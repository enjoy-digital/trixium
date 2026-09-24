# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiNormScratchpad (unit 12) against the firmware's definitions, compiled."""
import numpy as np
import pytest

from trixium.gateware.norm_sp import NORM_COMMAND, OP_L2, OP_RMS, BonsaiNormScratchpad
from rtl_client import build_client, run_client

f32 = np.float32
EPS = f32(1e-6)
SIZE = 256*1024


def canon(bits):
    bits = np.asarray(bits, dtype=np.uint32).copy()
    x = bits.view(f32)
    bits[np.isnan(x)] = 0x7FC00000
    return bits


def sumsq64(x):
    sq = (x*x).astype(f32)
    P = np.zeros(64, dtype=f32)
    full = len(sq)//64*64
    for row in sq[:full].reshape(-1, 64):
        P = (P + row).astype(f32)
    tail = len(sq) - full
    if tail:
        P[:tail] = (P[:tail] + sq[full:]).astype(f32)
    w = 32
    while w:
        P[:w] = (P[:w] + P[w:2*w]).astype(f32)
        w //= 2
    return P[0]


def reference(mem, cmd):
    """Apply one command to a uint32 word image (hybrid.c rmsnorm_fv / l2norm_fv)."""
    n, nseg = cmd["n"], cmd["nseg"]
    ss_ = (cmd.get("src_stride") or n//8)*8           # floats between segment starts
    ds_ = (cmd.get("dst_stride") or n//8)*8
    eps = np.uint32(cmd["eps"]).view(f32)
    w = mem[cmd["weight"]//4:cmd["weight"]//4 + n].view(f32) if cmd["op"] == OP_RMS else np.ones(n, f32)
    out = []
    with np.errstate(all="ignore"):
        for s in range(nseg):
            x = mem[cmd["src"]//4 + s*ss_:cmd["src"]//4 + s*ss_ + n].view(f32).copy()
            ss = sumsq64(x)
            if cmd["op"] == OP_RMS:
                r = f32(f32(1)/f32(np.sqrt(f32(f32(ss/f32(n)) + eps))))
            else:
                nrm = f32(np.sqrt(ss))
                if nrm < eps:
                    nrm = eps
                r = f32(f32(1)/nrm)
            out.append(canon((((x*r).astype(f32))*w).astype(f32).view(np.uint32)))
    for s, y in enumerate(out):
        mem[cmd["dst"]//4 + s*ds_:cmd["dst"]//4 + s*ds_ + n] = y


def command(op, src, dst, n, nseg, weight=0, **extra):
    c = dict(op=op, src=src, dst=dst, weight=weight, n=n, nseg=nseg,
             eps=int(EPS.view(np.uint32)), src_stride=0, dst_stride=0, protect_start=0, protect_end=0)
    c.update(extra)
    return c


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_client(BonsaiNormScratchpad(SIZE), NORM_COMMAND, tmp_path_factory.mktemp("norm"), "norm_sp")


def activations(rng, n, kind):
    if kind == "normal": return rng.normal(0, 2, n).astype(f32)
    if kind == "zero":   return np.zeros(n, f32)
    if kind == "tiny":   return (rng.normal(0, 1, n)*1e-22).astype(f32)      # squares underflow
    if kind == "huge":   return (rng.normal(0, 1, n)*3e19).astype(f32)       # ss overflows
    if kind == "nan":
        x = rng.normal(0, 1, n).astype(f32); x[n//3] = np.nan; return x
    raise ValueError(kind)


def test_compiled_norms_equal_the_firmware_definition(tmp_path, binary):
    rng = np.random.default_rng(12)
    mem = rng.integers(0, 1 << 32, SIZE//4, dtype=np.uint64).astype(np.uint32)
    def put(addr, values):
        mem[addr//4:addr//4 + len(values)] = np.asarray(values, f32).view(np.uint32)
    cmds = []
    # RMS over the model width (attention/FFN norms): n = 5120, one segment.
    put(0x00000, activations(rng, 5120, "normal"))
    put(0x05000, rng.normal(1, 0.2, 5120))
    cmds.append((command(OP_RMS, 0x00000, 0x0A000, 5120, 1, weight=0x05000), 0))
    # Per-head RMS (attention q heads): 24 x 256, special segments mixed in.
    heads = np.concatenate([activations(rng, 256, k) for k in
                            ["normal"]*18 + ["zero", "tiny", "huge", "nan", "normal", "normal"]])
    put(0x10000, heads)
    put(0x16000, rng.normal(1, 0.3, 256))
    cmds.append((command(OP_RMS, 0x10000, 0x18000, 256, 24, weight=0x16000), 0))
    # DeltaNet gated norm shape, in place: 48 x 128.
    put(0x20000, np.concatenate([activations(rng, 128, "normal") for _ in range(48)]))
    put(0x26000, rng.normal(1, 0.3, 128))
    cmds.append((command(OP_RMS, 0x20000, 0x20000, 128, 48, weight=0x26000), 0))
    # DeltaNet q/k L2: 32 x 128 (one zero segment: sqrt(0) < eps).
    put(0x28000, np.concatenate([activations(rng, 128, "zero" if i == 5 else "normal") for i in range(32)]))
    cmds.append((command(OP_L2, 0x28000, 0x2C000, 128, 32), 0))
    # The attention q heads in place inside [q | gate] (24 x 256, stride 2*256 floats = 64
    # beats); the gates between them stay untouched.
    put(0x30000, rng.normal(0, 1, 2*24*256 - 256))
    put(0x2F000, rng.normal(1, 0.3, 256))
    cmds.append((command(OP_RMS, 0x30000, 0x30000, 256, 24, weight=0x2F000, src_stride=64, dst_stride=64), 0))
    # A stride shorter than a segment is a geometry error.
    cmds.append((command(OP_RMS, 0x30000, 0x30000, 256, 24, weight=0x2F000, src_stride=16, dst_stride=16), 1))
    # Rejects: misaligned, n % 8, bounds, overlap, protected output.
    cmds.append((command(OP_RMS, 0x20004, 0x30000, 128, 1, weight=0x26000), 1))
    cmds.append((command(OP_RMS, 0x20000, 0x30000, 12, 1, weight=0x26000), 1))
    cmds.append((command(OP_L2, 0x3F000, 0x30000, 1024, 8), 2))
    cmds.append((command(OP_RMS, 0x20000, 0x20020, 128, 2, weight=0x26000), 4))
    cmds.append((command(OP_L2, 0x28000, 0x30000, 128, 1, protect_start=0x30010, protect_end=0x30020), 64))
    want = mem.copy()
    for c, err in cmds:
        if not err:
            reference(want, c)
    got, lines = run_client(binary, NORM_COMMAND, mem, cmds, tmp_path)
    print("\n".join(lines))
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:12]]
    # Throughput: the 5120-wide norm is two passes of 640 beats plus the scalar work.
    cycles = int(lines[0].split(" cycles ")[1].split()[0])
    assert cycles < 3*640 + 1000, lines[0]

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""NormScratchpad (unit 12) against the firmware's definitions, compiled."""

import numpy as np
import pytest

from gateware.units.norm import NORM_COMMAND, OP_L2, OP_RMS, NormScratchpad

from common.rtl_client import build_client, run_client

f32 = np.float32
EPS = f32(1e-6)
SIZE = 256*1024


# Helpers ------------------------------------------------------------------------------------------

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
            y = (((x*r).astype(f32))*w).astype(f32)
            hb = cmd.get("rope_hb", 0)
            if hb:      # hybrid.c rope_apply_partial: neox pairs over the first 16*hb values
                half = 8*hb
                tab = mem[cmd["rope_tab"]*8:cmd["rope_tab"]*8 + 2*half].view(f32)
                c, sn = tab[:half], tab[half:]
                y0, y1 = y[:half].copy(), y[half:2*half].copy()
                y[:half] = ((y0*c).astype(f32) - (y1*sn).astype(f32)).astype(f32)
                y[half:2*half] = ((y0*sn).astype(f32) + (y1*c).astype(f32)).astype(f32)
            out.append(canon(y.view(np.uint32)))
    for s, y in enumerate(out):
        mem[cmd["dst"]//4 + s*ds_:cmd["dst"]//4 + s*ds_ + n] = y


def command(op, src, dst, n, nseg, weight=0, **extra):
    c = dict(op=op, src=src, dst=dst, weight=weight, n=n, nseg=nseg,
             eps=int(EPS.view(np.uint32)), src_stride=0, dst_stride=0, protect_start=0, protect_end=0,
             rope_hb=0, rope_tab=0)
    c.update(extra)
    return c


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_client(NormScratchpad(SIZE), NORM_COMMAND, tmp_path_factory.mktemp("norm"), "norm_sp")


def activations(rng, n, kind):
    if kind == "normal": return rng.normal(0, 2, n).astype(f32)
    if kind == "zero":   return np.zeros(n, f32)
    if kind == "tiny":   return (rng.normal(0, 1, n)*1e-22).astype(f32)      # squares underflow
    if kind == "huge":   return (rng.normal(0, 1, n)*3e19).astype(f32)       # ss overflows
    if kind == "nan":
        x = rng.normal(0, 1, n).astype(f32); x[n//3] = np.nan; return x
    raise ValueError(kind)


# Tests --------------------------------------------------------------------------------------------

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


def test_rope_after_the_norm_equals_the_firmware(tmp_path, binary):
    """The attention's q/k heads: RMS per head, then RoPE over the first 2*8*hb values
    (neox pairs), cos then sin beats at rope_tab; strided q in place, k out of place, a
    full-width rotation, and the rejects."""
    rng = np.random.default_rng(31)
    mem = rng.integers(0, 1 << 32, SIZE//4, dtype=np.uint64).astype(np.uint32)
    def put(addr, values):
        mem[addr//4:addr//4 + len(values)] = np.asarray(values, f32).view(np.uint32)
    th = rng.uniform(-40, 40, 128).astype(f32)
    table = np.concatenate([np.cos(th[:32]), np.sin(th[:32])]).astype(f32)   # hb = 4
    table[5] = -0.0; table[40] = 0.0
    put(0x3E000, table)
    full = np.concatenate([np.cos(th), np.sin(th)]).astype(f32)               # hb = 16
    put(0x3C000, full)
    cmds = []
    # q heads in place inside [q | gate]: 24 x 256, stride 64 beats, rope_n 64.
    heads = np.concatenate([activations(rng, 256, k) for k in ["normal"]*20 + ["zero", "tiny", "huge", "nan"]])
    q = rng.normal(0, 1, 2*24*256).astype(f32)
    for h in range(24):
        q[2*256*h:2*256*h + 256] = heads[256*h:256*(h + 1)]
    put(0x00000, q)
    put(0x18000, rng.normal(1, 0.3, 256))
    cmds.append((command(OP_RMS, 0x00000, 0x00000, 256, 24, weight=0x18000, src_stride=64, dst_stride=64,
                         rope_hb=4, rope_tab=0x3E000//32), 0))
    # k heads out of place, 4 x 256, the whole head rotated (hb = 16).
    put(0x20000, rng.normal(0, 2, 4*256))
    put(0x19000, rng.normal(1, 0.3, 256))
    cmds.append((command(OP_RMS, 0x20000, 0x22000, 256, 4, weight=0x19000, rope_hb=16, rope_tab=0x3C000//32), 0))
    # L2 with a rotation (the unit allows it).
    put(0x24000, rng.normal(0, 1, 8*128))
    cmds.append((command(OP_L2, 0x24000, 0x26000, 128, 8, rope_hb=4, rope_tab=0x3E000//32), 0))
    # Rejects: hb past the segment, hb > 16, the table out of bounds.
    cmds.append((command(OP_RMS, 0x20000, 0x22000, 128, 1, weight=0x19000, rope_hb=9, rope_tab=0x3C000//32), 1))
    cmds.append((command(OP_RMS, 0x20000, 0x22000, 1024, 1, weight=0x19000, rope_hb=17, rope_tab=0x3C000//32), 1))
    cmds.append((command(OP_RMS, 0x20000, 0x22000, 256, 1, weight=0x19000, rope_hb=4, rope_tab=SIZE//32 - 4), 2))
    want = mem.copy()
    for c, err in cmds:
        if not err:
            reference(want, c)
    got, lines = run_client(binary, NORM_COMMAND, mem, cmds, tmp_path)
    print("\n".join(lines))
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:12]]

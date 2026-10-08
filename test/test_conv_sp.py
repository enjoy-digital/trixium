# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiConvScratchpad (unit 13) against hybrid.c's 4-tap causal conv, compiled."""
import numpy as np
import pytest

from trixium.units.conv import CONV_COMMAND, BonsaiConvScratchpad
from rtl_client import build_client, run_client

f32 = np.float32
SIZE = 256*1024


def canon(bits):
    bits = np.asarray(bits, dtype=np.uint32).copy()
    bits[np.isnan(bits.view(f32))] = 0x7FC00000
    return bits


def reference(mem, cmd):
    n = cmd["channels"]
    if cmd.get("op"):
        a = mem[cmd["x0"]//4:cmd["x0"]//4 + n].view(f32)
        b = mem[cmd["x3"]//4:cmd["x3"]//4 + n].view(f32)
        with np.errstate(all="ignore"):
            mem[cmd["dst"]//4:cmd["dst"]//4 + n] = canon((a + b).astype(f32).view(np.uint32))
        return
    x = [mem[cmd[k]//4:cmd[k]//4 + n].view(f32) for k in ("x0", "x1", "x2", "x3")]
    w = mem[cmd["w"]//4:cmd["w"]//4 + 4*n].view(f32).reshape(n, 4)
    with np.errstate(all="ignore"):
        a = (f32(0) + (w[:, 0]*x[0]).astype(f32)).astype(f32)
        for t in (1, 2, 3):
            a = (a + (w[:, t]*x[t]).astype(f32)).astype(f32)
    mem[cmd["dst"]//4:cmd["dst"]//4 + n] = canon(a.view(np.uint32))


def command(x0, x1, x2, x3, w, dst, channels, **extra):
    c = dict(op=0, x0=x0, x1=x1, x2=x2, x3=x3, w=w, dst=dst, channels=channels, protect_start=0, protect_end=0)
    c.update(extra)
    return c


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_client(BonsaiConvScratchpad(SIZE), CONV_COMMAND, tmp_path_factory.mktemp("conv"), "conv_sp")


def test_compiled_conv_equals_the_firmware(tmp_path, binary):
    rng = np.random.default_rng(5)
    mem = rng.integers(0, 1 << 32, SIZE//4, dtype=np.uint64).astype(np.uint32)
    def put(addr, v):
        mem[addr//4:addr//4 + len(v)] = np.asarray(v, f32).view(np.uint32)
    n = 2048                                     # one 27B chunk
    for k, a in enumerate((0x00000, 0x02000, 0x04000, 0x06000)):
        v = rng.normal(0, 1.5, n).astype(f32)
        if k == 3:
            v[:8] = [0, -0.0, 1e-42, -1e-40, np.inf, -np.inf, np.nan, 3e38]
        if k == 0:
            v[:4] = [-0.0, -0.0, 0.0, 5.0]
        put(a, v)
    taps = rng.normal(0, 0.5, 4*n).astype(f32)
    taps[:4] = [0, -0.0, 2.0, -1.0]
    put(0x08000, taps)
    cmds = [(command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x10000, n), 0),
            # In place over x3, a short tail-free run.
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x06000, 64), 0),
            # The residual add (x1, x2, w ignored), in place over x0, with signed zeros.
            (command(0x00000, 0x3FFE0, 0x3FFE0, 0x06000, 0x3FFE0, 0x00000, 2048, op=1), 0),
            # Rejects.
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x10000, 12), 1),
            (command(0x00004, 0x02000, 0x04000, 0x06000, 0x08000, 0x10000, 8), 1),
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x3FF00, 0x10000, 64), 2),
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x02000, 64), 4),
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x06020, 64), 4),
            (command(0x00000, 0x02000, 0x04000, 0x06000, 0x08000, 0x10000, 64,
                     protect_start=0x10000, protect_end=0x10004), 64)]
    want = mem.copy()
    for c, err in cmds:
        if not err:
            reference(want, c)
    got, lines = run_client(binary, CONV_COMMAND, mem, cmds, tmp_path)
    print("\n".join(lines))
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:12]]
    cycles = int(lines[0].split(" cycles ")[1].split()[0])
    assert cycles < 1.3*n + 200, lines[0]          # ~one channel per clock

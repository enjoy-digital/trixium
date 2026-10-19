#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""SoftmaxScratchpad (unit 14) against the firmware's softmax loop, compiled."""

import numpy as np
import pytest

from gateware.units.softmax import SOFTMAX_COMMAND, SoftmaxScratchpad
from model.rv32_math import softmax_exp_rv32

from common.rtl_client import build_client, run_client

f32 = np.float32
SIZE = 256*1024


# Reference ----------------------------------------------------------------------------------------

def canon(bits):
    bits = np.asarray(bits, dtype=np.uint32).copy()
    x = bits.view(f32)
    bits[np.isnan(x)] = 0x7FC00000
    return bits


def softmax_row(s):
    """hybrid.c softmax_rows, one row: max (a NaN never wins), canonical exp, sequential sum,
    one reciprocal, multiplies."""
    mx = f32(-1e30)
    for v in s:
        if v > mx:
            mx = v
    e = softmax_exp_rv32((s - mx).astype(f32))
    total = f32(0)
    for v in e:
        total = f32(total + v)
    with np.errstate(all="ignore"):
        inv = f32(f32(1)/total)
        return (e*inv).astype(f32)


def reference(mem, cmd):
    n, rows = cmd["n"], cmd["rows"]
    stride = (cmd["stride"] or (n + 7)//8)*8
    base = cmd["src"]//4
    with np.errstate(all="ignore"):
        for r in range(rows):
            s = mem[base + r*stride:base + r*stride + n].view(f32).copy()
            mem[base + r*stride:base + r*stride + n] = canon(softmax_row(s).view(np.uint32))


def command(src, n, rows, stride=0, **extra):
    c = dict(src=src, n=n, rows=rows, stride=stride, protect_start=0, protect_end=0)
    c.update(extra)
    return c


@pytest.fixture(scope="module")
def binary():
    return build_client(lambda: SoftmaxScratchpad(SIZE), SOFTMAX_COMMAND, "softmax_sp",
                        key=f"SoftmaxScratchpad({SIZE})")


def scores(rng, n, kind):
    if kind == "normal": return rng.normal(0, 3, n).astype(f32)
    if kind == "flat":   return np.full(n, 2.5, f32)
    if kind == "wide":   return rng.normal(0, 60, n).astype(f32)         # many exps below -80
    if kind == "zeros":
        x = rng.normal(0, 3, n).astype(f32); x[::3] = 0.0; x[1::3] = -0.0; return x
    if kind == "nan":
        x = rng.normal(0, 3, n).astype(f32); x[n//2] = np.nan; return x
    if kind == "allnan": return np.full(n, np.nan, f32)
    if kind == "huge":
        x = rng.normal(0, 3, n).astype(f32); x[0] = 3e38; x[1] = -3e38; return x
    if kind == "tiny":   return (rng.normal(0, 1, n)*1e-30).astype(f32)
    raise ValueError(kind)


# Tests --------------------------------------------------------------------------------------------

def test_compiled_softmax_equals_the_firmware_loop(tmp_path, binary):
    rng = np.random.default_rng(14)
    mem = rng.integers(0, 1 << 32, SIZE//4, dtype=np.uint64).astype(np.uint32)
    def put(addr, rows, stride_floats):
        for r, x in enumerate(rows):
            mem[addr//4 + r*stride_floats:addr//4 + r*stride_floats + len(x)] = np.asarray(x, f32).view(np.uint32)
    cmds = []
    # The 2B's attention at a short context: 16 heads, T = 37 (a partial last beat), Tp = 40.
    kinds = ["normal"]*10 + ["flat", "wide", "zeros", "nan", "huge", "tiny"]
    put(0x00000, [scores(rng, 37, k) for k in kinds], 40)
    cmds.append((command(0x00000, 37, 16), 0))
    # A long context: 16 heads, T = 256.
    put(0x04000, [scores(rng, 256, "normal" if r % 5 else "wide") for r in range(16)], 256)
    cmds.append((command(0x04000, 256, 16), 0))
    # Fewer rows than a group (3), and two groups with a short second one (9), T = 100.
    put(0x10000, [scores(rng, 100, k) for k in ("normal", "allnan", "normal")], 104)
    cmds.append((command(0x10000, 100, 3), 0))
    put(0x12000, [scores(rng, 100, "normal") for _ in range(9)], 104)
    cmds.append((command(0x12000, 100, 9), 0))
    # Strided rows (the gates between heads untouched), T = 8 (one full beat), 4 rows.
    put(0x18000, [scores(rng, 8, "normal") for _ in range(4)], 32)
    cmds.append((command(0x18000, 8, 4, stride=4), 0))
    # One element per row.
    put(0x1A000, [scores(rng, 1, "normal") for _ in range(5)], 8)
    cmds.append((command(0x1A000, 1, 5), 0))
    # Rejects: misaligned, n = 0, a stride shorter than the row, bounds, protected output.
    cmds.append((command(0x1A004, 8, 1), 1))
    cmds.append((command(0x1A000, 0, 1), 1))
    cmds.append((command(0x18000, 16, 4, stride=1), 1))
    cmds.append((command(0x3FF00, 256, 2), 2))
    cmds.append((command(0x18000, 8, 4, stride=4, protect_start=0x18040, protect_end=0x18060), 64))
    want = mem.copy()
    for c, err in cmds:
        if not err:
            reference(want, c)
    got, lines = run_client(binary, SOFTMAX_COMMAND, mem, cmds, tmp_path)
    print("\n".join(lines))
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:12]]
    # Throughput of the T = 256 x 16 command: pass 2 at one element per clock dominates
    # (4096), the max and scale passes a beat per clock (512 each), 16 reciprocals.
    cycles = int(lines[1].split(" cycles ")[1].split()[0])
    assert cycles < 4096 + 2*512 + 16*40 + 1500, lines[1]

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/fp32.py: the exact FP32 operators and FP32ArgmaxTracker."""

import subprocess

import numpy as np
import pytest

from migen import Module, Signal, run_simulation

from gateware.units.fp32 import CANONICAL_NAN, FP32Add, FP32DivSqrt, FP32Mul, FP32ArgmaxTracker

from common import RTL
from common.verilator import Simulator

# FP32Add / FP32Mul / FP32DivSqrt ------------------------------------------------------------------
# gateware/units/fp32.py against IEEE float32 (numpy), NaNs canonical as on RV32F.

f32 = np.float32


def canon(x):
    x = np.asarray(x, dtype=f32)
    bits = x.view(np.uint32).copy()
    bits[np.isnan(x)] = CANONICAL_NAN
    return bits


def operands(seed, n):
    """Random bit patterns across every class, plus hand-picked edges and near-ties."""
    rng = np.random.default_rng(seed)
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x00800001,
                        0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FA00000,
                        0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF, 0x34000000, 0x33800000,
                        0x40000000, 0x3FFFFFFF, 0x0DA24260, 0x7E967699], dtype=np.uint32)
    a = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    # Close exponents (cancellation, ties) and small exponents (subnormal results).
    k = n//4
    b[:k] = (a[:k] & 0xFF800000) | rng.integers(0, 1 << 23, k, dtype=np.uint32)
    b[k:2*k] ^= rng.integers(0, 8, k, dtype=np.uint32) << 23 & 0x7F800000 ^ (a[k:2*k] & 0x7F800000)
    small = rng.integers(0, 0x0B000000, k, dtype=np.uint32) | (rng.integers(0, 2, k, dtype=np.uint32) << 31)
    a[2*k:3*k] = small
    ea, eb = np.meshgrid(special, special)
    return (np.concatenate([a, ea.ravel()]).astype(np.uint32),
            np.concatenate([b, eb.ravel()]).astype(np.uint32))


def expected(op, a, b):
    fa, fb = a.view(f32), b.view(f32)
    with np.errstate(all="ignore"):
        if op == "mul": r = fa*fb
        elif op == "add": r = fa + fb
        elif op == "div": r = fa/fb
        else: r = np.sqrt(fa)
    return canon(r)


@pytest.fixture(scope="module")
def compiled():
    top = Module()
    top.submodules.mul = mul = FP32Mul()
    top.submodules.add = add = FP32Add()
    top.submodules.ds = ds = FP32DivSqrt()
    ports = {"mul_a": mul.a, "mul_b": mul.b, "mul_r": mul.r, "mul_ce": mul.ce,
             "add_a": add.a, "add_b": add.b, "add_r": add.r, "add_ce": add.ce,
             "ds_a": ds.a, "ds_b": ds.b, "ds_op": ds.op, "ds_start": ds.start, "ds_done": ds.done,
             "ds_r": ds.r}
    lat = {"mul_latency": FP32Mul.LATENCY, "add_latency": FP32Add.LATENCY}
    for name, sig in ports.items():
        sig.name_override = name
    consts = {}
    for name, value in lat.items():
        s = Signal(8, name_override=name)
        top.comb += s.eq(value)
        consts[name] = s
    return Simulator(top, ios=dict(ports, **consts), name="fp32x",
                     driver=(RTL / "fp32_exact.cpp").read_text()).binary


@pytest.mark.parametrize("op,n", [("mul", 400000), ("add", 400000), ("div", 60000), ("sqrt", 60000)])
def test_compiled_operator_equals_ieee(tmp_path, compiled, op, n):
    a, b = operands({"mul": 1, "add": 2, "div": 3, "sqrt": 4}[op], n)
    if op == "sqrt":
        a = np.concatenate([a, np.abs(a.view(f32)).view(np.uint32)])     # plenty of positives
        b = np.concatenate([b, b])
    want = expected(op, a, b)
    vec = tmp_path / f"{op}.bin"
    np.stack([a, b, want], axis=1).astype("<u4").tofile(vec)
    run = subprocess.run([str(compiled), op, str(vec)], capture_output=True, text=True, timeout=900)
    print(run.stdout, run.stderr[-3000:])
    assert run.returncode == 0, run.stdout + run.stderr[-3000:]


def test_migen_simulation_smoke():
    """A few values through the Migen simulator (the compiled test is the real check)."""
    dut = Module()
    dut.submodules.m = m = FP32Mul()
    dut.submodules.a = a = FP32Add()
    xs = [(1.5, 2.25), (-3.0, 0.125), (1e-40, 3.0), (3e38, 10.0)]
    got = []
    def run():
        for x, y in xs + [(0, 0)]*8:
            yield m.a.eq(int(f32(x).view(np.uint32))); yield m.b.eq(int(f32(y).view(np.uint32)))
            yield a.a.eq(int(f32(x).view(np.uint32))); yield a.b.eq(int(f32(y).view(np.uint32)))
            yield
            got.append(((yield m.r), (yield a.r)))
    run_simulation(dut, run())
    for i, (x, y) in enumerate(xs):
        with np.errstate(all="ignore"):
            wm, wa = canon(f32(x)*f32(y)), canon(f32(x) + f32(y))
        assert got[i + FP32Mul.LATENCY][0] == int(wm), (x, y, hex(got[i + FP32Mul.LATENCY][0]))
        assert got[i + FP32Add.LATENCY][1] == int(wa), (x, y, hex(got[i + FP32Add.LATENCY][1]))


# FP32ArgmaxTracker --------------------------------------------------------------------------------
# FP32ArgmaxTracker against the firmware's greedy argmax loop.

def cpu_argmax(x):
    best, idx = x[0], 0
    for i in range(1, len(x)):
        if x[i] > best:
            best, idx = x[i], i
    return idx


def run_tracker(values, chunks):
    """Feed `values` as consecutive commands of the given element counts (partial last beats)."""
    dut = FP32ArgmaxTracker()
    bits = np.asarray(values, f32).view(np.uint32)
    out = {}
    def gen():
        yield dut.clear.eq(1); yield
        yield dut.clear.eq(0)
        pos = 0
        for n in chunks:
            for b in range(0, n, 8):
                lanes = min(8, n - b)
                word = sum(int(bits[pos + b + j]) << (32*j) for j in range(lanes))
                yield dut.valid.eq(1); yield dut.data.eq(word); yield dut.mask.eq((1 << lanes) - 1)
                yield
                yield dut.valid.eq(0)
                if b % 3 == 0:
                    yield                               # gaps between beats
            pos += n
            # a whole chunk ends on a beat boundary in the unit (one command per chunk)
        for _ in range(8):
            yield
        out["idx"] = (yield dut.best_idx)
        out["valid"] = (yield dut.best_valid)
    run_simulation(dut, gen())
    return out


@pytest.mark.parametrize("case", ["random", "ties", "zeros", "nan_first", "nan_inside", "neg"])
def test_tracker_equals_the_cpu_loop(case):
    rng = np.random.default_rng(hash(case) % 1000)
    x = rng.normal(0, 3, 200).astype(f32)
    if case == "ties":
        x[:] = f32(1.5); x[37] = f32(2.0); x[150] = f32(2.0)
    elif case == "zeros":
        x[:] = f32(-1); x[5] = f32(-0.0); x[9] = f32(0.0); x[100] = f32(-0.0)
    elif case == "nan_first":
        x[0] = np.nan
    elif case == "nan_inside":
        x[17] = np.nan; x[18] = f32(1e30); x[64] = np.nan
    elif case == "neg":
        x = -np.abs(x) - 1
    chunks = [48, 64, 88]                      # element counts of consecutive commands (8-aligned)
    got = run_tracker(x, chunks)
    assert got["valid"] == 1
    assert got["idx"] == cpu_argmax(x), (case, got["idx"], cpu_argmax(x))


def test_tracker_partial_last_beat():
    x = np.full(19, f32(-5)); x[18] = f32(7)
    assert run_tracker(x, [19])["idx"] == 18


def test_tracker_partial_beats_between_commands():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 60).astype(f32); x[45] = f32(9)
    assert run_tracker(x, [13, 21, 26])["idx"] == 45

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/fp32_exact.py against IEEE float32 (numpy), NaNs canonical as on RV32F."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from migen import Module, Signal, run_simulation
from migen.fhdl import verilog

from trixium.units.fp32 import CANONICAL_NAN, FP32Add, FP32DivSqrt, FP32Mul

f32 = np.float32


# Helpers ------------------------------------------------------------------------------------------

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
def compiled(tmp_path_factory):
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.skip("Verilator required")
    tmp = tmp_path_factory.mktemp("fp32x")
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
    rtl = tmp / "fp32x.v"
    verilog.convert(top, ios=set(ports.values()) | set(consts.values()), name="fp32x").write(str(rtl))
    harness = Path(__file__).parent / "rtl/fp32_exact.cpp"
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "4", "-Wno-fatal", "-O3",
                            "--Mdir", str(tmp / "obj"), "--top-module", "fp32x", str(rtl),
                            str(harness.resolve())], capture_output=True, text=True, timeout=900)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
    return tmp / "obj/Vfp32x"


# Tests --------------------------------------------------------------------------------------------

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

# SPDX-License-Identifier: BSD-2-Clause
"""SiluMulKernel (portable RTL) against the bit-exact model, compiled with Verilator.

The chain of evidence: hybrid.c == silu_mul_fast == silu_mul_fixed (test_silu_hw.py),
and here RTL == silu_mul_fixed on every element, with random input gaps and output
stalls. A smaller Migen-simulated run checks the same kernel without Verilator.
"""
import random
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from migen import run_simulation
from migen.fhdl import verilog

from trixium.gateware.silu_mul import SiluMulKernel
from trixium.model.silu_hw import LUT_N, f32_bits, silu_mul_firmware, silu_mul_fixed, tables


def stress_vectors(seed=3, n_random=120000):
    rng = np.random.default_rng(seed)
    gates, mults = [], []
    # Every table node +-2 ulp, each with a few multipliers.
    ms = [f32_bits(np.float32(v)) for v in (1.0, -1.0, 0.7, -3.25, 1e-3, 1234.5, 0.0, -0.0)]
    for i in range(LUT_N + 1):
        b = f32_bits(np.float32(-16 + i/128))
        for dlt in (-2, -1, 0, 1, 2):
            for m in ms[:3] if dlt else ms:
                gates.append((b + dlt) & 0xFFFFFFFF); mults.append(m)
    # Random gates across the table range and across magnitudes down to subnormals.
    gates += [f32_bits(x) for x in rng.uniform(-16, 16, n_random).astype(np.float32)]
    tiny = rng.normal(0, 1, n_random//2) * np.exp2(rng.integers(-150, 4, n_random//2).astype(float))
    gates += [f32_bits(x) for x in tiny.astype(np.float32)]
    # Tails, specials, boundaries.
    for x in (16.0, -16.0, 80.0, -80.0, 16.000002, -16.000002, -80.00001, -79.99999, 17.0, -100.0,
              1e30, -1e30, 3.4e38, -3.4e38, 0.0, -0.0):
        gates.append(f32_bits(np.float32(x)))
    gates += [0x417FFFFF, 0xC17FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80000001]
    while len(mults) < len(gates):
        mults.append(f32_bits(np.float32(rng.normal(0, 3))))
    # Multipliers that push the product out of the normal range, and special multipliers.
    for g in (f32_bits(np.float32(v)) for v in (-10.0, -3.0, 0.5, 5.0, 20.0)):
        for m in (f32_bits(np.float32(v)) for v in (1e-35, -1e-37, 1e36, -3e38)) :
            gates.append(g); mults.append(m)
        for m in (0x7F800000, 0x7FC00000, 0x00000003, 0x80400000, 0x00800000):
            gates.append(g); mults.append(m)
    order = list(range(len(gates)))
    random.Random(seed).shuffle(order)
    return [gates[i] for i in order], [mults[i] for i in order]


def expected(gates, mults):
    lut, dlut = tables()
    out = []
    for g, m in zip(gates, mults):
        r, fix = silu_mul_fixed(g, m, lut, dlut)
        out.append((r, int(fix)))
    return out


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.skip("Verilator required")
    tmp = tmp_path_factory.mktemp("silu-mul")
    k = SiluMulKernel()
    ports = {"sink_valid": k.sink.valid, "sink_ready": k.sink.ready, "sink_g": k.sink.g,
             "sink_m": k.sink.m, "source_valid": k.source.valid, "source_ready": k.source.ready,
             "source_r": k.source.r, "source_fixup": k.source.fixup,
             "lut_we": k.lut_we, "lut_adr": k.lut_adr, "lut_dat": k.lut_dat,
             "dlut_we": k.dlut_we, "dlut_adr": k.dlut_adr, "dlut_dat": k.dlut_dat}
    for name, sig in ports.items():
        sig.name_override = name
    rtl = tmp / "silu_mul.v"
    verilog.convert(k, ios=set(ports.values()), name="silu_mul").write(str(rtl))
    harness = Path(__file__).parent / "rtl/silu_mul.cpp"
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "4", "-Wno-fatal", "-O3",
                            "--Mdir", str(tmp / "obj"), "--top-module", "silu_mul", str(rtl),
                            str(harness.resolve())], capture_output=True, text=True, timeout=600)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
    return tmp / "obj/Vsilu_mul"


def write_vectors(path, gates, mults, want):
    lut, dlut = tables()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(gates)))
        f.write(struct.pack(f"<{len(lut)}I", *lut))
        f.write(struct.pack(f"<{len(dlut)}I", *dlut))
        for g, m, (r, fix) in zip(gates, mults, want):
            f.write(struct.pack("<4I", g, m, r, fix))


def test_compiled_kernel_equals_the_model(tmp_path, compiled):
    gates, mults = stress_vectors()
    want = expected(gates, mults)
    vec = tmp_path / "vectors.bin"
    write_vectors(vec, gates, mults, want)
    run = subprocess.run([str(compiled), str(vec)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)
    fix = sum(f for _, f in want)
    assert f"{len(gates)} elements" in run.stdout and f"{fix} fixups" in run.stdout


def test_model_reference_spot_checked_against_firmware():
    # The bulk reference above is silu_mul_fixed; re-check a sample against the firmware
    # semantics directly so this file stands on its own.
    gates, mults = stress_vectors(seed=9, n_random=2000)
    lut, dlut = tables()
    for g, m in list(zip(gates, mults))[:6000]:
        r, fix = silu_mul_fixed(g, m, lut, dlut)
        if not fix:
            assert r == silu_mul_firmware(g, m), (hex(g), hex(m))


def test_migen_simulation_small():
    """The same kernel in Migen's simulator: a few hundred elements with stalls."""
    k = SiluMulKernel()
    lut, dlut = tables()
    gates, mults = stress_vectors(seed=21, n_random=200)
    gates, mults = gates[:400], mults[:400]
    want = expected(gates, mults)
    rng = random.Random(4)
    got = []

    def drive():
        for i in range(LUT_N + 1):
            yield k.lut_we.eq(1); yield k.lut_adr.eq(i); yield k.lut_dat.eq(lut[i])
            yield k.dlut_we.eq(i < LUT_N); yield k.dlut_adr.eq(i & 0xFFF)
            yield k.dlut_dat.eq(dlut[i] if i < LUT_N else 0)
            yield
        yield k.lut_we.eq(0); yield k.dlut_we.eq(0)
        sent = 0
        for _ in range(8000):
            offer = sent < len(gates) and rng.randrange(6) != 0
            yield k.sink.valid.eq(offer)
            if offer:
                yield k.sink.g.eq(gates[sent]); yield k.sink.m.eq(mults[sent])
            yield k.source.ready.eq(rng.randrange(4) != 0)
            yield
            if (yield k.sink.valid) and (yield k.sink.ready):
                sent += 1
            if (yield k.source.valid) and (yield k.source.ready):
                got.append(((yield k.source.r), (yield k.source.fixup)))
            if len(got) == len(gates):
                return

    run_simulation(k, drive())
    assert len(got) == len(want)
    for idx, ((r, f), (wr, wf)) in enumerate(zip(got, want)):
        assert f == wf, (idx, hex(gates[idx]), hex(mults[idx]))
        if not wf:
            assert r == wr, (idx, hex(gates[idx]), hex(mults[idx]), hex(r), hex(wr))

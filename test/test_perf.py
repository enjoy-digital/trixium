#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Rollover, atomicity and legacy compatibility for actual RTL/C profiling code."""

import shutil
import subprocess
from pathlib import Path

import pytest

from migen import Module, Constant, run_simulation

from litex.soc.interconnect.axi import AXIInterface

from gateware.core.csr import EngineCSR
from gateware.core.engine import MatVecEngine
from gateware.core.scratchpad import Scratchpad


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("total", "eng", "vec", "both")


# Helpers ------------------------------------------------------------------------------------------

def make_engine(sp):
    return MatVecEngine(sp, [AXIInterface(data_width=256, address_width=33,
                                              id_width=6)], xbuf_groups=8)


# Tests --------------------------------------------------------------------------------------------

def test_snapshot_is_atomic_and_stable():
    class DUT(Module):
        def __init__(self):
            # Only the CSR front end is simulated here; the generator drives the engine
            # signals directly to inject all four independent rollover cases.
            sp = Scratchpad(size=16*1024)
            engine = make_engine(sp)
            self.counter_inputs = [getattr(engine, "cyc_" + n) for n in NAMES]
            self.submodules.csr = EngineCSR(engine)
    dut = DUT()
    outputs = [getattr(dut.csr, "snap_" + n).status for n in NAMES]

    def gen():
        for sample in [(0xffffffff, 0xfffffffe, 12, 3),
                       (0x200000014, 0x100000002, 0x100000001, 0xfffffffc),
                       (0x1000000000014, 0x1000000000002, 100, 42)]:
            for signal, value in zip(dut.counter_inputs, sample):
                assert len(signal) == 64
                yield signal.eq(value)
            yield dut.csr.perf_snapshot.storage.eq(1)
            yield dut.csr.perf_snapshot.re.eq(1)
            yield
            yield dut.csr.perf_snapshot.re.eq(0)
            yield
            for signal, value in zip(outputs, sample):
                assert (yield signal) == value
            # Split RV32 reads must remain stable even if every live input changes.
            for signal, value in zip(dut.counter_inputs, sample):
                yield signal.eq(value + 1000)
            for _ in range(5):
                yield
            for signal, value in zip(outputs, sample):
                assert (yield signal) == value
        # A write of zero is not a snapshot.
        yield dut.csr.perf_snapshot.storage.eq(0)
        yield dut.csr.perf_snapshot.re.eq(1)
        yield
        yield
        for signal, value in zip(outputs, sample):
            assert (yield signal) == value
    run_simulation(dut, gen())

    # CSR bank ordering is append-only: every old software-visible offset is preserved.
    names = [c.name for c in dut.csr.get_csrs()]
    assert names[-9:] == ["cyc_total", "cyc_eng", "cyc_vec", "cyc_both",
                         "perf_snapshot", "snap_total", "snap_eng", "snap_vec", "snap_both"]


def test_live_total_counter_crosses_48_bits():
    class DUT(Module):
        def __init__(self):
            self.submodules.sp = Scratchpad(size=16*1024)
            self.submodules.engine = make_engine(self.sp)
            self.engine.cyc_total.reset = Constant((1 << 48) - 2, 64)
    dut = DUT()

    def gen():
        for _ in range(6):
            yield
        assert (yield dut.engine.cyc_total) >= (1 << 48)
    run_simulation(dut, gen())


@pytest.mark.parametrize("mode", ["wide", "legacy", "absent"])
def test_firmware_profiler(tmp_path, mode):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler unavailable")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "soc.h").write_text("")
    header = """
#include <stdint.h>
extern uint64_t test_live[4], test_snap[4];
static inline void timer0_en_write(uint32_t v) { (void)v; }
static inline void timer0_load_write(uint32_t v) { (void)v; }
static inline void timer0_reload_write(uint32_t v) { (void)v; }
"""
    flags = []
    if mode != "absent":
        header += "#define CSR_TRIXIUM_ENGINE_CSR_CYC_TOTAL_ADDR 1\n#define CSR_TRIXIUM_ENGINE_CSR_CYC_ENG_ADDR 1\n"
        for i, name in enumerate(NAMES):
            header += (f"static inline uint32_t trixium_engine_csr_cyc_{name}_read(void) "
                       f"{{ return (uint32_t)test_live[{i}]; }}\n")
    else:
        flags.append("-DTEST_NO_COUNTERS")
    if mode == "wide":
        flags.append("-DTEST_WIDE_COUNTERS")
        header += """
#define CSR_TRIXIUM_ENGINE_CSR_PERF_SNAPSHOT_ADDR 1
static inline void trixium_engine_csr_perf_snapshot_write(uint32_t v) {
    if (v) for (int i = 0; i < 4; i++) test_snap[i] = test_live[i];
}
"""
        for i, name in enumerate(NAMES):
            header += (f"static inline uint64_t trixium_engine_csr_snap_{name}_read(void) "
                       f"{{ return test_snap[{i}]; }}\n")
    (generated / "csr.h").write_text(header)
    exe = tmp_path / "perf_test"
    subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-DTRIXIUM_PERF_TEST",
                    *flags, "-I" + str(tmp_path), "-I" + str(ROOT / "firmware"),
                    str(ROOT / "firmware/perf.c"),
                    str(ROOT / "test/perf_host.c"),
                    "-o", str(exe)], check=True, capture_output=True, text=True)
    output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    assert "4294967344" in output
    assert "format: 0 18446744073709551615" in output
    if mode == "wide":
        assert "coherent64" in output and "unavailable" not in output
    else:
        assert "overlap: unavailable" in output
    if mode == "legacy":
        assert "8589934592 cycles; legacy32 approximate" in output

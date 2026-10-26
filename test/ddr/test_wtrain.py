#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""liblitedram's round-trip write training (sdram_write_training, the C port of
software/host/rdlevel.py's `train` and `check`) against rdlevel.py itself, on one eye model.

model.c is a DFI injector plus a US+ PHY's delay CSRs with a write eye and a read eye per
module. LiteX's sdram.c and accessors.c compile natively against the shim generated headers
(shim/) into a shared library that also exports the model's controls; rdlevel.py drives the
same model through a Bar0 whose reads and writes call it. The eyes cover the normal path, a
wrapped write window, the find_writes/find_reads fallbacks, the write-bitslip loop and a lane
that never passes. The C trainer must make rdlevel's choices exactly when its scans probe
every seed pair (SDRAM_WRITE_TRAINING_COARSE_SEEDS=3) and land in the same window with one (the
default); `check` must agree bit for bit; the CSR writes of a full run are the same.

Run: python3 -m pytest test/ddr -q (the never-passing lane through rdlevel.py is --runslow)."""

import io
import os
import contextlib
import ctypes as ct
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from software.host import rdlevel

HERE  = Path(__file__).resolve().parent
SHIM  = HERE / "shim"
CSR_H = SHIM / "generated/csr.h"

CSR_BASE = 0xf0000000
MODULES  = 8
STEP     = 4            # board.py trains the card with step 4.

# Eyes ---------------------------------------------------------------------------------------------

# Per module: the write eye (write bitslip, DQS range, DQ-DQS offset window centre/half-width,
# margin), the read eye (read bitslip, delay centre/half-width, margin) and the PHY state the
# training starts from (write bitslip, DQS, DQ, read bitslip, read delay): the DQ at the DQS as
# write leveling leaves them, the reads as the MPR read leveling set them. Positions are taps.
# On the US+ PHY the DQS reset pulse does nothing, so the write map (DQ reset to 0, the DQS
# where it is) runs at the offset -DQS: the eyes below are set around that; the read-side
# fallbacks keep the DQS where find_writes leaves it (+8), inside the range. `path` is the
# fallback message the lane must take (None: the map finds it).
EYES = {
    0: dict(what="wide, centred",            w=(2, 100, 400,   0, 120, 8), r=(3, 250, 100, 8), s=(0, 40, 40, 3, 250), path=None),
    1: dict(what="narrow (find_writes)",     w=(5, 200, 240,  20,  24, 4), r=(1, 300,  12, 2), s=(0, 10, 10, 1, 300), path="write side found"),
    2: dict(what="wrapped write window",     w=(2, 100, 400, 500,  40, 8), r=(3, 250, 100, 8), s=(0, 40, 40, 3, 250), path=None),
    3: dict(what="DQ off its eye",           w=(4, 100, 400, 200,  40, 8), r=(2, 200,  90, 8), s=(0, 40, 40, 2, 200), path="write side found"),
    4: dict(what="reads off (find_reads)",   w=(0, 100, 400,   0, 120, 8), r=(6, 100,  60, 8), s=(0, 150, 150, 0, 0), path="read side found"),
    5: dict(what="reads and bitslip off",    w=(3, 100, 400,   0,  60, 8), r=(6, 100,  60, 8), s=(0, 150, 150, 0, 0), path="write bitslip 3 with a new read window"),
    6: dict(what="never passes",             w=None,                        r=(3, 250, 100, 8), s=(0, 40, 40, 3, 250), path="FAILED - three passes did not converge"),
    7: dict(what="read window past tap 511", w=(2, 380, 500,   0, 120, 8), r=(4, 480,  60, 8), s=(0, 40, 40, 4, 480), path=None),
}
FAST_MODULES = [m for m in EYES if EYES[m]["w"] is not None]   # rdlevel.py on the lane that never passes: --runslow

# Wide eyes on every lane, for sdram_init() (write leveling leaves the DQ and DQS at tap 384 in
# the model, so the training takes the write-side search with the reads the MPR leveling set).
WIDE = {m: dict(what="wide", w=(0, 100, 400, 0, 120, 8), r=(3, 250, 100, 8), s=(0, 40, 40, 3, 250), path=None)
        for m in range(MODULES)}

# The library --------------------------------------------------------------------------------------

def litex_software():
    import litex
    return Path(litex.__file__).resolve().parent / "soc/software"


def c_capture(lib, fn):
    """fn()'s result and what the library printed meanwhile (fd 1, around pytest's capture)."""
    lib.c.model_flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as f:
        os.dup2(f.fileno(), 1)
        try:
            result = fn()
            lib.c.model_flush()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        return result, f.read().decode(errors="replace")


class Lib:
    """sdram.c + accessors.c + the model: one shared library, its own model instance."""

    def __init__(self, path):
        self.c = ct.CDLL(str(path), mode=ct.RTLD_LOCAL)
        c = self.c
        c.csr_read_simple.argtypes  = [ct.c_ulong]
        c.csr_read_simple.restype   = ct.c_ulong
        c.csr_write_simple.argtypes = [ct.c_ulong, ct.c_ulong]
        c.csr_write_simple.restype  = None
        c.model_set_write_eye.argtypes = [ct.c_int]*7
        c.model_set_read_eye.argtypes  = [ct.c_int]*5
        c.model_set_state.argtypes     = [ct.c_int]*6
        c.model_get_state.argtypes     = [ct.c_int, ct.POINTER(ct.c_int)]
        c.model_pass_write.argtypes    = [ct.c_int]*4
        c.model_pass_read.argtypes     = [ct.c_int]*3
        c.model_set_dqs_rst_wired.argtypes = [ct.c_int]
        c.model_set_memtest_fail.argtypes  = [ct.c_int]
        for f in ("model_reads", "model_writes", "model_unknown", "model_checks"):
            getattr(c, f).restype = ct.c_ulong
        self.training = hasattr(c, "sdram_write_training")
        if self.training:
            c.sdram_write_training_check.argtypes = [ct.POINTER(ct.c_int)]

    def setup(self, eyes, wired=0):
        c = self.c
        c.model_reset()
        c.model_set_dqs_rst_wired(wired)
        for m, e in eyes.items():
            if e["w"] is not None:
                c.model_set_write_eye(m, *e["w"])
            c.model_set_read_eye(m, *e["r"])
            c.model_set_state(m, *e["s"])
        c.model_reset_counters()

    def state(self, m):
        out = (ct.c_int*5)()
        self.c.model_get_state(m, out)
        return tuple(out)

    def in_eyes(self, m, state):
        return (self.c.model_pass_write(m, *state[:3]) and self.c.model_pass_read(m, *state[3:]))

    def counts(self):
        c = self.c
        return dict(reads=c.model_reads(), writes=c.model_writes(), checks=c.model_checks(),
                    unknown=c.model_unknown())

    def train(self):
        """sdram_write_training() under software control (as rdlevel's train_controller): the
        failing-module mask, the output, the CSR access counts."""
        c = self.c
        c.model_reset_counters()
        def run():
            c.sdram_software_control_on()
            mask = c.sdram_write_training()
            c.sdram_software_control_off()
            return mask
        mask, out = c_capture(self, run)
        return mask, out, self.counts()

    def check(self):
        c = self.c
        c.model_reset_counters()
        errors = (ct.c_int*MODULES)()
        def run():
            c.sdram_software_control_on()
            mask = c.sdram_write_training_check(errors)
            c.sdram_software_control_off()
            return mask
        mask, _ = c_capture(self, run)
        return mask, list(errors), self.counts()

    def init(self, memtest_fail):
        """sdram_init() with the memtest stub failing its first memtest_fail calls."""
        c = self.c
        c.model_set_memtest_fail(memtest_fail)
        ok, out = c_capture(self, c.sdram_init)
        return ok, out, c.model_memtest_calls()


def build(tmp, coarse_seeds=3, training=True, retries=3, after_leveling=False):
    cc = shutil.which("gcc")
    if cc is None:
        pytest.skip("gcc required")
    sw = litex_software()
    if not (sw / "liblitedram/sdram.c").exists():
        pytest.skip("LiteX sources not found")
    lib = tmp / f"wtrain_{'s%d' % coarse_seeds if training else 'off'}_r{retries}{'_after' if after_leveling else ''}.so"
    defs = [f"-DSDRAM_LEVELING_RETRIES={retries}"]
    if training:
        defs += ["-DSDRAM_WRITE_TRAINING", f"-DSDRAM_WRITE_TRAINING_COARSE_SEEDS={coarse_seeds}",
                 f"-DSDRAM_WRITE_TRAINING_STEP={STEP}"]
    if after_leveling:
        defs.append("-DSDRAM_WRITE_TRAINING_AFTER_LEVELING")
    cmd = [cc, "-std=gnu11", "-O1", "-g", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-Wall",
           "-Wno-unused-function", "-I", str(SHIM), "-I", str(sw / "include"), "-I", str(sw), *defs,
           str(HERE / "model.c"), str(sw / "liblitedram/sdram.c"), str(sw / "liblitedram/accessors.c"),
           "-o", str(lib)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, " ".join(cmd) + "\n" + r.stdout + r.stderr
    assert "warning" not in r.stderr, r.stderr
    return Lib(lib)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("wtrain")

@pytest.fixture(scope="module")
def lib3(tmp):
    """Every seed pair on the scans: rdlevel exactly."""
    return build(tmp, coarse_seeds=3)

@pytest.fixture(scope="module")
def lib1(tmp):
    """The default: one seed pair on the scans."""
    return build(tmp, coarse_seeds=1)

@pytest.fixture(scope="module")
def lib_off(tmp):
    """Without SDRAM_WRITE_TRAINING (and the default SDRAM_LEVELING_RETRIES): today's sdram.c."""
    return build(tmp, training=False, retries=1)

@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()

# rdlevel.py on the model --------------------------------------------------------------------------

class ModelBar0(rdlevel.Bar0):
    """rdlevel's BAR0 on the model: the same one-32-bit-access-per-CSR reads and writes (rd128 and
    wr128 inherited), through the library instead of the PCIe resource."""
    lib = None

    def __init__(self, base):   # no PCIe device to open
        self.base = base

    def rd(self, off):
        return self.lib.c.csr_read_simple(CSR_BASE + self.base + off)

    def wr(self, off, v):
        self.lib.c.csr_write_simple(v & 0xffffffff, CSR_BASE + self.base + off)


def rdlevel_train(lib, monkeypatch, modules):
    """rdlevel.train_controller on the model (its csr.h is the shim's): the output and the CSR
    access counts."""
    ModelBar0.lib = lib
    monkeypatch.setattr(rdlevel, "Bar0", ModelBar0)
    lib.c.model_reset_counters()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rdlevel.train_controller(0, step=STEP, modules=modules, csr_h=str(CSR_H))
    return out.getvalue(), lib.counts()


def rdlevel_check(lib, monkeypatch):
    ModelBar0.lib = lib
    monkeypatch.setattr(rdlevel, "Bar0", ModelBar0)
    lib.c.model_reset_counters()
    with contextlib.redirect_stdout(io.StringIO()):
        errors = rdlevel.check_controller(0, csr_h=str(CSR_H))
    return errors, lib.counts()

# Training results ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(lib3, lib1, monkeypatch_module):
    """The eyes trained by the exact C port, the default C port and rdlevel.py; the checks."""
    r = {}
    lib3.setup(EYES)
    r["mask3"], r["out3"], r["counts3"] = lib3.train()
    r["state3"] = {m: lib3.state(m) for m in EYES}
    r["check3"] = lib3.check()                       # on the trained lanes, lane 6 untrained
    r["check_py"] = rdlevel_check(lib3, monkeypatch_module)

    lib1.setup(EYES)
    r["mask1"], r["out1"], r["counts1"] = lib1.train()
    r["state1"] = {m: lib1.state(m) for m in EYES}

    lib3.setup(EYES)
    r["out_py"], r["counts_py"] = rdlevel_train(lib3, monkeypatch_module, FAST_MODULES)
    r["state_py"] = {m: lib3.state(m) for m in EYES}
    return r


def fmt(state):
    return "wbs %d dqs %d dq %d rbs %d rd %d" % state


def test_eyes_are_what_they_say(lib3):
    """The model: the lanes meant for the normal path start with their reads right, the lanes
    meant for the read-side fallbacks with their reads wrong; no CSR access lands nowhere."""
    lib3.setup(EYES)
    for m, e in EYES.items():
        read_ok = lib3.c.model_pass_read(m, *e["s"][3:])
        if e["path"] is None:
            assert read_ok, m
        if e["path"] and "read" in e["path"]:
            assert not read_ok, m
    assert lib3.counts()["unknown"] == 0


def test_c_port_matches_rdlevel_exactly(trained):
    """SDRAM_WRITE_TRAINING_COARSE_SEEDS=3: the same (write bitslip, DQS, DQ, read bitslip, delay)
    per module as rdlevel.py, the same path messages, the lane that never passes in the mask."""
    r = trained
    assert r["mask3"] == 1 << 6, r["out3"]
    assert r["counts3"]["unknown"] == 0 and r["counts_py"]["unknown"] == 0
    for m in FAST_MODULES:
        assert r["state3"][m] == r["state_py"][m], \
            f"m{m} ({EYES[m]['what']}): C {fmt(r['state3'][m])}, rdlevel {fmt(r['state_py'][m])}"
        path = EYES[m]["path"]
        if path is not None:
            assert f"m{m}: {path}" in r["out3"], r["out3"]
            assert f"m{m}: {path}" in r["out_py"], r["out_py"]
        wbs, _, _, rbs, rd = r["state3"][m]
        assert f"  m{m}: write bitslip {wbs}, " in r["out3"], r["out3"]
        assert f"read bitslip {rbs} delay {rd} (window " in r["out3"], r["out3"]
    assert "  m6: FAILED - three passes did not converge" in r["out3"]
    assert "  m6: verification failed after pass 3, retrying" in r["out3"]
    print("\nCSR accesses, step %d: C exact (8 lanes) %s; rdlevel.py (%d lanes) %s"
          % (STEP, r["counts3"], len(FAST_MODULES), r["counts_py"]))


def test_trained_points_are_inside_the_eyes(trained, lib3):
    for m in FAST_MODULES:
        assert lib3.in_eyes(m, trained["state3"][m]), (m, fmt(trained["state3"][m]))


def test_coarse_seeds_land_in_the_same_window(trained, lib1):
    """The default (one seed pair on the scans) picks the same bitslips and a point inside the same
    eyes, within the eyes' marginal taps of the exact run."""
    r = trained
    assert r["mask1"] == 1 << 6, r["out1"]
    for m in FAST_MODULES:
        s1, s3 = r["state1"][m], r["state3"][m]
        assert (s1[0], s1[3]) == (s3[0], s3[3]), f"m{m}: coarse {fmt(s1)}, exact {fmt(s3)}"
        assert lib1.in_eyes(m, s1), (m, fmt(s1))
        margin = max(EYES[m]["w"][5], EYES[m]["r"][3])
        for i in (1, 2, 4):
            d = abs(s1[i] - s3[i]) % 512
            assert min(d, 512 - d) <= 2*margin + 2*STEP, f"m{m}: coarse {fmt(s1)}, exact {fmt(s3)}"
    print("\nCSR accesses, coarse seeds 1: %s (exact: %s)" % (r["counts1"], r["counts3"]))


def test_check_matches_rdlevel(trained):
    """sdram_write_training_check: rdlevel's check, error bit for error bit, the same CSR writes."""
    r = trained
    mask, errors, counts = r["check3"]
    errors_py, counts_py = r["check_py"]
    assert errors == errors_py
    assert mask == 1 << 6
    assert all(errors[m] == 0 for m in FAST_MODULES) and errors[6] > 0
    assert counts["writes"] == counts_py["writes"]
    # liblitedram's phase helpers read RDPHASE/WRPHASE per access (6 per round trip, rdlevel 2),
    # and its software control reads DFII_CONTROL before writing it (on and off).
    assert counts["reads"] == counts_py["reads"] + 4*counts["checks"] + 2


def test_dqs_reset_wired(lib3, monkeypatch):
    """A PHY whose DQS write delay does reset (the pulse rdlevel assumes): the same choices, the
    narrow lane found by the map (the DQ-DQS offset is 0 from the reset)."""
    modules = [0, 1, 2]
    eyes = {m: EYES[m] for m in modules}
    lib3.setup(eyes, wired=1)
    mask, out, _ = lib3.train()
    assert mask & 0b111 == 0, out
    state_c = {m: lib3.state(m) for m in modules}
    lib3.setup(eyes, wired=1)
    out_py, _ = rdlevel_train(lib3, monkeypatch, modules)
    for m in modules:
        assert lib3.state(m) == state_c[m], f"m{m}: C {fmt(state_c[m])}, rdlevel {fmt(lib3.state(m))}"
        assert lib3.in_eyes(m, state_c[m]), (m, fmt(state_c[m]))
        assert f"m{m}: write bitslip" in out_py
    assert "write side found" not in out


def test_sdram_init_relevels_until_the_memtest_passes(lib3):
    """SDRAM_LEVELING_RETRIES=3: a failing memtest runs the leveling (with the training) again,
    attempts printed; the third failure is today's init error."""
    for fail, ok, calls in [(0, 1, 1), (2, 1, 3), (3, 0, 3)]:
        lib3.setup(WIDE)
        result, out, memtests = lib3.init(fail)
        assert (result, memtests) == (ok, calls), out
        assert out.count("Write training:") == calls
        assert ("Memtest failed, leveling again (attempt 2/3)." in out) == (fail >= 1)
        assert ("Memtest failed, leveling again (attempt 3/3)." in out) == (fail >= 2)
        assert "attempt 4/3" not in out
        for m in range(MODULES):
            assert lib3.in_eyes(m, lib3.state(m)), (m, fmt(lib3.state(m)))


def test_training_after_the_stock_leveling(tmp):
    """SDRAM_WRITE_TRAINING_AFTER_LEVELING: the stock stages, then the training."""
    lib = build(tmp, after_leveling=True)
    lib.setup(WIDE)
    result, out, memtests = lib.init(0)
    assert (result, memtests) == (1, 1), out
    stages = ["Write leveling:", "Read leveling (MPR):", "Write latency calibration:",
              "Write DQ-DQS training:", "Read leveling:\n", "Write training:"]
    at = [out.index(s) for s in stages]
    assert at == sorted(at), out
    for m in range(MODULES):
        assert lib.in_eyes(m, lib.state(m)), (m, fmt(lib.state(m)))


def test_without_the_constant_sdram_c_is_unchanged(lib_off):
    """No SDRAM_WRITE_TRAINING: no training symbols, the stock stages, one memtest, no retry."""
    assert not lib_off.training
    lib_off.setup(WIDE)
    result, out, memtests = lib_off.init(1)
    assert (result, memtests) == (0, 1), out
    assert "Write training" not in out and "leveling again" not in out
    for stage in ("Write leveling:", "Read leveling (MPR):", "Write latency calibration:",
                  "Write DQ-DQS training:", "Read leveling:"):
        assert stage in out, out


@pytest.mark.slow
def test_never_passing_lane_through_rdlevel(lib3, monkeypatch):
    """rdlevel.py on every lane including the one that never passes (its three full searches take
    a while through ctypes): the same end state as the C port and the same CSR writes."""
    lib3.setup(EYES)
    mask, out_c, counts_c = lib3.train()
    state_c = {m: lib3.state(m) for m in EYES}
    lib3.setup(EYES)
    out_py, counts_py = rdlevel_train(lib3, monkeypatch, list(EYES))
    assert mask == 1 << 6
    assert "m6: FAILED - three passes did not converge" in out_py
    for m in EYES:
        assert lib3.state(m) == state_c[m], f"m{m}: C {fmt(state_c[m])}, rdlevel {fmt(lib3.state(m))}"
    assert counts_c["writes"] == counts_py["writes"]
    assert counts_c["reads"] == counts_py["reads"] + 4*counts_c["checks"] + 2
    print("\nCSR accesses, 8 lanes, step %d: C %s; rdlevel.py %s" % (STEP, counts_c, counts_py))

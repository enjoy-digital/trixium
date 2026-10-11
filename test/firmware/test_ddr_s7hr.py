#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The YPCB's DDR3 calibration (firmware/ddr_s7hr.c) against a fake channel.

The calibration is compiled for the host with the fake board of test/common/s7hr/s7hr_fake.c in
place of the CSRs: per-lane tDQSS crossings along a fly-by (a write latency step and a failing
zone around each, wider under traffic than for a single burst), per-bit read eyes moving with
the CK phase, some bits framed a CLK off their lane. The program checks the CK phase (inside the
window common to every lane, at its centre), every lane's write latency and every bit's read
framing/tap; a channel without a common window must be refused.
"""

import shutil
import subprocess

import pytest

from common import ROOT, COMMON


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("no host C compiler")
    exe = tmp_path_factory.mktemp("s7hr")/"s7hr_fake"
    subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(COMMON/"s7hr"), "-I", str(ROOT/"firmware"),
        "-o", str(exe), str(COMMON/"s7hr"/"s7hr_fake.c"), "-lm"], check=True)
    return exe


@pytest.mark.parametrize("seed", range(1, 25))
def test_calibration_finds_the_common_window(fake, seed):
    out = subprocess.run([str(fake), str(seed)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:]
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith(("PASS", "SKIP")), out.stdout[-3000:]


def test_calibration_refuses_without_a_window(fake):
    out = subprocess.run([str(fake), "1", "nowindow"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:]
    assert "S7HR: no CK phase where every lane passes" in out.stdout

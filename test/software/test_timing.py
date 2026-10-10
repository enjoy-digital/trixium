#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/lab/timing.py: the timing gate (strict, serializer-skew approvals, experiment envelope) and its CLI."""

import sys
import subprocess

import pytest

from software.lab.timing import parse_full, strict_failures

from common import ROOT


PASS = "0.083 0.000 0 1154468 0.009 0.000 0 1154465 0.003 0.000 0 314334"
V8 = "0.083 0.000 0 1154468 0.009 0.000 0 1154465 -0.003 -0.036 16 314334"


# Helpers ------------------------------------------------------------------------------------------

def report_row(row, trailer=""):
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".rpt", delete=False)
    handle.write("Design Timing Summary\n --- header ---\n" + row + "\n" + trailer)
    handle.close()
    return handle.name


def report(tmp_path, row=PASS, trailer=""):
    path = tmp_path / "timing.rpt"
    path.write_text("Design Timing Summary\n --- header ---\n" + row + "\n" + trailer)
    return path


# Tests --------------------------------------------------------------------------------------------

def test_v8_serializer_violations_rejected(tmp_path):
    parsed = parse_full(report(tmp_path, V8, "Timing constraints are not met.\n"))
    failures = strict_failures(*parsed)
    assert any("serializer-skew" in f and "16/314334" in f for f in failures)


@pytest.mark.parametrize("column,value", [(0, "-0.01"), (1, "-1.0"), (2, "1"),
                                         (4, "-0.01"), (5, "-1.0"), (6, "1"),
                                         (8, "-0.01"), (9, "-1.0"), (10, "1"),
                                         (3, "0"), (7, "0"), (11, "0")])
def test_all_timing_classes_required(tmp_path, column, value):
    fields = PASS.split()
    fields[column] = value
    assert strict_failures(*parse_full(report(tmp_path, " ".join(fields))))


def test_pass_and_explicit_failed_verdict(tmp_path):
    assert strict_failures(*parse_full(report(tmp_path))) == []
    assert strict_failures(*parse_full(report(tmp_path, trailer="Timing constraints are not met.")))


@pytest.mark.parametrize("row", ["", "0.1 0.0 0 123", PASS.replace("0.083", "nan"),
                                 PASS.replace("1154468", "-1")])
def test_incomplete_and_invalid_reports_fail_closed(tmp_path, row):
    assert parse_full(report(tmp_path, row, "Clock Summary\n" + PASS)) is None


def test_cli_strict_not_overridable(tmp_path):
    path = report(tmp_path, V8)
    result = subprocess.run([sys.executable, str(ROOT / "software/lab/timing.py"),
                             "--strict", "--force", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "cannot override" in result.stdout


V8_PLUS = "0.033 0.000 0 1339665 0.010 0.000 0 1339662 -0.003 -0.036 16 383698"


def test_serializer_baseline_waiver_scope(tmp_path):
    # The exact v8 numbers ("v8-exact"); the tolerant "v8" baseline is tested below.
    from software.lab.timing import SERIALIZER_BASELINES
    v8 = SERIALIZER_BASELINES["v8-exact"]
    # Exactly the v8 family on a new build: waived, and reported as such.
    parsed = parse_full(report(tmp_path, V8_PLUS, "Timing constraints are not met.\n"))
    assert strict_failures(*parsed, serializer_baseline=v8) == []
    assert any("serializer-skew" in w for w in strict_failures.waived)
    # Worse on any one number is not waived.
    for column, value in ((8, "-0.004"), (9, "-0.040"), (10, "17")):
        fields = V8_PLUS.split()
        fields[column] = value
        parsed = parse_full(report(tmp_path, " ".join(fields), "Timing constraints are not met.\n"))
        assert strict_failures(*parsed, serializer_baseline=v8)
    # A setup miss is never waived, even with a clean serializer family.
    fields = V8_PLUS.split()
    fields[0], fields[1], fields[2] = "-0.010", "-0.020", "2"
    parsed = parse_full(report(tmp_path, " ".join(fields), "Timing constraints are not met.\n"))
    bad = strict_failures(*parsed, serializer_baseline=v8)
    assert any("setup" in b for b in bad)
    # Without the flag the same report is rejected as before.
    parsed = parse_full(report(tmp_path, V8_PLUS, "Timing constraints are not met.\n"))
    assert strict_failures(*parsed)


def test_cli_waiver_verdict(tmp_path):
    path = report(tmp_path, V8_PLUS, "Timing constraints are not met.\n")
    tool = ROOT / "software/lab/timing.py"
    result = subprocess.run([sys.executable, str(tool), "--strict", "--serializer-baseline", "v8",
                             str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and "PASS WITH WAIVER" in result.stdout and "-0.003" in result.stdout


def test_serializer_tolerance_accepts_placement_drift():
    from software.lab.timing import SERIALIZER_BASELINES
    drifted = "0.033 0.000 0 1339665 0.010 0.000 0 1339662 -0.005 -0.068 16 381185"
    parsed = parse_full(report_row(drifted))
    assert strict_failures(*parsed, serializer_baseline=SERIALIZER_BASELINES["v8"]) == []
    assert strict_failures(*parsed, serializer_baseline=SERIALIZER_BASELINES["v8-exact"])
    for column, value in ((8, "-0.011"), (10, "17")):  # Past the tolerance: still refused.
        fields = drifted.split(); fields[column] = value
        assert strict_failures(*parse_full(report_row(" ".join(fields))),
                               serializer_baseline=SERIALIZER_BASELINES["v8"])


@pytest.mark.parametrize("row,ok", [
    ("-0.506 -722.0 5206 1154468 0.009 0.000 0 1154465 -0.003 -0.036 16 314334", True),   # ran, stable
    ("-0.413 -861.0 7186 1154468 0.009 0.000 0 1154465 0.003 0.000 0 314334", True),     # ran, 5.5 tok/s
    ("-0.819 -16532.0 87320 1154468 0.009 0.000 0 1154465 0.003 0.000 0 314334", False), # killed the host
    ("-1.519 -37174.0 62478 1338955 0.010 0.000 0 1338952 -0.005 -0.068 16 381185", False),  # setup far out
    ("-0.100 -10.0 50 1154468 -0.010 -0.020 3 1154465 0.003 0.000 0 314334", False),     # hold miss
    ("-0.100 -10.0 50 1154468 0.009 0.000 0 1154465 -0.146 -38.8 428 383698", False),    # serializers past the baseline
])
def test_experiment_gate_is_the_host_safety_envelope(row, ok):
    from software.lab.timing import experiment_failures, SERIALIZER_BASELINES
    parsed = parse_full(report_row(row))
    assert (experiment_failures(parsed[0], SERIALIZER_BASELINES["v8"]) == []) == ok


def test_cli_experiment_verdicts(tmp_path):
    tool = ROOT / "software/lab/timing.py"
    near = report(tmp_path, "-0.300 -500.0 3000 1154468 0.009 0.000 0 1154465 -0.004 -0.050 16 314334",
                  "Timing constraints are not met.\n")
    r = subprocess.run([sys.executable, str(tool), "--experiment", "--serializer-baseline", "v8", str(near)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "EXPERIMENT ONLY" in r.stdout
    far = report(tmp_path, "-0.900 -9000.0 30000 1154468 0.009 0.000 0 1154465 0.003 0.000 0 314334")
    r = subprocess.run([sys.executable, str(tool), "--experiment", "--force", str(far)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "DO NOT DEPLOY" in r.stdout


# Board clock coverage -----------------------------------------------------------------------------

CLOCKS_YPCB = """Clock Summary
-------------
Clock                        Waveform(ns)         Period(ns)      Frequency(MHz)
-----                        ------------         ----------      --------------
clk50                        {0.000 10.000}       20.000          50.000
  main_crg_clkout0           {0.000 3.333}        6.667           150.000
  main_crg_clkout1           {0.000 3.750}        7.500           133.333
    main_s7hrddrphy0_outs1   {0.000 0.937}        1.875           533.333
  main_crg_clkout2           {0.000 2.500}        5.000           200.000
pcie_x81_clk_p               {0.000 5.000}        10.000          100.000
Intra Clock Table
"""

CLOCKS_PCIE_ONLY = """Clock Summary
-------------
Clock                        Waveform(ns)         Period(ns)      Frequency(MHz)
pcie_x81_clk_p               {0.000 5.000}        10.000          100.000
  main_s7pciephy_clkout0     {0.000 4.000}        8.000           125.000
Intra Clock Table
"""


def test_board_clock_coverage(tmp_path):
    """A report that does not time the board's clocks is refused before any verdict: the YPCB's
    first build had its 50 MHz input unconstrained, and the summary (the PCIe clocks alone)
    passed the strict gate."""
    from software.lab.timing import gate, missing_clocks
    good = report(tmp_path, trailer=CLOCKS_YPCB)
    assert missing_clocks(good, "ypcb") == []
    assert gate(good, strict=True, board="ypcb") == 0
    bad = tmp_path / "bad.rpt"
    bad.write_text("Design Timing Summary\n --- header ---\n" + PASS + "\n" + CLOCKS_PCIE_ONLY)
    assert missing_clocks(bad, "ypcb") == [20.0, 6.667, 7.5, 1.875, 5.0]
    assert gate(bad, strict=True, board="ypcb") == 1
    assert gate(bad, strict=True) == 0 # Without a board: the summary alone, as before.

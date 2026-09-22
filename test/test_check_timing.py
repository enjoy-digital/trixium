# SPDX-License-Identifier: BSD-2-Clause
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_timing import parse_full, strict_failures


PASS = "0.083 0.000 0 1154468 0.009 0.000 0 1154465 0.003 0.000 0 314334"
V8 = "0.083 0.000 0 1154468 0.009 0.000 0 1154465 -0.003 -0.036 16 314334"
ROOT = Path(__file__).resolve().parents[1]


def report(tmp_path, row=PASS, trailer=""):
    path = tmp_path / "timing.rpt"
    path.write_text("Design Timing Summary\n --- header ---\n" + row + "\n" + trailer)
    return path


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
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_timing.py"),
                             "--strict", "--force", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "cannot override" in result.stdout


def test_real_v8_report_if_available():
    path = ROOT / "prog/bitstreams/xcu1525_n16_150_v8_timing.rpt"
    if not path.exists():
        pytest.skip("local rollback report not present")
    row, failed = parse_full(path)
    assert row[0] > 0 and row[4] > 0 and row[8] < 0
    assert row[10] == 16 and failed
    assert strict_failures(row, failed)


V8_PLUS = "0.033 0.000 0 1339665 0.010 0.000 0 1339662 -0.003 -0.036 16 383698"


def test_serializer_baseline_waiver_scope(tmp_path):
    from tools.check_timing import SERIALIZER_BASELINES
    v8 = SERIALIZER_BASELINES["v8"]
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


def test_cli_waiver_verdict_and_v10(tmp_path):
    path = report(tmp_path, V8_PLUS, "Timing constraints are not met.\n")
    tool = ROOT / "tools/check_timing.py"
    result = subprocess.run([sys.executable, str(tool), "--strict", "--serializer-baseline", "v8",
                             str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and "PASS WITH WAIVER" in result.stdout and "-0.003" in result.stdout
    v10 = ROOT / "build/xcu1525_n16_150_v10_dualdiv_native/gateware/sqrl_xcu1525_mem48_timing.rpt"
    if v10.exists():
        result = subprocess.run([sys.executable, str(tool), "--strict", "--serializer-baseline", "v8",
                                 str(v10)], capture_output=True, text=True)
        assert result.returncode == 1 and "DO NOT SHIP" in result.stdout and "setup" in result.stdout

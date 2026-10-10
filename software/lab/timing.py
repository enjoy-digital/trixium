#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Decide whether a build is safe to put on the card, from its timing report.

A bitstream that misses timing badly can take the host down when it reconfigures the FPGA,
so the deploy path acts on the timing report instead of only printing it.

The thresholds are measured, not guessed: builds with a few thousand failing endpoints around
-0.5 ns WNS ran correct tokens for hours on this design, one with ~87k failing endpoints at
-0.82 ns took the host down on reconfigure. The line is drawn between them, deliberately closer
to the known-good side. This is a heuristic about *this* board and design, not a general rule:
a passing verdict means "no worse than builds that have run", not "meets timing".

  python3 -m software.lab.timing build/fk33/gateware/sqrl_fk33_timing.rpt
  python3 -m software.lab.timing --strict ...  # new releases: setup, hold and pulse/skew closure
  python3 -m software.lab.timing --strict --serializer-baseline v8 ...  # see below
  python3 -m software.lab.timing --experiment --serializer-baseline v8 ...  # concept tests, see below
  python3 -m software.lab.timing --force ...   # deploy anyway, and say so in the log

The targets run the gate after a build (`./trixium_xcu1525.py --build`) and the deploy
runs it again before configuring the card (software/lab/deploy.py); gate() is their entry point.

The experiment gate. To iterate fast, concept work goes on the card even when timing does not
close: a design that works on a near-miss bitstream has shown the concept, and one that fails
is only weak evidence against it (it may be the timing). --experiment passes anything inside
the measured envelope above - setup WNS >= WNS_FLOOR and at most FAILING_CAP failing endpoints
- with hold met and the serializer family inside the baseline, and prints EXPERIMENT ONLY when
the strict gate would refuse it. Outside the envelope it refuses and --force cannot override
it: a reconfigure there can take the remote host down, which costs far more than a failed
experiment.

The serializer-skew waiver. XCU1525 builds do not close the OSERDESE3 CLK/CLKDIV skew checks
in the DDR4 PHYs by a few picoseconds, and builds carrying them have run token gates and DDR
tests on the board for days; requiring their closure for every release means no release.
`--serializer-baseline <name>` scopes the strict gate: setup and hold must fully close, and
the pulse-width/skew family must be NO WORSE than the named baseline; the verdict then says
PASS WITH WAIVER and names the numbers. Anything worse is still DO NOT SHIP, and --force still
cannot override strict mode.
"""

import re
import sys
import math
import argparse

# Thresholds ---------------------------------------------------------------------------------------

WNS_FLOOR   = -0.75   # ns
FAILING_CAP = 20000   # endpoints

# Pulse-width/serializer-skew baselines: (worst slack ns, total ns, failing). The names are the
# builds they were measured on (the XCU1525 target and the deploy pass them).
SERIALIZER_BASELINES = {
    # The 16 OSERDESE3 CLK/CLKDIV endpoints; the family moves by a few picoseconds with
    # placement, so "v8" allows -0.010 ns per endpoint and "v8-exact" keeps the measured numbers.
    "v8"           : (-0.010, -0.160, 16),
    "v8-exact"     : (-0.003, -0.036, 16),
}

# Per-build approvals of a report outside the envelope: (setup WNS floor, serializer baseline).
# Each names one build and its exact numbers, so it cannot drift onto another report.
APPROVALS = {
    # DDR4 past DDR4-1250 on the component-mode PHY (the shipping DDR4-1500, up to DDR4-1600):
    # the ISERDESE3 Min Period overclock (1.600 ns, at most -0.350 ns on the ~940 serializer
    # clock pins) plus the OSERDES skew family; setup keeps the floor.
    "ddr4-overclock"    : (WNS_FLOOR, (-0.351, -360.0, 1000)),
}

# Report Parsing -----------------------------------------------------------------------------------

def parse(path):
    """Return (wns, tns, failing) from the Design Timing Summary's first data row."""
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    i = text.find("Design Timing Summary")
    if i < 0:
        return None
    for line in text[i:].splitlines():
        m = re.match(r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s", line)
        if m:
            return float(m.group(1)), float(m.group(2)), int(m.group(3))
    return None


def parse_full(path):
    """Vivado's twelve-column summary, bounded to its section (fail closed)."""
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    section = text.partition("Design Timing Summary")[2]
    if not section:
        return None
    # Never mistake an individual clock's passing row for the design-wide summary.
    section = section.split("Clock Summary", 1)[0]
    for line in section.splitlines():
        fields = line.split()
        if len(fields) != 12:
            continue
        try:
            row = tuple(float(v) if i % 4 < 2 else int(v)
                        for i, v in enumerate(fields))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in row):
            continue
        if any(v < 0 for i, v in enumerate(row) if i % 4 >= 2):
            continue
        return row, "Timing constraints are not met." in section
    return None

# Gates --------------------------------------------------------------------------------------------

def strict_failures(row, constraints_failed=False, serializer_baseline=None):
    """Return the strict gate's complaints (empty = pass) and, via waivers(), what was waived.

    serializer_baseline: a (worst, total, failing) triple. The pulse-width/skew family is
    then accepted when no worse than it on all three numbers, and Vivado's "constraints
    are not met" is accepted when that family is the only reason for it."""
    bad, waived = [], []
    for offset, name in ((0, "setup"), (4, "hold"), (8, "pulse-width/serializer-skew")):
        slack, total_slack, failing, endpoints = row[offset:offset + 4]
        if slack < 0 or total_slack < 0 or failing != 0 or endpoints == 0:
            text = (f"{name}: worst {slack:+.3f} ns, total {total_slack:.3f} ns, "
                    f"{failing}/{endpoints} failing endpoints")
            if (offset == 8 and serializer_baseline is not None and endpoints
                    and slack >= serializer_baseline[0] and total_slack >= serializer_baseline[1]
                    and failing <= serializer_baseline[2]):
                waived.append(text)
            else:
                bad.append(text)
    if constraints_failed and not waived:
        bad.append("Vivado reports timing constraints are not met")
    elif constraints_failed and bad:
        bad.append("Vivado reports timing constraints are not met")
    strict_failures.waived = waived
    return bad


strict_failures.waived = []


def experiment_failures(row, serializer_baseline=None, wns_floor=WNS_FLOOR):
    """The host-safety envelope for concept experiments: timing need NOT close.

    Setup may fail inside the measured line between builds that ran and the build that
    took the host down on reconfigure (WNS_FLOOR, FAILING_CAP above); hold must pass (a
    hold miss fails at any clock); the serializer family must be inside the baseline.
    A pass means "safe enough to try on the card", never "meets timing": a wrong result
    from such a bitstream is not evidence against the design by itself."""
    bad = []
    wns, _, failing, endpoints = row[0:4]
    if endpoints == 0:
        bad.append("setup: no endpoints (incomplete report)")
    if wns < wns_floor:
        bad.append(f"setup WNS {wns:+.3f} ns is below the {wns_floor:+.3f} ns host-safety floor")
    if failing > FAILING_CAP:
        bad.append(f"{failing} failing setup endpoints is above the {FAILING_CAP} host-safety cap")
    whs, ths, hfail, hend = row[4:8]
    if whs < 0 or ths < 0 or hfail != 0 or hend == 0:
        bad.append(f"hold: worst {whs:+.3f} ns, {hfail}/{hend} failing (hold must pass)")
    pw, ptot, pfail, pend = row[8:12]
    if pw < 0 or ptot < 0 or pfail != 0:
        if serializer_baseline is None or not (pw >= serializer_baseline[0] and ptot >= serializer_baseline[1]
                                                and pfail <= serializer_baseline[2]):
            bad.append(f"pulse-width/serializer-skew: worst {pw:+.3f} ns, {pfail}/{pend} failing")
    return bad

# Gate ---------------------------------------------------------------------------------------------

def gate(report, strict=False, force=False, experiment=False, serializer_baseline=None, approval=None):
    """Print the verdict on a Vivado timing report; return the exit code (0 = deploy)."""
    try:
        full = parse_full(report)
        got  = parse(report)
    except OSError as exc:
        print(f"timing: {exc}")
        return 2
    if experiment:
        if full is None:
            print(f"timing: no complete Design Timing Summary in {report}")
            return 2
        row, failed = full
        baseline = SERIALIZER_BASELINES.get(serializer_baseline)
        floor = WNS_FLOOR
        if approval:
            floor, baseline = APPROVALS[approval]
            print(f"approval {approval}: WNS floor {floor:+.3f} ns, serializer baseline {baseline}")
        strict_bad = strict_failures(row, failed, baseline)
        bad        = experiment_failures(row, baseline, floor)
        if bad:
            print("verdict: DO NOT DEPLOY, not even as an experiment - " + "; ".join(bad))
            print("  Outside the measured envelope a reconfigure has taken the host down; "
                  "--force cannot override this.")
            return 1
        if strict_bad:
            print("verdict: EXPERIMENT ONLY (timing does not close; inside the host-safety envelope) - "
                  + "; ".join(strict_bad))
        else:
            print("verdict: PASS (meets the strict release gate)")
        return 0
    if strict:
        if full is None:
            print(f"timing: no complete Design Timing Summary in {report}")
            return 2
        row, failed = full
        baseline = SERIALIZER_BASELINES.get(serializer_baseline)
        bad    = strict_failures(row, failed, baseline)
        waived = strict_failures.waived
        if bad:
            print("verdict: DO NOT SHIP - " + "; ".join(bad))
            if force:
                print("--force cannot override the strict release gate")
            return 1
        if waived:
            print(f"verdict: PASS WITH WAIVER - {'; '.join(waived)}; within the "
                  f"{serializer_baseline} baseline "
                  f"({baseline[0]:+.3f} ns, {baseline[1]:.3f} ns, {baseline[2]} endpoints) "
                  "that has run on the board; setup and hold fully close")
            return 0
        print("verdict: PASS (setup, hold and pulse-width/skew summary; "
              "constraint coverage still requires review)")
        return 0
    if got is None:
        print(f"timing: no Design Timing Summary in {report}")
        return 2
    wns, tns, failing = got
    print(f"WNS {wns:+.3f} ns   TNS {tns:.1f} ns   {failing} failing endpoints")

    bad = []
    if wns < WNS_FLOOR:
        bad.append(f"WNS {wns:+.3f} is below {WNS_FLOOR:+.2f}")
    if failing > FAILING_CAP:
        bad.append(f"{failing} failing endpoints is above {FAILING_CAP}")
    if not bad:
        if wns < 0:
            print("verdict: SHIP (misses timing, but no worse than builds that have run)")
        else:
            print("verdict: SHIP (legacy setup-only gate; NOT full timing closure)")
        if full and strict_failures(*full):
            print("strict release gate would reject this report")
        return 0

    print("verdict: DO NOT SHIP - " + "; ".join(bad))
    print("  A build this far out took the host down on reconfigure and needed a cold power "
          "cycle. Close timing first (explore directives, or the RTL fix for whatever the\n"
          "  critical path is), or pass --force if you mean it.")
    return 0 if force else 1


# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Decide whether a build is safe to put on the card, from its timing report.")
    parser.add_argument("report",                                                      help="Vivado timing report.")
    parser.add_argument("--strict",              action="store_true",                  help="Require full setup, hold and pulse-width/serializer-skew closure.")
    parser.add_argument("-f", "--force",         action="store_true",                  help="Deploy despite a failing verdict (says so on stdout).")
    parser.add_argument("--experiment",          action="store_true",                  help="Concept experiment: timing need not close, but setup must stay inside the host-safety envelope and hold must pass.")
    parser.add_argument("--serializer-baseline", choices=sorted(SERIALIZER_BASELINES), help="Strict mode: accept serializer-skew violations no worse than this board-validated build's.")
    parser.add_argument("--approval",            choices=sorted(APPROVALS),            help="Experiment mode: per-build approval (its own WNS floor and serializer baseline).")
    args = parser.parse_args(argv)
    return gate(args.report,
        strict              = args.strict,
        force               = args.force,
        experiment          = args.experiment,
        serializer_baseline = args.serializer_baseline,
        approval            = args.approval)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Run the test suite to completion, in parallel, on a machine shared with Vivado and simulators.

One pytest process per test file (per test for the files listed in SPLIT and for the files
the previous run recorded as longer than --split-above seconds), so peak memory is the
largest units running together rather than everything pytest has accumulated, and a unit
that has to be stopped loses only itself.

Up to -j units run at once, longest first (from the record of the previous run). A unit
starts only when the memory it is expected to need fits: its peak RSS from the record
(--min-free GB when it has none) plus --margin GB must be available after the units already
running have grown to their own expected peaks, and the expected peaks of everything running
stay under --max-gb. While units run, a watchdog stops the most recent one when availability
falls below --kill-free GB (protecting a concurrent route); stopped units are retried one at a
time at the end. Every unit's outcome, duration and peak RSS go to the summary and update the
record (<out>/record.json) the next run schedules from.

  python3 test/run_guarded.py                                       # fast tier (build/tests)
  python3 test/run_guarded.py --runslow                             # full tier
  python3 test/run_guarded.py -j 4 test/gateware/units              # every test file below
  python3 test/run_guarded.py --out build/t test/model/test_asm.py -- --durations=0

Arguments after `--` go to every pytest process. Exit 0 only when every unit passed (skips
inside a unit are pytest's own business).
"""

import os
import re
import sys
import glob
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path

# Constants ----------------------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[1]
SPLIT    = ("test/gateware/memory/test_ddr.py",)   # ~9.5 GB per test in Migen's simulator
DESELECT = ()

# Helpers ------------------------------------------------------------------------------------------

def find_test_files(paths):
    """The test files of `paths` (files, or directories searched recursively), relative to ROOT."""
    files = []
    for path in paths or ["test"]:
        if (ROOT / path).is_dir():
            files += sorted(glob.glob(os.path.join(path, "**", "test_*.py"), root_dir=ROOT, recursive=True))
        else:
            files.append(path)
    return files


def available_gb():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])/1024/1024
    return 0.0


def group_rss_gb(pgid):
    """Current RSS of every process of the session `pgid` (a unit's pytest and its children)."""
    total = 0
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(stat, encoding="utf-8") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid:
            total += int(fields[21])*os.sysconf("SC_PAGE_SIZE")
    return total/1e9


def node_ids(path, pytest_args):
    out = subprocess.run([sys.executable, "-m", "pytest", path, "--collect-only", "-p", "no:cacheprovider"] +
                         pytest_args, cwd=ROOT, capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if "::" in line and line.startswith("test/")]


def load_record(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

# Units --------------------------------------------------------------------------------------------

class Unit:
    def __init__(self, name, expected_gb, expected_s):
        self.name        = name
        self.expected_gb = expected_gb
        self.expected_s  = expected_s
        self.attempt     = 0

    def start(self, out, pytest_args, deselect):
        self.attempt += 1
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)[:180]
        self.log_path = out / "logs" / f"{stem}.log"
        self.rss_path = out / "logs" / f"{stem}.rss"
        # /usr/bin/time gives this process's own peak; RUSAGE_CHILDREN is a running maximum
        # over every child so far and would hide any unit lighter than an earlier one.
        cmd = ["/usr/bin/time", "-o", str(self.rss_path), "-f", "%M",
               sys.executable, "-m", "pytest", self.name, "-p", "no:cacheprovider"] + pytest_args
        for d in deselect:
            if d.split("::")[0] == self.name.split("::")[0]:
                cmd += ["--deselect", d]
        self.log    = open(self.log_path, "w", encoding="utf-8")
        self.proc   = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.begin  = time.time()
        self.killed = False

    def kill(self):
        self.killed = True
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def still_needs_gb(self):
        """What this unit may still grow by before reaching its expected peak."""
        return max(0.0, self.expected_gb - group_rss_gb(self.proc.pid))

    def result(self):
        self.log.close()
        try:
            peak = round(int(self.rss_path.read_text(encoding="utf-8").split()[-1])/1e6, 2)
        except (OSError, ValueError, IndexError):
            peak = None              # killed before /usr/bin/time could write it
        return dict(unit=self.name, rc=self.proc.returncode, killed=self.killed, attempt=self.attempt,
                    seconds=round(time.time() - self.begin, 1), peak_gb=peak)

# Scheduler ----------------------------------------------------------------------------------------

def run_units(queue, jobs, args, out, pytest_args, deselect, results):
    """Run `queue` (longest first) with up to `jobs` units at once; return the killed units."""
    running, killed = [], []
    log = open(out / "pytest.log", "a", encoding="utf-8")
    while queue or running:
        # Reap finished units.
        for unit in list(running):
            if unit.proc.poll() is None:
                continue
            running.remove(unit)
            r = unit.result()
            log.write(f"\n===== {unit.name} (attempt {unit.attempt})\n")
            log.write(unit.log_path.read_text(encoding="utf-8", errors="replace"))
            log.flush()
            status = "KILLED" if r["killed"] else ("ok" if r["rc"] == 0 else f"FAIL rc={r['rc']}")
            print(f"{status:10s} {r['seconds']:7.1f}s  peak {r['peak_gb'] or '-':>5} GB  {unit.name}", flush=True)
            if r["killed"]:
                killed.append(unit)
            results.append(r)

        # Watchdog: stop the most recent unit while memory runs out.
        alive = [u for u in running if not u.killed]
        if alive and available_gb() < args.kill_free:
            alive[-1].kill()
            time.sleep(5)
            continue

        # Admission: start the next unit whose expected peak fits.
        started = False
        if queue and len(running) < jobs:
            reserved = sum(u.still_needs_gb() for u in running)
            budget   = sum(u.expected_gb for u in running)
            for unit in queue:
                fits_free = available_gb() - reserved >= unit.expected_gb + args.margin
                fits_cap  = budget + unit.expected_gb <= args.max_gb
                if (fits_free and fits_cap) or (not running and available_gb() >= args.kill_free + args.margin):
                    queue.remove(unit)
                    unit.start(out, pytest_args, deselect)
                    running.append(unit)
                    started = True
                    break
        if not started:
            time.sleep(1 if running else 5)
    log.close()
    return killed

# Run ----------------------------------------------------------------------------------------------

def main():
    argv, pytest_args = sys.argv[1:], []
    if "--" in argv:
        argv, pytest_args = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files",         nargs="*",                     help="Test files or directories (default: every test/**/test_*.py).")
    parser.add_argument("-j", "--jobs",  type=int, default=max(1, (os.cpu_count() or 2)//2), help="Units run at once (default: cores/2).")
    parser.add_argument("--out",         default="build/tests",         help="Output directory (logs, summary, record).")
    parser.add_argument("--runslow",     action="store_true",           help="Full tier: include the tests marked slow.")
    parser.add_argument("--min-free",    type=float, default=4.0,       help="Expected peak GB of a unit the record does not know.")
    parser.add_argument("--margin",      type=float, default=10.0,      help="GB left available after every running unit reaches its peak.")
    parser.add_argument("--max-gb",      type=float, default=40.0,      help="Cap on the sum of the running units' expected peaks.")
    parser.add_argument("--kill-free",   type=float, default=6.0,       help="GB available below which the most recent unit is stopped.")
    parser.add_argument("--split-above", type=float, default=240.0,     help="Run a file per test when the record says it took longer (s).")
    parser.add_argument("--with-27b",    action="store_true",           help="Include the local-27B GGUF test.")
    args = parser.parse_args(argv)
    if args.runslow:
        pytest_args = ["--runslow"] + pytest_args
    out = ROOT / args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "pytest.log").write_text("", encoding="utf-8")
    record_path = out / "record.json"
    record      = load_record(record_path)
    deselect    = () if args.with_27b else DESELECT

    # Units: files, or their tests for the long/heavy ones.
    names = []
    for f in find_test_files(args.files):
        rec = record.get(f, {})
        if f in SPLIT or rec.get("seconds", 0) > args.split_above or rec.get("split"):
            names += node_ids(f, pytest_args) or [f]
        else:
            names.append(f)
    units = []
    for name in names:
        rec = record.get(name, {})
        units.append(Unit(name, rec.get("peak_gb") or args.min_free, rec.get("seconds", 60.0)))
    units.sort(key=lambda u: -u.expected_s)

    start   = time.time()
    results = []
    killed  = run_units(units, args.jobs, args, out, pytest_args, deselect, results)
    if killed:
        print(f"\nRetrying {len(killed)} stopped unit(s) one at a time.", flush=True)
        run_units(killed, 1, args, out, pytest_args, deselect, results)
    wall = time.time() - start

    # Record: keep the last completed measurement of every unit (a file that was split keeps a
    # flag so the next run splits it without re-measuring the whole file).
    for r in results:
        if not r["killed"] and r["peak_gb"] is not None:
            record[r["unit"]] = dict(seconds=r["seconds"], peak_gb=r["peak_gb"])
            if "::" in r["unit"]:
                record.setdefault(r["unit"].split("::")[0], {})["split"] = True
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")

    final  = {r["unit"]: r for r in results}         # A retried unit's last attempt wins.
    failed = [r for r in final.values() if r["rc"] != 0 or r["killed"]]
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    busy = sum(r["seconds"] for r in results)
    print(f"\n{len(final) - len(failed)}/{len(final)} units passed in {wall:.0f} s wall "
          f"({busy:.0f} s of unit time, {busy/max(wall, 1):.1f}x)" +
          (": " + ", ".join(r["unit"] for r in failed) if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Run the test suite to completion on a machine shared with Vivado and simulators.

One pytest process per test file (per test for the files listed in SPLIT), so peak memory
is the largest single unit rather than everything pytest has accumulated, and a unit
that has to be stopped loses only itself. Before each unit the runner waits until
--min-free GB are available; while a unit runs, a watchdog stops it if availability
falls below --kill-free GB (protecting a concurrent route), and stopped units are retried
at the end. Every unit's outcome, duration and peak RSS go to the summary.

  python3 test/run_guarded.py                                       # whole suite (build/tests)
  python3 test/run_guarded.py --out build/t test/model/test_asm.py  # selected files
  python3 test/run_guarded.py test/gateware/units                   # every test file below

Exit 0 only when every unit passed (skips inside a unit are pytest's own business).
Whole-suite runs in one process get killed by the memory guard during a route:
test/gateware/memory/test_ddr.py alone peaks at 21 GB in one process.
"""

import os
import sys
import glob
import json
import time
import signal
import argparse
import threading
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


def node_ids(path):
    out = subprocess.run([sys.executable, "-m", "pytest", path, "--collect-only", "-p", "no:cacheprovider"],
                         cwd=ROOT, capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if "::" in line and line.startswith("test/")]

# Units --------------------------------------------------------------------------------------------

def run_unit(unit, log, kill_free, deselect, rss_file):
    # /usr/bin/time gives this process's own peak; RUSAGE_CHILDREN is a running maximum
    # over every child so far and would hide any unit lighter than an earlier one.
    cmd = ["/usr/bin/time", "-o", str(rss_file), "-f", "%M",
           sys.executable, "-m", "pytest", unit, "-p", "no:cacheprovider"]
    for d in deselect:
        if d.split("::")[0] == unit.split("::")[0]:
            cmd += ["--deselect", d]
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    killed = threading.Event()

    def watchdog():
        while proc.poll() is None:
            if available_gb() < kill_free:
                killed.set()
                os.killpg(proc.pid, signal.SIGTERM)
                return
            time.sleep(5)

    threading.Thread(target=watchdog, daemon=True).start()
    rc = proc.wait()
    try:
        peak = round(int(Path(rss_file).read_text(encoding="utf-8").split()[-1])/1e6, 2)
    except (OSError, ValueError, IndexError):
        peak = None                  # killed before /usr/bin/time could write it
    return dict(unit=unit, rc=rc, killed=killed.is_set(), seconds=round(time.time() - start, 1),
                peak_gb=peak)

# Run ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files",       nargs="*",                 help="Test files or directories (default: every test/**/test_*.py).")
    parser.add_argument("--out",       default="build/tests",     help="Output directory (log, summary).")
    parser.add_argument("--min-free",  type=float, default=20.0,  help="GB available before a unit starts.")
    parser.add_argument("--kill-free", type=float, default=8.0,   help="GB available below which a unit is stopped.")
    parser.add_argument("--with-27b",  action="store_true",       help="Include the local-27B GGUF test.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = find_test_files(args.files)
    deselect = () if args.with_27b else DESELECT
    units = []
    for f in files:
        units += node_ids(f) if f in SPLIT else [f]
    log = open(out / "pytest.log", "w", encoding="utf-8")
    results, retry = [], []
    for attempt, queue in ((1, units), (2, retry)):
        for unit in list(queue):
            while available_gb() < args.min_free:
                time.sleep(15)
            log.write(f"\n===== {unit} (attempt {attempt})\n")
            log.flush()
            r = run_unit(unit, log, args.kill_free, deselect, out / "rss.txt")
            r["attempt"] = attempt
            status = "KILLED" if r["killed"] else ("ok" if r["rc"] == 0 else f"FAIL rc={r['rc']}")
            print(f"{status:10s} {r['seconds']:7.1f}s  peak {r['peak_gb'] or '-':>5} GB  {unit}", flush=True)
            if r["killed"] and attempt == 1:
                retry.append(unit)
            else:
                results.append(r)
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failed = [r for r in results if r["rc"] != 0 or r["killed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} units passed" +
          (": " + ", ".join(r["unit"] for r in failed) if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

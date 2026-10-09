#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Full release validation (hours): both bitstreams, the test suite, the simulation gate, then
the XCU1525 board with the three models and the FK33 board with two. Every step logs to
build/validate/<step>.log and the summary goes to build/validate/summary.txt; a failed step does
not stop the independent ones.

  trixium validate                                  # everything
  trixium validate --steps sim,tests                # a subset of: build-xcu1525 build-fk33
                                                    #   tests sim board board-fk33

Long: run it detached, e.g.
  systemd-run --user -E PATH="$PATH" -E HOME="$HOME" --working-directory=$PWD trixium validate

Board steps: the hosts of ~/.config/trixium/boards.ini (key-based ssh, software/deploy.py).
DDR4-1500 is an approved out-of-spec overclock (the XCU1525 build and deploy gate it under the
ddr1600-overclock approval); the reference is v30b: 27B 32.2 M cycles/token, tokens
"11751 13 198 760 6511 314 9564 369". FK33 reference v16b: 1.7B 2,101k cycles/token,
101.2 tok/s at 212.5 MHz; 8B 30.0 tok/s.
"""

import os
import re
import sys
import glob
import time
import shlex
import argparse
import threading
import subprocess

# Paths --------------------------------------------------------------------------------------------

REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ["build-xcu1525", "build-fk33", "tests", "sim", "board", "board-fk33"]

# Helpers ------------------------------------------------------------------------------------------

class Validation:
    def __init__(self, out):
        self.out     = out
        self.summary = os.path.join(out, "summary.txt")
        self.lock    = threading.Lock()
        os.makedirs(out, exist_ok=True)
        open(self.summary, "w", encoding="utf-8").close()

    def note(self, step, text):
        line = f"{step:16s} {text}"
        with self.lock:
            print(line, flush=True)
            with open(self.summary, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log(self, name):
        return os.path.join(self.out, f"{name}.log")

    def run(self, name, cmd, append=False):
        """Run cmd from the repository root, its output to <name>.log; returns the exit code."""
        with open(self.log(name), "a" if append else "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            f.flush()
            return subprocess.call(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)

    def start(self, name, cmd):
        """Start cmd in the background, its output to <name>.log."""
        f = open(self.log(name), "w", encoding="utf-8")
        return subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)

    def grep(self, name, pattern, n=2):
        try:
            with open(self.log(name), encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if re.search(pattern, l)]
        except OSError:
            return ""
        return " ".join(lines[-n:])


def free_gb():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // (1 << 20)
    return 0

# Steps --------------------------------------------------------------------------------------------

def utilization(build):
    for path in glob.glob(os.path.join(REPO, "build", build, "gateware", "*utilization_place.rpt")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if re.match(r"\| (CLB LUTs|LUT as Logic)", line):
                    return " ".join(line.split())
    return ""


def tests(v):
    rc = v.run("tests", [sys.executable, "test/run_guarded.py", "--out", os.path.join(v.out, "tests"), "--min-free", "16"])
    with open(v.log("tests"), encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()[1:]
    ok = sum(l.startswith("ok") for l in lines)
    v.note("tests", f"{'PASS' if rc == 0 else 'FAIL'} ({ok} units ok, {len(lines) - ok} other lines)")


def sim(v):
    rc = v.run("sim", [sys.executable, "trixium_sim.py", "--build"])
    rc = rc or v.run("sim", [sys.executable, "trixium_sim.py", "--gate"], append=True)
    result = v.grep("sim", r"[0-9]+ checksums identical, [0-9]+ different", 1)
    ok     = rc == 0 and "checksums identical, 0 different" in result
    v.note("sim", f"{'PASS' if ok else 'FAIL'} ({result})")
    rc = v.run("sim-fk33", [sys.executable, "trixium_sim.py", "--build", "--board", "fk33",
        "--image", "models/tiny/bonsai-tiny-n8.img"])
    v.note("sim-fk33", "BUILT" if rc == 0 else "BUILD FAIL")


def deploy(v, name, *args):
    return v.run(name, [sys.executable, "-m", "software.cli", "deploy", *args])


def smi(v, board, name):
    from software.deploy import board_config
    conf = board_config(board)
    cmd  = ["ssh", "-o", "BatchMode=yes", conf["target"], f"{shlex.quote(conf['dir'])}/user/trixium_smi -i 1"]
    v.run(name, cmd)


def board_xcu1525(v):
    # The 27B through the full deploy, then the 8B and the 1.7B on the same configuration.
    deploy(v, "board-27b", "xcu1525", "--board", "xcu1525", "--model", "27b")
    v.note("board-27b", v.grep("board-27b", r"^GEN:|TOKENS (OK|WRONG)|cycles/token"))
    for m in ["8b", "1p7b"]:
        deploy(v, f"board-{m}", "xcu1525", "--board", "xcu1525", "--model", m, "--no-reconfigure")
        v.note(f"board-{m}", v.grep(f"board-{m}", r"^GEN:|TOKENS (OK|WRONG)|cycles/token"))
    smi(v, "xcu1525", "smi-xcu1525")


def board_fk33(v):
    # The 1.7B and the 8B, each through the full deploy.
    for m in ["1p7b", "8b"]:
        deploy(v, f"board-fk33-{m}", "fk33", "--board", "fk33", "--model", m)
        v.note(f"board-fk33-{m}", v.grep(f"board-fk33-{m}", r"TOKENS (OK|WRONG)|BENCH"))
    smi(v, "fk33", "smi-fk33")

# Validate -----------------------------------------------------------------------------------------

def validate(steps, out):
    v = Validation(out)

    # Bitstreams (in parallel: a VU9P route peaks ~25 GB, the VU33P ~15 GB). Each target runs its
    # own timing gate and writes build/<name>.verdict.
    builds = {}
    if "build-xcu1525" in steps:
        builds["xcu1525"] = v.start("build-xcu1525", [sys.executable, "trixium_xcu1525.py", "--build", "--output-dir", "build/xcu1525"])
    if "build-fk33" in steps:
        while free_gb() < 20:
            time.sleep(60)
        builds["fk33"] = v.start("build-fk33", [sys.executable, "trixium_fk33.py", "--build", "--output-dir", "build/fk33"])

    # Tests and simulation, while the bitstreams build.
    if "tests" in steps:
        tests(v)
    if "sim" in steps:
        sim(v)

    # Bitstream verdicts.
    for b, p in builds.items():
        rc = p.wait()
        verdict = "PASS (timing gate)" if rc == 0 else f"verdict {rc} (see {v.log('build-' + b)})"
        if b == "xcu1525" and rc == 0:
            verdict = "PASS (timing gate, ddr1600-overclock approval)"
        v.note(f"build-{b}", f"{verdict}; {utilization(b)}")

    # Boards (in parallel: two hosts).
    threads = []
    if "board" in steps:
        threads.append(threading.Thread(target=board_xcu1525, args=(v,)))
    if "board-fk33" in steps:
        threads.append(threading.Thread(target=board_fk33, args=(v,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"summary: {v.summary}")
    return 0

# Run ----------------------------------------------------------------------------------------------

def add_arguments(parser):
    parser.add_argument("--steps", default=",".join(STEPS), help="Comma-separated subset of: " + ", ".join(STEPS) + ".")
    parser.add_argument("--out",   default="build/validate", help="Logs and summary directory.")


def run(args):
    steps = [s for s in args.steps.split(",") if s]
    for s in steps:
        if s not in STEPS:
            raise SystemExit(f"unknown step {s} (steps: {', '.join(STEPS)})")
    return validate(steps, os.path.join(REPO, args.out))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full release validation (bitstreams, tests, simulation, boards).")
    add_arguments(parser)
    sys.exit(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()

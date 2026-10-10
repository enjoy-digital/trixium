#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Drive a built Verilator simulation through its console (./trixium_sim.py --gate / --bench).

gate : the Python mirror on the image, then one traced generation on the simulated card with the
       resident path on (`dbg N; gen 1 <ids>`), then a checkpoint-by-checkpoint comparison
       (software/lab/mirror.py): every checkpoint must be bit-identical, the resident path on (and
       the unit 11 FFN activation lists when the SoC has them). The firmware's `stats` profile
       (hybprof: cycles per forward) is printed after the run.
bench: the firmware's own cycles/forward with a console toggle (resident, silu, dnstream) off
       then on. The default simulation's DDR4 model is not the board's: treat the ratio as a
       signal about the CPU work removed, never as a board number. With toggle none on a
       board-timing simulation of a shape-true slice: one run, the W/P split, the per-layer
       comparison with the board and the predicted board tok/s (software/lab/simcal.py).

The firmware's boot legs (`BOOT-LEGS (q to skip)`, ~70 min of Verilator time on the 1k model) are
skipped: these runs only need the console.
"""

import os
import re
import time

import pexpect

from software.lab.mirror import mirror, compare

# Constants ----------------------------------------------------------------------------------------

READY   = r"TRIXIUM-READY[ \t]*\r?\n"
PROMPT  = r"trixium> "
TIMEOUT = 3*3600 # Boot / generation timeout (s).

# Console ------------------------------------------------------------------------------------------

def spawn(gateware_dir, out_dir):
    binary = os.path.join(os.path.abspath(gateware_dir), "obj_dir", "Vsim")
    if not os.path.exists(binary):
        raise SystemExit(f"no simulator at {binary}: build it first (./trixium_sim.py --build).")
    os.makedirs(out_dir, exist_ok=True)
    child = pexpect.spawn(binary, cwd=gateware_dir, encoding="utf-8", codec_errors="replace",
        timeout=TIMEOUT, maxread=65536, echo=False)
    child.logfile_read = open(os.path.join(out_dir, "console.log"), "w", encoding="utf-8")
    return child


def boot(child):
    """Skip the boot legs, wait for TRIXIUM-READY and return the boot output.

    The skip key is only sent once the firmware offers it: earlier, it reaches the LiteX BIOS,
    where Q aborts the boot. A firmware without the offer goes straight to READY."""
    start = time.time()
    index = child.expect([r"BOOT-LEGS \(q to skip\)", READY])
    text  = child.before + child.after
    if index == 0:
        child.send("q")
        child.expect(READY)
        text += child.before + child.after
    print(f"console ready after {time.time() - start:.0f} s", flush=True)
    return text


def command(child, cmd, timeout=300):
    """Run one console command and return its output, consuming exactly its own prompt."""
    child.sendline(cmd)
    child.expect(PROMPT, timeout=timeout)
    return child.before

# Gate ---------------------------------------------------------------------------------------------

def gate(gateware_dir, image, out_dir, ids=(1, 2, 3), dbg=3, expect_silu=False):
    """Mirror vs traced simulation run; returns 0 when everything matches."""
    os.makedirs(out_dir, exist_ok=True)
    mirror_trace = os.path.join(out_dir, "mirror.txt")
    sim_trace    = os.path.join(out_dir, "trace.txt")
    ids          = [str(i) for i in ids]

    # Mirror.
    mirror(image, [int(i) for i in ids], mirror_trace)

    # Simulation.
    child = spawn(gateware_dir, out_dir)
    try:
        boot_text = boot(child)
        child.sendline(f"dbg {dbg}")
        child.expect(r"DBG armed for \d+ forwards", timeout=60)
        # `gen` takes the token COUNT first, then the prompt ids: "gen 1 1 2 3" is one token from
        # [1, 2, 3] (sending "gen 1 2 3" traces one forward fewer than the mirror).
        child.sendline("gen 1 " + " ".join(ids))
        child.expect(r"GEN (DONE|ERR)")
        trace = child.before + child.after
        child.expect(PROMPT)
        stats = command(child, "stats")
    finally:
        child.sendline("")
        child.close(force=True)
    with open(sim_trace, "w", encoding="utf-8") as f:
        f.write(trace)

    # Checks.
    ok = "GEN DONE" in trace
    tokens = re.search(r"(-?\d+(?:[ \t]+-?\d+)*)\s*GEN DONE", trace) # The line before GEN DONE.
    print("tokens:", tokens.group(1) if tokens else "(none)")
    resident = "resident projections" in boot_text and "resident path off" not in trace
    print("resident path:", "on" if resident else "OFF/absent")
    ok &= resident
    if expect_silu:
        silu = re.search(r"siluprof: per fwd (\d+) ffn lists", stats)
        silu = int(silu.group(1)) if silu and "resident FFN activation" in boot_text else 0
        print("SiLU FFN lists per forward:", silu)
        ok &= silu > 0
    for line in stats.splitlines():
        if line.strip().startswith("hybprof:"):
            print(line.strip())
    ok &= compare(mirror_trace, sim_trace) == 0
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1

# Bench --------------------------------------------------------------------------------------------

PROFILE_LINES = r"^(resprof|siluprof|listprof|hybprof|attnprof|floor|dnprof|dnhw|quant8|norm_i32|embed|lmhead|overlap|total \(sum)"

def bench(gateware_dir, out_dir, toggle="resident", tokens=2, image=None):
    """A/B a console toggle on cycles/forward; returns 0 when both runs reported.

    toggle "none": one run as built, then the W/P split and, for a shape-true slice image (a
    board-timing simulation), the per-layer normalization, the full-model board prediction and
    the comparison with the board (software/lab/simcal.py)."""
    child   = spawn(gateware_dir, out_dir)
    results = {}
    modes   = ("as-built",) if toggle == "none" else ("off", "on")
    try:
        boot(child)
        child.expect(PROMPT)
        for mode in modes:
            if toggle != "none":
                answer = command(child, f"{toggle} {mode}")
                if f"{toggle.upper()} {mode}" not in answer:
                    raise SystemExit(f"{toggle} {mode} was not acknowledged: {answer[-200:]!r}")
            command(child, "stats reset")
            text  = command(child, f"bench {tokens}", timeout=2*TIMEOUT)
            text += command(child, "stats", timeout=900)
            with open(os.path.join(out_dir, f"bench-{mode}.log"), "w", encoding="utf-8") as f:
                f.write(text)
            cycles = re.findall(r"(\d+) cycles/fwd", text)
            results[mode] = int(cycles[-1]) if cycles else None
            print(f"{toggle} {mode:3s}: {results[mode] or 'no BENCH line'} cycles/fwd", flush=True)
            for line in text.splitlines():
                if re.match(PROFILE_LINES, line.strip()):
                    print("   ", line.strip(), flush=True)
    finally:
        child.sendline("")
        child.close(force=True)
    if toggle == "none":
        if not results.get("as-built"):
            return 1
        if image is not None:
            from software.lab import simcal
            with open(os.path.join(out_dir, "bench-as-built.log"), encoding="utf-8") as f:
                simcal.report(simcal.parse_bench(f.read()), image)
        return 0
    off, on = results.get("off"), results.get("on")
    if not (off and on):
        return 1
    print(f"cycles/forward: {toggle} off {off}, on {on} ({off/on:.3f}x, {100*(on - off)/off:+.1f}%)")
    return 0

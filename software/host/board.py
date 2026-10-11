#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The board-side session, run ON THE BOARD HOST as root from the deploy directory (the deploy
copies software/ there: `trixium deploy`, software/lab/deploy.py).

  sudo python3 -m software.host.board reconfigure --board xcu1525 prog/csr.bit
  sudo python3 -m software.host.board boot 27b [--firmware trixium.bin]
  sudo python3 -m software.host.board boot 1.7b --board fk33
  sudo python3 -m software.host.board boot 1.7b --board ypcb
  sudo python3 -m software.host.board load bonsai-1p7b-n16.img
  sudo python3 -m software.host.board gen 16 785 6722 315 9625 374 [--want "12095 11 ..."]
  sudo python3 -m software.host.board model bonsai-27b-n16-xcu1525.img 16 760 6511 314 9338 369
  sudo python3 -m software.host.board bench 4
  sudo python3 -m software.host.board abtoggle silu [-n 4] [--rounds 2] [--want "..."]

`reconfigure`: drivers out, PCI device removed, the bitstream programmed (XCU1525:
openFPGALoader, then the JTAG IR capture must read 0x35d75 = DONE in the three SLRs; FK33:
OpenOCD pld load), then a warm reboot (PERST# and the PCIe link come back with it). Exits 1
when the FPGA is not configured: never go on against a removed device.

`boot` on the XCU1525 is the DDR4 boot order: CPU reset -> BIOS prompt -> host-side training of
controller 0 -> memtest -> firmware boot without a CPU reset (-R) -> the firmware calibrates
controllers 1..3 -> host-side training of 1..3 -> engreset -> ddrtest. The host-side training
(software/host/rdlevel.py) must never touch controller 0 while the firmware runs from it: `boot`
trains it between the BIOS's calibration and the jump. On the FK33: trixium_chat -k (CPU reset,
serial boot), retried once. On the YPCB-00338 there is no host training: the BIOS calibrates DDR3
controller 0 itself (the S7HR CK-phase scan, firmware/ddr_s7hr.c), the firmware boots without a
CPU reset and calibrates controller 1 the same way, then engreset -> ddrtest.
"""

import io
import re
import sys
import time
import argparse
import subprocess
import contextlib

from software.host import console
from software.host import rdlevel
from software.host.cpumon import cpumon
from software.host.cpurst import cpu_reset, find_xilinx_device

# Constants ----------------------------------------------------------------------------------------

PROMPT = "trixium> "

# The XCU1525's JTAG is the board's FT4232H (0403:6011, "Xilinx CB-U1"), channel A, plain MPSSE.
# openocd 0.11's virtex2 pld driver drives a single 6-bit IR and leaves this three-SLR device
# unconfigured (IR capture 0x11 per SLR) while reporting "loaded": programming goes through
# openFPGALoader and openocd only reads the IR capture (18 bits, three 6-bit SLR registers):
# 0x35 = DONE+INIT+ISC_DONE per SLR, 0x35d75 for the three. openocd 0.11 syntax.
XCU1525_IR_CHECK_CFG = """\
interface ftdi
ftdi_vid_pid 0x0403 0x6011
ftdi_channel 0
ftdi_layout_init 0x0008 0x000b
reset_config none
adapter_khz 5000
set _CHIPNAME xcvu9p
jtag newtap $_CHIPNAME tap -irlen 18 -expected-id 0x04b31093 -ignore-version -ircapture 0x00 -irmask 0x3ffff
"""
XCU1525_CONFIGURED = "0x35d75"

# The YPCB-00338's JTAG: an external Digilent HS2-style FT232H cable on the card's header (its
# chain shows the FPGA alone; a Xilinx Platform Cable's chain also has the card's CPLD).
YPCB_PROGRAM = ["openFPGALoader", "-c", "digilent_hs2", "--fpga-part", "xc7k480tffg1156"]

BENCH_LINES = (r"BENCH|SPTEST|^(op|embed|rmsnorm|qknorm|kvquant|scores|wv|quant8|norm_i32|silu_mul|"
    r"resid_add|lmhead|engine_wait|total|poll|dnhw|hybprof|dnprof|attnprof|floor|resprof|siluprof|"
    r"listprof|overlap)")

# Helpers ------------------------------------------------------------------------------------------

def say(text):
    print(text, flush=True)


def grep(text, pattern, limit=None):
    """Print the lines of `text` matching `pattern`; returns them."""
    lines = [l for l in text.splitlines() if re.search(pattern, l)]
    for l in lines[:limit]:
        say(l)
    return lines


def con(cmds, wait=20.0, quiet_after=2.0, until=None):
    return console.run(cmds, wait=wait, quiet_after=quiet_after, until=until)


def run(cmd, timeout, **kwargs):
    """A host tool with a timeout; returns its output (stdout + stderr, \\r removed)."""
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout, **kwargs)
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + f"\n{cmd[0]}: timeout after {timeout} s\n".encode()
    return out.decode(errors="replace").replace("\r", "")


def quietly(fn, *args, **kwargs):
    """Call fn with its prints captured: (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def train(ctrl, pattern):
    _, out = quietly(rdlevel.train_controller, ctrl, step=4)
    grep(out, pattern)


def clean_lanes(ctrl):
    return sum(e == 0 for e in rdlevel.check_controller(ctrl))


def flat(text):
    """A transcript on one line. The firmware prints "GEN:" and the ids with separate printfs:
    they do not reliably land on one line."""
    return " ".join(text.split())


def tokens(text):
    """The ids after the last "GEN:" of a transcript."""
    words = flat(text).replace("GEN:", " GEN: ").split()
    if "GEN:" not in words:
        return ""
    ids = []
    for w in words[len(words) - words[::-1].index("GEN:"):]:
        if not w.isdigit():
            break
        ids.append(w)
    return " ".join(ids)

# Reconfigure --------------------------------------------------------------------------------------

def reconfigure(board, bitstream):
    for module in ["liteuart", "trixium"]:
        subprocess.run(["rmmod", module], stderr=subprocess.DEVNULL)
    dev = find_xilinx_device()
    if dev is not None:
        with open(f"/sys/bus/pci/devices/{dev}/remove", "w", encoding="utf-8") as f:
            f.write("1")
        time.sleep(3)
    if board == "xcu1525":
        out = run(["openFPGALoader", "-c", "ft4232", "--fpga-part", "xcvu9p", bitstream], 600)
        say("\n".join(out.strip().splitlines()[-2:]))
        with open("ir_check.cfg", "w", encoding="utf-8") as f:
            f.write(XCU1525_IR_CHECK_CFG)
        out = run(["openocd", "-f", "ir_check.cfg", "-c", "init", "-c", "irscan xcvu9p.tap 0x3ffff",
            "-c", "shutdown"], 60)
        cap = re.search(r"saw (0x[0-9a-f]*)", out)
        cap = cap.group(1) if cap else "none"
        say(f"IR capture: {cap} ({XCU1525_CONFIGURED} = all three SLRs DONE)")
        if cap != XCU1525_CONFIGURED:
            say("NOT CONFIGURED - stop")
            return 1
    elif board == "ypcb":
        # openFPGALoader checks DONE after the load on this single-device chain.
        out = run(YPCB_PROGRAM + [bitstream], 600)
        grep(out, r"(?i)done|error|fail")
        if re.search(r"(?i)error|fail", out) or not re.search(r"(?i)done", out):
            say("NOT CONFIGURED - stop")
            return 1
    else:
        # openFPGALoader reports "done 1" on the FK33 without configuring it: its JPROGRAM
        # restarts the master-SPI boot and the factory flash image wins. OpenOCD's pld load
        # (~34 s), with the configuration the deploy copies (trixium_fk33.py's).
        out = run(["openocd", "-f", "openocd_fk33.cfg", "-c", f"init; pld load 0 {bitstream}; exit"], 300)
        grep(out, r"loaded|Error")
        if "loaded" not in out:
            say("NOT LOADED - stop")
            return 1
    # Warm reboot, detached: ssh drops with it.
    subprocess.Popen(["sh", "-c", "sleep 2; reboot"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    say("rebooting")
    time.sleep(3)
    return 0

# Boot ---------------------------------------------------------------------------------------------

def kill_consoles():
    for pattern in ["software.host.consol[e]", "trixium_cha[t]"]:
        subprocess.run(["pkill", "-f", pattern], stderr=subprocess.DEVNULL)


def boot_xcu1525(tok, firmware):
    kill_consoles()
    cpu_reset()
    # The console ring keeps the previous session's lines, prompts included: only the text
    # after the NEW banner counts, and the BIOS is done when its memtest verdict is there
    # (the calibration takes ~100 s; training a controller the BIOS is still calibrating
    # leaves both confused).
    acc = new = ""
    for i in range(1, 31):
        acc += "\n" + con([], wait=10, quiet_after=3)
        new = acc[acc.rfind("BIOS built on"):] if "BIOS built on" in acc else ""
        if new and re.search(r"Memtest (OK|KO)", new) and re.search(r"litex.*> ?$", new, re.M):
            say(f"BIOS prompt after ~{i*10} s")
            break
    memtest = re.findall(r".*Memtest (?:OK|KO).*", new)
    if memtest:
        say("BIOS " + memtest[-1])
    if not new:
        # No BIOS: where is the CPU? (the bus monitor, gateware/common.py)
        say("no BIOS banner")
        try:
            say("\n".join(cpumon()[:20]))
        except OSError as e:
            say(str(e))

    # Train until the 4 MB memtest passes (a pass can centre a lane on a marginal spot; the
    # second pass has always fixed it) - never boot the firmware into a main_ram the memtest
    # rejects.
    mt = ""
    for n in range(1, 4):
        train(0, r"FAIL|found|retry")
        say(f"{clean_lanes(0)}/8 lanes clean (controller 0)")
        out = con(["mem_test 0x40000000 0x400000"], wait=60, quiet_after=3)
        mt  = (re.findall(r".*Memtest (?:OK|KO).*", out) or [""])[-1]
        say(f"host-trained pass {n}: {mt}")
        if "OK" in mt:
            break
    if "OK" not in mt:
        say("STOP: main_ram does not pass the memtest after three training passes")
        return 1

    # No model at boot: a staged header from a previous session describes weights at the
    # previous bitstream's addresses and the firmware's self-test can wedge on them. The
    # model step uploads and loads afresh.
    run(["./user/trixium_chat", "-C"], 60)
    out = run(["stdbuf", "-oL", "./user/trixium_chat", "-R", "-W", "2000", "-k", firmware,
        "-z", f"models/{tok}/tokenizer.bin"], 2400)
    with open("/tmp/fw_boot.txt", "w", encoding="utf-8") as f:
        f.write(out)
    grep(out, r"\[boot\]|fastboot|ddr: controller|kept MPR|DDRTEST|SPTEST|VECTEST|READY|rror|ENGSTALL", 60)

    # Controllers 1..3 only from here on: the injector test (check/train) takes a controller
    # out of hardware control for ~100 ms - no refresh, no CPU access - and the firmware
    # runs from controller 0 (a check on it after the boot kills the firmware).
    for k in [1, 2, 3]:
        train(k, r"FAIL|found")
    for k in [1, 2, 3]:
        say(f"ctrl {k}:{clean_lanes(k)}/8 lanes clean")
    grep(con(["engreset", "ddrtest"], wait=200, quiet_after=5), r"ENGRESET|DDRTEST")
    return 0


def boot_ypcb(tok, firmware):
    kill_consoles()
    cpu_reset()
    # The BIOS calibrates controller 0 (sdram_custom_init: the S7HR CK-phase scan, a BIST per
    # step) then runs its memtest; only the text after the NEW banner counts.
    acc = new = ""
    for i in range(1, 31):
        acc += "\n" + con([], wait=10, quiet_after=3)
        new = acc[acc.rfind("BIOS built on"):] if "BIOS built on" in acc else ""
        if new and re.search(r"Memtest (OK|KO)", new) and re.search(r"litex.*> ?$", new, re.M):
            say(f"BIOS prompt after ~{i*10} s")
            break
    if not new:
        say("no BIOS banner")
        try:
            say("\n".join(cpumon()[:20]))
        except OSError as e:
            say(str(e))
        return 1
    grep(new, r"^S7HR|Memtest")
    if not re.search(r"S7HR: OK", new) or not re.search(r"Memtest OK", new):
        say("STOP: controller 0 did not calibrate (S7HR) or main_ram fails the memtest")
        return 1

    # No model at boot (a staged header from a previous session points at stale weights); the
    # firmware boots without a CPU reset, calibrates controller 1 and runs the round trip.
    run(["./user/trixium_chat", "-C"], 60)
    out = run(["stdbuf", "-oL", "./user/trixium_chat", "-R", "-W", "2000", "-k", firmware,
        "-z", f"models/{tok}/tokenizer.bin"], 2400)
    with open("/tmp/fw_boot.txt", "w", encoding="utf-8") as f:
        f.write(out)
    grep(out, r"\[boot\]|fastboot|^S7HR|ddr: controller|DDRTEST|SPTEST|VECTEST|READY|rror|ENGSTALL", 60)
    if not re.search(r"ddr: controller 1 OK", out):
        say("STOP: controller 1 did not calibrate")
        return 1
    out = con(["engreset", "ddrtest"], wait=200, quiet_after=5)
    lines = grep(out, r"ENGRESET|DDRTEST")
    if any("FAIL" in l for l in lines):
        say("STOP: DDRTEST errors")
        return 1
    return 0


def boot_fk33(tok, firmware):
    # trixium_chat -k: CPU reset, loader into main_ram, BIOS boot (~1.5 s); the serial boot
    # NAKs part-way often enough to retry.
    for attempt in [1, 2]:
        out = run(["./user/trixium_chat", "-k", firmware, "-z", f"models/{tok}/tokenizer.bin"], 300)
        grep(out, r"\[boot\]|infer: low heap|SPTEST|VECTEST|TRIXIUM-READY|fastboot")
        if "TRIXIUM-READY" in out:
            return 0
        say(f"  (attempt {attempt} did not reach TRIXIUM-READY, retrying)")
    return 1

# Model / Generation / Bench -----------------------------------------------------------------------

def load(image):
    out = run(["./user/trixium_load", image], 3600)
    grep(out, r"^  -> base|done:|rror|timeout")
    return 0 if "done:" in out else 1


def gen(args, want=None, timeout=1200):
    """hdrload + gen; with `want`, the tokens must contain it (tokens before numbers)."""
    out = con(["hdrload", "gen " + " ".join(args)], wait=timeout, until=PROMPT)
    grep(out, r"hdrload|hybrid|resident|inference|GEN|ENGSTALL: giving|rror")
    say(f"TOKENS: {tokens(out)}")
    if want is None:
        return 0
    if want in flat(out):
        say(f"TOKENS OK: ...{want}...")
        return 0
    say(f"TOKENS WRONG (want ...{want}...) - stop here and find out why before trusting any number")
    say(f"  transcript tail: ...{flat(out)[-200:]}")
    return 1


def bench(n):
    out = con(["stats reset", f"bench {n}", "stats", "sptest"], wait=3500, until=PROMPT)
    grep(out, BENCH_LINES)
    return 0


def abtoggle(toggle, n, rounds, want=None):
    """Interleaved A/B of ONE console on/off toggle (e.g. `silu`), everything else as booted.
    Tokens must match between off and on before any rate is reported."""
    ref, fail = None, False
    for v in ["off", "on"]:
        out = con([f"{toggle} {v}", "gen 8 760 6511 314 9338 369"], wait=800, until=PROMPT)
        grep(out, rf"^{toggle} ")
        tok = tokens(out)
        say(f"tokens {toggle} {v:3s}: {tok}")
        ref = tok if ref is None else ref
        if tok != ref:
            say("  TOKENS DIFFER between off and on")
            fail = True
        if want and want not in tok:
            say("  TOKENS != expected")
            fail = True
    if fail:
        say(f"ABTOGGLE STOP: {toggle} is not bit-exact; no rate is measured")
        return 1
    say(f"tokens identical with {toggle} off and on")
    for r in range(1, rounds + 1):
        for v in ["off", "on"]:
            out = con([f"{toggle} {v}", "stats reset", f"bench {n}", "stats"], wait=1100, until=PROMPT)
            cycles = re.findall(r"[0-9]+ cycles/fwd", out)
            say(f"round {r} {toggle} {v:3s}: {cycles[-1] if cycles else ''}")
            for l in out.splitlines():
                if re.match(r"(BENCH_PHASE|resprof|siluprof|listprof|dnhw|hybprof|dnprof|attnprof|floor)", l):
                    say("    " + l)
    return 0

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="TriXium board-side session (run as root on the board host).")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("reconfigure", help="Program the FPGA and warm-reboot the host.")
    p.add_argument("bitstream",                                                  help="Bitstream.")
    p.add_argument("--board",    default="xcu1525", choices=["xcu1525", "fk33", "ypcb"], help="Board.")

    p = sub.add_parser("boot", help="Boot the firmware (XCU1525: with the DDR4 training).")
    p.add_argument("tok",        default="1.7b", nargs="?",                      help="Tokenizer directory in models/.")
    p.add_argument("--board",    default="xcu1525", choices=["xcu1525", "fk33", "ypcb"], help="Board.")
    p.add_argument("--firmware", default="trixium.bin",                          help="Firmware binary.")

    p = sub.add_parser("load", help="Upload a weight image (trixium_load).")
    p.add_argument("image",                                                       help="BONS image.")

    p = sub.add_parser("gen", help="hdrload + gen <args> on the console.")
    p.add_argument("args",       nargs="+",              help="gen arguments (n, prompt ids).")
    p.add_argument("--want",                             help="Expected token ids (exit 1 otherwise).")
    p.add_argument("--timeout",  default=1200, type=int, help="Generation timeout (s).")

    p = sub.add_parser("model", help="load + gen.")
    p.add_argument("image",                              help="BONS image.")
    p.add_argument("args",       nargs="+",              help="gen arguments (n, prompt ids).")
    p.add_argument("--want",                             help="Expected token ids (exit 1 otherwise).")
    p.add_argument("--timeout",  default=1200, type=int, help="Generation timeout (s).")

    p = sub.add_parser("bench", help="stats reset / bench n / stats / sptest.")
    p.add_argument("n",          default=64, type=int, nargs="?",                 help="Forwards.")

    p = sub.add_parser("abtoggle", help="Interleaved A/B of one console on/off toggle.")
    p.add_argument("toggle",                          help="Console toggle command (e.g. silu).")
    p.add_argument("-n",         default=4, type=int, help="bench n.")
    p.add_argument("--rounds",   default=2, type=int, help="A/B rounds.")
    p.add_argument("--want",                          help="Expected token ids.")
    args = parser.parse_args(argv)

    if args.cmd == "reconfigure":
        rc = reconfigure(args.board, args.bitstream)
    elif args.cmd == "boot":
        rc = {"xcu1525": boot_xcu1525, "fk33": boot_fk33, "ypcb": boot_ypcb}[args.board](args.tok, args.firmware)
    elif args.cmd == "load":
        rc = load(args.image)
    elif args.cmd == "gen":
        rc = gen(args.args, args.want, args.timeout)
    elif args.cmd == "model":
        rc = load(args.image) or gen(args.args, args.want, args.timeout)
    elif args.cmd == "bench":
        rc = bench(args.n)
    else:
        rc = abtoggle(args.toggle, args.n, args.rounds, args.want)
    sys.exit(rc)


if __name__ == "__main__":
    main()

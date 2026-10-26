#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""liblitedram's read-leveling scan, driven from the host over BAR0 - for a byte lane the
BIOS cannot level.

A module can come out of the BIOS's read leveling with no window at any bitslip (all zeros)
while a walking pattern read through the L2 shows its lane exactly one beat late (beat 0 = the
preamble 0xff, beat k = beat k-1's data). The BIOS has no command for the read bitslip/delay
of one module, so this replays the same DFI-injector test pattern as liblitedram
(sdram_write_read_check_test_pattern, the same LFSR and seeds) from the host, with every CSR
write visible, and can pin a module to a chosen (bitslip, delay).

  sudo python3 -m software.host.rdlevel scan 7                  # the scan for module 7, all bitslips
  sudo python3 -m software.host.rdlevel scan 7 --bitslips 4 5 6
  sudo python3 -m software.host.rdlevel set 7 6 350             # module 7: bitslip 6, delay 350
  sudo python3 -m software.host.rdlevel phases                  # rdphase/wrphase
  sudo python3 -m software.host.rdlevel train                   # all modules: write DQ-DQS centring + read leveling
  sudo python3 -m software.host.rdlevel train --bitslips 7      # one module
  sudo python3 -m software.host.rdlevel check                   # error bits per module on the injector pattern

The CSR offsets are controller 0's (csr.h); --ctrl k adds k*0x1000 for the
others (DDR_CSR_STRIDE). The controller is put under software control for the scan and
handed back to the hardware at the end (the BIOS's own sequence).
"""

import os
import re
import sys
import mmap
import ctypes
import argparse

from software.host.cpurst import find_xilinx_device

# CSRs ---------------------------------------------------------------------------------------------

# Controller 0 (pages 8/9); DDR_CSR_STRIDE = 0x1000 per controller.
DFII_CONTROL      = 0x4000
PI_STRIDE         = 0x30
PI_COMMAND        = 0x4004
PI_COMMAND_ISSUE  = 0x4008
PI_ADDRESS        = 0x400c
PI_BADDRESS       = 0x4010
PI_WRDATA         = 0x4014
PI_RDDATA         = 0x4024
DLY_SEL           = 0x4820
RDLY_DQ_RST       = 0x4824
RDLY_DQ_INC       = 0x4828
RDLY_BITSLIP_RST  = 0x482c
RDLY_BITSLIP_INC  = 0x4830
RDPHASE           = 0x4850
WRPHASE           = 0x4854
EN_VTC            = 0x4804
WDLY_DQ_RST       = 0x4834
WDLY_DQ_INC       = 0x4838
WDLY_DQS_RST      = 0x483c
WDLY_DQS_INC      = 0x4840
WDLY_BITSLIP_RST  = 0x4848
WDLY_BITSLIP_INC  = 0x484c

DFII_CONTROL_SEL, DFII_CONTROL_CKE, DFII_CONTROL_ODT, DFII_CONTROL_RESET_N = 1, 2, 4, 8
CMD_CS, CMD_WE, CMD_CAS, CMD_RAS, CMD_WRDATA, CMD_RDDATA = 1, 2, 4, 8, 0x10, 0x20
PHASES, DATA_BYTES, MODULES, DELAYS, BITSLIPS = 4, 16, 8, 512, 8
SEEDS = (42, 84, 36)

# Test Pattern -------------------------------------------------------------------------------------

def lfsr32(prev):
    lsb = prev & 1
    prev >>= 1
    if lsb:
        prev ^= 0x80200003
    return prev & 0xffffffff


def pattern(seed):
    prv = seed
    prs = []
    for p in range(PHASES):
        row = bytearray(DATA_BYTES)
        for i in range(DATA_BYTES):
            v = 0
            for bit in range(8):
                prv = lfsr32(prv)
                v |= (prv & 1) << bit
            row[i] = v
        prs.append(bytes(row))
    return prs

# BAR0 ---------------------------------------------------------------------------------------------

class Bar0:
    def __init__(self, base):
        dev = find_xilinx_device()
        if dev is None:
            raise OSError("no Xilinx device found")
        self.fd = os.open(f"/sys/bus/pci/devices/{dev}/resource0", os.O_RDWR | os.O_SYNC)
        size = os.fstat(self.fd).st_size or 0x20000     # the whole BAR (0x40000 on the XCU1525)
        self.m    = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.base = base

    # One 32-bit access per CSR: mmap.write()'s memcpy can split a 4-byte store into two 16-bit
    # stores, i.e. two CSR writes, and every pulse CSR (delay inc, bitslip) would fire twice.
    def rd(self, off):
        return ctypes.c_uint32.from_buffer(self.m, self.base + off).value

    def wr(self, off, v):
        ctypes.c_uint32.from_buffer(self.m, self.base + off).value = v & 0xffffffff

    # A 128-bit CSR as 4 words, big ordering: word j holds bits 32*(3-j)..
    def rd128(self, off):
        v = 0
        for j in range(4):
            v |= self.rd(off + 4*j) << (32*(3 - j))
        return v.to_bytes(16, "little")

    def wr128(self, off, data16):
        v = int.from_bytes(data16, "little")
        for j in range(4):
            self.wr(off + 4*j, (v >> (32*(3 - j))) & 0xffffffff)

class RemoteBus:
    """The same accessors over a litex_server (Etherbone on the standalone build's QSFP Ethernet,
    or JTAGBone): `base` is the controller's absolute CSR address (CSR_BASE + its page)."""
    def __init__(self, base, csr_csv=None):
        from litex.tools.litex_client import RemoteClient
        self.bus = RemoteClient(csr_csv=csr_csv) if csr_csv else RemoteClient()
        self.bus.open()
        self.base = base

    def rd(self, off):
        return self.bus.read(self.base + off)

    def wr(self, off, v):
        self.bus.write(self.base + off, v & 0xffffffff)

    rd128 = Bar0.rd128
    wr128 = Bar0.wr128

# Controller ---------------------------------------------------------------------------------------

class Ctrl:
    def __init__(self, bar):
        self.b = bar

    def pi(self, reg, phase):
        return reg + PI_STRIDE*phase

    def command(self, phase, value):
        self.b.wr(self.pi(PI_COMMAND, phase), value)
        self.b.wr(self.pi(PI_COMMAND_ISSUE, phase), 1)

    # liblitedram's pair: the delay lines are VARIABLE/ASYNC with EN_VTC, and UG571 only
    # honours CE/INC/RST with voltage-temperature compensation off - a scan with EN_VTC
    # high is flat (every delay ignored).
    def software_control_on(self):
        self.b.wr(DFII_CONTROL, DFII_CONTROL_CKE | DFII_CONTROL_ODT | DFII_CONTROL_RESET_N)
        self.b.wr(EN_VTC, 0)

    def software_control_off(self):
        self.b.wr(DFII_CONTROL, DFII_CONTROL_SEL)
        self.b.wr(EN_VTC, 1)

    def phases(self):
        return self.b.rd(RDPHASE), self.b.rd(WRPHASE)

    def sel(self, module, reg):
        self.b.wr(DLY_SEL, 1 << module)
        self.b.wr(reg, 1)
        self.b.wr(DLY_SEL, 0)

    def activate_test_row(self):
        self.b.wr(self.pi(PI_ADDRESS, 0), 0)
        self.b.wr(self.pi(PI_BADDRESS, 0), 0)
        self.command(0, CMD_RAS | CMD_CS)

    def precharge_test_row(self):
        self.b.wr(self.pi(PI_ADDRESS, 0), 0)
        self.b.wr(self.pi(PI_BADDRESS, 0), 0)
        self.command(0, CMD_RAS | CMD_WE | CMD_CS)

    def check_pattern(self, module, seed, invert=False):
        """liblitedram's sdram_write_read_check_test_pattern: the module's bytes (positive
        edge byte = module, negative edge byte = module + 8) of every phase; returns the
        error bit count. invert: the pattern's complement (see module_ok)."""
        rdphase, wrphase = self.phases()
        prs = pattern(seed)
        if invert:
            prs = [bytes(b ^ 0xFF for b in row) for row in prs]
        self.activate_test_row()
        for p in range(PHASES):
            self.b.wr128(self.pi(PI_WRDATA, p), prs[p])
        self.b.wr(self.pi(PI_ADDRESS, wrphase), 0)
        self.b.wr(self.pi(PI_BADDRESS, wrphase), 0)
        self.command(wrphase, CMD_CAS | CMD_WE | CMD_CS | CMD_WRDATA)
        self.b.wr(self.pi(PI_ADDRESS, rdphase), 0)
        self.b.wr(self.pi(PI_BADDRESS, rdphase), 0)
        self.command(rdphase, CMD_CAS | CMD_CS | CMD_RDDATA)
        self.precharge_test_row()
        errors = 0
        got    = []
        for p in range(PHASES):
            tst = self.b.rd128(self.pi(PI_RDDATA, p))
            got.append(tst)
            for bo in (module, module + 8):
                errors += bin(prs[p][bo] ^ tst[bo]).count("1")
        return errors, prs, got

    def module_ok(self, module):
        """Every seed, then its complement: the same patterns at the same address read back
        right after ANY setting once wrote them, so a write side that writes nothing would
        pass (a lane "clean" here while every CPU write to it is lost). The complement must
        overwrite what the seed left."""
        return all(self.check_pattern(module, s, inv)[0] == 0 for s in SEEDS for inv in (False, True))

# Scan / Leveling ----------------------------------------------------------------------------------

def scan(ctrl, module, bitslips, step, show, errors_mode=False):
    ctrl.sel(module, RDLY_BITSLIP_RST)
    for bs in range(BITSLIPS):
        if bs in bitslips:
            ctrl.sel(module, RDLY_DQ_RST)
            line = []
            for d in range(0, DELAYS, step):
                if errors_mode:
                    e = sum(ctrl.check_pattern(module, s)[0] for s in SEEDS)
                    line.append(str(min(e, 9)) if e else ".")
                else:
                    line.append("1" if ctrl.module_ok(module) else "0")
                for _ in range(step):
                    ctrl.sel(module, RDLY_DQ_INC)
            print(f"  m{module}, b{bs:02d}: |{''.join(line)}|", flush=True)
            if show:
                ctrl.sel(module, RDLY_DQ_RST)
                e, prs, got = ctrl.check_pattern(module, SEEDS[0])
                for p in range(PHASES):
                    print(f"     phase {p}: want {prs[p].hex()} got {got[p].hex()}")
        ctrl.sel(module, RDLY_BITSLIP_INC)


def _windows(line):
    """Runs of passing samples in a scan line (index, length), wrap-around ignored."""
    runs, start = [], None
    for i, ok in enumerate(line + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def read_scan(ctrl, module, step):
    """Per bitslip, the passing delays (3 seeds); the bitslip counter is left at 0."""
    res = []
    ctrl.sel(module, RDLY_BITSLIP_RST)
    for bs in range(BITSLIPS):
        ctrl.sel(module, RDLY_DQ_RST)
        line = []
        for d in range(0, DELAYS, step):
            line.append(ctrl.module_ok(module))
            for _ in range(step):
                ctrl.sel(module, RDLY_DQ_INC)
        res.append(line)
        ctrl.sel(module, RDLY_BITSLIP_INC)   # 8 increments = back to the start
    return res


def read_center(ctrl, module, step=8, quiet=False):
    """Pick the bitslip with the widest window and centre the delay in it. Returns the
    window width in taps (0 = none)."""
    res = read_scan(ctrl, module, step)
    best = (0, 0, 0)   # width, bitslip, start
    for bs, line in enumerate(res):
        for start, length in _windows(line):
            if length > best[0]:
                best = (length, bs, start)
    width, bs, start = best
    if width == 0:
        return 0
    center = (start*step + (width*step)//2)
    ctrl.sel(module, RDLY_BITSLIP_RST)
    for _ in range(bs):
        ctrl.sel(module, RDLY_BITSLIP_INC)
    ctrl.sel(module, RDLY_DQ_RST)
    for _ in range(center):
        ctrl.sel(module, RDLY_DQ_INC)
    if not quiet:
        print(f"  m{module}: read bitslip {bs}, delay {center} (window {width*step} taps)")
    return width*step


def write_center(ctrl, module, step=8):
    """Scan the write DQ delay over its 512 taps (the DQS stays), pick the widest window
    and move DQ to its centre. The delay wraps at 512, so the net move is (center) taps."""
    line = []
    for k in range(0, DELAYS, step):
        line.append(ctrl.module_ok(module))
        for _ in range(step):
            ctrl.sel(module, WDLY_DQ_INC)
    # Wrap-around: the passing run at the start may continue the one at the end.
    runs = _windows(line)
    if not runs:
        return 0
    if line[0] and line[-1] and len(runs) > 1:
        (_, l0), (s1, l1) = runs[0], runs[-1]
        runs = [(s1, l1 + l0)] + runs[1:-1]
    start, length = max(runs, key=lambda r: r[1])
    center = (start*step + (length*step)//2) % DELAYS
    for _ in range(center):
        ctrl.sel(module, WDLY_DQ_INC)
    print(f"  m{module}: write DQ +{center} taps (window {length*step} taps of 512)")
    return length*step


def find_reads(ctrl, m):
    """A usable read setting for module m, stepping the write DQ delay when the leveled
    point is outside the write eye. True when found."""
    if read_center(ctrl, m, 8, quiet=True):
        return True
    for k in range(0, DELAYS, 16):
        for _ in range(16):
            ctrl.sel(m, WDLY_DQ_INC)
        if read_center(ctrl, m, 16, quiet=True):
            return True
    return False


def find_writes(ctrl, m, dqs_step=40, dq_step=16):
    """With the read side trusted (liblitedram's MPR read leveling, or a previous pass),
    search the write side: the DQS position against CK (DQ moves with it, as write
    leveling moves them), the write bitslip (the write latency), the DQ against DQS.
    liblitedram's write leveling fails on modules whose DQS-to-CK transition sits just
    before delay 0 ("delay: -", controllers 2 and 3 here) and its latency calibration
    then guesses; a pattern that reads back right is the only thing that matters, and
    tDQSS gives the DQS position +-0.27 tCK of slack. Returns True when found."""
    for _ in range(0, DELAYS, dqs_step):
        ctrl.sel(m, WDLY_BITSLIP_RST)
        for wbs in range(BITSLIPS):
            for _k in range(0, DELAYS, dq_step):
                if ctrl.module_ok(m):
                    return True
                for _ in range(dq_step):
                    ctrl.sel(m, WDLY_DQ_INC)
            ctrl.sel(m, WDLY_BITSLIP_INC)
        for _ in range(dqs_step):
            ctrl.sel(m, WDLY_DQS_INC)
            ctrl.sel(m, WDLY_DQ_INC)
    return False


def write_map_best(ctrl, m, step=16):
    """Map the write side - every write bitslip x DQS position (DQ moving with DQS, from
    reset) - and settle on the centre of the widest passing run. find_writes stops at the
    first passing point and dqs_center then centres locally: at DDR4-1500 a lane can pass
    only at one bitslip over a narrow DQS range, and the first-pass search leaves it on that
    run's edge, where every CPU write to the lane is lost. True when something passes."""
    best = None
    ctrl.sel(m, WDLY_BITSLIP_RST)
    for wbs in range(BITSLIPS):
        ctrl.sel(m, WDLY_DQS_RST)
        ctrl.sel(m, WDLY_DQ_RST)
        line = []
        for k in range(0, DELAYS, step):
            line.append(ctrl.module_ok(m))
            for _ in range(step):
                ctrl.sel(m, WDLY_DQS_INC)
                ctrl.sel(m, WDLY_DQ_INC)
        for start, length in _windows(line):
            if best is None or length > best[2]:
                best = (wbs, start, length)
        ctrl.sel(m, WDLY_BITSLIP_INC)
    if best is None:
        return False
    wbs, start, length = best
    ctrl.sel(m, WDLY_BITSLIP_RST)
    for _ in range(wbs):
        ctrl.sel(m, WDLY_BITSLIP_INC)
    ctrl.sel(m, WDLY_DQS_RST)
    ctrl.sel(m, WDLY_DQ_RST)
    center = start*step + (length*step)//2
    for _ in range(center):
        ctrl.sel(m, WDLY_DQS_INC)
        ctrl.sel(m, WDLY_DQ_INC)
    print(f"  m{m}: write bitslip {wbs}, DQS {center} (run {start*step}..{(start + length)*step - step}, "
          f"{length*step} taps)")
    return True


def dqs_center(ctrl, m, step=8):
    """Centre the DQS position against CK inside the range where the pattern round-trips
    (DQ moves with DQS). Write leveling puts DQS on the CK edge - the middle of the tDQSS
    window - for the modules it manages; find_writes stops at the first passing position,
    which can be the edge of that window, and a DQS at the edge of tDQSS fails now and
    then (intermittent bit errors)."""
    if not ctrl.module_ok(m):
        return 0
    def move(n):   # n taps forward on DQS and DQ (mod 512)
        for _ in range(n % DELAYS):
            ctrl.sel(m, WDLY_DQS_INC)
            ctrl.sel(m, WDLY_DQ_INC)
    fwd = 0
    while fwd + step < DELAYS // 2:
        move(step)
        fwd += step
        if not ctrl.module_ok(m):
            break
    move(DELAYS - fwd)          # back to the origin
    back = 0
    while back + step < DELAYS // 2:
        move(DELAYS - step)
        back += step
        if not ctrl.module_ok(m):
            break
    move(back)                  # back to the origin
    # The passing range is (-back, +fwd) exclusive of the failing ends; its centre:
    center = (fwd - back) // 2
    move(center % DELAYS)
    print(f"  m{m}: DQS window -{back}..+{fwd} taps around the origin, moved {center:+d}")
    return fwd + back

# Training -----------------------------------------------------------------------------------------

def train_module(ctrl, m, step):
    if write_map_best(ctrl, m):
        write_center(ctrl, m, step)
        read_center(ctrl, m, step)
        return all(ctrl.module_ok(m) for _ in range(4))
    if not ctrl.module_ok(m):
        if find_writes(ctrl, m):
            print(f"  m{m}: write side found (DQS position / write bitslip / DQ)")
        elif find_reads(ctrl, m):
            print(f"  m{m}: read side found")
        else:
            found = False
            ctrl.sel(m, WDLY_BITSLIP_RST)
            for wbs in range(BITSLIPS):
                if find_reads(ctrl, m):
                    print(f"  m{m}: write bitslip {wbs} with a new read window")
                    found = True
                    break
                ctrl.sel(m, WDLY_BITSLIP_INC)
            if not found:
                print(f"  m{m}: FAILED - no combination reads back the pattern")
                return False
    dqs_center(ctrl, m)
    write_center(ctrl, m, step)
    read_center(ctrl, m, step)
    return all(ctrl.module_ok(m) for _ in range(4))


def train(ctrl, modules, step):
    """Per module: if the pattern does not round-trip at the current settings, search the
    write side with the reads as they are (find_writes), then fall back to the read
    side search (find_reads: read bitslip/delay, write DQ, write bitslip); then centre
    the DQS, the write DQ and the reads, and verify - a pass whose centring landed on a
    marginal spot is retried."""
    for m in modules:
        for attempt in range(3):
            if train_module(ctrl, m, step):
                break
            print(f"  m{m}: verification failed after pass {attempt + 1}, retrying")
        else:
            print(f"  m{m}: FAILED - three passes did not converge")

# CSR Location -------------------------------------------------------------------------------------

def csr_shift(path):
    """The controllers' CSR pages relative to the offsets above (controller 0 at 0x4000 of
    the CSR region), or their own bus at DDR_CSR_WINDOW (the XCU1525 target: the controllers
    in their ddr domain). Read from the build's generated csr.h (the deploy copies it to
    gen/software/include/generated/)."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return 0
    m = re.search(r"#define CSR_SDRAM_BASE \(CSR_BASE \+ (0x[0-9a-fA-F]+)L?\)", text)
    if not m:
        return 0
    return int(m.group(1), 16) - DFII_CONTROL

# Controller Access --------------------------------------------------------------------------------

CSR_H = "gen/software/include/generated/csr.h"

def csr_base(csr_h):
    """CSR_BASE from the generated mem.h beside csr.h (the absolute address a litex_server needs)."""
    try:
        with open(os.path.join(os.path.dirname(csr_h), "mem.h"), encoding="utf-8") as f:
            m = re.search(r"#define CSR_BASE (0x[0-9a-fA-F]+)L?", f.read())
            return int(m.group(1), 16) if m else 0xf0000000
    except OSError:
        return 0xf0000000


def open_controller(index=0, csr_h=CSR_H, bus="pci", csr_csv=None):
    """The DFI controls of controller `index` (CSR stride 0x1000): over BAR0 (`bus` "pci", the
    card in a host) or a litex_server on localhost:1234 ("remote": Etherbone over the standalone
    build's Ethernet, or JTAGBone), the fallback trainer when the card's own training fails."""
    if bus == "pci":
        bar = Bar0(index*0x1000 + csr_shift(csr_h))
    else:
        bar = RemoteBus(csr_base(csr_h) + index*0x1000 + csr_shift(csr_h), csr_csv)
    return bar, Ctrl(bar)


def train_controller(index=0, step=8, modules=None, csr_h=CSR_H, bus="pci"):
    """Write DQ-DQS centring + read leveling of `modules` (default: all) of one controller."""
    _, ctrl = open_controller(index, csr_h, bus)
    ctrl.software_control_on()
    try:
        train(ctrl, modules or list(range(MODULES)), step)
    finally:
        ctrl.software_control_off()


def check_controller(index=0, csr_h=CSR_H):
    """Error bits per module on the injector pattern (every seed and its complement)."""
    _, ctrl = open_controller(index, csr_h)
    ctrl.software_control_on()
    try:
        return [sum(ctrl.check_pattern(m, s, inv)[0] for s in SEEDS for inv in (False, True))
            for m in range(MODULES)]
    finally:
        ctrl.software_control_off()

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="liblitedram's read-leveling scan, driven from the host over BAR0 (or a litex_server).")
    parser.add_argument("cmd",        choices=["scan", "set", "phases", "raw", "train", "check"],  help="Command.")
    parser.add_argument("module",     type=int, nargs="?", default=7,                              help="Module (byte lane).")
    parser.add_argument("bitslip",    type=int, nargs="?",                                         help="Read bitslip (set).")
    parser.add_argument("delay",      type=int, nargs="?",                                         help="Read delay (set).")
    parser.add_argument("--ctrl",     type=int, default=0,                                         help="Controller index (CSR stride 0x1000).")
    parser.add_argument("--csr-h",    default=CSR_H,                                               help="Build's csr.h (locates the controllers' CSR pages).")
    parser.add_argument("--bus",      default="pci", choices=["pci", "remote"],                     help="pci: BAR0 (the card in a host); remote: a litex_server on localhost:1234 (Etherbone/JTAGBone, the standalone card).")
    parser.add_argument("--bitslips", type=int, nargs="*", default=None,                           help="Bitslips to scan (train: modules to train).")
    parser.add_argument("--step",     type=int, default=8,                                         help="Delay step of the scan.")
    parser.add_argument("--show",     action="store_true",                                         help="Print want/got at delay 0 per bitslip.")
    parser.add_argument("--errors",   action="store_true",                                         help="Print the error-bit count (3 seeds, capped at 9; '.' = none) instead of 1/0.")
    a = parser.parse_args(argv)
    try:
        bar, ctrl = open_controller(a.ctrl, a.csr_h, a.bus)
    except OSError as e:
        sys.exit(str(e))
    if a.cmd == "phases":
        print("rdphase %d wrphase %d control 0x%x" % (*ctrl.phases(), bar.rd(DFII_CONTROL)))
        return
    ctrl.software_control_on()
    try:
        if a.cmd == "scan":
            bl = set(a.bitslips) if a.bitslips is not None else set(range(BITSLIPS))
            scan(ctrl, a.module, bl, a.step, a.show, a.errors)
        elif a.cmd == "raw":
            e, prs, got = ctrl.check_pattern(a.module, SEEDS[0])
            print(f"module {a.module}: {e} error bits")
            for p in range(PHASES):
                print(f"  phase {p}: want {prs[p].hex()} got {got[p].hex()}")
        elif a.cmd == "train":
            mods = a.bitslips if a.bitslips else list(range(MODULES))   # --bitslips doubles as the module list
            train(ctrl, mods, a.step)
        elif a.cmd == "check":
            for m in range(MODULES):
                e = sum(ctrl.check_pattern(m, s, inv)[0] for s in SEEDS for inv in (False, True))
                print(f"  m{m}: {e} error bits")
        elif a.cmd == "set":
            if a.bitslip is None or a.delay is None:
                sys.exit("set <module> <bitslip> <delay>")
            ctrl.sel(a.module, RDLY_BITSLIP_RST)
            for _ in range(a.bitslip):
                ctrl.sel(a.module, RDLY_BITSLIP_INC)
            ctrl.sel(a.module, RDLY_DQ_RST)
            for _ in range(a.delay):
                ctrl.sel(a.module, RDLY_DQ_INC)
            print(f"module {a.module}: bitslip {a.bitslip}, delay {a.delay}: "
                  f"{'OK' if ctrl.module_ok(a.module) else 'errors'} on the injector pattern")
    finally:
        ctrl.software_control_off()


if __name__ == "__main__":
    main()

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""trixium smi: the state of the TriXium cards in this machine, like nvidia-smi.

    trixium smi                 one table per card (/dev/trixium*)
    trixium smi -l 1            refresh every second
    trixium smi --json          the same, as JSON (one object per card)
    trixium smi --fake          a synthetic card (demo, tests)

A monitor: it only reads registers through the driver (/dev/trixiumN, no root needed once
init.sh opened the nodes up), plus `perf_chan` (the per-channel counter mux, which the
firmware only uses in its stall diagnostics). It never takes the console lock.

- Card:         the bitstream's identifier, the sys clock measured from the cycle counter,
                the PCIe link (sysfs) against what the slot can do.
- CPU / units:  CPU in reset or running, engine / sequencer / vector unit state and errors.
- Utilization:  engine-busy, vector-busy and overlap cycles, ops/s, over an interval.
- Memory:       beats delivered per channel -> GB/s, against the peak of the channels
                (32 bytes per channel per cycle), and the read-stall share.
- Model:        what the host tools recorded in /tmp/trixium/: the image last uploaded and
                the chat process (pid, tokens, tok/s).

The register map comes from the build's csr.csv: --csr, else the first csr.csv whose
identifier matches the card's (./, ~/trixium/, build/*/).
"""

import os
import sys
import csv
import glob
import json
import time
import fcntl
import struct
import random
import argparse
import datetime

# Constants ----------------------------------------------------------------------------------------

STATE_DIR = "/tmp/trixium"          # Written by trixium_load / trixium_chat / trixium.client.
BEAT_BYTES = 32                     # 256-bit AXI memory ports.
WIDTH      = 88                     # Table width.

# _IOWR('S', 0, struct litepcie_ioctl_reg { u32 addr; u32 val; u8 is_write; }) -> 12 bytes.
LITEPCIE_IOCTL_REG = (3 << 30) | (12 << 16) | (ord("S") << 8) | 0

ENGINE_STATUS = {"busy": 0, "error": 1, "pending": 2, "fill": 7, "run": 8}
SEQ_STATUS    = {"busy": 0}
VEC_STATUS    = {"busy": 0, "bad_op": 1}

# Register Map -------------------------------------------------------------------------------------

class CSRMap:
    """Registers and constants of one build, from its csr.csv."""
    def __init__(self, filename):
        self.filename  = filename
        self.base      = 0
        self.regs      = {}
        self.constants = {}
        with open(filename, encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                kind, name = row[0], row[1]
                if kind == "csr_base" and name == "ctrl":
                    self.base = int(row[2], 0)
                elif kind == "csr_register":
                    self.regs[name] = (int(row[2], 0), int(row[3]))
                elif kind == "constant":
                    self.constants[name] = row[2]
        if "identifier_mem" not in self.regs:
            self.regs["identifier_mem"] = (self.base + 0x800, 1)

    @property
    def identifier(self):
        return self.constants.get("config_identifier", "")

    def constant(self, name, default=None):
        value = self.constants.get(name)
        return default if value in (None, "None") else int(value)

    def offset(self, name):
        return self.regs[name][0] - self.base

def find_csr_maps(paths=None):
    candidates = list(paths or [])
    candidates += ["csr.csv", os.path.expanduser("~/trixium/csr.csv")]
    candidates += sorted(glob.glob("build/*/csr.csv"), key=os.path.getmtime, reverse=True)
    maps = []
    for filename in candidates:
        if os.path.isfile(filename):
            try:
                maps.append(CSRMap(filename))
            except (OSError, ValueError, IndexError):
                pass
    return maps

# Cards --------------------------------------------------------------------------------------------

class DriverCard:
    """A card behind the TriXium driver: BAR0 register reads/writes through the REG ioctl."""
    def __init__(self, device):
        self.device = device
        self.fd     = os.open(device, os.O_RDWR)

    def read(self, offset):
        buf = bytearray(struct.pack("<IIB3x", offset, 0, 0))
        fcntl.ioctl(self.fd, LITEPCIE_IOCTL_REG, buf)
        return struct.unpack("<IIB3x", buf)[1]

    def write(self, offset, value):
        buf = bytearray(struct.pack("<IIB3x", offset, value, 1))
        fcntl.ioctl(self.fd, LITEPCIE_IOCTL_REG, buf)

    def pci_device(self):
        """The card's sysfs PCI device directory (via the driver's class device)."""
        name = os.path.basename(self.device)
        path = os.path.realpath(f"/sys/class/trixium/{name}/device")
        return path if os.path.isdir(path) else None

    def close(self):
        os.close(self.fd)

class FakeCard:
    """A synthetic XCU1525-like card generating a token: counters advance with wall time."""
    def __init__(self, csr_map, device="/dev/fake0"):
        self.device = device
        self.map    = csr_map
        self.t0     = time.monotonic()
        self.chan   = 0
        self.clk    = csr_map.constant("config_clock_frequency", 125_000_000)
        self.ident  = csr_map.identifier.encode() + b"\0"
        self.rng    = random.Random(0)

    def _cycles(self):
        return int((time.monotonic() - self.t0) * self.clk)

    def read(self, offset):
        ident_base = self.map.offset("identifier_mem")
        if ident_base <= offset < ident_base + 4*len(self.ident):
            return self.ident[(offset - ident_base)//4]
        names = {self.map.offset(n): n for n in self.map.regs}
        name  = names.get(offset, "")
        c     = self._cycles()
        value = {
            "trixium_engine_csr_cyc_total"  : c,
            "trixium_engine_csr_cyc_eng"    : int(c*0.71),
            "trixium_engine_csr_cyc_vec"    : int(c*0.12),
            "trixium_engine_csr_cyc_both"   : int(c*0.05),
            "trixium_engine_csr_done_count" : c//9000,
            "trixium_engine_csr_status"     : 0x101,
            "trixium_seq_status"            : 0x1 | ((c//20000) & 0xffff) << 8,
            "trixium_engine_csr_perf_beats" : int(c*0.42*(1 - 0.02*self.chan)),
            "trixium_engine_csr_perf_r"     : int(c*0.20),
        }.get(name, 0)
        return value & 0xffffffff

    def write(self, offset, value):
        if offset == self.map.offset("trixium_engine_csr_perf_chan"):
            self.chan = value

    def pci_device(self):
        return None

    def close(self):
        pass

def find_devices():
    return sorted(glob.glob("/dev/trixium[0-9]*"), key=lambda d: int(d[len("/dev/trixium"):]))

# Sampling -----------------------------------------------------------------------------------------

def read_identifier(card, csr_map, length=64):
    base  = csr_map.offset("identifier_mem")
    chars = []
    for i in range(length):
        c = card.read(base + 4*i) & 0xff
        if c == 0:
            break
        chars.append(chr(c))
    return "".join(chars)

def pcie_link(card):
    path = card.pci_device()
    if path is None:
        return None
    def attr(name):
        try:
            with open(os.path.join(path, name), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None
    return {
        "address"   : os.path.basename(path),
        "speed"     : attr("current_link_speed"),
        "width"     : attr("current_link_width"),
        "max_speed" : attr("max_link_speed"),
        "max_width" : attr("max_link_width"),
    }

def snapshot(card, csr_map, channels):
    """One sample of every counter (wall time first, so rates use the matching window)."""
    rd = lambda name: card.read(csr_map.offset(name))
    s  = {"time": time.monotonic()}
    for name in ("cyc_total", "cyc_eng", "cyc_vec", "cyc_both", "done_count"):
        s[name] = rd(f"trixium_engine_csr_{name}")
    s["seq_status"] = rd("trixium_seq_status")
    s["beats"], s["r_stalls"] = [], []
    for c in range(channels):
        card.write(csr_map.offset("trixium_engine_csr_perf_chan"), c)
        s["beats"].append(rd("trixium_engine_csr_perf_beats"))
        s["r_stalls"].append(rd("trixium_engine_csr_perf_r"))
    return s

def _delta(a, b):
    return (b - a) & 0xffffffff  # 32-bit counters, wrap-safe over a short interval.

def _bits(value, fields):
    return {name: bool((value >> bit) & 1) for name, bit in fields.items()}

def write_state(devnum, kind, **fields):
    """Record the card's state (the Python twin of trixium_loader.c's loader_state_write)."""
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        os.chmod(STATE_DIR, 0o777)
    except OSError:
        pass
    path = os.path.join(STATE_DIR, f"trixium{devnum}.{kind}.json")
    tmp  = f"{path}.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), **fields}, f)
        os.chmod(tmp, 0o666)
        os.replace(tmp, path)
    except OSError:
        pass  # Best effort: the monitor is optional.

def read_state(device):
    """The host tools' records for this card: last upload and chat process (or None)."""
    name  = os.path.basename(device)
    state = {}
    for kind in ("load", "chat"):
        try:
            with open(os.path.join(STATE_DIR, f"{name}.{kind}.json"), encoding="utf-8") as f:
                state[kind] = json.load(f)
        except (OSError, ValueError):
            state[kind] = None
    chat = state["chat"]
    if chat and chat.get("pid"):
        chat["alive"] = os.path.exists(f"/proc/{chat['pid']}")
    return state

def query(card, csr_map, interval=0.25):
    channels = csr_map.constant("trixium_n_channels", 1)
    rd       = lambda name: card.read(csr_map.offset(name))
    d = {
        "device"     : card.device,
        "time"       : datetime.datetime.now().isoformat(timespec="seconds"),
        "identifier" : read_identifier(card, csr_map),
        "csr_csv"    : csr_map.filename,
        "pcie"       : pcie_link(card),
    }
    if not d["identifier"].startswith("TriXium"):
        d["ok"] = False
        return d
    d["ok"] = True

    # Status registers.
    cpu_reset = rd("cpumon_cpu_reset") & 1 if "cpumon_cpu_reset" in csr_map.regs else None
    d["cpu"]  = {"running": None if cpu_reset is None else not cpu_reset,
                 "bus_errors": rd("ctrl_bus_errors")}
    d["engine"]    = _bits(rd("trixium_engine_csr_status"), ENGINE_STATUS)
    d["sequencer"] = _bits(rd("trixium_seq_status"), SEQ_STATUS)
    d["vector"]    = _bits(rd("trixium_vecunit_status"), VEC_STATUS)

    # Counters over the interval.
    s0 = snapshot(card, csr_map, channels)
    time.sleep(interval)
    s1 = snapshot(card, csr_map, channels)
    seconds = s1["time"] - s0["time"]
    cycles  = _delta(s0["cyc_total"], s1["cyc_total"])
    clk     = cycles/seconds if seconds > 0 else 0
    frac    = lambda name: _delta(s0[name], s1[name])/cycles if cycles else 0.0
    beats   = [_delta(a, b) for a, b in zip(s0["beats"],    s1["beats"])]
    stalls  = [_delta(a, b) for a, b in zip(s0["r_stalls"], s1["r_stalls"])]
    d["clock_mhz"] = clk/1e6
    d["sample"]    = {"seconds": seconds, "cycles": cycles}
    d["util"] = {
        "engine"  : frac("cyc_eng"),
        "vector"  : frac("cyc_vec"),
        "overlap" : frac("cyc_both"),
        "ops_s"   : _delta(s0["done_count"], s1["done_count"])/seconds if seconds else 0.0,
        "lists_s" : ((s1["seq_status"] >> 8) - (s0["seq_status"] >> 8)) % (1 << 16)/seconds
                    if seconds else 0.0,
    }
    gbs = [b*BEAT_BYTES/seconds/1e9 if seconds else 0.0 for b in beats]
    d["memory"] = {
        "kind"          : "HBM2" if "hbm_channels" in csr_map.constants else "DDR4",
        "channels"      : channels,
        "gbs"           : sum(gbs),
        "peak_gbs"      : channels*BEAT_BYTES*clk/1e9,
        "channel_gbs"   : gbs,
        "channel_util"  : [b/cycles if cycles else 0.0 for b in beats],
        "r_stall"       : sum(stalls)/(channels*cycles) if cycles else 0.0,
    }
    d.update(read_state(card.device))
    return d

# Output -------------------------------------------------------------------------------------------

def _bar(x, n=10):
    k = max(0, min(n, round(x*n)))
    return "█"*k + "░"*(n - k)

def _pct(x):
    return f"{100*x:3.0f}%"

def _line(text=""):
    return "│ " + text[:WIDTH - 4].ljust(WIDTH - 4) + " │"

def _head(title, first=False):
    left, right = ("╭", "╮") if first else ("├", "┤")
    title = f"─ {title} " if title else ""
    return left + title + "─"*(WIDTH - 2 - len(title)) + right

def _kv(name, *cells):
    return _line(f"{name:<12}" + "".join(f"{c:<24}  " for c in cells).rstrip())

def _link(p):
    if not p or not p.get("speed"):
        return "PCIe n/a"
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4}
    g   = lambda s: f"Gen{gen.get(s.split()[0], '?')}"
    out = f"PCIe {g(p['speed'])} x{p['width']}"
    if (p["speed"], p["width"]) != (p["max_speed"], p["max_width"]):
        out += f" (slot {g(p['max_speed'])} x{p['max_width']})"
    return out

def _age(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s ago"
    if seconds < 7200:
        return f"{seconds/60:.0f} min ago"
    return f"{seconds/3600:.1f} h ago"

def table(devices):
    now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    out = ["trixium-smi".ljust(WIDTH - len(now)) + now]
    for n, d in enumerate(devices):
        title = f"Card {n} · {d['device']}"
        if d.get("pcie"):
            title += f" · {d['pcie']['address']}"
        out.append(_head(title, first=True))
        if not d.get("ok"):
            out.append(_kv("Status", d.get("error") or f"no TriXium bitstream ({d.get('identifier')!r})"))
            out.append("╰" + "─"*(WIDTH - 2) + "╯")
            continue
        out.append(_kv("Bitstream", d["identifier"]))
        out.append(_kv("Link", _link(d.get("pcie")), f"sys {d['clock_mhz']:.1f} MHz (measured)"))
        cpu, eng, vec = d["cpu"], d["engine"], d["vector"]
        cpu_txt = {True: "running", False: "in reset", None: "n/a"}[cpu["running"]]
        errors  = [n for n, bad in (("engine error", eng["error"]), ("vector bad op", vec["bad_op"]),
                                    (f"{cpu['bus_errors']} bus errors", cpu["bus_errors"])) if bad]
        out.append(_kv("State", f"CPU {cpu_txt}",
                       "engine " + ("streaming" if eng["run"] else "busy" if eng["busy"] else "idle"),
                       ", ".join(errors) if errors else "no errors"))

        # Utilization.
        u, s = d["util"], d["sample"]
        out.append(_head(f"Utilization over {1e3*s['seconds']:.0f} ms"))
        out.append(_line(f"Engine  {_bar(u['engine'])} {_pct(u['engine'])}     "
                         f"Vector  {_bar(u['vector'])} {_pct(u['vector'])}     "
                         f"Overlap {_pct(u['overlap'])}"))
        out.append(_line(f"{u['ops_s']:,.0f} engine ops/s   {u['lists_s']:,.0f} descriptors/s"))

        # Memory.
        m = d["memory"]
        out.append(_head(f"Memory · {m['kind']} · {m['channels']} channels"))
        out.append(_line(f"Read    {_bar(m['gbs']/m['peak_gbs'] if m['peak_gbs'] else 0)} "
                         f"{m['gbs']:6.2f} GB/s of {m['peak_gbs']:.1f} GB/s port peak   "
                         f"read-stall {_pct(m['r_stall'])}"))
        cells = [f"{c:2d} {_bar(x, 4)}{100*x:3.0f}%" for c, x in enumerate(m["channel_util"])]
        for i in range(0, len(cells), 6):
            out.append(_line("  ".join(cells[i:i + 6])))

        # Model / process.
        out.append(_head("Model"))
        load, chat = d.get("load"), d.get("chat")
        if load:
            age = _age(time.time() - load.get("time", time.time()))
            out.append(_kv("Weights", os.path.basename(load.get("image", "?")),
                           f"{load.get('bytes', 0)/2**30:.2f} GiB, uploaded {age}"))
        else:
            out.append(_kv("Weights", "unknown (no upload recorded)"))
        if chat and chat.get("alive"):
            out.append(_kv("Process", f"pid {chat['pid']} {chat.get('tool', '')}",
                           f"{chat.get('state', '?')}, {chat.get('tokens', 0)} tokens",
                           f"{chat.get('tok_s', 0):.2f} tok/s"))
        else:
            out.append(_kv("Process", "none"))
        out.append("╰" + "─"*(WIDTH - 2) + "╯")
    return "\n".join(out)

# Run ----------------------------------------------------------------------------------------------

def open_cards(args):
    if args.fake:
        maps = find_csr_maps(args.csr)
        if not maps:
            raise SystemExit("trixium smi --fake: needs a csr.csv (--csr).")
        return [(FakeCard(maps[0]), maps[0])]
    devices = args.device or find_devices()
    if not devices:
        raise SystemExit("trixium smi: no /dev/trixium* (is the driver loaded? "
                         "trixium/software/kernel/init.sh)")
    maps  = find_csr_maps(args.csr)
    cards = []
    for device in devices:
        card = DriverCard(device)
        # The register map whose identifier is the card's (identifier_mem sits at the same
        # place in every LiteX SoC), else the first one given.
        match = None
        for m in maps:
            if read_identifier(card, m) == m.identifier:
                match = m
                break
        if match is None and args.csr and maps:
            match = maps[0]
        cards.append((card, match))
    return cards

def run(args):
    cards = open_cards(args)
    while True:
        devices = []
        for card, csr_map in cards:
            if csr_map is None:
                devices.append({"device": card.device, "ok": False,
                                "error": "no csr.csv matching the card's identifier (--csr)"})
                continue
            try:
                devices.append(query(card, csr_map, args.interval))
            except OSError as e:
                devices.append({"device": card.device, "ok": False, "error": str(e)})
        if args.json:
            print(json.dumps(devices, indent=1))
        else:
            if args.loop:
                print("\033[H\033[J", end="")
            print(table(devices))
        sys.stdout.flush()
        if not args.loop:
            return 0
        time.sleep(args.loop)

def add_arguments(parser):
    parser.add_argument("--device",   action="append",           help="Card device, e.g. /dev/trixium0 (repeatable; default: all).")
    parser.add_argument("--csr",      action="append",           help="Build csr.csv (repeatable; default: ./, ~/trixium/, build/*/).")
    parser.add_argument("-l", "--loop", type=float, metavar="SEC", help="Refresh every SEC seconds.")
    parser.add_argument("--interval", type=float, default=0.25,  help="Counter sampling interval (s).")
    parser.add_argument("--json",     action="store_true",       help="JSON output.")
    parser.add_argument("--fake",     action="store_true",       help="Synthetic card (demo, tests).")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="trixium-smi", description="TriXium card status.")
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    sys.exit(main())

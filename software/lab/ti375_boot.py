#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Ti375 kit's power-on -> chat-ready timeline, measured from the workstation its USB is on.

    python3 -m software.lab.ti375_boot [--name trixium.local | --ip A.B.C.D] [--log boot.log]

Sequence: the script finds the kit on USB (its FT4232H's product string names the Ti375), opens
the console, and if the board is running (answers on the LAN) asks for a power-off and waits for
it (the USB device goes away with the board's power, or the LAN goes silent). Then it takes the
power-on: t=0 is the kit's FT4232H re-enumerating on USB (a few hundred ms after the switch), or
the first console byte when the USB chip stays powered (then the FPGA configuration time is not
in the measure), or the trigger file `poweron` in the current directory (touch it as you flip the
switch, for a network-only measure with no USB). From t=0 it logs every console line with its
timestamp, the first ICMP reply, the first HTTP 200 on GET / and the moment the mDNS name
resolves, and stops a few seconds after TRIXIUM-READY and the first page served. The summary
lists the boot's milestones (FPGA configured, BIOS, Liftoff, eMMC load, tokenizer, web server,
READY, DHCP lease, first page).

2026-10-26, v23 in the kit's flash: FPGA configured 4.9 s after power-on, Liftoff 5.7 s, the 2B
from the eMMC 6.9 -> 16.4 s (9.6 s), TRIXIUM-READY 18.2 s, the page served 19.4 s.
"""
import os, sys, glob, time, socket, argparse, subprocess, threading, urllib.request

BAUD     = 115200
WAIT_RDY = 150      # s after power-on to wait for READY + HTTP
TAIL     = 6        # s of extra console after both are seen (DHCP lease line)
TRIG     = "poweron"

KEYS = (("Liftoff", "liftoff"), ("SDINIT ok", "sdinit"), ("SDBOOT", "sdboot"), ("TOKLOAD", "tokload"),
        ("inference ready", "hdrload"), ("WEB DHCP", "webinit"), ("TRIXIUM-READY", "ready"),
        ("WEB http", "lease"), ("BIOS", "bios"), ("Serial boot", "serialboot"))
NAMES = [("trigger", "poweron trigger file"), ("usb", "kit enumerated on USB"),
         ("uart", "first console byte (FPGA configured)"),
         ("bios", "BIOS banner"), ("serialboot", "BIOS serial boot attempt"), ("liftoff", "Liftoff (firmware starts)"),
         ("sdinit", "eMMC initialised"), ("sdboot", "model load from the eMMC starts"),
         ("hdrload", "inference ready"), ("tokload", "tokenizer loaded"), ("webinit", "web server up"),
         ("ready", "TRIXIUM-READY"), ("icmp", "first ICMP reply"), ("lease", "DHCP lease (WEB http line)"),
         ("mdns", "mDNS name resolves"), ("http", "first HTTP 200 on GET /")]


class Measure:
    def __init__(self, ip, name, log, wait):
        self.ip, self.name, self.wait = ip, name, wait
        self.log = open(log, "w")
        self.t0 = None
        self.marks = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # Output ----------------------------------------------------------------------------------

    def out(self, s, t=None):
        t = time.time() if t is None else t
        stamp = f"{t - self.t0:8.3f}" if self.t0 is not None else time.strftime("%H:%M:%S")
        line = f"[{stamp}] {s}"
        with self.lock:
            print(line, flush=True)
            self.log.write(line + "\n"); self.log.flush()

    def mark(self, name, t=None):
        t = time.time() if t is None else t
        with self.lock:
            self.marks.setdefault(name, t)

    # The kit on USB ----------------------------------------------------------------------------

    @staticmethod
    def find_kit():
        for d in glob.glob("/sys/bus/usb/devices/*"):
            try:
                if "Ti375" in open(os.path.join(d, "product")).read():
                    return d
            except OSError:
                pass
        return None

    @staticmethod
    def kit_tty(d):
        base = os.path.basename(d)
        for pat in (base + ":1.2/ttyUSB*", base + ":1.2/tty/ttyUSB*"):   # The third port: the UART.
            for t in glob.glob(os.path.join(d, pat)):
                return "/dev/" + os.path.basename(t)
        return None

    def open_uart(self, d, wait=10):
        import serial
        t_start = time.time()
        while time.time() - t_start < wait:
            tty = self.kit_tty(d)
            if tty and os.path.exists(tty):
                try:
                    return serial.Serial(tty, BAUD, timeout=0), tty
                except Exception:
                    pass
            time.sleep(0.05)
        return None, None

    # The network -------------------------------------------------------------------------------

    def ping_once(self, w="0.3"):
        return subprocess.run(["ping", "-c", "1", "-W", w, "-n", self.ip], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0

    def ping_loop(self):
        while not self.stop.is_set():
            t = time.time()
            if self.ping_once():
                self.mark("icmp", t); self.out(f"ICMP: {self.ip} answers", t)
                return
            time.sleep(0.15)

    def http_loop(self):
        while not self.stop.is_set():
            t = time.time()
            try:
                with urllib.request.urlopen(f"http://{self.ip}/", timeout=0.6) as r:
                    body = r.read()
                    if r.status == 200:
                        self.mark("http", t); self.out(f"HTTP: GET / -> 200, {len(body)} bytes", t)
                        return
            except Exception:
                pass
            time.sleep(0.15)

    def mdns_loop(self):
        if not self.name:
            return
        while not self.stop.is_set():
            t = time.time()
            try:
                a = socket.getaddrinfo(self.name, 80, socket.AF_INET)
                self.mark("mdns", t); self.out(f"mDNS: {self.name} -> {a[0][4][0]}", t)
                return
            except Exception:
                pass
            time.sleep(0.3)

    @staticmethod
    def trigger_time():
        try:
            return os.path.getmtime(TRIG)
        except OSError:
            return None

    # The measure -------------------------------------------------------------------------------

    def run(self):
        if os.path.exists(TRIG):
            os.remove(TRIG)
        self.out(f"waiting for the kit on USB (product 'Ti375')... or the {TRIG} trigger file")
        t_start = time.time()
        d = None
        while True:
            d = self.find_kit()
            if d:
                break
            tt = self.trigger_time()
            if tt:
                self.t0 = tt; self.mark("trigger", tt)
                self.out(f"TRIGGER: {TRIG} file (t=0), no kit on USB: network probes only")
                break
            if time.time() - t_start > self.wait:
                self.out("no kit on USB, giving up")
                return 1
            time.sleep(0.05)

        ser = tty = None
        buf = b""
        first_byte = None
        if d:
            self.out(f"USB: kit found as {os.path.basename(d)}")
            ser, tty = self.open_uart(d)
            self.out(f"UART: {tty} open" if ser else "UART: no tty for the kit (if02)")

            # If the board runs, wait for it to go off.
            if self.ping_once("0.5") or self.ping_once("0.5"):
                self.out("the board is ON (answers on the LAN): POWER IT OFF now")
                t_start = t_ok = time.time()
                usb_gone = False
                while True:
                    if not self.find_kit():
                        usb_gone = True
                        break
                    if self.ping_once():
                        t_ok = time.time()
                    elif time.time() - t_ok > 2.0:
                        break
                    if ser:
                        try:
                            ser.read(65536)      # discard: the board is still running
                        except Exception:
                            usb_gone = True
                            break
                    if time.time() - t_start > self.wait:
                        self.out("the board never went off, giving up")
                        return 1
                    time.sleep(0.1)
                if usb_gone:
                    try:
                        ser and ser.close()
                    except Exception:
                        pass
                    ser = None
                    self.out("kit gone from USB (board off): POWER IT ON now")
                else:
                    self.out("the board is silent on the LAN (off): POWER IT ON now")
            else:
                self.out("the board is OFF (no answer on the LAN): POWER IT ON now")

            # The power-on: USB (re)enumeration, the trigger file, or the first console byte.
            t_start = time.time()
            while True:
                d2 = self.find_kit()
                if ser is None and d2:
                    self.t0 = time.time(); self.mark("usb", self.t0)
                    self.out(f"USB: kit enumerated as {os.path.basename(d2)} (t=0, the power-on reference)")
                    ser, tty = self.open_uart(d2)
                    self.out(f"UART: {tty} open" if ser else "UART: no tty for the kit (if02)")
                    break
                if ser is not None:
                    try:
                        data = ser.read(4096)
                    except Exception:
                        try: ser.close()
                        except Exception: pass
                        ser = None
                        self.out("kit gone from USB (board off): POWER IT ON now")
                        continue
                    if data:
                        self.t0 = time.time(); self.mark("uart", self.t0)
                        first_byte = self.t0; buf = data
                        self.out("UART: first byte (t=0: the FPGA is configured; its configuration time is not measured)")
                        break
                tt = self.trigger_time()
                if tt:
                    self.t0 = tt; self.mark("trigger", tt)
                    self.out(f"TRIGGER: {TRIG} file (t=0)")
                    break
                if time.time() - t_start > self.wait:
                    self.out("no power-on seen, giving up")
                    return 1
                time.sleep(0.02)

        for th in (self.ping_loop, self.http_loop, self.mdns_loop):
            threading.Thread(target=th, daemon=True).start()

        # The boot, until READY on the console and the web page answers.
        line_t = self.t0 if buf else None
        t_done = None
        while True:
            t = time.time()
            if ser:
                try:
                    data = ser.read(4096)
                except Exception:
                    self.out("UART: lost"); ser = None; data = b""
                if data:
                    if first_byte is None:
                        first_byte = t; self.mark("uart", t)
                        self.out("UART: first byte (FPGA configured, BIOS running)", t)
                    if not buf:
                        line_t = t
                    buf += data
                while b"\n" in buf:
                    ln, buf = buf.split(b"\n", 1)
                    s = ln.decode("utf-8", "replace").rstrip("\r")
                    if s.strip():
                        self.out("  | " + s, line_t)
                        for k, tag in KEYS:
                            if k in s:
                                self.mark(tag, line_t)
                    line_t = t
            with self.lock:
                have = set(self.marks)
            need = {"ready", "http"} if ser else {"http"}
            if t_done is None and need <= have:
                t_done = t
            if t_done is not None and t - t_done > TAIL:
                break
            if t - self.t0 > WAIT_RDY:
                self.out("timeout waiting for READY/HTTP")
                break
            time.sleep(0.01)
        self.stop.set()
        if ser:
            ser.close()

        self.out("SUMMARY (seconds after t=0):")
        for k, label in NAMES:
            if k in self.marks:
                self.out(f"  {self.marks[k] - self.t0:7.2f}  {label}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Ti375 kit: power-on -> chat-ready timeline.")
    parser.add_argument("--name", default="trixium.local", help="The board's mDNS name (resolved at start for "
                        "the probes, and watched during the boot).")
    parser.add_argument("--ip",   default=None, help="The board's address, if the name does not resolve.")
    parser.add_argument("--log",  default="ti375_boot.log", help="Timestamped console + probe log.")
    parser.add_argument("--wait", type=int, default=1800, help="Seconds to wait for each step (USB, off, on).")
    args = parser.parse_args()
    ip = args.ip
    if ip is None:
        try:
            ip = socket.getaddrinfo(args.name, 80, socket.AF_INET)[0][4][0]
        except OSError:
            sys.exit(f"{args.name} does not resolve: pass --ip (the board must have run once for its DHCP lease)")
    return Measure(ip, args.name, args.log, args.wait).run()

if __name__ == "__main__":
    sys.exit(main())

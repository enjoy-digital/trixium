#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Efinix Ti375 C529 kit from the workstation it is plugged into: no board host, no PCIe.

The kit's FT4232H gives JTAG (the bitstream, loaded volatile: the SPI flash is never written) and
the SoC's UART (the console, 115200); its RJ45 gives Etherbone (the CSRs) and the UDP weight port
(software/host/udpload.py). Used by `trixium deploy --board ti375` (software/lab/deploy.py).

  python3 -m software.host.ti375 load build/ti375_v7/gateware/efinix_ti375_c529_dev_kit.bit
  python3 -m software.host.ti375 boot build/ti375_v7/trixium.bin
  python3 -m software.host.ti375 server --iface enp5s0      # litex_server on that interface

A workstation with several interfaces on the board's subnet must reach it through the wired
one: `--iface` binds the Etherbone server's and the uploader's sockets to it (SO_BINDTODEVICE,
no root needed).
"""

import os
import sys
import time
import socket
import argparse
import subprocess

TTY     = "/dev/ttyUSB2"   # The FT4232H's third port: the SoC's UART.
BAUD    = 115200
BOARD_IP = "<board-host>"
MAIN_RAM = 0x4000_0000     # Where the BIOS serial-boots the firmware.

# JTAG ---------------------------------------------------------------------------------------------

def load(bit):
    """The bitstream into the FPGA's configuration memory (volatile; the flash is untouched)."""
    from litex.build.efinix.programmer import EfinixProgrammer
    EfinixProgrammer(family="Titanium").load_bitstream(bit)

# Serial Boot --------------------------------------------------------------------------------------

def boot(firmware, tty=TTY, timeout=240):
    """The firmware through the BIOS's serial boot (litex_term in a pty, as it wants a terminal):
    from the BIOS prompt (after `load`, once the BIOS has given up booting on its own). Returns
    the boot transcript; raises when the firmware does not come up."""
    import pexpect
    c = pexpect.spawn(f"{sys.executable} -m litex.tools.litex_term {tty} --speed {BAUD} "
                      f"--kernel {firmware} --kernel-adr {MAIN_RAM:#x}",
                      encoding="utf-8", codec_errors="replace", timeout=timeout)
    log = []
    try:
        time.sleep(2)
        c.send("\r")
        time.sleep(0.5)
        c.send("serialboot\r")
        i = c.expect([r"TRIXIUM-READY", pexpect.TIMEOUT, pexpect.EOF], timeout=timeout)
        log.append(c.before or "")
        if i != 0:
            raise RuntimeError("the firmware did not come up:\n" + "".join(log)[-2000:])
        c.expect([r"trixium> ", pexpect.TIMEOUT], timeout=120)
        log.append(c.before or "")
    finally:
        c.close(force=True)
    return "".join(log).replace("\r", "")

# Etherbone ----------------------------------------------------------------------------------------

def bind_sockets(iface):
    """Bind every socket this process creates to `iface` (SO_BINDTODEVICE)."""
    orig = socket.socket.__init__
    def init(self, *a, **kw):
        orig(self, *a, **kw)
        try:
            self.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode() + b"\0")
        except OSError:
            pass
    socket.socket.__init__ = init

def server(ip=BOARD_IP, iface=None):
    """litex_server over UDP to the board (blocks)."""
    if iface:
        bind_sockets(iface)
    import runpy
    sys.argv = ["litex_server", "--udp", "--udp-ip", ip]
    runpy.run_module("litex.tools.litex_server", run_name="__main__")

def start_server(ip=BOARD_IP, iface=None, port=1234, wait=10.0):
    """`server` in a child process; returns it once the TCP port answers."""
    cmd = [sys.executable, "-m", "software.host.ti375", "server", "--ip", ip] + (["--iface", iface] if iface else [])
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            socket.create_connection(("localhost", port), timeout=0.5).close()
            return p
        except OSError:
            time.sleep(0.2)
    p.kill()
    raise RuntimeError("litex_server did not start")

def ping(ip=BOARD_IP, iface=None):
    cmd = ["ping", "-c", "2", "-W", "1"] + (["-I", iface] if iface else []) + [ip]
    return subprocess.run(cmd, capture_output=True).returncode == 0

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load");   p.add_argument("bit")
    p = sub.add_parser("boot");   p.add_argument("firmware"); p.add_argument("--tty", default=TTY)
    p = sub.add_parser("server"); p.add_argument("--ip", default=BOARD_IP); p.add_argument("--iface", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "load":
        load(args.bit)
    elif args.cmd == "boot":
        print(boot(args.firmware, args.tty))
    else:
        server(args.ip, args.iface)

if __name__ == "__main__":
    main()

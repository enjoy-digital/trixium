#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Efinix Ti375 C529 kit from the workstation it is plugged into: no board host, no PCIe.

The kit's FT4232H gives JTAG (the bitstream, volatile, or into the SPI flash with `flash`) and
the SoC's UART (the console, 115200); its RJ45 gives Etherbone (the CSRs) and the UDP weight port
(software/host/udpload.py). Used by `trixium deploy --board ti375` (software/lab/deploy.py) and
`trixium chat --board ti375` (ensure_model: the weights uploaded only when not already there).

  python3 -m software.host.ti375 load build/ti375_v7/gateware/efinix_ti375_c529_dev_kit.bit
  python3 -m software.host.ti375 boot build/ti375_v7/trixium.bin
  python3 -m software.host.ti375 flash build/ti375_v10 firmware/trixium.bin   # power-up boot
  python3 -m software.host.ti375 model models/bons/bonsai2-2b-n4-ti375.img    # upload unless there
  python3 -m software.host.ti375 model <img> --tokenizer models/2b/tokenizer.bin # + on-board chat
  python3 -m software.host.ti375 model <img> --tokenizer <btok> --persist     # + to the SD card
  python3 -m software.host.ti375 server --iface enp5s0      # litex_server on that interface
  python3 -m software.host.ti375 smi [-w 1]                 # trixium smi over Etherbone
  python3 -m software.host.ti375 dashboard [--watts 7.5]    # the live page, http://localhost:8375/

A build with `--with-spi-flash` boots the firmware from the flash at power-up (LiteX's
flashboot, after the BIOS's serial boot: a host answering that one still boots its own).

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
import configparser

TTY     = "/dev/ttyUSB2"   # The FT4232H's third port: the SoC's UART.
BAUD    = 115200
BOARD_IP = "<board-host>"
MAIN_RAM = 0x4000_0000     # Where the BIOS serial-boots the firmware.
JTAG_HZ  = 15_000_000      # JTAG clock for loads: 11.5 s (Efinity's default 6 MHz: ~17 s; at 30 MHz
                           # the programmer reports success and nothing is configured).
DEVICE_ID = 0x006A0A79     # The Ti375's JTAG ID (the flash bridge image's name).
FIRMWARE_FLASH_OFFSET = 0x0200_0000   # trixium_ti375.py's (the BIOS's FLASH_BOOT_ADDRESS).

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def config():
    """(tty, ip, iface) of the kit: ~/.config/trixium/boards.ini [ti375] (doc/boards.ini.example),
    TRIXIUM_TI375_<KEY> overriding, then the defaults above."""
    ini = configparser.ConfigParser()
    ini.read(os.environ.get("TRIXIUM_BOARDS_INI", os.path.expanduser("~/.config/trixium/boards.ini")))
    sec = ini["ti375"] if ini.has_section("ti375") else {}
    get = lambda k, d=None: os.environ.get(f"TRIXIUM_TI375_{k.upper()}", sec.get(k, d))
    return get("tty", TTY), get("ip", BOARD_IP), get("iface")

# JTAG ---------------------------------------------------------------------------------------------

def _programmer():
    from litex.build.efinix.programmer import EfinixProgrammer
    return EfinixProgrammer(family="Titanium")

def kit_usb():
    """(bus, address) of the kit's FT4232H, found by its USB product string: other FTDI devices on
    the machine (without serial numbers, the kit's has none either) must not be programmed."""
    import glob
    for d in glob.glob("/sys/bus/usb/devices/*"):
        try:
            if "Ti375" in open(os.path.join(d, "product")).read():
                return int(open(os.path.join(d, "busnum")).read()), int(open(os.path.join(d, "devnum")).read())
        except OSError:
            continue
    raise RuntimeError("no Ti375 kit on USB")

def _efx(args, prog=None):
    """Efinity's ftdi_program.py with only the kit visible (software/host/efx_select.py)."""
    prog = prog or _programmer()
    bus, address = kit_usb()
    cmd = [prog.efinity_path + "/bin/python3", os.path.join(REPO, "software", "host", "efx_select.py"),
           str(bus), str(address), prog.efinity_path + "/pgm/bin/efx_pgm/ftdi_program.py"] + args
    r = subprocess.run(cmd, env=prog.env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    return r

def _openfpgaloader():
    """openFPGALoader with the Ti375 (>= 1.2: build/tools/ first, then PATH), or None."""
    import shutil
    for exe in (os.path.join(REPO, "build", "tools", "openFPGALoader"), shutil.which("openFPGALoader")):
        if exe and os.path.exists(exe):
            v = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout + \
                subprocess.run([exe, "--version"], capture_output=True, text=True).stderr
            m = __import__("re").search(r"v(\d+)\.(\d+)", v)
            if m and (int(m.group(1)), int(m.group(2))) >= (1, 2):
                return exe
    return None

def load(bit, hz=JTAG_HZ):
    """The bitstream into the FPGA's configuration memory (volatile): openFPGALoader when it knows
    the Ti375 (~4.5 s, the kit picked by its USB bus/address), else Efinity's programmer at `hz`
    (6 MHz retried; a load counts only when it read the device's ID: its exit code alone does not
    tell)."""
    bus, address = kit_usb()
    ofl = _openfpgaloader()
    if ofl:
        r = subprocess.run([ofl, "-c", "efinix_jtag_ft4232", "--busdev-num", f"{bus}:{address}",
                            "--freq", str(int(hz)), bit], capture_output=True, text=True)
        if r.returncode == 0:
            return
        sys.stdout.write(r.stdout + r.stderr)
    for f in ([hz, 6_000_000] if hz != 6_000_000 else [hz]):
        r = _efx([bit, "-m", "jtag", "--jtag_clock_freq", str(int(f))])
        if r.returncode == 0 and f"{DEVICE_ID:#010X}"[2:] in r.stdout.upper():
            return
    raise OSError(f"JTAG load of {bit} failed")

def _hex_bytes(data, path):
    """Efinity's flash .hex: one byte per line, upper-case hex."""
    with open(path, "w") as f:
        f.write("".join(f"{b:02X}\n" for b in data))

def flash(build, firmware):
    """Bitstream + firmware into the kit's configuration flash (through Efinity's JTAG bridge
    image): at power-up the FPGA configures itself and the BIOS boots the firmware (flashboot,
    LiteX's FBI: length + CRC-32 + the binary). The FPGA runs the bridge image afterwards:
    reload the bitstream (`load`) or power-cycle."""
    import tempfile
    from litex.soc.software.crcfbigen import insert_crc
    hexf = os.path.join(build, "gateware", "efinix_ti375_c529_dev_kit.hex")
    if not os.path.exists(hexf):
        raise SystemExit(f"{hexf} missing (the build's flash image)")
    prog = _programmer()
    with tempfile.TemporaryDirectory() as d:
        fbi = os.path.join(d, "firmware.fbi")
        insert_crc(firmware, fbi_mode=True, o_filename=fbi, little_endian=True)
        fw_hex = os.path.join(d, "firmware.hex")
        with open(fbi, "rb") as f:
            _hex_bytes(f.read(), fw_hex)
        # Efinity's JTAG bridge image first, then each image at its offset (verified).
        load(prog._bridge_image_path(None, DEVICE_ID))
        for addr, f in ((0, hexf), (FIRMWARE_FLASH_OFFSET, fw_hex)):
            r = _efx([f, "-m", "jtag_bridge", "--address", hex(addr)], prog)
            if r.returncode != 0 or "verify successful" not in r.stdout:
                raise OSError(f"flash write at {addr:#x} failed")

# Serial Boot --------------------------------------------------------------------------------------

def boot(firmware, tty=TTY, timeout=240, bit=None):
    """The firmware through the BIOS's serial boot (litex_term in a pty, as it wants a terminal).
    With `bit`: litex_term is attached first, then the bitstream loaded, so it answers the BIOS's
    serial boot as soon as the BIOS starts (no wait, and a firmware in the flash cannot boot
    instead); without, from the BIOS prompt. Returns the boot transcript; raises when the
    firmware does not come up."""
    import pexpect
    c = pexpect.spawn(f"{sys.executable} -m litex.tools.litex_term {tty} --speed {BAUD} "
                      f"--kernel {firmware} --kernel-adr {MAIN_RAM:#x}",
                      encoding="utf-8", codec_errors="replace", timeout=timeout)
    log = []
    try:
        time.sleep(1)
        if bit:
            load(bit)
        else:
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
    """Bind every UDP socket this process creates to `iface` (SO_BINDTODEVICE): the server's
    link to the board, not its TCP listener (its clients connect over loopback)."""
    orig = socket.socket.__init__
    def init(self, *a, **kw):
        orig(self, *a, **kw)
        if self.type == socket.SOCK_DGRAM:
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

# Model --------------------------------------------------------------------------------------------

# ctrl_scratch (LiteX's SoC scratch register) resets with the SoC (power-up, a bitstream load, the
# reset button): an upload leaves a random tag in it and in its host record, and a later start
# finds the weights still in the DRAM when both still agree (the same image, file unchanged).
SCRATCH_RESET = 0x12345678

def image_id(image):
    st = os.stat(image)
    return dict(image=os.path.realpath(image), bytes=st.st_size, mtime=int(st.st_mtime))

def ensure_model(image, csr_csv=None, force=False, verbose=False, tokenizer=None):
    """The weights of `image` on the kit: uploaded over UDP (each transfer CRC-checked on the
    builds with the port's CRC) unless the kit still holds them, then `hdrload`. With a
    `tokenizer` (BTOK), staged with the weights and loaded on the board (`tokload`): a plain
    serial terminal can then chat (the firmware's `chat`). Returns (uploaded, seconds)."""
    from litex.tools.litex_client import RemoteClient
    from software.host.ebsmi import state_read, state_write
    from software.host.udpload import UDPLink, upload, upload_tokenizer
    from software.host.console import run as console_run
    import random
    tty, ip, iface = config()
    csr_csv = csr_csv or loaded_csr_csv()
    server  = None
    try:
        socket.create_connection(("localhost", 1234), timeout=0.5).close()
    except OSError:
        server = start_server(ip, iface)
    t0 = time.time()
    uploaded = False
    try:
        bus = RemoteClient(csr_csv=csr_csv)
        bus.open()
        rec = state_read("load") or {}
        same = (not force and rec.get("board") == "ti375" and rec.get("tag")
                and bus.regs.ctrl_scratch.read() == rec["tag"]
                and {k: rec.get(k) for k in ("image", "bytes", "mtime")} == image_id(image)
                and (tokenizer is None or rec.get("tokenizer") == image_id(tokenizer)))
        if not same:
            bus.regs.ctrl_scratch.write(SCRATCH_RESET)     # Invalid while the weights change.
            n, resent = upload(bus, UDPLink(ip, iface=iface), image, verbose=verbose)
            if tokenizer:
                upload_tokenizer(bus, UDPLink(ip, iface=iface), tokenizer, verbose=verbose)
            tag = random.randrange(1, 1 << 32)
            while tag == SCRATCH_RESET:
                tag = random.randrange(1, 1 << 32)
            bus.regs.ctrl_scratch.write(tag)
            state_write("load", board="ti375", tag=tag, csr_csv=csr_csv, time=int(time.time()),
                        **image_id(image), build=os.path.basename(os.path.dirname(csr_csv)),
                        tokenizer=image_id(tokenizer) if tokenizer else None)
            uploaded = True
        bus.close()
    finally:
        if server:
            server.kill()
    out = console_run(["hdrload"], tty, wait=20, until="inference ready", baud=BAUD)
    if "inference ready" not in out:
        raise RuntimeError("hdrload failed:\n" + out[-800:])
    if tokenizer:
        out = console_run(["tokload"], tty, wait=20, until="TOKLOAD", baud=BAUD)
        if "TOKLOAD" not in out:
            raise RuntimeError("tokload failed:\n" + out[-800:])
    return uploaded, time.time() - t0

# Status -------------------------------------------------------------------------------------------

def loaded_csr_csv():
    """The csr.csv of the build the deploy last put on the kit (its "load" record), else the
    newest build/ti375_*."""
    from software.host.ebsmi import state_read
    load = state_read("load")
    if load and load.get("board") == "ti375" and os.path.exists(load.get("csr_csv", "")):
        return load["csr_csv"]
    import glob
    builds = sorted(glob.glob(os.path.join(REPO, "build", "ti375_*", "csr.csv")), key=os.path.getmtime)
    if not builds:
        raise SystemExit("no build/ti375_*/csr.csv: pass --csr-csv")
    return builds[-1]

def smi(csr_csv=None, interval=0.25, watch=0.0, as_json=False):
    """trixium smi over Etherbone (software/host/ebsmi.py), starting a litex_server on the kit's
    interface when none answers on localhost:1234."""
    from software.host import ebsmi
    tty, ip, iface = config()
    csr_csv = csr_csv or loaded_csr_csv()
    server  = None
    try:
        socket.create_connection(("localhost", 1234), timeout=0.5).close()
    except OSError:
        server = start_server(ip, iface)
    try:
        # The identifier first: a csr.csv of another build would read wrong addresses.
        card = ebsmi.Card(csr_csv)
        ident, want = card.identifier(), str(card.bus.constants.config_identifier)
        card.close()
        if ident != want:
            raise SystemExit(f"the kit runs '{ident}', {csr_csv} is '{want}': pass the loaded build's --csr-csv")
        ebsmi.run(csr_csv, interval=interval, watch=watch, as_json=as_json,
                  where=f"Ti375 · {ip}{' via ' + iface if iface else ''}")
    finally:
        if server:
            server.kill()

def dashboard(csr_csv=None, port=8375, bind="127.0.0.1", watts=None, interval=0.25):
    """The live page (software/host/dashboard.py) for the kit, starting a litex_server on its
    interface when none answers on localhost:1234; it runs next to `trixium chat`."""
    from software.host import ebsmi
    from software.host.dashboard import BoardSource, serve
    tty, ip, iface = config()
    csr_csv = csr_csv or loaded_csr_csv()
    server  = None
    try:
        socket.create_connection(("localhost", 1234), timeout=0.5).close()
    except OSError:
        server = start_server(ip, iface)
    try:
        card = ebsmi.Card(csr_csv)
        ident, want = card.identifier(), str(card.bus.constants.config_identifier)
        card.close()
        if ident != want:
            raise SystemExit(f"the kit runs '{ident}', {csr_csv} is '{want}': pass the loaded build's --csr-csv")
        serve(BoardSource(csr_csv, interval=interval), port, bind, watts)
    finally:
        if server:
            server.kill()

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load");   p.add_argument("bit")
    p = sub.add_parser("boot");   p.add_argument("firmware"); p.add_argument("--tty", default=TTY)
    p.add_argument("--bit", default=None, help="Load this bitstream first (litex_term already attached).")
    p = sub.add_parser("flash");  p.add_argument("build"); p.add_argument("firmware")
    p = sub.add_parser("model");  p.add_argument("image"); p.add_argument("--csr-csv", default=None)
    p.add_argument("--force", action="store_true", help="Upload even when the kit holds the image.")
    p.add_argument("--tokenizer", default=None, help="BTOK to stage and load on the board (its `chat` command).")
    p.add_argument("--persist", action="store_true", help="Then save it to the SD card (sdsave): the kit boots it at power-up.")
    p = sub.add_parser("server"); p.add_argument("--ip", default=BOARD_IP); p.add_argument("--iface", default=None)
    p = sub.add_parser("smi");    p.add_argument("--csr-csv", default=None); p.add_argument("-i", default=0.25, type=float, dest="interval")
    p.add_argument("-w", default=0.0, type=float, dest="watch"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("dashboard"); p.add_argument("--csr-csv", default=None)
    p.add_argument("--port", default=8375, type=int); p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--watts", default=None, type=float); p.add_argument("-i", default=0.25, type=float, dest="interval")
    args = parser.parse_args(argv)
    if args.cmd == "load":
        load(args.bit)
    elif args.cmd == "boot":
        print(boot(args.firmware, args.tty, bit=args.bit))
    elif args.cmd == "flash":
        flash(args.build, args.firmware)
    elif args.cmd == "model":
        uploaded, dt = ensure_model(args.image, args.csr_csv, args.force, verbose=True,
                                    tokenizer=args.tokenizer)
        print(f"{'uploaded' if uploaded else 'already on the kit'}, ready in {dt:.1f} s")
        if args.persist:
            # The firmware copies what the upload staged to the card (a bitstream built with
            # --with-sdcard): minutes, once per model.
            from software.host.console import run as console_run
            tty = config()[0]
            out = console_run(["sdsave"], tty, wait=1800, quiet_after=1800, until="trixium>", baud=BAUD)
            print("\n".join(l for l in out.splitlines() if "chan" in l or "SD" in l or "sd" in l))
            if "SDSAVE" not in out:
                raise SystemExit("sdsave failed")
    elif args.cmd == "smi":
        smi(args.csr_csv, args.interval, args.watch, args.json)
    elif args.cmd == "dashboard":
        dashboard(args.csr_csv, args.port, args.bind, args.watts, args.interval)
    else:
        server(args.ip, args.iface)

if __name__ == "__main__":
    main()

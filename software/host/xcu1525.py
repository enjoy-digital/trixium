#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The standalone XCU1525 (trixium_xcu1525.py --standalone) from a workstation on its network:
no PCIe host. Its FT4232H gives JTAG (the bitstream, volatile, or the configuration flash with
`flash`) and the console; its QSFP1 Ethernet (a QSFP-to-SFP+ adapter and a 1G copper SFP) gives
Etherbone (the CSRs) and the UDP weight port; its QSFP0 (an adapter and an SFP-to-SATA module)
holds the SSD the models boot from. Nothing here has run on hardware yet (doc/plan-xcu1525-
standalone.md): every command takes --dry-run, which prints what it would do.

  python3 -m software.host.xcu1525 load build/xcu1525_standalone/gateware/sqrl_xcu1525.bit
  python3 -m software.host.xcu1525 flash build/xcu1525_standalone firmware/trixium.bin   # power-up boot
  python3 -m software.host.xcu1525 model models/bons/bonsai-27b-n16-xcu1525.img \\
      --tokenizer models/27b/tokenizer.bin --persist --slot 0 [--default]   # onto the SSD
  python3 -m software.host.xcu1525 model <img> --tokenizer <btok>           # upload, no SSD
  python3 -m software.host.xcu1525 server --iface enp5s0                     # litex_server
  python3 -m software.host.xcu1525 smi [-w 1]                                # trixium smi

The flash (MT25QU01G, 128 MB): the bitstream at 0, the firmware as LiteX's flashboot image at
FIRMWARE_FLASH_OFFSET (trixium_xcu1525.py's). openFPGALoader writes it through a spiOverJtag
helper bitstream for the card's package (xcvu9p-fsgd2104), which its release does not ship:
build it with openFPGALoader/spiOverJtag/build.py (Vivado) and install it next to the others
(/usr/local/share/openFPGALoader/). `flash` reads the whole flash back first (--no-backup skips
it), since the card's factory image is overwritten.

The SSD (firmware/storage.h: slots of TRIXIUM_STORAGE_SLOT_SECTORS, version-3 tables) is written
from here: each entry streamed over UDP into the SoC's `satastore` (gateware/memory/sata_upload.py)
at its sectors, its CRC checked against the file's, then the table and the boot sector the same
way; the firmware only reads them (`stboot`, `stls`).
"""

import os
import re
import sys
import time
import socket
import struct
import argparse
import subprocess
import configparser

TTY      = "/dev/ttyUSB1"         # The card's FT4232H UART port (boards.ini [xcu1525] tty).
BAUD     = 115200
BOARD_IP = "<board-host>"          # --eth-ip of the build (Etherbone, the upload port).
OFL_BOARD = "vcu1525"             # openFPGALoader's name for the card (ft4232, xcvu9p-fsgd2104).
FLASH_SIZE = 128 << 20
HELPER   = "spiOverJtag_xcvu9p-fsgd2104.bit.gz"
FIRMWARE_FLASH_OFFSET = 0x0500_0000   # trixium_xcu1525.py's (the BIOS's FLASH_BOOT_ADDRESS).

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def config():
    """(tty, ip, iface) of the card: ~/.config/trixium/boards.ini [xcu1525] (doc/boards.ini.example),
    TRIXIUM_XCU1525_<KEY> overriding, then the defaults above."""
    ini = configparser.ConfigParser()
    ini.read(os.environ.get("TRIXIUM_BOARDS_INI", os.path.expanduser("~/.config/trixium/boards.ini")))
    sec = ini["xcu1525"] if ini.has_section("xcu1525") else {}
    get = lambda k, d=None: os.environ.get(f"TRIXIUM_XCU1525_{k.upper()}", sec.get(k, d))
    return get("tty", TTY), get("ip", BOARD_IP), get("iface")


def run(cmd, dry_run=False, check=True):
    print("+ " + " ".join(cmd))
    if dry_run:
        return None
    r = subprocess.run(cmd, text=True)
    if check and r.returncode != 0:
        raise OSError(f"{cmd[0]} failed ({r.returncode})")
    return r

# JTAG ---------------------------------------------------------------------------------------------

def openfpgaloader():
    """openFPGALoader: build/tools/ first, then PATH."""
    import shutil
    for exe in (os.path.join(REPO, "build", "tools", "openFPGALoader"), shutil.which("openFPGALoader")):
        if exe and os.path.exists(exe):
            return exe
    raise SystemExit("no openFPGALoader (build/tools/ or PATH)")


def helper_present(exe):
    """Where openFPGALoader keeps its spiOverJtag helpers, and whether the card's is there."""
    share = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "openFPGALoader")
    return share, os.path.exists(os.path.join(share, HELPER))


def load(bit, dry_run=False):
    """The bitstream into the FPGA (volatile), as `trixium deploy` does on the lab card."""
    run([openfpgaloader(), "-b", OFL_BOARD, bit], dry_run)


def flash(build, firmware, backup=True, dry_run=False):
    """Bitstream + firmware into the configuration flash: at power-up the FPGA configures itself
    and the BIOS boots the firmware (flashboot, LiteX's FBI: length + CRC-32 + the binary)."""
    import tempfile
    from litex.soc.software.crcfbigen import insert_crc
    exe = openfpgaloader()
    bit = os.path.join(build, "gateware", "sqrl_xcu1525.bit")
    if not os.path.exists(bit) and not dry_run:
        raise SystemExit(f"{bit} missing (the build's bitstream)")
    share, ok = helper_present(exe)
    if not ok:
        print(f"no {HELPER} in {share}: build it with openFPGALoader/spiOverJtag/build.py (Vivado) and "
              f"copy it there (or use Vivado's hardware manager: program_hw_cfgmem)", file=sys.stderr)
        if not dry_run:
            raise SystemExit(1)
    with tempfile.TemporaryDirectory() as d:
        fbi = os.path.join(d, "firmware.fbi")
        if not dry_run:
            insert_crc(firmware, fbi_mode=True, o_filename=fbi, little_endian=True)
        if backup:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            dump = os.path.join(build, f"flash-backup-{stamp}.bin")
            run([exe, "-b", OFL_BOARD, "--dump-flash", "--file-size", str(FLASH_SIZE), dump], dry_run)
        run([exe, "-b", OFL_BOARD, "-f", bit], dry_run)
        run([exe, "-b", OFL_BOARD, "-f", "--offset", hex(FIRMWARE_FLASH_OFFSET), "--file-type", "raw", fbi], dry_run)
    print("flashed: power-cycle the card (the FPGA runs the helper image until then)")

# Console ------------------------------------------------------------------------------------------

def console(cmds, tty=None, wait=60, until="trixium>"):
    from software.host.console import run as console_run
    tty = tty or config()[0]
    return console_run(cmds, tty, wait=wait, until=until, baud=BAUD)

# Etherbone ----------------------------------------------------------------------------------------

def bind_sockets(iface):
    """Bind every UDP socket this process creates to `iface` (SO_BINDTODEVICE)."""
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
    """litex_server over UDP to the card (blocks)."""
    if iface:
        bind_sockets(iface)
    import runpy
    sys.argv = ["litex_server", "--udp", "--udp-ip", ip]
    runpy.run_module("litex.tools.litex_server", run_name="__main__")


def start_server(ip=BOARD_IP, iface=None, port=1234, wait=10.0):
    cmd = [sys.executable, "-m", "software.host.xcu1525", "server", "--ip", ip] + (["--iface", iface] if iface else [])
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


def with_bus(csr_csv, ip, iface, fn):
    """`fn(bus)` with a RemoteClient on a litex_server (started here when none answers)."""
    from litex.tools.litex_client import RemoteClient
    server_p = None
    try:
        socket.create_connection(("localhost", 1234), timeout=0.5).close()
    except OSError:
        server_p = start_server(ip, iface)
    try:
        bus = RemoteClient(csr_csv=csr_csv)
        bus.open()
        try:
            return fn(bus)
        finally:
            bus.close()
    finally:
        if server_p:
            server_p.kill()


def latest_csr_csv():
    import glob
    builds = sorted(glob.glob(os.path.join(REPO, "build", "xcu1525_standalone*", "csr.csv")), key=os.path.getmtime)
    if not builds:
        raise SystemExit("no build/xcu1525_standalone*/csr.csv: pass --csr-csv")
    return builds[-1]

# Models -------------------------------------------------------------------------------------------

STORE_DONE, STORE_ERROR, STORE_BUSY = 1, 2, 8   # satastore_status bits (sata_upload.py: done, error, failed, busy)

def store_hooks(bus, sector0, verbose=False):
    """The `satastore` as udpload.transfer's sink: the entry's sectors from `sector0`, the
    chunk's offset from its base in bytes."""
    regs = bus.regs
    base0 = [None]
    def arm(base, beats):
        if base0[0] is None:
            base0[0] = base
        sector = sector0 + (base - base0[0])//512
        regs.satastore_sector.write(sector)
        regs.satastore_nsectors.write(beats*32//512)
        regs.satastore_crc_reset.write(1)
        regs.satastore_enable.write(1)
        regs.satastore_start.write(1)
    def wait(base, beats):
        t0 = time.time()
        while regs.satastore_status.read() & STORE_BUSY:
            if time.time() - t0 > 120:
                raise TimeoutError(f"satastore: sector {sector0}: not done ({regs.satastore_sectors_done.read()} sectors)")
            time.sleep(0.002)
        st = regs.satastore_status.read()
        regs.satastore_enable.write(0)
        if (st & STORE_ERROR) or not (st & STORE_DONE):
            raise RuntimeError(f"satastore: the device refused a write near sector {sector0} (status {st:#x})")
    return arm, wait


def persist(bus, link, image, tokenizer, slot, label, default=False, verbose=False, dry_run=False):
    """The image (and tokenizer) into SSD slot `slot`: every entry streamed into the SATA store at
    its sectors (the firmware's slot layout, software/host/sdimage.py), then the table, then the
    boot sector when `default`."""
    from software.host import sdimage, udpload
    cc = dict(weight_base=int(bus.constants.trixium_chan_weight_base),
              stride=int(getattr(bus.constants, "trixium_chan_stride", 0)),
              adds_offset=bool(getattr(bus.constants, "trixium_loader_adds_chan_offset", 0)),
              n_channels=int(getattr(bus.constants, "trixium_n_channels", 0)),
              slot_sectors=int(getattr(bus.constants, "trixium_storage_slot_sectors", sdimage.SLOT_SECTORS)))
    transfers = sdimage.entries(image, tokenizer, cc)
    table_sector, table, ents = sdimage.slot_layout(transfers, version=3, slot=slot,
                                                    slot_sectors=cc["slot_sectors"], label=label)
    total = sum(len(e[5]) for e in ents)
    print(f"slot {slot} at sector {table_sector}: {len(ents)} entries, {total/2**20:.1f} MB, label \"{label}\"")
    if dry_run:
        for chan, base, blocks, sector, crc, data in ents:
            print(f"  chan {chan}: base {base:#010x}, {blocks} blocks at sector {sector}, CRC {crc:08x}")
        return
    t0 = time.time()
    # The old table first invalidated: a transfer cut short leaves no image to boot.
    arm, wait = store_hooks(bus, table_sector)
    udpload.transfer(bus, link, 0, 0, bytes(len(table)), arm=arm, wait=wait, verbose=verbose)
    for chan, base, blocks, sector, crc, data in ents:
        padded = data + bytes(-len(data) % 512)
        arm, wait = store_hooks(bus, sector)
        for attempt in range(3):
            udpload.transfer(bus, link, chan, base, padded, arm=arm, wait=wait, verbose=verbose)
            got = bus.regs.satastore_crc.read()
            if got == crc:
                break
            print(f"  chan {chan}: the SSD stream's CRC {got:08x} != {crc:08x}, resending", file=sys.stderr)
        else:
            raise RuntimeError(f"chan {chan}: the SATA store's CRC never matched")
        if verbose:
            print(f"  chan {chan}: {blocks} blocks at sector {sector} OK", file=sys.stderr)
    arm, wait = store_hooks(bus, table_sector)
    udpload.transfer(bus, link, 0, 0, table, arm=arm, wait=wait, verbose=verbose)
    if default:
        arm, wait = store_hooks(bus, sdimage.BOOT_SECTOR)
        udpload.transfer(bus, link, 0, 0, sdimage.boot_sector(slot), arm=arm, wait=wait, verbose=verbose)
    dt = time.time() - t0
    print(f"slot {slot} written: {total/2**20:.1f} MB in {dt:.1f} s ({total/2**20/dt:.1f} MB/s)")


def model(image, csr_csv=None, tokenizer=None, persist_slot=None, label=None, default=False,
          verbose=False, dry_run=False):
    """The weights uploaded over UDP (the loader), `hdrload` (+ `tokload`); with `persist_slot`,
    the image goes to that SSD slot instead (the firmware's `stboot` loads it from there)."""
    from software.host.udpload import UDPLink, upload, upload_tokenizer
    tty, ip, iface = config()
    csr_csv = csr_csv or latest_csr_csv()
    label = label or re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(image))[0])[:31]
    if dry_run:
        from software.host import sdimage
        cc = sdimage.csr_constants(csr_csv)
        print(f"would use {csr_csv}, card {ip}{' via ' + iface if iface else ''}, console {tty}")
        if persist_slot is not None:
            transfers = sdimage.entries(image, tokenizer, cc)
            ts, table, ents = sdimage.slot_layout(transfers, version=3, slot=persist_slot,
                                                  slot_sectors=cc["slot_sectors"], label=label)
            print(f"slot {persist_slot} at sector {ts}: {len(ents)} entries, {sum(len(e[5]) for e in ents)/2**20:.1f} MB")
            for chan, base, blocks, sector, crc, data in ents:
                print(f"  chan {chan}: base {base:#010x}, {blocks} blocks at sector {sector}, CRC {crc:08x}")
        return
    def go(bus):
        link = UDPLink(ip, iface=iface)
        if persist_slot is not None:
            persist(bus, link, image, tokenizer, persist_slot, label, default, verbose)
            out = console([f"stboot {persist_slot}"], tty, wait=600, until="trixium>")
            if "BOOT slot" not in out:
                raise RuntimeError("stboot failed:\n" + out[-1500:])
            return
        n, resent = upload(bus, link, image, verbose=verbose)
        if tokenizer:
            upload_tokenizer(bus, link, tokenizer, verbose=verbose)
        out = console(["hdrload"], tty, wait=20, until="inference ready")
        if "inference ready" not in out:
            raise RuntimeError("hdrload failed:\n" + out[-800:])
        if tokenizer:
            console(["tokload"], tty, wait=20, until="TOKLOAD")
    with_bus(csr_csv, ip, iface, go)

# Status -------------------------------------------------------------------------------------------

def smi(csr_csv=None, interval=0.25, watch=0.0, as_json=False):
    from software.host import ebsmi
    tty, ip, iface = config()
    csr_csv = csr_csv or latest_csr_csv()
    def go(bus):
        ebsmi.run(csr_csv, interval=interval, watch=watch, as_json=as_json,
                  where=f"XCU1525 standalone · {ip}{' via ' + iface if iface else ''}")
    with_bus(csr_csv, ip, iface, go)

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print what would run, run nothing.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load");  p.add_argument("bit")
    p = sub.add_parser("flash"); p.add_argument("build"); p.add_argument("firmware")
    p.add_argument("--no-backup", action="store_true", help="Skip reading the flash back first.")
    p = sub.add_parser("model"); p.add_argument("image"); p.add_argument("--csr-csv", default=None)
    p.add_argument("--tokenizer", default=None, help="BTOK to stage/store with the model (the firmware's chat).")
    p.add_argument("--persist", action="store_true", help="Write the model to an SSD slot instead of the DRAM, then stboot it.")
    p.add_argument("--slot", default=0, type=int, help="--persist: the SSD slot.")
    p.add_argument("--label", default=None, help="--persist: the slot's label (default: the image's name).")
    p.add_argument("--default", action="store_true", help="--persist: make it the slot booted at power-up.")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("console"); p.add_argument("command", nargs="+"); p.add_argument("--tty", default=None)
    p = sub.add_parser("server"); p.add_argument("--ip", default=BOARD_IP); p.add_argument("--iface", default=None)
    p = sub.add_parser("smi"); p.add_argument("--csr-csv", default=None); p.add_argument("-i", default=0.25, type=float, dest="interval")
    p.add_argument("-w", default=0.0, type=float, dest="watch"); p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "load":
        load(args.bit, args.dry_run)
    elif args.cmd == "flash":
        flash(args.build, args.firmware, backup=not args.no_backup, dry_run=args.dry_run)
    elif args.cmd == "model":
        model(args.image, args.csr_csv, args.tokenizer, args.slot if args.persist else None, args.label,
              args.default, args.verbose, args.dry_run)
    elif args.cmd == "console":
        print(console(args.command, args.tty))
    elif args.cmd == "smi":
        smi(args.csr_csv, args.interval, args.watch, args.json)
    else:
        server(args.ip, args.iface)


if __name__ == "__main__":
    main()

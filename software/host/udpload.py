#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Weight upload over UDP (gateware/memory/udp_upload.py) for boards with Ethernet and no PCIe.

The placement is wbload.py's (header+table to channel 0's staging area, each channel to its
window); the loader is armed through its CSRs (Etherbone: LiteX's RemoteClient), the bytes go as
datagrams `[seq, payload]` of `beats_per_datagram` 256-bit beats to the board's upload port. The
port forwards only the next expected datagram, so the host sends a window ahead, reads the
port's `expected` back, and resends from there whatever was lost; the loader's `done` closes each
transfer. Builds with the port's `crc` check every transfer as it lands (zlib's CRC-32 of the
bytes sent against the port's, a mismatch resends the transfer): no read-back, no `imgcheck`
needed on the way; the datagram size is the port's `max_beats` (32 on builds without it).

  litex_server --udp --udp-ip <board-host>
  python3 -m software.host.udpload --csr-csv build/<name>/csr.csv --board-ip <board-host> image.img
"""

import argparse
import socket
import struct
import sys
import time
import zlib

from software.host.wbload import BEAT_BYTES, CHUNK_BEATS, HDR_STAGING_BELOW, constants, wait_done
from model.bons import BONSReader

UPLOAD_PORT = 2345
# The on-board tokenizer's staging (firmware/tokenizer.h): channel 0, 16 MB below the weights, a
# 32-byte record ("TOKS", length) then the BTOK tables; the firmware's `tokload` copies it.
TOK_STAGING_BELOW = 0x1000000
TOK_MAGIC         = 0x534b4f54


class UDPLink:
    """The datagram side: send(seq, payload bytes)."""
    def __init__(self, ip, port=UPLOAD_PORT, iface=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if iface:   # A host with several interfaces on the board's subnet: send through this one.
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode() + b"\0")
        self.addr = (ip, port)
    def send(self, seq, payload):
        self.sock.sendto(struct.pack("<I", seq) + payload, self.addr)


def port_status(regs):
    """The port's counters, and its last datagram when the build has the debug CSRs."""
    st = {k: getattr(regs, "udpload_" + k).read() for k in ("expected", "accepted", "dropped")}
    if hasattr(regs, "udpload_last_flags"):
        flags = regs.udpload_last_flags.read()
        st["last"] = dict(seq=regs.udpload_last_seq.read(), words=regs.udpload_last_words.read(),
            seq_ok=flags & 1, whole_beats=(flags >> 1) & 1, lost=(flags >> 2) & 1, no_slot=(flags >> 3) & 1)
    return st


class CRCError(RuntimeError):
    pass


def transfer(bus, link, chan, base, data, beats_per_datagram=None, window=512, verbose=False,
             stall_s=5.0, retries=3, arm=None, wait=None):
    """One channel's bytes to `base`, in 128 MB armed transfers, each checked by the port's CRC
    when the build has it (a mismatch resends it, up to `retries` times); returns the datagrams
    resent. `arm(base, beats)` / `wait(base, beats)` replace the loader's arming and done wait
    for another sink of the port's stream (the SATA store of the standalone XCU1525)."""
    regs   = bus.regs
    if beats_per_datagram is None:
        beats_per_datagram = regs.udpload_max_beats.read() if hasattr(regs, "udpload_max_beats") else 32
    data   = bytes(data) + bytes(-len(data) % BEAT_BYTES)
    beats  = len(data)//BEAT_BYTES
    resent = 0
    done   = 0
    tries  = 0
    while done < beats:
        n = min(beats - done, CHUNK_BEATS)
        chunk = data[done*BEAT_BYTES:(done + n)*BEAT_BYTES]
        resent += _send_chunk(bus, link, chan, base + done*BEAT_BYTES, chunk, beats_per_datagram,
                              window, stall_s, arm, wait)
        if hasattr(regs, "udpload_crc"):
            got, want = regs.udpload_crc.read(), zlib.crc32(chunk)
            if got != want:
                tries += 1
                if tries > retries:
                    raise CRCError(f"chan {chan}: CRC {got:08x}, want {want:08x} after {retries} resends")
                if verbose:
                    print(f"    chan {chan}: CRC {got:08x} != {want:08x}, resending", file=sys.stderr)
                continue
        tries = 0
        done += n
        if verbose:
            print(f"    chan {chan}: {done}/{beats} beats, {resent} datagrams resent", file=sys.stderr)
    return resent


def _send_chunk(bus, link, chan, base, chunk, beats_per_datagram, window, stall_s, arm=None, wait=None):
    """One armed transfer (<= 128 MB): its datagrams until the port expects no more, then the
    sink's done (the loader's, or `wait`); returns the datagrams resent."""
    regs   = bus.regs
    dgram  = beats_per_datagram*BEAT_BYTES
    n      = len(chunk)//BEAT_BYTES
    resent = 0
    count  = (len(chunk) + dgram - 1)//dgram
    if arm is None:
        regs.trixium_loader_src_sel.write(0)
        regs.trixium_loader_chan.write(chan)
        regs.trixium_loader_base.write(base)
        regs.trixium_loader_length.write(n)
        regs.trixium_loader_resync.write(0)
    regs.udpload_restart.write(0)
    if arm is None:
        regs.trixium_loader_start.write(1)
    else:
        arm(base, n)
    sent, last = 0, -1
    progress = time.time()
    while True:
        expected = regs.udpload_expected.read()
        if expected >= count:
            break
        if expected != last:
            progress = time.time()
        elif time.time() - progress > stall_s:
            raise RuntimeError(f"chan {chan}: no datagram accepted for {stall_s:.0f} s at "
                f"{expected}/{count}: {port_status(regs)}")
        if expected == last and sent > expected:   # No progress: the rest was lost.
            resent += sent - expected
            sent = expected
        last = expected
        stop = min(count, expected + window)
        while sent < stop:
            link.send(sent, chunk[sent*dgram:(sent + 1)*dgram])
            sent += 1
        time.sleep(0.0005)
    if wait is None:
        wait_done(bus, chan, base, n)
    else:
        wait(base, n)
    return resent


def upload(bus, link, image, verbose=False, **kw):
    img = BONSReader(image)
    h   = img.header
    c   = constants(bus)
    if c["n_channels"] and c["n_channels"] != h["n_channels"]:
        raise ValueError(f"image for {h['n_channels']} channels, the SoC has {c['n_channels']}")
    total, resent = 0, 0
    meta = bytes(img.data[:h["payload_offset"]])
    resent += transfer(bus, link, 0, c["weight_base"] - HDR_STAGING_BELOW, meta, verbose=verbose, **kw)
    total  += len(meta)
    for ch in range(h["n_channels"]):
        base = c["weight_base"] if c["adds_offset"] else c["weight_base"] + ch*c["stride"]
        payload = img.channel_payload(ch)
        resent += transfer(bus, link, ch, base, payload, verbose=verbose, **kw)
        total  += len(payload)
    return total, resent


def tokenizer_record(path):
    """The staged form of a BTOK file: the 32-byte record, the tables, padded to whole beats."""
    btok = open(path, "rb").read()
    if btok[:4] != b"BTOK":
        raise ValueError(f"{path}: not a BTOK tokenizer (trixium tokenizer)")
    if len(btok) > TOK_STAGING_BELOW - HDR_STAGING_BELOW - 32:
        raise ValueError(f"{path}: {len(btok)} bytes, more than the staging area")
    data = struct.pack("<II", TOK_MAGIC, len(btok)) + bytes(24) + btok
    return data + bytes(-len(data) % BEAT_BYTES)


def upload_tokenizer(bus, link, path, verbose=False, **kw):
    """The tokenizer to its staging area on channel 0, for the firmware's `tokload`."""
    c = constants(bus)
    data = tokenizer_record(path)
    resent = transfer(bus, link, 0, c["weight_base"] - TOK_STAGING_BELOW, data, verbose=verbose, **kw)
    return len(data), resent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image")
    parser.add_argument("--csr-csv",  default="csr.csv")
    parser.add_argument("--host",     default="localhost", help="litex_server host (Etherbone, the CSRs).")
    parser.add_argument("--port",     default=1234, type=int)
    parser.add_argument("--board-ip", required=True,       help="The board's IP (the upload datagrams).")
    parser.add_argument("--upload-port", default=UPLOAD_PORT, type=int)
    parser.add_argument("--iface",    default=None,         help="Send the datagrams through this interface (SO_BINDTODEVICE).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    from litex.tools.litex_client import RemoteClient
    bus = RemoteClient(host=args.host, port=args.port, csr_csv=args.csr_csv)
    bus.open()
    t0 = time.time()
    n, resent = upload(bus, UDPLink(args.board_ip, args.upload_port, args.iface), args.image, args.verbose)
    dt = time.time() - t0
    print(f"done: {n/1e6:.1f} MB in {dt:.1f} s ({n/1e6/dt:.1f} MB/s), {resent} datagrams resent")
    bus.close()


if __name__ == "__main__":
    main()

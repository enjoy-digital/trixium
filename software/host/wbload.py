#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Weight upload over the CSR bus (LiteX's RemoteClient: Etherbone over UDP, or JTAGBone through
`litex_server --jtag`): a host link for boards without PCIe.

The same staging as the PCIe loader (software/user/trixium_loader.c): the image's header and
table go to channel 0's staging area (the firmware's `hdrload` copies them), each channel's
payload to its window, through the gateware loader's CSR poke path (gateware/memory/loader.py:
`src_sel = 1`, eight 32-bit pokes per 256-bit beat, nothing dropped or reordered, the FIFO's
`poke_full` polled every 256 words, `done` awaited per transfer, 128 MB per armed transfer: the
loader counts beats in 24 bits). Bring-up grade: one CSR write per word (an Etherbone packet
each; litex_server writes every record as an incrementing burst, so pokes to one address cannot
be batched): measured over JTAGBone on the YPCB-00338, ~1,900 writes/s, 10 KB/s - a tiny model
in ~16 minutes, then `hdrload` and `gen` exact; the LiteEth weight port is the fast path.
litex_server --jtag needs an OpenOCD config that sets `_CHIPNAME` (it names the TAP
`$_CHIPNAME.tap`), and the JTAG cable to itself: stop the server and its OpenOCD (by PID) before
programming the card.

  litex_server --udp --udp-ip <board-host>      # or --jtag --jtag-config openocd_xxx.cfg
  python3 -m software.host.wbload --csr-csv build/<name>/csr.csv models/tiny/bonsai-tiny-n2.img
"""

import argparse
import sys
import time

from model.bons import BONSReader

BEAT_BYTES  = 32
CHUNK_BEATS = 4 << 20            # 128 MB per armed transfer (trixium_loader.c).
HDR_STAGING_BELOW = 0x100000     # The header+table staging: 1 MB below the weight base.


def constants(bus):
    """The loader's placement constants from the SoC's csr.csv (lowercase keys)."""
    d = bus.constants.d
    def get(name, default=None):
        if name in d:
            return int(d[name])
        if default is None:
            raise KeyError(f"{name} not in the SoC's constants (csr.csv)")
        return default
    return dict(weight_base = get("trixium_chan_weight_base"),
                stride      = get("trixium_chan_stride", 0),
                adds_offset = bool(get("trixium_loader_adds_chan_offset", 0)),
                n_channels  = get("trixium_n_channels", 0))


def transfer(bus, chan, base, data, verbose=False):
    """One channel's bytes to `base` through the poke path, in 128 MB armed transfers."""
    regs  = bus.regs
    data  = bytes(data)
    pad   = -len(data) % BEAT_BYTES
    data += bytes(pad)
    beats = len(data)//BEAT_BYTES
    done  = 0
    while done < beats:
        n = min(beats - done, CHUNK_BEATS)
        regs.trixium_loader_src_sel.write(1)
        regs.trixium_loader_chan.write(chan)
        regs.trixium_loader_base.write(base + done*BEAT_BYTES)
        regs.trixium_loader_length.write(n)
        regs.trixium_loader_resync.write(0)
        regs.trixium_loader_start.write(1)
        words = memoryview(data)[done*BEAT_BYTES:(done + n)*BEAT_BYTES].cast("I")
        for i, w in enumerate(words):
            if (i & 0xFF) == 0:
                # The FIFO drains at memory speed: it only fills if the link is unusually fast.
                while regs.trixium_loader_status.read() & 4:
                    pass
            regs.trixium_loader_poke.write(int(w))
        wait_done(bus, chan, base + done*BEAT_BYTES, n)
        done += n
        if verbose:
            print(f"    chan {chan}: {done}/{beats} beats", file=sys.stderr)
    return beats


def wait_done(bus, chan, base, beats, timeout=60.0):
    t0 = time.time()
    while not (bus.regs.trixium_loader_status.read() & 1):
        if time.time() - t0 > timeout:
            st = bus.regs.trixium_loader_status.read()
            raise TimeoutError(f"loader timeout chan {chan} addr {base:08x} (status {st:08x}, "
                               f"{(st >> 8) & 0xFFFFFF} of {beats} beats)")
        time.sleep(0.001)


def upload(bus, image, verbose=False):
    """The whole image: header+table to the staging area, every channel to its window.
    Returns the bytes uploaded."""
    img = BONSReader(image)
    h   = img.header
    c   = constants(bus)
    if c["n_channels"] and c["n_channels"] != h["n_channels"]:
        raise ValueError(f"image for {h['n_channels']} channels, the SoC has {c['n_channels']}")
    meta = bytes(img.data[:h["payload_offset"]])
    if verbose:
        print(f"  header+table -> ch0 staging ({len(meta)//1024} KB)", file=sys.stderr)
    total = BEAT_BYTES*transfer(bus, 0, c["weight_base"] - HDR_STAGING_BELOW, meta, verbose)
    for ch in range(h["n_channels"]):
        base = c["weight_base"] if c["adds_offset"] else c["weight_base"] + ch*c["stride"]
        payload = img.channel_payload(ch)
        if verbose:
            print(f"  channel {ch} -> {base:08x} ({len(payload) >> 20} MB)", file=sys.stderr)
        total += BEAT_BYTES*transfer(bus, ch, base, payload, verbose)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image",                           help="BONS image.")
    parser.add_argument("--csr-csv", default="csr.csv",    help="The SoC's csr.csv (registers and constants).")
    parser.add_argument("--host",    default="localhost",  help="litex_server host.")
    parser.add_argument("--port",    default=1234, type=int, help="litex_server port.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    from litex.tools.litex_client import RemoteClient
    bus = RemoteClient(host=args.host, port=args.port, csr_csv=args.csr_csv)
    bus.open()
    t0 = time.time()
    n  = upload(bus, args.image, args.verbose)
    dt = time.time() - t0
    print(f"done: {n/1e6:.1f} MB in {dt:.1f} s ({n/1e3/dt:.1f} KB/s)")
    bus.close()


if __name__ == "__main__":
    main()

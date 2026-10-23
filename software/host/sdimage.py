#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""An SD card image the firmware's `sdboot` loads (firmware/sdload.h): the model (and its
tokenizer) placed where a UDP upload puts them, read by the card straight into the channels'
windows through the weight loader (gateware/memory/sd_upload.py).

  python3 -m software.host.sdimage models/bons/bonsai2-2b-n4-ti375.img \\
      --tokenizer models/2b/tokenizer.bin --csr-csv build/ti375_v20/csr.csv -o card.img
  sudo dd if=card.img of=/dev/sdX bs=4M conv=fsync          # the whole card: no partitions kept

Layout: zeros up to sector 2048 (1 MB), there a 512-byte table (magic "TXSD", version 1, the
entry count, the CRC-32 of the entries, then (chan, base, blocks, sector) per entry), then each
entry's data from its first sector, padded to whole blocks: the BONS header+table (channel 0, the
header staging), each channel's payload (its weight window), the tokenizer record (channel 0, the
tokenizer staging). The card must be SDHC/SDXC (block addressing).
"""

import argparse
import struct
import zlib

from model.bons import BONSReader
from software.host.wbload import HDR_STAGING_BELOW
from software.host.udpload import TOK_STAGING_BELOW, tokenizer_record

SECTOR      = 512
TABLE_AT    = 2048                 # firmware/sdload.h SD_IMAGE_SECTOR
MAGIC       = 0x44535854           # "TXSD"
MAX_ENTRIES = 31


def csr_constants(csr_csv):
    """The loader's placement constants (wbload.constants' keys) from a build's csr.csv."""
    d = {}
    for line in open(csr_csv):
        f = line.strip().split(",")
        if len(f) >= 3 and f[0] == "constant":
            d[f[1]] = f[2]
    get = lambda k, default=None: int(d[k]) if k in d else default
    if get("trixium_chan_weight_base") is None:
        raise SystemExit(f"{csr_csv}: no trixium_chan_weight_base")
    return dict(weight_base=get("trixium_chan_weight_base"), stride=get("trixium_chan_stride", 0),
                adds_offset=bool(get("trixium_loader_adds_chan_offset", 0)),
                n_channels=get("trixium_n_channels", 0))


def entries(image, tokenizer, c):
    """(chan, base, bytes) of each transfer, in load order."""
    img = BONSReader(image)
    h   = img.header
    if c["n_channels"] and c["n_channels"] != h["n_channels"]:
        raise SystemExit(f"image for {h['n_channels']} channels, the SoC has {c['n_channels']}")
    out = [(0, c["weight_base"] - HDR_STAGING_BELOW, bytes(img.data[:h["payload_offset"]]))]
    for ch in range(h["n_channels"]):
        base = c["weight_base"] if c["adds_offset"] else c["weight_base"] + ch*c["stride"]
        out.append((ch, base, bytes(img.channel_payload(ch))))
    if tokenizer:
        out.append((0, c["weight_base"] - TOK_STAGING_BELOW, tokenizer_record(tokenizer)))
    return out


def write_image(path, transfers):
    assert len(transfers) <= MAX_ENTRIES
    table, sector = [], TABLE_AT + 1
    for chan, base, data in transfers:
        blocks = (len(data) + SECTOR - 1)//SECTOR
        table.append((chan, base, blocks, sector))
        sector += blocks
    body = b"".join(struct.pack("<4I", *e) for e in table)
    head = struct.pack("<4I", MAGIC, 1, len(table), zlib.crc32(body)) + body
    with open(path, "wb") as f:
        f.write(bytes(TABLE_AT*SECTOR))
        f.write(head + bytes(SECTOR - len(head)))
        for (chan, base, data), e in zip(transfers, table):
            assert f.tell() == e[3]*SECTOR
            f.write(data + bytes(-len(data) % SECTOR))
    return table, sector*SECTOR


def main(argv=None):
    parser = argparse.ArgumentParser(description="SD card image for the firmware's sdboot.")
    parser.add_argument("image",                       help="BONS image (.img).")
    parser.add_argument("--tokenizer", default=None,   help="BTOK tokenizer to put on the card too.")
    parser.add_argument("--csr-csv",   required=True,  help="The build's csr.csv (the weight placement).")
    parser.add_argument("-o", "--output", required=True, help="Card image to write.")
    args = parser.parse_args(argv)
    table, size = write_image(args.output, entries(args.image, args.tokenizer, csr_constants(args.csr_csv)))
    for chan, base, blocks, sector in table:
        print(f"  chan {chan}: base {base:#010x}, {blocks*SECTOR/2**20:7.2f} MB at sector {sector}")
    print(f"{args.output}: {size/2**20:.1f} MB (dd it to the card's whole device)")


if __name__ == "__main__":
    main()

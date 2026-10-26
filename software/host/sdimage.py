#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""A block-device image the firmware's `stboot`/`sdboot` loads (firmware/storage.h): the model
(and its tokenizer) placed where a UDP upload puts them, read by the device straight into the
channels' windows through the weight loader (gateware/memory/sd_upload.py, sata_upload.py).

  python3 -m software.host.sdimage models/bons/bonsai2-2b-n4-ti375.img \\
      --tokenizer models/2b/tokenizer.bin --csr-csv build/ti375_v20/csr.csv -o card.img
  sudo dd if=card.img of=/dev/sdX bs=4M conv=fsync          # the whole card: no partitions kept

Layout: zeros up to sector 2048 (1 MB), there slot 0's table, then each entry's data from its
first sector, padded to whole blocks: the BONS header+table (channel 0, the header staging),
each channel's payload (its weight window), the tokenizer record (channel 0, the tokenizer
staging). Table version 3 (default; 4 sectors, the data from the 5th): magic "TXSD", 3, the
entry count, the CRC-32 of the entries, a 32-byte label, then (chan, base, blocks, sector low
word, CRC-32 of the data, sector high word) per entry; an entry holds at most MAX_BLOCKS blocks
(one loader transfer, 512 MB), so a longer payload spans several. Version 1 (one sector, no
label, no CRCs, (chan, base, blocks, sector) entries) for the first firmwares. The card must be
SDHC/SDXC (block addressing). `slot_layout` is what software/host/xcu1525.py streams onto a
SATA SSD slot by slot.
"""

import argparse
import struct
import zlib

from model.bons import BONSReader
from software.host.wbload import HDR_STAGING_BELOW
from software.host.udpload import TOK_STAGING_BELOW, tokenizer_record

SECTOR           = 512
TABLE_AT         = 2048                # firmware/storage.h STORAGE_IMAGE_SECTOR
BOOT_SECTOR      = 2047                # STORAGE_BOOT_SECTOR: {"TXBT", slot, ~slot}
BOOT_MAGIC       = 0x54425854          # "TXBT"
MAGIC            = 0x44535854          # "TXSD"
MAX_ENTRIES      = 31                  # version 1
V3_TABLE_SECTORS = 4
V3_MAX_ENTRIES   = 83
MAX_BLOCKS       = (1 << 24)//16       # one loader transfer: 24 bits of 32-byte beats
SLOT_SECTORS     = 1 << 21             # 1 GB: the Ti375's eMMC (TRIXIUM_STORAGE_SLOT_SECTORS elsewhere)


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
                n_channels=get("trixium_n_channels", 0),
                slot_sectors=get("trixium_storage_slot_sectors", SLOT_SECTORS))


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


def split_transfers(transfers, max_blocks=None):
    """The transfers cut into loader-sized pieces: (chan, base, bytes), bases advancing."""
    out = []
    step = (MAX_BLOCKS if max_blocks is None else max_blocks)*SECTOR
    for chan, base, data in transfers:
        for off in range(0, max(len(data), 1), step):
            out.append((chan, base + off, data[off:off + step]))
    return out


def slot_layout(transfers, version=3, slot=0, slot_sectors=SLOT_SECTORS, label=""):
    """One slot: `table` (bytes, the table's sectors) at `table_sector`, and the entries
    [(chan, base, blocks, sector, crc, data)] at their sectors; the firmware's view."""
    table_sector = TABLE_AT + slot*slot_sectors
    if version == 1:
        pieces, sector = transfers, table_sector + 1
        assert len(pieces) <= MAX_ENTRIES
    else:
        pieces, sector = split_transfers(transfers), table_sector + V3_TABLE_SECTORS
        assert len(pieces) <= V3_MAX_ENTRIES, f"{len(pieces)} entries, the table holds {V3_MAX_ENTRIES}"
    ents = []
    for chan, base, data in pieces:
        blocks = (len(data) + SECTOR - 1)//SECTOR
        assert blocks <= MAX_BLOCKS
        ents.append((chan, base, blocks, sector, zlib.crc32(data + bytes(-len(data) % SECTOR)), data))
        sector += blocks
    assert sector - table_sector <= slot_sectors, "the model is larger than a slot"
    if version == 1:
        body  = b"".join(struct.pack("<4I", chan, base, blocks, sec) for chan, base, blocks, sec, _, _ in ents)
        table = struct.pack("<4I", MAGIC, 1, len(ents), zlib.crc32(body)) + body
        table += bytes(SECTOR - len(table))
    else:
        body  = b"".join(struct.pack("<6I", chan, base, blocks, sec & 0xffffffff, crc, sec >> 32)
                         for chan, base, blocks, sec, crc, _ in ents)
        lbl   = label.encode()[:31]
        table = struct.pack("<4I", MAGIC, 3, len(ents), zlib.crc32(body)) + lbl + bytes(32 - len(lbl)) + body
        table += bytes(V3_TABLE_SECTORS*SECTOR - len(table))
    return table_sector, table, ents


def boot_sector(slot):
    """The boot sector's bytes: the slot the power-up boots."""
    return struct.pack("<3I", BOOT_MAGIC, slot, ~slot & 0xffffffff) + bytes(SECTOR - 12)


def write_image(path, transfers, version=3, label=""):
    """The device image (slot 0 at TABLE_AT): (the entries' (chan, base, blocks, sector), the size)."""
    table_sector, table, ents = slot_layout(transfers, version=version, label=label)
    with open(path, "wb") as f:
        f.write(bytes(table_sector*SECTOR))
        f.write(table)
        for chan, base, blocks, sector, crc, data in ents:
            assert f.tell() == sector*SECTOR
            f.write(data + bytes(-len(data) % SECTOR))
        size = f.tell()
    return [e[:4] for e in ents], size


def main(argv=None):
    parser = argparse.ArgumentParser(description="Block-device image for the firmware's stboot/sdboot.")
    parser.add_argument("image",                       help="BONS image (.img).")
    parser.add_argument("--tokenizer", default=None,   help="BTOK tokenizer to put on the device too.")
    parser.add_argument("--csr-csv",   required=True,  help="The build's csr.csv (the weight placement).")
    parser.add_argument("--version",   default=3, type=int, choices=[1, 3], help="Table version (1: the first firmwares).")
    parser.add_argument("--label",     default="",     help="The slot's label (31 characters).")
    parser.add_argument("-o", "--output", required=True, help="Image to write.")
    args = parser.parse_args(argv)
    table, size = write_image(args.output, entries(args.image, args.tokenizer, csr_constants(args.csr_csv)),
                              version=args.version, label=args.label)
    for chan, base, blocks, sector in table:
        print(f"  chan {chan}: base {base:#010x}, {blocks*SECTOR/2**20:7.2f} MB at sector {sector}")
    print(f"{args.output}: {size/2**20:.1f} MB (dd it to the device's whole disk)")


if __name__ == "__main__":
    main()

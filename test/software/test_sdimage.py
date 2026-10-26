#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/host/sdimage.py: the device layout the firmware's stboot reads (firmware/storage.h),
on the tiny hybrid image: the table's place, magic and CRC, each entry on its block, its bytes;
version 1 and version 3 (the 4-sector table, CRCs, 48-bit sectors, entries split at the loader's
longest transfer)."""

import os
import re
import struct
import zlib

import pytest

from model.bons import BONSReader
from software.host import sdimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE = os.path.join(REPO, "models", "tiny", "bonsai-tiny-hybrid-1k-t5-n4.img")
TOKENIZER = os.path.join(REPO, "models", "2b", "tokenizer.bin")


def test_matches_firmware():
    h = open(os.path.join(REPO, "firmware", "storage.h")).read()
    get = lambda name, base=10: int(re.search(r"#define " + name + r"\s+(\(?[0-9a-fA-Fx]+)", h).group(1).lstrip("("), 0)
    assert get("STORAGE_IMAGE_SECTOR") == sdimage.TABLE_AT
    assert get("STORAGE_BOOT_SECTOR") == sdimage.BOOT_SECTOR
    assert get("STORAGE_IMAGE_MAGIC") == sdimage.MAGIC
    assert get("STORAGE_BOOT_MAGIC") == sdimage.BOOT_MAGIC
    assert get("STORAGE_V1_MAX_ENTRIES") == sdimage.MAX_ENTRIES
    assert get("STORAGE_V3_MAX_ENTRIES") == sdimage.V3_MAX_ENTRIES
    assert get("STORAGE_V3_TABLE_SECTORS") == sdimage.V3_TABLE_SECTORS
    assert re.search(r"#define STORAGE_MAX_BLOCKS\s+\(\(1u << 24\)/16\)", h) and sdimage.MAX_BLOCKS == (1 << 24)//16


def _transfers():
    if not os.path.exists(IMAGE):
        pytest.skip("no tiny image")
    c = dict(weight_base=0x1400_0000, stride=0x0B00_0000, adds_offset=True, n_channels=4, slot_sectors=1 << 21)
    tok = TOKENIZER if os.path.exists(TOKENIZER) else None
    return c, tok, sdimage.entries(IMAGE, tok, c)


def test_layout_v1(tmp_path):
    c, tok, transfers = _transfers()
    out = str(tmp_path / "card.img")
    table, size = sdimage.write_image(out, transfers, version=1)
    raw = open(out, "rb").read()
    assert len(raw) == size and raw[:sdimage.TABLE_AT*512] == bytes(sdimage.TABLE_AT*512)
    t = raw[sdimage.TABLE_AT*512:(sdimage.TABLE_AT + 1)*512]
    magic, ver, n, crc = struct.unpack("<4I", t[:16])
    assert (magic, ver, n) == (sdimage.MAGIC, 1, len(transfers))
    assert crc == zlib.crc32(t[16:16 + 16*n])
    img = BONSReader(IMAGE)
    for i, (chan, base, data) in enumerate(transfers):
        e = struct.unpack("<4I", t[16 + 16*i:32 + 16*i])
        assert e[:2] == (chan, base) and e[2] == (len(data) + 511)//512
        assert raw[e[3]*512:e[3]*512 + len(data)] == data
    assert transfers[1][2] == bytes(img.channel_payload(0))
    assert transfers[0] == (0, c["weight_base"] - 0x100000, bytes(img.data[:img.header["payload_offset"]]))
    if tok:
        assert transfers[-1][:2] == (0, c["weight_base"] - 0x1000000)


def test_layout_v3(tmp_path):
    c, tok, transfers = _transfers()
    out = str(tmp_path / "ssd.img")
    table, size = sdimage.write_image(out, transfers, version=3, label="tiny-hybrid")
    raw = open(out, "rb").read()
    assert len(raw) == size
    t = raw[sdimage.TABLE_AT*512:(sdimage.TABLE_AT + sdimage.V3_TABLE_SECTORS)*512]
    magic, ver, n, crc = struct.unpack("<4I", t[:16])
    assert (magic, ver, n) == (sdimage.MAGIC, 3, len(transfers))      # small transfers: no split
    assert t[16:48] == b"tiny-hybrid" + bytes(32 - 11)
    assert crc == zlib.crc32(t[48:48 + 24*n])
    first_data = sdimage.TABLE_AT + sdimage.V3_TABLE_SECTORS
    for i, (chan, base, data) in enumerate(transfers):
        e = struct.unpack("<6I", t[48 + 24*i:72 + 24*i])
        sector = e[3] | e[5] << 32
        assert e[:2] == (chan, base) and e[2] == (len(data) + 511)//512
        assert sector >= first_data
        padded = data + bytes(-len(data) % 512)
        assert raw[sector*512:sector*512 + len(padded)] == padded
        assert e[4] == zlib.crc32(padded)                                # the firmware's read-stream CRC
    assert table[0][3] == first_data


def test_split_and_wide_sectors():
    # A channel payload beyond one loader transfer spans two entries (the second at base + 512 MB),
    # and a slot far into a big SSD has 48-bit sectors.
    big = bytes(range(256))*4*1024                         # 1 MB of pattern
    small = sdimage.MAX_BLOCKS                             # the real limit...
    try:
        sdimage.MAX_BLOCKS = 1024                          # ... shrunk to 512 KB for the test
        pieces = sdimage.split_transfers([(3, 0x1000, big)], max_blocks=sdimage.MAX_BLOCKS)
        assert [(c, b, len(d)) for c, b, d in pieces] == [(3, 0x1000, 512*1024), (3, 0x1000 + 512*1024, 512*1024)]
        slot_sectors = 1 << 25
        table_sector, table, ents = sdimage.slot_layout([(3, 0x1000, big)], version=3, slot=200,
                                                         slot_sectors=slot_sectors, label="x")
        assert table_sector == sdimage.TABLE_AT + 200*slot_sectors > 1 << 32
        assert [e[:4] for e in ents] == [(3, 0x1000, 1024, table_sector + 4), (3, 0x1000 + 512*1024, 1024, table_sector + 4 + 1024)]
        e0 = struct.unpack("<6I", table[48:72])
        assert e0[3] | e0[5] << 32 == table_sector + 4 and e0[5] != 0
        assert e0[4] == zlib.crc32(big[:512*1024])
    finally:
        sdimage.MAX_BLOCKS = small
    b = sdimage.boot_sector(5)
    assert len(b) == 512 and struct.unpack("<3I", b[:12]) == (sdimage.BOOT_MAGIC, 5, ~5 & 0xffffffff)

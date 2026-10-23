#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/host/sdimage.py: the card layout the firmware's sdboot reads (firmware/sdload.h), on
the tiny hybrid image: the table's place, magic and CRC, each entry on its block, its bytes."""

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
    h = open(os.path.join(REPO, "firmware", "sdload.h")).read()
    assert int(re.search(r"#define SD_IMAGE_SECTOR\s+(\d+)", h).group(1)) == sdimage.TABLE_AT
    assert int(re.search(r"#define SD_IMAGE_MAGIC\s+(0x[0-9a-fA-F]+)u", h).group(1), 16) == sdimage.MAGIC
    assert int(re.search(r"#define SD_IMAGE_MAX_ENTRIES\s+(\d+)", h).group(1)) == sdimage.MAX_ENTRIES


def test_layout(tmp_path):
    if not os.path.exists(IMAGE):
        pytest.skip("no tiny image")
    c = dict(weight_base=0x1400_0000, stride=0x0B00_0000, adds_offset=True, n_channels=4)
    tok = TOKENIZER if os.path.exists(TOKENIZER) else None
    transfers = sdimage.entries(IMAGE, tok, c)
    out = str(tmp_path / "card.img")
    table, size = sdimage.write_image(out, transfers)
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

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/sd_upload.py CRC32Byte: zlib.crc32 of a byte stream with gaps, then a reset."""

import random
import zlib

from migen import *
from migen.sim import run_simulation

from gateware.memory.sd_upload import CRC32Byte


def test_crc32_matches_zlib():
    rng = random.Random(7)
    data = bytes(rng.randrange(256) for _ in range(300))
    dut = CRC32Byte()
    got = {}

    def gen():
        for b in data:
            yield dut.data.eq(b)
            yield dut.ce.eq(1)
            yield
            if rng.random() < 0.3:                      # Idle cycles between bytes.
                yield dut.ce.eq(0)
                yield
        yield dut.ce.eq(0)
        yield
        got["all"] = (yield dut.value)
        yield dut.reset.eq(1)
        yield
        yield dut.reset.eq(0)
        for b in b"TriXium":
            yield dut.data.eq(b)
            yield dut.ce.eq(1)
            yield
        yield dut.ce.eq(0)
        yield
        got["after_reset"] = (yield dut.value)

    run_simulation(dut, gen())
    assert got["all"] == zlib.crc32(data)
    assert got["after_reset"] == zlib.crc32(b"TriXium")

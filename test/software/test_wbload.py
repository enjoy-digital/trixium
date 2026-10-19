#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/host/wbload.py against a mock of LiteX's RemoteClient: the loader's CSR sequence
per transfer, the beats rebuilt from the pokes equal to the image's bytes at the PCIe loader's
placement (header+table at the staging area, every channel at its window)."""

import os
import types

import numpy as np
import pytest

from model.bons import BONSReader
from software.host import wbload

IMAGE = "models/tiny/bonsai-tiny-hybrid-1k-t5-n2.img"


class _Reg:
    def __init__(self, bus, name):
        self.bus, self.name = bus, name
    def write(self, v):
        self.bus.log.append((self.name, int(v)))
        if self.name == "trixium_loader_poke":
            self.bus.pokes.append(int(v))
        elif self.name == "trixium_loader_start":
            self.bus.on_start()
    def read(self):
        return 1 if self.name == "trixium_loader_status" else 0   # done, FIFO never full


class MockBus:
    """Records the CSR writes; a start arms (chan, base, length), pokes land in that transfer."""
    def __init__(self, consts):
        self.constants = types.SimpleNamespace(d=consts)
        self.regs = types.SimpleNamespace(**{n: _Reg(self, n) for n in (
            "trixium_loader_src_sel", "trixium_loader_chan", "trixium_loader_base",
            "trixium_loader_length", "trixium_loader_resync", "trixium_loader_start",
            "trixium_loader_poke", "trixium_loader_status")})
        self.log, self.pokes, self.transfers = [], [], []
    def on_start(self):
        armed = {k: v for k, v in self.log[-6:]}
        assert armed["trixium_loader_src_sel"] == 1 and armed["trixium_loader_resync"] == 0
        self.transfers.append(dict(chan=armed["trixium_loader_chan"], base=armed["trixium_loader_base"],
                                   length=armed["trixium_loader_length"], first_poke=len(self.pokes)))
    def memory(self):
        """{(chan, base): bytes} of every transfer, rebuilt from its pokes."""
        out = {}
        for i, t in enumerate(self.transfers):
            end = self.transfers[i + 1]["first_poke"] if i + 1 < len(self.transfers) else len(self.pokes)
            words = np.array(self.pokes[t["first_poke"]:end], dtype=np.uint32)
            assert len(words) == 8*t["length"], (t, len(words))
            out[(t["chan"], t["base"])] = words.tobytes()
        return out


@pytest.mark.skipif(not os.path.exists(IMAGE), reason="tiny image not generated (cli models --tiny)")
@pytest.mark.parametrize("adds_offset", [0, 1])
def test_upload_places_header_and_channels(adds_offset):
    weight_base, stride = 0x00C0_0000, 0x0080_0000
    bus = MockBus(dict(trixium_chan_weight_base=weight_base, trixium_chan_stride=stride,
                       trixium_loader_adds_chan_offset=adds_offset, trixium_n_channels=2))
    n = wbload.upload(bus, IMAGE)
    img = BONSReader(IMAGE); h = img.header
    mem = bus.memory()
    # Header + table at the staging area, padded to a beat.
    meta = bytes(img.data[:h["payload_offset"]])
    got  = mem[(0, weight_base - wbload.HDR_STAGING_BELOW)]
    assert got[:len(meta)] == meta and not any(got[len(meta):])
    # Every channel's payload at its window (the loader adds the offset itself or the host does).
    for c in range(h["n_channels"]):
        base = weight_base if adds_offset else weight_base + c*stride
        assert mem[(c, base)] == bytes(img.channel_payload(c))
    assert n == sum(len(v) for v in mem.values())
    assert len(bus.transfers) == 1 + h["n_channels"]


def test_chunks_at_128mb():
    bus = MockBus(dict(trixium_chan_weight_base=0, trixium_chan_stride=0))
    data = bytes(range(256))*(wbload.CHUNK_BEATS*wbload.BEAT_BYTES//256) + b"\x55"*64
    beats = wbload.transfer(bus, 3, 0x1000, data)
    assert beats == wbload.CHUNK_BEATS + 2
    assert [(t["chan"], t["base"], t["length"]) for t in bus.transfers] == \
        [(3, 0x1000, wbload.CHUNK_BEATS), (3, 0x1000 + wbload.CHUNK_BEATS*32, 2)]
    assert b"".join(bus.memory()[k] for k in sorted(bus.memory())) == data


def test_channel_count_mismatch_refused():
    bus = MockBus(dict(trixium_chan_weight_base=0, trixium_chan_stride=0, trixium_n_channels=4))
    with pytest.raises(ValueError):
        wbload.upload(bus, IMAGE)

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Loader stream resynchronisation.

The host->device DMA cannot be flow-controlled: the driver programs every descriptor and then
enables the reader, so the card is fed the ring's contents whether or not software has written
them. A transfer armed on the wrong beat writes plausible rubbish to HBM - that is how the
staged BONS header and the first megabyte of channel 0 were lost on hardware. Armed with
`sync`, the loader discards beats until it sees SYNC_MAGIC and takes the payload from the beat
after it, so whatever the DMA invents beforehand is discarded rather than written.
"""

import random

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium.memory.hbm_model import HBMChannelModel
from trixium.memory.loader import Loader, SYNC_MAGIC

MEM_SIZE = 64*1024


class _DUT(Module):
    # data_width=128 matches the SoC (the DMA sink is 128-bit). Testing at 256 hid a real bug:
    # the CSR poke path used to feed the 128->256 upconverter, which duplicated every word
    # across both halves of the beat on hardware while a 256-wide DUT saw no converter at all.
    def __init__(self, n=2, data_width=128):
        axis = []
        self.models = []
        for i in range(n):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            axis.append(axi)
            model = HBMChannelModel(axi, size=MEM_SIZE, latency=4)
            self.submodules += model
            self.models.append(model)
        self.loader = Loader(axis, data_width=data_width)
        self.submodules += self.loader


def _run(garbage_beats, payload, chan=0, addr=0x400, use_sync=True, markers=1):
    """Feed junk, then `markers` marker beats, then `payload`; return what landed in HBM."""
    dut = _DUT(data_width=256)
    rng = random.Random(1234)

    def driver():
        yield dut.loader.chan.storage.eq(chan)
        yield dut.loader.base.storage.eq(addr)
        yield dut.loader.length.storage.eq(len(payload))
        yield dut.loader.resync.storage.eq(1 if use_sync else 0)
        yield
        yield dut.loader.start.storage.eq(1)
        yield dut.loader.start.re.eq(1)
        yield
        yield dut.loader.start.re.eq(0)
        yield

        stream_beats = [rng.getrandbits(256) for _ in range(garbage_beats)]
        if use_sync:
            stream_beats += [SYNC_MAGIC]*markers
        stream_beats += payload

        for w in stream_beats:
            # The loader is deliberately not ready while idle (commit 458b070), so a stream
            # longer than the transfer - the no-sync case, where the garbage IS the transfer -
            # is never fully consumed: stop feeding once the loader reports done, and bound
            # the ready wait so a real stall fails instead of hanging the suite.
            if (yield dut.loader.status.fields.done):
                break
            yield dut.loader.sink.data.eq(w)
            yield dut.loader.sink.valid.eq(1)
            yield
            waited = 0
            while not (yield dut.loader.sink.ready):
                if (yield dut.loader.status.fields.done):
                    break
                waited += 1
                assert waited < 2000, "loader stalled mid-transfer"
                yield
        yield dut.loader.sink.valid.eq(0)

        for _ in range(4000):
            if (yield dut.loader.status.fields.done):
                break
            yield
        assert (yield dut.loader.status.fields.done), "loader did not finish"

    got = []

    def driver_then_read():
        yield from driver()
        for i in range(len(payload)):
            got.append((yield dut.models[chan].mem[addr//32 + i]))

    run_simulation(dut, [driver_then_read()])
    return got


def test_sync_skips_garbage_before_the_marker():
    payload = [0x1000 + i for i in range(8)]
    assert _run(garbage_beats=37, payload=payload) == payload


def test_sync_skips_a_run_of_markers():
    """A long marker prefix is how the host survives the dropped head of the stream."""
    payload = [0x4000 + i for i in range(6)]
    assert _run(garbage_beats=11, payload=payload, markers=64) == payload


def test_sync_with_no_garbage():
    payload = [0x2000 + i for i in range(4)]
    assert _run(garbage_beats=0, payload=payload) == payload


def test_without_sync_the_garbage_is_written():
    """The failure mode sync exists to prevent, pinned so the flag cannot silently regress."""
    payload = [0x3000 + i for i in range(4)]
    assert _run(garbage_beats=5, payload=payload, use_sync=False) != payload


# CSR upload path ------------------------------------------------------------------------------
#
# The host poking words through BAR0 instead of streaming over DMA. Slower - a posted PCIe
# write per 32 bits - but nothing can be dropped, reordered or invented on the way, so it needs
# neither the resync marker nor any pacing.

def test_csr_poke_upload():
    dut = _DUT()
    payload = [0x11110000 + i for i in range(16)]   # 16 x 256-bit beats, as 32-bit words
    words = []
    for beat in payload:
        for w in range(8):
            words.append((beat + w) & 0xFFFFFFFF)
    addr, chan = 0x800, 1
    got = []

    def driver():
        yield dut.loader.src_sel.storage.eq(1)
        yield dut.loader.chan.storage.eq(chan)
        yield dut.loader.base.storage.eq(addr)
        yield dut.loader.length.storage.eq(len(payload))
        yield dut.loader.resync.storage.eq(0)
        yield
        yield dut.loader.start.storage.eq(1)
        yield dut.loader.start.re.eq(1)
        yield
        yield dut.loader.start.re.eq(0)
        yield

        for w in words:
            while (yield dut.loader.status.fields.poke_full):
                yield
            yield dut.loader.poke.storage.eq(w)
            yield dut.loader.poke.re.eq(1)
            yield
            yield dut.loader.poke.re.eq(0)
            yield

        for _ in range(8000):
            if (yield dut.loader.status.fields.done):
                break
            yield
        assert (yield dut.loader.status.fields.done), "CSR upload did not finish"
        for i in range(len(payload)):
            got.append((yield dut.models[chan].mem[addr//32 + i]))

    run_simulation(dut, [driver()])
    expect = []
    for beat in payload:
        v = 0
        for w in range(8):
            v |= ((beat + w) & 0xFFFFFFFF) << (32*w)
        expect.append(v)
    assert got == expect, "CSR-poked payload landed wrong"

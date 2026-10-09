#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/loader.py: the Loader's upload, STORE and stream resynchronisation paths."""

import random

import numpy as np
import pytest

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from gateware.memory.loader import Loader, SYNC_MAGIC
from gateware.memory.hbm_model import HBMChannelModel

# Upload -------------------------------------------------------------------------------------------
# Host stream -> HBM.

def test_loader():
    class DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(2)]
            self.submodules.loader = Loader(axis, data_width=128)
            self.submodules.model0 = HBMChannelModel(axis[0], size=8*1024, latency=4)
            self.submodules.model1 = HBMChannelModel(axis[1], size=8*1024, latency=4)

    dut    = DUT()
    rng    = np.random.default_rng(31)
    nbeats = 6
    data   = rng.integers(0, 256, nbeats*32, dtype=np.uint8).tobytes()
    words128 = [int.from_bytes(data[i*16:(i+1)*16], "little") for i in range(nbeats*2)]
    base   = 32*5

    def gen():
        yield dut.loader.chan.storage.eq(1)
        yield dut.loader.base.storage.eq(base)
        yield dut.loader.length.storage.eq(nbeats)
        yield dut.loader.start.storage.eq(1)
        yield dut.loader.start.re.eq(1)
        yield
        yield dut.loader.start.re.eq(0)
        yield
        for w in words128:
            yield dut.loader.sink.data.eq(w)
            yield dut.loader.sink.valid.eq(1)
            yield
            while not (yield dut.loader.sink.ready):
                yield
        yield dut.loader.sink.valid.eq(0)
        for _ in range(500):
            if (yield dut.loader.status.fields.done):
                break
            yield
        else:
            raise AssertionError("loader timeout")
        for i in range(nbeats):
            got    = (yield dut.model1.mem[base//32 + i])
            expect = int.from_bytes(data[32*i:32*(i+1)], "little")
            assert got == expect, f"beat {i}"

    run_simulation(dut, [gen()])


# Store --------------------------------------------------------------------------------------------
# Loader STORE path (scratchpad -> HBM, the KV-cache append).
#
# On hardware a KV store wedges the engine: it sits in ST_DRAIN waiting for `store_busy` to
# drop, which means the loader is stuck in EST_DRAIN waiting for the AXI writer's B responses.
# The descriptor sequencer then stalls partway through a token (`SEQSTALL dcount=29/84 eng=01`)
# while matvec, fetch and the vector unit are all fine.
#
# The token gate runs the HBM model at latency=16 with no R stalls, which is nothing like the
# real part, so this exercises short stores against a slow, stalling channel - the shape of the
# traffic the KV append actually generates.

MEM_SIZE = 64*1024


class _StoreDUT(Module):
    def __init__(self, n=2, latency=100, stall_rate=8):
        axis = []
        self.models = []
        for i in range(n):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            axis.append(axi)
            model = HBMChannelModel(axi, size=MEM_SIZE, latency=latency,
                                          stall_rate=stall_rate)
            self.submodules += model
            self.models.append(model)
        self.loader = Loader(axis, data_width=128)
        self.submodules += self.loader


def _store(nbeats, chan=0, addr=0x400, latency=100, stall_rate=8, gap=0):
    """Drive one STORE of `nbeats`; return (completed, memory contents)."""
    dut = _StoreDUT(latency=latency, stall_rate=stall_rate)
    payload = [0x5000 + i for i in range(nbeats)]
    got = []
    done = []

    def driver():
        yield dut.loader.store_cmd.chan.eq(chan)
        yield dut.loader.store_cmd.addr.eq(addr)
        yield dut.loader.store_cmd.len.eq(nbeats)
        yield dut.loader.store_cmd.valid.eq(1)
        for _ in range(200):
            if (yield dut.loader.store_cmd.ready):
                break
            yield
        yield
        yield dut.loader.store_cmd.valid.eq(0)

        for w in payload:
            # `gap` models the scratchpad handing beats over with bubbles between them.
            for _ in range(gap):
                yield dut.loader.store_data.valid.eq(0)
                yield
            yield dut.loader.store_data.data.eq(w)
            yield dut.loader.store_data.valid.eq(1)
            yield
            while not (yield dut.loader.store_data.ready):
                yield
        yield dut.loader.store_data.valid.eq(0)

        for _ in range(20000):
            if not (yield dut.loader.store_busy):
                done.append(True)
                break
            yield
        for i in range(nbeats):
            got.append((yield dut.models[chan].mem[addr//32 + i]))

    run_simulation(dut, [driver()])
    return bool(done), got, payload


@pytest.mark.parametrize("nbeats", [1, 5, 16, 17, 64])
def test_store_completes_under_slow_hbm(nbeats):
    """A KV append is a handful of beats; the wedged op on hardware was 5."""
    ok, got, want = _store(nbeats, latency=100, stall_rate=8)
    assert ok, f"store of {nbeats} beats never drained (store_busy stuck high)"
    assert got == want, f"store of {nbeats} beats landed wrong"


@pytest.mark.parametrize("gap", [0, 1, 4])
def test_store_tolerates_bubbles_from_the_scratchpad(gap):
    """The engine feeds ST_RUN from the scratchpad, which does not deliver every cycle."""
    ok, got, want = _store(5, latency=100, stall_rate=8, gap=gap)
    assert ok, f"store with {gap}-cycle bubbles never drained"
    assert got == want


# Stream Resynchronisation -------------------------------------------------------------------------
# Loader stream resynchronisation.
#
# The host->device DMA cannot be flow-controlled: the driver programs every descriptor and then
# enables the reader, so the card is fed the ring's contents whether or not software has written
# them. A transfer armed on the wrong beat writes plausible rubbish to HBM - that is how the
# staged BONS header and the first megabyte of channel 0 were lost on hardware. Armed with
# `sync`, the loader discards beats until it sees SYNC_MAGIC and takes the payload from the beat
# after it, so whatever the DMA invents beforehand is discarded rather than written.

class _SyncDUT(Module):
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
    dut = _SyncDUT(data_width=256)
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


# CSR upload path ----------------------------------------------------------------------------------
#
# The host poking words through BAR0 instead of streaming over DMA. Slower - a posted PCIe
# write per 32 bits - but nothing can be dropped, reordered or invented on the way, so it needs
# neither the resync marker nor any pacing.

def test_csr_poke_upload():
    dut = _SyncDUT()
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

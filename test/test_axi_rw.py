#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# AXIReader/Writer unit tests against HBMChannelModel (Migen simulation):
# exact data/last sequences under latency + R-stall jitter + consumer backpressure,
# streamer utilization on the clean model, write-then-readback integrity.

import random

import numpy as np
import pytest

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium.memory.axi import AXIReader
from trixium.memory.axi import AXIWriter
from trixium.memory.hbm_model import HBMChannelModel, words_from_bytes

MEM_SIZE = 16*1024 # 512 beats.

def _pattern_bytes(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, MEM_SIZE, dtype=np.uint8).tobytes()

class _ReaderDUT(Module):
    def __init__(self, latency, stall_rate, init_bytes):
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.reader = AXIReader(axi, burst_beats=16, fifo_depth=64)
        self.submodules.model  = HBMChannelModel(axi, size=MEM_SIZE,
            init=words_from_bytes(init_bytes), latency=latency, stall_rate=stall_rate)

class _WriterDUT(Module):
    def __init__(self):
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.writer = AXIWriter(axi, burst_beats=16)
        self.submodules.model  = HBMChannelModel(axi, size=MEM_SIZE, latency=4)

def _expected_beats(data, addr, nbeats):
    return [int.from_bytes(data[addr + 32*i:addr + 32*(i+1)], "little") for i in range(nbeats)]

def _run_reader_case(latency, stall_rate, backpressure, seed):
    data = _pattern_bytes(seed)
    dut  = _ReaderDUT(latency, stall_rate, data)
    rng  = random.Random(seed)
    cmds = []
    for _ in range(6):
        nbeats = rng.randrange(1, 70)
        addr   = 32*rng.randrange(0, MEM_SIZE//32 - nbeats)
        cmds.append((addr, nbeats))

    received = []

    def driver():
        for addr, nbeats in cmds:
            yield dut.reader.sink.addr.eq(addr)
            yield dut.reader.sink.len.eq(nbeats)
            yield dut.reader.sink.valid.eq(1)
            yield
            while not (yield dut.reader.sink.ready):
                yield
            yield dut.reader.sink.valid.eq(0)
            # Wait for this command's last beat before issuing the next (checks `last`).
            while True:
                if (yield dut.reader.source.valid) and (yield dut.reader.source.ready) \
                   and (yield dut.reader.source.last):
                    break
                yield
            yield

    def consumer():
        bp_rng = random.Random(seed + 1000)
        yield dut.reader.source.ready.eq(1)
        while len(received) < sum(n for _, n in cmds):
            if backpressure:
                yield dut.reader.source.ready.eq(bp_rng.randrange(2))
            if (yield dut.reader.source.valid) and (yield dut.reader.source.ready):
                received.append(((yield dut.reader.source.data), (yield dut.reader.source.last)))
            yield

    run_simulation(dut, [driver(), consumer()])

    expected = []
    for addr, nbeats in cmds:
        beats = _expected_beats(data, addr, nbeats)
        expected += [(b, 1 if i == nbeats - 1 else 0) for i, b in enumerate(beats)]
    assert received == expected, f"mismatch latency={latency} stall={stall_rate} bp={backpressure}"

def test_reader_clean():
    _run_reader_case(latency=4, stall_rate=0, backpressure=False, seed=1)

def test_reader_latency_jitter():
    _run_reader_case(latency=24, stall_rate=6, backpressure=False, seed=2)

def test_reader_full_stress():
    _run_reader_case(latency=16, stall_rate=9, backpressure=True, seed=3)

def test_reader_utilization():
    # One long linear command on the clean model: >= 90% beats/cycle in steady state.
    data = _pattern_bytes(7)
    dut  = _ReaderDUT(8, 0, data)
    nbeats = 256
    stats  = {}

    def driver():
        yield dut.reader.sink.addr.eq(0)
        yield dut.reader.sink.len.eq(nbeats)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)

    def consumer():
        yield dut.reader.source.ready.eq(1)
        got   = 0
        cycle = 0
        first = None
        while got < nbeats:
            if (yield dut.reader.source.valid):
                if first is None:
                    first = cycle
                got += 1
            cycle += 1
            yield
        stats["cycles"] = cycle - first

    run_simulation(dut, [driver(), consumer()])
    utilization = nbeats/stats["cycles"]
    assert utilization >= 0.90, f"streamer utilization {utilization:.2f}"

def test_writer_readback():
    dut = _WriterDUT()
    rng = np.random.default_rng(11)
    payload = rng.integers(0, 256, 100*32, dtype=np.uint8).tobytes() # 100 beats.
    words   = words_from_bytes(payload)
    addr    = 32*17 # Unaligned vs bursts, exercises 4KB-safe splitting.

    def driver():
        yield dut.writer.sink_cmd.addr.eq(addr)
        yield dut.writer.sink_cmd.len.eq(len(words))
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        for w in words:
            yield dut.writer.sink_data.data.eq(w)
            yield dut.writer.sink_data.valid.eq(1)
            yield
            while not (yield dut.writer.sink_data.ready):
                yield
        yield dut.writer.sink_data.valid.eq(0)
        # Wait for completion.
        for _ in range(200):
            if (yield dut.writer.idle):
                break
            yield
        assert (yield dut.writer.idle)
        # Readback directly from the model memory.
        for i, w in enumerate(words):
            got = (yield dut.model.mem[addr//32 + i])
            assert got == w, f"beat {i}"

    run_simulation(dut, [driver()])


# AXI3 burst-length cap ----------------------------------------------------------------------------
#
# The HBM controller's ports are AXI3: AXI_nn_AxLEN is 4 bits (see ip/hbm*/hbm_*_stub.v), so a
# burst can never exceed 16 beats. LiteX's AXIInterface carries an 8-bit `len` and the upper
# bits are dropped where it meets the IP, which is silent and ugly: reads stall forever waiting
# for beats the IP will not send, and writes place WLAST late and corrupt memory (on hardware:
# ENGSTALL, 32 of 128 beats delivered), so the limit is pinned here.

@pytest.mark.parametrize("cls", [AXIReader, AXIWriter])
def test_burst_beats_capped_at_axi3_limit(cls):
    axi = AXIInterface(data_width=256, address_width=33, id_width=6)
    with pytest.raises(AssertionError):
        cls(axi, burst_beats=64)

@pytest.mark.parametrize("cls", [AXIReader, AXIWriter])
def test_burst_beats_16_is_accepted(cls):
    axi = AXIInterface(data_width=256, address_width=33, id_width=6)
    cls(axi, burst_beats=16)

def test_hbm_model_flags_axlen_overflow():
    """The model truncates AxLEN like the IP does, and latches the protocol violation."""
    axi = AXIInterface(data_width=256, address_width=33, id_width=6)
    dut = Module()
    dut.submodules.model = model = HBMChannelModel(axi, size=MEM_SIZE, latency=1)

    def driver():
        yield axi.ar.addr.eq(0)
        yield axi.ar.len.eq(63)          # 64-beat burst: illegal on an AXI3 port.
        yield axi.ar.valid.eq(1)
        for _ in range(8):
            yield
        yield axi.ar.valid.eq(0)
        for _ in range(4):
            yield
        assert (yield model.axlen_overflow), "model accepted an out-of-range AxLEN"

    run_simulation(dut, [driver()])

# Multi-ARID reads ---------------------------------------------------------------------------------
#
# The Xilinx HBM controller strictly serializes same-ARID reads: measured on the FK33, a
# single port streamed 33.7 cycles/beat (one 16-beat burst per round trip) while the same
# RTL sustained 2.35 cycles/beat in the full-SoC sim against the idealized model. The
# reorder_ids model mode reproduces that behavior (same-ID serialized with a latency gap,
# different IDs overlapped, response order shuffled across IDs); these tests pin both the
# correctness of the reorder drain and the performance claim, so the pathology can never
# ship silently again.

class _ReaderIDDUT(Module):
    def __init__(self, latency, stall_rate, init_bytes, n_ids, reorder_ids):
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.reader = AXIReader(axi, burst_beats=16, fifo_depth=64,
                                                 n_ids=n_ids)
        self.submodules.model  = HBMChannelModel(axi, size=MEM_SIZE,
            init=words_from_bytes(init_bytes), latency=latency, stall_rate=stall_rate,
            reorder_ids=reorder_ids)

def _run_id_case(latency, stall_rate, n_ids, reorder_ids, seed, backpressure=False):
    data = _pattern_bytes(seed)
    dut  = _ReaderIDDUT(latency, stall_rate, data, n_ids, reorder_ids)
    rng  = random.Random(seed)
    cmds = []
    for _ in range(6):
        nbeats = rng.randrange(1, 70)
        addr   = 32*rng.randrange(0, MEM_SIZE//32 - nbeats)
        cmds.append((addr, nbeats))

    received = []
    cycles   = [0]

    def driver():
        for addr, nbeats in cmds:
            yield dut.reader.sink.addr.eq(addr)
            yield dut.reader.sink.len.eq(nbeats)
            yield dut.reader.sink.valid.eq(1)
            yield
            while not (yield dut.reader.sink.ready):
                yield
            yield dut.reader.sink.valid.eq(0)
            got = 0
            while got < nbeats:
                if backpressure and (cycles[0] % 7 < 2):
                    yield dut.reader.source.ready.eq(0)
                else:
                    yield dut.reader.source.ready.eq(1)
                yield
                cycles[0] += 1
                if (yield dut.reader.source.valid) and (yield dut.reader.source.ready):
                    d = (yield dut.reader.source.data)
                    l = (yield dut.reader.source.last)
                    received.append((d, l))
                    got += 1
                if cycles[0] > 200000:
                    raise AssertionError("timeout")
            yield dut.reader.source.ready.eq(0)

    run_simulation(dut, [driver()])
    exp = []
    for addr, nbeats in cmds:
        beats = _expected_beats(data, addr, nbeats)
        exp += [(b, 1 if i == nbeats - 1 else 0) for i, b in enumerate(beats)]
    assert received == exp, "beat/last sequence mismatch"
    return cycles[0], sum(n for _, n in cmds)

def test_reader_multi_id_correct_under_reorder():
    # Byte-exact and last-exact through cross-ID response shuffling, at several latencies
    # and with stall + backpressure jitter.
    for latency, stall, bp, seed in [(8, 0, False, 1), (40, 0, False, 2),
                                     (40, 4, True, 3), (64, 2, False, 4)]:
        _run_id_case(latency, stall, 4, 4, seed, bp)

def test_reader_multi_id_correct_on_inorder_model():
    # The same reader against the legacy in-order model (reorder off): drain order and
    # tags must be identical - covers HBM controllers that happen not to interleave.
    _run_id_case(24, 2, 4, 0, 5)

def test_reader_single_id_slow_multi_id_fast_under_reorder():
    # The performance claim itself. Same command stream, same model: the single-ID reader
    # pays a full round trip per burst; four IDs must recover most of the line rate.
    c1, beats = _run_id_case(40, 0, 1, 4, 6)
    c4, _     = _run_id_case(40, 0, 4, 4, 6)
    r1, r4 = c1/beats, c4/beats
    assert r1 > 2.5, f"single-ID unexpectedly fast ({r1:.2f} cyc/beat) - model not serializing?"
    assert r4 < r1/2, f"multi-ID did not fix it: {r4:.2f} vs {r1:.2f} cyc/beat"

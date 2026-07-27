#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# BonsaiAXIReader/Writer unit tests against BonsaiHBMChannelModel (Migen simulation):
# exact data/last sequences under latency + R-stall jitter + consumer backpressure,
# streamer utilization on the clean model, write-then-readback integrity.

import random

import numpy as np
from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.axi_writer import BonsaiAXIWriter
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes

MEM_SIZE = 16*1024 # 512 beats.

def _pattern_bytes(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, MEM_SIZE, dtype=np.uint8).tobytes()

class _ReaderDUT(Module):
    def __init__(self, latency, stall_rate, init_bytes):
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.reader = BonsaiAXIReader(axi, burst_beats=16, fifo_depth=64)
        self.submodules.model  = BonsaiHBMChannelModel(axi, size=MEM_SIZE,
            init=words_from_bytes(init_bytes), latency=latency, stall_rate=stall_rate)

class _WriterDUT(Module):
    def __init__(self):
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.writer = BonsaiAXIWriter(axi, burst_beats=16)
        self.submodules.model  = BonsaiHBMChannelModel(axi, size=MEM_SIZE, latency=4)

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

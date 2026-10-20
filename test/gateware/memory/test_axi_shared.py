#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/axi_shared.py: AXISharedPort - channel readers and a writer on one AXI port
(the single-controller boards), against HBMChannelModel: every beat back to its master in order,
the writes landed, and the reads interleaved (the port stays busy while one master waits),
where LiteX's AXIArbiter streams one master's bursts at a time."""

import random

import numpy as np
import pytest

from migen import *
from migen.sim import run_simulation

from litex.soc.interconnect.axi import AXIInterface, AXIArbiter

from gateware.memory.axi import AXIReader, AXIWriter
from gateware.memory.axi_shared import AXISharedPort, slave_id_width
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes

MEM_SIZE = 256*1024
REGION   = 32*1024        # Per master.


class _DUT(Module):
    def __init__(self, n_readers, init, latency, reorder_ids, arbiter="shared"):
        self.raxi = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(n_readers)]
        self.waxi = AXIInterface(data_width=256, address_width=33, id_width=6)
        masters = self.raxi + [self.waxi]
        self.readers = [AXIReader(a, burst_beats=16, fifo_depth=128, n_ids=4) for a in self.raxi]
        for i, r in enumerate(self.readers):
            setattr(self.submodules, f"reader{i}", r)
        self.submodules.writer = AXIWriter(self.waxi, burst_beats=16)
        if arbiter == "shared":
            self.slave = AXIInterface(data_width=256, address_width=33, id_width=slave_id_width(masters))
            self.submodules.port = AXISharedPort(masters, self.slave)
        else:
            self.slave = AXIInterface(data_width=256, address_width=33, id_width=6)
            self.submodules.port = AXIArbiter(masters, self.slave)
        self.submodules.model = HBMChannelModel(self.slave, size=MEM_SIZE, init=words_from_bytes(init),
                                                latency=latency, reorder_ids=reorder_ids)


def _run(n_readers, latency, reorder_ids, backpressure, seed, beats=128, arbiter="shared"):
    rng  = np.random.default_rng(seed)
    data = rng.integers(0, 256, MEM_SIZE, dtype=np.uint8).tobytes()
    dut  = _DUT(n_readers, data, latency, reorder_ids, arbiter)
    wbase = n_readers*REGION
    wdata = [int.from_bytes(rng.integers(0, 256, 32, dtype=np.uint8).tobytes(), "little") for _ in range(beats)]
    got   = [[] for _ in range(n_readers)]
    state = dict(first=None, last=None, wdone=False)

    def reader_cmd(i):
        r = dut.readers[i]
        yield r.sink.addr.eq(i*REGION)
        yield r.sink.len.eq(beats)
        yield r.sink.valid.eq(1)
        yield
        while not (yield r.sink.ready):
            yield
        yield r.sink.valid.eq(0)

    def reader_sink(i):
        r = dut.readers[i]
        prng = random.Random(seed*7 + i)
        yield r.source.ready.eq(1)
        while len(got[i]) < beats:
            if backpressure:
                yield r.source.ready.eq(prng.randrange(4) != 0)
            if (yield r.source.valid) and (yield r.source.ready):
                got[i].append((yield r.source.data))
            yield

    def writer():
        w = dut.writer
        yield w.sink_cmd.addr.eq(wbase)
        yield w.sink_cmd.len.eq(beats)
        yield w.sink_cmd.valid.eq(1)
        yield
        while not (yield w.sink_cmd.ready):
            yield
        yield w.sink_cmd.valid.eq(0)
        for d in wdata:
            yield w.sink_data.data.eq(d)
            yield w.sink_data.valid.eq(1)
            yield
            while not (yield w.sink_data.ready):
                yield
        yield w.sink_data.valid.eq(0)

    def monitor():
        cycle = 0
        while not all(len(g) == beats for g in got):
            assert cycle < 40*beats*n_readers + 4000, f"stalled: {[len(g) for g in got]} of {beats} beats"
            if (yield dut.slave.r.valid) and (yield dut.slave.r.ready):
                if state["first"] is None:
                    state["first"] = cycle
                state["last"] = cycle
            cycle += 1
            yield
        for _ in range(600):       # Let the writes drain.
            yield
        written = []
        for k in range(beats):
            written.append((yield dut.model.mem[wbase//32 + k]))
        state["written"] = written

    gens = [reader_cmd(i) for i in range(n_readers)] + [reader_sink(i) for i in range(n_readers)]
    gens += [writer(), monitor()]
    run_simulation(dut, gens)
    for i in range(n_readers):
        want = [int.from_bytes(data[i*REGION + 32*k:i*REGION + 32*(k + 1)], "little") for k in range(beats)]
        assert got[i] == want, f"reader {i}: {sum(a != b for a, b in zip(got[i], want))} beats wrong"
    assert state["written"] == wdata, "the writes did not land"
    return dut, wdata, wbase, (n_readers*beats)/(state["last"] - state["first"] + 1)


@pytest.mark.parametrize("latency,reorder_ids,backpressure", [(20, 0, False), (40, 4, False), (20, 4, True)])
def test_readers_and_writer_share_one_port(latency, reorder_ids, backpressure):
    dut, wdata, wbase, rate = _run(3, latency, reorder_ids, backpressure, seed=latency + reorder_ids)
    print(f"latency {latency}, reorder {reorder_ids}, backpressure {backpressure}: {rate:.2f} read beats per clock")
    if not backpressure and not reorder_ids:
        assert rate > 0.95, rate        # (the reordering model caps itself: same-ID bursts serialised)


def test_interleaving_beats_the_lock_step_arbiter():
    """Same traffic, LiteX's AXIArbiter (one master until its requests complete) vs the shared port."""
    _, _, _, shared = _run(3, 40, 0, False, seed=5)
    _, _, _, locked = _run(3, 40, 0, False, seed=5, arbiter="litex")
    print(f"shared port {shared:.2f}, AXIArbiter {locked:.2f} read beats per clock")
    assert shared > locked + 0.1, (shared, locked)

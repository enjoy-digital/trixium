#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""LiteX's AXIArbiter in its interleaved mode (litex#2689, how the single-controller boards put
channel readers, writers and the CPU on one AXI port) and gateware/memory/axi_shared.py's
AXIIDSquash, against HBMChannelModel: every beat back to its master in order, the writes landed,
and the reads interleaved (the port stays busy while one master waits), where the default
AXIArbiter streams one master's bursts at a time."""

import random

import numpy as np
import pytest

from migen import *
from migen.sim import run_simulation

from litex.soc.interconnect.axi import AXIInterface, AXIArbiter
from litex.soc.interconnect.axi import axi_interleaved_id_width as slave_id_width

from gateware.memory.axi import AXIReader, AXIWriter
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
            self.submodules.port = AXIArbiter(masters, self.slave, interleave=True)
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


# The Ti375 attachment: 256-bit engine ports merged two beats per 512-bit word (AXIWide), on one
# 512-bit AXI4 port through the interleaved AXIArbiter.

class _WideDUT(Module):
    def __init__(self, n_readers, init, latency):
        from gateware.memory.axi import AXIWide
        self.raxi = [AXIInterface(data_width=256, address_width=32, id_width=6) for _ in range(n_readers)]
        self.waxi = AXIInterface(data_width=256, address_width=32, id_width=6)
        self.readers = [AXIReader(a, burst_beats=16, fifo_depth=128, n_ids=4) for a in self.raxi]
        for i, r in enumerate(self.readers):
            setattr(self.submodules, f"reader{i}", r)
        self.submodules.writer = AXIWriter(self.waxi, burst_beats=16)
        wides = []
        for i, a in enumerate(self.raxi + [self.waxi]):
            w = AXIInterface(data_width=512, address_width=32, id_width=6)
            setattr(self.submodules, f"wide{i}", AXIWide(a, w))
            wides.append(w)
        self.slave = AXIInterface(data_width=512, address_width=32, id_width=slave_id_width(wides))
        self.submodules.port  = AXIArbiter(wides, self.slave, r_buffer=96, interleave=True)
        words = [int.from_bytes(init[i:i + 64], "little") for i in range(0, len(init), 64)]
        self.submodules.model = HBMChannelModel(self.slave, size=MEM_SIZE, init=words, latency=latency,
                                                data_width=512, axi3=False)


@pytest.mark.parametrize("offset_beats", [0, 1, 5])
def test_wide_ports_on_one_512bit_port(offset_beats):
    """Reads and a write at word-aligned and half-word offsets (a burst entered in the upper half
    of a 512-bit word: 9 words for 16 beats), all through AXIWide and the shared port."""
    n_readers, beats = 2, 48
    rng  = np.random.default_rng(offset_beats)
    data = rng.integers(0, 256, MEM_SIZE, dtype=np.uint8).tobytes()
    dut  = _WideDUT(n_readers, data, latency=24)
    wbase = n_readers*REGION + 32*offset_beats
    wdata = [int.from_bytes(rng.integers(0, 256, 32, dtype=np.uint8).tobytes(), "little") for _ in range(beats)]
    got, state = [[] for _ in range(n_readers)], {}

    def reader(i):
        r = dut.readers[i]
        yield r.sink.addr.eq(i*REGION + 32*offset_beats)
        yield r.sink.len.eq(beats)
        yield r.sink.valid.eq(1)
        yield
        while not (yield r.sink.ready):
            yield
        yield r.sink.valid.eq(0)
        yield r.source.ready.eq(1)
        for _ in range(20000):
            if (yield r.source.valid):
                got[i].append((yield r.source.data))
                if len(got[i]) == beats:
                    return
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
            for _ in range(5000):
                if (yield w.sink_data.ready):
                    break
                yield
        yield w.sink_data.valid.eq(0)
        for _ in range(3000):
            yield
        words = []
        for k in range(wbase//64, (wbase + 32*beats + 63)//64):
            words.append((yield dut.model.mem[k]))
        state["mem"] = b"".join(x.to_bytes(64, "little") for x in words)

    run_simulation(dut, [reader(i) for i in range(n_readers)] + [writer()])
    for i in range(n_readers):
        a = i*REGION + 32*offset_beats
        want = [int.from_bytes(data[a + 32*k:a + 32*(k + 1)], "little") for k in range(beats)]
        assert got[i] == want, f"reader {i}: {len(got[i])} beats, {sum(x != y for x, y in zip(got[i], want))} wrong"
    off = wbase % 64
    written = state["mem"][off:off + 32*beats]
    assert written == b"".join(d.to_bytes(32, "little") for d in wdata), "the merged writes did not land"
    # The other half of a partially written word is untouched.
    if off:
        assert state["mem"][:off] == data[wbase - off:wbase]


def test_write_data_of_one_master_waiting_on_another_does_not_deadlock():
    """Master A's AW comes first but its W data only once master B's burst has been accepted (the
    DeltaNet stream's write-back feeds every channel's writer in lockstep): W in AW order at the
    slave would deadlock; the port buffers each master's burst before issuing its AW."""
    a = AXIInterface(data_width=256, address_width=33, id_width=6)
    b = AXIInterface(data_width=256, address_width=33, id_width=6)
    slave = AXIInterface(data_width=256, address_width=33, id_width=slave_id_width([a, b]))
    dut = Module()
    dut.submodules.port  = AXIArbiter([a, b], slave, interleave=True)
    dut.submodules.model = HBMChannelModel(slave, size=MEM_SIZE, latency=8)
    state = {"b_done": False, "a_done": False}
    BEATS = 8

    def master(m, addr, base_word, wait_for=None, aw_delay=0, key=None):
        for _ in range(aw_delay):
            yield
        yield m.aw.addr.eq(addr); yield m.aw.len.eq(BEATS - 1); yield m.aw.size.eq(5)
        yield m.aw.burst.eq(1); yield m.aw.valid.eq(1)
        yield
        while not (yield m.aw.ready):
            yield
        yield m.aw.valid.eq(0)

    def wdata(m, base_word, wait_for=None, key=None):
        if wait_for:
            for _ in range(4000):
                if state[wait_for]:
                    break
                yield
        for k in range(BEATS):
            yield m.w.data.eq(base_word + k); yield m.w.strb.eq((1 << 32) - 1)
            yield m.w.last.eq(k == BEATS - 1); yield m.w.valid.eq(1)
            yield
            for _ in range(4000):
                if (yield m.w.ready):
                    break
                yield
        yield m.w.valid.eq(0)
        state[key] = True

    def bresp(m):
        yield m.b.ready.eq(1)

    def watch():
        for _ in range(6000):
            if state["a_done"] and state["b_done"]:
                break
            yield
        for _ in range(200):
            yield
        state["mem"] = []
        for k in range(2*BEATS):
            state["mem"].append((yield dut.model.mem[k]))

    run_simulation(dut, [master(a, 0, 0x100), master(b, 32*BEATS, 0x200, aw_delay=2),
                         wdata(a, 0x100, wait_for="b_done", key="a_done"), wdata(b, 0x200, key="b_done"),
                         bresp(a), bresp(b), watch()])
    assert state["b_done"] and state["a_done"], f"deadlock: {state}"
    assert state["mem"] == [0x100 + k for k in range(BEATS)] + [0x200 + k for k in range(BEATS)]


def test_id_squash_restores_the_ids():
    """AXIIDSquash: reads and writes under several IDs go out under ID 0 and come back with their
    own IDs, in request order."""
    from gateware.memory.axi_shared import AXIIDSquash
    m = AXIInterface(data_width=256, address_width=33, id_width=8)
    s = AXIInterface(data_width=256, address_width=33, id_width=1)
    rng = np.random.default_rng(4)
    data = rng.integers(0, 256, MEM_SIZE, dtype=np.uint8).tobytes()
    dut = Module()
    dut.submodules.sq = AXIIDSquash(m, s)
    dut.submodules.model = HBMChannelModel(s, size=MEM_SIZE, init=words_from_bytes(data), latency=6)
    ids = [0x5a, 0x03, 0xff, 0x10]
    got = []

    def ar():
        for k, i in enumerate(ids):
            yield m.ar.addr.eq(32*4*k); yield m.ar.len.eq(1); yield m.ar.id.eq(i); yield m.ar.valid.eq(1)
            yield
            while not (yield m.ar.ready):
                yield
        yield m.ar.valid.eq(0)

    def r():
        yield m.r.ready.eq(1)
        for _ in range(400):
            if (yield m.r.valid):
                got.append(((yield m.r.id), (yield m.r.data), (yield m.r.last)))
            yield

    run_simulation(dut, [ar(), r()])
    want = []
    for k, i in enumerate(ids):
        for b in range(2):
            a = 32*(4*k + b)
            want.append((i, int.from_bytes(data[a:a + 32], "little"), b == 1))
    assert got == want


@pytest.mark.parametrize("write_order", ["aw_first", "w_first", "aw_after_w"])
def test_writes_with_every_slave_write_order(write_order):
    """A slave may take W before AW, or wait for WVALID before AWREADY (the Efinix hard DDR
    controller: its first write deadlocked the port, which sent W only after the AW): two writers'
    bursts land, in order, whatever the slave's discipline."""
    a = AXIInterface(data_width=256, address_width=33, id_width=6)
    b = AXIInterface(data_width=256, address_width=33, id_width=6)
    slave = AXIInterface(data_width=256, address_width=33, id_width=slave_id_width([a, b]))
    dut = Module()
    dut.submodules.wa = AXIWriter(a, burst_beats=16)
    dut.submodules.wb = AXIWriter(b, burst_beats=16)
    dut.submodules.port = AXIArbiter([a, b], slave, interleave=True)
    dut.submodules.model = HBMChannelModel(slave, size=MEM_SIZE, latency=4, write_order=write_order)
    rng = np.random.default_rng(len(write_order))
    beats = 40
    data = {w: [int.from_bytes(rng.integers(0, 256, 32, dtype=np.uint8).tobytes(), "little") for _ in range(beats)]
            for w in ("a", "b")}
    bases = {"a": 0, "b": 32*1024}
    state = {}

    def writer(w, base, words):
        yield w.sink_cmd.addr.eq(base); yield w.sink_cmd.len.eq(beats); yield w.sink_cmd.valid.eq(1)
        yield
        while not (yield w.sink_cmd.ready):
            yield
        yield w.sink_cmd.valid.eq(0)
        for d in words:
            yield w.sink_data.data.eq(d); yield w.sink_data.valid.eq(1)
            yield
            for _ in range(3000):
                if (yield w.sink_data.ready):
                    break
                yield
            else:
                state["stall"] = True
                return
        yield w.sink_data.valid.eq(0)

    def check():
        for _ in range(4000):
            yield
        for k, base in bases.items():
            got = []
            for i in range(beats):
                got.append((yield dut.model.mem[base//32 + i]))
            state[k] = got

    run_simulation(dut, [writer(dut.wa, 0, data["a"]), writer(dut.wb, bases["b"], data["b"]), check()])
    assert "stall" not in state, "a writer stalled: the port deadlocked"
    assert state["a"] == data["a"] and state["b"] == data["b"]


# The read merge needs the AR order back (Ti375) ---------------------------------------------------
#
# AXIWide's read merge (AXIReadMerge) keeps each narrow burst's half/length in AR order and
# applies them to the R bursts as they come: right only if the slave answers in AR order. The
# Efinix hard controller may serve different IDs out of order, so a reader rotating 4 ARIDs got
# other bursts' halves - wrong weights, intermittently (the 2B's tokens changed run to run on the
# kit). With one ARID per reader (TriXiumConfig reader_n_ids=1, the Ti375's) AXI keeps the order.

def _merge_reorder_case(n_ids, seed=7):
    import random
    from gateware.memory.axi import AXIReader, AXIWide
    from gateware.memory.hbm_model import HBMChannelModel
    rng  = random.Random(seed)
    size = 64*1024
    data = bytes(rng.getrandbits(8) for _ in range(size))
    narrow = AXIInterface(data_width=256, address_width=32, id_width=3)
    wide   = AXIInterface(data_width=512, address_width=32, id_width=3)
    dut = Module()
    dut.submodules.reader = reader = AXIReader(narrow, burst_beats=16, fifo_depth=128, n_ids=n_ids)
    dut.submodules.wide   = AXIWide(narrow, wide)
    words = [int.from_bytes(data[i:i+64], "little") for i in range(0, size, 64)]
    dut.submodules.model  = HBMChannelModel(wide, size=size, init=words, latency=12,
                                            data_width=512, axi3=False, reorder_ids=4)
    cmds = [(32*rng.randrange(0, size//32 - 80), rng.randrange(17, 80)) for _ in range(5)]
    got  = []

    def driver():
        for addr, nbeats in cmds:
            yield reader.sink.addr.eq(addr)
            yield reader.sink.len.eq(nbeats)
            yield reader.sink.valid.eq(1)
            yield
            while not (yield reader.sink.ready):
                yield
            yield reader.sink.valid.eq(0)
            yield reader.source.ready.eq(1)
            n, t = 0, 0
            while n < nbeats and t < 20000:
                yield
                t += 1
                if (yield reader.source.valid):
                    got.append((yield reader.source.data))
                    n += 1
            yield reader.source.ready.eq(0)

    run_simulation(dut, [driver()])
    want = [int.from_bytes(data[a + 32*i:a + 32*i + 32], "little") for a, n in cmds for i in range(n)]
    return got == want

def test_read_merge_single_id_exact_under_reorder():
    assert _merge_reorder_case(n_ids=1)

def test_read_merge_multi_id_breaks_under_reorder():
    # Pins the reason for reader_n_ids=1 on the Ti375: drop this test if the merge learns IDs.
    assert not _merge_reorder_case(n_ids=4)

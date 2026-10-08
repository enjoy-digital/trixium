"""AXIResetShim: the HBM port's protocol survives a master reset mid-transaction.

A reader and a writer behind the shim, in a ResetInserter domain; the HBM model can freeze
its read server. Scenarios: (1) the port stops answering, the master is reset, the port
answers later - the late beats are discarded and the next command's data is exact; (2) a
write burst cut in half by the reset is completed with wstrb=0 (memory untouched past the
beats that landed) and later writes are exact; (3) a burst that never returns leaves the
shim busy and raises `stuck`."""

import random

from migen import *
from migen.sim import run_simulation

from litex.soc.interconnect.axi import AXIInterface

from trixium.memory.axi import AXIResetShim
from trixium.memory.axi import BonsaiAXIReader
from trixium.memory.axi import BonsaiAXIWriter
from trixium.memory.hbm_model import BonsaiHBMChannelModel, words_from_bytes


class _DUT(Module):
    def __init__(self, init):
        port = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.submodules.model = BonsaiHBMChannelModel(port, size=16*1024, init=init, latency=4)
        self.submodules.shim  = AXIResetShim(port, stuck_cycles=2000)
        self.rst = Signal()
        self.submodules.reader = ResetInserter()(BonsaiAXIReader(self.shim.master, fifo_depth=64))
        self.submodules.writer = ResetInserter()(BonsaiAXIWriter(self.shim.master))
        self.comb += [self.reader.reset.eq(self.rst), self.writer.reset.eq(self.rst)]


def _read(dut, addr, beats, timeout=5000):
    got = []
    yield dut.reader.sink.addr.eq(addr)
    yield dut.reader.sink.len.eq(beats)
    yield dut.reader.sink.valid.eq(1)
    yield
    while not (yield dut.reader.sink.ready):
        yield
    yield dut.reader.sink.valid.eq(0)
    yield dut.reader.source.ready.eq(1)
    for _ in range(timeout):
        if (yield dut.reader.source.valid):
            got.append((yield dut.reader.source.data))
            if len(got) == beats:
                break
        yield
    yield dut.reader.source.ready.eq(0)
    yield
    return got


def _soft_reset(dut, timeout=5000):
    yield dut.rst.eq(1)
    yield dut.shim.hold.eq(1)
    for _ in range(4):
        yield
    yield dut.rst.eq(0)
    yield dut.shim.hold.eq(0)
    for _ in range(timeout):
        if not (yield dut.shim.busy):
            return True
        yield
    return False


def test_shim_read_reset_then_late_beats():
    rng  = random.Random(1)
    data = bytes(rng.getrandbits(8) for _ in range(4096))
    words = words_from_bytes(data)
    dut  = _DUT(words)
    out = {}

    def gen():
        # Warm-up: a normal read is exact through the shim.
        out["a"] = (yield from _read(dut, 0, 20))
        # The port stops answering in the middle of a 40-beat read.
        yield dut.reader.sink.addr.eq(64*32)
        yield dut.reader.sink.len.eq(40)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)
        yield dut.reader.source.ready.eq(1)
        for _ in range(3):
            yield
        yield dut.model.freeze.eq(1)
        for _ in range(100):
            yield
        # Reset the master while the port owes bursts; the shim must stay busy...
        yield dut.rst.eq(1); yield dut.shim.hold.eq(1)
        for _ in range(4):
            yield
        yield dut.rst.eq(0); yield dut.shim.hold.eq(0)
        for _ in range(50):
            yield
        out["busy_while_frozen"] = (yield dut.shim.busy)
        # ...and drain the late beats once the port answers again.
        yield dut.model.freeze.eq(0)
        for _ in range(3000):
            if not (yield dut.shim.busy):
                break
            yield
        out["drained"] = not (yield dut.shim.busy)
        out["stuck"] = (yield dut.shim.stuck)
        # A fresh read is exact: none of the late beats leaked into it.
        out["b"] = (yield from _read(dut, 32*32, 24))

    run_simulation(dut, gen())
    assert out["a"] == words[0:20]
    assert out["busy_while_frozen"] == 1
    assert out["drained"] and not out["stuck"]
    assert out["b"] == words[32:56], "late R beats leaked into the next read"


def test_shim_write_cut_by_reset():
    rng  = random.Random(2)
    words = words_from_bytes(bytes(rng.getrandbits(8) for _ in range(4096)))
    dut  = _DUT([0]*128)
    out = {}

    def gen():
        # A 24-beat write; the writer is reset after ~5 beats have gone out.
        yield dut.writer.sink_cmd.addr.eq(0)
        yield dut.writer.sink_cmd.len.eq(24)
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        sent = 0
        for _ in range(60):
            yield dut.writer.sink_data.valid.eq(1)
            yield dut.writer.sink_data.data.eq(words[sent])
            yield
            if (yield dut.writer.sink_data.ready):
                sent += 1
            if sent == 5:
                break
        yield dut.writer.sink_data.valid.eq(0)
        yield
        ok = yield from _soft_reset(dut)
        out["drained"] = ok
        # Memory: the beats that landed are exact, the padded rest is untouched (zero).
        mem = []
        for i in range(24):
            mem.append((yield dut.model.mem[i]))
        out["landed"] = mem[:sent] == words[:sent]
        out["rest_untouched"] = all(v == 0 for v in mem[sent:])
        # A later write of 10 beats at 40 is exact.
        yield dut.writer.sink_cmd.addr.eq(40*32)
        yield dut.writer.sink_cmd.len.eq(10)
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        i = 0
        while i < 10:
            yield dut.writer.sink_data.valid.eq(1)
            yield dut.writer.sink_data.data.eq(words[100 + i])
            yield
            if (yield dut.writer.sink_data.ready):
                i += 1
        yield dut.writer.sink_data.valid.eq(0)
        for _ in range(200):
            if (yield dut.writer.idle):
                break
            yield
        later = []
        for i in range(10):
            later.append((yield dut.model.mem[40 + i]))
        out["later"] = later == words[100:110]

    run_simulation(dut, gen())
    assert out["drained"]
    assert out["landed"] and out["rest_untouched"]
    assert out["later"]


def test_shim_stuck_when_port_never_answers():
    dut = _DUT([0]*128)
    out = {}

    def gen():
        yield dut.model.freeze.eq(1)
        yield dut.reader.sink.addr.eq(0)
        yield dut.reader.sink.len.eq(16)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)
        for _ in range(40):
            yield
        ok = yield from _soft_reset(dut, timeout=3000)
        out["drained"] = ok
        out["stuck"] = (yield dut.shim.stuck)

    run_simulation(dut, gen())
    assert not out["drained"] and out["stuck"]


import pytest

@pytest.mark.parametrize("order", ["aw_first", "w_first", "aw_after_w"])
def test_shim_writes_under_every_slave_handshake_order(order):
    """The shim in front of a slave that takes AW first (the legacy model), W first
    (LiteDRAM's bridge: W into a FIFO ahead of a one-deep AW buffer) or AW only after
    WVALID (a slave that waits for data - what the HBM IP's behaviour on the card implied,
    2026-09-18). Three multi-burst stores back to back, then a read-back: exact bytes, one
    B per AW (the writer goes idle), no dependence of any W handshake on the AW one
    (the aw_after_w slave would deadlock a master that had one)."""
    import numpy as np
    from litex.soc.interconnect.axi import AXIInterface
    rng = np.random.default_rng(int(sum(map(ord, order))))

    class _D(Module):
        def __init__(self):
            port = AXIInterface(data_width=256, address_width=33, id_width=6)
            self.submodules.model = BonsaiHBMChannelModel(port, size=16*1024, latency=3,
                                                          write_order=order)
            self.submodules.shim  = AXIResetShim(port, stuck_cycles=4000)
            self.submodules.writer = BonsaiAXIWriter(self.shim.master)
            self.submodules.reader = BonsaiAXIReader(self.shim.master)
            self.comb += self.reader.source.ready.eq(1)
            self.port = port
            self.bcount = Signal(8)
            self.sync += If(port.b.valid & port.b.ready, self.bcount.eq(self.bcount + 1))

    dut = _D()
    stores = [(0x1000*i, rng.integers(0, 2**32, 8*n, dtype=np.uint32)) for i, n in
              enumerate((3, 40, 5))]   # 1, 3 and 1 bursts.
    got = []

    def driver():
        for addr, words in stores:
            beats = len(words)//8
            yield dut.writer.sink_cmd.addr.eq(addr)
            yield dut.writer.sink_cmd.len.eq(beats)
            yield dut.writer.sink_cmd.valid.eq(1)
            yield
            while not (yield dut.writer.sink_cmd.ready):
                yield
            yield dut.writer.sink_cmd.valid.eq(0)
            for b in range(beats):
                w = int.from_bytes(words[8*b:8*b + 8].tobytes(), "little")
                yield dut.writer.sink_data.data.eq(w)
                yield dut.writer.sink_data.valid.eq(1)
                yield
                while not (yield dut.writer.sink_data.ready):
                    yield
            yield dut.writer.sink_data.valid.eq(0)
            for _ in range(3000):
                if (yield dut.writer.idle):
                    break
                yield
            else:
                raise AssertionError(f"{order}: the writer never went idle")
        assert (yield dut.bcount) == 5, f"{order}: {(yield dut.bcount)} B responses for 5 bursts"
        for addr, words in stores:
            beats = len(words)//8
            yield dut.reader.sink.addr.eq(addr)
            yield dut.reader.sink.len.eq(beats)
            yield dut.reader.sink.valid.eq(1)
            yield
            while not (yield dut.reader.sink.ready):
                yield
            yield dut.reader.sink.valid.eq(0)
            n = 0
            for _ in range(4000):
                if (yield dut.reader.source.valid):
                    got.append((yield dut.reader.source.data)); n += 1
                if n == beats:
                    break
                yield
            assert n == beats, order
        want = []
        for _, words in stores:
            for b in range(len(words)//8):
                want.append(int.from_bytes(words[8*b:8*b + 8].tobytes(), "little"))
        assert got == want, f"{order}: read-back differs"

    run_simulation(dut, [driver()], clocks={"sys": 10, "eng": 10})

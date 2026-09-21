# SPDX-License-Identifier: BSD-2-Clause
"""Passive traffic counting, simultaneous snapshots, reset epochs and RV32 reads."""
from types import SimpleNamespace

import pytest
from migen import Constant, Module, Signal, run_simulation
from litex.soc.interconnect.axi import AXIInterface
from litex.gen import LiteXModule

from trixium.gateware.traffic import (
    METRICS, BonsaiAXITraffic, BonsaiLiteDRAMTraffic, BonsaiTrafficCSR, add_memory_traffic,
)


def settle(n=4):
    for _ in range(n):
        yield


def snap(csr, value=1):
    yield csr.snapshot.storage.eq(value)
    yield csr.snapshot.re.eq(1)
    yield
    yield csr.snapshot.re.eq(0)
    yield


@pytest.mark.parametrize('width', [32, 256, 512])
def test_axi_handshake_strobes_stalls_errors(width):
    axi = AXIInterface(data_width=width, address_width=33)
    dut = BonsaiAXITraffic(axi, index=7)
    word = width//8

    def gen():
        # Backpressured channels: no bytes accepted, both stall classes count once.
        for channel in (axi.ar, axi.aw, axi.r, axi.w):
            yield channel.valid.eq(1)
        yield axi.w.strb.eq((1 << word) - 1)
        yield from settle(5)
        for channel in (axi.ar, axi.aw, axi.r, axi.w):
            yield channel.valid.eq(0)
        yield from settle()
        assert (yield dut.counters[0]) == 0
        assert (yield dut.counters[1]) == 0
        assert (yield dut.counters[2]) == 5
        assert (yield dut.counters[3]) == 5

        # Full, sparse and empty strobes; accepted R and B errors simultaneously.
        for channel in (axi.r, axi.w, axi.b):
            yield channel.valid.eq(1)
            yield channel.ready.eq(1)
        yield axi.r.resp.eq(2)
        yield axi.b.resp.eq(3)
        for strobe in [(1 << word) - 1, 5, 0]:
            yield axi.w.strb.eq(strobe)
            yield
        for channel in (axi.r, axi.w, axi.b):
            yield channel.valid.eq(0)
        yield from settle()
        assert (yield dut.counters[0]) == 3*word
        assert (yield dut.counters[1]) == word + 2
        assert (yield dut.counters[4]) == 6
        # No control signals were driven by the monitor.
        assert (yield axi.r.ready) == 1
        assert (yield axi.aw.ready) == 0
    run_simulation(dut, gen())
    assert dut.info == 1 | (7 << 8) | (word << 16)
    assert dut.mask == 31


def test_controller_payload_and_mask():
    interface = SimpleNamespace(nbanks=4, data_width=512)
    banks = []
    for i in range(4):
        bank = SimpleNamespace(**{name: Signal() for name in
            ['valid', 'ready', 'rdata_valid', 'wdata_ready']})
        setattr(interface, 'bank' + str(i), bank)
        banks.append(bank)
    dut = BonsaiLiteDRAMTraffic(interface, index=2)

    def gen():
        # Multiple issue events must sum, not collapse to one. Stalls count cycles.
        for bank in banks:
            yield bank.valid.eq(1)
        yield banks[0].rdata_valid.eq(1)
        yield banks[1].rdata_valid.eq(1)
        yield banks[2].wdata_ready.eq(1)
        yield from settle(3)
        for bank in banks:
            yield bank.valid.eq(0)
            yield bank.rdata_valid.eq(0)
            yield bank.wdata_ready.eq(0)
        yield from settle()
        values = []
        for c in dut.counters:
            values.append((yield c))
        assert values == [384, 192, 3, 0, 0]
    run_simulation(dut, gen())
    assert dut.info == 2 | (2 << 8) | (64 << 16)
    assert dut.mask == 7


def test_atomic_snapshot_above_48_bits_and_reset_epoch():
    class DUT(Module):
        def __init__(self):
            self.axis = [AXIInterface(data_width=256) for _ in range(2)]
            self.sources = []
            for i, axi in enumerate(self.axis):
                source = BonsaiAXITraffic(axi, index=i)
                setattr(self.submodules, 'source' + str(i), source)
                source.counters[0].reset = Constant((1 << 48) - 32, 64)
                self.sources.append(source)
            self.submodules.csr = BonsaiTrafficCSR(self.sources)
    dut = DUT()

    def gen():
        csr = dut.csr
        assert not (yield csr.valid.status)
        for axi in dut.axis:
            yield axi.r.valid.eq(1)
            yield axi.r.ready.eq(1)
        yield from settle(8)
        yield from snap(csr)
        assert (yield csr.valid.status)
        stamp = (yield csr.cycles.status)
        value = (yield csr.read_bytes.status)
        assert value > (1 << 48)
        for index in [1, 0, 1]:
            yield csr.select.storage.eq(index)
            yield from settle()
            assert (yield csr.read_bytes.status) == value
            assert (yield csr.cycles.status) == stamp
            assert (yield csr.info.status) == 1 | (index << 8) | (32 << 16)
        yield from snap(csr, 0)
        assert (yield csr.read_bytes.status) == value
        yield csr.select.storage.eq(255)
        yield from settle()
        assert not (yield csr.valid.status)
        assert (yield csr.info.status) == 0
        assert (yield csr.mask.status) == 0
        for name in METRICS:
            assert (yield getattr(csr, name).status) == 0
        yield csr.select.storage.eq(0)
        yield csr.reset_event.eq(1)
        yield from snap(csr)
        assert not (yield csr.valid.status)
        yield csr.reset_event.eq(0)
        yield csr.reset_busy.eq(1)
        yield from snap(csr)
        assert not (yield csr.valid.status)
        yield csr.reset_busy.eq(0)
        yield from snap(csr)
        assert (yield csr.valid.status)
        assert (yield csr.epoch.status) > 0
        assert (yield csr.read_bytes.status) > value
    run_simulation(dut, gen())


def test_invalid_sources():
    with pytest.raises(ValueError):
        BonsaiTrafficCSR([])
    source = BonsaiAXITraffic(AXIInterface(data_width=256))
    with pytest.raises(ValueError, match='duplicate'):
        BonsaiTrafficCSR([source, source])
    with pytest.raises(ValueError, match='sys-domain'):
        BonsaiAXITraffic(AXIInterface(data_width=256, clock_domain='eng'))


def test_target_helper_connects_reset_and_registers_sources():
    class SoC(LiteXModule):
        def __init__(self):
            self.engine_csr = SimpleNamespace(rst_trigger=Signal())
            self.engrst = SimpleNamespace(busy=Signal())
            self.axi = AXIInterface(data_width=256)
            add_memory_traffic(self, [self.axi])
    dut = SoC()

    def gen():
        yield dut.engine_csr.rst_trigger.eq(1)
        yield dut.engrst.busy.eq(1)
        yield
        yield dut.engine_csr.rst_trigger.eq(0)
        yield from snap(dut.ztraffic)
        assert not (yield dut.ztraffic.valid.status)
        assert (yield dut.ztraffic.epoch.status) == 1
        yield dut.engrst.busy.eq(0)
        yield from snap(dut.ztraffic)
        assert (yield dut.ztraffic.valid.status)
        assert (yield dut.ztraffic.count.status) == 1
    run_simulation(dut, gen())
    with pytest.raises(ValueError, match='already installed'):
        add_memory_traffic(dut, [dut.axi])

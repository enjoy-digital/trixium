# SPDX-License-Identifier: BSD-2-Clause
"""The real shared integration, including public CSRs and reset-domain wiring."""
import pytest

from migen import run_simulation
from litex.gen import LiteXModule
from litex.soc.interconnect.axi import AXIInterface

from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster, native_read_options
from trixium.gateware.hbm_model import BonsaiHBMChannelModel
from trixium.model import asm


class SoC(LiteXModule):
    def __init__(self, config, n=2):
        self.constants = {}
        self.ports = [AXIInterface(data_width=256, address_width=33, id_width=6)
                      for _ in range(n)]
        add_bonsai_cluster(self, self.ports, config)
        self.models = []
        for i, port in enumerate(self.ports):
            model = BonsaiHBMChannelModel(port, size=256*1024, latency=7, stall_rate=3)
            setattr(self, f'model{i}', model)
            self.models.append(model)

    def add_constant(self, name, value=None):
        assert name not in self.constants
        self.constants[name] = value


def write(csr, value):
    yield csr.storage.eq(value)
    yield csr.re.eq(1)
    yield
    yield csr.re.eq(0)
    yield


def submit(dut, **fields):
    before = (yield dut.engine_csr.done_count.status)
    for name, value in fields.items():
        yield from write(getattr(dut.engine_csr, name), value)
    yield from write(dut.engine_csr.submit, 1)
    for _ in range(2000):
        if (yield dut.engine_csr.done_count.status) != before:
            yield
            return
        yield
    raise AssertionError('shared-cluster CSR command timed out')


@pytest.mark.parametrize('config', [
    BonsaiClusterConfig(with_vecunit=False, scratchpad_size=16*1024),
    BonsaiClusterConfig(with_vecunit=True, scratchpad_size=16*1024,
                        chan_stride=1 << 32),  # address >4 GiB, not an extra compute pipe
    BonsaiClusterConfig(with_vecunit=True, with_dual=True, scratchpad_size=16*1024,
                        chan_stride=0x10000, chan_mod=2, chan_skew=0x1000),
])
def test_cluster_fetch_store_and_reset(config):
    dut = SoC(config)
    offset = config.chan_stride + config.chan_skew
    payload = [0x12340000 + i for i in range(3)]
    addresses = []

    def gen():
        for i, value in enumerate(payload):
            yield dut.models[1].mem[((offset + 64)//32 + i) % (256*1024//32)].eq(value)
        yield from submit(dut, kind=2, chan=1, wbase=64, wlen=3, dst=0x800)
        for i, value in enumerate(payload):
            assert (yield dut.scratchpad.mem[0x800//32 + i]) == value
        yield from submit(dut, kind=3, chan=1, wbase=0xc00, wlen=3, xsrc=0x800)
        for i, value in enumerate(payload):
            index = ((offset + 0xc00)//32 + i) % (256*1024//32)
            assert (yield dut.models[1].mem[index]) == value

        # Wedge a fetch, trigger reset through public CSRs, then let its port drain.
        yield dut.models[1].freeze.eq(1)
        for name, value in dict(kind=2, chan=1, wbase=64, wlen=3, dst=0x800).items():
            yield from write(getattr(dut.engine_csr, name), value)
        yield from write(dut.engine_csr.submit, 1)
        for _ in range(40):
            yield
        assert (yield dut.engine.busy)
        yield from write(dut.engine_csr.soft_reset, 1)
        for _ in range(80):
            yield
        assert not (yield dut.engine.busy)
        assert (yield dut.engine_csr.rst_busy)
        yield dut.models[1].freeze.eq(0)
        for _ in range(1000):
            if not (yield dut.engine_csr.rst_busy):
                break
            yield
        else:
            raise AssertionError('reset shims did not drain')
        assert not (yield dut.engine_csr.rst_stuck)
        yield from submit(dut, kind=2, chan=1, wbase=64, wlen=3, dst=0x800)
        for i, value in enumerate(payload):
            assert (yield dut.scratchpad.mem[0x800//32 + i]) == value

    from migen.sim import passive
    @passive
    def monitor():
        while True:
            if (yield dut.ports[1].ar.valid) and (yield dut.ports[1].ar.ready):
                addresses.append((yield dut.ports[1].ar.addr))
            yield

    run_simulation(dut, [gen(), monitor()], clocks={'sys': 10, 'eng': 10})
    assert addresses and set(addresses) == {offset + 64}
    assert ('BONSAI_HAS_VECUNIT' in dut.constants) == config.with_vecunit
    assert ('BONSAI_HAS_DUALQ' in dut.constants) == config.with_dual


def test_shared_sequencer_seed_mux_and_arguments():
    dut = SoC(BonsaiClusterConfig(scratchpad_size=16*1024))
    set6 = asm.assemble([('ARG', 's6', 0), ('DONE',)])
    spill = asm.assemble(asm.prog_spill())
    descriptors = []
    for control, pc, arg0 in [(2, 0, 0x1234), (2, len(set6), 0x800),
                               (2 | (1 << 4), len(set6), 0x820)]:
        words = [control, pc, arg0] + [0]*13
        for half in (words[:8], words[8:]):
            descriptors.append(sum(v << (32*i) for i, v in enumerate(half)))

    def gen():
        yield from write(dut.vecunit.prog_adr, 0)
        for word in set6 + spill:
            yield from write(dut.vecunit.prog, word)
        for i, word in enumerate(descriptors):
            yield dut.scratchpad.mem[i].eq(word)
        yield from write(dut.seq.base, 0)
        yield from write(dut.seq.count, 3)
        yield from write(dut.seq.start, 1)
        for _ in range(1000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('shared sequencer did not complete')
        yield
        assert (yield dut.seq.status.fields.dcount) == 3
        assert (yield dut.vecunit.done_count.status) == 3
        assert not (yield dut.vecunit.status.fields.bad_op)
        # Chained res6=0x1234 vs engine absmax=0: the w0[4] selection must differ.
        assert (yield dut.scratchpad.mem[0x800//32]) == sum(0x1234 << (32*i) for i in range(8))
        assert (yield dut.scratchpad.mem[0x820//32]) == 0
    run_simulation(dut, gen(), clocks={'sys': 10, 'eng': 10})


@pytest.mark.parametrize('kwargs', [
    {'with_vecunit': False, 'with_dnrec': True}, {'chan_stride': 96},
    {'chan_stride': -32}, {'chan_mod': 3}, {'chan_skew': 4096},
    {'chan_mod': 2, 'chan_skew': 1}, {'scratchpad_size': 33}, {'loader_data_width': 96},
    {'native_endpoints': [0]}, {'native_reservations': [object()]},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BonsaiClusterConfig(**kwargs)


def test_rejects_overwrite_and_invalid_ports():
    dut = SoC(BonsaiClusterConfig(with_vecunit=False))
    with pytest.raises(ValueError, match='overwrite'):
        add_bonsai_cluster(dut, dut.ports)
    for ports in ([], [AXIInterface(data_width=128)],
                  [AXIInterface(data_width=256, clock_domain='other')]):
        with pytest.raises(ValueError):
            add_bonsai_cluster(LiteXModule(), ports)


def test_native_configuration_is_explicit_frozen_and_separate_from_public_caps():
    from trixium.capabilities import Access
    from test_native_reader import region
    regions, reservations, endpoints = [region(access=Access(0))], [], [0]
    opts = native_read_options(regions, reservations, endpoints)
    config = BonsaiClusterConfig(with_vecunit=False, **opts)
    regions.clear()
    endpoints[0] = 9
    assert config.native_regions[0].access == Access.READ
    assert config.native_endpoints == (0,)
    assert opts['native_regions'][0].access == Access.READ
    original = region(access=Access(0))
    native_read_options([original], [], [0])
    assert original.access == Access(0)
    with pytest.raises(ValueError, match='every compute port'):
        add_bonsai_cluster(LiteXModule(), [AXIInterface(data_width=256)]*2, config)
    default = SoC(BonsaiClusterConfig(with_vecunit=False), n=1)
    assert not hasattr(default.engine.cmd, 'native')
    assert not hasattr(default, 'zznative')
    assert 'BONSAI_HAS_NATIVE_ENGINE' not in default.constants

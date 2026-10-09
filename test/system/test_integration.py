#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The real shared integration (gateware/core/core.py), including public CSRs and reset-domain
wiring. Its harness (SoC on HBM channel models, CSR write/submit) is common/cluster.py."""

import pytest

from migen import run_simulation

from litex.soc.interconnect.axi import AXIInterface

from gateware.core.core import TriXiumConfig, TriXiumCore
from model import asm

from common.cluster import SoC, write, submit


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('config', [
    TriXiumConfig(with_vecunit=False, scratchpad_size=16*1024),
    TriXiumConfig(with_vecunit=True, scratchpad_size=16*1024,
                        chan_stride=1 << 32),  # address >4 GiB, not an extra compute pipe
    TriXiumConfig(with_vecunit=True, scratchpad_size=16*1024,
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
    assert ('TRIXIUM_HAS_VECUNIT' in dut.constants) == config.with_vecunit


def test_shared_sequencer_seed_mux_and_arguments():
    dut = SoC(TriXiumConfig(scratchpad_size=16*1024))
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
    {'chan_stride': 96},
    {'chan_stride': -32}, {'chan_mod': 3}, {'chan_skew': 4096},
    {'chan_mod': 2, 'chan_skew': 1}, {'scratchpad_size': 33}, {'loader_data_width': 96},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TriXiumConfig(**kwargs)


def test_rejects_invalid_ports():
    for ports in ([], [AXIInterface(data_width=128)],
                  [AXIInterface(data_width=256, clock_domain='other')]):
        with pytest.raises(ValueError):
            TriXiumCore(ports)

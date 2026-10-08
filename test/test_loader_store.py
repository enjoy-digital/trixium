#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Loader STORE path (scratchpad -> HBM, the KV-cache append).

On hardware a KV store wedges the engine: it sits in ST_DRAIN waiting for `store_busy` to
drop, which means the loader is stuck in EST_DRAIN waiting for the AXI writer's B responses.
The descriptor sequencer then stalls partway through a token (`SEQSTALL dcount=29/84 eng=01`)
while matvec, fetch and the vector unit are all fine.

The token gate runs the HBM model at latency=16 with no R stalls, which is nothing like the
real part, so this exercises short stores against a slow, stalling channel - the shape of the
traffic the KV append actually generates.
"""

import pytest

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium.memory.hbm_model import BonsaiHBMChannelModel
from trixium.memory.loader import BonsaiHBMLoader

MEM_SIZE = 64*1024


class _DUT(Module):
    def __init__(self, n=2, latency=100, stall_rate=8):
        axis = []
        self.models = []
        for i in range(n):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            axis.append(axi)
            model = BonsaiHBMChannelModel(axi, size=MEM_SIZE, latency=latency,
                                          stall_rate=stall_rate)
            self.submodules += model
            self.models.append(model)
        self.loader = BonsaiHBMLoader(axis, data_width=128)
        self.submodules += self.loader


def _store(nbeats, chan=0, addr=0x400, latency=100, stall_rate=8, gap=0):
    """Drive one STORE of `nbeats`; return (completed, memory contents)."""
    dut = _DUT(latency=latency, stall_rate=stall_rate)
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

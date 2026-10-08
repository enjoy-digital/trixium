#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Infrastructure unit tests: engine FETCH op (HBM -> scratchpad), BonsaiHBMLoader
# (host stream -> HBM).

import numpy as np
from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium.core.scratchpad import BonsaiScratchpad
from trixium.core.engine import BonsaiMatVecEngine
from trixium.memory.loader import BonsaiHBMLoader
from trixium.memory.hbm_model import BonsaiHBMChannelModel, words_from_bytes

def test_engine_fetch():
    rng  = np.random.default_rng(21)
    data = rng.integers(0, 256, 8*1024, dtype=np.uint8).tobytes()

    class DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(2)]
            self.submodules.scratchpad = BonsaiScratchpad(size=16*1024)
            self.submodules.engine     = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=8)
            self.submodules.model0 = BonsaiHBMChannelModel(axis[0], size=8*1024, latency=4)
            self.submodules.model1 = BonsaiHBMChannelModel(axis[1], size=8*1024,
                init=words_from_bytes(data), latency=12)

    dut  = DUT()
    src  = 32*7   # Beat-aligned source in channel 1.
    dst  = 0x1000
    nbeats = 10

    def gen():
        yield dut.engine.cmd.kind.eq(2)
        yield dut.engine.cmd.chan.eq(1)
        yield dut.engine.cmd.wbase.eq(src)
        yield dut.engine.cmd.wlen.eq(nbeats)
        yield dut.engine.cmd.dst.eq(dst)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(2000):
            if (yield dut.engine.done):
                break
            yield
        else:
            raise AssertionError("fetch timeout")
        yield
        for i in range(nbeats):
            got    = (yield dut.scratchpad.mem[dst//32 + i])
            expect = int.from_bytes(data[src + 32*i:src + 32*(i+1)], "little")
            assert got == expect, f"beat {i}"

    run_simulation(dut, [gen()])

def test_loader():
    class DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(2)]
            self.submodules.loader = BonsaiHBMLoader(axis, data_width=128)
            self.submodules.model0 = BonsaiHBMChannelModel(axis[0], size=8*1024, latency=4)
            self.submodules.model1 = BonsaiHBMChannelModel(axis[1], size=8*1024, latency=4)

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

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The shared-cluster harness: the real TriXiumCore on HBM channel models, its CSR and list
helpers, and the small integer operators the stream-unit plumbing tests plug in."""

from migen import Signal

from litex.gen import LiteXModule

from litex.soc.interconnect.axi import AXIInterface

from gateware.core.core import TriXiumCore
from gateware.memory.hbm_model import HBMChannelModel

# SoC ----------------------------------------------------------------------------------------------

class SoC(TriXiumCore):
    """The core with an HBM channel model on each memory port."""
    def __init__(self, config, n=2):
        self.ports = [AXIInterface(data_width=256, address_width=33, id_width=6)
                      for _ in range(n)]
        TriXiumCore.__init__(self, self.ports, config)
        self.models = []
        for i, port in enumerate(self.ports):
            model = HBMChannelModel(port, size=256*1024, latency=7, stall_rate=3)
            setattr(self, f'model{i}', model)
            self.models.append(model)

# Helpers ------------------------------------------------------------------------------------------

def pack(words):
    """Eight 32-bit words -> one 256-bit scratchpad beat."""
    return sum((int(w) & 0xFFFFFFFF) << (32*i) for i, w in enumerate(words))


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


def install_list(dut, descriptors):
    """Write a descriptor list at scratchpad 0 and start the sequencer on it."""
    for index, words in enumerate(descriptors):
        assert len(words) == 16
        yield dut.scratchpad.mem[index*2].eq(pack(words[:8]))
        yield dut.scratchpad.mem[index*2 + 1].eq(pack(words[8:]))
    yield from write(dut.seq.base, 0)
    yield from write(dut.seq.count, len(descriptors))
    yield from write(dut.seq.start, 1)

# Integer Operators --------------------------------------------------------------------------------

# Five-cycle integer add/multiply: the stream units' plumbing tests run on these (the units'
# arithmetic is qualified bit for bit on the exact FP32 operators in their own tests).

class IntAdd5(LiteXModule):
    latency = 5

    def __init__(self):
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a + self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]


class IntMul5(LiteXModule):
    latency = 5

    def __init__(self):
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a*self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]

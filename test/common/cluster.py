#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The shared-cluster harness: the real TriXiumCore on HBM channel models, its CSR and list
helpers, and the small integer operators the stream-unit plumbing tests plug in.

The cluster tests are Migen testbench generators; run() runs them on either backend:

- "verilator": the core compiled as a co-simulation library (common.verilator.Compiled,
  cached), the channels answered by C++ AXI slaves. The default.
- "migen": Migen's simulator with HBMChannelModel channels - the slow cross-check.

The generators see the same objects on both: dut.models[i].mem / .freeze are the channel
memory and its read freeze (C++ AXI slave "axi<i>" when compiled), every CSR, the units' and
engine's status signals, the scratchpad memory.
"""

import pytest

from migen import Memory, Signal, run_simulation

from litex.gen import LiteXModule

from litex.soc.interconnect.axi import AXIInterface
from litex.soc.interconnect.csr import CSRStatus, CSRStorage

from gateware.core.core import TriXiumCore
from gateware.memory.hbm_model import HBMChannelModel

from common.verilator import Compiled

# Constants ----------------------------------------------------------------------------------------

MODEL_SIZE    = 256*1024
MODEL_LATENCY = 7
MODEL_STALL   = 3         # Sixteenths of the cycles (HBMChannelModel's stall_rate).

# The compiled core in the fast tier, Migen's simulator in the slow one (one representative
# test per file: the cross-check of the emitted Verilog against Migen's own semantics).
MIGEN   = pytest.param("migen", marks=pytest.mark.slow)
BACKEND = pytest.mark.parametrize("backend", ["verilator", MIGEN])

# SoC ----------------------------------------------------------------------------------------------

class ChannelHandle:
    """An HBMChannelModel's testbench face (mem, freeze) for compiled runs, where the channel is
    a C++ AXI slave: run() maps both onto it."""
    def __init__(self, size):
        self.mem    = Memory(256, size//32)
        self.freeze = Signal()


class SoC(TriXiumCore):
    """The core with an HBM channel model on each memory port (backend "migen") or with its
    ports open for the C++ AXI slaves (backend "verilator")."""
    def __init__(self, config, n=2, backend="migen"):
        self.ports = [AXIInterface(data_width=256, address_width=33, id_width=6)
                      for _ in range(n)]
        TriXiumCore.__init__(self, self.ports, config)
        self.config  = config
        self.backend = backend
        self.models  = []
        for i, port in enumerate(self.ports):
            if backend == "migen":
                model = HBMChannelModel(port, size=MODEL_SIZE, latency=MODEL_LATENCY,
                                        stall_rate=MODEL_STALL)
                setattr(self, f'model{i}', model)
            else:
                assert backend == "verilator", backend
                model = ChannelHandle(MODEL_SIZE)
            self.models.append(model)

# Run ----------------------------------------------------------------------------------------------

def cluster_signals(dut):
    """{name: Signal} a cluster testbench may touch: every CSR (storage/re, status and its
    fields) and the Signal attributes (and command handshake) of the core's modules."""
    signals = {}
    for csr in dut.get_csrs():
        if isinstance(csr, CSRStorage):
            signals[csr.name + "_storage"] = csr.storage
            signals[csr.name + "_re"]      = csr.re
        elif isinstance(csr, CSRStatus):
            signals[csr.name + "_status"]  = csr.status
            for field in getattr(getattr(csr, "fields", None), "fields", []):
                signals[f"{csr.name}_{field.name}"] = field
    for name in ("engine", "engine_csr", "seq", "vecunit", "loader", "engrst", "scratchpad", "hadamard",
                 "activation_quantize", "projection_restore", "silu", "norm", "conv"):
        module = getattr(dut, name, None)
        if module is None:
            continue
        for attr, value in vars(module).items():
            if isinstance(value, Signal) and not attr.startswith("_"):
                signals[f"{name}_{attr}"] = value
        cmd = getattr(module, "cmd", None)
        if cmd is not None and hasattr(cmd, "valid"):
            signals[f"{name}_cmd_valid"] = cmd.valid
            signals[f"{name}_cmd_ready"] = cmd.ready
    return signals


_compiled = {}

def run(dut, generators, signals=None, memories=None):
    """run_simulation(dut, generators) on the SoC's backend (see the module doc). signals /
    memories: {name: object} the generators touch beyond cluster_signals() and the
    scratchpad."""
    if dut.backend == "migen":
        run_simulation(dut, generators, clocks={"sys": 10, "eng": 10})
        return
    sigs = cluster_signals(dut)
    sigs.update(signals or {})
    mems = {"sp": dut.scratchpad.mem}
    mems.update(memories or {})
    axi  = {f"axi{i}": port for i, port in enumerate(dut.ports)}
    key  = f"cluster{dut.config!r}n{len(dut.ports)}"
    compiled = Compiled(dut, sigs, memories=mems, axi=axi, name="cluster", key=key)
    aliases, config = {}, {}
    for i, model in enumerate(dut.models):
        aliases[model.mem]    = f"axi{i}"
        aliases[model.freeze] = f"axi{i}_freeze"
        config[f"axi{i}"] = dict(size=MODEL_SIZE, latency=MODEL_LATENCY,
                                 r_stall=(MODEL_STALL*100 + 8)//16, max_len=16, seed=i + 1)
    compiled.run(generators, axi=config, aliases=aliases)

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

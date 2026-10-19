#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""EngineResetDomain: the soft-resettable clock domain the engine cluster runs in.

`eng` is the `sys` clock with a reset the firmware can pulse (the engine soft-reset CSR): the
matvec engine, the loader (whose per-channel writers carry the engine's KV stores), the
sequencer and the vector unit are instantiated under `ClockDomainsRenamer("eng")`. Their
memory ports go through one AXIResetShim each, which keeps the AXI protocol whole while they
reset and reports when a port cannot be drained. The scratchpad, the CSR front and the SoC stay
in `sys`: the vector programs and the descriptor lists live in memories that a reset does not
touch, so the firmware needs no re-upload afterwards.

A wedged op (a stalled sequencer, a fetch of a row that does not exist) thus costs a CSR write
and a few microseconds of drain instead of an FPGA reconfigure.
"""

from operator import or_
from functools import reduce

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer
from migen.genlib.cdc import MultiReg

from litex.gen import *

from gateware.memory.axi import AXIResetShim

# Engine Reset Domain ------------------------------------------------------------------------------

class EngineResetDomain(LiteXModule):
    def __init__(self, axis, hold_cycles=64, stuck_cycles=1 << 20, pipe_clk=None, read_axis=None,
                 read_clk=None):
        # pipe_clk: a clock domain of the SoC the channel pipes run in (PipeIsland); the engine
        # domain `peng` is that clock with this module's reset (the soft reset holds the pipes).
        # read_axis/read_clk: the channels' read-only AXI ports in the memory's own clock domain
        # (ReaderIsland): their shims' read halves run there (never reset, hold and status
        # crossed by MultiReg), the `axis` shims keep the write half only, and the readers'
        # domain `reng` is that clock with this module's reset.
        self.cd_eng = ClockDomain("eng")
        # The reset register implements its own reset equation below: Migen's sys reset
        # insertion would append eng_rst <= 0 and override eng_rst <= sys_rst | holding.
        self.cd_eng.rst.reset_less = True
        self.trigger = Signal()      # One-cycle pulse: start a soft reset.
        self.holding = Signal()      # The reset is asserted (hold_cycles after trigger).
        self.busy    = Signal()      # Some port is still draining: do not submit yet.
        self.stuck   = Signal()      # Some port will never drain: reconfigure.
        self.masters = []            # The shimmed AXI interfaces the engine cluster uses.
        self.read_masters = []       # With read_axis: the readers' (read-only, in reng).

        # # #

        shims = []
        for i, axi in enumerate(axis):
            shim = AXIResetShim(axi, stuck_cycles=stuck_cycles, reads=read_axis is None)
            setattr(self, f"shim{i}", shim)
            shims.append(shim)
            self.masters.append(shim.master)
        busy_terms, stuck_terms = [], []
        if read_axis is not None:
            self.cd_reng = ClockDomain("reng")
            self.cd_reng.rst.reset_less = True
            self.comb += self.cd_reng.clk.eq(ClockSignal(read_clk))
            self.specials += AsyncResetSynchronizer(self.cd_reng,
                ResetSignal(read_clk, allow_reset_less=True) | self.holding)
            hold_rd = Signal()
            self.specials += MultiReg(self.holding, hold_rd, odomain=read_clk)
            for i, axi in enumerate(read_axis):
                shim = ClockDomainsRenamer(read_clk)(
                    AXIResetShim(axi, stuck_cycles=stuck_cycles, writes=False))
                setattr(self, f"rshim{i}", shim)
                self.read_masters.append(shim.master)
                busy, stuck = Signal(), Signal()
                self.comb += shim.hold.eq(hold_rd)
                self.specials += [
                    MultiReg(shim.busy,  busy,  odomain="sys"),
                    MultiReg(shim.stuck, stuck, odomain="sys"),
                ]
                busy_terms.append(busy)
                stuck_terms.append(stuck)
            # The readers' reset releases after the hold ends (the synchronizer's latency): busy
            # covers it, so a submit never meets a reader still in reset.
            reng_rst = Signal()
            self.specials += MultiReg(self.cd_reng.rst, reng_rst, odomain="sys")
            busy_terms.append(reng_rst)

        hold_cnt = Signal(max=hold_cycles + 1)
        self.comb += [
            self.cd_eng.clk.eq(ClockSignal("sys")),
            self.holding.eq(hold_cnt != 0),
        ]
        self.sync += [
            If(self.trigger,
                hold_cnt.eq(hold_cycles),
            ).Elif(hold_cnt != 0,
                hold_cnt.eq(hold_cnt - 1),
            ),
            # Registered: one clean high-fanout net, the sys reset ORed in.
            self.cd_eng.rst.eq(ResetSignal("sys", allow_reset_less=True) | self.holding),
        ]
        if pipe_clk:
            self.cd_peng = ClockDomain("peng")
            self.cd_peng.rst.reset_less = True
            self.comb += self.cd_peng.clk.eq(ClockSignal(pipe_clk))
            self.specials += AsyncResetSynchronizer(self.cd_peng,
                ResetSignal(pipe_clk, allow_reset_less=True) | self.holding)
        if shims:
            self.comb += [
                self.busy.eq(reduce(or_, [s.busy for s in shims] + busy_terms) | self.holding),
                self.stuck.eq(reduce(or_, [s.stuck for s in shims] + stuck_terms)),
            ]
            for s in shims:
                self.comb += s.hold.eq(self.holding)

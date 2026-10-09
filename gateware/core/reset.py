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

from litex.gen import *

from gateware.memory.axi import AXIResetShim

# Engine Reset Domain ------------------------------------------------------------------------------

class EngineResetDomain(LiteXModule):
    def __init__(self, axis, hold_cycles=64, stuck_cycles=1 << 20):
        self.cd_eng = ClockDomain("eng")
        # The reset register implements its own reset equation below: Migen's sys reset
        # insertion would append eng_rst <= 0 and override eng_rst <= sys_rst | holding.
        self.cd_eng.rst.reset_less = True
        self.trigger = Signal()      # One-cycle pulse: start a soft reset.
        self.holding = Signal()      # The reset is asserted (hold_cycles after trigger).
        self.busy    = Signal()      # Some port is still draining: do not submit yet.
        self.stuck   = Signal()      # Some port will never drain: reconfigure.
        self.masters = []            # The shimmed AXI interfaces the engine cluster uses.

        # # #

        shims = []
        for i, axi in enumerate(axis):
            shim = AXIResetShim(axi, stuck_cycles=stuck_cycles)
            setattr(self, f"shim{i}", shim)
            shims.append(shim)
            self.masters.append(shim.master)

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
        if shims:
            self.comb += [
                self.busy.eq(reduce(or_, [s.busy for s in shims]) | self.holding),
                self.stuck.eq(reduce(or_, [s.stuck for s in shims])),
            ]
            for s in shims:
                self.comb += s.hold.eq(self.holding)

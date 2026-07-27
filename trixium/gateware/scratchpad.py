#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiScratchpad: on-chip activation memory (URAM class on hardware).

256-bit wide memory with:
- a read port for the engine's x-buffer fill,
- a 32-bit-granular write port for the engine's result gather,
- (added at SoC integration) a bus slave port for CPU/host/JTAG access.

Addresses are in 256-bit words; the gather port addresses int32 slots."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import wishbone

# BonsaiScratchpad ---------------------------------------------------------------------------------

class BonsaiScratchpad(LiteXModule):
    def __init__(self, size=256*1024, init=None):
        self.size = size
        depth     = size//32
        # The raw Memory must not be auto-exported over the CSR bus (AutoCSR scans Memory
        # attributes of LiteXModules and would add an unwanted CSR-bus port + page register).
        self.autocsr_exclude = {"mem"}

        # # #

        self.mem = mem = Memory(256, depth, init=init)
        self.specials += mem

        # Engine x-fill read port.
        self.fill = mem.get_port()
        self.specials += self.fill

        # Engine gather write port (int32 granularity).
        self.gather = mem.get_port(write_capable=True, we_granularity=32)
        self.specials += self.gather

    def add_wishbone_port(self):
        """32-bit Wishbone slave onto the 256-bit memory (CPU/host/JTAG access).

        Declared with byte addressing (what the SoC bus hands slaves in current LiteX -
        verified empirically in simulation: adr LSBs advance by 4 per 32-bit word)."""
        bus  = wishbone.Interface(data_width=32, address_width=32, addressing="byte")
        port = self.mem.get_port(write_capable=True, we_granularity=8)
        self.specials += port

        sel = Signal(3) # 32-bit slice within the 256-bit word.
        we  = Signal()
        self.comb += [
            port.adr.eq(bus.adr[5:]),  # 32-byte beat index.
            sel.eq(bus.adr[2:5]),      # 32-bit word within the beat.
            port.dat_w.eq(Replicate(bus.dat_w, 8)),
            # Constant-slice muxes: variable shift amounts computed with '*' are self-determined
            # (truncated) in the emitted Verilog - never use them.
            Case(sel, {i: bus.dat_r.eq(port.dat_r[32*i:32*(i+1)]) for i in range(8)}),
            If(we,
                Case(sel, {i: port.we.eq(bus.sel << (4*i)) for i in range(8)}),
            ),
        ]
        self.bus_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(bus.cyc & bus.stb,
                If(bus.we,
                    we.eq(1),
                    NextState("ACK"), # Registered ack (comb ack races pipelined masters).
                ).Else(
                    NextState("READ"), # One cycle of read latency.
                )
            )
        )
        fsm.act("READ",
            bus.ack.eq(1),
            NextState("IDLE"),
        )
        fsm.act("ACK",
            bus.ack.eq(1),
            NextState("IDLE"),
        )
        return bus

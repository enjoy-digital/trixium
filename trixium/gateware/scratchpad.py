#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiScratchpad: on-chip activation memory (URAM class on hardware).

256-bit wide memory with exactly TWO physical ports (Vivado's BRAM inference limit -
a 4-port Memory trips 'Unsupported RAM template', dissolves into FF+LUT emulation and
blew the N=16 LUT budget):
- port R: the engine's x-buffer fill reads (also vec/seq via the integrator's muxes),
- port W: the engine's result gather writes (byte enables; the 32-bit lane enables the
  engine drives are expanded to bytes here).

CPU/host access (wishbone slave, CSR peek/poke) rides a SIDEBAND that borrows both
ports when `units_idle` is high (integrator: no engine/vec/seq activity). Wishbone
masters tolerate the resulting wait-states natively; in resident mode the CPU only
touches the scratchpad between hardware ops, so waits are boot/debug-path only.

Addresses are in 256-bit words; the gather facade addresses int32 slots."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import wishbone
from litex.soc.interconnect.csr import *

# BonsaiScratchpad ---------------------------------------------------------------------------------

class _Facade:
    pass

class BonsaiScratchpad(LiteXModule):
    def __init__(self, size=256*1024, init=None):
        self.size = size
        depth     = size//32
        # The raw Memory must not be auto-exported over the CSR bus (AutoCSR scans Memory
        # attributes of LiteXModules and would add an unwanted CSR-bus port + page register).
        self.autocsr_exclude = {"mem"}

        # Integrator input: no engine/vec/seq scratchpad activity this cycle (sideband
        # borrow window). Left at 0 (never granted) when unwired - unit-test DUTs and
        # configurations without a bus port need no changes.
        self.units_idle = Signal()

        # # #

        self.mem = mem = Memory(256, depth, init=init)
        self.specials += mem
        port_r = mem.get_port()
        port_w = mem.get_port(write_capable=True, we_granularity=8)
        self.specials += port_r, port_w

        # Engine-facing facades (attribute-compatible with the old Memory ports).
        self.fill = _Facade()
        self.fill.adr   = Signal(max=depth)
        self.fill.dat_r = Signal(256)
        self.gather = _Facade()
        self.gather.adr   = Signal(max=depth)
        self.gather.dat_w = Signal(256)
        self.gather.we    = Signal(8)   # int32 lane enables (expanded to bytes below).

        # Sideband slots: 0 = wishbone (priority), 1 = CSR peek/poke. Each carries one
        # 32-bit access; grant only while every unit is idle.
        self._sb_req  = [Signal(name=f"sb{i}_req")  for i in range(2)]
        self._sb_we   = [Signal(name=f"sb{i}_we")   for i in range(2)]
        self._sb_adr  = [Signal(max=depth, name=f"sb{i}_adr") for i in range(2)]
        self._sb_sel  = [Signal(3, name=f"sb{i}_sel")  for i in range(2)]
        self._sb_bsel = [Signal(4, name=f"sb{i}_bsel") for i in range(2)]
        self._sb_wdat = [Signal(32, name=f"sb{i}_wdat") for i in range(2)]
        self._sb_gnt  = [Signal(name=f"sb{i}_gnt")  for i in range(2)]
        self._sb_rdat = Signal(256) # port_r.dat_r, valid the cycle after a granted read.

        gather_bwe = Signal(32)
        self.comb += [
            gather_bwe.eq(Cat(*[Replicate(self.gather.we[i], 4) for i in range(8)])),
            self._sb_gnt[0].eq(self._sb_req[0] & self.units_idle),
            self._sb_gnt[1].eq(self._sb_req[1] & self.units_idle & ~self._sb_req[0]),
            self.fill.dat_r.eq(port_r.dat_r),
            self._sb_rdat.eq(port_r.dat_r),
            If(self._sb_gnt[0],
                port_r.adr.eq(self._sb_adr[0]),
                port_w.adr.eq(self._sb_adr[0]),
                port_w.dat_w.eq(Replicate(self._sb_wdat[0], 8)),
                If(self._sb_we[0],
                    Case(self._sb_sel[0],
                         {i: port_w.we.eq(self._sb_bsel[0] << (4*i)) for i in range(8)}),
                )
            ).Elif(self._sb_gnt[1],
                port_r.adr.eq(self._sb_adr[1]),
                port_w.adr.eq(self._sb_adr[1]),
                port_w.dat_w.eq(Replicate(self._sb_wdat[1], 8)),
                If(self._sb_we[1],
                    Case(self._sb_sel[1],
                         {i: port_w.we.eq(self._sb_bsel[1] << (4*i)) for i in range(8)}),
                )
            ).Else(
                port_r.adr.eq(self.fill.adr),
                port_w.adr.eq(self.gather.adr),
                port_w.dat_w.eq(self.gather.dat_w),
                port_w.we.eq(gather_bwe),
            )
        ]

    def add_csr_port(self):
        """CSR-based peek/poke (host access over the PCIe BAR0/JTAG CSR window). Write
        `adr` to trigger a read; poll `data_r` (valid once the sideband was granted -
        debug path, host pollers naturally retry). Write `data_w` to store 32 bits."""
        return _ScratchpadCSR(self)

    def add_wishbone_port(self):
        """32-bit Wishbone slave onto the 256-bit memory (CPU/host/JTAG access) via the
        sideband: accesses wait (wishbone-native stalls) while engine/vec/seq are active.

        Declared with byte addressing (what the SoC bus hands slaves in current LiteX -
        verified empirically in simulation: adr LSBs advance by 4 per 32-bit word)."""
        bus = wishbone.Interface(data_width=32, address_width=32, addressing="byte")
        # The request is REGISTERED at the scratchpad (n16 timing #3c: the CPU's bus
        # decode reached the memory address pins combinationally - 5 ns of route from the
        # core to a 64-BRAM address net, the largest failing family). adr/sel/dat_w/we are
        # latched in IDLE and presented from flops; the read data is latched too, so the
        # return trip is FF -> FF as well. +2 cycles per CPU access (a few thousand per
        # token: noise).
        adr_r  = Signal(len(self._sb_adr[0]))
        sel_r  = Signal(3)
        bsel_r = Signal(4)
        wdat_r = Signal(32)
        we_r   = Signal()
        rdat_r = Signal(32)
        self.comb += [
            self._sb_adr[0].eq(adr_r),
            self._sb_sel[0].eq(sel_r),
            self._sb_bsel[0].eq(bsel_r),
            self._sb_wdat[0].eq(wdat_r),
            self._sb_we[0].eq(we_r),
            bus.dat_r.eq(rdat_r),
        ]
        self.bus_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(bus.cyc & bus.stb,
                NextValue(adr_r,  bus.adr[5:]),   # 32-byte beat index.
                NextValue(sel_r,  bus.adr[2:5]),  # 32-bit word within the beat.
                NextValue(bsel_r, bus.sel),
                NextValue(wdat_r, bus.dat_w),
                NextValue(we_r,   bus.we),
                NextState("REQ"),
            )
        )
        fsm.act("REQ",
            self._sb_req[0].eq(1),
            If(self._sb_gnt[0],
                If(we_r,
                    NextState("ACK"), # Registered ack (comb ack races pipelined masters).
                ).Else(
                    NextState("READ"), # One cycle of read latency.
                )
            )
        )
        fsm.act("READ",
            Case(sel_r, {i: NextValue(rdat_r, self._sb_rdat[32*i:32*(i+1)]) for i in range(8)}),
            NextState("ACK"),
        )
        fsm.act("ACK",
            bus.ack.eq(1),
            NextState("IDLE"),
        )
        return bus

# _ScratchpadCSR -----------------------------------------------------------------------------------

class _ScratchpadCSR(LiteXModule):
    def __init__(self, scratchpad):
        self.adr    = CSRStorage(24, description="Byte address (4-aligned). Writing triggers a read.")
        self.data_r = CSRStatus(32,  description="Read data for the last triggered read.")
        self.data_w = CSRStorage(32, description="Writing stores 32 bits at adr.")

        # # #

        sp    = scratchpad
        sel_r = Signal(3)
        rd_p  = Signal() # Read pending (adr written, not yet granted).
        wr_p  = Signal() # Write pending.
        cap   = Signal() # Capture read data this cycle.
        self.comb += [
            sp._sb_adr[1].eq(self.adr.storage[5:]),
            sp._sb_sel[1].eq(self.adr.storage[2:5]),
            sp._sb_bsel[1].eq(0xF),
            sp._sb_wdat[1].eq(self.data_w.storage),
            sp._sb_req[1].eq(rd_p | wr_p),
            sp._sb_we[1].eq(wr_p),
        ]
        self.sync += [
            If(self.adr.re,
                rd_p.eq(1),
            ).Elif(sp._sb_gnt[1] & ~wr_p,
                rd_p.eq(0),
            ),
            If(self.data_w.re,
                wr_p.eq(1),
            ).Elif(sp._sb_gnt[1] & wr_p,
                wr_p.eq(0),
            ),
            cap.eq(sp._sb_gnt[1] & ~wr_p),
            sel_r.eq(sp._sb_sel[1]),
            If(cap,
                Case(sel_r, {i: self.data_r.status.eq(sp._sb_rdat[32*i:32*(i+1)])
                             for i in range(8)}),
            ),
        ]

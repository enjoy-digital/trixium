#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Scratchpad: on-chip activation memory, and the arbiter of its unit-side client port.

256-bit wide memory with exactly two physical ports (more ports defeat Vivado's block RAM
inference and fall back to FF+LUT emulation):
- port R: the engine's x-buffer fill reads (also vec/seq through the integrator's muxes),
- port W: the engine's result gather writes (the 32-bit lane enables the engine drives are
  expanded to byte enables here).

CPU/host access (wishbone slave, CSR peek/poke) rides a sideband: writes are posted through a
small FIFO drained onto the W port when the integrator grants it, reads borrow the R port when
it is free (`units_idle` or `sbr_allow`) and the FIFO is empty.

Addresses are in 256-bit words; the gather facade addresses int32 slots.
"""

from operator import or_
from functools import reduce

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import *

from litex.soc.interconnect import wishbone
from litex.soc.interconnect.csr import *

# Helpers ------------------------------------------------------------------------------------------

class _Facade:
    """Attribute container standing in for a Memory port (adr/dat_r/dat_w/we)."""

# Scratchpad ---------------------------------------------------------------------------------------

class Scratchpad(LiteXModule):
    """256-bit scratchpad memory: engine fill/gather ports plus a CPU/host sideband."""
    def __init__(self, size=256*1024, init=None, xpm=False):
        self.size = size
        depth     = size//32

        # The raw Memory must not be auto-exported over the CSR bus (AutoCSR scans Memory
        # attributes of LiteXModules and would add an unwanted CSR-bus port + page register).
        self.autocsr_exclude = {"mem"}

        # Integrator input: no engine/vec/seq scratchpad activity this cycle (sideband
        # borrow window). Left at 0 (never granted) when unwired.
        self.units_idle = Signal()
        # Posted sideband writes: the wishbone ack fires on FIFO accept and the integrator
        # drains the FIFO onto the W port with an engine-local grant (sbw_gnt), keeping
        # CPU-side terms out of the port cones (congestion). Reads wait for FIFO-empty, so
        # they always see every posted write, and the integrator holds the sequencer's
        # descriptor fetch while the FIFO is non-empty (no stale descriptor).
        self.sbw_pending = Signal() # FIFO non-empty (integrator: drain + seq interlock).
        self.sbw_gnt     = Signal() # Integrator grant: drain one write this cycle.
        # Read slot: the integrator may raise sbr_allow on any cycle the R port is free, so
        # CPU/host reads need not wait for all units idle. Reads only borrow the R port.
        self.sbr_allow   = Signal()
        self.sbr_gnt_any = Signal() # For the integrator's rd_free gating.

        # # #

        if xpm:
            # Xilinx builds: an explicit simple dual-port block RAM (inside the full SoC,
            # inference turned this memory into LUT RAM). read_first: a read of the address
            # being written returns the old data, as the Memory below does in simulation.
            assert init is None
            aw = (depth - 1).bit_length()
            port_r, port_w = _Facade(), _Facade()
            port_r.adr, port_r.dat_r = Signal(aw), Signal(256)
            port_w.adr, port_w.dat_w, port_w.we = Signal(aw), Signal(256), Signal(32)
            self.mem = None
            self.specials += Instance("xpm_memory_sdpram",
                # Memory.
                p_MEMORY_PRIMITIVE    = "block",
                p_MEMORY_SIZE         = depth*256,
                p_CLOCKING_MODE       = "common_clock",
                p_ECC_MODE            = "no_ecc",
                p_MEMORY_INIT_FILE    = "none",
                p_MEMORY_INIT_PARAM   = "0",
                p_USE_MEM_INIT        = 0,
                p_MESSAGE_CONTROL     = 0,
                p_AUTO_SLEEP_TIME     = 0,
                p_WAKEUP_TIME         = "disable_sleep",
                p_RST_MODE_A          = "SYNC",
                p_RST_MODE_B          = "SYNC",

                # Port A (write).
                p_ADDR_WIDTH_A        = aw,
                p_BYTE_WRITE_WIDTH_A  = 8,
                p_WRITE_DATA_WIDTH_A  = 256,
                i_clka                = ClockSignal("sys"),
                i_ena                 = 1,
                i_wea                 = port_w.we,
                i_addra               = port_w.adr,
                i_dina                = port_w.dat_w,

                # Port B (read).
                p_ADDR_WIDTH_B        = aw,
                p_READ_DATA_WIDTH_B   = 256,
                p_READ_LATENCY_B      = 1,
                p_READ_RESET_VALUE_B  = "0",
                p_WRITE_MODE_B        = "read_first",
                i_clkb                = ClockSignal("sys"),
                i_enb                 = 1,
                i_addrb               = port_r.adr,
                o_doutb               = port_r.dat_r,
                i_rstb                = 0,
                i_regceb              = 1,

                # Unused.
                i_sleep               = 0,
                i_injectsbiterra      = 0,
                i_injectdbiterra      = 0,
            )
        else:
            self.mem = mem = Memory(256, depth, init=init, name="trixium_scratchpad")
            self.specials += mem
            port_r = mem.get_port()
            port_w = mem.get_port(write_capable=True, we_granularity=8)
            self.specials += port_r, port_w

        # Engine-facing facades (attribute-compatible with Memory ports).
        self.fill = _Facade()
        self.fill.adr   = Signal(max=depth)
        self.fill.dat_r = Signal(256)
        self.gather = _Facade()
        self.gather.adr   = Signal(max=depth)
        self.gather.dat_w = Signal(256)
        self.gather.we    = Signal(8) # int32 lane enables (expanded to bytes below).

        # Sideband slots: 0 = wishbone (priority), 1 = CSR peek/poke. Each carries one
        # 32-bit access.
        self._sb_req  = [Signal(           name=f"sb{i}_req")  for i in range(2)]
        self._sb_we   = [Signal(           name=f"sb{i}_we")   for i in range(2)]
        self._sb_adr  = [Signal(max=depth, name=f"sb{i}_adr")  for i in range(2)]
        self._sb_sel  = [Signal(3,         name=f"sb{i}_sel")  for i in range(2)]
        self._sb_bsel = [Signal(4,         name=f"sb{i}_bsel") for i in range(2)]
        self._sb_wdat = [Signal(32,        name=f"sb{i}_wdat") for i in range(2)]
        self._sb_gnt  = [Signal(           name=f"sb{i}_gnt")  for i in range(2)]
        self._sb_rdat = Signal(256) # port_r.dat_r, valid the cycle after a granted read.

        # Posted-write FIFO: two push slots (0 = wishbone, priority; 1 = CSR poke).
        aw = len(self._sb_adr[0])
        self._sbw_req  = [Signal(    name=f"sbw{i}_req")  for i in range(2)]
        self._sbw_ok   = [Signal(    name=f"sbw{i}_ok")   for i in range(2)]
        self._sbw_adr  = [Signal(aw, name=f"sbw{i}_adr")  for i in range(2)]
        self._sbw_sel  = [Signal(3,  name=f"sbw{i}_sel")  for i in range(2)]
        self._sbw_bsel = [Signal(4,  name=f"sbw{i}_bsel") for i in range(2)]
        self._sbw_wdat = [Signal(32, name=f"sbw{i}_wdat") for i in range(2)]
        sbw = SyncFIFO(width=aw + 3 + 4 + 32, depth=4)
        self += sbw
        dr_adr  = Signal(aw)
        dr_sel  = Signal(3)
        dr_bsel = Signal(4)
        dr_wdat = Signal(32)
        self.comb += [
            self._sbw_ok[0].eq(self._sbw_req[0] & sbw.writable),
            self._sbw_ok[1].eq(self._sbw_req[1] & ~self._sbw_req[0] & sbw.writable),
            sbw.we.eq(self._sbw_req[0] | self._sbw_req[1]),
            sbw.din.eq(Mux(self._sbw_req[0],
                Cat(self._sbw_adr[0], self._sbw_sel[0], self._sbw_bsel[0], self._sbw_wdat[0]),
                Cat(self._sbw_adr[1], self._sbw_sel[1], self._sbw_bsel[1], self._sbw_wdat[1]))),
            Cat(dr_adr, dr_sel, dr_bsel, dr_wdat).eq(sbw.dout),
            self.sbw_pending.eq(sbw.readable),
            sbw.re.eq(self.sbw_gnt),
        ]

        # Port muxing: posted-write drain, sideband reads, engine fill/gather.
        gather_bwe = Signal(32)
        self.comb += [
            gather_bwe.eq(Cat(*[Replicate(self.gather.we[i], 4) for i in range(8)])),
            self._sb_gnt[0].eq(self._sb_req[0] & (self.units_idle | self.sbr_allow)
                               & ~sbw.readable),
            self._sb_gnt[1].eq(self._sb_req[1] & (self.units_idle | self.sbr_allow)
                               & ~sbw.readable & ~self._sb_req[0]),
            self.sbr_gnt_any.eq(self._sb_gnt[0] | self._sb_gnt[1]),
            self.fill.dat_r.eq(port_r.dat_r),
            self._sb_rdat.eq(port_r.dat_r),
            # W port: a granted drain outranks everything (the integrator only grants
            # when the vector store slot is free and the gather FIFO has headroom).
            If(self.sbw_gnt & sbw.readable,
                port_w.adr.eq(dr_adr),
                port_w.dat_w.eq(Replicate(dr_wdat, 8)),
                Case(dr_sel, {i: port_w.we.eq(dr_bsel << (4*i)) for i in range(8)}),
                port_r.adr.eq(self.fill.adr),
            ).Else(
                port_w.adr.eq(self.gather.adr),
                port_w.dat_w.eq(self.gather.dat_w),
                port_w.we.eq(gather_bwe),
                # R port: granted reads take it for exactly one cycle; else fill.
                If(self._sb_gnt[0],
                    port_r.adr.eq(self._sb_adr[0]),
                ).Elif(self._sb_gnt[1],
                    port_r.adr.eq(self._sb_adr[1]),
                ).Else(
                    port_r.adr.eq(self.fill.adr),
                )
            )
        ]

    def add_csr_port(self):
        """CSR-based peek/poke (host access over the PCIe BAR0/JTAG CSR window). Write
        `adr` to trigger a read; poll `data_r` (valid once the sideband was granted). Write
        `data_w` to store 32 bits."""
        return _ScratchpadCSR(self)

    def add_wishbone_port(self):
        """32-bit byte-addressed Wishbone slave onto the 256-bit memory (CPU/host/JTAG
        access) via the sideband: accesses wait (wishbone-native stalls) for their slot."""
        bus = wishbone.Interface(data_width=32, address_width=32, addressing="byte")
        # Request and read data are registered for timing: the CPU's bus decode reached the
        # BRAM address pins combinationally. Costs 2 cycles per CPU access.
        # Reset-less: the request is latched in IDLE, the read data qualified by ack.
        adr_r  = Signal(len(self._sb_adr[0]), reset_less=True)
        sel_r  = Signal(3,  reset_less=True)
        bsel_r = Signal(4,  reset_less=True)
        wdat_r = Signal(32, reset_less=True)
        rdat_r = Signal(32, reset_less=True)
        self.comb += [
            self._sb_adr[0].eq(adr_r),
            self._sb_sel[0].eq(sel_r),
            self._sb_bsel[0].eq(bsel_r),
            self._sb_wdat[0].eq(wdat_r),
            # Writes are posted through the sbw FIFO (slot 0); _sb_we[0] stays 0.
            self._sbw_adr[0].eq(adr_r),
            self._sbw_sel[0].eq(sel_r),
            self._sbw_bsel[0].eq(bsel_r),
            self._sbw_wdat[0].eq(wdat_r),
            bus.dat_r.eq(rdat_r),
        ]
        self.bus_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(bus.cyc & bus.stb,
                NextValue(adr_r,  bus.adr[5:]),   # 32-byte beat index.
                NextValue(sel_r,  bus.adr[2:5]),  # 32-bit word within the beat.
                NextValue(bsel_r, bus.sel),
                NextValue(wdat_r, bus.dat_w),
                If(bus.we,
                    NextState("WPUSH"),
                ).Else(
                    NextState("REQ"),
                )
            )
        )
        fsm.act("WPUSH",
            # Posted: ack as soon as the write is in the sbw FIFO.
            self._sbw_req[0].eq(1),
            If(self._sbw_ok[0],
                NextState("ACK"),
            )
        )
        fsm.act("REQ",
            self._sb_req[0].eq(1),
            If(self._sb_gnt[0],
                NextState("READ"), # One cycle of read latency.
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

# Scratchpad CSR -----------------------------------------------------------------------------------

class _ScratchpadCSR(LiteXModule):
    """CSR peek/poke onto the scratchpad sideband (slot 1)."""
    def __init__(self, scratchpad):
        self.adr    = CSRStorage(24, description="Byte address (4-aligned). Writing triggers a read.")
        self.data_r = CSRStatus(32,  description="Read data for the last triggered read.")
        self.data_w = CSRStorage(32, description="Writing stores 32 bits at adr.")

        # # #

        sp         = scratchpad
        sel_r      = Signal(3, reset_less=True) # Registered every cycle.
        rd_p       = Signal() # Read pending (adr written, not yet granted).
        wr_p       = Signal() # Write pending.
        cap        = Signal() # Capture read data this cycle.
        reset_read = C(0, 1)  # Read-reset hook, tied off (kept: the generated Verilog depends on it).
        self.comb += [
            sp._sb_adr[1].eq(self.adr.storage[5:]),
            sp._sb_sel[1].eq(self.adr.storage[2:5]),
            sp._sb_bsel[1].eq(0xF),
            sp._sb_req[1].eq(rd_p & ~reset_read),
            sp._sbw_adr[1].eq(self.adr.storage[5:]),
            sp._sbw_sel[1].eq(self.adr.storage[2:5]),
            sp._sbw_bsel[1].eq(0xF),
            sp._sbw_wdat[1].eq(self.data_w.storage),
            sp._sbw_req[1].eq(wr_p),
        ]
        self.sync += [
            If(reset_read,
                rd_p.eq(0),
            ).Elif(self.adr.re,
                rd_p.eq(1),
            ).Elif(sp._sb_gnt[1] & ~wr_p,
                rd_p.eq(0),
            ),
            If(self.data_w.re,
                wr_p.eq(1),
            ).Elif(sp._sbw_ok[1],
                wr_p.eq(0),
            ),
            cap.eq(sp._sb_gnt[1] & ~wr_p & ~reset_read),
            sel_r.eq(sp._sb_sel[1]),
            If(cap & ~reset_read,
                Case(sel_r, {i: self.data_r.status.eq(sp._sb_rdat[32*i:32*(i+1)])
                             for i in range(8)}),
            ),
        ]

# Scratchpad Arbiter -------------------------------------------------------------------------------

class ScratchpadArbiter(LiteXModule):
    """Flat priority arbiter for the shared scratchpad client port.

    `base` is the vector unit (None without one); among `clients` (in attach order) the last
    busy one wins, else the base. Each client gets a one-hot select from a priority reduction
    over the busy bits and the outputs are one AND-OR level, so the logic depth on the BRAM
    address/data/enable pins grows with log2 of the client count (a serial mux chain failed
    timing).
    """
    def __init__(self, base, clients):
        clients = list(clients)
        ref = base if base is not None else clients[0]
        self.busy      = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal.like(ref.sp_rd_adr)
        self.sp_rd_dat = Signal.like(ref.sp_rd_dat)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal.like(ref.sp_wr_adr)
        self.sp_wr_we  = Signal.like(ref.sp_wr_we)
        self.sp_wr_dat = Signal.like(ref.sp_wr_dat)
        self.sp_wr_gnt = Signal()
        # Write room, driven by the integrator (its grant without the request): a client's
        # grant is room & select & its own request - the same as sp_wr_gnt & select, as the
        # integrator grants sp_wr_req & room and the selects are one-hot, but no client's
        # request reaches another client's grant (a 24-level cross-client chain on the K7).
        self.sp_wr_room = Signal()
        self.aux_busy  = Signal() # Any attached (non-base) client busy.

        # # #

        # One-hot select: client i wins when busy and no LATER client is busy.
        n = len(clients)
        later    = [Signal(name=f"arb_later{i}") for i in range(n)]
        sel      = [Signal(name=f"arb_sel{i}")   for i in range(n)]
        base_sel = Signal()
        for i in range(n):
            self.comb += later[i].eq(reduce(or_, [c.busy for c in clients[i + 1:]], C(0, 1)))
            self.comb += sel[i].eq(clients[i].busy & ~later[i])
        any_aux = reduce(or_, [c.busy for c in clients], C(0, 1))
        self.comb += [
            self.aux_busy.eq(any_aux),
            base_sel.eq(~any_aux if base is not None else 0),
            self.busy.eq((base.busy if base is not None else 0) | any_aux),
        ]

        # One AND-OR level per output.
        def pick(field):
            terms  = [Mux(base_sel, getattr(base, field), 0)] if base is not None else []
            terms += [Mux(sel[i], getattr(clients[i], field), 0) for i in range(n)]
            return reduce(or_, terms)

        for field in ("sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat"):
            self.comb += getattr(self, field).eq(pick(field))
        if base is not None:
            self.comb += [
                base.sp_rd_dat.eq(self.sp_rd_dat),
                base.sp_wr_gnt.eq(self.sp_wr_room & base_sel & base.sp_wr_req),
            ]
        for i, c in enumerate(clients):
            self.comb += [c.sp_rd_dat.eq(self.sp_rd_dat),
                          c.sp_wr_gnt.eq(self.sp_wr_room & sel[i] & c.sp_wr_req)]

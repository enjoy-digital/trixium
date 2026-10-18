#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""DDR4 attachment of the engine (the XCU1525): channel ports and the split `ddr` domain.

Engine ports. An engine channel's 256-bit AXI reaches a LiteDRAM controller in one of two
ways, measured on the real LiteDRAM core in simulation:

- a 256-bit crossbar port: LiteDRAM's native up-converter groups two 256-bit commands into
  one 512-bit controller word through a five-state FSM - ~5 cycles per word even with reads
  pipelined (LiteDRAMNativePortUpConverter read_depth). 2.55
  cycles/beat for a lone stream, 4.3 with four ports on one controller;
- a 512-bit crossbar port behind LiteX's AXIUpConverter: every 256-bit beat becomes a
  single 512-bit transaction on the lane its address selects. Twice the DRAM traffic, but
  the bridge issues a controller command per cycle: 1.31 cycles/beat alone, 2.9 with four
  ports. engine_ddr_port uses the 512-bit port and, by default, merges the read bursts two
  beats per word (AXIWide) to avoid the doubled read traffic.

DDR domain. add_ddr_domain_controllers places the controllers in their own `ddr` clock
domain (the SoC's add_sdram, by hand). It is shared by the XCU1525 target (--ddr-clk-freq,
USPDDRPHYs) and the simulation (trixium_sim.py --board xcu1525, LiteDRAM's PHY model on a second sim clock), so
the split SoC boots in simulation before it goes on the board.

Each core is renamed {sys: ddr, sys_port: sys}: its crossbar ports, requested with
clock_domain="sys_port", get LiteDRAM's native-port CDC between the two. Controller 0 also
carries main_ram (the CPU's direct memory buses and the L2 + wishbone bridge, as add_sdram
connects them). The cores' and PHYs' CSRs live on their own bank array in the ddr domain,
reached through a wishbone crossing at a CSR window; its regions are registered under the
usual names (the BIOS, firmware/ddr.c and the host trainer find sdram*/ddrphy* where
csr.h / csr.csv say).
"""

import math
from types import SimpleNamespace
from operator import or_
from functools import reduce

from migen import *
from migen.fhdl.simplify import FullMemoryWE

from litex.gen import LiteXModule

from litex.soc.interconnect import csr_bus, wishbone, axi, stream
from litex.soc.interconnect.csr import CSR, CSRStatus, CSRStorage
from litex.soc.interconnect.axi import AXIInterface, AXIUpConverter
from litex.soc.integration.soc import SoCRegion, SoCCSRRegion

from litedram.core import LiteDRAMCore
from litedram.frontend.axi import LiteDRAMAXI2Native
from litedram.frontend.wishbone import LiteDRAMWishbone2Native
from litedram.frontend.bist import LiteDRAMBISTGenerator, LiteDRAMBISTChecker

from gateware.memory.axi import AXIWide

# AXI Crossing -------------------------------------------------------------------------------------

class AXICrossing(LiteXModule):
    """An AXI interface from cd_from to cd_to: LiteX's AXIClockDomainCrossing (one buffered async
    FIFO per channel), and with register=True a register slice (valid and ready piped) on each
    request channel (AW, W, AR) on the cd_to side.

    The split DDR domain crosses here, not with the crossbar's native-port CDC: that one's
    read-data FIFO (16 words) has no backpressure - the controller cannot be stalled - so a
    LiteDRAMAXI2Native holding more reads in flight than it (the engine ports: 256) loses
    data once the ddr clock outruns sys. The AXI R channel has a ready; the bridge then sits
    in the ddr domain on a native port with no CDC at all.

    The slices take the FIFOs' block RAMs off the bridge's paths: a FIFO's output is its RAM's
    clock-to-out, and its ready computes the RAM's next read address. Without them, a bridge
    with read-modify-write runs a W FIFO's strobes through its partial-write detect, the RMW
    request and the command into the AW FIFO's read address in one ddr cycle."""
    def __init__(self, master, slave, cd_from, cd_to, depth=16, register=False):
        # # #

        mid = slave
        if register:
            mid = AXIInterface(data_width=slave.data_width, address_width=slave.address_width,
                               id_width=slave.id_width, version=slave.version)
        self.cdc = axi.AXIClockDomainCrossing(master, mid, cd_from, cd_to, depth=depth,
                                              buffered=True)
        if register:
            for ch in ("aw", "w", "ar"):
                buf = stream.Buffer(getattr(mid, ch).description, pipe_valid=True, pipe_ready=True)
                self.add_module(f"{ch}_slice", ClockDomainsRenamer(cd_to)(buf))
                self.comb += [
                    getattr(mid, ch).connect(buf.sink),
                    buf.source.connect(getattr(slave, ch)),
                ]
            self.comb += [
                slave.b.connect(mid.b),
                slave.r.connect(mid.r),
            ]

# Engine DDR Port ----------------------------------------------------------------------------------

def engine_ddr_port(core, axi, port_width=512, merge=True, w_buffer_depth=64,
                    r_buffer_depth=256, clock_domain="sys", rmw=False):
    """Return the modules (to register as submodules) that connect `axi` (the engine channel's
    256-bit AXI) to a `port_width`-bit crossbar port of `core`.

    merge:        on a wider port, merge the read bursts two beats per word (AXIWide) instead of
                  one wide transaction per beat (LiteX's AXIUpConverter, twice the traffic).
    clock_domain: the port's side when the controller runs in its own `ddr` domain (the AXI
                  then crosses with AXICrossing; see the XCU1525 target's --ddr-clk-freq).
    rmw:          read-modify-write the partially-strobed words (a memory without DM pins, the
                  YPCB's DDR3: the 256-bit beats reach the 512-bit word with half strobes, and
                  KV rows of an odd number of beats share words).
    """
    split = clock_domain != "sys"
    # Split: a native port on the controller's own side (the core renames its sys to ddr),
    # the bridge there too, and the AXI crossing into it (AXICrossing).
    port = core.crossbar.get_port(data_width=port_width)
    mods = []
    wide = axi
    if port_width != len(axi.r.data):
        wide = AXIInterface(data_width=port_width, address_width=axi.address_width,
                            id_width=axi.id_width)
        if merge:
            mods.append(AXIWide(axi, wide))
        else:
            mods.append(AXIUpConverter(axi, wide))
    if split:
        far = AXIInterface(data_width=port_width, address_width=axi.address_width,
                           id_width=axi.id_width, version=wide.version)
        mods.append(AXICrossing(wide, far, "sys", "ddr", register=True))
        wide = far
    bridge = LiteDRAMAXI2Native(wide, port, w_buffer_depth=w_buffer_depth,
                                r_buffer_depth=r_buffer_depth, with_read_modify_write=rmw)
    mods.append(ClockDomainsRenamer("ddr")(bridge) if split else bridge)
    return mods

# PHY Delay Controls -------------------------------------------------------------------------------

def register_delay_controls(phy):
    """Register the CE of each IDELAYE3/ODELAYE3 of a LiteDRAM US/US+ PHY next to it.

    The PHY drives them with `dly_sel & csr.wr_stb`: decoded straight from the CSR bus address,
    one net to the ~170 delay elements (data, strobes, clock, commands) over the channel's two
    or three I/O banks, 5 ns of route at the DDR clock that no CSR pipelining can shorten. One
    register per element (in the PHY's own domain, so call this before renaming it) lets each
    sit by its delay; the increment lands one cycle after the CSR write, which the calibration
    never sees (the PHY's own tap counters count the strobe itself). Returns the number of CEs registered."""
    n = 0
    for special in phy._fragment.specials:
        if not isinstance(special, Instance) or special.of not in ("IDELAYE3", "ODELAYE3"):
            continue
        for item in special.items:
            if isinstance(item, Instance.Input) and item.name == "CE" and not isinstance(item.expr, Constant):
                ce = Signal(name_override=f"{special.of.lower()}_ce_q{n}")
                phy.sync += ce.eq(item.expr)
                item.expr = ce
                n += 1
    return n

# DDR CSR ------------------------------------------------------------------------------------------

class PipelinedDDRCSR(LiteXModule):
    """Wishbone -> the ddr-domain CSR banks, one registered bus copy per controller.

    A shared CSR bus decoded straight into the PHYs' IDELAY/ODELAY CE pins of four
    controllers over three SLRs does not meet timing at the DDR clock (long routes). Here each
    controller's banks get their own `hops` registered copies of adr/we/re/dat_w and as many
    registered stages of dat_r: with two, one copy can sit by the shared logic and one by the
    PHY's I/O in its own SLR. An access takes 2*hops + 2 cycles (the controllers' CSRs are only
    touched by calibration and the BIOS)."""
    def __init__(self, wb, groups, address_width, data_width, hops=2):
        assert wb.data_width == data_width and wb.addressing == "word"

        # # #

        # Registered request, `hops` copies per controller group; registered read data back.
        # The address and data are reset-less (qualified by we/re), the strobes keep their reset.
        adr, dat_w = Signal(address_width, reset_less=True), Signal(data_width, reset_less=True)
        we, re     = Signal(), Signal()
        rdats      = []
        for buses in groups:
            src = SimpleNamespace(adr=adr, dat_w=dat_w, we=we, re=re)
            for h in range(hops):
                s = csr_bus.Interface(address_width=address_width, data_width=data_width)
                s.adr.reset_less = s.dat_w.reset_less = True
                self.sync += [s.adr.eq(src.adr), s.dat_w.eq(src.dat_w), s.we.eq(src.we),
                              s.re.eq(src.re)]
                src = s
            for b in buses:
                self.comb += [b.adr.eq(s.adr), b.dat_w.eq(s.dat_w), b.we.eq(s.we), b.re.eq(s.re)]
            rd = reduce(or_, [b.dat_r for b in buses])
            for h in range(hops):
                rd_h = Signal(data_width, reset_less=True)
                self.sync += rd_h.eq(rd)
                rd = rd_h
            rdats.append(rd)
        rdat = Signal(data_width)
        self.comb += rdat.eq(reduce(or_, rdats))

        # Wishbone FSM: request, 2*hops + 1 cycles of pipeline (adr, the hops out, the bank's
        # registered dat_r, the hops back), ack with the registered read data.
        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(wb.cyc & wb.stb,
                NextValue(adr,   wb.adr),
                NextValue(dat_w, wb.dat_w),
                NextValue(we,    wb.we  & (wb.sel != 0)),
                NextValue(re,    ~wb.we & (wb.sel != 0)),
                NextState("WAIT"),
            )
        )
        wait = Signal(max=2*hops + 2)
        fsm.act("WAIT",
            NextValue(we, 0),
            NextValue(re, 0),
            NextValue(wait, wait + 1),
            If(wait == 2*hops,
                NextValue(wait, 0),
                NextState("ACK"),
            )
        )
        fsm.act("ACK",
            wb.ack.eq(1),
            wb.dat_r.eq(rdat),
            NextState("IDLE"),
        )

class _DDRCSRHolder(Module):
    """A plain Module: the SoC's CSR scan does not look inside it, the ddr-domain
    CSRBankArray does (its attributes sdram, ddrphy, sdram1, ... are the bank names)."""

# DDR Statistics -----------------------------------------------------------------------------------

class DFIStats(LiteXModule):
    """Command statistics of one controller's DFI, in its clock: the cycles with a read, a
    write, an activate, a precharge or a refresh on any phase, and the cycles with no command
    at all. `snap` latches the counts for the CPU and restarts them, so two snaps bracket a
    workload: the accounting of the cycles the engine's streams did not get a beat in (refresh,
    row switches, nothing to issue)."""
    def __init__(self, dfi):
        self.snap   = CSR()
        self.cycles = CSRStatus(32)
        self.reads  = CSRStatus(32)
        self.writes = CSRStatus(32)
        self.acts   = CSRStatus(32)
        self.pres   = CSRStatus(32)
        self.refs   = CSRStatus(32)
        self.idle   = CSRStatus(32)

        # # #

        def on_any_phase(ras_n, cas_n, we_n):
            # Active-low pins: a rank selected and this command on the phase.
            return reduce(or_, [(p.cs_n != 2**len(p.cs_n) - 1) & (p.ras_n == ras_n) &
                                (p.cas_n == cas_n) & (p.we_n == we_n) for p in dfi.phases])
        # Every increment a 1-bit Signal: an expression such as ~(a | b) widens to the adder's
        # 32 bits in the emitted Verilog (the Migen width pitfall) and counts garbage.
        rd, wr, act, pre, ref, none = [Signal() for _ in range(6)]
        self.comb += [
            rd.eq(on_any_phase(1, 0, 1)),
            wr.eq(on_any_phase(1, 0, 0)),
            act.eq(on_any_phase(0, 1, 1)),
            pre.eq(on_any_phase(0, 1, 0)),
            ref.eq(on_any_phase(0, 0, 1)),
            none.eq(~(rd | wr | act | pre | ref)),
        ]
        for status, inc in ((self.cycles, 1), (self.reads, rd), (self.writes, wr), (self.acts, act),
                            (self.pres, pre), (self.refs, ref), (self.idle, none)):
            count = Signal(32)
            self.sync += If(self.snap.re,
                status.status.eq(count), count.eq(0),
            ).Else(
                count.eq(count + inc),
            )

class EnginePortStats(LiteXModule):
    """Per engine channel, in the engine's clock: the read beats it received, the cycles its
    read requests waited on the memory side, the cycles read data waited on the engine, the
    write beats it sent. `sel` picks the channel the counts show; `snap` latches every
    channel's and restarts them (two snaps bracket a workload)."""
    def __init__(self, axis):
        n = len(axis)
        self.snap    = CSR()
        self.sel     = CSRStorage(max(1, (n - 1).bit_length()))
        self.cycles  = CSRStatus(32)
        self.beats   = CSRStatus(32)   # Read beats delivered to the channel.
        self.ar_wait = CSRStatus(32)   # Cycles a read request waited (memory side busy).
        self.r_wait  = CSRStatus(32)   # Cycles read data waited (engine side busy).
        self.wbeats  = CSRStatus(32)   # Write beats sent.

        # # #

        cycles = Signal(32)
        self.sync += If(self.snap.re, self.cycles.status.eq(cycles), cycles.eq(0)).Else(cycles.eq(cycles + 1))
        holds = {name: [] for name in ("beats", "ar_wait", "r_wait", "wbeats")}
        for axi_if in axis:
            for name, expr in (("beats",   axi_if.r.valid & axi_if.r.ready),
                               ("ar_wait", axi_if.ar.valid & ~axi_if.ar.ready),
                               ("r_wait",  axi_if.r.valid & ~axi_if.r.ready),
                               ("wbeats",  axi_if.w.valid & axi_if.w.ready)):
                inc, count, hold = Signal(), Signal(32), Signal(32)   # inc 1-bit: see DFIStats
                self.comb += inc.eq(expr)
                self.sync += If(self.snap.re, hold.eq(count), count.eq(0)).Else(count.eq(count + inc))
                holds[name].append(hold)
        for name, regs in holds.items():
            self.comb += getattr(self, name).status.eq(Array(regs)[self.sel.storage])

# DDR Domain Controllers ---------------------------------------------------------------------------

def add_ddr_domain_controllers(soc, phys, modules, clk_freq, main_ram_size, l2_size,
                               csr_window, phy_renames={"sys": "ddr"}, controller_settings=None,
                               with_bist=False, with_ecc=False, with_stats=False):
    """Add one LiteDRAM controller per phy/module in the `ddr` domain (controller 0 carries
    main_ram). Fills soc.sdrams and soc.ddr_csr_stride (controller k's CSR pages start at
    controller 0's + k*stride); returns the per-controller memory sizes.

    controller_settings: a function returning the ControllerSettings of a controller (one
                         instance each), None for LiteDRAM's defaults.
    with_bist:           a LiteDRAM BIST generator/checker per controller on its own crossbar
                         port (sdram_generator{k}/sdram_checker{k}), for the calibration's
                         traffic test; their CSR pages follow the controller's two, so
                         controller k's CSRs sit at controller 0's + k*4 pages.
    with_ecc:            LiteDRAM's DFI ECC (SECDED on each PHY's ECC byte lane, 72-bit beats;
                         the controllers see 64-bit beats, the calibration the raw 72).
    with_stats:          DFI command statistics per controller (DFIStats, sdram_stats{k}: one
                         more CSR page per controller), the console's `ddrstat`.
    """
    n            = len(phys)
    holder       = _DDRCSRHolder()
    memory_sizes = []

    # Controllers.
    for i, (phy, module) in enumerate(zip(phys, modules)):
        sfx = "" if i == 0 else str(i)
        g   = module.geom_settings
        core_kwargs = {"with_ecc": True} if with_ecc else {}
        if controller_settings is not None:
            core_kwargs["controller_settings"] = controller_settings()
        core = LiteDRAMCore(phy=phy, geom_settings=module.geom_settings,
                            timing_settings=module.timing_settings, clk_freq=clk_freq,
                            **core_kwargs)
        memory_sizes.append((1 << (g.bankbits + g.rowbits + g.colbits)) *
                            core.controller.settings.phy.nranks*core.controller.settings.phy.databits//8)
        setattr(holder, f"ddrphy{sfx}", phy)
        setattr(holder, f"sdram{sfx}", core)
        soc.submodules += ClockDomainsRenamer(phy_renames)(phy)
        soc.submodules += ClockDomainsRenamer({"sys": "ddr", "sys_port": "sys"})(core)
        soc.sdrams.append(core)
    if with_bist:
        for i, core in enumerate(soc.sdrams):
            sfx = "" if i == 0 else str(i)
            generator = LiteDRAMBISTGenerator(core.crossbar.get_port())
            checker   = LiteDRAMBISTChecker(core.crossbar.get_port())
            setattr(holder, f"sdram_generator{sfx}", generator)
            setattr(holder, f"sdram_checker{sfx}",   checker)
            soc.submodules += ClockDomainsRenamer("ddr")(generator)
            soc.submodules += ClockDomainsRenamer("ddr")(checker)
    if with_stats:
        for i, phy in enumerate(phys):
            sfx = "" if i == 0 else str(i)
            stats = DFIStats(phy.dfi)
            setattr(holder, f"sdram_stats{sfx}", stats)
            soc.submodules += ClockDomainsRenamer("ddr")(stats)
    soc.submodules += holder
    core0 = soc.sdrams[0]
    # The builder writes sdram_phy.h from soc.sdram.controller.settings; the CSR bridge's
    # with_register follows hasattr(soc, "sdram"). A stand-in, invisible to the CSR scan.
    soc.sdram = SimpleNamespace(controller=core0.controller, crossbar=core0.crossbar)

    # main_ram on controller 0 (add_sdram's connection, the ports crossing into sys).
    sdram_size = min(memory_sizes[0], main_ram_size)
    soc.bus.add_region("main_ram", SoCRegion(origin=soc.mem_map["main_ram"], size=sdram_size,
                                             mode="rwx"))
    base = soc.bus.regions["main_ram"].origin
    if hasattr(soc.cpu, "add_memory_buses"):
        soc.cpu.add_memory_buses(address_width=32, data_width=core0.crossbar.controller.data_width)
    for mem_bus in soc.cpu.memory_buses:
        # The AXI crosses (AXICrossing: the R channel has backpressure, the native port CDC's
        # read data has none); the bridge sits on a native port in ddr.
        port = core0.crossbar.get_port()
        assert isinstance(mem_bus, axi.AXIInterface) and mem_bus.data_width == port.data_width, \
            "the ddr-domain main_ram expects the CPU's AXI memory bus at the controller's width"
        far = axi.AXIInterface(data_width=mem_bus.data_width,
                               address_width=mem_bus.address_width,
                               id_width=mem_bus.id_width, version=mem_bus.version)
        soc.submodules += AXICrossing(mem_bus, far, "sys", "ddr")
        soc.submodules += ClockDomainsRenamer("ddr")(
            LiteDRAMAXI2Native(axi=far, port=port, base_address=base))
    if (not len(soc.cpu.memory_buses)) or not hasattr(soc.cpu, "dma_bus"):
        port = core0.crossbar.get_port(clock_domain="sys_port")
        wb_sdram = wishbone.Interface(data_width=soc.bus.data_width, address_width=32,
                                      addressing="word")
        soc.bus.add_slave(name="main_ram", slave=wb_sdram)
        l2_size = 2**int(math.log2(max(l2_size, int(2*port.data_width/8))))
        # reverse=False: word i of a line at bits [32*i +: 32], as the AXI paths (the CPU's
        # memory bus, the engine) and LiteX's add_sdram have it; the default reversed the words
        # of every line between the bus (PCIe/JTAG) and the DRAM.
        l2 = wishbone.Cache(cachesize=l2_size//4, master=wb_sdram,
            slave=wishbone.Interface(data_width=max(port.data_width, 128), address_width=32,
                                     addressing="word"),
            reverse=False)
        soc.l2_cache = FullMemoryWE()(l2)
        soc.add_config("L2_SIZE", l2_size)
        soc.wishbone_bridge = LiteDRAMWishbone2Native(wishbone=soc.l2_cache.slave, port=port,
                                                      base_address=base)

    # The ddr-domain CSR bus: per controller, in order, sdram{k}, ddrphy{k}, with the BIST its
    # generator and checker, with the statistics sdram_stats{k}; controller k's pages start at
    # k*per (liblitedram's SDRAM_CSR_STRIDE = per pages).
    names = (["sdram", "ddrphy"] + (["sdram_generator", "sdram_checker"] if with_bist else [])
             + (["sdram_stats"] if with_stats else []))
    per   = len(names)
    pages = {}
    for i in range(n):
        sfx = "" if i == 0 else str(i)
        for j, name in enumerate(names):
            pages[f"{name}{sfx}"] = per*i + j
    bank = csr_bus.CSRBankArray(holder, lambda name, memory: pages[name],
        data_width=soc.csr.data_width, address_width=soc.csr.address_width,
        alignment=soc.csr.alignment, paging=soc.csr.paging, ordering=soc.csr.ordering)
    wb = wishbone.Interface(data_width=soc.bus.data_width, address_width=32, addressing="word")
    assert not bank.srams
    by_page = {mapaddr: b for (_, _, mapaddr, b) in bank.banks}
    # (LiteDRAM's PHY model has no CSRs: its ddrphy page is then empty.)
    groups = [[by_page[p].bus for p in range(per*i, per*i + per) if p in by_page] for i in range(n)]
    bridge = PipelinedDDRCSR(wb, groups, soc.csr.address_width, soc.csr.data_width)
    soc.submodules += ClockDomainsRenamer("ddr")(bank)
    soc.submodules += ClockDomainsRenamer("ddr")(bridge)
    window = soc.csr.paging*per*n
    soc.bus.add_slave(name="ddrcsr", slave=wb,
        region=SoCRegion(origin=csr_window, size=max(window, 0x10000), cached=False),
        clock_domain="ddr")
    for name, csrs, mapaddr, rmap in bank.banks:
        soc.csr.add_region(name, SoCCSRRegion(origin=csr_window + soc.csr.paging*mapaddr,
                                              busword=soc.csr.data_width, obj=csrs))
    for name, constant in bank.constants:
        soc.add_constant(name + "_" + constant.name, constant.value.value)
    soc.ddr_csr_stride = soc.csr.paging*per
    return memory_sizes

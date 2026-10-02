#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""DDR controllers in their own `ddr` clock domain (the SoC's add_sdram, by hand).

Shared by the XCU1525 target (--ddr-clk-freq, USPDDRPHYs) and the simulation (--ddr-split,
LiteDRAM's PHY model on a second sim clock), so the split SoC boots in simulation before it
goes on the board.

Each core is renamed {sys: ddr, sys_port: sys}: its crossbar ports, requested with
clock_domain="sys_port", get LiteDRAM's native-port CDC between the two. Controller 0 also
carries main_ram (the CPU's direct memory buses and the L2 + wishbone bridge, as add_sdram
connects them). The cores' and PHYs' CSRs live on their own bank array in the ddr domain,
reached through a wishbone crossing at a CSR window; its regions are registered under the
usual names (the BIOS, firmware/ddr.c and the host trainer find sdram*/ddrphy* where
csr.h / csr.csv say).
"""

import math
from functools import reduce
from operator import or_
from types import SimpleNamespace

from migen import *
from migen.fhdl.simplify import FullMemoryWE

from litex.soc.interconnect import csr_bus, wishbone, axi
from litex.soc.integration.soc import SoCRegion, SoCCSRRegion

from litedram.core import LiteDRAMCore
from litedram.frontend.axi import LiteDRAMAXI2Native
from litedram.frontend.wishbone import LiteDRAMWishbone2Native


class PipelinedDDRCSR(Module):
    """Wishbone -> the ddr-domain CSR banks, one registered bus copy per controller.

    v27 (ddr at 200 MHz): the shared CSR bus decoded straight into the PHYs' IDELAY/ODELAY CE
    pins of four controllers over three SLRs (worst -0.801 ns, 5.1 ns of route). Here each
    controller's banks get their own registered adr/we/re/dat_w and a registered dat_r, so
    every hop is one register to the next: an access takes 4 cycles instead of 2 (the
    controllers' CSRs are only touched by calibration and the BIOS)."""
    def __init__(self, wb, groups, address_width, data_width):
        adr, dat_w = Signal(address_width), Signal(data_width)
        we, re = Signal(), Signal()
        rdats = []
        for buses in groups:
            s = csr_bus.Interface(address_width=address_width, data_width=data_width)
            rd = Signal(data_width)
            self.sync += [s.adr.eq(adr), s.dat_w.eq(dat_w), s.we.eq(we), s.re.eq(re)]
            self.sync += rd.eq(reduce(or_, [b.dat_r for b in buses]))
            for b in buses:
                self.comb += [b.adr.eq(s.adr), b.dat_w.eq(s.dat_w), b.we.eq(s.we), b.re.eq(s.re)]
            rdats.append(rd)
        rdat = Signal(data_width)
        self.comb += rdat.eq(reduce(or_, rdats))
        assert wb.data_width == data_width and wb.addressing == "word"
        self.submodules.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(wb.cyc & wb.stb,
                NextValue(adr, wb.adr),
                NextValue(dat_w, wb.dat_w),
                NextValue(we, wb.we & (wb.sel != 0)),
                NextValue(re, ~wb.we & (wb.sel != 0)),
                NextState("S0")))
        fsm.act("S0", NextValue(we, 0), NextValue(re, 0), NextState("S1"))
        fsm.act("S1", NextState("S2"))
        fsm.act("S2", NextState("ACK"))
        fsm.act("ACK", wb.ack.eq(1), wb.dat_r.eq(rdat), NextState("IDLE"))


class _DDRCSRHolder(Module):
    """A plain Module: the SoC's CSR scan does not look inside it, the ddr-domain
    CSRBankArray does (its attributes sdram, ddrphy, sdram1, ... are the bank names)."""


def add_ddr_domain_controllers(soc, phys, modules, clk_freq, main_ram_size, l2_size,
                               csr_window, phy_renames={"sys": "ddr"}):
    """phys/modules: one per controller (controller 0 carries main_ram). Fills soc.sdrams;
    returns the per-controller memory sizes."""
    n = len(phys)
    holder = _DDRCSRHolder()
    memory_sizes = []
    for i, (phy, module) in enumerate(zip(phys, modules)):
        sfx = "" if i == 0 else str(i)
        g = module.geom_settings
        memory_sizes.append((1 << (g.bankbits + g.rowbits + g.colbits))*phy.settings.nranks*phy.settings.databits//8)
        core = LiteDRAMCore(phy=phy, geom_settings=module.geom_settings,
                            timing_settings=module.timing_settings, clk_freq=clk_freq)
        setattr(holder, f"ddrphy{sfx}", phy)
        setattr(holder, f"sdram{sfx}", core)
        soc.submodules += ClockDomainsRenamer(phy_renames)(phy)
        soc.submodules += ClockDomainsRenamer({"sys": "ddr", "sys_port": "sys"})(core)
        soc.sdrams.append(core)
    soc.submodules += holder
    core0 = soc.sdrams[0]
    # The builder writes sdram_phy.h from soc.sdram.controller.settings; the CSR bridge's
    # with_register follows hasattr(soc, "sdram"). A stand-in, invisible to the CSR scan.
    soc.sdram = SimpleNamespace(controller=core0.controller, crossbar=core0.crossbar)

    # main_ram on controller 0 (add_sdram's connection, the ports crossing into sys).
    sdram_size = min(memory_sizes[0], main_ram_size)
    soc.bus.add_region("main_ram", SoCRegion(origin=soc.mem_map["main_ram"], size=sdram_size, mode="rwx"))
    base = soc.bus.regions["main_ram"].origin
    if hasattr(soc.cpu, "add_memory_buses"):
        soc.cpu.add_memory_buses(address_width=32, data_width=core0.crossbar.controller.data_width)
    for mem_bus in soc.cpu.memory_buses:
        port = core0.crossbar.get_port(clock_domain="sys_port")
        assert isinstance(mem_bus, axi.AXIInterface) and mem_bus.data_width == port.data_width, \
            "the ddr-domain main_ram expects the CPU's AXI memory bus at the controller's width"
        soc.submodules += LiteDRAMAXI2Native(axi=mem_bus, port=port, base_address=base)
    if (not len(soc.cpu.memory_buses)) or not hasattr(soc.cpu, "dma_bus"):
        port = core0.crossbar.get_port(clock_domain="sys_port")
        wb_sdram = wishbone.Interface(data_width=soc.bus.data_width, address_width=32, addressing="word")
        soc.bus.add_slave(name="main_ram", slave=wb_sdram)
        l2_size = 2**int(math.log2(max(l2_size, int(2*port.data_width/8))))
        l2 = wishbone.Cache(cachesize=l2_size//4, master=wb_sdram,
            slave=wishbone.Interface(data_width=max(port.data_width, 128), address_width=32, addressing="word"))
        soc.l2_cache = FullMemoryWE()(l2)
        soc.add_config("L2_SIZE", l2_size)
        soc.wishbone_bridge = LiteDRAMWishbone2Native(wishbone=soc.l2_cache.slave, port=port, base_address=base)

    # The ddr-domain CSR bus: sdram{k} on page 2k, ddrphy{k} on 2k + 1.
    pages = {}
    for i in range(n):
        sfx = "" if i == 0 else str(i)
        pages[f"sdram{sfx}"], pages[f"ddrphy{sfx}"] = 2*i, 2*i + 1
    bank = csr_bus.CSRBankArray(holder, lambda name, memory: pages[name],
        data_width=soc.csr.data_width, address_width=soc.csr.address_width,
        alignment=soc.csr.alignment, paging=soc.csr.paging, ordering=soc.csr.ordering)
    wb = wishbone.Interface(data_width=soc.bus.data_width, address_width=32, addressing="word")
    assert not bank.srams
    by_page = {mapaddr: b for (_, _, mapaddr, b) in bank.banks}
    # (LiteDRAM's PHY model has no CSRs: its ddrphy page is then empty.)
    groups = [[by_page[p].bus for p in (2*i, 2*i + 1) if p in by_page] for i in range(n)]
    bridge = PipelinedDDRCSR(wb, groups, soc.csr.address_width, soc.csr.data_width)
    soc.submodules += ClockDomainsRenamer("ddr")(bank)
    soc.submodules += ClockDomainsRenamer("ddr")(bridge)
    window = soc.csr.paging*2*n
    soc.bus.add_slave(name="ddrcsr", slave=wb,
        region=SoCRegion(origin=csr_window, size=max(window, 0x10000), cached=False),
        clock_domain="ddr")
    for name, csrs, mapaddr, rmap in bank.banks:
        soc.csr.add_region(name, SoCCSRRegion(origin=csr_window + soc.csr.paging*mapaddr,
                                              busword=soc.csr.data_width, obj=csrs))
    for name, constant in bank.constants:
        soc.add_constant(name + "_" + constant.name, constant.value.value)
    return memory_sizes

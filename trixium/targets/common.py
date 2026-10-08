#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Target-side pieces shared by the Xilinx UltraScale+ boards.

The physical constraints of the TriXium core and the CPU bus monitor. The constraints are NOT a
timing waiver: new integrations need routed timing and selector coverage checks. In particular
a signal-name refactor is not physically neutral (the constraints match cell names).
"""

from migen import *

from litex.gen import LiteXModule

from litex.soc.interconnect.csr import CSRStatus

# Xilinx Constraints -------------------------------------------------------------------------------

def add_cluster_constraints(platform):
    # engreset holds reset for 64 sys cycles. Preserve setup=4 / hold=3 for the
    # reset launch net; firmware cannot submit during the reset/drain handshake.
    platform.add_platform_command(
        "set_multicycle_path -setup 4 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    platform.add_platform_command(
        "set_multicycle_path -hold 3 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    # The 2 Mbit shared scratchpad and the norm unit's weight buffer in BLOCK RAM. Left to
    # inference, part of the scratchpad goes to LUT RAM: every client address then fans out to
    # thousands of LUT-RAM cells across the die and those paths miss timing by nanoseconds.
    for name in ("trixium_scratchpad", "norm_wbuf"):
        platform.add_platform_command(
            "set_property RAM_STYLE block [get_cells -quiet -hierarchical -filter "
            "{{NAME =~ *" + name + "*_reg*}}]")
    # Preserve distributed vector imem: a BRAM output register changes the
    # decode launch's clock-to-out and placement. No vector multicycle paths.
    platform.add_platform_command(
        "set_property RAM_STYLE distributed "
        "[get_cells -hierarchical -filter {{NAME =~ *imem_reg*}}]")

# CPU Bus Monitor ----------------------------------------------------------------------------------

class CPUBusMonitor(LiteXModule):
    """Where is the CPU? Bus handshake counters and last addresses, readable over PCIe.

    When the CPU prints nothing while PCIe CSR access works (and JTAGBone is not available),
    these CSRs tell from the host (tools/cpumon.py) whether it is fetching, stalled on a bus
    access, or held in reset: AXI address/response handshake counts and the last addresses on
    the CPU's peripheral bus (ROM, SRAM, CSRs) and, when present, its direct memory bus
    (main_ram). A request count ahead of its response count, frozen across reads, is a stalled
    access.
    """
    def __init__(self, cpu):
        self.cpu_reset = CSRStatus(1, description="The CPU's reset input.")
        self.comb += self.cpu_reset.status.eq(cpu.reset)
        buses = [("p", cpu.pbus)] + [("m", b) for b in getattr(cpu, "memory_buses", [])[:1]]
        for name, bus in buses:
            for ch in ("ar", "r", "aw", "b"):
                cnt = CSRStatus(32, name=f"{name}{ch}", description=f"{name}Bus {ch} handshakes.")
                setattr(self, f"{name}{ch}", cnt)
                ep = getattr(bus, ch)
                fire = ep.valid & ep.ready
                if ch == "r" and hasattr(ep, "last"):
                    fire = fire & ep.last
                self.sync += If(fire, cnt.status.eq(cnt.status + 1))
            for ch in ("ar", "aw"):
                adr = CSRStatus(32, name=f"{name}{ch}addr", description=f"{name}Bus last {ch} address.")
                setattr(self, f"{name}{ch}addr", adr)
                ep = getattr(bus, ch)
                self.sync += If(ep.valid, adr.status.eq(ep.addr))

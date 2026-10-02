#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Where is the CPU? Bus handshake counters and last addresses, readable over PCIe.

v27 (2026-10-02): the CPU never printed the BIOS banner while PCIe CSR access worked and the
DDR trained; JTAGBone read zeros on that board and nothing else showed whether the CPU was
fetching, stalled on a bus access, or held in reset. These CSRs answer that from the host
(tools/cpumon.py): AXI address/response handshake counts and the last addresses on the
CPU's peripheral bus (ROM, SRAM, CSRs) and, when present, its direct memory bus (main_ram).
A request count ahead of its response count, frozen across reads, is a stalled access.
"""

from migen import *

from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRStatus


class CPUBusMonitor(LiteXModule):
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

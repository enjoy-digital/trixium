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

# BonsaiScratchpad ---------------------------------------------------------------------------------

class BonsaiScratchpad(LiteXModule):
    def __init__(self, size=256*1024, init=None):
        self.size = size
        depth     = size//32

        # # #

        self.mem = mem = Memory(256, depth, init=init)
        self.specials += mem

        # Engine x-fill read port.
        self.fill = mem.get_port()
        self.specials += self.fill

        # Engine gather write port (int32 granularity).
        self.gather = mem.get_port(write_capable=True, we_granularity=32)
        self.specials += self.gather

    def get_bus_port(self):
        """Extra RW port for SoC bus bridging (wishbone.SRAM(self.mem))."""
        return self.mem

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""SDCardUpload: the weights from an SD card, through the same loader as the UDP upload.

LiteSDCard's PHY and core (the CSRs LiteX's liblitesdcard drives: sdcard_init, the commands,
block length and count) with the card's read stream (8-bit) widened to the loader's and offered
as an upload source: with `upload` set, the loader takes it instead of the UDP port's. The
firmware arms the loader (chan, base, length) as for a UDP transfer, then issues a multi-block
read (CMD18): the blocks land in the channel's window with no CPU copy and no DMA through the
CPU's bus (which reaches main_ram only, not the engine's windows). A transfer is a whole number
of 512-byte blocks: the loader is armed for the padded length.

One module named `sdcard` in the SoC, so the CSR names are liblitesdcard's (sdcard_phy_*,
sdcard_core_*). `use_emulator`: LiteSDCard's card model instead of pads (simulation).
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRStorage

# SD Card Upload -----------------------------------------------------------------------------------

class SDCardUpload(LiteXModule):
    def __init__(self, platform, sys_clk_freq, pads=None, out_dw=128, use_emulator=False):
        from litesdcard.phy import SDPHY
        from litesdcard.core import SDCore
        self.upload = CSRStorage(1, description="1: the card's read data feeds the weight loader "
                                                "(instead of the UDP port).")
        self.source = stream.Endpoint([("data", out_dw)])

        # # #

        if use_emulator:
            from litesdcard.emulator import SDEmulator
            self.emulator = SDEmulator(platform)
            pads = self.emulator.pads
        self.phy  = SDPHY(pads, platform.device, sys_clk_freq, cmd_timeout=10e-1, data_timeout=10e-1)
        self.core = SDCore(self.phy)
        # Bytes in card order, the first in the low bits: the order the UDP port gives the loader.
        # Without `upload` the card's data is dropped before the converter: the init's own data
        # reads (the switch status, the SCR: liblitesdcard takes them by a DMA this design does
        # not have) must not leave bytes for the first transfer, whole or in a partial word.
        self.conv = conv = stream.Converter(8, out_dw)
        self.comb += [
            If(self.upload.storage,
                self.core.source.connect(conv.sink, keep={"valid", "ready", "data"}),
            ).Else(
                self.core.source.ready.eq(1),
            ),
            conv.source.connect(self.source, keep={"valid", "ready", "data"}),
        ]

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""SDCardUpload: the weights from an SD card, through the same loader as the UDP upload.

LiteSDCard's PHY and core (the firmware's firmware/sdload.c drives the commands, block length
and count) with the card's read stream (8-bit) widened to the loader's and offered
as an upload source: with `upload` set, the loader takes it instead of the UDP port's. The
firmware arms the loader (chan, base, length) as for a UDP transfer, then issues a multi-block
read (CMD18): the blocks land in the channel's window with no CPU copy and no DMA through the
CPU's bus (which reaches main_ram only, not the engine's windows). A transfer is a whole number
of 512-byte blocks: the loader is armed for the padded length.

Writes (the firmware's `sdsave`: the weights a UDP upload put in DRAM, copied to the card once):
the CPU pushes 32-bit words into `wdata` (a FIFO of `wfifo_depth`, its fill in `wlevel`), taken
little-endian into the core's write stream; the firmware fetches the DRAM through the engine into
the scratchpad and pushes from there. Slow (the CPU's CSR writes) but small, and once per model.

The SoC names it `sd`, not `sdcard`: under that name LiteX's BIOS links liblitesdcard's FatFs disk,
which needs the block2mem DMA (the link fails without it); the firmware has its own commands.
`use_emulator`: LiteSDCard's card model instead of pads (simulation).
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRStatus, CSRStorage

# SD Card Upload -----------------------------------------------------------------------------------

class SDCardUpload(LiteXModule):
    def __init__(self, platform, sys_clk_freq, pads=None, out_dw=128, use_emulator=False,
                 wfifo_depth=1024):
        from litesdcard.phy import SDPHY
        from litesdcard.core import SDCore
        self.upload = CSRStorage(1, description="1: the card's read data feeds the weight loader "
                                                "(instead of the UDP port).")
        self.source = stream.Endpoint([("data", out_dw)])
        self.wdata  = CSRStorage(32, description="Write: one 32-bit word to the card's write stream "
                                                 "(little-endian bytes).")
        self.wlevel = CSRStatus(bits_for(wfifo_depth), description="Words waiting in the write FIFO.")

        # # #

        if use_emulator:
            from litesdcard.emulator import SDEmulator
            self.emulator = SDEmulator(platform)
            pads = self.emulator.pads
        self.phy  = SDPHY(pads, platform.device, sys_clk_freq, cmd_timeout=10e-1, data_timeout=10e-1)
        self.core = SDCore(self.phy)
        # Bytes in card order, the first in the low bits: the order the UDP port gives the loader.
        # Without `upload` the card's data is dropped before the converter: the init's own data
        # reads (the switch status) must not leave bytes for the first transfer, whole or in a
        # partial word.
        self.conv = conv = stream.Converter(8, out_dw)
        self.comb += [
            If(self.upload.storage,
                self.core.source.connect(conv.sink, keep={"valid", "ready", "data"}),
            ).Else(
                self.core.source.ready.eq(1),
            ),
            conv.source.connect(self.source, keep={"valid", "ready", "data"}),
        ]
        # Writes: CSR words -> FIFO -> bytes into the core (it marks each block's last byte). A
        # block is released only once all its 128 words are in the FIFO: inside a block the PHY's
        # data writer clocks out whatever its input holds, valid or not, and the CPU's pushes are
        # slower than the card's bus - an underrun mid-block is a corrupted block (the card's CRC
        # check refuses it).
        assert wfifo_depth >= 2*128
        self.wfifo = wfifo = stream.SyncFIFO([("data", 32)], wfifo_depth)
        self.wconv = wconv = stream.Converter(32, 8)
        words_left = Signal(8)                   # The current block's words not yet released.
        self.sync += If(words_left == 0,
            If(wfifo.level >= 128, words_left.eq(128)),
        ).Elif(wfifo.source.valid & wfifo.source.ready,
            words_left.eq(words_left - 1),
        )
        self.comb += [
            wfifo.sink.valid.eq(self.wdata.re),
            wfifo.sink.data.eq(self.wdata.storage),
            self.wlevel.status.eq(wfifo.level),
            If(words_left != 0,
                wfifo.source.connect(wconv.sink, keep={"valid", "ready", "data"}),
            ),
            wconv.source.connect(self.core.sink, keep={"valid", "ready", "data"}),
        ]

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiHBMBIST: per-channel HBM bandwidth/integrity tester.

One reader + one writer, muxed onto the CSR-selected channel (channels are characterized one
at a time on the FK33 - conservative for the undersized VCCHBM regulator; aggregate bandwidth
is measured later with the engine's own streamers). Deterministic data: beat i = the 32-bit
multiplicative hash of (i + seed), replicated 8x per beat.

Modes: write (fill), read (bandwidth), check (read + compare). `cycles` counts busy sys
cycles: bandwidth = beats*32B * f_sys / cycles."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface
from litex.soc.interconnect.csr import *

from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.axi_writer import BonsaiAXIWriter

# BonsaiHBMBIST ------------------------------------------------------------------------------------

GOLDEN = 0x9E3779B1 # 32-bit golden-ratio multiplier.

class BonsaiHBMBIST(LiteXModule):
    def __init__(self, hbm_axis, channels):
        """hbm_axis: full list of HBM AXI ports; channels: indices this BIST may drive."""
        self.chan   = CSRStorage(5,  description="Channel under test (must be in the BIST set).")
        self.base   = CSRStorage(32, description="Byte address (beat-aligned).")
        self.length = CSRStorage(24, description="Length in 256-bit beats.")
        self.seed   = CSRStorage(32, description="Pattern seed.")
        self.mode   = CSRStorage(2,  description="0: write, 1: read, 2: read+check.")
        self.start  = CSRStorage(1)
        self.status = CSRStatus(fields=[CSRField("done", size=1)])
        self.errors = CSRStatus(32)
        self.cycles = CSRStatus(32, description="Busy cycles for the last run.")
        self.beats  = CSRStatus(32)

        # # #

        # Internal AXI port, muxed onto the selected channel.
        axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.reader = BonsaiAXIReader(axi)
        self.writer = BonsaiAXIWriter(axi)

        chan = self.chan.storage
        for i in channels:
            hbm = hbm_axis[i]
            sel = Signal()
            self.comb += sel.eq(chan == i)
            # Drive request channels + gather responses only when selected.
            self.comb += [
                If(sel,
                    hbm.ar.valid.eq(axi.ar.valid),
                    axi.ar.ready.eq(hbm.ar.ready),
                    hbm.ar.addr.eq(axi.ar.addr),
                    hbm.ar.len.eq(axi.ar.len),
                    hbm.ar.size.eq(axi.ar.size),
                    hbm.ar.burst.eq(axi.ar.burst),
                    hbm.ar.id.eq(axi.ar.id),
                    axi.r.valid.eq(hbm.r.valid),
                    hbm.r.ready.eq(axi.r.ready),
                    axi.r.data.eq(hbm.r.data),
                    axi.r.last.eq(hbm.r.last),
                    axi.r.id.eq(hbm.r.id),
                    hbm.aw.valid.eq(axi.aw.valid),
                    axi.aw.ready.eq(hbm.aw.ready),
                    hbm.aw.addr.eq(axi.aw.addr),
                    hbm.aw.len.eq(axi.aw.len),
                    hbm.aw.size.eq(axi.aw.size),
                    hbm.aw.burst.eq(axi.aw.burst),
                    hbm.aw.id.eq(axi.aw.id),
                    hbm.w.valid.eq(axi.w.valid),
                    axi.w.ready.eq(hbm.w.ready),
                    hbm.w.data.eq(axi.w.data),
                    hbm.w.strb.eq(axi.w.strb),
                    hbm.w.last.eq(axi.w.last),
                    axi.b.valid.eq(hbm.b.valid),
                    hbm.b.ready.eq(axi.b.ready),
                )
            ]

        # Pattern: beat index -> 32-bit hash replicated.
        def pattern(idx):
            h = Signal(32)
            return h, [h.eq(((idx + self.seed.storage)*GOLDEN)[:32])]

        gen_idx  = Signal(24)
        chk_idx  = Signal(24)
        errors   = Signal(32)
        cycles   = Signal(32)
        running  = Signal()

        gen_h, gen_stmt = pattern(gen_idx)
        chk_h, chk_stmt = pattern(chk_idx)
        self.comb += gen_stmt + chk_stmt

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.start.re & self.start.storage,
                NextValue(gen_idx, 0),
                NextValue(chk_idx, 0),
                NextValue(errors, 0),
                NextValue(cycles, 0),
                If(self.mode.storage == 0,
                    NextState("WRITE-CMD"),
                ).Else(
                    NextState("READ-CMD"),
                )
            )
        )
        fsm.act("WRITE-CMD",
            self.writer.sink_cmd.valid.eq(1),
            self.writer.sink_cmd.addr.eq(self.base.storage),
            self.writer.sink_cmd.len.eq(self.length.storage),
            If(self.writer.sink_cmd.ready,
                NextState("WRITE-DATA"),
            )
        )
        fsm.act("WRITE-DATA",
            running.eq(1),
            self.writer.sink_data.valid.eq(1),
            self.writer.sink_data.data.eq(Replicate(gen_h, 8)),
            If(self.writer.sink_data.ready,
                NextValue(gen_idx, gen_idx + 1),
                If(gen_idx == self.length.storage - 1,
                    NextState("WRITE-DRAIN"),
                )
            )
        )
        fsm.act("WRITE-DRAIN",
            running.eq(1),
            If(self.writer.idle,
                NextState("IDLE"),
            )
        )
        fsm.act("READ-CMD",
            self.reader.sink.valid.eq(1),
            self.reader.sink.addr.eq(self.base.storage),
            self.reader.sink.len.eq(self.length.storage),
            If(self.reader.sink.ready,
                NextState("READ-DATA"),
            )
        )
        fsm.act("READ-DATA",
            running.eq(1),
            self.reader.source.ready.eq(1),
            If(self.reader.source.valid,
                NextValue(chk_idx, chk_idx + 1),
                If((self.mode.storage == 2) &
                   (self.reader.source.data != Replicate(chk_h, 8)),
                    NextValue(errors, errors + 1),
                ),
                If(self.reader.source.last,
                    NextState("IDLE"),
                )
            )
        )

        self.sync += If(running, cycles.eq(cycles + 1))
        self.comb += [
            self.status.fields.done.eq(fsm.ongoing("IDLE")),
            self.errors.status.eq(errors),
            self.cycles.status.eq(cycles),
            self.beats.status.eq(chk_idx + gen_idx),
        ]

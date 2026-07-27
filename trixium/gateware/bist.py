#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiHBMBIST: per-channel HBM bandwidth/integrity tester.

One reader + one writer, muxed onto the CSR-selected channel (channels are characterized one
at a time on the FK33 - conservative for the undersized VCCHBM regulator; aggregate bandwidth
is measured with the engine's own streamers). Deterministic data: beat i carries the 32-bit
multiplicative hash of (i + seed) replicated 8x.

Timing-critical paths are pipelined: the pattern hash always feeds a register (the multiplier
retimes into a DSP), channel returns are Array-muxed (balanced, no priority chain) and the
read-check compare runs on registered data one cycle behind the handshake.

Modes: 0 write (fill), 1 read (bandwidth), 2 read+check. bandwidth = beats*32B*f_sys/cycles."""

from migen import *

from litex.gen import *

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
        n    = len(hbm_axis)

        # Every path between the BIST core and the 32-channel mux/demux gets a register stage
        # (225 MHz across all HBM port columns): AR/AW/W outward through pipe stages, R/B
        # inward through registered muxes. Full throughput is kept (ready = ~v | downstream).
        def _pipe(fields, src_valid, src_ready, dst_ready_muxed):
            """Register slice: core channel -> registered copy driving the demux."""
            v    = Signal()
            regs = [Signal.like(f) for f in fields]
            self.comb += src_ready.eq(~v | dst_ready_muxed)
            self.sync += [
                If(src_valid & src_ready,
                    v.eq(1),
                    *[r.eq(f) for r, f in zip(regs, fields)],
                ).Elif(dst_ready_muxed,
                    v.eq(0),
                )
            ]
            return v, regs

        ar_ready_m = Signal()
        aw_ready_m = Signal()
        w_ready_m  = Signal()
        r_advance  = Signal()
        self.comb += [
            ar_ready_m.eq(Array([hbm_axis[i].ar.ready for i in range(n)])[chan]),
            aw_ready_m.eq(Array([hbm_axis[i].aw.ready for i in range(n)])[chan]),
            w_ready_m.eq(Array([hbm_axis[i].w.ready   for i in range(n)])[chan]),
        ]
        ar_v, (ar_addr, ar_len) = _pipe([axi.ar.addr, axi.ar.len],
                                        axi.ar.valid, axi.ar.ready, ar_ready_m)
        aw_v, (aw_addr, aw_len) = _pipe([axi.aw.addr, axi.aw.len],
                                        axi.aw.valid, axi.aw.ready, aw_ready_m)
        w_v, (w_data, w_last)   = _pipe([axi.w.data, axi.w.last],
                                        axi.w.valid, axi.w.ready, w_ready_m)

        # Outward demux: registered request copies driven per BIST channel, valid gated by sel.
        for i in channels:
            hbm = hbm_axis[i]
            sel = Signal()
            self.comb += [
                sel.eq(chan == i),
                hbm.ar.valid.eq(ar_v & sel),
                hbm.ar.addr.eq(ar_addr),
                hbm.ar.len.eq(ar_len),
                hbm.ar.size.eq(axi.ar.size),
                hbm.ar.burst.eq(axi.ar.burst),
                hbm.ar.id.eq(0),
                hbm.r.ready.eq(r_advance & sel), # Acceptance into the R register stage.
                hbm.aw.valid.eq(aw_v & sel),
                hbm.aw.addr.eq(aw_addr),
                hbm.aw.len.eq(aw_len),
                hbm.aw.size.eq(axi.aw.size),
                hbm.aw.burst.eq(axi.aw.burst),
                hbm.aw.id.eq(0),
                hbm.w.valid.eq(w_v & sel),
                hbm.w.data.eq(w_data),
                hbm.w.strb.eq(2**32 - 1),
                hbm.w.last.eq(w_last),
                hbm.b.ready.eq(axi.b.ready & sel),
            ]
        # B responses: registered (b.ready is held high on both sides, so each response is a
        # single-cycle pulse at the source and a plain register preserves counts - same
        # treatment as the R return mux).
        b_valid_m = Signal()
        self.comb += b_valid_m.eq(Array([hbm_axis[i].b.valid for i in range(n)])[chan])
        self.sync += axi.b.valid.eq(b_valid_m) # b.ready is driven high by the writer.

        # The R return path (256-bit 32:1 mux into the reader FIFO) needs a pipeline register
        # stage to close timing at 225 MHz: mux -> reg -> reader.
        r_valid_m = Signal()
        r_data_m  = Signal(256)
        r_last_m  = Signal()
        r_v       = Signal()
        r_d       = Signal(256)
        r_l       = Signal()
        self.comb += [
            r_valid_m.eq(Array([hbm_axis[i].r.valid for i in range(n)])[chan]),
            r_data_m.eq(Array([hbm_axis[i].r.data   for i in range(n)])[chan]),
            r_last_m.eq(Array([hbm_axis[i].r.last   for i in range(n)])[chan]),
            r_advance.eq(~r_v | axi.r.ready),
            axi.r.valid.eq(r_v),
            axi.r.data.eq(r_d),
            axi.r.last.eq(r_l),
        ]
        self.sync += [
            If(r_advance,
                r_v.eq(r_valid_m),
                r_d.eq(r_data_m),
                r_l.eq(r_last_m),
            )
        ]

        # Pattern pipeline: pat register always holds hash(idx); the multiply feeds a register
        # so it retimes into a DSP. On accept, idx advances and pat reloads with hash(idx+1).
        gen_idx = Signal(24)
        gen_pat = Signal(32)
        chk_idx = Signal(24)
        chk_pat = Signal(32)
        gen_next = Signal(32)
        chk_next = Signal(32)
        load_pats = Signal()
        self.comb += [
            gen_next.eq(((gen_idx + 1 + self.seed.storage)*GOLDEN)[:32]),
            chk_next.eq(((chk_idx + 1 + self.seed.storage)*GOLDEN)[:32]),
        ]

        errors  = Signal(32)
        cycles  = Signal(32)
        running = Signal()

        # Registered read-check: compare one cycle behind the handshake.
        r_fire   = Signal()
        cmp_v    = Signal()
        r_data_d = Signal(256)
        exp_d    = Signal(256)
        self.comb += r_fire.eq(axi.r.valid & axi.r.ready)
        self.sync += [
            cmp_v.eq(r_fire & (self.mode.storage == 2)),
            r_data_d.eq(axi.r.data),
            exp_d.eq(Replicate(chk_pat, 8)),
            If(cmp_v & (r_data_d != exp_d),
                errors.eq(errors + 1),
            ),
            If(load_pats,
                errors.eq(0),
                gen_pat.eq(((self.seed.storage)*GOLDEN)[:32]),
                chk_pat.eq(((self.seed.storage)*GOLDEN)[:32]),
                gen_idx.eq(0),
                chk_idx.eq(0),
                cycles.eq(0),
            ),
            If(self.writer.sink_data.valid & self.writer.sink_data.ready,
                gen_idx.eq(gen_idx + 1),
                gen_pat.eq(gen_next),
            ),
            If(r_fire,
                chk_idx.eq(chk_idx + 1),
                chk_pat.eq(chk_next),
            ),
            If(running, cycles.eq(cycles + 1)),
        ]

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.start.re & self.start.storage,
                load_pats.eq(1),
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
            self.writer.sink_data.data.eq(Replicate(gen_pat, 8)),
            If(self.writer.sink_data.ready,
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
            # r.last marks burst ends, not the command end: count beats instead.
            If(r_fire & (chk_idx == self.length.storage - 1),
                NextState("READ-FLUSH"),
            )
        )
        fsm.act("READ-FLUSH", # Let the registered compare of the final beat land.
            running.eq(1),
            NextState("READ-FLUSH2"),
        )
        fsm.act("READ-FLUSH2",
            NextState("IDLE"),
        )

        self.comb += [
            self.status.fields.done.eq(fsm.ongoing("IDLE")),
            self.errors.status.eq(errors),
            self.cycles.status.eq(cycles),
            self.beats.status.eq(chk_idx + gen_idx),
        ]

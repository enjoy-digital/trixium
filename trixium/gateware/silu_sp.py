# SPDX-License-Identifier: BSD-2-Clause
"""Resident SiLU-multiply over the shared two-port scratchpad (future sequencer unit 11).

RUN (op 0): for i < elements, dst[i] = m[i] * siluf(g[i]) with g at src_g and m at src_m
(binary32 vectors, 8 per 256-bit beat), bit-exact with hybrid.c's silu_mul_fv through
SiluMulKernel. An element outside the kernel's fast path (see gateware/silu_mul.py) is a
FIXUP: its dst lane receives g unchanged and bit i of the bitmap at `fixmap` is set; the
firmware recomputes those elements from the untouched g/m. `fixups` counts them. The
bitmap is ceil(elements/32) little-endian words (bit i%32 of word i//32); every word of
it is written, so no stale bit survives. dst == src_g (in place, as the firmware does) is
allowed; any other overlap between dst, fixmap and the sources is rejected.

LOAD (op 1): read the kernel's tables from src_g: beats 0..512 hold lut[0..4096] (lut[4096]
in lane 0 of beat 512), beats 513..1024 hold dlut[0..4095]. Once per boot.

Contract with the integrator, as for the other resident clients: exclusive ownership while
busy; every read is granted and returns data one clock later; a write happens only on
sp_wr_req & sp_wr_gnt and its payload is held until then. error bits: 0 geometry /
alignment, 1 bounds, 2 forbidden overlap, 6 protected output. A failure stays busy until
cancel; all partial output is invalid.
"""
from migen import *
from migen.genlib.fifo import SyncFIFO
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.silu_mul import SiluMulKernel, LUT_N

SILU_COMMAND = [("op", 1), ("src_g", 24), ("src_m", 24), ("dst", 24), ("fixmap", 24),
                ("elements", 16), ("unit_m", 1), ("seg_beats", 16), ("perm_a", 8), ("perm_b", 8),
                ("protect_start", 24), ("protect_end", 24), ("table_only", 1), ("table_set", 1)]
OP_RUN, OP_LOAD = 0, 1
LOAD_BEATS = 513 + 512


class BonsaiSiluScratchpad(LiteXModule):
    def __init__(self, scratchpad_size=256*1024, tables=None, sets=2):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client address space")
        self.cmd = stream.Endpoint(SILU_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.fixups = Signal(16)
        self.read_beats, self.output_beats = Signal(32), Signal(32)
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)
        self.kernel = kernel = ResetInserter()(SiluMulKernel(tables, sets=sets))
        self.sets = sets

        IDLE, VALIDATE, LOAD, RUN, FINISH, FAILED = range(6)
        state = Signal(3, reset=IDLE)
        cap = {name: Signal(width) for name, width in SILU_COMMAND}
        beats = Signal(14)                         # data beats of each vector
        words = Signal(12)                         # bitmap words
        fix_beats = Signal(10)
        ends = {name: Signal(64) for name in ("src_g", "src_m", "dst", "fixmap")}
        geometry_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))

        def overlap(a, b):
            return (cap[a] < ends[b]) & (cap[b] < ends[a])

        def protected(a):
            return ((cap["protect_start"] != cap["protect_end"]) &
                    (cap[a] < cap["protect_end"]) & (cap["protect_start"] < ends[a]))

        load = cap["op"] == OP_LOAD
        unit_m = cap["unit_m"]
        perm = cap["perm_a"] > 1
        perm_elems = Signal(40)
        step_a = Signal(24)
        self.comb += perm_elems.eq(cap["seg_beats"]*cap["perm_a"]*cap["perm_b"]*8)
        self.comb += [
            ends["src_g"].eq(cap["src_g"] + Mux(load, C(LOAD_BEATS*32, 17), beats << 5)),
            ends["src_m"].eq(cap["src_m"] + (beats << 5)),
            ends["dst"].eq(cap["dst"] + (beats << 5)),
            ends["fixmap"].eq(cap["fixmap"] + (fix_beats << 5)),
            geometry_bad.eq((cap["src_g"][:5] != 0) | (cap["protect_start"] > cap["protect_end"]) |
                (cap["table_set"] >= sets) |
                (~load & ((cap["elements"] == 0) | (~unit_m & (cap["src_m"][:5] != 0)) |
                          (cap["dst"][:5] != 0) | (cap["fixmap"][:5] != 0) |
                          # A permuted output: whole segments of whole beats, a full grid.
                          (perm & ((cap["seg_beats"] == 0) | (cap["perm_b"] == 0) |
                                   (perm_elems != cap["elements"])))))),
            bounds_bad.eq((ends["src_g"] > scratchpad_size) | (cap["protect_end"] > scratchpad_size) |
                (~load & ((~unit_m & (ends["src_m"] > scratchpad_size)) | (ends["dst"] > scratchpad_size) |
                          (ends["fixmap"] > scratchpad_size)))),
            overlap_bad.eq(~load & (
                (overlap("dst", "src_g") & ((cap["dst"] != cap["src_g"]) | perm)) |
                (~unit_m & overlap("dst", "src_m")) | overlap("fixmap", "src_g") |
                (~unit_m & overlap("fixmap", "src_m")) | overlap("fixmap", "dst"))),
            protected_bad.eq(~load & (protected("dst") | protected("fixmap"))),
        ]

        # Read side: pairs of beats (g, m) for RUN, single beats for LOAD --------------------
        # One read per cycle at most; the response is captured a cycle later. `rd_second`
        # says the next request is the m beat of the current pair.
        pairs = ResetInserter()(SyncFIFO(width=512, depth=2))
        self.pairs = pairs
        issued = Signal(14)                        # beats (pairs) requested
        rd_second = Signal()
        rd_pending, rd_was_m = Signal(), Signal()
        g_hold = Signal(256)
        in_flight = Signal(2)                      # pairs requested but not yet in the FIFO
        can_issue = Signal()
        run_active = Signal()
        clear = Signal()
        self.comb += [
            run_active.eq(state == RUN),
            clear.eq(self.cancel | ((state != RUN) & (state != LOAD))),
            pairs.reset.eq(clear),
            kernel.reset.eq(clear),
            kernel.table_only.eq(cap["table_only"]),
            kernel.table_sel.eq(cap["table_set"] if sets > 1 else 0),
            kernel.lut_set.eq(cap["table_set"] if sets > 1 else 0),
            can_issue.eq((issued < beats) & ((pairs.level + in_flight) < 2)),
        ]

        # LOAD: one beat at a time, eight table words out of it, then the next beat.
        ld_beat = Signal(11)
        ld_lane = Signal(4)
        ld_data = Signal(256)
        ld_have = Signal()
        ld_req = Signal()
        self.comb += ld_req.eq((state == LOAD) & ~ld_have & ~rd_pending & (ld_beat < LOAD_BEATS))

        self.comb += [
            self.sp_rd_req.eq(~self.cancel & (ld_req | (run_active & (rd_second | can_issue)))),
            # unit_m: the pair is the g beat and eight 1.0s; no m read.
            self.sp_rd_adr.eq(Mux(state == LOAD, (cap["src_g"] >> 5) + ld_beat,
                                   Mux(rd_second, (cap["src_m"] >> 5) + issued,
                                       (cap["src_g"] >> 5) + issued))),
            pairs.we.eq(rd_pending & (rd_was_m | unit_m) & run_active),
            pairs.din.eq(Mux(unit_m, Cat(self.sp_rd_dat, Replicate(C(0x3F800000, 32), 8)),
                             Cat(g_hold, self.sp_rd_dat))),
        ]
        self.sync += If(clear,
            rd_pending.eq(0), rd_second.eq(0), in_flight.eq(0),
        ).Else(
            rd_pending.eq(self.sp_rd_req),
            rd_was_m.eq(rd_second),
            If(rd_pending & ~rd_was_m & run_active, g_hold.eq(self.sp_rd_dat)),
            If(self.sp_rd_req & run_active,
                If(rd_second, rd_second.eq(0), issued.eq(issued + 1)
                ).Elif(unit_m, issued.eq(issued + 1)
                ).Else(rd_second.eq(1)),
            ),
            in_flight.eq(in_flight + (self.sp_rd_req & run_active & ~rd_second) - pairs.we),
        )

        # LOAD writes into the kernel's tables. One read is outstanding at most and ld_beat
        # does not move while a beat is unpacked, so the beat in hand is ld_beat - 1.
        word = Signal(32)
        cur = Signal(11)
        dcur = Signal(11)
        unpacking = Signal()
        self.comb += [
            cur.eq(ld_beat - 1),
            dcur.eq(cur - 513),
            unpacking.eq((state == LOAD) & ld_have),
            word.eq(Array(ld_data[32*j:32*j + 32] for j in range(8))[ld_lane[:3]]),
            kernel.lut_we.eq(unpacking & ((cur < 512) | ((cur == 512) & (ld_lane == 0)))),
            kernel.lut_adr.eq(Cat(ld_lane[:3], cur[:10])),
            kernel.lut_dat.eq(word),
            kernel.dlut_we.eq(unpacking & (cur >= 513)),
            kernel.dlut_adr.eq(Cat(ld_lane[:3], dcur[:9])),
            kernel.dlut_dat.eq(word),
        ]

        # Unpack pairs into the kernel, one lane per cycle ----------------------------------
        lane = Signal(3)
        fed = Signal(17)                           # elements fed
        head_g, head_m = Signal(256), Signal(256)
        last_lane = Signal()
        self.comb += [
            head_g.eq(pairs.dout[0:256]), head_m.eq(pairs.dout[256:512]),
            last_lane.eq((lane == 7) | (fed + 1 == cap["elements"])),
            kernel.sink.valid.eq(run_active & pairs.readable & ~self.cancel),
            kernel.sink.g.eq(Array(head_g[32*j:32*j + 32] for j in range(8))[lane]),
            kernel.sink.m.eq(Array(head_m[32*j:32*j + 32] for j in range(8))[lane]),
            pairs.re.eq(kernel.sink.valid & kernel.sink.ready & last_lane),
        ]
        self.sync += If(clear, lane.eq(0), fed.eq(0)).Elif(kernel.sink.valid & kernel.sink.ready,
            fed.eq(fed + 1), lane.eq(Mux(last_lane, 0, lane + 1)))

        # Collect results into output beats and bitmap words --------------------------------
        out_lane = Signal(3)
        got = Signal(17)                           # results collected
        acc = Signal(256)
        acc_we = Signal(8)
        beat_full = Signal()                       # an output beat waits for its write
        beat_adr = Signal(13)
        beat_dat, beat_we = Signal(256), Signal(8)
        bits = Signal(32)
        word_full = Signal()                       # a bitmap word waits for its write
        word_idx = Signal(12)
        word_val = Signal(32)
        written_beats = Signal(14)
        written_words = Signal(12)
        take = Signal()
        res = Signal(32)
        ends_beat, ends_word = Signal(), Signal()
        close_beat, close_word = Signal(), Signal()
        closed_beats = Signal(14)
        closed_words = Signal(12)
        # Lane inserts by decode, not by variable shift (see the Verilog width pitfall).
        acc_next, acc_we_next, bits_next = Signal(256), Signal(8), Signal(32)
        self.comb += [
            res.eq(Mux(kernel.source.fixup, kernel.source.g, kernel.source.r)),
            acc_next.eq(acc), acc_we_next.eq(acc_we), bits_next.eq(bits),
            [If(out_lane == j, acc_next[32*j:32*j + 32].eq(res), acc_we_next[j].eq(1))
             for j in range(8)],
            [If(got[0:5] == j, bits_next[j].eq(kernel.source.fixup)) for j in range(32)],
            ends_beat.eq((out_lane == 7) | (got + 1 == cap["elements"])),
            ends_word.eq((got[0:5] == 31) | (got + 1 == cap["elements"])),
            # The accumulators keep filling while the previous beat/word waits for the
            # write port; a result that would close a second one waits instead.
            kernel.source.ready.eq(run_active & ~self.cancel &
                                   ~(beat_full & ends_beat) & ~(word_full & ends_word)),
            take.eq(kernel.source.valid & kernel.source.ready),
            close_beat.eq(take & ends_beat),
            close_word.eq(take & ends_word),
        ]
        # Output beat offset. Without a permutation it is the beat count; with one, source
        # segment s (seg_beats beats) goes to destination segment (s % a)*b + s // a, tracked
        # incrementally: `seg_off` is the destination segment's first beat, `div_off` the
        # offset of segment s // a, `in_seg` the beat within the segment, `s_mod` = s % a.
        in_seg = Signal(16)
        s_mod = Signal(8)
        seg_off, div_off = Signal(24), Signal(24)
        out_off = Signal(24)
        self.comb += out_off.eq(Mux(perm, seg_off + in_seg, closed_beats))
        self.sync += If(clear,
            in_seg.eq(0), s_mod.eq(0), seg_off.eq(0), div_off.eq(0),
        ).Elif(take & close_beat & perm,
            If(in_seg + 1 == cap["seg_beats"],
                in_seg.eq(0),
                If(s_mod + 1 == cap["perm_a"],
                    s_mod.eq(0), div_off.eq(div_off + cap["seg_beats"]),
                    seg_off.eq(div_off + cap["seg_beats"]),
                ).Else(
                    s_mod.eq(s_mod + 1), seg_off.eq(seg_off + step_a),
                ),
            ).Else(in_seg.eq(in_seg + 1)),
        )
        self.sync += If(state == VALIDATE, step_a.eq(cap["perm_b"]*cap["seg_beats"]))
        self.sync += If(clear,
            out_lane.eq(0), got.eq(0), acc.eq(0), acc_we.eq(0), bits.eq(0),
            closed_beats.eq(0), closed_words.eq(0),
        ).Elif(take,
            got.eq(got + 1),
            out_lane.eq(Mux(close_beat, 0, out_lane + 1)),
            If(close_beat,
                beat_adr.eq((cap["dst"] >> 5) + out_off),
                beat_dat.eq(acc_next), beat_we.eq(acc_we_next),
                closed_beats.eq(closed_beats + 1),
                acc.eq(0), acc_we.eq(0),
            ).Else(acc.eq(acc_next), acc_we.eq(acc_we_next)),
            If(close_word,
                word_idx.eq(closed_words), word_val.eq(bits_next),
                closed_words.eq(closed_words + 1),
                bits.eq(0),
            ).Else(bits.eq(bits_next)),
        )

        # Write port: a pending bitmap word first, then the pending data beat.
        write_word = Signal()
        self.comb += [
            write_word.eq(word_full),
            self.sp_wr_req.eq(run_active & ~self.cancel & (word_full | beat_full)),
            self.sp_wr_adr.eq(Mux(write_word, (cap["fixmap"] >> 5) + word_idx[3:12], beat_adr)),
            self.sp_wr_dat.eq(Mux(write_word, Replicate(word_val, 8), beat_dat)),
            self.sp_wr_we.eq(Mux(write_word, Cat(*[word_idx[0:3] == j for j in range(8)]), beat_we)),
        ]
        wrote = Signal()
        self.comb += wrote.eq(self.sp_wr_req & self.sp_wr_gnt)
        self.sync += If(clear,
            beat_full.eq(0), word_full.eq(0), written_beats.eq(0), written_words.eq(0),
        ).Else(
            # A close and the write of the previous one can coincide; the close wins.
            If(close_beat, beat_full.eq(1)).Elif(wrote & ~write_word, beat_full.eq(0)),
            If(close_word, word_full.eq(1)).Elif(wrote & write_word, word_full.eq(0)),
            If(wrote & write_word, written_words.eq(written_words + 1)),
            If(wrote & ~write_word, written_beats.eq(written_beats + 1),
               self.output_beats.eq(self.output_beats + 1)),
        )
        self.sync += If(~clear & take & kernel.source.fixup, self.fixups.eq(self.fixups + 1))

        # Control ---------------------------------------------------------------------------
        all_written = Signal()
        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel),
            self.busy.eq(state != IDLE),
            all_written.eq((written_beats == beats) & (written_words == words) &
                           ~beat_full & ~word_full),
        ]
        self.sync += If(self.cancel,
            state.eq(IDLE), self.done.eq(0), self.error.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, self.read_beats.eq(self.read_beats + 1)),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in cap.items()],
                    beats.eq((self.cmd.elements + C(7, 17)) >> 3),
                    words.eq((self.cmd.elements + C(31, 17)) >> 5),
                    fix_beats.eq((self.cmd.elements + C(255, 17)) >> 8),
                    issued.eq(0), ld_beat.eq(0), ld_lane.eq(0), ld_have.eq(0),
                    self.fixups.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
                    self.error.eq(0), state.eq(VALIDATE)),
                VALIDATE: If(geometry_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(FAILED),
                ).Else(state.eq(Mux(load, LOAD, RUN))),
                LOAD: [
                    If(ld_req, ld_beat.eq(ld_beat + 1)),
                    If(rd_pending, ld_data.eq(self.sp_rd_dat), ld_have.eq(1), ld_lane.eq(0)),
                    If(ld_have,
                        # Beat 512 carries only lut[4096], in lane 0.
                        If((ld_lane == 7) | (cur == 512),
                            ld_have.eq(0),
                            If(ld_beat == LOAD_BEATS, state.eq(FINISH)),
                        ).Else(ld_lane.eq(ld_lane + 1)),
                    ),
                ],
                RUN: If(all_written & (got == cap["elements"]), state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
        )

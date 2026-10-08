#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Resident RMSNorm / L2 norm (+ optional RoPE) over the shared scratchpad (sequencer unit 12).

For each of `nseg` segments of `n` FP32 values at src + seg*n*4, bit-exact with hybrid.c:

  RMS (op 0):  ss = sum64(x*x); r = 1/sqrt(ss/n + eps);   dst[i] = (x[i]*r)*w[i]
  L2  (op 1):  ss = sum64(x*x); inv = 1/max(sqrt(ss), eps); dst[i] = (x[i]*inv)*1.0

sum64 is the canonical reduction (docs/units.md, Canonical reductions: 64 partials, i % 64 in
index order, then a halving tree); every operation is an exact IEEE binary32 operation
(trixium/units/fp32.py), so there are no fixups. `w` (RMS only, n values) is read once
per command into an internal buffer and shared by every segment. dst == src (in place)
or disjoint. Segments may be strided (src_stride / dst_stride in beats, 0 = contiguous):
the attention's q heads sit between their gates in the Q projection output.

Per segment: pass 1 streams the segment through 8 multipliers and 8 adders into the 64
partials (each revisited every 8 beats, past the adder's latency), the tree reuses those
8 adders (8 of a level's independent pairs per clock), the scalar chain runs on its own
adder and the sequential divide/sqrt, pass 2 streams the segment again through two
multipliers per lane into a credit-counted output FIFO. Two stages overlap consecutive
segments: pass 1 and the tree of segment i+1 run under the scalar chain and pass 2 of
segment i (without the overlap, short segments are ~85% fixed cost: 226 cycles for 32 beats
at n = 128).

With rope_hb != 0, pass 2 also applies the firmware's neox RoPE to the first 2*rope_hb
beats of every segment, from cos/sin tables at beat address rope_tab (rope_hb cos beats,
then rope_hb sin beats).

The command (NORM_COMMAND) is the firmware's descriptor layout.

Contract with the integrator as for the other resident clients (one-cycle reads, writes
on sp_wr_req & sp_wr_gnt with the payload held). error bits: 0 geometry, 1 bounds,
2 forbidden overlap, 6 protected output; a failure stays busy until cancel.
"""

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import LiteXModule

from litex.soc.interconnect import stream

from trixium.units.fp32 import FP32Add, FP32DivSqrt, FP32Mul, _lzc, _shift_left

# Constants ----------------------------------------------------------------------------------------

NORM_COMMAND = [("op", 1), ("src", 24), ("dst", 24), ("weight", 24), ("n", 16), ("nseg", 16),
                ("eps", 32), ("src_stride", 16), ("dst_stride", 16),
                ("protect_start", 24), ("protect_end", 24),
                ("rope_hb", 5), ("rope_tab", 13)]
OP_RMS, OP_L2 = 0, 1
WMAX = 8192                     # largest n (27B: n_embd 5120)
HB_MAX = 16                     # RoPE: half the rotated dims, in beats (<= 128 elements)
F32_ONE = 0x3F800000

# Helpers ------------------------------------------------------------------------------------------

def _delay(module, value, cycles, name):
    regs = [Signal.like(value, name=f"{name}{i}") for i in range(cycles)]
    prev = value
    for r in regs:
        module.sync += r.eq(prev)
        prev = r
    return prev

# Norm Scratchpad ----------------------------------------------------------------------------------

class NormScratchpad(LiteXModule):
    """Resident RMSNorm / L2 norm with optional RoPE (see the module docstring)."""
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client "
                             "address space")
        # Command / status.
        self.cmd = stream.Endpoint(NORM_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.read_beats, self.output_beats = Signal(32), Signal(32)

        # Scratchpad client ports.
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)

        # # #

        # Command capture and validation.
        cap = {name: Signal(width, name=f"cap_{name}") for name, width in NORM_COMMAND}
        rms = cap["op"] == OP_RMS
        nbeats = Signal(13)                          # beats per segment
        seg_bytes = Signal(24)
        total = Signal(64)                           # bytes of all segments
        s_stride, d_stride = Signal(16), Signal(16)  # beats between segment starts
        s_span, d_span = Signal(64), Signal(64)      # bytes from the first to the last's end
        ends = {k: Signal(64, name=f"end_{k}") for k in ("src", "dst", "weight")}
        geometry_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))
        self.comb += [
            nbeats.eq(cap["n"][3:16]),
            seg_bytes.eq(cap["n"] << 2),
            total.eq(seg_bytes*cap["nseg"]),
            s_stride.eq(Mux(cap["src_stride"] == 0, nbeats, cap["src_stride"])),
            d_stride.eq(Mux(cap["dst_stride"] == 0, nbeats, cap["dst_stride"])),
            s_span.eq(((cap["nseg"] - 1)*s_stride << 5) + seg_bytes),
            d_span.eq(((cap["nseg"] - 1)*d_stride << 5) + seg_bytes),
            ends["src"].eq(cap["src"] + s_span),
            ends["dst"].eq(cap["dst"] + d_span),
            ends["weight"].eq(cap["weight"] + seg_bytes),
            geometry_bad.eq((cap["n"][:3] != 0) | (cap["n"] == 0) | (cap["n"] > WMAX) |
                            (cap["rope_hb"] > HB_MAX) | ((cap["rope_hb"] << 1) > nbeats) |
                            (s_stride < nbeats) | (d_stride < nbeats) |
                            (cap["nseg"] == 0) | (cap["src"][:5] != 0) | (cap["dst"][:5] != 0) |
                            (rms & (cap["weight"][:5] != 0)) |
                            (cap["protect_start"] > cap["protect_end"])),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                          (rms & (ends["weight"] > scratchpad_size)) |
                          (cap["protect_end"] > scratchpad_size) |
                          ((cap["rope_hb"] != 0) &
                           (((cap["rope_tab"] + (cap["rope_hb"] << 1)) << 5) > scratchpad_size))),
            overlap_bad.eq(
                (((cap["dst"] != cap["src"]) | (s_stride != d_stride)) &
                 (cap["dst"] < ends["src"]) & (cap["src"] < ends["dst"])) |
                (rms & (cap["dst"] < ends["weight"]) & (cap["weight"] < ends["dst"]))),
            protected_bad.eq((cap["protect_start"] != cap["protect_end"]) &
                             (cap["dst"] < cap["protect_end"]) &
                             (cap["protect_start"] < ends["dst"])),
        ]

        # Arithmetic -------------------------------------------------------------------------------
        L_MUL, L_ADD = FP32Mul.LATENCY, FP32Add.LATENCY
        sq = [FP32Mul() for _ in range(8)]           # pass 1: x*x
        acc = [FP32Add() for _ in range(8)]          # pass 1: partial + x*x
        m1 = [FP32Mul() for _ in range(8)]           # pass 2: x*scale
        m2 = [FP32Mul() for _ in range(8)]           # pass 2: (x*scale)*w
        tree = FP32Add()                             # tree and the scalar add
        self.ds = ds = FP32DivSqrt()
        self.submodules += sq + acc + m1 + m2 + [tree]

        # The 64 partials: P[8*slot + lane].
        P = [Signal(32, name=f"part{k}") for k in range(64)]

        # Weight buffer: one 256-bit word per beat.
        wbuf = Memory(256, WMAX//8, name="norm_wbuf")
        wr_port = wbuf.get_port(write_capable=True)
        rd_port = wbuf.get_port()
        self.specials += wbuf, wr_port, rd_port

        # Output FIFO (pass 2) with credits for the in-flight beats.
        FIFO_DEPTH = 32
        self.submodules.ofifo = ofifo = ResetInserter()(SyncFIFO(width=256 + 13, depth=FIFO_DEPTH))
        inflight = Signal(max=FIFO_DEPTH + 1)

        # Control ----------------------------------------------------------------------------------
        # Top level: IDLE -> VALIDATE -> WLOAD (RMS) -> RUN -> FINISH. In RUN two stages
        # work on consecutive segments: A (pass 1, tree) on segment i+1 while B (scalar
        # chain, pass 2) finishes segment i. A hands B the sum of squares and the segment
        # addresses once B is waiting; B changes `scale` and the output base only after
        # its previous segment's writes drained.
        states = ["IDLE", "VALIDATE", "WLOAD", "TLOAD", "RUN", "FINISH", "FAILED"]
        S = {name: i for i, name in enumerate(states)}
        A = {name: i for i, name in enumerate(["P1", "DRAIN", "TREE", "HAND", "DONE"])}
        B = {name: i for i, name in enumerate(["WAIT", "SCALAR", "P2"])}
        state, a_st, b_st = Signal(3), Signal(3), Signal(2)
        run = Signal()
        seg_a, seg_b = Signal(16), Signal(16)        # segments handed off / issued by B
        beat = Signal(13)                            # weight reads issued
        beat_a, beat_b = Signal(13), Signal(13)      # pass-1 / pass-2 reads issued
        rd_pending = Signal()
        rd_kind = Signal(2)                          # 0 weight, 1 pass-1 x, 2 pass-2 x
        rd_beat = Signal(13)
        clear = Signal()
        self.comb += [
            clear.eq(self.cancel | (state == S["IDLE"])),
            run.eq((state == S["RUN"]) & ~self.cancel),
            ofifo.reset.eq(clear),
            self.cmd.ready.eq((state == S["IDLE"]) & ~self.cancel),
            self.busy.eq(state != S["IDLE"]),
        ]
        # Segment bases, advanced by one stride per segment: a seg*stride product feeding the
        # read address every clock was the unit's critical path.
        src_a, dst_a = Signal(24), Signal(24)        # stage A's segment
        src_b, dst_b = Signal(24), Signal(24)        # stage B's segment

        # Read issue: one scratchpad read per clock; pass 2 (it frees output credits and
        # ends a segment) before pass 1.
        issue_w, issue_t, issue_a, issue_b, issue = Signal(), Signal(), Signal(), Signal(), Signal()
        can_p2 = Signal()
        hb = cap["rope_hb"]
        rope = Signal()
        after2 = Signal()        # a RoPE second-half beat issued last clock: it writes two beats
        self.comb += can_p2.eq((ofifo.level + inflight) < FIFO_DEPTH)
        self.comb += [
            rope.eq(hb != 0),
            issue_w.eq(~self.cancel & (state == S["WLOAD"]) & (beat < nbeats)),
            issue_t.eq(~self.cancel & (state == S["TLOAD"]) & (beat < (hb << 1))),
            issue_b.eq(run & (b_st == B["P2"]) & (beat_b < nbeats) & can_p2 & ~after2),
            issue_a.eq(run & (a_st == A["P1"]) & (beat_a < nbeats) & ~issue_b),
            issue.eq(issue_w | issue_t | issue_a | issue_b),
            self.sp_rd_req.eq(issue),
            self.sp_rd_adr.eq(Mux(issue_w, (cap["weight"] >> 5) + beat,
                              Mux(issue_t, cap["rope_tab"] + beat,
                              Mux(issue_b, (src_b >> 5) + beat_b, (src_a >> 5) + beat_a)))),
            rd_port.adr.eq(beat_b),
        ]
        self.sync += [
            rd_pending.eq(issue & ~clear),
            rd_kind.eq(Mux(issue_w, 0, Mux(issue_t, 3, Mux(issue_b, 2, 1)))),
            rd_beat.eq(Mux(issue_w | issue_t, beat, Mux(issue_b, beat_b, beat_a))),
            If(issue, self.read_beats.eq(self.read_beats + 1)),
            If(issue_w | issue_t, beat.eq(beat + 1)),
            after2.eq(issue_b & rope & (beat_b >= hb) & (beat_b < (hb << 1)) & ~clear),
        ]
        # RoPE tables (cos beats, then sin beats) and the held first-half beats.
        ctab, stab, rbuf = (Memory(256, HB_MAX, name=f"norm_{n}") for n in ("ctab", "stab", "rbuf"))
        ctab_w, stab_w, rbuf_w = (m.get_port(write_capable=True) for m in (ctab, stab, rbuf))
        ctab_r, stab_r, rbuf_r = (m.get_port(async_read=True) for m in (ctab, stab, rbuf))
        self.specials += ctab, stab, rbuf, ctab_w, stab_w, rbuf_w, ctab_r, stab_r, rbuf_r
        t_valid = Signal()
        self.comb += [
            t_valid.eq(rd_pending & (rd_kind == 3)),
            ctab_w.we.eq(t_valid & (rd_beat < hb)), ctab_w.adr.eq(rd_beat[:4]),
            ctab_w.dat_w.eq(self.sp_rd_dat),
            stab_w.we.eq(t_valid & (rd_beat >= hb)), stab_w.adr.eq((rd_beat - hb)[:4]),
            stab_w.dat_w.eq(self.sp_rd_dat),
        ]
        # Weight load writes.
        self.comb += [
            wr_port.we.eq(rd_pending & (rd_kind == 0)),
            wr_port.adr.eq(rd_beat),
            wr_port.dat_w.eq(self.sp_rd_dat),
        ]

        # The eight accumulate adders: pass 1 (partial + x*x), then the tree.
        p1_valid = Signal()
        self.comb += p1_valid.eq(rd_pending & (rd_kind == 1))
        slot_sq = _delay(self, rd_beat[:3], L_MUL, "slot_sq")
        valid_sq = _delay(self, p1_valid, L_MUL, "valid_sq")
        slot_acc = _delay(self, slot_sq, L_ADD, "slot_acc")
        valid_acc = _delay(self, valid_sq, L_ADD, "valid_acc")
        p1_inflight = Signal(8)
        self.sync += If(clear, p1_inflight.eq(0)).Else(
            p1_inflight.eq(p1_inflight + p1_valid - valid_acc))

        # Tree: level w = 32 >> level (0..5), P[j] = P[j] + P[j + w] for j < w, eight j per
        # clock (j = tj + lane, tj a multiple of 8; lanes >= w idle below w = 8). The same
        # pairs as one add per clock, so the same bits.
        level = Signal(3)
        w_cur = Signal(7)
        self.comb += w_cur.eq(Array([C(32 >> i, 7) for i in range(6)] + [C(0, 7)]*2)[level])
        tj = Signal(7)
        tree_mode = Signal()
        tree_issue = Signal()
        lane_mask = Signal(8)
        self.comb += [
            tree_mode.eq(a_st == A["TREE"]),
            tree_issue.eq(run & tree_mode & (tj < w_cur)),
            lane_mask.eq(Array([C(0xFF, 8)]*3 +
                               [C(0x0F, 8), C(0x03, 8), C(0x01, 8), C(0, 8), C(0, 8)])[level]),
        ]
        tree_valid = _delay(self, tree_issue, L_ADD, "tree_valid")
        tree_slot = _delay(self, tj[3:5], L_ADD, "tree_slot")
        tree_mask = _delay(self, lane_mask, L_ADD, "tree_mask")
        tree_inflight = Signal(4)
        self.sync += If(clear, tree_inflight.eq(0)).Else(
            tree_inflight.eq(tree_inflight + tree_issue - tree_valid))
        # Operand b's index: 0..3 = lane + 32 + 8*s, 4..5 = lane + 16 + 8*s, 6..9 = lane + 8,
        # 4, 2, 1 (s = tj >> 3).
        s_tj = tj[3:5]
        b_sel = Signal(4)
        self.comb += b_sel.eq(Mux(level == 0, s_tj, Mux(level == 1, 4 + s_tj[0], 4 + level)))
        for lane in range(8):
            x = self.sp_rd_dat[32*lane:32*lane + 32]
            a_tree = Array(P[8*s + lane] for s in range(4))[s_tj]
            b_tree = Array([P[lane + 32 + 8*s] for s in range(4)] +
                           [P[lane + 16 + 8*s] for s in range(2)] +
                           [P[lane + d] for d in (8, 4, 2, 1)])[b_sel]
            self.comb += [sq[lane].a.eq(x), sq[lane].b.eq(x),
                          acc[lane].a.eq(Mux(tree_mode, a_tree,
                                             Array(P[8*s + lane] for s in range(8))[slot_sq])),
                          acc[lane].b.eq(Mux(tree_mode, b_tree, sq[lane].r))]
        # Scalar chain on its own adder: ss/n + eps.
        sc_add = Signal()
        sc_step = Signal(3)
        sc_wait = Signal()
        ss, tmp, scale = Signal(32), Signal(32), Signal(32)
        sc_add_valid = _delay(self, sc_add, L_ADD, "sc_add_valid")
        n_f32 = Signal(32)
        nlz, nlz_s = _lzc(cap["n"], 16)
        self.comb += nlz_s
        nman, nman_c = _shift_left(cap["n"], nlz, 15, 16)
        self.comb += [nman_c, n_f32.eq(Cat(C(0, 8), nman[:15], (127 + 15) - nlz, C(0, 1)))]
        self.comb += [tree.a.eq(tmp), tree.b.eq(cap["eps"])]
        partial_writes = []
        for k in range(64):
            lane, s = k % 8, k // 8
            partial_writes.append(
                If(clear | (run & (a_st == A["P1"]) & (beat_a == 0)),
                    P[k].eq(0),
                ).Elif(valid_acc & (slot_acc == s),
                    P[k].eq(acc[lane].r),
                ).Elif(tree_valid & (tree_slot == s) & tree_mask[lane],
                    P[k].eq(acc[lane].r),
                ))
        self.sync += partial_writes

        # Pass 2: m1 = x*scale, m2 = m1*w; w = 1.0 for L2.
        p2_valid = Signal()
        self.comb += p2_valid.eq(rd_pending & (rd_kind == 2))
        w_word = Signal(256)
        self.comb += w_word.eq(Mux(rms, rd_port.dat_r, Replicate(C(F32_ONE, 32), 8)))
        w_hold = _delay(self, w_word, L_MUL, "w_hold")
        for lane in range(8):
            self.comb += [
                m1[lane].a.eq(self.sp_rd_dat[32*lane:32*lane + 32]), m1[lane].b.eq(scale),
                m2[lane].a.eq(m1[lane].r), m2[lane].b.eq(w_hold[32*lane:32*lane + 32]),
            ]
        out_valid = _delay(self, p2_valid, 2*L_MUL, "out_valid")
        out_beat = _delay(self, rd_beat, 2*L_MUL, "out_beat")
        y = Signal(256)
        self.comb += y.eq(Cat(*[m2[lane].r for lane in range(8)]))
        # RoPE (neox pairs over the first 2*hb beats of each segment, the firmware's
        # rope_apply_partial): a first-half beat waits in rbuf; its second-half partner
        # produces both, o0 = y0*c - y1*s at the first beat's place, o1 = y0*s + y1*c at its
        # own. Every beat takes the same L_MUL + L_ADD path so the writes keep their order;
        # the second of a pair's two writes goes one clock later (issue skips that clock).
        first, second, pas = Signal(), Signal(), Signal()
        ridx = Signal(4)
        self.comb += [
            first.eq(out_valid & rope & (out_beat < hb)),
            second.eq(out_valid & rope & (out_beat >= hb) & (out_beat < (hb << 1))),
            pas.eq(out_valid & ~first & ~second),
            ridx.eq((out_beat - hb)[:4]),
            rbuf_w.we.eq(first & ~clear), rbuf_w.adr.eq(out_beat[:4]), rbuf_w.dat_w.eq(y),
            rbuf_r.adr.eq(ridx), ctab_r.adr.eq(ridx), stab_r.adr.eq(ridx),
        ]
        rmul = [[FP32Mul() for _ in range(8)] for _ in range(4)]     # y0*c, y1*s, y0*s, y1*c
        radd = [[FP32Add() for _ in range(8)] for _ in range(2)]     # o0, o1
        self.submodules += [m for row in rmul for m in row] + [a for row in radd for a in row]
        for lane in range(8):
            sl = slice(32*lane, 32*lane + 32)
            y0, y1 = rbuf_r.dat_r[sl], y[sl]
            c, sn = ctab_r.dat_r[sl], stab_r.dat_r[sl]
            self.comb += [
                rmul[0][lane].a.eq(y0), rmul[0][lane].b.eq(c),
                rmul[1][lane].a.eq(y1), rmul[1][lane].b.eq(sn),
                rmul[2][lane].a.eq(y0), rmul[2][lane].b.eq(sn),
                rmul[3][lane].a.eq(y1), rmul[3][lane].b.eq(c),
                radd[0][lane].a.eq(rmul[0][lane].r),
                radd[0][lane].b.eq(rmul[1][lane].r ^ 0x80000000),
                radd[1][lane].a.eq(rmul[2][lane].r), radd[1][lane].b.eq(rmul[3][lane].r),
            ]
        LR = L_MUL + L_ADD
        pas_d, second_d = _delay(self, pas, LR, "rope_pas"), _delay(self, second, LR, "rope_second")
        beat_d, y_d = _delay(self, out_beat, LR, "rope_beat"), _delay(self, y, LR, "rope_y")
        o0 = Cat(*[radd[0][lane].r for lane in range(8)])
        o1 = Cat(*[radd[1][lane].r for lane in range(8)])
        pend_v, pend_adr, pend_dat = Signal(), Signal(13), Signal(256)
        self.sync += If(clear, pend_v.eq(0)).Else(
            pend_v.eq(second_d), pend_adr.eq(beat_d), pend_dat.eq(o1))
        wr_now = Signal()
        self.comb += [
            wr_now.eq((pas_d | second_d | pend_v) & ~clear),
            ofifo.we.eq(wr_now),
            If(pend_v,
                ofifo.din.eq(Cat(pend_dat, pend_adr)),
            ).Elif(second_d,
                ofifo.din.eq(Cat(o0, beat_d - hb)),
            ).Else(
                ofifo.din.eq(Cat(y_d, beat_d)),
            ),
        ]
        self.sync += If(clear, inflight.eq(0)).Else(
            inflight.eq(inflight + issue_b - wr_now))
        # Output beats carry their own address: dst + segment offset + beat. The FIFO holds
        # only data; addresses follow the write counter within the segment.
        wr_beat = Signal(13)
        wr_seg_base = Signal(13)
        drained = Signal()                           # every pass-2 beat issued so far written
        self.comb += [
            self.sp_wr_req.eq(ofifo.readable & ~self.cancel),
            self.sp_wr_adr.eq(wr_seg_base + ofifo.dout[256:]),
            self.sp_wr_dat.eq(ofifo.dout[:256]),
            self.sp_wr_we.eq(0xFF),
            ofifo.re.eq(self.sp_wr_req & self.sp_wr_gnt),
            drained.eq((inflight == 0) & ~ofifo.readable & ~(rd_pending & (rd_kind == 2)) &
                       ~pend_v),
        ]
        self.sync += If(clear, wr_beat.eq(0)).Elif(ofifo.re,
            wr_beat.eq(wr_beat + 1), self.output_beats.eq(self.output_beats + 1))

        # Sequencing -------------------------------------------------------------------------------
        ds_start = Signal()
        self.comb += ds.start.eq(ds_start)
        stage_a = Case(a_st, {
            A["P1"]: If(issue_a, beat_a.eq(beat_a + 1)).Elif(beat_a == nbeats, a_st.eq(A["DRAIN"])),
            A["DRAIN"]: If((p1_inflight == 0) & ~p1_valid,
                level.eq(0), tj.eq(0), a_st.eq(A["TREE"])),
            A["TREE"]: If(tree_issue,
                tj.eq(tj + 8),
            ).Elif(tree_inflight == 0,
                If(level == 5, a_st.eq(A["HAND"])).Else(level.eq(level + 1), tj.eq(0)),
            ),
            A["HAND"]: If(b_st == B["WAIT"],
                ss.eq(P[0]), src_b.eq(src_a), dst_b.eq(dst_a),
                sc_step.eq(0), sc_wait.eq(0), b_st.eq(B["SCALAR"]),
                If(seg_a + 1 == cap["nseg"], a_st.eq(A["DONE"])).Else(
                    seg_a.eq(seg_a + 1), beat_a.eq(0),
                    src_a.eq(src_a + (s_stride << 5)), dst_a.eq(dst_a + (d_stride << 5)),
                    a_st.eq(A["P1"]))),
        })
        stage_b = Case(b_st, {
            # RMS: ss/n -> +eps -> sqrt -> 1/x.  L2: sqrt(ss) -> max(eps) -> 1/x.
            B["SCALAR"]: If(~sc_wait,
                Case(sc_step, {
                    0: If(rms,
                            ds.a.eq(ss), ds.b.eq(n_f32), ds.op.eq(0), ds_start.eq(1),
                            sc_wait.eq(1),
                        ).Else(
                            ds.a.eq(ss), ds.op.eq(1), ds_start.eq(1), sc_wait.eq(1),
                            sc_step.eq(2)),
                    1: [sc_add.eq(1), sc_wait.eq(1)],                       # tmp + eps
                    2: [ds.a.eq(tmp), ds.op.eq(1), ds_start.eq(1), sc_wait.eq(1)],   # sqrt
                    3: [ds.a.eq(C(F32_ONE, 32)), ds.b.eq(tmp), ds.op.eq(0), ds_start.eq(1),
                        sc_wait.eq(1)],                                     # 1/x
                    # The previous segment's pass 2 still reads `scale` and the output base.
                    4: If(drained, scale.eq(tmp), beat_b.eq(0), wr_beat.eq(0),
                          wr_seg_base.eq(dst_b >> 5), b_st.eq(B["P2"])),
                }),
            ).Else(
                If(ds.done,
                    sc_wait.eq(0),
                    # L2 after the sqrt: nrm = max(sqrt, eps) (bit order = value order for
                    # the non-negative results; a NaN stays).
                    If(~rms & (sc_step == 2) & (ds.r < cap["eps"]),
                        tmp.eq(cap["eps"])).Else(tmp.eq(ds.r)),
                    sc_step.eq(Mux(~rms & (sc_step == 2), 3, sc_step + 1)),
                ).Elif(sc_add_valid,
                    sc_wait.eq(0), tmp.eq(tree.r), sc_step.eq(sc_step + 1),
                ),
            ),
            B["P2"]: If(issue_b, beat_b.eq(beat_b + 1)).Elif(beat_b == nbeats,
                seg_b.eq(seg_b + 1), b_st.eq(B["WAIT"])),
        })
        self.sync += [
            self.done.eq(0), ds_start.eq(0), sc_add.eq(0),
            If(self.cancel, state.eq(S["IDLE"]), self.error.eq(0)).Else(
            Case(state, {
                S["IDLE"]: If(self.cmd.valid & self.cmd.ready,
                    *[v.eq(getattr(self.cmd, k)) for k, v in cap.items()],
                    self.error.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
                    beat.eq(0),
                    state.eq(S["VALIDATE"])),
                S["VALIDATE"]: If(geometry_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3),
                                      protected_bad)),
                    state.eq(S["FAILED"]),
                ).Else(beat.eq(0), src_a.eq(cap["src"]), dst_a.eq(cap["dst"]),
                       seg_a.eq(0), seg_b.eq(0), beat_a.eq(0), a_st.eq(A["P1"]), b_st.eq(B["WAIT"]),
                       state.eq(Mux(rms, S["WLOAD"],
                                    Mux(cap["rope_hb"] != 0, S["TLOAD"], S["RUN"])))),
                S["WLOAD"]: If((beat == nbeats) & ~rd_pending,
                    beat.eq(0), state.eq(Mux(cap["rope_hb"] != 0, S["TLOAD"], S["RUN"]))),
                S["TLOAD"]: If((beat == (cap["rope_hb"] << 1)) & ~rd_pending, state.eq(S["RUN"])),
                S["RUN"]: [stage_a, stage_b,
                    If((seg_b == cap["nseg"]) & drained, state.eq(S["FINISH"]))],
                S["FINISH"]: [self.done.eq(1), state.eq(S["IDLE"])],
            })),
        ]

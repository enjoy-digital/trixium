#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Streaming exact-FP32 attention passes: the two O(T) loops of hybrid.c's attention core.

One lane per query head. K or V rows arrive as ONE 256-bit stream in position order (the
engine merges the position-striped channels); a row is ``nkv*hd/8`` beats, kv head ``j``
in beats ``j*hd/8 ..``. Lane ``l`` reads the row part of its kv head ``l // n_rep``.

  mode 0, scores:  s[l][t] = (sum_i q[l][i]*k[t][i], in i order from +0) * scale
                   8 positions interleave per lane (the accumulator feedback), one
                   multiply-add per lane per clock: hd*8 clocks per 8 positions.
  mode 1, values:  o[l][i] = sum_t p[l][t]*v[t][i], in t order from +0
                   hd independent chains per lane: hd clocks per position.

The softmax between them (max, exp, sum, p = e/sum) stays with the caller, so both passes
are bit-exact against the firmware's own loops. Operands come from the scratchpad:
``rec`` points at one 256-bit record::

    w0 = mode | nq << 8 | n_rep << 16 | nkv << 24
    w1 = T (positions)     w2 = hd (8, 16, 32, ... 8*hb_max floats; a power of two)
    w3 = src byte address  (mode 0: q rows, mode 1: p rows)   w4 = src lane stride (bytes)
    w5 = dst byte address  (mode 0: scores, mode 1: outputs)  w6 = dst lane stride (bytes)
    w7 = scale (mode 0)

Scores are written as whole beats of 8 positions (a row of ceil(T/8) beats per lane; the
padding positions hold junk), outputs as hd/8 beats per lane.
"""

from functools import reduce
from operator import and_

import numpy as np

from migen import *

from litex.gen import *
from litex.soc.interconnect import stream

from trixium.units.fp32 import ExactFP32Add, ExactFP32Mul

MODE_SCORES = 0
MODE_VALUES = 1


def scores_reference(q, K, scale):
    """q [nq, hd], K [T, nkv, hd] (kv head l // n_rep): hybrid.c's score loop."""
    f32 = np.float32
    nq, hd = q.shape
    n_rep = nq//K.shape[1]
    s = np.zeros((nq, K.shape[0]), f32)
    with np.errstate(all="ignore"):
        for l in range(nq):
            acc = np.zeros(K.shape[0], f32)
            for i in range(hd):
                acc = (acc + (q[l, i]*K[:, l//n_rep, i]).astype(f32)).astype(f32)
            s[l] = (acc*f32(scale)).astype(f32)
    return s


def values_reference(p, V):
    """p [nq, T], V [T, nkv, hd]: hybrid.c's weighted-V loop (oh = 0; oh += p*v, t order)."""
    f32 = np.float32
    nq, T = p.shape
    n_rep = nq//V.shape[1]
    o = np.zeros((nq, V.shape[2]), f32)
    with np.errstate(all="ignore"):
        for l in range(nq):
            for t in range(T):
                o[l] = (o[l] + (p[l, t]*V[t, l//n_rep]).astype(f32)).astype(f32)
    return o


class AttentionStream(LiteXModule):
    def __init__(self, lanes=24, groups=4, hb_max=32, t_max=2048, adr_bits=13,
                 add_factory=ExactFP32Add, mul_factory=ExactFP32Mul):
        L, P = lanes, 8
        pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        assert pow2(hb_max) and pow2(t_max) and t_max % 8 == 0
        La, Lm = add_factory.latency, mul_factory.latency
        assert P >= La + 2, "8 interleaved positions must cover the accumulator feedback"
        assert 8*hb_max >= La + Lm + 8
        hbb, tb = (hb_max - 1).bit_length(), (t_max//8 - 1).bit_length()
        pw = max(hbb, tb)                       # Per-lane parameter memory address width.
        self.lanes, self.groups, self.hb_max, self.t_max = L, groups, hb_max, t_max

        self.start = Signal()
        self.rec   = Signal(adr_bits)          # Scratchpad beat address of the record.
        self.busy  = Signal()
        self.done  = Signal()
        self.sink  = stream.Endpoint([("data", 256)])
        self.sp_rd_ok  = Signal(reset=1)
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(adr_bits)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(adr_bits)
        self.sp_wr_we  = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()
        self.error = Signal()                  # Sticky: the record is outside the geometry.

        # # #

        def dly(value, n, name):
            for i in range(n):
                r = Signal.like(value, name=f"{name}_d{i}")
                self.sync += r.eq(value)
                value = r
            return value

        def fsel(word, index):
            return Array([word[32*i:32*(i + 1)] for i in range(8)])[index]

        dat = self.sp_rd_dat
        run, mode = Signal(), Signal()
        nq, n_rep, nkv = Signal(8), Signal(8), Signal(8)
        T = Signal(16)
        hd = Signal(16)                     # Floats per head.
        hb = Signal(max=hb_max + 1)         # Beats per head.
        src, sstr, dst, dstr = Signal(adr_bits), Signal(adr_bits), Signal(adr_bits), Signal(adr_bits)
        scale = Signal(32)
        rowbeats = Signal(12)
        tbeats = Signal(max=t_max//8 + 1)   # ceil(T/8)
        ngroups = Signal(max=t_max//8 + 1)

        # Record, then the per-lane operands: registered read requests ---------------------------
        ld_st = Signal(3)       # 0 idle, 1 record, 2 decode, 3 operands, 4 last, 5 ready
        rq_v, rq_adr, rq_lane, rq_word, rq_last = (Signal(), Signal(adr_bits), Signal(max=L),
                                                   Signal(pw), Signal())
        rsp_v, rsp_lane, rsp_word, rsp_last, rsp_rec = (Signal(), Signal(max=L), Signal(pw),
                                                       Signal(), Signal())
        taken, free = Signal(), Signal()
        ld_lane, ld_word = Signal(max=L + 1), Signal(pw + 1)
        ld_words = Signal(pw + 1)           # Operand beats per lane.
        lane_base = Signal(adr_bits)
        params_ready = Signal()     # Record decoded, operands loaded, kv heads mapped.
        self.comb += [
            taken.eq(rq_v & self.sp_rd_ok),
            free.eq(~rq_v | taken),
            self.sp_rd_req.eq(rq_v),
            self.sp_rd_adr.eq(rq_adr),
            ld_words.eq(Mux(mode, tbeats, hb)),
        ]
        rec_w = [dat[32*i:32*(i + 1)] for i in range(8)]
        geometry_bad = Signal()
        self.comb += geometry_bad.eq(
            (rec_w[0][8:16] == 0) | (rec_w[0][8:16] > L) | (rec_w[0][16:24] == 0) |
            (rec_w[0][24:32] == 0) | (rec_w[0][24:32] > groups) |
            (rec_w[0][8:16] > rec_w[0][16:24]*rec_w[0][24:32]) |
            (rec_w[1] == 0) | (rec_w[1] > t_max) |
            (rec_w[2] < 8) | (rec_w[2] > 8*hb_max) | ((rec_w[2] & (rec_w[2] - 1)) != 0))
        self.sync += [
            rsp_v.eq(taken), rsp_lane.eq(rq_lane), rsp_word.eq(rq_word), rsp_last.eq(rq_last),
            rsp_rec.eq(taken & (ld_st == 1)),
            If(self.start,
                ld_st.eq(1), rq_v.eq(1), rq_adr.eq(self.rec), self.error.eq(0),
            ).Else(
                If(taken, rq_v.eq(0)),
                Case(ld_st, {
                    1: If(rsp_rec,
                        If(geometry_bad,
                            self.error.eq(1), ld_st.eq(0),
                        ).Else(
                            mode.eq(rec_w[0][0]),
                            nq.eq(rec_w[0][8:16]), n_rep.eq(rec_w[0][16:24]), nkv.eq(rec_w[0][24:32]),
                            T.eq(rec_w[1]), hd.eq(rec_w[2]), hb.eq(rec_w[2] >> 3),
                            src.eq(rec_w[3] >> 5), sstr.eq(rec_w[4] >> 5),
                            dst.eq(rec_w[5] >> 5), dstr.eq(rec_w[6] >> 5),
                            scale.eq(rec_w[7]),
                            ld_st.eq(2),
                        )
                    ),
                    2: [
                        rowbeats.eq(nkv*hb),
                        tbeats.eq((T + 7) >> 3),
                        ngroups.eq((T + 7) >> 3),
                        ld_lane.eq(0), ld_word.eq(0), lane_base.eq(src),
                        ld_st.eq(3),
                    ],
                    3: If(free,
                        If(ld_lane == nq,
                            ld_st.eq(4),
                        ).Else(
                            rq_v.eq(1), rq_adr.eq(lane_base + ld_word),
                            rq_lane.eq(ld_lane), rq_word.eq(ld_word),
                            rq_last.eq((ld_lane == nq - 1) & (ld_word == ld_words - 1)),
                            If(ld_word == ld_words - 1,
                                ld_word.eq(0), ld_lane.eq(ld_lane + 1),
                                lane_base.eq(lane_base + sstr),
                            ).Else(
                                ld_word.eq(ld_word + 1),
                            )
                        )
                    ),
                    4: If(rsp_v & rsp_last, ld_st.eq(5)),   # Written at this edge.
                    5: If(self.done | self.error, ld_st.eq(0)),
                })
            )
        ]

        # Row buffers: one per kv head, two halves of 8 rows (mode 0: a group of 8
        # positions; mode 1: one position per half) --------------------------------------------
        rb_bits = hbb + 3 + 1
        rowmems, rowrd = [], []
        fill_beat = Signal(12)          # Beat within the row.
        fill_row = Signal(16)           # Rows accepted.
        fill_units = Signal(2)          # Completed units (groups or rows), mod 4.
        cons_units = Signal(2)          # Units consumed by the issue, mod 4.
        fill_j, fill_idx = Signal(max=max(2, groups)), Signal(hbb if hbb else 1)
        fill_p, fill_half = Signal(3), Signal()
        unit_done = Signal()
        self.comb += [
            # Beat -> (kv head, word): hb is a power of two, so a Case over its log.
            Case(hb, {1 << b: [fill_j.eq(fill_beat >> b), fill_idx.eq(fill_beat & ((1 << b) - 1))]
                      for b in range(hbb + 1)}),
            fill_p.eq(Mux(mode, 0, fill_row[:3])),
            fill_half.eq(Mux(mode, fill_row[0], fill_row[3])),
            self.sink.ready.eq(run & params_ready & ((fill_units - cons_units)[:2] != 2)),
            unit_done.eq((fill_beat == rowbeats - 1) &
                         (mode | (fill_row[:3] == 7) | (fill_row == T - 1))),
        ]
        self.sync += If(self.start,
            fill_beat.eq(0), fill_row.eq(0), fill_units.eq(0),
        ).Elif(self.sink.valid & self.sink.ready,
            If(fill_beat == rowbeats - 1,
                fill_beat.eq(0), fill_row.eq(fill_row + 1),
                If(unit_done, fill_units.eq(fill_units + 1)),
            ).Else(
                fill_beat.eq(fill_beat + 1),
            )
        )
        rd_adr = Signal(rb_bits)
        for j in range(groups):
            m = Memory(256, 2*8*hb_max, name=f"attn_row{j}")
            wp = m.get_port(write_capable=True)
            rp = m.get_port()
            self.specials += m, wp, rp
            rowrd.append(rp)
            self.comb += [
                wp.adr.eq(Cat(fill_idx, fill_p, fill_half)),
                wp.dat_w.eq(self.sink.data),
                wp.we.eq(self.sink.valid & self.sink.ready & (fill_j == j)),
                rp.adr.eq(rd_adr),
            ]

        # Issue -----------------------------------------------------------------------------------
        # mode 0: units = groups of 8 positions; per unit i in [0, hd), p in [0, 8).
        # mode 1: units = positions; per unit i in [0, hd).
        is_run, is_i, is_p, is_unit = Signal(), Signal(16), Signal(3), Signal(16)
        is_units = Signal(16)
        issue = Signal()
        unit_last = Signal()
        snap_ok = Signal()
        self.comb += [
            is_units.eq(Mux(mode, T, ngroups)),
            issue.eq(is_run & (fill_units != cons_units) & snap_ok),
            unit_last.eq((is_i == hd - 1) & (mode | (is_p == 7))),
            rd_adr.eq(Cat(is_i[3:3 + hbb], Mux(mode, 0, is_p), is_unit[0])),
        ]
        self.sync += If(self.start,
            is_run.eq(0), is_unit.eq(0), cons_units.eq(0),
        ).Elif(~is_run,
            If(run & params_ready & (is_unit < is_units),
                is_run.eq(1), is_i.eq(0), is_p.eq(0),
            )
        ).Elif(issue,
            If(mode,
                is_i.eq(is_i + 1),
            ).Else(
                is_p.eq(is_p + 1),
                If(is_p == 7, is_i.eq(is_i + 1)),
            ),
            If(unit_last,
                is_run.eq(0), is_unit.eq(is_unit + 1), cons_units.eq(cons_units + 1),
            )
        )
        # Stage 1 (registered): the row word is on the memories; select per lane below.
        s1_v, s1_sub, s1_first, s1_p, s1_i, s1_last, s1_unit = (Signal(), Signal(3), Signal(),
            Signal(3), Signal(16), Signal(), Signal(16))
        self.sync += [
            s1_v.eq(issue), s1_sub.eq(is_i[:3]), s1_p.eq(is_p), s1_i.eq(is_i),
            s1_first.eq(Mux(mode, is_unit == 0, is_i == 0)),
            s1_last.eq(unit_last), s1_unit.eq(is_unit),
        ]
        # Stage 2: operands registered per lane; the multiply issues.
        s2_v, s2_first, s2_p, s2_i, s2_last, s2_unit = (dly(s1_v, 1, "s2v"), dly(s1_first, 1, "s2f"),
            dly(s1_p, 1, "s2p"), dly(s1_i, 1, "s2i"), dly(s1_last, 1, "s2l"), dly(s1_unit, 1, "s2u"))
        # Add stage (after the multiply) and write-back stage (after the add).
        a_v, a_first = dly(s2_v, Lm, "av"), dly(s2_first, Lm, "af")
        a_p, a_i = dly(s2_p, Lm, "ap"), dly(s2_i, Lm, "ai")
        w_v, w_p, w_i = dly(a_v, La, "wv"), dly(a_p, La, "wp"), dly(a_i, La, "wi")
        w_last, w_unit = dly(s2_last, Lm + La, "wl"), dly(s2_unit, Lm + La, "wu")
        # mode 1 accumulator read, one clock ahead of the add stage (sync read).
        acc_rd_i = dly(s2_i, Lm - 1, "ari") if Lm > 1 else s2_i

        muls = [mul_factory() for _ in range(L)]
        adds = [add_factory() for _ in range(L)]
        for l in range(L):
            setattr(self.submodules, f"mul{l}", muls[l])
            setattr(self.submodules, f"add{l}", adds[l])

        # Per-lane operand memory: q (mode 0, hd/8 words) or p (mode 1, ceil(T/8) words).
        opmems = []
        kv_of = [Signal(max=max(2, groups), name=f"attn_kv{l}") for l in range(L)]
        acc0 = [[Signal(32, name=f"attn_acc{l}_{p}") for p in range(P)] for l in range(L)]
        snap = [[[Signal(32, name=f"attn_snap{b}_{l}_{p}") for p in range(P)] for l in range(L)]
                for b in range(2)]
        accmems = []
        capture = Signal()          # mode 0: a group's last add retires.
        cap_bank = Signal()
        self.comb += [capture.eq(w_v & w_last & ~mode), cap_bank.eq(w_unit[0])]
        # kv head of each lane: l // n_rep, computed once per command (a running count).
        kv_cnt, kv_rep = Signal(8), Signal(8)
        kv_ptr = Signal(max=L + 1)
        self.sync += If(self.start | (ld_st == 2),
            kv_ptr.eq(0), kv_cnt.eq(0), kv_rep.eq(0),
        ).Elif(kv_ptr < L,
            kv_ptr.eq(kv_ptr + 1),
            Case(kv_ptr, {l: kv_of[l].eq(kv_cnt) for l in range(L)}),
            If(kv_rep == n_rep - 1, kv_rep.eq(0), kv_cnt.eq(kv_cnt + 1)).Else(kv_rep.eq(kv_rep + 1)),
        )
        self.comb += params_ready.eq((ld_st == 5) & (kv_ptr == L))

        for l in range(L):
            om = Memory(256, 1 << pw, name=f"attn_op{l}")
            ow = om.get_port(write_capable=True)
            orp = om.get_port(async_read=True)
            self.specials += om, ow, orp
            self.comb += [
                ow.adr.eq(rsp_word), ow.dat_w.eq(dat),
                ow.we.eq(rsp_v & ~rsp_rec & (rsp_lane == l)),
                orp.adr.eq(Mux(mode, is_unit[3:3 + pw], is_i[3:3 + pw])),
            ]
            opnd = Signal(32, name=f"attn_opnd{l}")
            opnd2 = Signal(32, name=f"attn_opnd2_{l}")
            self.sync += [
                opnd.eq(fsel(orp.dat_r, Mux(mode, is_unit[:3], is_i[:3]))),
                opnd2.eq(opnd),
            ]
            row = Signal(32, name=f"attn_rowv{l}")
            self.sync += row.eq(fsel(Array([r.dat_r for r in rowrd])[kv_of[l]], s1_sub))
            am = Memory(256, hb_max, name=f"attn_acc{l}")
            aw = am.get_port(write_capable=True, we_granularity=32)
            ar = am.get_port()
            self.specials += am, aw, ar
            accmems.append(ar)
            acc_old = Signal(32, name=f"attn_accold{l}")
            ar_sub = dly(acc_rd_i[:3], 1, f"arsub{l}")
            self.comb += acc_old.eq(fsel(ar.dat_r, ar_sub))
            a0 = Array(acc0[l])
            self.comb += [
                muls[l].a.eq(opnd2), muls[l].b.eq(row), muls[l].valid.eq(s2_v),
                adds[l].a.eq(Mux(a_first, 0, Mux(mode, acc_old, a0[a_p]))),
                adds[l].b.eq(muls[l].result),
                adds[l].valid.eq(a_v),
                aw.adr.eq(w_i[3:3 + hbb] if hbb else 0),
                aw.dat_w.eq(Replicate(adds[l].result, 8)),
                aw.we.eq(Mux(w_v & mode, 1 << w_i[:3], 0)),
            ]
            self.sync += If(w_v & ~mode, a0[w_p].eq(adds[l].result))
            for b in range(2):
                self.sync += If(capture & (cap_bank == b),
                    *[snap[b][l][p].eq(adds[l].result if p == P - 1 else acc0[l][p]) for p in range(P)])

        # mode 0 output: scale the snapshot (shared multiplier), a beat per lane -------------------
        sc_run, sc_l, sc_p, sc_unit = Signal(), Signal(max=L + 1), Signal(3), Signal(16)
        caps = Signal(16)
        ofifo = stream.SyncFIFO([("data", 256), ("adr", adr_bits)], 16)
        self.submodules.attn_ofifo = ofifo
        mul_s = mul_factory()
        self.submodules.mul_s = mul_s
        sc_issue = Signal()
        self.comb += [
            sc_issue.eq(sc_run & (ofifo.level < 10)),
            # A group may start once the snapshot bank it will fill is drained.
            snap_ok.eq(mode | (is_unit < sc_unit + 2)),
        ]
        self.sync += [
            If(self.start, caps.eq(0)).Elif(capture, caps.eq(caps + 1)),
            If(self.start,
                sc_run.eq(0), sc_unit.eq(0),
            ).Elif(~sc_run,
                If(caps > sc_unit, sc_run.eq(1), sc_l.eq(0), sc_p.eq(0)),
            ).Elif(sc_issue,
                sc_p.eq(sc_p + 1),
                If(sc_p == 7,
                    sc_l.eq(sc_l + 1),
                    If(sc_l == nq - 1, sc_run.eq(0), sc_unit.eq(sc_unit + 1)),
                )
            )
        ]
        snap_sel = Array(Array(Array(snap[b][l]) for l in range(L)) for b in range(2))
        h1_v, h1_a, h1_p, h1_adr = Signal(), Signal(32), Signal(3), Signal(adr_bits)
        sc_ptr = Signal(adr_bits)   # Lane sc_l's beat of group sc_unit: dst + l*dstr + g.
        self.sync += [
            If(~sc_run & (caps > sc_unit),
                sc_ptr.eq(dst + sc_unit),
            ).Elif(sc_issue & (sc_p == 7),
                sc_ptr.eq(sc_ptr + dstr),
            ),
            h1_v.eq(sc_issue),
            h1_a.eq(snap_sel[sc_unit[0]][sc_l][sc_p]),
            h1_p.eq(sc_p),
            h1_adr.eq(sc_ptr),
        ]
        h2_v, h2_p, h2_adr = dly(h1_v, Lm, "h2v"), dly(h1_p, Lm, "h2p"), dly(h1_adr, Lm, "h2a")
        pack = [Signal(32, name=f"attn_pack{i}") for i in range(7)]
        self.comb += [mul_s.a.eq(h1_a), mul_s.b.eq(scale), mul_s.valid.eq(h1_v)]
        self.sync += If(h2_v, Case(h2_p, {i: pack[i].eq(mul_s.result) for i in range(7)}))

        # mode 1 output: drain the accumulators, hd/8 beats per lane -----------------------------
        dr_run, dr_l, dr_w, dr_v = Signal(), Signal(max=L + 1), Signal(max=hb_max + 1), Signal()
        dr_adr, dr_lane = Signal(adr_bits), Signal(max=L)
        dr_base = Signal(adr_bits)
        dr_done = Signal()
        v_retired = Signal(16)
        self.sync += If(self.start, v_retired.eq(0)).Elif(w_v & w_last & mode, v_retired.eq(v_retired + 1))
        dr_issue = Signal()
        self.comb += dr_issue.eq(dr_run & (ofifo.level < 10))
        for l in range(L):
            self.comb += accmems[l].adr.eq(Mux(dr_run, dr_w, acc_rd_i[3:3 + hbb] if hbb else 0))
        self.sync += [
            dr_v.eq(dr_issue), dr_lane.eq(dr_l), dr_adr.eq(dr_base + dr_w),
            If(self.start,
                dr_run.eq(0), dr_done.eq(0),
            ).Elif(mode & ~dr_run & ~dr_done & params_ready & (v_retired == T) & (T != 0),
                dr_run.eq(1), dr_l.eq(0), dr_w.eq(0), dr_base.eq(dst),
            ).Elif(dr_issue,
                If(dr_w == hb - 1,
                    dr_w.eq(0), dr_l.eq(dr_l + 1), dr_base.eq(dr_base + dstr),
                    If(dr_l == nq - 1, dr_run.eq(0), dr_done.eq(1)),
                ).Else(
                    dr_w.eq(dr_w + 1),
                )
            )
        ]
        self.comb += [
            If(mode,
                ofifo.sink.valid.eq(dr_v),
                ofifo.sink.data.eq(Array([a.dat_r for a in accmems])[dr_lane]),
                ofifo.sink.adr.eq(dr_adr),
            ).Else(
                ofifo.sink.valid.eq(h2_v & (h2_p == 7)),
                ofifo.sink.data.eq(Cat(*pack, mul_s.result)),
                ofifo.sink.adr.eq(h2_adr),
            ),
            self.sp_wr_req.eq(ofifo.source.valid),
            self.sp_wr_adr.eq(ofifo.source.adr),
            self.sp_wr_we.eq(0xff),
            self.sp_wr_dat.eq(ofifo.source.data),
            ofifo.source.ready.eq(self.sp_wr_gnt),
        ]

        # Command -------------------------------------------------------------------------------
        finish = Signal()
        outputs_done = Signal()
        self.comb += [
            outputs_done.eq(Mux(mode, dr_done, (sc_unit == ngroups) & ~sc_run)),
            finish.eq(run & params_ready & outputs_done & (fill_row == T) &
                      ~ofifo.source.valid & ~h1_v & ~h2_v & ~dr_v),
            self.done.eq(finish),
            self.busy.eq(run),
        ]
        self.sync += If(self.start,
            run.eq(1),
        ).Elif(finish | (self.error & run),
            run.eq(0),
        )
        self.progress = [fill_row, is_unit, caps, sc_unit, v_retired, ld_st]

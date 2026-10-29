#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Integer attention passes: attention_stream.py's two O(T) loops on int8 K/V rows.

The FP32 unit does one multiply-add per lane per clock, so at long contexts the attention is
compute-bound (hd clocks per position and pass). Here q, K and V are int8 rows with one
power-of-two exponent per head row (the activation quantize unit's format: e8 =
floor(log2 absmax) - 6, ties away from zero), and each lane does EIGHT integer products per
clock, summed exactly: hd/8 clocks per position and pass. The sums are integers, so the
result does not depend on their order; one FP32 rounding per output.

Rows in engine DDR (both nkv*hd/32 + 1 beats; nkv <= 4):

  K row:  beat 0 = the position's exponents (word j: ek of kv head j, word nkv + j: ev of kv
          head j), then kv head j's hd int8 in beats 1 + j*hd/32 ..
  V row:  kv head j's hd int8 in beats j*hd/32 .., the last beat ignored.

(One quantize descriptor of the four blocks K0 K1 V0 V1 writes exactly this: its exponent
beat in front of the K int8, the V int8 after them.)

  mode 0, scores:  s[l][t] = RNE(f32(dot) * 2^(eq[l] + ek[t]) * scale),
                   dot = sum_i q8[l][i]*k8[t][i] (exact int32). 8 positions interleave per
                   lane; per position hd/8 clocks. The pass also records every position's V
                   exponents and their maximum EV per kv head for the values pass after it.
  mode 1, values:  o[l][i] = RNE(f32(acc[i]) * 2^(EV + E[l] - 14)),
                   acc[i] = sum_t p'[t]*v8[t][i] (exact), p'[t] = min(32767,
                   floor(p[t]*2^(ev[t] - EV + 14 - E[l]) + 1/2)), E[l] = floor(log2 max_t p[t]).
                   One position at a time, hd/8 clocks each.

The record is attention_stream.py's (w0 mode | kv0 << 1 | nq << 8 | n_rep << 16 | nkv << 24,
w1 T, w2 hd, w3 src, w4 src lane stride, w5 dst, w6 dst lane stride, w7 scale). Lane operands:
mode 0, q8 = hd/32 beats then one beat with eq in word 0; mode 1, the FP32 probabilities as the
FP32 unit reads them (ceil(T/8) beats). A values command must follow the scores command of the
same rows (the softmax between them is the caller's).
"""

from functools import reduce
from operator import or_

import numpy as np

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

from gateware.units.fp32 import ExactFP32Mul

# Constants ----------------------------------------------------------------------------------------

MODE_SCORES = 0
MODE_VALUES = 1
P_FRAC      = 14          # p' = p*2^(14 - E): [2^14, 2^15) for the largest p.

# Helpers ------------------------------------------------------------------------------------------

f32 = np.float32

def quantize_rows(x):
    """FP32 rows (last axis) -> (int8, int32 exponents): the activation quantize unit, per row."""
    x = np.asarray(x, f32)
    amax = np.abs(x).max(axis=-1)
    e = np.where(amax > 0, (amax.view(np.uint32) >> 23).astype(np.int64) - 133, 0)
    v = x.astype(np.float64)*np.exp2(-e.astype(np.float64))[..., None]
    q = np.clip(np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)), -127, 127)
    return q.astype(np.int8), e.astype(np.int32)

def _pow2(x, e):
    """f32(x) * 2^e exactly (x a float32 array, e integers): ldexp keeps it exact."""
    return np.ldexp(np.asarray(x, np.float64), np.asarray(e, np.int64)).astype(f32)

def scores_reference(q8, eq, K8, ek, scale):
    """q8 [nq, hd] int8, eq [nq]; K8 [T, nkv, hd] int8, ek [T, nkv] (kv head l // n_rep)."""
    nq, hd = q8.shape
    n_rep = nq//K8.shape[1]
    s = np.zeros((nq, K8.shape[0]), f32)
    for l in range(nq):
        dot = (K8[:, l//n_rep].astype(np.int64) @ q8[l].astype(np.int64))
        s[l] = (_pow2(dot.astype(f32), eq[l] + ek[:, l//n_rep])*f32(scale)).astype(f32)
    return s

def p_exponent(p):
    """floor(log2 max p) of a row of non-negative FP32 (0 when all are zero or subnormal)."""
    b = int(np.asarray(p, f32).view(np.uint32).max() >> 23 & 0xff)
    return b - 127 if b else 0

def p_quantize(p, ev, EV, E):
    """p' of FP32 probabilities p with their positions' V exponents ev (the unit's shifter:
    floor(m*2^s + 1/2) from the FP32 mantissa, subnormals as zero, clipped to 32767)."""
    bits = np.asarray(p, f32).view(np.uint32).astype(np.int64)
    b = bits >> 23 & 0xff
    m = np.where(b > 0, (bits & 0x7fffff) | 0x800000, 0)
    rs = -(b - 127 - 23 + ev - EV + P_FRAC - E)                 # right shift, >= 9 in range
    rs = np.maximum(rs, 1)
    pq = np.where(rs > 26, 0, ((m >> np.minimum(rs - 1, 62)) + 1) >> 1)
    return np.minimum(pq, 32767)

def values_reference(p, V8, ev):
    """p [nq, T] FP32, V8 [T, nkv, hd] int8, ev [T, nkv]; EV = max_t ev per kv head."""
    nq, T = p.shape
    n_rep = nq//V8.shape[1]
    o = np.zeros((nq, V8.shape[2]), f32)
    for l in range(nq):
        j = l//n_rep
        EV, E = int(ev[:, j].max()), p_exponent(p[l])
        pq = p_quantize(p[l], ev[:, j].astype(np.int64), EV, E)
        acc = pq @ V8[:, j].astype(np.int64)
        o[l] = _pow2(acc.astype(f32), EV + E - P_FRAC)
    return o

def k_rows(K8, ek, ev):
    """K8 [T, nkv, hd], ek/ev [T, nkv] -> the K rows' beats as uint32 words [T, rowbeats*8]."""
    T, nkv, hd = K8.shape
    exp = np.zeros((T, 8), np.int32)
    exp[:, :nkv], exp[:, nkv:2*nkv] = ek, ev
    return np.concatenate([exp.view(np.uint32), K8.reshape(T, -1).view(np.uint32)], axis=1)

def v_rows(V8):
    T = V8.shape[0]
    return np.concatenate([V8.reshape(T, -1).view(np.uint32), np.zeros((T, 8), np.uint32)], axis=1)

# Int To FP32 --------------------------------------------------------------------------------------

class IntToFP32Pow2(LiteXModule):
    """r = RNE(f32(x)) * 2^e: x a signed integer of `width` bits, e a signed exponent (the
    result's exponent stays in the normal range: no overflow or underflow handling). Three
    register stages, interval one, no reset (the caller tracks validity)."""
    LATENCY = 3

    def __init__(self, width=40):
        self.x = Signal((width, True))
        self.e = Signal((10, True))
        self.r = Signal(32)

        # # #

        W = width
        # S1: sign and magnitude, leading-zero count.
        s1_sign, s1_mag, s1_e = Signal(), Signal(W), Signal((10, True))
        s1_lz = Signal(max=W + 1)
        mag = Signal(W)
        self.comb += mag.eq(Mux(self.x < 0, -self.x, self.x))
        lz = Signal(max=W + 1)
        lz_cases = W
        for i in range(W):                      # Highest set bit wins (last assignment).
            lz_cases = Mux(mag[i], W - 1 - i, lz_cases)
        self.comb += lz.eq(lz_cases)
        self.sync += [s1_sign.eq(self.x < 0), s1_mag.eq(mag), s1_e.eq(self.e), s1_lz.eq(lz)]
        # S2: normalize (the leading one to bit W-1), round to 24 bits RNE.
        s2_sign, s2_zero, s2_m, s2_exp = Signal(), Signal(), Signal(25), Signal((12, True))
        norm = Signal(W)
        self.comb += norm.eq(s1_mag << s1_lz)
        if W > 24:
            keep, rest = norm[W - 24:], norm[:W - 24]
            half = 1 << (W - 25)
            rnd = (rest > half) | ((rest == half) & keep[0])
        else:
            keep, rnd = norm << (24 - W), 0
        lz_s = Signal((8, True))
        self.comb += lz_s.eq(s1_lz)
        self.sync += [
            s2_sign.eq(s1_sign), s2_zero.eq(s1_mag == 0),
            s2_m.eq(keep + rnd),
            s2_exp.eq(127 + W - 1 - lz_s + s1_e),
        ]
        # S3: a carry out of the rounding renormalizes; pack.
        frac, bexp = Signal(23), Signal((12, True))
        self.comb += If(s2_m[24],
            frac.eq(s2_m[1:24]), bexp.eq(s2_exp + 1),
        ).Else(
            frac.eq(s2_m[:23]), bexp.eq(s2_exp),
        )
        self.sync += self.r.eq(Mux(s2_zero, 0, Cat(frac, bexp[:8], s2_sign)))

# Attention Stream (int8) --------------------------------------------------------------------------

class AttentionStreamI8(LiteXModule):
    """Streaming integer attention passes over `lanes` query heads; see the module docstring.
    Same ports, record and engine op as AttentionStream."""
    def __init__(self, lanes=8, groups=2, hd_max=256, t_max=2048, adr_bits=13,
                 mul_factory=ExactFP32Mul):
        L, P, M = lanes, 8, 8                   # M: products per lane per clock.
        pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        assert pow2(hd_max) and 64 <= hd_max <= 512 and pow2(t_max) and t_max % 8 == 0
        hb_max = hd_max//32                     # int8 beats per head.
        hbb = max((hb_max - 1).bit_length(), 0)
        tb = (t_max//8 - 1).bit_length()
        pw = max((hb_max + 1 - 1).bit_length(), tb)   # Per-lane operand memory address width.
        Lm = mul_factory.latency
        self.lanes, self.groups, self.hd_max, self.t_max = L, groups, hd_max, t_max

        # Control.
        self.start = Signal()
        self.rec   = Signal(adr_bits)
        self.busy  = Signal()
        self.done  = Signal()
        self.sink  = stream.Endpoint([("data", 256)])

        # Scratchpad client ports.
        self.sp_rd_ok  = Signal(reset=1)
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(adr_bits)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(adr_bits)
        self.sp_wr_we  = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()

        # Status.
        self.error = Signal()

        # # #

        def dly(value, n, name, reset_less=True):
            for i in range(n):
                r = Signal.like(value, name=f"{name}_d{i}", reset_less=reset_less)
                self.sync += r.eq(value)
                value = r
            return value

        def chain(value, n, name):
            # A valid's delay line, every stage returned (the command's drain check).
            out = []
            for i in range(n):
                r = Signal(name=f"{name}_c{i}")
                self.sync += r.eq(value)
                out.append(r)
                value = r
            return out

        def word(w, i):
            return w[32*i:32*(i + 1)]

        dat = self.sp_rd_dat
        run, mode = Signal(), Signal()
        nq, n_rep, nkv = (Signal(8, reset_less=True), Signal(8, reset_less=True),
                          Signal(8, reset_less=True))
        kv0 = Signal(7, reset_less=True)
        T = Signal(16, reset_less=True)
        hd = Signal(16, reset_less=True)
        hb = Signal(max=hb_max + 1, reset_less=True)           # int8 beats per head.
        src, sstr, dst, dstr = [Signal(adr_bits, reset_less=True) for _ in range(4)]
        scale = Signal(32, reset_less=True)
        rowbeats = Signal(12, reset_less=True)
        tbeats = Signal(max=t_max//8 + 1, reset_less=True)

        # Record, then the per-lane operands ----------------------------------------------------------
        ld_st = Signal(3)       # 0 idle, 1 record, 6 check, 2 decode, 3 operands, 4 last, 5 ready
        rq_v, rq_adr, rq_lane, rq_word, rq_last = (Signal(), Signal(adr_bits, reset_less=True),
            Signal(max=L, reset_less=True), Signal(pw, reset_less=True), Signal(reset_less=True))
        rsp_v, rsp_lane, rsp_word, rsp_last, rsp_rec = (Signal(), Signal(max=L, reset_less=True),
            Signal(pw, reset_less=True), Signal(reset_less=True), Signal())
        taken, free = Signal(), Signal()
        ld_lane, ld_word = Signal(max=L + 1), Signal(pw + 1)
        ld_words = Signal(pw + 1)
        lane_base = Signal(adr_bits, reset_less=True)
        params_ready = Signal()
        self.comb += [
            taken.eq(rq_v & self.sp_rd_ok),
            free.eq(~rq_v | taken),
            self.sp_rd_req.eq(rq_v),
            self.sp_rd_adr.eq(rq_adr),
            ld_words.eq(Mux(mode, tbeats, hb + 1)),
        ]
        rec_q = Signal(256, reset_less=True)
        rec_w = [rec_q[32*i:32*(i + 1)] for i in range(8)]
        geometry_bad = Signal()
        self.comb += geometry_bad.eq(
            (rec_w[0][8:16] == 0) | (rec_w[0][8:16] > L) | (rec_w[0][16:24] == 0) |
            (rec_w[0][24:32] == 0) | (rec_w[0][24:32] > 4) | (rec_w[0][1:8] >= rec_w[0][24:32]) |
            (rec_w[0][8:16] > rec_w[0][16:24]*groups) |
            (rec_w[0][8:16] > rec_w[0][16:24]*(rec_w[0][24:32] - rec_w[0][1:8])) |
            (rec_w[1] == 0) | (rec_w[1] > t_max) |
            (rec_w[2] < 32) | (rec_w[2] > hd_max) | ((rec_w[2] & (rec_w[2] - 1)) != 0))
        op_v, op_last = Signal(), Signal()
        op_lane, op_word = Signal(max=L, reset_less=True), Signal(pw, reset_less=True)
        op_dat = Signal(256, reset_less=True)
        self.sync += [
            op_v.eq(rsp_v & ~rsp_rec), op_last.eq(rsp_last), op_lane.eq(rsp_lane),
            op_word.eq(rsp_word), op_dat.eq(dat),
        ]
        self.sync += [
            rsp_v.eq(taken), rsp_lane.eq(rq_lane), rsp_word.eq(rq_word), rsp_last.eq(rq_last),
            rsp_rec.eq(taken & (ld_st == 1)),
            If(self.start,
                ld_st.eq(1), rq_v.eq(1), rq_adr.eq(self.rec), self.error.eq(0),
            ).Else(
                If(taken, rq_v.eq(0)),
                Case(ld_st, {
                    1: If(rsp_rec, rec_q.eq(dat), ld_st.eq(6)),
                    6: [
                        If(geometry_bad,
                            self.error.eq(1), ld_st.eq(0),
                        ).Else(
                            mode.eq(rec_w[0][0]), kv0.eq(rec_w[0][1:8]),
                            nq.eq(rec_w[0][8:16]), n_rep.eq(rec_w[0][16:24]),
                            nkv.eq(rec_w[0][24:32]),
                            T.eq(rec_w[1]), hd.eq(rec_w[2]), hb.eq(rec_w[2] >> 5),
                            src.eq(rec_w[3] >> 5), sstr.eq(rec_w[4] >> 5),
                            dst.eq(rec_w[5] >> 5), dstr.eq(rec_w[6] >> 5),
                            scale.eq(rec_w[7]),
                            ld_st.eq(2),
                        )
                    ],
                    2: [
                        rowbeats.eq(nkv*hb + 1),
                        tbeats.eq((T + 7) >> 3),
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
                    4: If(op_v & op_last, ld_st.eq(5)),
                    5: If(self.done | self.error, ld_st.eq(0)),
                })
            )
        ]

        # Per-lane scalars from the operands: eq (mode 0, the exponent beat) and E (mode 1, the
        # largest FP32 exponent field among the T probabilities; padding positions masked).
        eq = [Signal((8, True), name=f"atti_eq{l}", reset_less=True) for l in range(L)]
        pmax = [Signal(8, name=f"atti_pmax{l}", reset_less=True) for l in range(L)]
        wmax = Signal(8)
        wm_expr = 0
        for k in range(8):
            fld = Signal(8, name=f"atti_pf{k}")
            self.comb += fld.eq(Mux((op_word*8 + k) < T, op_dat[32*k + 23:32*k + 31], 0))
            wm_expr = fld if k == 0 else Mux(fld > wm_expr, fld, wm_expr)
        self.comb += wmax.eq(wm_expr)
        for l in range(L):
            self.sync += If(self.start,
                pmax[l].eq(0),
            ).Elif(op_v & (op_lane == l),
                If(~mode & (op_word == hb), eq[l].eq(op_dat[:8])),
                If(mode & (wmax > pmax[l]), pmax[l].eq(wmax)),
            )

        # Row buffers ---------------------------------------------------------------------------------
        rb_bits = hbb + 3 + 1
        rowrd = []
        fill_beat = Signal(12)
        fill_row = Signal(16)
        fill_units = Signal(2)
        cons_units = Signal(2)
        fill_j, fill_idx = Signal(8), Signal(max(hbb, 1))
        fill_p, fill_half = Signal(3), Signal()
        fill_data = Signal()                   # The beat is int8 data (not exponents/padding).
        unit_done = Signal()
        db = Signal(12)                        # Data beat index within the row.
        self.comb += [
            db.eq(Mux(mode, fill_beat, fill_beat - 1)),
            Case(hb, {1 << b: [fill_j.eq(db >> b), fill_idx.eq(db & ((1 << b) - 1))]
                      for b in range(hbb + 1)}),
            fill_data.eq(Mux(mode, fill_beat != rowbeats - 1, fill_beat != 0)),
            fill_p.eq(Mux(mode, 0, fill_row[:3])),
            fill_half.eq(Mux(mode, fill_row[0], fill_row[3])),
            self.sink.ready.eq(run & params_ready & ((fill_units - cons_units)[:2] != 2)),
            unit_done.eq((fill_beat == rowbeats - 1) &
                         (mode | (fill_row[:3] == 7) | (fill_row == T - 1))),
        ]
        take = Signal()
        self.comb += take.eq(self.sink.valid & self.sink.ready)
        self.sync += If(self.start,
            fill_beat.eq(0), fill_row.eq(0), fill_units.eq(0),
        ).Elif(take,
            If(fill_beat == rowbeats - 1,
                fill_beat.eq(0), fill_row.eq(fill_row + 1),
                If(unit_done, fill_units.eq(fill_units + 1)),
            ).Else(
                fill_beat.eq(fill_beat + 1),
            )
        )
        rd_adr = Signal(rb_bits)
        for j in range(groups):
            m = Memory(256, 2*8*hb_max, name=f"atti_row{j}")
            wp = m.get_port(write_capable=True)
            rp = m.get_port()
            self.specials += m, wp, rp
            rowrd.append(rp)
            self.comb += [
                wp.adr.eq(Cat(fill_idx, fill_p, fill_half)),
                wp.dat_w.eq(self.sink.data),
                wp.we.eq(take & fill_data & (fill_j == kv0 + j)),
                rp.adr.eq(rd_adr),
            ]

        # Exponents (mode 0's exponent beats): ek per buffered kv head for the group being
        # filled (4 groups deep: the output may trail the fill by up to three groups), and
        # every position's V exponents with their maximum for the values pass.
        ek = [[[Signal((8, True), name=f"atti_ek{g}_{p}_{j}", reset_less=True) for j in range(groups)]
               for p in range(P)] for g in range(4)]
        exp_beat = Signal()
        self.comb += exp_beat.eq(take & ~mode & (fill_beat == 0))
        fill_group = Signal(2)
        self.comb += fill_group.eq(fill_row[3:5])
        for g in range(4):
            for p in range(P):
                for j in range(groups):
                    self.sync += If(exp_beat & (fill_group == g) & (fill_row[:3] == p),
                        ek[g][p][j].eq(Array([word(self.sink.data, i)[:8] for i in range(4)])[kv0 + j]))
        evm = Memory(32, t_max, name="atti_ev")
        evw = evm.get_port(write_capable=True)
        evr = evm.get_port()
        self.specials += evm, evw, evr
        evs = Array([word(self.sink.data, i)[:8] for i in range(8)])
        self.comb += [
            evw.adr.eq(fill_row), evw.we.eq(exp_beat),
            evw.dat_w.eq(Cat(*[evs[nkv + j] for j in range(4)])),
        ]
        EV = [Signal((8, True), name=f"atti_EV{j}", reset_less=True) for j in range(4)]
        ev_first = Signal()
        self.sync += If(self.start, ev_first.eq(1)).Elif(exp_beat, ev_first.eq(0))
        for j in range(4):
            evj = Signal((8, True), name=f"atti_evj{j}")
            self.comb += evj.eq(evs[nkv + j])
            self.sync += If(exp_beat & (ev_first | (evj > EV[j])), EV[j].eq(evj))

        # Issue -----------------------------------------------------------------------------------------
        # mode 0: unit = 8 positions; per unit (w, s, p) with p fastest: 32*hb clocks.
        # mode 1: unit = a position; per unit (w, s): 4*hb clocks.
        is_run, is_w, is_s, is_p, is_unit = Signal(), Signal(max(hbb, 1)), Signal(2), Signal(3), Signal(16)
        is_units = Signal(16)
        issue = Signal()
        unit_last = Signal()
        snap_ok = Signal()
        self.comb += [
            is_units.eq(Mux(mode, T, tbeats)),
            issue.eq(is_run & (fill_units != cons_units) & snap_ok),
            unit_last.eq((is_w == hb - 1) & (is_s == 3) & (mode | (is_p == 7))),
            rd_adr.eq(Cat(is_w, Mux(mode, 0, is_p), is_unit[0])),
        ]
        self.sync += If(self.start,
            is_run.eq(0), is_unit.eq(0), cons_units.eq(0),
        ).Elif(~is_run,
            If(run & params_ready & (is_unit < is_units),
                is_run.eq(1), is_w.eq(0), is_s.eq(0), is_p.eq(0),
            )
        ).Elif(issue,
            If(mode,
                is_s.eq(is_s + 1),
                If(is_s == 3, is_w.eq(is_w + 1)),
            ).Else(
                is_p.eq(is_p + 1),
                If(is_p == 7,
                    is_s.eq(is_s + 1),
                    If(is_s == 3, is_w.eq(is_w + 1)),
                ),
            ),
            If(unit_last,
                is_run.eq(0), is_unit.eq(is_unit + 1), cons_units.eq(cons_units + 1),
            )
        )
        # S1: the memories' registered read data (row word, operand word, ev word).
        s1_v = Signal()
        s1_s, s1_p, s1_first, s1_last = (Signal(2, reset_less=True), Signal(3, reset_less=True),
                                         Signal(reset_less=True), Signal(reset_less=True))
        s1_aw = Signal(max(hbb, 1) + 2, reset_less=True)        # mode 1 accumulator word.
        s1_unit = Signal(16, reset_less=True)
        s1_tsub = Signal(3, reset_less=True)
        self.sync += [
            s1_v.eq(issue), s1_s.eq(is_s), s1_p.eq(is_p),
            s1_first.eq(Mux(mode, is_unit == 0, (is_w == 0) & (is_s == 0))),
            s1_last.eq(unit_last), s1_aw.eq(Cat(is_s, is_w)), s1_unit.eq(is_unit),
            s1_tsub.eq(is_unit[:3]),
        ]
        self.comb += evr.adr.eq(is_unit)
        # S2: per lane, the selected 64-bit row chunk and operand (q chunk, or p and its ev).
        s2_v = dly(s1_v, 1, "s2v", False)
        s2_first, s2_p, s2_last, s2_aw, s2_unit = (dly(s1_first, 1, "s2f"), dly(s1_p, 1, "s2p"),
            dly(s1_last, 1, "s2l"), dly(s1_aw, 1, "s2aw"), dly(s1_unit, 1, "s2u"))
        # S3: mode 1's p' (the shifter); S4: the products; S5/S6: the adder tree (mode 0).
        s3_v, s3_first, s3_p, s3_last, s3_aw, s3_unit = (dly(s2_v, 1, "s3v", False),
            dly(s2_first, 1, "s3f"), dly(s2_p, 1, "s3p"), dly(s2_last, 1, "s3l"),
            dly(s2_aw, 1, "s3aw"), dly(s2_unit, 1, "s3u"))
        s4_v, s4_first, s4_p, s4_last, s4_aw, s4_unit = (dly(s3_v, 1, "s4v", False),
            dly(s3_first, 1, "s4f"), dly(s3_p, 1, "s4p"), dly(s3_last, 1, "s4l"),
            dly(s3_aw, 1, "s4aw"), dly(s3_unit, 1, "s4u"))
        s6_v, s6_first, s6_p, s6_last, s6_unit = (dly(s4_v, 2, "s6v", False),
            dly(s4_first, 2, "s6f"), dly(s4_p, 2, "s6p"), dly(s4_last, 2, "s6l"),
            dly(s4_unit, 2, "s6u"))

        kv_of = [Signal(max=max(2, groups), name=f"atti_kv{l}", reset_less=True) for l in range(L)]
        kvh_of = [Signal(3, name=f"atti_kvh{l}", reset_less=True) for l in range(L)]   # absolute
        kv_cnt, kv_rep = Signal(8), Signal(8)
        kv_ptr = Signal(max=L + 1)
        self.sync += If(self.start | (ld_st == 2),
            kv_ptr.eq(0), kv_cnt.eq(0), kv_rep.eq(0),
        ).Elif(kv_ptr < L,
            kv_ptr.eq(kv_ptr + 1),
            Case(kv_ptr, {l: [kv_of[l].eq(kv_cnt), kvh_of[l].eq(kv0 + kv_cnt)] for l in range(L)}),
            If(kv_rep == n_rep - 1,
                kv_rep.eq(0), kv_cnt.eq(kv_cnt + 1),
            ).Else(kv_rep.eq(kv_rep + 1)),
        )
        self.comb += params_ready.eq((ld_st == 5) & (kv_ptr == L))

        acc0 = [[Signal((32, True), name=f"atti_acc{l}_{p}", reset_less=True) for p in range(P)]
                for l in range(L)]
        snap = [[[Signal((32, True), name=f"atti_snap{b}_{l}_{p}", reset_less=True) for p in range(P)]
                 for l in range(L)] for b in range(2)]
        capture = Signal()
        cap_bank = Signal()
        self.comb += [capture.eq(s6_v & s6_last & ~mode), cap_bank.eq(s6_unit[0])]
        accrd = []
        AW = 40                                     # mode 1 accumulator width
        s5_aw, s5_v = dly(s4_aw, 1, "s5aw"), dly(s4_v, 1, "s5v", False)
        for l in range(L):
            om = Memory(256, 1 << pw, name=f"atti_op{l}")
            ow = om.get_port(write_capable=True)
            orp = om.get_port(mode=READ_FIRST)       # synchronous: block RAM on any part
            self.specials += om, ow, orp
            self.comb += [
                ow.adr.eq(op_word), ow.dat_w.eq(op_dat),
                ow.we.eq(op_v & (op_lane == l)),
                orp.adr.eq(Mux(mode, is_unit[3:3 + pw], is_w)),
            ]
            ow1 = orp.dat_r
            rw1 = Array([r.dat_r for r in rowrd])[kv_of[l]]
            # S2 registers.
            qa2 = Signal(64, name=f"atti_qa{l}", reset_less=True)
            kb2 = Signal(64, name=f"atti_kb{l}", reset_less=True)
            pv2 = Signal(32, name=f"atti_pv{l}", reset_less=True)
            ev2 = Signal((8, True), name=f"atti_ev2_{l}", reset_less=True)
            self.sync += [
                qa2.eq(Array([ow1[64*i:64*(i + 1)] for i in range(4)])[s1_s]),
                kb2.eq(Array([rw1[64*i:64*(i + 1)] for i in range(4)])[s1_s]),
                pv2.eq(Array([ow1[32*i:32*(i + 1)] for i in range(8)])[s1_tsub]),
                ev2.eq(Array([evr.dat_r[8*i:8*(i + 1)] for i in range(4)])[kvh_of[l]]),
            ]
            # S3: p' = min(32767, floor(m*2^-rs + 1/2)), rs = 136 + EV + E - b - ev (mode 1).
            pb = Signal(8, name=f"atti_pb{l}")
            self.comb += pb.eq(pv2[23:31])
            pm = Signal(24, name=f"atti_pm{l}")
            self.comb += pm.eq(Mux(pb == 0, 0, Cat(pv2[:23], 1)))
            rs = Signal((12, True), name=f"atti_rs{l}")
            pmx_s = Signal((10, True), name=f"atti_pmxs{l}")
            pb_s = Signal((10, True), name=f"atti_pbs{l}")
            EVl = Signal((8, True), name=f"atti_EVl{l}")
            self.comb += [
                pmx_s.eq(pmax[l]), pb_s.eq(pb),
                EVl.eq(Array(EV)[kvh_of[l]]),
                # E = pmax - 127 (0 when pmax == 0): rs = 136 + EV + (pmax - 127) - b - ev.
                rs.eq(Mux(pmax[l] == 0, 136 + EVl - pb_s - ev2, 9 + EVl + pmx_s - pb_s - ev2)),
            ]
            rsc = Signal(5, name=f"atti_rsc{l}")
            self.comb += rsc.eq(Mux(rs < 1, 0, Mux(rs > 26, 26, rs - 1)))
            sh = Signal(25, name=f"atti_sh{l}")
            self.comb += sh.eq((pm >> rsc) + 1)
            pq = Signal(16, name=f"atti_pq{l}")
            pq_full = Signal(25, name=f"atti_pqf{l}")
            self.comb += pq_full.eq(Mux(rs > 26, 0, sh >> 1))
            pa3 = Signal(16, name=f"atti_pa{l}", reset_less=True)
            qa3 = Signal(64, name=f"atti_qa3{l}", reset_less=True)
            kb3 = Signal(64, name=f"atti_kb3{l}", reset_less=True)
            self.sync += [
                pa3.eq(Mux(pq_full > 32767, 32767, pq_full)),
                qa3.eq(qa2), kb3.eq(kb2),
            ]
            # S4: eight products, the a operand a signed 17-bit (q byte or p').
            prods = []
            for k in range(M):
                a = Signal((17, True), name=f"atti_a{l}_{k}")
                b = Signal((8, True), name=f"atti_b{l}_{k}")
                qb_s = Signal((8, True), name=f"atti_qs{l}_{k}")
                pa_s = Signal((17, True), name=f"atti_ps{l}_{k}")
                self.comb += [
                    qb_s.eq(qa3[8*k:8*(k + 1)]), pa_s.eq(pa3),
                    a.eq(Mux(mode, pa_s, qb_s)),
                    b.eq(kb3[8*k:8*(k + 1)]),
                ]
                pr = Signal((25, True), name=f"atti_pr{l}_{k}", reset_less=True)
                self.sync += pr.eq(a*b)
                prods.append(pr)
            # S5/S6: mode 0's tree (8 -> 2 -> 1).
            t5 = [Signal((27, True), name=f"atti_t5_{l}_{i}", reset_less=True) for i in range(2)]
            self.sync += [t5[i].eq(prods[4*i] + prods[4*i + 1] + prods[4*i + 2] + prods[4*i + 3])
                          for i in range(2)]
            t6 = Signal((28, True), name=f"atti_t6_{l}", reset_less=True)
            self.sync += t6.eq(t5[0] + t5[1])
            a0 = Array(acc0[l])
            self.sync += If(s6_v & ~mode, a0[s6_p].eq(Mux(s6_first, 0, a0[s6_p]) + t6))
            for b in range(2):
                self.sync += If(capture & (cap_bank == b),
                    *[snap[b][l][p].eq((Mux(s6_first, 0, acc0[l][p]) + t6) if p == P - 1 else acc0[l][p])
                      for p in range(P)])
            # mode 1: the accumulator word (8 x 40 bits), read at S3, updated at S4, written S5.
            am = Memory(M*AW, 4*hb_max, name=f"atti_vacc{l}")
            aw = am.get_port(write_capable=True)
            ar = am.get_port()
            self.specials += am, aw, ar
            accrd.append(ar)
            new = Signal(M*AW, name=f"atti_vnew{l}", reset_less=True)
            for k in range(M):
                old = Signal((AW, True), name=f"atti_vold{l}_{k}")
                self.comb += old.eq(ar.dat_r[AW*k:AW*(k + 1)])
                self.sync += new[AW*k:AW*(k + 1)].eq(Mux(s4_first, 0, old) + prods[k])
            self.comb += [
                aw.adr.eq(s5_aw), aw.dat_w.eq(new), aw.we.eq(s5_v & mode),
            ]

        # Outputs: one shared converter (int -> FP32 * 2^e) and the scale multiplier ---------------
        conv = IntToFP32Pow2(width=AW)
        self.submodules.conv = conv
        mul_s = mul_factory()
        self.submodules.mul_s = mul_s
        ofifo = stream.SyncFIFO([("data", 256), ("adr", adr_bits)], 16)
        self.submodules.atti_ofifo = ofifo
        one = 0x3f800000

        # mode 0: per captured group, lane by lane, 8 positions.
        sc_run, sc_l, sc_p, sc_unit = Signal(), Signal(max=L + 1), Signal(3), Signal(16)
        caps = Signal(16)
        sc_issue = Signal()
        self.comb += [
            sc_issue.eq(sc_run & (ofifo.level < 8)),
            snap_ok.eq(mode | (is_unit < sc_unit + 2)),
        ]
        sc_ptr = Signal(adr_bits, reset_less=True)
        self.sync += [
            If(self.start, caps.eq(0)).Elif(capture, caps.eq(caps + 1)),
            If(self.start,
                sc_run.eq(0), sc_unit.eq(0),
            ).Elif(~sc_run,
                If(caps > sc_unit, sc_run.eq(1), sc_l.eq(0), sc_p.eq(0), sc_ptr.eq(dst + sc_unit)),
            ).Elif(sc_issue,
                sc_p.eq(sc_p + 1),
                If(sc_p == 7,
                    sc_l.eq(sc_l + 1), sc_ptr.eq(sc_ptr + dstr),
                    If(sc_l == nq - 1, sc_run.eq(0), sc_unit.eq(sc_unit + 1)),
                )
            )
        ]
        snap_sel = Array(Array(Array(snap[b][l]) for l in range(L)) for b in range(2))
        ek_sel = Array(Array(Array(ek[g][p]) for p in range(P)) for g in range(4))
        eq_sel, kvo_sel = Array(eq), Array(kv_of)

        # mode 1: drain the accumulators, lane by lane, hd/8 words of 8 values.
        dr_run, dr_l, dr_w, dr_k, dr_done = Signal(), Signal(max=L + 1), Signal(max(hbb, 1) + 2), Signal(3), Signal()
        dr_issue = Signal()
        dr_base = Signal(adr_bits, reset_less=True)
        v_retired = Signal(16)
        s5_last = dly(s4_last, 1, "s5l")
        self.sync += If(self.start, v_retired.eq(0)).Elif(s5_v & s5_last & mode,
                                                          v_retired.eq(v_retired + 1))
        self.comb += dr_issue.eq(dr_run & (ofifo.level < 8))
        n_words = Signal(max(hbb, 1) + 3)
        self.comb += n_words.eq(hb << 2)
        dr_word = Signal(M*AW, reset_less=True)
        for l in range(L):
            self.comb += accrd[l].adr.eq(Mux(dr_run, dr_w, s3_aw))
        self.sync += [
            If(self.start,
                dr_run.eq(0), dr_done.eq(0),
            ).Elif(mode & ~dr_run & ~dr_done & params_ready & (v_retired == T) & (T != 0),
                dr_run.eq(1), dr_l.eq(0), dr_w.eq(0), dr_k.eq(0), dr_base.eq(dst),
            ).Elif(dr_issue,
                dr_k.eq(dr_k + 1),
                If(dr_k == 7,
                    If(dr_w == n_words - 1,
                        dr_w.eq(0), dr_l.eq(dr_l + 1), dr_base.eq(dr_base + dstr),
                        If(dr_l == nq - 1, dr_run.eq(0), dr_done.eq(1)),
                    ).Else(
                        dr_w.eq(dr_w + 1),
                    )
                )
            )
        ]
        # The accumulator word is read at dr_k == 0 (sync: valid at dr_k == 1); values k are
        # taken one clock after their issue from the registered word, k = 0 straight from
        # the memory.
        d1_v, d1_k, d1_l, d1_adr = (Signal(), Signal(3, reset_less=True),
            Signal(max=L, reset_less=True), Signal(adr_bits, reset_less=True))
        self.sync += [
            d1_v.eq(dr_issue), d1_k.eq(dr_k), d1_l.eq(dr_l), d1_adr.eq(dr_base + dr_w),
        ]
        acc_now = Signal(M*AW)
        self.comb += acc_now.eq(Mux(d1_k == 0, Array([a.dat_r for a in accrd])[d1_l], dr_word))
        self.sync += If(d1_v & (d1_k == 0), dr_word.eq(Array([a.dat_r for a in accrd])[d1_l]))

        # Converter input: mode 0 from the snapshot (registered issue), mode 1 from the drain.
        c_v, c_k, c_adr = Signal(), Signal(3, reset_less=True), Signal(adr_bits, reset_less=True)
        cx = Signal((AW, True), reset_less=True)
        ce = Signal((10, True), reset_less=True)
        evd = Signal((10, True))
        pmd = Signal((10, True))
        self.comb += [
            evd.eq(Array(EV)[Array(kvh_of)[d1_l]]),
            pmd.eq(Array(pmax)[d1_l]),
        ]
        s0_x = Signal((32, True))
        s0_e = Signal((10, True))
        self.comb += [
            s0_x.eq(snap_sel[sc_unit[0]][sc_l][sc_p]),
            s0_e.eq(eq_sel[sc_l] + ek_sel[sc_unit[:2]][sc_p][kvo_sel[sc_l]]),
        ]
        vk = Signal((AW, True))
        self.comb += vk.eq(Array([acc_now[AW*k:AW*(k + 1)] for k in range(M)])[d1_k])
        self.sync += [
            c_v.eq(Mux(mode, d1_v, sc_issue)),
            c_k.eq(Mux(mode, d1_k, sc_p)),
            c_adr.eq(Mux(mode, d1_adr, sc_ptr)),
            cx.eq(Mux(mode, vk, s0_x)),
            ce.eq(Mux(mode, evd + Mux(pmd == 0, 0, pmd - 127) - P_FRAC, s0_e)),
        ]
        self.comb += [conv.x.eq(cx), conv.e.eq(ce)]
        k_vs = chain(c_v, conv.LATENCY, "kv")
        k_v = k_vs[-1]
        k_k, k_adr = dly(c_k, conv.LATENCY, "kk"), dly(c_adr, conv.LATENCY, "kadr")
        self.comb += [mul_s.a.eq(conv.r), mul_s.b.eq(Mux(mode, one, scale)), mul_s.valid.eq(k_v)]
        h_vs = chain(k_v, Lm, "hv")
        h_v = h_vs[-1]
        h_k, h_adr = dly(k_k, Lm, "hk"), dly(k_adr, Lm, "hadr")
        pack = [Signal(32, name=f"atti_pack{i}", reset_less=True) for i in range(7)]
        self.sync += If(h_v, Case(h_k, {i: pack[i].eq(mul_s.result) for i in range(7)}))
        self.comb += [
            ofifo.sink.valid.eq(h_v & (h_k == 7)),
            ofifo.sink.data.eq(Cat(*pack, mul_s.result)),
            ofifo.sink.adr.eq(h_adr),
            self.sp_wr_req.eq(ofifo.source.valid),
            self.sp_wr_adr.eq(ofifo.source.adr),
            self.sp_wr_we.eq(0xff),
            self.sp_wr_dat.eq(ofifo.source.data),
            ofifo.source.ready.eq(self.sp_wr_gnt),
        ]

        # Command ---------------------------------------------------------------------------------------
        finish = Signal()
        outputs_done = Signal()
        pipe_busy = Signal()
        self.comb += [
            pipe_busy.eq(reduce(or_, [s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, d1_v, c_v] + k_vs + h_vs)),
            outputs_done.eq(Mux(mode, dr_done, (sc_unit == tbeats) & ~sc_run)),
            finish.eq(run & params_ready & outputs_done & (fill_row == T) &
                      ~ofifo.source.valid & ~pipe_busy),
            self.done.eq(finish),
            self.busy.eq(run),
        ]
        self.sync += If(self.start, run.eq(1)).Elif(finish | (self.error & run), run.eq(0))
        self.progress = [fill_row, is_unit, caps, sc_unit, v_retired, ld_st]

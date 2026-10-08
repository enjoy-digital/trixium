#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Portable, exact IEEE-754 binary32 arithmetic for the resident units.

Every operation rounds to nearest-even, handles subnormal inputs and results, and returns
RISC-V's canonical quiet NaN (0x7FC00000) for every NaN result - the semantics of the
firmware's RV32F CPU, so a unit built from these reproduces the firmware bit for bit. No
vendor floating-point IP (the Xilinx operators flush subnormals and exist only in Vivado):
the same Verilog is simulated by Verilator and implemented on the FPGA.

  FP32Mul           pipelined a*b: LATENCY register stages (a result is visible after its
                    entry clock plus LATENCY-1 further enabled clocks)
  FP32Add           pipelined a+b (a-b: flip b's sign bit), same interface
  FP32DivSqrt       sequential a/b or sqrt(a) for per-vector scalars, ~30 clocks
  ExactFP32Add/Mul  the same pipelines behind a valid/result_valid/`latency` interface
  FP32ArgmaxTracker the CPU's greedy argmax over a stream of FP32 beats

Shifts are explicit Case multiplexers, never Migen variable shifts (the Verilog width
pitfall). Tests: test/test_fp32_exact.py, compiled, against numpy's IEEE float32.
"""

from migen import *

from litex.gen import LiteXModule

# Constants ----------------------------------------------------------------------------------------

CANONICAL_NAN = 0x7FC00000

# Helpers ------------------------------------------------------------------------------------------

def _shift_right_sticky(value, width, amount, amount_max, out_width):
    """Combinational (value >> amount) truncated to out_width bits, plus a sticky bit that
    is the OR of every bit shifted out below the kept field. amount in [0, amount_max];
    larger amounts behave as amount_max (callers clamp). A logarithmic barrel shifter of
    fixed-width constant-shift stages: no Migen variable shift (the Verilog width pitfall),
    a fraction of the area of a one-hot case mux."""
    q, sticky = Signal(out_width), Signal()
    bits = max(1, amount_max.bit_length())
    stmts = []
    cur = Signal(width, name="rs_in")
    stmts.append(cur.eq(value))
    st = C(0, 1)
    for k in range(bits):
        sh = 1 << k
        nxt, nst = Signal(width, name=f"rs{k}"), Signal(name=f"rs_st{k}")
        lost = cur[:sh] if sh < width else cur
        kept = Cat(cur[sh:], C(0, sh)) if sh < width else C(0, width)
        stmts += [nxt.eq(Mux(amount[k], kept, cur)), nst.eq(st | (amount[k] & (lost != 0)))]
        cur, st = nxt, nst
    over = amount > amount_max                   # beyond the range: everything shifts out
    stmts += [q.eq(Mux(over, 0, cur)), sticky.eq(Mux(over, value != 0, st))]
    return q, sticky, stmts

def _shift_left(value, amount, amount_max, out_width):
    """Combinational (value << amount) truncated to out_width bits, 0 above amount_max: the same
    logarithmic barrel shifter as _shift_right_sticky."""
    out = Signal(out_width)
    bits = max(1, amount_max.bit_length())
    stmts = []
    cur = Signal(out_width, name="ls_in")
    stmts.append(cur.eq(value))
    for k in range(bits):
        sh = 1 << k
        nxt = Signal(out_width, name=f"ls{k}")
        shifted = Cat(C(0, sh), cur[:out_width - sh]) if sh < out_width else C(0, out_width)
        stmts.append(nxt.eq(Mux(amount[k], shifted, cur)))
        cur = nxt
    stmts.append(out.eq(Mux(amount > amount_max, 0, cur)))
    return out, stmts

def _lzc(value, width):
    """Combinational leading-zero count of `value` (width bits); later statements win, so
    scanning from bit 0 upwards leaves the count of the highest set bit."""
    count = Signal(max=width + 1)
    stmts = [count.eq(width)]
    for bit in range(width):
        stmts.append(If(value[bit], count.eq(width - 1 - bit)))
    return count, stmts

# Unpack -------------------------------------------------------------------------------------------

class _Unpack(LiteXModule):
    """Classify and expose a binary32 word: significand with hidden bit, effective biased
    exponent (1 for subnormals), flags."""
    def __init__(self, x):
        self.sign = x[31]
        self.exp  = Signal(8)
        self.frac = Signal(23)
        self.man  = Signal(24)
        self.eff  = Signal(8)
        self.is_nan, self.is_inf, self.is_zero = Signal(), Signal(), Signal()

        # # #

        self.comb += [
            self.exp.eq(x[23:31]), self.frac.eq(x[0:23]),
            self.man.eq(Cat(self.frac, self.exp != 0)),
            self.eff.eq(Mux(self.exp == 0, 1, self.exp)),
            self.is_nan.eq((self.exp == 0xFF) & (self.frac != 0)),
            self.is_inf.eq((self.exp == 0xFF) & (self.frac == 0)),
            self.is_zero.eq((self.exp == 0) & (self.frac == 0)),
        ]

# FP32 Mul -----------------------------------------------------------------------------------------

class FP32Mul(LiteXModule):
    """r = RNE(a * b). Register stages advance on `ce`; results are LATENCY enabled clocks
    behind their operands."""
    LATENCY = 5

    def __init__(self):
        self.a, self.b, self.r = Signal(32), Signal(32), Signal(32)
        self.ce = Signal(reset=1)

        # # #

        # S1: unpack, special cases.
        ua, ub = _Unpack(self.a), _Unpack(self.b)
        self.submodules += ua, ub
        s1_sign, s1_special, s1_sval = Signal(), Signal(), Signal(32)
        s1_ma, s1_mb = Signal(24), Signal(24)
        s1_esum = Signal(10)                      # eff_a + eff_b (2..508)
        sign = Signal()
        self.comb += sign.eq(ua.sign ^ ub.sign)
        self.sync += If(self.ce,
            s1_sign.eq(sign), s1_ma.eq(ua.man), s1_mb.eq(ub.man), s1_esum.eq(ua.eff + ub.eff),
            s1_special.eq(ua.is_nan | ub.is_nan | ua.is_inf | ub.is_inf | ua.is_zero | ub.is_zero),
            If(ua.is_nan | ub.is_nan | (ua.is_inf & ub.is_zero) | (ua.is_zero & ub.is_inf),
                s1_sval.eq(CANONICAL_NAN),
            ).Elif(ua.is_inf | ub.is_inf,
                s1_sval.eq(Cat(C(0, 23), C(0xFF, 8), sign)),
            ).Else(
                s1_sval.eq(Cat(C(0, 31), sign)),
            ),
        )
        # S2: the exact 48-bit product.
        s2_p, s2_sign, s2_special, s2_sval, s2_esum = (Signal(48), Signal(), Signal(), Signal(32),
                                                       Signal(10))
        self.sync += If(self.ce, s2_p.eq(s1_ma*s1_mb), s2_sign.eq(s1_sign),
                        s2_special.eq(s1_special), s2_sval.eq(s1_sval), s2_esum.eq(s1_esum))
        # S3: shift amount. value = P * 2**(esum - 300); keep the lsb at or above 2**-149:
        # s = max(msb - 23, 151 - esum); s < 0 is an exact left shift.
        lz, lz_stmts = _lzc(s2_p, 48)
        self.comb += lz_stmts
        s_norm, s_sub, s_sel = Signal((11, True)), Signal((11, True)), Signal((11, True))
        self.comb += [
            s_norm.eq(C(24, 11) - lz),                       # (47 - lz) - 23
            s_sub.eq(C(151, 11) - s2_esum),
            s_sel.eq(Mux(s_norm > s_sub, s_norm, s_sub)),
        ]
        s3_p, s3_s, s3_sign, s3_special, s3_sval, s3_esum = (Signal(48), Signal((11, True)),
                                                              Signal(), Signal(), Signal(32),
                                                              Signal(10))
        self.sync += If(self.ce, s3_p.eq(s2_p), s3_s.eq(s_sel), s3_sign.eq(s2_sign),
                        s3_special.eq(s2_special), s3_sval.eq(s2_sval), s3_esum.eq(s2_esum))
        # S4: shift to 24 bits + guard + sticky.
        rs = Signal(6)
        self.comb += rs.eq(Mux(s3_s < 0, 0, Mux(s3_s > 50, 50, s3_s[:6])))
        # sh_q: [0] guard, [1:25] kept significand.
        sh_q, sh_sticky, sh_case = _shift_right_sticky(Cat(C(0, 1), s3_p), 49, rs, 50, 25)
        ls = Signal(5)
        self.comb += ls.eq(Mux(s3_s < 0, -s3_s, 0))
        left, left_case = _shift_left(s3_p, ls, 24, 25)
        self.comb += [sh_case, left_case]
        s4_q, s4_g, s4_st, s4_s = Signal(25), Signal(), Signal(), Signal((11, True))
        s4_sign, s4_special, s4_sval, s4_esum = Signal(), Signal(), Signal(32), Signal(10)
        self.sync += If(self.ce,
            If(s3_s < 0, s4_q.eq(left), s4_g.eq(0), s4_st.eq(0),
            ).Else(s4_q.eq(sh_q[1:]), s4_g.eq(sh_q[0]), s4_st.eq(sh_sticky)),
            s4_s.eq(s3_s), s4_sign.eq(s3_sign), s4_special.eq(s3_special),
            s4_sval.eq(s3_sval), s4_esum.eq(s3_esum))
        # S5: round, renormalise, pack.
        self.pack = pack = _RoundPack(s4_sign, s4_q[:24], s4_g, s4_st,
                                       s4_esum, s4_s, bias=-150)
        self.sync += If(self.ce, self.r.eq(Mux(s4_special, s4_sval, pack.word)))

# Round Pack ---------------------------------------------------------------------------------------

class _RoundPack(LiteXModule):
    """RNE-round a 24-bit significand q (plus guard g and sticky st) whose lsb weighs
    2**(esum + s + bias - 127 ... ) and pack: exponent field = esum + s + bias when q is
    normal (bit 23 set) after rounding, 0 for a subnormal; overflow to infinity."""
    def __init__(self, sign, q, g, st, esum, s, bias):
        self.word = Signal(32)

        # # #

        up = Signal()
        q1 = Signal(25)
        e = Signal((12, True))
        man = Signal(24)
        ef = Signal((12, True))
        self.comb += [
            up.eq(g & (st | q[0])),
            q1.eq(q + up),
            e.eq(esum + s + bias),
            If(q1[24], man.eq(q1[1:25]), ef.eq(e + 1)).Else(man.eq(q1[:24]), ef.eq(e)),
            If(~man[23],
                self.word.eq(Cat(man[:23], C(0, 8), sign)),         # subnormal or zero
            ).Elif(ef >= 255,
                self.word.eq(Cat(C(0, 23), C(0xFF, 8), sign)),
            ).Else(
                self.word.eq(Cat(man[:23], ef[:8], sign)),
            ),
        ]

# FP32 Add -----------------------------------------------------------------------------------------

class FP32Add(LiteXModule):
    """r = RNE(a + b), IEEE signed-zero rules (x + -x = +0; -0 + -0 = -0)."""
    LATENCY = 5

    def __init__(self):
        self.a, self.b, self.r = Signal(32), Signal(32), Signal(32)
        self.ce = Signal(reset=1)

        # # #

        ua, ub = _Unpack(self.a), _Unpack(self.b)
        self.submodules += ua, ub

        # S1: order by magnitude, specials, alignment distance.
        swap = Signal()
        self.comb += swap.eq(self.b[:31] > self.a[:31])
        s1_bs, s1_ss = Signal(), Signal()
        s1_bm, s1_sm = Signal(24), Signal(24)
        s1_be = Signal(8)
        s1_d = Signal(8)
        s1_sub = Signal()
        s1_special, s1_sval = Signal(), Signal(32)
        s1_zsign = Signal()
        self.sync += If(self.ce,
            If(swap,
                s1_bs.eq(ub.sign), s1_bm.eq(ub.man), s1_be.eq(ub.eff),
                s1_ss.eq(ua.sign), s1_sm.eq(ua.man), s1_d.eq(ub.eff - ua.eff),
            ).Else(
                s1_bs.eq(ua.sign), s1_bm.eq(ua.man), s1_be.eq(ua.eff),
                s1_ss.eq(ub.sign), s1_sm.eq(ub.man), s1_d.eq(ua.eff - ub.eff),
            ),
            s1_sub.eq(ua.sign ^ ub.sign),
            s1_zsign.eq(ua.sign & ub.sign),
            s1_special.eq(ua.is_nan | ub.is_nan | ua.is_inf | ub.is_inf),
            If(ua.is_nan | ub.is_nan | (ua.is_inf & ub.is_inf & (ua.sign != ub.sign)),
                s1_sval.eq(CANONICAL_NAN),
            ).Elif(ua.is_inf,
                s1_sval.eq(self.a),
            ).Else(
                s1_sval.eq(self.b),
            ),
        )
        # S2: align the smaller significand (3 extra bits: guard, round, sticky).
        dq, dst, dcase = _shift_right_sticky(Cat(C(0, 3), s1_sm), 27, s1_d, 27, 27)
        self.comb += dcase
        s2_big, s2_small = Signal(27), Signal(27)
        s2_bs, s2_sub, s2_be, s2_special, s2_sval, s2_zsign = (Signal(), Signal(), Signal(8),
                                                              Signal(), Signal(32), Signal())
        self.sync += If(self.ce,
            s2_big.eq(Cat(C(0, 3), s1_bm)), s2_small.eq(dq | dst),
            s2_bs.eq(s1_bs), s2_sub.eq(s1_sub), s2_be.eq(s1_be),
            s2_special.eq(s1_special), s2_sval.eq(s1_sval), s2_zsign.eq(s1_zsign))
        # S3: add or subtract (|big| >= |small|, so the difference is non-negative).
        s3_sum = Signal(28)
        s3_bs, s3_be, s3_special, s3_sval, s3_zsign, s3_sub = (Signal(), Signal(8), Signal(),
                                                               Signal(32), Signal(), Signal())
        self.sync += If(self.ce,
            s3_sum.eq(Mux(s2_sub, s2_big - s2_small, s2_big + s2_small)),
            s3_bs.eq(s2_bs), s3_be.eq(s2_be), s3_special.eq(s2_special), s3_sval.eq(s2_sval),
            s3_zsign.eq(s2_zsign), s3_sub.eq(s2_sub))
        # S4: normalise. A carry shifts right one (sticky kept); otherwise shift left by the
        # leading zeros, but never below the subnormal exponent (1).
        lz, lz_stmts = _lzc(s3_sum[:27], 27)
        self.comb += lz_stmts
        lmax = Signal(9)
        lsh = Signal(5)
        self.comb += [
            lmax.eq(s3_be - 1),
            lsh.eq(Mux(lz > lmax, lmax, lz)),
        ]
        left, lcase = _shift_left(s3_sum[:27], lsh, 26, 27)
        self.comb += lcase
        s4_n = Signal(27)
        s4_e = Signal((11, True))
        s4_bs, s4_special, s4_sval, s4_zero, s4_zsign = (Signal(), Signal(), Signal(32), Signal(),
                                                         Signal())
        self.sync += If(self.ce,
            If(s3_sum[27],
                s4_n.eq(Cat(s3_sum[1] | s3_sum[0], s3_sum[2:28])), s4_e.eq(s3_be + 1),
            ).Else(
                s4_n.eq(left), s4_e.eq(s3_be - lsh),
            ),
            s4_bs.eq(s3_bs), s4_special.eq(s3_special), s4_sval.eq(s3_sval),
            s4_zero.eq(s3_sum == 0), s4_zsign.eq(s3_zsign))
        # S5: round (n = 24-bit significand, guard, round|sticky) and pack.
        self.pack = pack = _RoundPack(s4_bs, s4_n[3:27], s4_n[2], s4_n[1] | s4_n[0],
                                       s4_e, C(0, 2), bias=0)
        self.sync += If(self.ce, self.r.eq(
            Mux(s4_special, s4_sval, Mux(s4_zero, Cat(C(0, 31), s4_zsign), pack.word))))

# FP32 Div/Sqrt ------------------------------------------------------------------------------------

class FP32DivSqrt(LiteXModule):
    """Sequential r = RNE(a / b) (op 0) or RNE(sqrt(a)) (op 1). Pulse `start` with the
    operands valid; `done` pulses (r valid) ~30 clocks later; `busy` in between.

    Both reduce to an integer result R (27 bits for the quotient, 26 for the root), an
    exact-remainder flag, and the weight 2**L0 of R's lsb; the common packer keeps 24 bits,
    rounds with guard and sticky, and handles the subnormal range and overflow.
      div:  na, nb normalised (leading one at bit 23), R = floor(na * 2**26 / nb),
            L0 = (Ea - Eb) - 26 with Ex = eff - lz (the operands' leading-bit exponents).
      sqrt: a = na * 2**(E - 23), E = eff - lz - 127; X = na * 2**k, k in {26, 27} with
            E - 23 - k even; R = floor(sqrt(X)) (25..26 bits), L0 = (E - 23 - k) / 2.
    """

    def __init__(self):
        self.a, self.b, self.op = Signal(32), Signal(32), Signal()
        self.start, self.busy, self.done = Signal(), Signal(), Signal()
        self.r = Signal(32)

        # # #

        # Operand normalization.
        ua, ub = _Unpack(self.a), _Unpack(self.b)
        self.submodules += ua, ub
        lza, lza_s = _lzc(ua.man, 24)
        lzb, lzb_s = _lzc(ub.man, 24)
        self.comb += lza_s + lzb_s
        na, la = _shift_left(ua.man, lza, 23, 24)
        nb, lb = _shift_left(ub.man, lzb, 23, 24)
        self.comb += [la, lb]
        ex_a, ex_b, e_sq = Signal((11, True)), Signal((11, True)), Signal((11, True))
        self.comb += [
            ex_a.eq(ua.eff - lza), ex_b.eq(ub.eff - lzb),
            e_sq.eq(ex_a - 127),
        ]

        # Iteration state.
        op, sign, special = Signal(), Signal(), Signal()
        sval = Signal(32)
        rem = Signal(30)
        den = Signal(24)
        quo = Signal(27)
        rad = Signal(52)
        cnt = Signal(5)
        l0 = Signal((12, True))
        IDLE, RUN, PACK, PACK2 = range(4)
        state = Signal(2)
        self.comb += self.busy.eq(state != IDLE)

        # Division step: bit = rem >= den; rem = (rem - bit*den) << 1.
        ge = Signal()
        diff = Signal(30)
        self.comb += [ge.eq(rem >= den), diff.eq(rem - den)]
        # Square-root step: r2 = (rem << 2) | next two radicand bits; t = (root << 2) | 1.
        r2 = Signal(30)
        t = Signal(30)
        sq_ge = Signal()
        self.comb += [
            r2.eq(Cat(rad[50:52], rem[:28])),
            t.eq(Cat(C(1, 1), C(0, 1), quo)),
            sq_ge.eq(r2 >= t),
        ]

        # Packing: keep 24 bits of quo (leading one at `msb`) with guard and sticky, shifted
        # further right when the lsb weight would fall below 2**-149.
        nq_lz, nq_s = _lzc(quo, 27)
        self.comb += nq_s
        s_n, s_s, s_sel = Signal((12, True)), Signal((12, True)), Signal((12, True))
        self.comb += [
            s_n.eq(C(3, 12) - nq_lz),               # (26 - lz) - 23
            s_s.eq(-149 - l0),
            s_sel.eq(Mux(s_n > s_s, s_n, s_s)),
        ]
        rsh = Signal(6)
        self.comb += rsh.eq(Mux(s_sel < 0, 0, Mux(s_sel > 30, 30, s_sel[:6])))
        kq, ks, kcase = _shift_right_sticky(Cat(C(0, 1), quo), 28, rsh, 30, 25)
        lsq = Signal(5)
        self.comb += lsq.eq(Mux(s_sel < 0, -s_sel, 0))
        kleft, klcase = _shift_left(quo, lsq, 23, 24)
        self.comb += [kcase, klcase]
        pk_q, pk_g, pk_st = Signal(24), Signal(), Signal()
        self.comb += If(s_sel < 0,
            pk_q.eq(kleft), pk_g.eq(0), pk_st.eq(rem != 0),
        ).Else(
            pk_q.eq(kq[1:25]), pk_g.eq(kq[0]), pk_st.eq(ks | (rem != 0)),
        )
        shift_used = Signal((12, True))
        self.comb += shift_used.eq(Mux(s_sel < 0, s_sel, rsh))
        # The count/shift and the round/pack are split over two clocks (PACK, PACK2) for timing:
        # as one they are ~26 logic levels.
        r_q, r_g, r_st, r_s = Signal(24), Signal(), Signal(), Signal((12, True))
        self.pack = pack = _RoundPack(sign, r_q, r_g, r_st, l0, r_s, bias=150)

        # Control.
        self.sync += [
            self.done.eq(0),
            Case(state, {
                IDLE: If(self.start,
                    op.eq(self.op), cnt.eq(0), quo.eq(0),
                    If(self.op == 0,
                        sign.eq(ua.sign ^ ub.sign),
                        special.eq(ua.is_nan | ub.is_nan | ua.is_inf | ub.is_inf |
                                   ua.is_zero | ub.is_zero),
                        If(ua.is_nan | ub.is_nan | (ua.is_inf & ub.is_inf) |
                           (ua.is_zero & ub.is_zero),
                            sval.eq(CANONICAL_NAN),
                        ).Elif(ua.is_inf | ub.is_zero,
                            sval.eq(Cat(C(0, 23), C(0xFF, 8), ua.sign ^ ub.sign)),
                        ).Else(
                            sval.eq(Cat(C(0, 31), ua.sign ^ ub.sign)),
                        ),
                        rem.eq(na), den.eq(nb),
                        l0.eq(ex_a - ex_b - 26),
                    ).Else(
                        sign.eq(0),
                        special.eq(ua.is_nan | ua.is_inf | ua.is_zero | ua.sign),
                        If(ua.is_nan | (ua.sign & ~ua.is_zero),
                            sval.eq(CANONICAL_NAN),
                        ).Else(sval.eq(self.a)),               # +inf, +0, -0 stay
                        rem.eq(0),
                        If(e_sq[0],                            # E odd: E - 23 even, k = 26
                            rad.eq(na << 26), l0.eq((e_sq - 23 - 26) >> 1),
                        ).Else(                                # E even: k = 27
                            rad.eq(na << 27), l0.eq((e_sq - 23 - 27) >> 1),
                        ),
                    ),
                    state.eq(RUN),
                ),
                RUN: [
                    cnt.eq(cnt + 1),
                    If(op == 0,
                        quo.eq(Cat(ge, quo[:26])),
                        rem.eq(Mux(ge, diff, rem) << 1),
                        If(cnt == 26, state.eq(PACK)),
                    ).Else(
                        quo.eq(Cat(sq_ge, quo[:26])),
                        rem.eq(Mux(sq_ge, r2 - t, r2)),
                        rad.eq(rad << 2),
                        If(cnt == 25, state.eq(PACK)),
                    ),
                ],
                PACK: [
                    r_q.eq(pk_q), r_g.eq(pk_g), r_st.eq(pk_st), r_s.eq(shift_used),
                    state.eq(PACK2),
                ],
                PACK2: [
                    self.r.eq(Mux(special, sval, pack.word)),
                    self.done.eq(1), state.eq(IDLE),
                ],
            }),
        ]

# Valid Adapters -----------------------------------------------------------------------------------

class _ValidAdapter(LiteXModule):
    """A vendor-operator style interface (a, b, valid -> result, result_valid, `latency`) over
    an exact pipeline, for the streaming units (DeltaNet recurrence, attention passes)."""
    def __init__(self, op_cls):
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()

        # # #

        self.op = op = op_cls()
        self.comb += [op.a.eq(self.a), op.b.eq(self.b), self.result.eq(op.r)]
        prev = self.valid
        for i in range(op_cls.LATENCY):
            v = Signal(name=f"valid{i}")
            self.sync += v.eq(prev)
            prev = v
        self.comb += self.result_valid.eq(prev)

class ExactFP32Add(_ValidAdapter):
    latency = FP32Add.LATENCY

    def __init__(self):
        _ValidAdapter.__init__(self, FP32Add)

class ExactFP32Mul(_ValidAdapter):
    latency = FP32Mul.LATENCY

    def __init__(self):
        _ValidAdapter.__init__(self, FP32Mul)

# FP32 Argmax Tracker ------------------------------------------------------------------------------

class FP32ArgmaxTracker(LiteXModule):
    """The CPU's greedy argmax over a stream of FP32 beats, bit for bit:

        best = x[0]; idx = 0; for i >= 1: if (x[i] > best) { best = x[i]; idx = i; }

    - a NaN never compares greater, and a NaN as the very first element freezes the result
      at index 0 (every later comparison with it is false);
    - -0 and +0 are equal (the earlier one stays); ties keep the earlier index.
    `beat` delivers 8 lanes (lane j = element count + j) with `mask` of active lanes (a
    prefix); state persists across beats and commands until `clear`. Pipelined: a 3-level
    tree over the lanes, then the combination with the running best, 5 clocks behind.
    """
    def __init__(self):
        self.clear = Signal()
        self.valid, self.data, self.mask = Signal(), Signal(256), Signal(8)
        self.best_valid, self.best_val, self.best_idx = Signal(), Signal(32), Signal(32)

        # # #

        count = Signal(32)
        # S0: register the beat with its base index and whether it starts the stream.
        s0_v, s0_d, s0_m, s0_base, s0_first = Signal(), Signal(256), Signal(8), Signal(32), Signal()
        started = Signal()
        active = Signal(4)                     # lanes in this beat (the mask is a prefix)
        self.comb += active.eq(sum(self.mask[j] for j in range(8)))
        self.sync += If(self.clear, count.eq(0), started.eq(0), s0_v.eq(0)).Else(
            s0_v.eq(self.valid), s0_d.eq(self.data), s0_m.eq(self.mask), s0_base.eq(count),
            s0_first.eq(self.valid & ~started),
            If(self.valid, count.eq(count + active), started.eq(1)))

        def key(x):
            k = Signal(32)
            zero = x[:31] == 0
            self.comb += k.eq(Mux(zero, 0x80000000, Mux(x[31], ~x, x | 0x80000000)))
            return k

        def nan(x):
            return (x[23:31] == 0xFF) & (x[:23] != 0)

        # S1: per-lane candidates (valid, key, value, lane) and the first lane's NaN flag.
        cands = []
        for j in range(8):
            x = s0_d[32*j:32*j + 32]
            cands.append((s0_m[j] & ~nan(x), key(x), x, C(j, 3)))
        s1 = []
        for a, b in zip(cands[0::2], cands[1::2]):
            v, k, x, l = Signal(), Signal(32), Signal(32), Signal(3)
            take_a = a[0] & (~b[0] | (a[1] >= b[1]))
            self.sync += [v.eq(a[0] | b[0]), k.eq(Mux(take_a, a[1], b[1])),
                          x.eq(Mux(take_a, a[2], b[2])), l.eq(Mux(take_a, a[3], b[3]))]
            s1.append((v, k, x, l))
        s1_v, s1_base, s1_first, s1_nan0 = Signal(), Signal(32), Signal(), Signal()
        s1_x0 = Signal(32)
        self.sync += [s1_v.eq(s0_v & ~self.clear), s1_base.eq(s0_base), s1_first.eq(s0_first),
                      s1_nan0.eq(nan(s0_d[:32])), s1_x0.eq(s0_d[:32])]
        level, pipe = s1, [(s1_v, s1_base, s1_first, s1_nan0, s1_x0)]
        for depth in (2, 3):
            nxt = []
            for a, b in zip(level[0::2], level[1::2]):
                v, k, x, l = Signal(), Signal(32), Signal(32), Signal(3)
                take_a = a[0] & (~b[0] | (a[1] >= b[1]))
                self.sync += [v.eq(a[0] | b[0]), k.eq(Mux(take_a, a[1], b[1])),
                              x.eq(Mux(take_a, a[2], b[2])), l.eq(Mux(take_a, a[3], b[3]))]
                nxt.append((v, k, x, l))
            level = nxt
            pv, pb, pf, pn, px = pipe[-1]
            q = (Signal(), Signal(32), Signal(), Signal(), Signal(32))
            self.sync += [q[0].eq(pv & ~self.clear), q[1].eq(pb), q[2].eq(pf), q[3].eq(pn),
                          q[4].eq(px)]
            pipe.append(q)
        (wv, wk, wx, wl), = level
        bv, bbase, bfirst, bnan0, bx0 = pipe[-1]
        # Combine with the running best.
        frozen, rkey = Signal(), Signal(32)
        self.sync += If(self.clear,
            self.best_valid.eq(0), frozen.eq(0),
        ).Elif(bv,
            If(bfirst,
                self.best_valid.eq(1),
                If(bnan0,
                    frozen.eq(1), self.best_val.eq(bx0), self.best_idx.eq(bbase),
                ).Else(
                    rkey.eq(wk), self.best_val.eq(wx), self.best_idx.eq(bbase + wl),
                ),
            ).Elif(~frozen & wv & (wk > rkey),
                rkey.eq(wk), self.best_val.eq(wx), self.best_idx.eq(bbase + wl),
            ),
        )

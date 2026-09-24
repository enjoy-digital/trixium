# SPDX-License-Identifier: BSD-2-Clause
"""Portable, exact IEEE-754 binary32 arithmetic for the resident units.

Every operation rounds to nearest-even, handles subnormal inputs and results, and returns
RISC-V's canonical quiet NaN (0x7FC00000) for every NaN result - the semantics of the
firmware's RV32F CPU, so a unit built from these reproduces the firmware bit for bit. No
vendor floating-point IP (the Xilinx operators flush subnormals and exist only in Vivado):
the same Verilog is simulated by Verilator and implemented on the FPGA.

  FP32Mul     pipelined a*b: LATENCY register stages (a result is visible after its
              entry clock plus LATENCY-1 further enabled clocks)
  FP32Add     pipelined a+b (a-b: flip b's sign bit), same interface
  FP32DivSqrt sequential a/b or sqrt(a) for per-vector scalars, ~30 clocks

Shifts are explicit Case multiplexers, never Migen variable shifts (the Verilog width
pitfall). Tests: test/test_fp32_exact.py, compiled, against numpy's IEEE float32.
"""
from migen import *
from litex.gen import LiteXModule

CANONICAL_NAN = 0x7FC00000


def _shift_right_sticky(value, width, amount, amount_max, out_width):
    """Combinational (value >> amount) truncated to out_width bits, plus a sticky bit that
    is the OR of every bit shifted out below the kept field. amount in [0, amount_max]."""
    q, sticky = Signal(out_width), Signal()
    cases = {}
    for s in range(amount_max + 1):
        kept = value[s:] if s < width else C(0, 1)
        lost = value[:s] if s else C(0, 1)
        cases[s] = [q.eq(kept), sticky.eq(lost != 0 if s else 0)]
    cases["default"] = [q.eq(0), sticky.eq(value != 0)]
    return q, sticky, Case(amount, cases)


def _shift_left(value, amount, amount_max, out_width):
    out = Signal(out_width)
    cases = {s: out.eq(value << s) for s in range(amount_max + 1)}
    cases["default"] = out.eq(0)
    return out, Case(amount, cases)


def _lzc(value, width):
    count = Signal(max=width + 1)
    stmts = [count.eq(width)]
    for bit in range(width):
        stmts.append(If(value[bit], count.eq(width - 1 - bit)))
    return count, stmts


class _Unpack(Module):
    """Classify and expose a binary32 word: significand with hidden bit, effective biased
    exponent (1 for subnormals), flags."""
    def __init__(self, x):
        self.sign = x[31]
        self.exp = Signal(8)
        self.frac = Signal(23)
        self.man = Signal(24)
        self.eff = Signal(8)
        self.is_nan, self.is_inf, self.is_zero = Signal(), Signal(), Signal()
        self.comb += [
            self.exp.eq(x[23:31]), self.frac.eq(x[0:23]),
            self.man.eq(Cat(self.frac, self.exp != 0)),
            self.eff.eq(Mux(self.exp == 0, 1, self.exp)),
            self.is_nan.eq((self.exp == 0xFF) & (self.frac != 0)),
            self.is_inf.eq((self.exp == 0xFF) & (self.frac == 0)),
            self.is_zero.eq((self.exp == 0) & (self.frac == 0)),
        ]


class FP32Mul(LiteXModule):
    """r = RNE(a * b). Register stages advance on `ce`; results are LATENCY enabled clocks
    behind their operands."""
    LATENCY = 5

    def __init__(self):
        self.a, self.b, self.r = Signal(32), Signal(32), Signal(32)
        self.ce = Signal(reset=1)

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
        s2_p, s2_sign, s2_special, s2_sval, s2_esum = Signal(48), Signal(), Signal(), Signal(32), Signal(10)
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
        s3_p, s3_s, s3_sign, s3_special, s3_sval, s3_esum = (Signal(48), Signal((11, True)), Signal(),
                                                              Signal(), Signal(32), Signal(10))
        self.sync += If(self.ce, s3_p.eq(s2_p), s3_s.eq(s_sel), s3_sign.eq(s2_sign),
                        s3_special.eq(s2_special), s3_sval.eq(s2_sval), s3_esum.eq(s2_esum))
        # S4: shift to 24 bits + guard + sticky.
        rs = Signal(6)
        self.comb += rs.eq(Mux(s3_s < 0, 0, Mux(s3_s > 50, 50, s3_s[:6])))
        sh = Signal(49)                          # [0] guard, [1:25] kept significand
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
        self.submodules.pack = pack = _RoundPack(s4_sign, s4_q[:24], s4_g, s4_st,
                                                  s4_esum, s4_s, bias=-150)
        self.sync += If(self.ce, self.r.eq(Mux(s4_special, s4_sval, pack.word)))


class _RoundPack(Module):
    """RNE-round a 24-bit significand q (plus guard g and sticky st) whose lsb weighs
    2**(esum + s + bias - 127 ... ) and pack: exponent field = esum + s + bias when q is
    normal (bit 23 set) after rounding, 0 for a subnormal; overflow to infinity."""
    def __init__(self, sign, q, g, st, esum, s, bias):
        self.word = Signal(32)
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


class FP32Add(LiteXModule):
    """r = RNE(a + b), IEEE signed-zero rules (x + -x = +0; -0 + -0 = -0)."""
    LATENCY = 5

    def __init__(self):
        self.a, self.b, self.r = Signal(32), Signal(32), Signal(32)
        self.ce = Signal(reset=1)
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
        s2_bs, s2_sub, s2_be, s2_special, s2_sval, s2_zsign = (Signal(), Signal(), Signal(8), Signal(),
                                                              Signal(32), Signal())
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
        s4_bs, s4_special, s4_sval, s4_zero, s4_zsign = Signal(), Signal(), Signal(32), Signal(), Signal()
        self.sync += If(self.ce,
            If(s3_sum[27],
                s4_n.eq(Cat(s3_sum[1] | s3_sum[0], s3_sum[2:28])), s4_e.eq(s3_be + 1),
            ).Else(
                s4_n.eq(left), s4_e.eq(s3_be - lsh),
            ),
            s4_bs.eq(s3_bs), s4_special.eq(s3_special), s4_sval.eq(s3_sval),
            s4_zero.eq(s3_sum == 0), s4_zsign.eq(s3_zsign))
        # S5: round (n = 24-bit significand, guard, round|sticky) and pack.
        self.submodules.pack = pack = _RoundPack(s4_bs, s4_n[3:27], s4_n[2], s4_n[1] | s4_n[0],
                                                  s4_e, C(0, 2), bias=0)
        self.sync += If(self.ce, self.r.eq(
            Mux(s4_special, s4_sval, Mux(s4_zero, Cat(C(0, 31), s4_zsign), pack.word))))


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

        op, sign, special = Signal(), Signal(), Signal()
        sval = Signal(32)
        rem = Signal(30)
        den = Signal(24)
        quo = Signal(27)
        rad = Signal(52)
        cnt = Signal(5)
        l0 = Signal((12, True))
        IDLE, RUN, PACK = range(3)
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
        self.submodules.pack = pack = _RoundPack(sign, pk_q, pk_g, pk_st, l0, shift_used, bias=150)

        self.sync += [
            self.done.eq(0),
            Case(state, {
                IDLE: If(self.start,
                    op.eq(self.op), cnt.eq(0), quo.eq(0),
                    If(self.op == 0,
                        sign.eq(ua.sign ^ ub.sign),
                        special.eq(ua.is_nan | ub.is_nan | ua.is_inf | ub.is_inf |
                                   ua.is_zero | ub.is_zero),
                        If(ua.is_nan | ub.is_nan | (ua.is_inf & ub.is_inf) | (ua.is_zero & ub.is_zero),
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
                    self.r.eq(Mux(special, sval, pack.word)),
                    self.done.eq(1), state.eq(IDLE),
                ],
            }),
        ]


class _ValidAdapter(LiteXModule):
    """The vendor-operator interface (a, b, valid -> result, result_valid, `latency`) over
    an exact pipeline, for units written against gateware/fp32.py (the DeltaNet recurrence)."""
    def __init__(self, op_cls):
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()
        self.submodules.op = op = op_cls()
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

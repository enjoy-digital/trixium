# SPDX-License-Identifier: BSD-2-Clause
"""SiluMulKernel: r = m * siluf(g) on IEEE-754 binary32, bit-exact with the firmware.

Portable Migen (no vendor floating-point IP), one element per cycle, a globally stalled
pipeline. The contract and the arithmetic are in model/silu_hw.py; this is
`silu_mul_fixed()` stage for stage:

  S1  classify g/m; a20 = RNE_f32(g + 16) * 2**20 as a shift-and-round of g's significand
  S2  i = a20 >> 13, F = a20 & 0x1FFF; read lut[i] (lut[4096] at the endpoint) and dlut[i]
  S3  table words arrive; P = F * d.significand (13 x 24 bits); unpack c = lut[i]
  S4  align the smaller of P and c to the larger with a sticky bit
  S5  add or subtract, take the magnitude and sign
  S6  count leading zeros
  S7  normalize and round to 24 bits (RNE): s = fma(F/8192, d, c); select s (pass / -0 / endpoint)
  S8  m.significand * s.significand (24 x 24 bits)
  S9  normalize and round the product (RNE), encode r; flag a non-normal product

`fixup` is set when the element is outside the fast path (the polynomial tails
-80 <= g <= -16 and g == 16, non-finite g or m, a subnormal m, a product that is not a
normal number): `r` is then meaningless and the CPU recomputes that element with the
firmware's own code. Every other `r` equals hybrid.c's `mul * siluf(gate)`.

Tables: write lut[0..4096] and dlut[0..4095] (dlut[i] = lut[i+1] - lut[i], computed by the
firmware with one FP32 subtraction) through the write ports before streaming; they are
not modified by streaming. Both are read in S2 with synchronous read ports.
"""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

LUT_N = 4096
IN_LAYOUT = [("g", 32), ("m", 32)]
OUT_LAYOUT = [("r", 32), ("fixup", 1)]

KIND_TABLE, KIND_PASS, KIND_NEGZERO, KIND_FIXUP = range(4)


def _lzc(value, width):
    """Leading-zero count of `value` (width bits), combinational. Later statements win in
    Migen, so scanning from bit 0 upwards leaves the count of the highest set bit."""
    count = Signal(max=width + 1)
    stmts = [count.eq(width)]
    for bit in range(width):
        stmts.append(If(value[bit], count.eq(width - 1 - bit)))
    return count, stmts


class SiluMulKernel(LiteXModule):
    LATENCY = 9

    def __init__(self):
        self.sink = sink = stream.Endpoint(IN_LAYOUT)
        self.source = source = stream.Endpoint(OUT_LAYOUT)
        self.lut_we, self.lut_adr, self.lut_dat = Signal(), Signal(13), Signal(32)
        self.dlut_we, self.dlut_adr, self.dlut_dat = Signal(), Signal(12), Signal(32)

        lut = Memory(32, LUT_N + 1)
        dlut = Memory(32, LUT_N)
        lut_w, dlut_w = lut.get_port(write_capable=True), dlut.get_port(write_capable=True)
        lut_r = lut.get_port(has_re=True, mode=READ_FIRST)
        dlut_r = dlut.get_port(has_re=True, mode=READ_FIRST)
        self.specials += lut, dlut, lut_w, dlut_w, lut_r, dlut_r
        self.comb += [
            lut_w.we.eq(self.lut_we), lut_w.adr.eq(self.lut_adr), lut_w.dat_w.eq(self.lut_dat),
            dlut_w.we.eq(self.dlut_we), dlut_w.adr.eq(self.dlut_adr), dlut_w.dat_w.eq(self.dlut_dat),
        ]

        # Global stall: every stage advances together when the output can move.
        n = self.LATENCY
        valid = [Signal(name=f"v{k}") for k in range(n + 1)]   # valid[k]: output of stage k
        ce = Signal()
        self.comb += [
            ce.eq(~valid[n] | source.ready),
            sink.ready.eq(ce),
            valid[0].eq(sink.valid),
            source.valid.eq(valid[n]),
        ]
        for k in range(1, n + 1):
            self.sync += If(ce, valid[k].eq(valid[k - 1]))

        def carry(name, width, src):
            """A signal registered through the remaining stages: list indexed by stage."""
            regs = [src]
            for k in range(1, n + 1):
                r = Signal(width, name=f"{name}{k}")
                self.sync += If(ce, r.eq(regs[-1]))
                regs.append(r)
            return regs

        g = carry("g", 32, sink.g)
        m = carry("m", 32, sink.m)

        # S1: classify and add 16 ------------------------------------------------------------
        gs, ge, gf = sink.g[31], sink.g[23:31], sink.g[0:23]
        me, mf = sink.m[23:31], sink.m[0:23]
        mag = sink.g[0:31]
        k_table, k_pass, k_negzero = Signal(), Signal(), Signal()
        bad = Signal()
        kind0 = Signal(2)
        self.comb += [
            bad.eq((ge == 0xFF) | (me == 0xFF) | ((me == 0) & (mf != 0))),
            k_table.eq(mag < 0x41800000),
            k_pass.eq(~gs & (mag > 0x41800000)),
            k_negzero.eq(gs & (mag > 0x42A00000)),
            If(bad, kind0.eq(KIND_FIXUP)
            ).Elif(k_table, kind0.eq(KIND_TABLE)
            ).Elif(k_pass, kind0.eq(KIND_PASS)
            ).Elif(k_negzero, kind0.eq(KIND_NEGZERO)
            ).Else(kind0.eq(KIND_FIXUP)),
        ]
        kind = carry("kind", 2, kind0)

        # a20 = RNE_f32(g + 16) * 2**20 (see model.silu_hw.add16_fixed).
        sig = Signal(24)
        ge_eff = Signal(8)
        sh = Signal(8)                  # right shift of the significand, capped
        wide = Signal(50)               # sig << 26, then shifted right by sh
        shifted = Signal(50)
        q, q_up = Signal(24), Signal(25)
        guard, sticky = Signal(), Signal()
        a20_0 = Signal(26)
        self.comb += [
            sig.eq(Mux(ge != 0, Cat(gf, C(1, 1)), Cat(gf, C(0, 1)))),
            ge_eff.eq(Mux(ge != 0, ge, 1)),
            # positive: shift 131 - ge_eff (>= 1); negative: 130 - ge_eff (>= 0). |g| < 16.
            If(gs, sh.eq(Mux(ge_eff >= 80, 130 - ge_eff, 50))
            ).Else(sh.eq(Mux(ge_eff >= 81, 131 - ge_eff, 50))),
            wide.eq(Cat(C(0, 26), sig)),
            shifted.eq(wide >> sh),
            q.eq(shifted[26:50]),
            guard.eq(shifted[25]),
            sticky.eq(shifted[0:25] != 0),
            q_up.eq(q + (guard & (sticky | q[0]))),
            If(gs, a20_0.eq((1 << 24) - q_up)
            ).Else(a20_0.eq(Cat(C(0, 1), (1 << 23) + q_up))),
        ]
        a20 = Signal(26)
        self.sync += If(ce, a20.eq(a20_0))

        # S2: index, fraction, table read --------------------------------------------------
        idx = Signal(13)
        endpoint2 = Signal()
        self.comb += [
            idx.eq(a20[13:26]),
            endpoint2.eq(idx >= LUT_N),
            lut_r.adr.eq(Mux(endpoint2, LUT_N, idx)),
            dlut_r.adr.eq(idx[0:12]),
            lut_r.re.eq(ce), dlut_r.re.eq(ce),
        ]
        F2 = Signal(13)
        endpoint = Signal()
        self.sync += If(ce, F2.eq(a20[0:13]), endpoint.eq(endpoint2))
        # Stage 3 sees lut_r.dat_r / dlut_r.dat_r, F2 and endpoint (all stage-2 outputs).

        # S3: product P = F * d, unpack c -----------------------------------------------------
        c_w, d_w = lut_r.dat_r, dlut_r.dat_r
        P3 = Signal(37)
        pe3 = Signal((10, True))        # value = P * 2**pe (d's exponent minus 13)
        ps3 = Signal()
        cn3 = Signal(37)                # c's significand widened by 13 bits
        ce3 = Signal((10, True))
        cs3 = Signal()
        cw3 = Signal(32)
        ep3 = Signal()
        d_sig, c_sig = Signal(24), Signal(24)
        self.comb += [
            d_sig.eq(Mux(d_w[23:31] != 0, Cat(d_w[0:23], C(1, 1)), Cat(d_w[0:23], C(0, 1)))),
            c_sig.eq(Mux(c_w[23:31] != 0, Cat(c_w[0:23], C(1, 1)), Cat(c_w[0:23], C(0, 1)))),
        ]
        d_e, c_e = Signal((10, True)), Signal((10, True))
        self.comb += [
            d_e.eq(Mux(d_w[23:31] != 0, d_w[23:31], 1) - 150),
            c_e.eq(Mux(c_w[23:31] != 0, c_w[23:31], 1) - 150),
        ]
        self.sync += If(ce,
            P3.eq(F2 * d_sig), pe3.eq(d_e - 13), ps3.eq(d_w[31]),
            cn3.eq(Cat(C(0, 13), c_sig)), ce3.eq(c_e - 13), cs3.eq(c_w[31]),
            cw3.eq(c_w), ep3.eq(endpoint),
        )

        # S4: align (both significands are 37 bits, exponents pe3 / ce3) --------------------
        p_big = Signal()
        diff = Signal(10)
        big_m, small_m = Signal(37), Signal(37)
        big_s, small_s = Signal(), Signal()
        big_e = Signal((10, True))
        small_ext = Signal(104)
        small_shr = Signal(104)
        sticky4 = Signal()
        p_zero = Signal()
        self.comb += [
            p_zero.eq(P3 == 0),
            # The larger exponent is "big"; a zero product never is (its exponent is arbitrary).
            p_big.eq(~p_zero & ((pe3 > ce3) | (cn3 == 0))),
            If(p_big,
                big_m.eq(P3), big_e.eq(pe3), big_s.eq(ps3),
                small_m.eq(cn3), small_s.eq(cs3), diff.eq(pe3 - ce3),
            ).Else(
                big_m.eq(cn3), big_e.eq(ce3), big_s.eq(cs3),
                small_m.eq(P3), small_s.eq(ps3), diff.eq(Mux(p_zero, 0, ce3 - pe3)),
            ),
            # small << 3 (to meet big4 = big << 3), above a 64-bit field that catches the
            # bits the alignment shifts out: small_m << 67 in 104 bits.
            small_ext.eq(Cat(C(0, 67), small_m)),
            small_shr.eq(small_ext >> Mux(diff > 64, 64, diff)),
            sticky4.eq((small_shr[0:64] != 0) | ((diff > 64) & (small_m != 0))),
        ]
        big4, small4 = Signal(40), Signal(40)
        bs4, ss4 = Signal(), Signal()
        be4 = Signal((10, True))
        zero_both4 = Signal()
        zsign4 = Signal()
        self.sync += If(ce,
            big4.eq(Cat(C(0, 3), big_m)),
            small4.eq(small_shr[64:104] | sticky4),
            bs4.eq(big_s), ss4.eq(small_s), be4.eq(big_e),
            zero_both4.eq(p_zero & (cn3 == 0)),
            zsign4.eq(ps3 & cs3),        # sign of 0 + 0 under RNE: negative only if both are
        )
        ep4 = Signal()
        cw4 = Signal(32)
        self.sync += If(ce, ep4.eq(ep3), cw4.eq(cw3))

        # S5: add or subtract -------------------------------------------------------------
        sum5 = Signal(42)
        diff5 = Signal((42, True))
        res5 = Signal(41)
        rs5 = Signal()
        self.comb += [
            sum5.eq(big4 + small4),
            diff5.eq(big4 - small4),
        ]
        res_c, rs_c = Signal(41), Signal()
        self.comb += If(bs4 == ss4,
            res_c.eq(sum5), rs_c.eq(bs4),
        ).Elif(diff5 < 0,
            res_c.eq(-diff5), rs_c.eq(ss4),
        ).Else(
            res_c.eq(diff5), rs_c.eq(bs4),
        )
        be5 = Signal((10, True))
        zb5, zs5, ep5 = Signal(), Signal(), Signal()
        cw5 = Signal(32)
        self.sync += If(ce, res5.eq(res_c), rs5.eq(rs_c), be5.eq(be4),
                        zb5.eq(zero_both4), zs5.eq(zsign4), ep5.eq(ep4), cw5.eq(cw4))

        # S6: leading zeros ---------------------------------------------------------------
        lz, lz_stmts = _lzc(res5, 41)
        self.comb += lz_stmts
        res6 = Signal(41)
        lz6 = Signal(6)
        rs6, zb6, zs6, ep6 = Signal(), Signal(), Signal(), Signal()
        be6 = Signal((10, True))
        cw6 = Signal(32)
        self.sync += If(ce, res6.eq(res5), lz6.eq(lz), rs6.eq(rs5), be6.eq(be5),
                        zb6.eq(zb5), zs6.eq(zs5), ep6.eq(ep5), cw6.eq(cw5))

        # S7: normalize, round, select s ----------------------------------------------------
        # value = res6 * 2**(be6 - 3); leading one at bit L = 40 - lz6.
        norm = Signal(41 + 1)
        q7, q7r = Signal(24), Signal(25)
        g7, st7 = Signal(), Signal()
        fma_bits = Signal(32)
        L = Signal(7)
        e_biased = Signal((12, True))
        self.comb += [
            L.eq(40 - lz6),
            # Put the leading one at bit 40, so the 24-bit result is norm[17:41],
            # the guard bit norm[16] and the sticky norm[0:16].
            norm.eq(res6 << lz6),
            q7.eq(norm[17:41]),
            g7.eq(norm[16]),
            st7.eq(norm[0:16] != 0),
            q7r.eq(q7 + (g7 & (st7 | q7[0]))),
            # value = q * 2**(be6 - 3 + L - 23); biased = that exponent + 150 (+1 on carry).
            e_biased.eq(be6 - 3 + L - 23 + 150 + q7r[24]),
            If(res6 == 0,
                fma_bits.eq(Cat(C(0, 31), Mux(zb6, zs6, 0))),
            ).Else(
                fma_bits.eq(Cat(Mux(q7r[24], q7r[1:24], q7r[0:23]), e_biased[0:8], rs6)),
            ),
        ]
        s7 = Signal(32)
        s_sel = Signal(32)
        self.comb += Case(kind[6], {
            KIND_PASS: s_sel.eq(g[6]),
            KIND_NEGZERO: s_sel.eq(0x80000000),
            "default": s_sel.eq(Mux(ep6, cw6, fma_bits)),
        })
        self.sync += If(ce, s7.eq(s_sel))

        # S8: significand product ----------------------------------------------------------
        m7 = m[7]
        prod8 = Signal(48)
        zero8 = Signal()
        sign8 = Signal()
        ebsum8 = Signal((11, True))
        msig, ssig = Signal(24), Signal(24)
        self.comb += [
            msig.eq(Cat(m7[0:23], C(1, 1))),          # m subnormal is a fixup
            ssig.eq(Cat(s7[0:23], C(1, 1))),          # s is normal or zero here
        ]
        self.sync += If(ce,
            prod8.eq(msig * ssig),
            zero8.eq((m7[0:31] == 0) | (s7[0:31] == 0)),
            sign8.eq(m7[31] ^ s7[31]),
            # prod in [2**46, 2**48): the biased result exponent is mb + sb - 127 (+1 when
            # the product reaches 2**47, +1 again on a rounding carry). 1.0*1.0 -> 127.
            ebsum8.eq(m7[23:31] + s7[23:31] - 127),
        )

        # S9: normalize, round, encode, flag ------------------------------------------------
        top = Signal()
        qm, qmr = Signal(24), Signal(25)
        gm, stm = Signal(), Signal()
        eb9 = Signal((11, True))
        r9 = Signal(32)
        underflow = Signal()
        self.comb += [
            top.eq(prod8[47]),
            If(top,
                qm.eq(prod8[24:48]), gm.eq(prod8[23]), stm.eq(prod8[0:23] != 0),
            ).Else(
                qm.eq(prod8[23:47]), gm.eq(prod8[22]), stm.eq(prod8[0:22] != 0),
            ),
            qmr.eq(qm + (gm & (stm | qm[0]))),
            eb9.eq(ebsum8 + top + qmr[24]),
            underflow.eq(~zero8 & ((eb9 <= 0) | (eb9 >= 255))),
            If(zero8,
                r9.eq(Cat(C(0, 31), sign8)),
            ).Else(
                r9.eq(Cat(Mux(qmr[24], qmr[1:24], qmr[0:23]), eb9[0:8], sign8)),
            ),
        ]
        out_r = Signal(32)
        out_fix = Signal()
        self.sync += If(ce, out_r.eq(r9), out_fix.eq((kind[8] == KIND_FIXUP) | underflow))
        self.comb += [source.r.eq(out_r), source.fixup.eq(out_fix)]

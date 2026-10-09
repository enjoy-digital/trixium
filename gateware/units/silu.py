#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Resident SiLU-multiply (sequencer unit 11): r = m * siluf(g) on binary32, bit-exact.

SiluMulKernel is portable Migen (no vendor floating-point IP), one element per cycle in a
globally stalled pipeline. The contract and the arithmetic are in model/silu_hw.py;
the kernel is `silu_mul_fixed()` stage for stage:

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

SiluScratchpad runs `ways` kernels over the shared two-port scratchpad. Its command
(SILU_COMMAND, the firmware's descriptor layout) has two operations:

RUN (op 0): for i < elements, dst[i] = m[i] * siluf(g[i]) with g at src_g and m at src_m
(binary32 vectors, 8 per 256-bit beat), bit-exact with hybrid.c's silu_mul_fv. An element
outside the kernel's fast path is a FIXUP: its dst lane receives g unchanged and bit i of
the bitmap at `fixmap` is set; the firmware recomputes those elements from the untouched
g/m. `fixups` counts them. The bitmap is ceil(elements/32) little-endian words (bit i%32 of
word i//32); every word of it is written, so no stale bit survives. dst == src_g (in place,
as the firmware does) is allowed; any other overlap between dst, fixmap and the sources is
rejected. unit_m replaces m by 1.0 (no m read); perm_a/perm_b/seg_beats permute output
segments (see the output beat offset below); table_only/table_set select the table pair.

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

from gateware.units.fp32 import _lzc

# Constants ----------------------------------------------------------------------------------------

LUT_N = 4096
IN_LAYOUT = [("g", 32), ("m", 32)]
OUT_LAYOUT = [("r", 32), ("fixup", 1), ("g", 32)]   # g: the input gate, for fixup lanes

KIND_TABLE, KIND_PASS, KIND_NEGZERO, KIND_FIXUP = range(4)

# SiLU Mul Kernel ----------------------------------------------------------------------------------

class SiluMulKernel(LiteXModule):
    """r = m * siluf(g), one element per clock, LATENCY stages (see the module docstring).

    tables=(lut, dlut) preloads the memories (simulation/tests); hardware loads them through the
    write ports. sets > 1: that many resident table pairs (the SiLU's and the attention gate's
    sigmoid), written through lut_set, read through table_sel.
    """
    LATENCY = 9

    def __init__(self, tables=None, sets=1):
        # Streams.
        self.sink   = sink   = stream.Endpoint(IN_LAYOUT)
        self.source = source = stream.Endpoint(OUT_LAYOUT)

        # Table write ports and selection (table_only/table_sel held for a whole run;
        # table_only turns the pass-through and -0 kinds into fixups).
        self.lut_we, self.lut_adr, self.lut_dat = Signal(), Signal(13), Signal(32)
        self.table_only = Signal()
        sb = max(1, (sets - 1).bit_length())
        self.table_sel = Signal(sb) # Table pair read.
        self.lut_set   = Signal(sb) # Table pair written by lut_we/dlut_we.
        self.dlut_we, self.dlut_adr, self.dlut_dat = Signal(), Signal(12), Signal(32)

        # # #

        # Tables.
        if sets > 1:
            # Set s at [s*8192, s*8192 + 4097) and [s*4096, (s+1)*4096): a set bit on top.
            lut = Memory(32, sets*2*LUT_N, init=list(tables[0]) if tables else None)
            dlut = Memory(32, sets*LUT_N, init=list(tables[1]) if tables else None)
            lut_wadr, dlut_wadr = Cat(self.lut_adr, self.lut_set), Cat(self.dlut_adr, self.lut_set)
        else:
            lut = Memory(32, LUT_N + 1, init=list(tables[0]) if tables else None)
            dlut = Memory(32, LUT_N, init=list(tables[1]) if tables else None)
            lut_wadr, dlut_wadr = self.lut_adr, self.dlut_adr
        lut_w, dlut_w = lut.get_port(write_capable=True), dlut.get_port(write_capable=True)
        lut_r = lut.get_port(has_re=True, mode=READ_FIRST)
        dlut_r = dlut.get_port(has_re=True, mode=READ_FIRST)
        self.specials += lut, dlut, lut_w, dlut_w, lut_r, dlut_r
        # Handles for tests to preload the tables. Never CSR-exported: AutoCSR turns a Memory
        # attribute into a CSR-bus SRAM with a THIRD (sys, read/write) port, which Verilator
        # simulates exactly but Vivado cannot map onto a BRAM (the tables then read wrong on
        # the board while every simulation passes).
        self.autocsr_exclude = {"lut_mem", "dlut_mem"}
        self.lut_mem, self.dlut_mem = lut, dlut
        self.comb += [
            lut_w.we.eq(self.lut_we), lut_w.adr.eq(lut_wadr), lut_w.dat_w.eq(self.lut_dat),
            dlut_w.we.eq(self.dlut_we), dlut_w.adr.eq(dlut_wadr), dlut_w.dat_w.eq(self.dlut_dat),
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

        # S1: classify and add 16 ------------------------------------------------------------------
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
            ).Elif(k_pass & ~self.table_only, kind0.eq(KIND_PASS)
            ).Elif(k_negzero & ~self.table_only, kind0.eq(KIND_NEGZERO)
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

        # S2: index, fraction, table read ----------------------------------------------------------
        idx = Signal(13)
        endpoint2 = Signal()
        self.comb += [
            idx.eq(a20[13:26]),
            endpoint2.eq(idx >= LUT_N),
            lut_r.adr.eq(Cat(Mux(endpoint2, LUT_N, idx)[:13], self.table_sel) if sets > 1
                         else Mux(endpoint2, LUT_N, idx)),
            dlut_r.adr.eq(Cat(idx[0:12], self.table_sel) if sets > 1 else idx[0:12]),
            lut_r.re.eq(ce), dlut_r.re.eq(ce),
        ]
        F2 = Signal(13)
        endpoint = Signal()
        self.sync += If(ce, F2.eq(a20[0:13]), endpoint.eq(endpoint2))
        # Stage 3 sees lut_r.dat_r / dlut_r.dat_r, F2 and endpoint (all stage-2 outputs).

        # S3: product P = F * d, unpack c ----------------------------------------------------------
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

        # S4: align (both significands are 37 bits, exponents pe3 / ce3) ---------------------------
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

        # S5: add or subtract ----------------------------------------------------------------------
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

        # S6: leading zeros ------------------------------------------------------------------------
        lz, lz_stmts = _lzc(res5, 41)
        self.comb += lz_stmts
        res6 = Signal(41)
        lz6 = Signal(6)
        rs6, zb6, zs6, ep6 = Signal(), Signal(), Signal(), Signal()
        be6 = Signal((10, True))
        cw6 = Signal(32)
        self.sync += If(ce, res6.eq(res5), lz6.eq(lz), rs6.eq(rs5), be6.eq(be5),
                        zb6.eq(zb5), zs6.eq(zs5), ep6.eq(ep5), cw6.eq(cw5))

        # S7: normalize, round, select s -----------------------------------------------------------
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

        # S8: significand product ------------------------------------------------------------------
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

        # S9: normalize, round, encode, flag -------------------------------------------------------
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
        # The input gate travels with its result: a client writes it back unchanged for a
        # fixup lane, so the CPU can recompute that element from the original inputs.
        self.comb += [source.r.eq(out_r), source.fixup.eq(out_fix), source.g.eq(g[n])]

# SiLU Scratchpad ----------------------------------------------------------------------------------

SILU_COMMAND = [("op", 1), ("src_g", 24), ("src_m", 24), ("dst", 24), ("fixmap", 24),
                ("elements", 16), ("unit_m", 1), ("seg_beats", 16), ("perm_a", 8), ("perm_b", 8),
                ("protect_start", 24), ("protect_end", 24), ("table_only", 1), ("table_set", 1)]
OP_RUN, OP_LOAD = 0, 1
LOAD_BEATS = 513 + 512

class SiluScratchpad(LiteXModule):
    """Resident SiLU-multiply over the shared scratchpad, `ways` kernels in lockstep (see the
    module docstring)."""
    def __init__(self, scratchpad_size=256*1024, tables=None, sets=2, ways=4):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client "
                             "address space")
        # Command / status.
        self.cmd = stream.Endpoint(SILU_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error  = Signal(8)
        self.fixups = Signal(16)
        self.read_beats, self.output_beats = Signal(32), Signal(32)

        # Scratchpad client ports.
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)

        # ways kernels in lockstep (same valid, same ready, their own table copies loaded
        # together): `ways` elements per clock. 4 = the read port's pace (a g and an m beat
        # per 8 elements); 1 element per clock cost ~1.9M cycles per 27B forward.
        assert ways in (1, 2, 4)
        W = ways
        self.kernels = kernels = [ResetInserter()(SiluMulKernel(tables, sets=sets))
                                  for _ in range(W)]
        self.kernel = kernel = kernels[0]             # (LiteXModule registers it)
        for k, kern in enumerate(kernels[1:], 1):
            setattr(self.submodules, f"kernel{k}", kern)
        self.sets, self.ways = sets, ways

        # # #

        # Command capture and validation.
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
            bounds_bad.eq((ends["src_g"] > scratchpad_size) |
                (cap["protect_end"] > scratchpad_size) |
                (~load & ((~unit_m & (ends["src_m"] > scratchpad_size)) |
                          (ends["dst"] > scratchpad_size) |
                          (ends["fixmap"] > scratchpad_size)))),
            overlap_bad.eq(~load & (
                (overlap("dst", "src_g") & ((cap["dst"] != cap["src_g"]) | perm)) |
                (~unit_m & overlap("dst", "src_m")) | overlap("fixmap", "src_g") |
                (~unit_m & overlap("fixmap", "src_m")) | overlap("fixmap", "dst"))),
            protected_bad.eq(~load & (protected("dst") | protected("fixmap"))),
        ]

        # Read side: pairs of beats (g, m) for RUN, single beats for LOAD --------------------------
        # One read per cycle at most; the response is captured a cycle later. `rd_second`
        # says the next request is the m beat of the current pair.
        pairs = ResetInserter()(SyncFIFO(width=512, depth=4))
        self.pairs = pairs
        issued = Signal(14)                        # beats (pairs) requested
        rd_second = Signal()
        rd_pending, rd_was_m = Signal(), Signal()
        g_hold = Signal(256)
        in_flight = Signal(3)                      # pairs requested but not yet in the FIFO
        can_issue = Signal()
        run_active = Signal()
        clear = Signal()
        self.comb += [
            run_active.eq(state == RUN),
            clear.eq(self.cancel | ((state != RUN) & (state != LOAD))),
            pairs.reset.eq(clear),
            *[[kern.reset.eq(clear),
               kern.table_only.eq(cap["table_only"]),
               kern.table_sel.eq(cap["table_set"] if sets > 1 else 0),
               kern.lut_set.eq(cap["table_set"] if sets > 1 else 0)] for kern in kernels],
            can_issue.eq((issued < beats) & ((pairs.level + in_flight) < 4)),
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
            *[[kern.lut_we.eq(unpacking & ((cur < 512) | ((cur == 512) & (ld_lane == 0)))),
               kern.lut_adr.eq(Cat(ld_lane[:3], cur[:10])),
               kern.lut_dat.eq(word),
               kern.dlut_we.eq(unpacking & (cur >= 513)),
               kern.dlut_adr.eq(Cat(ld_lane[:3], dcur[:9])),
               kern.dlut_dat.eq(word)] for kern in kernels],
        ]

        # Unpack pairs into the kernels, `ways` lanes per cycle (a multiple of ways: lanes past
        # the vector's end are fed and their results dropped) --------------------------------------
        lane = Signal(3)
        fed = Signal(17)                           # elements fed
        head_g, head_m = Signal(256), Signal(256)
        last_lane = Signal()
        feed = Signal()
        self.comb += [
            head_g.eq(pairs.dout[0:256]), head_m.eq(pairs.dout[256:512]),
            last_lane.eq((lane == 8 - W) | (fed + W >= cap["elements"])),
            feed.eq(kernel.sink.valid & kernel.sink.ready),
            pairs.re.eq(feed & last_lane),
        ]
        for k, kern in enumerate(kernels):
            self.comb += [
                kern.sink.valid.eq(run_active & pairs.readable & ~self.cancel),
                kern.sink.g.eq(Array(head_g[32*j:32*j + 32] for j in range(8))[lane + k]),
                kern.sink.m.eq(Array(head_m[32*j:32*j + 32] for j in range(8))[lane + k]),
            ]
        self.sync += If(clear, lane.eq(0), fed.eq(0)).Elif(feed,
            fed.eq(fed + W), lane.eq(Mux(last_lane, 0, lane + W)))

        # Collect results into output beats and bitmap words ---------------------------------------
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
        live = [Signal(name=f"silu_live{k}") for k in range(W)]      # element got + k exists
        ress = [Signal(32, name=f"silu_res{k}") for k in range(W)]
        fixes = [Signal(name=f"silu_fix{k}") for k in range(W)]
        out_ready = Signal()
        self.comb += [
            acc_next.eq(acc), acc_we_next.eq(acc_we), bits_next.eq(bits),
            ends_beat.eq((out_lane == 8 - W) | (got + W >= cap["elements"])),
            ends_word.eq((got[0:5] == 32 - W) | (got + W >= cap["elements"])),
            # The accumulators keep filling while the previous beat/word waits for the
            # write port; a result that would close a second one waits instead.
            out_ready.eq(run_active & ~self.cancel &
                         ~(beat_full & ends_beat) & ~(word_full & ends_word)),
            take.eq(kernel.source.valid & out_ready),
            close_beat.eq(take & ends_beat),
            close_word.eq(take & ends_word),
        ]
        for k, kern in enumerate(kernels):
            self.comb += [
                kern.source.ready.eq(out_ready),
                live[k].eq(got + k < cap["elements"]),
                fixes[k].eq(kern.source.fixup & live[k]),
                ress[k].eq(Mux(kern.source.fixup, kern.source.g, kern.source.r)),
                [If(live[k] & (out_lane + k == j), acc_next[32*j:32*j + 32].eq(ress[k]),
                    acc_we_next[j].eq(1)) for j in range(8)],
                [If(live[k] & (got[0:5] + k == j), bits_next[j].eq(kern.source.fixup))
                 for j in range(32)],
            ]
        self.comb += res.eq(ress[0])
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
            got.eq(got + W),
            out_lane.eq(Mux(close_beat, 0, out_lane + W)),
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
            self.sp_wr_we.eq(Mux(write_word, Cat(*[word_idx[0:3] == j for j in range(8)]),
                                 beat_we)),
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
        self.sync += If(~clear & take, self.fixups.eq(self.fixups + sum(fixes)))

        # Control ----------------------------------------------------------------------------------
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
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3),
                                      protected_bad)),
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
                # got counts whole groups of `ways` elements.
                RUN: If(all_written & (got >= cap["elements"]), state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
        )

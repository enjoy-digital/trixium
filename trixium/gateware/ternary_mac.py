#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiChannelPipe: one HBM channel's compute pipeline (unpack + dot + scale-accumulate).

Consumes the weight-beat stream of one channel (row packets: scale beats then code beats, see
trixium.layout), computes one dot product per row against the local x-buffer and emits
int32 results, implementing trixium.model.bitexact.matvec_ternary()/matvec_q8() bit-exactly:

  ternary: 128 lanes of {-x, 0, +x} (2-bit codes) -> pipelined adder tree -> group_sum
  q8     : 32 lanes of w8*x8 -> adder tree -> group_sum
  both   : group_sum * m11(fp16 mantissa) shifted by (e16 + B) into a 48-bit accumulator,
           per-row result = acc >> R, truncated to int32.

The per-row/beat structure (scale_beats, code_beats, rows) and B/R come from quasi-static
op parameter signals set by the engine before `start`. Fully pipelined: accepts one beat per
cycle; results (1 per row) go through a small FIFO to the gather stage.
"""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

# Helpers ------------------------------------------------------------------------------------------

def _delay(module, sig, cycles):
    """Return `sig` delayed by `cycles` sync stages."""
    cur = sig
    for _ in range(cycles):
        nxt = Signal.like(cur)
        module.sync += nxt.eq(cur)
        cur = nxt
    return cur

def _adder_tree(module, lanes, reg_every=2):
    """Pairwise adder tree; registers every `reg_every` levels. Returns (out, n_reg_stages)."""
    level  = list(lanes)
    n_lvl  = 0
    n_regs = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                s = Signal((len(level[i]) + 1, True))
                module.comb += s.eq(level[i] + level[i+1])
            else:
                s = level[i]
            nxt.append(s)
        n_lvl += 1
        if n_lvl % reg_every == 0:
            regs = []
            for s in nxt:
                r = Signal.like(s)
                module.sync += r.eq(s)
                regs.append(r)
            nxt = regs
            n_regs += 1
        level = nxt
    return level[0], n_regs

# BonsaiChannelPipe --------------------------------------------------------------------------------

KIND_OP_TERNARY = 0
KIND_OP_Q8      = 1
KIND_OP_AXPY    = 2 # Weighted-V: acc[lane] += (p8 * v8) << (10 + e16 + B - Rpre).

class BonsaiChannelPipe(LiteXModule):
    def __init__(self, xbuf_groups=96, out_fifo_depth=24, chan_id=0, n_channels=1):
        # Op parameters (quasi-static while busy).
        self.kind        = Signal(2)
        self.rows        = Signal(16)
        self.scale_beats = Signal(4)   # Scale beats per row (>= 1).
        self.code_beats  = Signal(8)   # Code beats (= groups) per row.
        self.bias        = Signal(7)   # B.
        self.rshift      = Signal(6)   # R.
        self.qn          = Signal()    # Dual-Q GQA mode (Q8/AXPY): 2 heads per stream.
        self.rshift2     = Signal(6)   # Head-B AXPY pre-shift.
        self.start       = Signal()    # Pulse: reset counters/accumulator/error.

        # Weight beats in, results out.
        self.sink   = stream.Endpoint([("data", 256)])
        # Matvec results present one int32 in data[:32] (wide=0); AXPY drain entries carry
        # 8 accumulators = one full scratchpad beat (wide=1, row = first lane index).
        # head=1 marks dual-Q head-B results (gather adds the qstride offset).
        self.source = stream.Endpoint([("row", 16), ("data", 256), ("wide", 1), ("head", 1)])

        # x-buffer broadcast write port (from the engine's fill FSM).
        self.xw_group = Signal(max=max(xbuf_groups, 2))
        self.xw_sub   = Signal(2)
        self.xw_data  = Signal(256)
        self.xw_we    = Signal()

        # Status.
        self.error    = Signal() # Sticky: reserved ternary code 3 seen.
        self.overflow = Signal() # Sticky: result FIFO overflow (rate mismatch, debug).

        # # #

        out_fifo_rdy_ax = Signal() # Driven at the FIFO hookup below (drain flow control).
        out_fifo_room   = Signal() # Driven at the FIFO hookup below (beat-intake flow control).

        # x-buffer: 1024-bit words, one per ternary group (4 q8 groups). --------------------------
        xbuf = Memory(1024, xbuf_groups)
        self.specials += xbuf
        xbuf_w  = xbuf.get_port(write_capable=True, we_granularity=256)
        xbuf_r  = xbuf.get_port()
        # Second read port for dual-Q ternary (head B's word): the x-buffer is distributed
        # RAM (1024b wide), where extra read ports are native - the no-new-port rule guards
        # the BRAM scratchpad, not this.
        xbuf_r2 = xbuf.get_port()
        self.specials += xbuf_w, xbuf_r, xbuf_r2
        self.comb += [
            xbuf_w.adr.eq(self.xw_group),
            xbuf_w.dat_w.eq(Replicate(self.xw_data, 4)),
            xbuf_w.we.eq(Mux(self.xw_we, 1 << self.xw_sub, 0)),
        ]

        # Scale RAM: 16 fp16 per beat, double-buffered by row parity. -----------------------------
        scale_ram = Memory(256, 16) # 8 words per bank, 2 banks.
        self.specials += scale_ram
        scale_w = scale_ram.get_port(write_capable=True)
        scale_r = scale_ram.get_port()
        self.specials += scale_w, scale_r

        # Beat parser -----------------------------------------------------------------------------
        phase   = Signal() # 0: scale beats, 1: code beats.
        s_cnt   = Signal(4)
        g_cnt   = Signal(8)
        row_cnt = Signal(16)

        accept = Signal()
        self.comb += [
            # Beat intake throttles on result-FIFO headroom: at N channels the aggregate
            # result rate (N/row_beats per cycle) can exceed the 1/cycle gather drain
            # (first bites at N=4 ternary, 3 beats/row) - without this the FIFO drops
            # results and the op never completes.
            self.sink.ready.eq(out_fifo_room),
            accept.eq(self.sink.valid & self.sink.ready),
        ]

        is_code   = Signal()
        row_last  = Signal()
        self.comb += [
            is_code.eq(phase == 1),
            row_last.eq(is_code & (g_cnt == self.code_beats - 1)),
        ]

        self.sync += [
            If(self.start,
                phase.eq(0),
                s_cnt.eq(0),
                g_cnt.eq(0),
                row_cnt.eq(0),
            ).Elif(accept,
                If(~is_code,
                    s_cnt.eq(s_cnt + 1),
                    If(s_cnt == self.scale_beats - 1,
                        phase.eq(1),
                        g_cnt.eq(0),
                    )
                ).Else(
                    g_cnt.eq(g_cnt + 1),
                    If(row_last,
                        phase.eq(0),
                        s_cnt.eq(0),
                        row_cnt.eq(row_cnt + 1),
                    )
                )
            )
        ]

        # Scale writes (bank = row parity).
        self.comb += [
            scale_w.adr.eq(Cat(s_cnt[:3], row_cnt[0])),
            scale_w.dat_w.eq(self.sink.data),
            scale_w.we.eq(accept & ~is_code),
        ]

        # T0: code-beat acceptance: launch x-buffer + scale reads, register codes. ----------------
        t0_fire = Signal()
        pidx    = Signal(16) # AXPY: probs byte index = row*n_channels + chan_id.
        # Dual-Q x layout: the fill interleaves the two heads' vectors at 32-byte
        # granularity (position 2g = head A group g, 2g+1 = head B group g), so one xbuf
        # read serves both heads. AXPY bytes get the same treatment: byte t of head A
        # lives at (t>>5)*64 + (t&31), head B at +32 - always inside the same 1024b word.
        pidx_d = Signal(16) # AXPY dual: interleaved byte index of head A.
        self.comb += [
            t0_fire.eq(accept & is_code),
            pidx.eq(row_cnt*n_channels + chan_id),
            pidx_d.eq(Cat(pidx[0:5], C(0, 1), pidx[5:15])),
            xbuf_r.adr.eq(Mux(self.kind == KIND_OP_AXPY,
                              Mux(self.qn, pidx_d[7:], pidx[7:]),
                          Mux(self.kind == KIND_OP_Q8,
                              Mux(self.qn, g_cnt[1:], g_cnt[2:]),
                              Mux(self.qn, Cat(C(0, 1), g_cnt[:7]), g_cnt)))),
            # Dual ternary: head A at even words (2g), head B at odd (2g+1).
            xbuf_r2.adr.eq(Cat(C(1, 1), g_cnt[:7])),
            scale_r.adr.eq(Cat(g_cnt[4:7], row_cnt[0])), # Scale g at halfword g%16 of word g//16.
        ]
        codes_r   = Signal(256)
        g_lo_r    = Signal(4) # Scale index within the 16-scale word.
        q8_sub_r  = Signal(2) # 256-bit slice of the 1024-bit x word (q8 mode).
        pidx_r    = Signal(7) # AXPY: byte within the 1024-bit x word.
        g_ax_r    = Signal(3) # AXPY: accumulator group of this beat.
        self.sync += [
            If(t0_fire,
                codes_r.eq(self.sink.data),
                g_lo_r.eq(g_cnt[:4]),
                q8_sub_r.eq(Mux(self.qn, Cat(C(0, 1), g_cnt[0]), g_cnt[:2])),
                pidx_r.eq(Mux(self.qn, pidx_d[:7], pidx[:7])),
                g_ax_r.eq(g_cnt[:3]),
            )
        ]
        v1        = _delay(self, t0_fire, 1)             # T1 valid.
        row1      = _delay(self, row_cnt, 1)
        row_last1 = _delay(self, row_last, 1)

        # T1: lane select + scale fp16 mux; register lanes. ---------------------------------------
        # Ternary lanes: code 00 -> -x, 01 -> 0, 10 -> +x, 11 -> 0 (+ sticky error).
        xword  = xbuf_r.dat_r  # 1024-bit, valid at T1.
        xword2 = xbuf_r2.dat_r # Head B's word (dual ternary), valid at T1.

        tern_lanes  = []
        tern_lanesb = []
        code3_seen = Signal()
        code3_any  = []
        for j in range(128):
            x_j  = Signal((8, True))
            c_j  = Signal(2)
            lane = Signal((9, True))
            self.comb += [
                x_j.eq(xword[8*j:8*j+8]),
                c_j.eq(codes_r[2*j:2*j+2]),
                Case(c_j, {
                    0 : lane.eq(-x_j),
                    1 : lane.eq(0),
                    2 : lane.eq(x_j),
                    3 : lane.eq(0),
                }),
            ]
            code3_any.append(c_j == 3)
            r = Signal((9, True))
            self.sync += If(v1, r.eq(lane))
            tern_lanes.append(r)
            xb_t  = Signal((8, True))
            laneb = Signal((9, True))
            self.comb += [
                xb_t.eq(xword2[8*j:8*j+8]),
                Case(c_j, {
                    0 : laneb.eq(-xb_t),
                    1 : laneb.eq(0),
                    2 : laneb.eq(xb_t),
                    3 : laneb.eq(0),
                }),
            ]
            rbt = Signal((9, True))
            self.sync += If(v1 & self.qn, rbt.eq(laneb))
            tern_lanesb.append(rbt)
        self.comb += code3_seen.eq(v1 & (self.kind == KIND_OP_TERNARY) & reduce(or_, code3_any))
        self.sync += If(self.start, self.error.eq(0)).Elif(code3_seen, self.error.eq(1))

        # Q8 lanes: w8 * x8 over the selected 256-bit x slice. Head B (dual-Q) uses the
        # adjacent slice / adjacent p8 byte (+32) of the same 1024-bit x word.
        xslice  = Signal(256)
        xsliceb = Signal(256)
        q8_subb = Signal(2)
        self.comb += q8_subb.eq(q8_sub_r | 1)
        self.comb += Case(q8_sub_r, {i: xslice.eq(xword[256*i:256*(i+1)]) for i in range(4)})
        self.comb += Case(q8_subb,  {i: xsliceb.eq(xword[256*i:256*(i+1)]) for i in range(4)})
        # AXPY coefficient byte: two-stage constant-slice mux out of the 1024-bit x word.
        p8_grp  = Signal(64)
        p8_ax   = Signal((8, True))
        p8_grpb = Signal(64)
        p8_axb  = Signal((8, True))
        pidx_rb = Signal(7)
        self.comb += pidx_rb.eq(pidx_r | 32) # Dual layout: head B byte at +32 (bit 5 clear on A).
        self.comb += Case(pidx_r[3:], {i: p8_grp.eq(xword[64*i:64*(i+1)]) for i in range(16)})
        self.comb += Case(pidx_r[:3], {i: p8_ax.eq(p8_grp[8*i:8*(i+1)]) for i in range(8)})
        self.comb += Case(pidx_rb[3:], {i: p8_grpb.eq(xword[64*i:64*(i+1)]) for i in range(16)})
        self.comb += Case(pidx_rb[:3], {i: p8_axb.eq(p8_grpb[8*i:8*(i+1)]) for i in range(8)})
        is_axpy = Signal()
        self.comb += is_axpy.eq(self.kind == KIND_OP_AXPY)
        q8_lanes  = []
        q8_lanesb = []
        for j in range(32):
            w_j = Signal((8, True))
            x_j = Signal((8, True))
            p   = Signal((16, True))
            self.comb += [
                w_j.eq(codes_r[8*j:8*j+8]),
                x_j.eq(Mux(is_axpy, p8_ax, xslice[8*j:8*j+8])),
                p.eq(w_j*x_j),
            ]
            r = Signal((16, True))
            self.sync += If(v1, r.eq(p))
            q8_lanes.append(r)
            xb_j = Signal((8, True))
            pb   = Signal((16, True))
            self.comb += [
                xb_j.eq(Mux(is_axpy, p8_axb, xsliceb[8*j:8*j+8])),
                pb.eq(w_j*xb_j),
            ]
            rb = Signal((16, True))
            self.sync += If(v1 & self.qn, rb.eq(pb))
            q8_lanesb.append(rb)

        # Scale fp16: mux halfword out of the scale word (registered at T1 for use at T2+).
        scale_word_r = Signal(256)
        g_lo_r2      = Signal(4)
        self.sync += If(v1, scale_word_r.eq(scale_r.dat_r), g_lo_r2.eq(g_lo_r))
        scale_f16 = Signal(16)
        self.comb += Case(g_lo_r2, {i: scale_f16.eq(scale_word_r[16*i:16*(i+1)]) for i in range(16)})

        # AXPY datapath (kind 2): per-lane shift + int32 accumulators + drain. --------------------
        # Row weight is one shift (pow-2 row scales, m11 == 1024): sh = (f_exp - 15) + B - Rpre
        # with B on self.bias and Rpre on self.rshift. Lane products (v8 * p8, registered at
        # T2) shift at T2->T3, accumulate per (group, lane) at T3; after the final row a
        # drain FSM streams code_beats*32 accumulators into the out FIFO (row = g*32 + j).
        # Signed-only arithmetic through dedicated intermediates (unsigned operands mixed
        # into signed expressions emit corrupt Verilog - see the fp16 decompose below).
        ax_c31n  = Signal((9, True), reset=-31)
        ax_sh    = Signal((9, True))
        ax_exp   = Signal(5)
        ax_exp_s = Signal((7, True))
        ax_bias  = Signal((8, True))
        ax_rpre  = Signal((8, True))
        self.comb += [
            ax_exp.eq(scale_f16[10:15]),
            ax_exp_s.eq(ax_exp),          # Zero-extend into a signed signal.
            ax_bias.eq(self.bias),
            ax_rpre.eq(self.rshift),
            ax_sh.eq(ax_exp_s - 15 + ax_bias - ax_rpre),
        ]
        ax_shl = Signal(4)
        ax_shr = Signal(5)
        self.comb += [
            ax_shl.eq(Mux(ax_sh >= 0, Mux(ax_sh > 15, 15, ax_sh[:4]), 0)),
            ax_shr.eq(Mux(ax_sh < 0, Mux(ax_sh < ax_c31n, 31, (-ax_sh)[:5]), 0)),
        ]
        # Head B (dual-Q): its own pre-shift (Rpre differs per head), same exponent path.
        ax_sh2    = Signal((9, True))
        ax_rpre2  = Signal((8, True))
        ax_shl2   = Signal(4)
        ax_shr2   = Signal(5)
        self.comb += [
            ax_rpre2.eq(self.rshift2),
            ax_sh2.eq(ax_exp_s - 15 + ax_bias - ax_rpre2),
            ax_shl2.eq(Mux(ax_sh2 >= 0, Mux(ax_sh2 > 15, 15, ax_sh2[:4]), 0)),
            ax_shr2.eq(Mux(ax_sh2 < 0, Mux(ax_sh2 < ax_c31n, 31, (-ax_sh2)[:5]), 0)),
        ]
        v2a    = _delay(self, v1, 1)              # Lane products valid (T2).
        v3a    = _delay(self, v1, 2)              # Shifted values valid (T3).
        g_ax2  = _delay(self, g_ax_r, 2)          # Beat group at T3.
        ax_shifted  = []
        ax_shifted2 = []
        for j in range(32):
            sh_full = Signal((32, True))
            r       = Signal((32, True))
            self.comb += sh_full.eq(Mux(ax_sh >= 0, q8_lanes[j] << ax_shl,
                                                    q8_lanes[j] >> ax_shr))
            self.sync += If(v2a, r.eq(sh_full))
            ax_shifted.append(r)
            sh_full2 = Signal((32, True))
            r2       = Signal((32, True))
            self.comb += sh_full2.eq(Mux(ax_sh2 >= 0, q8_lanesb[j] << ax_shl2,
                                                      q8_lanesb[j] >> ax_shr2))
            self.sync += If(v2a & self.qn, r2.eq(sh_full2))
            ax_shifted2.append(r2)

        AX_GROUPS = 4 # head_dim <= 128 for AXPY ops (kvpack contract).
        acc_ax = [[Signal((32, True), name=f"axacc{g}_{j}") for j in range(32)]
                  for g in range(AX_GROUPS)]
        acc_flat = Array([acc_ax[g][j] for g in range(AX_GROUPS) for j in range(32)])
        acc_ax2 = [[Signal((32, True), name=f"axacc2_{g}_{j}") for j in range(32)]
                   for g in range(AX_GROUPS)]
        acc2_flat = Array([acc_ax2[g][j] for g in range(AX_GROUPS) for j in range(32)])

        # Final-beat tag: last code beat of the last row, aligned to the accumulate stage.
        fin_t0  = Signal()
        self.comb += fin_t0.eq(t0_fire & row_last & (row_cnt == self.rows - 1))
        fin_acc = _delay(self, fin_t0, 3)         # Accumulate of the final beat done (T4).

        d_cnt   = Signal(8)
        d_total = Signal(8)
        # Wide drain: 8 accumulators (one scratchpad beat) per FIFO entry, so the gather
        # stage retires 8 partials/cycle instead of 1. d_cnt counts 8-lane groups.
        self.comb += d_total.eq(Cat(C(0, 2), self.code_beats[:3])) # code_beats * 4 = cols/8.
        self.ax_fsm = ax_fsm = FSM(reset_state="AX_IDLE")
        drain_val  = Signal()
        drain_head = Signal() # Dual-Q: drain bank A fully, then bank B.
        ax_fsm.act("AX_IDLE",
            If(self.start & (self.kind == KIND_OP_AXPY),
                NextValue(d_cnt, 0),
                NextValue(drain_head, 0),
                If(self.rows == 0,
                    NextState("AX_DRAIN"), # No rows on this channel: emit zero partials.
                ).Else(
                    NextState("AX_ACC"),
                )
            )
        )
        ax_fsm.act("AX_ACC",
            If(fin_acc,
                NextState("AX_DRAIN"),
            )
        )
        ax_fsm.act("AX_DRAIN",
            drain_val.eq(1),
            If(out_fifo_rdy_ax,
                NextValue(d_cnt, d_cnt + 1),
                If(d_cnt == d_total - 1,
                    If(self.qn & ~drain_head,
                        NextValue(drain_head, 1),
                        NextValue(d_cnt, 0),
                    ).Else(
                        NextValue(drain_head, 0),
                        NextState("AX_IDLE"),
                    )
                )
            )
        )

        self.sync += [
            If(self.start,
                *[acc_ax[g][j].eq(0) for g in range(AX_GROUPS) for j in range(32)],
                *[acc_ax2[g][j].eq(0) for g in range(AX_GROUPS) for j in range(32)],
            ).Elif(v3a & (self.kind == KIND_OP_AXPY),
                Case(g_ax2[:2], {g: [acc_ax[g][j].eq(acc_ax[g][j] + ax_shifted[j])
                                     for j in range(32)] for g in range(AX_GROUPS)}),
                If(self.qn,
                    Case(g_ax2[:2], {g: [acc_ax2[g][j].eq(acc_ax2[g][j] + ax_shifted2[j])
                                         for j in range(32)] for g in range(AX_GROUPS)}),
                )
            ),
        ]

        # T2..: adder trees (both built; result muxed by kind). -----------------------------------
        tern_sum, tern_regs = _adder_tree(self, tern_lanes) # 7 levels -> 16b, n regs.
        q8_sum,   q8_regs   = _adder_tree(self, q8_lanes)   # 5 levels -> 21b, n regs.
        q8_sumb,  q8_regsb  = _adder_tree(self, q8_lanesb)  # Head B (dual-Q).
        tern_sumb, tern_regsb = _adder_tree(self, tern_lanesb)
        n_tree = max(tern_regs, q8_regs)
        tern_sum  = _delay(self, tern_sum,  n_tree - tern_regs)
        q8_sum    = _delay(self, q8_sum,    n_tree - q8_regs)
        q8_sumb   = _delay(self, q8_sumb,   n_tree - q8_regsb)
        tern_sumb = _delay(self, tern_sumb, n_tree - tern_regsb)

        gs  = Signal((22, True))
        gsb = Signal((22, True))
        self.comb += [
            gs.eq(Mux(self.kind == KIND_OP_Q8, q8_sum, tern_sum)),
            gsb.eq(Mux(self.kind == KIND_OP_Q8, q8_sumb, tern_sumb)),
        ]

        # Align tags and scale with the tree output. Lanes are registered on v1 (valid T2),
        # the tree adds n_tree stages -> gs valid at T{2+n_tree}. v1 is high at T1, so tags
        # need n_tree+1 delays; scale_f16 (valid T2) needs n_tree.
        v2        = _delay(self, v1,        n_tree + 1)
        row2      = _delay(self, row1,      n_tree + 1)
        row_last2 = _delay(self, row_last1, n_tree + 1)
        scale2    = _delay(self, scale_f16, n_tree)

        # fp16 decompose (comb) at the tree output. All arithmetic goes through dedicated
        # signed intermediates: Migen's Verilog emission computes sub-expressions at
        # self-determined widths (e.g. `f_exp - 25` at 5 bits inside a concat) and mixes
        # signed/unsigned operands into unsigned expressions - both silently corrupt values.
        f_sign  = Signal()
        f_exp   = Signal(5)
        f_frac  = Signal(10)
        m11u    = Signal(11)
        m11s    = Signal((12, True))
        m11     = Signal((13, True))
        f_exp_s = Signal((7, True))
        e16n    = Signal((7, True))
        e16     = Signal((7, True))
        self.comb += [
            f_sign.eq(scale2[15]),
            f_exp.eq(scale2[10:15]),
            f_frac.eq(scale2[:10]),
            m11u.eq(Mux(f_exp != 0, Cat(f_frac, 1), Cat(f_frac, 0))),
            m11s.eq(m11u),                # Zero-extend into a signed signal.
            m11.eq(Mux(f_sign, -m11s, m11s)),
            f_exp_s.eq(f_exp),            # Zero-extend into a signed signal.
            e16n.eq(f_exp_s - 25),        # Full-width signed subtract (dedicated assign).
            e16.eq(Mux(f_exp != 0, e16n, -24)),
        ]

        # T+1: product (DSP). ---------------------------------------------------------------------
        prod   = Signal((34, True))
        prod2  = Signal((34, True)) # Head B (dual-Q Q8): shares the row scale.
        bias_s = Signal((8, True)) # Signed copy of B: signed-only arithmetic for sh_r.
        sh_r   = Signal((8, True)) # 12 - (e16 + B) in [0, 44].
        v3     = Signal()
        row3   = Signal(16)
        rlast3 = Signal()
        self.comb += bias_s.eq(self.bias)
        self.sync += [
            prod.eq(gs*m11),
            prod2.eq(gsb*m11),
            sh_r.eq(12 - e16 - bias_s),
            v3.eq(v2),
            row3.eq(row2),
            rlast3.eq(row_last2),
        ]

        # T+2: shift to accumulator scale (contribution = prod << (e16+B), via <<12 then >>). -----
        contrib  = Signal((46, True))
        contrib2 = Signal((46, True))
        v4      = Signal()
        row4    = Signal(16)
        rlast4  = Signal()
        shamt   = Signal(6)
        self.comb += shamt.eq(Mux(sh_r > 45, 45, sh_r)) # >>45 of a 46-bit value = sign only.
        self.sync += [
            contrib.eq((prod << 12) >> shamt),
            contrib2.eq((prod2 << 12) >> shamt),
            v4.eq(v3),
            row4.eq(row3),
            rlast4.eq(rlast3),
        ]

        # T+3: 48-bit accumulate; emit on row_last. -----------------------------------------------
        acc      = Signal((48, True))
        acc_new  = Signal((48, True))
        result   = Signal((48, True))
        acc2     = Signal((48, True))
        acc2_new = Signal((48, True))
        result2  = Signal((48, True))
        self.comb += [
            acc_new.eq(acc + contrib),
            result.eq(acc_new >> self.rshift),
            acc2_new.eq(acc2 + contrib2),
            result2.eq(acc2_new >> self.rshift), # Scores share R across the head pair.
        ]

        out_fifo = stream.SyncFIFO([("row", 16), ("data", 32), ("head", 1)], out_fifo_depth)
        # AXPY wide-drain FIFO: 8 accumulators per entry (one full beat). Matvec and AXPY
        # entries never interleave (drain runs only in AXPY mode, row-last pushes only in
        # matvec kinds), so the source muxes by kind.
        axw_fifo = stream.SyncFIFO([("row", 16), ("data", 256), ("head", 1)], 4)
        self.submodules += out_fifo, axw_fifo

        # Dual-Q: head B's result is staged one cycle behind head A's push (rows are >= 2
        # beats apart, so the slot is always free before the next row-last).
        pendb_v   = Signal()
        pendb_row = Signal(16)
        pendb_dat = Signal(32)
        self.sync += [
            If(self.start,
                acc.eq(0),
                acc2.eq(0),
                pendb_v.eq(0),
            ).Elif(v4,
                acc.eq(Mux(rlast4, 0, acc_new)),
                acc2.eq(Mux(rlast4, 0, acc2_new)),
                If(rlast4 & self.qn
                   & ((self.kind == KIND_OP_Q8) | (self.kind == KIND_OP_TERNARY)),
                    pendb_v.eq(1),
                    pendb_row.eq(row4),
                    pendb_dat.eq(result2[:32]),
                )
            ).Elif(pendb_v,
                pendb_v.eq(0),
            ),
            If(self.start,
                self.overflow.eq(0),
            ).Elif(out_fifo.sink.valid & ~out_fifo.sink.ready & (self.kind != KIND_OP_AXPY),
                # Matvec pushes don't wait: valid & !ready here is a dropped result. The
                # AXPY drain FSM holds valid until accepted - backpressure, not a drop.
                self.overflow.eq(1),
            ),
        ]
        self.comb += [
            If(self.kind == KIND_OP_AXPY,
                axw_fifo.sink.valid.eq(drain_val),
                axw_fifo.sink.row.eq(Cat(C(0, 3), d_cnt)), # First lane index = d_cnt*8.
                axw_fifo.sink.head.eq(drain_head),
                axw_fifo.sink.data.eq(Mux(drain_head,
                    Cat(*[acc2_flat[(d_cnt << 3) + i][0:32] for i in range(8)]),
                    Cat(*[acc_flat[(d_cnt << 3) + i][0:32] for i in range(8)]))),
            ).Elif(pendb_v,
                out_fifo.sink.valid.eq(1),
                out_fifo.sink.row.eq(pendb_row),
                out_fifo.sink.data.eq(pendb_dat),
                out_fifo.sink.head.eq(1),
            ).Else(
                out_fifo.sink.valid.eq(v4 & rlast4),
                out_fifo.sink.row.eq(row4),
                out_fifo.sink.data.eq(result[:32]),
            ),
            out_fifo_rdy_ax.eq(axw_fifo.sink.ready),
            # 8 slots of slack cover in-flight results (<= 3 row-lasts in the T+0..T+3
            # stages, x2 for the dual-Q staged head-B pushes).
            out_fifo_room.eq(out_fifo.level < (out_fifo_depth - 8)),
            If(self.kind == KIND_OP_AXPY,
                self.source.valid.eq(axw_fifo.source.valid),
                self.source.row.eq(axw_fifo.source.row),
                self.source.data.eq(axw_fifo.source.data),
                self.source.head.eq(axw_fifo.source.head),
                self.source.wide.eq(1),
                axw_fifo.source.ready.eq(self.source.ready),
            ).Else(
                self.source.valid.eq(out_fifo.source.valid),
                self.source.row.eq(out_fifo.source.row),
                self.source.data.eq(out_fifo.source.data),
                self.source.head.eq(out_fifo.source.head),
                out_fifo.source.ready.eq(self.source.ready),
            ),
        ]

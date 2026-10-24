#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""ChannelPipe: one memory channel's compute pipeline (unpack + dot + scale-accumulate).

Consumes the weight-beat stream of one channel (row packets: scale beats then code beats, see
model.layout), computes one dot product per row against the local x-buffer and emits
int32 results, implementing model.bitexact.matvec_ternary()/matvec_q8() bit-exactly:

  ternary: 128 lanes of {-x, 0, +x} (2-bit codes) -> pipelined adder tree -> group_sum
  q8     : 32 lanes of w8*x8 -> adder tree -> group_sum
  both   : group_sum * m11(fp16 mantissa) shifted by (e16 + B) into a 48-bit accumulator,
           per-row result = acc >> R, truncated to int32.

The per-row/beat structure (scale_beats, code_beats, rows) and B/R come from quasi-static
op parameter signals set by the engine before `start`. Fully pipelined: accepts one beat per
cycle; results (1 per row) go through a small FIFO to the gather stage.
"""

from operator import or_
from functools import reduce

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

# Constants ----------------------------------------------------------------------------------------

KIND_OP_TERNARY = 0
KIND_OP_Q8      = 1
KIND_OP_AXPY    = 2 # Weighted-V: acc[lane] += (p8 * v8) << (10 + e16 + B - Rpre).

# Helpers ------------------------------------------------------------------------------------------

def _delay(module, sig, cycles, reset_less=False):
    """Return `sig` delayed by `cycles` sync stages (reset_less: a datapath delay line)."""
    cur = sig
    for _ in range(cycles):
        nxt = Signal.like(cur, reset_less=reset_less)
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
                r = Signal.like(s, reset_less=True)
                module.sync += r.eq(s)
                regs.append(r)
            nxt = regs
            n_regs += 1
        level = nxt
    return level[0], n_regs

# Channel Pipe -------------------------------------------------------------------------------------

class ChannelPipe(LiteXModule):
    """One memory channel's compute pipeline: weight beats in, one int32 result per row out.

    axpy: the weighted-V accumulate (kind 2: per-lane shifters, 4 x 32 int32 accumulators, the
    wide drain) - the Qwen3 path's attention; a hybrid build never issues it and leaves it out.

    nrows: activation rows per weight beat (doc/plan-multirow.md). Each code beat is applied to
    `nrows` x vectors at once: the x-buffer is 1024*nrows bits wide (row r's word in bits
    [1024r, 1024r+1024), written as quarter 4r + sub through xw_sub), and the ternary lanes,
    adder tree, product, shift and accumulator are replicated per row; the beat parser, scale
    path and codes are shared. A result entry then carries row r's int32 in data lane r (instead
    of one int32 replicated in all 8 lanes). Q8 and AXPY stay single-row (row 0's x word).
    nrows=1 is the single-row pipe, netlist unchanged."""
    def __init__(self, xbuf_groups=96, out_fifo_depth=24, chan_id=0, n_channels=1, axpy=True,
                 xw_cd="sys", nrows=1):
        assert 1 <= nrows <= 8 # One int32 per row in the 256-bit result entry.
        # xw_cd: the x-buffer write port's clock domain (PipeIsland: the engine's, the rest of
        # the pipe in its own).
        self.axpy  = axpy
        self.nrows = nrows
        # Op parameters (quasi-static while busy; registered by the engine every cycle, reset-less:
        # an op only reads them from `start` on).
        self.kind        = Signal(2,  reset_less=True)
        self.rows        = Signal(16, reset_less=True)
        self.scale_beats = Signal(4,  reset_less=True) # Scale beats per row (>= 1).
        self.code_beats  = Signal(8,  reset_less=True) # Code beats (= groups) per row.
        self.bias        = Signal(7,  reset_less=True) # B.
        self.rshift      = Signal(6,  reset_less=True) # R.
        self.qn          = Signal(reset_less=True)     # Dual-Q GQA mode (Q8/AXPY): 2 heads/stream.
        self.rshift2     = Signal(6,  reset_less=True) # Head-B AXPY pre-shift.
        self.start       = Signal()    # Pulse: reset counters/accumulator/error.

        # Weight beats in, results out.
        self.sink = stream.Endpoint([("data", 256)])
        # Matvec results present one int32 in data[:32] (wide=0; nrows > 1: row r's in lane r);
        # AXPY drain entries carry 8 accumulators = one full scratchpad beat (wide=1, row = first
        # lane index).
        # head (gather adds the qstride offset when set) is always 0.
        self.source = stream.Endpoint([("row", 16), ("data", 256), ("wide", 1), ("head", 1)])

        # x-buffer broadcast write port (from the engine's fill FSM).
        self.xw_group = Signal(max=max(xbuf_groups, 2))
        self.xw_sub   = Signal(max=4*nrows) # Quarter of the x word (+ 4*row when nrows > 1).
        self.xw_data  = Signal(256)
        self.xw_we    = Signal()

        # Status.
        self.error    = Signal() # Sticky: reserved code 3.
        self.overflow = Signal() # Sticky: result FIFO overflow (rate mismatch, debug).

        # # #

        out_fifo_rdy_ax = Signal() # Drain flow control (driven at the FIFO hookup below).
        out_fifo_room   = Signal() # Beat-intake flow control (driven at the FIFO hookup below).

        # x-buffer: 1024-bit words, one per ternary group (4 q8 groups), x nrows ------------------
        xbuf   = Memory(1024*nrows, xbuf_groups)
        self.specials += xbuf
        xbuf_w = xbuf.get_port(write_capable=True, we_granularity=256, clock_domain=xw_cd)
        xbuf_r = xbuf.get_port()
        self.specials += xbuf_w, xbuf_r
        self.comb += [
            xbuf_w.adr.eq(self.xw_group),
            xbuf_w.dat_w.eq(Replicate(self.xw_data, 4*nrows)),
            xbuf_w.we.eq(Mux(self.xw_we, 1 << self.xw_sub, 0)),
        ]

        # Scale RAM: 16 fp16 per beat, double-buffered by row parity -------------------------------
        scale_ram = Memory(256, 16) # 8 words per bank, 2 banks.
        self.specials += scale_ram
        scale_w = scale_ram.get_port(write_capable=True)
        scale_r = scale_ram.get_port()
        self.specials += scale_w, scale_r

        # Beat parser ------------------------------------------------------------------------------
        phase   = Signal() # 0: scale beats, 1: code beats.
        s_cnt   = Signal(4)
        g_cnt   = Signal(8)
        row_cnt = Signal(16)

        accept   = Signal()
        is_code  = Signal()
        row_last = Signal()
        self.comb += [
            # Beat intake throttles on result-FIFO headroom: at N channels the aggregate
            # result rate (N/row_beats per cycle) can exceed the 1/cycle gather drain.
            self.sink.ready.eq(out_fifo_room),
            accept.eq(self.sink.valid & self.sink.ready),
        ]
        self.comb += [
            is_code.eq(phase == 1),
            row_last.eq(is_code & (g_cnt == self.code_beats - 1)),
        ]
        self.sync += [
            If(self.start,
                phase.eq(0), s_cnt.eq(0), g_cnt.eq(0), row_cnt.eq(0),
            ).Elif(accept,
                If(~is_code,
                    s_cnt.eq(s_cnt + 1),
                    If(s_cnt == self.scale_beats - 1, phase.eq(1), g_cnt.eq(0)),
                ).Else(
                    g_cnt.eq(g_cnt + 1),
                    If(row_last, phase.eq(0), s_cnt.eq(0), row_cnt.eq(row_cnt + 1)),
                ),
            ),
        ]
        # Scale writes (bank = row parity).
        self.comb += [
            scale_w.adr.eq(Cat(s_cnt[:3], row_cnt[0])),
            scale_w.dat_w.eq(self.sink.data),
            scale_w.we.eq(accept & ~is_code),
        ]

        # T0: code-beat acceptance: launch x-buffer + scale reads, register codes ------------------
        t0_fire = Signal()
        pidx    = Signal(16) # AXPY: probs byte index = row*n_channels + chan_id.
        # Dual-Q x layout: the fill interleaves the two heads' vectors at 32-byte
        # granularity (position 2g = head A group g, 2g+1 = head B group g), so one xbuf
        # read serves both heads. AXPY byte t of head A lives at (t>>5)*64 + (t&31), head B
        # at +32: always inside the same 1024-bit word.
        pidx_d = Signal(16) # AXPY dual: interleaved byte index of head A.
        self.comb += [
            t0_fire.eq(accept & is_code),
            pidx.eq(row_cnt*n_channels + chan_id),
            pidx_d.eq(Cat(pidx[0:5], C(0, 1), pidx[5:15])),
            xbuf_r.adr.eq(Mux((self.kind == KIND_OP_AXPY) if axpy else 0,
                              Mux(self.qn, pidx_d[7:], pidx[7:]),
                          Mux(self.kind == KIND_OP_Q8,
                              Mux(self.qn, g_cnt[1:], g_cnt[2:]),
                              Mux(self.qn, Cat(C(0, 1), g_cnt[:7]), g_cnt)))),
        ]
        self.comb += scale_r.adr.eq(Cat(g_cnt[4:7], row_cnt[0]))

        # T0 payload (reset-less: qualified by v1 and the valid tags below).
        codes_r  = Signal(256, reset_less=True)
        g_lo_r   = Signal(4, reset_less=True) # Scale index within the 16-scale word.
        q8_sub_r = Signal(2, reset_less=True) # 256-bit slice of the 1024-bit x word (q8 mode).
        pidx_r   = Signal(7, reset_less=True) # AXPY: byte within the 1024-bit x word.
        g_ax_r   = Signal(3, reset_less=True) # AXPY: accumulator group of this beat.
        self.sync += [
            If(t0_fire,
                codes_r.eq(self.sink.data),
                g_lo_r.eq(g_cnt[:4]),
                q8_sub_r.eq(Mux(self.qn, Cat(C(0, 1), g_cnt[0]), g_cnt[:2])),
                pidx_r.eq(Mux(self.qn, pidx_d[:7], pidx[:7])),
                g_ax_r.eq(g_cnt[:3]),
            )
        ]
        v1 = _delay(self, t0_fire, 1) # T1 valid.
        # v1 is a high-fanout enable (~160 lane registers plus the scale capture): per-group
        # dont_touch replicas let each copy sit with its consumers (timing).
        # _v1_copies is unused, but kept: Migen's tracer derives signal names from this function's
        # variable tables, so removing it renames signals in the generated Verilog.
        _v1_copies = []
        def v1_rep(tag):
            c = Signal(name=f"v1_rep_{tag}")
            c.attr.add(("dont_touch", "true"))
            self.sync += c.eq(t0_fire)
            _v1_copies.append(c)
            return c
        v1_tern  = [v1_rep(f"t{k}") for k in range(8)] # One per 16 ternary lanes.
        v1_q8    = [v1_rep(f"q{k}") for k in range(2)] # One per 16 Q8 lanes.
        v1_scale = v1_rep("scale")
        row1      = _delay(self, row_cnt, 1, reset_less=True)
        row_last1 = _delay(self, row_last, 1, reset_less=True) # A tag: qualified by v4.

        # T1: lane select + scale fp16 mux; register lanes -----------------------------------------
        # Ternary lanes: code 00 -> -x, 01 -> 0, 10 -> +x, 11 -> 0 (+ sticky error).
        xword = xbuf_r.dat_r if nrows == 1 else xbuf_r.dat_r[:1024] # Row 0, 1024-bit, valid at T1.

        tern_lanes = []
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
            r = Signal((9, True), reset_less=True)
            self.sync += If(v1_tern[j // 16], r.eq(lane))
            tern_lanes.append(r)
        self.comb += code3_seen.eq(v1 & (self.kind == KIND_OP_TERNARY) & reduce(or_, code3_any))
        self.sync += If(self.start,
            self.error.eq(0),
        ).Elif(code3_seen,
            self.error.eq(1),
        )

        # Extra activation rows (nrows > 1): the same lanes on row r's x word. Each row has its
        # own codes_r copy and T1 enables, so every copy keeps the single-row fan-out (128 lanes).
        # Code 3 is a property of the beat: flagged by row 0's lanes only.
        tern_lanes_x = []
        for r in range(1, nrows):
            codes_x = Signal(256, name=f"codes_r{r}", reset_less=True)
            codes_x.attr.add(("dont_touch", "true"))
            self.sync += If(t0_fire, codes_x.eq(self.sink.data))
            v1_x    = [v1_rep(f"t{k}_r{r}") for k in range(8)]
            xword_x = xbuf_r.dat_r[1024*r:1024*(r+1)]
            lanes_x = []
            for j in range(128):
                x_j  = Signal((8, True))
                c_j  = Signal(2)
                lane = Signal((9, True))
                self.comb += [
                    x_j.eq(xword_x[8*j:8*j+8]),
                    c_j.eq(codes_x[2*j:2*j+2]),
                    Case(c_j, {
                        0 : lane.eq(-x_j),
                        1 : lane.eq(0),
                        2 : lane.eq(x_j),
                        3 : lane.eq(0),
                    }),
                ]
                lr = Signal((9, True), reset_less=True)
                self.sync += If(v1_x[j // 16], lr.eq(lane))
                lanes_x.append(lr)
            tern_lanes_x.append(lanes_x)

        # Q8 lanes: w8 * x8 over the selected 256-bit x slice.
        xslice = Signal(256)
        self.comb += Case(q8_sub_r, {i: xslice.eq(xword[256*i:256*(i+1)]) for i in range(4)})
        # AXPY coefficient byte: two-stage constant-slice mux out of the 1024-bit x word.
        p8_grp = Signal(64)
        p8_ax  = Signal((8, True))
        self.comb += Case(pidx_r[3:], {i: p8_grp.eq(xword[64*i:64*(i+1)]) for i in range(16)})
        self.comb += Case(pidx_r[:3], {i: p8_ax.eq(p8_grp[8*i:8*(i+1)]) for i in range(8)})
        # Op-stable decode, registered: kind loads at START+1 and the first T1 use is at
        # START+5 or later, so a flop copy is exact and keeps the kind compare (and the
        # engine-side flop it would otherwise reach) out of the 32-lane operand mux.
        is_axpy = Signal(reset_less=True)
        if axpy:
            self.sync += is_axpy.eq(self.kind == KIND_OP_AXPY)
        else:
            self.comb += is_axpy.eq(0)
        q8_lanes = []
        # The 32 signed 8x8 products go to DSPs (saves ~2.6k LUT per channel; Vivado folds
        # the first Q8 tree levels into the DSP ALUs). Operands are registered at T1 (the
        # DSP's A/B registers), product at T2 -> valid T3, one cycle after the ternary lanes:
        # the multiply sits entirely inside the DSP (the single-cycle form failed timing).
        # Costs one cycle of Q8/AXPY op latency, not throughput.
        v2_q8 = [_delay(self, c, 1) for c in v1_q8]
        for j in range(32):
            w_j = Signal((8, True))
            x_j = Signal((8, True))
            w_r = Signal((8, True), reset_less=True) # DSP A/B registers.
            x_r = Signal((8, True), reset_less=True)
            p   = Signal((16, True))
            p.attr.add(("use_dsp", "yes"))
            self.comb += [
                w_j.eq(codes_r[8*j:8*j+8]),
                x_j.eq(Mux(is_axpy, p8_ax, xslice[8*j:8*j+8]) if axpy else xslice[8*j:8*j+8]),
                p.eq(w_r*x_r),
            ]
            self.sync += If(v1_q8[j // 16], w_r.eq(w_j), x_r.eq(x_j))
            r = Signal((16, True), reset_less=True) # DSP P register.
            self.sync += If(v2_q8[j // 16], r.eq(p))
            q8_lanes.append(r)

        # Scale fp16: the beat's halfword selected at T1 straight from the scale RAM read
        # (scale_r.dat_r and g_lo_r are both valid then) and registered as 16 bits, valid
        # at T2+ (keeps the 16:1 mux out of the AXPY shift-amount cone).
        scale_f16_t1 = Signal(16)
        self.comb += Case(g_lo_r, {i: scale_f16_t1.eq(scale_r.dat_r[16*i:16*(i+1)])
                                   for i in range(16)})
        scale_f16 = Signal(16, reset_less=True)
        self.sync += If(v1_scale, scale_f16.eq(scale_f16_t1))

        # AXPY datapath (kind 2): per-lane shift + int32 accumulators + drain ----------------------
        # Row weight is one shift (pow-2 row scales, m11 == 1024): sh = (f_exp - 15) + B - Rpre
        # with B on self.bias and Rpre on self.rshift. Lane products (v8 * p8, registered at
        # T2) shift at T2->T3, accumulate per (group, lane) at T3; after the final row a
        # drain FSM streams code_beats*32 accumulators into the out FIFO (row = g*32 + j).
        # Signed-only arithmetic through dedicated intermediates (unsigned operands mixed
        # into signed expressions emit corrupt Verilog - see the fp16 decompose below).
        # For timing, the op-stable part (B - Rpre - 15) is a register, the exponent part
        # is added at T1 from the RAM read data, and the resulting direction/amount bits are
        # registered (replicated per 8 lanes, dont_touch): T2 is register -> shifter ->
        # register.
        drain_val = Signal()
        d_cnt     = Signal(8)
        acc_flat  = None
        if not axpy:
            self.comb += drain_val.eq(0)
        if axpy:
            ax_c31n  = Signal((9, True), reset=-31)
            ax_bias  = Signal((8, True))
            ax_rpre  = Signal((8, True))
            ax_bmr   = Signal((9, True), reset_less=True) # B - Rpre - 15, registered (op-stable).
            ax_exp   = Signal(5)
            ax_exp_s = Signal((7, True))
            ax_sh    = Signal((9, True)) # T1: exponent + (B - Rpre - 15), same range as before.
            self.comb += [
                ax_bias.eq(self.bias),
                ax_rpre.eq(self.rshift),
                ax_exp.eq(scale_f16_t1[10:15]),
                ax_exp_s.eq(ax_exp),          # Zero-extend into a signed signal.
                ax_sh.eq(ax_exp_s + ax_bmr),
            ]
            self.sync += ax_bmr.eq(ax_bias - ax_rpre - 15)
            def amount_regs(sh, tag):
                """(neg, shl, shr) for `sh`: registered at T1 (valid T2), then one more stage
                so they meet the products at T3; one dont_touch copy per 8 lanes."""
                copies = []
                for k in range(4):
                    neg1 = Signal(name=f"axneg1_{tag}{k}", reset_less=True)
                    shl1 = Signal(4, name=f"axshl1_{tag}{k}", reset_less=True)
                    shr1 = Signal(5, name=f"axshr1_{tag}{k}", reset_less=True)
                    self.sync += If(v1_scale,
                        neg1.eq(sh < 0),
                        shl1.eq(Mux(sh >= 0, Mux(sh > 15, 15, sh[:4]), 0)),
                        shr1.eq(Mux(sh < 0, Mux(sh < ax_c31n, 31, (-sh)[:5]), 0)),
                    )
                    neg = Signal(name=f"axneg_{tag}{k}", reset_less=True)
                    shl = Signal(4, name=f"axshl_{tag}{k}", reset_less=True)
                    shr = Signal(5, name=f"axshr_{tag}{k}", reset_less=True)
                    for c in (neg, shl, shr):
                        c.attr.add(("dont_touch", "true"))
                    self.sync += [neg.eq(neg1), shl.eq(shl1), shr.eq(shr1)]
                    copies.append((neg, shl, shr))
                return copies
            ax_amt = amount_regs(ax_sh, "a")
            v2a = _delay(self, v1, 2) # Lane products valid (T3).
            ax_shifted = []
            for j in range(32):
                neg, shl, shr = ax_amt[j // 8]
                sh_full = Signal((32, True))
                r       = Signal((32, True), reset_less=True)
                self.comb += sh_full.eq(Mux(neg, q8_lanes[j] >> shr, q8_lanes[j] << shl))
                self.sync += If(v2a, r.eq(sh_full))
                ax_shifted.append(r)

            AX_GROUPS = 4 # head_dim <= 128 for AXPY ops (kvpack contract).
            # Accumulators: cleared by `start`, reset-less.
            acc_ax = [[Signal((32, True), name=f"axacc{g}_{j}", reset_less=True) for j in range(32)]
                      for g in range(AX_GROUPS)]
            acc_flat = Array([acc_ax[g][j] for g in range(AX_GROUPS) for j in range(32)])

            # Final-beat tag: last code beat of the last row, aligned to the accumulate stage.
            fin_t0 = Signal()
            self.comb += fin_t0.eq(t0_fire & row_last & (row_cnt == self.rows - 1))
            fin_acc = _delay(self, fin_t0, 4) # Accumulate of the final beat done (T5).

            d_total = Signal(8)
            # Wide drain: 8 accumulators (one scratchpad beat) per FIFO entry, so the gather
            # stage retires 8 partials/cycle instead of 1. d_cnt counts 8-lane groups.
            self.comb += d_total.eq(Cat(C(0, 2), self.code_beats[:3])) # code_beats * 4 = cols/8.
            self.ax_fsm = ax_fsm = FSM(reset_state="AX_IDLE")
            ax_fsm.act("AX_IDLE",
                If(self.start & (self.kind == KIND_OP_AXPY),
                    NextValue(d_cnt, 0),
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
                        NextState("AX_IDLE"),
                    )
                )
            )

            # Accumulate: one adder per lane, shared by the four groups. The beat's bank is
            # selected (4:1 per lane, select replicated per 8 lanes), added once, and written
            # back under a registered per-bank enable. Exact: the group sequence is cyclic within
            # a row and rows are separated by at least one scale beat, so the same bank is never
            # updated on consecutive beats and the single-cycle read-modify-write sees its own
            # previous write. `start` keeps priority over the enables.
            g_ax1 = _delay(self, g_ax_r, 2, reset_less=True) # Beat group at T3 (decode below -> T4).
            ax_we = [Signal(name=f"ax_we{g}") for g in range(AX_GROUPS)]
            self.sync += [ax_we[g].eq(v2a & is_axpy & (g_ax1[:2] == g)) for g in range(AX_GROUPS)]
            g_sel = []
            for k in range(4):
                c = Signal(2, name=f"ax_gsel{k}", reset_less=True)
                c.attr.add(("dont_touch", "true"))
                self.sync += c.eq(g_ax1[:2]) # The beat's group at T3, one copy per 8 lanes.
                g_sel.append(c)
            for j in range(32):
                acc_sel = Signal((32, True))
                acc_sum = Signal((32, True))
                self.comb += [
                    Case(g_sel[j // 8], {g: acc_sel.eq(acc_ax[g][j]) for g in range(AX_GROUPS)}),
                    acc_sum.eq(acc_sel + ax_shifted[j]),
                ]
                for g in range(AX_GROUPS):
                    self.sync += If(self.start,
                        acc_ax[g][j].eq(0),
                    ).Elif(ax_we[g],
                        acc_ax[g][j].eq(acc_sum),
                    )
        # (end of the AXPY datapath)

        # T2..: adder trees (both built; result muxed by kind) -------------------------------------
        tern_sum, tern_regs = _adder_tree(self, tern_lanes) # 7 levels -> 16b, n regs.
        q8_sum,   q8_regs   = _adder_tree(self, q8_lanes)   # 5 levels -> 21b, n regs.
        # Q8 lanes are valid one cycle after the ternary lanes (the DSP operand stage).
        n_tree = max(tern_regs, q8_regs + 1)
        assert n_tree == 3 and q8_regs + 1 <= n_tree
        tern_sum = _delay(self, tern_sum, n_tree - tern_regs,  reset_less=True)
        q8_sum   = _delay(self, q8_sum,   n_tree - q8_regs - 1, reset_less=True)
        gs = Signal((22, True))
        self.comb += gs.eq(Mux(self.kind == KIND_OP_Q8, q8_sum, tern_sum))
        # Extra rows: ternary only, aligned with row 0's tree output.
        gs_x = []
        for lanes_x in tern_lanes_x:
            sum_x, regs_x = _adder_tree(self, lanes_x)
            assert regs_x == tern_regs
            gs_x.append(_delay(self, sum_x, n_tree - tern_regs, reset_less=True))

        # Align tags and scale with the tree output. Lanes are registered on v1 (valid T2),
        # the tree adds n_tree stages -> gs valid at T{2+n_tree}. v1 is high at T1, so tags
        # need n_tree+1 delays; scale_f16 (valid T2) needs n_tree.
        v2        = _delay(self, v1,        n_tree + 1)
        row2      = _delay(self, row1,      n_tree + 1, reset_less=True)
        row_last2 = _delay(self, row_last1, n_tree + 1, reset_less=True)
        scale2    = _delay(self, scale_f16, n_tree,     reset_less=True)

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

        # T+1: product (DSP) -----------------------------------------------------------------------
        prod   = Signal((34, True), reset_less=True)
        bias_s = Signal((8, True)) # Signed copy of B: signed-only arithmetic for sh_c.
        sh_c   = Signal((8, True)) # 12 - (e16 + B) in [0, 44] (comb).
        shamt  = Signal(6, reset_less=True) # Saturated at T+1 (>>45 of a 46-bit value = sign only),
                                   # so T+2 is register -> 46-bit shifter -> register.
        v3     = Signal()
        row3   = Signal(16, reset_less=True)
        rlast3 = Signal(reset_less=True)
        self.comb += [
            bias_s.eq(self.bias),
            sh_c.eq(12 - e16 - bias_s),
        ]
        self.sync += [
            prod.eq(gs*m11),
            shamt.eq(Mux(sh_c > 45, 45, sh_c)),
            v3.eq(v2),
            row3.eq(row2),
            rlast3.eq(row_last2),
        ]
        prod_x = [Signal((34, True), reset_less=True) for _ in gs_x]
        self.sync += [p.eq(g*m11) for p, g in zip(prod_x, gs_x)]

        # T+2: shift to accumulator scale (contribution = prod << (e16+B), via <<12 then >>) -------
        contrib = Signal((46, True), reset_less=True)
        v4      = Signal()
        row4    = Signal(16, reset_less=True)
        rlast4  = Signal(reset_less=True)
        self.sync += [
            contrib.eq((prod << 12) >> shamt),
            v4.eq(v3),
            row4.eq(row3),
            rlast4.eq(rlast3),
        ]
        contrib_x = [Signal((46, True), reset_less=True) for _ in prod_x]
        self.sync += [c.eq((p << 12) >> shamt) for c, p in zip(contrib_x, prod_x)]

        # T+3: 48-bit accumulate; emit on row_last -------------------------------------------------
        acc     = Signal((48, True), reset_less=True) # Cleared by `start`.
        acc_new = Signal((48, True))
        result  = Signal((48, True))
        self.comb += result.eq(acc_new >> self.rshift)
        self.comb += acc_new.eq(acc + contrib)
        # Extra rows: own accumulators, same clear/emit timing as row 0.
        results = [result]
        for contrib_r in contrib_x:
            acc_r     = Signal((48, True), reset_less=True)
            acc_new_r = Signal((48, True))
            result_r  = Signal((48, True))
            self.comb += [
                result_r.eq(acc_new_r >> self.rshift),
                acc_new_r.eq(acc_r + contrib_r),
            ]
            self.sync += If(self.start,
                acc_r.eq(0),
            ).Elif(v4,
                acc_r.eq(Mux(rlast4, 0, acc_new_r)),
            )
            results.append(result_r)

        out_fifo = stream.SyncFIFO([("row", 16), ("data", 32*nrows), ("head", 1)], out_fifo_depth)
        # AXPY wide-drain FIFO: 8 accumulators per entry (one full beat). Matvec and AXPY
        # entries never interleave (drain runs only in AXPY mode, row-last pushes only in
        # matvec kinds), so the source muxes by kind.
        self.submodules += out_fifo
        axw_fifo = None
        if axpy:
            axw_fifo = stream.SyncFIFO([("row", 16), ("data", 256), ("head", 1)], 4)
            self.submodules += axw_fifo

        self.sync += [
            If(self.start,
                acc.eq(0),
            ).Elif(v4,
                acc.eq(Mux(rlast4, 0, acc_new)),
            ),
            If(self.start,
                self.overflow.eq(0),
            ).Elif(out_fifo.sink.valid & ~out_fifo.sink.ready & ((self.kind != KIND_OP_AXPY) if axpy else 1),
                # Matvec pushes don't wait: valid & !ready here is a dropped result. The
                # AXPY drain FSM holds valid until accepted - backpressure, not a drop.
                self.overflow.eq(1),
            ),
        ]
        matvec_sink = [
            out_fifo.sink.valid.eq(v4 & rlast4),
            out_fifo.sink.row.eq(row4),
            out_fifo.sink.data.eq(result[:32] if nrows == 1 else Cat(*[r[:32] for r in results])),
        ]
        matvec_source = [
            self.source.valid.eq(out_fifo.source.valid),
            self.source.row.eq(out_fifo.source.row),
            # The 32-bit result in every lane: the engine's result packer keeps a
            # per-lane register and enables only the lane this result belongs to, so
            # replicating here (pure fanout) lets the packer carry no data mux.
            # nrows > 1: row r's result in lane r (the engine splits them per row).
            self.source.data.eq(Replicate(out_fifo.source.data, 8) if nrows == 1 else
                                out_fifo.source.data),
            self.source.head.eq(out_fifo.source.head),
            out_fifo.source.ready.eq(self.source.ready),
        ]
        # 9 slots of slack cover in-flight results (<= 4 row-lasts in the T+0..T+4
        # stages with the Q8 operand register, x2 margin).
        self.comb += out_fifo_room.eq(out_fifo.level < (out_fifo_depth - 9))
        if axpy:
            self.comb += [
                If(self.kind == KIND_OP_AXPY,
                    axw_fifo.sink.valid.eq(drain_val),
                    axw_fifo.sink.row.eq(Cat(C(0, 3), d_cnt)), # First lane index = d_cnt*8.
                    axw_fifo.sink.data.eq(Cat(*[acc_flat[(d_cnt << 3) + i][0:32]
                        for i in range(8)])),
                ).Else(*matvec_sink),
                out_fifo_rdy_ax.eq(axw_fifo.sink.ready),
                If(self.kind == KIND_OP_AXPY,
                    self.source.valid.eq(axw_fifo.source.valid),
                    self.source.row.eq(axw_fifo.source.row),
                    self.source.data.eq(axw_fifo.source.data),
                    self.source.head.eq(axw_fifo.source.head),
                    self.source.wide.eq(1),
                    axw_fifo.source.ready.eq(self.source.ready),
                ).Else(*matvec_source),
            ]
        else:
            self.comb += matvec_sink + matvec_source + [out_fifo_rdy_ax.eq(0)]

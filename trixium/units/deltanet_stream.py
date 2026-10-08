#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Streaming exact-FP32 DeltaNet recurrence: one command runs every head of a layer.

The state never sits in the scratchpad. Lane ``l`` owns rows ``l, l+L, l+2L, ...`` of every
head (``R = rows/L`` of them) and receives them as its own 256-bit stream, straight from
the engine reader of channel ``l``; the updated rows leave on the lane's source stream to
the same channel's writer. Per lane and head the stream carries ``G*R`` beats (``G =
cols/8``) in column-group-major order::

    beat g*R + r = row (r*L + l), columns 8g .. 8g+7

so the recurrence walks columns in order over R interleaved rows without a transpose
buffer (R >= adder latency + 1 covers the accumulator feedback). The layout exists only
in engine DDR and only this unit reads or writes it (``state_to_lanes`` /
``lanes_to_state`` convert).

Per row the arithmetic is the firmware's loop, rounded after every operation::

    m = m*decay; d = d + m*k[i]            (P1, i in order, d from +0)
    delta = (v - d)*beta
    m = m + delta*k[i]; o = o + m*q[i]     (P2, i in order, o from +0)
    out = o*scale

P1 of head h+1 overlaps P2 of head h, so a head costs ``cols*R`` clocks: 1,024 for the 27B
at 16 lanes. delta and the output scale run on three shared operators over snapshots of
the per-row accumulators. Buffer ownership, each a head-count comparison:

    decayed rows (md)   4 quarters   P1(h) starts once P2(h-3) has issued (P2 trails P1 by
                                     the delta latency, so 2 halves would stall P1)
    d snapshot          1 bank       P1(h) ends >= cols*R clocks after delta(h-1) could
                                     start, and delta(h-1) is free to start then (below)
    delta registers     2 banks      delta(h) starts once P2(h-2) has issued
    o snapshot          2 banks      P2(h) starts once scale(h-2) has issued
    parameters          `slots`      head h loads once head h-slots is retired

Parameters come from the scratchpad: ``table`` points at one 256-bit record per head,
``[q, k, v, out byte addresses, decay, beta, scale, 0]``, and each vector is read in
beats (q/k: cols floats, v and out: rows floats). Records may alias (GQA heads share q/k).
With ``zero`` set at start every incoming state value is replaced by +0.0 (the same bits as
a cleared state: +0 * decay = +0); the state streams are still consumed.
"""

from functools import reduce
from operator import and_, or_

import numpy as np

from migen import *

from litex.gen import *
from litex.soc.interconnect import stream

from trixium.units.fp32 import ExactFP32Add, ExactFP32Mul


def state_to_lanes(states, lanes):
    """[heads, rows, cols] float32 -> [lanes, heads*G*R*8] uint32 (each lane's stream)."""
    states = np.asarray(states, dtype=np.float32)
    h, rows, cols = states.shape
    r, g = rows//lanes, cols//8
    s = states.view(np.uint32).reshape(h, r, lanes, g, 8)
    return np.ascontiguousarray(s.transpose(2, 0, 3, 1, 4).reshape(lanes, -1))


def lanes_to_state(words, lanes, rows, cols):
    words = np.asarray(words, dtype=np.uint32).reshape(lanes, -1)
    r, g = rows//lanes, cols//8
    h = words.shape[1]//(g*r*8)
    s = words.reshape(lanes, h, g, r, 8).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(s.reshape(h, rows, cols)).view(np.float32)


def reference(state, q, k, v, decay, beta, scale):
    """One head, the firmware loop (hybrid.c), vectorised over rows. Returns (state, out)."""
    f32 = np.float32
    m = np.array(state, dtype=f32)
    rows, cols = m.shape
    with np.errstate(all="ignore"):
        d = np.zeros(rows, f32)
        for i in range(cols):
            m[:, i] = (m[:, i]*f32(decay)).astype(f32)
            d = (d + (m[:, i]*f32(k[i])).astype(f32)).astype(f32)
        delta = ((np.asarray(v, f32) - d).astype(f32)*f32(beta)).astype(f32)
        o = np.zeros(rows, f32)
        for i in range(cols):
            m[:, i] = (m[:, i] + (delta*f32(k[i])).astype(f32)).astype(f32)
            o = (o + (m[:, i]*f32(q[i])).astype(f32)).astype(f32)
        out = (o*f32(scale)).astype(f32)
    return m, out


class BonsaiDeltaNetStream(LiteXModule):
    def __init__(self, lanes=16, rows=128, cols=128, slots=4, adr_bits=13,
                 add_factory=ExactFP32Add, mul_factory=ExactFP32Mul,
                 negate=lambda x: x ^ 0x80000000, fifo_depth=8):
        L = lanes
        pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        assert pow2(L) and pow2(rows) and pow2(cols) and pow2(slots)
        assert rows % L == 0 and cols >= 16 and rows >= 16
        R, G, VB = rows//L, cols//8, rows//8
        La, Lm = add_factory.latency, mul_factory.latency
        assert R >= La + 1, "rows per lane must cover the accumulator feedback latency"
        assert cols*R >= rows + La + Lm + 8, "delta must finish before the next snapshot"
        rb, gb, sb = (R - 1).bit_length(), (G - 1).bit_length(), max(1, (slots - 1).bit_length())
        vbb = (VB - 1).bit_length()
        self.lanes, self.rows, self.cols, self.rows_per_lane = L, rows, cols, R
        self.beats_per_head = G*R   # Per lane.

        self.start = Signal()
        self.heads = Signal(16)
        self.table = Signal(adr_bits)   # Scratchpad beat address of the head records.
        # Sampled with start: the incoming state is taken as +0.0 (the stream is still read and
        # discarded), so a generation's first token needs no 144 MiB clear of the DDR state.
        self.zero  = Signal()
        self.busy  = Signal()
        self.done  = Signal()           # Pulse: every head's outputs and state written.
        self.sinks   = [stream.Endpoint([("data", 256)], name=f"dns_sink{l}") for l in range(L)]
        self.sources = [stream.Endpoint([("data", 256)], name=f"dns_source{l}") for l in range(L)]
        # Scratchpad client ports. Reads: sp_rd_req/sp_rd_adr come from flops and hold until
        # a cycle with sp_rd_ok (the grant); the data is on sp_rd_dat one clock later.
        # Writes: held until granted.
        self.sp_rd_ok  = Signal(reset=1)
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(adr_bits)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(adr_bits)
        self.sp_wr_we  = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()
        self.heads_done = Signal(16)

        # # #

        def dly(value, n, name):
            for i in range(n):
                r = Signal.like(value, name=f"{name}_d{i}")
                self.sync += r.eq(value)
                value = r
            return value

        def fsel(word, index):
            return Array([word[32*i:32*(i + 1)] for i in range(8)])[index]

        run   = Signal()
        heads = Signal(16)
        table = Signal(adr_bits)
        dat   = self.sp_rd_dat

        # Parameters: `slots` heads resident (loading, P1, P2/delta, scale+write).
        qmem = Memory(256, slots*G, name="dns_q")
        kmem = Memory(256, slots*G, name="dns_k")
        vmem = Memory(256, slots*VB, name="dns_v")
        q_w, k_w, v_w = [m.get_port(write_capable=True) for m in (qmem, kmem, vmem)]
        q_r  = qmem.get_port(async_read=True)
        k_r1 = kmem.get_port(async_read=True)
        k_r2 = kmem.get_port(async_read=True)
        v_r  = vmem.get_port(async_read=True)
        self.specials += qmem, kmem, vmem, q_w, k_w, v_w, q_r, k_r1, k_r2, v_r
        dec = Array(Signal(32, name=f"dns_decay{s}") for s in range(slots))
        bet = Array(Signal(32, name=f"dns_beta{s}") for s in range(slots))
        scl = Array(Signal(32, name=f"dns_scale{s}") for s in range(slots))
        oad = Array(Signal(adr_bits, name=f"dns_out{s}") for s in range(slots))

        # Head counters (all reset by start).
        ld_head  = Signal(16)   # Heads whose parameters are loaded.
        p1_head  = Signal(16)   # Heads whose P1 issue finished.
        p1_done  = Signal(16)   # Heads whose d snapshot was taken.
        dl_head  = Signal(16)   # Heads whose delta issue finished.
        dl_done  = Signal(16)   # Heads whose deltas are all written.
        p2_head  = Signal(16)   # Heads whose P2 issue finished (their md quarter is free).
        p2_done  = Signal(16)   # Heads whose o snapshot was taken.
        sc_head  = Signal(16)   # Heads whose scale issue finished (their o bank is free).
        retired  = Signal(16)   # Heads whose outputs are in the scratchpad.
        em_cnt   = Signal(16)   # State groups emitted to the sources (all lanes).

        zero_r = Signal()
        self.sync += If(self.start, zero_r.eq(self.zero))

        # Loader: one registered read request (address from a flop, granted by sp_rd_ok),
        # a record then q, k, v beats per head; responses one clock after the grant.
        nvec = 2*G + VB
        ld_idx = Signal(max=nvec + 1)
        ld_qb, ld_kb, ld_vb = Signal(adr_bits), Signal(adr_bits), Signal(adr_bits)
        ld_st = Signal(2)       # 0 idle, 1 record in flight, 2 vectors, 3 last vector in flight
        ld_slot = Signal(sb)
        self.comb += ld_slot.eq(ld_head[:sb])
        vk, voff = Signal(2), Signal(max(gb, vbb))
        self.comb += If(ld_idx < G, vk.eq(0), voff.eq(ld_idx)
            ).Elif(ld_idx < 2*G, vk.eq(1), voff.eq(ld_idx - G)
            ).Else(vk.eq(2), voff.eq(ld_idx - 2*G))
        rq_v, rq_adr, rq_kind, rq_off, rq_last = (Signal(), Signal(adr_bits), Signal(2),
                                                  Signal(max(gb, vbb)), Signal())
        rsp_v, rsp_kind, rsp_off, rsp_last = Signal(), Signal(2), Signal(max(gb, vbb)), Signal()
        taken, free = Signal(), Signal()
        self.comb += [
            taken.eq(rq_v & self.sp_rd_ok),
            free.eq(~rq_v | taken),
            self.sp_rd_req.eq(rq_v),
            self.sp_rd_adr.eq(rq_adr),
        ]
        self.sync += [
            rsp_v.eq(taken), rsp_kind.eq(rq_kind), rsp_off.eq(rq_off), rsp_last.eq(rq_last),
            If(self.start,
                ld_st.eq(0), rq_v.eq(0), ld_head.eq(0),
            ).Else(
                If(taken, rq_v.eq(0)),
                Case(ld_st, {
                    0: If(run & (ld_head < heads) & (ld_head < retired + slots) & free,
                        rq_v.eq(1), rq_adr.eq(table + ld_head), rq_kind.eq(3), rq_last.eq(0),
                        ld_st.eq(1),
                    ),
                    1: If(rsp_v & (rsp_kind == 3),
                        ld_qb.eq(dat[5:5 + adr_bits]),
                        ld_kb.eq(dat[32 + 5:32 + 5 + adr_bits]),
                        ld_vb.eq(dat[64 + 5:64 + 5 + adr_bits]),
                        oad[ld_slot].eq(dat[96 + 5:96 + 5 + adr_bits]),
                        dec[ld_slot].eq(dat[128:160]),
                        bet[ld_slot].eq(dat[160:192]),
                        scl[ld_slot].eq(dat[192:224]),
                        ld_idx.eq(0),
                        ld_st.eq(2),
                    ),
                    2: If(free,
                        If(ld_idx == nvec,
                            ld_st.eq(3),
                        ).Else(
                            rq_v.eq(1),
                            rq_adr.eq(Mux(vk == 0, ld_qb, Mux(vk == 1, ld_kb, ld_vb)) + voff),
                            rq_kind.eq(vk), rq_off.eq(voff), rq_last.eq(ld_idx == nvec - 1),
                            ld_idx.eq(ld_idx + 1),
                        )
                    ),
                    3: If(rsp_v & rsp_last,     # Written at this edge.
                        ld_head.eq(ld_head + 1),
                        ld_st.eq(0),
                    ),
                })
            )
        ]
        self.comb += [
            q_w.adr.eq(Cat(rsp_off[:gb], ld_slot)), q_w.dat_w.eq(dat),
            q_w.we.eq(rsp_v & (rsp_kind == 0)),
            k_w.adr.eq(Cat(rsp_off[:gb], ld_slot)), k_w.dat_w.eq(dat),
            k_w.we.eq(rsp_v & (rsp_kind == 1)),
            v_w.adr.eq(Cat(rsp_off[:vbb], ld_slot)), v_w.dat_w.eq(dat),
            v_w.we.eq(rsp_v & (rsp_kind == 2)),
        ]

        # Operators ---------------------------------------------------------------------------
        def op(factory, name):
            o = factory()
            setattr(self.submodules, name, o)
            return o
        mul_dec = [op(mul_factory, f"mul_dec{l}") for l in range(L)]
        mul_k   = [op(mul_factory, f"mul_k{l}")   for l in range(L)]
        add_d   = [op(add_factory, f"add_d{l}")   for l in range(L)]
        mul_dk  = [op(mul_factory, f"mul_dk{l}")  for l in range(L)]
        add_m   = [op(add_factory, f"add_m{l}")   for l in range(L)]
        mul_q   = [op(mul_factory, f"mul_q{l}")   for l in range(L)]
        add_o   = [op(add_factory, f"add_o{l}")   for l in range(L)]
        add_v   = op(add_factory, "add_v")
        mul_b   = op(mul_factory, "mul_b")
        mul_s   = op(mul_factory, "mul_s")

        # Input half-buffers (per lane, two column groups) -------------------------------------
        p1_grp = Signal(2)       # Column groups consumed by P1 (mod 4, all lanes).
        fill_grp = []
        inbufs = []
        for l, sink in enumerate(self.sinks):
            mem = Memory(256, 2*R, name=f"dns_in{l}")
            wp = mem.get_port(write_capable=True)
            rp = mem.get_port()
            self.specials += mem, wp, rp
            inbufs.append(rp)
            fi = Signal(rb + 1, name=f"dns_fill{l}")
            fg = Signal(2, name=f"dns_fillgrp{l}")
            fill_grp.append(fg)
            self.comb += [
                sink.ready.eq(run & ((fg - p1_grp)[:2] != 2)),
                wp.adr.eq(fi), wp.dat_w.eq(sink.data), wp.we.eq(sink.valid & sink.ready),
            ]
            self.sync += If(self.start,
                fi.eq(0), fg.eq(0),
            ).Elif(sink.valid & sink.ready,
                fi.eq(fi + 1),
                If(fi[:rb] == R - 1, fg.eq(fg + 1)),
            )
        grp_ok = reduce(and_, [fg != p1_grp for fg in fill_grp])

        # P1: decay, k dot ---------------------------------------------------------------------
        p1_run = Signal()
        p1_g, p1_jj, p1_r = Signal(gb), Signal(3), Signal(rb)
        p1_issue = Signal()
        self.comb += p1_issue.eq(p1_run & grp_ok)
        p1_slot = p1_head[:sb]
        self.comb += k_r1.adr.eq(Cat(p1_g, p1_slot))
        for rp in inbufs:
            self.comb += rp.adr.eq(Cat(p1_r, p1_grp[0]))
        self.sync += If(self.start,
            p1_run.eq(0), p1_head.eq(0), p1_grp.eq(0),
        ).Elif(~p1_run,
            If(run & (p1_head < heads) & (p1_head < ld_head) & (p1_head < p2_head + 3),
                p1_run.eq(1), p1_g.eq(0), p1_jj.eq(0), p1_r.eq(0),
            )
        ).Elif(p1_issue,
            p1_r.eq(p1_r + 1),
            If(p1_r == R - 1,
                p1_jj.eq(p1_jj + 1),
                If(p1_jj == 7,
                    p1_grp.eq(p1_grp + 1),
                    p1_g.eq(p1_g + 1),
                    If(p1_g == G - 1,
                        p1_run.eq(0),
                        p1_head.eq(p1_head + 1),
                    )
                )
            )
        )
        s1_v, s1_first, s1_last = Signal(), Signal(), Signal()
        s1_r, s1_jj = Signal(rb), Signal(3)
        s1_mda = Signal(rb + 3 + gb + 2)
        s1_k, s1_dec = Signal(32), Signal(32)
        self.sync += [
            s1_v.eq(p1_issue),
            s1_first.eq((p1_g == 0) & (p1_jj == 0)),
            s1_last.eq((p1_g == G - 1) & (p1_jj == 7) & (p1_r == R - 1)),
            s1_r.eq(p1_r), s1_jj.eq(p1_jj),
            s1_mda.eq(Cat(p1_r, p1_jj, p1_g, p1_head[:2])),
            s1_k.eq(fsel(k_r1.dat_r, p1_jj)),
            s1_dec.eq(dec[p1_slot]),
        ]
        # u: one register after the input buffer's float select (memory -> 8:1 mux ->
        # multiplier is not one clock at speed).
        u_v, u_dec = dly(s1_v, 1, "uv"), dly(s1_dec, 1, "udec")
        t2_v, t2_mda, t2_k = dly(s1_v, Lm + 1, "t2v"), dly(s1_mda, Lm + 1, "t2mda"), dly(s1_k, Lm + 1, "t2k")
        t2_first, t2_last, t2_r = (dly(s1_first, Lm + 1, "t2f"), dly(s1_last, Lm + 1, "t2l"),
                                   dly(s1_r, Lm + 1, "t2r"))
        t3_v, t3_first = dly(t2_v, Lm, "t3v"), dly(t2_first, Lm, "t3f")
        t3_last, t3_r = dly(t2_last, Lm, "t3l"), dly(t2_r, Lm, "t3r")
        t4_v, t4_last, t4_r = dly(t3_v, La, "t4v"), dly(t3_last, La, "t4l"), dly(t3_r, La, "t4r")
        p1_capture = Signal()
        self.comb += p1_capture.eq(t4_v & t4_last)

        # Per-lane accumulators and the snapshot the shared delta pipe drains.
        dacc = [[Signal(32, name=f"dns_d{l}_{r}") for r in range(R)] for l in range(L)]
        oacc = [[Signal(32, name=f"dns_o{l}_{r}") for r in range(R)] for l in range(L)]
        dsnap = [Signal(32, name=f"dns_dsnap{i}") for i in range(rows)]     # i = r*L + l
        osnap = [[Signal(32, name=f"dns_osnap{b}_{i}") for i in range(rows)] for b in range(2)]
        dreg  = [[Signal(32, name=f"dns_delta{b}_{i}") for i in range(rows)] for b in range(2)]

        mdmems = []
        for l in range(L):
            md = Memory(32, 4*cols*R, name=f"dns_md{l}")
            md_w = md.get_port(write_capable=True)
            md_r = md.get_port()
            self.specials += md, md_w, md_r
            mdmems.append(md_r)
            da = Array(dacc[l])
            u_m = Signal(32, name=f"dns_um{l}")
            self.sync += u_m.eq(Mux(zero_r, 0, fsel(inbufs[l].dat_r, s1_jj)))
            self.comb += [
                mul_dec[l].a.eq(u_m),
                mul_dec[l].b.eq(u_dec),
                mul_dec[l].valid.eq(u_v),
                md_w.adr.eq(t2_mda), md_w.dat_w.eq(mul_dec[l].result), md_w.we.eq(t2_v),
                mul_k[l].a.eq(mul_dec[l].result),
                mul_k[l].b.eq(t2_k),
                mul_k[l].valid.eq(t2_v),
                add_d[l].a.eq(Mux(t3_first, 0, da[t3_r])),
                add_d[l].b.eq(mul_k[l].result),
                add_d[l].valid.eq(t3_v),
            ]
            self.sync += If(t4_v, da[t4_r].eq(add_d[l].result))

        # Delta (shared): delta = (v - d)*beta over rows in natural order --------------------
        dl_run = Signal()
        dl_idx = Signal(max=rows)
        dl_slot = dl_head[:sb]
        self.comb += v_r.adr.eq(Cat(dl_idx[3:], dl_slot))
        dl_issue = Signal()
        self.comb += dl_issue.eq(dl_run)
        snap_d = []
        for r in range(R):
            for l in range(L):
                snap_d.append(add_d[l].result if r == R - 1 else dacc[l][r])
        self.sync += If(p1_capture,
            *[dsnap[i].eq(snap_d[i]) for i in range(rows)],
        ).Elif(dl_issue,
            *[dsnap[i].eq(dsnap[i + 1]) for i in range(rows - 1)],
        )
        self.sync += If(self.start,
            p1_done.eq(0), dl_run.eq(0), dl_head.eq(0),
        ).Else(
            If(p1_capture, p1_done.eq(p1_done + 1)),
            If(~dl_run,
                If((p1_done > dl_head) & (dl_head < p2_head + 2), dl_run.eq(1), dl_idx.eq(0)),
            ).Else(
                dl_idx.eq(dl_idx + 1),
                If(dl_idx == rows - 1, dl_run.eq(0), dl_head.eq(dl_head + 1)),
            )
        )
        e1_v, e1_a, e1_b, e1_last = Signal(), Signal(32), Signal(32), Signal()
        e1_slot, e1_bank = Signal(sb), Signal()
        self.sync += [
            e1_v.eq(dl_issue),
            e1_a.eq(fsel(v_r.dat_r, dl_idx[:3])),
            e1_b.eq(negate(dsnap[0])),
            e1_last.eq(dl_idx == rows - 1),
            e1_slot.eq(dl_slot), e1_bank.eq(dl_head[0]),
        ]
        e2_v, e2_last = dly(e1_v, La, "e2v"), dly(e1_last, La, "e2l")
        e2_slot, e2_bank = dly(e1_slot, La, "e2s"), dly(e1_bank, La, "e2b")
        e3_v, e3_last, e3_bank = dly(e2_v, Lm, "e3v"), dly(e2_last, Lm, "e3l"), dly(e2_bank, Lm, "e3b")
        self.comb += [
            add_v.a.eq(e1_a), add_v.b.eq(e1_b), add_v.valid.eq(e1_v),
            mul_b.a.eq(add_v.result), mul_b.b.eq(bet[e2_slot]), mul_b.valid.eq(e2_v),
        ]
        for b in range(2):
            self.sync += If(e3_v & (e3_bank == b),
                *[dreg[b][i].eq(dreg[b][i + 1]) for i in range(rows - 1)],
                dreg[b][rows - 1].eq(mul_b.result),
            )
        self.sync += If(self.start,
            dl_done.eq(0),
        ).Elif(e3_v & e3_last,
            dl_done.eq(dl_done + 1),
        )

        # P2: rank-1 update, q dot -------------------------------------------------------------
        p2_run = Signal()
        p2_g, p2_jj, p2_r = Signal(gb), Signal(3), Signal(rb)
        p2_grp = Signal()        # Output half of the current column group.
        half_busy = Array(Signal(name=f"dns_hbusy{i}") for i in range(2))
        half_full = Array(Signal(name=f"dns_hfull{i}") for i in range(2))
        grp_first = Signal()
        p2_issue = Signal()
        p2_slot = p2_head[:sb]
        self.comb += [
            grp_first.eq((p2_jj == 0) & (p2_r == 0)),
            p2_issue.eq(p2_run & (~grp_first | ~half_busy[p2_grp])),
            k_r2.adr.eq(Cat(p2_g, p2_slot)),
            q_r.adr.eq(Cat(p2_g, p2_slot)),
        ]
        for md_r in mdmems:
            self.comb += md_r.adr.eq(Cat(p2_r, p2_jj, p2_g, p2_head[:2]))
        self.sync += If(self.start,
            p2_run.eq(0), p2_head.eq(0), p2_grp.eq(0),
        ).Elif(~p2_run,
            If(run & (p2_head < heads) & (dl_done > p2_head) & (p2_head < sc_head + 2),
                p2_run.eq(1), p2_g.eq(0), p2_jj.eq(0), p2_r.eq(0),
            )
        ).Elif(p2_issue,
            p2_r.eq(p2_r + 1),
            If(p2_r == R - 1,
                p2_jj.eq(p2_jj + 1),
                If(p2_jj == 7,
                    p2_grp.eq(~p2_grp),
                    p2_g.eq(p2_g + 1),
                    If(p2_g == G - 1,
                        p2_run.eq(0),
                        p2_head.eq(p2_head + 1),
                    )
                )
            )
        )
        f1_v, f1_first, f1_last, f1_glast = Signal(), Signal(), Signal(), Signal()
        f1_r, f1_jj, f1_half, f1_bank = Signal(rb), Signal(3), Signal(), Signal()
        f1_k, f1_q = Signal(32), Signal(32)
        self.sync += [
            f1_v.eq(p2_issue),
            f1_first.eq((p2_g == 0) & (p2_jj == 0)),
            f1_last.eq((p2_g == G - 1) & (p2_jj == 7) & (p2_r == R - 1)),
            f1_glast.eq((p2_jj == 7) & (p2_r == R - 1)),
            f1_r.eq(p2_r), f1_jj.eq(p2_jj), f1_half.eq(p2_grp), f1_bank.eq(p2_head[0]),
            f1_k.eq(fsel(k_r2.dat_r, p2_jj)),
            f1_q.eq(fsel(q_r.dat_r, p2_jj)),
        ]
        g2_v = dly(f1_v, Lm, "g2v")
        g3_v, g3_r, g3_jj = dly(g2_v, La, "g3v"), dly(f1_r, Lm + La, "g3r"), dly(f1_jj, Lm + La, "g3jj")
        g3_half, g3_glast = dly(f1_half, Lm + La, "g3h"), dly(f1_glast, Lm + La, "g3gl")
        g3_q = dly(f1_q, Lm + La, "g3q")
        g4_v, g4_first, g4_r = dly(g3_v, Lm, "g4v"), dly(f1_first, 2*Lm + La, "g4f"), dly(g3_r, Lm, "g4r")
        assert La >= 1
        g5_ve, g5_r = dly(g4_v, La - 1, "g5ve"), dly(g4_r, La, "g5r")
        g5_laste = dly(f1_last, 2*Lm + 2*La - 1, "g5le")
        g5_banke = dly(f1_bank, 2*Lm + 2*La - 1, "g5be")
        g5_v, g5_last, g5_bank = dly(g5_ve, 1, "g5v"), dly(g5_laste, 1, "g5l"), dly(g5_banke, 1, "g5b")
        p2_capture = Signal()
        self.comb += p2_capture.eq(g5_v & g5_last)
        # The snapshot capture enables 2 x rows x 32 flops spread over the unit (v27: an 8.8 ns
        # route from g5l): one register per bank, decided a stage early, replicated by Vivado.
        cap = [Signal(name=f"dns_cap{b}", attr={("max_fanout", 128)}) for b in range(2)]
        self.sync += [cap[b].eq(g5_ve & g5_laste & (g5_banke == b)) for b in range(2)]

        outbufs = []
        for l in range(L):
            ob = Memory(256, 2*R, name=f"dns_ob{l}")
            ob_w = ob.get_port(write_capable=True, we_granularity=32)
            ob_r = ob.get_port()
            self.specials += ob, ob_w, ob_r
            outbufs.append(ob_r)
            f1_delta = Signal(32, name=f"dns_f1delta{l}")
            self.sync += f1_delta.eq(Mux(p2_head[0],
                Array(dreg[1][r*L + l] for r in range(R))[p2_r],
                Array(dreg[0][r*L + l] for r in range(R))[p2_r]))
            md_d = dly(mdmems[l].dat_r, Lm, f"dns_mdd{l}")
            oa = Array(oacc[l])
            self.comb += [
                mul_dk[l].a.eq(f1_delta), mul_dk[l].b.eq(f1_k), mul_dk[l].valid.eq(f1_v),
                add_m[l].a.eq(md_d), add_m[l].b.eq(mul_dk[l].result), add_m[l].valid.eq(g2_v),
                ob_w.adr.eq(Cat(g3_r, g3_half)),
                ob_w.dat_w.eq(Replicate(add_m[l].result, 8)),
                ob_w.we.eq(Mux(g3_v, 1 << g3_jj, 0)),
                mul_q[l].a.eq(add_m[l].result), mul_q[l].b.eq(g3_q), mul_q[l].valid.eq(g3_v),
                add_o[l].a.eq(Mux(g4_first, 0, oa[g4_r])),
                add_o[l].b.eq(mul_q[l].result),
                add_o[l].valid.eq(g4_v),
            ]
            self.sync += If(g5_v, oa[g5_r].eq(add_o[l].result))
        snap_o = []
        for r in range(R):
            for l in range(L):
                snap_o.append(add_o[l].result if r == R - 1 else oacc[l][r])
        self.sync += If(self.start,
            p2_done.eq(0),
        ).Elif(p2_capture,
            p2_done.eq(p2_done + 1),
        )

        # Emission: completed output halves -> per-lane source FIFOs -----------------------------
        em_run, em_half, em_i, em_push = Signal(), Signal(), Signal(rb), Signal()
        fifos = []
        for l in range(L):
            f = stream.SyncFIFO([("data", 256)], fifo_depth)
            setattr(self.submodules, f"dns_src{l}", f)
            fifos.append(f)
            self.comb += [
                f.sink.valid.eq(em_push), f.sink.data.eq(outbufs[l].dat_r),
                f.source.connect(self.sources[l]),
                outbufs[l].adr.eq(Cat(em_i, em_half)),
            ]
        room = reduce(and_, [f.level < fifo_depth - 2 for f in fifos])
        em_issue = Signal()
        self.comb += em_issue.eq(em_run & room)
        self.sync += [
            em_push.eq(em_issue),
            If(self.start,
                em_run.eq(0), em_half.eq(0), em_cnt.eq(0),
                half_busy[0].eq(0), half_busy[1].eq(0), half_full[0].eq(0), half_full[1].eq(0),
            ).Else(
                If(p2_issue & grp_first, half_busy[p2_grp].eq(1)),
                If(g3_v & g3_glast, half_full[g3_half].eq(1)),
                If(~em_run,
                    If(half_full[em_half], em_run.eq(1), em_i.eq(0)),
                ).Elif(em_issue,
                    em_i.eq(em_i + 1),
                    If(em_i == R - 1,
                        em_run.eq(0),
                        half_full[em_half].eq(0),
                        half_busy[em_half].eq(0),
                        em_half.eq(~em_half),
                        em_cnt.eq(em_cnt + 1),
                    )
                )
            )
        ]

        # Scale (shared) and the output writes ---------------------------------------------------
        sc_run = Signal()
        sc_idx = Signal(max=rows)
        sc_slot = sc_head[:sb]
        ofifo = stream.SyncFIFO([("data", 256), ("adr", adr_bits), ("fin", 1)], 16)
        self.submodules.dns_out_fifo = ofifo
        sc_issue = Signal()
        self.comb += sc_issue.eq(sc_run & (ofifo.level < 12))
        for b in range(2):
            self.sync += If(cap[b],
                *[osnap[b][i].eq(snap_o[i]) for i in range(rows)],
            ).Elif(sc_issue & (sc_head[0] == b),
                *[osnap[b][i].eq(osnap[b][i + 1]) for i in range(rows - 1)],
            )
        self.sync += If(self.start,
            sc_run.eq(0), sc_head.eq(0),
        ).Elif(~sc_run,
            If(p2_done > sc_head, sc_run.eq(1), sc_idx.eq(0)),
        ).Elif(sc_issue,
            sc_idx.eq(sc_idx + 1),
            If(sc_idx == rows - 1, sc_run.eq(0), sc_head.eq(sc_head + 1)),
        )
        h1_v, h1_a, h1_b, h1_i, h1_adr, h1_last = (Signal(), Signal(32), Signal(32), Signal(3),
                                                  Signal(adr_bits), Signal())
        self.sync += [
            h1_v.eq(sc_issue),
            h1_a.eq(Mux(sc_head[0], osnap[1][0], osnap[0][0])),
            h1_b.eq(scl[sc_slot]),
            h1_i.eq(sc_idx[:3]),
            h1_adr.eq(oad[sc_slot] + (sc_idx >> 3)),
            h1_last.eq(sc_idx == rows - 1),
        ]
        h2_v, h2_i = dly(h1_v, Lm, "h2v"), dly(h1_i, Lm, "h2i")
        h2_adr, h2_last = dly(h1_adr, Lm, "h2a"), dly(h1_last, Lm, "h2l")
        self.comb += [mul_s.a.eq(h1_a), mul_s.b.eq(h1_b), mul_s.valid.eq(h1_v)]
        pack = [Signal(32, name=f"dns_pack{i}") for i in range(7)]
        self.sync += If(h2_v, Case(h2_i, {i: pack[i].eq(mul_s.result) for i in range(7)}))
        self.comb += [
            ofifo.sink.valid.eq(h2_v & (h2_i == 7)),
            ofifo.sink.data.eq(Cat(*pack, mul_s.result)),
            ofifo.sink.adr.eq(h2_adr),
            ofifo.sink.fin.eq(h2_last),
            self.sp_wr_req.eq(ofifo.source.valid),
            self.sp_wr_adr.eq(ofifo.source.adr),
            self.sp_wr_we.eq(0xff),
            self.sp_wr_dat.eq(ofifo.source.data),
            ofifo.source.ready.eq(self.sp_wr_gnt),
        ]
        self.sync += If(self.start,
            retired.eq(0),
        ).Elif(ofifo.source.valid & self.sp_wr_gnt & ofifo.source.fin,
            retired.eq(retired + 1),
        )

        # Command -----------------------------------------------------------------------------
        drained = reduce(and_, [~f.source.valid for f in fifos])
        finish = Signal()
        self.comb += [
            finish.eq(run & (retired == heads) & (em_cnt == (heads << gb)) & drained),
            self.done.eq(finish),
            self.busy.eq(run),
            self.heads_done.eq(retired),
        ]
        # Progress counters, for tests and a stall report.
        self.progress = [ld_head, p1_head, p1_done, dl_head, dl_done, p2_head, p2_done,
                         sc_head, retired, em_cnt]
        self.sync += If(self.start,
            run.eq(1), heads.eq(self.heads), table.eq(self.table),
        ).Elif(finish,
            run.eq(0),
        )

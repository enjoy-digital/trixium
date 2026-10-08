#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Resident DeltaNet causal convolution over the shared scratchpad (sequencer unit 13).

For `channels` channels (a multiple of 8), bit-exact with hybrid.c's 4-tap conv:

  y[c] = (((+0 + w[4c]*x0[c]) + w[4c+1]*x1[c]) + w[4c+2]*x2[c]) + w[4c+3]*x3[c]

op 1 (ADD) is the residual add instead: y[c] = x0[c] + x3[c] (one exact add; x1, x2, w
unused), dst == x0 or x3 in place.

x0, x1, x2 are the window slots oldest to newest, x3 the new input (the QKV projection),
w the per-channel taps interleaved as in the model (4 FP32 per channel). The SiLU that
follows is unit 11 with a unit multiplier. Every operation is exact IEEE binary32
(trixium/units/fp32.py): no fixups.

Eight channels need eight scratchpad beats (x0..x3 and four beats of taps) and produce
one; one channel per clock matches that, so the datapath is a single lane (four
multipliers, the leading +0 add as a sign fix, three adders) fed from double-buffered
gather registers. y may overwrite x3 in place (each output beat is written after its own
input beats were read); every other overlap is rejected.

Integrator contract and error bits as for the other resident clients. The command
(CONV_COMMAND) is the firmware's descriptor layout.
"""

from functools import reduce
from operator import or_

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import LiteXModule

from litex.soc.interconnect import stream

from trixium.units.fp32 import FP32Add, FP32Mul

# Constants ----------------------------------------------------------------------------------------

CONV_COMMAND = [("op", 1), ("x0", 24), ("x1", 24), ("x2", 24), ("x3", 24), ("w", 24),
                ("dst", 24), ("channels", 16), ("protect_start", 24), ("protect_end", 24)]
OP_CONV, OP_ADD = 0, 1

# Helpers ------------------------------------------------------------------------------------------

def _delay(module, value, cycles, name):
    prev = value
    for i in range(cycles):
        r = Signal.like(value, name=f"{name}{i}")
        module.sync += r.eq(prev)
        prev = r
    return prev

# Conv Scratchpad ----------------------------------------------------------------------------------

class ConvScratchpad(LiteXModule):
    """Resident DeltaNet causal convolution / residual add (see the module docstring)."""
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client "
                             "address space")
        # Command / status.
        self.cmd = stream.Endpoint(CONV_COMMAND)
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
        cap = {name: Signal(width, name=f"cap_{name}") for name, width in CONV_COMMAND}
        groups = Signal(13)
        vbytes, wbytes = Signal(24), Signal(24)
        names = ("x0", "x1", "x2", "x3", "w", "dst")
        add = cap["op"] == OP_ADD
        used = {"x0": C(1, 1), "x1": ~add, "x2": ~add, "x3": C(1, 1), "w": ~add, "dst": C(1, 1)}
        ends = {k: Signal(64, name=f"end_{k}") for k in names}
        geometry_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))

        def disjoint(a, b):
            return (cap[a] >= ends[b]) | (cap[b] >= ends[a])

        self.comb += [
            groups.eq(cap["channels"][3:16]),
            vbytes.eq(cap["channels"] << 2), wbytes.eq(cap["channels"] << 4),
            *[ends[k].eq(cap[k] + (wbytes if k == "w" else vbytes)) for k in names],
            geometry_bad.eq((cap["channels"] == 0) | (cap["channels"][:3] != 0) |
                            (cap["protect_start"] > cap["protect_end"]) |
                            reduce(or_, [used[k] & (cap[k][:5] != 0) for k in names])),
            bounds_bad.eq(reduce(or_, [used[k] & (ends[k] > scratchpad_size) for k in names]) |
                          (cap["protect_end"] > scratchpad_size)),
            overlap_bad.eq(Mux(add,
                (~disjoint("dst", "x0") & (cap["dst"] != cap["x0"])) |
                (~disjoint("dst", "x3") & (cap["dst"] != cap["x3"])),
                ~disjoint("dst", "x0") | ~disjoint("dst", "x1") | ~disjoint("dst", "x2") |
                ~disjoint("dst", "w") |
                (~disjoint("dst", "x3") & (cap["dst"] != cap["x3"])))),
            protected_bad.eq((cap["protect_start"] != cap["protect_end"]) &
                             (cap["dst"] < cap["protect_end"]) &
                             (cap["protect_start"] < ends["dst"])),
        ]

        # Read side: 8 beats per group (x0, x1, x2, x3, w0..w3) into one of two gather banks.
        grp = Signal(13)                            # group being read
        sub = Signal(3)                             # beat within the group
        bank_full = Array([Signal(name="bank_full0"), Signal(name="bank_full1")])
        rbank = Signal()                            # bank being filled
        rd_pending, rd_sub, rd_bank = Signal(), Signal(3), Signal()
        banks = [[Signal(256, name=f"gb{b}_{i}") for i in range(8)] for b in range(2)]

        states = ["IDLE", "VALIDATE", "RUN", "FINISH", "FAILED"]
        S = {n: i for i, n in enumerate(states)}
        state = Signal(3)
        clear = Signal()
        self.comb += [
            clear.eq(self.cancel | (state != S["RUN"])),
            self.cmd.ready.eq((state == S["IDLE"]) & ~self.cancel),
            self.busy.eq(state != S["IDLE"]),
        ]
        issue = Signal()
        radr = Signal(13)
        self.comb += [
            issue.eq((state == S["RUN"]) & ~self.cancel & (grp < groups) & ~bank_full[rbank]),
            Case(sub, {
                0: radr.eq((cap["x0"] >> 5) + grp), 1: radr.eq((cap["x1"] >> 5) + grp),
                2: radr.eq((cap["x2"] >> 5) + grp), 3: radr.eq((cap["x3"] >> 5) + grp),
                "default": radr.eq((cap["w"] >> 5) + (grp << 2) + (sub - 4)),
            }),
            self.sp_rd_req.eq(issue), self.sp_rd_adr.eq(radr),
        ]
        self.sync += [
            rd_pending.eq(issue), rd_sub.eq(sub), rd_bank.eq(rbank),
            If(clear, grp.eq(0), sub.eq(0), rbank.eq(0),
            ).Elif(issue,
                self.read_beats.eq(self.read_beats + 1),
                sub.eq(Mux(add, Mux(sub == 0, 3, 0), sub + 1)),
                If(Mux(add, sub == 3, sub == 7), grp.eq(grp + 1), rbank.eq(~rbank)),
            ),
        ]
        for b in range(2):
            for i in range(8):
                self.sync += If(rd_pending & (rd_bank == b) & (rd_sub == i),
                    banks[b][i].eq(self.sp_rd_dat))

        # Compute side: one channel per clock from the full bank.
        cbank = Signal()
        lane = Signal(3)
        cgrp = Signal(13)
        feed = Signal()
        out_room = Signal()
        FIFO_DEPTH = 8
        self.submodules.ofifo = ofifo = ResetInserter()(SyncFIFO(width=256, depth=FIFO_DEPTH))
        groups_in_flight = Signal(4)                 # groups fed but not yet in the FIFO
        self.comb += [
            ofifo.reset.eq(clear),
            out_room.eq((ofifo.level + groups_in_flight) < FIFO_DEPTH),
            feed.eq((state == S["RUN"]) & bank_full[cbank] & ~self.cancel &
                    (out_room | (lane != 0))),
        ]
        # ADD takes a whole group (beat) per clock through eight adders of its own - reads
        # then pace it (x0 and x3, two beats per group): one lane at a time, the residual adds
        # cost ~4.2k clocks per 5,120 floats.
        last_lane = Signal()
        self.comb += last_lane.eq(add | (lane == 7))
        # A bank is full once its 8th beat has landed; it empties when its last lane is fed.
        for b in range(2):
            self.sync += If(clear, bank_full[b].eq(0)).Else(
                If(rd_pending & (rd_bank == b) & (rd_sub == Mux(add, 3, 7)), bank_full[b].eq(1)),
                If(feed & (cbank == b) & last_lane, bank_full[b].eq(0)))
        self.sync += If(clear, cbank.eq(0), lane.eq(0), cgrp.eq(0)).Elif(feed,
            lane.eq(Mux(add, 0, lane + 1)),
            If(last_lane, cbank.eq(~cbank), cgrp.eq(cgrp + 1)))

        bank_sel = [Array([banks[0][i], banks[1][i]])[cbank] for i in range(8)]

        def lane_word(beat, idx):
            return Array([beat[32*j:32*j + 32] for j in range(8)])[idx]
        xs = [lane_word(bank_sel[i], lane) for i in range(4)]
        # Tap t of channel `lane` (0..7) is float 4*lane + t of the 32 tap floats in beats 4..7.
        taps = []
        wflat = Cat(*bank_sel[4:8])
        wfl = Array([wflat[32*j:32*j + 32] for j in range(32)])
        for t in range(4):
            taps.append(wfl[Cat(C(t, 2), lane)])
        muls = [FP32Mul() for _ in range(4)]
        adds = [FP32Add() for _ in range(3)]
        self.submodules += muls + adds
        L_M, L_A = FP32Mul.LATENCY, FP32Add.LATENCY
        for t in range(4):
            self.comb += [muls[t].a.eq(taps[t]), muls[t].b.eq(xs[t])]
        # a1 = +0 + p0: p0, except -0 -> +0.
        p0 = muls[0].r
        a1 = Signal(32)
        self.comb += a1.eq(Mux(p0 == 0x80000000, 0, p0))
        p1 = muls[1].r
        p2 = _delay(self, muls[2].r, L_A, "p2d")
        p3 = _delay(self, muls[3].r, 2*L_A, "p3d")
        self.comb += [
            adds[0].a.eq(a1), adds[0].b.eq(p1),
            adds[1].a.eq(adds[0].r), adds[1].b.eq(p2),
            adds[2].a.eq(adds[1].r), adds[2].b.eq(p3),
        ]
        LAT = L_M + 3*L_A
        y_valid = _delay(self, feed & ~add, LAT, "yv")
        y_lane = _delay(self, lane, LAT, "yl")
        acc = [Signal(32, name=f"yacc{j}") for j in range(7)]
        for j in range(7):
            self.sync += If(y_valid & (y_lane == j), acc[j].eq(adds[2].r))
        # ADD: x0 + x3, eight lanes at once.
        vadds = [FP32Add() for _ in range(8)]
        self.submodules += vadds
        for j in range(8):
            self.comb += [vadds[j].a.eq(bank_sel[0][32*j:32*j + 32]),
                          vadds[j].b.eq(bank_sel[3][32*j:32*j + 32])]
        v_valid = _delay(self, feed & add, L_A, "vv")
        self.comb += [
            ofifo.we.eq(((y_valid & (y_lane == 7)) | v_valid) & ~clear),
            ofifo.din.eq(Mux(add, Cat(*[v.r for v in vadds]), Cat(*acc, adds[2].r))),
        ]
        self.sync += If(clear, groups_in_flight.eq(0)).Else(
            groups_in_flight.eq(groups_in_flight + (feed & (lane == 0)) - ofifo.we))

        # Writes.
        wgrp = Signal(13)
        self.comb += [
            self.sp_wr_req.eq(ofifo.readable & ~self.cancel),
            self.sp_wr_adr.eq((cap["dst"] >> 5) + wgrp),
            self.sp_wr_dat.eq(ofifo.dout), self.sp_wr_we.eq(0xFF),
            ofifo.re.eq(self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear, wgrp.eq(0)).Elif(ofifo.re,
            wgrp.eq(wgrp + 1), self.output_beats.eq(self.output_beats + 1))

        # Control.
        self.sync += [
            self.done.eq(0),
            If(self.cancel, state.eq(S["IDLE"]), self.error.eq(0)).Else(Case(state, {
                S["IDLE"]: If(self.cmd.valid & self.cmd.ready,
                    *[v.eq(getattr(self.cmd, k)) for k, v in cap.items()],
                    self.error.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
                    state.eq(S["VALIDATE"])),
                S["VALIDATE"]: If(geometry_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3),
                                      protected_bad)),
                    state.eq(S["FAILED"]),
                ).Else(state.eq(S["RUN"])),
                S["RUN"]: If((wgrp == groups) & ~ofifo.readable, state.eq(S["FINISH"])),
                S["FINISH"]: [self.done.eq(1), state.eq(S["IDLE"])],
            })),
        ]

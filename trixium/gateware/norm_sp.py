# SPDX-License-Identifier: BSD-2-Clause
"""Resident RMSNorm / L2 norm over the shared scratchpad (future sequencer unit 12).

For each of `nseg` segments of `n` FP32 values at src + seg*n*4, bit-exact with hybrid.c:

  RMS (op 0):  ss = sum64(x*x); r = 1/sqrt(ss/n + eps);   dst[i] = (x[i]*r)*w[i]
  L2  (op 1):  ss = sum64(x*x); inv = 1/max(sqrt(ss), eps); dst[i] = (x[i]*inv)*1.0

sum64 is the canonical reduction of docs/canonical-reductions.md (64 partials, i % 64 in
index order, then a halving tree); every operation is an exact IEEE binary32 operation
(gateware/fp32_exact.py), so there are no fixups. `w` (RMS only, n values) is read once
per command into an internal buffer and shared by every segment. dst == src (in place)
or disjoint.

Per segment: pass 1 streams the segment through 8 multipliers and 8 adders into the 64
partials (each revisited every 8 beats, past the adder's latency), the tree and the
scalar chain run on one adder and the sequential divide/sqrt, pass 2 streams it again
through two multipliers per lane into a credit-counted output FIFO.

Contract with the integrator as for the other resident clients (one-cycle reads, writes
on sp_wr_req & sp_wr_gnt with the payload held). error bits: 0 geometry, 1 bounds,
2 forbidden overlap, 6 protected output; a failure stays busy until cancel.
"""
from migen import *
from migen.genlib.fifo import SyncFIFO
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.fp32_exact import FP32Add, FP32DivSqrt, FP32Mul, _lzc, _shift_left

NORM_COMMAND = [("op", 1), ("src", 24), ("dst", 24), ("weight", 24), ("n", 16), ("nseg", 16),
                ("eps", 32), ("protect_start", 24), ("protect_end", 24)]
OP_RMS, OP_L2 = 0, 1
WMAX = 8192                     # largest n (27B: n_embd 5120)
F32_ONE = 0x3F800000


def _delay(module, value, cycles, name):
    regs = [Signal.like(value, name=f"{name}{i}") for i in range(cycles)]
    prev = value
    for r in regs:
        module.sync += r.eq(prev)
        prev = r
    return prev


class BonsaiNormScratchpad(LiteXModule):
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client address space")
        self.cmd = stream.Endpoint(NORM_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.read_beats, self.output_beats = Signal(32), Signal(32)
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)

        cap = {name: Signal(width, name=f"cap_{name}") for name, width in NORM_COMMAND}
        rms = cap["op"] == OP_RMS
        nbeats = Signal(13)                          # beats per segment
        seg_bytes = Signal(24)
        total = Signal(64)                           # bytes of all segments
        ends = {k: Signal(64, name=f"end_{k}") for k in ("src", "dst", "weight")}
        geometry_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))
        self.comb += [
            nbeats.eq(cap["n"][3:16]),
            seg_bytes.eq(cap["n"] << 2),
            total.eq(seg_bytes*cap["nseg"]),
            ends["src"].eq(cap["src"] + total),
            ends["dst"].eq(cap["dst"] + total),
            ends["weight"].eq(cap["weight"] + seg_bytes),
            geometry_bad.eq((cap["n"][:3] != 0) | (cap["n"] == 0) | (cap["n"] > WMAX) |
                            (cap["nseg"] == 0) | (cap["src"][:5] != 0) | (cap["dst"][:5] != 0) |
                            (rms & (cap["weight"][:5] != 0)) |
                            (cap["protect_start"] > cap["protect_end"])),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                          (rms & (ends["weight"] > scratchpad_size)) |
                          (cap["protect_end"] > scratchpad_size)),
            overlap_bad.eq(
                ((cap["dst"] != cap["src"]) & (cap["dst"] < ends["src"]) & (cap["src"] < ends["dst"])) |
                (rms & (cap["dst"] < ends["weight"]) & (cap["weight"] < ends["dst"]))),
            protected_bad.eq((cap["protect_start"] != cap["protect_end"]) &
                             (cap["dst"] < cap["protect_end"]) & (cap["protect_start"] < ends["dst"])),
        ]

        # Arithmetic ------------------------------------------------------------------------
        L_MUL, L_ADD = FP32Mul.LATENCY, FP32Add.LATENCY
        sq = [FP32Mul() for _ in range(8)]           # pass 1: x*x
        acc = [FP32Add() for _ in range(8)]          # pass 1: partial + x*x
        m1 = [FP32Mul() for _ in range(8)]           # pass 2: x*scale
        m2 = [FP32Mul() for _ in range(8)]           # pass 2: (x*scale)*w
        tree = FP32Add()                             # tree and the scalar add
        self.ds = ds = FP32DivSqrt()
        self.submodules += sq + acc + m1 + m2 + [tree]

        # The 64 partials: P[8*slot + lane].
        P = [Signal(32, name=f"part{k}") for k in range(64)]
        Pa = Array(P)

        # Weight buffer: one 256-bit word per beat.
        wbuf = Memory(256, WMAX//8)
        wr_port = wbuf.get_port(write_capable=True)
        rd_port = wbuf.get_port()
        self.specials += wbuf, wr_port, rd_port

        # Output FIFO (pass 2) with credits for the in-flight beats.
        FIFO_DEPTH = 32
        self.submodules.ofifo = ofifo = ResetInserter()(SyncFIFO(width=256, depth=FIFO_DEPTH))
        P2_LAT = 1 + L_MUL + L_MUL                   # read -> m1 -> m2
        inflight = Signal(max=FIFO_DEPTH + 1)

        # Control ---------------------------------------------------------------------------
        states = ["IDLE", "VALIDATE", "WLOAD", "P1", "P1_DRAIN", "TREE", "SCALAR", "P2", "P2_DRAIN",
                  "FINISH", "FAILED"]
        S = {name: i for i, name in enumerate(states)}
        state = Signal(4)
        seg = Signal(16)
        beat = Signal(13)                            # reads issued in the current phase
        rd_pending = Signal()
        rd_kind = Signal(2)                          # 0 weight, 1 pass-1 x, 2 pass-2 x
        rd_beat = Signal(13)
        clear = Signal()
        self.comb += [
            clear.eq(self.cancel | (state == S["IDLE"])),
            ofifo.reset.eq(clear),
            self.cmd.ready.eq((state == S["IDLE"]) & ~self.cancel),
            self.busy.eq(state != S["IDLE"]),
        ]
        seg_src = Signal(24)
        seg_dst = Signal(24)
        self.comb += [
            seg_src.eq(cap["src"] + seg*seg_bytes),
            seg_dst.eq(cap["dst"] + seg*seg_bytes),
        ]

        # Read issue per phase.
        issue = Signal()
        can_p2 = Signal()
        self.comb += can_p2.eq((ofifo.level + inflight) < FIFO_DEPTH)
        self.comb += [
            issue.eq(~self.cancel & (
                ((state == S["WLOAD"]) & (beat < nbeats)) |
                ((state == S["P1"]) & (beat < nbeats)) |
                ((state == S["P2"]) & (beat < nbeats) & can_p2))),
            self.sp_rd_req.eq(issue),
            self.sp_rd_adr.eq(Mux(state == S["WLOAD"], (cap["weight"] >> 5) + beat,
                                  (seg_src >> 5) + beat)),
            rd_port.adr.eq(beat),
        ]
        self.sync += [
            rd_pending.eq(issue & ~clear),
            rd_kind.eq(Mux(state == S["WLOAD"], 0, Mux(state == S["P1"], 1, 2))),
            rd_beat.eq(beat),
            If(issue, beat.eq(beat + 1), self.read_beats.eq(self.read_beats + 1)),
        ]
        # Weight load writes.
        self.comb += [
            wr_port.we.eq(rd_pending & (rd_kind == 0)),
            wr_port.adr.eq(rd_beat),
            wr_port.dat_w.eq(self.sp_rd_dat),
        ]

        # Pass 1: squares, then accumulation into the partials of this beat's slot.
        p1_valid = Signal()
        self.comb += p1_valid.eq(rd_pending & (rd_kind == 1))
        slot_sq = _delay(self, rd_beat[:3], L_MUL, "slot_sq")
        valid_sq = _delay(self, p1_valid, L_MUL, "valid_sq")
        for lane in range(8):
            x = self.sp_rd_dat[32*lane:32*lane + 32]
            self.comb += [sq[lane].a.eq(x), sq[lane].b.eq(x),
                          acc[lane].a.eq(Array(P[8*s + lane] for s in range(8))[slot_sq]),
                          acc[lane].b.eq(sq[lane].r)]
        slot_acc = _delay(self, slot_sq, L_ADD, "slot_acc")
        valid_acc = _delay(self, valid_sq, L_ADD, "valid_acc")
        p1_inflight = Signal(8)
        self.sync += If(clear, p1_inflight.eq(0)).Else(
            p1_inflight.eq(p1_inflight + p1_valid - valid_acc))

        # Tree: level w (32..1), j in [0, w): P[j] = P[j] + P[j + w], one add per clock.
        level = Signal(3)                            # 0..5 -> w = 32 >> level
        w_cur = Signal(7)
        self.comb += w_cur.eq(Array([C(32 >> i, 7) for i in range(6)] + [C(0, 7)]*2)[level])
        tj = Signal(7)
        tree_issue = Signal()
        tree_valid = _delay(self, tree_issue, L_ADD, "tree_valid")
        tree_dst = _delay(self, tj[:6], L_ADD, "tree_dst")
        tree_inflight = Signal(4)
        self.comb += tree_issue.eq((state == S["TREE"]) & (tj < w_cur) & ~self.cancel)
        # Scalar chain: the tree adder also does ss/n + eps.
        sc_add = Signal()
        sc_step = Signal(3)
        sc_wait = Signal()
        ss, tmp, scale = Signal(32), Signal(32), Signal(32)
        sc_add_valid = _delay(self, sc_add, L_ADD, "sc_add_valid")
        n_f32 = Signal(32)
        nlz, nlz_s = _lzc(cap["n"], 16)
        self.comb += nlz_s
        nman, nman_c = _shift_left(cap["n"], nlz, 15, 16)
        self.comb += [nman_c, n_f32.eq(Cat(C(0, 8), nman[:15], (127 + 15) - nlz, C(0, 1)))]
        tree_a, tree_b = Signal(32), Signal(32)
        self.comb += [
            tree.a.eq(tree_a), tree.b.eq(tree_b),
            If(state == S["TREE"],
                tree_a.eq(Pa[tj[:6]]), tree_b.eq(Pa[tj[:6] + w_cur]),
            ).Else(
                tree_a.eq(tmp), tree_b.eq(cap["eps"]),
            ),
        ]
        partial_writes = []
        for k in range(64):
            lane, s = k % 8, k // 8
            partial_writes.append(
                If(clear | ((state == S["P1"]) & (beat == 0) & ~rd_pending),
                    P[k].eq(0),
                ).Elif(valid_acc & (slot_acc == s),
                    P[k].eq(acc[lane].r),
                ).Elif(tree_valid & (tree_dst == k),
                    P[k].eq(tree.r),
                ))
        self.sync += partial_writes
        self.sync += If(clear, tree_inflight.eq(0)).Else(
            tree_inflight.eq(tree_inflight + tree_issue - tree_valid))

        # Pass 2: m1 = x*scale, m2 = m1*w; w = 1.0 for L2.
        p2_valid = Signal()
        self.comb += p2_valid.eq(rd_pending & (rd_kind == 2))
        w_word = Signal(256)
        self.comb += w_word.eq(Mux(rms, rd_port.dat_r, Replicate(C(F32_ONE, 32), 8)))
        w_hold = _delay(self, w_word, L_MUL, "w_hold")
        for lane in range(8):
            self.comb += [
                m1[lane].a.eq(self.sp_rd_dat[32*lane:32*lane + 32]), m1[lane].b.eq(scale),
                m2[lane].a.eq(m1[lane].r), m2[lane].b.eq(w_hold[32*lane:32*lane + 32]),
            ]
        out_valid = _delay(self, p2_valid, 2*L_MUL, "out_valid")
        out_beat = _delay(self, rd_beat, 2*L_MUL, "out_beat")
        self.comb += [
            ofifo.we.eq(out_valid & ~clear),
            ofifo.din.eq(Cat(*[m2[lane].r for lane in range(8)])),
        ]
        self.sync += If(clear, inflight.eq(0)).Else(
            inflight.eq(inflight + (issue & (state == S["P2"])) - out_valid))
        # Output beats carry their own address: dst + segment offset + beat. The FIFO holds
        # only data; addresses follow the write counter within the segment.
        wr_beat = Signal(13)
        wr_seg_base = Signal(13)
        self.comb += [
            self.sp_wr_req.eq(ofifo.readable & ~self.cancel),
            self.sp_wr_adr.eq(wr_seg_base + wr_beat),
            self.sp_wr_dat.eq(ofifo.dout),
            self.sp_wr_we.eq(0xFF),
            ofifo.re.eq(self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear, wr_beat.eq(0)).Elif(ofifo.re,
            wr_beat.eq(wr_beat + 1), self.output_beats.eq(self.output_beats + 1))

        # Sequencing ------------------------------------------------------------------------
        ds_start = Signal()
        self.comb += ds.start.eq(ds_start)
        self.sync += [
            self.done.eq(0), ds_start.eq(0), sc_add.eq(0),
            If(self.cancel, state.eq(S["IDLE"]), self.error.eq(0)).Else(
            Case(state, {
                S["IDLE"]: If(self.cmd.valid & self.cmd.ready,
                    *[v.eq(getattr(self.cmd, k)) for k, v in cap.items()],
                    self.error.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
                    seg.eq(0), beat.eq(0),
                    state.eq(S["VALIDATE"])),
                S["VALIDATE"]: If(geometry_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(S["FAILED"]),
                ).Else(beat.eq(0), state.eq(Mux(rms, S["WLOAD"], S["P1"]))),
                S["WLOAD"]: If((beat == nbeats) & ~rd_pending, beat.eq(0), state.eq(S["P1"])),
                S["P1"]: If(beat == nbeats, state.eq(S["P1_DRAIN"])),
                S["P1_DRAIN"]: If((p1_inflight == 0) & ~rd_pending,
                    level.eq(0), tj.eq(0), state.eq(S["TREE"])),
                S["TREE"]: If(tree_issue,
                    tj.eq(tj + 1),
                ).Elif(tree_inflight == 0,
                    If(level == 5,
                        sc_step.eq(0), sc_wait.eq(0), state.eq(S["SCALAR"]),
                    ).Else(level.eq(level + 1), tj.eq(0)),
                ),
                # RMS: ss/n -> +eps -> sqrt -> 1/x.  L2: sqrt(ss) -> max(eps) -> 1/x.
                S["SCALAR"]: If(~sc_wait,
                    Case(sc_step, {
                        0: [ss.eq(P[0]),
                            If(rms, ds.a.eq(P[0]), ds.b.eq(n_f32), ds.op.eq(0), ds_start.eq(1), sc_wait.eq(1),
                            ).Else(ds.a.eq(P[0]), ds.op.eq(1), ds_start.eq(1), sc_wait.eq(1), sc_step.eq(2))],
                        1: [sc_add.eq(1), sc_wait.eq(1)],                       # tmp + eps
                        2: [ds.a.eq(tmp), ds.op.eq(1), ds_start.eq(1), sc_wait.eq(1)],   # sqrt
                        3: [ds.a.eq(C(F32_ONE, 32)), ds.b.eq(tmp), ds.op.eq(0), ds_start.eq(1), sc_wait.eq(1)],
                        4: [scale.eq(tmp), beat.eq(0), wr_beat.eq(0),
                            wr_seg_base.eq(seg_dst >> 5), state.eq(S["P2"])],
                    }),
                ).Else(
                    If(ds.done,
                        sc_wait.eq(0),
                        # L2 after the sqrt: nrm = max(sqrt, eps) (bit order = value order for
                        # the non-negative results; a NaN stays).
                        If(~rms & (sc_step == 2) & (ds.r < cap["eps"]),
                            tmp.eq(cap["eps"])).Else(tmp.eq(ds.r)),
                        sc_step.eq(Mux(~rms & (sc_step == 2), 3, sc_step + 1)),
                    ).Elif(sc_add_valid,
                        sc_wait.eq(0), tmp.eq(tree.r), sc_step.eq(sc_step + 1),
                    ),
                ),
                S["P2"]: If(beat == nbeats, state.eq(S["P2_DRAIN"])),
                S["P2_DRAIN"]: If((inflight == 0) & ~ofifo.readable & ~rd_pending,
                    If(seg + 1 == cap["nseg"],
                        state.eq(S["FINISH"]),
                    ).Else(seg.eq(seg + 1), beat.eq(0), state.eq(S["P1"]))),
                S["FINISH"]: [self.done.eq(1), state.eq(S["IDLE"])],
            })),
        ]

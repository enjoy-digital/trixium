#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiSequencer: hardware execution of descriptor lists (engine + vector-unit ops).

Op dispatch from the CPU costs ~1k cycles per op (CSR writes + completion polling) and the
engine/vector units serialize behind it - at N >= 16 channels dispatch dominates the token
time. The sequencer executes a scratchpad-resident descriptor list back-to-back: the CPU
builds the (mostly static) list once, patches the few data-dependent words, writes
base/count and one start, then polls a single completion.

Descriptor: 64 bytes = 2 scratchpad beats, 16 little-endian words.
  w0:  [3:0] unit: 0 = stop, 1 = engine op, 2 = vector-unit program.
       [5] (engine) capture k(absmax) into kreg[w0[7:6]] when this op completes.
       [8] (engine) ALSO accumulate that k into kreg[3] (kreg3 += k) - lets one register
           collect a SUM of shifts across ops (silu's eg+eu term) while [7:6] keeps each
           op's own k addressable. Firmware zeroes kreg3 by capturing with [7:6]=3 first.
       (vec) w1[31:16]: per-arg 2-bit kreg-add select (arg i at [16+2i +: 2]): 0 = none,
           n = add kreg[n-1] to the arg AFTER its 2-bit source resolves. Turns
           "imm + earlier k" - the FFN exponent algebra that forced a CPU round-trip per
           op - into descriptor-resident arithmetic. kreg3 is reachable via capture only.
       [4] seed S7 from the ENGINE ABSMAX instead of the previous program's res6. Used by
           the absmax spill once cmd.xred does the cross-channel sum in hardware and the
           value stops flowing through a vector program's res6.
       [31:16] vec arg sources, 2 bits each (arg i at [16+2i +: 2]):
               0 = immediate (w2+i), 1 = k(engine absmax) = max(0, bitlen(absmax) - 14),
               2 = rpre(prev vec res6) = bitlen(res6) + 5 (bitexact.axpy_rpre),
               3 = k(prev vec res6) = max(0, bitlen(res6) - 14) (norm shift from a
                   reduce_acc absmax result).
  engine: w1 = kind | ilv<<8 | supw<<9 | rshift_rpre<<10 (rshift taken from
          rpre(prev vec res6) instead of w7) | qn<<11 (dual-Q GQA)
          | xred<<12 (AXPY: sum the per-channel partials in hardware);
          w2 = rows | rem<<16;  w3 = cols;  w4 = wbase;  w5 = wlen;
          w6 = sbeats | cbeats<<8 | qstride<<16 (head-B result offset, words);
          w7 = bias | rshift<<16 | rshift2<<22 (head-B AXPY pre-shift);
          w8 = xsrc;  w9 = dst;  w10 = chan.
  vec:    w1 = pc;  w2..w9 = args 0..7.

Numerics are UNCHANGED - sequencing is transparent; the golden mirrors need no update.

Scratchpad access rides the engine fill port (never a fifth RAM port): descriptors are
fetched only while both units are idle, so the integrator grants the port as
vec.busy > seq.sp_active > engine."""

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

# BonsaiSequencer ----------------------------------------------------------------------------------

class BonsaiSequencer(LiteXModule):
    def __init__(self):
        self.base   = CSRStorage(24, description="Descriptor list byte address (32B aligned).")
        self.count  = CSRStorage(16, description="Descriptor count (stop descriptor also ends).")
        self.start  = CSRStorage(1,  description="Write 1 to run the list.")
        self.status = CSRStatus(fields=[
            CSRField("busy",   size=1),
            CSRField("dcount", size=16, offset=8, description="Descriptors completed."),
        ])

        # Engine command side (muxed into engine.cmd by BonsaiEngineCSR while busy).
        self.eng_valid  = Signal()
        self.eng_ready  = Signal()
        self.eng_kind   = Signal(3)
        self.eng_rows   = Signal(16)
        self.eng_cols   = Signal(14)
        self.eng_wbase  = Signal(32)
        self.eng_wlen   = Signal(24)
        self.eng_sbeats = Signal(4)
        self.eng_cbeats = Signal(8)
        self.eng_bias   = Signal(7)
        self.eng_rshift = Signal(6)
        self.eng_xsrc   = Signal(24)
        self.eng_dst    = Signal(24)
        self.eng_chan   = Signal(5)
        self.eng_rem    = Signal(5)
        self.eng_ilv    = Signal()
        self.eng_supw   = Signal()
        self.eng_qn      = Signal()
        self.eng_xred   = Signal()  # AXPY cross-channel reduce (engine sums the partials).
        self.eng_qstride = Signal(16)
        self.eng_rshift2 = Signal(6)
        self.eng_done   = Signal()   # From the engine (pulse).
        self.absmax     = Signal(32) # From the engine tracker.

        # Vector-unit direct-drive side.
        self.vec_start = Signal()    # Pulse.
        self.vec_pc    = Signal(16)
        self.vec_args  = [Signal(32, name=f"seq_vec_arg{i}") for i in range(8)]
        self.vec_done  = Signal()    # From the vector unit (pulse).
        self.vec_res6  = Signal(48)  # Vector unit S6 result (registered at DONE).
        self.vec_seed7 = Signal(48)  # S7 seed for the next program: res6, or the engine
                                     # absmax when w0[4] is set (see below).

        # Scratchpad read (through the engine's fill-port mux).
        self.sp_adr    = Signal(13)
        self.sp_dat    = Signal(256)
        self.sp_active = Signal()

        self.busy = Signal()

        # # #

        dptr   = Signal(13) # Beat index of the current descriptor.
        left   = Signal(16)
        dcount = Signal(16)
        d0     = Signal(256)
        d1     = Signal(256)

        def w(beat, i): # Word i of a captured beat.
            return (d0 if beat == 0 else d1)[32*(i % 8):32*(i % 8) + 32]

        unit    = Signal(4)
        argsrc  = Signal(16)
        self.comb += [
            unit.eq(w(0, 0)[0:4]),
            argsrc.eq(w(0, 0)[16:32]),
        ]

        # Special arg sources (pure bit-length functions, matching the firmware/bitexact).
        def bitlen(sig):
            n   = len(sig)
            res = Signal(7)
            cases = If(sig[n-1], res.eq(n))
            for b in range(n - 2, -1, -1):
                cases = cases.Elif(sig[b], res.eq(b + 1))
            cases = cases.Else(res.eq(0))
            self.comb += cases
            return res

        absmax_bl = bitlen(self.absmax)
        res6_lo   = Signal(48)
        self.comb += res6_lo.eq(self.vec_res6)
        res6_bl   = bitlen(res6_lo)

        k_absmax = Signal(32)
        rpre_v   = Signal(32)
        k_res6   = Signal(32)
        self.comb += [
            k_absmax.eq(Mux(absmax_bl > 14, absmax_bl - 14, 0)),
            rpre_v.eq(res6_bl + 5),
            k_res6.eq(Mux(res6_bl > 14, res6_bl - 14, 0)),
        ]

        # S7 seed select (w0[4]). Normally the previous program's res6, which is how a list
        # chains a hardware result into the next program. With cmd.xred the engine does the
        # cross-channel sum itself, so the value the CPU needs afterwards - the absmax of the
        # summed vector - lives in the engine tracker and no longer passes through a vector
        # program's res6. Seeding S7 from it lets the existing spill program persist it
        # unchanged, and because spill ends with SMOV s6, s7 the value also lands in res6, so
        # a following norm_i32 taking k from KRES6 sees exactly what the reduce_acc chain
        # used to give it. That is what keeps the numerics bit-identical across the switch.
        self.comb += self.vec_seed7.eq(Mux(w(0, 0)[4], self.absmax, self.vec_res6))

        # Exponent register file. The FFN chain needs "per-layer constant + an EARLIER op's
        # k" as vec args (eg = ex8 - bias + rshift + kg; silu needs both eg and eu); the
        # KABS/KRES6 sources only expose the LATEST k, so every such arg forced a CPU
        # round-trip - measured as 43% of the GEN-HW window in 200-5k-cycle gaps.
        self.kregs = kregs = Array([Signal(32, name=f"kreg{i}") for i in range(4)])
        # Gated on self.busy, not an FSM state: eng_done is wired from engine.done
        # unconditionally, so it also pulses for CPU-dispatched ops - but those never run
        # while the sequencer is busy (csr_front muxes the command interface to the
        # sequencer for the whole list), so busy & eng_done is exactly "our engine
        # descriptor completed" and d0 still holds its words.
        self.sync += If(self.busy & self.eng_done,
            If(w(0, 0)[5], kregs[w(0, 0)[6:8]].eq(k_absmax)),
            If(w(0, 0)[8], kregs[3].eq(kregs[3] + k_absmax)),
        )

        for i in range(8):
            src  = Signal(2)
            imm  = Signal(32)
            base = Signal(32)
            kadd = Signal(2)
            self.comb += [
                src.eq(argsrc[2*i:2*i+2]),
                imm.eq(w(0, 2 + i) if (2 + i) < 8 else w(1, 2 + i)),
                Case(src, {
                    0: base.eq(imm),
                    1: base.eq(k_absmax),
                    2: base.eq(rpre_v),
                    3: base.eq(k_res6),
                }),
                kadd.eq(w(0, 1)[16 + 2*i:18 + 2*i]),
                self.vec_args[i].eq(base + Mux(kadd == 0, 0, kregs[kadd - 1])),
            ]

        # Engine fields straight from the descriptor.
        self.comb += [
            self.eng_kind.eq(w(0, 1)[0:3]),
            self.eng_ilv.eq(w(0, 1)[8]),
            self.eng_supw.eq(w(0, 1)[9]),
            self.eng_rows.eq(w(0, 2)[0:16]),
            self.eng_rem.eq(w(0, 2)[16:21]),
            self.eng_cols.eq(w(0, 3)[0:14]),
            self.eng_wbase.eq(w(0, 4)),
            self.eng_wlen.eq(w(0, 5)[0:24]),
            self.eng_sbeats.eq(w(0, 6)[0:4]),
            self.eng_cbeats.eq(w(0, 6)[8:16]),
            self.eng_bias.eq(w(0, 7)[0:7]),
            self.eng_rshift.eq(Mux(w(0, 1)[10], rpre_v[0:6], w(0, 7)[16:22])),
            self.eng_qn.eq(w(0, 1)[11]),
            self.eng_xred.eq(w(0, 1)[12]),
            self.eng_qstride.eq(w(0, 6)[16:32]),
            self.eng_rshift2.eq(w(0, 7)[22:28]),
            self.eng_xsrc.eq(w(1, 8)[0:24]),
            self.eng_dst.eq(w(1, 9)[0:24]),
            self.eng_chan.eq(w(1, 10)[0:5]),
            self.vec_pc.eq(w(0, 1)[0:16]),
        ]

        self.comb += [
            self.status.fields.busy.eq(self.busy),
            self.status.fields.dcount.eq(dcount),
        ]

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.start.re & self.start.storage,
                NextValue(dptr, self.base.storage[5:]),
                NextValue(left, self.count.storage),
                NextValue(dcount, 0),
                NextState("FETCH0"),
            )
        )
        fsm.act("FETCH0",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr),
            If(left == 0,
                NextState("IDLE"),
            ).Else(
                NextState("FETCH0_L"),
            )
        )
        fsm.act("FETCH0_L", # dat valid this cycle (address held from FETCH0's registered mux path).
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr), # Hold.
            NextValue(d0, self.sp_dat),
            NextState("FETCH1"),
        )
        fsm.act("FETCH1",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr + 1),
            NextState("FETCH1_L"),
        )
        fsm.act("FETCH1_L",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr + 1), # Hold.
            NextValue(d1, self.sp_dat),
            NextState("DISPATCH"),
        )
        fsm.act("DISPATCH",
            self.busy.eq(1),
            If(unit == 0,
                NextState("IDLE"),
            ).Elif(unit == 1,
                self.eng_valid.eq(1),
                If(self.eng_ready,
                    NextState("WAIT_ENG"),
                )
            ).Elif(unit == 2,
                self.vec_start.eq(1),
                NextState("WAIT_VEC"),
            ).Else(
                NextState("IDLE"), # Unknown unit: stop (firmware bug guard).
            )
        )
        fsm.act("WAIT_ENG",
            self.busy.eq(1),
            If(self.eng_done,
                NextState("NEXT"),
            )
        )
        fsm.act("WAIT_VEC",
            self.busy.eq(1),
            If(self.vec_done,
                NextState("NEXT"),
            )
        )
        fsm.act("NEXT",
            self.busy.eq(1),
            NextValue(dptr, dptr + 2),
            NextValue(left, left - 1),
            NextValue(dcount, dcount + 1),
            NextState("FETCH0"),
        )

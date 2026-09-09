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
       [9] (engine) NOWAIT: dispatch and move on instead of waiting for done - following
           VEC descriptors execute while the engine op streams (the per-cycle scratchpad
           arbitration makes that safe, docs/overlap-design.md). The list author asserts
           independence; the hardware still interlocks the obvious hazards: a later
           ENGINE descriptor waits in DISPATCH (engine ready = idle), a VEC descriptor
           whose args read engine state (KABS source, S7-from-absmax w0[4], any kreg-add)
           stalls until the in-flight op completes, and list end drains before idle.
       [10] WAIT-ENGINE barrier: this descriptor does not dispatch until any in-flight
           NOWAIT engine op has completed. For data dependences the arg-source interlock
           cannot see (e.g. softmax reading the scores buffer the engine is filling).
       [5] (engine) capture k(absmax) into kreg[w0[7:6]] when this op completes.
       [8] (engine) ALSO accumulate that k into kreg[3] (kreg3 += k) - lets one register
           collect a SUM of shifts across ops (silu's eg+eu term) while [7:6] keeps each
           op's own k addressable. Firmware zeroes kreg3 by capturing with [7:6]=3 first.
       (vec) w1[31:16]: per-arg 2-bit kreg-add select (arg i at [16+2i +: 2]): 0 = none,
           n = SUBTRACT kreg[n-1] from the arg AFTER its 2-bit source resolves - or ADD it
           when w0[11] is set (the 2026-08-28 VSHLS convention turned silu's A4 into
           imm + kg, which the subtract-only form cannot express). Turns
           "imm - earlier k" - the FFN exponent algebra that forced a CPU round-trip per
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
          w8 = xsrc;  w9 = dst;  w10 = chan;
          w11 = rep, w12 = dstride, w13 = sstride (multi-packet STORE, F1-1).
  vec:    w1 = pc;  w2..w9 = args 0..7.

Numerics are UNCHANGED - sequencing is transparent; the golden mirrors need no update.

Scratchpad access rides the engine fill port (never a fifth RAM port): descriptors are
fetched only while both units are idle, so the integrator grants the port as
vec.busy > seq.sp_active > engine."""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

# BonsaiSequencer ----------------------------------------------------------------------------------

class BonsaiSequencer(LiteXModule):
    def __init__(self):
        self.base   = CSRStorage(24, description="Descriptor list byte address (32B aligned).")
        self.count  = CSRStorage(16, description="Descriptor count (stop descriptor also ends).")
        self.start  = CSRStorage(1,  description="Write 1 to run the list.")
        self.sbmode = CSRStorage(1, reset=0, description=
            "Posted-write ordering. 0 = legacy: the descriptor fetch stalls while any "
            "posted write is in flight. 1 = drain-at-start only (see sbw_pending). Exists "
            "so the two can be A/B'd in one boot, which is the only measurement this "
            "project trusts; the start drain runs in both modes (harmless under legacy, "
            "which is strictly stronger), so this bit is the whole behavioural difference.")
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
        self.eng_rep     = Signal(5)   # Multi-packet STORE (words 11-13).
        self.eng_dstride = Signal(24)
        self.eng_sstride = Signal(8)
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
        # Posted sideband writes still sitting in the scratchpad's FIFO. The descriptor
        # fetch used to be held for as long as ANY write was in flight, which was correct
        # but made the CPU's patch burst starve a running list: ~7.5k cycles/layer for 80
        # writes (v8 qtrace), and two attempts to cut the CPU work around those writes
        # each cost ~5% because they only bunched the writes tighter. The ordering that is
        # actually required is "every patch for a list has landed before that list starts",
        # so the FIFO is drained once at START instead (SBDRAIN below).
        #
        # TWO INVARIANTS this relies on, both firmware-side:
        #  1. A list's descriptors are never patched while THAT list is running. Patching
        #     other lists under a running one is the whole point of the sideband slot and
        #     stays safe.
        #  2. A patch store is in the FIFO before the `start` CSR write is issued - i.e.
        #     the CPU's descriptor stores are ordered ahead of the start. The scratchpad
        #     window is mapped cached=False (trixium.py) and the core is in-order, so
        #     this holds. NOT a new requirement: a store still sitting in a CPU-side buffer
        #     never raised sbw_pending either, so the old fetch interlock needed it too.
        #     What is new is only (1).
        # Unwired (reset 0) this is a no-op, which keeps unit-test DUTs on FETCH-at-start.
        self.sbw_pending = Signal()
        self.sp_gnt    = Signal(reset=1) # Integrator: port is the seq's this cycle (a vec
                                         # load outranks descriptor fetches). Reset-1 keeps
                                         # unwired unit-test DUTs on the old behavior.

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
        nowait  = Signal()
        self.comb += [
            unit.eq(w(0, 0)[0:4]),
            argsrc.eq(w(0, 0)[16:32]),
            nowait.eq(w(0, 0)[9]),
        ]

        # NOWAIT bookkeeping. eng_disp fires on the engine-dispatch handshake; a NOWAIT
        # dispatch leaves the op in flight. done can never coincide with the next accept
        # (the engine's cmd.ready is an IDLE-state signal, one cycle after done), so the
        # set/clear priority below is safe.
        eng_disp     = Signal()
        eng_inflight = Signal()
        # Capture target of the IN-FLIGHT op (for the per-register kadd hazard below).
        infl_cap_en  = Signal()
        infl_cap_sel = Signal(2)
        self.sync += [
            If(eng_disp & nowait,
                eng_inflight.eq(1),
                infl_cap_en.eq(w(0, 0)[5]),
                infl_cap_sel.eq(w(0, 0)[6:8]),
            ).Elif(self.eng_done,
                eng_inflight.eq(0),
            ),
        ]
        # A vec descriptor whose arguments read engine state must not race an in-flight
        # engine op: KABS source (code 1) and S7-seed-from-absmax (w0[4]) read the LIVE
        # tracker, so any in-flight op poisons them. A kreg-add only hazards against the
        # SPECIFIC register the in-flight op will capture (or a capture still in its
        # done+2 pipe) - the first interlock was register-blind and serialized the
        # FFN list's norm-under-stream, costing 4% (2026-08-29). kadd reaches kreg0-2
        # only, so the kreg3 accumulator never hazards a kadd.
        vec_needs_eng = Signal()
        kadd_hazard   = Signal() # Driven after the capture block (needs cap_d/cap_sel).

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
        # Two subtleties, both load-bearing:
        # - Gated on self.busy, not an FSM state: eng_done also pulses for CPU-dispatched
        #   ops, but those never run while the sequencer owns the command interface, so
        #   busy & eng_done is exactly "our engine descriptor completed".
        # - The capture is DELAYED two cycles. The engine absmax tracker lags its write
        #   stage by two cycles (matvec.py: "the tracker lags the write by two cycles;
        #   every reader trails done by more") - the KABS arg source qualifies because the
        #   next DISPATCH is >2 cycles away, but a same-edge capture at done would miss a
        #   max arriving in the final beat. Control bits are latched at done (d0 is
        #   overwritten by the next descriptor fetch soon after); k is applied at done+2.
        cap_d   = Signal(2)
        cap_en  = Signal()
        cap_acc = Signal()
        cap_sel = Signal(2)
        self.sync += [
            cap_d.eq(Cat(self.busy & self.eng_done, cap_d[0])),
            # Latched at the DISPATCH handshake: with NOWAIT the descriptor buffer holds a
            # LATER descriptor when done pulses. Serial lists latch identical values (the
            # buffer does not move during WAIT_ENG).
            If(eng_disp,
                cap_en.eq(w(0, 0)[5]),
                cap_acc.eq(w(0, 0)[8]),
                cap_sel.eq(w(0, 0)[6:8]),
            ),
            If(cap_d[1],
                If(cap_en,  kregs[cap_sel].eq(k_absmax)),
                If(cap_acc, kregs[3].eq(kregs[3] + k_absmax)),
            ),
        ]
        kadds_i = [w(0, 1)[16 + 2*i:18 + 2*i] for i in range(8)]
        self.comb += [
            kadd_hazard.eq(reduce(or_, [
                (ka != 0) & (
                    (eng_inflight & infl_cap_en & (infl_cap_sel == (ka - 1)))
                    | ((cap_d != 0) & cap_en & (cap_sel == (ka - 1))))
                for ka in kadds_i])),
            vec_needs_eng.eq(
                ((w(0, 0)[4]
                  | reduce(or_, [argsrc[2*i:2*i+2] == 1 for i in range(8)]))
                 & eng_inflight)
                | (w(0, 0)[10] & eng_inflight) # Explicit barrier.
                | kadd_hazard),
        ]

        # A5 from a scratchpad LUT (w0[12], 2026-09-09): A5 = LUT[imm5 + kreg[sel5]], the
        # LUT's beat index in w(1,10) (an engine-only word, free on vec descriptors). The
        # one arg the k/v chain could not derive in-sequencer was rmsnorm's eps, a
        # power-of-two scaling of rms_eps by the source exponent - which is a per-layer
        # constant plus the shift the sequencer already holds in a kreg. The firmware
        # writes 32 entries per norm per layer (under the previous FFN list) and patches
        # the per-token part of the exponent into imm5, as it patched eps itself before.
        # The kreg comes from A5's own kadd select, so the capture-pipe interlock
        # (kadd_hazard) covers the read; the index is clamped to the 32-entry table.
        lut_mode = Signal()
        lut_base = Signal(13)
        lut_sel  = Signal(2)
        lut_sum  = Signal(32)
        lut_idx  = Signal(5)
        lut_word = Signal(32)
        # Timing: the two things this must not do on a 98%-full die are add an adder to
        # the scratchpad address cone or a mux to its 256-bit read bus (the vec unit's
        # ex1_r family lives there). So the address is registered in FETCHL from a
        # comb index (lut_adr_r), the fetched beat is captured whole (lut_beat_r, like
        # d0/d1) and the word select runs off that register a cycle later.
        lut_adr_r  = Signal(13)
        lut_sel_r  = Signal(3)
        lut_beat_r = Signal(256)
        self.comb += [
            lut_mode.eq((unit == 2) & w(0, 0)[12]),
            lut_base.eq(w(1, 10)[0:13]),
            lut_sel.eq(w(0, 1)[26:28]),
            lut_sum.eq(w(0, 7) + Mux(lut_sel == 0, 0, kregs[lut_sel - 1])),
            # Negative or >31 both clamp to the table's ends (the firmware never asks
            # for either; a wrong index is a wrong eps, not a hang).
            lut_idx.eq(Mux(lut_sum[31], 0, Mux(lut_sum[5:31] != 0, 31, lut_sum[0:5]))),
            lut_word.eq(Array([lut_beat_r[32*j:32*j + 32] for j in range(8)])[lut_sel_r]),
        ]

        for i in range(8):
            src  = Signal(2)
            imm  = Signal(32)
            base = Signal(32)
            full = Signal(32)
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
                # SUBTRACT by default: the old exponent args had the shape 'constant - k'
                # (pre-VSHLS silu A4 = -(eg+6) = C_layer - kg). w0[11] flips this
                # descriptor's kreg terms to ADD for the direct-shift convention
                # (A4 = eg + 6 = C_layer + kg). The '+latest k' case without a register
                # is already covered by SEQ_SRC_KABS.
                full.eq(Mux(w(0, 0)[11],
                    base + Mux(kadd == 0, 0, kregs[kadd - 1]),
                    base - Mux(kadd == 0, 0, kregs[kadd - 1]))),
                # LUT mode replaces the whole A5 formula (its kreg is the index).
                self.vec_args[i].eq(Mux(lut_mode, lut_word, full) if i == 5 else full),
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
            self.eng_rep.eq(w(1, 11)[0:5]),
            self.eng_dstride.eq(w(1, 12)[0:24]),
            self.eng_sstride.eq(w(1, 13)[0:8]),
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
                NextState("SBDRAIN"),
            )
        )
        fsm.act("SBDRAIN", # Let every patch posted for this list land before fetching it.
            self.busy.eq(1),
            If(~self.sbw_pending,
                NextState("FETCH0"),
            )
        )
        fsm.act("FETCH0",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr),
            If(left == 0,
                If(~eng_inflight, # Drain a NOWAIT op before reporting idle.
                    NextState("IDLE"),
                )
            ).Elif(self.sp_gnt, # Hold while a vec load owns the port.
                NextState("FETCH0_L"),
            )
        )
        fsm.act("FETCH0_L", # dat valid this cycle: it reflects the address sampled at the
            # end of the GRANTED FETCH0 cycle, whoever owns the port now.
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
            If(self.sp_gnt,
                NextState("FETCH1_L"),
            )
        )
        fsm.act("FETCH1_L",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(dptr + 1), # Hold.
            NextValue(d1, self.sp_dat),
            # d1 lands this edge; lut_mode reads d0 (captured two states ago). w(1,10)
            # is read in FETCHL from the registered d1.
            If((unit == 2) & w(0, 0)[12],
                NextState("FETCHL"),
            ).Else(
                NextState("DISPATCH"),
            )
        )
        fsm.act("FETCHL", # Resolve the LUT index (waits for its kreg), register the address.
            self.busy.eq(1),
            # The index reads a kreg: hold while its capture is still in flight or in the
            # done+2 pipe (the same interlock DISPATCH applies to kadd consumers).
            If(~kadd_hazard,
                NextValue(lut_adr_r, lut_base + lut_idx[3:5]),
                NextValue(lut_sel_r, lut_idx[0:3]),
                NextState("FETCHL_A"),
            )
        )
        fsm.act("FETCHL_A", # One more beat: the LUT entry for this descriptor's A5.
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(lut_adr_r),
            If(self.sp_gnt,
                NextState("FETCHL_L"),
            )
        )
        fsm.act("FETCHL_L",
            self.busy.eq(1),
            self.sp_active.eq(1),
            self.sp_adr.eq(lut_adr_r), # Hold.
            NextValue(lut_beat_r, self.sp_dat), # lut_word reads it in DISPATCH.
            NextState("DISPATCH"),
        )
        fsm.act("DISPATCH",
            self.busy.eq(1),
            If(unit == 0,
                If(~eng_inflight, # Drain a NOWAIT op before stopping.
                    NextState("IDLE"),
                )
            ).Elif(unit == 1,
                # A second engine descriptor waits here naturally: eng_ready is the
                # engine's IDLE-state cmd.ready.
                self.eng_valid.eq(1),
                If(self.eng_ready,
                    eng_disp.eq(1),
                    If(nowait,
                        NextState("NEXT"),
                    ).Else(
                        NextState("WAIT_ENG"),
                    )
                )
            ).Elif(unit == 2,
                # Hazard interlock: args reading engine state wait for the in-flight op
                # (and kadd consumers for their register's capture pipe).
                If(~vec_needs_eng,
                    self.vec_start.eq(1),
                    NextState("WAIT_VEC"),
                )
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

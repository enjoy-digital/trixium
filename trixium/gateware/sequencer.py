#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiSequencer: hardware execution of descriptor lists (engine/vector/rotation ops).

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
  had:    Optional unit 3, always serialized; w0 exactly 3, w1 src, w2 signs,
          w3 dst, w4 block count, w5 stats, w6 inverse (0/1), w7..w15 zero.
          See docs/hadamard-scratchpad.md for bounds, ownership and raw numerics.
  quant:  Optional unit 4, always serialized; w0 exactly 4, w1 FP32 src,
          w2 Q8 dst, w3 exponent array, w4 elements/block, w5 blocks, w6..w15 zero.
  requant: Unit 5 (quant ABI >=1.1), serialized; w0 exactly 5, w1 raw int32 src,
           w2 Q8 dst, w3 input exponent array, w4 raw absmax array, w5 result
           exponent, w6 blocks of 1024, w7..w15 zero. See docs/rotation-requantization.md.
  native: Optional units 6 (pipe region/offset/stride setup) and 7 (projection),
          always serialized. See docs/native-sequencer.md for their strict ABI.
  restore: Optional unit 8, serialized; w0 exactly 8, w1 int32 src, w2 FP32 dst,
           w3 resident exponent beat, w4 elements, w5 signed adjustment (R-B),
           w6..w15 zero. See docs/projection-restoration.md.
  embedding: Optional unit 9, serialized; w0 exactly 9, w1 compact PTQ src,
             w2 inverse signs, w3 FP32 dst, w4 groups, w5..w15 zero.
             See docs/native-embedding-restoration.md.
  region fetch: Optional unit 10, serialized; w0 exactly 10, w1 region ID,
                w2/w3 64-bit byte offset low/high, w4 full 32-byte beat count,
                w5 scratchpad dst, w6..w15 zero. Bounds/reservations/transport
                checks use the shared region reader; no legacy physical address.
  silu:   Optional unit 11 (ABI 1.1), serialized; w0 exactly 11, w1 FP32 gate src, w2
          FP32 multiplier src, w3 dst (== w1 or disjoint), w4 fixup bitmap, w5 elements,
          w6 op (bit0: 0 RUN, 1 LOAD tables from w1) | unit_m << 1 (m = 1.0, w2 unused)
          | table_only << 2 (ABI 1.2: pass-through/-0 kinds are fixups - any table)
          | table_set << 3 (ABI 1.3: the resident table pair a LOAD writes and a RUN reads),
          w7 seg_beats | perm_a << 16 | perm_b << 24 (permuted output segments, 0 = none),
          w8..w15 zero. seq.silu_fixups sums the flagged elements of the list.
  norm:   Optional unit 12 (ABI 1.1), serialized; w0 exactly 12, w1 src, w2 dst, w3 weight,
          w4 n | nseg << 16, w5 FP32 eps, w6 op (0 RMS, 1 L2), w7 src_stride | dst_stride
          << 16 (beats between segments, 0 = contiguous), w8..w15 zero. norm_sp.py.
  conv:   Optional unit 13, serialized; w0 exactly 13, w1..w3 window slots x0..x2, w4 new
          input x3, w5 taps, w6 dst, w7 channels | op << 16 (op 1: dst = x0 + x3, the
          residual add), w8..w15 zero. gateware/conv_sp.py.

Legacy engine/vector numerics are unchanged. Unit 3 returns raw integer rotations;
it does not silently replace a legacy dense rotation or its quantization boundary.

Scratchpad access rides the engine fill port (never a fifth RAM port): descriptors are
fetched only while both units are idle, so the integrator grants the port as
vec.busy > seq.sp_active > engine."""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *
from litex.soc.interconnect import stream

# BonsaiSequencer ----------------------------------------------------------------------------------

class BonsaiSequencer(LiteXModule):
    def __init__(self, *, with_hadamard=False, with_activation_quantize=False, native_pipes=0,
                 with_projection_restore=False, with_embedding_restore=False,
                 with_region_fetch=False, with_silu=False, with_norm=False, with_conv=False,
                 scratchpad_size=256*1024):
        if not isinstance(native_pipes, int) or not 0 <= native_pipes <= 31:
            raise ValueError('native_pipes must be in [0, 31]')
        if with_region_fetch and not native_pipes:
            raise ValueError('Sequenced region fetch requires native memory support')
        extended = (with_hadamard or with_activation_quantize or native_pipes or with_projection_restore or
                    with_embedding_restore or with_silu or with_norm or with_conv)
        if extended and (scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32):
            raise ValueError("Resident sequencer needs an aligned scratchpad within its 13-bit address space")
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
        extension_errors = []
        if with_hadamard:
            from trixium.gateware.hadamard_sp import HADAMARD_COMMAND
            self.hadamard_cmd = stream.Endpoint(HADAMARD_COMMAND)
            self.hadamard_done = Signal()
            self.hadamard_failed = Signal(8)
            self.hadamard_abi = CSRStatus(32, reset=0x00010000)
            self.hadamard_error = CSRStatus(32, description=
                "Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, "
                "bit2 output overlaps descriptor list, bits8..15 rotation-unit error.")
            extension_errors.append(self.hadamard_error)
        if with_activation_quantize:
            from trixium.gateware.activation_sp import ACTIVATION_COMMAND
            self.activation_quantize_cmd = stream.Endpoint(ACTIVATION_COMMAND)
            self.activation_quantize_done = Signal()
            self.activation_quantize_failed = Signal(8)
            self.activation_quantize_abi = CSRStatus(32, reset=0x00010001)
            self.activation_quantize_error = CSRStatus(32, description=
                "Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, "
                "bits8..15 conversion-unit error (bit14 protects descriptor list; "
                "bit15 rejects invalid rotation metadata/raw values).")
            extension_errors.append(self.activation_quantize_error)
        if native_pipes:
            self.native_abi = CSRStatus(32, reset=0x00010000)
            self.native_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bit2 output overlaps descriptor list, bit3 config channel, bit8 engine failure.')
            extension_errors.append(self.native_error)
            self.native_cfg = stream.Endpoint([('chan', 5), ('region', 32), ('offset', 64), ('stride', 64)])
            self.native_cfg_error = Signal()
            self.native_failed = Signal()
            self.native_manual_pending = Signal()
            self.native_engine_busy = Signal()
            self.native_manual_drain = Signal()
            self.eng_native = Signal()
        if with_projection_restore:
            # Append after native CSRs so all previously existing profiles keep
            # their register addresses when this independent option is enabled.
            from trixium.gateware.projection_restore_sp import RESTORE_COMMAND
            self.projection_restore_cmd = stream.Endpoint(RESTORE_COMMAND)
            self.projection_restore_done = Signal()
            self.projection_restore_failed = Signal(8)
            self.projection_restore_abi = CSRStatus(32, reset=0x00010000)
            self.projection_restore_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bits8..15 restoration-client error (bit14 protects the full descriptor list).')
            extension_errors.append(self.projection_restore_error)
        if with_embedding_restore:
            # Append-only optional ABI. The shared-cluster profile is currently
            # max_groups=64/input_width=16; reject unsupported model spans before
            # execution rather than silently falling back to lossy arithmetic.
            from trixium.gateware.embedding_restore_sp import EMBEDDING_COMMAND
            self.embedding_restore_cmd = stream.Endpoint(EMBEDDING_COMMAND)
            self.embedding_restore_done = Signal()
            self.embedding_restore_failed = Signal(16)
            self.embedding_restore_abi = CSRStatus(32, reset=0x00010000)
            self.embedding_restore_limits = CSRStatus(32, reset=64 | (16 << 16), description=
                'Bits0..15 maximum PTQ groups; bits16..23 exact aligned integer input width.')
            self.embedding_restore_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bits8..23 embedding-client error (bit10 protects the full descriptor list).')
            extension_errors.append(self.embedding_restore_error)
        if with_region_fetch:
            self.region_fetch_cmd = stream.Endpoint([
                ('region', 32), ('offset', 64), ('length', 24), ('dst', 24)])
            self.region_fetch_done = Signal()
            self.region_fetch_failed = Signal(9)
            self.region_fetch_manual_pending = Signal()
            self.region_fetch_manual_failed = Signal()
            self.region_fetch_abi = CSRStatus(32, reset=0x00010000)
            self.region_fetch_error = CSRStatus(32, description=
                'Sticky until common reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bit2 output overlaps list, bit3 older manual read failure, '
                'bits8..15 transport/config errors, bit16 failed reader state.')
            extension_errors.append(self.region_fetch_error)
        if with_silu:
            # Append-only, after every other optional block.
            from trixium.gateware.silu_sp import SILU_COMMAND
            self.silu_cmd = stream.Endpoint(SILU_COMMAND)
            self.silu_done = Signal()
            self.silu_failed = Signal(8)
            self.silu_unit_fixups = Signal(16)
            self.silu_abi = CSRStatus(32, reset=0x00010003)
            self.silu_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bits8..15 SiLU-client error (bit14 protects the full descriptor list).')
            self.silu_fixups = CSRStatus(32, description=
                'Fixup elements flagged by unit 11 since the last list start (all its RUN descriptors).')
            extension_errors.append(self.silu_error)
        if with_norm:
            from trixium.gateware.norm_sp import NORM_COMMAND
            self.norm_cmd = stream.Endpoint(NORM_COMMAND)
            self.norm_done, self.norm_failed = Signal(), Signal(8)
            self.norm_abi = CSRStatus(32, reset=0x00010001)
            self.norm_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bits8..15 norm-client error (bit14 protects the full descriptor list).')
            extension_errors.append(self.norm_error)
        if with_conv:
            from trixium.gateware.conv_sp import CONV_COMMAND
            self.conv_cmd = stream.Endpoint(CONV_COMMAND)
            self.conv_done, self.conv_failed = Signal(), Signal(8)
            self.conv_abi = CSRStatus(32, reset=0x00010000)
            self.conv_error = CSRStatus(32, description=
                'Sticky until cluster reset: bit0 list bounds/alignment, bit1 malformed descriptor, '
                'bits8..15 conv-client error (bit14 protects the full descriptor list).')
            extension_errors.append(self.conv_error)

        if with_projection_restore:
            # Appended last: every earlier register keeps its address.
            self.restore_argmax_clear = Signal()
            self.restore_argmax_valid_in = Signal()
            self.restore_argmax_idx_in, self.restore_argmax_val_in = Signal(32), Signal(32)
            self.restore_argmax_idx = CSRStatus(32, description=
                'Greedy argmax (first max, NaN-first frozen) of every unit-8 output since the '
                'last list start: the element index, 0xFFFFFFFF before any output.')
            self.restore_argmax_val = CSRStatus(32, description='Its FP32 value (0 when none yet).')

        # Engine command side (muxed into engine.cmd by BonsaiEngineCSR while busy).
        self.eng_valid  = Signal()
        self.eng_ready  = Signal()
        self.eng_kind   = Signal(3)
        self.eng_rows   = Signal(16)
        self.eng_cols   = Signal(17 if native_pipes else 14)
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

        list_ok = C(1, 1)
        had_dispatch, quant_dispatch, restore_dispatch, embedding_dispatch, bad_list = [], [], [], [], []
        region_fetch_dispatch, silu_dispatch, norm_dispatch, conv_dispatch = [], [], [], []
        unknown_unit = [NextState("IDLE")]
        if extended:
            list_base = Signal(24)
            list_end = Signal(32)
            start_end = Signal(32)
            self.comb += start_end.eq(self.base.storage + (self.count.storage << 6))
            list_ok = (self.base.storage[:5] == 0) & (start_end <= scratchpad_size)
            self.sync += If(self.start.re & self.start.storage & ~self.busy,
                list_base.eq(self.base.storage), list_end.eq(start_end))
            unknown_unit = [*[NextValue(error.status, 2) for error in extension_errors], NextState("EXT_FAILED")]
            bad_list = [*[NextValue(error.status, 1) for error in extension_errors], NextState("EXT_FAILED")]
        if with_hadamard:
            dst_end, stats_end = Signal(64), Signal(64)
            malformed, list_overlap = Signal(), Signal()
            self.comb += [
                dst_end.eq(w(0, 3) + (w(0, 4) << 12)),
                stats_end.eq(w(0, 5) + (((w(0, 4) + C(7, 33)) >> 3) << 5)),
                malformed.eq((w(0, 0) != 3) | (w(0, 1)[24:] != 0) |
                    (w(0, 2)[24:] != 0) | (w(0, 3)[24:] != 0) | (w(0, 4)[16:] != 0) |
                    (w(0, 5)[24:] != 0) | (w(0, 6)[1:] != 0) | (w(0, 7) != 0) | (d1 != 0)),
                list_overlap.eq(((w(0, 3) < list_end) & (list_base < dst_end)) |
                                ((w(0, 5) < list_end) & (list_base < stats_end))),
                self.hadamard_cmd.src.eq(w(0, 1)[:24]), self.hadamard_cmd.signs.eq(w(0, 2)[:24]),
                self.hadamard_cmd.dst.eq(w(0, 3)[:24]), self.hadamard_cmd.blocks.eq(w(0, 4)[:16]),
                self.hadamard_cmd.stats.eq(w(0, 5)[:24]), self.hadamard_cmd.inverse.eq(w(0, 6)[0]),
            ]
            had_dispatch = [If(malformed | list_overlap,
                NextValue(self.hadamard_error.status, Cat(C(0, 1), malformed, list_overlap)),
                NextState("EXT_FAILED"),
            ).Else(NextState("DISPATCH_HAD"))]
        if with_activation_quantize:
            malformed = Signal()
            requantize = Signal()
            command = self.activation_quantize_cmd
            self.comb += [
                requantize.eq(w(0, 0)[0:4] == 5),
                malformed.eq((w(0, 0) != Mux(requantize, 5, 4)) | (w(0, 1)[24:] != 0) | (w(0, 2)[24:] != 0) |
                    (w(0, 3)[24:] != 0) | (w(0, 7) != 0) | (d1 != 0) |
                    Mux(requantize, (w(0, 4)[24:] != 0) | (w(0, 5)[24:] != 0) | (w(0, 6)[16:] != 0),
                        (w(0, 4)[16:] != 0) | (w(0, 5)[16:] != 0) | (w(0, 6) != 0))),
                command.src.eq(w(0, 1)[:24]), command.dst.eq(w(0, 2)[:24]),
                command.exponents.eq(w(0, 3)[:24]), command.elements.eq(Mux(requantize, 1024, w(0, 4)[:16])),
                command.blocks.eq(Mux(requantize, w(0, 6)[:16], w(0, 5)[:16])),
                command.requantize.eq(requantize), command.maxima.eq(Mux(requantize, w(0, 4)[:24], 0)),
                command.result_exponent.eq(Mux(requantize, w(0, 5)[:24], 0)), command.protect_start.eq(list_base),
                command.protect_end.eq(list_end),
            ]
            quant_dispatch = [If(malformed,
                NextValue(self.activation_quantize_error.status, 2), NextState("EXT_FAILED"),
            ).Else(NextState("DISPATCH_QUANT"))]
        if with_projection_restore:
            malformed = Signal()
            command = self.projection_restore_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 8) | (w(0, 1)[24:] != 0) | (w(0, 2)[24:] != 0) |
                    (w(0, 3)[24:] != 0) | (w(0, 4)[16:] != 0) |
                    (w(0, 6) != 0) | (w(0, 7) != 0) | (d1 != 0)),
                command.src.eq(w(0, 1)[:24]), command.dst.eq(w(0, 2)[:24]),
                command.exponents.eq(w(0, 3)[:24]), command.elements.eq(w(0, 4)[:16]),
                command.adjustment.eq(w(0, 5)),
                command.protect_start.eq(list_base), command.protect_end.eq(list_end),
            ]
            restore_dispatch = [If(malformed,
                NextValue(self.projection_restore_error.status, 2), NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_RESTORE'))]
            self.comb += [
                self.restore_argmax_clear.eq(self.start.re & self.start.storage & ~self.busy),
                self.restore_argmax_idx.status.eq(Mux(self.restore_argmax_valid_in,
                                                      self.restore_argmax_idx_in, 0xFFFFFFFF)),
                self.restore_argmax_val.status.eq(self.restore_argmax_val_in),
            ]
        if with_embedding_restore:
            malformed = Signal()
            command = self.embedding_restore_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 9) | (w(0, 1)[24:] != 0) | (w(0, 2)[24:] != 0) |
                    (w(0, 3)[24:] != 0) | (w(0, 4)[16:] != 0) |
                    (w(0, 5) != 0) | (w(0, 6) != 0) | (w(0, 7) != 0) | (d1 != 0)),
                command.src.eq(w(0, 1)[:24]), command.signs.eq(w(0, 2)[:24]),
                command.dst.eq(w(0, 3)[:24]), command.groups.eq(w(0, 4)[:16]),
                command.protect_start.eq(list_base), command.protect_end.eq(list_end),
            ]
            embedding_dispatch = [If(malformed,
                NextValue(self.embedding_restore_error.status, 2), NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_EMBEDDING'))]
        if with_silu:
            malformed = Signal()
            command = self.silu_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 11) | (w(0, 1)[24:] != 0) | (w(0, 2)[24:] != 0) |
                    (w(0, 3)[24:] != 0) | (w(0, 4)[24:] != 0) | (w(0, 5)[16:] != 0) |
                    (w(0, 6)[4:] != 0) | (d1 != 0)),
                command.src_g.eq(w(0, 1)[:24]), command.src_m.eq(w(0, 2)[:24]),
                command.dst.eq(w(0, 3)[:24]), command.fixmap.eq(w(0, 4)[:24]),
                command.elements.eq(w(0, 5)[:16]), command.op.eq(w(0, 6)[0]),
                command.unit_m.eq(w(0, 6)[1]), command.table_only.eq(w(0, 6)[2]),
                command.table_set.eq(w(0, 6)[3]),
                command.seg_beats.eq(w(0, 7)[:16]),
                command.perm_a.eq(w(0, 7)[16:24]), command.perm_b.eq(w(0, 7)[24:32]),
                command.protect_start.eq(list_base), command.protect_end.eq(list_end),
            ]
            self.sync += If(self.start.re & self.start.storage & ~self.busy,
                self.silu_fixups.status.eq(0))
            silu_dispatch = [If(malformed,
                NextValue(self.silu_error.status, 2), NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_SILU'))]
        if with_norm:
            malformed = Signal()
            command = self.norm_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 12) | (w(0, 1)[24:] != 0) | (w(0, 2)[24:] != 0) |
                    (w(0, 3)[24:] != 0) | (w(0, 6)[1:] != 0) | (d1 != 0)),
                command.src.eq(w(0, 1)[:24]), command.dst.eq(w(0, 2)[:24]),
                command.weight.eq(w(0, 3)[:24]), command.n.eq(w(0, 4)[:16]),
                command.nseg.eq(w(0, 4)[16:32]), command.eps.eq(w(0, 5)), command.op.eq(w(0, 6)[0]),
                command.src_stride.eq(w(0, 7)[:16]), command.dst_stride.eq(w(0, 7)[16:32]),
                command.protect_start.eq(list_base), command.protect_end.eq(list_end),
            ]
            norm_dispatch = [If(malformed,
                NextValue(self.norm_error.status, 2), NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_NORM'))]
        if with_conv:
            malformed = Signal()
            command = self.conv_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 13) | reduce(or_, [w(0, i)[24:] != 0 for i in range(1, 7)]) |
                    (w(0, 7)[17:] != 0) | (d1 != 0)),
                command.x0.eq(w(0, 1)[:24]), command.x1.eq(w(0, 2)[:24]), command.x2.eq(w(0, 3)[:24]),
                command.x3.eq(w(0, 4)[:24]), command.w.eq(w(0, 5)[:24]), command.dst.eq(w(0, 6)[:24]),
                command.channels.eq(w(0, 7)[:16]), command.op.eq(w(0, 7)[16]),
                command.protect_start.eq(list_base), command.protect_end.eq(list_end),
            ]
            conv_dispatch = [If(malformed,
                NextValue(self.conv_error.status, 2), NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_CONV'))]
        if with_region_fetch:
            malformed, list_overlap = Signal(), Signal()
            destination_end = Signal(64)
            command = self.region_fetch_cmd
            self.comb += [
                malformed.eq((w(0, 0) != 10) | (w(0, 4)[24:] != 0) | (w(0, 5)[24:] != 0) |
                    (w(0, 6) != 0) | (w(0, 7) != 0) | (d1 != 0)),
                destination_end.eq(w(0, 5) + (w(0, 4) << 5)),
                list_overlap.eq((w(0, 4) != 0) & (w(0, 5) < list_end) & (list_base < destination_end)),
                command.region.eq(w(0, 1)), command.offset.eq(Cat(w(0, 2), w(0, 3))),
                command.length.eq(w(0, 4)[:24]), command.dst.eq(w(0, 5)[:24]),
            ]
            region_fetch_dispatch = [If(malformed | list_overlap,
                NextValue(self.region_fetch_error.status, Cat(C(0, 1), malformed, list_overlap)),
                NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_REGION_FETCH'))]
        if native_pipes:
            cfg_malformed, cfg_channel_bad = Signal(), Signal()
            projection_malformed, projection_overlap = Signal(), Signal()
            projection_rows = Signal(32)
            projection_end, head_b_start, head_b_end = Signal(64), Signal(64), Signal(64)
            self.comb += [
                cfg_malformed.eq((w(0, 0) != 6) | (w(0, 7) != 0) | (d1 != 0)),
                cfg_channel_bad.eq(w(0, 1) >= native_pipes),
                self.native_cfg.chan.eq(w(0, 1)[:5]), self.native_cfg.region.eq(w(0, 2)),
                self.native_cfg.offset.eq(Cat(w(0, 3), w(0, 4))),
                self.native_cfg.stride.eq(Cat(w(0, 5), w(0, 6))),
                self.eng_native.eq(w(0, 0) == 7),
                projection_malformed.eq((w(0, 0) != 7) | ((w(0, 1) & 0xfffff4fe) != 0) |
                    (w(0, 2)[21:] != 0) | (w(0, 3) != 0) | (w(0, 4) != 0) | (w(0, 5) != 0) |
                    (w(0, 6)[:8] != 0) | ((w(0, 7) & 0xffc0ff80) != 0) |
                    (w(1, 8)[24:] != 0) | (w(1, 9)[24:] != 0) | (d1[64:] != 0)),
                projection_rows.eq(w(0, 2)[:16]*native_pipes + w(0, 2)[16:21]),
                projection_end.eq(w(1, 9) + (projection_rows << 2)),
                head_b_start.eq(w(1, 9) + (w(0, 6)[16:] << 2)),
                head_b_end.eq(head_b_start + (projection_rows << 2)),
                projection_overlap.eq(~w(0, 1)[9] &
                    (((w(1, 9) < list_end) & (list_base < projection_end)) |
                     (w(0, 1)[11] & (head_b_start < list_end) & (list_base < head_b_end)))),
            ]
            cfg_dispatch = [If(cfg_malformed | cfg_channel_bad,
                NextValue(self.native_error.status, Cat(C(0, 1), cfg_malformed, C(0, 1), cfg_channel_bad)),
                NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_NATIVE_CFG'))]
            native_dispatch = [If(projection_malformed | projection_overlap,
                NextValue(self.native_error.status, Cat(C(0, 1), projection_malformed, projection_overlap)),
                NextState('EXT_FAILED'),
            ).Else(NextState('DISPATCH_NATIVE'))]

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
            # SBDRAIN can finish older manual commands while busy is asserted.
            # Their completion must never replay a previous list's k capture.
            cap_d.eq(Cat(self.busy & self.eng_done &
                         (~self.native_manual_drain if native_pipes else C(1, 1)), cap_d[0])),
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
            self.eng_cols.eq(Mux(self.eng_native,
                Mux(w(0, 1)[0], w(0, 6)[8:16] << 5, w(0, 6)[8:16] << 7),
                w(0, 3)[0:14]) if native_pipes else w(0, 3)[0:14]),
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
                If(list_ok, NextState("SBDRAIN")).Else(*bad_list),
            )
        )
        fsm.act("SBDRAIN", # Let every patch posted for this list land before fetching it.
            self.busy.eq(1),
            *([self.native_manual_drain.eq(1)] if native_pipes else []),
            If(~self.sbw_pending &
               (~(self.native_manual_pending | self.native_engine_busy) if native_pipes else C(1, 1)) &
               (~self.region_fetch_manual_pending if with_region_fetch else C(1, 1)),
                NextState("FETCH0"),
            ),
            *([If(self.native_failed, NextValue(self.native_error.status, 256), NextState('EXT_FAILED'))]
              if native_pipes else []),
            *([If(self.region_fetch_manual_failed,
                NextValue(self.region_fetch_error.status, 8 | (self.region_fetch_failed << 8)),
                NextState('EXT_FAILED'))] if with_region_fetch else []),
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
            ).Elif(unit == 3, *(had_dispatch if with_hadamard else unknown_unit)
            ).Elif(unit == 4, *(quant_dispatch if with_activation_quantize else unknown_unit)
            ).Elif(unit == 5, *(quant_dispatch if with_activation_quantize else unknown_unit)
            ).Elif(unit == 6, *(cfg_dispatch if native_pipes else unknown_unit)
            ).Elif(unit == 7, *(native_dispatch if native_pipes else unknown_unit)
            ).Elif(unit == 8, *(restore_dispatch if with_projection_restore else unknown_unit)
            ).Elif(unit == 9, *(embedding_dispatch if with_embedding_restore else unknown_unit)
            ).Elif(unit == 10, *(region_fetch_dispatch if with_region_fetch else unknown_unit)
            ).Elif(unit == 11, *(silu_dispatch if with_silu else unknown_unit)
            ).Elif(unit == 12, *(norm_dispatch if with_norm else unknown_unit)
            ).Elif(unit == 13, *(conv_dispatch if with_conv else unknown_unit)
            ).Else(*unknown_unit)
        )
        if with_hadamard:
            fsm.act("DISPATCH_HAD",
                self.busy.eq(1),
                # Always serialize, irrespective of an earlier engine's NOWAIT.
                If(~eng_inflight,
                    self.hadamard_cmd.valid.eq(1),
                    If(self.hadamard_cmd.ready, NextState("WAIT_HAD")),
                ),
            )
            fsm.act("WAIT_HAD",
                self.busy.eq(1),
                If(self.hadamard_failed != 0,
                    NextValue(self.hadamard_error.status, self.hadamard_failed << 8),
                    NextState("EXT_FAILED"),
                ).Elif(self.hadamard_done, NextState("NEXT")),
            )
        if with_activation_quantize:
            fsm.act("DISPATCH_QUANT", self.busy.eq(1),
                If(~eng_inflight, self.activation_quantize_cmd.valid.eq(1),
                    If(self.activation_quantize_cmd.ready, NextState("WAIT_QUANT"))))
            fsm.act("WAIT_QUANT", self.busy.eq(1),
                If(self.activation_quantize_failed != 0,
                    NextValue(self.activation_quantize_error.status, self.activation_quantize_failed << 8),
                    NextState("EXT_FAILED"),
                ).Elif(self.activation_quantize_done, NextState("NEXT")))
        if extended:
            fsm.act("EXT_FAILED", self.busy.eq(1))
        if with_projection_restore:
            fsm.act('DISPATCH_RESTORE', self.busy.eq(1),
                If(~eng_inflight, self.projection_restore_cmd.valid.eq(1),
                    If(self.projection_restore_cmd.ready, NextState('WAIT_RESTORE'))))
            fsm.act('WAIT_RESTORE', self.busy.eq(1),
                If(self.projection_restore_failed != 0,
                    NextValue(self.projection_restore_error.status, self.projection_restore_failed << 8),
                    NextState('EXT_FAILED'),
                ).Elif(self.projection_restore_done, NextState('NEXT')))
        if with_embedding_restore:
            fsm.act('DISPATCH_EMBEDDING', self.busy.eq(1),
                If(~eng_inflight, self.embedding_restore_cmd.valid.eq(1),
                    If(self.embedding_restore_cmd.ready, NextState('WAIT_EMBEDDING'))))
            fsm.act('WAIT_EMBEDDING', self.busy.eq(1),
                If(self.embedding_restore_failed != 0,
                    NextValue(self.embedding_restore_error.status, self.embedding_restore_failed << 8),
                    NextState('EXT_FAILED'),
                ).Elif(self.embedding_restore_done, NextState('NEXT')))
        if with_silu:
            fsm.act('DISPATCH_SILU', self.busy.eq(1),
                If(~eng_inflight, self.silu_cmd.valid.eq(1),
                    If(self.silu_cmd.ready, NextState('WAIT_SILU'))))
            fsm.act('WAIT_SILU', self.busy.eq(1),
                If(self.silu_failed != 0,
                    NextValue(self.silu_error.status, self.silu_failed << 8),
                    NextState('EXT_FAILED'),
                ).Elif(self.silu_done,
                    NextValue(self.silu_fixups.status, self.silu_fixups.status + self.silu_unit_fixups),
                    NextState('NEXT')))
        for name, enabled in (("norm", with_norm), ("conv", with_conv)):
            if not enabled:
                continue
            cmd, failed, done = (getattr(self, name + "_cmd"), getattr(self, name + "_failed"),
                                 getattr(self, name + "_done"))
            error = getattr(self, name + "_error")
            fsm.act('DISPATCH_' + name.upper(), self.busy.eq(1),
                If(~eng_inflight, cmd.valid.eq(1), If(cmd.ready, NextState('WAIT_' + name.upper()))))
            fsm.act('WAIT_' + name.upper(), self.busy.eq(1),
                If(failed != 0,
                    NextValue(error.status, failed << 8), NextState('EXT_FAILED'),
                ).Elif(done, NextState('NEXT')))
        if with_region_fetch:
            fsm.act('DISPATCH_REGION_FETCH', self.busy.eq(1),
                If(self.region_fetch_failed != 0,
                    NextValue(self.region_fetch_error.status, self.region_fetch_failed << 8),
                    NextState('EXT_FAILED'),
                ).Elif(~eng_inflight,
                    self.region_fetch_cmd.valid.eq(1),
                    If(self.region_fetch_cmd.ready,
                        # Raw reads pulse eng_done too. Clear stale k-capture
                        # controls via the ordinary engine-dispatch bookkeeping;
                        # exact w0=10 has no NOWAIT or capture flags set.
                        eng_disp.eq(1), NextState('WAIT_REGION_FETCH'))))
            fsm.act('WAIT_REGION_FETCH', self.busy.eq(1),
                If(self.region_fetch_failed != 0,
                    NextValue(self.region_fetch_error.status, self.region_fetch_failed << 8),
                    NextState('EXT_FAILED'),
                ).Elif(self.region_fetch_done, NextState('NEXT')))
        if native_pipes:
            fsm.act('DISPATCH_NATIVE_CFG', self.busy.eq(1),
                If(self.native_failed,
                    NextValue(self.native_error.status, 256), NextState('EXT_FAILED'),
                ).Elif(~eng_inflight,
                    self.native_cfg.valid.eq(1),
                    If(self.native_cfg.ready, NextState('WAIT_NATIVE_CFG'))))
            fsm.act('WAIT_NATIVE_CFG', self.busy.eq(1),
                If(self.native_cfg_error,
                    NextValue(self.native_error.status, 8), NextState('EXT_FAILED'),
                ).Else(NextState('NEXT')))
            fsm.act('DISPATCH_NATIVE', self.busy.eq(1),
                If(self.native_failed,
                    NextValue(self.native_error.status, 256), NextState('EXT_FAILED'),
                ).Elif(~eng_inflight,
                    self.eng_valid.eq(1),
                    If(self.eng_ready, eng_disp.eq(1), NextState('WAIT_NATIVE'))))
            fsm.act('WAIT_NATIVE', self.busy.eq(1),
                If(self.native_failed,
                    NextValue(self.native_error.status, 256), NextState('EXT_FAILED'),
                ).Elif(self.eng_done, NextState('NEXT')))
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

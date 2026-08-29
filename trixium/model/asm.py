#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiVectorUnit microcode: ISA, assembler, golden interpreter and stock programs.

The vector unit is a microcoded 16-lane SIMD engine working on the scratchpad:
- 8 vector registers V0..V7, 16 lanes x int32 (LDV sign-extends int16 beats, LDW loads
  8 x int32 into lanes 0..7).
- 8 scalar registers S0..S7, 48-bit signed (reduction accumulators need > 32 bits).
- 8 argument registers A0..A7 written via CSR before VECRUN (addresses, lengths, shifts).
- One hardware loop (LOOPS count / LOOPE).
- Q15 LUT ops (EXP2F/RSQRT/RECIP) with linear interpolation, tables from
  trixium.model.bitexact (identical ROM contents in RTL, interpreter and firmware).

The INTERPRETER IS THE GOLDEN MODEL: the RTL must execute the same binary with identical
results; programs are validated against numpy/float references here first (accuracy study),
then the RTL against the interpreter (bit-exact).

Instruction word (32 bits):
  [31:26] opcode | [25:23] vd | [22:20] va | [19:17] vb | [16:14] sd | [13:11] sa | [10:0] imm
imm is signed 11-bit unless noted. Memory ops use byte addresses in scalar regs,
post-incremented by 32 (one 256-bit beat).
"""

import numpy as np

from trixium.model import bitexact as bx

LANES = 16

# BonsaiVectorUnit.imem_depth: the programs are uploaded into a 256-word instruction BRAM at
# init. Overflowing it does NOT fail loudly - the write pointer wraps and quietly overwrites
# the programs at offset 0 (one added program once corrupted resid_add's ARG setup, and only
# that program's CRC moved). emit_c() asserts against this, so keep it in step with the RTL.
IMEM_DEPTH = 512 # 256 until 2026-08-29 (V1 pipeline; 512 x 32 distributed RAM).

# Opcodes ------------------------------------------------------------------------------------------

OPCODES = [
    "NOP",    "DONE",
    # Scalar.
    "MOVSI",  # sd = imm (sign-extended).
    "ARG",    # sd = args[imm].
    "SADDI",  # sd += imm.
    "SMOV",   # sd = S[sa].
    # Loop.
    "LOOPS",  # loop count = S[sa] if imm == 0 else imm; loop body starts at next instr.
    "LOOPE",  # if --count > 0: jump to loop start.
    # Memory (S[sa] = byte address, post-incremented by 32).
    "LDV",    # vd = 16 x int16 (sign-extended).
    "LDW",    # vd lanes 0..7 = 8 x int32 (lanes 8..15 = 0).
    "STV",    # store 16 lanes as int16 (32 bytes).
    "STB",    # store 16 lanes as int8 (16 bytes; S[sa] += 16).
    # Lanewise ALU (int32, wrap on overflow unless noted).
    "BCAST",  # vd = broadcast S[sa] (low 32 bits).
    "VADD", "VSUB", "VMUL",  # VMUL = low 32 bits of product.
    "VMULH15",               # (a*b) >> 15 (arith), for Q15 math.
    "VMIN", "VMAX",
    "VSHRI",  # vd = va >> imm (arith).
    "VSHLI",  # vd = va << imm.
    "VSHLV",  # vd = va shifted by lane amounts in vb: left if >= 0, arith right if < 0 (clamped +-31).
    "VRNE",   # vd = rne_shift(va, imm) (round-nearest-even right shift).
    "VCLAMP8",   # clamp to [-127, 127].
    "VCLAMP16",  # clamp to [-32767, 32767].
    "VABS",
    "VCLZN",  # vd = shift s such that (va << s) normalizes |va| into [2^14, 2^15) (0 for va == 0).
    "VPACK",  # vd = lanes 0..7 of va and vb interleaved into 16 lanes (va[0..7] -> 0..7, vb[0..7] -> 8..15).
    # Reductions (accumulate into sd, 48-bit).
    "REDSUM",   # sd += sum(lanes)
    "REDSSQ",   # sd += sum(lanes^2)
    "REDMAX",   # sd = max(sd, max(|lanes|))
    # Q15 LUTs with linear interpolation (input low 15 bits used as documented per table).
    "LUTE2",    # vd = exp2f(va & 0x7FFF)   : 2^f, f in [0,1) Q15 -> Q15 result [2^15, 2^16).
    "LUTRECIP", # vd = recip(va)            : va Q15 mantissa in [2^14, 2^15) meaning [0.5,1) -> Q15 of 1/m in (2^15, 2^16].
    "LUTRSQRT", # vd = rsqrt(va)            : va Q30-normalized in [2^30, 2^32) meaning [1,4) -> Q15 of 1/sqrt.
    # Scalar (48-bit) ops added for rmsnorm (M4).
    "SADD",     # sd += S[sa].
    "SSHLV",    # sd shifted by S[sa] (left if >= 0, arith right if < 0; clamped +-47; wraps to 48 bits).
    "SNORME",   # sd = EVEN shift e such that (S[sa] << e) lands in [2^30, 2^32); 0 for S[sa] <= 0.
                # (Even so the shift splits cleanly into the rsqrt exponent half; the caller
                #  guarantees S[sa] > 0 - firmware clamps the eps term to >= 1.)
    # M8 additions (softmax/reduce programs).
    "REDSMAX",  # sd = max(sd, max(lanes)) SIGNED (REDMAX is an abs-max).
    "STW",      # store lanes 0..7 as 8 x int32 (32 bytes; S[sa] += 32).
    # Scalar-B ALU forms: b comes from S[sa] instead of a vector register. The BCAST+V<op>
    # idiom costs 1-3 body instructions per use to materialise a per-iteration-constant
    # vector; at 4 cycles/instruction on a non-overlapped FSM that idiom was 8 of
    # silu_mul's 36 body instructions. RTL cost: one 2:1 mux on the shared b-lane feed.
    "VSHLS",  # vd = va shifted lanewise by S[sa]: left if >= 0, arith right if < 0 (+-31).
    "VMULS",  # vd = low 32 bits of va * S[sa].
]
OP = {name: i for i, name in enumerate(OPCODES)}

MASK32 = 0xFFFFFFFF
MASK48 = 0xFFFFFFFFFFFF

def _s32(v):
    v = int(v) & MASK32
    return v - (1 << 32) if v & (1 << 31) else v

def _s48(v):
    v = int(v) & MASK48
    return v - (1 << 48) if v & (1 << 47) else v

# Assembler ----------------------------------------------------------------------------------------

def _reg(name, prefix):
    assert isinstance(name, str) and name[0].lower() == prefix and 0 <= int(name[1:]) <= 7, name
    return int(name[1:])

def assemble(program):
    """program: list of tuples (op, operands...) with named regs 'v0'..'v7'/'s0'..'s7' and
    python-int immediates. Returns list of 32-bit words."""
    words = []
    for ins in program:
        op = ins[0].upper()
        vd = va = vb = sd = sa = 0
        imm = 0
        args = list(ins[1:])
        def take_v():  return _reg(args.pop(0), "v")
        def take_s():  return _reg(args.pop(0), "s")
        def take_i():  return int(args.pop(0))
        if op in ("NOP", "DONE"):
            pass
        elif op in ("MOVSI", "ARG", "SADDI"):
            sd  = take_s()
            imm = take_i()
        elif op in ("SMOV", "SADD", "SSHLV", "SNORME"):
            sd = take_s(); sa = take_s()
        elif op == "LOOPS":
            if isinstance(args[0], str):
                sa = take_s()
            else:
                imm = take_i()
        elif op == "LOOPE":
            pass
        elif op in ("LDV", "LDW"):
            vd = take_v(); sa = take_s()
        elif op in ("STV", "STB", "STW"):
            va = take_v(); sa = take_s()
        elif op == "BCAST":
            vd = take_v(); sa = take_s()
        elif op in ("VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHLV", "VPACK"):
            vd = take_v(); va = take_v(); vb = take_v()
        elif op in ("VSHRI", "VSHLI", "VRNE"):
            vd = take_v(); va = take_v(); imm = take_i()
        elif op in ("VSHLS", "VMULS"):
            vd = take_v(); va = take_v(); sa = take_s()
        elif op in ("VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "LUTE2", "LUTRECIP", "LUTRSQRT"):
            vd = take_v(); va = take_v()
        elif op in ("REDSUM", "REDSSQ", "REDMAX", "REDSMAX"):
            sd = take_s(); va = take_v()
        else:
            raise ValueError(op)
        assert not args, f"extra operands in {ins}"
        assert -1024 <= imm <= 1023 or op in ("MOVSI", "LOOPS"), f"imm out of range in {ins}"
        if op in ("MOVSI", "LOOPS"):
            assert -1024 <= imm <= 1023, f"imm out of range in {ins} (11-bit)"
        words.append((OP[op] << 26) | (vd << 23) | (va << 20) | (vb << 17)
                     | (sd << 14) | (sa << 11) | (imm & 0x7FF))
    return words

# Interpreter (the golden model) -------------------------------------------------------------------

_EXP2  = bx.exp2_lut()   # 256 x Q15 for f in [0,1).
_RECIP = bx.recip_lut()  # 256 x Q15 for m in [1,2).
_RSQRT = bx.rsqrt_lut()  # 256 x Q15 for x in [1,4).

def _lut_interp(table, idx8, frac, next_edge):
    """Linear interpolation: table[idx8] + ((next - table[idx8]) * frac) >> shift, where
    frac has 7 bits (memory ops use 15-bit inputs: idx8 = top 8 bits, frac = low 7)."""
    a = int(table[idx8])
    b = int(table[idx8 + 1]) if idx8 < 255 else next_edge
    return a + (((b - a)*frac) >> 7)

def exp2f_q15(x):
    """2^f for f = (x & 0x7FFF)/2^15 -> Q15 in [2^15, 2^16)."""
    x = int(x) & 0x7FFF
    return _lut_interp(_EXP2, x >> 7, x & 0x7F, 1 << 16)

def recip_q15(m):
    """1/m for Q15 mantissa m in [2^14, 2^15) (value [0.5, 1)) -> Q15 of (1/value)/2 in [2^14, 2^15].
    Table is indexed over [1,2): m14 in [0.5,1) maps as m*2 in [1,2)."""
    m = int(m) & 0x7FFF
    idx = (m - (1 << 14)) >> 6          # 8-bit index over [2^14, 2^15).
    frac = (m - (1 << 14)) & 0x3F
    a = int(_RECIP[idx])
    b = int(_RECIP[idx + 1]) if idx < 255 else (1 << 14)
    return a + (((b - a)*frac) >> 6)

def rsqrt_q15(x32):
    """1/sqrt(v) for v = x32/2^30 in [1,4) -> Q15. MSB-split index (see bx.rsqrt_lut)."""
    x = int(x32) & MASK32
    if x < (1 << 31):                   # [1, 2): 128 entries, step 1/128.
        idx  = (x - (1 << 30)) >> 23
        frac = (x >> 16) & 0x7F
    else:                               # [2, 4): 128 entries, step 1/64.
        idx  = 128 + ((x - (1 << 31)) >> 24)
        frac = (x >> 17) & 0x7F
    a = int(_RSQRT[idx])
    b = int(_RSQRT[idx + 1]) if idx < 255 else int(round((2**15)/2.0))
    return a + (((b - a)*frac) >> 7)

class VecInterp:
    """Executes assembled binaries against a bytearray scratchpad. Bit-exact spec for the RTL."""

    def __init__(self, scratchpad):
        self.sp = scratchpad # bytearray.

    def run(self, words, args, max_steps=1_000_000, s7_seed=0):
        # s7_seed models the sequencer's res6-chaining (S7 starts as the previous
        # program's res6 under direct drive; 0 for CSR dispatch).
        V = [np.zeros(LANES, dtype=np.int64) for _ in range(8)]
        S = [0]*7 + [_s48(s7_seed)]
        A = list(args) + [0]*(8 - len(args))
        pc = 0
        loop_start = loop_cnt = 0
        steps = 0
        while steps < max_steps:
            steps += 1
            w = words[pc]
            op   = OPCODES[(w >> 26) & 0x3F]
            vd   = (w >> 23) & 7
            va   = (w >> 20) & 7
            vb   = (w >> 17) & 7
            sd   = (w >> 14) & 7
            sa   = (w >> 11) & 7
            imm  = w & 0x7FF
            if imm & 0x400:
                imm -= 0x800
            npc = pc + 1

            if op == "NOP":
                pass
            elif op == "DONE":
                return S, steps
            elif op == "MOVSI":
                S[sd] = imm
            elif op == "ARG":
                S[sd] = _s32(A[imm & 7]) # Args are 32-bit CSRs: sign-extend into the scalar.
            elif op == "SADDI":
                S[sd] = _s48(S[sd] + imm)
            elif op == "SMOV":
                S[sd] = S[sa]
            elif op == "SADD":
                S[sd] = _s48(S[sd] + S[sa])
            elif op == "SSHLV":
                amt = max(-47, min(47, S[sa]))
                S[sd] = _s48(S[sd] << amt) if amt >= 0 else _s48(S[sd] >> -amt)
            elif op == "SNORME":
                v = S[sa]
                if v <= 0:
                    S[sd] = 0
                else:
                    p = v.bit_length() - 1
                    S[sd] = (30 - p) if (p % 2 == 0) else (31 - p)
            elif op == "LOOPS":
                loop_cnt   = (S[sa] & MASK32) if imm == 0 else imm
                loop_start = npc
            elif op == "LOOPE":
                loop_cnt -= 1
                if loop_cnt > 0:
                    npc = loop_start
            elif op == "LDV":
                a = S[sa] & 0xFFFFFF
                assert a % 32 == 0, f"LDV address {a:#x} not 32-byte aligned (RTL is beat-addressed)"
                V[vd] = np.frombuffer(bytes(self.sp[a:a+32]), dtype="<i2").astype(np.int64)
                S[sa] = _s48(S[sa] + 32)
            elif op == "LDW":
                a = S[sa] & 0xFFFFFF
                assert a % 32 == 0, f"LDW address {a:#x} not 32-byte aligned (RTL is beat-addressed)"
                v = np.zeros(LANES, dtype=np.int64)
                v[:8] = np.frombuffer(bytes(self.sp[a:a+32]), dtype="<i4").astype(np.int64)
                V[vd] = v
                S[sa] = _s48(S[sa] + 32)
            elif op == "STV":
                a = S[sa] & 0xFFFFFF
                assert a % 32 == 0, f"STV address {a:#x} not 32-byte aligned (RTL is beat-addressed)"
                self.sp[a:a+32] = np.asarray([_s32(x) for x in V[va]], dtype="<i4") \
                    .astype("<i2").tobytes()
                S[sa] = _s48(S[sa] + 32)
            elif op == "STB":
                a = S[sa] & 0xFFFFFF
                assert a % 16 == 0, f"STB address {a:#x} not 16-byte aligned (RTL writes half-beats)"
                self.sp[a:a+16] = np.asarray([_s32(x) for x in V[va]], dtype="<i4") \
                    .astype("<i1").tobytes()
                S[sa] = _s48(S[sa] + 16)
            elif op == "BCAST":
                V[vd] = np.full(LANES, _s32(S[sa]), dtype=np.int64)
            elif op in ("VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHLV", "VPACK",
                        "VSHLS", "VMULS"):
                a = V[va]
                b = (np.full(LANES, _s32(S[sa]), dtype=np.int64)
                     if op in ("VSHLS", "VMULS") else V[vb])
                if   op == "VADD":    r = a + b
                elif op == "VSUB":    r = a - b
                elif op == "VMUL":    r = a*b
                elif op == "VMULH15": r = (a*b) >> 15
                elif op == "VMIN":    r = np.minimum(a, b)
                elif op == "VMAX":    r = np.maximum(a, b)
                elif op in ("VSHLV", "VSHLS"):
                    amt = np.clip(b, -31, 31)
                    r = np.where(amt >= 0, a << np.maximum(amt, 0), a >> np.maximum(-amt, 0))
                elif op == "VMULS":  r = a*b
                elif op == "VPACK":
                    r = np.concatenate([a[:8], b[:8]])
                V[vd] = np.asarray([_s32(x) for x in r], dtype=np.int64)
            elif op == "VSHRI":
                V[vd] = np.asarray([_s32(_s32(x) >> imm) for x in V[va]], dtype=np.int64)
            elif op == "VSHLI":
                V[vd] = np.asarray([_s32(_s32(x) << imm) for x in V[va]], dtype=np.int64)
            elif op == "VRNE":
                V[vd] = np.asarray([_s32(bx.rne_shift(_s32(x), imm)) for x in V[va]],
                                   dtype=np.int64)
            elif op == "VCLAMP8":
                V[vd] = np.clip(V[va], -127, 127)
            elif op == "VCLAMP16":
                V[vd] = np.clip(V[va], -32767, 32767)
            elif op == "VABS":
                V[vd] = np.abs([_s32(x) for x in V[va]])
            elif op == "VCLZN":
                r = []
                for x in V[va]:
                    x = abs(_s32(x))
                    r.append(0 if x == 0 else 14 - (x.bit_length() - 1))
                V[vd] = np.asarray(r, dtype=np.int64)
            elif op == "REDSUM":
                S[sd] = _s48(S[sd] + int(np.sum([_s32(x) for x in V[va]])))
            elif op == "REDSSQ":
                S[sd] = _s48(S[sd] + int(np.sum([_s32(x)*_s32(x) for x in V[va]])))
            elif op == "REDMAX":
                S[sd] = max(S[sd], int(np.max(np.abs([_s32(x) for x in V[va]]))))
            elif op == "REDSMAX":
                S[sd] = max(S[sd], int(np.max([_s32(x) for x in V[va]])))
            elif op == "STW":
                a = S[sa] & 0xFFFFFF
                assert a % 32 == 0, f"STW address {a:#x} not 32-byte aligned (RTL is beat-addressed)"
                self.sp[a:a+32] = np.asarray([_s32(x) for x in V[va][:8]],
                                             dtype="<i4").tobytes()
                S[sa] = _s48(S[sa] + 32)
            elif op == "LUTE2":
                V[vd] = np.asarray([exp2f_q15(x) for x in V[va]], dtype=np.int64)
            elif op == "LUTRECIP":
                V[vd] = np.asarray([recip_q15(x) for x in V[va]], dtype=np.int64)
            elif op == "LUTRSQRT":
                V[vd] = np.asarray([rsqrt_q15(x) for x in V[va]], dtype=np.int64)
            else:
                raise ValueError(op)
            pc = npc
        raise RuntimeError("max_steps exceeded")

# Stock programs -----------------------------------------------------------------------------------
#
# Calling convention (args A0..A7 set via CSR):
#   A0/A1/A2 = source/source2/destination byte addresses, A3 = beat count,
#   A4..A7   = op-specific shifts/constants computed by the firmware exponent algebra.

def prog_resid_add():
    """dst_i16 = clamp16((a << A4) + (b << A5)), shifts signed (VSHLV: left if >= 0).

    Firmware picks A4/A5 = e_a - e_out / e_b - e_out from the operands' *measured* absmax,
    so the residual keeps its full 15-bit mantissa. Taking e_out = max(e_a, e_b) + 1 instead
    - the bound for two full mantissas - spends a bit on every call, and the loop calls this
    twice per layer: the 1.7B's residual was down to one bit by layer 11.

    Returns S6 = output absmax, which is what lets the *next* call size its own shifts."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5),
        ("MOVSI", "s6", 0),
        # Both shift amounts are loop constants; broadcasting them per iteration cost 2 of a
        # 10-instruction body. v2/v3 stay live across the loop instead.
        ("BCAST", "v2", "s4"),
        ("BCAST", "v3", "s5"),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("LDV",   "v1", "s1"),
        ("VSHLV", "v0", "v0", "v2"),
        ("VSHLV", "v1", "v1", "v3"),
        ("VADD",  "v0", "v0", "v1"),
        ("VCLAMP16", "v0", "v0"),
        ("REDMAX", "s6", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_quant8():
    """int16 BFP -> int8 matvec input: dst_i8 = clamp8(rne(src >> s)), out exponent = e + s.

    A1 selects where s comes from:
      A1 = 1  measure it (s = bitlen(absmax) - 7, a first pass over the source). After
              rmsnorm the mantissas sit near 2**11, so the fixed 8 below would leave the
              matvec int8 inputs of about +-9 - 3 of the 7 bits available - and cost ~0.30
              rel-rms on every matvec output. s may be negative: a small source scales up.
      A1 = 0  fixed s = 8. The attention path needs the shift known *before* the data
              exists, because the sequencer patches the softmax scale (which folds in the q
              exponent) into a program it then runs head after head with no CPU in the loop.

    VCLZN(absmax) is exactly the pre-shift wanted - it normalizes the absmax into
    [2^14, 2^15), i.e. shifts left by 15 - bitlen - and keeping the fixed RNE at bit 8 then
    rounds at the same place as rne_shift(src, bitlen - 7), with no variable-rounding
    microcode. Multiplying it by A1 selects the mode without a branch (the ISA has none).

      A0 src, A1 mode, A2 dst, A3 beats.  Returns S6 = source absmax (firmware derives the
      same s from it: 8 if absmax == 0 else bitlen(absmax) - 7)."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("MOVSI", "s6", 0),
        ("LOOPS", "s3"),               # Pass 1: absmax.
        ("LDV",   "v0", "s0"),
        ("REDMAX", "s6", "v0"),
        ("LOOPE",),
        ("BCAST", "v1", "s6"),
        ("VCLZN", "v1", "v1"),         # 15 - bitlen(absmax) in every lane.
        ("BCAST", "v2", "s1"),
        ("VMUL",  "v1", "v1", "v2"),   # x1 = measure, x0 = fixed >>8.
        ("ARG",   "s0", 0),            # Rewind src.
        ("LOOPS", "s3"),               # Pass 2: scale, round, store.
        ("LDV",   "v0", "s0"),
        ("VSHLV", "v0", "v0", "v1"),
        ("VRNE",  "v0", "v0", 8),
        ("VCLAMP8", "v0", "v0"),
        ("STB",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_norm_i32():
    """Engine int32 results -> normalized int16 BFP (floor-shift semantics; the interpreter
    IS the spec). A0 src (int32), A2 dst, A3 = loop count = source beats / 2 (two 8xint32
    source beats -> one 16xint16 beat out), A4 = right-shift k >= 0 (firmware computes
    bitlen(absmax) - 14 from the engine absmax tracker; output exponent += k).

    CONTRACT (2026-08-29): k MUST come from the absmax of the source - the engine tracker,
    a spilled copy of it, or a CPU scan - i.e. bitlen(absmax) - 14. Then |v >> k| < 2^14
    for every lane, so the old VCLAMP16 could never fire and was dropped; an out-of-
    contract smaller k now wraps in the int16 store instead of saturating. The old REDMAX
    into S6 (output absmax) had no consumer anywhere - every KRES6 reader follows a spill
    and both CPU call sites discard the result - so it was dropped too. Body 8 -> 6
    instructions on a 4-cycle-per-instruction unit: -25% on the top op of the token.
    Returns S6 = 0."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3), ("ARG", "s4", 4),
        ("MOVSI", "s6", 0),
        ("MOVSI", "s5", 0),
        # -k is a loop constant: build it once. It used to cost two BCASTs and a VSUB on every
        # iteration, 3 of a 10-instruction body, for a value that never changes.
        ("BCAST", "v3", "s5"),          # 0
        ("BCAST", "v2", "s4"),
        ("VSUB",  "v2", "v3", "v2"),    # -k
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s0"),
        ("VPACK", "v4", "v0", "v1"),
        ("VSHLV", "v4", "v4", "v2"),    # arith shift right by k (floor)
        ("STV",   "v4", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_silu_mul():
    """dst_i16 = clamp16(silu(gate) * up >> 15-ish), all Q15 fixed point.
    gate value = mg * 2^eg, up value = mu * 2^eu. Firmware supplies:
      A4 = eg + 6, the DIRECT (usually negative => right) shift for f = (mg*C_LOG2E) << A4.
           CONVENTION CHANGE 2026-08-28: this used to be -(eg+6), negated in-loop; VSHLS
           shifts by the scalar directly, so the negation moved to the caller.
      A6 = output shift so that dst mantissa = (sig_q15 * mg * mu-ish) fits int16 (direct).
    Body is 21 instructions (was 27 on 2026-08-28, 36 before that). 2026-08-29: the two
    clamp constants are hoisted - v1 = -15*2^15 and v7 = 14*2^15 live across the loop; the
    register plan that frees them: VCLZN's n goes to v4 (i = floor(f) is dead after the
    e2 shift) and mu is loaded late into v3 (f is dead after LUTE2). Same arithmetic,
    instruction for instruction, as the pinned pre-hoist program in test_asm_hoist.
    """
    C_LOG2E = 739 # round(log2(e) * 2^9) - fits the 11-bit immediate; A4 folds the 2^9 scale.
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s6", 6),
        ("MOVSI", "s5", C_LOG2E),
        ("MOVSI", "s7", 1),    ("BCAST", "v2", "s7"), ("VSHLI", "v2", "v2", 15), # v2 = 2^15
        ("MOVSI", "s7", -480), ("BCAST", "v1", "s7"), ("VSHLI", "v1", "v1", 10), # -15*2^15
        ("MOVSI", "s7", 448),  ("BCAST", "v7", "s7"), ("VSHLI", "v7", "v7", 10), # 14*2^15
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),          # mg
        ("VMULS", "v3", "v0", "s5"),    # mg * C_LOG2E
        ("VSHLS", "v3", "v3", "s4"),    # f (Q15): << A4 (negative = right shift)
        ("VMAX",  "v3", "v3", "v1"),    # clamp low at -15*2^15
        ("VMIN",  "v3", "v3", "v7"),    # clamp high at 14*2^15 (e2 <= 2^29: int32-safe)
        ("VSHRI", "v4", "v3", 15),      # i = floor(f)
        ("LUTE2", "v5", "v3"),          # 2^r (LUTE2 masks to low 15 bits itself)
        ("VSHLV", "v5", "v5", "v4"),    # e2 = 2^f (Q15 integer-scaled); i dead from here
        ("VADD",  "v6", "v5", "v2"),    # den = e2 + 2^15 (v2 persistent)
        ("VCLZN", "v4", "v6"),          # n (<= 0: den >= 2^15)
        ("VSHLV", "v6", "v6", "v4"),    # mantissa in [2^14, 2^15)
        ("LUTRECIP", "v6", "v6"),
        ("VMULH15", "v5", "v5", "v6"),  # (e2 * rec) >> 15
        ("VSHLV", "v5", "v5", "v4"),
        ("VSHLI", "v5", "v5", 1),       # Q14 -> Q15
        ("VMULH15", "v0", "v0", "v5"),
        ("LDV",   "v3", "s1"),          # mu (late: f is dead after LUTE2)
        ("VMUL",  "v0", "v0", "v3"),
        ("VSHLS", "v0", "v0", "s6"),    # >> A6
        ("VCLAMP16", "v0", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_rmsnorm():
    """dst_i16 @ E_NORM = clamp16(round_half_up(m * w * rsqrt(mean(m^2) + eps))).

    Two passes over the source: REDSSQ then apply. Source m is int16 @ es, weights w are
    int16 @ e_w (the image's NORM16 rows, fixed exponent). All the es-dependence folds into
    two firmware-computed args; the normalize shift e from SNORME carries the data-dependent
    part (out = m*w*r_q15 * 2^(A6 + e/2) with the >>15 of VMULH15 folded into A6).

      A0 = src, A1 = w, A2 = dst, A3 = beats (n/16)
      A4 = -log2(n)                                (n is always a power of two here)
      A5 = eps_int = max(1, round(eps * 2^(-2*es)))  (>= 1 keeps SNORME's input positive)
      A6 = e_w - E_NORM - 15

    Final shift K = A6 + e/2 is always negative for sane inputs (right shift with
    round-half-up bias; half-up not RNE - the interpreter is the spec for ties)."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5), ("ARG", "s6", 6),
        ("MOVSI", "s7", 0),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("REDSSQ", "s7", "v0"),
        ("LOOPE",),
        ("SSHLV", "s7", "s4"),      # ss / n (A4 negative).
        ("SADD",  "s7", "s5"),      # + eps_int.
        ("SNORME", "s4", "s7"),     # e (even).
        ("SSHLV", "s7", "s4"),      # normalize into [2^30, 2^32).
        ("BCAST", "v7", "s7"),
        ("LUTRSQRT", "v7", "v7"),   # r_q15 in every lane.
        ("MOVSI", "s5", -1),
        ("SSHLV", "s4", "s5"),      # e/2 (e even: arith halving is exact).
        ("SADD",  "s4", "s6"),      # K = A6 + e/2.
        ("BCAST", "v6", "s4"),      # K lanes.
        ("VSUB",  "v5", "v5", "v5"),
        ("VSUB",  "v5", "v5", "v6"),  # -K.
        ("MOVSI", "s5", 1),
        ("BCAST", "v4", "s5"),        # ones.
        ("VSUB",  "v3", "v5", "v4"),  # -K - 1.
        ("VSHLV", "v4", "v4", "v3"),  # bias = 1 << (-K-1) (0 if K >= 0).
        ("ARG",   "s0", 0),           # rewind src.
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),        # m.
        ("LDV",   "v1", "s1"),        # w.
        ("VMUL",  "v2", "v0", "v1"),  # m*w (<= 2^30: no wrap).
        ("VMULH15", "v2", "v2", "v7"),# * r_q15 >> 15 (<= 2^30 post-shift: no wrap).
        ("VADD",  "v2", "v2", "v4"),  # + bias.
        ("VSHLV", "v2", "v2", "v6"),  # >> -K.
        ("VCLAMP16", "v2", "v2"),
        ("STV",   "v2", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_softmax8():
    """int32 scores (position-major, padded to a 16-multiple with INT32_MIN) -> int8 probs
    relative to the max entry: p8_i = clamp8(rne(2^((s_i - smax)*log2e*2^sigma) * 2^15 >> 8)).
    The 1/sum normalization is NOT applied here - S6 returns the exact sum of the emitted
    p8 integers (the firmware folds 1/S6 into the attention scale and uses S6 as the AXPY
    Rpre bound); S7 returns smax.

      A0 = scores, A2 = p8 dst, A3 = blocks (padded/16)
      A4 = -s1 (score pre-shift, s1 >= 1: keeps the subtract and *739 in int32;
                firmware picks s1 = max(1, 22 - floor(log2(2^31)) + ...) = enough that
                |s>>s1| <= 2^21)
      A1 = C: 739 = round(log2(e)*2^9) for even log2(head_dim); 523 = round(739/sqrt(2))
                when the 1/sqrt(head_dim) attention scale carries a half-power of two.
      A5 = -s2 with s1 + s2 = -(sigma + 6): f = ((s>>s1 - smax>>s1)*C) >> s2 is Q15 of
                (s - smax)*2^sigma*log2(e), sigma including the whole-power part of the
                attention scale.

    Padding entries clamp to f = -15*2^15 -> p8 = 0.

      A6 = spill address (32 B beat): the exact p8 sum (S6) is ALSO written there as 8 int32
           copies, replacing the separate PROG_SPILL descriptor the attention list used to
           run after every softmax (F0-5, 2026-08-29). S6/S7 are unchanged by the tail."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s3", 3),
        ("MOVSI", "s5", 47),
        ("MOVSI", "s7", -1),
        ("SSHLV", "s7", "s5"),          # smax accumulator = -2^47.
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s0"),
        ("VPACK", "v0", "v0", "v1"),
        ("REDSMAX", "s7", "v0"),
        ("LOOPE",),
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5),
        ("MOVSI", "s6", 0),
        ("BCAST", "v7", "s7"),          # smax.
        ("BCAST", "v6", "s4"),
        ("VSHLV", "v7", "v7", "v6"),    # smax >> s1.
        ("ARG",   "s1", 1),
        ("BCAST", "v5", "s1"),
        ("MOVSI", "s1", -480),
        ("BCAST", "v4", "s1"),
        ("VSHLI", "v4", "v4", 10),      # -15*2^15.
        # Both shift amounts are loop constants; they used to be rebuilt into v1 on every
        # iteration of a 19-instruction body. v3/v6 are free across this loop.
        ("BCAST", "v3", "s4"),
        ("BCAST", "v6", "s5"),
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s0"),
        ("VPACK", "v0", "v0", "v1"),
        ("VSHLV", "v0", "v0", "v3"),    # s >> s1.
        ("VSUB",  "v0", "v0", "v7"),    # d' <= 0.
        ("VMUL",  "v0", "v0", "v5"),    # d'*739.
        ("VSHLV", "v0", "v0", "v6"),    # f (Q15).
        ("VMAX",  "v0", "v0", "v4"),    # clamp at -15*2^15.
        ("VSHRI", "v1", "v0", 15),      # i = floor(f) in [-15, 0].
        # Same as silu_mul: LUTE2 masks to the low 15 bits, so r = f - ((f>>15)<<15) was
        # two instructions per iteration computing what the LUT already does.
        ("LUTE2", "v2", "v0"),
        ("VSHLV", "v2", "v2", "v1"),    # 2^f, Q15.
        ("VRNE",  "v2", "v2", 8),
        ("VCLAMP8", "v2", "v2"),        # p8 in [0, 127] (max entry 128 -> 127).
        ("REDSUM", "s6", "v2"),         # Exact sum of emitted p8.
        ("STB",   "v2", "s2"),
        ("LOOPE",),
        ("ARG",   "s1", 6),             # Self-spill: 8 copies of S6 at A6 (was PROG_SPILL).
        ("BCAST", "v0", "s6"),
        ("STW",   "v0", "s1"),
        ("DONE",),
    ]

def prog_reduce_acc():
    """acc_i32 += src_i32 (8 int32 per iteration); S6 = |max| of the post-add values
    (the last call of an N-way reduce returns the true absmax for the norm shift).
    A0 = acc read ptr, A1 = src, A2 = acc write ptr (same address as A0), A3 = beats."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("MOVSI", "s6", 0),
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s1"),
        ("VADD",  "v0", "v0", "v1"),
        ("REDMAX", "s6", "v0"),
        ("STW",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_rope():
    """In-place rotate of one head's halves against Q14 cos/sin tables:
      out0 = clamp16(rne(x0*c - x1*s, 14)),  out1 = clamp16(rne(x0*s + x1*c, 14)).
    Tables are int16 Q14 (c14 = round(cos*mscale*2^14) - Q14 absorbs YaRN's mscale > 1),
    fetched from the image's rope_tab row for the position (half-split pairing:
    pair p = (x[p], x[p + hd/2])).
      A0 = x0 (first half), A1 = x1 (second half), A2 = c table, A3 = s table,
      A4 = beats (hd/2/16), A5 = x0 write ptr, A6 = x1 write ptr."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5), ("ARG", "s6", 6),
        ("LOOPS", "s4"),
        ("LDV",   "v0", "s0"),          # x0.
        ("LDV",   "v1", "s1"),          # x1.
        ("LDV",   "v2", "s2"),          # c.
        ("LDV",   "v3", "s3"),          # s.
        ("VMUL",  "v4", "v0", "v2"),    # x0*c.
        ("VMUL",  "v5", "v1", "v3"),    # x1*s.
        ("VSUB",  "v4", "v4", "v5"),
        ("VRNE",  "v4", "v4", 14),
        ("VCLAMP16", "v4", "v4"),
        ("VMUL",  "v5", "v0", "v3"),    # x0*s.
        ("VMUL",  "v6", "v1", "v2"),    # x1*c.
        ("VADD",  "v5", "v5", "v6"),
        ("VRNE",  "v5", "v5", 14),
        ("VCLAMP16", "v5", "v5"),
        ("STV",   "v4", "s5"),
        ("STV",   "v5", "s6"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_scale8():
    """int16 src -> int8 dst via one Q15 multiplier: dst = clamp8(rne(src*M >> 15)).
    M (A4, < 2^15) folds a per-head non-power-of-2 factor (e.g. 1/sum-of-p8 and the
    common-output-scale alignment) computed by the firmware; |src| <= 2^14 keeps the
    product int32-safe. A0 = src, A2 = dst, A3 = beats (n/16)."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3), ("ARG", "s4", 4),
        ("BCAST", "v3", "s4"),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("VMUL",  "v0", "v0", "v3"),
        ("VRNE",  "v0", "v0", 15),
        ("VCLAMP8", "v0", "v0"),
        ("STB",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_spill():
    """Store S7's low 32 bits (sequencer-seeded with the previous program's res6) to A0.
    Lets a descriptor list persist per-head hardware results (softmax sums, reduce
    absmaxes) for one CPU readout after the whole list completes. Writes 8 copies
    (one 32-byte beat)."""
    return [
        ("ARG",   "s0", 0),
        ("BCAST", "v0", "s7"),
        ("STW",   "v0", "s0"),
        ("SMOV",  "s6", "s7"),  # Preserve the chained value in res6 for the next program.
        ("DONE",),
    ]

PROGRAMS = {
    "resid_add":  prog_resid_add,
    "quant8":     prog_quant8,
    "norm_i32":   prog_norm_i32,
    "silu_mul":   prog_silu_mul,
    "rmsnorm":    prog_rmsnorm,
    "softmax8":   prog_softmax8,
    "reduce_acc": prog_reduce_acc,
    "rope":       prog_rope,
    "scale8":     prog_scale8,
    "spill":      prog_spill,
}

def emit_c(out):
    """Emit firmware/progs.h with assembled program words + entry offsets."""
    lines = ["// Generated by trixium.model.asm - do not edit.", "#pragma once",
             "#include <stdint.h>", ""]
    all_words = []
    offsets = {}
    for name, fn in PROGRAMS.items():
        offsets[name] = len(all_words)
        all_words += assemble(fn())
    assert len(all_words) <= IMEM_DEPTH, (
        f"programs need {len(all_words)} words but the instruction BRAM holds {IMEM_DEPTH}: "
        f"the upload would wrap and silently overwrite the first program")
    lines.append(f"static const uint32_t bonsai_progs[{len(all_words)}] = {{")
    for i in range(0, len(all_words), 8):
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in all_words[i:i+8]) + ",")
    lines.append("};")
    for name, off in offsets.items():
        lines.append(f"#define PROG_{name.upper()} {off}")
    lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    import sys
    emit_c(sys.argv[1] if len(sys.argv) > 1 else "firmware/progs.h")
    print("emitted")

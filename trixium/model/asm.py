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
        elif op == "SMOV":
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
        elif op in ("STV", "STB"):
            va = take_v(); sa = take_s()
        elif op == "BCAST":
            vd = take_v(); sa = take_s()
        elif op in ("VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHLV", "VPACK"):
            vd = take_v(); va = take_v(); vb = take_v()
        elif op in ("VSHRI", "VSHLI", "VRNE"):
            vd = take_v(); va = take_v(); imm = take_i()
        elif op in ("VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "LUTE2", "LUTRECIP", "LUTRSQRT"):
            vd = take_v(); va = take_v()
        elif op in ("REDSUM", "REDSSQ", "REDMAX"):
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
    """1/sqrt(v) for v = x32/2^30 in [1,4) -> Q15."""
    x = int(x32) & MASK32
    idx = (x - (1 << 30)) >> 22         # 8-bit index over [2^30, 2^32).
    frac = ((x - (1 << 30)) >> 15) & 0x7F
    a = int(_RSQRT[idx])
    b = int(_RSQRT[idx + 1]) if idx < 255 else int(round((2**15)/2.0))
    return a + (((b - a)*frac) >> 7)

class VecInterp:
    """Executes assembled binaries against a bytearray scratchpad. Bit-exact spec for the RTL."""

    def __init__(self, scratchpad):
        self.sp = scratchpad # bytearray.

    def run(self, words, args, max_steps=1_000_000):
        V = [np.zeros(LANES, dtype=np.int64) for _ in range(8)]
        S = [0]*8
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
                S[sd] = _s48(A[imm & 7])
            elif op == "SADDI":
                S[sd] = _s48(S[sd] + imm)
            elif op == "SMOV":
                S[sd] = S[sa]
            elif op == "LOOPS":
                loop_cnt   = (S[sa] & MASK32) if imm == 0 else imm
                loop_start = npc
            elif op == "LOOPE":
                loop_cnt -= 1
                if loop_cnt > 0:
                    npc = loop_start
            elif op == "LDV":
                a = S[sa] & 0xFFFFFF
                V[vd] = np.frombuffer(bytes(self.sp[a:a+32]), dtype="<i2").astype(np.int64)
                S[sa] = _s48(S[sa] + 32)
            elif op == "LDW":
                a = S[sa] & 0xFFFFFF
                v = np.zeros(LANES, dtype=np.int64)
                v[:8] = np.frombuffer(bytes(self.sp[a:a+32]), dtype="<i4").astype(np.int64)
                V[vd] = v
                S[sa] = _s48(S[sa] + 32)
            elif op == "STV":
                a = S[sa] & 0xFFFFFF
                self.sp[a:a+32] = np.asarray([_s32(x) for x in V[va]], dtype="<i4") \
                    .astype("<i2").tobytes()
                S[sa] = _s48(S[sa] + 32)
            elif op == "STB":
                a = S[sa] & 0xFFFFFF
                self.sp[a:a+16] = np.asarray([_s32(x) for x in V[va]], dtype="<i4") \
                    .astype("<i1").tobytes()
                S[sa] = _s48(S[sa] + 16)
            elif op == "BCAST":
                V[vd] = np.full(LANES, _s32(S[sa]), dtype=np.int64)
            elif op in ("VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHLV", "VPACK"):
                a, b = V[va], V[vb]
                if   op == "VADD":    r = a + b
                elif op == "VSUB":    r = a - b
                elif op == "VMUL":    r = a*b
                elif op == "VMULH15": r = (a*b) >> 15
                elif op == "VMIN":    r = np.minimum(a, b)
                elif op == "VMAX":    r = np.maximum(a, b)
                elif op == "VSHLV":
                    amt = np.clip(b, -31, 31)
                    r = np.where(amt >= 0, a << np.maximum(amt, 0), a >> np.maximum(-amt, 0))
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
    """dst_i16 = clamp16((a >> A4) + (b >> A5)) — conservative pre-shifts (firmware picks
    A4/A5 = e_out - e_a / e_out - e_b with e_out = max(e_a, e_b) + 1)."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("LDV",   "v1", "s1"),
        ("BCAST", "v2", "s4"), ("VSHLV", "v0", "v0", "v2"),
        ("BCAST", "v2", "s5"), ("VSHLV", "v1", "v1", "v2"),
        ("VADD",  "v0", "v0", "v1"),
        ("VCLAMP16", "v0", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_quant8():
    """int16 BFP -> int8 matvec input: dst_i8 = clamp8(rne(src >> 8)). A0 src, A2 dst, A3 beats."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
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
    Returns S6 = output absmax (post-shift sanity/renorm check)."""
    return [
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3), ("ARG", "s4", 4),
        ("MOVSI", "s6", 0),
        ("MOVSI", "s5", 0),
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s0"),
        ("VPACK", "v4", "v0", "v1"),
        ("BCAST", "v3", "s5"),          # 0
        ("BCAST", "v2", "s4"),
        ("VSUB",  "v2", "v3", "v2"),    # -k
        ("VSHLV", "v4", "v4", "v2"),    # arith shift right by k (floor)
        ("VCLAMP16", "v4", "v4"),
        ("REDMAX", "s6", "v4"),
        ("STV",   "v4", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

def prog_silu_mul():
    """dst_i16 = clamp16(silu(gate) * up >> 15-ish), all Q15 fixed point.
    gate value = mg * 2^A4 (A4 = e_gate), up value = mu * 2^A5. Firmware supplies:
      A4 = right-shift such that f = (mg * C_LOG2E) >> A4 is the Q15 value of gate*log2(e)
           (C_LOG2E = round(log2(e) * 2^9) = 739 => f_int = mg*739*2^(eg+6); A4 = -(eg+6)).
      A6 = output shift so that dst mantissa = (sig_q15 * mg * mu-ish) fits int16
    Pipeline per beat:
      f   = clamp((mg * C) >> A4, -15*2^15, 14*2^15)  # log2-domain input
      i   = f >> 15 (floor), r = f & 0x7FFF
      e2  = EXP2F(r) << i                              # 2^f, integer-scaled Q15
      den = e2 + 2^15                                  # 1 + 2^f
      n   = VCLZN(den); m = den << n (Q15 mantissa in [2^14, 2^15))
      rec = RECIP(m)                                   # Q15-ish of 1/m
      sig = (e2 * rec) >> 15, then >> (correction from n and den exponent) via VSHLV
      out = ((mg * sig) >> 15) * mu >> A6
    """
    C_LOG2E = 739 # round(log2(e) * 2^9) - fits the 11-bit immediate; A4 folds the 2^9 scale.
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s6", 6),
        ("MOVSI", "s5", C_LOG2E),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),          # mg
        ("LDV",   "v1", "s1"),          # mu
        # f = (mg * C_LOG2E) >> A4, clamped to [-15, 8] in Q15.
        ("BCAST", "v2", "s5"),
        ("VMUL",  "v3", "v0", "v2"),
        ("BCAST", "v2", "s4"),
        ("VSUB",  "v4", "v4", "v4"),
        ("VSUB",  "v2", "v4", "v2"),    # -A4
        ("VSHLV", "v3", "v3", "v2"),    # f (Q15)
        ("MOVSI", "s7", -480), ("BCAST", "v2", "s7"), ("VSHLI", "v2", "v2", 10),
        ("VMAX",  "v3", "v3", "v2"),    # clamp low at -480*2^10 = -15*2^15
        ("MOVSI", "s7", 448), ("BCAST", "v2", "s7"), ("VSHLI", "v2", "v2", 10),
        ("VMIN",  "v3", "v3", "v2"),    # clamp high at 14*2^15 (e2 <= 2^29: int32-safe)
        # split i/r; e2 = EXP2F(r) << i
        ("VSHRI", "v4", "v3", 15),      # i = floor(f)
        ("VSHLI", "v5", "v4", 15),
        ("VSUB",  "v5", "v3", "v5"),    # r = f - (i << 15) in [0, 2^15)
        ("LUTE2", "v5", "v5"),          # 2^r, Q15 in [2^15, 2^16)
        ("VSHLV", "v5", "v5", "v4"),    # e2 = 2^f (Q15 integer-scaled)
        # den = e2 + 2^15 ; normalize ; recip
        ("MOVSI", "s7", 1), ("BCAST", "v2", "s7"), ("VSHLI", "v2", "v2", 15),
        ("VADD",  "v6", "v5", "v2"),    # den
        ("VCLZN", "v7", "v6"),          # n (<= 0 here: den >= 2^15)
        ("VSHLV", "v6", "v6", "v7"),    # mantissa in [2^14, 2^15)
        ("LUTRECIP", "v6", "v6"),       # ~ (2^29)/den_mantissa, Q15 of 1/m in [2^14, 2^15]
        # sig = e2 * rec adjusted: sig_q15 = (e2 * rec) >> (15 - n)  (n <= 0)
        ("VMULH15", "v5", "v5", "v6"),  # (e2 * rec) >> 15
        ("VSHLV", "v5", "v5", "v7"),    # >> -n more  => sigmoid in Q14 (rec is Q15 of 1/(2m))
        ("VSHLI", "v5", "v5", 1),       # Q14 -> Q15
        # out = ((mg * sig) >> 15) * mu >> A6
        ("VMULH15", "v0", "v0", "v5"),
        ("VMUL",  "v0", "v0", "v1"),
        ("BCAST", "v2", "s6"),
        ("VSHLV", "v0", "v0", "v2"),    # >> A6 (A6 negative shift amount from firmware)
        ("VCLAMP16", "v0", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

PROGRAMS = {
    "resid_add": prog_resid_add,
    "quant8":    prog_quant8,
    "norm_i32":  prog_norm_i32,
    "silu_mul":  prog_silu_mul,
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

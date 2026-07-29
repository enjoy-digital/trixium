#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Bit-exact numerics specification for the TriXium engine.

This module IS the hardware specification: every function here defines, to the bit, what the
RTL (ternary_mac/requant/vector unit) must produce. The RTL testbenches compare against these
functions; the converter uses the bound/bias helpers to guarantee no accumulator saturation.

Value representation ("BFP vector"): int16 mantissas + one per-vector exponent e, value_i =
m_i * 2^e, normalized so absmax(m) is in [2^14, 2^15) (15-bit mantissas). Matvec inputs are
int8 (m >> 8 with round-to-nearest-even, exponent += 8).

Ternary matvec (per output row m):
  group_sum(g) = sum_{j in group g} (code-1) * x8_j                 exact int, |gs| <= 16256
  s_fp16(g)    = m11(g) * 2^e16(g)                                  exact fp16 decomposition
  contrib(g)   = (group_sum * m11) <<or>> (e16 + B)                 B = per-tensor scale bias
  acc          = sum_g contrib                                       int48, no saturation by
                                                                     construction (see bounds)
  result       = acc >> R                                            per-tensor result shift,
                                                                     fits int32 by construction
  semantic scale of result = 2^(ex - B + R), ex = exponent of x8.

All right shifts on signed values are arithmetic (floor); "RNE" = round to nearest, ties to
even, implemented as shift with round+sticky bits.
"""

import numpy as np

# Integer primitives -------------------------------------------------------------------------------

def rne_shift(v, k):
    """Round-to-nearest-even right shift by k (k >= 0). v: int array/scalar. Exact RTL spec:
    result = (v >> k) + round_up, round_up = round_bit & (sticky | lsb)."""
    v = np.asarray(v, dtype=np.int64)
    if k == 0:
        return v.copy()
    q      = v >> k                      # Floor shift.
    round_ = (v >> (k - 1)) & 1          # Round bit.
    sticky = (v & ((1 << (k - 1)) - 1)) != 0 if k > 1 else np.zeros_like(v, dtype=bool)
    lsb    = q & 1
    return q + (round_ & (sticky | (lsb == 1)).astype(np.int64))

def arith_shift(v, k):
    """Shift by k: left if k >= 0 (exact), arithmetic right (floor) if k < 0."""
    v = np.asarray(v, dtype=np.int64)
    return v << k if k >= 0 else v >> (-k)

def clamp(v, lo, hi):
    return np.clip(v, lo, hi)

# fp16 decomposition -------------------------------------------------------------------------------

FP16_EXP_BIAS = 25 # value = m11 * 2^(E - 25) for normals (m11 includes the implicit bit).

def fp16_decompose(s):
    """fp16 -> (m11 signed int, e16 int): value = m11 * 2^e16 exactly.
    Normal:    m11 = +-(1024 + frac), e16 = E - 25.
    Subnormal: m11 = +-frac,          e16 = -24.  Zero: (0, -24). Inf/NaN unsupported."""
    s    = np.asarray(s, dtype=np.float16)
    bits = s.view(np.uint16).astype(np.int64)
    sign = np.where(bits & 0x8000, -1, 1)
    E    = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    assert not np.any(E == 0x1F), "fp16 inf/nan in scales"
    m11 = np.where(E > 0, 1024 + frac, frac)*sign
    e16 = np.where(E > 0, E - FP16_EXP_BIAS, -24)
    return m11, e16

# Per-tensor bias/shift selection (used by the converter; stored in the manifest) ------------------

ACC_BITS    = 48
RESULT_BITS = 32
GS_MAX      = 128*127     # Max |group_sum|, ternary mode (128 lanes x |x8|).
GS_MAX_Q8   = 32*127*127  # Max |group_sum|, Q8 row-dot mode (32 lanes x |w8|x|x8|).

def choose_scale_bias(e16_all, target_top=12):
    """Pick B so max(e16) + B == target_top: largest-scale groups get a 12-bit left shift,
    bounding contributions to |gs*m11| * 2^12 < 2^37 and 96-group sums < 2^44 (int48 safe)."""
    return int(target_top - np.max(e16_all))

def tensor_acc_bound(scales_f16, B, gs_max=GS_MAX):
    """Worst-case |acc| over all rows of a tensor. scales_f16: (rows, G) fp16."""
    m11, e16 = fp16_decompose(scales_f16)
    shift    = e16 + B
    # Bound per group: gs_max * |m11| shifted; sum per row.
    mag = np.abs(m11).astype(np.float64)*gs_max*np.exp2(shift.astype(np.float64))
    per_row = np.ceil(mag).sum(axis=1)
    return per_row.max()

def choose_result_shift(scales_f16, B, gs_max=GS_MAX):
    """Smallest R such that (acc >> R) always fits int32."""
    bound = tensor_acc_bound(scales_f16, B, gs_max)
    R = 0
    while bound/(1 << R) >= 2**(RESULT_BITS - 1):
        R += 1
    return R

# Ternary matvec (the engine spec) -----------------------------------------------------------------

def matvec_ternary(scales_f16, codes, x8, B, R):
    """Exact engine matvec: scales (M, G) fp16, codes (M, K) uint8 {0,1,2}, x8 (K,) int8.
    Returns int32 results (M,). Asserts the converter-chosen B/R prevent saturation."""
    M, K = codes.shape
    G    = K//128
    x    = x8.astype(np.int64)
    w    = codes.astype(np.int64) - 1                            # {-1, 0, +1}.
    gs   = (w*x[None, :]).reshape(M, G, 128).sum(axis=2)         # Exact group sums.
    m11, e16 = fp16_decompose(scales_f16)
    prod = gs*m11.astype(np.int64)                               # |.| < 2^25.
    acc  = np.zeros(M, dtype=np.int64)
    for g in range(G):                                           # Per-group shift can differ.
        sh = e16[:, g] + B
        c  = prod[:, g]
        acc += np.where(sh >= 0, c << np.maximum(sh, 0), c >> np.maximum(-sh, 0))
    assert np.all(np.abs(acc) < 2**(ACC_BITS - 1)), "int48 accumulator overflow"
    res = acc >> R
    assert np.all(np.abs(res) < 2**(RESULT_BITS - 1)), "int32 result overflow"
    return res.astype(np.int32)

def matvec_q8(scales_f16, q8w, x8, B, R):
    """Q8 row-dot mode (K-cache scores, Q8 tensors): per-group-of-32 fp16 scales.
    scales (M, K//32), q8w (M, K) int8, x8 (K,) int8 -> int32 (M,)."""
    M, K = q8w.shape
    G    = K//32
    x    = x8.astype(np.int64)
    gs   = (q8w.astype(np.int64)*x[None, :]).reshape(M, G, 32).sum(axis=2) # |gs| <= 32*127*127.
    m11, e16 = fp16_decompose(scales_f16)
    prod = gs*m11.astype(np.int64)
    acc  = np.zeros(M, dtype=np.int64)
    for g in range(G):
        sh = e16[:, g] + B
        c  = prod[:, g]
        acc += np.where(sh >= 0, c << np.maximum(sh, 0), c >> np.maximum(-sh, 0))
    assert np.all(np.abs(acc) < 2**(ACC_BITS - 1))
    res = acc >> R
    assert np.all(np.abs(res) < 2**(RESULT_BITS - 1))
    return res.astype(np.int32)

def axpy_rows(scales_f16, v8w, p8, B, Rpre):
    """AXPY mode (weighted-V): accumulate coeff_t * v_t into per-lane int32 accumulators.
    scales (M, K//32) fp16 (per-row 2^ev, all groups equal), v8w (M, K) int8 V rows,
    p8 (M,) int8 probability coefficients (position t -> row t of this channel's stripe).
    Per-lane (RTL-faithful) form - pow-2 row scales only (m11 == 1024, the kvpack
    contract), so the whole row weight is one shift:
      acc[d] += (p8_t * v8[t, d]) << (10 + e16_t + B - Rpre)   (arith >> if negative)
    Firmware picks B = 12 - max(e16) (as Q8) and Rpre from axpy_rpre(). Returns int32 (K,)."""
    M, K = v8w.shape
    m11, e16 = fp16_decompose(scales_f16)
    assert np.all(np.abs(m11) == 1024), "axpy requires pow-2 row scales"
    acc = np.zeros(K, dtype=np.int64)
    for t in range(M):
        sh = 10 + int(e16[t, 0]) + B - Rpre  # Row-constant lane shift.
        prod = int(p8[t])*v8w[t].astype(np.int64)
        acc += (prod << sh) if sh >= 0 else (prod >> -sh)
        assert np.all(np.abs(acc) < 2**31), "axpy accumulator overflow"
    return acc.astype(np.int32)

def axpy_rpre(p8_sum_bound, B_top=22):
    """Rpre keeping worst-case |acc| < 2^31: per unit of p8 the contribution is bounded by
    127*1024*127*2^12 < 2^36, so r = max(0, bitlen(sum|p8|) + 36 - 31). Pure bit-length
    form (a priority encoder in hardware - the sequencer computes it from the softmax's
    returned sum), conservative by at most one bit vs the exact product loop."""
    return max(0, int(p8_sum_bound).bit_length() + 5)

# BFP vector helpers (vector-unit spec) ------------------------------------------------------------

def normalize_i32_to_bfp(v32, e_in):
    """int32 vector -> normalized (int16 mantissas, exponent). absmax lands in [2^14, 2^15).
    Zero vector -> (zeros, e_in)."""
    v = np.asarray(v32, dtype=np.int64)
    amax = int(np.abs(v).max())
    if amax == 0:
        return v.astype(np.int16), e_in
    k = amax.bit_length() - 15 # Shift so absmax has 15 bits.
    if k > 0:
        m = rne_shift(v, k)
        # RNE can push absmax to 2^15: renormalize once more.
        if int(np.abs(m).max()) >= 2**15:
            m = rne_shift(v, k + 1)
            k += 1
    else:
        m = v << (-k)
    return m.astype(np.int16), e_in + k

def bfp_to_int8(m16):
    """Matvec input quantization: int16 mantissa -> int8, exponent caller-side += 8."""
    return clamp(rne_shift(np.asarray(m16, dtype=np.int64), 8), -127, 127).astype(np.int8)

def f32_to_bfp(vec):
    """One-time load conversion of f32 vectors (norm gains, rope tables) to BFP int16.
    Spec: e = ceil(log2(absmax)) - 15 computed via frexp; m = RNE(v * 2^-e)."""
    v = np.asarray(vec, dtype=np.float64)
    amax = np.abs(v).max()
    if amax == 0:
        return np.zeros(v.shape, dtype=np.int16), 0
    e = int(np.floor(np.log2(amax))) - 14 # absmax mantissa in [2^14, 2^15).
    m = np.rint(v*np.exp2(-e)).astype(np.int64) # Round-half-even (np.rint).
    if int(np.abs(m).max()) >= 2**15:
        e += 1
        m = np.rint(v*np.exp2(-e)).astype(np.int64)
    return m.astype(np.int16), e

def bfp_value(m, e):
    """Float value of a BFP vector (for comparisons against the float model)."""
    return np.asarray(m, dtype=np.float64)*np.exp2(e)

# LUT generators (shared ROM contents for golden, vector-unit BRAM init and firmware) --------------

def rsqrt_lut(entries=256):
    """1/sqrt(x) for x in [1, 4), input u in [2^30, 2^32) (30 fractional bits). MSB-split
    layout so the index needs no multiplier: entries 0..127 cover [1, 2) at step 1/128
    (idx = (u - 2^30) >> 23), entries 128..255 cover [2, 4) at step 1/64
    (idx = 128 + ((u - 2^31) >> 24)); value = round(2^15 / sqrt(x_mid))."""
    assert entries == 256
    lo = 1.0 + np.arange(128)/128.0     # Edge values (the interpolator assumes edges).
    hi = 2.0 + np.arange(128)/64.0
    return np.round((2**15)/np.sqrt(np.concatenate([lo, hi]))).astype(np.int32)

def exp2_lut(entries=256):
    """2^f for f in [0, 1): entry i = round(2^15 * 2^(i/entries))."""
    f = np.arange(entries)/entries
    return np.round((2**15)*np.exp2(f)).astype(np.int32)

def recip_lut(entries=256):
    """1/x for x in [1, 2): entry i = round(2^15 / (1 + i/entries))."""
    x = 1.0 + np.arange(entries)/entries
    return np.round((2**15)/x).astype(np.int32)

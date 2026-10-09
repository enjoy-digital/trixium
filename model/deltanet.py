#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Gated DeltaNet recurrence on the vector unit: the numerics, to the bit.

One value head holds a state M (hv rows x hk columns). Per token, with k and q the head's
L2-normalised key and query, v its value row, alpha the decay and beta the write gate
(llama.cpp's kernel, row j independent of every other row):

    M_j <- alpha M_j;  d_j = (v_j - M_j . k) beta;  M_j <- M_j + d_j k;  o_j = M_j . q

The fast path keeps M as int16 in the scratchpad at one exponent eM per head (the value
is M_int * 2^eM), k and q as int16 with their own exponents, and runs two microcode
passes per head (asm.prog_dn_a / prog_dn_b) with the CPU doing only per-head scalar
algebra between them:

  pass A  dot_j = M_j . k16 exact in the 48-bit scalar (units 2^(eM+ek)); SSHLV to units
          of 2^-19; VMULH15 by the Q15 decay (floor); + the CPU-provided -v_j (units
          2^-19, one int32 per 32-byte beat); VMULH15 by -beta (Q15, floor) = delta_j
          in units of 2^-19; REDMAX of |delta_j| into S6; delta_j stored as 8 copies.
  CPU     the exponent ratchet: from max |M| (pass B of the previous token) and
          max |delta| (S6), the number n of halvings that keeps every updated entry
          inside int16 - a bound (|alpha M| + |delta| |k|), not the exact new maximum,
          so it can ratchet a bit early; then the pass-B constants.
  pass B  the 8 copies summed back (8 delta_j), rounded and shifted to delta_s (16 bits,
          so delta_s * k16 fits a lane); per beat M' = rne((M * d15n + delta_s * k16),
          15) with d15n = alpha * 2^(15-n) carrying the ratchet; REDMAX |M'| into S5 for
          the next token; o_j = M'_j . q16 exact, >> 8, stored as 8 copies.

The functions below are the specification: `head_step` is what the two passes plus the
CPU compute for one head, `pass_a_args` / `ratchet` /
`pass_b_args` are the CPU's algebra, and test/model/test_asm.py proves the interpreter's execution
of the two programs equals head_step to the bit. Every shift is a floor (the unit's
SSHLV/VSHLS/VMULH15) except the update's VRNE; the two multiplies by Q15 constants are
floors on scalars, i.e. rounding-level noise per row, not a drift per element (the
update's rounding is nearest-even for that reason: a floor there would walk every entry
down half a unit per token).
"""

import numpy as np

from model import bitexact as bx

# Constants ----------------------------------------------------------------------------------------

DELTA_FRAC = 19            # delta_j, v_j and the decayed dot live in units of 2^-19.
O_SHIFT    = 8             # o_j leaves the unit as (M'_j . q16) >> 8 (fits int32 for any state).
STATE_LIM  = (1 << 15) - 4 # The ratchet keeps |M'| below this (rounding adds < 2).

# Quantization -------------------------------------------------------------------------------------

def lroundf(x):
    """C lroundf: round half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)

def q16_head(x):
    """A head's vector as int16 mantissas with one power-of-two exponent: |m| in
    [2^14, 2^15) at the max, the same rule as bx.f32_to_bfp, on float32 inputs."""
    x = np.asarray(x, dtype=np.float32)
    amax = np.abs(x).max()
    if amax == 0:
        return np.zeros(len(x), dtype=np.int64), 0
    e = int(np.frexp(np.float32(amax))[1]) - 15
    return np.clip(lroundf(x*np.float32(np.exp2(-e))), -32767, 32767), e

def q15(x):
    """A [0, 1] constant as Q15 (32767 at most)."""
    return int(np.clip(lroundf(np.float32(x)*np.float32(32768.0)), 0, 32767))

# CPU Algebra --------------------------------------------------------------------------------------

def pass_a_args(eM, ek, decay, beta):
    """(dot shift for SSHLV, d15, -beta_q) for prog_dn_a."""
    r = -DELTA_FRAC - (eM + ek)          # dot (units 2^(eM+ek)) >> r -> units 2^-19
    return -r, q15(decay), -q15(beta)

def neg_v_int(vh):
    """The CPU's -v_j in units of 2^-19 (one int32 per beat)."""
    nv = -lroundf(np.asarray(vh, dtype=np.float32).astype(np.float64)*float(1 << DELTA_FRAC))
    assert np.abs(nv).max() < (1 << 31)
    return nv

def ratchet(maxM, maxD, eM, ek, k16max, decay):
    """Halvings n so that every updated entry fits int16: |alpha M| + |delta| |k| < lim
    with |delta| <= maxD 2^-19 and |k| <= k16max 2^ek, both in the state's units."""
    kmax = float(k16max)*float(np.exp2(ek))
    t = (maxM*float(decay) + maxD*float(np.exp2(-DELTA_FRAC))*kmax*float(np.exp2(-eM)))/float(STATE_LIM)
    return 0 if t <= 1.0 else int(np.ceil(np.log2(t)))

def pass_b_args(eM2, ek, n, decay):
    """(delta shift for VSHLS, d15n, rounding constant) for prog_dn_b."""
    sh = eM2 + 7 - ek                    # 8 delta_j (units 2^-19) -> delta_s: (.. * k16) >> 15 in 2^eM2
    d15n = int(np.clip(lroundf(np.float32(decay)*np.float32(np.exp2(15 - n))), 0, 32767))
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    return -sh, d15n, rnd

# Head Step ----------------------------------------------------------------------------------------

def head_step(M, eM, kk, qh, vh, decay, beta, maxM=None, stats=None, trace=None):
    """One head, one token: M (int64 array hv x hk, updated in place, int16 range), eM
    its exponent, kk/qh/vh float32, decay/beta float32 -> (o float32, eM2, maxM2).
    maxM: max |M| as pass B of the previous token left it (None = compute it). trace (a
    dict) receives every intermediate the two passes and the CPU exchange."""
    hv, hk = M.shape
    k16, ek = q16_head(kk)
    q16, eq = q16_head(qh)
    if maxM is None:
        maxM = int(np.abs(M).max())
    shift_a, d15, nbq = pass_a_args(eM, ek, decay, beta)
    # Pass A.
    dot = M.astype(np.int64) @ k16.astype(np.int64)
    dot_r = bx.arith_shift(dot, shift_a)
    assert np.abs(dot_r).max() < (1 << 31), (eM, ek, np.abs(dot_r).max())
    ddr = (dot_r*d15) >> 15
    s3 = ddr + neg_v_int(vh)
    delta = (s3*nbq) >> 15
    maxD = int(np.abs(delta).max())
    # The CPU.
    n = ratchet(maxM, maxD, eM, ek, int(np.abs(k16).max()), decay)
    eM2 = eM + n
    shift_b, d15n, rnd = pass_b_args(eM2, ek, n, decay)
    # Pass B.
    d8 = 8*delta
    assert np.abs(d8 + rnd).max() < (1 << 31)
    delta_s = bx.arith_shift(d8 + rnd, shift_b)
    assert np.abs(delta_s).max() < (1 << 16), (np.abs(delta_s).max(), shift_b, n)
    prod = M.astype(np.int64)*d15n + np.outer(delta_s, k16.astype(np.int64))
    newM = bx.rne_shift(prod, 15)
    assert np.abs(newM).max() <= 32767, (np.abs(newM).max(), n)
    M[:] = newM
    maxM2 = int(np.abs(newM).max())
    o = newM @ q16.astype(np.int64)
    o_r = o >> O_SHIFT
    assert np.abs(o_r).max() < (1 << 31)
    if stats is not None:
        stats.setdefault("eM", []).append(eM2)
        stats.setdefault("n", []).append(n)
    if trace is not None:
        trace.update(k16=k16, ek=ek, q16=q16, eq=eq, a_args=(shift_a, d15, nbq),
                     nv=neg_v_int(vh), delta=delta, maxD=maxD, n=n, eM2=eM2,
                     b_args=(shift_b, d15n, rnd), delta_s=delta_s, maxM2=maxM2, o_r=o_r)
    of = (o_r.astype(np.float64)*float(np.exp2(eM2 + eq + O_SHIFT))).astype(np.float32)
    return of, eM2, maxM2

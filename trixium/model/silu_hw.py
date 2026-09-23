# SPDX-License-Identifier: BSD-2-Clause
"""Bit-level model of the hardware SiLU-multiply kernel, and its exactness contract.

The firmware's hybrid path computes `dst[i] = mul[i] * siluf(gate[i])` (hybrid.c
silu_mul_fv; fastmath.h bonsai_silu_lookup): for -16 < g < 16

    u = (g + 16) * 128          FP32 add (RNE), then an exact power-of-two scale
    i = (int)u                  truncation; u >= 0
    f = u - i                   exact
    s = fmaf(f, lut[i+1] - lut[i], lut[i])   one FP32 subtract, one fused multiply-add
    r = mul * s                 FP32 multiply (RNE; the firmware is built without contraction)

with lut[] the 4,097-entry table the firmware builds once at boot with the target libm.
Outside (-16, 16) it uses a polynomial exp and a divide.

The hardware takes that table (and the difference table d[i] = lut[i+1] - lut[i], which
the firmware computes with the same single FP32 subtraction) from the firmware, so it
needs only an add of the constant 16, truncation, one FMA and one multiply to reproduce
the firmware bit for bit. Everything else is a FIXUP: the element is flagged, the
hardware result is not used, and the CPU recomputes it with the firmware's own code.
FIXUP covers the polynomial tails (-80 <= g <= -16 and g == 16), non-finite inputs,
a subnormal multiplier, and a product that is not a normal FP32 number. g > 16 (s = g)
and g < -80 (s = -0.0) are exact without the polynomial and are NOT fixups.

Everything here is exact integer arithmetic on IEEE-754 binary32 bit patterns; there is
no host floating point on the checked path. `silu_mul_fast` is what the RTL implements
stage for stage; `silu_mul_firmware` is what the CPU computes; the contract, tested in
test/test_silu_hw.py, is that they agree on every element the fast path does not flag.
"""
from functools import lru_cache

import numpy as np

from trixium.model import rv32_math

F32_ONE = 0x3F800000
F32_16 = 0x41800000
F32_NEG_ZERO = 0x80000000
LUT_N = 4096          # 2 * 16 * 128 intervals, 4,097 points


# Exact binary32 <-> (sign, integer significand, power of two) -------------------------------

def decode(bits):
    """(sign, N, E, kind) with value (-1)**sign * N * 2**E; kind in normal/subnormal/zero/inf/nan."""
    bits &= 0xFFFFFFFF
    sign, exp, frac = bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exp == 0xFF:
        return sign, 0, 0, "nan" if frac else "inf"
    if exp == 0:
        return sign, frac, -149, "zero" if frac == 0 else "subnormal"
    return sign, frac | 0x800000, exp - 150, "normal"


def round_to_f32(sign, n, e):
    """RNE rounding of the exact value (-1)**sign * n * 2**e (n >= 0) to binary32 bits.

    Gradual underflow and overflow to infinity follow IEEE-754 (the reference needs them;
    the fast path flags those results instead of producing them)."""
    if n == 0:
        return sign << 31
    p = n.bit_length() - 1
    ulp_e = max(p + e - 23, -149)
    shift = ulp_e - e
    if shift > 0:
        q, rem = n >> shift, n & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
    else:
        q = n << (-shift)
    if q >= 1 << 24:                    # the rounding carried into a new binade
        q >>= 1
        ulp_e += 1
    if q < 1 << 23:                     # subnormal (only possible at ulp_e == -149)
        return (sign << 31) | q
    biased = ulp_e + 150
    if biased >= 0xFF:
        return (sign << 31) | 0x7F800000
    return (sign << 31) | (biased << 23) | (q & 0x7FFFFF)


def _exact(bits):
    sign, n, e, kind = decode(bits)
    if kind in ("nan", "inf"):
        raise ValueError("exact arithmetic needs finite operands")
    return sign, n, e


def _sum(terms):
    """Exact signed sum of (sign, n, e) terms -> (sign, magnitude, e); zero sign by RNE rules."""
    base = min(e for _, _, e in terms)
    total = sum((-1 if s else 1) * (n << (e - base)) for s, n, e in terms)
    if total == 0:
        # x + (-x) is +0 under RNE; the sum of two same-signed zeros keeps that sign.
        signs = {s for s, n, _ in terms if n == 0} if all(n == 0 for _, n, _ in terms) else {0}
        return (1 if signs == {1} else 0), 0, base
    return (1 if total < 0 else 0), abs(total), base


def f32_add(a, b):
    return round_to_f32(*_sum([_exact(a), _exact(b)]))


def f32_sub(a, b):
    return f32_add(a, b ^ 0x80000000)


def f32_mul(a, b):
    sa, na, ea = _exact(a)
    sb, nb, eb = _exact(b)
    return round_to_f32(sa ^ sb, na * nb, ea + eb)


def f32_fma(a, b, c):
    """a*b + c with a single rounding."""
    sa, na, ea = _exact(a)
    sb, nb, eb = _exact(b)
    return round_to_f32(*_sum([(sa ^ sb, na * nb, ea + eb), _exact(c)]))


def f32_bits(x):
    return int(np.float32(x).view(np.uint32))


def bits_f32(b):
    return np.uint32(b).view(np.float32)


# Tables -------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def tables():
    """(lut, dlut) as tuples of bits. lut is the firmware's table (rv32_math.silu_table,
    checked against the target libm); dlut[i] = lut[i+1] - lut[i] for i < 4096."""
    lut = tuple(f32_bits(v) for v in rv32_math.silu_table())
    dlut = tuple(f32_sub(lut[i + 1], lut[i]) for i in range(LUT_N))
    return lut, dlut


# The two sides of the contract ---------------------------------------------------------------

def silu_mul_firmware(g, m):
    """What hybrid.c computes for finite inputs: mul * siluf(gate), as bits."""
    s = f32_bits(rv32_math.silu_rv32([bits_f32(g)])[0])
    return f32_mul(m, s)


def silu_mul_fast(g, m, lut=None, dlut=None):
    """The hardware fast path: (result bits, fixup). The result is meaningless when fixup."""
    if lut is None:
        lut, dlut = tables()
    gs, gn, ge, gk = decode(g)
    ms, mn, me, mk = decode(m)
    if gk in ("nan", "inf") or mk in ("nan", "inf", "subnormal"):
        return 0, True
    mag = g & 0x7FFFFFFF
    if mag >= F32_16:                               # |g| >= 16
        if not gs and mag > F32_16:
            s = g                                   # g > 16: siluf returns g itself
        elif gs and mag > 0x42A00000:               # g < -80: siluf returns -0.0
            s = F32_NEG_ZERO
        else:
            return 0, True                          # -80 <= g <= -16 or g == 16: polynomial
    else:
        a = f32_add(g, F32_16)                      # (0, 32]
        _, an, ae, ak = decode(a)
        if ak != "normal":
            return 0, True                          # cannot happen for |g| < 16; checked below
        # u = a * 128 exactly: the same significand two binades up. i = floor(u), f = u - i.
        ue = ae + 7
        if ue >= 0:
            i, frac_n, frac_e = an << ue, 0, 0
        else:
            i = an >> (-ue)
            frac_n, frac_e = an & ((1 << (-ue)) - 1), ue
        if i >= LUT_N:
            s = lut[LUT_N]                          # the largest finite x < 16 rounds u to 4096
        else:
            f = round_to_f32(0, frac_n, frac_e)     # exact: fewer than 24 significant bits
            s = f32_fma(f, dlut[i], lut[i])
    r = f32_mul(m, s)
    _, rn, re_, rk = decode(r)
    # The multiplier handles normal results and signed zeros; a subnormal or infinite
    # product (from a nonzero s) is left to the CPU.
    if rk in ("subnormal", "inf", "nan"):
        return 0, True
    if rk == "zero" and (m & 0x7FFFFFFF) and (s & 0x7FFFFFFF):
        return 0, True                              # underflow of a nonzero product
    return r, False

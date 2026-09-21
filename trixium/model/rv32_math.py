# SPDX-License-Identifier: BSD-2-Clause
"""Numerical profile of the baseline RV32F firmware, not a generic libm claim.

GCC 10.1's rv32imf/ilp32f newlib expf contains fused operations even though our
firmware is built with -ffp-contract=off. NumPy exp (or a host libm LUT) does not
necessarily produce those bits. This explicit profile is checked against the
actual target archive under QEMU, and against compiled full-SoC checkpoints.
Other target libms/ABIs must be qualified separately. Only finite exp inputs in
[-80, 80] are modeled; SiLU accepts all finite FP32 inputs.
"""
import ctypes
import ctypes.util
from functools import lru_cache

import numpy as np

f32 = np.float32


@lru_cache(maxsize=1)
def _host_fmaf():
    # FMA must round ONCE to FP32. float64 a*b+c followed by float32 can double
    # round. IEEE libm fmaf supplies the primitive, not the target exp algorithm.
    library = ctypes.util.find_library("m")
    fn = ctypes.CDLL(library).fmaf
    fn.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fn.restype = ctypes.c_float
    return fn


def fma32(a, b, c):
    return f32(_host_fmaf()(float(a), float(b), float(c)))


def _bits(x):
    return int(f32(x).view(np.uint32))


def _word(x):
    return np.uint32(x).view(f32)


# expf's approximation/reduction is adapted from SunPro fdlibm/newlib ef_exp.c,
# with fusion and the small-input cutoff following the actual target archive.
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
def expf_rv32(x):
    x = f32(x)
    if not np.isfinite(x) or not -80 <= x <= 80:
        raise ValueError("RV32 exp profile requires finite input in [-80, 80]")
    hx = _bits(x) & 0x7fffffff
    sign = bool(_bits(x) >> 31)
    k = 0
    if hx > 0x3eb17218:
        if hx < 0x3f851592:
            hi = f32(x - _word(0xbf317180 if sign else 0x3f317180))
            lo = _word(0xb717f7d1 if sign else 0x3717f7d1)
            k = -1 if sign else 1
        else:
            k = int(fma32(x, _word(0x3fb8aa3b), -0.5 if sign else 0.5))
            hi = fma32(-f32(k), _word(0x3f317180), x)
            lo = f32(f32(k)*_word(0x3717f7d1))
        x = f32(hi - lo)
    elif hx < 0x31800000:
        return f32(1 + x)
    t = f32(x*x)
    p = _word(0x3331bb4c)
    for word in (0xb5ddea0e, 0x388ab355, 0xbb360b61, 0x3e2aaaab):
        p = fma32(p, t, _word(word))
    c = fma32(-p, t, x)
    xc = f32(x*c)
    if not k:
        return f32(f32(1) - f32(f32(xc/f32(c - f32(2))) - x))
    y = f32(f32(1) - f32(f32(lo - f32(xc/f32(f32(2) - c))) - hi))
    # The bounded domain above keeps this a normal result, as in the target.
    return _word(_bits(y) + (k << 23))


def expf_bounded(x):
    """firmware/fastmath.h, with explicit fused polynomial stages only."""
    x = f32(x)
    if not np.isfinite(x) or not -80 <= x <= 16:
        raise ValueError("fast exp requires finite input in [-80, 16]")
    z = f32(x*f32(float.fromhex("0x1.715476p+0")))
    n = int(f32(z + f32(0.5 if z >= 0 else -0.5)))
    fn = f32(n)
    r = f32(f32(x - f32(fn*f32(float.fromhex("0x1.62e400p-1"))))
            - f32(fn*f32(float.fromhex("0x1.7f7d1cp-20"))))
    p = f32(float.fromhex("0x1.6c16c2p-10"))
    for c in ("0x1.111112p-7", "0x1.555556p-5", "0x1.555556p-3", "0x1p-1", "0x1p0", "0x1p0"):
        p = fma32(p, r, f32(float.fromhex(c)))
    return f32(p*_word((n + 127) << 23))


def silu_polynomial(x):
    x = f32(x)
    if not np.isfinite(x):
        raise ValueError("SiLU reference requires finite input")
    if x > 16:
        return x
    if x < -80:
        return f32(-0.0)
    if x < -16:
        return f32(x*expf_bounded(x))
    return f32(x/f32(f32(1) + expf_bounded(-x)))


@lru_cache(maxsize=1)
def silu_table():
    table = np.empty(4097, dtype=f32)
    for i in range(len(table)):
        x = f32(-16 + i/128)  # Grid points are exact binary fractions.
        table[i] = f32(x/f32(f32(1) + expf_rv32(-x)))
    table.flags.writeable = False
    return table


def silu_rv32(values):
    """The baseline hybrid.c LUT, fused interpolation, and polynomial tails."""
    values = np.asarray(values, dtype=f32)
    table = silu_table()
    result = np.empty_like(values)
    for index in np.ndindex(values.shape):
        x = values[index]
        if not np.isfinite(x):
            raise ValueError("SiLU reference requires finite input")
        if x >= 16 or x <= -16:
            result[index] = silu_polynomial(x)
        else:
            u = f32(f32(x + f32(16))*f32(128))
            i = int(u)
            # The final finite interior x can round u to the endpoint. Match the
            # firmware's bounds fix, without accessing table[4097].
            if i >= 4096:
                result[index] = table[4096]
                continue
            f = f32(u - f32(i))
            result[index] = fma32(f, f32(table[i + 1] - table[i]), table[i])
    return result

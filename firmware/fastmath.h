/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium bounded FP32 exp, SiLU and sigmoid (canonical forms shared with the host oracles)
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_FASTMATH_H
#define TRIXIUM_FASTMATH_H

#include <stdint.h>

/* Exp */
/*-----*/

// expf() is about 250 cycles on the XCU1525's RV32F VexiiRiscv.  Hybrid SiLU calls it
// 1.9 million times per 27B forward, so a general libm implementation dominates the CPU
// even though all ordinary FP32 operations are native instructions.  SiLU only needs exp
// on a bounded interval: outside [-16, 16] its saturated form is both stable and cheaper.
//
// Reduce x to r in [-ln(2)/2, ln(2)/2], evaluate exp(r) through degree six, and construct
// 2^n exactly.  The omitted Taylor term is below 1.3e-7 over this interval.  Explicit
// fmaf() lets RV32F issue one instruction per polynomial stage; the firmware is built with
// -ffp-contract=off so changing this hot approximation cannot silently change other math.
static inline float trixium_expf_bounded(float x)
{
    const float log2e = 0x1.715476p+0f;
    const float ln2_hi = 0x1.62e400p-1f;
    const float ln2_lo = 0x1.7f7d1cp-20f;
    float z = x*log2e;
    int n = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
    float fn = (float)n;
    float r = (x - fn*ln2_hi) - fn*ln2_lo;

    float p = 0x1.6c16c2p-10f;       // 1/720
    p = __builtin_fmaf(p, r, 0x1.111112p-7f);  // 1/120
    p = __builtin_fmaf(p, r, 0x1.555556p-5f);  // 1/24
    p = __builtin_fmaf(p, r, 0x1.555556p-3f);  // 1/6
    p = __builtin_fmaf(p, r, 0x1.000000p-1f);  // 1/2
    p = __builtin_fmaf(p, r, 1.0f);
    p = __builtin_fmaf(p, r, 1.0f);

    union { uint32_t u; float f; } scale = { (uint32_t)(n + 127) << 23 };
    return p*scale.f;
}

// Canonical softmax exp (doc/units.md (Canonical reductions), "softmax"): the bounded exp above
// on [-80, 0], +0 below (a term under 2^-115 of the maximum's 1.0, below every FP32 sum
// that holds the maximum), a NaN argument returned (s - max: already the canonical NaN).
// Softmax arguments are s - max <= 0, exactly +0 at the maximum (exp(+0) = 1 here too).
static inline float trixium_softmax_exp(float x)
{
    if (x != x) return x;
    if (x < -80.0f) return 0.0f;
    return trixium_expf_bounded(x);
}

// The DeltaNet decay exp(g), g = softplus(a + dt)*A <= 0 (A = -exp(A_log)): the same
// canonical exp (doc/units.md (Canonical reductions)) - ~40 cycles against libm's ~250, 2,304
// times per 27B forward.
static inline float trixium_decay_exp(float g) { return trixium_softmax_exp(g); }

/* SiLU */
/*------*/

static inline float trixium_siluf(float x)
{
    // These tails avoid overflow and keep the range-reduction exponent normal.  At 16,
    // the discarded sigmoid correction is about one ulp of x; the polynomial path is used
    // at the boundary so the function remains continuous to FP32 precision.
    if (x > 16.0f) return x;
    if (x < -80.0f) return -0.0f;
    if (x < -16.0f) return x*trixium_expf_bounded(x);
    return x/(1.0f + trixium_expf_bounded(-x));
}

#define TRIXIUM_SILU_LUT_SCALE 128
#define TRIXIUM_SILU_LUT_LIMIT 16
#define TRIXIUM_SILU_LUT_N (2*TRIXIUM_SILU_LUT_LIMIT*TRIXIUM_SILU_LUT_SCALE)

// Finite-input lookup shared by the firmware and target-math regression harness.
static inline float trixium_silu_lookup(float x, const float *lut)
{
    if (x >= TRIXIUM_SILU_LUT_LIMIT || x <= -TRIXIUM_SILU_LUT_LIMIT)
        return trixium_siluf(x);
    float u = (x + TRIXIUM_SILU_LUT_LIMIT)*TRIXIUM_SILU_LUT_SCALE;
    int i = (int)u;
    // The largest float below 16 rounds (x + 16) to 32, making i == N.
    // Its interpolation fraction is zero; return the endpoint without reading N+1.
    if (i >= TRIXIUM_SILU_LUT_N) return lut[TRIXIUM_SILU_LUT_N];
    float f = u - (float)i;
    return __builtin_fmaf(f, lut[i + 1] - lut[i], lut[i]);
}

/* Sigmoid */
/*---------*/

// Canonical sigmoid (doc/units.md (Canonical reductions), "sigmoid"): the SiLU lookup's form on a
// sigmoid table, with tails from the same bounded exp - no libm expf anywhere, so the
// firmware, the mirror and a table-driven unit agree bit for bit. The table is built by
// trixium_sigmoid_point at the 4,097 grid points.
static inline float trixium_sigmoid_point(float x)          // x in [-16, 80]
{
    return 1.0f/(1.0f + trixium_expf_bounded(-x));
}

static inline float trixium_sigmoid_tail(float x)           // |x| >= 16, finite
{
    if (x > 80.0f) return 1.0f;
    if (x >= 16.0f) return trixium_sigmoid_point(x);
    if (x < -80.0f) return 0.0f;
    float e = trixium_expf_bounded(x);
    return e/(1.0f + e);
}

static inline float trixium_sigmoid_lookup(float x, const float *lut)
{
    if (x >= TRIXIUM_SILU_LUT_LIMIT || x <= -TRIXIUM_SILU_LUT_LIMIT)
        return trixium_sigmoid_tail(x);
    float u = (x + TRIXIUM_SILU_LUT_LIMIT)*TRIXIUM_SILU_LUT_SCALE;
    int i = (int)u;
    if (i >= TRIXIUM_SILU_LUT_N) return lut[TRIXIUM_SILU_LUT_N];
    float f = u - (float)i;
    return __builtin_fmaf(f, lut[i + 1] - lut[i], lut[i]);
}

#endif /* TRIXIUM_FASTMATH_H */

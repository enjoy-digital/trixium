// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#ifndef BONSAI_FASTMATH_H
#define BONSAI_FASTMATH_H

#include <stdint.h>

// expf() is about 250 cycles on the XCU1525's RV32F VexiiRiscv.  Hybrid SiLU calls it
// 1.9 million times per 27B forward, so a general libm implementation dominates the CPU
// even though all ordinary FP32 operations are native instructions.  SiLU only needs exp
// on a bounded interval: outside [-16, 16] its saturated form is both stable and cheaper.
//
// Reduce x to r in [-ln(2)/2, ln(2)/2], evaluate exp(r) through degree six, and construct
// 2^n exactly.  The omitted Taylor term is below 1.3e-7 over this interval.  Explicit
// fmaf() lets RV32F issue one instruction per polynomial stage; the rest of hybrid.c keeps
// -ffp-contract=off so changing this hot approximation cannot silently change other math.
static inline float bonsai_expf_bounded(float x)
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

static inline float bonsai_siluf(float x)
{
    // These tails avoid overflow and keep the range-reduction exponent normal.  At 16,
    // the discarded sigmoid correction is about one ulp of x; the polynomial path is used
    // at the boundary so the function remains continuous to FP32 precision.
    if (x > 16.0f) return x;
    if (x < -80.0f) return -0.0f;
    if (x < -16.0f) return x*bonsai_expf_bounded(x);
    return x/(1.0f + bonsai_expf_bounded(-x));
}

#define BONSAI_SILU_LUT_SCALE 128
#define BONSAI_SILU_LUT_LIMIT 16
#define BONSAI_SILU_LUT_N (2*BONSAI_SILU_LUT_LIMIT*BONSAI_SILU_LUT_SCALE)

// Finite-input lookup shared by the firmware and target-math regression harness.
static inline float bonsai_silu_lookup(float x, const float *lut)
{
    if (x >= BONSAI_SILU_LUT_LIMIT || x <= -BONSAI_SILU_LUT_LIMIT)
        return bonsai_siluf(x);
    float u = (x + BONSAI_SILU_LUT_LIMIT)*BONSAI_SILU_LUT_SCALE;
    int i = (int)u;
    // The largest float below 16 rounds (x + 16) to 32, making i == N.
    // Its interpolation fraction is zero; return the endpoint without reading N+1.
    if (i >= BONSAI_SILU_LUT_N) return lut[BONSAI_SILU_LUT_N];
    float f = u - (float)i;
    return __builtin_fmaf(f, lut[i + 1] - lut[i], lut[i]);
}

// Canonical sigmoid (docs/canonical-reductions.md, "sigmoid"): the SiLU lookup's form on a
// sigmoid table, with tails from the same bounded exp - no libm expf anywhere, so the
// firmware, the mirror and a table-driven unit agree bit for bit. The table is built by
// bonsai_sigmoid_point at the 4,097 grid points.
static inline float bonsai_sigmoid_point(float x)          // x in [-16, 80]
{
    return 1.0f/(1.0f + bonsai_expf_bounded(-x));
}

static inline float bonsai_sigmoid_tail(float x)           // |x| >= 16, finite
{
    if (x > 80.0f) return 1.0f;
    if (x >= 16.0f) return bonsai_sigmoid_point(x);
    if (x < -80.0f) return 0.0f;
    float e = bonsai_expf_bounded(x);
    return e/(1.0f + e);
}

static inline float bonsai_sigmoid_lookup(float x, const float *lut)
{
    if (x >= BONSAI_SILU_LUT_LIMIT || x <= -BONSAI_SILU_LUT_LIMIT)
        return bonsai_sigmoid_tail(x);
    float u = (x + BONSAI_SILU_LUT_LIMIT)*BONSAI_SILU_LUT_SCALE;
    int i = (int)u;
    if (i >= BONSAI_SILU_LUT_N) return lut[BONSAI_SILU_LUT_N];
    float f = u - (float)i;
    return __builtin_fmaf(f, lut[i + 1] - lut[i], lut[i]);
}

#endif

/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium Qwen3 token loop and the float matvec primitives of the hybrid path
 *
 * The engine does every matvec (ternary weights streamed from the weight memory); the vector
 * unit and the descriptor sequencer run the vector ops and attention on resident
 * block-floating-point activations in the scratchpad. Every op keeps a CPU reference
 * implementation (hwops). The hybrid models' forward lives in hybrid.c, on top of the float
 * matvec primitives exported here.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>
#include <generated/mem.h>

#include <system.h>

#include "bons.h"
#include "engine.h"
#include "hybrid.h"
#include "model.h"
#include "infer.h"
#include "static_memory.h"
#include "ops.h"
#include "perf.h"
#include "vec.h"
#include "seq.h"
#include "progs.h"

/* Code Layout */
/*-------------*/

// The per-layer hot path is ~28 KB of text against a 32 KB I$ with no L2 behind it (a miss
// is a ~150-cycle HBM round trip). These hints and the cold attributes below move the CPU
// fallback paths out of the hot text (-freorder-blocks-and-partition, see the Makefile):
// same instructions, same numerics.
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define COLD        __attribute__((cold, noinline))

// A call to a cold function is the one thing GCC's partitioner treats as "never executed"
// (__builtin_expect is only 90%): placed first in a branch that is dead on the card (the
// op-by-op paths behind the sequencer lists), it moves that whole branch to .text.unlikely.
static COLD void cold_path(void) { __asm__ volatile("" ::: "memory"); }

/* Float Helpers */
/*---------------*/

// newlib-libm compatibility: the toolchain libm.a references __errno, absent from the
// picolibc environment LiteX links against.
int *__errno(void);
static int __errno_storage;
int *__errno(void) { return &__errno_storage; }

// The LiteX firmware is built with -fno-builtin, so even an RV32F build would call libm for
// every fabsf/lroundf in float->int8 quantization. RISC-V's static RMM rounding mode is
// exactly lroundf's finite-input rule (nearest, ties away from zero); libm stays as the
// bit-identical soft-float fallback.
static inline float f32_abs(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    v.u &= 0x7fffffff;
    return v.f;
}

static inline uint32_t f32_abs_bits(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    return v.u & 0x7fffffff;
}

static inline int32_t f32_lround_i32(float x)
{
#if defined(__riscv_flen) && __riscv_flen >= 32
    int32_t r;
    __asm__ volatile("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
#else
    return (int32_t)lroundf(x);
#endif
}

static inline int32_t f32_q8(float x, float is)
{
    int32_t q = f32_lround_i32(x*is);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    return q;
}

static inline uint32_t f32_pack_q8x4(const float *src, float is)
{
    // Spell out four independent lanes.  GCC 10 otherwise retains an inner b=0..3 loop
    // with a variable shift, serialising the RV32F conversions and adding four branches
    // to every scratchpad word.
    uint32_t q0 = (uint8_t)(int8_t)f32_q8(src[0], is);
    uint32_t q1 = (uint8_t)(int8_t)f32_q8(src[1], is);
    uint32_t q2 = (uint8_t)(int8_t)f32_q8(src[2], is);
    uint32_t q3 = (uint8_t)(int8_t)f32_q8(src[3], is);
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

static inline uint32_t i32_pack_q8x4(const int32_t *src, float is)
{
    // Fused Hadamard boundary: instead of int32 -> FP32*2^e in main RAM, then *2^-e8, both
    // power-of-two scales are combined after the same int32->FP32 conversion, which is
    // bit-identical for the normal finite values produced by the engine.
    uint32_t q0 = (uint8_t)(int8_t)f32_q8((float)src[0], is);
    uint32_t q1 = (uint8_t)(int8_t)f32_q8((float)src[1], is);
    uint32_t q2 = (uint8_t)(int8_t)f32_q8((float)src[2], is);
    uint32_t q3 = (uint8_t)(int8_t)f32_q8((float)src[3], is);
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

static int f32_e8_from_amax_bits(uint32_t amax)
{
    if (!amax) return 0;
    uint32_t exp = amax >> 23;
    if (exp) return (int)exp - 133; // unbiased exponent + 1 (frexpf), then -7.
    union { uint32_t u; float f; } v = { .u = amax };
    int e8;
    frexpf(v.f, &e8);               // Subnormal activations are not expected; stay exact.
    return e8 - 7;
}

static int f32_quant_e8(const float *src, uint32_t n)
{
    uint32_t amax = 0;
    uint32_t i = 0;
    // The reduction is over ordered positive IEEE-754 bit patterns, so four independent
    // loads/comparisons are bit-identical and remove three quarters of the loop control.
    for (; i + 3 < n; i += 4) {
        uint32_t a0 = f32_abs_bits(src[i + 0]), a1 = f32_abs_bits(src[i + 1]);
        uint32_t a2 = f32_abs_bits(src[i + 2]), a3 = f32_abs_bits(src[i + 3]);
        if (a0 <= 0x7f800000u && a0 > amax) amax = a0;
        if (a1 <= 0x7f800000u && a1 > amax) amax = a1;
        if (a2 <= 0x7f800000u && a2 > amax) amax = a2;
        if (a3 <= 0x7f800000u && a3 > amax) amax = a3;
    }
    for (; i < n; i++) {
        uint32_t a = f32_abs_bits(src[i]);
        // Positive finite IEEE-754 bit patterns have the same order as their values.
        // Ignore NaNs as an FP compare would; retain infinity for equivalent overflow.
        if (a <= 0x7f800000u && a > amax) amax = a;
    }
    return f32_e8_from_amax_bits(amax);
}

/* Scratchpad Layout */
/*-------------------*/

// Byte offsets of the resident-BFP regions (int16 mantissas + firmware-tracked exponents;
// sized for the 8B: E=4096, F=12288).
#define SP_X    0x00000 // int8 matvec input (<= 12 KB).
#define SP_Y    0x04000 // Engine int32 results (<= 48 KB).
#define SP_YU   0x10000 // Engine int32 results, second stream (up proj).
#define SP_H    0x1C000 // Normed activations int16.
#define SP_XR   0x1E000 // Resident x (residual stream) int16.
#define SP_G    0x20000 // Gate int16 (silu in/out).
#define SP_U    0x26000 // Up int16.
#define SP_D    0x2C000 // Projection-result int16 (residual addend).
#define SP_FET  0x30000 // Fetch staging (embedding rows, norm16 rows).
#define SP_KVP  0x34000 // KV row-packet staging (built by CPU, engine-stored to HBM).
#define SP_SC   0x36000 // Engine score results (int32, position-major).
#define SP_AW   SP_Y            // Attention: AXPY partials + reduce/norm scratch (free here).
#define SP_AWP8 (SP_Y + 0x2000) // Per-head softmax p8-sum spill beats (32 B each).
#define SP_AWAM (SP_Y + 0x2400) // Per-head reduce-absmax spill beats (32 B each).
#define SP_Q    0x38000 // Resident q int16 (q_dim <= 4096: 8 KB).
#define SP_K    0x3A000 // Resident k int16 (kv_dim <= 1024: 2 KB).
#define SP_V    0x3B000 // Resident v int16.
#define SP_TAB  0x3B800 // Rope table row for the current position ([c14 | s14]).
#define SP_FFSP (SP_TAB + 0x400) // FFN/QKV seq spill beats (5 x 32 B): the tail of the rope
                        // table slot (the row is 2*head_dim bytes, 256 on these models). Not
                        // between SP_H and SP_XR: the 8B's SP_H (8 KB) fills that gap.
#define SP_FET2 0x3C000 // Second norm16 staging slot (k_norm while q_norm sits in SP_FET).
// Resident qk-norm weight rows (qres): head_dim int16 = 256 B per layer per kind, 28 layers =
// 7 KB each, in the tails of the SP_FET and SP_D regions (their live occupants are <= 4 KB).
// Fetched once; the qkr list's rmsnorm_h A1 is patched per layer instead of two engine
// fetches per layer.
#define SP_QNW  (SP_FET + 0x1000)
#define SP_KNW  (SP_D + 0x1000)

// Fixed exponent for CPU-produced normed vectors (values bounded ~16 by construction).
#define E_NORM  (-11)

/* Heap and Channel Windows */
/*--------------------------*/

// Heap: after the BONS header/table area. The CPU's RAM is HBM pseudo-channel 4 and engine
// channel 3's weight window overlaps it at [64 MB, ~141 MB) (see layout.py HBM_WEIGHT_BASE):
// the low heap must stay under 64 MB, and anything big goes to the HIGH heap above the
// weights+KV tail. The low heap holds a few MB of norms and tables; the only big
// consumer is the CPU-side KV cache, which is a debug fallback (HW attention keeps its own
// KV in HBM at full context) and is capped at CPU_KV_CTX positions to fit.
#define HEAP_BASE    (TRIXIUM_HDR_ADDR + 0x100000)
#define CPU_KV_CTX   512
// The high heap starts above the highest weights+KV tail any engine channel puts inside
// main_ram (channel 3, PC 4): computed per image at infer_init, 1 MB aligned (the 8B alone
// has 136 MB/channel of weights).
static uint32_t heap_hi_base;

// Engine-side addresses vs the CPU's. On the FK33 the CPU window passes through 1:1 (main_ram
// at 0x4000_0000 IS HBM address 0x4000_0000 on port 0, inside pseudo-channel 4 = engine
// channel 3's window). On a board with several memory controllers (the XCU1525's four DDR4
// channels) engine channel c lives on controller c / CHAN_MOD at that controller's local
// BASE + (c % CHAN_MOD)*STRIDE, and main_ram is controller 0's local [MEM_CPU_OFFSET, +size).
// Two SoC constants describe both; absent, the FK33's geometry.
#ifndef TRIXIUM_CHAN_MOD
#define TRIXIUM_CHAN_MOD 0                       // 0: one flat engine address space.
#endif
#ifndef TRIXIUM_MEM_CPU_OFFSET
#define TRIXIUM_MEM_CPU_OFFSET 0x40000000ull      // Engine-side address of main_ram's origin.
#endif
#ifndef TRIXIUM_CHAN_SKEW
#define TRIXIUM_CHAN_SKEW 0                       // Per-channel bank skew inside a controller.
#endif
static inline uint64_t chan_window(uint32_t c)  // Engine-side start of channel c's window.
{
    uint32_t local = TRIXIUM_CHAN_MOD ? (c % TRIXIUM_CHAN_MOD) : c;
    return (uint64_t)TRIXIUM_CHAN_WEIGHT_BASE + (uint64_t)local*TRIXIUM_CHAN_STRIDE
           + (uint64_t)local*TRIXIUM_CHAN_SKEW;
}
static inline int chan_on_cpu_memory(uint32_t c) // Same controller as main_ram?
{
    return TRIXIUM_CHAN_MOD ? (c / TRIXIUM_CHAN_MOD) == 0 : 1;
}
static inline uint32_t cpu_addr_of(uint64_t eng) // Engine-side -> CPU address (main_ram).
{
    return (uint32_t)(0x40000000ull + (eng - TRIXIUM_MEM_CPU_OFFSET));
}

static uint8_t *heap;
static void *halloc(uint32_t n)
{
    void *p = heap;
    heap = heap + ((n + 63) & ~63u);
    return p;
}

static uint8_t *heap_hi;
static void *halloc_hi(uint32_t n)
{
    if (heap_hi == 0)          // Small main_ram (the sim's 8 MB): no channel-3 overlap to
        return halloc(n);      // dodge there, the low heap serves everything.
    void *p = heap_hi;
    heap_hi = heap_hi + ((n + 63) & ~63u);
    return p;
}

/* Model State */
/*-------------*/

static const struct bons_header *H;
static int n_ch, head_dim, n_head, n_kv_head, n_embd, ffn_dim, n_layer;
static int max_ctx;
static int cpu_kv_ctx; // CPU-side KV cache depth (== max_ctx unless HW attention exists).

static float **attn_norm, **ffn_norm, **q_norm, **k_norm; // Per layer.
static float *out_norm;

static int8_t *k_cache, *v_cache;    // [layer][pos][kv_dim] int8.
static float  *k_scale, *v_scale;    // [layer][pos][kv_head].
static int    n_pos;

static float *x, *h, *q, *k, *v, *attn, *gate, *up, *scores;
static int ex_state; // Exponent of the resident x (SP_XR).
// Absmax of the resident x mantissas: resid_add sizes its shifts from it so the residual
// keeps its full 15-bit mantissa (see op_resid_add_sp). Both embedding paths normalize the
// mantissas into [2^14, 2^15), so 32767 is a tight starting bound and resid_add returns the
// exact value from then on.
#define X_AMAX_AFTER_EMBED 32767u
static uint32_t x_amax;

/* Helpers */
/*---------*/

static float fp16_to_float(uint16_t b)
{
    uint32_t sign = (uint32_t)(b & 0x8000) << 16;
    uint32_t exp  = (b >> 10) & 0x1F;
    uint32_t frac = b & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) bits = sign;
        else { // Subnormal.
            float f = (float)frac*(1.0f/16777216.0f); // frac * 2^-24
            union { float f; uint32_t u; } u = { .f = f };
            bits = sign | u.u;
        }
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } u = { .u = bits };
    return u.f;
}

COLD static void rmsnorm_f(float *dst, const float *src, const float *w, int n, float eps)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += src[i]*src[i];
    float r = 1.0f/sqrtf(ss/n + eps);
    for (int i = 0; i < n; i++) dst[i] = src[i]*r*w[i];
}

// Quantize a float vector to int8 with a power-of-two scale 2^e8 (so downstream exponent
// algebra stays integer). Returns e8.
COLD static int quantize_f_pow2(const float *src, int n, uint32_t sp_off)
{
    perf_enter(OP_QUANT8);
    int e8 = f32_quant_e8(src, n);
    float is = ldexpf(1.0f, -e8);
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_off);
    for (int i = 0; i < n; i += 4) sp[i/4] = f32_pack_q8x4(src + i, is);
    perf_exit(OP_QUANT8);
    return e8;
}

/* KV Cache */
/*----------*/

// KV-to-HBM: per (layer, kv-head, K|V) region of Q8 row packets, position-striped
// (position t -> channel t % n_ch, local row t / n_ch). All groups of a row share one
// power-of-two scale 2^ek, stored as an exactly-representable fp16 ((ek+15)<<10), so the
// engine's Q8 scale path applies it bit-exactly and the exponent algebra stays integer.
static uint32_t kv_base;    // Per-channel byte address of the KV region (4 KB aligned).
static uint32_t kv_rb;      // Beats per KV row packet (cols = head_dim, Q8).
static uint32_t kv_region;  // Beats per (layer, kv-head, K|V) region.
static int8_t  *kv_ekmax;   // Running max ek per region (B selection); -128 = empty.

static uint32_t kv_region_addr(int l, int kvh, int kind)
{
    return kv_base + ((uint32_t)((l*n_kv_head + kvh)*2 + kind)*kv_region)*32;
}

static int kv_ek_idx(int l, int kvh, int kind)
{
    return (l*n_kv_head + kvh)*2 + kind;
}

// Quantize one head slice to a Q8 row packet in SP_KVP and store it to HBM.
COLD static void kv_store_row(int l, int kvh, int kind, int t, const float *vec)
{
    float amax = 0;
    for (int i = 0; i < head_dim; i++) {
        float a = f32_abs(vec[i]);
        if (a > amax) amax = a;
    }
    int ek = 0;
    if (amax > 0) {
        frexpf(amax, &ek);
        ek -= 7; // Mantissas in [2^6, 2^7).
    }
    volatile uint16_t *ps = (volatile uint16_t *)(SCRATCHPAD_BASE + SP_KVP);
    int groups = head_dim/32;
    for (int g = 0; g < 16; g++) // Scale beat: fp16(2^ek) per group, zero padding.
        ps[g] = (g < groups) ? (uint16_t)((ek + 15) << 10) : 0;
    float is = ldexpf(1.0f, -ek);
    volatile uint32_t *pc = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_KVP + 32);
    for (int i = 0; i < head_dim; i += 4) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++) {
            int q8 = f32_lround_i32(vec[i+b]*is);
            if (q8 >  127) q8 =  127;
            if (q8 < -127) q8 = -127;
            w |= ((uint32_t)(uint8_t)(int8_t)q8) << (8*b);
        }
        pc[i/4] = w;
    }
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_STORE;
    op.chan  = t % n_ch;
    op.wbase = kv_region_addr(l, kvh, kind) + (uint32_t)(t/n_ch)*kv_rb*32;
    op.wlen  = kv_rb;
    op.xsrc  = SP_KVP;
    engine_run(&op);
    int idx = kv_ek_idx(l, kvh, kind);
    if (ek > kv_ekmax[idx]) kv_ekmax[idx] = ek;
}

/* Attention Fold */
/*----------------*/

// The per-head attention fold in integers, emulating the float32 path bit-for-bit so tokens
// stay byte-identical, without the soft-float divide/ldexpf/frexpf/lroundf per head:
//   q  = RN24(1.0f / (float)P8)            -> (qm, qe): q = qm * 2^qe, qm in [2^23, 2^24)
//   c2 = ldexpf(q, E1)                     -> (qm, qe + E1), no rounding
//   hm = (float)(am >> k2) * c2            -> RN24 of the product: (m2, e2)
//   ea8 = frexpf(max hm) - 7               -> e2max + 24 - 7
//   Mh  = lroundf(ldexpf(c2, 15 - ea8))    -> qm * 2^(qe + E1 + 15 - ea8), half away from 0
#ifdef TRIXIUM_HAS_ZBB
// Zbb clz. GCC 10.1 knows no Zb*, so the encoding is spelled out:
// .insn i 0x13, 1, rd, rs1, 0x600 == clz rd, rs1 (0x60059513 for a0, a1). Defined for 0
// (returns 32), so no zero test is needed.
static inline __attribute__((always_inline)) int bitlen32(uint32_t x)
{
    uint32_t r;
    __asm__ (".insn i 0x13, 1, %0, %1, 0x600" : "=r"(r) : "r"(x));
    return 32 - (int)r;
}
#else
static inline __attribute__((always_inline)) int bitlen32(uint32_t x)
{
    return x ? 32 - __builtin_clz(x) : 0;
}
#endif

static void attn_f32_recip(uint32_t p8, uint32_t *qm, int *qe)
{
    // m = 2^(23+L)/p8 in (2^23, 2^24] with an exact remainder, on the 32-bit divider:
    // for sh = 23+L <= 31 directly; above that 2^sh = 2^k * 2^32 with k = sh-32 <= 9 and
    // 2^32 = q1*p8 + r1, so m = (q1 << k) + ((r1 << k) / p8) (r1 << k < 2^27 fits).
    int L = bitlen32(p8);                        // 2^(L-1) <= p8 < 2^L
    int sh = 23 + L;
    uint32_t m, r;
    if (sh <= 31) {
        uint32_t num = 1u << sh;
        m = num/p8; r = num - m*p8;
    } else {
        int k = sh - 32;
        uint32_t q1 = 0xFFFFFFFFu/p8;
        uint32_t r1 = 0xFFFFFFFFu - q1*p8 + 1u;  // 2^32 - q1*p8 in [1, p8]
        if (r1 == p8) { q1++; r1 = 0; }
        uint32_t t = r1 << k;
        m = (q1 << k) + t/p8; r = t - (t/p8)*p8;
    }
    if (2*r > p8 || (2*r == p8 && (m & 1))) m++; // round to nearest even
    int e = -sh;
    if (m == (1u << 24)) { m >>= 1; e++; }
    *qm = m;
    *qe = e;
}

static void attn_f32_mul(uint32_t a, uint32_t qm, int qe, uint32_t *m2, int *e2)
{
    // 32-bit throughout: 64-bit variable shifts are libgcc calls on RV32. a <= 2^14 and
    // qm < 2^24 give p < 2^38 and sh in 0..14, so m = p >> sh is (pl >> sh) | (ph << (32-sh))
    // exactly (the bits of ph that would overflow are zero because p < 2^L), and the
    // remainder/half live entirely in pl: bit-identical to the 64-bit form.
    uint64_t p = (uint64_t)a*qm;                 // mul + mulhu, no library call
    uint32_t ph = (uint32_t)(p >> 32), pl = (uint32_t)p;
    int L = ph ? 32 + bitlen32(ph) : bitlen32(pl);
    int sh = L - 24;                             // p >= 2^23 -> sh >= 0
    uint32_t m = sh ? ((pl >> sh) | (ph << (32 - sh))) : pl;
    if (sh > 0) {
        uint32_t rem = pl & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
    }
    if (m == (1u << 24)) { m >>= 1; sh++; }
    *m2 = m;
    *e2 = qe + sh;
}

static int32_t attn_mh_int(uint32_t qm, int sh)
{
    if (sh >= 0) {
        if (sh >= 8) return 32767;               // qm >= 2^23 -> >= 2^31 -> clamps
        uint32_t v = qm << sh;                   // qm <= 2^24, sh <= 7: fits.
        return (v > 32767) ? 32767 : (int32_t)v;
    }
    int r = -sh;
    if (r >= 26) return 0;                       // qm <= 2^24 < 2^(r-1): rounds to 0.
    uint32_t v = (qm + (1u << (r - 1))) >> r;    // lroundf: half away from 0; <= 2^25.
    return (v > 32767) ? 32767 : (int32_t)v;
}

static uint32_t aw_qm[32];   // Per-head c2 mantissa (float32 emulation).
static int      aw_qe[32];   // Per-head c2 exponent (qe + E1).
static uint32_t aw_hm2;      // max hm mantissa/exponent over heads.
static int      aw_he2, aw_hvalid;

// AXPY pre-shift keeping the int32 lane accumulators safe (bitexact.axpy_rpre): pure
// bit-length form so the descriptor sequencer can compute it in hardware.
static int axpy_rpre_c(uint64_t p8sum)
{
    // Bit length via clz: a 64-bit shift loop is several hundred cycles on RV32.
    uint32_t hi = (uint32_t)(p8sum >> 32), lo = (uint32_t)p8sum;
    int bl = hi ? 32 + bitlen32(hi) : bitlen32(lo);
    return bl + 5;
}

// B/R for a scores op over a region: pow-2 scales make e16 + B == 12 exactly
// (fp16_decompose: 2^ek = 1024 * 2^(ek-10); B = 12 - (ekmax - 10) = 22 - ekmax), so the
// worst-row acc bound is closed-form: groups * (32*127*127) * 1024 << 12.
static int kv_br_r = -1; // Model constant (head_dim only), computed once at infer_init.

static void kv_scores_br(int l, int kvh, int kind, int *B, int *R)
{
    *B = 22 - kv_ekmax[kv_ek_idx(l, kvh, kind)];
    *R = kv_br_r;
}

/* Float Matvec */
/*--------------*/

// Hybrid float-result staging.  CPU reads of the scratchpad use a registered 32-bit
// Wishbone sideband (several round trips per word).  On the XCU1525 the engine can STORE
// the gathered SP_Y stream into controller-0 main RAM, where the CPU consumes it through
// its burst-filled D-cache.  This buffer is read-only to the CPU, so post-DMA invalidation
// cannot discard dirty data.  Targets without STORE + Zicbom keep the direct SP path.
static int32_t *i32_stage;

static void matvec_engine_rows(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                               uint32_t sp_dst)
{
    uint32_t rb = bons_row_beats(e->cols, e->kind, e->group);
    struct engine_op op = {0};
    op.kind   = (e->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
    op.rows   = rows;
    op.cols   = e->cols;
    op.wbase  = TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)row0*rb)*32;
    op.wlen   = rows*rb;
    op.sbeats = bons_scale_beats(e->cols, e->group);
    op.cbeats = bons_code_beats(e->cols, e->kind);
    op.bias   = e->bias;
    op.rshift = e->rshift;
    op.xsrc   = SP_X;
    op.dst    = sp_dst;
    engine_run(&op);
}

static const volatile int32_t *stage_i32_results_to(uint32_t sp_src, uint32_t count,
                                                     int dma_result, int32_t *stage)
{
    const volatile int32_t *y = (volatile int32_t *)(SCRATCHPAD_BASE + sp_src);
#if defined(HAS_INVD_CPU_DCACHE_RANGE)
    if (dma_result) {
        uint32_t eng_addr = (uint32_t)((uintptr_t)stage - MAIN_RAM_BASE
                                      + TRIXIUM_MEM_CPU_OFFSET);
        engine_store(0, eng_addr, (count*4 + 31)/32, sp_src);
        invd_cpu_dcache_range(stage, count*4);
        y = stage;
    }
#else
    (void)dma_result;
    (void)stage;
#endif
    return y;
}

static const volatile int32_t *stage_i32_results(uint32_t sp_src, uint32_t count,
                                                  int dma_result)
{
    return stage_i32_results_to(sp_src, count, dma_result, i32_stage);
}

// Run an engine matvec for tensor e on rows [row0, row0+rows) per channel; results (float,
// scaled by xscale) into dst[gathered global order: chan*rows + row]. With `acc` the int32
// results are added into acc[] instead (a K-segment of a wide matvec; the caller scales).
COLD static void matvec_rows_f(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                               float xscale, float *dst, int32_t *acc, int acc_add, int dma_result)
{
    matvec_engine_rows(e, row0, rows, SP_Y);

    perf_enter(OP_NORM_I32);
    uint32_t count = rows*n_ch;
    const volatile int32_t *y = stage_i32_results(SP_Y, count, dma_result);
    if (acc) {
        if (acc != (const int32_t *)y || acc_add) {
            uint32_t i = 0;
            if (acc_add) {
                for (; i + 3 < count; i += 4) {
                    acc[i + 0] += y[i + 0]; acc[i + 1] += y[i + 1];
                    acc[i + 2] += y[i + 2]; acc[i + 3] += y[i + 3];
                }
                for (; i < count; i++) acc[i] += y[i];
            } else {
                for (; i + 3 < count; i += 4) {
                    acc[i + 0] = y[i + 0]; acc[i + 1] = y[i + 1];
                    acc[i + 2] = y[i + 2]; acc[i + 3] = y[i + 3];
                }
                for (; i < count; i++) acc[i] = y[i];
            }
        }
    } else {
        float sc = ldexpf(xscale, (int)e->rshift - (int)e->bias);
        uint32_t i = 0;
        // Four independent conversions let the in-order RV32F core overlap fcvt/fmul
        // latency and remove three quarters of the loop-control instructions.
        for (; i + 3 < count; i += 4) {
            float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
            float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
            dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
            dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
        }
        for (; i < count; i++) dst[i] = (float)y[i]*sc;
    }
    perf_exit(OP_NORM_I32);
}

// Full matvec (all rows), output in original row order (block shard: chan-major == row order).
// The float matvec the hybrid CPU path runs (hybrid.c): quantize the input to int8 with a
// power-of-two scale, engine, float results in row order - chunked so the int32 results
// stay inside SP_Y..SP_H (96 KB) whatever the row count (the 27B's LM head is 15,520 rows
// per channel).
static float *rot_scratch;   // (SP_H - SP_Y)/4 floats/ints, allocated on first use.

static void ensure_float_matvec_scratch(void)
{
    if (!rot_scratch) {
        rot_scratch = halloc((SP_H - SP_Y));
        i32_stage   = halloc((SP_H - SP_Y));
    }
}

static void matvec_f_one(const struct bons_entry *e, float xs, float *dst, int32_t *acc,
                         int acc_add)
{
    uint32_t rpc   = e->rows/n_ch;
    uint32_t chunk = (SP_H - SP_Y)/(4*n_ch);
    if (chunk > rpc) chunk = rpc;
    for (uint32_t r0 = 0; r0 < rpc; r0 += chunk) {
        uint32_t n = (rpc - r0 < chunk) ? rpc - r0 : chunk;
        // matvec_rows_f gathers chan*n + row: unshuffle into row order per channel.
        if (n == rpc) {
            matvec_rows_f(e, r0, n, xs, dst, acc, acc_add, 1);
        } else if (acc) {
            // K-segment: the int32 results arrive as [chan][n]; add them at [chan][rpc].
            matvec_rows_f(e, r0, n, xs, 0, (int32_t *)rot_scratch, 0, 1);
            for (int c = 0; c < n_ch; c++)
                for (uint32_t i = 0; i < n; i++)
                    if (acc_add)
                        acc[(uint32_t)c*rpc + r0 + i] += ((int32_t *)rot_scratch)[(uint32_t)c*n + i];
                    else
                        acc[(uint32_t)c*rpc + r0 + i] = ((int32_t *)rot_scratch)[(uint32_t)c*n + i];
        } else {
            // Results arrive as [chan][n] in rot_scratch; scatter them into the caller's
            // [chan][rpc] row order.
            matvec_rows_f(e, r0, n, xs, rot_scratch, 0, 0, 1);
            for (int c = 0; c < n_ch; c++)
                for (uint32_t i = 0; i < n; i++)
                    dst[(uint32_t)c*rpc + r0 + i] = rot_scratch[(uint32_t)c*n + i];
        }
    }
}

// K-segments (FLAG_KSEG, the 27B's FFN down projection): the whole input is quantised once
// with one exponent, each segment's int8 slice goes to SP_X, the engine's int32 results add
// up in acc[] (every segment carries the tensor's bias/rshift, so they are in one fixed
// point) and are scaled once - the unsplit result up to one LSB of the per-segment shift.
uint32_t infer_matvec_cols(const struct bons_entry *e)   // K of the whole matvec.
{
    if (!(e->flags & FLAG_KSEG))
        return e->cols;
    uint32_t cols = 0;
    int nseg = model_entry_nkseg(e);   // once, not in the loop condition
    for (int k = 0; k < nseg; k++)
        cols += model_entry_kseg(e, k)->cols;
    return cols;
}

static int32_t *kseg_acc;    // rows int32, sized on first use to the widest such tensor.

int infer_quantize_f(const float *src, uint32_t n)
{
    return quantize_f_pow2(src, n, SP_X);
}

/* Fused Prism Rotation */
/*----------------------*/

// Rather than materialising every Hadamard block as FP32 in main RAM, scanning the
// concatenated vector for its Q8 exponent, then reading it all again to feed the immediately
// following projection: a rotation result is an int32 engine value times a power of two, so
// retain those integers in the cacheable DMA buffer and remember the exponent per block.
// The global exponent comes from the engine's exact absmax, and packing combines the two
// power-of-two scales around the same RV32F conversion.  Thus the Q8 bytes are identical,
// without the intermediate float write, float scan and float reload.
#define ROT_BLOCK_MAX 64
static int rot_block_e[ROT_BLOCK_MAX];
static int rot_stage_width, rot_stage_block, rot_stage_e8;

static void pack_rotated_range(uint32_t c0, uint32_t n, uint32_t sp_dst)
{
    const int32_t *raw = (const int32_t *)rot_scratch;
    volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_dst);
    uint32_t end = c0 + n, out = 0;
    for (uint32_t i = c0; i < end;) {
        uint32_t b = i/(uint32_t)rot_stage_block;
        uint32_t bend = (b + 1u)*(uint32_t)rot_stage_block;
        if (bend > end) bend = end;
        float is = ldexpf(1.0f, rot_block_e[b] - rot_stage_e8);
        for (; i < bend; i += 4)
            x[out++] = i32_pack_q8x4(raw + i, is);
    }
}

int infer_rotate_quantize_f(const float *src, uint32_t width)
{
#if !defined(HAS_INVD_CPU_DCACHE_RANGE)
    (void)src;
    (void)width;
    return INFER_ROTATE_UNSUPPORTED;
#else
    ensure_float_matvec_scratch();
    int B = H->hadamard_block, wi = -1;
    for (int i = 0; i < 4; i++)
        if (H->hadamard_widths[i] == width) wi = i;
    if (!B || wi < 0 || width % (uint32_t)B || B % n_ch ||
        width/(uint32_t)B > ROT_BLOCK_MAX || width*4u > (SP_H - SP_Y)) {
        printf("hybrid: unsupported fused rotation width %lu block %d\n",
               (unsigned long)width, B);
        return INFER_ROTATE_UNSUPPORTED;
    }

    rot_stage_width = (int)width;
    rot_stage_block = B;
    uint32_t max_bits = 0;
    int32_t *raw = (int32_t *)rot_scratch;
    for (uint32_t b = 0; b < width/(uint32_t)B; b++) {
        const struct bons_entry *re = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, b));
        int xe = quantize_f_pow2(src + b*(uint32_t)B, B, SP_X);
        matvec_engine_rows(re, 0, B/n_ch, SP_Y);
#ifdef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
        uint32_t amax = trixium_engine_csr_absmax_read();
#else
        uint32_t amax = 0;
#endif
        perf_enter(OP_NORM_I32);
        const volatile int32_t *y = stage_i32_results_to(
            SP_Y, B, 1, raw + b*(uint32_t)B);
#ifndef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
        for (int i = 0; i < B; i++) {
            int64_t v = y[i];
            uint32_t a = (uint32_t)(v < 0 ? -v : v);
            if (a > amax) amax = a;
        }
#else
        (void)y;
#endif
        perf_exit(OP_NORM_I32);
        int oe = xe + (int)re->rshift - (int)re->bias;
        rot_block_e[b] = oe;
        float mf = (float)amax*ldexpf(1.0f, oe);
        uint32_t bits = f32_abs_bits(mf);
        if (bits <= 0x7f800000u && bits > max_bits) max_bits = bits;
    }
    rot_stage_e8 = f32_e8_from_amax_bits(max_bits);
    if (width <= SP_Y - SP_X) {
        perf_enter(OP_QUANT8);
        pack_rotated_range(0, width, SP_X);
        perf_exit(OP_QUANT8);
    }
    return rot_stage_e8;
#endif
}

void infer_matvec_f_rotated_quantized(const struct bons_entry *e, int e8, float *dst)
{
    ensure_float_matvec_scratch();
    uint32_t cols = infer_matvec_cols(e);
    if (cols <= SP_Y - SP_X) {
        infer_matvec_f_quantized(e, e8, dst);
        return;
    }
    if (!(e->flags & FLAG_KSEG) || cols != (uint32_t)rot_stage_width ||
        e8 != rot_stage_e8 || (uint64_t)e->rows*4 > (SP_H - SP_YU)) {
        printf("hybrid: unsupported fused wide projection K%lu R%lu\n",
               (unsigned long)cols, (unsigned long)e->rows);
        return;
    }

    int nseg = model_entry_nkseg(e);
    uint32_t c0 = 0;
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = model_entry_kseg(e, k);
        perf_enter(OP_QUANT8);
        pack_rotated_range(c0, seg->cols, SP_X);
        perf_exit(OP_QUANT8);
        matvec_engine_rows(seg, 0, seg->rows/n_ch, k ? SP_YU : SP_Y);
        if (k) {
            perf_enter(OP_NORM_I32);
            int32_t a[4] = {SP_Y, SP_YU, SP_Y, (int32_t)((e->rows + 7)/8)};
            vec_run(PROG_REDUCE_ACC, a, 4);
            perf_exit(OP_NORM_I32);
        }
        c0 += seg->cols;
    }
    perf_enter(OP_NORM_I32);
    const volatile int32_t *y = stage_i32_results(SP_Y, e->rows, 1);
    float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
    uint32_t i = 0;
    for (; i + 3 < e->rows; i += 4) {
        float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
        float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
        dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
        dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
    }
    for (; i < e->rows; i++) dst[i] = (float)y[i]*sc;
    perf_exit(OP_NORM_I32);
}

/* Hybrid Path Primitives */
/*------------------------*/

void infer_matvec_f_quantized(const struct bons_entry *e, int e8, float *dst)
{
    // SP_X is read-only to the engine, so consecutive non-K-segment projections with the
    // same input can reuse its exact bytes and exponent.  hybrid.c is the only caller.
    ensure_float_matvec_scratch();
    matvec_f_one(e, ldexpf(1.0f, e8), dst, 0, 0);
}

void infer_matvec_f(const struct bons_entry *e, const float *src, float *dst)
{
    ensure_float_matvec_scratch();
    if (!(e->flags & FLAG_KSEG)) {
        int e8 = quantize_f_pow2(src, e->cols, SP_X);
        matvec_f_one(e, ldexpf(1.0f, e8), dst, 0, 0);
        return;
    }
    int nseg = model_entry_nkseg(e);
    uint32_t cols = infer_matvec_cols(e);
    if (!kseg_acc) {
        // The widest K-split tensor decides both buffers (the table is small; once).
        uint32_t max_rows = 0;
        const struct bons_entry *tab = model_entry_at(0);
        for (uint32_t i = 0; i < H->n_tensors; i++) {
            const struct bons_entry *t = &tab[i];
            if (!(t->flags & FLAG_KSEG) || t->kseg != 0) continue;
            if (t->rows > max_rows) max_rows = t->rows;
        }
        kseg_acc = halloc(max_rows*4);
    }
    // One exponent for the whole vector (quantize_f_pow2's rule: amax's exponent - 7).
    int e8 = f32_quant_e8(src, cols);
    float is = ldexpf(1.0f, -e8);
    uint32_t c0 = 0;
    // The 27B's two FFN-down segments each produce one 20 KiB int32 vector.  Keep both
    // inside the two scratchpad result banks, add later segments into the first with the
    // existing 8-lane vector program, then DMA/convert only the final sum.  This removes
    // one controller-0 STORE plus both CPU-side accumulator passes per layer.  Wider result
    // tensors retain the generic main-RAM accumulator below.
    if ((uint64_t)e->rows*4 <= (SP_H - SP_YU)) {
        for (int k = 0; k < nseg; k++) {
            const struct bons_entry *seg = model_entry_kseg(e, k);
            perf_enter(OP_QUANT8);
            volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_X);
            for (uint32_t i = 0; i < seg->cols; i += 4)
                x[i/4] = f32_pack_q8x4(src + c0 + i, is);
            perf_exit(OP_QUANT8);
            matvec_engine_rows(seg, 0, seg->rows/n_ch, k ? SP_YU : SP_Y);
            if (k) {
                perf_enter(OP_NORM_I32);
                int32_t a[4] = {SP_Y, SP_YU, SP_Y, (int32_t)((e->rows + 7)/8)};
                vec_run(PROG_REDUCE_ACC, a, 4);
                perf_exit(OP_NORM_I32);
            }
            c0 += seg->cols;
        }
        perf_enter(OP_NORM_I32);
        const volatile int32_t *y = stage_i32_results(SP_Y, e->rows, 1);
        float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
        uint32_t i = 0;
        for (; i + 3 < e->rows; i += 4) {
            float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
            float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
            dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
            dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
        }
        for (; i < e->rows; i++) dst[i] = (float)y[i]*sc;
        perf_exit(OP_NORM_I32);
        return;
    }
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = model_entry_kseg(e, k);
        // The shared exponent is what makes quantisation global; quantising each slice
        // directly to SP_X produces the same bytes as quantising the whole vector first,
        // without the extra memory passes.
        perf_enter(OP_QUANT8);
        volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_X);
        for (uint32_t i = 0; i < seg->cols; i += 4)
            x[i/4] = f32_pack_q8x4(src + c0 + i, is);
        perf_exit(OP_QUANT8);
        matvec_f_one(seg, 1.0f, 0, kseg_acc, k != 0);
        c0 += seg->cols;
    }
    float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
    uint32_t i = 0;
    for (; i + 3 < e->rows; i += 4) {
        float f0 = (float)kseg_acc[i + 0], f1 = (float)kseg_acc[i + 1];
        float f2 = (float)kseg_acc[i + 2], f3 = (float)kseg_acc[i + 3];
        dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
        dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
    }
    for (; i < e->rows; i++) dst[i] = (float)kseg_acc[i]*sc;
}

void *infer_halloc(uint64_t n, int big)
{
    uint8_t *lim = big && heap_hi ? (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (2u << 20))
                                  : (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (1u << 20));
    uint8_t *p = big && heap_hi ? heap_hi : heap;
    if ((uint64_t)(lim - p) < n) return NULL;
    return big && heap_hi ? halloc_hi((uint32_t)n) : halloc((uint32_t)n);
}

uint64_t infer_heap_room(int big)
{
    uint8_t *lim = (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (1u << 20));
#if !defined(TRIXIUM_CHAN_MOD) || TRIXIUM_CHAN_MOD == 0
    // One flat memory (the FK33): the low heap ends where the first engine window inside
    // main_ram starts (channel 3's, at +64 MB) - hybrid models are not this board's.
    if (MAIN_RAM_SIZE > (64u << 20)) lim = (uint8_t *)(0x40000000u + (64u << 20));
#endif
    uint8_t *p = big && heap_hi ? heap_hi : heap;
    if (big && heap_hi) lim = (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (2u << 20));
    return p < lim ? (uint64_t)(lim - p) : 0;
}

void infer_fetch_f32(uint32_t addr, uint32_t beats, float *dst, int n)
{
    // Through the fetch staging region, 8 KB at a time.
    int done = 0;
    while (done < n) {
        uint32_t b = beats - (uint32_t)done/8;
        if (b > 256) b = 256;
        engine_fetch(0, addr + (uint32_t)done*4, b, SP_FET);
        volatile float *p = (volatile float *)(SCRATCHPAD_BASE + SP_FET);
        int m = (int)b*8;
        if (m > n - done) m = n - done;
        for (int i = 0; i < m; i++) dst[done + i] = p[i];
        done += m;
    }
}

/* CPU Embedding and RoPE */
/*------------------------*/

// Embedding row: fetch the row packet from HBM and dequantize (ternary).
COLD static void embed_row_f(const struct bons_entry *e, int row, float *dst)
{
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch; // Rows per channel.
    uint32_t ch  = row/rpc;
    uint32_t loc = row % rpc;
    engine_fetch(ch, TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)loc*rb)*32,
                 rb, SP_FET);
    volatile uint8_t *p = (volatile uint8_t *)(SCRATCHPAD_BASE + SP_FET);
    uint32_t sb = bons_scale_beats(e->cols, e->group);
    const volatile uint16_t *scales = (const volatile uint16_t *)p;
    const volatile uint8_t  *codes  = p + sb*32;
    for (int g = 0; g < (int)(e->cols/128); g++) {
        float s = fp16_to_float(scales[g]);
        for (int j = 0; j < 128; j++) {
            int q8 = (codes[g*32 + j/4] >> ((j & 3)*2)) & 3;
            dst[g*128 + j] = (float)(q8 - 1)*s;
        }
    }
}

// RoPE with YaRN (transliterated from the golden model / llama.cpp).
// Soft-float transcendentals dominate the CPU cost, so everything position-independent is
// precomputed once (rope_init), and cos/sin are computed once per position (rope_pos) and
// shared by all heads and by the Q and K applications.
static float *rope_freq_i;  // Per pair: interpolated angular frequency.
static float *rope_freq_e;  // Per pair: extrapolated (ramp-blended in).
static float  rope_mscale;
static float *rope_c, *rope_s; // Per-pair cos/sin for the current position.
static int    rope_pos_cached;

static float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base)
{
    return n_dims*logf(n_ctx_orig/(n_rot*2.0f*(float)M_PI))/(2.0f*logf(base));
}

static void rope_init(void)
{
    // Bonsai GGUFs use yarn factor 4; tiny sim models use plain rope (vocab heuristic until
    // the BONS header carries yarn fields).
    float theta_base = (float)H->rope_theta;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    if (H->vocab > 100000) {
        freq_scale = 0.25f;
        ext_factor = 1.0f;
    }
    int   D   = head_dim;
    float low = 0, high = D - 1;
    rope_mscale = 1.0f;
    if (ext_factor != 0.0f) {
        int orig = (n_embd >= 4096) ? 16384 : 8192;
        low  = floorf(yarn_corr_dim(D, orig, 32.0f, theta_base));
        high = ceilf(yarn_corr_dim(D, orig, 1.0f, theta_base));
        if (low  < 0)     low  = 0;
        if (high > D - 1) high = D - 1;
        rope_mscale *= 1.0f + 0.1f*logf(1.0f/freq_scale);
    }
    rope_freq_i = halloc((D/2)*sizeof(float));
    rope_freq_e = halloc((D/2)*sizeof(float));
    rope_c      = halloc((D/2)*sizeof(float));
    rope_s      = halloc((D/2)*sizeof(float));
    for (int p2 = 0; p2 < D/2; p2++) {
        float fe   = powf(theta_base, -2.0f*p2/D); // Extrapolated frequency.
        float ramp = 0.0f;
        if (ext_factor != 0.0f) {
            float y = (p2 - low)/((high - low) > 0.001f ? (high - low) : 0.001f);
            ramp    = (1.0f - fminf(1.0f, fmaxf(0.0f, y)))*ext_factor;
        }
        // theta(pos) = pos * (freq_scale*fe*(1-ramp) + fe*ramp).
        rope_freq_i[p2] = freq_scale*fe*(1.0f - ramp) + fe*ramp;
        rope_freq_e[p2] = fe;
    }
    rope_pos_cached = -1;
}

static void rope_pos(int pos)
{
    if (pos == rope_pos_cached)
        return;
    for (int p2 = 0; p2 < head_dim/2; p2++) {
        float theta = pos*rope_freq_i[p2];
        rope_c[p2] = cosf(theta)*rope_mscale;
        rope_s[p2] = sinf(theta)*rope_mscale;
    }
    rope_pos_cached = pos;
}

COLD static void rope_apply(float *vec, int n_heads_v, int pos)
{
    int D = head_dim;
    rope_pos(pos);
    for (int hh = 0; hh < n_heads_v; hh++) {
        float *vh = vec + hh*D;
        for (int p2 = 0; p2 < D/2; p2++) {
            float c = rope_c[p2], s = rope_s[p2];
            float x0 = vh[p2], x1 = vh[p2 + D/2];
            vh[p2]       = x0*c - x1*s;
            vh[p2 + D/2] = x0*s + x1*c;
        }
    }
}

/* Resident BFP Ops */
/*------------------*/

// Every op has a CPU implementation (float math over the int16 scratchpad regions - the
// always-available fallback) and, when the vector unit exists, a HW implementation running
// the corresponding microcode program. Exponent algebra stays in this file; ops receive
// shift amounts only.

// Score padding granule: softmax8's x2 unroll (ISA rev 2 microcode) walks two 16-entry
// blocks per iteration and has no odd-tail form, so scores are padded to a 32-multiple
// with INT32_MIN there (those entries emit p8 = 0 - the sum and the AXPY do not change).
#if TRIXIUM_PROGS_VREGS >= 16
#define SC_PAD 32
#else
#define SC_PAD 16
#endif
#define SC_PADN(cnt) ((SC_PAD - ((cnt) & (SC_PAD - 1))) & (SC_PAD - 1))

static inline volatile int16_t *sp16(uint32_t off) { return (volatile int16_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int32_t *sp32(uint32_t off) { return (volatile int32_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int8_t  *sp8(uint32_t off)  { return (volatile int8_t  *)(SCRATCHPAD_BASE + off); }

// rmsnorm: SP int16(src,esrc) -> SP int16(dst) at fixed E_NORM. HW path: fetch the
// image's int16 norm rows (kind norm16) into SP_FET and run the vector unit's two-pass
// program; CPU path: soft-float with the F32 weights from RAM.
//
// The int16 rows carry a per-tensor exponent in the entry's bias byte (BONS v2): a fixed
// exponent would clip the gammas (they reach 153 on the 1.7B). The program's A6 is the
// combined shift; the sequencer lists get theirs patched per layer next to their A5 (eps)
// and A1 (resident rows).
static inline int w16_shift(const struct bons_entry *e)
{
    return (int)(int8_t)e->bias - E_NORM - 15;
}

// rms_eps as an exact integer mantissa/exponent, so the per-call eps does not cost soft-float.
//
// The value wanted is lroundf(rms_eps * 2^-2esrc) - a pure power-of-two scaling, so float32
// computes it with no rounding beyond rms_eps's own representation. Decomposing once into a
// 24-bit mantissa (all a float32 has) makes the per-call work a shift and a rounding add,
// bit-identical to the float path, instead of soft-float ldexpf+lroundf per head norm.
static uint32_t eps_m;   // rms_eps = eps_m * 2^(eps_e - 24), eps_m in [2^23, 2^24)
static int      eps_e;

static void eps_init(void)
{
    float m = frexpf(H->rms_eps, &eps_e);       // m in [0.5, 1)
    eps_m   = (uint32_t)(ldexpf(m, 24) + 0.5f); // exact: m has at most 24 significant bits
}

static int eps_for(int esrc)
{
    int s = 24 + 2*esrc - eps_e;
    if (s <= 0) {
        // eps_m < 2^24, so the value fits int32 up to a 7-bit left shift (the float path
        // does not overflow here: lroundf(1e-6 * 2^44) is 1.76e7). It fires when the
        // residual absmax is below 2^-4, i.e. on the tiny embeddings of the chat template's
        // special tokens (<|im_start|> at 2^-22), never on a plain-text prompt.
        return s < -7 ? 0x7FFFFFFF : (int)(eps_m << (-s));
    }
    if (s > 31)
        return 1;                                // rounds to 0; callers clamp to >= 1 anyway
    uint32_t v = (eps_m + (1u << (s - 1))) >> s; // round half away from zero (values > 0)
    return (int)(v < 1 ? 1 : v);
}

COLD static void op_rmsnorm_sp(uint32_t src, int esrc, const float *w,
                               const struct bons_entry *w16, int n, uint32_t dst)
{
    if (op_on_hw(OP_RMSNORM) && w16) {
        engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + w16->chan_base_beats*32,
                     w16->chan_len_beats, SP_FET);
        int eps_int = eps_for(esrc);
        int ln = 0;
        while ((1 << ln) < n) ln++;
        int32_t args[7] = {(int32_t)src, (int32_t)SP_FET, (int32_t)dst, n/16,
                           -ln, eps_int, w16_shift(w16)};
        vec_run(PROG_RMSNORM, args, 7);
        return;
    }
    volatile int16_t *ps = sp16(src);
    volatile int16_t *pd = sp16(dst);
    float es = ldexpf(1.0f, esrc);
    float ss = 0;
    for (int i = 0; i < n; i++) {
        float v = (float)ps[i]*es;
        ss += v*v;
    }
    float r  = 1.0f/sqrtf(ss/n + H->rms_eps);
    float os = ldexpf(1.0f, -E_NORM);
    for (int i = 0; i < n; i++) {
        float v = (float)ps[i]*es*r*w[i]*os;
        int32_t q = (int32_t)lroundf(v);
        if (q >  32767) q =  32767;
        if (q < -32767) q = -32767;
        pd[i] = (int16_t)q;
    }
}

// quant8: SP int16 -> SP int8 (RNE >> s). Returns s; caller: out exponent = e + s.
//
// s comes from the source magnitude, not a fixed 8. After rmsnorm the mantissas sit near
// 2**11, so >>8 would hand the matvec int8 inputs of about +-9 - 3 of the 7 bits available -
// and cost ~0.30 rel-rms on every matvec output. bit_length(amax) - 7 is clip-free by
// construction (amax >> s < 128) and recovers a full order of magnitude of accuracy.
// The program measures its own absmax (no producer has a spare scalar to return one in), so
// both paths here derive the same shift from the same absmax. s may be negative: a small
// source is scaled *up* into the int8 range.
static int quant8_shift(uint32_t amax)
{
    if (amax == 0) return 8; // VCLZN(0) == 0, i.e. pre-shift 0 - keep the paths identical.
    return bitlen32(amax) - 7;
}

static int op_quant8_sp(uint32_t src, uint32_t dst, int n);

// The norm-weight row fetch that opens every rmsnorm_q8 would otherwise be a blocking engine
// op between two vector ops, with both units idle. The caller submits it BEFORE the
// preceding resid_add so it streams underneath, joins, and flags it; the op then skips its
// own fetch.
static int rq8_prefetched;

static uint32_t norm_fetch_submit(const struct bons_entry *w16)
{
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_FETCH;
    op.wbase = TRIXIUM_CHAN_WEIGHT_BASE + w16->chan_base_beats*32;
    op.wlen  = w16->chan_len_beats;
    op.dst   = SP_FET;
    return engine_submit(&op);
}

static int op_resid_add_sp(uint32_t a, int ea, uint32_t b, int eb, int n,
                           uint32_t bmax, uint32_t *amax_io);

// resid_add with the NEXT norm's weight row streaming underneath. SP_FET is idle here (its
// last reader, the previous rmsnorm or the qkr list, has finished) and the engine is idle
// (the preceding matvec/list joined) - same numerics either way.
static int resid_add_pf(uint32_t src, int esrc, uint32_t d, int ed, int n, uint32_t dmax,
                        uint32_t *xamax, const struct bons_entry *next_w16)
{
    uint32_t df = 0;
    int pf = next_w16 && op_on_hw(OP_RMSNORM) && op_on_hw(OP_QUANT8);
    if (pf) df = norm_fetch_submit(next_w16);
    perf_enter(OP_RESID_ADD);
    int e = op_resid_add_sp(src, esrc, d, ed, n, dmax, xamax);
    perf_exit(OP_RESID_ADD);
    if (pf) { engine_join(df); rq8_prefetched = 1; }
    return e;
}

// rmsnorm + measured quant8 in one program: saves a dispatch and quant8's absmax pass at
// every norm->x8 site. Returns the quant8 shift like op_quant8_sp; falls back to the two ops
// when either is off the hardware or the NORM16 row is absent.
static int op_rmsnorm_q8_sp(uint32_t src, int esrc, const float *w,
                            const struct bons_entry *w16, int n, uint32_t dst16,
                            uint32_t dst8)
{
    if (op_on_hw(OP_RMSNORM) && op_on_hw(OP_QUANT8) && w16) {
        if (rq8_prefetched)
            rq8_prefetched = 0;                  // Row already in SP_FET (resid_add_pf).
        else
            engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + w16->chan_base_beats*32,
                         w16->chan_len_beats, SP_FET);
        int eps_int = eps_for(esrc);
        int ln = 0;
        while ((1 << ln) < n) ln++;
        int32_t args[8] = {(int32_t)src, (int32_t)SP_FET, (int32_t)dst16, n/16,
                           -ln, eps_int, w16_shift(w16), (int32_t)dst8};
        perf_enter(OP_RMSNORM);
        uint32_t amax = (uint32_t)vec_run(PROG_RMSNORM_Q8, args, 8);
        perf_exit(OP_RMSNORM);
        return quant8_shift(amax);
    }
    perf_enter(OP_RMSNORM);
    op_rmsnorm_sp(src, esrc, w, w16, n, dst16);
    perf_exit(OP_RMSNORM);
    perf_enter(OP_QUANT8);
    int s = op_quant8_sp(dst16, dst8, n);
    perf_exit(OP_QUANT8);
    return s;
}

static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n);

// silu_mul + measured quant8 in one program. Returns the quant8 shift.
static int op_silu_q8_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n, uint32_t dst8)
{
    if (op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
        int32_t args[8] = {(int32_t)gsrc, (int32_t)usrc, (int32_t)gsrc, n/16,
                           eg + 6, 0, -15, (int32_t)dst8};
        perf_enter(OP_SILU_MUL);
        uint32_t amax = (uint32_t)vec_run(PROG_SILU_Q8, args, 8);
        perf_exit(OP_SILU_MUL);
        return quant8_shift(amax);
    }
    perf_enter(OP_SILU_MUL);
    op_silu_mul_sp(gsrc, eg, usrc, eu, n);
    perf_exit(OP_SILU_MUL);
    perf_enter(OP_QUANT8);
    int s = op_quant8_sp(gsrc, dst8, n);
    perf_exit(OP_QUANT8);
    return s;
}

COLD static int op_quant8_sp(uint32_t src, uint32_t dst, int n)
{
    if (op_on_hw(OP_QUANT8)) {
        int32_t args[4] = {(int32_t)src, 1, (int32_t)dst, n/16}; // A1 = 1: measure.
        return quant8_shift((uint32_t)vec_run(PROG_QUANT8, args, 4)); // S6 = source absmax.
    }
    volatile int16_t *ps = sp16(src);
    volatile int8_t  *pd = sp8(dst);
    uint32_t amax = 0;
    for (int i = 0; i < n; i++) {
        int32_t a = ps[i];
        uint32_t m = (uint32_t)(a < 0 ? -a : a);
        if (m > amax) amax = m;
    }
    int s = quant8_shift(amax);
    for (int i = 0; i < n; i++) {
        int32_t v = (s <= 8) ? ((int32_t)ps[i] << (8 - s)) : ((int32_t)ps[i] >> (s - 8));
        int32_t q = (v >> 8) + (((v >> 7) & 1) & (((v & 0x7F) != 0) | ((v >> 8) & 1)));
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        pd[i] = (int8_t)q;
    }
    return s;
}

// norm_i32: SP int32 -> SP int16 (floor >> k chosen from the engine absmax CSR).
// Returns k (caller: out exponent = in exponent + k); *omax (optional) = output absmax,
// which resid_add needs to size its own shifts.
// _a: absmax supplied by the caller - the overlap path captures the PREVIOUS engine
// op's absmax before submitting the next one (the tracker resets on command accept).
static int op_norm_i32_sp_a(uint32_t src, uint32_t dst, int n, uint32_t amax,
                            uint32_t *omax)
{
    int k = 0;
    while ((amax >> k) > 16383u) k++;
    if (omax) *omax = amax >> k;
    if (op_on_hw(OP_NORM_I32)) {
        int32_t args[5] = {(int32_t)src, 0, (int32_t)dst, n/16, k};
        vec_run(PROG_NORM_I32, args, 5);
        return k;
    }
    volatile int32_t *ps = sp32(src);
    volatile int16_t *pd = sp16(dst);
    for (int i = 0; i < n; i++) {
        int32_t v = ps[i] >> k;
        if (v >  32767) v =  32767;
        if (v < -32767) v = -32767;
        pd[i] = (int16_t)v;
    }
    return k;
}

static int op_norm_i32_sp_m(uint32_t src, uint32_t dst, int n, uint32_t *omax)
{
#ifdef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
    uint32_t amax = trixium_engine_csr_absmax_read();
#else
    // Gateware without the engine absmax tracker: scan on the CPU.
    uint32_t amax = 0;
    {
        volatile int32_t *pa = sp32(src);
        for (int i = 0; i < n; i++) {
            uint32_t a = (uint32_t)((pa[i] < 0) ? -pa[i] : pa[i]);
            if (a > amax) amax = a;
        }
    }
#endif
    return op_norm_i32_sp_a(src, dst, n, amax, omax);
}

static inline int op_norm_i32_sp(uint32_t src, uint32_t dst, int n)
{
    return op_norm_i32_sp_m(src, dst, n, 0);
}

// silu_mul: SP int16(gate,eg) x SP int16(up,eu) -> SP int16(gate) at eg+eu+15.
COLD static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n)
{
    if (op_on_hw(OP_SILU_MUL)) {
        int32_t args[7] = {(int32_t)gsrc, (int32_t)usrc, (int32_t)gsrc, n/16,
                           eg + 6, 0, -15}; // A4 = direct shift (VSHLS convention).
        vec_run(PROG_SILU_MUL, args, 7);
        return;
    }
    volatile int16_t *pg = sp16(gsrc);
    volatile int16_t *pu = sp16(usrc);
    float gs = ldexpf(1.0f, eg), us = ldexpf(1.0f, eu);
    float os = ldexpf(1.0f, -(eg + eu + 15));
    for (int i = 0; i < n; i++) {
        float g = (float)pg[i]*gs;
        float v = g/(1.0f + expf(-g))*(float)pu[i]*us*os;
        int32_t q = (int32_t)lroundf(v);
        if (q >  32767) q =  32767;
        if (q < -32767) q = -32767;
        pg[i] = (int16_t)q;
    }
}

// resid_add: SP int16(a,ea) + SP int16(b,eb) -> SP int16(a) at eo. Returns eo and, via
// *amax_io, the output absmax (in: the absmax of a).
//
// eo comes from the operands' *measured* magnitudes. A fixed eo = max(ea,eb)+1 (the bound
// for two full 15-bit mantissas) spends a mantissa bit on every call, twice per layer: on the
// 1.7B the residual would be down to a single bit by layer 11 and the output garbage. Sizing
// eo from the real absmax keeps ~14 bits at every layer. Shifts are signed (VSHLV: left when
// negative here).
static int op_resid_add_sp(uint32_t a, int ea, uint32_t b, int eb, int n,
                           uint32_t bmax, uint32_t *amax_io)
{
    int emin = (ea < eb) ? ea : eb;
    // Bound the sum at the common exponent, then keep only the headroom it actually needs.
    uint64_t bound = ((uint64_t)*amax_io << (ea - emin)) + ((uint64_t)bmax << (eb - emin));
    int k = 0;
    while ((bound >> k) > 32767) k++;
    int eo = emin + k;
    int sa = eo - ea, sb = eo - eb;
    if (op_on_hw(OP_RESID_ADD)) {
        int32_t args[6] = {(int32_t)a, (int32_t)b, (int32_t)a, n/16, -sa, -sb};
        *amax_io = (uint32_t)vec_run(PROG_RESID_ADD, args, 6); // S6 = output absmax.
        return eo;
    }
    volatile int16_t *pa = sp16(a);
    volatile int16_t *pb = sp16(b);
    uint32_t omax = 0;
    for (int i = 0; i < n; i++) {
        int32_t va = (sa >= 0) ? ((int32_t)pa[i] >> sa) : ((int32_t)pa[i] << -sa);
        int32_t vb = (sb >= 0) ? ((int32_t)pb[i] >> sb) : ((int32_t)pb[i] << -sb);
        int32_t v  = va + vb;
        if (v >  32767) v =  32767;
        if (v < -32767) v = -32767;
        pa[i] = (int16_t)v;
        uint32_t m = (uint32_t)(v < 0 ? -v : v);
        if (m > omax) omax = m;
    }
    *amax_io = omax;
    return eo;
}

/* Resident Attention and KV Store */
/*---------------------------------*/

// Resident attention front (all attention hwops on): q/k/v live in the scratchpad as
// int16 from the matvec on; qk-norm/rope/kv-pack run on the vector unit; the CPU float
// K/V cache is not maintained (attention op bits are latched per generation - flipping
// them mid-generation is a debug-only move).
static int attn_all_hw(void)
{
    return op_on_hw(OP_QKNORM) && op_on_hw(OP_ROPE) && op_on_hw(OP_KVQUANT)
        && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV);
}

// vec quant8 head slice -> int8 at dst (exponent = es + 8). Fixed shift on purpose: the
// sequencer patches the softmax scale (which folds in this exponent) into a program it then
// runs head after head with no CPU in the loop, so the shift has to be known in advance.
// The measuring quant8 is used where firmware does the exponent algebra itself.
static void vec_quant8_head(uint32_t sp, uint32_t dst, int n)
{
    int32_t args[4] = {(int32_t)sp, 0, (int32_t)dst, n/16}; // A1 = 0: fixed >>8.
    vec_run(PROG_QUANT8, args, 4);
}

// Pack a resident head as a KV row packet and store it (scale = 2^(es+8) exactly).
//
// Staging slots so the scale beat is written only when the exponent CHANGES, not per store:
// K's exponent is the constant E_NORM+8 and V's changes once per layer, and rewriting 16
// scratchpad halfwords is costly uncached bus traffic. One staging slot per (kv_head, kind):
// 16 x 512 B fills SP_KVP exactly. Two slots would do if pack and store alternated, but they
// must not: the vector unit and the engine share the scratchpad port, and interleaving them
// is ~1.6x slower than packing everything first and then storing.
#define KVP_SLOTS 16
static int kvp_scale_e8[KVP_SLOTS];

static uint32_t kvp_slot(int kvh, int kind)
{
    return SP_KVP + (uint32_t)(2*kvh + kind)*0x200;
}

static void kv_store_scale_prep(int kvh, int kind, int e8)
{
    int slot = 2*kvh + kind;
    int prev = kvp_scale_e8[slot];
    if (prev == e8)
        return;
    // Eight word stores instead of sixteen halfword stores: the beat is byte-identical
    // (halfword g = scale for g < groups, else 0) and uncached scratchpad stores are the
    // cost here, not the arithmetic. Only the words that carry a scale change with the
    // exponent - ceil(groups/2) = 2 of 8 on the 1.7B (head_dim 128 -> 4 groups). The others
    // are constant zero and nothing else ever writes the scale beat, so they are staged once
    // per slot epoch (a sentinel in kvp_scale_e8 at init) and left alone after.
    volatile uint32_t *pw = (volatile uint32_t *)(SCRATCHPAD_BASE + kvp_slot(kvh, kind));
    int groups = head_dim/32;
    int nw = (prev <= -1000) ? 8 : (groups + 1)/2;
    uint32_t sc = (uint32_t)(uint16_t)((e8 + 15) << 10);
    for (int w = 0; w < nw; w++) {
        uint32_t lo = (2*w     < groups) ? sc : 0;
        uint32_t hi = (2*w + 1 < groups) ? sc : 0;
        pw[w] = lo | (hi << 16);
    }
    kvp_scale_e8[slot] = e8;
}

// Pack prep only: the qkr sequencer list carries the quant8 itself (seq_build_qkr);
// the CPU side is the scale beat and the exponent ratchet - identical bookkeeping.
static void kv_pack_prep_res(int l, int kvh, int kind, int es)
{
    int e8 = es + 8;
    kv_store_scale_prep(kvh, kind, e8);
    int idx = kv_ek_idx(l, kvh, kind);
    if (e8 > kv_ekmax[idx]) kv_ekmax[idx] = e8;
}

#ifndef CSR_TRIXIUM_ENGINE_CSR_REP_ADDR
// Store phase: stream the already-packed slots out (one store per slot without the
// multi-store CSR). Engine only.
static void kv_store_row_res(int l, int kvh, int kind, int t)
{
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_STORE;
    op.chan  = t % n_ch;
    op.wbase = kv_region_addr(l, kvh, kind) + (uint32_t)(t/n_ch)*kv_rb*32;
    op.wlen  = kv_rb;
    op.xsrc  = kvp_slot(kvh, kind);
    engine_run(&op);
}
#endif

/* Resident Embedding */
/*--------------------*/

static int embed_row_bfp(const struct bons_entry *e, int row, uint32_t sp_dst);

// The embedding row already in int16 BFP form in the image (ROLE_TOK_EMB16, bit-exact with
// embed_row_bfp below - bons.py embed_bfp_rows): one engine fetch straight into the
// residual buffer plus one beat for the exponent, instead of the CPU decode and its
// uncached scratchpad stores. Falls back to the decode when the image has no such table.
// Both entries are resolved once at init (norm16_cache_build): the lookups are HBM-backed
// table walks, and this runs once per forward.
static const struct bons_entry *emb16_ent, *emb16x_ent;

static int embed_row_fetch(const struct bons_entry *e, int row, uint32_t sp_dst)
{
    const struct bons_entry *e16 = emb16_ent;
    const struct bons_entry *ee  = emb16x_ent;
    if (!e16 || !ee || e16->kind != KIND_NORM16 || ee->kind != KIND_NORM16
        || (uint32_t)e16->cols != (uint32_t)e->cols || row < 0 || (uint32_t)row >= e16->rows)
        return embed_row_bfp(e, row, sp_dst);
    uint32_t rb  = bons_row_beats(e16->cols, e16->kind, e16->group); // 128 at cols 2048.
    uint32_t rpc = e16->rows/n_ch;
    engine_fetch((uint32_t)row/rpc,
                 TRIXIUM_CHAN_WEIGHT_BASE + (e16->chan_base_beats + (uint64_t)((uint32_t)row % rpc)*rb)*32,
                 rb, sp_dst);
    // Exponents: 16 per beat, replicated - one beat from channel 0.
    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + (ee->chan_base_beats + (uint64_t)((uint32_t)row/16))*32,
                 1, SP_FET);
    return *(volatile int16_t *)(SCRATCHPAD_BASE + SP_FET + 2*((uint32_t)row % 16));
}

// Integer embedding: fetch the ternary row, dequantize with exact fp16 decomposition
// ((q-1)*m11 aligned to the smallest group exponent), then normalize to int16 BFP.
// Pure integer (~6 ops/element vs ~50 soft-float ones); exponent returned.
COLD static int embed_row_bfp(const struct bons_entry *e, int row, uint32_t sp_dst)
{
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch;
    engine_fetch(row/rpc,
                 TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)(row % rpc)*rb)*32,
                 rb, SP_FET);
    volatile uint8_t *pb = (volatile uint8_t *)(SCRATCHPAD_BASE + SP_FET);
    uint32_t sb = bons_scale_beats(e->cols, e->group);
    const volatile uint16_t *scales = (const volatile uint16_t *)pb;
    const volatile uint8_t  *codes  = pb + sb*32;
    int G = e->cols/128;
    int32_t m11[32];
    int     e16[32], e16min = 1000;
    for (int g = 0; g < G; g++) {
        uint16_t b2 = scales[g];
        int exp2f_ = (b2 >> 10) & 0x1F;
        int frac   = b2 & 0x3FF;
        int m      = (exp2f_ > 0) ? (1024 + frac) : frac;
        if (b2 & 0x8000) m = -m;
        m11[g] = m;
        e16[g] = (exp2f_ > 0) ? (exp2f_ - 25) : -24;
        if (e16[g] < e16min) e16min = e16[g];
    }
    // Two passes (no buffer: SRAM is tight): amax of |(q-1)*m11 << sh| per group is just
    // |m11[g]| << sh over any nonzero code - scan codes once for group nonzero-ness.
    // Scratchpad accesses are uncached bus transactions (~30 cycles each), so the codes
    // are read as 32-bit words (16 codes per read) and the int16 results are stored two
    // per word.
    const volatile uint32_t *cw = (const volatile uint32_t *)(pb + sb*32);
    int32_t amax = 0;
    for (int g = 0; g < G; g++) {
        int sh = e16[g] - e16min;
        if (sh > 20) sh = 20;
        int32_t gm = ((m11[g] < 0) ? -m11[g] : m11[g]) << sh;
        if (gm <= amax) continue;
        for (int j = 0; j < 8; j++) {  // Any nonzero code in the group? (0x55 = "1" x4.)
            if (cw[g*8 + j] != 0x55555555u) { amax = gm; break; }
        }
    }
    int bl = bitlen32((uint32_t)amax);   // Bit length (amax >= 0).
    int k2 = bl - 15;                    // Mantissas into [2^14, 2^15); may be negative.
    volatile uint32_t *pw = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_dst);
    for (int g = 0; g < G; g++) {
        int sh = e16[g] - e16min;
        if (sh > 20) sh = 20;
        int net = sh - k2;               // Combined shift, sign either way.
        int32_t m = m11[g];
        volatile uint32_t *dst = pw + g*64;
        for (int w = 0; w < 8; w++) {
            uint32_t c = cw[g*8 + w];    // 16 codes, element i at bits [2i+1:2i].
            for (int i = 0; i < 16; i += 2) {
                int32_t v0 = (int32_t)(((c >> (2*i)) & 3) - 1)*m;
                int32_t v1 = (int32_t)(((c >> (2*i + 2)) & 3) - 1)*m;
                int16_t o0 = (int16_t)((net >= 0) ? (v0 << net) : (v0 >> -net));
                int16_t o1 = (int16_t)((net >= 0) ? (v1 << net) : (v1 >> -net));
                dst[w*8 + i/2] = (uint32_t)(uint16_t)o0 | ((uint32_t)(uint16_t)o1 << 16);
            }
        }
    }
    return e16min + k2;
}

int infer_embed_f(int token, float *dst)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    if (!emb) return -1;
    int e = embed_row_fetch(emb, token, SP_XR);
    volatile int16_t *p = sp16(SP_XR);
    float s = ldexpf(1.0f, e);
    for (int i = 0; i < n_embd; i++) dst[i] = (float)p[i]*s;
    return 0;
}

/* Sequencer: QK-Norm/RoPE List */
/*------------------------------*/

// Sequencer descriptor list for the per-head qk-norm + rope loops. Addresses are
// head-relative inside SP_Q/SP_K, so the list is layer-independent; per layer only the
// rmsnorm eps_int args (A5 = descriptor word 7) change. Built once at init.
static int seq_qkr_count;
// Descriptor indices of the q-norm and k-norm in the qkr list (their A1/A5 are patched
// per layer).
static int seq_qkr_di_qn = 0, seq_qkr_di_kn = 1;

// qkr list order: K chain -> packs -> multi-store (NOWAIT) -> Q chain -> Q quant8. The q-side
// work (rmsnorm_h/rope_h over the q heads) is independent of K/V, so behind the store it
// absorbs the store's write-latency tail that the stop descriptor would otherwise wait out.
// Same ops, same numerics.
// qkvx (TRIXIUM_SEQ_LUT bitstreams): the K chain (rmsnorm_h, rope_h, pack) and the V pack
// move INTO the qkv list, under the q projection's stream, in v-k-q order; the qkr list
// keeps the multi-store and the Q chain. rmsnorm's eps - a per-layer constant scaled by the
// norm shift the sequencer holds in a kreg - comes from a 32-entry table
// (seq_desc_vec_lut) the CPU fills per layer under the previous FFN list. Same programs,
// same eps values: byte-identical. Requires the resident norm rows, the multi-store and
// the reordered Q chain.
static int qkvx_ok  = 0;      // What the built lists use.
static int qkvx_blocked = 0;  // The 14-descriptor qkv list did not fit: rebuilt without.
static int seq_qkv_len = 10;
static int seq_qkv_di_kn = -1;
#define SP_LUTQ (SP_DESC + 188*64) // 32 x int32 eps for the q norm (descriptor slots
#define SP_LUTK (SP_LUTQ + 128)    // 188-191, past every list).

// In-list KV stores: each (kvh, kind) staging slot's engine STORE rides the qkr list as a
// NOWAIT descriptor right after the pack quant8 that fills the slot - the next head's
// quant8 (vec) runs under the store's stream, so the serial CPU-dispatched stores hide under
// vector work the layer does anyway. Needs the scale-beat prep to run BEFORE the list (the
// store streams the scale beat), which is equally correct in the CPU-store layout.
static int kv_pipe_active = 0;  // What the built list uses.

// One in-list MULTI-STORE for all 2*n_kv_head slots instead of one NOWAIT store per slot:
// the staging slots are 0x200 B apart and a layer's (kvh, kind) regions kv_region beats
// apart, so the engine's rep/dstride/sstride store scatters them all from one descriptor
// (one dispatch on the list side, one per-layer address patch on the CPU side). Needs the
// multi-store CSR.
#ifdef CSR_TRIXIUM_ENGINE_CSR_REP_ADDR
static const int kv_mstore = 1;
#else
static const int kv_mstore = 0;
#endif
static int kv_mstore_active = 0; // What the built list uses.
static uint8_t seq_kvst_di[32]; // Descriptor index of the store for slot (kvh*2 + kind).
// What the store descriptors' constant words currently hold (~0 = unknown/rebuilt).
static uint32_t kvst_rb_set = 0xFFFFFFFFu, kvst_chan_set = 0xFFFFFFFFu;

static int qkvx_possible(void)
{
    // The K chain lives inside the qkv list, so qkvx only holds while that list runs:
    // with `seq off` the qkr list (built without the K chain) would run alone, K would
    // never be normed and the q-norm's A6 never patched.
    return !qkvx_blocked && seq_use
        && (uint32_t)n_layer*head_dim*2 <= 0x1C00
        && model_entry_kind(ROLE_ATTN_Q_NORM, 0, KIND_NORM16) != 0
        && model_entry_kind(ROLE_ATTN_K_NORM, 0, KIND_NORM16) != 0
        && kv_mstore && 2*n_kv_head <= 31;
}

static void seq_build_qkr(void)
{
    kvst_rb_set = kvst_chan_set = 0xFFFFFFFFu; // The store slots below are rewritten.
    qkvx_ok = qkvx_possible();
    int ln = 0;
    while ((1 << ln) < head_dim) ln++;
    uint32_t di = 0;
    // Whole-tensor per-head programs: the 2*(n_head+n_kv_head) rmsnorm/rope descriptors and
    // the 2*n_kv_head KV-pack ones are one program each - a per-head descriptor's fixed cost
    // is 8-12x its work. Same instruction sequence per head, so byte-identical by
    // construction; only A5 (eps) of the two norms is patched per layer.
    int32_t nq[8] = {(int32_t)SP_Q, (int32_t)SP_FET, (int32_t)SP_Q, head_dim/16,
                     -ln, 1, 0, n_head};                       // A5, A6 patched per layer
    int32_t rq[8] = {(int32_t)SP_Q, (int32_t)(SP_Q + head_dim), (int32_t)SP_TAB,
                     (int32_t)(SP_TAB + head_dim), (head_dim/2)/16, (int32_t)SP_Q,
                     head_dim, n_head};
    {
        int32_t nk[8] = {(int32_t)SP_K, (int32_t)SP_FET2, (int32_t)SP_K, head_dim/16,
                         -ln, 1, 0, n_kv_head};                // A5, A6 patched per layer
        int32_t rk[8] = {(int32_t)SP_K, (int32_t)(SP_K + head_dim), (int32_t)SP_TAB,
                         (int32_t)(SP_TAB + head_dim), (head_dim/2)/16, (int32_t)SP_K,
                         head_dim, n_kv_head};
        int32_t pk[8] = {(int32_t)SP_K, 0, (int32_t)(kvp_slot(0, 0) + 32), head_dim/16, 0, 0,
                         0x400 - 16*(head_dim/16), n_kv_head};
        int32_t pv[8] = {(int32_t)SP_V, 0, (int32_t)(kvp_slot(0, 1) + 32), head_dim/16, 0, 0,
                         0x400 - 16*(head_dim/16), n_kv_head};
        if (!qkvx_ok) { // qkvx: the K chain and both packs ride the qkv list instead.
            seq_qkr_di_kn = (int)di;
            seq_desc_vec(di++, PROG_RMSNORM_H, nk, 8, 0);              // k-norm (A5 patched)
            seq_desc_vec(di++, PROG_ROPE_H, rk, 8, 0);                 // k-rope
            seq_desc_vec(di++, PROG_KVPACK_H, pk, 8, 0);               // 4: K packs
            seq_desc_vec(di++, PROG_KVPACK_H, pv, 8, 0);               // 5: V packs
        }
    }
    // The fixed-shift KV-pack quant8s ride the same list: the whole-tensor PROG_KVPACK_H
    // descriptors 4/5 pack every kv head (K's after K's rope; V is resident before the list
    // runs). The CPU keeps only the scale-beat prep and the kv_ekmax ratchet
    // (kv_pack_prep_res). The ENGINE STORES below are needed without the multi-store.
    // kv_pipe_active must be reset unconditionally: a stale value would make a list
    // rebuild patch descriptors that now belong to the attention list.
    kv_pipe_active = 0;
    for (int kvh = 0; kvh < n_kv_head; kvh++)
        for (int kind = 0; kind < 2; kind++) {
            if (!kv_mstore && 2*n_kv_head <= 32) {
                // NOWAIT store of the slot just packed; wbase/wlen/chan patched per
                // layer/forward (kv_rb is set after the first build).
                struct engine_op st = {0};
                st.kind = ENGINE_KIND_STORE;
                st.xsrc = kvp_slot(kvh, kind);
                seq_desc_engine(di, &st);
                seq_patch(di, 0, 1u | (1u << 9)); // NOWAIT.
                seq_kvst_di[kvh*2 + kind] = (uint8_t)di;
                kv_pipe_active = 1;
                di++;
            }
        }
    kv_mstore_active = 0;
    if (kv_mstore && 2*n_kv_head <= 31) {
        // ONE NOWAIT multi-store for every slot (see kv_mstore): rep packets, kv_rb beats
        // each, from kvp_slot(0,0) stepping 0x200 B, to kv_region_addr(l,0,0) + rowoff
        // stepping kv_region beats. Word 4 (base) is patched per layer, 5/10 on change.
        struct engine_op st = {0};
        st.kind    = ENGINE_KIND_STORE;
        st.xsrc    = kvp_slot(0, 0);
        st.rep     = (uint8_t)(2*n_kv_head);
        st.dstride = kv_region;
        st.sstride = 0x200/32;
        seq_desc_engine(di, &st);
        seq_patch(di, 0, 1u | (1u << 9)); // NOWAIT: the whole-Q quant8 below runs under it.
        seq_kvst_di[0] = (uint8_t)di;
        kv_pipe_active = 1;
        kv_mstore_active = 1;
        di++;
    }
    // Q chain behind the (NOWAIT) store: absorbs its write-latency tail.
    seq_qkr_di_qn = (int)di;
    if (qkvx_ok) { // A5 = LUTQ[sx8 + kreg0], kreg0 captured by the qkv list's q matvec.
        int32_t nql[8] = {nq[0], nq[1], nq[2], nq[3], nq[4], 0, nq[6], nq[7]};
        seq_desc_vec_lut(di++, PROG_RMSNORM_H, nql, 8, 0, SP_LUTQ/32, 1);
    } else
        seq_desc_vec(di++, PROG_RMSNORM_H, nq, 8, 0);                  // q-norm (A5 patched)
    seq_desc_vec(di++, PROG_ROPE_H, rq, 8, 0);                         // q-rope
    // ONE fixed-shift quant8 over all of resident q -> SP_X + 0x1000 (STB post-increments
    // 16 B, so head j lands at +j*head_dim); the attention list's scores read x from there.
    // The region is dead during attention: softmax p8 lives at SP_X[0, ctx) and nothing else
    // touches SP_X until the FFN quant8 overwrites it.
    {
        // Fixed shift only: PROG_QUANT8 with A1 = 0 would still run its absmax pass over
        // all 128 beats before the fixed >>8 pass, and nothing reads the absmax here.
        // PROG_KVPACK_H with one "head" is the same LDV / VRNE 8 / VCLAMP8 / STB chain
        // without the scan - byte-identical, half the loads.
        int32_t fargs[8] = {(int32_t)SP_Q, 0, (int32_t)(SP_X + 0x1000),
                            (n_head*head_dim)/16, 0, 0, 0, 1};
        seq_desc_vec(di++, PROG_KVPACK_H, fargs, 8, 0);
    }
    seq_desc_stop(di++);
    seq_qkr_count = di;
}

// seq_eps_int cache, index es+64; 0 = not computed yet. int32 over [-64, 63]: the qkvx LUT
// fill asks for 32 es values per norm per layer (including the big-eps, very negative es
// entries) and a miss is soft-float work. In the main_ram static window (not SRAM .bss,
// which is at its 4 KiB stack floor), so NOT zero-initialized: infer_static_init() clears
// it.
#define eps_memo ((int32_t *)TRIXIUM_EPS_MEMO_ADDR)

void infer_static_init(void)
{
    for (int i = 0; i < 128; i++) eps_memo[i] = 0;
}

static int seq_eps_int(int es)
{
    // Soft-float lroundf/ldexpf costs hundreds of cycles and es takes a handful of values:
    // memoize. e >= 1 always, so 0 doubles as the empty marker.
    int mi = es + 64;
    if (mi >= 0 && mi < 128 && eps_memo[mi])
        return eps_memo[mi];
    int e = (int)lroundf(H->rms_eps*ldexpf(1.0f, -2*es));
    if (e < 1) e = 1;
    if (mi >= 0 && mi < 128)
        eps_memo[mi] = e;
    return e;
}

/* Sequencer: Attention List */
/*---------------------------*/

// Sequencer descriptor list for the fused per-head attention chain:
//   quant8 -> scores (engine) -> softmax8 -> spill(p8 sum) -> AXPY (rshift from rpre of
//   the chained res6 = p8 sum) -> reduce x(n_ch-1) -> spill(absmax) -> norm_i32 (k via
//   res6). One block per head slot, reused across chunks; head/token/layer-dependent
//   words are patched before each run. Spills feed the one CPU float fold per head
//   (c2/hmax) after the list - identical numerics to the CPU-dispatched fused path.
static int seq_attn_base, seq_attn_dph, seq_attn_heads;
// Pipelined attention list: head j+1's scores op is dispatched NOWAIT before head j's
// softmax8, so the engine streams K(j+1) while the vector unit works. Scores double-buffer
// between two buffers written only by scores ops, so the INT32_MIN padding written once per
// forward survives across layers, the same invariant the single buffer relies on (not SP_G:
// the FFN rewrites it every layer). The serialization softmax8(j) needs against scores(j)
// comes free from the dispatch of scores(j+1) - an engine descriptor only dispatches when
// the engine is idle - so no barrier bits are required. Layout when active:
//   base+0                 scores(slot 0), NOWAIT                    (prologue)
//   base+1+j*5 .. +4       [spill(j-1)|nopd] [scores(j+1) NOWAIT|nopd] [norm(j-1)|nopd]
//                          softmax8(j)  wv(j) NOWAIT
//   base+1+chunk*5 .. +1   spill(chunk-1)  norm(chunk-1)              (epilogue, static)
// Deep form: wv(j) is NOWAIT; the next group's spill(j) stalls on the
// absmax interlock until wv(j) completes (the tracker is safe - scores(j+2) sits AFTER
// the spill in list order), and norm(j-1)+softmax8(j) execute under scores(j+1)'s
// stream. The engine idles only for the ~20-cycle spill instead of spill+norm, and the
// vector unit fills the scores streams completely. Non-chunked only.
static int attn_pipe = 1;        // What the NEXT build will use. attn_list_prep_token
                                 // clears it when the context outgrows the second scores
                                 // buffer and restores it when it fits again.
static int attn_prep_pos = -1;   // Position attn_list_prep_token last ran for (dedup).
static int attn_pipe_active = 0; // What the built list uses.
static int attn_chunk_cap = 0;   // Debug: cap the heads per list (0 = as many as fit).
// The second scores buffer (odd heads, deep layout): its INT32_MIN padding is written once
// per token, so it must stay untouched for the whole forward, and its capacity bounds the
// context the deep layout serves (attn_list_prep_token). Its address follows the model:
// SP_SC + 0x3800 up to SP_K (0x800 bytes = 512 int32 entries) is the tail of the
// resident-q region, free on the 1.7B (q 4 KB) but inside SP_Q on the 8B (q 8 KB); the 8B
// gets the free half of SP_D instead (the resident k-norm rows that live there on the 1.7B
// are impossible at 36 layers, see qres), 8 KB = the full 2048-position budget. Not SP_YU:
// the FFN's up results overwrite it every layer. Neither fits: the deep layout stays off.
// Buffer A is safe: ceil32(cnt) <= 2048 = max_ctx ends it exactly at SP_Q.
static uint32_t sp_scb2;          // Second scores buffer (0 = no deep layout possible).
static uint32_t sc_b_cap_bytes;   // Its capacity.
#define SP_SCB2        sp_scb2
#define SC_B_CAP_BYTES sc_b_cap_bytes
static inline uint32_t attn_grp(int j)
{
    return (uint32_t)seq_attn_base + (attn_pipe_active ? 1u : 0u)
           + (uint32_t)j*(uint32_t)seq_attn_dph;
}

static inline uint32_t attn_di_sc(int j) // Scores descriptor serving slot j.
{
    if (!attn_pipe_active) return attn_grp(j);
    // Deep layout: slot 1 of the previous group (slot 0 is that group's spill; an
    // unwritten slot would be w0 = 0, a STOP).
    return (j == 0) ? (uint32_t)seq_attn_base : attn_grp(j - 1) + 1u;
}

static inline uint32_t attn_di_sm(int j) // softmax8 slot.
{
    return attn_grp(j) + (attn_pipe_active ? 3u : 1u);
}

static inline uint32_t attn_di_wv(int j) // wv (AXPY) slot.
{
    return attn_grp(j) + (attn_pipe_active ? 4u : 2u);
}

static inline uint32_t attn_sc_dst(int j)
{
    // Double-buffered ONLY in the pipelined layout: the INT32_MIN padding is only
    // maintained for the active layout, so the plain list must not read the second buffer.
    return (attn_pipe_active && (j & 1)) ? (uint32_t)SP_SCB2 : (uint32_t)SP_SC;
}

static inline int attn_list_len(int nh)
{
    return nh*seq_attn_dph + (attn_pipe_active ? 3 : 0); // Prologue + 2-slot epilogue.
}

static void seq_build_attn(void)
{
    seq_attn_heads = 0;
    attn_prep_pos = -1; // Rebuild wipes the per-token patch words; force a re-prep.
    if (n_ch < 2)
        return; // n1's absmax lives in the engine CSR; keep CPU-dispatched ops there.
    // 5 descriptors per head: scores, softmax8 (self-spills its p8 sum), axpy(xred),
    // spill(absmax from the engine tracker), norm_i32 (the q quant8 is batched in the qkr
    // list, the p8 spill is softmax8's tail and the engine reduces across channels).
    // 16*5 + 66 = 146 of 192, so the FFN and scale8 lists fit behind it.
    int dph = 5;
    int cap = (192 - seq_qkr_count - (attn_pipe ? 3 : 0))/dph;
    if (attn_chunk_cap > 0 && cap > attn_chunk_cap)
        cap = attn_chunk_cap;                    // `attnchunk N`: force a chunked list
    if (cap <= 0)
        return;
    int chunk = (cap < n_head) ? cap : n_head;
    int lg = 0;
    while ((1 << lg) < head_dim) lg++;
    int Ceff = (lg & 1) ? 523 : 739;
    seq_attn_base  = seq_qkr_count;
    seq_attn_dph   = dph;
    seq_attn_heads = chunk;
    attn_pipe_active = attn_pipe && (chunk == n_head); // Deep layout: non-chunked only.
    int32_t z = 0;
    for (int j = 0; j < chunk; j++) {
        uint32_t di = attn_grp(j);
        struct engine_op sc = {0};
        sc.kind   = ENGINE_KIND_Q8;
        sc.ilv    = 1;
        sc.cols   = head_dim;
        sc.sbeats = 1;
        sc.cbeats = head_dim/32;
        sc.xsrc   = SP_X + 0x1000 + (uint32_t)j*head_dim; // Batched q8 (qkr list).
        sc.dst    = attn_sc_dst(j);
        seq_desc_engine(attn_di_sc(j), &sc);
        if (attn_pipe_active) {
            seq_patch(attn_di_sc(j), 0, 1u | (1u << 9)); // NOWAIT.
            // No scores(chunk) exists: the placeholder goes in the LAST group's slot 2,
            // and its norm(chunk-2) moves up to slot 1 (see below). A NOPD is a vec
            // program and a program start ZEROES the scalar file, so a NOPD between a
            // spill and its KRES6-consuming norm would hand the norm k = 0.
            if (j == chunk - 1)
                seq_desc_vec(di + 2, PROG_NOPD, &z, 0, 0);
        }
        int32_t sargs[7] = {(int32_t)attn_sc_dst(j), Ceff, (int32_t)SP_X, 0, -11, 0,
                            (int32_t)(SP_AWP8 + (uint32_t)j*32)}; // A6: p8-sum self-spill.
        seq_desc_vec(attn_di_sm(j), PROG_SOFTMAX8, sargs, 7, 0);
        struct engine_op wv = {0};
        wv.kind        = ENGINE_KIND_AXPY;
        wv.cols        = head_dim;
        wv.sbeats      = 1;
        wv.cbeats      = head_dim/32;
        wv.xsrc        = SP_X;
        wv.dst         = SP_AW;
        wv.rshift_rpre = 1;
        wv.xred        = 1;
        seq_desc_engine(attn_di_wv(j), &wv);
        // wv stays BLOCKING even in the deep layout: the next descriptor is its own
        // spill either way, so NOWAIT buys nothing - and it bites: the absmax tracker
        // settles at done+2, the kreg capture path delays for exactly that (cap_d), but
        // a NOWAIT-released dispatch fires within a cycle of eng_inflight clearing, so
        // the spill's S7 seed would sample a not-yet-settled absmax. The WAIT_ENG path's
        // descriptor fetch gives the settle time for free.
        // The engine sums the channels itself (xred), so the absmax the CPU fold needs
        // comes from the engine tracker. seq_desc_vec_seed puts it in S7 for the spill, and
        // spill's trailing SMOV s6, s7 leaves it in res6 - so the norm below takes k from
        // KRES6. Numerics are identical by construction.
        // Deep layout: head j's spill/norm live in group j+1 (or the epilogue), so they
        // run while scores(j+1) streams; their content is head-static either way.
        int32_t p2 = (int32_t)(SP_AWAM + (uint32_t)j*32);
        int32_t nargs[5] = {(int32_t)SP_AW, 0, (int32_t)(SP_H + (uint32_t)j*head_dim*2),
                            head_dim/16, 0};
        if (!attn_pipe_active) {
            seq_desc_vec_seed(di + 3, PROG_SPILL, &p2, 1, 0, 1);
            seq_desc_vec(di + 4, PROG_NORM_I32, nargs, 5,
                         (uint16_t)(SEQ_SRC_KRES6 << 8)); // A4 = k(res6).
        } else {
            uint32_t sp_di = (j == chunk - 1)
                ? (uint32_t)seq_attn_base + 1u + (uint32_t)chunk*5u      // epilogue+0
                : attn_grp(j + 1) + 0u;
            uint32_t nm_di = (j == chunk - 1)
                ? (uint32_t)seq_attn_base + 1u + (uint32_t)chunk*5u + 1u // epilogue+1
                : ((j == chunk - 2) ? attn_grp(j + 1) + 1u   // last group: norm at slot 1
                                    : attn_grp(j + 1) + 2u); // (slot 2 holds the NOPD)
            seq_desc_vec_seed(sp_di, PROG_SPILL, &p2, 1, 0, 1);
            seq_desc_vec(nm_di, PROG_NORM_I32, nargs, 5,
                         (uint16_t)(SEQ_SRC_KRES6 << 8)); // A4 = k(res6).
            if (j == 0) { // Group 0 has no previous head: placeholders.
                seq_desc_vec(di + 0, PROG_NOPD, &z, 0, 0);
                seq_desc_vec(di + 2, PROG_NOPD, &z, 0, 0);
            }
        }
    }
}

/* FFN Profiling */
/*---------------*/

// FFN diagnosis (fprof console command): windowed overlap accumulators over the FFN block on
// whichever path ran, plus one layer's fine-grained trace per forward (layer 0): descriptor
// timestamps for the list path, step timestamps for the op path.
int fprof_use = 0;
static uint64_t *const fprof_acc_list = (uint64_t *)(TRIXIUM_FPROF_BUFFER_ADDR);
static uint64_t *const fprof_acc_ops = (uint64_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 32);
static uint32_t fprof_nlist, fprof_nops;
static uint32_t *const fprof_ts = (uint32_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 64);
static int fprof_nts;
static uint32_t *const fprof_steps = (uint32_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 128);
static int fprof_nsteps;

void fprof_reset(void)
{
    for (int i = 0; i < 4; i++) fprof_acc_list[i] = fprof_acc_ops[i] = 0;
    fprof_nlist = fprof_nops = 0; fprof_nts = 0; fprof_nsteps = 0;
}

void fprof_print(void)
{
    if (fprof_nlist) {
        perf_ovl_report("FFN-list", fprof_acc_list);
        printf("  windows %lu\n", (unsigned long)fprof_nlist);
    }
    if (fprof_nops) {
        perf_ovl_report("FFN-ops", fprof_acc_ops);
        printf("  windows %lu\n", (unsigned long)fprof_nops);
    }
    if (fprof_nts) {
        printf("FFNTRACE list dcounts (deltas):");
        for (int i = 1; i < fprof_nts; i++)
            printf(" %lu", (unsigned long)(fprof_ts[i] - fprof_ts[i-1]));
        printf("\n");
    }
    if (fprof_nsteps) {
        printf("FFNTRACE ops steps (deltas):");
        for (int i = 1; i < fprof_nsteps; i++)
            printf(" %lu", (unsigned long)(fprof_steps[i] - fprof_steps[i-1]));
        printf("\n");
    }
}

/* Sequencer: FFN List */
/*---------------------*/

// FFN chain as ONE descriptor list (kreg exponent registers, sequencer w0[5]/w1[16+2i]),
// layout in seq_build_ffn: the silu shift eg = ex8 - bias + rshift + kg mixes a per-layer
// constant with the gate matvec's k - inexpressible with KABS alone, hence the kreg.
// Post-list the CPU recovers: kg/ku from the two spilled absmaxes, s from vec res6, kd and
// dmax from the engine absmax CSR (down ran last) - the SAME values the op-by-op path
// computes, so the numerics are identical by construction. The final norm -> SP_D and
// resid_add stay CPU-side (they need *omax and the residual ratchet state).
static int seq_ffn_base = -1;
static int seq_ffn_len  = 0;
static int ffn_shape_base = -1; // seq_ffn_base the FFN engine shape images were written for.

// The SiLU split: everything in silu_q8 up to the x*sigmoid(x) product is a function of the
// GATE alone, so that half (silu_gate) runs UNDER the up matvec's stream (where norm_g
// already runs) and only the multiply/absmax/quant8 tail (mul_q8) waits for up.
// Byte-identical by construction (asm.py, test_silu_split_matches_fused). It relies on the
// matvec result packer: without it the vector unit runs ~2.9x slower under an engine stream
// and norm_g alone fills the window.
static int ffn_di_down = 7; // Down matvec's descriptor (seq_build_ffn).
static int ffn_di_a4   = 6; // Descriptor whose A4 takes the per-layer gate shift.

static void seq_build_ffn(void)
{
    int di = seq_qkr_count + attn_list_len(seq_attn_heads);
    int len = 10;
    if (di + len > 192 || seq_attn_heads == 0)
        return;
    seq_ffn_base = di;
    seq_ffn_len  = len;
    ffn_shape_base = -1; // Engine slots are placeholders until ffn_prep_layer.
    // Pipelined order: the up matvec is NOWAIT and the gate norm and silu_gate run UNDER
    // its stream. The norms take k via KRES6 (the preceding spill's absmax) instead of
    // KABS, so they do not read the live absmax and can float across engine ops. kreg
    // consumers are all >= one interlocked spill program past the capturing op's done,
    // which covers the capture-at-done+2 lag.
    //   0 gate mv [cap k->kreg0]    1 spill gate-absmax (seed_abs, engine idle here)
    //   2 up mv NOWAIT
    //   3 norm_g SP_Y -> SP_G [k via KRES6]            <- runs under the up stream
    //   4 silu_gate SP_G in place [A4 = imm + kreg0]   <- runs under the up stream
    //   5 spill up-absmax (seed_abs: interlock waits the stream)
    //   6 norm_u SP_YU -> SP_U [k via KRES6]
    //   7 mul_q8 -> SP_X [res6 = source absmax, read by the CPU after the list -
    //     desc 8 is an engine op and leaves it alone]
    //   8 down mv   9 stop
    struct engine_op z = {0};
    seq_desc_engine(di + 0, &z);
    int32_t pa = (int32_t)SP_FFSP;
    seq_desc_vec_seed(di + 1, PROG_SPILL, &pa, 1, 0, 1);
    seq_desc_engine(di + 2, &z);
    // norm k via KRES6, NOT a kreg-add: desc1's spill left the gate absmax in res6
    // (SMOV s6,s7), and KRES6 is not in the hazard interlock - so this descriptor
    // dispatches UNDER the NOWAIT up stream. A kadd arg would stall on the interlock and
    // serialize the list.
    int32_t na[5] = {(int32_t)SP_Y, 0, (int32_t)SP_G, ffn_dim/16, 0};
    seq_desc_vec(di + 3, PROG_NORM_I32, na, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
    int d = di + 4; // Next free descriptor.
    {
        // 4: silu_gate SP_G -> SP_G in place [A4 = imm + kreg0], still UNDER the up
        //    stream. kreg0 was captured by desc 0 (long done) and the per-register
        //    interlock lets a kreg0 consumer dispatch under kreg1's NOWAIT stream
        //    (test_sequencer_kadd_interlock_precise).
        int32_t ga[5] = {(int32_t)SP_G, 0, (int32_t)SP_G, ffn_dim/16, 0};
        seq_desc_vec(d, PROG_SILU_GATE, ga, 5, 0);
        seq_patch(d, 1, (uint32_t)PROG_SILU_GATE | (1u << (16 + 2*4))); // A4 kreg0 term...
        seq_patch(d, 0, 2u | (1u << 11)); // ...ADDS (w0[11]): A4 = imm + kg (VSHLS conv).
        ffn_di_a4 = d - di;
        d++;
    }
    int32_t pb = (int32_t)(SP_FFSP + 32);
    seq_desc_vec_seed(d, PROG_SPILL, &pb, 1, 0, 1);                      // spill up-absmax
    d++;
    int32_t nb[5] = {(int32_t)SP_YU, 0, (int32_t)SP_U, ffn_dim/16, 0};
    seq_desc_vec(d, PROG_NORM_I32, nb, 5, (uint16_t)(SEQ_SRC_KRES6 << 8)); // norm_u
    d++;
    {
        // mul_q8: (silu * up) << A6, clamp, absmax (res6 for the CPU), quant8 -> SP_X.
        int32_t ma[8] = {(int32_t)SP_G, (int32_t)SP_U, (int32_t)SP_G, ffn_dim/16, 0, 0, -15,
                         (int32_t)SP_X};
        seq_desc_vec(d, PROG_MUL_Q8, ma, 8, 0);
    }
    d++;
    seq_desc_engine(d, &z);                                               // down mv
    ffn_di_down = d - di;
    d++;
    seq_desc_stop(d);
}

/* Sequencer: Scale8 List */
/*------------------------*/

// The per-head attention-output scale8 as one n_head-descriptor list. Only A4 (the per-head
// Q15 multiplier Mh) changes per layer; an in-list descriptor dispatches ~10x faster than a
// CPU vec_run. Placed behind the FFN list (which is behind the attention list).
static int seq_scale8_base = -1;

static void seq_build_scale8(void)
{
    // Invalidate first, like seq_build_qkv does with its own base: seq_build_qkv derives
    // seq_qkv_base from it, and a stale (smaller) value after an early return could place the
    // qkv list on top of the FFN list's descriptors - which qkv_prep_layer(l+1) patches while
    // the FFN list RUNS. Every consumer guards on >= 0, so -1 degrades cleanly to the CPU
    // path instead of corrupting a list.
    seq_scale8_base = -1;
    if (seq_attn_heads == 0)
        return;
    int di = (seq_ffn_base >= 0) ? seq_ffn_base + seq_ffn_len
                                 : seq_qkr_count + attn_list_len(seq_attn_heads);
    if (di + n_head + 1 > 192)
        return;
    seq_scale8_base = di;
    for (int hh = 0; hh < n_head; hh++) {
        int32_t sa[5] = {(int32_t)(SP_H + hh*head_dim*2), 0, (int32_t)(SP_X + hh*head_dim),
                         head_dim/16, 0};                   // A4 = Mh, patched per layer.
        seq_desc_vec(di + hh, PROG_SCALE8, sa, 5, 0);
    }
    seq_desc_stop(di + n_head);
}

/* Sequencer: QKV List */
/*---------------------*/

// QKV projection list ("qkvl"): the three attention projections and their norm_i32s as one
// sequencer run, the FFN list's shape - each norm runs UNDER the next projection's NOWAIT
// stream with its shift derived in-sequencer (KRES6) from the absmax the preceding spill
// left in res6. Replaces six CPU dispatches per layer. The three absmaxes are spilled to
// SP_QKSP so the CPU recovers kq/kk/kv exactly as the op path computes them.
//   0 q mv            1 spill q-absmax -> SP_QKSP+0
//   2 k mv NOWAIT     3 norm_q SP_Y -> SP_Q [k via KRES6]      <- under the k stream
//   4 spill k-absmax  5 v mv NOWAIT   6 norm_k SP_Y+0x4000 -> SP_K [KRES6] <- under v
//   7 spill v-absmax  8 norm_v SP_Y+0x5000 -> SP_V [KRES6]    9 stop
// Engine words (wbase/bias per layer) come from qkv_prep_layer, posted under the
// previous layer's FFN list (or the LM-head stream for layer 0).
#define SP_QKSP (SP_FFSP + 64)
#define NRM_MAXL 64

static int seq_qkv_base = -1;
static int qkv_prepped  = -1; // Layer whose projections are currently in the slots.

static void seq_build_qkv(void)
{
    seq_qkv_base = -1;
    qkv_prepped  = -1;
    if (seq_scale8_base < 0)
        return;
    int di = seq_scale8_base + n_head + 1;
    int len = qkvx_ok ? 14 : 10;
    if (di + len > 192) {
        if (qkvx_ok) { // Fit without qkvx: every list rebuilds (bases shift with qkr).
            qkvx_blocked = 1;
            seq_ffn_base = -1;
            seq_build_qkr(); seq_build_attn(); seq_build_ffn(); seq_build_scale8();
            seq_build_qkv();
        }
        return;
    }
    seq_qkv_base = di;
    seq_qkv_len  = len;
    struct engine_op z = {0};
    int32_t pq = (int32_t)(SP_QKSP + 0), pk = (int32_t)(SP_QKSP + 32), pv = (int32_t)(SP_QKSP + 64);
    int q_dim = n_head*head_dim, kv_dim = n_kv_head*head_dim;
    int32_t nq[5] = {(int32_t)SP_Y,            0, (int32_t)SP_Q, q_dim/16,  0};
    int32_t nk[5] = {(int32_t)(SP_Y + 0x4000), 0, (int32_t)SP_K, kv_dim/16, 0};
    int32_t nv[5] = {(int32_t)(SP_Y + 0x5000), 0, (int32_t)SP_V, kv_dim/16, 0};
    if (qkvx_ok) {
        // v-k-q order: V's pack rides under the k stream, the whole K chain under the q
        // stream (its eps from LUTK[sx8 + kreg1], kreg1 captured by the k matvec); q's
        // own chain stays in the qkr list behind the store, its kreg0 captured here.
        //   0 v mv            1 spill v   2 k mv NOWAIT [cap kreg1]   3 norm_v   4 V packs
        //   5 spill k         6 q mv NOWAIT [cap kreg0]   7 norm_k   8 k-norm [LUT]
        //   9 k-rope         10 K packs  11 spill q   12 norm_q   13 stop
        int ln = 0;
        while ((1 << ln) < head_dim) ln++;
        int32_t nkl[8] = {(int32_t)SP_K, (int32_t)SP_KNW, (int32_t)SP_K, head_dim/16,
                          -ln, 0, 0, n_kv_head}; // A1, A6 per layer (qkv_prep_layer), A5 per token.
        int32_t rk[8]  = {(int32_t)SP_K, (int32_t)(SP_K + head_dim), (int32_t)SP_TAB,
                          (int32_t)(SP_TAB + head_dim), (head_dim/2)/16, (int32_t)SP_K,
                          head_dim, n_kv_head};
        int32_t pkk[8] = {(int32_t)SP_K, 0, (int32_t)(kvp_slot(0, 0) + 32), head_dim/16, 0, 0,
                          0x400 - 16*(head_dim/16), n_kv_head};
        int32_t pvv[8] = {(int32_t)SP_V, 0, (int32_t)(kvp_slot(0, 1) + 32), head_dim/16, 0, 0,
                          0x400 - 16*(head_dim/16), n_kv_head};
        seq_desc_engine(di + 0, &z);
        seq_desc_vec_seed(di + 1, PROG_SPILL, &pv, 1, 0, 1);
        seq_desc_engine(di + 2, &z);
        seq_desc_vec(di + 3, PROG_NORM_I32, nv, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
        seq_desc_vec(di + 4, PROG_KVPACK_H, pvv, 8, 0);
        seq_desc_vec_seed(di + 5, PROG_SPILL, &pk, 1, 0, 1);
        seq_desc_engine(di + 6, &z);
        seq_desc_vec(di + 7, PROG_NORM_I32, nk, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
        seq_qkv_di_kn = di + 8;
        seq_desc_vec_lut(di + 8, PROG_RMSNORM_H, nkl, 8, 0, SP_LUTK/32, 2);
        seq_desc_vec(di + 9, PROG_ROPE_H, rk, 8, 0);
        seq_desc_vec(di + 10, PROG_KVPACK_H, pkk, 8, 0);
        seq_desc_vec_seed(di + 11, PROG_SPILL, &pq, 1, 0, 1);
        seq_desc_vec(di + 12, PROG_NORM_I32, nq, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
        seq_desc_stop(di + 13);
        // No print here: the lists rebuild during a generation (the deep attention layout
        // toggles with the context), and anything printed between "GEN:" and "GEN DONE"
        // is parsed as token ids by the chat tools. The state is reported once by
        // infer_init instead.
        return;
    }
    seq_desc_engine(di + 0, &z);
    seq_desc_vec_seed(di + 1, PROG_SPILL, &pq, 1, 0, 1);
    seq_desc_engine(di + 2, &z);
    seq_desc_vec(di + 3, PROG_NORM_I32, nq, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
    seq_desc_vec_seed(di + 4, PROG_SPILL, &pk, 1, 0, 1);
    seq_desc_engine(di + 5, &z);
    seq_desc_vec(di + 6, PROG_NORM_I32, nk, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
    seq_desc_vec_seed(di + 7, PROG_SPILL, &pv, 1, 0, 1);
    seq_desc_vec(di + 8, PROG_NORM_I32, nv, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
    seq_desc_stop(di + 9);
}

static void qkv_prep_layer(int l)
{
    if (seq_qkv_base < 0 || qkv_prepped == l)
        return;
    struct engine_op mo = {0};
    uint32_t b = (uint32_t)seq_qkv_base;
    if (qkvx_ok) {
        const struct bons_entry *eq = model_entry(ROLE_ATTN_Q, l);
        const struct bons_entry *ek = model_entry(ROLE_ATTN_K, l);
        model_matvec_op(model_entry(ROLE_ATTN_V, l), SP_X, SP_Y + 0x5000, &mo);
        seq_desc_engine(b + 0, &mo);
        model_matvec_op(ek, SP_X, SP_Y + 0x4000, &mo);
        seq_desc_engine(b + 2, &mo);
        seq_patch(b + 2, 0, 1u | (1u << 9) | (1u << 5) | (1u << 6)); // NOWAIT, k -> kreg1.
        model_matvec_op(eq, SP_X, SP_Y, &mo);
        seq_desc_engine(b + 6, &mo);
        seq_patch(b + 6, 0, 1u | (1u << 9) | (1u << 5));             // NOWAIT, k -> kreg0.
        // The layer's eps tables: LUT[s] = eps(E_NORM + s - bias + rshift), s = sx8 + k
        // with sx8 the per-token part (patched into A5's immediate before the list) and k
        // the norm shift the sequencer adds from the kreg. 32 entries cover sx8 <= 8 and
        // k <= 18 (int32 sources); seq_eps_int is the same function the qkr list's A5
        // patch uses, so the values are identical by construction.
        volatile uint32_t *lq = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_LUTQ);
        volatile uint32_t *lk = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_LUTK);
        int eq0 = E_NORM - (int)eq->bias + (int)eq->rshift;
        int ek0 = E_NORM - (int)ek->bias + (int)ek->rshift;
        for (int sidx = 0; sidx < 32; sidx++) {
            lq[sidx] = (uint32_t)seq_eps_int(eq0 + sidx);
            lk[sidx] = (uint32_t)seq_eps_int(ek0 + sidx);
        }
        // Resident norm rows (qres is a qkvx precondition): A1 of both rmsnorm_h, and
        // A6 = the layer's per-tensor gamma exponent shift.
        seq_patch((uint32_t)seq_qkv_di_kn, 3, SP_KNW + (uint32_t)l*head_dim*2);
        seq_patch((uint32_t)seq_qkr_di_qn, 3, SP_QNW + (uint32_t)l*head_dim*2);
        seq_patch((uint32_t)seq_qkv_di_kn, 8,
                  (uint32_t)w16_shift(model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16)));
        seq_patch((uint32_t)seq_qkr_di_qn, 8,
                  (uint32_t)w16_shift(model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16)));
        qkv_prepped = l;
        return;
    }
    model_matvec_op(model_entry(ROLE_ATTN_Q, l), SP_X, SP_Y, &mo);
    seq_desc_engine(b + 0, &mo);
    model_matvec_op(model_entry(ROLE_ATTN_K, l), SP_X, SP_Y + 0x4000, &mo);
    seq_desc_engine(b + 2, &mo);
    seq_patch(b + 2, 0, 1u | (1u << 9));               // NOWAIT: norm_q under the k stream.
    model_matvec_op(model_entry(ROLE_ATTN_V, l), SP_X, SP_Y + 0x5000, &mo);
    seq_desc_engine(b + 5, &mo);
    seq_patch(b + 5, 0, 1u | (1u << 9));               // NOWAIT: norm_k under the v stream.
    qkv_prepped = l;
}

/* Sequencer Control */
/*-------------------*/

// The descriptor lists (attention, FFN, qkv, scale8) are on by default: they collapse the
// per-head and per-op dispatch round-trips into one submit per list, same tokens either way.
// `seq off` selects the CPU-dispatched op-by-op path, which still overlaps the engine and
// the vector unit (docs/overlap-design.md): e.g. the FFN up-projection stream runs UNDER the
// gate-norm vector program (disjoint regions; gate absmax captured before the submit, so
// numerics are byte-identical by construction).
int seq_use = 1;

void infer_seq(int on)
{
    seq_use = on ? 1 : 0;
    qkvx_blocked = 0;
    seq_ffn_base = -1;
    seq_build_qkr(); seq_build_attn(); seq_build_ffn(); seq_build_scale8(); seq_build_qkv();
}

void infer_qkvx_report(void)
{
    if (qkvx_ok)
        puts("qkvx: K chain + V pack in the qkv list (14 descriptors), eps via LUT");
}

// Console `attnchunk N` (debug): cap the heads per attention list, rebuild the lists.
void infer_attn_chunk(int cap)
{
    attn_chunk_cap = cap;
    seq_ffn_base = -1;
    seq_build_qkr(); seq_build_attn(); seq_build_ffn(); seq_build_scale8(); seq_build_qkv();
    printf("ATTNCHUNK cap %d: list holds %d of %d heads, pipelined %d\n", cap, seq_attn_heads,
           n_head, attn_pipe_active);
}

/* Per-Layer List Patching */
/*-------------------------*/

// Per-layer attention slot patches, extracted so they can run UNDER the qkr list
// (per-cycle sideband slot): the sequencer is executing qkr descriptors and never
// dispatches attention ones, so patching them mid-list is safe; every input
// (kv_ekmax ratchet, region addresses) is CPU-side state settled before the list
// started. attn_prepatched tells the attention section its work is already done.
static int attn_prepatched;
static int ffn_prepatched;
static int qkn_resident;   // 1 once every layer's q/k norm rows sit in SP_QNW/SP_KNW.

// List-relative dcount ("descriptors completed") threshold after which head j's fold
// inputs (softmax8's AWP8 self-spill and spill(j)'s AWAM) are certainly written: one
// past norm(j), whose producers are all earlier vec descriptors in the same list.
static inline uint32_t attn_fold_ready(int j, int chunk)
{
    if (!attn_pipe_active) return (uint32_t)(j*5 + 5);
    if (j >= chunk - 2)    return (uint32_t)(1 + (j + 1)*5 + 2); // last-group/epilogue slots.
    return (uint32_t)(1 + (j + 1)*5 + 3);
}

// Per-layer FFN list rebuild (three engine descriptors + their flag words), extracted so it
// can POST under the qkr list (all inputs are layer-known: model entries and static
// scratchpad addresses; only the silu A4 imm needs ex8 and stays at FFN time). The
// sequencer-fetch interlock orders these writes before the FFN list runs.
static inline void ffn_patch_lw(uint32_t di, const struct engine_op *mo)
{
    seq_patch(di, 4, mo->wbase);
    seq_patch(di, 7, mo->bias | ((uint32_t)mo->rshift << 16) | ((uint32_t)mo->rshift2 << 22));
}

static void ffn_prep_layer(int l)
{
    // Only wbase (4) and bias|rshift (7) change with the layer; the 13 shape words are
    // per-role constants and the flag patches are build-time. Full images once per list
    // build, then 6 posted words per layer instead of 45.
    struct engine_op mo = {0};
    int full = (ffn_shape_base != seq_ffn_base);
    uint32_t b = (uint32_t)seq_ffn_base;
    model_matvec_op(model_entry(ROLE_FFN_GATE, l), SP_X, SP_Y, &mo);
    if (full) { seq_desc_engine(b + 0, &mo); seq_patch(b + 0, 0, 1u | (1u << 5)); }
    else ffn_patch_lw(b + 0, &mo);
    model_matvec_op(model_entry(ROLE_FFN_UP, l), SP_X, SP_YU, &mo);
    if (full) { seq_desc_engine(b + 2, &mo); seq_patch(b + 2, 0, 1u | (1u << 9)); }
    else ffn_patch_lw(b + 2, &mo);
    model_matvec_op(model_entry(ROLE_FFN_DOWN, l), SP_X, SP_Y, &mo);
    if (full) seq_desc_engine(b + (uint32_t)ffn_di_down, &mo);
    else ffn_patch_lw(b + (uint32_t)ffn_di_down, &mo);
    ffn_shape_base = seq_ffn_base;
}

static void attn_patch_slots(int l, int h0, int nh, int chunked)
{
    int lg = 0;
    while ((1 << lg) < head_dim) lg++;
    int nrep = n_head/n_kv_head;
    for (int j = 0; j < nh; j++) {
        int hh = h0 + j, kvh = hh/nrep;
        uint32_t di = attn_grp(j);
        uint32_t ds = attn_di_sc(j);
        int Bk, Rk, Bv, Rd;
        kv_scores_br(l, kvh, 0, &Bk, &Rk);
        kv_scores_br(l, kvh, 1, &Bv, &Rd);
        int sigma = (E_NORM + 8) - Bk + Rk - (lg >> 1);
        if (chunked) { // Chunked implies the plain layout (grp offsets 1/3/4).
            seq_patch(ds, 8, SP_X + 0x1000 + (uint32_t)hh*head_dim);
            seq_patch(di + 1, 8, SP_AWP8 + (uint32_t)hh*32);
            seq_patch(di + 3, 2, SP_AWAM + (uint32_t)hh*32);
            seq_patch(di + 4, 4, SP_H + (uint32_t)hh*head_dim*2);
        }
        seq_patch(ds, 4, kv_region_addr(l, kvh, 0));
        seq_patch(ds, 7, (uint32_t)Bk | ((uint32_t)Rk << 16));
        seq_patch(attn_di_sm(j), 7, (uint32_t)(sigma + 6 + 11)); // A5 (s2 shift).
        seq_patch(attn_di_wv(j), 4, kv_region_addr(l, kvh, 1));
        seq_patch(attn_di_wv(j), 7, (uint32_t)Bv);
        (void)Rd;
    }
}

// Token-invariant words of the attention list, patched ONCE per forward instead of once per
// layer per head: rowsw/wlen (scores + AXPY) and the softmax block count depend on pos only,
// and the score-slot padding survives across heads and layers because only the scores op
// writes SP_SC (exactly cnt slots). Uncached scratchpad stores are expensive.
static void attn_list_prep_token(int pos)
{
    if (seq_attn_heads <= 0)
        return;
    if (pos == attn_prep_pos)
        return; // Already posted under the previous forward's LM-head stream.
    int cnt  = pos + 1;
    int padn = SC_PADN(cnt);
    // The deep layout needs the padded score count to fit the second buffer's REAL
    // capacity (SC_B_CAP_BYTES, see above). Beyond it fall back to the plain list, and
    // come back to the deep one as soon as a new generation fits again - both directions
    // shift the FFN/scale8/qkv bases with the prologue, so all four lists rebuild.
    int want_pipe = SP_SCB2 != 0 && ((uint32_t)(cnt + padn)*4u <= SC_B_CAP_BYTES);
    if (want_pipe != attn_pipe) {
        attn_pipe = want_pipe;
        seq_ffn_base = -1;
        seq_build_attn(); seq_build_ffn(); seq_build_scale8(); seq_build_qkv();
    }
    volatile int32_t *sc32 = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
    for (int i2 = 0; i2 < padn; i2++)
        sc32[cnt + i2] = (int32_t)0x80000000;
    if (attn_pipe_active) {
        volatile int32_t *sc32b = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SCB2);
        for (int i2 = 0; i2 < padn; i2++)
            sc32b[cnt + i2] = (int32_t)0x80000000;
    }
    uint32_t rows  = (uint32_t)cnt/n_ch, rem = (uint32_t)cnt % n_ch;
    uint32_t wlen  = rows*kv_rb;
    uint32_t rowsw = rows | (rem << 16);
    uint32_t blocks = (uint32_t)(cnt + padn)/16;
    for (int j = 0; j < seq_attn_heads; j++) {
        seq_patch(attn_di_sc(j), 2, rowsw);    // scores rows|rem.
        seq_patch(attn_di_sc(j), 5, wlen);
        seq_patch(attn_di_sm(j), 5, blocks);   // softmax8 A3 (blocks).
        seq_patch(attn_di_wv(j), 2, rowsw);    // AXPY rows|rem.
        seq_patch(attn_di_wv(j), 5, wlen);
    }
    attn_prep_pos = pos;
}

/* Attention Fold Inputs */
/*-----------------------*/

static int fold_nrep_sh = 1; // log2(n_head/n_kv_head) when a power of two (set at init).

// The per-head fold from its two hardware results (softmax8's p8 sum, the AXPY absmax),
// in the integer form the output projection consumes. ONE producer for both the list
// path (spilled values, attn_fold_head) and the seq-off op-by-op path, so both fill the
// integer fold the consumer reads.
static void attn_fold_vals(int l, int hh, uint32_t P8, uint32_t am)
{
    int kvh = hh >> fold_nrep_sh;
    int Bv = 22 - kv_ekmax[kv_ek_idx(l, kvh, 1)];   // kv_scores_br's B; R unused here.
    int Rpre = axpy_rpre_c(P8 ? P8 : 1);
    int k2 = (bitlen32(am) > 14) ? bitlen32(am) - 14 : 0; // == the >16383 scan, via clz.
    int E1 = Rpre - Bv + k2;
    // Integer emulation of the float32 fold: same bits, no soft-float.
    uint32_t qm; int qe;
    attn_f32_recip(P8 ? P8 : 1, &qm, &qe);
    aw_qm[hh] = qm;
    aw_qe[hh] = qe + E1;
    uint32_t a = am >> k2;
    if (a) {
        uint32_t m2; int e2;
        attn_f32_mul(a, qm, qe + E1, &m2, &e2);
        if (!aw_hvalid || e2 > aw_he2 || (e2 == aw_he2 && m2 > aw_hm2)) {
            aw_hm2 = m2; aw_he2 = e2; aw_hvalid = 1;
        }
    }
}

static void attn_fold_head(int l, int hh)
{
    uint32_t P8 = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWP8 + (uint32_t)hh*32);
    uint32_t am = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWAM + (uint32_t)hh*32);
    attn_fold_vals(l, hh, P8, am);
}

/* Norm16 Cache */
/*--------------*/

// Per-layer norm16 entries, resolved once at init into .bss - which lives in the
// tightly-coupled SRAM. model_entry_kind() walks the entry cache and the tensor table, both
// in main_ram behind an L1 with no L2, so each of these lookups would be a couple of cache
// misses, several times a layer.
static const struct bons_entry *nrm_attn16[NRM_MAXL], *nrm_ffn16[NRM_MAXL], *nrm_out16;

static void norm16_cache_build(void)
{
    for (int l = 0; l < n_layer && l < NRM_MAXL; l++) {
        nrm_attn16[l] = model_entry_kind(ROLE_ATTN_NORM, l, KIND_NORM16);
        nrm_ffn16[l]  = model_entry_kind(ROLE_FFN_NORM,  l, KIND_NORM16);
    }
    nrm_out16 = model_entry_kind(ROLE_OUT_NORM, 0, KIND_NORM16);
    emb16_ent  = model_entry(ROLE_TOK_EMB16, 0);
    emb16x_ent = model_entry(ROLE_TOK_EMB16_EXP, 0);
}

// The norm16 row the layer after l opens with (attention norm), or the final norm.
static const struct bons_entry *next_norm16(int l)
{
    return (l + 1 < n_layer) ? nrm_attn16[l + 1] : nrm_out16;
}

uint32_t infer_forwards; // Forwards run since boot (bench reports cycles per forward too).

/* Sampling */
/*----------*/

// `sample <temp> <top_p> <rep> [seed]`; temp 0 = greedy, the default.
//
// The LM head's argmax tap returns one (value, index); sampling needs a candidate set. The
// head runs as SMP_SLICES row-range ops instead of one, and each tap returns the best of
// its slice: 32 candidates that always include the global best (so temperature 0 stays
// bit-exact greedy) and approximate the top-32 (the true second best sits in the best's
// own slice with probability 1/32). One op's dispatch is ~0.5k cycles, so the cost is
// ~16k cycles a token (<1%); the stream itself is the same 151k rows. No RTL, no logits
// in the scratchpad. The repetition penalty (llama.cpp's form: logit/rep above zero,
// logit*rep below) applies to candidates seen in the last smp_window tokens of the
// buffer the caller registers (infer_set_recent); then temperature, top-p, one draw from
// an xorshift32 stream seeded by `sample`'s seed (reproducible).
#define SMP_SLICES 32
static float    smp_temp = 0.0f, smp_top_p = 1.0f, smp_rep = 1.0f;
static uint32_t smp_rng = 0x9E3779B9u;
static int      smp_window = 64;
static const int *smp_recent; static const int *smp_recent_np;

void infer_set_sampling(float temp, float top_p, float rep, uint32_t seed)
{
    smp_temp = temp < 0 ? 0 : temp;
    smp_top_p = (top_p <= 0 || top_p > 1) ? 1.0f : top_p;
    smp_rep = rep < 1.0f ? 1.0f : rep;
    if (seed) smp_rng = seed;
}

void infer_get_sampling(float *temp, float *top_p, float *rep)
{
    *temp = smp_temp; *top_p = smp_top_p; *rep = smp_rep;
}

void infer_set_recent(const int *buf, const int *n_ptr)
{
    smp_recent    = buf;
    smp_recent_np = n_ptr;
}

static int smp_on(void)
{
    return smp_temp > 0.0f || smp_rep > 1.0f;
}

static uint32_t smp_rand(void)
{
    uint32_t x = smp_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return smp_rng = x;
}

// Pick a token among the slice maxima. val[i] are the engine's int32 logits (one scale
// for every row: the same B/R), lscale their float scale; tok[i] < vocab.
static int smp_pick(const int32_t *val, const int *tok, int n, float lscale)
{
    float lg[SMP_SLICES];
    int   best = 0;
    for (int i = 0; i < n; i++) {
        float v = (float)val[i]*lscale;
        if (smp_rep > 1.0f) {
            int seen = 0, rn = smp_recent_np ? *smp_recent_np : 0;
            for (int j = rn - smp_window; j < rn; j++)
                if (j >= 0 && smp_recent[j] == tok[i]) { seen = 1; break; }
            if (seen) v = (v > 0) ? v/smp_rep : v*smp_rep;
        }
        lg[i] = v;
        if (v > lg[best]) best = i;
    }
    if (smp_temp <= 0.0f)
        return tok[best];                        // greedy (with the penalty applied)
    // softmax(lg / temp) over the candidates, sorted descending for top-p.
    int   ord[SMP_SLICES];
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = 1; i < n; i++) {                // insertion sort, n <= 32
        int o = ord[i], j = i;
        while (j > 0 && lg[ord[j - 1]] < lg[o]) { ord[j] = ord[j - 1]; j--; }
        ord[j] = o;
    }
    float p[SMP_SLICES], sum = 0;
    for (int i = 0; i < n; i++) {
        p[i] = expf((lg[ord[i]] - lg[ord[0]])/smp_temp);
        sum += p[i];
    }
    float acc = 0, cut = smp_top_p*sum;
    int keep = n;
    for (int i = 0; i < n; i++) {                // smallest prefix with mass >= top_p
        acc += p[i];
        if (acc >= cut) { keep = i + 1; break; }
    }
    float r = (float)(smp_rand() >> 8)*(1.0f/16777216.0f)*acc;
    float c = 0;
    for (int i = 0; i < keep; i++) {
        c += p[i];
        if (r < c) return tok[ord[i]];
    }
    return tok[ord[keep - 1]];
}

/* Debug Traces */
/*--------------*/

// `dbg N`: checksum the resident intermediates of the next N forwards, the same regions at
// the same points as the Python mirror's raw trace (test_sim_smoke.hybrid_generate, tag
// names shared) - tools/card_vs_mirror.py lines the two up and names the first divergence.
// The only bit-exact reference the 1.7B has: the tiny sim model answers token 0 to
// everything, so its token gate cannot see a numerics deviation.
static int dbg_armed;

void infer_dbg(int n)
{
    dbg_armed = n;
}

int infer_dbg_armed(void)
{
    return dbg_armed;
}

static void dbg_typed(const char *tag, int layer, const void *values, uint32_t n,
                      const char *dtype)
{
    if (!dbg_armed) return;
    // f32/u32 are both four bytes, little-endian on the supported RISC-V targets.
    // Do not convert floats to integers: +/-zero and exact FP32 rounding matter.
    const uint8_t *p = values;
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < 4*n; i++) hash = (hash ^ p[i])*16777619u;
    printf("DBG pos=%d l=%d %s dtype=%s n=%lu e=0 %08lx\n", n_pos, layer, tag, dtype,
           (unsigned long)n, (unsigned long)hash);
}

void infer_dbg_f32(const char *tag, int layer, const float *values, uint32_t n)
{
    _Static_assert(sizeof(float) == 4, "trace ABI requires float32");
    dbg_typed(tag, layer, values, n, "f32");
}

static void dbg_sum(const char *tag, int l, uint32_t off, uint32_t bytes, int e)
{
    if (!dbg_armed)
        return;
    const volatile uint8_t *p = (const volatile uint8_t *)(SCRATCHPAD_BASE + off);
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < bytes; i++)
        h = (h ^ p[i])*16777619u;
    // The first four elements and the absmax, so a divergence shows its magnitude at once
    // (int8 regions carry "x8" in the tag).
    int is8 = 0;                                  // libbase has no strstr: "x8" by hand.
    for (const char *q = tag; q[0] && q[1]; q++)
        if (q[0] == 'x' && q[1] == '8') is8 = 1;
    int32_t v[4], amax = 0;
    uint32_t n = is8 ? bytes : bytes/2;
    for (uint32_t i = 0; i < n; i++) {
        int32_t x = is8 ? (int32_t)((const volatile int8_t *)p)[i]
                        : (int32_t)((const volatile int16_t *)p)[i];
        if (i < 4) v[i] = x;
        if (x < 0) x = -x;
        if (x > amax) amax = x;
    }
    printf("DBG pos=%d l=%d %s e=%d %08lx  amax=%ld v0..3=%ld,%ld,%ld,%ld\n", n_pos, l, tag, e,
           (unsigned long)h, (long)amax, (long)v[0], (long)v[1], (long)v[2], (long)v[3]);
}

/* Forward */
/*---------*/

// Forward one token; returns the next token id (only meaningful when want_logits).
static int forward(int token, int want_logits)
{
    if (hybrid_present()) {          // The 27B's layers: hybrid.c, the CPU path.
        infer_forwards++;
        int t = hybrid_forward(token, want_logits);
        if (dbg_armed) {
            if (want_logits) {
                uint32_t token_bits = (uint32_t)t;
                dbg_typed("argmax", -1, &token_bits, 1, "u32");
            }
            dbg_armed--;
        }
        n_pos = hybrid_pos();
        return t;
    }
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    infer_forwards++;
    if (tprof_armed && want_logits && --tprof_armed == 0)
        tprof_begin(); // armed=N traces the Nth generated token; disarms itself.
    rq8_prefetched = 0; // A logits-less forward leaves the out-norm row prefetched; drop it.
    perf_enter(OP_EMBED);
    if (likely(attn_all_hw())) {
        ex_state = embed_row_fetch(emb, token, SP_XR);
        x_amax   = X_AMAX_AFTER_EMBED;
        perf_exit(OP_EMBED);
        goto embed_done;
    }
    embed_row_f(emb, token, x);
    // Residual stream lives in the scratchpad as int16 BFP from here on.
    {
        float amax = 0;
        for (int i = 0; i < n_embd; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        int e = 0;
        if (amax > 0) { frexpf(amax, &e); e -= 15; }
        float is = ldexpf(1.0f, -e);
        volatile int16_t *pd = sp16(SP_XR);
        for (int i = 0; i < n_embd; i++)
            pd[i] = (int16_t)lroundf(x[i]*is);
        ex_state = e;
        x_amax   = X_AMAX_AFTER_EMBED;
    }
    perf_exit(OP_EMBED);
embed_done: ;
    dbg_sum("emb", -1, SP_XR, 2*n_embd, ex_state);

    int   kv_dim = n_kv_head*head_dim;
    int   q_dim  = n_head*head_dim;
    float scale  = 1.0f/sqrtf((float)head_dim);
    int   pos    = n_pos;

    int attn_hw = attn_all_hw();
    if (likely(attn_hw)) {
        // Rope table row for this position ([c14 | s14], Q14, from the image).
        const struct bons_entry *rt = model_entry_kind(ROLE_ROPE_TAB, 0, KIND_NORM16);
        uint32_t rb = (2u*head_dim + 31)/32;
        engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + (rt->chan_base_beats + (uint64_t)pos*rb)*32,
                     rb, SP_TAB);
        if (likely(seq_use)) {
            perf_enter(OP_SCORES);
            attn_list_prep_token(pos);
            perf_exit(OP_SCORES);
        }
    }

    for (int l = 0; l < n_layer; l++) {
        if (unlikely(tprof_active))
            tprof_event((3u << 28) | ((uint32_t)l << 8));
        // Attention block.
        int sx8 = op_rmsnorm_q8_sp(SP_XR, ex_state, attn_norm[l],
                                   nrm_attn16[l],
                                   n_embd, SP_H, SP_X);
        dbg_sum("attn_in_x8", l, SP_X, n_embd, E_NORM + sx8);
        int ev_res = 0;
        if (likely(attn_hw)) {
            // q/k/v: engine matvec -> vec norm_i32 -> int16 resident; qk-norm + rope on
            // the vector unit; K/V packed and stored to HBM. No CPU floats anywhere.
            const struct bons_entry *eqt = model_entry(ROLE_ATTN_Q, l);
            const struct bons_entry *ekt = model_entry(ROLE_ATTN_K, l);
            const struct bons_entry *evt = model_entry(ROLE_ATTN_V, l);
            perf_enter(OP_NORM_I32);
            int kq, kk2, kv2;
            // Resident q/k norm rows: one-time fetch of every layer's row. Ahead of the qkv
            // list, since qkvx runs the k-norm INSIDE that list.
            int qres = 0;
            {
                const struct bons_entry *qn16r = model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16);
                const struct bons_entry *kn16r = model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16);
                qres = (uint32_t)n_layer*head_dim*2 <= 0x1C00 && qn16r && kn16r;
            }
            if (qres && !qkn_resident) {
                for (int ll = 0; ll < n_layer; ll++) {
                    const struct bons_entry *q = model_entry_kind(ROLE_ATTN_Q_NORM, ll, KIND_NORM16);
                    const struct bons_entry *k = model_entry_kind(ROLE_ATTN_K_NORM, ll, KIND_NORM16);
                    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + q->chan_base_beats*32,
                                 q->chan_len_beats, SP_QNW + (uint32_t)ll*head_dim*2);
                    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + k->chan_base_beats*32,
                                 k->chan_len_beats, SP_KNW + (uint32_t)ll*head_dim*2);
                }
                qkn_resident = 1;
            }
            if (likely(seq_use && seq_qkv_base >= 0 && op_on_hw(OP_NORM_I32))) {
                // One sequencer run for q/k/v + norms (seq_build_qkv). The engine words
                // were posted under the previous FFN list (or the LM-head stream); the
                // call below is a no-op then. kq/kk/kv from the spilled absmaxes, the
                // same bitlen-14 the op path derives - byte-identical.
                qkv_prep_layer(l);
                if (qkvx_ok) { // Per-token part of both norms' eps index (LUT[sx8 + k]).
                    // The layer's tables cover s = sx8 + k in [0, 31], i.e. sx8 in [0, 13]
                    // with k <= 18 (an int32 source). quant8_shift can go negative for a
                    // tiny source, so outside that window refill both tables around this
                    // token's sx8 and index by k alone - the sequencer clamps, and a
                    // clamped index would be a silently wrong eps. Rare: after the
                    // attention rmsnorm the mantissas sit near 2^11..2^15 (sx8 4..8).
                    int imm = sx8;
                    if (sx8 < 0 || sx8 > 13) {
                        volatile uint32_t *lq = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_LUTQ);
                        volatile uint32_t *lk = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_LUTK);
                        int eq0 = E_NORM + sx8 - (int)eqt->bias + (int)eqt->rshift;
                        int ek0 = E_NORM + sx8 - (int)ekt->bias + (int)ekt->rshift;
                        for (int j = 0; j < 32; j++) {
                            lq[j] = (uint32_t)seq_eps_int(eq0 + j);
                            lk[j] = (uint32_t)seq_eps_int(ek0 + j);
                        }
                        qkv_prepped = -1; // The next layer's prep rewrites its own tables.
                        imm = 0;
                    }
                    seq_patch((uint32_t)seq_qkv_di_kn, 7, (uint32_t)imm);
                    seq_patch((uint32_t)seq_qkr_di_qn, 7, (uint32_t)imm);
                }
                seq_run((uint32_t)seq_qkv_base, (uint32_t)seq_qkv_len);
                uint32_t aq = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP);
                uint32_t ak = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP + 32);
                uint32_t av = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP + 64);
                kq  = (bitlen32(aq) > 14) ? bitlen32(aq) - 14 : 0;
                kk2 = (bitlen32(ak) > 14) ? bitlen32(ak) - 14 : 0;
                kv2 = (bitlen32(av) > 14) ? bitlen32(av) - 14 : 0;
            } else
#if defined(CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR)
            if (op_on_hw(OP_NORM_I32)) {
                cold_path(); // Dead on the card: the qkv list above runs instead.
                // Overlap: each norm runs UNDER the next projection's stream. Distinct
                // result buffers inside the SP_Y region (q 16 KB max, then k/v) keep the
                // in-flight dst disjoint from the norm's src; each op's absmax is captured
                // before the next submit. Same k's, same programs - byte-identical.
                struct engine_op mo = {0};
                model_matvec(eqt, SP_X, SP_Y);
                uint32_t aq = trixium_engine_csr_absmax_read();
                model_matvec_op(ekt, SP_X, SP_Y + 0x4000, &mo);
                uint32_t dk = engine_submit(&mo);
                kq = op_norm_i32_sp_a(SP_Y, SP_Q, q_dim, aq, 0);
                perf_enter(OP_ENGINE_WAIT);
                engine_join(dk);
                perf_exit(OP_ENGINE_WAIT);
                uint32_t ak = trixium_engine_csr_absmax_read();
                model_matvec_op(evt, SP_X, SP_Y + 0x5000, &mo);
                uint32_t dv = engine_submit(&mo);
                kk2 = op_norm_i32_sp_a(SP_Y + 0x4000, SP_K, kv_dim, ak, 0);
                perf_enter(OP_ENGINE_WAIT);
                engine_join(dv);
                perf_exit(OP_ENGINE_WAIT);
                kv2 = op_norm_i32_sp(SP_Y + 0x5000, SP_V, kv_dim);
            } else
#endif
            {
                cold_path();
                model_matvec(eqt, SP_X, SP_Y);
                kq = op_norm_i32_sp(SP_Y, SP_Q, q_dim);
                model_matvec(ekt, SP_X, SP_Y);
                kk2 = op_norm_i32_sp(SP_Y, SP_K, kv_dim);
                model_matvec(evt, SP_X, SP_Y);
                kv2 = op_norm_i32_sp(SP_Y, SP_V, kv_dim);
            }
            int eq_r = (E_NORM + sx8) - (int)eqt->bias + (int)eqt->rshift + kq;
            int ek_r = (E_NORM + sx8) - (int)ekt->bias + (int)ekt->rshift + kk2;
            ev_res = (E_NORM + sx8) - (int)evt->bias + (int)evt->rshift + kv2;
            perf_exit(OP_NORM_I32);
            perf_enter(OP_QKNORM);
            const struct bons_entry *qn16 = model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16);
            const struct bons_entry *kn16 = model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16);
            uint32_t dqn = 0;
            (void)qn16; (void)kn16; // qres and the residency fetch: ahead of the qkv list.
            if (qres && qkvx_ok) {
                // Both A1s were patched with the layer's engine words (qkv_prep_layer);
                // the K chain has already run inside the qkv list.
            } else if (qres) {
                seq_patch((uint32_t)seq_qkr_di_qn, 3, SP_QNW + (uint32_t)l*head_dim*2); // rmsnorm_h A1 (q).
                seq_patch((uint32_t)seq_qkr_di_kn, 3, SP_KNW + (uint32_t)l*head_dim*2); // rmsnorm_h A1 (k).
            } else {  // The q-norm row streams under the eps/KV prep below.
                dqn = norm_fetch_submit(qn16);
            }
            {
                // One sequencer run replaces the 2*(n_head+n_kv) per-head dispatches;
                // only the eps args change per layer (descriptor word 7 = A5).
                int epq = seq_eps_int(eq_r), epk = seq_eps_int(ek_r);
                // qkvx: the sequencer reads both from the layer's LUTs, and qkv_prep_layer
                // patched both A6s.
                if (!qkvx_ok) {
                    seq_patch((uint32_t)seq_qkr_di_qn, 7, (uint32_t)epq);   // rmsnorm_h A5 (q heads).
                    seq_patch((uint32_t)seq_qkr_di_kn, 7, (uint32_t)epk);   // rmsnorm_h A5 (k heads).
                    seq_patch((uint32_t)seq_qkr_di_qn, 8, (uint32_t)w16_shift(qn16)); // A6 (gamma exp).
                    seq_patch((uint32_t)seq_qkr_di_kn, 8, (uint32_t)w16_shift(kn16));
                }
                if (kv_pipe_active) {
                    // In-list stores: the scale beats must exist before the list runs
                    // (the store streams them), and each store's target row moves per
                    // layer/token. The exponent ratchet is identical bookkeeping to a
                    // post-list placement.
                    // Only the target address is per (layer, slot): the beat count is a
                    // model constant once kv_rb is known and the channel is per TOKEN, so
                    // those are rewritten only on change (posted writes under a running list
                    // are expensive). This removes writes, not the arithmetic between them:
                    // that arithmetic is what keeps the sequencer's descriptor fetch alive.
                    uint32_t rowoff = (uint32_t)(pos/n_ch)*kv_rb*32;
                    uint32_t chan   = (uint32_t)(pos % n_ch);
                    int wr_rb   = (kvst_rb_set   != (uint32_t)kv_rb);
                    int wr_chan = (kvst_chan_set != chan);
                    for (int hh = 0; hh < n_kv_head; hh++) {
                        kv_pack_prep_res(l, hh, 0, E_NORM);
                        kv_pack_prep_res(l, hh, 1, ev_res);
                        if (kv_mstore_active)
                            continue; // One descriptor for all slots, patched below.
                        for (int kind = 0; kind < 2; kind++) {
                            uint32_t sdi = seq_kvst_di[hh*2 + kind];
                            seq_patch(sdi, 4, kv_region_addr(l, hh, kind) + rowoff);
                            if (wr_rb)   seq_patch(sdi, 5, (uint32_t)kv_rb);
                            if (wr_chan) seq_patch(sdi, 10, chan);
                        }
                    }
                    if (kv_mstore_active) {
                        uint32_t sdi = seq_kvst_di[0];
                        seq_patch(sdi, 4, kv_region_addr(l, 0, 0) + rowoff);
                        if (wr_rb)   seq_patch(sdi, 5, (uint32_t)kv_rb);
                        if (wr_chan) seq_patch(sdi, 10, chan);
                    }
                    kvst_rb_set   = (uint32_t)kv_rb;
                    kvst_chan_set = chan;
                }
                if (!qres) {
                    engine_join(dqn);
                    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                                 kn16->chan_len_beats, SP_FET2);
                }
                if (likely(seq_use && seq_attn_heads >= n_head
                    && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV))) {
                    // Overlap (sideband slot): the attention list's per-layer patches
                    // run while the sequencer executes the qkr list. Without the slot
                    // the first patch would simply stall until the list finished.
                    seq_list_start(0, seq_qkr_count);
                    attn_patch_slots(l, 0, n_head, 0);
                    attn_prepatched = 1;
                    if (seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                        && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
                        ffn_prep_layer(l); // The FFN rebuild rides the same window.
                        ffn_prepatched = 1;
                    }
                    seq_join();
                } else {
                    seq_run(0, seq_qkr_count);
                }
            }
            perf_exit(OP_QKNORM);
            dbg_sum("q16", l, SP_Q, 2*q_dim, E_NORM);      // q heads normed + roped
            dbg_sum("k16", l, SP_K, 2*kv_dim, E_NORM);     // k heads normed + roped
            dbg_sum("v16", l, SP_V, 2*kv_dim, ev_res);     // v after norm_i32
            perf_enter(OP_KVQUANT);
            for (int hh = 0; hh < n_kv_head && !kv_pipe_active; hh++) {
                // ...the quant8s already ran inside the qkr list; prep only.
                // (With the in-list stores the prep AND the stores already ran with the list.)
                kv_pack_prep_res(l, hh, 0, E_NORM);
                kv_pack_prep_res(l, hh, 1, ev_res);
            }
#ifdef CSR_TRIXIUM_ENGINE_CSR_REP_ADDR
            // Only without the in-list NOWAIT stores (the default): with them, the list has
            // already written every KV row (a second store would cost time, not tokens).
            if (!kv_pipe_active) {
                // ...then ONE multi-packet store for all 2*n_kv_head slots: the (kvh, kind)
                // regions of a layer are consecutive (kv_region beats apart) and the staging
                // slots 0x200 B apart, so one op scatters them all.
                struct engine_op op = {0};
                op.kind    = ENGINE_KIND_STORE;
                op.chan    = pos % n_ch;
                op.wbase   = kv_region_addr(l, 0, 0) + (uint32_t)(pos/n_ch)*kv_rb*32;
                op.wlen    = kv_rb;
                op.xsrc    = kvp_slot(0, 0);
                op.rep     = (uint8_t)(2*n_kv_head);
                op.dstride = kv_region;
                op.sstride = 0x200/32;
                engine_run(&op);
            }
#else
            if (!kv_pipe_active)
                for (int hh = 0; hh < n_kv_head; hh++) { // ...hand the port to the engine.
                    kv_store_row_res(l, hh, 0, pos);
                    kv_store_row_res(l, hh, 1, pos);
                }
#endif
            perf_exit(OP_KVQUANT);
            goto attn_scores;
        }
        {
            float sx = ldexpf(1.0f, E_NORM + sx8);
            matvec_rows_f(model_entry(ROLE_ATTN_Q, l), 0, q_dim/n_ch,  sx, q, 0, 0, 0);
            matvec_rows_f(model_entry(ROLE_ATTN_K, l), 0, kv_dim/n_ch, sx, k, 0, 0, 0);
            matvec_rows_f(model_entry(ROLE_ATTN_V, l), 0, kv_dim/n_ch, sx, v, 0, 0, 0);
        }

        perf_enter(OP_QKNORM);
        for (int hh = 0; hh < n_head; hh++)
            rmsnorm_f(q + hh*head_dim, q + hh*head_dim, q_norm[l], head_dim, H->rms_eps);
        for (int hh = 0; hh < n_kv_head; hh++)
            rmsnorm_f(k + hh*head_dim, k + hh*head_dim, k_norm[l], head_dim, H->rms_eps);
        perf_exit(OP_QKNORM);
        perf_enter(OP_ROPE);
        rope_apply(q, n_head,    pos);
        rope_apply(k, n_kv_head, pos);
        perf_exit(OP_ROPE);

        // Append K/V (int8 + per-row scale) to the CPU-side caches.
        perf_enter(OP_KVQUANT);
        {
            if (pos >= cpu_kv_ctx)
                goto cpu_kv_full; // Cache capped; HW KV (full ctx) is unaffected.
            int8_t *kd = k_cache + ((uint64_t)l*cpu_kv_ctx + pos)*kv_dim;
            int8_t *vd = v_cache + ((uint64_t)l*cpu_kv_ctx + pos)*kv_dim;
            // Per-head scales, not one per token: the head spread of K is large (16x at layer 0
            // of the 1.7B), so a single tensor scale leaves the quiet heads under 3 bits and
            // wrecks their scores.
            for (int h = 0; h < n_kv_head; h++) {
                const float *kh = k + h*head_dim, *vh = v + h*head_dim;
                float ka = 0, va = 0;
                for (int i = 0; i < head_dim; i++) {
                    float a = fabsf(kh[i]); if (a > ka) ka = a;
                    a = fabsf(vh[i]);       if (a > va) va = a;
                }
                uint64_t si = ((uint64_t)l*cpu_kv_ctx + pos)*n_kv_head + h;
                float ks = (ka > 0) ? ka/127.0f : 1.0f, is = 1.0f/ks;
                for (int i = 0; i < head_dim; i++)
                    kd[h*head_dim + i] = (int8_t)lroundf(kh[i]*is);
                k_scale[si] = ks;
                float vs = (va > 0) ? va/127.0f : 1.0f; is = 1.0f/vs;
                for (int i = 0; i < head_dim; i++)
                    vd[h*head_dim + i] = (int8_t)lroundf(vh[i]*is);
                v_scale[si] = vs;
            }
        }
cpu_kv_full: ;
        // Dual-KV: also append Q8 row packets to HBM for the engine scores path.
        if (op_on_hw(OP_KVQUANT)) {
            for (int h = 0; h < n_kv_head; h++) {
                kv_store_row(l, h, 0, pos, k + h*head_dim);
                kv_store_row(l, h, 1, pos, v + h*head_dim);
            }
        }
        perf_exit(OP_KVQUANT);

attn_scores: ;
        // Resident attention output: per-head int16 in SP_H + per-head value scales,
        // aligned to a common ea8 and scale8-ed into SP_X after the head loop.
        aw_hvalid = 0; // The fold starts empty for every path below (the list paths reset
                       // it again where they fold; the op-by-head path relies on this).
        // Attention (CPU, GQA).
        int n_rep = n_head/n_kv_head;
        if (likely(seq_use && attn_hw && seq_attn_heads > 0
            && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV))) {
            // Whole-head-loop descriptor lists: ~6 dispatch/poll round-trips per head
            // collapse into one submit per chunk; the AXPY rshift and norm k are computed
            // in-sequencer from the chained res6. Numerics identical to the fused path.
            perf_enter(OP_SCORES);
            // Token-invariant words (rowsw/wlen/blocks, score padding) were patched once
            // per forward by attn_list_prep_token(); per layer only the KV addresses and
            // the B/R/sigma exponents move. The four head-static words (q src, spill
            // addresses, norm dst) are already what seq_build_attn wrote for slot j
            // whenever the list holds every head (hh == j); they are re-patched only in
            // the chunked case, where a slot serves several heads.
            int chunked = (seq_attn_heads < n_head);
#ifdef TRIXIUM_SP_SB_RSLOT
            if (likely(!chunked)) {
                // Fold head j while heads j+2.. still execute. The read slot lets
                // the spill reads grant mid-list; dcount ("descriptors completed")
                // past norm(j) proves j's spills are written. The fold work and the
                // per-head CSR polls replace seq_join's idle spin.
                if (!attn_prepatched)
                    attn_patch_slots(l, 0, n_head, 0);
                attn_prepatched = 0;
                seq_list_start((uint32_t)seq_attn_base, (uint32_t)attn_list_len(n_head));
                perf_exit(OP_SCORES);
                perf_enter(OP_WV);
                aw_hvalid = 0;
                for (int hh = 0; hh < n_head; hh++) {
                    uint32_t ready = attn_fold_ready(hh, n_head), spins = 0;
                    for (;;) {
                        uint32_t st = trixium_seq_status_read();
                        tprof_dcount(st);
                        if (!(st & 1) || ((st >> 8) >= ready))
                            break;
                        if (++spins > 4000000u) break; // seq_join below reports the wedge.
                    }
                    attn_fold_head(l, hh);
                }
                perf_exit(OP_WV);
                perf_enter(OP_SCORES);
                seq_join();
                perf_exit(OP_SCORES);
                goto attn_heads_done;
            }
#endif
            for (int h0 = 0; h0 < n_head; h0 += seq_attn_heads) {
                int nh = n_head - h0;
                if (nh > seq_attn_heads) nh = seq_attn_heads;
                if (!attn_prepatched)
                    attn_patch_slots(l, h0, nh, chunked);
                seq_run((uint32_t)seq_attn_base, (uint32_t)attn_list_len(nh));
            }
            attn_prepatched = 0;
            perf_exit(OP_SCORES);
            // One fold per head from the spilled p8 sums / absmaxes (attn_fold_head).
            perf_enter(OP_WV);
            aw_hvalid = 0;
            for (int hh = 0; hh < n_head; hh++)
                attn_fold_head(l, hh);
            perf_exit(OP_WV);
            goto attn_heads_done;
        }
        cold_path(); // Dead on the card: the attention list above runs instead.
        for (int hh = 0; hh < n_head; hh++) {
            int kvh = hh/n_rep;
            const float *qh = q + hh*head_dim;
            float smax = -1e30f;
            if (likely(op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV))) {
                // Fused resident path: scores (engine) -> softmax8 (vec, p8 + exact sum)
                // -> weighted-V (engine AXPY) -> cross-channel reduce (vec). Scores never
                // touch the CPU; the 1/sum fold is one float divide per head.
                perf_enter(OP_SCORES);
                int eq;
                if (likely(attn_hw)) {
                    vec_quant8_head(SP_Q + hh*head_dim*2, SP_X, head_dim);
                    eq = E_NORM + 8;
                } else {
                    eq = quantize_f_pow2(qh, head_dim, SP_X);
                }
                int Bk, Rk;
                kv_scores_br(l, kvh, 0, &Bk, &Rk);
                struct engine_op op = {0};
                op.kind   = ENGINE_KIND_Q8;
                op.rows   = (pos + 1)/n_ch;
                op.rem    = (pos + 1) % n_ch;
                op.ilv    = 1;
                op.cols   = head_dim;
                op.wbase  = kv_region_addr(l, kvh, 0);
                op.wlen   = (uint32_t)op.rows*kv_rb;
                op.sbeats = 1;
                op.cbeats = head_dim/32;
                op.bias   = Bk;
                op.rshift = Rk;
                op.xsrc   = SP_X;
                op.dst    = SP_SC;
                engine_run(&op);
                perf_exit(OP_SCORES);
                perf_enter(OP_SOFTMAX);
                int cnt = pos + 1;
                int padn = SC_PADN(cnt);
                volatile int32_t *sc32 = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
                for (int i2 = 0; i2 < padn; i2++)
                    sc32[cnt + i2] = (int32_t)0x80000000;
                // Attention scale 1/sqrt(head_dim): whole powers of two fold into
                // sigma, an odd half folds into the log2e constant (739 -> 523).
                int lg = 0;
                while ((1 << lg) < head_dim) lg++;
                int Ceff  = (lg & 1) ? 523 : 739;
                int sigma = eq - Bk + Rk - (lg >> 1);
                int s1v = 11;
                int32_t sargs[7] = {(int32_t)SP_SC, Ceff, (int32_t)SP_X, (cnt + padn)/16,
                                    -s1v, sigma + 6 + s1v, (int32_t)(SP_AWP8 + hh*32)};
                uint64_t P8 = vec_run(PROG_SOFTMAX8, sargs, 7);
                perf_exit(OP_SOFTMAX);
                perf_enter(OP_WV);
                int Bv, Rd;
                kv_scores_br(l, kvh, 1, &Bv, &Rd);
                int Rpre = axpy_rpre_c(P8 ? P8 : 1);
                struct engine_op wv = {0};
                wv.kind   = ENGINE_KIND_AXPY;
                wv.rows   = (pos + 1)/n_ch;
                wv.rem    = (pos + 1) % n_ch;
                wv.cols   = head_dim;
                wv.wbase  = kv_region_addr(l, kvh, 1);
                wv.wlen   = (uint32_t)wv.rows*kv_rb;
                wv.sbeats = 1;
                wv.cbeats = head_dim/32;
                wv.bias   = Bv;
                wv.rshift = Rpre;
                wv.xsrc   = SP_X;
                wv.dst    = SP_SC;
                engine_run(&wv);
                uint64_t am = 0;
                for (int c = 1; c < n_ch; c++) {
                    int32_t rargs[4] = {(int32_t)SP_SC, (int32_t)(SP_SC + c*head_dim*4),
                                        (int32_t)SP_SC, head_dim/8};
                    am = vec_run(PROG_REDUCE_ACC, rargs, 4);
                }
                if (n_ch == 1)
                    am = trixium_engine_csr_absmax_read(); // AXPY partials tracked too.
                int k2 = (bitlen32(am) > 14) ? bitlen32(am) - 14 : 0; // == the >16383 scan, via clz.
                int32_t nargs[5] = {(int32_t)SP_SC, 0,
                                    (int32_t)(SP_H + hh*head_dim*2), head_dim/16, k2};
                vec_run(PROG_NORM_I32, nargs, 5);
                // Both fold forms, same producer as the list path (see attn_fold_vals):
                // Bv/Rpre/k2 there are exactly the ones computed above.
                attn_fold_vals(l, hh, (uint32_t)P8, (uint32_t)am);
                perf_exit(OP_WV);
                continue;
            }
            perf_enter(OP_SCORES);
            if (likely(op_on_hw(OP_SCORES))) {
                // q head -> int8 @ 2^eq in SP_X, one striped+interleaved Q8 op over the
                // K region: int32 scores land position-major in SP_SC.
                int eq = quantize_f_pow2(qh, head_dim, SP_X);
                int B, R;
                kv_scores_br(l, kvh, 0, &B, &R);
                struct engine_op op = {0};
                op.kind   = ENGINE_KIND_Q8;
                op.rows   = (pos + 1)/n_ch;
                op.rem    = (pos + 1) % n_ch;
                op.ilv    = 1;
                op.cols   = head_dim;
                op.wbase  = kv_region_addr(l, kvh, 0);
                op.wlen   = (uint32_t)op.rows*kv_rb;
                op.sbeats = 1;
                op.cbeats = head_dim/32;
                op.bias   = B;
                op.rshift = R;
                op.xsrc   = SP_X;
                op.dst    = SP_SC;
                engine_run(&op);
                volatile int32_t *sc32 = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
                float fs = ldexpf(scale, eq - B + R);
                for (int t = 0; t <= pos; t++) {
                    float sc = (float)sc32[t]*fs;
                    scores[t] = sc;
                    if (sc > smax) smax = sc;
                }
            } else {
                // The whole CPU fallback (scores from the CPU-side k_cache).
                if (pos >= cpu_kv_ctx) {
                    // The CPU fallback cache is capped (see CPU_KV_CTX); results past it
                    // would read out of bounds. The HW path carries full context.
                    puts("CPU attention past CPU_KV_CTX: use hwops attention or shorter prompts");
                    pos = cpu_kv_ctx - 1;
                }
                for (int t = 0; t <= pos; t++) {
                    const int8_t *kt = k_cache + ((uint64_t)l*cpu_kv_ctx + t)*kv_dim + kvh*head_dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qh[d]*kt[d];
                    float sc = dot*k_scale[((uint64_t)l*cpu_kv_ctx + t)*n_kv_head + kvh]*scale;
                    scores[t] = sc;
                    if (sc > smax) smax = sc;
                }
            }
            perf_exit(OP_SCORES);
            perf_enter(OP_SOFTMAX);
            float sum = 0;
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - smax);
                sum += scores[t];
            }
            perf_exit(OP_SOFTMAX);
            float inv = 1.0f/sum;
            perf_enter(OP_WV);
            float *ah = attn + hh*head_dim;
            if (likely(op_on_hw(OP_WV))) {
                // Probs -> int8 @ 2^ep (position-major in SP_X), one AXPY op over the V
                // region; each channel lands head_dim int32 partials chan-major in SP_SC.
                float pmax = 0;
                for (int t = 0; t <= pos; t++) {
                    float p = scores[t]*inv;
                    if (p > pmax) pmax = p;
                }
                int ep = 0;
                if (pmax > 0) { frexpf(pmax, &ep); ep -= 7; }
                float pis = ldexpf(1.0f, -ep);
                volatile uint32_t *px = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_X);
                for (int t = 0; t <= pos; t += 4) {
                    uint32_t w = 0;
                    for (int b = 0; b < 4 && t + b <= pos; b++) {
                        int p8 = (int)lroundf(scores[t+b]*inv*pis);
                        if (p8 > 127) p8 = 127;
                        if (p8 < 0)   p8 = 0;
                        w |= ((uint32_t)p8) << (8*b);
                    }
                    px[t/4] = w;
                }
                int B, Rd;
                kv_scores_br(l, kvh, 1, &B, &Rd);
                int Rpre = axpy_rpre_c((uint64_t)(1u << -ep) + (uint32_t)(pos + 1));
                struct engine_op op = {0};
                op.kind   = ENGINE_KIND_AXPY;
                op.rows   = (pos + 1)/n_ch;
                op.rem    = (pos + 1) % n_ch;
                op.cols   = head_dim;
                op.wbase  = kv_region_addr(l, kvh, 1);
                op.wlen   = (uint32_t)op.rows*kv_rb;
                op.sbeats = 1;
                op.cbeats = head_dim/32;
                op.bias   = B;
                op.rshift = Rpre;
                op.xsrc   = SP_X;
                op.dst    = SP_SC;
                engine_run(&op);
                volatile int32_t *pt = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
                float fs = ldexpf(1.0f, ep + Rpre - B);
                for (int d = 0; d < head_dim; d++) {
                    float acc = 0;
                    for (int c = 0; c < n_ch; c++)
                        acc += (float)pt[c*head_dim + d];
                    ah[d] = acc*fs;
                }
                perf_exit(OP_WV);
                continue;
            }
            for (int d = 0; d < head_dim; d++) ah[d] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *vt = v_cache + ((uint64_t)l*cpu_kv_ctx + t)*kv_dim + kvh*head_dim;
                float p = scores[t]*inv*v_scale[((uint64_t)l*cpu_kv_ctx + t)*n_kv_head + kvh];
                for (int d = 0; d < head_dim; d++) ah[d] += p*vt[d];
            }
            perf_exit(OP_WV);
        }
attn_heads_done: ;

        // Output projection + residual (resident).
        {
            const struct bons_entry *eo = model_entry(ROLE_ATTN_OUT, l);
            int ea8;
            if (likely(attn_hw)) {
                perf_enter(OP_QUANT8);
                ea8 = 0;
                if (aw_hvalid) ea8 = aw_he2 + 24 - 7;   // frexpf exponent of max hm, -7.
                for (int hh = 0; hh < n_head; hh++) {
                    int32_t Mh = attn_mh_int(aw_qm[hh], aw_qe[hh] + 15 - ea8);
                    if (likely(seq_use && seq_scale8_base >= 0)) {
                        seq_patch((uint32_t)(seq_scale8_base + hh), 6, (uint32_t)Mh);
                        continue;                       // One list run below.
                    }
                    int32_t sa2[5] = {(int32_t)(SP_H + hh*head_dim*2), 0,
                                      (int32_t)(SP_X + hh*head_dim), head_dim/16, Mh};
                    vec_run(PROG_SCALE8, sa2, 5);
                }
                if (likely(seq_use && seq_scale8_base >= 0))
                    seq_run((uint32_t)seq_scale8_base, (uint32_t)n_head);
                perf_exit(OP_QUANT8);
            } else {
                ea8 = quantize_f_pow2(attn, q_dim, SP_X);
            }
            dbg_sum("attn_x8", l, SP_X, q_dim, ea8);
            // The attention list's per-head spills (int32 as int16 pairs: the low half of
            // each 32 B beat's first word is what the fold reads) and scores buffer B.
            dbg_sum("awp8", l, SP_AWP8, n_head*32, 0);
            dbg_sum("awam", l, SP_AWAM, n_head*32, 0);
            dbg_sum("scb",  l, SP_SCB2, 64, 0);
            model_matvec(eo, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            uint32_t dmax = 0;
            int kk = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
            perf_exit(OP_NORM_I32);
            int ed = ea8 - (int)eo->bias + (int)eo->rshift + kk;
            ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                    nrm_ffn16[l]);
            dbg_sum("attn_res", l, SP_XR, 2*n_embd, ex_state);
        }

        // FFN (fully resident: no float round-trips).
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            int ex8 = E_NORM + op_rmsnorm_q8_sp(SP_XR, ex_state, ffn_norm[l],
                                                nrm_ffn16[l],
                                                n_embd, SP_H, SP_X);
            dbg_sum("ffn_x8", l, SP_X, n_embd, ex8);

            if (likely(seq_use && seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8))) {
                // One list for gate->norm->up->norm->silu->quant8->down (see seq_build_ffn).
                // Whole-list time is charged to OP_SILU_MUL: per-op shares are documented as
                // unreliable under sequencing; totals are the measurement.
                if (!ffn_prepatched)
                    ffn_prep_layer(l);
                ffn_prepatched = 0;
                // silu A4 = eg + 6 = imm + kg (VSHLS convention); the descriptor carries
                // w0[11] so its kreg0 term ADDS.
                seq_patch((uint32_t)(seq_ffn_base + ffn_di_a4), 6,
                          (uint32_t)(ex8 - (int)eg_t->bias + (int)eg_t->rshift + 6));
                perf_enter(OP_SILU_MUL);
                if (unlikely(fprof_use)) {
                    perf_ovl_mark();
                    if (l == 0)
                        fprof_nts = seq_run_trace((uint32_t)seq_ffn_base,
                                                  (uint32_t)seq_ffn_len, fprof_ts, 16);
                    else
                        seq_run((uint32_t)seq_ffn_base, (uint32_t)seq_ffn_len);
                    perf_ovl_acc(fprof_acc_list); fprof_nlist++;
                } else {
                    seq_list_start((uint32_t)seq_ffn_base, (uint32_t)seq_ffn_len);
                    if (l + 1 < n_layer)
                        qkv_prep_layer(l + 1); // Next layer's projections, under this stream.
                    seq_join();
                }
                perf_exit(OP_SILU_MUL);
                // Recover the folds the op-by-op path computes - identical values:
                // kg/ku from the spilled engine absmaxes, s from quant8's surviving res6.
                uint32_t absg = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP);
                uint32_t absu = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP + 32);
                int kg = (bitlen32(absg) > 14) ? bitlen32(absg) - 14 : 0; // == the >16383 scan.
                int ku = (bitlen32(absu) > 14) ? bitlen32(absu) - 14 : 0;
                int eg  = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;
                int eu  = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;
                int es8 = (eg + eu + 15)
                        + quant8_shift((uint32_t)trixium_vecunit_res6_read());
                dbg_sum("ffn_g16", l, SP_G, 2*ffn_dim, eg);      // silu(gate)*up, int16
                dbg_sum("silu_x8", l, SP_X, ffn_dim, es8);
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;
                ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                        next_norm16(l));
                dbg_sum("ffn_res", l, SP_XR, 2*n_embd, ex_state);
                goto ffn_done;
            }
            cold_path(); // Dead on the card: the FFN list above runs instead.
            if (unlikely(fprof_use)) {
                perf_ovl_mark();
                if (l == 0) { fprof_nsteps = 0; fprof_steps[fprof_nsteps++] = perf_now(); }
            }
            model_matvec(eg_t, SP_X, SP_Y);
            if (unlikely(fprof_use) && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int kg;
#if defined(CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR)
            if (op_on_hw(OP_NORM_I32)) {
                // Overlap: the up stream runs UNDER the gate norm. Gate absmax is read
                // BEFORE the submit (the tracker resets on accept); regions are disjoint
                // (up: SP_X -> SP_YU, norm: SP_Y -> SP_G). Same kg, same program, same
                // args as the serial order - byte-identical by construction.
                uint32_t ag = trixium_engine_csr_absmax_read();
                struct engine_op mo = {0};
                model_matvec_op(eu_t, SP_X, SP_YU, &mo);
                uint32_t d0 = engine_submit(&mo);
                perf_enter(OP_NORM_I32);
                kg = op_norm_i32_sp_a(SP_Y, SP_G, ffn_dim, ag, 0);
                perf_exit(OP_NORM_I32);
                perf_enter(OP_ENGINE_WAIT);
                engine_join(d0);
                perf_exit(OP_ENGINE_WAIT);
            } else
#endif
            {
                perf_enter(OP_NORM_I32);
                kg = op_norm_i32_sp(SP_Y, SP_G, ffn_dim);
                perf_exit(OP_NORM_I32);
                model_matvec(eu_t, SP_X, SP_YU);
            }
            if (unlikely(fprof_use) && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int eg = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;

            perf_enter(OP_NORM_I32);
            int ku = op_norm_i32_sp(SP_YU, SP_U, ffn_dim);
            perf_exit(OP_NORM_I32);
            if (unlikely(fprof_use) && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int eu = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;

            int es = eg + eu + 15;
            int es8 = es + op_silu_q8_sp(SP_G, eg, SP_U, eu, ffn_dim, SP_X);
            dbg_sum("silu_x8", l, SP_X, ffn_dim, es8);
            if (unlikely(fprof_use) && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();

            model_matvec(ed_t, SP_X, SP_Y);
            if (unlikely(fprof_use)) {
                if (l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
                perf_ovl_acc(fprof_acc_ops); fprof_nops++;
            }
            perf_enter(OP_NORM_I32);
            uint32_t dmax = 0;
            int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
            perf_exit(OP_NORM_I32);
            int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;
            ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                    next_norm16(l));
            dbg_sum("ffn_res", l, SP_XR, 2*n_embd, ex_state);
            ffn_done: ;
        }
    }
    if (dbg_armed && !want_logits)
        dbg_armed--;
    n_pos++;

    if (!want_logits)
        return -1;

    // Final norm + LM head (tied or separate), chunked over rows to fit the scratchpad.
    int lm_sx8 = op_rmsnorm_q8_sp(SP_XR, ex_state, out_norm,
                                  nrm_out16, n_embd, SP_H, SP_X);
    perf_enter(OP_LMHEAD);
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head)
        head = model_entry(ROLE_TOK_EMB, 0);
    uint32_t rpc       = head->rows/n_ch;
    uint32_t chunk_rpc = 0x20000/(4*n_ch); // 128 KB of int32 results per chunk.
    if (chunk_rpc > rpc) chunk_rpc = rpc;

    if (op_on_hw(OP_LMHEAD) && smp_on()) {
        // Sampling: SMP_SLICES row-range ops, one candidate per slice from the tap.
        uint32_t rb = bons_row_beats(head->cols, head->kind, head->group);
        uint32_t srows = (rpc + SMP_SLICES - 1)/SMP_SLICES;
        int32_t  cval[SMP_SLICES]; int ctok[SMP_SLICES]; int nc = 0;
        for (uint32_t r0 = 0; r0 < rpc; r0 += srows) {
            uint32_t rows = (r0 + srows <= rpc) ? srows : (rpc - r0);
            struct engine_op op = {0};
            op.kind   = ENGINE_KIND_TERNARY;
            op.rows   = rows;
            op.cols   = head->cols;
            op.wbase  = TRIXIUM_CHAN_WEIGHT_BASE + (head->chan_base_beats + (uint64_t)r0*rb)*32;
            op.wlen   = rows*rb;
            op.sbeats = bons_scale_beats(head->cols, head->group);
            op.cbeats = bons_code_beats(head->cols, head->kind);
            op.bias   = head->bias;
            op.rshift = head->rshift;
            op.xsrc   = SP_X;
            op.dst    = SP_Y;
            op.supw   = 1;
            engine_run(&op);
            uint32_t idx = trixium_engine_csr_argmax_idx_read();   // grant*rows + row
            uint32_t tok = (idx/rows)*rpc + r0 + idx % rows;
            if (idx != 0xFFFFFF && tok < H->vocab) {
                cval[nc] = (int32_t)trixium_engine_csr_argmax_val_read();
                ctok[nc] = (int)tok;
                nc++;
            }
        }
        if (likely(seq_use)) {   // the same next-token prep the greedy path posts
            attn_list_prep_token(n_pos);
            qkv_prep_layer(0);
        }
        perf_exit(OP_LMHEAD);
        if (unlikely(tprof_active))
            tprof_end();
        float lscale = ldexpf(1.0f, E_NORM + lm_sx8 - (int)head->bias + (int)head->rshift);
        int pick = nc ? smp_pick(cval, ctok, nc, lscale) : 0;
        if (dbg_armed) {
            printf("DBG pos=%d l=-1 argmax e=0 %08lx\n", n_pos - 1, (unsigned long)pick);
            dbg_armed--;
        }
        return pick;
    }
    if (likely(op_on_hw(OP_LMHEAD))) {
        // One suppressed-write op over every row; the argmax tap returns value+index
        // (global row order, ties to smallest - same as the CPU scan). Padding rows
        // (vocab not divisible by n_ch) can win with garbage logits: fall through to
        // the chunked CPU scan in that rare case.
        uint32_t rb = bons_row_beats(head->cols, head->kind, head->group);
        struct engine_op op = {0};
        op.kind   = ENGINE_KIND_TERNARY;
        op.rows   = rpc;
        op.cols   = head->cols;
        op.wbase  = TRIXIUM_CHAN_WEIGHT_BASE + head->chan_base_beats*32;
        op.wlen   = rpc*rb;
        op.sbeats = bons_scale_beats(head->cols, head->group);
        op.cbeats = bons_code_beats(head->cols, head->kind);
        op.bias   = head->bias;
        op.rshift = head->rshift;
        op.xsrc   = SP_X;
        op.dst    = SP_Y;
        op.supw   = 1;
        if (likely(seq_use)) {
            // The next forward's per-token list prep (score padding + the
            // rowsw/wlen/blocks patches) depends only on the position, never on the
            // token this argmax is about to pick - post it under the argmax stream.
            // supw=1 leaves the W port idle, so the posted writes drain freely.
            uint32_t dl = engine_submit(&op);
            attn_list_prep_token(n_pos);
            qkv_prep_layer(0);   // Next forward's layer-0 projections ride here too.
            engine_join(dl);
        } else {
            engine_run(&op);
        }
        uint32_t idx = trixium_engine_csr_argmax_idx_read();
        if (idx < H->vocab) {
            perf_exit(OP_LMHEAD);
            if (unlikely(tprof_active))
                tprof_end();
            if (dbg_armed) {
                printf("DBG pos=%d l=-1 argmax e=0 %08lx\n", n_pos - 1, (unsigned long)idx);
                dbg_armed--;
            }
            return (int)idx;
        }
    }
    int32_t best = -0x7FFFFFFF;
    int     best_tok = 0;
    volatile int32_t *y = (volatile int32_t *)(SCRATCHPAD_BASE + SP_Y);
    for (uint32_t r0 = 0; r0 < rpc; r0 += chunk_rpc) {
        uint32_t rows = (r0 + chunk_rpc <= rpc) ? chunk_rpc : (rpc - r0);
        uint32_t rb   = bons_row_beats(head->cols, head->kind, head->group);
        struct engine_op op = {0};
        op.kind   = ENGINE_KIND_TERNARY;
        op.rows   = rows;
        op.cols   = head->cols;
        op.wbase  = TRIXIUM_CHAN_WEIGHT_BASE + (head->chan_base_beats + (uint64_t)r0*rb)*32;
        op.wlen   = rows*rb;
        op.sbeats = bons_scale_beats(head->cols, head->group);
        op.cbeats = bons_code_beats(head->cols, head->kind);
        op.bias   = head->bias;
        op.rshift = head->rshift;
        op.xsrc   = SP_X;
        op.dst    = SP_Y;
        engine_run(&op);
        // Results: chan-major [ch][row]; global token id = ch*rpc + (r0 + row).
        for (int ch = 0; ch < n_ch; ch++) {
            for (uint32_t r = 0; r < rows; r++) {
                int32_t val = y[ch*rows + r];
                uint32_t tok = ch*rpc + r0 + r;
                if (tok < H->vocab && val > best) {
                    best = val;
                    best_tok = tok;
                }
            }
        }
    }
    perf_exit(OP_LMHEAD);
    if (unlikely(tprof_active))
        tprof_end();
    if (dbg_armed) {
        printf("DBG pos=%d l=-1 argmax e=0 %08lx\n", n_pos - 1, (unsigned long)best_tok);
        dbg_armed--;
    }
    return best_tok;
}

/* Init */
/*------*/

int infer_init(void)
{
    for (int i = 0; i < KVP_SLOTS; i++)
        kvp_scale_e8[i] = -1000; // New image: restage every KV scale beat.
    H = model_header();
    if (H->magic != BONS_MAGIC)
        return -1;
    if (H->version < BONS_VERSION_MIN || H->version > BONS_VERSION) { // Never misread a layout.
        printf("infer: BONS version %lu, this firmware reads version %u\n",
               (unsigned long)H->version, BONS_VERSION);
        return -2;
    }
#ifdef TRIXIUM_N_CHANNELS
    if (H->n_channels != TRIXIUM_N_CHANNELS) {
        printf("infer: image channels (%lu) != engine channels (%u)\n",
               (unsigned long)H->n_channels, TRIXIUM_N_CHANNELS);
        return -3;
    }
#endif
#if TRIXIUM_PROGS_VREGS >= 16
    // The x2-unrolled programs walk two beats per iteration and have no odd tail:
    // silu_q8 needs an even ffn_dim/16; rmsnorm_h and kvpack_h an even head_dim/16;
    // rope_h an even head_dim/32 (its loop is the half-span).
    if (H->ffn_dim % 32 || H->head_dim % 64) {
        printf("infer: ffn_dim %lu / head_dim %lu do not meet the ISA rev 2 microcode "
               "contracts (multiples of 32 / 64)\n",
               (unsigned long)H->ffn_dim, (unsigned long)H->head_dim);
        return -4;
    }
#endif
    n_ch      = H->n_channels;
    n_layer   = H->n_layer;
    n_embd    = H->n_embd;
    n_head    = H->n_head;
    n_kv_head = H->n_kv_head;
    head_dim  = H->head_dim;
    ffn_dim   = H->ffn_dim;

    heap = (uint8_t *)HEAP_BASE;
    // The integer attention's per-head fold shifts by log2(n_rep); the hybrid CPU path takes
    // any ratio (the 27B: 24/4 = 6).
    if (!hybrid_present()) {
        int nrep = n_head/n_kv_head, sh = 0;
        while ((1 << sh) < nrep) sh++;
        if ((1 << sh) != nrep) {
            printf("infer: n_head/n_kv_head must be a power of two\n");
            return -5;
        }
        fold_nrep_sh = sh;
    }
    qkn_resident = 0; // New image / re-init: the resident qk-norm rows must be refetched.
    {
        int q_bytes = 2*n_head*head_dim, kn_bytes = 2*n_layer*head_dim;
        if (q_bytes <= 0x1800) {                       // 1.7B: the tail of the q region
            sp_scb2 = (uint32_t)SP_SC + 0x3800u;
            sc_b_cap_bytes = (uint32_t)SP_K - sp_scb2;
        } else if (2*n_embd <= 0x2000 && kn_bytes > 0x1C00) { // 8B: SP_D's free half
            sp_scb2 = (uint32_t)SP_D + 0x2000u;
            sc_b_cap_bytes = (uint32_t)SP_FET - sp_scb2;
        } else {
            sp_scb2 = 0;
            sc_b_cap_bytes = 0;
        }
    }
    norm16_cache_build();
    tprof_setbuf(halloc(6000*16), 6000); // 96 KB trace buffer (main_ram, not sram .bss).
    {
        // kv_scores_br's R: depends only on head_dim.
        uint64_t bound = (uint64_t)(head_dim/32)*32u*127u*127u*1024u << 12;
        int r = 0;
        while ((bound >> r) >= (1ull << 31)) r++;
        kv_br_r = r;
        for (int i = 0; i < 128; i++) eps_memo[i] = 0; // New image: new rms_eps.
    }
    int kv_dim = n_kv_head*head_dim;
    int q_dim  = n_head*head_dim;

    if (hybrid_present()) {
        // The hybrid models run on their own CPU path (hybrid.c) - none of the Qwen3
        // lists, caches or resident regions below apply. The embedding comes from the
        // restored emb16 rows (embed_row_fetch).
        emb16_ent  = model_entry(ROLE_TOK_EMB16, 0);
        emb16x_ent = model_entry(ROLE_TOK_EMB16_EXP, 0);
        max_ctx = H->max_ctx;
        n_pos = 0;
        heap_hi = 0;
        int rc = hybrid_init();
        // The hybrid allocator may halve the requested context to fit main_ram. The
        // request/generation bounds must use that allocation too, otherwise forward()
        // hits its smaller cap and its -1 sentinel can be emitted as a token.
        if (rc == 0) max_ctx = hybrid_context_capacity();
        return rc;
    }

    // Norms: fetch each F32 tensor (replicated; channel 0 copy) to the CPU.
    attn_norm = halloc(n_layer*sizeof(float *));
    ffn_norm  = halloc(n_layer*sizeof(float *));
    q_norm    = halloc(n_layer*sizeof(float *));
    k_norm    = halloc(n_layer*sizeof(float *));

    for (int l = 0; l < n_layer; l++) {
        struct { int role; float **arr; int n; } norms[] = {
            { ROLE_ATTN_NORM,   &attn_norm[l], n_embd },
            { ROLE_FFN_NORM,    &ffn_norm[l],  n_embd },
            { ROLE_ATTN_Q_NORM, &q_norm[l],    head_dim },
            { ROLE_ATTN_K_NORM, &k_norm[l],    head_dim },
        };
        for (int i = 0; i < 4; i++) {
            const struct bons_entry *e = model_entry(norms[i].role, l);
            if (!e) return -2;
            float *dst = halloc(norms[i].n*sizeof(float));
            engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32,
                         e->chan_len_beats, SP_FET);
            volatile float *p = (volatile float *)(SCRATCHPAD_BASE + SP_FET);
            for (int j = 0; j < norms[i].n; j++) dst[j] = p[j];
            *norms[i].arr = dst;
        }
    }
    const struct bons_entry *e = model_entry(ROLE_OUT_NORM, 0);
    out_norm = halloc(n_embd*sizeof(float));
    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32, e->chan_len_beats, SP_FET);
    volatile float *p = (volatile float *)(SCRATCHPAD_BASE + SP_FET);
    for (int j = 0; j < n_embd; j++) out_norm[j] = p[j];

    // Activations.
    x     = halloc(n_embd*sizeof(float));
    h     = halloc(n_embd*sizeof(float));
    q     = halloc(q_dim*sizeof(float));
    k     = halloc(kv_dim*sizeof(float));
    v     = halloc(kv_dim*sizeof(float));
    attn  = halloc(q_dim*sizeof(float));
    gate  = halloc(ffn_dim*sizeof(float));
    up    = halloc(ffn_dim*sizeof(float));

    // KV caches: size to the header ctx (tiny in sim; on hardware capped by the converter).
    max_ctx = H->max_ctx;
    // The CPU-side KV cache is the debug fallback when HW attention exists (the fast path
    // keeps its own KV in HBM at full context); cap it so it fits the high heap window
    // above engine channel 3's weight tail. Without HW attention it is the primary cache
    // and keeps the full context.
    cpu_kv_ctx = (max_ctx < CPU_KV_CTX) ? max_ctx : CPU_KV_CTX;
    // KV region: above the per-channel weight payload, 4 KB aligned.
    kv_base   = (uint32_t)((TRIXIUM_CHAN_WEIGHT_BASE + H->payload_beats*32 + 0xFFF) & ~0xFFFull);
    kv_rb     = 1 + head_dim/32; // One scale beat + code beats (cols = head_dim, Q8).
    kv_region = (uint32_t)((max_ctx + n_ch - 1)/n_ch)*kv_rb;
    // A channel window that lies inside main_ram (the FK33's channel 3, in PC 4): its
    // payload + KV tail decides where the high heap can start, and the image is refused only
    // when that tail leaves main_ram. On a board whose windows sit above main_ram (the XCU1525) no
    // channel qualifies and the high heap keeps its fixed base.
    {
        uint64_t top = 0;   // Engine-side.
        const uint64_t ram_lo = TRIXIUM_MEM_CPU_OFFSET, ram_hi = ram_lo + MAIN_RAM_SIZE;
        for (uint32_t c = 0; c < (uint32_t)n_ch; c++) {
            if (!chan_on_cpu_memory(c)) continue;
            uint64_t win  = chan_window(c);
            uint64_t tail = win + H->payload_beats*32
                          + (uint64_t)n_layer*n_kv_head*2*kv_region*32;
            if (win < ram_hi && tail > ram_lo) {
                if (tail > ram_hi) {
                    printf("infer: channel %lu's weights+KV reach %08lx, past the end of "
                           "main_ram\n", (unsigned long)c, (unsigned long)cpu_addr_of(tail));
                    return -4;
                }
                if (tail > top) top = tail;
            }
        }
        heap_hi_base = top ? (uint32_t)((cpu_addr_of(top) + 0xFFFFFu) & ~0xFFFFFu) : 0;
        if (heap_hi_base < 0x40000000u + 0x09000000u)
            heap_hi_base = 0x40000000u + 0x09000000u;   // floor: main_ram + 144 MB
        heap_hi = (MAIN_RAM_SIZE >= 0x0A000000) ? (uint8_t *)heap_hi_base : 0;
        // The CPU-side KV cache (the debug fallback) must fit between the high heap's base
        // and the end of main_ram: halve its depth until it does.
        if (heap_hi) {
            uint64_t room = 0x40000000ull + MAIN_RAM_SIZE - heap_hi_base - (2u << 20);
            while (cpu_kv_ctx > 64
                   && (uint64_t)n_layer*cpu_kv_ctx*(2ull*kv_dim + 2ull*n_kv_head*4) > room)
                cpu_kv_ctx /= 2;
        }
    }
    kv_ekmax = halloc(n_layer*n_kv_head*2);
    for (int i = 0; i < n_layer*n_kv_head*2; i++) kv_ekmax[i] = -128;
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
    infer_qkvx_report();
    k_cache = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*kv_dim);
    v_cache = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*kv_dim);
    k_scale = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*n_kv_head*sizeof(float));
    v_scale = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*n_kv_head*sizeof(float));
    scores  = halloc(max_ctx*sizeof(float));
    n_pos   = 0;
    eps_init();
    rope_init();

    // 16 KV staging slots fill SP_KVP exactly (16 x 512 B, ending at SP_SC). A model with
    // more KV heads would run the last slots into the score buffer and corrupt attention with
    // no other symptom, so refuse rather than discover it as wrong tokens.
    if (2*n_kv_head > KVP_SLOTS) {
        printf("infer: %d KV heads needs %d staging slots, SP_KVP holds %d\n",
               n_kv_head, 2*n_kv_head, KVP_SLOTS);
        return -1;
    }

    // The low heap lives BELOW the lowest engine window inside main_ram (on the FK33
    // channel 3's, at +64 MB: HBM_WEIGHT_BASE is port-matched, see layout.py). Overrunning
    // it would scribble on that channel's weights and show up as wrong tokens with
    // everything else looking healthy. Check it, loudly, once per init; the numbers are cheap
    // to print and worth having in every boot log. A board whose windows sit above main_ram
    // has no such limit.
    {
        uint32_t lo_used = (uint32_t)(heap - (uint8_t *)HEAP_BASE);
        uint64_t lo_win  = 0;   // Engine-side start of the lowest window inside main_ram.
        for (uint32_t c = 0; c < (uint32_t)n_ch; c++) {
            uint64_t win = chan_window(c);
            if (chan_on_cpu_memory(c) && win >= TRIXIUM_MEM_CPU_OFFSET
                && win < TRIXIUM_MEM_CPU_OFFSET + MAIN_RAM_SIZE && (!lo_win || win < lo_win))
                lo_win = win;
        }
        uint32_t lo_room = lo_win ? cpu_addr_of(lo_win) - (uint32_t)HEAP_BASE : 0xFFFFFFFFu;
        printf("infer: low heap %lu KB", (unsigned long)(lo_used >> 10));
        if (heap_hi)
            printf(", high heap %lu KB @%08lx (CPU KV fallback %d positions)",
                   (unsigned long)((uint32_t)(heap_hi - (uint8_t *)heap_hi_base) >> 10),
                   (unsigned long)heap_hi_base, cpu_kv_ctx);
        printf("\n");
        if (lo_win && lo_used > lo_room) {
            printf("infer: LOW HEAP OVERRUNS the engine window at %08lx "
                   "(%lu KB used, %lu KB available) - tokens will be wrong\n",
                   (unsigned long)cpu_addr_of(lo_win),
                   (unsigned long)(lo_used >> 10), (unsigned long)(lo_room >> 10));
            return -1;
        }
    }
    return 0;
}

/* Generation */
/*------------*/

static int gen_stop[INFER_STOP_MAX], gen_nstop;

void infer_set_stop(const int *ids, int n)
{
    gen_nstop = n < INFER_STOP_MAX ? n : INFER_STOP_MAX;
    for (int i = 0; i < gen_nstop; i++) gen_stop[i] = ids[i];
}

static int gen_is_stop(int tok)
{
    for (int i = 0; i < gen_nstop; i++)
        if (gen_stop[i] == tok) return 1;
    return 0;
}

static uint32_t gen_serial;   // Bumped per generation: a continuation must follow its own.
static struct infer_prefill_profile last_prefill;

int infer_kv_resident(void)
{
    return n_pos;
}

uint32_t infer_gen_serial(void)
{
    return gen_serial;
}

int infer_context_capacity(void)
{
    return max_ctx;
}

const struct infer_prefill_profile *infer_last_prefill(void)
{
    return &last_prefill;
}

int infer_generate(const int *prompt, int n_prompt, int max_new, void (*emit)(int tok))
{
    return infer_generate_from(prompt, n_prompt, 0, max_new, emit);
}

// buf[0..resident) are already in the KV cache from the previous call (n_pos == resident:
// every forwarded token, i.e. all but the last one emitted), buf[resident..n) are new.
// resident == 0 is a fresh generation. Byte-identical to re-forwarding the whole buffer
// from position 0: the rows and the kv_ekmax ratchet evolve in the same order either way.
// The chat clients keep the conversation on the card (tokclr/tok/genb), so without this
// every turn would re-run the whole history and the first-token latency would grow with it.
int infer_generate_from(const int *buf, int n, int resident, int max_new, void (*emit)(int tok))
{
    memset(&last_prefill, 0, sizeof(last_prefill));
    // An id past the vocabulary makes the embedding fetch read a row that does not exist -
    // an engine stall only reconfiguring the FPGA clears. Refuse the request instead; the
    // console commands report it as GEN ERR.
    if (!H || H->magic != BONS_MAGIC || n_layer == 0) {
        printf(" [no model: hdrload first]");
        return -1;
    }
    for (int i = 0; i < n; i++)
        if (buf[i] < 0 || (uint32_t)buf[i] >= H->vocab) {
            printf(" [bad token id %d at %d, vocab %lu]", buf[i], i, (unsigned long)H->vocab);
            return -1;
        }
    if (n > max_ctx) { // KV rows live at [0, max_ctx); the generation loop stops there too.
        printf(" [%d tokens exceed the KV budget of %d - tokclr]", n, max_ctx);
        return -1;
    }
    if (resident <= 0 || resident > n || resident != n_pos)
        resident = 0;
    gen_serial++;
    uint64_t reset_start = perf_now64();
    if (resident == 0) {
        n_pos = 0; // Reset the KV cache (fresh generation).
        if (hybrid_present()) hybrid_reset();
        // ...and its exponent ratchet. kv_ekmax is the running max over every row ever
        // stored in a region and picks the KV scale bias (B = 22 - ekmax), so without this
        // a generation would inherit the previous request's maxima and its arithmetic would
        // depend on request history. Rows are overwritten from position 0 anyway.
        if (kv_ekmax)
            for (int i = 0; i < n_layer*n_kv_head*2; i++) kv_ekmax[i] = -128;
    }
    last_prefill.reset_cycles = perf_now64() - reset_start;
    if (resident == n) {
        printf(" [nothing new to forward]");
        return -1;
    }
    int tok = -1;
    uint32_t prefill_f0 = infer_forwards;
    uint64_t prefill_start = perf_now64();
    for (int i = resident; i < n; i++) {
        tok = forward(buf[i], i == n - 1);
        if (engine_wedged) { printf(" [engine wedged during the prompt]"); return -1; }
    }
    last_prefill.prefill_cycles = perf_now64() - prefill_start;
    last_prefill.tokens = n - resident;
    last_prefill.forwards = infer_forwards - prefill_f0;
    last_prefill.valid = 1;
    int count = 0;
    for (int i = 0; i < max_new; i++) {
        if (n_pos >= max_ctx)
            break;
        if (engine_wedged) { printf(" [engine wedged]"); return -1; } // the token is garbage
        emit(tok);
        count++;
        if (gen_is_stop(tok))
            break;
        // The forward after the LAST emitted token computes a successor nobody emits and
        // stores a KV row nobody reads (a continuation forwards that token itself), so skip
        // it.
        if (i + 1 < max_new)
            tok = forward(tok, 1);
    }
    return count;
}

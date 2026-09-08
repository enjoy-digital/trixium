// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Qwen3 token loop, phase 1: the engine does every matvec (ternary weights streamed from
// HBM); the CPU does the vector ops (RMSNorm, RoPE with YaRN, softmax, SiLU) and attention
// over CPU-side K/V caches in soft-float. Slow (seconds/token) but exact enough for tokens -
// the hardware vector unit + engine-side attention are the phase-2 speedup.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>
#include <generated/mem.h>

#include "bons.h"
#include "engine.h"
#include "model.h"
#include "infer.h"
#include "ops.h"
#include "perf.h"
#include "vec.h"
#include "seq.h"

// newlib-libm compatibility: the toolchain libm.a references __errno, absent from the
// picolibc environment LiteX links against.
static int __errno_storage;
int *__errno(void) { return &__errno_storage; }

// Scratchpad layout (byte offsets) - resident-BFP regions (int16 mantissas + firmware-
// tracked exponents; sized for the 8B: E=4096, F=12288).
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
// Dual-Q (n_rep == 2) attention map: scores A at SP_Y, scores B at SP_Y + 8 KB (qstride
// 0x800 words, bounds ctx at 2048); AXPY partials in SP_YU (A at 0, B at n_ch*hd words);
// spill tables above them. All regions are dead outside the attention block.
#define SP_SCB   (SP_Y + 0x2000)
#define SP_AW2   SP_YU
#define SP_A2P8  (SP_YU + 0x4000) // Per-head p8-sum spills (32 B each, indexed by head).
#define SP_A2AM  (SP_YU + 0x4800) // Per-head absmax spills.
#define SP_Q    0x38000 // Resident q int16 (q_dim <= 4096: 8 KB).
#define SP_K    0x3A000 // Resident k int16 (kv_dim <= 1024: 2 KB).
#define SP_V    0x3B000 // Resident v int16.
#define SP_TAB  0x3B800 // Rope table row for the current position ([c14 | s14]).
#define SP_FFSP 0x1D000 // FFN seq spill beats (gate absmax, up absmax): the 0x1D000..0x1E000
                        // gap between SP_H (ends 0x1C000 + 2*n_embd <= 0x1D000) and SP_XR.
#define SP_FET2 0x3C000 // Second norm16 staging slot (k_norm while q_norm sits in SP_FET).
// Resident qk-norm weight rows (qres, 2026-09-03): head_dim int16 = 256 B per layer per
// kind, 28 layers = 7 KB each, in the tails of the SP_FET and SP_D regions (their live
// occupants are <= 4 KB). Fetched once; the qkr list's rmsnorm_h A1 is patched per layer
// instead of two 8-beat engine fetches paid at ~1.5k cycles of op latency each, x56/token.
#define SP_QNW  (SP_FET + 0x1000)
#define SP_KNW  (SP_D + 0x1000)

// Fixed exponent for CPU-produced normed vectors (values bounded ~16 by construction).
#define E_NORM  (-11)

// Heap: after the BONS header/table area. The CPU's RAM is HBM pseudo-channel 4 and engine
// channel 3's weight window overlaps it at [64 MB, ~141 MB) (see layout.py HBM_WEIGHT_BASE):
// the low heap must stay under 64 MB, and anything big goes to the HIGH heap above the
// weights+KV tail at +144 MB. The low heap holds a few MB of norms and tables; the only big
// consumer is the CPU-side KV cache, which is a debug fallback (HW attention keeps its own
// KV in HBM at full context) and is capped at CPU_KV_CTX positions to fit.
#define HEAP_BASE    (BONSAI_HDR_ADDR + 0x100000)
#define HEAP_HI_BASE (0x40000000u + 0x09000000u) // Main_ram + 144 MB: above ch3's tail.
#define CPU_KV_CTX   512

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

// Model state --------------------------------------------------------------------------------

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

// Helpers ------------------------------------------------------------------------------------

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

static void rmsnorm_f(float *dst, const float *src, const float *w, int n, float eps)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += src[i]*src[i];
    float r = 1.0f/sqrtf(ss/n + eps);
    for (int i = 0; i < n; i++) dst[i] = src[i]*r*w[i];
}

// Quantize a float vector to int8 with a power-of-two scale 2^e8 (so downstream exponent
// algebra stays integer). Returns e8.
static int quantize_f_pow2(const float *src, int n, uint32_t sp_off)
{
    perf_enter(OP_QUANT8);
    float amax = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    int e8 = 0;
    if (amax > 0) {
        frexpf(amax, &e8);
        e8 -= 7; // mantissas in [2^6, 2^7).
    }
    float is = ldexpf(1.0f, -e8);
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_off);
    for (int i = 0; i < n; i += 4) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++) {
            int q8 = (int)lroundf(src[i+b]*is);
            if (q8 >  127) q8 =  127;
            if (q8 < -127) q8 = -127;
            w |= ((uint32_t)(uint8_t)(int8_t)q8) << (8*b);
        }
        sp[i/4] = w;
    }
    perf_exit(OP_QUANT8);
    return e8;
}

// KV-to-HBM (M5): per (layer, kv-head, K|V) region of Q8 row packets, position-striped
// (position t -> channel t % n_ch, local row t / n_ch). All groups of a row share one
// power-of-two scale 2^ek, stored as an exactly-representable fp16 ((ek+15)<<10), so the
// engine's Q8 scale path applies it bit-exactly and the exponent algebra stays integer.
#ifdef BONSAI_HAS_KVSTORE
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
static void kv_store_row(int l, int kvh, int kind, int t, const float *vec)
{
    float amax = 0;
    for (int i = 0; i < head_dim; i++) {
        float a = fabsf(vec[i]);
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
            int q8 = (int)lroundf(vec[i+b]*is);
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

// AXPY pre-shift keeping the int32 lane accumulators safe (bitexact.axpy_rpre):
// pure bit-length form so the descriptor sequencer can compute it in hardware.
// F1-5 (2026-08-29): the per-head attention fold in integers, emulating the float32 path
// bit-for-bit so tokens stay byte-identical. The float path (kept below `ifold_use`) is
// soft-float on VexiiRiscv - a divide, ldexpf, a multiply and frexpf per head, then
// lroundf(ldexpf()) per head for the scale8 multiplier: ~1.4k cycles per head, 448 per token.
//   q  = RN24(1.0f / (float)P8)            -> (qm, qe): q = qm * 2^qe, qm in [2^23, 2^24)
//   c2 = ldexpf(q, E1)                     -> (qm, qe + E1), no rounding
//   hm = (float)(am >> k2) * c2            -> RN24 of the product: (m2, e2)
//   ea8 = frexpf(max hm) - 7               -> e2max + 24 - 7
//   Mh  = lroundf(ldexpf(c2, 15 - ea8))    -> qm * 2^(qe + E1 + 15 - ea8), half away from 0
int ifold_use = 1; // Second version (2026-08-30): 32-bit hardware divides + clz.
static inline __attribute__((always_inline)) int bitlen32(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
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
    // 32-bit throughout (2026-09-03): the 64-bit variable shifts this used to do are
    // libgcc calls on RV32 and were most of the fold's 640 cycles. a <= 2^14 and
    // qm < 2^24 give p < 2^38 and sh in 0..14, so m = p >> sh is (pl >> sh) | (ph <<
    // (32-sh)) exactly (the bits of ph that would overflow are zero because p < 2^L),
    // and the remainder/half live entirely in pl. Proven bit-identical to the 64-bit
    // form over 20M random inputs plus the edges.
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

static int axpy_rpre_c(uint64_t p8sum)
{
    // Bit length via clz - the 64-bit shift loop it replaces cost ~300-450 cycles per
    // fold on RV32 (etrace, 2026-09-02). Identical result for every input.
    uint32_t hi = (uint32_t)(p8sum >> 32), lo = (uint32_t)p8sum;
    int bl = hi ? 32 + bitlen32(hi) : bitlen32(lo);
    return bl + 5;
}

// B/R for a scores op over a region: pow-2 scales make e16 + B == 12 exactly
// (fp16_decompose: 2^ek = 1024 * 2^(ek-10); B = 12 - (ekmax - 10) = 22 - ekmax), so the
// worst-row acc bound is closed-form: groups * (32*127*127) * 1024 << 12.
static int kv_br_r = -1; // Model constant (head_dim only); computed once at infer_init.
                         // Recomputing it here was 13 iterations of 64-bit shifts per
                         // call x 32 calls/layer - ~7k cycles/layer of pure waste
                         // (found by qtrace, 2026-09-02).

static void kv_scores_br(int l, int kvh, int kind, int *B, int *R)
{
    *B = 22 - kv_ekmax[kv_ek_idx(l, kvh, kind)];
    *R = kv_br_r;
}
#endif

// Run an engine matvec for tensor e on rows [row0, row0+rows) per channel; results (float,
// scaled by xscale) into dst[gathered global order: chan*rows + row].
static void matvec_rows_f(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                          float xscale, float *dst)
{
    uint32_t rb = bons_row_beats(e->cols, e->kind, e->group);
    struct engine_op op = {0};
    op.kind   = (e->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
    op.rows   = rows;
    op.cols   = e->cols;
    op.wbase  = BONSAI_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)row0*rb)*32;
    op.wlen   = rows*rb;
    op.sbeats = bons_scale_beats(e->cols, e->group);
    op.cbeats = bons_code_beats(e->cols, e->kind);
    op.bias   = e->bias;
    op.rshift = e->rshift;
    op.xsrc   = SP_X;
    op.dst    = SP_Y;
    engine_run(&op);

    perf_enter(OP_NORM_I32);
    float sc = ldexpf(xscale, (int)e->rshift - (int)e->bias);
    volatile int32_t *y = (volatile int32_t *)(SCRATCHPAD_BASE + SP_Y);
    for (uint32_t i = 0; i < rows*n_ch; i++)
        dst[i] = (float)y[i]*sc;
    perf_exit(OP_NORM_I32);
}

// Full matvec (all rows), output in original row order (block shard: chan-major == row order).
static void matvec_f(const struct bons_entry *e, const float *src, float *dst)
{
    float s = quantize_to_sp(src, e->cols, SP_X);
    matvec_rows_f(e, 0, e->rows/n_ch, s, dst);
}

// Embedding row: fetch the row packet from HBM and dequantize (ternary).
static void embed_row_f(const struct bons_entry *e, int row, float *dst)
{
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch; // Rows per channel.
    uint32_t ch  = row/rpc;
    uint32_t loc = row % rpc;
    engine_fetch(ch, BONSAI_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)loc*rb)*32,
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

static void rope_apply(float *vec, int n_heads_v, int pos)
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

#ifdef CSR_VECUNIT_BASE
#include "progs.h"

// Score padding granule: softmax8's x2 unroll (ISA rev 2 microcode) walks two 16-entry
// blocks per iteration and has no odd-tail form, so scores are padded to a 32-multiple
// with INT32_MIN there (those entries emit p8 = 0 - the sum and the AXPY do not change).
#if BONSAI_PROGS_VREGS >= 16
#define SC_PAD 32
#else
#define SC_PAD 16
#endif
#define SC_PADN(cnt) ((SC_PAD - ((cnt) & (SC_PAD - 1))) & (SC_PAD - 1))
#endif

// Resident-BFP op layer: every op has a CPU implementation (float math over the int16
// scratchpad regions - the always-available fallback) and, when the vector unit exists,
// a HW implementation running the corresponding microcode program. Exponent algebra stays
// in this file; ops receive shift amounts only.

static inline volatile int16_t *sp16(uint32_t off) { return (volatile int16_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int32_t *sp32(uint32_t off) { return (volatile int32_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int8_t  *sp8(uint32_t off)  { return (volatile int8_t  *)(SCRATCHPAD_BASE + off); }

// rmsnorm: SP int16(src,esrc) -> SP int16(dst) at fixed E_NORM. HW path: fetch the
// image's int16 norm rows (kind norm16, exponent E_W16) into SP_FET and run the vector
// unit's two-pass program; CPU path: soft-float with the F32 weights from RAM.
#define E_W16 (-13) // layout.NORM16_EXP.

// rms_eps as an exact integer mantissa/exponent, so the per-call eps does not cost soft-float.
//
// The value wanted is lroundf(rms_eps * 2^-2esrc) - a pure power-of-two scaling, so float32
// computes it with no rounding beyond rms_eps's own representation. Decomposing once into a
// 24-bit mantissa (all a float32 has) makes the per-call work a shift and a rounding add, and
// bit-identical to what the float path produced. vec_rmsnorm_head runs per head, ~700 times a
// forward, and ldexpf+lroundf are hundreds of cycles each on a core with no FPU.
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
    if (s <= 0)                                  // float path overflowed int here too
        return 0x7FFFFFFF;                       // (verified bit-exact for every s > 0)
    if (s > 31)
        return 1;                                // rounds to 0; callers clamp to >= 1 anyway
    uint32_t v = (eps_m + (1u << (s - 1))) >> s; // round half away from zero (values > 0)
    return (int)(v < 1 ? 1 : v);
}
static void op_rmsnorm_sp(uint32_t src, int esrc, const float *w,
                              const struct bons_entry *w16, int n, uint32_t dst)
{
#if defined(CSR_VECUNIT_BASE) && defined(BONSAI_HAS_VECUNIT2)
    if (op_on_hw(OP_RMSNORM) && w16) {
        engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + w16->chan_base_beats*32,
                     w16->chan_len_beats, SP_FET);
        int eps_int = eps_for(esrc);
        int ln = 0;
        while ((1 << ln) < n) ln++;
        int32_t args[7] = {(int32_t)src, (int32_t)SP_FET, (int32_t)dst, n/16,
                           -ln, eps_int, E_W16 - E_NORM - 15};
        vec_run(PROG_RMSNORM, args, 7);
        return;
    }
#endif
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
// 2**11, so >>8 handed the matvec int8 inputs of about +-9 - 3 of the 7 bits available -
// and cost ~0.30 rel-rms on every matvec output. bit_length(amax) - 7 is clip-free by
// construction (amax >> s < 128) and recovers a full order of magnitude of accuracy.
// The program measures its own absmax (no producer has a spare scalar to return one in), so
// both paths here derive the same shift from the same absmax. s may be negative: a small
// source is scaled *up* into the int8 range.
static int quant8_shift(uint32_t amax)
{
    if (amax == 0) return 8; // VCLZN(0) == 0, i.e. pre-shift 0 - keep the paths identical.
    int b = 0;
    while ((amax >> b) != 0) b++; // bitlen.
    return b - 7;
}

// rmsnorm + measured quant8 in one program (F1-3, 2026-08-30): saves a dispatch and
// quant8's absmax pass at every norm->x8 site. Returns the quant8 shift like op_quant8_sp;
// falls back to the two ops when either is off the hardware or the NORM16 row is absent.
static int op_quant8_sp(uint32_t src, uint32_t dst, int n);
// fovl (2026-09-02): the norm-weight row fetch that opens every rmsnorm_q8 used to be a
// blocking engine op between two vector ops (tprof: 4k cycles x 56 per token of glue
// with both units idle). The caller submits it BEFORE the preceding resid_add so it
// streams underneath, joins, and flags it; the op then skips its own fetch.
int fovl_use = 1;
static int rq8_prefetched;
static uint32_t norm_fetch_submit(const struct bons_entry *w16)
{
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_FETCH;
    op.wbase = BONSAI_CHAN_WEIGHT_BASE + w16->chan_base_beats*32;
    op.wlen  = w16->chan_len_beats;
    op.dst   = SP_FET;
    return engine_submit(&op);
}
static int op_resid_add_sp(uint32_t a, int ea, uint32_t b, int eb, int n,
                           uint32_t bmax, uint32_t *amax_io);
// resid_add with the NEXT norm's weight row streaming underneath (fovl). SP_FET is
// idle here (its last reader, the previous rmsnorm or the qkr list, has finished) and
// the engine is idle (the preceding matvec/list joined) - same numerics either way.
static int resid_add_pf(uint32_t src, int esrc, uint32_t d, int ed, int n, uint32_t dmax,
                        uint32_t *xamax, const struct bons_entry *next_w16)
{
    uint32_t df = 0;
    int pf = fovl_use && next_w16 && op_on_hw(OP_RMSNORM) && op_on_hw(OP_QUANT8);
    if (pf) df = norm_fetch_submit(next_w16);
    perf_enter(OP_RESID_ADD);
    int e = op_resid_add_sp(src, esrc, d, ed, n, dmax, xamax);
    perf_exit(OP_RESID_ADD);
    if (pf) { engine_join(df); rq8_prefetched = 1; }
    return e;
}

static int op_rmsnorm_q8_sp(uint32_t src, int esrc, const float *w,
                            const struct bons_entry *w16, int n, uint32_t dst16,
                            uint32_t dst8)
{
#if defined(CSR_VECUNIT_BASE) && defined(BONSAI_HAS_VECUNIT2)
    if (op_on_hw(OP_RMSNORM) && op_on_hw(OP_QUANT8) && w16) {
        if (rq8_prefetched)
            rq8_prefetched = 0;                  // Row already in SP_FET (resid_add_pf).
        else
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + w16->chan_base_beats*32,
                         w16->chan_len_beats, SP_FET);
        int eps_int = eps_for(esrc);
        int ln = 0;
        while ((1 << ln) < n) ln++;
        int32_t args[8] = {(int32_t)src, (int32_t)SP_FET, (int32_t)dst16, n/16,
                           -ln, eps_int, E_W16 - E_NORM - 15, (int32_t)dst8};
        perf_enter(OP_RMSNORM);
        uint32_t amax = (uint32_t)vec_run(PROG_RMSNORM_Q8, args, 8);
        perf_exit(OP_RMSNORM);
        return quant8_shift(amax);
    }
#endif
    perf_enter(OP_RMSNORM);
    op_rmsnorm_sp(src, esrc, w, w16, n, dst16);
    perf_exit(OP_RMSNORM);
    perf_enter(OP_QUANT8);
    int s = op_quant8_sp(dst16, dst8, n);
    perf_exit(OP_QUANT8);
    return s;
}

// silu_mul + measured quant8 in one program (2026-08-30). Returns the quant8 shift.
static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n);
static int op_silu_q8_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n, uint32_t dst8)
{
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
        int32_t args[8] = {(int32_t)gsrc, (int32_t)usrc, (int32_t)gsrc, n/16,
                           eg + 6, 0, -15, (int32_t)dst8};
        perf_enter(OP_SILU_MUL);
        uint32_t amax = (uint32_t)vec_run(PROG_SILU_Q8, args, 8);
        perf_exit(OP_SILU_MUL);
        return quant8_shift(amax);
    }
#endif
    perf_enter(OP_SILU_MUL);
    op_silu_mul_sp(gsrc, eg, usrc, eu, n);
    perf_exit(OP_SILU_MUL);
    perf_enter(OP_QUANT8);
    int s = op_quant8_sp(gsrc, dst8, n);
    perf_exit(OP_QUANT8);
    return s;
}

static int op_quant8_sp(uint32_t src, uint32_t dst, int n)
{
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_QUANT8)) {
        int32_t args[4] = {(int32_t)src, 1, (int32_t)dst, n/16}; // A1 = 1: measure.
        return quant8_shift((uint32_t)vec_run(PROG_QUANT8, args, 4)); // S6 = source absmax.
    }
#endif
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
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_NORM_I32)) {
        int32_t args[5] = {(int32_t)src, 0, (int32_t)dst, n/16, k};
        vec_run(PROG_NORM_I32, args, 5);
        return k;
    }
#endif
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
#ifdef CSR_ENGINE_CSR_ABSMAX_ADDR
    uint32_t amax = engine_csr_absmax_read();
#else
    // Pre-absmax gateware (the archived variant-b bitstream): scan on the CPU.
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
static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n)
{
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_SILU_MUL)) {
        int32_t args[7] = {(int32_t)gsrc, (int32_t)usrc, (int32_t)gsrc, n/16,
                           eg + 6, 0, -15}; // A4 = direct shift (VSHLS convention 2026-08-28).
        vec_run(PROG_SILU_MUL, args, 7);
        return;
    }
#endif
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
// eo comes from the operands' *measured* magnitudes. The old eo = max(ea,eb)+1 is the bound
// for two full 15-bit mantissas, so it spent a mantissa bit on every call - and the loop
// calls this twice per layer. On the 1.7B the residual was down to a single bit by layer 11
// (every element 0 or +-2**ex); its own quantization error then grew 4x per layer, reaching
// 6e11 by layer 27 and turning the output into garbage. Sizing eo from the real absmax keeps
// ~14 bits at every layer. Shifts are signed (VSHLV: left when negative here).
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
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_RESID_ADD)) {
        int32_t args[6] = {(int32_t)a, (int32_t)b, (int32_t)a, n/16, -sa, -sb};
        *amax_io = (uint32_t)vec_run(PROG_RESID_ADD, args, 6); // S6 = output absmax.
        return eo;
    }
#endif
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

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
// Resident attention front (all attention hwops on): q/k/v live in the scratchpad as
// int16 from the matvec on; qk-norm/rope/kv-pack run on the vector unit; the CPU float
// K/V cache is not maintained (attention op bits are latched per generation - flipping
// them mid-generation is a debug-only move).
static int attn_all_hw(void)
{
    return op_on_hw(OP_QKNORM) && op_on_hw(OP_ROPE) && op_on_hw(OP_KVQUANT)
        && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV);
}

// vec rmsnorm of one head slice in place (w16 rows already staged at w_sp).
static void vec_rmsnorm_head(uint32_t sp, int es, uint32_t w_sp, int n)
{
    int eps_int = eps_for(es);
    int ln = 0;
    while ((1 << ln) < n) ln++;
    int32_t args[7] = {(int32_t)sp, (int32_t)w_sp, (int32_t)sp, n/16,
                       -ln, eps_int, E_W16 - E_NORM - 15};
    vec_run(PROG_RMSNORM, args, 7);
}

static void vec_rope_head(uint32_t sp, int hd)
{
    int32_t args[7] = {(int32_t)sp, (int32_t)(sp + hd), (int32_t)SP_TAB,
                       (int32_t)(SP_TAB + hd), (hd/2)/16,
                       (int32_t)sp, (int32_t)(sp + hd)};
    vec_run(PROG_ROPE, args, 7);
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
// Two staging slots (K and V) so the scale beat is written only when the exponent CHANGES,
// not per store: K's exponent is the constant E_NORM+8 and V's changes once per layer, but
// this used to rewrite 16 scratchpad halfwords (~1.1k cycles of uncached bus traffic) for
// every one of the 448 stores per forward.
// One staging slot per (kv_head, kind) - 16 x 512 B fills SP_KVP exactly. Two slots would
// do if pack and store alternated, but they must not: the vector unit and the engine share
// the scratchpad port, and interleaving them 32 times per layer measures 1.6x the cost of
// packing everything first and then storing (vbench kv_pair_alt 74,948 vs 46,167 phased).
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
    // cost here, not the arithmetic (F0-3, 2026-08-29).
    // Only the words that carry a scale change with the exponent - ceil(groups/2) = 2 of
    // 8 on the 1.7B (head_dim 128 -> 4 groups). The other six are constant zero and
    // nothing else ever writes the scale beat, so they are staged once per slot epoch (a
    // sentinel in kvp_scale_e8: init, vbench) and left alone after: 6 posted stores per
    // exponent change, V changes every layer (2026-09-07, from the external review).
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

// Pack phase: quantize the head into its own staging slot. No engine op, so the vector unit
// keeps the scratchpad for the whole run of heads.
static void kv_pack_row_res(int l, int kvh, int kind, uint32_t sp, int es)
{
    int e8 = es + 8;
    kv_store_scale_prep(kvh, kind, e8);
    vec_quant8_head(sp, kvp_slot(kvh, kind) + 32, head_dim);
    int idx = kv_ek_idx(l, kvh, kind);
    if (e8 > kv_ekmax[idx]) kv_ekmax[idx] = e8;
}

// Pack prep only: the qkr sequencer list carries the quant8 itself (seq_build_qkr, F0-4);
// the CPU side is the scale beat and the exponent ratchet - identical bookkeeping.
static void kv_pack_prep_res(int l, int kvh, int kind, int es)
{
    int e8 = es + 8;
    kv_store_scale_prep(kvh, kind, e8);
    int idx = kv_ek_idx(l, kvh, kind);
    if (e8 > kv_ekmax[idx]) kv_ekmax[idx] = e8;
}

// Store phase: stream the already-packed slots out. Engine only.
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

// Integer embedding: fetch the ternary row, dequantize with exact fp16 decomposition
// ((q-1)*m11 aligned to the smallest group exponent), then normalize to int16 BFP.
// Pure integer (~6 ops/element vs ~50 soft-float ones); exponent returned.
static int embed_row_bfp(const struct bons_entry *e, int row, uint32_t sp_dst)
{
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch;
    engine_fetch(row/rpc,
                 BONSAI_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)(row % rpc)*rb)*32,
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
    // per word: the byte/halfword version of these loops was ~85k cycles per token
    // for 2048 elements (2026-09-04). Same arithmetic, same bytes.
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

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
// Sequencer descriptor list for the per-head qk-norm + rope loops. Addresses are
// head-relative inside SP_Q/SP_K, so the list is layer-independent; per layer only the
// rmsnorm eps_int args (A5 = descriptor word 7) change. Built once at init.
static int seq_qkr_count;
// Descriptor indices of the q-norm and k-norm in the qkr list (their A1/A5 are patched
// per layer). Fixed at 0/1 until the reorder below moved the q chain behind the store.
static int seq_qkr_di_qn = 0, seq_qkr_di_kn = 1;
// qkr list order (2026-09-08): K chain -> packs -> multi-store (NOWAIT) -> Q chain -> Q
// quant8. The q-side work (rmsnorm_h/rope_h over 16 heads, ~3k cycles) is independent of
// K/V, so behind the store it absorbs the store's ~1k tail that the stop descriptor used
// to wait out (16 packets at ~65 cycles of write latency each, per-descriptor trace).
// Same ops, same numerics. Toggle qkro selects the old order for a same-boot A/B.
static int qkr_reorder = 1;
static int qfix_use = 1;     // Console toggle (qfix): scan-free fixed-shift Q quant8.
// In-list KV stores (probe-1 follow-up, 2026-08-31): each (kvh, kind) staging slot's
// engine STORE rides the qkr list as a NOWAIT descriptor right after the pack quant8
// that fills the slot - the next head's quant8 (vec) runs under the store's stream, so
// the ~9k cycles/layer of serial CPU-dispatched stores hide under vector work the layer
// does anyway. Needs the scale-beat prep to run BEFORE the list (the store streams the
// scale beat), which is equally correct in the CPU-store layout. kovl on|off rebuilds
// the four lists (the attention/FFN/scale8 bases shift with the 2*n_kv_head extras).
static int kv_pipe = 1;         // Requested (kovl console toggle).
static int kv_pipe_active = 0;  // What the built list uses.
// Emit the per-slot KV-pack quant8 descriptors that PROG_KVPACK_H makes redundant (see
// the loop below). Only useful to A/B the two list shapes in one boot; without LOOPS2
// there is no kvpack_h descriptor and they are the real thing, so it is forced on.
#ifdef BONSAI_VEC_LOOPS2
static int kvq_redundant = 0;
#else
static const int kvq_redundant = 1;
#endif
// One in-list MULTI-STORE for all 2*n_kv_head slots instead of 16 NOWAIT store ops
// (2026-09-08): the staging slots are 0x200 B apart and a layer's (kvh, kind) regions
// kv_region beats apart, so the engine's rep/dstride/sstride store (F1-1) scatters them
// all from one descriptor. 16 dispatches -> 1 on the list side, 16 per-layer address
// patches -> 1 on the CPU side - and the qkr window is the one where both sides are the
// long pole within ~850 cycles of each other (qtrace, v11). Needs the multi-store CSR.
#ifdef CSR_ENGINE_CSR_REP_ADDR
static int kv_mstore = 1;
#else
static const int kv_mstore = 0;
#endif
static int kv_mstore_active = 0; // What the built list uses.
static uint8_t seq_kvst_di[32]; // Descriptor index of the store for slot (kvh*2 + kind).
// What the store descriptors' constant words currently hold (~0 = unknown/rebuilt).
static uint32_t kvst_rb_set = 0xFFFFFFFFu, kvst_chan_set = 0xFFFFFFFFu;
static void seq_build_qkr(void)
{
    kvst_rb_set = kvst_chan_set = 0xFFFFFFFFu; // The store slots below are rewritten.
    int ln = 0;
    while ((1 << ln) < head_dim) ln++;
    uint32_t di = 0;
#ifdef BONSAI_VEC_LOOPS2
    // Whole-tensor per-head programs (2026-08-30): the 2*(n_head+n_kv_head) rmsnorm/rope
    // descriptors and the 2*n_kv_head KV-pack ones become one program each - the
    // per-descriptor fixed cost was 8-12x the work at V1 rates (qknorm 749k, kvquant 514k
    // cycles/token measured vs 87k + 44k attainable). Same instruction sequence per head,
    // so byte-identical by construction; only A5 (eps) of the two norms is patched per layer.
    int32_t nq[8] = {(int32_t)SP_Q, (int32_t)SP_FET, (int32_t)SP_Q, head_dim/16,
                     -ln, 1, E_W16 - E_NORM - 15, n_head};
    int32_t rq[8] = {(int32_t)SP_Q, (int32_t)(SP_Q + head_dim), (int32_t)SP_TAB,
                     (int32_t)(SP_TAB + head_dim), (head_dim/2)/16, (int32_t)SP_Q,
                     head_dim, n_head};
    {
        if (!qkr_reorder) {
            seq_qkr_di_qn = (int)di;
            seq_desc_vec(di++, PROG_RMSNORM_H, nq, 8, 0);              // q-norm (A5 patched)
        }
        int32_t nk[8] = {(int32_t)SP_K, (int32_t)SP_FET2, (int32_t)SP_K, head_dim/16,
                         -ln, 1, E_W16 - E_NORM - 15, n_kv_head};
        seq_qkr_di_kn = (int)di;
        seq_desc_vec(di++, PROG_RMSNORM_H, nk, 8, 0);                  // k-norm (A5 patched)
        if (!qkr_reorder)
            seq_desc_vec(di++, PROG_ROPE_H, rq, 8, 0);                 // q-rope
        int32_t rk[8] = {(int32_t)SP_K, (int32_t)(SP_K + head_dim), (int32_t)SP_TAB,
                         (int32_t)(SP_TAB + head_dim), (head_dim/2)/16, (int32_t)SP_K,
                         head_dim, n_kv_head};
        seq_desc_vec(di++, PROG_ROPE_H, rk, 8, 0);                     // k-rope
        int32_t pk[8] = {(int32_t)SP_K, 0, (int32_t)(kvp_slot(0, 0) + 32), head_dim/16, 0, 0,
                         0x400 - 16*(head_dim/16), n_kv_head};
        seq_desc_vec(di++, PROG_KVPACK_H, pk, 8, 0);                   // 4: K packs
        int32_t pv[8] = {(int32_t)SP_V, 0, (int32_t)(kvp_slot(0, 1) + 32), head_dim/16, 0, 0,
                         0x400 - 16*(head_dim/16), n_kv_head};
        seq_desc_vec(di++, PROG_KVPACK_H, pv, 8, 0);                   // 5: V packs
    }
    if (0)
#endif
    for (int hh = 0; hh < n_head + n_kv_head; hh++) {
        int      is_q = hh < n_head;
        uint32_t sp   = is_q ? (SP_Q + hh*head_dim*2)
                             : (SP_K + (hh - n_head)*head_dim*2);
        uint32_t wsp  = is_q ? SP_FET : SP_FET2;
        int32_t rargs[7] = {(int32_t)sp, (int32_t)wsp, (int32_t)sp, head_dim/16,
                            -ln, 1, E_W16 - E_NORM - 15};
        seq_desc_vec(di++, PROG_RMSNORM, rargs, 7, 0);
        int32_t pargs[7] = {(int32_t)sp, (int32_t)(sp + head_dim), (int32_t)SP_TAB,
                            (int32_t)(SP_TAB + head_dim), (head_dim/2)/16,
                            (int32_t)sp, (int32_t)(sp + head_dim)};
        seq_desc_vec(di++, PROG_ROPE, pargs, 7, 0);
    }
    // F0-4 (2026-08-29): the 2*n_kv_head fixed-shift KV-pack quant8s ride the same list.
    // K's pack follows K's rope in list order; V is resident before the list runs. The
    // CPU keeps only the scale-beat prep and the kv_ekmax ratchet (kv_pack_prep_res).
    // An in-list descriptor measures ~50 cycles against ~420 for a CPU vec_run.
    // The per-slot quant8s above were meant to disappear under BONSAI_VEC_LOOPS2 - the
    // whole-tensor PROG_KVPACK_H descriptors 4/5 already pack every kv head - but the
    // `if (0)` that should have guarded this loop landed on the `kv_pipe_active = 0`
    // statement that was later inserted between it and the loop, so on v8 all 16 ran and
    // redid work descriptor 4/5 had just done (2026-09-06). Proven redundant in the
    // interpreter: kvpack_h over n_kv_head heads and the 8 per-slot quant8s leave a
    // byte-identical scratchpad. The ENGINE STORES in this loop are NOT redundant, which
    // is why the loop still runs and only the quant8 is skipped.
    // Note kv_pipe_active must be reset unconditionally: leaving it stale is what made
    // `kovl off` patch descriptors that had become the attention list's.
    kv_pipe_active = 0;
    for (int kvh = 0; kvh < n_kv_head; kvh++)
        for (int kind = 0; kind < 2; kind++) {
            uint32_t src = (kind ? SP_V : SP_K) + (uint32_t)kvh*head_dim*2;
            int32_t qargs[4] = {(int32_t)src, 0, (int32_t)(kvp_slot(kvh, kind) + 32),
                                head_dim/16};                       // A1 = 0: fixed >>8.
            if (kvq_redundant)
                seq_desc_vec(di++, PROG_QUANT8, qargs, 4, 0);
            if (kv_pipe && !kv_mstore && 2*n_kv_head <= 32) {
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
    if (kv_pipe && kv_mstore && 2*n_kv_head <= 31) {
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
#ifdef BONSAI_VEC_LOOPS2
    if (qkr_reorder) { // Q chain behind the (NOWAIT) store: absorbs its write-latency tail.
        seq_qkr_di_qn = (int)di;
        seq_desc_vec(di++, PROG_RMSNORM_H, nq, 8, 0);                  // q-norm (A5 patched)
        seq_desc_vec(di++, PROG_ROPE_H, rq, 8, 0);                     // q-rope
    }
#endif
    // F0-5: ONE fixed-shift quant8 over all of resident q -> SP_X + 0x1000 (STB
    // post-increments 16 B, so head j lands at +j*head_dim); the attention list's
    // per-head quant8 descriptor is gone and its scores read x from there. The region
    // is dead during attention: softmax p8 lives at SP_X[0, ctx) and nothing else
    // touches SP_X until the FFN quant8 overwrites it.
    {
#ifdef BONSAI_VEC_LOOPS2
        if (qfix_use) {
            // Fixed shift only (2026-09-08): PROG_QUANT8 with A1 = 0 still runs its absmax
            // pass over all 128 beats before the fixed >>8 pass, and nothing reads the
            // absmax it returns here. PROG_KVPACK_H with one "head" is the same LDV / VRNE 8
            // / VCLAMP8 / STB chain without the scan - byte-identical, half the loads.
            int32_t fargs[8] = {(int32_t)SP_Q, 0, (int32_t)(SP_X + 0x1000),
                                (n_head*head_dim)/16, 0, 0, 0, 1};
            seq_desc_vec(di++, PROG_KVPACK_H, fargs, 8, 0);
        } else
#endif
        {
            int32_t qargs[4] = {(int32_t)SP_Q, 0, (int32_t)(SP_X + 0x1000),
                                (n_head*head_dim)/16};
            seq_desc_vec(di++, PROG_QUANT8, qargs, 4, 0);
        }
    }
    seq_desc_stop(di++);
    seq_qkr_count = di;
}

static int16_t eps_memo[64]; // seq_eps_int cache, index es+32; 0 = not computed yet.

static int seq_eps_int(int es)
{
    // Soft-float (lroundf/ldexpf) costs hundreds of cycles on this FPU-less core and
    // es takes a handful of values - memoize (qtrace, 2026-09-02). e >= 1 always, so
    // 0 doubles as the empty marker.
    int mi = es + 32;
    if (mi >= 0 && mi < 64 && eps_memo[mi])
        return eps_memo[mi];
    int e = (int)lroundf(H->rms_eps*ldexpf(1.0f, -2*es));
    if (e < 1) e = 1;
    if (mi >= 0 && mi < 64 && e <= 0x7FFF)
        eps_memo[mi] = (int16_t)e;
    return e;
}

// Sequencer descriptor list for the fused per-head attention chain:
//   quant8 -> scores (engine) -> softmax8 -> spill(p8 sum) -> AXPY (rshift from rpre of
//   the chained res6 = p8 sum) -> reduce x(n_ch-1) -> spill(absmax) -> norm_i32 (k via
//   res6). One block per head slot, reused across chunks; head/token/layer-dependent
//   words are patched before each run. Spills feed the one CPU float fold per head
//   (c2/hmax) after the list - identical numerics to the CPU-dispatched fused path.
static int seq_attn_base, seq_attn_dph, seq_attn_heads;
// Pipelined attention list (probe 1, 2026-08-31): head j+1's scores op is dispatched
// NOWAIT before head j's softmax8, so the engine streams K(j+1) while the vector unit
// works. Scores double-buffer between the two HALVES of the SP_SC region - both halves
// are written only by scores ops, so the INT32_MIN padding written once per forward
// survives across layers, the same invariant the single buffer relied on. (SP_G was the
// first choice and was wrong: the FFN rewrites it every layer, so from layer 1 on the
// odd heads' softmax read gate garbage where the padding had been - the n8 sim's GEN-HW
// tokens caught it.) The serialization softmax8(j) needs against scores(j) comes free from the
// dispatch of scores(j+1) - an engine descriptor only dispatches when the engine is
// idle - so no barrier bits are required. Layout when active:
//   base+0                 scores(slot 0), NOWAIT                    (prologue)
//   base+1+j*5 .. +4       [spill(j-1)|nopd] [scores(j+1) NOWAIT|nopd] [norm(j-1)|nopd]
//                          softmax8(j)  wv(j) NOWAIT
//   base+1+chunk*5 .. +1   spill(chunk-1)  norm(chunk-1)              (epilogue, static)
// Deep form (2026-08-31 night): wv(j) is NOWAIT; the next group's spill(j) stalls on the
// absmax interlock until wv(j) completes (the tracker is safe - scores(j+2) sits AFTER
// the spill in list order), and norm(j-1)+softmax8(j) execute under scores(j+1)'s
// stream. The engine idles only for the ~20-cycle spill instead of spill+norm, and the
// vector unit fills the scores streams completely. Non-chunked only.
static int attn_pipe = 1;        // What the NEXT build will use (see attn_pipe_req).
static int attn_pipe_req = 1;    // Requested (aovl console toggle). attn_list_prep_token
                                 // clears attn_pipe when the context outgrows the second
                                 // scores buffer and restores it when it fits again; the
                                 // request itself must survive that, or one long generation
                                 // would leave aovl off for every generation after it.
static int attn_prep_pos = -1;   // Position attn_list_prep_token last ran for (lovl dedup).
static int attn_pipe_active = 0; // What the built list uses.
// The second scores buffer (odd heads, deep layout) sits at SP_SC + 0x3800 and resident K
// starts at SP_K: 0x800 bytes = 512 int32 entries, NOT the 14 KB the "halves" comment
// used to claim. Its INT32_MIN padding is written once per token, and every layer's QKV
// stage rewrites SP_K - so from ctx 513 the padding beyond entry 512 was K data by the
// time softmax read it (and V data from 1537). Bench runs at ctx~67 and the tokens gate
// is short, so it never showed. Buffer A is safe: ceil32(cnt) <= 2048 = max_ctx ends it
// exactly at SP_Q. (Found by the 2026-09-07 external review.)
#define SC_B_CAP_BYTES ((uint32_t)SP_K - ((uint32_t)SP_SC + 0x3800u))
static inline uint32_t attn_grp(int j) {
    return (uint32_t)seq_attn_base + (attn_pipe_active ? 1u : 0u)
           + (uint32_t)j*(uint32_t)seq_attn_dph;
}
static inline uint32_t attn_di_sc(int j) { // scores descriptor serving slot j
    if (!attn_pipe_active) return attn_grp(j);
    // Deep layout: slot 1 of the previous group (slot 0 is that group's spill). The
    // aovl-era +0 here made sc(j) clobber spill(j-1) and left every slot 1 unwritten -
    // w0 = 0 is a STOP, so the list ended after two scores ops; the descriptor dump
    // made it obvious after three wrong-token hunts in the execution semantics.
    return (j == 0) ? (uint32_t)seq_attn_base : attn_grp(j - 1) + 1u;
}
static inline uint32_t attn_di_sm(int j) { // softmax8 slot
    return attn_grp(j) + (attn_pipe_active ? 3u : 1u);
}
static inline uint32_t attn_di_wv(int j) { // wv (AXPY) slot
    return attn_grp(j) + (attn_pipe_active ? 4u : 2u);
}
static inline uint32_t attn_sc_dst(int j) {
    // Double-buffered ONLY in the pipelined layout. The plain list never needed it, and
    // its INT32_MIN padding is only maintained for the active layout - alternating here
    // unconditionally left aovl-off reading an unpadded second buffer for odd heads
    // (latent since the aovl commit: every sim gate ran the pipelined default and the
    // card's aovl-off benches never token-checked; the pipe-off bisect exposed it).
    return (uint32_t)SP_SC
           + ((attn_pipe_active && (j & 1)) ? 0x3800u : 0u); // B holds 512 entries, not 14 KB
                                                             // (SC_B_CAP_BYTES): K is at SP_K.
}
static inline int attn_list_len(int nh) {
    return nh*seq_attn_dph + (attn_pipe_active ? 3 : 0); // Prologue + 2-slot epilogue.
}
static void seq_build_attn(void)
{
    seq_attn_heads = 0;
    attn_prep_pos = -1; // Rebuild wipes the per-token patch words; force a re-prep.
    if (n_ch < 2)
        return; // n1's absmax lives in the engine CSR; keep CPU-dispatched ops there.
    // 5 descriptors per head: scores, softmax8 (self-spills its p8 sum), axpy(xred),
    // spill(absmax from the engine tracker), norm_i32. It was 7 until F0-5 moved the
    // per-head quant8 into the qkr list (one batched call) and the p8 spill into
    // softmax8's tail, and 6 + n_ch = 14 before the engine learned to reduce across
    // channels. 16*5 + 66 = 146 of 192, so the FFN and scale8 lists fit behind it.
    int dph = 5;
    int cap = (192 - seq_qkr_count - (attn_pipe ? 3 : 0))/dph;
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
            // spill and its KRES6-consuming norm hands the norm k = 0 - the tiny sim's
            // GEN-HW tokens caught exactly that.
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
        // the spill's S7 seed sampled a not-yet-settled absmax (tiny-sim GEN-HW tokens
        // caught it - deterministically wrong on every head). The WAIT_ENG path's
        // descriptor fetch gives the settle time for free, as it always has.
        // The engine summed the channels itself, so the n_ch-1 reduce_acc calls are gone and
        // the absmax the CPU fold needs comes from the engine tracker instead of the last
        // reduce's res6. seq_desc_vec_seed puts it in S7 for the spill, and spill's trailing
        // SMOV s6, s7 leaves it in res6 - so the norm below keeps taking k from KRES6 and
        // sees the identical value. Numerics are unchanged by construction, not by luck.
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

// Dual-Q (S3) attention lists, two phases per layer. Phase 1 per kv-head: quant8 both
// sibling Q-heads -> one dual-Q scores op over the shared K stream -> softmax8 + spill
// per head. The CPU then reads the p8-sum spills, computes each head's Rpre and patches
// the AXPY descriptors. Phase 2 per kv-head: one dual-Q AXPY over the shared V stream ->
// per head: reduce chain -> spill(absmax) -> norm_i32 (k from res6). Requires n_rep == 2
// and n_ch >= 2; other configs use the v1 list or CPU dispatch.
// Sequencer attention: the descriptor lists collapse ~6 dispatch round-trips per head into
// one submit. On by default - measured 1.415 tok/s against 1.175 with the lists off, same
// tokens either way.
//
// They were held off while the interleaved gather could wedge the engine mid-op, which
// looked like a sequencer bug and was not: the gather consumed a merged result on merge_mask
// but counted it on grant_valid, so a result could vanish without incrementing result_cnt.
// `seq off` still selects the CPU-dispatched path per head.
// FFN chain as ONE descriptor list (kreg exponent registers, sequencer w0[5]/w1[16+2i]).
// The gap histogram put 43% of the GEN-HW window in 200-5k-cycle inter-op gaps; the FFN's
// share exists because eg = ex8 - bias + rshift + kg mixes a per-layer constant with the
// gate matvec's k - inexpressible with KABS alone. Layout (10 descriptors):
//   0 gate matvec [capture k -> kreg0]      1 spill engine-absmax -> SP_FFSP+0
//   2 norm_i32 SP_Y  -> SP_G [A4 = KABS]    3 up matvec
//   4 spill engine-absmax -> SP_FFSP+32     5 norm_i32 SP_YU -> SP_U [A4 = KABS]
//   6 silu_mul [A4 = imm(-(ex8 - bias_g + rshift_g + 6)) MINUS kreg0]
//   7 quant8 SP_G -> SP_X (measured; its res6 = source absmax survives the list)
//   8 down matvec                           9 stop
// Post-list the CPU recovers: kg/ku from the two spilled absmaxes, s from vec res6, kd and
// dmax from the engine absmax CSR (down ran last) - the SAME values the removed round-trips
// produced, so the numerics are identical by construction. The final norm -> SP_D and
// resid_add stay CPU-side (they need *omax and the residual ratchet state).
// C2-1a FFN diagnosis (fprof console command): windowed overlap accumulators over the
// FFN block on whichever path ran, plus one layer's fine-grained trace per forward
// (layer 0): descriptor timestamps for the list path, step timestamps for the op path.
int fprof_use = 0;
static uint32_t fprof_acc_list[4], fprof_acc_ops[4];
static uint32_t fprof_nlist, fprof_nops;
static uint32_t fprof_ts[16]; static int fprof_nts;
static uint32_t fprof_steps[8]; static int fprof_nsteps;
void fprof_reset(void)
{
    for (int i = 0; i < 4; i++) fprof_acc_list[i] = fprof_acc_ops[i] = 0;
    fprof_nlist = fprof_nops = 0; fprof_nts = 0; fprof_nsteps = 0;
}
void fprof_print(void)
{
    if (fprof_nlist) { perf_ovl_report("FFN-list", fprof_acc_list);
                       printf("  windows %lu\n", (unsigned long)fprof_nlist); }
    if (fprof_nops)  { perf_ovl_report("FFN-ops", fprof_acc_ops);
                       printf("  windows %lu\n", (unsigned long)fprof_nops); }
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

static int seq_ffn_base = -1;
static int seq_ffn_len  = 0;
static int ffn_shape_base = -1; // seq_ffn_base the FFN engine shape images were written for.
// The SiLU split (2026-09-07, from the external review): everything in silu_q8 up to the
// x*sigmoid(x) product is a function of the GATE alone, so that half could run UNDER the
// up matvec's stream (where norm_g already runs) and only the multiply/absmax/quant8 tail
// would wait for up. Byte-identical by construction (asm.py, test_silu_split_matches_fused);
// the list grows 9 -> 10 descriptors. On v9 it measured +0.42% (slower): the vector unit
// ran ~2.9x slower under an engine stream (norm_g 6,096 cycles under the up stream vs the
// identical norm_u 2,133 idle), so norm_g alone filled the window and silu_gate ran after
// the stream. The v11 result packer (gateware/matvec.py _ResultPacker) removed that
// throttle - norm_g under the stream is 2,108 on v11 - and the split is now worth
// -4.07% same-boot (2,477,618 vs 2,582,600 cycles/token, 3 pairs, 2026-09-08). Default ON.
static int ffn_split = 1;
static int ffn_di_down = 7; // Down matvec's descriptor (b + 7 fused, b + 8 split).
static int ffn_di_a4   = 6; // Descriptor whose A4 takes the per-layer gate shift.
static void seq_build_ffn(void)
{
#ifdef BONSAI_HAS_DUALQ
    return; // Dual builds use attn2 lists; FFN list only wired on the shipping non-dual path.
#else
    int di = seq_qkr_count + attn_list_len(seq_attn_heads);
    int len = ffn_split ? 10 : 9;
    if (di + len > 192 || seq_attn_heads == 0)
        return;
    seq_ffn_base = di;
    seq_ffn_len  = len;
    ffn_shape_base = -1; // Engine slots are placeholders until ffn_prep_layer.
    // Pipelined order (2026-08-29): the up matvec is NOWAIT and the gate norm runs UNDER
    // its stream - the list keeps the `ovl` overlap it used to lose. The norms take k
    // from kregs (captured at each matvec) instead of KABS, so they no longer read the
    // live absmax and can float across engine ops. kreg consumers are all >= one
    // interlocked spill program past the capturing op's done, which covers the
    // capture-at-done+2 lag on this bitstream.
    //   0 gate mv [cap k->kreg0]    1 spill gate-absmax (seed_abs, engine idle here)
    //   2 up mv NOWAIT [cap k->kreg1]
    //   3 norm_g SP_Y -> SP_G [A4 = 0 + kreg0]   <- runs under the up stream
    //   4 spill up-absmax (seed_abs: interlock waits the stream)
    //   5 norm_u SP_YU -> SP_U [A4 = 0 + kreg1]
    //   6 silu_q8 fused [A4 = imm + kreg0; res6 = source absmax, read by the CPU
    //     after the list - desc 7 is an engine op and leaves it alone]
    //   7 down mv   8 stop
    // (C2-1b: descriptors 6+7 used to be the UNFUSED silu_mul + measured quant8 pair
    // while the op-by-op path ran the fused PROG_SILU_Q8 - a whole extra ffn_dim pass
    // x28 layers rode the list side of every A/B. Same kreg0 A4 mechanics.)
    struct engine_op z = {0};
    seq_desc_engine(di + 0, &z);
    int32_t pa = (int32_t)SP_FFSP;
    seq_desc_vec_seed(di + 1, PROG_SPILL, &pa, 1, 0, 1);
    seq_desc_engine(di + 2, &z);
    // norm k via KRES6, NOT a kreg-add: desc1's spill left the gate absmax in res6
    // (SMOV s6,s7), and KRES6 is not in the hazard interlock - so this descriptor
    // dispatches UNDER the NOWAIT up stream. A kadd arg would stall on the (currently
    // register-blind) interlock and serialize the list - measured 13.60M vs 13.45M.
    int32_t na[5] = {(int32_t)SP_Y, 0, (int32_t)SP_G, ffn_dim/16, 0};
    seq_desc_vec(di + 3, PROG_NORM_I32, na, 5, (uint16_t)(SEQ_SRC_KRES6 << 8));
    int d = di + 4; // Next free descriptor.
    if (ffn_split) {
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
    if (ffn_split) {
        // mul_q8: (silu * up) << A6, clamp, absmax (res6 for the CPU), quant8 -> SP_X.
        int32_t ma[8] = {(int32_t)SP_G, (int32_t)SP_U, (int32_t)SP_G, ffn_dim/16, 0, 0, -15,
                         (int32_t)SP_X};
        seq_desc_vec(d, PROG_MUL_Q8, ma, 8, 0);
    } else {
        int32_t sa[8] = {(int32_t)SP_G, (int32_t)SP_U, (int32_t)SP_G, ffn_dim/16, 0, 0, -15,
                         (int32_t)SP_X};
        seq_desc_vec(d, PROG_SILU_Q8, sa, 8, 0);
        seq_patch(d, 1, (uint32_t)PROG_SILU_Q8 | (1u << (16 + 2*4))); // A4 kreg0 term...
        seq_patch(d, 0, 2u | (1u << 11)); // ...ADDS (w0[11]): A4 = imm + kg (VSHLS conv).
        ffn_di_a4 = d - di;
    }
    d++;
    seq_desc_engine(d, &z);                                               // down mv
    ffn_di_down = d - di;
    d++;
    seq_desc_stop(d);
#endif
}


// F0-6 (2026-08-29): the per-head attention-output scale8 as one n_head-descriptor list.
// Only A4 (the per-head Q15 multiplier Mh) changes per layer; a CPU vec_run costs ~500
// cycles of dispatch against ~50 for an in-list descriptor, x16 heads x 28 layers.
// Placed behind the FFN list (which is behind the attention list); 156..172 of 192 on
// the 1.7B. Not built on dual-Q images (their two-phase lists own that region).
static int seq_scale8_base = -1;
static void seq_build_scale8(void)
{
#ifndef BONSAI_HAS_DUALQ
    // Invalidate first, like seq_build_qkv does with its own base. Both early returns below
    // used to leave a STALE base behind, and seq_build_qkv derives seq_qkv_base from it - so
    // a stale (smaller) value could place the qkv list on top of the FFN list's descriptors.
    // qkv_prep_layer(l+1) runs UNDER the FFN list, which would then be a patch of a RUNNING
    // list: exactly the invariant the start-drain interlock relies on (gateware/sequencer.py).
    // Not reachable on the 1.7B - 159 of 192 descriptors, neither return fires - but it is
    // one larger model away, and every consumer already guards on >= 0, so -1 degrades
    // cleanly to the CPU path instead of corrupting a list.
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
#endif
}

// QKV projection list (2026-09-02, "qkvl"): the three attention projections and their
// norm_i32s as one sequencer run, the FFN list's shape - each norm runs UNDER the next
// projection's NOWAIT stream with its shift derived in-sequencer (KRES6) from the
// absmax the preceding spill left in res6. Replaces six CPU dispatches per layer
// (tprof: the norm_i32 windows sat 55% idle around them). The three absmaxes are
// spilled to SP_QKSP so the CPU recovers kq/kk/kv exactly as the op path computes them.
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
#ifndef BONSAI_HAS_DUALQ
    seq_qkv_base = -1;
    qkv_prepped  = -1;
    if (seq_scale8_base < 0)
        return;
    int di = seq_scale8_base + n_head + 1;
    if (di + 10 > 192)
        return;
    seq_qkv_base = di;
    struct engine_op z = {0};
    int32_t pq = (int32_t)(SP_QKSP + 0), pk = (int32_t)(SP_QKSP + 32), pv = (int32_t)(SP_QKSP + 64);
    int q_dim = n_head*head_dim, kv_dim = n_kv_head*head_dim;
    int32_t nq[5] = {(int32_t)SP_Y,            0, (int32_t)SP_Q, q_dim/16,  0};
    int32_t nk[5] = {(int32_t)(SP_Y + 0x4000), 0, (int32_t)SP_K, kv_dim/16, 0};
    int32_t nv[5] = {(int32_t)(SP_Y + 0x5000), 0, (int32_t)SP_V, kv_dim/16, 0};
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
#endif
}

static void qkv_prep_layer(int l)
{
    if (seq_qkv_base < 0 || qkv_prepped == l)
        return;
    struct engine_op mo = {0};
    uint32_t b = (uint32_t)seq_qkv_base;
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

int seq_use = 1;
// FFN descriptor list: DEFAULT ON since C2-1b/1c (2026-09-01). History, kept because it
// carries two lessons: the list was default-off first for a 1.7B tokens failure (fixed
// long ago by the silu rewrite) and then because it measured +8.5% SLOWER - but that
// comparison was never like-for-like (the list ran the UNFUSED silu_mul+quant8 pair
// while the op path ran the fused silu_q8). With the fused descriptor the same-boot
// windowed counters read list 12.03M vs ops 14.32M per bench-8 (neither-busy 4% vs 19%)
// and bench-64 is -2.0%/token. `ffnseq off` selects the op-by-op path.
int seq_ffn_use = 1;
// Engine/vec overlap (docs/overlap-design.md): run the FFN up-projection stream UNDER the
// gate-norm vector program (disjoint regions; gate absmax captured before the submit, so
// numerics are byte-identical by construction). `ovl off` for same-binary A/B.
int ovl_use = 1;

static int seq_at2_base, seq_at2_p1n, seq_at2_p2n, seq_at2_kvchunk;
static int seq_at2_p1dn, seq_at2_p2dn;
static void seq_build_attn2(void)
{
    seq_at2_kvchunk = 0;
#ifndef BONSAI_HAS_DUALQ
    return; // Engine built without the head-B datapath: use the v1 single-head lists.
#else
    if (n_ch < 2 || n_head != 2*n_kv_head)
        return;
    int p1dn = 7;            // q8A q8B scores smA spA smB spB.
    int p2dn = 2*n_ch + 3;   // axpy redA* spA normA redB* spB normB.
    int cap  = (192 - seq_qkr_count)/(p1dn + p2dn);
    if (cap <= 0)
        return;
    int chunk = (cap < n_kv_head) ? cap : n_kv_head;
    int lg = 0;
    while ((1 << lg) < head_dim) lg++;
    int Ceff = (lg & 1) ? 523 : 739;
    seq_at2_base    = seq_qkr_count;
    seq_at2_p1dn    = p1dn;
    seq_at2_p2dn    = p2dn;
    seq_at2_p1n     = chunk*p1dn;
    seq_at2_p2n     = chunk*p2dn;
    seq_at2_kvchunk = chunk;
    uint32_t qs2 = (uint32_t)n_ch*head_dim; // AXPY head-B partial offset (words).
    for (int j = 0; j < chunk; j++) {
        int hA = 2*j, hB = 2*j + 1;
        uint32_t d1 = (uint32_t)seq_at2_base + (uint32_t)j*p1dn;
        int32_t qa[4] = {(int32_t)(SP_Q + (uint32_t)hA*head_dim*2), 0,
                         (int32_t)SP_X, head_dim/16};
        seq_desc_vec(d1 + 0, PROG_QUANT8, qa, 4, 0);
        int32_t qb[4] = {(int32_t)(SP_Q + (uint32_t)hB*head_dim*2), 0,
                         (int32_t)(SP_X + head_dim), head_dim/16};
        seq_desc_vec(d1 + 1, PROG_QUANT8, qb, 4, 0);
        struct engine_op sc = {0};
        sc.kind    = ENGINE_KIND_Q8;
        sc.ilv     = 1;
        sc.cols    = head_dim;
        sc.sbeats  = 1;
        sc.cbeats  = head_dim/32;
        sc.xsrc    = SP_X;
        sc.dst     = SP_Y;
        sc.qn      = 1;
        sc.qstride = 0x800;
        seq_desc_engine(d1 + 2, &sc);
        int32_t sa[7] = {(int32_t)SP_Y, Ceff, (int32_t)SP_X, 0, -11, 0,
                         (int32_t)(SP_A2P8 + (uint32_t)hA*32)};
        seq_desc_vec(d1 + 3, PROG_SOFTMAX8, sa, 7, 0);
        int32_t pa = (int32_t)(SP_A2P8 + (uint32_t)hA*32);
        seq_desc_vec(d1 + 4, PROG_SPILL, &pa, 1, 0);
        int32_t sb[7] = {(int32_t)SP_SCB, Ceff, 0, 0, -11, 0,     // A2 (p8B dst) per token.
                         (int32_t)(SP_A2P8 + (uint32_t)hB*32)};
        seq_desc_vec(d1 + 5, PROG_SOFTMAX8, sb, 7, 0);
        int32_t pb = (int32_t)(SP_A2P8 + (uint32_t)hB*32);
        seq_desc_vec(d1 + 6, PROG_SPILL, &pb, 1, 0);

        uint32_t d2 = (uint32_t)seq_at2_base + (uint32_t)seq_at2_p1n + (uint32_t)j*p2dn;
        struct engine_op wv = {0};
        wv.kind    = ENGINE_KIND_AXPY;
        wv.cols    = head_dim;
        wv.sbeats  = 1;
        wv.cbeats  = head_dim/32;
        wv.xsrc    = SP_X;
        wv.dst     = SP_AW2;
        wv.qn      = 1;
        wv.qstride = (uint16_t)qs2;
        seq_desc_engine(d2 + 0, &wv);
        for (int c = 1; c < n_ch; c++) {
            int32_t ra[4] = {(int32_t)SP_AW2, (int32_t)(SP_AW2 + (uint32_t)c*head_dim*4),
                             (int32_t)SP_AW2, head_dim/8};
            seq_desc_vec(d2 + c, PROG_REDUCE_ACC, ra, 4, 0);
        }
        int32_t ma = (int32_t)(SP_A2AM + (uint32_t)hA*32);
        seq_desc_vec(d2 + n_ch, PROG_SPILL, &ma, 1, 0);
        int32_t na[5] = {(int32_t)SP_AW2, 0, (int32_t)(SP_H + (uint32_t)hA*head_dim*2),
                         head_dim/16, 0};
        seq_desc_vec(d2 + n_ch + 1, PROG_NORM_I32, na, 5,
                     (uint16_t)(SEQ_SRC_KRES6 << 8));
        uint32_t b2 = SP_AW2 + 4*qs2;
        for (int c = 1; c < n_ch; c++) {
            int32_t rb2[4] = {(int32_t)b2, (int32_t)(b2 + (uint32_t)c*head_dim*4),
                              (int32_t)b2, head_dim/8};
            seq_desc_vec(d2 + n_ch + 1 + c, PROG_REDUCE_ACC, rb2, 4, 0);
        }
        int32_t mb = (int32_t)(SP_A2AM + (uint32_t)hB*32);
        seq_desc_vec(d2 + 2*n_ch + 1, PROG_SPILL, &mb, 1, 0);
        int32_t nb[5] = {(int32_t)b2, 0, (int32_t)(SP_H + (uint32_t)hB*head_dim*2),
                         head_dim/16, 0};
        seq_desc_vec(d2 + 2*n_ch + 2, PROG_NORM_I32, nb, 5,
                     (uint16_t)(SEQ_SRC_KRES6 << 8));
    }
#endif
}
#endif

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
// One layer of dual-Q sequencer attention (S3): shared by forward() and
// forward_pair(). Fills aw_c[] and raises *aw_hmax.
static void attn_seq2_layer(int l, int pos, float *aw_c, float *aw_hmax)
{
    int n_rep = n_head/n_kv_head;
    // S3 dual-Q path: one K stream scores both sibling heads, one V stream feeds
    // both AXPYs - halves the KV restream of GQA. Two list runs per layer with a
    // CPU Rpre patch between them; numerics identical to the single-head chain.
    perf_enter(OP_SCORES);
    int cnt  = pos + 1;
    int padn = SC_PADN(cnt);
    volatile int32_t *scA = (volatile int32_t *)(SCRATCHPAD_BASE + SP_Y);
    volatile int32_t *scB = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SCB);
    for (int i2 = 0; i2 < padn; i2++) {
        scA[cnt + i2] = (int32_t)0x80000000;
        scB[cnt + i2] = (int32_t)0x80000000;
    }
    uint32_t rows  = (uint32_t)cnt/n_ch, rem = (uint32_t)cnt % n_ch;
    uint32_t wlen  = rows*kv_rb;
    uint32_t rowsw = rows | (rem << 16);
    int lg = 0;
    while ((1 << lg) < head_dim) lg++;
    uint32_t blocks = (uint32_t)(cnt + padn)/16;
    uint32_t p8pad  = ((uint32_t)cnt + 31) & ~31u; // 32B-padded p8 vector size.
    for (int k0 = 0; k0 < n_kv_head; k0 += seq_at2_kvchunk) {
        int nk = n_kv_head - k0;
        if (nk > seq_at2_kvchunk) nk = seq_at2_kvchunk;
        for (int j = 0; j < nk; j++) {
            int kvh = k0 + j, hA = 2*kvh, hB = hA + 1;
            uint32_t d1 = (uint32_t)seq_at2_base + (uint32_t)j*seq_at2_p1dn;
            int Bk, Rk;
            kv_scores_br(l, kvh, 0, &Bk, &Rk);
            int sigma = (E_NORM + 8) - Bk + Rk - (lg >> 1);
            seq_patch(d1 + 0, 2, SP_Q + (uint32_t)hA*head_dim*2);
            seq_patch(d1 + 1, 2, SP_Q + (uint32_t)hB*head_dim*2);
            seq_patch(d1 + 2, 2, rowsw);
            seq_patch(d1 + 2, 4, kv_region_addr(l, kvh, 0));
            seq_patch(d1 + 2, 5, wlen);
            seq_patch(d1 + 2, 7, (uint32_t)Bk | ((uint32_t)Rk << 16));
            // p8 pool: per-kv-head buffers in the FFN scratch (dead here) so
            // phase 2's AXPYs still see them after later heads' softmaxes ran.
            uint32_t pbase = SP_G + (uint32_t)kvh*2*p8pad;
            seq_patch(d1 + 3, 4, pbase);
            seq_patch(d1 + 3, 5, blocks);
            seq_patch(d1 + 3, 7, (uint32_t)(sigma + 6 + 11));
            seq_patch(d1 + 4, 2, SP_A2P8 + (uint32_t)hA*32);
            seq_patch(d1 + 5, 4, pbase + p8pad);
            seq_patch(d1 + 5, 5, blocks);
            seq_patch(d1 + 5, 7, (uint32_t)(sigma + 6 + 11));
            seq_patch(d1 + 6, 2, SP_A2P8 + (uint32_t)hB*32);
        }
        seq_run((uint32_t)seq_at2_base, (uint32_t)nk*seq_at2_p1dn);
        for (int j = 0; j < nk; j++) {
            int kvh = k0 + j, hA = 2*kvh, hB = hA + 1;
            uint32_t d2 = (uint32_t)seq_at2_base + (uint32_t)seq_at2_p1n
                          + (uint32_t)j*seq_at2_p2dn;
            uint32_t P8a = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_A2P8
                                                  + (uint32_t)hA*32);
            uint32_t P8b = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_A2P8
                                                  + (uint32_t)hB*32);
            int Bv, Rd;
            kv_scores_br(l, kvh, 1, &Bv, &Rd);
            uint32_t Ra = (uint32_t)axpy_rpre_c(P8a ? P8a : 1);
            uint32_t Rb = (uint32_t)axpy_rpre_c(P8b ? P8b : 1);
            seq_patch(d2 + 0, 2, rowsw);
            seq_patch(d2 + 0, 4, kv_region_addr(l, kvh, 1));
            seq_patch(d2 + 0, 5, wlen);
            seq_patch(d2 + 0, 7, (uint32_t)Bv | (Ra << 16) | (Rb << 22));
            seq_patch(d2 + 0, 8, SP_G + (uint32_t)kvh*2*p8pad);
            seq_patch(d2 + n_ch, 2, SP_A2AM + (uint32_t)hA*32);
            seq_patch(d2 + n_ch + 1, 4, SP_H + (uint32_t)hA*head_dim*2);
            seq_patch(d2 + 2*n_ch + 1, 2, SP_A2AM + (uint32_t)hB*32);
            seq_patch(d2 + 2*n_ch + 2, 4, SP_H + (uint32_t)hB*head_dim*2);
            (void)Rd;
        }
        seq_run((uint32_t)seq_at2_base + (uint32_t)seq_at2_p1n,
                (uint32_t)nk*seq_at2_p2dn);
    }
    perf_exit(OP_SCORES);
    perf_enter(OP_WV);
    for (int hh = 0; hh < n_head; hh++) {
        int kvh = hh/n_rep;
        uint32_t P8 = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_A2P8
                                             + (uint32_t)hh*32);
        uint32_t am = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_A2AM
                                             + (uint32_t)hh*32);
        int Bv, Rd;
        kv_scores_br(l, kvh, 1, &Bv, &Rd);
        int Rpre = axpy_rpre_c(P8 ? P8 : 1);
        int k2 = 0;
        k2 = (bitlen32(am) > 14) ? bitlen32(am) - 14 : 0; // == the >16383 scan, via clz.
        float c2 = ldexpf(1.0f/(float)(P8 ? P8 : 1), Rpre - Bv + k2);
        aw_c[hh] = c2;
        float hm2 = (float)(am >> k2)*c2;
        if (hm2 > (*aw_hmax)) (*aw_hmax) = hm2;
        (void)Rd;
    }
    perf_exit(OP_WV);
}

#endif

// Speculative-decode state: successor draft table + acceptance counters (the
// counters exist on every build; the machinery only under the full-HW guard).
static int       spec_on, spec_acc, spec_rej;
static uint16_t *spec_succ;
static uint8_t  *spec_succ_v;

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && defined(BONSAI_HAS_TOPK) && defined(BONSAI_HAS_DUALQ)
// S6 speculative decode: forward TWO positions through ONE weight stream per matvec
// (dual-x ternary, model_matvec2). Positions are independent along the residual; the only
// coupling is attention, which runs per position in-order (store KV p0 -> attn p0 ->
// store KV p1 -> attn p1) so each position's Bk/Rk view matches the non-speculative
// mirror exactly. Greedy speculation is lossless: accepted streams are token-identical
// to plain decode, so the existing gate references validate this path.
//
// Per-position regions: residual XR + p*0x1000; x8 pairs contiguous at SP_X (+p*width);
// matvec results A/B at dst + p*4*rows; SP_Q/SP_K staging halves; V p1 staged in the
// KV-pack region; FFN G/U halves; SP_D halves. kv_ekmax entries touched by p1's stores
// are captured per layer for exact rollback on draft rejection.
// Undo records: (index, value) pairs.
static int     ek_undo_n;
// In the main_ram static window, not .bss: sram is 8 KB *including the stack*, and these
// 3 KB were a third of it. Same window as model.c's entry cache (BONSAI_HDR_ADDR + 0x80000
// + 4 KB); data accesses to main_ram are the norm here - the heap is next door.
#define EK_UNDO_BASE (BONSAI_HDR_ADDR + 0x90000)
static int16_t *const ek_undo_idx = (int16_t *)EK_UNDO_BASE;
static int8_t  *const ek_undo_val = (int8_t  *)(EK_UNDO_BASE + sizeof(int16_t)*64*8*2);
static void ek_capture(int l)
{
    for (int h = 0; h < n_kv_head; h++)
        for (int k = 0; k < 2; k++) {
            int idx = kv_ek_idx(l, h, k);
            ek_undo_idx[ek_undo_n] = (int16_t)idx;
            ek_undo_val[ek_undo_n] = kv_ekmax[idx];
            ek_undo_n++;
        }
}
static void ek_restore(void)
{
    for (int i = 0; i < ek_undo_n; i++)
        kv_ekmax[ek_undo_idx[i]] = ek_undo_val[i];
    ek_undo_n = 0;
}

// Forward positions n_pos (token t0) and n_pos+1 (token t1); returns argmaxes for both.
// Returns 0 on success, -1 if the argmax tap hit a padding row (caller falls back).
static int forward_pair(int t0, int t1, int *am0, int *am1)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    const struct bons_entry *rt  = model_entry_kind(ROLE_ROPE_TAB, 0, KIND_NORM16);
    int   kv_dim = n_kv_head*head_dim;
    int   q_dim  = n_head*head_dim;
    int   pos    = n_pos;
    int   ex_st[2];
    float aw_cp[2][32];
    uint32_t xr[2]  = {SP_XR, SP_XR + 0x1000};
    uint32_t vst[2] = {SP_V, SP_KVP + 0x1000};
    ek_undo_n = 0;

    perf_enter(OP_EMBED);
    uint32_t xa_st[2] = {X_AMAX_AFTER_EMBED, X_AMAX_AFTER_EMBED};
    ex_st[0] = embed_row_bfp(emb, t0, xr[0]);
    ex_st[1] = embed_row_bfp(emb, t1, xr[1]);
    perf_exit(OP_EMBED);

    for (int l = 0; l < n_layer; l++) {
        const struct bons_entry *eqt = model_entry(ROLE_ATTN_Q, l);
        const struct bons_entry *ekt = model_entry(ROLE_ATTN_K, l);
        const struct bons_entry *evt = model_entry(ROLE_ATTN_V, l);
        // Attention rmsnorm + quant8 per position -> x8 pair.
        int sx8[2];
        for (int p = 0; p < 2; p++) {
            perf_enter(OP_RMSNORM);
            op_rmsnorm_sp(xr[p], ex_st[p], attn_norm[l],
                          model_entry_kind(ROLE_ATTN_NORM, l, KIND_NORM16), n_embd, SP_H);
            perf_exit(OP_RMSNORM);
            perf_enter(OP_QUANT8);
            sx8[p] = op_quant8_sp(SP_H, SP_X + (uint32_t)p*n_embd, n_embd);
            perf_exit(OP_QUANT8);
        }
        // q/k/v: one weight stream each, both positions.
        perf_enter(OP_NORM_I32);
        int eq_r[2], ek_r[2], ev_r[2];
        model_matvec2(eqt, SP_X, SP_Y, (uint16_t)eqt->rows);
        for (int p = 0; p < 2; p++) {
            int kq = op_norm_i32_sp(SP_Y + (uint32_t)p*4*eqt->rows,
                                    SP_Q + (uint32_t)p*q_dim*2, q_dim);
            eq_r[p] = (E_NORM + sx8[p]) - (int)eqt->bias + (int)eqt->rshift + kq;
        }
        model_matvec2(ekt, SP_X, SP_Y, (uint16_t)ekt->rows);
        for (int p = 0; p < 2; p++) {
            int kk = op_norm_i32_sp(SP_Y + (uint32_t)p*4*ekt->rows,
                                    SP_K + (uint32_t)p*kv_dim*2, kv_dim);
            ek_r[p] = (E_NORM + sx8[p]) - (int)ekt->bias + (int)ekt->rshift + kk;
        }
        model_matvec2(evt, SP_X, SP_Y, (uint16_t)evt->rows);
        for (int p = 0; p < 2; p++) {
            int kv2 = op_norm_i32_sp(SP_Y + (uint32_t)p*4*evt->rows, vst[p], kv_dim);
            ev_r[p] = (E_NORM + sx8[p]) - (int)evt->bias + (int)evt->rshift + kv2;
        }
        perf_exit(OP_NORM_I32);
        // qk-norm weight rows (shared by both positions).
        const struct bons_entry *qn16 = model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16);
        const struct bons_entry *kn16 = model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16);
        engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + qn16->chan_base_beats*32,
                     qn16->chan_len_beats, SP_FET);
        engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                     kn16->chan_len_beats, SP_FET2);
        uint32_t rb = (2u*head_dim + 31)/32;
        float aw_hmaxp[2] = {0, 0};
        int   ea8p[2] = {0, 0};
        for (int p = 0; p < 2; p++) {
            // Stage this position's K into SP_K[0..) and V into SP_V (the qkr list and
            // kv-pack operate on the fixed regions).
            if (p == 1) {
                volatile uint32_t *d = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_K);
                volatile uint32_t *s2 = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_K
                                                              + (uint32_t)kv_dim*2);
                for (int i = 0; i < kv_dim/2; i++) d[i] = s2[i];
                volatile uint32_t *dv = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_V);
                volatile uint32_t *sv = (volatile uint32_t *)(SCRATCHPAD_BASE + vst[1]);
                for (int i = 0; i < kv_dim/2; i++) dv[i] = sv[i];
                volatile uint32_t *dq = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_Q);
                volatile uint32_t *sq = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_Q
                                                              + (uint32_t)q_dim*2);
                for (int i = 0; i < q_dim/2; i++) dq[i] = sq[i];
            }
            perf_enter(OP_QKNORM);
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE
                            + (rt->chan_base_beats + (uint64_t)(pos + p)*rb)*32,
                         rb, SP_TAB);
            int epq = seq_eps_int(eq_r[p]), epk = seq_eps_int(ek_r[p]);
#ifdef BONSAI_VEC_LOOPS2
            seq_patch((uint32_t)seq_qkr_di_qn, 7, (uint32_t)epq);
            seq_patch((uint32_t)seq_qkr_di_kn, 7, (uint32_t)epk);
#else
            for (int hh = 0; hh < n_head + n_kv_head; hh++)
                seq_patch(2*hh, 7, (hh < n_head) ? epq : epk);
#endif
            seq_run(0, seq_qkr_count);
            perf_exit(OP_QKNORM);
            perf_enter(OP_KVQUANT);
            if (p == 1)
                ek_capture(l);
            for (int hh = 0; hh < n_kv_head; hh++) {   // Pack every head first...
                kv_pack_row_res(l, hh, 0, SP_K + hh*head_dim*2, E_NORM);
                kv_pack_row_res(l, hh, 1, SP_V + hh*head_dim*2, ev_r[p]);
            }
            for (int hh = 0; hh < n_kv_head; hh++) {   // ...then hand the port to the engine.
                kv_store_row_res(l, hh, 0, pos + p);
                kv_store_row_res(l, hh, 1, pos + p);
            }
            perf_exit(OP_KVQUANT);
            attn_seq2_layer(l, pos + p, aw_cp[p], &aw_hmaxp[p]);
            // Head outputs -> int8 at the position's x8 slot.
            perf_enter(OP_QUANT8);
            int ea8 = 0;
            if (aw_hmaxp[p] > 0) { frexpf(aw_hmaxp[p], &ea8); ea8 -= 7; }
            for (int hh = 0; hh < n_head; hh++) {
                int32_t Mh = (int32_t)lroundf(ldexpf(aw_cp[p][hh], 15 - ea8));
                if (Mh > 32767) Mh = 32767;
                if (Mh < 0)     Mh = 0;
                int32_t sa2[5] = {(int32_t)(SP_H + hh*head_dim*2), 0,
                                  (int32_t)(SP_X + 0x800 + (uint32_t)p*q_dim
                                            + hh*head_dim),
                                  head_dim/16, Mh};
                vec_run(PROG_SCALE8, sa2, 5);
            }
            perf_exit(OP_QUANT8);
            ea8p[p] = ea8;
        }
        // Output projection: one stream, both positions.
        {
            const struct bons_entry *eo = model_entry(ROLE_ATTN_OUT, l);
            // x pair sits above the attention scratch (position 1's head quant8 writes
            // SP_X[0, head_dim) while position 0's staged vector must survive).
            model_matvec2(eo, SP_X + 0x800, SP_Y, (uint16_t)eo->rows);
            for (int p = 0; p < 2; p++) {
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kk = op_norm_i32_sp_m(SP_Y + (uint32_t)p*4*eo->rows,
                                          SP_D + (uint32_t)p*n_embd*2, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = ea8p[p] - (int)eo->bias + (int)eo->rshift + kk;
                perf_enter(OP_RESID_ADD);
                ex_st[p] = op_resid_add_sp(xr[p], ex_st[p],
                                           SP_D + (uint32_t)p*n_embd*2, ed, n_embd,
                                           dmax, &xa_st[p]);
                perf_exit(OP_RESID_ADD);
            }
        }
        // FFN: norm/quant per position, gate/up/down shared streams.
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            int fx8[2];
            for (int p = 0; p < 2; p++) {
                perf_enter(OP_RMSNORM);
                op_rmsnorm_sp(xr[p], ex_st[p], ffn_norm[l],
                              model_entry_kind(ROLE_FFN_NORM, l, KIND_NORM16), n_embd, SP_H);
                perf_exit(OP_RMSNORM);
                perf_enter(OP_QUANT8);
                fx8[p] = E_NORM + op_quant8_sp(SP_H, SP_X + (uint32_t)p*n_embd, n_embd);
                perf_exit(OP_QUANT8);
            }
            int eg[2], eu[2];
            model_matvec2(eg_t, SP_X, SP_Y, (uint16_t)eg_t->rows);
            perf_enter(OP_NORM_I32);
            for (int p = 0; p < 2; p++) {
                int kg = op_norm_i32_sp(SP_Y + (uint32_t)p*4*eg_t->rows,
                                        SP_G + (uint32_t)p*ffn_dim*2, ffn_dim);
                eg[p] = fx8[p] - (int)eg_t->bias + (int)eg_t->rshift + kg;
            }
            perf_exit(OP_NORM_I32);
            model_matvec2(eu_t, SP_X, SP_YU, (uint16_t)eu_t->rows);
            perf_enter(OP_NORM_I32);
            for (int p = 0; p < 2; p++) {
                int ku = op_norm_i32_sp(SP_YU + (uint32_t)p*4*eu_t->rows,
                                        SP_U + (uint32_t)p*ffn_dim*2, ffn_dim);
                eu[p] = fx8[p] - (int)eu_t->bias + (int)eu_t->rshift + ku;
            }
            perf_exit(OP_NORM_I32);
            int es8[2];
            for (int p = 0; p < 2; p++) {
                perf_enter(OP_SILU_MUL);
                op_silu_mul_sp(SP_G + (uint32_t)p*ffn_dim*2, eg[p],
                               SP_U + (uint32_t)p*ffn_dim*2, eu[p], ffn_dim);
                perf_exit(OP_SILU_MUL);
                perf_enter(OP_QUANT8);
                int ss = op_quant8_sp(SP_G + (uint32_t)p*ffn_dim*2,
                                      SP_X + (uint32_t)p*ffn_dim, ffn_dim);
                perf_exit(OP_QUANT8);
                es8[p] = eg[p] + eu[p] + 15 + ss;
            }
            model_matvec2(ed_t, SP_X, SP_Y, (uint16_t)ed_t->rows);
            for (int p = 0; p < 2; p++) {
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kd = op_norm_i32_sp_m(SP_Y + (uint32_t)p*4*ed_t->rows,
                                          SP_D + (uint32_t)p*n_embd*2, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = es8[p] - (int)ed_t->bias + (int)ed_t->rshift + kd;
                perf_enter(OP_RESID_ADD);
                ex_st[p] = op_resid_add_sp(xr[p], ex_st[p],
                                           SP_D + (uint32_t)p*n_embd*2, ed, n_embd,
                                           dmax, &xa_st[p]);
                perf_exit(OP_RESID_ADD);
            }
        }
    }
    n_pos += 2;

    // Final norm + LM head per position (argmax tap; vocab-padding edge bails out).
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head)
        head = model_entry(ROLE_TOK_EMB, 0);
    uint32_t rpc  = head->rows/n_ch;
    uint32_t hrb  = bons_row_beats(head->cols, head->kind, head->group);
    int am[2];
    for (int p = 0; p < 2; p++) {
        perf_enter(OP_RMSNORM);
        op_rmsnorm_sp(xr[p], ex_st[p], out_norm,
                      model_entry_kind(ROLE_OUT_NORM, 0, KIND_NORM16), n_embd, SP_H);
        perf_exit(OP_RMSNORM);
        perf_enter(OP_QUANT8);
        op_quant8_sp(SP_H, SP_X, n_embd);
        perf_exit(OP_QUANT8);
        perf_enter(OP_LMHEAD);
        struct engine_op op = {0};
        op.kind   = ENGINE_KIND_TERNARY;
        op.rows   = rpc;
        op.cols   = head->cols;
        op.wbase  = BONSAI_CHAN_WEIGHT_BASE + head->chan_base_beats*32;
        op.wlen   = rpc*hrb;
        op.sbeats = bons_scale_beats(head->cols, head->group);
        op.cbeats = bons_code_beats(head->cols, head->kind);
        op.bias   = head->bias;
        op.rshift = head->rshift;
        op.xsrc   = SP_X;
        op.dst    = SP_Y;
        op.supw   = 1;
        engine_run(&op);
        uint32_t idx = engine_csr_argmax_idx_read();
        perf_exit(OP_LMHEAD);
        if (idx >= H->vocab)
            return -1;
        am[p] = (int)idx;
    }
    *am0 = am[0];
    *am1 = am[1];
    return 0;
}
#endif

// Forward one token; returns argmax token id (only meaningful when wanted). ------------------

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
// Token-invariant words of the attention list, patched ONCE per forward instead of once
// per layer per head (F0-2 patch hygiene, 2026-08-29): rowsw/wlen (scores + AXPY) and the
// softmax block count depend on pos only, and the score-slot padding survives across heads
// and layers because only the scores op writes SP_SC (exactly cnt slots). Same values as
// before, written 28x less often - uncached scratchpad stores are ~25-70 cycles each.
// Console toggle (kovl): the qkr list length changes, so every downstream base shifts.
void infer_kv_pipe(int on)
{
    kv_pipe = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
}
int infer_kv_pipe_state(void) { return kv_pipe_active; }

// Console toggle (kvq): re-emit the redundant per-slot KV-pack quant8 descriptors. Same
// base shifts as kovl, so the same five rebuilds.
void infer_kvq_redundant(int on)
{
#ifdef BONSAI_VEC_LOOPS2
    kvq_redundant = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
#else
    (void)on;
#endif
}
int infer_kvq_redundant_state(void) { return kvq_redundant; }

// Console toggle (kvms): in-list multi-store vs 16 per-slot stores. Same rebuilds as kovl.
void infer_kv_mstore(int on)
{
#ifdef CSR_ENGINE_CSR_REP_ADDR
    kv_mstore = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
#else
    (void)on;
#endif
}
int infer_kv_mstore_state(void) { return kv_mstore_active; }

// Console toggle (qkro): qkr list order. Same base shifts as kovl (count unchanged, but
// the norm indices move), so the same five rebuilds.
void infer_qkr_reorder(int on)
{
    qkr_reorder = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
}
int infer_qkr_reorder_state(void) { return qkr_reorder; }
void infer_qfix(int on)
{
    qfix_use = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_qkr(); seq_build_attn(); seq_build_ffn(); seq_build_scale8(); seq_build_qkv();
}
int infer_qfix_state(void) { return qfix_use; }

// Console toggle (fsplit): same base shifts as the other list toggles.
void infer_ffn_split(int on)
{
    ffn_split = on ? 1 : 0;
    seq_ffn_base = -1;
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
}
int infer_ffn_split_state(void) { return ffn_split; }

// Console toggle (aovl): rebuild the three dependent lists in either layout.
void infer_attn_pipe(int on)
{
    attn_pipe_req = on ? 1 : 0;
    attn_pipe = attn_pipe_req;
    seq_ffn_base = -1;
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
}
int infer_attn_pipe_state(void) { return attn_pipe_active; }

// Per-layer attention slot patches, extracted so they can run UNDER the qkr list
// (per-cycle sideband slot): the sequencer is executing qkr descriptors and never
// dispatches attention ones, so patching them mid-list is safe; every input
// (kv_ekmax ratchet, region addresses) is CPU-side state settled before the list
// started. attn_prepatched tells the attention section its work is already done.
static int attn_prepatched;
static int ffn_prepatched;
int eovl_use = 1; // C2-2: fold head j under the still-running attention list (needs RSLOT).
int lovl_use = 1; // C2-3: next forward's list prep posts under the LM-head argmax stream.
int qkvl_use = 1; // QKV projections + norms as one sequencer list (seq_build_qkv).
int qres_use = 1; // Resident qk-norm rows (SP_QNW/SP_KNW) instead of per-layer fetches.
static int qkn_resident;   // 1 once every layer's q/k norm rows sit in SP_QNW/SP_KNW.
int atrace_use;   // Diagnostic: =N traces layer 0's attention list (per-descriptor deltas)
                  // on the Nth generated token, one-shot, serial folds for that layer.
// List-relative dcount ("descriptors completed") threshold after which head j's fold
// inputs (softmax8's AWP8 self-spill and spill(j)'s AWAM) are certainly written: one
// past norm(j), whose producers are all earlier vec descriptors in the same list.
static inline uint32_t attn_fold_ready(int j, int chunk) {
    if (!attn_pipe_active) return (uint32_t)(j*5 + 5);
    if (j >= chunk - 2)    return (uint32_t)(1 + (j + 1)*5 + 2); // last-group/epilogue slots.
    return (uint32_t)(1 + (j + 1)*5 + 3);
}
// Per-layer FFN list rebuild (three engine descriptors + their flag words), extracted so
// it can POST under the qkr list in the povl window (all inputs are layer-known: model
// entries and static scratchpad addresses; only the silu A4 imm needs ex8 and stays at
// FFN time). The sequencer-fetch interlock orders these writes before the FFN list runs.
static inline void ffn_patch_lw(uint32_t di, const struct engine_op *mo)
{
    seq_patch(di, 4, mo->wbase);
    seq_patch(di, 7, mo->bias | ((uint32_t)mo->rshift << 16) | ((uint32_t)mo->rshift2 << 22));
}

static void ffn_prep_layer(int l)
{
    // Only wbase (4) and bias|rshift (7) change with the layer; the 13 shape words are
    // per-role constants and the flag patches are build-time. Full images once per list
    // build, then 6 posted words per layer instead of 45. (Measured neutral on v5d -
    // absorbed downstream, an I$ artefact; re-measured on v7c with the 32 KB I$.)
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
int prep_ovl = 1; // povl console toggle: patch-under-list overlap (needs BONSAI_SP_SB_SLOT).
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

static void attn_list_prep_token(int pos)
{
    if (seq_attn_heads <= 0)
        return;
    if (pos == attn_prep_pos)
        return; // Already posted under the previous forward's LM-head stream (lovl).
    int cnt  = pos + 1;
    int padn = SC_PADN(cnt);
    // The deep layout needs the padded score count to fit the second buffer's REAL
    // capacity (SC_B_CAP_BYTES, see above). Beyond it fall back to the plain list, and
    // come back to the deep one as soon as a new generation fits again - both directions
    // shift the FFN/scale8/qkv bases with the prologue, so all four lists rebuild.
    int want_pipe = attn_pipe_req && ((uint32_t)(cnt + padn)*4u <= SC_B_CAP_BYTES);
    if (want_pipe != attn_pipe) {
        attn_pipe = want_pipe;
        seq_ffn_base = -1;
        seq_build_attn(); seq_build_ffn(); seq_build_scale8(); seq_build_qkv();
    }
    volatile int32_t *sc32 = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
    for (int i2 = 0; i2 < padn; i2++)
        sc32[cnt + i2] = (int32_t)0x80000000;
    if (attn_pipe_active) {
        volatile int32_t *sc32b = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC + 0x3800);
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
#endif

static int fold_nrep_sh = 1; // log2(n_head/n_kv_head) when a power of two (set at init).
// The per-head fold from its two hardware results (softmax8's p8 sum, the AXPY absmax),
// in whichever form the output projection will consume (ifold_use). ONE producer for
// both the list path (spilled values, attn_fold_head) and the seq-off op-by-op path:
// the latter used to fill only the float fold while the consumer read the integer one
// under the default ifold_use, so `seq off` ran the projection on stale or zero fold
// state (2026-09-07, from the external review).
static void attn_fold_vals(int l, int hh, uint32_t P8, uint32_t am,
                           float *aw_c, float *aw_hmax_p)
{
    int kvh = hh >> fold_nrep_sh;
    int Bv = 22 - kv_ekmax[kv_ek_idx(l, kvh, 1)];   // kv_scores_br's B; R unused here.
    int Rpre = axpy_rpre_c(P8 ? P8 : 1);
    int k2 = (bitlen32(am) > 14) ? bitlen32(am) - 14 : 0; // == the >16383 scan, via clz.
    int E1 = Rpre - Bv + k2;
    if (ifold_use) {
        // Integer emulation of the float32 fold (F1-5): same bits, no soft-float.
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
    } else {
        float c2 = ldexpf(1.0f/(float)(P8 ? P8 : 1), E1);
        aw_c[hh] = c2;
        float hm2 = (float)(am >> k2)*c2;
        if (hm2 > *aw_hmax_p) *aw_hmax_p = hm2;
    }
}
static void attn_fold_head(int l, int hh, float *aw_c, float *aw_hmax_p)
{
    uint32_t P8 = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWP8 + (uint32_t)hh*32);
    uint32_t am = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWAM + (uint32_t)hh*32);
    attn_fold_vals(l, hh, P8, am, aw_c, aw_hmax_p);
}

// Per-layer norm16 entries, resolved once at init into .bss - which lives in the 8 KB
// tightly-coupled SRAM. model_entry_kind() walks the entry cache and the tensor table,
// both in HBM-backed main_ram behind a 2-way L1 with no L2, so each of these lookups
// was a couple of ~150-cycle misses, 5+ times a layer (2026-09-04).
static const struct bons_entry *nrm_attn16[NRM_MAXL], *nrm_ffn16[NRM_MAXL], *nrm_out16;
static void norm16_cache_build(void)
{
    for (int l = 0; l < n_layer && l < NRM_MAXL; l++) {
        nrm_attn16[l] = model_entry_kind(ROLE_ATTN_NORM, l, KIND_NORM16);
        nrm_ffn16[l]  = model_entry_kind(ROLE_FFN_NORM,  l, KIND_NORM16);
    }
    nrm_out16 = model_entry_kind(ROLE_OUT_NORM, 0, KIND_NORM16);
}

// The norm16 row the layer after l opens with (attention norm), or the final norm.
static const struct bons_entry *next_norm16(int l)
{
    return (l + 1 < n_layer) ? nrm_attn16[l + 1] : nrm_out16;
}

uint32_t infer_forwards; // Forwards run since boot (bench reports cycles per forward too).
static int forward(int token, int want_logits)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    infer_forwards++;
    if (tprof_armed && want_logits && --tprof_armed == 0)
        tprof_begin(); // armed=N traces the Nth generated token; disarms itself.
    rq8_prefetched = 0; // A logits-less forward leaves the out-norm row prefetched; drop it.
    perf_enter(OP_EMBED);
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
    if (attn_all_hw()) {
        ex_state = embed_row_bfp(emb, token, SP_XR);
        x_amax   = X_AMAX_AFTER_EMBED;
        perf_exit(OP_EMBED);
        goto embed_done;
    }
#endif
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
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
embed_done: ;
#endif

    int   kv_dim = n_kv_head*head_dim;
    int   q_dim  = n_head*head_dim;
    float scale  = 1.0f/sqrtf((float)head_dim);
    int   pos    = n_pos;

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
    int attn_hw = attn_all_hw();
    if (attn_hw) {
        // Rope table row for this position ([c14 | s14], Q14, from the image).
        const struct bons_entry *rt = model_entry_kind(ROLE_ROPE_TAB, 0, KIND_NORM16);
        uint32_t rb = (2u*head_dim + 31)/32;
        engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + (rt->chan_base_beats + (uint64_t)pos*rb)*32,
                     rb, SP_TAB);
#ifdef CSR_SEQ_BASE
        if (seq_use) {
            perf_enter(OP_SCORES);
            attn_list_prep_token(pos);
            perf_exit(OP_SCORES);
        }
#endif
    }
#endif

    for (int l = 0; l < n_layer; l++) {
        if (tprof_active)
            tprof_event((3u << 28) | ((uint32_t)l << 8));
        // Attention block.
        int sx8 = op_rmsnorm_q8_sp(SP_XR, ex_state, attn_norm[l],
                                   nrm_attn16[l],
                                   n_embd, SP_H, SP_X);
        int eq_res = 0, ev_res = 0;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
        if (attn_hw) {
            // q/k/v: engine matvec -> vec norm_i32 -> int16 resident; qk-norm + rope on
            // the vector unit; K/V packed and stored to HBM. No CPU floats anywhere.
            const struct bons_entry *eqt = model_entry(ROLE_ATTN_Q, l);
            const struct bons_entry *ekt = model_entry(ROLE_ATTN_K, l);
            const struct bons_entry *evt = model_entry(ROLE_ATTN_V, l);
            perf_enter(OP_NORM_I32);
            int kq, kk2, kv2;
#ifdef CSR_SEQ_BASE
            if (qkvl_use && seq_use && seq_qkv_base >= 0 && op_on_hw(OP_NORM_I32)) {
                // One sequencer run for q/k/v + norms (seq_build_qkv). The engine words
                // were posted under the previous FFN list (or the LM-head stream); the
                // call below is a no-op then. kq/kk/kv from the spilled absmaxes, the
                // same bitlen-14 the op path derives - byte-identical.
                qkv_prep_layer(l);
                seq_run((uint32_t)seq_qkv_base, 10);
                uint32_t aq = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP);
                uint32_t ak = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP + 32);
                uint32_t av = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_QKSP + 64);
                kq  = (bitlen32(aq) > 14) ? bitlen32(aq) - 14 : 0;
                kk2 = (bitlen32(ak) > 14) ? bitlen32(ak) - 14 : 0;
                kv2 = (bitlen32(av) > 14) ? bitlen32(av) - 14 : 0;
            } else
#endif
#if defined(CSR_ENGINE_CSR_ABSMAX_ADDR)
            if (ovl_use && op_on_hw(OP_NORM_I32)) {
                // Overlap: each norm runs UNDER the next projection's stream. Distinct
                // result buffers inside the SP_Y region (q 16 KB max, then k/v) keep the
                // in-flight dst disjoint from the norm's src; each op's absmax is captured
                // before the next submit. Same k's, same programs - byte-identical.
                struct engine_op mo = {0};
                model_matvec(eqt, SP_X, SP_Y);
                uint32_t aq = engine_csr_absmax_read();
                model_matvec_op(ekt, SP_X, SP_Y + 0x4000, &mo);
                uint32_t dk = engine_submit(&mo);
                kq = op_norm_i32_sp_a(SP_Y, SP_Q, q_dim, aq, 0);
                perf_enter(OP_ENGINE_WAIT);
                engine_join(dk);
                perf_exit(OP_ENGINE_WAIT);
                uint32_t ak = engine_csr_absmax_read();
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
            uint32_t qt[8]; int qtn = 0;
            // qtrace piggybacks on the atrace arm: fires at layer 1 one forward early.
            int qtr = (atrace_use == 1 && l == 1);
            if (qtr) qt[qtn++] = perf_now();
            const struct bons_entry *qn16 = model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16);
            const struct bons_entry *kn16 = model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16);
            uint32_t dqn = 0;
            int qres = 0;
#ifdef BONSAI_VEC_LOOPS2
            qres = qres_use && (uint32_t)n_layer*head_dim*2 <= 0x1C00 && qn16 && kn16;
            if (qres && !qkn_resident) {
                // One-time residency: every layer's q/k norm row (8 beats each).
                for (int ll = 0; ll < n_layer; ll++) {
                    const struct bons_entry *q = model_entry_kind(ROLE_ATTN_Q_NORM, ll, KIND_NORM16);
                    const struct bons_entry *k = model_entry_kind(ROLE_ATTN_K_NORM, ll, KIND_NORM16);
                    engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + q->chan_base_beats*32,
                                 q->chan_len_beats, SP_QNW + (uint32_t)ll*head_dim*2);
                    engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + k->chan_base_beats*32,
                                 k->chan_len_beats, SP_KNW + (uint32_t)ll*head_dim*2);
                }
                qkn_resident = 1;
            }
#endif
            if (qres) {
                seq_patch((uint32_t)seq_qkr_di_qn, 3, SP_QNW + (uint32_t)l*head_dim*2); // rmsnorm_h A1 (q).
                seq_patch((uint32_t)seq_qkr_di_kn, 3, SP_KNW + (uint32_t)l*head_dim*2); // rmsnorm_h A1 (k).
            } else if (fovl_use) {  // fovl: the q-norm row streams under the eps/KV prep below.
                dqn = norm_fetch_submit(qn16);
            } else {
                engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + qn16->chan_base_beats*32,
                             qn16->chan_len_beats, SP_FET);
                engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                             kn16->chan_len_beats, SP_FET2);
            }
#ifdef CSR_SEQ_BASE
            {
                // One sequencer run replaces the 2*(n_head+n_kv) per-head dispatches;
                // only the eps args change per layer (descriptor word 7 = A5).
                if (qtr) qt[qtn++] = perf_now(); // [1] fetches + entry lookups done.
                int epq = seq_eps_int(eq_r), epk = seq_eps_int(ek_r);
#ifdef BONSAI_VEC_LOOPS2
                seq_patch((uint32_t)seq_qkr_di_qn, 7, (uint32_t)epq);   // rmsnorm_h A5 (q heads).
                seq_patch((uint32_t)seq_qkr_di_kn, 7, (uint32_t)epk);   // rmsnorm_h A5 (k heads).
#else
                for (int hh = 0; hh < n_head + n_kv_head; hh++)
                    seq_patch(2*hh, 7, (hh < n_head) ? epq : epk);
#endif
                if (kv_pipe_active) {
                    // In-list stores: the scale beats must exist before the list runs
                    // (the store streams them), and each store's target row moves per
                    // layer/token. The exponent ratchet is identical bookkeeping to the
                    // post-list placement it replaces.
                    // Only the target address is per (layer, slot). The beat count is a
                    // model constant once kv_rb is known, and the channel is per TOKEN -
                    // both used to be rewritten for all 16 slots every layer, i.e. 32 of
                    // the 48 writes here were rewriting values that were already there.
                    // Posted writes under a running list cost ~80 cycles each (v8 qtrace),
                    // so they were ~2.6k cycles a layer. Note this removes writes, not the
                    // arithmetic between them: that arithmetic is what keeps the
                    // sequencer's descriptor fetch alive (see the patch-write wall).
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
                if (fovl_use && !qres) {
                    engine_join(dqn);
                    engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                                 kn16->chan_len_beats, SP_FET2);
                }
                if (qtr) qt[qtn++] = perf_now(); // [2] kv scale-preps + store patches done.
#ifdef BONSAI_SP_SB_SLOT
                if (prep_ovl && seq_use && seq_attn_heads >= n_head
                    && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV)) {
                    // Overlap (sideband slot): the attention list's per-layer patches
                    // run while the sequencer executes the qkr list. Without the slot
                    // the first patch would simply stall until the list finished.
                    seq_list_start(0, seq_qkr_count);
                    attn_patch_slots(l, 0, n_head, 0);
                    attn_prepatched = 1;
                    if (qtr) qt[qtn++] = perf_now(); // [3] attn_patch_slots done.
                    if (seq_ffn_use && seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                        && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
                        ffn_prep_layer(l); // C2-1c: the FFN rebuild rides the same window.
                        ffn_prepatched = 1;
                    }
                    if (qtr) qt[qtn++] = perf_now(); // [4] ffn_prep_layer done.
                    seq_join();
                    if (qtr) {
                        qt[qtn++] = perf_now(); // [5] qkr list joined.
                        printf("QTRACE fetch+kv %lu %lu attnpatch %lu ffnprep %lu join %lu\n",
                               (unsigned long)(qt[1]-qt[0]), (unsigned long)(qt[2]-qt[1]),
                               (unsigned long)(qt[3]-qt[2]), (unsigned long)(qt[4]-qt[3]),
                               (unsigned long)(qt[5]-qt[4]));
                    }
                } else
#endif
                seq_run(0, seq_qkr_count);
            }
            perf_exit(OP_QKNORM);
#else
            if (fovl_use && !qres) {
                engine_join(dqn);
                engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                             kn16->chan_len_beats, SP_FET2);
            }
            for (int hh = 0; hh < n_head; hh++)
                vec_rmsnorm_head(SP_Q + hh*head_dim*2, eq_r, SP_FET, head_dim);
            for (int hh = 0; hh < n_kv_head; hh++)
                vec_rmsnorm_head(SP_K + hh*head_dim*2, ek_r, SP_FET2, head_dim);
            perf_exit(OP_QKNORM);
            perf_enter(OP_ROPE);
            for (int hh = 0; hh < n_head; hh++)
                vec_rope_head(SP_Q + hh*head_dim*2, head_dim);
            for (int hh = 0; hh < n_kv_head; hh++)
                vec_rope_head(SP_K + hh*head_dim*2, head_dim);
            perf_exit(OP_ROPE);
#endif
            perf_enter(OP_KVQUANT);
            for (int hh = 0; hh < n_kv_head && !kv_pipe_active; hh++) {
#ifdef CSR_SEQ_BASE
                // ...the quant8s already ran inside the qkr list (F0-4); prep only.
                // (With kovl the prep AND the stores already ran with the list.)
                kv_pack_prep_res(l, hh, 0, E_NORM);
                kv_pack_prep_res(l, hh, 1, ev_res);
#else
                kv_pack_row_res(l, hh, 0, SP_K + hh*head_dim*2, E_NORM);
                kv_pack_row_res(l, hh, 1, SP_V + hh*head_dim*2, ev_res);
#endif
            }
#ifdef CSR_ENGINE_CSR_REP_ADDR
            // The !kv_pipe_active guard was on the #else twin below but not here, so with
            // the in-list NOWAIT stores on (the shipping default) BOTH paths ran and every
            // KV row was written to HBM twice per layer - same bytes to the same addresses,
            // so it cost time and never showed up as a wrong token (2026-09-06).
            if (!kv_pipe_active)
            {   // ...then ONE multi-packet store for all 2*n_kv_head slots (F1-1): the
                // (kvh, kind) regions of a layer are consecutive (kv_region beats apart)
                // and the staging slots 0x200 B apart, so one op scatters them all.
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
            eq_res = E_NORM;
            (void)eq_res;
            goto attn_scores;
        }
#endif
        {
        float sx = ldexpf(1.0f, E_NORM + sx8);
        matvec_rows_f(model_entry(ROLE_ATTN_Q, l), 0, q_dim/n_ch,  sx, q);
        matvec_rows_f(model_entry(ROLE_ATTN_K, l), 0, kv_dim/n_ch, sx, k);
        matvec_rows_f(model_entry(ROLE_ATTN_V, l), 0, kv_dim/n_ch, sx, v);
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
            // Per-head scales, not one per token: the head spread of K is large (16x at
            // layer 0 of the 1.7B, head 4 peaking at 311 against head 6 at 18.9), so a single
            // tensor scale leaves the quiet heads under 3 bits and wrecks their scores.
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
#ifdef BONSAI_HAS_KVSTORE
        // Dual-KV: also append Q8 row packets to HBM for the engine scores path.
        if (op_on_hw(OP_KVQUANT)) {
            for (int h = 0; h < n_kv_head; h++) {
                kv_store_row(l, h, 0, pos, k + h*head_dim);
                kv_store_row(l, h, 1, pos, v + h*head_dim);
            }
        }
#endif
        perf_exit(OP_KVQUANT);

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
attn_scores: ;
        // Resident attention output: per-head int16 in SP_H + per-head value scales,
        // aligned to a common ea8 and scale8-ed into SP_X after the head loop.
        float aw_c[32];
        float aw_hmax = 0;
        aw_hvalid = 0; // Both fold forms start empty for every path below (the list paths
                       // reset it again where they fold; the op-by-head path relies on this).
#endif
        // Attention (CPU, GQA).
        int n_rep = n_head/n_kv_head;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
        if (seq_use && attn_hw && seq_at2_kvchunk > 0
            && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV)) {
            attn_seq2_layer(l, pos, aw_c, &aw_hmax);
            goto attn_heads_done;
        }
        if (seq_use && attn_hw && seq_attn_heads > 0
            && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV)) {
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
            if (atrace_use && l == 0 && !chunked && --atrace_use == 0) {
                // One-shot per-descriptor wall-clock of the attention list: run it under
                // seq_run_trace (tight dcount poll, no folds) and fold serially after.
                // Numerics identical - only this layer's fold overlap is lost.
                if (!attn_prepatched)
                    attn_patch_slots(l, 0, n_head, 0);
                attn_prepatched = 0;
                static uint32_t ats[192];
                int an = seq_run_trace((uint32_t)seq_attn_base,
                                       (uint32_t)attn_list_len(n_head), ats, 192);
                printf("ATRACE %d marks (deltas):", an);
                for (int i = 1; i < an; i++)
                    printf(" %lu", (unsigned long)(ats[i] - ats[i-1]));
                printf("\n");
                perf_exit(OP_SCORES);
                perf_enter(OP_WV);
                aw_hvalid = 0;
                for (int hh = 0; hh < n_head; hh++)
                    attn_fold_head(l, hh, aw_c, &aw_hmax);
                perf_exit(OP_WV);
                goto attn_heads_done;
            }
#ifdef BONSAI_SP_SB_RSLOT
            if (eovl_use && !chunked) {
                // C2-2: fold head j while heads j+2.. still execute. The read slot lets
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
                // etrace: one-shot split of this loop into wait vs fold time (piggybacks
                // the atrace arm at layer 2, one forward early - like qtrace).
                int etr = (atrace_use == 1 && l == 2);
                uint32_t et0 = etr ? perf_now() : 0, et_fold = 0, et_spins = 0;
                for (int hh = 0; hh < n_head; hh++) {
                    uint32_t ready = attn_fold_ready(hh, n_head), spins = 0;
                    for (;;) {
                        uint32_t st = seq_status_read();
                        tprof_dcount(st);
                        if (!(st & 1) || ((st >> 8) >= ready))
                            break;
                        if (++spins > 4000000u) break; // seq_join below reports the wedge.
                    }
                    et_spins += spins;
                    uint32_t f0 = etr ? perf_now() : 0;
                    attn_fold_head(l, hh, aw_c, &aw_hmax);
                    if (etr) et_fold += perf_now() - f0;
                }
                perf_exit(OP_WV);
                perf_enter(OP_SCORES);
                uint32_t j0 = etr ? perf_now() : 0;
                seq_join();
                if (etr)
                    printf("ETRACE loop %lu fold16 %lu spins %lu join %lu\n",
                           (unsigned long)(j0 - et0), (unsigned long)et_fold,
                           (unsigned long)et_spins, (unsigned long)(perf_now() - j0));
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
                attn_fold_head(l, hh, aw_c, &aw_hmax);
            perf_exit(OP_WV);
            goto attn_heads_done;
        }
#endif
        for (int hh = 0; hh < n_head; hh++) {
            int kvh = hh/n_rep;
            const float *qh = q + hh*head_dim;
            float smax = -1e30f;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
            if (op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV)) {
                // Fused resident path: scores (engine) -> softmax8 (vec, p8 + exact sum)
                // -> weighted-V (engine AXPY) -> cross-channel reduce (vec). Scores never
                // touch the CPU; the 1/sum fold is one float divide per head.
                float *ah = attn + hh*head_dim;
                perf_enter(OP_SCORES);
                int eq;
                if (attn_hw) {
                    vec_quant8_head(SP_Q + hh*head_dim*2, SP_X, head_dim);
                    eq = E_NORM + 8;
                } else {
                    eq = quantize_f_pow2(qh, head_dim, SP_X);
                }
                int Bk, Rk;
                kv_scores_br(l, kvh, 0, &Bk, &Rk);
                // This op is where the card wedges once rows >= 2, i.e. from position 9
                // at 8 channels. Not the ragged shape: padding so every channel carries the
                // same count (rem = 0) still wedges, with all channels then delivering
                // 10/10 beats. All the data arrives; the results stop retiring. See
                // docs/bringup.md before spending time on it.
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
                    am = engine_csr_absmax_read(); // AXPY partials tracked too.
                int k2 = 0;
                k2 = (bitlen32(am) > 14) ? bitlen32(am) - 14 : 0; // == the >16383 scan, via clz.
                int32_t nargs[5] = {(int32_t)SP_SC, 0,
                                    (int32_t)(SP_H + hh*head_dim*2), head_dim/16, k2};
                vec_run(PROG_NORM_I32, nargs, 5);
                // Both fold forms, same producer as the list path (see attn_fold_vals):
                // Bv/Rpre/k2 there are exactly the ones computed above.
                attn_fold_vals(l, hh, (uint32_t)P8, (uint32_t)am, aw_c, &aw_hmax);
                (void)Rpre; (void)Bv; (void)ah;
                perf_exit(OP_WV);
                continue;
            }
#endif
            perf_enter(OP_SCORES);
#ifdef BONSAI_HAS_KVSTORE
            if (op_on_hw(OP_SCORES)) {
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
            } else
#endif
            { // The whole CPU fallback. This brace used to be missing: the else bound only
              // to the context-limit check below and the CPU score loop ran after HARDWARE
              // scores too, overwriting them from a k_cache the HW path never fills and
              // skipping the bounds check (2026-09-07, from the external review). Only the
              // mixed hwops configurations (HW scores, CPU softmax/WV) reached it.
            if (pos >= cpu_kv_ctx) {
                // The CPU fallback cache is capped (see CPU_KV_CTX); results past it would
                // read out of bounds. The HW path carries full context - switch back.
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
#ifdef BONSAI_HAS_KVSTORE
            if (op_on_hw(OP_WV)) {
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
#endif
            for (int d = 0; d < head_dim; d++) ah[d] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *vt = v_cache + ((uint64_t)l*cpu_kv_ctx + t)*kv_dim + kvh*head_dim;
                float p = scores[t]*inv*v_scale[((uint64_t)l*cpu_kv_ctx + t)*n_kv_head + kvh];
                for (int d = 0; d < head_dim; d++) ah[d] += p*vt[d];
            }
            perf_exit(OP_WV);
        }
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
attn_heads_done: ;
#endif

        // Output projection + residual (resident).
        {
            const struct bons_entry *eo = model_entry(ROLE_ATTN_OUT, l);
            int ea8;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
            if (attn_hw) {
                perf_enter(OP_QUANT8);
                ea8 = 0;
                if (ifold_use) {
                    if (aw_hvalid) ea8 = aw_he2 + 24 - 7;   // frexpf exponent of max hm, -7.
                } else {
                    if (aw_hmax > 0) { frexpf(aw_hmax, &ea8); ea8 -= 7; }
                }
                for (int hh = 0; hh < n_head; hh++) {
                    int32_t Mh;
                    if (ifold_use) {
                        Mh = attn_mh_int(aw_qm[hh], aw_qe[hh] + 15 - ea8);
                    } else {
                        Mh = (int32_t)lroundf(ldexpf(aw_c[hh], 15 - ea8));
                        if (Mh > 32767) Mh = 32767;
                        if (Mh < 0)     Mh = 0;
                    }
#ifdef CSR_SEQ_BASE
                    if (seq_use && seq_scale8_base >= 0) {
                        seq_patch((uint32_t)(seq_scale8_base + hh), 6, (uint32_t)Mh);
                        continue;                       // One list run below (F0-6).
                    }
#endif
                    int32_t sa2[5] = {(int32_t)(SP_H + hh*head_dim*2), 0,
                                      (int32_t)(SP_X + hh*head_dim), head_dim/16, Mh};
                    vec_run(PROG_SCALE8, sa2, 5);
                }
#ifdef CSR_SEQ_BASE
                if (seq_use && seq_scale8_base >= 0)
                    seq_run((uint32_t)seq_scale8_base, (uint32_t)n_head);
#endif
                perf_exit(OP_QUANT8);
            } else
#endif
            ea8 = quantize_f_pow2(attn, q_dim, SP_X);
            model_matvec(eo, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            uint32_t dmax = 0;
            int kk = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
            perf_exit(OP_NORM_I32);
            int ed = ea8 - (int)eo->bias + (int)eo->rshift + kk;
            ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                    nrm_ffn16[l]);
        }

        // FFN (fully resident: no float round-trips).
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            int ex8 = E_NORM + op_rmsnorm_q8_sp(SP_XR, ex_state, ffn_norm[l],
                                                nrm_ffn16[l],
                                                n_embd, SP_H, SP_X);

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && !defined(BONSAI_HAS_DUALQ)
            if (seq_use && seq_ffn_use && seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
                // One list for gate->norm->up->norm->silu->quant8->down (see seq_build_ffn).
                // Whole-list time is charged to OP_SILU_MUL: per-op shares are documented as
                // unreliable under sequencing; totals are the measurement.
                if (!ffn_prepatched)
                    ffn_prep_layer(l);
                ffn_prepatched = 0;
                // silu A4 = eg + 6 = imm + kg (VSHLS convention); the descriptor carries
                // w0[11] so its kreg0 term ADDS. Requires a post-2026-08-29 bitstream -
                // earlier sequencers only subtract and the list computes garbage. The list
                // stays default-off pending the 1.7B divergence diagnosis either way.
                seq_patch((uint32_t)(seq_ffn_base + ffn_di_a4), 6,
                          (uint32_t)(ex8 - (int)eg_t->bias + (int)eg_t->rshift + 6));
                perf_enter(OP_SILU_MUL);
                if (fprof_use) {
                    perf_ovl_mark();
                    if (l == 0)
                        fprof_nts = seq_run_trace((uint32_t)seq_ffn_base,
                                                  (uint32_t)seq_ffn_len, fprof_ts, 16);
                    else
                        seq_run((uint32_t)seq_ffn_base, (uint32_t)seq_ffn_len);
                    perf_ovl_acc(fprof_acc_list); fprof_nlist++;
                } else {
#ifdef BONSAI_SP_SB_SLOT
                    seq_list_start((uint32_t)seq_ffn_base, (uint32_t)seq_ffn_len);
                    if (qkvl_use && l + 1 < n_layer)
                        qkv_prep_layer(l + 1); // Next layer's projections, under this stream.
                    seq_join();
#else
                    seq_run((uint32_t)seq_ffn_base, (uint32_t)seq_ffn_len);
#endif
                }
                perf_exit(OP_SILU_MUL);
                // Recover the folds the removed round-trips produced - identical values:
                // kg/ku from the spilled engine absmaxes, s from quant8's surviving res6.
                uint32_t absg = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP);
                uint32_t absu = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP + 32);
                int kg = (bitlen32(absg) > 14) ? bitlen32(absg) - 14 : 0; // == the >16383 scan.
                int ku = (bitlen32(absu) > 14) ? bitlen32(absu) - 14 : 0;
                int eg  = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;
                int eu  = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;
                int es8 = (eg + eu + 15)
                        + quant8_shift((uint32_t)vecunit_res6_read());
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;
                ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                        next_norm16(l));
                goto ffn_done;
            }
#endif
            if (fprof_use) {
                perf_ovl_mark();
                if (l == 0) { fprof_nsteps = 0; fprof_steps[fprof_nsteps++] = perf_now(); }
            }
            model_matvec(eg_t, SP_X, SP_Y);
            if (fprof_use && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int kg;
#if defined(CSR_ENGINE_CSR_ABSMAX_ADDR) && defined(CSR_VECUNIT_BASE)
            if (ovl_use && op_on_hw(OP_NORM_I32)) {
                // Overlap: the up stream runs UNDER the gate norm. Gate absmax is read
                // BEFORE the submit (the tracker resets on accept); regions are disjoint
                // (up: SP_X -> SP_YU, norm: SP_Y -> SP_G). Same kg, same program, same
                // args as the serial order - byte-identical by construction.
                uint32_t ag = engine_csr_absmax_read();
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
            if (fprof_use && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int eg = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;

            perf_enter(OP_NORM_I32);
            int ku = op_norm_i32_sp(SP_YU, SP_U, ffn_dim);
            perf_exit(OP_NORM_I32);
            if (fprof_use && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();
            int eu = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;

            int es = eg + eu + 15;
            int es8 = es + op_silu_q8_sp(SP_G, eg, SP_U, eu, ffn_dim, SP_X);
            if (fprof_use && l == 0 && fprof_nsteps < 8) fprof_steps[fprof_nsteps++] = perf_now();

            model_matvec(ed_t, SP_X, SP_Y);
            if (fprof_use) {
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
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && !defined(BONSAI_HAS_DUALQ)
            ffn_done: ;
#endif
        }
    }
    n_pos++;

    if (!want_logits)
        return -1;

    // Final norm + LM head (tied or separate), chunked over rows to fit the scratchpad.
    (void)op_rmsnorm_q8_sp(SP_XR, ex_state, out_norm,
                           nrm_out16, n_embd, SP_H, SP_X);
    perf_enter(OP_LMHEAD);
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head)
        head = model_entry(ROLE_TOK_EMB, 0);
    float sc  = 0; // Argmax runs on raw int32 (same scale across chunks).
    uint32_t rpc       = head->rows/n_ch;
    uint32_t chunk_rpc = 0x20000/(4*n_ch); // 128 KB of int32 results per chunk.
    if (chunk_rpc > rpc) chunk_rpc = rpc;

#ifdef BONSAI_HAS_TOPK
    if (op_on_hw(OP_LMHEAD)) {
        // One suppressed-write op over every row; the argmax tap returns value+index
        // (global row order, ties to smallest - same as the CPU scan). Padding rows
        // (vocab not divisible by n_ch) can win with garbage logits: fall through to
        // the chunked CPU scan in that rare case.
        uint32_t rb = bons_row_beats(head->cols, head->kind, head->group);
        struct engine_op op = {0};
        op.kind   = ENGINE_KIND_TERNARY;
        op.rows   = rpc;
        op.cols   = head->cols;
        op.wbase  = BONSAI_CHAN_WEIGHT_BASE + head->chan_base_beats*32;
        op.wlen   = rpc*rb;
        op.sbeats = bons_scale_beats(head->cols, head->group);
        op.cbeats = bons_code_beats(head->cols, head->kind);
        op.bias   = head->bias;
        op.rshift = head->rshift;
        op.xsrc   = SP_X;
        op.dst    = SP_Y;
        op.supw   = 1;
#ifdef CSR_SEQ_BASE
        if (lovl_use && seq_use) {
            // C2-3: the next forward's per-token list prep (score padding + the
            // rowsw/wlen/blocks patches) depends only on the position, never on the
            // token this argmax is about to pick - post it under the argmax stream.
            // supw=1 leaves the W port idle, so the posted writes drain freely.
            uint32_t dl = engine_submit(&op);
            attn_list_prep_token(n_pos);
            if (qkvl_use)
                qkv_prep_layer(0);   // Next forward's layer-0 projections ride here too.
            engine_join(dl);
        } else
#endif
        engine_run(&op);
        uint32_t idx = engine_csr_argmax_idx_read();
        if (idx < H->vocab) {
            perf_exit(OP_LMHEAD);
            if (tprof_active)
                tprof_end();
            return (int)idx;
        }
    }
#endif
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
        op.wbase  = BONSAI_CHAN_WEIGHT_BASE + (head->chan_base_beats + (uint64_t)r0*rb)*32;
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
    (void)sc;
    perf_exit(OP_LMHEAD);
    if (tprof_active)
        tprof_end();
    return best_tok;
}

// Public API ---------------------------------------------------------------------------------

int infer_init(void)
{
    for (int i = 0; i < KVP_SLOTS; i++)
        kvp_scale_e8[i] = -1000; // New image: restage every KV scale beat.
    H = model_header();
    if (H->magic != BONS_MAGIC)
        return -1;
#ifdef BONSAI_N_CHANNELS
    if (H->n_channels != BONSAI_N_CHANNELS) {
        printf("infer: image channels (%lu) != engine channels (%u)\n",
               (unsigned long)H->n_channels, BONSAI_N_CHANNELS);
        return -3;
    }
#endif
#if BONSAI_PROGS_VREGS >= 16
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
    n_ch     = H->n_channels;
    n_layer  = H->n_layer;
    n_embd   = H->n_embd;
    n_head   = H->n_head;
    n_kv_head= H->n_kv_head;
    head_dim = H->head_dim;
    ffn_dim  = H->ffn_dim;

    heap    = (uint8_t *)HEAP_BASE;
    {
        int nrep = n_head/n_kv_head, sh = 0;
        while ((1 << sh) < nrep) sh++;
        if ((1 << sh) != nrep) { printf("infer: n_head/n_kv_head must be a power of two\n"); return -5; }
        fold_nrep_sh = sh;
    }
    qkn_resident = 0; // New image / re-init: the resident qk-norm rows must be refetched.
    norm16_cache_build();
    tprof_setbuf(halloc(6000*16), 6000); // 96 KB trace buffer (main_ram, not sram .bss).
    {
        // kv_scores_br's R: depends only on head_dim - once, not 1792x/token.
        uint64_t bound = (uint64_t)(head_dim/32)*32u*127u*127u*1024u << 12;
        int r = 0;
        while ((bound >> r) >= (1ull << 31)) r++;
        kv_br_r = r;
        for (int i = 0; i < 64; i++) eps_memo[i] = 0; // New image: new rms_eps.
    }
    heap_hi = (MAIN_RAM_SIZE >= 0x0A000000) ? (uint8_t *)HEAP_HI_BASE : 0;
    int kv_dim = n_kv_head*head_dim;
    int q_dim  = n_head*head_dim;

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
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32,
                         e->chan_len_beats, SP_FET);
            volatile float *p = (volatile float *)(SCRATCHPAD_BASE + SP_FET);
            for (int j = 0; j < norms[i].n; j++) dst[j] = p[j];
            *norms[i].arr = dst;
        }
    }
    const struct bons_entry *e = model_entry(ROLE_OUT_NORM, 0);
    out_norm = halloc(n_embd*sizeof(float));
    engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32, e->chan_len_beats, SP_FET);
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
#ifdef BONSAI_HAS_KVSTORE
    cpu_kv_ctx = (max_ctx < CPU_KV_CTX) ? max_ctx : CPU_KV_CTX;
#else
    cpu_kv_ctx = max_ctx;
#endif
#ifdef BONSAI_HAS_KVSTORE
    // KV region: above the per-channel weight payload, 4 KB aligned.
    kv_base   = (uint32_t)((BONSAI_CHAN_WEIGHT_BASE + H->payload_beats*32 + 0xFFF) & ~0xFFFull);
    kv_rb     = 1 + head_dim/32; // One scale beat + code beats (cols = head_dim, Q8).
    kv_region = (uint32_t)((max_ctx + n_ch - 1)/n_ch)*kv_rb;
    kv_ekmax  = halloc(n_layer*n_kv_head*2);
    for (int i = 0; i < n_layer*n_kv_head*2; i++) kv_ekmax[i] = -128;
#ifdef CSR_SEQ_BASE
    seq_build_qkr();
    seq_build_attn();
    seq_build_ffn();
    seq_build_scale8();
    seq_build_qkv();
    seq_build_attn2();
#endif
#endif
    k_cache = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*kv_dim);
    v_cache = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*kv_dim);
    k_scale = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*n_kv_head*sizeof(float));
    v_scale = halloc_hi((uint64_t)n_layer*cpu_kv_ctx*n_kv_head*sizeof(float));
    scores  = halloc(max_ctx*sizeof(float));
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && defined(BONSAI_HAS_TOPK) && defined(BONSAI_HAS_DUALQ)
    spec_succ   = halloc((uint64_t)H->vocab*2);
    spec_succ_v = halloc(H->vocab);
#endif
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

    // The low heap now lives BELOW engine channel 3's weight window (see layout.py:
    // HBM_WEIGHT_BASE is port-matched, so channel 3 lands inside main_ram at +64 MB).
    // Overrunning it would scribble on that channel's weights and show up as wrong tokens
    // with everything else looking healthy - which is exactly the failure mode that cost a
    // day. Check it, loudly, once per init; the numbers are cheap to print and worth having
    // in every boot log.
    {
        uint32_t lo_used = (uint32_t)(heap - (uint8_t *)HEAP_BASE);
        uint32_t lo_room = (uint32_t)(BONSAI_CHAN_WEIGHT_BASE + 3*BONSAI_CHAN_STRIDE
                                      - (uint32_t)HEAP_BASE);
        printf("infer: low heap %lu KB", (unsigned long)(lo_used >> 10));
        if (heap_hi)
            printf(", high heap %lu KB @%08lx",
                   (unsigned long)((uint32_t)(heap_hi - (uint8_t *)HEAP_HI_BASE) >> 10),
                   (unsigned long)HEAP_HI_BASE);
        printf("\n");
        if ((uint32_t)MAIN_RAM_BASE + MAIN_RAM_SIZE > BONSAI_CHAN_WEIGHT_BASE + 3*BONSAI_CHAN_STRIDE
            && lo_used > lo_room) {
            printf("infer: LOW HEAP OVERRUNS engine channel 3's weights "
                   "(%lu KB used, %lu KB available) - tokens will be wrong\n",
                   (unsigned long)(lo_used >> 10), (unsigned long)(lo_room >> 10));
            return -1;
        }
    }
    return 0;
}

// Microbenchmark: time each stage of the stack in isolation, so "where do the cycles go"
// is measured on the card instead of argued about. Run after hdrload.
void infer_vbench(void)
{
    if (!n_layer) { puts("vbench: run hdrload first"); return; }
    uint32_t t0, dt;
    printf("VB clk=%d MHz  (cycles, avg over reps)\n",
           (int)(CONFIG_CLOCK_FREQUENCY/1000000));

#ifdef CSR_VECUNIT_BASE
    { // Vec dispatch floor: smallest real program run (resid_add, 1 beat).
        int32_t a[6] = {(int32_t)SP_H, (int32_t)SP_D, (int32_t)SP_H, 1, 0, -1};
        t0 = perf_now();
        for (int i = 0; i < 16; i++) vec_run(PROG_RESID_ADD, a, 6);
        dt = perf_now() - t0;
        printf("VB vec_1beat        %8lu\n", (unsigned long)(dt/16));
    }
    { // Uncached scratchpad word stores: the cost of every seq_patch / scale-prep write.
        volatile uint32_t *pw = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_FET);
        t0 = perf_now();
        for (int i = 0; i < 64; i++) pw[i] = (uint32_t)i;
        dt = perf_now() - t0;
        printf("VB sp_write_64w     %8lu  (per store %lu)\n", (unsigned long)dt,
               (unsigned long)(dt/64));
    }
#ifdef CSR_SEQ_BASE
    { // One-descriptor sequencer list (spill): the in-list dispatch floor.
        int32_t pa = (int32_t)SP_FET;
        seq_desc_vec(191, PROG_SPILL, &pa, 1, 0);
        t0 = perf_now();
        for (int i = 0; i < 16; i++) seq_run(191, 1);
        dt = perf_now() - t0;
        printf("VB seq_1desc        %8lu\n", (unsigned long)(dt/16));
    }
#endif
#if defined(BONSAI_HAS_KVSTORE)
    { // KV scale-beat prep with a forced exponent change every call.
        t0 = perf_now();
        for (int i = 0; i < 16; i++) kv_store_scale_prep(0, 0, (i & 1) ? -20 : -21);
        dt = perf_now() - t0;
        kvp_scale_e8[0] = -9999; // Leave the slot marked stale for the next real pack.
        printf("VB scale_prep       %8lu\n", (unsigned long)(dt/16));
    }
#endif
    { // Vec throughput: resid_add over n_embd (128 beats, ~1.3k instructions).
        int32_t a[6] = {(int32_t)SP_H, (int32_t)SP_D, (int32_t)SP_H, 128, 0, -1};
        t0 = perf_now();
        for (int i = 0; i < 16; i++) vec_run(PROG_RESID_ADD, a, 6);
        dt = perf_now() - t0;
        printf("VB vec_resid_128b   %8lu\n", (unsigned long)(dt/16));
    }
#endif
    { // Engine fetch: 128 beats from channel 0 (per-op latency + short stream).
        t0 = perf_now();
        for (int i = 0; i < 16; i++)
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE, 128, SP_FET);
        dt = perf_now() - t0;
        printf("VB fetch_128b       %8lu\n", (unsigned long)(dt/16));
    }
    { // Engine fetch: 1 beat (pure per-op overhead + HBM latency).
        t0 = perf_now();
        for (int i = 0; i < 16; i++)
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE, 1, SP_FET);
        dt = perf_now() - t0;
        printf("VB fetch_1b         %8lu\n", (unsigned long)(dt/16));
    }
    { // Long single-channel fetch: sustained per-beat read rate of ONE port. The 8-channel
      // matvec streams run at ~4.6 cycles/beat with zero AR stalls; if one port alone
      // sustains ~1 cycle/beat the loss is contention in the HBM AXI switch, if one port
      // alone is also slow it is a per-port burst-gap problem.
        t0 = perf_now();
        for (int i = 0; i < 4; i++)
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE, 512, SP_Y); // 16 KB into SP_Y.
        dt = perf_now() - t0;
        printf("VB fetch_512b       %8lu  (%lu.%02lu cyc/beat)\n", (unsigned long)(dt/4),
               (unsigned long)(dt/4/512), (unsigned long)((dt/4*100/512) % 100));
    }
    { // rmsnorm, staged: the op is fetch + eps glue + one vec program.
        const struct bons_entry *an = model_entry_kind(ROLE_ATTN_NORM, 0, KIND_NORM16);
        t0 = perf_now();
        for (int i = 0; i < 8; i++)
            op_rmsnorm_sp(SP_XR, -15, attn_norm[0], an, n_embd, SP_H);
        dt = perf_now() - t0;
        printf("VB rmsnorm_full     %8lu\n", (unsigned long)(dt/8));
#if defined(CSR_VECUNIT_BASE) && defined(BONSAI_HAS_VECUNIT2)
        if (an) {
            t0 = perf_now();
            for (int i = 0; i < 8; i++)
                engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + an->chan_base_beats*32,
                             an->chan_len_beats, SP_FET);
            dt = perf_now() - t0;
            printf("VB rmsnorm_fetch    %8lu  (%lu beats)\n",
                   (unsigned long)(dt/8), (unsigned long)an->chan_len_beats);
            int eps_int = (int)lroundf(H->rms_eps*ldexpf(1.0f, -2*(-15)));
            if (eps_int < 1) eps_int = 1;
            int ln = 0;
            while ((1 << ln) < n_embd) ln++;
            int32_t a[7] = {(int32_t)SP_XR, (int32_t)SP_FET, (int32_t)SP_H, n_embd/16,
                            -ln, eps_int, E_W16 - E_NORM - 15};
            t0 = perf_now();
            for (int i = 0; i < 8; i++) vec_run(PROG_RMSNORM, a, 7);
            dt = perf_now() - t0;
            printf("VB rmsnorm_vecrun   %8lu\n", (unsigned long)(dt/8));
        }
#endif
    }
    { // Big weight stream: ffn_gate matvec (the shape the token loop lives in), with the
      // per-channel stream counters around it: AR-stall cycles say whether the HBM side
      // limits, near-zero everywhere says the engine's own drain does.
        const struct bons_entry *eg = model_entry(ROLE_FFN_GATE, 0);
        uint32_t b0[32], a0[32];
        uint32_t r0[32];
        for (int c = 0; c < n_ch && c < 32; c++) {
            engine_csr_perf_chan_write(c);
            b0[c] = engine_csr_perf_beats_read();
            a0[c] = engine_csr_perf_ar_read();
            r0[c] = engine_csr_perf_r_read();
        }
        t0 = perf_now();
        for (int i = 0; i < 4; i++) model_matvec(eg, SP_X, SP_Y);
        dt = perf_now() - t0;
        printf("VB matvec_gate      %8lu  (%lu beats/chan)\n",
               (unsigned long)(dt/4), (unsigned long)eg->chan_len_beats);
        for (int c = 0; c < n_ch && c < 32; c += 4) {
            engine_csr_perf_chan_write(c);
            printf("VB   ch%d beats %lu arstall %lu rstall %lu\n", c,
                   (unsigned long)((engine_csr_perf_beats_read() - b0[c])/4),
                   (unsigned long)((engine_csr_perf_ar_read()   - a0[c])/4),
                   (unsigned long)((engine_csr_perf_r_read()    - r0[c])/4));
        }
    }
    { // Small weight stream: attn_q matvec.
        const struct bons_entry *eq = model_entry(ROLE_ATTN_Q, 0);
        t0 = perf_now();
        for (int i = 0; i < 8; i++) model_matvec(eq, SP_X, SP_Y);
        dt = perf_now() - t0;
        printf("VB matvec_q         %8lu  (%lu beats/chan)\n",
               (unsigned long)(dt/8), (unsigned long)eq->chan_len_beats);
    }
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
    { // KV store: the pair the token loop runs 448 times per forward (kvquant is 16% of it).
      // Split so the engine STORE and the vec quant8 that feeds it are separated - the whole
      // pair measures ~8.5k cycles in `stats` and only ~3.6k of that is accounted for by the
      // op floor plus the pack, so the remainder has to be named before it can be attacked.
        int32_t qa[4] = {(int32_t)SP_K, 0, (int32_t)(SP_KVP + 32), head_dim/16};
        t0 = perf_now();
        for (int i = 0; i < 16; i++) vec_run(PROG_QUANT8, qa, 4);
        dt = perf_now() - t0;
        printf("VB kv_pack_quant8   %8lu\n", (unsigned long)(dt/16));

        struct engine_op op = {0};
        op.kind  = ENGINE_KIND_STORE;
        op.chan  = 0;
        op.wbase = kv_region_addr(0, 0, 0);
        op.wlen  = kv_rb;
        op.xsrc  = kvp_slot(0, 0);
        t0 = perf_now();
        for (int i = 0; i < 16; i++) engine_run(&op);
        dt = perf_now() - t0;
        printf("VB kv_store_1region %8lu\n", (unsigned long)(dt/16));

        // Faithful to the token loop: 16 stores spread over the (kvh, kind) regions of a
        // layer, not the same address 16 times. If this is much dearer than the single-region
        // loop, the cost is in the scatter (row/bank behaviour), not the op.
        t0 = perf_now();
        for (int h = 0; h < n_kv_head; h++)
            for (int kind = 0; kind < 2; kind++) {
                op.wbase = kv_region_addr(0, h, kind);
                op.xsrc  = kvp_slot(h, kind);
                engine_run(&op);
            }
        dt = perf_now() - t0;
        printf("VB kv_store_layer   %8lu  (%d stores)\n",
               (unsigned long)dt, 2*n_kv_head);

        // The token loop ALTERNATES pack/store; the two loops above run them in phases. The
        // vector unit and the engine share the scratchpad port, so alternating may cost
        // arbitration that phase-separated batches do not. Same 16 pairs either way, so the
        // difference is the interleaving alone.
        t0 = perf_now();
        for (int h = 0; h < n_kv_head; h++)
            for (int kind = 0; kind < 2; kind++) {
                int32_t qq[4] = {(int32_t)((kind ? SP_V : SP_K) + (uint32_t)h*head_dim*2), 0,
                                 (int32_t)(kvp_slot(h, kind) + 32), head_dim/16};
                vec_run(PROG_QUANT8, qq, 4);
                op.wbase = kv_region_addr(0, h, kind);
                op.xsrc  = kvp_slot(h, kind);
                engine_run(&op);
            }
        dt = perf_now() - t0;
        printf("VB kv_pair_alt      %8lu  (%d pack+store pairs, interleaved)\n",
               (unsigned long)dt, 2*n_kv_head);
    }
#endif
    { // perf_now itself (measurement overhead per probe pair).
        t0 = perf_now();
        for (int i = 0; i < 64; i++) (void)perf_now();
        dt = perf_now() - t0;
        printf("VB perf_now         %8lu\n", (unsigned long)(dt/64));
    }
    { // CSR write cost by bank: engine op field vs timer vs a scratchpad posted word.
        t0 = perf_now();
        for (int i = 0; i < 64; i++) engine_csr_rows_write((uint32_t)(i & 1));
        dt = perf_now() - t0;
        printf("VB csr_wr_engine    %8lu\n", (unsigned long)(dt/64));
        t0 = perf_now();
        for (int i = 0; i < 64; i++) timer0_load_write(0xFFFFFFFFu - (uint32_t)(i & 1));
        dt = perf_now() - t0;
        printf("VB csr_wr_timer     %8lu\n", (unsigned long)(dt/64));
        t0 = perf_now();
        for (int i = 0; i < 64; i++) (void)engine_csr_status_read();
        dt = perf_now() - t0;
        printf("VB csr_rd_engine    %8lu\n", (unsigned long)(dt/64));
        t0 = perf_now();
        for (int i = 0; i < 64; i++) (void)seq_status_read();
        dt = perf_now() - t0;
        printf("VB csr_rd_seq       %8lu\n", (unsigned long)(dt/64));
    }
    puts("VB done");
}

void infer_spec(int on)
{
    spec_on  = on;
    spec_acc = spec_rej = 0;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && defined(BONSAI_HAS_TOPK) && defined(BONSAI_HAS_DUALQ)
    if (on && spec_succ_v)
        for (uint32_t i = 0; i < H->vocab; i++)
            spec_succ_v[i] = 0;
#endif
}

void infer_spec_stats(int *acc, int *rej)
{
    *acc = spec_acc;
    *rej = spec_rej;
}

int infer_generate(const int *prompt, int n_prompt, int max_new, void (*emit)(int tok))
{
    n_pos = 0; // Reset the KV cache (fresh generation).
#ifdef BONSAI_HAS_KVSTORE
    // ...and its exponent ratchet. kv_ekmax is the running max over every row ever
    // stored in a region and picks the KV scale bias (B = 22 - ekmax), so without this
    // a generation inherited the previous request's maxima: its arithmetic depended on
    // request history, and only the FIRST generation after a boot matched the reference
    // (2026-09-07, from the external review). Rows are overwritten from position 0 anyway.
    if (kv_ekmax)
        for (int i = 0; i < n_layer*n_kv_head*2; i++) kv_ekmax[i] = -128;
#endif
    int tok = -1;
    for (int i = 0; i < n_prompt; i++)
        tok = forward(prompt[i], i == n_prompt - 1);
    int count = 0;
    for (int i = 0; i < max_new; i++) {
        if (n_pos >= max_ctx)
            break;
        emit(tok);
        count++;
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && defined(BONSAI_HAS_TOPK) && defined(BONSAI_HAS_DUALQ)
        // Speculative decode: draft the successor seen last time this token appeared;
        // one dual-x pair-forward verifies the draft and computes the next token. On
        // accept, two tokens per weight-stream pass. Greedy -> lossless (token stream
        // identical to plain decode). Rejection restores the kv_ekmax entries the
        // speculated position touched; its KV rows are overwritten by the re-forward.
        if (spec_on && seq_use && attn_all_hw() && seq_at2_kvchunk > 0 && spec_succ != 0
            && n_pos + 2 <= max_ctx && i + 1 < max_new) {
            int draft = spec_succ_v[tok] ? (int)spec_succ[tok] : -1;
            if (draft >= 0) {
                int am0 = -1, am1 = -1;
                int save_pos = n_pos;
                if (forward_pair(tok, draft, &am0, &am1) == 0) {
                    if (am0 == draft) {
                        spec_acc++;
                        spec_succ[tok] = (uint16_t)draft; spec_succ_v[tok] = 1;
                        emit(draft);
                        count++; i++;
                        spec_succ[draft] = (uint16_t)am1; spec_succ_v[draft] = 1;
                        tok = am1;
                        continue;
                    }
                    // Reject: keep position n_pos-2 (t0 accepted), drop the draft's.
                    spec_rej++;
                    ek_restore();
                    n_pos = save_pos + 1;
                    spec_succ[tok] = (uint16_t)am0; spec_succ_v[tok] = 1;
                    tok = am0;
                    continue;
                }
                n_pos = save_pos; // Padding-row bail: redo plainly.
                ek_restore();
            }
        }
        if (spec_on && spec_succ != 0) {
            // Track successors even on the plain path so drafts warm up.
            int nt = forward(tok, 1);
            spec_succ[tok] = (uint16_t)nt; spec_succ_v[tok] = 1;
            tok = nt;
            continue;
        }
#endif
        // The forward after the LAST emitted token computes a successor nobody emits and
        // stores a KV row nobody reads (every caller restarts at n_pos = 0), so skip it:
        // one full forward per request, ~1.5% of a 64-token reply. bench 64 therefore runs
        // 66 forwards where it ran 67 - compare cycles/FORWARD across that boundary, not
        // cycles/token (2026-09-07, from the external review).
        if (i + 1 < max_new)
            tok = forward(tok, 1);
    }
    return count;
}

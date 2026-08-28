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
static int axpy_rpre_c(uint64_t p8sum)
{
    int bl = 0;
    while ((p8sum >> bl) != 0) bl++;
    return bl + 5;
}

// B/R for a scores op over a region: pow-2 scales make e16 + B == 12 exactly
// (fp16_decompose: 2^ek = 1024 * 2^(ek-10); B = 12 - (ekmax - 10) = 22 - ekmax), so the
// worst-row acc bound is closed-form: groups * (32*127*127) * 1024 << 12.
static void kv_scores_br(int l, int kvh, int kind, int *B, int *R)
{
    int ekmax = kv_ekmax[kv_ek_idx(l, kvh, kind)];
    *B = 22 - ekmax;
    uint64_t bound = (uint64_t)(head_dim/32)*32u*127u*127u*1024u << 12;
    int r = 0;
    while ((bound >> r) >= (1ull << 31)) r++;
    *R = r;
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
    if (kvp_scale_e8[slot] == e8)
        return;
    volatile uint16_t *ps = (volatile uint16_t *)(SCRATCHPAD_BASE + kvp_slot(kvh, kind));
    int groups = head_dim/32;
    for (int g = 0; g < 16; g++)
        ps[g] = (g < groups) ? (uint16_t)((e8 + 15) << 10) : 0;
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
    int32_t amax = 0;
    for (int g = 0; g < G; g++) {
        int sh = e16[g] - e16min;
        if (sh > 20) sh = 20;
        int32_t gm = ((m11[g] < 0) ? -m11[g] : m11[g]) << sh;
        if (gm <= amax) continue;
        for (int j = 0; j < 32; j++) { // Any nonzero 2-bit code pair in the group?
            uint8_t b3 = codes[g*32 + j];
            if (b3 != 0x55) { amax = gm; break; } // 0x55 = four "code 1" (zero weight).
        }
    }
    int bl = 0;
    while ((amax >> bl) != 0) bl++;      // Bit length.
    int k2 = bl - 15;                    // Mantissas into [2^14, 2^15); may be negative.
    volatile int16_t *pd = sp16(sp_dst);
    for (int g = 0; g < G; g++) {
        int sh = e16[g] - e16min;
        if (sh > 20) sh = 20;
        int net = sh - k2;               // Combined shift, sign either way.
        for (int j = 0; j < 128; j++) {
            int q2 = (codes[g*32 + j/4] >> ((j & 3)*2)) & 3;
            int32_t v2 = (int32_t)(q2 - 1)*m11[g];
            pd[g*128 + j] = (int16_t)((net >= 0) ? (v2 << net) : (v2 >> -net));
        }
    }
    return e16min + k2;
}

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE)
// Sequencer descriptor list for the per-head qk-norm + rope loops. Addresses are
// head-relative inside SP_Q/SP_K, so the list is layer-independent; per layer only the
// rmsnorm eps_int args (A5 = descriptor word 7) change. Built once at init.
static int seq_qkr_count;
static void seq_build_qkr(void)
{
    int ln = 0;
    while ((1 << ln) < head_dim) ln++;
    uint32_t di = 0;
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
    seq_desc_stop(di++);
    seq_qkr_count = di;
}

static int seq_eps_int(int es)
{
    int e = (int)lroundf(H->rms_eps*ldexpf(1.0f, -2*es));
    return (e < 1) ? 1 : e;
}

// Sequencer descriptor list for the fused per-head attention chain:
//   quant8 -> scores (engine) -> softmax8 -> spill(p8 sum) -> AXPY (rshift from rpre of
//   the chained res6 = p8 sum) -> reduce x(n_ch-1) -> spill(absmax) -> norm_i32 (k via
//   res6). One block per head slot, reused across chunks; head/token/layer-dependent
//   words are patched before each run. Spills feed the one CPU float fold per head
//   (c2/hmax) after the list - identical numerics to the CPU-dispatched fused path.
static int seq_attn_base, seq_attn_dph, seq_attn_heads;
static void seq_build_attn(void)
{
    seq_attn_heads = 0;
    if (n_ch < 2)
        return; // n1's absmax lives in the engine CSR; keep CPU-dispatched ops there.
    // 7 descriptors per head: quant8, scores, softmax8, spill(p8), axpy(xred),
    // spill(absmax from the engine tracker), norm_i32. It was 6 + n_ch = 14 before the
    // engine learned to reduce across channels; at N=8 that halving is what lets all 16
    // heads fit one list (16*7 + 49 = 161 of 192) so the chunking below stops splitting.
    int dph = 7;
    int cap = (192 - seq_qkr_count)/dph;
    if (cap <= 0)
        return;
    int chunk = (cap < n_head) ? cap : n_head;
    int lg = 0;
    while ((1 << lg) < head_dim) lg++;
    int Ceff = (lg & 1) ? 523 : 739;
    seq_attn_base  = seq_qkr_count;
    seq_attn_dph   = dph;
    seq_attn_heads = chunk;
    for (int j = 0; j < chunk; j++) {
        uint32_t di = (uint32_t)seq_attn_base + (uint32_t)j*dph;
        int32_t qargs[4] = {(int32_t)(SP_Q + (uint32_t)j*head_dim*2), 0,
                            (int32_t)SP_X, head_dim/16};
        seq_desc_vec(di + 0, PROG_QUANT8, qargs, 4, 0);
        struct engine_op sc = {0};
        sc.kind   = ENGINE_KIND_Q8;
        sc.ilv    = 1;
        sc.cols   = head_dim;
        sc.sbeats = 1;
        sc.cbeats = head_dim/32;
        sc.xsrc   = SP_X;
        sc.dst    = SP_SC;
        seq_desc_engine(di + 1, &sc);
        int32_t sargs[6] = {(int32_t)SP_SC, Ceff, (int32_t)SP_X, 0, -11, 0};
        seq_desc_vec(di + 2, PROG_SOFTMAX8, sargs, 6, 0);
        int32_t p1 = (int32_t)(SP_AWP8 + (uint32_t)j*32);
        seq_desc_vec(di + 3, PROG_SPILL, &p1, 1, 0);
        struct engine_op wv = {0};
        wv.kind        = ENGINE_KIND_AXPY;
        wv.cols        = head_dim;
        wv.sbeats      = 1;
        wv.cbeats      = head_dim/32;
        wv.xsrc        = SP_X;
        wv.dst         = SP_AW;
        wv.rshift_rpre = 1;
        wv.xred        = 1;
        seq_desc_engine(di + 4, &wv);
        // The engine summed the channels itself, so the n_ch-1 reduce_acc calls are gone and
        // the absmax the CPU fold needs comes from the engine tracker instead of the last
        // reduce's res6. seq_desc_vec_seed puts it in S7 for the spill, and spill's trailing
        // SMOV s6, s7 leaves it in res6 - so the norm below keeps taking k from KRES6 and
        // sees the identical value. Numerics are unchanged by construction, not by luck.
        int32_t p2 = (int32_t)(SP_AWAM + (uint32_t)j*32);
        seq_desc_vec_seed(di + 5, PROG_SPILL, &p2, 1, 0, 1);
        int32_t nargs[5] = {(int32_t)SP_AW, 0, (int32_t)(SP_H + (uint32_t)j*head_dim*2),
                            head_dim/16, 0};
        seq_desc_vec(di + 6, PROG_NORM_I32, nargs, 5,
                     (uint16_t)(SEQ_SRC_KRES6 << 8)); // A4 = k(res6).
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
static int seq_ffn_base = -1;
static void seq_build_ffn(void)
{
#ifdef BONSAI_HAS_DUALQ
    return; // Dual builds use attn2 lists; FFN list only wired on the shipping non-dual path.
#else
    int di = seq_qkr_count + seq_attn_heads*seq_attn_dph;
    if (di + 10 > 192 || seq_attn_heads == 0)
        return;
    seq_ffn_base = di;
    // Engine descriptors 0/3/8 are fully rebuilt per layer (entries differ); placeholders.
    struct engine_op z = {0};
    seq_desc_engine(di + 0, &z);
    int32_t pa = (int32_t)SP_FFSP;
    seq_desc_vec_seed(di + 1, PROG_SPILL, &pa, 1, 0, 1);
    int32_t na[5] = {(int32_t)SP_Y, 0, (int32_t)SP_G, ffn_dim/16, 0};
    seq_desc_vec(di + 2, PROG_NORM_I32, na, 5, (uint16_t)(SEQ_SRC_KABS << 8));
    seq_desc_engine(di + 3, &z);
    int32_t pb = (int32_t)(SP_FFSP + 32);
    seq_desc_vec_seed(di + 4, PROG_SPILL, &pb, 1, 0, 1);
    int32_t nb[5] = {(int32_t)SP_YU, 0, (int32_t)SP_U, ffn_dim/16, 0};
    seq_desc_vec(di + 5, PROG_NORM_I32, nb, 5, (uint16_t)(SEQ_SRC_KABS << 8));
    int32_t sa[7] = {(int32_t)SP_G, (int32_t)SP_U, (int32_t)SP_G, ffn_dim/16, 0, 0, -15};
    seq_desc_vec(di + 6, PROG_SILU_MUL, sa, 7, 0);
    seq_patch(di + 6, 1, (uint32_t)PROG_SILU_MUL | (1u << (16 + 2*4))); // A4 -= kreg0.
    int32_t qa[4] = {(int32_t)SP_G, 1, (int32_t)SP_X, ffn_dim/16};
    seq_desc_vec(di + 7, PROG_QUANT8, qa, 4, 0);
    seq_desc_engine(di + 8, &z);
    seq_desc_stop(di + 9);
#endif
}

int seq_use = 1;
// FFN descriptor list: DEFAULT OFF. Sim tokens (n8 tiny, all legs) are byte-identical and
// the tiny-sim window improves -3.3% - but the 1.7B card A/B fails TOKENS. That is exactly
// the exponent-path blind spot the BFP-invariants note warns the tiny gate cannot see
// (residual ratchet / shift magnitudes the tiny shapes never reach). Off until the 1.7B
// divergence is diagnosed; enable with `ffnseq on` to reproduce.
int seq_ffn_use = 0;
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
        int32_t sa[6] = {(int32_t)SP_Y, Ceff, (int32_t)SP_X, 0, -11, 0};
        seq_desc_vec(d1 + 3, PROG_SOFTMAX8, sa, 6, 0);
        int32_t pa = (int32_t)(SP_A2P8 + (uint32_t)hA*32);
        seq_desc_vec(d1 + 4, PROG_SPILL, &pa, 1, 0);
        int32_t sb[6] = {(int32_t)SP_SCB, Ceff, 0, 0, -11, 0}; // A2 (p8B dst) per token.
        seq_desc_vec(d1 + 5, PROG_SOFTMAX8, sb, 6, 0);
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
    int padn = (16 - (cnt & 15)) & 15;
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
        while ((am >> k2) > 16383u) k2++;
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
            for (int hh = 0; hh < n_head + n_kv_head; hh++)
                seq_patch(2*hh, 7, (hh < n_head) ? epq : epk);
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

static int forward(int token, int want_logits)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
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
    }
#endif

    for (int l = 0; l < n_layer; l++) {
        // Attention block.
        perf_enter(OP_RMSNORM);
        op_rmsnorm_sp(SP_XR, ex_state, attn_norm[l],
                      model_entry_kind(ROLE_ATTN_NORM, l, KIND_NORM16), n_embd, SP_H);
        perf_exit(OP_RMSNORM);
        perf_enter(OP_QUANT8);
        int sx8 = op_quant8_sp(SP_H, SP_X, n_embd);
        perf_exit(OP_QUANT8);
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
            const struct bons_entry *qn16 = model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_NORM16);
            const struct bons_entry *kn16 = model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_NORM16);
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + qn16->chan_base_beats*32,
                         qn16->chan_len_beats, SP_FET);
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + kn16->chan_base_beats*32,
                         kn16->chan_len_beats, SP_FET2);
#ifdef CSR_SEQ_BASE
            {
                // One sequencer run replaces the 2*(n_head+n_kv) per-head dispatches;
                // only the eps args change per layer (descriptor word 7 = A5).
                int epq = seq_eps_int(eq_r), epk = seq_eps_int(ek_r);
                for (int hh = 0; hh < n_head + n_kv_head; hh++)
                    seq_patch(2*hh, 7, (hh < n_head) ? epq : epk);
                seq_run(0, seq_qkr_count);
            }
            perf_exit(OP_QKNORM);
#else
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
            for (int hh = 0; hh < n_kv_head; hh++) {   // Pack every head first...
                kv_pack_row_res(l, hh, 0, SP_K + hh*head_dim*2, E_NORM);
                kv_pack_row_res(l, hh, 1, SP_V + hh*head_dim*2, ev_res);
            }
            for (int hh = 0; hh < n_kv_head; hh++) {   // ...then hand the port to the engine.
                kv_store_row_res(l, hh, 0, pos);
                kv_store_row_res(l, hh, 1, pos);
            }
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
            int cnt  = pos + 1;
            int padn = (16 - (cnt & 15)) & 15;
            volatile int32_t *sc32 = (volatile int32_t *)(SCRATCHPAD_BASE + SP_SC);
            for (int i2 = 0; i2 < padn; i2++)
                sc32[cnt + i2] = (int32_t)0x80000000;
            uint32_t rows  = (uint32_t)cnt/n_ch, rem = (uint32_t)cnt % n_ch;
            uint32_t wlen  = rows*kv_rb;
            uint32_t rowsw = rows | (rem << 16);
            int lg = 0;
            while ((1 << lg) < head_dim) lg++;
            uint32_t blocks = (uint32_t)(cnt + padn)/16;
            for (int h0 = 0; h0 < n_head; h0 += seq_attn_heads) {
                int nh = n_head - h0;
                if (nh > seq_attn_heads) nh = seq_attn_heads;
                for (int j = 0; j < nh; j++) {
                    int hh = h0 + j, kvh = hh/n_rep;
                    uint32_t di = (uint32_t)seq_attn_base + (uint32_t)j*seq_attn_dph;
                    int Bk, Rk, Bv, Rd;
                    kv_scores_br(l, kvh, 0, &Bk, &Rk);
                    kv_scores_br(l, kvh, 1, &Bv, &Rd);
                    int sigma = (E_NORM + 8) - Bk + Rk - (lg >> 1);
                    seq_patch(di + 0, 2, SP_Q + (uint32_t)hh*head_dim*2);
                    seq_patch(di + 1, 2, rowsw);
                    seq_patch(di + 1, 4, kv_region_addr(l, kvh, 0));
                    seq_patch(di + 1, 5, wlen);
                    seq_patch(di + 1, 7, (uint32_t)Bk | ((uint32_t)Rk << 16));
                    seq_patch(di + 2, 5, blocks);          // softmax8 A3 (blocks).
                    seq_patch(di + 2, 7, (uint32_t)(sigma + 6 + 11)); // A5 (s2 shift).
                    seq_patch(di + 3, 2, SP_AWP8 + (uint32_t)hh*32);
                    seq_patch(di + 4, 2, rowsw);
                    seq_patch(di + 4, 4, kv_region_addr(l, kvh, 1));
                    seq_patch(di + 4, 5, wlen);
                    seq_patch(di + 4, 7, (uint32_t)Bv);
                    seq_patch(di + 5, 2, SP_AWAM + (uint32_t)hh*32);
                    seq_patch(di + 6, 4, SP_H + (uint32_t)hh*head_dim*2);
                    (void)Rd;
                }
                seq_run((uint32_t)seq_attn_base, (uint32_t)nh*seq_attn_dph);
            }
            perf_exit(OP_SCORES);
            // One float fold per head from the spilled p8 sums / absmaxes.
            perf_enter(OP_WV);
            for (int hh = 0; hh < n_head; hh++) {
                int kvh = hh/n_rep;
                uint32_t P8 = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWP8
                                                     + (uint32_t)hh*32);
                uint32_t am = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_AWAM
                                                     + (uint32_t)hh*32);
                int Bv, Rd;
                kv_scores_br(l, kvh, 1, &Bv, &Rd);
                int Rpre = axpy_rpre_c(P8 ? P8 : 1);
                int k2 = 0;
                while ((am >> k2) > 16383u) k2++;
                float c2 = ldexpf(1.0f/(float)(P8 ? P8 : 1), Rpre - Bv + k2);
                aw_c[hh] = c2;
                float hm2 = (float)(am >> k2)*c2;
                if (hm2 > aw_hmax) aw_hmax = hm2;
                (void)Rd;
            }
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
                int padn = (16 - (cnt & 15)) & 15;
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
                int32_t sargs[6] = {(int32_t)SP_SC, Ceff, (int32_t)SP_X, (cnt + padn)/16,
                                    -s1v, sigma + 6 + s1v};
                uint64_t P8 = vec_run(PROG_SOFTMAX8, sargs, 6);
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
                while ((am >> k2) > 16383u) k2++;
                int32_t nargs[5] = {(int32_t)SP_SC, 0,
                                    (int32_t)(SP_H + hh*head_dim*2), head_dim/16, k2};
                vec_run(PROG_NORM_I32, nargs, 5);
                float c2 = ldexpf(1.0f/(float)(uint32_t)P8, Rpre - Bv + k2);
                aw_c[hh] = c2;
                float hm2 = (float)(uint32_t)(am >> k2)*c2;
                if (hm2 > aw_hmax) aw_hmax = hm2;
                (void)ah;
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
                if (aw_hmax > 0) { frexpf(aw_hmax, &ea8); ea8 -= 7; }
                for (int hh = 0; hh < n_head; hh++) {
                    int32_t Mh = (int32_t)lroundf(ldexpf(aw_c[hh], 15 - ea8));
                    if (Mh > 32767) Mh = 32767;
                    if (Mh < 0)     Mh = 0;
                    int32_t sa2[5] = {(int32_t)(SP_H + hh*head_dim*2), 0,
                                      (int32_t)(SP_X + hh*head_dim), head_dim/16, Mh};
                    vec_run(PROG_SCALE8, sa2, 5);
                }
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
            perf_enter(OP_RESID_ADD);
            ex_state = op_resid_add_sp(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax);
            perf_exit(OP_RESID_ADD);
        }

        // FFN (fully resident: no float round-trips).
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            perf_enter(OP_RMSNORM);
            op_rmsnorm_sp(SP_XR, ex_state, ffn_norm[l],
                          model_entry_kind(ROLE_FFN_NORM, l, KIND_NORM16), n_embd, SP_H);
            perf_exit(OP_RMSNORM);
            perf_enter(OP_QUANT8);
            int ex8 = E_NORM + op_quant8_sp(SP_H, SP_X, n_embd);
            perf_exit(OP_QUANT8);

#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2) && defined(CSR_SEQ_BASE) \
    && !defined(BONSAI_HAS_DUALQ)
            if (seq_use && seq_ffn_use && seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
                // One list for gate->norm->up->norm->silu->quant8->down (see seq_build_ffn).
                // Whole-list time is charged to OP_SILU_MUL: per-op shares are documented as
                // unreliable under sequencing; totals are the measurement.
                struct engine_op mo = {0};
                model_matvec_op(eg_t, SP_X, SP_Y, &mo);
                seq_desc_engine(seq_ffn_base + 0, &mo);
                seq_patch(seq_ffn_base + 0, 0, 1u | (1u << 5)); // capture k -> kreg0.
                model_matvec_op(eu_t, SP_X, SP_YU, &mo);
                seq_desc_engine(seq_ffn_base + 3, &mo);
                model_matvec_op(ed_t, SP_X, SP_Y, &mo);
                seq_desc_engine(seq_ffn_base + 8, &mo);
                // silu A4 = eg + 6 = imm + kg under the 2026-08-28 VSHLS convention - but the
                // kreg arg path SUBTRACTS (chosen for the old -(eg+6) shape). The list is
                // default-off with an undiagnosed 1.7B divergence anyway; when revived, the
                // kadd polarity needs a descriptor bit (or flip to add) alongside that fix.
                seq_patch(seq_ffn_base + 6, 6,
                          (uint32_t)(ex8 - (int)eg_t->bias + (int)eg_t->rshift + 6));
                perf_enter(OP_SILU_MUL);
                seq_run((uint32_t)seq_ffn_base, 10);
                perf_exit(OP_SILU_MUL);
                // Recover the folds the removed round-trips produced - identical values:
                // kg/ku from the spilled engine absmaxes, s from quant8's surviving res6.
                uint32_t absg = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP);
                uint32_t absu = *(volatile uint32_t *)(SCRATCHPAD_BASE + SP_FFSP + 32);
                int kg = 0, ku = 0;
                while ((absg >> kg) > 16383u) kg++;
                while ((absu >> ku) > 16383u) ku++;
                int eg  = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;
                int eu  = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;
                int es8 = (eg + eu + 15)
                        + quant8_shift((uint32_t)vecunit_res6_read());
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;
                perf_enter(OP_RESID_ADD);
                ex_state = op_resid_add_sp(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax);
                perf_exit(OP_RESID_ADD);
                goto ffn_done;
            }
#endif
            model_matvec(eg_t, SP_X, SP_Y);
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
            int eg = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;

            perf_enter(OP_NORM_I32);
            int ku = op_norm_i32_sp(SP_YU, SP_U, ffn_dim);
            perf_exit(OP_NORM_I32);
            int eu = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;

            perf_enter(OP_SILU_MUL);
            op_silu_mul_sp(SP_G, eg, SP_U, eu, ffn_dim);
            perf_exit(OP_SILU_MUL);
            int es = eg + eu + 15;

            perf_enter(OP_QUANT8);
            int es8 = es + op_quant8_sp(SP_G, SP_X, ffn_dim);
            perf_exit(OP_QUANT8);

            model_matvec(ed_t, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            uint32_t dmax = 0;
            int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
            perf_exit(OP_NORM_I32);
            int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;

            perf_enter(OP_RESID_ADD);
            ex_state = op_resid_add_sp(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax);
            perf_exit(OP_RESID_ADD);
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
    perf_enter(OP_RMSNORM);
    op_rmsnorm_sp(SP_XR, ex_state, out_norm,
                  model_entry_kind(ROLE_OUT_NORM, 0, KIND_NORM16), n_embd, SP_H);
    perf_exit(OP_RMSNORM);
    perf_enter(OP_QUANT8);
    op_quant8_sp(SP_H, SP_X, n_embd);
    perf_exit(OP_QUANT8);
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
        engine_run(&op);
        uint32_t idx = engine_csr_argmax_idx_read();
        if (idx < H->vocab) {
            perf_exit(OP_LMHEAD);
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
    n_ch     = H->n_channels;
    n_layer  = H->n_layer;
    n_embd   = H->n_embd;
    n_head   = H->n_head;
    n_kv_head= H->n_kv_head;
    head_dim = H->head_dim;
    ffn_dim  = H->ffn_dim;

    heap    = (uint8_t *)HEAP_BASE;
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
        uint32_t b0[8], a0[8];
        for (int c = 0; c < n_ch && c < 8; c++) {
            engine_csr_perf_chan_write(c);
            b0[c] = engine_csr_perf_beats_read();
            a0[c] = engine_csr_perf_ar_read();
        }
        t0 = perf_now();
        for (int i = 0; i < 4; i++) model_matvec(eg, SP_X, SP_Y);
        dt = perf_now() - t0;
        printf("VB matvec_gate      %8lu  (%lu beats/chan)\n",
               (unsigned long)(dt/4), (unsigned long)eg->chan_len_beats);
        for (int c = 0; c < n_ch && c < 8; c += 4) {
            engine_csr_perf_chan_write(c);
            printf("VB   ch%d beats %lu arstall %lu\n", c,
                   (unsigned long)((engine_csr_perf_beats_read() - b0[c])/4),
                   (unsigned long)((engine_csr_perf_ar_read()   - a0[c])/4));
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
        tok = forward(tok, 1);
    }
    return count;
}

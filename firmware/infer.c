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

#include "bons.h"
#include "engine.h"
#include "model.h"
#include "infer.h"
#include "ops.h"
#include "perf.h"
#include "vec.h"

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
#define SP_FET  0x30000 // Fetch staging (embedding rows).

// Fixed exponent for CPU-produced normed vectors (values bounded ~16 by construction).
#define E_NORM  (-11)

// Heap: after the BONS header/table area.
#define HEAP_BASE (BONSAI_HDR_ADDR + 0x100000)

static uint8_t *heap;
static void *halloc(uint32_t n)
{
    void *p = heap;
    heap = heap + ((n + 63) & ~63u);
    return p;
}

// Model state --------------------------------------------------------------------------------

static const struct bons_header *H;
static int n_ch, head_dim, n_head, n_kv_head, n_embd, ffn_dim, n_layer;
static int max_ctx;

static float **attn_norm, **ffn_norm, **q_norm, **k_norm; // Per layer.
static float *out_norm;

static int8_t *k_cache, *v_cache;    // [layer][pos][kv_dim] int8.
static float  *k_scale, *v_scale;    // [layer][pos].
static int    n_pos;

static float *x, *h, *q, *k, *v, *attn, *gate, *up, *scores;
static int ex_state; // Exponent of the resident x (SP_XR).

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

// rmsnorm: SP int16(src,esrc) -> SP int16(dst) at fixed E_NORM (CPU float math; HW in M4).
static void op_rmsnorm_sp(uint32_t src, int esrc, const float *w, int n, uint32_t dst)
{
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

// quant8: SP int16 -> SP int8 (RNE >> 8); returns nothing (out exponent = e + 8).
static void op_quant8_sp(uint32_t src, uint32_t dst, int n)
{
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_QUANT8)) {
        int32_t args[4] = {(int32_t)src, 0, (int32_t)dst, n/16};
        vec_run(PROG_QUANT8, args, 4);
        return;
    }
#endif
    volatile int16_t *ps = sp16(src);
    volatile int8_t  *pd = sp8(dst);
    for (int i = 0; i < n; i++) {
        int32_t v = ps[i];
        int32_t q = (v >> 8) + (((v >> 7) & 1) & (((v & 0x7F) != 0) | ((v >> 8) & 1)));
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        pd[i] = (int8_t)q;
    }
}

// norm_i32: SP int32 -> SP int16 (floor >> k chosen from the engine absmax CSR).
// Returns k (caller: out exponent = in exponent + k).
static int op_norm_i32_sp(uint32_t src, uint32_t dst, int n)
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
    int k = 0;
    while ((amax >> k) > 16383u) k++;
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

// silu_mul: SP int16(gate,eg) x SP int16(up,eu) -> SP int16(gate) at eg+eu+15.
static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n)
{
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_SILU_MUL)) {
        int32_t args[7] = {(int32_t)gsrc, (int32_t)usrc, (int32_t)gsrc, n/16,
                           -(eg + 6), 0, -15};
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

// resid_add: SP int16(a,ea) + SP int16(b,eb) -> SP int16(a) at eo = max(ea,eb)+1.
// Returns eo.
static int op_resid_add_sp(uint32_t a, int ea, uint32_t b, int eb, int n)
{
    int eo = ((ea > eb) ? ea : eb) + 1;
#ifdef CSR_VECUNIT_BASE
    if (op_on_hw(OP_RESID_ADD)) {
        int32_t args[6] = {(int32_t)a, (int32_t)b, (int32_t)a, n/16, ea - eo, eb - eo};
        vec_run(PROG_RESID_ADD, args, 6);
        return eo;
    }
#endif
    volatile int16_t *pa = sp16(a);
    volatile int16_t *pb = sp16(b);
    for (int i = 0; i < n; i++) {
        int32_t v = (pa[i] >> (eo - ea)) + (pb[i] >> (eo - eb));
        if (v >  32767) v =  32767;
        if (v < -32767) v = -32767;
        pa[i] = (int16_t)v;
    }
    return eo;
}

// Forward one token; returns argmax token id (only meaningful when wanted). ------------------

static int forward(int token, int want_logits)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    perf_enter(OP_EMBED);
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
    }
    perf_exit(OP_EMBED);

    int   kv_dim = n_kv_head*head_dim;
    int   q_dim  = n_head*head_dim;
    float scale  = 1.0f/sqrtf((float)head_dim);
    int   pos    = n_pos;

    for (int l = 0; l < n_layer; l++) {
        // Attention block.
        perf_enter(OP_RMSNORM);
        op_rmsnorm_sp(SP_XR, ex_state, attn_norm[l], n_embd, SP_H);
        perf_exit(OP_RMSNORM);
        perf_enter(OP_QUANT8);
        op_quant8_sp(SP_H, SP_X, n_embd);
        perf_exit(OP_QUANT8);
        float sx = ldexpf(1.0f, E_NORM + 8);
        matvec_rows_f(model_entry(ROLE_ATTN_Q, l), 0, q_dim/n_ch,  sx, q);
        matvec_rows_f(model_entry(ROLE_ATTN_K, l), 0, kv_dim/n_ch, sx, k);
        matvec_rows_f(model_entry(ROLE_ATTN_V, l), 0, kv_dim/n_ch, sx, v);

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
            int8_t *kd = k_cache + ((uint64_t)l*max_ctx + pos)*kv_dim;
            int8_t *vd = v_cache + ((uint64_t)l*max_ctx + pos)*kv_dim;
            float ka = 0, va = 0;
            for (int i = 0; i < kv_dim; i++) {
                float a = fabsf(k[i]); if (a > ka) ka = a;
                a = fabsf(v[i]);       if (a > va) va = a;
            }
            float ks = (ka > 0) ? ka/127.0f : 1.0f, is = 1.0f/ks;
            for (int i = 0; i < kv_dim; i++) kd[i] = (int8_t)lroundf(k[i]*is);
            k_scale[l*max_ctx + pos] = ks;
            float vs = (va > 0) ? va/127.0f : 1.0f; is = 1.0f/vs;
            for (int i = 0; i < kv_dim; i++) vd[i] = (int8_t)lroundf(v[i]*is);
            v_scale[l*max_ctx + pos] = vs;
        }
        perf_exit(OP_KVQUANT);

        // Attention (CPU, GQA).
        int n_rep = n_head/n_kv_head;
        for (int hh = 0; hh < n_head; hh++) {
            int kvh = hh/n_rep;
            const float *qh = q + hh*head_dim;
            float smax = -1e30f;
            perf_enter(OP_SCORES);
            for (int t = 0; t <= pos; t++) {
                const int8_t *kt = k_cache + ((uint64_t)l*max_ctx + t)*kv_dim + kvh*head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qh[d]*kt[d];
                float sc = dot*k_scale[l*max_ctx + t]*scale;
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
            for (int d = 0; d < head_dim; d++) ah[d] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *vt = v_cache + ((uint64_t)l*max_ctx + t)*kv_dim + kvh*head_dim;
                float p = scores[t]*inv*v_scale[l*max_ctx + t];
                for (int d = 0; d < head_dim; d++) ah[d] += p*vt[d];
            }
            perf_exit(OP_WV);
        }

        // Output projection + residual (resident).
        {
            const struct bons_entry *eo = model_entry(ROLE_ATTN_OUT, l);
            int ea8 = quantize_f_pow2(attn, q_dim, SP_X);
            model_matvec(eo, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            int kk = op_norm_i32_sp(SP_Y, SP_D, n_embd);
            perf_exit(OP_NORM_I32);
            int ed = ea8 - (int)eo->bias + (int)eo->rshift + kk;
            perf_enter(OP_RESID_ADD);
            ex_state = op_resid_add_sp(SP_XR, ex_state, SP_D, ed, n_embd);
            perf_exit(OP_RESID_ADD);
        }

        // FFN (fully resident: no float round-trips).
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            perf_enter(OP_RMSNORM);
            op_rmsnorm_sp(SP_XR, ex_state, ffn_norm[l], n_embd, SP_H);
            perf_exit(OP_RMSNORM);
            perf_enter(OP_QUANT8);
            op_quant8_sp(SP_H, SP_X, n_embd);
            perf_exit(OP_QUANT8);
            int ex8 = E_NORM + 8;

            model_matvec(eg_t, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            int kg = op_norm_i32_sp(SP_Y, SP_G, ffn_dim);
            perf_exit(OP_NORM_I32);
            int eg = ex8 - (int)eg_t->bias + (int)eg_t->rshift + kg;

            model_matvec(eu_t, SP_X, SP_YU);
            perf_enter(OP_NORM_I32);
            int ku = op_norm_i32_sp(SP_YU, SP_U, ffn_dim);
            perf_exit(OP_NORM_I32);
            int eu = ex8 - (int)eu_t->bias + (int)eu_t->rshift + ku;

            perf_enter(OP_SILU_MUL);
            op_silu_mul_sp(SP_G, eg, SP_U, eu, ffn_dim);
            perf_exit(OP_SILU_MUL);
            int es = eg + eu + 15;

            perf_enter(OP_QUANT8);
            op_quant8_sp(SP_G, SP_X, ffn_dim);
            perf_exit(OP_QUANT8);
            int es8 = es + 8;

            model_matvec(ed_t, SP_X, SP_Y);
            perf_enter(OP_NORM_I32);
            int kd = op_norm_i32_sp(SP_Y, SP_D, n_embd);
            perf_exit(OP_NORM_I32);
            int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;

            perf_enter(OP_RESID_ADD);
            ex_state = op_resid_add_sp(SP_XR, ex_state, SP_D, ed, n_embd);
            perf_exit(OP_RESID_ADD);
        }
    }
    n_pos++;

    if (!want_logits)
        return -1;

    // Final norm + LM head (tied or separate), chunked over rows to fit the scratchpad.
    perf_enter(OP_RMSNORM);
    op_rmsnorm_sp(SP_XR, ex_state, out_norm, n_embd, SP_H);
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

    heap = (uint8_t *)HEAP_BASE;
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
    k_cache = halloc((uint64_t)n_layer*max_ctx*kv_dim);
    v_cache = halloc((uint64_t)n_layer*max_ctx*kv_dim);
    k_scale = halloc((uint64_t)n_layer*max_ctx*sizeof(float));
    v_scale = halloc((uint64_t)n_layer*max_ctx*sizeof(float));
    scores  = halloc(max_ctx*sizeof(float));
    n_pos   = 0;
    rope_init();
    return 0;
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
        tok = forward(tok, 1);
    }
    return count;
}

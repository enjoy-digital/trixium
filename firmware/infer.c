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

// newlib-libm compatibility: the toolchain libm.a references __errno, absent from the
// picolibc environment LiteX links against.
static int __errno_storage;
int *__errno(void) { return &__errno_storage; }

// Scratchpad layout (byte offsets).
#define SP_X    0x00000 // Quantized activations (engine matvec input).
#define SP_Y    0x08000 // Engine results (int32).
#define SP_FET  0x30000 // Fetch staging (embedding rows).

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

// Quantize float vector to int8 in the scratchpad; returns the scale (absmax/127).
static float quantize_to_sp(const float *src, int n, uint32_t sp_off)
{
    float amax = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float s  = (amax > 0) ? amax/127.0f : 1.0f;
    float is = 1.0f/s;
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
    return s;
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

    float sc = ldexpf(xscale, (int)e->rshift - (int)e->bias);
    volatile int32_t *y = (volatile int32_t *)(SCRATCHPAD_BASE + SP_Y);
    for (uint32_t i = 0; i < rows*n_ch; i++)
        dst[i] = (float)y[i]*sc;
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
static float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base)
{
    return n_dims*logf(n_ctx_orig/(n_rot*2.0f*(float)M_PI))/(2.0f*logf(base));
}

static void rope_apply(float *vec, int n_heads_v, int pos)
{
    // Uses model constants: head_dim, rope theta from header, yarn 4x (Bonsai family).
    float theta_base = (float)H->rope_theta;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float mscale     = 1.0f;
    // Bonsai GGUFs use yarn factor 4; the original ctx differs per model. Encode via
    // max_ctx? Keep simple: yarn only when header vocab is the real one (>100k = real
    // model). Tiny models use plain rope.
    if (H->vocab > 100000) {
        freq_scale = 0.25f;
        ext_factor = 1.0f;
    }
    int   D    = head_dim;
    float low  = 0, high = D - 1;
    if (ext_factor != 0.0f) {
        int orig = (H->vocab > 100000) ? ((n_embd >= 4096) ? 16384 : 8192) : 0;
        low  = floorf(yarn_corr_dim(D, orig, 32.0f, theta_base));
        high = ceilf(yarn_corr_dim(D, orig, 1.0f, theta_base));
        if (low  < 0)     low  = 0;
        if (high > D - 1) high = D - 1;
        mscale *= 1.0f + 0.1f*logf(1.0f/freq_scale);
    }
    for (int hh = 0; hh < n_heads_v; hh++) {
        float *vh = vec + hh*D;
        for (int p2 = 0; p2 < D/2; p2++) {
            float theta_extrap = pos*powf(theta_base, -2.0f*p2/D);
            float theta = freq_scale*theta_extrap;
            if (ext_factor != 0.0f) {
                float y    = (p2 - low)/((high - low) > 0.001f ? (high - low) : 0.001f);
                float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
                theta = theta*(1.0f - ramp*ext_factor) + theta_extrap*ramp*ext_factor;
            }
            float c = cosf(theta)*mscale;
            float s = sinf(theta)*mscale;
            float x0 = vh[p2], x1 = vh[p2 + D/2];
            vh[p2]       = x0*c - x1*s;
            vh[p2 + D/2] = x0*s + x1*c;
        }
    }
}

// Forward one token; returns argmax token id (only meaningful when wanted). ------------------

static int forward(int token, int want_logits)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    embed_row_f(emb, token, x);

    int   kv_dim = n_kv_head*head_dim;
    int   q_dim  = n_head*head_dim;
    float scale  = 1.0f/sqrtf((float)head_dim);
    int   pos    = n_pos;

    for (int l = 0; l < n_layer; l++) {
        // Attention block.
        rmsnorm_f(h, x, attn_norm[l], n_embd, H->rms_eps);
        float sx = quantize_to_sp(h, n_embd, SP_X);
        matvec_rows_f(model_entry(ROLE_ATTN_Q, l), 0, q_dim/n_ch,  sx, q);
        matvec_rows_f(model_entry(ROLE_ATTN_K, l), 0, kv_dim/n_ch, sx, k);
        matvec_rows_f(model_entry(ROLE_ATTN_V, l), 0, kv_dim/n_ch, sx, v);

        for (int hh = 0; hh < n_head; hh++)
            rmsnorm_f(q + hh*head_dim, q + hh*head_dim, q_norm[l], head_dim, H->rms_eps);
        for (int hh = 0; hh < n_kv_head; hh++)
            rmsnorm_f(k + hh*head_dim, k + hh*head_dim, k_norm[l], head_dim, H->rms_eps);
        rope_apply(q, n_head,    pos);
        rope_apply(k, n_kv_head, pos);

        // Append K/V (int8 + per-row scale) to the CPU-side caches.
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

        // Attention (CPU, GQA).
        int n_rep = n_head/n_kv_head;
        for (int hh = 0; hh < n_head; hh++) {
            int kvh = hh/n_rep;
            const float *qh = q + hh*head_dim;
            float smax = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const int8_t *kt = k_cache + ((uint64_t)l*max_ctx + t)*kv_dim + kvh*head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qh[d]*kt[d];
                float sc = dot*k_scale[l*max_ctx + t]*scale;
                scores[t] = sc;
                if (sc > smax) smax = sc;
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - smax);
                sum += scores[t];
            }
            float inv = 1.0f/sum;
            float *ah = attn + hh*head_dim;
            for (int d = 0; d < head_dim; d++) ah[d] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *vt = v_cache + ((uint64_t)l*max_ctx + t)*kv_dim + kvh*head_dim;
                float p = scores[t]*inv*v_scale[l*max_ctx + t];
                for (int d = 0; d < head_dim; d++) ah[d] += p*vt[d];
            }
        }

        // Output projection + residual.
        matvec_f(model_entry(ROLE_ATTN_OUT, l), attn, h);
        for (int i = 0; i < n_embd; i++) x[i] += h[i];

        // FFN.
        rmsnorm_f(h, x, ffn_norm[l], n_embd, H->rms_eps);
        sx = quantize_to_sp(h, n_embd, SP_X);
        matvec_rows_f(model_entry(ROLE_FFN_GATE, l), 0, ffn_dim/n_ch, sx, gate);
        matvec_rows_f(model_entry(ROLE_FFN_UP,   l), 0, ffn_dim/n_ch, sx, up);
        for (int i = 0; i < ffn_dim; i++) {
            float g = gate[i];
            gate[i] = g/(1.0f + expf(-g))*up[i];
        }
        matvec_f(model_entry(ROLE_FFN_DOWN, l), gate, h);
        for (int i = 0; i < n_embd; i++) x[i] += h[i];
    }
    n_pos++;

    if (!want_logits)
        return -1;

    // Final norm + LM head (tied or separate), chunked over rows to fit the scratchpad.
    rmsnorm_f(h, x, out_norm, n_embd, H->rms_eps);
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head)
        head = model_entry(ROLE_TOK_EMB, 0);
    float sx  = quantize_to_sp(h, n_embd, SP_X);
    float sc  = ldexpf(sx, (int)head->rshift - (int)head->bias);
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
    return best_tok;
}

// Public API ---------------------------------------------------------------------------------

int infer_init(void)
{
    H = model_header();
    if (H->magic != BONS_MAGIC)
        return -1;
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

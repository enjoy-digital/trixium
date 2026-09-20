// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// The hybrid models (Bonsai 2 = Qwen3.5: Gated DeltaNet layers, gated attention with partial
// RoPE, PrismML's rotated weight basis) on the CPU path: float math over main_ram, the engine
// for every matvec (the projections and the rotation blocks). This is the correctness path -
// the golden's semantics (trixium/model/golden.py) transliterated, gated on the tiny
// hybrid model in the Verilator sim - and the reference the fast path (vector unit + engine,
// docs/xcu1525.md) is measured against. It is slow by design: soft-float on VexiiRiscv, one
// engine op per rotation block, the DeltaNet state walked head by head from main_ram.
//
// What a hybrid image adds (BONS v3, layout.py): per layer either the attention tensors
// (attn_q carries [q | gate] per head when the header says attn_gated) or the DeltaNet set
// (attn_qkv = [q | k | v], attn_gate = z, ssm_beta/ssm_alpha as Q8 head projections, the
// conv taps, A = -exp(A_log), the dt bias, the head norm, ssm_out); post_attention_norm as the
// FFN norm; the ROLE_HADAMARD block tensors R diag(s) per input width; the emb16 rows already
// restored to the plain basis (HYB_EMB16_RESTORED - the ternary embedding table stays
// rotated and cannot be decoded row by row).

#include <generated/csr.h>
#include <generated/soc.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "bons.h"
#include "model.h"
#include "engine.h"
#include "infer.h"
#include "hybrid.h"

static const struct bons_header *H;
static int n_ch, n_embd, n_head, n_kv_head, head_dim, ffn_dim, n_layer, q_dim, kv_dim, n_rep;
static int nk, hk, nv, hv, nq, nvh, conv_k, rope_n;
static int n_dn, n_attn, ctx_cap, pos;
static float rms_eps;

// Float work buffers (main_ram).
static float *xb, *hb, *rot, *qkv, *zb, *betab, *alphab, *gb, *convb, *qb, *kb, *vb, *ob, *og,
             *gateb, *upb, *attnb, *logits, *scores, *delta, *qfull, *tmp_h;
// Per-layer tables: index by layer (NULL where the layer has no such tensor).
static float **norm_attn, **norm_ffn, **norm_q, **norm_k, **norm_ssm, **ssm_a, **ssm_dt, **conv_w;
static int   *dn_slot, *attn_slot;   // layer -> DeltaNet / attention index, -1 otherwise.
static float *dn_state;              // [n_dn][nv][hv][hk]  (row j = value index, as llama.cpp)
static float *dn_conv;               // [n_dn][conv_k - 1][C]
static float *kc, *vc;               // [n_attn][ctx_cap][kv_dim]
static float *rope_c, *rope_s, *rope_freq;
static int    rope_pos_cached;

int hybrid_present(void)
{
    const struct bons_header *h = model_header();
    return h->magic == BONS_MAGIC && h->version >= 3
        && (h->layer_types[0] || h->layer_types[1] || h->hadamard_block);
}

// Helpers -------------------------------------------------------------------------------------

static float sigmoidf(float x) { return 1.0f/(1.0f + expf(-x)); }
static float siluf(float x)    { return x/(1.0f + expf(-x)); }
static float softplusf(float x){ return x > 20.0f ? x : log1pf(expf(x)); }

static void rmsnorm_fv(float *dst, const float *src, const float *w, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += src[i]*src[i];
    float r = 1.0f/sqrtf(ss/n + rms_eps);
    for (int i = 0; i < n; i++) dst[i] = src[i]*r*w[i];
}

static void l2norm_fv(float *v, int n)   // ggml_l2_norm: v / max(|v|, eps)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += v[i]*v[i];
    float nrm = sqrtf(ss);
    if (nrm < rms_eps) nrm = rms_eps;
    for (int i = 0; i < n; i++) v[i] /= nrm;
}

// A replicated F32 tensor's rows into a float array (chunked through the fetch region).
static float *fetch_f32(const struct bons_entry *e)
{
    if (!e) return NULL;
    uint32_t rb = bons_row_beats(e->cols, e->kind, e->group);
    float *dst = infer_halloc((uint64_t)e->rows*e->cols*sizeof(float), 0);
    if (!dst) return NULL;
    for (uint32_t r = 0; r < e->rows; r++) {
        infer_fetch_f32(BONSAI_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)r*rb)*32, rb,
                        dst + (uint64_t)r*e->cols, e->cols);
    }
    return dst;
}

// The rotation: x' = R (s * x) per block - one engine matvec per block tensor.
static void rotate_f(const float *src, float *dst, int width)
{
    int B = H->hadamard_block;
    if (!B) { memcpy(dst, src, (size_t)width*sizeof(float)); return; }
    int wi = -1;
    for (int i = 0; i < 4; i++) if ((int)H->hadamard_widths[i] == width) wi = i;
    if (wi < 0) { printf("hybrid: no rotation for width %d\n", width); return; }
    for (int b = 0; b < width/B; b++) {
        const struct bons_entry *e = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, b));
        infer_matvec_f(e, src + b*B, dst + b*B);
    }
}

// A projection with a folded weight: rotate the input, then the matvec.
static void proj_f(const struct bons_entry *e, const float *src, float *dst)
{
    if (H->hadamard_block) {
        rotate_f(src, rot, infer_matvec_cols(e));   // The whole width, K-segments included.
        infer_matvec_f(e, rot, dst);
    } else {
        infer_matvec_f(e, src, dst);
    }
}

static void rope_prepare(int p)
{
    if (p == rope_pos_cached) return;
    for (int i = 0; i < rope_n/2; i++) {
        float th = (float)p*rope_freq[i];
        rope_c[i] = cosf(th);
        rope_s[i] = sinf(th);
    }
    rope_pos_cached = p;
}

static void rope_apply_partial(float *vh)   // one head: the first rope_n dims, neox pairs
{
    int half = rope_n/2;
    for (int i = 0; i < half; i++) {
        float x0 = vh[i], x1 = vh[i + half];
        vh[i]        = x0*rope_c[i] - x1*rope_s[i];
        vh[i + half] = x0*rope_s[i] + x1*rope_c[i];
    }
}

// Init ------------------------------------------------------------------------------------------

int hybrid_init(void)
{
    H = model_header();
    n_ch = H->n_channels; n_layer = H->n_layer; n_embd = H->n_embd; n_head = H->n_head;
    n_kv_head = H->n_kv_head; head_dim = H->head_dim; ffn_dim = H->ffn_dim;
    q_dim = n_head*head_dim; kv_dim = n_kv_head*head_dim; n_rep = n_head/n_kv_head;
    nk = H->dn_num_k_heads; hk = H->dn_k_head_dim; nv = H->dn_num_v_heads; hv = H->dn_v_head_dim;
    nq = nk*hk; nvh = nv*hv; conv_k = H->dn_conv_kernel;
    rope_n = H->rope_dims ? H->rope_dims : head_dim;
    rms_eps = H->rms_eps;
    if (H->hadamard_block && !(H->hyb_flags & HYB_EMB16_RESTORED)) {
        printf("hybrid: rotated basis but the emb16 rows are not restored - reconvert\n");
        return -1;
    }
    if (H->hadamard_block && !model_entry_kind(ROLE_TOK_EMB16, 0, KIND_NORM16)) {
        printf("hybrid: rotated basis needs the emb16 rows in the image (window too small?)\n");
        return -1;
    }
    n_dn = n_attn = 0;
    dn_slot = infer_halloc(n_layer*sizeof(int), 0);
    attn_slot = infer_halloc(n_layer*sizeof(int), 0);
    for (int l = 0; l < n_layer; l++) {
        if (bons_layer_is_deltanet(H, l)) { dn_slot[l] = n_dn++; attn_slot[l] = -1; }
        else                              { attn_slot[l] = n_attn++; dn_slot[l] = -1; }
    }
    // The CPU state: fp32 everything. main_ram decides how deep the KV cache goes.
    uint64_t state_b = (uint64_t)n_dn*nv*hv*hk*4 + (uint64_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4;
    ctx_cap = H->max_ctx;
    uint64_t kv_b = (uint64_t)n_attn*ctx_cap*kv_dim*4*2;
    uint64_t room = infer_heap_room(1);
    // Work buffers: the widest activation a few times over, plus the logits.
    uint64_t work_b = 24u*(uint64_t)(ffn_dim > 2*nq + nvh ? ffn_dim : 2*nq + nvh)*4 + 4u*H->vocab + (256u << 10);
    while (ctx_cap > 16 && state_b + kv_b + work_b > room) {
        ctx_cap /= 2;
        kv_b = (uint64_t)n_attn*ctx_cap*kv_dim*4*2;
    }
    if (state_b + kv_b + work_b > room) {
        printf("hybrid: the CPU path needs %lu KB (state %lu, KV %lu, work %lu), main_ram has %lu KB free\n",
               (unsigned long)((state_b + kv_b + work_b) >> 10), (unsigned long)(state_b >> 10),
               (unsigned long)(kv_b >> 10), (unsigned long)(work_b >> 10), (unsigned long)(room >> 10));
        return -1;
    }
    dn_state = n_dn ? infer_halloc((uint64_t)n_dn*nv*hv*hk*4, 1) : NULL;
    dn_conv  = n_dn ? infer_halloc((uint64_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4, 1) : NULL;
    kc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    vc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;

    int wmax = n_embd;
    if (ffn_dim > wmax) wmax = ffn_dim;
    if (q_dim > wmax) wmax = q_dim;
    if (nvh > wmax) wmax = nvh;
    xb = infer_halloc(n_embd*4, 0); hb = infer_halloc(n_embd*4, 0); tmp_h = infer_halloc(n_embd*4, 0);
    rot = infer_halloc(wmax*4, 0);
    qkv = infer_halloc((2*nq + nvh + 16)*4, 0); zb = infer_halloc((nvh + 16)*4, 0);
    betab = infer_halloc((nv + 16)*4, 0); alphab = infer_halloc((nv + 16)*4, 0); gb = infer_halloc((nv + 16)*4, 0);
    convb = infer_halloc((2*nq + nvh + 16)*4, 0);
    qb = infer_halloc((nq > q_dim ? nq : q_dim)*4, 0); kb = infer_halloc((nq > kv_dim ? nq : kv_dim)*4, 0);
    vb = infer_halloc((nvh > kv_dim ? nvh : kv_dim)*4, 0);
    ob = infer_halloc((nvh + 16)*4, 0); og = infer_halloc((nvh + 16)*4, 0);
    delta = infer_halloc((hv + 16)*4, 0);
    qfull = infer_halloc((2*q_dim + 16)*4, 0);
    attnb = infer_halloc((q_dim + 16)*4, 0);
    gateb = infer_halloc(ffn_dim*4, 0); upb = infer_halloc(ffn_dim*4, 0);
    scores = infer_halloc(ctx_cap*4, 0);
    logits = infer_halloc((uint64_t)H->vocab*4 + 64*n_ch*4, 1);
    if (!logits) { printf("hybrid: no room for the logits\n"); return -1; }

    norm_attn = infer_halloc(n_layer*sizeof(float *), 0); norm_ffn = infer_halloc(n_layer*sizeof(float *), 0);
    norm_q = infer_halloc(n_layer*sizeof(float *), 0);    norm_k = infer_halloc(n_layer*sizeof(float *), 0);
    norm_ssm = infer_halloc(n_layer*sizeof(float *), 0);  ssm_a = infer_halloc(n_layer*sizeof(float *), 0);
    ssm_dt = infer_halloc(n_layer*sizeof(float *), 0);    conv_w = infer_halloc(n_layer*sizeof(float *), 0);
    for (int l = 0; l < n_layer; l++) {
        norm_attn[l] = fetch_f32(model_entry_kind(ROLE_ATTN_NORM, l, KIND_F32));
        norm_ffn[l]  = fetch_f32(model_entry_kind(ROLE_FFN_NORM, l, KIND_F32));
        if (!norm_attn[l] || !norm_ffn[l]) { printf("hybrid: layer %d norms missing\n", l); return -2; }
        if (dn_slot[l] >= 0) {
            norm_ssm[l] = fetch_f32(model_entry_kind(ROLE_SSM_NORM, l, KIND_F32));
            ssm_a[l]    = fetch_f32(model_entry_kind(ROLE_SSM_A, l, KIND_F32));
            ssm_dt[l]   = fetch_f32(model_entry_kind(ROLE_SSM_DT, l, KIND_F32));
            conv_w[l]   = fetch_f32(model_entry_kind(ROLE_SSM_CONV, l, KIND_F32));
            if (!norm_ssm[l] || !ssm_a[l] || !ssm_dt[l] || !conv_w[l]) {
                printf("hybrid: layer %d DeltaNet tables missing\n", l); return -2; }
            norm_q[l] = norm_k[l] = NULL;
        } else {
            norm_q[l] = fetch_f32(model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_F32));
            norm_k[l] = fetch_f32(model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_F32));
            if (!norm_q[l] || !norm_k[l]) { printf("hybrid: layer %d q/k norms missing\n", l); return -2; }
        }
    }
    // Plain RoPE over rope_n dims (the 27B: theta 1e7, no YaRN; v3 images carry the
    // scaling fields when a model needs them - none does today).
    rope_c = infer_halloc((rope_n/2)*4, 0); rope_s = infer_halloc((rope_n/2)*4, 0);
    rope_freq = infer_halloc((rope_n/2)*4, 0);
    for (int i = 0; i < rope_n/2; i++)
        rope_freq[i] = powf((float)H->rope_theta, -2.0f*i/rope_n);
    rope_pos_cached = -1;
    hybrid_reset();
    printf("hybrid: %d DeltaNet + %d attention layers, dn %d/%d heads x %d/%d, conv %d, "
           "rope %d/%d, gate %d, hadamard %d (widths", n_dn, n_attn, nv, nk, hv, hk, conv_k,
           rope_n, head_dim, H->attn_gated, H->hadamard_block);
    for (int i = 0; i < 4; i++) if (H->hadamard_widths[i]) printf(" %lu", (unsigned long)H->hadamard_widths[i]);
    printf("), ctx %d, state %lu KB - CPU path\n", ctx_cap, (unsigned long)(state_b >> 10));
    return 0;
}

void hybrid_reset(void)
{
    pos = 0;
    if (dn_state) memset(dn_state, 0, (size_t)n_dn*nv*hv*hk*4);
    if (dn_conv)  memset(dn_conv, 0, (size_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4);
}

int hybrid_pos(void) { return pos; }

// Layers ----------------------------------------------------------------------------------------

// Four rows have no data dependency between them.  Keep their FP32 operation and accumulation
// order exactly the same as the scalar reference, but expose four independent chains to the
// Vexii FPU.  A single row's dot += term chain otherwise serialises on the adder latency.  Do
// not replace these loops with a tree reduction or an FMA: both change the 27B's arithmetic.
static inline void deltanet_rows4(float *M, int stride, const float *kk, const float *qh,
                                  const float *vh, float decay, float beta, float scale,
                                  float *oh)
{
    float *m0 = M, *m1 = M + stride, *m2 = M + 2*stride, *m3 = M + 3*stride;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int i = 0; i < stride; i++) {
        float k = kk[i];
        m0[i] *= decay; d0 += m0[i]*k;
        m1[i] *= decay; d1 += m1[i]*k;
        m2[i] *= decay; d2 += m2[i]*k;
        m3[i] *= decay; d3 += m3[i]*k;
    }
    d0 = (vh[0] - d0)*beta;
    d1 = (vh[1] - d1)*beta;
    d2 = (vh[2] - d2)*beta;
    d3 = (vh[3] - d3)*beta;
    float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    for (int i = 0; i < stride; i++) {
        float k = kk[i], q = qh[i];
        m0[i] += d0*k; o0 += m0[i]*q;
        m1[i] += d1*k; o1 += m1[i]*q;
        m2[i] += d2*k; o2 += m2[i]*q;
        m3[i] += d3*k; o3 += m3[i]*q;
    }
    oh[0] = o0*scale; oh[1] = o1*scale; oh[2] = o2*scale; oh[3] = o3*scale;
}

static void deltanet_layer(int l, const float *h, float *out)
{
    int   s = dn_slot[l];
    int   C = 2*nq + nvh;
    float *S    = dn_state + (uint64_t)s*nv*hv*hk;
    float *win  = dn_conv + (uint64_t)s*(conv_k - 1)*C;

    proj_f(model_entry(ROLE_ATTN_QKV, l), h, qkv);
    proj_f(model_entry(ROLE_ATTN_ZGATE, l), h, zb);
    infer_matvec_f(model_entry(ROLE_SSM_BETA, l), h, betab);    // not folded
    infer_matvec_f(model_entry(ROLE_SSM_ALPHA, l), h, alphab);
    for (int hh = 0; hh < nv; hh++) {
        betab[hh] = sigmoidf(betab[hh]);
        gb[hh]    = softplusf(alphab[hh] + ssm_dt[l][hh])*ssm_a[l][hh];   // ssm_a = -exp(A_log)
    }
    // Causal conv over the last conv_k inputs of every channel, then SiLU.
    const float *w = conv_w[l];   // tap i of channel c at c*conv_k + i
    for (int c = 0; c < C; c++) {
        float acc = 0;
        for (int i = 0; i < conv_k - 1; i++) acc += w[c*conv_k + i]*win[i*C + c];
        acc += w[c*conv_k + conv_k - 1]*qkv[c];
        convb[c] = siluf(acc);
    }
    for (int i = 0; i < conv_k - 2; i++)
        memcpy(win + i*C, win + (i + 1)*C, (size_t)C*4);
    memcpy(win + (conv_k - 2)*C, qkv, (size_t)C*4);
    // q, k (per key head, L2-normalised), v.
    memcpy(qb, convb, (size_t)nq*4);
    memcpy(kb, convb + nq, (size_t)nq*4);
    memcpy(vb, convb + 2*nq, (size_t)nvh*4);
    for (int hh = 0; hh < nk; hh++) { l2norm_fv(qb + hh*hk, hk); l2norm_fv(kb + hh*hk, hk); }
    // The recurrence per value head (llama.cpp's autoregressive kernel): M = S[hh] rows j.
    float scale = 1.0f/sqrtf((float)hk);
    int rep = nv/nk;
    for (int hh = 0; hh < nv; hh++) {
        int kh = hh % nk;
        const float *qh = qb + kh*hk, *kk = kb + kh*hk, *vh = vb + hh*hv;
        float *M = S + (uint64_t)hh*hv*hk;
        float decay = expf(gb[hh]);
        float *oh = ob + hh*hv;
        // One pass over the state: row j's delta depends on row j alone, so the decay,
        // the k dot, the update and the q dot happen while the row (512 B) is in the
        // D$ - the two-loop form streamed the 64 KB head twice through a 32 KB cache.
        // Same operations in the same order per element: bit-identical results.
        int j = 0;
        for (; j + 3 < hv; j += 4)
            deltanet_rows4(M + j*hk, hk, kk, qh, vh + j, decay, betab[hh], scale, oh + j);
        for (; j < hv; j++) {
            float *Mj = M + j*hk, dot = 0;
            for (int i = 0; i < hk; i++) { Mj[i] *= decay; dot += Mj[i]*kk[i]; }
            float dj = (vh[j] - dot)*betab[hh];
            dot = 0;
            for (int i = 0; i < hk; i++) { Mj[i] += dj*kk[i]; dot += Mj[i]*qh[i]; }
            oh[j] = dot*scale;
        }
        // Gated RMSNorm over the head with the shared head norm, times SiLU(z); written in
        // grouped head order (key head major) - the order the folded out projection expects.
        rmsnorm_fv(delta, oh, norm_ssm[l], hv);
        int gpos = kh*rep + hh/nk;
        float *dst = og + gpos*hv;
        const float *z = zb + hh*hv;
        for (int j = 0; j < hv; j++) dst[j] = delta[j]*siluf(z[j]);
    }
    proj_f(model_entry(ROLE_SSM_OUT, l), og, out);
}

static void attention_layer(int l, const float *h, float *out)
{
    int a = attn_slot[l];
    float *K = kc + (uint64_t)a*ctx_cap*kv_dim, *V = vc + (uint64_t)a*ctx_cap*kv_dim;
    int gated = H->attn_gated;
    proj_f(model_entry(ROLE_ATTN_Q, l), h, gated ? qfull : qb);
    proj_f(model_entry(ROLE_ATTN_K, l), h, kb);
    proj_f(model_entry(ROLE_ATTN_V, l), h, vb);
    if (gated)   // [q | gate] per head
        for (int hh = 0; hh < n_head; hh++)
            memcpy(qb + hh*head_dim, qfull + hh*2*head_dim, (size_t)head_dim*4);
    rope_prepare(pos);
    for (int hh = 0; hh < n_head; hh++) {
        rmsnorm_fv(qb + hh*head_dim, qb + hh*head_dim, norm_q[l], head_dim);
        rope_apply_partial(qb + hh*head_dim);
    }
    for (int hh = 0; hh < n_kv_head; hh++) {
        rmsnorm_fv(kb + hh*head_dim, kb + hh*head_dim, norm_k[l], head_dim);
        rope_apply_partial(kb + hh*head_dim);
    }
    memcpy(K + (uint64_t)pos*kv_dim, kb, (size_t)kv_dim*4);
    memcpy(V + (uint64_t)pos*kv_dim, vb, (size_t)kv_dim*4);
    int   T = pos + 1;
    float scale = 1.0f/sqrtf((float)head_dim);
    for (int hh = 0; hh < n_head; hh++) {
        int kvh = hh/n_rep;
        const float *qh = qb + hh*head_dim;
        float mx = -1e30f;
        for (int t = 0; t < T; t++) {
            const float *kt = K + (uint64_t)t*kv_dim + kvh*head_dim;
            float sc = 0;
            for (int i = 0; i < head_dim; i++) sc += qh[i]*kt[i];
            scores[t] = sc*scale;
            if (scores[t] > mx) mx = scores[t];
        }
        float sum = 0;
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        float *oh = attnb + hh*head_dim;
        for (int i = 0; i < head_dim; i++) oh[i] = 0;
        for (int t = 0; t < T; t++) {
            const float *vt = V + (uint64_t)t*kv_dim + kvh*head_dim;
            float p = scores[t]/sum;
            for (int i = 0; i < head_dim; i++) oh[i] += p*vt[i];
        }
        if (gated) {
            const float *g = qfull + hh*2*head_dim + head_dim;
            for (int i = 0; i < head_dim; i++) oh[i] *= sigmoidf(g[i]);
        }
    }
    proj_f(model_entry(ROLE_ATTN_OUT, l), attnb, out);
}

int hybrid_forward(int token, int want_logits)
{
    if (pos >= ctx_cap) { printf("hybrid: context cap %d reached\n", ctx_cap); return -1; }
    if (infer_embed_f(token, xb) < 0) return -1;
    for (int l = 0; l < n_layer; l++) {
        rmsnorm_fv(hb, xb, norm_attn[l], n_embd);
        if (dn_slot[l] >= 0) deltanet_layer(l, hb, tmp_h);
        else                 attention_layer(l, hb, tmp_h);
        for (int i = 0; i < n_embd; i++) xb[i] += tmp_h[i];
        rmsnorm_fv(hb, xb, norm_ffn[l], n_embd);
        if (H->hadamard_block) rotate_f(hb, rot, n_embd); else memcpy(rot, hb, (size_t)n_embd*4);
        infer_matvec_f(model_entry(ROLE_FFN_GATE, l), rot, gateb);
        infer_matvec_f(model_entry(ROLE_FFN_UP, l), rot, upb);
        for (int i = 0; i < ffn_dim; i++) gateb[i] = siluf(gateb[i])*upb[i];
        proj_f(model_entry(ROLE_FFN_DOWN, l), gateb, tmp_h);
        for (int i = 0; i < n_embd; i++) xb[i] += tmp_h[i];
    }
    pos++;
    if (!want_logits) return -1;
    const struct bons_entry *on = model_entry_kind(ROLE_OUT_NORM, 0, KIND_F32);
    static float *out_norm; if (!out_norm) out_norm = fetch_f32(on);
    rmsnorm_fv(hb, xb, out_norm, n_embd);
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head) head = model_entry(ROLE_TOK_EMB, 0);
    proj_f(head, hb, logits);
    int best = 0; float bv = logits[0];
    for (uint32_t i = 1; i < H->vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    return best;
}

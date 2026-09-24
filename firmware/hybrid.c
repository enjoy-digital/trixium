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
#include "bonsai_sequence.h"

// The resident DeltaNet layer needs units 11 (SiLU), 12 (norm) and 13 (conv).
#if defined(BONSAI_HAS_NORM) && defined(BONSAI_HAS_CONV) && defined(BONSAI_HAS_SILU)
#define DNRES_BUILT 1
#endif
#include "perf.h"
#include "fastmath.h"
#include "resident.h"
#include "static_memory.h"

static const struct bons_header *H;
static int n_ch, n_embd, n_head, n_kv_head, head_dim, ffn_dim, n_layer, q_dim, kv_dim, n_rep;
static int nk, hk, nv, hv, nq, nvh, conv_k, rope_n;
static int n_dn, n_attn, ctx_cap, pos;
static float rms_eps;

// Float work buffers (main_ram).
static float *xb, *hb, *rot, *qkv, *zb, *betab, *alphab, *gb, *convb, *qb, *kb, *vb, *ob, *og,
             *gateb, *upb, *actb, *attnb, *logits, *scores, *delta, *qfull, *tmp_h;
// Per-layer tables: index by layer (NULL where the layer has no such tensor).
static float **norm_attn, **norm_ffn, **norm_q, **norm_k, **norm_ssm, **ssm_a, **ssm_dt, **conv_w;
static int   *dn_slot, *attn_slot;   // layer -> DeltaNet / attention index, -1 otherwise.
static float *dn_state;              // [n_dn][nv][hv][hk]  (row j = value index, as llama.cpp)
static float *dn_conv;               // [n_dn][conv_k - 1][C]
static uint8_t *dn_conv_pos;         // Oldest causal-convolution plane per DeltaNet layer.
static float *kc, *vc;               // [n_attn][ctx_cap][kv_dim]
static float *rope_c, *rope_s, *rope_freq;
static int    rope_pos_cached;

// Exact-FP32 recurrence accelerator. The fixed gateware geometry is the 27B's 128x128
// matrix; other hybrid test models keep using the scalar CPU implementation below.
static int      dn_hw;
static uint32_t dn_hbm_base;

// The first board run needs to distinguish a compute limit from a DDR/scratchpad/dispatch
// limit. rdcycle is cheap enough to bracket every head without perturbing the datapath;
// report per-head and per-forward averages so the 32-bit minimal printf remains sufficient
// even for a long 27B benchmark.
static uint64_t dn_cyc_fetch, dn_cyc_params, dn_cyc_core, dn_cyc_store, dn_cyc_output;
static uint32_t dn_heads_profiled;
static uint64_t dn_cyc_inproj, dn_cyc_act, dn_cyc_conv, dn_cyc_prep;
static uint64_t dn_cyc_heads, dn_cyc_gate, dn_cyc_outproj;
static uint64_t hyb_cyc_embed, hyb_cyc_anorm, hyb_cyc_dn, hyb_cyc_attn;
static uint64_t hyb_cyc_resid, hyb_cyc_fprep, hyb_cyc_fproj, hyb_cyc_fact;
static uint64_t hyb_cyc_fdown, hyb_cyc_logits;
// Attention sub-stages (sum = hyb_cyc_attn): prepare, Q/K/V projections, per-head norms +
// RoPE, scores/softmax/values/gate (incl. the KV write), output projection.
// Weight beats one forward streams per channel (every matvec tensor once, K-segments
// included, the LM head once): the bandwidth floor is this many cycles at one 32-byte
// beat per channel per clock. New profile counters live here, in the main_ram static
// window: the 8 KiB SRAM's .bss is at its 4 KiB stack floor (48 more bytes broke the link).
struct hyb_prof {
    uint64_t at_prep, at_qkv, at_rope, at_core, at_out;
    uint64_t fwd_weight_beats;
    uint32_t dn_state_sp;
    uint8_t dnres_enabled;           // console `dnres on|off`: the resident DeltaNet layer
    uint8_t xres_enabled;            // console `xres on|off`: the resident residual stream
};
#define HP ((struct hyb_prof *)BONSAI_HYBPROF_ADDR)
_Static_assert(sizeof(struct hyb_prof) <= 256, "hyb_prof window is 256 bytes");
static uint32_t hyb_forwards_profiled;

// The 27B evaluates SiLU about 1.9 million times per forward.  Even the range-reduced
// polynomial in fastmath.h is limited by its dependent FMA chain and FP divide.  Linear
// interpolation at 1/128 spacing has <4.4e-6 absolute error over [-16, 16], while this
// 16 KiB table and the activations' much smaller hot interval fit the 32 KiB D-cache.
static float *silu_lut;

// Console `dnpipe`: DeltaNet head state store/fetch pipelining (DNREC only). Default on since
// the v12 board run (2026-09-23): bit-exact in all four resident/dnpipe configurations,
// 23,709 -> 17,231 cycles per head (docs/measurements/v12-2026-09-23.md).
int dn_pipe_use = 1;

#ifdef BONSAI_HAS_DNREC
#define DN_SP_STATE 0x00000u
#define DN_SP_Q     0x10000u
#define DN_SP_K     0x10200u
#define DN_SP_V     0x10400u
#define DN_SP_OUT   0x10600u
#define DN_HW_ROWS  128u
#define DN_HW_COLS  128u
#define DN_HEAD_BYTES (DN_HW_ROWS*DN_HW_COLS*4u)
#define DN_HEAD_BEATS (DN_HEAD_BYTES/32u)

static uint32_t f32_bits(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    return v.u;
}

// Scratchpad staging of the head state (HP->dn_sp_state): DN_SP_STATE for the CPU-side
// layer, RS_F for the resident layer (whose q/k/v and outputs sit where DN_SP_STATE is).
#define dn_sp_state (HP->dn_state_sp)

static uint32_t dn_head_addr(uint32_t global_head)
{
    return dn_hbm_base + (global_head/(uint32_t)n_ch)*DN_HEAD_BYTES;
}

// Head pipelining (console `dnpipe on|off`, default on since the v12 board run): the
// state STORE of head i and the state FETCH of head i+1 are submitted back to back
// without waiting (the CSR front holds one pending op behind the running one, and the
// engine runs them in order, so the FETCH cannot overwrite the state before the STORE
// has read it) and run under the CPU's output read and next-head parameter writes.
// Measured v8: fetch 2650 + store 4915 cycles per head against 3114 + 4898 of CPU work
// (build/v0-826f4ae-bench8.log), 2304 heads per 27B forward.
static int dn_pipe_prefetched = -1;   // Global head whose state FETCH is in flight, or -1.
static uint32_t dn_pipe_done0;        // done_count before the STORE: FETCH done at +2.
static int dn_pipe_store_open;        // A STORE was submitted and never joined.

static void dn_pipe_join(void)
{
    if (!dn_pipe_store_open)
        return;
    uint32_t need = dn_pipe_prefetched >= 0 ? 2u : 1u, spins = 0;
    while ((uint32_t)(engine_csr_done_count_read() - dn_pipe_done0) < need) {
        if (++spins > 4000000u) {
            puts("DNSTALL: pipelined state store/fetch did not complete");
            engine_wedged = 1;
            break;
        }
    }
    dn_pipe_store_open = 0;
}

static void dn_hw_head(uint32_t global_head, const float *qh, const float *kh,
                       const float *vh, float decay, float beta, float scale, float *oh,
                       int load_qk, int next_head)
{
    uint8_t chan = global_head % (uint32_t)n_ch;
    uint64_t t0 = perf_now64();
    if (dn_pipe_prefetched == (int)global_head) {
        dn_pipe_join();                 // The prefetched state is in the scratchpad once both landed.
        dn_pipe_prefetched = -1;
    } else {
        dn_pipe_join();                 // Never let a FETCH race an unjoined STORE.
        dn_pipe_prefetched = -1;
        engine_fetch(chan, dn_head_addr(global_head), DN_HEAD_BEATS, dn_sp_state);
    }
    dn_cyc_fetch += perf_now64() - t0;

    t0 = perf_now64();
    volatile float *sq = (volatile float *)(SCRATCHPAD_BASE + DN_SP_Q);
    volatile float *sk = (volatile float *)(SCRATCHPAD_BASE + DN_SP_K);
    volatile float *sv = (volatile float *)(SCRATCHPAD_BASE + DN_SP_V);
    // qh == NULL: the caller pointed the unit's q/k/v bases at scratchpad data (resident
    // layer); nothing to copy.
    if (qh && load_qk)
        for (uint32_t i = 0; i < DN_HW_COLS; i++) { sq[i] = qh[i]; sk[i] = kh[i]; }
    if (qh)
        for (uint32_t i = 0; i < DN_HW_ROWS; i++) sv[i] = vh[i];

    dnunit_decay_write(f32_bits(decay));
    dnunit_beta_write(f32_bits(beta));
    dnunit_scale_write(f32_bits(scale));
    dn_cyc_params += perf_now64() - t0;

    t0 = perf_now64();
    uint32_t done = dnunit_done_count_read();
    dnunit_submit_write(1);
    uint32_t spins = 0;
    while (dnunit_done_count_read() == done && ++spins < 100000u) {}
    if (dnunit_done_count_read() == done) {
        puts("DNSTALL: recurrence command did not complete");
        engine_wedged = 1;
        return;
    }
    dn_cyc_core += perf_now64() - t0;

    t0 = perf_now64();
    if (dn_pipe_use && !engine_wedged) {
        struct engine_op st = {0};
        st.kind = ENGINE_KIND_STORE; st.chan = chan;
        st.wbase = dn_head_addr(global_head); st.wlen = DN_HEAD_BEATS; st.xsrc = dn_sp_state;
        dn_pipe_done0 = engine_submit(&st);
        dn_pipe_store_open = 1;
        if (next_head >= 0) {
            struct engine_op ft = {0};
            ft.kind = ENGINE_KIND_FETCH; ft.chan = (uint8_t)((uint32_t)next_head % (uint32_t)n_ch);
            ft.wbase = dn_head_addr((uint32_t)next_head); ft.wlen = DN_HEAD_BEATS; ft.dst = dn_sp_state;
            engine_submit(&ft);         // Queued behind the STORE; joined by the next head.
            dn_pipe_prefetched = next_head;
        }
    } else {
        engine_store(chan, dn_head_addr(global_head), DN_HEAD_BEATS, dn_sp_state);
    }
    dn_cyc_store += perf_now64() - t0;

    t0 = perf_now64();
    volatile float *so = (volatile float *)(SCRATCHPAD_BASE + DN_SP_OUT);
    if (oh)                                // NULL: the output base is the resident layer's
        for (uint32_t i = 0; i < DN_HW_ROWS; i++) oh[i] = so[i];
    dn_cyc_output += perf_now64() - t0;
    dn_heads_profiled++;
}

static void dn_hw_reset(void)
{
    // Fill one matrix once, then let the engine's multi-store repeat it across consecutive
    // per-channel slots (up to 31 packets per command). This clears the full 144 MiB 27B
    // state without walking it over the CPU/main-RAM path.
    volatile uint32_t *z = (volatile uint32_t *)(SCRATCHPAD_BASE + DN_SP_STATE);
    for (uint32_t i = 0; i < DN_HEAD_BYTES/4; i++) z[i] = 0;
    uint32_t total = (uint32_t)n_dn*(uint32_t)nv;
    for (uint32_t c = 0; c < (uint32_t)n_ch; c++) {
        uint32_t count = (total + (uint32_t)n_ch - 1u - c)/(uint32_t)n_ch;
        uint32_t slot = 0;
        while (slot < count) {
            uint32_t batch = count - slot;
            if (batch > 31) batch = 31;
            struct engine_op op = {0};
            op.kind    = ENGINE_KIND_STORE;
            op.chan    = c;
            op.wbase   = dn_hbm_base + slot*DN_HEAD_BYTES;
            op.wlen    = DN_HEAD_BEATS;
            op.xsrc    = DN_SP_STATE;
            op.rep     = batch;
            op.dstride = DN_HEAD_BEATS;
            op.sstride = 0;
            engine_run(&op);
            if (engine_wedged) return;
            slot += batch;
        }
    }
}
#endif

void hybrid_dnres_enable(int on) { HP->dnres_enabled = on ? 1 : 0; }
void hybrid_xres_enable(int on) { HP->xres_enabled = on ? 1 : 0; }
int hybrid_xres_enabled(void) { return HP->xres_enabled && resident_active(); }
int hybrid_dnres_enabled(void)
{
#ifdef DNRES_BUILT
    return HP->dnres_enabled && resident_silu_active();
#else
    return 0;
#endif
}

int hybrid_present(void)
{
    const struct bons_header *h = model_header();
    return h->magic == BONS_MAGIC && h->version >= 3
        && (h->layer_types[0] || h->layer_types[1] || h->hadamard_block);
}

// Helpers -------------------------------------------------------------------------------------

static float *sigmoid_lut;
static float sigmoidf(float x) { return bonsai_sigmoid_lookup(x, sigmoid_lut); }
static float softplusf(float x){ return x > 20.0f ? x : log1pf(expf(x)); }

static void silu_lut_init(void)
{
    for (int i = 0; i <= BONSAI_SILU_LUT_N; i++) {
        float x = -BONSAI_SILU_LUT_LIMIT + (float)i*(1.0f/BONSAI_SILU_LUT_SCALE);
        silu_lut[i] = x/(1.0f + expf(-x));
    }
}

static inline float siluf(float x)
{
    return bonsai_silu_lookup(x, silu_lut);
}

// The canonical FP32 sum of squares (docs/canonical-reductions.md): 64 partial sums, element
// i into partial i % 64 in index order (each term x*x rounded, then added, starting from
// +0), then a halving tree P[j] += P[j + w] for w = 32, 16, ..., 1. A scratchpad unit that
// takes one 8-float beat per clock revisits each partial every 8 clocks, so it can use a
// pipelined adder and still reproduce this bit for bit; the mirror does the same.
static float sumsq64(const float *v, int n)
{
    float P[64];
    for (int k = 0; k < 64; k++) {
        float acc = 0;
        for (int i = k; i < n; i += 64) acc += v[i]*v[i];
        P[k] = acc;
    }
    for (int w = 32; w; w >>= 1)
        for (int j = 0; j < w; j++) P[j] += P[j + w];
    return P[0];
}

static void rmsnorm_fv(float *dst, const float *src, const float *w, int n)
{
    float ss = sumsq64(src, n);
    int i;
    float r = 1.0f/sqrtf(ss/n + rms_eps);
    i = 0;
    for (; i + 3 < n; i += 4) {
        dst[i + 0] = src[i + 0]*r*w[i + 0];
        dst[i + 1] = src[i + 1]*r*w[i + 1];
        dst[i + 2] = src[i + 2]*r*w[i + 2];
        dst[i + 3] = src[i + 3]*r*w[i + 3];
    }
    for (; i < n; i++) dst[i] = src[i]*r*w[i];
}

static void l2norm_fv(float *v, int n)   // ggml_l2_norm: v / max(|v|, eps)
{
    float ss = sumsq64(v, n);
    int i;
    float nrm = sqrtf(ss);
    if (nrm < rms_eps) nrm = rms_eps;
    // Canonical (docs/canonical-reductions.md): one correctly rounded reciprocal, then a
    // multiply per element - what the norm unit does - not a divide per element.
    float inv = 1.0f/nrm;
    i = 0;
    for (; i + 3 < n; i += 4) {
        v[i + 0] *= inv; v[i + 1] *= inv;
        v[i + 2] *= inv; v[i + 3] *= inv;
    }
    for (; i < n; i++) v[i] *= inv;
}

static void silu_mul_fv(float *dst, const float *mul, const float *gate, int n)
{
    int i = 0;
    // Independent lanes hide LUT-load and FMA latency on the in-order RV32F core.  dst may
    // alias gate (the FFN case); no element reads another element, so this remains exact.
    for (; i + 3 < n; i += 4) {
        float s0 = siluf(gate[i + 0]), s1 = siluf(gate[i + 1]);
        float s2 = siluf(gate[i + 2]), s3 = siluf(gate[i + 3]);
        dst[i + 0] = mul[i + 0]*s0; dst[i + 1] = mul[i + 1]*s1;
        dst[i + 2] = mul[i + 2]*s2; dst[i + 3] = mul[i + 3]*s3;
    }
    for (; i < n; i++) dst[i] = mul[i]*siluf(gate[i]);
}

static void add_fv(float *dst, const float *src, int n)
{
    int i = 0;
    for (; i + 3 < n; i += 4) {
        dst[i + 0] += src[i + 0]; dst[i + 1] += src[i + 1];
        dst[i + 2] += src[i + 2]; dst[i + 3] += src[i + 3];
    }
    for (; i < n; i++) dst[i] += src[i];
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

// Portable fallback for targets without scratchpad STORE + cache invalidation.  The XCU1525
// uses the fused path below; keeping this implementation preserves every other hybrid target.
static void rotate_f_legacy(const float *src, float *dst, int width)
{
    int B = H->hadamard_block, wi = -1;
    for (int i = 0; i < 4; i++) if ((int)H->hadamard_widths[i] == width) wi = i;
    if (wi < 0) { printf("hybrid: no rotation for width %d\n", width); return; }
    for (int b = 0; b < width/B; b++) {
        const struct bons_entry *e = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, b));
        infer_matvec_f(e, src + b*B, dst + b*B);
    }
}

static int rotate_quantize_legacy(const float *src, int width)
{
    int e8 = infer_rotate_quantize_f(src, (uint32_t)width);
    if (e8 != INFER_ROTATE_UNSUPPORTED) return e8;
    rotate_f_legacy(src, rot, width);
    return infer_quantize_f(rot, (uint32_t)width);
}

// Resident preparation (resident.c: units 4/3/5 in the scratchpad) with the legacy CPU
// path as fallback. The prepared source is remembered so a matvec the resident path
// declines (LM head rows, a failed list) can requantize on the CPU and go the old way.
#define E8_RESIDENT 0x40000001

static int prepare_f(const float *src, int width, int rotate)
{
    struct resident_state *R = resident_state();
    R->prep_src = src; R->prep_width = width; R->prep_rotate = rotate;
    if (resident_prepare(src, (uint32_t)width, rotate) == 0)
        return E8_RESIDENT;
    return rotate ? rotate_quantize_legacy(src, width) : infer_quantize_f(src, (uint32_t)width);
}

static int rotate_quantize_f(const float *src, int width)
{
    return prepare_f(src, width, 1);
}

static int quantize_f(const float *src, int width)
{
    return prepare_f(src, width, 0);
}

static void matvec_q(const struct bons_entry *e, int e8, float *dst)
{
    if (e8 == E8_RESIDENT) {
        if (resident_matvec(e, dst) == 0)
            return;
        // Declined or failed: the CPU path needs its own Q8 in SP_X.
        const struct resident_state *R = resident_state();
        e8 = R->prep_rotate ? rotate_quantize_legacy(R->prep_src, R->prep_width)
                            : infer_quantize_f(R->prep_src, (uint32_t)R->prep_width);
        if (e8 == E8_RESIDENT) return;   // Cannot happen: the legacy path never returns it.
    }
    if (e->flags & FLAG_KSEG)
        infer_matvec_f_rotated_quantized(e, e8, dst);
    else
        infer_matvec_f_quantized(e, e8, dst);
}

// A projection with a folded weight: rotate the input, then the matvec.
static void proj_f(const struct bons_entry *e, const float *src, float *dst)
{
    if (H->hadamard_block) {
        uint32_t width = infer_matvec_cols(e);      // The whole width, K-segments included.
        int e8 = prepare_f(src, (int)width, 1);
        if (e8 == E8_RESIDENT) {
            matvec_q(e, e8, dst);
        } else if (e8 != INFER_ROTATE_UNSUPPORTED) {
            infer_matvec_f_rotated_quantized(e, e8, dst);
        } else {
            rotate_f_legacy(src, rot, (int)width);
            infer_matvec_f(e, rot, dst);
        }
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
    dn_hw = 0;
#ifdef BONSAI_HAS_DNREC
    dn_hw = n_dn && hv == DN_HW_ROWS && hk == DN_HW_COLS;
    if (dn_hw) {
        // Keep the same conservative KV reservation as the image converter, then place
        // whole recurrent matrices immediately above it. Hybrid attention currently keeps
        // its KV cache in main RAM, but reserving the range makes this layout future-proof.
        uint32_t kv_base = (uint32_t)((BONSAI_CHAN_WEIGHT_BASE + H->payload_beats*32
                                      + 0xFFFu) & ~0xFFFu);
        uint32_t kv_rb = 1u + (uint32_t)head_dim/32u;
        uint32_t kv_region = ((uint32_t)H->max_ctx + (uint32_t)n_ch - 1u)
                           /(uint32_t)n_ch*kv_rb;
        uint64_t base = (uint64_t)kv_base
                      + (uint64_t)n_layer*n_kv_head*2u*kv_region*32u;
        dn_hbm_base = (uint32_t)((base + 0xFFFu) & ~0xFFFull);
        uint64_t state_per_channel = ((uint64_t)n_dn*nv + n_ch - 1u)/n_ch*DN_HEAD_BYTES;
        uint32_t skew_count = BONSAI_CHAN_MOD ? BONSAI_CHAN_MOD - 1u : 0u;
        uint64_t limit = (uint64_t)BONSAI_CHAN_WEIGHT_BASE + BONSAI_CHAN_STRIDE
                       - (BONSAI_CHAN_WEIGHT_BASE % BONSAI_CHAN_STRIDE) - 0x1000u
                       - (uint64_t)skew_count*BONSAI_CHAN_SKEW;
        if ((uint64_t)dn_hbm_base + state_per_channel > limit) {
            printf("hybrid: DeltaNet state reaches %08lx, past channel limit %08lx - "
                   "reconvert for this board\n",
                   (unsigned long)((uint64_t)dn_hbm_base + state_per_channel),
                   (unsigned long)limit);
            return -1;
        }
        HP->dn_state_sp = DN_SP_STATE;
        HP->dnres_enabled = 1;
        dnunit_state_base_write(DN_SP_STATE);
        dnunit_q_base_write(DN_SP_Q);
        dnunit_k_base_write(DN_SP_K);
        dnunit_v_base_write(DN_SP_V);
        dnunit_out_base_write(DN_SP_OUT);
    }
#endif
    // FP32 state. The matrix lives in engine DDR when the recurrence accelerator is
    // available; the short causal-convolution history remains in main RAM either way.
    uint64_t matrix_b = (uint64_t)n_dn*nv*hv*hk*4;
    uint64_t conv_b = (uint64_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4;
    uint64_t state_b = (dn_hw ? 0 : matrix_b) + conv_b;
    ctx_cap = H->max_ctx;
    uint64_t kv_b = (uint64_t)n_attn*ctx_cap*kv_dim*4*2;
    uint64_t room = infer_heap_room(1);
    // Work buffers: the widest activation a few times over, plus the logits.
    uint64_t work_b = 24u*(uint64_t)(ffn_dim > 2*nq + nvh ? ffn_dim : 2*nq + nvh)*4
                    + 4u*H->vocab + (256u << 10) + sizeof(float)*(BONSAI_SILU_LUT_N + 1);
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
    dn_state = (n_dn && !dn_hw) ? infer_halloc(matrix_b, 1) : NULL;
    dn_conv  = n_dn ? infer_halloc((uint64_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4, 1) : NULL;
    dn_conv_pos = n_dn ? infer_halloc(n_dn, 0) : NULL;
    kc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    vc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    silu_lut = infer_halloc(sizeof(float)*(BONSAI_SILU_LUT_N + 1), 0);
    silu_lut_init();
    sigmoid_lut = infer_halloc(sizeof(float)*(BONSAI_SILU_LUT_N + 1), 0);
    for (int i = 0; i <= BONSAI_SILU_LUT_N; i++)
        sigmoid_lut[i] = bonsai_sigmoid_point(-BONSAI_SILU_LUT_LIMIT + (float)i*(1.0f/BONSAI_SILU_LUT_SCALE));

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
    actb = infer_halloc(ffn_dim*4, 0);   // Resident FFN activation (gateb keeps the gate trace).
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
    HP->dnres_enabled = 0;
    HP->xres_enabled = 0;
    if (H->hadamard_block && resident_available()) {
        if (resident_init() == 0) {
            puts("hybrid: resident projections (scratchpad quantize/rotate/restore lists) on");
            uint32_t *stage = infer_halloc(1025*32, 0);   // Table image for the DMA.
            if (resident_silu_init(silu_lut, stage) == 0)
                puts("hybrid: resident FFN activation (unit 11 SiLU-multiply) on");
#ifdef DNRES_BUILT
            HP->dnres_enabled = 1;
            HP->xres_enabled = 1;
            if (n_dn && resident_silu_active())
                puts("hybrid: resident DeltaNet layer (conv/norm/SiLU units, lists) on");
#endif
        }
    }
    {
        static const int roles[] = {ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
            ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN, ROLE_ATTN_QKV, ROLE_ATTN_ZGATE,
            ROLE_SSM_OUT, ROLE_SSM_BETA, ROLE_SSM_ALPHA};
        HP->fwd_weight_beats = 0;
        for (int l = 0; l < n_layer; l++)
            for (unsigned r = 0; r < sizeof roles/sizeof roles[0]; r++) {
                const struct bons_entry *e = model_entry(roles[r], l);
                int nseg = e && (e->flags & FLAG_KSEG) ? model_entry_nkseg(e) : 1;
                for (int k = 0; e && k < nseg; k++)
                    HP->fwd_weight_beats += (nseg > 1 ? model_entry_kseg(e, k) : e)->chan_len_beats;
            }
        const struct bons_entry *lm = model_entry(ROLE_LM_HEAD, 0);
        if (!lm) lm = model_entry(ROLE_TOK_EMB, 0);
        if (lm) HP->fwd_weight_beats += lm->chan_len_beats;
    }
    printf("hybrid: %d DeltaNet + %d attention layers, dn %d/%d heads x %d/%d, conv %d, "
           "rope %d/%d, gate %d, hadamard %d (widths", n_dn, n_attn, nv, nk, hv, hk, conv_k,
           rope_n, head_dim, H->attn_gated, H->hadamard_block);
    for (int i = 0; i < 4; i++) if (H->hadamard_widths[i]) printf(" %lu", (unsigned long)H->hadamard_widths[i]);
    if (dn_hw)
        printf("), ctx %d, state %lu KB + %lu KB engine DDR at %08lx - HW recurrence\n",
               ctx_cap, (unsigned long)(conv_b >> 10), (unsigned long)(matrix_b >> 10),
               (unsigned long)dn_hbm_base);
    else
        printf("), ctx %d, state %lu KB - CPU path\n", ctx_cap,
               (unsigned long)(state_b >> 10));
    return 0;
}

void hybrid_reset(void)
{
    pos = 0;
    if (dn_hw) {
#ifdef BONSAI_HAS_DNREC
        dn_hw_reset();
#endif
    } else if (dn_state) {
        memset(dn_state, 0, (size_t)n_dn*nv*hv*hk*4);
    }
    if (dn_conv)  memset(dn_conv, 0, (size_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4);
    if (dn_conv_pos) memset(dn_conv_pos, 0, (size_t)n_dn);
    hybrid_stats_reset();
}

void hybrid_stats_reset(void)
{
    dn_cyc_fetch = dn_cyc_params = dn_cyc_core = dn_cyc_store = dn_cyc_output = 0;
    dn_heads_profiled = 0;
    dn_cyc_inproj = dn_cyc_act = dn_cyc_conv = dn_cyc_prep = 0;
    dn_cyc_heads = dn_cyc_gate = dn_cyc_outproj = 0;
    hyb_cyc_embed = hyb_cyc_anorm = hyb_cyc_dn = hyb_cyc_attn = 0;
    hyb_cyc_resid = hyb_cyc_fprep = hyb_cyc_fproj = hyb_cyc_fact = 0;
    hyb_cyc_fdown = hyb_cyc_logits = 0;
    HP->at_prep = HP->at_qkv = HP->at_rope = HP->at_core = HP->at_out = 0;
    hyb_forwards_profiled = 0;
    resident_stats_reset();
}

void hybrid_stats_print(void)
{
    // The per-head recurrence line exists only with the hardware DeltaNet; the per-stage
    // and resident profiles below do not depend on it (they used to sit behind the same
    // early return, so a DNREC-less simulation printed neither).
    if (dn_hw && dn_heads_profiled) {
        uint32_t heads_per_forward = (uint32_t)n_dn*(uint32_t)nv;
        uint32_t forwards = heads_per_forward ? dn_heads_profiled/heads_per_forward : 0;
        uint64_t total = dn_cyc_fetch + dn_cyc_params + dn_cyc_core
                       + dn_cyc_store + dn_cyc_output;
        printf("dnhw: %lu heads/%lu forwards, avg cycles/head fetch %lu params %lu "
               "core %lu store %lu output %lu total %lu",
               (unsigned long)dn_heads_profiled, (unsigned long)forwards,
               (unsigned long)(dn_cyc_fetch/dn_heads_profiled),
               (unsigned long)(dn_cyc_params/dn_heads_profiled),
               (unsigned long)(dn_cyc_core/dn_heads_profiled),
               (unsigned long)(dn_cyc_store/dn_heads_profiled),
               (unsigned long)(dn_cyc_output/dn_heads_profiled),
               (unsigned long)(total/dn_heads_profiled));
        if (forwards)
            printf(", %lu cycles/forward", (unsigned long)(total/forwards));
        puts("");
    }
    if (hyb_forwards_profiled) {
        uint64_t ht = hyb_cyc_embed + hyb_cyc_anorm + hyb_cyc_dn + hyb_cyc_attn
                    + hyb_cyc_resid + hyb_cyc_fprep + hyb_cyc_fproj + hyb_cyc_fact
                    + hyb_cyc_fdown + hyb_cyc_logits;
        printf("hybprof: %lu forwards, avg cycles/fwd embed %lu anorm %lu dn %lu "
               "attn %lu resid %lu fprep %lu fproj %lu fact %lu fdown %lu "
               "logits %lu total %lu\n",
               (unsigned long)hyb_forwards_profiled,
               (unsigned long)(hyb_cyc_embed/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_anorm/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_dn/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_attn/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_resid/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_fprep/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_fproj/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_fact/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_fdown/hyb_forwards_profiled),
               (unsigned long)(hyb_cyc_logits/hyb_forwards_profiled),
               (unsigned long)(ht/hyb_forwards_profiled));
        if (hyb_cyc_attn)
            printf("attnprof: avg cycles/fwd prep %lu qkv %lu normrope %lu core %lu out %lu\n",
                   (unsigned long)(HP->at_prep/hyb_forwards_profiled),
                   (unsigned long)(HP->at_qkv/hyb_forwards_profiled),
                   (unsigned long)(HP->at_rope/hyb_forwards_profiled),
                   (unsigned long)(HP->at_core/hyb_forwards_profiled),
                   (unsigned long)(HP->at_out/hyb_forwards_profiled));
        if (HP->fwd_weight_beats) {
            // Floor: every channel streams its share of every tensor at one beat per clock.
            uint64_t fwd = ht/hyb_forwards_profiled;
            printf("floor: %lu MB weights/forward over %d channels = %lu cycles at 1 beat/"
                   "channel/clock; instrumented forward %lu cycles = %lu.%02lux the floor\n",
                   (unsigned long)(HP->fwd_weight_beats*32*(uint64_t)n_ch >> 20), n_ch,
                   (unsigned long)HP->fwd_weight_beats, (unsigned long)fwd,
                   (unsigned long)(fwd/HP->fwd_weight_beats),
                   (unsigned long)((fwd*100/HP->fwd_weight_beats) % 100));
        }
        const struct resident_state *R = resident_state();
        if (R->prepares || R->matvecs || R->fallbacks)
            printf("resprof: per fwd %lu prepares (%lu cycles) %lu matvecs (%lu cycles) "
                   "%lu fallbacks, resident %s\n",
                   (unsigned long)(R->prepares/hyb_forwards_profiled),
                   (unsigned long)(R->prepare_cycles/hyb_forwards_profiled),
                   (unsigned long)(R->matvecs/hyb_forwards_profiled),
                   (unsigned long)(R->matvec_cycles/hyb_forwards_profiled),
                   (unsigned long)(R->fallbacks/hyb_forwards_profiled),
                   resident_active() ? "on" : "off");
        resident_listprof_print(hyb_forwards_profiled);
        if (R->silu_lists)
            printf("siluprof: per fwd %lu ffn lists (%lu cycles), %lu fixups total\n",
                   (unsigned long)(R->silu_lists/hyb_forwards_profiled),
                   (unsigned long)(R->silu_cycles/hyb_forwards_profiled),
                   (unsigned long)R->silu_fixups);
        uint64_t dt = dn_cyc_inproj + dn_cyc_act + dn_cyc_conv + dn_cyc_prep
                    + dn_cyc_heads + dn_cyc_gate + dn_cyc_outproj;
        printf("dnprof: avg cycles/fwd inproj %lu act %lu conv %lu prep %lu "
               "heads %lu gate %lu outproj %lu total %lu\n",
               (unsigned long)(dn_cyc_inproj/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_act/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_conv/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_prep/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_heads/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_gate/hyb_forwards_profiled),
               (unsigned long)(dn_cyc_outproj/hyb_forwards_profiled),
               (unsigned long)(dt/hyb_forwards_profiled));
    }
}

int hybrid_pos(void) { return pos; }
int hybrid_context_capacity(void) { return ctx_cap; }

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
    float *S    = dn_hw ? NULL : dn_state + (uint64_t)s*nv*hv*hk;
    float *win  = dn_conv + (uint64_t)s*(conv_k - 1)*C;

    uint64_t t0 = perf_now64();
    int e8;
    if (H->hadamard_block) {
        // Both folded projections consume the same input basis.  Rotate and quantize once;
        // the exact Q8 bytes remain resident in SP_X for both engine projections.
        e8 = rotate_quantize_f(h, n_embd);
    } else {
        e8 = quantize_f(h, n_embd);
    }
    matvec_q(model_entry(ROLE_ATTN_QKV, l), e8, qkv);
    matvec_q(model_entry(ROLE_ATTN_ZGATE, l), e8, zb);
    e8 = quantize_f(h, n_embd);                                 // beta/alpha are not folded
    matvec_q(model_entry(ROLE_SSM_BETA, l), e8, betab);
    matvec_q(model_entry(ROLE_SSM_ALPHA, l), e8, alphab);
    dn_cyc_inproj += perf_now64() - t0;
    t0 = perf_now64();
    for (int hh = 0; hh < nv; hh++) {
        betab[hh] = sigmoidf(betab[hh]);
        gb[hh]    = softplusf(alphab[hh] + ssm_dt[l][hh])*ssm_a[l][hh];   // ssm_a = -exp(A_log)
    }
    dn_cyc_act += perf_now64() - t0;
    // Causal conv over the last conv_k inputs of every channel, then SiLU.
    t0 = perf_now64();
    const float *w = conv_w[l];   // tap i of channel c at c*conv_k + i
    if (conv_k == 4) {
        // The 27B's fixed four-tap kernel.  A circular history removes three 40 KiB
        // memcpys per layer, and spelling out the taps removes the runtime inner loop.
        // The oldest-to-newest accumulation order is unchanged, so FP32 results are exact.
        int p0 = dn_conv_pos[s], p1 = p0 == 2 ? 0 : p0 + 1;
        int p2 = p1 == 2 ? 0 : p1 + 1;
        const float *x0 = win + p0*C, *x1 = win + p1*C, *x2 = win + p2*C;
        int c = 0;
        // Four channels have no dependency.  Preserve each channel's tap order while
        // exposing independent multiply/add and LUT chains to the in-order RV32F core.
        for (; c + 3 < C; c += 4) {
            const float *cw0 = w + 4*(c + 0), *cw1 = w + 4*(c + 1);
            const float *cw2 = w + 4*(c + 2), *cw3 = w + 4*(c + 3);
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            a0 += cw0[0]*x0[c + 0]; a1 += cw1[0]*x0[c + 1];
            a2 += cw2[0]*x0[c + 2]; a3 += cw3[0]*x0[c + 3];
            a0 += cw0[1]*x1[c + 0]; a1 += cw1[1]*x1[c + 1];
            a2 += cw2[1]*x1[c + 2]; a3 += cw3[1]*x1[c + 3];
            a0 += cw0[2]*x2[c + 0]; a1 += cw1[2]*x2[c + 1];
            a2 += cw2[2]*x2[c + 2]; a3 += cw3[2]*x2[c + 3];
            a0 += cw0[3]*qkv[c + 0]; a1 += cw1[3]*qkv[c + 1];
            a2 += cw2[3]*qkv[c + 2]; a3 += cw3[3]*qkv[c + 3];
            convb[c + 0] = siluf(a0); convb[c + 1] = siluf(a1);
            convb[c + 2] = siluf(a2); convb[c + 3] = siluf(a3);
        }
        for (; c < C; c++) {
            const float *cw = w + 4*c;
            float acc = 0;
            acc += cw[0]*x0[c];
            acc += cw[1]*x1[c];
            acc += cw[2]*x2[c];
            acc += cw[3]*qkv[c];
            convb[c] = siluf(acc);
        }
        memcpy(win + p0*C, qkv, (size_t)C*4);
        dn_conv_pos[s] = p1;
    } else {
        for (int c = 0; c < C; c++) {
            float acc = 0;
            for (int i = 0; i < conv_k - 1; i++) acc += w[c*conv_k + i]*win[i*C + c];
            acc += w[c*conv_k + conv_k - 1]*qkv[c];
            convb[c] = siluf(acc);
        }
        for (int i = 0; i < conv_k - 2; i++)
            memcpy(win + i*C, win + (i + 1)*C, (size_t)C*4);
        memcpy(win + (conv_k - 2)*C, qkv, (size_t)C*4);
    }
    dn_cyc_conv += perf_now64() - t0;
    // q, k (per key head, L2-normalised), v.
    t0 = perf_now64();
    // The convolution output is dead after this layer.  Consume its Q/K/V slices in place
    // instead of copying all C floats into three work buffers (~6 MiB per 27B forward).
    float *dn_q = convb, *dn_k = convb + nq, *dn_v = convb + 2*nq;
    for (int hh = 0; hh < nk; hh++) {
        l2norm_fv(dn_q + hh*hk, hk);
        l2norm_fv(dn_k + hh*hk, hk);
    }
    dn_cyc_prep += perf_now64() - t0;
    // The recurrence per value head (llama.cpp's autoregressive kernel): M = S[hh] rows j.
    float scale = 1.0f/sqrtf((float)hk);
    int rep = nv/nk;
    for (int hi = 0; hi < nv; hi++) {
        int hh = hi, kh, gpos, load_qk = 1;
        if (dn_hw) {
            // Heads r*nk+kh share q/k for each kh. Visit those `rep` heads together so
            // q/k remain in the scratchpad, while their striped state matrices are also
            // consecutive slots in one DDR channel. `og`'s grouped order is exactly hi.
            kh = hi/rep;
            int r = hi - kh*rep;
            hh = r*nk + kh;
            gpos = hi;
            load_qk = (r == 0);
        } else {
            kh = hh % nk;
            gpos = kh*rep + hh/nk;
        }
        const float *qh = dn_q + kh*hk, *kk = dn_k + kh*hk, *vh = dn_v + hh*hv;
        float *M = dn_hw ? NULL : S + (uint64_t)hh*hv*hk;
        t0 = perf_now64();
        float decay = expf(gb[hh]);
        float *oh = ob + hh*hv;
        if (dn_hw) {
#ifdef BONSAI_HAS_DNREC
            // The next head in visiting order (hi+1 -> its hh), for the state prefetch.
            int next_head = -1;
            if (dn_pipe_use && hi + 1 < nv) {
                int nkh = (hi + 1)/rep, nr = hi + 1 - nkh*rep;
                next_head = (int)((uint32_t)s*nv) + nr*nk + nkh;
            }
            dn_hw_head((uint32_t)s*nv + (uint32_t)hh, qh, kk, vh, decay,
                       betab[hh], scale, oh, load_qk, next_head);
#endif
        } else {
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
        }
        dn_cyc_heads += perf_now64() - t0;
        // Gated RMSNorm over the head with the shared head norm, times SiLU(z); written in
        // grouped head order (key head major) - the order the folded out projection expects.
        t0 = perf_now64();
        rmsnorm_fv(delta, oh, norm_ssm[l], hv);
        float *dst = og + gpos*hv;
        const float *z = zb + hh*hv;
        silu_mul_fv(dst, delta, z, hv);
        dn_cyc_gate += perf_now64() - t0;
    }
#ifdef BONSAI_HAS_DNREC
    dn_pipe_join();                 // The last head's state must be in DDR before any other op.
#endif
    t0 = perf_now64();
    proj_f(model_entry(ROLE_SSM_OUT, l), og, out);
    dn_cyc_outproj += perf_now64() - t0;
}

#ifdef DNRES_BUILT
static uint32_t f32_bits_any(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    return v.u;
}
// The resident DeltaNet layer: the whole layer in scratchpad lists (units 1/3/4/5/8 for the
// projections, 13 conv, 11 SiLU, 12 norms, the recurrence unit per head), the CPU doing
// only the per-head scalars and the rare SiLU fixups. Bit-exact with deltanet_layer (the
// sim gate compares them checkpoint by checkpoint). Scratchpad map for this layer:
#define DNR_QKV   0x04800u        // qkv, then conv+SiLU output, q/k L2-normed in place (4C)
#define XRES_SP   0x36600u        // the resident residual stream (above the persistent map)
static uint32_t dnr_align(uint32_t x) { return (x + 31u) & ~31u; }

static int deltanet_layer_resident(int l, const float *h, float *out)
{
    int s = dn_slot[l];
    uint32_t C = (uint32_t)(2*nq + nvh);
    // z is restored at Z and parked in main RAM (zb) for the heads, whose outputs take its
    // place (HO == Z); two 64 KB head-state slots then fit above the small regions, over
    // RS_F and RS_XW (both idle during the heads); z comes back at Z2 (slot 0) for the
    // gated norm.
    uint32_t Z = dnr_align(DNR_QKV + 4*C), HO = Z, BA = dnr_align(Z + 4*(uint32_t)nvh);
    uint32_t AL = BA + dnr_align(4*(uint32_t)nv), FIX = AL + dnr_align(4*(uint32_t)nv);
    uint32_t NW = dnr_align(FIX + 4*((C > (uint32_t)nvh ? C : (uint32_t)nvh) + 31)/32);
    uint32_t SLOT0 = (NW + dnr_align(4*(uint32_t)hv) + 1023u) & ~1023u, Z2 = SLOT0;
    const uint32_t CH = 2048, W = RS_F_ADDR, X = RS_F_ADDR + 16*CH;
    uint32_t eps = f32_bits_any(rms_eps);
    if (!HP->dnres_enabled || !resident_silu_active() || !H->hadamard_block || conv_k != 4 ||
        Z2 + 4u*(uint32_t)nvh > RS_F_ADDR || C % 8 || hv % 8 || hk % 8 ||
        (seq_silu_abi_read() & 0xffff) < 1 || (seq_norm_abi_read() >> 16) != 1 ||
        (seq_conv_abi_read() >> 16) != 1)
        return -1;
#ifdef BONSAI_HAS_DNREC
    if (dn_hw && (SLOT0 + 2u*DN_HEAD_BYTES > XRES_SP || nv > 64))
        return -1;                     // the two head-state slots must fit below x
#endif
    float *win = dn_conv + (uint64_t)s*(conv_k - 1)*C;
    struct bns_seq_descriptor d;

    // A: projections. qkv and z restored in place where they will be used.
    uint64_t t0 = perf_now64();
    rl_begin();
    rl_prepare(h, (uint32_t)n_embd, 1);
    rl_matvec(model_entry(ROLE_ATTN_QKV, l), DNR_QKV, DNR_QKV);
    rl_matvec(model_entry(ROLE_ATTN_ZGATE, l), Z, Z);
    rl_prepare(h, (uint32_t)n_embd, 0);                              // beta/alpha are not folded
    rl_matvec(model_entry(ROLE_SSM_BETA, l), BA, BA);
    rl_matvec(model_entry(ROLE_SSM_ALPHA, l), AL, AL);
    rl_store(betab, dnr_align(4*(uint32_t)nv), BA);
    rl_store(alphab, dnr_align(4*(uint32_t)nv), AL);
    rl_store(zb, dnr_align(4*(uint32_t)nvh), Z);
    if (rl_run("deltanet projections"))
        return -1;                     // nothing of the layer's state has moved yet
    dn_cyc_inproj += perf_now64() - t0;
    t0 = perf_now64();
    for (int hh = 0; hh < nv; hh++) {
        betab[hh] = sigmoidf(betab[hh]);
        gb[hh]    = softplusf(alphab[hh] + ssm_dt[l][hh])*ssm_a[l][hh];
    }
    dn_cyc_act += perf_now64() - t0;

    // B: causal conv in chunks (window slots oldest to newest p0, p1, p2; the new input
    // replaces p0 after its chunk was read), then SiLU in place with a unit multiplier.
    t0 = perf_now64();
    int p0 = dn_conv_pos[s], p1 = p0 == 2 ? 0 : p0 + 1, p2 = p1 == 2 ? 0 : p1 + 1;
    float *x0 = win + (uint64_t)p0*C, *x1 = win + (uint64_t)p1*C, *x2 = win + (uint64_t)p2*C;
    rl_begin();
    for (uint32_t c0 = 0; c0 < C; c0 += CH) {
        uint32_t ch = C - c0 < CH ? C - c0 : CH;
        rl_fetch(conv_w[l] + 4*c0, 16*ch, W);
        rl_fetch(x0 + c0, 4*ch, X);
        rl_fetch(x1 + c0, 4*ch, X + 4*CH);
        rl_fetch(x2 + c0, 4*ch, X + 8*CH);
        rl_store(x0 + c0, 4*ch, DNR_QKV + 4*c0);
        bns_seq_desc_conv(&d, X, X + 4*CH, X + 8*CH, DNR_QKV + 4*c0, W, DNR_QKV + 4*c0, ch);
        rl_desc(&d);
    }
    bns_seq_desc_silu_ex(&d, DNR_QKV, DNR_QKV, DNR_QKV, FIX, C, 1, 0, 0, 0);
    rl_desc(&d);
    if (rl_run("deltanet conv"))
        goto fatal;
    dn_conv_pos[s] = (uint8_t)p1;
    if (seq_silu_fixups_read())
        resident_silu_patch(DNR_QKV, 0, DNR_QKV, FIX, C, 0, 0, 0);
    dn_cyc_conv += perf_now64() - t0;
    t0 = perf_now64();
    rl_begin();
    bns_seq_desc_norm(&d, 1, DNR_QKV, DNR_QKV, 0, (uint32_t)hk, 2u*(uint32_t)nk, eps);
    rl_desc(&d);
    if (rl_run("deltanet q/k l2"))
        goto fatal;
    dn_cyc_prep += perf_now64() - t0;

    // Heads: outputs in natural head order at HO.
    t0 = perf_now64();
    float scale = 1.0f/sqrtf((float)hk);
    int rep = nv/nk;
    if (dn_hw) {
#ifdef BONSAI_HAS_DNREC
        // Two state slots: the FETCH of head i+1 runs on the engine while the unit
        // computes head i; the STORE of head i follows. The engine runs ops in order and
        // holds one pending behind the running one, so a submit first waits for at most
        // one outstanding op, and a slot is reused only behind its own STORE.
        uint32_t slot[2] = {SLOT0, SLOT0 + DN_HEAD_BYTES};
        dn_pipe_join();
        uint32_t done0 = engine_csr_done_count_read(), sub = 0, fetched[2] = {0, 0};
#define DNR_WAIT(k) do { uint32_t sp_ = 0; \
            while ((uint32_t)(engine_csr_done_count_read() - done0) < (k)) \
                if (++sp_ > 4000000u) { puts("DNSTALL: head state transfer"); engine_wedged = 1; goto fatal; } \
        } while (0)
#define DNR_OP(kind_, head_, slot_) do { \
            if (sub > 1) DNR_WAIT(sub - 1); \
            struct engine_op o_ = {0}; uint32_t g_ = (uint32_t)(head_); \
            o_.kind = (kind_); o_.chan = (uint8_t)(g_ % (uint32_t)n_ch); o_.wbase = dn_head_addr(g_); \
            o_.wlen = DN_HEAD_BEATS; o_.dst = (slot_); o_.xsrc = (slot_); \
            engine_submit(&o_); sub++; \
        } while (0)
        int hh_of[64];
        for (int hi = 0; hi < nv && hi < 64; hi++) {
            int kh = hi/rep, r = hi - kh*rep;
            hh_of[hi] = r*nk + kh;
        }
        DNR_OP(ENGINE_KIND_FETCH, (uint32_t)s*nv + hh_of[0], slot[0]);
        fetched[0] = sub;
        for (int hi = 0; hi < nv; hi++) {
            int hh = hh_of[hi], kh = hi/rep, b = hi & 1;
            DNR_WAIT(fetched[b]);
            dnunit_state_base_write(slot[b]);
            dnunit_q_base_write(DNR_QKV + 4u*(uint32_t)(kh*hk));
            dnunit_k_base_write(DNR_QKV + 4u*(uint32_t)(nq + kh*hk));
            dnunit_v_base_write(DNR_QKV + 4u*(uint32_t)(2*nq + hh*hv));
            dnunit_out_base_write(HO + 4u*(uint32_t)(hh*hv));
            dnunit_decay_write(f32_bits(expf(gb[hh])));
            dnunit_beta_write(f32_bits(betab[hh]));
            dnunit_scale_write(f32_bits(scale));
            uint32_t udone = dnunit_done_count_read();
            dnunit_submit_write(1);
            if (hi + 1 < nv) {
                DNR_OP(ENGINE_KIND_FETCH, (uint32_t)s*nv + hh_of[hi + 1], slot[b ^ 1]);
                fetched[b ^ 1] = sub;
            }
            uint32_t spins = 0;
            while (dnunit_done_count_read() == udone)
                if (++spins > 4000000u) { puts("DNSTALL: recurrence command did not complete"); engine_wedged = 1; goto fatal; }
            DNR_OP(ENGINE_KIND_STORE, (uint32_t)s*nv + hh, slot[b]);
        }
        DNR_WAIT(sub);
#undef DNR_OP
#undef DNR_WAIT
        dnunit_state_base_write(DN_SP_STATE);
        dnunit_q_base_write(DN_SP_Q); dnunit_k_base_write(DN_SP_K);
        dnunit_v_base_write(DN_SP_V); dnunit_out_base_write(DN_SP_OUT);
        if (engine_wedged)
            goto fatal;
#endif
    } else {
        // No recurrence unit (a simulation without --with-dnrec): the CPU recurrence on
        // the normalised q/k/v, then its outputs back into the scratchpad.
        rl_begin();
        rl_store(convb, dnr_align(4*C), DNR_QKV);
        if (rl_run("deltanet qkv out"))
            goto fatal;
        const float *dn_q = convb, *dn_k = convb + nq, *dn_v = convb + 2*nq;
        for (int hh = 0; hh < nv; hh++) {
            int kh = hh % nk;
            const float *qh = dn_q + kh*hk, *kk = dn_k + kh*hk, *vh = dn_v + hh*hv;
            float *M = dn_state + ((uint64_t)s*nv + hh)*hv*hk, *oh = ob + hh*hv;
            float decay = expf(gb[hh]);
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
        }
        rl_begin();
        rl_fetch(ob, dnr_align(4*(uint32_t)nvh), HO);
        if (rl_run("deltanet heads in"))
            goto fatal;
    }
    dn_cyc_heads += perf_now64() - t0;

    // C: gated RMSNorm over the heads, x SiLU(z), into the grouped order at RS_F (og).
    t0 = perf_now64();
    rl_begin();
    rl_fetch(norm_ssm[l], dnr_align(4*(uint32_t)hv), NW);
    rl_fetch(zb, dnr_align(4*(uint32_t)nvh), Z2);
    bns_seq_desc_norm(&d, 0, HO, HO, NW, (uint32_t)hv, (uint32_t)nv, eps);
    rl_desc(&d);
    bns_seq_desc_silu_ex(&d, Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, 0, (uint32_t)hv/8, (uint32_t)nk,
                         (uint32_t)rep);
    rl_desc(&d);
    if (rl_run("deltanet gated norm"))
        goto fatal;
    if (seq_silu_fixups_read())
        resident_silu_patch(Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, (uint32_t)hv, (uint32_t)nk, (uint32_t)rep);
    // The CPU fallback of the out projection reads og from main RAM.
    rl_begin();
    rl_store(og, dnr_align(4*(uint32_t)nvh), RS_F_ADDR);
    if (rl_run("deltanet og out"))
        goto fatal;
    dn_cyc_gate += perf_now64() - t0;

    t0 = perf_now64();
    struct resident_state *R = resident_state();
    R->prep_src = og; R->prep_width = nvh; R->prep_rotate = 1;
    resident_mark_act();
    if (resident_prepare(NULL, (uint32_t)nvh, 1) == 0)
        matvec_q(model_entry(ROLE_SSM_OUT, l), E8_RESIDENT, out);
    else
        proj_f(model_entry(ROLE_SSM_OUT, l), og, out);
    dn_cyc_outproj += perf_now64() - t0;
    return 0;
fatal:
    // A list failed after the layer's window/state started to move: the forward cannot be
    // recomputed on the CPU. The resident path is off (run_list); report and stop.
    puts("hybrid: resident DeltaNet layer failed mid-layer - this token is invalid");
    return -2;
}
#endif

#ifdef DNRES_BUILT
// The resident residual stream: x lives in the scratchpad at XRES_SP for the whole forward;
// the norms (unit 12) read it and leave hb in RS_F (and main RAM, for the mixers and the
// traces); the residual adds (unit 13 op 1) add the layer outputs into it. xb (main RAM)
// is only brought up to date for traces and for the LM head.
static int xres_ok(void)
{
    return HP->xres_enabled && resident_active() && n_embd % 8 == 0 && n_embd <= 8192 &&
           XRES_SP + 4u*(uint32_t)n_embd <= 0x3C000u &&
           (seq_norm_abi_read() >> 16) == 1 && (seq_conv_abi_read() >> 16) == 1;
}

static int xres_list(const float *y, const float *w, int load, int sync)
{
    struct bns_seq_descriptor d;
    uint32_t n = (uint32_t)n_embd, bytes = 4*n;
    uint32_t wsp = RS_Y_ADDR + ((bytes + 31u) & ~31u);
    rl_begin();
    if (load)
        rl_fetch(xb, bytes, XRES_SP);
    if (y) {
        rl_fetch(y, bytes, RS_Y_ADDR);
        bns_seq_desc_add(&d, XRES_SP, RS_Y_ADDR, XRES_SP, n);
        rl_desc(&d);
    }
    if (sync)
        rl_store(xb, bytes, XRES_SP);
    if (w) {
        rl_fetch(w, bytes, wsp);
        bns_seq_desc_norm(&d, 0, XRES_SP, RS_F_ADDR, wsp, n, 1, f32_bits_any(rms_eps));
        rl_desc(&d);
        rl_store(hb, bytes, RS_F_ADDR);
    }
    return rl_run("residual stream");
}
#endif

static void attention_layer(int l, const float *h, float *out)
{
    int a = attn_slot[l];
    float *K = kc + (uint64_t)a*ctx_cap*kv_dim, *V = vc + (uint64_t)a*ctx_cap*kv_dim;
    int gated = H->attn_gated;
    int e8;
    uint64_t t0 = perf_now64(), t1;
    int normed = 0;
#ifdef DNRES_BUILT
    // Resident: Q/K/V in one list with the per-head q/k RMSNorms on unit 12 (the q heads
    // strided between their gates), everything stored for the CPU's RoPE and scores.
    uint32_t qrows = (uint32_t)(gated ? 2*q_dim : q_dim);
    uint32_t QS = RS_Y_ADDR, KS = QS + 4*qrows, VS = KS + 4*(uint32_t)kv_dim;
    uint32_t NQ = VS + 4*(uint32_t)kv_dim, NK = NQ + 4*(uint32_t)head_dim;
    if (H->hadamard_block && resident_active() && HP->dnres_enabled && head_dim % 8 == 0 &&
        NK + 4u*(uint32_t)head_dim <= RS_F_ADDR && (seq_norm_abi_read() & 0xffff) >= 1) {
        struct bns_seq_descriptor d;
        uint32_t eps = f32_bits_any(rms_eps), hb8 = (uint32_t)head_dim/8;
        rl_begin();
        rl_prepare(h, (uint32_t)n_embd, 1);
        rl_matvec(model_entry(ROLE_ATTN_Q, l), QS, QS);
        rl_matvec(model_entry(ROLE_ATTN_K, l), KS, KS);
        rl_matvec(model_entry(ROLE_ATTN_V, l), VS, VS);
        rl_fetch(norm_q[l], 4*(uint32_t)head_dim, NQ);
        rl_fetch(norm_k[l], 4*(uint32_t)head_dim, NK);
        bns_seq_desc_norm_strided(&d, 0, QS, QS, NQ, (uint32_t)head_dim, (uint32_t)n_head, eps,
                                  gated ? 2*hb8 : hb8, gated ? 2*hb8 : hb8);
        rl_desc(&d);
        bns_seq_desc_norm(&d, 0, KS, KS, NK, (uint32_t)head_dim, (uint32_t)n_kv_head, eps);
        rl_desc(&d);
        rl_store(gated ? qfull : qb, 4*qrows, QS);
        rl_store(kb, 4*(uint32_t)kv_dim, KS);
        rl_store(vb, 4*(uint32_t)kv_dim, VS);
        normed = rl_run("attention q/k/v") == 0;
    }
#endif
    if (normed) {
        t1 = perf_now64(); HP->at_qkv += t1 - t0; t0 = t1;
    } else {
    if (H->hadamard_block) {
        // Q/K/V share the input and folded basis: rotate once, not once per projection.
        // This removes another 32 full-width rotations per 27B forward.
        e8 = rotate_quantize_f(h, n_embd);
    } else {
        e8 = quantize_f(h, n_embd);
    }
    t1 = perf_now64(); HP->at_prep += t1 - t0; t0 = t1;
    matvec_q(model_entry(ROLE_ATTN_Q, l), e8, gated ? qfull : qb);
    matvec_q(model_entry(ROLE_ATTN_K, l), e8, kb);
    matvec_q(model_entry(ROLE_ATTN_V, l), e8, vb);
    t1 = perf_now64(); HP->at_qkv += t1 - t0; t0 = t1;
    }
    if (gated)   // [q | gate] per head
        for (int hh = 0; hh < n_head; hh++)
            memcpy(qb + hh*head_dim, qfull + hh*2*head_dim, (size_t)head_dim*4);
    rope_prepare(pos);
    for (int hh = 0; hh < n_head; hh++) {
        if (!normed)
            rmsnorm_fv(qb + hh*head_dim, qb + hh*head_dim, norm_q[l], head_dim);
        rope_apply_partial(qb + hh*head_dim);
    }
    for (int hh = 0; hh < n_kv_head; hh++) {
        if (!normed)
            rmsnorm_fv(kb + hh*head_dim, kb + hh*head_dim, norm_k[l], head_dim);
        rope_apply_partial(kb + hh*head_dim);
    }
    t1 = perf_now64(); HP->at_rope += t1 - t0; t0 = t1;
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
            int i = 0;
            // Keep the reference's sequential FP32 sum, with less loop control.
            for (; i + 3 < head_dim; i += 4) {
                sc += qh[i + 0]*kt[i + 0]; sc += qh[i + 1]*kt[i + 1];
                sc += qh[i + 2]*kt[i + 2]; sc += qh[i + 3]*kt[i + 3];
            }
            for (; i < head_dim; i++) sc += qh[i]*kt[i];
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
            int i = 0;
            for (; i + 3 < head_dim; i += 4) {
                oh[i + 0] += p*vt[i + 0]; oh[i + 1] += p*vt[i + 1];
                oh[i + 2] += p*vt[i + 2]; oh[i + 3] += p*vt[i + 3];
            }
            for (; i < head_dim; i++) oh[i] += p*vt[i];
        }
        if (gated) {
            const float *g = qfull + hh*2*head_dim + head_dim;
            int i = 0;
            for (; i + 3 < head_dim; i += 4) {
                oh[i + 0] *= sigmoidf(g[i + 0]); oh[i + 1] *= sigmoidf(g[i + 1]);
                oh[i + 2] *= sigmoidf(g[i + 2]); oh[i + 3] *= sigmoidf(g[i + 3]);
            }
            for (; i < head_dim; i++) oh[i] *= sigmoidf(g[i]);
        }
    }
    t1 = perf_now64(); HP->at_core += t1 - t0; t0 = t1;
    proj_f(model_entry(ROLE_ATTN_OUT, l), attnb, out);
    HP->at_out += perf_now64() - t0;
}

int hybrid_forward(int token, int want_logits)
{
    if (pos >= ctx_cap) { printf("hybrid: context cap %d reached\n", ctx_cap); return -1; }
    uint64_t t0 = perf_now64();
    if (infer_embed_f(token, xb) < 0) return -1;
    hyb_cyc_embed += perf_now64() - t0;
    infer_dbg_f32("emb", -1, xb, n_embd);
    int xres = 0;
#ifdef DNRES_BUILT
    xres = xres_ok();
#endif
    int traced = infer_dbg_armed();
    for (int l = 0; l < n_layer; l++) {
        t0 = perf_now64();
        if (xres) {
#ifdef DNRES_BUILT
            if (xres_list(NULL, norm_attn[l], l == 0, 0)) return -1;
#endif
        } else {
            rmsnorm_fv(hb, xb, norm_attn[l], n_embd);
        }
        hyb_cyc_anorm += perf_now64() - t0;
        infer_dbg_f32("attn_norm", l, hb, n_embd);
        t0 = perf_now64();
        if (dn_slot[l] >= 0) {
            int rc = -1;
#ifdef DNRES_BUILT
            rc = deltanet_layer_resident(l, hb, tmp_h);
            if (rc == -2) return -1;
#endif
            if (rc)
                deltanet_layer(l, hb, tmp_h);
            hyb_cyc_dn += perf_now64() - t0;
        } else {
            attention_layer(l, hb, tmp_h);
            hyb_cyc_attn += perf_now64() - t0;
        }
        infer_dbg_f32(dn_slot[l] >= 0 ? "dn_out" : "attn_out", l, tmp_h, n_embd);
        t0 = perf_now64();
        if (xres) {
#ifdef DNRES_BUILT
            // Residual add, the FFN norm, and xb for the trace when traced.
            if (xres_list(tmp_h, norm_ffn[l], 0, traced)) return -1;
#endif
            hyb_cyc_resid += perf_now64() - t0;
            infer_dbg_f32("attn_res", l, xb, n_embd);
            t0 = perf_now64();
        } else {
            add_fv(xb, tmp_h, n_embd);
            hyb_cyc_resid += perf_now64() - t0;
            infer_dbg_f32("attn_res", l, xb, n_embd);
            t0 = perf_now64();
            rmsnorm_fv(hb, xb, norm_ffn[l], n_embd);
        }
        infer_dbg_f32("ffn_norm", l, hb, n_embd);
        int e8;
        if (H->hadamard_block)
            e8 = rotate_quantize_f(hb, n_embd);
        else
            e8 = quantize_f(hb, n_embd);
        hyb_cyc_fprep += perf_now64() - t0;
        t0 = perf_now64();
        const struct bons_entry *down = model_entry(ROLE_FFN_DOWN, l);
        // Resident FFN: gate, up and act = up*silu(gate) in one scratchpad list, the act
        // then prepared for the down projection where it already is (no FETCH). The gate
        // and up projections reach main RAM only when traced.
        int fused = 0;
        if (e8 == E8_RESIDENT && H->hadamard_block && resident_silu_active() &&
            infer_matvec_cols(down) == (uint32_t)ffn_dim) {
            int traced = infer_dbg_armed();
            fused = resident_ffn_act(model_entry(ROLE_FFN_GATE, l), model_entry(ROLE_FFN_UP, l),
                                     actb, traced ? gateb : NULL, traced ? upb : NULL) == 0;
        }
        if (fused) {
            hyb_cyc_fproj += perf_now64() - t0;
            infer_dbg_f32("ffn_gate", l, gateb, ffn_dim);
            infer_dbg_f32("ffn_up", l, upb, ffn_dim);
            infer_dbg_f32("ffn_act", l, actb, ffn_dim);
            t0 = perf_now64();
            struct resident_state *R = resident_state();
            R->prep_src = actb; R->prep_width = ffn_dim; R->prep_rotate = 1;
            if (resident_prepare(NULL, (uint32_t)ffn_dim, 1) == 0)
                matvec_q(down, E8_RESIDENT, tmp_h);
            else
                proj_f(down, actb, tmp_h);
            hyb_cyc_fdown += perf_now64() - t0;
        } else {
            matvec_q(model_entry(ROLE_FFN_GATE, l), e8, gateb);
            matvec_q(model_entry(ROLE_FFN_UP, l), e8, upb);
            hyb_cyc_fproj += perf_now64() - t0;
            infer_dbg_f32("ffn_gate", l, gateb, ffn_dim);
            infer_dbg_f32("ffn_up", l, upb, ffn_dim);
            t0 = perf_now64();
            silu_mul_fv(gateb, upb, gateb, ffn_dim);
            hyb_cyc_fact += perf_now64() - t0;
            infer_dbg_f32("ffn_act", l, gateb, ffn_dim);
            t0 = perf_now64();
            proj_f(down, gateb, tmp_h);
            hyb_cyc_fdown += perf_now64() - t0;
        }
        infer_dbg_f32("ffn_down", l, tmp_h, n_embd);
        t0 = perf_now64();
        if (xres) {
#ifdef DNRES_BUILT
            // The last layer brings xb up to date for the LM head.
            if (xres_list(tmp_h, NULL, 0, traced || l == n_layer - 1)) return -1;
#endif
        } else {
            add_fv(xb, tmp_h, n_embd);
        }
        hyb_cyc_resid += perf_now64() - t0;
        infer_dbg_f32("ffn_res", l, xb, n_embd);
    }
    pos++;
    hyb_forwards_profiled++;
    if (!want_logits) return -1;
    t0 = perf_now64();
    const struct bons_entry *on = model_entry_kind(ROLE_OUT_NORM, 0, KIND_F32);
    static float *out_norm; if (!out_norm) out_norm = fetch_f32(on);
    rmsnorm_fv(hb, xb, out_norm, n_embd);
    infer_dbg_f32("out_norm", -1, hb, n_embd);
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head) head = model_entry(ROLE_TOK_EMB, 0);
    proj_f(head, hb, logits);
    infer_dbg_f32("logits", -1, logits, H->vocab);
    int best = 0; float bv = logits[0];
    for (uint32_t i = 1; i < H->vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    hyb_cyc_logits += perf_now64() - t0;
    return best;
}

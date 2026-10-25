/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium hybrid models forward (Bonsai 2: Gated DeltaNet + gated attention + rotated basis)
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

/*
 * The hybrid models (Bonsai 2 = Qwen3.5: Gated DeltaNet layers, gated attention with partial
 * RoPE, PrismML's rotated weight basis). The CPU path does float math over main_ram with the
 * engine for every matvec (the projections and the rotation blocks): the golden's semantics
 * (model/golden.py) transliterated, gated on the tiny hybrid model in the Verilator
 * sim, and the reference the hardware paths are checked against. Where the bitstream has the
 * units, the same layers run as scratchpad descriptor lists (resident.c), the DeltaNet stream
 * (engine op 5) and the attention stream (engine op 6) - bit-exact with the CPU path.
 *
 * What a hybrid image adds (BONS v3, layout.py): per layer either the attention tensors
 * (attn_q carries [q | gate] per head when the header says attn_gated) or the DeltaNet set
 * (attn_qkv = [q | k | v], attn_gate = z, ssm_beta/ssm_alpha as Q8 head projections, the
 * conv taps, A = -exp(A_log), the dt bias, the head norm, ssm_out); post_attention_norm as the
 * FFN norm; the ROLE_HADAMARD block tensors R diag(s) per input width; the emb16 rows already
 * restored to the plain basis (HYB_EMB16_RESTORED - the ternary embedding table stays
 * rotated and cannot be decoded row by row).
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include <system.h>

#include "bons.h"
#include "model.h"
#include "engine.h"
#include "infer.h"
#include "hybrid.h"
#include "trixium_sequence.h"
#include "perf.h"
#include "fastmath.h"
#include "resident.h"
#include "static_memory.h"

/* Configuration */
/*---------------*/

// The resident DeltaNet layer needs units 11 (SiLU), 12 (norm) and 13 (conv).
#if defined(TRIXIUM_HAS_NORM) && defined(TRIXIUM_HAS_CONV) && defined(TRIXIUM_HAS_SILU)
#define DNRES_BUILT 1
#endif

// The DeltaNet state in engine DDR, updated by the stream op (TRIXIUM_HAS_DNSTREAM): the
// heads run only there (`dnstream off` then stops with DNSTALL).
#if defined(TRIXIUM_HAS_DNSTREAM)
#define DN_STATE_HW 1
#endif

#if defined(DNRES_BUILT) && defined(CSR_TRIXIUM_ENGINE_CSR_NR_ADDR)
#define P2_BUILT 1
// The batched prefill's in-projections (doc/plan-multirow.md): the mixer of row 0 (P2_LEAD)
// also projects row 1 in the same weight pass - row 1's attention norm from its residual slot,
// its rotated prepare into Q8 slot 1 - and parks row 1's restored outputs in main RAM; the
// mixer of row 1 (P2_FOLLOW, only when the lead list ran: p2_parked) fetches them back where
// its own projections would have restored them. The outputs are the restores of the same int32
// sums with the same exponents, so the follower's list sees the single-token values.
#define P2_LEAD   1
#define P2_FOLLOW 2
static int p2_mode;                    // what pair_seg asks of the mixer it calls (0: nothing)
static int p2_parked;                  // the lead list ran: row 1's outputs are in p2_park
static float *p2_park;
static uint32_t p2_park_bytes;
#endif

/* Variables */
/*-----------*/

static const struct bons_header *H;
static int n_ch, n_embd, n_head, n_kv_head, head_dim, ffn_dim, n_layer, q_dim, kv_dim, n_rep;
static int nk, hk, nv, hv, nq, nvh, conv_k, rope_n;
static int n_dn, n_attn, ctx_cap, pos;
static float rms_eps;
static float *out_norm;              // fetched on the first forward with logits

// Float work buffers (main_ram).
static float *xb, *hb, *rot, *qkv, *zb, *betab, *alphab, *gb, *convb, *qb, *kb, *vb, *ob, *og,
             *gateb, *upb, *actb, *attnb, *logits, *scores, *delta, *qfull, *tmp_h;
// Per-layer tables: index by layer (NULL where the layer has no such tensor).
static float **norm_attn, **norm_ffn, **norm_q, **norm_k, **norm_ssm, **ssm_a, **ssm_dt, **conv_w;
static int   *dn_slot, *attn_slot;   // layer -> DeltaNet / attention index, -1 otherwise.
static float *dn_state;              // [n_dn][nv][hv][hk]  (row j = value index, as llama.cpp)
static float *dn_conv;               // [dn_conv_w][n_dn][C], slot-major (dn_conv_slot)
static uint8_t *dn_conv_pos;         // Oldest causal-convolution plane per DeltaNet layer.
static int dn_conv_w;                // Window ring slots: conv_k - 1.
static float *kc, *vc;               // [n_attn][ctx_cap][kv_dim]
// Rows [0, kc_rows[a]) of attention slot a are in kc/vc. With the attention stream the rows
// live in engine DDR and main RAM's copy (both sides miss the cache) is skipped; the CPU loop
// restores the missing rows before it runs (attn_kv_main).
static uint16_t *kc_rows;
static float *rope_c, *rope_s, *rope_freq;
static int    rope_pos_cached;

// The DeltaNet state in engine DDR (the stream op's fixed geometry: the 27B's 128x128
// matrices); other hybrid test models keep it in main RAM for the scalar CPU recurrence.
static int      dn_hw;
#ifdef TRIXIUM_HAS_ATTNSTREAM
static uint32_t at_hbm_base, at_slots, at_rowbeats;
static int at_geom_ok;

// The end (local address, a page below) of a channel's weight window, the smallest of a
// controller's (the last, shortened by the bank skew): TRIXIUM_CHAN_WINDOW when the target
// exports it (the YPCB's windows start at base + j*stride, not at stride multiples), else
// the stride boundary above the weight base (as ddr.c's DDR_TEST_WINDOW).
static uint64_t chan_limit(void)
{
#ifdef TRIXIUM_CHAN_WINDOW
    return (uint64_t)TRIXIUM_CHAN_WEIGHT_BASE + TRIXIUM_CHAN_WINDOW - 0x1000u;
#else
    uint32_t skew_count = TRIXIUM_CHAN_MOD ? TRIXIUM_CHAN_MOD - 1u : 0u;
    return (uint64_t)TRIXIUM_CHAN_WEIGHT_BASE + TRIXIUM_CHAN_STRIDE
         - (TRIXIUM_CHAN_WEIGHT_BASE % TRIXIUM_CHAN_STRIDE) - 0x1000u
         - (uint64_t)skew_count*TRIXIUM_CHAN_SKEW;
#endif
}
static int at_sg;                      // kv heads per op-6 command (n_kv_head: one command)
static float *at_buf;
#endif

static uint32_t dn_hbm_base;
// Two state sets per channel (TRIXIUM_DNSTREAM_WOFF): a forward reads set dn_par and writes
// the other, so a forward that saw an uncorrectable ECC read can run again from intact state
// (hybrid_ckpt_save). dn_set_bytes: one set's span per channel; 0 = one set, in place.
static uint32_t dn_set_bytes;
static int dn_par;
static int dn_hw_written;              // the span was cleared once since the boot
// DeltaNet slots whose state is still the reset's zero (bit s): with the stream layout the
// layer's op 5 takes it as +0.0 (TRIXIUM_DNSTREAM_ZERO) instead of hybrid_reset clearing the
// 144 MiB in engine DDR at every generation's first token.
static uint64_t dn_state_zero;
int hybrid_softmax_div;                  // Console `softmaxdiv on`: the per-position divide (A/B).
// Past the KV window (ctx_cap positions) the attention layers keep the first KV_SINKS positions
// and a ring of the most recent ones: the conversation goes on without a restart or a re-prefill
// (the DeltaNet layers' recurrent state carries the whole history anyway). K rows are stored
// rotated at their absolute position, so the scores stay exact relative attention over the
// slots kept. Console `kvring off`: the old hard stop at ctx_cap.
int hybrid_kv_ring = 1;
#define KV_SINKS 4
// Positions kept for good past the window: at least the KV_SINKS, a chat's whole first turn (its
// system prompt and first message, hybrid_set_pin before the generation's reset) up to a
// quarter of the window: what the conversation is about stays attended to.
static int kv_pin = KV_SINKS, kv_pin_next = KV_SINKS;

void hybrid_set_pin(int n) { kv_pin_next = n; }

static int kv_ring_on(void)
{
    return hybrid_kv_ring && ctx_cap > 2*kv_pin;
}

// The KV slot of position p (p < ctx_cap: p itself).
static uint32_t kv_slot(int p)
{
    if (p < ctx_cap)
        return (uint32_t)p;
    return (uint32_t)(kv_pin + (p - kv_pin) % (ctx_cap - kv_pin));
}

#ifdef P2_BUILT
// Two conversations decoded together (hybrid_forward_dual): everything one sequence carries
// from token to token, swapped in and out of the globals the forward reads (seq_swap), so the
// single-token code runs unchanged on either. Sequence 0 is the one every other path uses; the
// second set (engine-DDR DeltaNet state and attention K/V spans, the conv window, main RAM's K/V
// copy) is allocated by hybrid_dual_setup on the first `gen2`.
struct hyb_seq {
    int pos, dn_par, kv_pin, kv_pin_next, dn_hw_written;
    uint64_t dn_state_zero;
    uint32_t dn_hbm_base, at_hbm_base;
    float *dn_state, *dn_conv, *kc, *vc;
    uint8_t *dn_conv_pos;
    uint16_t *kc_rows;
};
static struct hyb_seq hseq[2];
static int hseq_cur;                   // the sequence whose state the globals hold
static int hseq_ready;                 // the second set: 0 not tried, 1 allocated, -1 refused
static int p2_dual;                    // pair_row swaps sequences (the dual decode), not positions
static void seq_swap(int s);
// The list image cache's salt of row r of the current sequence: the lists of sequence 1 name its
// own engine-DDR state (the DeltaNet core list's wbase), the second row the second residual slot.
static uint32_t seq_salt(int r)
{
    uint32_t k = (uint32_t)hseq_cur << 1 | (uint32_t)r;
    return k ? 0x9A1B0000u | k : 0u;
}
#endif
int hybrid_softmax_unit = 1;             // Console `softmaxunit off`: the CPU softmax instead of unit 14.
int hybrid_dn_list = 1;                  // Console `dnlist off`: the DeltaNet heads op through the CSRs
int hybrid_dn_zbeside = 0;               // Console `dnzbeside on`: z beside the conv's SiLU, not in the projection list (A/B pending on the card)
                                         // (its ENGSTALL diagnostics) instead of the merged core list.
int hybrid_fuse_out = 1;                 // Console `fuseout off`: the output projection, the residual add
                                         // and the next norm as separate lists (the A/B of out_resid_*).
int hybrid_lm_store = 0;                 // Console `lmstore on`: the LM head chunk lists store the logits
                                         // to main RAM even when the hardware argmax decides (the
                                         // untraced default stores nothing: the tracker reads RS_F).
int hybrid_hb_store = 0;                 // Console `hbstore on`: the residual list stores hb to main RAM
                                         // every call (the untraced default leaves it in RS_F only).
static int hb_stale;                     // hb's main-RAM copy lags the scratchpad (see hb_materialize)
static const float *hb_w;                // ... the norm weight that produced it
#ifdef TRIXIUM_SIM
static int hybrid_inits, power_on_reset; // Simulation: hybrid_init's first reset (below).
#endif

static inline float *dn_conv_slot(int s, int p, int C)
{
    return dn_conv + ((uint64_t)p*(uint64_t)n_dn + (uint64_t)s)*(uint64_t)C;
}

/* Profiling */
/*-----------*/

// Stage cycle counters: rdcycle is cheap enough to bracket every stage without perturbing
// the datapath; per-forward averages keep the 32-bit minimal printf sufficient even for a
// long 27B benchmark.
static uint64_t dn_cyc_inproj, dn_cyc_act, dn_cyc_conv, dn_cyc_prep;
static uint64_t dn_cyc_heads, dn_cyc_gate, dn_cyc_outproj;
static uint64_t hyb_cyc_embed, hyb_cyc_anorm, hyb_cyc_dn, hyb_cyc_attn;
static uint64_t hyb_cyc_resid, hyb_cyc_fprep, hyb_cyc_fproj, hyb_cyc_fact;
static uint64_t hyb_cyc_fdown, hyb_cyc_logits;
// Profile counters and console switches, in the main_ram static window: the 8 KiB SRAM's
// .bss has no room left above its 4 KiB stack floor, so new counters go here.
struct hyb_prof {
    // Attention sub-stages (sum = hyb_cyc_attn): prepare, Q/K/V projections, per-head norms +
    // RoPE, scores/softmax/values/gate (incl. the KV write), output projection.
    uint64_t at_prep, at_qkv, at_rope, at_core, at_out;
    uint64_t at_sub[6];              // inside at_core: kv copy, list 1 build+run, softmax,
                                     // list 2 build+run, fixups, the rest
    // Weight beats one forward streams per channel (every matvec tensor once, K-segments
    // included, the LM head once): the bandwidth floor is this many cycles at one 32-byte
    // beat per channel per clock.
    uint64_t fwd_weight_beats;
    uint8_t dnres_enabled;           // console `dnres on|off`: the resident DeltaNet layer
    uint8_t xres_enabled;            // the resident residual stream (set at init)
    uint8_t dnstream_enabled;        // console `dnstream on|off` (taken at the next reset)
    uint8_t dnstream_active;         // the layout in engine DDR since the last reset
    uint8_t attnstream_enabled;      // console `attnstream on|off` (taken at the next reset)
    uint8_t attnstream_active;       // K/V rows also appended to engine DDR since the reset
    uint8_t hwargmax;                // console `hwargmax on|off`: the LM head list's argmax
    uint32_t lm_chunk;               // console `lmchunk <rows>`: cap the LM head chunks (0 auto)
    uint8_t logits_clean;            // the CPU holds no dirty line of `logits` (lm_head_argmax)
    uint64_t fd_prep, fd_mv;         // inside hyb_cyc_fdown: the down prepare, the down matvec
    uint8_t keep_out;                // the mixer/FFN outputs stay in RS_F for the residual add
    uint8_t xres_on;                 // the resident residual stream runs this forward (out_resid_*)
    uint8_t xres_row;                // the batched prefill's row whose residual slot XRES_SP names
    uint8_t prefill2;                // console `prefill2 on|off`: two prompt tokens per weight pass
    uint8_t pair_geom;               // the model fits the batched FFN lists: 0 unknown, 1 yes, 2 no
    uint8_t pair_inproj;             // `prefill2 on` (1) also batches the mixers' in-projections; `prefill2 ffn` (0)
    uint8_t lmhead2;                 // `gen2 lmhead on|off`: the two conversations' LM heads in one weight pass
};
#define HP ((struct hyb_prof *)TRIXIUM_HYBPROF_ADDR)
_Static_assert(sizeof(struct hyb_prof) <= 256, "hyb_prof window is 256 bytes");
static uint32_t hyb_forwards_profiled;
// Live counters for a host monitor (software/host/dashboard.py reads them over Etherbone while
// the board runs): cumulative since hybrid_init, not cleared by `stats`, published at the end of
// each forward's layers (the LM head of a forward lands in the next publication). The block is
// the first thing in sram (linker.ld's .live): the shared bus reaches sram on every board, while
// main_ram is the CPU's own memory port on some (the Ti375). seq is odd while the block changes;
// a reader retries when it differs across its read. The CPU's L1 is write-back: each
// publication cleans the block's lines.
#define HYB_LIVE_MAGIC   0x4556494cu     // "LIVE"
#define HYB_LIVE_VERSION 1
#define HYB_LIVE_STAGES  10
struct hyb_live {
    uint32_t magic, version, seq, forwards;
    uint64_t now;                        // perf_now64() at the publication
    uint64_t cyc[HYB_LIVE_STAGES];       // embed anorm dn attn resid fprep fproj fact fdown logits
    uint64_t weight_beats;               // weight beats one forward streams (HP->fwd_weight_beats)
};
static uint64_t hyb_live_base[HYB_LIVE_STAGES];   // the totals before the last hybrid_stats_reset
static struct hyb_live hyb_live __attribute__((section(".live"), used));
#define HL ((volatile struct hyb_live *)&hyb_live)
_Static_assert(sizeof(struct hyb_live) <= 128, "linker.ld's .live is 128 bytes");

static void hyb_live_publish(void)
{
    const uint64_t c[HYB_LIVE_STAGES] = {hyb_cyc_embed, hyb_cyc_anorm, hyb_cyc_dn, hyb_cyc_attn,
        hyb_cyc_resid, hyb_cyc_fprep, hyb_cyc_fproj, hyb_cyc_fact, hyb_cyc_fdown, hyb_cyc_logits};
    HL->seq++;
    for (int i = 0; i < HYB_LIVE_STAGES; i++)
        HL->cyc[i] = hyb_live_base[i] + c[i];
    HL->forwards++;
    HL->weight_beats = HP->fwd_weight_beats;
    HL->now = perf_now64();
    HL->seq++;
#ifdef HAS_CLEAN_CPU_DCACHE_RANGE
    clean_cpu_dcache_range((void *)HL, sizeof(struct hyb_live));
#endif
}

/* Activation Tables */
/*-------------------*/

// The 27B evaluates SiLU about 1.9 million times per forward.  Even the range-reduced
// polynomial in fastmath.h is limited by its dependent FMA chain and FP divide.  Linear
// interpolation at 1/128 spacing has <4.4e-6 absolute error over [-16, 16], while this
// 16 KiB table and the activations' much smaller hot interval fit the 32 KiB D-cache.
static float *silu_lut;
static float *sigmoid_lut;

/* DeltaNet State In Engine DDR */
/*------------------------------*/

#ifdef DN_STATE_HW
#define DN_SP_STATE 0x00000u
#define DN_HW_ROWS  128u
#define DN_HW_COLS  128u
#define DN_HEAD_BYTES (DN_HW_ROWS*DN_HW_COLS*4u)
#define DN_HEAD_BEATS (DN_HEAD_BYTES/32u)

// DeltaNet stream (engine op 5, TRIXIUM_HAS_DNSTREAM): every head of a layer in one op, the
// state striped over the channels - channel c holds rows c, c+n_ch, ... of every head, in
// the unit's column-group-major order (gateware/units/deltanet_stream.py), heads in processing
// order, n_dn*nv head matrices per layer set; the layout is chosen at hybrid_reset
// (HP->dnstream_active) and only the resident layer runs it.
#define DN_STREAM_BEATS (DN_HEAD_BEATS/(uint32_t)n_ch)   // per channel per head

static void dn_hw_reset(void)
{
    // Fill one matrix once, then let the engine's multi-store repeat it across consecutive
    // per-channel slots (up to 31 packets per command). This clears the full 144 MiB 27B
    // state without walking it over the CPU/main-RAM path.
    volatile uint32_t *z = (volatile uint32_t *)(SCRATCHPAD_BASE + DN_SP_STATE);
    for (uint32_t i = 0; i < DN_HEAD_BYTES/4; i++) z[i] = 0;
    uint32_t total = (uint32_t)n_dn*(uint32_t)nv;
    for (uint32_t c = 0; c < (uint32_t)n_ch; c++) {
        // Every channel's full span: the stream layout uses total*DN_STREAM_BEATS beats of
        // each, i.e. at most ceil(total/n_ch) head-sized slots (per set).
        uint32_t count = (total + (uint32_t)n_ch - 1u)/(uint32_t)n_ch*(dn_set_bytes ? 2u : 1u);
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

/* Controls */
/*----------*/

void hybrid_dnres_enable(int on) { HP->dnres_enabled = on ? 1 : 0; }
void hybrid_dnstream_enable(int on)
{
#ifdef TRIXIUM_HAS_DNSTREAM
    HP->dnstream_enabled = on ? 1 : 0;
#else
    (void)on;
#endif
}
int hybrid_dnstream_enabled(void) { return HP->dnstream_enabled; }
void hybrid_attnstream_enable(int on)
{
#ifdef TRIXIUM_HAS_ATTNSTREAM
    HP->attnstream_enabled = on ? 1 : 0;
#else
    (void)on;
#endif
}
int hybrid_attnstream_enabled(void) { return HP->attnstream_enabled; }
void hybrid_hwargmax_enable(int on) { HP->hwargmax = on ? 1 : 0; }
void hybrid_lm_chunk(uint32_t rows) { HP->lm_chunk = rows; }
int hybrid_hwargmax_enabled(void) { return HP->hwargmax; }
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

/* Helpers */
/*---------*/

static float sigmoidf(float x) { return trixium_sigmoid_lookup(x, sigmoid_lut); }
static float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static void silu_lut_init(void)
{
    for (int i = 0; i <= TRIXIUM_SILU_LUT_N; i++) {
        float x = -TRIXIUM_SILU_LUT_LIMIT + (float)i*(1.0f/TRIXIUM_SILU_LUT_SCALE);
        silu_lut[i] = x/(1.0f + expf(-x));
    }
}

static inline float siluf(float x)
{
    return trixium_silu_lookup(x, silu_lut);
}

// The canonical FP32 sum of squares (doc/units.md (Canonical reductions)): 64 partial sums, element
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

#define XRES_SP0  0x36600u        // the resident residual stream (above the persistent map)
#define XRES_SP1  (XRES_SP0 + 0x2800u) // the batched prefill's second row's (n_embd <= 2560)
#define XRES_SP   (HP->xres_row ? XRES_SP1 : XRES_SP0)
// hb's main-RAM copy for the CPU paths. Untraced, the residual list leaves hb in RS_F and stores
// nothing (a STORE descriptor, 8 KB flushed and 8 KB invalidated at every list, 49 times a
// token); the resident consumers take it from RS_F (rl_rsf_is). A CPU reader of hb (a declined
// or failed list's fallback, the LM head's CPU loop) calls this first: one list stores x from
// XRES_SP (xb) and hb from RS_F while it is still there, else hb is renormed on the CPU (the
// same canonical sum as unit 12). If the list itself fails, the words are read from the
// scratchpad directly (slow, correct).
static int hb_materialize(void)
{
    if (!hb_stale)
        return 0;
    uint32_t n = (uint32_t)n_embd, bytes = 4u*n;
    int from_rsf = rl_rsf_is(hb);
    rl_begin();
    rl_store(xb, bytes, XRES_SP);
    if (from_rsf)
        rl_store(hb, bytes, RS_F_ADDR);
    if (rl_run("hb restore")) {
        const volatile uint32_t *x = (const volatile uint32_t *)(SCRATCHPAD_BASE + XRES_SP);
        const volatile uint32_t *f = (const volatile uint32_t *)(SCRATCHPAD_BASE + RS_F_ADDR);
        for (uint32_t i = 0; i < n; i++) ((uint32_t *)xb)[i] = x[i];
        if (from_rsf)
            for (uint32_t i = 0; i < n; i++) ((uint32_t *)hb)[i] = f[i];
    }
    if (!from_rsf)
        rmsnorm_fv(hb, xb, hb_w, n_embd);
    hb_stale = 0;
    return 0;
}

// Before a list that may FETCH hb from main RAM (its prepare, when RS_F no longer holds hb).
static void hb_ready(void)
{
    if (hb_stale && !rl_rsf_is(hb))
        hb_materialize();
}


static void l2norm_fv(float *v, int n)   // ggml_l2_norm: v / max(|v|, eps)
{
    float ss = sumsq64(v, n);
    int i;
    float nrm = sqrtf(ss);
    if (nrm < rms_eps) nrm = rms_eps;
    // Canonical (doc/units.md (Canonical reductions)): one correctly rounded reciprocal, then a
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
        infer_fetch_f32(TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)r*rb)*32, rb,
                        dst + (uint64_t)r*e->cols, e->cols);
    }
    return dst;
}

// Portable fallback for targets without scratchpad STORE + cache invalidation. The XCU1525
// uses the fused path (infer_rotate_quantize_f) and the resident lists.
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
    if (src == hb) hb_materialize();
    int e8 = infer_rotate_quantize_f(src, (uint32_t)width);
    if (e8 != INFER_ROTATE_UNSUPPORTED) return e8;
    rotate_f_legacy(src, rot, width);
    return infer_quantize_f(rot, (uint32_t)width);
}

// Resident preparation (resident.c: units 4/3/5 in the scratchpad) with the legacy CPU
// path as fallback. The prepared source is remembered so a matvec the resident path
// declines (LM head rows, a failed list) can requantize and take the CPU path.
#define E8_RESIDENT 0x40000001

// The list builder's prepare naming the matvec that consumes the vector: its weights are
// prefetched under the prepare's units (rl_prepare_pf; nothing for a NULL entry).
static void rl_prepare_for(const float *src, uint32_t width, int rotate, const struct bons_entry *e)
{
    rl_prepare_pf(src, width, rotate, e, 0, e ? e->rows : 0);
}
// A list image holding prefetch descriptors is named apart from one without (`prefetch off`):
// the bit goes into an aligned pointer of the list's name.
static inline uint32_t pf_key(void) { return (uint32_t)(engine_pf_use && hybrid_pf_use); }

static uint32_t f32_bits(float x);

/* Output Projection + Residual */
/*------------------------------*/

// A mixer's or the FFN's output projection, the residual add into the stream (XRES_SP) and the
// next norm as ONE list: staged (out_resid_build) under the list before it, started
// (out_resid_start) after that list's SiLU fixup patch, so neither its build nor the residual
// list's round trip is exposed. src: the projection's input in main RAM (fetched; the attention
// output), NULL: in RS_F (the gated SiLU's og, the FFN activation); e: the projection; w_next:
// the next norm's weight (NULL: no norm); sync: xb brought up to date (the trace, the last
// layer). Traced, the projection's output is stored to tmp_h and hb to main RAM for the trace's
// taps, so the list has one structure in both modes. out_resid_start: 0 done (RS_F holds hb when
// w_next), -1 declined (nothing ran: the caller's two-list path), -2 the list failed.
enum { OUT_SITE_DN, OUT_SITE_ATTN, OUT_SITE_FFN };
static struct { const float *w_next; int sync, traced, built; } out_resid;

static int out_resid_build(const float *src, uint32_t width, const struct bons_entry *e,
                           const float *w_next, int sync, int site, int l)
{
    struct bns_seq_descriptor d;
    uint32_t n = (uint32_t)n_embd, bytes = 4u*n, wsp = RS_Y_ADDR + ((bytes + 31u) & ~31u);
    out_resid.built = 0;
    if (!hybrid_fuse_out || !HP->xres_on || !e || !resident_active() || width % 32 ||
        width != infer_matvec_cols(e) || ((uintptr_t)src & 31))
        return -1;                                 // (rl_prepare_pf bounds the width itself)
    int traced = infer_dbg_armed() != 0;           // a count of forwards: one key bit
    // The name: the layer and site, the source's presence and whether RS_F holds it, sync, the
    // trace; the entry (its low bit the prefetch); the next norm's weight.
    int hb_store = traced || hybrid_hb_store;      // the hb store descriptor's presence (the key)
    rl_begin_keyed(0x0E000000u | (uint32_t)l << 8 | (uint32_t)hb_store << 7 | (uint32_t)site << 4 |
                   (uint32_t)(src != NULL) << 3 | (uint32_t)(src && rl_rsf_is(src)) << 2 |
                   (uint32_t)sync << 1 | (uint32_t)traced,
                   (uint32_t)(uintptr_t)e | pf_key(), (uint32_t)(uintptr_t)w_next);
    rl_prepare_pf(src, width, 1, e, 0, e->rows);
    rl_matvec(e, RS_Y_ADDR, RS_F_ADDR);
    if (traced)
        rl_store(tmp_h, bytes, RS_F_ADDR);         // the trace's dn_out / attn_out / ffn_down
    bns_seq_desc_add(&d, XRES_SP, RS_F_ADDR, XRES_SP, n);
    rl_desc(&d);
    if (sync)
        rl_store(xb, bytes, XRES_SP);
    if (w_next) {
        rl_fetch_raw(w_next, bytes, wsp);          // a norm weight: cleaned at load
        bns_seq_desc_norm(&d, 0, XRES_SP, RS_F_ADDR, wsp, n, 1, f32_bits(rms_eps));
        rl_desc(&d);
        if (hb_store)
            rl_store(hb, bytes, RS_F_ADDR);
    }
    out_resid.w_next = w_next; out_resid.sync = sync; out_resid.traced = traced;
    out_resid.built = 1;
    return 0;
}

static int out_resid_start(void)
{
    if (!out_resid.built)
        return -1;
    out_resid.built = 0;
    int post = !out_resid.sync && HP->keep_out;
    if (rl_start("out+residual"))
        return -1;                                 // declined: nothing ran
    if (out_resid.w_next) {
        resident_rsf_set(hb);                      // the next list's prepare of hb reads RS_F
        if (out_resid.traced || hybrid_hb_store) {
            hb_stale = 0;
        } else {
            hb_stale = 1;
            hb_w = out_resid.w_next;
        }
    }
    if (!post && rl_wait())
        return -2;
    return 0;
}


static int prepare_f(const float *src, int width, int rotate)
{
    struct resident_state *R = resident_state();
    R->prep_src = src; R->prep_width = width; R->prep_rotate = rotate;
    if (src == hb) hb_ready();
    if (resident_prepare(src, (uint32_t)width, rotate) == 0)
        return E8_RESIDENT;
    if (src == hb) hb_materialize();
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
        if (R->prep_src == hb) hb_materialize();
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
        // The prepare and the matvec in one list, the weights prefetched under the prepare;
        // declined (nothing run): the two-list path below.
        struct resident_state *R = resident_state();
        R->prep_src = src; R->prep_width = (int)width; R->prep_rotate = 1;
        if (resident_prepare_matvec(src, width, 1, e, dst) == 0)
            return;
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

/* Init */
/*------*/

int hybrid_init(void)
{
    H = model_header();
#ifdef P2_BUILT
    hseq_cur = 0;                      // a new heap: the second sequence's set is gone
    hseq_ready = 0;
    p2_dual = 0;
#endif
    memset((void *)HL, 0, sizeof(struct hyb_live));      // .live is not .bss
    memset(hyb_live_base, 0, sizeof hyb_live_base);
    HL->version = HYB_LIVE_VERSION;
    HL->magic   = HYB_LIVE_MAGIC;
    out_norm = NULL;
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
    embed_restore_off();
    if (H->hadamard_block && !model_entry_kind(ROLE_TOK_EMB16, 0, KIND_NORM16)) {
        // A 1 GB board's image: each embedding row is restored on the CPU (embed_row_restore).
        int wi = -1;
        for (int i = 0; i < 4; i++) if ((int)H->hadamard_widths[i] == n_embd) wi = i;
        if (wi < 0 || embed_restore_setup(H->hadamard_block, n_embd, wi)) {
            printf("hybrid: rotated basis, no emb16 rows and no rotation tensors to restore them\n");
            return -1;
        }
        printf("hybrid: embedding rows restored on the CPU (no emb16 table)\n");
    }
    n_dn = n_attn = 0;
    int n_lay = n_layer;
    dn_slot = infer_halloc(n_lay*sizeof(int), 0);
    attn_slot = infer_halloc(n_lay*sizeof(int), 0);
    for (int l = 0; l < n_lay; l++) {
        if (bons_layer_is_deltanet(H, l)) { dn_slot[l] = n_dn++; attn_slot[l] = -1; }
        else                              { attn_slot[l] = n_attn++; dn_slot[l] = -1; }
    }
    dn_hw = 0;
#ifdef DN_STATE_HW
    dn_hw = n_dn && hv == DN_HW_ROWS && hk == DN_HW_COLS;
    if (dn_hw) {
        // Keep the same conservative KV reservation as the image converter, then place
        // whole recurrent matrices immediately above it (the attention stream's K/V spans
        // follow above the state).
        uint32_t kv_base = (uint32_t)((TRIXIUM_CHAN_WEIGHT_BASE + H->payload_beats*32
                                      + 0xFFFu) & ~0xFFFu);
        uint32_t kv_rb = 1u + (uint32_t)head_dim/32u;
        uint32_t kv_region = ((uint32_t)H->max_ctx + (uint32_t)n_ch - 1u)
                           /(uint32_t)n_ch*kv_rb;
        uint64_t base = (uint64_t)kv_base
                      + (uint64_t)n_lay*n_kv_head*2u*kv_region*32u;
        dn_hbm_base = (uint32_t)((base + 0xFFFu) & ~0xFFFull);
        uint64_t state_per_channel = ((uint64_t)n_dn*nv + n_ch - 1u)/n_ch*DN_HEAD_BYTES;
        uint64_t limit = chan_limit();
        dn_set_bytes = 0;
#ifdef TRIXIUM_DNSTREAM_WOFF
        // The second set when it fits (the write offset is a signed 24-bit beat count).
        if ((uint64_t)dn_hbm_base + 2u*state_per_channel <= limit && state_per_channel < (8ull << 20)*32u) {
            dn_set_bytes = (uint32_t)state_per_channel;
            state_per_channel *= 2u;
        }
#endif
        if ((uint64_t)dn_hbm_base + state_per_channel > limit) {
            printf("hybrid: DeltaNet state reaches %08lx, past channel limit %08lx - "
                   "reconvert for this board\n",
                   (unsigned long)((uint64_t)dn_hbm_base + state_per_channel),
                   (unsigned long)limit);
            return -1;
        }
        HP->dnres_enabled = 1;
    }
#endif
#ifdef TRIXIUM_HAS_ATTNSTREAM
    // Attention stream (engine op 6): K and V rows position-striped over the channels (row
    // t on channel t % n_ch, slot t / n_ch), per attention layer a K span then a V span,
    // above the DeltaNet state. main RAM keeps its copy (the CPU path's, always complete).
    at_geom_ok = 0;
    {
        uint32_t kv_base = (uint32_t)((TRIXIUM_CHAN_WEIGHT_BASE + H->payload_beats*32 + 0xFFFu) & ~0xFFFu);
        uint32_t kv_rb = 1u + (uint32_t)head_dim/32u;
        uint32_t kv_region = ((uint32_t)H->max_ctx + (uint32_t)n_ch - 1u)/(uint32_t)n_ch*kv_rb;
        uint64_t base = (uint64_t)kv_base + (uint64_t)n_lay*n_kv_head*2u*kv_region*32u;
#ifdef DN_STATE_HW
        if (dn_hw)
            base = (uint64_t)dn_hbm_base + ((uint64_t)n_dn*nv + n_ch - 1u)/n_ch*DN_HEAD_BYTES
                 *(dn_set_bytes ? 2u : 1u);
#endif
        base = (base + 0xFFFu) & ~0xFFFull;
        uint64_t limit = chan_limit();
        at_rowbeats = (uint32_t)(n_kv_head*head_dim)/8u;
        at_slots = ((uint32_t)H->max_ctx + (uint32_t)n_ch - 1u)/(uint32_t)n_ch;
        uint64_t end = base + (uint64_t)n_attn*2u*at_slots*at_rowbeats*32u;
        at_hbm_base = (uint32_t)base;
        // kv heads per command: all of them, or (TRIXIUM_ATTN_KV_SLICE) as many as the unit's
        // buffers and lanes hold, one command per slice (K/V streamed once per slice).
        at_sg = n_kv_head;
#ifdef TRIXIUM_ATTN_KV_SLICE
        if (n_kv_head && n_head % n_kv_head == 0) {
            int rep = n_head/n_kv_head, sg = TRIXIUM_ATTN_GROUPS;
            if (sg > TRIXIUM_ATTN_LANES/rep) sg = TRIXIUM_ATTN_LANES/rep;
            while (sg > 1 && n_kv_head % sg) sg--;
            if (sg >= 1 && sg < n_kv_head && n_kv_head/sg <= 16) at_sg = sg;
        }
#endif
        at_geom_ok = n_attn && at_sg >= 1 && at_sg*(n_head/n_kv_head) <= TRIXIUM_ATTN_LANES &&
                     at_sg <= TRIXIUM_ATTN_GROUPS && head_dim >= 8 && head_dim <= TRIXIUM_ATTN_HD_MAX && !(head_dim & (head_dim - 1)) &&
                     n_head % n_kv_head == 0 && at_rowbeats <= 255 && end <= limit;
        if (at_geom_ok && at_sg < n_kv_head)
            printf("hybrid: attention stream in %d kv-head slices of %d (one command each)\n",
                   n_kv_head/at_sg, at_sg);
        if (n_attn && !at_geom_ok)
            printf("hybrid: attention stream off (geometry or K/V span %08lx..%08lx past %08lx)\n",
                   (unsigned long)base, (unsigned long)end, (unsigned long)limit);
    }
#endif
    // The four-tap kernel's ring has a fourth slot: the new input never overwrites the window
    // it was computed from, so a forward can be re-run (hybrid_ckpt_save).
    dn_conv_w = conv_k == 4 ? 4 : conv_k - 1;
    // FP32 state. The matrix lives in engine DDR when the stream op can update it; the
    // short causal-convolution history remains in main RAM either way.
    uint64_t matrix_b = (uint64_t)n_dn*nv*hv*hk*4;
    uint64_t conv_b = (uint64_t)n_dn*dn_conv_w*(2*nq + nvh)*4;
    uint64_t state_b = (dn_hw ? 0 : matrix_b) + conv_b;
    ctx_cap = H->max_ctx;
    uint64_t kv_b = (uint64_t)n_attn*ctx_cap*kv_dim*4*2;
    uint64_t room = infer_heap_room(1);
    // Work buffers: the widest activation a few times over, plus the logits.
    uint64_t work_b = 24u*(uint64_t)(ffn_dim > 2*nq + nvh ? ffn_dim : 2*nq + nvh)*4
                    + 4u*H->vocab + (256u << 10) + sizeof(float)*(TRIXIUM_SILU_LUT_N + 1);
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
    dn_conv  = n_dn ? infer_halloc((uint64_t)n_dn*dn_conv_w*(2*nq + nvh)*4, 1) : NULL;
    dn_conv_pos = n_dn ? infer_halloc(n_dn, 0) : NULL;
    kc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    vc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    kc_rows = n_attn ? infer_halloc(((uint64_t)n_attn*2 + 31) & ~31ull, 0) : NULL;
    silu_lut = infer_halloc(sizeof(float)*(TRIXIUM_SILU_LUT_N + 1), 0);
    silu_lut_init();
    sigmoid_lut = infer_halloc(sizeof(float)*(TRIXIUM_SILU_LUT_N + 1), 0);
    for (int i = 0; i <= TRIXIUM_SILU_LUT_N; i++)
        sigmoid_lut[i] = trixium_sigmoid_point(-TRIXIUM_SILU_LUT_LIMIT + (float)i*(1.0f/TRIXIUM_SILU_LUT_SCALE));

    SIM_MARK("hybrid: silu/sigmoid tables");
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
#ifdef P2_BUILT
    {
        // The batched prefill's park: row 1's DeltaNet qkv + z or attention q + k + v. Only the
        // engine writes and reads it (raw transfers): no dirty line of the CPU's may remain.
        uint32_t dn_b = 4u*(uint32_t)(2*nq + 2*nvh), at_b = 4u*(uint32_t)(2*q_dim + 2*kv_dim);
        p2_park_bytes = (dn_b > at_b ? dn_b : at_b) + 64u;
        p2_park = infer_halloc(p2_park_bytes, 0);
        if (p2_park)
            flush_cpu_dcache_range(p2_park, p2_park_bytes);
        else
            p2_park_bytes = 0;
    }
#endif
    gateb = infer_halloc(ffn_dim*4, 0); upb = infer_halloc(ffn_dim*4, 0);
    actb = infer_halloc(ffn_dim*4, 0);   // Resident FFN activation (gateb keeps the gate trace).
    scores = infer_halloc(ctx_cap*4, 0);
#ifdef TRIXIUM_HAS_ATTNSTREAM
    // Scores, then p, for every head: [n_head][ctx_cap rounded to 8].
    at_buf = at_geom_ok ? infer_halloc((uint64_t)n_head*((ctx_cap + 7) & ~7)*4, 1) : NULL;
    if (!at_buf) at_geom_ok = 0;
#endif
    logits = infer_halloc((uint64_t)H->vocab*4 + 64*n_ch*4, 1);
    if (!logits) { printf("hybrid: no room for the logits\n"); return -1; }

    norm_attn = infer_halloc(n_lay*sizeof(float *), 0); norm_ffn = infer_halloc(n_lay*sizeof(float *), 0);
    norm_q = infer_halloc(n_lay*sizeof(float *), 0);    norm_k = infer_halloc(n_lay*sizeof(float *), 0);
    norm_ssm = infer_halloc(n_lay*sizeof(float *), 0);  ssm_a = infer_halloc(n_lay*sizeof(float *), 0);
    ssm_dt = infer_halloc(n_lay*sizeof(float *), 0);    conv_w = infer_halloc(n_lay*sizeof(float *), 0);
    for (int l = 0; l < n_lay; l++) {
        // Norm weights are constants the lists fetch raw (rl_fetch_raw): cleaned once here.
        norm_attn[l] = fetch_f32(model_entry_kind(ROLE_ATTN_NORM, l, KIND_F32));
        norm_ffn[l]  = fetch_f32(model_entry_kind(ROLE_FFN_NORM, l, KIND_F32));
        if (norm_attn[l]) clean_cpu_dcache_range(norm_attn[l], (size_t)n_embd*4);
        if (norm_ffn[l])  clean_cpu_dcache_range(norm_ffn[l], (size_t)n_embd*4);
        if (!norm_attn[l] || !norm_ffn[l]) { printf("hybrid: layer %d norms missing\n", l); return -2; }
        if (dn_slot[l] >= 0) {
            norm_ssm[l] = fetch_f32(model_entry_kind(ROLE_SSM_NORM, l, KIND_F32));
            if (norm_ssm[l]) clean_cpu_dcache_range(norm_ssm[l], (size_t)hv*4);
            ssm_a[l]    = fetch_f32(model_entry_kind(ROLE_SSM_A, l, KIND_F32));
            ssm_dt[l]   = fetch_f32(model_entry_kind(ROLE_SSM_DT, l, KIND_F32));
            conv_w[l]   = fetch_f32(model_entry_kind(ROLE_SSM_CONV, l, KIND_F32));
            if (conv_w[l])             // the resident conv fetches the taps raw
                clean_cpu_dcache_range(conv_w[l], (size_t)(2*nq + nvh)*conv_k*4);
            if (!norm_ssm[l] || !ssm_a[l] || !ssm_dt[l] || !conv_w[l]) {
                printf("hybrid: layer %d DeltaNet tables missing\n", l);
                return -2;
            }
            norm_q[l] = norm_k[l] = NULL;
        } else {
            norm_q[l] = fetch_f32(model_entry_kind(ROLE_ATTN_Q_NORM, l, KIND_F32));
            norm_k[l] = fetch_f32(model_entry_kind(ROLE_ATTN_K_NORM, l, KIND_F32));
            if (norm_q[l]) clean_cpu_dcache_range(norm_q[l], (size_t)head_dim*4);
            if (norm_k[l]) clean_cpu_dcache_range(norm_k[l], (size_t)head_dim*4);
            if (!norm_q[l] || !norm_k[l]) { printf("hybrid: layer %d q/k norms missing\n", l); return -2; }
        }
    }
    SIM_MARK("hybrid: norms");
    // Plain RoPE over rope_n dims (the 27B: theta 1e7, no YaRN; the header has no scaling
    // fields).
    rope_c = infer_halloc((rope_n/2)*4, 0); rope_s = infer_halloc((rope_n/2)*4, 0);
    rope_freq = infer_halloc((rope_n/2)*4, 0);
    for (int i = 0; i < rope_n/2; i++)
        rope_freq[i] = powf((float)H->rope_theta, -2.0f*i/rope_n);
    rope_pos_cached = -1;
    HP->dnstream_active = 0;
    HP->attnstream_active = 0;
    HP->hwargmax = 1;
    HP->lm_chunk = 0;
    HP->logits_clean = 0;
#ifdef TRIXIUM_HAS_ATTNSTREAM
    HP->attnstream_enabled = 1;
#else
    HP->attnstream_enabled = 0;
#endif
#ifdef TRIXIUM_HAS_DNSTREAM
    HP->dnstream_enabled = 1;
#else
    HP->dnstream_enabled = 0;
#endif
#ifdef TRIXIUM_SIM
    // The first reset after power-on: the DeltaNet state (engine DDR) and the conv window
    // (fresh heap) still hold the memory models' initial zeros - nothing stores there before
    // the first generation, and the CPU has no line of either.
    power_on_reset = !hybrid_inits++;
#endif
    hybrid_reset();
#ifdef TRIXIUM_SIM
    power_on_reset = 0;
#endif
    SIM_MARK("hybrid: reset");
    HP->dnres_enabled = 0;
    HP->xres_enabled = 0;
    HP->xres_row = 0;
    HP->prefill2 = 1;
    HP->pair_geom = 0;
    HP->pair_inproj = 1;
    HP->lmhead2 = 0;                  // Off: on the card's 2B (24,576-row chunks) the two-row head
                                      // gave wrong tokens (the tiny image's 128-row chunks pass): to fix.
    if (H->hadamard_block && resident_available()) {
        if (resident_init() == 0) {
            puts("hybrid: resident projections (scratchpad quantize/rotate/restore lists) on");
            uint32_t *stage = infer_halloc(1025*32, 0);   // Table image for the DMA.
            if (resident_silu_init(silu_lut, stage) == 0) {
                puts("hybrid: resident FFN activation (unit 11 SiLU-multiply) on");
#ifdef TRIXIUM_HAS_ATTNSTREAM
                uint32_t *sig_image = n_attn && H->attn_gated ? infer_halloc(1025*32, 0) : NULL;
                if (sig_image && resident_gate_init(sigmoid_lut, sig_image) == 0)
                    puts("hybrid: attention output gate on unit 11 (sigmoid table, ABI 1.2)");
#endif
            }
#ifdef DNRES_BUILT
            HP->dnres_enabled = 1;
            HP->xres_enabled = 1;
            if (n_dn && resident_silu_active())
                puts("hybrid: resident DeltaNet layer (conv/norm/SiLU units, lists) on");
#ifdef TRIXIUM_HAS_DNSTREAM
            if (n_dn && dn_hw && TRIXIUM_DNSTREAM_LANES == n_ch)
                printf("hybrid: DeltaNet stream (engine op 5, %d lanes) on from the next reset\n", n_ch);
#endif
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

/* Reset */
/*-------*/

// The conv window (5.9 MB on the 27B) zeroed by the engine: 31-packet STOREs of one zeroed
// 4 KB scratchpad block, much faster than a CPU memset through cold lines plus the flush.
// The window is device-owned (the CPU path flushes around its own use), so no cache
// maintenance. 0 = done; -1: the caller's memset.
static int dn_conv_zero(void)
{
#ifdef DNRES_BUILT
    uint64_t bytes = (uint64_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4;   // slots 0..conv_k-2: read first
    const uint32_t blk = 4096u, per = 31u*blk;
    if (!resident_active() || ((uintptr_t)dn_conv & 31) || bytes % 32 ||
        (bytes + per - 1)/per + 1 > 60u)
        return -1;
    volatile uint32_t *z = (volatile uint32_t *)(SCRATCHPAD_BASE + RS_F_ADDR);
    for (uint32_t i = 0; i < blk/4; i++) z[i] = 0;
    (void)z[blk/4 - 1];                // the posted writes land before the list starts
    rl_begin();
    uint8_t *p = (uint8_t *)dn_conv;
    for (uint64_t off = 0; off < bytes; ) {
        uint64_t left = bytes - off;
        if (left >= blk) {
            uint32_t n = left/blk > 31u ? 31u : (uint32_t)(left/blk);
            rl_store_packets(p + off, blk, n, blk, RS_F_ADDR, 0, NULL, 0);
            off += (uint64_t)n*blk;
        } else {
            rl_store_raw(p + off, (uint32_t)left, RS_F_ADDR);
            off += left;
        }
    }
    return rl_run("conv window reset") ? -1 : 0;
#else
    return -1;
#endif
}

void hybrid_reset(void)
{
    pos = 0;
    kv_pin = kv_pin_next < KV_SINKS ? KV_SINKS : kv_pin_next > ctx_cap/4 ? ctx_cap/4 : kv_pin_next;
    if (kv_pin < KV_SINKS) kv_pin = KV_SINKS;
    kv_pin_next = KV_SINKS;            // One generation's: gen/bench take the default.
#ifdef TRIXIUM_HAS_ATTNSTREAM
    HP->attnstream_active = HP->attnstream_enabled && at_geom_ok && resident_active();
#endif
    if (dn_hw) {
        int cleared = 0;
        dn_state_zero = 0;
        dn_par = 0;
#ifdef TRIXIUM_HAS_DNSTREAM
        HP->dnstream_active = HP->dnstream_enabled && hybrid_dnres_enabled()
                            && TRIXIUM_DNSTREAM_LANES == n_ch && DN_HEAD_BEATS % (uint32_t)n_ch == 0;
#ifdef TRIXIUM_DNSTREAM_ZERO
        // Only op 5 touches the stream layout (every other path stops with DNSTALL), so the
        // first op of each layer can take its state as zero instead of the clear.
        if (HP->dnstream_active && n_dn <= 64) {
            dn_state_zero = n_dn == 64 ? ~0ull : (1ull << n_dn) - 1u;
            cleared = 1;
        }
#endif
#endif
#ifdef TRIXIUM_SIM
        // Simulation only: the boot-time clear (~0.8M cycles, ~40 s of Verilator: a CPU
        // zero-fill of 64 KB of scratchpad, then engine STOREs) would write zeros over the
        // memory model's power-on zeros. Every later reset clears as on the board.
        if (power_on_reset)
            cleared = dn_hw_written = 1;
#endif
#ifdef DN_STATE_HW
        // Once per boot even when op 5 takes the state as zero: never-written DDR holds no
        // valid ECC, so those reads would count as uncorrectable (and re-run the forward).
        if (!cleared || !dn_hw_written) {
            dn_hw_reset();
            dn_hw_written = !engine_wedged;
        }
#endif
        (void)cleared;
    } else if (dn_state) {
        memset(dn_state, 0, (size_t)n_dn*nv*hv*hk*4);
    }
#ifdef TRIXIUM_SIM
    // Simulation only, likewise: the boot-time memset + flush of the conv window
    // (~0.4M cycles: every line a main_ram miss, then a write-back).
    if (dn_conv && !power_on_reset && dn_conv_zero()) {
#else
    if (dn_conv && dn_conv_zero()) {
#endif
        memset(dn_conv, 0, (size_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4);
        flush_cpu_dcache_range(dn_conv, (size_t)n_dn*(conv_k - 1)*(2*nq + nvh)*4);   // device-owned
    }
    if (dn_conv_pos) memset(dn_conv_pos, 0, (size_t)n_dn);
    if (kc_rows) memset(kc_rows, 0, (size_t)n_attn*2);
    hybrid_stats_reset();
}

/* Profiling Report */
/*------------------*/

void hybrid_stats_reset(void)
{
    if (HL->magic == HYB_LIVE_MAGIC) {
        const uint64_t c[HYB_LIVE_STAGES] = {hyb_cyc_embed, hyb_cyc_anorm, hyb_cyc_dn, hyb_cyc_attn,
            hyb_cyc_resid, hyb_cyc_fprep, hyb_cyc_fproj, hyb_cyc_fact, hyb_cyc_fdown, hyb_cyc_logits};
        for (int i = 0; i < HYB_LIVE_STAGES; i++)
            hyb_live_base[i] += c[i];
    }
    dn_cyc_inproj = dn_cyc_act = dn_cyc_conv = dn_cyc_prep = 0;
    dn_cyc_heads = dn_cyc_gate = dn_cyc_outproj = 0;
    hyb_cyc_embed = hyb_cyc_anorm = hyb_cyc_dn = hyb_cyc_attn = 0;
    hyb_cyc_resid = hyb_cyc_fprep = hyb_cyc_fproj = hyb_cyc_fact = 0;
    hyb_cyc_fdown = hyb_cyc_logits = 0;
    HP->fd_prep = HP->fd_mv = 0;
    HP->at_prep = HP->at_qkv = HP->at_rope = HP->at_core = HP->at_out = 0;
    memset(HP->at_sub, 0, sizeof HP->at_sub);
    hyb_forwards_profiled = 0;
    resident_stats_reset();
}

void hybrid_stats_print(void)
{
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
        if (hyb_cyc_fdown)
            printf("hybprof fdown: avg cycles/fwd prepare %lu matvec %lu\n",
                   (unsigned long)(HP->fd_prep/hyb_forwards_profiled),
                   (unsigned long)(HP->fd_mv/hyb_forwards_profiled));
        if (hyb_cyc_attn && HP->at_core)
            printf("attnprof core: avg cycles/fwd kvcopy %lu scores-build %lu scores-run %lu softmax %lu "
                   "values-build %lu values-run %lu\n",
                   (unsigned long)(HP->at_sub[0]/hyb_forwards_profiled),
                   (unsigned long)(HP->at_sub[1]/hyb_forwards_profiled),
                   (unsigned long)(HP->at_sub[2]/hyb_forwards_profiled),
                   (unsigned long)(HP->at_sub[3]/hyb_forwards_profiled),
                   (unsigned long)(HP->at_sub[4]/hyb_forwards_profiled),
                   (unsigned long)(HP->at_sub[5]/hyb_forwards_profiled));
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
        resident_listcache_print();
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
int hybrid_context_unbounded(void) { return kv_ring_on(); }

/* DeltaNet Layer */
/*----------------*/

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
    resident_join();                   // h from a posted residual-stream list
    if (h == hb) hb_materialize();     // ... which left it in the scratchpad
    int   s = dn_slot[l];
    int   C = 2*nq + nvh;
    float *S    = dn_hw ? NULL : dn_state + (uint64_t)s*nv*hv*hk;

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
    // The window is device-owned while the resident layer runs (raw engine transfers):
    // take it back before reading, hand it back after writing.
    int W = dn_conv_w;
    for (int i = 0; i < W; i++)
        flush_cpu_dcache_range(dn_conv_slot(s, i, C), (size_t)C*4);
    if (conv_k == 4) {
        // The 27B's fixed four-tap kernel.  A circular history removes three 40 KiB
        // memcpys per layer, and spelling out the taps removes the runtime inner loop.
        // The oldest-to-newest accumulation order is unchanged, so FP32 results are exact.
        // The input goes into the slot after the window (the oldest one when W == 3).
        int p0 = dn_conv_pos[s], p1 = p0 + 1 == W ? 0 : p0 + 1;
        int p2 = p1 + 1 == W ? 0 : p1 + 1, pn = p2 + 1 == W ? 0 : p2 + 1;
        const float *x0 = dn_conv_slot(s, p0, C), *x1 = dn_conv_slot(s, p1, C), *x2 = dn_conv_slot(s, p2, C);
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
        memcpy(dn_conv_slot(s, pn, C), qkv, (size_t)C*4);
        dn_conv_pos[s] = p1;
    } else {
        // Other kernels: a shifted window.
        for (int c = 0; c < C; c++) {
            float acc = 0;
            for (int i = 0; i < conv_k - 1; i++) acc += w[c*conv_k + i]*dn_conv_slot(s, i, C)[c];
            acc += w[c*conv_k + conv_k - 1]*qkv[c];
            convb[c] = siluf(acc);
        }
        for (int i = 0; i < conv_k - 2; i++)
            memcpy(dn_conv_slot(s, i, C), dn_conv_slot(s, i + 1, C), (size_t)C*4);
        if (conv_k > 1)
            memcpy(dn_conv_slot(s, conv_k - 2, C), qkv, (size_t)C*4);
    }
    for (int i = 0; i < W; i++)
        flush_cpu_dcache_range(dn_conv_slot(s, i, C), (size_t)C*4);
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
    if (dn_hw) {
        // The state is in engine DDR, which only the stream op of the resident layer updates.
        if (!engine_wedged)
            puts("DNSTALL: the DeltaNet state is in engine DDR, the CPU layer cannot run it");
        engine_wedged = 1;
        return;
    }
    // The recurrence per value head (llama.cpp's autoregressive kernel): M = S[hh] rows j.
    float scale = 1.0f/sqrtf((float)hk);
    int rep = nv/nk;
    for (int hh = 0; hh < nv; hh++) {
        int kh = hh % nk, gpos = kh*rep + hh/nk;
        const float *qh = dn_q + kh*hk, *kk = dn_k + kh*hk, *vh = dn_v + hh*hv;
        float *M = S + (uint64_t)hh*hv*hk;
        t0 = perf_now64();
        float decay = trixium_decay_exp(gb[hh]);
        float *oh = ob + hh*hv;
        // One pass over the state: row j's delta depends on row j alone, so the decay,
        // the k dot, the update and the q dot happen while the row (512 B) is in the
        // D$ (a two-loop form streams the 64 KB head twice through the 32 KB cache).
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
    t0 = perf_now64();
    proj_f(model_entry(ROLE_SSM_OUT, l), og, out);
    dn_cyc_outproj += perf_now64() - t0;
}

/* Resident DeltaNet Layer */
/*-------------------------*/

#ifdef DNRES_BUILT
static uint32_t f32_bits(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    return v.u;
}

#ifdef P2_BUILT

// The lead's list fits: ternary single-K-segment entries, row 1's norm weight and every entry's
// two staged rows above RS_F (below the persistent Q8), the park big enough.
static int p2_lead_fits(const struct bons_entry *const *e, int ne)
{
    uint32_t n = (uint32_t)n_embd, stage = 0, park = 0;
    if (!p2_park || n % 32 || RS_F_ADDR + 8u*n > RS_XW_ADDR || RS_XW_ADDR + 2u*n > RS_E8_ADDR)
        return 0;
    for (int k = 0; k < ne; k++) {
        if (!e[k] || (e[k]->flags & FLAG_KSEG) || e[k]->cols != n ||
            (e[k]->kind != KIND_TERNARY && e[k]->kind != KIND_TERNARY5) ||
            e[k]->rows % 8 || e[k]->rows % (uint32_t)n_ch || e[k]->rows > 0xFFFFu)
            return 0;
        stage += 8u*e[k]->rows;
        park  += 4u*e[k]->rows;
    }
    return RS_F_ADDR + stage <= RS_XW_ADDR && park <= p2_park_bytes;
}

// After row 0's rotated prepare into slot 0 (RS_XW, RS_E8): row 1's attention norm (the
// residual list's descriptor on XRES_SP1) and prepare into slot 1, then each entry for both rows
// staged from RS_F up, row 0 restored to dst0[k], row 1 in place and stored to the park.
static void p2_lead_emit(int l, const struct bons_entry *const *e, const uint32_t *dst0, int ne)
{
    struct bns_seq_descriptor d;
    uint32_t n = (uint32_t)n_embd, w1 = RS_F_ADDR + ((4u*n + 31u) & ~31u), off = 0, S = RS_F_ADDR;
    const uint32_t e8[2] = {RS_E8_ADDR, RS_E8B_ADDR};
    rl_fetch_raw(norm_attn[l], 4u*n, w1);              // a norm weight: cleaned at load
    if (!rl_replaying())
        bns_seq_desc_norm(&d, 0, XRES_SP1, RS_F_ADDR, w1, n, 1, f32_bits(rms_eps));
    rl_desc(&d);
    rl_prepare_sp(RS_F_ADDR, n, 1, RS_XW_ADDR + n, RS_E8B_ADDR);
    for (int k = 0; k < ne; k++) {
        uint32_t rows = e[k]->rows, dst[2] = {dst0[k], S + 4u*rows};
        rl_matvec2(e[k], RS_XW_ADDR, n, e8, S, dst);
        rl_store_raw((uint8_t *)p2_park + off, 4u*rows, dst[1]);
        off += 4u*rows;
        S += 8u*rows;
    }
}

// Row 1: the parked outputs back where its projections would have restored them.
static void p2_follow_emit(const struct bons_entry *const *e, const uint32_t *dst0, int ne)
{
    for (uint32_t k = 0, off = 0; k < (uint32_t)ne; k++) {
        rl_fetch_raw((uint8_t *)p2_park + off, 4u*e[k]->rows, dst0[k]);
        off += 4u*e[k]->rows;
    }
}
#endif

// The resident DeltaNet layer: the whole layer in scratchpad lists (units 1/3/4/5/8 for the
// projections, 13 conv, 11 SiLU, 12 norms, engine op 5 for the heads), the CPU doing
// only the per-head scalars and the rare SiLU fixups. Bit-exact with deltanet_layer (the
// sim gate compares them checkpoint by checkpoint). Scratchpad map for this layer:
#define DNR_QKV   0x04800u        // qkv, then conv+SiLU output, q/k L2-normed in place (4C)
static uint32_t dnr_align(uint32_t x) { return (x + 31u) & ~31u; }

#ifdef TRIXIUM_HAS_DNSTREAM
// The stream op's records: one per head in processing order at slot0 (q, k, v, out byte
// addresses; decay, beta, scale), from betab/gb.
static void dn_stream_records(uint32_t slot0, uint32_t ho, float scale, int rep)
{
    volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + slot0);
    for (int hi = 0; hi < nv; hi++) {
        int kh = hi/rep, hh = (hi - kh*rep)*nk + kh;
        volatile uint32_t *w = rec + 8*hi;
        w[0] = DNR_QKV + 4u*(uint32_t)(kh*hk);
        w[1] = DNR_QKV + 4u*(uint32_t)(nq + kh*hk);
        w[2] = DNR_QKV + 4u*(uint32_t)(2*nq + hh*hv);
        w[3] = ho + 4u*(uint32_t)(hh*hv);
        w[4] = f32_bits(trixium_decay_exp(gb[hh]));
        w[5] = f32_bits(betab[hh]);
        w[6] = f32_bits(scale);
        w[7] = 0;
    }
}
#endif

static int deltanet_layer_resident(int l, const float *h, float *out)
{
    hb_ready();                        // a prepare of hb may FETCH it from main RAM
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
    // The stream op needs no head-state slots: its records sit at SLOT0 and the heads'
    // outputs right above them, so z stays at Z for the gated norm (no park in main RAM,
    // no fetch back).
    int z_stays = 0;
#ifdef TRIXIUM_HAS_DNSTREAM
    if (dn_hw && HP->dnstream_active &&
        SLOT0 + 0x800u + 4u*(uint32_t)nvh <= RS_F_ADDR && 32u*(uint32_t)nv <= 0x800u) {
        HO = SLOT0 + 0x800u;
        Z2 = Z;
        z_stays = 1;
    }
#endif
    // The conv's chunks: the taps in two buffers (W, W + 16*CH), so chunk c+1's taps stream
    // in (NOWAIT) while unit 13 convolves chunk c (OVERLAP), then the three window slots - 44*CH
    // bytes, inside RS_F.
    const uint32_t CH = 1920, W = RS_F_ADDR, X = RS_F_ADDR + 32*CH;
    _Static_assert(RS_F_ADDR + 44u*1920u <= 0x32000u, "conv buffers fit RS_F");
    uint32_t eps = f32_bits(rms_eps);
    if (!HP->dnres_enabled || !resident_silu_active() || !H->hadamard_block || conv_k != 4 ||
        Z2 + 4u*(uint32_t)nvh > RS_F_ADDR || C % 8 || hv % 8 || hk % 8 ||
        (trixium_seq_silu_abi_read() & 0xffff) < 1 || (trixium_seq_norm_abi_read() >> 16) != 1 ||
        (trixium_seq_conv_abi_read() >> 16) != 1)
        return -1;
#ifdef DN_STATE_HW
    if (dn_hw && (SLOT0 + 2u*DN_HEAD_BYTES > XRES_SP0 || nv > 64))
        return -1;                     // the two head-state slots must fit below x
#endif
    struct bns_seq_descriptor d;

    // A: projections. qkv and z restored in place where they will be used.
    // With the stream op, beta/alpha go first and the CPU computes the per-head scalars and
    // the op's records from the scratchpad while qkv and z stream (hidden behind the list).
    // Needs qkv and z in one K-segment each: several would stage partials above their
    // regions, over BA/AL and the records.
    const struct bons_entry *e_qkv = model_entry(ROLE_ATTN_QKV, l), *e_z = model_entry(ROLE_ATTN_ZGATE, l);
    int early = 0;
#ifdef TRIXIUM_HAS_DNSTREAM
    early = dn_hw && HP->dnstream_active && z_stays && e_qkv && e_z &&
            !(e_qkv->flags & FLAG_KSEG) && !(e_z->flags & FLAG_KSEG);
#endif
    // z beside the conv's SiLU (early only): its matvec streams while unit 11 works, instead
    // of lengthening the projection list. It reads the prepared input at RS_XW (the conv
    // buffers end below it) and writes Z; the SiLU touches DNR_QKV and FIX; z is used by the
    // gated norm in the core list.
    int zbeside = early && hybrid_dn_zbeside;
    // The batched prefill: qkv and z of both rows in row 0's list (lead), fetched back in row 1's.
    int lead = 0, follow = 0;
#ifdef P2_BUILT
    const struct bons_entry *p2e[2] = {e_qkv, e_z};
    const uint32_t p2d[2] = {DNR_QKV, Z};
    if (early && !zbeside) {
        lead   = p2_mode == P2_LEAD && p2_lead_fits(p2e, 2);
        follow = p2_mode == P2_FOLLOW;
    }
#endif
    float scale = 1.0f/sqrtf((float)hk);
    int rep = nv/nk;
    uint64_t t0 = perf_now64(), act_dt = 0, conv_dt = 0;
    // The causal conv's list (B below): the window slots (oldest to newest p0, p1, p2; the
    // new input replaces p0 after its chunk was read), then SiLU in place with a unit
    // multiplier. Built while the projections stream when they are posted (early).
    int CW = dn_conv_w, p0 = dn_conv_pos[s], p1 = p0 + 1 == CW ? 0 : p0 + 1;
    int p2 = p1 + 1 == CW ? 0 : p1 + 1, pn = p2 + 1 == CW ? 0 : p2 + 1;
    float *x0 = dn_conv_slot(s, p0, C), *x1 = dn_conv_slot(s, p1, C), *x2 = dn_conv_slot(s, p2, C);
    float *xn = dn_conv_slot(s, pn, C);
#define DN_CONV_LIST() do {                                                                      \
        rl_begin();                                                                              \
        /* Raw transfers: the taps were cleaned once at load, and the window is device-owned */  \
        /* (the CPU path flushes it around its own use, hybrid_reset after its memset).      */  \
        rl_fetch_raw(conv_w[l], 16*(C < CH ? C : CH), W);                                        \
        for (uint32_t c0 = 0, b = 0; c0 < C; c0 += CH, b ^= 1) {                                 \
            uint32_t ch = C - c0 < CH ? C - c0 : CH, wb = W + b*16*CH;                           \
            rl_fetch_raw(x0 + c0, 4*ch, X);                                                      \
            rl_fetch_raw(x1 + c0, 4*ch, X + 4*CH);                                               \
            rl_fetch_raw(x2 + c0, 4*ch, X + 8*CH);                                               \
            rl_store_raw(xn + c0, 4*ch, DNR_QKV + 4*c0);                                         \
            bns_seq_desc_conv(&d, X, X + 4*CH, X + 8*CH, DNR_QKV + 4*c0, wb, DNR_QKV + 4*c0, ch);\
            if (c0 + CH < C) {                                                                   \
                uint32_t cn = C - c0 - CH < CH ? C - c0 - CH : CH;                               \
                rl_fetch_raw(conv_w[l] + 4*(c0 + CH), 16*cn, W + (b ^ 1)*16*CH);                 \
                rl_nowait();                                                                     \
                rl_desc_overlap(&d);                                                             \
            } else {                                                                             \
                rl_desc(&d);                                                                     \
            }                                                                                    \
        }                                                                                        \
        bns_seq_desc_silu_ex(&d, DNR_QKV, DNR_QKV, DNR_QKV, FIX, C, 1, 0, 0, 0);                 \
        if (zbeside)                                                                             \
            rl_matvec_beside(e_z, Z, Z, &d);                                                     \
        else                                                                                     \
            rl_desc(&d);                                                                         \
    } while (0)
    perf_enter(OP_DN_PROJ);           // the token trace's phases (tprof): nested brackets
    rl_begin_keyed(0x04000000u | (uint32_t)l << 8 | (uint32_t)early | (uint32_t)z_stays << 1 |
                   (uint32_t)rl_rsf_is(h) << 2 | (uint32_t)zbeside << 3 | (uint32_t)lead << 4 |
                   (uint32_t)follow << 5,
                   (uint32_t)(uintptr_t)h | pf_key(), 0);
    if (early) {
        rl_prepare_for(h, (uint32_t)n_embd, 0, model_entry(ROLE_SSM_BETA, l)); // beta/alpha are not folded
        rl_matvec(model_entry(ROLE_SSM_BETA, l), BA, BA);
        rl_matvec(model_entry(ROLE_SSM_ALPHA, l), AL, AL);
        uint32_t k_ba = rl_count();
#ifdef P2_BUILT
        if (follow) {
            p2_follow_emit(p2e, p2d, 2);
        } else if (lead) {
            rl_prepare_for(h, (uint32_t)n_embd, 1, e_qkv);
            p2_lead_emit(l, p2e, p2d, 2);
        } else
#endif
        {
            rl_prepare_for(h, (uint32_t)n_embd, 1, e_qkv);
            rl_matvec(e_qkv, DNR_QKV, DNR_QKV);
            if (!zbeside)
                rl_matvec(e_z, Z, Z);
        }
        if (rl_start("deltanet projections"))
            return -1;
        if (rl_wait_desc(k_ba) == 0) {
            uint64_t ta = perf_now64();
            perf_enter(OP_DN_ACT);
            const volatile float *bsp = (const volatile float *)(SCRATCHPAD_BASE + BA);
            const volatile float *asp = (const volatile float *)(SCRATCHPAD_BASE + AL);
            for (int hh = 0; hh < nv; hh++) {
                betab[hh] = sigmoidf(bsp[hh]);
                gb[hh]    = softplusf(asp[hh] + ssm_dt[l][hh])*ssm_a[l][hh];
            }
            dn_stream_records(SLOT0, HO, scale, rep);
            perf_exit(OP_DN_ACT);
            act_dt = perf_now64() - ta;
        }
        // The conv list into the other descriptor bank while qkv and z stream.
        uint64_t tc = perf_now64();
        perf_enter(OP_DN_CONV);
        DN_CONV_LIST();
        perf_exit(OP_DN_CONV);
        conv_dt = perf_now64() - tc;
        if (rl_wait())
            return -1;                 // nothing of the layer's state has moved yet
#ifdef P2_BUILT
        if (lead)
            p2_parked = 1;
#endif
    } else {
        rl_prepare_for(h, (uint32_t)n_embd, 1, e_qkv);
        rl_matvec(e_qkv, DNR_QKV, DNR_QKV);
        rl_matvec(e_z, Z, Z);
        rl_prepare_for(h, (uint32_t)n_embd, 0, model_entry(ROLE_SSM_BETA, l)); // beta/alpha are not folded
        rl_matvec(model_entry(ROLE_SSM_BETA, l), BA, BA);
        rl_matvec(model_entry(ROLE_SSM_ALPHA, l), AL, AL);
        rl_store(betab, dnr_align(4*(uint32_t)nv), BA);
        rl_store(alphab, dnr_align(4*(uint32_t)nv), AL);
        if (!z_stays)
            rl_store(zb, dnr_align(4*(uint32_t)nvh), Z);
        if (rl_run("deltanet projections"))
            return -1;                 // nothing of the layer's state has moved yet
    }
    dn_cyc_act += act_dt;
    dn_cyc_inproj += perf_now64() - t0 - act_dt - conv_dt;
    perf_exit(OP_DN_PROJ);

    // B: causal conv in chunks, then SiLU. Without the early projections the list runs while
    // the CPU computes the per-head scalars and the stream op's records (none of it touches
    // the list's regions or the descriptor slots).
    t0 = perf_now64() - conv_dt;
    perf_enter(OP_DN_CONV);
    if (!early)
        DN_CONV_LIST();
#undef DN_CONV_LIST
    if (rl_start("deltanet conv"))
        goto fatal;
    uint64_t ta = perf_now64();
    perf_enter(OP_DN_ACT);
    if (!early) {
        for (int hh = 0; hh < nv; hh++) {
            betab[hh] = sigmoidf(betab[hh]);
            gb[hh]    = softplusf(alphab[hh] + ssm_dt[l][hh])*ssm_a[l][hh];
        }
#ifdef TRIXIUM_HAS_DNSTREAM
        if (dn_hw && HP->dnstream_active)
            dn_stream_records(SLOT0, HO, scale, rep);
#endif
    }
    perf_exit(OP_DN_ACT);
    act_dt = perf_now64() - ta;
    dn_cyc_act += act_dt;
    // The q/k L2 norm, the heads and the gated norm as one keyed list, built into the other
    // bank while the conv runs (dnlist on, the stream op in the list as an engine descriptor:
    // the same fields as the CSR path's; its key carries the state set and the zero flag, which
    // change wbase/dstride/bias): no list round trip for the L2, no CSR dispatch and poll for
    // the heads, the gated norm's build hidden. The CSR path stays (dnlist off) for its
    // ENGSTALL diagnostics and the CPU recurrence.
    int dn_list = 0, core_posted = 0;
#ifdef TRIXIUM_HAS_DNSTREAM
    dn_list = hybrid_dn_list && dn_hw && HP->dnstream_active;
#endif
    uint64_t tl = perf_now64();
    perf_enter(OP_DN_PREP);
    int core_replay = 0;
    if (dn_list) {
        uint32_t zero = (uint32_t)(dn_state_zero >> s & 1u);
        core_replay = rl_begin_keyed(0x0D000000u | (uint32_t)l << 8 | (uint32_t)z_stays |
                                     (uint32_t)dn_par << 1 | zero << 2, 0, 0);
        if (!core_replay)
            bns_seq_desc_norm(&d, 1, DNR_QKV, DNR_QKV, 0, (uint32_t)hk, 2u*(uint32_t)nk, eps);
        rl_desc(&d);
        rl_fetch_raw(norm_ssm[l], dnr_align(4*(uint32_t)hv), NW);
        if (!z_stays)
            rl_fetch(zb, dnr_align(4*(uint32_t)nvh), Z2);
#ifdef TRIXIUM_HAS_DNSTREAM
        struct engine_op o = {0};
        o.kind  = ENGINE_KIND_DNSTREAM;
        o.rows  = (uint16_t)nv;
        o.wbase = dn_hbm_base + (uint32_t)dn_par*dn_set_bytes + (uint32_t)s*(uint32_t)nv*DN_STREAM_BEATS*32u;
        o.dstride = (dn_par ? -(int32_t)(dn_set_bytes/32u) : (int32_t)(dn_set_bytes/32u)) & 0xFFFFFF;
        o.wlen  = (uint32_t)nv*DN_STREAM_BEATS;
        o.xsrc  = SLOT0;
        o.bias  = (uint8_t)zero;
        rl_engine(&o);
#endif
        if (!core_replay)
            bns_seq_desc_norm(&d, 0, HO, HO, NW, (uint32_t)hv, (uint32_t)nv, eps);
        rl_desc(&d);
        if (!core_replay)
            bns_seq_desc_silu_ex(&d, Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, 0, (uint32_t)hv/8,
                                 (uint32_t)nk, (uint32_t)rep);
        rl_desc(&d);
    } else {
        // The q/k L2 list alone into the other bank while the conv runs.
        if (!rl_begin_keyed(0x06000000u, 0, 0))
            bns_seq_desc_norm(&d, 1, DNR_QKV, DNR_QKV, 0, (uint32_t)hk, 2u*(uint32_t)nk, eps);
        rl_desc(&d);
    }
    perf_exit(OP_DN_PREP);
    uint64_t l2_dt = perf_now64() - tl;
    if (resident_join())
        goto fatal;
    dn_conv_pos[s] = (uint8_t)p1;
    if (trixium_seq_silu_fixups_read())
        resident_silu_patch(DNR_QKV, 0, DNR_QKV, FIX, C, 0, 0, 0);
    dn_cyc_conv += perf_now64() - t0 - act_dt - l2_dt;
    perf_exit(OP_DN_CONV);
    if (dn_list) {
        // The records at SLOT0 were posted writes during the conv list: the read-back fences
        // them (the sequencer drains posted writes before a list too).
        volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + SLOT0);
        (void)rec[8*(nv - 1) + 7];
        t0 = perf_now64() - l2_dt;
        perf_enter(OP_DN_HEADS);
        if (rl_start("deltanet core"))             // posted: the out list is staged under it
            goto fatal;
        core_posted = 1;
    } else {
    perf_enter(OP_DN_PREP);
    t0 = perf_now64() - l2_dt;
    if (rl_run("deltanet q/k l2"))
        goto fatal;
    dn_cyc_prep += perf_now64() - t0;
    perf_exit(OP_DN_PREP);
    // Heads: outputs in natural head order at HO.
    t0 = perf_now64();
    perf_enter(OP_DN_HEADS);
#ifdef TRIXIUM_HAS_DNSTREAM
    if (dn_hw && HP->dnstream_active) {
        // Every head in one engine op, the records written during the conv list; the
        // state streamed from and back to every channel. The read-back fences the CPU's
        // posted scratchpad writes.
        volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + SLOT0);
        (void)rec[8*(nv - 1) + 7];   // the last record written
        struct engine_op o = {0};
        o.kind  = ENGINE_KIND_DNSTREAM;
        o.rows  = (uint16_t)nv;
        o.wbase = dn_hbm_base + (uint32_t)dn_par*dn_set_bytes + (uint32_t)s*(uint32_t)nv*DN_STREAM_BEATS*32u;
        // Written back into the other set (0: in place).
        o.dstride = (dn_par ? -(int32_t)(dn_set_bytes/32u) : (int32_t)(dn_set_bytes/32u)) & 0xFFFFFF;
        o.wlen  = (uint32_t)nv*DN_STREAM_BEATS;
        o.xsrc  = SLOT0;
        o.bias  = (uint8_t)(dn_state_zero >> s & 1u);    // the state is still the reset's zero
        engine_run(&o);
        if (engine_wedged) {
            puts("DNSTALL: DeltaNet stream op");
            goto fatal;
        }
        dn_state_zero &= ~(1ull << s);
    } else
#endif
    if (dn_hw) {
        puts("DNSTALL: the DeltaNet state is in engine DDR and the stream is off (dnstream off?)");
        engine_wedged = 1;
        goto fatal;
    } else {
        // The state in main RAM (a model without the stream's geometry): the CPU
        // recurrence on the normalised q/k/v, then its outputs back into the scratchpad.
        rl_begin();
        rl_store(convb, dnr_align(4*C), DNR_QKV);
        if (rl_run("deltanet qkv out"))
            goto fatal;
        const float *dn_q = convb, *dn_k = convb + nq, *dn_v = convb + 2*nq;
        for (int hh = 0; hh < nv; hh++) {
            int kh = hh % nk;
            const float *qh = dn_q + kh*hk, *kk = dn_k + kh*hk, *vh = dn_v + hh*hv;
            float *M = dn_state + ((uint64_t)s*nv + hh)*hv*hk, *oh = ob + hh*hv;
            float decay = trixium_decay_exp(gb[hh]);
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
    perf_exit(OP_DN_HEADS);

    // C: gated RMSNorm over the heads, x SiLU(z), into the grouped order at RS_F (og).
    t0 = perf_now64();
    perf_enter(OP_DN_GATE);
    int gate_replay = rl_begin_keyed(0x07000000u | (uint32_t)l << 8 | (uint32_t)z_stays, 0, 0);
    rl_fetch_raw(norm_ssm[l], dnr_align(4*(uint32_t)hv), NW);
    if (!z_stays)
        rl_fetch(zb, dnr_align(4*(uint32_t)nvh), Z2);
    if (!gate_replay)
        bns_seq_desc_norm(&d, 0, HO, HO, NW, (uint32_t)hv, (uint32_t)nv, eps);
    rl_desc(&d);
    if (!gate_replay)
        bns_seq_desc_silu_ex(&d, Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, 0, (uint32_t)hv/8,
                             (uint32_t)nk, (uint32_t)rep);
    rl_desc(&d);
    if (rl_run("deltanet gated norm"))
        goto fatal;
    if (trixium_seq_silu_fixups_read())
        resident_silu_patch(Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, (uint32_t)hv, (uint32_t)nk, (uint32_t)rep);
    // og stays in RS_F for the out projection's prepare (never stored to main RAM): a failure
    // past this point invalidates the token anyway - the layer's window and state have moved.
    dn_cyc_gate += perf_now64() - t0;
    perf_exit(OP_DN_GATE);
    }

    // The out projection, the residual add and the FFN norm as one list (out_resid_build):
    // staged while the posted core list runs, started after the gated SiLU's fixup patch.
    uint64_t to = perf_now64();
    struct resident_state *R = resident_state();
    R->prep_src = og; R->prep_width = nvh; R->prep_rotate = 1;
    resident_mark_act();
    int fused = out_resid_build(NULL, (uint32_t)nvh, model_entry(ROLE_SSM_OUT, l), norm_ffn[l],
                                infer_dbg_armed() != 0, OUT_SITE_DN, l) == 0;
    uint64_t tb = perf_now64() - to;
    if (core_posted) {
        if (resident_join())
            goto fatal;
        dn_state_zero &= ~(1ull << s);
        dn_cyc_heads += perf_now64() - t0 - tb;
        perf_exit(OP_DN_HEADS);
        t0 = perf_now64();
        perf_enter(OP_DN_GATE);
        if (trixium_seq_silu_fixups_read())
            resident_silu_patch(Z2, HO, RS_F_ADDR, FIX, (uint32_t)nvh, (uint32_t)hv, (uint32_t)nk, (uint32_t)rep);
        dn_cyc_gate += perf_now64() - t0;
        perf_exit(OP_DN_GATE);
    }
    t0 = perf_now64() - tb;
    perf_enter(OP_DN_OUT);
    if (fused) {
        int rc = out_resid_start();
        if (rc == -2)
            goto fatal;
        if (rc == 0) {
            dn_cyc_outproj += perf_now64() - t0;
            perf_exit(OP_DN_OUT);
            return 1;                              // the residual add and the FFN norm are done
        }
        resident_mark_act();                       // declined: the two-list path below
    }
    if (HP->keep_out)
        resident_keep_rsf(out);
    // The prepare and the out projection in one list, its weights prefetched under the prepare.
    if (resident_prepare_matvec(NULL, (uint32_t)nvh, 1, model_entry(ROLE_SSM_OUT, l), out))
        goto fatal;
    dn_cyc_outproj += perf_now64() - t0;
    perf_exit(OP_DN_OUT);
    return 0;
fatal:
    // A list failed after the layer's window/state started to move: the forward cannot be
    // recomputed on the CPU. The resident path is off (run_list); report and stop.
    puts("hybrid: resident DeltaNet layer failed mid-layer - this token is invalid");
    return -2;
}
#endif

/* Resident Residual Stream */
/*--------------------------*/

#ifdef DNRES_BUILT
// The resident residual stream: x lives in the scratchpad at XRES_SP for the whole forward;
// the norms (unit 12) read it and leave hb in RS_F (and main RAM, for the mixers and the
// traces); the residual adds (unit 13 op 1) add the layer outputs into it. xb (main RAM)
// is only brought up to date for traces and for the LM head.
static int xres_ok(void)
{
    return HP->xres_enabled && resident_active() && n_embd % 8 == 0 && n_embd <= 8192 &&
           XRES_SP0 + 4u*(uint32_t)n_embd <= 0x3C000u &&
           (trixium_seq_norm_abi_read() >> 16) == 1 && (trixium_seq_conv_abi_read() >> 16) == 1;
}

static int xres_list(const float *y, const float *w, int load, int sync)
{
    struct bns_seq_descriptor d;
    uint32_t n = (uint32_t)n_embd, bytes = 4*n;
    int traced_now = infer_dbg_armed() != 0;
    uint32_t wsp = RS_Y_ADDR + ((bytes + 31u) & ~31u);
    rl_begin();
    if (load)
        rl_fetch(xb, bytes, XRES_SP);
    if (y) {
        // The mixer or FFN output: still in RS_F when its list restored it there (no FETCH).
        uint32_t ysp = RS_F_ADDR;
        if (!rl_rsf_is(y)) {
            rl_fetch(y, bytes, RS_Y_ADDR);
            ysp = RS_Y_ADDR;
        }
        bns_seq_desc_add(&d, XRES_SP, ysp, XRES_SP, n);
        rl_desc(&d);
    }
    if (sync)
        rl_store(xb, bytes, XRES_SP);
    if (w) {
        rl_fetch_raw(w, bytes, wsp);                  // a norm weight: cleaned at load
        bns_seq_desc_norm(&d, 0, XRES_SP, RS_F_ADDR, wsp, n, 1, f32_bits(rms_eps));
        rl_desc(&d);
        // hb to main RAM only for the trace (or `hbstore on`): the next lists read it in RS_F,
        // a CPU reader calls hb_materialize() first.
        if (traced_now || hybrid_hb_store) {
            rl_store(hb, bytes, RS_F_ADDR);
            hb_stale = 0;
        } else {
            hb_stale = 1;
            hb_w = w;
        }
    }
    // Untraced and not syncing xb, nothing reads this list's outputs but the next list:
    // posted, so the CPU builds that one while this runs.
    if (!sync && HP->keep_out ? rl_post("residual stream") : rl_run("residual stream"))
        return -1;
    if (w)
        resident_rsf_set(hb);         // the next list's prepare of hb reads RS_F, no FETCH
    return 0;
}
#endif

/* Attention Stream */
/*------------------*/

#ifdef TRIXIUM_HAS_ATTNSTREAM
// Attention stream (engine op 6, gateware/units/attention_stream.py). Scratchpad while the attention
// core runs (the q/k/v list has stored its outputs; RS_Y and RS_F are free until the out
// projection's prepare):
#define AT_Q    RS_Y_ADDR              // q heads (in [q | gate] when gated): <= 48 KB
#define AT_KST  (RS_Y_ADDR + 0xC000u)  // this token's K row (kv_dim floats)
#define AT_VST  (RS_Y_ADDR + 0xD000u)  // this token's V row
#define AT_REC  (RS_Y_ADDR + 0xE000u)  // the scores record, then the values record
#define AT_O    (RS_Y_ADDR + 0xE400u)  // the outputs: n_head*hd floats, <= 24 KB
#define AT_O_END (RS_Y_ADDR + 0x18000u)
#define AT_SP   RS_F_ADDR              // scores, then p: [n_head][T rounded to 8]; then
#define AT_D    RS_F_ADDR              // the gated outputs,
#define AT_FX   (RS_F_ADDR + 0x6000u)  // their fixup maps (32 B per head),
#define AT_IMG  (RS_F_ADDR + 0x8000u)  // and the unit-11 table image (1,025 beats)
#define AT_SP_END 0x32000u

// Appends this token's K/V row to engine DDR (every token while the stream is on), then runs
// the two passes around the firmware's own softmax, and the output gate on unit 11 when it
// can. q: the roped query heads in main RAM, qstride floats apart (2*hd inside [q | gate],
// the gates behind each head). 0: attnb holds the ungated outputs; 1: the gated outputs;
// -1: the CPU loop runs on main RAM's copy (a long context, or a list failure, which also
// turns the stream off until the next reset).
// The attention softmax over `rows` score rows of T positions (Tp floats apart), in place: mx =
// max (a NaN never wins), e = softmax_exp(s - mx), one sequential sum per row from +0, one
// reciprocal per row, then multiplies (ggml scales by 1/sum too; the mirror's order). Unit 14
// does the same in the scratchpad when the bitstream has it (softmaxunit). Four rows per pass
// gained 2% on the card (the exp's branches keep the chains serial): one row at a time.
static void softmax_rows(float *buf, uint32_t rows, uint32_t stride, int T)
{
    for (uint32_t hh = 0; hh < rows; hh++) {
        float *sc = buf + hh*stride, mx = -1e30f, sum = 0;
        for (int t = 0; t < T; t++) if (sc[t] > mx) mx = sc[t];
        for (int t = 0; t < T; t++) { sc[t] = trixium_softmax_exp(sc[t] - mx); sum += sc[t]; }
        // `softmaxdiv on` keeps the old divide per position, for a same-boot comparison only.
        if (hybrid_softmax_div) {
            for (int t = 0; t < T; t++) sc[t] = sc[t]/sum;
        } else {
            float inv = 1.0f/sum;
            for (int t = 0; t < T; t++) sc[t] = sc[t]*inv;
        }
    }
}

static int attn_stream(int a, int T, uint32_t slot, float scale, int gated, const float *q,
                       uint32_t qstride, uint32_t qsp, uint32_t ksp, uint32_t vsp)
{
    // T slots attended (all of them once the ring wrapped), this token's K/V stored at `slot`.
    // qsp/ksp/vsp != 0: q (in [q | gate] at qstride), this token's K and V rows are already in
    // the scratchpad there (the q/k/v list with RoPE on unit 12): no FETCH from main RAM.
    uint32_t QB = qsp ? qsp : AT_Q, KB = ksp ? ksp : AT_KST, VB = vsp ? vsp : AT_VST;
    uint32_t hd = (uint32_t)head_dim, nq = (uint32_t)n_head, nkv = (uint32_t)n_kv_head;
    uint32_t Tp = ((uint32_t)T + 7u) & ~7u, rb = at_rowbeats, rowbytes = rb*32u;
    uint32_t kspan = at_hbm_base + (uint32_t)(2*a)*at_slots*rowbytes;
    uint32_t vspan = kspan + at_slots*rowbytes;
    uint32_t p = slot, nch = (uint32_t)n_ch;
    int fits = (qsp ? QB + nq*qstride*4u <= AT_REC : nq*qstride*4u <= AT_KST - AT_Q) &&
               (ksp ? KB + (uint32_t)kv_dim*4u <= AT_REC && VB + (uint32_t)kv_dim*4u <= AT_REC
                    : (uint32_t)kv_dim*4u <= AT_VST - AT_KST) &&
               nq*hd*4u <= AT_O_END - AT_O && nq*Tp*4u <= AT_SP_END - AT_SP &&
               T <= TRIXIUM_ATTN_T_MAX;
    struct engine_op o;
    uint64_t ts = perf_now64(), tn;
#define AT_SUB(i) do { tn = perf_now64(); HP->at_sub[i] += tn - ts; ts = tn; } while (0)
    // The softmax on unit 14 (its ABI 1 present, the geometry within its bounds, the console
    // toggle on): the scores, the softmax and the values run as one list.
    int usm = 0;
#ifdef CSR_TRIXIUM_SEQ_SOFTMAX_ABI_ADDR
    usm = hybrid_softmax_unit && (trixium_seq_softmax_abi_read() >> 16) == 1 &&
          T <= 4096 && nq <= 256u;
#endif
    rl_begin();
    if (!ksp) {
        rl_fetch(kb, (uint32_t)kv_dim*4u, AT_KST);
        rl_fetch(vb, (uint32_t)kv_dim*4u, AT_VST);
    }
    memset(&o, 0, sizeof o);
    o.kind = ENGINE_KIND_STORE; o.chan = (uint8_t)(p % nch); o.wlen = rb;
    o.wbase = kspan + p/nch*rowbytes; o.xsrc = KB; rl_engine(&o);
    o.wbase = vspan + p/nch*rowbytes; o.xsrc = VB; rl_engine(&o);
    volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + AT_REC);
    // One scores and one values record per kv-head slice (at_sg heads, ns query heads): the
    // scores records at AT_REC, the values records after them.
    uint32_t nsl = nkv/(uint32_t)at_sg, ns = nq/nsl, vrec = AT_REC + 32u*nsl;
    if (fits) {
        for (uint32_t sl = 0; sl < nsl; sl++) {
            volatile uint32_t *rs = rec + 8u*sl, *rv = rec + 8u*(nsl + sl);
            uint32_t w0 = (sl*(uint32_t)at_sg) << 1 | ns << 8 | (nq/nkv) << 16 | nkv << 24;
            rs[0] = w0 | 0;  rs[1] = (uint32_t)T; rs[2] = hd; rs[3] = QB + sl*ns*qstride*4u;
            rs[4] = qstride*4u; rs[5] = AT_SP + sl*ns*Tp*4u; rs[6] = Tp*4u; rs[7] = f32_bits(scale);
            rv[0] = w0 | 1;  rv[1] = (uint32_t)T; rv[2] = hd; rv[3] = AT_SP + sl*ns*Tp*4u;
            rv[4] = Tp*4u;   rv[5] = AT_O + sl*ns*hd*4u; rv[6] = hd*4u; rv[7] = 0;
        }
        if (!qsp)
            rl_fetch(q, nq*qstride*4u, AT_Q);
        memset(&o, 0, sizeof o);
        o.kind = ENGINE_KIND_ATTN; o.rows = (uint16_t)((uint32_t)T/nch); o.rem = (uint8_t)((uint32_t)T % nch);
        o.wbase = kspan; o.wlen = (uint32_t)o.rows*rb; o.cbeats = (uint8_t)rb;
        for (uint32_t sl = 0; sl < nsl; sl++) {
            o.xsrc = AT_REC + 32u*sl;
            rl_engine(&o);
        }
        if (!usm)
            rl_store(at_buf, nq*Tp*4u, AT_SP);
    }
    AT_SUB(1);
    if (!usm) {
        // Posted: the values list is built while the scores run (it fetches the probabilities
        // only after the softmax below has cleaned them to main RAM).
        if (rl_post("attention scores")) {
            HP->attnstream_active = 0;
            return -1;
        }
    }
    if (!fits) {
        if (usm ? rl_run("attention kv") : resident_join())
            HP->attnstream_active = 0;
        return -1;
    }
    // The gate on unit 11 when it can: one run per head over the gates in AT_Q, into AT_D
    // (disjoint: the outputs in AT_O and the gates stay for the fixups).
    int unit_gate = gated && qstride == 2u*hd && resident_gate_active() && hd <= 256u &&
                    nq*hd*4u <= AT_FX - AT_D && nq*32u <= AT_IMG - AT_FX;
    if (usm) {
#ifdef CSR_TRIXIUM_SEQ_SOFTMAX_ABI_ADDR
        // The softmax on unit 14 over the scores in AT_SP, in the same list as the scores and
        // the values: no main-RAM round trip, no CPU between them.
        struct bns_seq_descriptor d;
        if (bns_seq_desc_softmax(&d, AT_SP, (uint32_t)T, nq, Tp/8u)) {
            HP->attnstream_active = 0;
            return -1;
        }
        rl_desc(&d);
#endif
    } else {
        rl_begin();
        rl_fetch_raw(at_buf, nq*Tp*4u, AT_SP);   // cleaned after the softmax
    }
    o.wbase = vspan;
    for (uint32_t sl = 0; sl < nsl; sl++) {
        o.xsrc = vrec + 32u*sl;
        rl_engine(&o);
    }
    if (unit_gate) {
        rl_gate_begin(AT_IMG);
        for (uint32_t hh = 0; hh < nq; hh++)
            rl_gate_seg(QB + (hh*qstride + hd)*4u, AT_O + hh*hd*4u, AT_D + hh*hd*4u,
                        AT_FX + hh*32u, hd);
        rl_gate_end(AT_IMG);
        rl_store(attnb, nq*hd*4u, AT_D);
    } else {
        rl_store(attnb, nq*hd*4u, AT_O);
    }
    AT_SUB(4);
    if (!usm) {
        if (resident_join()) {
            HP->attnstream_active = 0;
            return -1;
        }
        AT_SUB(2);
        // The softmax exactly as the CPU loop computes it (softmax_rows above).
        perf_enter(OP_SOFTMAX);
        softmax_rows(at_buf, nq, Tp, T);
        clean_cpu_dcache_range(at_buf, nq*Tp*4u);
        perf_exit(OP_SOFTMAX);
        AT_SUB(3);
    }
    if (rl_run(usm ? "attention" : unit_gate ? "attention values+gate" : "attention values")) {
        HP->attnstream_active = 0;
        if (unit_gate)
            resident_silu_restore();
        return -1;
    }
#ifdef CSR_TRIXIUM_SEQ_SILU_FIXUPS_ADDR
    if (unit_gate && trixium_seq_silu_fixups_read())
        for (uint32_t hh = 0; hh < nq; hh++)
            resident_gate_patch(AT_FX + hh*32u, QB + (hh*qstride + hd)*4u, AT_O + hh*hd*4u,
                                attnb + hh*hd, hd);
#endif
    AT_SUB(5);
#undef AT_SUB
    return unit_gate ? 1 : 0;
}
#endif

/* Attention Layer */
/*-----------------*/

// Rows [0, T) of attention slot a in kc/vc for the CPU loop, this token's at row `slot` from
// kb/vb, the others back from the stream's copy in engine DDR (row t on channel t % n_ch) from
// kc_rows[a] on. Before the ring wraps slot = T - 1 and rows [kc_rows[a], T - 1) are missing;
// once wrapped any row may have been rewritten by the stream (kc_rows[a] = 0 then): all T come
// back. -1 when a row cannot be brought back (the loop must not run on stale rows).
static int attn_kv_main(int a, int T, uint32_t slot, float *K, float *V)
{
    uint32_t have = kc_rows ? kc_rows[a] : 0, last = slot == (uint32_t)T - 1u ? slot : (uint32_t)T;
    if (kc_rows && have < last) {
#ifdef TRIXIUM_HAS_ATTNSTREAM
        uint32_t rb = at_rowbeats, rowbytes = rb*32u, nch = (uint32_t)n_ch;
        uint32_t kspan = at_hbm_base + (uint32_t)(2*a)*at_slots*rowbytes, vspan = kspan + at_slots*rowbytes;
        uint32_t bytes = (uint32_t)kv_dim*4u, per = 8;          // 8 rows of K and V per list: 64 KB of RS_F
        if (!at_geom_ok || bytes != rowbytes || 2u*per*bytes > 0x32000u - RS_F_ADDR)
            return -1;
        for (uint32_t t0 = have; t0 < last; t0 += per) {
            uint32_t n = last - t0 < per ? last - t0 : per;
            rl_begin();
            for (uint32_t i = 0; i < n; i++) {
                uint32_t t = t0 + i, spk = RS_F_ADDR + 2u*i*bytes, spv = spk + bytes;
                struct engine_op o = {0};
                o.kind = ENGINE_KIND_FETCH; o.chan = (uint8_t)(t % nch); o.wlen = rb;
                o.wbase = kspan + t/nch*rowbytes; o.dst = spk; rl_engine(&o);
                o.wbase = vspan + t/nch*rowbytes; o.dst = spv; rl_engine(&o);
                rl_store(K + (uint64_t)t*kv_dim, bytes, spk);
                rl_store(V + (uint64_t)t*kv_dim, bytes, spv);
            }
            if (rl_run("attention kv restore"))
                return -1;
        }
        printf("hybrid: attention slot %d: K/V rows %lu..%lu restored from engine DDR\n", a,
               (unsigned long)have, (unsigned long)(last - 1u));
        have = last;
#else
        return -1;
#endif
    }
    if (kc_rows && have < last)
        return -1;
    memcpy(K + (uint64_t)slot*kv_dim, kb, (size_t)kv_dim*4);
    memcpy(V + (uint64_t)slot*kv_dim, vb, (size_t)kv_dim*4);
    if (kc_rows) kc_rows[a] = (uint16_t)T;
    return 0;
}

static int attention_layer(int l, const float *h, float *out)
{
    hb_ready();                        // a prepare of hb may FETCH it from main RAM
    int a = attn_slot[l];
    float *K = kc + (uint64_t)a*ctx_cap*kv_dim, *V = vc + (uint64_t)a*ctx_cap*kv_dim;
    int gated = H->attn_gated;
    int e8;
    uint64_t t0 = perf_now64(), t1;
    int normed = 0, rope_hw = 0;
    uint32_t r_q = 0, r_k = 0, r_v = 0, r_qbytes = 0;   // rope_hw: q/k/v resident there
#ifdef DNRES_BUILT
    // Resident: Q/K/V in one list with the per-head q/k RMSNorms on unit 12 (the q heads
    // strided between their gates), everything stored for the CPU's RoPE and scores.
    uint32_t qrows = (uint32_t)(gated ? 2*q_dim : q_dim);
    uint32_t QS = RS_Y_ADDR, KS = QS + 4*qrows, VS = KS + 4*(uint32_t)kv_dim;
    uint32_t NQ = VS + 4*(uint32_t)kv_dim, NK = NQ + 4*(uint32_t)head_dim;
    if (H->hadamard_block && resident_active() && HP->dnres_enabled && head_dim % 8 == 0 &&
        NK + 4u*(uint32_t)head_dim <= RS_F_ADDR && (trixium_seq_norm_abi_read() & 0xffff) >= 1) {
        struct bns_seq_descriptor d;
        uint32_t eps = f32_bits(rms_eps), hb8 = (uint32_t)head_dim/8;
        // With the attention stream and norm ABI 1.2 the RoPE runs on unit 12 too and q, k
        // and v stay in the scratchpad for op 6 (no stores, no CPU RoPE, no fetches back):
        // the cos/sin of this position go behind the k norm weight.
        uint32_t RT = (NK + 4u*(uint32_t)head_dim + 31u) & ~31u;
        int want_rope = 0;
#ifdef TRIXIUM_HAS_ATTNSTREAM
        want_rope = HP->attnstream_active && kc_rows && (trixium_seq_norm_abi_read() & 0xffff) >= 2 &&
                    rope_n % 16 == 0 && rope_n <= 256 && rope_n <= head_dim &&
                    RT + 4u*(uint32_t)rope_n <= RS_F_ADDR && VS + 4u*(uint32_t)kv_dim <= AT_REC;
#endif
        if (want_rope) {
            rope_prepare(pos);
            volatile uint32_t *tab = (volatile uint32_t *)(SCRATCHPAD_BASE + RT);
            for (int i = 0; i < rope_n/2; i++) {
                tab[i] = f32_bits(rope_c[i]);
                tab[rope_n/2 + i] = f32_bits(rope_s[i]);
            }
        }
        // q's norm (with the RoPE when the encoder takes it) is encoded first: its verdict is
        // part of the list's name (the list image cache).
        struct bns_seq_descriptor dq;
        bns_seq_desc_norm_strided(&dq, 0, QS, QS, NQ, (uint32_t)head_dim, (uint32_t)n_head, eps,
                                  gated ? 2*hb8 : hb8, gated ? 2*hb8 : hb8);
        if (want_rope && bns_seq_desc_norm_rope(&dq, (uint32_t)rope_n, RT)) want_rope = 0;
        // The batched prefill: q, k and v of both rows in row 0's list (lead), fetched back in
        // row 1's (the norms, the RoPE and everything after stay per row).
        int lead = 0, follow = 0;
#ifdef P2_BUILT
        const struct bons_entry *p2e[3] = {model_entry(ROLE_ATTN_Q, l), model_entry(ROLE_ATTN_K, l),
                                           model_entry(ROLE_ATTN_V, l)};
        const uint32_t p2d[3] = {QS, KS, VS};
        lead   = p2_mode == P2_LEAD && p2_lead_fits(p2e, 3);
        follow = p2_mode == P2_FOLLOW;
#endif
        int qkv_replay = rl_begin_keyed(0x08000000u | (uint32_t)l << 8 | (uint32_t)gated |
                                        (uint32_t)want_rope << 1 | (uint32_t)rl_rsf_is(h) << 2 |
                                        (uint32_t)lead << 3 | (uint32_t)follow << 4,
                                        (uint32_t)(uintptr_t)h | pf_key(), 0);
        rl_fetch_raw(norm_q[l], 4*(uint32_t)head_dim, NQ);
        rl_fetch_raw(norm_k[l], 4*(uint32_t)head_dim, NK);
#ifdef P2_BUILT
        if (follow) {
            p2_follow_emit(p2e, p2d, 3);
        } else if (lead) {
            rl_prepare_for(h, (uint32_t)n_embd, 1, p2e[0]);
            p2_lead_emit(l, p2e, p2d, 3);
        } else
#endif
        {
            rl_prepare_for(h, (uint32_t)n_embd, 1, model_entry(ROLE_ATTN_Q, l));
            rl_matvec(model_entry(ROLE_ATTN_Q, l), QS, QS);
            rl_matvec(model_entry(ROLE_ATTN_K, l), KS, KS);
            rl_matvec(model_entry(ROLE_ATTN_V, l), VS, VS);
        }
        rl_desc(&dq);
        if (!qkv_replay) {
            bns_seq_desc_norm(&d, 0, KS, KS, NK, (uint32_t)head_dim, (uint32_t)n_kv_head, eps);
            if (want_rope)       // same n and table as q's, which the encoder accepted
                (void)bns_seq_desc_norm_rope(&d, (uint32_t)rope_n, RT);
        }
        rl_desc(&d);
        if (!want_rope) {
            rl_store(gated ? qfull : qb, 4*qrows, QS);
            rl_store(kb, 4*(uint32_t)kv_dim, KS);
            rl_store(vb, 4*(uint32_t)kv_dim, VS);
        }
        normed = rl_run(want_rope ? "attention q/k/v+rope" : "attention q/k/v") == 0;
#ifdef P2_BUILT
        if (normed && lead)
            p2_parked = 1;
#endif
        rope_hw = normed && want_rope;
        if (rope_hw) { r_q = QS; r_k = KS; r_v = VS; r_qbytes = 4*qrows; }
    }
#endif
    if (normed) {
        t1 = perf_now64(); HP->at_qkv += t1 - t0; t0 = t1;
    } else {
        if (H->hadamard_block) {
            // Q/K/V share the input and folded basis: rotate once, not once per projection.
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
    // The stream reads the q heads where they are (inside [q | gate] when gated, the
    // unit's lane stride); the CPU loop wants them contiguous in qb, copied only if it runs.
    int q_in_place = 0;
#ifdef TRIXIUM_HAS_ATTNSTREAM
    q_in_place = gated && HP->attnstream_active;
#endif
    float *qheads = q_in_place ? qfull : qb;
    int qstride = q_in_place ? 2*head_dim : head_dim;
    if (!rope_hw) {                        // else normed and rotated on unit 12, resident
        if (gated && !q_in_place)          // [q | gate] per head
            for (int hh = 0; hh < n_head; hh++)
                memcpy(qb + hh*head_dim, qfull + hh*2*head_dim, (size_t)head_dim*4);
        rope_prepare(pos);
        for (int hh = 0; hh < n_head; hh++) {
            float *qh = qheads + hh*qstride;
            if (!normed)
                rmsnorm_fv(qh, qh, norm_q[l], head_dim);
            rope_apply_partial(qh);
        }
        for (int hh = 0; hh < n_kv_head; hh++) {
            if (!normed)
                rmsnorm_fv(kb + hh*head_dim, kb + hh*head_dim, norm_k[l], head_dim);
            rope_apply_partial(kb + hh*head_dim);
        }
    }
    t1 = perf_now64(); HP->at_rope += t1 - t0; t0 = t1;
    int T = pos < ctx_cap ? pos + 1 : ctx_cap;      // Slots attended (the ring: all of them).
    uint32_t slot = kv_slot(pos);
    int stream_on = 0;
#ifdef TRIXIUM_HAS_ATTNSTREAM
    stream_on = HP->attnstream_active && kc_rows;
#endif
    if (!stream_on && attn_kv_main(a, T, slot, K, V)) {
        puts("hybrid: attention K/V rows could not be restored");
        engine_wedged = 1;
        return 0;
    }
    HP->at_sub[0] += perf_now64() - t0;
    float scale = 1.0f/sqrtf((float)head_dim);
    int streamed = 0;                  // 1: attnb holds the outputs, 2: gated too
#ifdef TRIXIUM_HAS_ATTNSTREAM
    if (HP->attnstream_active) {
        int rc = attn_stream(a, T, slot, scale, gated, qheads, (uint32_t)qstride, r_q, r_k, r_v);
        streamed = rc < 0 ? 0 : 1 + rc;
        // Wrapped, the stream rewrote a row kc may hold: a CPU loop later restores them all.
        if (streamed && pos >= ctx_cap && kc_rows) kc_rows[a] = 0;
    }
#endif
    if (!streamed && rope_hw) {
        // The CPU loop after all: the rotated q, k and v are only in the scratchpad.
        rl_begin();
        rl_store(gated ? qfull : qb, r_qbytes, r_q);
        rl_store(kb, 4*(uint32_t)kv_dim, r_k);
        rl_store(vb, 4*(uint32_t)kv_dim, r_v);
        if (rl_run("attention q/k/v store")) {
            puts("hybrid: attention q/k/v could not be stored for the CPU loop");
            engine_wedged = 1;
            return 0;
        }
    }
    if (!streamed && stream_on && attn_kv_main(a, T, slot, K, V)) {   // the CPU loop after all
        puts("hybrid: attention K/V rows could not be restored");
        engine_wedged = 1;
        return 0;
    }
    if (!streamed && q_in_place)           // the CPU loop after all: its contiguous q
        for (int hh = 0; hh < n_head; hh++)
            memcpy(qb + hh*head_dim, qfull + hh*2*head_dim, (size_t)head_dim*4);
    for (int hh = 0; hh < n_head; hh++) {
        float *oh = attnb + hh*head_dim;
        if (!streamed) {
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
            for (int t = 0; t < T; t++) { scores[t] = trixium_softmax_exp(scores[t] - mx); sum += scores[t]; }
            float inv = 1.0f/sum;      // one divide per head (the stream path's and the mirror's order)
            for (int i = 0; i < head_dim; i++) oh[i] = 0;
            for (int t = 0; t < T; t++) {
                const float *vt = V + (uint64_t)t*kv_dim + kvh*head_dim;
                float p = hybrid_softmax_div ? scores[t]/sum : scores[t]*inv;
                int i = 0;
                for (; i + 3 < head_dim; i += 4) {
                    oh[i + 0] += p*vt[i + 0]; oh[i + 1] += p*vt[i + 1];
                    oh[i + 2] += p*vt[i + 2]; oh[i + 3] += p*vt[i + 3];
                }
                for (; i < head_dim; i++) oh[i] += p*vt[i];
            }
        }
        if (gated && streamed != 2) {
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
    // The out projection (attnb fetched), the residual add and the FFN norm as one list.
    if (out_resid_build(attnb, (uint32_t)n_embd, model_entry(ROLE_ATTN_OUT, l), norm_ffn[l],
                        infer_dbg_armed() != 0, OUT_SITE_ATTN, l) == 0) {
        int rc = out_resid_start();
        if (rc != -1) {
            HP->at_out += perf_now64() - t0;
            return rc == 0 ? 1 : -2;
        }
    }
    if (HP->keep_out)
        resident_keep_rsf(out);
    proj_f(model_entry(ROLE_ATTN_OUT, l), attnb, out);
    HP->at_out += perf_now64() - t0;
    return 0;
}

/* LM Head */
/*---------*/

// LM head chunk: the rows that fit RS_F (restored FP32) and the int32 staging below it, a
// multiple of 8*n_ch; `lmchunk` caps it (the tiny simulation model's head is one chunk
// otherwise, so the cap is how the sim exercises the 27B's multi-chunk path).
static uint32_t lm_chunk_rows(void)
{
    uint32_t unit = 8u*(uint32_t)n_ch, room = (0x32000u - RS_F_ADDR)/4;
    uint32_t chunk = (room < 24576u ? room : 24576u)/unit*unit;
    if (HP->lm_chunk && HP->lm_chunk/unit*unit != 0 && HP->lm_chunk/unit*unit < chunk)
        chunk = HP->lm_chunk/unit*unit;
    return chunk;
}

// The LM head as ONE list: the rotated input once, then row chunks, each stored into dst.
// 0 = done.
static int lm_head_list(const struct bons_entry *head, float *dst)
{
    if (!(H->hadamard_block && resident_active() && head &&
          (head->kind == KIND_TERNARY || head->kind == KIND_TERNARY5 || head->kind == KIND_Q8) && !(head->flags & FLAG_KSEG)))
        return -1;
    uint32_t unit = 8u*(uint32_t)n_ch, chunk = lm_chunk_rows();
    uint32_t nch = (uint32_t)n_ch, per = head->rows/nch;
    if (head->rows % unit || !chunk)
        return -1;
    rl_begin_keyed(0x0B000000u | (chunk/32u & 0x7FFu) << 1 | (uint32_t)rl_rsf_is(hb),
                   (uint32_t)(uintptr_t)head | pf_key(), (uint32_t)(uintptr_t)dst);
    rl_prepare_pf(hb, (uint32_t)n_embd, 1, head, 0, head->rows < chunk ? head->rows : chunk);
    if (!HP->logits_clean) {                  // CPU-written lines must not land over the stores
        flush_cpu_dcache_range(dst, (size_t)head->rows*4u);
        HP->logits_clean = 1;
    }
    for (uint32_t r0 = 0; r0 < head->rows; r0 += chunk) {
        uint32_t rows = head->rows - r0 < chunk ? head->rows - r0 : chunk, rpc = rows/nch;
        rl_matvec_rows(head, r0, rows, RS_Y_ADDR, RS_F_ADDR);
        // Channel-major result: channel c's rpc rows are model rows c*per + r0/nch + j, one
        // store per channel (a packet store's source stride is 8 bits of beats: 8,160 bytes).
        for (uint32_t c = 0; c < nch; c++)
            rl_store_raw(dst + c*per + r0/nch, rpc*4u, RS_F_ADDR + c*rpc*4u);
    }
    int rc = rl_run("lm head");
    if (!rc)
        invd_cpu_dcache_range(dst, (size_t)head->rows*4u);
    return rc;
}

#ifdef CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR
// The greedy argmax of the LM head without the CPU's pass over ~250k logits: one list per
// chunk, the restore unit's tracker read after each. Within a chunk the list order
// (channel-major) is increasing model row, so the tracker's first-occurrence tie-break is
// the firmware loop's; across chunks the smaller model row wins a tie. A chunk the tracker
// cannot answer for (a NaN that is not the model's first logit, padding rows past the
// vocabulary winning) is scanned on the CPU. Same result as the loop in hybrid_forward, bit
// for bit, including its NaN semantics (a NaN first logit freezes it at row 0). Returns the
// row, or -1 when the lists failed (dst is then incomplete).
static int lm_head_argmax(const struct bons_entry *head, float *dst)
{
    if (!(H->hadamard_block && resident_active() && head &&
          (head->kind == KIND_TERNARY || head->kind == KIND_TERNARY5 || head->kind == KIND_Q8) && !(head->flags & FLAG_KSEG)))
        return -1;
    uint32_t unit = 8u*(uint32_t)n_ch, chunk = lm_chunk_rows();
    uint32_t nch = (uint32_t)n_ch, per = head->rows/nch, vocab = (uint32_t)H->vocab;
    if (head->rows % unit || !chunk)
        return -1;
    // The logits reach main RAM only for the trace (or `lmstore on`): the tracker reads the
    // restore's output in RS_F. A chunk the tracker cannot answer for is re-run with its stores
    // before the CPU scans it.
    int store = infer_dbg_armed() || hybrid_lm_store;
    int have = 0, frozen = 0;
    float bv = 0.0f;
    uint32_t brow = 0;
    // The chunks' stores carry no cache maintenance (a flush and an invalidate over ~1 MB
    // per chunk would cost millions of cycles per 27B token): the buffer is flushed once
    // while the CPU has written it, and only what the CPU reads is invalidated first (the
    // rare scan below; the trace, hybrid_forward).
    if (!HP->logits_clean) {
        flush_cpu_dcache_range(dst, (size_t)H->vocab*4 + 64u*nch*4u);
        HP->logits_clean = 1;
    }
    for (uint32_t r0 = 0; r0 < head->rows; r0 += chunk) {
        uint32_t rows = head->rows - r0 < chunk ? head->rows - r0 : chunk, rpc = rows/nch;
        int st = store, reran = 0;
        uint32_t idx, row;
        union { uint32_t u; float f; } v;
    rerun:
        // The list's name: the chunk (its index and size), whether hb is already in RS_F, the
        // stores (bit 1 of the aligned entry pointer).
        rl_begin_keyed(0x0A000000u | (r0/chunk & 0xFFFu) << 12 | (chunk/32u & 0x7FFu) << 1 |
                       (uint32_t)rl_rsf_is(hb), (uint32_t)(uintptr_t)head | pf_key() | (uint32_t)st << 1,
                       (uint32_t)(uintptr_t)dst);
        if (!r0 && !reran)
            rl_prepare_pf(hb, (uint32_t)n_embd, 1, head, 0, rows);
        rl_matvec_rows(head, r0, rows, RS_Y_ADDR, RS_F_ADDR);
        if (st)
            for (uint32_t c = 0; c < nch; c++)    // channel c's rpc rows: model rows c*per + r0/nch + j
                rl_store_raw(dst + c*per + r0/nch, rpc*4u, RS_F_ADDR + c*rpc*4u);
        if (rl_run("lm head"))
            return -1;
        idx = trixium_seq_restore_argmax_idx_read();
        v.u = trixium_seq_restore_argmax_val_read();
        row = idx < rows ? (idx/rpc)*per + r0/nch + idx % rpc : 0xFFFFFFFFu;
        if (frozen)
            continue;
        if (!r0 && row == 0 && v.f != v.f) {       // the model's first logit is NaN
            frozen = 1; have = 1; brow = 0;
            continue;
        }
        float cv = v.f;
        if (row >= vocab || cv != cv) {
            if (!st) {                             // the scan needs the chunk's logits in main RAM
                st = 1; reran = 1;
                goto rerun;
            }
            // Scan this chunk's rows (< vocab) in model order: NaN never wins.
            for (uint32_t c = 0; c < nch; c++)
                invd_cpu_dcache_range(dst + c*per + r0/nch, rpc*4u);
            int chave = 0;
            for (uint32_t c = 0; c < nch; c++)
                for (uint32_t j = 0; j < rpc; j++) {
                    uint32_t r = c*per + r0/nch + j;
                    float x = dst[r];
                    if (r >= vocab || x != x) continue;
                    if (!chave || x > cv) { cv = x; row = r; chave = 1; }
                }
            if (!chave)
                continue;
        }
        if (!have || cv > bv || (cv == bv && row < brow)) { bv = cv; brow = row; have = 1; }
    }
    return have ? (int)brow : 0;
}
#endif

// The LM head on hb (the normed hidden state) and its argmax. t0: the logits phase's start.
static int head_argmax(uint64_t t0)
{
    hb_ready();                        // a prepare of hb may FETCH it from main RAM
    const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
    if (!head) head = model_entry(ROLE_TOK_EMB, 0);
    int best = -1;
#ifdef CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR
    // Several chunks and the hardware argmax on: one list per chunk (lm_head_argmax).
    if (HP->hwargmax && head && head->rows > lm_chunk_rows()) {
        best = lm_head_argmax(head, logits);
        if (best >= 0) {
            if (infer_dbg_armed())
                invd_cpu_dcache_range(logits, (size_t)H->vocab*4);
            infer_dbg_f32("logits", -1, logits, H->vocab);
            hyb_cyc_logits += perf_now64() - t0;
            return best;
        }
    }
#endif
    int head_done = lm_head_list(head, logits) == 0;
#ifdef CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR
    // The restore unit's argmax indexes the list's (channel-major) order: the model's row
    // order only when the head is one chunk (several: lm_head_argmax above, or the loop).
    int one_chunk = head && head->rows <= lm_chunk_rows();
#endif
    if (!head_done) {
        hb_materialize();
        proj_f(head, hb, logits);
        HP->logits_clean = 0;            // CPU-written: lm_head_argmax flushes it first
    }
    infer_dbg_f32("logits", -1, logits, H->vocab);
#ifdef CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR
    // The restore unit tracked the firmware loop's greedy argmax over everything the LM
    // head list restored; rows past the vocabulary (padding) winning means: loop instead.
    if (head_done && HP->hwargmax && one_chunk) {
        uint32_t i = trixium_seq_restore_argmax_idx_read();
        if (i < H->vocab) best = (int)i;
    }
#endif
    if (best < 0) {
        float bv = logits[0];
        best = 0;
        for (uint32_t i = 1; i < H->vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    }
    hyb_cyc_logits += perf_now64() - t0;
    return best;
}

/* FFN */
/*-----*/

// The FFN of hb (ffn-normed, the token's residual in xb) into tmp_h; t0: the fprep start.
// w_next / sync: the residual list that would follow (the next norm's weight, xb synced); with
// the fused down list (out_resid_*) it is this function's. 1 = residual done, -2 = failed.
static int ffn_tok(int l, uint64_t t0, const float *w_next, int sync)
{
    hb_ready();                        // a prepare of hb may FETCH it from main RAM
    int e8 = E8_RESIDENT, fused = 0, pre = 0, fused_down = 0;
    const struct bons_entry *down = model_entry(ROLE_FFN_DOWN, l);
    // Resident FFN: gate, up and act = up*silu(gate) in one scratchpad list, the act
    // then prepared for the down projection where it already is (no FETCH). The gate
    // and up projections reach main RAM only when traced. The activation stays in RS_F
    // (never stored to main RAM): the trace reads it there, and a failed down projection
    // recomputes it on the CPU path. The input's prepare sits at the head of that list,
    // the gate's weights prefetched under it (pre); declined (nothing run), the separate
    // prepare and the list without it follow.
    if (H->hadamard_block && resident_silu_active() && infer_matvec_cols(down) == (uint32_t)ffn_dim) {
        struct resident_state *R = resident_state();
        R->prep_src = hb; R->prep_width = n_embd; R->prep_rotate = 1;
        int traced = infer_dbg_armed();
        perf_enter(OP_FFN_PROJ);
        pre = resident_ffn_act_post(model_entry(ROLE_FFN_GATE, l), model_entry(ROLE_FFN_UP, l), hb,
                                    (uint32_t)n_embd, NULL, traced ? gateb : NULL, traced ? upb : NULL) == 0;
        if (pre) {
            // The down projection, the residual add and the next norm staged under the
            // activation list; started after its fixup patch (below).
            fused_down = out_resid_build(NULL, (uint32_t)ffn_dim, down, w_next, sync, OUT_SITE_FFN, l) == 0;
            if (resident_ffn_act_finish()) {
                pre = 0;                           // the activation list failed: the CPU path
                fused_down = 0;
            }
        }
        perf_exit(OP_FFN_PROJ);
    }
    if (!pre) {
        perf_enter(OP_QUANT8);
        if (H->hadamard_block)
            e8 = rotate_quantize_f(hb, n_embd);
        else
            e8 = quantize_f(hb, n_embd);
        hyb_cyc_fprep += perf_now64() - t0;
        perf_exit(OP_QUANT8);
        t0 = perf_now64();
    }
    perf_enter(OP_FFN_PROJ);
    if (pre) {
        fused = 1;
    } else if (e8 == E8_RESIDENT && H->hadamard_block && resident_silu_active() &&
        infer_matvec_cols(down) == (uint32_t)ffn_dim) {
        int traced = infer_dbg_armed();
        fused = resident_ffn_act(model_entry(ROLE_FFN_GATE, l), model_entry(ROLE_FFN_UP, l), NULL, 0,
                                 NULL, traced ? gateb : NULL, traced ? upb : NULL) == 0;
    }
    if (fused) {
        hyb_cyc_fproj += perf_now64() - t0;
        perf_exit(OP_FFN_PROJ);
        infer_dbg_f32("ffn_gate", l, gateb, ffn_dim);
        infer_dbg_f32("ffn_up", l, upb, ffn_dim);
        infer_dbg_f32("ffn_act", l, (const float *)(SCRATCHPAD_BASE + RS_F_ADDR), ffn_dim);
        t0 = perf_now64();
        perf_enter(OP_FFN_DOWN);
        struct resident_state *R = resident_state();
        R->prep_src = actb; R->prep_width = ffn_dim; R->prep_rotate = 1;
        int down_done = 0;
        uint64_t tf = perf_now64();
        if (fused_down) {
            int rc = out_resid_start();
            if (rc == -2)
                return -2;
            if (rc == 0) {
                HP->fd_mv += perf_now64() - tf;
                hyb_cyc_fdown += perf_now64() - t0;
                perf_exit(OP_FFN_DOWN);
                return 1;                          // the residual add and the next norm are done
            }
            resident_mark_act();                   // declined: the two-list path below
        }
        // The prepare and the down projection in one list, its weights prefetched under the
        // prepare (fd_prep stays 0: no separate prepare).
        if (HP->keep_out)
            resident_keep_rsf(tmp_h);
        down_done = resident_prepare_matvec(NULL, (uint32_t)ffn_dim, 1, down, tmp_h) == 0;
        HP->fd_mv += perf_now64() - tf;
        if (!down_done) {
            int e8c = rotate_quantize_legacy(hb, n_embd);
            matvec_q(model_entry(ROLE_FFN_GATE, l), e8c, gateb);
            matvec_q(model_entry(ROLE_FFN_UP, l), e8c, upb);
            silu_mul_fv(actb, upb, gateb, ffn_dim);
            proj_f(down, actb, tmp_h);
        }
        hyb_cyc_fdown += perf_now64() - t0;
        perf_exit(OP_FFN_DOWN);
    } else {
        matvec_q(model_entry(ROLE_FFN_GATE, l), e8, gateb);
        matvec_q(model_entry(ROLE_FFN_UP, l), e8, upb);
        hyb_cyc_fproj += perf_now64() - t0;
        perf_exit(OP_FFN_PROJ);
        infer_dbg_f32("ffn_gate", l, gateb, ffn_dim);
        infer_dbg_f32("ffn_up", l, upb, ffn_dim);
        t0 = perf_now64();
        perf_enter(OP_SILU_MUL);
        silu_mul_fv(gateb, upb, gateb, ffn_dim);
        hyb_cyc_fact += perf_now64() - t0;
        perf_exit(OP_SILU_MUL);
        infer_dbg_f32("ffn_act", l, gateb, ffn_dim);
        t0 = perf_now64();
        perf_enter(OP_FFN_DOWN);
        proj_f(down, gateb, tmp_h);
        hyb_cyc_fdown += perf_now64() - t0;
        perf_exit(OP_FFN_DOWN);
    }
    return 0;
}

/* Forward */
/*---------*/

/* Forward Checkpoint */
/*--------------------*/

// What a forward changes besides memory it rewrites identically when run again: the
// position, the DeltaNet set parity and zero flags, the conv ring positions, the CPU-side
// K/V row counts. The state matrices are double-buffered (dn_set_bytes), the conv ring has a
// spare slot, the K/V row at the position is rewritten.
struct hyb_ckpt {
    int pos, par;
    uint64_t zero;
    uint8_t conv_pos[64];
    uint16_t kc_rows[64];
};
#define CK (*(struct hyb_ckpt *)TRIXIUM_HYB_CKPT_ADDR)
_Static_assert(sizeof(struct hyb_ckpt) <= 1024, "hybrid checkpoint window is 1 KB");

int hybrid_ckpt_save(void)
{
    if (n_dn > 64 || n_attn > 64)
        return 0;
    if (n_dn && !(dn_hw && dn_set_bytes && HP->dnstream_active && dn_conv_w == conv_k))
        return 0;
    CK.pos  = pos;
    CK.par  = dn_par;
    CK.zero = dn_state_zero;
    if (dn_conv_pos) memcpy(CK.conv_pos, dn_conv_pos, (size_t)n_dn);
    if (kc_rows) memcpy(CK.kc_rows, kc_rows, (size_t)n_attn*2);
    return 1;
}

void hybrid_ckpt_restore(void)
{
    pos           = CK.pos;
    dn_par        = CK.par;
    dn_state_zero = CK.zero;
    if (dn_conv_pos) memcpy(dn_conv_pos, CK.conv_pos, (size_t)n_dn);
    if (kc_rows) memcpy(kc_rows, CK.kc_rows, (size_t)n_attn*2);
}

int hybrid_forward(int token, int want_logits)
{
    if (pos >= ctx_cap && !kv_ring_on()) {
        printf("hybrid: context cap %d reached\n", ctx_cap);
        return -1;
    }
    uint64_t t0 = perf_now64();
    perf_enter(OP_EMBED);
    if (infer_embed_f(token, xb) < 0) return -1;
    hyb_cyc_embed += perf_now64() - t0;
    perf_exit(OP_EMBED);
    infer_dbg_f32("emb", -1, xb, n_embd);
    int xres = 0;
#ifdef DNRES_BUILT
    xres = xres_ok();
#endif
    int traced = infer_dbg_armed();       // the resident residual stream syncs xb when traced
    (void)traced;
    // Untraced, the resident residual stream takes the mixer and FFN outputs where their
    // matvec left them (RS_F, never stored to main RAM), and the residual add closing layer l
    // runs layer l+1's attention norm in the same list.
    HP->keep_out = xres && !traced;
    HP->xres_on = (uint8_t)xres;
    int normed = 0;                       // hb already holds layer l's attention norm
    int out_normed = 0;                   // ... and after the loop the LM head's norm
    (void)normed;
    if (xres && want_logits && !out_norm)
        out_norm = fetch_f32(model_entry_kind(ROLE_OUT_NORM, 0, KIND_F32));
    for (int l = 0; l < n_layer; l++) {
        t0 = perf_now64();
        perf_enter(OP_RMSNORM);
        if (xres) {
#ifdef DNRES_BUILT
            if (!normed && xres_list(NULL, norm_attn[l], l == 0, 0)) return -1;
#endif
        } else {
            rmsnorm_fv(hb, xb, norm_attn[l], n_embd);
            hb_stale = 0;
        }
        hyb_cyc_anorm += perf_now64() - t0;
        perf_exit(OP_RMSNORM);
        infer_dbg_f32("attn_norm", l, hb, n_embd);
        t0 = perf_now64();
        int resid_done = 0;                   // the mixer's list did the residual add + FFN norm
        if (dn_slot[l] >= 0) {
            int rc = -1;
#ifdef DNRES_BUILT
            rc = deltanet_layer_resident(l, hb, tmp_h);
            if (rc == -2) return -1;
            if (rc == 1) { resid_done = 1; rc = 0; }
#endif
            if (rc)
                deltanet_layer(l, hb, tmp_h);
            hyb_cyc_dn += perf_now64() - t0;
        } else {
            perf_enter(OP_ATTN);
            int rc = attention_layer(l, hb, tmp_h);
            if (rc == -2) return -1;
            resid_done = rc == 1;
            hyb_cyc_attn += perf_now64() - t0;
            perf_exit(OP_ATTN);
        }
        infer_dbg_f32(dn_slot[l] >= 0 ? "dn_out" : "attn_out", l, tmp_h, n_embd);
        t0 = perf_now64();
        perf_enter(OP_RESID_ADD);
        if (xres) {
#ifdef DNRES_BUILT
            // Residual add, the FFN norm, and xb for the trace when traced (unless the mixer's
            // own list did them).
            if (!resid_done && xres_list(tmp_h, norm_ffn[l], 0, traced)) return -1;
#endif
            hyb_cyc_resid += perf_now64() - t0;
            perf_exit(OP_RESID_ADD);
            infer_dbg_f32("attn_res", l, xb, n_embd);
            t0 = perf_now64();
        } else {
            add_fv(xb, tmp_h, n_embd);
            hyb_cyc_resid += perf_now64() - t0;
            perf_exit(OP_RESID_ADD);
            infer_dbg_f32("attn_res", l, xb, n_embd);
            t0 = perf_now64();
            rmsnorm_fv(hb, xb, norm_ffn[l], n_embd);
            hb_stale = 0;
        }
        infer_dbg_f32("ffn_norm", l, hb, n_embd);
        // The residual list after the FFN: the last layer's runs the LM head's norm (hb in RS_F
        // for its prepare; the CPU's rmsnorm_fv below was the forward's largest engine-idle gap,
        // ~90k cycles); traced, or without the resident outputs, it brings xb up to date for the
        // CPU. The FFN's own list does it when it can (fdone).
        int last = l + 1 == n_layer, fdone = 0, fsync = 0;
        const float *w_next = NULL;
        if (xres) {
            normed = HP->keep_out && !last;
            out_normed = HP->keep_out && last && want_logits;
            w_next = normed ? norm_attn[l + 1] : out_normed ? out_norm : NULL;
            fsync = traced || (last && !out_normed);
        }
        fdone = ffn_tok(l, t0, w_next, fsync);
        if (fdone == -2) return -1;
        infer_dbg_f32("ffn_down", l, tmp_h, n_embd);
        t0 = perf_now64();
        perf_enter(OP_RESID_ADD);
        if (xres) {
#ifdef DNRES_BUILT
            if (fdone != 1 && xres_list(tmp_h, w_next, 0, fsync))
                return -1;
#endif
        } else {
            add_fv(xb, tmp_h, n_embd);
        }
        hyb_cyc_resid += perf_now64() - t0;
        perf_exit(OP_RESID_ADD);
        infer_dbg_f32("ffn_res", l, xb, n_embd);
    }
    pos++;
    if (dn_set_bytes)
        dn_par ^= 1;                      // every layer's state is now in the other set
    hyb_forwards_profiled++;
    hyb_live_publish();
    if (!want_logits) return -1;
    t0 = perf_now64();
    if (!out_norm)                        // model_entry_kind scans the table: once
        out_norm = fetch_f32(model_entry_kind(ROLE_OUT_NORM, 0, KIND_F32));
    perf_enter(OP_LMHEAD);
    if (!out_normed) {
        rmsnorm_fv(hb, xb, out_norm, n_embd);
        hb_stale = 0;
    }
    infer_dbg_f32("out_norm", -1, hb, n_embd);
    int r = head_argmax(t0);
    perf_exit(OP_LMHEAD);
    return r;
}

/* Batched Prefill */
/*-----------------*/

// Two prompt tokens (positions p, p+1) forwarded layer-major with the multi-row matvec
// (doc/plan-multirow.md, an engine built with nrows >= 2: CSR_TRIXIUM_ENGINE_CSR_NR_ADDR). Per
// layer, each token in turn runs the attention norm, its mixer (DeltaNet or attention: the state
// carried from token to token stays strictly sequential) and the residual add with the FFN norm,
// exactly as hybrid_forward does, on its own residual slot (XRES_SP0 / XRES_SP1); then one FFN
// pass streams the gate, up and down weights once for both: each token's FFN input normed again
// from its slot and prepared with its own exponent, the matvecs with two activation rows
// (rl_matvec2), every row restored with its own exponent beat, SiLU-multiply per row, the down
// projection added into each slot. Each row's arithmetic is the single-token arithmetic, so the
// activations, the K/V rows and the DeltaNet state are bit-identical to two hybrid_forward calls.
// No logits (prefill: the caller forwards the last prompt token alone).
//
// Per row: pos, the DeltaNet set parity (row 0 reads set p and writes p^1, row 1 reads p^1
// and writes p), the residual slot, the trace position (infer_dbg_row) and the list image
// cache's salt (row 1's lists name the second slot; row 0's are the single-token images).
#if defined(DNRES_BUILT) && defined(CSR_TRIXIUM_ENGINE_CSR_NR_ADDR)

#define P2_FIX1 (RS_Q8B_ADDR + 0x800u)   // row 1's SiLU fixup bitmap (row 0's at RS_Q8B)

static int pair_geom(void)
{
    if (HP->pair_geom)
        return HP->pair_geom == 1;
    HP->pair_geom = 2;
    uint32_t n = (uint32_t)n_embd, F = (uint32_t)ffn_dim;
    // Scratchpad: gate rows in RS_F, up rows then the activations in RS_Y, both rows' Q8 in the
    // persistent region, the second residual slot below the sign bits.
    if (n % 32 || F % 32 || F < n || F > 16384u || 4u*4u*F > RS_F_ADDR - RS_Y_ADDR ||
        2u*4u*F > 0x32000u - RS_F_ADDR || 2u*F > RS_E8_ADDR - RS_XW_ADDR || F/32u > 0xFFFu ||
        XRES_SP0 + 4u*n > XRES_SP1 || XRES_SP1 + 4u*n > 0x3C000u)
        return 0;
    for (int l = 0; l < n_layer; l++) {
        const struct bons_entry *e[3] = {model_entry(ROLE_FFN_GATE, l), model_entry(ROLE_FFN_UP, l),
                                         model_entry(ROLE_FFN_DOWN, l)};
        for (int k = 0; k < 3; k++) {
            uint32_t rows = k == 2 ? n : F, cols = k == 2 ? F : n;
            if (!e[k] || (e[k]->flags & FLAG_KSEG) || e[k]->rows != rows || e[k]->cols != cols ||
                (e[k]->kind != KIND_TERNARY && e[k]->kind != KIND_TERNARY5) ||
                rows % 8 || rows % (uint32_t)n_ch)
                return 0;
        }
        if (!norm_ffn[l] || !norm_attn[l])
            return 0;
    }
    HP->pair_geom = 1;
    return 1;
}

static int pair_ok(void)
{
    return HP->prefill2 && xres_ok() && resident_silu_active() && H->hadamard_block && pair_geom();
}

static void pair_row(int r, int p0, int par0)
{
    if (p2_dual) {                          // two sequences: row r is sequence r, at its own state
        seq_swap(r);
    } else {
        pos = p0 + r;
        dn_par = dn_set_bytes ? par0 ^ r : par0;
    }
    HP->xres_row = (uint8_t)r;
    resident_listcache_salt(seq_salt(r));
    infer_dbg_row(r);
}

// The current row's layer l up to the FFN (hybrid_forward's loop body on the resident residual
// stream, the attention norm never fused into the previous FFN's list).
static int pair_seg(int l)
{
    int traced = infer_dbg_armed();
    HP->keep_out = !traced;
    uint64_t t0 = perf_now64();
    perf_enter(OP_RMSNORM);
    int rc = xres_list(NULL, norm_attn[l], 0, 0);     // both slots loaded (hybrid_forward2)
    hyb_cyc_anorm += perf_now64() - t0;
    perf_exit(OP_RMSNORM);
    if (rc) return -1;
    infer_dbg_f32("attn_norm", l, hb, n_embd);
    // Row 0's mixer projects row 1 too (lead), row 1's fetches its outputs back (follow).
    if (HP->xres_row == 0) {
        p2_parked = 0;
        p2_mode = HP->pair_inproj ? P2_LEAD : 0;
    } else {
        p2_mode = p2_parked ? P2_FOLLOW : 0;
    }
    t0 = perf_now64();
    int resid_done = 0;
    if (dn_slot[l] >= 0) {
        rc = deltanet_layer_resident(l, hb, tmp_h);
        if (rc == -2) return -1;
        if (rc == 1) { resid_done = 1; rc = 0; }
        if (rc)
            deltanet_layer(l, hb, tmp_h);
        hyb_cyc_dn += perf_now64() - t0;
    } else {
        perf_enter(OP_ATTN);
        rc = attention_layer(l, hb, tmp_h);
        hyb_cyc_attn += perf_now64() - t0;
        perf_exit(OP_ATTN);
        if (rc == -2) return -1;
        resid_done = rc == 1;
    }
    p2_mode = 0;
    infer_dbg_f32(dn_slot[l] >= 0 ? "dn_out" : "attn_out", l, tmp_h, n_embd);
    t0 = perf_now64();
    perf_enter(OP_RESID_ADD);
    rc = !resid_done && xres_list(tmp_h, norm_ffn[l], 0, traced);
    hyb_cyc_resid += perf_now64() - t0;
    perf_exit(OP_RESID_ADD);
    if (rc) return -1;
    infer_dbg_f32("attn_res", l, xb, n_embd);
    infer_dbg_f32("ffn_norm", l, hb, n_embd);
    return 0;
}

// Both rows' FFN of layer l, added into their residual slots: two lists (gate/up/SiLU, then
// down/residual), the unit 11 fixups patched between them. The trace reads its taps from the
// scratchpad, so the lists are the same traced or not.
static int ffn_pair(int l, int p0, int par0)
{
    const struct bons_entry *gate = model_entry(ROLE_FFN_GATE, l), *up = model_entry(ROLE_FFN_UP, l);
    const struct bons_entry *down = model_entry(ROLE_FFN_DOWN, l);
    const uint32_t n = (uint32_t)n_embd, F = (uint32_t)ffn_dim, fq = 4u*F, nq = 4u*n;
    const uint32_t ACT = RS_Y_ADDR + 2u*fq;               // the activations, above the up rows
    const uint32_t slot[2] = {XRES_SP0, XRES_SP1}, e8[2] = {RS_E8_ADDR, RS_E8B_ADDR};
    const uint32_t fix[2] = {RS_Q8B_ADDR, P2_FIX1};
    const uint32_t gdst[2] = {RS_F_ADDR, RS_F_ADDR + fq}, udst[2] = {RS_Y_ADDR, RS_Y_ADDR + fq};
    const uint32_t adst[2] = {ACT, ACT + fq}, ddst[2] = {RS_F_ADDR, RS_F_ADDR + nq};
    struct bns_seq_descriptor d;
#define SPF(a) ((const float *)(SCRATCHPAD_BASE + (a)))
    resident_listcache_salt(0);
    uint64_t t0 = perf_now64();
    perf_enter(OP_FFN_PROJ);
    // A: both rows' FFN norm (from the slots: the segments left only the second row's in RS_F)
    // and prepare, gate and up for two rows (gate restored into RS_F, up in place), SiLU-multiply.
    rl_begin_keyed(0x0F000000u | (uint32_t)l << 8, (uint32_t)(uintptr_t)gate, (uint32_t)(uintptr_t)up);
    rl_fetch_raw(norm_ffn[l], nq, RS_Y_ADDR);             // the norm weight (cleaned at load)
    for (int r = 0; r < 2; r++) {
        bns_seq_desc_norm(&d, 0, slot[r], RS_F_ADDR, RS_Y_ADDR, n, 1, f32_bits(rms_eps));
        rl_desc(&d);
        rl_prepare_sp(RS_F_ADDR, n, 1, RS_XW_ADDR + (uint32_t)r*n, e8[r]);
    }
    rl_matvec2(gate, RS_XW_ADDR, n, e8, RS_Y_ADDR, gdst);
    rl_matvec2(up,   RS_XW_ADDR, n, e8, RS_Y_ADDR, udst);
    for (int r = 0; r < 2; r++) {
        bns_seq_desc_silu(&d, gdst[r], udst[r], adst[r], fix[r], F);
        rl_desc(&d);
    }
    // A runs while the CPU builds B (the other descriptor bank); the fixups are patched between
    // the two lists, B then posted (the next layer's first list waits for it) unless traced.
    int rc = rl_start("ffn2 activation");
    if (rc) {
        perf_exit(OP_FFN_PROJ);
        return -1;
    }
    int traced = 0;
    for (int r = 0; r < 2; r++) {
        pair_row(r, p0, par0);
        traced |= infer_dbg_armed() != 0;
    }
    resident_listcache_salt(0);
    // B: both rows' activations prepared, down for two rows (restored into RS_F), each added
    // into its slot.
    rl_begin_keyed(0x0F000001u | (uint32_t)l << 8, (uint32_t)(uintptr_t)down, 0);
    for (int r = 0; r < 2; r++)
        rl_prepare_sp(adst[r], F, 1, RS_XW_ADDR + (uint32_t)r*F, e8[r]);
    rl_matvec2(down, RS_XW_ADDR, F, e8, RS_Y_ADDR, ddst);
    for (int r = 0; r < 2; r++) {
        bns_seq_desc_add(&d, slot[r], ddst[r], slot[r], n);
        rl_desc(&d);
    }
    rc = rl_wait();
    if (!rc) {
        int fixups = 1;
#ifdef CSR_TRIXIUM_SEQ_SILU_FIXUPS_ADDR
        fixups = trixium_seq_silu_fixups_read() != 0;
#endif
        for (int r = 0; r < 2 && fixups; r++)
            resident_silu_patch(gdst[r], udst[r], adst[r], fix[r], F, 0, 0, 0);
    }
    hyb_cyc_fproj += perf_now64() - t0;
    perf_exit(OP_FFN_PROJ);
    if (rc) return -1;
    for (int r = 0; r < 2; r++) {          // the trace's taps (nothing when not armed)
        pair_row(r, p0, par0);
        infer_dbg_f32("ffn_gate", l, SPF(gdst[r]), F);
        infer_dbg_f32("ffn_up", l, SPF(udst[r]), F);
        infer_dbg_f32("ffn_act", l, SPF(adst[r]), F);
    }
    resident_listcache_salt(0);
    t0 = perf_now64();
    perf_enter(OP_FFN_DOWN);
    rc = rl_start("ffn2 down");
    if (!rc && traced)
        rc = rl_wait();
    hyb_cyc_fdown += perf_now64() - t0;
    perf_exit(OP_FFN_DOWN);
    if (rc) return -1;
    hb_stale = 0;                           // the next layer's segments norm again
    for (int r = 0; r < 2; r++) {
        pair_row(r, p0, par0);
        infer_dbg_f32("ffn_down", l, SPF(ddst[r]), n);
        infer_dbg_f32("ffn_res", l, SPF(slot[r]), n);
    }
#undef SPF
    return 0;
}
#endif

int hybrid_forward2_ok(void)
{
#if defined(DNRES_BUILT) && defined(CSR_TRIXIUM_ENGINE_CSR_NR_ADDR)
    return H && pair_ok() && (pos + 2 <= ctx_cap || kv_ring_on());
#else
    return 0;
#endif
}

void hybrid_prefill2_enable(int on)
{
    HP->prefill2 = on ? 1 : 0;
    HP->pair_inproj = on == 1;           // 2: the FFN only
}
int  hybrid_prefill2_enabled(void) { return HP->prefill2 ? (HP->pair_inproj ? 1 : 2) : 0; }

int hybrid_forward2(const int *tokens)
{
#if defined(DNRES_BUILT) && defined(CSR_TRIXIUM_ENGINE_CSR_NR_ADDR)
    if (!hybrid_forward2_ok())
        return -2;                          // nothing ran: the caller forwards one at a time
    int p0 = pos, par0 = dn_par, rc = 0;
    HP->xres_on = 1;
    // Both embeddings into their residual slots first: row 0's first mixer reads row 1's slot
    // (the batched in-projections).
    for (int r = 0; r < 2 && !rc; r++) {
        pair_row(r, p0, par0);
        // The embedding uses the engine and xb: the previous row's lists done first.
        resident_join();
        uint64_t t0 = perf_now64();
        perf_enter(OP_EMBED);
        rc = infer_embed_f(tokens[r], xb) < 0;
        if (!rc) {
            rl_begin();
            rl_fetch(xb, 4u*(uint32_t)n_embd, XRES_SP);
            rc = rl_run("residual load") != 0;
        }
        hyb_cyc_embed += perf_now64() - t0;
        perf_exit(OP_EMBED);
        if (!rc)
            infer_dbg_f32("emb", -1, xb, n_embd);
    }
    for (int l = 0; l < n_layer && !rc; l++) {
        for (int r = 0; r < 2 && !rc; r++) {
            pair_row(r, p0, par0);
            rc = pair_seg(l);
        }
        if (!rc)
            rc = ffn_pair(l, p0, par0);
    }
    resident_join();
    p2_mode = p2_parked = 0;
    HP->xres_row = 0;
    HP->keep_out = 0;
    resident_listcache_salt(seq_salt(0));
    pos = p0 + 2;
    dn_par = par0;                          // flipped once per token
    hyb_forwards_profiled += 2;
    hyb_live_publish();
    if (rc) {
        printf("hybrid: batched prefill failed at position %d\n", p0);
        return -1;
    }
    return 0;
#else
    (void)tokens;
    return -2;
#endif
}

/* Two Conversations */
/*-------------------*/

// Two independent sequences decoded together on the multi-row engine (doc/plan-multirow.md,
// "small batches"): one layer-major pass per step produces the next token of each. Per layer,
// each sequence in turn runs its attention norm, its mixer and the residual add with the FFN
// norm on its own state (seq_swap: position, DeltaNet state set and parity, conv window, K/V
// spans and ring) and its own residual slot (XRES_SP0 / XRES_SP1); the weight streams the
// batched prefill shares (DeltaNet qkv/z, attention q/k/v, FFN gate/up/down) run once for both
// rows with rl_matvec2, each row restored with its own exponent. Then each sequence's output
// norm and LM head. Each row's arithmetic is the single-token arithmetic, so both token streams
// and every checkpoint are bit-identical to decoding each sequence alone.
#ifdef P2_BUILT
static void seq_save(struct hyb_seq *q)
{
    q->pos = pos; q->dn_par = dn_par; q->kv_pin = kv_pin; q->kv_pin_next = kv_pin_next;
    q->dn_hw_written = dn_hw_written; q->dn_state_zero = dn_state_zero;
    q->dn_hbm_base = dn_hbm_base;
#ifdef TRIXIUM_HAS_ATTNSTREAM
    q->at_hbm_base = at_hbm_base;
#endif
    q->dn_state = dn_state; q->dn_conv = dn_conv; q->kc = kc; q->vc = vc;
    q->dn_conv_pos = dn_conv_pos; q->kc_rows = kc_rows;
}

static void seq_load(const struct hyb_seq *q)
{
    pos = q->pos; dn_par = q->dn_par; kv_pin = q->kv_pin; kv_pin_next = q->kv_pin_next;
    dn_hw_written = q->dn_hw_written; dn_state_zero = q->dn_state_zero;
    dn_hbm_base = q->dn_hbm_base;
#ifdef TRIXIUM_HAS_ATTNSTREAM
    at_hbm_base = q->at_hbm_base;
#endif
    dn_state = q->dn_state; dn_conv = q->dn_conv; kc = q->kc; vc = q->vc;
    dn_conv_pos = q->dn_conv_pos; kc_rows = q->kc_rows;
}

static void seq_swap(int s)
{
    if (s == hseq_cur || (s && hseq_ready <= 0))
        return;
    seq_save(&hseq[hseq_cur]);
    seq_load(&hseq[s]);
    hseq_cur = s;
    resident_listcache_salt(seq_salt(0));   // a single-token forward of this sequence
}
#endif

int hybrid_dual_setup(void)
{
#ifdef P2_BUILT
    if (hseq_ready)
        return hseq_ready > 0 ? 0 : -1;
    if (!H || !pair_ok()) {
        printf("hybrid: dual decode needs the resident lists and the batched FFN geometry\n");
        return -1;                          // not latched: `prefill2 on` may come later
    }
    hseq_ready = -1;
    seq_swap(0);
    seq_save(&hseq[0]);
    struct hyb_seq *b = &hseq[1];
    *b = hseq[0];
    // Engine DDR: a second DeltaNet state span (both sets) and a second pair of K/V spans per
    // attention layer, above everything the first sequence uses, inside the channel window.
    uint64_t end = 0, dn_span = 0, at_span = 0, limit = 0;
#ifdef DN_STATE_HW
    if (dn_hw) {
        dn_span = ((uint64_t)n_dn*nv + n_ch - 1u)/n_ch*DN_HEAD_BYTES*(dn_set_bytes ? 2u : 1u);
        end = (uint64_t)dn_hbm_base + dn_span;
    }
#endif
#ifdef TRIXIUM_HAS_ATTNSTREAM
    if (at_geom_ok) {
        at_span = (uint64_t)n_attn*2u*at_slots*at_rowbeats*32u;
        if ((uint64_t)at_hbm_base + at_span > end)
            end = (uint64_t)at_hbm_base + at_span;
    }
#endif
    if (end) {
        uint64_t base = (end + 0xFFFu) & ~0xFFFull;
#if defined(TRIXIUM_HAS_ATTNSTREAM) || defined(DN_STATE_HW)
        limit = chan_limit();
#endif
        if (dn_span) {
            b->dn_hbm_base = (uint32_t)base;
            base = (base + dn_span + 0xFFFu) & ~0xFFFull;
        }
        if (at_span) {
            b->at_hbm_base = (uint32_t)base;
            base += at_span;
        }
        if (base > limit) {
            printf("hybrid: dual decode: the second state set reaches %08lx, past the channel "
                   "window's end %08lx\n", (unsigned long)base, (unsigned long)limit);
            return -1;
        }
    }
    // main RAM: the conv window, the CPU path's K/V copy (and the state matrices without the
    // stream), at the first sequence's context capacity.
    uint64_t matrix_b = (uint64_t)n_dn*nv*hv*hk*4;
    uint64_t conv_b = (uint64_t)n_dn*dn_conv_w*(2*nq + nvh)*4;
    uint64_t kv_b = (uint64_t)n_attn*ctx_cap*kv_dim*4*2;
    uint64_t need = (dn_hw ? 0 : matrix_b) + conv_b + kv_b + 4096u, room = infer_heap_room(1);
    if (need > room) {
        printf("hybrid: dual decode: the second sequence needs %lu KB of main RAM, %lu KB free\n",
               (unsigned long)(need >> 10), (unsigned long)(room >> 10));
        return -1;
    }
    b->dn_state = (n_dn && !dn_hw) ? infer_halloc(matrix_b, 1) : NULL;
    b->dn_conv = n_dn ? infer_halloc(conv_b, 1) : NULL;
    b->dn_conv_pos = n_dn ? infer_halloc(n_dn, 0) : NULL;
    b->kc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    b->vc = n_attn ? infer_halloc(kv_b/2, 1) : NULL;
    b->kc_rows = n_attn ? infer_halloc(((uint64_t)n_attn*2 + 31) & ~31ull, 0) : NULL;
    if ((n_dn && (!b->dn_conv || !b->dn_conv_pos || (!dn_hw && !b->dn_state))) ||
        (n_attn && (!b->kc || !b->vc || !b->kc_rows))) {
        printf("hybrid: dual decode: no heap for the second sequence\n");
        return -1;
    }
    b->pos = 0;
    b->dn_par = 0;
    b->dn_state_zero = 0;
    b->kv_pin = b->kv_pin_next = KV_SINKS;
#ifdef TRIXIUM_SIM
    // Simulation: the second span was never written (the memory model's zeros, no ECC); the
    // board clears it once at the sequence's first reset (never-written DDR has no valid ECC).
    b->dn_hw_written = 1;
#else
    b->dn_hw_written = 0;
#endif
    if (b->dn_conv_pos) memset(b->dn_conv_pos, 0, (size_t)n_dn);
    if (b->kc_rows) memset(b->kc_rows, 0, (size_t)n_attn*2);
    hseq_ready = 1;
    printf("hybrid: dual decode: second sequence: engine DDR DeltaNet %lu KB at %08lx, K/V %lu KB "
           "at %08lx (window end %08lx); main RAM %lu KB (ctx %d)\n",
           (unsigned long)(dn_span >> 10), (unsigned long)b->dn_hbm_base,
           (unsigned long)(at_span >> 10), (unsigned long)b->at_hbm_base, (unsigned long)limit,
           (unsigned long)(need >> 10), ctx_cap);
    return 0;
#else
    return -1;
#endif
}

void hybrid_lmhead2_enable(int on) { HP->lmhead2 = on ? 1 : 0; }
int  hybrid_lmhead2_enabled(void) { return HP->lmhead2; }

void hybrid_seq_select(int s)
{
#ifdef P2_BUILT
    seq_swap(s);
#else
    (void)s;
#endif
}

int hybrid_seq_pos(int s)
{
#ifdef P2_BUILT
    if (s != hseq_cur && hseq_ready > 0)
        return hseq[s].pos;
#endif
    (void)s;
    return pos;
}

#if defined(P2_BUILT) && defined(CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR)
// Both sequences' LM heads in one weight pass per chunk: the first chunk's list norms each
// residual slot with the output norm and prepares it into its Q8 slot; each chunk streams the
// head once for both rows (rl_matvec2_rows, int32 staged from RS_Y up), its list restores row 0
// (in place) and the restore unit's tracker gives row 0's argmax; a second list restores row 1
// for its tracker. Same tie-breaks as lm_head_argmax (chunk order is model row order within a
// chunk, the smaller row across chunks). A chunk the tracker cannot answer for (a NaN, padding
// past the vocabulary) leaves best[r] = -1: the caller runs that row's single LM head.
static int lm_head_argmax2(const struct bons_entry *head, int best[2])
{
    if (!(H->hadamard_block && resident_active() && head && out_norm &&
          (head->kind == KIND_TERNARY || head->kind == KIND_TERNARY5) && !(head->flags & FLAG_KSEG)))
        return -1;
    uint32_t nch = (uint32_t)n_ch, unit = 8u*nch, per = head->rows/nch, vocab = (uint32_t)H->vocab;
    uint32_t n = (uint32_t)n_embd, chunk = lm_chunk_rows(), cap = (0x32000u - RS_Y_ADDR)/8u/unit*unit;
    if (chunk > cap)
        chunk = cap;
    if (head->rows % unit || !chunk || n % 32 || 4u*n > RS_F_ADDR - RS_Y_ADDR)
        return -1;
    const uint32_t slot[2] = {XRES_SP0, XRES_SP1}, e8[2] = {RS_E8_ADDR, RS_E8B_ADDR};
    int have[2] = {0, 0}, bad[2] = {0, 0};
    float bv[2] = {0.0f, 0.0f};
    uint32_t brow[2] = {0, 0};
    struct bns_seq_descriptor d;
    for (uint32_t r0 = 0; r0 < head->rows; r0 += chunk) {
        uint32_t rows = head->rows - r0 < chunk ? head->rows - r0 : chunk, rpc = rows/nch;
        const uint32_t ys[2] = {RS_Y_ADDR, RS_Y_ADDR + 4u*rows};
        for (int r = 0; r < 2; r++) {
            rl_begin();
            if (!r0 && !r) {
                rl_fetch_raw(out_norm, 4u*n, RS_Y_ADDR);   // fetched by the engine: clean
                for (int k = 0; k < 2; k++) {
                    bns_seq_desc_norm(&d, 0, slot[k], RS_F_ADDR, RS_Y_ADDR, n, 1, f32_bits(rms_eps));
                    rl_desc(&d);
                    rl_prepare_sp(RS_F_ADDR, n, 1, RS_XW_ADDR + (uint32_t)k*n, e8[k]);
                }
            }
            if (!r)
                rl_matvec2_rows(head, r0, rows, RS_XW_ADDR, n, e8, RS_Y_ADDR, ys, 1u);
            else
                rl_restore_rows(head, ys[1], ys[1], e8[1], rows);
            if (rl_run("lm head2"))
                return -1;
            uint32_t idx = trixium_seq_restore_argmax_idx_read();
            union { uint32_t u; float f; } v;
            v.u = trixium_seq_restore_argmax_val_read();
            uint32_t row = idx < rows ? (idx/rpc)*per + r0/nch + idx % rpc : 0xFFFFFFFFu;
            if (bad[r])
                continue;
            if (row >= vocab || v.f != v.f) {
                bad[r] = 1;
                continue;
            }
            if (!have[r] || v.f > bv[r] || (v.f == bv[r] && row < brow[r])) {
                bv[r] = v.f;
                brow[r] = row;
                have[r] = 1;
            }
        }
    }
    for (int r = 0; r < 2; r++)
        best[r] = bad[r] ? -1 : have[r] ? (int)brow[r] : 0;
    return 0;
}
#endif

int hybrid_forward_dual(const int *tokens, int *out)
{
#ifdef P2_BUILT
    if (hseq_ready <= 0 || !pair_ok())
        return -2;
    for (int s = 0; s < 2; s++) {
        seq_swap(s);
        if (pos >= ctx_cap && !kv_ring_on()) {
            seq_swap(0);
            return -2;
        }
    }
    int rc = 0;
    out[0] = out[1] = -1;
    resident_join();
    if (!out_norm)                          // model_entry_kind scans the table: once
        out_norm = fetch_f32(model_entry_kind(ROLE_OUT_NORM, 0, KIND_F32));
    p2_dual = 1;
    HP->xres_on = 1;
    // Both embeddings into their residual slots first (row 0's mixers read row 1's slot).
    for (int r = 0; r < 2 && !rc; r++) {
        pair_row(r, 0, 0);
        resident_join();
        uint64_t t0 = perf_now64();
        perf_enter(OP_EMBED);
        rc = infer_embed_f(tokens[r], xb) < 0;
        if (!rc) {
            rl_begin();
            rl_fetch(xb, 4u*(uint32_t)n_embd, XRES_SP);
            rc = rl_run("residual load") != 0;
        }
        hyb_cyc_embed += perf_now64() - t0;
        perf_exit(OP_EMBED);
        if (!rc)
            infer_dbg_f32("emb", -1, xb, n_embd);
    }
    for (int l = 0; l < n_layer && !rc; l++) {
        for (int r = 0; r < 2 && !rc; r++) {
            pair_row(r, 0, 0);
            rc = pair_seg(l);
        }
        if (!rc)
            rc = ffn_pair(l, 0, 0);
    }
    hyb_forwards_profiled += 2;
    hyb_live_publish();
    // Both LM heads in one weight pass (untraced: the trace's logits and out_norm taps are the
    // single head's), else, or for a row it could not answer, each sequence's output norm (from
    // its slot) and LM head.
    int traced2 = 0;
    for (int r = 0; r < 2; r++) {
        pair_row(r, 0, 0);
        traced2 |= infer_dbg_armed() != 0;
    }
#ifdef CSR_TRIXIUM_SEQ_RESTORE_ARGMAX_IDX_ADDR
    if (!rc && !traced2 && HP->hwargmax && HP->lmhead2) {
        const struct bons_entry *head = model_entry(ROLE_LM_HEAD, 0);
        if (!head) head = model_entry(ROLE_TOK_EMB, 0);
        uint64_t t0 = perf_now64();
        perf_enter(OP_LMHEAD);
        resident_listcache_salt(0);
        int best[2];
        if (resident_join() == 0 && lm_head_argmax2(head, best) == 0) {
            out[0] = best[0];
            out[1] = best[1];
        }
        hyb_cyc_logits += perf_now64() - t0;
        perf_exit(OP_LMHEAD);
    }
#endif
    for (int r = 0; r < 2 && !rc; r++) {
        if (out[r] >= 0)
            continue;
        pair_row(r, 0, 0);
        int traced = infer_dbg_armed() != 0;
        HP->keep_out = 0;
        uint64_t t0 = perf_now64();
        perf_enter(OP_LMHEAD);
        rc = xres_list(NULL, out_norm, 0, traced) != 0;
        if (!rc) {
            infer_dbg_f32("out_norm", -1, hb, n_embd);
            out[r] = head_argmax(t0);
        }
        perf_exit(OP_LMHEAD);
    }
    resident_join();
    p2_dual = 0;
    p2_mode = p2_parked = 0;
    HP->xres_row = 0;
    HP->keep_out = 0;
    for (int s = 1; s >= 0; s--) {          // every layer's state is now in the other set
        seq_swap(s);
        pos++;
        if (dn_set_bytes)
            dn_par ^= 1;
    }
    resident_listcache_salt(seq_salt(0));
    if (rc) {
        puts("hybrid: dual decode failed - these tokens are invalid");
        return -1;
    }
    return 0;
#else
    (void)tokens; (void)out;
    return -2;
#endif
}

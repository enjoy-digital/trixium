/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium Qwen3 token loop
 *
 * The engine does every matvec (ternary weights streamed from the weight memory); the vector
 * unit and the descriptor sequencer run the vector ops and attention on resident
 * block-floating-point activations in the scratchpad. Every op keeps a CPU reference
 * implementation (hwops). This file holds the model state, the KV cache, the forward, init
 * and generation; the ops are in infer_bfp.c, the sequencer lists in infer_lists.c and the
 * float matvec primitives the hybrid models' forward (hybrid.c) runs on in infer_float.c.
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
#include "infer_internal.h"

/* Code Layout */
/*-------------*/

// A call to a cold function is the one thing GCC's partitioner treats as "never executed"
// (__builtin_expect is only 90%): placed first in a branch that is dead on the card (the
// op-by-op paths behind the sequencer lists), it moves that whole branch to .text.unlikely.
static COLD void cold_path(void) { __asm__ volatile("" ::: "memory"); }

// newlib-libm compatibility: the toolchain libm.a references __errno, absent from the
// picolibc environment LiteX links against.
int *__errno(void);
static int __errno_storage;
int *__errno(void) { return &__errno_storage; }

/* Heap */
/*------*/

// The high heap starts above the highest weights+KV tail any engine channel puts inside
// main_ram (channel 3, PC 4): computed per image at infer_init, 1 MB aligned (the 8B alone
// has 136 MB/channel of weights).
static uint32_t heap_hi_base;

uint8_t *heap;
void *halloc(uint32_t n)
{
    void *p = heap;
    heap = heap + ((n + 63) & ~63u);
    return p;
}

uint8_t *heap_hi;
void *halloc_hi(uint32_t n)
{
    if (heap_hi == 0)          // Small main_ram (the sim's 8 MB): no channel-3 overlap to
        return halloc(n);      // dodge there, the low heap serves everything.
    void *p = heap_hi;
    heap_hi = heap_hi + ((n + 63) & ~63u);
    return p;
}

/* Model State */
/*-------------*/

const struct bons_header *H;
int n_ch, head_dim, n_head, n_kv_head, n_embd, ffn_dim, n_layer;
static int max_ctx;
static int cpu_kv_ctx; // CPU-side KV cache depth (capped at CPU_KV_CTX, halved to fit the heap).

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

float fp16_to_float(uint16_t b)
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
COLD int quantize_f_pow2(const float *src, int n, uint32_t sp_off)
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
uint32_t attn_delay = 0; // Console `attndelay`: cycles of wait before attention (diagnostic).
int attn_fold_early = 1; // Console `foldearly`: fold heads under the list (1) or after it (0).
static uint32_t kv_base;    // Per-channel byte address of the KV region (4 KB aligned).
uint32_t kv_rb;      // Beats per KV row packet (cols = head_dim, Q8).
uint32_t kv_region;  // Beats per (layer, kv-head, K|V) region.
int8_t  *kv_ekmax;   // Running max ek per region (B selection); -128 = empty.

uint32_t kv_region_addr(int l, int kvh, int kind)
{
    return kv_base + ((uint32_t)((l*n_kv_head + kvh)*2 + kind)*kv_region)*32;
}

int kv_ek_idx(int l, int kvh, int kind)
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

void kv_scores_br(int l, int kvh, int kind, int *B, int *R)
{
    *B = 22 - kv_ekmax[kv_ek_idx(l, kvh, kind)];
    *R = kv_br_r;
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
// the same points as the Python mirror's raw trace (model/mirror.py hybrid_generate, tag
// names shared) - software/lab/mirror.py lines the two up and names the first divergence.
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
    // FNV-1a over the 4*n bytes in memory order, one 32-bit load per element (byte 0 is
    // the low byte): a quarter of the loads of a byte loop - the ffn_act checkpoint reads
    // the scratchpad over the uncached bus - for the same hash.
    typedef uint32_t __attribute__((may_alias)) word_alias;
    const word_alias *p = values;
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t v = p[i];
        hash = (hash ^ (v & 0xffu))*16777619u;
        hash = (hash ^ ((v >> 8) & 0xffu))*16777619u;
        hash = (hash ^ ((v >> 16) & 0xffu))*16777619u;
        hash = (hash ^ (v >> 24))*16777619u;
    }
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
    if (hybrid_present()) {          // The 27B's layers: the hybrid forward (hybrid.c).
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
    // The quant8 shift of a norm the previous residual tail already ran (resid_norm_q8: the
    // int16/x8 outputs are in place), or RESID_TAIL_OFF: run the norm.
    int sx8_next = RESID_TAIL_OFF;
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
        int sx8 = sx8_next;
        if (sx8 == RESID_TAIL_OFF)
            sx8 = op_rmsnorm_q8_sp(SP_XR, ex_state, attn_norm[l], nrm_attn16[l], n_embd,
                                   SP_H, SP_X);
        sx8_next = RESID_TAIL_OFF;
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
                    if (kv_mstore_active) // One descriptor for all slots, patched below.
                        kv_pack_prep_layer(l, ev_res);
                    for (int hh = 0; hh < n_kv_head && !kv_mstore_active; hh++) {
                        kv_pack_prep_res(l, hh, 0, E_NORM);
                        kv_pack_prep_res(l, hh, 1, ev_res);
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
                    // run while the sequencer executes the qkr list.
                    // (The FFN rebuild rides the output projection's stream instead: the
                    // patches alone fill this window.)
                    seq_list_start(0, seq_qkr_count);
                    attn_patch_slots(l, 0, n_head, 0);
                    attn_prepatched = 1;
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
        if (attn_delay) { uint64_t t0 = perf_now64(); while (perf_now64() - t0 < attn_delay); }
        // Resident attention output: per-head int16 in SP_H + per-head value scales,
        // aligned to a common ea8 and scale8-ed into SP_X after the head loop.
        aw_hvalid = 0; // The fold starts empty for every path below (the list paths reset
                       // it again where they fold; the op-by-head path relies on this).
        // Attention (lists, then the op-by-op and CPU fallbacks; GQA).
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
                    uint32_t ready = attn_fold_early ? attn_fold_ready(hh, n_head) : 0xFFFFFFu, spins = 0;
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
            // The FFN norm's row streams under the scale8 work (the engine is idle there):
            // the residual tail after the projection then starts on it.
            const struct bons_entry *fn16 = nrm_ffn16[l];
            uint32_t dfn = fn16 ? norm_fetch_submit(fn16) : 0;
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
                if (likely(seq_use && seq_scale8_base >= 0)) {
                    // The output projection's weights stream under the scale8 list (the FFN
                    // norm row's fetch first: one op at a time on the readers).
                    if (engine_pf_use) {
                        if (fn16)
                            engine_join(dfn);
                        struct engine_op pfo = {0};
                        model_matvec_op(eo, SP_X, SP_Y, &pfo);
                        engine_prefetch(&pfo);
                    }
                    seq_run((uint32_t)seq_scale8_base, (uint32_t)n_head);
                }
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
            if (fn16)
                engine_join(dfn);
            {
                // The projection streams while the CPU rebuilds the FFN list's engine words.
                struct engine_op mo = {0};
                model_matvec_op(eo, SP_X, SP_Y, &mo);
                perf_enter(OP_ENGINE_WAIT);
                uint32_t dwo = engine_submit(&mo);
                if (seq_use && seq_ffn_base >= 0 && !ffn_prepatched && op_on_hw(OP_NORM_I32)
                    && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8)) {
                    ffn_prep_layer(l, next_norm16(l));
                    ffn_prepatched = 1;
                }
                engine_join(dwo);
                perf_exit(OP_ENGINE_WAIT);
            }
            // The residual tail and the FFN norm in one program when it can (resid_norm_q8),
            // the FFN gate's weights prefetched under it.
            struct engine_op pfg = {0};
            model_matvec_op(model_entry(ROLE_FFN_GATE, l), SP_X, SP_Y, &pfg);
            sx8_next = resid_norm_q8(&pfg, SP_Y, ea8 - (int)eo->bias + (int)eo->rshift,
                                     &ex_state, &x_amax, fn16, n_embd, 1);
            if (sx8_next == RESID_TAIL_OFF) {
                perf_enter(OP_NORM_I32);
                uint32_t dmax = 0;
                int kk = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
                perf_exit(OP_NORM_I32);
                int ed = ea8 - (int)eo->bias + (int)eo->rshift + kk;
                ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                        nrm_ffn16[l]);
            }
            dbg_sum("attn_res", l, SP_XR, 2*n_embd, ex_state);
        }

        // FFN (fully resident: no float round-trips).
        {
            const struct bons_entry *eg_t = model_entry(ROLE_FFN_GATE, l);
            const struct bons_entry *eu_t = model_entry(ROLE_FFN_UP, l);
            const struct bons_entry *ed_t = model_entry(ROLE_FFN_DOWN, l);
            int sx8f = sx8_next;
            if (sx8f == RESID_TAIL_OFF)
                sx8f = op_rmsnorm_q8_sp(SP_XR, ex_state, ffn_norm[l], nrm_ffn16[l], n_embd,
                                        SP_H, SP_X);
            sx8_next = RESID_TAIL_OFF;
            int ex8 = E_NORM + sx8f;
            dbg_sum("ffn_x8", l, SP_X, n_embd, ex8);

            if (likely(seq_use && seq_ffn_base >= 0 && op_on_hw(OP_NORM_I32)
                && op_on_hw(OP_SILU_MUL) && op_on_hw(OP_QUANT8))) {
                // One list for gate->norm->up->norm->silu->quant8->down (see seq_build_ffn).
                // Whole-list time is charged to OP_SILU_MUL: per-op shares are documented as
                // unreliable under sequencing; totals are the measurement.
                if (!ffn_prepatched)
                    ffn_prep_layer(l, next_norm16(l));
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
                // The residual tail and the next norm in one program - the last layer's
                // final norm only when the logits will read it.
                // The next layer's first projection prefetched under it: the qkv list's first
                // matvec (v with qkvx's v-k-q order, else q, as the op paths).
                struct engine_op pfq = {0};
                int qkv_first = (seq_use && seq_qkv_base >= 0 && qkvx_ok && op_on_hw(OP_NORM_I32))
                              ? ROLE_ATTN_V : ROLE_ATTN_Q;
                if (l + 1 < n_layer)
                    model_matvec_op(model_entry(qkv_first, l + 1), SP_X, SP_Y, &pfq);
                if (l + 1 < n_layer || want_logits)
                    sx8_next = resid_norm_q8(l + 1 < n_layer ? &pfq : 0, SP_Y,
                                             es8 - (int)ed_t->bias + (int)ed_t->rshift,
                                             &ex_state, &x_amax, next_norm16(l), n_embd, 1);
                if (sx8_next == RESID_TAIL_OFF) {
                    perf_enter(OP_NORM_I32);
                    uint32_t dmax = 0;
                    int kd = op_norm_i32_sp_m(SP_Y, SP_D, n_embd, &dmax);
                    perf_exit(OP_NORM_I32);
                    int ed = es8 - (int)ed_t->bias + (int)ed_t->rshift + kd;
                    ex_state = resid_add_pf(SP_XR, ex_state, SP_D, ed, n_embd, dmax, &x_amax,
                                            next_norm16(l));
                }
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
    int lm_sx8 = sx8_next;
    if (lm_sx8 == RESID_TAIL_OFF)
        lm_sx8 = op_rmsnorm_q8_sp(SP_XR, ex_state, out_norm, nrm_out16, n_embd, SP_H, SP_X);
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
            op.pk     = (head->kind == KIND_TERNARY5);
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
        op.pk     = (head->kind == KIND_TERNARY5);
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
        op.pk     = (head->kind == KIND_TERNARY5);
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
    tprof_setbuf(halloc(6000*20), 6000); // 120 KB trace buffer (main_ram, not sram .bss).
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
        // The hybrid models run their own forward (hybrid.c) - none of the Qwen3
        // lists, caches or resident regions below apply. The embedding comes from the
        // restored emb16 rows (embed_row_fetch).
        emb16_ent  = model_entry(ROLE_TOK_EMB16, 0);
        emb16x_ent = model_entry(ROLE_TOK_EMB16_EXP, 0);
        max_ctx = H->max_ctx;
        n_pos = 0;
        heap_hi = 0;
        int rc = hybrid_init();
        SIM_MARK("hybrid_init");
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
    // above engine channel 3's weight tail (capped at CPU_KV_CTX, halved below until it
    // fits).
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

// Diagnostic (`kvfill`): overwrite every channel's KV region with `pattern` (0: zeros, else a
// pseudo-random stream seeded by it), so a generation's dependence on stale rows shows.
void infer_kv_fill(uint32_t pattern)
{
    // Source: the KV row-packet staging (8 KB, rebuilt every forward); not SP_FET, which
    // keeps resident norm rows across forwards.
    volatile uint32_t *src = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_KVP);
    uint32_t x = pattern;
    for (int i = 0; i < 0x2000/4; i++) {
        if (pattern) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; }
        src[i] = pattern ? x : 0;
    }
    uint32_t beats = (uint32_t)n_layer*n_kv_head*2*kv_region;
    for (int c = 0; c < n_ch; c++)
        for (uint32_t b = 0; b < beats; b += 256) {
            uint32_t n = beats - b < 256 ? beats - b : 256;
            engine_store((uint8_t)c, kv_base + b*32, n, SP_KVP);
        }
    n_pos = 0;
    printf("KVFILL %08lx: %lu beats per channel\n", (unsigned long)pattern, (unsigned long)beats);
}

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

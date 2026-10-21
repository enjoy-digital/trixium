/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium resident block-floating-point ops
 *
 * The vector-unit ops on int16-mantissa activations resident in the scratchpad (exponents
 * tracked by the firmware), the resident attention and KV store, the resident embedding.
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

/* Resident BFP Ops */
/*------------------*/

// Every op has a CPU implementation (float math over the int16 scratchpad regions - the
// always-available fallback) and, when the vector unit exists, a HW implementation running
// the corresponding microcode program. Exponent algebra stays in this file; ops receive
// shift amounts only.

// rmsnorm: SP int16(src,esrc) -> SP int16(dst) at fixed E_NORM. HW path: fetch the
// image's int16 norm rows (kind norm16) into SP_FET and run the vector unit's two-pass
// program; CPU path: soft-float with the F32 weights from RAM.
//
// The int16 rows carry a per-tensor exponent in the entry's bias byte (BONS v2): a fixed
// exponent would clip the gammas (they reach 153 on the 1.7B). The program's A6 is the
// combined shift; the sequencer lists get theirs patched per layer next to their A5 (eps)
// and A1 (resident rows).

// rms_eps as an exact integer mantissa/exponent, so the per-call eps does not cost soft-float.
//
// The value wanted is lroundf(rms_eps * 2^-2esrc) - a pure power-of-two scaling, so float32
// computes it with no rounding beyond rms_eps's own representation. Decomposing once into a
// 24-bit mantissa (all a float32 has) makes the per-call work a shift and a rounding add,
// bit-identical to the float path, instead of soft-float ldexpf+lroundf per head norm.
static uint32_t eps_m;   // rms_eps = eps_m * 2^(eps_e - 24), eps_m in [2^23, 2^24)
static int      eps_e;

void eps_init(void)
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
int quant8_shift(uint32_t amax)
{
    if (amax == 0) return 8; // VCLZN(0) == 0, i.e. pre-shift 0 - keep the paths identical.
    return bitlen32(amax) - 7;
}

static int op_quant8_sp(uint32_t src, uint32_t dst, int n);

// The norm-weight row fetch that opens every rmsnorm_q8 would otherwise be a blocking engine
// op between two vector ops, with both units idle. The caller submits it BEFORE the
// preceding resid_add so it streams underneath, joins, and flags it; the op then skips its
// own fetch.
int rq8_prefetched;

uint32_t norm_fetch_submit(const struct bons_entry *w16)
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

// resid_add's output exponent (see op_resid_add_sp): bound the sum at the common exponent,
// then keep only the headroom it actually needs.
static int resid_eo(int ea, uint32_t amax, int eb, uint32_t bmax)
{
    int emin = (ea < eb) ? ea : eb;
    uint64_t bound = ((uint64_t)amax << (ea - emin)) + ((uint64_t)bmax << (eb - emin));
    int k = 0;
    while ((bound >> k) > 32767) k++;
    return emin + k;
}

// resid_add with the NEXT norm's weight row streaming underneath. SP_FET is idle here (its
// last reader, the previous rmsnorm or the qkr list, has finished) and the engine is idle
// (the preceding matvec/list joined) - same numerics either way.
int resid_add_pf(uint32_t src, int esrc, uint32_t d, int ed, int n, uint32_t dmax,
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
int op_rmsnorm_q8_sp(uint32_t src, int esrc, const float *w,
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

// The residual tail in one program (prog_resid_rmsnorm_q8): the projection's norm_i32, resid_add
// into the residual and the next rmsnorm_q8 - byte-identical to op_norm_i32_sp_m + resid_add_pf
// + op_rmsnorm_q8_sp, one dispatch instead of three, and the addend never goes through SP_D.
// The shifts ride one argument as 6-bit fields (asm.resid_tail_pack).
static int32_t resid_tail_pack(int nk, int nsa, int nsb, int nln, int wsh)
{
    int v[5] = {nk, nsa, nsb, nln, wsh};
    uint32_t word = 0;
    for (int i = 0; i < 5; i++) {
        if (v[i] < -32 || v[i] > 31)
            return -1; // Never a packing: bits 31:30 are always clear.
        word |= ((uint32_t)v[i] & 0x3F) << (6*i);
    }
    return (int32_t)word;
}

// src = the engine results (the projection that just ran: its absmax is still in the tracker,
// read before the norm-row fetch is submitted - the tracker resets on accept), eproj = their
// exponent before the norm shift (input exponent - bias + rshift), *eres/*xamax = the residual
// exponent/absmax (updated), w16 = the next norm's row (fetched: already in SP_FET). Returns the next norm's quant8 shift
// (int16 out in SP_H, x8 in SP_X), or RESID_TAIL_OFF with nothing run - an op off the hardware
// or a shift that does not fit its field - and the caller runs the three ops. pf (or NULL): the
// next matvec, prefetched under the program once it is sure to run (the three ops' fallback
// fetches for itself, so nothing is prefetched then).
int resid_norm_q8(const struct engine_op *pf, uint32_t src, int eproj, int *eres, uint32_t *xamax,
                  const struct bons_entry *w16, int n, int fetched)
{
#ifdef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
    if (!w16 || !op_on_hw(OP_NORM_I32) || !op_on_hw(OP_RESID_ADD) || !op_on_hw(OP_RMSNORM)
        || !op_on_hw(OP_QUANT8))
        return RESID_TAIL_OFF;
    uint32_t amax = trixium_engine_csr_absmax_read();
    int k = 0;
    while ((amax >> k) > 16383u) k++;
    uint32_t dmax = amax >> k;
    int ed = eproj + k;
    int eo = resid_eo(*eres, *xamax, ed, dmax);
    int ln = 0;
    while ((1 << ln) < n) ln++;
    int32_t a4 = resid_tail_pack(-k, *eres - eo, ed - eo, -ln, w16_shift(w16));
    if (a4 < 0)
        return RESID_TAIL_OFF;
    uint32_t df = fetched ? 0 : norm_fetch_submit(w16);
    int32_t args[8] = {(int32_t)src, (int32_t)SP_XR, (int32_t)SP_H, n/16, a4, eps_for(eo),
                       (int32_t)SP_FET, (int32_t)SP_X};
    if (!fetched)
        engine_join(df);
    if (pf)
        engine_prefetch(pf);
    perf_enter(OP_RESID_ADD);
    uint32_t r = (uint32_t)vec_run(PROG_RESID_RMSQ8, args, 8);
    perf_exit(OP_RESID_ADD);
    *eres  = eo;
    *xamax = r >> 16;
    return quant8_shift(r & 0xFFFF);
#else
    (void)pf; (void)src; (void)eproj; (void)eres; (void)xamax; (void)w16; (void)n; (void)fetched;
    return RESID_TAIL_OFF;
#endif
}

static void op_silu_mul_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n);

// silu_mul + measured quant8 in one program. Returns the quant8 shift.
int op_silu_q8_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n, uint32_t dst8)
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
int op_norm_i32_sp_a(uint32_t src, uint32_t dst, int n, uint32_t amax,
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

int op_norm_i32_sp_m(uint32_t src, uint32_t dst, int n, uint32_t *omax)
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
    int eo = resid_eo(ea, *amax_io, eb, bmax);
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
int attn_all_hw(void)
{
    return op_on_hw(OP_QKNORM) && op_on_hw(OP_ROPE) && op_on_hw(OP_KVQUANT)
        && op_on_hw(OP_SCORES) && op_on_hw(OP_SOFTMAX) && op_on_hw(OP_WV);
}

// vec quant8 head slice -> int8 at dst (exponent = es + 8). Fixed shift on purpose: the
// sequencer patches the softmax scale (which folds in this exponent) into a program it then
// runs head after head with no CPU in the loop, so the shift has to be known in advance.
// The measuring quant8 is used where firmware does the exponent algebra itself.
void vec_quant8_head(uint32_t sp, uint32_t dst, int n)
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
int kvp_scale_e8[KVP_SLOTS];

uint32_t kvp_slot(int kvh, int kind)
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
void kv_pack_prep_res(int l, int kvh, int kind, int es)
{
    int e8 = es + 8;
    kv_store_scale_prep(kvh, kind, e8);
    int idx = kv_ek_idx(l, kvh, kind);
    if (e8 > kv_ekmax[idx]) kv_ekmax[idx] = e8;
}

// kv_pack_prep_res for both kinds of every kv head of layer l (k at E_NORM, v at ev), the
// per-head bookkeeping in one loop: a slot's scale beat is restaged only when its exponent
// moved (k's never does after the first layer, v's once per layer).
void kv_pack_prep_layer(int l, int ev)
{
    int e8k = E_NORM + 8, e8v = ev + 8;
    int8_t *ek = &kv_ekmax[kv_ek_idx(l, 0, 0)];
    for (int hh = 0; hh < n_kv_head; hh++) {
        if (kvp_scale_e8[2*hh] != e8k)
            kv_store_scale_prep(hh, 0, e8k);
        if (kvp_scale_e8[2*hh + 1] != e8v)
            kv_store_scale_prep(hh, 1, e8v);
        if (e8k > ek[2*hh])     ek[2*hh]     = (int8_t)e8k;
        if (e8v > ek[2*hh + 1]) ek[2*hh + 1] = (int8_t)e8v;
    }
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
const struct bons_entry *emb16_ent, *emb16x_ent;

int embed_row_fetch(const struct bons_entry *e, int row, uint32_t sp_dst)
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
static int embed_row_restore(const struct bons_entry *e, int row, uint32_t sp_dst);
static int er_block;

COLD static int embed_row_bfp(const struct bons_entry *e, int row, uint32_t sp_dst)
{
    if (er_block)
        return embed_row_restore(e, row, sp_dst);
    if (e->kind == KIND_TERNARY5) { // Packed rows only come with the int16 table (bons.py).
        printf("embed: 5-trit token embedding without the int16 table\n");
        return 0;
    }
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch;
    engine_fetch(row/rpc,
                 TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)(row % rpc)*rb)*32,
                 rb, SP_FET);
    volatile uint8_t *pb = (volatile uint8_t *)(SCRATCHPAD_BASE + SP_FET);
    uint32_t sb = bons_scale_beats(e->cols, e->group);
    const volatile uint16_t *scales = (const volatile uint16_t *)pb;
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

/* Rotated Embedding Without The Int16 Table */
/*-------------------------------------------*/

// A 1 GB board's image leaves out the int16 table (ROLE_TOK_EMB16: ~1 GB for the 2B), and a
// rotated basis cannot be decoded group by group: the row is restored here in integers,
// bit-exact with bons.py restored_row_exact. v = (q-1)*m11 aligned to the row's smallest fp16
// exponent (exact in int64), y = s * (H v) per block with H the +-1 Sylvester matrix (a fast
// Walsh-Hadamard transform), the transform's normalization times the image's gain being the
// power of two 2^-floor(log2(block)/2); then int16 BFP: absmax mantissa in [2^14, 2^15),
// rounded half to even, clipped to +-32767. ~20k int64 butterflies per row at 2048.
static int8_t  *er_signs;  // s, from row 0 of the width's rotation tensors (codes s_j + 1).
static int64_t *er_acc;
static int      er_fexp;

void embed_restore_off(void)
{
    er_block = 0;
}

int embed_restore_setup(int block, int width, int wi)
{
    er_block = 0;
    if (block <= 0 || width % block) return -1;
    er_signs = infer_halloc((uint64_t)width, 0);
    er_acc   = infer_halloc((uint64_t)width*sizeof(int64_t), 0);
    if (!er_signs || !er_acc) return -1;
    for (int b = 0; b < width/block; b++) {
        const struct bons_entry *e = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, b));
        if (!e || e->kind != KIND_TERNARY || (int)e->cols != block) return -1;
        // Row 0 is channel 0's first row.
        engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + (uint64_t)e->chan_base_beats*32,
                     bons_row_beats(e->cols, e->kind, e->group), SP_FET);
        const volatile uint32_t *cw = (const volatile uint32_t *)(SCRATCHPAD_BASE + SP_FET
            + bons_scale_beats(e->cols, e->group)*32);
        for (int j = 0; j < block; j++)
            er_signs[b*block + j] = (int8_t)((int)((cw[j/16] >> (2*(j % 16))) & 3) - 1);
    }
    int lg = 0;
    while ((1 << lg) < block) lg++;
    er_fexp  = -(lg/2);
    er_block = block;
    return 0;
}

static int embed_row_restore(const struct bons_entry *e, int row, uint32_t sp_dst)
{
    if (e->kind != KIND_TERNARY || e->cols % er_block) {
        printf("embed: no restore for this embedding table\n");
        return 0;
    }
    int n = e->cols;
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch;
    engine_fetch(row/rpc,
                 TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)(row % rpc)*rb)*32,
                 rb, SP_FET);
    volatile uint8_t *pb = (volatile uint8_t *)(SCRATCHPAD_BASE + SP_FET);
    const volatile uint16_t *scales = (const volatile uint16_t *)pb;
    const volatile uint32_t *cw = (const volatile uint32_t *)(pb + bons_scale_beats(e->cols, e->group)*32);
    int G = n/128;
    int64_t m[64];
    int     e16[64], e16min = 1000;
    for (int g = 0; g < G; g++) {
        uint16_t b2 = scales[g];
        int ex   = (b2 >> 10) & 0x1F;
        int frac = b2 & 0x3FF;
        int mm   = (ex > 0) ? (1024 + frac) : frac;
        m[g]     = (b2 & 0x8000) ? -mm : mm;
        e16[g]   = (ex > 0) ? (ex - 25) : -24;
        if (e16[g] < e16min) e16min = e16[g];
    }
    // v, exact: |m11| < 2^11 aligned by at most 30 bits, sums of a block below 2^50.
    int64_t *v = er_acc;
    for (int g = 0; g < G; g++) {
        int64_t mg = m[g] << (e16[g] - e16min);
        for (int w = 0; w < 8; w++) {
            uint32_t c = cw[g*8 + w];
            for (int i = 0; i < 16; i++) {
                uint32_t q = (c >> (2*i)) & 3;
                v[g*128 + w*16 + i] = (q == 2) ? mg : (q == 0) ? -mg : 0;
            }
        }
    }
    // Walsh-Hadamard per block (h below the block: the strides never cross one).
    for (int h = 1; h < er_block; h <<= 1)
        for (int i = 0; i < n; i += 2*h)
            for (int j = i; j < i + h; j++) {
                int64_t a = v[j], b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
    uint64_t amax = 0;
    for (int j = 0; j < n; j++) {
        if (er_signs[j] < 0) v[j] = -v[j];
        uint64_t a = (uint64_t)((v[j] < 0) ? -v[j] : v[j]);
        if (a > amax) amax = a;
    }
    volatile uint32_t *pw = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_dst);
    if (amax == 0) {
        for (int j = 0; j < n/2; j++) pw[j] = 0;
        return 0;
    }
    int bl = (amax >> 32) ? 32 + bitlen32((uint32_t)(amax >> 32)) : bitlen32((uint32_t)amax);
    int r  = bl - 15;
    for (int j = 0; j < n; j += 2) {
        int32_t o[2];
        for (int k = 0; k < 2; k++) {
            int64_t y = v[j + k];
            int64_t mag = (y < 0) ? -y : y, qv;
            if (r > 0) {
                int64_t rem  = mag & ((1ll << r) - 1), half = 1ll << (r - 1);
                qv = mag >> r;
                if (rem > half || (rem == half && (qv & 1))) qv++;
            } else {
                qv = mag << -r;
            }
            if (qv > 32767) qv = 32767;
            o[k] = (int32_t)((y < 0) ? -qv : qv);
        }
        pw[j/2] = (uint32_t)(uint16_t)o[0] | ((uint32_t)(uint16_t)o[1] << 16);
    }
    return e16min + er_fexp + r;
}

int infer_embed_f(int token, float *dst)
{
    const struct bons_entry *emb = model_entry(ROLE_TOK_EMB, 0);
    if (!emb) return -1;
    int e = embed_row_fetch(emb, token, SP_XR);
    float s = ldexpf(1.0f, e);
    // The row to main RAM by one engine STORE (~1.5k cycles with the dispatch) and converted
    // from cached memory: read from the scratchpad it was 1,024 uncached bus reads of two
    // int16 (~30 cycles each), the hybrid forward's second-largest engine-idle gap (45k).
    uint32_t bytes = (2u*(uint32_t)n_embd + 31u) & ~31u;
    if (bytes <= 16384u) {
        int16_t *row = (int16_t *)TRIXIUM_EMB16_ROW_ADDR;
        engine_store(0, (uint32_t)((uintptr_t)row - MAIN_RAM_BASE + TRIXIUM_MEM_CPU_OFFSET),
                     bytes/32u, SP_XR);
        invd_cpu_dcache_range(row, bytes);
        for (int i = 0; i < n_embd; i++)
            dst[i] = (float)row[i]*s;
        return 0;
    }
    const volatile uint32_t *p = (const volatile uint32_t *)(SCRATCHPAD_BASE + SP_XR);
    int i = 0;
    for (; i + 1 < n_embd; i += 2) {
        uint32_t w = p[i/2];
        dst[i]     = (float)(int16_t)(w & 0xffffu)*s;
        dst[i + 1] = (float)(int16_t)(w >> 16)*s;
    }
    if (i < n_embd) dst[i] = (float)sp16(SP_XR)[i]*s;
    return 0;
}

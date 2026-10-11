/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium sequencer descriptor lists
 *
 * The per-layer op lists the descriptor sequencer runs (qk-norm/RoPE, attention, FFN,
 * scale8, QKV), their control, and the per-layer patching of their arguments.
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

/* Sequencer: QK-Norm/RoPE List */
/*------------------------------*/

// Sequencer descriptor list for the per-head qk-norm + rope loops. Addresses are
// head-relative inside SP_Q/SP_K, so the list is layer-independent; per layer only the
// rmsnorm eps_int args (A5 = descriptor word 7) change. Built once at init.
int seq_qkr_count;
// Descriptor indices of the q-norm and k-norm in the qkr list (their A1/A5 are patched
// per layer).
int seq_qkr_di_qn = 0, seq_qkr_di_kn = 1;

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
int qkvx_ok  = 0;      // What the built lists use.
static int qkvx_blocked = 0;  // The 14-descriptor qkv list did not fit: rebuilt without.
int seq_qkv_len = 10;
int seq_qkv_di_kn = -1;

// In-list KV stores: each (kvh, kind) staging slot's engine STORE rides the qkr list as a
// NOWAIT descriptor right after the pack quant8 that fills the slot - the next head's
// quant8 (vec) runs under the store's stream, so the serial CPU-dispatched stores hide under
// vector work the layer does anyway. Needs the scale-beat prep to run BEFORE the list (the
// store streams the scale beat), which is equally correct in the CPU-store layout.
int kv_pipe_active = 0;  // What the built list uses.

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
int kv_mstore_active = 0; // What the built list uses.
uint8_t seq_kvst_di[32]; // Descriptor index of the store for slot (kvh*2 + kind).
// What the store descriptors' constant words currently hold (~0 = unknown/rebuilt).
uint32_t kvst_rb_set = 0xFFFFFFFFu, kvst_chan_set = 0xFFFFFFFFu;

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

void seq_build_qkr(void)
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

void infer_static_init(void)
{
    for (int i = 0; i < 128; i++) eps_memo[i] = 0;
}

int seq_eps_int(int es)
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
int seq_attn_base, seq_attn_dph, seq_attn_heads;
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
int attn_pipe_active = 0; // What the built list uses.
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
uint32_t sp_scb2;          // Second scores buffer (0 = no deep layout possible).
uint32_t sc_b_cap_bytes;   // Its capacity.
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

void seq_build_attn(void)
{
    seq_attn_heads = 0;
    attn_prep_pos = -1; // Rebuild wipes the per-token patch words; force a re-prep.
    if (n_ch < 2)
        return; // n1's absmax lives in the engine CSR; keep CPU-dispatched ops there.
    // 5 descriptors per head: scores, softmax8 (self-spills its p8 sum), axpy(xred),
    // spill(absmax from the engine tracker), norm_i32 (the q quant8 is batched in the qkr
    // list, the p8 spill is softmax8's tail and the engine reduces across channels).
    // cap: what the 192-descriptor list leaves behind the qkr list.
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
uint64_t *const fprof_acc_list = (uint64_t *)(TRIXIUM_FPROF_BUFFER_ADDR);
uint64_t *const fprof_acc_ops = (uint64_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 32);
uint32_t fprof_nlist, fprof_nops;
uint32_t *const fprof_ts = (uint32_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 64);
int fprof_nts;
uint32_t *const fprof_steps = (uint32_t *)(TRIXIUM_FPROF_BUFFER_ADDR + 128);
int fprof_nsteps;

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
int seq_ffn_base = -1;
int seq_ffn_len  = 0;
static int ffn_shape_base = -1; // seq_ffn_base the FFN engine shape images were written for.

// The SiLU split: everything in silu_q8 up to the x*sigmoid(x) product is a function of the
// GATE alone, so that half (silu_gate) runs UNDER the up matvec's stream (where norm_g
// already runs) and only the multiply/absmax/quant8 tail (mul_q8) waits for up.
// Byte-identical by construction (asm.py, test_silu_split_matches_fused). It relies on the
// matvec result packer: without it the vector unit runs ~2.9x slower under an engine stream
// and norm_g alone fills the window.
static int ffn_di_down = 7; // Down matvec's descriptor (seq_build_ffn).
int ffn_di_a4   = 6; // Descriptor whose A4 takes the per-layer gate shift.

void seq_build_ffn(void)
{
    int di = seq_qkr_count + attn_list_len(seq_attn_heads);
    int len = 9;
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
    //   6 norm_mul_q8 SP_G x (SP_YU >> ku) -> SP_X [ku via KRES6; res6 = source absmax,
    //     read by the CPU after the list - desc 7 is an engine op and leaves it alone]
    //   7 down mv   8 stop
    // The up norm rides inside the multiply (prog_norm_mul_q8): no int16 up in SP_U, one pass
    // and one descriptor less on the serial tail between the up and down streams.
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
    {
        // norm_mul_q8: (silu * (up >> ku)) << A6, clamp, absmax (res6 for the CPU), quant8
        // -> SP_X; A4 = ku from the spill's res6 (KRES6), as norm_u took it.
        int32_t ma[8] = {(int32_t)SP_G, (int32_t)SP_YU, (int32_t)SP_G, ffn_dim/16, 0, 0, -15,
                         (int32_t)SP_X};
        seq_desc_vec(d, PROG_NORM_MUL_Q8, ma, 8, (uint16_t)(SEQ_SRC_KRES6 << 8));
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
int seq_scale8_base = -1;

void seq_build_scale8(void)
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
int seq_qkv_base = -1;
int qkv_prepped  = -1; // Layer whose projections are currently in the slots.

void seq_build_qkv(void)
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

void qkv_prep_layer(int l)
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
        // the norm shift the sequencer adds from the kreg. 32 entries cover sx8 in [0, 13]
        // with k <= 18 (int32 sources); seq_eps_int is the same function the qkr list's A5
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
// the vector unit (doc/units.md (Overlap and scratchpad arbitration)): e.g. the FFN up-projection stream runs UNDER the
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
int attn_prepatched;
int ffn_prepatched;
int qkn_resident;   // 1 once every layer's q/k norm rows sit in SP_QNW/SP_KNW.

// Per-layer FFN list rebuild (three engine descriptors + their flag words), extracted so it
// can POST under the qkr list (all inputs are layer-known: model entries and static
// scratchpad addresses; only the silu A4 imm needs ex8 and stays at FFN time). The
// sequencer-fetch interlock orders these writes before the FFN list runs.
static inline void ffn_patch_lw(uint32_t di, const struct engine_op *mo)
{
    seq_patch(di, 4, mo->wbase);
    seq_patch(di, 7, mo->bias | ((uint32_t)mo->rshift << 16) | ((uint32_t)mo->rshift2 << 22));
}

void ffn_prep_layer(int l)
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

void attn_patch_slots(int l, int h0, int nh, int chunked)
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
void attn_list_prep_token(int pos)
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

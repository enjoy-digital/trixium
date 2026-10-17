/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium per-op cycle profiling
 *
 * Per-op cycles on the 64-bit architectural counter, engine/vector-unit overlap counters and
 * the full-token trace (tprof). The offload work is driven by these numbers: profile, move the
 * heaviest op to hardware, re-profile.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef PERF_H
#define PERF_H

#include <stdint.h>

/* Ops */
/*-----*/

// Op classes (also the hwops mask bit positions, see ops.h).
enum perf_op {
    OP_EMBED = 0,
    OP_RMSNORM,     // Full-width norms (attn/ffn/final).
    OP_QKNORM,      // Per-head Q/K norms.
    OP_ROPE,
    OP_KVQUANT,     // KV quantize + append (CPU cache and/or HBM pack+store).
    OP_SCORES,      // Attention QK^T.
    OP_SOFTMAX,
    OP_WV,          // Attention weighted-V sum.
    OP_QUANT8,      // Activation float->int8 quantize into the scratchpad.
    OP_NORM_I32,    // Engine result int32 -> activation conversion/scaling.
    OP_SILU_MUL,
    OP_RESID_ADD,
    OP_LMHEAD,      // Chunked LM head scan + argmax.
    // The hybrid forward's phases (hybrid.c; its attention softmax/scores use the ops above).
    OP_DN_PROJ,     // DeltaNet in-projections list (qkv, z, beta, alpha).
    OP_DN_ACT,      // Per-head sigmoid/softplus scalars and the stream records (CPU).
    OP_DN_CONV,     // Causal conv + SiLU list (its build included).
    OP_DN_PREP,     // q/k L2 norm list.
    OP_DN_HEADS,    // The recurrence: op 5 stream or the CPU path.
    OP_DN_GATE,     // Gated RMSNorm x SiLU(z) list.
    OP_DN_OUT,      // Out-projection prepare + matvec.
    OP_ATTN,        // An attention layer (softmax inside as OP_SOFTMAX).
    OP_FFN_PROJ,    // Gate/up projections (+ act when fused).
    OP_FFN_DOWN,    // Down projection (prepare + matvec).
    OP_ENGINE_WAIT, // Engine polling; may overlap op brackets, not always enclosed by one.
    OP_COUNT,
};

/* Profiling */
/*-----------*/

void     perf_init(void);
void     perf_reset(void);
uint32_t perf_now(void);   // Legacy timer: only intervals shorter than 2^32 cycles are valid.
uint64_t perf_now64(void); // Architectural cycle/cycleh counter for long benchmarks.
void     perf_enter(enum perf_op op);
void     perf_exit(enum perf_op op);
void     perf_print(void);          // Dump per-op {calls, cycles} table + engine counters.
void     perf_print_u64(uint64_t v); // LiteX printf does not implement %llu.
void     perf_ovl_mark(void);           // Overlap-window start.
void     perf_ovl_acc(uint64_t acc[4]); // Zero-init; acc[0] == UINT64_MAX means invalid.
void     perf_ovl_report(const char *tag, const uint64_t acc[4]);

// Boot-phase marks for the simulation (TRIXIUM_SIM, defined by trixium_sim.py's SoC only):
// the architectural cycle count since reset, so a slow simulated boot can be broken down.
// Compiled out of the board firmware.
#ifdef TRIXIUM_SIM
#define SIM_MARK(label) printf("SIMT %s %lu\n", (label), (unsigned long)perf_now64())
#else
#define SIM_MARK(label) ((void)0)
#endif

// Accumulators, exposed for bench.
extern uint64_t perf_cycles[OP_COUNT];
extern uint64_t perf_calls[OP_COUNT];

/* Token Trace */
/*-------------*/

// tprof: full-token trace. `tprof arm` on the console arms it; the next want_logits
// forward records events (tag, t, cyc_eng, cyc_vec) until it returns, then disarms.
// tag[31:28] = kind: 1 op-enter, 2 op-exit (op in tag[15:8]), 3 layer mark (layer in
// [15:8]), 4 dcount transition (dcount in [19:0]), 5 list start (idx0 in [15:8], count
// in [7:0]), 6 forward start, 7 forward end. Events cost ~2 CSR reads + 4 stores each, so a
// traced run is a diagnostic, not a benchmark. Inert when not tracing: one predictable branch
// per hook.
extern int tprof_armed, tprof_active;
void tprof_setbuf(void *p, int max_events); // Buffer lives in main_ram (halloc'd).
void tprof_begin(void);
void tprof_end(void);
void tprof_event(uint32_t tag);
void tprof_dcount(uint32_t st); // Feed seq_status; logs kind-4 on dcount change.
void tprof_dump(void);          // TP <tag> <t> <eng> <vec> <both> lines over the console.

#endif

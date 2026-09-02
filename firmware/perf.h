// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Per-op cycle profiling on timer0 (free-running down-counter). The whole offload ladder is
// driven by these numbers: profile -> move the heaviest op to hardware -> re-profile.

#ifndef PERF_H
#define PERF_H

#include <stdint.h>

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
    OP_ENGINE_WAIT, // Inside engine_run() polls (also counted within the enclosing op).
    OP_COUNT,
};

void     perf_init(void);
void     perf_reset(void);
uint32_t perf_now(void); // Free-running cycle counter (wraps; deltas are valid).
void     perf_enter(enum perf_op op);
void     perf_exit(enum perf_op op);
void     perf_print(void); // Dump per-op {calls, cycles} table + engine counters.
void     perf_ovl_mark(void);               // C2-1a: overlap-window start.
void     perf_ovl_acc(uint32_t acc[4]);     // C2-1a: add window deltas to acc[tot,eng,vec,both].
void     perf_ovl_report(const char *tag, const uint32_t acc[4]);

// Accumulators exposed for bench.
extern uint32_t perf_cycles[OP_COUNT];
extern uint32_t perf_calls[OP_COUNT];

// tprof: full-token trace. `tprof arm` on the console arms it; the next want_logits
// forward records events (tag, t, cyc_eng, cyc_vec) until it returns, then disarms.
// tag[31:28] = kind: 1 op-enter, 2 op-exit (op in tag[15:8]), 3 layer mark (layer in
// [15:8]), 4 dcount transition (dcount in [19:0]), 5 list start (idx0 in [15:8], count
// in [7:0]), 6 forward start, 7 forward end. Events cost ~2 CSR reads + 4 stores each
// (~7% of a traced token) - a diagnostic run, not a benchmark. Inert when not tracing:
// one predictable branch per hook.
extern int tprof_armed, tprof_active;
void tprof_setbuf(void *p, int max_events); // Buffer lives in main_ram (halloc'd).
void tprof_begin(void);
void tprof_end(void);
void tprof_event(uint32_t tag);
void tprof_dcount(uint32_t st); // Feed seq_status; logs kind-4 on dcount change.
void tprof_dump(void);          // TP <tag> <t> <eng> <vec> lines over the console.

#endif

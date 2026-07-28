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

// Accumulators exposed for bench.
extern uint32_t perf_cycles[OP_COUNT];
extern uint32_t perf_calls[OP_COUNT];

#endif

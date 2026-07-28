// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "perf.h"

uint32_t perf_cycles[OP_COUNT];
uint32_t perf_calls[OP_COUNT];
static uint32_t perf_t0[OP_COUNT];

static const char *op_names[OP_COUNT] = {
    "embed", "rmsnorm", "qknorm", "rope", "kvquant", "scores", "softmax", "wv",
    "quant8", "norm_i32", "silu_mul", "resid_add", "lmhead", "engine_wait",
};

void perf_init(void)
{
    // Free-running down-counter over the full 32-bit range.
    timer0_en_write(0);
    timer0_load_write(0xFFFFFFFF);
    timer0_reload_write(0xFFFFFFFF);
    timer0_en_write(1);
    perf_reset();
}

void perf_reset(void)
{
    for (int i = 0; i < OP_COUNT; i++) {
        perf_cycles[i] = 0;
        perf_calls[i]  = 0;
    }
}

uint32_t perf_now(void)
{
    timer0_update_value_write(1);
    return ~timer0_value_read(); // Down-counter -> up-counting cycles.
}

void perf_enter(enum perf_op op)
{
    perf_t0[op] = perf_now();
}

void perf_exit(enum perf_op op)
{
    perf_cycles[op] += perf_now() - perf_t0[op];
    perf_calls[op]  += 1;
}

void perf_print(void)
{
    uint32_t total = 0;
    for (int i = 0; i < OP_COUNT; i++)
        if (i != OP_ENGINE_WAIT)
            total += perf_cycles[i];
    puts("op            calls      cycles    %");
    for (int i = 0; i < OP_COUNT; i++) {
        if (!perf_calls[i])
            continue;
        printf("%-12s %6lu %11lu %4lu%s\n", op_names[i],
               (unsigned long)perf_calls[i], (unsigned long)perf_cycles[i],
               (unsigned long)(total ? (uint64_t)perf_cycles[i]*100/total : 0),
               i == OP_ENGINE_WAIT ? " (within ops above)" : "");
    }
    printf("total (excl. engine_wait overlap): %lu cycles\n", (unsigned long)total);
}

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

static uint32_t ovl0_tot, ovl0_eng, ovl0_vec, ovl0_both;

void perf_reset(void)
{
    // Snapshot the free-running overlap counters too. Without this the percentages are
    // computed over everything since reset - boot, selftest and the weight load - which on
    // the n8 sim is ~40M cycles against ~1.7M for a generation leg, so the token loop is
    // diluted to nothing and every share reads 0%.
#ifdef CSR_ENGINE_CSR_CYC_TOTAL_ADDR
    ovl0_tot  = engine_csr_cyc_total_read();
    ovl0_eng  = engine_csr_cyc_eng_read();
    ovl0_vec  = engine_csr_cyc_vec_read();
    ovl0_both = engine_csr_cyc_both_read();
#endif
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
    perf_print_overlap();
}

// How much of the run had the engine and the vector unit busy at the same time. The MAC
// array is idle ~92% of a token and the sequencer issues one descriptor at a time, so
// `neither` is the number that decides whether a dual-issue sequencer is worth building:
// it is the time no unit is doing work at all, i.e. pure serialisation and dispatch.
void perf_print_overlap(void)
{
#ifdef CSR_ENGINE_CSR_CYC_TOTAL_ADDR
    uint32_t tot  = engine_csr_cyc_total_read() - ovl0_tot;
    uint32_t eng  = engine_csr_cyc_eng_read()   - ovl0_eng;
    uint32_t vec  = engine_csr_cyc_vec_read()   - ovl0_vec;
    uint32_t both = engine_csr_cyc_both_read()  - ovl0_both;
    if (!tot)
        return;
    uint32_t neither = tot - (eng + vec - both);
    printf("overlap: eng %lu%%  vec %lu%%  both %lu%%  neither %lu%%  (of %lu cycles)\n",
           (unsigned long)((uint64_t)eng*100/tot),  (unsigned long)((uint64_t)vec*100/tot),
           (unsigned long)((uint64_t)both*100/tot),
           (unsigned long)((uint64_t)neither*100/tot), (unsigned long)tot);
#endif
}

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
    {
        extern uint32_t eng_poll_spins, eng_poll_waits, vec_poll_spins, vec_poll_waits,
                        seq_poll_spins, seq_poll_waits;
        eng_poll_spins = eng_poll_waits = 0;
        vec_poll_spins = vec_poll_waits = 0;
        seq_poll_spins = seq_poll_waits = 0;
    }
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
    if (tprof_active)
        tprof_event((1u << 28) | ((uint32_t)op << 8));
}

void perf_exit(enum perf_op op)
{
    perf_cycles[op] += perf_now() - perf_t0[op];
    perf_calls[op]  += 1;
    if (tprof_active)
        tprof_event((2u << 28) | ((uint32_t)op << 8));
}

// tprof ------------------------------------------------------------------------------------

int tprof_armed, tprof_active;
static uint32_t *tprof_buf;
static int tprof_n, tprof_cap; // Words used / word capacity (4 words per event).
static uint32_t tprof_last_dc;

void tprof_setbuf(void *p, int max_events)
{
    tprof_buf = (uint32_t *)p;
    tprof_cap = max_events*4;
}

void tprof_event(uint32_t tag)
{
    if (!tprof_active || !tprof_buf || tprof_n > tprof_cap - 4)
        return;
    tprof_buf[tprof_n++] = tag;
    tprof_buf[tprof_n++] = perf_now();
#ifdef CSR_ENGINE_CSR_CYC_ENG_ADDR
    tprof_buf[tprof_n++] = engine_csr_cyc_eng_read();
    tprof_buf[tprof_n++] = engine_csr_cyc_vec_read();
#else
    tprof_buf[tprof_n++] = 0;
    tprof_buf[tprof_n++] = 0;
#endif
}

void tprof_begin(void)
{
    tprof_armed   = 0;
    tprof_n       = 0;
    tprof_last_dc = 0xFFFFFFFF;
    tprof_active  = 1;
    tprof_event(6u << 28);
}

void tprof_end(void)
{
    tprof_event(7u << 28);
    tprof_active = 0;
}

void tprof_dcount(uint32_t st)
{
    if (!tprof_active)
        return;
    uint32_t dc = st >> 8;
    if (dc == tprof_last_dc)
        return;
    tprof_last_dc = dc;
    tprof_event((4u << 28) | (dc & 0xFFFFF));
}

void tprof_dump(void)
{
    printf("TPROF %d events (cap %d)%s\n", tprof_n/4, tprof_cap/4,
           tprof_n > tprof_cap - 4 ? " TRUNCATED" : "");
    for (int i = 0; i < tprof_n; i += 4)
        printf("TP %08lx %lu %lu %lu\n",
               (unsigned long)tprof_buf[i],   (unsigned long)tprof_buf[i+1],
               (unsigned long)tprof_buf[i+2], (unsigned long)tprof_buf[i+3]);
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
    {
        // Completion-poll census: iterations x ~70cy CSR read bounds what IRQ completion
        // could recover. Deltas since perf_reset.
        extern uint32_t eng_poll_spins, eng_poll_waits, vec_poll_spins, vec_poll_waits,
                        seq_poll_spins, seq_poll_waits;
        printf("poll: eng %lu/%lu  vec %lu/%lu  seq %lu/%lu (spins/waits; x~70cy = bound)\n",
               (unsigned long)eng_poll_spins, (unsigned long)eng_poll_waits,
               (unsigned long)vec_poll_spins, (unsigned long)vec_poll_waits,
               (unsigned long)seq_poll_spins, (unsigned long)seq_poll_waits);
    }
    perf_print_overlap();
}

// How much of the run had the engine and the vector unit busy at the same time. The MAC
// array is idle ~92% of a token and the sequencer issues one descriptor at a time, so
// `neither` is the number that decides whether a dual-issue sequencer is worth building:
// it is the time no unit is doing work at all, i.e. pure serialisation and dispatch.
// Windowed overlap accounting (C2-1a): mark at a window's start, accumulate the four
// counter deltas at its end. Callers keep their own acc[4] = {tot, eng, vec, both}.
static uint32_t ovlw_tot, ovlw_eng, ovlw_vec, ovlw_both;

void perf_ovl_mark(void)
{
#ifdef CSR_ENGINE_CSR_CYC_TOTAL_ADDR
    ovlw_tot  = engine_csr_cyc_total_read();
    ovlw_eng  = engine_csr_cyc_eng_read();
    ovlw_vec  = engine_csr_cyc_vec_read();
    ovlw_both = engine_csr_cyc_both_read();
#endif
}

void perf_ovl_acc(uint32_t acc[4])
{
#ifdef CSR_ENGINE_CSR_CYC_TOTAL_ADDR
    acc[0] += engine_csr_cyc_total_read() - ovlw_tot;
    acc[1] += engine_csr_cyc_eng_read()   - ovlw_eng;
    acc[2] += engine_csr_cyc_vec_read()   - ovlw_vec;
    acc[3] += engine_csr_cyc_both_read()  - ovlw_both;
#else
    (void)acc;
#endif
}

void perf_ovl_report(const char *tag, const uint32_t acc[4])
{
    uint32_t tot = acc[0] ? acc[0] : 1;
    printf("OVLW %-10s tot %9lu  eng %2lu%%  vec %2lu%%  both %2lu%%  neither %2lu%%\n",
           tag, (unsigned long)acc[0],
           (unsigned long)((uint64_t)acc[1]*100/tot),
           (unsigned long)((uint64_t)acc[2]*100/tot),
           (unsigned long)((uint64_t)acc[3]*100/tot),
           (unsigned long)((acc[0] + acc[3] >= acc[1] + acc[2])
               ? (uint64_t)(acc[0] + acc[3] - acc[1] - acc[2])*100/tot : 0));
}

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

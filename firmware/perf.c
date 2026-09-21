// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "perf.h"
#include "hybrid.h"

uint64_t perf_cycles[OP_COUNT];
uint64_t perf_calls[OP_COUNT];
static uint64_t perf_t0[OP_COUNT];
static uint64_t perf_start;

struct overlap_snapshot {
    uint64_t cycles[4];
    uint64_t stamp;
};
static struct overlap_snapshot ovl0, ovlw;

static void overlap_snapshot(struct overlap_snapshot *s)
{
    s->stamp = perf_now64();
#ifdef CSR_ENGINE_CSR_PERF_SNAPSHOT_ADDR
    engine_csr_perf_snapshot_write(1);
    s->cycles[0] = engine_csr_snap_total_read();
    s->cycles[1] = engine_csr_snap_eng_read();
    s->cycles[2] = engine_csr_snap_vec_read();
    s->cycles[3] = engine_csr_snap_both_read();
#elif defined(CSR_ENGINE_CSR_CYC_TOTAL_ADDR)
    s->cycles[0] = engine_csr_cyc_total_read();
    s->cycles[1] = engine_csr_cyc_eng_read();
    s->cycles[2] = engine_csr_cyc_vec_read();
    s->cycles[3] = engine_csr_cyc_both_read();
#else
    for (int i = 0; i < 4; i++) s->cycles[i] = 0;
#endif
}

static int overlap_delta(const struct overlap_snapshot *start,
                         const struct overlap_snapshot *end, uint64_t out[4])
{
#ifdef CSR_ENGINE_CSR_PERF_SNAPSHOT_ADDR
    for (int i = 0; i < 4; i++) {
        if (end->cycles[i] < start->cycles[i]) return 0; // reset during window
        out[i] = end->cycles[i] - start->cycles[i];
    }
    // Invariant for an atomic snapshot interval. Avoid overflowing intermediate sums.
    return out[1] <= out[0] && out[2] <= out[0] && out[3] <= out[1]
        && out[3] <= out[2] && out[1] - out[3] <= out[0] - out[2];
#elif defined(CSR_ENGINE_CSR_CYC_TOTAL_ADDR)
    // Cannot reconstruct wraps from two low-word samples. Include CSR-read time and
    // reject the whole interval if even one wrap could have occurred. Legacy reads are
    // not simultaneous, so their percentages are explicitly approximate.
    if (perf_now64() - start->stamp >= UINT32_MAX) return 0;
    for (int i = 0; i < 4; i++)
        out[i] = (uint32_t)(end->cycles[i] - start->cycles[i]);
    return 1;
#else
    (void)start; (void)end; (void)out;
    return 0;
#endif
}

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
    hybrid_stats_reset();
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
    overlap_snapshot(&ovl0);
    perf_start = perf_now64();
    for (int i = 0; i < OP_COUNT; i++) {
        perf_cycles[i] = 0;
        perf_calls[i]  = 0;
    }
}

#ifndef BONSAI_PERF_TEST // Host tests inject clocks; production always uses these readers.
uint32_t perf_now(void)
{
    timer0_update_value_write(1);
    return ~timer0_value_read(); // Down-counter -> up-counting cycles.
}

uint64_t perf_now64(void)
{
    // A 27B forward is over a billion cycles and even bench 1 can cross the 32-bit timer
    // wrap.  RV32 exposes the architectural counter as two CSRs; sample high/low/high so
    // a low-word rollover between reads cannot create a torn timestamp.
    uint32_t hi0, lo, hi1;
    do {
        __asm__ volatile("rdcycleh %0" : "=r"(hi0));
        __asm__ volatile("rdcycle %0"  : "=r"(lo));
        __asm__ volatile("rdcycleh %0" : "=r"(hi1));
    } while (hi0 != hi1);
    return ((uint64_t)hi0 << 32) | lo;
}
#endif

void perf_enter(enum perf_op op)
{
    perf_t0[op] = perf_now64();
    if (tprof_active)
        tprof_event((1u << 28) | ((uint32_t)op << 8));
}

void perf_exit(enum perf_op op)
{
    perf_cycles[op] += perf_now64() - perf_t0[op];
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

void perf_print_u64(uint64_t v)
{
    char buf[20];
    int n = 0;
    do {
        buf[n++] = '0' + (char)(v % 10);
        v /= 10;
    } while (v);
    while (n) putchar(buf[--n]);
}

static void overlap_print(const char *tag, const uint64_t acc[4]);

void perf_print(void)
{
    uint64_t wall = perf_now64() - perf_start;
    struct overlap_snapshot end;
    uint64_t overlap[4];
    overlap_snapshot(&end);
    int overlap_ok = overlap_delta(&ovl0, &end, overlap);
    uint64_t total = 0;
    for (int i = 0; i < OP_COUNT; i++)
        if (i != OP_ENGINE_WAIT)
            total += perf_cycles[i];
    puts("op            calls cycles %instrumented (not %wall)");
    for (int i = 0; i < OP_COUNT; i++) {
        if (!perf_calls[i])
            continue;
        printf("%-12s ", op_names[i]);
        perf_print_u64(perf_calls[i]);
        putchar(' ');
        perf_print_u64(perf_cycles[i]);
        printf(" %4lu%s\n", (unsigned long)(total ? perf_cycles[i]*100/total : 0),
               i == OP_ENGINE_WAIT ? " (separate; may overlap other ops)" : "");
    }
    printf("total (sum of instrumented ops; excludes engine_wait): ");
    perf_print_u64(total);
    printf(" cycles; wall ");
    perf_print_u64(wall);
    puts(" cycles");
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
    hybrid_stats_print();
    if (overlap_ok) overlap_print("overlap", overlap);
    else puts("overlap: unavailable (legacy 32-bit window too long, absent counters or reset)");
}

void perf_ovl_mark(void)
{
    overlap_snapshot(&ovlw);
}

void perf_ovl_acc(uint64_t acc[4])
{
    struct overlap_snapshot end;
    uint64_t delta[4];
    if (acc[0] == UINT64_MAX) return;
    overlap_snapshot(&end);
    if (!overlap_delta(&ovlw, &end, delta)) {
        acc[0] = UINT64_MAX;
        return;
    }
    for (int i = 0; i < 4; i++) acc[i] += delta[i];
}

static void overlap_print(const char *tag, const uint64_t acc[4])
{
    uint64_t tot = acc[0] ? acc[0] : 1;
    uint64_t active = acc[1] + acc[2] - acc[3];
    uint64_t neither = acc[0] >= active ? acc[0] - active : 0;
    printf("%s: eng %lu%%  vec/dn %lu%%  both %lu%%  neither %lu%%  (of ", tag,
           (unsigned long)(acc[1]*100/tot), (unsigned long)(acc[2]*100/tot),
           (unsigned long)(acc[3]*100/tot), (unsigned long)(neither*100/tot));
    perf_print_u64(acc[0]);
#ifdef CSR_ENGINE_CSR_PERF_SNAPSHOT_ADDR
    puts(" cycles; coherent64; neither includes CPU work)");
#else
    puts(" cycles; legacy32 approximate; neither includes CPU work)");
#endif
}

void perf_ovl_report(const char *tag, const uint64_t acc[4])
{
    printf("OVLW %s ", tag);
    if (acc[0] == UINT64_MAX) puts("unavailable (invalid counter window)");
    else overlap_print("window", acc);
}

void perf_print_overlap(void)
{
    struct overlap_snapshot end;
    uint64_t delta[4];
    overlap_snapshot(&end);
    if (overlap_delta(&ovl0, &end, delta)) overlap_print("overlap", delta);
    else puts("overlap: unavailable (legacy 32-bit window too long, absent counters or reset)");
}

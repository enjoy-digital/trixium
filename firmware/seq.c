/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium descriptor sequencer driver
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "seq.h"
#include "engine.h"
#include "perf.h"

/* Descriptors */
/*-------------*/

static volatile uint32_t *desc_w(uint32_t idx)
{
    return (volatile uint32_t *)(SCRATCHPAD_BASE + SP_DESC + idx*64);
}

void seq_desc_vec(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs)
{
    seq_desc_vec_seed(idx, pc, args, n, srcs, 0);
}

// seed_abs: take the S7 seed from the engine absmax rather than the previous program's res6.
// With cmd.xred the engine does the cross-channel sum, so the absmax of the summed vector lives
// in the engine tracker instead of a reduce_acc's res6. spill ends with SMOV s6, s7, so the
// value also lands in res6 and a following norm_i32 on KRES6 sees what a reduce chain would
// have handed it.
void seq_desc_vec_seed(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs,
                       int seed_abs)
{
    volatile uint32_t *d = desc_w(idx);
    d[0] = 2u | (seed_abs ? (1u << 4) : 0u) | ((uint32_t)srcs << 16);
    d[1] = pc;
    for (int i = 0; i < 8; i++)
        d[2 + i] = (i < n) ? (uint32_t)args[i] : 0;
}

void seq_desc_vec_lut(uint32_t idx, uint16_t pc, const int32_t *args, int n,
                      uint16_t srcs, uint32_t lut_beat, int lut_kreg)
{
    volatile uint32_t *d = desc_w(idx);
    seq_desc_vec_seed(idx, pc, args, n, srcs, 0);
    d[0] |= 1u << 12;                          // A5 from the LUT.
    d[1] |= (uint32_t)(lut_kreg & 3) << 26;    // A5's kreg select = the table index term.
    d[10] = lut_beat;
}

void seq_desc_engine(uint32_t idx, const struct engine_op *op)
{
    volatile uint32_t *d = desc_w(idx);
    d[0]  = 1;
    d[1]  = op->kind | ((uint32_t)op->ilv << 8) | ((uint32_t)op->supw << 9)
                     | ((uint32_t)op->rshift_rpre << 10) | ((uint32_t)op->qn << 11)
                     | ((uint32_t)op->xred << 12);
    d[2]  = op->rows | ((uint32_t)op->rem << 16);
    d[3]  = op->cols;
    d[4]  = op->wbase;
    d[5]  = op->wlen;
    d[6]  = op->sbeats | ((uint32_t)op->cbeats << 8) | ((uint32_t)op->qstride << 16);
    d[7]  = op->bias | ((uint32_t)op->rshift << 16) | ((uint32_t)op->rshift2 << 22);
    d[8]  = op->xsrc;
    d[9]  = op->dst;
    d[10] = op->chan;
    d[11] = op->rep;
    d[12] = op->dstride;
    d[13] = op->sstride;
}

void seq_desc_stop(uint32_t idx)
{
    desc_w(idx)[0] = 0;
}

/* Lists */
/*-------*/

int seq_wedged; // Sticky: a list never finished, so the engine state is not trustworthy.

uint32_t seq_poll_spins, seq_poll_waits;

// Split form (per-cycle sideband slot): seq_list_start() starts a list and returns; the CPU
// may then patch other lists' descriptors or do main_ram work while it runs, and seq_join()
// polls completion. Without TRIXIUM_SP_SB_SLOT the first scratchpad access after
// seq_list_start() stalls until the list finishes: correct either way, only the overlap is
// lost.
void seq_join(void)
{
    if (seq_wedged)
        return;
    uint32_t spins = 0, reports = 0;
    seq_poll_waits++;
    for (;;) {
        uint32_t st = trixium_seq_status_read();
        tprof_dcount(st);
        if (!(st & 1))
            break;
        seq_poll_spins++;
        if (++spins < 200000u)
            continue;
        spins = 0;
        printf("SEQSTALL dcount=%lu eng=%02lx\n",
               (unsigned long)(trixium_seq_status_read() >> 8),
               (unsigned long)trixium_engine_csr_status_read());
        if (++reports == 8) {
            seq_wedged = 1;
            engine_wedged = 1;
            puts("SEQSTALL: giving up - the generation aborts and the engine cluster is "
                 "soft-reset (engreset); the weights stay.");
            return;
        }
    }
}

// Traced run: like seq_run, but records perf_now() at every dcount transition
// (seq_status[8:...]) into ts[0..max-1]; ts[0] at start, last entry at done. Returns the
// entries used. Resolution is one CSR poll (~70 cycles).
int seq_run_trace(uint32_t idx0, uint32_t count, uint32_t *ts, int max)
{
    if (seq_wedged)
        return 0;
    int n = 0;
    uint32_t last = 0xFFFFFFFF, spins = 0;
    if (n < max)
        ts[n++] = perf_now();
    trixium_seq_base_write(SP_DESC + idx0*64);
    trixium_seq_count_write(count);
    trixium_seq_start_write(1);
    for (;;) {
        uint32_t st = trixium_seq_status_read();
        uint32_t dc = st >> 8;
        if (dc != last) {
            last = dc;
            if (n < max)
                ts[n++] = perf_now();
        }
        if (!(st & 1))
            break;
        if (++spins > 4000000u) { // Wedge guard (diagnostic path: just stop).
            seq_wedged = 1;
            engine_wedged = 1;
            puts("SEQSTALL (trace): giving up - reconfigure to recover.");
            break;
        }
    }
    if (n < max)
        ts[n++] = perf_now();
    return n;
}

void seq_list_start(uint32_t idx0, uint32_t count)
{
    if (seq_wedged)
        return;
    if (tprof_active)
        tprof_event((5u << 28) | ((idx0 & 0xFF) << 8) | (count & 0xFF));
    trixium_seq_base_write(SP_DESC + idx0*64);
    trixium_seq_count_write(count);
    trixium_seq_start_write(1);
}

void seq_run(uint32_t idx0, uint32_t count)
{
    if (seq_wedged) // Nothing will complete until the FPGA is reconfigured.
        return;
    trixium_seq_base_write(SP_DESC + idx0*64);
    trixium_seq_count_write(count);
    trixium_seq_start_write(1);
    // Spin-capped poll: a wedged list names the stuck descriptor instead of hanging, and
    // gives up: a stalled list keeps `busy` and the scratchpad grant (status bit 9), so the
    // CPU's scratchpad reads return garbage either way; giving up keeps the console alive.
    uint32_t spins = 0, reports = 0;
    seq_poll_waits++;
    while (trixium_seq_status_read() & 1) {
        seq_poll_spins++;
        if (++spins < 200000u)
            continue;
        spins = 0;
        printf("SEQSTALL dcount=%lu/%lu eng=%02lx\n",
               (unsigned long)(trixium_seq_status_read() >> 8), (unsigned long)count,
               (unsigned long)trixium_engine_csr_status_read());
        if (++reports == 8) {
            seq_wedged = 1;
            engine_wedged = 1;
            puts("SEQSTALL: giving up - the generation aborts and the engine cluster is "
                 "soft-reset (engreset); the weights stay.");
            return;
        }
    }
}


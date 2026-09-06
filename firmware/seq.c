// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "seq.h"
#include "perf.h"

#ifdef CSR_SEQ_BASE

static volatile uint32_t *desc_w(uint32_t idx)
{
    return (volatile uint32_t *)(SCRATCHPAD_BASE + SP_DESC + idx*64);
}

void seq_desc_vec(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs)
{
    seq_desc_vec_seed(idx, pc, args, n, srcs, 0);
}

// seed_abs: take the S7 seed from the ENGINE ABSMAX rather than the previous program's res6.
// Needed once cmd.xred does the cross-channel sum in hardware: the absmax of the summed
// vector then lives in the engine tracker instead of a reduce_acc's res6. spill ends with
// SMOV s6, s7, so the value also lands in res6 and a following norm_i32 on KRES6 sees
// exactly what the reduce chain used to hand it.
void seq_desc_vec_seed(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs,
                       int seed_abs)
{
    volatile uint32_t *d = desc_w(idx);
    d[0] = 2u | (seed_abs ? (1u << 4) : 0u) | ((uint32_t)srcs << 16);
    d[1] = pc;
    for (int i = 0; i < 8; i++)
        d[2 + i] = (i < n) ? (uint32_t)args[i] : 0;
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

void seq_patch(uint32_t idx, int word, uint32_t value)
{
    desc_w(idx)[word] = value;
}

// Posted-write ordering mode (bitstreams carrying the sbmode CSR only). 0 = legacy: the
// descriptor fetch stalls while any CPU patch is still in the scratchpad's posted-write
// FIFO. 1 = the sequencer drains that FIFO once at list START instead, so a patch burst
// no longer starves a RUNNING list. Mode 1 is only correct while firmware keeps the
// invariant recorded in gateware/sequencer.py: never patch a list's descriptors while
// THAT list is running (patching other lists under a running one is the design's whole
// point and is safe in both modes). Default legacy, so a bitstream boots into the
// behaviour every existing measurement was taken under.
int seq_sbdrain_use;

void seq_sbmode_set(int on)
{
    seq_sbdrain_use = on ? 1 : 0;
#ifdef CSR_SEQ_SBMODE_ADDR
    seq_sbmode_write(seq_sbdrain_use);
#endif
}

int seq_sbmode_available(void)
{
#ifdef CSR_SEQ_SBMODE_ADDR
    return 1;
#else
    return 0;
#endif
}

int seq_wedged; // Sticky: a list never finished, so the engine state is not trustworthy.

uint32_t seq_poll_spins, seq_poll_waits;

// Split form (2026-08-31, per-cycle sideband slot): start a list and return; the CPU
// may then patch OTHER lists' descriptors or do host-memory work while it runs, and
// seq_join() polls completion. On bitstreams without BONSAI_SP_SB_SLOT the first
// scratchpad access after seq_list_start() simply stalls until the list finishes -
// correct either way, only the overlap is lost.
void seq_join(void)
{
    if (seq_wedged)
        return;
    uint32_t spins = 0, reports = 0;
    seq_poll_waits++;
    for (;;) {
        uint32_t st = seq_status_read();
        tprof_dcount(st);
        if (!(st & 1))
            break;
        seq_poll_spins++;
        if (++spins < 200000u)
            continue;
        spins = 0;
        printf("SEQSTALL dcount=%lu eng=%02lx\n",
               (unsigned long)(seq_status_read() >> 8),
               (unsigned long)engine_csr_status_read());
        if (++reports == 8) {
            seq_wedged = 1;
            puts("SEQSTALL: giving up. The engine is stuck mid-op and holds the scratchpad;\n"
                 "results from here are garbage. Reconfigure the FPGA to recover (a CPU\n"
                 "reset will not do it) - and note this wipes HBM, so reupload the weights.");
            return;
        }
    }
}

// Traced run (C2-1a): like seq_run, but records perf_now() at every dcount transition
// (seq_status[8:...]) into ts[0..max-1]; ts[0] at start, last entry at done. Returns
// entries used. ~70-cycle CSR-poll resolution vs multi-thousand-cycle descriptors.
int seq_run_trace(uint32_t idx0, uint32_t count, uint32_t *ts, int max)
{
    if (seq_wedged)
        return 0;
    int n = 0;
    uint32_t last = 0xFFFFFFFF, spins = 0;
    if (n < max) ts[n++] = perf_now();
    seq_base_write(SP_DESC + idx0*64);
    seq_count_write(count);
    seq_start_write(1);
    for (;;) {
        uint32_t st = seq_status_read();
        uint32_t dc = st >> 8;
        if (dc != last) {
            last = dc;
            if (n < max) ts[n++] = perf_now();
        }
        if (!(st & 1))
            break;
        if (++spins > 4000000u) { // Wedge guard (diagnostic path: just stop).
            seq_wedged = 1;
            puts("SEQSTALL (trace): giving up - reconfigure to recover.");
            break;
        }
    }
    if (n < max) ts[n++] = perf_now();
    return n;
}

void seq_list_start(uint32_t idx0, uint32_t count)
{
    if (seq_wedged)
        return;
    if (tprof_active)
        tprof_event((5u << 28) | ((idx0 & 0xFF) << 8) | (count & 0xFF));
    seq_base_write(SP_DESC + idx0*64);
    seq_count_write(count);
    seq_start_write(1);
}

void seq_run(uint32_t idx0, uint32_t count)
{
    if (seq_wedged) // Nothing will complete until the FPGA is reconfigured.
        return;
    seq_base_write(SP_DESC + idx0*64);
    seq_count_write(count);
    seq_start_write(1);
    // Spin-capped poll: a wedged list names the stuck descriptor instead of hanging.
    //
    // And it does have to *give up*. Spinning forever here is what turned a stalled
    // descriptor into a dead card: the sequencer holds `busy`, the engine keeps the
    // scratchpad grant (status bit 9), so the CPU's scratchpad reads come back garbage -
    // SPTEST reports 64 errors - and a CPU reset does not clear either. Only reconfiguring
    // the FPGA does. Giving up at least keeps the console alive to say so.
    uint32_t spins = 0, reports = 0;
    seq_poll_waits++;
    while (seq_status_read() & 1) {
        seq_poll_spins++;
        if (++spins < 200000u)
            continue;
        spins = 0;
        printf("SEQSTALL dcount=%lu/%lu eng=%02lx\n",
               (unsigned long)(seq_status_read() >> 8), (unsigned long)count,
               (unsigned long)engine_csr_status_read());
        if (++reports == 8) {
            seq_wedged = 1;
            puts("SEQSTALL: giving up. The engine is stuck mid-op and holds the scratchpad;\n"
                 "results from here are garbage. Reconfigure the FPGA to recover (a CPU\n"
                 "reset will not do it) - and note this wipes HBM, so reupload the weights.");
            return;
        }
    }
}

#endif

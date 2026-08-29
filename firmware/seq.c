// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "seq.h"

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
}

void seq_desc_stop(uint32_t idx)
{
    desc_w(idx)[0] = 0;
}

void seq_patch(uint32_t idx, int word, uint32_t value)
{
    desc_w(idx)[word] = value;
}

int seq_wedged; // Sticky: a list never finished, so the engine state is not trustworthy.

uint32_t seq_poll_spins, seq_poll_waits;

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

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
    volatile uint32_t *d = desc_w(idx);
    d[0] = 2u | ((uint32_t)srcs << 16);
    d[1] = pc;
    for (int i = 0; i < 8; i++)
        d[2 + i] = (i < n) ? (uint32_t)args[i] : 0;
}

void seq_desc_engine(uint32_t idx, const struct engine_op *op)
{
    volatile uint32_t *d = desc_w(idx);
    d[0]  = 1;
    d[1]  = op->kind | ((uint32_t)op->ilv << 8) | ((uint32_t)op->supw << 9)
                     | ((uint32_t)op->rshift_rpre << 10) | ((uint32_t)op->qn << 11);
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

void seq_run(uint32_t idx0, uint32_t count)
{
    seq_base_write(SP_DESC + idx0*64);
    seq_count_write(count);
    seq_start_write(1);
    // Spin-capped poll: a wedged list names the stuck descriptor instead of hanging.
    uint32_t spins = 0;
    while (seq_status_read() & 1) {
        if (++spins == 200000u) {
            printf("SEQSTALL dcount=%lu/%lu eng=%02lx\n",
                   (unsigned long)(seq_status_read() >> 8), (unsigned long)count,
                   (unsigned long)engine_csr_status_read());
            spins = 0;
        }
    }
}

#endif

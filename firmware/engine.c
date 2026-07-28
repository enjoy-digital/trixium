// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Engine CSR driver: program one op, submit, poll completion.

#include <generated/csr.h>

#include "engine.h"
#include "perf.h"

void engine_run(const struct engine_op *op)
{
    perf_enter(OP_ENGINE_WAIT);
    uint32_t done = engine_csr_done_count_read();

    engine_csr_kind_write(op->kind);
    engine_csr_rows_write(op->rows);
    engine_csr_cols_write(op->cols);
    engine_csr_wbase_write(op->wbase);
    engine_csr_wlen_write(op->wlen);
    engine_csr_sbeats_write(op->sbeats);
    engine_csr_cbeats_write(op->cbeats);
    engine_csr_bias_write(op->bias);
    engine_csr_rshift_write(op->rshift);
    engine_csr_xsrc_write(op->xsrc);
    engine_csr_dst_write(op->dst);
    engine_csr_chan_write(op->chan);
    engine_csr_rem_write(op->rem);
    engine_csr_ilv_write(op->ilv);
    engine_csr_supw_write(op->supw);
    engine_csr_submit_write(1);

    while (engine_csr_done_count_read() == done);
    perf_exit(OP_ENGINE_WAIT);
}

void engine_fetch(uint8_t chan, uint32_t addr, uint32_t beats, uint32_t dst)
{
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_FETCH;
    op.chan  = chan;
    op.wbase = addr;
    op.wlen  = beats;
    op.dst   = dst;
    engine_run(&op);
}

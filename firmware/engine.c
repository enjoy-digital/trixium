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

    // Delta CSR writes (nothing else touches the engine op registers).
    static struct engine_op c;
    static int c_valid;
#define EW(field, wrfn) do { \
        if (!c_valid || c.field != op->field) { wrfn(op->field); c.field = op->field; } \
    } while (0)
    EW(kind,   engine_csr_kind_write);
    EW(rows,   engine_csr_rows_write);
    EW(cols,   engine_csr_cols_write);
    EW(wbase,  engine_csr_wbase_write);
    EW(wlen,   engine_csr_wlen_write);
    EW(sbeats, engine_csr_sbeats_write);
    EW(cbeats, engine_csr_cbeats_write);
    EW(bias,   engine_csr_bias_write);
    EW(rshift, engine_csr_rshift_write);
    EW(xsrc,   engine_csr_xsrc_write);
    EW(dst,    engine_csr_dst_write);
    EW(chan,   engine_csr_chan_write);
    EW(rem,    engine_csr_rem_write);
    EW(ilv,    engine_csr_ilv_write);
    EW(supw,   engine_csr_supw_write);
#undef EW
    c_valid = 1;
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

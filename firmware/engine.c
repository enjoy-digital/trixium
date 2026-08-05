// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Engine CSR driver: program one op, submit, poll completion.

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "engine.h"

#include "perf.h"

int engine_wedged; // Sticky: an op never completed, so engine results are not trustworthy.

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
#ifdef CSR_ENGINE_CSR_QN_ADDR
    engine_csr_qn_write(op->qn);
    engine_csr_qstride_write(op->qstride);
    engine_csr_rshift2_write(op->rshift2);
#endif
    // Snapshot each reader's beat counter before the op: on a stall the deltas say exactly
    // which channel came up short, which is the one fact the status bits cannot give.
    // Expected delta per channel = wlen + (chan < rem ? sbeats + cbeats : 0).
    uint32_t beats0[BONSAI_N_CHANNELS];
    for (int c = 0; c < BONSAI_N_CHANNELS; c++) {
        engine_csr_perf_chan_write(c);
        beats0[c] = engine_csr_perf_beats_read();
    }

    // Mark the result slots so a stall shows which ones were retired. The interleaved
    // gather writes slot = row*n_channels + chan, so the pattern says exactly which
    // (row, channel) never produced a result.
    volatile uint32_t *rslot = (volatile uint32_t *)(SCRATCHPAD_BASE + op->dst);
    uint32_t nslot = 0;
    if (op->kind == 1 && op->ilv) {
        nslot = op->rows*BONSAI_N_CHANNELS + op->rem;
        if (nslot > 64) nslot = 64;
        for (uint32_t i = 0; i < nslot; i++)
            rslot[i] = 0xDEADBEEFu;
    }

    engine_csr_submit_write(1);

    // Spin-capped poll: a wedged op names itself (status + stream counters) instead of
    // hanging the boot silently.
    uint32_t spins = 0, reports = 0;
    while (engine_csr_done_count_read() == done) {
        if (++spins < 200000u)
            continue;
        spins = 0;
        // The op's own parameters, because a stall is nearly always a parameter the
        // firmware computed rather than the datapath: the same shapes pass the Migen tests
        // when the parameters are supplied by construction.
        printf("ENGSTALL st=%02lx seq=%02lx kind=%d rows=%lu rem=%lu cols=%lu wlen=%lu "
               "wbase=%08lx sb=%lu cb=%lu bias=%ld rsh=%ld ilv=%d qn=%d xsrc=%lx dst=%lx",
               (unsigned long)engine_csr_status_read(),
               (unsigned long)seq_status_read(), op->kind,
               (unsigned long)op->rows, (unsigned long)op->rem, (unsigned long)op->cols,
               (unsigned long)op->wlen, (unsigned long)op->wbase,
               (unsigned long)op->sbeats, (unsigned long)op->cbeats,
               (long)op->bias, (long)op->rshift, op->ilv, op->qn,
               (unsigned long)op->xsrc, (unsigned long)op->dst);
        for (int c = 0; c < BONSAI_N_CHANNELS; c++) {
            engine_csr_perf_chan_write(c);
            uint32_t got  = engine_csr_perf_beats_read() - beats0[c];
            uint32_t want = op->wlen + ((uint32_t)c < op->rem
                                        ? (uint32_t)op->sbeats + op->cbeats : 0);
            printf(" ch%d=%lu/%lu%s", c, (unsigned long)got, (unsigned long)want,
                   (got == want) ? "" : " <<");
        }
        if (nslot) { // '.' = retired, 'X' = still the marker (no result).
            printf(" slots=");
            for (uint32_t i = 0; i < nslot; i++)
                printf("%c", (rslot[i] == 0xDEADBEEFu) ? 'X' : '.');
        }
        printf("\n");
        // Give up rather than spin forever. A stuck op holds the scratchpad grant, so the
        // CPU's own scratchpad reads return garbage from here on and nothing downstream is
        // trustworthy - but spinning turns that into a card that answers nothing at all,
        // and only reconfiguring the FPGA recovers either way.
        if (++reports == 8) {
            engine_wedged = 1;
            puts("ENGSTALL: giving up. Reconfigure the FPGA to recover (a CPU reset will "
                 "not do it), and reupload the weights - HBM is wiped with it.");
            break;
        }
    }
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

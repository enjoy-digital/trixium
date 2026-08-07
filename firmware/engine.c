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
// Per-op stall diagnostics (pre-op beat snapshots, result-slot markers) cost ~2k cycles of
// CSR/scratchpad traffic per engine op - ~1.3M cycles per forward at ~760 ops. They were
// indispensable while the gather wedge was live; now they are opt-in (`engdiag on`). The
// stall printout itself stays: without the snapshot it shows absolute beat counters, which
// still names the channel that came up short.
int engine_diag;

// Shadow of the last-written op CSRs: a CSR access costs ~70 cycles on this SoC (measured
// via vbench: a minimal vec_run of ~8 accesses is ~570 cycles), and the token loop issues
// ~760 engine ops per forward whose fields mostly repeat - consecutive KV stores differ
// only in wbase/chan, a layer's matvecs share every shape field. Writing only what changed
// cuts an op's programming cost from ~13 accesses to the few that moved. The shadow starts
// invalid (force = every field written once) and the CSRs are only ever written here or by
// the sequencer mux - which drives the engine's command wires directly, not the CSRs - so
// the shadow cannot go stale.
static struct engine_op eng_shadow;
static int eng_shadow_valid;

#define ENG_WR(field) do { \
    if (!eng_shadow_valid || eng_shadow.field != op->field) { \
        engine_csr_##field##_write(op->field); \
        eng_shadow.field = op->field; \
    } } while (0)

void engine_run(const struct engine_op *op)
{
    perf_enter(OP_ENGINE_WAIT);
    uint32_t done = engine_csr_done_count_read();

    ENG_WR(kind);
    ENG_WR(rows);
    ENG_WR(cols);
    ENG_WR(wbase);
    ENG_WR(wlen);
    ENG_WR(sbeats);
    ENG_WR(cbeats);
    ENG_WR(bias);
    ENG_WR(rshift);
    ENG_WR(xsrc);
    ENG_WR(dst);
    ENG_WR(chan);
    ENG_WR(rem);
    ENG_WR(ilv);
    ENG_WR(supw);
#ifdef CSR_ENGINE_CSR_QN_ADDR
    ENG_WR(qn);
    ENG_WR(qstride);
    ENG_WR(rshift2);
#endif
#ifdef CSR_ENGINE_CSR_XRED_ADDR
    // Must be written unconditionally, not only when set: the CSR is sticky, so an op that
    // left xred=1 would silently reshape the next AXPY's result layout.
    ENG_WR(xred);
#endif
    // Snapshot each reader's beat counter before the op: on a stall the deltas say exactly
    // which channel came up short, which is the one fact the status bits cannot give.
    // Expected delta per channel = wlen + (chan < rem ? sbeats + cbeats : 0).
    uint32_t beats0[BONSAI_N_CHANNELS];
    if (engine_diag) {
        for (int c = 0; c < BONSAI_N_CHANNELS; c++) {
            engine_csr_perf_chan_write(c);
            beats0[c] = engine_csr_perf_beats_read();
        }
    } else {
        for (int c = 0; c < BONSAI_N_CHANNELS; c++)
            beats0[c] = 0; // Stall print falls back to absolute counters.
    }

    // Mark the result slots so a stall shows which ones were retired. The interleaved
    // gather writes slot = row*n_channels + chan, so the pattern says exactly which
    // (row, channel) never produced a result.
    volatile uint32_t *rslot = (volatile uint32_t *)(SCRATCHPAD_BASE + op->dst);
    uint32_t nslot = 0;
    if (engine_diag && op->kind == 1 && op->ilv) {
        nslot = op->rows*BONSAI_N_CHANNELS + op->rem;
        if (nslot > 64) nslot = 64;
        for (uint32_t i = 0; i < nslot; i++)
            rslot[i] = 0xDEADBEEFu;
    }

    engine_csr_submit_write(1);
    eng_shadow_valid = 1;

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

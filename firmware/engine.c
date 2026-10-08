/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium engine driver: program one op, submit, poll completion
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "engine.h"
#include "perf.h"

/* Variables */
/*-----------*/

int engine_wedged; // Sticky: an op never completed, so engine results are not trustworthy.

/* Recovery */
/*----------*/

// Soft reset of the engine cluster (engine, loader, sequencer, vector unit): the `eng` clock
// domain, trixium/core/reset.py. The HBM ports stay whole behind their AXI shims; the
// scratchpad, the vector programs and the descriptor lists survive. Returns 0 when the ports
// drained, -1 when a port never answers (the shim's `stuck`: the FPGA needs a reconfigure) or
// the drain timed out.
int engine_recover(void)
{
    trixium_engine_csr_soft_reset_write(1);
    trixium_engine_csr_soft_reset_write(0);
    uint32_t spins = 0;
    while (trixium_engine_csr_rst_status_read() & 1) {  // busy: hold + drain
        if (trixium_engine_csr_rst_status_read() & 2) { // stuck: a burst that never returns
            puts("ENGRESET: a port did not drain (a burst the HBM never answered) - "
                 "reconfigure the FPGA");
            return -1;
        }
        if (++spins > 2000000u) {
            puts("ENGRESET: drain timeout - reconfigure the FPGA");
            return -1;
        }
    }
    engine_wedged = 0;
    extern int seq_wedged;
    seq_wedged = 0;
    // The op CSRs live in the never-reset CSR front, so the shadow stays valid.
    puts("ENGRESET ok");
    return 0;
}

/* Submit / Join */
/*---------------*/

// Per-op stall diagnostics (pre-op beat snapshots, result-slot markers) cost ~2k cycles of
// CSR/scratchpad traffic per engine op, so they are opt-in (`engdiag on`). The stall printout
// itself stays: without the snapshot it shows absolute beat counters, which still name the
// channel that came up short.
int engine_diag;

// Completion-poll census (spins x ~70 cycles per CSR read: the IRQ opportunity, upper bound).
uint32_t eng_poll_spins, eng_poll_waits;

// Shadow of the last-written op CSRs: a CSR access costs ~70 cycles and consecutive ops mostly
// repeat their fields (KV stores differ only in wbase/chan, a layer's matvecs share every
// shape field), so only the fields that changed are written. The shadow starts invalid (every
// field written once) and the CSRs are only written here (the sequencer drives the engine's
// command wires directly, not the CSRs), so it cannot go stale.
static struct engine_op eng_shadow;
static int eng_shadow_valid;

#define ENG_WR(field) do { \
    if (!eng_shadow_valid || eng_shadow.field != op->field) { \
        trixium_engine_csr_##field##_write(op->field); \
        eng_shadow.field = op->field; \
    } } while (0)

// engine_submit programs and launches the op and returns the done_count to hand to
// engine_join; the caller may run independent vector-unit work in between (the scratchpad
// ports arbitrate per cycle, docs/overlap-design.md). Between submit and join:
//  - do not read the engine absmax/argmax (they belong to the streaming op),
//  - do not submit another engine op,
//  - only touch scratchpad regions disjoint from the op's xsrc/dst,
//  - CPU sideband scratchpad accesses stall until the engine idles, so keep the in-between
//    work on the vector unit (op_on_hw), not in CPU loops.
static uint32_t eng_beats0[TRIXIUM_N_CHANNELS];
static uint32_t eng_nslot;

uint32_t engine_submit(const struct engine_op *op)
{
    uint32_t done = trixium_engine_csr_done_count_read();

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
#ifdef CSR_TRIXIUM_ENGINE_CSR_QN_ADDR
    ENG_WR(qn);
    ENG_WR(qstride);
    ENG_WR(rshift2);
#endif
#ifdef CSR_TRIXIUM_ENGINE_CSR_REP_ADDR
    ENG_WR(rep);
    ENG_WR(dstride);
    ENG_WR(sstride);
#endif
#ifdef CSR_TRIXIUM_ENGINE_CSR_XRED_ADDR
    // Written on every op, not only when set: the CSR is sticky, so an op that left xred=1
    // would silently reshape the next AXPY's result layout.
    ENG_WR(xred);
#endif
    // Snapshot each reader's beat counter before the op: on a stall the deltas say which
    // channel came up short, which the status bits cannot tell.
    // Expected delta per channel = wlen + (chan < rem ? sbeats + cbeats : 0).
    if (engine_diag) {
        for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
            trixium_engine_csr_perf_chan_write(c);
            eng_beats0[c] = trixium_engine_csr_perf_beats_read();
        }
    } else {
        for (int c = 0; c < TRIXIUM_N_CHANNELS; c++)
            eng_beats0[c] = 0; // Stall print falls back to absolute counters.
    }

    // Mark the result slots so a stall shows which ones were retired. The interleaved
    // gather writes slot = row*n_channels + chan, so the pattern names the (row, channel)
    // that never produced a result.
    volatile uint32_t *rslot = (volatile uint32_t *)(SCRATCHPAD_BASE + op->dst);
    eng_nslot = 0;
    if (engine_diag && op->kind == ENGINE_KIND_Q8 && op->ilv) {
        eng_nslot = op->rows*TRIXIUM_N_CHANNELS + op->rem;
        if (eng_nslot > 64) eng_nslot = 64;
        for (uint32_t i = 0; i < eng_nslot; i++)
            rslot[i] = 0xDEADBEEFu;
    }

    trixium_engine_csr_submit_write(1);
    eng_shadow_valid = 1;
    return done;
}

void engine_join(uint32_t done)
{
    const struct engine_op *op = &eng_shadow; // Valid: submit always fills the shadow.
    volatile uint32_t *rslot = (volatile uint32_t *)(SCRATCHPAD_BASE + op->dst);
    uint32_t nslot = eng_nslot;
    uint32_t *beats0 = eng_beats0;

    // Spin-capped poll: a wedged op names itself (status + stream counters) instead of
    // hanging silently. 200k spins (~4M cycles) covers every op at the HBM's rate; ops
    // streaming more than 100k beats per channel (an 8B LM head on DDR4) scale the threshold.
    uint32_t spin_limit = 200000u;
    if (op->wlen > 100000u)
        spin_limit = op->wlen*2u;
    uint32_t spins = 0, reports = 0;
    eng_poll_waits++;
    while (trixium_engine_csr_done_count_read() == done) {
        eng_poll_spins++;
        if (++spins < spin_limit)
            continue;
        spins = 0;
        // Print the op's own parameters: a stall is nearly always a parameter the firmware
        // computed rather than the datapath.
        printf("ENGSTALL st=%02lx seq=%02lx kind=%d rows=%lu rem=%lu cols=%lu wlen=%lu "
               "wbase=%08lx sb=%lu cb=%lu bias=%ld rsh=%ld ilv=%d qn=%d xsrc=%lx dst=%lx",
               (unsigned long)trixium_engine_csr_status_read(),
               (unsigned long)trixium_seq_status_read(), op->kind,
               (unsigned long)op->rows, (unsigned long)op->rem, (unsigned long)op->cols,
               (unsigned long)op->wlen, (unsigned long)op->wbase,
               (unsigned long)op->sbeats, (unsigned long)op->cbeats,
               (long)op->bias, (long)op->rshift, op->ilv, op->qn,
               (unsigned long)op->xsrc, (unsigned long)op->dst);
        for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
            trixium_engine_csr_perf_chan_write(c);
            uint32_t got  = trixium_engine_csr_perf_beats_read() - beats0[c];
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
        // Give up rather than spin forever: a stuck op holds the scratchpad grant, so nothing
        // downstream is trustworthy, and spinning would leave a card that answers nothing.
        if (++reports == 8) {
            engine_wedged = 1;
            puts("ENGSTALL: giving up - the generation aborts and the engine is soft-reset "
                 "(engreset); the weights stay.");
            break;
        }
    }
}

void engine_run(const struct engine_op *op)
{
    perf_enter(OP_ENGINE_WAIT);
    engine_join(engine_submit(op));
    perf_exit(OP_ENGINE_WAIT);
}

/* Fetch / Store */
/*---------------*/

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

void engine_store(uint8_t chan, uint32_t addr, uint32_t beats, uint32_t src)
{
    struct engine_op op = {0};
    op.kind  = ENGINE_KIND_STORE;
    op.chan  = chan;
    op.wbase = addr;
    op.wlen  = beats;
    op.xsrc  = src;
    engine_run(&op);
}

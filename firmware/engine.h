/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium engine driver
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>

/* Op kinds */
/*----------*/

#define ENGINE_KIND_TERNARY  0
#define ENGINE_KIND_Q8       1
#define ENGINE_KIND_FETCH    2
#define ENGINE_KIND_STORE    3
#define ENGINE_KIND_AXPY     4
#define ENGINE_KIND_DNSTREAM 5 // DeltaNet stream (TRIXIUM_HAS_DNSTREAM): rows = heads,
                               // xsrc = head records, wbase/wlen = state span.
#define ENGINE_KIND_ATTN     6 // Attention passes (TRIXIUM_HAS_ATTNSTREAM): xsrc = the unit's
                               // record, K/V rows position-striped (rows/rem, cbeats per row).

// Scratchpad CPU window (uncached; generated/mem.h defines it too when included first).
#ifndef SCRATCHPAD_BASE
#define SCRATCHPAD_BASE 0xa0000000UL
#endif

/* Op */
/*----*/

struct engine_op {
    uint8_t  kind;
    uint16_t rows;        // Rows per channel.
    uint8_t  rem;         // Channels < rem stream one extra row (striped KV).
    uint8_t  ilv;         // Interleaved (position-major) gather.
    uint8_t  supw;        // Suppress scratchpad result writes (argmax tap only).
    uint16_t cols;
    uint32_t wbase;       // Per-channel weight byte address.
    uint32_t wlen;        // Beats per channel.
    uint8_t  sbeats;
    uint8_t  cbeats;
    uint8_t  bias;
    uint8_t  rshift;
    uint8_t  rshift_rpre; // Sequencer only: rshift = bitlen(prev vec res6) + 5 (AXPY Rpre).
    uint8_t  qn;          // Reserved descriptor field: always 0.
    uint8_t  xred;        // AXPY: the engine sums the per-channel partials and writes one
                          // cols-long vector; its absmax is then the summed vector's.
    uint16_t qstride;     // Reserved descriptor field: always 0.
    uint8_t  rshift2;     // Reserved descriptor field: always 0.
    uint32_t xsrc;        // Scratchpad byte address of x8.
    uint32_t dst;         // Scratchpad byte address for results.
    uint8_t  chan;        // Fetch source channel.
    uint8_t  rep;         // STORE: packets per op (0/1 = one).
    uint32_t dstride;     // STORE: HBM beats between packets.
    uint8_t  sstride;     // STORE: scratchpad beats between source slots.
};

/* Functions */
/*-----------*/

void engine_run(const struct engine_op *op);
// Overlap split (docs/overlap-design.md): submit returns the pre-op done_count for
// engine_join; independent vector-unit work may run in between. Not for CPU loops (the
// scratchpad sideband stalls while the engine is busy).
uint32_t engine_submit(const struct engine_op *op);
void engine_join(uint32_t done);
void engine_fetch(uint8_t chan, uint32_t addr, uint32_t beats, uint32_t dst);
void engine_store(uint8_t chan, uint32_t addr, uint32_t beats, uint32_t src);

extern int engine_wedged; // Set once an op stalled (see ENGSTALL in engine.c).
int engine_recover(void); // Soft reset of the engine cluster; 0 = recovered, <0 = reconfigure.
extern int engine_diag;   // Opt-in per-op stall diagnostics (~2k cycles/op).

#endif

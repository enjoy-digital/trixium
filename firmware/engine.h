// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>

#define ENGINE_KIND_TERNARY 0
#define ENGINE_KIND_Q8      1
#define ENGINE_KIND_FETCH   2
#define ENGINE_KIND_STORE   3
#define ENGINE_KIND_AXPY    4

// Scratchpad CPU window (uncached region defined in trixium.py / bonsai_sim.py).
#define SCRATCHPAD_BASE 0xa0000000UL

struct engine_op {
    uint8_t  kind;
    uint16_t rows;      // Rows per channel.
    uint8_t  rem;       // Channels < rem stream one extra row (striped KV).
    uint8_t  ilv;       // Interleaved (position-major) gather.
    uint8_t  supw;      // Suppress scratchpad result writes (argmax tap only).
    uint16_t cols;
    uint32_t wbase;     // Per-channel weight byte address.
    uint32_t wlen;      // Beats per channel.
    uint8_t  sbeats;
    uint8_t  cbeats;
    uint8_t  bias;
    uint8_t  rshift;
    uint8_t  rshift_rpre; // Sequencer only: rshift = bitlen(prev vec res6) + 5 (AXPY Rpre).
    uint8_t  qn;          // Dual-Q GQA mode: one K/V stream serves 2 sibling Q-heads.
    uint16_t qstride;     // Head-B result offset in words.
    uint8_t  rshift2;     // Head-B AXPY pre-shift.
    uint32_t xsrc;      // Scratchpad byte address of x8.
    uint32_t dst;       // Scratchpad byte address for results.
    uint8_t  chan;      // Fetch source channel.
};

void engine_run(const struct engine_op *op);
void engine_fetch(uint8_t chan, uint32_t addr, uint32_t beats, uint32_t dst);

#endif

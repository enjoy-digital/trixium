// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Descriptor sequencer driver: build 64-byte descriptors in the scratchpad, run lists.

#ifndef SEQ_H
#define SEQ_H

#include <stdint.h>

#include "engine.h"

#define SP_DESC 0x3D000 // Descriptor region (up to 192 descriptors to 0x40000).

// Vec arg sources (2 bits per arg in the descriptor header).
#define SEQ_SRC_IMM    0
#define SEQ_SRC_KABS   1 // k = max(0, bitlen(engine absmax) - 14).
#define SEQ_SRC_RPRE   2 // bitlen(prev vec res6) + 5 (axpy_rpre).
#define SEQ_SRC_KRES6  3 // k = max(0, bitlen(prev vec res6) - 14).

void     seq_desc_vec(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs);
void     seq_desc_vec_seed(uint32_t idx, uint16_t pc, const int32_t *args, int n,
                           uint16_t srcs, int seed_abs); // S7 <- engine absmax.
// A5 = LUT[args[5] + kreg[lut_kreg]] (w0[12], TRIXIUM_SEQ_LUT bitstreams): lut_beat is the
// 32-entry table's scratchpad beat index, lut_kreg 1..3 selects kreg0..2 (0 = none).
void     seq_desc_vec_lut(uint32_t idx, uint16_t pc, const int32_t *args, int n,
                          uint16_t srcs, uint32_t lut_beat, int lut_kreg);
void     seq_desc_engine(uint32_t idx, const struct engine_op *op);
void     seq_desc_stop(uint32_t idx);
// Patch one descriptor word. Inline: it is one posted store, and it used to be a
// cross-file call at every one of the ~150 patches a layer (2026-09-08).
#include <generated/soc.h>
static inline void seq_patch(uint32_t idx, int word, uint32_t value)
{
    ((volatile uint32_t *)(SCRATCHPAD_BASE + SP_DESC + idx*64))[word] = value;
}
void     seq_run(uint32_t idx0, uint32_t count);            // Run + wait (gives up if wedged).
void     seq_list_start(uint32_t idx0, uint32_t count);     // Start only; overlap CPU work...
int      seq_run_trace(uint32_t idx0, uint32_t count,
                       uint32_t *ts, int max);              // C2-1a: per-dcount timestamps.
void     seq_join(void);                                    // ...then wait (same wedge guard).
extern int seq_wedged; // Set once a list stalled: every later seq_run() is a no-op.

#endif

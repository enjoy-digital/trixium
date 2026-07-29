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
#define SEQ_SRC_RES6   3 // prev vec res6 low 32 bits.

void     seq_desc_vec(uint32_t idx, uint16_t pc, const int32_t *args, int n, uint16_t srcs);
void     seq_desc_engine(uint32_t idx, const struct engine_op *op);
void     seq_desc_stop(uint32_t idx);
void     seq_patch(uint32_t idx, int word, uint32_t value); // Patch one descriptor word.
void     seq_run(uint32_t idx0, uint32_t count);            // Run + wait.

#endif

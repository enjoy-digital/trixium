// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#include "bons.h"

// The BONS header/table location in CPU address space and the per-channel byte offset where
// channel payloads live are provided as SoC constants (BONSAI_HDR_ADDR/BONSAI_CHAN_WEIGHT_BASE).

const struct bons_header *model_header(void);
const struct bons_entry  *model_entry(int role, int layer);
void model_info(void);

// Run a ternary/q8 matvec for one tensor: x8 at scratchpad `xsrc`, int32 results at `dst`.
void model_matvec(const struct bons_entry *e, uint32_t xsrc, uint32_t dst);

#endif

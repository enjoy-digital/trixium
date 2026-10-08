/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium BONS model access: header/table lookups and per-tensor engine ops
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#include "bons.h"

// The BONS header/table CPU address and the per-channel payload byte offset are SoC constants
// (TRIXIUM_HDR_ADDR, TRIXIUM_CHAN_WEIGHT_BASE).

/* Header / Table */
/*----------------*/

const struct bons_header *model_header(void);
const struct bons_entry  *model_entry(int role, int layer);
const struct bons_entry  *model_entry_kind(int role, int layer, int kind);
// K-segment k of a wide matvec (FLAG_KSEG); k = 0 is what model_entry returns.
const struct bons_entry  *model_entry_kseg(const struct bons_entry *e0, int k);
int                       model_entry_nkseg(const struct bons_entry *e0);
const struct bons_entry  *model_entry_at(uint32_t i); // The table itself, entry i.

void model_info(void);
void model_cache_reset(void); // Invalidate the (role, layer) lookup cache (call on hdrload).

/* Matvec */
/*--------*/

// Run a ternary/q8 matvec for one tensor: x8 at scratchpad `xsrc`, int32 results at `dst`.
void model_matvec(const struct bons_entry *e, uint32_t xsrc, uint32_t dst);
struct engine_op; // engine.h; forward-declared so this header stays include-order free.
void model_matvec_op(const struct bons_entry *e, uint32_t xsrc, uint32_t dst,
                     struct engine_op *op); // Fill without running (descriptor lists).

#endif

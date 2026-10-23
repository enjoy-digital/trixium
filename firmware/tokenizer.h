/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium on-board tokenizer
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_TOKENIZER_H
#define TRIXIUM_TOKENIZER_H

#include "trixium_tok.h"

// The host stages a BTOK file on engine channel 0, 16 MB below the weights (the BONS header's
// staging is the last 1 MB): a 32-byte record (magic, length) then the tables
// (software/host/udpload.py upload_tokenizer).
#define TOK_STAGING        (TRIXIUM_CHAN_WEIGHT_BASE - 0x1000000)
#define TOK_STAGING_MAGIC  0x534b4f54u   // "TOKS"
#define TOK_STAGING_MAX    (0x1000000 - 0x100000 - 32)

// tokload: copy the staged tables to the top 16 MB of main_ram and index them. 0 on success.
int tok_load(void);

// The loaded tokenizer, or NULL (no tokload yet, or the heap has grown into its region).
const struct ttok *tok_get(void);

#endif

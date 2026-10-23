/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium on-board tokenizer
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>
#include <string.h>

#include <generated/soc.h>
#include <generated/mem.h>
#include <system.h>

#include "bons.h"
#include "engine.h"
#include "model.h"
#include "tokenizer.h"

// The tables and their index in the top 16 MB of main_ram: the 2B's BTOK is 5.5 MB and its
// index (merge hash, vocabulary offsets, BPE scratch) 9 MB. Above the heap, which the model's
// init grows from the bottom; tok_get() checks the two never met.
#define TOK_REGION_SIZE 0x1000000u
#define TOK_REGION      (MAIN_RAM_BASE + MAIN_RAM_SIZE - TOK_REGION_SIZE)

extern uint8_t *heap, *heap_hi;   // infer.c's bump allocators.

static struct ttok tok;
static int tok_ok;

const struct ttok *tok_get(void)
{
    if (!tok_ok)
        return NULL;
    if ((uint32_t)heap > TOK_REGION || (uint32_t)heap_hi > TOK_REGION) {
        printf("tokenizer: the heap reached its region (%08lx), tokload again\n",
               (unsigned long)TOK_REGION);
        tok_ok = 0;
        return NULL;
    }
    return &tok;
}

int tok_load(void)
{
    tok_ok = 0;
#if MAIN_RAM_SIZE < 0x8000000
    puts("tokload: main_ram too small for the tables (128 MB needed)");
    return -1;
#else
    engine_fetch(0, TOK_STAGING, 1, 0xC000);
    volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
    uint32_t len = rec[1];
    if (rec[0] != TOK_STAGING_MAGIC || len < 16 || len > TOK_STAGING_MAX) {
        printf("tokload: no tokenizer staged (magic %08lx, %lu bytes)\n",
               (unsigned long)rec[0], (unsigned long)len);
        return -1;
    }
    uint8_t *blob = (uint8_t *)TOK_REGION;
    uint8_t *arena = blob + ((len + 63) & ~63u);
    if ((uint32_t)heap > TOK_REGION || (uint32_t)heap_hi > TOK_REGION) {
        printf("tokload: the heap reaches %08lx, past the tokenizer region\n",
               (unsigned long)((uint32_t)heap > (uint32_t)heap_hi ? (uint32_t)heap : (uint32_t)heap_hi));
        return -1;
    }
    // 64 KB engine fetches through the scratchpad (hdrload's window), as whole beats.
    for (uint32_t off = 0; off < len; off += 0x10000) {
        uint32_t chunk = len - off > 0x10000 ? 0x10000 : len - off;
        engine_fetch(0, TOK_STAGING + 32 + off, (chunk + 31)/32, 0xC000);
        volatile uint32_t *src = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
        for (uint32_t i = 0; i < (chunk + 3)/4; i++)      // Word reads, as hdrload's.
            ((uint32_t *)(blob + off))[i] = src[i];
    }
    size_t need = ttok_arena_size(blob, len);
    if (!need || (uint32_t)(arena + need) > MAIN_RAM_BASE + MAIN_RAM_SIZE) {
        printf("tokload: %lu bytes of tables need %lu KB of index, more than the region\n",
               (unsigned long)len, (unsigned long)(need >> 10));
        return -1;
    }
    if (ttok_init(&tok, blob, len, arena) < 0) {
        puts("tokload: the staged tables are not a BTOK file");
        return -1;
    }
    tok_ok = 1;
    printf("TOKLOAD %lu tokens, %lu merges, %lu KB", (unsigned long)tok.nvocab,
           (unsigned long)tok.nmerges, (unsigned long)((len + need) >> 10));
    // The image header carries the converter's tokenizer hash: ids from other tables would
    // still run, as the wrong words.
    const struct bons_header *h = model_header();
    if (tok.sha_ok && h && h->magic == BONS_MAGIC && h->version >= 2)
        printf(", %s the image", ttok_same(tok.sha, h->tokenizer_sha256, 32) ? "matches" : "DIFFERENT FROM");
    printf("\n");
    return 0;
#endif
}

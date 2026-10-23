/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium byte-level BPE tokenizer (Qwen3 / Qwen3.5 BTOK tables)
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 * One encoder for the host chat (software/user/trixium_chat.c) and the firmware: no stdio, no
 * allocation, no libc beyond memcpy/memmove/memset/strlen. The tables are a BTOK file
 * (model/tokenizer.py) held in memory, read in place; the caller gives an arena of
 * ttok_arena_size() bytes for the vocabulary index, the merge hash and the BPE scratch.
 *
 * The pre-tokenizer follows the Qwen/GPT-2 split with approximate Unicode letter classes and
 * no NFC normalization (test/software/test_tokenizer.py: the reference encoder's ids on the
 * repo's docs and multilingual text; decomposed accents, rare symbol classes and added-token
 * text typed by the user can differ). A chat turn adds the template's special tokens by id,
 * never by parsing text.
 */

#ifndef TRIXIUM_TOK_H
#define TRIXIUM_TOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TTOK_CHUNK_MAX 16384 /* Bytes per BPE pre-token; longer runs go in windows of this size. */

struct ttok_merge { uint64_t key; uint32_t rank, result; };

struct ttok {
    const uint8_t     *blob;     /* The BTOK vocabulary records, then the merges. */
    uint32_t           nvocab, nmerges;
    uint32_t          *voff;     /* [nvocab] offset of each record (u32 length, bytes) in blob. */
    struct ttok_merge *mtab;     /* Open addressing, (a, b) -> rank, result. */
    uint32_t           mmask;
    int32_t           *scratch;  /* [TTOK_CHUNK_MAX] BPE ids. */
    int32_t            byte_id[256];
    uint8_t            sha[32];  /* BTOK v2: the source tokenizer.json's SHA-256. */
    int                sha_ok;
    /* Chat-template tokens, -1 when the vocabulary has none (raw completion). */
    int                im_start, im_end, think, think_end;
};

/* Arena bytes ttok_init needs for these tables, 0 if they are not a BTOK file. */
size_t ttok_arena_size(const uint8_t *btok, size_t len);

/* Reads the BTOK tables in place (they must outlive t). 0 on success, -1 on a bad file. */
int ttok_init(struct ttok *t, const uint8_t *btok, size_t len, void *arena);

/* Token ids of n bytes of UTF-8 text, at most max; returns the count. */
int ttok_encode(const struct ttok *t, const uint8_t *text, size_t n, int *out, int max);

/* One user turn in the Qwen3 chat template, as the reference tokenizer encodes it:
 *   [<|im_end|>]\n            when continuing (first == 0; <|im_end|> if the reply was cut)
 *   <|im_start|>system\n{system}<|im_end|>\n                 first turn, system non-empty
 *   <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * The empty think block is the template's enable_thinking=False. Without the template tokens,
 * the text alone. */
int ttok_turn(const struct ttok *t, const char *text, const char *system, int first,
              int prev_closed, int *out, int max);

/* The bytes of token id (NULL if out of range); *len its length. */
const uint8_t *ttok_piece(const struct ttok *t, int id, uint32_t *len);

/* The highest id whose bytes are lit (added tokens are stored as their UTF-8 text), or -1. */
int ttok_find(const struct ttok *t, const char *lit);

/* 1 if two n-byte strings are equal (the firmware's libc has no memcmp). */
int ttok_same(const uint8_t *a, const uint8_t *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif

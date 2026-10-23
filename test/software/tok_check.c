/* SPDX-License-Identifier: BSD-2-Clause
 *
 * software/lib/trixium_tok.c on stdin: paragraphs separated by 0x1e, one line of ids per
 * paragraph on stdout (test_tokenizer.py compares them with the reference encoder).
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 */

#include <stdio.h>
#include <stdlib.h>

#include "trixium_tok.h"

static uint8_t *slurp(FILE *f, size_t *n)
{
    size_t cap = 1 << 20, len = 0, got;
    uint8_t *b = malloc(cap);
    while ((got = fread(b + len, 1, cap - len, f)) > 0)
        if ((len += got) == cap) b = realloc(b, cap *= 2);
    *n = len;
    return b;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: tok_check tokenizer.bin < paragraphs\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    size_t blen, tlen;
    uint8_t *btok = slurp(f, &blen);
    fclose(f);
    struct ttok t;
    void *arena = malloc(ttok_arena_size(btok, blen));
    if (!arena || ttok_init(&t, btok, blen, arena)) { fprintf(stderr, "bad BTOK\n"); return 2; }
    uint8_t *text = slurp(stdin, &tlen);
    int *ids = malloc(sizeof(int)*(tlen + 16));
    for (size_t s = 0; s <= tlen; ) {
        size_t e = s;
        while (e < tlen && text[e] != 0x1e) e++;
        int n = ttok_encode(&t, text + s, e - s, ids, (int)(e - s) + 16);
        for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
        s = e + 1;
    }
    return 0;
}

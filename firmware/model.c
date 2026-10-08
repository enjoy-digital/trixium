// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// BONS model walking: header/table access + per-tensor engine ops.

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "model.h"
#include "engine.h"

const struct bons_header *model_header(void)
{
    return (const struct bons_header *)BONSAI_HDR_ADDR;
}

static const struct bons_entry *model_table(void)
{
    return (const struct bons_entry *)(BONSAI_HDR_ADDR + model_header()->table_offset);
}

// (role, layer) -> table-index cache. The linear scan below walks up to n_tensors entries
// of the DRAM-resident table; measured on the card a token-loop rmsnorm call was 157k cycles
// against 27k for the identical call with the entry pointer in hand - the difference is this
// scan falling out of the D-cache. The loop does a dozen lookups per layer, so the scans
// were 29% of token time, hiding as "unaccounted" because no perf region covered them.
// `any` holds the first (role, layer) match in table order (model_entry semantics verbatim);
// `n16` the first NORM16 match (the only kind model_entry_kind is used with - anything else
// falls back to the scan). Rebuilt lazily; hdrload resets it when the header changes.
// NOT in .bss: the sram region is 8 KB *including the stack* (_fstack sits at its top), and
// these 4 KB of tables left it no room - the sim build failed to link and the card build
// linked with so little stack that boot corrupted itself and hung. They live instead in the
// unused main_ram window between the BONS header/table (~32 KB at BONSAI_HDR_ADDR) and the
// inference heap (BONSAI_HDR_ADDR + 0x100000, see infer.c HEAP_BASE).
// 32 roles: ROLE_TOK_EMB16_EXP is role 16 and sat just past the old bound of 16, so its
// lookup fell to the 426-entry scan (~27 KB through the D$) once every forward - 33k
// cycles of the embed window on the card (2026-09-08). 8 KB of the 64 KB window.
#define ECACHE_ROLES  32
#define ECACHE_LAYERS 64
_Static_assert(ROLE_TOK_EMB16_EXP < ECACHE_ROLES, "entry cache must cover every role");
#define ECACHE_BASE   (BONSAI_HDR_ADDR + 0x80000)
#define ecache_any ((int16_t (*)[ECACHE_LAYERS])(ECACHE_BASE))
#define ecache_n16 ((int16_t (*)[ECACHE_LAYERS])(ECACHE_BASE + \
                        ECACHE_ROLES*ECACHE_LAYERS*sizeof(int16_t)))
// K-segments: the first FLAG_KSEG entry of each (role, layer, kseg < 4), 16 KB more. Every
// K-segmented lookup used to scan the whole table (~1,100 entries on the 27B), and
// infer_matvec_cols re-counted the segments in its loop condition: ~11 scans per FFN down
// lookup, ~330k cycles per layer outside the FFN lists on the v17 board (2026-09-25).
#define ECACHE_KSEG 4
#define ecache_kseg ((int16_t (*)[ECACHE_LAYERS][ECACHE_KSEG])(ECACHE_BASE + \
                        2*ECACHE_ROLES*ECACHE_LAYERS*sizeof(int16_t)))
_Static_assert(2*ECACHE_ROLES*ECACHE_LAYERS*2 + ECACHE_ROLES*ECACHE_LAYERS*ECACHE_KSEG*2 <= 0x10000,
               "entry cache window is 64 KB");
static int ecache_ok;

void model_cache_reset(void)
{
    ecache_ok = 0;
}

static void ecache_build(void)
{
    const struct bons_header *h = model_header();
    const struct bons_entry  *t = model_table();
    for (int r = 0; r < ECACHE_ROLES; r++)
        for (int l = 0; l < ECACHE_LAYERS; l++) {
            ecache_any[r][l] = ecache_n16[r][l] = -1;
            for (int k = 0; k < ECACHE_KSEG; k++) ecache_kseg[r][l][k] = -1;
        }
    for (uint32_t i = 0; i < h->n_tensors && i < 32767; i++) {
        int r = t[i].role, l = t[i].layer;
        if (r < 0 || r >= ECACHE_ROLES || l < 0 || l >= ECACHE_LAYERS)
            continue;
        if (ecache_any[r][l] < 0)
            ecache_any[r][l] = (int16_t)i;
        if (t[i].kind == KIND_NORM16 && ecache_n16[r][l] < 0)
            ecache_n16[r][l] = (int16_t)i;
        if ((t[i].flags & FLAG_KSEG) && t[i].kseg < ECACHE_KSEG && ecache_kseg[r][l][t[i].kseg] < 0)
            ecache_kseg[r][l][t[i].kseg] = (int16_t)i;
    }
    ecache_ok = 1;
}

const struct bons_entry *model_entry(int role, int layer)
{
    const struct bons_header *h = model_header();
    const struct bons_entry  *t = model_table();
    if (h->magic == BONS_MAGIC &&
        role >= 0 && role < ECACHE_ROLES && layer >= 0 && layer < ECACHE_LAYERS) {
        if (!ecache_ok)
            ecache_build();
        int16_t i = ecache_any[role][layer];
        return (i >= 0) ? &t[i] : 0;
    }
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (t[i].role == role && t[i].layer == layer)
            return &t[i];
    return 0;
}

const struct bons_entry *model_entry_kind(int role, int layer, int kind)
{
    const struct bons_header *h = model_header();
    const struct bons_entry  *t = model_table();
    if (kind == KIND_NORM16 && h->magic == BONS_MAGIC &&
        role >= 0 && role < ECACHE_ROLES && layer >= 0 && layer < ECACHE_LAYERS) {
        if (!ecache_ok)
            ecache_build();
        int16_t i = ecache_n16[role][layer];
        return (i >= 0) ? &t[i] : 0;
    }
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (t[i].role == role && t[i].layer == layer && t[i].kind == kind)
            return &t[i];
    return 0;
}

const struct bons_entry *model_entry_at(uint32_t i)
{
    return &model_table()[i];
}

const struct bons_entry *model_entry_kseg(const struct bons_entry *e0, int k)
{
    const struct bons_header *h = model_header();
    const struct bons_entry  *t = model_table();
    if (!(e0->flags & FLAG_KSEG))
        return k == 0 ? e0 : 0;
    if (h->magic == BONS_MAGIC && e0->role < ECACHE_ROLES && e0->layer < ECACHE_LAYERS &&
        k >= 0 && k < ECACHE_KSEG) {
        if (!ecache_ok)
            ecache_build();
        int16_t i = ecache_kseg[e0->role][e0->layer][k];
        if (i < 0)
            return 0;                  // No entry of this (role, layer) has segment k.
        if (t[i].kind == e0->kind)
            return &t[i];              // The scan's first match (a different kind: scan).
    }
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (t[i].role == e0->role && t[i].layer == e0->layer && t[i].kind == e0->kind
            && (t[i].flags & FLAG_KSEG) && t[i].kseg == k)
            return &t[i];
    return 0;
}

int model_entry_nkseg(const struct bons_entry *e0)
{
    if (!(e0->flags & FLAG_KSEG))
        return 1;
    int n = 0;
    while (model_entry_kseg(e0, n))
        n++;
    return n;
}

void model_info(void)
{
    const struct bons_header *h = model_header();
    if (h->magic != BONS_MAGIC) {
        printf("no BONS image (magic %08lx)\n", (unsigned long)h->magic);
        return;
    }
    printf("BONS v%lu: %lu layers, embd %lu, heads %lu/%lu, ffn %lu, vocab %lu, "
           "%lu tensors, %lu channels, %lu beats/channel\n",
           (unsigned long)h->version, (unsigned long)h->n_layer, (unsigned long)h->n_embd,
           (unsigned long)h->n_head, (unsigned long)h->n_kv_head, (unsigned long)h->ffn_dim,
           (unsigned long)h->vocab, (unsigned long)h->n_tensors, (unsigned long)h->n_channels,
           (unsigned long)h->payload_beats);
    printf("tokenizer sha256 ");                     // The converter's --tokenizer file;
    for (int i = 0; i < 32; i++)                     // bonsai_chat compares it with its own.
        printf("%02x", h->tokenizer_sha256[i]);
    printf("\n");
}

// Fill the engine op for a tensor without running it: the FFN descriptor list rebuilds its
// three matvec descriptors from this per layer, so the encoding lives in exactly one place.
void model_matvec_op(const struct bons_entry *e, uint32_t xsrc, uint32_t dst,
                     struct engine_op *op)
{
    const struct bons_header *h = model_header();
    op->kind   = (e->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
    op->rows   = e->rows/h->n_channels;
    op->cols   = e->cols;
    op->wbase  = BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32;
    op->wlen   = e->chan_len_beats;
    op->sbeats = bons_scale_beats(e->cols, e->group);
    op->cbeats = bons_code_beats(e->cols, e->kind);
    op->bias   = e->bias;
    op->rshift = e->rshift;
    op->xsrc   = xsrc;
    op->dst    = dst;
}

void model_matvec(const struct bons_entry *e, uint32_t xsrc, uint32_t dst)
{
    struct engine_op op = {0};
    model_matvec_op(e, xsrc, dst, &op);
    engine_run(&op);
}

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

const struct bons_entry *model_entry(int role, int layer)
{
    const struct bons_header *h = model_header();
    const struct bons_entry  *t = model_table();
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (t[i].role == role && t[i].layer == layer)
            return &t[i];
    return 0;
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
}

void model_matvec(const struct bons_entry *e, uint32_t xsrc, uint32_t dst)
{
    const struct bons_header *h = model_header();
    struct engine_op op = {0};
    op.kind   = (e->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
    op.rows   = e->rows/h->n_channels;
    op.cols   = e->cols;
    op.wbase  = BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32;
    op.wlen   = e->chan_len_beats;
    op.sbeats = bons_scale_beats(e->cols, e->group);
    op.cbeats = bons_code_beats(e->cols, e->kind);
    op.bias   = e->bias;
    op.rshift = e->rshift;
    op.xsrc   = xsrc;
    op.dst    = dst;
    engine_run(&op);
}

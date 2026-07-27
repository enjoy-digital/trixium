// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// BONS weight image structures - must match trixium/layout.py exactly.

#ifndef BONS_H
#define BONS_H

#include <stdint.h>

#define BONS_MAGIC 0x534E4F42 // "BONS"

// Tensor kinds.
#define KIND_TERNARY 0
#define KIND_Q8      1
#define KIND_F16     2
#define KIND_F32     3

#define FLAG_REPLICATED 0x01

// Tensor roles.
enum {
    ROLE_TOK_EMB = 0, ROLE_OUT_NORM, ROLE_LM_HEAD,
    ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
    ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
    ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
};

struct __attribute__((packed)) bons_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t vocab;
    uint32_t rope_theta;
    float    rms_eps;
    uint32_t max_ctx;
    uint32_t n_channels;
    uint32_t n_tensors;
    uint32_t table_offset;
    uint32_t payload_offset;
    uint64_t payload_beats;
    uint32_t mode;
    uint32_t table_crc32;
    uint8_t  tokenizer_sha256[32];
};

struct __attribute__((packed)) bons_entry {
    uint8_t  role;
    uint8_t  kind;
    uint8_t  flags;
    uint8_t  pad0;
    uint16_t layer;
    uint16_t group;
    uint32_t rows;
    uint32_t cols;
    uint64_t chan_base_beats;
    uint64_t chan_len_beats;
    uint32_t crc32;
    uint8_t  bias;
    uint8_t  rshift;
    uint8_t  pad1[2];
    char     name[24];
};

// Row-packet beat math (mirrors layout.py).
static inline uint32_t bons_scale_beats(uint32_t cols, uint32_t group) {
    uint32_t n_scales = cols/group;
    return (2*n_scales + 31)/32;
}

static inline uint32_t bons_code_beats(uint32_t cols, uint8_t kind) {
    if (kind == KIND_TERNARY) return cols/128;
    if (kind == KIND_Q8)      return cols/32;
    if (kind == KIND_F16)     return (2*cols + 31)/32;
    return (4*cols + 31)/32; // F32
}

static inline uint32_t bons_row_beats(uint32_t cols, uint8_t kind, uint32_t group) {
    uint32_t sb = (kind == KIND_TERNARY || kind == KIND_Q8) ? bons_scale_beats(cols, group) : 0;
    return sb + bons_code_beats(cols, kind);
}

#endif

// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// BONS weight image structures - must match trixium/layout.py exactly.

#ifndef BONS_H
#define BONS_H

#include <stdint.h>

#define BONS_MAGIC   0x534E4F42 // "BONS"
#define BONS_STAGING_CLEARED 0x44524c43u // "CLRD": trixium_loader.c loader_clear_staging()
#define BONS_VERSION 3          // layout.BONS_VERSION; infer_init accepts 2 and 3.
                                // 2: NORM16 .w16 entries carry their exponent in bias.
                                // 3: the hybrid fields (a v2 image reads as all-zero ones).
#define BONS_VERSION_MIN 2

// Tensor kinds.
#define KIND_TERNARY 0
#define KIND_Q8      1
#define KIND_F16     2
#define KIND_F32     3
#define KIND_NORM16  4 // int16 rows at 2^-13 (layout.NORM16_EXP) for the vector unit.

#define FLAG_REPLICATED 0x01
#define FLAG_KSEG       0x02   // One K-segment of a wide matvec (layout.FLAG_KSEG): siblings
                               // share role/layer/kind/bias/rshift; sum their int32 results.

// Tensor roles.
enum {
    ROLE_TOK_EMB = 0, ROLE_OUT_NORM, ROLE_LM_HEAD,
    ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
    ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
    ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
    ROLE_ROPE_TAB,
    ROLE_TOK_EMB16, ROLE_TOK_EMB16_EXP, // Pre-dequantized int16 embedding rows + exponents.
    // The 27B's hybrid layers (BONS v3): see layout.py.
    ROLE_ATTN_QKV, ROLE_ATTN_ZGATE, ROLE_SSM_OUT, ROLE_SSM_BETA, ROLE_SSM_ALPHA,
    ROLE_SSM_CONV, ROLE_SSM_A, ROLE_SSM_DT, ROLE_SSM_NORM,
    ROLE_HADAMARD,
    // Roles 27-30 are reserved (unused).
};

#define HYB_EMB16_RESTORED 0x01   // hyb_flags: the emb16 rows are in the plain basis.

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
    // v3 (offset 112): zero on a v2 image and on a plain Qwen3 model.
    uint64_t layer_types[2];      // bit l set = layer l is Gated DeltaNet.
    uint16_t dn_num_v_heads;
    uint16_t dn_num_k_heads;
    uint16_t dn_k_head_dim;
    uint16_t dn_v_head_dim;
    uint16_t dn_conv_kernel;
    uint16_t rope_dims;           // 0 = the whole head is rotated.
    uint16_t hadamard_block;      // 0 = plain basis.
    uint8_t  attn_gated;
    uint8_t  hyb_flags;
    uint32_t hadamard_widths[4];  // ROLE_HADAMARD tensors by layer index.
    uint8_t  reserved160;         // Reserved byte, always 0.
};

_Static_assert(__builtin_offsetof(struct bons_header, layer_types) == 112, "BONS v3 fields at 112");
_Static_assert(__builtin_offsetof(struct bons_header, hadamard_widths) == 144, "BONS v3 widths at 144");
_Static_assert(__builtin_offsetof(struct bons_header, reserved160) == 160, "BONS v3 reserved byte at 160");

// ROLE_HADAMARD block tensors: entry layer = width index (header list) * 256 + block.
#define HADAMARD_LAYER(wi, b) ((int)((wi)*256 + (b)))

static inline int bons_layer_is_deltanet(const struct bons_header *h, unsigned l) {
    return (int)((h->layer_types[l >> 6] >> (l & 63)) & 1);
}

struct __attribute__((packed)) bons_entry {
    uint8_t  role;
    uint8_t  kind;
    uint8_t  flags;
    uint8_t  kseg;     // K-segment index with FLAG_KSEG (0 otherwise).
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
    // NORM16 is int16 too (layout.code_beats agrees), not F32.
    if (kind == KIND_NORM16)  return (2*cols + 31)/32;
    return (4*cols + 31)/32; // F32
}

static inline uint32_t bons_row_beats(uint32_t cols, uint8_t kind, uint32_t group) {
    uint32_t sb = (kind == KIND_TERNARY || kind == KIND_Q8) ? bons_scale_beats(cols, group) : 0;
    return sb + bons_code_beats(cols, kind);
}

#endif

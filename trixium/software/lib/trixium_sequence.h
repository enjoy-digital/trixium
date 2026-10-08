/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium sequencer descriptor encoders
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 * Pure descriptor encoders: no MMIO, allocation, board names or model policy. Words are
 * host-native uint32_t values; use bns_seq_desc_encode_le for a byte stream, or ordered 32-bit
 * MMIO stores on the target. All reserved words are zero. Failure leaves the output unchanged;
 * a NULL output is invalid.
 *
 * Before submission the caller must check each optional unit's hardware ABI, physical
 * capacity/profile, placement, numerical admission and full-list ownership: these functions
 * check field encoding, not execution validity. Addresses are scratchpad-relative bytes; the
 * hardware checks full spans and list protection. Live descriptors and input buffers must
 * remain immutable; a failure invalidates the output prefixes and requires a common
 * reset/drain. There is no implicit quantization fallback.
 */

#ifndef TRIXIUM_SEQUENCE_H
#define TRIXIUM_SEQUENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor */
/*------------*/

#define BNS_SEQ_DESC_WORDS 16u
#define BNS_SEQ_DESC_BYTES 64u

struct bns_seq_descriptor {
    uint32_t word[BNS_SEQ_DESC_WORDS];
};

enum bns_seq_desc_result {
    BNS_SEQ_DESC_OK      =  0,
    BNS_SEQ_DESC_INVALID = -1,
};

/* List end */
/*----------*/

int bns_seq_desc_stop(struct bns_seq_descriptor *out);

/* Hadamard (unit 3) */
/*-------------------*/

/* ABI 1: 1024-element int8 blocks -> raw int32 + absmax. */
int bns_seq_desc_hadamard(struct bns_seq_descriptor *out, uint32_t src, uint32_t signs,
                          uint32_t dst, uint32_t stats, uint32_t blocks, uint32_t inverse);

/* Activation quantize (units 4/5) */
/*---------------------------------*/

/* Unit 4, ABI 1: one exponent per independent FP32 block. */
int bns_seq_desc_quantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                          uint32_t exponents, uint32_t elements, uint32_t blocks);

/* Unit 5, needs activation_quantize ABI 1, minor >= 1: whole-vector post-Hadamard exponent
 * over ALL 1024-element blocks, before any projection K segmentation. */
int bns_seq_desc_requantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                            uint32_t exponents, uint32_t maxima, uint32_t result_exponent,
                            uint32_t blocks);

/* Projection restore (unit 8) */
/*-----------------------------*/

/* ABI 1: int32 -> FP32 using resident e8 + (R-B). */
int bns_seq_desc_restore(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                         uint32_t exponents, uint32_t elements, int32_t adjustment);

/* SiLU (unit 11) */
/*----------------*/

/* ABI 1: dst[i] = m[i]*silu(g[i]) bit-exact with the firmware's lookup; fixup elements keep g
 * in dst and set bit i of the fixmap bitmap (ceil(elements/32) words, all written).
 * dst == src_g or disjoint. */
int bns_seq_desc_silu(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                      uint32_t dst, uint32_t fixmap, uint32_t elements);

/* Table load: 1,025 beats at src, lut[0..4096] then dlut from beat 513. */
int bns_seq_desc_silu_load(struct bns_seq_descriptor *out, uint32_t src);

/* ABI >= 1.1, with options: unit_m (m = 1.0, src_m unused) and permuted output segments
 * (source segment s of seg_beats beats -> destination (s % a)*b + s / a). */
int bns_seq_desc_silu_ex(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                         uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t unit_m,
                         uint32_t seg_beats, uint32_t perm_a, uint32_t perm_b);

/* ABI >= 1.2, table only: dst = m * table(g) with every element outside the table range a
 * fixup (no SiLU pass-through or -0), for a table other than SiLU's (the sigmoid). */
int bns_seq_desc_silu_table(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                            uint32_t dst, uint32_t fixmap, uint32_t elements);

/* ABI >= 1.3: a second resident table pair (set 1), loaded once, selected per run. */
int bns_seq_desc_silu_table_set(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                                uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t set);
int bns_seq_desc_silu_load_set(struct bns_seq_descriptor *out, uint32_t src, uint32_t set);

/* Norm (unit 12) */
/*----------------*/

/* ABI 1: RMSNorm (l2 = 0, weight) or L2 (l2 = 1) over nseg segments of n. */
int bns_seq_desc_norm(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src, uint32_t dst,
                      uint32_t weight, uint32_t n, uint32_t nseg, uint32_t eps_bits);

/* ABI >= 1.1, with strided segments (beats between segment starts, 0 = n/8). */
int bns_seq_desc_norm_strided(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src,
                              uint32_t dst, uint32_t weight, uint32_t n, uint32_t nseg,
                              uint32_t eps_bits, uint32_t src_stride, uint32_t dst_stride);

/* ABI >= 1.2: RoPE after the norm, on a built norm descriptor. Neox pairs over the first
 * rope_n values of every segment (a multiple of 16, <= 256 and <= n), cos[rope_n/2] then
 * sin[rope_n/2] at the 32-byte-aligned scratchpad address tab (< 256 KB). */
int bns_seq_desc_norm_rope(struct bns_seq_descriptor *d, uint32_t rope_n, uint32_t tab);

/* Conv / Add (unit 13) */
/*----------------------*/

/* ABI 1: the DeltaNet 4-tap causal conv, window x0..x2, input x3, taps w. */
int bns_seq_desc_conv(struct bns_seq_descriptor *out, uint32_t x0, uint32_t x1, uint32_t x2,
                      uint32_t x3, uint32_t w, uint32_t dst, uint32_t channels);

/* Op 1: dst = a + b over n FP32 values (the residual add); dst == a or b. */
int bns_seq_desc_add(struct bns_seq_descriptor *out, uint32_t a, uint32_t b, uint32_t dst,
                     uint32_t n);

/* Serialization */
/*---------------*/

/* Exactly 64 little-endian bytes; bytes may be unaligned. The two objects must not overlap.
 * A NULL input/output is rejected without modifying the output. */
int bns_seq_desc_encode_le(void *bytes, const struct bns_seq_descriptor *descriptor);

#ifdef __cplusplus
}
#endif

#endif /* TRIXIUM_SEQUENCE_H */

/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef BONSAI_SEQUENCE_H
#define BONSAI_SEQUENCE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BNS_SEQ_DESC_WORDS 16u
#define BNS_SEQ_DESC_BYTES 64u
struct bns_seq_descriptor { uint32_t word[BNS_SEQ_DESC_WORDS]; };
enum bns_seq_desc_result { BNS_SEQ_DESC_OK=0, BNS_SEQ_DESC_INVALID=-1 };

/* Pure descriptor encoders: no MMIO, allocation, board names or model policy.
 * Words are host-native uint32_t values. Use bns_seq_desc_encode_le for a byte
 * stream, or ordered 32-bit MMIO stores on the target. All reserved words are
 * zero. Failure leaves the output unchanged; NULL output is invalid.
 *
 * Before submission the caller must check each optional unit's hardware ABI,
 * physical capacity/profile, placement, numerical admission and full-list
 * ownership. These functions check field encoding, not execution validity.
 * Addresses are scratchpad-relative bytes except region-relative uint64_t
 * offsets/strides. Hardware checks full spans/list protection. Live descriptors
 * and input buffers must remain immutable; failure invalidates output prefixes
 * and requires common reset/drain. There is no implicit quantization fallback.
 */
int bns_seq_desc_stop(struct bns_seq_descriptor *out);
/* Unit 3, hadamard ABI 1: 1024-element int8 blocks -> raw int32 + absmax. */
int bns_seq_desc_hadamard(struct bns_seq_descriptor *out, uint32_t src, uint32_t signs,
                         uint32_t dst, uint32_t stats, uint32_t blocks, uint32_t inverse);
/* Unit 4, activation_quantize ABI 1: one exponent per independent FP32 block. */
int bns_seq_desc_quantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                         uint32_t exponents, uint32_t elements, uint32_t blocks);
/* Unit 5 needs activation_quantize ABI 1, minor >= 1. Whole-vector post-Hadamard
 * exponent over ALL 1024-element blocks, before any projection K segmentation. */
int bns_seq_desc_requantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                           uint32_t exponents, uint32_t maxima, uint32_t result_exponent,
                           uint32_t blocks);
/* Unit 6, native ABI 1: full 64-bit row offset/stride, both multiples of 32 (device rows are
 * beat-aligned; the reader rejects unaligned values as a configuration error). */
int bns_seq_desc_native_pipe(struct bns_seq_descriptor *out, uint32_t pipe, uint32_t region,
                            uint64_t offset, uint64_t stride);
struct bns_seq_projection {
    uint32_t kind, rows, rem, groups, bias, rshift, xsrc, dst;
    uint32_t interleaved, suppress_writes, dual, qstride;
};
/* Unit 7, native ABI 1: kind 0 PTQ1_0 or 1 Q8_0; flags exactly 0/1.
 * Keep whole-matrix B/R and one whole-activation exponent across K segments;
 * reduce wrapping int32 segment outputs before restoring FP32 scale. */
int bns_seq_desc_native_projection(struct bns_seq_descriptor *out,
                                  const struct bns_seq_projection *projection);
/* Unit 8, projection_restore ABI 1: int32 -> FP32 using resident e8 + (R-B). */
int bns_seq_desc_restore(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                        uint32_t exponents, uint32_t elements, int32_t adjustment);
/* Unit 9, embedding_restore ABI 1 and admitted limits/profile. Groups 8..512
 * encode the generic kernel range; the shared hardware currently allows <=64. */
int bns_seq_desc_embedding(struct bns_seq_descriptor *out, uint32_t src, uint32_t signs,
                          uint32_t dst, uint32_t groups);
/* Unit 10, region_fetch ABI 1: positive full 32-byte beats, aligned offset/dst.
 * Region READ access, reservations and endpoint/address width are checked in
 * hardware. Read success alone does not publish a valid model allocation. */
int bns_seq_desc_region_fetch(struct bns_seq_descriptor *out, uint32_t region,
                             uint64_t offset, uint32_t length, uint32_t dst);
/* Unit 11, silu ABI 1: dst[i] = m[i]*silu(g[i]) bit-exact with the firmware's
 * lookup; fixup elements keep g in dst and set bit i of the fixmap bitmap
 * (ceil(elements/32) words, all written). dst == src_g or disjoint. */
int bns_seq_desc_silu(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                     uint32_t dst, uint32_t fixmap, uint32_t elements);
/* Unit 11 table load: 1,025 beats at src, lut[0..4096] then dlut from beat 513. */
int bns_seq_desc_silu_load(struct bns_seq_descriptor *out, uint32_t src);
/* Unit 11 (ABI >= 1.1) with options: unit_m (m = 1.0, src_m unused) and permuted output
 * segments (source segment s of seg_beats beats -> destination (s % a)*b + s / a). */
int bns_seq_desc_silu_ex(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                        uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t unit_m,
                        uint32_t seg_beats, uint32_t perm_a, uint32_t perm_b);
/* Unit 11 (ABI >= 1.2), table only: dst = m * table(g) with every element outside the table
 * range a fixup (no SiLU pass-through or -0) - for a table other than SiLU's (the sigmoid). */
int bns_seq_desc_silu_table(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                           uint32_t dst, uint32_t fixmap, uint32_t elements);
/* ABI >= 1.3: a second resident table pair (set 1) - loaded once, selected per run. */
int bns_seq_desc_silu_table_set(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                               uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t set);
int bns_seq_desc_silu_load_set(struct bns_seq_descriptor *out, uint32_t src, uint32_t set);
/* Unit 12, norm ABI 1: RMSNorm (l2 = 0, weight) or L2 (l2 = 1) over nseg segments of n. */
int bns_seq_desc_norm(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src, uint32_t dst,
                     uint32_t weight, uint32_t n, uint32_t nseg, uint32_t eps_bits);
/* Unit 12 (ABI >= 1.1) with strided segments (beats between segment starts, 0 = n/8). */
int bns_seq_desc_norm_strided(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src,
                             uint32_t dst, uint32_t weight, uint32_t n, uint32_t nseg,
                             uint32_t eps_bits, uint32_t src_stride, uint32_t dst_stride);
/* Unit 13, conv ABI 1: the DeltaNet 4-tap causal conv, window x0..x2, input x3, taps w. */
int bns_seq_desc_conv(struct bns_seq_descriptor *out, uint32_t x0, uint32_t x1, uint32_t x2,
                     uint32_t x3, uint32_t w, uint32_t dst, uint32_t channels);
/* Unit 13 op 1: dst = a + b over n FP32 values (the residual add); dst == a or b. */
int bns_seq_desc_add(struct bns_seq_descriptor *out, uint32_t a, uint32_t b, uint32_t dst, uint32_t n);
/* Exactly 64 little-endian bytes; bytes may be unaligned. The two objects must
 * not overlap. NULL input/output is rejected without modifying the output. */
int bns_seq_desc_encode_le(void *bytes, const struct bns_seq_descriptor *descriptor);
#ifdef __cplusplus
}
#endif
#endif

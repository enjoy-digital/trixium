/* SPDX-License-Identifier: BSD-2-Clause */
#include "trixium_sequence.h"

static int sp_address(uint32_t value, uint32_t alignment)
{
    return value < (1u << 24) && value % alignment == 0;
}
static int count16(uint32_t value) { return value > 0 && value < (1u << 16); }

int bns_seq_desc_stop(struct bns_seq_descriptor *out)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out) return BNS_SEQ_DESC_INVALID;
    *out = d;
    return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_hadamard(struct bns_seq_descriptor *out, uint32_t src, uint32_t signs,
                         uint32_t dst, uint32_t stats, uint32_t blocks, uint32_t inverse)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32) || !sp_address(signs, 32) || !sp_address(dst, 32)
        || !sp_address(stats, 32) || !count16(blocks) || inverse > 1) return BNS_SEQ_DESC_INVALID;
    d.word[0]=3; d.word[1]=src; d.word[2]=signs; d.word[3]=dst;
    d.word[4]=blocks; d.word[5]=stats; d.word[6]=inverse;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_quantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                         uint32_t exponents, uint32_t elements, uint32_t blocks)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32) || !sp_address(dst, 32) || !sp_address(exponents, 32)
        || !count16(elements) || elements % 32 || !count16(blocks)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=4; d.word[1]=src; d.word[2]=dst; d.word[3]=exponents;
    d.word[4]=elements; d.word[5]=blocks;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_requantize(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                           uint32_t exponents, uint32_t maxima, uint32_t result_exponent,
                           uint32_t blocks)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32) || !sp_address(dst, 32) || !sp_address(exponents, 32)
        || !sp_address(maxima, 32) || !sp_address(result_exponent, 32) || !count16(blocks))
        return BNS_SEQ_DESC_INVALID;
    d.word[0]=5; d.word[1]=src; d.word[2]=dst; d.word[3]=exponents;
    d.word[4]=maxima; d.word[5]=result_exponent; d.word[6]=blocks;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_restore(struct bns_seq_descriptor *out, uint32_t src, uint32_t dst,
                        uint32_t exponents, uint32_t elements, int32_t adjustment)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32) || !sp_address(dst, 32) || !sp_address(exponents, 32)
        || !count16(elements)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=8; d.word[1]=src; d.word[2]=dst; d.word[3]=exponents;
    d.word[4]=elements; d.word[5]=(uint32_t)adjustment;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_silu(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                     uint32_t dst, uint32_t fixmap, uint32_t elements)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src_g, 32) || !sp_address(src_m, 32) || !sp_address(dst, 32)
        || !sp_address(fixmap, 32) || !count16(elements)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=11; d.word[1]=src_g; d.word[2]=src_m; d.word[3]=dst; d.word[4]=fixmap;
    d.word[5]=elements;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_silu_load(struct bns_seq_descriptor *out, uint32_t src)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=11; d.word[1]=src; d.word[6]=1;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_silu_ex(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                        uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t unit_m,
                        uint32_t seg_beats, uint32_t perm_a, uint32_t perm_b)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src_g, 32) || !sp_address(src_m, 32) || !sp_address(dst, 32)
        || !sp_address(fixmap, 32) || !count16(elements) || unit_m > 1 || seg_beats > 0xFFFF
        || perm_a > 0xFF || perm_b > 0xFF
        || (perm_a > 1 && (uint64_t)seg_beats*perm_a*perm_b*8 != elements)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=11; d.word[1]=src_g; d.word[2]=src_m; d.word[3]=dst; d.word[4]=fixmap;
    d.word[5]=elements; d.word[6]=unit_m << 1; d.word[7]=seg_beats | perm_a << 16 | perm_b << 24;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_silu_table(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                           uint32_t dst, uint32_t fixmap, uint32_t elements)
{
    return bns_seq_desc_silu_table_set(out, src_g, src_m, dst, fixmap, elements, 0);
}
int bns_seq_desc_silu_table_set(struct bns_seq_descriptor *out, uint32_t src_g, uint32_t src_m,
                               uint32_t dst, uint32_t fixmap, uint32_t elements, uint32_t set)
{
    if (set > 1) return BNS_SEQ_DESC_INVALID;
    int rc = bns_seq_desc_silu_ex(out, src_g, src_m, dst, fixmap, elements, 0, 0, 0, 0);
    if (rc == BNS_SEQ_DESC_OK) out->word[6] |= 1u << 2 | set << 3;
    return rc;
}
int bns_seq_desc_silu_load_set(struct bns_seq_descriptor *out, uint32_t src, uint32_t set)
{
    if (set > 1) return BNS_SEQ_DESC_INVALID;
    int rc = bns_seq_desc_silu_load(out, src);
    if (rc == BNS_SEQ_DESC_OK) out->word[6] |= set << 3;
    return rc;
}
int bns_seq_desc_norm(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src, uint32_t dst,
                     uint32_t weight, uint32_t n, uint32_t nseg, uint32_t eps_bits)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || l2 > 1 || !sp_address(src, 32) || !sp_address(dst, 32) || !sp_address(weight, 32)
        || n < 8 || n > 8192 || n % 8 || !count16(nseg)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=12; d.word[1]=src; d.word[2]=dst; d.word[3]=weight; d.word[4]=n | nseg << 16;
    d.word[5]=eps_bits; d.word[6]=l2;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_norm_strided(struct bns_seq_descriptor *out, uint32_t l2, uint32_t src,
                             uint32_t dst, uint32_t weight, uint32_t n, uint32_t nseg,
                             uint32_t eps_bits, uint32_t src_stride, uint32_t dst_stride)
{
    struct bns_seq_descriptor d = {{0}};
    if (bns_seq_desc_norm(&d, l2, src, dst, weight, n, nseg, eps_bits)
        || (src_stride && (src_stride < n/8 || src_stride > 0xFFFF))
        || (dst_stride && (dst_stride < n/8 || dst_stride > 0xFFFF))) return BNS_SEQ_DESC_INVALID;
    if (!out) return BNS_SEQ_DESC_INVALID;
    d.word[7] = src_stride | dst_stride << 16;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_norm_rope(struct bns_seq_descriptor *d, uint32_t rope_n, uint32_t tab)
{
    uint32_t n;
    if (!d || d->word[0] != 12 || rope_n < 16 || rope_n > 256 || rope_n % 16 || !sp_address(tab, 32)
        || tab >= 1u << 18)                     /* a 13-bit beat index: the 256 KB scratchpad */
        return BNS_SEQ_DESC_INVALID;
    n = d->word[4] & 0xFFFF;
    if (rope_n > n || (d->word[6] & ~1u)) return BNS_SEQ_DESC_INVALID;
    d->word[6] |= (rope_n/16) << 8 | (tab/32) << 16;
    return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_conv(struct bns_seq_descriptor *out, uint32_t x0, uint32_t x1, uint32_t x2,
                     uint32_t x3, uint32_t w, uint32_t dst, uint32_t channels)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(x0, 32) || !sp_address(x1, 32) || !sp_address(x2, 32)
        || !sp_address(x3, 32) || !sp_address(w, 32) || !sp_address(dst, 32)
        || channels < 8 || channels > 0xFFFF || channels % 8) return BNS_SEQ_DESC_INVALID;
    d.word[0]=13; d.word[1]=x0; d.word[2]=x1; d.word[3]=x2; d.word[4]=x3; d.word[5]=w;
    d.word[6]=dst; d.word[7]=channels;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_add(struct bns_seq_descriptor *out, uint32_t a, uint32_t b, uint32_t dst, uint32_t n)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(a, 32) || !sp_address(b, 32) || !sp_address(dst, 32)
        || n < 8 || n > 0xFFFF || n % 8) return BNS_SEQ_DESC_INVALID;
    d.word[0]=13; d.word[1]=a; d.word[4]=b; d.word[6]=dst; d.word[7]=n | 1u << 16;
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_encode_le(void *bytes, const struct bns_seq_descriptor *descriptor)
{
    uint8_t *p=(uint8_t *)bytes;
    if (!bytes || !descriptor) return BNS_SEQ_DESC_INVALID;
    for (unsigned i=0; i<BNS_SEQ_DESC_WORDS; i++)
        for (unsigned j=0; j<4; j++)
            p[4*i+j]=(uint8_t)(descriptor->word[i] >> (8*j));
    return BNS_SEQ_DESC_OK;
}

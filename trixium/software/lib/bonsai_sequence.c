/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_sequence.h"

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
int bns_seq_desc_native_pipe(struct bns_seq_descriptor *out, uint32_t pipe, uint32_t region,
                            uint64_t offset, uint64_t stride)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || pipe >= 32) return BNS_SEQ_DESC_INVALID;
    d.word[0]=6; d.word[1]=pipe; d.word[2]=region;
    d.word[3]=(uint32_t)offset; d.word[4]=(uint32_t)(offset >> 32);
    d.word[5]=(uint32_t)stride; d.word[6]=(uint32_t)(stride >> 32);
    *out=d; return BNS_SEQ_DESC_OK;
}
int bns_seq_desc_native_projection(struct bns_seq_descriptor *out,
                                  const struct bns_seq_projection *p)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !p || p->kind > 1 || p->rows >= (1u << 16) || p->rem >= 32
        || (!p->rows && !p->rem) || !p->groups || p->groups >= 256
        || p->bias >= 128 || p->rshift >= 64 || !sp_address(p->xsrc, 32)
        || !sp_address(p->dst, 4) || p->interleaved > 1 || p->suppress_writes > 1
        || p->dual > 1 || p->qstride >= (1u << 16)) return BNS_SEQ_DESC_INVALID;
    d.word[0]=7;
    d.word[1]=p->kind | (p->interleaved << 8) | (p->suppress_writes << 9) | (p->dual << 11);
    d.word[2]=p->rows | (p->rem << 16);
    d.word[6]=(p->groups << 8) | (p->qstride << 16);
    d.word[7]=p->bias | (p->rshift << 16); d.word[8]=p->xsrc; d.word[9]=p->dst;
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
int bns_seq_desc_embedding(struct bns_seq_descriptor *out, uint32_t src, uint32_t signs,
                          uint32_t dst, uint32_t groups)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || !sp_address(src, 32) || !sp_address(signs, 32) || !sp_address(dst, 32)
        || groups < 8 || groups > 512 || groups % 8) return BNS_SEQ_DESC_INVALID;
    d.word[0]=9; d.word[1]=src; d.word[2]=signs; d.word[3]=dst; d.word[4]=groups;
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
int bns_seq_desc_region_fetch(struct bns_seq_descriptor *out, uint32_t region,
                             uint64_t offset, uint32_t length, uint32_t dst)
{
    struct bns_seq_descriptor d = {{0}};
    if (!out || offset % 32 || !length || length >= (1u << 24) || !sp_address(dst, 32))
        return BNS_SEQ_DESC_INVALID;
    d.word[0]=10; d.word[1]=region; d.word[2]=(uint32_t)offset;
    d.word[3]=(uint32_t)(offset >> 32); d.word[4]=length; d.word[5]=dst;
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

/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_region_read.h"
#include <stddef.h>

static uint32_t reg(const struct bns_read_io *io, uint32_t offset)
{
    return io->read32(io->context, BNS_RB_VERIFY_BANK, offset);
}
static void put(const struct bns_read_io *io, uint32_t offset, uint32_t value)
{
    io->write32(io->context, BNS_RB_VERIFY_BANK, offset, value);
}
static int status(const struct bns_read_io *io, struct bns_read_progress *progress, uint32_t *st)
{
    *st=reg(io, BNS_RB_STATUS);
    uint32_t error=reg(io, BNS_RB_ERROR);
    if (progress) progress->hardware_error=error;
    if ((*st&4) || error) return BNS_RB_HARDWARE;
    return (*st&~7u) ? BNS_RB_PROTOCOL : BNS_RB_OK;
}
static int deadline(const struct bns_read_io *io, uint64_t started, uint64_t timeout)
{
    return io->now_us(io->context)-started >= timeout ? BNS_RB_TIMEOUT : BNS_RB_OK;
}
static int transfer(const struct bns_read_io *io, uint32_t region, uint64_t offset,
                    void *payload, const void *reference, uint64_t bytes, uint32_t scratch_dst,
                    uint32_t scratch_bytes, uint64_t timeout, struct bns_read_progress *progress)
{
    if (progress) {
        progress->completed_bytes=0; progress->hardware_error=0; progress->mismatch_offset=UINT64_MAX;
    }
    if (!io || !io->read32 || !io->write32 || !io->now_us || (!payload && !reference)
        || !bytes || !timeout || (offset&31) || (bytes&31) || (scratch_dst&31)
        || !scratch_bytes || (scratch_bytes&31) || (uint64_t)scratch_dst+scratch_bytes > (1u<<24)
        || bytes>SIZE_MAX || bytes>UINT64_MAX-offset) return BNS_RB_INVALID;
    if (reg(io, BNS_RB_ABI)!=0x00010001) return BNS_RB_UNSUPPORTED;
    uint8_t *out=(uint8_t *)payload;
    const uint8_t *expected=(const uint8_t *)reference;
    for (uint64_t done=0; done<bytes;) {
        uint32_t st;
        int rc=status(io, progress, &st);
        if (rc) return rc;
        if ((st&3) || reg(io, BNS_RB_SP_PENDING)) return BNS_RB_BUSY;
        uint32_t chunk=(bytes-done < scratch_bytes) ? (uint32_t)(bytes-done) : scratch_bytes;
        uint32_t before=reg(io, BNS_RB_COMPLETED), accepted=reg(io, BNS_RB_ACCEPTED);
        uint64_t started=io->now_us(io->context), address=offset+done;
        put(io, BNS_RB_REGION, region);
        put(io, BNS_RB_OFFSET_HI, (uint32_t)(address>>32));
        put(io, BNS_RB_OFFSET_LO, (uint32_t)address);
        put(io, BNS_RB_LENGTH, chunk/32);
        put(io, BNS_RB_DST, scratch_dst);
        put(io, BNS_RB_START, 1);
        for (;;) {
            rc=status(io, progress, &st);
            if (rc) return rc;
            if ((rc=deadline(io, started, timeout))) return rc;
            uint32_t completed=reg(io, BNS_RB_COMPLETED);
            if (completed==(uint32_t)(before+1)) {
                rc=status(io, progress, &st);
                if (rc) return rc;
                if ((st&3) || reg(io, BNS_RB_BEATS)!=chunk/32
                    || reg(io, BNS_RB_ACCEPTED)!=(uint32_t)(accepted+1)) return BNS_RB_PROTOCOL;
                break;
            }
            if (completed!=before) return BNS_RB_PROTOCOL;
            if (io->wait) io->wait(io->context);
        }
        for (uint32_t pos=0; pos<chunk; pos+=4) {
            if ((rc=deadline(io, started, timeout))) return rc;
            if (reg(io, BNS_RB_SP_PENDING)) return BNS_RB_PROTOCOL;
            uint32_t sequence=reg(io, BNS_RB_SP_COMPLETED);
            io->write32(io->context, BNS_RB_SCRATCHPAD_BANK, BNS_RB_SP_ADR, scratch_dst+pos);
            for (;;) {
                rc=status(io, progress, &st);
                if (rc) return rc;
                if (st&3) return BNS_RB_PROTOCOL;
                if ((rc=deadline(io, started, timeout))) return rc;
                uint32_t completed=reg(io, BNS_RB_SP_COMPLETED);
                if (completed==(uint32_t)(sequence+1)) {
                    if (reg(io, BNS_RB_SP_PENDING)) return BNS_RB_PROTOCOL;
                    break;
                }
                if (completed!=sequence) return BNS_RB_PROTOCOL;
                if (io->wait) io->wait(io->context);
            }
            uint32_t value=io->read32(io->context, BNS_RB_SCRATCHPAD_BANK, BNS_RB_SP_DATA);
            for (unsigned lane=0; lane<4; lane++) {
                uint8_t byte=(uint8_t)(value>>(8*lane));
                uint64_t index=done+pos+lane;
                if (expected) {
                    if (expected[index]!=byte) {
                        if (progress) progress->mismatch_offset=index;
                        return BNS_RB_MISMATCH;
                    }
                } else out[index]=byte;
            }
        }
        rc=status(io, progress, &st);
        if (rc) return rc;
        if (st&3) return BNS_RB_PROTOCOL;
        if ((rc=deadline(io, started, timeout))) return rc;
        done+=chunk;
        if (progress) progress->completed_bytes=done;
    }
    return BNS_RB_OK;
}
int bns_region_readback(const struct bns_read_io *io, uint32_t region, uint64_t offset,
                        void *payload, uint64_t bytes, uint32_t scratch_dst,
                        uint32_t scratch_bytes, uint64_t timeout_us, struct bns_read_progress *progress)
{
    return transfer(io, region, offset, payload, NULL, bytes, scratch_dst, scratch_bytes, timeout_us, progress);
}
int bns_region_verify(const struct bns_read_io *io, uint32_t region, uint64_t offset,
                      const void *reference, uint64_t bytes, uint32_t scratch_dst,
                      uint32_t scratch_bytes, uint64_t timeout_us, struct bns_read_progress *progress)
{
    return transfer(io, region, offset, NULL, reference, bytes, scratch_dst, scratch_bytes, timeout_us, progress);
}

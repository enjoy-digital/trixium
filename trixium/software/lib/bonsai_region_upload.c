/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_region_upload.h"
#include <stddef.h>

#define UPLOAD_CHUNK_BYTES (UINT64_C(128) << 20)
#define UPLOAD_ABI UINT32_C(0x00010000)

static int status(const struct bns_upload_io *io, struct bns_upload_progress *progress, uint32_t *st)
{
    *st = io->read32(io->context, BNS_UP_STATUS);
    uint32_t error = io->read32(io->context, BNS_UP_ERROR);
    if (progress) progress->hardware_error = error;
    return ((*st & 4) || error) ? BNS_UP_HARDWARE : BNS_UP_OK;
}

static int wait_step(const struct bns_upload_io *io, uint64_t started, uint64_t timeout)
{
    if (io->now_us(io->context) - started >= timeout) return BNS_UP_TIMEOUT;
    if (io->wait) io->wait(io->context);
    return BNS_UP_OK;
}

int bns_region_upload(const struct bns_upload_io *io, uint32_t region, uint64_t offset,
                      const void *payload, uint64_t bytes, uint64_t timeout_us,
                      struct bns_upload_progress *progress)
{
    if (progress) { progress->completed_bytes = 0; progress->hardware_error = 0; }
    if (!io || !io->read32 || !io->write32 || !io->now_us || !payload || !bytes || !timeout_us
        || (offset & 31) || (bytes & 31) || bytes > SIZE_MAX || bytes > UINT64_MAX-offset)
        return BNS_UP_INVALID;
    if (io->read32(io->context, BNS_UP_ABI) != UPLOAD_ABI) return BNS_UP_UNSUPPORTED;
    const uint8_t *data = (const uint8_t *)payload;
    uint64_t done = 0;
    while (done < bytes) {
        uint32_t st;
        int rc = status(io, progress, &st);
        if (rc) return rc;
        if (st & 3) return BNS_UP_BUSY;
        uint64_t chunk = bytes-done;
        if (chunk > UPLOAD_CHUNK_BYTES) chunk = UPLOAD_CHUNK_BYTES;
        uint32_t beats = (uint32_t)(chunk/32), words = (uint32_t)(chunk/4), sent = 0;
        uint32_t before = io->read32(io->context, BNS_UP_COMPLETED);
        uint32_t accepted = io->read32(io->context, BNS_UP_ACCEPTED);
        uint64_t started = io->now_us(io->context), address = offset+done;
        io->write32(io->context, BNS_UP_REGION, region);
        io->write32(io->context, BNS_UP_OFFSET_HI, (uint32_t)(address >> 32));
        io->write32(io->context, BNS_UP_OFFSET_LO, (uint32_t)address);
        io->write32(io->context, BNS_UP_LENGTH, beats);
        io->write32(io->context, BNS_UP_START, 1);
        while (sent < words) {
            rc = status(io, progress, &st);
            if (rc) return rc;
            if (io->now_us(io->context)-started >= timeout_us) return BNS_UP_TIMEOUT;
            uint32_t credits = io->read32(io->context, BNS_UP_SPACE);
            /* ABI permits 2..256 beats of FIFO capacity. Reject impossible
             * credits; do not trust a stale/mismatched register map. */
            if (credits > 2048 || credits > words-sent) return BNS_UP_PROTOCOL;
            if (!credits) {
                rc = wait_step(io, started, timeout_us);
                if (rc) return rc;
                continue;
            }
            for (uint32_t i=0; i<credits; i++, sent++) {
                const uint8_t *p = data+done+(uint64_t)sent*4;
                uint32_t value = (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
                io->write32(io->context, BNS_UP_POKE, value);
            }
        }
        for (;;) {
            rc = status(io, progress, &st);
            if (rc) return rc;
            if (io->now_us(io->context)-started >= timeout_us) return BNS_UP_TIMEOUT;
            uint32_t completed = io->read32(io->context, BNS_UP_COMPLETED);
            if (completed == (uint32_t)(before+1)) {
                rc = status(io, progress, &st);
                if (rc) return rc;
                if ((st & 3) || io->read32(io->context, BNS_UP_WORDS) != words
                    || io->read32(io->context, BNS_UP_BEATS) != beats
                    || io->read32(io->context, BNS_UP_ACCEPTED) != (uint32_t)(accepted+1))
                    return BNS_UP_PROTOCOL;
                break;
            }
            if (completed != before) return BNS_UP_PROTOCOL;
            rc = wait_step(io, started, timeout_us);
            if (rc) return rc;
        }
        done += chunk;
        if (progress) progress->completed_bytes = done;
    }
    return BNS_UP_OK;
}

/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef BONSAI_REGION_UPLOAD_H
#define BONSAI_REGION_UPLOAD_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Experimental zzregion bank ABI, independent of board names and MMIO mapping.
 * The caller owns the device exclusively, supplies a monotonic microsecond
 * clock, and validates the full allocation/placement before loading it. Callbacks
 * preserve MMIO order; reads must complete preceding posted writes. This
 * write-only extension does NOT imply the public REGION_IO capability.
 * Multi-chunk transfers are not atomic. Any failure invalidates the target
 * allocation, may leave a written prefix, and requires outer reset/drain before
 * reuse. The function never resets someone else's running engine implicitly.
 */
enum bns_upload_register {
    BNS_UP_ABI=0x00, BNS_UP_REGION=0x04, BNS_UP_OFFSET_HI=0x08, BNS_UP_OFFSET_LO=0x0c,
    BNS_UP_LENGTH=0x10, BNS_UP_START=0x14, BNS_UP_POKE=0x18, BNS_UP_STATUS=0x1c,
    BNS_UP_ERROR=0x20, BNS_UP_SPACE=0x24, BNS_UP_WORDS=0x28, BNS_UP_BEATS=0x2c,
    BNS_UP_ACCEPTED=0x30, BNS_UP_COMPLETED=0x34
};
enum bns_upload_result {
    BNS_UP_OK=0, BNS_UP_INVALID=-1, BNS_UP_UNSUPPORTED=-2, BNS_UP_BUSY=-3,
    BNS_UP_TIMEOUT=-4, BNS_UP_HARDWARE=-5, BNS_UP_PROTOCOL=-6
};
struct bns_upload_io {
    void *context;
    uint32_t (*read32)(void *context, uint32_t byte_offset);
    void (*write32)(void *context, uint32_t byte_offset, uint32_t value);
    uint64_t (*now_us)(void *context);
    void (*wait)(void *context); /* Optional yielding/pacing while waiting, never a deadline substitute. */
};
struct bns_upload_progress {
    uint64_t completed_bytes; /* Only fully acknowledged chunks, not a validity guarantee after error. */
    uint32_t hardware_error;
};

/* Full 32-byte beats only; caller owns/pads final allocation bytes explicitly.
 * Splits at 128 MiB, below the 24-bit beat counter. timeout_us is per chunk.
 * Payload can be unaligned; its little-endian words are assembled bytewise.
 */
int bns_region_upload(const struct bns_upload_io *io, uint32_t region, uint64_t offset,
                      const void *payload, uint64_t bytes, uint64_t timeout_us,
                      struct bns_upload_progress *progress);
#ifdef __cplusplus
}
#endif
#endif

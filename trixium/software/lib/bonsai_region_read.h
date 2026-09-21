/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef BONSAI_REGION_READ_H
#define BONSAI_REGION_READ_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Bank selectors are logical, not CSR addresses or board names. Callbacks map
 * them to generated bank bases. Ordered reads complete preceding posted writes.
 * Caller owns the device AND scratch workspace exclusively; stop inference,
 * finish uploads and prevalidate the full placement before verification.
 * ABI 1.1 adds completion-aware scratchpad peeks to the ABI 1.0 read transport.
 */
enum bns_read_bank { BNS_RB_VERIFY_BANK=0, BNS_RB_SCRATCHPAD_BANK=1 };
enum bns_read_register {
    BNS_RB_ABI=0x00, BNS_RB_REGION=0x04, BNS_RB_OFFSET_HI=0x08, BNS_RB_OFFSET_LO=0x0c,
    BNS_RB_LENGTH=0x10, BNS_RB_DST=0x14, BNS_RB_START=0x18, BNS_RB_STATUS=0x1c,
    BNS_RB_ERROR=0x20, BNS_RB_BEATS=0x24, BNS_RB_ACCEPTED=0x28, BNS_RB_COMPLETED=0x2c,
    BNS_RB_SP_PENDING=0x30, BNS_RB_SP_COMPLETED=0x34
};
enum bns_read_sp_register { BNS_RB_SP_ADR=0x00, BNS_RB_SP_DATA=0x04 };
enum bns_read_result {
    BNS_RB_OK=0, BNS_RB_INVALID=-1, BNS_RB_UNSUPPORTED=-2, BNS_RB_BUSY=-3,
    BNS_RB_TIMEOUT=-4, BNS_RB_HARDWARE=-5, BNS_RB_PROTOCOL=-6, BNS_RB_MISMATCH=-7
};
struct bns_read_io {
    void *context;
    uint32_t (*read32)(void *context, uint32_t bank, uint32_t byte_offset);
    void (*write32)(void *context, uint32_t bank, uint32_t byte_offset, uint32_t value);
    uint64_t (*now_us)(void *context);
    void (*wait)(void *context); /* Optional pacing; must return, not replace the deadline. */
};
struct bns_read_progress {
    uint64_t completed_bytes; /* Fully copied/compared chunks only; failure invalidates the whole operation. */
    uint64_t mismatch_offset; /* Byte offset relative to the input span, UINT64_MAX unless a mismatch. */
    uint32_t hardware_error;
};

/* Read or byte-compare a region span using a caller-owned scratchpad workspace.
 * Source offset, bytes, scratch_dst and scratch_bytes are 32-byte aligned.
 * Scratch size is NOT discovered/inferred from the compute-pipe count. Supply
 * a reserved, capability-validated workspace. Transfer chunks fit that space.
 * Payload/reference pointers may be unaligned. No allocation or implicit reset.
 * timeout_us bounds each chunk, including scratchpad extraction/comparison;
 * the monotonic clock may wrap. Callbacks themselves must not block forever.
 * On read failure discard any copied prefix. On any verify failure the model
 * is not verified, even if completed_bytes is nonzero. Transport/protocol errors
 * or timeout require full reset/drain before reuse. A mismatch returns only
 * after its scratchpad read has completed; it does not alter source weights.
 */
int bns_region_readback(const struct bns_read_io *io, uint32_t region, uint64_t offset,
                        void *payload, uint64_t bytes, uint32_t scratch_dst,
                        uint32_t scratch_bytes, uint64_t timeout_us,
                        struct bns_read_progress *progress);
int bns_region_verify(const struct bns_read_io *io, uint32_t region, uint64_t offset,
                      const void *reference, uint64_t bytes, uint32_t scratch_dst,
                      uint32_t scratch_bytes, uint64_t timeout_us,
                      struct bns_read_progress *progress);
#ifdef __cplusplus
}
#endif
#endif

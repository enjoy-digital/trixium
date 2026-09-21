/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef BONSAI_CAPS_H
#define BONSAI_CAPS_H
#include <stddef.h>
#include <stdint.h>

#define BNS_CAP_MAGIC UINT32_C(0x50434e42)
#define BNS_CAP_ABI 1u
#define BNS_CAP_MAX_BYTES 65536u
#define BNS_CAP_HEADER_BYTES 64u
#define BNS_CAP_REGION_BYTES 40u
#define BNS_CAP_RESERVATION_BYTES 24u

enum bns_feature {
    BNS_SOFT_RESET = 1u << 0, BNS_VECTOR = 1u << 1, BNS_SEQUENCER = 1u << 2,
    BNS_DUAL_Q = 1u << 3, BNS_DELTANET_FP32 = 1u << 4,
    BNS_COHERENT_OVERLAP = 1u << 5, BNS_AXI_TRAFFIC = 1u << 6,
    BNS_DDR_TRAFFIC = 1u << 7, BNS_REGION_IO = 1u << 8
};
enum bns_weight_encoding {
    BNS_TERNARY_ROWS, BNS_Q8_ROWS, BNS_F16, BNS_F32, BNS_BFP16,
    BNS_PTQ1_0, BNS_Q8_0, BNS_Q4_0
};
enum bns_memory_kind { BNS_DDR4 = 1, BNS_HBM, BNS_SIMULATION };
enum bns_access { BNS_READ = 1, BNS_WRITE = 2 };
enum bns_reservation_kind { BNS_CPU = 1, BNS_UPLOAD_STAGING, BNS_DIAGNOSTIC, BNS_SYSTEM };
enum bns_caps_result {
    BNS_CAP_OK = 0, BNS_CAP_INVALID = -1, BNS_CAP_UNSUPPORTED = -2,
    BNS_CAP_BOUNDS = -3, BNS_CAP_RESERVED = -4
};

struct bns_memory_region {
    uint32_t id, kind, endpoint, access;
    uint64_t base, size;
    uint32_t alignment, locality;
};
struct bns_reservation {
    uint32_t region_id, kind;
    uint64_t offset, size;
};
struct bns_caps {
    /* Borrowed immutable buffer, valid until this view is no longer used. */
    const uint8_t *data;
    uint32_t sys_hz, compute_hz, compute_pipes, scratchpad_bytes;
    uint64_t features, weight_encodings, physical_bytes, unreserved_bytes;
    uint32_t region_count, reservation_count;
};

/* No dynamic allocation or MMIO dependency. Basic compiler memory helpers such
 * as memset may be needed on bare-metal targets. Input may be unaligned. On
 * failure the view is invalidated. Accessors require a successfully opened view. */
int bns_caps_open(struct bns_caps *out, const void *data, size_t length);
int bns_caps_region(const struct bns_caps *caps, uint32_t index, struct bns_memory_region *out);
int bns_caps_reservation(const struct bns_caps *caps, uint32_t index, struct bns_reservation *out);
/* Pure validation/address translation, never submits a transfer or truncates an
 * offset to legacy wbase. Legacy devices return BNS_CAP_UNSUPPORTED. */
int bns_caps_resolve(const struct bns_caps *caps, uint32_t region_id, uint64_t offset,
                     uint64_t size, uint32_t access, uint32_t *endpoint, uint64_t *address);
#endif

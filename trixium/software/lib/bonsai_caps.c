/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_caps.h"

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32; }

static uint32_t checksum(const uint8_t *data, size_t n)
{
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < n; i++) {
        crc ^= i >= 12 && i < 16 ? 0 : data[i];
        for (int j = 0; j < 8; j++) crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (0u - (crc & 1)));
    }
    return ~crc;
}

int bns_caps_region(const struct bns_caps *c, uint32_t index, struct bns_memory_region *r)
{
    if (!c || !c->data || !r || index >= c->region_count) return BNS_CAP_BOUNDS;
    const uint8_t *p = c->data + BNS_CAP_HEADER_BYTES + index*BNS_CAP_REGION_BYTES;
    r->id = le32(p); r->kind = le32(p + 4); r->endpoint = le32(p + 8); r->access = le32(p + 12);
    r->base = le64(p + 16); r->size = le64(p + 24);
    r->alignment = le32(p + 32); r->locality = le32(p + 36);
    return BNS_CAP_OK;
}

int bns_caps_reservation(const struct bns_caps *c, uint32_t index, struct bns_reservation *r)
{
    if (!c || !c->data || !r || index >= c->reservation_count) return BNS_CAP_BOUNDS;
    const uint8_t *p = c->data + BNS_CAP_HEADER_BYTES + c->region_count*BNS_CAP_REGION_BYTES
                       + index*BNS_CAP_RESERVATION_BYTES;
    r->region_id = le32(p); r->kind = le32(p + 4); r->offset = le64(p + 8); r->size = le64(p + 16);
    return BNS_CAP_OK;
}

/* These ranges have positive sizes. Subtraction avoids end-address overflow. */
static int overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m)
{
    return a <= b ? b - a < n : a - b < m;
}

int bns_caps_open(struct bns_caps *out, const void *buffer, size_t length)
{
    if (!out) return BNS_CAP_INVALID;
    *out = (struct bns_caps){0};
    const uint8_t *p = buffer;
    if (!p || length < BNS_CAP_HEADER_BYTES || length > BNS_CAP_MAX_BYTES) return BNS_CAP_INVALID;
    if (le32(p) != BNS_CAP_MAGIC || le32(p + 4) != BNS_CAP_ABI) return BNS_CAP_UNSUPPORTED;
    if (le32(p + 8) != length || le32(p + 52) > BNS_CAP_MAX_BYTES/BNS_CAP_RESERVATION_BYTES ||
        le32(p + 48) > BNS_CAP_MAX_BYTES/BNS_CAP_REGION_BYTES ||
        le32(p + 56) != BNS_CAP_REGION_BYTES || le32(p + 60) != BNS_CAP_RESERVATION_BYTES)
        return BNS_CAP_INVALID;
    uint32_t nr = le32(p + 48), ns = le32(p + 52);
    if (!nr || BNS_CAP_HEADER_BYTES + nr*BNS_CAP_REGION_BYTES + ns*BNS_CAP_RESERVATION_BYTES != length ||
        checksum(p, length) != le32(p + 12)) return BNS_CAP_INVALID;
    struct bns_caps c = {0};
    c.data = p;
    c.sys_hz = le32(p + 16); c.compute_hz = le32(p + 20);
    c.compute_pipes = le32(p + 24); c.scratchpad_bytes = le32(p + 28);
    c.features = le64(p + 32); c.weight_encodings = le64(p + 40);
    c.region_count = nr; c.reservation_count = ns;
    if (!c.sys_hz || !c.compute_hz || !c.compute_pipes || !c.scratchpad_bytes ||
        (c.features & ~UINT64_C(0x1ff)) || (c.weight_encodings & ~UINT64_C(0xff))) return BNS_CAP_INVALID;
    for (uint32_t i = 0; i < nr; i++) {
        struct bns_memory_region r;
        bns_caps_region(&c, i, &r);
        if (r.kind < BNS_DDR4 || r.kind > BNS_SIMULATION || (r.access & ~3u) ||
            (r.access && !(c.features & BNS_REGION_IO)) || !r.size ||
            r.base > UINT64_MAX - (r.size - 1) || !r.alignment || (r.alignment & (r.alignment - 1)) ||
            (r.base & (r.alignment - 1)) || (r.size & (r.alignment - 1)) ||
            r.size > UINT64_MAX - c.physical_bytes) return BNS_CAP_INVALID;
        c.physical_bytes += r.size;
        for (uint32_t j = 0; j < i; j++) {
            struct bns_memory_region a;
            bns_caps_region(&c, j, &a);
            if (a.id == r.id || (a.endpoint == r.endpoint && overlaps(a.base, a.size, r.base, r.size)))
                return BNS_CAP_INVALID;
        }
    }
    c.unreserved_bytes = c.physical_bytes;
    for (uint32_t i = 0; i < ns; i++) {
        struct bns_reservation r;
        bns_caps_reservation(&c, i, &r);
        if (r.kind < BNS_CPU || r.kind > BNS_SYSTEM || !r.size) return BNS_CAP_INVALID;
        int found = 0;
        for (uint32_t j = 0; j < nr; j++) {
            struct bns_memory_region a;
            bns_caps_region(&c, j, &a);
            if (a.id == r.region_id) {
                if (r.offset > a.size || r.size > a.size - r.offset) return BNS_CAP_INVALID;
                found = 1;
                break;
            }
        }
        if (!found) return BNS_CAP_INVALID;
        for (uint32_t j = 0; j < i; j++) {
            struct bns_reservation a;
            bns_caps_reservation(&c, j, &a);
            if (a.region_id == r.region_id && overlaps(a.offset, a.size, r.offset, r.size))
                return BNS_CAP_INVALID;
        }
        c.unreserved_bytes -= r.size;
    }
    *out = c;
    return BNS_CAP_OK;
}

int bns_caps_resolve(const struct bns_caps *c, uint32_t id, uint64_t offset,
                     uint64_t size, uint32_t access, uint32_t *endpoint, uint64_t *address)
{
    if (!c || !c->data || !endpoint || !address) return BNS_CAP_INVALID;
    if (!(c->features & BNS_REGION_IO)) return BNS_CAP_UNSUPPORTED;
    if (!access || (access & ~3u)) return BNS_CAP_INVALID;
    for (uint32_t i = 0; i < c->region_count; i++) {
        struct bns_memory_region r;
        bns_caps_region(c, i, &r);
        if (r.id != id) continue;
        if ((r.access & access) != access) return BNS_CAP_UNSUPPORTED;
        if (!size || offset > r.size || size > r.size - offset ||
            (offset & (r.alignment - 1)) || (size & (r.alignment - 1))) return BNS_CAP_BOUNDS;
        for (uint32_t j = 0; j < c->reservation_count; j++) {
            struct bns_reservation a;
            bns_caps_reservation(c, j, &a);
            if (a.region_id == id && overlaps(offset, size, a.offset, a.size)) return BNS_CAP_RESERVED;
        }
        *endpoint = r.endpoint;
        *address = r.base + offset;
        return BNS_CAP_OK;
    }
    return BNS_CAP_BOUNDS;
}

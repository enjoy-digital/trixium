/* SPDX-License-Identifier: BSD-2-Clause
 * Inspect the SAME immutable capability ABI from a file or a live PCIe device.
 * Live access needs generated headers matching the programmed bitstream.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include "liblitepcie.h"
#include "bonsai_caps.h"

static int read_device(const char *path, uint8_t *buffer, size_t *length)
{
#ifdef CSR_ZZCAPS_MAGIC_ADDR
    int fd = open(path, O_RDWR);
    if (fd < 0) { perror(path); return -1; }
    if (flock(fd, LOCK_EX | LOCK_NB)) { perror("capability reader lock"); close(fd); return -1; }
    int rc = -1;
    if (litepcie_readl(fd, CSR_ZZCAPS_MAGIC_ADDR) != BNS_CAP_MAGIC ||
        litepcie_readl(fd, CSR_ZZCAPS_ABI_ADDR) != BNS_CAP_ABI) goto done;
    uint32_t n = litepcie_readl(fd, CSR_ZZCAPS_LENGTH_ADDR);
    if (n < BNS_CAP_HEADER_BYTES || n > BNS_CAP_MAX_BYTES || n % 4) goto done;
    for (uint32_t i = 0; i < n/4; i++) {
        litepcie_writel(fd, CSR_ZZCAPS_INDEX_ADDR, i);
        if (!litepcie_readl(fd, CSR_ZZCAPS_VALID_ADDR)) goto done;
        uint32_t word = litepcie_readl(fd, CSR_ZZCAPS_WORD_ADDR);
        for (int j = 0; j < 4; j++) buffer[4*i + j] = word >> (8*j);
    }
    *length = n;
    rc = 0;
done:
    close(fd);
    return rc;
#else
    (void)path; (void)buffer; (void)length;
    fputs("This build has no capability CSR map; use explicit legacy configuration.\n", stderr);
    return -1;
#endif
}

int main(int argc, char **argv)
{
    if (argc != 3 || (strcmp(argv[1], "-f") && strcmp(argv[1], "-d"))) {
        fprintf(stderr, "usage: %s -f capabilities.bin | -d /dev/litepcie0\n", argv[0]);
        return 2;
    }
    uint8_t *buffer = malloc(BNS_CAP_MAX_BYTES);
    if (!buffer) return 1;
    size_t length = 0;
    int rc = -1;
    if (!strcmp(argv[1], "-f")) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) perror(argv[2]);
        else {
            length = fread(buffer, 1, BNS_CAP_MAX_BYTES, f);
            if (fgetc(f) == EOF && !ferror(f)) rc = 0;
            fclose(f);
        }
    } else rc = read_device(argv[2], buffer, &length);
    struct bns_caps caps;
    if (rc || bns_caps_open(&caps, buffer, length)) {
        fputs("Capabilities absent, malformed or unsupported; no capacity/feature inference.\n", stderr);
        free(buffer);
        return 1;
    }
    printf("BNCP ABI 1.0: pipes=%" PRIu32 " sys_hz=%" PRIu32 " compute_hz=%" PRIu32
           " scratchpad_bytes=%" PRIu32 "\n", caps.compute_pipes, caps.sys_hz,
           caps.compute_hz, caps.scratchpad_bytes);
    printf("features=0x%" PRIx64 " weight_encodings=0x%" PRIx64
           " physical_bytes=%" PRIu64 " unreserved_bytes=%" PRIu64 "\n",
           caps.features, caps.weight_encodings, caps.physical_bytes, caps.unreserved_bytes);
    for (uint32_t i = 0; i < caps.region_count; i++) {
        struct bns_memory_region r;
        bns_caps_region(&caps, i, &r);
        printf("region id=%" PRIu32 " kind=%" PRIu32 " endpoint=%" PRIu32
               " base=%" PRIu64 " size=%" PRIu64 " alignment=%" PRIu32
               " locality=%" PRIu32 " access=%" PRIu32 "\n",
               r.id, r.kind, r.endpoint, r.base, r.size, r.alignment, r.locality, r.access);
    }
    for (uint32_t i = 0; i < caps.reservation_count; i++) {
        struct bns_reservation r;
        bns_caps_reservation(&caps, i, &r);
        printf("reservation region=%" PRIu32 " kind=%" PRIu32 " offset=%" PRIu64 " size=%" PRIu64 "\n",
               r.region_id, r.kind, r.offset, r.size);
    }
    if (!(caps.features & BNS_REGION_IO)) puts("Region-addressed I/O is NOT implemented by this device.");
    free(buffer);
    return 0;
}

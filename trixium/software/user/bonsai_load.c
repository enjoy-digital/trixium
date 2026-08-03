/* This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * bonsai_load: upload a BONS weight image to HBM.
 *
 * For each channel c of the image, programs the loader (chan=c, base=BONSAI_CHAN_WEIGHT_BASE)
 * and streams the channel payload through the PCIe DMA (write() on /dev/bonsaiN feeds
 * pcie_dma0.source -> loader). The header+table area is staged into engine channel 0 at
 * HDR_STAGING; the firmware's `hdrload` command copies it to main_ram.
 *
 * Usage: bonsai_load [-c devnum] image.img
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "liblitepcie.h"

/* Keep in sync with trixium/layout.py and generated soc.h. */
#include "soc.h"
#ifndef BONSAI_CHAN_WEIGHT_BASE
#error "SoC headers lack BONSAI_CHAN_WEIGHT_BASE (build with variant b)"
#endif
#define HDR_STAGING (BONSAI_CHAN_WEIGHT_BASE - 0x100000)

struct bons_header {
    uint32_t magic, version, n_layer, n_embd, n_head, n_kv_head, head_dim, ffn_dim, vocab;
    uint32_t rope_theta;
    float    rms_eps;
    uint32_t max_ctx, n_channels, n_tensors, table_offset, payload_offset;
    uint64_t payload_beats;
} __attribute__((packed));

/* The LitePCIe kernel write() only consumes whole DMA buffers (8192 B) and returns 0 for a
 * final partial chunk - all transfers are therefore padded to DMA_BUFFER_SIZE multiples
 * (the loader writes the padding beats past the payload into free HBM space, harmless). */
#define DMA_CHUNK 8192

static void write_all(int fd, const uint8_t *buf, size_t len)
{
    while (len) {
        ssize_t r = write(fd, buf, len);
        if (r < 0) {
            if (errno == EAGAIN) { usleep(100); continue; }
            perror("write");
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "write() stalled with %zu bytes left (partial DMA chunk?)\n", len);
            exit(1);
        }
        buf += r;
        len -= r;
    }
}

static void loader_transfer(int fd, int chan, uint32_t base, const uint8_t *data, uint64_t beats)
{
    litepcie_writel(fd, CSR_LOADER_CHAN_ADDR,   chan);
    litepcie_writel(fd, CSR_LOADER_BASE_ADDR,   base);
    litepcie_writel(fd, CSR_LOADER_LENGTH_ADDR, beats);
    litepcie_writel(fd, CSR_LOADER_START_ADDR,  1);

    write_all(fd, data, beats*32);

    int timeout = 10000; /* 10 s */
    while (!(litepcie_readl(fd, CSR_LOADER_STATUS_ADDR) & 1)) {
        usleep(1000);
        if (--timeout == 0) {
            uint32_t st = litepcie_readl(fd, CSR_LOADER_STATUS_ADDR);
            fprintf(stderr, "loader timeout (status %08x, %u beats written)\n",
                    st, (st >> 8) & 0xFFFFFF);
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    int devnum = 0;
    int opt;
    while ((opt = getopt(argc, argv, "c:h")) != -1) {
        if (opt == 'c') devnum = atoi(optarg);
        else { printf("usage: bonsai_load [-c devnum] image.img\n"); exit(0); }
    }
    if (optind >= argc) {
        fprintf(stderr, "usage: bonsai_load [-c devnum] image.img\n");
        exit(1);
    }

    char devname[64];
    snprintf(devname, sizeof(devname), "/dev/bonsai%d", devnum);
    int fd = open(devname, O_RDWR);
    if (fd < 0) { perror(devname); exit(1); }

    FILE *f = fopen(argv[optind], "rb");
    if (!f) { perror(argv[optind]); exit(1); }
    struct bons_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != 0x534E4F42) {
        fprintf(stderr, "not a BONS image\n");
        exit(1);
    }
    printf("BONS image: %u layers, %u channels, %.1f MB/channel\n",
           hdr.n_layer, hdr.n_channels, hdr.payload_beats*32/1e6);
    if (hdr.n_channels != BONSAI_N_CHANNELS) {
        fprintf(stderr, "image has %u channels but the gateware has %u engine channels - "
                "convert with --channels %u\n",
                hdr.n_channels, BONSAI_N_CHANNELS, BONSAI_N_CHANNELS);
        exit(1);
    }

    /* DMA setup: user stream (loopback off), reader (host->device) enabled. */
    litepcie_dma_set_loopback(fd, 0);
    struct litepcie_ioctl_dma_reader reader;
    reader.enable = 1;
    ioctl(fd, LITEPCIE_IOCTL_DMA_READER, &reader);

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    /* All transfers padded to whole DMA buffers (kernel write() granularity); the padding
     * beats land in free HBM space past the payload. */
    #define PAD(bytes) (((uint64_t)(bytes) + DMA_CHUNK - 1) & ~(uint64_t)(DMA_CHUNK - 1))
    uint32_t meta_bytes  = hdr.payload_offset;
    uint64_t chan_bytes  = hdr.payload_beats*32;
    uint64_t buf_size    = PAD(chan_bytes > meta_bytes ? chan_bytes : meta_bytes);
    uint8_t *buf = malloc(buf_size);

    /* Header + table -> channel 0 staging area (firmware copies with `hdrload`). */
    fseek(f, 0, SEEK_SET);
    if (fread(buf, 1, meta_bytes, f) != meta_bytes) { perror("fread"); exit(1); }
    memset(buf + meta_bytes, 0, PAD(meta_bytes) - meta_bytes);
    printf("  header+table -> ch0 staging (%u KB)\n", meta_bytes/1024);
    loader_transfer(fd, 0, HDR_STAGING, buf, PAD(meta_bytes)/32);

    /* Channel payloads. */
    for (uint32_t c = 0; c < hdr.n_channels; c++) {
        fseek(f, (uint64_t)hdr.payload_offset + c*chan_bytes, SEEK_SET);
        if (fread(buf, 1, chan_bytes, f) != chan_bytes) {
            perror("fread");
            exit(1);
        }
        memset(buf + chan_bytes, 0, PAD(chan_bytes) - chan_bytes);
        printf("  channel %u payload (%.1f MB)\n", c, chan_bytes/1e6);
        /* The HBM AXI switch makes the address select the pseudo-channel, so each channel
         * needs its own window - writing them all at one base put every payload on top of
         * the last (confirmed on hardware). The engine adds the same offset on its side. */
        loader_transfer(fd, c, BONSAI_CHAN_WEIGHT_BASE + (uint32_t)c*BONSAI_CHAN_STRIDE,
                        buf, PAD(chan_bytes)/32);
    }

    reader.enable = 0;
    ioctl(fd, LITEPCIE_IOCTL_DMA_READER, &reader);

    gettimeofday(&t1, NULL);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)/1e6;
    double mb = (meta_bytes + (double)hdr.n_channels*hdr.payload_beats*32)/1e6;
    printf("done: %.1f MB in %.2f s (%.2f GB/s)\n", mb, dt, mb/dt/1000);

    free(buf);
    fclose(f);
    close(fd);
    return 0;
}

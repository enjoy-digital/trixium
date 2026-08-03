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

/* Scratch HBM address for the DMA priming transfer below: clear of every channel window
 * (which start at BONSAI_CHAN_WEIGHT_BASE) and of the header staging area beneath them. */
#define LOADER_SCRATCH 0x10000000u

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

/* One transfer: marker beat, then payload.
 *
 * The host->device DMA cannot be flow-controlled. litepcie_dma_reader_start() programs every
 * descriptor and *then* enables the reader, so the card is fed the ring's contents whether or
 * not we have written them, and litepcie_write() silently drops our data whenever the hardware
 * laps it (its underflow path advances the buffer counter without copying). Either way the
 * loader cannot tell payload from filler on its own, and a transfer that starts on the wrong
 * beat writes plausible rubbish - that is how the staged BONS header and the first megabyte of
 * channel 0 were lost, while this tool cheerfully reported success because the byte count it
 * prints is its own.
 *
 * So the stream says where it starts: arm with resync and prefix one SYNC_MAGIC beat. The
 * loader discards everything until it sees the marker and takes the payload from the beat
 * after it. Whatever the DMA invented beforehand is discarded rather than written. */
static const uint8_t SYNC_BEAT[32] = {
    0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0,
    0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0, 0xa5, 0xa5, 0xff, 0xb0,
};

static void loader_transfer(int fd, int chan, uint32_t base, const uint8_t *data, uint64_t beats)
{
    litepcie_writel(fd, CSR_LOADER_CHAN_ADDR,   chan);
    litepcie_writel(fd, CSR_LOADER_BASE_ADDR,   base);
    litepcie_writel(fd, CSR_LOADER_LENGTH_ADDR, beats);
    litepcie_writel(fd, CSR_LOADER_RESYNC_ADDR, 1);
    litepcie_writel(fd, CSR_LOADER_START_ADDR,  1);

    /* The loader discards until it sees a marker, then discards the whole run of them, so a
     * prefix longer than the dropped head of the stream always leaves one standing. Only the
     * first transfer needs that: the drop happens once, while the hardware is lapping a ring
     * software has not filled yet. */
    static int primed = 0;
    uint64_t prefix = primed ? 32 : 2*1024*1024;
    primed = 1;

    uint8_t *markers = malloc(prefix);
    if (!markers) { perror("malloc"); exit(1); }
    for (uint64_t o = 0; o < prefix; o += 32)
        memcpy(markers + o, SYNC_BEAT, 32);
    write_all(fd, markers, prefix);
    free(markers);

    write_all(fd, data, beats*32);

    int timeout = 20000;
    while (!(litepcie_readl(fd, CSR_LOADER_STATUS_ADDR) & 1)) {
        usleep(1000);
        if (--timeout == 0) {
            uint32_t st = litepcie_readl(fd, CSR_LOADER_STATUS_ADDR);
            fprintf(stderr, "loader timeout chan %d addr %08x (status %08x, %u of %llu beats%s)\n",
                    chan, base, st, (st >> 8) & 0xFFFFFF, (unsigned long long)beats,
                    (st & 2) ? ", still hunting for the sync marker" : "");
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

    /* Align the DMA stream with our writes before the first real transfer.
     *
     * litepcie_dma_reader_start() programs all DMA_BUFFER_COUNT descriptors and *then* enables
     * the reader, so the hardware immediately streams a whole ring of buffers we have not
     * written yet - 2 MB of leftovers from whatever ran before. The loader back-pressures
     * rather than discarding while idle, so those leftovers became the head of the first armed
     * transfer: on hardware the staged header and the first megabyte of channel 0 read back as
     * zeros while everything after them was correct and correctly ordered.
     *
     * The loader back-pressures while idle, so the stream stalls between transfers and the
     * misalignment is exactly one ring lap, taken by whichever transfer is armed first. Give
     * that lap a throwaway transfer of its own - armed for exactly one ring, writing nothing -
     * and the hardware wraps back to buffer 0 just as our first write() lands there. */
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

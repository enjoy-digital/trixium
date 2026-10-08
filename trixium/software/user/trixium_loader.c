/* Host-side HBM loader (see trixium_loader.h). */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "trixium_loader.h"

/* Keep in sync with trixium/layout.py and the generated soc.h / csr.h. */
#include "soc.h"
#include "csr.h"
#include "mem.h"
#ifndef TRIXIUM_CHAN_WEIGHT_BASE
#error "SoC headers lack TRIXIUM_CHAN_WEIGHT_BASE (build with variant b)"
#endif
#define HDR_STAGING (TRIXIUM_CHAN_WEIGHT_BASE - 0x100000)

/* main_ram: the CPU's HBM window passes the CPU address through unmodified
 * (add_usphbm2_pseudochannels -> AXILiteOffset(offset=0)), so main_ram is HBM global address
 * MAIN_RAM_BASE = 0x4000_0000 (pseudo-channel 4), and the loader's own port reaches it
 * because the HBM AXI switch routes by address, not by port (layout.py, "port-matched
 * addressing"). Read back through the BIOS to confirm on 2026-09-11: address 0 was not it. */
/* Where the firmware goes: the ENGINE-side address of main_ram's origin. On the FK33 the CPU
 * window passes through 1:1 (HBM address = CPU address); on the XCU1525 main_ram is
 * controller 0's local address 0 and the SoC exports it (TRIXIUM_MEM_CPU_OFFSET = 0) - the
 * first XCU1525 boot put the firmware at DRAM 0x40000000, in channel 1's window, and the BIOS
 * jumped into the memtest's leftovers (2026-09-19). */
#ifdef TRIXIUM_MEM_CPU_OFFSET
#define FW_HBM_ADDR ((uint32_t)TRIXIUM_MEM_CPU_OFFSET)
#else
#define FW_HBM_ADDR ((uint32_t)MAIN_RAM_BASE)
#endif

struct bons_header {
    uint32_t magic, version, n_layer, n_embd, n_head, n_kv_head, head_dim, ffn_dim, vocab;
    uint32_t rope_theta;
    float    rms_eps;
    uint32_t max_ctx, n_channels, n_tensors, table_offset, payload_offset;
    uint64_t payload_beats;
} __attribute__((packed));

/* liblitepcie's litepcie_writel() is an ioctl - a syscall per 32-bit word - which for a
 * gigabyte upload is hundreds of millions of syscalls. BAR0 is mapped uncached through
 * sysfs instead: each store is its own posted write, exactly what a FIFO register wants. */
static volatile uint32_t *bar;
#define BAR_W(a, v) do { bar[((a) - CSR_BASE)/4] = (v); } while (0)
#define BAR_R(a)    (bar[((a) - CSR_BASE)/4])

void loader_bar_map(int devnum)
{
    char path[512], vend[16];
    DIR *d = opendir("/sys/bus/pci/devices");
    struct dirent *e;
    int found = 0, n = 0;

    if (!d) { perror("opendir"); exit(1); }
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        if (fgets(vend, sizeof(vend), f) && strtoul(vend, NULL, 0) == 0x10ee && n++ == devnum) {
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", e->d_name);
            found = 1;
        }
        fclose(f);
        if (found)
            break;
    }
    closedir(d);
    if (!found) { fprintf(stderr, "no Xilinx device %d found\n", devnum); exit(1); }

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror(path); exit(1); }
    void *m = mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap BAR0"); exit(1); }
    close(fd);
    bar = (volatile uint32_t *)m;
}

void loader_cpu_reset(void)
{
    /* Bit 1 = cpu_rst. Bit 0 (soc_rst) would take the PCIe endpoint down with it. */
    BAR_W(CSR_CTRL_RESET_ADDR, 1u << 1);
    usleep(50000);
    BAR_W(CSR_CTRL_RESET_ADDR, 0);
}

static void loader_wait(int chan, uint32_t base, uint64_t beats)
{
    int timeout = 60000;
    while (!(BAR_R(CSR_TRIXIUM_LOADER_STATUS_ADDR) & 1)) {
        usleep(1000);
        if (--timeout == 0) {
            uint32_t st = BAR_R(CSR_TRIXIUM_LOADER_STATUS_ADDR);
            fprintf(stderr, "loader timeout chan %d addr %08x (status %08x, %u of %llu beats%s)\n",
                    chan, base, st, (st >> 8) & 0xFFFFFF, (unsigned long long)beats,
                    (st & 2) ? ", still hunting for the sync marker" : "");
            exit(1);
        }
    }
}

static void loader_transfer_one(int chan, uint32_t base, const uint8_t *data, uint64_t beats);

/* The gateware loader counts a transfer in 24 bits of beats (512 MB); the 27B's 641 MB
 * channel payloads wrapped it and the host spun forever on a full FIFO (2026-09-19).
 * Chunk at 128 MB; each chunk is its own armed transfer at the advanced base. */
void loader_transfer(int chan, uint32_t base, const uint8_t *data, uint64_t beats)
{
    const uint64_t chunk = 4u << 20;   /* beats: 128 MB */
    while (beats > chunk) {
        loader_transfer_one(chan, base, data, chunk);
        base  += (uint32_t)(chunk*32);
        data  += chunk*32;
        beats -= chunk;
    }
    loader_transfer_one(chan, base, data, beats);
}

static void loader_transfer_one(int chan, uint32_t base, const uint8_t *data, uint64_t beats)
{
    BAR_W(CSR_TRIXIUM_LOADER_SRC_SEL_ADDR, 1);
    BAR_W(CSR_TRIXIUM_LOADER_CHAN_ADDR,     chan);
    BAR_W(CSR_TRIXIUM_LOADER_BASE_ADDR,     base);
    BAR_W(CSR_TRIXIUM_LOADER_LENGTH_ADDR,   beats);
    BAR_W(CSR_TRIXIUM_LOADER_RESYNC_ADDR,   0);
    BAR_W(CSR_TRIXIUM_LOADER_START_ADDR,    1);

    const uint32_t *w = (const uint32_t *)data;
    uint64_t nwords = beats*8;
    for (uint64_t i = 0; i < nwords; i++) {
        /* The FIFO drains at HBM speed, so it only fills if the link is unusually fast;
         * check periodically rather than per word. */
        if ((i & 0xFF) == 0)
            while (BAR_R(CSR_TRIXIUM_LOADER_STATUS_ADDR) & 4)
                ;
        BAR_W(CSR_TRIXIUM_LOADER_POKE_ADDR, w[i]);
    }
    loader_wait(chan, base, beats);
}

#define BEATS(bytes) (((uint64_t)(bytes) + 31)/32)

uint64_t loader_upload_image(const char *path, int verbose)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    struct bons_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != 0x534E4F42) {
        fprintf(stderr, "%s: not a BONS image\n", path);
        fclose(f);
        return 0;
    }
    if (hdr.version != 2 && hdr.version != 3) {
        fprintf(stderr, "%s: legacy loader accepts BONS v2/v3, not v%u; "
                "v4 needs the region-aware runtime\n", path, hdr.version);
        fclose(f);
        return 0;
    }
    if (verbose)
        printf("BONS image: %u layers, %u channels, %.1f MB/channel\n",
               hdr.n_layer, hdr.n_channels, hdr.payload_beats*32/1e6);
    if (hdr.n_channels != TRIXIUM_N_CHANNELS) {
        fprintf(stderr, "image has %u channels but the gateware has %u engine channels - "
                "convert with --channels %u\n",
                hdr.n_channels, TRIXIUM_N_CHANNELS, TRIXIUM_N_CHANNELS);
        fclose(f);
        return 0;
    }

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    uint32_t meta_bytes = hdr.payload_offset;
    uint64_t chan_bytes = hdr.payload_beats*32;
    uint64_t buf_size   = 32*BEATS(chan_bytes > meta_bytes ? chan_bytes : meta_bytes);
    uint8_t *buf = malloc(buf_size);
    if (!buf) { perror("malloc"); exit(1); }

    /* Header + table -> channel 0 staging area (the firmware copies with `hdrload`). */
    fseek(f, 0, SEEK_SET);
    if (fread(buf, 1, meta_bytes, f) != meta_bytes) { perror("fread"); exit(1); }
    memset(buf + meta_bytes, 0, 32*BEATS(meta_bytes) - meta_bytes);
    if (verbose)
        printf("  header+table -> ch0 staging (%u KB)\n", meta_bytes/1024);
    loader_transfer(0, HDR_STAGING, buf, BEATS(meta_bytes));

    /* Channel payloads, each to its own window: the HBM AXI switch makes the address
     * select the pseudo-channel, so one shared base put every payload on top of the last
     * (confirmed on hardware, 2026-08-03). */
    for (uint32_t c = 0; c < hdr.n_channels; c++) {
        fseek(f, (uint64_t)hdr.payload_offset + c*chan_bytes, SEEK_SET);
        if (fread(buf, 1, chan_bytes, f) != chan_bytes) { perror("fread"); exit(1); }
        memset(buf + chan_bytes, 0, 32*BEATS(chan_bytes) - chan_bytes);
#ifdef TRIXIUM_LOADER_ADDS_CHAN_OFFSET
        uint32_t csr_base = TRIXIUM_CHAN_WEIGHT_BASE;
#else
        uint32_t csr_base = TRIXIUM_CHAN_WEIGHT_BASE + (uint32_t)c*TRIXIUM_CHAN_STRIDE;
#endif
        if (verbose)
            printf("  -> base %09llx (csr %08x), first words %08x %08x\n",
                   (unsigned long long)TRIXIUM_CHAN_WEIGHT_BASE
                       + (unsigned long long)c*TRIXIUM_CHAN_STRIDE,
                   (unsigned)csr_base, ((uint32_t *)buf)[0], ((uint32_t *)buf)[1]);
        loader_transfer((int)c, csr_base, buf, BEATS(chan_bytes));
    }

    gettimeofday(&t1, NULL);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)/1e6;
    double mb = (meta_bytes + (double)hdr.n_channels*chan_bytes)/1e6;
    printf("done: %.1f MB in %.2f s (%.2f GB/s)\n", mb, dt, mb/dt/1000);
    free(buf);
    fclose(f);
    return (uint64_t)hdr.n_channels*chan_bytes;
}

void loader_clear_staging(void)
{
    /* Zeros with a "cleared" marker in the magic's place: the firmware then also drops a
       header main_ram kept from an earlier session (it would stream weights that are not
       there). The simulation stages nothing and keeps its preloaded header. */
    static uint8_t zeros[4096];
    const uint32_t cleared = 0x44524c43u; /* "CLRD", firmware/bons.h BONS_STAGING_CLEARED */
    memcpy(zeros, &cleared, 4);
    loader_transfer(0, HDR_STAGING, zeros, BEATS(sizeof(zeros)));
}

uint64_t loader_upload_firmware(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > (16 << 20)) {
        fprintf(stderr, "%s: implausible firmware size %ld\n", path, len);
        exit(1);
    }
    uint8_t *img = calloc(BEATS(len), 32);
    if (!img) { perror("calloc"); exit(1); }
    if (fread(img, 1, len, f) != (size_t)len) { perror("fread"); exit(1); }
    fclose(f);
    loader_transfer(0, FW_HBM_ADDR, img, BEATS(len));
    free(img);
    return (uint64_t)len;
}

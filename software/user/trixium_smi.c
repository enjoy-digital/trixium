/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium smi: card status (like nvidia-smi) and LitePCIe diagnostics on /dev/trixiumN
 *
 * This file is part of TriXium, derived from LitePCIe's litepcie_util.
 *
 * Copyright (C) 2018-2023 / EnjoyDigital  / florent@enjoy-digital.fr
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include "liblitepcie.h"

/* Parameters */
/*------------*/

#define DMA_CHECK_DATA   /* Un-comment to disable data check */
#define DMA_RANDOM_DATA  /* Un-comment to disable data random */

/* Variables */
/*-----------*/

static char litepcie_device[1024];
static int litepcie_device_num;

sig_atomic_t keep_running = 1;

void intHandler(int dummy) {
    keep_running = 0;
}

/* Info */
/*------*/

static void info(void)
{
    int fd;
    int i;
    unsigned char fpga_identifier[256];

    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not init driver\n");
        exit(1);
    }


    printf("\e[1m[> FPGA/SoC Info:\e[0m\n");
    printf("-----------------\n");

    for (i = 0; i < 256; i ++)
        fpga_identifier[i] = litepcie_readl(fd, CSR_IDENTIFIER_MEM_BASE + 4 * i);
    printf("SoC Identifier   : %s.\n", fpga_identifier);
#ifdef CSR_DNA_BASE
    printf("FPGA DNA         : 0x%08x%08x\n",
        litepcie_readl(fd, CSR_DNA_ID_ADDR + 4 * 0),
        litepcie_readl(fd, CSR_DNA_ID_ADDR + 4 * 1)
    );
#endif
#ifdef CSR_XADC_BASE
    printf("FPGA Temperature : %0.1f °C\n",
           (double)litepcie_readl(fd, CSR_XADC_TEMPERATURE_ADDR) * 503.975/4096 - 273.15);
    printf("FPGA VCC-INT     : %0.2f V\n",
           (double)litepcie_readl(fd, CSR_XADC_VCCINT_ADDR) / 4096 * 3);
    printf("FPGA VCC-AUX     : %0.2f V\n",
           (double)litepcie_readl(fd, CSR_XADC_VCCAUX_ADDR) / 4096 * 3);
    printf("FPGA VCC-BRAM    : %0.2f V\n",
           (double)litepcie_readl(fd, CSR_XADC_VCCBRAM_ADDR) / 4096 * 3);
#endif
    close(fd);
}

/* Status */
/*--------*/

/* The card's state, like nvidia-smi: identity and link, then the core's counters sampled twice
 * over an interval (engine/vector busy cycles, done ops, beats and request-wait cycles per memory
 * channel), then what the host tools recorded in TRIXIUM_STATE_DIR (weights, chat process). A
 * monitor: it only reads registers, plus perf_chan (the per-channel counter mux, which the
 * firmware only uses in its stall diagnostics). */

#define TRIXIUM_STATE_DIR "/tmp/trixium"
#define STATUS_WIDTH      88
#define BEAT_BYTES        32 /* 256-bit AXI memory ports. */

struct status_sample {
    double   time;
    uint32_t cyc_total, cyc_eng, cyc_vec, cyc_both, done;
    uint32_t beats[TRIXIUM_N_CHANNELS];
    uint32_t ar_wait[TRIXIUM_N_CHANNELS];
    uint32_t pf_hits;
};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void status_sample(int fd, struct status_sample *s)
{
    s->time      = now_s();
    s->cyc_total = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_CYC_TOTAL_ADDR);
    s->cyc_eng   = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_CYC_ENG_ADDR);
    s->cyc_vec   = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_CYC_VEC_ADDR);
    s->cyc_both  = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_CYC_BOTH_ADDR);
    s->done      = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_DONE_COUNT_ADDR);
#ifdef CSR_TRIXIUM_ENGINE_CSR_PF_HITS_ADDR
    s->pf_hits   = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_PF_HITS_ADDR);
#else
    s->pf_hits   = 0;
#endif
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        litepcie_writel(fd, CSR_TRIXIUM_ENGINE_CSR_PERF_CHAN_ADDR, c);
        s->beats[c]   = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_PERF_BEATS_ADDR);
        s->ar_wait[c] = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_PERF_AR_ADDR);
    }
}

/* PCIe link of the N-th Xilinx device (the driver creates its nodes without a PCI parent). */
static int pcie_link(int devnum, char *addr, char *link, size_t n)
{
    struct dirent **names;
    char path[512], speed[32] = "", width[16] = "", max_speed[32] = "", max_width[16] = "";
    int count = scandir("/sys/bus/pci/devices", &names, NULL, alphasort);
    int index = 0, found = 0;

    for (int i = 0; i < count; i++) {
        if (!found && names[i]->d_name[0] != '.') {
            char vendor[16] = "";
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", names[i]->d_name);
            FILE *f = fopen(path, "r");
            if (f) {
                if (fgets(vendor, sizeof(vendor), f) && strtoul(vendor, NULL, 0) == 0x10ee &&
                    index++ == devnum) {
                    snprintf(addr, 32, "%.31s", names[i]->d_name);
                    found = 1;
                }
                fclose(f);
            }
        }
        free(names[i]);
    }
    if (count >= 0)
        free(names);
    if (!found)
        return 0;

    const char *attrs[4] = {"current_link_speed", "current_link_width",
                            "max_link_speed", "max_link_width"};
    char *values[4] = {speed, width, max_speed, max_width};
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", addr, attrs[i]);
        FILE *f = fopen(path, "r");
        if (f) {
            if (fgets(values[i], i % 2 ? 16 : 32, f))
                values[i][strcspn(values[i], "\n")] = 0;
            fclose(f);
        }
    }
    /* "8.0 GT/s PCIe" -> Gen3. */
    int gen = (int)(atof(speed)/2.5 + 0.5), max_gen = (int)(atof(max_speed)/2.5 + 0.5);
    gen     = gen     >= 6 ? 4 : gen     >= 3 ? 3 : gen;
    max_gen = max_gen >= 6 ? 4 : max_gen >= 3 ? 3 : max_gen;
    if (gen == max_gen && !strcmp(width, max_width))
        snprintf(link, n, "PCIe Gen%d x%s", gen, width);
    else
        snprintf(link, n, "PCIe Gen%d x%s (slot Gen%d x%s)", gen, width, max_gen, max_width);
    return 1;
}

/* The host tools' records: {"key": value, ...} as written by loader_state_write. */
static int state_read(int devnum, const char *kind, char *buf, size_t n)
{
    char path[256];
    snprintf(path, sizeof(path), TRIXIUM_STATE_DIR "/trixium%d.%s.json", devnum, kind);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    size_t len = fread(buf, 1, n - 1, f);
    buf[len] = 0;
    fclose(f);
    return len > 0;
}

static double state_num(const char *buf, const char *key)
{
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\": ", key);
    const char *p = strstr(buf, pattern);
    return p ? atof(p + strlen(pattern)) : 0.0;
}

static void state_str(const char *buf, const char *key, char *out, size_t n)
{
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\": \"", key);
    const char *p = strstr(buf, pattern);
    out[0] = 0;
    if (p) {
        p += strlen(pattern);
        size_t len = strcspn(p, "\"");
        snprintf(out, n, "%.*s", (int)(len < n ? len : n - 1), p);
    }
}

/* Table output (UTF-8 box drawing, padded by visible characters). In a terminal (NO_COLOR
 * unset): the diagrams' palette, teal bars and section titles, muted frame. */
static int g_color;
static const char *ui(const char *seq) { return g_color ? seq : ""; }
#define UI_TEAL  ui("\033[38;2;53;194;174m")
#define UI_DIM   ui("\033[38;2;95;120;113m")
#define UI_BOLD  ui("\033[1m")
#define UI_RESET ui("\033[0m")

static int utf8_width(const char *s)
{
    int n = 0;
    for (; *s; s++) {
        if (*s == '\033') {                     /* Colour escapes take no column. */
            while (*s && *s != 'm') s++;
            if (!*s) break;
            continue;
        }
        n += ((*s & 0xc0) != 0x80);
    }
    return n;
}

static void status_line(const char *fmt, ...)
{
    char text[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    int pad = STATUS_WIDTH - 4 - utf8_width(text);
    printf("%s│%s %s%*s %s│%s\n", UI_DIM, UI_RESET, text, pad > 0 ? pad : 0, "", UI_DIM, UI_RESET);
}

static void status_head(const char *title, int first)
{
    int n = STATUS_WIDTH - 2;
    printf("%s%s", UI_DIM, first ? "╭" : "├");
    if (title[0]) {
        printf("─ %s%s%s%s%s ", UI_RESET, UI_BOLD, UI_TEAL, title, UI_DIM);
        n -= utf8_width(title) + 3;
    }
    for (int i = 0; i < n; i++)
        printf("─");
    printf("%s%s\n", first ? "╮" : "┤", UI_RESET);
}

static void status_bar(char *out, double x, int n)
{
    int k = (int)(x*n + 0.5);
    k = k < 0 ? 0 : k > n ? n : k;
    strcpy(out, UI_TEAL);
    for (int i = 0; i < n; i++) {
        if (i == k)
            strcat(out, UI_DIM);
        strcat(out, i < k ? "█" : "░");
    }
    strcat(out, UI_RESET);
}

static void status(int devnum, double interval, int json)
{
    struct status_sample s0, s1;
    char ident[256], addr[32] = "", link[96] = "PCIe n/a";
    int fd;

    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not open %s (is the driver loaded?)\n", litepcie_device);
        exit(1);
    }

    /* Identity and state. */
    for (int i = 0; i < 255; i++)
        ident[i] = litepcie_readl(fd, CSR_IDENTIFIER_MEM_BASE + 4*i);
    ident[255] = 0;
    pcie_link(devnum, addr, link, sizeof(link));
    uint32_t eng_status = litepcie_readl(fd, CSR_TRIXIUM_ENGINE_CSR_STATUS_ADDR);
    uint32_t vec_status = litepcie_readl(fd, CSR_TRIXIUM_VECUNIT_STATUS_ADDR);
    uint32_t bus_errors = litepcie_readl(fd, CSR_CTRL_BUS_ERRORS_ADDR);
    int eng_error = (eng_status >> CSR_TRIXIUM_ENGINE_CSR_STATUS_ERROR_OFFSET) & 1;
    int eng_run   = (eng_status >> CSR_TRIXIUM_ENGINE_CSR_STATUS_RUN_OFFSET) & 1;
    int eng_busy  = (eng_status >> CSR_TRIXIUM_ENGINE_CSR_STATUS_BUSY_OFFSET) & 1;
    int vec_bad   = (vec_status >> CSR_TRIXIUM_VECUNIT_STATUS_BAD_OP_OFFSET) & 1;
#ifdef CSR_CPUMON_CPU_RESET_ADDR
    const char *cpu = (litepcie_readl(fd, CSR_CPUMON_CPU_RESET_ADDR) & 1) ? "in reset" : "running";
#else
    const char *cpu = "n/a";
#endif

    /* Counters over the interval (32-bit, wrap-safe over a short interval). */
    status_sample(fd, &s0);
    usleep((useconds_t)(interval*1e6));
    status_sample(fd, &s1);
    close(fd);
    double   dt     = s1.time - s0.time;
    uint32_t cycles = s1.cyc_total - s0.cyc_total;
    double   clk    = dt > 0 ? cycles/dt : 0;
    double   u_eng  = cycles ? (double)(uint32_t)(s1.cyc_eng  - s0.cyc_eng)/cycles  : 0;
    double   u_vec  = cycles ? (double)(uint32_t)(s1.cyc_vec  - s0.cyc_vec)/cycles  : 0;
    double   u_both = cycles ? (double)(uint32_t)(s1.cyc_both - s0.cyc_both)/cycles : 0;
    double   ops_s  = dt > 0 ? (uint32_t)(s1.done - s0.done)/dt : 0;
    double   ch_util[TRIXIUM_N_CHANNELS], gbs = 0, ar_wait = 0;
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        uint32_t beats = s1.beats[c] - s0.beats[c];
        ch_util[c] = cycles ? (double)beats/cycles : 0;
        gbs       += dt > 0 ? beats*(double)BEAT_BYTES/dt/1e9 : 0;
        ar_wait   += cycles ? (double)(uint32_t)(s1.ar_wait[c] - s0.ar_wait[c])/cycles : 0;
    }
    ar_wait /= TRIXIUM_N_CHANNELS;
    double peak_gbs = TRIXIUM_N_CHANNELS*BEAT_BYTES*clk/1e9; /* The engine ports'. */
    double pf_s     = dt > 0 ? (uint32_t)(s1.pf_hits - s0.pf_hits)/dt : 0;
    /* The memory and, where the engine ports outrun it, the DRAM's own peak (the bar's
     * reference): the YPCB's two 64-bit DDR3-1066 channels, 17.1 GB/s. */
#ifdef HBM_CHANNELS
    const char *memory = "HBM2";
    const char *peak_what = "port peak";
#else
    const char *memory = "DDR4";
    const char *peak_what = "port peak";
#ifdef CONFIG_PLATFORM_NAME
    if (strstr(CONFIG_PLATFORM_NAME, "ypcb")) {
        memory    = "DDR3-1066";
        peak_gbs  = SDRAM_CONTROLLERS*8*(3200e6/3)/1e9;
        peak_what = "DRAM peak";
    }
#endif
#endif

    /* Host records. */
    char load[2048], chat[2048], image[1024] = "", tool[64] = "", state[32] = "", model[64] = "";
    int has_load = state_read(devnum, "load", load, sizeof(load));
    int has_chat = state_read(devnum, "chat", chat, sizeof(chat));
    int chat_alive = 0;
    if (has_load)
        state_str(load, "image", image, sizeof(image));
    if (has_chat) {
        state_str(chat, "tool",  tool,  sizeof(tool));
        state_str(chat, "state", state, sizeof(state));
        state_str(chat, "model", model, sizeof(model));
        chat_alive = kill((pid_t)state_num(chat, "pid"), 0) == 0 || errno == EPERM;
    }

    if (json) {
        printf("{\"device\": \"%s\", \"pci\": \"%s\", \"link\": \"%s\", \"identifier\": \"%s\",\n",
               litepcie_device, addr, link, ident);
        printf(" \"clock_mhz\": %.3f, \"cpu\": \"%s\", \"bus_errors\": %u, \"engine_error\": %d, "
               "\"vector_bad_op\": %d,\n", clk/1e6, cpu, bus_errors, eng_error, vec_bad);
        printf(" \"seconds\": %.3f, \"engine\": %.4f, \"vector\": %.4f, \"overlap\": %.4f, "
               "\"ops_s\": %.1f, \"prefetch_s\": %.1f,\n", dt, u_eng, u_vec, u_both, ops_s, pf_s);
        printf(" \"memory\": \"%s\", \"channels\": %d, \"gbs\": %.3f, \"peak_gbs\": %.3f, "
               "\"request_wait\": %.4f, \"channel_util\": [", memory, TRIXIUM_N_CHANNELS, gbs,
               peak_gbs, ar_wait);
        for (int c = 0; c < TRIXIUM_N_CHANNELS; c++)
            printf("%s%.4f", c ? ", " : "", ch_util[c]);
        printf("],\n \"load\": %s,\n \"chat\": %s}\n",
               has_load ? load : "null", has_chat && chat_alive ? chat : "null");
        return;
    }

    char bar[192], title[160], date[64];
    time_t t = time(NULL);
    strftime(date, sizeof(date), "%a %b %d %H:%M:%S %Y", localtime(&t));
    printf("%-*s%s\n", STATUS_WIDTH - (int)strlen(date), "trixium_smi", date);
    snprintf(title, sizeof(title), "Card %d · %.64s%s%s", devnum, litepcie_device,
             addr[0] ? " · " : "", addr);
    status_head(title, 1);
    status_line("%-12s%s", "Bitstream", ident);
    status_line("%-12s%-26ssys %.1f MHz (measured)", "Link", link, clk/1e6);
    /* Sticky flags: the engine's error bit (reserved ternary code, result FIFO overflow or an
     * attention record outside its geometry), the vector unit's unknown opcode, bus errors. */
    char errors[96] = "";
    if (eng_error)
        strcat(errors, "engine-err ");
    if (vec_bad)
        strcat(errors, "vec-badop ");
    if (bus_errors)
        snprintf(errors + strlen(errors), sizeof(errors) - strlen(errors), "bus-err %u", bus_errors);
    if (!errors[0])
        strcpy(errors, "no errors");
    status_line("%-12sCPU %-22sengine %-19s%s", "State", cpu,
                eng_run ? "streaming" : eng_busy ? "busy" : "idle", errors);

    snprintf(title, sizeof(title), "Utilization over %.0f ms", dt*1e3);
    status_head(title, 0);
    char bar2[192];
    status_bar(bar, u_eng, 10);
    status_bar(bar2, u_vec, 10);
    status_line("Engine  %s %3.0f%%     Vector  %s %3.0f%%     Overlap %3.0f%%",
                bar, 100*u_eng, bar2, 100*u_vec, 100*u_both);
#ifdef CSR_TRIXIUM_ENGINE_CSR_PF_HITS_ADDR
    status_line("%-10.0f engine ops/s   %.0f served from prefetch/s", ops_s, pf_s);
#else
    status_line("%.0f engine ops/s", ops_s);
#endif

    snprintf(title, sizeof(title), "Memory · %s · %d channels", memory, TRIXIUM_N_CHANNELS);
    status_head(title, 0);
    status_bar(bar, peak_gbs > 0 ? gbs/peak_gbs : 0, 10);
    status_line("Read    %s %6.2f GB/s of %.1f GB/s %s   request wait %3.0f%%",
                bar, gbs, peak_gbs, peak_what, 100*ar_wait);
    for (int c0 = 0; c0 < TRIXIUM_N_CHANNELS; c0 += 6) {
        char row[1536] = "", cell[256];
        for (int c = c0; c < c0 + 6 && c < TRIXIUM_N_CHANNELS; c++) {
            status_bar(bar, ch_util[c], 4);
            snprintf(cell, sizeof(cell), "%s%2d %s%3.0f%%", c > c0 ? "  " : "", c, bar,
                     100*ch_util[c]);
            strcat(row, cell);
        }
        status_line("%s", row);
    }

    status_head("Model", 0);
    if (has_load) {
        const char *name = strrchr(image, '/');
        double age = difftime(time(NULL), (time_t)state_num(load, "time"));
        char ago[32];
        if (age < 120)
            snprintf(ago, sizeof(ago), "%.0f s ago", age);
        else if (age < 7200)
            snprintf(ago, sizeof(ago), "%.0f min ago", age/60);
        else
            snprintf(ago, sizeof(ago), "%.1f h ago", age/3600);
        status_line("%-12s%s  %.2f GiB, uploaded %s", "Weights", name ? name + 1 : image,
                    state_num(load, "bytes")/(1 << 30), ago);
    } else
        status_line("%-12sunknown (no upload recorded)", "Weights");
    if (has_chat && chat_alive) {
        status_line("%-12spid %.0f %s%s%s  %s, %.0f tokens", "Process",
                    state_num(chat, "pid"), tool, model[0] ? " · " : "", model, state,
                    state_num(chat, "tokens"));
        /* The reply's decode rate (live while generating) and its time to first token. */
        double rate = state_num(chat, "tok_s");
        status_bar(bar, rate > 0 ? (rate >= 40 ? 1.0 : rate/40) : 0, 10);
        status_line("%-12s%s %6.2f tok/s   first token %.2f s", "Decode", bar, rate,
                    state_num(chat, "ttft"));
    } else
        status_line("%-12snone", "Process");
    printf("%s╰", UI_DIM);
    for (int i = 0; i < STATUS_WIDTH - 2; i++)
        printf("─");
    printf("╯%s\n", UI_RESET);
}

/* Scratch */
/*---------*/

void scratch_test(void)
{
    int fd;

    printf("\e[1m[> Scratch register test:\e[0m\n");
    printf("-------------------------\n");

    /* Open LitePCIe device. */
    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not init driver\n");
        exit(1);
    }

    /* Write to scratch register. */
    printf("Write 0x12345678 to Scratch register:\n");
    litepcie_writel(fd, CSR_CTRL_SCRATCH_ADDR, 0x12345678);
    printf("Read: 0x%08x\n", litepcie_readl(fd, CSR_CTRL_SCRATCH_ADDR));

    /* Read from scratch register. */
    printf("Write 0xdeadbeef to Scratch register:\n");
    litepcie_writel(fd, CSR_CTRL_SCRATCH_ADDR, 0xdeadbeef);
    printf("Read: 0x%08x\n", litepcie_readl(fd, CSR_CTRL_SCRATCH_ADDR));

    /* Close LitePCIe device. */
    close(fd);
}

/* SPI Flash */
/*-----------*/

#ifdef CSR_FLASH_BASE

static void flash_progress(void *opaque, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    fflush(stdout);
    va_end(ap);
}

static void flash_program(uint32_t base, const uint8_t *buf1, int size1)
{
    int fd;

    uint32_t size;
    uint8_t *buf;
    int sector_size;
    int errors;

    /* Open LitePCIe device. */
    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not init driver\n");
        exit(1);
    }

    /* Get flash sector size and pad size to it. */
    sector_size = litepcie_flash_get_erase_block_size(fd);
    size = ((size1 + sector_size - 1) / sector_size) * sector_size;

    /* Alloc buffer and copy data to it. */
    buf = calloc(1, size);
    if (!buf) {
        fprintf(stderr, "%d: alloc failed\n", __LINE__);
        exit(1);
    }
    memcpy(buf, buf1, size1);

    /* Program flash. */
    printf("Programming (%d bytes at 0x%08x)...\n", size, base);
    errors = litepcie_flash_write(fd, buf, base, size, flash_progress, NULL);
    if (errors) {
        printf("Failed %d errors.\n", errors);
        exit(1);
    } else {
        printf("Success.\n");
    }

    /* Free buffer and close LitePCIe device. */
    free(buf);
    close(fd);
}

static void flash_write(const char *filename, uint32_t offset)
{
    uint8_t *data;
    int size;
    FILE * f;

    /* Open data source file. */
    f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    /* Get size, alloc buffer and copy data to it. */
    fseek(f, 0L, SEEK_END);
    size = ftell(f);
    fseek(f, 0L, SEEK_SET);
    data = malloc(size);
    if (!data) {
        fprintf(stderr, "%d: malloc failed\n", __LINE__);
        exit(1);
    }
    ssize_t ret = fread(data, size, 1, f);
    fclose(f);

    /* Program file to flash */
    if (ret != 1)
        perror(filename);
    else
        flash_program(offset, data, size);

    /* Free buffer */
    free(data);
}

static void flash_read(const char *filename, uint32_t size, uint32_t offset)
{
    int fd;
    FILE * f;
    uint32_t base;
    uint32_t sector_size;
    uint8_t byte;
    int i;

    /* Open data destination file. */
    f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    /* Open LitePCIe device. */
    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not init driver\n");
        exit(1);
    }

    /* Get flash sector size. */
    sector_size = litepcie_flash_get_erase_block_size(fd);

    /* Read flash and write to destination file. */
    base = offset;
    for (i = 0; i < size; i++) {
        if ((i % sector_size) == 0) {
            printf("Reading 0x%08x\r", base + i);
            fflush(stdout);
        }
        byte = litepcie_flash_read(fd, base + i);
        fwrite(&byte, 1, 1, f);
    }

    /* Close destination file and LitePCIe device. */
    fclose(f);
    close(fd);
}

static void flash_reload(void)
{
    int fd;

    /* Open LitePCIe device. */
    fd = open(litepcie_device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not init driver\n");
        exit(1);
    }

    /* Reload FPGA through ICAP.*/
    litepcie_reload(fd);

    /* Notice user to reboot/rescan the hardware.*/
    printf("===========================================================================\n");
    printf("= PLEASE REBOOT YOUR HARDWARE OR RESCAN PCIe BUS TO USE NEW FPGA GATEWARE =\n");
    printf("===========================================================================\n");

    /* Close LitePCIe device. */
    close(fd);
}
#endif

/* DMA */
/*-----*/

static inline int64_t add_mod_int(int64_t a, int64_t b, int64_t m)
{
    a += b;
    if (a >= m)
        a -= m;
    return a;
}

static int get_next_pow2(int data_width)
{
    int x = 1;
    while (x < data_width)
        x <<= 1;
    return x;
}

#ifdef DMA_CHECK_DATA

static inline uint32_t seed_to_data(uint32_t seed)
{
#ifdef DMA_RANDOM_DATA
    /* Return pseudo random data from seed. */
    return seed * 69069 + 1;
#else
    /* Return seed. */
    return seed;
#endif
}

static uint32_t get_data_mask(int data_width)
{
    int i;
    uint32_t mask;
    mask = 0;
    for (i = 0; i < 32/get_next_pow2(data_width); i++) {
        mask <<= get_next_pow2(data_width);
        mask |= ((uint64_t) 1 << data_width) - 1;
    }
    return mask;
}

static void write_pn_data(uint32_t *buf, int count, uint32_t *pseed, int data_width)
{
    int i;
    uint32_t seed;
    uint32_t mask = get_data_mask(data_width);

    seed = *pseed;
    for(i = 0; i < count; i++) {
        buf[i] = (seed_to_data(seed) & mask);
        seed = add_mod_int(seed, 1, DMA_BUFFER_SIZE / sizeof(uint32_t));
    }
    *pseed = seed;
}

static int check_pn_data(const uint32_t *buf, int count, uint32_t *pseed, int data_width)
{
    int i, errors;
    uint32_t seed;
    uint32_t mask = get_data_mask(data_width);

    errors = 0;
    seed = *pseed;
    for (i = 0; i < count; i++) {
        if ((buf[i] & mask) != (seed_to_data(seed) & mask)) {
            errors ++;
        }
        seed = add_mod_int(seed, 1, DMA_BUFFER_SIZE / sizeof(uint32_t));
    }
    *pseed = seed;
    return errors;
}
#endif

static void dma_test(uint8_t zero_copy, uint8_t external_loopback, int data_width, int auto_rx_delay, int duration)
{
    static struct litepcie_dma_ctrl dma = {.use_reader = 1, .use_writer = 1};
    dma.loopback = external_loopback ? 0 : 1;

    if (data_width > 32 || data_width < 1) {
        fprintf(stderr, "Invalid data width %d\n", data_width);
        exit(1);
    }

    /* Statistics */
    int i = 0;
    int64_t reader_sw_count_last = 0;
    int64_t last_time;
    uint32_t errors = 0;
    int64_t end_time = (duration > 0) ? get_time_ms() + duration * 1000 : 0;

#ifdef DMA_CHECK_DATA
    uint32_t seed_wr = 0;
    uint32_t seed_rd = 0;
    uint8_t  run = (auto_rx_delay == 0);
#else
    uint8_t run = 1;
#endif

    signal(SIGINT, intHandler);

    printf("\e[1m[> DMA loopback test:\e[0m\n");
    printf("---------------------\n");

    if (litepcie_dma_init(&dma, litepcie_device, zero_copy))
        exit(1);

    dma.reader_enable = 1;
    dma.writer_enable = 1;

    /* Test loop. */
    last_time = get_time_ms();
    for (;;) {
        /* Exit loop on CTRL+C or when the duration is over. */
        if (!keep_running || (duration > 0 && get_time_ms() >= end_time))
            break;

        /* Update DMA status. */
        litepcie_dma_process(&dma);

#ifdef DMA_CHECK_DATA
        char *buf_wr;
        char *buf_rd;

        /* DMA-TX Write. */
        while (1) {
            /* Get Write buffer. */
            buf_wr = litepcie_dma_next_write_buffer(&dma);
            /* Break when no buffer available for Write. */
            if (!buf_wr)
                break;
            /* Write data to buffer. */
            write_pn_data((uint32_t *) buf_wr, DMA_BUFFER_SIZE / sizeof(uint32_t), &seed_wr, data_width);
        }

        /* DMA-RX Read/Check */
        while (1) {
            /* Get Read buffer. */
            buf_rd = litepcie_dma_next_read_buffer(&dma);
            /* Break when no buffer available for Read. */
            if (!buf_rd)
                break;
            /* Skip the first 128 DMA loops. */
            if (dma.writer_hw_count < 128*DMA_BUFFER_COUNT)
                break;
            /* When running... */
            if (run) {
                /* Check data in Read buffer. */
                errors += check_pn_data((uint32_t *) buf_rd, DMA_BUFFER_SIZE / sizeof(uint32_t), &seed_rd, data_width);
                /* Clear Read buffer */
                memset(buf_rd, 0, DMA_BUFFER_SIZE);
            } else {
                /* Find initial Delay/Seed (Useful when loopback is introducing delay). */
                uint32_t errors_min = 0xffffffff;
                for (int delay = 0; delay < DMA_BUFFER_SIZE / sizeof(uint32_t); delay++) {
                    seed_rd = delay;
                    errors = check_pn_data((uint32_t *) buf_rd, DMA_BUFFER_SIZE / sizeof(uint32_t), &seed_rd, data_width);
                    //printf("delay: %d / errors: %d\n", delay, errors);
                    if (errors < errors_min)
                        errors_min = errors;
                    if (errors < (DMA_BUFFER_SIZE / sizeof(uint32_t)) / 2) {
                        printf("RX_DELAY: %d (errors: %d)\n", delay, errors);
                        run = 1;
                        break;
                    }
                }
                if (!run) {
                    printf("Unable to find DMA RX_DELAY (min errors: %d/%ld), exiting.\n",
                        errors_min,
                        DMA_BUFFER_SIZE / sizeof(uint32_t));
                    goto end;
                }
            }

        }
#endif

        /* Statistics every 200ms. */
        int64_t duration_ms = get_time_ms() - last_time;
        if (run & (duration_ms > 200)) {
            /* Print banner every 10 lines. */
            if (i % 10 == 0)
                printf("\e[1mDMA_SPEED(Gbps)\tTX_BUFFERS\tRX_BUFFERS\tDIFF\tERRORS\e[0m\n");
            i++;
            /* Print statistics. */
            printf("%14.2f\t%10" PRIu64 "\t%10" PRIu64 "\t%4" PRIu64 "\t%6u\n",
                   (double)(dma.reader_sw_count - reader_sw_count_last) * DMA_BUFFER_SIZE * 8 * data_width / (get_next_pow2(data_width) * (double)duration_ms * 1e6),
                   dma.reader_sw_count,
                   dma.writer_sw_count,
                   (uint64_t) abs(dma.reader_sw_count - dma.writer_sw_count),
                   errors);
            /* Update errors/time/count. */
            errors = 0;
            last_time = get_time_ms();
            reader_sw_count_last = dma.reader_sw_count;
        }
    }

    /* Cleanup DMA. */
#ifdef DMA_CHECK_DATA
end:
#endif
    litepcie_dma_cleanup(&dma);
}

/* Help */
/*------*/

static void help(int status)
{
    fprintf(status ? stderr : stdout,"TriXium smi: card status and diagnostics\n"
           "usage: trixium_smi [options] [cmd [args...]]\n"
           "\n"
           "options:\n"
           "-h                                Help.\n"
           "-c device_num                     Select the device (default = 0).\n"
           "-l seconds                        Status: refresh every N seconds.\n"
           "-i seconds                        Status: counter sampling interval (default = 0.25).\n"
           "-j                                Status: JSON output.\n"
           "-z                                DMA test: zero-copy mode.\n"
           "-e                                DMA test: external loopback (default = internal).\n"
           "-w data_width                     DMA test: data bus width (default = 16).\n"
           "-a                                DMA test: automatic RX-delay calibration.\n"
           "-t duration                       DMA test: duration in seconds (default = 0, infinite).\n"
           "\n"
           "available commands:\n"
           "(none)                            Card status: bitstream, link, clock, utilization,\n"
           "                                  memory GB/s per channel, model and process.\n"
           "info                              FPGA identifier, DNA, temperature.\n"
           "scratch_test                      Test the scratch register.\n"
           "dma_test                          Test the PCIe DMA.\n"
#ifdef CSR_FLASH_BASE
           "flash_write filename [offset]     Write file contents to SPI Flash.\n"
           "flash_read filename size [offset] Read from SPI Flash and write contents to file.\n"
           "flash_reload                      Reload FPGA Image.\n"
#endif
           );
    exit(status);
}

/* Main */
/*------*/

int main(int argc, char **argv)
{
    g_color = isatty(STDOUT_FILENO) && !getenv("NO_COLOR");
    const char *cmd;
    int c;
    static uint8_t litepcie_device_zero_copy;
    static uint8_t litepcie_device_external_loopback;
    static int litepcie_data_width;
    static int litepcie_auto_rx_delay;
    static int test_duration = 0; /* Default to 0 for infinite duration.*/
    double status_loop     = 0;
    double status_interval = 0.25;
    int    status_json     = 0;

    litepcie_device_num = 0;
    litepcie_data_width = 16;
    litepcie_auto_rx_delay = 0;
    litepcie_device_zero_copy = 0;
    litepcie_device_external_loopback = 0;

    /* Parameters. */
    for (;;) {
        c = getopt(argc, argv, "hc:w:zeat:l:i:j");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help(0);
            break;
        case 'c':
            litepcie_device_num = atoi(optarg);
            break;
        case 'w':
            litepcie_data_width = atoi(optarg);
            break;
        case 'z':
            litepcie_device_zero_copy = 1;
            break;
        case 'e':
            litepcie_device_external_loopback = 1;
            break;
        case 'a':
            litepcie_auto_rx_delay = 1;
            break;
        case 't':
            test_duration = atoi(optarg);
            break;
        case 'l':
            status_loop = atof(optarg);
            break;
        case 'i':
            status_interval = atof(optarg);
            break;
        case 'j':
            status_json = 1;
            break;
        default:
            help(1);
        }
    }

    /* Select device. */
    snprintf(litepcie_device, sizeof(litepcie_device), "/dev/trixium%d", litepcie_device_num);

    /* Status (default command). */
    if (optind >= argc) {
        do {
            if (status_loop > 0 && !status_json)
                printf("\033[H\033[J");
            if (status_json) g_color = 0;
            status(litepcie_device_num, status_interval, status_json);
            fflush(stdout);
            if (status_loop > 0)
                usleep((useconds_t)(status_loop*1e6));
        } while (status_loop > 0);
        return 0;
    }

    cmd = argv[optind++];

    /* Info cmds. */
    if (!strcmp(cmd, "info"))
        info();
    /* Scratch cmds. */
    else if (!strcmp(cmd, "scratch_test"))
        scratch_test();
    /* SPI Flash cmds. */
#if CSR_FLASH_BASE
    else if (!strcmp(cmd, "flash_write")) {
        const char *filename;
        uint32_t offset = 0;
        if (optind + 1 > argc)
            goto show_help;
        filename = argv[optind++];
        if (optind < argc)
            offset = strtoul(argv[optind++], NULL, 0);
        flash_write(filename, offset);
    }
    else if (!strcmp(cmd, "flash_read")) {
        const char *filename;
        uint32_t size = 0;
        uint32_t offset = 0;
        if (optind + 2 > argc)
            goto show_help;
        filename = argv[optind++];
        size = strtoul(argv[optind++], NULL, 0);
        if (optind < argc)
            offset = strtoul(argv[optind++], NULL, 0);
        flash_read(filename, size, offset);
    }
    else if (!strcmp(cmd, "flash_reload"))
        flash_reload();
#endif

    /* DMA cmds. */
    else if (!strcmp(cmd, "dma_test"))
        dma_test(
            litepcie_device_zero_copy,
            litepcie_device_external_loopback,
            litepcie_data_width,
            litepcie_auto_rx_delay,
            test_duration);

    /* Show help otherwise. */
    else
        goto show_help;

    return 0;

show_help:
        help(1);

    return 0;
}

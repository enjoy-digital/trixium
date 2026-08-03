/* This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * bonsai_chat: one-stop host tool - configure the accelerator, then talk to the model.
 *
 *   1. (-m) upload a BONS weight image to HBM over the PCIe DMA (same path as bonsai_load)
 *   2. (-k) boot the firmware over the LiteX serial-boot protocol on the console UART
 *   3.      `hdrload` so the firmware picks up the staged BONS header/table
 *   4.      REPL: your text -> BPE ids -> `tok`/`genb` -> ids back -> text
 *
 * The model runs entirely on the card; this tool only tokenises, drives the console and
 * detokenises. Tokeniser data comes from tools/export_tokenizer.py (raw-byte vocab +
 * merge ranks), so no JSON or unicode tables are needed here.
 *
 * Usage:
 *   bonsai_chat -T                                   # tokeniser self-test, no hardware
 *   bonsai_chat -m models/bons/bonsai-1p7b-n8.img \
 *               -k firmware/bonsai.bin               # full bring-up, then chat
 *   bonsai_chat                                      # attach to an already-configured card
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#include "liblitepcie.h"
#include "soc.h"
#include "mem.h"

/* Firmware load/jump address: the BIOS serialboot target is main_ram (HBM channel 0 on
 * hardware). ROM_BOOT_ADDRESS only exists in the simulation SoC. */
#ifndef ROM_BOOT_ADDRESS
#define ROM_BOOT_ADDRESS MAIN_RAM_BASE
#endif

#define HDR_STAGING  (BONSAI_CHAN_WEIGHT_BASE - 0x100000)
#define DMA_CHUNK    8192
#define PAD(b)       (((uint64_t)(b) + DMA_CHUNK - 1) & ~(uint64_t)(DMA_CHUNK - 1))

/*---------------------------------------------------------------------------------------*/
/* Tokeniser: byte-level BPE over the exported vocab (raw bytes) + merge ranks.           */
/*---------------------------------------------------------------------------------------*/

struct tok_entry { uint32_t len; const uint8_t *bytes; };

struct merge_slot { uint64_t key; uint32_t rank, result; };

static struct tok_entry *g_vocab;
static uint32_t          g_nvocab;
static uint8_t          *g_blob;          /* Owns all token bytes. */
static struct merge_slot *g_mtab;         /* Open-addressed (a,b) -> rank/result. */
static uint32_t          g_mmask;
static int32_t           g_byte_id[256];  /* Single-byte token ids. */

static uint64_t mkey(uint32_t a, uint32_t b) { return ((uint64_t)a << 32) | b; }

static uint64_t hash64(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

static const struct merge_slot *merge_find(uint32_t a, uint32_t b)
{
    uint64_t k = mkey(a, b);
    uint32_t i = (uint32_t)(hash64(k) & g_mmask);
    while (g_mtab[i].key != UINT64_MAX) {
        if (g_mtab[i].key == k)
            return &g_mtab[i];
        i = (i + 1) & g_mmask;
    }
    return NULL;
}

static int tok_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    char magic[4];
    uint32_t ver, nvocab, nmerges;
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, "BTOK", 4) ||
        fread(&ver, 4, 1, f) != 1 || fread(&nvocab, 4, 1, f) != 1 ||
        fread(&nmerges, 4, 1, f) != 1) {
        fprintf(stderr, "%s: not a BTOK tokeniser (run tools/export_tokenizer.py)\n", path);
        fclose(f); return -1;
    }

    long here = ftell(f);
    fseek(f, 0, SEEK_END);
    long rest = ftell(f) - here;
    fseek(f, here, SEEK_SET);
    g_blob = malloc(rest);
    if (fread(g_blob, 1, rest, f) != (size_t)rest) { perror("fread"); fclose(f); return -1; }
    fclose(f);

    g_vocab  = calloc(nvocab, sizeof(*g_vocab));
    g_nvocab = nvocab;
    memset(g_byte_id, -1, sizeof(g_byte_id));
    uint8_t *p = g_blob;
    for (uint32_t i = 0; i < nvocab; i++) {
        uint32_t len; memcpy(&len, p, 4); p += 4;
        g_vocab[i].len   = len;
        g_vocab[i].bytes = p;
        if (len == 1 && g_byte_id[p[0]] < 0)
            g_byte_id[p[0]] = (int32_t)i;
        p += len;
    }

    uint32_t cap = 1;
    while (cap < nmerges*2) cap <<= 1;
    g_mtab  = malloc(cap*sizeof(*g_mtab));
    g_mmask = cap - 1;
    for (uint32_t i = 0; i < cap; i++) g_mtab[i].key = UINT64_MAX;
    for (uint32_t r = 0; r < nmerges; r++) {
        uint32_t a, b, ab;
        memcpy(&a, p, 4); memcpy(&b, p + 4, 4); memcpy(&ab, p + 8, 4); p += 12;
        uint64_t k = mkey(a, b);
        uint32_t i = (uint32_t)(hash64(k) & g_mmask);
        while (g_mtab[i].key != UINT64_MAX) i = (i + 1) & g_mmask;
        g_mtab[i].key = k; g_mtab[i].rank = r; g_mtab[i].result = ab;
    }
    return 0;
}

/* BPE over one pre-token: start from single bytes, repeatedly merge the best-ranked pair. */
static int bpe_chunk(const uint8_t *s, size_t n, int *out, int max)
{
    static int   ids[4096];
    if (n == 0 || n > sizeof(ids)/sizeof(ids[0])) return 0;
    size_t cnt = 0;
    for (size_t i = 0; i < n; i++) {
        int id = g_byte_id[s[i]];
        if (id < 0) continue; /* Should not happen: byte-level vocab covers 0..255. */
        ids[cnt++] = id;
    }
    while (cnt > 1) {
        uint32_t best = UINT32_MAX; size_t at = 0; uint32_t res = 0;
        for (size_t i = 0; i + 1 < cnt; i++) {
            const struct merge_slot *m = merge_find(ids[i], ids[i+1]);
            if (m && m->rank < best) { best = m->rank; at = i; res = m->result; }
        }
        if (best == UINT32_MAX) break;
        ids[at] = (int)res;
        memmove(&ids[at+1], &ids[at+2], (cnt - at - 2)*sizeof(int));
        cnt--;
    }
    int w = 0;
    for (size_t i = 0; i < cnt && w < max; i++) out[w++] = ids[i];
    return w;
}

/* Minimal UTF-8 decode: returns the codepoint and advances *len. */
static uint32_t utf8_cp(const uint8_t *s, size_t n, int *len)
{
    if (n == 0) { *len = 0; return 0; }
    if (s[0] < 0x80)                    { *len = 1; return s[0]; }
    if ((s[0] & 0xE0) == 0xC0 && n >= 2) { *len = 2; return ((s[0]&0x1F)<<6)|(s[1]&0x3F); }
    if ((s[0] & 0xF0) == 0xE0 && n >= 3) { *len = 3;
        return ((s[0]&0x0F)<<12)|((s[1]&0x3F)<<6)|(s[2]&0x3F); }
    if ((s[0] & 0xF8) == 0xF0 && n >= 4) { *len = 4;
        return ((uint32_t)(s[0]&0x07)<<18)|((s[1]&0x3F)<<12)|((s[2]&0x3F)<<6)|(s[3]&0x3F); }
    *len = 1; return s[0];
}

/* \p{L} approximation: good enough to separate letters (incl. accented, Greek, Cyrillic,
 * CJK) from symbols/emoji, which is what the pre-tokeniser split actually depends on. */
static int cp_is_letter(uint32_t cp)
{
    if (cp < 0x80)    return isalpha((int)cp) != 0;
    if (cp < 0x2000)  return 1;   /* Latin-1/Extended, Greek, Cyrillic, Hebrew, Arabic... */
    if (cp < 0x3040)  return 0;   /* Punctuation, arrows, math, misc symbols.             */
    if (cp < 0xD800)  return 1;   /* Kana, Hangul, CJK.                                   */
    if (cp < 0x1F000) return 1;   /* CJK compat, presentation forms.                      */
    return 0;                     /* Emoji and pictographs.                               */
}
static int cp_is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static int cp_is_space(uint32_t cp)
{
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x0b || cp == 0x0c;
}

/* Follows the Qwen/GPT-2 pre-tokeniser alternation, in order:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Order matters: a symbol run absorbs one leading space, so " + " is [" +"][" "].
 * `-T` checks this against the reference encoder. */
static int tok_encode(const char *text, int *out, int max)
{
    const uint8_t *s = (const uint8_t *)text;
    size_t n = strlen(text), i = 0;
    int w = 0;
    static const char *contr[] = {"'s","'t","'re","'ve","'m","'ll","'d", NULL};

    while (i < n && w < max) {
        size_t start = i;
        int l;
        uint32_t cp = utf8_cp(s + i, n - i, &l);

        /* 1. contractions */
        int done = 0;
        for (int c = 0; contr[c]; c++) {
            size_t cl = strlen(contr[c]);
            if (i + cl <= n && strncasecmp((const char *)s + i, contr[c], cl) == 0) {
                i += cl; done = 1; break;
            }
        }

        /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
        if (!done) {
            size_t k = i; int kl; uint32_t kc = cp;
            if (!cp_is_letter(kc) && !cp_is_digit(kc) && kc != '\r' && kc != '\n') {
                k += l;
                kc = utf8_cp(s + k, n - k, &kl);
            }
            if (k < n && cp_is_letter(kc)) {
                while (k < n) {
                    uint32_t c2 = utf8_cp(s + k, n - k, &kl);
                    if (!cp_is_letter(c2)) break;
                    k += kl;
                }
                i = k; done = 1;
            }
        }
        /* 3. \p{N} : one digit at a time */
        if (!done && cp_is_digit(cp)) { i += l; done = 1; }

        /* 4. ' ?' [^\s\p{L}\p{N}]+ [\r\n]*  (symbol run may absorb ONE leading space) */
        if (!done) {
            size_t k = i; int kl; uint32_t kc = cp;
            if (kc == ' ') { k += l; kc = utf8_cp(s + k, n - k, &kl); }
            if (k < n && !cp_is_space(kc) && !cp_is_letter(kc) && !cp_is_digit(kc)) {
                while (k < n) {
                    uint32_t c2 = utf8_cp(s + k, n - k, &kl);
                    if (cp_is_space(c2) || cp_is_letter(c2) || cp_is_digit(c2)) break;
                    k += kl;
                }
                while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
                i = k; done = 1;
            }
        }
        /* 5/6/7. whitespace: \s*[\r\n]+ , then \s+(?!\S) , then \s+ */
        if (!done) {
            size_t k = i;
            while (k < n && cp_is_space(utf8_cp(s + k, n - k, &l))) k += l;
            size_t nl_end = k;
            while (nl_end > i && (s[nl_end-1] == '\r' || s[nl_end-1] == '\n')) nl_end--;
            if (k > i && nl_end < k) {
                i = k;                       /* run ends in newline(s) */
            } else if (k < n && k > i + 1) {
                i = k - 1;                   /* \s+(?!\S): last space belongs to the word */
            } else {
                i = (k > i) ? k : i + 1;
            }
        }
        if (i <= start) i = start + 1;       /* Never stall. */
        w += bpe_chunk(s + start, i - start, out + w, max - w);
    }
    return w;
}

static void tok_decode(const int *ids, int n, char *out, size_t max)
{
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= g_nvocab) continue;
        const struct tok_entry *t = &g_vocab[ids[i]];
        if (w + t->len + 1 >= max) break;
        memcpy(out + w, t->bytes, t->len);
        w += t->len;
    }
    out[w] = 0;
}

/*---------------------------------------------------------------------------------------*/
/* Console UART: serial boot + line protocol                                              */
/*---------------------------------------------------------------------------------------*/

static int   g_tty = -1;

static const char sfl_magic_req[] = "sL5DdSMmkekro\n";
static const char sfl_magic_ack[] = "z6IHG7cYDID6o\n";
#define SFL_CMD_LOAD 0x01
#define SFL_CMD_JUMP 0x02

static uint16_t crc16(const uint8_t *d, int n) /* CRC-16/XMODEM, as litex_term uses. */
{
    uint16_t crc = 0;
    for (int i = 0; i < n; i++) {
        crc ^= (uint16_t)d[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

static int tty_open(const char *path)
{
    int fd = open(path, O_RDWR | O_NOCTTY);
    if (fd < 0) { perror(path); return -1; }
    struct termios t;
    if (tcgetattr(fd, &t) == 0) {
        cfmakeraw(&t);
        t.c_cc[VMIN] = 0; t.c_cc[VTIME] = 1;      /* 100 ms read timeout. */
        tcsetattr(fd, TCSANOW, &t);
    }
    return fd;
}

static void tty_write(const void *buf, size_t n)
{
    const uint8_t *p = buf;
    while (n) {
        ssize_t r = write(g_tty, p, n);
        if (r <= 0) { if (errno == EAGAIN) continue; perror("write(tty)"); exit(1); }
        p += r; n -= r;
    }
}

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec/1e9;
}

/* Read until `needle` is seen or timeout; optionally echo to stdout. Returns captured text. */
static int tty_expect(const char *needle, double timeout, char *cap, size_t capmax, int echo)
{
    double t0 = now_s();
    size_t w = 0;
    size_t nl = strlen(needle);
    if (cap && capmax) cap[0] = 0;
    while (now_s() - t0 < timeout) {
        uint8_t c;
        ssize_t r = read(g_tty, &c, 1);
        if (r <= 0) continue;
        if (echo) { fputc(c, stdout); fflush(stdout); }
        if (cap && w + 1 < capmax) { cap[w++] = (char)c; cap[w] = 0; }
        if (w >= nl && cap && strcmp(cap + w - nl, needle) == 0)
            return 1;
    }
    return 0;
}

static int sfl_frame(uint8_t cmd, const uint8_t *payload, int len)
{
    uint8_t frame[4 + 255];
    uint8_t body[1 + 255];
    body[0] = cmd;
    memcpy(body + 1, payload, len);
    uint16_t crc = crc16(body, len + 1);
    frame[0] = (uint8_t)len;
    frame[1] = (uint8_t)(crc >> 8);
    frame[2] = (uint8_t)(crc & 0xFF);
    frame[3] = cmd;
    memcpy(frame + 4, payload, len);
    tty_write(frame, 4 + len);
    /* Wait for the BIOS ack ('K' = success). */
    double t0 = now_s();
    while (now_s() - t0 < 5.0) {
        uint8_t c;
        if (read(g_tty, &c, 1) == 1) {
            if (c == 'K') return 0;
            if (c == 'C' || c == 'E' || c == 'U') {
                fprintf(stderr, "serialboot: NAK '%c'\n", c);
                return -1;
            }
        }
    }
    fprintf(stderr, "serialboot: ack timeout\n");
    return -1;
}

static int serialboot(const char *fw)
{
    FILE *f = fopen(fw, "rb");
    if (!f) { perror(fw); return -1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *img = malloc(len);
    if (fread(img, 1, len, f) != (size_t)len) { perror("fread"); return -1; }
    fclose(f);

    printf("[boot] waiting for the BIOS serialboot prompt (reset the board if idle)...\n");
    /* The BIOS emits the magic periodically while waiting; answer it. */
    if (!tty_expect(sfl_magic_req, 30.0, NULL, 0, 0)) {
        char buf[256];
        tty_write("\nserialboot\n", 12);          /* Nudge it from the BIOS prompt. */
        if (!tty_expect(sfl_magic_req, 15.0, buf, sizeof(buf), 0)) {
            fprintf(stderr, "serialboot: no magic from the BIOS - is the console right?\n");
            return -1;
        }
    }
    tty_write(sfl_magic_ack, strlen(sfl_magic_ack));

    uint32_t addr = ROM_BOOT_ADDRESS;
    const int chunk = 64;                          /* sfl_safe_data_length */
    for (long off = 0; off < len; off += chunk) {
        int n = (int)((len - off < chunk) ? (len - off) : chunk);
        uint8_t pl[4 + 64];
        uint32_t a = addr + (uint32_t)off;
        pl[0] = a >> 24; pl[1] = a >> 16; pl[2] = a >> 8; pl[3] = a;
        memcpy(pl + 4, img + off, n);
        if (sfl_frame(SFL_CMD_LOAD, pl, 4 + n) < 0) { free(img); return -1; }
        if ((off % (chunk*256)) == 0) {
            printf("\r[boot] %ld/%ld KB", (off + n)/1024, len/1024);
            fflush(stdout);
        }
    }
    printf("\r[boot] %ld/%ld KB - jumping\n", len/1024, len/1024);
    uint8_t pl[4];
    pl[0] = addr >> 24; pl[1] = addr >> 16; pl[2] = addr >> 8; pl[3] = addr;
    sfl_frame(SFL_CMD_JUMP, pl, 4);
    free(img);
    return 0;
}

/*---------------------------------------------------------------------------------------*/
/* Weight upload (PCIe DMA -> loader -> HBM), same sequence as bonsai_load                */
/*---------------------------------------------------------------------------------------*/

struct bons_header {
    uint32_t magic, version, n_layer, n_embd, n_head, n_kv_head, head_dim, ffn_dim, vocab;
    uint32_t rope_theta;
    float    rms_eps;
    uint32_t max_ctx, n_channels, n_tensors, table_offset, payload_offset;
    uint64_t payload_beats;
} __attribute__((packed));

static void dma_write_all(int fd, const uint8_t *buf, size_t len)
{
    while (len) {
        ssize_t r = write(fd, buf, len);
        if (r < 0) { if (errno == EAGAIN) { usleep(100); continue; } perror("write"); exit(1); }
        if (r == 0) { fprintf(stderr, "DMA stalled, %zu left\n", len); exit(1); }
        buf += r; len -= r;
    }
}

static void loader_transfer(int fd, int chan, uint32_t base, const uint8_t *d, uint64_t beats)
{
    litepcie_writel(fd, CSR_LOADER_CHAN_ADDR,   chan);
    litepcie_writel(fd, CSR_LOADER_BASE_ADDR,   base);
    litepcie_writel(fd, CSR_LOADER_LENGTH_ADDR, beats);
    litepcie_writel(fd, CSR_LOADER_START_ADDR,  1);
    dma_write_all(fd, d, beats*32);
    int timeout = 10000;
    while (!(litepcie_readl(fd, CSR_LOADER_STATUS_ADDR) & 1)) {
        usleep(1000);
        if (--timeout == 0) { fprintf(stderr, "loader timeout\n"); exit(1); }
    }
}

static int upload_model(int devnum, const char *path)
{
    char dev[64];
    snprintf(dev, sizeof(dev), "/dev/bonsai%d", devnum);
    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror(dev); return -1; }

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    struct bons_header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != 0x534E4F42) {
        fprintf(stderr, "%s: not a BONS image\n", path); return -1;
    }
    if (h.n_channels != BONSAI_N_CHANNELS) {
        fprintf(stderr, "image has %u channels, gateware has %u - use the matching image\n",
                h.n_channels, BONSAI_N_CHANNELS);
        return -1;
    }
    printf("[load] %s: %u layers, %u channels, %.0f MB total\n", path, h.n_layer,
           h.n_channels, (h.payload_beats*32.0*h.n_channels + h.payload_offset)/1e6);

    litepcie_dma_set_loopback(fd, 0);
    struct litepcie_ioctl_dma_reader rd = { .enable = 1 };
    ioctl(fd, LITEPCIE_IOCTL_DMA_READER, &rd);

    uint64_t chan_bytes = h.payload_beats*32;
    uint64_t bufsz = PAD(chan_bytes > h.payload_offset ? chan_bytes : h.payload_offset);
    uint8_t *buf = malloc(bufsz);
    double t0 = now_s();

    fseek(f, 0, SEEK_SET);
    if (fread(buf, 1, h.payload_offset, f) != h.payload_offset) { perror("fread"); return -1; }
    memset(buf + h.payload_offset, 0, PAD(h.payload_offset) - h.payload_offset);
    loader_transfer(fd, 0, HDR_STAGING, buf, PAD(h.payload_offset)/32);

    for (uint32_t c = 0; c < h.n_channels; c++) {
        fseek(f, (uint64_t)h.payload_offset + (uint64_t)c*chan_bytes, SEEK_SET);
        if (fread(buf, 1, chan_bytes, f) != chan_bytes) { perror("fread"); return -1; }
        memset(buf + chan_bytes, 0, PAD(chan_bytes) - chan_bytes);
        printf("\r[load] channel %u/%u", c + 1, h.n_channels); fflush(stdout);
        loader_transfer(fd, c, BONSAI_CHAN_WEIGHT_BASE, buf, PAD(chan_bytes)/32);
    }
    rd.enable = 0;
    ioctl(fd, LITEPCIE_IOCTL_DMA_READER, &rd);
    double dt = now_s() - t0;
    double mb = (h.payload_offset + (double)h.n_channels*chan_bytes)/1e6;
    printf("\r[load] %.0f MB in %.1f s (%.2f GB/s)\n", mb, dt, mb/dt/1000);
    free(buf); fclose(f); close(fd);
    return 0;
}

/*---------------------------------------------------------------------------------------*/
/* Chat                                                                                   */
/*---------------------------------------------------------------------------------------*/

static void send_line(const char *fmt, ...)
{
    char line[1200];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line) - 2, fmt, ap);
    va_end(ap);
    line[n++] = '\n'; line[n] = 0;
    tty_write(line, n);
}

/* Firmware console line buffer is 1024 B; send the prompt in comfortably-sized chunks. */
static int send_prompt(const int *ids, int n)
{
    send_line("tokclr");
    if (!tty_expect("TOKCLR", 5.0, NULL, 0, 0)) return -1;
    char line[900];
    int i = 0;
    while (i < n) {
        int w = snprintf(line, sizeof(line), "tok");
        while (i < n && w < 850)
            w += snprintf(line + w, sizeof(line) - w, " %d", ids[i++]);
        send_line("%s", line);
        if (!tty_expect("TOK ", 10.0, NULL, 0, 0)) return -1;
        tty_expect("\n", 2.0, NULL, 0, 0);
    }
    return 0;
}

static int parse_gen(const char *buf, int *ids, int max)
{
    const char *p = strstr(buf, "GEN:");
    if (!p) return 0;
    p += 4;
    int n = 0;
    while (*p && n < max) {
        while (*p == ' ' || *p == '\r') p++;
        if (!isdigit((unsigned char)*p)) break;
        ids[n++] = atoi(p);
        while (isdigit((unsigned char)*p)) p++;
    }
    return n;
}

static int self_test(const char *tests_path)
{
    FILE *f = fopen(tests_path, "r");
    if (!f) { perror(tests_path); return 1; }
    char line[4096];
    int pass = 0, fail = 0;
    while (fgets(line, sizeof(line), f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        /* The prompt is JSON-quoted; decode the simple escapes we emit. */
        char text[2048]; int w = 0;
        for (char *p = line + 1; *p && *(p+1); p++) {
            if (*p == '"' && *(p+1) == 0) break;
            if (*p == '\\') {
                p++;
                if (*p == 'n') text[w++] = '\n';
                else if (*p == 't') text[w++] = '\t';
                else if (*p == 'u') { /* \uXXXX -> UTF-8, incl. surrogate pairs */
                    unsigned cp = (unsigned)strtoul((char[5]){p[1],p[2],p[3],p[4],0}, NULL, 16);
                    p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p[1] == '\\' && p[2] == 'u') {
                        unsigned lo = (unsigned)strtoul(
                            (char[5]){p[3],p[4],p[5],p[6],0}, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                    }
                    if (cp >= 0x10000) {
                        text[w++] = 0xF0|(cp>>18);        text[w++] = 0x80|((cp>>12)&0x3F);
                        text[w++] = 0x80|((cp>>6)&0x3F);  text[w++] = 0x80|(cp&0x3F);
                        continue;
                    }
                    if (cp < 0x80) text[w++] = (char)cp;
                    else if (cp < 0x800) { text[w++] = 0xC0|(cp>>6); text[w++] = 0x80|(cp&0x3F); }
                    else { text[w++] = 0xE0|(cp>>12); text[w++] = 0x80|((cp>>6)&0x3F);
                           text[w++] = 0x80|(cp&0x3F); }
                }
                else text[w++] = *p;
            } else text[w++] = *p;
        }
        text[w] = 0;

        int exp[512], nexp = 0;
        for (char *p = tab + 1; *p && nexp < 512; ) {
            while (*p == ' ') p++;
            if (!isdigit((unsigned char)*p)) break;
            exp[nexp++] = atoi(p);
            while (isdigit((unsigned char)*p)) p++;
        }
        int got[512];
        int ngot = tok_encode(text, got, 512);
        int ok = (ngot == nexp);
        for (int i = 0; ok && i < ngot; i++) ok = (got[i] == exp[i]);
        if (ok) { pass++; }
        else {
            fail++;
            printf("MISMATCH %-42.42s\n  want:", text);
            for (int i = 0; i < nexp; i++) printf(" %d", exp[i]);
            printf("\n  got :");
            for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
            printf("\n");
        }
    }
    fclose(f);
    printf("tokeniser self-test: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}

static void usage(void)
{
    printf("usage: bonsai_chat [options]\n"
           "  -c N       PCIe device number (default 0)\n"
           "  -t TTY     console tty (default /dev/ttyLXU0)\n"
           "  -m IMAGE   upload a BONS weight image to HBM first\n"
           "  -k FW      serial-boot this firmware binary first\n"
           "  -z FILE    tokeniser (default models/1.7b/tokenizer.bin)\n"
           "  -n N       tokens to generate per turn (default 64)\n"
           "  -T [FILE]  tokeniser self-test against reference encodings, then exit\n");
}

int main(int argc, char **argv)
{
    int devnum = 0, ntok = 64, do_test = 0;
    const char *tty  = "/dev/ttyLXU0";
    const char *img  = NULL, *fw = NULL;
    const char *tokf = "models/1.7b/tokenizer.bin";
    const char *testf = "models/1.7b/tokenizer_tests.txt";
    int opt;
    while ((opt = getopt(argc, argv, "c:t:m:k:z:n:T::h")) != -1) {
        switch (opt) {
        case 'c': devnum = atoi(optarg); break;
        case 't': tty = optarg; break;
        case 'm': img = optarg; break;
        case 'k': fw  = optarg; break;
        case 'z': tokf = optarg; break;
        case 'n': ntok = atoi(optarg); break;
        case 'T': do_test = 1; if (optarg) testf = optarg; break;
        default: usage(); return 0;
        }
    }

    if (tok_load(tokf) < 0) return 1;
    if (do_test) return self_test(testf);

    if (img && upload_model(devnum, img) < 0) return 1;

    g_tty = tty_open(tty);
    if (g_tty < 0) return 1;

    if (fw && serialboot(fw) < 0) return 1;

    /* Sync with the firmware prompt. */
    send_line("");
    if (!tty_expect("bonsai>", 10.0, NULL, 0, 0)) {
        fprintf(stderr,
            "no 'bonsai>' prompt on %s.\n"
            "  - firmware not running? re-run with -k firmware/bonsai.bin\n"
            "  - wrong console? try -t /dev/ttyLXU0 (PCIe) or use litex_term crossover\n", tty);
        return 1;
    }
    printf("[init] firmware is up; loading the model header\n");
    send_line("hdrload");
    char cap[4096];
    tty_expect("bonsai>", 30.0, cap, sizeof(cap), 0);
    if (strstr(cap, "inference ready") == NULL)
        fprintf(stderr, "[warn] hdrload did not report 'inference ready':\n%s\n", cap);

    printf("\nTriXium chat - type a prompt, Ctrl-D to quit.\n"
           "(each turn continues the conversation on-card; /reset clears it)\n\n");

    char *linebuf = NULL; size_t linecap = 0;
    int   ids[1024], out[1024];
    char  text[8192];
    int   fresh = 1;

    while (1) {
        printf("you> "); fflush(stdout);
        ssize_t ln = getline(&linebuf, &linecap, stdin);
        if (ln <= 0) break;
        if (linebuf[ln-1] == '\n') linebuf[--ln] = 0;
        if (ln == 0) continue;
        if (!strcmp(linebuf, "/reset")) { fresh = 1; printf("[context cleared]\n"); continue; }

        int n = tok_encode(linebuf, ids, 1024);
        if (n == 0) { printf("[empty after tokenising]\n"); continue; }

        double t0 = now_s();
        if (fresh) {
            if (send_prompt(ids, n) < 0) { fprintf(stderr, "prompt send failed\n"); break; }
            fresh = 0;
        } else {
            /* Continue: the card still holds the conversation, just append this turn. */
            char line[900]; int i = 0;
            while (i < n) {
                int w = snprintf(line, sizeof(line), "tok");
                while (i < n && w < 850) w += snprintf(line + w, sizeof(line) - w, " %d", ids[i++]);
                send_line("%s", line);
                if (!tty_expect("TOK ", 10.0, NULL, 0, 0)) { fprintf(stderr, "tok failed\n"); return 1; }
                tty_expect("\n", 2.0, NULL, 0, 0);
            }
        }

        send_line("genb %d", ntok);
        if (!tty_expect("GEN DONE", 600.0, cap, sizeof(cap), 0)) {
            fprintf(stderr, "\n[timeout waiting for generation]\n");
            continue;
        }
        double dt = now_s() - t0;
        int ng = parse_gen(cap, out, 1024);
        tok_decode(out, ng, text, sizeof(text));
        printf("bot> %s\n", text);
        printf("     [%d tokens, %.1f s, %.1f tok/s]\n\n", ng, dt, ng/dt);
        tty_expect("bonsai>", 5.0, NULL, 0, 0);
    }
    free(linebuf);
    close(g_tty);
    return 0;
}

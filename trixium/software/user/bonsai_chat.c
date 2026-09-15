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
#include <sys/file.h>
#include <sys/time.h>

#include "liblitepcie.h"
#include "soc.h"
#include "mem.h"
#include "bonsai_loader.h"

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

static uint8_t g_tok_sha[32];
static int     g_tok_sha_ok;

/* Chat-template tokens (Qwen3: <|im_start|> 151644, <|im_end|> 151645, <think> 151667,
 * </think> 151668), found by their literal text in the table - added tokens are stored as
 * UTF-8, and the added-token id is the highest one carrying that text. -1 = not this
 * tokenizer, and the tool falls back to raw completion. */
static int g_im_start = -1, g_im_end = -1, g_think = -1, g_think_end = -1;

static int tok_find(const char *lit)
{
    size_t n = strlen(lit);
    for (uint32_t i = g_nvocab; i-- > 0; )
        if (g_vocab[i].len == n && memcmp(g_vocab[i].bytes, lit, n) == 0)
            return (int)i;
    return -1;
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

    if (ver >= 2) {                                  /* v2: the source tokenizer.json's hash. */
        if (fread(g_tok_sha, 32, 1, f) != 1) { perror("fread"); fclose(f); return -1; }
        g_tok_sha_ok = 1;
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
    g_im_start  = tok_find("<|im_start|>");
    g_im_end    = tok_find("<|im_end|>");
    g_think     = tok_find("<think>");
    g_think_end = tok_find("</think>");
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

static int g_chat = 1;   /* 0 = raw completion (-r): the text goes to the model as typed. */
static const char *g_system = NULL;   /* -y: a system turn ahead of the first user turn. */

/* One user turn in the Qwen3 chat template, as the HF tokenizer would encode it (it splits
 * at the added tokens and BPE-encodes the runs between - same ids as this, piece by piece):
 *   [\n] <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * The empty think block is the template's "enable_thinking=False": without it the model
 * spends the whole budget inside <think>. On a continuing conversation the previous reply
 * is closed first (the card keeps it in the buffer; a reply cut by -n has no <|im_end|>). */
static int encode_turn(const char *text, int *out, int max, int first, int prev_closed)
{
    int n = 0;
    if (!g_chat || g_im_start < 0)
        return tok_encode(text, out, max);
    if (!first) {
        if (!prev_closed && n < max) out[n++] = g_im_end;
        n += tok_encode("\n", out + n, max - n);
    }
    if (first && g_system && *g_system) {         /* <|im_start|>system\n...<|im_end|>\n */
        if (n < max) out[n++] = g_im_start;
        n += tok_encode("system\n", out + n, max - n);
        n += tok_encode(g_system, out + n, max - n);
        if (n < max) out[n++] = g_im_end;
        n += tok_encode("\n", out + n, max - n);
    }
    if (n < max) out[n++] = g_im_start;
    n += tok_encode("user\n", out + n, max - n);
    n += tok_encode(text, out + n, max - n);
    if (n < max) out[n++] = g_im_end;
    n += tok_encode("\n", out + n, max - n);
    if (n < max) out[n++] = g_im_start;
    n += tok_encode("assistant\n", out + n, max - n);
    if (g_think >= 0 && g_think_end >= 0) {
        if (n < max) out[n++] = g_think;
        n += tok_encode("\n\n", out + n, max - n);
        if (n < max) out[n++] = g_think_end;
        n += tok_encode("\n\n", out + n, max - n);
    }
    return n;
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
    /* One console session at a time: the firmware reads one byte stream, and two readers
     * split its replies between them (an idle bonsai_chat left at its prompt did exactly
     * that, 2026-09-14). Advisory lock, shared by every console tool. */
    if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
        fprintf(stderr, "%s is in use by another console session (bonsai_chat, pcie_console.py, "
                        "the Python client, litex_term?) - close it first\n", path);
        close(fd);
        return -1;
    }
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
    /* Match against a private sliding window, NOT the caller's capture buffer. Keying the
     * comparison off `cap` meant every tty_expect(..., NULL, 0, ...) could never match - the
     * prompt sync is exactly that call, so attaching to a running card always burned the full
     * 120 s and reported "firmware not running" for a perfectly healthy board. It also failed
     * once a capture buffer filled, because w stops advancing there. */
    char win[128];
    size_t ww = 0;
    if (nl >= sizeof(win)) nl = sizeof(win) - 1;
    if (cap && capmax) cap[0] = 0;
    while (now_s() - t0 < timeout) {
        uint8_t c;
        ssize_t r = read(g_tty, &c, 1);
        if (r <= 0) continue;
        if (echo) { fputc(c, stdout); fflush(stdout); }
        if (cap && w + 1 < capmax) { cap[w++] = (char)c; cap[w] = 0; }
        if (ww < nl) {
            win[ww++] = (char)c;
        } else {
            memmove(win, win + 1, nl - 1);
            win[nl - 1] = (char)c;
        }
        if (ww == nl && memcmp(win, needle, nl) == 0)
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
/* Weight upload and firmware boot, through the shared loader (bonsai_loader.c)            */
/*---------------------------------------------------------------------------------------*/

static int upload_model(int devnum, const char *path)
{
    (void)devnum; /* BAR0 is mapped once in main() for every loader use. */
    printf("[load] %s\n", path);
    return loader_upload_image(path, 0) ? 0 : -1;
}

/* Fast boot: CPU reset -> BIOS prompt -> firmware into main_ram through the loader ->
 * BIOS `boot`. The serial-boot protocol moved 100 KB in 64-byte frames with an ack each
 * (23-44 s, and NAKs part-way often enough that the deploy script retries three times);
 * the loader does it in about a millisecond. Cache coherency: the BIOS memtest/memspeed
 * stream 2 MB through the 32 KB write-back D$ before the prompt, so no line of the
 * firmware's first 100 KB is left in it (dirty or clean) when the loader writes past the
 * cache, and the BIOS never executes from main_ram (no stale I$ lines). */
static int fastboot(const char *fw)
{
    printf("[boot] CPU reset, waiting for the BIOS prompt...\n");
    loader_cpu_reset();
    /* The prompt is coloured ("\e[92;1mlitex\e[0m> "), so match its uncoloured tail. It
     * comes after the 2 MB memtest/memspeed and the serial-boot timeout: 1.0 s. */
    if (!tty_expect("[0m> ", 30.0, NULL, 0, 0)) {
        tty_write("\n", 1);
        if (!tty_expect("[0m> ", 10.0, NULL, 0, 0)) {
            fprintf(stderr, "fastboot: no BIOS prompt on the console - is the tty right?\n");
            return -1;
        }
    }
    uint64_t n = loader_upload_firmware(fw);
    printf("[boot] %llu KB into main_ram - jumping\n", (unsigned long long)(n/1024));
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "boot 0x%08lx\n", (unsigned long)MAIN_RAM_BASE);
    tty_write(cmd, strlen(cmd));
    return 0;
}

/*---------------------------------------------------------------------------------------*/
/* Chat                                                                                   */
/*---------------------------------------------------------------------------------------*/

/* Console lines go out in small chunks, each acknowledged by its echo before the next.
 *
 * The firmware's UART RX ring is 128 bytes (libbase uart.c) and the ISR DROPS bytes when it
 * is full; the console loop drains it one byte per iteration and echoes each through a TX
 * ring the host driver empties on a poll timer. A long `tok` line written in one burst thus
 * loses characters as soon as the echo stalls - two ids merge into one out-of-range id, and
 * an id past the vocabulary sends the engine to fetch a row that does not exist: ENGSTALL,
 * reconfigure the FPGA. Seen with 130-character lines on 2026-09-11; the 850-character
 * lines both clients allowed were never safe. Waiting for the echo bounds the bytes in
 * flight to one chunk, well inside the ring. */
#define PACE_CHUNK 48
static void tty_write_paced(const char *buf, size_t n)
{
    while (n) {
        size_t k = n < PACE_CHUNK ? n : PACE_CHUNK;
        tty_write(buf, k);
        double t0 = now_s();
        size_t got = 0;
        while (got < k && now_s() - t0 < 3.0) {  /* the echo, byte for byte */
            uint8_t c;
            if (read(g_tty, &c, 1) == 1) got++;
        }
        buf += k; n -= k;
    }
}

static void send_line(const char *fmt, ...)
{
    char line[1200];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line) - 2, fmt, ap);
    va_end(ap);
    line[n++] = '\n'; line[n] = 0;
    tty_write_paced(line, n);
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

/* How many bytes the UTF-8 sequence starting with `c` occupies (1 for ASCII, continuation
 * bytes and anything malformed - those just pass through). */
static int utf8_len(unsigned char c)
{
    if (c < 0x80)          return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* Read the generation as it arrives and print each token the moment it lands.
 *
 * The firmware already emits one printf per token, so the ids were streaming over the UART
 * all along - the tool simply buffered to "GEN DONE" and printed the reply in one lump. The
 * only real work here is UTF-8: the vocabulary is byte-level BPE, so a single token can end
 * halfway through a multi-byte character. Bytes are held back until the character is
 * complete, otherwise the terminal shows replacement junk mid-word.
 *
 * Returns the token count, or -1 on timeout. *first_dt gets the time to the first token,
 * which is the latency a user actually feels. */
static int stream_generate(double timeout, int *ids, int max, double *first_dt)
{
    double t0 = now_s();
    char win[16];                 /* sliding window, to spot "GEN:" then "GEN DONE" */
    size_t ww = 0;
    int started = 0, n = 0;
    long acc = -1;                /* digits of the id being parsed, -1 = not in a number */
    int  prev = ':';              /* the byte before the digits: an id follows ':' or ' ' */
    unsigned char pend[8];        /* incomplete UTF-8 tail */
    size_t pendn = 0;

    *first_dt = 0;
    while (now_s() - t0 < timeout) {
        unsigned char c;
        ssize_t r = read(g_tty, &c, 1);
        if (r <= 0) continue;

        if (ww < sizeof(win) - 1) win[ww++] = (char)c;
        else { memmove(win, win + 1, sizeof(win) - 2); win[sizeof(win) - 2] = (char)c; }
        win[ww] = 0;

        if (!started) {                        /* skip the echoed command, wait for GEN: */
            if (strstr(win, "GEN:")) { started = 1; ww = 0; win[0] = 0; }
            continue;
        }
        if (strstr(win, "GEN DONE")) break;

        /* Ids are printed as " %d" after "GEN:". Digits reached any other way - inside
         * a stray message ("(14 descriptors)") - are not ids: the firmware once printed
         * such a line inside the stream and its 14 became a "/" in every reply. */
        if (isdigit(c)) {
            if (acc >= 0 || prev == ' ' || prev == ':')
                acc = (acc < 0 ? 0 : acc)*10 + (c - '0');
            prev = c;
            continue;
        }
        int delim = (c == ' ' || c == '\n' || c == '\r');
        prev = c;
        if (acc < 0) continue;                 /* delimiter with no digits pending */
        if (!delim) { acc = -1; continue; }    /* digits glued to text: not an id */

        /* One complete id: record it, then print its bytes. */
        if (n < max) ids[n] = (int)acc;
        n++;
        if (n == 1) *first_dt = now_s() - t0;
        if (acc == g_im_end || acc == g_im_start) {
            /* Template tokens are structure, not text (the firmware stops on <|im_end|>). */
        } else if (acc >= 0 && (uint32_t)acc < g_nvocab) {
            const struct tok_entry *t = &g_vocab[acc];
            for (size_t i = 0; i < t->len; i++) {
                if (pendn < sizeof(pend)) pend[pendn++] = (unsigned char)t->bytes[i];
                size_t need = utf8_len(pend[0]);
                if (pendn >= need) {           /* character complete: emit it */
                    fwrite(pend, 1, need, stdout);
                    memmove(pend, pend + need, pendn - need);
                    pendn -= need;
                }
            }
            fflush(stdout);
        }
        acc = -1;
    }
    if (pendn) { fwrite(pend, 1, pendn, stdout); fflush(stdout); }   /* malformed tail */
    return (now_s() - t0 < timeout) ? n : -1;
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
           "  -k FW      boot this firmware first (CPU reset, loader into main_ram, BIOS boot)\n"
           "  -K FW      boot it the old way (serial-boot frames over the console, 30-45 s)\n"
           "  -z FILE    tokeniser (default models/1.7b/tokenizer.bin)\n"
           "  -n N       reply cap in tokens (default 256; the reply ends at <|im_end|> anyway,\n"
           "             the cap is for runaways; the card holds 2048 tokens of conversation)\n"
           "  -r         raw completion: no chat template, the text continues as typed\n"
           "  -S T,P,R   sampling: temperature (0 = greedy), top-p, repetition penalty, e.g.\n"
           "             -S 0.7,0.9,1.1 (greedy loops on lists; 1.1 alone already breaks them)\n"
           "  -s SEED    sampling PRNG seed (reproducible runs)\n"
           "  -y TEXT    system prompt (a system turn ahead of the conversation)\n"
           "In the chat a line ending in a backslash continues on the next line.\n"
           "  -T [FILE]  tokeniser self-test against reference encodings, then exit\n"
           "  -E TEXT    print the ids one chat turn of TEXT sends (template included), exit\n"
           "In the chat: /reset clears the conversation, /sample T,P,R changes the sampling,\n"
           "/cmd <line> runs a firmware console\n"
           "command (e.g. /cmd stats, /cmd hwops), /stats and /help are shortcuts.\n");
}

int main(int argc, char **argv)
{
    int devnum = 0, ntok = 256, do_test = 0;
    const char *tty  = "/dev/ttyLXU0";
    const char *img  = NULL, *fw = NULL, *fw_serial = NULL;
    const char *tokf = "models/1.7b/tokenizer.bin";
    const char *testf = "models/1.7b/tokenizer_tests.txt";
    const char *enc_text = NULL;
    const char *sampling = NULL;
    unsigned    seed = 0;
    int opt;
    while ((opt = getopt(argc, argv, "c:t:m:k:K:U:z:n:T::E:S:s:y:rh")) != -1) {
        switch (opt) {
        case 'c': devnum = atoi(optarg); break;
        case 't': tty = optarg; break;
        case 'm': img = optarg; break;
        case 'k': fw  = optarg; break;
        case 'K': fw_serial = optarg; break;
        case 'U': /* Upload the firmware through the loader and exit (bring-up aid). */
            loader_bar_map(devnum);
            printf("%llu bytes\n", (unsigned long long)loader_upload_firmware(optarg));
            return 0;
        case 'z': tokf = optarg; break;
        case 'n': ntok = atoi(optarg); break;
        case 'r': g_chat = 0; break;
        case 'S': sampling = optarg; break;
        case 's': seed = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'y': g_system = optarg; break;
        case 'E': enc_text = optarg; break;
        case 'T': do_test = 1; if (optarg) testf = optarg; break;
        default: usage(); return 0;
        }
    }

    if (tok_load(tokf) < 0) return 1;
    if (do_test) return self_test(testf);
    if (enc_text) {
        int ids[2048];
        int n = encode_turn(enc_text, ids, 2048, 1, 1);
        for (int i = 0; i < n; i++) printf("%d%c", ids[i], i + 1 < n ? ' ' : '\n');
        return 0;
    }

    if (img || fw)
        loader_bar_map(devnum);
    if (img && upload_model(devnum, img) < 0) return 1;

    g_tty = tty_open(tty);
    if (g_tty < 0) return 1;

    if (fw && fastboot(fw) < 0) return 1;
    if (fw_serial && serialboot(fw_serial) < 0) return 1;

    /* Sync with the firmware prompt. The firmware runs its power-on self-tests (scratchpad,
     * vector-unit programs, and with a model present the engine matvec/fetch checksums and
     * a two-token generation) before it prints a prompt: 0.2 s on the 1.7B now that the
     * all-CPU reference leg is skipped there, but a firmware that still runs it takes 15 s,
     * and waiting too briefly reports "firmware not running" for a board that is merely
     * busy, eating the startup log on the way out. */
    send_line("");
    if (!tty_expect("bonsai>", 120.0, NULL, 0, 1)) { /* echo: show the self-tests as they run */
        fprintf(stderr,
            "no 'bonsai>' prompt on %s after 120 s.\n"
            "  - firmware not running? re-run with -k firmware/bonsai.bin\n"
            "  - self-test wedged? watch the console directly and look for ENGSTALL\n"
            "  - wrong console? try -t /dev/ttyLXU0 (PCIe) or use litex_term crossover\n", tty);
        return 1;
    }
    printf("[init] firmware is up; loading the model header\n");
    send_line("hdrload");
    char cap[4096];
    tty_expect("bonsai>", 30.0, cap, sizeof(cap), 0);
    if (strstr(cap, "inference ready") == NULL)
        fprintf(stderr, "[warn] hdrload did not report 'inference ready':\n%s\n", cap);
    if (g_tok_sha_ok) {
        /* The image header carries the converter's tokenizer hash; a mismatch means the
         * ids sent and the ids the model was converted with do not agree. */
        send_line("info");
        tty_expect("bonsai>", 10.0, cap, sizeof(cap), 0);
        char *p = strstr(cap, "tokenizer sha256 ");
        char hex[65];
        for (int i = 0; i < 32; i++) sprintf(hex + 2*i, "%02x", g_tok_sha[i]);
        if (!p)
            fprintf(stderr, "[warn] firmware did not report the image's tokenizer hash\n");
        else if (strncmp(p + 17, hex, 64) != 0)
            fprintf(stderr, "[warn] TOKENIZER MISMATCH: image %.16s..., %s %.16s...\n"
                            "       (convert with --tokenizer <the json this .bin came from>)\n",
                    p + 17, tokf, hex);
        else
            printf("[init] tokenizer matches the image (%.16s...)\n", hex);
    }

    if (g_chat && g_im_start < 0) {
        fprintf(stderr, "[warn] no <|im_start|> in %s - raw completion mode\n", tokf);
        g_chat = 0;
    }
    if (sampling || seed) {
        double t = 0, pp = 1, r = 1;
        if (sampling) sscanf(sampling, "%lf,%lf,%lf", &t, &pp, &r);
        send_line("sample %d %d %d %u", (int)(t*1000 + 0.5), (int)(pp*1000 + 0.5),
                  (int)(r*1000 + 0.5), seed);
        tty_expect("bonsai>", 5.0, cap, sizeof(cap), 0);
        char *nl = strchr(cap, '\n');
        printf("[init] %s", nl ? nl + 1 : cap);
    }
    printf("\nTriXium chat - type a prompt, Ctrl-D to quit.\n"
           "(%s; each turn continues the conversation on-card; /reset clears it)\n\n",
           g_chat ? "chat template, thinking off" : "raw completion, -r");

    char *linebuf = NULL; size_t linecap = 0;
    int   ids[2048], out[2048];         /* PBUF_N in firmware/main.c: the on-card buffer. */
    int   fresh = 1, prev_closed = 1;

    while (1) {
        printf("you> "); fflush(stdout);
        ssize_t ln = getline(&linebuf, &linecap, stdin);
        if (ln <= 0) break;
        if (linebuf[ln-1] == '\n') linebuf[--ln] = 0;
        /* Multi-line input: a trailing backslash continues the prompt on the next line. */
        while (ln > 0 && linebuf[ln-1] == '\\') {
            linebuf[ln-1] = '\n';
            printf("...> "); fflush(stdout);
            char *more = NULL; size_t morecap = 0;
            ssize_t lm = getline(&more, &morecap, stdin);
            if (lm <= 0) { free(more); break; }
            if (more[lm-1] == '\n') more[--lm] = 0;
            if ((size_t)(ln + lm + 1) > linecap) {
                linecap = ln + lm + 1024;
                linebuf = realloc(linebuf, linecap);
            }
            memcpy(linebuf + ln, more, lm + 1);
            ln += lm;
            free(more);
        }
        if (ln == 0) continue;
        if (!strcmp(linebuf, "/reset")) { fresh = 1; prev_closed = 1; printf("[context cleared]\n"); continue; }
        if (!strcmp(linebuf, "/help"))  { strcpy(linebuf, "/cmd help"); }
        if (!strcmp(linebuf, "/stats")) { strcpy(linebuf, "/cmd stats"); }
        if (!strncmp(linebuf, "/cmd ", 5)) {
            /* A firmware console command, its output echoed until the next prompt. */
            char cap[8192];
            send_line("%s", linebuf + 5);
            if (tty_expect("bonsai>", 30.0, cap, sizeof(cap), 0)) {
                char *nl = strchr(cap, '\n');            /* Drop the echoed command line. */
                fputs(nl ? nl + 1 : cap, stdout);
            } else
                fprintf(stderr, "[no prompt back after 30 s]\n");
            continue;
        }
        if (!strncmp(linebuf, "/sample", 7)) {           /* /sample T,P,R  or  /sample T P R */
            double t = 0, pp = 1, r = 1;
            if (sscanf(linebuf + 7, "%lf%*[ ,]%lf%*[ ,]%lf", &t, &pp, &r) < 1) {
                printf("[usage: /sample temp,top_p,rep  e.g. /sample 0.7,0.9,1.1]\n"); continue;
            }
            send_line("sample %d %d %d 0", (int)(t*1000 + 0.5), (int)(pp*1000 + 0.5), (int)(r*1000 + 0.5));
            if (tty_expect("bonsai>", 5.0, cap, sizeof(cap), 0)) {
                char *nl = strchr(cap, '\n');
                fputs(nl ? nl + 1 : cap, stdout);
            }
            continue;
        }
        if (linebuf[0] == '/') { printf("[unknown: /reset /sample /cmd <line> /stats /help]\n"); continue; }

        int n = encode_turn(linebuf, ids, 2048, fresh, prev_closed);
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

        if (g_chat) send_line("genb %d %d", ntok, g_im_end);   /* stop at <|im_end|> */
        else        send_line("genb %d", ntok);
        printf("bot> ");
        fflush(stdout);
        double first_dt = 0;
        int ng = stream_generate(600.0, out, 2048, &first_dt);
        if (ng < 0) {
            fprintf(stderr, "\n[timeout waiting for generation]\n");
            continue;
        }
        double dt = now_s() - t0;
        prev_closed = (ng > 0 && ng <= 2048 && out[ng - 1] == g_im_end);
        /* The rate is the decode rate: tokens after the first over the time after it. The
         * first token's latency is this turn's prefill (the card keeps the conversation's
         * KV cache across turns since 2026-09-11; before, it re-ran the whole history and
         * that latency grew with the chat), so folding it in would misreport the decode. */
        double rate = (ng > 1 && dt > first_dt) ? (ng - 1)/(dt - first_dt) : 0.0;
        printf("\n     [%d tokens in %.1f s: first after %.2f s, then %.1f tok/s%s]\n\n",
               ng, dt, first_dt, rate, prev_closed || !g_chat ? "" : ", cut by -n");
        tty_expect("bonsai>", 5.0, NULL, 0, 0);
    }
    free(linebuf);
    close(g_tty);
    return 0;
}

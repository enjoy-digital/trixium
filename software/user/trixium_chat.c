/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium chat: configure the accelerator, then talk to the model
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 *   1. (-m) Upload a BONS weight image through the BAR0 loader (same path as trixium_load).
 *   2. (-k) Boot the firmware: CPU reset, loader copy into main_ram, BIOS `boot` (-K: the
 *           LiteX serial-boot protocol on the console UART).
 *   3.      `hdrload` so the firmware picks up the staged BONS header/table.
 *   4.      REPL: text -> BPE ids -> `tok`/`genb` -> ids back -> text.
 *
 * The model runs entirely on the card; this tool only tokenizes, drives the console and
 * detokenizes. The tokenizer data comes from `trixium tokenizer` (model/tokenizer.py: raw-byte
 * vocab + merge ranks), so no JSON or Unicode tables are needed here.
 *
 *   trixium_chat -T                                   # Tokenizer self-test, no hardware.
 *   trixium_chat -m models/bons/bonsai-1p7b-n8.img \
 *                -k firmware/trixium.bin              # Full bring-up, then chat.
 *   trixium_chat                                      # Attach to an already-configured card.
 *   trixium_chat -t /dev/ttyUSB2 -z models/2b/tokenizer.bin   # A board on a serial console.
 *
 * Built against a gateware build without the LitePCIe driver (the Efinix Ti375 kit), it is a
 * console-only tool (TRIXIUM_UART_ONLY): -m/-k/-U/-C refuse, the weights come from
 * `trixium deploy`, -K serial-boots a firmware.
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
#include <sys/ioctl.h>
#include <sys/time.h>

#ifdef TRIXIUM_UART_ONLY
#define MAIN_RAM_BASE 0x40000000 /* The BIOS serial-boot target (every TriXium SoC). */
#else
#include "soc.h"
#include "mem.h"
#include "csr.h"
#include "liblitepcie.h"
#endif
#include "trixium_loader.h"
#include "trixium_tok.h"

/* Parameters */
/*------------*/

/* Firmware load/jump address: the BIOS serialboot target is main_ram. ROM_BOOT_ADDRESS only
 * exists in the simulation SoC. */
#ifndef ROM_BOOT_ADDRESS
#define ROM_BOOT_ADDRESS MAIN_RAM_BASE
#endif

/* Variables */
/*-----------*/

/* Terminal style: the diagrams' palette (teal TriXium, gold memory, muted grey), off when stdout
 * is not a terminal or NO_COLOR is set. */
static int g_color;
static const char *ui(const char *seq) { return g_color ? seq : ""; }
#define UI_TEAL  ui("\033[38;2;53;194;174m")
#define UI_GOLD  ui("\033[38;2;216;178;94m")
#define UI_DIM   ui("\033[38;2;140;158;152m")
#define UI_BOLD  ui("\033[1m")
#define UI_RESET ui("\033[0m")

static int  g_devnum;
static const char *g_tty_name = "";
static char g_model[48] = "Bonsai"; /* From the firmware's `info` (layers x embd). */
static char g_short[16] = "bonsai";
static int  g_layers;

/* -r: 0 = raw completion, the text goes to the model as typed. */
static int g_chat = 1;

/* -y: a system turn ahead of the first user turn. */
static const char *g_system = NULL;

/* -W: seconds to wait for the firmware prompt (the XCU1525 firmware calibrates three DDR4
 * controllers before it prints one: minutes). */
static double g_prompt_wait = 120.0;

/* -R: the BIOS is already at its prompt (a host-side DDR training ran after its calibration). */
static int g_no_cpu_reset;

/* Tokenizer: software/lib/trixium_tok.c, the firmware's encoder too */
/*-------------------------------------------------------------------*/

static struct ttok g_tok;
/* Chat-template tokens (Qwen3: <|im_start|> 151644, <|im_end|> 151645, <think> 151667,
 * </think> 151668), -1 when absent: the tool falls back to raw completion. */
static int g_im_start = -1, g_im_end = -1;

static int tok_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *btok = malloc(len > 0 ? len : 1);
    if (len <= 0 || fread(btok, 1, len, f) != (size_t)len) { perror(path); fclose(f); return -1; }
    fclose(f);
    size_t need = ttok_arena_size(btok, len);
    void *arena = need ? malloc(need) : NULL;
    if (!arena || ttok_init(&g_tok, btok, len, arena) < 0) {
        fprintf(stderr, "%s: not a BTOK tokenizer (run trixium tokenizer)\n", path);
        return -1;
    }
    g_im_start = g_tok.im_start;
    g_im_end   = g_tok.im_end;
    return 0;
}

static int tok_encode(const char *text, int *out, int max)
{
    return ttok_encode(&g_tok, (const uint8_t *)text, strlen(text), out, max);
}

/* One user turn in the Qwen3 chat template (ttok_turn); the text alone in raw mode (-r). On a
 * continuing conversation the previous reply is closed first (the card keeps it in the buffer;
 * a reply cut by -n has no <|im_end|>). */
static int encode_turn(const char *text, int *out, int max, int first, int prev_closed)
{
    if (!g_chat)
        return tok_encode(text, out, max);
    return ttok_turn(&g_tok, text, g_system, first, prev_closed, out, max);
}

/* Console UART */
/*--------------*/

static int g_tty = -1;

static const char sfl_magic_req[] = "sL5DdSMmkekro\n";
static const char sfl_magic_ack[] = "z6IHG7cYDID6o\n";
#define SFL_CMD_LOAD 0x01
#define SFL_CMD_JUMP 0x02

/* CRC-16/XMODEM, as litex_term uses. */
static uint16_t crc16(const uint8_t *d, int n)
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
     * split its replies between them. Advisory lock, shared by every console tool. */
    if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
        fprintf(stderr, "%s is in use by another console session (trixium_chat, trixium console, "
                        "the Python client, litex_term?) - close it first\n", path);
        close(fd);
        return -1;
    }
    struct termios t;
    if (tcgetattr(fd, &t) == 0) {
        cfmakeraw(&t);
        cfsetspeed(&t, B115200);                  /* A serial console (LiteUART ignores it). */
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
    /* Match against a private sliding window, not the caller's capture buffer: that one may be
     * NULL (the prompt sync) or full (w stops advancing). */
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

/* Serial boot */
/*-------------*/

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

/* Card state for trixium_smi */
/*----------------------------*/

static void chat_state(int devnum, const char *state, long tokens, double tok_s, double ttft)
{
    loader_state_write(devnum, "chat",
        "\"tool\": \"trixium_chat\", \"model\": \"%s\", \"state\": \"%s\", \"tokens\": %ld, "
        "\"tok_s\": %.3f, \"ttft\": %.3f, \"time\": %ld",
        g_model, state, tokens, tok_s, ttft, (long)time(NULL));
}

#ifdef TRIXIUM_UART_ONLY
/* No BAR0 loader on a console-only board. */
static void uart_only(void)
{
    fprintf(stderr, "this trixium_chat is a console-only build (no PCIe): upload the weights with "
                    "`trixium deploy`, boot a firmware with -K\n");
    exit(1);
}
void     loader_bar_map(int devnum)               { (void)devnum; uart_only(); }
void     loader_cpu_reset(void)                   { uart_only(); }
void     loader_clear_staging(void)               { uart_only(); }
uint64_t loader_upload_image(const char *p, int v) { (void)p; (void)v; uart_only(); return 0; }
uint64_t loader_upload_firmware(const char *p)    { (void)p; uart_only(); return 0; }
#endif

/* Weight upload and fast boot, through the shared loader (trixium_loader.c) */
/*----------------------------------------------------------------------------*/

/* BAR0 is mapped once in main() for every loader use. */
static int upload_model(const char *path)
{
    printf("[load] %s\n", path);
    return loader_upload_image(path, 0) ? 0 : -1;
}

/* Fast boot: CPU reset -> BIOS prompt -> firmware into main_ram through the loader -> BIOS
 * `boot`. The serial-boot protocol moves 100 KB in 64-byte acked frames (23-44 s, with
 * occasional NAKs); the loader does it in about a millisecond.
 *
 * Cache coherency: the BIOS memtest/memspeed stream 2 MB through the 32 KB write-back D$
 * before the prompt, so no line of the firmware's first 100 KB is left in it (dirty or clean)
 * when the loader writes past the cache, and the BIOS never executes from main_ram (no stale
 * I$ lines). */
static int fastboot(const char *fw)
{
    if (g_no_cpu_reset)
        printf("[boot] no CPU reset (-R), expecting the BIOS prompt...\n");
    else {
        printf("[boot] CPU reset, waiting for the BIOS prompt...\n");
        loader_cpu_reset();
    }
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

/* Chat protocol */
/*---------------*/

/* Console lines go out in small chunks, each acknowledged by its echo before the next.
 *
 * The firmware's UART RX ring is 128 bytes (libbase uart.c) and the ISR DROPS bytes when it
 * is full; the console loop drains it one byte per iteration and echoes each through a TX
 * ring the host driver empties on a poll timer. A long `tok` line written in one burst thus
 * loses characters as soon as the echo stalls - two ids merge into one out-of-range id, and
 * an id past the vocabulary sends the engine to fetch a row that does not exist: ENGSTALL,
 * reconfigure the FPGA. Waiting for the echo bounds the bytes in flight to one chunk, well
 * inside the ring. */
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

/* Reply text with word wrap: whole words go to the terminal, a word that would cross the right
 * edge starts a new line (indented under the reply). No wrap when stdout is not a terminal. */
static int  g_wrap_width, g_wrap_col, g_wrap_indent;
static char g_word[1024];
static int  g_wlen, g_wcols;

static void wrap_start(int indent)
{
    struct winsize ws;
    g_wrap_width  = (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) ? ws.ws_col : 0;
    g_wrap_col    = indent;
    g_wrap_indent = indent;
    g_wlen = g_wcols = 0;
}

static void wrap_flush(void)
{
    if (!g_wlen)
        return;
    if (g_wrap_width && g_wrap_col > g_wrap_indent && g_wrap_col + g_wcols > g_wrap_width - 1) {
        printf("\n%*s", g_wrap_indent, "");
        g_wrap_col = g_wrap_indent;
    }
    fwrite(g_word, 1, g_wlen, stdout);
    g_wrap_col += g_wcols;
    g_wlen = g_wcols = 0;
}

/* One complete UTF-8 character of the reply. */
static void wrap_char(const unsigned char *c, size_t n)
{
    if (n == 1 && (c[0] == ' ' || c[0] == '\n')) {
        wrap_flush();
        if (c[0] == '\n') {
            printf("\n%*s", g_wrap_indent, "");
            g_wrap_col = g_wrap_indent;
        } else if (g_wrap_col > g_wrap_indent) { /* No space at the start of a line. */
            putchar(' ');
            g_wrap_col++;
        }
        return;
    }
    if (g_wlen + (int)n >= (int)sizeof(g_word))
        wrap_flush();
    memcpy(g_word + g_wlen, c, n);
    g_wlen  += (int)n;
    g_wcols += 1;
}

/* Read the generation as it arrives and print each token the moment it lands.
 *
 * The firmware emits one printf per token. The only real work here is UTF-8: the vocabulary
 * is byte-level BPE, so a single token can end halfway through a multi-byte character. Bytes
 * are held back until the character is complete, otherwise the terminal shows replacement
 * junk mid-word.
 *
 * Returns the token count, or -1 on timeout. *first_dt gets the time to the first token,
 * which is the latency a user actually feels. */
static int stream_generate(double timeout, int *ids, int max, double *first_dt, long tokens_before)
{
    double last_state = 0;        /* trixium_smi's live view: the reply's rate so far. */
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

        /* Ids are printed as " %d" after "GEN:". Digits reached any other way, inside a
         * stray firmware message ("(14 descriptors)"), are not ids. */
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
        if (n > 1 && now_s() - last_state > 0.25) {
            double el = now_s() - t0 - *first_dt;
            chat_state(g_devnum, "generating", tokens_before + n, el > 0 ? (n - 1)/el : 0.0,
                       *first_dt);
            last_state = now_s();
        }
        if (acc == g_im_end || acc == g_im_start) {
            /* Template tokens are structure, not text (the firmware stops on <|im_end|>). */
        } else if (acc >= 0 && (uint32_t)acc < g_tok.nvocab) {
            uint32_t len;
            const uint8_t *bytes = ttok_piece(&g_tok, (int)acc, &len);
            for (size_t i = 0; i < len; i++) {
                if (pendn < sizeof(pend)) pend[pendn++] = (unsigned char)bytes[i];
                size_t need = utf8_len(pend[0]);
                if (pendn >= need) {           /* character complete: emit it */
                    wrap_char(pend, need);
                    memmove(pend, pend + need, pendn - need);
                    pendn -= need;
                }
            }
            fflush(stdout);
        }
        acc = -1;
    }
    if (pendn) wrap_char(pend, pendn);                               /* malformed tail */
    wrap_flush();
    fflush(stdout);
    return (now_s() - t0 < timeout) ? n : -1;
}

/* Tokenizer self-test */
/*---------------------*/

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
    printf("tokenizer self-test: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}

/* Help */
/*------*/

/* Header */
/*--------*/

/* The model's name from `info` ("BONS vN: L layers, embd E, ..."): the Bonsai family's shapes. */
static void model_name(const char *info)
{
    const char *p = strstr(info, "BONS v");
    unsigned v, layers = 0, embd = 0;
    if (!p || sscanf(p, "BONS v%u: %u layers, embd %u", &v, &layers, &embd) != 3)
        return;
    g_layers = (int)layers;
    const char *name = NULL, *s = NULL;
    if (layers == 28 && embd == 2048) { name = "Ternary-Bonsai 1.7B"; s = "1.7B"; }
    if (layers == 36 && embd == 4096) { name = "Ternary-Bonsai 8B";   s = "8B";   }
    if (embd == 5120)                 { name = "Bonsai 27B";          s = "27B";  }
    if (name) {
        snprintf(g_model, sizeof(g_model), "%s", name);
        snprintf(g_short, sizeof(g_short), "%s", s);
    } else {
        snprintf(g_model, sizeof(g_model), "Bonsai %ux%u", layers, embd);
        snprintf(g_short, sizeof(g_short), "bonsai");
    }
}

/* The card's SoC identifier ("TriXium on <board> ..."), read over /dev/trixiumN (no root). */
static void card_name(int devnum, char *out, size_t n)
{
#ifdef TRIXIUM_UART_ONLY
    (void)devnum;
    snprintf(out, n, "%s", g_tty_name);
#else
    char dev[64];
    snprintf(dev, sizeof(dev), "/dev/trixium%d", devnum);
    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        snprintf(out, n, "%s", g_tty_name); /* No PCIe card: a console-only board. */
        return;
    }
    snprintf(out, n, "%s", dev);
    char id[256];
    for (int i = 0; i < 255; i++)
        id[i] = (char)litepcie_readl(fd, CSR_IDENTIFIER_MEM_BASE + 4*i);
    id[255] = 0;
    close(fd);
    const char *on = strstr(id, " on ");   /* "TriXium on YPCB-00338-1P1 2026-..." */
    if (on) {
        char board[64];
        if (sscanf(on + 4, "%63s", board) == 1)
            snprintf(out, n, "%s · %s", board, dev);
    }
#endif
}

/* Visible width of a UTF-8 string (box drawing): colour escapes take no column. */
static int vis_len(const char *s)
{
    int w = 0;
    for (; *s; s++) {
        if (*s == '\033') {
            while (*s && *s != 'm') s++;
            if (!*s) break;
            continue;
        }
        if (((unsigned char)*s & 0xC0) != 0x80) w++;
    }
    return w;
}

static void box_line(int width, const char *fmt, ...)
{
    char text[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    printf("%s│%s %s", UI_TEAL, UI_RESET, text);
    for (int i = vis_len(text); i < width - 3; i++) putchar(' ');
    printf("%s│%s\n", UI_TEAL, UI_RESET);
}

static void box_rule(int width, const char *left, const char *title, const char *right)
{
    printf("%s%s─", UI_TEAL, left);
    int w = 2;
    if (title) { printf(" %s%s%s%s ", UI_BOLD, title, UI_RESET, UI_TEAL); w += vis_len(title) + 2; }
    for (; w < width - 1; w++) printf("─");
    printf("%s%s\n", right, UI_RESET);
}

static void header(int devnum, int ntok, const char *sampling)
{
    const int W = 72;
    char card[160];
    card_name(devnum, card, sizeof(card));
    putchar('\n');
    box_rule(W, "╭", "TriXium chat", "╮");
    box_line(W, "%s%s%s  ·  %s", UI_BOLD, g_model, UI_RESET, card);
    box_line(W, "%s%d layers · 2048-token context · %s · replies up to %d tokens%s", UI_DIM,
             g_layers, sampling ? "sampled" : "greedy", ntok, UI_RESET);
    box_line(W, "%sthe model runs on the card, this host only tokenizes%s%s", UI_DIM,
             g_chat ? " · thinking off" : " · raw completion", UI_RESET);
    box_rule(W, "╰", NULL, "╯");
    printf("%s  /reset clears the conversation · /stats card profile · Ctrl-D quits%s\n\n",
           UI_DIM, UI_RESET);
}

static void reply_stats(int ng, double dt, double first, double rate, int ctx, int cut)
{
    char bar[64] = "";
    int cells = 8, fill = (ctx*cells + 2047)/2048;
    if (fill > cells) fill = cells;
    for (int i = 0; i < cells; i++) strcat(bar, i < fill ? "█" : "░");
    (void)dt;
    /* Fits the 72-column box: the header already gives the 2048-token context. */
    printf("\n%s  %d tokens%s · first %.2f s · %s%.1f tok/s%s%s · context %d %s%s%s\n\n",
           UI_DIM, ng, cut ? ", capped" : "", first, UI_TEAL, rate, UI_RESET, UI_DIM, ctx,
           UI_TEAL, bar, UI_RESET);
}

static void usage(FILE *f)
{
    fprintf(f,"TriXium chat\n"
           "usage: trixium_chat [options]\n"
           "\n"
           "options:\n"
           "-h             Help.\n"
           "-c device_num  Select the device (default = 0).\n"
           "-t tty         Console tty (default = /dev/ttyLXU0).\n"
           "-z file        Tokenizer (default = models/1.7b/tokenizer.bin).\n"
           "\n"
           "bring-up:\n"
           "-m image       Upload a BONS weight image first.\n"
           "-k firmware    Boot this firmware first (CPU reset, loader into main_ram, boot).\n"
           "-K firmware    Boot it over serial-boot frames on the console instead (30-45 s).\n"
           "-R             With -k: no CPU reset, the BIOS is already at its prompt (a host-side\n"
           "               DDR training ran after its calibration; software/host/rdlevel.py).\n"
           "-W seconds     Firmware prompt timeout (default = 120; the XCU1525 firmware\n"
           "               calibrates three DDR4 controllers first).\n"
           "-C             Clear the staged BONS header and exit (before a boot on a new\n"
           "               memory layout).\n"
           "-U firmware    Upload the firmware into main_ram and exit.\n"
           "\n"
           "generation:\n"
           "-n tokens      Reply cap (default = 256; a reply ends at <|im_end|> anyway, the cap\n"
           "               is for runaways; the card holds 2048 tokens of conversation).\n"
           "-r             Raw completion: no chat template, the text continues as typed.\n"
           "-S T,P,R       Sampling: temperature (0 = greedy), top-p, repetition penalty,\n"
           "               e.g. -S 0.7,0.9,1.1 (1.1 alone already breaks greedy list loops).\n"
           "-s seed        Sampling PRNG seed (reproducible runs).\n"
           "-y text        System prompt (a system turn ahead of the conversation).\n"
           "\n"
           "tokenizer checks:\n"
           "-T[file]       Tokenizer self-test against reference encodings, then exit.\n"
           "-E text        Print the ids of one chat turn of text (template included), then exit.\n"
           "\n"
           "chat commands:\n"
           "/reset         Clear the conversation.\n"
           "/sample T,P,R  Change the sampling.\n"
           "/cmd line      Run a firmware console command (e.g. /cmd stats, /cmd hwops).\n"
           "/stats, /help  Shortcuts for /cmd stats, /cmd help.\n"
           "A line ending in a backslash continues on the next line.\n");
}

/* Main */
/*------*/

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
    while ((opt = getopt(argc, argv, "c:t:m:k:K:U:z:n:T::E:S:s:y:W:rRCh")) != -1) {
        switch (opt) {
        case 'c': devnum = atoi(optarg); break;
        case 't': tty = optarg; break;
        case 'm': img = optarg; break;
        case 'k': fw  = optarg; break;
        case 'K': fw_serial = optarg; break;
        case 'U': /* Upload the firmware through the loader and exit. */
            loader_bar_map(devnum);
            printf("%llu bytes\n", (unsigned long long)loader_upload_firmware(optarg));
            return 0;
        case 'C': /* Clear the staged BONS header and exit: a firmware that finds one runs its
                   * self-test on the weights it describes, and after a bitstream whose
                   * memory layout moved those are stale (the self-test would wedge). */
            loader_bar_map(devnum);
            loader_clear_staging();
            puts("staged header cleared");
            return 0;
        case 'z': tokf = optarg; break;
        case 'n': ntok = atoi(optarg); break;
        case 'r': g_chat = 0; break;
        case 'R': g_no_cpu_reset = 1; break;
        case 'W': g_prompt_wait = atof(optarg); break;
        case 'S': sampling = optarg; break;
        case 's': seed = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'y': g_system = optarg; break;
        case 'E': enc_text = optarg; break;
        case 'T': do_test = 1; if (optarg) testf = optarg; break;
        case 'h': usage(stdout); return 0;
        default:  usage(stderr); return 1;
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
    if (img && upload_model(img) < 0) return 1;

    g_tty = tty_open(tty);
    if (g_tty < 0) return 1;
    g_tty_name = tty;

    if (fw && fastboot(fw) < 0) return 1;
    if (fw_serial && serialboot(fw_serial) < 0) return 1;

    /* Sync with the firmware prompt. The firmware runs its power-on self-tests (scratchpad,
     * vector-unit programs, and with a model present the engine matvec/fetch checksums and
     * a two-token generation) before it prints a prompt - minutes on the XCU1525, which
     * calibrates its DDR4 controllers first - and waiting too briefly reports "firmware not
     * running" for a board that is merely busy, eating the startup log on the way out. */
    send_line("");
    /* Echo: show the self-tests as they run. */
    if (!tty_expect("trixium>", g_prompt_wait, NULL, 0, 1)) {
        fprintf(stderr,
            "no 'trixium>' prompt on %s after %.0f s (-W to wait longer).\n"
            "  - firmware not running? re-run with -k firmware/trixium.bin\n"
            "  - self-test wedged? watch the console directly and look for ENGSTALL\n"
            "  - wrong console? try -t /dev/ttyLXU0 (PCIe) or use litex_term crossover\n",
            tty, g_prompt_wait);
        return 1;
    }
    g_color  = isatty(STDOUT_FILENO) && !getenv("NO_COLOR");
    g_devnum = devnum;
    /* "\r\e[K": over the firmware prompt the sync just echoed. */
    printf("%s%s[init] firmware is up; loading the model header%s\n", g_color ? "\r\033[K" : "",
           UI_DIM, UI_RESET);
    send_line("hdrload");
    char cap[4096];
    tty_expect("trixium>", 30.0, cap, sizeof(cap), 0);
    if (strstr(cap, "inference ready") == NULL)
        fprintf(stderr, "[warn] hdrload did not report 'inference ready':\n%s\n", cap);
    send_line("info");
    tty_expect("trixium>", 10.0, cap, sizeof(cap), 0);
    model_name(cap);
    if (g_tok.sha_ok) {
        /* The image header carries the converter's tokenizer hash; a mismatch means the
         * ids sent and the ids the model was converted with do not agree. */
        char *p = strstr(cap, "tokenizer sha256 ");
        char hex[65];
        for (int i = 0; i < 32; i++) sprintf(hex + 2*i, "%02x", g_tok.sha[i]);
        if (!p)
            fprintf(stderr, "[warn] firmware did not report the image's tokenizer hash\n");
        else if (strncmp(p + 17, hex, 64) != 0)
            fprintf(stderr, "[warn] TOKENIZER MISMATCH: image %.16s..., %s %.16s...\n"
                            "       (convert with --tokenizer <the json this .bin came from>)\n",
                    p + 17, tokf, hex);
        else
            printf("%s[init] tokenizer matches the image (%.16s...)%s\n", UI_DIM, hex, UI_RESET);
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
        tty_expect("trixium>", 5.0, cap, sizeof(cap), 0);
        char *nl = strchr(cap, '\n');
        printf("%s[init] %s%s", UI_DIM, nl ? nl + 1 : cap, UI_RESET);
    }
    header(devnum, ntok, sampling);

    long tokens_total = 0;
    int  ctx = 0;                       /* Tokens of the conversation on the card. */
    chat_state(devnum, "idle", 0, 0.0, 0.0);

    char *linebuf = NULL; size_t linecap = 0;
    int   ids[2048], out[2048];         /* PBUF_N in firmware/main.c: the on-card buffer. */
    int   fresh = 1, prev_closed = 1;

    while (1) {
        printf("%s%syou ›%s ", UI_BOLD, UI_TEAL, UI_RESET); fflush(stdout);
        ssize_t ln = getline(&linebuf, &linecap, stdin);
        if (ln <= 0) break;
        if (linebuf[ln-1] == '\n') linebuf[--ln] = 0;
        /* Multi-line input: a trailing backslash continues the prompt on the next line. */
        while (ln > 0 && linebuf[ln-1] == '\\') {
            linebuf[ln-1] = '\n';
            printf("%s    ›%s ", UI_TEAL, UI_RESET); fflush(stdout);
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
        if (!strcmp(linebuf, "/reset")) {
            fresh = 1;
            prev_closed = 1;
            ctx = 0;
            printf("%s  conversation cleared%s\n\n", UI_DIM, UI_RESET);
            continue;
        }
        if (!strcmp(linebuf, "/help"))  { strcpy(linebuf, "/cmd help"); }
        if (!strcmp(linebuf, "/stats")) { strcpy(linebuf, "/cmd stats"); }
        if (!strncmp(linebuf, "/cmd ", 5)) {
            /* A firmware console command, its output echoed until the next prompt. */
            char cap[8192];
            send_line("%s", linebuf + 5);
            if (tty_expect("trixium>", 30.0, cap, sizeof(cap), 0)) {
                char *nl = strchr(cap, '\n');            /* Drop the echoed command line. */
                fputs(nl ? nl + 1 : cap, stdout);
            } else
                fprintf(stderr, "[no prompt back after 30 s]\n");
            continue;
        }
        if (!strncmp(linebuf, "/sample", 7)) {           /* /sample T,P,R  or  /sample T P R */
            double t = 0, pp = 1, r = 1;
            if (sscanf(linebuf + 7, "%lf%*[ ,]%lf%*[ ,]%lf", &t, &pp, &r) < 1) {
                printf("[usage: /sample temp,top_p,rep  e.g. /sample 0.7,0.9,1.1]\n");
                continue;
            }
            send_line("sample %d %d %d 0", (int)(t*1000 + 0.5), (int)(pp*1000 + 0.5),
                      (int)(r*1000 + 0.5));
            if (tty_expect("trixium>", 5.0, cap, sizeof(cap), 0)) {
                char *nl = strchr(cap, '\n');
                fputs(nl ? nl + 1 : cap, stdout);
            }
            continue;
        }
        if (linebuf[0] == '/') {
            printf("[unknown: /reset /sample /cmd <line> /stats /help]\n");
            continue;
        }

        int n = encode_turn(linebuf, ids, 2048, fresh, prev_closed);
        if (n == 0) { printf("[empty after tokenizing]\n"); continue; }

        double t0 = now_s();
        if (fresh)
            ctx = 0;
        ctx += n;
        if (fresh) {
            if (send_prompt(ids, n) < 0) { fprintf(stderr, "prompt send failed\n"); break; }
            fresh = 0;
        } else {
            /* Continue: the card still holds the conversation, just append this turn. */
            char line[900]; int i = 0;
            while (i < n) {
                int w = snprintf(line, sizeof(line), "tok");
                while (i < n && w < 850)
                    w += snprintf(line + w, sizeof(line) - w, " %d", ids[i++]);
                send_line("%s", line);
                if (!tty_expect("TOK ", 10.0, NULL, 0, 0)) {
                    fprintf(stderr, "tok failed\n");
                    return 1;
                }
                tty_expect("\n", 2.0, NULL, 0, 0);
            }
        }

        chat_state(devnum, "generating", tokens_total, 0.0, 0.0);
        if (g_chat) send_line("genb %d %d", ntok, g_im_end);   /* stop at <|im_end|> */
        else        send_line("genb %d", ntok);
        printf("%s%s%s ›%s ", UI_BOLD, UI_GOLD, g_short, UI_RESET);
        fflush(stdout);
        wrap_start(vis_len(g_short) + 3);
        double first_dt = 0;
        int ng = stream_generate(600.0, out, 2048, &first_dt, tokens_total);
        if (ng < 0) {
            fprintf(stderr, "\n[timeout waiting for generation]\n");
            continue;
        }
        double dt = now_s() - t0;
        prev_closed = (ng > 0 && ng <= 2048 && out[ng - 1] == g_im_end);
        /* The rate is the decode rate: tokens after the first over the time after it. The
         * first token's latency is this turn's prefill (the card keeps the conversation's
         * KV cache across turns), so folding it in would misreport the decode. */
        double rate = (ng > 1 && dt > first_dt) ? (ng - 1)/(dt - first_dt) : 0.0;
        tokens_total += ng;
        ctx += ng;
        chat_state(devnum, "idle", tokens_total, rate, first_dt);
        reply_stats(ng, dt, first_dt, rate, ctx, !(prev_closed || !g_chat));
        tty_expect("trixium>", 5.0, NULL, 0, 0);
    }
    free(linebuf);
    close(g_tty);
    return 0;
}

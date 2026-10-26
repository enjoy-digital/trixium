/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium firmware main: console shell, self-tests and inference entry points
 *
 * At boot, runs the self-test sequence (deterministic checksums that the simulation harness
 * verifies), then drops to the shell.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include <irq.h>
#include <libbase/uart.h>
#include <libbase/console.h>
#include <libbase/crc.h>
#include <system.h>

#include "bons.h"
#include "ddr.h"
#include "engine.h"
#include "model.h"
#include "infer.h"
#include "ops.h"
#include "perf.h"
#include "bench.h"
#include "seq.h"
#include "resident.h"
#include "hybrid.h"
#include "vec.h"
#include "progs.h"
#include "static_memory.h"
#include "tokenizer.h"
#include "chat.h"
#include "web.h"
#include "sdload.h"

/* Checks */
/*--------*/

#if defined(TRIXIUM_N_MEMCTRL) && TRIXIUM_N_MEMCTRL > 1
// ddr.c reaches controller k's registers as controller 0's shifted by k*STRIDE: the target
// pins the (sdram, ddrphy) CSR pairs to consecutive pages, and this is where that is checked.
// (The PHY model of the Verilator sim has no CSRs, hence the ddrphy guards. One controller - the
// small boards' topology, every channel on main_ram's controller - has no page to check.)
_Static_assert(CSR_SDRAM1_BASE - CSR_SDRAM_BASE == TRIXIUM_DDR_CSR_STRIDE, "sdram1 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY1_BASE - CSR_DDRPHY_BASE == TRIXIUM_DDR_CSR_STRIDE, "ddrphy1 CSR page");
#endif
#if TRIXIUM_N_MEMCTRL > 2
_Static_assert(CSR_SDRAM2_BASE - CSR_SDRAM_BASE == 2*TRIXIUM_DDR_CSR_STRIDE, "sdram2 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY2_BASE - CSR_DDRPHY_BASE == 2*TRIXIUM_DDR_CSR_STRIDE, "ddrphy2 CSR page");
#endif
#endif
#if TRIXIUM_N_MEMCTRL > 3
_Static_assert(CSR_SDRAM3_BASE - CSR_SDRAM_BASE == 3*TRIXIUM_DDR_CSR_STRIDE, "sdram3 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY3_BASE - CSR_DDRPHY_BASE == 3*TRIXIUM_DDR_CSR_STRIDE, "ddrphy3 CSR page");
#endif
#endif
#endif

/* Console */
/*---------*/

static char *readstr(void)
{
    char c[2];
    char *const s = (char *)TRIXIUM_CONSOLE_LINE_ADDR;
    static int ptr = 0;

    if (readchar_nonblock()) {
        c[0] = getchar();
        c[1] = 0;
        switch (c[0]) {
        case 0x7f:
        case 0x08:
            if (ptr > 0) {
                ptr--;
                fputs("\x08 \x08", stdout);
            }
            break;
        case '\r':
        case '\n':
            s[ptr] = 0x00;
            fputs("\n", stdout);
            ptr = 0;
            return s;
        default:
            if (ptr >= TRIXIUM_CONSOLE_LINE_SIZE - 1)
                break;
            fputs(c, stdout);
            s[ptr] = c[0];
            ptr++;
            break;
        }
    }
    return NULL;
}

static char *get_token(char **str)
{
    char *c, *d;
    c = (char *)strchr(*str, ' ');
    if (c == NULL) {
        d = *str;
        *str = *str + strlen(*str);
        return d;
    }
    *c = 0;
    d = *str;
    *str = c + 1;
    return d;
}

/* Helpers */
/*---------*/

static uint32_t fnv1a(const volatile uint8_t *data, uint32_t len)
{
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

// Deterministic x8 pattern shared with the python harness: byte i = (i*137 + 11) & 0xFF.
static void fill_x_pattern(uint32_t xsrc, uint32_t n)
{
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + xsrc);
    for (uint32_t i = 0; i < n/4; i++) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++)
            w |= (uint32_t)(((4*i + b)*137 + 11) & 0xFF) << (8*b);
        sp[i] = w;
    }
}

/* Commands */
/*----------*/

static void emit_token(int tok);
static int  parse_int(const char *s);

static void help(void)
{
    puts("TriXium firmware");
    puts("Inference:");
    puts("  info              - model info (BONS header)");
    puts("  hdrload           - copy the BONS header+table staged by trixium_load to CPU memory");
    puts("  tokclr/tok/genb   - chunked prompt buffer + generate [stop ids] (host chat tools)");
    puts("  gen n id id ...   - generate n tokens from prompt ids");
#ifdef CSR_TRIXIUM_ENGINE_CSR_NR_ADDR
    puts("  gen2 n ids | ids  - two conversations decoded together (multi-row matvec); gen2 dual|seq, lmhead on|off, trace a|b");
#endif
    puts("  tokload           - load the tokenizer staged with the image (trixium load --tokenizer)");
    puts("  chat [n]          - chat in text on this console, replies up to n tokens (default 512)");
#ifdef CSR_SD_BASE
    puts("  sdinit [mhz]      - SD card / eMMC init (the widest bus that reads; 50 MHz high speed when taken)");
    puts("  sdload c b s n    - n 512-byte blocks from sector s into channel c's window at b");
    puts("  sdboot [slot]     - a model on the SD card / eMMC: weights (CRC-checked), hdrload, tokload");
    puts("  sdsave [slot] [label] - the uploaded model (and tokenizer) to a slot");
    puts("  sdls              - the slots on the card; sddefault slot - the one booted at power-up");
    puts("  sdwtest s         - write a pattern block at sector s and read it back");
#endif
    puts("  sample T P R [S]  - temperature/top-p/repeat-penalty x1000 (0 1000 1000 = greedy), seed");
    puts("  bench [n]         - profiled generation benchmark (default 4 tokens)");
    puts("Hardware vs CPU reference (default: the hardware path where the bitstream has it):");
    puts("  hwops [op cpu|hw] - show/set per-op CPU/HW dispatch (op name or 'all')");
    puts("  seq resident silu dnres dnstream attnstream hwargmax [on|off]");
    puts("                    - unit or list path vs its CPU reference (prints the state)");
    puts("  embres unit|cpu   - embedding rows without the int16 table: transform on unit 3 or the CPU");
    puts("  lmchunk rows      - LM head list chunk cap in rows (0 = automatic)");
    puts("  attnchunk n       - cap the heads per attention list");
    puts("  kvring on|off     - past the KV window: 4 sink positions + a ring of the latest (on), or stop");
    puts("Diagnostics:");
    puts("  stats [reset]     - per-op cycle profile");
    puts("  fprof on|off|print, tprof arm [n]|dump, engdiag on|off");
    puts("                    - FFN windows, token timeline, per-op engine stall snapshots");
    puts("  dbg [n]           - checksum the next n forwards' intermediates (vs the Python mirror)");
    puts("  selftest          - engine self-tests (matvec + fetch checksums)");
    puts("  sptest            - scratchpad bus read/write test");
    puts("  peek c addr       - dump one weight-memory beat of channel c via the engine (hex addr)");
    puts("  imgcheck [i [n]]  - read every tensor back through the engine, check its table CRC");
    puts("  mvtest role [layer] - matvec one tensor with the test pattern, print checksum");
    puts("  mvbw role [layer] [reps] - engine streaming efficiency on one tensor");
    puts("  ddrstat          - DFI command cycles per controller and the engine ports' waits since the previous ddrstat");
    puts("  mvdiff role [layer] [reps] - repeat one matvec, report the rows that change");
    puts("  fetchtest role [layer] [reps] - re-read a tensor's weights, compare chunk checksums");
#ifdef TRIXIUM_N_MEMCTRL
    puts("  ddrinit [k]       - re-init DDR controller k (default: 1..N-1)");
    puts("  ddrtest [c]       - engine store/fetch round trip through channel c's window (all)");
#endif
    puts("  engreset          - soft reset of the engine cluster (automatic after a stall)");
    puts("  wedge             - stall the engine on purpose, recover, rerun the self-test");
    puts("  help              - this message");
}

static void hwops_cmd(char *args)
{
    char *op  = get_token(&args);
    char *mode = get_token(&args);
    if (*op == 0) {
        hwops_print();
        return;
    }
    int hw = (strcmp(mode, "hw") == 0);
    if (!hw && strcmp(mode, "cpu") != 0) {
        puts("usage: hwops [<op>|all cpu|hw]");
        return;
    }
    if (strcmp(op, "all") == 0) {
        hwops_set_all(hw);
    } else {
        int found = -1;
        for (int i = 0; i < OP_COUNT; i++)
            if (strcmp(op, op_name(i)) == 0)
                found = i;
        if (found < 0) {
            puts("unknown op");
            return;
        }
        hwops_set(found, hw);
    }
    hwops_print();
}

/* Bench */
/*-------*/

static struct bench_samples bench_samples;

static void bench_emit_token(int tok)
{
    bench_token_ready(&bench_samples, perf_now64());
    emit_token(tok);
    bench_token_emitted(&bench_samples, perf_now64());
}

static void bench_field(const char *name, uint64_t value)
{
    printf(" %s=", name);
    perf_print_u64(value);
}

static void bench_cmd(char *args)
{
    char *tok = get_token(&args);
    int n = (*tok) ? parse_int(tok) : 4;
    if (n <= 0) { puts("BENCH ERR: token count must be positive"); return; }
    // Optional prompt length P (default 3: ids 1, 2, 3): ids 1..P (the prefill measurement).
    int prompt[64];
    tok = get_token(&args);
    int np = (*tok) ? parse_int(tok) : 3;
    if (np < 1 || np > 64) { puts("BENCH ERR: prompt length 1..64"); return; }
    for (int i = 0; i < np; i++) prompt[i] = 1 + i;
    perf_reset();
    uint32_t f0 = infer_forwards;
    uint64_t t0 = perf_now64();
    memset(&bench_samples, 0, sizeof(bench_samples));
    bench_samples.start = t0;
    printf("GEN:");
    int count = infer_generate(prompt, np, n, bench_emit_token);
    uint64_t dt = perf_now64() - t0;
    uint32_t fwd = infer_forwards - f0;
    puts("");
    perf_print();
    printf("BENCH: %d tokens, ", count);
    perf_print_u64(dt);
    printf(" cycles total");
    if (count > 0) {
        printf(", ");
        perf_print_u64(dt/count);
        printf(" cycles/token (%lu.%03lu tok/s @%luMHz)",
               (unsigned long)(((uint64_t)CONFIG_CLOCK_FREQUENCY*count)/dt),
               (unsigned long)((((uint64_t)CONFIG_CLOCK_FREQUENCY*count*1000)/dt) % 1000),
               (unsigned long)(CONFIG_CLOCK_FREQUENCY/1000000));
    }
    // Per FORWARD: the figure that stays comparable across prompt and token counts
    // (bench 64 = 3 prompt + 63 = 66 forwards: the last token needs no forward).
    if (fwd > 0) {
        printf(", %lu forwards, ", (unsigned long)fwd);
        perf_print_u64(dt/fwd);
        printf(" cycles/fwd");
    }
    puts("");
    const struct infer_prefill_profile *p = infer_last_prefill();
    printf("BENCH_PHASE v=1");
    bench_field("valid", count > 0 && p->valid);
    bench_field("clock_hz", CONFIG_CLOCK_FREQUENCY);
    bench_field("requested_tokens", n);
    bench_field("generated_tokens", bench_samples.count);
    bench_field("prompt_tokens", p->tokens);
    bench_field("prefill_forwards", p->forwards);
    bench_field("reset_cycles", p->reset_cycles);
    bench_field("prefill_cycles", p->prefill_cycles);
    bench_field("ttft_cycles", bench_samples.count ? bench_samples.first - t0 : 0);
    bench_field("decode_tokens", bench_samples.count ? bench_samples.count - 1 : 0);
    bench_field("decode_cycles", bench_decode_compute(&bench_samples));
    bench_field("decode_wall_cycles", bench_decode_wall(&bench_samples));
    bench_field("emit_cycles", bench_samples.emit_cycles);
    bench_field("total_cycles", dt);
    bench_field("context_capacity", infer_context_capacity());
    bench_field("context_used", infer_kv_resident());
    puts("");
}

/* Prompt Buffer */
/*---------------*/

static int gen_emit_to_pbuf;
// Chunked prompt buffer (tok/genb), in the main_ram static window (static_memory.h), not
// .bss: TRIXIUM_HDR_ADDR + 0xA0000, clear of the entry cache (+0x80000).
// Sized to the KV budget (the image's max_ctx, 2048 for the 1.7B): a
// conversation can use the whole context.
#define PBUF_N 2048
static int *const pbuf = (int *)(TRIXIUM_HDR_ADDR + 0xA0000); // [PBUF_N]
static int pbuf_n;
static int      pbuf_kv;      // pbuf[0..pbuf_kv) are in the KV cache (from the last genb)...
static uint32_t pbuf_serial;  // ...as long as no other generation ran since.
static int      pbuf_base;    // pbuf[0] is this position (chat dropped older, resident tokens).

// Room for `need` more tokens in pbuf. With the KV ring (a hybrid model) the oldest tokens, already
// in the ring and the DeltaNet state, go (pbuf keeps a quarter of its size of the latest) and the
// conversation goes on from the same position; 0 when there is room.
static int pbuf_make_room(int need)
{
    int resident = pbuf_serial == infer_gen_serial() ? pbuf_kv : 0;
    if (pbuf_n + need <= PBUF_N)
        return 0;
    if (!infer_context_unbounded() || resident <= PBUF_N/4 || resident > pbuf_n)
        return -1;
    int drop = resident - PBUF_N/4;
    memmove(pbuf, pbuf + drop, (size_t)(pbuf_n - drop)*sizeof(pbuf[0]));
    pbuf_n -= drop; pbuf_kv -= drop; pbuf_base += drop;
    return pbuf_n + need <= PBUF_N ? 0 : -1;
}

static void emit_token(int tok)
{
    printf(" %d", tok);
    if (gen_emit_to_pbuf && pbuf_n < PBUF_N)
        pbuf[pbuf_n++] = tok;
}

/* Header Load */
/*-------------*/

// Copy the BONS header+table from the HBM staging area (engine channel 0, uploaded there by
// trixium_load) into the CPU-visible header location, then re-init inference.
#define HDR_STAGING (TRIXIUM_CHAN_WEIGHT_BASE - 0x100000)
// The staged copy (channel 0, uploaded with every image by trixium_load) is the truth about
// what is in HBM; the main_ram copy is derived from it. Trusting main_ram after a CPU reset
// that followed the upload of a DIFFERENT model would run the boot demo with the previous
// model's tables (its KV addresses point into the new model's weights, and the demo tokens
// overwrite them), so the boot refreshes main_ram from the staged header when they differ.
// The table as the converter wrote it (header table_crc32, zlib's CRC over the entries).
static int hdr_table_ok(const struct bons_header *h)
{
    return crc32((const unsigned char *)h + h->table_offset,
                 h->n_tensors*sizeof(struct bons_entry)) == h->table_crc32;
}

static int hdrload_staged(int quiet)
{
    // First fetch just the header to learn the table size.
    engine_fetch(0, HDR_STAGING, 128, 0xC000);
    volatile struct bons_header *sh =
        (volatile struct bons_header *)(SCRATCHPAD_BASE + 0xC000);
    if (sh->magic != BONS_MAGIC) {
        if (!quiet)
            printf("hdrload: no BONS header at staging (magic %08lx)\n",
                   (unsigned long)sh->magic);
        // Staging cleared by the host (trixium_chat -C writes this marker before a board
        // boot): a header left in main_ram by an earlier session describes weights that are
        // not there, and the boot demo would stream them from controllers not yet trained.
        // The simulation stages nothing and keeps the header it preloaded into main_ram.
        struct bons_header *mh = (struct bons_header *)TRIXIUM_HDR_ADDR;
        if (sh->magic == BONS_STAGING_CLEARED && mh->magic == BONS_MAGIC) {
            mh->magic = 0;
            model_cache_reset();
        }
        return -1;
    }
    if (sh->version < BONS_VERSION_MIN || sh->version > BONS_VERSION) {
        if (!quiet)
            printf("hdrload: unsupported BONS version %lu\n", (unsigned long)sh->version);
        return -1; // Reject before interpreting layout-specific offsets/copy sizes.
    }
    const struct bons_header *mh = model_header();
    if (quiet && mh->magic == BONS_MAGIC && mh->version == sh->version
        && mh->n_tensors == sh->n_tensors && mh->payload_beats == sh->payload_beats
        && mh->n_layer == sh->n_layer && mh->table_crc32 == sh->table_crc32
        && hdr_table_ok(mh))
        return 0;                              // main_ram already holds this image's header
    uint32_t bytes = sh->payload_offset;
    uint8_t *dst   = (uint8_t *)TRIXIUM_HDR_ADDR;
    uint32_t off   = 0;
    while (off < bytes) {
        uint32_t chunk = bytes - off > 0x10000 ? 0x10000 : bytes - off; // 64 KB chunks.
        engine_fetch(0, HDR_STAGING + off, chunk/32, 0xC000);
        volatile uint32_t *src = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
        for (uint32_t i = 0; i < chunk/4; i++)
            ((uint32_t *)(dst + off))[i] = src[i];
        off += chunk;
    }
#ifdef HAS_CLEAN_CPU_DCACHE_RANGE
    clean_cpu_dcache_range(dst, bytes);   // main_ram holds the table now, not just the cache.
#endif
    model_cache_reset(); // Header changed: rebuild the (role, layer) lookup cache.
    if (!hdr_table_ok((const struct bons_header *)dst)) {
        // The staged copy itself (the upload) or the copy into main_ram is damaged: a table
        // entry read wrong is a NULL tensor in the forward, an engine op on garbage.
        printf("hdrload: table CRC mismatch (%08lx, header says %08lx) - re-upload the image\n",
               (unsigned long)crc32((const unsigned char *)dst + ((struct bons_header *)dst)->table_offset,
                                    ((struct bons_header *)dst)->n_tensors*sizeof(struct bons_entry)),
               (unsigned long)((struct bons_header *)dst)->table_crc32);
        ((struct bons_header *)dst)->magic = 0;
        return -1;
    }
    printf("hdrload: %lu KB copied%s\n", (unsigned long)(bytes/1024),
           quiet ? " (staged header differed from main_ram's)" : "");
    return 1;
}

static void hdrload(void)
{
    if (hdrload_staged(0) < 0)
        return;
    model_info();
    if (infer_init() == 0)
        puts("hdrload: inference ready");
}

/* Diagnostics */
/*-------------*/

// zlib's CRC-32, chainable (the converter chains each tensor's channels): a nibble table, so
// no 1 KB table in the SRAM.
static uint32_t crc32_update(uint32_t crc, const volatile uint32_t *p, uint32_t n)
{
    static const uint32_t t[16] = {
        0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC, 0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
        0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C, 0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C,
    };
    crc = ~crc;
    for (uint32_t i = 0; i < n/4; i++) {   // Little-endian words: bytes in address order.
        uint32_t w = p[i];                 // One uncached scratchpad read per 4 bytes.
        for (int k = 0; k < 4; k++, w >>= 8) {
            crc ^= w & 0xff;
            crc = (crc >> 4) ^ t[crc & 15];
            crc = (crc >> 4) ^ t[crc & 15];
        }
    }
    return ~crc;
}

// Every tensor's payload read back through the engine and checked against its table CRC
// (the converter's: each channel's share, channel-major): proves the upload landed whole, and
// names the tensor when it did not.
//   imgcheck [first [count [chunk]]]   (chunk: beats per engine fetch, default 2048)
static void imgcheck_cmd(char *str)
{
    const struct bons_header *h = model_header();
    if (h->magic != BONS_MAGIC) {
        puts("imgcheck: no model (hdrload)");
        return;
    }
    char *f_s = get_token(&str);
    char *n_s = get_token(&str);
    char *k_s = get_token(&str);
    uint32_t chunk = *k_s ? (uint32_t)strtoul(k_s, NULL, 0) : 2048;
    if (chunk == 0 || chunk > 2048) chunk = 2048;
    uint32_t first = *f_s ? (uint32_t)strtoul(f_s, NULL, 0) : 0;
    uint32_t count = *n_s ? (uint32_t)strtoul(n_s, NULL, 0) : h->n_tensors;
    const struct bons_entry *t = (const struct bons_entry *)(TRIXIUM_HDR_ADDR + h->table_offset);
    uint32_t bad = 0, done = 0;
    uint64_t bytes = 0;
    for (uint32_t i = first; i < h->n_tensors && i < first + count; i++, done++) {
        const struct bons_entry *e = &t[i];
        uint32_t crc = 0;
        for (uint32_t c = 0; c < h->n_channels; c++) {
            for (uint64_t b = 0; b < e->chan_len_beats; ) {
                uint32_t n = e->chan_len_beats - b > chunk ? chunk : (uint32_t)(e->chan_len_beats - b);
                engine_fetch((uint8_t)c, TRIXIUM_CHAN_WEIGHT_BASE + (uint32_t)((e->chan_base_beats + b)*32), n, 0xC000);
                crc = crc32_update(crc, (const volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000), n*32);
                b += n;
            }
        }
        bytes += (uint64_t)e->chan_len_beats*32*h->n_channels;
        if (crc != e->crc32) {
            bad++;
            printf("imgcheck: [%lu] %.24s crc %08lx, table %08lx\n", (unsigned long)i, e->name,
                   (unsigned long)crc, (unsigned long)e->crc32);
        }
    }
    printf("imgcheck: %lu tensors, %lu MB, %lu bad\n", (unsigned long)done,
           (unsigned long)(bytes >> 20), (unsigned long)bad);
}

// Fetch one beat from (channel, HBM byte address) through the engine and dump it. Ground
// truth for "did the host's upload land where the engine looks for it?" - the two sides
// agree on paper (both byte addresses, both indexing hbm.axi[1+c]) but only this says so.
//   peek <chan> <hex-addr>
static void peek_cmd(char *str)
{
    char *c_s = get_token(&str);
    char *a_s = get_token(&str);
    if (*c_s == 0 || *a_s == 0) {
        puts("usage: peek <chan> <hex-addr>");
        return;
    }
    int      chan = (int)strtoul(c_s, NULL, 0);
    uint32_t addr = (uint32_t)strtoul(a_s, NULL, 16);
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
    for (int i = 0; i < 8; i++)
        sp[i] = 0xdeadbeef;                 // so "unchanged" is distinguishable from "read 0"
    engine_fetch(chan, addr, 1, 0xC000);
    printf("peek ch%d %08lx:", chan, (unsigned long)addr);
    for (int i = 0; i < 8; i++)
        printf(" %08lx", (unsigned long)sp[i]);
    printf("\n");
}

// Matvec one tensor (by role and layer) with the deterministic x pattern and report its
// checksum. selftest only covers blk.0.attn_q; the real models have other shapes (cols 6144
// for ffn_down, rows 1024 for attn_k/v, the tied LM head) that the tiny sim model lacks.
//   mvtest <role> <layer>      roles: 4 attn_q  5 attn_k  6 attn_v  7 attn_out
//                                    11 ffn_gate 12 ffn_up 13 ffn_down  2 lm_head
static void mvtest_cmd(char *str)
{
    char *r = get_token(&str);
    char *l = get_token(&str);
    if (*r == 0) { puts("usage: mvtest <role> <layer>"); return; }
    if (model_header()->magic != BONS_MAGIC) { puts("mvtest: no model"); return; }
    int role  = (int)strtoul(r, NULL, 0);
    int layer = (*l) ? (int)strtoul(l, NULL, 0) : 0;
    const struct bons_entry *e = model_entry(role, layer);
    if (!e) { printf("mvtest: no tensor role %d layer %d\n", role, layer); return; }
    if (e->kind != KIND_TERNARY && e->kind != KIND_TERNARY5 && e->kind != KIND_Q8) {
        printf("mvtest: role %d layer %d is kind %u, not a matvec\n", role, layer, e->kind);
        return;
    }
    uint32_t xsrc = 0, dst = 0x8000;
    fill_x_pattern(xsrc, e->cols);
    model_matvec(e, xsrc, dst);
    uint32_t csum = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + dst), e->rows*4);
    printf("MV role=%d layer=%d rows=%lu cols=%lu bias=%d rshift=%d CSUM=%08lx\n",
           role, layer, (unsigned long)e->rows, (unsigned long)e->cols,
           e->bias, e->rshift, (unsigned long)csum);
}

// Engine streaming efficiency on one real tensor: `mvbw <role> <layer> [reps]` times reps
// matvecs through the legacy command path and compares the cycles with the tensor's beats
// per channel (the floor at one 32-byte beat per channel per clock).
static void mvbw_cmd(char *str)
{
    char *r = get_token(&str), *l = get_token(&str), *n = get_token(&str);
    if (*r == 0) { puts("usage: mvbw <role> <layer> [reps]"); return; }
    if (model_header()->magic != BONS_MAGIC) { puts("mvbw: no model"); return; }
    const struct bons_entry *e = model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0);
    int reps = *n ? (int)strtoul(n, NULL, 0) : 4;
    if (!e || (e->kind != KIND_TERNARY && e->kind != KIND_TERNARY5 && e->kind != KIND_Q8) || e->rows*4 > 0x30000) {
        puts("mvbw: no such matvec tensor (or its result does not fit the scratchpad)");
        return;
    }
    fill_x_pattern(0, e->cols);
    model_matvec(e, 0, 0x8000);                    // warm: descriptors, first-touch
    uint64_t t0 = perf_now64();
    for (int i = 0; i < reps; i++)
        model_matvec(e, 0, 0x8000);
    uint64_t dt = (perf_now64() - t0)/(uint64_t)(reps > 0 ? reps : 1);
    uint64_t beats = e->chan_len_beats;
    // The pipe's beats: one 128-trit (or 32-byte Q8) beat per clock per channel, scale beats
    // included; a trit5 row's memory beats are 26/32 of its code beats (bons_row_beats), so the
    // pipe at 100% reads 0.81-0.85 memory beats per clock: compare the two rates.
    uint64_t pipe = (e->flags & FLAG_KSEG) ? 0 : (uint64_t)(e->rows/TRIXIUM_N_CHANNELS)*
                    (bons_scale_beats(e->cols, e->group) + bons_code_beats(e->cols, e->kind));
    printf("MVBW role=%u layer=%u rows=%lu cols=%lu kind=%u: %lu cycles/matvec, %lu beats/channel, "
           "%lu.%02lu beats/channel/clock (%lu%% of 1.0); pipe beats %lu, %lu.%02lu per clock\n", e->role, e->layer,
           (unsigned long)e->rows, (unsigned long)e->cols, e->kind, (unsigned long)dt,
           (unsigned long)beats, (unsigned long)(beats/dt), (unsigned long)(beats*100/dt % 100),
           (unsigned long)(beats*100/dt), (unsigned long)pipe, (unsigned long)(pipe/dt),
           (unsigned long)(pipe*100/dt % 100));
}

// Weight read consistency: `fetchtest <role> <layer> [reps]` fetches a tensor's beats on
// each channel (one channel at a time, 32 KB chunks) and checksums every chunk; later passes
// must match the first. Catches read errors ddrtest's small window does not reach.
static void fetchtest_cmd(char *str)
{
    char *r = get_token(&str), *l = get_token(&str), *n = get_token(&str);
    if (*r == 0) { puts("usage: fetchtest <role> <layer> [reps]"); return; }
    if (model_header()->magic != BONS_MAGIC) { puts("fetchtest: no model"); return; }
    const struct bons_entry *e = model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0);
    if (!e) { puts("fetchtest: no such tensor"); return; }
    int reps = *n ? (int)strtoul(n, NULL, 0) : 4;
    enum { CHUNK = 1024, MAXCK = 256, DST = 0x10000, REF = 0x20000 }; // REF: 8 KB per 8 channels
    volatile uint32_t (*ref)[MAXCK] = (volatile uint32_t (*)[MAXCK])(SCRATCHPAD_BASE + REF);
    uint32_t beats = e->chan_len_beats, nck = (beats + CHUNK - 1)/CHUNK;
    if (nck > MAXCK) nck = MAXCK;
    uint32_t base = TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32;
    int bad_total = 0;
    for (int rep = 0; rep <= reps; rep++) {
        int bad = 0;
        for (int c = 0; c < TRIXIUM_N_CHANNELS; c++)
            for (uint32_t k = 0; k < nck; k++) {
                uint32_t nb = (k + 1)*CHUNK > beats ? beats - k*CHUNK : CHUNK;
                engine_fetch(c, base + k*CHUNK*32, nb, DST);
                uint32_t h = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + DST), nb*32);
                if (rep == 0) ref[c][k] = h;
                else if (h != ref[c][k]) {
                    if (bad < 4) printf("  pass %d chan %d chunk %lu differs\n", rep, c, (unsigned long)k);
                    bad++;
                }
            }
        bad_total += bad;
    }
    printf("FETCHTEST role=%u layer=%u %lu beats/channel x %d passes: %d chunk mismatches\n",
           e->role, e->layer, (unsigned long)beats, reps, bad_total);
}

#if defined(CSR_SDRAM_STATS_SNAP_ADDR) && defined(TRIXIUM_DDR_CSR_STRIDE) && defined(TRIXIUM_N_MEMCTRL)
#define STATS(k, reg) (CSR_SDRAM_STATS_##reg##_ADDR + (unsigned long)(k)*TRIXIUM_DDR_CSR_STRIDE)
#define PCT10(x, n) (unsigned long)((uint64_t)(x)*1000u/(n)) / 10, (unsigned long)((uint64_t)(x)*1000u/(n)) % 10
// `ddrstat`: the DFI command cycles of every controller (gateware DFIStats) and the engine
// ports' waits (EnginePortStats) since the previous call - each call snaps and restarts the
// counters, so call it before and after a workload (bench, mvbw) and read the second.
static void ddrstat_cmd(void)
{
    for (int k = 0; k < TRIXIUM_N_MEMCTRL; k++) {
        csr_write_simple(1, STATS(k, SNAP));
        uint32_t cyc = csr_read_simple(STATS(k, CYCLES));
        if (!cyc) cyc = 1;
        uint32_t rd = csr_read_simple(STATS(k, READS)), wr = csr_read_simple(STATS(k, WRITES));
        uint32_t act = csr_read_simple(STATS(k, ACTS)), pre = csr_read_simple(STATS(k, PRES));
        uint32_t ref = csr_read_simple(STATS(k, REFS)), idle = csr_read_simple(STATS(k, IDLE));
        printf("DDRSTAT ctrl %d: %lu ddr cycles: read %lu.%lu%% write %lu.%lu%% act %lu.%lu%% pre %lu.%lu%% "
               "ref %lu.%lu%% no command %lu.%lu%% (counts %lu %lu %lu %lu %lu %lu)\n", k, (unsigned long)cyc,
               PCT10(rd, cyc), PCT10(wr, cyc), PCT10(act, cyc), PCT10(pre, cyc), PCT10(ref, cyc), PCT10(idle, cyc),
               (unsigned long)rd, (unsigned long)wr, (unsigned long)act, (unsigned long)pre, (unsigned long)ref,
               (unsigned long)idle);
    }
#ifdef CSR_ENGSTAT_BASE
    engstat_snap_write(1);
    uint32_t cyc = engstat_cycles_read();
    if (!cyc) cyc = 1;
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        engstat_sel_write(c);
        uint32_t beats = engstat_beats_read(), ar = engstat_ar_wait_read(), r = engstat_r_wait_read();
        uint32_t wb = engstat_wbeats_read();
        printf("  chan %d: %lu read beats (%lu.%02lu per clock), requests waited %lu.%lu%% of the cycles (%lu), "
               "data waited on the engine %lu.%lu%% (%lu), %lu write beats; %lu cycles\n", c, (unsigned long)beats,
               (unsigned long)((uint64_t)beats*100u/cyc/100), (unsigned long)((uint64_t)beats*100u/cyc % 100),
               PCT10(ar, cyc), (unsigned long)ar, PCT10(r, cyc), (unsigned long)r, (unsigned long)wb, (unsigned long)cyc);
    }
#endif
#ifdef CSR_WIDESTAT_BASE
    // The 512-bit side of each engine port, before the clock crossing (the sim): words the
    // crossing delivered, requests that waited on it, words that waited on the merge.
    widestat_snap_write(1);
    uint32_t wcyc = widestat_cycles_read();
    if (!wcyc) wcyc = 1;
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        widestat_sel_write(c);
        uint32_t words = widestat_beats_read(), ar = widestat_ar_wait_read(), r = widestat_r_wait_read();
        printf("  wide %d: %lu words (%lu.%02lu per clock), requests waited %lu.%lu%% (%lu), "
               "words waited on the merge %lu.%lu%% (%lu); %lu cycles\n", c, (unsigned long)words,
               (unsigned long)((uint64_t)words*100u/wcyc/100), (unsigned long)((uint64_t)words*100u/wcyc % 100),
               PCT10(ar, wcyc), (unsigned long)ar, PCT10(r, wcyc), (unsigned long)r, (unsigned long)wcyc);
    }
#endif
#ifdef CSR_NATSTAT_BASE
    // Each engine channel's native port, in the controller's clock: commands issued, cycles a
    // command waited on the controller, cycles with nothing to issue, read words returned.
    natstat_snap_write(1);
    uint32_t ncyc = natstat_cycles_read();
    if (!ncyc) ncyc = 1;
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        natstat_sel_write(c);
        uint32_t iss = natstat_issued_read(), wt = natstat_waited_read(), st = natstat_starved_read();
        uint32_t rw = natstat_rwords_read();
        printf("  native %d: %lu read commands (%lu.%lu%% of the cycles), waited on the controller %lu.%lu%% (%lu), "
               "nothing to issue %lu.%lu%% (%lu), %lu words back; %lu ddr cycles\n", c, (unsigned long)iss,
               PCT10(iss, ncyc), PCT10(wt, ncyc), (unsigned long)wt, PCT10(st, ncyc), (unsigned long)st,
               (unsigned long)rw, (unsigned long)ncyc);
    }
#endif
#ifdef CSR_RDSTAT_BASE
    // Each channel reader's output to its pipe: beats, cycles a beat waited on the pipe,
    // cycles the reader had no beat (its FIFO empty).
    rdstat_snap_write(1);
    uint32_t rcyc = rdstat_cycles_read();
    if (!rcyc) rcyc = 1;
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        rdstat_sel_write(c);
        uint32_t b = rdstat_beats_read(), st = rdstat_stalled_read(), sv = rdstat_starved_read();
        printf("  reader %d: %lu beats (%lu.%lu%% of the cycles), waited on the pipe %lu.%lu%% (%lu), "
               "no beat %lu.%lu%% (%lu); %lu cycles\n", c, (unsigned long)b, PCT10(b, rcyc),
               PCT10(st, rcyc), (unsigned long)st, PCT10(sv, rcyc), (unsigned long)sv, (unsigned long)rcyc);
    }
#endif
}
#endif


#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
#define BIST(k, reg) (CSR_SDRAM_##reg##_ADDR + (unsigned long)(k)*SDRAM_CSR_STRIDE)
// Controller k's BIST checker re-reading the 32 MB that `ddrstress` wrote (background load and
// an error count): bist_kick restarts it when done, bist_report prints the total.
static const uint32_t bist_len = 32u << 20, bist_base = (2u << 30) - (32u << 20) - (64u << 20);
static uint32_t bist_errs, bist_runs;
static void bist_kick(int k)
{
    if (bist_runs && !csr_read_simple(BIST(k, CHECKER_DONE))) return;
    if (bist_runs) bist_errs += csr_read_simple(BIST(k, CHECKER_ERRORS));
    csr_write_simple(1, BIST(k, CHECKER_RESET)); csr_write_simple(0, BIST(k, CHECKER_RESET));
    csr_write_simple(1, BIST(k, CHECKER_RANDOM));
    csr_write_simple(bist_base, BIST(k, CHECKER_BASE));
    csr_write_simple(bist_base + bist_len, BIST(k, CHECKER_END));
    csr_write_simple(bist_len, BIST(k, CHECKER_LENGTH));
    csr_write_simple(1, BIST(k, CHECKER_START));
    bist_runs++;
}
static void bist_report(int k)
{
    while (!csr_read_simple(BIST(k, CHECKER_DONE)));
    bist_errs += csr_read_simple(BIST(k, CHECKER_ERRORS));
    printf("BIST ctrl %d under load: %lu runs, %lu errors\n", k, (unsigned long)bist_runs,
           (unsigned long)bist_errs);
    bist_errs = bist_runs = 0;
}

// BIST traffic patterns for the ECC counters (`eccbits clear` / `eccbits` around it):
// `bistpat <k> <mode> [runs]`. The generator writes the `ddrstress` region (32 MB above the
// weights), then the checker reads it `runs` times; mode bit 0: at random addresses (the
// LFSR's, so the checker's own count means nothing - the ECC's per-DQ counts do), bit 1: with
// the generator writing the next 32 MB at the same time (read/write turnarounds). Which
// generic pattern reproduces the read eye the engine's concurrent streams see (ddr_ecc_scan).
static void bistpat_cmd(char *str)
{
    char *kk = get_token(&str), *m = get_token(&str), *n = get_token(&str);
    int k = (int)strtoul(kk, NULL, 0), mode = *m ? (int)strtoul(m, NULL, 0) : 0;
    int runs = *n ? (int)strtoul(n, NULL, 0) : 8;
    if (k < 0 || k >= SDRAM_CONTROLLERS) { puts("bistpat: no such controller"); return; }
    uint64_t t0 = perf_now64();
    uint32_t errs = 0;
    // Random addresses: the generator must visit the checker's LFSR sequence of addresses
    // (a sequential write leaves most of a 32 MB region's random addresses unwritten).
    csr_write_simple(1, BIST(k, GENERATOR_RESET));
    csr_write_simple((mode & 1) ? 3 : 1, BIST(k, GENERATOR_RANDOM));
    csr_write_simple(bist_base, BIST(k, GENERATOR_BASE));
    csr_write_simple(bist_base + bist_len, BIST(k, GENERATOR_END));
    csr_write_simple(bist_len, BIST(k, GENERATOR_LENGTH));
    csr_write_simple(1, BIST(k, GENERATOR_START));
    while (!csr_read_simple(BIST(k, GENERATOR_DONE)));
    for (int i = 0; i < runs; i++) {
        if (mode & 2) {
            csr_write_simple(1, BIST(k, GENERATOR_RESET));
            csr_write_simple(1, BIST(k, GENERATOR_RANDOM));
            csr_write_simple(bist_base + bist_len, BIST(k, GENERATOR_BASE));
            csr_write_simple(bist_base + 2*bist_len, BIST(k, GENERATOR_END));
            csr_write_simple(bist_len, BIST(k, GENERATOR_LENGTH));
            csr_write_simple(1, BIST(k, GENERATOR_START));
        }
        csr_write_simple(1, BIST(k, CHECKER_RESET));
        csr_write_simple((mode & 1) ? 3 : 1, BIST(k, CHECKER_RANDOM));   // data | addr
        csr_write_simple(bist_base, BIST(k, CHECKER_BASE));
        csr_write_simple(bist_base + bist_len, BIST(k, CHECKER_END));
        csr_write_simple(bist_len, BIST(k, CHECKER_LENGTH));
        csr_write_simple(1, BIST(k, CHECKER_START));
        while (!csr_read_simple(BIST(k, CHECKER_DONE)));
        while (!csr_read_simple(BIST(k, GENERATOR_DONE)));
        errs += csr_read_simple(BIST(k, CHECKER_ERRORS));
    }
    printf("BISTPAT ctrl %d mode %d: %d runs of %lu MB in %lu ms, %lu checker errors%s\n", k, mode,
           runs, (unsigned long)(bist_len >> 20),
           (unsigned long)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000)), (unsigned long)errs,
           (mode & 1) ? " (random addresses: no data check)" : "");
}
#endif

// Matvec repeatability: `mvdiff <role> <layer> [reps]` runs one matvec as the reference, then
// reps more on the same input, and reports the output rows that changed (row, old, new).
static void mvdiff_cmd(char *str)
{
    char *r = get_token(&str), *l = get_token(&str), *n = get_token(&str), *m = get_token(&str);
    if (*r == 0) { puts("usage: mvdiff <role> <layer> [reps] [mode: 1 refill x, 2 idle 2 ms, 4 zero dst]"); return; }
    int mode = *m ? (int)strtoul(m, NULL, 0) : 0;
    if (model_header()->magic != BONS_MAGIC) { puts("mvdiff: no model"); return; }
    const struct bons_entry *e = model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0);
    if (!e || e->rows*4 > 0x18000) { puts("mvdiff: no such tensor (or too many rows)"); return; }
    int reps = *n ? (int)strtoul(n, NULL, 0) : 8;
    enum { REF = 0x8000, DST = 0x20000 };
    volatile int32_t *ref = (volatile int32_t *)(SCRATCHPAD_BASE + REF);
    volatile int32_t *out = (volatile int32_t *)(SCRATCHPAD_BASE + DST);
    fill_x_pattern(0, e->cols);
    model_matvec(e, 0, REF);
    int bad_runs = 0;
    for (int rep = 0; rep < reps; rep++) {
        if (mode & 4) for (uint32_t i = 0; i < e->rows; i++) out[i] = 0x5a5a5a5a;
        if (mode & 2) { uint64_t t0 = perf_now64(); while (perf_now64() - t0 < CONFIG_CLOCK_FREQUENCY/500); }
        if (mode & 1) fill_x_pattern(0, e->cols);
#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
        if (mode & 8) bist_kick(1);
#endif
        model_matvec(e, 0, DST);
        int bad = 0;
        for (uint32_t i = 0; i < e->rows; i++)
            if (out[i] != ref[i]) {
                if (bad < 6) printf("  run %d row %lu: %ld -> %ld\n", rep, (unsigned long)i,
                                    (long)ref[i], (long)out[i]);
                bad++;
            }
        if (bad) { printf("  run %d: %d rows differ\n", rep, bad); bad_runs++; }
    }
    printf("MVDIFF role=%u layer=%u rows=%lu mode=%d: %d of %d runs differ from the first\n",
           e->role, e->layer, (unsigned long)e->rows, mode, bad_runs, reps);
#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
    if (mode & 8) bist_report(1);
#endif
}

#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
// DDR read margin under load: `ddrstress <ctrl> <role> <layer> [passes]`. The controller's
// LiteDRAM BIST writes 32 MB of random data at its top (above the weights), then its checker
// re-reads that region in the background while the engine fetches the tensor's chunks on the
// controller's channels; every fetch is compared word by word with a quiet fetch of the same
// chunk, and the differing words are printed with their XOR (the failing DQ bits).
static void ddrstress_cmd(char *str)
{
    char *kk = get_token(&str), *r = get_token(&str), *l = get_token(&str), *n = get_token(&str);
    if (*kk == 0 || *r == 0) { puts("usage: ddrstress <ctrl> <role> <layer> [passes]"); return; }
    int k = (int)strtoul(kk, NULL, 0);
    const struct bons_entry *e = model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0);
    if (!e || k < 0 || k >= SDRAM_CONTROLLERS) { puts("ddrstress: no such tensor or controller"); return; }
    int passes = *n ? (int)strtoul(n, NULL, 0) : 4;
    const uint32_t len = 32u << 20, base = (2u << 30) - len - (64u << 20); // controller bytes
    csr_write_simple(1, BIST(k, GENERATOR_RESET)); csr_write_simple(0, BIST(k, GENERATOR_RESET));
    csr_write_simple(1, BIST(k, GENERATOR_RANDOM));
    csr_write_simple(base, BIST(k, GENERATOR_BASE));
    csr_write_simple(base + len, BIST(k, GENERATOR_END));
    csr_write_simple(len, BIST(k, GENERATOR_LENGTH));
    csr_write_simple(1, BIST(k, GENERATOR_START));
    while (!csr_read_simple(BIST(k, GENERATOR_DONE)));
    enum { CHUNK = 1024, REF = 0x20000, DST = 0x10000 };
    volatile uint32_t *ref = (volatile uint32_t *)(SCRATCHPAD_BASE + REF);
    volatile uint32_t *got = (volatile uint32_t *)(SCRATCHPAD_BASE + DST);
    uint32_t beats = e->chan_len_beats, wbase = TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32;
    uint32_t bad = 0, checks = 0, bist_err = 0, bist_runs = 0;
    uint64_t bitmask = 0;
    int c0 = TRIXIUM_CHAN_MOD ? k*TRIXIUM_CHAN_MOD : 0, c1 = TRIXIUM_CHAN_MOD ? c0 + TRIXIUM_CHAN_MOD : TRIXIUM_N_CHANNELS;
    for (int c = c0; c < c1; c++)
        for (uint32_t ck = 0; ck*CHUNK < beats; ck++) {
            uint32_t nb = (ck + 1)*CHUNK > beats ? beats - ck*CHUNK : CHUNK;
            engine_fetch(c, wbase + ck*CHUNK*32, nb, REF);       // quiet reference
            for (int p = 0; p < passes; p++) {
                if (csr_read_simple(BIST(k, CHECKER_DONE)) || bist_runs == 0) {
                    if (bist_runs) bist_err += csr_read_simple(BIST(k, CHECKER_ERRORS));
                    csr_write_simple(1, BIST(k, CHECKER_RESET)); csr_write_simple(0, BIST(k, CHECKER_RESET));
                    csr_write_simple(1, BIST(k, CHECKER_RANDOM));
                    csr_write_simple(base, BIST(k, CHECKER_BASE));
                    csr_write_simple(base + len, BIST(k, CHECKER_END));
                    csr_write_simple(len, BIST(k, CHECKER_LENGTH));
                    csr_write_simple(1, BIST(k, CHECKER_START));
                    bist_runs++;
                }
                engine_fetch(c, wbase + ck*CHUNK*32, nb, DST);
                checks++;
                for (uint32_t i = 0; i < nb*8; i++)
                    if (got[i] != ref[i]) {
                        uint32_t x = got[i] ^ ref[i];
                        // Word i of a 256-bit beat: the beat spans 4 DFI 64-bit words (2 per
                        // DDR edge pair); bit b of 64-bit word w = DQ b.
                        bitmask |= (uint64_t)x << (32*(i & 1));
                        if (bad < 12) printf("  chan %d chunk %lu word %lu: %08lx ^ %08lx\n", c,
                                             (unsigned long)ck, (unsigned long)i, (unsigned long)ref[i], (unsigned long)x);
                        bad++;
                    }
            }
        }
    while (!csr_read_simple(BIST(k, CHECKER_DONE)));
    bist_err += csr_read_simple(BIST(k, CHECKER_ERRORS));
    printf("DDRSTRESS ctrl %d: %lu fetches, %lu bad words, DQ mask %08lx%08lx; BIST %lu runs, %lu errors\n",
           k, (unsigned long)checks, (unsigned long)bad, (unsigned long)(bitmask >> 32),
           (unsigned long)bitmask, (unsigned long)bist_runs, (unsigned long)bist_err);
}
#endif

// `mvdump <role> <layer>`: one matvec on the self-test input, every output row in hex (16 per
// line), to compare boards row by row.
static void mvdump_cmd(char *str)
{
    char *r = get_token(&str), *l = get_token(&str);
    const struct bons_entry *e = (*r && model_header()->magic == BONS_MAGIC) ?
        model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0) : 0;
    if (!e || e->rows*4 > 0x18000) { puts("usage: mvdump <role> <layer> (a loaded matvec tensor)"); return; }
    fill_x_pattern(0, e->cols);
    model_matvec(e, 0, 0x8000);
    volatile uint32_t *out = (volatile uint32_t *)(SCRATCHPAD_BASE + 0x8000);
    for (uint32_t i = 0; i < e->rows; i++)
        printf("%08lx%c", (unsigned long)out[i], (i % 16 == 15) ? '\n' : ' ');
    printf("MVDUMP role=%u layer=%u rows=%lu\n", e->role, e->layer, (unsigned long)e->rows);
}

// `wsum <role> <layer>`: FNV-1a of the tensor's bytes on every channel (fetched in 32 KB
// chunks), to compare with the image file (the weights as they are in memory).
static void wsum_cmd(char *str)
{
    char *r = get_token(&str), *l = get_token(&str);
    const struct bons_entry *e = (*r && model_header()->magic == BONS_MAGIC) ?
        model_entry((int)strtoul(r, NULL, 0), *l ? (int)strtoul(l, NULL, 0) : 0) : 0;
    if (!e) { puts("usage: wsum <role> <layer> (a loaded tensor)"); return; }
    uint32_t beats = e->chan_len_beats, base = TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32;
    printf("WSUM role=%u layer=%u beats=%lu:", e->role, e->layer, (unsigned long)beats);
    for (int c = 0; c < TRIXIUM_N_CHANNELS; c++) {
        uint32_t h = 2166136261u;
        for (uint32_t b = 0; b < beats; b += 1024) {
            uint32_t n = beats - b < 1024 ? beats - b : 1024;
            engine_fetch(c, base + b*32, n, 0x10000);
            volatile uint8_t *d = (volatile uint8_t *)(SCRATCHPAD_BASE + 0x10000);
            for (uint32_t i = 0; i < n*32; i++) { h ^= d[i]; h *= 16777619u; }
        }
        printf(" %08lx", (unsigned long)h);
    }
    printf("\n");
}

// `chdump <chan> <beat> [n]`: n beats (default 1, max 64) of a channel's weight window from
// payload beat `beat`, as 8 words each - the raw memory behind wsum.
static void chdump_cmd(char *str)
{
    char *c = get_token(&str), *b = get_token(&str), *nn = get_token(&str);
    if (*c == 0 || *b == 0) { puts("usage: chdump <chan> <beat> [n]"); return; }
    int ch = (int)strtoul(c, NULL, 0);
    uint32_t beat = (uint32_t)strtoul(b, NULL, 0), n = *nn ? (uint32_t)strtoul(nn, NULL, 0) : 1;
    if (n > 64) n = 64;
    engine_fetch(ch, TRIXIUM_CHAN_WEIGHT_BASE + beat*32, n, 0x10000);
    volatile uint32_t *d = (volatile uint32_t *)(SCRATCHPAD_BASE + 0x10000);
    for (uint32_t i = 0; i < n; i++) {
        printf("CH%d %lu:", ch, (unsigned long)(beat + i));
        for (int w = 0; w < 8; w++) printf(" %08lx", (unsigned long)d[8*i + w]);
        printf("\n");
    }
}

static int parse_int(const char *s)
{
    int v = 0;
    while (*s >= '0' && *s <= '9')
        v = v*10 + (*s++ - '0');
    return v;
}

/* Generation */
/*------------*/

static void tokclr_cmd(void)
{
    pbuf_n = 0;
    pbuf_kv = 0;
    pbuf_base = 0;
    printf("TOKCLR %d\n", pbuf_n);
}

static void tok_cmd(char *args)
{
    char *tok;
    uint32_t vocab = model_header() ? model_header()->vocab : 0;
    while (1) {
        tok = get_token(&args);
        if (*tok == 0)
            break;
        int id = parse_int(tok);
        // A garbled line (the console drops characters from long bursts) can merge two
        // ids into one past the vocabulary; refuse the whole line so the host resends it.
        if (vocab && (uint32_t)id >= vocab) {
            printf("TOK ERR bad id %d (vocab %lu)\n", id, (unsigned long)vocab);
            return;
        }
        if (pbuf_n == PBUF_N)
            pbuf_make_room(1);
        if (pbuf_n < PBUF_N)
            pbuf[pbuf_n++] = id;
    }
    printf("TOK %d\n", pbuf_n); // Host uses this to confirm the running length.
}

// Generate from the accumulated prompt buffer; appends the emitted tokens to it so the
// next turn continues the conversation without the host resending history. Optional stop
// ids (the chat template's <|im_end|>) end the reply early; the stop token is emitted and
// kept in the buffer so the next turn's context stays well-formed.
static void genb_cmd(char *args)
{
    char *tok = get_token(&args);
    int n = parse_int(tok);
    int stop[INFER_STOP_MAX], nstop = 0;
    while (nstop < INFER_STOP_MAX) {
        tok = get_token(&args);
        if (*tok == 0)
            break;
        stop[nstop++] = parse_int(tok);
    }
    if (n <= 0 || pbuf_n == 0) {
        puts("usage: tok <ids...> ; genb <n> [stop ids...]");
        return;
    }
    pbuf_make_room(n);                    // The reply's tokens join pbuf too.
    printf("GEN:");
    gen_emit_to_pbuf = 1;
    infer_set_stop(stop, nstop);
    infer_set_recent(pbuf, &pbuf_n);      // the repetition penalty's window
    // Continue from the KV cache when the buffer's head is still what the last genb left
    // there (gen/bench in between reset it); otherwise the whole buffer is re-forwarded.
    int resident = (pbuf_serial == infer_gen_serial()) ? pbuf_kv : 0;
    if (resident == 0)
        infer_set_pin(pbuf_n);            // A fresh conversation: its first turn stays attended.
    infer_set_base(pbuf_base);
    int r = infer_generate_from(pbuf, pbuf_n, resident, n, emit_token);
    pbuf_kv     = infer_kv_resident();     // Relative to pbuf: before the base goes back to 0.
    infer_set_base(0);
    pbuf_serial = infer_gen_serial();
    infer_set_stop(stop, 0);
    gen_emit_to_pbuf = 0;
    puts(r < 0 ? "\nGEN ERR" : "\nGEN DONE");
    if (engine_wedged) {          // Recover here, after the stream: the client reads GEN ERR.
        engine_recover();
        pbuf_kv = 0;              // The KV cache is not trustworthy: the next genb re-forwards.
    }
}

/* On-board Chat */
/*---------------*/

// The tokenizer runs here: a plain serial terminal is the whole client. The conversation lives
// in the prompt buffer and continues from the KV cache, as genb's.

static void (*chat_out)(const uint8_t *b, uint32_t n);   // chat_turn's reply bytes.

static void chat_emit(int id)
{
    const struct ttok *t = tok_get();
    if (pbuf_n < PBUF_N)
        pbuf[pbuf_n++] = id;
    if (!t || id == t->im_end || id == t->im_start)
        return;                   // Template tokens are structure, not text.
    uint32_t len;
    const uint8_t *b = ttok_piece(t, id, &len);
    chat_out(b, len);             // UTF-8 bytes: the reader joins a split character.
}

static void chat_putbytes(const uint8_t *b, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        putchar(b[i]);
}

// The conversation (pbuf) the console `chat` and the web page share.
static int chat_first = 1, chat_prev_closed = 1;

int chat_ready(void)
{
    const struct ttok *t = tok_get();
    const struct bons_header *h = model_header();
    return t && t->im_end >= 0 && h && h->magic == BONS_MAGIC;
}

void chat_reset(void)
{
    pbuf_n = pbuf_kv = pbuf_base = 0;
    chat_first = chat_prev_closed = 1;
}

// One turn: `line` in, the reply's bytes to `out` as they come (nmax tokens at most).
int chat_turn(const char *line, int nmax, void (*out)(const uint8_t *b, uint32_t n),
              struct chat_result *res)
{
    const struct ttok *t = tok_get();
    memset(res, 0, sizeof(*res));
    // Past half the buffer with the KV ring: room for a long turn and its reply.
    if (pbuf_n > PBUF_N/2)
        pbuf_make_room(PBUF_N/2);
    int n = ttok_turn(t, line, NULL, chat_first, chat_prev_closed, pbuf + pbuf_n, PBUF_N - pbuf_n);
    if (pbuf_n + n + nmax > PBUF_N) {        // The context is full: start over with this turn.
        res->restarted = 1;
        pbuf_n = pbuf_kv = pbuf_base = 0;
        n = ttok_turn(t, line, NULL, 1, 1, pbuf, PBUF_N);
    }
    pbuf_n += n;
    int before = pbuf_n, stop = t->im_end;
    infer_set_stop(&stop, 1);
    infer_set_recent(pbuf, &pbuf_n);
    int resident = (pbuf_serial == infer_gen_serial()) ? pbuf_kv : 0;
    if (resident == 0)
        infer_set_pin(before);              // A fresh conversation: its first turn stays attended.
    uint64_t t0 = perf_now64();
    chat_out = out;
    infer_set_base(pbuf_base);
    int r = infer_generate_from(pbuf, before, resident, nmax, chat_emit);
    pbuf_kv     = infer_kv_resident();      // Relative to pbuf: before the base goes back to 0.
    infer_set_base(0);
    uint64_t dt = perf_now64() - t0;
    pbuf_serial = infer_gen_serial();
    infer_set_stop(&stop, 0);
    res->tokens  = pbuf_n - before;
    res->ms      = (uint32_t)(dt/(CONFIG_CLOCK_FREQUENCY/1000));
    chat_prev_closed = res->tokens > 0 && pbuf[pbuf_n - 1] == t->im_end;
    res->capped  = !chat_prev_closed;
    chat_first = 0;
    res->context = pbuf_base + pbuf_n;
    if (r < 0 || engine_wedged) {
        res->error = 1;
        if (engine_wedged) engine_recover();
        chat_reset();
    }
    return res->error ? -1 : 0;
}

// One line from the console, echoed; backspace removes a whole UTF-8 character. -1 on Ctrl-D.
static int chat_readline(char *buf, int cap)
{
    int n = 0;
    while (1) {
        int c = getchar();
        if (c == '\r' || c == '\n') {
            if (n == 0 && c == '\n')
                continue;         // The \n of a terminal's \r\n.
            putchar('\n');
            buf[n] = 0;
            return n;
        }
        if (c == 0x04)
            return -1;
        if (c == 0x7f || c == 0x08) {
            if (n > 0) {
                while (--n > 0 && ((uint8_t)buf[n] & 0xC0) == 0x80);
                fputs("\x08 \x08", stdout);
            }
            continue;
        }
        if (n < cap - 1) {
            buf[n++] = (char)c;
            putchar(c);
        }
    }
}

static void chat_cmd(char *args)
{
    if (!chat_ready()) {
        puts(tok_get() ? "chat: no model (hdrload)" : "chat: no tokenizer (tokload)");
        return;
    }
    char *a = get_token(&args);
    int nmax = *a ? parse_int(a) : 512;
    if (nmax < 1 || nmax > PBUF_N/2) nmax = 512;
    char *line = (char *)TRIXIUM_CHAT_LINE_ADDR;
    chat_reset();
    puts("TriXium chat, the tokenizer on the board: /reset clears the conversation, /exit or Ctrl-D leaves");
    while (1) {
        printf("\nyou> ");
        int ln = chat_readline(line, TRIXIUM_CHAT_LINE_SIZE);
        // A trailing backslash continues the message on the next line.
        while (ln > 0 && line[ln - 1] == '\\') {
            line[ln - 1] = '\n';
            printf("   > ");
            int m = chat_readline(line + ln, TRIXIUM_CHAT_LINE_SIZE - ln);
            if (m < 0) { ln = -1; break; }
            ln += m;
        }
        if (ln < 0 || strcmp(line, "/exit") == 0)
            break;
        if (ln == 0)
            continue;
        if (strcmp(line, "/reset") == 0) {
            chat_reset();
            puts("(conversation cleared)");
            continue;
        }
        struct chat_result res;
        printf("bonsai> ");
        chat_turn(line, nmax, chat_putbytes, &res);
        if (res.restarted)
            printf("\n(context full: conversation restarted)");
        printf("\n[%d tokens in %lu.%01lu s, context %d%s]\n", res.tokens, (unsigned long)(res.ms/1000),
               (unsigned long)(res.ms%1000/100), res.context, res.capped ? ", capped" : "");
        if (res.error)
            puts("(generation error: conversation restarted)");
    }
    puts("(chat ended)");
}

static void gen_cmd(char *args)
{
    int *const prompt = (int *)TRIXIUM_CONSOLE_PROMPT_ADDR;
    char *tok = get_token(&args);
    int n = parse_int(tok);
    int np = 0;
    while (np < 64) {
        tok = get_token(&args);
        if (*tok == 0)
            break;
        prompt[np++] = parse_int(tok);
    }
    if (n <= 0 || np == 0) {
        puts("usage: gen n id id ...");
        return;
    }
    printf("GEN:");
    int r = infer_generate(prompt, np, n, emit_token);
    puts(r < 0 ? "\nGEN ERR" : "\nGEN DONE");
    if (engine_wedged)
        engine_recover();
}

#ifdef CSR_TRIXIUM_ENGINE_CSR_NR_ADDR
// `gen2 n idsA... | idsB...`: two fresh conversations decoded together, one weight pass per step
// for both (infer_generate_dual); prints each stream, then the cycles. `gen2 seq` runs the same
// steps as two single forwards (the A/B, same tokens), `gen2 dual` back; `gen2 trace a|b`: the
// conversation `dbg N` traces.
static int gen2_dual = 1, gen2_trace;
static void gen2_cmd(char *args)
{
    // In the main_ram static window: 2.5 KB on the 4 KiB SRAM stack overflowed into .bss on the
    // single-forward path (`gen2 seq`, a sequence alone), corrupting later runs.
    int *const buf = (int *)TRIXIUM_GEN2_BUF_ADDR;
    int *const prompt[2] = {buf, buf + 64}, *const out[2] = {buf + 128, buf + 384};
    int np[2] = {0, 0}, s = 0;
    char *tok = get_token(&args);
    if (strcmp(tok, "dual") == 0 || strcmp(tok, "seq") == 0) {
        gen2_dual = tok[0] == 'd';
        printf("GEN2 %s\n", gen2_dual ? "dual (one weight pass per step for both)" : "seq (two single forwards per step)");
        return;
    }
    if (strcmp(tok, "lmhead") == 0) {
        tok = get_token(&args);
        if (strncmp(tok, "on", 2) == 0)  hybrid_lmhead2_enable(1);
        if (strncmp(tok, "off", 3) == 0) hybrid_lmhead2_enable(0);
        printf("GEN2 lmhead %s\n", hybrid_lmhead2_enabled() ? "on (both LM heads in one weight pass)" : "off (one LM head per conversation)");
        return;
    }
    if (strcmp(tok, "trace") == 0) {
        tok = get_token(&args);
        gen2_trace = tok[0] == 'b';
        printf("GEN2 trace %c\n", gen2_trace ? 'b' : 'a');
        return;
    }
    int n = parse_int(tok);
    while (1) {
        tok = get_token(&args);
        if (*tok == 0)
            break;
        if (tok[0] == '|') {
            if (++s > 1) break;
            continue;
        }
        if (np[s] < 64)
            prompt[s][np[s]++] = parse_int(tok);
    }
    if (n <= 0 || n > 256 || s != 1 || !np[0] || !np[1]) {
        puts("usage: gen2 n id id ... | id id ...  (n <= 256, prompts <= 64 ids)");
        return;
    }
    int count[2], steps;
    uint64_t cyc[2];
    printf("GEN2:");
    int r = infer_generate_dual((const int *const *)prompt, np, n, gen2_dual, gen2_trace, out, count, cyc, &steps);
    for (int k = 0; k < 2; k++) {
        printf("\nGEN2 %c:", 'A' + k);
        for (int i = 0; i < count[k]; i++)
            printf(" %d", out[k][i]);
    }
    printf("\nGEN2 %s: prefill %lu cycles, %d steps with both %lu cycles (%lu per step)\n",
           gen2_dual ? "dual" : "seq", (unsigned long)cyc[0], steps, (unsigned long)cyc[1],
           (unsigned long)(steps ? cyc[1]/(uint64_t)steps : 0));
    puts(r < 0 ? "GEN2 ERR" : "GEN2 DONE");
    if (engine_wedged)
        engine_recover();
}
#endif

/* Self-tests */
/*------------*/

static void sptest(void)
{
    volatile uint32_t *sp = (volatile uint32_t *)SCRATCHPAD_BASE;
    int errors = 0;
    for (uint32_t i = 0; i < 64; i++)
        sp[i] = i*0x9E3779B1u;
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t got = sp[i], exp = i*0x9E3779B1u;
        if (got != exp) {
            if (errors < 4)
                printf("  sp[%lu]: got %08lx expect %08lx\n",
                       (unsigned long)i, (unsigned long)got, (unsigned long)exp);
            errors++;
        }
    }
    printf("SPTEST %s (%d errors)\n", errors ? "FAIL" : "OK", errors);
}

static void selftest(void)
{
    const struct bons_header *h = model_header();
    if (h->magic != BONS_MAGIC) {
        puts("SELFTEST SKIP (no model)");
        return;
    }
    const struct bons_entry *e = model_entry(ROLE_ATTN_Q, 0);
    if (!e) {
        puts("SELFTEST SKIP (no attn_q)");
        return;
    }

    // 1. Ternary matvec on blk.0.attn_q with the deterministic x pattern.
    uint32_t xsrc = 0, dst = 0x8000;
    puts("ST:fill");
    fill_x_pattern(xsrc, e->cols);
    puts("ST:mv");
    model_matvec(e, xsrc, dst);
    uint32_t n = e->rows*4; // rows int32 results (all channels).
    uint32_t csum = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + dst), n);
    volatile uint32_t *res = (volatile uint32_t *)(SCRATCHPAD_BASE + dst);
    printf("MATVEC R0=%08lx R1=%08lx R64=%08lx R128=%08lx R255=%08lx\n",
           (unsigned long)res[0], (unsigned long)res[1], (unsigned long)res[64],
           (unsigned long)res[128], (unsigned long)res[e->rows - 1]);
    printf("MATVEC CSUM=%08lx\n", (unsigned long)csum);

    // 2. Fetch the first 8 beats of channel 0's payload into the scratchpad.
    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE + e->chan_base_beats*32, 8, 0xC000);
    csum = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + 0xC000), 8*32);
    printf("FETCH CSUM=%08lx\n", (unsigned long)csum);

    puts("SELFTEST DONE");
}

/* Console Service */
/*-----------------*/

// Stack tripwire: the stack sits at the top of the SRAM above .bss with a 4 KiB floor
// (linker.ld) and an overflow corrupts .bss silently (a later run wrong, or a hang). The bytes
// just above .bss are painted at boot and checked after every console poll.
extern char _ebss[];
#define STACK_GUARD 256
static void stack_guard_paint(void)
{
    memset(_ebss, 0x5a, STACK_GUARD);
}
static void stack_guard_check(void)
{
    for (int i = 0; i < STACK_GUARD; i++)
        if (_ebss[i] != 0x5a) {
            printf("\nSTACK OVERFLOW: the SRAM stack reached .bss; results from here are suspect, "
                   "reboot\n");
            stack_guard_paint();
            return;
        }
}

static void console_service(void)
{
    char *str, *token;

    str = readstr();
    if (str == NULL)
        return;
    token = get_token(&str);
    if (strcmp(token, "help") == 0)
        help();
    else if (strcmp(token, "info") == 0)
        model_info();
    else if (strcmp(token, "sptest") == 0)
        sptest();
    else if (strcmp(token, "peek") == 0)
        peek_cmd(str);
    else if (strcmp(token, "imgcheck") == 0)
        imgcheck_cmd(str);
#ifdef CSR_TRIXIUM_ENGINE_CSR_CYC_VEC_ADDR
    else if (strcmp(token, "spcost") == 0) { // spcost [n]: CPU scratchpad word accesses vs the busy counters
        char *nn = get_token(&str);
        uint32_t n = *nn ? (uint32_t)strtoul(nn, NULL, 0) : 1024, acc = 0;
        volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
        uint32_t e0 = trixium_engine_csr_cyc_eng_read(), v0 = trixium_engine_csr_cyc_vec_read();
        uint64_t t0 = perf_now64();
        for (uint32_t i = 0; i < n; i++) acc += sp[i & 1023];
        uint64_t t1 = perf_now64();
        for (uint32_t i = 0; i < n; i++) sp[i & 1023] = i;
        uint64_t t2 = perf_now64();
        uint32_t e1 = trixium_engine_csr_cyc_eng_read(), v1 = trixium_engine_csr_cyc_vec_read();
        printf("SPCOST %lu reads %lu cycles, %lu writes %lu cycles; busy deltas eng %lu vec %lu (acc %08lx)\n",
               (unsigned long)n, (unsigned long)(t1 - t0), (unsigned long)n, (unsigned long)(t2 - t1),
               (unsigned long)(e1 - e0), (unsigned long)(v1 - v0), (unsigned long)acc);
    }
#endif
#ifdef TRIXIUM_CHAN_WEIGHT_BASE
    else if (strcmp(token, "fetchload") == 0) { // fetchload <chan> [mb] [passes]: one channel's sequential FETCH stream (the read eye under a single stream: eccbits around it)
        char *cc = get_token(&str), *mm = get_token(&str), *pp = get_token(&str);
        uint32_t chan = (uint32_t)strtoul(cc, NULL, 0), mb = *mm ? (uint32_t)strtoul(mm, NULL, 0) : 32;
        uint32_t passes = *pp ? (uint32_t)strtoul(pp, NULL, 0) : 4, beats = 1024;   // 32 KB into RS_Y (transient)
        uint64_t t0 = perf_now64();
        for (uint32_t p = 0; p < passes && !engine_wedged; p++)
            for (uint32_t off = 0; off < (mb << 20) && !engine_wedged; off += beats*32)
                engine_fetch((uint8_t)chan, TRIXIUM_CHAN_WEIGHT_BASE + off, beats, 0xC000);
        printf("FETCHLOAD chan %lu: %lu x %lu MB in %lu ms%s\n", (unsigned long)chan, (unsigned long)passes,
               (unsigned long)mb, (unsigned long)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000)),
               engine_wedged ? " WEDGED" : "");
    }
#endif
    else if (strcmp(token, "mvtest") == 0)
        mvtest_cmd(str);
    else if (strcmp(token, "mvbw") == 0)
        mvbw_cmd(str);
#if defined(CSR_SDRAM_STATS_SNAP_ADDR) && defined(TRIXIUM_DDR_CSR_STRIDE) && defined(TRIXIUM_N_MEMCTRL)
    else if (strcmp(token, "ddrstat") == 0)   // ddrstat: DFI command cycles per controller + engine port waits since the previous call
        ddrstat_cmd();
#endif
#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
    else if (strcmp(token, "ddrstress") == 0)
        ddrstress_cmd(str);
#endif
#if defined(CSR_SDRAM_GENERATOR_BASE_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
    else if (strcmp(token, "bistload") == 0) { // bistload <k> <runs>: the BIST alone, no engine
        char *kk = get_token(&str), *nn = get_token(&str);
        int k = (int)strtoul(kk, NULL, 0), runs = *nn ? (int)strtoul(nn, NULL, 0) : 16;
        for (int i = 0; i < runs; i++) {
            bist_kick(k);
            while (!csr_read_simple(BIST(k, CHECKER_DONE)));
        }
        bist_report(k);
    }
    else if (strcmp(token, "fetchbist") == 0) { // fetchbist <k> <chan> [mb] [runs]: the checker's sequential stream on controller k while one engine channel streams: two ports (eccbits around it)
        char *kk = get_token(&str), *cc = get_token(&str), *mm = get_token(&str), *nn = get_token(&str);
        int k = (int)strtoul(kk, NULL, 0), runs = *nn ? (int)strtoul(nn, NULL, 0) : 8;
        uint32_t chan = (uint32_t)strtoul(cc, NULL, 0), mb = *mm ? (uint32_t)strtoul(mm, NULL, 0) : 32, beats = 1024;
        if (k < 0 || k >= SDRAM_CONTROLLERS) { puts("fetchbist: no such controller"); return; }
        csr_write_simple(1, BIST(k, GENERATOR_RESET)); csr_write_simple(0, BIST(k, GENERATOR_RESET));
        csr_write_simple(1, BIST(k, GENERATOR_RANDOM));
        csr_write_simple(bist_base, BIST(k, GENERATOR_BASE));
        csr_write_simple(bist_base + bist_len, BIST(k, GENERATOR_END));
        csr_write_simple(bist_len, BIST(k, GENERATOR_LENGTH));
        csr_write_simple(1, BIST(k, GENERATOR_START));
        while (!csr_read_simple(BIST(k, GENERATOR_DONE)));
        uint64_t t0 = perf_now64();
        uint32_t passes = 0;
        bist_kick(k);
        while (bist_runs < (uint32_t)runs && !engine_wedged) {
            for (uint32_t off = 0; off < (mb << 20) && !engine_wedged; off += beats*32)
                engine_fetch((uint8_t)chan, TRIXIUM_CHAN_WEIGHT_BASE + off, beats, 0xC000);   // 32 KB into RS_Y (transient)
            passes++;
            bist_kick(k);
        }
        printf("FETCHBIST chan %lu: %lu x %lu MB in %lu ms%s; ", (unsigned long)chan, (unsigned long)passes,
               (unsigned long)mb, (unsigned long)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000)),
               engine_wedged ? " WEDGED" : "");
        bist_report(k);
    }
    else if (strcmp(token, "bistpat") == 0) // bistpat <k> <mode> [runs]: BIST traffic patterns, for eccbits
        bistpat_cmd(str);
#endif
#if defined(SDRAM_CSR_STRIDE)
    else if (strcmp(token, "ddrmr") == 0) { // ddrmr <k> <mr> <value>
        char *kk = get_token(&str), *m = get_token(&str), *v = get_token(&str);
        ddr_mode_register((int)strtoul(kk, NULL, 0), (int)strtoul(m, NULL, 0), (int)strtoul(v, NULL, 0));
    }
    else if (strcmp(token, "ddrbit") == 0) { // ddrbit <k> <dq> <off>
        char *kk = get_token(&str), *d = get_token(&str), *o = get_token(&str);
        int off = *o == '-' ? -(int)strtoul(o + 1, NULL, 0) : (int)strtoul(o, NULL, 0);
        int t = ddr_bit_offset((int)strtoul(kk, NULL, 0), (int)strtoul(d, NULL, 0), off);
        printf("DDRBIT ctrl %s dq %s off %+d -> tap %d\n", kk, d, off, t);
    }
    else if (strcmp(token, "eccscan") == 0) { // eccscan [k] [range] [passes]: per-bit read eye under engine traffic
        char *kk = get_token(&str), *r = get_token(&str), *p = get_token(&str);
        int n = ddr_ecc_scan(*kk ? (int)strtoul(kk, NULL, 0) : 1, *r ? (int)strtoul(r, NULL, 0) : 3,
                             *p ? (int)strtoul(p, NULL, 0) : 2);
        printf("ECCSCAN %d\n", n);
    }
    else if (strcmp(token, "eyescan") == 0) { // eyescan [k] [range] [forwards]: per-bit read eye under the model's forwards
        char *kk = get_token(&str), *r = get_token(&str), *p = get_token(&str);
        int n = ddr_eye_scan(*kk ? (int)strtoul(kk, NULL, 0) : 1, *r ? (int)strtoul(r, NULL, 0) : 3,
                             *p ? (int)strtoul(p, NULL, 0) : 2);
        printf("EYESCAN %d\n", n);
    }
    else if (strcmp(token, "ddrtap") == 0) {
        char *kk = get_token(&str), *o = get_token(&str);
        ddr_tap_offset((int)strtoul(kk, NULL, 0), *o == '-' ? -(int)strtoul(o + 1, NULL, 0) : (int)strtoul(o, NULL, 0));
    }
#endif
    else if (strcmp(token, "attndelay") == 0) {
        extern uint32_t attn_delay;
        char *v = get_token(&str);
        if (*v) attn_delay = (uint32_t)strtoul(v, NULL, 0);
        printf("ATTNDELAY %lu\n", (unsigned long)attn_delay);
    }
    else if (strcmp(token, "kvfill") == 0) {
        extern void infer_kv_fill(uint32_t pattern);
        char *v = get_token(&str);
        infer_kv_fill(*v ? (uint32_t)strtoul(v, NULL, 0) : 0);
    }
    else if (strcmp(token, "foldearly") == 0) {
        extern int attn_fold_early;
        char *v = get_token(&str);
        if (*v) attn_fold_early = (int)strtoul(v, NULL, 0);
        printf("FOLDEARLY %d\n", attn_fold_early);
    }
    else if (strcmp(token, "chdump") == 0)
        chdump_cmd(str);
    else if (strcmp(token, "wsum") == 0)
        wsum_cmd(str);
    else if (strcmp(token, "eccforce") == 0) { // eccforce N: re-run each of the next N forwards
        char *c = get_token(&str);
        infer_ecc_force = c ? (uint32_t)strtoul(c, NULL, 0) : 0;
        printf("ECC retry forced on the next %lu forwards (retries so far %lu)\n",
               (unsigned long)infer_ecc_force, (unsigned long)infer_ecc_retries);
    }
#if defined(CSR_SDRAM_ECC_BITSEL_ADDR) && defined(SDRAM_CSR_STRIDE) && defined(SDRAM_CONTROLLERS)
    else if (strcmp(token, "eccbits") == 0) { // eccbits [clear]: ECC counters, per DQ bit
        char *c = get_token(&str);
        for (int k = 0; k < SDRAM_CONTROLLERS; k++) {
            unsigned long st = (unsigned long)k*SDRAM_CSR_STRIDE;
            if (c && strcmp(c, "clear") == 0) { csr_write_simple(1, CSR_SDRAM_ECC_CLEAR_ADDR + st); continue; }
            printf("ECC ctrl %d: %lu corrected, %lu uncorrectable; per DQ (nonzero):", k,
                   (unsigned long)csr_read_simple(CSR_SDRAM_ECC_SEC_ADDR + st),
                   (unsigned long)csr_read_simple(CSR_SDRAM_ECC_DED_ADDR + st));
            for (int b = 0; b < 72; b++) {
                csr_write_simple(b, CSR_SDRAM_ECC_BITSEL_ADDR + st);
                (void)csr_read_simple(CSR_SDRAM_ECC_BITCNT_ADDR + st); // registered mux: settle
                uint32_t n = csr_read_simple(CSR_SDRAM_ECC_BITCNT_ADDR + st);
                if (n) printf(" %s%d:%lu", b >= 64 ? "E" : "", b >= 64 ? b - 64 : b, (unsigned long)n);
            }
            printf("\n");
        }
        printf("ECC forward retries %lu, unrecovered %lu\n", (unsigned long)infer_ecc_retries,
               (unsigned long)infer_ecc_unrecovered);
        if (c && strcmp(c, "clear") == 0) {
            infer_ecc_retries = infer_ecc_unrecovered = 0;
            puts("ECC counters cleared");
        }
    }
#endif
    else if (strcmp(token, "mvdump") == 0)
        mvdump_cmd(str);
    else if (strcmp(token, "mvdiff") == 0)
        mvdiff_cmd(str);
    else if (strcmp(token, "fetchtest") == 0)
        fetchtest_cmd(str);
    else if (strcmp(token, "selftest") == 0)
        selftest();
    else if (strcmp(token, "gen") == 0)
        gen_cmd(str);
#ifdef CSR_TRIXIUM_ENGINE_CSR_NR_ADDR
    else if (strcmp(token, "gen2") == 0)
        gen2_cmd(str);
#endif
    else if (strcmp(token, "tok") == 0)
        tok_cmd(str);
    else if (strcmp(token, "tokclr") == 0)
        tokclr_cmd();
    else if (strcmp(token, "genb") == 0)
        genb_cmd(str);
    else if (strcmp(token, "hwops") == 0)
        hwops_cmd(str);
    else if (strcmp(token, "sample") == 0) { // sample <temp> <top_p> <rep> [seed], all x1000
        char *a = get_token(&str), *b = get_token(&str), *c = get_token(&str), *d = get_token(&str);
        if (*a) // Integers scaled by 1000: the console parser has no floats.
            infer_set_sampling(parse_int(a)/1000.0f, *b ? parse_int(b)/1000.0f : 1.0f,
                               *c ? parse_int(c)/1000.0f : 1.0f, *d ? (uint32_t)parse_int(d) : 0);
        float t, p, r;
        infer_get_sampling(&t, &p, &r);
        printf("SAMPLE temp=%d.%03d top_p=%d.%03d rep=%d.%03d (x1000: sample 700 900 1100 [seed])\n",
               (int)t, (int)(t*1000) % 1000, (int)p, (int)(p*1000) % 1000, (int)r, (int)(r*1000) % 1000);
    } else if (strcmp(token, "attnchunk") == 0)
        infer_attn_chunk(parse_int(str));
    else if (strcmp(token, "engreset") == 0)
        engine_recover();
#ifdef TRIXIUM_N_MEMCTRL
    else if (strcmp(token, "ddrinit") == 0) {
        char *arg = get_token(&str);
        if (arg && *arg) ddr_init(parse_int(arg)); else ddr_init_all();
    } else if (strcmp(token, "ddrtest") == 0) {
        char *arg = get_token(&str);
        ddr_test((arg && *arg) ? parse_int(arg) : -1);
    } else if (strcmp(token, "ddrdeep") == 0) { // ddrdeep <blocks> [chan]: destructive
        char *b = get_token(&str), *c = get_token(&str);
        ddr_test_deep((b && *b) ? parse_int(b) : 256, (c && *c) ? parse_int(c) : -1);
    }
#endif
    else if (strcmp(token, "wedge") == 0) {
        // Deliberately wedge the engine (an op whose stream is 2 beats short of its rows),
        // let the stall reporter give up, recover, and rerun the self-test: the checksums
        // must match the boot's. The recovery path's own test, on the sim and the card.
        struct engine_op op = {0};
        op.kind = ENGINE_KIND_TERNARY; op.rows = 8; op.cols = 256; op.wbase = 0;
        op.wlen = 2; op.sbeats = 1; op.cbeats = 2; op.xsrc = 0; op.dst = 0x4000;
        puts("WEDGE: submitting an op that cannot complete");
        engine_run(&op);
        if (!engine_wedged) { puts("WEDGE: the op completed?!"); return; }
        if (engine_recover() == 0) {
            puts("WEDGE: recovered - self-test follows");
            selftest();
        }
    } else if (strcmp(token, "dbg") == 0) {
        int n = (*str) ? parse_int(str) : 1;
        infer_dbg(n);
        printf("DBG armed for %d forwards\n", n);
    } else if (strcmp(token, "stats") == 0) {
        if (strncmp(str, "reset", 5) == 0)
            perf_reset();
        else
            perf_print();
    } else if (strcmp(token, "bench") == 0)
        bench_cmd(str);
    else if (strcmp(token, "fprof") == 0) {
        extern int fprof_use; extern void fprof_reset(void), fprof_print(void);
        if (strncmp(str, "on", 2) == 0)  { fprof_use = 1; fprof_reset(); }
        if (strncmp(str, "off", 3) == 0) fprof_use = 0;
        if (strncmp(str, "print", 5) == 0) fprof_print();
        printf("FPROF %s\n", fprof_use ? "on" : "off");
    } else if (strcmp(token, "listcache") == 0) { // listcache [off|on|verify]: the list image cache
        if (strncmp(str, "off", 3) == 0)    resident_listcache_mode(0);
        if (strncmp(str, "on", 2) == 0)     resident_listcache_mode(1);
        if (strncmp(str, "verify", 6) == 0) resident_listcache_mode(2);
        resident_listcache_print();
    } else if (strcmp(token, "tprof") == 0) {
        if (strncmp(str, "arm", 3) == 0) {
            const char *p = str + 3;
            while (*p == ' ') p++;
            int n = parse_int(p);
            tprof_armed = n > 0 ? n : 1;
            printf("TPROF armed: generated token #%d traces itself (gen or bench)\n",
                   tprof_armed);
        }
        if (strncmp(str, "dump", 4) == 0) { tprof_dump(); resident_listprof_names(); }
        if (str[0] == 0)
            printf("TPROF %s (arm | dump)\n",
                   tprof_armed ? "armed" : tprof_active ? "tracing" : "idle");
    } else if (strcmp(token, "listlat") == 0) {
        // listlat [n]: the round trip of an empty resident list (its loader FETCH and stop), n
        // times: the fixed cost of every list start.
        int n = *str ? (int)strtoul(str, NULL, 0) : 64, bad = 0;
        uint64_t t0 = perf_now64();
        for (int i = 0; i < n && !bad; i++) { rl_begin(); bad = rl_run("empty list"); }
        uint64_t dt = perf_now64() - t0;
        printf("LISTLAT %d empty lists: %lu cycles each%s\n", n, (unsigned long)(dt/(uint32_t)(n > 0 ? n : 1)),
               bad ? " (FAILED)" : "");
    } else if (strcmp(token, "prefetch") == 0) {
        // prefetch [on|off|hybrid on|hybrid off]: the engine's weight prefetch (the Qwen3 path's
        // calls), and the hybrid lists' prefetch descriptors (resident.c, default off).
        if (strncmp(str, "on", 2) == 0)  engine_pf_use = 1;
        if (strncmp(str, "off", 3) == 0) engine_pf_use = 0;
        if (strncmp(str, "hybrid on", 9) == 0)   hybrid_pf_use = 1;
        if (strncmp(str, "hybrid off", 10) == 0) hybrid_pf_use = 0;
#ifdef TRIXIUM_HAS_PREFETCH
        printf("PREFETCH %s, hybrid lists %s (hits %lu, mismatch %lu, reader FIFO %d beats)\n",
               engine_pf_use ? "on" : "off", hybrid_pf_use ? "on" : "off",
               (unsigned long)trixium_engine_csr_pf_hits_read(),
               (unsigned long)trixium_engine_csr_pf_mismatch_read(), TRIXIUM_READER_FIFO_DEPTH);
#else
        engine_pf_use = 0;
        puts("PREFETCH off (no weight prefetch in this bitstream)");
#endif
    } else if (strcmp(token, "mmio") == 0) {
        // Cycles per CPU access on each path the per-token firmware uses: a TriXium CSR read /
        // write, a scratchpad word read / write (posted), and a main_ram (cached) read.
        volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_DESC + 191*64);
        uint32_t keep = sp[0], acc = 0, n = 256;
        uint32_t t0 = perf_now();
        for (uint32_t i = 0; i < n; i++) acc += trixium_seq_status_read();
        uint32_t t1 = perf_now();
        for (uint32_t i = 0; i < n; i++) trixium_seq_count_write(i & 1);
        seq_csr_forget();
        uint32_t t2 = perf_now();
        for (uint32_t i = 0; i < n; i++) acc += sp[i & 15];
        uint32_t t3 = perf_now();
        for (uint32_t i = 0; i < n; i++) sp[i & 7] = i;
        uint32_t t4 = perf_now();
        static volatile uint32_t ram[16];
        for (uint32_t i = 0; i < n; i++) acc += ram[i & 15];
        uint32_t t5 = perf_now();
        extern uint8_t *heap;
        volatile uint32_t *mr = (volatile uint32_t *)heap; // main_ram (cached, behind the L2).
        uint32_t m0 = perf_now();
        for (uint32_t i = 0; i < n; i++) acc += mr[i & 15];
        uint32_t m1 = perf_now();
        for (uint32_t i = 0; i < n; i++) mr[i & 15] = i;
        uint32_t m2 = perf_now();
        uint32_t t5b = perf_now();
        for (uint32_t i = 0; i < n; i++) __asm__ volatile("");
        uint32_t t6 = perf_now();
        sp[0] = keep;
        uint32_t loop = (t6 - t5b);
        printf("MMIO main_ram (heap) rd %lu, wr %lu; sram .bss at %p, heap at %p\n",
               (unsigned long)((m1 - m0 - loop)/n), (unsigned long)((m2 - m1 - loop)/n),
               (void *)ram, (void *)heap);
        printf("MMIO cycles/access: csr rd %lu, csr wr %lu, sp rd %lu, sp wr %lu, ram rd %lu (loop %lu/%lu, acc %lx)\n",
               (unsigned long)((t1 - t0 - loop)/n), (unsigned long)((t2 - t1 - loop)/n),
               (unsigned long)((t3 - t2 - loop)/n), (unsigned long)((t4 - t3 - loop)/n),
               (unsigned long)((t5 - t4 - loop)/n), (unsigned long)loop, (unsigned long)n,
               (unsigned long)acc);
    } else if (strcmp(token, "seq") == 0) {
        if (strncmp(str, "on", 2) == 0)
            infer_seq(1);
        else if (strncmp(str, "off", 3) == 0)
            infer_seq(0);
        printf("SEQ %s (off runs the same ops CPU-dispatched per head, ~20%% slower)\n",
               seq_use ? "on" : "off");
    } else if (strcmp(token, "lmchunk") == 0) {
        uint32_t rows = (uint32_t)strtoul(str, NULL, 0);
        hybrid_lm_chunk(rows);
        printf("LMCHUNK %lu (LM head list chunk cap in rows, 0 = automatic)\n", (unsigned long)rows);
    } else if (strcmp(token, "softmaxdiv") == 0) { // softmaxdiv on|off: the softmax's divide per position (on) or one reciprocal per head (off, the default and the mirror's): a same-boot comparison
        extern int hybrid_softmax_div;
        if (strncmp(str, "on", 2) == 0)  hybrid_softmax_div = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_softmax_div = 0;
        printf("SOFTMAXDIV %s\n", hybrid_softmax_div ? "on (a divide per position)" : "off (one reciprocal per head)");
    } else if (strcmp(token, "kvring") == 0) { // kvring on|off: past the KV window the attention keeps 4 sink positions + a ring of the latest (on) or the generation stops (off)
        extern int hybrid_kv_ring;
        if (strncmp(str, "on", 2) == 0)  hybrid_kv_ring = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_kv_ring = 0;
        printf("KVRING %s\n", hybrid_kv_ring ? "on (4 sinks + a ring past the KV window)" : "off (the generation stops at the KV window)");
    } else if (strcmp(token, "softmaxunit") == 0) { // softmaxunit on|off: the attention softmax on unit 14 (on, the default where the unit exists) or on the CPU
        extern int hybrid_softmax_unit;
        if (strncmp(str, "on", 2) == 0)  hybrid_softmax_unit = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_softmax_unit = 0;
#ifdef CSR_TRIXIUM_SEQ_SOFTMAX_ABI_ADDR
        printf("SOFTMAXUNIT %s (unit 14 ABI %lu.%lu)\n", hybrid_softmax_unit ? "on" : "off",
               (unsigned long)(trixium_seq_softmax_abi_read() >> 16), (unsigned long)(trixium_seq_softmax_abi_read() & 0xFFFF));
#else
        printf("SOFTMAXUNIT %s (no unit 14 in this bitstream: the CPU softmax)\n", hybrid_softmax_unit ? "on" : "off");
#endif
    } else if (strcmp(token, "prefill2") == 0) { // prefill2 on|ffn|off: two prompt tokens per weight pass (the multi-row matvec)
        if (strncmp(str, "on", 2) == 0)  hybrid_prefill2_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_prefill2_enable(0);
        if (strncmp(str, "ffn", 3) == 0) hybrid_prefill2_enable(2);   // the FFN only (A/B)
#ifdef CSR_TRIXIUM_ENGINE_CSR_NR_ADDR
        printf("PREFILL2 %s\n", hybrid_prefill2_enabled() == 1 ? "on (two prompt tokens per weight pass: FFN, DeltaNet qkv/z, attention q/k/v)" :
                                hybrid_prefill2_enabled() == 2 ? "ffn (two prompt tokens per weight pass: the FFN only)" :
                                "off (one token per forward)");
#else
        printf("PREFILL2 off (no multi-row matvec in this bitstream)\n");
#endif
    } else if (strcmp(token, "fuseout") == 0) { // fuseout on|off: a mixer's or the FFN's output projection, the residual add and the next norm as one list (on) or as separate lists (off)
        extern int hybrid_fuse_out;
        if (strncmp(str, "on", 2) == 0)  hybrid_fuse_out = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_fuse_out = 0;
        printf("FUSEOUT %s\n", hybrid_fuse_out ? "on (out projection + residual + norm in one list)" : "off (separate lists)");
    } else if (strcmp(token, "lmstore") == 0) { // lmstore on|off: the LM head's logits stored to main RAM when the hardware argmax decides (off: the tracker alone; traced runs always store)
        extern int hybrid_lm_store;
        if (strncmp(str, "on", 2) == 0)  hybrid_lm_store = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_lm_store = 0;
        printf("LMSTORE %s\n", hybrid_lm_store ? "on (logits stored every chunk)" : "off (the argmax tracker alone)");
    } else if (strcmp(token, "dnzbeside") == 0) { // dnzbeside on|off: the DeltaNet z projection beside the conv's SiLU (on) or in the projection list
        extern int hybrid_dn_zbeside;
        if (strncmp(str, "on", 2) == 0)  hybrid_dn_zbeside = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_dn_zbeside = 0;
        printf("DNZBESIDE %s\n", hybrid_dn_zbeside ? "on (z streams beside the conv's SiLU)" : "off (z in the projection list)");
    } else if (strcmp(token, "dnlist") == 0) { // dnlist on|off: the DeltaNet q/k norm, heads op and gated norm as one list (on) or the heads op through the CSRs (off)
        extern int hybrid_dn_list;
        if (strncmp(str, "on", 2) == 0)  hybrid_dn_list = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_dn_list = 0;
        printf("DNLIST %s\n", hybrid_dn_list ? "on (one list: l2 norm, heads, gated norm)" : "off (the heads op through the CSRs)");
    } else if (strcmp(token, "hbstore") == 0) { // hbstore on|off: the residual list stores the normed hidden state to main RAM every call (off: in RS_F only, materialized for CPU readers)
        extern int hybrid_hb_store;
        if (strncmp(str, "on", 2) == 0)  hybrid_hb_store = 1;
        if (strncmp(str, "off", 3) == 0) hybrid_hb_store = 0;
        printf("HBSTORE %s\n", hybrid_hb_store ? "on (hb stored every residual list)" : "off (hb stays in RS_F; CPU readers restore it)");
    } else if (strcmp(token, "embres") == 0) { // embres unit|cpu: a rotated image's embedding rows without the int16 table, transformed on unit 3 or on the CPU
        extern int embed_restore_unit;
        if (strncmp(str, "unit", 4) == 0) embed_restore_unit = 1;
        if (strncmp(str, "cpu", 3) == 0)  embed_restore_unit = 0;
        printf("EMBRES %s\n", embed_restore_unit ? "unit (the rows' Hadamard transform on unit 3)" : "cpu (the transform on the CPU)");
        embed_restore_print();
    } else if (strcmp(token, "hwargmax") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_hwargmax_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_hwargmax_enable(0);
        printf("HWARGMAX %s (the LM head list's restore-unit argmax; off: the CPU loop over the logits)\n",
               hybrid_hwargmax_enabled() ? "on" : "off");
    } else if (strcmp(token, "attnstream") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_attnstream_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_attnstream_enable(0);
        printf("ATTNSTREAM %s (attention scores/values on engine op 6; takes effect at the next reset)\n",
               hybrid_attnstream_enabled() ? "on" : "off");
    } else if (strcmp(token, "dnstream") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_dnstream_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_dnstream_enable(0);
        printf("DNSTREAM %s (DeltaNet heads as one engine op per layer; takes effect at the next reset)\n",
               hybrid_dnstream_enabled() ? "on" : "off");
    } else if (strcmp(token, "dnres") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_dnres_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_dnres_enable(0);
        printf("DNRES %s (DeltaNet layers as scratchpad lists)\n", hybrid_dnres_enabled() ? "on" : "off");
    } else if (strcmp(token, "silu") == 0) {
        if (strncmp(str, "on", 2) == 0)  resident_silu_enable(1);
        if (strncmp(str, "off", 3) == 0) resident_silu_enable(0);
        printf("SILU %s (FFN activation through the unit 11 SiLU-multiply list)\n",
               resident_silu_active() ? "on" : "off");
    } else if (strcmp(token, "resident") == 0) {
        if (strncmp(str, "on", 2) == 0)  resident_enable(1);
        if (strncmp(str, "off", 3) == 0) resident_enable(0);
        printf("RESIDENT %s (27B projections through scratchpad quantize/rotate/restore "
               "lists; hardware %s)\n", resident_active() ? "on" : "off",
               resident_available() ? "present" : "absent");
    } else if (strcmp(token, "engdiag") == 0) {
        if (strncmp(str, "on", 2) == 0)  engine_diag = 1;
        if (strncmp(str, "off", 3) == 0) engine_diag = 0;
        printf("ENGDIAG %s (per-op stall snapshots, ~2k cycles/op)\n",
               engine_diag ? "on" : "off");
    } else if (strcmp(token, "hdrload") == 0)
        hdrload();
    else if (strcmp(token, "tokload") == 0)
        tok_load();
    else if (strcmp(token, "chat") == 0)
        chat_cmd(str);
#ifdef CSR_SD_BASE
    else if (strcmp(token, "sdinit") == 0) {
        char *m = get_token(&str);
        sd_init(*m ? parse_int(m) : 50, 1000);
    } else if (strcmp(token, "sdload") == 0) {
        char *c = get_token(&str), *b = get_token(&str), *sc = get_token(&str), *n = get_token(&str);
        int ch = (int)strtoul(c, NULL, 0);
        uint32_t base = strtoul(b, NULL, 0);
        if (sd_load(ch, base, strtoul(sc, NULL, 0), strtoul(n, NULL, 0)) == 0) {
            engine_fetch(ch, base, 1, 0xC000);       // What landed: the first beat.
            printf("SDLOAD ok:");
            for (int w = 0; w < 8; w++)
                printf(" %08lx", (unsigned long)((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[w]);
            printf("\n");
        }
    } else if (strcmp(token, "sdwtest") == 0) {   // sdwtest <sector>: write, read back, compare.
        uint32_t sec = strtoul(get_token(&str), NULL, 0), w[128];
        for (int i = 0; i < 128; i++) w[i] = 0xA5000000u ^ (sec << 8) ^ (uint32_t)i;
        if (sd_write(sec, 1, 0, 0, w) == 0 &&
            sd_load(0, TRIXIUM_CHAN_WEIGHT_BASE - 0x200000, sec, 1) == 0) {
            engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE - 0x200000, 16, 0xC000);
            int bad = 0;
            for (int i = 0; i < 128; i++)
                bad += ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i] != w[i];
            printf("SDWTEST %s (%d of 128 words differ)\n", bad ? "FAIL" : "ok", bad);
        }
    }
#if TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000
    else if (strcmp(token, "sdsave") == 0) {      // sdsave [slot] [label]
        char *a = get_token(&str), *label = get_token(&str);
        sd_save(*a ? (int)strtoul(a, NULL, 0) : 0, label);
    } else if (strcmp(token, "sdls") == 0)
        sd_list();
    else if (strcmp(token, "sddefault") == 0) {   // sddefault slot: the one the power-up boots
        char *a = get_token(&str);
        sd_set_default(*a ? (int)strtoul(a, NULL, 0) : 0);
    } else if (strcmp(token, "sdboot") == 0) {     // sdboot [slot] (default: the boot slot)
        char *a = get_token(&str);
        int slot = *a ? (int)strtoul(a, NULL, 0) : -1;
        if (slot < 0)
            slot = (sd_init_done() || sd_init(50, 1000) == 0) ? sd_default_slot() : 0;
        if (sd_boot(slot) == 0) {
            hdrload();
            engine_fetch(0, TOK_STAGING, 1, 0xC000);   // A tokenizer on the card: load it too.
            if (*(volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000) == TOK_STAGING_MAGIC)
                tok_load();
        }
    }
#endif
#endif
    else if (strlen(token))
        puts("unknown command");
    printf("trixium> ");
}

/* Main */
/*------*/

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
    irq_setmask(0);
    irq_setie(1);
#endif
    uart_init();
    SIM_MARK("main");                    // Reset to here: crt0 (no BIOS in the simulation).
#ifdef TRIXIUM_DDR_ONLY
    // Board bring-up: the memory controllers alone, before any unit is touched (no engine or
    // vector-unit init; ddrtest drives the engine's STORE/FETCH only).
    puts("\nTriXium DDR bring-up "__DATE__" "__TIME__"\n");
#ifdef TRIXIUM_N_MEMCTRL
    ddr_boot_init();
    ddr_test(-1);
#endif
    puts("ddr bring-up: done");
    while (1)
        console_service();
#endif
    stack_guard_paint();
    perf_init();
    infer_static_init();
    vec_init();

    puts("\nTriXium firmware "__DATE__" "__TIME__"\n");
#ifdef TRIXIUM_N_MEMCTRL
#ifndef TRIXIUM_SIM
    // The BIOS brought up controller 0 (main_ram); the other controllers carry the engine
    // channels' weights: init them here unless the BIOS did (the YPCB's CK phase leveling),
    // then prove every channel's window with the path the weights take (ddr.c).
    ddr_boot_init();
    ddr_test(-1);
#else
    // Simulation: LiteDRAM's PHY model has no reset/MRS/ZQ/leveling state and the DFII
    // starts in hardware control, so the init sequence (~1.6M cycles, ~80 s of Verilator)
    // has nothing to set up. The round trip (~0.5M cycles) is covered by the gate itself:
    // every channel's weights are fetched and its K/V and DeltaNet state stored through
    // the same windows, so an aliased or broken channel changes the checkpoints. Both
    // stay on the console (`ddrinit`, `ddrtest`).
    puts("ddr: simulation (PHY model) - controller init and DDRTEST skipped");
#endif
#endif
#if defined(CSR_SD_BASE) && TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000 && !defined(TRIXIUM_SIM)
    // A card answers: its image is the model, a host-free start. Whatever the DRAM still holds
    // is from before this boot: a bitstream reload (and a short power cut) keeps the header
    // but not all the weights (after a reload: 3 of 470 tensors off their table CRC, and
    // `imgcheck` takes 100 s, the card 74). `s` within a second keeps the staged model (the
    // host's serial boot answers it: it uploads right after). Three quick probes without a card.
    // Not in the simulation (a second is 100M cycles there): `sdinit`/`sdboot` on its console.
    if (sd_init(50, 3) == 0) {
        puts("SDBOOT in 1 s (s to skip: keep the staged model)");
        int skip = 0;
        uint64_t t_wait = perf_now64();
        while (!skip && perf_now64() - t_wait < CONFIG_CLOCK_FREQUENCY)
            if (readchar_nonblock())
                skip = getchar() == 's';
        if (!skip && sd_boot(sd_default_slot()) == 0 && hdrload_staged(1) >= 0) {
            engine_fetch(0, TOK_STAGING, 1, 0xC000);
            if (*(volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000) == TOK_STAGING_MAGIC)
                tok_load();
        }
    }
#endif
    hdrload_staged(1);   // A different image than main_ram's header says: take the staged one.
    model_info();
    sptest();
#ifndef CSR_TRIXIUM_VECUNIT_PROG_ADDR
    puts("VECTEST skipped (no vector unit in this bitstream)");
#else
    { // Vec unit smoke test: quant8 of a known ramp via the CSR dispatch path. Catches
      // emission-level datapath breaks that Migen-sim unit tests cannot see.
        volatile int16_t *pi = (volatile int16_t *)(SCRATCHPAD_BASE + 0x100);
        volatile int8_t  *po = (volatile int8_t  *)(SCRATCHPAD_BASE + 0x200);
        // TWO beats: quant8 is x2-unrolled (asm.py) and processes beats in pairs, so a
        // 1-beat call would also read 0x120-0x13F - whatever a previous session left there.
        for (int i = 0; i < 32; i++) pi[i] = (int16_t)((i & 15)*1000 - 8000);
        for (int i = 0; i < 32; i++) po[i] = 0x5A;
        int32_t va[4] = {0x100, 1, 0x200, 2}; // A1 = 1: measured shift.
        uint64_t amax = vec_run(PROG_QUANT8, va, 4); // S6 = source absmax (pass 1).
        // quant8 picks its shift from that absmax (see prog_quant8), so the expectation has
        // to track it - this also puts VCLZN and REDMAX under the smoke test.
        int bl = 0;
        while ((amax >> bl) != 0) bl++;
        int sh = (amax == 0) ? 8 : bl - 7;
        int ok = (amax == 8000);
        for (int i = 0; i < 32; i++) {
            int32_t v = (int32_t)((i & 15)*1000 - 8000) << (8 - sh);
            int32_t q = (v >> 8) + (((v >> 7) & 1) & (((v & 0x7F) != 0) | ((v >> 8) & 1)));
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            if (po[i] != (int8_t)q) ok = 0;
        }
        printf("VECTEST %s\n", ok ? "OK" : "FAIL");
        // Per-program CRC sweep: run every program on a seeded pattern; the sim harness
        // compares each printed CRC against the interpreter (named failure per program).
        {
            volatile uint32_t *sp32b = (volatile uint32_t *)(SCRATCHPAD_BASE + 0x8000);
            uint32_t seed = 0x12345678u;
            for (int i = 0; i < 1024; i++) { // 4 KB of LCG pattern at 0x8000.
                seed = seed*1664525u + 1013904223u;
                sp32b[i] = seed;
            }
            // `dst` is explicit because it is not always A2: rope takes A2 as its *cos
            // table* and writes via A5/A6; clearing A2 would corrupt that table (and reduce's
            // A1 source, the same region).
            struct { const char *name; uint16_t pc; int32_t a[8]; int na; uint32_t dst; }
            progs[] = {
                {"quant8",   PROG_QUANT8,   {0x8000, 1, 0xA000, 8}, 4, 0xA000},  // A1 = 1.
                {"quant8f",  PROG_QUANT8,   {0x8000, 0, 0xB200, 8}, 4, 0xB200},  // A1 = 0.
                {"norm_i32", PROG_NORM_I32, {0x8000, 0, 0xA200, 4, 13}, 5, 0xA200},
                {"resid",    PROG_RESID_ADD,{0x8000, 0x8400, 0xA400, 8, 1, 2}, 6, 0xA400},
                {"silu",     PROG_SILU_MUL, {0x8000, 0x8400, 0xA600, 8, -6, 0, -15}, 7, 0xA600},
                {"rmsnorm",  PROG_RMSNORM,  {0x8000, 0x8400, 0xA800, 8, -7, 17, -17}, 7, 0xA800},
                {"rope",     PROG_ROPE,     {0x8000, 0x8080, 0x8400, 0x8480, 4,
                                             0xAA00, 0xAA80}, 7, 0xAA00},
                {"softmax8", PROG_SOFTMAX8, {0x8000, 739, 0xAC00, 4, -11, 17, 0xB400}, 7, 0xAC00},
                {"reduce",   PROG_REDUCE_ACC,{0xA200, 0x8400, 0xAE00, 4}, 4, 0xAE00},
                {"scale8",   PROG_SCALE8,   {0x8000, 0, 0xB000, 8, 19571}, 5, 0xB000},
            };
            for (unsigned pi2 = 0; pi2 < sizeof(progs)/sizeof(progs[0]); pi2++) {
                volatile uint32_t *dst = (volatile uint32_t *)(SCRATCHPAD_BASE
                                          + progs[pi2].dst);
                // Clear the CRC window first. Programs write less than 128 words, so
                // otherwise the checksum folds in whatever was left in the scratchpad
                // (history-dependent on a card, incomparable to the interpreter).
                for (int i = 0; i < 128; i++)
                    dst[i] = 0;
                uint64_t r6 = vec_run(progs[pi2].pc, progs[pi2].a, progs[pi2].na);
                uint32_t crc = 0;
                for (int i = 0; i < 128; i++)
                    crc = (crc ^ dst[i])*16777619u;
                printf("VECCRC %s %08lx %08lx\n", progs[pi2].name,
                       (unsigned long)crc, (unsigned long)(uint32_t)r6);
            }
        }
    }
#endif // CSR_TRIXIUM_VECUNIT_PROG_ADDR
    selftest();
    SIM_MARK("selftests");
    hwops_init();
    if (infer_init() == 0) {
        // Boot generation demo (deterministic; checked by the sim harness).
        static const int prompt[3] = {1, 2, 3};
        // The legs that only exist for the gate are skipped on big models (the all-CPU
        // reference alone takes seconds on the 1.7B); the tiny sim model runs every leg.
        // GEN-HW stays: two tokens through the hardware path are cheap and prove the units
        // are alive.
        const int big_model = model_header()->n_layer > 8;
        if (!big_model) {
            // Harness hook for the simulation tools that need only the console (the legs
            // cost tens of minutes of Verilator time). Offered here, after the firmware is
            // running: an early key would reach the LiteX BIOS, where Q aborts the boot.
            // With no key within 2M cycles, or any other key, the legs run.
            puts("BOOT-LEGS (q to skip)");
            uint64_t t_wait = perf_now64();
            while (perf_now64() - t_wait < 2000000u) {
                if (readchar_nonblock()) {
                    char c = getchar();
                    if (c == 'q' || c == 'Q') {
                        puts("boot legs skipped");
                        goto boot_demo_done;
                    }
                    break;
                }
            }
        }
        if (big_model) {
            puts("GEN skipped at boot on this model size (`hwops all cpu` + `gen` runs the CPU reference)");
        } else {
            // The GEN leg is the all-CPU reference the gate compares against
            // hybrid_generate's CPU modes, so pin it here rather than inheriting
            // whatever the default mask is.
            hwops_set_all(0);
            printf("GEN:");
            perf_reset();
            infer_generate(prompt, 3, 2, emit_token);
            puts("\nGEN DONE");
            perf_print();
        }
        if (hw_caps()) {
            hwops_set_all(1);
            printf("GEN-HW:");
            perf_reset();
            infer_generate(prompt, 3, 2, emit_token);
            puts("\nGEN-HW DONE");
            perf_print();
        }
    }
boot_demo_done:
    hwops_init(); // Leave the shell on the fast path, not on the debug fallbacks.
    SIM_MARK("ready");
#ifdef TRIXIUM_WEB_IP
    // The chat page on the CPU's own MAC (net.c, web.c), served between console commands.
    {
        uint8_t mac[6] = {TRIXIUM_WEB_MAC_HI >> 8, TRIXIUM_WEB_MAC_HI & 0xff, TRIXIUM_WEB_MAC_LO >> 24,
                          (TRIXIUM_WEB_MAC_LO >> 16) & 0xff, (TRIXIUM_WEB_MAC_LO >> 8) & 0xff, TRIXIUM_WEB_MAC_LO & 0xff};
        net_platform_init();
        web_init(mac, (uint32_t)TRIXIUM_WEB_IP);
        uint32_t ip = TRIXIUM_WEB_IP;
        printf("WEB DHCP, else http://%d.%d.%d.%d/ in 4 s; http://trixium.local/\n", (int)(ip >> 24),
               (int)((ip >> 16) & 0xff), (int)((ip >> 8) & 0xff), (int)(ip & 0xff));
    }
#endif
    puts("TRIXIUM-READY");
    printf("trixium> ");

    while (1) {
        console_service();
        stack_guard_check();
#ifdef TRIXIUM_WEB_IP
        web_service();
#endif
    }
    return 0;
}

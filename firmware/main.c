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

/* Checks */
/*--------*/

#ifdef TRIXIUM_N_MEMCTRL
// ddr.c reaches controller k's registers as controller 0's shifted by k*STRIDE: the target
// pins the (sdram, ddrphy) CSR pairs to consecutive pages, and this is where that is checked.
// (The PHY model of the Verilator sim has no CSRs, hence the ddrphy guards.)
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
    puts("  sample T P R [S]  - temperature/top-p/repeat-penalty x1000 (0 1000 1000 = greedy), seed");
    puts("  bench [n]         - profiled generation benchmark (default 4 tokens)");
    puts("Hardware vs CPU reference (default: the hardware path where the bitstream has it):");
    puts("  hwops [op cpu|hw] - show/set per-op CPU/HW dispatch (op name or 'all')");
    puts("  seq resident silu dnres dnstream attnstream hwargmax [on|off]");
    puts("                    - unit or list path vs its CPU reference (prints the state)");
    puts("  lmchunk rows      - LM head list chunk cap in rows (0 = automatic)");
    puts("  attnchunk n       - cap the heads per attention list");
    puts("Diagnostics:");
    puts("  stats [reset]     - per-op cycle profile");
    puts("  fprof on|off|print, tprof arm [n]|dump, engdiag on|off");
    puts("                    - FFN windows, token timeline, per-op engine stall snapshots");
    puts("  dbg [n]           - checksum the next n forwards' intermediates (vs the Python mirror)");
    puts("  selftest          - engine self-tests (matvec + fetch checksums)");
    puts("  sptest            - scratchpad bus read/write test");
    puts("  peek c addr       - dump one weight-memory beat of channel c via the engine (hex addr)");
    puts("  mvtest role [layer] - matvec one tensor with the test pattern, print checksum");
    puts("  mvbw role [layer] [reps] - engine streaming efficiency on one tensor");
#ifdef TRIXIUM_N_MEMCTRL
    puts("  ddrinit [k]       - re-init DDR4 controller k (default: 1..N-1, as at boot)");
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
    static const int prompt[3] = {1, 2, 3};
    perf_reset();
    uint32_t f0 = infer_forwards;
    uint64_t t0 = perf_now64();
    memset(&bench_samples, 0, sizeof(bench_samples));
    bench_samples.start = t0;
    printf("GEN:");
    int count = infer_generate(prompt, 3, n, bench_emit_token);
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
        && mh->n_layer == sh->n_layer && mh->table_crc32 == sh->table_crc32)
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
    model_cache_reset(); // Header changed: rebuild the (role, layer) lookup cache.
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
    if (e->kind != KIND_TERNARY && e->kind != KIND_Q8) {
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
    if (!e || (e->kind != KIND_TERNARY && e->kind != KIND_Q8) || e->rows*4 > 0x30000) {
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
    printf("MVBW role=%u layer=%u rows=%lu cols=%lu kind=%u: %lu cycles/matvec, %lu beats/channel, "
           "%lu.%02lu beats/channel/clock (%lu%% of 1.0)\n", e->role, e->layer,
           (unsigned long)e->rows, (unsigned long)e->cols, e->kind, (unsigned long)dt,
           (unsigned long)beats, (unsigned long)(beats/dt), (unsigned long)(beats*100/dt % 100),
           (unsigned long)(beats*100/dt));
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
    printf("GEN:");
    gen_emit_to_pbuf = 1;
    infer_set_stop(stop, nstop);
    infer_set_recent(pbuf, &pbuf_n);      // the repetition penalty's window
    // Continue from the KV cache when the buffer's head is still what the last genb left
    // there (gen/bench in between reset it); otherwise the whole buffer is re-forwarded.
    int resident = (pbuf_serial == infer_gen_serial()) ? pbuf_kv : 0;
    int r = infer_generate_from(pbuf, pbuf_n, resident, n, emit_token);
    pbuf_kv     = infer_kv_resident();
    pbuf_serial = infer_gen_serial();
    infer_set_stop(stop, 0);
    gen_emit_to_pbuf = 0;
    puts(r < 0 ? "\nGEN ERR" : "\nGEN DONE");
    if (engine_wedged) {          // Recover here, after the stream: the client reads GEN ERR.
        engine_recover();
        pbuf_kv = 0;              // The KV cache is not trustworthy: the next genb re-forwards.
    }
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
    else if (strcmp(token, "mvtest") == 0)
        mvtest_cmd(str);
    else if (strcmp(token, "mvbw") == 0)
        mvbw_cmd(str);
    else if (strcmp(token, "selftest") == 0)
        selftest();
    else if (strcmp(token, "gen") == 0)
        gen_cmd(str);
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
    } else if (strcmp(token, "tprof") == 0) {
        if (strncmp(str, "arm", 3) == 0) {
            const char *p = str + 3;
            while (*p == ' ') p++;
            int n = parse_int(p);
            tprof_armed = n > 0 ? n : 1;
            printf("TPROF armed: generated token #%d traces itself (gen or bench)\n",
                   tprof_armed);
        }
        if (strncmp(str, "dump", 4) == 0) tprof_dump();
        if (str[0] == 0)
            printf("TPROF %s (arm | dump)\n",
                   tprof_armed ? "armed" : tprof_active ? "tracing" : "idle");
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
    perf_init();
    infer_static_init();
    vec_init();

    puts("\nTriXium firmware "__DATE__" "__TIME__"\n");
#ifdef TRIXIUM_N_MEMCTRL
#ifndef TRIXIUM_SIM
    // The BIOS brought up controller 0 (main_ram); the other DDR4 controllers carry
    // the engine channels' weights and have no BIOS support: init them here, then prove
    // every channel's window with the path the weights take (ddr.c).
    ddr_init_all();
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
    hdrload_staged(1);   // A different image than main_ram's header says: take the staged one.
    model_info();
    sptest();
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
    puts("TRIXIUM-READY");
    printf("trixium> ");

    while (1) {
        console_service();
    }
    return 0;
}

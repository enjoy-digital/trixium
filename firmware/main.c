// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// TriXium firmware: console shell + engine self-tests (token loop lands with the
// vector unit). At boot, runs the self-test sequence (deterministic checksums that the
// simulation harness verifies), then drops to the shell.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <irq.h>
#include <libbase/uart.h>
#include <libbase/console.h>
#include <generated/csr.h>
#include <generated/soc.h>

#include "bons.h"
#include "ddr.h"

#ifdef BONSAI_N_MEMCTRL
// ddr.c reaches controller k's registers as controller 0's shifted by k*STRIDE: the target
// pins the (sdram, ddrphy) CSR pairs to consecutive pages, and this is where that is checked.
// (The PHY model of the Verilator sim has no CSRs, hence the ddrphy guards.)
_Static_assert(CSR_SDRAM1_BASE - CSR_SDRAM_BASE == BONSAI_DDR_CSR_STRIDE, "sdram1 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY1_BASE - CSR_DDRPHY_BASE == BONSAI_DDR_CSR_STRIDE, "ddrphy1 CSR page");
#endif
#if BONSAI_N_MEMCTRL > 2
_Static_assert(CSR_SDRAM2_BASE - CSR_SDRAM_BASE == 2*BONSAI_DDR_CSR_STRIDE, "sdram2 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY2_BASE - CSR_DDRPHY_BASE == 2*BONSAI_DDR_CSR_STRIDE, "ddrphy2 CSR page");
#endif
#endif
#if BONSAI_N_MEMCTRL > 3
_Static_assert(CSR_SDRAM3_BASE - CSR_SDRAM_BASE == 3*BONSAI_DDR_CSR_STRIDE, "sdram3 CSR page");
#ifdef CSR_DDRPHY_BASE
_Static_assert(CSR_DDRPHY3_BASE - CSR_DDRPHY_BASE == 3*BONSAI_DDR_CSR_STRIDE, "ddrphy3 CSR page");
#endif
#endif
#endif
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

/*-----------------------------------------------------------------------*/
/* Console                                                               */
/*-----------------------------------------------------------------------*/

static char *readstr(void)
{
    char c[2];
    char *const s = (char *)BONSAI_CONSOLE_LINE_ADDR;
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
            if (ptr >= BONSAI_CONSOLE_LINE_SIZE - 1)
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

/*-----------------------------------------------------------------------*/
/* Helpers                                                               */
/*-----------------------------------------------------------------------*/

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

/*-----------------------------------------------------------------------*/
/* Commands                                                              */
/*-----------------------------------------------------------------------*/

static void emit_token(int tok);
static int  gen_emit_to_pbuf;
static int  parse_int(const char *s);

static void help(void)
{
    puts("TriXium firmware");
    puts("help            - this message");
    puts("info            - model info (BONS header)");
    puts("hdrload         - copy BONS header+table staged by bonsai_load to CPU memory");
    puts("sptest          - scratchpad bus read/write test");
#ifdef BONSAI_N_MEMCTRL
    puts("ddrinit [k]     - re-init DDR4 controller k (default: 1..N-1, as at boot)");
    puts("ddrtest [c]     - engine store/fetch round trip through channel c's window (all)");
#endif
    puts("peek c addr     - dump one HBM beat via the engine (hex addr)");
    puts("mvtest idx      - matvec tensor <idx> with the test pattern, print checksum");
    puts("selftest        - engine self-tests (matvec + fetch checksums)");
    puts("spec [on|off]   - speculative decode toggle + acc/rej counters");
    puts("seq [on|off]    - sequencer attention lists (on by default; off: CPU-dispatched)");
    puts("tokclr/tok/genb - chunked prompt buffer + generate [stop ids] (host chat tools)");
    puts("gen n id id ... - generate n tokens (greedy) from prompt ids");
    puts("hwops [op cpu|hw] - show/set per-op CPU/HW dispatch (op name or 'all')");
    puts("stats [reset]   - per-op cycle profile");
    puts("bench [n]       - profiled generation benchmark (default 4 tokens)");
    puts("sample T P R [S] - temperature/top-p/repeat-penalty x1000 (0 1000 1000 = greedy), seed");
    puts("engreset        - soft reset of the engine cluster (automatic after a stall)");
    puts("wedge           - stall the engine on purpose, recover, rerun the self-test");
    puts("dbg [n]         - checksum the next n forwards' intermediates (vs the Python mirror)");
    puts("vbench          - stage-by-stage cycle microbenchmark");
    puts("Experiment toggles (all `name on|off`, each printing its state; defaults ship):");
    puts("  ffnseq fsplit     - FFN as one sequencer list; SiLU split under the up stream");
    puts("  qkvl qkvx qkro    - QKV projections as a list; K chain inside it; Q chain last");
    puts("  qfix kvms kvq     - fixed-shift Q quant8; one KV multi-store; redundant KV quant8");
    puts("  aovl kovl povl lovl eovl ovl fovl - overlap windows (attention, KV, patches,");
    puts("                      LM head, early fold, op-by-op norms, norm-row prefetch)");
    puts("  qres ifold embf sbdrain - resident qk-norm rows; integer folds; int16 embedding");
    puts("                      table; sideband drain-at-start (needs the sbmode CSR)");
    puts("  fprof tprof atrace engdiag - profiling: FFN windows, token timeline, attention");
    puts("                      list trace (atrace N arms), engine diagnostics");
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
    // Per FORWARD: the comparable figure across the 2026-09-07 change that stopped
    // running the unused final forward (bench 64 = 3 prompt + 63 = 66 forwards, was 67).
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

static int gen_emit_to_pbuf;
// Chunked prompt buffer (tok/genb) - main_ram static window, not .bss (sram is tight;
// see firmware/linker.ld). BONSAI_HDR_ADDR + 0xA0000, clear of the entry cache (+0x80000)
// and ek_undo (+0x90000); the heap starts at +0x100000. Sized to the KV budget (the image's
// max_ctx, 2048 for the 1.7B): a conversation can use the whole context.
#define PBUF_N 2048
static int *const pbuf = (int *)(BONSAI_HDR_ADDR + 0xA0000); // [PBUF_N]
static int pbuf_n;
static int      pbuf_kv;      // pbuf[0..pbuf_kv) are in the KV cache (from the last genb)...
static uint32_t pbuf_serial;  // ...as long as no other generation ran since.

static void emit_token(int tok)
{
    printf(" %d", tok);
    if (gen_emit_to_pbuf && pbuf_n < PBUF_N)
        pbuf[pbuf_n++] = tok;
}

// Copy the BONS header+table from the HBM staging area (engine channel 0, uploaded there by
// bonsai_load) into the CPU-visible header location, then re-init inference.
#define HDR_STAGING (BONSAI_CHAN_WEIGHT_BASE - 0x100000)
// The staged copy (channel 0, uploaded with every image by bonsai_load) is the truth about
// what is in HBM; the main_ram copy is derived from it. The boot used to trust main_ram: a
// CPU reset after `bonsai_load` of a DIFFERENT model then ran the boot demo with the
// previous model's tables - the 1.7B's KV addresses, which on the 8B point into layer 18's
// weights - and the two demo tokens overwrote them (2026-09-18: mvtest of blk.18.attn_v
// wrong until re-uploaded, the corruption starting exactly at the 1.7B's kv_base). Now the
// boot refreshes main_ram from the staged header when the two differ.
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
        // Staging cleared by the host (bonsai_chat -C writes this marker before a board
        // boot): a header left in main_ram by an earlier session describes weights that are
        // not there - the boot demo then streamed them from controllers not yet trained and
        // the engine stalled (v22 restore, 2026-10-02). The simulation stages nothing and
        // keeps the header it preloaded into main_ram.
        struct bons_header *mh = (struct bons_header *)BONSAI_HDR_ADDR;
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
    uint8_t *dst   = (uint8_t *)BONSAI_HDR_ADDR;
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
// checksum. selftest only ever covered blk.0.attn_q; the 1.7B model has several other shapes
// - cols 6144 for ffn_down, rows 1024 for attn_k/v, the tied LM head - and none of them had
// been checked on silicon, while the sim token gate runs the tiny model's small dimensions.
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
    int *const prompt = (int *)BONSAI_CONSOLE_PROMPT_ADDR;
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
    engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32, 8, 0xC000);
    csum = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + 0xC000), 8*32);
    printf("FETCH CSUM=%08lx\n", (unsigned long)csum);

    puts("SELFTEST DONE");
}

/*-----------------------------------------------------------------------*/
/* Main                                                                  */
/*-----------------------------------------------------------------------*/

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
    }
    else if (strcmp(token, "attnchunk") == 0)
        infer_attn_chunk(parse_int(str));
    else if (strcmp(token, "engreset") == 0)
        engine_recover();
#ifdef BONSAI_N_MEMCTRL
    else if (strcmp(token, "ddrinit") == 0) {
        char *arg = get_token(&str);
        if (arg && *arg) ddr_init(parse_int(arg)); else ddr_init_all();
    }
    else if (strcmp(token, "ddrtest") == 0) {
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
    }
    else if (strcmp(token, "dbg") == 0) {
        int n = (*str) ? parse_int(str) : 1;
        infer_dbg(n);
        printf("DBG armed for %d forwards\n", n);
    }
    else if (strcmp(token, "dbgdump") == 0) { // dbgdump <tag> <layer> <pos>
        char *tag = get_token(&str);
        char *ls  = get_token(&str);
        char *ps  = get_token(&str);
        infer_dbg_dump(tag, parse_int(ls), parse_int(ps));
        printf("DBG dump %s l=%d pos=%d\n", tag, parse_int(ls), parse_int(ps));
    }
    else if (strcmp(token, "stats") == 0) {
        if (strncmp(str, "reset", 5) == 0)
            perf_reset();
        else
            perf_print();
    }
    else if (strcmp(token, "bench") == 0)
        bench_cmd(str);
    else if (strcmp(token, "spec") == 0) {
        if (strncmp(str, "on", 2) == 0)
            infer_spec(1);
        else if (strncmp(str, "off", 3) == 0)
            infer_spec(0);
        int acc, rej;
        infer_spec_stats(&acc, &rej);
        printf("SPEC acc=%d rej=%d\n", acc, rej);
    }
    else if (strcmp(token, "ffnseq") == 0) {
        extern int seq_ffn_use;
        if (strncmp(str, "on", 2) == 0)  seq_ffn_use = 1;
        if (strncmp(str, "off", 3) == 0) seq_ffn_use = 0;
        printf("FFNSEQ %s (whole-layer FFN list; default on since C2-1c)\n",
               seq_ffn_use ? "on" : "off");
    }
    else if (strcmp(token, "povl") == 0) {
        extern int prep_ovl;
        if (strncmp(str, "on", 2) == 0)  prep_ovl = 1;
        if (strncmp(str, "off", 3) == 0) prep_ovl = 0;
        printf("POVL %s (attention patches under the qkr list)\n", prep_ovl ? "on" : "off");
    }
    else if (strcmp(token, "eovl") == 0) {
        extern int eovl_use;
        if (strncmp(str, "on", 2) == 0)  eovl_use = 1;
        if (strncmp(str, "off", 3) == 0) eovl_use = 0;
        printf("EOVL %s (fold heads under the running attention list)\n",
               eovl_use ? "on" : "off");
    }
    else if (strcmp(token, "qres") == 0) {
        extern int qres_use;
        if (strncmp(str, "on", 2) == 0)  qres_use = 1;
        if (strncmp(str, "off", 3) == 0) qres_use = 0;
        printf("QRES %s (resident qk-norm rows)\n", qres_use ? "on" : "off");
    }
    else if (strcmp(token, "fovl") == 0) {
        extern int fovl_use;
        if (strncmp(str, "on", 2) == 0)  fovl_use = 1;
        if (strncmp(str, "off", 3) == 0) fovl_use = 0;
        printf("FOVL %s (norm-row fetches under resid_add / KV prep)\n", fovl_use ? "on" : "off");
    }
    else if (strcmp(token, "qkvl") == 0) {
        extern int qkvl_use;
        if (strncmp(str, "on", 2) == 0)  infer_qkvl(1);
        if (strncmp(str, "off", 3) == 0) infer_qkvl(0);
        printf("QKVL %s (q/k/v projections + norms as one sequencer list)\n",
               qkvl_use ? "on" : "off");
    }
#ifdef CSR_SEQ_BASE
    else if (strcmp(token, "sbdrain") == 0) {
        if (!seq_sbmode_available())
            puts("SBDRAIN unsupported on this bitstream (no sbmode CSR)");
        else {
            if (strncmp(str, "on", 2) == 0)  seq_sbmode_set(1);
            if (strncmp(str, "off", 3) == 0) seq_sbmode_set(0);
            printf("SBDRAIN %s (%s)\n", seq_sbdrain_use ? "on" : "off",
                   seq_sbdrain_use ? "posted patches drained at list start only"
                                   : "legacy: descriptor fetch stalls on any posted write");
        }
    }
#endif
    else if (strcmp(token, "lovl") == 0) {
        extern int lovl_use;
        if (strncmp(str, "on", 2) == 0)  lovl_use = 1;
        if (strncmp(str, "off", 3) == 0) lovl_use = 0;
        printf("LOVL %s (next-token list prep under the LM-head stream)\n",
               lovl_use ? "on" : "off");
    }
    else if (strcmp(token, "fprof") == 0) {
        extern int fprof_use; extern void fprof_reset(void), fprof_print(void);
        if (strncmp(str, "on", 2) == 0)  { fprof_use = 1; fprof_reset(); }
        if (strncmp(str, "off", 3) == 0) fprof_use = 0;
        if (strncmp(str, "print", 5) == 0) fprof_print();
        printf("FPROF %s\n", fprof_use ? "on" : "off");
    }
    else if (strcmp(token, "tprof") == 0) {
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
    }
    else if (strcmp(token, "atrace") == 0) {
        extern int atrace_use;
        int n = parse_int(str);
        atrace_use = n > 0 ? n : 1;
        printf("ATRACE armed: layer 0 of generated token #%d traces per-descriptor\n",
               atrace_use);
    }
    else if (strcmp(token, "kovl") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_kv_pipe(1);
        if (strncmp(str, "off", 3) == 0) infer_kv_pipe(0);
        printf("KOVL %s (in-list NOWAIT KV stores under the pack quant8s)\n",
               infer_kv_pipe_state() ? "on" : "off");
    }
    else if (strcmp(token, "fsplit") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_ffn_split(1);
        if (strncmp(str, "off", 3) == 0) infer_ffn_split(0);
        printf("FSPLIT %s (gate-only SiLU under the up stream; mul+quant8 after)\n",
               infer_ffn_split_state() ? "on" : "off");
    }
    else if (strcmp(token, "embf") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_embfetch(1);
        if (strncmp(str, "off", 3) == 0) infer_embfetch(0);
        printf("EMBF %s (embedding row fetched pre-dequantized)\n",
               infer_embfetch_state() ? "on" : "off");
    }
    else if (strcmp(token, "qkvx") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_qkvx(1);
        if (strncmp(str, "off", 3) == 0) infer_qkvx(0);
        printf("QKVX %s (K chain and V pack inside the qkv list, eps via LUT)\n",
               infer_qkvx_state() ? "on" : "off");
    }
    else if (strcmp(token, "qfix") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_qfix(1);
        if (strncmp(str, "off", 3) == 0) infer_qfix(0);
        printf("QFIX %s (scan-free fixed-shift Q quant8)\n", infer_qfix_state() ? "on" : "off");
    }
    else if (strcmp(token, "qkro") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_qkr_reorder(1);
        if (strncmp(str, "off", 3) == 0) infer_qkr_reorder(0);
        printf("QKRO %s (qkr list: Q chain behind the KV store)\n",
               infer_qkr_reorder_state() ? "on" : "off");
    }
    else if (strcmp(token, "kvms") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_kv_mstore(1);
        if (strncmp(str, "off", 3) == 0) infer_kv_mstore(0);
        printf("KVMS %s (one in-list multi-store for all KV slots vs 16 stores)\n",
               infer_kv_mstore_state() ? "on" : "off");
    }
    else if (strcmp(token, "kvq") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_kvq_redundant(1);
        if (strncmp(str, "off", 3) == 0) infer_kvq_redundant(0);
        printf("KVQ %s (per-slot KV-pack quant8s; kvpack_h already does them)\n",
               infer_kvq_redundant_state() ? "on" : "off");
    }
    else if (strcmp(token, "aovl") == 0) {
        if (strncmp(str, "on", 2) == 0)  infer_attn_pipe(1);
        if (strncmp(str, "off", 3) == 0) infer_attn_pipe(0);
        printf("AOVL %s (pipelined attention list: next head's scores stream under softmax)\n",
               infer_attn_pipe_state() ? "on" : "off");
    }
    else if (strcmp(token, "ifold") == 0) {
        extern int ifold_use;
        if (strncmp(str, "on", 2) == 0)  ifold_use = 1;
        if (strncmp(str, "off", 3) == 0) ifold_use = 0;
        printf("IFOLD %s (integer emulation of the per-head float fold)\n",
               ifold_use ? "on" : "off");
    }
    else if (strcmp(token, "ovl") == 0) {
        extern int ovl_use;
        if (strncmp(str, "on", 2) == 0)  ovl_use = 1;
        if (strncmp(str, "off", 3) == 0) ovl_use = 0;
        printf("OVL %s (engine/vec overlap - off serializes for A/B)\n",
               ovl_use ? "on" : "off");
    }
    else if (strcmp(token, "seq") == 0) {
        if (strncmp(str, "on", 2) == 0)
            infer_seq(1);
        else if (strncmp(str, "off", 3) == 0)
            infer_seq(0);
        printf("SEQ %s (off runs the same ops CPU-dispatched per head, ~20%% slower)\n",
               seq_use ? "on" : "off");
    }
    else if (strcmp(token, "dnpipe") == 0) {
        extern int dn_pipe_use;
        if (strncmp(str, "on", 2) == 0)  dn_pipe_use = 1;
        if (strncmp(str, "off", 3) == 0) dn_pipe_use = 0;
        printf("DNPIPE %s (DeltaNet head state store/fetch under the CPU's head work)\n",
               dn_pipe_use ? "on" : "off");
    }
    else if (strcmp(token, "xres") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_xres_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_xres_enable(0);
        printf("XRES %s (the residual stream and its norms resident)\n", hybrid_xres_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "lmcheck") == 0) {
        hybrid_lm_check();
    }
    else if (strcmp(token, "lmchunk") == 0) {
        uint32_t rows = (uint32_t)strtoul(str, NULL, 0);
        hybrid_lm_chunk(rows);
        printf("LMCHUNK %lu (LM head list chunk cap in rows, 0 = automatic)\n", (unsigned long)rows);
    }
    else if (strcmp(token, "prefetch") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_prefetch_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_prefetch_enable(0);
        printf("PREFETCH %s (run-ahead weight reads: recorded on the next forward with logits)\n",
               hybrid_prefetch_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "hwargmax") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_hwargmax_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_hwargmax_enable(0);
        printf("HWARGMAX %s (the LM head list's restore-unit argmax; off: the CPU loop over the logits)\n",
               hybrid_hwargmax_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "descbench") == 0) {
        resident_desc_bench();
    }
    else if (strcmp(token, "attnfail") == 0) {
        int p = (int)strtoul(str, NULL, 0);
        hybrid_attn_fail_at(p);
        printf("attnfail: the attention stream fails at position %d (0 = never)\n", p);
    }
    else if (strcmp(token, "attnstream") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_attnstream_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_attnstream_enable(0);
        printf("ATTNSTREAM %s (attention scores/values on engine op 6; takes effect at the next reset)\n",
               hybrid_attnstream_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "dnstream") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_dnstream_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_dnstream_enable(0);
        printf("DNSTREAM %s (DeltaNet heads as one engine op per layer; takes effect at the next reset)\n",
               hybrid_dnstream_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "dnres") == 0) {
        if (strncmp(str, "on", 2) == 0)  hybrid_dnres_enable(1);
        if (strncmp(str, "off", 3) == 0) hybrid_dnres_enable(0);
        printf("DNRES %s (DeltaNet layers as scratchpad lists)\n", hybrid_dnres_enabled() ? "on" : "off");
    }
    else if (strcmp(token, "silu") == 0) {
        if (strncmp(str, "on", 2) == 0)  resident_silu_enable(1);
        if (strncmp(str, "off", 3) == 0) resident_silu_enable(0);
        printf("SILU %s (FFN activation through the unit 11 SiLU-multiply list)\n",
               resident_silu_active() ? "on" : "off");
    }
    else if (strcmp(token, "resident") == 0) {
        if (strncmp(str, "on", 2) == 0)  resident_enable(1);
        if (strncmp(str, "off", 3) == 0) resident_enable(0);
        printf("RESIDENT %s (27B projections through scratchpad quantize/rotate/restore "
               "lists; hardware %s)\n", resident_active() ? "on" : "off",
               resident_available() ? "present" : "absent");
    }
    else if (strcmp(token, "engdiag") == 0) {
        if (strncmp(str, "on", 2) == 0)  engine_diag = 1;
        if (strncmp(str, "off", 3) == 0) engine_diag = 0;
        printf("ENGDIAG %s (per-op stall snapshots, ~2k cycles/op)\n",
               engine_diag ? "on" : "off");
    }
    else if (strcmp(token, "hdrload") == 0)
        hdrload();
    else if (strcmp(token, "vbench") == 0)
        infer_vbench();
    else if (strlen(token))
        puts("unknown command");
    printf("bonsai> ");
}

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
    irq_setmask(0);
    irq_setie(1);
#endif
    uart_init();
    perf_init();
    infer_static_init();
#ifdef CSR_VECUNIT_BASE
    vec_init();
#endif

    puts("\nTriXium firmware "__DATE__" "__TIME__"\n");
#ifdef BONSAI_N_MEMCTRL
    // The BIOS brought up controller 0 (main_ram); the other DDR4 controllers carry
    // the engine channels' weights and have no BIOS support: init them here, then prove
    // every channel's window with the path the weights take (ddr.c).
    ddr_init_all();
    ddr_test(-1);
#endif
    hdrload_staged(1);   // A different image than main_ram's header says: take the staged one.
    model_info();
    sptest();
#ifdef CSR_VECUNIT_BASE
    { // Vec unit smoke test: quant8 of a known ramp via the CSR dispatch path. Catches
      // emission-level datapath breaks that Migen-sim unit tests cannot see (the VCLAMP8
      // unsigned-literal emission bug shipped through six token-gated milestones).
        volatile int16_t *pi = (volatile int16_t *)(SCRATCHPAD_BASE + 0x100);
        volatile int8_t  *po = (volatile int8_t  *)(SCRATCHPAD_BASE + 0x200);
        // TWO beats: quant8 is x2-unrolled (asm.py, 2026-08-31) and processes beats in
        // pairs, so a 1-beat call also read 0x120-0x13F - zero after a reconfigure, the
        // previous session's data after a warm boot, which failed this test on every
        // serialboot after inference had run (found 2026-09-09; tokens never affected).
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
            // table* and writes via A5/A6, so checksumming A2 was reading an input, and
            // clearing it corrupted the table - which also broke reduce, whose A1 source is
            // that same region.
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
                // otherwise the checksum folds in whatever was left in the scratchpad -
                // deterministic in a fresh sim, but history-dependent on a card that has
                // run before, which made the hardware CRCs incomparable to the interpreter.
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
#endif
    selftest();
    hwops_init();
    if (infer_init() == 0) {
        // Boot generation demo (deterministic; checked by the sim harness).
        static const int prompt[3] = {1, 2, 3};
        // The legs that only exist for the gate are skipped on big models: on the 1.7B
        // the all-CPU reference alone is 3.2 G cycles - 15 s of what was a 16 s boot -
        // and the tiny sim model (2 layers) still runs every leg. GEN-HW stays: two
        // tokens through the hardware path cost 35 ms and prove the units are alive.
        const int big_model = model_header()->n_layer > 8;
        if (!big_model) {
            // Harness hook for the simulation tools (sim_resident_gate/bench): they need only
            // the console, and on the 1k tiny model the three legs cost ~70 minutes of
            // Verilator time. Offer the skip here, after the firmware is running - an early
            // key would reach the LiteX BIOS, where Q aborts the boot. Only tiny models get
            // here; 2 M cycles is ~30 s of simulation wall time and 13 ms on a board. With
            // no key, or any other key, the legs run exactly as before.
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
            // Speculative-decode leg: greedy spec is lossless, so the tokens must match
            // the same references; the acc/rej counters show the machinery engaged.
            //
            // Skipped on big models at boot purely for boot time - six more 1.7B tokens.
            // (It used to be skipped because the sequencer stalled here; that was the
            // interleaved-gather bug, now fixed.) `spec on` reaches it deliberately, and
            // the sim's tiny model still runs the leg so the gate keeps checking it.
            if (big_model) {
                puts("GEN-SP skipped at boot on this model size (use `spec on` to run it)");
                goto boot_demo_done;
            }
            infer_spec(1);
            printf("GEN-SP:");
            perf_reset();
            infer_generate(prompt, 3, 6, emit_token);
            puts("\nGEN-SP DONE");
            {
                int acc, rej;
                infer_spec_stats(&acc, &rej);
                printf("SPEC acc=%d rej=%d\n", acc, rej);
            }
            perf_print();
            infer_spec(0);
        }
    }
boot_demo_done:
    hwops_init(); // Leave the shell on the fast path, not on the debug fallbacks.
    puts("BONSAI-READY");
    printf("bonsai> ");

    while (1) {
        console_service();
    }
    return 0;
}

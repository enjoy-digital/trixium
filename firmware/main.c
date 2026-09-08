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
#include "engine.h"
#include "model.h"
#include "infer.h"
#include "ops.h"
#include "perf.h"
#include "seq.h"
#include "vec.h"
#include "progs.h"

/*-----------------------------------------------------------------------*/
/* Console                                                               */
/*-----------------------------------------------------------------------*/

static char *readstr(void)
{
    char c[2];
    static char s[1024];
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
            if (ptr >= (int)(sizeof(s) - 1))
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
    puts("peek c addr     - dump one HBM beat via the engine (hex addr)");
    puts("mvtest idx      - matvec tensor <idx> with the test pattern, print checksum");
    puts("selftest        - engine self-tests (matvec + fetch checksums)");
    puts("spec [on|off]   - speculative decode toggle + acc/rej counters");
    puts("seq [on|off]    - sequencer attention lists (on by default; off: CPU-dispatched)");
    puts("tokclr/tok/genb - chunked prompt buffer + generate (host chat tool)");
    puts("gen n id id ... - generate n tokens (greedy) from prompt ids");
    puts("hwops [op cpu|hw] - show/set per-op CPU/HW dispatch (op name or 'all')");
    puts("stats [reset]   - per-op cycle profile");
    puts("bench [n]       - profiled generation benchmark (default 4 tokens)");
    puts("vbench          - stage-by-stage cycle microbenchmark");
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

static void bench_cmd(char *args)
{
    char *tok = get_token(&args);
    int n = (*tok) ? parse_int(tok) : 4;
    static const int prompt[3] = {1, 2, 3};
    perf_reset();
    uint32_t f0 = infer_forwards;
    uint32_t t0 = perf_now();
    printf("GEN:");
    int count = infer_generate(prompt, 3, n, emit_token);
    uint32_t dt = perf_now() - t0;
    uint32_t fwd = infer_forwards - f0;
    puts("");
    perf_print();
    printf("BENCH: %d tokens, %lu cycles total", count, (unsigned long)dt);
    if (count > 0)
        printf(", %lu cycles/token (%lu.%03lu tok/s @%luMHz)",
               (unsigned long)(dt/count),
               (unsigned long)(((uint64_t)CONFIG_CLOCK_FREQUENCY*count)/dt),
               (unsigned long)((((uint64_t)CONFIG_CLOCK_FREQUENCY*count*1000)/dt) % 1000),
               (unsigned long)(CONFIG_CLOCK_FREQUENCY/1000000));
    // Per FORWARD: the comparable figure across the 2026-09-07 change that stopped
    // running the unused final forward (bench 64 = 3 prompt + 63 = 66 forwards, was 67).
    if (fwd > 0)
        printf(", %lu forwards, %lu cycles/fwd", (unsigned long)fwd, (unsigned long)(dt/fwd));
    puts("");
}

static int gen_emit_to_pbuf;
// Chunked prompt buffer (tok/genb) - main_ram static window, not .bss (sram is tight;
// see firmware/linker.ld). BONSAI_HDR_ADDR + 0xA0000, clear of the entry cache (+0x80000)
// and ek_undo (+0x90000).
#define PBUF_N 512
static int *const pbuf = (int *)(BONSAI_HDR_ADDR + 0xA0000); // [PBUF_N]
static int pbuf_n;

static void emit_token(int tok)
{
    printf(" %d", tok);
    if (gen_emit_to_pbuf && pbuf_n < PBUF_N)
        pbuf[pbuf_n++] = tok;
}

// Copy the BONS header+table from the HBM staging area (engine channel 0, uploaded there by
// bonsai_load) into the CPU-visible header location, then re-init inference.
#define HDR_STAGING (BONSAI_CHAN_WEIGHT_BASE - 0x100000)
static void hdrload(void)
{
    // First fetch just the header to learn the table size.
    engine_fetch(0, HDR_STAGING, 128, 0xC000);
    volatile struct bons_header *sh =
        (volatile struct bons_header *)(SCRATCHPAD_BASE + 0xC000);
    if (sh->magic != BONS_MAGIC) {
        printf("hdrload: no BONS header at staging (magic %08lx)\n",
               (unsigned long)sh->magic);
        return;
    }
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
    printf("hdrload: %lu KB copied\n", (unsigned long)(bytes/1024));
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
    printf("TOKCLR %d\n", pbuf_n);
}

static void tok_cmd(char *args)
{
    char *tok;
    while (1) {
        tok = get_token(&args);
        if (*tok == 0)
            break;
        if (pbuf_n < PBUF_N)
            pbuf[pbuf_n++] = parse_int(tok);
    }
    printf("TOK %d\n", pbuf_n); // Host uses this to confirm the running length.
}

// Generate from the accumulated prompt buffer; appends the emitted tokens to it so the
// next turn continues the conversation without the host resending history.
static void genb_cmd(char *args)
{
    char *tok = get_token(&args);
    int n = parse_int(tok);
    if (n <= 0 || pbuf_n == 0) {
        puts("usage: tok <ids...> ; genb <n>");
        return;
    }
    printf("GEN:");
    gen_emit_to_pbuf = 1;
    infer_generate(pbuf, pbuf_n, n, emit_token);
    gen_emit_to_pbuf = 0;
    puts("\nGEN DONE");
}

static void gen_cmd(char *args)
{
    static int prompt[64];
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
    infer_generate(prompt, np, n, emit_token);
    puts("\nGEN DONE");
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
        if (strncmp(str, "on", 2) == 0)  qkvl_use = 1;
        if (strncmp(str, "off", 3) == 0) qkvl_use = 0;
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
            seq_use = 1;
        else if (strncmp(str, "off", 3) == 0)
            seq_use = 0;
        printf("SEQ %s (off runs the same ops CPU-dispatched per head, ~20%% slower)\n",
               seq_use ? "on" : "off");
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
#ifdef CSR_VECUNIT_BASE
    vec_init();
#endif

    puts("\nTriXium firmware "__DATE__" "__TIME__"\n");
    model_info();
    sptest();
#ifdef CSR_VECUNIT_BASE
    { // Vec unit smoke test: quant8 of a known ramp via the CSR dispatch path. Catches
      // emission-level datapath breaks that Migen-sim unit tests cannot see (the VCLAMP8
      // unsigned-literal emission bug shipped through six token-gated milestones).
        volatile int16_t *pi = (volatile int16_t *)(SCRATCHPAD_BASE + 0x100);
        volatile int8_t  *po = (volatile int8_t  *)(SCRATCHPAD_BASE + 0x200);
        for (int i = 0; i < 16; i++) pi[i] = (int16_t)(i*1000 - 8000);
        for (int i = 0; i < 16; i++) po[i] = 0x5A;
        int32_t va[4] = {0x100, 1, 0x200, 1}; // A1 = 1: measured shift.
        uint64_t amax = vec_run(PROG_QUANT8, va, 4); // S6 = source absmax (pass 1).
        // quant8 picks its shift from that absmax (see prog_quant8), so the expectation has
        // to track it - this also puts VCLZN and REDMAX under the smoke test.
        int bl = 0;
        while ((amax >> bl) != 0) bl++;
        int sh = (amax == 0) ? 8 : bl - 7;
        int ok = (amax == 8000);
        for (int i = 0; i < 16; i++) {
            int32_t v = (int32_t)(i*1000 - 8000) << (8 - sh);
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
        // The GEN leg is the all-CPU reference the gate compares against hybrid_generate's
        // CPU modes, so pin it here rather than inheriting whatever the default mask is.
        hwops_set_all(0);
        printf("GEN:");
        perf_reset();
        infer_generate(prompt, 3, 2, emit_token);
        puts("\nGEN DONE");
        perf_print();
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
            if (model_header()->n_layer > 8) {
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

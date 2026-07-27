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

/*-----------------------------------------------------------------------*/
/* Console                                                               */
/*-----------------------------------------------------------------------*/

static char *readstr(void)
{
    char c[2];
    static char s[64];
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

static void help(void)
{
    puts("TriXium firmware");
    puts("help       - this message");
    puts("info       - model info (BONS header)");
    puts("sptest     - scratchpad bus read/write test");
    puts("selftest   - engine self-tests (matvec + fetch checksums)");
}

static void sptest(void)
{
    volatile uint32_t *sp = (volatile uint32_t *)SCRATCHPAD_BASE;
    int errors = 0;
    for (uint32_t i = 0; i < 64; i++)
        sp[i] = i*0x9E3779B1u;
    for (uint32_t i = 0; i < 64; i++)
        if (sp[i] != i*0x9E3779B1u)
            errors++;
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
    fill_x_pattern(xsrc, e->cols);
    model_matvec(e, xsrc, dst);
    uint32_t n = e->rows*4; // rows int32 results (all channels).
    uint32_t csum = fnv1a((volatile uint8_t *)(SCRATCHPAD_BASE + dst), n);
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
    else if (strcmp(token, "selftest") == 0)
        selftest();
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

    puts("\nTriXium firmware "__DATE__" "__TIME__"\n");
    model_info();
    sptest();
    selftest();
    puts("BONSAI-READY");
    printf("bonsai> ");

    while (1) {
        console_service();
    }
    return 0;
}

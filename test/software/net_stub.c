/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Host harness for firmware/net.c and web.c (test_net.py): frames out to a queue, a settable
 * clock, a chat that streams a fixed reply.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdint.h>
#include <string.h>

#include "chat.h"
#include "net.h"
#include "web.h"

#define QN 256
static uint8_t  q[QN][1600];
static int      qlen[QN], qn;
static uint64_t now;
static int      resets, turns;
static char     last_line[4096];

void net_tx(const uint8_t *f, int len) { if (qn < QN) { memcpy(q[qn], f, len); qlen[qn++] = len; } }
uint64_t net_now_us(void) { return now; }
void net_platform_init(void) {}
void net_platform_poll(void) { net_poll(); }

int  stub_count(void) { return qn; }
int  stub_frame(int i, uint8_t *out) { memcpy(out, q[i], qlen[i]); return qlen[i]; }
void stub_clear(void) { qn = 0; }
void stub_set_now(uint64_t t) { now = t; }
int  stub_resets(void) { return resets; }
int  stub_turns(void) { return turns; }
const char *stub_last_line(void) { return last_line; }

int  chat_ready(void) { return 1; }
void chat_reset(void) { resets++; }
int  chat_turn(const char *line, int nmax, void (*out)(const uint8_t *, uint32_t), struct chat_result *res)
{
    (void)nmax;
    strncpy(last_line, line, sizeof(last_line) - 1);
    turns++;
    const char *words[] = {"Bonjour", " ! ", "Je suis ", "TriXium", "."};
    for (int i = 0; i < 5; i++)
        out((const uint8_t *)words[i], (uint32_t)strlen(words[i]));
    memset(res, 0, sizeof(*res));
    res->tokens = 5; res->ms = 500; res->context = 42;
    return 0;
}

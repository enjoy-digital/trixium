/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium's web page: the chat in a browser, served by the board
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

// HTTP/1.1 on port 80, one request per connection (Connection: close):
//   GET  /       the page (web_page.h)
//   POST /chat   the message in the body; the reply streams back as UTF-8 text, then a record
//                separator (0x1e) and {"tokens","ms","context","capped","restarted"}
//   POST /reset  a new conversation
// The receive path only buffers requests; web_service runs them from the main loop, so a reply
// that streams for a minute keeps the stack polled (ACKs, retransmissions) without another
// generation starting inside it: a second /chat waits for the first.

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "chat.h"
#include "net.h"
#include "web.h"
#include "web_page.h"

#define REQ_MAX   4096
#define REPLY_MAX 1024     // Tokens per reply.
#define STALL_US  10000000 // A peer that takes nothing for 10 s is gone.

struct req {
    int  n, done;
    char buf[REQ_MAX + 1];
};

#ifdef TRIXIUM_FIRMWARE
#include <generated/soc.h>
#include "static_memory.h"
static struct req *const reqs = (struct req *)TRIXIUM_WEB_REQS_ADDR;       // Not .bss (sram).
#else
static struct req reqs[NET_CONNS];
#endif

// strstr (the firmware's libc has none).
static char *find(const char *h, const char *n)
{
    size_t k = strlen(n);
    for (; *h; h++)
        if (strncmp(h, n, k) == 0)
            return (char *)h;
    return NULL;
}
static int cur = -1;                 // The connection a reply streams to.

static void on_data(int c, const uint8_t *d, int n)
{
    struct req *r = &reqs[c];
    if (r->done)
        return;
    if (r->n + n > REQ_MAX)
        n = REQ_MAX - r->n;
    memcpy(r->buf + r->n, d, n);
    r->n += n;
    r->buf[r->n] = 0;
    char *end = find(r->buf, "\r\n\r\n");
    if (!end && r->n < REQ_MAX)
        return;
    int body = 0;
    if (end) {
        const char *cl = find(r->buf, "Content-Length:");
        if (!cl) cl = find(r->buf, "content-length:");
        if (cl && cl < end) {
            cl += 15;
            while (*cl == ' ') cl++;
            while (*cl >= '0' && *cl <= '9') body = body*10 + (*cl++ - '0');
        }
        if ((end + 4 - r->buf) + body > r->n && r->n < REQ_MAX)
            return;                                     // The body is still coming.
    }
    r->done = 1;
}

static void on_closed(int c)
{
    reqs[c].n = reqs[c].done = 0;
}

// Everything into the connection's queue, the stack polled while it fills.
static int web_write(int c, const void *d, int n)
{
    const uint8_t *p = d;
    uint64_t t0 = net_now_us();
    while (n > 0) {
        if (!net_alive(c))
            return -1;
        int k = net_send(c, p, n);
        p += k; n -= k;
        if (k)
            t0 = net_now_us();
        else if (net_now_us() - t0 > STALL_US)
            return -1;
        if (n)
            net_platform_poll();
    }
    return 0;
}

static void web_puts(int c, const char *s) { web_write(c, s, (int)strlen(s)); }

// Text building without snprintf (the firmware's libc has none).
static char *put_s(char *p, const char *s) { while (*s) *p++ = *s++; *p = 0; return p; }
static char *put_u(char *p, unsigned long v)
{
    char t[12]; int n = 0;
    do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) *p++ = t[--n];
    *p = 0;
    return p;
}

static void chat_out(const uint8_t *b, uint32_t n)
{
    if (cur >= 0)
        web_write(cur, b, (int)n);
    net_platform_poll();
}

static void reply(int c, const char *status, const char *type, const char *body, int n)
{
    char h[256], *p = h;
    p = put_s(p, "HTTP/1.1 "); p = put_s(p, status);
    p = put_s(p, "\r\nContent-Type: "); p = put_s(p, type);
    p = put_s(p, "\r\nContent-Length: "); p = put_u(p, (unsigned long)n);
    put_s(p, "\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
    web_puts(c, h);
    if (n)
        web_write(c, body, n);
}

static void handle(int c)
{
    struct req *r = &reqs[c];
    char *body = find(r->buf, "\r\n\r\n");
    body = body ? body + 4 : r->buf + r->n;
    if (strncmp(r->buf, "GET / ", 6) == 0 || strncmp(r->buf, "GET /index.html ", 16) == 0) {
        reply(c, "200 OK", "text/html; charset=utf-8", web_page, (int)sizeof(web_page) - 1);
    } else if (strncmp(r->buf, "POST /reset ", 12) == 0) {
        chat_reset();
        reply(c, "200 OK", "text/plain", "ok", 2);
    } else if (strncmp(r->buf, "POST /chat ", 11) == 0) {
        if (!chat_ready()) {
            const char *m = "the board has no model or tokenizer loaded";
            reply(c, "503 Service Unavailable", "text/plain; charset=utf-8", m, (int)strlen(m));
        } else if (*body == 0) {
            reply(c, "400 Bad Request", "text/plain", "empty message", 13);
        } else {
            web_puts(c, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
                        "Cache-Control: no-cache\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n");
            struct chat_result res;
            cur = c;
            chat_turn(body, REPLY_MAX, chat_out, &res);
            cur = -1;
            char s[160], *p = s;
            p = put_s(p, "\x1e{\"tokens\":"); p = put_u(p, (unsigned long)res.tokens);
            p = put_s(p, ",\"ms\":");        p = put_u(p, res.ms);
            p = put_s(p, ",\"context\":");   p = put_u(p, (unsigned long)res.context);
            p = put_s(p, ",\"capped\":");    p = put_u(p, (unsigned long)res.capped);
            p = put_s(p, ",\"restarted\":"); p = put_u(p, (unsigned long)res.restarted);
            p = put_s(p, ",\"error\":");     p = put_u(p, (unsigned long)res.error);
            put_s(p, "}");
            web_puts(c, s);
        }
    } else {
        reply(c, "404 Not Found", "text/plain", "not found", 9);
    }
    net_close(c);
    r->done = 0;
    r->n = 0;
}

void web_init(const uint8_t mac[6], uint32_t ip)
{
    memset(reqs, 0, NET_CONNS*sizeof(struct req));
    net_init(mac, ip);
    net_listen(80, on_data, on_closed);
}

void web_service(void)
{
    net_platform_poll();
    if (net_ip_changed()) {               // A DHCP lease, or the static fallback.
        uint32_t ip = net_ip();
        printf("\nWEB http://%d.%d.%d.%d/ (%s), http://trixium.local/\n", (int)(ip >> 24),
               (int)((ip >> 16) & 0xff), (int)((ip >> 8) & 0xff), (int)(ip & 0xff),
               net_ip_dhcp() ? "DHCP" : "static: no DHCP server");
    }
    for (int c = 0; c < NET_CONNS; c++)
        if (reqs[c].done && net_alive(c)) {
            handle(c);
            break;
        }
}

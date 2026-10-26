/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium's small network stack: ARP, ICMP echo and a few TCP connections, for the web page
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

// Enough TCP for a browser on the same LAN: a passive open, in-order receive (anything else is
// dropped and the expected sequence acknowledged again), a send queue per connection with go-back-N
// retransmission, FIN both ways. Replies go to the MAC address the peer's frames came from (no ARP
// table: the server only answers). Portable C: the platform gives net_tx() and net_now_us(), the
// board's in net_ethmac.c, the host test's in test/software/test_net.py.

#include <stdint.h>
#include <string.h>

#include "net.h"

#define ETH_HDR  14
#define IP_HDR   20
#define TCP_HDR  20

#define TCP_FIN  0x01
#define TCP_SYN  0x02
#define TCP_RST  0x04
#define TCP_PSH  0x08
#define TCP_ACK  0x10

#define RTO_US        300000u    // Retransmission timeout (a LAN).
#define RETRIES       10
#define RX_WINDOW     4096u      // What the server advertises (requests are small).

enum { CLOSED = 0, SYN_RCVD, ESTABLISHED, CLOSING };

struct conn {
    int      state;
    uint8_t  peer_mac[6];
    uint32_t peer_ip;
    uint16_t peer_port;
    uint32_t rcv_nxt;            // Next sequence expected from the peer.
    uint32_t snd_una, snd_nxt;   // Oldest unacknowledged, next to send (our sequence space).
    uint32_t iss;
    uint32_t peer_wnd;
    uint32_t head;               // Bytes ever queued (the queue holds [snd_una - iss - 1, head)).
    int      fin_queued, fin_sent, peer_fin;
    uint64_t sent_at;            // Last (re)transmission of unacknowledged data.
    int      retries;
    uint8_t  buf[NET_TXBUF];
};

static uint8_t  my_mac[6];
static uint32_t my_ip;
static uint16_t listen_port;
static void   (*app_data)(int c, const uint8_t *d, int n);
static void   (*app_closed)(int c);
#ifdef TRIXIUM_FIRMWARE
#include <generated/soc.h>
#include "static_memory.h"
static struct conn *const conns = (struct conn *)TRIXIUM_NET_CONNS_ADDR;   // Not .bss (sram).
#else
static struct conn conns[NET_CONNS];
#endif
static uint32_t iss_seed = 0x54525800u;
static uint16_t ip_id;
#ifdef TRIXIUM_FIRMWARE
static uint8_t *const txf = (uint8_t *)TRIXIUM_NET_TXF_ADDR;   // Not .bss either (1.5 KB).
#else
static uint8_t  txf[ETH_HDR + IP_HDR + TCP_HDR + 4 + NET_MSS];
#endif

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] << 8 | p[1]); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static void wr16(uint8_t *p, uint16_t v) { p[0] = v >> 8; p[1] = (uint8_t)v; }
static void wr32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = (uint8_t)v; }

static uint32_t csum_add(uint32_t sum, const uint8_t *p, int n)
{
    for (int i = 0; i + 1 < n; i += 2)
        sum += rd16(p + i);
    if (n & 1)
        sum += (uint32_t)p[n - 1] << 8;
    return sum;
}

static uint16_t csum_fold(uint32_t sum)
{
    while (sum >> 16)
        sum = (sum & 0xffff) + (sum >> 16);
    return (uint16_t)~sum;
}

static void eth_ip_header_ttl(uint8_t *f, const uint8_t *dst_mac, uint32_t src_ip, uint32_t dst_ip,
                              uint8_t proto, int ip_len, uint8_t ttl);

/* UDP: DHCP client, mDNS responder */
/*----------------------------------*/

// The address comes from a DHCP server when the LAN has one (DISCOVER, OFFER, REQUEST, ACK,
// renewal at half the lease); without an answer within DHCP_WAIT_US the static address given to
// net_init is used (a cable straight to a PC) and DHCP keeps trying in the background. mDNS
// answers "<hostname>.local" (A records), so the page is http://trixium.local/ on any LAN.

enum { DHCP_DISCOVER = 1, DHCP_REQUEST, DHCP_BOUND, DHCP_STATIC };
#define DHCP_WAIT_US   4000000u
#define DHCP_RETRY_US  1000000u
#define DHCP_BG_US     30000000u    // Background DISCOVERs on the static address.

static int      dhcp_state;
static uint32_t dhcp_xid, dhcp_server, dhcp_offer, static_ip;
static uint64_t dhcp_start, dhcp_sent, dhcp_renew_at, dhcp_expire_at;
static int      ip_changed;
static const char hostname[] = "trixium";
static const uint8_t bcast_mac[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static const uint8_t mdns_mac[6]  = {0x01, 0x00, 0x5e, 0x00, 0x00, 0xfb};
#define MDNS_IP 0xe00000fbu         // 224.0.0.251

static void udp_out(const uint8_t *dst_mac, uint32_t src, uint32_t dst, uint16_t sport, uint16_t dport,
                    const uint8_t *d, int n, uint8_t ttl)
{
    static uint8_t f[ETH_HDR + IP_HDR + 8 + 600];
    if (n > 600)
        return;
    eth_ip_header_ttl(f, dst_mac, src, dst, 17, IP_HDR + 8 + n, ttl);
    uint8_t *u = f + ETH_HDR + IP_HDR;
    wr16(u, sport); wr16(u + 2, dport); wr16(u + 4, (uint16_t)(8 + n)); wr16(u + 6, 0);
    memcpy(u + 8, d, n);
    uint8_t pseudo[12];
    wr32(pseudo, src); wr32(pseudo + 4, dst); pseudo[8] = 0; pseudo[9] = 17; wr16(pseudo + 10, (uint16_t)(8 + n));
    uint16_t c = csum_fold(csum_add(csum_add(0, pseudo, 12), u, 8 + n));
    wr16(u + 6, c ? c : 0xffff);
    int len = ETH_HDR + IP_HDR + 8 + n;
    if (len < 60) { memset(f + len, 0, 60 - len); len = 60; }
    net_tx(f, len);
}

static void dhcp_send(int type)         // 1 DISCOVER, 3 REQUEST
{
    uint8_t m[300];
    memset(m, 0, sizeof(m));
    m[0] = 1; m[1] = 1; m[2] = 6;
    wr32(m + 4, dhcp_xid);
    int renew = dhcp_state == DHCP_BOUND;
    if (!renew) wr16(m + 10, 0x8000);   // Broadcast replies: no address to receive them at yet.
    if (renew) wr32(m + 12, my_ip);     // ciaddr
    memcpy(m + 28, my_mac, 6);
    wr32(m + 236, 0x63825363u);
    int o = 240;
    m[o++] = 53; m[o++] = 1; m[o++] = (uint8_t)type;
    if (type == 3 && !renew) {
        m[o++] = 50; m[o++] = 4; wr32(m + o, dhcp_offer); o += 4;
        m[o++] = 54; m[o++] = 4; wr32(m + o, dhcp_server); o += 4;
    }
    m[o++] = 12; m[o++] = sizeof(hostname) - 1; memcpy(m + o, hostname, sizeof(hostname) - 1); o += sizeof(hostname) - 1;
    m[o++] = 55; m[o++] = 3; m[o++] = 1; m[o++] = 3; m[o++] = 51;
    m[o++] = 255;
    udp_out(bcast_mac, renew ? my_ip : 0, 0xffffffffu, 68, 67, m, sizeof(m), 64);
    dhcp_sent = net_now_us();
}

static void dhcp_begin(void)
{
    dhcp_xid = 0x54580000u ^ ((uint32_t)my_mac[4] << 8 | my_mac[5]) ^ (uint32_t)net_now_us();
    if (dhcp_state != DHCP_STATIC)
        dhcp_state = DHCP_DISCOVER;
    dhcp_send(1);
}

static void dhcp_in(const uint8_t *m, int n)
{
    if (n < 240 || m[0] != 2 || rd32(m + 4) != dhcp_xid || rd32(m + 236) != 0x63825363u)
        return;
    for (int i = 0; i < 6; i++)                 // Our chaddr (the firmware's libc has no memcmp).
        if (m[28 + i] != my_mac[i])
            return;
    int type = 0;
    uint32_t server = 0, lease = 3600;
    for (int o = 240; o < n && m[o] != 255; ) {
        if (m[o] == 0) { o++; continue; }
        if (o + 2 > n || o + 2 + m[o + 1] > n) break;
        const uint8_t *v = m + o + 2;
        if (m[o] == 53 && m[o + 1] >= 1) type = v[0];
        if (m[o] == 54 && m[o + 1] >= 4) server = rd32(v);
        if (m[o] == 51 && m[o + 1] >= 4) lease = rd32(v);
        o += 2 + m[o + 1];
    }
    uint64_t now = net_now_us();
    if (type == 2 && (dhcp_state == DHCP_DISCOVER || dhcp_state == DHCP_STATIC)) {   // OFFER
        dhcp_offer = rd32(m + 16);
        dhcp_server = server;
        dhcp_state = DHCP_REQUEST;
        dhcp_send(3);
    } else if (type == 5 && (dhcp_state == DHCP_REQUEST || dhcp_state == DHCP_BOUND)) {   // ACK
        uint32_t ip = rd32(m + 16);
        if (ip != my_ip) ip_changed = 1;
        my_ip = ip;
        dhcp_state = DHCP_BOUND;
        if (lease < 60) lease = 60;
        dhcp_renew_at  = now + (uint64_t)lease*500000u;
        dhcp_expire_at = now + (uint64_t)lease*1000000u;
    } else if (type == 6) {                                                           // NAK
        dhcp_state = DHCP_DISCOVER;
        dhcp_begin();
    }
}

static void dhcp_poll(void)
{
    uint64_t now = net_now_us();
    switch (dhcp_state) {
    case DHCP_DISCOVER:
    case DHCP_REQUEST:
        if (now - dhcp_start > DHCP_WAIT_US && my_ip == 0) {      // No server: the static address.
            my_ip = static_ip;
            ip_changed = 1;
            dhcp_state = DHCP_STATIC;
            dhcp_sent = now;
        } else if (now - dhcp_sent > DHCP_RETRY_US) {
            if (dhcp_state == DHCP_DISCOVER) dhcp_begin(); else dhcp_send(3);
        }
        break;
    case DHCP_STATIC:
        if (now - dhcp_sent > DHCP_BG_US)
            dhcp_begin();
        break;
    case DHCP_BOUND:
        if (now > dhcp_expire_at) {
            dhcp_state = DHCP_DISCOVER;
            dhcp_start = now;
            dhcp_begin();
        } else if (now > dhcp_renew_at && now - dhcp_sent > 10*DHCP_RETRY_US) {
            dhcp_send(3);
        }
        break;
    }
}

// "<label><label>" of a DNS name at m + o against hostname.local; returns the offset after it or -1.
static int mdns_name_is_ours(const uint8_t *m, int n, int o)
{
    const char *want[2] = {hostname, "local"};
    for (int k = 0; k < 2; k++) {
        int len = (int)strlen(want[k]);
        if (o >= n || m[o] != len || o + 1 + len > n)
            return -1;
        for (int i = 0; i < len; i++) {
            char c = (char)m[o + 1 + i];
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            if (c != want[k][i])
                return -1;
        }
        o += 1 + len;
    }
    return (o < n && m[o] == 0) ? o + 1 : -1;
}

static void mdns_in(const uint8_t *f, const uint8_t *ip, const uint8_t *m, int n, uint16_t sport)
{
    if (n < 12 || my_ip == 0 || (m[2] & 0x80) || rd16(m + 4) == 0)   // Responses and empty queries.
        return;
    int q = mdns_name_is_ours(m, n, 12);
    if (q < 0 || q + 4 > n)
        return;
    uint16_t qtype = rd16(m + q);
    if (qtype != 1 && qtype != 255)                                  // A or ANY.
        return;
    int legacy = sport != 5353;                                      // A plain resolver's query.
    uint8_t r[128];
    int o = 0;
    wr16(r, legacy ? rd16(m) : 0); wr16(r + 2, 0x8400);
    wr16(r + 4, legacy ? 1 : 0); wr16(r + 6, 1); wr16(r + 8, 0); wr16(r + 10, 0);
    o = 12;
    int name_len = q - 12;
    if (legacy) {                                                    // The question echoed.
        memcpy(r + o, m + 12, name_len + 4); o += name_len + 4;
    }
    memcpy(r + o, m + 12, name_len); o += name_len;
    wr16(r + o, 1); wr16(r + o + 2, legacy ? 0x0001 : 0x8001);        // A, IN (+ cache flush).
    wr32(r + o + 4, 120); wr16(r + o + 8, 4); wr32(r + o + 10, my_ip);
    o += 14;
    if (legacy)
        udp_out(f + 6, my_ip, rd32(ip + 12), 5353, sport, r, o, 64);
    else
        udp_out(mdns_mac, my_ip, MDNS_IP, 5353, 5353, r, o, 255);
}

void net_init(const uint8_t mac[6], uint32_t ip)
{
    memcpy(my_mac, mac, 6);
    static_ip = ip;
    my_ip = 0;
    memset(conns, 0, NET_CONNS*sizeof(struct conn));
    dhcp_state = DHCP_DISCOVER;
    dhcp_start = net_now_us();
    dhcp_begin();
}

uint32_t net_ip(void) { return my_ip; }
int net_ip_dhcp(void) { return dhcp_state == DHCP_BOUND; }

int net_ip_changed(void)
{
    int c = ip_changed;
    ip_changed = 0;
    return c;
}

void net_listen(uint16_t port, void (*on_data)(int, const uint8_t *, int), void (*on_closed)(int))
{
    listen_port = port;
    app_data    = on_data;
    app_closed  = on_closed;
}

/* Frames */
/*--------*/

static void eth_ip_header(uint8_t *f, const uint8_t *dst_mac, uint32_t dst_ip, uint8_t proto, int ip_len)
{
    eth_ip_header_ttl(f, dst_mac, my_ip, dst_ip, proto, ip_len, 64);
}

static void eth_ip_header_ttl(uint8_t *f, const uint8_t *dst_mac, uint32_t src_ip, uint32_t dst_ip,
                              uint8_t proto, int ip_len, uint8_t ttl)
{
    memcpy(f, dst_mac, 6);
    memcpy(f + 6, my_mac, 6);
    wr16(f + 12, 0x0800);
    uint8_t *ip = f + ETH_HDR;
    ip[0] = 0x45; ip[1] = 0;
    wr16(ip + 2, (uint16_t)ip_len);
    wr16(ip + 4, ip_id++);
    wr16(ip + 6, 0x4000);        // Don't fragment.
    ip[8] = ttl; ip[9] = proto;
    wr16(ip + 10, 0);
    wr32(ip + 12, src_ip);
    wr32(ip + 16, dst_ip);
    wr16(ip + 10, csum_fold(csum_add(0, ip, IP_HDR)));
}

static void tcp_send(const uint8_t *mac, uint32_t ip, uint16_t port, uint32_t rcv_nxt,
                     uint32_t seq, uint8_t flags, const uint8_t *data, int n)
{
    uint8_t *tcp = txf + ETH_HDR + IP_HDR;
    int opt = (flags & TCP_SYN) ? 4 : 0;       // MSS option on the SYN-ACK.
    int tcp_len = TCP_HDR + opt + n;
    eth_ip_header(txf, mac, ip, 6, IP_HDR + tcp_len);
    wr16(tcp, listen_port);
    wr16(tcp + 2, port);
    wr32(tcp + 4, seq);
    wr32(tcp + 8, (flags & TCP_ACK) ? rcv_nxt : 0);
    tcp[12] = (uint8_t)(((TCP_HDR + opt)/4) << 4);
    tcp[13] = flags;
    wr16(tcp + 14, RX_WINDOW);
    wr16(tcp + 16, 0);
    wr16(tcp + 18, 0);
    if (opt) { tcp[20] = 2; tcp[21] = 4; wr16(tcp + 22, NET_MSS); }
    if (n) memcpy(tcp + TCP_HDR + opt, data, n);
    uint8_t pseudo[12];
    wr32(pseudo, my_ip); wr32(pseudo + 4, ip);
    pseudo[8] = 0; pseudo[9] = 6; wr16(pseudo + 10, (uint16_t)tcp_len);
    wr16(tcp + 16, csum_fold(csum_add(csum_add(0, pseudo, 12), tcp, tcp_len)));
    int len = ETH_HDR + IP_HDR + tcp_len;
    if (len < 60) { memset(txf + len, 0, 60 - len); len = 60; }
    net_tx(txf, len);
}

static void tcp_out(const struct conn *k, uint32_t seq, uint8_t flags, const uint8_t *data, int n)
{
    tcp_send(k->peer_mac, k->peer_ip, k->peer_port, k->rcv_nxt, seq, flags, data, n);
}

// A RST for a segment that belongs to no connection (no struct conn on the stack: its 16 KB
// queue is four times the SRAM stack).
static void tcp_reset(const uint8_t *src_mac, uint32_t ip, uint16_t port, uint32_t seq, uint32_t ack, int with_ack)
{
    tcp_send(src_mac, ip, port, ack, seq, TCP_RST | (with_ack ? TCP_ACK : 0), NULL, 0);
}

/* TCP */
/*-----*/

static void conn_free(int c)
{
    if (conns[c].state != CLOSED && app_closed)
        app_closed(c);
    conns[c].state = CLOSED;
}

// Queued bytes not yet acknowledged (0 once our FIN, one past the queue, is acknowledged).
static uint32_t queued(const struct conn *k)
{
    uint32_t base = k->snd_una - k->iss - 1;
    return (int32_t)(k->head - base) > 0 ? k->head - base : 0;
}

// Send what the window and the queue allow, from snd_nxt.
static void tcp_push(int c)
{
    struct conn *k = &conns[c];
    if (k->state != ESTABLISHED && k->state != CLOSING)
        return;
    uint32_t base = k->snd_una - k->iss - 1;           // Queue offset of snd_una.
    uint32_t end  = k->head;
    while (1) {
        uint32_t off = k->snd_nxt - k->iss - 1;
        if (off >= end || off - base >= k->peer_wnd)
            break;
        uint32_t n = end - off;
        if (n > NET_MSS) n = NET_MSS;
        if (n > k->peer_wnd - (off - base)) n = k->peer_wnd - (off - base);
        uint8_t seg[NET_MSS];
        for (uint32_t i = 0; i < n; i++)
            seg[i] = k->buf[(off + i) & (NET_TXBUF - 1)];
        if (k->snd_una == k->snd_nxt)
            k->sent_at = net_now_us();
        tcp_out(k, k->snd_nxt, TCP_ACK | TCP_PSH, seg, (int)n);
        k->snd_nxt += n;
    }
    if (k->fin_queued && !k->fin_sent && k->snd_nxt - k->iss - 1 == end) {
        if (k->snd_una == k->snd_nxt)
            k->sent_at = net_now_us();
        tcp_out(k, k->snd_nxt, TCP_ACK | TCP_FIN, NULL, 0);
        k->snd_nxt++;
        k->fin_sent = 1;
    }
}

static void tcp_in(const uint8_t *f, const uint8_t *ip, const uint8_t *tcp, int tcp_len)
{
    uint32_t src_ip = rd32(ip + 12);
    uint16_t sport = rd16(tcp), dport = rd16(tcp + 2);
    uint32_t seq = rd32(tcp + 4), ack = rd32(tcp + 8);
    int off = (tcp[12] >> 4)*4;
    uint8_t fl = tcp[13];
    if (dport != listen_port || off < TCP_HDR || off > tcp_len)
        return;
    const uint8_t *data = tcp + off;
    int n = tcp_len - off;
    int c;
    for (c = 0; c < NET_CONNS; c++)
        if (conns[c].state != CLOSED && conns[c].peer_ip == src_ip && conns[c].peer_port == sport)
            break;
    if (c == NET_CONNS) {
        if (fl & TCP_RST)
            return;
        if (!(fl & TCP_SYN) || (fl & TCP_ACK)) {
            tcp_reset(f + 6, src_ip, sport, (fl & TCP_ACK) ? ack : 0, seq + n + ((fl & TCP_FIN) ? 1 : 0), !(fl & TCP_ACK));
            return;
        }
        for (c = 0; c < NET_CONNS && conns[c].state != CLOSED; c++);
        if (c == NET_CONNS) {                           // Full: refuse.
            tcp_reset(f + 6, src_ip, sport, 0, seq + 1, 1);
            return;
        }
        struct conn *k = &conns[c];
        memset(k, 0, sizeof(*k) - sizeof(k->buf));
        memcpy(k->peer_mac, f + 6, 6);
        k->peer_ip = src_ip; k->peer_port = sport;
        k->rcv_nxt = seq + 1;
        k->iss = iss_seed; iss_seed += 0x01000193u;
        k->snd_una = k->iss; k->snd_nxt = k->iss + 1;
        k->peer_wnd = rd16(tcp + 14);
        k->state = SYN_RCVD;
        k->sent_at = net_now_us();
        tcp_out(k, k->iss, TCP_SYN | TCP_ACK, NULL, 0);
        return;
    }
    struct conn *k = &conns[c];
    if (fl & TCP_RST) {
        conn_free(c);
        return;
    }
    if (fl & TCP_SYN) {                                 // A SYN again: our SYN-ACK was lost.
        if (k->state == SYN_RCVD)
            tcp_out(k, k->iss, TCP_SYN | TCP_ACK, NULL, 0);
        return;
    }
    if (!(fl & TCP_ACK))
        return;
    // The acknowledgement: anything in (snd_una, snd_nxt].
    if ((int32_t)(ack - k->snd_una) > 0 && (int32_t)(ack - k->snd_nxt) <= 0) {
        k->snd_una = ack;
        k->retries = 0;
        k->sent_at = net_now_us();
        if (k->state == SYN_RCVD)
            k->state = ESTABLISHED;
    }
    k->peer_wnd = rd16(tcp + 14);
    if (k->state == SYN_RCVD)
        return;
    // In-order payload only; anything else gets the expected sequence acknowledged again.
    int ack_now = 0;
    if (n > 0 || (fl & TCP_FIN)) {
        if (seq == k->rcv_nxt) {
            if (n > 0) {
                k->rcv_nxt += (uint32_t)n;
                if (app_data && !k->peer_fin)
                    app_data(c, data, n);
            }
            if (fl & TCP_FIN) {
                k->rcv_nxt++;
                k->peer_fin = 1;
            }
        }
        ack_now = 1;
    }
    if (ack_now && k->state != CLOSED)
        tcp_out(k, k->snd_nxt, TCP_ACK, NULL, 0);
    // Both FINs out and ours acknowledged: done (no TIME_WAIT on a server that answers).
    if (k->fin_sent && k->snd_una == k->snd_nxt && k->peer_fin) {
        conn_free(c);
        return;
    }
    tcp_push(c);
}

void net_rx(const uint8_t *f, int len)
{
    if (len < ETH_HDR)
        return;
    uint16_t type = rd16(f + 12);
    if (type == 0x0806 && len >= ETH_HDR + 28) {        // ARP
        const uint8_t *a = f + ETH_HDR;
        if (rd16(a + 6) != 1 || my_ip == 0 || rd32(a + 24) != my_ip)
            return;
        uint8_t r[60] = {0};
        memcpy(r, a + 8, 6); memcpy(r + 6, my_mac, 6); wr16(r + 12, 0x0806);
        uint8_t *p = r + ETH_HDR;
        wr16(p, 1); wr16(p + 2, 0x0800); p[4] = 6; p[5] = 4; wr16(p + 6, 2);
        memcpy(p + 8, my_mac, 6); wr32(p + 14, my_ip);
        memcpy(p + 18, a + 8, 6); memcpy(p + 24, a + 14, 4);
        net_tx(r, 60);
        return;
    }
    if (type != 0x0800 || len < ETH_HDR + IP_HDR)
        return;
    const uint8_t *ip = f + ETH_HDR;
    int ihl = (ip[0] & 0xf)*4, ip_len = rd16(ip + 2);
    if ((ip[0] >> 4) != 4 || ihl < IP_HDR || ip_len < ihl || ETH_HDR + ip_len > len)
        return;
    uint32_t dst = rd32(ip + 16);
    int unicast = my_ip && dst == my_ip;
    if ((!unicast && dst != 0xffffffffu && dst != MDNS_IP) || (rd16(ip + 6) & 0x3fff))
        return;                                         // Not ours, or a fragment.
    if (csum_fold(csum_add(0, ip, ihl)) != 0)
        return;
    const uint8_t *pl = ip + ihl;
    int pl_len = ip_len - ihl;
    if (ip[9] == 17 && pl_len >= 8) {                   // UDP: DHCP replies, mDNS queries
        uint16_t sport = rd16(pl), dport = rd16(pl + 2), ulen = rd16(pl + 4);
        if (ulen < 8 || ulen > pl_len)
            return;
        if (dport == 68 && sport == 67)
            dhcp_in(pl + 8, ulen - 8);
        else if (dport == 5353)
            mdns_in(f, ip, pl + 8, ulen - 8, sport);
        return;
    }
    if (!unicast)
        return;
    if (ip[9] == 1 && pl_len >= 8 && pl[0] == 8) {      // ICMP echo request
        static uint8_t r[ETH_HDR + IP_HDR + 1500];
        if (pl_len > 1500 - IP_HDR)
            return;
        eth_ip_header(r, f + 6, rd32(ip + 12), 1, IP_HDR + pl_len);
        uint8_t *icmp = r + ETH_HDR + IP_HDR;
        memcpy(icmp, pl, pl_len);
        icmp[0] = 0; wr16(icmp + 2, 0);
        wr16(icmp + 2, csum_fold(csum_add(0, icmp, pl_len)));
        int n = ETH_HDR + IP_HDR + pl_len;
        if (n < 60) { memset(r + n, 0, 60 - n); n = 60; }
        net_tx(r, n);
        return;
    }
    if (ip[9] == 6 && pl_len >= TCP_HDR) {
        uint8_t pseudo[12];
        memcpy(pseudo, ip + 12, 8); pseudo[8] = 0; pseudo[9] = 6; wr16(pseudo + 10, (uint16_t)pl_len);
        if (csum_fold(csum_add(csum_add(0, pseudo, 12), pl, pl_len)) != 0)
            return;
        tcp_in(f, ip, pl, pl_len);
    }
}

void net_poll(void)
{
    dhcp_poll();
    uint64_t now = net_now_us();
    for (int c = 0; c < NET_CONNS; c++) {
        struct conn *k = &conns[c];
        if (k->state == CLOSED)
            continue;
        if (k->snd_una != k->snd_nxt && now - k->sent_at > RTO_US) {
            if (++k->retries > RETRIES) {
                conn_free(c);
                continue;
            }
            k->sent_at = now;
            if (k->state == SYN_RCVD) {
                tcp_out(k, k->iss, TCP_SYN | TCP_ACK, NULL, 0);
                continue;
            }
            k->snd_nxt = k->snd_una;                    // Go back N (the FIN again too).
            k->fin_sent = 0;
        }
        tcp_push(c);
        // Closed by us, everything acknowledged, the peer never sent its FIN: give up after a while.
        if (k->fin_sent && k->snd_una == k->snd_nxt && !k->peer_fin && now - k->sent_at > 10*RTO_US)
            conn_free(c);
    }
}

int net_send(int c, const void *d, int n)
{
    struct conn *k = &conns[c];
    if (k->state != ESTABLISHED || k->fin_queued)
        return 0;
    int room = net_tx_room(c);
    if (n > room) n = room;
    const uint8_t *p = d;
    for (int i = 0; i < n; i++)
        k->buf[(k->head + (uint32_t)i) & (NET_TXBUF - 1)] = p[i];
    k->head += (uint32_t)n;
    tcp_push(c);
    return n;
}

int net_tx_room(int c)
{
    return (int)(NET_TXBUF - queued(&conns[c]));
}

int net_tx_idle(int c)
{
    const struct conn *k = &conns[c];
    return k->state == CLOSED || k->snd_una == k->snd_nxt;
}

void net_close(int c)
{
    struct conn *k = &conns[c];
    if (k->state == ESTABLISHED) {
        k->fin_queued = 1;
        k->state = CLOSING;
        tcp_push(c);
    }
}

int net_alive(int c)
{
    return conns[c].state != CLOSED;
}

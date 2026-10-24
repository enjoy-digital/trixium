/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium's small network stack: ARP, ICMP echo and a few TCP connections, for the web page
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_NET_H
#define TRIXIUM_NET_H

#include <stdint.h>

#define NET_CONNS   4        // TCP connections at once (a browser opens two or three).
#define NET_TXBUF   16384    // Bytes queued per connection (a power of two).
#define NET_MSS     1400

// Platform (the CPU's Ethernet MAC on the board, a test harness on the host).
void     net_tx(const uint8_t *frame, int len);
uint64_t net_now_us(void);

// The address from DHCP, else (no answer within 4 s) `ip`, the static fallback (0xc0a80133 =
// <board-host>); mDNS answers trixium.local.
void net_init(const uint8_t mac[6], uint32_t ip);
uint32_t net_ip(void);                                // 0 until DHCP or the fallback gave one.
int  net_ip_dhcp(void);                               // The address is a DHCP lease.
int  net_ip_changed(void);                            // Since the last call (then cleared).
void net_rx(const uint8_t *frame, int len);           // One received Ethernet frame.
void net_poll(void);                                  // Retransmissions, queued data.

// TCP server: one listening port. on_data gets the in-order payload of connection c; on_closed
// tells the connection is gone (closed, reset or timed out).
void net_listen(uint16_t port, void (*on_data)(int c, const uint8_t *d, int n),
                void (*on_closed)(int c));
int  net_send(int c, const void *d, int n);           // Queue; returns the bytes queued.
int  net_tx_room(int c);
int  net_tx_idle(int c);                              // Everything queued is acknowledged.
void net_close(int c);                                // FIN once the queue is out.
int  net_alive(int c);

#endif

/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium's web page: the chat in a browser, served by the board
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_WEB_H
#define TRIXIUM_WEB_H

#include <stdint.h>

// Platform: frames in and timers (net_ethmac.c on the board).
void net_platform_init(void);
void net_platform_poll(void);

void web_init(const uint8_t mac[6], uint32_t ip);
void web_service(void);            // From the main loop: network, then one pending request.

#endif

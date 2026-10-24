/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium's network stack on the CPU's LiteEth MAC (the SRAM slots of add_etherbone(with_ethmac))
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdint.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/mem.h>
#include <generated/soc.h>

#include "net.h"
#include "perf.h"
#include "web.h"

#ifdef CSR_ETHMAC_BASE

#include <libbase/crc.h>

#define ETHMAC_EV_SRAM_WRITER 0x1   // libliteeth/udp.h's (not linked here).
#define ETHMAC_EV_SRAM_READER 0x1

// Without the MAC's own preamble/CRC handling the frames in the slots carry both (as libliteeth).
#ifdef CSR_ETHMAC_PREAMBLE_CRC_ADDR
#define PRE 0
#define FCS 0
#else
#define PRE 8
#define FCS 4
#endif

static uint32_t txslot;

uint64_t net_now_us(void)
{
    return perf_now64()/(CONFIG_CLOCK_FREQUENCY/1000000);
}

void net_tx(const uint8_t *f, int len)
{
    if (PRE + len + FCS > ETHMAC_SLOT_SIZE)
        return;
    for (int t = 0; !ethmac_sram_reader_ready_read(); t++)
        if (t > 1000000) return;                        // The MAC never frees a slot: drop.
    volatile uint8_t *slot = (volatile uint8_t *)(ETHMAC_BASE + ETHMAC_SLOT_SIZE*(ETHMAC_RX_SLOTS + txslot));
    uint8_t *b = (uint8_t *)slot;
#if PRE
    for (int i = 0; i < 7; i++) b[i] = 0x55;
    b[7] = 0xd5;
#endif
    memcpy(b + PRE, f, len);
#if FCS
    uint32_t crc = crc32(b + PRE, len);
    b[PRE + len + 0] = crc; b[PRE + len + 1] = crc >> 8;
    b[PRE + len + 2] = crc >> 16; b[PRE + len + 3] = crc >> 24;
#endif
    ethmac_sram_reader_slot_write(txslot);
    ethmac_sram_reader_length_write(PRE + len + FCS);
    ethmac_sram_reader_start_write(1);
    txslot = (txslot + 1) % ETHMAC_TX_SLOTS;
}

// Every frame waiting in the RX slots into the stack, then its timers.
void net_platform_poll(void)
{
    while (ethmac_sram_writer_ev_pending_read() & ETHMAC_EV_SRAM_WRITER) {
        uint32_t slot = ethmac_sram_writer_slot_read();
        uint32_t len  = ethmac_sram_writer_length_read();
        const uint8_t *b = (const uint8_t *)(ETHMAC_BASE + ETHMAC_SLOT_SIZE*slot);
        if (len >= (uint32_t)(PRE + 14 + FCS) && len <= ETHMAC_SLOT_SIZE)
            net_rx(b + PRE, (int)(len - PRE - FCS));
        ethmac_sram_writer_ev_pending_write(ETHMAC_EV_SRAM_WRITER);
    }
    net_poll();
}

void net_platform_init(void)
{
    ethmac_sram_reader_ev_pending_write(ETHMAC_EV_SRAM_READER);
    ethmac_sram_writer_ev_pending_write(ETHMAC_EV_SRAM_WRITER);
    txslot = 0;
}

#endif

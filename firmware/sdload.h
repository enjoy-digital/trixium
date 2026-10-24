/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium weights from an SD card
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_SDLOAD_H
#define TRIXIUM_SDLOAD_H

#include <stdint.h>

// The card image: up to SD_MAX_SLOTS models, slot k's 512-byte table at sector SD_IMAGE_SECTOR +
// k*SD_SLOT_SECTORS (1 MB, then every 1 GB), its entries' data right after it; sector
// SD_BOOT_SECTOR names the slot the power-up boots (slot 0 without it). Table version 2: magic
// "TXSD", 2, n entries, the CRC-32 of the entries, a 32-byte label, then n entries of (chan,
// channel-relative base, 512-byte blocks, first sector, the CRC-32 of the blocks' bytes). Version 1
// (software/host/sdimage.py, the first sdsave): no label, 4-word entries without CRCs. The BONS
// header+table, every channel's payload and the tokenizer are entries, staged where a UDP upload
// puts them.
#define SD_IMAGE_SECTOR        2048
#define SD_SLOT_SECTORS        (1u << 21)    // 1 GB.
#define SD_MAX_SLOTS           7             // 7 x 1 GB fit the kit's 7.28 GB eMMC.
#define SD_BOOT_SECTOR         2047
#define SD_BOOT_MAGIC          0x54425854u   // "TXBT"
#define SD_IMAGE_MAGIC         0x44535854u   // "TXSD"
#define SD_IMAGE_MAX_ENTRIES   31
#define SD_V2_MAX_ENTRIES      23
#define SD_TABLE_STAGING_BELOW 0x1100000     // The table's own landing place on channel 0.

// tries: CMD8 attempts (each waits out the command timeout without a card).
int sd_init(int mhz, int tries);
int sd_init_done(void);
int sd_load(int chan, uint32_t base, uint32_t sector, uint32_t blocks);
// n blocks to the card from a DRAM window (chan, base), or from CPU memory when cpu != NULL.
int sd_write(uint32_t sector, uint32_t blocks, int chan, uint32_t base, const uint32_t *cpu);
// The staged model (and tokenizer) to slot `slot`; sd_boot loads a slot back (CRC-checked).
int sd_save(int slot, const char *label);
int sd_boot(int slot);
void sd_list(void);
int sd_default_slot(void);
int sd_set_default(int slot);

#endif

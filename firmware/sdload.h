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

// The card image (software/host/sdimage.py): at sector SD_IMAGE_SECTOR (1 MB in) a 512-byte
// table: magic "TXSD", version 1, n entries, the CRC-32 of the entries, then n entries of
// (chan, channel-relative base, 512-byte blocks, first sector); the entries' data each start on a
// block. The BONS header+table, every channel's payload and the tokenizer are entries, staged where
// a UDP upload puts them.
#define SD_IMAGE_SECTOR        2048
#define SD_IMAGE_MAGIC         0x44535854u   // "TXSD"
#define SD_IMAGE_MAX_ENTRIES   31
#define SD_TABLE_STAGING_BELOW 0x1100000     // The table's own landing place on channel 0.

// tries: CMD8 attempts (each waits out the command timeout without a card).
int sd_init(int mhz, int tries);
int sd_load(int chan, uint32_t base, uint32_t sector, uint32_t blocks);
int sd_boot(void);

#endif

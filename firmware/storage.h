/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium models on a block device: the slots and tables shared by the SD/eMMC and SATA paths
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_STORAGE_H
#define TRIXIUM_STORAGE_H

#include <stdint.h>
#include <generated/soc.h>

// The device image (512-byte sectors, any block device): up to STORAGE_MAX_SLOTS models, slot k's
// table at sector STORAGE_IMAGE_SECTOR + k*STORAGE_SLOT_SECTORS (1 MB, then every slot: 1 GB on the
// Ti375's eMMC, the target's TRIXIUM_STORAGE_SLOT_SECTORS elsewhere), its entries' data right
// after it; sector STORAGE_BOOT_SECTOR names the slot the power-up boots (slot 0 without it).
// Table version 3 (4 sectors): magic "TXSD", 3, n entries, the CRC-32 of the entries, a 32-byte
// label, then n entries of 6 words (chan, channel-relative base, 512-byte blocks, first sector's
// low word, the CRC-32 of the blocks' bytes, the sector's high word), the data from the slot's
// 5th sector. Version 2 (one sector): the same with 5-word entries (32-bit sectors), the data
// from the 2nd sector. Version 1 (software/host/sdimage.py's first, the first sdsave): no label,
// 4-word entries without CRCs. An entry is at most STORAGE_MAX_BLOCKS blocks (the loader's
// length is 24 bits of 32-byte beats: 512 MB), so a channel payload beyond that spans several
// entries. The BONS header+table, every channel's payload and the tokenizer are entries, staged
// where a UDP upload puts them.
#define STORAGE_IMAGE_SECTOR        2048
#ifdef TRIXIUM_STORAGE_SLOT_SECTORS
#define STORAGE_SLOT_SECTORS        ((uint64_t)TRIXIUM_STORAGE_SLOT_SECTORS)
#else
#define STORAGE_SLOT_SECTORS        ((uint64_t)1 << 21)   // 1 GB: the Ti375's 7.28 GB eMMC.
#endif
#ifdef TRIXIUM_STORAGE_MAX_SLOTS
#define STORAGE_MAX_SLOTS           TRIXIUM_STORAGE_MAX_SLOTS
#else
#define STORAGE_MAX_SLOTS           7
#endif
#define STORAGE_BOOT_SECTOR         2047
#define STORAGE_BOOT_MAGIC          0x54425854u   // "TXBT"
#define STORAGE_IMAGE_MAGIC         0x44535854u   // "TXSD"
#define STORAGE_V1_MAX_ENTRIES      31
#define STORAGE_V2_MAX_ENTRIES      23
#define STORAGE_V3_TABLE_SECTORS    4
#define STORAGE_V3_MAX_ENTRIES      83            // (4*128 - 12 words)/6
#define STORAGE_MAX_BLOCKS          ((1u << 24)/16)
#define STORAGE_TABLE_STAGING_BELOW 0x1100000     // The table's own landing place on channel 0.

// A block device the firmware can boot a model from. Reads go through the weight loader into the
// channel windows (no CPU copy); the running CRC-32 of the read stream (crc_reset/crc_read, NULL
// without one) checks each entry against its table. `write` takes blocks from a DRAM window
// (chan, base; `bulk_write` devices only: the CPU pushes the words, slow) or from CPU memory
// (cpu != NULL: the table and boot sectors); with `crc`, the running CRC register (0xffffffff to
// start, the caller inverts) of what was pushed. NULL `write`: a read-only device, written from
// the host (software/host/xcu1525.py model --persist).
struct storage_ops {
    const char *name;            // "SD card / eMMC", "SATA"
    const char *tag;             // The console's prefixes: SDBOOT / SATABOOT, SDSLOT / SATASLOT
    const char *cmd;             // The messages' prefix: "sdboot: ..."
    int  (*init)(int fast, int tries);
    int  (*ready)(void);
    int  (*read_to_loader)(uint64_t sector, uint32_t blocks, int chan, uint32_t base);
    int  (*write)(uint64_t sector, uint32_t blocks, int chan, uint32_t base, const uint32_t *cpu,
                  uint32_t *crc);
    int  bulk_write;
    void (*crc_reset)(void);
    uint32_t (*crc_read)(void);
};

extern const struct storage_ops *storage;              // The selected device (NULL: none yet).
void storage_select(const struct storage_ops *ops);
int  storage_probe(int fast, int tries);               // The first device that answers: SATA, then SD.
int  storage_ready(void);                              // Selected and initialised (probing when not).
int  storage_arm_loader(int chan, uint32_t base, uint32_t blocks);
int  storage_wait_loader(void);                        // 0 once the loader wrote its last beat.
uint64_t storage_slot_sector(int slot);
int  storage_read_sectors(uint64_t sector, uint32_t n, uint32_t *dst);   // 1..4 sectors into dst.
int  storage_boot(int slot);
int  storage_save(int slot, const char *label);
void storage_list(void);
int  storage_default_slot(void);
int  storage_set_default(int slot);

// The devices (sdload.c, sataload.c).
extern const struct storage_ops sd_storage_ops;
extern const struct storage_ops sata_storage_ops;

#endif

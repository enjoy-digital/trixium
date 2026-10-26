/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium models on a block device: the slots and tables shared by the SD/eMMC and SATA paths
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>
#include <system.h>

#include "bons.h"
#include "engine.h"
#include "perf.h"
#include "storage.h"
#include "static_memory.h"
#include "tokenizer.h"

#if defined(CSR_SD_BASE) || defined(CSR_SATAUP_BASE)

#include <libbase/crc.h>

// The device the slot commands act on: the first one that answered (storage_probe at boot:
// SATA first, then the SD/eMMC) or the one `sdinit`/`satainit` last brought up.
const struct storage_ops *storage;

void storage_select(const struct storage_ops *ops) { storage = ops; }

int storage_probe(int fast, int tries)
{
#ifdef CSR_SATAUP_BASE
    if (sata_storage_ops.init(fast, tries) == 0) { storage = &sata_storage_ops; return 0; }
#endif
#ifdef CSR_SD_BASE
    if (sd_storage_ops.init(fast, tries) == 0) { storage = &sd_storage_ops; return 0; }
#endif
    return -1;
}

int storage_ready(void)
{
    if (storage == NULL)
        return storage_probe(1, 1000) == 0;
    return storage->ready() || storage->init(1, 1000) == 0;
}

// The loader armed for a channel window, as the host does for a UDP transfer: the device's read
// stream then lands there (`blocks` 512-byte blocks = 16 beats each).
int storage_arm_loader(int chan, uint32_t base, uint32_t blocks)
{
    if (blocks == 0 || blocks > STORAGE_MAX_BLOCKS) {
        printf("storage: 1..%lu blocks per transfer\n", (unsigned long)STORAGE_MAX_BLOCKS);
        return -1;
    }
    trixium_loader_src_sel_write(0);
    trixium_loader_chan_write(chan);
    trixium_loader_base_write(base);
    trixium_loader_length_write(blocks*16);
    trixium_loader_resync_write(0);
    trixium_loader_start_write(1);
    return 0;
}

// The loader drains its last beats into the memory (its writers' responses).
int storage_wait_loader(void)
{
    for (int t = 0; t < 100000 && !(trixium_loader_status_read() & 1); t++)
        busy_wait_us(10);
    return (trixium_loader_status_read() & 1) ? 0 : -1;
}

#if TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000   // The staging areas below the weights.

uint64_t storage_slot_sector(int slot)
{
    return STORAGE_IMAGE_SECTOR + (uint64_t)slot*STORAGE_SLOT_SECTORS;
}

// n (1..4) sectors through the loader into `dst` (the table's landing place on channel 0).
int storage_read_sectors(uint64_t sector, uint32_t n, uint32_t *dst)
{
    uint32_t scratch = TRIXIUM_CHAN_WEIGHT_BASE - STORAGE_TABLE_STAGING_BELOW;
    if (n < 1 || n > STORAGE_V3_TABLE_SECTORS || storage == NULL)
        return -1;
    if (storage->read_to_loader(sector, n, 0, scratch) < 0)
        return -1;
    engine_fetch(0, scratch, n*16, 0xC000);
    for (uint32_t i = 0; i < n*128; i++)
        dst[i] = ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i];
    return 0;
}

// The slot table (2 KB) and the save's zero sectors in the main_ram static window: not on the
// 4 KB stack, not in the 16 KB SRAM's .bss (static_memory.h; nothing there is zeroed).
#define table_buf ((uint32_t *)TRIXIUM_STORAGE_TABLE_ADDR)

// A slot's table into table_buf: its version (1, 2, 3) when valid, -1 when the slot holds no
// image (or a damaged one), -2 when the sector cannot be read (past the device's end). Version
// 3 spans 4 sectors: the first one tells.
static int table_read(int slot)
{
    uint64_t s = storage_slot_sector(slot);
    uint32_t *t = table_buf;
    if (storage_read_sectors(s, 1, t) < 0)
        return -2;
    if (t[0] != STORAGE_IMAGE_MAGIC || t[2] == 0)
        return -1;
    if (t[1] == 1)
        return t[2] <= STORAGE_V1_MAX_ENTRIES && crc32((unsigned char *)&t[4], t[2]*16) == t[3] ? 1 : -1;
    if (t[1] == 2)
        return t[2] <= STORAGE_V2_MAX_ENTRIES && crc32((unsigned char *)&t[12], t[2]*20) == t[3] ? 2 : -1;
    if (t[1] == 3) {
        if (t[2] > STORAGE_V3_MAX_ENTRIES || storage_read_sectors(s, STORAGE_V3_TABLE_SECTORS, t) < 0)
            return -1;
        return crc32((unsigned char *)&t[12], t[2]*24) == t[3] ? 3 : -1;
    }
    return -1;
}

struct storage_entry { uint32_t chan, base, blocks, crc; uint64_t sector; };

static void entry_get(const uint32_t *t, int ver, uint32_t e, struct storage_entry *x)
{
    const uint32_t *w = ver == 1 ? &t[4 + 4*e] : ver == 2 ? &t[12 + 5*e] : &t[12 + 6*e];
    x->chan   = w[0];
    x->base   = w[1];
    x->blocks = w[2];
    x->sector = w[3];
    x->crc    = ver == 1 ? 0 : w[4];
    if (ver == 3)
        x->sector |= (uint64_t)w[5] << 32;
}

static const char *table_label(const uint32_t *t, int ver) { return ver == 1 ? "" : (const char *)&t[4]; }

int storage_boot(int slot)
{
    if (storage == NULL && !storage_ready())
        return -1;
    const char *cmd = storage->cmd, *tag = storage->tag;
    if (slot < 0 || slot >= STORAGE_MAX_SLOTS) {
        printf("%sboot: slot 0..%d\n", cmd, STORAGE_MAX_SLOTS - 1);
        return -1;
    }
    if (!storage_ready())
        return -1;
    int ver = table_read(slot);
    const uint32_t *t = table_buf;
    if (ver < 0) {
        printf("%sboot: no TriXium image in slot %d (sector %lu, magic %08lx)\n", cmd, slot,
               (unsigned long)storage_slot_sector(slot), (unsigned long)t[0]);
        return -1;
    }
    uint64_t t0 = perf_now64(), bytes = 0;
    int checked = 0;
    for (uint32_t e = 0; e < t[2]; e++) {
        struct storage_entry x;
        entry_get(t, ver, e, &x);
        printf("  chan %lu: %lu KB from sector %lu\n", (unsigned long)x.chan,
               (unsigned long)(x.blocks/2), (unsigned long)x.sector);
        if (storage->crc_reset)
            storage->crc_reset();
        if (storage->read_to_loader(x.sector, x.blocks, (int)x.chan, x.base) < 0)
            return -1;
        // The data went to the DRAM without the CPU: the read stream's CRC against the table's.
        if (ver >= 2 && storage->crc_read) {
            uint32_t got = storage->crc_read();
            if (got != x.crc) {
                printf("%sboot: entry %lu CRC %08lx, the table says %08lx: the device's data is damaged\n",
                       cmd, (unsigned long)e, (unsigned long)got, (unsigned long)x.crc);
                return -1;
            }
            checked++;
        }
        bytes += (uint64_t)x.blocks*512;
    }
    uint32_t ms = (uint32_t)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000));
    printf("%sBOOT slot %d \"%s\": %lu MB in %lu.%01lu s (%lu MB/s)%s\n", tag, slot, table_label(t, ver),
           (unsigned long)(bytes >> 20), (unsigned long)(ms/1000), (unsigned long)(ms%1000/100),
           (unsigned long)(ms ? (bytes >> 10)/ms : 0), checked ? ", CRC ok" : "");
    return 0;
}

// One transfer of `bytes` from (chan, base) as table entries from `*sector` on (several when
// longer than the loader's longest transfer); the entry count in *n.
static int add_entries(uint32_t *t, uint32_t *n, uint32_t chan, uint32_t base, uint64_t bytes,
                       uint64_t *sector)
{
    do {
        uint64_t blocks = (bytes + 511)/512;
        if (blocks > STORAGE_MAX_BLOCKS)
            blocks = STORAGE_MAX_BLOCKS;
        if (*n >= STORAGE_V3_MAX_ENTRIES)
            return -1;
        uint32_t *x = &t[12 + 6*(*n)++];
        x[0] = chan;
        x[1] = base;
        x[2] = (uint32_t)blocks;
        x[3] = (uint32_t)*sector;
        x[5] = (uint32_t)(*sector >> 32);
        *sector += blocks;
        base    += (uint32_t)(blocks*512);
        bytes    = bytes > blocks*512 ? bytes - blocks*512 : 0;
    } while (bytes);
    return 0;
}

int storage_save(int slot, const char *label)
{
    if (storage == NULL && !storage_ready())
        return -1;
    const char *cmd = storage->cmd, *tag = storage->tag;
    if (!storage->bulk_write) {
        printf("%ssave: this device is written from the host (model --persist)\n", cmd);
        return -1;
    }
    if (slot < 0 || slot >= STORAGE_MAX_SLOTS) {
        printf("%ssave: slot 0..%d\n", cmd, STORAGE_MAX_SLOTS - 1);
        return -1;
    }
    if (!storage_ready())
        return -1;
    // What a UDP upload staged: the BONS header+table, each channel's payload, the tokenizer.
    _Static_assert(sizeof(struct bons_header) <= 512, "the header in 16 beats");
    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE - 0x100000, 16, 0xC000);
    struct bons_header h;
    for (uint32_t i = 0; i < sizeof(h)/4; i++)
        ((uint32_t *)&h)[i] = ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i];
    if (h.magic != BONS_MAGIC) {
        printf("%ssave: no staged model (upload one first)\n", cmd);
        return -1;
    }
    uint32_t *t = table_buf;
    memset(t, 0, STORAGE_V3_TABLE_SECTORS*512);
    uint32_t n = 0;
    uint64_t sector = storage_slot_sector(slot) + STORAGE_V3_TABLE_SECTORS;
    int rc = add_entries(t, &n, 0, TRIXIUM_CHAN_WEIGHT_BASE - 0x100000, h.payload_offset, &sector);
    for (uint32_t c = 0; c < h.n_channels && rc == 0; c++)
        rc = add_entries(t, &n, c, TRIXIUM_CHAN_WEIGHT_BASE
#ifndef TRIXIUM_LOADER_ADDS_CHAN_OFFSET
                         + c*TRIXIUM_CHAN_STRIDE
#endif
                         , (uint64_t)h.payload_beats*32, &sector);
    engine_fetch(0, TOK_STAGING, 1, 0xC000);
    volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
    if (rc == 0 && rec[0] == TOK_STAGING_MAGIC && rec[1] <= TOK_STAGING_MAX)
        rc = add_entries(t, &n, 0, TOK_STAGING, 32 + rec[1], &sector);
    if (rc < 0) {
        printf("%ssave: more than %d entries\n", cmd, STORAGE_V3_MAX_ENTRIES);
        return -1;
    }
    if (sector - storage_slot_sector(slot) > STORAGE_SLOT_SECTORS) {
        printf("%ssave: the model is larger than a slot\n", cmd);
        return -1;
    }
    t[0] = STORAGE_IMAGE_MAGIC; t[1] = 3; t[2] = n;
    for (int i = 0; i < 31 && label && label[i]; i++)
        ((char *)&t[4])[i] = label[i];
    // The old table first invalidated, the table last: a save cut short leaves no image to boot.
    uint32_t *zero = (uint32_t *)TRIXIUM_STORAGE_ZERO_ADDR;
    memset(zero, 0, STORAGE_V3_TABLE_SECTORS*512);
    if (storage->write(storage_slot_sector(slot), STORAGE_V3_TABLE_SECTORS, 0, 0, zero, NULL) < 0)
        return -1;
    uint64_t t0 = perf_now64(), bytes = 0;
    for (uint32_t e = 0; e < n; e++) {
        uint32_t *x = &t[12 + 6*e], crc = 0xffffffffu;
        uint64_t s = x[3] | (uint64_t)x[5] << 32;
        printf("  chan %lu: %lu KB to sector %lu\n", (unsigned long)x[0], (unsigned long)(x[2]/2),
               (unsigned long)s);
        if (storage->write(s, x[2], (int)x[0], x[1], NULL, &crc) < 0)
            return -1;
        x[4] = ~crc;
        bytes += (uint64_t)x[2]*512;
    }
    t[3] = crc32((unsigned char *)&t[12], n*24);
    if (storage->write(storage_slot_sector(slot), STORAGE_V3_TABLE_SECTORS, 0, 0, t, NULL) < 0)
        return -1;
    uint32_t ms = (uint32_t)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000));
    printf("%sSAVE slot %d \"%s\": %lu MB in %lu s (%lu KB/s)\n", tag, slot, (char *)&t[4],
           (unsigned long)(bytes >> 20), (unsigned long)(ms/1000), (unsigned long)(ms ? bytes/ms : 0));
    return 0;
}

void storage_list(void)
{
    if (!storage_ready())
        return;
    int def = storage_default_slot();
    for (int slot = 0; slot < STORAGE_MAX_SLOTS; slot++) {
        int ver = table_read(slot);
        if (ver == -2)
            break;                                       // Past the device's end.
        if (ver < 0)
            continue;
        const uint32_t *t = table_buf;
        uint64_t bytes = 0;
        for (uint32_t e = 0; e < t[2]; e++) {
            struct storage_entry x;
            entry_get(t, ver, e, &x);
            bytes += (uint64_t)x.blocks*512;
        }
        printf("%sSLOT %d%s \"%s\" %lu MB, %lu entries%s\n", storage->tag, slot, slot == def ? " (boot)" : "",
               table_label(t, ver), (unsigned long)(bytes >> 20), (unsigned long)t[2],
               ver == 1 ? ", version 1 (no CRCs)" : "");
    }
}

// The slot booted at power-up: the boot sector's, 0 without one.
int storage_default_slot(void)
{
    uint32_t *t = table_buf;
    if (storage == NULL || storage_read_sectors(STORAGE_BOOT_SECTOR, 1, t) < 0 ||
        t[0] != STORAGE_BOOT_MAGIC || t[2] != ~t[1] || t[1] >= STORAGE_MAX_SLOTS)
        return 0;
    return (int)t[1];
}

int storage_set_default(int slot)
{
    if (storage == NULL && !storage_ready())
        return -1;
    const char *cmd = storage->cmd;
    if (slot < 0 || slot >= STORAGE_MAX_SLOTS) {
        printf("%sdefault: slot 0..%d\n", cmd, STORAGE_MAX_SLOTS - 1);
        return -1;
    }
    if (storage->write == NULL) {
        printf("%sdefault: this device is written from the host (model --persist)\n", cmd);
        return -1;
    }
    if (!storage_ready())
        return -1;
    uint32_t *t = (uint32_t *)TRIXIUM_STORAGE_BOOT_ADDR;
    memset(t, 0, 512);
    t[0] = STORAGE_BOOT_MAGIC; t[1] = (uint32_t)slot; t[2] = ~(uint32_t)slot;
    if (storage->write(STORAGE_BOOT_SECTOR, 1, 0, 0, t, NULL) < 0)
        return -1;
    printf("%sDEFAULT slot %d\n", storage->tag, slot);
    return 0;
}

#endif // TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000

#endif // CSR_SD_BASE || CSR_SATAUP_BASE

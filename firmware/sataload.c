/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium weights from a SATA SSD
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

#include "engine.h"
#include "perf.h"
#include "storage.h"

#ifdef CSR_SATAUP_BASE

// The SSD's read stream feeds the weight loader (gateware/memory/sata_upload.py `SATAUpload`):
// the firmware arms the loader for a channel window, as the host does for a UDP transfer, gives
// the module a sector and a count, and the sectors land there - no CPU copy, the stream's CRC-32
// against the slot table's. LiteSATA's own init (liblitesata: PHY enable, link, IDENTIFY) on the
// boards with the PHY; the simulation runs the module on a port model with no PHY. The SSD is
// written from the host: an upload streamed into `satastore` (software/host/xcu1525.py model
// --persist), or the table and boot sectors from CPU memory through LiteSATA's mem2sector DMA.

#ifdef CSR_SATA_PHY_BASE
#include <liblitesata/sata.h>
#endif

// sataup_status (gateware/memory/sata_upload.py): done, error, failed (the device's), busy.
#define SATAUP_DONE   1
#define SATAUP_ERROR  2
#define SATAUP_FAILED 4
#define SATAUP_BUSY   8

static int sata_ok;

static int sata_ops_ready(void) { return sata_ok; }

static int sata_ops_init(int fast, int tries)
{
    (void)fast;
    sata_ok = 0;
    sataup_upload_write(0);
#ifdef CSR_SATA_PHY_BASE
    // liblitesata: the PHY reset and enabled, the link waited for, IDENTIFY read (the device's
    // model and capacity printed when asked for more than a quick probe); ~0.3 s with a device,
    // 3 s without.
    sata_ok = sata_init(tries > 3);
#else
    sata_ok = 1;                                   // The simulation's port model.
#endif
    if (!sata_ok) {
        if (tries > 3) puts("satainit: no SATA device answers (link or IDENTIFY)");
        return -1;
    }
    puts("SATAINIT ok");
    return 0;
}

// `blocks` sectors from `sector` into channel `chan`'s window at `base`, through the loader.
static int sata_read_to_loader(uint64_t sector, uint32_t blocks, int chan, uint32_t base)
{
    if (!sata_ok) { puts("sataload: satainit first"); return -1; }
    if (storage_arm_loader(chan, base, blocks) < 0)
        return -1;
    sataup_upload_write(1);
    sataup_sector_write(sector);
    sataup_nsectors_write(blocks);
    sataup_start_write(1);
    // The transfer at no less than 20 MB/s, plus a second (the device's own latency).
    uint64_t t0 = perf_now64();
    uint64_t limit = ((uint64_t)blocks*512/20 + 1000000u)*(CONFIG_CLOCK_FREQUENCY/1000000);
    int timeout = 0;
    uint32_t st;
    while ((st = sataup_status_read()) & SATAUP_BUSY)
        if (perf_now64() - t0 > limit) { timeout = 1; break; }
    int error   = !timeout && ((st & SATAUP_ERROR) || !(st & SATAUP_DONE));
    int drained = !timeout && !error && storage_wait_loader() == 0;
    if (timeout)
        sataup_abort_write(1);
    sataup_upload_write(0);
    if (timeout || error || !drained) {
        printf("sataload: %s at sector %lu (%lu of %lu sectors, loader status %08lx)\n",
               timeout ? "timeout" : (st & SATAUP_FAILED) ? "the device refused the read" :
               error ? "the transfer ended in error" : "the loader did not finish",
               (unsigned long)sector, (unsigned long)sataup_sectors_done_read(), (unsigned long)blocks,
               (unsigned long)trixium_loader_status_read());
        engine_recover();                          // A short transfer leaves the loader armed.
        return -1;
    }
    return 0;
}

// Writes from CPU memory only (the boot sector, a table): LiteSATA's mem2sector DMA. The bulk
// (a model's entries from the DRAM windows, which the CPU does not reach) comes from the host.
static int sata_ops_write(uint64_t sector, uint32_t blocks, int chan, uint32_t base,
                          const uint32_t *cpu, uint32_t *crc)
{
    (void)chan; (void)base; (void)crc;
    if (cpu == NULL) {
        puts("satawrite: the model's data is written from the host (model --persist)");
        return -1;
    }
#ifdef CSR_SATA_MEM2SECTOR_BASE
    if (!sata_ok) { puts("satawrite: satainit first"); return -1; }
    if (sector >> 32) { puts("satawrite: sector beyond 32 bits"); return -1; }
    // The DMA reads main_ram through the bus: the words the CPU just wrote must be out of its
    // caches (liblitesata flushes after a read, not before a write).
    flush_cpu_dcache();
    flush_l2_cache();
    if (sata_write((uint32_t)sector, blocks, (uint8_t *)cpu) != 0) {
        printf("satawrite: write failed at sector %lu\n", (unsigned long)sector);
        return -1;
    }
    return 0;
#else
    puts("satawrite: no mem2sector DMA in this bitstream");
    return -1;
#endif
}

static void     sata_ops_crc_reset(void) { sataup_crc_reset_write(1); }
static uint32_t sata_ops_crc_read(void)  { return sataup_crc_read(); }

const struct storage_ops sata_storage_ops = {
    .name           = "SATA",
    .tag            = "SATA",
    .cmd            = "sata",
    .init           = sata_ops_init,
    .ready          = sata_ops_ready,
    .read_to_loader = sata_read_to_loader,
    .write          = sata_ops_write,
    .bulk_write     = 0,
    .crc_reset      = sata_ops_crc_reset,
    .crc_read       = sata_ops_crc_read,
};

#endif // CSR_SATAUP_BASE

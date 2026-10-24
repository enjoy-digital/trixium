/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium weights from an SD card
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
#include "sdload.h"
#include "tokenizer.h"

#ifdef CSR_SD_BASE

#include <libbase/crc.h>

// The card's read stream feeds the weight loader (gateware/memory/sd_upload.py): the firmware
// arms the loader for a channel window, as the host does for a UDP transfer, and a multi-block
// read (CMD18) lands there - no CPU copy. The SoC's module is `sd`, not `sdcard`: under that
// name LiteX's BIOS builds its FatFs disk on liblitesdcard, which needs the block2mem DMA this
// design does not have (the link fails). So the commands are here, after liblitesdcard's: init
// without its two data reads into CPU memory (the switch status, the SCR; the gateware drops
// that data): 4-bit assumed, which every SD memory card supports.

#define SD_OK       0
#define SD_CRCERROR 1
#define SD_TIMEOUT  2
#define SD_CMD_RESPONSE_SIZE 16
#define RSP_NONE  0
#define RSP_SHORT 1
#define RSP_LONG  2
#define RSP_BUSY  3
#define RSP_CRC   4
#define DATA_READ  (1 << 5)
#define DATA_WRITE (2 << 5)
#define PHY_SPEED_1X 0
#define PHY_SPEED_4X 1
#define PHY_SPEED_8X 2

static int sd_ok;

static int sd_wait(uint32_t (*event)(void))
{
    uint32_t e;
    while (!((e = event()) & 1))
        busy_wait_us(10);
    return (e & 4) ? SD_TIMEOUT : (e & 8) ? SD_CRCERROR : SD_OK;
}

static int sd_cmd(uint32_t arg, int cmd, int rsp)
{
    sd_core_cmd_argument_write(arg);
    sd_core_cmd_command_write((cmd << 8) | rsp);
    sd_core_cmd_send_write(1);
    return sd_wait(sd_core_cmd_event_read);
}

static int sd_cmd_retry(uint32_t arg, int cmd, int rsp)
{
    int rc = SD_TIMEOUT;
    for (int i = 0; i < 16 && rc != SD_OK; i++)
        rc = sd_cmd(arg, cmd, rsp);
    return rc;
}

static void sd_clk(uint32_t hz)
{
    uint32_t div = (CONFIG_CLOCK_FREQUENCY + hz - 1)/hz;
    sd_phy_clocker_divider_write(div < 2 ? 2 : div > 256 ? 256 : div);
}

#ifndef TRIXIUM_SD_EMMC
// CMD6 (switch to high speed): its 64-byte status reply is dropped by the gateware.
static int sd_switch_hs(void)
{
    sd_core_block_length_write(64);
    sd_core_block_count_write(1);
    if (sd_cmd_retry(0x80fffff1, 6, DATA_READ | RSP_SHORT | RSP_CRC) != SD_OK)
        return SD_TIMEOUT;
    return sd_wait(sd_core_data_event_read);
}
#endif

// The bus width (SD: ACMD6; eMMC: CMD6 SWITCH into EXT_CSD BUS_WIDTH) and the PHY's setting.
static int sd_bus_width(uint32_t rca, int width)
{
#ifdef TRIXIUM_SD_EMMC
    (void)rca;
    uint32_t bw = width == 8 ? 2 : width == 4 ? 1 : 0;
    if (sd_cmd((3u << 24) | (183u << 16) | (bw << 8), 6, RSP_BUSY | RSP_CRC) != SD_OK)
        return SD_TIMEOUT;
#else
    if (sd_cmd(rca << 16, 55, RSP_SHORT | RSP_CRC) != SD_OK ||
        sd_cmd(width == 4 ? 2 : 0, 6, RSP_SHORT | RSP_CRC) != SD_OK)
        return SD_TIMEOUT;
#endif
    sd_phy_settings_write(width == 8 ? PHY_SPEED_8X : width == 4 ? PHY_SPEED_4X : PHY_SPEED_1X);
    return SD_OK;
}

// One block read of sector 0 (the gateware drops it: sd_upload is clear).
static int sd_read_check(void)
{
    sd_core_block_length_write(512);
    sd_core_block_count_write(1);
    int rc = sd_cmd_retry(0, 17, DATA_READ | RSP_SHORT | RSP_CRC);
    if (rc == SD_OK)
        rc = sd_wait(sd_core_data_event_read);
    if (rc != SD_OK)
        (void)sd_cmd(0, 12, RSP_BUSY | RSP_CRC);              // Back to transfer state.
    return rc;
}

int sd_init_done(void) { return sd_ok; }

int sd_init(int mhz, int tries)
{
    int force_1bit = mhz < 0;
    if (mhz < 0) mhz = -mhz;
    uint32_t r[SD_CMD_RESPONSE_SIZE/4];
    int timeout;
    sd_ok = 0;
    sd_upload_write(0);
    sd_phy_settings_write(PHY_SPEED_1X);
    sd_clk(400000);
#ifdef TRIXIUM_SD_EMMC
    // eMMC (CMD1 instead of CMD8/ACMD41, the RCA assigned by the host, EXT_CSD switches).
    uint32_t rca = 1;
    int answered = 0;
    sd_phy_init_initialize_write(1);               // 80 clocks.
    busy_wait(1);
    (void)sd_cmd(0, 0, RSP_NONE);                  // CMD0: idle.
    for (timeout = 1000; timeout > 0; timeout--) {
        if (sd_cmd(0x40ff8080, 1, RSP_SHORT) == SD_OK) {   // CMD1: sector mode, 1.8/3.3 V.
            answered = 1;
            csr_rd_buf_uint32(CSR_SD_CORE_CMD_RESPONSE_ADDR, r, SD_CMD_RESPONSE_SIZE/4);
            if (r[0] & 0x80000000)                 // Power-up done.
                break;
        } else if (!answered && 1000 - timeout >= tries - 1) {
            if (tries > 3) puts("sdinit: no eMMC answers (CMD1)");
            return -1;
        }
        busy_wait(1);
    }
    if (!timeout) { puts("sdinit: the eMMC stays busy (CMD1)"); return -1; }
    if (((r[0] >> 29) & 3) != 2) {
        puts("sdinit: a byte-addressed eMMC (2 GB or less): not supported");
        return -1;
    }
    if (sd_cmd(0, 2, RSP_LONG | RSP_CRC) != SD_OK ||              // CMD2: CID.
        sd_cmd(rca << 16, 3, RSP_SHORT | RSP_CRC) != SD_OK ||     // CMD3: set RCA.
        sd_cmd(rca << 16, 9, RSP_LONG | RSP_CRC) != SD_OK ||      // CMD9: CSD.
        sd_cmd(rca << 16, 7, RSP_BUSY | RSP_CRC) != SD_OK) {      // CMD7: select.
        puts("sdinit: eMMC identification failed");
        return -1;
    }
    sd_clk(25000000);
    // High speed timing (EXT_CSD HS_TIMING): up to 52 MHz.
    int hs = mhz > 25 && sd_cmd((3u << 24) | (185u << 16) | (1u << 8), 6, RSP_BUSY | RSP_CRC) == SD_OK;
    const int widest = 8;
#else
    for (timeout = tries; timeout > 0; timeout--) {
        busy_wait(1);
        sd_phy_init_initialize_write(1);           // 80 clocks.
        busy_wait(1);
        (void)sd_cmd(0, 0, RSP_NONE);              // CMD0: idle.
        busy_wait(1);
        if (sd_cmd(0x1aa, 8, RSP_SHORT | RSP_CRC) == SD_OK)   // CMD8: a v2.00+ card.
            break;
    }
    if (!timeout) { if (tries > 3) puts("sdinit: no card answers (CMD8)"); return -1; }
    sd_clk(25000000);
    busy_wait(1);
    for (timeout = 1000; timeout > 0; timeout--) {
        sd_cmd(0, 55, RSP_SHORT | RSP_CRC);        // ACMD41, high capacity.
        if (sd_cmd(0x70ff8000, 41, RSP_SHORT) == SD_OK) {
            csr_rd_buf_uint32(CSR_SD_CORE_CMD_RESPONSE_ADDR, r, SD_CMD_RESPONSE_SIZE/4);
            if (r[0] & 0x80000000)                 // Power-up done.
                break;
        }
        busy_wait(1);
    }
    if (!timeout) { puts("sdinit: the card stays busy (ACMD41)"); return -1; }
    if (!((r[0] >> 30) & 1)) {
        puts("sdinit: a standard-capacity card (byte addresses): use an SDHC/SDXC card");
        return -1;
    }
    if (sd_cmd(0, 2, RSP_LONG | RSP_CRC) != SD_OK ||          // CMD2: CID.
        sd_cmd(0, 3, RSP_SHORT | RSP_CRC) != SD_OK) {         // CMD3: RCA.
        puts("sdinit: identification failed");
        return -1;
    }
    csr_rd_buf_uint32(CSR_SD_CORE_CMD_RESPONSE_ADDR, r, SD_CMD_RESPONSE_SIZE/4);
    uint32_t rca = (r[0] >> 16) & 0xffff;
    if (sd_cmd(rca << 16, 9, RSP_LONG | RSP_CRC) != SD_OK ||  // CMD9: CSD.
        sd_cmd(rca << 16, 7, RSP_BUSY | RSP_CRC) != SD_OK) {  // CMD7: select.
        puts("sdinit: selection failed");
        return -1;
    }
    // High speed (50 MHz) when asked for and the card takes the switch; 25 MHz otherwise.
    int hs = mhz > 25 && sd_switch_hs() == SD_OK;
    const int widest = 4;
#endif
    if (sd_cmd(512, 16, RSP_SHORT | RSP_CRC) != SD_OK) {      // CMD16: 512-byte blocks.
        puts("sdinit: block length refused");
        return -1;
    }
    sd_clk(hs ? (uint32_t)mhz*1000000u : 25000000u);
    // The widest bus that reads one block back (dropped): the Ti375 kit's microSD slot switches
    // to 4-bit but DAT2 never reaches the FPGA (software/lab/ti375_sd.py), so 1-bit there.
    int width = force_1bit ? 1 : widest;
    while (sd_bus_width(rca, width) != SD_OK || sd_read_check() != SD_OK) {
        if (width == 1) {
            puts("sdinit: no block reads at any bus width");
            return -1;
        }
        width = width == 8 ? 4 : 1;
    }
    sd_ok = 1;
    printf("SDINIT ok, %d-bit at %d MHz\n", width, hs ? mhz : 25);
    return 0;
}

int sd_load(int chan, uint32_t base, uint32_t sector, uint32_t blocks)
{
    if (!sd_ok) { puts("sdload: sdinit first"); return -1; }
    if (blocks == 0 || blocks > (1u << 24)/16) { puts("sdload: 1..1M blocks"); return -1; }
    trixium_loader_src_sel_write(0);
    trixium_loader_chan_write(chan);
    trixium_loader_base_write(base);
    trixium_loader_length_write(blocks*16);       // 512-byte blocks in 32-byte beats.
    trixium_loader_resync_write(0);
    trixium_loader_start_write(1);
    sd_upload_write(1);
    sd_core_block_length_write(512);
    sd_core_block_count_write(blocks);
    int rc = sd_cmd_retry(sector, 18, DATA_READ | RSP_SHORT | RSP_CRC);   // CMD18.
    if (rc == SD_OK)
        rc = sd_wait(sd_core_data_event_read);
    (void)sd_cmd(0, 12, RSP_BUSY | RSP_CRC);                              // CMD12: stop.
    // The loader drains its last beats into the memory (its writers' responses).
    for (int t = 0; t < 100000 && !(trixium_loader_status_read() & 1); t++)
        busy_wait_us(10);
    sd_upload_write(0);
    if (rc != SD_OK || !(trixium_loader_status_read() & 1)) {
        printf("sdload: %s (loader status %08lx, %lu beats armed)\n",
               rc == SD_CRCERROR ? "CRC error on the card's data" : rc != SD_OK ? "read timeout" :
               "the loader did not finish", (unsigned long)trixium_loader_status_read(),
               (unsigned long)(blocks*16));
        return -1;
    }
    return 0;
}

// Writes: blocks from a DRAM window (through the engine into the scratchpad) or from CPU memory
// (cpu != NULL), pushed word by word into the card's write FIFO (sd_upload.py's wdata).
#define SD_WFIFO_DEPTH 1024

static void sd_push(const volatile uint32_t *w, uint32_t n)
{
    uint32_t room = 0;
    for (uint32_t i = 0; i < n; i++) {
        while (room == 0)
            room = SD_WFIFO_DEPTH - sd_wlevel_read();
        sd_wdata_write(w[i]);
        room--;
    }
}

// CRC-32 (zlib's) a byte at a time from a 1 KB table: ~5 cycles a byte while sdsave pushes.
static uint32_t crc_tab[256];
static uint32_t crc_update(uint32_t crc, const volatile uint32_t *w, uint32_t words)
{
    if (!crc_tab[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0xedb88320u : c >> 1;
            crc_tab[i] = c;
        }
    for (uint32_t i = 0; i < words; i++) {
        uint32_t v = w[i];                       // The bytes in the order the write FIFO sends them.
        for (int b = 0; b < 4; b++, v >>= 8)
            crc = crc_tab[(crc ^ v) & 0xff] ^ (crc >> 8);
    }
    return crc;
}

static uint32_t *write_crc;                      // sd_write: the running CRC of what it pushes.

int sd_write(uint32_t sector, uint32_t blocks, int chan, uint32_t base, const uint32_t *cpu)
{
    if (!sd_ok) { puts("sdwrite: sdinit first"); return -1; }
    sd_core_block_length_write(512);
    sd_core_block_count_write(blocks);
    if (sd_cmd(sector, blocks > 1 ? 25 : 24, DATA_WRITE | RSP_SHORT | RSP_CRC) != SD_OK) {
        puts("sdwrite: write command refused");
        return -1;
    }
    uint32_t bytes = blocks*512;
    for (uint32_t off = 0; off < bytes; off += 0x10000) {
        uint32_t chunk = bytes - off > 0x10000 ? 0x10000 : bytes - off;
        const volatile uint32_t *src = (const volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
        if (cpu)
            src = cpu + off/4;
        else
            engine_fetch(chan, base + off, chunk/32, 0xC000);
        sd_push(src, chunk/4);
        if (write_crc)
            *write_crc = crc_update(*write_crc, src, chunk/4);
    }
    int rc = sd_wait(sd_core_data_event_read);
    if (blocks > 1)
        (void)sd_cmd(0, 12, RSP_BUSY | RSP_CRC);
    if (rc != SD_OK) {
        printf("sdwrite: %s at sector %lu\n", rc == SD_CRCERROR ? "the card refused a block (CRC)" :
               "write timeout", (unsigned long)sector);
        return -1;
    }
    return 0;
}

#if TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000   // The staging areas below the weights.

// One 512-byte sector through the loader into `t` (the table's landing place on channel 0).
static int sd_read_sector(uint32_t sector, uint32_t t[128])
{
    uint32_t scratch = TRIXIUM_CHAN_WEIGHT_BASE - SD_TABLE_STAGING_BELOW;
    if (sd_load(0, scratch, sector, 1) < 0)
        return -1;
    engine_fetch(0, scratch, 16, 0xC000);
    for (int i = 0; i < 128; i++)
        t[i] = ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i];
    return 0;
}

static uint32_t slot_sector(int slot) { return SD_IMAGE_SECTOR + (uint32_t)slot*SD_SLOT_SECTORS; }

// A slot's table: 0 and *v1 = 0/1 when valid (version 2: label and per-entry CRCs; 1: neither).
static int table_ok(const uint32_t *t, int *v1)
{
    if (t[0] != SD_IMAGE_MAGIC || t[2] == 0)
        return -1;
    if (t[1] == 1 && t[2] <= SD_IMAGE_MAX_ENTRIES && crc32((unsigned char *)&t[4], t[2]*16) == t[3]) {
        *v1 = 1;
        return 0;
    }
    if (t[1] == 2 && t[2] <= SD_V2_MAX_ENTRIES && crc32((unsigned char *)&t[12], t[2]*20) == t[3]) {
        *v1 = 0;
        return 0;
    }
    return -1;
}

int sd_save(int slot, const char *label)
{
    if (slot < 0 || slot >= SD_MAX_SLOTS) { printf("sdsave: slot 0..%d\n", SD_MAX_SLOTS - 1); return -1; }
    if (!sd_ok && sd_init(50, 1000) < 0)
        return -1;
    // What a UDP upload staged: the BONS header+table, each channel's payload, the tokenizer.
    _Static_assert(sizeof(struct bons_header) <= 512, "the header in 16 beats");
    engine_fetch(0, TRIXIUM_CHAN_WEIGHT_BASE - 0x100000, 16, 0xC000);
    struct bons_header h;
    for (uint32_t i = 0; i < sizeof(h)/4; i++)
        ((uint32_t *)&h)[i] = ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i];
    if (h.magic != BONS_MAGIC) { puts("sdsave: no staged model (upload one first)"); return -1; }
    uint32_t t[128];
    memset(t, 0, sizeof(t));
    uint32_t n = 0, sector = slot_sector(slot) + 1;
#define SD_ENTRY(c, b, bytes) do { uint32_t *x = &t[12 + 5*n++]; x[0] = (c); x[1] = (b); \
        x[2] = ((bytes) + 511)/512; x[3] = sector; sector += x[2]; } while (0)
    SD_ENTRY(0, TRIXIUM_CHAN_WEIGHT_BASE - 0x100000, h.payload_offset);
    for (uint32_t c = 0; c < h.n_channels; c++)
        SD_ENTRY(c, TRIXIUM_CHAN_WEIGHT_BASE
#ifndef TRIXIUM_LOADER_ADDS_CHAN_OFFSET
                    + c*TRIXIUM_CHAN_STRIDE
#endif
                 , (uint32_t)(h.payload_beats*32));
    engine_fetch(0, TOK_STAGING, 1, 0xC000);
    volatile uint32_t *rec = (volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000);
    if (rec[0] == TOK_STAGING_MAGIC && rec[1] <= TOK_STAGING_MAX)
        SD_ENTRY(0, TOK_STAGING, 32 + rec[1]);
    if (sector - slot_sector(slot) > SD_SLOT_SECTORS) { puts("sdsave: the model is larger than a slot"); return -1; }
    t[0] = SD_IMAGE_MAGIC; t[1] = 2; t[2] = n;
    for (int i = 0; i < 31 && label && label[i]; i++)
        ((char *)&t[4])[i] = label[i];
    // The old table first invalidated, the table last: a save cut short leaves no image to boot.
    uint32_t zero[128];
    memset(zero, 0, sizeof(zero));
    if (sd_write(slot_sector(slot), 1, 0, 0, zero) < 0)
        return -1;
    uint64_t t0 = perf_now64(), bytes = 0;
    for (uint32_t e = 0; e < n; e++) {
        uint32_t *x = &t[12 + 5*e], crc = 0xffffffffu;
        printf("  chan %lu: %lu KB to sector %lu\n", (unsigned long)x[0], (unsigned long)(x[2]/2),
               (unsigned long)x[3]);
        write_crc = &crc;
        int rc = sd_write(x[3], x[2], (int)x[0], x[1], NULL);
        write_crc = NULL;
        if (rc < 0)
            return -1;
        x[4] = ~crc;
        bytes += (uint64_t)x[2]*512;
    }
    t[3] = crc32((unsigned char *)&t[12], n*20);
    if (sd_write(slot_sector(slot), 1, 0, 0, t) < 0)
        return -1;
    uint32_t ms = (uint32_t)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000));
    printf("SDSAVE slot %d \"%s\": %lu MB in %lu s (%lu KB/s)\n", slot, (char *)&t[4],
           (unsigned long)(bytes >> 20), (unsigned long)(ms/1000), (unsigned long)(ms ? bytes/ms : 0));
    return 0;
}

int sd_boot(int slot)
{
    if (slot < 0 || slot >= SD_MAX_SLOTS) { printf("sdboot: slot 0..%d\n", SD_MAX_SLOTS - 1); return -1; }
    if (!sd_ok && sd_init(50, 1000) < 0)
        return -1;
    uint32_t t[128];
    int v1;
    if (sd_read_sector(slot_sector(slot), t) < 0)
        return -1;
    if (table_ok(t, &v1) < 0) {
        printf("sdboot: no TriXium image in slot %d (sector %lu, magic %08lx)\n", slot,
               (unsigned long)slot_sector(slot), (unsigned long)t[0]);
        return -1;
    }
    uint64_t t0 = perf_now64(), bytes = 0;
    int checked = 0;
    for (uint32_t e = 0; e < t[2]; e++) {
        const uint32_t *x = v1 ? &t[4 + 4*e] : &t[12 + 5*e];   // chan, base, blocks, sector[, crc]
        printf("  chan %lu: %lu KB from sector %lu\n", (unsigned long)x[0],
               (unsigned long)(x[2]/2), (unsigned long)x[3]);
#ifdef CSR_SD_CRC_ADDR
        sd_crc_reset_write(1);
#endif
        if (sd_load((int)x[0], x[1], x[3], x[2]) < 0)
            return -1;
#ifdef CSR_SD_CRC_ADDR
        // The data went to the DRAM without the CPU: the read stream's CRC against the table's.
        if (!v1) {
            uint32_t got = sd_crc_read();
            if (got != x[4]) {
                printf("sdboot: entry %lu CRC %08lx, the table says %08lx: the device's data is damaged\n",
                       (unsigned long)e, (unsigned long)got, (unsigned long)x[4]);
                return -1;
            }
            checked++;
        }
#endif
        bytes += (uint64_t)x[2]*512;
    }
    uint32_t ms = (uint32_t)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000));
    printf("SDBOOT slot %d \"%s\": %lu MB in %lu.%01lu s (%lu MB/s)%s\n", slot, v1 ? "" : (char *)&t[4],
           (unsigned long)(bytes >> 20), (unsigned long)(ms/1000), (unsigned long)(ms%1000/100),
           (unsigned long)(ms ? (bytes >> 10)/ms : 0), checked ? ", CRC ok" : "");
    return 0;
}

void sd_list(void)
{
    if (!sd_ok && sd_init(50, 1000) < 0)
        return;
    int def = sd_default_slot();
    for (int slot = 0; slot < SD_MAX_SLOTS; slot++) {
        uint32_t t[128];
        int v1;
        if (sd_read_sector(slot_sector(slot), t) < 0)
            break;                                       // Past the device's end.
        if (table_ok(t, &v1) < 0)
            continue;
        uint64_t bytes = 0;
        for (uint32_t e = 0; e < t[2]; e++)
            bytes += (uint64_t)(v1 ? t[4 + 4*e + 2] : t[12 + 5*e + 2])*512;
        printf("SDSLOT %d%s \"%s\" %lu MB, %lu entries%s\n", slot, slot == def ? " (boot)" : "",
               v1 ? "" : (char *)&t[4], (unsigned long)(bytes >> 20), (unsigned long)t[2],
               v1 ? ", version 1 (no CRCs)" : "");
    }
}

// The slot booted at power-up: the boot sector's, 0 without one.
int sd_default_slot(void)
{
    uint32_t t[128];
    if (sd_read_sector(SD_BOOT_SECTOR, t) < 0 || t[0] != SD_BOOT_MAGIC || t[2] != ~t[1] ||
        t[1] >= SD_MAX_SLOTS)
        return 0;
    return (int)t[1];
}

int sd_set_default(int slot)
{
    if (slot < 0 || slot >= SD_MAX_SLOTS) { printf("sddefault: slot 0..%d\n", SD_MAX_SLOTS - 1); return -1; }
    if (!sd_ok && sd_init(50, 1000) < 0)
        return -1;
    uint32_t t[128];
    memset(t, 0, sizeof(t));
    t[0] = SD_BOOT_MAGIC; t[1] = (uint32_t)slot; t[2] = ~(uint32_t)slot;
    if (sd_write(SD_BOOT_SECTOR, 1, 0, 0, t) < 0)
        return -1;
    printf("SDDEFAULT slot %d\n", slot);
    return 0;
}

#endif

#endif

/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium weights from an SD card / eMMC
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
#include "sdload.h"
#include "static_memory.h"
#include "storage.h"

#ifdef CSR_SD_BASE

// The card's read stream feeds the weight loader (gateware/memory/sd_upload.py): the firmware
// arms the loader for a channel window, as the host does for a UDP transfer, and a multi-block
// read (CMD18) lands there - no CPU copy. The SoC's module is `sd`, not `sdcard`: under that
// name LiteX's BIOS builds its FatFs disk on liblitesdcard, which needs the block2mem DMA this
// design does not have (the link fails). So the commands are here, after liblitesdcard's: init
// without its two data reads into CPU memory (the switch status, the SCR; the gateware drops
// that data): 4-bit assumed, which every SD memory card supports. The slots and tables are
// storage.c's (the same on a SATA SSD); this file is the device.

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
    if (storage_arm_loader(chan, base, blocks) < 0)
        return -1;
    sd_upload_write(1);
    sd_core_block_length_write(512);
    sd_core_block_count_write(blocks);
    int rc = sd_cmd_retry(sector, 18, DATA_READ | RSP_SHORT | RSP_CRC);   // CMD18.
    if (rc == SD_OK)
        rc = sd_wait(sd_core_data_event_read);
    (void)sd_cmd(0, 12, RSP_BUSY | RSP_CRC);                              // CMD12: stop.
    int drained = storage_wait_loader() == 0;
    sd_upload_write(0);
    if (rc != SD_OK || !drained) {
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


// CRC-32 (zlib's), slicing-by-4: one 32-bit word per 4 lookups (4 KB of tables in the main_ram
// static window) while sdsave pushes; a byte-at-a-time table loop made the save 108 s, not 39.
static uint32_t (*const crc_tab)[256] = (uint32_t (*)[256])TRIXIUM_CRC_TABLES_ADDR;
static int crc_ready;
static uint32_t crc_update(uint32_t crc, const volatile uint32_t *w, uint32_t words)
{
    if (!crc_ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0xedb88320u : c >> 1;
            crc_tab[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; i++)
            for (int t = 1; t < 4; t++)
                crc_tab[t][i] = (crc_tab[t - 1][i] >> 8) ^ crc_tab[0][crc_tab[t - 1][i] & 0xff];
        crc_ready = 1;
    }
    for (uint32_t i = 0; i < words; i++) {
        uint32_t v = crc ^ w[i];                 // Little-endian: the bytes the write FIFO sends.
        crc = crc_tab[3][v & 0xff] ^ crc_tab[2][(v >> 8) & 0xff] ^
              crc_tab[1][(v >> 16) & 0xff] ^ crc_tab[0][v >> 24];
    }
    return crc;
}

static uint32_t *write_crc;                      // sd_write: the running CRC of what it pushes.

// Words into the card's write FIFO (CSR writes), each read once: the running CRC (sdsave) on the
// same value (re-reading the uncached scratchpad for the CRC cost 45 s of the 2B's save).
static void sd_push(const volatile uint32_t *w, uint32_t n)
{
    uint32_t room = 0, crc = write_crc ? *write_crc : 0;
    crc_update(0, w, 0);                         // The tables, once.
    for (uint32_t i = 0; i < n; i++) {
        while (room == 0)
            room = SD_WFIFO_DEPTH - sd_wlevel_read();
        uint32_t x = w[i];
        sd_wdata_write(x);
        room--;
        if (write_crc) {
            uint32_t v = crc ^ x;
            crc = crc_tab[3][v & 0xff] ^ crc_tab[2][(v >> 8) & 0xff] ^
                  crc_tab[1][(v >> 16) & 0xff] ^ crc_tab[0][v >> 24];
        }
    }
    if (write_crc)
        *write_crc = crc;
}

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

// The device, for storage.c's slots and tables.
static int sd_ops_init(int fast, int tries) { return sd_init(fast ? 50 : 25, tries); }

static int sd_ops_read_to_loader(uint64_t sector, uint32_t blocks, int chan, uint32_t base)
{
    if (sector >> 32) { puts("sdload: sector beyond 32 bits"); return -1; }
    return sd_load(chan, base, (uint32_t)sector, blocks);
}

static int sd_ops_write(uint64_t sector, uint32_t blocks, int chan, uint32_t base,
                        const uint32_t *cpu, uint32_t *crc)
{
    if (sector >> 32) { puts("sdwrite: sector beyond 32 bits"); return -1; }
    write_crc = crc;
    int rc = sd_write((uint32_t)sector, blocks, chan, base, cpu);
    write_crc = NULL;
    return rc;
}

#ifdef CSR_SD_CRC_ADDR
static void     sd_ops_crc_reset(void) { sd_crc_reset_write(1); }
static uint32_t sd_ops_crc_read(void)  { return sd_crc_read(); }
#endif

const struct storage_ops sd_storage_ops = {
    .name           = "SD card / eMMC",
    .tag            = "SD",
    .cmd            = "sd",
    .init           = sd_ops_init,
    .ready          = sd_init_done,
    .read_to_loader = sd_ops_read_to_loader,
    .write          = sd_ops_write,
    .bulk_write     = 1,
#ifdef CSR_SD_CRC_ADDR
    .crc_reset      = sd_ops_crc_reset,
    .crc_read       = sd_ops_crc_read,
#endif
};

#endif // CSR_SD_BASE

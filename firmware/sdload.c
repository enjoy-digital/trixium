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

#include "engine.h"
#include "perf.h"
#include "sdload.h"

#ifdef CSR_SDCARD_BASE

#include <libbase/crc.h>
#include <liblitesdcard/sdcard.h>

// The card's read stream feeds the weight loader (gateware/memory/sd_upload.py): the firmware
// arms the loader for a channel window, as the host does for a UDP transfer, and a multi-block
// read (CMD18) lands there - no CPU copy. Init is liblitesdcard's sequence without its two data
// reads into CPU memory (the switch status, the SCR: they need the DMA this design does not
// have, the gateware drops that data): 4-bit assumed, which every SD memory card supports.

static int sd_ok;

int sd_init(int mhz, int tries)
{
    uint32_t r[SD_CMD_RESPONSE_SIZE/4];
    int timeout;
    sd_ok = 0;
    sdcard_upload_write(0);
    sdcard_phy_settings_write(SD_PHY_SPEED_1X);
    sdcard_set_clk_freq(400000, 0);
    for (timeout = tries; timeout > 0; timeout--) {
        busy_wait(1);
        sdcard_phy_init_initialize_write(1);       // 80 clocks.
        busy_wait(1);
        (void)sdcard_go_idle();
        busy_wait(1);
        if (sdcard_send_ext_csd() == SD_OK)        // CMD8: a v2.00+ card.
            break;
    }
    if (!timeout) { if (tries > 3) puts("sdinit: no card answers (CMD8)"); return -1; }
    sdcard_set_clk_freq(25000000, 0);
    busy_wait(1);
    for (timeout = 1000; timeout > 0; timeout--) {
        sdcard_app_cmd(0);
        if (sdcard_app_send_op_cond(1) == SD_OK) {
            csr_rd_buf_uint32(CSR_SDCARD_CORE_CMD_RESPONSE_ADDR, r, SD_CMD_RESPONSE_SIZE/4);
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
    if (sdcard_all_send_cid() != SD_OK || sdcard_set_relative_address() != SD_OK) {
        puts("sdinit: identification failed");
        return -1;
    }
    uint16_t rca = sdcard_decode_rca();
    if (sdcard_send_cid(rca) != SD_OK || sdcard_send_csd(rca) != SD_OK ||
        sdcard_select_card(rca) != SD_OK) {
        puts("sdinit: selection failed");
        return -1;
    }
    // High speed (50 MHz) when asked for and the card takes the switch; 25 MHz otherwise.
    int hs = mhz > 25 && sdcard_switch(SD_SWITCH_SWITCH, SD_GROUP_ACCESSMODE, SD_SPEED_SDR25) == SD_OK;
    if (sdcard_app_cmd(rca) != SD_OK || sdcard_app_set_bus_width() != SD_OK) {
        puts("sdinit: 4-bit bus refused");
        return -1;
    }
    sdcard_phy_settings_write(SD_PHY_SPEED_4X);
    if (sdcard_app_set_blocklen(512) != SD_OK) {
        puts("sdinit: block length refused");
        return -1;
    }
    sdcard_set_clk_freq(hs ? (uint32_t)mhz*1000000u : 25000000u, 0);
    sd_ok = 1;
    printf("SDINIT ok, 4-bit at %d MHz\n", hs ? mhz : 25);
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
    sdcard_upload_write(1);
    int rc = sdcard_read_multiple_block(sector, blocks);
    (void)sdcard_stop_transmission();
    // The loader drains its last beats into the memory (its writers' responses).
    for (int t = 0; t < 100000 && !(trixium_loader_status_read() & 1); t++)
        busy_wait_us(10);
    sdcard_upload_write(0);
    if (rc != SD_OK || !(trixium_loader_status_read() & 1)) {
        printf("sdload: %s (loader status %08lx, %lu beats armed)\n",
               rc == SD_CRCERROR ? "CRC error on the card's data" : rc != SD_OK ? "read timeout" :
               "the loader did not finish", (unsigned long)trixium_loader_status_read(),
               (unsigned long)(blocks*16));
        return -1;
    }
    return 0;
}

#if TRIXIUM_CHAN_WEIGHT_BASE >= 0x2000000   // The staging areas below the weights.
int sd_boot(void)
{
    if (!sd_ok && sd_init(50, 1000) < 0)
        return -1;
    // The image's table, through the loader to a scratch place on channel 0, then the scratchpad.
    uint32_t scratch = TRIXIUM_CHAN_WEIGHT_BASE - SD_TABLE_STAGING_BELOW;
    if (sd_load(0, scratch, SD_IMAGE_SECTOR, 1) < 0)
        return -1;
    engine_fetch(0, scratch, 16, 0xC000);
    uint32_t t[128];
    for (int i = 0; i < 128; i++)
        t[i] = ((volatile uint32_t *)(SCRATCHPAD_BASE + 0xC000))[i];
    if (t[0] != SD_IMAGE_MAGIC || t[1] != 1 || t[2] == 0 || t[2] > SD_IMAGE_MAX_ENTRIES ||
        crc32((unsigned char *)&t[4], t[2]*16) != t[3]) {
        printf("sdboot: no TriXium image at sector %u (magic %08lx)\n", SD_IMAGE_SECTOR,
               (unsigned long)t[0]);
        return -1;
    }
    uint64_t t0 = perf_now64(), bytes = 0;
    for (uint32_t e = 0; e < t[2]; e++) {
        const uint32_t *x = &t[4 + 4*e];         // chan, base, blocks, sector
        printf("  chan %lu: %lu KB from sector %lu\n", (unsigned long)x[0],
               (unsigned long)(x[2]/2), (unsigned long)x[3]);
        if (sd_load((int)x[0], x[1], x[3], x[2]) < 0)
            return -1;
        bytes += (uint64_t)x[2]*512;
    }
    uint32_t ms = (uint32_t)((perf_now64() - t0)/(CONFIG_CLOCK_FREQUENCY/1000));
    printf("SDBOOT %lu MB in %lu.%01lu s (%lu MB/s)\n", (unsigned long)(bytes >> 20),
           (unsigned long)(ms/1000), (unsigned long)(ms%1000/100),
           (unsigned long)(ms ? (bytes >> 10)/ms : 0));
    return 0;
}

#endif

#endif

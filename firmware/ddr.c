/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium extra DDR controllers (the XCU1525's four DDR4 channels, the YPCB's two DDR3)
 *
 * LiteX's own liblitedram, compiled once more here with the CSR base as a runtime variable.
 *
 * The BIOS initializes controller 0 (main_ram) the standard way. Controllers 1..N-1 have no CPU
 * window and no BIOS support; they have the same PHY, module and init sequence, and their CSRs
 * sit at controller 0's offsets shifted by k*TRIXIUM_DDR_CSR_STRIDE (the target pins the
 * (sdram, ddrphy) pairs to consecutive CSR pages). csr.h's accessors are all
 * `CSR_BASE + offset`, so redefining CSR_BASE as a variable for this translation unit turns the
 * unmodified sdram_init() (init sequence, write leveling, read leveling) into init(k). The
 * calibration never touches the CPU's memory bus (it goes through the DFI injector's CSRs),
 * which is why a controller without a CPU window can be calibrated at all. sdram_init()'s final
 * memtest would use main_ram, so it is compiled out and replaced by ddr_test(): an engine
 * STORE/FETCH round trip through every channel's own window, the path the weights take.
 *
 * With SDRAM_CSR_STRIDE (the YPCB, CK phase leveling: SDRAM_PHY_CKPHASE_STEPS), liblitedram
 * switches the CSR base itself and the BIOS levels every controller at boot (SDRAM_CONTROLLERS);
 * ddr_init(k) re-levels controller k with sdram_ckphase_leveling(k).
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

// The include order below is the mechanism: generated/mem.h's literal CSR_BASE is captured
// first, then CSR_BASE becomes a variable before csr.h and liblitedram are compiled.
#include <generated/soc.h>

#ifdef TRIXIUM_N_MEMCTRL

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifndef SDRAM_CSR_STRIDE
#include <generated/mem.h>              // CSR_BASE, the literal.
static const unsigned long ddr_csr_base0 = CSR_BASE;
#undef CSR_BASE
unsigned long ddr_csr_base;             // The controller being talked to.
#define CSR_BASE ddr_csr_base
#endif
// No csr.h (or sdram_phy.h, which includes it) before sdram.c: its inline accessors would bind
// to the literal base (with SDRAM_CSR_STRIDE, sdram.c includes it after its own CSR_BASE).
#define SDRAM_TEST_DISABLE              // No main_ram memtest at the end of sdram_init().
#include <liblitedram/sdram.c>          // The BIOS's init and calibration, at our base.
#include <liblitedram/accessors.c>      // The PHY delay accessors the leveling uses.
#if defined(SDRAM_CSR_STRIDE) && !defined(SDRAM_PHY_CKPHASE_STEPS)
#error "SDRAM_CSR_STRIDE: liblitedram levels other controllers with CK phase leveling only"
#endif

#include "engine.h"
#include "ddr.h"

#ifndef TRIXIUM_CHAN_MOD
#define TRIXIUM_CHAN_MOD 0
#endif

/* Init */
/*------*/

int ddr_init(int k)
{
    if (k < 0 || k >= TRIXIUM_N_MEMCTRL) {
        printf("ddr: controller %d does not exist (0..%d)\n", k, TRIXIUM_N_MEMCTRL - 1);
        return 0;
    }
#ifdef SDRAM_CSR_STRIDE
    // The leveling takes the controller out of hardware control for seconds: never the one the
    // firmware runs from (the BIOS levels controller 0).
    if (k == 0) {
        printf("ddr: controller 0 holds main_ram (the BIOS levels it)\n");
        return 0;
    }
    printf("ddr: controller %d\n", k);
    int ok = sdram_ckphase_leveling(k);
#else
    ddr_csr_base = ddr_csr_base0 + (unsigned long)k*TRIXIUM_DDR_CSR_STRIDE;
    printf("ddr: controller %d (CSRs at %08lx)\n", k, (unsigned long)CSR_SDRAM_BASE);
    int ok = sdram_init();
    ddr_csr_base = ddr_csr_base0;
#endif
    printf("ddr: controller %d %s\n", k, ok ? "OK" : "FAILED");
    return ok;
}

int ddr_init_all(void)
{
    int ok = 1;
    for (int k = 1; k < TRIXIUM_N_MEMCTRL; k++)
        ok &= ddr_init(k);
    return ok;
}

int ddr_boot_init(void)
{
#ifdef SDRAM_CSR_STRIDE
    // The BIOS leveled every controller (SDRAM_CONTROLLERS) before the firmware was loaded.
    printf("ddr: controllers 0..%d leveled by the BIOS\n", TRIXIUM_N_MEMCTRL - 1);
    return 1;
#else
    return ddr_init_all();
#endif
}

#if defined(SDRAM_CSR_STRIDE) && defined(SDRAM_PHY_CKPHASE_STEPS)
// Diagnostic: re-apply the last leveling's per-bit read taps (controller k, which must be the
// last one leveled) shifted by `off` taps; off 0 restores them.
void ddr_tap_offset(int k, int off)
{
    sdram_csr_base = sdram_csr_base0 + (unsigned long)k*SDRAM_CSR_STRIDE;
    for (int l = 0; l < CKPHASE_LANES; l++)
        for (int i = 0; i < 8; i++) {
            int t = ckphase.bit[8*l + i].tap + off;
            if (t < 0) t = 0;
            if (t > SDRAM_PHY_DELAYS - 1) t = SDRAM_PHY_DELAYS - 1;
            ckphase_set_tap(1u << l, 1u << i, t);
        }
    sdram_csr_base = sdram_csr_base0;
    printf("ddr: controller %d read taps %+d from the leveling\n", k, off);
}

// Diagnostic / retune: DQ bit `dq` of controller k (the last one leveled) at its leveled read tap
// + off (the others untouched); returns the tap set, -1 when out of range.
int ddr_bit_offset(int k, int dq, int off)
{
    if (dq < 0 || dq >= CKPHASE_BITS) return -1;
    int t = ckphase.bit[dq].tap + off;
    if (t < 0 || t > SDRAM_PHY_DELAYS - 1) return -1;
    sdram_csr_base = sdram_csr_base0 + (unsigned long)k*SDRAM_CSR_STRIDE;
    ckphase_set_tap(1u << (dq/8), 1u << (dq%8), t);
    sdram_csr_base = sdram_csr_base0;
    return t;
}
#endif

#if defined(SDRAM_CSR_STRIDE)
// Diagnostic: load DDR3 mode register `mr` of controller k (k > 0) with `value` (MRS through the
// DFI injector, the controller back under hardware control afterwards).
void ddr_mode_register(int k, int mr, int value)
{
    sdram_csr_base = sdram_csr_base0 + (unsigned long)k*SDRAM_CSR_STRIDE;
    sdram_software_control_on();
    sdram_dfii_pi0_address_write(0x400);         // Precharge all first.
    sdram_dfii_pi0_baddress_write(0);
    command_p0(DFII_COMMAND_RAS|DFII_COMMAND_WE|DFII_COMMAND_CS);
    cdelay(100);
    sdram_dfii_pi0_address_write(value);
    sdram_dfii_pi0_baddress_write(mr);
    command_p0(DFII_COMMAND_RAS|DFII_COMMAND_CAS|DFII_COMMAND_WE|DFII_COMMAND_CS);
    cdelay(200);
    sdram_software_control_off();
    sdram_csr_base = sdram_csr_base0;
    printf("ddr: controller %d MR%d = 0x%x\n", k, mr, value);
}
#endif

/* Test */
/*------*/

// Engine STORE/FETCH round trip through every channel's window with a distinct pattern per
// channel: all stores first, then every fetch, so a channel aliased onto another's memory
// (e.g. a wrong chan_mod) reads back its neighbour's pattern instead of its own.
// 64 beats in the last 4 KB of the channel's own window, base + (stride - base % stride) - 4 KB,
// which the converter reserves, so it is free on every board and for every model. Not below the
// window: the hardware adds the channel's offset, so that would be inside channel c-1's window.
// Not base + stride either: for the last XCU1525 window that is past the channel's 4 GB.
#define DDR_TEST_BEATS  64
#ifndef TRIXIUM_CHAN_SKEW
#define TRIXIUM_CHAN_SKEW 0
#endif
// The smallest window of a controller: the last one, which the bank skew shortens. A target
// that does not end its windows at a stride boundary (the YPCB's even split) exports it.
#ifdef TRIXIUM_CHAN_WINDOW
#define DDR_TEST_WINDOW TRIXIUM_CHAN_WINDOW
#else
#define DDR_TEST_WINDOW (TRIXIUM_CHAN_STRIDE - (TRIXIUM_CHAN_WEIGHT_BASE % TRIXIUM_CHAN_STRIDE) \
                         - (TRIXIUM_CHAN_MOD ? (TRIXIUM_CHAN_MOD - 1)*TRIXIUM_CHAN_SKEW : 0))
#endif
#define DDR_TEST_ADDR   (TRIXIUM_CHAN_WEIGHT_BASE + DDR_TEST_WINDOW - 0x1000u)
#define DDR_TEST_SRC    0x0000u // Scratchpad: the pattern (2 KB).
#define DDR_TEST_DST    0x4000u // Scratchpad: the readback.

static void ddr_pattern(uint32_t chan, uint32_t seed)
{
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_SRC);
    for (uint32_t i = 0; i < DDR_TEST_BEATS*8; i++)
        sp[i] = (seed + i)*0x9E3779B1u ^ (chan << 24);
}

int ddr_test(int chan)
{
    int n_ch = TRIXIUM_N_CHANNELS;
    int first = chan < 0 ? 0 : chan, last = chan < 0 ? n_ch - 1 : chan;
    if (first >= n_ch) {
        printf("ddrtest: channel %d does not exist (0..%d)\n", chan, n_ch - 1);
        return 0;
    }
    for (int c = first; c <= last; c++) {
        ddr_pattern(c, 0x1000u*c + 1);
        struct engine_op op = {0};
        op.kind  = ENGINE_KIND_STORE;
        op.chan  = c;
        op.wbase = DDR_TEST_ADDR;
        op.wlen  = DDR_TEST_BEATS;
        op.xsrc  = DDR_TEST_SRC;
        engine_run(&op);
    }
    int failed = 0;
    for (int c = first; c <= last; c++) {
        ddr_pattern(c, 0x1000u*c + 1); // The expected words, back in the source.
        engine_fetch(c, DDR_TEST_ADDR, DDR_TEST_BEATS, DDR_TEST_DST);
        volatile uint32_t *exp = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_SRC);
        volatile uint32_t *got = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_DST);
        int errors = 0;
        for (uint32_t i = 0; i < DDR_TEST_BEATS*8; i++) {
            if (got[i] != exp[i]) {
                if (errors < 2)
                    printf("  chan %d word %lu: got %08lx expect %08lx\n", c,
                           (unsigned long)i, (unsigned long)got[i], (unsigned long)exp[i]);
                errors++;
            }
        }
        if (errors)
            failed++;
        printf("DDRTEST chan %2d (controller %d) %s (%d errors)\n", c,
               TRIXIUM_CHAN_MOD ? c/TRIXIUM_CHAN_MOD : 0, errors ? "FAIL" : "OK", errors);
    }
    return failed == 0;
}

// Deep write/read test (`ddrdeep <blocks> [chan]`, destructive: before the weights upload):
// `blocks` 64-beat blocks spread over each channel's whole window, pseudo-random data seeded per
// (channel, block); every block of every channel is stored first, then fetched and compared (so
// aliasing and writes that land wrong both show). A controller trained on a marginal spot can
// pass the 64-beat ddrtest and still corrupt a fraction of its writes (the XCU1525's DDR4,
// 2026-10-15: one controller per boot, the weights silently wrong). Returns the failing
// controllers as a bit mask.
static void ddr_pattern_rand(uint32_t seed)
{
    volatile uint32_t *sp = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_SRC);
    uint32_t x = seed | 1;
    for (uint32_t i = 0; i < DDR_TEST_BEATS*8; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        sp[i] = x;
    }
}

uint32_t ddr_test_deep(int blocks, int chan)
{
    int n_ch = TRIXIUM_N_CHANNELS;
    int first = chan < 0 ? 0 : chan, last = chan < 0 ? n_ch - 1 : chan;
    if (blocks < 1) blocks = 1;
    uint32_t span = (DDR_TEST_WINDOW - 0x1000u)/(uint32_t)blocks & ~0x7FFu; // >= one block apart
    if (span < DDR_TEST_BEATS*32) span = DDR_TEST_BEATS*32;
    for (int c = first; c <= last; c++)
        for (int k = 0; k < blocks; k++) {
            ddr_pattern_rand(0x9E3779B1u*(uint32_t)(c*65536 + k + 1));
            engine_store((uint8_t)c, TRIXIUM_CHAN_WEIGHT_BASE + (uint32_t)k*span, DDR_TEST_BEATS, DDR_TEST_SRC);
        }
    uint32_t bad_ctrl = 0;
    for (int c = first; c <= last; c++) {
        uint32_t errors = 0, bad_blocks = 0;
        for (int k = 0; k < blocks; k++) {
            ddr_pattern_rand(0x9E3779B1u*(uint32_t)(c*65536 + k + 1));
            engine_fetch((uint8_t)c, TRIXIUM_CHAN_WEIGHT_BASE + (uint32_t)k*span, DDR_TEST_BEATS, DDR_TEST_DST);
            volatile uint32_t *exp = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_SRC);
            volatile uint32_t *got = (volatile uint32_t *)(SCRATCHPAD_BASE + DDR_TEST_DST);
            uint32_t e = 0;
            for (uint32_t i = 0; i < DDR_TEST_BEATS*8; i++)
                if (got[i] != exp[i]) {
                    if (errors + e < 2)
                        printf("  chan %d block %d word %lu: got %08lx expect %08lx\n", c, k,
                               (unsigned long)i, (unsigned long)got[i], (unsigned long)exp[i]);
                    e++;
                }
            errors += e;
            bad_blocks += (e != 0);
        }
        int ctrl = TRIXIUM_CHAN_MOD ? c/TRIXIUM_CHAN_MOD : 0;
        if (errors) bad_ctrl |= 1u << ctrl;
        printf("DDRDEEP chan %2d (controller %d) %s (%lu errors in %lu of %d blocks)\n", c, ctrl,
               errors ? "FAIL" : "OK", (unsigned long)errors, (unsigned long)bad_blocks, blocks);
    }
    printf("DDRDEEP %s (failing controllers mask 0x%lx)\n", bad_ctrl ? "FAIL" : "OK",
           (unsigned long)bad_ctrl);
    return bad_ctrl;
}

#endif // TRIXIUM_N_MEMCTRL

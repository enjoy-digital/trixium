// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// The extra DDR4 controllers (the XCU1525's four channels): LiteX's own liblitedram, compiled
// once more here with the CSR base as a runtime variable.
//
// The BIOS initialises controller 0 (main_ram) the standard way. Controllers 1..N-1 have no
// CPU window and no BIOS support; they have the same PHY, the same module and the same init
// sequence, and their CSRs sit at controller 0's offsets shifted by k*TRIXIUM_DDR_CSR_STRIDE
// (the target pins the (sdram, ddrphy) pairs to consecutive CSR pages). csr.h's accessors are
// all `CSR_BASE + offset`, so redefining CSR_BASE as a variable for this translation unit
// turns the unmodified sdram_init() - init sequence, write leveling, read leveling - into
// init(k). Nothing in the calibration touches the CPU's memory bus: it goes through the DFI
// injector's CSRs, which is why a controller without a CPU window can be calibrated at all.
// The final memtest of sdram_init() would use main_ram, so it is compiled out and replaced
// by ddr_test(): an engine STORE / FETCH round trip through every channel's own window, which
// is the path the weights take and the one that would alias channels onto one memory.

#include <generated/soc.h>

#ifdef TRIXIUM_N_MEMCTRL

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <generated/mem.h>              // CSR_BASE, the literal.
static const unsigned long ddr_csr_base0 = CSR_BASE;
#undef CSR_BASE
unsigned long ddr_csr_base;             // The controller being talked to.
#define CSR_BASE ddr_csr_base
#include <generated/csr.h>              // Every accessor: ddr_csr_base + offset.

#define SDRAM_TEST_DISABLE              // No main_ram memtest at the end of sdram_init().
#include <liblitedram/sdram.c>          // The BIOS's init and calibration, at our base.
#include <liblitedram/accessors.c>      // The PHY delay accessors the leveling uses.

#include "engine.h"
#include "ddr.h"

#ifndef TRIXIUM_CHAN_MOD
#define TRIXIUM_CHAN_MOD 0
#endif

int ddr_init(int k)
{
    if (k < 0 || k >= TRIXIUM_N_MEMCTRL) {
        printf("ddr: controller %d does not exist (0..%d)\n", k, TRIXIUM_N_MEMCTRL - 1);
        return 0;
    }
    ddr_csr_base = ddr_csr_base0 + (unsigned long)k*TRIXIUM_DDR_CSR_STRIDE;
    printf("ddr: controller %d (CSRs at %08lx)\n", k, (unsigned long)CSR_SDRAM_BASE);
    int ok = sdram_init();
    ddr_csr_base = ddr_csr_base0;
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

// Engine STORE / FETCH round trip through every channel's window, distinct pattern per
// channel: all stores first, then every fetch, so a channel aliased onto another's memory
// (e.g. a wrong chan_mod) reads back its neighbour's pattern instead of its own. 64 beats
// in the last 4 KB of the channel's own window - base + (stride - base % stride) - 4 KB,
// which the converter reserves (a model is capped at the window minus these 4 KB), so it is
// free on every board and for every model. NOT "below the window": the hardware adds the
// channel's offset, so an address below channel c's window is inside channel c-1's (with
// the sim's 2 MB stride, that is the neighbour's embedding table). And not "base + stride":
// with the XCU1525's 1 GB stride and 320 MB base that is past the channel's 4 GB for the
// last window.
#define DDR_TEST_BEATS  64
#ifndef TRIXIUM_CHAN_SKEW
#define TRIXIUM_CHAN_SKEW 0
#endif
// The smallest window of a controller: the last one, which the bank skew shortens.
#define DDR_TEST_WINDOW (TRIXIUM_CHAN_STRIDE - (TRIXIUM_CHAN_WEIGHT_BASE % TRIXIUM_CHAN_STRIDE) \
                         - (TRIXIUM_CHAN_MOD ? (TRIXIUM_CHAN_MOD - 1)*TRIXIUM_CHAN_SKEW : 0))
#define DDR_TEST_ADDR   (TRIXIUM_CHAN_WEIGHT_BASE + DDR_TEST_WINDOW - 0x1000u)
#define DDR_TEST_SRC   0x0000u   // Scratchpad: the pattern (2 KB).
#define DDR_TEST_DST   0x4000u   // Scratchpad: the readback.

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
        ddr_pattern(c, 0x1000u*c + 1);           // The expected words, back in the source.
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
        if (errors) failed++;
        printf("DDRTEST chan %2d (controller %d) %s (%d errors)\n", c,
               TRIXIUM_CHAN_MOD ? c/TRIXIUM_CHAN_MOD : 0, errors ? "FAIL" : "OK", errors);
    }
    return failed == 0;
}

#endif // TRIXIUM_N_MEMCTRL

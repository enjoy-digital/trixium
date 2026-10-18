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
#include "bons.h"
#include "infer.h"
#include "ddr.h"

// ECC-guided per-bit read tap centring (ddr_ecc_scan): a leveling the firmware owns, the
// controller's ECC per-bit counters and its BIST.
#if defined(SDRAM_CSR_STRIDE) && defined(SDRAM_PHY_CKPHASE_STEPS) && \
    defined(CSR_SDRAM_ECC_BITSEL_ADDR) && defined(CSR_SDRAM_GENERATOR_BASE_ADDR)
#define DDR_ECC_SCAN 1
#endif

#ifndef TRIXIUM_CHAN_MOD
#define TRIXIUM_CHAN_MOD 0
#endif

/* Init */
/*------*/

// The controller whose per-bit read taps the `ckphase` table holds: the last one the FIRMWARE
// leveled (-1: only the BIOS has, and its table is not ours - ddrtap/ddrbit refuse).
static int ckphase_ctrl = -1;

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
    ckphase_ctrl = ok ? k : -1;
#ifdef DDR_ECC_SCAN
    if (ok)
        ddr_ecc_scan(k, 6, 4);
#endif
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
#ifdef DDR_ECC_SCAN
    // ... in an eye its pattern test overestimates: level controllers 1.. again here, where the
    // ECC scan can then centre every bit (ddr_ecc_scan). Controller 0 holds main_ram: the BIOS's.
    return ddr_init_all();
#else
    return 1;
#endif
#else
    return ddr_init_all();
#endif
}

#if defined(SDRAM_CSR_STRIDE) && defined(SDRAM_PHY_CKPHASE_STEPS)
// Diagnostic: re-apply the last leveling's per-bit read taps (controller k, which must be the
// last one leveled) shifted by `off` taps; off 0 restores them.
void ddr_tap_offset(int k, int off)
{
    if (k != ckphase_ctrl) {
        printf("ddr: controller %d's taps are not the firmware's last leveling (ddrinit %d first)\n", k, k);
        return;
    }
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

#ifdef DDR_ECC_SCAN
// Read-eye scan under inference-like traffic, per DQ bit, with the controller's ECC as the
// oracle. Every bit's read tap goes to its leveled tap + off for off in [-range, range]; at
// each offset the engine streams (`passes` times, a ternary matvec with its result writes
// suppressed) a 32 MB region at the start of every channel window, written once by each
// controller's LiteDRAM BIST generator (ECC words: each corrected error is charged to its DQ by
// the per-bit counters); a bit is then set to the centre of its longest clean run of offsets
// (ties: toward the leveled tap). Why this traffic: the leveling's own window
// (ckphase_dfi_check: a few DFI-injected bursts per tap) is 10-13 taps wide and the BIST
// checker's single stream sees 6-8, where the engine's concurrent channel streams see 2-3 (the
// YPCB at DDR3-933); the leveling centred DQ14 one tap from that edge (~600 corrected / 2-7
// uncorrectable reads per 24 tokens on some boots, 64k / 26k on a bad one; none with every tap
// one earlier). Writes do not depend on the read taps. Destroys the first 32 MB of every
// channel's weights: at boot (before any model), or reload the model after. Returns the number
// of bits moved, -1 when controller k's taps are not the firmware's leveling (ddrinit k first).
// The scan's loop: every bit's tap at its current tap + off for off in [-range, range], the
// traffic callback `passes` times at each offset (0 = ran; nonzero stops the scan: the taps
// go back and -1 comes back), the ECC per-bit counters marking the bits in error; then each
// bit to the centre of its longest clean run of offsets (ties: toward the current tap).
static int ddr_scan_eye(int k, int range, int passes, int (*traffic)(void *), void *arg,
                        const char *what)
{
    if (k != ckphase_ctrl) {
        printf("ddr: controller %d's taps are not the firmware's last leveling (ddrinit %d first)\n", k, k);
        return -1;
    }
    if (range < 1) range = 1;
    if (range > 8) range = 8;
    if (passes < 1) passes = 1;
    sdram_csr_base = sdram_csr_base0 + (unsigned long)k*SDRAM_CSR_STRIDE;
    uint32_t clean[CKPHASE_BITS];               // bit (off + range): no corrected error on DQ b
    memset(clean, 0, sizeof clean);
    int failed = 0;
    for (int off = -range; off <= range && !failed; off++) {
        for (int b = 0; b < CKPHASE_BITS; b++) {
            int t = ckphase.bit[b].tap + off;
            if (t < 0) t = 0;
            if (t > SDRAM_PHY_DELAYS - 1) t = SDRAM_PHY_DELAYS - 1;
            ckphase_set_tap(1u << (b/8), 1u << (b%8), t);
        }
        sdram_ecc_clear_write(1);
        for (int p = 0; p < passes && !failed; p++) {
            sdram_csr_base = sdram_csr_base0;   // the traffic may use any controller's CSRs
            failed = traffic(arg);
            sdram_csr_base = sdram_csr_base0 + (unsigned long)k*SDRAM_CSR_STRIDE;
        }
        for (int b = 0; b < CKPHASE_BITS; b++) {
            sdram_ecc_bitsel_write(b);
            (void)sdram_ecc_bitcnt_read();      // registered mux: settle
            if (sdram_ecc_bitcnt_read() == 0)
                clean[b] |= 1u << (off + range);
        }
    }
    if (failed) {
        for (int b = 0; b < CKPHASE_BITS; b++)
            ckphase_set_tap(1u << (b/8), 1u << (b%8), ckphase.bit[b].tap);
        sdram_ecc_clear_write(1);
        sdram_csr_base = sdram_csr_base0;
        printf("ddr: controller %d eye scan under %s stopped (the traffic failed) - taps unchanged\n",
               k, what);
        return -1;
    }
    int moved = 0;
    printf("ddr: controller %d read eye under %s, offsets %+d..%+d from the leveling "
           "(. error, # clean), each bit's tap moved by:\n", k, what, -range, range);
    for (int l = 0; l < CKPHASE_LANES; l++) {
        printf("ddr:  m%d:", l);
        for (int i = 0; i < 8; i++) {
            int b = 8*l + i, lo = 0, hi = -1, run_lo = 0;
            // The longest clean run; on a tie the one holding the leveled tap wins.
            for (int o = -range; o <= range + 1; o++) {
                int c = o <= range && (clean[b] >> (o + range) & 1);
                if (c && (o == -range || !(clean[b] >> (o - 1 + range) & 1)))
                    run_lo = o;
                if (!c && o > -range && (clean[b] >> (o - 1 + range) & 1)) {
                    int run_hi = o - 1;
                    if (run_hi - run_lo > hi - lo ||
                        (run_hi - run_lo == hi - lo && run_lo <= 0 && run_hi >= 0)) {
                        lo = run_lo; hi = run_hi;
                    }
                }
            }
            int c = hi < lo ? 0 : (lo + hi)/2;   // no clean offset: the leveled tap stays
            int t = ckphase.bit[b].tap + c;
            if (t < 0) t = 0;
            if (t > SDRAM_PHY_DELAYS - 1) t = SDRAM_PHY_DELAYS - 1;
            ckphase.bit[b].tap = (uint8_t)t;     // the next scan's (and ddrbit/ddrtap's) origin
            ckphase_set_tap(1u << l, 1u << i, t);
            if (c) moved++;
            printf(" ");
            for (int o = -range; o <= range; o++)
                printf("%c", clean[b] >> (o + range) & 1 ? '#' : '.');
            printf("%+d", c);
        }
        printf("\n");
    }
    sdram_ecc_clear_write(1);
    sdram_csr_base = sdram_csr_base0;
    printf("ddr: controller %d: %d of %d bits moved\n", k, moved, CKPHASE_BITS);
    return moved;
}

// The synthetic traffic: a trit5 matvec with its result writes suppressed over a 32 MB
// region of every channel window, written once by each controller's BIST generator.
static int scan_matvec(void *arg)
{
    engine_run((struct engine_op *)arg);
    return engine_wedged ? -1 : 0;
}

int ddr_ecc_scan(int k, int range, int passes)
{
    if (k != ckphase_ctrl) {
        printf("ddr: controller %d's taps are not the firmware's last leveling (ddrinit %d first)\n", k, k);
        return -1;
    }
    // The engine op: trit5 rows of 2,048 columns, as many as fill 32 MB per channel.
    const uint32_t cols = 2048, group = 128, len = 32u << 20;
    uint32_t rb = bons_row_beats(cols, KIND_TERNARY5, group);   // memory beats per row
    uint32_t rows = len/(rb*32u);
    if (rows > 65535u) rows = 65535u;
    struct engine_op op = {0};
    op.kind = ENGINE_KIND_TERNARY; op.rows = (uint16_t)rows; op.cols = (uint16_t)cols;
    op.wbase = TRIXIUM_CHAN_WEIGHT_BASE; op.wlen = rows*rb;
    op.sbeats = (uint8_t)bons_scale_beats(cols, group);
    op.cbeats = (uint8_t)bons_code_beats(cols, KIND_TERNARY5);     // pipe beats per row
    op.pk = 1; op.supw = 1;
    // The region in every window of every controller (the op streams every channel).
    for (int c = 0; c < TRIXIUM_N_MEMCTRL; c++) {
        sdram_csr_base = sdram_csr_base0 + (unsigned long)c*SDRAM_CSR_STRIDE;
        for (int j = 0; j < (TRIXIUM_CHAN_MOD ? TRIXIUM_CHAN_MOD : 1); j++) {
            uint32_t base = (uint32_t)j*TRIXIUM_CHAN_STRIDE + TRIXIUM_CHAN_WEIGHT_BASE;
            sdram_generator_reset_write(1); sdram_generator_reset_write(0);
            sdram_generator_random_write(1);
            sdram_generator_base_write(base); sdram_generator_end_write(base + len);
            sdram_generator_length_write(len);
            sdram_generator_start_write(1);
            while (!sdram_generator_done_read());
        }
    }
    sdram_csr_base = sdram_csr_base0;
    return ddr_scan_eye(k, range, passes, scan_matvec, &op, "engine traffic");
}

// The inference's own traffic: a forward of the loaded model per pass (infer_eye_forward: run
// from the checkpoint and restored, as the ECC retry does), every weight stream, the DeltaNet
// state's reads and writes and the KV rows included - the pattern the synthetic matvec does
// not reproduce on every bitstream (ypcb_hyb7: its scan left three bits at 150-500 corrected
// reads per 16 tokens). Run at the end of every hybrid model load (infer_init).
static int scan_forward(void *arg)
{
    (void)arg;
    return infer_eye_forward();
}

int ddr_eye_scan(int k, int range, int forwards)
{
    return ddr_scan_eye(k, range, forwards, scan_forward, 0, "inference traffic");
}
#endif

// Diagnostic / retune: DQ bit `dq` of controller k (the last one leveled) at its leveled read tap
// + off (the others untouched); returns the tap set, -1 when out of range.
int ddr_bit_offset(int k, int dq, int off)
{
    if (dq < 0 || dq >= CKPHASE_BITS || k != ckphase_ctrl) return -1;
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

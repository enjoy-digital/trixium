/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Host test of firmware/ddr_s7hr.c: the calibration against a fake YPCB channel.
 *
 * The fake models what the calibration has to find on the card (docs: openTPU's card scans):
 *   - per lane, a tDQSS crossing: the CK phase where the lane's DQS is half a tCK from its
 *     chip's CK edge; below it the lane needs write bitslip base_wl, above it base_wl + 2, and
 *     within +-Z steps of it the lane's writes fail at any latency (DFI check) and within
 *     +-(Z + T) they fail under traffic (BIST) while a single burst may still pass;
 *   - per DQ bit, a read eye of 2H taps at a virtual delay V = bitslip*12 + tap that moves with
 *     the CK phase (the DRAM drives DQ from its CK), some bits framed a CLK (+-24) off their
 *     lane;
 *   - DFII commands and the BIST as the S7HRDDRPHY + LiteDRAM present them.
 * The program runs sdram_custom_init() and checks its choices against the model: the CK phase
 * inside the window common to all lanes (near its centre), each lane's write latency, each
 * bit's read bitslip/tap inside its eye. Exit status 0 on success.
 *
 *   s7hr_fake <seed> [nowindow]
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <generated/csr.h>
#include <generated/sdram_phy.h>

/* Model */
/*-------*/

#define LANES 8
#define BITS  64
#define STEPS 112

static int    cross[LANES];
static int    base_wl[LANES];
static double v0[BITS];
static int    Z = 3, T = 4;
static double H = 4.5;
static const double TAPS_PER_STEP = 16.74/78.0;

// PHY state.
static int      ck;
static int      wl[LANES];
static int      rb[BITS];
static int      tap[BITS];
static uint32_t dly_sel, dq_dly_sel;
static uint32_t address[4], baddress[4];
static uint32_t wrdata[16], rddata[16];
static uint64_t mem[2][8];
static long     csr_accesses, dfi_checks, bists;

static uint32_t rng_state = 1;
static uint32_t rng(void)
{
    rng_state = rng_state*1664525u + 1013904223u;
    return rng_state >> 8;
}

static int dist(int a, int b)
{
    int d = ((a - b) % STEPS + STEPS) % STEPS;
    return d < STEPS - d ? d : STEPS - d;
}

static int wl_needed(int l, int at)
{
    return (base_wl[l] + (at >= cross[l] ? 2 : 0)) & 7;
}

static int write_ok(int l, int margin)
{
    return wl[l] == wl_needed(l, ck) && dist(ck, cross[l]) > Z + margin;
}

static double eye_off(int b, int at, int bitslip, int t)
{
    return bitslip*12 + t - (v0[b] + TAPS_PER_STEP*at);
}

static int read_ok(int b, double margin)
{
    return fabs(eye_off(b, ck, rb[b], tap[b])) <= H - margin;
}

/* CSRs */
/*------*/

static int bit_selected(int b, int read_side)
{
    uint32_t mask = (read_side && dq_dly_sel) ? dq_dly_sel : 0xff;
    return ((dly_sel >> (b/8)) & 1) && ((mask >> (b%8)) & 1);
}

void csr_write_simple(uint32_t value, unsigned long addr)
{
    csr_accesses++;
    if (addr >= FAKE_WRDATA_BASE && addr < FAKE_WRDATA_BASE + 0x40)
        wrdata[(addr - FAKE_WRDATA_BASE)/4] = value;
}

uint32_t csr_read_simple(unsigned long addr)
{
    csr_accesses++;
    if (addr >= FAKE_RDDATA_BASE && addr < FAKE_RDDATA_BASE + 0x40)
        return rddata[(addr - FAKE_RDDATA_BASE)/4];
    return 0;
}

void ddrphy_rst_write(uint32_t v)
{
    if (v) {
        for (int l = 0; l < LANES; l++) wl[l] = 0;
        for (int b = 0; b < BITS; b++) rb[b] = tap[b] = 0;
    }
}
void ddrphy_dly_sel_write(uint32_t v)     { csr_accesses++; dly_sel = v; }
void ddrphy_dq_dly_sel_write(uint32_t v)  { csr_accesses++; dq_dly_sel = v; }
void ddrphy_rdly_dq_rst_write(uint32_t v) { (void)v; for (int b = 0; b < BITS; b++) if (bit_selected(b, 1)) tap[b] = 0; }
void ddrphy_rdly_dq_inc_write(uint32_t v) { (void)v; for (int b = 0; b < BITS; b++) if (bit_selected(b, 1) && tap[b] < 31) tap[b]++; }
void ddrphy_rdly_dq_bitslip_rst_write(uint32_t v) { (void)v; for (int b = 0; b < BITS; b++) if (bit_selected(b, 1)) rb[b] = 0; }
void ddrphy_rdly_dq_bitslip_write(uint32_t v)     { (void)v; for (int b = 0; b < BITS; b++) if (bit_selected(b, 1)) rb[b] = (rb[b] + 1) & 7; }
void ddrphy_wdly_dq_bitslip_rst_write(uint32_t v) { (void)v; for (int l = 0; l < LANES; l++) if ((dly_sel >> l) & 1) wl[l] = 0; }
void ddrphy_wdly_dq_bitslip_write(uint32_t v)     { (void)v; for (int l = 0; l < LANES; l++) if ((dly_sel >> l) & 1) wl[l] = (wl[l] + 1) & 7; }
uint32_t ddrphy_rdphase_read(void) { return SDRAM_PHY_RDPHASE; }
uint32_t ddrphy_wrphase_read(void) { return SDRAM_PHY_WRPHASE; }
void ddrphy_ckphase_inc_write(uint32_t v) { (void)v; ck++; }
void ddrphy_ckphase_dec_write(uint32_t v) { (void)v; ck--; }
uint32_t ddrphy_ckphase_status_read(void) { return ((uint32_t)ck & 0xffff) | (1u << CSR_DDRPHY_CKPHASE_STATUS_LOCKED_OFFSET); }

void sdram_dfii_control_write(uint32_t v) { (void)v; }
void sdram_dfii_pi_address_write(int p, uint32_t v)  { address[p]  = v; }
void sdram_dfii_pi_baddress_write(int p, uint32_t v) { baddress[p] = v; }

void sdram_dfii_pi_command_issue(int p, uint32_t cmd)
{
    csr_accesses += 2;
    if (cmd & DFII_COMMAND_WRDATA) {
        if (p != SDRAM_PHY_WRPHASE) { printf("FAIL: write on phase %d\n", p); exit(2); }
        int burst = (address[p] >> 3) & 1;
        for (int q = 0; q < 4; q++) {
            mem[burst][2*q + 1] = ((uint64_t)wrdata[4*q + 0] << 32) | wrdata[4*q + 1];
            mem[burst][2*q + 0] = ((uint64_t)wrdata[4*q + 2] << 32) | wrdata[4*q + 3];
        }
        // A lane whose write latency or tDQSS is wrong writes garbage.
        for (int l = 0; l < LANES; l++)
            if (!write_ok(l, 0))
                for (int n = 0; n < 8; n++)
                    mem[burst][n] ^= (uint64_t)(rng() & 0xff) << (8*l) | (uint64_t)1 << (8*l);
    } else if (cmd & DFII_COMMAND_RDDATA) {
        if (p != SDRAM_PHY_RDPHASE) { printf("FAIL: read on phase %d\n", p); exit(2); }
        int burst = (address[p] >> 3) & 1;
        uint64_t beats[8];
        for (int n = 0; n < 8; n++) {
            beats[n] = mem[burst][n];
            for (int b = 0; b < BITS; b++)
                if (!read_ok(b, 0))
                    beats[n] = (beats[n] & ~((uint64_t)1 << b)) | ((uint64_t)(rng() & 1) << b);
        }
        for (int q = 0; q < 4; q++) {
            rddata[4*q + 0] = beats[2*q + 1] >> 32;
            rddata[4*q + 1] = beats[2*q + 1];
            rddata[4*q + 2] = beats[2*q + 0] >> 32;
            rddata[4*q + 3] = beats[2*q + 0];
        }
        dfi_checks++;
    }
}

void cdelay(int i) { (void)i; }
void fake_init_sequence(void) { }

// The BIST: clean only with every lane's writes and every bit's reads inside their traffic margin.
static uint32_t bist_errors;
void sdram_generator_reset_write(uint32_t v)  { (void)v; }
void sdram_generator_random_write(uint32_t v) { (void)v; }
void sdram_generator_base_write(uint32_t v)   { (void)v; }
void sdram_generator_end_write(uint32_t v)    { (void)v; }
void sdram_generator_length_write(uint32_t v) { (void)v; }
void sdram_generator_start_write(uint32_t v)  { (void)v; bists++; }
uint32_t sdram_generator_done_read(void)      { return 1; }
void sdram_checker_reset_write(uint32_t v)    { (void)v; }
void sdram_checker_random_write(uint32_t v)   { (void)v; }
void sdram_checker_base_write(uint32_t v)     { (void)v; }
void sdram_checker_end_write(uint32_t v)      { (void)v; }
void sdram_checker_length_write(uint32_t v)   { (void)v; }
void sdram_checker_start_write(uint32_t v)
{
    (void)v;
    bist_errors = 0;
    for (int l = 0; l < LANES; l++)
        if (!write_ok(l, T))
            bist_errors += 100;
    for (int b = 0; b < BITS; b++)
        if (!read_ok(b, 1.0))
            bist_errors += 1;
}
uint32_t sdram_checker_done_read(void)   { return 1; }
uint32_t sdram_checker_errors_read(void) { return bist_errors; }

/* The code under test */
/*---------------------*/

#include "ddr_s7hr.c"

/* Test */
/*------*/

int main(int argc, char **argv)
{
    int seed     = argc > 1 ? atoi(argv[1]) : 1;
    int nowindow = argc > 2 && strcmp(argv[2], "nowindow") == 0;
    rng_state = 0x9E3779B9u*(uint32_t)seed + 7;

    // A fly-by: crossings ordered along the chain, ~40 steps apart end to end (openTPU's
    // channel 1 spans 34), shifted at random; base latencies 0 or 6 (-2).
    int origin = (int)(rng() % STEPS) - STEPS/2;
    for (int l = 0; l < LANES; l++) {
        cross[l]   = ((origin + l*5 + (int)(rng() % 3)) % STEPS + STEPS + STEPS/2) % STEPS - STEPS/2;
        base_wl[l] = (rng() & 1) ? 0 : 6;
        if (nowindow)
            cross[l] = -STEPS/2 + l*(STEPS/LANES);
    }
    for (int l = 0; l < LANES; l++) {
        double lane_v = 40 + (rng() % 16);
        for (int i = 0; i < 8; i++) {
            int b = 8*l + i;
            v0[b] = lane_v + ((int)(rng() % 31) - 15)/10.0;
            if ((rng() % 13) == 0)          // A bit framed a CLK off its lane.
                v0[b] += (rng() & 1) ? 24 : -24;
        }
    }
    ck = 0;

    int ok = sdram_custom_init();
    printf("fake: %ld CSR accesses, %ld DFI checks, %ld BISTs\n", csr_accesses, dfi_checks, bists);

    // The window common to every lane under traffic.
    int pass[STEPS], best_len = 0, best_start = 0;
    for (int i = 0; i < STEPS; i++) {
        pass[i] = 1;
        for (int l = 0; l < LANES; l++)
            if (dist(i - STEPS/2, cross[l]) <= Z + T)
                pass[i] = 0;
    }
    for (int s = 0; s < STEPS; s++) {
        int len = 0;
        while (len < STEPS && pass[(s + len) % STEPS])
            len++;
        if (len > best_len) { best_len = len; best_start = s; }
    }
    if (nowindow) {
        if (ok) { printf("FAIL: calibrated without a common window\n"); return 1; }
        printf("PASS: no common window, refused\n");
        return 0;
    }
    if (best_len < 6) {
        printf("SKIP: seed %d has a %d-step window\n", seed, best_len);
        return 0;
    }
    int centre = (best_start + best_len/2) % STEPS - STEPS/2;
    printf("model: window %d steps, centre %+d\n", best_len, centre);
    if (!ok) { printf("FAIL: calibration failed\n"); return 1; }
    if (!pass[s7hr.ckphase + STEPS/2]) { printf("FAIL: CK %+d outside the window\n", s7hr.ckphase); return 1; }
    if (dist(s7hr.ckphase, centre) > S7HR_SCAN_STRIDE + 1) {
        printf("FAIL: CK %+d off the window centre %+d\n", s7hr.ckphase, centre);
        return 1;
    }
    for (int l = 0; l < LANES; l++)
        if (wl[l] != wl_needed(l, ck)) { printf("FAIL: lane %d write bitslip %d, needs %d\n", l, wl[l], wl_needed(l, ck)); return 1; }
    for (int b = 0; b < BITS; b++)
        if (!read_ok(b, 1.5)) {
            printf("FAIL: bit %d read bitslip %d tap %d is %.1f taps off its eye centre\n", b, rb[b], tap[b],
                eye_off(b, ck, rb[b], tap[b]));
            return 1;
        }
    printf("PASS: CK %+d (window centre %+d), %d-step run\n", s7hr.ckphase, centre, s7hr.run);
    return 0;
}

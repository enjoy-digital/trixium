/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Compiled-RTL check of SiluMulKernel against model/silu_hw.silu_mul_fixed.
 * Vector file: u32 count; u32 lut[4097]; u32 dlut[4096]; count x {u32 g, m, r, fixup}.
 * Loads the tables through the write ports, then streams every element with random input
 * gaps and random output stalls, and checks each output in order: fixup must match
 * exactly, and r must match wherever fixup is clear. Output must stay stable under stall.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
#include "Vsilu_mul.h"

static uint32_t word(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
static void require(bool ok, const char *what) { if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); } }

int main(int argc, char **argv) {
    require(argc == 2, "vector file required");
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    require(data.size() >= 4, "short file");
    const uint32_t count = word(&data[0]);
    const size_t base = 4 + 4*(4097 + 4096);
    require(data.size() == base + 16ull*count, "vector length");
    Vsilu_mul d;
    auto tick = [&]() { d.sys_clk = 0; d.eval(); d.sys_clk = 1; d.eval(); };
    d.sys_rst = 1; tick(); tick(); d.sys_rst = 0;
    for (uint32_t i = 0; i < 4097; i++) {
        d.lut_we = 1; d.lut_adr = i; d.lut_dat = word(&data[4 + 4*i]);
        d.dlut_we = i < 4096; d.dlut_adr = i & 0xFFF; d.dlut_dat = i < 4096 ? word(&data[4 + 4*(4097 + i)]) : 0;
        tick();
    }
    d.lut_we = 0; d.dlut_we = 0;
    uint32_t state = 12345, sent = 0, got = 0, checked = 0, fixups = 0, cycles = 0;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return state >> 8; };
    bool held = false; uint32_t held_r = 0, held_f = 0;
    while (got < count) {
        require(++cycles < 20*count + 100000, "timeout");
        bool offer = sent < count && (rnd() % 8) != 0;
        const uint8_t *in = &data[base + 16ull*sent];
        d.sink_valid = offer;
        if (offer) { d.sink_g = word(in); d.sink_m = word(in + 4); }
        d.source_ready = (rnd() % 5) != 0;
        d.sys_clk = 0; d.eval();
        if (held) require(d.source_valid && d.source_r == held_r && d.source_fixup == held_f,
                          "output changed under backpressure");
        bool take_in = d.sink_valid && d.sink_ready;
        bool take_out = d.source_valid && d.source_ready;
        if (take_out) {
            const uint8_t *e = &data[base + 16ull*got];
            uint32_t want_r = word(e + 8), want_f = word(e + 12);
            if (d.source_fixup != want_f) {
                std::fprintf(stderr, "element %u g=%08x m=%08x: fixup %u, model %u\n",
                             got, word(e), word(e + 4), d.source_fixup, want_f);
                std::exit(1);
            }
            if (!want_f && d.source_r != want_r) {
                std::fprintf(stderr, "element %u g=%08x m=%08x: r=%08x, model %08x\n",
                             got, word(e), word(e + 4), d.source_r, want_r);
                std::exit(1);
            }
            checked += !want_f; fixups += want_f; got++;
        }
        held = d.source_valid && !d.source_ready;
        held_r = d.source_r; held_f = d.source_fixup;
        d.sys_clk = 1; d.eval();
        if (take_in) sent++;
    }
    std::printf("%u elements: %u exact, %u fixups, %u cycles\n", count, checked, fixups, cycles);
    return 0;
}

/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Compiled check of gateware/units/fp32.py against numpy float32 (IEEE, canonical NaN).
 * usage: Vfp32x <op> vectors.bin    op: mul | add | fma | div | sqrt
 * vectors.bin: count x {u32 a, u32 b, u32 expected} (fma: {a, b, c, expected}). Pipelined ops
 * run with random clock-enable gaps; div/sqrt run start -> done one at a time.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "Vfp32x.h"

static void require(bool ok, const char *what) { if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); } }

int main(int argc, char **argv) {
    require(argc == 3, "usage: op vectors");
    const char *op = argv[1];
    bool mul = !std::strcmp(op, "mul"), add = !std::strcmp(op, "add"), fma = !std::strcmp(op, "fma");
    bool div = !std::strcmp(op, "div"), sq = !std::strcmp(op, "sqrt");
    require(mul || add || fma || div || sq, "op");
    const size_t W = fma ? 4 : 3;                     // words per record
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    require(raw.size() % (4*W) == 0, "vector length");
    size_t n = raw.size()/(4*W);
    std::vector<uint32_t> v(n*W);
    std::memcpy(v.data(), raw.data(), raw.size());
    Vfp32x d;
    auto tick = [&]() { d.sys_clk = 0; d.eval(); d.sys_clk = 1; d.eval(); };
    d.sys_rst = 1; tick(); tick(); d.sys_rst = 0;
    uint32_t state = 99, bad = 0;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return state >> 8; };
    auto report = [&](size_t i, uint32_t got) {
        if (bad++ < 8) std::fprintf(stderr, "%s %zu: a=%08x b=%08x c=%08x got %08x want %08x\n", op, i,
                                    v[W*i], v[W*i + 1], fma ? v[W*i + 2] : 0, got, v[W*i + W - 1]);
    };
    if (mul || add || fma) {
        const int L = (mul ? d.mul_latency : add ? d.add_latency : d.fma_latency) - 1;   // entry clock counts as the first stage
        std::vector<long> slots;          // index entering at each enabled clock
        size_t issued = 0, checked = 0;
        while (checked < n) {
            bool ce = (rnd() % 5) != 0;
            long idx = -1;
            if (ce && issued < n) idx = (long)issued++;
            uint32_t a = idx >= 0 ? v[W*idx] : 0, b = idx >= 0 ? v[W*idx + 1] : 0, c = idx >= 0 ? v[W*idx + 2] : 0;
            if (mul)      { d.mul_ce = ce; d.mul_a = a; d.mul_b = b; }
            else if (add) { d.add_ce = ce; d.add_a = a; d.add_b = b; }
            else          { d.fma_ce = ce; d.fma_a = a; d.fma_b = b; d.fma_c = c; }
            tick();
            if (!ce) continue;
            slots.push_back(idx);
            if ((long)slots.size() > L) {
                long out = slots[slots.size() - 1 - L];
                if (out >= 0) {
                    uint32_t got = mul ? d.mul_r : add ? d.add_r : d.fma_r;
                    if (got != v[W*out + W - 1]) report(out, got);
                    checked++;
                }
            }
            if (issued == n && (long)slots.size() > (long)n + L + 10) break;
        }
        // Drain.
        while (checked < n) {
            if (mul) d.mul_ce = 1; else if (add) d.add_ce = 1; else d.fma_ce = 1;
            tick();
            slots.push_back(-1);
            long out = slots[slots.size() - 1 - L];
            if (out >= 0) {
                uint32_t got = mul ? d.mul_r : add ? d.add_r : d.fma_r;
                if (got != v[W*out + W - 1]) report(out, got);
                checked++;
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            d.ds_a = v[W*i]; d.ds_b = v[W*i + 1]; d.ds_op = sq; d.ds_start = 1;
            tick(); d.ds_start = 0;
            int c = 0;
            while (!d.ds_done) { tick(); require(++c < 100, "div/sqrt timeout"); }
            if (d.ds_r != v[W*i + 2]) report(i, d.ds_r);
        }
    }
    std::printf("%s: %zu vectors, %u mismatches\n", op, n, bad);
    return bad ? 1 : 0;
}

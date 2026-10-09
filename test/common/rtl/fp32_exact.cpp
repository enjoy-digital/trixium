/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Compiled check of gateware/units/fp32.py against numpy float32 (IEEE, canonical NaN).
 * usage: Vfp32x <op> vectors.bin    op: mul | add | div | sqrt
 * vectors.bin: count x {u32 a, u32 b, u32 expected}. Pipelined ops run with random clock-
 * enable gaps; div/sqrt run start -> done one at a time.
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
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    require(raw.size() % 12 == 0, "vector length");
    size_t n = raw.size()/12;
    std::vector<uint32_t> v(n*3);
    std::memcpy(v.data(), raw.data(), raw.size());
    const char *op = argv[1];
    bool mul = !std::strcmp(op, "mul"), add = !std::strcmp(op, "add");
    bool div = !std::strcmp(op, "div"), sq = !std::strcmp(op, "sqrt");
    require(mul || add || div || sq, "op");
    Vfp32x d;
    auto tick = [&]() { d.sys_clk = 0; d.eval(); d.sys_clk = 1; d.eval(); };
    d.sys_rst = 1; tick(); tick(); d.sys_rst = 0;
    uint32_t state = 99, bad = 0;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return state >> 8; };
    auto report = [&](size_t i, uint32_t got) {
        if (bad++ < 8) std::fprintf(stderr, "%s %zu: a=%08x b=%08x got %08x want %08x\n", op, i,
                                    v[3*i], v[3*i + 1], got, v[3*i + 2]);
    };
    if (mul || add) {
        const int L = (mul ? d.mul_latency : d.add_latency) - 1;   // entry clock counts as the first stage
        std::vector<long> slots;          // index entering at each enabled clock
        size_t issued = 0, checked = 0;
        while (checked < n) {
            bool ce = (rnd() % 5) != 0;
            long idx = -1;
            if (ce && issued < n) idx = (long)issued++;
            if (mul) { d.mul_ce = ce; d.mul_a = idx >= 0 ? v[3*idx] : 0; d.mul_b = idx >= 0 ? v[3*idx + 1] : 0; }
            else     { d.add_ce = ce; d.add_a = idx >= 0 ? v[3*idx] : 0; d.add_b = idx >= 0 ? v[3*idx + 1] : 0; }
            tick();
            if (!ce) continue;
            slots.push_back(idx);
            if ((long)slots.size() > L) {
                long out = slots[slots.size() - 1 - L];
                if (out >= 0) {
                    uint32_t got = mul ? d.mul_r : d.add_r;
                    if (got != v[3*out + 2]) report(out, got);
                    checked++;
                }
            }
            if (issued == n && (long)slots.size() > (long)n + L + 10) break;
        }
        // Drain.
        while (checked < n) {
            if (mul) d.mul_ce = 1; else d.add_ce = 1;
            tick();
            slots.push_back(-1);
            long out = slots[slots.size() - 1 - L];
            if (out >= 0) {
                uint32_t got = mul ? d.mul_r : d.add_r;
                if (got != v[3*out + 2]) report(out, got);
                checked++;
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            d.ds_a = v[3*i]; d.ds_b = v[3*i + 1]; d.ds_op = sq; d.ds_start = 1;
            tick(); d.ds_start = 0;
            int c = 0;
            while (!d.ds_done) { tick(); require(++c < 100, "div/sqrt timeout"); }
            if (d.ds_r != v[3*i + 2]) report(i, d.ds_r);
        }
    }
    std::printf("%s: %zu vectors, %u mismatches\n", op, n, bad);
    return bad ? 1 : 0;
}

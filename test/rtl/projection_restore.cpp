// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Independent native binary32 cast and multiply, not RTL significand shifts.
#include "Vprojection_restore.h"
#include "verilated.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t seed=81525;
static uint32_t random_word() { seed=seed*1664525u+1013904223u; return seed; }
struct Word {
    std::array<uint32_t, 8> x;
    int exponent;
    bool first, last;
};
struct Result {
    std::array<uint32_t, 8> data;
    unsigned exponent, error;
    bool first, last;
    bool operator==(const Result &r) const {
        return data==r.data && exponent==r.exponent && error==r.error && first==r.first && last==r.last;
    }
};
static Result oracle(const Word &w) {
    const bool bad=w.exponent < -149 || w.exponent > 127;
    Result r={{}, unsigned(w.exponent) & 1023, bad ? 256u : 0u, w.first, w.last};
    for (unsigned lane=0; lane<8; lane++) {
        if (bad) continue;
        int32_t x;
        std::memcpy(&x, &w.x[lane], 4);
        volatile float converted=float(x); // Keep both firmware rounding boundaries.
        volatile float scale=std::ldexp(1.0f, w.exponent);
        float restored=converted*scale;
        if (!std::isfinite(restored)) r.error |= 1u << lane;
        else std::memcpy(&r.data[lane], &restored, 4);
    }
    return r;
}
static Result output(const Vprojection_restore &d) {
    Result r={{}, d.out_exponent, d.out_error, bool(d.out_first), bool(d.out_last)};
    for (unsigned lane=0; lane<8; lane++) r.data[lane]=d.out_data[lane];
    return r;
}
int main() {
    require(std::fegetround()==FE_TONEAREST, "oracle must use nearest-even rounding");
    Vprojection_restore d;
    d.sys_rst=1; d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); d.sys_rst=0;
    unsigned sent=0, received=0, discarded=0, cancels=0, resets=0;
    const unsigned count=40000;
    std::deque<Result> pending;
    std::deque<unsigned> accepted_cycles;
    Word offered{};
    Result held{};
    bool offering=false, stalled=false;
    for (unsigned cycle=0; cycle<400000; cycle++) {
        const bool cancel=(cycle==1 || cycle==17 || cycle==101 || cycle==501 || cycle==1001);
        const bool reset=(cycle==59 || cycle==555 || cycle==2003);
        d.cancel=cancel; d.sys_rst=reset;
        if (cancel || reset) {
            discarded += pending.size(); pending.clear(); accepted_cycles.clear(); stalled=false;
            cancels += cancel; resets += reset;
        }
        if (!offering && sent<count && (cycle<7000 || (random_word() & 3))) {
            offered.exponent=sent<8192 ? int(sent % 1024)-512 : int(random_word()%277)-149;
            offered.first=sent%17==0; offered.last=sent%17==16;
            for (auto &x : offered.x) x=random_word();
            if (sent<8192) {
                const unsigned p=24 + (sent/1024)%7;
                const uint32_t half=1u << (p-24), base=(1u << p)+half;
                offered.x={base-1,base,base+1,base+2*half,
                           0u,0x7fffffffu,0x80000000u,uint32_t(-int32_t(base))};
            }
            if (sent>=8192 && sent<16384) {
                offered.exponent=-149 + (sent % 25);
                // Real subnormal outputs and their neighboring normal values.
                const uint32_t base=1u << (sent % 24);
                offered.x={base-1,base,base+1,uint32_t(-int32_t(base)),
                           1u,0xffffffffu,0x7fffffu,0x800001u};
            }
            offering=true;
        }
        d.in_valid=offering && !reset;
        for (unsigned lane=0; lane<8; lane++) d.in_data[lane]=offered.x[lane];
        d.in_exponent=unsigned(offered.exponent) & 1023;
        d.in_first=offered.first; d.in_last=offered.last;
        d.out_ready=cycle<7000 || (random_word() & 3);
        d.sys_clk=0; d.eval();
        if (cancel) require(!d.in_ready && !d.out_valid, "cancel failed to mask handshakes");
        if (!cancel && !reset) {
            if (stalled) require(d.out_valid && output(d)==held, "stalled output changed");
            if (d.out_valid && d.out_ready) {
                require(!pending.empty(), "stale output after reset/cancel");
                if (cycle>2100 && cycle<6900)
                    require(cycle-accepted_cycles.front()==4, "wrong unstalled latency");
                if (!(output(d)==pending.front())) {
                    const auto r=pending.front();
                    std::fprintf(stderr, "cycle=%u exponent=%u error=%x/%x\n",
                                 cycle, d.out_exponent, r.error, d.out_error);
                    for (unsigned i=0; i<8; i++)
                        std::fprintf(stderr, " lane%u expected=%08x actual=%08x\n", i, r.data[i], d.out_data[i]);
                    require(false, "binary32 oracle mismatch");
                }
                pending.pop_front(); accepted_cycles.pop_front(); received++;
            }
            stalled=d.out_valid && !d.out_ready; held=output(d);
            if (d.in_valid && d.in_ready) {
                pending.push_back(oracle(offered)); accepted_cycles.push_back(cycle); sent++; offering=false;
            }
            if (cycle>2100 && cycle<6900) require(d.in_ready && d.out_valid, "bubble in sustained stream");
        }
        d.sys_clk=1; d.eval();
        if (sent==count && pending.empty()) {
            require(received+discarded==sent, "lost or duplicated words");
            std::printf("PROJECTION_RESTORE_PASS words=%u lanes=8 received=%u cancelled_words=%u cancels=%u resets=%u latency=4 interval=1\n",
                        sent, received, discarded, cancels, resets);
            return 0;
        }
    }
    require(false, "restoration timeout");
}

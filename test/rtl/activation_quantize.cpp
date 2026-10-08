// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Independent binary32 multiply + std::lround (not the RTL's significand shifts).
#include "Vactivation_quantize.h"
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t seed=983;
static uint32_t random_word() { seed=seed*1664525u+1013904223u; return seed; }
struct Word {
    std::array<uint32_t, 8> x;
    int exponent;
    bool first, last;
};
struct Result {
    uint64_t data;
    unsigned exponent, error;
    bool first, last;
    bool operator==(const Result &r) const {
        return data==r.data && exponent==r.exponent && error==r.error && first==r.first && last==r.last;
    }
};
static Result oracle(const Word &w) {
    const bool bad=w.exponent < -127 || w.exponent > 121;
    Result r={0, unsigned(w.exponent) & 1023, bad ? 256u : 0u, w.first, w.last};
    for (unsigned lane=0; lane<8; lane++) {
        float x;
        std::memcpy(&x, &w.x[lane], 4);
        int q=0;
        if (!std::isfinite(x)) r.error |= 1u << lane;
        else if (!bad) {
            const float scale=std::ldexp(1.0f, -w.exponent);
            volatile float scaled=x*scale;  // Force the firmware's binary32 multiply boundary.
            if (scaled >= 127.0f) q=127;
            else if (scaled <= -127.0f) q=-127;
            else q=int(std::lround(scaled));
        }
        r.data |= uint64_t(uint8_t(q)) << (lane*8);
    }
    return r;
}
static Result output(const Vactivation_quantize &d) {
    return {d.out_data, d.out_e8, d.out_error, bool(d.out_first), bool(d.out_last)};
}
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
int main() {
    Vactivation_quantize d;
    d.sys_rst=1; d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); d.sys_rst=0;
    unsigned sent=0, received=0, discarded=0, cancels=0, resets=0;
    const unsigned count=20000;
    std::deque<Result> pending;
    std::deque<unsigned> accepted_cycles;
    Word offered{};
    Result held{};
    bool offering=false, stalled=false;
    for (unsigned cycle=0; cycle<200000; cycle++) {
        // Separate reset and cancel while words occupy different pipeline stages.
        const bool cancel=(cycle==1 || cycle==17 || cycle==101 || cycle==501 || cycle==1001);
        const bool reset=(cycle==59 || cycle==555 || cycle==2003);
        d.cancel=cancel; d.sys_rst=reset;
        if (cancel || reset) {
            discarded += pending.size(); pending.clear(); accepted_cycles.clear(); stalled=false;
            cancels += cancel; resets += reset;
        }
        if (!offering && sent<count && (cycle<7000 || (random_word() & 3))) {
            offered.exponent=sent<16000 ? int(sent % 249)-127 : int(random_word() % 1024)-512;
            offered.first=sent%17==0; offered.last=sent%17==16;
            for (auto &x : offered.x) x=random_word();
            if (sent<16000) {
                // Exact half integers and adjacent binary32 values, both signs,
                // all supported exponents. Also exercise nonfinite/denormal lanes.
                float value=std::ldexp(float((sent/249)%127) + 0.5f, offered.exponent);
                const uint32_t u=bits(value);
                offered.x={u-1,u,u+1,(u-1)|0x80000000u,u|0x80000000u,(u+1)|0x80000000u,
                           sent%2 ? 0x7f800000u : 1u, sent%2 ? 0xffc00001u : 0x80000000u};
            }
            offering=true;
        }
        d.in_valid=offering && !reset;
        for (unsigned lane=0; lane<8; lane++) d.in_data[lane]=offered.x[lane];
        d.in_e8=unsigned(offered.exponent) & 1023;
        d.in_first=offered.first; d.in_last=offered.last;
        d.out_ready=cycle<7000 || (random_word() & 3);
        d.sys_clk=0; d.eval();
        if (cancel) require(!d.in_ready && !d.out_valid, "cancel failed to mask handshakes");
        if (!cancel && !reset) {
            if (stalled) require(d.out_valid && output(d)==held, "stalled payload changed");
            if (d.out_valid && d.out_ready) {
                require(!pending.empty(), "stale output after reset/cancel");
                if (cycle>2100 && cycle<6900)
                    require(cycle-accepted_cycles.front()==3, "wrong unstalled pipeline latency");
                if (!(output(d)==pending.front())) {
                    const auto r=pending.front();
                    std::fprintf(stderr, "cycle=%u expected=%016llx got=%016llx exponent=%u error=%x/%x\n",
                        cycle, (unsigned long long)r.data, (unsigned long long)d.out_data,
                        d.out_e8, r.error, d.out_error);
                    require(false, "binary32 oracle mismatch");
                }
                pending.pop_front(); accepted_cycles.pop_front(); received++;
            }
            stalled=d.out_valid && !d.out_ready; held=output(d);
            if (d.in_valid && d.in_ready) {
                pending.push_back(oracle(offered)); accepted_cycles.push_back(cycle); sent++; offering=false;
            }
            if (cycle>2100 && cycle<6900) require(d.in_ready && d.out_valid, "bubble in sustained one-word-per-cycle stream");
        }
        d.sys_clk=1; d.eval();
        if (sent==count && pending.empty()) {
            require(received+discarded==sent, "lost or duplicated words");
            std::printf("ACTIVATION_QUANTIZE_PASS words=%u lanes=8 received=%u cancelled_words=%u cancels=%u resets=%u latency=3 interval=1\n",
                        sent, received, discarded, cancels, resets);
            return 0;
        }
    }
    require(false, "quantizer timeout");
}

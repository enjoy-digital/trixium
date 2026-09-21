// SPDX-License-Identifier: BSD-2-Clause
// Independent dense parity oracle, including full signed-byte range and RAM reuse.
#include "Vinteger_hadamard.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static uint32_t rng = 19;
static uint32_t random_word() { rng = rng*1664525u + 1013904223u; return rng; }
static void tick(Vinteger_hadamard &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }

static void begin(Vinteger_hadamard &d, bool inverse) {
    d.cmd_payload_inverse=inverse; d.cmd_valid=1;
    d.sink_valid=0; d.source_ready=0;
    d.sys_clk=0; d.eval();
    require(d.cmd_ready, "not ready for a new block");
    tick(d); d.cmd_valid=0;
    require(d.busy, "command did not start");
}

static void cancel(Vinteger_hadamard &d) {
    d.cancel=1; d.eval();
    require(!d.source_valid && !d.sink_ready && !d.cmd_ready, "cancel did not mask interfaces");
    tick(d); d.cancel=0; d.sink_valid=0; d.cmd_valid=0; d.source_ready=1;
    tick(d);
    require(!d.busy && d.cancelled && !d.error && !d.done && !d.source_valid,
            "cancel did not flush state");
}

static void complete(Vinteger_hadamard &d, unsigned pattern, bool inverse, bool stalls) {
    std::array<int, 1024> x{}, signs{}, expected{};
    for (unsigned j=0; j<1024; j++) {
        unsigned r=random_word();
        x[j]=pattern==0 ? 0 : pattern==1 ? -128 : pattern==2 ? 127 : int(r >> 24)-128;
        signs[j]=pattern<2 ? -1 : (random_word() & 0x10000) ? 1 : -1;
    }
    unsigned maximum=0;
    for (unsigned i=0; i<1024; i++) {
        for (unsigned j=0; j<1024; j++)
            expected[i] += (__builtin_parity(i & j) ? -1 : 1)*x[j]*(inverse ? 1 : signs[j]);
        if (inverse) expected[i] *= signs[i];
        maximum=std::max(maximum, unsigned(std::abs(expected[i])));
    }
    begin(d, inverse);
    unsigned sent=0, received=0, first=0, last=0;
    bool offered=false, held=false;
    std::array<uint32_t, 7> previous{};
    for (unsigned cycle=0; cycle<20000; cycle++) {
        d.cmd_payload_inverse=!inverse; // The mode must have been latched.
        offered=offered || !stalls || random_word()%4 != 0;
        d.sink_valid=sent<128 && offered;
        d.sink_last=sent==127;
        d.sink_payload_data=0; d.sink_payload_negate=0;
        if (sent<128) for (unsigned lane=0; lane<8; lane++) {
            d.sink_payload_data |= uint64_t(uint8_t(x[sent*8+lane])) << (lane*8);
            d.sink_payload_negate |= (signs[sent*8+lane]<0) << lane;
        }
        d.source_ready=!stalls || random_word()%3 != 0;
        d.sys_clk=0; d.eval();
        std::array<uint32_t, 7> now{};
        for (unsigned i=0; i<5; i++) now[i]=d.source_payload_data[i];
        now[5]=d.source_first; now[6]=d.source_last;
        if (held) require(d.source_valid && now==previous, "unstable stalled output");
        held=d.source_valid && !d.source_ready; previous=now;
        if (d.sink_valid && d.sink_ready) { sent++; offered=false; }
        if (d.source_valid && d.source_ready) {
            require(received<128, "extra output word");
            require(d.source_first==(received==0) && d.source_last==(received==127), "bad output flags");
            for (unsigned lane=0; lane<8; lane++) {
                unsigned bit=lane*19, word=bit/32, shift=bit%32;
                uint64_t raw=uint64_t(d.source_payload_data[word]) >> shift;
                if (shift+19>32) raw |= uint64_t(d.source_payload_data[word+1]) << (32-shift);
                int value=int(raw & 0x7ffff);
                if (value & 0x40000) value -= 0x80000;
                if (value != expected[received*8+lane]) {
                    std::fprintf(stderr, "pattern=%u inverse=%u index=%u want=%d got=%d\n",
                                 pattern, inverse, received*8+lane, expected[received*8+lane], value);
                    require(false, "integer mismatch");
                }
            }
            if (!received) first=cycle;
            last=cycle; received++;
        }
        d.sys_clk=1; d.eval();
        require(!d.error, "unexpected error");
        if (d.done) {
            require(!d.busy && !d.cancelled && sent==128 && received==128, "premature completion");
            require(d.absmax==maximum, "wrong block absmax");
            if (!stalls) require(first==779 && last==906 && cycle==909, "unexpected dense cycle count");
            d.sink_valid=0; tick(d);
            require(!d.done && !d.source_valid, "completion/output repeated");
            return;
        }
    }
    require(false, "command timeout");
}

int main() {
    Vinteger_hadamard d;
    d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    unsigned completed=0;
    for (unsigned pattern=0; pattern<12; pattern++) for (bool inverse : {false, true}) {
        complete(d, pattern, inverse, pattern>=3); completed++;
    }
    for (unsigned stop : {0u, 3u, 127u, 128u, 192u, 400u, 777u, 778u, 800u, 950u}) {
        begin(d, true);
        for (unsigned cycle=0; cycle<stop; cycle++) {
            d.sink_valid=cycle<128; d.sink_last=cycle==127;
            d.sink_payload_data=UINT64_C(0x8080808080808080); d.sink_payload_negate=255;
            d.source_ready=0; tick(d);
        }
        cancel(d); complete(d, 3, false, true); completed++;
    }
    for (unsigned stop : {800u, 850u, 905u}) {
        begin(d, false);
        unsigned prefix=0;
        for (unsigned cycle=0; cycle<stop; cycle++) {
            d.sink_valid=cycle<128; d.sink_last=cycle==127;
            d.sink_payload_data=UINT64_C(0x7f807f807f807f80); d.sink_payload_negate=0x55;
            d.source_ready=1; d.sys_clk=0; d.eval();
            if (d.source_valid) prefix++;
            d.sys_clk=1; d.eval();
            require(!d.done, "partial-output cancellation test completed too soon");
        }
        require(prefix>0 && prefix<128, "cancellation did not exercise a partial result");
        cancel(d); complete(d, 4, true, true); completed++;
    }
    for (unsigned stop : {907u, 908u, 909u}) {
        begin(d, true);
        for (unsigned cycle=0; cycle<stop; cycle++) {
            d.sink_valid=cycle<128; d.sink_last=cycle==127;
            d.sink_payload_data=UINT64_C(0x8080808080808080); d.sink_payload_negate=255;
            d.source_ready=1; tick(d);
            require(!d.done, "absmax-drain cancellation completed too soon");
        }
        cancel(d); complete(d, 5, false, true); completed++;
    }
    for (unsigned stop : {500u, 850u, 908u}) {
        begin(d, true);
        for (unsigned cycle=0; cycle<stop; cycle++) {
            d.sink_valid=cycle<128; d.sink_last=cycle==127;
            d.sink_payload_data=UINT64_C(0x8080808080808080); d.sink_payload_negate=255;
            d.source_ready=stop==908; tick(d);
        }
        d.sys_rst=1; tick(d); d.sys_rst=0; d.sink_valid=0; tick(d);
        require(!d.busy && !d.cancelled && !d.error && !d.done && !d.source_valid,
                "system reset did not flush pending state");
        complete(d, 5, false, true); completed++;
    }
    for (bool early : {false, true}) {
        begin(d, false);
        for (unsigned word=0; word<(early ? 1u : 128u); word++) {
            d.sink_valid=1; d.sink_last=early; tick(d);
        }
        d.sink_valid=0;
        for (unsigned i=0; i<5; i++) {
            tick(d); require(d.error && d.busy && !d.done && !d.source_valid, "framing not rejected");
        }
        cancel(d); complete(d, 2, true, false); completed++;
    }
    std::printf("HADAMARD_COMPILED_PASS blocks=%u cancelled=16 reset=3 framing=2 dense_first=779 dense_done=909\n", completed);
}

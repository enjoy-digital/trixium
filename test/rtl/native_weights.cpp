// SPDX-License-Identifier: BSD-2-Clause
// Compiled-Verilog check independent of Migen's arithmetic expression evaluator.
#include "Vnative_weights.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}

static void reference(const uint8_t *p, bool ptq, uint32_t words[8], unsigned &scale) {
    for (int i = 0; i < 8; i++) words[i] = 0;
    if (ptq) {
        // Scalar transliteration of the reference's run ordering, not an RTL mux.
        unsigned element = 0, byte = 0;
        for (unsigned run : {32u, 16u, 8u}) {
            while (byte + run <= 24) {
                unsigned power = 1;
                for (unsigned trit = 0; trit < 5; trit++, power *= 3) {
                    for (unsigned m = 0; m < run; m++, element++) {
                        unsigned code = (((unsigned(p[byte + m])*power) & 255)*3) >> 8;
                        words[element / 16] |= code << (2*(element % 16));
                    }
                }
                byte += run;
            }
        }
        unsigned power = 1;
        for (unsigned trit = 0; trit < 4; trit++, power *= 3) {
            for (unsigned m = 0; m < 2; m++, element++) {
                unsigned code = (((unsigned(p[24 + m])*power) & 255)*3) >> 8;
                words[element / 16] |= code << (2*(element % 16));
            }
        }
        require(element == 128, "reference element count");
        scale = p[26] | (unsigned(p[27]) << 8);
    } else {
        scale = p[0] | (unsigned(p[1]) << 8);
        for (unsigned i = 0; i < 32; i++) words[i/4] |= unsigned(p[i+2]) << (8*(i % 4));
    }
}

int main(int argc, char **argv) {
    require(argc == 5, "arguments: raw encoding rows groups");
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), {});
    unsigned encoding = std::strtoul(argv[2], nullptr, 0);
    unsigned rows = std::strtoul(argv[3], nullptr, 0), groups = std::strtoul(argv[4], nullptr, 0);
    const bool ptq = encoding == 5;
    unsigned block = ptq ? 28 : 34, total = rows*groups;
    require(data.size() == total*block, "input bytes");
    while (data.size() % 32) data.push_back(0xa5);
    Vnative_weights dut;
    auto clock = [&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1;
    clock(); clock();
    dut.sys_rst=0;
    dut.rows=rows; dut.groups=groups; dut.encoding=encoding; dut.start=1;
    clock();
    dut.start=0;
    dut.rows=65535; dut.groups=65535; dut.encoding=255; // Quasi-static values were latched.
    unsigned sent=0, received=0, state=17, held[15] = {};
    bool offered=false, stalled=false;
    auto random = [&]() { state = state*1664525u + 1013904223u; return state; };
    for (unsigned cycle=0; cycle<20000; cycle++) {
        offered = offered || random()%4 != 0;
        dut.sink_valid = sent < data.size()/32 && offered;
        dut.sink_last = sent + 1 == data.size()/32;
        if (sent < data.size()/32) {
            for (unsigned i=0; i<8; i++) {
                unsigned word=0;
                for (unsigned k=0; k<4; k++) word |= unsigned(data[sent*32+i*4+k]) << (8*k);
                dut.sink_payload_data[i] = word;
            }
        }
        dut.source_ready = cycle >= 60 && random()%3 != 0;
        dut.sys_clk=0; dut.eval();
        unsigned out[15];
        for (unsigned i=0; i<8; i++) out[i]=dut.source_payload_codes[i];
        out[8]=dut.source_payload_scale; out[9]=dut.source_payload_row;
        out[10]=dut.source_payload_group; out[11]=dut.source_payload_row_last;
        out[12]=dut.source_first; out[13]=dut.source_last;
        out[14]=dut.source_payload_kind;
        if (stalled) {
            require(dut.source_valid, "valid withdrawn under backpressure");
            for (unsigned i=0; i<15; i++) require(out[i] == held[i], "payload changed under backpressure");
        }
        stalled = dut.source_valid && !dut.source_ready;
        for (unsigned i=0; i<15; i++) held[i]=out[i];
        if (dut.sink_valid && dut.sink_ready) { sent++; offered=false; }
        if (dut.source_valid && dut.source_ready) {
            require(received < total, "duplicate output");
            uint32_t words[8]; unsigned scale;
            reference(&data[received*block], ptq, words, scale);
            for (unsigned i=0; i<8; i++) {
                if (words[i] != out[i]) {
                    std::fprintf(stderr, "group %u word %u: expected %08x, got %08x, cycle %u\n",
                                 received, i, words[i], out[i], cycle);
                    require(false, "wrong code bits");
                }
            }
            require(scale == out[8], "wrong scale bits");
            require(out[9] == received/groups && out[10] == received%groups, "wrong row/group");
            require(out[11] == (received%groups == groups-1), "wrong row-last");
            require(out[12] == (received == 0) && out[13] == (received == total-1), "wrong command tags");
            require(out[14] == !ptq && !dut.source_payload_error, "wrong kind/error");
            received++;
        }
        dut.sys_clk=1; dut.eval();
        require(dut.buffered_bytes <= 64, "reservoir overflow");
        if (dut.done) {
            require(!dut.busy && !dut.error && !dut.cancelled, "failed status");
            require(received == total && sent == data.size()/32, "incomplete operation");
            std::printf("%u groups, %u beats: exact\n", received, sent);
            return 0;
        }
    }
    require(false, "timeout");
}

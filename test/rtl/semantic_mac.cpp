// SPDX-License-Identifier: BSD-2-Clause
#include "Vsemantic_mac.h"
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
static uint32_t word(const uint8_t *p) {
    return p[0] | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
int main(int argc, char **argv) {
    require(argc == 2 || argc == 3, "vector file required");
    const bool expect_error = argc == 3;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    require(data.size() >= 40, "short vector");
    unsigned h[10];
    for (int i=0; i<10; i++) h[i]=word(&data[4*i]);
    unsigned kind=h[0], rows=h[1], groups=h[2], qn=h[3], nx=h[7], nw=h[8], nout=h[9];
    require(data.size() == 40+nx+nw+4*nout && nx%32 == 0 && nw%32 == 0, "vector size mismatch");
    const uint8_t *x=&data[40], *weights=x+nx, *expected=weights+nw;
    Vsemantic_mac dut;
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0;
    for (unsigned offset=0; offset<nx; offset+=32) {
        dut.p_xw_group=offset/128; dut.p_xw_sub=(offset/32)%4; dut.p_xw_we=1;
        for (unsigned i=0; i<8; i++) dut.p_xw_data[i]=word(x+offset+4*i);
        clock();
    }
    dut.p_xw_we=0;
    dut.p_kind=kind; dut.p_rows=rows; dut.p_code_beats=groups; dut.p_qn=qn;
    dut.p_bias=h[4]; dut.p_rshift=h[5]; dut.p_rshift2=h[6];
    dut.d_rows=rows; dut.d_groups=groups; dut.d_encoding=kind == 0 ? 5 : 6;
    dut.p_start=1; dut.d_start=1; clock(); dut.p_start=0; dut.d_start=0;
    unsigned state=41, sent=0, received=0;
    auto random=[&]() { state=state*1664525u+1013904223u; return state; };
    std::vector<bool> seen(nout, false);
    bool offered=false, decoded=false;
    for (unsigned cycle=0; cycle<20000; cycle++) {
        offered = offered || random()%4 != 0;
        dut.sink_valid=sent<nw/32 && offered; dut.sink_last=sent+1==nw/32;
        if (sent<nw/32) for (unsigned i=0; i<8; i++) dut.sink_data[i]=word(weights+sent*32+4*i);
        dut.source_ready=cycle>=200 && random()%3 != 0;
        dut.sys_clk=0; dut.eval();
        if (dut.sink_valid && dut.sink_ready) { sent++; offered=false; }
        if (dut.source_valid && dut.source_ready && !expect_error) {
            require(dut.source_wide == (kind == 2), "wrong wide result flag");
            unsigned stride=kind == 2 ? groups*32 : rows;
            for (unsigned lane=0; lane<(dut.source_wide ? 8u : 1u); lane++) {
                unsigned i=dut.source_head*stride+dut.source_row+lane;
                require(i<nout && !seen[i], "out of range or duplicate result");
                if (dut.source_data[lane] != word(expected+4*i)) {
                    std::fprintf(stderr, "result %u: expected %08x, got %08x\n", i,
                                 word(expected+4*i), unsigned(dut.source_data[lane]));
                    require(false, "numerical mismatch");
                }
                seen[i]=true; received++;
            }
        }
        dut.sys_clk=1; dut.eval();
        decoded = decoded || dut.d_done;
        require(!dut.d_error && !dut.p_overflow, "framing/FIFO error");
        if (dut.p_error) {
            if (!expect_error || !dut.p_arithmetic_error)
                std::fprintf(stderr, "cycle %u: MAC error, arithmetic=%u\n", cycle, unsigned(dut.p_arithmetic_error));
            require(expect_error && dut.p_arithmetic_error, "unexpected MAC error");
            std::puts("arithmetic overflow detected");
            return 0;
        }
        if (received==nout && decoded && !expect_error) {
            require(sent==nw/32, "premature result");
            dut.sink_valid=0; dut.source_ready=1;
            for (int i=0; i<16; i++) { clock(); require(!dut.source_valid, "late duplicate result"); }
            std::printf("%u exact results\n", received);
            return 0;
        }
    }
    require(false, "timeout");
}

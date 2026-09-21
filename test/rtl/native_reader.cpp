// SPDX-License-Identifier: BSD-2-Clause
// Compiled region->AXI->byte spans->native groups; independent clock/AXI driver.
#include "Vnative_reader.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iterator>
#include <unordered_map>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t word(const uint8_t *p) {
    return p[0] | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
static uint64_t wide(const uint8_t *p) { return word(p) | uint64_t(word(p+4))<<32; }
struct Burst { uint64_t addr; unsigned left, ready_at; };

int main(int argc, char **argv) {
    require(argc == 2, "vector file required");
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    require(data.size() >= 56, "short header");
    unsigned h[10];
    for (unsigned i=0; i<10; i++) h[i]=word(&data[4*i]);
    uint64_t offset=wide(&data[40]), stride=wide(&data[48]);
    const unsigned error=h[4], flags=h[5], nmem=h[6], nout=h[7], nread=h[8];
    require(data.size() == 56+40*nmem+64*nout+8*nread, "vector length");
    std::unordered_map<uint64_t, std::array<uint32_t, 8>> memory;
    for (unsigned i=0; i<nmem; i++) {
        const uint8_t *p=&data[56+40*i];
        std::array<uint32_t, 8> words;
        for (unsigned j=0; j<8; j++) words[j]=word(p+8+4*j);
        memory.emplace(wide(p), words);
    }
    const uint8_t *expected=&data[56+40*nmem], *reads=expected+64*nout;
    Vnative_reader dut;
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0;
    dut.cmd_encoding=h[0]; dut.cmd_rows=h[1]; dut.cmd_groups=h[2]; dut.cmd_region=h[3];
    dut.cmd_offset=offset; dut.cmd_stride=stride; dut.cmd_valid=1;
    dut.sys_clk=0; dut.eval(); require(dut.cmd_ready, "initial command not ready");
    dut.sys_clk=1; dut.eval(); dut.cmd_valid=0;
    dut.cmd_encoding=255; dut.cmd_rows=0; dut.cmd_groups=0;
    dut.cmd_offset=~uint64_t(0); dut.cmd_stride=~uint64_t(0); dut.cmd_region=0;
    unsigned state=41, received=0, issued=0, held_id=0;
    auto random=[&]() { state=state*1664525u+1013904223u; return state; };
    std::array<std::deque<Burst>, 4> pending;
    bool response_held=false, output_held=false, completed=dut.done;
    std::array<uint32_t, 16> saved;
    for (unsigned cycle=0; cycle<30000; cycle++) {
        dut.ar_ready=random()%4 != 0;
        if (!response_held) {
            unsigned start=random()%4;
            for (unsigned j=0; j<4; j++) {
                unsigned id=(start+j)%4;
                if (!pending[id].empty() && pending[id].front().ready_at <= cycle) {
                    held_id=id; response_held=true; break;
                }
            }
        }
        dut.r_valid=response_held;
        if (response_held) {
            const auto &burst=pending[held_id].front();
            auto entry=memory.find(burst.addr);
            require(entry != memory.end(), "unexpected memory address");
            for (unsigned j=0; j<8; j++) dut.r_data[j]=entry->second[j];
            dut.r_id=held_id+(flags & 4 ? 4 : 0);
            dut.r_last=burst.left == 1; dut.r_resp=flags & 1 ? 2 : 0;
        }
        dut.cancel=(flags & 2) && cycle == 60;
        dut.source_ready=cycle>=90 && random()%3 != 0;
        dut.sys_clk=0; dut.eval();
        if (dut.ar_valid && dut.ar_ready) {
            const uint64_t addr=dut.ar_addr;
            const unsigned beats=dut.ar_len+1, id=dut.ar_id;
            require(id<4 && beats<=16 && addr%32 == 0 && (addr & 4095)+beats*32<=4096,
                    "illegal AXI burst");
            require(dut.ar_size == 5 && dut.ar_burst == 1, "wrong AXI attributes");
            for (unsigned j=0; j<beats; j++) {
                require(issued<nread && addr+32*j == wide(reads+8*issued), "wrong read order/range");
                issued++;
            }
            pending[id].push_back({addr, beats, cycle+10});
        }
        if (response_held && dut.r_ready) {
            auto &burst=pending[held_id].front();
            burst.addr+=32; burst.left--;
            if (!burst.left) pending[held_id].pop_front();
            response_held=false;
        }
        std::array<uint32_t, 16> out;
        for (unsigned j=0; j<8; j++) out[j]=dut.source_codes[j];
        out[8]=dut.source_scale; out[9]=dut.source_row; out[10]=dut.source_group;
        out[11]=dut.source_row_last; out[12]=dut.source_first; out[13]=dut.source_last;
        out[14]=dut.source_kind; out[15]=dut.source_error;
        if (output_held && !flags && !error)
            require(dut.source_valid && out == saved, "unstable stalled output");
        output_held=dut.source_valid && !dut.source_ready; saved=out;
        if (dut.source_valid && dut.source_ready) {
            require(received<nout, "duplicate output");
            for (unsigned j=0; j<16; j++) {
                if (out[j] != word(expected+64*received+4*j)) {
                    std::fprintf(stderr, "group %u field %u: %08x != %08x\n", received, j,
                                 unsigned(out[j]), word(expected+64*received+4*j));
                    require(false, "native group mismatch");
                }
            }
            received++;
        }
        dut.sys_clk=1; dut.eval();
        completed=completed || dut.done;
        if (completed) {
            if (dut.error != error)
                std::fprintf(stderr, "error %u != %u\n", unsigned(dut.error), error);
            require(dut.error == error && bool(dut.cancelled) == bool(flags & 2), "wrong status");
            require(!dut.busy && dut.reader_idle && !response_held, "completed before AXI drain");
            for (const auto &queue:pending) require(queue.empty(), "outstanding burst at completion");
            if (!flags && !error) require(received == nout && issued == nread, "premature completion");
            std::printf("%u native groups, %u aligned read beats\n", received, issued);
            return 0;
        }
    }
    require(false, "reader timeout");
}

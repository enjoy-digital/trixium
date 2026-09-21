// SPDX-License-Identifier: BSD-2-Clause
#include "Vnative_engine.h"
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
struct Port {
    CData *ar_valid, *ar_ready, *ar_len, *ar_size, *ar_burst, *ar_id;
    QData *ar_addr;
    CData *r_valid, *r_ready, *r_last, *r_id, *r_resp;
    VlWide<8> *r_data;
};
int main(int argc, char **argv) {
    require(argc == 2, "vector file required");
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    require(data.size() >= 168, "short header");
    unsigned h[32]; for (unsigned i=0; i<32; i++) h[i]=word(&data[4*i]);
    require(h[0] == 0x42534e45 && data.size() == 168+h[19]+4*h[18]+40*h[15]+8*(h[16]+h[17]), "vector size");
    const uint8_t *descs=&data[128], *xb=&data[168], *expected=xb+h[19], *rawmem=expected+4*h[18];
    const uint8_t *reads[2]={rawmem+40*h[15], rawmem+40*h[15]+8*h[16]};
    std::unordered_map<uint64_t, std::array<uint32_t, 8>> memory;
    for (unsigned i=0; i<h[15]; i++) {
        const uint8_t *p=rawmem+40*i;
        std::array<uint32_t, 8> words; for (unsigned j=0; j<8; j++) words[j]=word(p+8+4*j);
        memory.emplace(wide(p), words);
    }
    Vnative_engine dut;
    Port ports[2]={
        {&dut.ar0_valid, &dut.ar0_ready, &dut.ar0_len, &dut.ar0_size, &dut.ar0_burst, &dut.ar0_id,
         &dut.ar0_addr, &dut.r0_valid, &dut.r0_ready, &dut.r0_last, &dut.r0_id, &dut.r0_resp, &dut.r0_data},
        {&dut.ar1_valid, &dut.ar1_ready, &dut.ar1_len, &dut.ar1_size, &dut.ar1_burst, &dut.ar1_id,
         &dut.ar1_addr, &dut.r1_valid, &dut.r1_ready, &dut.r1_last, &dut.r1_id, &dut.r1_resp, &dut.r1_data},
    };
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0;
    for (unsigned pos=0; pos<h[19]; pos+=32) {
        dut.sp_addr=pos/32; dut.sp_we=1;
        for (unsigned j=0; j<8; j++) dut.sp_data[j]=word(xb+pos+4*j);
        clock();
    }
    dut.sp_we=0;
    for (unsigned i=0; i<2; i++) {
        if (!(h[24] & (1u << i))) continue;
        dut.cfg_chan=i; dut.cfg_region=word(descs+20*i);
        dut.cfg_offset=wide(descs+20*i+4); dut.cfg_stride=wide(descs+20*i+12); dut.cfg_valid=1;
        dut.sys_clk=0; dut.eval(); require(dut.cfg_ready, "descriptor blocked");
        dut.sys_clk=1; dut.eval();
    }
    dut.cfg_valid=0;
    dut.cmd_native=h[1]; dut.cmd_kind=h[2]; dut.cmd_rows=h[3]; dut.cmd_rem=h[4]; dut.cmd_cols=h[5];
    dut.cmd_cbeats=h[6]; dut.cmd_bias=h[7]; dut.cmd_rshift=h[8]; dut.cmd_qn=h[9]; dut.cmd_ilv=h[10];
    dut.cmd_dst=h[11]; dut.cmd_qstride=h[12]; dut.cmd_wlen=h[13]; dut.cmd_sbeats=h[14]; dut.cmd_wbase=0x800;
    dut.cmd_rshift2=h[20]; dut.cmd_xsrc=h[21]; dut.cmd_supw=h[22]; dut.cmd_xred=h[23];
    dut.cmd_valid=1; dut.sys_clk=0; dut.eval(); require(dut.cmd_ready, "command blocked");
    dut.sys_clk=1; dut.eval(); dut.cmd_valid=0;
    dut.cmd_rows=0; dut.cmd_cols=0; dut.cmd_kind=7; dut.cmd_native=!h[1];
    std::array<std::array<std::deque<Burst>, 4>, 2> pending;
    unsigned state=h[27], issued[2]={0, 0}, held_id[2]={0, 0};
    bool held[2]={false, false}, completed=false;
    // Use the high bits: the low two LCG bits repeat every four calls, which
    // can phase-lock a two-port idle bus into permanently deasserting ARREADY.
    auto random=[&]() { state=state*1664525u+1013904223u; return state >> 16; };
    for (unsigned cycle=0; cycle<30000; cycle++) {
        for (unsigned port=0; port<2; port++) {
            auto &p=ports[port]; *p.ar_ready=random()%4 != 0;
            if (!held[port]) {
                unsigned start=random()%4;
                for (unsigned j=0; j<4; j++) {
                    unsigned id=(start+j)%4;
                    if (!pending[port][id].empty() && pending[port][id].front().ready_at <= cycle) {
                        held_id[port]=id; held[port]=true; break;
                    }
                }
            }
            *p.r_valid=held[port];
            if (held[port]) {
                auto &b=pending[port][held_id[port]].front();
                auto entry=memory.find(b.addr); require(entry != memory.end(), "unexpected read address");
                for (unsigned j=0; j<8; j++) (*p.r_data)[j]=entry->second[j];
                *p.r_id=held_id[port]; *p.r_last=b.left == 1; *p.r_resp=0;
            }
        }
        dut.sys_clk=0; dut.eval();
        for (unsigned port=0; port<2; port++) {
            auto &p=ports[port];
            if (*p.ar_valid && *p.ar_ready) {
                uint64_t addr=*p.ar_addr; unsigned beats=*p.ar_len+1, id=*p.ar_id;
                require(id<4 && beats<=16 && addr%32 == 0 && (addr & 4095)+beats*32<=4096, "illegal AXI burst");
                require(*p.ar_size == 5 && *p.ar_burst == 1, "wrong AXI attributes");
                for (unsigned j=0; j<beats; j++) {
                    require(issued[port]<h[16+port] && addr+32*j == wide(reads[port]+8*issued[port]), "wrong physical read order/range");
                    issued[port]++;
                }
                pending[port][id].push_back({addr, beats, cycle+6+random()%15});
            }
            if (held[port] && *p.r_ready) {
                auto &b=pending[port][held_id[port]].front(); b.addr+=32; b.left--;
                if (!b.left) pending[port][held_id[port]].pop_front();
                held[port]=false;
            }
        }
        const bool done_before_edge=dut.done;
        if (h[22]) require(!dut.gather_write, "suppressed command wrote results");
        if (h[25]) require(!done_before_edge && !dut.gather_write, "failed command committed results");
        dut.sys_clk=1; dut.eval();
        if (h[25]) {
            require(!dut.done && !dut.gather_write, "failed command committed results");
            if (!h[26]) require(!issued[0] && !issued[1], "invalid command issued memory reads");
            if (dut.native_failed) {
                require(dut.error && dut.busy && !dut.cmd_ready, "failed command did not remain locked");
                // Failure must not auto-complete or reopen the command port.
                for (unsigned j=0; j<20; j++) {
                    clock();
                    require(dut.native_failed && dut.error && dut.busy && !dut.done && !dut.cmd_ready,
                            "failed command silently recovered");
                }
                std::printf("invalid command rejected without successful completion\n");
                return 0;
            }
            continue;
        }
        require(!dut.error && !dut.native_failed, "engine failed");
        if (dut.done || done_before_edge) { completed=true; break; }
    }
    if (!completed) std::fprintf(stderr, "state=%u, results=%u expected=%u, issued=%u complete=%u, native busy=%u/%u, reader idle=%u/%u, reads=%u/%u\n",
        unsigned(dut.state), unsigned(dut.result_count), unsigned(dut.expected_count), unsigned(dut.issued), unsigned(dut.completed),
        unsigned(dut.native_busy0), unsigned(dut.native_busy1), unsigned(dut.reader_idle0), unsigned(dut.reader_idle1), issued[0], issued[1]);
    require(completed, "engine timeout");
    for (unsigned i=0; i<2; i++) {
        require(issued[i] == h[16+i] && !held[i], "premature completion");
        for (const auto &q:pending[i]) require(q.empty(), "outstanding read at completion");
    }
    clock(); clock(); clock();
    const unsigned rows=h[3]*2+h[4];
    int32_t maximum=INT32_MIN; unsigned argmax=0;
    for (unsigned i=0; i<h[18]; i++) {
        unsigned slot=h[11]+4*(i%rows+(i/rows)*h[12]);
        dut.sp_addr=slot/32; dut.eval();
        uint32_t wanted=word(expected+4*i), got=dut.sp_result[slot%32/4];
        if (!h[22] && wanted != got) std::fprintf(stderr, "result %u: %08x != %08x\n", i, got, wanted);
        if (!h[22]) require(wanted == got, "scratchpad numerical mismatch");
        if (int32_t(wanted)>maximum) { maximum=int32_t(wanted); argmax=i; }
    }
    if (!h[9] && !h[10]) require(dut.argmax_idx == argmax && dut.argmax_val == uint32_t(maximum), "wrong argmax");
    std::printf("%u exact results; %u/%u physical beats\n", h[18], issued[0], issued[1]);
    return 0;
}

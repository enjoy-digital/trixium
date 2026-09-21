// SPDX-License-Identifier: BSD-2-Clause
#include "Vregion_read.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
struct Port {
    CData *av, *ar, *alen, *aid, *asz, *aburst;
    QData *addr;
    CData *rv, *rr, *rl, *rid, *resp;
    VlWide<8> *data;
};
struct Burst { uint64_t addr; unsigned left, ready_at; };
static uint32_t payload(uint64_t addr, unsigned lane) {
    return uint32_t(addr) ^ uint32_t(addr >> 32)*0xa6398b17u ^ (lane*7919+0x5b0b0000u);
}
int main(int argc, char **argv) {
    require(argc == 8, "region offset length dst physical port fault required");
    const unsigned region=std::strtoul(argv[1], nullptr, 0), count=std::strtoul(argv[3], nullptr, 0);
    const unsigned dst=std::strtoul(argv[4], nullptr, 0), selected=std::strtoul(argv[6], nullptr, 0);
    const unsigned fault=std::strtoul(argv[7], nullptr, 0);
    const uint64_t offset=std::strtoull(argv[2], nullptr, 0), physical=std::strtoull(argv[5], nullptr, 0);
    Vregion_read dut;
    Port ports[2]={
        {&dut.ar0_valid,&dut.ar0_ready,&dut.ar0_len,&dut.ar0_id,&dut.ar0_size,&dut.ar0_burst,&dut.ar0_addr,
         &dut.r0_valid,&dut.r0_ready,&dut.r0_last,&dut.r0_id,&dut.r0_resp,&dut.r0_data},
        {&dut.ar1_valid,&dut.ar1_ready,&dut.ar1_len,&dut.ar1_id,&dut.ar1_size,&dut.ar1_burst,&dut.ar1_addr,
         &dut.r1_valid,&dut.r1_ready,&dut.r1_last,&dut.r1_id,&dut.r1_resp,&dut.r1_data},
    };
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0;
    // Neither a vector operation, native descriptor, nor legacy command may
    // race a readback submission. Probe readiness without clocking acceptance.
    dut.cmd_valid=1;
    dut.vec_busy=1; dut.sys_clk=0; dut.eval(); require(!dut.cmd_ready, "read raced vector"); dut.vec_busy=0;
    dut.cfg_valid=1; dut.eval(); require(!dut.cmd_ready, "read raced descriptor"); dut.cfg_valid=0;
    dut.legacy_valid=1; dut.eval(); require(!dut.cmd_ready, "read raced legacy command"); dut.legacy_valid=0;
    dut.cmd_region=region; dut.cmd_offset=offset; dut.cmd_length=count; dut.cmd_dst=dst;
    dut.eval(); require(dut.cmd_ready, "command blocked"); dut.sys_clk=1; dut.eval();
    dut.cmd_valid=0; dut.cmd_region=99; dut.cmd_offset=~uint64_t(0); dut.cmd_length=0; dut.cmd_dst=0;
    std::array<std::array<std::deque<Burst>,4>,2> pending;
    unsigned issued[2]={0,0}, delivered[2]={0,0}, held_id[2]={0,0}, state=197;
    bool held[2]={false,false}, stalled[2]={false,false};
    std::array<uint64_t,6> saved[2];
    auto random=[&]() { state=1664525u*state+1013904223u; return state >> 16; };
    bool failed=false, complete=false;
    for (unsigned cycle=0; cycle<50000; cycle++) {
        for (unsigned i=0; i<2; i++) {
            auto &p=ports[i]; *p.ar=random()%4!=0;
            if (!held[i]) {
                unsigned start=random()%4;
                for (unsigned j=0; j<4; j++) {
                    unsigned id=(start+j)%4;
                    if (!pending[i][id].empty() && pending[i][id].front().ready_at<=cycle && random()%4!=0) {
                        held_id[i]=id; held[i]=true; break;
                    }
                }
            }
            *p.rv=held[i] && fault!=6; // missing responses must not fabricate completion
            if (held[i]) {
                auto &b=pending[i][held_id[i]].front();
                for (unsigned j=0; j<8; j++) (*p.data)[j]=payload(b.addr,j);
                *p.rid=fault==3 ? held_id[i]+4 : held_id[i];
                *p.resp=fault==2 ? 2 : 0;
                *p.rl=fault==4 ? 1 : fault==5 ? 0 : b.left==1;
            }
        }
        dut.sys_clk=0; dut.eval();
        if (fault==1) require(!dut.gather_write, "invalid span wrote scratchpad");
        for (unsigned i=0; i<2; i++) {
            auto &p=ports[i];
            std::array<uint64_t,6> addr={*p.av,*p.addr,*p.alen,*p.aid,*p.asz,*p.aburst};
            if (stalled[i]) require(addr==saved[i], "unstable stalled AR");
            stalled[i]=*p.av && !*p.ar; saved[i]=addr;
            if (*p.av && *p.ar) {
                require(i==selected && fault!=1, "invalid region issued AR");
                const unsigned beats=*p.alen+1;
                require(beats<=16 && *p.addr%32==0 && (*p.addr%4096)+beats*32<=4096, "invalid burst");
                require(*p.aid<4 && *p.asz==5 && *p.aburst==1, "invalid AXI attributes");
                require(*p.addr==physical+32*issued[i] && issued[i]+beats<=count, "physical address mismatch");
                pending[i][*p.aid].push_back({*p.addr,beats,cycle+10+random()%30}); issued[i]+=beats;
            }
            if (*p.rv && *p.rr) {
                auto &queue=pending[i][held_id[i]]; auto &b=queue.front();
                delivered[i]++; b.addr+=32; if (!--b.left) queue.pop_front(); held[i]=false;
            }
        }
        const bool done=dut.done;
        dut.sys_clk=1; dut.eval();
        if (fault) {
            require(!done && !dut.done, "failed read reported success");
            if (dut.failed) { require(dut.error && dut.busy && !dut.cmd_ready, "unlatched failure"); failed=true; break; }
            if (fault==6 && cycle>=2000) { require(dut.busy && !dut.cmd_ready, "missing R lost ownership"); failed=true; break; }
        } else {
            require(!dut.error && !dut.failed, "valid read failed");
            if (done || dut.done) {
                require(issued[selected]==count && delivered[selected]==count && dut.beats==count, "short read");
                require(selected ? dut.idle1 && dut.drained1 : dut.idle0 && dut.drained0, "early completion");
                complete=true; break;
            }
        }
    }
    require(failed || complete, "read timed out");
    if (complete) {
        clock();
        for (unsigned b=0; b<count; b++) {
            dut.sp_addr=dst/32+b; dut.sys_clk=0; dut.eval();
            for (unsigned j=0; j<8; j++) require(dut.sp_result[j]==payload(physical+32*b,j), "scratchpad data mismatch");
        }
    }
    std::printf("%s: issued=%u delivered=%u error=%u\n", failed ? "failed safely" : "exact readback",
                issued[selected],delivered[selected],unsigned(dut.error));
}

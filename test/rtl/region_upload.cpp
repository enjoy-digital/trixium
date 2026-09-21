// SPDX-License-Identifier: BSD-2-Clause
#include "Vregion_upload.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
struct Port {
    CData *av, *ar, *alen, *aid, *asz, *aburst;
    QData *addr;
    CData *wv, *wr, *wl, *bv, *br, *bid, *bresp;
    IData *strb;
    VlWide<8> *data;
};
struct AW { uint64_t addr; unsigned beats; };
struct W { unsigned beats, first; };
int main(int argc, char **argv) {
    require(argc == 7, "region offset beats physical port fault required");
    const unsigned region=std::strtoul(argv[1], nullptr, 0), count=std::strtoul(argv[3], nullptr, 0);
    const uint64_t offset=std::strtoull(argv[2], nullptr, 0), physical=std::strtoull(argv[4], nullptr, 0);
    const unsigned selected=std::strtoul(argv[5], nullptr, 0), fault=std::strtoul(argv[6], nullptr, 0);
    Vregion_upload dut;
    Port ports[2]={
        {&dut.aw0_valid, &dut.aw0_ready, &dut.aw0_len, &dut.aw0_id, &dut.aw0_size, &dut.aw0_burst, &dut.aw0_addr,
         &dut.w0_valid, &dut.w0_ready, &dut.w0_last, &dut.b0_valid, &dut.b0_ready, &dut.b0_id, &dut.b0_resp,
         &dut.w0_strb, &dut.w0_data},
        {&dut.aw1_valid, &dut.aw1_ready, &dut.aw1_len, &dut.aw1_id, &dut.aw1_size, &dut.aw1_burst, &dut.aw1_addr,
         &dut.w1_valid, &dut.w1_ready, &dut.w1_last, &dut.b1_valid, &dut.b1_ready, &dut.b1_id, &dut.b1_resp,
         &dut.w1_strb, &dut.w1_data},
    };
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0;
    dut.cmd_region=region; dut.cmd_offset=offset; dut.cmd_length=count; dut.cmd_valid=1;
    dut.sys_clk=0; dut.eval(); require(dut.cmd_ready, "command blocked");
    dut.sys_clk=1; dut.eval(); dut.cmd_valid=0;
    dut.cmd_region=99; dut.cmd_offset=~uint64_t(0); dut.cmd_length=0;
    std::array<std::deque<AW>,2> aw;
    std::array<std::deque<W>,2> w;
    unsigned issued[2]={0,0}, received[2]={0,0}, in_burst[2]={0,0}, responses[2]={0,0};
    unsigned sent=0, state=97;
    bool seen_w[2]={false,false}, aw_held[2]={false,false}, w_held[2]={false,false};
    std::array<uint64_t,6> saved_aw[2];
    std::array<uint32_t,10> saved_w[2];
    auto random=[&]() { state=1664525u*state+1013904223u; return state >> 16; };
    for (unsigned cycle=0; cycle<30000; cycle++) {
        dut.data_valid=sent<count;
        dut.data_last=(sent==count-1) != (fault==2);
        for (unsigned j=0; j<8; j++) dut.data_data[j]=0x5b0b0000u+sent*8+j;
        dut.cancel=fault==5 && cycle>=60;
        for (unsigned i=0; i<2; i++) {
            auto &p=ports[i]; seen_w[i]=seen_w[i] || *p.wv;
            *p.ar=seen_w[i] && random()%4!=0; *p.wr=random()%4!=0;
            *p.bv=!aw[i].empty() && !w[i].empty() && cycle>=300;
            *p.bid=fault==4 ? 3 : 0; *p.bresp=fault==3 ? 2 : 0;
        }
        dut.sys_clk=0; dut.eval();
        if (dut.data_valid && dut.data_ready) sent++;
        for (unsigned i=0; i<2; i++) {
            auto &p=ports[i];
            std::array<uint64_t,6> addr={*p.addr,*p.alen,*p.aid,*p.asz,*p.aburst,*p.av};
            if (aw_held[i]) require(addr==saved_aw[i], "unstable stalled AW");
            aw_held[i]=*p.av && !*p.ar; saved_aw[i]=addr;
            std::array<uint32_t,10> data;
            for (unsigned j=0; j<8; j++) data[j]=(*p.data)[j];
            data[8]=*p.wl; data[9]=*p.strb;
            if (w_held[i]) require(*p.wv && data==saved_w[i], "unstable stalled W");
            w_held[i]=*p.wv && !*p.wr; saved_w[i]=data;
            if (*p.av && *p.ar) {
                require(i==selected && fault!=1, "invalid destination issued AW");
                const unsigned beats=*p.alen+1;
                require(beats<=16 && *p.addr%32==0 && (*p.addr%4096)+beats*32<=4096, "invalid write burst");
                require(*p.aid==0 && *p.asz==5 && *p.aburst==1, "AXI attributes");
                require(*p.addr==physical+32*issued[i] && issued[i]+beats<=count, "physical write mismatch");
                aw[i].push_back({*p.addr,beats}); issued[i]+=beats;
                require(aw[i].size()<=8, "write credits exceeded");
            }
            if (*p.wv && *p.wr) {
                require(i==selected && fault!=1 && fault!=2, "invalid payload issued W");
                require(received[i]<count && *p.strb==0xffffffffu, "unexpected write strobe/beat");
                for (unsigned j=0; j<8; j++) require(data[j]==0x5b0b0000u+received[i]*8+j, "payload mismatch");
                received[i]++; in_burst[i]++;
                if (*p.wl) { w[i].push_back({in_burst[i], received[i]-in_burst[i]}); in_burst[i]=0; }
            }
            if (*p.bv && *p.br) {
                require(aw[i].front().beats==w[i].front().beats, "wrong WLAST");
                require(aw[i].front().addr==physical+32*w[i].front().first, "AW/W order mismatch");
                aw[i].pop_front(); w[i].pop_front(); responses[i]++;
            }
        }
        const bool done=dut.done;
        dut.sys_clk=1; dut.eval();
        if (fault) {
            require(!done && !dut.done, "failed upload reported success");
            if (dut.failed) {
                require(dut.error && dut.busy && !dut.cmd_ready, "unlatched upload failure");
                std::printf("failed safely: %u\n", unsigned(dut.error));
                return 0;
            }
        } else {
            require(!dut.error && !dut.failed, "valid upload failed");
            if (done || dut.done) {
                require(sent==count && received[selected]==count && issued[selected]==count, "short upload");
                require(aw[selected].empty() && w[selected].empty() && !in_burst[selected] && responses[selected], "early done before B drain");
                std::printf("%u physical beats exactly written\n", count);
                return 0;
            }
        }
    }
    require(false, "upload timed out");
}

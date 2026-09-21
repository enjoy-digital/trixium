// SPDX-License-Identifier: BSD-2-Clause
#include "Vregion_fetch.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"%s\n",why); std::exit(1); }
}
struct Port {
    CData *av,*ar,*alen,*aid,*asz,*aburst; QData *addr;
    CData *rv,*rr,*rl,*rid,*resp; VlWide<8> *data;
};
struct Burst { uint64_t addr; unsigned left,ready_at; };
static uint32_t payload(uint64_t addr, unsigned lane) {
    return uint32_t(addr) ^ uint32_t(addr>>32)*0xa6398b17u ^ (lane*7919+0x5b0b0000u);
}
class Rig {
public:
    Vregion_fetch d;
    std::array<Port,2> ports{{
        {&d.ar0_valid,&d.ar0_ready,&d.ar0_len,&d.ar0_id,&d.ar0_size,&d.ar0_burst,&d.ar0_addr,
         &d.r0_valid,&d.r0_ready,&d.r0_last,&d.r0_id,&d.r0_resp,&d.r0_data},
        {&d.ar1_valid,&d.ar1_ready,&d.ar1_len,&d.ar1_id,&d.ar1_size,&d.ar1_burst,&d.ar1_addr,
         &d.r1_valid,&d.r1_ready,&d.r1_last,&d.r1_id,&d.r1_resp,&d.r1_data}}};
    std::array<std::array<std::deque<Burst>,4>,2> pending;
    unsigned held_id[2]={0,0},held_rid[2]={0,0},held_resp[2]={0,0},held_last[2]={0,0};
    bool held[2]={false,false};
    unsigned cycle=0,rng=139,fault=0,selected=0,count=0,dst=0,issued=0,delivered=0,writes=0;
    uint64_t physical=0;
    bool freeze=false,monitor=false;
    unsigned random() { rng=1664525u*rng+1013904223u; return rng>>16; }
    void low() {
        for (unsigned i=0;i<2;i++) {
            auto &p=ports[i]; *p.ar=random()%4!=0;
            if (!held[i] && !freeze) {
                unsigned first=random()%4;
                for (unsigned n=0;n<4;n++) {
                    unsigned id=(first+n)%4;
                    if (!pending[i][id].empty() && pending[i][id].front().ready_at<=cycle && random()%4!=0) {
                        held_id[i]=id; held[i]=true;
                        held_rid[i]=fault==3 ? id+4 : id;
                        held_resp[i]=fault==2 ? 2 : 0;
                        held_last[i]=fault==4 ? 1 : fault==5 ? 0 : pending[i][id].front().left==1;
                        break;
                    }
                }
            }
            *p.rv=held[i] && !freeze;
            if (held[i]) {
                const auto &b=pending[i][held_id[i]].front();
                for (unsigned j=0;j<8;j++) (*p.data)[j]=payload(b.addr,j);
                *p.rid=held_rid[i]; *p.resp=held_resp[i]; *p.rl=held_last[i];
            }
        }
        d.sys_clk=0; d.eval();
    }
    void high() {
        if (!d.sys_rst) for (unsigned i=0;i<2;i++) {
            auto &p=ports[i];
            if (*p.av && *p.ar) {
                require(monitor && fault!=1 && i==selected,"unexpected external read");
                unsigned beats=*p.alen+1;
                require(beats<=16 && *p.addr%32==0 && (*p.addr%4096)+beats*32<=4096,"invalid burst");
                require(*p.aid<4 && *p.asz==5 && *p.aburst==1,"invalid AXI attributes");
                require(*p.addr==physical+32*issued && issued+beats<=count,"wrong physical region span");
                pending[i][*p.aid].push_back({*p.addr,beats,cycle+10+random()%20}); issued+=beats;
            }
            if (*p.rv && *p.rr) {
                auto &q=pending[i][held_id[i]]; auto &b=q.front();
                delivered++; b.addr+=32; if (!--b.left) q.pop_front(); held[i]=false;
            }
        }
        if (monitor && d.gather_we) {
            require(fault!=1 && d.gather_adr*32>=dst && uint64_t(d.gather_adr)*32<uint64_t(dst)+32*count,
                    "read wrote outside its admitted scratchpad span"); writes++;
        }
        d.sys_clk=1; d.eval(); cycle++;
    }
    void tick() { low(); high(); }
    uint32_t access(unsigned address, bool write, uint32_t value=0) {
        d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
        for (unsigned n=0;n<2000;n++) {
            low(); bool ack=d.host_ack; uint32_t result=d.host_dat_r; high();
            if (ack) { d.host_cyc=0; d.host_stb=0; tick(); return result; }
        }
        require(false,"Wishbone timeout"); return 0;
    }
    void configure(unsigned rid, uint64_t offset, unsigned length, unsigned destination,
                   uint64_t address, unsigned port, unsigned fault_mode) {
        fault=fault_mode; selected=port; count=length; dst=destination; physical=address;
        issued=delivered=writes=0; monitor=true;
        std::array<uint32_t,32> desc{};
        desc[0]=10; desc[1]=rid; desc[2]=offset; desc[3]=offset>>32; desc[4]=length; desc[5]=dst;
        for (unsigned i=0;i<desc.size();i++) access(i*4,true,desc[i]);
        for (unsigned guard:{0x7fc,0x10000,0x3fffc})
            if (guard<dst || uint64_t(guard)>=uint64_t(dst)+uint64_t(count)*32) access(guard,true,0xdeadbeef);
        require(access(0,false)==10,"posted write barrier");
    }
    void start() {
        d.seq_base=0; d.seq_count=2; d.seq_start=1; d.seq_start_re=1;
        tick(); d.seq_start_re=0; require(d.seq_status&1,"sequence not started");
    }
    void reset(bool global=false) {
        if (global) {
            // Malformed RID/RLAST cannot promise bus resynchronization via a
            // client-only reset. Reset both DUT and modeled external transport.
            d.sys_rst=1; d.host_cyc=0; d.host_stb=0; freeze=false;
            for (auto &p:pending) for (auto &q:p) q.clear();
            held[0]=held[1]=false; monitor=false; tick(); tick(); d.sys_rst=0;
        } else {
            d.reset_value=1; d.reset_re=1; tick(); d.reset_re=0;
            if (freeze) {
                for (unsigned n=0;n<15;n++) tick();
                require(d.reset_busy && !(d.seq_status&1),"reset did not wait for outstanding traffic");
            }
            freeze=false; fault=0; // Drain valid responses through the actual reset shims.
        }
        for (unsigned n=0;n<3000;n++) {
            tick(); if (n>=30 && !d.reset_busy) break;
        }
        require(!d.reset_busy && !d.reset_stuck && !(d.seq_status&1) && !d.seq_error,"reset/drain failed");
        for (auto &p:pending) for (auto &q:p) require(q.empty(),"reset left external traffic");
        require(!held[0] && !held[1],"reset left held response");
    }
    void finish(bool inject_manual=false, unsigned manual_before=0) {
        bool injected=false;
        for (unsigned n=0;n<50000;n++) {
            if (inject_manual && !injected && issued) {
                d.manual_region=7; d.manual_offset=128; d.manual_length=1; d.manual_dst=0x2000;
                d.manual_start=1; d.manual_start_re=1; injected=true;
            }
            tick(); d.manual_start_re=0;
            if (d.seq_error) std::fprintf(stderr,"sequence error=%x transport=%x\n",d.seq_error,d.read_error);
            require(!d.seq_error,"valid sequence failed");
            if (!(d.seq_status&1)) break;
        }
        require(!(d.seq_status&1) && (d.seq_status>>8)==1,"sequence did not retire read before stop");
        require(issued==count && delivered==count && writes==count,"short/extra region transfer");
        if (inject_manual) require(injected && d.manual_error==256,"conflicting manual read not rejected");
        require(d.manual_accepted==manual_before && d.manual_completed==manual_before,"sequence credited manual read counters");
        for (unsigned b=0;b<count;b++) for (unsigned lane=0;lane<8;lane++)
            require(access(dst+b*32+lane*4,false)==payload(physical+b*32,lane),"scratchpad payload mismatch");
        for (unsigned guard:{0x7fc,0x10000,0x3fffc})
            if (guard<dst || uint64_t(guard)>=uint64_t(dst)+uint64_t(count)*32)
                require(access(guard,false)==0xdeadbeef,"scratchpad guard changed");
        for (unsigned i=16;i<32;i++) require(access(i*4,false)==0,"later descriptor overwritten");
    }
};
int main(int argc,char **argv) {
    require(argc==8,"region offset length dst physical port fault required");
    unsigned rid=std::strtoul(argv[1],nullptr,0),count=std::strtoul(argv[3],nullptr,0);
    unsigned dst=std::strtoul(argv[4],nullptr,0),selected=std::strtoul(argv[6],nullptr,0),fault=std::strtoul(argv[7],nullptr,0);
    uint64_t offset=std::strtoull(argv[2],nullptr,0),physical=std::strtoull(argv[5],nullptr,0);
    Rig r; r.d.sys_rst=1; r.tick(); r.tick(); r.d.sys_rst=0;
    r.configure(rid,offset,count,dst,physical,selected,fault);
    r.freeze=fault==6 || fault==7;
    if (fault==10) {
        r.freeze=true;
        r.d.manual_region=rid; r.d.manual_offset=offset; r.d.manual_length=count; r.d.manual_dst=dst;
        r.d.manual_start=1; r.d.manual_start_re=1; r.tick(); r.d.manual_start_re=0;
        for (unsigned n=0;n<100 && !r.issued;n++) r.tick();
        require(r.issued && r.d.manual_accepted==1,"older manual read not issued");
    }
    if (fault==11) {
        r.d.manual_region=7; r.d.manual_offset=128; r.d.manual_length=1; r.d.manual_dst=0x2000;
        r.d.manual_start=1; r.d.manual_start_re=1;
    }
    r.start();
    r.d.manual_start_re=0;
    if (fault==10) {
        for (unsigned n=0;n<30;n++) {
            r.tick(); require(r.d.seq_draining && !(r.d.seq_status>>8) && !r.d.manual_completed,
                              "list did not wait for older manual read");
        }
        r.freeze=false;
        for (unsigned n=0;n<10000 && !r.d.manual_completed;n++) r.tick();
        require(r.d.manual_completed==1 && r.delivered==count && r.issued==count,"older read not drained exactly");
        r.issued=r.delivered=r.writes=0;
        r.finish(false,1);
    } else if (fault==0 || fault==9 || fault==11) {
        r.finish(fault==9);
        if (fault==11) require(r.d.manual_error==256,"same-cycle manual/list start was not rejected");
    }
    else {
        bool reached=false;
        for (unsigned n=0;n<10000;n++) {
            r.tick();
            require((r.d.seq_status&1) && !(r.d.seq_status>>8),"failed/cancelled read was published");
            if (fault==1 && r.d.seq_error) {
                require(!r.issued && !r.writes,"invalid descriptor accessed data"); reached=true; break;
            }
            if (fault>=2 && fault<=5 && r.d.seq_error) { reached=true; break; }
            if (((fault==6 && n>=2000) || (fault==7 && n>=8)) && r.issued) {
                require(!r.d.seq_error && !r.delivered,"missing response fabricated progress"); reached=true; break;
            }
            if (fault==8 && r.writes>=5) { reached=true; break; }
        }
        require(reached,"fault/cancel point not reached");
        r.reset(fault>=3 && fault<=5);
        r.configure(42,128,3,0x800,(uint64_t(1)<<32)+128,0,0);
        r.start(); r.finish();
    }
    std::printf("REGION_FETCH_PASS fault=%u issued=%u delivered=%u cycles=%u\n",fault,r.issued,r.delivered,r.cycle);
}

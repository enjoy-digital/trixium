// SPDX-License-Identifier: BSD-2-Clause
#include "Vembedding_sp.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static void tick(Vembedding_sp &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t word(std::ifstream &f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4); require(bool(f),"short vector file");
    return b[0] | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
static uint32_t access(Vembedding_sp &d, uint32_t address, bool write, uint32_t value=0) {
    d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
    for (unsigned i=0; i<2000; i++) {
        d.sys_clk=0; d.eval(); bool ack=d.host_ack; uint32_t result=d.host_dat_r;
        d.sys_clk=1; d.eval();
        if (ack) { d.host_cyc=0; d.host_stb=0; tick(d); return result; }
    }
    require(false,"Wishbone timeout"); return 0;
}
struct Case {
    uint32_t src, signs, dst, groups, pstart, pend, error, exponent, reads, outputs;
    std::map<uint32_t,uint32_t> writes, checks;
};
static void install(Vembedding_sp &d, const Case &c) {
    for (auto [a,v]:c.writes) access(d,a,true,v);
    // A sideband read is the posted-write barrier; no host access occurs from
    // command admission until completion or cancellation.
    auto first=c.writes.begin(); require(access(d,first->first,false)==first->second,"setup barrier");
}
static void start(Vembedding_sp &d, const Case &c) {
    d.cmd_src=c.src; d.cmd_signs=c.signs; d.cmd_dst=c.dst; d.cmd_groups=c.groups;
    d.cmd_protect_start=c.pstart; d.cmd_protect_end=c.pend; d.cmd_valid=1;
    for (unsigned i=0; i<100; i++) {
        d.sys_clk=0; d.eval(); bool ready=d.cmd_ready; d.sys_clk=1; d.eval();
        if (ready) { d.cmd_valid=0; require(d.busy,"command not captured"); return; }
    }
    require(false,"command timeout");
}
static void clear(Vembedding_sp &d, bool reset=false) {
    d.cmd_valid=0; d.cancel=!reset; d.sys_rst=reset;
    d.sys_clk=0; d.eval();
    if (!reset) require(!d.rd_req && !d.wr_req && !d.cmd_ready,"cancel did not mask transactions");
    tick(d); d.cancel=0; d.sys_rst=0;
    for (unsigned i=0; i<20; i++) {
        tick(d); require(!d.busy && !d.done && !d.error && !d.rd_req && !d.wr_req,"stale work after reset/cancel");
    }
    require(d.read_beats==0 && d.output_beats==0,"cancel counters not cleared");
}
static void check(Vembedding_sp &d, const std::map<uint32_t,uint32_t> &expected, unsigned id) {
    for (auto [a,v]:expected) {
        uint32_t got=access(d,a,false);
        if (got!=v) std::fprintf(stderr,"case=%u addr=%x got=%08x expected=%08x\n",id,a,got,v);
        require(got==v,"scratchpad memory mismatch");
    }
}
// abort: 1 raw return, 2 signs return, 3 transform, 4 held output after
// a committed prefix, 5/6 global reset during raw return / held output.
static unsigned run(Vembedding_sp &d, const Case &c, unsigned id, unsigned abort=0, unsigned seed=0) {
    install(d,c); start(d,c);
    // An accepted command owns a snapshot, never live host endpoint fields.
    d.cmd_src=0xffffff; d.cmd_signs=0xffffff; d.cmd_dst=0xffffff; d.cmd_groups=0;
    d.cmd_protect_start=0xffffff; d.cmd_protect_end=0;
    unsigned reads=0, outputs=0, rng=113+seed, held_addr=0;
    bool held=false, aborted=false, previous_raw=false, previous_sign=false;
    std::array<uint32_t,8> held_data{};
    std::map<uint32_t,uint32_t> prefix=c.writes;
    for (unsigned cycle=0; cycle<50000; cycle++) {
        rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
        d.write_allow=(abort==4 || abort==6) ? outputs<17 : (!seed || rng%4!=0);
        d.sys_clk=0; d.eval();
        if ((abort==1 || abort==5) && previous_raw) aborted=true;
        if (abort==2 && previous_sign) aborted=true;
        if (abort==3 && reads==c.reads && !d.wr_req && cycle>500) aborted=true;
        if ((abort==4 || abort==6) && d.wr_req && outputs==17 && held) aborted=true;
        if (aborted) {
            clear(d,abort>=5); check(d,prefix,id);
            require((abort!=4 && abort!=6) || outputs==17,"abort missed committed prefix");
            return cycle;
        }
        if (d.error) {
            if (d.error!=c.error) std::fprintf(stderr,"case=%u cycle=%u error=%x expected=%x reads=%u outputs=%u\n",
                                              id,cycle,d.error,c.error,reads,outputs);
            require(d.error==c.error && c.error,"unexpected client failure");
            require(d.busy && !d.done && !d.rd_req && !d.wr_req && !outputs,"invalid output publication");
            if (c.error&7) require(!reads,"invalid command touched memory");
            for (unsigned i=0; i<5; i++) { tick(d); require(d.error==c.error && d.busy,"failure not sticky"); }
            clear(d); check(d,c.writes,id); return cycle;
        }
        if (d.done) {
            require(!c.error && !abort && !d.busy,"bad completion");
            require(reads==c.reads && outputs==c.outputs,"transaction count mismatch");
            require(d.read_beats==reads && d.output_beats==outputs,"hardware counter mismatch");
            require(d.exponent==(c.exponent&1023),"row exponent mismatch");
            tick(d); require(!d.done && d.cmd_ready,"completion not a pulse");
            check(d,c.checks,id); return cycle;
        }
        if (held) {
            require(d.wr_req && d.wr_adr==held_addr,"stalled output address changed");
            for (unsigned lane=0; lane<8; lane++) require(d.wr_data[lane]==held_data[lane],"stalled output changed");
        }
        held=d.wr_req && !d.write_allow;
        if (held) { held_addr=d.wr_adr; for (unsigned lane=0; lane<8; lane++) held_data[lane]=d.wr_data[lane]; }
        previous_raw=false; previous_sign=false;
        if (d.rd_req) {
            require(!outputs,"input read after output overwrite");
            unsigned address=d.rd_adr*32;
            previous_raw=address>=c.src && address<c.src+c.groups*28;
            previous_sign=address>=c.signs && address<c.signs+c.groups*16;
            require(previous_raw || previous_sign,"unexpected read address"); reads++;
        }
        if (d.wr_req && d.write_allow) {
            require(reads==c.reads,"output before all input reads");
            require(d.wr_we==255 && d.wr_adr*32==c.dst+outputs*32,"output address/enable mismatch");
            for (unsigned lane=0; lane<8; lane++) {
                unsigned a=d.wr_adr*32+lane*4;
                if (d.wr_data[lane]!=c.checks.at(a))
                    std::fprintf(stderr,"case=%u cycle=%u output=%u lane=%u got=%08x expected=%08x\n",
                                 id,cycle,outputs,lane,d.wr_data[lane],c.checks.at(a));
                require(d.wr_data[lane]==c.checks.at(a),"output data mismatch");
                prefix[a]=d.wr_data[lane];
            }
            outputs++;
        }
        d.sys_clk=1; d.eval();
    }
    std::fprintf(stderr,"case=%u abort=%u reads=%u/%u outputs=%u/%u error=%x\n",
                 id,abort,reads,c.reads,outputs,c.outputs,d.error);
    require(false,"embedding scratchpad timed out"); return 0;
}
int main() {
    std::ifstream f("vectors.bin",std::ios::binary);
    require(word(f)==0x45535031,"bad vector magic"); unsigned count=word(f), rejected=0;
    Vembedding_sp d; d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    for (unsigned id=0; id<count; id++) {
        Case c;
        c.src=word(f); c.signs=word(f); c.dst=word(f); c.groups=word(f); c.pstart=word(f); c.pend=word(f);
        c.error=word(f); c.exponent=word(f); c.reads=word(f); c.outputs=word(f);
        unsigned nw=word(f), nc=word(f);
        for (unsigned i=0; i<nw; i++) { unsigned a=word(f), v=word(f); c.writes[a]=v; }
        for (unsigned i=0; i<nc; i++) { unsigned a=word(f), v=word(f); c.checks[a]=v; }
        if (id==0) {
            for (unsigned abort=1; abort<=6; abort++) {
                run(d,c,id,abort); run(d,c,id,0,abort);
            }
            unsigned cycles=run(d,c,id);
            std::printf("EMBEDDING_SP dense_groups=%u reads=%u outputs=%u cycles=%u\n",c.groups,c.reads,c.outputs,cycles);
        } else run(d,c,id,0,id%3);
        rejected+=c.error!=0;
    }
    require(f.peek()==std::char_traits<char>::eof(),"trailing vector bytes");
    std::printf("EMBEDDING_SP_PASS cases=%u rejects=%u cancel_reuse=4 reset_reuse=2\n",count,rejected);
}

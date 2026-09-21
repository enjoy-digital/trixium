// SPDX-License-Identifier: BSD-2-Clause
#include "Vembedding_native.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"%s\n",why); std::exit(1); }
}
static uint32_t rng=914;
static uint32_t random_word() { rng=1664525u*rng+1013904223u; return rng>>16; }
static void tick(Vembedding_native &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t word(std::ifstream &f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4); require(bool(f),"short vector");
    return b[0] | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
struct Case {
    unsigned groups, beats, exponent, error, frame, sign_groups;
    std::vector<std::array<uint32_t,4>> signs;
    std::vector<std::array<uint32_t,8>> data;
    std::vector<uint32_t> expected;
};
static void cancel(Vembedding_native &d) {
    d.cancel=1; d.eval(); require(!d.source_valid && !d.sink_ready && !d.cmd_ready,"cancel did not mask handshakes");
    tick(d); d.cancel=0; d.sink_valid=0; d.cmd_valid=0; tick(d); tick(d);
    // NativeWeightStream deliberately retains its diagnostic error until the
    // next start/reset. Cancellation must flush validity/ownership and the
    // embedding owner's errors, not rewrite that existing upstream contract.
    require(!d.busy && !(d.error&63) && !d.done && !d.source_valid && d.cancelled,"cancel did not flush");
}
static void run(Vembedding_native &d, const Case &c, unsigned index, unsigned abort_cycle=0,
                bool reset_abort=false, bool stalls=true) {
    d.source_ready=0; d.sink_valid=0;
    for (unsigned g=0; g<c.sign_groups; g++) {
        d.signs_address=g; d.signs_we=1;
        for (unsigned j=0; j<4; j++) d.signs_data[j]=c.signs[g][j];
        tick(d);
    }
    d.signs_we=0; d.cmd_groups=c.groups; d.cmd_valid=1;
    d.sys_clk=0; d.eval(); require(d.cmd_ready,"command blocked");
    d.sys_clk=1; d.eval(); d.cmd_valid=0; d.cmd_groups=0;
    unsigned sent=0, received=0, first=0;
    bool offering=false, held=false;
    std::array<uint32_t,11> previous{};
    for (unsigned cycle=0; cycle<50000; cycle++) {
        if (abort_cycle && cycle==abort_cycle) {
            require(d.busy && !d.done,"cancellation did not interrupt a live row");
            if (abort_cycle==9000) require(received==17*8,"late cancellation missed consumed prefix");
            if (reset_abort) {
                d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
                d.sink_valid=0; d.cmd_valid=0; tick(d);
                require(!d.busy && !d.error && !d.done && !d.source_valid,"global reset did not flush");
            } else cancel(d);
            return;
        }
        offering=offering || !stalls || random_word()%4 != 0;
        d.sink_valid=sent<c.beats && offering;
        d.sink_last=(sent+1==c.beats && c.frame!=2) || (c.frame==1 && sent==0);
        if (sent<c.beats) for (unsigned j=0; j<8; j++) d.sink_data[j]=c.data[sent][j];
        d.source_ready=!stalls || ((cycle%97<40 || cycle%97>=73) && random_word()%3 != 0);
        if (abort_cycle && received>=17*8) d.source_ready=0; // Retain an invalidated prefix plus held word.
        d.sys_clk=0; d.eval();
        std::array<uint32_t,11> now{};
        for (unsigned j=0; j<8; j++) now[j]=d.source_data[j];
        now[8]=d.source_exponent; now[9]=d.source_first; now[10]=d.source_last;
        if (held) require(d.source_valid && now==previous,"stalled embedding output changed");
        held=d.source_valid && !d.source_ready; previous=now;
        if (d.sink_valid && d.sink_ready) { sent++; offering=false; }
        if (d.source_valid) require(!c.error,"invalid row produced output");
        if (d.source_valid && d.source_ready) {
            require(received+8<=c.expected.size(),"extra output");
            require(d.source_first==(received==0) && d.source_last==(received+8==c.expected.size()),"output framing");
            if (d.source_exponent!=(c.exponent & 1023))
                std::fprintf(stderr,"case=%u abort=%u cycle=%u exponent=%u expected=%u\n",index,abort_cycle,cycle,d.source_exponent,c.exponent & 1023);
            require(d.source_exponent==(c.exponent & 1023),"wrong whole-row BFP exponent");
            for (unsigned j=0; j<8; j++) {
                if (d.source_data[j]!=c.expected[received+j])
                    std::fprintf(stderr,"case=%u element=%u got=%08x expected=%08x\n",index,received+j,d.source_data[j],c.expected[received+j]);
                require(d.source_data[j]==c.expected[received+j],"restored embedding mismatch");
            }
            if (!received) first=cycle;
            received+=8;
        }
        d.sys_clk=1; d.eval();
        if (d.error) {
            if (d.error!=c.error) std::fprintf(stderr,"case=%u error=%u expected=%u cycle=%u\n",index,d.error,c.error,cycle);
            require(c.error && d.error==c.error && !received && !d.done && d.busy,"wrong failure contract");
            d.sink_valid=0;
            for (unsigned i=0; i<20; i++) { tick(d); require(d.busy && !d.done && !d.source_valid,"failure was published"); }
            cancel(d); return;
        }
        if (d.done) {
            require(!abort_cycle && !c.error && sent==c.beats && received==c.expected.size() && !d.busy,"premature completion");
            d.sink_valid=0; tick(d); require(!d.done && !d.source_valid,"stale output/completion");
            if (index<3) std::printf("EMBEDDING_ROW case=%u groups=%u stalls=%u first=%u cycles=%u\n",index,c.groups,stalls,first,cycle+1);
            return;
        }
    }
    require(false,"embedding timeout");
}
int main() {
    std::ifstream f("vectors.bin",std::ios::binary);
    require(word(f)==0x454d4231,"bad vector magic"); unsigned count=word(f), width=word(f);
    Vembedding_native d; d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    unsigned failures=0;
    for (unsigned index=0; index<count; index++) {
        Case c; c.groups=word(f); c.beats=word(f); c.exponent=word(f); c.error=word(f); c.frame=word(f); c.sign_groups=word(f);
        require(c.sign_groups<=64 && c.beats<1000,"bad vector geometry");
        c.signs.resize(c.sign_groups); for (auto &g:c.signs) for (auto &v:g) v=word(f);
        c.data.resize(c.beats); for (auto &b:c.data) for (auto &v:b) v=word(f);
        if (!c.error) { c.expected.resize(c.groups*128); for (auto &v:c.expected) v=word(f); }
        if (!index) {
            for (unsigned cycle:{1,4,80,300,950,2400,4600,9000}) run(d,c,index,cycle);
            for (unsigned cycle:{300,9000}) run(d,c,index,cycle,true);
            run(d,c,index,0,false,false);
        }
        run(d,c,index);
        if (c.error) failures++;
    }
    require(f.peek()==std::char_traits<char>::eof(),"trailing vectors");
    std::printf("EMBEDDING_NATIVE_PASS width=%u cases=%u failures=%u cancel_reuse=8 reset_reuse=2\n",width,count,failures);
}

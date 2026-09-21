// SPDX-License-Identifier: BSD-2-Clause
#include "Vbf16_resident.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <utility>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static void tick(Vbf16_resident &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t word(std::ifstream &f) {
    uint8_t bytes[4]; f.read(reinterpret_cast<char*>(bytes),4); require(bool(f),"short vector file");
    return bytes[0] | uint32_t(bytes[1])<<8 | uint32_t(bytes[2])<<16 | uint32_t(bytes[3])<<24;
}
static uint32_t access(Vbf16_resident &d, uint32_t address, bool write, uint32_t value=0) {
    d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
    for (unsigned cycle=0; cycle<2000; cycle++) {
        d.sys_clk=0; d.eval(); bool ack=d.host_ack; uint32_t result=d.host_dat_r;
        d.sys_clk=1; d.eval();
        if (ack) { d.host_cyc=0; d.host_stb=0; tick(d); return result; }
    }
    require(false,"Wishbone timeout"); return 0;
}
static void reset(Vbf16_resident &d) {
    d.reset_value=1; d.reset_re=1; tick(d); d.reset_re=0;
    for (unsigned i=0; i<100; i++) tick(d);
    require(!d.reset_busy && !d.reset_stuck && !(d.seq_status&1) && !d.seq_error
        && !d.native_error && !d.restore_error,"reset failed");
}
static void start(Vbf16_resident &d, unsigned descriptors) {
    d.seq_base=0; d.seq_count=descriptors; d.seq_start=1; d.seq_start_re=1;
    tick(d); d.seq_start_re=0; require(d.seq_status&1,"sequence not started");
}
int main() {
    std::ifstream f("vectors.bin",std::ios::binary);
    require(word(f)==0x42465231,"bad vector magic"); unsigned profiles=word(f);
    Vbf16_resident d; d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    for (unsigned profile=0; profile<profiles; profile++) {
        unsigned nw=word(f), nc=word(f), nd=word(f);
        std::vector<std::pair<uint32_t,uint32_t>> writes, checks;
        for (unsigned i=0; i<nw; i++) { uint32_t a=word(f), v=word(f); writes.emplace_back(a,v); }
        for (unsigned i=0; i<nc; i++) { uint32_t a=word(f), v=word(f); checks.emplace_back(a,v); }
        auto install=[&]() { for (auto [a,v] : writes) access(d,a,true,v); };
        if (profile<2) {
            install(); start(d,nd);
            bool interrupted=false;
            for (unsigned cycle=0; cycle<200000; cycle++) {
                require(!d.seq_error && !d.native_error && !d.restore_error,"abort sequence failed early");
                if ((profile==0 && d.engine_busy && (d.seq_status>>8)==3) ||
                    (profile==1 && d.restore_busy)) { interrupted=true; break; }
                tick(d);
            }
            require(interrupted && (d.seq_status&1),"reset did not interrupt requested resident stage");
            reset(d);
        }
        install(); unsigned before=d.engine_done; start(d,nd);
        unsigned cycles=0;
        // No host memory access, conversion or command submission between start
        // and completion. Both projections reuse the first resident quantization.
        for (; cycles<200000; cycles++) {
            if (d.seq_error || d.native_error || d.restore_error)
                std::fprintf(stderr,"errors q=%x native=%x restore=%x descriptors=%u\n",
                             d.seq_error,d.native_error,d.restore_error,d.seq_status>>8);
            require(!d.seq_error && !d.native_error && !d.restore_error,"resident sequence failed");
            if (!(d.seq_status&1)) break;
            tick(d);
        }
        require(cycles<200000 && (d.seq_status>>8)==nd,"sequence completion mismatch");
        require(d.engine_done-before==2,"wrong projection retirement count");
        require(d.quant_blocks==1 && d.quant_reads==1280 && d.quant_writes==160,"activation was not quantized once as a whole");
        for (auto [a,v] : checks) {
            uint32_t got=access(d,a,false);
            if (got!=v) std::fprintf(stderr,"profile=%u address=%x got=%x expected=%x\n",profile,a,got,v);
            require(got==v,"resident scratchpad mismatch");
        }
        std::printf("BF16_PAIR profile=%u rows=48 columns=5120 descriptors=%u cycles=%u\n",profile,nd,cycles);
    }
    require(f.peek()==std::char_traits<char>::eof(),"trailing vector bytes");
    std::puts("BF16_RESIDENT_PASS profiles=3 quantizations_per_pair=1 reset_reuse=2");
}

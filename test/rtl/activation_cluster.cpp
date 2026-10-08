// SPDX-License-Identifier: BSD-2-Clause
// Full shared scratchpad: FP32 -> block Q8 -> Hadamard -> whole-vector Q8,
// -> restored FP32.
#include "Vactivation_cluster.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t rng=537;
static uint32_t random_word() { rng=rng*1664525u+1013904223u; return rng; }
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static void tick(Vactivation_cluster &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t access(Vactivation_cluster &d, uint32_t address, bool write, uint32_t value=0) {
    d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
    for (unsigned cycle=0; cycle<2000; cycle++) {
        d.sys_clk=0; d.eval(); bool ack=d.host_ack; uint32_t result=d.host_dat_r;
        d.sys_clk=1; d.eval();
        if (ack) { d.host_cyc=0; d.host_stb=0; tick(d); return result; }
    }
    require(false, "Wishbone timeout"); return 0;
}
static std::array<uint32_t, 16> quant(unsigned elements, unsigned blocks) {
    return {4, 0x10000, 0x4000, 0x800, elements, blocks, 0,0,0,0,0,0,0,0,0,0};
}
static std::array<uint32_t, 16> requant(unsigned blocks) {
    return {5, 0x22000, 0xa000, 0x800, 0x900, 0xa00, blocks, 0,0,0,0,0,0,0,0,0};
}
static void start(Vactivation_cluster &d, const std::array<uint32_t, 16> &words, bool rotate=false) {
    unsigned count=rotate ? 3 : 1;
    for (unsigned i=0; i<16; i++) access(d, i*4, true, words[i]);
    if (rotate) {
        const std::array<uint32_t, 16> had={3,0x4000,0x1000,0x22000,words[4]*words[5]/1024,0x900,0,0,0,0,0,0,0,0,0,0};
        for (unsigned i=0; i<16; i++) access(d, 64+i*4, true, had[i]);
        auto post=requant(words[4]*words[5]/1024);
        for (unsigned i=0; i<16; i++) access(d, 128+i*4, true, post[i]);
    }
    d.seq_base=0; d.seq_count=count; d.seq_start=1; d.seq_start_re=1; tick(d); d.seq_start_re=0;
    require(d.seq_status & 1, "list not started");
}
static void reset_cluster(Vactivation_cluster &d) {
    d.reset_value=1; d.reset_re=1; tick(d); d.reset_re=0;
    for (unsigned i=0; i<100; i++) tick(d);
    require(!d.reset_busy && !d.reset_stuck && !(d.seq_status & 1) && !d.seq_error && !d.had_error,
            "cluster reset did not recover");
}
static void complete(Vactivation_cluster &d, unsigned elements, unsigned blocks, bool rotate=false, bool wide_scales=true) {
    const unsigned length=elements*blocks;
    std::vector<float> values(length);
    std::vector<int> expected(length), exponents(blocks), signs(length);
    for (unsigned i=0; i<length; i++) {
        // Whole-vector mode also spans very different magnitude regions. Its
        // largest region arrives last and must set the scale for every element.
        const unsigned scale_group=i/(elements>=1024 ? 1024 : elements);
        int physical_exponent=wide_scales ? int(scale_group)*5-40 : int(scale_group%3)-1;
        values[i]=std::ldexp((int(random_word() >> 16)-32768)/256.0f, physical_exponent);
        signs[i]=(random_word() & 0x10000) ? 1 : -1;
        access(d, 0x10000+i*4, true, bits(values[i]));
    }
    for (unsigned b=0; b<blocks; b++) {
        float maximum=0;
        for (unsigned i=0; i<elements; i++) maximum=std::max(maximum,std::fabs(values[b*elements+i]));
        int e=0; if (maximum) { std::frexp(maximum,&e); e-=7; }
        exponents[b]=e;
        const float scale=std::ldexp(1.0f,-e);
        for (unsigned i=0; i<elements; i++) {
            volatile float scaled=values[b*elements+i]*scale;
            expected[b*elements+i]=std::clamp(int(std::lround(scaled)), -127,127);
        }
    }
    for (unsigned i=0; i<((blocks+7)/8)*8; i++) access(d, 0x800+i*4, true, 0xdeadbeef);
    if (rotate) for (unsigned i=0; i<length; i+=32) {
        uint32_t word=0;
        for (unsigned lane=0; lane<32; lane++) word |= uint32_t(signs[i+lane]<0) << lane;
        access(d, 0x1000+i/8, true, word);
    }
    access(d, 0x3ffc, true, 0xdeadbeef); access(d, 0x4000+length, true, 0xdeadbeef);
    if (rotate) {
        for (unsigned i=0; i<8; i++) access(d,0xa00+i*4,true,0xdeadbeef);
        access(d,0x9ffc,true,0xdeadbeef); access(d,0xa000+length,true,0xdeadbeef);
    }
    start(d, quant(elements,blocks), rotate);
    unsigned cycles=0;
    for (; cycles<100000; cycles++) {
        require(!d.seq_error && !d.had_error, "resident conversion/rotation failed");
        if (!(d.seq_status & 1)) break;
        tick(d);
    }
    unsigned descriptors=rotate ? 3 : 1;
    require(cycles<100000 && ((d.seq_status >> 8) & 0xffff)==descriptors, "list did not retire exactly");
    require(d.quant_blocks==blocks && d.quant_reads==(rotate ? blocks*130 : length/4) && d.quant_writes==length/32,
            "incorrect conversion counters");
    for (unsigned i=0; i<length; i+=4) {
        uint32_t word=0;
        for (unsigned lane=0; lane<4; lane++) word |= uint32_t(uint8_t(expected[i+lane])) << (lane*8);
        require(access(d,0x4000+i,false)==word,"incorrect resident quantized bytes");
    }
    for (unsigned b=0; b<blocks; b++)
        require(int32_t(access(d,0x800+b*4,false))==exponents[b],"incorrect resident block exponent");
    for (unsigned b=blocks; b<((blocks+7)/8)*8; b++)
        require(access(d,0x800+b*4,false)==0xdeadbeef,"exponent tail clobbered");
    require(access(d,0x3ffc,false)==0xdeadbeef && access(d,0x4000+length,false)==0xdeadbeef,"output guard clobbered");
    std::vector<float> rotated(length);
    if (rotate) for (unsigned b=0; b<length/1024; b++) {
        unsigned maximum=0;
        for (unsigned i=0; i<1024; i++) {
            int total=0;
            for (unsigned j=0; j<1024; j++)
                total += (__builtin_parity(i&j) ? -1 : 1)*expected[b*1024+j]*signs[b*1024+j];
            require(int32_t(access(d,0x22000+(b*1024+i)*4,false))==total,"wrong quantize/Hadamard chain");
            rotated[b*1024+i]=std::ldexp(float(total),exponents[b]-5);
            maximum=std::max(maximum,unsigned(std::abs(total)));
        }
        require(access(d,0x900+b*4,false)==maximum,"wrong rotation block maximum");
    }
    if (rotate) {
        float maximum=0;
        for (float value : rotated) maximum=std::max(maximum,std::fabs(value));
        int exponent=0; if (maximum) { std::frexp(maximum,&exponent); exponent-=7; }
        require(int32_t(access(d,0xa00,false))==exponent,"wrong whole-rotation exponent");
        for (unsigned i=1; i<8; i++) require(access(d,0xa00+i*4,false)==0xdeadbeef,"result exponent tail clobbered");
        const float scale=std::ldexp(1.0f,-exponent);
        // Consume the final bytes in the two exact 68-group K segments when
        // this is the full FFN. The split crosses the ninth Hadamard block.
        unsigned segment=length==17408 ? 8704 : length;
        unsigned nonzero[2]={0,0};
        int projections[2][3]={{0,0,0},{0,0,0}};
        for (unsigned begin=0; begin<length; begin+=segment) for (unsigned i=begin; i<begin+segment; i+=4) {
            uint32_t word=0;
            for (unsigned lane=0; lane<4; lane++) {
                volatile float scaled=rotated[i+lane]*scale;
                int q=std::clamp(int(std::lround(scaled)),-127,127);
                nonzero[begin/segment] += q!=0;
                word |= uint32_t(uint8_t(q)) << (lane*8);
                for (unsigned row=0; row<3; row++) {
                    unsigned k=i+lane;
                    projections[begin/segment][row] += q*(int((k*7+k/13+row)%3)-1);
                }
            }
            require(access(d,0xa000+i,false)==word,"wrong whole-vector requantization/FFN split bytes");
        }
        if (!wide_scales && length==17408)
            require(nonzero[0]>segment/2 && nonzero[1]>segment/2,"both FFN K segments must exercise nonzero data");
        require(access(d,0x9ffc,false)==0xdeadbeef && access(d,0xa000+length,false)==0xdeadbeef,"requant output guard clobbered");
    }
    std::printf("resident_quant elements=%u blocks=%u rotate=%u wide_scales=%u sequence_cycles=%u\n",elements,blocks,rotate,wide_scales,cycles);
}
int main() {
    Vactivation_cluster d;
    d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    for (unsigned i=0; i<100; i++) tick(d);
    complete(d,1024,17,true); complete(d,1024,17,true,false);
    complete(d,17408,1); complete(d,5120,1); complete(d,32,5);
    // Reset points across the 17-block list (14,024 cycles with the 32-lane Hadamard).
    for (unsigned stop : {20u,160u,260u,1300u,6000u,11000u,13500u}) {
        start(d,quant(1024,17),true);
        for (unsigned i=0; i<stop; i++) tick(d);
        require(d.seq_status&1,"list completed before reset case");
        reset_cluster(d); complete(d,1024,1,true);
    }
    for (unsigned mode=0; mode<15; mode++) {
        auto words=quant(1024,1);
        for (unsigned i=0; i<1024; i++) access(d,0x10000+i*4,true,bits(1.0f));
        if (mode==0) words[0] |= 1u<<10;   // bit 9 is OVERLAP (legal); 10 is reserved on units
        if (mode==1) words[1] |= 1u<<24;
        if (mode==2) words[4]=65536;
        if (mode==3) words[5]=65536;
        if (mode==4) words[6]=1;
        if (mode==5) words[15]=1;
        if (mode==6) words[4]=40;
        if (mode==7) words[1]=0xffffe0;
        if (mode==8) words[2]=64;  // Protect the later descriptor, not only the active one.
        if (mode==9) words[3]=0;
        if (mode==10) words[2]=words[1];
        if (mode==11) access(d,0x10000,true,0x7f800000);
        if (mode==12) for (unsigned i=0; i<1024; i++) access(d,0x10000+i*4,true,1);
        if (mode>=13) {
            words[5]=2;
            for (unsigned i=1024; i<2048; i++) access(d,0x10000+i*4,true,mode==13 ? 0x7f800000 : 1);
        }
        access(d,0x4000,true,0xdeadbeef);
        start(d,words,true);
        for (unsigned i=0; i<500; i++) tick(d);
        require((d.seq_status&1) && d.seq_error && !((d.seq_status>>8)&0xffff),"invalid conversion was published");
        require(access(d,0,false)==words[0],"rejected command overwrote list");
        if (mode<13) require(access(d,0x4000,false)==0xdeadbeef,"rejected command wrote output");
        else require(d.quant_blocks==1 && access(d,0x4000,false)==0x40404040,"partial prefix failure not exercised");
        reset_cluster(d);
    }
    for (unsigned mode=0; mode<18; mode++) {
        auto words=requant(1);
        access(d,0x800,true,0); access(d,0x900,true,128); access(d,0xa00,true,0xdeadbeef);
        access(d,0xa000,true,0xdeadbeef);
        for (unsigned i=0; i<1024; i++) access(d,0x22000+i*4,true,128);
        if (mode==0) words[0] |= 1u<<10;   // bit 9 is OVERLAP (legal); 10 is reserved on units
        if (mode>=1 && mode<=5) words[mode] |= 1u<<24;
        if (mode==6) words[6]=65536;
        if (mode==7) words[7]=1;
        if (mode==8) words[15]=1;
        if (mode==9) words[6]=0;
        if (mode==10) words[4]=0xffffe0;
        if (mode==11) words[2]=0;
        if (mode==12) words[5]=0;
        if (mode==13) words[2]=words[4];
        if (mode==14) access(d,0x800,true,1024);  // Reject full int32, not truncated e8.
        if (mode==15) access(d,0x900,true,131073);
        if (mode==16) access(d,0x22000,true,0x80000000);
        if (mode==17) {
            words[6]=2; access(d,0x804,true,0); access(d,0x904,true,129);
            for (unsigned i=1024; i<2048; i++) access(d,0x22000+i*4,true,128);
        }
        start(d,words);
        for (unsigned i=0; i<1000; i++) tick(d);
        require((d.seq_status&1) && d.seq_error && !((d.seq_status>>8)&0xffff),"invalid requantization was published");
        require(access(d,0,false)==words[0] && access(d,0xa00,false)==0xdeadbeef,"rejected requantization corrupted publication");
        if (mode<17) require(access(d,0xa000,false)==0xdeadbeef,"early-rejected requantization wrote output");
        else require(d.quant_blocks==1 && access(d,0xa000,false)==0x40404040,"requant partial prefix failure not exercised");
        reset_cluster(d);
    }
    std::puts("ACTIVATION_CLUSTER_PASS ff_blocks=17 stages=3 scale_profiles=2 reset_reuse=7 rejected=33 partial_prefix_failure=3");
}

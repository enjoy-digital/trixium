/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Real sequencer, shared engine port arbitration, two-port scratchpad and Wishbone sideband.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include "Vhadamard_cluster.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t rng=971;
static uint32_t random_word() { rng=rng*1664525u+1013904223u; return rng; }
static void tick(Vhadamard_cluster &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t access(Vhadamard_cluster &d, uint32_t address, bool write, uint32_t value=0) {
    d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
    for (unsigned cycle=0; cycle<2000; cycle++) {
        d.sys_clk=0; d.eval(); bool ack=d.host_ack; uint32_t result=d.host_dat_r;
        d.sys_clk=1; d.eval();
        if (ack) { d.host_cyc=0; d.host_stb=0; tick(d); return result; }
    }
    require(false, "Wishbone timeout"); return 0;
}
static void reset_cluster(Vhadamard_cluster &d) {
    d.reset_value=1; d.reset_re=1; tick(d); d.reset_re=0;
    for (unsigned i=0; i<100; i++) tick(d);
    require(!d.reset_busy && !d.reset_stuck && !(d.seq_status & 1) && !d.seq_error,
            "cluster reset did not recover");
}
static std::array<uint32_t, 16> descriptor(unsigned blocks, bool inverse) {
    return {3, 0x2000, 0x1000, 0x10000, blocks, 0x800, unsigned(inverse), 0,0,0,0,0,0,0,0,0};
}
static void start(Vhadamard_cluster &d, const std::array<uint32_t, 16> &words) {
    for (unsigned i=0; i<16; i++) access(d, i*4, true, words[i]);
    d.seq_base=0; d.seq_count=1; d.seq_start=1; d.seq_start_re=1; tick(d); d.seq_start_re=0;
    require(d.seq_status & 1, "list not started");
}
static void complete(Vhadamard_cluster &d, unsigned blocks, bool inverse) {
    const unsigned length=blocks*1024;
    std::vector<int> values(length), signs(length), expected(length);
    for (unsigned i=0; i<length; i++) {
        values[i]=int(random_word() >> 24)-128;
        signs[i]=(random_word() & 0x10000) ? 1 : -1;
    }
    for (unsigned i=0; i<length; i+=4) {
        uint32_t word=0;
        for (unsigned lane=0; lane<4; lane++) word |= uint32_t(uint8_t(values[i+lane])) << (8*lane);
        access(d, 0x2000+i, true, word);
    }
    for (unsigned i=0; i<length; i+=32) {
        uint32_t word=0;
        for (unsigned lane=0; lane<32; lane++) word |= uint32_t(signs[i+lane]<0) << lane;
        access(d, 0x1000+i/8, true, word);
    }
    for (unsigned b=0; b<blocks; b++) for (unsigned i=0; i<1024; i++) {
        int total=0;
        for (unsigned j=0; j<1024; j++)
            total += (__builtin_parity(i & j) ? -1 : 1)*values[b*1024+j]*(inverse ? 1 : signs[b*1024+j]);
        expected[b*1024+i]=total*(inverse ? signs[b*1024+i] : 1);
    }
    for (unsigned i=0; i<((blocks+7)/8)*8; i++) access(d, 0x800+i*4, true, 0xdeadbeef);
    access(d, 0xfffc, true, 0xdeadbeef); access(d, 0x10000+length*4, true, 0xdeadbeef);
    start(d, descriptor(blocks, inverse));
    unsigned cycles=0;
    for (; cycles<100000; cycles++) {
        require(!d.seq_error, "rotation failed");
        if (!(d.seq_status & 1)) break;
        tick(d);
    }
    require(cycles<100000 && ((d.seq_status >> 8) & 0xffff)==1, "list did not retire once");
    require(d.blocks_done==blocks && d.output_beats==length/8, "bad command progress counts");
    for (unsigned i=0; i<length; i++) {
        int32_t actual=int32_t(access(d, 0x10000+i*4, false));
        if (actual!=expected[i]) {
            std::fprintf(stderr, "blocks=%u inverse=%u index=%u expected=%d got=%d\n", blocks, inverse, i, expected[i], actual);
            require(false, "wrong rotated scratchpad value");
        }
    }
    for (unsigned b=0; b<blocks; b++) {
        unsigned maximum=0;
        for (unsigned i=0; i<1024; i++) maximum=std::max(maximum, unsigned(std::abs(expected[b*1024+i])));
        require(access(d, 0x800+b*4, false)==maximum, "wrong per-block absmax");
    }
    for (unsigned i=blocks; i<((blocks+7)/8)*8; i++)
        require(access(d, 0x800+i*4, false)==0xdeadbeef, "stats tail clobbered");
    require(access(d, 0xfffc, false)==0xdeadbeef && access(d, 0x10000+length*4, false)==0xdeadbeef,
            "output guard clobbered");
    std::printf("rotation blocks=%u inverse=%u sequence_cycles=%u\n", blocks, inverse, cycles);
}
int main() {
    Vhadamard_cluster d;
    d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    for (unsigned i=0; i<100; i++) tick(d);
    complete(d, 17, false); complete(d, 1, true); complete(d, 5, false);
    for (unsigned stop : {30u, 400u, 1500u, 5000u}) {
        start(d, descriptor(17, false));
        for (unsigned i=0; i<stop; i++) tick(d);
        require((d.seq_status & 1) && !((d.seq_status >> 8) & 0xffff), "cancelled list was already published");
        reset_cluster(d); complete(d, 1, stop & 4);
    }
    for (unsigned mode=0; mode<4; mode++) {
        auto words=descriptor(1, false);
        if (mode==0) words[6]=4;   // w6 bit 2: reserved (bit 1 is the half-block flag, ABI 1.1)
        if (mode==1) words[5]=0;
        if (mode==2) words[4]=65535;
        if (mode==3) words[1]=0xffffe0;
        access(d, 0x10000, true, 0xdeadbeef);
        start(d, words);
        for (unsigned i=0; i<100; i++) tick(d);
        require((d.seq_status & 1) && d.seq_error && !((d.seq_status >> 8) & 0xffff), "invalid command retired");
        require(access(d, 0x10000, false)==0xdeadbeef, "invalid command wrote output");
        reset_cluster(d);
    }
    std::puts("HADAMARD_CLUSTER_PASS full_ffn_blocks=17 reset_reuse=4 rejected=4");
}

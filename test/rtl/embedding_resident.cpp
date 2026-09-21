// SPDX-License-Identifier: BSD-2-Clause
#include "Vembedding_resident.h"
#include "bonsai_sequence.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"%s\n",why); std::exit(1); }
}
static void tick(Vembedding_resident &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
static uint32_t word(std::ifstream &f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4); require(bool(f),"short vector file");
    return b[0] | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
static uint32_t access(Vembedding_resident &d, uint32_t address, bool write, uint32_t value=0) {
    d.host_cyc=1; d.host_stb=1; d.host_we=write; d.host_adr=address; d.host_dat_w=value; d.host_sel=15;
    for (unsigned i=0; i<2000; i++) {
        d.sys_clk=0; d.eval(); bool ack=d.host_ack; uint32_t result=d.host_dat_r;
        d.sys_clk=1; d.eval();
        if (ack) { d.host_cyc=0; d.host_stb=0; tick(d); return result; }
    }
    require(false,"Wishbone timeout"); return 0;
}
static void reset(Vembedding_resident &d) {
    d.reset_value=1; d.reset_re=1; tick(d); d.reset_re=0;
    for (unsigned i=0; i<100; i++) tick(d);
    require(!d.reset_busy && !d.reset_stuck && !(d.seq_status&1) && !d.embedding_error
        && !d.quant_error && !d.region_error && !d.embedding_busy && !d.quant_busy && !d.engine_busy,"common reset failed");
}
static void start(Vembedding_resident &d, unsigned descriptors) {
    d.seq_base=0; d.seq_count=descriptors; d.seq_start=1; d.seq_start_re=1;
    tick(d); d.seq_start_re=0; require(d.seq_status&1,"sequence not started");
}
int main() {
    std::ifstream f("vectors.bin",std::ios::binary);
    require(word(f)==0x45525331,"bad vector magic"); unsigned profiles=word(f), region_fetch=word(f);
    Vembedding_resident d; d.sys_rst=1; tick(d); tick(d); d.sys_rst=0;
    for (unsigned profile=0; profile<profiles; profile++) {
        unsigned nw=word(f), nc=word(f), nd=word(f), exponent=word(f);
        std::map<uint32_t,uint32_t> writes, checks;
        for (unsigned i=0; i<nw; i++) { unsigned a=word(f), v=word(f); writes[a]=v; }
        for (unsigned i=0; i<nc; i++) { unsigned a=word(f), v=word(f); checks[a]=v; }
        // Execute descriptors built by the portable C firmware library. The
        // Python vector file independently supplies the exact encoding oracle.
        unsigned c_descriptors=0;
        for (unsigned index=0; index<nd; index++) {
            unsigned base=index*64;
            auto w=[&](unsigned i) { return writes.at(base+4*i); };
            bns_seq_descriptor desc;
            int result;
            if (w(0)==4) result=bns_seq_desc_quantize(&desc,w(1),w(2),w(3),w(4),w(5));
            else if (w(0)==9) result=bns_seq_desc_embedding(&desc,w(1),w(2),w(3),w(4));
            else if (w(0)==10) result=bns_seq_desc_region_fetch(&desc,w(1),
                uint64_t(w(2)) | (uint64_t(w(3))<<32),w(4),w(5));
            else continue; // Unchanged legacy physical fetch/store descriptors.
            require(result==BNS_SEQ_DESC_OK,"C descriptor encoder rejected valid command");
            for (unsigned i=0;i<BNS_SEQ_DESC_WORDS;i++) {
                require(desc.word[i]==w(i),"C/Python descriptor mismatch");
                writes[base+4*i]=desc.word[i];
            }
            c_descriptors++;
        }
        require(c_descriptors==(region_fetch ? 6u : 2u),"missing C descriptor coverage");
        auto install=[&]() { for (auto [a,v]:writes) access(d,a,true,v); };
        // A malformed sequencer descriptor never reaches the client, whose
        // counters otherwise retain the previous successful operation.
        reset(d);
        // Fail before any embedding input read: protected later descriptor,
        // reserved high address bits, then invalid group geometry.
        install();
        unsigned field=profile==0 ? 3 : (profile==1 ? 1 : 4);
        unsigned invalid=profile==0 ? 64 : (profile==1 ? 0x1004000 : 0);
        unsigned expected_error=profile==0 ? 0x400 : (profile==1 ? 2 : 0x100);
        access(d,128+4*field,true,invalid); start(d,nd);
        unsigned cycle=0;
        for (; cycle<10000 && !d.embedding_error; cycle++) tick(d);
        if (d.embedding_error!=expected_error)
            std::fprintf(stderr,"profile=%u error=%x expected=%x dcount=%u\n",profile,d.embedding_error,expected_error,d.seq_status>>8);
        require(cycle<10000 && d.embedding_error==expected_error && (d.seq_status&1)
            && (d.seq_status>>8)==2,"bad descriptor was not rejected");
        require(!d.embedding_reads && !d.embedding_writes,"rejected descriptor touched embedding data");
        reset(d);
        for (auto [a,v]:writes) if (a>=0x8000) require(access(d,a,false)==v,"failed operation modified outputs");

        // In-flight common reset while reading, after a committed FP32 prefix,
        // or in its downstream quantizer, followed by full reuse of all units.
        install(); start(d,nd); bool interrupted=false;
        for (cycle=0; cycle<30000; cycle++) {
            require(!d.embedding_error && !d.quant_error,"abort sequence failed early");
            if ((profile==0 && d.embedding_busy && d.embedding_reads>=1) ||
                (profile==1 && d.embedding_busy && d.embedding_writes>=17) ||
                (profile==2 && d.quant_busy && d.quant_reads>=1)) { interrupted=true; break; }
            tick(d);
        }
        require(interrupted && (d.seq_status&1),"reset missed requested stage"); reset(d);

        install(); unsigned before=d.engine_done; start(d,nd); bool saw_embedding=false;
        unsigned input_beats=0, readback_beats=0;
        for (cycle=0; cycle<100000; cycle++) {
            if (d.embedding_error || d.quant_error)
                std::fprintf(stderr,"profile=%u cycle=%u embedding=%x quant=%x dcount=%u\n",
                             profile,cycle,d.embedding_error,d.quant_error,d.seq_status>>8);
            require(!d.embedding_error && !d.quant_error && !d.region_error,"resident sequence failed");
            if (d.embedding_busy) {
                saw_embedding=true;
                require(d.engine_done-before==2 && !d.engine_busy && !d.quant_busy,"embedding ownership overlap");
            }
            if (!(d.seq_status&1)) break;
            d.sys_clk=0; d.eval();
            if (d.ar_valid && d.ar_ready) {
                unsigned count=d.ar_len+1;
                unsigned retired=d.engine_done-before;
                uint64_t base=retired==0 ? 0x1000+profile*0x2000 : 0x2000+profile*0x2000;
                if (retired<2) {
                    if (region_fetch) base+=uint64_t(1)<<32;
                    unsigned offset=retired==0 ? input_beats : input_beats-35;
                    require(d.ar_addr==base+offset*32,"model fetch lost region offset or high address bits");
                    input_beats+=count;
                } else readback_beats+=count;
            }
            d.sys_clk=1; d.eval();
        }
        require(cycle<100000 && saw_embedding && (d.seq_status>>8)==nd,"sequence did not retire");
        require(d.engine_done-before==6,"wrong fetch/store retirement count");
        require(input_beats==55 && readback_beats==800,"wrong physical read traffic");
        require(!d.manual_read_accepted && !d.manual_read_completed,"sequenced read credited manual CSR completion");
        require(d.embedding_reads==55 && d.embedding_writes==640,"embedding transaction count");
        require(d.embedding_exponent==(exponent&1023),"embedding exponent mismatch");
        // FP32 absmax and conversion make two complete input passes.
        require(d.quant_blocks==1 && d.quant_reads==1280 && d.quant_writes==160,"quantization transaction count");
        for (auto [a,v]:checks) {
            unsigned got=access(d,a,false);
            if (got!=v) std::fprintf(stderr,"profile=%u addr=%x got=%08x expected=%08x\n",profile,a,got,v);
            require(got==v,"resident memory mismatch");
        }
        std::printf("EMBEDDING_RESIDENT profile=%u descriptors=%u cycles=%u c_descriptors=%u\n",
                    profile,nd,cycle,c_descriptors);
    }
    require(f.peek()==std::char_traits<char>::eof(),"trailing vector bytes");
    std::printf("EMBEDDING_RESIDENT_PASS profiles=%u rejected=3 common_reset_reuse=3 region_fetch=%u\n",profiles,region_fetch);
}

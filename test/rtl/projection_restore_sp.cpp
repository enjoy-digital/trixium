// SPDX-License-Identifier: BSD-2-Clause
// Compiled client with one-cycle reads, masked writes and an IEEE FP32 oracle.
#include "Vprojection_restore_sp.h"
#include "verilated.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static uint32_t seed=771525;
static uint32_t random_word() { seed=seed*1664525u+1013904223u; return seed; }
static uint32_t oracle(uint32_t word, int exponent) {
    int32_t x;
    std::memcpy(&x, &word, 4);
    volatile float converted=float(x), scale=std::ldexp(1.0f, exponent);
    float restored=converted*scale;
    require(std::isfinite(restored), "bad finite oracle input");
    uint32_t result;
    std::memcpy(&result, &restored, 4);
    return result;
}
struct Write {
    unsigned address, mask;
    std::array<uint32_t, 8> data;
    bool operator==(const Write &b) const { return address==b.address && mask==b.mask && data==b.data; }
};
struct Fixture {
    Vprojection_restore_sp d;
    std::vector<uint32_t> mem=std::vector<uint32_t>(65536, 0xdeadbeefu);
    unsigned cycles=0, reads=0, writes=0, runs=0, cancellations=0, resets=0;
    bool held=false;
    Write previous{};
    void tick() {
        d.sys_clk=0; d.eval();
        Write output{d.sp_wr_adr, d.sp_wr_we, {}};
        for (unsigned i=0; i<8; i++) output.data[i]=d.sp_wr_dat[i];
        std::array<uint32_t, 8> response{};
        if (d.cancel) require(!d.sp_wr_req && !d.sp_rd_req && !d.cmd_ready, "cancel did not mask IO");
        if (d.cancel || d.sys_rst) held=false;
        else {
            if (held) require(d.sp_wr_req && output==previous, "stalled write changed");
            held=d.sp_wr_req && !d.sp_wr_gnt;
            previous=output;
            if (d.sp_rd_req) {
                require(d.sp_rd_adr < 8192, "read out of bounds");
                for (unsigned i=0; i<8; i++) response[i]=mem[8*d.sp_rd_adr+i];
                reads++;
            }
            if (d.sp_wr_req && d.sp_wr_gnt) {
                require(d.sp_wr_adr < 8192, "write out of bounds");
                for (unsigned i=0; i<8; i++) if (d.sp_wr_we & (1u << i)) mem[8*d.sp_wr_adr+i]=d.sp_wr_dat[i];
                writes++;
            }
        }
        d.sys_clk=1; d.eval();
        for (unsigned i=0; i<8; i++) d.sp_rd_dat[i]=response[i];
        d.eval(); cycles++;
    }
    void reset(bool cancel) {
        d.cmd_valid=0; d.cancel=cancel; d.sys_rst=!cancel;
        tick();
        d.cancel=0; d.sys_rst=0; tick();
        require(!d.busy && !d.error && !d.done, "reset did not clear controller");
        require(!d.read_beats && !d.output_beats, "reset did not clear counters");
        if (cancel) cancellations++; else resets++;
    }
    void submit(unsigned src, unsigned dst, unsigned count, int32_t adjustment=0,
                unsigned metadata=0x3ffe0, unsigned protect_start=0, unsigned protect_end=0) {
        d.cmd_src=src; d.cmd_dst=dst; d.cmd_elements=count; d.cmd_adjustment=uint32_t(adjustment);
        d.cmd_exponents=metadata; d.cmd_protect_start=protect_start; d.cmd_protect_end=protect_end;
        d.cmd_valid=1; d.eval();
        require(d.cmd_ready, "command not ready");
        tick();
        d.cmd_valid=0;
        // Captured command, not live host fields.
        d.cmd_src=0xffffff; d.cmd_dst=0xffffff; d.cmd_elements=0;
        d.cmd_adjustment=0x7fffffff; d.cmd_exponents=0xffffff;
    }
    void valid(unsigned count, int32_t metadata, int32_t adjustment, bool inplace, bool stalled) {
        const int64_t sum=int64_t(metadata)+adjustment;
        require(sum>=-149 && sum<=127, "invalid test exponent");
        const int exponent=int(sum);
        const unsigned src=count>30000 ? 0 : 0x1000, dst=inplace ? src : 0x20000;
        const unsigned beats=(count+7)/8;
        std::vector<uint32_t> inputs(count), expected(count);
        for (unsigned i=0; i<8*beats; i++) mem[dst/4+i]=0xdeadbeef;
        for (unsigned i=0; i<8*beats; i++) {
            const uint32_t x=i>=count ? 0x7fffffff : exponent==127 ? 0 : random_word();
            mem[src/4+i]=x;
            if (i<count) { inputs[i]=x; expected[i]=oracle(x, exponent); }
        }
        mem[0x3ffe0/4]=uint32_t(metadata);
        const unsigned before_reads=reads, before_writes=writes, before_cycles=cycles;
        submit(src, dst, count, adjustment);
        bool finished=false;
        for (unsigned cycle=0; cycle<100000; cycle++) {
            d.sp_wr_gnt=!stalled || (cycle%97>=32 && (random_word() & 3));
            tick();
            require(!d.error, "valid restoration failed");
            if (d.done) { finished=true; break; }
        }
        require(finished, "valid restoration timeout");
        for (unsigned i=0; i<count; i++) {
            if (mem[dst/4+i] != expected[i]) {
                std::fprintf(stderr, "i=%u count=%u exponent=%d in=%08x expect=%08x got=%08x\n",
                             i, count, exponent, inputs[i], expected[i], mem[dst/4+i]);
                require(false, "memory result mismatch");
            }
        }
        for (unsigned i=count; i<8*beats; i++)
            require(mem[dst/4+i]==(inplace ? 0x7fffffffu : 0xdeadbeefu), "tail lanes changed");
        require(mem[0x3ffe0/4]==uint32_t(metadata), "exponent metadata changed");
        require(reads-before_reads==beats+1 && d.read_beats==beats+1, "read credit count mismatch");
        require(writes-before_writes==beats && d.output_beats==beats, "write count mismatch");
        require(!d.busy, "completion published while busy");
        if (count==65528 && !stalled)
            std::printf("RESTORE_SP_FULL_INPLACE elements=%u cycles=%u reads=%u writes=%u\n",
                        count, cycles-before_cycles, reads-before_reads, writes-before_writes);
        tick(); runs++;
    }
    void rejected(unsigned mask, unsigned expected_reads) {
        for (unsigned i=0; i<100; i++) {
            d.sp_wr_gnt=1; tick();
            require(!d.done && !d.sp_wr_req, "failed command wrote or published success");
        }
        require(d.busy && (d.error & mask), "missing sticky failure");
        require(d.read_beats==expected_reads && !d.output_beats, "rejected command performed memory IO");
        reset(true);
    }
};
int main() {
    require(std::fegetround()==FE_TONEAREST, "oracle must use RNE");
    Fixture f;
    f.reset(false);
    for (bool inplace : {false, true}) {
        for (unsigned count : {1u, 3u, 8u, 65u, 1023u, 8191u}) {
            f.valid(count, -149, 0, inplace, true);
            f.valid(count, 31, -47, inplace, false);
        }
        f.valid(3, 127, 0, inplace, true); // Inactive INT_MAX must not cause overflow.
    }
    f.valid(65528, -31, 0, true, false); // Largest valid in-place span + metadata beat.
    f.valid(65528, -149, 0, true, true);
    f.valid(9, INT32_MAX, -INT32_MAX, false, true);
    f.valid(9, INT32_MIN, INT32_MAX, true, true);
    for (auto pair : std::array<std::array<int32_t, 2>, 7>{{
            {{-150,0}}, {{128,0}}, {{0x40000000,0}}, {{INT32_MAX,1}},
            {{INT32_MIN,-1}}, {{INT32_MAX,INT32_MAX}}, {{INT32_MIN,INT32_MIN}}}}) {
        f.mem[0x3ffe0/4]=uint32_t(pair[0]); f.submit(0x1000, 0x20000, 3, pair[1]); f.rejected(8,1);
    }
    f.submit(0xffffe0, 0x20000, 65); f.rejected(2,0);
    f.submit(0x1000, 0xffffe0, 65); f.rejected(2,0);
    f.submit(0x1000, 0x1020, 65); f.rejected(4,0);
    f.submit(0x1000, 0x20000, 3, 0, 0x3ffe0, 0x2001f, 0x20020); f.rejected(64,0);
    for (unsigned delay : {0u, 1u, 2u, 3u, 5u, 8u, 12u, 24u, 50u}) {
        f.mem[0x3ffe0/4]=0; f.d.sp_wr_gnt=0;
        f.submit(0x1000, 0x20000, 128);
        for (unsigned i=0; i<delay; i++) f.tick();
        f.reset(delay%2==0);
        f.valid(3, -31, 0, false, true);
    }
    // A late overflow must invalidate a written prefix, never publish completion.
    for (unsigned i=0; i<128; i++) { f.mem[0x1000/4+i]=(i==112 ? 0x7fffffffu : 0); f.mem[0x20000/4+i]=0xdeadbeef; }
    f.mem[0x3ffe0/4]=127; f.submit(0x1000, 0x20000, 128);
    for (unsigned i=0; i<100; i++) { f.d.sp_wr_gnt=1; f.tick(); require(!f.d.done, "overflow published success"); }
    require(f.d.busy && f.d.error==16 && f.d.output_beats>0 && f.d.output_beats<16, "overflow failed to invalidate prefix");
    require(f.mem[0x20000/4+112]==0xdeadbeef && !f.d.sp_wr_req && !f.d.sp_rd_req, "overflow beat wrote or kept streaming");
    f.reset(true); f.valid(3, -149, 0, false, true);
    std::printf("PROJECTION_RESTORE_SP_PASS runs=%u cancellations=%u resets=%u cycles=%u\n",
                f.runs, f.cancellations, f.resets, f.cycles);
    return 0;
}

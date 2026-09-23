// SPDX-License-Identifier: BSD-2-Clause
// Compiled-RTL driver for BonsaiSiluScratchpad over a modelled scratchpad.
// usage: Vsilu_sp image.bin commands.bin out.bin
//   image.bin    initial scratchpad contents (whole beats)
//   commands.bin records of 9 x u32: op src_g src_m dst fixmap elements protect_start
//                protect_end want_error (0 = must complete; else the exact error bits)
//   out.bin      final scratchpad contents
// Reads return one clock after the request; the write grant is random and the payload
// must hold while ungranted. Prints one line per command: fixups and cycles.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "Vsilu_sp.h"

static void require(bool ok, const char *what) { if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); } }

static std::vector<uint32_t> load(const char *path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    require(b.size() % 4 == 0, "file length");
    std::vector<uint32_t> w(b.size()/4);
    std::memcpy(w.data(), b.data(), b.size());
    return w;
}

int main(int argc, char **argv) {
    require(argc == 4, "usage: image commands out");
    std::vector<uint32_t> mem = load(argv[1]);
    std::vector<uint32_t> cmds = load(argv[2]);
    require(mem.size() % 8 == 0 && cmds.size() % 9 == 0, "shapes");
    const uint32_t beats = mem.size()/8;
    Vsilu_sp d;
    uint32_t state = 777;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return state >> 8; };
    uint32_t pending[8] = {0};
    bool held = false; uint32_t held_adr = 0, held_we = 0, held_dat[8] = {0};
    uint64_t cycle = 0;
    auto tick = [&](bool grant) {
        d.sp_wr_gnt = grant;
        d.sys_clk = 0; d.eval();
        if (held) {
            require(d.sp_wr_req && d.sp_wr_adr == held_adr && d.sp_wr_we == held_we &&
                    !std::memcmp(held_dat, &d.sp_wr_dat[0], 32), "write payload changed while ungranted");
        }
        if (d.sp_wr_req && grant) {
            require(d.sp_wr_adr < beats, "write out of range");
            for (int l = 0; l < 8; l++)
                if (d.sp_wr_we >> l & 1) mem[8*d.sp_wr_adr + l] = d.sp_wr_dat[l];
        }
        held = d.sp_wr_req && !grant;
        held_adr = d.sp_wr_adr; held_we = d.sp_wr_we;
        std::memcpy(held_dat, &d.sp_wr_dat[0], 32);
        bool rd = d.sp_rd_req;
        if (rd) {
            require(d.sp_rd_adr < beats, "read out of range");
            for (int l = 0; l < 8; l++) pending[l] = mem[8*d.sp_rd_adr + l];
        }
        d.sys_clk = 1; d.eval();
        for (int l = 0; l < 8; l++) d.sp_rd_dat[l] = rd ? pending[l] : 0xA5A5A5A5u;
        cycle++;
    };
    d.sys_rst = 1; tick(true); tick(true); d.sys_rst = 0;
    for (size_t c = 0; c < cmds.size(); c += 9) {
        const uint32_t *r = &cmds[c];
        d.cmd_op = r[0]; d.cmd_src_g = r[1]; d.cmd_src_m = r[2]; d.cmd_dst = r[3]; d.cmd_fixmap = r[4];
        d.cmd_elements = r[5]; d.cmd_protect_start = r[6]; d.cmd_protect_end = r[7];
        d.cmd_valid = 1;
        d.sys_clk = 0; d.eval();
        require(d.cmd_ready, "command not accepted");
        tick(true);
        d.cmd_valid = 0; d.cmd_elements = 0; d.cmd_dst = 0xFFFFFF;   // captured, not live
        const uint64_t start = cycle;
        bool done = false;
        for (uint64_t n = 0; n < 40ull*r[5] + 40000 && !done; n++) {
            tick((rnd() % 4) != 0);
            if (r[8]) {
                require(!d.done && !d.sp_wr_req, "rejected command touched memory");
                if (n == 30) break;
            } else {
                require(d.error == 0, "unexpected error");
            }
            done = d.done;
        }
        if (r[8]) {
            if (d.error != r[8]) { std::fprintf(stderr, "cmd %zu: error %u, want %u\n", c/9, d.error, r[8]); return 1; }
            require(d.busy, "failed command must stay busy");
            d.cancel = 1; tick(true); d.cancel = 0; tick(true);
            require(!d.busy && !d.error, "cancel");
            std::printf("cmd %zu rejected %u\n", c/9, r[8]);
        } else {
            require(done, "timeout");
            std::printf("cmd %zu op %u elements %u fixups %u cycles %llu\n", c/9, r[0], r[5],
                        d.fixups, (unsigned long long)(cycle - start));
        }
    }
    std::ofstream o(argv[3], std::ios::binary);
    o.write(reinterpret_cast<const char *>(mem.data()), 4*mem.size());
    return 0;
}

// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// VexRiscv PC profiler for the Verilator sim: a `perf record` for the softcore.
//
// Why. Measured on the card, a token is engine-busy 7%, vector-busy 22%, both 0%, and
// NEITHER 69%. That 69% is the softcore running firmware and nothing in the system could
// attribute it - perf_print's buckets are op-class wall time that already includes the CPU
// glue, so they cannot separate "computing" from "polling" from "stalled".
//
// This samples the retired-instruction PC every cycle and buckets it. Two counters per PC:
//
//   cycles[pc]   incremented when the writeback stage holds this instruction (lastStageIsValid)
//   retired[pc]  incremented when it actually retires    (lastStageIsFiring)
//
// so cycles/retired is per-PC CPI, which is the difference between "this code is slow because
// there is a lot of it" and "this code is slow because it is stalled on a cache miss". A plain
// histogram cannot tell those apart and they need opposite fixes.
//
// Mechanism. LiteX declares three weak hooks in build/sim/core/veril.cpp; litex_sim_eval()
// calls pre/post around every model eval. Defining the strong symbol here overrides the weak
// default at link time - no sim_config entry, no Python-side module. (The dlopen'd sim module
// pattern in build/sim/core/modules/ is NOT usable for this: those only get `pad` pointers and
// never see the Vsim*, so they cannot reach a CPU-internal signal at all.)
//
// The signal path is `sim->sim->VexRiscv->lastStagePc`, verified against the generated
// obj_dir/Vsim_VexRiscv.h (lastStageIsFiring:155, lastStagePc:252, lastStageIsValid:339,
// CsrPlugin_inWfi:501) and obj_dir/Vsim.h:43 / Vsim_sim.h:18 for the cell pointers. VexRiscv
// escapes inlining precisely because those carry /* verilator public */, which is also why the
// names are unmangled. If a Verilator upgrade changes that, this fails to COMPILE rather than
// silently mis-sampling - which is the failure mode we want.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "Vsim.h"
#include "Vsim_sim.h"
#include "Vsim_VexRiscv.h"

namespace {

struct Region { uint32_t base, size; uint64_t *cyc, *ret; };

// Matches MAIN_RAM_BASE/SIZE and integrated_rom_size in trixium/targets/sim.py. A PC outside both
// lands in g_oor, which the dump header reports - so a layout change shouts instead of
// silently mis-attributing.
Region   g_rom = { 0x00000000u, 0x00020000u, nullptr, nullptr };
Region   g_ram = { 0x40000000u, 0x00800000u, nullptr, nullptr };

uint8_t  g_prev_clk = 0;
uint64_t g_cycles = 0, g_bubble = 0, g_wfi = 0, g_oor = 0, g_retired = 0, g_time_ps = 0;
unsigned g_snap = 0;
double   g_last_ckpt_s = 0.0, g_ckpt_period_s = 60.0;
char     g_dir[512] = ".";

// Timeline events for tools/trace_decode.py: 8 bytes each, little-endian.
//   cycle:u32 | kind:u8 (0 = busy-state edge, 1 = marker) | payload:u8[3]
//   kind 0: payload[0] = eng | vec<<1 | seq<<2   (state AFTER the edge)
//   kind 1: payload[0] = marker index into the g_markers order
FILE    *g_evf = nullptr;
uint8_t  g_busy_prev = 0xFF;

inline void event(uint32_t cyc, uint8_t kind, uint8_t p0) {
    if (!g_evf) return;
    uint8_t rec[8] = {(uint8_t)cyc, (uint8_t)(cyc>>8), (uint8_t)(cyc>>16), (uint8_t)(cyc>>24),
                      kind, p0, 0, 0};
    fwrite(rec, 1, 8, g_evf);
}

char     g_tail[128]; size_t g_tail_n = 0;
char    *g_markers[32]; size_t g_nmarkers = 0;

double wall_s() { timeval tv; gettimeofday(&tv, nullptr);
                  return tv.tv_sec + tv.tv_usec*1e-6; }

bool alloc_region(Region &r) {
    const size_t n = r.size >> 2;
    r.cyc = (uint64_t *)calloc(n, sizeof(uint64_t));
    r.ret = (uint64_t *)calloc(n, sizeof(uint64_t));
    return r.cyc && r.ret;
}

void dump(const char *tag) {
    char path[640];
    snprintf(path, sizeof(path), "%s/pcprof-%03u-%s.txt", g_dir, g_snap, tag);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[pcprof] cannot write %s\n", path); return; }
    fprintf(f, "# trixium pc-profile v1\n# snapshot %u\n# marker %s\n", g_snap, tag);
    fprintf(f, "# time_ps %llu\n# cycles %llu\n# bubble_cycles %llu\n# wfi_cycles %llu\n"
               "# out_of_range_cycles %llu\n# retired %llu\n",
            (unsigned long long)g_time_ps, (unsigned long long)g_cycles,
            (unsigned long long)g_bubble,  (unsigned long long)g_wfi,
            (unsigned long long)g_oor,     (unsigned long long)g_retired);
    fprintf(f, "# regions 0x%08x:0x%08x 0x%08x:0x%08x\n",
            g_rom.base, g_rom.size, g_ram.base, g_ram.size);
    fprintf(f, "# columns pc cycles retired\n");
    const Region *rs[2] = { &g_rom, &g_ram };
    for (int k = 0; k < 2; k++)
        for (size_t i = 0, n = rs[k]->size >> 2; i < n; i++)
            if (rs[k]->cyc[i])
                fprintf(f, "%u %llu %llu\n",
                        (unsigned)(rs[k]->base + (uint32_t)(i << 2)),
                        (unsigned long long)rs[k]->cyc[i],
                        (unsigned long long)rs[k]->ret[i]);
    fclose(f);
    fprintf(stderr, "[pcprof] %s: %llu cycles, %llu retired\n", path,
            (unsigned long long)g_cycles, (unsigned long long)g_retired);
    g_snap++;
}

void dump_final() { dump("final"); if (g_evf) fclose(g_evf); g_evf = nullptr; }

inline void sample(Vsim_VexRiscv *cpu) {
    g_cycles++;
    if (cpu->CsrPlugin_inWfi) g_wfi++;
    // Not valid = the writeback stage is empty: i-cache miss, flush, branch bubble. Not
    // attributable to any PC, and the fraction is itself a headline number.
    if (!cpu->lastStageIsValid) { g_bubble++; return; }
    const uint32_t pc = cpu->lastStagePc;
    Region *r;
    if      ((uint32_t)(pc - g_ram.base) < g_ram.size) r = &g_ram;
    else if ((uint32_t)(pc - g_rom.base) < g_rom.size) r = &g_rom;
    else { g_oor++; return; }
    const size_t i = (size_t)((pc - r->base) >> 2);
    r->cyc[i]++;
    if (cpu->lastStageIsFiring) { r->ret[i]++; g_retired++; }
}

void sanitize(const char *in, char *out, size_t n) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < n; i++) {
        const char c = in[i];
        out[j++] = ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '-') ? c : '_';
    }
    out[j] = 0;
}

// Snoop the console byte stream to window the capture. The firmware already prints unique
// markers (SELFTEST DONE, GEN:, GEN DONE, ...) and serial_source_* are top-level Vsim ports,
// which is the most stable thing in the build. On a hit we dump a CUMULATIVE snapshot and do
// NOT reset: the Python side subtracts consecutive snapshots, so windowing is exact and a
// missed marker costs resolution rather than correctness.
inline void uart_sniff(Vsim *sim) {
    if (!sim->serial_source_valid) return;
    if (g_tail_n == sizeof(g_tail) - 1) { memmove(g_tail, g_tail + 1, g_tail_n - 1); g_tail_n--; }
    g_tail[g_tail_n++] = (char)sim->serial_source_data;
    g_tail[g_tail_n]   = 0;
    for (size_t m = 0; m < g_nmarkers; m++) {
        const size_t L = strlen(g_markers[m]);
        // Anchored at the newest byte, so a marker that is a prefix of another cannot fire early.
        if (g_tail_n >= L && !memcmp(g_tail + g_tail_n - L, g_markers[m], L)) {
            char tag[64]; sanitize(g_markers[m], tag, sizeof(tag));
            event((uint32_t)g_cycles, 1, (uint8_t)m);
            dump(tag);
            g_tail_n = 0; g_tail[0] = 0;
            return;
        }
    }
}

} // namespace

extern "C" void litex_sim_user_init(void *vsim) {
    (void)vsim;
    if (const char *d = getenv("LITEX_PC_PROF_DIR"))          snprintf(g_dir, sizeof(g_dir), "%s", d);
    if (const char *c = getenv("LITEX_PC_PROF_CHECKPOINT_S")) g_ckpt_period_s = atof(c);
    static const char *defaults =
        "SELFTEST DONE,GEN:,GEN DONE,GEN-HW:,GEN-HW DONE,TRIXIUM-READY";
    const char *ms = getenv("LITEX_PC_PROF_MARKERS");
    char *buf = strdup(ms ? ms : defaults);   // intentionally leaked: lives for the run
    for (char *t = strtok(buf, ","); t && g_nmarkers < 32; t = strtok(nullptr, ","))
        g_markers[g_nmarkers++] = t;
    if (!alloc_region(g_rom) || !alloc_region(g_ram)) {
        fprintf(stderr, "[pcprof] DISABLED: cannot allocate histogram\n");
        g_nmarkers = 0; g_rom.cyc = g_ram.cyc = nullptr;
        return;
    }
    g_last_ckpt_s = wall_s();
    // Covers SIGTERM/SIGINT/$finish, which exit through main(). It does NOT cover SIGKILL -
    // test/test_sim_smoke.py uses p.kill() - which is why the marker snapshots above are the
    // primary mechanism rather than a convenience.
    {
        char path[640];
        snprintf(path, sizeof(path), "%s/trace_events.bin", g_dir);
        g_evf = fopen(path, "wb");
    }
    atexit(dump_final);
    fprintf(stderr, "[pcprof] armed: %zu markers, dir=%s\n", g_nmarkers, g_dir);
}

extern "C" void litex_sim_user_post_eval(void *vsim, uint64_t time_ps) {
    Vsim *sim = static_cast<Vsim *>(vsim);
    const uint8_t clk = sim->sys_clk;
    // post_eval with clk high: the writeback flops hold the instruction now in the last stage
    // and lastStageIsFiring is the settled verdict on whether it retires this cycle.
    if (clk && !g_prev_clk && g_ram.cyc) {
        g_time_ps = time_ps;
        sample(sim->sim->VexRiscv);
        const uint8_t busy = (uint8_t)((sim->prof_eng_busy) | (sim->prof_vec_busy<<1)
                                       | (sim->prof_seq_busy<<2));
        if (busy != g_busy_prev) { event((uint32_t)g_cycles, 0, busy); g_busy_prev = busy; }
        uart_sniff(sim);
        if (g_ckpt_period_s > 0.0 && (g_cycles & 0xFFFFF) == 0) {
            const double now = wall_s();
            if (now - g_last_ckpt_s >= g_ckpt_period_s) {
                g_last_ckpt_s = now;
                const unsigned keep = g_snap; dump("checkpoint"); g_snap = keep;
            }
        }
    }
    g_prev_clk = clk;
}

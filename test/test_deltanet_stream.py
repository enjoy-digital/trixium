# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiDeltaNetStream against the firmware's recurrence loop, compiled (Verilator).

Every lane's state stream is fed with random gaps, every output stream drained with
random back-pressure, the scratchpad write grant is random: the state, the outputs and the
cycle count come back from the compiled netlist and must equal the float32 reference bit
for bit.
"""
import shutil
import subprocess

import numpy as np
import pytest
from migen.fhdl import verilog

from trixium.gateware.deltanet_stream import (
    BonsaiDeltaNetStream, lanes_to_state, reference, state_to_lanes)

f32 = np.float32

_CPP = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "V{top}.h"
static void require(bool ok, const char *what) {{ if (!ok) {{ std::fprintf(stderr, "%s\n", what); std::exit(1); }} }}
static std::vector<uint32_t> load(const char *path) {{
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {{}});
    std::vector<uint32_t> w(b.size()/4); std::memcpy(w.data(), b.data(), b.size()); return w;
}}
int main(int argc, char **argv) {{
    require(argc == 10, "usage: sp lanes sp_out lanes_out heads table in_pct out_pct grant_pct");
    std::vector<uint32_t> mem = load(argv[1]), in = load(argv[2]);
    const uint32_t heads = std::atoi(argv[5]), table = std::atoi(argv[6]);
    const uint32_t in_pct = std::atoi(argv[7]), out_pct = std::atoi(argv[8]), grant_pct = std::atoi(argv[9]);
    const uint32_t L = {lanes}, per = in.size()/L/8, beats = mem.size()/8;
    V{top} d;
    CData *sv[L] = {{ {sink_valid} }}; CData *sr[L] = {{ {sink_ready} }};
    EData *sd[L] = {{ {sink_data} }};
    CData *ov[L] = {{ {src_valid} }}; CData *orr[L] = {{ {src_ready} }};
    EData *od[L] = {{ {src_data} }};
    std::vector<uint32_t> out(in.size(), 0xDEADBEEFu);
    std::vector<uint32_t> ip(L, 0), op(L, 0), took(L, 0);
    uint32_t state = 12345; auto rnd = [&]() {{ state = state*1664525u + 1013904223u; return state >> 8; }};
    uint32_t pending[8] = {{0}};
    uint64_t cycle = 0;
    auto tick = [&]() {{
        for (uint32_t l = 0; l < L; l++) {{
            if (!*sv[l] && ip[l] < per && (rnd() % 100) < in_pct) *sv[l] = 1;
            if (*sv[l]) for (int w = 0; w < 8; w++) sd[l][w] = in[(size_t)l*per*8 + (size_t)ip[l]*8 + w];
            *orr[l] = (rnd() % 100) < out_pct;
        }}
        d.sp_wr_gnt = (rnd() % 100) < grant_pct;
        d.sys_clk = 0; d.eval();
        for (uint32_t l = 0; l < L; l++) {{
            took[l] = *sv[l] && *sr[l];
            if (took[l]) ip[l]++;
            if (*ov[l] && *orr[l]) {{
                require(op[l] < per, "too many state beats");
                for (int w = 0; w < 8; w++) out[(size_t)l*per*8 + (size_t)op[l]*8 + w] = od[l][w];
                op[l]++;
            }}
        }}
        if (d.sp_wr_req && d.sp_wr_gnt) {{
            require(d.sp_wr_adr < beats, "write out of range");
            for (int w = 0; w < 8; w++) if (d.sp_wr_we >> w & 1) mem[8*d.sp_wr_adr + w] = d.sp_wr_dat[w];
        }}
        bool rd = d.sp_rd_req;
        if (rd) {{ require(d.sp_rd_adr < beats, "read out of range"); for (int w = 0; w < 8; w++) pending[w] = mem[8*d.sp_rd_adr + w]; }}
        d.sys_clk = 1; d.eval();
        for (uint32_t l = 0; l < L; l++) if (took[l]) *sv[l] = 0;
        for (int w = 0; w < 8; w++) d.sp_rd_dat[w] = rd ? pending[w] : 0xA5A5A5A5u;
        cycle++;
    }};
    d.sys_rst = 1; tick(); tick(); d.sys_rst = 0;
    d.heads = heads; d.table_adr = table; d.start = 1; tick(); d.start = 0;
    const uint64_t t0 = cycle;
    bool done = false;
    for (uint64_t n = 0; n < 50000000 && !done; n++) {{ tick(); done = d.done; }}
    if (!done) {{ std::fprintf(stderr, "timeout: progress {progress} in %u/%u out %u/%u ready %u valid %u\n", {progress_args}, ip[0], ip[L-1], op[0], op[L-1], *sr[0], *sv[0]); std::exit(1); }}
    tick(); require(!d.busy, "busy after done");
    for (uint32_t l = 0; l < L; l++) require(ip[l] == per && op[l] == per, "stream lengths");
    std::printf("cycles %llu\n", (unsigned long long)(cycle - t0));
    std::ofstream o1(argv[3], std::ios::binary); o1.write(reinterpret_cast<const char *>(mem.data()), 4*mem.size());
    std::ofstream o2(argv[4], std::ios::binary); o2.write(reinterpret_cast<const char *>(out.data()), 4*out.size());
    return 0;
}}
'''


def build(tmp_path, lanes, rows, cols):
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.skip("Verilator required")
    dut = BonsaiDeltaNetStream(lanes=lanes, rows=rows, cols=cols)
    sig = {n: getattr(dut, n) for n in ("start", "heads", "table", "busy", "done", "sp_rd_req",
           "sp_rd_adr", "sp_rd_dat", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat", "sp_wr_gnt")}
    for l in range(lanes):
        for kind, ep in (("sink", dut.sinks[l]), ("source", dut.sources[l])):
            for f in ("valid", "ready", "data"):
                sig[f"{kind}{l}_{f}"] = getattr(ep, f)
    for i, s in enumerate(dut.progress):
        sig[f"progress{i}"] = s
    for n, s in sig.items():
        s.name_override = n if n != "table" else "table_adr"
    top = f"dns_{lanes}_{rows}_{cols}"
    rtl = tmp_path / f"{top}.v"
    verilog.convert(dut, ios=set(sig.values()) | {dut.sp_rd_ok}, name=top).write(str(rtl))
    j = lambda fmt: ", ".join(fmt.format(l) for l in range(lanes))
    cpp = tmp_path / f"{top}.cpp"
    cpp.write_text(_CPP.format(top=top, lanes=lanes,
        sink_valid=j("&d.sink{}_valid"), sink_ready=j("&d.sink{}_ready"), sink_data=j("&d.sink{}_data[0]"),
        src_valid=j("&d.source{}_valid"), src_ready=j("&d.source{}_ready"), src_data=j("&d.source{}_data[0]"),
        progress=" ".join(["%u"]*len(dut.progress)),
        progress_args=", ".join(f"(unsigned)d.progress{i}" for i in range(len(dut.progress)))))
    # sp_rd_ok is an input left at its reset value 1 by the driver (Verilator zero-inits, so
    # the harness drives it explicitly below).
    text = cpp.read_text().replace("d.sys_rst = 1;", "d.sp_rd_ok = 1; d.sys_rst = 1;", 1)
    cpp.write_text(text)
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "8", "-Wno-fatal", "-O3",
                            "--x-assign", "fast", "--x-initial", "fast",
                            "--Mdir", str(tmp_path / "obj"), "--top-module", top, str(rtl), str(cpp)],
                           capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
    return tmp_path / f"obj/V{top}"


def scenario(rng, heads, rows, cols, nk):
    """A layer's worth of heads: GQA q/k shared by head pairs, a few special values."""
    SP = 256*1024
    mem = rng.integers(0, 1 << 32, SP//4, dtype=np.uint64).astype(np.uint32)
    put = lambda a, v: mem.__setitem__(slice(a//4, a//4 + len(v)), np.asarray(v, f32).view(np.uint32))
    qk_base, v_base, out_base, table = 0x00000, 0x10000, 0x20000, 0x30000
    states = rng.normal(0, 0.5, (heads, rows, cols)).astype(f32)
    states[0, 3, :] = 0                                    # a zero row
    states[1, 5, :7] = [1e-40, -1e-41, 0, -0.0, 3e-39, 1e30, -1e30]
    q = rng.normal(0, 1, (nk, cols)).astype(f32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    k = rng.normal(0, 1, (nk, cols)).astype(f32); k /= np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.normal(0, 1, (heads, rows)).astype(f32)
    decay = rng.uniform(0.3, 1.0, heads).astype(f32)
    beta = rng.uniform(0.0, 1.0, heads).astype(f32)
    scale = f32(1/np.sqrt(cols))
    for kh in range(nk):
        put(qk_base + 8*cols*kh, q[kh]); put(qk_base + 8*cols*kh + 4*cols, k[kh])
    recs = []
    for h in range(heads):
        kh = h % nk
        put(v_base + 4*rows*h, v[h])
        rec = np.array([qk_base + 8*cols*kh, qk_base + 8*cols*kh + 4*cols, v_base + 4*rows*h,
                        out_base + 4*rows*h, 0, 0, 0, 0], np.uint32)
        rec[4:7] = np.array([decay[h], beta[h], scale], f32).view(np.uint32)
        recs.append(rec)
    mem[table//4:table//4 + 8*heads] = np.concatenate(recs)
    want_mem, want_states = mem.copy(), np.empty_like(states)
    for h in range(heads):
        m, o = reference(states[h], q[h % nk], k[h % nk], v[h], decay[h], beta[h], scale)
        want_states[h] = m
        want_mem[(out_base + 4*rows*h)//4:(out_base + 4*rows*(h + 1))//4] = o.view(np.uint32)
    return mem, states, table, want_mem, want_states


def run(binary, tmp_path, mem, states, lanes, heads, table, pcts):
    sp, ln, spo, lno = (tmp_path / n for n in ("sp.bin", "lanes.bin", "sp_out.bin", "lanes_out.bin"))
    mem.astype("<u4").tofile(sp)
    state_to_lanes(states, lanes).astype("<u4").tofile(ln)
    r = subprocess.run([str(binary), str(sp), str(ln), str(spo), str(lno), str(heads), str(table//32),
                        *[str(p) for p in pcts]], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr[-4000:]
    cycles = int(r.stdout.split("cycles ")[1].split()[0])
    return np.fromfile(spo, dtype="<u4"), np.fromfile(lno, dtype="<u4"), cycles


def check(got_mem, got_lanes, want_mem, want_states, lanes, rows, cols):
    got_states = lanes_to_state(got_lanes, lanes, rows, cols).view(np.uint32)
    want = want_states.view(np.uint32)
    bad = np.argwhere(got_states != want)
    assert not len(bad), [(tuple(i), hex(int(got_states[tuple(i)])), hex(int(want[tuple(i)]))) for i in bad[:8]]
    bad = np.nonzero(got_mem != want_mem)[0]
    assert not len(bad), [(hex(4*i), hex(int(got_mem[i])), hex(int(want_mem[i]))) for i in bad[:8]]


@pytest.mark.parametrize("pcts", [(100, 100, 100), (40, 60, 50)])
def test_small_geometry(tmp_path_factory, pcts):
    tmp = tmp_path_factory.mktemp("dns_small")
    lanes, rows, cols, heads = 2, 16, 16, 9
    binary = build(tmp, lanes, rows, cols)
    mem, states, table, want_mem, want_states = scenario(np.random.default_rng(3), heads, rows, cols, 3)
    got_mem, got_lanes, cycles = run(binary, tmp, mem, states, lanes, heads, table, pcts)
    check(got_mem, got_lanes, want_mem, want_states, lanes, rows, cols)


@pytest.fixture(scope="module")
def binary27(tmp_path_factory):
    return build(tmp_path_factory.mktemp("dns27"), 16, 128, 128)


def test_27b_geometry_streams_a_head_per_1024_clocks(tmp_path, binary27):
    lanes, rows, cols = 16, 128, 128
    cycles = {}
    for heads in (6, 12):
        mem, states, table, want_mem, want_states = scenario(np.random.default_rng(heads), heads, rows, cols, 2)
        got_mem, got_lanes, cycles[heads] = run(binary27, tmp_path, mem, states, lanes, heads, table,
                                                (100, 100, 100))
        check(got_mem, got_lanes, want_mem, want_states, lanes, rows, cols)
    # P1 and P2 overlap: one head per cols*R clocks once the pipe is full, plus one head of
    # fill (the last P2) and the load/delta/scale latencies.
    per_head = (cycles[12] - cycles[6])/6
    print("27B geometry:", cycles, "cycles;", per_head, "per head in steady state")
    assert per_head < 1024*1.02, cycles
    assert cycles[6] < 7*1024 + 800, cycles


def test_27b_geometry_under_backpressure(tmp_path, binary27):
    lanes, rows, cols, heads = 16, 128, 128, 5
    mem, states, table, want_mem, want_states = scenario(np.random.default_rng(8), heads, rows, cols, 5)
    got_mem, got_lanes, cycles = run(binary27, tmp_path, mem, states, lanes, heads, table, (30, 50, 40))
    check(got_mem, got_lanes, want_mem, want_states, lanes, rows, cols)

# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiAttnStream against hybrid.c's attention loops, compiled (Verilator).

The K/V rows stream in with random gaps, the scratchpad read and write grants are random;
scores and outputs must equal the float32 reference of the firmware loops bit for bit.
"""
import shutil
import subprocess

import numpy as np
import pytest
from migen.fhdl import verilog

from trixium.gateware.attn_stream import (
    BonsaiAttnStream, MODE_SCORES, MODE_VALUES, scores_reference, values_reference)

f32 = np.float32
SP = 256*1024

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
    require(argc == 8, "usage: sp rows sp_out rec in_pct grant_pct want_error");
    std::vector<uint32_t> mem = load(argv[1]), in = load(argv[2]);
    const uint32_t rec = std::atoi(argv[4]), in_pct = std::atoi(argv[5]), grant_pct = std::atoi(argv[6]);
    const bool want_error = std::atoi(argv[7]);
    const uint32_t beats = mem.size()/8, n = in.size()/8;
    V{top} d;
    uint32_t state = 4242; auto rnd = [&]() {{ state = state*1664525u + 1013904223u; return state >> 8; }};
    uint32_t pending[8] = {{0}}, ip = 0;
    uint64_t cycle = 0;
    auto tick = [&]() {{
        if (!d.sink_valid && ip < n && (rnd() % 100) < in_pct) d.sink_valid = 1;
        if (d.sink_valid) for (int w = 0; w < 8; w++) d.sink_data[w] = in[8*ip + w];
        d.sp_wr_gnt = (rnd() % 100) < grant_pct;
        d.sp_rd_ok = (rnd() % 100) < grant_pct;
        d.sys_clk = 0; d.eval();
        bool took = d.sink_valid && d.sink_ready;
        if (d.sp_wr_req && d.sp_wr_gnt) {{
            require(d.sp_wr_adr < beats, "write out of range");
            for (int w = 0; w < 8; w++) if (d.sp_wr_we >> w & 1) mem[8*d.sp_wr_adr + w] = d.sp_wr_dat[w];
        }}
        bool rd = d.sp_rd_req && d.sp_rd_ok;
        if (rd) {{ require(d.sp_rd_adr < beats, "read out of range"); for (int w = 0; w < 8; w++) pending[w] = mem[8*d.sp_rd_adr + w]; }}
        d.sys_clk = 1; d.eval();
        if (took) {{ ip++; d.sink_valid = 0; }}
        for (int w = 0; w < 8; w++) d.sp_rd_dat[w] = rd ? pending[w] : 0xA5A5A5A5u;
        cycle++;
    }};
    d.sys_rst = 1; tick(); tick(); d.sys_rst = 0;
    d.rec = rec; d.start = 1; tick(); d.start = 0;
    const uint64_t t0 = cycle;
    bool done = false;
    for (uint64_t k = 0; k < 20000000 && !done; k++) {{
        tick();
        done = d.done;
        if (want_error && d.error) break;
    }}
    if (want_error) {{ require(d.error && !done, "expected a geometry error"); tick(); require(!d.busy, "busy after error"); std::printf("error ok\n"); return 0; }}
    if (!done) {{ std::fprintf(stderr, "timeout: progress {progress} in %u/%u\n", {progress_args}, ip, n); std::exit(1); }}
    tick(); require(!d.busy, "busy after done");
    require(ip == n, "rows not all consumed");
    std::printf("cycles %llu\n", (unsigned long long)(cycle - t0));
    std::ofstream o(argv[3], std::ios::binary); o.write(reinterpret_cast<const char *>(mem.data()), 4*mem.size());
    return 0;
}}
'''


def build(tmp_path, **geometry):
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.skip("Verilator required")
    dut = BonsaiAttnStream(**geometry)
    sig = {n: getattr(dut, n) for n in ("start", "rec", "busy", "done", "error", "sp_rd_ok", "sp_rd_req",
           "sp_rd_adr", "sp_rd_dat", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat", "sp_wr_gnt")}
    for f in ("valid", "ready", "data"):
        sig["sink_" + f] = getattr(dut.sink, f)
    for i, s in enumerate(dut.progress):
        sig[f"progress{i}"] = s
    for n, s in sig.items():
        s.name_override = n
    top = "attn_" + "_".join(str(v) for v in geometry.values())
    rtl = tmp_path / f"{top}.v"
    verilog.convert(dut, ios=set(sig.values()), name=top).write(str(rtl))
    cpp = tmp_path / f"{top}.cpp"
    cpp.write_text(_CPP.format(top=top, progress=" ".join(["%u"]*len(dut.progress)),
        progress_args=", ".join(f"(unsigned)d.progress{i}" for i in range(len(dut.progress)))))
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "8", "-Wno-fatal", "-O3",
                            "--x-assign", "fast", "--x-initial", "fast",
                            "--Mdir", str(tmp_path / "obj"), "--top-module", top, str(rtl), str(cpp)],
                           capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
    return tmp_path / f"obj/V{top}"


def layout(rng, nq, n_rep, nkv, hd, T, mode, specials=True):
    """Scratchpad image, row stream and expected image for one command."""
    mem = rng.integers(0, 1 << 32, SP//4, dtype=np.uint64).astype(np.uint32)
    put = lambda a, v: mem.__setitem__(slice(a//4, a//4 + len(v)), np.asarray(v, f32).view(np.uint32))
    rows = rng.normal(0, 1, (T, nkv, hd)).astype(f32)
    if specials:
        rows[0, 0, :4] = [0, -0.0, 1e-40, 3e38]
    SRC, DST, REC = 0x00000, 0x20000, 0x3FF00
    tb = (T + 7)//8
    if mode == MODE_SCORES:
        q = rng.normal(0, 1, (nq, hd)).astype(f32)
        sstr, dstr = 4*hd + 64, 32*tb + 32         # strided lanes, with gaps
        for l in range(nq):
            put(SRC + l*sstr, q[l])
        scale = f32(1/np.sqrt(hd))
        want = scores_reference(q, rows, scale)
    else:
        p = rng.uniform(0, 1, (nq, T)).astype(f32)
        p[0, :min(2, T)] = [0, 1e-39][:T]
        sstr, dstr = 32*tb + 96, 4*hd + 32
        for l in range(nq):
            put(SRC + l*sstr, p[l])
        scale = f32(0)
        want = values_reference(p, rows)
    rec = np.array([mode | nq << 8 | n_rep << 16 | nkv << 24, T, hd, SRC, sstr, DST, dstr, 0], np.uint32)
    rec[7] = np.array([scale], f32).view(np.uint32)[0]
    mem[REC//4:REC//4 + 8] = rec
    expect = mem.copy()
    for l in range(nq):
        if mode == MODE_SCORES:
            base = (DST + l*dstr)//4
            got_slice = slice(base, base + T)              # padding positions are junk
            expect[got_slice] = want[l].view(np.uint32)
        else:
            base = (DST + l*dstr)//4
            expect[base:base + hd] = want[l].view(np.uint32)
    # Padding of the last score beat is junk: mark it "don't care".
    care = np.ones(SP//4, bool)
    if mode == MODE_SCORES:
        for l in range(nq):
            base = (DST + l*dstr)//4
            care[base + T:base + 8*tb] = False
    return mem, rows.reshape(-1).view(np.uint32), REC//32, expect, care


def run(binary, tmp_path, mem, rows, rec, pcts=(100, 100), want_error=0):
    sp, rw, out = tmp_path / "sp.bin", tmp_path / "rows.bin", tmp_path / "out.bin"
    mem.astype("<u4").tofile(sp)
    rows.astype("<u4").tofile(rw)
    r = subprocess.run([str(binary), str(sp), str(rw), str(out), str(rec), *[str(p) for p in pcts],
                        str(want_error)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr[-4000:]
    if want_error:
        return None, None
    return np.fromfile(out, dtype="<u4"), int(r.stdout.split("cycles ")[1].split()[0])


def check(got, expect, care):
    bad = np.nonzero((got != expect) & care)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(expect[i]))) for i in bad[:8]]


SMALL = dict(lanes=4, groups=2, hb_max=4, t_max=64)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return build(tmp_path_factory.mktemp("attn_small"), **SMALL)


@pytest.mark.parametrize("mode", [MODE_SCORES, MODE_VALUES])
@pytest.mark.parametrize("nq,n_rep,nkv,hd,T,pcts", [
    (4, 2, 2, 32, 13, (100, 100)),
    (4, 2, 2, 16, 1, (50, 40)),
    (3, 3, 1, 8, 64, (30, 60)),
    (4, 4, 1, 32, 20, (70, 30)),
])
def test_small_geometry(tmp_path, small, mode, nq, n_rep, nkv, hd, T, pcts):
    mem, rows, rec, expect, care = layout(np.random.default_rng(T*7 + hd), nq, n_rep, nkv, hd, T, mode)
    got, cycles = run(small, tmp_path, mem, rows, rec, pcts)
    check(got, expect, care)


def test_bad_geometry_is_an_error(tmp_path, small):
    mem, rows, rec, expect, care = layout(np.random.default_rng(1), 4, 2, 2, 32, 13, MODE_SCORES)
    mem[rec*8 + 2] = 48                                     # hd not a power of two
    run(small, tmp_path, mem, rows[:0], rec, want_error=1)


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    return build(tmp_path_factory.mktemp("attn_27b"), lanes=24, groups=4, hb_max=32, t_max=2048)


@pytest.mark.parametrize("mode", [MODE_SCORES, MODE_VALUES])
def test_27b_geometry_and_rate(tmp_path, big, mode):
    T = 37
    mem, rows, rec, expect, care = layout(np.random.default_rng(5 + mode), 24, 6, 4, 256, T, mode)
    got, cycles = run(big, tmp_path, mem, rows, rec)
    check(got, expect, care)
    # One multiply-add per lane per clock: scores 8*hd per 8 positions, values hd per one.
    compute = (T + 7)//8*8*256 if mode == MODE_SCORES else T*256
    print("27B geometry, mode", mode, ":", cycles, "cycles for T =", T, "(compute", compute, ")")
    assert cycles < compute*1.1 + 1200, cycles


def test_tiny_geometry_on_the_27b_build(tmp_path, big):
    for mode in (MODE_SCORES, MODE_VALUES):
        mem, rows, rec, expect, care = layout(np.random.default_rng(11 + mode), 8, 4, 2, 128, 9, mode)
        got, cycles = run(big, tmp_path, mem, rows, rec, (60, 70))
        check(got, expect, care)

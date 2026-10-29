#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/attention_i8.py: AttentionStreamI8, compiled, against its integer references.

Each run is a scores command over K rows followed by a values command over V rows (the values
pass takes its V exponents from the scores pass before it). Rows stream in with random gaps, the
scratchpad grants are random; scores and outputs must equal the references bit for bit."""

import subprocess

import numpy as np
import pytest

from gateware.units.attention_i8 import (AttentionStreamI8, MODE_SCORES, MODE_VALUES,
    quantize_rows, scores_reference, values_reference, k_rows, v_rows, p_exponent)

from common.verilator import Simulator, file_key

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
// usage: sp rows out in_pct grant_pct want_error rec0 beats0 [rec1 beats1 ...]
int main(int argc, char **argv) {{
    require(argc >= 9 && (argc - 7) % 2 == 0, "usage");
    std::vector<uint32_t> mem = load(argv[1]), in = load(argv[2]);
    const uint32_t in_pct = std::atoi(argv[4]), grant_pct = std::atoi(argv[5]);
    const bool want_error = std::atoi(argv[6]);
    const uint32_t beats = mem.size()/8;
    V{top} d;
    uint32_t state = 4242; auto rnd = [&]() {{ state = state*1664525u + 1013904223u; return state >> 8; }};
    uint32_t pending[8] = {{0}}, ip = 0, lim = 0;
    uint64_t cycle = 0;
    auto tick = [&]() {{
        if (!d.sink_valid && ip < lim && (rnd() % 100) < in_pct) d.sink_valid = 1;
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
    for (int c = 7; c < argc; c += 2) {{
        lim += std::atoi(argv[c + 1]);
        d.rec = std::atoi(argv[c]); d.start = 1; tick(); d.start = 0;
        const uint64_t t0 = cycle;
        bool done = false;
        for (uint64_t k = 0; k < 20000000 && !done; k++) {{
            tick();
            done = d.done;
            if (want_error && d.error) break;
        }}
        if (want_error) {{ require(d.error && !done, "expected a geometry error"); tick(); require(!d.busy, "busy after error"); std::printf("error ok\n"); return 0; }}
        if (!done) {{ std::fprintf(stderr, "timeout: progress {progress} in %u/%u\n", {progress_args}, ip, lim); std::exit(1); }}
        tick(); require(!d.busy, "busy after done");
        require(ip == lim, "rows not all consumed");
        std::printf("cycles %llu\n", (unsigned long long)(cycle - t0));
    }}
    std::ofstream o(argv[3], std::ios::binary); o.write(reinterpret_cast<const char *>(mem.data()), 4*mem.size());
    return 0;
}}
'''


def build(**geometry):
    top = "atti_" + "_".join(str(v) for v in geometry.values())
    return Simulator(lambda: _construct(**geometry), name=top, key=file_key(__file__, top, geometry),
                     driver=lambda built: _driver(top, built["dut"])).binary


def _construct(**geometry):
    dut = AttentionStreamI8(**geometry)
    sig = {n: getattr(dut, n) for n in ("start", "rec", "busy", "done", "error", "sp_rd_ok", "sp_rd_req",
           "sp_rd_adr", "sp_rd_dat", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat", "sp_wr_gnt")}
    for f in ("valid", "ready", "data"):
        sig["sink_" + f] = getattr(dut.sink, f)
    for i, s in enumerate(dut.progress):
        sig[f"progress{i}"] = s
    for n, s in sig.items():
        s.name_override = n
    return dict(dut=dut, ios=sig)


def _driver(top, dut):
    return _CPP.format(top=top, progress=" ".join(["%u"]*len(dut.progress)),
        progress_args=", ".join(f"(unsigned)d.progress{i}" for i in range(len(dut.progress))))


def layout(rng, nq, n_rep, nkv, hd, T, kv0=0, ev_spread=3):
    """Scratchpad image, rows and expected image for a scores + values pair (kv heads from kv0)."""
    mem = rng.integers(0, 1 << 32, SP//4, dtype=np.uint64).astype(np.uint32)
    put = lambda a, v: mem.__setitem__(slice(a//4, a//4 + len(v)), np.asarray(v).view(np.uint32))
    # K and V with per-position magnitudes, so the exponents differ along the rows.
    K = (rng.normal(0, 1, (T, nkv, hd))*np.exp2(rng.integers(-ev_spread, ev_spread + 1, (T, nkv, 1)))).astype(f32)
    V = (rng.normal(0, 1, (T, nkv, hd))*np.exp2(rng.integers(-ev_spread, ev_spread + 1, (T, nkv, 1)))).astype(f32)
    K[0, 0, :4] = [0, -0.0, 1e-30, 3.0]
    K8, ek = quantize_rows(K); V8, ev = quantize_rows(V)
    q = rng.normal(0, 1, (nq, hd)).astype(f32)
    q8, eq = quantize_rows(q)
    tb = (T + 7)//8
    QSRC, PSRC, SDST, ODST, REC = 0x00000, 0x08000, 0x20000, 0x30000, 0x3FF00
    hb = hd//32
    qstr, sstr_s, pstr, odstr = 32*(hb + 1) + 64, 32*tb + 32, 32*tb + 96, 4*hd + 32
    for l in range(nq):
        put(QSRC + l*qstr, q8[l].view(np.uint32))
        put(QSRC + l*qstr + 32*hb, np.array([eq[l], 0, 0, 0, 0, 0, 0, 0], np.int32))
    # Probabilities: a spread of magnitudes, zeros and a subnormal, junk in the padding.
    p = (rng.uniform(0, 1, (nq, T))*np.exp2(rng.integers(-12, 1, (nq, T)))).astype(f32)
    p[0, :min(2, T)] = np.array([0, 1e-39], f32)[:min(2, T)]
    for l in range(nq):
        row = np.frombuffer(rng.bytes(4*8*tb), np.uint32).copy()
        row[:T] = p[l].view(np.uint32)
        put(PSRC + l*pstr, row)
    scale = f32(1/np.sqrt(hd))
    sl = slice(kv0, kv0 + -(-nq//n_rep))
    want_s = scores_reference(q8, eq, K8[:, sl], ek[:, sl], scale)
    want_o = values_reference(p, V8[:, sl], ev[:, sl])
    w0 = kv0 << 1 | nq << 8 | n_rep << 16 | nkv << 24
    recs = [np.array([w0 | MODE_SCORES, T, hd, QSRC, qstr, SDST, sstr_s, scale.view(np.uint32)], np.uint32),
            np.array([w0 | MODE_VALUES, T, hd, PSRC, pstr, ODST, odstr, 0], np.uint32)]
    for i, r in enumerate(recs):
        mem[REC//4 + 8*i:REC//4 + 8*i + 8] = r
    expect, care = mem.copy(), np.ones(SP//4, bool)
    for l in range(nq):
        b = (SDST + l*sstr_s)//4
        expect[b:b + T] = want_s[l].view(np.uint32)
        care[b + T:b + 8*tb] = False                   # padding positions: junk
        b = (ODST + l*odstr)//4
        expect[b:b + hd] = want_o[l].view(np.uint32)
    rows = np.concatenate([k_rows(K8, ek, ev).reshape(-1), v_rows(V8).reshape(-1)])
    rb = nkv*hb + 1
    return mem, rows, [(REC//32, T*rb), (REC//32 + 1, T*rb)], expect, care


def run(binary, tmp_path, mem, rows, cmds, pcts=(100, 100), want_error=0):
    sp, rw, out = tmp_path / "sp.bin", tmp_path / "rows.bin", tmp_path / "out.bin"
    mem.astype("<u4").tofile(sp)
    rows.astype("<u4").tofile(rw)
    args = [str(binary), str(sp), str(rw), str(out), *[str(p) for p in pcts], str(want_error)]
    for rec, n in cmds:
        args += [str(rec), str(n)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr[-4000:]
    if want_error:
        return None, None
    return np.fromfile(out, dtype="<u4"), [int(x.split()[0]) for x in r.stdout.split("cycles ")[1:]]


def check(got, expect, care):
    bad = np.nonzero((got != expect) & care)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(expect[i]))) for i in bad[:8]]


SMALL = dict(lanes=4, groups=2, hd_max=128, t_max=64)


@pytest.fixture(scope="module")
def small():
    return build(**SMALL)


@pytest.mark.parametrize("nq,n_rep,nkv,hd,T,pcts", [
    (4, 2, 2, 64, 13, (100, 100)),
    (4, 2, 2, 128, 1, (50, 40)),
    (3, 3, 1, 32, 64, (30, 60)),
    (4, 4, 1, 64, 20, (70, 30)),
    (4, 2, 4, 128, 37, (60, 80)),
])
def test_small_geometry(tmp_path, small, nq, n_rep, nkv, hd, T, pcts):
    mem, rows, cmds, expect, care = layout(np.random.default_rng(T*7 + hd), nq, n_rep, nkv, hd, T)
    got, cycles = run(small, tmp_path, mem, rows, cmds, pcts)
    check(got, expect, care)


def test_kv_slice(tmp_path, small):
    """A command over kv heads 2.. of a 4-head row (the buffers keep only the slice)."""
    mem, rows, cmds, expect, care = layout(np.random.default_rng(5), 4, 2, 4, 64, 24, kv0=2)
    got, _ = run(small, tmp_path, mem, rows, cmds, (80, 70))
    check(got, expect, care)


def test_geometry_error(tmp_path, small):
    mem, rows, cmds, _, _ = layout(np.random.default_rng(1), 4, 2, 2, 64, 8)
    mem[0x3FF00//4 + 2] = 48                            # hd not a power of two
    run(small, tmp_path, mem, rows, cmds[:1], want_error=1)


def test_throughput(tmp_path):
    """The Ti375 geometry at full rate: hd/8 clocks per position and pass (+ the drain)."""
    dut = build(lanes=8, groups=2, hd_max=256, t_max=256)
    mem, rows, cmds, expect, care = layout(np.random.default_rng(9), 8, 4, 2, 256, 256)
    got, cycles = run(dut, tmp_path, mem, rows, cmds)
    check(got, expect, care)
    assert cycles[0] < 256*32*1.15 + 600, cycles            # scores: 32 clocks per position
    assert cycles[1] < 256*32*1.15 + 8*256 + 600, cycles    # values + the serial drain

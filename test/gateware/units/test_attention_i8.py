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


# The firmware's list through the shared cluster ------------------------------------------------
# Scores (engine op 6), the softmax (unit 14) in place, values (op 6): the int8 rows in the
# channel models, position t on channel t % n, as hybrid.c's attention list runs them.

def _desc_engine(kind, rows=0, rem=0, wbase=0, wlen=0, cbeats=0, xsrc=0):
    return [1, kind, rows | rem << 16, 0, wbase, wlen, cbeats << 8, 0, xsrc] + [0]*7

def _desc_softmax(src, n, rows, stride):
    return [14, src, n | rows << 16, stride] + [0]*12

def _desc_stop():
    return [0]*16

@pytest.mark.parametrize("T,hd,kit", [(2, 64, 0), (3, 64, 0), (9, 64, 0), (2, 256, 0), (3, 256, 0),
                                      (2, 256, 1)])
def test_scores_softmax_values_list(T, hd, kit):
    """kit: the Ti375 firmware's addresses (256 KB scratchpad, the list at SP_DESC 0x3D000, the
    scores at RS_F 0x1C800)."""
    from gateware.core.core import TriXiumConfig
    from common.cluster import SoC, install_list, pack
    from common.cluster import run as run_cluster
    import test_softmax as tsm
    n, nq, n_rep, nkv = 2, 8, 4, 2
    hb, rb = hd//32, nkv*hd//32 + 1
    config = TriXiumConfig(with_vecunit=False, with_softmax=True, scratchpad_size=(256 if kit else 64)*1024,
        with_attnstream=True, attnstream_options=(("lanes", 8), ("groups", 2), ("hb_max", hd//8),
                                                  ("t_max", 64), ("int8", True)))
    dut = SoC(config, n=n, backend="verilator")
    rng = np.random.default_rng(T)
    K = (rng.normal(0, 2, (T, nkv, hd))).astype(f32)
    V = (rng.normal(0, 1, (T, nkv, hd))*np.exp2(rng.integers(-2, 3, (T, nkv, 1)))).astype(f32)
    q = rng.normal(0, 1, (nq, hd)).astype(f32)
    K8, ek = quantize_rows(K); V8, ev = quantize_rows(V); q8, eq = quantize_rows(q)
    scale = f32(1/np.sqrt(hd))
    s = scores_reference(q8, eq, K8, ek, scale)
    p = np.stack([tsm.softmax_row(s[l]) for l in range(nq)])
    o = values_reference(p, V8, ev)
    Tp = (T + 7)//8*8
    QSRC, SP, ODST, REC, KB, VB = 0x1000, 0x4000, 0x5000, 0x3800, 0x4000, 0x8000
    LIST = 0
    if kit:
        QSRC, SP, ODST, REC, LIST = 0xC800, 0x1C800, 0x12C00, 0x12800, 0x3D000
    qop = 32*(hb + 1)
    words = {}
    def put(addr, u32):
        u32 = np.asarray(u32, np.uint32)
        for i in range(0, len(u32), 8):
            words[addr//32 + i//8] = pack(list(u32[i:i + 8]) + [0]*(8 - len(u32[i:i + 8])))
    for l in range(nq):
        put(QSRC + l*qop, q8[l].view(np.uint32))
        put(QSRC + l*qop + 32*hb, [int(eq[l]) & 0xFFFFFFFF])
    for l in range(nq):                                  # junk in the score rows
        put(SP + l*Tp*4, rng.integers(0, 1 << 32, Tp, dtype=np.uint64).astype(np.uint32))
    w0 = nq << 8 | n_rep << 16 | nkv << 24
    put(REC, [w0, T, hd, QSRC, qop, SP, Tp*4, int(scale.view(np.uint32))])
    put(REC + 32, [w0 | 1, T, hd, SP, Tp*4, ODST, hd*4, 0])
    krows, vrows = k_rows(K8, ek, ev), v_rows(V8)
    lst = [_desc_engine(6, T//n, T % n, KB, (T//n)*rb, rb, REC),
           _desc_softmax(SP, T, nq, Tp//8),
           _desc_engine(6, T//n, T % n, VB, (T//n)*rb, rb, REC + 32),
           _desc_stop()]

    def gen():
        for a, v in words.items():
            yield dut.scratchpad.mem[a].eq(v)
        for t in range(T):
            c, slot = t % n, t//n
            for b in range(rb):
                yield dut.models[c].mem[KB//32 + slot*rb + b].eq(pack(list(krows[t][8*b:8*b + 8])))
                yield dut.models[c].mem[VB//32 + slot*rb + b].eq(pack(list(vrows[t][8*b:8*b + 8])))
        yield from install_list(dut, lst, base=LIST)
        for _ in range(200000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("list did not complete")
        bad = []
        for l in range(nq):
            got = []
            for w in range(Tp//8):
                v = yield dut.scratchpad.mem[(SP + l*Tp*4)//32 + w]
                got += [(v >> (32*i)) & 0xFFFFFFFF for i in range(8)]
            want = p[l].view(np.uint32)
            if got[:T] != list(want):
                bad.append(("p", l, [hex(x) for x in got[:T]], [hex(int(x)) for x in want]))
            got = []
            for w in range(hd//8):
                v = yield dut.scratchpad.mem[(ODST + l*hd*4)//32 + w]
                got += [(v >> (32*i)) & 0xFFFFFFFF for i in range(8)]
            if got != list(o[l].view(np.uint32)):
                bad.append(("o", l))
        assert not bad, bad[:6]

    run_cluster(dut, [gen()])


@pytest.mark.parametrize("T", [2, 3])
def test_firmware_list_full_units(T):
    """hybrid.c's whole int8 attention list on the kit's unit set: this token's FP32 K/V and q
    quantized (unit 4) into the row layout and the lane operands, the rows stored to their
    channel, then scores / softmax (unit 14) / values."""
    from gateware.core.core import TriXiumConfig
    from common.cluster import SoC, install_list, pack
    from common.cluster import run as run_cluster
    from model.activation import scratchpad_quantize_descriptor
    import test_softmax as tsm
    n, nq, n_rep, nkv, hd = 2, 8, 4, 2, 64
    hb, rb, kv_dim = hd//32, nkv*hd//32 + 1, nkv*hd
    config = TriXiumConfig(with_vecunit=False, scratchpad_size=64*1024, with_hadamard=True,
        with_activation_quantize=True, with_projection_restore=True, with_silu=True, with_norm=True,
        with_conv=True, with_softmax=True, norm_rope=False, hadamard_lanes=8, silu_ways=1,
        with_attnstream=True, attnstream_options=(("lanes", 8), ("groups", 2), ("hb_max", hd//8),
                                                  ("t_max", 64), ("int8", True)))
    dut = SoC(config, n=n, backend="verilator")
    rng = np.random.default_rng(100 + T)
    K = rng.normal(0, 2, (T, nkv, hd)).astype(f32)
    V = (rng.normal(0, 1, (T, nkv, hd))*np.exp2(rng.integers(-2, 3, (T, nkv, 1)))).astype(f32)
    qg = rng.normal(0, 1, (nq, 2*hd)).astype(f32)          # [q | gate] per head
    q = qg[:, :hd]
    K8, ek = quantize_rows(K); V8, ev = quantize_rows(V); q8, eq = quantize_rows(q)
    scale = f32(1/np.sqrt(hd))
    s = scores_reference(q8, eq, K8, ek, scale)
    p = np.stack([tsm.softmax_row(s[l]) for l in range(nq)])
    o = values_reference(p, V8, ev)
    Tp = (T + 7)//8*8
    QB, KVF, QQ, KVQ, SP, ODST, REC = 0x1000, 0x2000, 0x2800, 0x3000, 0x4000, 0x5000, 0x3800
    KB, VB = 0x4000, 0x8000
    qop = 32*(hb + 1)
    words = {}
    def put(addr, u32):
        u32 = np.asarray(u32, np.uint32)
        for i in range(0, len(u32), 8):
            words[addr//32 + i//8] = pack(list(u32[i:i + 8]) + [0]*(8 - len(u32[i:i + 8])))
    put(QB, qg.reshape(-1).view(np.uint32))
    put(KVF, np.concatenate([K[T - 1].reshape(-1), V[T - 1].reshape(-1)]).view(np.uint32))
    for l in range(nq):
        put(SP + l*Tp*4, rng.integers(0, 1 << 32, Tp, dtype=np.uint64).astype(np.uint32))
    w0 = nq << 8 | n_rep << 16 | nkv << 24
    put(REC, [w0, T, hd, QQ, qop, SP, Tp*4, int(scale.view(np.uint32))])
    put(REC + 32, [w0 | 1, T, hd, SP, Tp*4, ODST, hd*4, 0])
    krows, vrows = k_rows(K8, ek, ev), v_rows(V8)
    t1 = T - 1
    qd = lambda **f: list(np.frombuffer(scratchpad_quantize_descriptor(**f), np.uint32)) \
        if isinstance(scratchpad_quantize_descriptor(**f), (bytes, bytearray)) else list(scratchpad_quantize_descriptor(**f))
    lst = [qd(src=KVF, dst=KVQ + 32, exponents=KVQ, elements=hd, blocks=2*nkv),
           [1, 3, 0, 0, KB + (t1//n)*rb*32, rb, 0, 0, KVQ, 0, t1 % n] + [0]*5,
           [1, 3, 0, 0, VB + (t1//n)*rb*32, rb, 0, 0, KVQ + 32 + kv_dim, 0, t1 % n] + [0]*5]
    lst += [qd(src=QB + l*2*hd*4, dst=QQ + l*qop, exponents=QQ + l*qop + 32*hb, elements=hd, blocks=1)
            for l in range(nq)]
    lst += [_desc_engine(6, T//n, T % n, KB, (T//n)*rb, rb, REC),
            _desc_softmax(SP, T, nq, Tp//8),
            _desc_engine(6, T//n, T % n, VB, (T//n)*rb, rb, REC + 32),
            _desc_stop()]
    lst = [[int(x) for x in d] + [0]*(16 - len(d)) for d in lst]

    def gen():
        for a, v in words.items():
            yield dut.scratchpad.mem[a].eq(v)
        for t in range(T - 1):                           # the earlier rows, already stored
            c, slot = t % n, t//n
            for b in range(rb):
                yield dut.models[c].mem[KB//32 + slot*rb + b].eq(pack(list(krows[t][8*b:8*b + 8])))
                yield dut.models[c].mem[VB//32 + slot*rb + b].eq(pack(list(vrows[t][8*b:8*b + 8])))
        yield from install_list(dut, lst)
        for _ in range(400000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("list did not complete")
        assert not (yield dut.seq.activation_quantize_error.status)
        bad = []
        for l in range(nq):
            got = []
            for w in range(Tp//8):
                v = yield dut.scratchpad.mem[(SP + l*Tp*4)//32 + w]
                got += [(v >> (32*i)) & 0xFFFFFFFF for i in range(8)]
            if got[:T] != list(p[l].view(np.uint32)):
                bad.append(("p", l, [hex(x) for x in got[:T]], [hex(int(x)) for x in p[l].view(np.uint32)]))
            got = []
            for w in range(hd//8):
                v = yield dut.scratchpad.mem[(ODST + l*hd*4)//32 + w]
                got += [(v >> (32*i)) & 0xFFFFFFFF for i in range(8)]
            if got != list(o[l].view(np.uint32)):
                bad.append(("o", l))
        assert not bad, bad[:6]

    run_cluster(dut, [gen()])

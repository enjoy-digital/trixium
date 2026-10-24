#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/engine.py with nrows > 1: the multi-row matvec (doc/plan-multirow.md, step 3).

One weight pass, nr+1 activation rows: row r's x8 at xsrc + r*xstride beats, its results at
dst + 4*r*qstride bytes. Every row is compared BIT-EXACTLY against
model.bitexact.matvec_ternary on the same weights; the result regions' gaps must stay
untouched (no stray lane). Covered: 2-bit and trit5 codes, 2 and 4 channels, nrows 2 and 3
(nr below nrows: unused rows must not leak or be counted), per-channel latency skew and read
stalls, gather pressure (short rows), a single-row op right after a multi-row one on the same
engine, a prefetch then a multi-row matvec, the argmax tap restricted to row 0 and the absmax
over every row, the sticky error (Q8, ilv, rem, nr >= nrows) with the op run single-row, and
the pipe/reader islands on the Migen backend."""

import numpy as np
import pytest

from model import layout, bitexact as bx
from model.bons import TensorSpec, encode_rows

from common.engine import EngineBench, submit
from common.verilator import Script

T2 = layout.KIND_TERNARY
T5 = layout.KIND_TERNARY5

# Every command field the tests drive (the others stay 0): a command after a multi-row one
# must not inherit its nr/xstride/qstride.
FIELDS = ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats", "pk", "bias", "rshift",
          "xsrc", "dst", "chan", "rem", "ilv", "supw", "qn", "qstride", "nr", "xstride")

SENTINEL = 0xA5

# Helpers ------------------------------------------------------------------------------------------

def _i32(data):
    return np.frombuffer(data, dtype="<i4").astype(np.int64)

class _Tensor:
    """A random ternary tensor sharded over the channels (BONS rows), its command and the exact
    results for any x8."""
    def __init__(self, n_channels, rows_per_chan, cols, seed, kind=T2):
        rng = np.random.default_rng(seed)
        self.M = M = n_channels*rows_per_chan
        G = cols//layout.GROUP_TERNARY
        self.scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
        self.codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
        _, e16 = bx.fp16_decompose(self.scales)
        self.B = bx.choose_scale_bias(e16)
        self.R = bx.choose_result_shift(self.scales, self.B)
        spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                          scales=self.scales, values=self.codes)
        spec.kind = kind
        self.shards = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                       for c in range(n_channels)]
        self.cols = cols
        self.cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                        wlen=rows_per_chan*layout.row_beats(cols, kind),
                        sbeats=layout.scale_beats(cols, kind), cbeats=layout.code_beats(cols, kind),
                        pk=int(kind == T5), bias=self.B, rshift=self.R)

    def expected(self, x8):
        return bx.matvec_ternary(self.scales, self.codes, x8, self.B, self.R)

    def expected_rows(self, xs):
        return bx.matvec_ternary_rows(self.scales, self.codes, np.stack(xs), self.B, self.R)

def _xs(nvec, cols, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(-127, 128, cols, dtype=np.int8) for _ in range(nvec)]

def _cmd(base, **kw):
    cmd = {k: 0 for k in FIELDS}
    cmd.update(base)
    cmd.update(kw)
    return cmd

class _Scratchpad:
    """Scratchpad image: x8 vectors placed at byte addresses, result regions pre-filled with
    a sentinel (so a stray write shows)."""
    def __init__(self, size=0xC000):
        self.img = bytearray(size)
    def put(self, addr, data):
        self.img[addr:addr + len(data)] = bytes(data)
    def poison(self, addr, length):
        self.img[addr:addr + length] = bytes([SENTINEL])*length

def _rows_op(s, sp, t, xs, nr, xsrc, xstride, dst, qstride, tag, cap=60000, **kw):
    """One multi-row op: rows 0..nr's x8 placed at xsrc + r*xstride beats, results read back at
    dst + 4*r*qstride (and the whole span, for the gaps). Returns the read tags' expectations."""
    for r in range(nr + 1):
        sp.put(xsrc + 32*r*xstride, xs[r].tobytes())
    span = 4*(nr*qstride + t.M)
    sp.poison(dst, span)
    submit(s, _cmd(t.cmd, nr=nr, xsrc=xsrc, xstride=xstride, dst=dst, qstride=qstride, **kw))
    s.wait("done", cap=cap, tag=tag)
    s.step(2)
    s.read("sp", dst, span, tag)
    return dict(tag=tag, nr=nr, dst=dst, qstride=qstride, rows=t.expected_rows(xs[:nr + 1]), M=t.M)

def _check(r, exp):
    """Every row exact at its stride, every byte between/after untouched."""
    raw = r.reads[exp["tag"]]
    span = bytearray(raw)
    for k, want in enumerate(exp["rows"]):
        off = 4*k*exp["qstride"]
        got = _i32(raw[off:off + 4*exp["M"]])
        assert np.array_equal(got, want), (exp["tag"], "row", k)
        span[off:off + 4*exp["M"]] = bytes([SENTINEL])*(4*exp["M"])
    assert span == bytes([SENTINEL])*len(span), (exp["tag"], "stray write in a gap")

# Compiled engine (Verilator) ----------------------------------------------------------------------

def test_multirow_ternary_n2_r2():
    """2 channels, nrows 2, 2-bit codes: nr=1 with a gapped xstride and an odd qstride, then a
    single-row op (nr=0) on the same engine, then nr=1 again with another tensor (cols 512)
    under per-channel latency skew and read stalls."""
    n = 2
    a = _Tensor(n, 12, 256, seed=101)
    b = _Tensor(n, 9, 512, seed=102)
    off_b = (max(len(sh) for sh in a.shards) + 31)//32*32
    payloads = [sa.ljust(off_b, b"\0") + sb for sa, sb in zip(a.shards, b.shards)]
    xa = _xs(2, a.cols, 201)
    xb = _xs(2, b.cols, 202)
    sp = _Scratchpad()
    s = Script()
    exps = [
        _rows_op(s, sp, a, xa, nr=1, xsrc=0x0000, xstride=a.cols//32 + 3, dst=0x4000,
                 qstride=a.M + 13, tag="a_r2"),
        _rows_op(s, sp, a, xa[1:], nr=0, xsrc=0x1000, xstride=0, dst=0x5000, qstride=0,
                 tag="a_r1"),
        _rows_op(s, sp, b, xb, nr=1, xsrc=0x2000, xstride=b.cols//32, dst=0x6000,
                 qstride=b.M + 8, tag="b_r2", wbase=off_b),
    ]
    s.peek("error")
    r = EngineBench(n, nrows=2).run("verilator", s, sp=sp.img, payloads=payloads,
                                    latency=[8, 30], stall_rate=[0, 4])
    assert r.peeks["error"] == 0
    for e in exps:
        _check(r, e)

def test_multirow_trit5_n4_r3():
    """4 channels, nrows 3, trit5 codes, no AXPY (the hybrid pipe): nr=2 (K = 2048), nr=1
    (one row unused: it must neither leak nor be counted), gather pressure (cols 128, 48 rows
    per channel, 3 rows: 6 results per clock against the drain), argmax on row 0 only and the
    absmax over all rows."""
    n = 4
    a = _Tensor(n, 6, 2048, seed=111, kind=T5)
    b = _Tensor(n, 48, 128, seed=112, kind=T5)
    off_b = (max(len(sh) for sh in a.shards) + 31)//32*32
    payloads = [sa.ljust(off_b, b"\0") + sb for sa, sb in zip(a.shards, b.shards)]
    xa = _xs(3, a.cols, 211)
    xb = _xs(3, b.cols, 212)
    sp = _Scratchpad()
    s = Script()
    exps = []
    exps.append(_rows_op(s, sp, a, xa, nr=2, xsrc=0x0000, xstride=a.cols//32, dst=0x4000,
                         qstride=a.M + 5, tag="a_r3"))
    s.peek("argmax_val", "a_amv")
    s.peek("argmax_idx", "a_ami")
    s.peek("absmax", "a_abs")
    exps.append(_rows_op(s, sp, a, xa, nr=1, xsrc=0x0000, xstride=a.cols//32, dst=0x5000,
                         qstride=a.M, tag="a_r2"))
    exps.append(_rows_op(s, sp, b, xb, nr=2, xsrc=0x3000, xstride=b.cols//32 + 1, dst=0x7000,
                         qstride=b.M + 3, tag="b_r3", wbase=off_b))
    s.peek("error")
    r = EngineBench(n, nrows=3, trit5=True, axpy=False).run(
        "verilator", s, sp=sp.img, payloads=payloads, latency=[8, 20, 12, 40], stall_rate=[0, 3, 1, 0])
    assert r.peeks["error"] == 0
    for e in exps:
        _check(r, e)
    rows = exps[0]["rows"]
    assert r.peek("a_amv", signed_width=32) == int(rows[0].max())
    assert r.peeks["a_ami"] == int(np.argmax(rows[0]))
    assert r.peeks["a_abs"] == int(max(np.abs(row).max() for row in rows))

def test_multirow_prefetch_then_rows():
    """The hybrid engine (trit5, no AXPY, prefetch): a prefetch (op 7, the multi-row op's
    fields) then the multi-row matvec served from it, then a single-row op."""
    n = 2
    a = _Tensor(n, 8, 1024, seed=121, kind=T5)
    xa = _xs(2, a.cols, 221)
    sp = _Scratchpad()
    s = Script()
    submit(s, _cmd(a.cmd, kind=7, nr=1, xstride=a.cols//32, qstride=a.M))
    s.wait("done", cap=5000, tag="pf")
    s.step(300)
    exps = [
        _rows_op(s, sp, a, xa, nr=1, xsrc=0x0000, xstride=a.cols//32, dst=0x4000, qstride=a.M,
                 tag="pf_r2"),
        _rows_op(s, sp, a, xa[1:], nr=0, xsrc=0x1000, xstride=0, dst=0x6000, qstride=0,
                 tag="r1"),
    ]
    s.peek("error")
    s.peek("pf_hits")
    s.peek("pf_mismatch")
    r = EngineBench(n, nrows=2, trit5=True, axpy=False, prefetch=True).run(
        "verilator", s, sp=sp.img, payloads=[sh for sh in a.shards], latency=30)
    assert r.peeks["error"] == 0
    assert r.peeks["pf_hits"] == 1 and r.peeks["pf_mismatch"] == 0
    for e in exps:
        _check(r, e)

@pytest.mark.parametrize("case", ["q8", "ilv", "rem", "nr_over"])
def test_multirow_unsupported_sticky_error(case):
    """nr > 0 with Q8, ilv, rem, or nr >= nrows: the sticky error, the op runs single-row
    (completes, row 0 exact, nothing written at row 1's slot); the next valid op clears it."""
    n = 2
    if case == "q8":
        rows, cols = 4, 256
        rng = np.random.default_rng(131)
        M = n*rows
        w8 = rng.integers(-127, 128, (M, cols), dtype=np.int8)
        sc = rng.uniform(0.005, 0.02, (M, cols//layout.GROUP_Q8)).astype(np.float16)
        _, e16 = bx.fp16_decompose(sc)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(sc, B, gs_max=bx.GS_MAX_Q8)
        spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8, M, cols, scales=sc, values=w8)
        shards = [encode_rows(spec, *layout.shard_rows(M, n, c)).tobytes() for c in range(n)]
        base = dict(kind=1, rows=rows, cols=cols, wbase=0,
                    wlen=rows*layout.row_beats(cols, layout.KIND_Q8),
                    sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                    cbeats=layout.code_beats(cols, layout.KIND_Q8), bias=B, rshift=R)
        x8 = _xs(1, cols, 231)[0]
        expect = bx.matvec_q8(sc, w8, x8, B, R)
        bad = {}
    else:
        t = _Tensor(n, 8, 256, seed=132)
        shards, base, M = t.shards, t.cmd, t.M
        x8 = _xs(1, t.cols, 232)[0]
        expect = t.expected(x8)
        # rem: channel 0 streams one extra row (the striped KV layout; its beats read as zeros
        # past the shard): the op must run single-row and flag; only the error, completion
        # and row 1's untouched slot are checked (the striped slots overlap).
        bad = {"ilv": dict(ilv=1), "rem": dict(rem=1), "nr_over": dict(nr=2)}[case]
    sp = _Scratchpad()
    sp.put(0, x8.tobytes())
    sp.put(0x200, _xs(1, len(x8), 233)[0].tobytes())
    sp.poison(0x4000, 0x1000)
    s = Script()
    cmd = _cmd(base, xsrc=0, xstride=0x200//32, dst=0x4000, qstride=0x400//4, nr=1)
    cmd.update(bad)
    submit(s, cmd)
    s.wait("done", cap=60000, tag="bad")
    s.step(2)
    s.peek("error", "err_bad")
    s.read("sp", 0x4000, 4*M, "row0")
    s.read("sp", 0x4400, 4*M, "row1")
    # The next valid single-row op clears it.
    submit(s, _cmd(base, xsrc=0, dst=0x6000))
    s.wait("done", cap=60000, tag="good")
    s.step(2)
    s.peek("error", "err_good")
    s.read("sp", 0x6000, 4*M, "good")
    r = EngineBench(n, nrows=2).run("verilator", s, sp=sp.img, payloads=shards)
    assert r.peeks["err_bad"] == 1, case
    assert r.reads["row1"] == bytes([SENTINEL])*(4*M), case
    if case in ("q8", "nr_over"):
        assert np.array_equal(_i32(r.reads["row0"]), expect), case
    assert r.peeks["err_good"] == 0, case
    assert np.array_equal(_i32(r.reads["good"]), expect), case

# Islands (Migen backend) --------------------------------------------------------------------------

@pytest.mark.parametrize("phase", [0, pytest.param(5, marks=pytest.mark.slow)])
def test_multirow_trit5_reader_island(phase):
    """ReaderIsland + PipeIsland with nrows 2 (the YPCB hyb20 structure: readers in `ddr`, pipes
    in `peng`, engine in sys; periods 70 / 60 / 56): a multi-row trit5 op, a single-row op, the
    multi-row op again; every row exact."""
    n = 2
    a = _Tensor(n, 4, 1024, seed=141, kind=T5)
    xa = _xs(2, a.cols, 241)
    sp = _Scratchpad(size=64*1024)
    s = Script()
    s.step(phase)
    exps = [
        _rows_op(s, sp, a, xa, nr=1, xsrc=0x0000, xstride=a.cols//32 + 2, dst=0x4000,
                 qstride=a.M + 3, tag="r2"),
        _rows_op(s, sp, a, xa[1:], nr=0, xsrc=0x1000, xstride=0, dst=0x5000, qstride=0,
                 tag="r1"),
        _rows_op(s, sp, a, xa, nr=1, xsrc=0x0000, xstride=a.cols//32 + 2, dst=0x6000,
                 qstride=a.M, tag="r2b"),
    ]
    s.peek("error")
    bench = EngineBench(n, xbuf_groups=32, trit5=True, axpy=False, pipe_cd="peng",
                        readers_cd="ddr", nrows=2)
    r = bench.run("migen", s, sp=sp.img, payloads=a.shards, clocks={"sys": 70, "ddr": 60, "peng": 56})
    assert r.peeks["error"] == 0
    for e in exps:
        _check(r, e)

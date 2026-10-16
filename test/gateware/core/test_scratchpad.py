#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/scratchpad.py: the flat ScratchpadArbiter and the per-cycle arbitration between the
engine and the vector unit."""

import random

import numpy as np
import pytest

from migen import *

from litex.gen import LiteXModule

from gateware.core.scratchpad import ScratchpadArbiter
from model import layout, bitexact as bx
from model.bons import TensorSpec, encode_rows

from common.engine import EngineBench, submit, vec_submit, vec_upload
from common.verilator import Script

# ScratchpadArbiter --------------------------------------------------------------------------------
# The flat scratchpad arbiter must be the nested 2:1 client-mux chain, bit for bit.
#
# ScratchpadArbiter replaces one ScratchpadClientMux per attached unit (serial mux levels
# fail setup timing on the full SoC). It is only safe to swap in if
# every output toward the scratchpad and every return toward every client is identical to
# the chain's, for every combination of busy bits - including two units busy at once,
# which the sequencer never does but a manual CSR start could. Checked exhaustively over
# the busy combinations for one to six attached clients, with random payloads.

# Reference: the nested 2:1 client mux the arbiter replaced.
class ScratchpadClientMux(LiteXModule):
    """Give an auxiliary unit priority over the vector-unit scratchpad client."""

    def __init__(self, vector, auxiliary):
        self.busy = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal.like(vector.sp_rd_adr)
        self.sp_rd_dat = Signal.like(vector.sp_rd_dat)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal.like(vector.sp_wr_adr)
        self.sp_wr_we  = Signal.like(vector.sp_wr_we)
        self.sp_wr_dat = Signal.like(vector.sp_wr_dat)
        self.sp_wr_gnt = Signal()
        sel = auxiliary.busy
        self.comb += [
            self.busy.eq(vector.busy | auxiliary.busy),
            self.sp_rd_req.eq(Mux(sel, auxiliary.sp_rd_req, vector.sp_rd_req)),
            self.sp_rd_adr.eq(Mux(sel, auxiliary.sp_rd_adr, vector.sp_rd_adr)),
            vector.sp_rd_dat.eq(self.sp_rd_dat),
            auxiliary.sp_rd_dat.eq(self.sp_rd_dat),
            self.sp_wr_req.eq(Mux(sel, auxiliary.sp_wr_req, vector.sp_wr_req)),
            self.sp_wr_adr.eq(Mux(sel, auxiliary.sp_wr_adr, vector.sp_wr_adr)),
            self.sp_wr_we.eq(Mux(sel, auxiliary.sp_wr_we, vector.sp_wr_we)),
            self.sp_wr_dat.eq(Mux(sel, auxiliary.sp_wr_dat, vector.sp_wr_dat)),
            vector.sp_wr_gnt.eq(self.sp_wr_gnt & ~sel),
            auxiliary.sp_wr_gnt.eq(self.sp_wr_gnt & sel),
        ]


WIDTHS = dict(sp_rd_adr=13, sp_rd_dat=256, sp_wr_adr=13, sp_wr_we=8, sp_wr_dat=256)
DRIVEN = ("busy", "sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat")
TO_SP = ("busy", "sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat")
RETURNED = ("sp_rd_dat", "sp_wr_gnt")


class Client:
    def __init__(self):
        for name in ("busy", "sp_rd_req", "sp_wr_req", "sp_wr_gnt"):
            setattr(self, name, Signal())
        for name, width in WIDTHS.items():
            setattr(self, name, Signal(width))


class DUT(Module):
    def __init__(self, n):
        self.chain_base, self.chain = Client(), [Client() for _ in range(n)]
        self.flat_base, self.flat = Client(), [Client() for _ in range(n)]
        top = self.chain_base
        for i, client in enumerate(self.chain):
            mux = ScratchpadClientMux(top, client)
            setattr(self.submodules, f"mux{i}", mux)
            top = mux
        self.chain_top = top
        self.submodules.arb = ScratchpadArbiter(self.flat_base, self.flat)
        # The scratchpad side feeds both trees the same read data and write grant.
        self.rd_dat, self.wr_gnt = Signal(256), Signal()
        self.comb += [self.chain_top.sp_rd_dat.eq(self.rd_dat), self.chain_top.sp_wr_gnt.eq(self.wr_gnt),
                      self.arb.sp_rd_dat.eq(self.rd_dat), self.arb.sp_wr_gnt.eq(self.wr_gnt)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_flat_arbiter_equals_the_nested_chain(n):
    dut = DUT(n)
    rng = random.Random(1000 + n)

    def drive():
        checked = 0
        for busy_bits in range(1 << (n + 1)):
            for _ in range(3):
                for k, (c_chain, c_flat) in enumerate(zip([dut.chain_base] + dut.chain,
                                                          [dut.flat_base] + dut.flat)):
                    values = {"busy": (busy_bits >> k) & 1}
                    for name in DRIVEN[1:]:
                        width = WIDTHS.get(name, 1)
                        values[name] = rng.getrandbits(width)
                    for name, value in values.items():
                        yield getattr(c_chain, name).eq(value)
                        yield getattr(c_flat, name).eq(value)
                yield dut.rd_dat.eq(rng.getrandbits(256))
                yield dut.wr_gnt.eq(rng.getrandbits(1))
                yield
                for name in TO_SP:
                    a, b = (yield getattr(dut.chain_top, name)), (yield getattr(dut.arb, name))
                    assert a == b, f"n={n} busy={busy_bits:b}: {name} chain {a:#x} flat {b:#x}"
                for k, (c_chain, c_flat) in enumerate(zip([dut.chain_base] + dut.chain,
                                                          [dut.flat_base] + dut.flat)):
                    for name in RETURNED:
                        a, b = (yield getattr(c_chain, name)), (yield getattr(c_flat, name))
                        assert a == b, f"n={n} busy={busy_bits:b} client {k}: {name} {a:#x} != {b:#x}"
                checked += 1
        assert checked == 3 << (n + 1)

    run_simulation(dut, drive())


def test_priority_is_last_attached():
    # Two units busy together: the one attached last owns the port, as in the chain.
    dut = DUT(3)

    def drive():
        for c in [dut.flat_base] + dut.flat:
            yield c.busy.eq(0)
        for k, c in enumerate(dut.flat):
            yield c.sp_rd_adr.eq(0x100 + k)
        yield dut.flat_base.sp_rd_adr.eq(0x0FF)
        yield dut.flat[0].busy.eq(1)
        yield dut.flat[2].busy.eq(1)
        yield dut.wr_gnt.eq(1)
        yield
        assert (yield dut.arb.sp_rd_adr) == 0x102
        assert (yield dut.flat[2].sp_wr_gnt) == 1
        assert (yield dut.flat[0].sp_wr_gnt) == 0 and (yield dut.flat_base.sp_wr_gnt) == 0
        yield dut.flat[0].busy.eq(0)
        yield dut.flat[2].busy.eq(0)
        yield
        assert (yield dut.arb.sp_rd_adr) == 0x0FF and (yield dut.flat_base.sp_wr_gnt) == 1

    run_simulation(dut, drive())


# Per-Cycle Arbitration (Engine + Vector Unit Overlap) ---------------------------------------------
# Per-cycle scratchpad arbitration (doc/units.md (Overlap and scratchpad arbitration)).
#
# Engine + vector unit sharing the scratchpad WITHOUT a sequencer or CSR front: engine.cmd is
# testbench-driven, the vector unit runs from its own CSRs, and the two are launched
# concurrently - the per-cycle port arbitration is what keeps both results bit-exact. The
# bench mirrors the SoC's sideband idle wiring (no sequencer in this DUT): the posted-write
# drain works without it, but read grants are units_idle-gated. Each test runs on the
# compiled design (common.engine.EngineBench); the concurrent engine/vector test also on
# Migen's simulator in the slow tier.

# Every test compiled; Migen's simulator (minutes per test) for the representative only.
BACKEND = pytest.mark.parametrize("backend", ["verilator", pytest.param("migen", marks=pytest.mark.slow)])
COMPILED = pytest.mark.parametrize("backend", ["verilator"])

def _overlap_bench(n_channels):
    return EngineBench(n_channels, vecunit=True, overlap=True)

def _overlap_vec_golden(sp, vec_args):
    from model import asm as vasm
    mirror = bytearray(sp)
    vasm.VecInterp(mirror).run(vasm.assemble(vasm.PROGRAMS["resid_add"]()), list(vec_args))
    return mirror

def _sb_post(s, beat, sel, wdat, bound, tag):
    """A posted sideband write (slot 1): accepted within `bound` cycles."""
    s.set("sbw1_adr", beat)
    s.set("sbw1_sel", sel)
    s.set("sbw1_bsel", 0xF)
    s.set("sbw1_wdat", wdat)
    s.handshake("sbw1_req", "sbw1_ok", cap=bound, tag=tag)
    s.step()

def _sb_read(s, beat, sel, bound, tag):
    """A sideband read (slot 1): granted within `bound` cycles; the beat is peeked as `tag`."""
    s.set("sb1_adr", beat)
    s.set("sb1_sel", sel)
    s.handshake("sb1_req", "sb1_gnt", cap=bound, tag=f"{tag}_grant")
    s.peek("sb_rdat", tag)

def _lane(beat, sel):
    return (beat >> (32*sel)) & 0xFFFFFFFF

def _overlap_case(seed):
    """The matvec + resid_add scenario the sideband tests share."""
    n_channels, rows_per_chan, cols = 2, 32, 128
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols).astype(np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    res_exp = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    nv = 1024
    vec_args = [0x4000, 0x4800, 0x5000, nv//16, -1, -3]
    sp = bytearray(64*1024)
    sp[0x1000:0x1000 + cols] = x8.tobytes()
    sp[0x4000:0x4000 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x4800:0x4800 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    mirror = _overlap_vec_golden(sp, vec_args)
    sb_src = rng.integers(0, 2**32, 64, dtype=np.uint32)
    sp[0x7000:0x7000 + 256] = sb_src.astype("<u4").tobytes()
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, rem=0, sbeats=(G*2 + 31)//32,
               cbeats=(cols + 127)//128, wbase=0, wlen=len(payloads[0])//32, bias=B, rshift=R,
               xsrc=0x1000, dst=0x2000)
    return dict(n_channels=n_channels, M=M, nv=nv, vec_args=vec_args, sp=sp, mirror=mirror,
                sb_src=sb_src, payloads=payloads, res_exp=res_exp, cmd=cmd)

@COMPILED
def test_overlap_sideband_slot(backend):
    # Posted sideband writes stream in DURING a concurrent engine matvec + vec resid_add
    # (2026-08-31 second form): each push must be accepted within a bounded wait, the
    # FIFO must drain (sbw_pending falls), reads AFTER the run (units_idle + FIFO-empty
    # gated) must see every posted write and the untouched known block, both unit
    # results must equal their serial goldens, and no gather write may be lost.
    from model import asm as vasm
    c  = _overlap_case(95)
    nv = c["nv"]
    s  = Script()
    vec_upload(s, vasm.assemble(vasm.PROGRAMS["resid_add"]()))
    submit(s, c["cmd"])
    vec_submit(s, c["vec_args"])
    # Posted writes streamed while both units run.
    wrote = {}
    for i in range(48):
        beat, sel = 0x7800//32 + (i % 4), (i // 4) % 8
        w = (0xC0DE0000 + i) & 0xFFFFFFFF
        _sb_post(s, beat, sel, w, 300, "posted_write_starved_during_concurrent_units")
        wrote[(beat, sel)] = w
        s.step(2)
    s.wait("done", cap=30000, tag="engine", fatal=False)
    # FIFO must drain once granted cycles exist.
    s.wait({"sbw_pending": 0}, cap=1000, tag="posted-write_FIFO_never_drained")
    s.step(200)
    s.peek("gw_ovf")
    s.read("sp", 0x2000, 4*c["M"], "res")
    s.read("sp", 0x5000, 2*nv, "vec")
    for n, (beat, sel) in enumerate(wrote):
        _sb_read(s, beat, sel, 3000, f"posted{n}")
    for i in range(8):
        _sb_read(s, 0x7000//32 + i, i, 3000, f"known{i}")
    r = _overlap_bench(c["n_channels"]).run(backend, s, sp=c["sp"], payloads=c["payloads"])

    assert r.peeks["gw_ovf"] == 0, "gather write lost under sideband traffic"
    assert np.array_equal(np.frombuffer(r.reads["res"], dtype="<i4"), c["res_exp"]), \
        "matvec result changed under sideband traffic"
    assert r.reads["vec"] == bytes(c["mirror"][0x5000:0x5000 + 2*nv]), \
        "vec result changed under sideband traffic"
    for n, ((beat, sel), w) in enumerate(wrote.items()):
        v = _lane(r.peeks[f"posted{n}"], sel)
        assert v == w, f"posted write at beat {beat} sel {sel}: {v:#x} != {w:#x}"
    for i in range(8):
        v = _lane(r.peeks[f"known{i}"], i)
        assert v == int(c["sb_src"][i*8 + i]), f"known block read {i} returned {v:#x}"

@COMPILED
def test_overlap_sideband_read_slot(backend):
    # C2-2: sideband READS grant during a concurrent engine matvec + vec resid_add (the
    # R-port-only slot), return correct data, and never outrun the posted-write FIFO -
    # a read issued right after a posted write to the same word must return the NEW
    # value (the FIFO-empty gate is a tested contract). Unit results stay bit-exact and
    # no gather write is lost (reads must never disturb the W port).
    from model import asm as vasm
    c  = _overlap_case(97)
    nv = c["nv"]
    s  = Script()
    vec_upload(s, vasm.assemble(vasm.PROGRAMS["resid_add"]()))
    submit(s, c["cmd"])
    vec_submit(s, c["vec_args"])
    # Mid-activity reads of the known block: must grant within a small bound.
    for i in range(24):
        _sb_read(s, 0x7000//32 + (i % 8), i % 8, 300, f"mid{i}")
        s.step(2)
    # Read-after-posted-write: the read must wait for the drain and see the NEW value.
    for i in range(8):
        _sb_post(s, 0x7800//32, i, (0xFEED0000 + i) & 0xFFFFFFFF, 300, "posted_write_starved")
        _sb_read(s, 0x7800//32, i, 600, f"after{i}")
    s.wait("done", cap=30000, tag="engine", fatal=False)
    # The vector job too: when the engine's done pulse came during the reads above, the wait
    # for it ran its full 30000 cycles and covered the job; when it comes after, it may not.
    s.wait({"busy": 0, "vu_busy": 0}, cap=30000, tag="units_timeout")
    s.step(200)
    s.peek("gw_ovf")
    s.read("sp", 0x2000, 4*c["M"], "res")
    s.read("sp", 0x5000, 2*nv, "vec")
    r = _overlap_bench(c["n_channels"]).run(backend, s, sp=c["sp"], payloads=c["payloads"])

    for i in range(24):
        v = _lane(r.peeks[f"mid{i}"], i % 8)
        assert v == int(c["sb_src"][(i % 8)*8 + (i % 8)]), f"read {i} returned {v:#x}"
    for i in range(8):
        w = (0xFEED0000 + i) & 0xFFFFFFFF
        v = _lane(r.peeks[f"after{i}"], i)
        assert v == w, f"read-after-post {i}: {v:#x} != {w:#x}"
    assert r.peeks["gw_ovf"] == 0, "gather write lost under read traffic"
    assert np.array_equal(np.frombuffer(r.reads["res"], dtype="<i4"), c["res_exp"]), \
        "matvec result changed under sideband reads"
    assert r.reads["vec"] == bytes(c["mirror"][0x5000:0x5000 + 2*nv]), \
        "vec result changed under sideband reads"

@BACKEND
def test_overlap_engine_vec_concurrent(backend):
    # Engine ternary matvec and a vector resid_add run CONCURRENTLY on disjoint regions.
    # Both results must equal their serial goldens, cyc_both must show real overlap, and
    # no gather write may be lost (gw_ovf).
    n_channels, rows_per_chan, cols, seed = 2, 32, 128, 93
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols).astype(np.int8)

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    res_exp = bx.matvec_ternary(scales, codes, x8, B, R)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]

    # Vec job: resid_add over 1024 int16 (64 iterations, LDV/LDV/STV per beat - the most
    # port-dense stock program). Regions disjoint from the matvec's x/dst.
    nv = 1024
    vec_args = [0x4000, 0x4800, 0x5000, nv//16, -1, -3]
    sp = bytearray(64*1024)
    sp[0x1000:0x1000 + cols] = x8.tobytes()
    sp[0x4000:0x4000 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x4800:0x4800 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    mirror = _overlap_vec_golden(sp, vec_args)

    from model import asm as vasm
    words = vasm.assemble(vasm.PROGRAMS["resid_add"]())
    rb    = layout.row_beats(cols, layout.KIND_TERNARY)
    s     = Script()
    vec_upload(s, words)
    # Vec first (it runs ~2k cycles), engine a few cycles later - contention spans the
    # engine's whole FILL/RUN window.
    vec_submit(s, vec_args)
    submit(s, dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0x1000, dst=0x2000))
    s.wait({"busy": 0, "vu_busy": 0}, cap=50000, tag="overlap_run_timeout")
    s.step()
    s.peek("cyc_both")
    s.peek("gw_ovf")
    s.read("sp", 0x2000, 4*M, "res")
    s.read("sp", 0x5000, 2*nv, "vec")
    r = _overlap_bench(n_channels).run(backend, s, sp=sp, payloads=payloads)
    res = np.frombuffer(r.reads["res"], dtype="<i4")
    assert r.peeks["gw_ovf"] == 0, "gather FIFO overflow"
    assert np.array_equal(res, res_exp), (res[:8], res_exp[:8])
    assert r.reads["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert r.peeks["cyc_both"] > 50, f"no real overlap measured (cyc_both={r.peeks['cyc_both']})"

@COMPILED
def test_overlap_fetch_under_vec(backend):
    # Engine FETCH (HBM -> scratchpad, 1 write/cycle - the worst-case W-port load, which
    # must backpressure the reader when a vec store steals the port) under the same
    # port-dense vec program. Fetched bytes and vec results both bit-exact; no overflow.
    n_channels, seed = 2, 47
    rng      = np.random.default_rng(seed)
    payloads = [rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
                for _ in range(n_channels)]
    fetch_beats = 64  # 2 KB from channel 0.

    nv = 1024
    vec_args = [0x4000, 0x4800, 0x5000, nv//16, -1, -3]
    sp = bytearray(64*1024)
    sp[0x4000:0x4000 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x4800:0x4800 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    mirror = _overlap_vec_golden(sp, vec_args)

    from model import asm as vasm
    words = vasm.assemble(vasm.PROGRAMS["resid_add"]())
    s     = Script()
    vec_upload(s, words)
    vec_submit(s, vec_args)
    submit(s, dict(kind=2, wbase=0, wlen=fetch_beats, dst=0x3000, chan=0))
    s.wait({"busy": 0, "vu_busy": 0}, cap=50000, tag="fetch-under-vec_timeout")
    s.step()
    s.peek("cyc_both")
    s.peek("gw_ovf")
    s.read("sp", 0x3000, 32*fetch_beats, "fetch")
    s.read("sp", 0x5000, 2*nv, "vec")
    r = _overlap_bench(n_channels).run(backend, s, sp=sp, payloads=payloads)
    assert r.peeks["gw_ovf"] == 0, "gather FIFO overflow"
    assert r.reads["fetch"] == payloads[0][:32*fetch_beats], "fetched data corrupted"
    assert r.reads["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert r.peeks["cyc_both"] > 20, f"no real overlap measured (cyc_both={r.peeks['cyc_both']})"

@COMPILED
def test_overlap_vec_throughput_under_stream(backend):
    # The mechanism behind the result packer (doc/method.md, "Lessons from the climb"): with one
    # result per W-port write, a saturating plain-matvec stream held the port ~92% of the
    # time and a vector STORE was granted about one cycle in twelve - norm_i32 ran 2.9x
    # slower under the up projection than idle. This pins the property: the same
    # store-dense vector program (resid_add: 2 loads + 1 store per beat) must complete
    # under a saturating stream within 1.3x its idle time. Eight channels x 5-beat rows
    # demand 1.6 results/cycle, so the grant runs every cycle - the card's condition.
    n_channels, rows_per_chan, cols, seed = 8, 160, 128, 94
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols).astype(np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16); R = bx.choose_result_shift(scales, B)
    res_exp = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    nv = 512 # ~300 cycles of vector work: the 160-row stream (800 beats/channel, the sim
             # streams ~1 beat/cycle/channel) outlasts it, and the test asserts that, so the
             # ratio is measured entirely under the stream.
    vec_args = [0x8000, 0x9000, 0xA000, nv//16, -1, -3]
    sp = bytearray(64*1024)
    sp[0x1000:0x1000 + cols] = x8.tobytes()
    sp[0x8000:0x8000 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x9000:0x9000 + 2*nv] = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    mirror = _overlap_vec_golden(sp, vec_args)
    from model import asm as vasm
    words = vasm.assemble(vasm.PROGRAMS["resid_add"]())
    rb    = layout.row_beats(cols, layout.KIND_TERNARY)
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0x1000, dst=0x2000)

    def measure(with_engine):
        s = Script()
        vec_upload(s, words)
        if with_engine:
            submit(s, cmd)
            # Let the stream reach steady state before the vector job starts.
            s.step(40)
        vec_submit(s, vec_args)
        if with_engine:
            s.count("dbg_gw_we", "writes")
        s.wait({"vu_busy": 0}, cap=60000, tag="vec_timeout")
        if with_engine:
            s.peek("busy", "eng_busy_at_vec_end")
            s.wait({"busy": 0}, cap=60000, tag="engine_timeout")
            s.count_end("writes")
            s.step()
            s.peek("gw_ovf")
            s.read("sp", 0x2000, 4*M, "res")
        s.read("sp", 0xA000, 2*nv, "vec")
        return _overlap_bench(n_channels).run(backend, s, sp=sp, payloads=payloads)

    idle  = measure(False)
    under = measure(True)
    idle_cycles, under_cycles = idle.waits["vec_timeout"][0], under.waits["vec_timeout"][0]
    writes = under.counts["writes"]
    assert under.peeks["gw_ovf"] == 0, "gather FIFO overflow"
    assert np.array_equal(np.frombuffer(under.reads["res"], dtype="<i4"), res_exp)
    assert idle.reads["vec"] == bytes(mirror[0xA000:0xA000 + 2*nv])
    assert under.reads["vec"] == bytes(mirror[0xA000:0xA000 + 2*nv])
    ratio = under_cycles/idle_cycles
    print(f"\nvec under stream: {ratio:.2f}x ({under_cycles} vs {idle_cycles} "
          f"cycles); engine writes {writes} for {M} results")
    assert under.peeks["eng_busy_at_vec_end"], "stream ended before the vector job: not measured under stream"
    assert writes <= M//8 + n_channels, (writes, M)
    assert ratio <= 1.3, (f"vector program {ratio:.2f}x slower under the stream "
                          f"({under_cycles} vs {idle_cycles} cycles)")

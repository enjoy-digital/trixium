#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/sequencer.py: descriptor lists through the sequencer, the engine and the vector unit.

The list tests run on the compiled design (scratchpad + vector unit + sequencer + engine +
CSR front, common.engine.EngineBench); the mixed list also on Migen's simulator in the slow
tier."""

import numpy as np
import pytest

from migen import *

from model import layout, bitexact as bx
from model.bons import TensorSpec, encode_rows

from common.engine import EngineBench, seq_start, vec_upload
from common.verilator import Script

# Backends -----------------------------------------------------------------------------------------

# Every test compiled; Migen's simulator (minutes per test) for the representative only.
BACKEND = pytest.mark.parametrize("backend", ["verilator", pytest.param("migen", marks=pytest.mark.slow)])
COMPILED = pytest.mark.parametrize("backend", ["verilator"])


def _run_list(backend, n_channels, payloads, sp, words, count, cap, reads, peeks=(), seed7=True,
              dumps=(), tag="sequencer_timeout"):
    """Upload the vector programs, run the descriptor list at 0x6000 until the sequencer
    drops busy, then read the scratchpad regions {tag: (addr, length)} and the peeks.
    seed7: wire the sequencer's S7 seed into the vector unit (as the SoC does)."""
    s = Script()
    vec_upload(s, words)
    seq_start(s, 0x6000, count)
    s.dump(*dumps)
    s.wait({"seq_busy": 0}, cap=cap, tag=tag)
    s.step()
    for name in peeks:
        s.peek(name)
    for name, (addr, length) in reads.items():
        s.read("sp", addr, length, name)
    bench = EngineBench(n_channels, vecunit=True, seq=True, front=True, seed7=seed7)
    return bench.run(backend, s, sp=sp, payloads=payloads)

# Descriptor Lists ---------------------------------------------------------------------------------

def _desc_engine(**f):
    w = [0]*16
    w[0] = (1 | (f.get("kcap", 0) << 5) | (f.get("kreg", 0) << 6)
            | (f.get("kacc", 0) << 8) | (f.get("nowait", 0) << 9))
    w[1] = (f.get("kind", 0) | (f.get("ilv", 0) << 8) | (f.get("supw", 0) << 9)
            | (f.get("rshift_rpre", 0) << 10) | (f.get("qn", 0) << 11)
            | (f.get("xred", 0) << 12))
    w[2] = f.get("rows", 0) | (f.get("rem", 0) << 16)
    w[3] = f.get("cols", 0)
    w[4] = f.get("wbase", 0)
    w[5] = f.get("wlen", 0)
    w[6] = f.get("sbeats", 0) | (f.get("cbeats", 0) << 8) | (f.get("qstride", 0) << 16)
    w[7] = f.get("bias", 0) | (f.get("rshift", 0) << 16) | (f.get("rshift2", 0) << 22)
    w[8] = f.get("xsrc", 0)
    w[9] = f.get("dst", 0)
    w[10] = f.get("chan", 0)
    return b"".join(x.to_bytes(4, "little") for x in w)

def _desc_vec(pc, args, srcs=None, seed_abs=0, kadds=None, barrier=0, kadd_add=0,
              lut_beat=None):
    w = [0]*16
    sel = 0
    for i, sv in enumerate(srcs or []):
        sel |= (sv & 3) << (2*i)
    w[0] = (2 | ((1 << 4) if seed_abs else 0) | ((1 << 10) if barrier else 0)
            | ((1 << 11) if kadd_add else 0) | (sel << 16))
    if lut_beat is not None:                  # w0[12]: A5 = LUT[imm5 + kreg[kadds[5]-1]]
        w[0] |= 1 << 12
        w[10] = lut_beat
    w[1] = pc
    for i, kv in enumerate(kadds or []):     # 0 = none, n = add kreg[n-1] after the source
        w[1] |= (kv & 3) << (16 + 2*i)
    for i, a in enumerate(args):
        w[2 + i] = a & 0xFFFFFFFF
    return b"".join(x.to_bytes(4, "little") for x in w)

def _desc_stop():
    return (0).to_bytes(4, "little")*16

def test_sequencer_start_drains_posted_writes():
    # The list-start ordering contract. CPU patches reach the scratchpad
    # through a posted-write FIFO; the descriptor fetch no longer waits on that FIFO
    # cycle by cycle (matvec.py sp_gnt), so the guarantee that a list sees its own
    # patches rests entirely on this: START must not fetch descriptor 0 while any
    # posted write is still in flight.
    #
    # Driven at the FSM boundary on purpose. The integration-level version of this test
    # has no teeth: sbw_gnt is blocked only by a vec store or a full write FIFO, so on
    # any quiet cycle the patch lands within a cycle or two and a broken design passes
    # by luck. Holding sbw_pending here makes the contract the only thing under test.
    from gateware.core.sequencer import Sequencer

    class _DUT(Module):
        def __init__(self):
            self.submodules.seq = Sequencer()

    dut = _DUT()

    def tb():
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(4)
        yield dut.seq.sbw_pending.eq(1)      # a patch for this list is still in the FIFO
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        # While the write is in flight the list may not read a single descriptor word,
        # and it must already report busy (firmware polls this to know the list is live).
        for cyc in range(20):
            yield
            assert not (yield dut.seq.sp_active), \
                f"descriptor fetched at cycle {cyc} with a posted write still in flight"
            assert (yield dut.seq.busy), f"not busy at cycle {cyc} of the drain"
        # The write lands: the list proceeds, and reads descriptor 0 at its own base.
        yield dut.seq.sbw_pending.eq(0)
        for _ in range(8):
            yield
            if (yield dut.seq.sp_active):
                assert (yield dut.seq.sp_adr) == 0x6000 >> 5, (yield dut.seq.sp_adr)
                return
        raise AssertionError("the sequencer never left the drain")

    run_simulation(dut, tb())

@COMPILED
def test_sequencer_v2_chaining(backend):
    # Sequencer v2 primitives in one list, the way seq_build_attn uses them:
    #   reduce_acc (res6 = absmax) -> spill (S7 seeded with prev res6, persisted to SP,
    #   res6 preserved) -> norm_i32 with k from src=3 (k(res6)) -> reduce_acc over a
    #   crafted buffer (res6 = sum p8, standing in for softmax8's S6) -> engine AXPY with
    #   rshift_rpre=1 (rshift = bitlen(res6)+5 from the sequencer, w7 rshift left 0).
    from model import asm as vasm

    n_channels, cols, T, seed = 2, 64, 5, 83
    rng = np.random.default_rng(seed)

    # reduce_acc data: 16 int32 acc + src, absmax sets the norm k.
    acc = rng.integers(-(1 << 30), 1 << 30, 16).astype("<i4")
    src = rng.integers(-(1 << 30), 1 << 30, 16).astype("<i4")
    summed = acc.astype(np.int64) + src.astype(np.int64)
    K1     = int(np.abs(summed).max())
    k_exp  = max(0, K1.bit_length() - 14)
    nrm_exp = np.clip(summed >> k_exp, -32767, 32767).astype(np.int16)

    # AXPY data (test_engine_axpy shape): res6 for rpre comes from a reduce over [S, 0..].
    ev  = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    p8  = rng.integers(1, 128, T, dtype=np.int8)
    S   = int(p8.astype(np.int64).sum())
    _, e16 = bx.fp16_decompose(scales)
    B_ax = 12 - int(e16.max())
    Rpre = bx.axpy_rpre(S)
    assert Rpre == S.bit_length() + 5  # The sequencer computes bitlen(res6) + 5.

    rows, rem = T//n_channels, T % n_channels
    payloads, axpy_exp = [], []
    for c in range(n_channels):
        pos = [t for t in range(T) if t % n_channels == c]
        if pos:
            spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                len(pos), cols, scales=scales[pos], values=v8[pos])
            payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
            axpy_exp.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B_ax, Rpre))
        else:
            payloads.append(b"")
            axpy_exp.append(np.zeros(cols, dtype=np.int32))
    maxlen   = max(len(p) for p in payloads)
    payloads = [p.ljust(maxlen, b"\0") for p in payloads]
    axpy_exp = np.concatenate(axpy_exp)

    words = []
    offs  = {}
    for name in ("reduce_acc", "norm_i32", "spill"):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())

    # SP map: acc 0x0, src 0x100, B2 [S,0..] 0x180, zeros 0x1C0, p8 0x200,
    # norm out 0x3000, spill out 0x3800, axpy out 0x4000, descriptors 0x6000.
    b2 = np.zeros(8, dtype="<i4"); b2[0] = S
    rb = layout.row_beats(cols, layout.KIND_Q8)
    descs = (_desc_vec(offs["reduce_acc"], [0x0, 0x100, 0x0, 2])
             + _desc_vec(offs["spill"], [0x3800])
             + _desc_vec(offs["norm_i32"], [0x0, 0, 0x3000, 1, 0],
                         srcs=[0, 0, 0, 0, 3])  # A4 = k(res6).
             + _desc_vec(offs["reduce_acc"], [0x180, 0x1C0, 0x180, 1])
             + _desc_engine(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                            sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                            cbeats=layout.code_beats(cols, layout.KIND_Q8),
                            bias=B_ax, rshift=0, rshift_rpre=1, rem=rem,
                            xsrc=0x200, dst=0x4000)
             + _desc_stop())

    sp = bytearray(64*1024)
    sp[0x0:0x0+64]      = acc.tobytes()
    sp[0x100:0x100+64]  = src.tobytes()
    sp[0x180:0x180+32]  = b2.tobytes()
    sp[0x200:0x200+T]   = p8.tobytes()
    sp[0x6000:0x6000+len(descs)] = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=6, cap=100000,
                  reads={"data": (0, 0x5000)}, peeks=["seq_dcount"])
    assert r.peeks["seq_dcount"] == 5
    data  = r.reads["data"]
    spill = np.frombuffer(bytes(data[0x3800:0x3800+32]), dtype="<u4")
    nrm   = np.frombuffer(bytes(data[0x3000:0x3000+32]), dtype="<i2")[:16]
    axp   = np.frombuffer(bytes(data[0x4000:0x4000+4*n_channels*cols]), dtype="<i4")
    assert np.all(spill == K1 & 0xFFFFFFFF), ("spill S7 seed", spill, K1)
    assert np.array_equal(nrm, nrm_exp), ("norm_i32 k via src=3", nrm[:8], nrm_exp[:8], k_exp)
    assert np.array_equal(axp, axpy_exp), ("axpy rshift_rpre", axp[:8], axpy_exp[:8], Rpre)

@BACKEND
def test_sequencer_mixed_list(backend):
    # Descriptor list: vec quant8 -> engine ternary matvec -> vec norm_i32 with the k
    # auto-arg from the engine absmax tracker -> stop. Results must match the individually
    # computed goldens; numerics are the same as CPU-dispatched ops by construction.
    from model import asm as vasm

    n_channels, rows_per_chan, cols, seed = 2, 4, 128, 71
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x16    = rng.integers(-32000, 32000, cols).astype("<i2")

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    x8_exp  = np.clip(bx.rne_shift(x16.astype(np.int64), 8), -127, 127)
    res_exp = bx.matvec_ternary(scales, codes, x8_exp.astype(np.int8), B, R)
    amax    = int(np.abs(res_exp).max())
    k_exp   = max(0, amax.bit_length() - 14)
    nrm_exp = np.clip(res_exp.astype(np.int64) >> k_exp, -32767, 32767)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]

    words = []
    offs  = {}
    for name in ("quant8", "norm_i32"):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())

    # Scratchpad map: x16 @ 0x0, x8 @ 0x1000, matvec out @ 0x2000, norm out @ 0x3000,
    # descriptors @ 0x6000.
    rb = layout.row_beats(cols, layout.KIND_TERNARY)
    descs = (_desc_vec(offs["quant8"], [0x0, 0, 0x1000, cols//16])
             + _desc_engine(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                            wlen=rows_per_chan*rb,
                            sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                            cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                            bias=B, rshift=R, xsrc=0x1000, dst=0x2000)
             + _desc_vec(offs["norm_i32"], [0x2000, 0, 0x3000, M//16 if M >= 16 else 1, 0],
                         srcs=[0, 0, 0, 0, 1])  # A4 = k(absmax).
             + _desc_stop())

    sp = bytearray(64*1024)
    sp[0:2*cols]              = x16.tobytes()
    sp[0x6000:0x6000+len(descs)] = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=7, cap=100000,
                  reads={"data": (0, 0x4000)}, peeks=["seq_dcount"], seed7=False)
    assert r.peeks["seq_dcount"] == 3
    data = r.reads["data"]
    out = {}
    out["x8"]  = np.frombuffer(bytes(data[0x1000:0x1000+cols]), dtype=np.int8)
    out["res"] = np.frombuffer(bytes(data[0x2000:0x2000+4*M]), dtype="<i4")
    out["nrm"] = np.frombuffer(bytes(data[0x3000:0x3000+2*16*max(1, M//16 if M >= 16 else 1)]),
                               dtype="<i2")[:M]
    assert np.array_equal(out["x8"], x8_exp.astype(np.int8)), "quant8 via sequencer"
    assert np.array_equal(out["res"], res_exp), "matvec via sequencer"
    assert np.array_equal(out["nrm"], nrm_exp.astype(np.int16)), ("norm_i32 auto-k", out["nrm"][:8], nrm_exp[:8], k_exp, amax)

@COMPILED
def test_sequencer_q8_scores_ragged(backend):
    """Scores op through the *sequencer* at the shipping channel count.

    The engine's own tests cover this op shape (test_matvec_q8_striped_interleave /
    _dual_head, both extended to 8 channels), and the sequencer tests cover mixed lists at 2
    channels - but nothing covered the scores op *driven by the sequencer* at 8. That is the
    combination the card wedges on: once T is not a multiple of the channel count and every
    channel already carries a row (rows>=1, rem>=1), the engine sits in RUN holding the
    scratchpad grant and never retires (SEQSTALL dcount=n/84 eng=301).

    T = 8 is the last working case on hardware, 9 and 10 are the failing ones.

    These all pass, so the wedge needs the full design: reproduce it in the Verilator sim
    with a tiny model sharded 8 ways (the n2 sim does not reach a failing shape):

      .venv/bin/python -m model.tinygen --channels 8
      .venv/bin/python trixium_sim.py --image models/tiny/bonsai-tiny-n8.img --build --run
    """
    from model import asm as vasm

    vec_words = vasm.assemble(vasm.PROGRAMS["quant8"]())
    n_channels, cols, seed = 8, 128, 91
    for T in (8, 9, 10, 17):
        rng    = np.random.default_rng(seed + T)
        scales = rng.uniform(0.005, 0.02, (T, cols//layout.GROUP_Q8)).astype(np.float16)
        q8w    = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        x8     = rng.integers(-127, 128, cols, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
        expected = bx.matvec_q8(scales, q8w, x8, B, R)

        rows, rem = T//n_channels, T % n_channels
        payloads = []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            spec_c = TensorSpec("kv", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                max(len(pos), 1), cols,
                                scales=scales[pos] if pos else scales[:1],
                                values=q8w[pos] if pos else q8w[:1])
            payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes() if pos else b"")
        maxlen   = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]

        rb  = layout.row_beats(cols, layout.KIND_Q8)
        # Twice back to back, as the per-head list does: one head's scores op is followed by
        # the next head's, so a handshake that only settles between *lists* would pass a
        # single-op test and hang here.
        # Each head in the real list is a vec op then the engine op, so include the vec op:
        # a list of engine ops alone never exercises the scratchpad hand-off between the
        # vector unit and the engine.
        one = (_desc_vec(0, [0x2000, 0, 0x1000, cols//16])
               + _desc_engine(kind=1, rows=rows, rem=rem, ilv=1, cols=cols, wbase=0,
                              wlen=rows*rb,
                              sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                              cbeats=layout.code_beats(cols, layout.KIND_Q8),
                              bias=B, rshift=R, xsrc=0x1000, dst=0x4000))
        descs = one + one + _desc_stop()
        sp = bytearray(64*1024)
        # int16 source for quant8; it writes the int8 x vector the engine op then consumes.
        sp[0x2000:0x2000 + 2*cols]     = (x8.astype("<i2")*256).tobytes()
        sp[0x6000:0x6000 + len(descs)] = descs

        # A list that never finishes reports the engine status: this is the SEQSTALL wedge.
        r = _run_list(backend, n_channels, payloads, sp, vec_words, count=5, cap=200000,
                      reads={"got": (0x4000, 4*T)}, seed7=False, dumps=["front_status"],
                      tag=f"T={T}_rows={rows}_rem={rem}:_sequencer_never_finished")
        got = np.frombuffer(r.reads["got"], dtype="<i4")
        assert np.array_equal(got, expected), f"T={T}: {got} != {expected}"

@COMPILED
def test_sequencer_xred_absmax_spill(backend):
    # The exact descriptor sequence seq_build_attn runs per head once cmd.xred is on:
    #     axpy(xred) -> spill(seed S7 from the ENGINE absmax) -> norm_i32(k from res6)
    # This is the mechanism that replaces the n_ch-1 reduce_acc chain, and it has three
    # separate places to go wrong that the engine unit test cannot see: the w0[4] seed
    # select, spill's SMOV s6,s7 handing the value to norm via KRES6, and the absmax
    # surviving from the AXPY's write stage to the next descriptor's dispatch.
    #
    # It exists because the full-SoC no-dual gate caught a regression here that every
    # engine-level test passed, and a 12-minute Verilator run is the wrong place to debug
    # a three-descriptor question.
    import model.asm as vasm
    n_channels, cols, T = 2, 64, 5
    rng = np.random.default_rng(97)

    ev     = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8     = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    p8     = rng.integers(1, 128, T, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B_ax   = 12 - int(e16.max())
    Rpre   = bx.axpy_rpre(int(p8.astype(np.int64).sum()))

    rows, rem = T//n_channels, T % n_channels
    payloads, parts = [], []
    for c in range(n_channels):
        pos = [t for t in range(T) if t % n_channels == c]
        if pos:
            spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                len(pos), cols, scales=scales[pos], values=v8[pos])
            payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
            parts.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B_ax, Rpre))
        else:
            payloads.append(b"")
            parts.append(np.zeros(cols, dtype=np.int32))
    maxlen   = max(len(p) for p in payloads)
    payloads = [p.ljust(maxlen, b"\0") for p in payloads]
    summed   = np.sum(np.stack(parts).astype(np.int64), axis=0).astype(np.int32)
    am_exp   = int(np.abs(summed.astype(np.int64)).max())

    words, offs = [], {}
    for name in ("norm_i32", "spill"):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())

    rb = layout.row_beats(cols, layout.KIND_Q8)
    block = lambda spill_at: (
        _desc_engine(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                     sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                     cbeats=layout.code_beats(cols, layout.KIND_Q8),
                     bias=B_ax, rshift=Rpre, rem=rem, xred=1,
                     xsrc=0x200, dst=0x4000)
        + _desc_vec(offs["spill"], [spill_at], seed_abs=1)
        + _desc_vec(offs["norm_i32"], [0x4000, 0, 0x3000, cols//16, 0],
                    srcs=[0, 0, 0, 0, 3]))       # A4 = k(res6), res6 = spill's SMOV.
    descs = block(0x3800) + block(0x3820) + _desc_stop()

    sp = bytearray(64*1024)
    sp[0x200:0x200+T]            = p8.tobytes()
    sp[0x6000:0x6000+len(descs)] = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=7, cap=200000,
                  reads={"data": (0, 0x5000)})
    data   = r.reads["data"]
    axp    = np.frombuffer(bytes(data[0x4000:0x4000+4*cols]), dtype="<i4")
    spill  = np.frombuffer(bytes(data[0x3800:0x3800+32]), dtype="<u4")
    spill2 = np.frombuffer(bytes(data[0x3820:0x3820+32]), dtype="<u4")
    assert np.array_equal(axp, summed), ("xred sum", axp[:8], summed[:8])
    assert np.all(spill == am_exp),  ("head 0 spilled absmax", spill[0], am_exp)
    assert np.all(spill2 == am_exp), ("head 1 spilled absmax", spill2[0], am_exp)

@COMPILED
def test_sequencer_kreg_capture_and_add(backend):
    # The exponent register file: an engine op captures k(absmax) into kreg[i] (w0[5],[7:6])
    # and a LATER vec descriptor adds it to an arg (w1[16+2i]). This is what lets the FFN
    # chain's "per-layer constant + an EARLIER op's k" args live in descriptors instead of
    # forcing a CPU round-trip per op - the 43% mid-band in the gap histogram.
    #
    # Shape: axpy(xred, capture kreg0) -> norm_i32 with A4 = imm(0) + kreg0. The reference k
    # is bitlen(summed absmax) - 14, i.e. exactly what SEQ_SRC_KABS would give the very next
    # op - but here it survives an arbitrary distance and combines with an immediate.
    import model.asm as vasm
    n_channels, cols, T = 2, 64, 5
    rng = np.random.default_rng(131)
    ev     = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8     = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    # Force the absmax into the LAST column, i.e. the final drained beat: the engine
    # absmax tracker lags its write stage by two cycles, so a same-edge kreg capture reads
    # a value missing exactly that beat. Random data let the race pass on luck.
    v8[:, -1] = 127
    p8     = rng.integers(1, 128, T, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B_ax   = 12 - int(e16.max())
    Rpre   = bx.axpy_rpre(int(p8.astype(np.int64).sum()))
    rows, rem = T//n_channels, T % n_channels
    payloads, parts = [], []
    for c in range(n_channels):
        pos = [t for t in range(T) if t % n_channels == c]
        spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                            len(pos), cols, scales=scales[pos], values=v8[pos])
        payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
        parts.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B_ax, Rpre))
    maxlen   = max(len(pp) for pp in payloads)
    payloads = [pp.ljust(maxlen, b"\0") for pp in payloads]
    summed   = np.sum(np.stack(parts).astype(np.int64), axis=0).astype(np.int32)
    am       = int(np.abs(summed.astype(np.int64)).max())
    k_exp    = max(0, am.bit_length() - 14)

    words, offs = [], {}
    for name in ("norm_i32",):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())
    # Golden: interpreter on the summed buffer with the expected k.
    ref_sp = bytearray(64*1024)
    ref_sp[0x4000:0x4000+4*cols] = summed.astype("<i4").tobytes()
    vasm.VecInterp(ref_sp).run(vasm.assemble(vasm.PROGRAMS["norm_i32"]()),
                               [0x4000, 0, 0x3000, cols//16, k_exp])
    nrm_exp = np.frombuffer(bytes(ref_sp[0x3000:0x3000+2*cols]), dtype="<i2")

    rb = layout.row_beats(cols, layout.KIND_Q8)
    descs = (_desc_engine(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                          sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                          cbeats=layout.code_beats(cols, layout.KIND_Q8),
                          bias=B_ax, rshift=Rpre, rem=rem, xred=1,
                          kcap=1, kreg=0,                       # capture k -> kreg0
                          xsrc=0x200, dst=0x4000)
             + _desc_vec(offs["norm_i32"], [0x4000, 0, 0x3000, cols//16, 2*k_exp],
                         kadds=[0, 0, 0, 0, 1])                 # A4 = 2k - kreg0 = k
             + _desc_vec(offs["norm_i32"], [0x4000, 0, 0x3400, cols//16, 0],
                         kadds=[0, 0, 0, 0, 1], kadd_add=1)     # w0[11]: A4 = 0 + kreg0 = k
             + _desc_stop())
    sp = bytearray(64*1024)
    sp[0x200:0x200+T]            = p8.tobytes()
    sp[0x6000:0x6000+len(descs)] = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=4, cap=200000,
                  reads={"data": (0, 0x5000)}, peeks=["seq_kreg0"])
    data = r.reads["data"]
    nrm  = np.frombuffer(bytes(data[0x3000:0x3000+2*cols]), dtype="<i2")
    nrmA = np.frombuffer(bytes(data[0x3400:0x3400+2*cols]), dtype="<i2")
    kr0  = r.peeks["seq_kreg0"]
    assert kr0 == k_exp, ("kreg0 capture", kr0, k_exp)
    assert np.array_equal(nrm, nrm_exp), ("norm via kreg", nrm[:8], nrm_exp[:8], k_exp)
    assert np.array_equal(nrmA, nrm_exp), ("norm via kreg ADD (w0[11])",
                                           nrmA[:8], nrm_exp[:8], k_exp)

@COMPILED
def test_sequencer_lut_arg(backend):
    # A5 from a scratchpad LUT indexed by imm5 + kreg (w0[12]): the one arg
    # the k/v chain could not derive in-sequencer was rmsnorm's eps, a per-layer table
    # indexed by the exponent shift the sequencer already holds in a kreg. Shape: the
    # kreg-capture test's AXPY (k -> kreg0), then rmsnorm with A5 = LUT[3 + kreg0]; the
    # same rmsnorm with the immediate the LUT holds must match bit for bit, and an index
    # past the table clamps to entry 31 rather than reading past it.
    import model.asm as vasm
    n_channels, cols, T = 2, 64, 5
    rng = np.random.default_rng(137)
    ev     = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8     = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    v8[:, -1] = 127
    p8     = rng.integers(1, 128, T, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B_ax   = 12 - int(e16.max())
    Rpre   = bx.axpy_rpre(int(p8.astype(np.int64).sum()))
    rows, rem = T//n_channels, T % n_channels
    payloads, parts = [], []
    for c in range(n_channels):
        pos = [t for t in range(T) if t % n_channels == c]
        spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                            len(pos), cols, scales=scales[pos], values=v8[pos])
        payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
        parts.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B_ax, Rpre))
    maxlen   = max(len(pp) for pp in payloads)
    payloads = [pp.ljust(maxlen, b"\0") for pp in payloads]
    summed   = np.sum(np.stack(parts).astype(np.int64), axis=0).astype(np.int32)
    am       = int(np.abs(summed.astype(np.int64)).max())
    k_exp    = max(0, am.bit_length() - 14)
    assert 3 + k_exp <= 31

    # rmsnorm input: an int16 row and a norm-weight row (E_W16 units, all ones).
    n = 128
    # Small source so eps (100..1247) is comparable with mean(m^2) and visibly changes
    # the normalization; A6 is any sane (negative) final shift - the firmware's is
    # e_w - E_NORM - 15 = -17; the test only needs the LUT and immediate runs to agree.
    x16 = rng.integers(-40, 41, n, dtype=np.int16)
    w16 = np.full(n, 1 << 13, dtype=np.int16)
    ln  = 7
    A6  = -15
    lut = [100 + 37*i for i in range(32)]             # Distinct eps per entry.
    words, offs = [], {}
    for name in ("rmsnorm",):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())
    def ref(eps):
        sp = bytearray(64*1024)
        sp[0x1000:0x1000+2*n] = x16.tobytes()
        sp[0x1800:0x1800+2*n] = w16.tobytes()
        vasm.VecInterp(sp).run(vasm.assemble(vasm.PROGRAMS["rmsnorm"]()),
                               [0x1000, 0x1800, 0x2000, n//16, -ln, eps, A6])
        return np.frombuffer(bytes(sp[0x2000:0x2000+2*n]), dtype="<i2")
    exp_lut   = ref(lut[3 + k_exp])
    exp_clamp = ref(lut[31])
    assert not np.array_equal(exp_lut, ref(lut[3]))   # The kreg term must matter.

    rb = layout.row_beats(cols, layout.KIND_Q8)
    LUT_ADDR = 0x5800
    descs = (_desc_engine(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                          sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                          cbeats=layout.code_beats(cols, layout.KIND_Q8),
                          bias=B_ax, rshift=Rpre, rem=rem, xred=1,
                          kcap=1, kreg=0, xsrc=0x200, dst=0x4000)
             + _desc_vec(offs["rmsnorm"], [0x1000, 0x1800, 0x2000, n//16, -ln, 3, A6],
                         kadds=[0, 0, 0, 0, 0, 1], lut_beat=LUT_ADDR//32)   # A5 = LUT[3+k]
             + _desc_vec(offs["rmsnorm"], [0x1000, 0x1800, 0x2400, n//16, -ln,
                                           lut[3 + k_exp], A6])             # same eps, imm
             + _desc_vec(offs["rmsnorm"], [0x1000, 0x1800, 0x2800, n//16, -ln, 40, A6],
                         kadds=[0, 0, 0, 0, 0, 1], lut_beat=LUT_ADDR//32)   # clamps to 31
             + _desc_stop())
    sp = bytearray(64*1024)
    sp[0x200:0x200+T]              = p8.tobytes()
    sp[0x1000:0x1000+2*n]          = x16.tobytes()
    sp[0x1800:0x1800+2*n]          = w16.tobytes()
    sp[LUT_ADDR:LUT_ADDR+4*32]     = b"".join(int(v).to_bytes(4, "little") for v in lut)
    sp[0x6000:0x6000+len(descs)]   = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=5, cap=200000,
                  reads={"data": (0, 0x3000)}, peeks=["seq_kreg0"])
    data = r.reads["data"]
    kr0  = r.peeks["seq_kreg0"]
    assert kr0 == k_exp, ("kreg0 capture", kr0, k_exp)
    out_lut   = np.frombuffer(bytes(data[0x2000:0x2000+2*n]), dtype="<i2")
    out_imm   = np.frombuffer(bytes(data[0x2400:0x2400+2*n]), dtype="<i2")
    out_clamp = np.frombuffer(bytes(data[0x2800:0x2800+2*n]), dtype="<i2")
    assert np.array_equal(out_lut, exp_lut), ("A5 via LUT[3+kreg0]", out_lut[:6], exp_lut[:6])
    assert np.array_equal(out_imm, exp_lut), ("same eps as immediate", out_imm[:6])
    assert np.array_equal(out_clamp, exp_clamp), ("index 40+k clamps to 31",
                                                  out_clamp[:6], exp_clamp[:6])

@COMPILED
def test_sequencer_nowait_overlap(backend):
    # NOWAIT engine descriptor (w0[9]): the following independent vec descriptor runs
    # WHILE the matvec streams; the next descriptor (norm_i32 with A4 = KABS) trips the
    # hazard interlock and waits for the in-flight op. All three results bit-exact,
    # real overlap measured, and the interlock's k matches the serial golden.
    from model import asm as vasm

    n_channels, rows_per_chan, cols, seed = 2, 32, 128, 57
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
    k_exp   = max(0, int(np.abs(res_exp).max()).bit_length() - 14)
    nrm_exp = np.clip(res_exp.astype(np.int64) >> k_exp, -32767, 32767).astype("<i2")

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]

    words, offs = [], {}
    for name in ("resid_add", "norm_i32"):
        offs[name] = len(words)
        words += vasm.assemble(vasm.PROGRAMS[name]())

    nv = 1024
    vec_args = [0x4000, 0x4800, 0x5000, nv//16, -1, -3]
    rb = layout.row_beats(cols, layout.KIND_TERNARY)
    descs = (_desc_engine(nowait=1, kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                          wlen=rows_per_chan*rb,
                          sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                          cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                          bias=B, rshift=R, xsrc=0x1000, dst=0x2000)
             + _desc_vec(offs["resid_add"], vec_args)           # Runs UNDER the matvec.
             + _desc_vec(offs["norm_i32"], [0x2000, 0, 0x3800, M//16, k_exp],
                         barrier=1)                             # w0[10]: explicit barrier.
             + _desc_vec(offs["norm_i32"], [0x2000, 0, 0x3000, M//16, 0],
                         srcs=[0, 0, 0, 0, 1])                  # KABS: interlock waits.
             + _desc_stop())

    sp = bytearray(64*1024)
    sp[0x1000:0x1000 + cols]  = x8.tobytes()
    sp[0x4000:0x4000 + 2*nv]  = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x4800:0x4800 + 2*nv]  = rng.integers(-30000, 30000, nv).astype("<i2").tobytes()
    sp[0x6000:0x6000 + len(descs)] = descs
    mirror = bytearray(sp)
    vasm.VecInterp(mirror).run(vasm.assemble(vasm.PROGRAMS["resid_add"]()),
                               list(vec_args))

    r = _run_list(backend, n_channels, payloads, sp, words, count=5, cap=100000,
                  reads={"res": (0x2000, 4*M), "nrm": (0x3000, 2*M), "nrmb": (0x3800, 2*M),
                         "vec": (0x5000, 2*nv)},
                  peeks=["seq_dcount", "cyc_both", "gw_ovf"])
    out = {}
    out["dcount"] = r.peeks["seq_dcount"]
    out["both"]   = r.peeks["cyc_both"]
    out["ovf"]    = r.peeks["gw_ovf"]
    out["res"]    = np.frombuffer(r.reads["res"], dtype="<i4")
    out["nrm"]    = np.frombuffer(r.reads["nrm"], dtype="<i2")
    out["nrmb"]   = np.frombuffer(r.reads["nrmb"], dtype="<i2")
    out["vec"]    = r.reads["vec"]
    assert out["ovf"] == 0, "gather FIFO overflow"
    assert out["dcount"] == 4, out["dcount"]
    assert np.array_equal(out["res"], res_exp), (out["res"][:8], res_exp[:8])
    assert np.array_equal(out["nrm"], nrm_exp), (out["nrm"][:8], nrm_exp[:8], k_exp)
    assert np.array_equal(out["nrmb"], nrm_exp), "w0[10] barrier saw partial results"
    assert out["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert out["both"] > 50, f"no overlap through the sequencer (cyc_both={out['both']})"

@COMPILED
def test_sequencer_kadd_interlock_precise(backend):
    # The kadd hazard interlock is per-REGISTER: matvec A captures k -> kreg0 (serial),
    # matvec B is NOWAIT and captures k -> kreg1. A norm reading kreg0 must dispatch UNDER
    # B's stream (real overlap), while a norm reading kreg1 must wait for B's capture
    # (done+2) and see the right k. The register-blind interlock serialized the FFN
    # list's norm-under-stream on the card (13.60M vs 13.45M cycles/token, 2026-08-29).
    from model import asm as vasm

    n_channels, rows_per_chan, cols = 2, 32, 128
    rng = np.random.default_rng(211)
    M   = n_channels*rows_per_chan
    G   = cols//layout.GROUP_TERNARY
    x8  = rng.integers(-127, 128, cols).astype(np.int8)

    def build(seed_scale, tag):
        scales = rng.uniform(seed_scale, seed_scale*4, (M, G)).astype(np.float16)
        codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B)
        res = bx.matvec_ternary(scales, codes, x8, B, R)
        k   = max(0, int(np.abs(res).max()).bit_length() - 14)
        nrm = np.clip(res.astype(np.int64) >> k, -32767, 32767).astype("<i2")
        spec = TensorSpec(tag, layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                          scales=scales, values=codes)
        pay = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
               for c in range(n_channels)]
        return B, R, k, nrm, pay

    # Two magnitudes so the two k's differ - a norm on the wrong register is visible.
    Ba, Ra, ka, nrm_a, pay_a = build(0.01, "a")
    Bb, Rb, kb, nrm_b, pay_b = build(0.5,  "b")
    assert ka != kb, (ka, kb)
    rb = layout.row_beats(cols, layout.KIND_TERNARY)
    off_b = len(pay_a[0])                     # B's weights follow A's in every channel.
    assert off_b % 32 == 0
    payloads = [pay_a[c] + pay_b[c] for c in range(n_channels)]

    words = vasm.assemble(vasm.PROGRAMS["norm_i32"]())
    common = dict(kind=0, rows=rows_per_chan, cols=cols, wlen=rows_per_chan*rb,
                  sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                  cbeats=layout.code_beats(cols, layout.KIND_TERNARY), xsrc=0x1000)
    descs = (_desc_engine(wbase=0, bias=Ba, rshift=Ra, dst=0x2000,
                          kcap=1, kreg=0, **common)              # A: k -> kreg0 (serial)
             + _desc_engine(wbase=off_b, bias=Bb, rshift=Rb, dst=0x2800,
                            kcap=1, kreg=1, nowait=1, **common)  # B: k -> kreg1, NOWAIT
             + _desc_vec(0, [0x2000, 0, 0x3000, M//16, 0],
                         kadds=[0, 0, 0, 0, 1], kadd_add=1)      # kreg0: runs UNDER B
             + _desc_vec(0, [0x2800, 0, 0x3800, M//16, 0],
                         kadds=[0, 0, 0, 0, 2], kadd_add=1)      # kreg1: waits for B
             + _desc_stop())
    sp = bytearray(64*1024)
    sp[0x1000:0x1000 + cols] = x8.tobytes()
    sp[0x6000:0x6000 + len(descs)] = descs

    r = _run_list(backend, n_channels, payloads, sp, words, count=5, cap=200000,
                  reads={"na": (0x3000, 2*M), "nb": (0x3800, 2*M)},
                  peeks=["cyc_both", "seq_kreg0", "seq_kreg1"])
    out = {}
    out["both"] = r.peeks["cyc_both"]
    out["k0"]   = r.peeks["seq_kreg0"]
    out["k1"]   = r.peeks["seq_kreg1"]
    out["na"]   = np.frombuffer(r.reads["na"], dtype="<i2")
    out["nb"]   = np.frombuffer(r.reads["nb"], dtype="<i2")
    assert (out["k0"], out["k1"]) == (ka, kb), (out["k0"], out["k1"], ka, kb)
    assert np.array_equal(out["na"], nrm_a), "norm via kreg0 (under the stream)"
    assert np.array_equal(out["nb"], nrm_b), "norm via kreg1 (after the capture)"
    # The norm under the stream is short (V1 pipelined unit: ~8 cy/beat), so the overlap
    # window is small in absolute cycles; > 10 is well above zero and below any run.
    assert out["both"] > 10, f"kreg0 consumer did not overlap B (cyc_both={out['both']})"

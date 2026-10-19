#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/engine.py: MatVecEngine end to end (matvec, fetch, store, AXPY, argmax, channel
offsets) against model.bitexact, on memory channel models.

Every test runs as a common.verilator.Script on the compiled engine (C++ AXI slaves); the
slim-engine test also runs, in the slow tier, on Migen's simulator with HBMChannelModel
channels: the same golden checks on Migen's own semantics."""

import os

import numpy as np
import pytest

from model import layout, bitexact as bx
from model.bons import TensorSpec, encode_rows
from model.layout import HBM_CHAN_STRIDE, HBM_WEIGHT_BASE

from common.engine import CMD_FIELDS, EngineBench, csr_write, submit
from common.verilator import Script

# Backends -----------------------------------------------------------------------------------------
# The compiled engine in the fast tier; Migen's simulator (minutes per test) in the slow one
# for the representative test only.

BACKEND = pytest.mark.parametrize("backend", ["verilator", pytest.param("migen", marks=pytest.mark.slow)])
COMPILED = pytest.mark.parametrize("backend", ["verilator"])

# MatVec / Store / AXPY / Argmax -------------------------------------------------------------------
# MatVecEngine end-to-end unit test: BONS-encoded weight shards in memory channel models,
# x8 in the scratchpad, engine command -> results in scratchpad, compared BIT-EXACTLY against
# model.bitexact.matvec_ternary()/matvec_q8().
#
# latency/stall_rate may be a per-channel list: real HBM channels do not respond in lockstep,
# and several engine bugs only appear once the readers desynchronize.

def _i32(data):
    return np.frombuffer(data, dtype="<i4").astype(np.int64)

def _run_engine(backend, n_channels, payloads, sp, cmd, n_results, dst, latency=8, stall_rate=0,
                xbuf_groups=32, reorder_ids=0, absmax=False, trit5=False, axpy=True):
    """Results array, or (results, engine.absmax) when absmax=True."""
    s = Script()
    submit(s, cmd)
    s.wait("done", cap=50000, tag="engine")
    s.step(2)
    s.peek("error")
    s.peek("absmax")
    s.read("sp", dst, 4*n_results, "res")
    bench = EngineBench(n_channels, xbuf_groups=xbuf_groups, **({"trit5": True} if trit5 else {}),
                        **({"axpy": False} if not axpy else {}))
    r = bench.run(backend, s, sp=sp, payloads=payloads,
        latency=latency, stall_rate=stall_rate, reorder_ids=reorder_ids)
    assert r.peeks["error"] == 0
    arr = _i32(r.reads["res"])
    return (arr, r.peeks["absmax"]) if absmax else arr

def _case_ternary(backend, n_channels, rows_per_chan, cols, latency, stall_rate, seed,
                  kind=layout.KIND_TERNARY, trit5=False, xbuf_groups=32, axpy=True):
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    spec.kind = kind # KIND_TERNARY5: the same tensor stored 5 trits per byte.
    payloads = []
    for c in range(n_channels):
        row0, nrows = layout.shard_rows(M, n_channels, c)
        payloads.append(encode_rows(spec, row0, nrows).tobytes())

    xsrc, dst = 0, 0x4000
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, kind),
               sbeats=layout.scale_beats(cols, kind),
               cbeats=layout.code_beats(cols, kind),
               bias=B, rshift=R, xsrc=xsrc, dst=dst)
    if trit5:
        cmd["pk"] = int(kind == layout.KIND_TERNARY5)
    got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, M, dst, latency, stall_rate,
                      xbuf_groups=xbuf_groups, trit5=trit5, axpy=axpy)
    assert np.array_equal(got, expected), f"N={n_channels} rows={rows_per_chan} cols={cols}"

@COMPILED
def test_matvec_ternary_n1(backend):
    _case_ternary(backend, n_channels=1, rows_per_chan=8, cols=256, latency=8, stall_rate=0, seed=1)

@COMPILED
def test_matvec_ternary_n2(backend):
    _case_ternary(backend, n_channels=2, rows_per_chan=8, cols=256, latency=8, stall_rate=0, seed=2)

@COMPILED
def test_matvec_ternary_n4(backend):
    _case_ternary(backend, n_channels=4, rows_per_chan=4, cols=256, latency=8, stall_rate=3, seed=7)

@COMPILED
def test_matvec_ternary_n4_gather_pressure(backend):
    # 4 channels x 2-beat rows demand 2 results/cycle against the 1/cycle gather drain:
    # pipes must throttle beat intake on result-FIFO headroom instead of dropping results
    # (without backpressure out_fifo overflows and the op never completes).
    _case_ternary(backend, n_channels=4, rows_per_chan=48, cols=128, latency=8, stall_rate=0, seed=9)

# Trit5 (KIND_TERNARY5 through the unpacker) ------------------------------------------------------

T5 = layout.KIND_TERNARY5

@COMPILED
def test_matvec_trit5_aligned(backend):
    # K = 2048: 16 groups in 13 memory beats, no row padding (the Qwen3 widths).
    _case_ternary(backend, n_channels=2, rows_per_chan=6, cols=2048, latency=8, stall_rate=0, seed=21,
                  kind=T5, trit5=True)

@COMPILED
def test_matvec_trit5_no_axpy(backend):
    # The hybrid builds' pipe: no AXPY datapath (ChannelPipe axpy=False), the ternary and
    # trit5 matvecs unchanged bit for bit.
    _case_ternary(backend, n_channels=2, rows_per_chan=6, cols=2048, latency=8, stall_rate=2, seed=23,
                  kind=T5, trit5=True, axpy=False)

# Pipe island: the pipes (and unpackers) in their own clock domain ---------------------------------

def _trit5_case(n_channels, rows_per_chan, cols, seed):
    """One trit5 matvec: per-channel shards, x8, the command's fields and its exact result."""
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    spec.kind = T5
    shards = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
              for c in range(n_channels)]
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, T5),
               sbeats=layout.scale_beats(cols, T5), cbeats=layout.code_beats(cols, T5),
               bias=B, rshift=R, pk=1)
    return shards, x8, cmd, expected

@pytest.mark.parametrize("phase", [0] + [pytest.param(p, marks=pytest.mark.slow) for p in range(1, 5)])
def test_matvec_trit5_island(phase):
    """PipeIsland (gateware/core/island.py): the pipes and unpackers in their own clock domain,
    on the Migen backend with sys at 10 ns and the pipe clock at 8 ns (the YPCB's 100/125 MHz
    ratio: the alignments repeat every 4 sys cycles, `phase` walks them). Two trit5 tensors
    back to back, a FETCH between (it pulses the pipes' start as well), the first tensor again:
    every result exact, no sticky error."""
    n = 2
    a = _trit5_case(n, 6, 2048, 31)
    b = _trit5_case(n, 4, 1024, 32)
    # Memory: tensor A at 0, tensor B behind it on a beat boundary, in every channel.
    off_b = (max(len(sh) for sh in a[0]) + 31)//32*32
    payloads = [sa.ljust(off_b, b"\0") + sb for sa, sb in zip(a[0], b[0])]
    # Scratchpad: x8 of A at 0, of B at 0x1000; results at 0x4000/0x5000/0x6000, the fetch at 0x7000.
    sp    = a[1].tobytes().ljust(0x1000, b"\0") + b[1].tobytes()
    cmd_a = dict(a[2], xsrc=0, dst=0x4000)
    cmd_b = dict(b[2], wbase=off_b, xsrc=0x1000, dst=0x5000)
    s = Script()
    s.step(phase)
    submit(s, cmd_a)
    s.wait("done", cap=50000, tag="a")
    submit(s, cmd_b)                                       # Back to back.
    s.wait("done", cap=50000, tag="b")
    submit(s, dict(kind=2, chan=1, wbase=0, wlen=4, dst=0x7000), fields=CMD_FIELDS)
    s.wait("done", cap=5000, tag="fetch")
    s.step(phase)
    submit(s, dict(cmd_a, dst=0x6000))
    s.wait("done", cap=50000, tag="a2")
    s.step(2)
    s.peek("error")
    s.read("sp", 0x4000, 4*len(a[3]), "ra")
    s.read("sp", 0x5000, 4*len(b[3]), "rb")
    s.read("sp", 0x6000, 4*len(a[3]), "ra2")
    s.read("sp", 0x7000, 128, "fetched")
    bench = EngineBench(n, xbuf_groups=32, trit5=True, axpy=False, pipe_cd="peng")
    r = bench.run("migen", s, sp=sp, payloads=payloads, clocks={"sys": 10, "peng": 8})
    assert r.peeks["error"] == 0
    assert np.array_equal(_i32(r.reads["ra"]),  a[3]), "tensor A"
    assert np.array_equal(_i32(r.reads["rb"]),  b[3]), "tensor B after A"
    assert np.array_equal(_i32(r.reads["ra2"]), a[3]), "tensor A after the fetch"
    assert r.reads["fetched"] == payloads[1][:128]

@COMPILED
def test_matvec_trit5_padded_stalls(backend):
    # K = 384: 3 groups in 78 bytes, each row padded to a whole beat; stalls desynchronize the
    # channels.
    _case_ternary(backend, n_channels=4, rows_per_chan=8, cols=384, latency=8, stall_rate=3, seed=22,
                  kind=T5, trit5=True)

@COMPILED
def test_matvec_trit5_gather_pressure(backend):
    # One group per row (26 bytes in a padded beat): results faster than the gather drains.
    _case_ternary(backend, n_channels=4, rows_per_chan=48, cols=128, latency=8, stall_rate=0, seed=23,
                  kind=T5, trit5=True)

@COMPILED
def test_matvec_trit5_engine_unpacked_op(backend):
    # A trit5 engine still runs 2-bit ternary ops (pk = 0: the unpacker passes beats through).
    _case_ternary(backend, n_channels=2, rows_per_chan=8, cols=256, latency=8, stall_rate=2, seed=24,
                  trit5=True)

@BACKEND
def test_matvec_slim_no_dual(backend):
    # Ternary + Q8 + striped/ilv scores + AXPY on one engine configuration.
    _case_ternary(backend, n_channels=4, rows_per_chan=8, cols=256, latency=8, stall_rate=2,
                  seed=31)

    cols, T, n_channels, seed = 128, 6, 2, 33
    rng    = np.random.default_rng(seed)
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
        spec_c = TensorSpec("kv", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8, len(pos), cols,
                            scales=scales[pos], values=q8w[pos])
        payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
    maxlen = max(len(p) for p in payloads)
    payloads = [p.ljust(maxlen, b"\0") for p in payloads]
    rb  = layout.row_beats(cols, layout.KIND_Q8)
    cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
    got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, T, 0x4000)
    assert np.array_equal(got, expected), "slim ilv scores"

    # AXPY (single head) on the slim engine.
    cols = 64
    ev  = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    p8  = rng.integers(1, 128, T, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B    = 12 - int(e16.max())
    Rpre = bx.axpy_rpre(int(p8.astype(np.int64).sum()))
    payloads, expected = [], []
    for c in range(n_channels):
        pos = [t for t in range(T) if t % n_channels == c]
        spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8, len(pos), cols,
                            scales=scales[pos], values=v8[pos])
        payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
        expected.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B, Rpre))
    maxlen = max(len(p) for p in payloads)
    payloads = [p.ljust(maxlen, b"\0") for p in payloads]
    expected = np.concatenate(expected)
    rb  = layout.row_beats(cols, layout.KIND_Q8)
    cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem)
    got = _run_engine(backend, n_channels, payloads, p8.tobytes().ljust(32, b"\0"), cmd,
                      n_channels*cols, 0x4000)
    assert np.array_equal(got, expected), "slim axpy"

@COMPILED
def test_matvec_ternary_multi_scaleword(backend):
    # cols=4096: 32 groups -> 2 scale beats/row, exercises scale-word switching + row parity.
    _case_ternary(backend, n_channels=2, rows_per_chan=2, cols=4096, latency=16, stall_rate=5, seed=3)

@COMPILED
def test_matvec_q8(backend):
    n_channels, rows_per_chan, cols, seed = 2, 4, 128, 4
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_Q8
    scales = rng.uniform(0.005, 0.02, (M, G)).astype(np.float16)
    q8w    = rng.integers(-127, 128, (M, cols), dtype=np.int8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
    expected = bx.matvec_q8(scales, q8w, x8, B, R)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8, M, cols,
                      scales=scales, values=q8w)
    payloads = []
    for c in range(n_channels):
        row0, nrows = layout.shard_rows(M, n_channels, c)
        payloads.append(encode_rows(spec, row0, nrows).tobytes())

    xsrc, dst = 0, 0x4000
    cmd = dict(kind=1, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_Q8),
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=R, xsrc=xsrc, dst=dst)
    got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, M, dst)
    assert np.array_equal(got, expected)

@pytest.mark.slow  # Needs a converted 1.7B image.
@COMPILED
def test_matvec_real_1p7b_slice(backend):
    # Real Ternary-Bonsai-1.7B layer-0 attn_q data (real fp16 scale distribution, real code
    # statistics) through the engine, bit-exact vs the golden model. Uses a 64-rows-per-
    # channel slice of the N=2 layer-0 image so the memory model init stays small.
    from model.bons import BONSReader
    img_path = "models/bons/bonsai-1p7b-l1-n2.img"
    if not os.path.exists(img_path):
        pytest.skip("1.7B layer-0 image not converted")
    img  = BONSReader(img_path)
    name = "blk.0.attn_q.weight"
    e    = img.by_name[name]
    scales, codes = img.tensor(name)

    rows_per_chan = 64
    cols          = e["cols"] # 2048.
    rb            = layout.row_beats(cols, layout.KIND_TERNARY)
    rpc_full      = e["rows"]//2

    # Shard slices: channel c rows [c*rpc_full, c*rpc_full + rows_per_chan).
    payloads, sel_rows = [], []
    for c in range(2):
        raw = bytes(img.tensor_channel_bytes(name, c))[:rows_per_chan*rb*32]
        payloads.append(raw)
        sel_rows.append(slice(c*rpc_full, c*rpc_full + rows_per_chan))

    rng = np.random.default_rng(77)
    x8  = rng.integers(-127, 128, cols, dtype=np.int8)

    scl = np.concatenate([scales[s] for s in sel_rows], axis=0)
    cds = np.concatenate([codes[s]  for s in sel_rows], axis=0)
    expected = bx.matvec_ternary(scl, cds, x8, e["bias"], e["rshift"])

    dst = 0x8000
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=e["bias"], rshift=e["rshift"], xsrc=0, dst=dst)
    got = _run_engine(backend, 2, payloads, x8.tobytes(), cmd, 2*rows_per_chan, dst,
                      latency=16, stall_rate=4)
    assert np.array_equal(got, expected)

@COMPILED
def test_engine_absmax_tracker(backend):
    # The absmax CSR input signal must equal max(|results|) after a matvec.
    rng    = np.random.default_rng(31)
    M, K   = 16, 256
    G      = K//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, K), dtype=np.uint8)
    x8     = rng.integers(-127, 128, K, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, K,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, 2, c)).tobytes() for c in range(2)]
    cmd = dict(kind=0, rows=M//2, cols=K, wbase=0,
               wlen=(M//2)*layout.row_beats(K, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(K, layout.KIND_TERNARY),
               cbeats=layout.code_beats(K, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)
    s = Script()
    submit(s, cmd)
    s.wait("done", cap=20000, tag="done", fatal=False)
    s.step()
    s.peek("absmax")
    r = EngineBench(2).run(backend, s, sp=x8.tobytes(), payloads=payloads)
    assert r.peeks["absmax"] == int(np.abs(expected).max())

@COMPILED
def test_engine_csr_single_run_per_submit(backend):
    # Regression: submitting while the engine is idle used to latch `pending` even though
    # the command was accepted on the submit cycle itself, replaying every op once the
    # engine returned to idle (identical results, but the replay's command accept reset
    # the absmax tracker mid-read and doubled the stream time).
    n_channels, rows_per_chan, cols, seed = 2, 8, 256, 11
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = []
    for c in range(n_channels):
        row0, nrows = layout.shard_rows(M, n_channels, c)
        payloads.append(encode_rows(spec, row0, nrows).tobytes())

    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)

    s = Script()
    for k, v in cmd.items():
        s.set(f"front_{k}", v)
    s.step()
    csr_write(s, "front_submit", 1)
    s.step()
    # Every done over the next 20000 cycles; absmax as it stands at the first one.
    s.count("done", "dones")
    s.wait("done", cap=20000, tag="first", fatal=False)
    s.peek("absmax")
    s.step(20000)
    s.count_end("dones")
    s.peek("front_pending")
    r = EngineBench(n_channels, front=True).run(backend, s, sp=x8.tobytes(), payloads=payloads)
    assert r.counts["dones"] == 1, f"engine ran {r.counts['dones']} times for one submit"
    assert r.peeks["absmax"] == int(np.abs(expected).max())
    assert r.peeks["front_pending"] == 0

def _store_fetch_wait(s, tag, cap=50000):
    s.wait("done", cap=cap, tag=tag)
    s.step()

@COMPILED
@pytest.mark.parametrize("chan_stride", [0, 0x1000])
def test_engine_store_roundtrip(chan_stride, backend):
    # kind-3 STORE (scratchpad -> HBM via the loader's writers), then FETCH back: the KV
    # append datapath. Also checks the done handshake orders after the B drain.
    #
    # chan_stride is parametrised because the built SoC uses a non-zero one (each channel owns
    # its own HBM window - the AXI switch makes the address, not the port, select the
    # pseudo-channel) while this test only ever ran at 0. On hardware the KV STORE wedges the
    # engine mid-token, and an untested address path is the first suspect.
    n_channels = 2
    rng   = np.random.default_rng(17)
    src   = rng.integers(0, 256, 512, dtype=np.uint8)  # 16 beats.
    beats = len(src)//32

    s = Script()
    # Store 16 beats from SP 0x0 to channel 1 @ 0x2000.
    submit(s, dict(kind=3, chan=1, wbase=0x2000, wlen=beats, xsrc=0), fields=CMD_FIELDS)
    _store_fetch_wait(s, "store")
    # Fetch them back to SP 0x4000.
    submit(s, dict(kind=2, chan=1, wbase=0x2000, wlen=beats, dst=0x4000), fields=CMD_FIELDS)
    _store_fetch_wait(s, "fetch")
    s.read("sp", 0x4000, len(src), "got")
    bench = EngineBench(n_channels, chan_stride=chan_stride, loader=True)
    r = bench.run(backend, s, sp=src.tobytes())
    assert r.reads["got"] == src.tobytes(), "store/fetch roundtrip mismatch"

@COMPILED
def test_matvec_q8_striped_interleave(backend):
    # Scores-op shape: position-striped rows (uneven per channel via rem) + interleaved
    # (position-major) gather, Q8 kind - each row carries its own scale, per-position KV.
    cols, seed = 128, 23
    # n_channels=8 is the shipping configuration and was missing here: the card wedges on
    # this exact op once a channel carries an extra KV row (rows>=1 with rem>=1).
    for n_channels, T in [(2, 1), (2, 2), (2, 5), (2, 8),
                          (4, 1), (4, 3), (4, 6), (4, 9),
                          (8, 1), (8, 7), (8, 8), (8, 9), (8, 10), (8, 16), (8, 17)]:
        # positions (total rows); T < N -> rows=0 on some channels.
        rng    = np.random.default_rng(seed + T)
        scales = rng.uniform(0.005, 0.02, (T, cols//layout.GROUP_Q8)).astype(np.float16)
        q8w    = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        x8     = rng.integers(-127, 128, cols, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
        expected = bx.matvec_q8(scales, q8w, x8, B, R)  # Position order.

        rows, rem = T//n_channels, T % n_channels
        payloads = []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]  # local row r <-> t = r*N + c
            spec_c = TensorSpec("kv", layout.ROLE_ATTN_K if hasattr(layout, "ROLE_ATTN_K")
                                else layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                max(len(pos), 1), cols,
                                scales=scales[pos] if pos else scales[:1],
                                values=q8w[pos] if pos else q8w[:1])
            payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes() if pos else b"")

        # Pad payloads to equal length for the DUT init (readers stop at their own len).
        maxlen = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]

        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
        got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, T, 0x4000)
        assert np.array_equal(got, expected), f"T={T}: {got} != {expected}"

@COMPILED
def test_engine_axpy(backend):
    # Weighted-V: kind-4 op accumulates p8-weighted V rows per lane; each channel emits
    # cols partial sums (chan-major), bit-exact vs bx.axpy_rows on its position stripe.
    cols, seed = 64, 41
    # 8 channels is the shipping configuration and was untested here; T=9,10,17 are the
    # ragged cases (rows>=1, rem>=1) the card wedges on.
    for n_channels, T in [(2, 1), (2, 3), (2, 8), (4, 1), (4, 5), (4, 9),
                          (8, 8), (8, 9), (8, 10), (8, 17)]:
        rng = np.random.default_rng(seed + T)
        ev  = rng.integers(-9, -4, T)
        scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
        v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        p8  = rng.integers(0, 128, T, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B    = 12 - int(e16.max())
        Rpre = bx.axpy_rpre(int(p8.astype(np.int64).sum()))

        rows, rem = T//n_channels, T % n_channels
        payloads, expected = [], []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            if pos:
                spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                    len(pos), cols, scales=scales[pos], values=v8[pos])
                payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
                expected.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B, Rpre))
            else:
                payloads.append(b"")
                expected.append(np.zeros(cols, dtype=np.int32))
        maxlen = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]
        expected = np.concatenate(expected)

        x_init = p8.tobytes().ljust(64, b"\0")  # Probs vector, position-major.
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem)
        got = _run_engine(backend, n_channels, payloads, x_init, cmd, n_channels*cols, 0x4000)
        assert np.array_equal(got, expected), f"T={T}"

@COMPILED
def test_engine_axpy_xred(backend):
    # cmd.xred: the engine sums the per-channel AXPY partials itself and writes ONE
    # cols-long vector, replacing the n_channels-1 reduce_acc calls the firmware runs per
    # head. The reference is the exact int32-wrapping sum of the partials that the
    # unmodified op produces - the same arithmetic the vector unit's VADD does - so this
    # pins bit-exactness, not just plausibility.
    #
    # Covers the ragged cases (rem != 0) on purpose: `rem` changes how many POSITIONS each
    # channel accumulates, and the lockstep drain is only correct because it does NOT change
    # how many beats each channel emits. If that ever stops holding, T=9/10/17 catch it.
    cols, seed = 64, 41
    # (2,1), (2,2), (8,3): shapes where some channel gets ZERO positions - rows=0,
    # wlen=0, it streams nothing. The tiny sim model starts at T=1 on 2 channels and
    # hits this on the very first token, so it is not an exotic case; the original
    # case list missed it because every entry gave each channel at least one row.
    for n_channels, T in [(2, 1), (2, 2), (8, 3), (2, 3), (4, 5), (4, 9),
                          (8, 8), (8, 9), (8, 10), (8, 17)]:
        rng = np.random.default_rng(seed + T)
        ev  = rng.integers(-9, -4, T)
        scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
        v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        p8  = rng.integers(0, 128, T, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B    = 12 - int(e16.max())
        Rpre = bx.axpy_rpre(int(p8.astype(np.int64).sum()))

        rows, rem = T//n_channels, T % n_channels
        payloads, parts = [], []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            if pos:
                spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                    len(pos), cols, scales=scales[pos], values=v8[pos])
                payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
                parts.append(bx.axpy_rows(scales[pos], v8[pos], p8[pos], B, Rpre))
            else:
                payloads.append(b"")
                parts.append(np.zeros(cols, dtype=np.int32))
        maxlen = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]
        # int32 wrap, exactly as VADD accumulates it.
        expected = np.sum(np.stack(parts).astype(np.int64), axis=0).astype(np.int32)

        x_init = p8.tobytes().ljust(64, b"\0")
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem, xred=1)
        got = _run_engine(backend, n_channels, payloads, x_init, cmd, cols, 0x4000, absmax=True)
        assert np.array_equal(got[0].astype(np.int32), expected), \
            f"xred N={n_channels} T={T}"
        # The engine absmax tracker must hold the absmax of the SUMMED vector, because that
        # is what norm_i32's k comes from once the reduce_acc chain is gone (SEQ_SRC_KABS).
        # This is only true with xred: without it the tracker sees the per-channel partials,
        # whose absmax is not the absmax of their sum.
        assert got[1] == int(np.abs(expected.astype(np.int64)).max()), \
            f"xred absmax N={n_channels} T={T}"

@COMPILED
def test_engine_argmax_tap_and_suppress(backend):
    # Argmax tap: value + global index (ties to smallest index) over a suppressed-write
    # ternary op - the scratchpad result region must stay untouched.
    n_channels, rows_per_chan, cols, seed = 2, 16, 256, 51
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    # Force a tie for the max between two rows (row 5 copies row 20's content).
    codes[5]  = codes[20]
    scales[5] = scales[20]

    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    exp_idx  = int(np.argmax(expected))
    exp_val  = int(expected[exp_idx])

    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = []
    for c in range(n_channels):
        row0, nrows = layout.shard_rows(M, n_channels, c)
        payloads.append(encode_rows(spec, row0, nrows).tobytes())

    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000, supw=1)
    s = Script()
    submit(s, cmd)
    s.wait("done", cap=50000, tag="timeout")
    s.step(2)
    s.peek("argmax_val")
    s.peek("argmax_idx")
    # Result region untouched (suppressed): scratchpad still zero there.
    s.read("sp", 0x4000, 8*32, "region")
    r = EngineBench(n_channels).run(backend, s, sp=x8.tobytes(), payloads=payloads)
    val = r.peek("argmax_val", signed_width=32)
    assert val == exp_val, (val, exp_val)
    assert r.peeks["argmax_idx"] == exp_idx, (r.peeks["argmax_idx"], exp_idx)
    assert r.reads["region"] == bytes(8*32)

@COMPILED
def test_engine_back_to_back_scores_axpy_ragged(backend):
    """The real per-head chain: scores -> AXPY -> scores, back to back, 8 channels, ragged.

    Each of those ops passes on its own. This runs them consecutively the way one head's
    attention does, because ISSUE only waits on readers[0].sink.ready before moving to RUN -
    "all readers are idle here" - and with rem>=1 the readers are handed *different* lengths,
    so they do not finish together. If a reader is still busy when the next op issues, its
    command is dropped, its pipe never produces results, and the engine sits in RUN forever.
    """
    cols, seed = 64, 131
    n_channels = 8
    for T in (9, 17):
        rng    = np.random.default_rng(seed + T)
        ev     = rng.integers(-9, -4, T)
        scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
        w8     = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        x8     = rng.integers(-127, 128, cols, dtype=np.int8)
        p8     = rng.integers(0, 128, T, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        Bs   = bx.choose_scale_bias(e16)
        Rs   = bx.choose_result_shift(scales, Bs, gs_max=bx.GS_MAX_Q8)
        Ba   = 12 - int(e16.max())
        Rpre = bx.axpy_rpre(int(p8.astype(np.int64).sum()))
        exp_scores = bx.matvec_q8(scales, w8, x8, Bs, Rs)
        exp_axpy   = []

        rows, rem = T//n_channels, T % n_channels
        payloads = []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            if pos:
                spec_c = TensorSpec("kv", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                    len(pos), cols, scales=scales[pos], values=w8[pos])
                payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
                exp_axpy.append(bx.axpy_rows(scales[pos], w8[pos], p8[pos], Ba, Rpre))
            else:
                payloads.append(b"")
                exp_axpy.append(np.zeros(cols, dtype=np.int32))
        maxlen   = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]
        exp_axpy = np.concatenate(exp_axpy)

        rb = layout.row_beats(cols, layout.KIND_Q8)
        sb = layout.scale_beats(cols, layout.KIND_Q8)
        cb = layout.code_beats(cols, layout.KIND_Q8)
        # x holds the score vector at 0 and the probs at 0x100, as the head chain does.
        x_init = x8.tobytes().ljust(0x100, b"\0") + p8.tobytes()
        # Per-channel latency/stalls so the readers finish at different times, which is what
        # real HBM does. A reader only accepts a command while idle (sink.ready = ~active),
        # so one still draining when the next op issues is the failure mode under test.
        lat = [4, 20, 6, 28, 8, 32, 10, 36][:n_channels]
        stl = [0, 2, 0, 3, 0, 4, 0, 5][:n_channels]

        sc_cmd = dict(kind=1, rows=rows, rem=rem, ilv=1, cols=cols, wbase=0, wlen=rows*rb,
                      sbeats=sb, cbeats=cb, bias=Bs, rshift=Rs, xsrc=0, dst=0x4000)
        ax_cmd = dict(kind=4, rows=rows, rem=rem, cols=cols, wbase=0, wlen=rows*rb,
                      sbeats=sb, cbeats=cb, bias=Ba, rshift=Rpre, xsrc=0x100, dst=0x5000)
        fetch_cmd = dict(kind=2, rows=0, rem=0, cols=cols, wbase=0, wlen=4,
                         sbeats=sb, cbeats=cb, bias=0, rshift=0, xsrc=0, dst=0x7000,
                         chan=n_channels - 1)  # last channel, so reader 0 sits out

        s = Script()
        # If an op never completes, say which readers are still holding it and which pipes
        # hold a result the gather will not take: the card showed every channel delivering
        # exactly what it was commanded, so a match here means the same failure. `wide` is in
        # the merge condition (merge_mask requires ~wide), so a stale wide flag from a
        # previous AXPY would strand exactly this.
        s.dump("dbg_result_cnt", "dbg_mv_total", "dbg_ax_total",
               *[f"reader{i}_idle" for i in range(n_channels)],
               *[f"pipe{i}_{f}" for i in range(n_channels) for f in ("valid", "row", "wide", "head")])
        # A fetch commands ONE channel and leaves the rest idle, so it is the most likely
        # thing to leave a reader busy while the next op issues - and firmware fetches
        # constantly (rope rows, norm weights, embeddings).
        for tag, cmd, dst in (("fetch",   fetch_cmd, 0x7000),
                              ("scores",  sc_cmd,    0x4000),
                              ("fetch2",  fetch_cmd, 0x7000),
                              ("axpy",    ax_cmd,    0x5000),
                              ("scores2", sc_cmd,    0x6000)):
            submit(s, dict(cmd, dst=dst))
            s.wait("done", cap=30000, tag=f"T={T}_rows={rows}_rem={rem}:_{tag}_never_completed")
            s.step(2)
            s.peek("error", tag)
        s.read("sp", 0x4000, 4*T, "scores")
        s.read("sp", 0x5000, 4*n_channels*cols, "axpy")
        s.read("sp", 0x6000, 4*T, "scores2")
        r = EngineBench(n_channels).run(backend, s, sp=x_init, payloads=payloads,
                                        latency=lat, stall_rate=stl)
        for tag in ("fetch", "scores", "fetch2", "axpy", "scores2"):
            assert r.peeks[tag] == 0, tag
        assert np.array_equal(_i32(r.reads["scores"]),  exp_scores), f"T={T} scores"
        assert np.array_equal(_i32(r.reads["axpy"]),    exp_axpy),   f"T={T} axpy"
        assert np.array_equal(_i32(r.reads["scores2"]), exp_scores), f"T={T} scores again"

@COMPILED
def test_matvec_q8_ilv_multirow_gaps(backend):
    """Q8 interleaved scores with more than one row per channel, and gaps in the read data.

    On the card the scores op stalls the moment a pipe has to process two rows: rows=1 works
    at every rem, rows=2 wedges even with rem=0 and every channel delivering exactly the
    beats it was commanded (measured ch0..7 = 10/10). The uniform-latency, gapless model
    passes rows=2, so the difference to look for is timing: real HBM returns a burst with
    gaps between beats, which is what stall_rate models.
    """
    cols, seed = 128, 151
    n_channels = 8
    for T, stall in [(8, 0), (16, 0), (16, 4), (24, 4), (17, 4)]:
        rows, rem = T//n_channels, T % n_channels
        rng    = np.random.default_rng(seed + T + stall)
        scales = rng.uniform(0.005, 0.02, (T, cols//layout.GROUP_Q8)).astype(np.float16)
        q8w    = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        x8     = rng.integers(-127, 128, cols, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
        expected = bx.matvec_q8(scales, q8w, x8, B, R)

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

        # xbuf_groups=96 is what the SoC builds (the default). The tests had been running
        # 32, and the Verilator sim - which uses 96 - is the only place the wedge reproduced.
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
        got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, T, 0x4000, 8, stall,
                          xbuf_groups=96)
        assert np.array_equal(got, expected), f"T={T} stall={stall}: {got} != {expected}"

@COMPILED
def test_matvec_ternary_under_id_reorder(backend):
    """Full matvec path against the same-ID-serializing memory model: the multi-ID readers see
    cross-ID response shuffling at high latency and the results must stay bit-exact. This is
    the engine-level guard for the multi-ARID reorder drain."""
    n_channels, rows_per_chan, cols, seed = 8, 12, 2048, 41
    rng = np.random.default_rng(seed)
    M = n_channels*rows_per_chan
    G = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    for latency in (8, 48):
        rb  = layout.row_beats(cols, layout.KIND_TERNARY)
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=0x4000)
        got = _run_engine(backend, n_channels, payloads, x8.tobytes(), cmd, M, 0x4000, latency, 0,
                          reorder_ids=4)
        # Position order per channel-stripe layout (kind=0 plain: rows land per channel).
        assert np.array_equal(got, expected), f"latency {latency}"

@COMPILED
@pytest.mark.parametrize("chan_stride", [0, 0x1000])
def test_engine_multi_store(chan_stride, backend):
    # F1-1: one STORE op scatters `rep` packets - packet r from scratchpad
    # xsrc + r*sstride beats to HBM wbase + r*dstride beats - the per-layer KV append as
    # a single dispatch. Fetch every region back and compare; a wrong stride or a packet
    # count off by one shows as a mismatch in exactly one region.
    n_channels = 2
    rng   = np.random.default_rng(29)
    rep   = 5
    beats = 3                      # kv_rb-sized packets.
    sstride = 16                   # 0x200 B slots, as kvp_slot().
    dstride = 64                   # 2 KB regions.
    sp = bytearray(64*1024)
    pk = []
    for r in range(rep):
        p = rng.integers(0, 256, 32*beats, dtype=np.uint8).tobytes()
        pk.append(p)
        sp[r*32*sstride:r*32*sstride + len(p)] = p

    fields = CMD_FIELDS + ("rep", "dstride", "sstride")
    s = Script()
    submit(s, dict(kind=3, chan=1, wbase=0x2000, wlen=beats, xsrc=0,
                   rep=rep, dstride=dstride, sstride=sstride), fields=fields)
    _store_fetch_wait(s, "store")
    for r in range(rep):
        submit(s, dict(kind=2, chan=1, wbase=0x2000 + r*32*dstride, wlen=beats,
                       dst=0x8000 + r*0x100), fields=fields)
        _store_fetch_wait(s, f"fetch{r}")
        s.read("sp", 0x8000 + r*0x100, 32*beats, f"packet{r}")
    bench = EngineBench(n_channels, chan_stride=chan_stride, loader=True)
    res = bench.run(backend, s, sp=bytes(sp))
    for r in range(rep):
        assert res.reads[f"packet{r}"] == pk[r], f"multi-store packet {r} mismatch"

@COMPILED
def test_engine_n16_ternary_and_store_window(backend):
    # N=16 enablement (2026-08-29): the shipping sharding at sixteen channels - a ternary
    # matvec across 16 readers with the 16-input xred-free gather - and the 33-bit STORE
    # address: channel 15's port-matched window is 0x1_0400_0000 (PC16, stack 1), which
    # a 32-bit write path silently wraps into PC0. The AW address on channel 15's AXI is
    # captured and checked against the full 33-bit value.
    n_channels, rows_per_chan, cols, seed = 16, 4, 256, 173
    rng    = np.random.default_rng(seed)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols).astype(np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    maxlen   = max(len(p) for p in payloads)
    payloads = [p.ljust(maxlen, b"\0") for p in payloads]

    # The models are 64 KB windows; reads/writes are addressed modulo the window, so the
    # 33-bit AW address is checked on the bus, not the model.
    rb = layout.row_beats(cols, layout.KIND_TERNARY)
    s  = Script()
    submit(s, dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=0x4000), fields=CMD_FIELDS)
    s.wait("done", cap=60000, tag="matvec")
    s.step()
    s.read("sp", 0x4000, 4*M, "res")
    # STORE 2 beats to channel 15 at the port-matched weight base: the bus must see the
    # 33-bit address.
    submit(s, dict(kind=3, chan=15, wbase=HBM_WEIGHT_BASE, wlen=2, xsrc=0), fields=CMD_FIELDS)
    s.wait("done", cap=60000, tag="store")
    s.step()
    bench = EngineBench(n_channels, chan_stride=HBM_CHAN_STRIDE, loader=True)
    r = bench.run(backend, s, sp=x8.tobytes(), payloads=payloads, latency=4, axi_log=[15])
    got = np.frombuffer(r.reads["res"], dtype="<i4")
    assert np.array_equal(got, expected), (got[:8], expected[:8])
    want = HBM_WEIGHT_BASE + 15*HBM_CHAN_STRIDE
    assert want >= (1 << 32), "test premise: channel 15 is above 4 GB"
    aw = r.axi_addresses("axi15", "aw")
    assert aw and aw[0] == want, ([hex(a) for a in aw[:2]], hex(want))

@COMPILED
def test_matvec_packed_writes(backend):
    # The result packer (2026-09-07): a plain matvec must reach the W port as full 8-lane
    # beats, not one lane per result. Counts the cycles the engine drives a write during
    # the op and checks the results are still byte-exact. Two shapes: rows a multiple of 8
    # (every beat full: writes == results/8) and rows = 12 (each channel's last beat is
    # shared with the next channel's first: two partial flushes per boundary).
    for n_channels, rows_per_chan, max_writes in ((2, 16, 4), (4, 12, 6 + 8)):
        rng    = np.random.default_rng(300 + rows_per_chan)
        cols   = 128
        M      = n_channels*rows_per_chan
        G      = cols//layout.GROUP_TERNARY
        scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
        codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
        x8     = rng.integers(-127, 128, cols, dtype=np.int8)
        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16); R = bx.choose_result_shift(scales, B)
        expected = bx.matvec_ternary(scales, codes, x8, B, R)
        spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                          scales=scales, values=codes)
        payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                    for c in range(n_channels)]
        dst = 0x4000
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                   wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=dst)
        s = Script()
        submit(s, cmd)
        s.count("dbg_gw_we", "writes")
        s.wait("done", cap=50000, tag="engine_timeout")
        s.step()                       # The done cycle's write counts too.
        s.count_end("writes")
        s.step()
        s.peek("error")
        s.read("sp", dst, 4*M, "res")
        r = EngineBench(n_channels).run(backend, s, sp=x8.tobytes(), payloads=payloads)
        assert r.peeks["error"] == 0
        assert np.array_equal(_i32(r.reads["res"]), expected), rows_per_chan
        writes = r.counts["writes"]
        assert writes <= max_writes, (n_channels, rows_per_chan, writes, max_writes)
        assert writes >= M//8, (n_channels, rows_per_chan, writes)


# Fetch --------------------------------------------------------------------------------------------
# HBM -> scratchpad.

@COMPILED
def test_engine_fetch(backend):
    rng  = np.random.default_rng(21)
    data = rng.integers(0, 256, 8*1024, dtype=np.uint8).tobytes()
    src  = 32*7   # Beat-aligned source in channel 1.
    dst  = 0x1000
    nbeats = 10

    s = Script()
    submit(s, dict(kind=2, chan=1, wbase=src, wlen=nbeats, dst=dst))
    s.wait("done", cap=2000, tag="fetch_timeout")
    s.step()
    s.read("sp", dst, 32*nbeats, "got")
    bench = EngineBench(2, sp_size=16*1024, xbuf_groups=8)
    r = bench.run(backend, s, payloads=[b"", data], latency=[4, 12], model_size=8*1024)
    for i in range(nbeats):
        assert r.reads["got"][32*i:32*(i + 1)] == data[src + 32*i:src + 32*(i + 1)], f"beat {i}"


# Channel Offsets (chan_mod) -----------------------------------------------------------------------
# chan_mod: the per-controller channel offset (the XCU1525's four DDR4 controllers).
#
# With chan_mod = 4 engine channel c addresses controller c // 4 at the FK33's constants for
# channel c % 4. The three address paths - the readers (matvec/fetch), the engine STORE and
# the host upload - are probed on the bus, since each channel's model is its own memory and
# a data round trip would pass with any offset at all. chan_mod = 0 must leave the FK33's
# addresses untouched.

@COMPILED
@pytest.mark.parametrize("chan_mod, chan_skew, chan_stride", [(0, 0, HBM_CHAN_STRIDE),
    (4, 0, HBM_CHAN_STRIDE), (4, 0x2000, HBM_CHAN_STRIDE), (4, 0x4000, 432 << 20)])
def test_chan_mod_addresses_every_path(chan_mod, chan_skew, chan_stride, backend):
    """Read (FETCH), STORE and host upload each add (c % chan_mod)*stride + (c % chan_mod)*skew
    in hardware - the bank skew (XCU1525: 8 KB) on top of the window stride. With chan_mod the
    stride need not be a power of two (the YPCB's even split of a controller: 432 MB)."""
    n_channels = 8
    wbase = HBM_WEIGHT_BASE + 0x8000

    def local(c):
        return (c % chan_mod) if chan_mod else c

    def off(c):
        return local(c)*chan_stride + local(c)*chan_skew

    s = Script()
    # FETCH from channel 6: one reader's AR carries wbase + local(6)*stride.
    submit(s, dict(kind=2, chan=6, wbase=wbase, wlen=2, dst=0x4000), fields=CMD_FIELDS)
    s.wait("done", cap=8000, tag="fetch")
    s.step()
    # STORE to channel 5: the loader's writer on channel 5 sees the masked offset.
    submit(s, dict(kind=3, chan=5, wbase=wbase, wlen=2, xsrc=0), fields=CMD_FIELDS)
    s.wait("done", cap=8000, tag="store")
    s.step()
    # Host upload to channel 7 through the loader's CSR path (base + offset in hardware).
    s.set("loader_chan", 7)
    s.set("loader_base", wbase)
    s.set("loader_length", 2)
    csr_write(s, "loader_start", 1)
    for beat in range(2):
        s.set("loader_sink_data", beat + 1)
        s.handshake("loader_sink_valid", "loader_sink_ready", tag="upload")
    s.wait("loader_done", cap=2000, tag="upload_done", fatal=False)
    s.step(20)

    bench = EngineBench(n_channels, chan_stride=chan_stride, chan_mod=chan_mod,
                        chan_skew=chan_skew, loader=True, loader_width=256)
    r = bench.run(backend, s, sp=np.arange(1024, dtype=np.uint8).tobytes(), latency=4,
                  axi_log=[5, 6, 7])
    ar6, aw5, aw7 = r.axi_addresses("axi6", "ar"), r.axi_addresses("axi5", "aw"), r.axi_addresses("axi7", "aw")
    assert ar6 and ar6[0] == wbase + off(6), [hex(a) for a in ar6[:2]]
    assert aw5 and aw5[0] == wbase + off(5), [hex(a) for a in aw5[:2]]
    assert aw7 and aw7[0] == wbase + off(7), [hex(a) for a in aw7[:2]]
    # Nothing else was addressed on the probed channels' write side.
    assert not r.axi_addresses("axi6", "aw")


@COMPILED
def test_engine_weight_prefetch(backend):
    """Op 7 issues a matvec's reads ahead: during an idle gap the readers fill their FIFOs, the
    matvec that follows skips ISSUE and runs faster, bit-exact; a prefetch is consumed once,
    and the next matvec issues its own reads again."""
    n_channels, rows, cols = 4, 8, 512
    rng    = np.random.default_rng(77)
    M      = n_channels*rows
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    cmd = dict(kind=0, rows=rows, cols=cols, wbase=0,
               wlen=rows*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)
    s = Script()
    for tag, pre, dst in (("plain", False, 0x4000), ("pref", True, 0x5000), ("again", False, 0x6000)):
        if pre:
            s.peek("absmax", "am_before")
            submit(s, dict(cmd, kind=7, dst=dst))
            s.wait("done", cap=5000, tag="prefetch_done")
            s.step(2)
            s.peek("absmax", "am_after") # The tracker survives a prefetch.
        s.step(400) # The serial gap the prefetch fills.
        s.peek("cyc_total", f"{tag}_t0")
        submit(s, dict(cmd, dst=dst))
        s.wait("done", cap=50000, tag=f"{tag}_done")
        s.peek("cyc_total", f"{tag}_t1")
        s.step(2)
        s.read("sp", dst, 4*M, tag)
    s.peek("pf_hits")
    s.peek("pf_mismatch")
    s.peek("error")
    r = EngineBench(n_channels, prefetch=True).run(backend, s, sp=x8.tobytes(),
                                                   payloads=payloads, latency=40)
    for tag in ("plain", "pref", "again"):
        assert np.array_equal(_i32(r.reads[tag]), expected), tag
    assert r.peeks["pf_hits"] == 1 and r.peeks["pf_mismatch"] == 0 and r.peeks["error"] == 0
    assert r.peeks["am_after"] == r.peeks["am_before"] != 0
    t = {tag: r.peeks[f"{tag}_t1"] - r.peeks[f"{tag}_t0"] for tag in ("plain", "pref", "again")}
    assert t["pref"] < t["plain"] - 30, t     # The reads' latency and first beats were hidden.
    assert abs(t["again"] - t["plain"]) <= 4, t

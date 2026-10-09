#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/engine.py: MatVecEngine end to end (matvec, fetch, store, AXPY, argmax, channel
offsets) against model.bitexact, on HBM channel models."""

import os

import numpy as np
import pytest

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from gateware.core.scratchpad import Scratchpad
from gateware.core.engine import MatVecEngine
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes
from gateware.memory.loader import Loader
from model import layout, bitexact as bx
from model.bons import TensorSpec, encode_rows
from model.layout import HBM_CHAN_STRIDE, HBM_WEIGHT_BASE

from common.engine import read_sp

# MatVec / Store / AXPY / Argmax -------------------------------------------------------------------
# MatVecEngine end-to-end unit test: BONS-encoded weight shards in HBM channel models,
# x8 in the scratchpad, engine command -> results in scratchpad, compared BIT-EXACTLY against
# model.bitexact.matvec_ternary()/matvec_q8().

class _EngineDUT(Module):
    def __init__(self, n_channels, chan_payloads, sp_init, latency=8, stall_rate=0,
                 xbuf_groups=32, reorder_ids=0):
        # latency/stall_rate may be a per-channel list: real HBM channels do not respond in
        # lockstep, and several engine bugs only appear once the readers desynchronize.
        lat = latency if isinstance(latency, (list, tuple)) else [latency]*n_channels
        stl = stall_rate if isinstance(stall_rate, (list, tuple)) else [stall_rate]*n_channels
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.submodules.scratchpad = Scratchpad(size=64*1024, init=sp_init)
        self.submodules.engine     = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=xbuf_groups)
        for i, axi in enumerate(axis):
            model = HBMChannelModel(axi, size=64*1024,
                init=words_from_bytes(chan_payloads[i]), latency=lat[i], stall_rate=stl[i],
                reorder_ids=reorder_ids)
            setattr(self.submodules, f"model{i}", model)

def _run_engine(dut, cmd, n_results, dst, absmax=False):
    """Results array, or (results, engine.absmax) when absmax=True."""
    results = []
    extra   = {}

    def driver():
        for k, v in cmd.items():
            yield getattr(dut.engine.cmd, k).eq(v)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(50000):
            if (yield dut.engine.done):
                break
            yield
        else:
            raise AssertionError("engine timeout")
        yield
        yield
        assert (yield dut.engine.error) == 0
        # Read back int32 results from the scratchpad.
        for i in range(n_results):
            word = (yield dut.scratchpad.mem[(dst + 4*i)//32])
            v    = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
            results.append(v - (1 << 32) if v & (1 << 31) else v)
        extra["absmax"] = (yield dut.engine.absmax)

    run_simulation(dut, [driver()])
    arr = np.array(results, dtype=np.int64)
    return (arr, extra["absmax"]) if absmax else arr

def _case_ternary(n_channels, rows_per_chan, cols, latency, stall_rate, seed):
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

    xsrc, dst = 0, 0x4000
    sp_init   = words_from_bytes(x8.tobytes())

    dut = _EngineDUT(n_channels, payloads, sp_init, latency, stall_rate)
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=xsrc, dst=dst)
    got = _run_engine(dut, cmd, M, dst)
    assert np.array_equal(got, expected), f"N={n_channels} rows={rows_per_chan} cols={cols}"

def test_matvec_ternary_n1():
    _case_ternary(n_channels=1, rows_per_chan=8, cols=256, latency=8, stall_rate=0, seed=1)

def test_matvec_ternary_n2():
    _case_ternary(n_channels=2, rows_per_chan=8, cols=256, latency=8, stall_rate=0, seed=2)

def test_matvec_ternary_n4():
    _case_ternary(n_channels=4, rows_per_chan=4, cols=256, latency=8, stall_rate=3, seed=7)

def test_matvec_ternary_n4_gather_pressure():
    # 4 channels x 2-beat rows demand 2 results/cycle against the 1/cycle gather drain:
    # pipes must throttle beat intake on result-FIFO headroom instead of dropping results
    # (without backpressure out_fifo overflows and the op never completes).
    _case_ternary(n_channels=4, rows_per_chan=48, cols=128, latency=8, stall_rate=0, seed=9)

def test_matvec_slim_no_dual():
    # Ternary + Q8 + striped/ilv scores + AXPY on one engine configuration.
    _case_ternary(n_channels=4, rows_per_chan=8, cols=256, latency=8, stall_rate=2,
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
    dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, 0)
    rb  = layout.row_beats(cols, layout.KIND_Q8)
    cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
    got = _run_engine(dut, cmd, T, 0x4000)
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
    dut = _EngineDUT(n_channels, payloads, words_from_bytes(p8.tobytes().ljust(32, b"\0")),
                     8, 0)
    rb  = layout.row_beats(cols, layout.KIND_Q8)
    cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem)
    got = _run_engine(dut, cmd, n_channels*cols, 0x4000)
    assert np.array_equal(got, expected), "slim axpy"

def test_matvec_ternary_multi_scaleword():
    # cols=4096: 32 groups -> 2 scale beats/row, exercises scale-word switching + row parity.
    _case_ternary(n_channels=2, rows_per_chan=2, cols=4096, latency=16, stall_rate=5, seed=3)

def test_matvec_q8():
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
    dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, 0)
    cmd = dict(kind=1, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_Q8),
               sbeats=layout.scale_beats(cols, layout.KIND_Q8),
               cbeats=layout.code_beats(cols, layout.KIND_Q8),
               bias=B, rshift=R, xsrc=xsrc, dst=dst)
    got = _run_engine(dut, cmd, M, dst)
    assert np.array_equal(got, expected)

@pytest.mark.slow
def test_matvec_real_1p7b_slice():
    # Real Ternary-Bonsai-1.7B layer-0 attn_q data (real fp16 scale distribution, real code
    # statistics) through the Migen engine, bit-exact vs the golden model. Uses a 64-rows-per-
    # channel slice of the N=2 layer-0 image so the HBM model init stays small.
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
    dut = _EngineDUT(2, payloads, words_from_bytes(x8.tobytes()), latency=16, stall_rate=4)
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*rb,
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=e["bias"], rshift=e["rshift"], xsrc=0, dst=dst)
    got = _run_engine(dut, cmd, 2*rows_per_chan, dst)
    assert np.array_equal(got, expected)

def test_engine_absmax_tracker():
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
    dut = _EngineDUT(2, payloads, words_from_bytes(x8.tobytes()))
    cmd = dict(kind=0, rows=M//2, cols=K, wbase=0,
               wlen=(M//2)*layout.row_beats(K, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(K, layout.KIND_TERNARY),
               cbeats=layout.code_beats(K, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)
    absmax = {}

    def check():
        yield
        for _ in range(20000):
            if (yield dut.engine.done):
                break
            yield
        yield
        absmax["v"] = (yield dut.engine.absmax)

    def driver():
        for k, v in cmd.items():
            yield getattr(dut.engine.cmd, k).eq(v)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)

    run_simulation(dut, [driver(), check()])
    assert absmax["v"] == int(np.abs(expected).max())

def test_engine_csr_single_run_per_submit():
    # Regression: submitting while the engine is idle used to latch `pending` even though
    # the command was accepted on the submit cycle itself, replaying every op once the
    # engine returned to idle (identical results, but the replay's command accept reset
    # the absmax tracker mid-read and doubled the stream time).
    from gateware.core.csr import EngineCSR

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

    dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()))
    dut.submodules.front = EngineCSR(dut.engine)

    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)

    def driver():
        for k, v in cmd.items():
            yield getattr(dut.front, k).storage.eq(v)
        yield
        yield dut.front.submit.storage.eq(1)
        yield dut.front.submit.re.eq(1)
        yield
        yield dut.front.submit.re.eq(0)
        yield
        dones = 0
        absmax_at_done = None
        for cyc in range(20000):
            if (yield dut.engine.done):
                dones += 1
                if absmax_at_done is None:
                    absmax_at_done = (yield dut.engine.absmax)
            yield
        assert dones == 1, f"engine ran {dones} times for one submit"
        assert absmax_at_done == int(np.abs(expected).max())
        assert (yield dut.front.status.fields.pending) == 0

    run_simulation(dut, [driver()])

@pytest.mark.parametrize("chan_stride", [0, 0x1000])
def test_engine_store_roundtrip(chan_stride):
    # kind-3 STORE (scratchpad -> HBM via the loader's writers), then FETCH back: the KV
    # append datapath. Also checks the done handshake orders after the B drain.
    #
    # chan_stride is parametrised because the built SoC uses a non-zero one (each channel owns
    # its own HBM window - the AXI switch makes the address, not the port, select the
    # pseudo-channel) while this test only ever ran at 0. On hardware the KV STORE wedges the
    # engine mid-token, and an untested address path is the first suspect.
    from gateware.memory.loader import Loader

    n_channels = 2
    rng   = np.random.default_rng(17)
    src   = rng.integers(0, 256, 512, dtype=np.uint8)  # 16 beats.
    beats = len(src)//32

    class _StoreDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(
                size=64*1024, init=words_from_bytes(src.tobytes()))
            self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                        chan_stride=chan_stride)
            self.submodules.loader = Loader(axis)
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
            for i, axi in enumerate(axis):
                model = HBMChannelModel(axi, size=64*1024, latency=8)
                setattr(self.submodules, f"model{i}", model)

    dut = _StoreDUT()

    def submit(c):
        for k in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                  "bias", "rshift", "xsrc", "dst", "chan"):
            yield getattr(dut.engine.cmd, k).eq(c.get(k, 0))
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)

    def wait_done():
        for _ in range(50000):
            if (yield dut.engine.done):
                return
            yield
        raise AssertionError("timeout")

    def driver():
        # Store 16 beats from SP 0x0 to channel 1 @ 0x2000.
        yield from submit(dict(kind=3, chan=1, wbase=0x2000, wlen=beats, xsrc=0))
        yield from wait_done()
        yield
        # Fetch them back to SP 0x4000.
        yield from submit(dict(kind=2, chan=1, wbase=0x2000, wlen=beats, dst=0x4000))
        yield from wait_done()
        yield
        got = bytearray()
        for w in range(len(src)//32):
            v = (yield dut.scratchpad.mem[(0x4000 + 32*w)//32])
            got += int(v).to_bytes(32, "little")
        assert bytes(got) == src.tobytes(), "store/fetch roundtrip mismatch"

    run_simulation(dut, [driver()])

def test_matvec_q8_striped_interleave():
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

        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, 0)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
        got = _run_engine(dut, cmd, T, 0x4000)
        assert np.array_equal(got, expected), f"T={T}: {got} != {expected}"

def test_engine_axpy():
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
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x_init), 8, 0)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem)
        got = _run_engine(dut, cmd, n_channels*cols, 0x4000)
        assert np.array_equal(got, expected), f"T={T}"

def test_engine_axpy_xred():
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
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x_init), 8, 0)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=Rpre, xsrc=0, dst=0x4000, rem=rem, xred=1)
        got = _run_engine(dut, cmd, cols, 0x4000, absmax=True)
        assert np.array_equal(got[0].astype(np.int32), expected), \
            f"xred N={n_channels} T={T}"
        # The engine absmax tracker must hold the absmax of the SUMMED vector, because that
        # is what norm_i32's k comes from once the reduce_acc chain is gone (SEQ_SRC_KABS).
        # This is only true with xred: without it the tracker sees the per-channel partials,
        # whose absmax is not the absmax of their sum.
        assert got[1] == int(np.abs(expected.astype(np.int64)).max()), \
            f"xred absmax N={n_channels} T={T}"

def test_engine_argmax_tap_and_suppress():
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

    dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()))
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000, supw=1)
    out = {}

    def driver():
        for k, v in cmd.items():
            yield getattr(dut.engine.cmd, k).eq(v)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(50000):
            if (yield dut.engine.done):
                break
            yield
        else:
            raise AssertionError("timeout")
        yield
        yield
        out["val"] = (yield dut.engine.argmax_val)
        out["idx"] = (yield dut.engine.argmax_idx)
        # Result region untouched (suppressed): scratchpad still zero there.
        untouched = 0
        for w in range(8):
            untouched |= (yield dut.scratchpad.mem[(0x4000 + 32*w)//32])
        out["untouched"] = untouched

    run_simulation(dut, [driver()])
    val = out["val"] - (1 << 32) if out["val"] & (1 << 31) else out["val"]
    assert val == exp_val, (val, exp_val)
    assert out["idx"] == exp_idx, (out["idx"], exp_idx)
    assert out["untouched"] == 0

def test_engine_back_to_back_scores_axpy_ragged():
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
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x_init), lat, stl)

        sc_cmd = dict(kind=1, rows=rows, rem=rem, ilv=1, cols=cols, wbase=0, wlen=rows*rb,
                      sbeats=sb, cbeats=cb, bias=Bs, rshift=Rs, xsrc=0, dst=0x4000)
        ax_cmd = dict(kind=4, rows=rows, rem=rem, cols=cols, wbase=0, wlen=rows*rb,
                      sbeats=sb, cbeats=cb, bias=Ba, rshift=Rpre, xsrc=0x100, dst=0x5000)
        fetch_cmd = dict(kind=2, rows=0, rem=0, cols=cols, wbase=0, wlen=4,
                         sbeats=sb, cbeats=cb, bias=0, rshift=0, xsrc=0, dst=0x7000,
                         chan=n_channels - 1)  # last channel, so reader 0 sits out

        def read_i32(dst, n):
            out = []
            for i in range(n):
                word = (yield dut.scratchpad.mem[(dst + 4*i)//32])
                v    = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
                out.append(v - (1 << 32) if v & (1 << 31) else v)
            return np.array(out, dtype=np.int64)

        def driver():
            # A fetch commands ONE channel and leaves the rest idle, so it is the most
            # likely thing to leave a reader busy while the next op issues - and firmware
            # fetches constantly (rope rows, norm weights, embeddings).
            for tag, cmd, dst in (("fetch",   fetch_cmd, 0x7000),
                                  ("scores",  sc_cmd,    0x4000),
                                  ("fetch2",  fetch_cmd, 0x7000),
                                  ("axpy",    ax_cmd,    0x5000),
                                  ("scores2", sc_cmd,    0x6000)):
                for k, v in dict(cmd, dst=dst).items():
                    yield getattr(dut.engine.cmd, k).eq(v)
                yield dut.engine.cmd.valid.eq(1)
                yield
                while not (yield dut.engine.cmd.ready):
                    yield
                yield dut.engine.cmd.valid.eq(0)
                for _ in range(30000):
                    if (yield dut.engine.done):
                        break
                    yield
                else:
                    # Say which readers are still holding the op, and which ones never
                    # delivered: the card showed every channel delivering exactly what it
                    # was commanded, so a match here means the same failure, not a
                    # test-only artefact.
                    idle = []
                    for r in dut.engine.readers:
                        idle.append((yield r.idle))
                    rc = (yield dut.engine.dbg_result_cnt)
                    mv = (yield dut.engine.dbg_mv_total)
                    ax = (yield dut.engine.dbg_ax_total)
                    # Every reader idle means the data is all in and consumed, so the
                    # question is which pipe is still holding a result the gather will not
                    # take. `wide` is in the merge condition (merge_mask requires ~wide), so
                    # a stale wide flag from a previous AXPY would strand exactly this.
                    pipes = []
                    for pi, p in enumerate(dut.engine.pipes):
                        v = (yield p.source.valid)
                        if v:
                            pipes.append((pi, "row%d" % (yield p.source.row),
                                          "wide%d" % (yield p.source.wide),
                                          "head%d" % (yield p.source.head)))
                    raise AssertionError(
                        f"T={T} rows={rows} rem={rem}: '{tag}' never completed. "
                        f"result_cnt={rc} mv_total={mv} ax_total={ax} reader idle={idle} "
                        f"pipes waiting={pipes}")
                yield
                yield
                assert (yield dut.engine.error) == 0, tag
            got_s = yield from read_i32(0x4000, T)
            got_a = yield from read_i32(0x5000, n_channels*cols)
            got_2 = yield from read_i32(0x6000, T)
            assert np.array_equal(got_s, exp_scores), f"T={T} scores"
            assert np.array_equal(got_a, exp_axpy),   f"T={T} axpy"
            assert np.array_equal(got_2, exp_scores), f"T={T} scores again"

        run_simulation(dut, [driver()])

def test_matvec_q8_ilv_multirow_gaps():
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
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, stall,
                         xbuf_groups=96)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1)
        got = _run_engine(dut, cmd, T, 0x4000)
        assert np.array_equal(got, expected), f"T={T} stall={stall}: {got} != {expected}"

def test_matvec_ternary_under_id_reorder():
    """Full matvec path against the same-ID-serializing HBM model: the multi-ID readers see
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
    sp = bytearray(64*1024)
    sp[0:cols] = x8.tobytes()
    for latency in (8, 48):
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(bytes(sp)),
                         latency, 0, reorder_ids=4)
        rb  = layout.row_beats(cols, layout.KIND_TERNARY)
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=0x4000)
        got = _run_engine(dut, cmd, M, 0x4000)
        # Position order per channel-stripe layout (kind=0 plain: rows land per channel).
        assert np.array_equal(got, expected), f"latency {latency}"

@pytest.mark.parametrize("chan_stride", [0, 0x1000])
def test_engine_multi_store(chan_stride):
    # F1-1: one STORE op scatters `rep` packets - packet r from scratchpad
    # xsrc + r*sstride beats to HBM wbase + r*dstride beats - the per-layer KV append as
    # a single dispatch. Fetch every region back and compare; a wrong stride or a packet
    # count off by one shows as a mismatch in exactly one region.
    from gateware.memory.loader import Loader

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

    class _MStoreDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(
                size=64*1024, init=words_from_bytes(bytes(sp)))
            self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                        chan_stride=chan_stride)
            self.submodules.loader = Loader(axis)
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
            for i, axi in enumerate(axis):
                model = HBMChannelModel(axi, size=64*1024, latency=8)
                setattr(self.submodules, f"model{i}", model)

    dut = _MStoreDUT()

    def submit(c):
        for k in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                  "bias", "rshift", "xsrc", "dst", "chan", "rep", "dstride", "sstride"):
            yield getattr(dut.engine.cmd, k).eq(c.get(k, 0))
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)

    def wait_done():
        for _ in range(50000):
            if (yield dut.engine.done):
                return
            yield
        raise AssertionError("timeout")

    def driver():
        yield from submit(dict(kind=3, chan=1, wbase=0x2000, wlen=beats, xsrc=0,
                               rep=rep, dstride=dstride, sstride=sstride))
        yield from wait_done()
        yield
        for r in range(rep):
            yield from submit(dict(kind=2, chan=1, wbase=0x2000 + r*32*dstride, wlen=beats,
                                   dst=0x8000 + r*0x100))
            yield from wait_done()
            yield
            got = bytearray()
            for w in range(beats):
                v = (yield dut.scratchpad.mem[(0x8000 + r*0x100 + 32*w)//32])
                got += int(v).to_bytes(32, "little")
            assert bytes(got) == pk[r], f"multi-store packet {r} mismatch"

    run_simulation(dut, [driver()])

@pytest.mark.slow  # sixteen Migen channel models: hours; run explicitly (-m slow or -k n16)
def test_engine_n16_ternary_and_store_window():
    # N=16 enablement (2026-08-29): the shipping sharding at sixteen channels - a ternary
    # matvec across 16 readers with the 16-input xred-free gather - and the 33-bit STORE
    # address: channel 15's port-matched window is 0x1_0400_0000 (PC16, stack 1), which
    # a 32-bit write path silently wraps into PC0. The AW address on channel 15's AXI is
    # captured and checked against the full 33-bit value.
    from gateware.memory.loader import Loader
    from model.layout import HBM_CHAN_STRIDE, HBM_WEIGHT_BASE

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

    class _N16DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.axis = axis
            self.submodules.scratchpad = Scratchpad(
                size=64*1024, init=words_from_bytes(x8.tobytes()))
            self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                        chan_stride=HBM_CHAN_STRIDE)
            self.submodules.loader = Loader(axis)
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
            for i, axi in enumerate(axis):
                # The models are 64 KB windows; reads/writes are addressed modulo the
                # window, so the 33-bit AW address is checked on the bus, not the model.
                model = HBMChannelModel(axi, size=64*1024,
                                              init=words_from_bytes(payloads[i]), latency=4)
                setattr(self.submodules, f"model{i}", model)

    dut = _N16DUT()
    out = {"aw": []}

    def submit(c):
        for k in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                  "bias", "rshift", "xsrc", "dst", "chan"):
            yield getattr(dut.engine.cmd, k).eq(c.get(k, 0))
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)

    def wait_done():
        for _ in range(60000):
            if (yield dut.engine.done):
                return
            yield
        raise AssertionError("timeout")

    def aw_probe():
        # Record every AW handshake on channel 15.
        axi = dut.axis[15]
        for _ in range(60000):
            if (yield axi.aw.valid) and (yield axi.aw.ready):
                out["aw"].append((yield axi.aw.addr))
            yield

    def driver():
        rb = layout.row_beats(cols, layout.KIND_TERNARY)
        yield from submit(dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                               wlen=rows_per_chan*rb,
                               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                               bias=B, rshift=R, xsrc=0, dst=0x4000))
        yield from wait_done()
        yield
        got = np.frombuffer((yield from read_sp(dut, 0x4000, 4*M)), dtype="<i4")
        assert np.array_equal(got, expected), (got[:8], expected[:8])
        # STORE 2 beats to channel 15 at the port-matched weight base: the bus must see
        # the 33-bit address.
        yield from submit(dict(kind=3, chan=15, wbase=HBM_WEIGHT_BASE, wlen=2, xsrc=0))
        yield from wait_done()
        yield
        want = HBM_WEIGHT_BASE + 15*HBM_CHAN_STRIDE
        assert want >= (1 << 32), "test premise: channel 15 is above 4 GB"
        assert out["aw"] and out["aw"][0] == want, (
            [hex(a) for a in out["aw"][:2]], hex(want))

    run_simulation(dut, [driver(), aw_probe()])

def test_matvec_packed_writes():
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
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, 0)
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                   wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=dst)
        writes = []
        results = []

        def driver():
            for k, v in cmd.items():
                yield getattr(dut.engine.cmd, k).eq(v)
            yield dut.engine.cmd.valid.eq(1)
            yield
            while not (yield dut.engine.cmd.ready):
                yield
            yield dut.engine.cmd.valid.eq(0)
            w = 0
            for _ in range(50000):
                if (yield dut.engine.dbg_gw_we):
                    w += 1
                if (yield dut.engine.done):
                    break
                yield
            else:
                raise AssertionError("engine timeout")
            writes.append(w)
            yield; yield
            assert (yield dut.engine.error) == 0
            for i in range(M):
                word = (yield dut.scratchpad.mem[(dst + 4*i)//32])
                v = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
                results.append(v - (1 << 32) if v & (1 << 31) else v)

        run_simulation(dut, [driver()])
        assert np.array_equal(np.array(results, dtype=np.int64), expected), rows_per_chan
        assert writes[0] <= max_writes, (n_channels, rows_per_chan, writes[0], max_writes)
        assert writes[0] >= M//8, (n_channels, rows_per_chan, writes[0])


# Fetch --------------------------------------------------------------------------------------------
# HBM -> scratchpad.

def test_engine_fetch():
    rng  = np.random.default_rng(21)
    data = rng.integers(0, 256, 8*1024, dtype=np.uint8).tobytes()

    class DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(2)]
            self.submodules.scratchpad = Scratchpad(size=16*1024)
            self.submodules.engine     = MatVecEngine(self.scratchpad, axis, xbuf_groups=8)
            self.submodules.model0 = HBMChannelModel(axis[0], size=8*1024, latency=4)
            self.submodules.model1 = HBMChannelModel(axis[1], size=8*1024,
                init=words_from_bytes(data), latency=12)

    dut  = DUT()
    src  = 32*7   # Beat-aligned source in channel 1.
    dst  = 0x1000
    nbeats = 10

    def gen():
        yield dut.engine.cmd.kind.eq(2)
        yield dut.engine.cmd.chan.eq(1)
        yield dut.engine.cmd.wbase.eq(src)
        yield dut.engine.cmd.wlen.eq(nbeats)
        yield dut.engine.cmd.dst.eq(dst)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(2000):
            if (yield dut.engine.done):
                break
            yield
        else:
            raise AssertionError("fetch timeout")
        yield
        for i in range(nbeats):
            got    = (yield dut.scratchpad.mem[dst//32 + i])
            expect = int.from_bytes(data[src + 32*i:src + 32*(i+1)], "little")
            assert got == expect, f"beat {i}"

    run_simulation(dut, [gen()])


# Channel Offsets (chan_mod) -----------------------------------------------------------------------
# chan_mod: the per-controller channel offset (the XCU1525's four DDR4 controllers).
#
# With chan_mod = 4 engine channel c addresses controller c // 4 at the FK33's constants for
# channel c % 4. The three address paths - the readers (matvec/fetch), the engine STORE and
# the host upload - are probed on the bus, since each channel's model is its own memory and
# a data round trip would pass with any offset at all. chan_mod = 0 must leave the FK33's
# addresses untouched.

class _DUT(Module):
    def __init__(self, n_channels, chan_mod, chan_skew=0):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.axis = axis
        src = np.arange(1024, dtype=np.uint8)
        self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                      init=words_from_bytes(src.tobytes()))
        self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                    chan_stride=HBM_CHAN_STRIDE,
                                                    chan_mod=chan_mod, chan_skew=chan_skew)
        self.submodules.loader = Loader(axis, data_width=256,
                                                 chan_stride=HBM_CHAN_STRIDE,
                                                 chan_mod=chan_mod, chan_skew=chan_skew)
        self.comb += [
            self.engine.store_cmd.connect(self.loader.store_cmd),
            self.engine.store_data.connect(self.loader.store_data),
            self.engine.store_busy.eq(self.loader.store_busy),
        ]
        for i, axi in enumerate(axis):
            # 64 KB windows addressed modulo their size: the address is checked on the bus.
            model = HBMChannelModel(axi, size=64*1024, latency=4)
            setattr(self.submodules, f"model{i}", model)


def _submit(dut, c):
    for k in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
              "bias", "rshift", "xsrc", "dst", "chan"):
        yield getattr(dut.engine.cmd, k).eq(c.get(k, 0))
    yield dut.engine.cmd.valid.eq(1)
    yield
    while not (yield dut.engine.cmd.ready):
        yield
    yield dut.engine.cmd.valid.eq(0)


def _wait_done(dut):
    for _ in range(8000):
        if (yield dut.engine.done):
            return
        yield
    raise AssertionError("timeout")


def _probe(axi, chan, out, key):
    # Runs until the driver is done (out["done"]): an 8-channel engine is slow to simulate,
    # so the probes must not pin the run to a fixed cycle count.
    while not out.get("done"):
        if (yield axi.aw.valid) and (yield axi.aw.ready):
            out.setdefault(("aw", key), []).append((yield axi.aw.addr))
        if (yield axi.ar.valid) and (yield axi.ar.ready):
            out.setdefault(("ar", key), []).append((yield axi.ar.addr))
        yield


@pytest.mark.parametrize("chan_mod, chan_skew", [(0, 0), (4, 0), (4, 0x2000)])
def test_chan_mod_addresses_every_path(chan_mod, chan_skew):
    """Read (FETCH), STORE and host upload each add (c % chan_mod)*stride + (c % chan_mod)*skew
    in hardware - the bank skew (XCU1525: 8 KB) on top of the window stride."""
    n_channels = 8
    dut = _DUT(n_channels, chan_mod, chan_skew)
    out = {}
    wbase = HBM_WEIGHT_BASE + 0x8000

    def local(c):
        return (c % chan_mod) if chan_mod else c

    def off(c):
        return local(c)*HBM_CHAN_STRIDE + local(c)*chan_skew

    def driver():
        # FETCH from channel 6: one reader's AR carries wbase + local(6)*stride.
        yield from _submit(dut, dict(kind=2, chan=6, wbase=wbase, wlen=2, dst=0x4000))
        yield from _wait_done(dut)
        yield
        # STORE to channel 5: the loader's writer on channel 5 sees the masked offset.
        yield from _submit(dut, dict(kind=3, chan=5, wbase=wbase, wlen=2, xsrc=0))
        yield from _wait_done(dut)
        yield
        # Host upload to channel 7 through the loader's CSR path (base + offset in hardware).
        yield dut.loader.chan.storage.eq(7)
        yield dut.loader.base.storage.eq(wbase)
        yield dut.loader.length.storage.eq(2)
        yield dut.loader.start.storage.eq(1)
        yield dut.loader.start.re.eq(1)
        yield
        yield dut.loader.start.re.eq(0)
        for beat in range(2):
            yield dut.loader.sink.data.eq(beat + 1)
            yield dut.loader.sink.valid.eq(1)
            yield
            while not (yield dut.loader.sink.ready):
                yield
        yield dut.loader.sink.valid.eq(0)
        for _ in range(2000):
            if (yield dut.loader.status.fields.done):
                break
            yield
        for _ in range(20):
            yield
        out["done"] = True

        assert out[("ar", 6)][0] == wbase + off(6), \
            [hex(a) for a in out[("ar", 6)][:2]]
        assert out[("aw", 5)][0] == wbase + off(5), \
            [hex(a) for a in out[("aw", 5)][:2]]
        assert out[("aw", 7)][0] == wbase + off(7), \
            [hex(a) for a in out[("aw", 7)][:2]]
        # Nothing else was addressed on the probed channels' write side.
        assert ("aw", 6) not in out

    run_simulation(dut, [driver(),
                         _probe(dut.axis[5], 5, out, 5),
                         _probe(dut.axis[6], 6, out, 6),
                         _probe(dut.axis[7], 7, out, 7)])

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# MatVecEngine end-to-end unit test: BONS-encoded weight shards in HBM channel models,
# x8 in the scratchpad, engine command -> results in scratchpad, compared BIT-EXACTLY against
# model.bitexact.matvec_ternary()/matvec_q8().

import numpy as np

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from model import layout
from model import bitexact as bx
from model.bons import TensorSpec, encode_rows
from gateware.core.scratchpad import Scratchpad
from gateware.core.engine import MatVecEngine
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes

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

import os
import pytest

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

def test_sequencer_v2_chaining():
    # Sequencer v2 primitives in one list, the way seq_build_attn will use them:
    #   reduce_acc (res6 = absmax) -> spill (S7 seeded with prev res6, persisted to SP,
    #   res6 preserved) -> norm_i32 with k from src=3 (k(res6)) -> reduce_acc over a
    #   crafted buffer (res6 = sum p8, standing in for softmax8's S6) -> engine AXPY with
    #   rshift_rpre=1 (rshift = bitlen(res6)+5 from the sequencer, w7 rshift left 0).
    from gateware.core.vector import VectorUnit
    from gateware.core.csr import EngineCSR
    from gateware.core.sequencer import Sequencer
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

    class _SeqDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7), # res6 chaining, or engine absmax (w0[4]).
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                model = HBMChannelModel(axi, size=64*1024,
                    init=words_from_bytes(payloads[i]), latency=8)
                setattr(self.submodules, f"model{i}", model)

    dut = _SeqDUT()

    def driver():
        yield dut.vu.prog_adr.storage.eq(0)
        yield dut.vu.prog_adr.re.eq(1)
        yield
        yield dut.vu.prog_adr.re.eq(0)
        for wd in words:
            yield dut.vu.prog.storage.eq(wd)
            yield dut.vu.prog.re.eq(1)
            yield
        yield dut.vu.prog.re.eq(0)
        yield
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(6)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(100000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        assert (yield dut.seq.status.fields.dcount) == 5
        data = bytearray()
        for wd in range(0x5000//32):
            v = (yield dut.scratchpad.mem[wd])
            data += int(v).to_bytes(32, "little")
        spill = np.frombuffer(bytes(data[0x3800:0x3800+32]), dtype="<u4")
        nrm   = np.frombuffer(bytes(data[0x3000:0x3000+32]), dtype="<i2")[:16]
        axp   = np.frombuffer(bytes(data[0x4000:0x4000+4*n_channels*cols]), dtype="<i4")
        assert np.all(spill == K1 & 0xFFFFFFFF), ("spill S7 seed", spill, K1)
        assert np.array_equal(nrm, nrm_exp), ("norm_i32 k via src=3", nrm[:8], nrm_exp[:8], k_exp)
        assert np.array_equal(axp, axpy_exp), ("axpy rshift_rpre", axp[:8], axpy_exp[:8], Rpre)

    run_simulation(dut, [driver()])

def test_sequencer_mixed_list():
    # Descriptor list: vec quant8 -> engine ternary matvec -> vec norm_i32 with the k
    # auto-arg from the engine absmax tracker -> stop. Results must match the individually
    # computed goldens; numerics are the same as CPU-dispatched ops by construction.
    from gateware.core.vector import VectorUnit
    from gateware.core.csr import EngineCSR
    from gateware.core.sequencer import Sequencer
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

    class _SeqDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                model = HBMChannelModel(axi, size=64*1024,
                    init=words_from_bytes(payloads[i]), latency=8)
                setattr(self.submodules, f"model{i}", model)

    dut = _SeqDUT()

    def driver():
        # Upload the vec programs.
        yield dut.vu.prog_adr.storage.eq(0)
        yield dut.vu.prog_adr.re.eq(1)
        yield
        yield dut.vu.prog_adr.re.eq(0)
        for wd in words:
            yield dut.vu.prog.storage.eq(wd)
            yield dut.vu.prog.re.eq(1)
            yield
        yield dut.vu.prog.re.eq(0)
        yield
        # Kick the sequencer.
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(7)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(100000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        assert (yield dut.seq.status.fields.dcount) == 3
        data = bytearray()
        for wd in range(0x4000//32):
            v = (yield dut.scratchpad.mem[wd])
            data += int(v).to_bytes(32, "little")
        out = {}
        out["x8"]  = np.frombuffer(bytes(data[0x1000:0x1000+cols]), dtype=np.int8)
        out["res"] = np.frombuffer(bytes(data[0x2000:0x2000+4*M]), dtype="<i4")
        out["nrm"] = np.frombuffer(bytes(data[0x3000:0x3000+2*16*max(1, M//16 if M >= 16 else 1)]),
                                   dtype="<i2")[:M]
        assert np.array_equal(out["x8"], x8_exp.astype(np.int8)), "quant8 via sequencer"
        assert np.array_equal(out["res"], res_exp), "matvec via sequencer"
        assert np.array_equal(out["nrm"], nrm_exp.astype(np.int16)), ("norm_i32 auto-k", out["nrm"][:8], nrm_exp[:8], k_exp, amax)

    run_simulation(dut, [driver()])

def test_sequencer_q8_scores_ragged():
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
    from gateware.core.vector import VectorUnit
    from gateware.core.csr import EngineCSR
    from gateware.core.sequencer import Sequencer
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

        class _SeqQ8DUT(Module):
            def __init__(self):
                axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                        for _ in range(n_channels)]
                self.submodules.scratchpad = Scratchpad(
                    size=64*1024, init=words_from_bytes(bytes(sp)))
                self.submodules.vu  = VectorUnit()
                self.submodules.seq = Sequencer()
                self.submodules.engine = MatVecEngine(
                    self.scratchpad, axis, xbuf_groups=32, vecunit=self.vu, seq=self.seq)
                self.submodules.front = EngineCSR(self.engine, seq=self.seq)
                self.comb += [
                    self.vu.seq_en.eq(self.seq.busy),
                    self.vu.seq_start.eq(self.seq.vec_start),
                    self.vu.seq_pc.eq(self.seq.vec_pc),
                    self.seq.vec_done.eq(self.vu.done),
                    self.seq.vec_res6.eq(self.vu.res6.status),
                ]
                self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
                for i, axi in enumerate(axis):
                    model = HBMChannelModel(axi, size=64*1024,
                        init=words_from_bytes(payloads[i]), latency=8)
                    setattr(self.submodules, f"model{i}", model)

        dut = _SeqQ8DUT()

        def driver():
            yield dut.vu.prog_adr.storage.eq(0)
            yield dut.vu.prog_adr.re.eq(1)
            yield
            yield dut.vu.prog_adr.re.eq(0)
            for wd in vec_words:
                yield dut.vu.prog.storage.eq(wd)
                yield dut.vu.prog.re.eq(1)
                yield
            yield dut.vu.prog.re.eq(0)
            yield
            yield dut.seq.base.storage.eq(0x6000)
            yield dut.seq.count.storage.eq(5)
            yield
            yield dut.seq.start.storage.eq(1)
            yield dut.seq.start.re.eq(1)
            yield
            yield dut.seq.start.re.eq(0)
            yield
            for _ in range(200000):
                if not (yield dut.seq.busy):
                    break
                yield
            else:
                st = yield dut.front.status.status
                raise AssertionError(
                    f"T={T} rows={rows} rem={rem}: sequencer never finished "
                    f"(engine status {st:03x}) - this is the SEQSTALL wedge")
            yield
            data = bytearray()
            for wd in range(0x5000//32):
                v = (yield dut.scratchpad.mem[wd])
                data += int(v).to_bytes(32, "little")
            got = np.frombuffer(bytes(data[0x4000:0x4000 + 4*T]), dtype="<i4")
            assert np.array_equal(got, expected), f"T={T}: {got} != {expected}"

        run_simulation(dut, [driver()])

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

def test_sequencer_xred_absmax_spill():
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
    from gateware.core.vector import VectorUnit
    from gateware.core.sequencer import Sequencer
    from gateware.core.csr import EngineCSR
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

    class _DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                setattr(self.submodules, f"model{i}", HBMChannelModel(
                    axi, size=64*1024, init=words_from_bytes(payloads[i]), latency=8))

    dut = _DUT()

    def driver():
        yield dut.vu.prog_adr.storage.eq(0)
        yield dut.vu.prog_adr.re.eq(1)
        yield
        yield dut.vu.prog_adr.re.eq(0)
        for wd in words:
            yield dut.vu.prog.storage.eq(wd)
            yield dut.vu.prog.re.eq(1)
            yield
        yield dut.vu.prog.re.eq(0)
        yield
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(7)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(200000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        data = bytearray()
        for wd in range(0x5000//32):
            v = (yield dut.scratchpad.mem[wd])
            data += int(v).to_bytes(32, "little")
        axp   = np.frombuffer(bytes(data[0x4000:0x4000+4*cols]), dtype="<i4")
        spill = np.frombuffer(bytes(data[0x3800:0x3800+32]), dtype="<u4")
        spill2 = np.frombuffer(bytes(data[0x3820:0x3820+32]), dtype="<u4")
        assert np.array_equal(axp, summed), ("xred sum", axp[:8], summed[:8])
        assert np.all(spill == am_exp),  ("head 0 spilled absmax", spill[0], am_exp)
        assert np.all(spill2 == am_exp), ("head 1 spilled absmax", spill2[0], am_exp)

    run_simulation(dut, [driver()])

def test_sequencer_kreg_capture_and_add():
    # The exponent register file: an engine op captures k(absmax) into kreg[i] (w0[5],[7:6])
    # and a LATER vec descriptor adds it to an arg (w1[16+2i]). This is what lets the FFN
    # chain's "per-layer constant + an EARLIER op's k" args live in descriptors instead of
    # forcing a CPU round-trip per op - the 43% mid-band in the gap histogram.
    #
    # Shape: axpy(xred, capture kreg0) -> norm_i32 with A4 = imm(0) + kreg0. The reference k
    # is bitlen(summed absmax) - 14, i.e. exactly what SEQ_SRC_KABS would give the very next
    # op - but here it survives an arbitrary distance and combines with an immediate.
    import model.asm as vasm
    from gateware.core.vector import VectorUnit
    from gateware.core.sequencer import Sequencer
    from gateware.core.csr import EngineCSR
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

    class _DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                setattr(self.submodules, f"model{i}", HBMChannelModel(
                    axi, size=64*1024, init=words_from_bytes(payloads[i]), latency=8))

    dut = _DUT()

    def driver():
        yield dut.vu.prog_adr.storage.eq(0)
        yield dut.vu.prog_adr.re.eq(1)
        yield
        yield dut.vu.prog_adr.re.eq(0)
        for wd in words:
            yield dut.vu.prog.storage.eq(wd)
            yield dut.vu.prog.re.eq(1)
            yield
        yield dut.vu.prog.re.eq(0)
        yield
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(4)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(200000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        data = bytearray()
        for wd in range(0x5000//32):
            v = (yield dut.scratchpad.mem[wd])
            data += int(v).to_bytes(32, "little")
        nrm  = np.frombuffer(bytes(data[0x3000:0x3000+2*cols]), dtype="<i2")
        nrmA = np.frombuffer(bytes(data[0x3400:0x3400+2*cols]), dtype="<i2")
        kr0 = (yield dut.seq.kregs[0])
        assert kr0 == k_exp, ("kreg0 capture", kr0, k_exp)
        assert np.array_equal(nrm, nrm_exp), ("norm via kreg", nrm[:8], nrm_exp[:8], k_exp)
        assert np.array_equal(nrmA, nrm_exp), ("norm via kreg ADD (w0[11])",
                                               nrmA[:8], nrm_exp[:8], k_exp)

    run_simulation(dut, [driver()])


def test_sequencer_lut_arg():
    # A5 from a scratchpad LUT indexed by imm5 + kreg (w0[12]): the one arg
    # the k/v chain could not derive in-sequencer was rmsnorm's eps, a per-layer table
    # indexed by the exponent shift the sequencer already holds in a kreg. Shape: the
    # kreg-capture test's AXPY (k -> kreg0), then rmsnorm with A5 = LUT[3 + kreg0]; the
    # same rmsnorm with the immediate the LUT holds must match bit for bit, and an index
    # past the table clamps to entry 31 rather than reading past it.
    import model.asm as vasm
    from gateware.core.vector import VectorUnit
    from gateware.core.sequencer import Sequencer
    from gateware.core.csr import EngineCSR
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

    class _DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                setattr(self.submodules, f"model{i}", HBMChannelModel(
                    axi, size=64*1024, init=words_from_bytes(payloads[i]), latency=8))

    dut = _DUT()

    def driver():
        yield dut.vu.prog_adr.storage.eq(0)
        yield dut.vu.prog_adr.re.eq(1)
        yield
        yield dut.vu.prog_adr.re.eq(0)
        for wd in words:
            yield dut.vu.prog.storage.eq(wd)
            yield dut.vu.prog.re.eq(1)
            yield
        yield dut.vu.prog.re.eq(0)
        yield
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(5)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(200000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        data = bytearray()
        for wd in range(0x3000//32):
            v = (yield dut.scratchpad.mem[wd])
            data += int(v).to_bytes(32, "little")
        kr0 = (yield dut.seq.kregs[0])
        assert kr0 == k_exp, ("kreg0 capture", kr0, k_exp)
        out_lut   = np.frombuffer(bytes(data[0x2000:0x2000+2*n]), dtype="<i2")
        out_imm   = np.frombuffer(bytes(data[0x2400:0x2400+2*n]), dtype="<i2")
        out_clamp = np.frombuffer(bytes(data[0x2800:0x2800+2*n]), dtype="<i2")
        assert np.array_equal(out_lut, exp_lut), ("A5 via LUT[3+kreg0]", out_lut[:6], exp_lut[:6])
        assert np.array_equal(out_imm, exp_lut), ("same eps as immediate", out_imm[:6])
        assert np.array_equal(out_clamp, exp_clamp), ("index 40+k clamps to 31",
                                                      out_clamp[:6], exp_clamp[:6])

    run_simulation(dut, [driver()])


# Per-cycle scratchpad arbitration (docs/units.md (Overlap and scratchpad arbitration)) ----------------------------------------

class _OverlapDUT(Module):
    # Engine + vector unit sharing the scratchpad WITHOUT a sequencer or CSR front:
    # engine.cmd is testbench-driven, the vector unit runs from its own CSRs, and the
    # two are launched concurrently - the per-cycle port arbitration is what keeps
    # both results bit-exact.
    def __init__(self, n_channels, chan_payloads, sp_init, latency=8):
        from gateware.core.vector import VectorUnit
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.submodules.scratchpad = Scratchpad(size=64*1024, init=sp_init)
        self.submodules.vu = VectorUnit()
        self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                    xbuf_groups=32, vecunit=self.vu)
        # Mirror the SoC's sideband idle wiring (no sequencer in this DUT): the posted-
        # write drain works without it, but read grants are units_idle-gated.
        self.comb += self.scratchpad.units_idle.eq(~self.engine.sp_busy & ~self.vu.busy)
        for i, axi in enumerate(axis):
            model = HBMChannelModel(axi, size=64*1024,
                init=words_from_bytes(chan_payloads[i]), latency=latency)
            setattr(self.submodules, f"model{i}", model)


def _vec_csr_upload(dut, words):
    yield dut.vu.prog_adr.storage.eq(0)
    yield dut.vu.prog_adr.re.eq(1)
    yield
    yield dut.vu.prog_adr.re.eq(0)
    for wd in words:
        yield dut.vu.prog.storage.eq(wd)
        yield dut.vu.prog.re.eq(1)
        yield
    yield dut.vu.prog.re.eq(0)
    yield


def _vec_csr_submit(dut, args):
    arg_csrs = [dut.vu.arg0, dut.vu.arg1, dut.vu.arg2, dut.vu.arg3,
                dut.vu.arg4, dut.vu.arg5, dut.vu.arg6, dut.vu.arg7]
    for i, a in enumerate(args):
        yield arg_csrs[i].storage.eq(a & 0xFFFFFFFF)
    yield dut.vu.pc.storage.eq(0)
    yield dut.vu.submit.storage.eq(1)
    yield dut.vu.submit.re.eq(1)
    yield
    yield dut.vu.submit.re.eq(0)
    yield


def _overlap_vec_golden(sp, vec_args):
    from model import asm as vasm
    mirror = bytearray(sp)
    vasm.VecInterp(mirror).run(vasm.assemble(vasm.PROGRAMS["resid_add"]()), list(vec_args))
    return mirror


def _read_sp(dut, start, length):
    data = bytearray()
    for wd in range(start//32, (start + length + 31)//32):
        v = (yield dut.scratchpad.mem[wd])
        data += int(v).to_bytes(32, "little")
    return bytes(data[:length])


def test_overlap_sideband_slot():
    # Posted sideband writes stream in DURING a concurrent engine matvec + vec resid_add
    # (2026-08-31 second form): each push must be accepted within a bounded wait, the
    # FIFO must drain (sbw_pending falls), reads AFTER the run (units_idle + FIFO-empty
    # gated) must see every posted write and the untouched known block, both unit
    # results must equal their serial goldens, and no gather write may be lost.
    from model import asm as vasm
    n_channels, rows_per_chan, cols, seed = 2, 32, 128, 95
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
    dut = _OverlapDUT(n_channels, payloads, sp_init=words_from_bytes(bytes(sp)))

    wrote = {}

    def sb_post(beat, sel, wdat):
        yield dut.scratchpad._sbw_req[1].eq(1)
        yield dut.scratchpad._sbw_adr[1].eq(beat)
        yield dut.scratchpad._sbw_sel[1].eq(sel)
        yield dut.scratchpad._sbw_bsel[1].eq(0xF)
        yield dut.scratchpad._sbw_wdat[1].eq(wdat)
        yield
        waited = 0
        while not (yield dut.scratchpad._sbw_ok[1]):
            waited += 1
            assert waited < 300, "posted write starved during concurrent units"
            yield
        yield dut.scratchpad._sbw_req[1].eq(0)
        yield

    def sb_read(beat, sel):
        yield dut.scratchpad._sb_req[1].eq(1)
        yield dut.scratchpad._sb_adr[1].eq(beat)
        yield dut.scratchpad._sb_sel[1].eq(sel)
        yield
        waited = 0
        while not (yield dut.scratchpad._sb_gnt[1]):
            waited += 1
            assert waited < 3000, "sideband read never granted after done"
            yield
        yield dut.scratchpad._sb_req[1].eq(0)
        yield
        rd = (yield dut.scratchpad._sb_rdat)
        return (rd >> (32*sel)) & 0xFFFFFFFF

    def driver():
        yield from _vec_csr_upload(dut, vasm.assemble(vasm.PROGRAMS["resid_add"]()))
        yield dut.engine.cmd.kind.eq(0)
        yield dut.engine.cmd.rows.eq(rows_per_chan)
        yield dut.engine.cmd.cols.eq(cols)
        yield dut.engine.cmd.rem.eq(0)
        yield dut.engine.cmd.sbeats.eq((G*2 + 31)//32)
        yield dut.engine.cmd.cbeats.eq((cols + 127)//128)
        yield dut.engine.cmd.wbase.eq(0)
        yield dut.engine.cmd.wlen.eq(len(payloads[0])//32)
        yield dut.engine.cmd.bias.eq(B)
        yield dut.engine.cmd.rshift.eq(R)
        yield dut.engine.cmd.xsrc.eq(0x1000)
        yield dut.engine.cmd.dst.eq(0x2000)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        yield from _vec_csr_submit(dut, vec_args)
        # Posted writes streamed while both units run.
        for i in range(48):
            beat, sel = 0x7800//32 + (i % 4), (i // 4) % 8
            w = (0xC0DE0000 + i) & 0xFFFFFFFF
            yield from sb_post(beat, sel, w)
            wrote[(beat, sel)] = w
            for _ in range(2):
                yield
        for _ in range(30000):
            if (yield dut.engine.done):
                break
            yield
        # FIFO must drain once granted cycles exist.
        waited = 0
        while (yield dut.scratchpad.sbw_pending):
            waited += 1
            assert waited < 1000, "posted-write FIFO never drained"
            yield
        for _ in range(200):
            yield
        assert (yield dut.engine.gw_ovf) == 0, "gather write lost under sideband traffic"
        got_res = yield from _read_sp(dut, 0x2000, 4*M)
        assert np.array_equal(np.frombuffer(got_res, dtype="<i4"), res_exp), \
            "matvec result changed under sideband traffic"
        got_vec = yield from _read_sp(dut, 0x5000, 2*nv)
        assert got_vec == bytes(mirror[0x5000:0x5000 + 2*nv]), \
            "vec result changed under sideband traffic"
        for (beat, sel), w in wrote.items():
            v = yield from sb_read(beat, sel)
            assert v == w, f"posted write at beat {beat} sel {sel}: {v:#x} != {w:#x}"
        for i in range(8):
            v = yield from sb_read(0x7000//32 + i, i)
            assert v == int(sb_src[i*8 + i]), f"known block read {i} returned {v:#x}"

    run_simulation(dut, [driver()])


def test_overlap_sideband_read_slot():
    # C2-2: sideband READS grant during a concurrent engine matvec + vec resid_add (the
    # R-port-only slot), return correct data, and never outrun the posted-write FIFO -
    # a read issued right after a posted write to the same word must return the NEW
    # value (the FIFO-empty gate is a tested contract). Unit results stay bit-exact and
    # no gather write is lost (reads must never disturb the W port).
    from model import asm as vasm
    n_channels, rows_per_chan, cols, seed = 2, 32, 128, 97
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
    dut = _OverlapDUT(n_channels, payloads, sp_init=words_from_bytes(bytes(sp)))

    def sb_post(beat, sel, wdat):
        yield dut.scratchpad._sbw_req[1].eq(1)
        yield dut.scratchpad._sbw_adr[1].eq(beat)
        yield dut.scratchpad._sbw_sel[1].eq(sel)
        yield dut.scratchpad._sbw_bsel[1].eq(0xF)
        yield dut.scratchpad._sbw_wdat[1].eq(wdat)
        yield
        waited = 0
        while not (yield dut.scratchpad._sbw_ok[1]):
            waited += 1
            assert waited < 300, "posted write starved"
            yield
        yield dut.scratchpad._sbw_req[1].eq(0)
        yield

    def sb_read(beat, sel, bound, tag):
        yield dut.scratchpad._sb_req[1].eq(1)
        yield dut.scratchpad._sb_adr[1].eq(beat)
        yield dut.scratchpad._sb_sel[1].eq(sel)
        yield
        waited = 0
        while not (yield dut.scratchpad._sb_gnt[1]):
            waited += 1
            assert waited < bound, f"read not granted ({tag})"
            yield
        yield dut.scratchpad._sb_req[1].eq(0)
        yield
        rd = (yield dut.scratchpad._sb_rdat)
        return (rd >> (32*sel)) & 0xFFFFFFFF

    def driver():
        yield from _vec_csr_upload(dut, vasm.assemble(vasm.PROGRAMS["resid_add"]()))
        yield dut.engine.cmd.kind.eq(0)
        yield dut.engine.cmd.rows.eq(rows_per_chan)
        yield dut.engine.cmd.cols.eq(cols)
        yield dut.engine.cmd.rem.eq(0)
        yield dut.engine.cmd.sbeats.eq((G*2 + 31)//32)
        yield dut.engine.cmd.cbeats.eq((cols + 127)//128)
        yield dut.engine.cmd.wbase.eq(0)
        yield dut.engine.cmd.wlen.eq(len(payloads[0])//32)
        yield dut.engine.cmd.bias.eq(B)
        yield dut.engine.cmd.rshift.eq(R)
        yield dut.engine.cmd.xsrc.eq(0x1000)
        yield dut.engine.cmd.dst.eq(0x2000)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        yield from _vec_csr_submit(dut, vec_args)
        # Mid-activity reads of the known block: must grant within a small bound.
        for i in range(24):
            v = yield from sb_read(0x7000//32 + (i % 8), i % 8, 300, "mid-activity")
            assert v == int(sb_src[(i % 8)*8 + (i % 8)]), f"read {i} returned {v:#x}"
            for _ in range(2):
                yield
        # Read-after-posted-write: the read must wait for the drain and see the NEW value.
        for i in range(8):
            w = (0xFEED0000 + i) & 0xFFFFFFFF
            yield from sb_post(0x7800//32, i, w)
            v = yield from sb_read(0x7800//32, i, 600, "after-post")
            assert v == w, f"read-after-post {i}: {v:#x} != {w:#x}"
        for _ in range(30000):
            if (yield dut.engine.done):
                break
            yield
        for _ in range(200):
            yield
        assert (yield dut.engine.gw_ovf) == 0, "gather write lost under read traffic"
        got_res = yield from _read_sp(dut, 0x2000, 4*M)
        assert np.array_equal(np.frombuffer(got_res, dtype="<i4"), res_exp), \
            "matvec result changed under sideband reads"
        got_vec = yield from _read_sp(dut, 0x5000, 2*nv)
        assert got_vec == bytes(mirror[0x5000:0x5000 + 2*nv]), \
            "vec result changed under sideband reads"

    run_simulation(dut, [driver()])


def test_overlap_engine_vec_concurrent():
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
    dut   = _OverlapDUT(n_channels, payloads, words_from_bytes(bytes(sp)))
    out   = {}

    def driver():
        yield from _vec_csr_upload(dut, words)
        # Vec first (it runs ~2k cycles), engine a few cycles later - contention spans
        # the engine's whole FILL/RUN window.
        yield from _vec_csr_submit(dut, vec_args)
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0x1000, dst=0x2000)
        for k, v in cmd.items():
            yield getattr(dut.engine.cmd, k).eq(v)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(50000):
            eb = (yield dut.engine.busy)
            vb = (yield dut.vu.busy)
            if not eb and not vb:
                break
            yield
        else:
            raise AssertionError("overlap run timeout")
        yield
        out["both"] = (yield dut.engine.cyc_both)
        out["ovf"]  = (yield dut.engine.gw_ovf)
        out["res"]  = np.frombuffer((yield from _read_sp(dut, 0x2000, 4*M)), dtype="<i4")
        out["vec"]  = (yield from _read_sp(dut, 0x5000, 2*nv))

    run_simulation(dut, [driver()])
    assert out["ovf"] == 0, "gather FIFO overflow"
    assert np.array_equal(out["res"], res_exp), (out["res"][:8], res_exp[:8])
    assert out["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert out["both"] > 50, f"no real overlap measured (cyc_both={out['both']})"


def test_overlap_fetch_under_vec():
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
    dut   = _OverlapDUT(n_channels, payloads, words_from_bytes(bytes(sp)))
    out   = {}

    def driver():
        yield from _vec_csr_upload(dut, words)
        yield from _vec_csr_submit(dut, vec_args)
        cmd = dict(kind=2, wbase=0, wlen=fetch_beats, dst=0x3000, chan=0)
        for k, v in cmd.items():
            yield getattr(dut.engine.cmd, k).eq(v)
        yield dut.engine.cmd.valid.eq(1)
        yield
        while not (yield dut.engine.cmd.ready):
            yield
        yield dut.engine.cmd.valid.eq(0)
        for _ in range(50000):
            if not (yield dut.engine.busy) and not (yield dut.vu.busy):
                break
            yield
        else:
            raise AssertionError("fetch-under-vec timeout")
        yield
        out["both"]  = (yield dut.engine.cyc_both)
        out["ovf"]   = (yield dut.engine.gw_ovf)
        out["fetch"] = (yield from _read_sp(dut, 0x3000, 32*fetch_beats))
        out["vec"]   = (yield from _read_sp(dut, 0x5000, 2*nv))

    run_simulation(dut, [driver()])
    assert out["ovf"] == 0, "gather FIFO overflow"
    assert out["fetch"] == payloads[0][:32*fetch_beats], "fetched data corrupted"
    assert out["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert out["both"] > 20, f"no real overlap measured (cyc_both={out['both']})"


def test_sequencer_nowait_overlap():
    # NOWAIT engine descriptor (w0[9]): the following independent vec descriptor runs
    # WHILE the matvec streams; the next descriptor (norm_i32 with A4 = KABS) trips the
    # hazard interlock and waits for the in-flight op. All three results bit-exact,
    # real overlap measured, and the interlock's k matches the serial golden.
    from gateware.core.vector import VectorUnit
    from gateware.core.csr import EngineCSR
    from gateware.core.sequencer import Sequencer
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

    class _NwDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                model = HBMChannelModel(axi, size=64*1024,
                    init=words_from_bytes(payloads[i]), latency=8)
                setattr(self.submodules, f"model{i}", model)

    dut = _NwDUT()
    out = {}

    def driver():
        yield from _vec_csr_upload(dut, words)
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(5)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(100000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        out["dcount"] = (yield dut.seq.status.fields.dcount)
        out["both"]   = (yield dut.engine.cyc_both)
        out["ovf"]    = (yield dut.engine.gw_ovf)
        out["res"]  = np.frombuffer((yield from _read_sp(dut, 0x2000, 4*M)), dtype="<i4")
        out["nrm"]  = np.frombuffer((yield from _read_sp(dut, 0x3000, 2*M)), dtype="<i2")
        out["nrmb"] = np.frombuffer((yield from _read_sp(dut, 0x3800, 2*M)), dtype="<i2")
        out["vec"]  = (yield from _read_sp(dut, 0x5000, 2*nv))

    run_simulation(dut, [driver()])
    assert out["ovf"] == 0, "gather FIFO overflow"
    assert out["dcount"] == 4, out["dcount"]
    assert np.array_equal(out["res"], res_exp), (out["res"][:8], res_exp[:8])
    assert np.array_equal(out["nrm"], nrm_exp), (out["nrm"][:8], nrm_exp[:8], k_exp)
    assert np.array_equal(out["nrmb"], nrm_exp), "w0[10] barrier saw partial results"
    assert out["vec"] == bytes(mirror[0x5000:0x5000 + 2*nv]), "vec result corrupted"
    assert out["both"] > 50, f"no overlap through the sequencer (cyc_both={out['both']})"


def test_sequencer_kadd_interlock_precise():
    # The kadd hazard interlock is per-REGISTER: matvec A captures k -> kreg0 (serial),
    # matvec B is NOWAIT and captures k -> kreg1. A norm reading kreg0 must dispatch UNDER
    # B's stream (real overlap), while a norm reading kreg1 must wait for B's capture
    # (done+2) and see the right k. The register-blind interlock serialized the FFN
    # list's norm-under-stream on the card (13.60M vs 13.45M cycles/token, 2026-08-29).
    from gateware.core.vector import VectorUnit
    from gateware.core.csr import EngineCSR
    from gateware.core.sequencer import Sequencer
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

    class _DUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = VectorUnit()
            self.submodules.seq = Sequencer()
            self.submodules.engine = MatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = EngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_seed7),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                setattr(self.submodules, f"model{i}", HBMChannelModel(
                    axi, size=64*1024, init=words_from_bytes(payloads[i]), latency=8))

    dut = _DUT()
    out = {}

    def driver():
        yield from _vec_csr_upload(dut, words)
        yield dut.seq.base.storage.eq(0x6000)
        yield dut.seq.count.storage.eq(5)
        yield
        yield dut.seq.start.storage.eq(1)
        yield dut.seq.start.re.eq(1)
        yield
        yield dut.seq.start.re.eq(0)
        yield
        for _ in range(200000):
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError("sequencer timeout")
        yield
        out["both"] = (yield dut.engine.cyc_both)
        out["k0"]   = (yield dut.seq.kregs[0])
        out["k1"]   = (yield dut.seq.kregs[1])
        out["na"]   = np.frombuffer((yield from _read_sp(dut, 0x3000, 2*M)), dtype="<i2")
        out["nb"]   = np.frombuffer((yield from _read_sp(dut, 0x3800, 2*M)), dtype="<i2")

    run_simulation(dut, [driver()])
    assert (out["k0"], out["k1"]) == (ka, kb), (out["k0"], out["k1"], ka, kb)
    assert np.array_equal(out["na"], nrm_a), "norm via kreg0 (under the stream)"
    assert np.array_equal(out["nb"], nrm_b), "norm via kreg1 (after the capture)"
    # The norm under the stream is short (V1 pipelined unit: ~8 cy/beat), so the overlap
    # window is small in absolute cycles; > 10 is well above zero and below any run.
    assert out["both"] > 10, f"kreg0 consumer did not overlap B (cyc_both={out['both']})"


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
        got = np.frombuffer((yield from _read_sp(dut, 0x4000, 4*M)), dtype="<i4")
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


def test_overlap_vec_throughput_under_stream():
    # The mechanism behind the result packer (docs/history.md 2026-09-07): with one
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
        dut = _OverlapDUT(n_channels, payloads, words_from_bytes(bytes(sp)))
        out = {}

        def driver():
            yield from _vec_csr_upload(dut, words)
            if with_engine:
                for k, v in cmd.items():
                    yield getattr(dut.engine.cmd, k).eq(v)
                yield dut.engine.cmd.valid.eq(1)
                yield
                while not (yield dut.engine.cmd.ready):
                    yield
                yield dut.engine.cmd.valid.eq(0)
                # Let the stream reach steady state before the vector job starts.
                for _ in range(40):
                    yield
            yield from _vec_csr_submit(dut, vec_args)
            t = 0
            w = 0
            for _ in range(60000):
                if with_engine and (yield dut.engine.dbg_gw_we):
                    w += 1
                if not (yield dut.vu.busy):
                    break
                t += 1
                yield
            else:
                raise AssertionError("vec timeout")
            out["vec_cycles"] = t
            if with_engine:
                out["eng_busy_at_vec_end"] = (yield dut.engine.busy)
                for _ in range(60000):
                    if (yield dut.engine.dbg_gw_we):
                        w += 1
                    if not (yield dut.engine.busy):
                        break
                    yield
                else:
                    raise AssertionError("engine timeout")
                out["writes"] = w
                yield
                out["ovf"] = (yield dut.engine.gw_ovf)
                out["res"] = np.frombuffer((yield from _read_sp(dut, 0x2000, 4*M)), dtype="<i4")
            out["vec"] = (yield from _read_sp(dut, 0xA000, 2*nv))

        run_simulation(dut, [driver()])
        return out

    idle  = measure(False)
    under = measure(True)
    assert under["ovf"] == 0, "gather FIFO overflow"
    assert np.array_equal(under["res"], res_exp)
    assert idle["vec"] == bytes(mirror[0xA000:0xA000 + 2*nv])
    assert under["vec"] == bytes(mirror[0xA000:0xA000 + 2*nv])
    ratio = under["vec_cycles"]/idle["vec_cycles"]
    print(f"\nvec under stream: {ratio:.2f}x ({under['vec_cycles']} vs {idle['vec_cycles']} "
          f"cycles); engine writes {under['writes']} for {M} results")
    assert under["eng_busy_at_vec_end"], "stream ended before the vector job: not measured under stream"
    assert under["writes"] <= M//8 + n_channels, (under["writes"], M)
    assert ratio <= 1.3, (f"vector program {ratio:.2f}x slower under the stream "
                          f"({under['vec_cycles']} vs {idle['vec_cycles']} cycles)")

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# BonsaiMatVecEngine end-to-end unit test: BONS-encoded weight shards in HBM channel models,
# x8 in the scratchpad, engine command -> results in scratchpad, compared BIT-EXACTLY against
# trixium.model.bitexact.matvec_ternary()/matvec_q8().

import numpy as np
from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes

class _EngineDUT(Module):
    def __init__(self, n_channels, chan_payloads, sp_init, latency=8, stall_rate=0,
                 dual=True):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.submodules.scratchpad = BonsaiScratchpad(size=64*1024, init=sp_init)
        self.submodules.engine     = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                        dual=dual)
        for i, axi in enumerate(axis):
            model = BonsaiHBMChannelModel(axi, size=64*1024,
                init=words_from_bytes(chan_payloads[i]), latency=latency, stall_rate=stall_rate)
            setattr(self.submodules, f"model{i}", model)

def _run_engine(dut, cmd, n_results, dst):
    results = []

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

    run_simulation(dut, [driver()])
    return np.array(results, dtype=np.int64)

def _case_ternary(n_channels, rows_per_chan, cols, latency, stall_rate, seed,
                  dual=True):
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

    dut = _EngineDUT(n_channels, payloads, sp_init, latency, stall_rate, dual=dual)
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
    # (pre-backpressure this overflowed out_fifo and the op never completed - the n4 SoC
    # boot hang of 2026-07-30).
    _case_ternary(n_channels=4, rows_per_chan=48, cols=128, latency=8, stall_rate=0, seed=9)

def test_matvec_ternary_dual_x():
    # S6 enabler: one ternary weight stream verifies two speculated tokens (qn=1, two x
    # vectors word-interleaved in the x-buffer via its second read port). Chan-major dsts,
    # head B at dst + 4*qstride; bit-exact vs two independent references.
    cols, seed = 256, 91
    for (n_channels, rows_per_chan), lat in [((2, 4), 8), ((4, 3), 8),
                                             ((2, 4), 2), ((4, 6), 2)]:
        rng    = np.random.default_rng(seed + n_channels)
        M      = n_channels*rows_per_chan
        G      = cols//layout.GROUP_TERNARY
        scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
        codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
        xa     = rng.integers(-127, 128, cols, dtype=np.int8)
        xb     = rng.integers(-127, 128, cols, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B)
        exp_a = bx.matvec_ternary(scales, codes, xa, B, R)
        exp_b = bx.matvec_ternary(scales, codes, xb, B, R)

        spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                          scales=scales, values=codes)
        payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                    for c in range(n_channels)]

        qstride = 64
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(xa.tobytes() + xb.tobytes()),
                         lat, 0)
        cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
                   wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
                   sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                   cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, qn=1, qstride=qstride)
        got = _run_engine(dut, cmd, qstride + M, 0x4000)
        assert np.array_equal(got[:M], exp_a), f"N={n_channels} head A"
        assert np.array_equal(got[qstride:qstride + M], exp_b), f"N={n_channels} head B"

def test_matvec_slim_no_dual():
    # Hardware builds without the head-B datapath (--with-dual off): the engine is half
    # the area and every non-qn op must be bit-identical. Ternary + Q8 + striped/ilv
    # scores + AXPY, all with dual=False.
    _case_ternary(n_channels=4, rows_per_chan=8, cols=256, latency=8, stall_rate=2,
                  seed=31, dual=False)

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
    dut = _EngineDUT(n_channels, payloads, words_from_bytes(x8.tobytes()), 8, 0, dual=False)
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
                     8, 0, dual=False)
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
    from trixium.model.bons import BONSReader
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
    from trixium.gateware.csr_front import BonsaiEngineCSR

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
    dut.submodules.front = BonsaiEngineCSR(dut.engine)

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
    from trixium.gateware.loader import BonsaiHBMLoader

    n_channels = 2
    rng   = np.random.default_rng(17)
    src   = rng.integers(0, 256, 512, dtype=np.uint8)  # 16 beats.
    beats = len(src)//32

    class _StoreDUT(Module):
        def __init__(self):
            axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                    for _ in range(n_channels)]
            self.submodules.scratchpad = BonsaiScratchpad(
                size=64*1024, init=words_from_bytes(src.tobytes()))
            self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                        chan_stride=chan_stride)
            self.submodules.loader = BonsaiHBMLoader(axis)
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
            for i, axi in enumerate(axis):
                model = BonsaiHBMChannelModel(axi, size=64*1024, latency=8)
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

def test_matvec_q8_dual_head():
    # Dual-Q GQA scores: one K stream serves two Q-heads (qn=1). x holds both vectors
    # contiguously (head B at xsrc + cols); results land position-major per head, head B
    # offset by qstride words. Bit-exact vs two independent matvec_q8 references.
    cols, seed = 128, 57
    # 8 channels is the shipping configuration; T=9,10,17 are the ragged cases (rows>=1
    # with rem>=1) where the card's sequencer attention wedges.
    for (n_channels, T), lat in [((2, 5), 8), ((2, 8), 8), ((4, 3), 8), ((4, 9), 8),
                                 ((2, 8), 2), ((4, 9), 2),
                                 ((8, 8), 8), ((8, 9), 8), ((8, 10), 8), ((8, 17), 8),
                                 ((8, 9), 2), ((8, 17), 2)]:
        rng    = np.random.default_rng(seed + 16*n_channels + T)
        scales = rng.uniform(0.005, 0.02, (T, cols//layout.GROUP_Q8)).astype(np.float16)
        q8w    = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        xa     = rng.integers(-127, 128, cols, dtype=np.int8)
        xb     = rng.integers(-127, 128, cols, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
        exp_a = bx.matvec_q8(scales, q8w, xa, B, R)
        exp_b = bx.matvec_q8(scales, q8w, xb, B, R)

        rows, rem = T//n_channels, T % n_channels
        payloads = []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            spec_c = TensorSpec("kv", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                max(len(pos), 1), cols,
                                scales=scales[pos] if pos else scales[:1],
                                values=q8w[pos] if pos else q8w[:1])
            payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes() if pos else b"")
        maxlen = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]

        qstride = 64 # Head-B result offset in words.
        x_init  = xa.tobytes() + xb.tobytes()
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x_init), lat, 0)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=1, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=R, xsrc=0, dst=0x4000, rem=rem, ilv=1,
                   qn=1, qstride=qstride)
        got = _run_engine(dut, cmd, qstride + T, 0x4000)
        assert np.array_equal(got[:T], exp_a), f"N={n_channels} T={T} head A"
        assert np.array_equal(got[qstride:qstride + T], exp_b), \
            f"N={n_channels} T={T} head B"

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

def test_engine_axpy_dual_head():
    # Dual-Q GQA weighted-V: one V stream serves two heads' AXPYs (qn=1). p8 vectors
    # contiguous at xsrc (head B at +pad32(T)); per-head Rpre via rshift/rshift2; head B
    # partials land at dst + 4*qstride. Bit-exact vs two independent axpy_rows refs.
    cols, seed = 64, 67
    for n_channels, T in [(2, 3), (2, 8), (4, 5), (8, 9), (8, 17)]:
        rng = np.random.default_rng(seed + 16*n_channels + T)
        ev  = rng.integers(-9, -4, T)
        scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
        v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
        p8a = rng.integers(1, 128, T, dtype=np.int8)
        p8b = rng.integers(1, 128, T, dtype=np.int8)

        _, e16 = bx.fp16_decompose(scales)
        B     = 12 - int(e16.max())
        Rpre_a = bx.axpy_rpre(int(p8a.astype(np.int64).sum()))
        Rpre_b = bx.axpy_rpre(int(p8b.astype(np.int64).sum()) + 1000) # Force different Rpre.

        rows, rem = T//n_channels, T % n_channels
        payloads, exp_a, exp_b = [], [], []
        for c in range(n_channels):
            pos = [t for t in range(T) if t % n_channels == c]
            if pos:
                spec_c = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8,
                                    len(pos), cols, scales=scales[pos], values=v8[pos])
                payloads.append(encode_rows(spec_c, 0, len(pos)).tobytes())
                exp_a.append(bx.axpy_rows(scales[pos], v8[pos], p8a[pos], B, Rpre_a))
                exp_b.append(bx.axpy_rows(scales[pos], v8[pos], p8b[pos], B, Rpre_b))
            else:
                payloads.append(b"")
                exp_a.append(np.zeros(cols, dtype=np.int32))
                exp_b.append(np.zeros(cols, dtype=np.int32))
        maxlen   = max(len(p) for p in payloads)
        payloads = [p.ljust(maxlen, b"\0") for p in payloads]
        exp_a, exp_b = np.concatenate(exp_a), np.concatenate(exp_b)

        # Words; must cover the n_channels*cols head-A partials, so it has to scale with the
        # channel count - a fixed 256 silently overlapped the two heads at 8 channels.
        qstride = n_channels*cols
        x_init  = p8a.tobytes().ljust(32, b"\0") + p8b.tobytes().ljust(32, b"\0")
        dut = _EngineDUT(n_channels, payloads, words_from_bytes(x_init), 8, 0)
        rb  = layout.row_beats(cols, layout.KIND_Q8)
        cmd = dict(kind=4, rows=rows, cols=cols, wbase=0, wlen=rows*rb,
                   sbeats=layout.scale_beats(cols, layout.KIND_Q8),
                   cbeats=layout.code_beats(cols, layout.KIND_Q8),
                   bias=B, rshift=Rpre_a, rshift2=Rpre_b, qn=1, qstride=qstride,
                   xsrc=0, dst=0x4000, rem=rem)
        got = _run_engine(dut, cmd, qstride + n_channels*cols, 0x4000)
        assert np.array_equal(got[:n_channels*cols], exp_a), f"N={n_channels} T={T} head A"
        assert np.array_equal(got[qstride:qstride + n_channels*cols], exp_b), \
            f"N={n_channels} T={T} head B"

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
    w[0] = 1
    w[1] = (f.get("kind", 0) | (f.get("ilv", 0) << 8) | (f.get("supw", 0) << 9)
            | (f.get("rshift_rpre", 0) << 10) | (f.get("qn", 0) << 11))
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

def _desc_vec(pc, args, srcs=None):
    w = [0]*16
    sel = 0
    for i, sv in enumerate(srcs or []):
        sel |= (sv & 3) << (2*i)
    w[0] = 2 | (sel << 16)
    w[1] = pc
    for i, a in enumerate(args):
        w[2 + i] = a & 0xFFFFFFFF
    return b"".join(x.to_bytes(4, "little") for x in w)

def _desc_stop():
    return (0).to_bytes(4, "little")*16

def test_sequencer_v2_chaining():
    # Sequencer v2 primitives in one list, the way seq_build_attn will use them:
    #   reduce_acc (res6 = absmax) -> spill (S7 seeded with prev res6, persisted to SP,
    #   res6 preserved) -> norm_i32 with k from src=3 (k(res6)) -> reduce_acc over a
    #   crafted buffer (res6 = sum p8, standing in for softmax8's S6) -> engine AXPY with
    #   rshift_rpre=1 (rshift = bitlen(res6)+5 from the sequencer, w7 rshift left 0).
    from trixium.gateware.vector import BonsaiVectorUnit
    from trixium.gateware.csr_front import BonsaiEngineCSR
    from trixium.gateware.sequencer import BonsaiSequencer
    from trixium.model import asm as vasm

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
            self.submodules.scratchpad = BonsaiScratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = BonsaiVectorUnit()
            self.submodules.seq = BonsaiSequencer()
            self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = BonsaiEngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
                self.vu.seq_seed7.eq(self.seq.vec_res6), # res6 chaining (S7 seed).
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                model = BonsaiHBMChannelModel(axi, size=64*1024,
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
    from trixium.gateware.vector import BonsaiVectorUnit
    from trixium.gateware.csr_front import BonsaiEngineCSR
    from trixium.gateware.sequencer import BonsaiSequencer
    from trixium.model import asm as vasm

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
            self.submodules.scratchpad = BonsaiScratchpad(size=64*1024,
                                                          init=words_from_bytes(bytes(sp)))
            self.submodules.vu  = BonsaiVectorUnit()
            self.submodules.seq = BonsaiSequencer()
            self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis,
                                                        xbuf_groups=32,
                                                        vecunit=self.vu, seq=self.seq)
            self.submodules.front = BonsaiEngineCSR(self.engine, seq=self.seq)
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            for i, axi in enumerate(axis):
                model = BonsaiHBMChannelModel(axi, size=64*1024,
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
        yield dut.seq.count.storage.eq(4)
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

      .venv/bin/python -m trixium.model.tinygen --channels 8
      .venv/bin/python sim/bonsai_sim.py --image models/tiny/bonsai-tiny-n8.img --run
    """
    from trixium.gateware.vector import BonsaiVectorUnit
    from trixium.gateware.csr_front import BonsaiEngineCSR
    from trixium.gateware.sequencer import BonsaiSequencer
    from trixium.model import asm as vasm

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
                self.submodules.scratchpad = BonsaiScratchpad(
                    size=64*1024, init=words_from_bytes(bytes(sp)))
                self.submodules.vu  = BonsaiVectorUnit()
                self.submodules.seq = BonsaiSequencer()
                self.submodules.engine = BonsaiMatVecEngine(
                    self.scratchpad, axis, xbuf_groups=32, vecunit=self.vu, seq=self.seq)
                self.submodules.front = BonsaiEngineCSR(self.engine, seq=self.seq)
                self.comb += [
                    self.vu.seq_en.eq(self.seq.busy),
                    self.vu.seq_start.eq(self.seq.vec_start),
                    self.vu.seq_pc.eq(self.seq.vec_pc),
                    self.seq.vec_done.eq(self.vu.done),
                    self.seq.vec_res6.eq(self.vu.res6.status),
                ]
                self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
                for i, axi in enumerate(axis):
                    model = BonsaiHBMChannelModel(axi, size=64*1024,
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

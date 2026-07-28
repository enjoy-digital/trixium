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
    def __init__(self, n_channels, chan_payloads, sp_init, latency=8, stall_rate=0):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.submodules.scratchpad = BonsaiScratchpad(size=64*1024, init=sp_init)
        self.submodules.engine     = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=32)
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

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

def test_engine_store_roundtrip():
    # kind-3 STORE (scratchpad -> HBM via the loader's writers), then FETCH back: the KV
    # append datapath. Also checks the done handshake orders after the B drain.
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
            self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=32)
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
    n_channels, cols, seed = 2, 128, 23
    for T in (1, 2, 5, 8):  # positions (total rows); T=1 -> rows=0 on one channel.
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
    n_channels, cols, seed = 2, 64, 41
    for T in (1, 3, 8):
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
    w[1] = f.get("kind", 0) | (f.get("ilv", 0) << 8) | (f.get("supw", 0) << 9)
    w[2] = f.get("rows", 0) | (f.get("rem", 0) << 16)
    w[3] = f.get("cols", 0)
    w[4] = f.get("wbase", 0)
    w[5] = f.get("wlen", 0)
    w[6] = f.get("sbeats", 0) | (f.get("cbeats", 0) << 8)
    w[7] = f.get("bias", 0) | (f.get("rshift", 0) << 16)
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
        assert np.array_equal(out["nrm"], nrm_exp.astype(np.int16)), "norm_i32 auto-k"

    run_simulation(dut, [driver()])

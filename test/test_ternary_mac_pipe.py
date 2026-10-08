"""Pipe-level AXPY test: one BonsaiChannelPipe driven directly, bit-exact vs bx.axpy_rows.

Written for the 2026-09-10 area/timing changes in ternary_mac.py (registered shift
amounts, ONE accumulate adder per lane shared by the four groups): the engine-level AXPY
tests all use cols=64 (two groups) with lockstep readers and no stalls, so they could not
see a wrong bank select, a stale shift amount held across a gap, or the two-beat minimum
distance between updates of the same bank (cols=32: one code beat per row, one scale beat
between rows). This test drives every beat position with random valid gaps and random
source.ready toggling, covers cols in {32, 64, 128} (1..4 banks), rows in {1, 2, 5} with
distinct per-row exponents.
"""
import numpy as np
import pytest
from migen import *

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.ternary_mac import BonsaiChannelPipe, KIND_OP_AXPY


def _beats(payload):
    assert len(payload) % 32 == 0
    return [int.from_bytes(payload[i:i+32], "little") for i in range(0, len(payload), 32)]


def _run_pipe(cols, T, seed, gaps=True):
    rng = np.random.default_rng(seed)
    ev  = rng.integers(-9, -4, T)
    scales = np.array([[2.0**int(e)]*(cols//32) for e in ev], dtype=np.float16)
    v8  = rng.integers(-127, 128, (T, cols), dtype=np.int8)
    p8  = rng.integers(0, 128, T, dtype=np.int8)
    rng.integers(0, 128, T, dtype=np.int8)            # (Keeps the seeds' draw sequence.)
    _, e16 = bx.fp16_decompose(scales)
    B     = 12 - int(e16.max())
    Rpre  = bx.axpy_rpre(int(p8.astype(np.int64).sum()))
    spec = TensorSpec("v", layout.ROLE_ATTN_Q, 0, layout.KIND_Q8, T, cols,
                      scales=scales, values=v8)
    payload = encode_rows(spec, 0, T).tobytes()
    expA = bx.axpy_rows(scales, v8, p8, B, Rpre)

    # x buffer: probs byte t at index t (n_channels=1, chan_id=0).
    xb = bytearray(128)                               # One 1024-bit x word (T <= 32).
    assert T <= 32
    for t in range(T):
        xb[t] = int(p8[t])

    pipe = BonsaiChannelPipe(xbuf_groups=8, chan_id=0, n_channels=1)
    beats = _beats(payload)
    sbeats = layout.scale_beats(cols, layout.KIND_Q8)
    cbeats = layout.code_beats(cols, layout.KIND_Q8)
    got = {}

    def driver():
        # Fill the x buffer: 256-bit quarters of 1024-bit words.
        for q in range(len(xb)//32):
            yield pipe.xw_group.eq(q // 4)
            yield pipe.xw_sub.eq(q % 4)
            yield pipe.xw_data.eq(int.from_bytes(xb[32*q:32*q+32], "little"))
            yield pipe.xw_we.eq(1)
            yield
        yield pipe.xw_we.eq(0)
        yield pipe.kind.eq(KIND_OP_AXPY)
        yield pipe.rows.eq(T)
        yield pipe.scale_beats.eq(sbeats)
        yield pipe.code_beats.eq(cbeats)
        yield pipe.bias.eq(B)
        yield pipe.rshift.eq(Rpre)
        yield
        yield pipe.start.eq(1)
        yield
        yield pipe.start.eq(0)
        for _ in range(6):
            yield
        out = {}
        i = 0
        cycles = 0
        drain_seen = 0
        while cycles < 20000:
            cycles += 1
            # Beat source with random gaps.
            if i < len(beats):
                v = 1 if (not gaps or rng.integers(0, 3) != 0) else 0
                yield pipe.sink.valid.eq(v)
                yield pipe.sink.data.eq(beats[i])
            else:
                yield pipe.sink.valid.eq(0)
            rdy = 1 if (not gaps or rng.integers(0, 3) != 0) else 0
            yield pipe.source.ready.eq(rdy)
            yield
            if i < len(beats) and (yield pipe.sink.valid) and (yield pipe.sink.ready):
                i += 1
            if (yield pipe.source.valid) and (yield pipe.source.ready):
                assert (yield pipe.source.wide) == 1
                row  = (yield pipe.source.row)
                assert (yield pipe.source.head) == 0
                data = (yield pipe.source.data)
                for k in range(8):
                    w = (data >> (32*k)) & 0xFFFFFFFF
                    out[row + k] = w - (1 << 32) if w & (1 << 31) else w
                drain_seen += 1
            want = cols//8
            if i == len(beats) and drain_seen == want:
                break
        else:
            raise AssertionError(f"timeout: beats {i}/{len(beats)} drain {drain_seen}")
        assert (yield pipe.error) == 0 and (yield pipe.overflow) == 0
        got.update(out)

    run_simulation(pipe, [driver()])
    gotA = np.array([got[c] for c in range(cols)], dtype=np.int64)
    assert np.array_equal(gotA, expA), (cols, T, gotA[:8], expA[:8])


@pytest.mark.parametrize("cols", [32, 64, 128])
@pytest.mark.parametrize("T", [1, 2, 5])
def test_pipe_axpy_single(cols, T):
    _run_pipe(cols, T, seed=100 + cols + T)


def test_pipe_axpy_no_gaps():
    # Dense stream: the minimum two-beat distance between same-bank updates at cols=32.
    _run_pipe(32, 5, seed=7, gaps=False)

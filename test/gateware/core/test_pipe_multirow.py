#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Pipe-level multi-row matvec test (doc/plan-multirow.md, step 1): one ChannelPipe(nrows=R)
driven directly, R activation rows against one weight stream, bit-exact per row vs
bx.matvec_ternary.

The x-buffer is filled through xw_group/xw_sub (quarter 4r + q for row r), the weight rows are
BONS-encoded (2-bit, and trit5 through a Trit5Unpacker in front of the pipe), with random valid
gaps on the beat side and random/held-low source.ready on the result side (the intake must
throttle on the result FIFO, never drop: overflow stays 0). Each result entry is checked lane by
lane (row r's int32 in lane r, the unused lanes zero). Ragged row counts (1, 2, 9, 20) and
several K. Mutation checks: the results must not match the expectations of a pipe whose rows all
read row 0's word, whose lane order is reversed within a group, or whose rows are swapped; and
swapping two rows' x vectors must swap their results.
"""

import numpy as np
import pytest

from migen import *

from litex.gen import *

from model import layout
from model import bitexact as bx
from model.bons import TensorSpec, encode_rows
from gateware.core.pipe import ChannelPipe, KIND_OP_TERNARY
from gateware.core.trit5 import Trit5Unpacker


# Helpers ------------------------------------------------------------------------------------------

def _beats(payload):
    assert len(payload) % 32 == 0
    return [int.from_bytes(payload[i:i+32], "little") for i in range(0, len(payload), 32)]


def _s32(w):
    return w - (1 << 32) if w & (1 << 31) else w


class _Trit5Pipe(LiteXModule):
    """Trit5Unpacker -> ChannelPipe, as in PipeIsland (single clock)."""
    def __init__(self, **kw):
        self.unp  = unp  = Trit5Unpacker()
        self.pipe = pipe = ChannelPipe(**kw)
        self.comb += [
            unp.start.eq(pipe.start),
            unp.source.connect(pipe.sink),
        ]


def _tensor(rng, M, cols):
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    return scales, codes, B, R


def _run_pipe(nrows, cols, scales, codes, x8, B, R, seed, trit5=False, gaps=True, hold=0):
    """Stream the tensor through ChannelPipe(nrows) against x8 (nrows, cols).
    Returns (results (nrows, M) int64, error). hold: source.ready low for the first `hold`
    cycles of the stream (result-side back-pressure)."""
    rng  = np.random.default_rng(seed)
    M    = codes.shape[0]
    kind = layout.KIND_TERNARY5 if trit5 else layout.KIND_TERNARY
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    spec.kind = kind
    beats  = _beats(encode_rows(spec, 0, M).tobytes())
    sbeats = layout.scale_beats(cols, kind)
    cbeats = layout.code_beats(cols, kind)
    groups = cols//layout.GROUP_TERNARY

    kw = dict(xbuf_groups=max(groups, 2), chan_id=0, n_channels=1, axpy=False, nrows=nrows)
    if trit5:
        top  = _Trit5Pipe(**kw)
        pipe = top.pipe
        sink = top.unp.sink
    else:
        top  = pipe = ChannelPipe(**kw)
        sink = pipe.sink
    got = {}
    err = []

    def driver():
        # x-buffer: row r, group g, quarter q -> address g, sub 4r + q.
        for g in range(groups):
            for r in range(nrows):
                for q in range(4):
                    chunk = x8[r, 128*g + 32*q:128*g + 32*q + 32].tobytes()
                    yield pipe.xw_group.eq(g)
                    yield pipe.xw_sub.eq(4*r + q)
                    yield pipe.xw_data.eq(int.from_bytes(chunk, "little"))
                    yield pipe.xw_we.eq(1)
                    yield
        yield pipe.xw_we.eq(0)
        yield pipe.kind.eq(KIND_OP_TERNARY)
        yield pipe.rows.eq(M)
        yield pipe.scale_beats.eq(sbeats)
        yield pipe.code_beats.eq(cbeats)
        yield pipe.bias.eq(B)
        yield pipe.rshift.eq(R)
        if trit5:
            yield top.unp.packed.eq(1)
            yield top.unp.scale_beats.eq(sbeats)
            yield top.unp.code_beats.eq(cbeats)
        yield
        yield pipe.start.eq(1)
        yield
        yield pipe.start.eq(0)
        for _ in range(6):
            yield
        i, cycles = 0, 0
        while cycles < 40000:
            cycles += 1
            if i < len(beats):
                v = 1 if (not gaps or rng.integers(0, 3) != 0) else 0
                yield sink.valid.eq(v)
                yield sink.data.eq(beats[i])
            else:
                yield sink.valid.eq(0)
            if cycles <= hold:
                rdy = 0
            else:
                rdy = 1 if (not gaps or rng.integers(0, 3) != 0) else 0
            yield pipe.source.ready.eq(rdy)
            yield
            if i < len(beats) and (yield sink.valid) and (yield sink.ready):
                i += 1
            if (yield pipe.source.valid) and (yield pipe.source.ready):
                assert (yield pipe.source.wide) == 0 and (yield pipe.source.head) == 0
                row  = (yield pipe.source.row)
                data = (yield pipe.source.data)
                assert row not in got, f"row {row} emitted twice"
                lanes = [(data >> (32*k)) & 0xFFFFFFFF for k in range(8)]
                if nrows == 1:
                    assert all(w == lanes[0] for w in lanes), "nrows=1: result replicated in 8 lanes"
                else:
                    assert all(w == 0 for w in lanes[nrows:]), "unused lanes must be zero"
                got[row] = [_s32(w) for w in lanes[:nrows]]
            if i == len(beats) and len(got) == M:
                break
        else:
            raise AssertionError(f"timeout: beats {i}/{len(beats)} results {len(got)}/{M}")
        for _ in range(8):
            yield
        assert (yield pipe.source.valid) == 0, "extra result"
        assert (yield pipe.overflow) == 0
        err.append((yield pipe.error))

    run_simulation(top, [driver()])
    res = np.array([[got[m][r] for m in range(M)] for r in range(nrows)], dtype=np.int64)
    return res, err[0]


def _case(nrows, cols, M, seed, trit5=False, gaps=True, hold=0):
    rng = np.random.default_rng(seed)
    scales, codes, B, R = _tensor(rng, M, cols)
    x8  = rng.integers(-127, 128, (nrows, cols), dtype=np.int8)
    exp = np.array([bx.matvec_ternary(scales, codes, x8[r], B, R) for r in range(nrows)],
                   dtype=np.int64)
    got, err = _run_pipe(nrows, cols, scales, codes, x8, B, R, seed, trit5=trit5, gaps=gaps,
                         hold=hold)
    assert err == 0
    for r in range(nrows):
        assert np.array_equal(got[r], exp[r]), (nrows, cols, M, r, got[r][:4], exp[r][:4])
    return scales, codes, x8, B, R, got


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nrows", [1, 2, 3])
@pytest.mark.parametrize("cols,M", [(128, 1), (256, 2), (256, 9)])
def test_pipe_multirow(nrows, cols, M):
    _case(nrows, cols, M, seed=1000 + 17*nrows + cols + M)


def test_pipe_multirow_wide_k():
    # K = 2048: 16 groups, every x-buffer address and the full scale word (the sim is slow at
    # this width: one case).
    _case(2, 2048, 2, seed=2048)


@pytest.mark.parametrize("nrows", [2, 3])
def test_pipe_multirow_trit5(nrows):
    _case(nrows, 384, 5, seed=50 + nrows, trit5=True)


def test_pipe_multirow_backpressure():
    # 20 one-group rows with the result side held low: the intake must throttle on the result
    # FIFO's headroom (out_fifo_room) and resume without a drop or a duplicate.
    _case(2, 128, 20, seed=77, hold=120)


def test_pipe_multirow_no_gaps():
    _case(2, 128, 9, seed=78, gaps=False)


def test_pipe_multirow_code3():
    rng = np.random.default_rng(5)
    scales, codes, B, R = _tensor(rng, 3, 256)
    codes[1, 200] = 3
    x8 = rng.integers(-127, 128, (2, 256), dtype=np.int8)
    _, err = _run_pipe(2, 256, scales, codes, x8, B, R, seed=5)
    assert err == 1


def test_pipe_multirow_mutations():
    # The comparison can tell the plausible wiring mistakes apart...
    scales, codes, x8, B, R, got = _case(2, 256, 9, seed=91)
    mv = lambda x: bx.matvec_ternary(scales, codes, x, B, R).astype(np.int64)
    x_rev = x8.reshape(2, -1, 128)[:, :, ::-1].reshape(2, -1)
    assert not np.array_equal(got[1], mv(x8[0]))     # Every row reading row 0's x word.
    assert not np.array_equal(got[1], mv(x_rev[1]))  # Lane order reversed within a group.
    assert not np.array_equal(got[0], mv(x8[1]))     # Rows swapped.
    # ...and swapping the two rows' x vectors swaps the results.
    got_sw, err = _run_pipe(2, 256, scales, codes, x8[::-1].copy(), B, R, seed=92)
    assert err == 0
    assert np.array_equal(got_sw[0], got[1]) and np.array_equal(got_sw[1], got[0])

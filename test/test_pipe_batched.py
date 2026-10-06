# SPDX-License-Identifier: BSD-2-Clause
"""Batched matvec on one BonsaiChannelPipe (nbatch, docs/speculative-verify-design.md step 2).

One weight stream, nb token vectors: every code beat is replayed nb times, vector b reading
its x at x-buffer word b*xw and accumulating in its own rotating accumulator. Each vector's
results must equal the single-vector reference bit for bit, with random input gaps and
output backpressure, and nb=1 on an nbatch build must behave as the unbatched pipe."""
import random

import numpy as np
import pytest
from migen import *

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.ternary_mac import BonsaiChannelPipe, KIND_OP_TERNARY, KIND_OP_Q8


def make_case(kind, rows, groups, nb, seed):
    rng = np.random.default_rng(seed)
    ternary = kind == KIND_OP_TERNARY
    cols = groups*(128 if ternary else 32)
    scales = (rng.uniform(-2, 2, (rows, groups))*np.exp2(rng.integers(-14, 2, (rows, groups)))).astype(np.float16)
    values = (rng.integers(0, 3, (rows, cols), dtype=np.uint8) if ternary else
              rng.integers(-128, 128, (rows, cols), dtype=np.int8))
    xs = [rng.integers(-128, 128, cols, dtype=np.int8) for _ in range(nb)]
    _, exp = bx.fp16_decompose(scales)
    bias = bx.choose_scale_bias(exp)
    shift = bx.choose_result_shift(scales, bias, gs_max=bx.GS_MAX if ternary else bx.GS_MAX_Q8)
    fn = bx.matvec_ternary if ternary else bx.matvec_q8
    expected = [fn(scales, values, x, bias, shift) for x in xs]
    # Vector b at x-buffer word b*xw: one word per ternary group, one per 4 Q8 groups.
    xw = groups if ternary else (groups + 3)//4
    xb = bytearray(128*xw*nb)
    for b, x in enumerate(xs):
        xb[128*xw*b:128*xw*b + cols] = x.tobytes()
    spec = TensorSpec('w', layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY if ternary else layout.KIND_Q8,
                      rows, cols, scales=scales, values=values)
    payload = encode_rows(spec, 0, rows).tobytes()
    return dict(kind=kind, rows=rows, groups=groups, nb=nb, bias=bias, shift=shift, xb=bytes(xb),
                sbeats=layout.scale_beats(cols, spec.kind), payload=payload, expected=expected)


def drive(pipe, case, gaps, seed):
    rng = random.Random(seed)
    x = case['xb']
    for offset in range(0, len(x), 32):
        yield pipe.xw_group.eq(offset // 128)
        yield pipe.xw_sub.eq((offset // 32) % 4)
        yield pipe.xw_data.eq(int.from_bytes(x[offset:offset+32], 'little'))
        yield pipe.xw_we.eq(1)
        yield
    yield pipe.xw_we.eq(0)
    for name, value in [('kind', case['kind']), ('rows', case['rows']), ('code_beats', case['groups']),
                        ('scale_beats', case['sbeats']), ('bias', case['bias']),
                        ('rshift', case['shift']), ('nb', case['nb'] - 1)]:
        yield getattr(pipe, name).eq(value)
    yield
    yield pipe.start.eq(1)
    yield
    yield pipe.start.eq(0)
    for _ in range(4):
        yield
    payload = case['payload']
    beats = [int.from_bytes(payload[i:i+32], 'little') for i in range(0, len(payload), 32)]
    sent, offered, result = 0, False, {}
    wanted = case['rows']*case['nb']
    for cycle in range(40000):
        if sent < len(beats):
            # A source holds valid until accepted (stream semantics), gaps between beats.
            offered = offered or not gaps or rng.randrange(3) != 0
            yield pipe.sink.valid.eq(offered)
            yield pipe.sink.data.eq(beats[sent])
        else:
            yield pipe.sink.valid.eq(0)
        yield pipe.source.ready.eq(not gaps or rng.randrange(3) != 0)
        yield
        if (yield pipe.sink.valid) and (yield pipe.sink.ready):
            sent += 1
            offered = False
        assert not (yield pipe.error), f'pipe error at cycle {cycle}'
        assert not (yield pipe.overflow), f'FIFO overflow at cycle {cycle}'
        if (yield pipe.source.valid) and (yield pipe.source.ready):
            row, head = (yield pipe.source.row), (yield pipe.source.head)
            raw = (yield pipe.source.data) & 0xffffffff
            key = (head, row)
            assert key not in result, f'duplicate output {key}'
            result[key] = raw - (1 << 32) if raw & (1 << 31) else raw
        if len(result) == wanted and sent == len(beats):
            for b, expected in enumerate(case['expected']):
                got = np.array([result[b, r] for r in range(case['rows'])], dtype=np.int32)
                assert np.array_equal(got, expected), (case['kind'], case['groups'], b, got, expected)
            for _ in range(16):
                yield
                assert not (yield pipe.source.valid)
            return
    raise AssertionError(f'timeout: {sent}/{len(beats)} beats, {len(result)}/{wanted} results')


@pytest.mark.parametrize('kind', [KIND_OP_TERNARY, KIND_OP_Q8])
@pytest.mark.parametrize('nb,groups,rows,gaps', [
    (1, 3, 5, True),     # nbatch build, one vector: the unbatched behaviour.
    (2, 1, 9, False),    # One code beat per row, dense: back-to-back row-lasts per vector.
    (2, 5, 4, True),
    (3, 2, 6, True),     # Non-power-of-two ring.
    (4, 4, 5, False),
    (4, 7, 3, True),     # Q8: 7 groups -> 2 words per vector (rounded up).
])
def test_pipe_batched_matvec(kind, nb, groups, rows, gaps):
    case = make_case(kind, rows, groups, nb, seed=nb*100 + groups*10 + rows + kind)
    xw = groups if kind == KIND_OP_TERNARY else (groups + 3)//4
    pipe = BonsaiChannelPipe(xbuf_groups=max(8, xw*nb), dual=False, nbatch=4)
    run_simulation(pipe, [drive(pipe, case, gaps, seed=groups + rows)])


def test_pipe_batched_dual_build_nb1():
    # The dual build keeps head B's plumbing: nb=1 on a dual+nbatch pipe is the plain path.
    case = make_case(KIND_OP_TERNARY, 6, 3, 1, seed=5)
    pipe = BonsaiChannelPipe(xbuf_groups=8, dual=True, nbatch=2)
    run_simulation(pipe, [drive(pipe, case, True, seed=9)])

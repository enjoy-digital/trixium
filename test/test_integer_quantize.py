# SPDX-License-Identifier: BSD-2-Clause
import random

import pytest
from migen import run_simulation

from trixium.units.activation import IntegerToQ8


def expected(value, exponent):
    # Exact rational rounding, independent of any finite-width barrel shifter.
    magnitude = abs(value)
    if exponent > 0:
        denominator = 1 << exponent
        rounded = (2*magnitude + denominator)//(2*denominator)
    else:
        rounded = magnitude*(1 << -exponent)
    return min(127, rounded)*(-1 if value < 0 else 1)


def exercise(dut, vectors, stalls=True):
    rng = random.Random(145)
    pending, held, sent = [], None, 0
    for cycle in range(len(vectors)*10 + 100):
        cancel = stalls and cycle in (2, 13, 75, 181)
        yield dut.cancel.eq(cancel)
        yield dut.source.ready.eq(not stalls or rng.randrange(4) != 0)
        yield dut.sink.valid.eq(sent < len(vectors))
        if sent < len(vectors):
            values, exponent = vectors[sent]
            yield dut.sink.data.eq(sum((v & ((1 << dut.input_width) - 1)) << (dut.input_width*i) for i, v in enumerate(values)))
            yield dut.sink.e8.eq(exponent & 1023)
            yield dut.sink.first.eq(sent % 17 == 0)
            yield dut.sink.last.eq(sent % 17 == 16)
        yield
        if cancel:
            assert not (yield dut.sink.ready) and not (yield dut.source.valid)
            pending.clear()
            held = None
            continue
        payload = ((yield dut.source.data), (yield dut.source.e8), (yield dut.source.first), (yield dut.source.last))
        if held is not None:
            assert (yield dut.source.valid) and payload == held
        held = payload if (yield dut.source.valid) and not (yield dut.source.ready) else None
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            pending.append((sum((expected(v, exponent) & 255) << (8*i) for i, v in enumerate(values)),
                            exponent & 1023, sent % 17 == 0, sent % 17 == 16))
            sent += 1
        if (yield dut.source.valid) and (yield dut.source.ready):
            assert pending and payload == pending.pop(0), (cycle, payload)
        if sent == len(vectors) and not pending:
            return
    raise AssertionError('integer quantizer did not drain')


@pytest.mark.parametrize('lanes,width', [(1, 2), (2, 8), (8, 19), (16, 32)])
def test_integer_q8_all_exponents_extremes_backpressure_and_cancel(lanes, width):
    rng = random.Random(184)
    vectors = []
    for exponent in range(-512, 512):
        values = [rng.randrange(-(1 << (width-1)), 1 << (width-1)) for _ in range(lanes)]
        values[0] = -(1 << (width-1)) if exponent & 1 else (1 << (width-1)) - 1
        vectors.append((values, exponent))
    dut = IntegerToQ8(lanes, width)
    run_simulation(dut, exercise(dut, vectors))


def test_integer_q8_half_integer_neighbors():
    vectors = []
    for exponent in range(1, 12):
        for target in (0, 1, 2, 63, 126, 127):
            half = (target << exponent) + (1 << (exponent - 1))
            if half + 1 >= 1 << 18:
                continue
            vectors.append(([half-1, half, half+1, 1-half, -half, -half-1, 0, 1], exponent))
    dut = IntegerToQ8()
    run_simulation(dut, exercise(dut, vectors, stalls=False))


def test_integer_q8_rejects_invalid_configuration():
    for args in ({'lanes': 0}, {'lanes': 3}, {'input_width': 1}, {'input_width': 33}):
        with pytest.raises(ValueError):
            IntegerToQ8(**args)

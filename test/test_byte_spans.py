# SPDX-License-Identifier: BSD-2-Clause
import random

import pytest
from migen import run_simulation

from trixium.gateware.byte_spans import ByteSpanPacker


def drive_spans(dut, spans, seed=10):
    rng = random.Random(seed)
    wanted = b''.join(data[skip:skip+count] for data, skip, count in spans)
    wanted += bytes((-len(wanted)) % 32)
    yield dut.start.eq(1)
    yield
    yield dut.start.eq(0)
    sent, received, offered, held = 0, [], False, None
    for cycle in range(10000):
        if sent < len(spans):
            data, skip, count = spans[sent]
            offered |= rng.randrange(4) != 0
            yield dut.sink.valid.eq(offered)
            yield dut.sink.data.eq(int.from_bytes(data, 'little'))
            yield dut.sink.skip.eq(skip)
            yield dut.sink.count.eq(count)
            yield dut.sink.last.eq(sent == len(spans)-1)
        else:
            yield dut.sink.valid.eq(0)
        yield dut.source.ready.eq(cycle >= 80 and rng.randrange(3) != 0)
        yield
        assert not (yield dut.error)
        assert (yield dut.buffered_bytes) <= 64
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            sent += 1
            offered = False
        fields = ((yield dut.source.data), (yield dut.source.first), (yield dut.source.last))
        valid, ready = (yield dut.source.valid), (yield dut.source.ready)
        if held is not None:
            assert valid and fields == held
        held = fields if valid and not ready else None
        if valid and ready:
            data, first, last = fields
            assert bool(first) == (len(received) == 0)
            assert bool(last) == (len(received)+1 == len(wanted)//32)
            received.append(data.to_bytes(32, 'little'))
        if (yield dut.done):
            assert sent == len(spans)
            assert b''.join(received) == wanted
            return
    raise AssertionError('byte-span timeout')


def test_all_byte_prefix_and_length_combinations_and_reuse():
    dut = ByteSpanPacker()
    rng = random.Random(72)
    spans = [(rng.randbytes(32), skip, count) for skip in range(32) for count in range(1, 33-skip)]
    rng.shuffle(spans)
    def run():
        yield from drive_spans(dut, spans)
        yield from drive_spans(dut, [(b'\xff'*32, 31, 1), (b'\xa5'*32, 0, 32)])
    run_simulation(dut, run())


@pytest.mark.parametrize('skip,count', [(0, 0), (0, 33), (31, 2), (1, 63)])
def test_invalid_span_fails_without_output(skip, count):
    dut = ByteSpanPacker()
    def run():
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        yield dut.sink.valid.eq(1)
        yield dut.sink.skip.eq(skip)
        yield dut.sink.count.eq(count)
        yield dut.sink.last.eq(1)
        yield
        assert (yield dut.sink.ready)
        yield dut.sink.valid.eq(0)
        yield
        assert (yield dut.error) and (yield dut.done) and not (yield dut.busy)
        assert not (yield dut.source.valid)
    run_simulation(dut, run())


def test_cancel_flushes_held_full_and_partial_bytes():
    dut = ByteSpanPacker()
    def run():
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        yield dut.sink.valid.eq(1)
        yield dut.sink.count.eq(32)
        yield dut.sink.data.eq((1 << 256)-1)
        yield
        yield dut.sink.count.eq(17)
        yield
        yield dut.sink.valid.eq(0)
        yield
        assert (yield dut.source.valid) and (yield dut.buffered_bytes) == 49
        yield dut.cancel.eq(1)
        yield
        yield dut.cancel.eq(0)
        yield
        assert not (yield dut.busy) and not (yield dut.source.valid)
        assert (yield dut.buffered_bytes) == 0
        yield from drive_spans(dut, [(bytes(range(32)), 2, 28)])
    run_simulation(dut, run())

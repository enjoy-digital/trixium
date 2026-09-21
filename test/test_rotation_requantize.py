# SPDX-License-Identifier: BSD-2-Clause
"""Exact post-Hadamard whole-vector scaling through the shared scratchpad."""
import math
import random

import pytest
from migen import run_simulation

from test_activation_sp import QuantSP, bits, pack, reference, submit


FIELDS = dict(src=0x1000, dst=0xc000, exponents=0x400, maxima=0x440,
              result_exponent=0x480, elements=1024, requantize=1, protect_start=0, protect_end=128)
SENTINEL = pack([0xdeadbeef]*8, 32)


def prepare(dut, arrays, exponents, maxima=None):
    for index, array in enumerate(arrays):
        assert len(array) == 1024
        for beat in range(128):
            yield dut.sp.mem[FIELDS['src']//32 + index*128 + beat].eq(pack(array[beat*8:beat*8+8], 32))
    if maxima is None:
        maxima = [max(map(abs, array)) for array in arrays]
    for name, values in (('exponents', exponents), ('maxima', maxima)):
        for start in range(0, len(values), 8):
            yield dut.sp.mem[FIELDS[name]//32 + start//8].eq(pack(values[start:start+8], 32))
    for addr in (FIELDS['result_exponent'], FIELDS['dst'] - 32, FIELDS['dst'] + len(arrays)*1024):
        yield dut.sp.mem[addr//32].eq(SENTINEL)


def command(dut, block_count, **changes):
    fields = dict(FIELDS, blocks=block_count)
    fields.update(changes)
    yield from submit(dut, **fields)


def verify(dut, arrays, exponents, stalls=True):
    yield from prepare(dut, arrays, exponents)
    yield from command(dut, len(arrays))
    rng, held = random.Random(527), None
    for cycle in range(len(arrays)*1000 + 100):
        yield dut.grant.eq(not stalls or (cycle % 71 not in range(11, 41) and rng.randrange(3) != 0))
        yield dut.unit.cmd.src.eq(0xffffe0)
        yield dut.unit.cmd.requantize.eq(0)
        yield dut.unit.cmd.maxima.eq(0xffffe0)
        yield
        payload = (yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat)
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        assert not (yield dut.unit.error), ((yield dut.unit.error), cycle)
        if (yield dut.unit.done):
            break
    else:
        raise AssertionError('post-rotation requantization did not complete')
    # This oracle materializes the exact dequantized FP32 values, chooses a
    # whole-vector floating-point absmax and runs the independent FP32 oracle.
    words = [bits(math.ldexp(value, exponent - 5)) for array, exponent in zip(arrays, exponents) for value in array]
    expected_bytes, exponent = reference(words)
    for beat in range(len(words)//32):
        assert (yield dut.sp.mem[FIELDS['dst']//32 + beat]) == int.from_bytes(expected_bytes[32*beat:32*(beat+1)], 'little')
    result = yield dut.sp.mem[FIELDS['result_exponent']//32]
    assert result & 0xffffffff == exponent & 0xffffffff
    assert result >> 32 == SENTINEL >> 32
    assert (yield dut.unit.exponent) == exponent
    assert (yield dut.unit.blocks_done) == len(arrays)
    assert (yield dut.unit.read_beats) == len(arrays)*130
    assert (yield dut.unit.output_beats) == len(arrays)*32
    for addr in (FIELDS['dst'] - 32, FIELDS['dst'] + len(arrays)*1024):
        assert (yield dut.sp.mem[addr//32]) == SENTINEL
    yield


def test_requantization_global_scale_negative_ties_nine_blocks_and_zero_reuse():
    dut = QuantSP(size=65536)
    rng = random.Random(1709)
    arrays = [[rng.randrange(-100000, 100001) for _ in range(1024)] for _ in range(9)]
    # The largest raw maximum is NOT the largest physically scaled maximum.
    arrays[0][0] = 131072
    arrays[-1] = [127, 129, 128, -127, -129, -128, 0, 1]*128
    exponents = [-100, -20, -19, -18, -17, -16, -15, -14, 10]
    def run():
        yield from verify(dut, arrays, exponents)
        yield from verify(dut, [[0]*1024]*2, [-127, 121])
        yield from verify(dut, [[1, -1, 2, -2, 3, -3, 63, -63]*128], [-121], stalls=False)
        yield from verify(dut, [[4095, -4095]*512], [121], stalls=False)
    run_simulation(dut, run())


def test_requantization_exact_result_capacity_and_readonly_metadata_alias():
    dut = QuantSP(size=65536)
    def run():
        yield from prepare(dut, [[0]*1024], [0], [0])
        yield dut.sp.mem[65536//32 - 1].eq(SENTINEL)
        # Read-only arrays may alias; zero is valid for both metadata fields.
        yield from command(dut, 1, maxima=FIELDS['exponents'], result_exponent=65504)
        for _ in range(400):
            assert not (yield dut.unit.error)
            if (yield dut.unit.done):
                break
            yield
        else:
            raise AssertionError('exact-end result did not retire')
        assert (yield dut.sp.mem[65536//32 - 1]) == SENTINEL & ~0xffffffff
        for beat in range(32):
            assert (yield dut.sp.mem[FIELDS['dst']//32 + beat]) == 0
    run_simulation(dut, run())


@pytest.mark.parametrize('changes,mask', [
    ({'elements': 32}, 1), ({'maxima': 0x441}, 1), ({'result_exponent': 0x481}, 1),
    ({'blocks': 65535}, 2), ({'maxima': 0xffffe0}, 2), ({'result_exponent': 0x10000}, 2),
    ({'dst': 0x1000}, 4), ({'dst': 0x400}, 4), ({'dst': 0x440}, 4), ({'dst': 0x480}, 4),
    ({'result_exponent': 0x1000}, 4), ({'result_exponent': 0x400}, 4), ({'result_exponent': 0x440}, 4),
    ({'protect_start': 0xc000, 'protect_end': 0xc020}, 64),
    ({'protect_start': 0x480, 'protect_end': 0x4a0}, 64),
])
def test_requantization_rejects_bounds_before_any_access(changes, mask):
    dut = QuantSP(size=65536)
    def run():
        yield from command(dut, 1, **changes)
        for _ in range(10):
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            yield
        assert (yield dut.unit.busy) and (yield dut.unit.error) & mask
        assert not (yield dut.unit.done)
    run_simulation(dut, run())


@pytest.mark.parametrize('exponent,maximum,mask', [
    (-128, 1, 128), (122, 1, 128), (1024, 1, 128), (0x7fffffff, 1, 128), (-2147483648, 1, 128),
    (0, 131073, 128), (0, 0xffffffff, 128), (-127, 1, 16), (121, 131072, 16),
])
def test_requantization_rejects_all_metadata_before_writes(exponent, maximum, mask):
    dut = QuantSP(size=65536)
    def run():
        first_exponent = -127 if exponent == -127 and maximum == 1 else 0
        yield from prepare(dut, [[1]*1024]*2, [first_exponent, exponent], [1, maximum])
        yield from command(dut, 2)
        for _ in range(40):
            assert not (yield dut.unit.sp_wr_req) and not (yield dut.unit.done)
            yield
        assert (yield dut.unit.error) == mask
    run_simulation(dut, run())


@pytest.mark.parametrize('actual,declared', [(129, 128), (-129, 128), (0x80000000, 128),
                                          (0x7fffffff, 128), (128, 129), (1, 0)])
def test_requantization_detects_stale_metadata_without_publication(actual, declared):
    dut = QuantSP(size=65536)
    def run():
        array = [0]*1024
        array[-1] = actual
        yield from prepare(dut, [array], [0], [declared])
        yield from command(dut, 1)
        for _ in range(800):
            assert not (yield dut.unit.done)
            if (yield dut.unit.error):
                break
            yield
        assert (yield dut.unit.error) == 128 and (yield dut.unit.busy)
        assert (yield dut.sp.mem[FIELDS['result_exponent']//32]) == SENTINEL
        for _ in range(10):
            assert not (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_rd_req)
            yield
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        yield from verify(dut, [[63, -63]*512], [-5], stalls=False)
    run_simulation(dut, run())

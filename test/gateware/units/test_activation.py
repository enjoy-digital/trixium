#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/activation.py: Fp32ToQ8, Fp32AbsmaxWord, IntegerToQ8, ActivationScratchpad
(quantization, post-Hadamard requantization) and their sequencer lists on the shared cluster."""

import math
import random
import struct

import pytest

from migen import *

from litex.gen import LiteXModule

from gateware.units.activation import Fp32AbsmaxWord, Fp32ToQ8, IntegerToQ8, ActivationScratchpad
from gateware.core.scratchpad import Scratchpad
from gateware.core.core import TriXiumConfig
from model.activation import scratchpad_quantize_descriptor, scratchpad_requantize_descriptor

from common import RTL
from common.cluster import SoC, write, install_list
from common.cluster import run as run_cluster
from common.verilator import Simulator, file_key
from common.verilator import run as run_tb

# Fp32ToQ8 / Fp32AbsmaxWord ------------------------------------------------------------------------
# Scalar IEEE float/math oracle, independent of the hardware bit/shift path.

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def expected(words, e8):
    error = int(not -127 <= e8 <= 121) << len(words)
    result = 0
    for lane, word in enumerate(words):
        x = struct.unpack('<f', struct.pack('<I', word))[0]
        if not math.isfinite(x):
            error |= 1 << lane
        elif -127 <= e8 <= 121:
            # Binary32 * a power of two is exact in binary64 in this domain.
            # Values near half integers are exact, not approximated by decimals.
            scaled = math.ldexp(x, -e8)
            q = 127 if abs(scaled) >= 127 else math.floor(abs(scaled) + 0.5)
            if scaled < 0:
                q = -q
            result |= (q & 255) << (8*lane)
    return result, error


def exercise(dut, vectors, stalls=True, cancels=()):
    rng = random.Random(5009)
    pending = []
    index, retired, discarded = 0, 0, 0
    held = None
    offered = False
    for cycle in range(20*len(vectors) + 100):
        cancel = cycle in cancels
        yield dut.cancel.eq(cancel)
        yield dut.source.ready.eq(not stalls or rng.randrange(3) != 0)
        if index < len(vectors):
            words, e8, first, last = vectors[index]
            offered = offered or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offered)
            yield dut.sink.data.eq(sum(w << (32*i) for i, w in enumerate(words)))
            yield dut.sink.e8.eq(e8 & 1023)
            yield dut.sink.first.eq(first)
            yield dut.sink.last.eq(last)
        else:
            yield dut.sink.valid.eq(0)
        yield
        if cancel:
            assert not (yield dut.sink.ready) and not (yield dut.source.valid)
            discarded += len(pending)
            pending.clear()
            held = None
            continue
        payload = ((yield dut.source.data), (yield dut.source.error),
                   (yield dut.source.e8), (yield dut.source.first), (yield dut.source.last))
        if held is not None:
            assert (yield dut.source.valid) and payload == held
        held = payload if (yield dut.source.valid) and not (yield dut.source.ready) else None
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            pending.append((*expected(words, e8), e8 & 1023, first, last))
            index += 1
            offered = False
        if (yield dut.source.valid) and (yield dut.source.ready):
            assert pending, 'stale output after cancellation'
            assert payload == pending.pop(0), (cycle, index, payload)
            retired += 1
        if index == len(vectors) and not pending:
            assert retired + discarded == len(vectors)
            yield dut.sink.valid.eq(0)
            for _ in range(5):
                yield
                assert not (yield dut.source.valid)
            return cycle + 1
    raise AssertionError('quantizer did not drain')


@pytest.mark.parametrize('lanes', [1, 2, 8, 16])
def test_fp32_q8_exact_random_stalls_and_cancellation(lanes):
    rng = random.Random(301)
    vectors = [([rng.getrandbits(32) for _ in range(lanes)], rng.randrange(-140, 140),
                i % 7 == 0, i % 7 == 6) for i in range(300)]
    dut = Fp32ToQ8(lanes)
    run_simulation(dut, exercise(dut, vectors, cancels=(1, 2, 8, 47, 199, 401)))


def test_fp32_q8_all_scale_exponents_and_rounding_neighbors():
    vectors = []
    for e8 in range(-127, 122):
        for half in (0.5, 1.5, 2.5, 63.5, 126.5, 127.5):
            middle = bits(math.ldexp(half, e8))
            words = [middle - 1, middle, middle + 1,
                     (middle - 1) | (1 << 31), middle | (1 << 31), (middle + 1) | (1 << 31), 0, 1 << 31]
            vectors.append((words, e8, half == 0.5, half == 127.5))
    dut = Fp32ToQ8()
    run_tb(dut, exercise(dut, vectors, stalls=False))


@pytest.mark.parametrize('e8', [-512, -128, -127, 0, 121, 122, 511])
def test_fp32_q8_nonfinite_subnormal_and_scale_errors(e8):
    words = [0x7f800000, 0xff800000, 0x7fc00001, 0xff800001, 1, 0x807fffff, 0x7f7fffff, 0xff7fffff]
    dut = Fp32ToQ8()
    run_simulation(dut, exercise(dut, [(words, e8, 1, 1)]*7))


def test_fp32_q8_invalid_lane_configuration():
    for lanes in (0, 3, 32):
        with pytest.raises(ValueError):
            Fp32ToQ8(lanes)
        with pytest.raises(ValueError):
            Fp32AbsmaxWord(lanes)


@pytest.mark.parametrize('lanes', [1, 2, 8, 16])
def test_fp32_absmax_word_elastic_metadata_and_cancel(lanes):
    dut = Fp32AbsmaxWord(lanes)
    rng = random.Random(481)
    vectors = [[rng.getrandbits(32) for _ in range(lanes)] for _ in range(200)]
    vectors[0] = [0x7fc00001]*lanes
    vectors[1] = [0x80000000]*lanes
    def run():
        index, pending, held = 0, [], None
        for cycle in range(1500):
            cancel = cycle in (2, 9, 73, 123)
            yield dut.cancel.eq(cancel)
            yield dut.source.ready.eq(rng.randrange(4) != 0)
            yield dut.sink.valid.eq(index < len(vectors))
            if index < len(vectors):
                yield dut.sink.data.eq(sum(w << (32*i) for i, w in enumerate(vectors[index])))
                yield dut.sink.first.eq(index % 11 == 0)
                yield dut.sink.last.eq(index % 11 == 10)
            yield
            if cancel:
                assert not (yield dut.sink.ready) and not (yield dut.source.valid)
                pending.clear()
                held = None
                continue
            payload = ((yield dut.source.absmax), (yield dut.source.error),
                       (yield dut.source.first), (yield dut.source.last))
            if held is not None:
                assert (yield dut.source.valid) and payload == held
            held = payload if (yield dut.source.valid) and not (yield dut.source.ready) else None
            if (yield dut.sink.valid) and (yield dut.sink.ready):
                values = [struct.unpack('<f', struct.pack('<I', word))[0] for word in vectors[index]]
                finite = [abs(x) for x in values if math.isfinite(x)]
                pending.append((bits(max(finite, default=0.0)), len(finite) != lanes,
                                index % 11 == 0, index % 11 == 10))
                index += 1
            if (yield dut.source.valid) and (yield dut.source.ready):
                assert pending and payload == pending.pop(0)
            if index == len(vectors) and not pending:
                return
        raise AssertionError('word absmax did not drain')
    run_simulation(dut, run())


def test_fp32_q8_compiled_float_oracle_and_reset():
    dut = Fp32ToQ8()
    signals = {'cancel': dut.cancel}
    for direction, endpoint in (('in', dut.sink), ('out', dut.source)):
        for name in ('data', 'e8', 'first', 'last', 'valid', 'ready'):
            signals[f'{direction}_{name}'] = getattr(endpoint, name)
    signals['out_error'] = dut.source.error
    for name, signal in signals.items():
        signal.name_override = name
    sim = Simulator(dut, ios=signals, name='activation_quantize',
                    driver=(RTL / 'activation_quantize.cpp').read_text())
    result = sim.run(timeout=60)
    assert 'ACTIVATION_QUANTIZE_PASS' in result.stdout
    print(result.stdout)


# IntegerToQ8 --------------------------------------------------------------------------------------

def integer_expected(value, exponent):
    # Exact rational rounding, independent of any finite-width barrel shifter.
    magnitude = abs(value)
    if exponent > 0:
        denominator = 1 << exponent
        rounded = (2*magnitude + denominator)//(2*denominator)
    else:
        rounded = magnitude*(1 << -exponent)
    return min(127, rounded)*(-1 if value < 0 else 1)


def integer_exercise(dut, vectors, stalls=True):
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
            pending.append((sum((integer_expected(v, exponent) & 255) << (8*i) for i, v in enumerate(values)),
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
    run_tb(dut, integer_exercise(dut, vectors))


def test_integer_q8_half_integer_neighbors():
    vectors = []
    for exponent in range(1, 12):
        for target in (0, 1, 2, 63, 126, 127):
            half = (target << exponent) + (1 << (exponent - 1))
            if half + 1 >= 1 << 18:
                continue
            vectors.append(([half-1, half, half+1, 1-half, -half, -half-1, 0, 1], exponent))
    dut = IntegerToQ8()
    run_simulation(dut, integer_exercise(dut, vectors, stalls=False))


def test_integer_q8_rejects_invalid_configuration():
    for args in ({'lanes': 0}, {'lanes': 3}, {'input_width': 1}, {'input_width': 33}):
        with pytest.raises(ValueError):
            IntegerToQ8(**args)


# ActivationScratchpad -----------------------------------------------------------------------------

class QuantSP(LiteXModule):
    def __init__(self, size=8192):
        self.sp = Scratchpad(size=size)
        self.unit = ActivationScratchpad(size)
        self.grant = Signal(reset=1)
        self.comb += [self.sp.fill.adr.eq(self.unit.sp_rd_adr), self.unit.sp_rd_dat.eq(self.sp.fill.dat_r),
                     self.unit.sp_wr_gnt.eq(self.grant & self.unit.sp_wr_req),
                     self.sp.gather.adr.eq(self.unit.sp_wr_adr), self.sp.gather.dat_w.eq(self.unit.sp_wr_dat),
                     self.sp.gather.we.eq(Mux(self.grant & self.unit.sp_wr_req, self.unit.sp_wr_we, 0))]


def pack(values, width):
    return sum((v & ((1 << width) - 1)) << (width*i) for i, v in enumerate(values))


def reference(words):
    values = [struct.unpack('<f', struct.pack('<I', word))[0] for word in words]
    assert all(math.isfinite(value) for value in values)
    maximum = max(map(abs, values))
    e8 = math.frexp(maximum)[1] - 7 if maximum else 0
    result, error = expected(words, e8)
    assert error == 0
    return result.to_bytes(len(words), 'little'), e8


def submit(dut, **changed):
    fields = dict(src=0x400, dst=0x1800, exponents=0x1f00, elements=64, blocks=3,
                  protect_start=0, protect_end=64)
    fields.update(changed)
    for name, value in fields.items():
        yield getattr(dut.unit.cmd, name).eq(value)
    yield dut.unit.cmd.valid.eq(1)
    yield
    assert (yield dut.unit.cmd.ready)
    yield dut.unit.cmd.valid.eq(0)
    yield


def valid_command(dut, arrays, stall=True):
    elements, blocks = len(arrays[0]), len(arrays)
    words = [word for array in arrays for word in array]
    for beat in range(len(words)//8):
        yield dut.sp.mem[0x400//32 + beat].eq(pack(words[beat*8:beat*8 + 8], 32))
    sentinel = pack([0xdeadbeef]*8, 32)
    for address in (0x17e0, 0x1800 + len(words), 0x1f00):
        yield dut.sp.mem[address//32].eq(sentinel)
    yield from submit(dut, elements=elements, blocks=blocks)
    rng = random.Random(940)
    held = None
    for cycle in range(4000):
        # Long stalls fill the converter pipeline and reserved-read FIFO, not
        # only the final word register. The credit boundary must remain bounded.
        yield dut.grant.eq(not stall or (cycle % 83 not in range(9, 41) and rng.randrange(3) != 0))
        yield dut.unit.cmd.src.eq(0xffffe0)
        yield dut.unit.cmd.elements.eq(0)
        yield dut.unit.cmd.blocks.eq(65535)
        yield
        payload = (yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat)
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        assert not (yield dut.unit.error), ((yield dut.unit.error), cycle)
        if (yield dut.unit.done):
            break
    else:
        raise AssertionError('resident activation conversion did not complete')
    assert (yield dut.unit.blocks_done) == blocks
    # Eight inputs per read: two passes, or one in the ping-pong mode (several blocks of at
    # most 1,024 elements: the convert pass reads the unit's block buffer).
    passes = 1 if (blocks > 1 and elements <= 1024) else 2
    assert (yield dut.unit.read_beats) == passes*len(words)//8
    assert (yield dut.unit.output_beats) == len(words)//32
    expected_bytes = b''.join(reference(array)[0] for array in arrays)
    for beat in range(len(words)//32):
        assert (yield dut.sp.mem[0x1800//32 + beat]) == int.from_bytes(expected_bytes[32*beat:32*(beat+1)], 'little')
    stats = yield dut.sp.mem[0x1f00//32]
    for index, array in enumerate(arrays):
        assert (stats >> (32*index)) & 0xffffffff == reference(array)[1] & 0xffffffff
    assert stats >> (32*blocks) == sentinel >> (32*blocks)
    assert (yield dut.sp.mem[0x17e0//32]) == sentinel
    assert (yield dut.sp.mem[(0x1800 + len(words))//32]) == sentinel
    yield


@pytest.mark.parametrize("backend", ["verilator", pytest.param("migen", marks=pytest.mark.slow)])
def test_resident_quantization_per_block_and_whole_vector_reuse(backend):
    dut = QuantSP()
    rng = random.Random(500)
    arrays = [[bits(math.ldexp(rng.uniform(-127.9, 127.9), exponent)) for _ in range(256)]
              for exponent in (-100, 0, 100)]
    arrays[0][:4] = [bits(math.ldexp(x, -100)) for x in (0.5, -0.5, 2.5, -2.5)]
    def run():
        yield from valid_command(dut, arrays)
        yield from valid_command(dut, [[v for a in arrays for v in a]], stall=False)
        yield from valid_command(dut, [[0, 0x80000000]*32])
    run_tb(dut, run(), backend)


def test_resident_exponent_tail_at_exact_physical_end():
    dut = QuantSP(size=0x1f20)
    run_simulation(dut, valid_command(dut, [[bits(1.0)]*64]))


@pytest.mark.parametrize('word', [bits(2.0**-121), 0x7f7fffff, 0xff7fffff])
def test_resident_exponent_extremes(word):
    dut = QuantSP()
    run_simulation(dut, valid_command(dut, [[word]*64]))


@pytest.mark.parametrize('changed,mask', [({'elements': 0}, 1), ({'elements': 40}, 1),
    ({'blocks': 0}, 1), ({'src': 1}, 1), ({'dst': 0x1801}, 1), ({'exponents': 0x1f01}, 1),
    ({'elements': 65504, 'blocks': 65535}, 2), ({'src': 0xffffe0}, 2), ({'dst': 0x1fe0}, 2),
    ({'exponents': 0xffffe0}, 2), ({'dst': 0x400}, 4), ({'exponents': 0x400}, 4),
    ({'exponents': 0x1800}, 4), ({'protect_start': 64, 'protect_end': 32}, 1),
    ({'protect_end': 0x4000}, 2), ({'protect_start': 0x1800, 'protect_end': 0x1820}, 64),
    ({'protect_start': 0x1f00, 'protect_end': 0x1f20}, 64)])
def test_resident_quantization_rejects_complete_invalid_spans_before_access(changed, mask):
    dut = QuantSP()
    def run():
        yield from submit(dut, **changed)
        for _ in range(8):
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            yield
        assert (yield dut.unit.busy) and not (yield dut.unit.done)
        assert (yield dut.unit.error) & mask
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.busy) and not (yield dut.unit.error)
    run_simulation(dut, run())


@pytest.mark.parametrize('word,mask', [(0x7f800000, 8), (0xff800000, 8), (0x7fc00001, 8),
    (1, 16), (0x807fffff, 16), (bits(2.0**-122), 16)])
def test_resident_quantization_rejects_numerical_failure_without_writes(word, mask):
    dut = QuantSP()
    def run():
        for beat in range(8):
            yield dut.sp.mem[0x400//32 + beat].eq(pack([word]*8, 32))
        yield from submit(dut, blocks=1)
        for _ in range(80):
            assert not (yield dut.unit.sp_wr_req)
            assert not (yield dut.unit.done)
            yield
        assert (yield dut.unit.busy) and (yield dut.unit.error) == mask
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        yield from valid_command(dut, [[bits(0.5)]*64])
    run_simulation(dut, run())


# ActivationScratchpad Post-Hadamard Requantization ------------------------------------------------
# Exact post-Hadamard whole-vector scaling through the shared scratchpad.

FIELDS = dict(src=0x1000, dst=0xc000, exponents=0x400, maxima=0x440,
              result_exponent=0x480, elements=1024, requantize=1, protect_start=0, protect_end=128)
SENTINEL = pack([0xdeadbeef]*8, 32)


def prepare(dut, arrays, exponents, maxima=None, n=1024):
    for index, array in enumerate(arrays):
        assert len(array) == n
        for beat in range(n//8):
            yield dut.sp.mem[FIELDS['src']//32 + index*(n//8) + beat].eq(pack(array[beat*8:beat*8+8], 32))
    if maxima is None:
        maxima = [max(map(abs, array)) for array in arrays]
    for name, values in (('exponents', exponents), ('maxima', maxima)):
        for start in range(0, len(values), 8):
            yield dut.sp.mem[FIELDS[name]//32 + start//8].eq(pack(values[start:start+8], 32))
    for addr in (FIELDS['result_exponent'], FIELDS['dst'] - 32, FIELDS['dst'] + len(arrays)*n):
        yield dut.sp.mem[addr//32].eq(SENTINEL)


def command(dut, block_count, **changes):
    fields = dict(FIELDS, blocks=block_count)
    fields.update(changes)
    yield from submit(dut, **fields)


def verify(dut, arrays, exponents, stalls=True, n=1024):
    # n: elements per block, 1024 (normalization 2**-5) or 512 (2**-4, quant ABI 1.2).
    yield from prepare(dut, arrays, exponents, n=n)
    yield from command(dut, len(arrays), elements=n)
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
    norm = {1024: 5, 512: 4}[n]
    words = [bits(math.ldexp(value, exponent - norm)) for array, exponent in zip(arrays, exponents) for value in array]
    expected_bytes, exponent = reference(words)
    for beat in range(len(words)//32):
        assert (yield dut.sp.mem[FIELDS['dst']//32 + beat]) == int.from_bytes(expected_bytes[32*beat:32*(beat+1)], 'little')
    result = yield dut.sp.mem[FIELDS['result_exponent']//32]
    assert result & 0xffffffff == exponent & 0xffffffff
    assert result >> 32 == SENTINEL >> 32
    assert (yield dut.unit.exponent) == exponent
    assert (yield dut.unit.blocks_done) == len(arrays)
    assert (yield dut.unit.read_beats) == len(arrays)*(n//8 + 2)
    assert (yield dut.unit.output_beats) == len(arrays)*(n//32)
    for addr in (FIELDS['dst'] - 32, FIELDS['dst'] + len(arrays)*n):
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
    run_tb(dut, run())


def test_requantization_of_512_element_blocks():
    """Blocks of 512 (the Bonsai 2 2B/4B's Hadamard block on the 1,024 unit's half mode): the
    rotation's 2**-4 normalization, mixed exponents and a zero block."""
    dut = QuantSP(size=65536)
    rng = random.Random(512)
    arrays = [[rng.randrange(-65000, 65001) for _ in range(512)] for _ in range(5)]
    arrays[1][7] = 65024
    exponents = [-30, -12, -11, 0, 7]
    def run():
        yield from verify(dut, arrays, exponents, n=512)
        yield from verify(dut, [[0]*512]*3, [-127, 0, 121], n=512)
        yield from verify(dut, [[1, -1, 2, -2, 3, -3, 63, -63]*64], [-121], stalls=False, n=512)
    run_tb(dut, run())


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
    run_tb(dut, run())


# Sequencer Lists on the Shared Cluster ------------------------------------------------------------

def test_activation_descriptor_and_configuration():
    assert scratchpad_quantize_descriptor(src=32, dst=64, exponents=96, elements=1024, blocks=17) == [4, 32, 64, 96, 1024, 17] + [0]*10
    for changes in ({'src': 1}, {'dst': -32}, {'exponents': 1 << 24}, {'elements': 0},
                    {'elements': 40}, {'elements': 65536}, {'blocks': 0}, {'blocks': 65536}):
        args = dict(src=32, dst=64, exponents=96, elements=1024)
        args.update(changes)
        with pytest.raises(ValueError):
            scratchpad_quantize_descriptor(**args)
    # Resident units without the vector unit are a supported build since the YPCB's hyb7 (the
    # hybrid path never runs it); the scratchpad stays within the clients' 13-bit beat addresses.
    TriXiumConfig(with_activation_quantize=True, with_vecunit=False)
    with pytest.raises(ValueError):
        TriXiumConfig(with_activation_quantize=True, scratchpad_size=512*1024)
    assert scratchpad_requantize_descriptor(src=32, dst=64, exponents=96, maxima=128,
        result_exponent=160, blocks=17) == [5, 32, 64, 96, 128, 160, 17] + [0]*9
    for changes in ({'src': 1}, {'dst': -32}, {'exponents': 1 << 24}, {'maxima': 127},
                    {'result_exponent': 1 << 24}, {'blocks': 0}, {'blocks': 65536}):
        args = dict(src=32, dst=64, exponents=96, maxima=128, result_exponent=160, blocks=17)
        args.update(changes)
        with pytest.raises(ValueError):
            scratchpad_requantize_descriptor(**args)


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (0x3fe0, 1)])
def test_quantization_only_profile_validates_full_list_before_fetch(base, count):
    from gateware.core.sequencer import Sequencer
    dut = Sequencer(with_activation_quantize=True, scratchpad_size=16*1024)
    def run():
        yield from write(dut.base, base)
        yield from write(dut.count, count)
        yield from write(dut.start, 1)
        for _ in range(5):
            assert (yield dut.busy) and (yield dut.activation_quantize_error.status) == 1
            assert not (yield dut.sp_active)
            yield
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 128])
def test_requantization_dispatch_without_hadamard_and_completion_publication(failure):
    from gateware.core.sequencer import Sequencer
    dut = Module()
    dut.submodules.seq = seq = Sequencer(with_activation_quantize=True)
    words = scratchpad_requantize_descriptor(src=0x10000, dst=0x8000, exponents=0x800,
        maxima=0x900, result_exponent=0xa00, blocks=17)
    mem = Memory(256, 2, init=[pack(words[:8], 32), pack(words[8:], 32)])
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    def run():
        assert (yield seq.activation_quantize_abi.status) == 0x10002
        yield from write(seq.base, 0)
        yield from write(seq.count, 1)
        yield from write(seq.start, 1)
        cmd = seq.activation_quantize_cmd
        for _ in range(50):
            if (yield cmd.valid):
                break
            yield
        else:
            raise AssertionError('unit 5 did not dispatch without Hadamard')
        expected_fields = dict(src=0x10000, dst=0x8000, exponents=0x800, maxima=0x900,
                               result_exponent=0xa00, blocks=17, elements=1024, requantize=1,
                               protect_start=0, protect_end=64)
        for _ in range(5):
            assert (yield cmd.valid) and not (yield seq.status.fields.dcount)
            for name, value in expected_fields.items():
                assert (yield getattr(cmd, name)) == value
            yield
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        for _ in range(5):
            yield
            assert (yield seq.busy) and not (yield seq.status.fields.dcount)
        yield seq.activation_quantize_done.eq(not failure)
        yield seq.activation_quantize_failed.eq(failure)
        for _ in range(10):
            yield
        assert (yield seq.activation_quantize_error.status) == failure << 8
        assert (yield seq.status.fields.dcount) == (0 if failure else 1)
        assert bool((yield seq.busy)) == bool(failure)
    run_simulation(dut, run())


# Compiled cluster; without Hadamard also on Migen's simulator (~40 s).
@pytest.mark.parametrize('with_hadamard,backend', [(False, "verilator"), (True, "verilator"),
                                                   pytest.param(False, "migen", marks=pytest.mark.slow)])
def test_sequenced_fetch_nowait_quantize_store(with_hadamard, backend):
    dut = SoC(TriXiumConfig(with_activation_quantize=True, with_hadamard=with_hadamard,
                                 scratchpad_size=16*1024), n=1, backend=backend)
    arrays = [[bits(math.ldexp((i - 32)*0.5, e)) for i in range(64)] for e in (-7, 4)]
    raw = [word for a in arrays for word in a]
    expected = b''.join(reference(a)[0] for a in arrays)
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 16, 0, 0, 0, 0x1000, 0] + [0]*5
    quant = scratchpad_quantize_descriptor(src=0x1000, dst=0x800, exponents=0x700, elements=64, blocks=2)
    store = [1, 3, 0, 0, 0x1000, 4, 0, 0, 0x800, 0, 0] + [0]*5

    def run():
        for beat in range(16):
            yield dut.models[0].mem[0x100//32 + beat].eq(pack(raw[8*beat:8*(beat+1)], 32))
        yield from install_list(dut, [fetch, quant, store])
        accepted = False
        for _ in range(3000):
            unit = dut.activation_quantize
            if (yield unit.cmd.valid) and (yield unit.cmd.ready):
                assert not (yield dut.engine.sp_busy)
                assert (yield dut.engine_csr.done_count.status) == 1
                accepted = True
            assert not (yield dut.seq.activation_quantize_error.status)
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('sequenced conversion did not complete')
        assert accepted and (yield dut.seq.status.fields.dcount) == 3
        assert (yield dut.engine_csr.done_count.status) == 2
        for beat in range(4):
            word = int.from_bytes(expected[32*beat:32*(beat+1)], 'little')
            assert (yield dut.scratchpad.mem[0x800//32 + beat]) == word
            assert (yield dut.models[0].mem[0x1000//32 + beat]) == word
        exponents = yield dut.scratchpad.mem[0x700//32]
        assert exponents & ((1 << 64) - 1) == pack([reference(a)[1] for a in arrays], 32)
    run_cluster(dut, run())


def test_compiled_resident_conversion_rotation_full_ffn_and_reset():
    def construct():
        dut = SoC(TriXiumConfig(with_hadamard=True, with_activation_quantize=True), n=1)
        bus = dut.bus
        status = Signal(24)
        dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
        signals = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
        signals.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
            seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re, seq_status=status,
            seq_error=dut.seq.activation_quantize_error.status, had_error=dut.seq.hadamard_error.status,
            reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
            reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
            quant_blocks=dut.activation_quantize.blocks_done, quant_reads=dut.activation_quantize.read_beats,
            quant_writes=dut.activation_quantize.output_beats)
        for name, signal in signals.items():
            signal.name_override = name
        return dict(dut=dut, ios=signals)
    # Runs in the build directory, next to the memory .init files.
    sim = Simulator(construct, key=file_key(__file__, 'activation_cluster'), name='activation_cluster',
                    driver=(RTL / 'activation_cluster.cpp').read_text())
    result = sim.run(timeout=120)
    assert '$readmem file not found' not in result.stdout
    assert 'ACTIVATION_CLUSTER_PASS' in result.stdout
    print(result.stdout)


def test_resident_quantization_ping_pong_rate():
    # Several blocks of 1,024 FP32 (the rotated prepare's quantize): the convert pass reads
    # the unit's block buffer while the next block's scan reads the scratchpad, so a block
    # costs one scratchpad pass (128 reads) plus a few control clocks, not two. Bytes and
    # exponents as the one-block reference.
    elements, blocks = 1024, 4
    rng = random.Random(77)
    arrays = [[struct.unpack('<I', struct.pack('<f', rng.uniform(-3.0, 3.0)*(b + 1)))[0]
               for _ in range(elements)] for b in range(blocks)]
    words = [w for a in arrays for w in a]
    dut = QuantSP(size=64*1024)
    src, dst, exps = 0x400, 0x8400, 0xc400
    cycles = []
    def run():
        for beat in range(len(words)//8):
            yield dut.sp.mem[src//32 + beat].eq(pack(words[beat*8:beat*8 + 8], 32))
        yield from submit(dut, src=src, dst=dst, exponents=exps, elements=elements, blocks=blocks,
                          protect_start=0, protect_end=0)
        for cycle in range(5000):
            yield
            assert not (yield dut.unit.error)
            if (yield dut.unit.done):
                cycles.append(cycle)
                break
        assert (yield dut.unit.read_beats) == len(words)//8
        out = b''
        for i in range(len(words)//32):
            out += (yield dut.sp.mem[dst//32 + i]).to_bytes(32, 'little')
        exp_beat = yield dut.sp.mem[exps//32]
        exponents = [exp_beat >> (32*b) & 0xffffffff for b in range(blocks)]
        for b, array in enumerate(arrays):
            data, e8 = reference(array)
            assert out[b*elements:(b + 1)*elements] == data, b
            assert exponents[b] == e8 & 0xffffffff, (b, exponents[b], e8)
    run_tb(dut, run())
    per_block = cycles[0]/blocks
    print(f"\nquantize: {per_block:.0f} clocks per 1,024-element block")
    assert cycles[0] <= elements//8 + blocks*(elements//8 + 12) + 8, cycles[0]

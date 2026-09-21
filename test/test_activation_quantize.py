# SPDX-License-Identifier: BSD-2-Clause
"""Scalar IEEE float/math oracle, independent of the hardware bit/shift path."""
import math
import random
import struct
from pathlib import Path
import shutil
import subprocess

import pytest
from migen import run_simulation
from migen.fhdl import verilog

from trixium.gateware.activation_quantize import Fp32ToQ8


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
    run_simulation(dut, exercise(dut, vectors, stalls=False))


@pytest.mark.parametrize('e8', [-512, -128, -127, 0, 121, 122, 511])
def test_fp32_q8_nonfinite_subnormal_and_scale_errors(e8):
    words = [0x7f800000, 0xff800000, 0x7fc00001, 0xff800001, 1, 0x807fffff, 0x7f7fffff, 0xff7fffff]
    dut = Fp32ToQ8()
    run_simulation(dut, exercise(dut, [(words, e8, 1, 1)]*7))


def test_fp32_q8_invalid_lane_configuration():
    for lanes in (0, 3, 32):
        with pytest.raises(ValueError):
            Fp32ToQ8(lanes)


def test_fp32_q8_compiled_float_oracle_and_reset(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled quantization check')
    dut = Fp32ToQ8()
    signals = {'cancel': dut.cancel}
    for direction, endpoint in (('in', dut.sink), ('out', dut.source)):
        for name in ('data', 'e8', 'first', 'last', 'valid', 'ready'):
            signals[f'{direction}_{name}'] = getattr(endpoint, name)
    signals['out_error'] = dut.source.error
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / 'activation_quantize.v'
    verilog.convert(dut, ios=set(signals.values()), name='activation_quantize').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/activation_quantize.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'activation_quantize',
                            str(rtl), str(harness.resolve())], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vactivation_quantize')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ACTIVATION_QUANTIZE_PASS' in result.stdout
    print(result.stdout)

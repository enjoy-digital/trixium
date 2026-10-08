#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""IEEE binary32 scalar oracle, independent of the RTL normalization/shifts."""

import math
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from migen import run_simulation
from migen.fhdl import verilog

from trixium.units.restore import I32ToFp32Scaled


# Helpers ------------------------------------------------------------------------------------------

def bits(value):
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return 0xff800000 if value < 0 else 0x7f800000


def from_bits(value):
    return struct.unpack('<f', struct.pack('<I', value))[0]


def signed(value):
    return struct.unpack('<i', struct.pack('<I', value & 0xffffffff))[0]


def expected(words, exponent):
    bad = not -149 <= exponent <= 127
    error = int(bad) << len(words)
    data = 0
    for lane, word in enumerate(words):
        if bad:
            continue
        converted = from_bits(bits(signed(word)))
        restored = bits(converted * from_bits(bits(math.ldexp(1.0, exponent))))
        if restored & 0x7fffffff == 0x7f800000:
            error |= 1 << lane
            restored = 0
        data |= restored << (32*lane)
    return data, error


def vectors(lanes=8):
    rng = random.Random(271525)
    result = []
    # Every exponent encoding, signed INT_MIN, carry at INT_MAX, zero, odd/even
    # tie conversion and the smallest subnormal / normal result boundaries.
    edges = [0, 1, -1, -(1 << 31), (1 << 31) - 1, (1 << 24) + 1,
             (1 << 24) + 3, -((1 << 24) + 3)]
    for exponent in range(-512, 512):
        for start in range(0, 8, lanes):
            word = [edges[(start + i) % 8] & 0xffffffff for i in range(lanes)]
            result.append((word, exponent, exponent % 17 == 0, exponent % 17 == 16))
    for power in range(24, 31):
        quantum = 1 << (power - 23)
        for base in ((1 << power), (1 << (power + 1)) - 2*quantum):
            for offset in range(-3, 4):
                for parity in range(2):
                    tie = base + parity*quantum + quantum//2
                    value = tie + offset
                    for exponent in (-149, -148, -127, -126, -31, 0, 96, 97, 103, 127):
                        values = [signed(value if i % 2 == 0 else -value) & 0xffffffff
                                  for i in range(lanes)]
                        result.append((values, exponent, offset == -3, offset == 3))
    for i in range(700):
        result.append(([rng.getrandbits(32) for _ in range(lanes)],
                       rng.randrange(-149, 128), i % 11 == 0, i % 11 == 10))
    return result


def exercise(dut, inputs, stalls=True, cancels=()):
    rng = random.Random(317)
    pending, held = [], None
    index, retired, discarded, offering = 0, 0, 0, False
    for cycle in range(20*len(inputs) + 100):
        cancel = cycle in cancels
        yield dut.cancel.eq(cancel)
        yield dut.source.ready.eq(not stalls or rng.randrange(3) != 0)
        if index < len(inputs):
            words, exponent, first, last = inputs[index]
            offering = offering or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offering)
            yield dut.sink.data.eq(sum(word << (32*i) for i, word in enumerate(words)))
            yield dut.sink.exponent.eq(exponent & 1023)
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
                   (yield dut.source.exponent), (yield dut.source.first), (yield dut.source.last))
        if held is not None:
            assert (yield dut.source.valid) and payload == held
        held = payload if (yield dut.source.valid) and not (yield dut.source.ready) else None
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            pending.append((*expected(words, exponent), exponent & 1023, first, last))
            index += 1
            offering = False
        if (yield dut.source.valid) and (yield dut.source.ready):
            assert pending, 'stale result after cancellation'
            assert payload == pending.pop(0), (cycle, index, payload)
            retired += 1
        if index == len(inputs) and not pending:
            assert retired + discarded == len(inputs)
            yield dut.sink.valid.eq(0)
            for _ in range(6):
                yield
                assert not (yield dut.source.valid)
            return cycle + 1
    raise AssertionError('projection restoration did not drain')


# Tests --------------------------------------------------------------------------------------------

def test_projection_restore_all_exponents_rounding_boundaries():
    dut = I32ToFp32Scaled()
    run_simulation(dut, exercise(dut, vectors(), stalls=False))


@pytest.mark.parametrize('lanes', [1, 2, 8, 16])
def test_projection_restore_elastic_cancel(lanes):
    rng = random.Random(948)
    inputs = [([rng.getrandbits(32) for _ in range(lanes)], rng.randrange(-160, 140),
               i % 13 == 0, i % 13 == 12) for i in range(400)]
    dut = I32ToFp32Scaled(lanes)
    run_simulation(dut, exercise(dut, inputs, cancels=(1, 2, 11, 78, 143, 301)))


def test_projection_restore_invalid_lanes():
    for lanes in (0, 3, 32):
        with pytest.raises(ValueError):
            I32ToFp32Scaled(lanes)


def test_projection_restore_reference_operation_order():
    # This observable difference forbids independently restoring K-segments
    # and then adding floats. Reduction is first, including int32 wrap.
    a, b = (1 << 24) + 1, -(1 << 24)
    restored, error = expected([a + b], 0)
    assert not error and restored == bits(1.0)
    assert bits(from_bits(bits(a)) + from_bits(bits(b))) == bits(0.0)
    assert expected([signed(0x7fffffff + 1)], 0) == (bits(-(1 << 31)), 0)


def test_projection_restore_compiled_float_oracle(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled restoration check')
    dut = I32ToFp32Scaled()
    signals = {'cancel': dut.cancel}
    for direction, endpoint in (('in', dut.sink), ('out', dut.source)):
        for name in ('data', 'exponent', 'first', 'last', 'valid', 'ready'):
            signals[f'{direction}_{name}'] = getattr(endpoint, name)
    signals['out_error'] = dut.source.error
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / 'projection_restore.v'
    verilog.convert(dut, ios=set(signals.values()), name='projection_restore').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/projection_restore.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'projection_restore',
                            str(rtl), str(harness.resolve())], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vprojection_restore')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PROJECTION_RESTORE_PASS' in result.stdout
    print(result.stdout)


@pytest.mark.parametrize('abi', ['ilp32f', 'ilp32'])
def test_projection_restore_reference_against_rv32(tmp_path, abi):
    cc, qemu = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('qemu-riscv32')
    if not cc or not qemu:
        pytest.skip('target restoration check needs RV32 compiler and qemu-riscv32')
    source = Path(__file__).parent / 'rv32_projection_restore_oracle.c'
    exe = tmp_path / 'restore.elf'
    march = 'rv32imf' if abi == 'ilp32f' else 'rv32im'
    compiled = subprocess.run([cc, f'-march={march}', f'-mabi={abi}', '-O2', '-ffp-contract=off',
                               '-msmall-data-limit=0', '-nostartfiles', '-nostdlib',
                               '-Wl,--no-relax,-e,_start', str(source), '-lm', '-lgcc', '-o', str(exe)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    inputs = [(word, exponent) for words, exponent, _, _ in vectors()
              if -149 <= exponent <= 127 for word in words]
    output = subprocess.run([qemu, str(exe)],
                            input=b''.join(struct.pack('<Ii', word, exponent) for word, exponent in inputs),
                            check=True, capture_output=True, timeout=60).stdout
    assert len(output) == 4*len(inputs)
    for (word, exponent), (actual,) in zip(inputs, struct.iter_unpack('<I', output)):
        value, error = expected([word], exponent)
        if error:
            assert actual == (0xff800000 if word & 0x80000000 else 0x7f800000)
        else:
            assert actual == value, (hex(word), exponent, hex(actual), hex(value))

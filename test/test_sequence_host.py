# SPDX-License-Identifier: BSD-2-Clause
"""Shared C encoders must match Python and never publish partial descriptors."""
import ctypes as ct
from pathlib import Path
import random
import shutil
import struct
import subprocess

import pytest

from trixium.model.activation import (
    scratchpad_quantize_descriptor, scratchpad_requantize_descriptor, scratchpad_restore_descriptor,
)
from trixium.model.embedding import scratchpad_embedding_descriptor
from trixium.model.hadamard import scratchpad_rotation_descriptor
from trixium.model.native_sequence import (
    native_pipe_descriptor, native_projection_descriptor, region_fetch_descriptor,
)

LIB = Path(__file__).resolve().parents[1] / 'trixium/software/lib'


class Descriptor(ct.Structure):
    _fields_ = [('word', ct.c_uint32 * 16)]


class Projection(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'kind', 'rows', 'rem', 'groups', 'bias', 'rshift', 'xsrc', 'dst',
        'interleaved', 'suppress_writes', 'dual', 'qstride')]


SCHEMAS = {
    'hadamard': (scratchpad_rotation_descriptor, 'src signs dst stats blocks inverse'),
    'quantize': (scratchpad_quantize_descriptor, 'src dst exponents elements blocks'),
    'requantize': (scratchpad_requantize_descriptor, 'src dst exponents maxima result_exponent blocks'),
    'native_pipe': (native_pipe_descriptor, 'pipe region offset stride'),
    'native_projection': (native_projection_descriptor, ''),
    'restore': (scratchpad_restore_descriptor, 'src dst exponents elements adjustment'),
    'embedding': (scratchpad_embedding_descriptor, 'src signs dst groups'),
    'region_fetch': (region_fetch_descriptor, 'region offset length dst'),
}


@pytest.fixture(scope='module')
def encoder(tmp_path_factory):
    cc = shutil.which('gcc')
    if not cc:
        pytest.skip('C compiler required')
    library = tmp_path_factory.mktemp('sequence-host') / 'sequence.so'
    result = subprocess.run([cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
        '-shared', '-fPIC', str(LIB / 'bonsai_sequence.c'), '-o', str(library)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    lib = ct.CDLL(str(library))
    for name, (_, fields) in SCHEMAS.items():
        function = getattr(lib, 'bns_seq_desc_'+name)
        function.restype = ct.c_int
        function.argtypes = [ct.POINTER(Descriptor)] + ([ct.POINTER(Projection)] if name == 'native_projection' else [
            ct.c_uint64 if field in ('offset', 'stride') else ct.c_int32 if field == 'adjustment' else ct.c_uint32
            for field in fields.split()])
    lib.bns_seq_desc_stop.argtypes = [ct.POINTER(Descriptor)]
    lib.bns_seq_desc_encode_le.argtypes = [ct.c_void_p, ct.POINTER(Descriptor)]
    return lib


def call(lib, name, arguments, out):
    args = ([ct.byref(Projection(**arguments))] if name == 'native_projection' else
            [arguments[field] for field in SCHEMAS[name][1].split()])
    return getattr(lib, 'bns_seq_desc_'+name)(out, *args)


def base(name):
    return {
        'hadamard': dict(src=32, signs=64, dst=96, stats=128, blocks=17, inverse=1),
        'quantize': dict(src=32, dst=64, exponents=96, elements=17408, blocks=1),
        'requantize': dict(src=32, dst=64, exponents=96, maxima=128, result_exponent=160, blocks=17),
        'native_pipe': dict(pipe=2, region=0xfedcba98, offset=0x123456789abcdef0, stride=0xfedcba9876543210),
        'native_projection': dict(kind=0, rows=3, rem=1, groups=68, bias=17, rshift=5, xsrc=32, dst=68,
                                  interleaved=1, suppress_writes=0, dual=1, qstride=19),
        'restore': dict(src=32, dst=64, exponents=96, elements=173, adjustment=-13),
        'embedding': dict(src=32, signs=64, dst=96, groups=40),
        'region_fetch': dict(region=0xfedcba98, offset=0x123456789abcde00, length=35, dst=32),
    }[name]


@pytest.mark.parametrize('name', SCHEMAS)
def test_c_matches_python_for_boundary_and_random_descriptors(encoder, name):
    rng = random.Random(123)
    cases = [base(name)]
    for _ in range(64):
        args = base(name)
        for field in args:
            if field in ('src', 'signs', 'dst', 'stats', 'xsrc', 'exponents', 'maxima', 'result_exponent'):
                alignment = 4 if name == 'native_projection' and field == 'dst' else 32
                args[field] = rng.randrange((1 << 24)//alignment)*alignment
            elif field in ('offset', 'stride'):
                args[field] = rng.getrandbits(64) & (~31 if name == 'region_fetch' else -1)
            elif field == 'region': args[field] = rng.getrandbits(32)
            elif field in ('pipe', 'rem'): args[field] = rng.randrange(32)
            elif field in ('kind', 'inverse', 'interleaved', 'suppress_writes', 'dual'): args[field] = rng.randrange(2)
            elif field == 'groups': args[field] = rng.randrange(1, 65)*8 if name == 'embedding' else rng.randrange(1, 256)
            elif field == 'bias': args[field] = rng.randrange(128)
            elif field == 'rshift': args[field] = rng.randrange(64)
            elif field == 'length': args[field] = rng.randrange(1, 1 << 24)
            elif field == 'adjustment': args[field] = rng.randrange(-(1 << 31), 1 << 31)
            elif field == 'elements' and name == 'quantize': args[field] = rng.randrange(1, 2048)*32
            else: args[field] = rng.randrange(1, 65536)
        cases.append(args)
    boundaries = {
        'hadamard': [('blocks', 1), ('blocks', 65535), ('inverse', 0)],
        'quantize': [('elements', 32), ('elements', 65504), ('blocks', 65535)],
        'requantize': [('blocks', 1), ('blocks', 65535)],
        'native_pipe': [('pipe', 31), ('offset', 0), ('offset', (1 << 64)-1), ('stride', (1 << 64)-1)],
        'native_projection': [('rows', 0), ('rows', 65535), ('groups', 255), ('qstride', 65535), ('rem', 31)],
        'restore': [('adjustment', -(1 << 31)), ('adjustment', (1 << 31)-1), ('elements', 1), ('elements', 65535)],
        'embedding': [('groups', 8), ('groups', 512)],
        'region_fetch': [('offset', (1 << 64)-32), ('length', (1 << 24)-1), ('dst', (1 << 24)-32)],
    }
    cases += [dict(base(name), **{field: value}) for field, value in boundaries[name]]
    for args in cases:
        out = Descriptor((ct.c_uint32*16)(*[0xdeadbeef]*16))
        assert call(encoder, name, args, ct.byref(out)) == 0
        assert list(out.word) == SCHEMAS[name][0](**args)
        raw = (ct.c_ubyte*66)(*[0xa5]*66)
        assert encoder.bns_seq_desc_encode_le(ct.byref(raw, 1), ct.byref(out)) == 0
        assert bytes(raw)[1:-1] == struct.pack('<16I', *out.word)
        assert raw[0] == raw[-1] == 0xa5


@pytest.mark.parametrize('name', SCHEMAS)
def test_invalid_fields_and_null_never_modify_output(encoder, name):
    args = base(name)
    out = Descriptor((ct.c_uint32*16)(*[0xdeadbeef]*16))
    saved = bytes(out)
    assert call(encoder, name, args, None) == -1
    cases = []
    for field in args:
        if field in ('src', 'signs', 'dst', 'stats', 'xsrc', 'exponents', 'maxima', 'result_exponent'):
            cases += [dict(args, **{field: value}) for value in (1, 1 << 24, 0xffffffff)]
        elif field in ('kind', 'inverse', 'interleaved', 'suppress_writes', 'dual'):
            cases.append(dict(args, **{field: 2}))
        elif field == 'offset' and name == 'region_fetch': cases.append(dict(args, offset=1))
        elif field in ('offset', 'stride', 'region', 'adjustment'): pass  # Full declared C width is encodable.
        else:
            high = {'pipe': 32, 'rem': 32, 'bias': 128, 'rshift': 64, 'length': 1 << 24,
                    'groups': 520 if name == 'embedding' else 256}.get(field, 65536)
            cases.append(dict(args, **{field: high}))
            if field in ('groups', 'length', 'elements', 'blocks'):
                cases.append(dict(args, **{field: 0}))
    if name == 'native_projection':
        cases.append(dict(args, rows=0, rem=0))
        assert encoder.bns_seq_desc_native_projection(ct.byref(out), None) == -1
    if name == 'embedding': cases.append(dict(args, groups=9))
    if name == 'quantize': cases.append(dict(args, elements=33))
    for invalid in cases:
        with pytest.raises(ValueError):
            SCHEMAS[name][0](**invalid)
        assert call(encoder, name, invalid, ct.byref(out)) == -1
        assert bytes(out) == saved


def test_stop_and_null_byte_encoder(encoder):
    assert ct.sizeof(Descriptor) == 64
    out = Descriptor((ct.c_uint32*16)(*[0xdeadbeef]*16))
    assert encoder.bns_seq_desc_stop(None) == -1
    assert encoder.bns_seq_desc_stop(ct.byref(out)) == 0
    assert bytes(out) == bytes(64)
    raw = (ct.c_ubyte*64)(*[0xa5]*64)
    assert encoder.bns_seq_desc_encode_le(None, ct.byref(out)) == -1
    assert encoder.bns_seq_desc_encode_le(raw, None) == -1
    assert bytes(raw) == bytes([0xa5])*64


@pytest.mark.parametrize('target', ['rv32', 'cxx'])
def test_descriptor_library_cross_compiles_without_target_headers(tmp_path, target):
    cc = shutil.which('riscv64-unknown-elf-gcc' if target == 'rv32' else 'g++')
    if not cc and target == 'rv32' and Path('/opt/riscv/bin/riscv64-unknown-elf-gcc').exists():
        cc = '/opt/riscv/bin/riscv64-unknown-elf-gcc'
    if not cc:
        pytest.skip(target+' compiler unavailable')
    flags = ['-march=rv32ima', '-mabi=ilp32', '-std=c11', '-ffreestanding'] if target == 'rv32' else ['-x', 'c++', '-std=c++17']
    result = subprocess.run([cc, *flags, '-O2', '-Wall', '-Wextra', '-Werror', '-c',
        str(LIB / 'bonsai_sequence.c'), '-o', str(tmp_path / 'sequence.o')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Shared C encoders must match Python and never publish partial descriptors."""

import random
import shutil
import struct
import subprocess
import ctypes as ct
from pathlib import Path

import pytest

from trixium.model.activation import (
    scratchpad_quantize_descriptor, scratchpad_requantize_descriptor, scratchpad_restore_descriptor,
    scratchpad_silu_descriptor, scratchpad_norm_descriptor, scratchpad_conv_descriptor,
    scratchpad_add_descriptor,
)
from trixium.model.hadamard import scratchpad_rotation_descriptor

LIB = Path(__file__).resolve().parents[1] / 'trixium/software/lib'


# Helpers ------------------------------------------------------------------------------------------

def _norm_py(l2, src, dst, weight, n, nseg, eps):
    return scratchpad_norm_descriptor(src=src, dst=dst, weight=weight, n=n, nseg=nseg, eps=eps, l2=l2)


class Descriptor(ct.Structure):
    _fields_ = [('word', ct.c_uint32 * 16)]


SCHEMAS = {
    'hadamard': (scratchpad_rotation_descriptor, 'src signs dst stats blocks inverse'),
    'quantize': (scratchpad_quantize_descriptor, 'src dst exponents elements blocks'),
    'requantize': (scratchpad_requantize_descriptor, 'src dst exponents maxima result_exponent blocks'),
    'restore': (scratchpad_restore_descriptor, 'src dst exponents elements adjustment'),
    'silu': (scratchpad_silu_descriptor, 'src_g src_m dst fixmap elements'),
    'norm': (_norm_py, 'l2 src dst weight n nseg eps'),
    'conv': (scratchpad_conv_descriptor, 'x0 x1 x2 x3 w dst channels'),
    'add': (scratchpad_add_descriptor, 'a b dst n'),
}


@pytest.fixture(scope='module')
def encoder(tmp_path_factory):
    cc = shutil.which('gcc')
    if not cc:
        pytest.skip('C compiler required')
    library = tmp_path_factory.mktemp('sequence-host') / 'sequence.so'
    result = subprocess.run([cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
        '-shared', '-fPIC', str(LIB / 'trixium_sequence.c'), '-o', str(library)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    lib = ct.CDLL(str(library))
    for name, (_, fields) in SCHEMAS.items():
        function = getattr(lib, 'bns_seq_desc_'+name)
        function.restype = ct.c_int
        function.argtypes = [ct.POINTER(Descriptor)] + [
            ct.c_int32 if field == 'adjustment' else ct.c_uint32 for field in fields.split()]
    lib.bns_seq_desc_stop.argtypes = [ct.POINTER(Descriptor)]
    lib.bns_seq_desc_encode_le.argtypes = [ct.c_void_p, ct.POINTER(Descriptor)]
    return lib


def call(lib, name, arguments, out):
    args = [arguments[field] for field in SCHEMAS[name][1].split()]
    return getattr(lib, 'bns_seq_desc_'+name)(out, *args)


def base(name):
    return {
        'hadamard': dict(src=32, signs=64, dst=96, stats=128, blocks=17, inverse=1),
        'quantize': dict(src=32, dst=64, exponents=96, elements=17408, blocks=1),
        'requantize': dict(src=32, dst=64, exponents=96, maxima=128, result_exponent=160, blocks=17),
        'restore': dict(src=32, dst=64, exponents=96, elements=173, adjustment=-13),
        'silu': dict(src_g=32, src_m=64, dst=32, fixmap=96, elements=17408),
        'norm': dict(l2=0, src=32, dst=64, weight=96, n=5120, nseg=1, eps=0x358637bd),
        'conv': dict(x0=0, x1=32, x2=64, x3=96, w=128, dst=160, channels=2048),
        'add': dict(a=0, b=32, dst=0, n=5120),
    }[name]


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', SCHEMAS)
def test_c_matches_python_for_boundary_and_random_descriptors(encoder, name):
    rng = random.Random(123)
    cases = [base(name)]
    for _ in range(64):
        args = base(name)
        for field in args:
            if field in ('src', 'signs', 'dst', 'stats', 'xsrc', 'exponents', 'maxima', 'result_exponent',
                         'src_g', 'src_m', 'fixmap', 'x0', 'x1', 'x2', 'x3', 'w', 'weight', 'a', 'b'):
                args[field] = rng.randrange((1 << 24)//32)*32
            elif field == 'inverse': args[field] = rng.randrange(2)
            elif field == 'adjustment': args[field] = rng.randrange(-(1 << 31), 1 << 31)
            elif field == 'elements' and name == 'quantize': args[field] = rng.randrange(1, 2048)*32
            elif field == 'n' and name == 'add': args[field] = rng.randrange(1, 8192)*8
            elif field == 'n': args[field] = rng.randrange(1, 1025)*8
            elif field == 'channels': args[field] = rng.randrange(1, 8192)*8
            elif field == 'eps': args[field] = rng.getrandbits(32)
            elif field == 'l2': args[field] = rng.randrange(2)
            else: args[field] = rng.randrange(1, 65536)
        cases.append(args)
    boundaries = {
        'hadamard': [('blocks', 1), ('blocks', 65535), ('inverse', 0)],
        'quantize': [('elements', 32), ('elements', 65504), ('blocks', 65535)],
        'requantize': [('blocks', 1), ('blocks', 65535)],
        'restore': [('adjustment', -(1 << 31)), ('adjustment', (1 << 31)-1), ('elements', 1), ('elements', 65535)],
        'silu': [('elements', 1), ('elements', 65535), ('fixmap', (1 << 24)-32)],
        'norm': [('n', 8), ('n', 8192), ('nseg', 65535), ('l2', 1), ('eps', (1 << 32)-1)],
        'conv': [('channels', 8), ('channels', 65528)],
        'add': [('n', 8), ('n', 65528)],
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
        if field in ('src', 'signs', 'dst', 'stats', 'xsrc', 'exponents', 'maxima', 'result_exponent',
                         'src_g', 'src_m', 'fixmap', 'x0', 'x1', 'x2', 'x3', 'w', 'weight', 'a', 'b'):
            cases += [dict(args, **{field: value}) for value in (1, 1 << 24, 0xffffffff)]
        elif field == 'inverse':
            cases.append(dict(args, **{field: 2}))
        elif field in ('adjustment', 'eps'): pass  # Full C width is encodable.
        elif field == 'l2': cases.append(dict(args, l2=2))
        elif field == 'n' and name == 'add': cases += [dict(args, n=v) for v in (0, 12, 65536)]
        elif field == 'n': cases += [dict(args, n=v) for v in (0, 12, 8200)]
        elif field == 'channels': cases += [dict(args, channels=v) for v in (0, 12, 65536)]
        else:
            cases.append(dict(args, **{field: 65536}))
            if field in ('elements', 'blocks', 'nseg'):
                cases.append(dict(args, **{field: 0}))
    if name == 'quantize': cases.append(dict(args, elements=33))
    for invalid in cases:
        with pytest.raises(ValueError):
            SCHEMAS[name][0](**invalid)
        assert call(encoder, name, invalid, ct.byref(out)) == -1
        assert bytes(out) == saved


def test_silu_table_load_encoder(encoder):
    encoder.bns_seq_desc_silu_load.argtypes = [ct.POINTER(Descriptor), ct.c_uint32]
    out = Descriptor((ct.c_uint32*16)(*[0xdeadbeef]*16))
    for src in (0, 32, (1 << 24)-32):
        assert encoder.bns_seq_desc_silu_load(ct.byref(out), src) == 0
        assert list(out.word) == scratchpad_silu_descriptor(src_g=src, load=True)
    saved = bytes(out)
    for src in (1, 1 << 24):
        assert encoder.bns_seq_desc_silu_load(ct.byref(out), src) == -1
        assert bytes(out) == saved


def test_silu_table_only_encoder(encoder):
    encoder.bns_seq_desc_silu_table.argtypes = [ct.POINTER(Descriptor)] + [ct.c_uint32]*5
    out = Descriptor((ct.c_uint32*16)(*[0xdeadbeef]*16))
    args = dict(src_g=32, src_m=64, dst=32, fixmap=96, elements=6144)
    assert encoder.bns_seq_desc_silu_table(ct.byref(out), *args.values()) == 0
    assert list(out.word) == scratchpad_silu_descriptor(**args, table_only=True)
    assert out.word[6] == 4
    saved = bytes(out)
    assert encoder.bns_seq_desc_silu_table(ct.byref(out), 33, 64, 32, 96, 8) == -1
    assert bytes(out) == saved
    encoder.bns_seq_desc_silu_table_set.argtypes = [ct.POINTER(Descriptor)] + [ct.c_uint32]*6
    encoder.bns_seq_desc_silu_load_set.argtypes = [ct.POINTER(Descriptor), ct.c_uint32, ct.c_uint32]
    assert encoder.bns_seq_desc_silu_table_set(ct.byref(out), *args.values(), 1) == 0
    assert list(out.word) == scratchpad_silu_descriptor(**args, table_only=True, table_set=1)
    assert encoder.bns_seq_desc_silu_load_set(ct.byref(out), 0x1000, 1) == 0
    assert list(out.word) == scratchpad_silu_descriptor(src_g=0x1000, load=True, table_set=1)
    saved = bytes(out)
    assert encoder.bns_seq_desc_silu_load_set(ct.byref(out), 0x1000, 2) == -1
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
        str(LIB / 'trixium_sequence.c'), '-o', str(tmp_path / 'sequence.o')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_norm_rope_encoder(encoder):
    """ABI 1.2: RoPE added to a built norm descriptor; invalid arguments leave it as it was."""
    encoder.bns_seq_desc_norm.argtypes = [ct.POINTER(Descriptor)] + [ct.c_uint32]*7
    encoder.bns_seq_desc_norm_rope.argtypes = [ct.POINTER(Descriptor), ct.c_uint32, ct.c_uint32]
    out = Descriptor()
    for rope_n, tab in ((64, 0x3E000), (256, 32), (16, (1 << 18) - 32)):
        assert encoder.bns_seq_desc_norm(ct.byref(out), 0, 32, 64, 96, 256, 24, 0x358637bd) == 0
        assert encoder.bns_seq_desc_norm_rope(ct.byref(out), rope_n, tab) == 0
        assert list(out.word) == scratchpad_norm_descriptor(src=32, dst=64, weight=96, n=256, nseg=24,
                                                            eps=0x358637bd, rope_n=rope_n, rope_tab=tab)
    for rope_n, tab in ((8, 32), (272, 32), (24, 32), (64, 33), (64, 1 << 18)):
        assert encoder.bns_seq_desc_norm(ct.byref(out), 0, 32, 64, 96, 256, 24, 0x358637bd) == 0
        saved = bytes(out)
        assert encoder.bns_seq_desc_norm_rope(ct.byref(out), rope_n, tab) == -1
        assert bytes(out) == saved
        if rope_n <= 256:
            with pytest.raises(ValueError):
                scratchpad_norm_descriptor(src=32, dst=64, weight=96, n=256, nseg=24, eps=0x358637bd,
                                           rope_n=rope_n, rope_tab=tab)
    # rope_n past the segment
    assert encoder.bns_seq_desc_norm(ct.byref(out), 0, 32, 64, 96, 128, 1, 0x358637bd) == 0
    assert encoder.bns_seq_desc_norm_rope(ct.byref(out), 256, 32) == -1

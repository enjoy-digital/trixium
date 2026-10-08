#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Independent host profile vs the target libm and actual firmware math helpers."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trixium.model.rv32_math import (
    expf_rv32, expf_bounded, fma32, sigmoid_rv32, sigmoid_table, silu_polynomial, silu_rv32,
    softmax_exp_rv32,
    silu_table,
)

ROOT = Path(__file__).resolve().parents[1]
f32 = np.float32


def test_fma_is_fused_and_preserves_zero_sign():
    a, b = f32(1 + 2**-23), f32(1 - 2**-23)
    assert fma32(a, b, -1) == f32(-2**-46)
    assert f32(f32(a*b) - f32(1)) == 0
    assert fma32(-0.0, 1, -0.0).view('u4') == 0x80000000


def test_silu_grid_endpoints_and_shapes():
    table = silu_table()
    assert not table.flags.writeable
    grid = np.arange(-2047, 2048, dtype=f32)/f32(128)
    assert np.array_equal(silu_rv32(grid).view('u4'), table[1:-1].view('u4'))
    upper = np.nextafter(f32(16), f32(0))
    assert f32(upper + f32(16)) == f32(32)
    assert silu_rv32(upper).view('u4') == table[-1].view('u4')
    assert silu_rv32(np.zeros((2, 3), dtype=f32)).shape == (2, 3)
    assert silu_rv32(-81).view('u4') == 0x80000000
    assert silu_rv32(f32(np.finfo(f32).max)) == np.finfo(f32).max
    for value in (-16, 16):
        assert silu_rv32(value) == silu_polynomial(value)


@pytest.mark.parametrize('x', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_inputs_fail_closed(x):
    for fn in (expf_rv32, expf_bounded, silu_rv32, silu_polynomial):
        with pytest.raises(ValueError, match='finite'):
            fn(x)


def test_exp_profile_domain():
    for x in (-81, 81):
        with pytest.raises(ValueError):
            expf_rv32(x)
    with pytest.raises(ValueError):
        expf_bounded(17)
    assert expf_rv32(0) == 1
    assert expf_rv32(1).view('u4') == 0x402df854
    assert expf_rv32(-1).view('u4') == 0x3ebc5ab2


def test_profile_against_target_libm(tmp_path):
    cc, qemu = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('qemu-riscv32')
    if not cc or not qemu:
        pytest.skip('target-math gate needs the RV32 toolchain and qemu-riscv32')
    exe = tmp_path / 'math.elf'
    # No CRT => no gp initialization. Disable linker relaxation as well as compiler
    # small-data accesses, including accesses originating inside the libm archive.
    subprocess.run([cc, '-march=rv32imf', '-mabi=ilp32f', '-O2', '-ffp-contract=off',
                    '-msmall-data-limit=0', '-nostartfiles', '-nostdlib',
                    '-Wl,--no-relax,-e,_start', '-I', str(ROOT / 'firmware'),
                    str(ROOT / 'test/rv32_math_oracle.c'), '-lm', '-lgcc', '-o', str(exe)],
                   check=True, capture_output=True, timeout=60)
    rng = np.random.default_rng(2701525)
    edges = np.array([-100, -80, -16, -1, -0.0, 0, 1, 16, 80, 100, 1e30, -1e30], dtype=f32)
    near_edges = np.concatenate([edges, np.nextafter(edges, f32(-np.inf)),
                                 np.nextafter(edges, f32(np.inf))])
    bits = rng.integers(0, 2**32, 8192, dtype=np.uint32).view(f32)
    bits = bits[np.isfinite(bits) & (np.abs(bits) <= 80)]
    x = np.concatenate([np.arange(-4096, 4097, dtype=f32)/f32(256),
                        rng.uniform(-80, 80, 8192).astype(f32), bits, near_edges])
    output = subprocess.run([qemu, str(exe)], input=x.astype('<f4').tobytes(),
                            check=True, capture_output=True, timeout=60).stdout
    assert len(output) == len(x)*5*4
    target = np.frombuffer(output, dtype='<u4').reshape(-1, 5)
    for column, fn in enumerate((expf_rv32, silu_polynomial, silu_rv32, sigmoid_rv32)):
        mask = (x >= -80) & (x <= 80) if column == 0 else np.ones(len(x), dtype=bool)
        actual = np.array([fn(v) for v in x[mask]], dtype=f32).view('u4')
        np.testing.assert_array_equal(actual, target[mask, column], err_msg=fn.__name__)
    neg = np.where(x < 0, x, -x).astype(f32)
    np.testing.assert_array_equal(softmax_exp_rv32(neg).view('u4'), target[:, 4], err_msg='softmax_exp')


def test_silu_endpoint_has_no_out_of_bounds_read(tmp_path):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('native C compiler unavailable')
    source = tmp_path / 'endpoint.c'
    source.write_text('''
#include <math.h>
#include "fastmath.h"
int main(void) {
    float lut[TRIXIUM_SILU_LUT_N + 1] = {0};
    lut[TRIXIUM_SILU_LUT_N] = 15.999998f;
    volatile float x = nextafterf(16.0f, 0.0f);
    return trixium_silu_lookup(x, lut) != lut[TRIXIUM_SILU_LUT_N];
}
''')
    exe = tmp_path / 'endpoint'
    subprocess.run([cc, '-O1', '-g', '-ffp-contract=off', '-fsanitize=address,undefined',
                    '-I', str(ROOT / 'firmware'), str(source), '-lm', '-o', str(exe)],
                   check=True, capture_output=True, timeout=60)
    # This stack-only bounds test needs ASan, not LeakSanitizer (which cannot run
    # inside a ptrace-based sandbox). Keep address/undefined checks enabled.
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=60,
                   env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'})


def test_sigmoid_grid_and_tails():
    table = sigmoid_table()
    grid = np.arange(-2047, 2048, dtype=f32)/f32(128)
    assert np.array_equal(sigmoid_rv32(grid).view('u4'), table[1:-1].view('u4'))
    assert sigmoid_rv32(f32(0)) == f32(0.5)
    assert sigmoid_rv32(f32(81)) == f32(1) and sigmoid_rv32(f32(-81)) == f32(0)
    s = sigmoid_rv32(np.linspace(-90, 90, 2001, dtype=f32))
    assert np.all(np.diff(s.astype(np.float64)) >= 0)          # monotone across the seams
    ref = 1/(1 + np.exp(-np.linspace(-20, 20, 4001)))
    got = sigmoid_rv32(np.linspace(-20, 20, 4001).astype(f32)).astype(np.float64)
    assert np.max(np.abs(got - ref)) < 2e-6

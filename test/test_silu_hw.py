# SPDX-License-Identifier: BSD-2-Clause
"""The hardware SiLU-multiply fast path must equal the firmware wherever it does not flag.

Three layers: the integer-exact binary32 add/multiply/FMA helpers against the host FPU
and IEEE fmaf; the fast path (what the RTL implements) against hybrid.c's silu_mul_fv
semantics (rv32_math.silu_rv32, itself checked against the target libm under QEMU and
the compiled FPU SoC); and the fixup set against its documented definition.
"""
import random

import numpy as np
import pytest

from trixium.model import rv32_math
from trixium.model.silu_hw import (
    F32_16, LUT_N, bits_f32, decode, f32_add, f32_bits, f32_fma, f32_mul, f32_sub,
    silu_mul_fast, silu_mul_firmware, tables,
)


def random_finite(rng, n, spread=True):
    out = []
    while len(out) < n:
        b = rng.getrandbits(32)
        if (b >> 23) & 0xFF == 0xFF:
            continue
        out.append(b)
    return out


def test_exact_helpers_match_the_host_fpu():
    rng = random.Random(7)
    a = random_finite(rng, 40000)
    b = random_finite(rng, 40000)
    c = random_finite(rng, 40000)
    for x, y, z in zip(a, b, c):
        fx, fy, fz = bits_f32(x), bits_f32(y), bits_f32(z)
        with np.errstate(all="ignore"):
            assert f32_add(x, y) == f32_bits(fx + fy), (hex(x), hex(y))
            assert f32_mul(x, y) == f32_bits(fx * fy), (hex(x), hex(y))
        want = rv32_math.fma32(fx, fy, fz)
        if np.isfinite(want):
            assert f32_fma(x, y, z) == f32_bits(want), (hex(x), hex(y), hex(z))


def test_signed_zero_rules():
    pz, nz = 0x00000000, 0x80000000
    assert f32_add(pz, nz) == pz and f32_add(nz, nz) == nz
    one = 0x3F800000
    assert f32_sub(one, one) == pz
    assert f32_fma(pz, one, nz) == pz            # +0*1 + -0 = +0
    assert f32_fma(nz, one, nz) == nz            # -0*1 + -0 = -0


def firmware_grid():
    """Every table node, its neighbours, a dense sweep, and the boundaries."""
    xs = set()
    for i in range(LUT_N + 1):
        x = np.float32(-16 + i/128)
        b = f32_bits(x)
        for d in (-2, -1, 0, 1, 2):
            xs.add((b + d) & 0xFFFFFFFF)
    for x in np.linspace(-15.999, 15.999, 20001, dtype=np.float32):
        xs.add(f32_bits(x))
    for x in (16.0, -16.0, 80.0, -80.0, 16.000002, -16.000002, -80.00001, -79.99999,
              0.0, -0.0, 1e-30, -1e-30, 1e-45, -1e-45, 3.4e38, -3.4e38):
        xs.add(f32_bits(np.float32(x)))
    xs.add(0x417FFFFF)                          # the largest float below 16
    xs.add(0xC17FFFFF)                          # and its negative
    return sorted(b for b in xs if (b >> 23) & 0xFF != 0xFF)


MULTIPLIERS = [f32_bits(np.float32(v)) for v in (1.0, -1.0, 0.5, 3.75, -1234.5, 1e-3, 7e-20, 0.0, -0.0)]


def test_fast_path_equals_firmware_on_every_node_and_boundary():
    lut, dlut = tables()
    checked = fixups = 0
    for g in firmware_grid():
        for m in MULTIPLIERS:
            r, fixup = silu_mul_fast(g, m, lut, dlut)
            if fixup:
                fixups += 1
                continue
            want = silu_mul_firmware(g, m)
            assert r == want, (hex(g), hex(m), hex(r), hex(want))
            checked += 1
    assert checked > 150_000, checked


def test_fast_path_equals_firmware_on_random_activations():
    lut, dlut = tables()
    rng = np.random.default_rng(11)
    gates = rng.normal(0, 4, 60000).astype(np.float32)
    mults = rng.normal(0, 2, 60000).astype(np.float32)
    fixups = 0
    for g, m in zip(gates, mults):
        gb, mb = f32_bits(g), f32_bits(m)
        r, fixup = silu_mul_fast(gb, mb, lut, dlut)
        if fixup:
            fixups += 1
            continue
        assert r == silu_mul_firmware(gb, mb), (g, m)
    # Only |g| >= 16 tails flag here: about 6e-5 of a N(0, 4) sample.
    assert fixups < 60, fixups


def test_fixup_set_is_exactly_the_documented_one():
    lut, dlut = tables()
    one = f32_bits(np.float32(1.0))
    def flagged(x):
        return silu_mul_fast(f32_bits(np.float32(x)), one, lut, dlut)[1]
    for x in (-80.0, -79.5, -40.0, -16.5, -16.0, 16.0):
        assert flagged(x), x                    # polynomial tails
    for x in (-80.5, -1e30, 16.000002, 17.0, 1e30, -15.999999, 15.999999, 0.0, -0.0):
        assert not flagged(x), x                # exact without the polynomial, or in the table
    assert silu_mul_fast(0x7FC00000, one, lut, dlut)[1]           # NaN gate
    assert silu_mul_fast(one, 0x7F800000, lut, dlut)[1]           # infinite multiplier
    assert silu_mul_fast(one, 0x00000001, lut, dlut)[1]           # subnormal multiplier
    # A product that underflows to a subnormal is left to the CPU.
    assert silu_mul_fast(f32_bits(np.float32(-10.0)), f32_bits(np.float32(1e-35)), lut, dlut)[1]


def test_fast_path_never_needs_subnormal_fma_operands():
    # The RTL FMA assumes normal operands in the table range: f = u - floor(u) is either
    # zero or at least 2**-13 (a = g + 16 >= 2**-20), and the tables hold normal values.
    lut, dlut = tables()
    for word in lut + dlut:
        assert decode(word)[3] in ("normal", "zero"), hex(word)
    # Smallest a: g = -16 + 2**-20 (the ulp of |g| in [8, 16)).
    a = f32_add(f32_bits(np.float32(-16 + 2.0**-20)), F32_16)
    assert decode(a)[3] == "normal"

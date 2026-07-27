#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Bit-exact numerics spec tests: primitive correctness (RNE, fp16 decomposition), matvec
# accuracy against the float reference, and overflow-bound guarantees.

import numpy as np

from trixium.model import bitexact as bx

def test_rne_shift():
    # Ties to even, floor base, negatives.
    assert bx.rne_shift(3, 1) == 2      # 1.5 -> 2
    assert bx.rne_shift(1, 1) == 0      # 0.5 -> 0 (even)
    assert bx.rne_shift(5, 1) == 2      # 2.5 -> 2 (even)
    assert bx.rne_shift(7, 1) == 4      # 3.5 -> 4
    assert bx.rne_shift(-1, 1) == 0     # -0.5 -> 0
    assert bx.rne_shift(-3, 1) == -2    # -1.5 -> -2
    assert bx.rne_shift(-5, 2) == -1    # -1.25 -> -1
    assert bx.rne_shift(6, 2) == 2      # 1.5 -> 2 (even)
    v = np.arange(-1000, 1000)
    for k in (1, 3, 7):
        expect = np.rint(v/2**k).astype(np.int64) # np.rint = round-half-even.
        assert np.array_equal(bx.rne_shift(v, k), expect), k

def test_fp16_decompose_exact():
    rng = np.random.default_rng(5)
    s   = rng.uniform(-1.0, 1.0, 1000).astype(np.float16)
    s   = np.concatenate([s, np.array([0.0, 6e-8, -6e-8, 6e-5, 65504.0], dtype=np.float16)])
    m11, e16 = bx.fp16_decompose(s)
    recon = m11.astype(np.float64)*np.exp2(e16.astype(np.float64))
    assert np.array_equal(recon, s.astype(np.float64))

def test_matvec_ternary_accuracy():
    rng = np.random.default_rng(42)
    M, K = 64, 512
    G    = K//128
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, K), dtype=np.uint8)
    x8     = rng.integers(-127, 128, K, dtype=np.int8)

    m11, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    res = bx.matvec_ternary(scales, codes, x8, B, R)

    # Float reference of the same computation.
    w   = codes.astype(np.float64) - 1
    ref = ((w*x8.astype(np.float64)[None, :]).reshape(M, G, 128).sum(axis=2)
           *scales.astype(np.float64)).sum(axis=1)
    got = res.astype(np.float64)*np.exp2(-B + R)
    # Error only from per-group shift truncation: tiny relative to result magnitude.
    denom = np.maximum(np.abs(ref), 1e-3)
    assert np.max(np.abs(got - ref)/denom) < 1e-3

def test_matvec_ternary_bounds_hold():
    # Adversarial: max-magnitude activations and codes, wide scale dynamic range.
    rng = np.random.default_rng(7)
    M, K = 16, 12288 # Largest K in the 8B model.
    G    = K//128
    scales = (np.exp2(rng.uniform(-10, 0, (M, G)))*0.5).astype(np.float16)
    codes  = np.full((M, K), 2, dtype=np.uint8)  # All +1.
    x8     = np.full(K, 127, dtype=np.int8)
    m11, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    res = bx.matvec_ternary(scales, codes, x8, B, R) # Asserts internally on overflow.
    assert res.dtype == np.int32

def test_matvec_q8_accuracy():
    rng = np.random.default_rng(3)
    M, K = 32, 256
    G    = K//32
    scales = rng.uniform(0.005, 0.02, (M, G)).astype(np.float16)
    q8w    = rng.integers(-127, 128, (M, K), dtype=np.int8)
    x8     = rng.integers(-127, 128, K, dtype=np.int8)
    m11, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX_Q8)
    res = bx.matvec_q8(scales, q8w, x8, B, R)
    ref = ((q8w.astype(np.float64)*x8.astype(np.float64)[None, :]).reshape(M, G, 32).sum(axis=2)
           *scales.astype(np.float64)).sum(axis=1)
    got = res.astype(np.float64)*np.exp2(-B + R)
    assert np.max(np.abs(got - ref)/np.maximum(np.abs(ref), 1e-2)) < 1e-3

def test_bfp_normalize_roundtrip():
    rng = np.random.default_rng(11)
    v   = rng.integers(-2**28, 2**28, 1024).astype(np.int64)
    m, e = bx.normalize_i32_to_bfp(v, 0)
    assert 2**14 <= np.abs(m.astype(np.int64)).max() < 2**15
    # Quantization error is bounded by half an LSB (2^(e-1)) in absolute terms.
    err = np.abs(bx.bfp_value(m, e) - v)
    assert err.max() <= 2**(e - 1)
    # Zero vector.
    m, e = bx.normalize_i32_to_bfp(np.zeros(8, dtype=np.int64), 3)
    assert e == 3 and np.all(m == 0)

def test_f32_to_bfp():
    rng = np.random.default_rng(13)
    v   = rng.uniform(0.5, 1.5, 2048).astype(np.float32) # Norm-gain-like.
    m, e = bx.f32_to_bfp(v)
    assert 2**14 <= np.abs(m.astype(np.int64)).max() < 2**15
    rel = np.abs(bx.bfp_value(m, e) - v)/np.abs(v)
    assert rel.max() < 2**-13

def test_luts():
    assert bx.rsqrt_lut().shape == (256,)
    assert bx.exp2_lut()[0] == 2**15
    assert bx.recip_lut()[0] == 2**15
    assert np.all(np.diff(bx.exp2_lut()) > 0)
    assert np.all(np.diff(bx.rsqrt_lut()) < 0)

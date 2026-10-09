#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# qformats unit tests: pack/unpack roundtrips and a scalar transliteration of the PrismML
# llama.cpp fork's C reference (ggml-quants.c) to pin the exact bit layout.

import numpy as np

from model.qformats import (
    QK_Q2_0, BLOCK_Q2_0, q2_0_pack, q2_0_unpack, q2_0_dequant,
    QK_Q8_0, q8_0_pack, q8_0_unpack, q8_0_dequant, q8_0_quant,
)

# Scalar transliteration of dequantize_row_q2_0 (the C reference).
def _q2_0_dequant_scalar(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    nb  = raw.size//BLOCK_Q2_0
    out = np.zeros(nb*QK_Q2_0, dtype=np.float32)
    for i in range(nb):
        block = raw[i*BLOCK_Q2_0:(i+1)*BLOCK_Q2_0]
        d     = block[0:2].copy().view(np.float16)[0].astype(np.float32)
        qs    = block[2:]
        for j in range(QK_Q2_0):
            q = (int(qs[j//4]) >> ((j % 4)*2)) & 0x03
            out[i*QK_Q2_0 + j] = (q - 1)*d
    return out

def test_q2_0_roundtrip():
    rng    = np.random.default_rng(1234)
    nb     = 8
    scales = rng.uniform(0.001, 0.1, nb).astype(np.float16)
    codes  = rng.integers(0, 3, (nb, QK_Q2_0), dtype=np.uint8) # Ternary: 0..2.
    raw    = q2_0_pack(scales, codes)
    assert raw.size == nb*BLOCK_Q2_0
    s2, c2 = q2_0_unpack(raw)
    assert np.array_equal(s2, scales)
    assert np.array_equal(c2, codes)

def test_q2_0_matches_c_reference():
    rng    = np.random.default_rng(42)
    nb     = 4
    scales = rng.uniform(0.001, 0.1, nb).astype(np.float16)
    codes  = rng.integers(0, 4, (nb, QK_Q2_0), dtype=np.uint8) # Include reserved code 3.
    raw    = q2_0_pack(scales, codes)
    assert np.array_equal(q2_0_dequant(raw), _q2_0_dequant_scalar(raw))

def test_q2_0_dequant_values():
    # One block, known codes: q=0 -> -d, q=1 -> 0, q=2 -> +d.
    codes       = np.ones((1, QK_Q2_0), dtype=np.uint8)
    codes[0, 0] = 0
    codes[0, 1] = 2
    raw = q2_0_pack(np.array([0.5], dtype=np.float16), codes)
    w   = q2_0_dequant(raw)
    assert w[0] == -0.5 and w[1] == +0.5 and np.all(w[2:] == 0.0)

def test_q8_0_roundtrip():
    rng    = np.random.default_rng(7)
    nb     = 16
    scales = rng.uniform(0.001, 0.1, nb).astype(np.float16)
    quants = rng.integers(-127, 128, (nb, QK_Q8_0), dtype=np.int8)
    raw    = q8_0_pack(scales, quants)
    s2, q2 = q8_0_unpack(raw)
    assert np.array_equal(s2, scales)
    assert np.array_equal(q2, quants)

def test_q8_0_quant_dequant_close():
    rng = np.random.default_rng(3)
    w   = rng.normal(0, 1, 4*QK_Q8_0).astype(np.float32)
    wq  = q8_0_dequant(q8_0_quant(w))
    assert np.abs(w - wq).max() < 0.02 # |err| <= d/2 with d ~= amax/127.

def test_q1_0_roundtrip_and_semantics():
    from model.qformats import (
        QK_Q1_0, BLOCK_Q1_0, q1_0_pack, q1_0_unpack, q1_0_dequant, q1_0_to_ternary_codes)
    rng    = np.random.default_rng(9)
    nb     = 6
    scales = rng.uniform(0.001, 0.1, nb).astype(np.float16)
    bits   = rng.integers(0, 2, (nb, QK_Q1_0), dtype=np.uint8)
    raw    = q1_0_pack(scales, bits)
    assert raw.size == nb*BLOCK_Q1_0
    s2, b2 = q1_0_unpack(raw)
    assert np.array_equal(s2, scales)
    assert np.array_equal(b2, bits)
    # C reference semantics: bit j at (qs[j//8] >> (j%8)) & 1; w = bit ? d : -d.
    w = q1_0_dequant(raw).reshape(nb, QK_Q1_0)
    for i in (0, nb - 1):
        for j in (0, 1, 7, 8, 127):
            byte = raw[i*BLOCK_Q1_0 + 2 + j//8]
            bit  = (int(byte) >> (j % 8)) & 1
            expect = float(scales[i]) if bit else -float(scales[i])
            assert w[i, j] == np.float32(expect)
    # Ternary-code bridge: 0 -> code 0 (-1), 1 -> code 2 (+1).
    s3, codes = q1_0_to_ternary_codes(raw)
    assert np.array_equal(s3, scales)
    assert np.array_equal(codes, bits*2)


# PrismML's 2026-06 formats: PQ2_0 (type 142) is the g128 block under its current name;
# PTQ1_0 (type 143) packs the same trits 5 per byte; type 42 became upstream's g64 block.

def _ptq1_0_decode_reference(raw):
    """Straight port of dequantize_row_ptq1_0 (ggml-quants.c, prism branch), scalar."""
    from model.qformats import BLOCK_PTQ1_0
    raw = np.frombuffer(raw, dtype=np.uint8)
    pow3 = [1, 3, 9, 27, 81, 243]
    out = []
    for i in range(raw.size//BLOCK_PTQ1_0):
        b  = raw[i*BLOCK_PTQ1_0:(i + 1)*BLOCK_PTQ1_0]
        qs, qh = b[0:24], b[24:26]
        j = 0
        for c in (32, 16, 8):
            while j + c <= 24:
                for n in range(5):
                    for m in range(c):
                        q  = (int(qs[j + m])*pow3[n]) & 0xFF
                        out.append(((q*3) >> 8) - 1)
                j += c
        for n in range(4):
            for h in range(2):
                q = (int(qh[h])*pow3[n]) & 0xFF
                out.append(((q*3) >> 8) - 1)
    return np.array(out, dtype=np.int64)


def test_ptq1_0_pack_unpack_matches_c_reference():
    from model.qformats import ptq1_0_pack, ptq1_0_unpack, ptq1_0_dequant, BLOCK_PTQ1_0
    rng    = np.random.default_rng(7)
    codes  = rng.integers(0, 3, (37, 128), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.1, 37).astype(np.float16)
    raw    = ptq1_0_pack(scales, codes)
    assert raw.size == 37*BLOCK_PTQ1_0
    s2, c2 = ptq1_0_unpack(raw)
    assert np.array_equal(c2, codes) and np.array_equal(s2, scales)
    # The vectorized decoder against the scalar port of the C loop, element order included.
    ref = _ptq1_0_decode_reference(raw)
    assert np.array_equal(ref, codes.astype(np.int64).reshape(-1) - 1)
    w = ptq1_0_dequant(raw)
    assert np.allclose(w, ((codes.astype(np.float32) - 1)*scales.astype(np.float32)[:, None]).reshape(-1))


def test_ptq1_0_and_pq2_0_agree_on_the_same_weights():
    from model.qformats import (ptq1_0_pack, ptq1_0_dequant, q2_0_pack, q2_0_dequant,
                                             DEQUANT, GGML_PQ2_0, GGML_PTQ1_0, GGML_Q2_0)
    rng    = np.random.default_rng(8)
    codes  = rng.integers(0, 3, (16, 128), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.1, 16).astype(np.float16)
    assert np.array_equal(ptq1_0_dequant(ptq1_0_pack(scales, codes)),
                          q2_0_dequant(q2_0_pack(scales, codes)))
    assert DEQUANT[GGML_PQ2_0] is DEQUANT[GGML_Q2_0] and DEQUANT[GGML_PTQ1_0] is ptq1_0_dequant


def test_q2_0_group_detection():
    from model.qformats import q2_0_group
    assert q2_0_group(np.zeros(34*10, np.uint8), 1280) == 128
    assert q2_0_group(np.zeros(18*20, np.uint8), 1280) == 64
    import pytest
    with pytest.raises(ValueError):
        q2_0_group(np.zeros(100, np.uint8), 1280)

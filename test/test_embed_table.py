#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The pre-dequantized embedding table (ROLE_TOK_EMB16 / _EXP) is bit-exact with the
firmware's integer decode.

bons.embed_bfp_rows is a vectorized transcription of embed_row_bfp (firmware/infer.c) - the
scalar reference below is the sim's embed_bfp_int (test_sim_smoke.py) lifted verbatim. The
table lets the card fetch a token's residual row with two engine fetches instead of ~47k
cycles of CPU decode (-2.6%/token); a mismatch here would be a silent numerics
change that the tiny-model token gate could only see for the tokens it happens to use.
"""

import types

import numpy as np
import pytest

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import embed_bfp_rows


# Helpers ------------------------------------------------------------------------------------------

def embed_bfp_int_ref(codes, scales):
    """Scalar reference (test_sim_smoke.embed_bfp_int) for one row."""
    m11, e16 = bx.fp16_decompose(scales)
    e16min = int(e16.min())
    sh  = np.minimum(e16 - e16min, 20).astype(np.int64)
    q   = codes.astype(np.int64).reshape(-1, 128) - 1
    amax = 0
    for g in range(q.shape[0]):
        if np.any(q[g] != 0):
            gm = int(abs(int(m11[g]))) << int(sh[g])
            amax = max(amax, gm)
    bl = amax.bit_length()
    k2 = bl - 15
    out = np.zeros(q.size, dtype=np.int64)
    for g in range(q.shape[0]):
        net = int(sh[g]) - k2
        v2 = q[g]*int(m11[g])
        out[g*128:(g+1)*128] = (v2 << net) if net >= 0 else (v2 >> -net)
    return out.astype(np.int16), e16min + k2


def _rows(rng, n, cols):
    G = cols//layout.GROUP_TERNARY
    codes  = rng.integers(0, 3, size=(n, cols)).astype(np.uint8)
    scales = np.exp2(rng.uniform(-14, 4, size=(n, G))).astype(np.float16)
    scales *= rng.choice([-1.0, 1.0], size=(n, G)).astype(np.float16)
    # Edge cases: an all-zero group, an all-zero row, a group whose scale is exactly a power
    # of two (the bit_length guard), an exponent spread past the 20-bit cap, a group with
    # codes only where the scale is tiny.
    codes[0, :128] = 1
    codes[1, :] = 1
    scales[2, :] = np.float16(2.0**-6)
    scales[3, 0] = np.float16(2.0**-14); scales[3, 1:] = np.float16(8.0)
    scales[4, 0] = np.float16(2.0**-14); codes[4, 128:] = 1
    scales[5, :] = np.float16(1.0)
    return codes, scales


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cols", [256, 2048])
def test_embed_bfp_rows_matches_scalar_reference(cols):
    rng = np.random.default_rng(7)
    n = 300
    codes, scales = _rows(rng, n, cols)
    spec = types.SimpleNamespace(values=codes, scales=scales)
    mant, exps = embed_bfp_rows(spec)
    assert mant.shape == (n, cols) and mant.dtype == np.int16
    assert exps.shape == (n,)
    for r in range(n):
        m_ref, e_ref = embed_bfp_int_ref(codes[r], scales[r])
        assert np.array_equal(mant[r], m_ref), f"row {r} mantissas differ"
        assert int(exps[r]) == e_ref, f"row {r} exponent {exps[r]} != {e_ref}"


def test_embed_bfp_rows_mantissa_range():
    """Rows with any nonzero code land in [2^14, 2^15) at their absmax (the rmsnorm/residual
    ratchet in the firmware assumes the embedding arrives normalized)."""
    rng = np.random.default_rng(11)
    codes, scales = _rows(rng, 64, 2048)
    mant, _ = embed_bfp_rows(types.SimpleNamespace(values=codes, scales=scales))
    for r in range(64):
        if not np.any(codes[r] != 1):
            assert not mant[r].any()
            continue
        amax = int(np.abs(mant[r].astype(np.int32)).max())
        assert (1 << 14) <= amax < (1 << 15), (r, amax)

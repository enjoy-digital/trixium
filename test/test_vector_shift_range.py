#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The rmsnorm programs' A6 (gamma-exponent shift) over the range the per-tensor NORM16
exponents produce (layout.norm16_exp: the 1.7B spans -15..-7, i.e. A6 from -19 to -11),
RTL against the interpreter. The suites only ever ran A6 = -17 (the old fixed -13)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from trixium.model import asm
from test_vector import _compare, _compare_words, SP_SIZE

E_NORM = -11


@pytest.mark.parametrize("e16", [-15, -14, -13, -10, -8, -7])
def test_rtl_rmsnorm_shift_range(e16):
    rng = np.random.default_rng(100 + e16)
    n   = 128
    es  = -12
    m   = np.clip(rng.normal(0, 9000, n), -32767, 32767).astype("<i2")
    # Gammas that use the tensor's range: absmax near 32767 at this exponent.
    w   = np.abs(rng.normal(1.0, 0.3, n)) + 0.05
    w16 = np.clip(np.round(w/w.max()*32000), -32767, 32767).astype("<i2")
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]         = m.tobytes()
    sp[4096:4096+2*n] = w16.tobytes()
    eps_int = max(1, int(round(1e-6*2.0**(-2*es))))
    args = [0, 4096, 8192, n//16, -int(np.log2(n)), eps_int, e16 - E_NORM - 15]
    _compare("rmsnorm", bytes(sp), args)

@pytest.mark.parametrize("e16", [-15, -13, -8])
def test_rtl_rmsnorm_h_shift_range(e16):
    rng = np.random.default_rng(200 + e16)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*hd] = rng.integers(-32000, 32000, hd).astype("<i2").tobytes()
    _compare_words(asm.prog_rmsnorm_h(), sp, [0, 0x2000, 0, hd//16, -7, 17, e16 - E_NORM - 15, H])

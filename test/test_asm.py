#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Microcode interpreter tests: the interpreter is the golden model for the vector unit RTL;
# these tests validate the STOCK PROGRAMS against independent numpy/float references
# (semantics + accuracy), before any RTL exists.

import numpy as np

from trixium.model import asm
from trixium.model import bitexact as bx

def _run(prog_name, sp, args):
    words = asm.assemble(asm.PROGRAMS[prog_name]())
    return asm.VecInterp(sp).run(words, args)

def test_assembler_encodes():
    words = asm.assemble(asm.prog_quant8())
    assert all(0 <= w < 2**32 for w in words)
    assert (words[-1] >> 26) == asm.OP["DONE"]

def test_resid_add():
    rng = np.random.default_rng(1)
    n   = 128 # ints (8 beats).
    a   = rng.integers(-30000, 30000, n).astype(np.int16)
    b   = rng.integers(-30000, 30000, n).astype(np.int16)
    sp  = bytearray(4096)
    sp[0:2*n]        = a.tobytes()
    sp[1024:1024+2*n] = b.tobytes()
    # e_a = -3, e_b = -5 -> e_out = max+1 = -2: shifts -1 and -3.
    _run("resid_add", sp, [0, 1024, 2048, n//16, -1, -3])
    got = np.frombuffer(bytes(sp[2048:2048+2*n]), dtype=np.int16)
    expect = np.clip((a.astype(np.int64) >> 1) + (b.astype(np.int64) >> 3), -32767, 32767)
    assert np.array_equal(got, expect)

def test_quant8():
    rng = np.random.default_rng(2)
    n   = 256
    x   = rng.integers(-32767, 32768, n).astype(np.int16)
    sp  = bytearray(4096)
    sp[0:2*n] = x.tobytes()
    _run("quant8", sp, [0, 0, 2048, n//16])
    got = np.frombuffer(bytes(sp[2048:2048+n]), dtype=np.int8)
    expect = np.clip(bx.rne_shift(x.astype(np.int64), 8), -127, 127)
    assert np.array_equal(got, expect)

def test_norm_i32():
    rng = np.random.default_rng(3)
    n   = 64 # int32 count (8 source beats -> 4 loop iterations).
    v   = rng.integers(-2**28, 2**28, n).astype(np.int32)
    sp  = bytearray(4096)
    sp[0:4*n] = v.tobytes()
    k = max(0, int(np.abs(v).max()).bit_length() - 14)
    S, _ = _run("norm_i32", sp, [0, 0, 2048, n//16, k])
    got = np.frombuffer(bytes(sp[2048:2048+2*n]), dtype=np.int16)
    expect = np.clip(v.astype(np.int64) >> k, -32767, 32767)
    assert np.array_equal(got, expect)
    assert S[6] == int(np.abs(expect).max())

def test_silu_mul_accuracy():
    rng = np.random.default_rng(4)
    n   = 256
    # Gate/up as BFP: values in ~[-8, 8] with exponents eg = eu = -12.
    eg = eu = -12
    g16 = rng.integers(-30000, 30000, n).astype(np.int16)
    u16 = rng.integers(-30000, 30000, n).astype(np.int16)
    gv  = g16.astype(np.float64)*2.0**eg
    uv  = u16.astype(np.float64)*2.0**eu

    sp = bytearray(8192)
    sp[0:2*n]         = g16.tobytes()
    sp[2048:2048+2*n] = u16.tobytes()
    # A4 = right-shift for f: f_int = mg*739*2^(eg+6) => shift right by -(eg+6).
    a4 = -(eg + 6)
    a6 = -15 # (mg*sig>>15)*mu >> 15; output exponent eo = eg+eu+15.
    _run("silu_mul", sp, [0, 2048, 4096, n//16, a4, 0, a6])
    got = np.frombuffer(bytes(sp[4096:4096+2*n]), dtype=np.int16).astype(np.float64)
    got_val = got*2.0**(eg + eu + 15)

    expect = gv/(1.0 + np.exp(-gv))*uv
    # Accuracy: absolute error bounded by the output LSB scale plus Q15 sigmoid error.
    scale = 2.0**(eg + eu + 15)
    err   = np.abs(got_val - expect)
    tol   = 4*scale + 2e-3*np.abs(expect)
    assert np.all(err < tol), f"max err {err.max()}, worst rel {np.max(err/np.maximum(np.abs(expect), 1e-9))}"

def test_silu_mul_zero_and_extremes():
    # Zero gate -> zero output; large |gate| -> silu ~ g (pos) or ~0 (neg).
    eg = eu = -12
    g16 = np.array([0, 32000, -32000, 1, -1]*4 + [0]*12, dtype=np.int16)[:16]
    u16 = np.full(16, 16384, dtype=np.int16)
    sp = bytearray(4096)
    sp[0:32]   = g16.tobytes()
    sp[512:544] = u16.tobytes()
    asm.VecInterp(sp).run(asm.assemble(asm.prog_silu_mul()),
                          [0, 512, 1024, 1, -(eg + 6), 0, -15])
    got = np.frombuffer(bytes(sp[1024:1056]), dtype=np.int16).astype(np.float64)
    got_val = got*2.0**(eg + eu + 15)
    gv = g16.astype(np.float64)*2.0**eg
    uv = u16.astype(np.float64)*2.0**eu
    expect = gv/(1.0 + np.exp(-gv))*uv
    assert np.all(np.abs(got_val - expect) < 4*2.0**(eg+eu+15) + 3e-3*np.abs(expect))

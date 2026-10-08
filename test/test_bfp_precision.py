#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Precision invariants of the resident-BFP loop.

These exist because the token gate could not see the bug they cover. That gate runs the
tiny model - two layers - and asserts the sim matches `hybrid_generate`; but the failure
was mantissa bits leaking away *per layer*, so it needed a dozen layers to show, and both
sides of the comparison leaked identically. The 1.7B produced fluent garbage while every
test passed.

So: assert the arithmetic invariants directly, over a realistic layer count, and assert the
microcode agrees with the exponent algebra firmware does in C.
"""

import numpy as np
import pytest

from trixium.model import asm as vasm
from trixium.model import bitexact as bx

E_NORM = -11


# Reference implementations ------------------------------------------------------------------------
#
# The same rules firmware applies (infer.c: op_quant8_sp, op_resid_add_sp) and the mirror
# applies (test_sim_smoke.hybrid_generate).

def quant8_shift(amax):
    return 8 if amax == 0 else int(amax).bit_length() - 7


def quant8(m, measure=True):
    sh  = quant8_shift(int(np.abs(m).max())) if measure else 8
    m64 = np.asarray(m, dtype=np.int64)
    v   = (m64 << (8 - sh)) if sh <= 8 else (m64 >> (sh - 8))
    return np.clip(bx.rne_shift(v, 8), -127, 127).astype(np.int8), sh


def resid_add(am, ea, bm, eb):
    emin  = min(ea, eb)
    bound = ((int(np.abs(am).max()) << (ea - emin)) +
             (int(np.abs(bm).max()) << (eb - emin)))
    eo     = emin + max(0, bound.bit_length() - 15)
    sa, sb = eo - ea, eo - eb
    a64, b64 = np.asarray(am, dtype=np.int64), np.asarray(bm, dtype=np.int64)
    v = (((a64 >> sa) if sa >= 0 else (a64 << -sa)) +
         ((b64 >> sb) if sb >= 0 else (b64 << -sb)))
    return np.clip(v, -32767, 32767), eo


def mantissa_bits(m):
    a = int(np.abs(m).max())
    return a.bit_length()


# Tests --------------------------------------------------------------------------------------------

def test_programs_fit_the_instruction_bram():
    """The upload wraps instead of failing: overflow silently overwrites program 0.

    That is not hypothetical - adding one program pushed the total to 259 words against a
    256-word BRAM, which clobbered resid_add's ARG setup, and the only symptom was one
    changed CRC in a ten-program sweep.
    """
    total = sum(len(vasm.assemble(fn())) for fn in vasm.PROGRAMS.values())
    assert total <= vasm.IMEM_DEPTH, (
        f"{total} words of microcode against a {vasm.IMEM_DEPTH}-word BRAM")


def test_resid_add_keeps_the_residual_mantissa():
    """Twenty-eight layers of residual updates must not grind the mantissa away.

    e_out = max(e_a, e_b) + 1 - the bound for two full 15-bit mantissas - costs a bit on
    every call regardless of what the values need. At two calls per layer that exhausted an
    int16 by layer 11 of the 1.7B: every element became 0 or +-2**ex, and the residual's own
    quantization error then grew 4x per layer.

    Both operands have to carry *real* magnitudes for this to mean anything: if the delta is
    regenerated at full mantissa scale every call, it keeps re-floating the sum and the
    ratchet stays hidden. What actually happens is that the delta's real size stays roughly
    constant while e_out climbs, so the delta lands further and further down the mantissa.
    """
    def to_bfp(v):
        amax = float(np.abs(v).max())
        e    = int(np.floor(np.log2(amax))) - 14
        return np.clip(np.rint(v/2.0**e), -32767, 32767).astype(np.int64), e

    rng    = np.random.default_rng(0)
    n      = 256
    xf     = rng.normal(0.0, 1.0, n)          # Residual as a real vector.
    xm, ex = to_bfp(xf)
    worst  = 15
    for layer in range(28):
        for _ in range(2):                    # Attention residual, then FFN residual.
            df     = rng.normal(0.0, 0.5, n)  # A layer's contribution: real size ~constant.
            dm, ed = to_bfp(df)
            xf     = xf + df
            xm, ex = resid_add(xm, ex, dm, ed)
            worst  = min(worst, mantissa_bits(xm))
    assert worst >= 14, f"residual mantissa fell to {worst} bits over 28 layers"
    # And the value it represents must still be the residual, not just wide. 56 int16
    # roundings leave ~0.3%; the collapsed-mantissa failure is order 100%, so the two are
    # nowhere near each other and this threshold does not need to be tight.
    got = xm*2.0**ex
    rel = float(np.sqrt(((got - xf)**2).mean())/np.sqrt((xf**2).mean()))
    assert rel < 0.02, f"residual drifted from the value it should hold (rel {rel:.4f})"


def test_resid_add_preserves_value():
    """Exponent bookkeeping has to be right, not just wide."""
    rng = np.random.default_rng(1)
    for ea, eb in ((-5, -5), (0, 0), (2, -4), (-1, -1), (-8, 3)):
        am = rng.integers(-30000, 30000, 128)
        bm = rng.integers(-16383, 16383, 128)
        vm, eo = resid_add(am, ea, bm, eb)
        want = am*2.0**ea + bm*2.0**eb
        got  = vm*2.0**eo
        assert np.allclose(got, want, rtol=0, atol=2.0**eo), (ea, eb)


def test_quant8_uses_the_int8_range():
    """A fixed >>8 wasted most of int8 on the values this actually sees.

    After rmsnorm the mantissas sit near 2**11 (E_NORM = -11, activations of order 1), so
    >>8 produced int8 inputs of about +-9 - 3 of the 7 bits - and ~0.30 rel-rms on every
    matvec output, which compounded into garbage tokens.
    """
    rng = np.random.default_rng(2)
    for scale in (2**11, 2**9, 2**14, 2**5):
        m = rng.integers(-scale, scale, 2048)
        q, sh = quant8(m)
        assert np.abs(q).max() >= 64, (
            f"|x8| peaks at {np.abs(q).max()} for a source peaking at {np.abs(m).max()}: "
            f"only {int(np.abs(q).max()).bit_length()} of 7 bits used")


def test_quant8_clamp_never_discards_magnitude():
    """bit_length - 7 is clip-free by construction: absmax >> sh < 128.

    So the clamp can only ever shave the rounding overshoot at the very peak (an RNE of
    something just under 128 landing on 128), never real magnitude. Counting how many
    elements *reach* 127 would prove nothing - with a full-scale source plenty of them do,
    and that is the range being used properly.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        m  = rng.integers(-32767, 32767, 512)
        sh = quant8_shift(int(np.abs(m).max()))
        raw = bx.rne_shift(np.asarray(m, dtype=np.int64) << (8 - sh), 8)
        assert np.abs(raw).max() <= 128, (
            f"pre-clamp peak {int(np.abs(raw).max())} at shift {sh}: the clamp is losing "
            f"magnitude, not just a rounding tie")


@pytest.mark.parametrize("mode", [0, 1])
def test_quant8_microcode_matches_firmware_algebra(mode):
    """The RTL/interpreter shift and the shift firmware derives from S6 must agree.

    They are computed differently on purpose - the program uses VCLZN(absmax) as a pre-shift
    and keeps the fixed RNE at bit 8, firmware does bit_length(absmax) - 7 in C - so a
    divergence would be silent and would corrupt exponents only for some inputs.
    """
    words = vasm.assemble(vasm.PROGRAMS["quant8"]())
    rng   = np.random.default_rng(4)
    cases = [rng.integers(-2200, 2200, 128), rng.integers(-32767, 32767, 128),
             rng.integers(-9, 9, 128), np.zeros(128, dtype=np.int64),
             np.concatenate([rng.integers(-5, 5, 127), [32767]])]
    for m in cases:
        m  = np.asarray(m, dtype=np.int64)
        sp = bytearray(8192)
        sp[0:2*len(m)] = m.astype("<i2").tobytes()
        S, _ = vasm.VecInterp(sp).run(words, [0, mode, 4096, len(m)//16])
        got  = np.frombuffer(bytes(sp[4096:4096 + len(m)]), dtype=np.int8)
        ref, sh = quant8(m, measure=bool(mode))
        assert np.array_equal(got, ref), f"mode {mode}, |m|max {int(np.abs(m).max())}"
        if mode:
            assert quant8_shift(int(S[6])) == sh, "firmware would derive a different shift"


def test_resid_add_microcode_matches():
    """resid_add's shifts are signed (VSHLV): firmware may need a *left* shift now."""
    words = vasm.assemble(vasm.PROGRAMS["resid_add"]())
    rng   = np.random.default_rng(5)
    for ea, eb in ((-5, -5), (0, 0), (2, -4), (-1, -1)):
        am = rng.integers(-30000, 30000, 128)
        bm = rng.integers(-16383, 16383, 128)
        ref, eo = resid_add(am, ea, bm, eb)
        sp = bytearray(8192)
        sp[0:256]           = am.astype("<i2").tobytes()
        sp[4096:4096 + 256] = bm.astype("<i2").tobytes()
        S, _ = vasm.VecInterp(sp).run(words, [0, 4096, 0, 8, ea - eo, eb - eo])
        got  = np.frombuffer(bytes(sp[0:256]), dtype="<i2").astype(np.int64)
        assert np.array_equal(got, ref), (ea, eb)
        assert int(S[6]) == int(np.abs(ref).max()), "S6 must return the output absmax"

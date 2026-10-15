#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""model/asm.py: the microcode assembler/interpreter, its stock programs, the hoisted programs and
the resident-BFP precision invariants."""

import numpy as np
import pytest

from model import asm, bitexact as bx, asm as vasm

# Interpreter and Stock Programs -------------------------------------------------------------------
# Microcode interpreter tests: the interpreter is the golden model for the vector unit RTL;
# these tests validate the STOCK PROGRAMS against independent numpy/float references
# (semantics + accuracy), before any RTL exists.

def _run(prog_name, sp, args):
    """Run a program; its fused form (asm.fuse) runs on a copy and must leave the scratchpad
    and the scalar results byte-identical."""
    prog  = asm.PROGRAMS[prog_name]()
    ref   = bytearray(sp)
    res   = asm.VecInterp(sp).run(asm.assemble(prog), args)
    fused = asm.fuse(prog)
    if fused != prog:
        res_f = asm.VecInterp(ref).run(asm.assemble(fused), args)
        assert bytes(ref) == bytes(sp), f"{prog_name}: fused scratchpad differs"
        assert list(res_f[0]) == list(res[0]), f"{prog_name}: fused scalars differ"
    return res

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
    # The program's contract (asm.py): k = bitlen(absmax) - 14 keeps every lane inside
    # 2^14, so the clip above is a no-op and the program carries no clamp. S6 is 0 (the
    # output-absmax REDMAX had no consumer).
    assert int(np.abs(v.astype(np.int64) >> k).max()) < 2**14
    assert S[6] == 0

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
    # A4 = eg + 6, the DIRECT (negative = right) shift for f (VSHLS convention).
    a4 = eg + 6
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
                          [0, 512, 1024, 1, eg + 6, 0, -15])
    got = np.frombuffer(bytes(sp[1024:1056]), dtype=np.int16).astype(np.float64)
    got_val = got*2.0**(eg + eu + 15)
    gv = g16.astype(np.float64)*2.0**eg
    uv = u16.astype(np.float64)*2.0**eu
    expect = gv/(1.0 + np.exp(-gv))*uv
    assert np.all(np.abs(got_val - expect) < 4*2.0**(eg+eu+15) + 3e-3*np.abs(expect))

def test_scalar_ops_m4():
    # SADD / SSHLV / SNORME semantics (48-bit).
    prog = [
        ("ARG",   "s0", 0),
        ("ARG",   "s1", 1),
        ("SADD",  "s0", "s1"),      # s0 = a0 + a1
        ("SMOV",  "s5", "s0"),
        ("ARG",   "s2", 2),
        ("SSHLV", "s0", "s2"),      # s6 = (a0+a1) shifted by a2
        ("SMOV",  "s6", "s0"),
        ("SNORME", "s7", "s0"),     # s7 = even normalize shift of s0
        ("DONE",),
    ]
    words = asm.assemble(prog)
    for a0, a1, sh in [(100, 23, 3), (1 << 30, 5, -12), (7, 0, 30), (-50, 20, 2),
                       (0x3FFFFFFF, 0, -20), (12345, 6789, 0)]:
        S, _ = asm.VecInterp(bytearray(64)).run(words, [a0, a1, sh])
        v = a0 + a1
        exp_shift = (v << sh) if sh >= 0 else (v >> -sh)
        exp_shift = ((exp_shift + (1 << 47)) & ((1 << 48) - 1)) - (1 << 47)
        assert S[5] == v
        assert S[6] == exp_shift
        if exp_shift <= 0:
            assert S[7] == 0
        else:
            e = S[7]
            assert e % 2 == 0
            nv = (exp_shift << e) if e >= 0 else (exp_shift >> -e)
            assert (1 << 30) <= nv < (1 << 32), (exp_shift, e)

def test_snorme_full_range():
    # Args are int32 (sign-extended); values above 2^31 are built in-program via SSHLV.
    prog = [("ARG", "s0", 0), ("ARG", "s1", 1), ("SSHLV", "s0", "s1"),
            ("SNORME", "s1", "s0"), ("SSHLV", "s0", "s1"),
            ("SMOV", "s6", "s0"), ("SMOV", "s7", "s1"), ("DONE",)]
    words = asm.assemble(prog)
    for p in range(0, 47):
        for base in {1 << min(p, 30), (1 << min(p, 30)) + (1 << max(0, min(p, 30) - 3)),
                     (1 << (min(p, 30) + 1)) - 1}:
            up = p - min(p, 30)
            S, _ = asm.VecInterp(bytearray(64)).run(words, [base, up])
            if (base << up) >= (1 << 47):
                continue
            assert S[7] % 2 == 0
            assert (1 << 30) <= S[6] < (1 << 32), (base, up, S[6], S[7])

E_NORM_T = -11
E_W_T    = -13

def _rmsnorm_hw(m, es, w16, n):
    beats = n//16
    sp = bytearray(64*1024)
    sp[0:2*n]         = m.astype("<i2").tobytes()
    sp[8192:8192+2*n] = w16.astype("<i2").tobytes()
    eps  = 1e-6
    eps_int = max(1, int(round(eps*2.0**(-2*es))))
    ln   = int(np.log2(n))
    args = [0, 8192, 16384, beats, -ln, eps_int, E_W_T - E_NORM_T - 15]
    _run("rmsnorm", sp, args)
    return np.frombuffer(bytes(sp[16384:16384+2*n]), dtype="<i2").astype(np.int64)

def _rmsnorm_float_ref(m, es, w16, n):
    x  = m.astype(np.float64)*2.0**es
    wf = w16.astype(np.float64)*2.0**E_W_T
    r  = 1.0/np.sqrt(np.mean(x*x) + 1e-6)
    return x*r*wf*2.0**(-E_NORM_T)

def test_rmsnorm_accuracy():
    rng = np.random.default_rng(42)
    for es, n, wl in [(-11, 256, 0.9), (-15, 2048, 1.4), (-18, 256, 0.4), (-12, 1024, 1.0)]:
        m   = np.clip(rng.normal(0, 9000, n), -32767, 32767).astype(np.int64)
        wf  = np.abs(rng.normal(wl, 0.25, n)) + 0.05
        w16 = np.clip(np.round(wf*2.0**(-E_W_T)), -32767, 32767).astype(np.int64)
        got = _rmsnorm_hw(m, es, w16, n)
        ref = _rmsnorm_float_ref(m, es, w16, n)
        err = np.abs(got - ref)
        # LUT rsqrt (256-entry interp) ~1e-4 relative + rounding.
        tol = 1.0 + 3e-4*np.abs(ref)
        assert np.all(err < tol), (es, n, err.max(), np.abs(ref).max())

def test_rmsnorm_edges():
    n = 256
    # All-zero input: eps floor keeps SNORME's input positive; output all zero.
    w16 = np.full(n, 8192, dtype=np.int64)
    got = _rmsnorm_hw(np.zeros(n, dtype=np.int64), -11, w16, n)
    assert np.all(got == 0)
    # Saturated input: scale-invariance keeps output identical for m and m/2 @ es+1.
    rng = np.random.default_rng(7)
    m   = np.clip(rng.normal(0, 16000, n), -32766, 32766).astype(np.int64)
    m   = m - (m % 2) # make halving exact
    a   = _rmsnorm_hw(m, -13, w16, n)
    b   = _rmsnorm_hw(m >> 1, -12, w16, n)
    assert np.max(np.abs(a - b)) <= 1

def test_softmax8_accuracy():
    rng = np.random.default_rng(9)
    for T, sigma in [(5, -18), (30, -20), (64, -16), (1, -18)]:
        s32 = rng.integers(-2**29, 2**29, T).astype(np.int64)
        pad = ((T + 15)//16)*16
        buf = np.full(pad, -2**31, dtype=np.int64)
        buf[:T] = s32
        sp = bytearray(8192)
        sp[0:4*pad] = buf.astype("<i4").tobytes()
        s1 = 11                      # |s>>s1| <= 2^18: subtract and *739 int32-safe.
        s2 = -(sigma + 6) - s1
        S, _ = _run("softmax8", sp, [0, 739, 4096, pad//16, -s1, -s2, 6144])
        got  = np.frombuffer(bytes(sp[4096:4096+pad]), dtype=np.int8).astype(np.float64)
        ref  = np.exp((s32 - s32.max()).astype(np.float64)*(2.0**sigma))
        # got/128 approximates ref (relative to the max entry).
        err = np.abs(got[:T]/128.0 - ref)
        assert np.all(err < 0.02), (T, err.max())
        assert got[T:].sum() == 0                       # Padding contributes nothing.
        assert S[6] == int(got[:pad].sum())             # Exact emitted sum.
        assert S[7] == int(s32.max())
        # Normalized probs vs softmax: relative error after the 1/S6 fold.
        pn  = got[:T]/S[6]
        rn  = ref/ref.sum()
        assert np.abs(pn - rn).max() < 0.02

def test_reduce_acc():
    rng = np.random.default_rng(10)
    n   = 64
    a   = rng.integers(-2**28, 2**28, n).astype(np.int64)
    b   = rng.integers(-2**28, 2**28, n).astype(np.int64)
    sp  = bytearray(4096)
    sp[0:4*n]         = a.astype("<i4").tobytes()
    sp[1024:1024+4*n] = b.astype("<i4").tobytes()
    S, _ = _run("reduce_acc", sp, [0, 1024, 0, n//8])
    got = np.frombuffer(bytes(sp[0:4*n]), dtype="<i4").astype(np.int64)
    assert np.array_equal(got, a + b)
    assert S[6] == int(np.abs(a + b).max())

def test_rope_accuracy():
    rng = np.random.default_rng(11)
    hd  = 64
    # Normed inputs (rmsnorm output <= ~2^14): rotations stay inside int16.
    x   = rng.integers(-20000, 20000, hd).astype(np.int64)
    ang = rng.uniform(0, 2*np.pi, hd//2)
    c15 = np.round(np.cos(ang)*(1 << 14)).astype(np.int64)
    s15 = np.round(np.sin(ang)*(1 << 14)).astype(np.int64)
    sp  = bytearray(4096)
    sp[0:hd]              = x[:hd//2].astype("<i2").tobytes()
    sp[256:256+hd]        = x[hd//2:].astype("<i2").tobytes()
    sp[512:512+hd]        = c15.astype("<i2").tobytes()
    sp[768:768+hd]        = s15.astype("<i2").tobytes()
    _run("rope", sp, [0, 256, 512, 768, hd//2//16, 1024, 1280])
    o0 = np.frombuffer(bytes(sp[1024:1024+hd]), dtype="<i2").astype(np.float64)
    o1 = np.frombuffer(bytes(sp[1280:1280+hd]), dtype="<i2").astype(np.float64)
    r0 = x[:hd//2]*np.cos(ang) - x[hd//2:]*np.sin(ang)
    r1 = x[:hd//2]*np.sin(ang) + x[hd//2:]*np.cos(ang)
    assert np.abs(o0 - r0).max() < 3.0   # Q15 table + rounding.
    assert np.abs(o1 - r1).max() < 3.0

def test_scale8():
    rng = np.random.default_rng(12)
    n   = 64
    m   = rng.integers(-16000, 16000, n).astype(np.int64)
    M   = 19571
    sp  = bytearray(4096)
    sp[0:2*n] = m.astype("<i2").tobytes()
    _run("scale8", sp, [0, 0, 1024, n//16, M])
    got = np.frombuffer(bytes(sp[1024:1024+n]), dtype=np.int8).astype(np.int64)
    exp = np.clip(bx.rne_shift(m*M, 15), -127, 127)
    assert np.array_equal(got, exp)


def test_nested_loops_loops2():
    # LOOPS2/LOOPE2: an outer loop over "heads" around an inner loop over
    # beats, the shape the whole-tensor per-head programs use. Inner count re-initialises
    # on every outer iteration; the counters are independent.
    import numpy as np
    sp = bytearray(4096)
    src = np.arange(1, 65, dtype="<i2")            # 4 beats of int16.
    sp[0:128] = src.tobytes()
    prog = [("ARG", "s0", 0), ("ARG", "s2", 2), ("MOVSI", "s5", 0),
            ("LOOPS2", 3),                          # 3 "heads"
            ("ARG", "s0", 0),                       # rewind the source each head
            ("LOOPS", 4),                           # 4 beats
            ("LDV", "v0", "s0"),
            ("VSHLI", "v0", "v0", 1),
            ("STV", "v0", "s2"),
            ("REDSUM", "s5", "v0"),
            ("LOOPE",),
            ("LOOPE2",),
            ("DONE",)]
    S, _ = asm.VecInterp(sp).run(asm.assemble(prog), [0, 0, 1024])
    out = np.frombuffer(bytes(sp[1024:1024 + 3*128]), dtype="<i2")
    assert np.array_equal(out, np.tile(src*2, 3))
    assert S[5] == 3*int((src*2).sum())


def _run_words(words, sp, args):
    return asm.VecInterp(sp).run(words, list(args))


def test_rmsnorm_h_matches_per_head_rmsnorm():
    # One program over H contiguous heads == H runs of prog_rmsnorm (in place), byte for byte.
    import numpy as np
    rng = np.random.default_rng(31)
    hd, H = 128, 3
    x = rng.integers(-20000, 20000, hd*H).astype("<i2")
    w = rng.integers(-8000, 8000, hd).astype("<i2")
    base = bytearray(16*1024)
    base[0:2*hd*H] = x.tobytes()
    base[0x2000:0x2000 + 2*hd] = w.tobytes()
    A4, A5, A6 = -7, 17, -17
    ref = bytearray(base)
    for h in range(H):
        _run_words(asm.assemble(asm.prog_rmsnorm()), ref,
                   [2*hd*h, 0x2000, 2*hd*h, hd//16, A4, A5, A6])
    got = bytearray(base)
    _run_words(asm.assemble(asm.prog_rmsnorm_h()), got,
               [0, 0x2000, 0, hd//16, A4, A5, A6, H])
    assert bytes(got) == bytes(ref)


def test_rope_h_matches_per_head_rope():
    import numpy as np
    rng = np.random.default_rng(32)
    hd, H = 128, 3
    x = rng.integers(-20000, 20000, hd*H).astype("<i2")
    c = rng.integers(-16384, 16384, hd//2).astype("<i2")
    s = rng.integers(-16384, 16384, hd//2).astype("<i2")
    base = bytearray(16*1024)
    base[0:2*hd*H] = x.tobytes()
    base[0x2000:0x2000 + hd] = c.tobytes()
    base[0x2000 + hd:0x2000 + 2*hd] = s.tobytes()
    ref = bytearray(base)
    for h in range(H):
        x0 = 2*hd*h
        _run_words(asm.assemble(asm.prog_rope()), ref,
                   [x0, x0 + hd, 0x2000, 0x2000 + hd, (hd//2)//16, x0, x0 + hd])
    got = bytearray(base)
    _run_words(asm.assemble(asm.prog_rope_h()), got,
               [0, hd, 0x2000, 0x2000 + hd, (hd//2)//16, 0, hd, H])
    assert bytes(got) == bytes(ref)


def test_kvpack_h_matches_per_head_fixed_quant8():
    import numpy as np
    rng = np.random.default_rng(33)
    hd, H, stride = 128, 4, 0x400
    x = rng.integers(-30000, 30000, hd*H).astype("<i2")
    base = bytearray(16*1024)
    base[0:2*hd*H] = x.tobytes()
    ref = bytearray(base)
    for h in range(H):
        _run_words(asm.assemble(asm.prog_quant8()), ref,
                   [2*hd*h, 0, 0x2020 + h*stride, hd//16])
    got = bytearray(base)
    _run_words(asm.assemble(asm.prog_kvpack_h()), got,
               [0, 0, 0x2020, hd//16, 0, 0, stride - 16*(hd//16), H])
    assert bytes(got) == bytes(ref)


def test_rmsnorm_q8_matches_rmsnorm_then_quant8():
    import numpy as np
    rng = np.random.default_rng(34)
    n = 256
    x = rng.integers(-20000, 20000, n).astype("<i2")
    w = rng.integers(-8000, 8000, n).astype("<i2")
    base = bytearray(16*1024)
    base[0:2*n] = x.tobytes()
    base[0x2000:0x2000 + 2*n] = w.tobytes()
    A4, A5, A6 = -8, 17, -17
    ref = bytearray(base)
    _run_words(asm.assemble(asm.prog_rmsnorm()), ref, [0, 0x2000, 0x1000, n//16, A4, A5, A6])
    Sr, _ = _run_words(asm.assemble(asm.prog_quant8()), ref, [0x1000, 1, 0x3000, n//16])
    got = bytearray(base)
    Sg, _ = _run_words(asm.assemble(asm.prog_rmsnorm_q8()), got,
                       [0, 0x2000, 0x1000, n//16, A4, A5, A6, 0x3000])
    assert bytes(got) == bytes(ref)
    assert Sg[6] == Sr[6]


def test_silu_q8_matches_silu_then_quant8():
    import numpy as np
    rng = np.random.default_rng(35)
    n = 256
    base = bytearray(16*1024)
    base[0:2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    base[0x1000:0x1000 + 2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    a4, a6 = -12, -15
    ref = bytearray(base)
    _run_words(asm.assemble(asm.prog_silu_mul()), ref, [0, 0x1000, 0, n//16, a4, 0, a6])
    Sr, _ = _run_words(asm.assemble(asm.prog_quant8()), ref, [0, 1, 0x3000, n//16])
    got = bytearray(base)
    Sg, _ = _run_words(asm.assemble(asm.prog_silu_q8()), got,
                       [0, 0x1000, 0, n//16, a4, 0, a6, 0x3000])
    assert bytes(got) == bytes(ref)
    assert Sg[6] == Sr[6]


def test_silu_split_matches_fused():
    """The FFN SiLU split: prog_silu_gate then prog_mul_q8 must leave the
    scratchpad and res6 byte-identical to the fused prog_silu_q8, in place (A2 == A0, as
    the FFN list runs it), for every shift the firmware can hand it and at the int16
    extremes - x1 and x2 flavours alike. The split's only new rounding boundary is the
    int16 round trip of x*sigmoid(x), which is lossless iff |mulh15(x, p)| <= |x|."""
    import numpy as np
    rng = np.random.default_rng(36)
    n = 512
    fused    = asm.assemble(asm.prog_silu_q8())
    fused_x2 = asm.assemble(asm.prog_silu_q8_x2())
    gate, gate_x2 = asm.assemble(asm.prog_silu_gate()), asm.assemble(asm.prog_silu_gate_x2())
    mul, mul_x2   = asm.assemble(asm.prog_mul_q8()), asm.assemble(asm.prog_mul_q8_x2())
    # The fused-instruction forms (asm.fuse) of all six.
    ff, ffx = (asm.assemble(asm.fuse(p())) for p in (asm.prog_silu_q8, asm.prog_silu_q8_x2))
    gf, gfx = (asm.assemble(asm.fuse(p())) for p in (asm.prog_silu_gate, asm.prog_silu_gate_x2))
    mf, mfx = (asm.assemble(asm.fuse(p())) for p in (asm.prog_mul_q8, asm.prog_mul_q8_x2))
    cases = 0
    for trial in range(6):
        base = bytearray(16*1024)
        if trial == 0:   # every int16 extreme in both operands
            g = np.array([-32768, -32767, 32767, 32766, 0, 1, -1] * (n//7 + 1))[:n].astype("<i2")
            u = np.array([32767, -32768, -32768, 32767, 32767, -1, 1] * (n//7 + 1))[:n].astype("<i2")
        else:
            g = rng.integers(-32768, 32768, n).astype("<i2")
            u = rng.integers(-32768, 32768, n).astype("<i2")
        base[0:2*n] = g.tobytes()
        base[0x1000:0x1000 + 2*n] = u.tobytes()
        for a4 in (-14, -12, -8, -4, 0, 2):        # gate shift (imm + k)
            for a6 in (-15, -13, -10):             # output shift
                for f, gp, mp in ((fused, gate, mul), (fused_x2, gate_x2, mul_x2),
                                  (ff, gf, mf), (ffx, gfx, mfx)):
                    ref = bytearray(base)
                    Sr, _ = _run_words(f, ref, [0, 0x1000, 0, n//16, a4, 0, a6, 0x3000])
                    got = bytearray(base)
                    _run_words(gp, got, [0, 0, 0, n//16, a4])
                    Sg, _ = _run_words(mp, got, [0, 0x1000, 0, n//16, 0, 0, a6, 0x3000])
                    assert bytes(got) == bytes(ref), (trial, a4, a6, len(f))
                    assert Sg[6] == Sr[6], (trial, a4, a6, Sg[6], Sr[6])
                    cases += 1
    assert cases == 6*6*3*4


def test_resid_rmsnorm_q8_matches_the_three_programs():
    """The residual tail: prog_resid_rmsnorm_q8 (x1 and x2) leaves the residual, the int16
    normed output and the int8 x8 byte-identical to prog_norm_i32 -> prog_resid_add ->
    prog_rmsnorm_q8, and its S6 packs both programs' absmaxes - over the shift range the
    firmware hands it, left shifts of either operand included, and the int16 extremes."""
    rng = np.random.default_rng(47)
    n = 512
    X, D, H, W, Y, X8 = 0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x6000
    fused = tuple(asm.assemble(f(p())) for p in (asm.prog_resid_rmsnorm_q8, asm.prog_resid_rmsnorm_q8_x2)
                  for f in (list, asm.fuse))
    cases = 0
    for trial in range(5):
        base = bytearray(32*1024)
        if trial == 0:
            x = np.array([-32767, 32767, 0, 1, -1, 16384] * (n//6 + 1))[:n].astype("<i2")
        else:
            x = rng.integers(-32767, 32768, n).astype("<i2")
        base[X:X + 2*n] = x.tobytes()
        base[W:W + 2*n] = rng.integers(-8000, 8000, n).astype("<i2").tobytes()
        for k in (0, 3, 9):
            # Engine results within norm_i32's contract for this k: |y >> k| < 2^14.
            y = rng.integers(-(1 << (13 + k)), 1 << (13 + k), n).astype("<i4")
            base[Y:Y + 4*n] = y.tobytes()
            for sa, sb in ((0, 1), (1, 0), (2, -1), (-1, 3), (0, 0)):
                for eps, wsh in ((17, -17), (1, -12)):
                    ref = bytearray(base)
                    _run_words(asm.assemble(asm.prog_norm_i32()), ref, [Y, 0, D, n//16, k])
                    Sr1, _ = _run_words(asm.assemble(asm.prog_resid_add()), ref,
                                        [X, D, X, n//16, -sa, -sb])
                    Sr2, _ = _run_words(asm.assemble(asm.prog_rmsnorm_q8()), ref,
                                        [X, W, H, n//16, -9, eps, wsh, X8])
                    a4 = asm.resid_tail_pack(-k, -sa, -sb, -9, wsh)
                    for words in fused:
                        got = bytearray(base)
                        Sg, _ = _run_words(words, got, [Y, X, H, n//16, a4, eps, W, X8])
                        # SP_D is the only region the fused program leaves untouched.
                        got[D:D + 2*n] = ref[D:D + 2*n]
                        assert bytes(got) == bytes(ref), (trial, k, sa, sb, eps)
                        assert Sg[6] == (Sr1[6] << 16) | Sr2[6], (Sg[6], Sr1[6], Sr2[6])
                        cases += 1
    assert cases == 5*3*5*2*4


def test_norm_mul_q8_matches_norm_i32_then_mul_q8():
    """The FFN tail: prog_norm_mul_q8 (x1 and x2) == prog_norm_i32 (up -> int16) then
    prog_mul_q8, in place as the FFN list runs it - scratchpad (but the int16 up it no
    longer writes) and res6."""
    rng = np.random.default_rng(49)
    n = 512
    G, U, YU, X8 = 0x0000, 0x1000, 0x2000, 0x4000
    fused = tuple(asm.assemble(f(p())) for p in (asm.prog_norm_mul_q8, asm.prog_norm_mul_q8_x2)
                  for f in (list, asm.fuse))
    cases = 0
    for trial in range(4):
        base = bytearray(24*1024)
        base[G:G + 2*n] = rng.integers(-32768, 32768, n).astype("<i2").tobytes()
        for k in (0, 2, 11):
            base[YU:YU + 4*n] = rng.integers(-(1 << (13 + k)), 1 << (13 + k), n).astype("<i4").tobytes()
            for a6 in (-15, -13, -10):
                ref = bytearray(base)
                _run_words(asm.assemble(asm.prog_norm_i32()), ref, [YU, 0, U, n//16, k])
                Sr, _ = _run_words(asm.assemble(asm.prog_mul_q8()), ref,
                                   [G, U, G, n//16, 0, 0, a6, X8])
                for words in fused:
                    got = bytearray(base)
                    Sg, _ = _run_words(words, got, [G, YU, G, n//16, k, 0, a6, X8])
                    got[U:U + 2*n] = ref[U:U + 2*n]
                    assert bytes(got) == bytes(ref), (trial, k, a6)
                    assert Sg[6] == Sr[6]
                    cases += 1
    assert cases == 4*3*3*4


def test_resid_tail_pack_fields():
    assert asm.resid_tail_pack(-32, 31, 0, -12, -17) is not None
    assert asm.resid_tail_pack(-33, 0, 0, 0, 0) is None
    assert asm.resid_tail_pack(0, 0, 0, 0, 32) is None


def _dn_scratchpad(M, k16, q16, nv, rows, beats, base=0x10000, size=0x14000):
    """The two passes' scratchpad: state at 0, k at K, k/q interleaved at KQ, -v beats at
    NV (word 0 of each, the rest zero), delta/o beats at DL."""
    K, KQ, NV, DL = base, base + 0x400, base + 0x1000, base + 0x2000
    sp = bytearray(size)
    sp[0:rows*beats*32] = np.asarray(M, dtype="<i2").tobytes()
    sp[K:K + beats*32] = np.asarray(k16, dtype="<i2").tobytes()
    kq = np.empty((beats, 2, 16), dtype="<i2")
    kq[:, 0, :] = np.asarray(k16).reshape(beats, 16)
    kq[:, 1, :] = np.asarray(q16).reshape(beats, 16)
    sp[KQ:KQ + 2*beats*32] = kq.tobytes()
    for j in range(rows):
        sp[NV + 32*j:NV + 32*j + 4] = np.asarray([nv[j]], dtype="<i4").tobytes()
    return sp, K, KQ, NV, DL

@pytest.mark.parametrize("hv,hk,seed", [(64, 64, 1), (128, 128, 2), (128, 128, 3), (32, 64, 4)])
def test_dn_passes_match_head_step(hv, hk, seed):
    """prog_dn_a / prog_dn_b executed by the interpreter equal deltanet.head_step to the
    bit: the delta beats and max |delta| out of pass A, the updated state, max |M'| and
    the o beats out of pass B - over a few tokens so the ratchet fires (the state starts
    at a small exponent and the first updates overflow it) and a token where it does not."""
    from model import deltanet as dn
    rng = np.random.default_rng(seed)
    beats = hk//16
    M = np.zeros((hv, hk), dtype=np.int64)
    eM, maxM = -20, 0
    for tok in range(4):
        kk = rng.standard_normal(hk).astype(np.float32); kk /= np.linalg.norm(kk)
        qh = rng.standard_normal(hk).astype(np.float32); qh /= np.linalg.norm(qh)
        vh = (rng.standard_normal(hv)*(3.0 if tok < 2 else 0.05)).astype(np.float32)
        decay, beta = np.float32(rng.uniform(0.3, 1.0)), np.float32(rng.uniform(0.2, 1.0))
        tr = {}
        Mref = M.copy()
        o_ref, eM2, maxM2 = dn.head_step(Mref, eM, kk, qh, vh, decay, beta, maxM=maxM, trace=tr)
        sp, K, KQ, NV, DL = _dn_scratchpad(M, tr["k16"], tr["q16"], tr["nv"], hv, beats)
        geom = hv | (beats << 16)
        shift_a, d15, nbq = tr["a_args"]
        S, _ = _run_words(asm.assemble(asm.prog_dn_a()), sp, [0, K, NV, DL, shift_a, d15, nbq, geom])
        assert S[6] == tr["maxD"], (S[6], tr["maxD"])
        got_delta = np.frombuffer(bytes(sp[DL:DL + hv*32]), dtype="<i4").reshape(hv, 8)
        assert (got_delta == tr["delta"][:, None]).all()
        shift_b, d15n, rnd = tr["b_args"]
        S, _ = _run_words(asm.assemble(asm.prog_dn_b()), sp, [0, KQ, 0, DL, shift_b, d15n, rnd, geom])
        assert S[5] == tr["maxM2"], (S[5], tr["maxM2"])
        got_M = np.frombuffer(bytes(sp[0:hv*hk*2]), dtype="<i2").reshape(hv, hk).astype(np.int64)
        assert (got_M == Mref).all(), np.argwhere(got_M != Mref)[:4]
        got_o = np.frombuffer(bytes(sp[DL:DL + hv*32]), dtype="<i4").reshape(hv, 8)
        assert (got_o == tr["o_r"][:, None]).all()
        M, eM, maxM = Mref, eM2, maxM2
        if tok == 0:
            assert tr["n"] > 0, "the first token must ratchet from the -20 start"
    assert tr["n"] == 0, "the small last update must not ratchet"


# Hoisted Microcode --------------------------------------------------------------------------------
# Optimized microcode must compute exactly what the version it replaced computed.
#
# The existing gates cannot catch a semantic change here. They compare the RTL against the
# interpreter, and both execute the *same* program words - change the program and they move
# together, still agreeing, while the model quietly computes something else. The token gates
# have the same blind spot: `hybrid_generate` runs the interpreter too.
#
# So this pins the pre-optimization programs as literals and asserts the current ones produce
# byte-identical scratchpad output on random inputs. When a program is deliberately changed in a
# way that alters results, the literal here has to be updated in the same commit - which is the
# point: it makes that a visible decision rather than an invisible drift.

def _norm_i32_before():
    return [
        ("ARG",   "s0", 0), ("ARG", "s2", 2), ("ARG", "s3", 3), ("ARG", "s4", 4),
        ("MOVSI", "s6", 0),
        ("MOVSI", "s5", 0),
        ("LOOPS", "s3"),
        ("LDW",   "v0", "s0"),
        ("LDW",   "v1", "s0"),
        ("VPACK", "v4", "v0", "v1"),
        ("BCAST", "v3", "s5"),
        ("BCAST", "v2", "s4"),
        ("VSUB",  "v2", "v3", "v2"),
        ("VSHLV", "v4", "v4", "v2"),
        ("VCLAMP16", "v4", "v4"),
        ("REDMAX", "s6", "v4"),
        ("STV",   "v4", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]


def _resid_add_before():
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s5", 5),
        ("MOVSI", "s6", 0),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("LDV",   "v1", "s1"),
        ("BCAST", "v2", "s4"), ("VSHLV", "v0", "v0", "v2"),
        ("BCAST", "v2", "s5"), ("VSHLV", "v1", "v1", "v2"),
        ("VADD",  "v0", "v0", "v1"),
        ("VCLAMP16", "v0", "v0"),
        ("REDMAX", "s6", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]


def _softmax8_before():
    """Current softmax8 with the two hoisted broadcasts put back inside the loop.

    Derived by inverting the hoist rather than pasted as a literal, deliberately: the risk
    being tested is that v3/v6 get clobbered *inside* the loop, and recomputing them per
    iteration is exactly what would mask that. (The other risk - that v6's pre-loop value was
    still needed - is not covered here; it is ruled out by v6's only consumer sitting one
    instruction after its pre-loop broadcast.)
    """
    prog = list(vasm.PROGRAMS["softmax8"]())
    hoisted = [("BCAST", "v3", "s4"), ("BCAST", "v6", "s5")]
    for h in hoisted:
        prog.remove(h)
    # Re-insert them at the top of the second loop body.
    n = 0
    for i, ins in enumerate(prog):
        if ins[0].upper() == "LOOPS":
            n += 1
            if n == 2:
                break
    return prog[:i+1] + hoisted + prog[i+1:]


def _silu_mul_before():
    # The two clamp constants (-15*2^15, 14*2^15) were rebuilt in v4 every
    # iteration (MOVSI/BCAST/VSHLI x2 = 6 of 27 body instructions) because no vreg was free.
    C_LOG2E = 739
    return [
        ("ARG",   "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
        ("ARG",   "s4", 4), ("ARG", "s6", 6),
        ("MOVSI", "s5", C_LOG2E),
        ("MOVSI", "s7", 1), ("BCAST", "v2", "s7"), ("VSHLI", "v2", "v2", 15),
        ("LOOPS", "s3"),
        ("LDV",   "v0", "s0"),
        ("LDV",   "v1", "s1"),
        ("VMULS", "v3", "v0", "s5"),
        ("VSHLS", "v3", "v3", "s4"),
        ("MOVSI", "s7", -480), ("BCAST", "v4", "s7"), ("VSHLI", "v4", "v4", 10),
        ("VMAX",  "v3", "v3", "v4"),
        ("MOVSI", "s7", 448), ("BCAST", "v4", "s7"), ("VSHLI", "v4", "v4", 10),
        ("VMIN",  "v3", "v3", "v4"),
        ("VSHRI", "v4", "v3", 15),
        ("LUTE2", "v5", "v3"),
        ("VSHLV", "v5", "v5", "v4"),
        ("VADD",  "v6", "v5", "v2"),
        ("VCLZN", "v7", "v6"),
        ("VSHLV", "v6", "v6", "v7"),
        ("LUTRECIP", "v6", "v6"),
        ("VMULH15", "v5", "v5", "v6"),
        ("VSHLV", "v5", "v5", "v7"),
        ("VSHLI", "v5", "v5", 1),
        ("VMULH15", "v0", "v0", "v5"),
        ("VMUL",  "v0", "v0", "v1"),
        ("VSHLS", "v0", "v0", "s6"),
        ("VCLAMP16", "v0", "v0"),
        ("STV",   "v0", "s2"),
        ("LOOPE",),
        ("DONE",),
    ]

BEFORE = {"norm_i32": _norm_i32_before, "resid_add": _resid_add_before,
          "softmax8": _softmax8_before, "silu_mul": _silu_mul_before}

# Argument sets and the destination window each program writes, per program.
CASES = {
    # (args, dst, dst_bytes, src_setup)
    "norm_i32":  ([0x8000, 0, 0xA000, 8, 5],            0xA000, 8*32),
    "resid_add": ([0x8000, 0x9000, 0xA000, 8, 1, 2],    0xA000, 8*32),
    "softmax8":  ([0x8000, 739, 0xA000, 4, -11, 17, 0xB800], 0xA000, 4*32),
    # A4 = -12 lands random int16 gates across the clamp window and its interior; A6 = -15.
    "silu_mul":  ([0x8000, 0x9000, 0xA000, 8, -12, 0, -15], 0xA000, 8*32),
}


def _seeded_sp(seed):
    sp = bytearray(64*1024)
    rng = np.random.default_rng(seed)
    vals = rng.integers(-2**30, 2**30, 0x4000//4).astype("<i4")
    sp[0x8000:0x8000 + len(vals)*4] = vals.tobytes()
    return sp


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_optimized_program_matches_the_one_it_replaced(name):
    args, dst, nbytes = CASES[name]
    old_words = vasm.assemble(BEFORE[name]())
    new_words = vasm.assemble(vasm.PROGRAMS[name]())

    for seed in range(6):
        sp_old, sp_new = _seeded_sp(seed), _seeded_sp(seed)
        args = list(args)
        if name == "norm_i32":
            # In-contract k only (asm.py): k = bitlen(absmax) - 14 of the data
            # the program will read. The pre-hoist program saturated out-of-contract
            # inputs; the current one carries no clamp because the contract makes it dead.
            n_src = 2*args[3]*8
            v = np.frombuffer(bytes(sp_new[0x8000:0x8000 + 4*n_src]), dtype="<i4")
            args[4] = max(0, int(np.abs(v.astype(np.int64)).max()).bit_length() - 14)
        s_old, _ = vasm.VecInterp(sp_old).run(old_words, list(args))
        s_new, _ = vasm.VecInterp(sp_new).run(new_words, list(args))
        assert bytes(sp_new[dst:dst + nbytes]) == bytes(sp_old[dst:dst + nbytes]), (
            f"{name}: scratchpad output changed (seed {seed})")
        if name != "norm_i32": # norm_i32's S6 (output absmax) was dropped: no consumer.
            assert int(s_new[6]) == int(s_old[6]), f"{name}: S6 result changed (seed {seed})"

    # And the point of the exercise: it should be shorter.
    # Deliberate unrolls trade words for cycles; list them here so accidental bloat
    # still fails. Value = the expected word count of the unrolled form.
    UNROLLED = {"norm_i32": 24}
    if name in UNROLLED:
        assert len(new_words) == UNROLLED[name], (
            f"{name}: {len(new_words)} words vs the documented unroll {UNROLLED[name]}")
        return
    assert len(new_words) <= len(old_words), (
        f"{name}: {len(new_words)} words vs {len(old_words)} before - no saving")


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_optimized_program_has_no_loop_invariant_broadcasts(name):
    """A BCAST/MOVSI inside the loop whose source never changes is recomputed per iteration."""
    prog  = vasm.PROGRAMS[name]()
    depth = 0
    body  = []
    for ins in prog:
        op = ins[0].upper()
        if op == "LOOPE":
            depth -= 1
        if depth > 0:
            body.append(ins)
        if op == "LOOPS":
            depth += 1
    written = {ins[1] for ins in body if ins[0].upper() in ("MOVSI", "ARG", "SADDI", "SMOV",
                                                           "SADD", "SSHLV", "SNORME",
                                                           "REDSUM", "REDSSQ", "REDMAX",
                                                           "REDSMAX")}
    stale = [ins for ins in body if ins[0].upper() == "BCAST" and ins[2] not in written]
    assert not stale, f"{name}: loop-invariant broadcasts still in the body: {stale}"


# BFP Precision Invariants -------------------------------------------------------------------------
# Precision invariants of the resident-BFP loop.
#
# These exist because the token gate could not see the bug they cover. That gate runs the
# tiny model - two layers - and asserts the sim matches `hybrid_generate`; but the failure
# was mantissa bits leaking away *per layer*, so it needed a dozen layers to show, and both
# sides of the comparison leaked identically. The 1.7B produced fluent garbage while every
# test passed.
#
# So: assert the arithmetic invariants directly, over a realistic layer count, and assert the
# microcode agrees with the exponent algebra firmware does in C.

E_NORM = -11


# Reference implementations ------------------------------------------------------------------------
#
# The same rules firmware applies (infer_bfp.c: op_quant8_sp, op_resid_add_sp) and the mirror
# applies (model/mirror.py hybrid_generate).

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

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
    # The program's contract (asm.py): k = bitlen(absmax) - 14 keeps every lane inside
    # 2^14, so the clip above is a no-op and the program carries no clamp. S6 is 0 since
    # 2026-08-29 (the output-absmax REDMAX had no consumer).
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
    # A4 = eg + 6, the DIRECT (negative = right) shift for f (2026-08-28 VSHLS convention).
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
    # LOOPS2/LOOPE2 (2026-08-30): an outer loop over "heads" around an inner loop over
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
    """The FFN SiLU split (2026-09-07): prog_silu_gate then prog_mul_q8 must leave the
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
                for f, gp, mp in ((fused, gate, mul), (fused_x2, gate_x2, mul_x2)):
                    ref = bytearray(base)
                    Sr, _ = _run_words(f, ref, [0, 0x1000, 0, n//16, a4, 0, a6, 0x3000])
                    got = bytearray(base)
                    _run_words(gp, got, [0, 0, 0, n//16, a4])
                    Sg, _ = _run_words(mp, got, [0, 0x1000, 0, n//16, 0, 0, a6, 0x3000])
                    assert bytes(got) == bytes(ref), (trial, a4, a6, len(f))
                    assert Sg[6] == Sr[6], (trial, a4, a6, Sg[6], Sr[6])
                    cases += 1
    assert cases == 6*6*3*2

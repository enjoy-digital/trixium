#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# BonsaiVectorUnit RTL vs the asm.py interpreter: every stock program runs on both with the
# same scratchpad contents and arguments; the resulting scratchpad bytes must be IDENTICAL
# (the interpreter is the golden model).

import numpy as np
from migen import *

from trixium.model import asm
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.vector import BonsaiVectorUnit
from trixium.gateware.hbm_model import words_from_bytes

SP_SIZE = 16*1024

class _VecDUT(Module):
    def __init__(self, sp_init):
        self.submodules.sp = BonsaiScratchpad(size=SP_SIZE, init=sp_init)
        self.submodules.vu = BonsaiVectorUnit()
        self.comb += [
            self.sp.fill.adr.eq(self.vu.sp_rd_adr),
            self.vu.sp_rd_dat.eq(self.sp.fill.dat_r),
            self.sp.gather.adr.eq(self.vu.sp_wr_adr),
            self.sp.gather.we.eq(self.vu.sp_wr_we),
            self.sp.gather.dat_w.eq(self.vu.sp_wr_dat),
        ]

def _csr_write(csr, value):
    yield csr.storage.eq(value)
    yield csr.re.eq(1)
    yield
    yield csr.re.eq(0)
    yield

def _run_rtl(prog_name, sp_bytes, args, max_cycles=200000):
    words = asm.assemble(asm.PROGRAMS[prog_name]())
    dut   = _VecDUT(words_from_bytes(bytes(sp_bytes)))
    out   = {}

    def gen():
        # Upload program.
        yield from _csr_write(dut.vu.prog_adr, 0)
        for w in words:
            yield from _csr_write(dut.vu.prog, w)
        # Args.
        arg_csrs = [dut.vu.arg0, dut.vu.arg1, dut.vu.arg2, dut.vu.arg3,
                    dut.vu.arg4, dut.vu.arg5, dut.vu.arg6, dut.vu.arg7]
        for i, a in enumerate(args):
            yield from _csr_write(arg_csrs[i], a & 0xFFFFFFFF)
        yield from _csr_write(dut.vu.pc, 0)
        # Run.
        yield from _csr_write(dut.vu.submit, 1)
        for _ in range(max_cycles):
            if not (yield dut.vu.busy):
                break
            yield
        else:
            raise AssertionError("vector unit timeout")
        yield
        # Read back the whole scratchpad.
        data = bytearray()
        for w in range(SP_SIZE//32):
            v = (yield dut.sp.mem[w])
            data += int(v).to_bytes(32, "little")
        out["sp"] = bytes(data)
        out["s6"] = (yield dut.vu.res6.status)

    run_simulation(dut, [gen()])
    return out

def _compare(prog_name, sp_bytes, args):
    ref_sp = bytearray(sp_bytes)
    S, _   = asm.VecInterp(ref_sp).run(asm.assemble(asm.PROGRAMS[prog_name]()), list(args))
    got    = _run_rtl(prog_name, sp_bytes, args)
    if got["sp"] != bytes(ref_sp):
        gb, rb = got["sp"], bytes(ref_sp)
        diffs = [i for i in range(len(gb)) if gb[i] != rb[i]][:8]
        raise AssertionError(f"{prog_name}: sp mismatch at {diffs}: "
                             f"got {[hex(gb[i]) for i in diffs]} "
                             f"ref {[hex(rb[i]) for i in diffs]}")
    return S, got

def test_rtl_resid_add():
    rng = np.random.default_rng(21)
    n   = 128
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]        = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    sp[1024:1024+2*n] = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    _compare("resid_add", sp, [0, 1024, 2048, n//16, -1, -3])

def test_rtl_quant8():
    rng = np.random.default_rng(22)
    n   = 256
    sp  = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-32767, 32768, n).astype(np.int16).tobytes()
    _compare("quant8", sp, [0, 0, 2048, n//16])

def test_rtl_norm_i32():
    rng = np.random.default_rng(23)
    n   = 64
    v   = rng.integers(-2**28, 2**28, n).astype(np.int32)
    sp  = bytearray(SP_SIZE)
    sp[0:4*n] = v.tobytes()
    k = max(0, int(np.abs(v).max()).bit_length() - 14)
    S, got = _compare("norm_i32", sp, [0, 0, 2048, n//16, k])
    assert got["s6"] == S[6] # Output absmax matches the interpreter.

def test_rtl_silu_mul():
    rng = np.random.default_rng(24)
    n   = 128
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]         = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    sp[2048:2048+2*n] = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    _compare("silu_mul", sp, [0, 2048, 4096, n//16, -6, 0, -15]) # eg = -12 -> A4 = eg+6 = -6.

def test_rtl_rmsnorm():
    rng = np.random.default_rng(31)
    n   = 128
    es  = -12
    m   = np.clip(rng.normal(0, 9000, n), -32767, 32767).astype("<i2")
    w16 = np.clip(np.round((np.abs(rng.normal(1.0, 0.3, n)) + 0.05)*(2.0**13)),
                  -32767, 32767).astype("<i2")
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]         = m.tobytes()
    sp[4096:4096+2*n] = w16.tobytes()
    eps_int = max(1, int(round(1e-6*2.0**(-2*es))))
    args = [0, 4096, 8192, n//16, -int(np.log2(n)), eps_int, (-13) - (-11) - 15]
    _compare("rmsnorm", bytes(sp), args)

def test_rtl_scalar_ops_m4():
    # SADD/SSHLV/SNORME + sign-extending ARG, RTL vs interpreter (scalar results via res5-7).
    prog = [
        ("ARG",   "s0", 0),
        ("ARG",   "s1", 1),
        ("SADD",  "s0", "s1"),
        ("SMOV",  "s5", "s0"),
        ("ARG",   "s2", 2),
        ("SSHLV", "s0", "s2"),
        ("SMOV",  "s6", "s0"),
        ("SNORME", "s7", "s0"),
        ("DONE",),
    ]
    words = asm.PROGRAMS["_m4_scalar"] = lambda: prog
    try:
        for a in ([100, 23, 3], [1 << 20, 5, 12], [7, 0, 30], [-50, 20, 2],
                  [123456, -456, -3], [-17, 0, 0]):
            ref_S, _ = asm.VecInterp(bytearray(64)).run(asm.assemble(prog), list(a))
            got = _run_rtl("_m4_scalar", bytes(SP_SIZE), a)
            m48 = (1 << 48) - 1
            assert got["s6"] == ref_S[6] & m48, (a, got["s6"], ref_S[6])
    finally:
        del asm.PROGRAMS["_m4_scalar"]

def test_rtl_softmax8():
    rng = np.random.default_rng(61)
    T   = 30
    pad = 32
    s32 = rng.integers(-2**29, 2**29, T).astype(np.int64)
    buf = np.full(pad, -2**31, dtype=np.int64)
    buf[:T] = s32
    sp  = bytearray(SP_SIZE)
    sp[0:4*pad] = buf.astype("<i4").tobytes()
    sigma, s1 = -18, 11
    _compare("softmax8", bytes(sp), [0, 739, 8192, pad//16, -s1, -(-(sigma + 6) - s1),
                                     12288])

def test_rtl_reduce_acc():
    rng = np.random.default_rng(62)
    n   = 64
    sp  = bytearray(SP_SIZE)
    sp[0:4*n]         = rng.integers(-2**28, 2**28, n).astype("<i4").tobytes()
    sp[1024:1024+4*n] = rng.integers(-2**28, 2**28, n).astype("<i4").tobytes()
    _compare("reduce_acc", bytes(sp), [0, 1024, 0, n//8])

def test_rtl_rope():
    rng = np.random.default_rng(63)
    hd  = 64
    sp  = bytearray(SP_SIZE)
    sp[0:hd]       = rng.integers(-30000, 30000, hd//2).astype("<i2").tobytes()
    sp[256:256+hd] = rng.integers(-30000, 30000, hd//2).astype("<i2").tobytes()
    sp[512:512+hd] = rng.integers(-32767, 32768, hd//2).astype("<i2").tobytes()
    sp[768:768+hd] = rng.integers(-32767, 32768, hd//2).astype("<i2").tobytes()
    _compare("rope", bytes(sp), [0, 256, 512, 768, hd//2//16, 1024, 1280])

def test_rtl_scale8():
    rng = np.random.default_rng(64)
    n   = 64
    sp  = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-16000, 16000, n).astype("<i2").tobytes()
    _compare("scale8", bytes(sp), [0, 1024, 2048, n//16, 19571])


def _compare_words(prog, sp_bytes, args):
    # Like _compare but for programs not (yet) in the per-op RTL list: whole scratchpad.
    ref = bytearray(sp_bytes)
    asm.VecInterp(ref).run(asm.assemble(prog), list(args))
    words = asm.assemble(prog)
    dut   = _VecDUT(words_from_bytes(bytes(sp_bytes)))
    out   = {}

    def gen():
        yield from _csr_write(dut.vu.prog_adr, 0)
        for w in words:
            yield from _csr_write(dut.vu.prog, w)
        arg_csrs = [dut.vu.arg0, dut.vu.arg1, dut.vu.arg2, dut.vu.arg3,
                    dut.vu.arg4, dut.vu.arg5, dut.vu.arg6, dut.vu.arg7]
        for i, a in enumerate(args):
            yield from _csr_write(arg_csrs[i], a & 0xFFFFFFFF)
        yield from _csr_write(dut.vu.pc, 0)
        yield from _csr_write(dut.vu.submit, 1)
        for _ in range(200000):
            if not (yield dut.vu.busy):
                break
            yield
        else:
            raise AssertionError("vector unit timeout")
        yield
        data = bytearray()
        for w in range(SP_SIZE//32):
            v = (yield dut.sp.mem[w])
            data += int(v).to_bytes(32, "little")
        out["sp"] = bytes(data)

    run_simulation(dut, [gen()])
    assert out["sp"] == bytes(ref), "RTL differs from the interpreter"


def test_rtl_rmsnorm_h():
    rng = np.random.default_rng(41)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*hd] = rng.integers(-8000, 8000, hd).astype("<i2").tobytes()
    _compare_words(asm.prog_rmsnorm_h(), sp, [0, 0x2000, 0, hd//16, -7, 17, -17, H])


def test_rtl_rope_h():
    rng = np.random.default_rng(42)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + hd] = rng.integers(-16384, 16384, hd//2).astype("<i2").tobytes()
    sp[0x2000 + hd:0x2000 + 2*hd] = rng.integers(-16384, 16384, hd//2).astype("<i2").tobytes()
    _compare_words(asm.prog_rope_h(), sp, [0, hd, 0x2000, 0x2000 + hd, (hd//2)//16, 0, hd, H])


def test_rtl_kvpack_h():
    rng = np.random.default_rng(43)
    hd, H, stride = 128, 4, 0x400
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-30000, 30000, hd*H).astype("<i2").tobytes()
    _compare_words(asm.prog_kvpack_h(), sp,
                   [0, 0, 0x2020, hd//16, 0, 0, stride - 16*(hd//16), H])


def test_rtl_rmsnorm_q8():
    rng = np.random.default_rng(44)
    n = 256
    sp = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*n] = rng.integers(-8000, 8000, n).astype("<i2").tobytes()
    _compare_words(asm.prog_rmsnorm_q8(), sp, [0, 0x2000, 0x1000, n//16, -8, 17, -17, 0x3000])


def test_rtl_silu_q8():
    rng = np.random.default_rng(45)
    n = 256
    sp = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    sp[0x1000:0x1000 + 2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    _compare_words(asm.prog_silu_q8(), sp, [0, 0x1000, 0, n//16, -12, 0, -15, 0x3000])


def test_rtl_silu_gate_and_mul_q8():
    # The FFN SiLU split (2026-09-07): both halves, x1 and x2, RTL vs interpreter over the
    # whole scratchpad, run in place the way the FFN list runs them (A2 == A0).
    rng = np.random.default_rng(46)
    n = 256
    for gate, mul in ((asm.prog_silu_gate(), asm.prog_mul_q8()),
                      (asm.prog_silu_gate_x2(), asm.prog_mul_q8_x2())):
        sp = bytearray(SP_SIZE)
        sp[0:2*n] = rng.integers(-32768, 32768, n).astype("<i2").tobytes()
        sp[0x1000:0x1000 + 2*n] = rng.integers(-32768, 32768, n).astype("<i2").tobytes()
        _compare_words(gate, sp, [0, 0, 0, n//16, -12])
        # Feed the mul half the gate half's OUTPUT (interpreter-produced, already proven
        # equal to the RTL's by the line above).
        asm.VecInterp(sp).run(asm.assemble(gate), [0, 0, 0, n//16, -12])
        _compare_words(mul, sp, [0, 0x1000, 0, n//16, 0, 0, -15, 0x3000])


def _dn_case(seed, hv, hk):
    """A DeltaNet head mid-life: a state at its exponent, unit k/q, a value row - the
    scratchpad and both passes' arguments from deltanet.head_step's trace."""
    from trixium.model import deltanet as dn
    from test_asm import _dn_scratchpad
    rng = np.random.default_rng(seed)
    M = rng.integers(-30000, 30000, (hv, hk)).astype(np.int64)
    kk = rng.standard_normal(hk).astype(np.float32); kk /= np.linalg.norm(kk)
    qh = rng.standard_normal(hk).astype(np.float32); qh /= np.linalg.norm(qh)
    vh = (rng.standard_normal(hv)*2.0).astype(np.float32)
    tr = {}
    dn.head_step(M.copy(), -14, kk, qh, vh, np.float32(0.9), np.float32(0.7), trace=tr)
    sp, K, KQ, NV, DL = _dn_scratchpad(M, tr["k16"], tr["q16"], tr["nv"], hv, hk//16,
                                       base=0x1000, size=SP_SIZE)
    return sp, K, KQ, NV, DL, tr, hv | ((hk//16) << 16)


def test_rtl_dn_a():
    sp, K, KQ, NV, DL, tr, geom = _dn_case(45, 32, 64)
    shift_a, d15, nbq = tr["a_args"]
    _compare_words(asm.prog_dn_a(), sp, [0, K, NV, DL, shift_a, d15, nbq, geom])


def test_rtl_dn_b():
    sp, K, KQ, NV, DL, tr, geom = _dn_case(46, 32, 64)
    # Pass A's delta beats as pass B finds them.
    for j in range(32):
        sp[DL + 32*j:DL + 32*j + 32] = np.full(8, tr["delta"][j], dtype="<i4").tobytes()
    shift_b, d15n, rnd = tr["b_args"]
    _compare_words(asm.prog_dn_b(), sp, [0, KQ, 0, DL, shift_b, d15n, rnd, geom])

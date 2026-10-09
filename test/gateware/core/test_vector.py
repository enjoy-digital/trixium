#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/core/vector.py: the VectorUnit RTL against the asm.py interpreter (stock programs, random
programs for the pipeline hazards, the rmsnorm shift range)."""

import random

import numpy as np
import pytest

from migen import *

from gateware.core.scratchpad import Scratchpad
from gateware.core.vector import VectorUnit
from gateware.memory.hbm_model import words_from_bytes
from model import asm

from common import import_test
from common.verilator import Script, Simulator, run_migen

# Stock Programs -----------------------------------------------------------------------------------
# VectorUnit RTL vs the asm.py interpreter: every stock program runs on both with the
# same scratchpad contents and arguments; the resulting scratchpad bytes must be IDENTICAL
# (the interpreter is the golden model).

SP_SIZE = 16*1024

class _DUT(Module):
    def __init__(self, sp_init=None):
        self.submodules.sp = Scratchpad(size=SP_SIZE, init=sp_init)
        self.submodules.vu = VectorUnit()
        self.comb += [
            self.sp.fill.adr.eq(self.vu.sp_rd_adr),
            self.vu.sp_rd_dat.eq(self.sp.fill.dat_r),
            self.sp.gather.adr.eq(self.vu.sp_wr_adr),
            # The integrator's rule: a withheld grant means the write does not happen (the
            # grant is held at 1 unless a test withholds it).
            self.sp.gather.we.eq(Mux(self.vu.sp_wr_gnt, self.vu.sp_wr_we, 0)),
            self.sp.gather.dat_w.eq(self.vu.sp_wr_dat),
        ]
        vu = self.vu
        self.signals = dict(vu_busy=vu.sp_wr_gnt, vu_sp_wr_gnt=vu.sp_wr_gnt)
        for name in ("prog", "prog_adr", "pc", "submit") + tuple(f"arg{i}" for i in range(8)):
            self.signals[f"vu_{name}"]    = getattr(vu, name).storage
            self.signals[f"vu_{name}_re"] = getattr(vu, name).re
        self.signals.update(vu_busy=vu.busy, vu_res5=vu.res5.status, vu_res6=vu.res6.status,
                            vu_res7=vu.res7.status, vu_bad_op=vu.status.fields.bad_op,
                            vu_done_count=vu.done_count.status)

# Backends -----------------------------------------------------------------------------------------
# Every RTL test runs on the compiled vector unit: one Script, the same scratchpad image in, the
# whole scratchpad and S5..S7 out. The same Script runs on Migen's simulator (minutes per test)
# for one random program in the slow tier: the cross-check of the emitted Verilog.

COMPILED = pytest.mark.parametrize("backend", ["verilator"])

def _csr_write(s, name, value):
    s.set(name, value)
    s.set(name + "_re", 1)
    s.step()
    s.set(name + "_re", 0)
    s.step()

def _simulator():
    def build():
        dut = _DUT()
        return dict(dut=dut, ios=dut.signals, memories={"sp": dut.sp.mem})
    return Simulator(build, name="vecunit", key="test_vector._DUT")

def _vec_run(backend, words, sp_bytes, args, withhold_seed=None, max_cycles=200000):
    """Upload `words`, run them on `args` over the scratchpad image; withhold_seed: withhold
    the store grant ~40% of the cycles while the program runs."""
    s = Script()
    s.reset(2)
    s.set("vu_sp_wr_gnt", 1)
    _csr_write(s, "vu_prog_adr", 0)
    for w in words:
        _csr_write(s, "vu_prog", w)
    for i, a in enumerate(args):
        _csr_write(s, f"vu_arg{i}", a & 0xFFFFFFFF)
    _csr_write(s, "vu_pc", 0)
    if withhold_seed is not None:
        s.random("vu_sp_wr_gnt", 60, withhold_seed)
    _csr_write(s, "vu_submit", 1)
    s.wait({"vu_busy": 0}, cap=max_cycles, tag="vector_unit_timeout")
    s.random("vu_sp_wr_gnt", -1)
    s.set("vu_sp_wr_gnt", 1)
    s.step()
    s.read("sp", 0, SP_SIZE, "sp")
    for name in ("res5", "res6", "res7", "bad_op", "done_count"):
        s.peek(f"vu_{name}", name)
    if backend == "migen":
        dut = _DUT(words_from_bytes(bytes(sp_bytes)))
        r = run_migen(dut, dut.signals, s, memories={"sp": dut.sp.mem})
    else:
        r = _simulator().run(s, images={"sp": bytes(sp_bytes)})
    p = r.peeks
    return dict(sp=r.reads["sp"], s5=p["res5"], s6=p["res6"], s7=p["res7"], bad=p["bad_op"],
                dc=p["done_count"])

def _run_rtl(backend, prog_name, sp_bytes, args, max_cycles=200000):
    return _vec_run(backend, asm.assemble(asm.PROGRAMS[prog_name]()), sp_bytes, args,
                    max_cycles=max_cycles)

def _compare(backend, prog_name, sp_bytes, args):
    ref_sp = bytearray(sp_bytes)
    S, _   = asm.VecInterp(ref_sp).run(asm.assemble(asm.PROGRAMS[prog_name]()), list(args))
    got    = _run_rtl(backend, prog_name, sp_bytes, args)
    if got["sp"] != bytes(ref_sp):
        gb, rb = got["sp"], bytes(ref_sp)
        diffs = [i for i in range(len(gb)) if gb[i] != rb[i]][:8]
        raise AssertionError(f"{prog_name}: sp mismatch at {diffs}: "
                             f"got {[hex(gb[i]) for i in diffs]} "
                             f"ref {[hex(rb[i]) for i in diffs]}")
    return S, got

@COMPILED
def test_rtl_resid_add(backend):
    rng = np.random.default_rng(21)
    n   = 128
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]        = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    sp[1024:1024+2*n] = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    _compare(backend, "resid_add", sp, [0, 1024, 2048, n//16, -1, -3])

@COMPILED
def test_rtl_quant8(backend):
    rng = np.random.default_rng(22)
    n   = 256
    sp  = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-32767, 32768, n).astype(np.int16).tobytes()
    _compare(backend, "quant8", sp, [0, 0, 2048, n//16])

@COMPILED
def test_rtl_norm_i32(backend):
    rng = np.random.default_rng(23)
    n   = 64
    v   = rng.integers(-2**28, 2**28, n).astype(np.int32)
    sp  = bytearray(SP_SIZE)
    sp[0:4*n] = v.tobytes()
    k = max(0, int(np.abs(v).max()).bit_length() - 14)
    S, got = _compare(backend, "norm_i32", sp, [0, 0, 2048, n//16, k])
    assert got["s6"] == S[6] # Output absmax matches the interpreter.

@COMPILED
def test_rtl_silu_mul(backend):
    rng = np.random.default_rng(24)
    n   = 128
    sp  = bytearray(SP_SIZE)
    sp[0:2*n]         = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    sp[2048:2048+2*n] = rng.integers(-30000, 30000, n).astype(np.int16).tobytes()
    _compare(backend, "silu_mul", sp, [0, 2048, 4096, n//16, -6, 0, -15]) # eg = -12 -> A4 = eg+6 = -6.

@COMPILED
def test_rtl_rmsnorm(backend):
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
    _compare(backend, "rmsnorm", bytes(sp), args)

@COMPILED
def test_rtl_scalar_ops_m4(backend):
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
    asm.PROGRAMS["_m4_scalar"] = lambda: prog
    try:
        for a in ([100, 23, 3], [1 << 20, 5, 12], [7, 0, 30], [-50, 20, 2],
                  [123456, -456, -3], [-17, 0, 0]):
            ref_S, _ = asm.VecInterp(bytearray(64)).run(asm.assemble(prog), list(a))
            got = _run_rtl(backend, "_m4_scalar", bytes(SP_SIZE), a)
            m48 = (1 << 48) - 1
            assert got["s6"] == ref_S[6] & m48, (a, got["s6"], ref_S[6])
    finally:
        del asm.PROGRAMS["_m4_scalar"]

@COMPILED
def test_rtl_softmax8(backend):
    rng = np.random.default_rng(61)
    T   = 30
    pad = 32
    s32 = rng.integers(-2**29, 2**29, T).astype(np.int64)
    buf = np.full(pad, -2**31, dtype=np.int64)
    buf[:T] = s32
    sp  = bytearray(SP_SIZE)
    sp[0:4*pad] = buf.astype("<i4").tobytes()
    sigma, s1 = -18, 11
    _compare(backend, "softmax8", bytes(sp), [0, 739, 8192, pad//16, -s1, -(-(sigma + 6) - s1),
                                     12288])

@COMPILED
def test_rtl_reduce_acc(backend):
    rng = np.random.default_rng(62)
    n   = 64
    sp  = bytearray(SP_SIZE)
    sp[0:4*n]         = rng.integers(-2**28, 2**28, n).astype("<i4").tobytes()
    sp[1024:1024+4*n] = rng.integers(-2**28, 2**28, n).astype("<i4").tobytes()
    _compare(backend, "reduce_acc", bytes(sp), [0, 1024, 0, n//8])

@COMPILED
def test_rtl_rope(backend):
    rng = np.random.default_rng(63)
    hd  = 64
    sp  = bytearray(SP_SIZE)
    sp[0:hd]       = rng.integers(-30000, 30000, hd//2).astype("<i2").tobytes()
    sp[256:256+hd] = rng.integers(-30000, 30000, hd//2).astype("<i2").tobytes()
    sp[512:512+hd] = rng.integers(-32767, 32768, hd//2).astype("<i2").tobytes()
    sp[768:768+hd] = rng.integers(-32767, 32768, hd//2).astype("<i2").tobytes()
    _compare(backend, "rope", bytes(sp), [0, 256, 512, 768, hd//2//16, 1024, 1280])

@COMPILED
def test_rtl_scale8(backend):
    rng = np.random.default_rng(64)
    n   = 64
    sp  = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-16000, 16000, n).astype("<i2").tobytes()
    _compare(backend, "scale8", bytes(sp), [0, 1024, 2048, n//16, 19571])


def _compare_words(backend, prog, sp_bytes, args):
    # Like _compare but for programs not (yet) in the per-op RTL list: whole scratchpad.
    ref = bytearray(sp_bytes)
    asm.VecInterp(ref).run(asm.assemble(prog), list(args))
    out = _vec_run(backend, asm.assemble(prog), sp_bytes, args)
    assert out["sp"] == bytes(ref), "RTL differs from the interpreter"


@COMPILED
def test_rtl_rmsnorm_h(backend):
    rng = np.random.default_rng(41)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*hd] = rng.integers(-8000, 8000, hd).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_rmsnorm_h(), sp, [0, 0x2000, 0, hd//16, -7, 17, -17, H])


@COMPILED
def test_rtl_rope_h(backend):
    rng = np.random.default_rng(42)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + hd] = rng.integers(-16384, 16384, hd//2).astype("<i2").tobytes()
    sp[0x2000 + hd:0x2000 + 2*hd] = rng.integers(-16384, 16384, hd//2).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_rope_h(), sp, [0, hd, 0x2000, 0x2000 + hd, (hd//2)//16, 0, hd, H])


@COMPILED
def test_rtl_kvpack_h(backend):
    rng = np.random.default_rng(43)
    hd, H, stride = 128, 4, 0x400
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-30000, 30000, hd*H).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_kvpack_h(), sp,
                   [0, 0, 0x2020, hd//16, 0, 0, stride - 16*(hd//16), H])


@COMPILED
def test_rtl_rmsnorm_q8(backend):
    rng = np.random.default_rng(44)
    n = 256
    sp = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*n] = rng.integers(-8000, 8000, n).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_rmsnorm_q8(), sp, [0, 0x2000, 0x1000, n//16, -8, 17, -17, 0x3000])


@COMPILED
def test_rtl_silu_q8(backend):
    rng = np.random.default_rng(45)
    n = 256
    sp = bytearray(SP_SIZE)
    sp[0:2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    sp[0x1000:0x1000 + 2*n] = rng.integers(-20000, 20000, n).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_silu_q8(), sp, [0, 0x1000, 0, n//16, -12, 0, -15, 0x3000])


@COMPILED
def test_rtl_silu_gate_and_mul_q8(backend):
    # The FFN SiLU split: both halves, x1 and x2, RTL vs interpreter over the
    # whole scratchpad, run in place the way the FFN list runs them (A2 == A0).
    rng = np.random.default_rng(46)
    n = 256
    for gate, mul in ((asm.prog_silu_gate(), asm.prog_mul_q8()),
                      (asm.prog_silu_gate_x2(), asm.prog_mul_q8_x2())):
        sp = bytearray(SP_SIZE)
        sp[0:2*n] = rng.integers(-32768, 32768, n).astype("<i2").tobytes()
        sp[0x1000:0x1000 + 2*n] = rng.integers(-32768, 32768, n).astype("<i2").tobytes()
        _compare_words(backend, gate, sp, [0, 0, 0, n//16, -12])
        # Feed the mul half the gate half's OUTPUT (interpreter-produced, already proven
        # equal to the RTL's by the line above).
        asm.VecInterp(sp).run(asm.assemble(gate), [0, 0, 0, n//16, -12])
        _compare_words(backend, mul, sp, [0, 0x1000, 0, n//16, 0, 0, -15, 0x3000])


def _dn_case(seed, hv, hk):
    """A DeltaNet head mid-life: a state at its exponent, unit k/q, a value row - the
    scratchpad and both passes' arguments from deltanet.head_step's trace."""
    from model import deltanet as dn
    _dn_scratchpad = import_test("model/test_asm.py")._dn_scratchpad
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


@COMPILED
def test_rtl_dn_a(backend):
    sp, K, KQ, NV, DL, tr, geom = _dn_case(45, 32, 64)
    shift_a, d15, nbq = tr["a_args"]
    _compare_words(backend, asm.prog_dn_a(), sp, [0, K, NV, DL, shift_a, d15, nbq, geom])


@COMPILED
def test_rtl_dn_b(backend):
    sp, K, KQ, NV, DL, tr, geom = _dn_case(46, 32, 64)
    # Pass A's delta beats as pass B finds them.
    for j in range(32):
        sp[DL + 32*j:DL + 32*j + 32] = np.full(8, tr["delta"][j], dtype="<i4").tobytes()
    shift_b, d15n, rnd = tr["b_args"]
    _compare_words(backend, asm.prog_dn_b(), sp, [0, KQ, 0, DL, shift_b, d15n, rnd, geom])


# Random-Program Hazards ---------------------------------------------------------------------------
# Random-program hazard stress for the pipelined vector unit.
#
# The stock programs (above) exercise each op, but a 1-instruction/cycle pipeline
# is only correct if EVERY dependency distance forwards or stalls right: RAW on V at
# distance 1/2/3 (ALU, multiply, load, LUT producers), reductions followed by scalar
# readers/writers of the same S, `producer; STx`, loads one behind a store to the same
# beat, 2-8 instruction loop bodies, DONE right after a reduction or a store, and stores
# held by a withheld sp_wr_gnt. Programs are generated in the interpreter's ISA with those
# shapes biased in, and the RTL must match the interpreter on the whole scratchpad plus
# res5/6/7 - byte for byte, program after program.

SRC0, SRC1, DST = 0x0000, 0x0800, 0x1000


# The constant register is the top vreg; random operands span the rest (ISA rev 2: 16).
VCONST = f"v{asm.VREGS - 1}"
VMAX   = asm.VREGS - 2


def gen_program(rng):
    """One random program. v7 holds a positive constant (LUTRECIP operands must be
    nonzero); s0/s1 are load pointers, s2 the store pointer, s3 the loop count, s4 a
    constant, s5..s7 the reduction/scalar scratch."""
    prog = [("ARG", "s0", 0), ("ARG", "s1", 1), ("ARG", "s2", 2), ("ARG", "s3", 3),
            ("MOVSI", "s4", rng.randint(1, 1000)),
            ("MOVSI", "s5", rng.randint(-1000, 1000)), ("MOVSI", "s6", 0),
            ("MOVSI", "s7", rng.randint(-5, 5)),
            ("BCAST", VCONST, "s4")]
    last = ["v0", "v1"]   # Recently written vregs, most recent first (dependency bias).

    def vsrc():
        r = rng.random()
        if r < 0.55 and last:
            return last[0]
        if r < 0.75 and len(last) > 1:
            return last[1]
        return f"v{rng.randint(0, VMAX)}"

    def vdst():
        return f"v{rng.randint(0, VMAX)}"

    def wrote(v):
        if v in last:
            last.remove(v)
        last.insert(0, v)
        del last[3:]

    sregs = ["s5", "s6", "s7"]

    def emit_random(n, allow_mem=True):
        out = []
        for _ in range(n):
            k = rng.random()
            if k < 0.34:
                op = rng.choice(["VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX",
                                 "VSHLV", "VPACK"])
                d = vdst(); out.append((op, d, vsrc(), vsrc())); wrote(d)
            elif k < 0.50:
                op = rng.choice(["VSHRI", "VSHLI", "VRNE"])
                d = vdst()
                imm = rng.randint(0, 31) if op != "VSHLI" else rng.randint(0, 15)
                out.append((op, d, vsrc(), imm)); wrote(d)
            elif k < 0.60:
                op = rng.choice(["VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "LUTE2"])
                d = vdst(); out.append((op, d, vsrc())); wrote(d)
            elif k < 0.66:
                op = rng.choice(["BCAST", "VSHLS", "VMULS"])
                d = vdst()
                if op == "BCAST":
                    out.append((op, d, rng.choice(sregs + ["s4"])))
                else:
                    out.append((op, d, vsrc(), rng.choice(sregs + ["s4"])))
                wrote(d)
            elif k < 0.74 and allow_mem:
                op = rng.choice(["LDV", "LDW"])
                d = vdst(); out.append((op, d, rng.choice(["s0", "s1"]))); wrote(d)
            elif k < 0.82 and allow_mem:
                op = rng.choice(["STV", "STB", "STW"])
                out.append((op, vsrc(), "s2"))
                if op == "STB":   # Pairs keep s2 beat-aligned for the STV/STW that follow
                    out.append(("STB", vsrc(), "s2"))   # (the ports are beat-addressed).
            elif k < 0.90:
                op = rng.choice(["REDSUM", "REDSSQ", "REDMAX", "REDSMAX"])
                out.append((op, rng.choice(sregs), vsrc()))
                # A scalar consumer of the same S right behind it, sometimes.
                if rng.random() < 0.5:
                    sop = rng.choice(["SADDI", "SMOV", "SADD", "SSHLV", "SNORME"])
                    sd, sa = rng.choice(sregs), rng.choice(sregs)
                    if sop == "SADDI":
                        out.append((sop, sd, rng.randint(-100, 100)))
                    else:
                        out.append((sop, sd, sa))
            elif k < 0.95:
                # LUTRECIP idiom on a normalized nonzero mantissa.
                x, z, w = vdst(), vdst(), vdst()
                out += [("VABS", x, vsrc()), ("VMAX", x, x, VCONST), ("VCLZN", z, x),
                        ("VSHLV", x, x, z), ("LUTRECIP", w, x)]
                wrote(x); wrote(z); wrote(w)
            else:
                sop = rng.choice(["SADDI", "SMOV", "SADD", "SSHLV", "SNORME"])
                sd, sa = rng.choice(sregs), rng.choice(sregs)
                out.append((sop, sd, rng.randint(-100, 100)) if sop == "SADDI"
                           else (sop, sd, sa))
        return out

    prog += emit_random(rng.randint(3, 10))
    # Store then load the SAME beat one instruction later (the memory RAW hazard).
    if rng.random() < 0.4:
        prog += [("SMOV", "s1", "s2"), ("STV", vsrc(), "s2"), ("LDV", "v3", "s1")]
        wrote("v3")
    # Loop, sometimes nested in an outer LOOPS2..LOOPE2 (the per-head shape).
    nested = rng.random() < 0.5
    if nested:
        prog.append(("LOOPS2", rng.randint(2, 3)))
        prog += emit_random(rng.randint(0, 3), allow_mem=False)
    if rng.random() < 0.5:
        prog.append(("LOOPS", "s3"))
    else:
        prog.append(("LOOPS", rng.randint(2, 4)))   # Immediate-count form.
    prog += emit_random(rng.randint(2, 8))
    prog.append(("LOOPE",))
    if nested:
        prog += emit_random(rng.randint(0, 3), allow_mem=False)
        prog.append(("LOOPE2",))
    prog += emit_random(rng.randint(1, 6))
    # Sometimes end right after a reduction or a store.
    tail = rng.random()
    if tail < 0.3:
        prog.append((rng.choice(["REDSUM", "REDMAX"]), rng.choice(sregs), vsrc()))
    elif tail < 0.6:
        prog.append(("STV", vsrc(), "s2"))
    prog.append(("DONE",))
    return prog


def _seeded_sp(rng):
    sp = bytearray(SP_SIZE)
    vals = np.array([rng.randint(-32768, 32767) for _ in range(0x1000//2)], dtype="<i2")
    sp[0:0x1000] = vals.tobytes()
    return sp


def _check(backend, seed, withhold):
    rng  = random.Random(seed)
    prog = gen_program(rng)
    words = asm.assemble(prog)
    sp   = _seeded_sp(rng)
    args = [SRC0, SRC1, DST, rng.randint(2, 4)]
    ref  = bytearray(sp)
    S, _ = asm.VecInterp(ref).run(words, list(args))
    got  = _vec_run(backend, words, sp, args, withhold_seed=seed + 7 if withhold else None,
                    max_cycles=100000)
    m48  = (1 << 48) - 1
    diffs = [i for i in range(SP_SIZE) if got["sp"][i] != ref[i]]
    assert not diffs, (seed, prog, f"first diffs at {diffs[:8]}")
    assert got["s5"] == (S[5] & m48), (seed, prog, "s5", got["s5"], S[5] & m48)
    assert got["s6"] == (S[6] & m48), (seed, prog, "s6", got["s6"], S[6] & m48)
    assert got["s7"] == (S[7] & m48), (seed, prog, "s7", got["s7"], S[7] & m48)
    assert got["bad"] == 0 and got["dc"] == 1, (seed, prog, got["bad"], got["dc"])


@pytest.mark.parametrize("seed,backend", [(seed, "verilator") for seed in range(24)] +
                         [pytest.param(0, "migen", marks=pytest.mark.slow)])
def test_random_program_matches_interpreter(seed, backend):
    _check(backend, seed, withhold=False)


@COMPILED
@pytest.mark.parametrize("seed", list(range(100, 112)))
def test_random_program_under_withheld_store_grants(seed, backend):
    _check(backend, seed, withhold=True)


# RMSNorm Shift Range ------------------------------------------------------------------------------
# The rmsnorm programs' A6 (gamma-exponent shift) over the range the per-tensor NORM16
# exponents produce (layout.norm16_exp: the 1.7B spans -15..-7, i.e. A6 from -19 to -11),
# RTL against the interpreter. The suites only ever ran A6 = -17 (the old fixed -13).

E_NORM = -11


@COMPILED
@pytest.mark.parametrize("e16", [-15, -14, -13, -10, -8, -7])
def test_rtl_rmsnorm_shift_range(e16, backend):
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
    _compare(backend, "rmsnorm", bytes(sp), args)

@COMPILED
@pytest.mark.parametrize("e16", [-15, -13, -8])
def test_rtl_rmsnorm_h_shift_range(e16, backend):
    rng = np.random.default_rng(200 + e16)
    hd, H = 128, 3
    sp = bytearray(SP_SIZE)
    sp[0:2*hd*H] = rng.integers(-20000, 20000, hd*H).astype("<i2").tobytes()
    sp[0x2000:0x2000 + 2*hd] = rng.integers(-32000, 32000, hd).astype("<i2").tobytes()
    _compare_words(backend, asm.prog_rmsnorm_h(), sp, [0, 0x2000, 0, hd//16, -7, 17, e16 - E_NORM - 15, H])

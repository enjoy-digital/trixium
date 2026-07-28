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
    assert got["sp"] == bytes(ref_sp), f"{prog_name}: scratchpad mismatch"
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
    _compare("silu_mul", sp, [0, 2048, 4096, n//16, 6, 0, -15]) # eg = -12 -> A4 = 6.

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Random-program hazard stress for the pipelined vector unit.

The stock programs (test_vector.py) exercise each op, but a 1-instruction/cycle pipeline
is only correct if EVERY dependency distance forwards or stalls right: RAW on V at
distance 1/2/3 (ALU, multiply, load, LUT producers), reductions followed by scalar
readers/writers of the same S, `producer; STx`, loads one behind a store to the same
beat, 2-8 instruction loop bodies, DONE right after a reduction or a store, and stores
held by a withheld sp_wr_gnt. Programs are generated in the interpreter's ISA with those
shapes biased in, and the RTL must match the interpreter on the whole scratchpad plus
res5/6/7 - byte for byte, program after program."""

import random

import numpy as np
import pytest

from migen import *

from model import asm
from gateware.core.scratchpad import Scratchpad
from gateware.core.vector import VectorUnit
from gateware.memory.hbm_model import words_from_bytes

SP_SIZE = 16*1024
SRC0, SRC1, DST = 0x0000, 0x0800, 0x1000


# The constant register is the top vreg; random operands span the rest (ISA rev 2: 16).
VCONST = f"v{asm.VREGS - 1}"
VMAX   = asm.VREGS - 2

# Helpers ------------------------------------------------------------------------------------------

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


class _DUT(Module):
    def __init__(self, sp_init):
        self.submodules.sp = Scratchpad(size=SP_SIZE, init=sp_init)
        self.submodules.vu = VectorUnit()
        self.comb += [
            self.sp.fill.adr.eq(self.vu.sp_rd_adr),
            self.vu.sp_rd_dat.eq(self.sp.fill.dat_r),
            self.sp.gather.adr.eq(self.vu.sp_wr_adr),
            # The integrator's rule: a withheld grant means the write does not happen.
            self.sp.gather.we.eq(Mux(self.vu.sp_wr_gnt, self.vu.sp_wr_we, 0)),
            self.sp.gather.dat_w.eq(self.vu.sp_wr_dat),
        ]


def _csr_write(csr, value):
    yield csr.storage.eq(value)
    yield csr.re.eq(1)
    yield
    yield csr.re.eq(0)
    yield


def _run_rtl(words, sp_bytes, args, gnt_rng=None, max_cycles=100000):
    dut = _DUT(words_from_bytes(bytes(sp_bytes)))
    out = {}

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
        for _ in range(max_cycles):
            if not (yield dut.vu.busy):
                break
            yield
        else:
            raise AssertionError("vector unit timeout")
        yield dut.vu.sp_wr_gnt.eq(1)
        yield
        data = bytearray()
        for w in range(SP_SIZE//32):
            v = (yield dut.sp.mem[w])
            data += int(v).to_bytes(32, "little")
        out["sp"] = bytes(data)
        out["s5"] = (yield dut.vu.res5.status)
        out["s6"] = (yield dut.vu.res6.status)
        out["s7"] = (yield dut.vu.res7.status)
        out["bad"] = (yield dut.vu.status.fields.bad_op)
        out["dc"]  = (yield dut.vu.done_count.status)

    gens = [gen()]
    if gnt_rng is not None:
        # Withhold the store grant ~40% of cycles. Bounded by the main generator's
        # completion: run_simulation only returns once EVERY generator has finished.
        def gnt_gen():
            while "sp" not in out:
                yield dut.vu.sp_wr_gnt.eq(1 if gnt_rng.random() < 0.6 else 0)
                yield
        gens.append(gnt_gen())
    run_simulation(dut, gens)
    return out


def _check(seed, withhold):
    rng  = random.Random(seed)
    prog = gen_program(rng)
    words = asm.assemble(prog)
    sp   = _seeded_sp(rng)
    args = [SRC0, SRC1, DST, rng.randint(2, 4)]
    ref  = bytearray(sp)
    S, _ = asm.VecInterp(ref).run(words, list(args))
    got  = _run_rtl(words, sp, args, gnt_rng=random.Random(seed + 7) if withhold else None)
    m48  = (1 << 48) - 1
    diffs = [i for i in range(SP_SIZE) if got["sp"][i] != ref[i]]
    assert not diffs, (seed, prog, f"first diffs at {diffs[:8]}")
    assert got["s5"] == (S[5] & m48), (seed, prog, "s5", got["s5"], S[5] & m48)
    assert got["s6"] == (S[6] & m48), (seed, prog, "s6", got["s6"], S[6] & m48)
    assert got["s7"] == (S[7] & m48), (seed, prog, "s7", got["s7"], S[7] & m48)
    assert got["bad"] == 0 and got["dc"] == 1, (seed, prog, got["bad"], got["dc"])


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(24)))
def test_random_program_matches_interpreter(seed):
    _check(seed, withhold=False)


@pytest.mark.parametrize("seed", list(range(100, 112)))
def test_random_program_under_withheld_store_grants(seed):
    _check(seed, withhold=True)

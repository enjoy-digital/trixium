#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Optimized microcode must compute exactly what the version it replaced computed.

The existing gates cannot catch a semantic change here. They compare the RTL against the
interpreter, and both execute the *same* program words - change the program and they move
together, still agreeing, while the model quietly computes something else. The token gates
have the same blind spot: `hybrid_generate` runs the interpreter too.

So this pins the pre-optimization programs as literals and asserts the current ones produce
byte-identical scratchpad output on random inputs. When a program is deliberately changed in a
way that alters results, the literal here has to be updated in the same commit - which is the
point: it makes that a visible decision rather than an invisible drift.
"""

import numpy as np
import pytest

from trixium.model import asm as vasm

# Pre-hoist programs, verbatim as they were before loop-invariant work was moved out. ---------

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
    # 2026-08-29: the two clamp constants (-15*2^15, 14*2^15) were rebuilt in v4 every
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
            # In-contract k only (asm.py, 2026-08-29): k = bitlen(absmax) - 14 of the data
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

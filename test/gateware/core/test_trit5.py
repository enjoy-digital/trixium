#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Trit5Unpacker (gateware/core/trit5.py): rows of 5-trits-per-byte code beats back to the pipe's
2-bit code beats, scale beats passed through, row padding dropped, under random stalls on both
sides; and the pass-through mode."""

import random

import pytest

from migen import *

from gateware.core.trit5 import Trit5Unpacker, trit5_pack, trit5_decode_table, BEAT_BYTES

# Reference ----------------------------------------------------------------------------------------

def codes_word(codes):
    return sum(c << (2*j) for j, c in enumerate(codes))


def make_rows(rng, rows, scale_beats, code_beats):
    """(memory beats as ints, expected pipe beats as ints) for `rows` packed rows."""
    mem, exp = bytearray(), []
    for _ in range(rows):
        row = bytearray()
        for _ in range(scale_beats):
            s = bytes(rng.getrandbits(8) for _ in range(BEAT_BYTES))
            row += s
            exp.append(int.from_bytes(s, "little"))
        for _ in range(code_beats):
            codes = [rng.randrange(3) for _ in range(128)]
            row += trit5_pack(codes)
            exp.append(codes_word(codes))
        row += bytes(-len(row) % BEAT_BYTES) # Row padding.
        mem += row
    beats = [int.from_bytes(mem[i:i + BEAT_BYTES], "little") for i in range(0, len(mem), BEAT_BYTES)]
    return beats, exp


def test_decode_table():
    table = trit5_decode_table()
    rng = random.Random(1)
    for _ in range(200):
        codes = [rng.randrange(3) for _ in range(128)]
        packed = trit5_pack(codes)
        got = []
        for i, b in enumerate(packed):
            w = table[b]
            got += [(w >> (2*k)) & 3 for k in range(min(5, 128 - 5*i))]
        assert got == codes
    assert table[243] == table[255] == 0x3ff

# Simulation ---------------------------------------------------------------------------------------

def run(beats, packed, scale_beats, code_beats, n_out, seed, in_rate=0.7, out_rate=0.7):
    dut = Trit5Unpacker()
    rng = random.Random(seed)
    got = []
    cycles = {"first": None, "last": None, "n": 0}

    def feed():
        yield dut.packed.eq(packed)
        yield dut.scale_beats.eq(scale_beats)
        yield dut.code_beats.eq(code_beats)
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        yield
        for beat in beats:
            while rng.random() > in_rate:
                yield dut.sink.valid.eq(0)
                yield
            yield dut.sink.valid.eq(1)
            yield dut.sink.data.eq(beat)
            yield
            while not (yield dut.sink.ready):
                yield
        yield dut.sink.valid.eq(0)

    def drain():
        timeout = 0
        while len(got) < n_out and timeout < 20000:
            ready = rng.random() < out_rate
            yield dut.source.ready.eq(ready)
            yield
            if ready and (yield dut.source.valid):
                got.append((yield dut.source.data))
                cycles["first"] = cycles["first"] or cycles["n"]
                cycles["last"] = cycles["n"]
            cycles["n"] += 1
            timeout += 1
        # Nothing more comes out.
        yield dut.source.ready.eq(1)
        for _ in range(20):
            yield
            assert not (yield dut.source.valid)

    run_simulation(dut, [feed(), drain()])
    run.cycles = cycles["last"] - cycles["first"] + 1
    return got


@pytest.mark.parametrize("scale_beats,code_beats", [(1, 16), (1, 3), (2, 48), (1, 1), (3, 7)])
def test_unpack_rows(scale_beats, code_beats):
    rng = random.Random(scale_beats*100 + code_beats)
    rows = 5
    beats, exp = make_rows(rng, rows, scale_beats, code_beats)
    got = run(beats, 1, scale_beats, code_beats, len(exp), seed=code_beats)
    assert len(got) == len(exp)
    for i, (g, e) in enumerate(zip(got, exp)):
        assert g == e, f"beat {i}"


def test_full_rate():
    """No stalls: one pipe beat per clock while the memory keeps up (code beats need 26/32)."""
    rng = random.Random(5)
    beats, exp = make_rows(rng, 8, 1, 16)
    got = run(beats, 1, 1, 16, len(exp), seed=5, in_rate=1.0, out_rate=1.0)
    assert got == exp
    assert run.cycles == len(exp), (run.cycles, len(exp))


def test_pass_through():
    rng = random.Random(7)
    beats = [rng.getrandbits(256) for _ in range(40)]
    got = run(beats, 0, 1, 16, len(beats), seed=7)
    assert got == beats


def test_invalid_byte_is_code_3():
    codes = [1]*128
    packed = bytearray(trit5_pack(codes))
    packed[0] = 250
    row = bytes(BEAT_BYTES) + bytes(packed)
    row += bytes(-len(row) % BEAT_BYTES)
    beats = [int.from_bytes(row[i:i + BEAT_BYTES], "little") for i in range(0, len(row), BEAT_BYTES)]
    got = run(beats, 1, 1, 1, 2, seed=3)
    assert got[1] & 0x3ff == 0x3ff

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import math
import random
import struct

import pytest

from migen import *

from litex.gen import LiteXModule

from gateware.units.activation import ActivationScratchpad
from gateware.core.scratchpad import Scratchpad
from test_activation_quantize import bits, expected


# Helpers ------------------------------------------------------------------------------------------

class QuantSP(LiteXModule):
    def __init__(self, size=8192):
        self.sp = Scratchpad(size=size)
        self.unit = ActivationScratchpad(size)
        self.grant = Signal(reset=1)
        self.comb += [self.sp.fill.adr.eq(self.unit.sp_rd_adr), self.unit.sp_rd_dat.eq(self.sp.fill.dat_r),
                     self.unit.sp_wr_gnt.eq(self.grant & self.unit.sp_wr_req),
                     self.sp.gather.adr.eq(self.unit.sp_wr_adr), self.sp.gather.dat_w.eq(self.unit.sp_wr_dat),
                     self.sp.gather.we.eq(Mux(self.grant & self.unit.sp_wr_req, self.unit.sp_wr_we, 0))]


def pack(values, width):
    return sum((v & ((1 << width) - 1)) << (width*i) for i, v in enumerate(values))


def reference(words):
    values = [struct.unpack('<f', struct.pack('<I', word))[0] for word in words]
    assert all(math.isfinite(value) for value in values)
    maximum = max(map(abs, values))
    e8 = math.frexp(maximum)[1] - 7 if maximum else 0
    result, error = expected(words, e8)
    assert error == 0
    return result.to_bytes(len(words), 'little'), e8


def submit(dut, **changed):
    fields = dict(src=0x400, dst=0x1800, exponents=0x1f00, elements=64, blocks=3,
                  protect_start=0, protect_end=64)
    fields.update(changed)
    for name, value in fields.items():
        yield getattr(dut.unit.cmd, name).eq(value)
    yield dut.unit.cmd.valid.eq(1)
    yield
    assert (yield dut.unit.cmd.ready)
    yield dut.unit.cmd.valid.eq(0)
    yield


def valid_command(dut, arrays, stall=True):
    elements, blocks = len(arrays[0]), len(arrays)
    words = [word for array in arrays for word in array]
    for beat in range(len(words)//8):
        yield dut.sp.mem[0x400//32 + beat].eq(pack(words[beat*8:beat*8 + 8], 32))
    sentinel = pack([0xdeadbeef]*8, 32)
    for address in (0x17e0, 0x1800 + len(words), 0x1f00):
        yield dut.sp.mem[address//32].eq(sentinel)
    yield from submit(dut, elements=elements, blocks=blocks)
    rng = random.Random(940)
    held = None
    for cycle in range(4000):
        # Long stalls fill the converter pipeline and reserved-read FIFO, not
        # only the final word register. The credit boundary must remain bounded.
        yield dut.grant.eq(not stall or (cycle % 83 not in range(9, 41) and rng.randrange(3) != 0))
        yield dut.unit.cmd.src.eq(0xffffe0)
        yield dut.unit.cmd.elements.eq(0)
        yield dut.unit.cmd.blocks.eq(65535)
        yield
        payload = (yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat)
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        assert not (yield dut.unit.error), ((yield dut.unit.error), cycle)
        if (yield dut.unit.done):
            break
    else:
        raise AssertionError('resident activation conversion did not complete')
    assert (yield dut.unit.blocks_done) == blocks
    assert (yield dut.unit.read_beats) == len(words)//4  # Two passes, eight inputs/read.
    assert (yield dut.unit.output_beats) == len(words)//32
    expected_bytes = b''.join(reference(array)[0] for array in arrays)
    for beat in range(len(words)//32):
        assert (yield dut.sp.mem[0x1800//32 + beat]) == int.from_bytes(expected_bytes[32*beat:32*(beat+1)], 'little')
    stats = yield dut.sp.mem[0x1f00//32]
    for index, array in enumerate(arrays):
        assert (stats >> (32*index)) & 0xffffffff == reference(array)[1] & 0xffffffff
    assert stats >> (32*blocks) == sentinel >> (32*blocks)
    assert (yield dut.sp.mem[0x17e0//32]) == sentinel
    assert (yield dut.sp.mem[(0x1800 + len(words))//32]) == sentinel
    yield


# Tests --------------------------------------------------------------------------------------------

def test_resident_quantization_per_block_and_whole_vector_reuse():
    dut = QuantSP()
    rng = random.Random(500)
    arrays = [[bits(math.ldexp(rng.uniform(-127.9, 127.9), exponent)) for _ in range(256)]
              for exponent in (-100, 0, 100)]
    arrays[0][:4] = [bits(math.ldexp(x, -100)) for x in (0.5, -0.5, 2.5, -2.5)]
    def run():
        yield from valid_command(dut, arrays)
        yield from valid_command(dut, [[v for a in arrays for v in a]], stall=False)
        yield from valid_command(dut, [[0, 0x80000000]*32])
    run_simulation(dut, run())


def test_resident_exponent_tail_at_exact_physical_end():
    dut = QuantSP(size=0x1f20)
    run_simulation(dut, valid_command(dut, [[bits(1.0)]*64]))


@pytest.mark.parametrize('word', [bits(2.0**-121), 0x7f7fffff, 0xff7fffff])
def test_resident_exponent_extremes(word):
    dut = QuantSP()
    run_simulation(dut, valid_command(dut, [[word]*64]))


@pytest.mark.parametrize('changed,mask', [({'elements': 0}, 1), ({'elements': 40}, 1),
    ({'blocks': 0}, 1), ({'src': 1}, 1), ({'dst': 0x1801}, 1), ({'exponents': 0x1f01}, 1),
    ({'elements': 65504, 'blocks': 65535}, 2), ({'src': 0xffffe0}, 2), ({'dst': 0x1fe0}, 2),
    ({'exponents': 0xffffe0}, 2), ({'dst': 0x400}, 4), ({'exponents': 0x400}, 4),
    ({'exponents': 0x1800}, 4), ({'protect_start': 64, 'protect_end': 32}, 1),
    ({'protect_end': 0x4000}, 2), ({'protect_start': 0x1800, 'protect_end': 0x1820}, 64),
    ({'protect_start': 0x1f00, 'protect_end': 0x1f20}, 64)])
def test_resident_quantization_rejects_complete_invalid_spans_before_access(changed, mask):
    dut = QuantSP()
    def run():
        yield from submit(dut, **changed)
        for _ in range(8):
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            yield
        assert (yield dut.unit.busy) and not (yield dut.unit.done)
        assert (yield dut.unit.error) & mask
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.busy) and not (yield dut.unit.error)
    run_simulation(dut, run())


@pytest.mark.parametrize('word,mask', [(0x7f800000, 8), (0xff800000, 8), (0x7fc00001, 8),
    (1, 16), (0x807fffff, 16), (bits(2.0**-122), 16)])
def test_resident_quantization_rejects_numerical_failure_without_writes(word, mask):
    dut = QuantSP()
    def run():
        for beat in range(8):
            yield dut.sp.mem[0x400//32 + beat].eq(pack([word]*8, 32))
        yield from submit(dut, blocks=1)
        for _ in range(80):
            assert not (yield dut.unit.sp_wr_req)
            assert not (yield dut.unit.done)
            yield
        assert (yield dut.unit.busy) and (yield dut.unit.error) == mask
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        yield from valid_command(dut, [[bits(0.5)]*64])
    run_simulation(dut, run())

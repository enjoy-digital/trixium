#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import random

import pytest

from migen import *

from litex.gen import LiteXModule

from trixium.units.hadamard import HadamardScratchpad
from trixium.core.scratchpad import Scratchpad
from test_hadamard import dense, packed, unpacked


# Helpers ------------------------------------------------------------------------------------------

class RotationSP(LiteXModule):
    def __init__(self, size=8192, block_size=256):
        self.sp = Scratchpad(size=size)
        self.unit = HadamardScratchpad(size, block_size)
        self.grant = Signal(reset=1)
        self.comb += [self.sp.fill.adr.eq(self.unit.sp_rd_adr), self.unit.sp_rd_dat.eq(self.sp.fill.dat_r),
                     self.unit.sp_wr_gnt.eq(self.grant & self.unit.sp_wr_req),
                     self.sp.gather.adr.eq(self.unit.sp_wr_adr), self.sp.gather.dat_w.eq(self.unit.sp_wr_dat),
                     self.sp.gather.we.eq(Mux(self.grant & self.unit.sp_wr_req, self.unit.sp_wr_we, 0))]


def submit(dut, **fields):
    for name, value in fields.items():
        yield getattr(dut.unit.cmd, name).eq(value)
    yield dut.unit.cmd.valid.eq(1)
    yield
    assert (yield dut.unit.cmd.ready)
    yield dut.unit.cmd.valid.eq(0)
    yield


def run_command(dut, blocks=3, inverse=False, size=256, seed=3):
    rng = random.Random(seed)
    src, signs, dst, stats = 0x100, 0x500, 0x800, 0x1800
    values = [rng.randrange(-128, 128) for _ in range(blocks*size)]
    sg = [rng.choice([-1, 1]) for _ in values]
    expected = [dense(values[b*size:(b+1)*size], sg[b*size:(b+1)*size], inverse) for b in range(blocks)]
    for beat in range(blocks*size//32):
        yield dut.sp.mem[src//32 + beat].eq(packed(values[beat*32:(beat+1)*32], 8))
    for beat in range(blocks*size//256):
        yield dut.sp.mem[signs//32 + beat].eq(sum((s < 0) << i for i, s in enumerate(sg[beat*256:(beat+1)*256])))
    sentinel = packed([0xdeadbeef]*8, 32)
    yield dut.sp.mem[stats//32].eq(sentinel)
    yield dut.sp.mem[dst//32 - 1].eq(sentinel)
    yield dut.sp.mem[(dst + blocks*size*4)//32].eq(sentinel)
    yield from submit(dut, src=src, signs=signs, dst=dst, stats=stats, blocks=blocks, inverse=inverse)
    held = None
    for cycle in range(10000):
        yield dut.grant.eq(rng.randrange(4) != 0)
        # Command fields are snapshotted, never live during execution.
        yield dut.unit.cmd.src.eq(0xffffff)
        yield dut.unit.cmd.blocks.eq(0xffff)
        yield
        payload = (yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat)
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        if (yield dut.unit.done):
            break
        assert not (yield dut.unit.error)
    else:
        raise AssertionError('scratchpad rotation timed out')
    assert (yield dut.unit.blocks_done) == blocks
    assert (yield dut.unit.output_beats) == blocks*size//8
    for block, expected_block in enumerate(expected):
        actual = []
        for beat in range(size//8):
            actual += unpacked((yield dut.sp.mem[(dst + block*size*4)//32 + beat]), 32, 8)
        assert actual == expected_block
    stat_words = unpacked((yield dut.sp.mem[stats//32]), 32, 8)
    assert stat_words[:blocks] == [max(map(abs, values)) for values in expected]
    assert stat_words[blocks:] == [0xdeadbeef - (1 << 32)]*(8 - blocks)
    assert (yield dut.sp.mem[dst//32 - 1]) == sentinel
    assert (yield dut.sp.mem[(dst + blocks*size*4)//32]) == sentinel


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('inverse', [False, True])
def test_hadamard_shared_scratchpad_words_signs_bounds_and_reuse(inverse):
    dut = RotationSP()
    def run():
        yield from run_command(dut, inverse=inverse)
        yield
        yield from run_command(dut, inverse=not inverse, seed=11)
    run_simulation(dut, run())


@pytest.mark.parametrize('changed,mask', [
    ({'blocks': 0}, 1), ({'src': 1}, 1), ({'signs': 0x501}, 1),
    ({'dst': 0x801}, 1), ({'stats': 0x1801}, 1), ({'blocks': 65535}, 2),
    ({'src': 0xffffe0}, 2), ({'dst': 0xffffe0}, 2), ({'signs': 0xffffe0}, 2),
    ({'stats': 0xffffe0}, 2), ({'dst': 0x100}, 4), ({'dst': 0x500}, 4),
    ({'stats': 0x100}, 4), ({'stats': 0x500}, 4), ({'stats': 0x800}, 4),
])
def test_hadamard_rejects_full_span_before_any_scratchpad_access(changed, mask):
    dut = RotationSP()
    def run():
        fields = dict(src=0x100, signs=0x500, dst=0x800, stats=0x1800, blocks=3, inverse=0)
        fields.update(changed)
        yield from submit(dut, **fields)
        for _ in range(8):
            assert (yield dut.unit.error) & mask
            assert (yield dut.unit.busy) and not (yield dut.unit.done)
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            yield
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.busy) and not (yield dut.unit.error)
    run_simulation(dut, run())

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/hadamard.py: IntegerHadamard, HadamardScratchpad and the rotation lists on the
shared cluster."""

import random

import numpy as np
import pytest

from migen import *

from litex.gen import LiteXModule

from gateware.units.hadamard import IntegerHadamard, HadamardScratchpad
from gateware.core.scratchpad import Scratchpad
from gateware.core.core import TriXiumConfig
from model.hadamard import scratchpad_rotation_descriptor

from common import RTL
from common.cluster import SoC, write, install_list
from common.verilator import Simulator

# IntegerHadamard ----------------------------------------------------------------------------------
# Independent dense integer oracle; no floating-point butterfly reference.

def dense(values, signs, inverse=False):
    size = len(values)
    h = np.array([[1 - 2*((i & j).bit_count() & 1) for j in range(size)]
                  for i in range(size)], dtype=np.int64)
    x, s = np.asarray(values, dtype=np.int64), np.asarray(signs, dtype=np.int64)
    return ((h @ x)*s if inverse else h @ (s*x)).tolist()


def packed(values, width):
    return sum((int(value) & ((1 << width) - 1)) << (i*width) for i, value in enumerate(values))


def unpacked(value, width, count):
    words = [(value >> (i*width)) & ((1 << width) - 1) for i in range(count)]
    return [word - (1 << width) if word & (1 << (width-1)) else word for word in words]


def begin(dut, inverse=False):
    yield dut.cmd.inverse.eq(inverse)
    yield dut.cmd.valid.eq(1)
    yield
    assert (yield dut.cmd.ready)
    yield dut.cmd.valid.eq(0)
    yield
    assert (yield dut.busy)


def command(dut, values, signs, inverse=False, stalls=True, seed=7):
    rng = random.Random(seed)
    expected = dense(values, signs, inverse)
    yield dut.sink.valid.eq(0)
    yield dut.source.ready.eq(0)
    yield from begin(dut, inverse)
    sent, received, held = 0, [], None
    lanes, count = dut.lanes, len(values)//dut.lanes
    offered = False
    transfer_cycles = []
    for cycle in range(10*len(values) + 100):
        # Changing the command payload while busy must not affect the latched mode.
        yield dut.cmd.inverse.eq(not inverse)
        # Withdraw the extra offer once the final result is transferred; leaving
        # valid high after returning to IDLE would legitimately start a new block.
        yield dut.cmd.valid.eq(len(received) < len(values))
        if sent < count:
            offered = offered or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offered)
            yield dut.sink.data.eq(packed(values[sent*lanes:(sent+1)*lanes], dut.input_width))
            yield dut.sink.negate.eq(sum((s < 0) << i for i, s in enumerate(signs[sent*lanes:(sent+1)*lanes])))
            yield dut.sink.last.eq(sent == count - 1)
        else:
            yield dut.sink.valid.eq(0)
        yield dut.source.ready.eq(not stalls or rng.randrange(3) != 0)
        yield
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            sent += 1
            offered = False
        valid, ready = (yield dut.source.valid), (yield dut.source.ready)
        current = (yield dut.source.data), (yield dut.source.first), (yield dut.source.last)
        if held is not None:
            assert valid and current == held, "output changed while stalled"
        held = current if valid and not ready else None
        if valid and ready:
            assert current[1] == (not received)
            assert current[2] == (len(received) + lanes == len(values))
            received.extend(unpacked(current[0], dut.output_width, lanes))
            transfer_cycles.append(cycle)
        if (yield dut.done):
            assert not (yield dut.busy)
            assert not (yield dut.error)
            assert not (yield dut.cancelled)
            assert received == expected
            assert (yield dut.absmax) == max(map(abs, expected))
            yield dut.cmd.valid.eq(0)
            yield dut.sink.valid.eq(0)
            yield
            assert not (yield dut.done)
            assert not (yield dut.source.valid)
            return transfer_cycles
        assert not (yield dut.cmd.ready), "second command accepted while busy"
    raise AssertionError(f"stalled: sent {sent} words, received {len(received)} values")


@pytest.mark.parametrize("size,lanes,width", [(4, 2, 8), (4, 2, 32), (16, 2, 8), (16, 8, 8),
                                               (128, 4, 16), (1024, 8, 8)])
@pytest.mark.parametrize("inverse", [False, True])
def test_integer_hadamard_dense_equivalence(size, lanes, width, inverse):
    rng = random.Random(size + width)
    values = [rng.randrange(-(1 << (width-1)), 1 << (width-1)) for _ in range(size)]
    values[0] = -(1 << (width-1))
    signs = [rng.choice([-1, 1]) for _ in range(size)]
    dut = IntegerHadamard(size, lanes, width)
    run_simulation(dut, command(dut, values, signs, inverse))


def test_integer_hadamard_extremes_reuse_and_dense_rate():
    dut = IntegerHadamard()
    size, lanes = dut.block_size, dut.lanes

    def run():
        for values, signs, inverse in [
            ([0]*size, [-1]*size, False),
            ([-128]*size, [-1]*size, False),  # +131072: requires the extra sign bit.
            ([-128]*size, [-1]*size, True),
            ([127]*size, [1 if i % 2 else -1 for i in range(size)], False),
        ]:
            cycles = yield from command(dut, values, signs, inverse, stalls=False)
            assert all(b == a + 1 for a, b in zip(cycles, cycles[1:]))
            # 128 input cycles + 10*(64 issue + 1 drain) + one output-read cycle.
            assert cycles[0] <= size//lanes + 10*(size//(2*lanes) + 1) + 2
    run_simulation(dut, run())


def test_integer_hadamard_matches_dense_engine_rotation_contract():
    from model import bitexact as bx

    size = 1024
    rng = np.random.default_rng(2714)
    signs = rng.choice([-1, 1], size)
    h = np.array([[1 - 2*((i & j).bit_count() & 1) for j in range(size)]
                  for i in range(size)], dtype=np.int64)
    codes = (h*signs[None, :] + 1).astype(np.uint8)
    scales = np.full((size, size//128), 1/32, dtype=np.float16)
    _, exponent = bx.fp16_decompose(scales)
    bias = bx.choose_scale_bias(exponent)
    shift = bx.choose_result_shift(scales, bias)
    assert (bias, shift) == (27, 8)
    dut = IntegerHadamard()

    def run():
        # The existing quantizer uses [-127,127]. -128 is safe in the new raw
        # core but not necessarily after the legacy h<<14 int32 presentation.
        for values in [np.zeros(size, dtype=np.int8),
                       rng.integers(-127, 128, size, dtype=np.int8),
                       (127*signs).astype(np.int8)]:
            raw = h @ (signs*values.astype(np.int64))
            engine = bx.matvec_ternary(scales, codes, values, bias, shift)
            assert np.array_equal(engine.astype(np.int64), raw << 14)
            yield from command(dut, values.tolist(), signs.tolist(), stalls=False)
    run_simulation(dut, run())


@pytest.mark.parametrize("last_word", [0, None])
def test_integer_hadamard_rejects_early_or_missing_last_and_cancel_recovers(last_word):
    dut = IntegerHadamard(16, 4)

    def run():
        yield from begin(dut)
        for word in range(4):
            yield dut.sink.valid.eq(1)
            yield dut.sink.data.eq(0)
            yield dut.sink.last.eq(word == last_word)
            yield
            if word == last_word:
                break
        yield dut.sink.valid.eq(0)
        for _ in range(5):
            yield
            assert (yield dut.busy) and (yield dut.error)
            assert not (yield dut.cmd.ready)
            assert not (yield dut.source.valid) and not (yield dut.done)
        yield dut.cancel.eq(1)
        yield
        yield dut.cancel.eq(0)
        yield
        assert not (yield dut.busy) and (yield dut.cancelled)
        yield from command(dut, list(range(16)), [-1, 1]*8)
    run_simulation(dut, run())


@pytest.mark.parametrize("stop_cycle", [0, 2, 5, 10, 20, 30])
def test_integer_hadamard_cancel_flushes_loading_compute_and_stalled_output(stop_cycle):
    dut = IntegerHadamard(16, 4)

    def run():
        yield from begin(dut, True)
        for cycle in range(stop_cycle):
            yield dut.sink.valid.eq(cycle < 4)
            yield dut.sink.last.eq(cycle == 3)
            yield dut.sink.data.eq(0x80808080)
            yield dut.sink.negate.eq(15)
            yield dut.source.ready.eq(0)
            yield
        yield dut.cancel.eq(1)
        yield
        assert not (yield dut.source.valid)
        assert not (yield dut.sink.ready) and not (yield dut.cmd.ready)
        yield dut.cancel.eq(0)
        yield dut.sink.valid.eq(0)
        yield
        assert not (yield dut.busy) and (yield dut.cancelled)
        for _ in range(3):
            yield
            assert not (yield dut.source.valid) and not (yield dut.done)
        yield from command(dut, [-128, 127]*8, [1, -1]*8, inverse=False)
    run_simulation(dut, run())


@pytest.mark.parametrize("args", [(3, 2, 8), (16, 3, 8), (16, 16, 8), (16, 2, 1), (16, 2, 33)])
def test_integer_hadamard_parameter_validation(args):
    with pytest.raises(ValueError):
        IntegerHadamard(*args)


def test_compiled_integer_hadamard():
    dut = IntegerHadamard()
    ios = set(dut.cmd.flatten() + dut.sink.flatten() + dut.source.flatten()) | {
        dut.cancel, dut.busy, dut.done, dut.error, dut.cancelled, dut.absmax}
    sim = Simulator(dut, ios=ios, name="integer_hadamard",
                    driver=(RTL / "integer_hadamard.cpp").read_text())
    result = sim.run(timeout=30)
    assert "HADAMARD_COMPILED_PASS" in result.stdout
    print(result.stdout)


# HadamardScratchpad -------------------------------------------------------------------------------

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


# Sequencer Lists on the Shared Cluster ------------------------------------------------------------

def test_sequenced_fetch_nowait_rotate_store_uses_actual_shared_ports():
    dut = SoC(TriXiumConfig(with_hadamard=True, scratchpad_size=16*1024), n=1)
    rng = random.Random(941)
    values = [rng.randrange(-128, 128) for _ in range(1024)]
    signs = [rng.choice([-1, 1]) for _ in values]
    expected = dense(values, signs)
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 32, 0, 0, 0, 0x800, 0] + [0]*5
    rotation = scratchpad_rotation_descriptor(src=0x800, signs=0x600, dst=0x2000, stats=0x700, blocks=1)
    store = [1, 3, 0, 0, 0x1000, 128, 0, 0, 0x2000, 0, 0] + [0]*5

    def run():
        for beat in range(32):
            yield dut.models[0].mem[0x100//32 + beat].eq(packed(values[beat*32:(beat+1)*32], 8))
            yield dut.scratchpad.mem[0x800//32 + beat].eq(0xa5)
        for beat in range(4):
            yield dut.scratchpad.mem[0x600//32 + beat].eq(sum((s < 0) << i for i, s in enumerate(signs[beat*256:(beat+1)*256])))
        yield from install_list(dut, [fetch, rotation, store])
        seen_rotation = False
        for cycle in range(12000):
            if (yield dut.hadamard.cmd.valid) and (yield dut.hadamard.cmd.ready):
                assert not (yield dut.engine.sp_busy)
                assert (yield dut.engine_csr.done_count.status) == 1
                seen_rotation = True
            assert not (yield dut.seq.hadamard_error.status)
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('sequenced rotation did not complete')
        assert seen_rotation
        assert (yield dut.seq.status.fields.dcount) == 3
        assert (yield dut.engine_csr.done_count.status) == 2
        assert (yield dut.hadamard.blocks_done) == 1
        assert (yield dut.scratchpad.mem[0x700//32]) & 0xffffffff == max(map(abs, expected))
        for beat in range(128):
            word = yield dut.scratchpad.mem[0x2000//32 + beat]
            assert unpacked(word, 32, 8) == expected[beat*8:(beat+1)*8]
            assert (yield dut.models[0].mem[0x1000//32 + beat]) == word
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


@pytest.mark.parametrize('field,value,error', [(1, 1 << 24, 2), (4, 65536, 2),
    (6, 2, 2), (7, 1, 2), (15, 1, 2), (3, 0, 4), (5, 0, 4), (1, 0x801, 1 << 8)])
def test_sequenced_rotation_invalid_descriptor_is_not_completed(field, value, error):
    dut = SoC(TriXiumConfig(with_hadamard=True, scratchpad_size=16*1024), n=1)
    words = scratchpad_rotation_descriptor(src=0x800, signs=0x600, dst=0x2000, stats=0x700, blocks=1)
    words[field] = value
    def run():
        yield from install_list(dut, [words])
        for _ in range(80):
            assert not (yield dut.hadamard.sp_wr_req)
            yield
        assert (yield dut.seq.busy)
        assert (yield dut.seq.hadamard_error.status) & error
        assert (yield dut.seq.status.fields.dcount) == 0
        yield from write(dut.engine_csr.soft_reset, 1)
        for _ in range(90):
            yield
        assert not (yield dut.seq.busy)
        assert not (yield dut.seq.hadamard_error.status)
        assert not (yield dut.hadamard.busy)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (0x3fe0, 1)])
def test_rotation_profile_checks_whole_descriptor_list_before_fetch(base, count):
    from gateware.core.sequencer import Sequencer
    dut = Sequencer(with_hadamard=True, scratchpad_size=16*1024)
    def run():
        yield from write(dut.base, base)
        yield from write(dut.count, count)
        yield from write(dut.start, 1)
        for _ in range(5):
            assert (yield dut.busy) and (yield dut.hadamard_error.status) == 1
            assert not (yield dut.sp_active)
            yield
    run_simulation(dut, run())


def test_rotation_descriptor_encoding_and_configuration():
    from gateware.core.sequencer import Sequencer
    assert scratchpad_rotation_descriptor(src=32, signs=64, dst=96, stats=128, blocks=17, inverse=True) == [3, 32, 64, 96, 17, 128, 1] + [0]*9
    for changed in ({'src': 1}, {'signs': -32}, {'dst': 1 << 24}, {'stats': 7}, {'blocks': 0}, {'blocks': 65536}, {'inverse': 2}):
        args = dict(src=32, signs=64, dst=96, stats=128, blocks=1)
        args.update(changed)
        with pytest.raises(ValueError):
            scratchpad_rotation_descriptor(**args)
    with pytest.raises(ValueError):
        TriXiumConfig(with_hadamard=True, with_vecunit=False)
    with pytest.raises(ValueError):
        TriXiumConfig(with_hadamard=True, scratchpad_size=512*1024)
    for size in (0, 31, 33, 256*1024 + 32):
        with pytest.raises(ValueError):
            Sequencer(with_hadamard=True, scratchpad_size=size)


def test_compiled_full_vector_rotation_through_shared_cluster_and_wishbone():
    dut = SoC(TriXiumConfig(with_hadamard=True), n=1)
    bus = dut.bus
    # A bare RTL fixture has no CSRBank finalization to assemble fielded status.
    # Export the real driven fields, not CSRStatus.status (otherwise an input).
    status = Signal(24)
    dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
    signals = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
    signals.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
                   seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re,
                   seq_status=status, seq_error=dut.seq.hadamard_error.status,
                   reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
                   reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
                   blocks_done=dut.hadamard.blocks_done, output_beats=dut.hadamard.output_beats)
    for name, signal in signals.items():
        signal.name_override = name
    # Runs in the build directory, next to the memory .init files.
    sim = Simulator(dut, ios=signals, name='hadamard_cluster',
                    driver=(RTL / 'hadamard_cluster.cpp').read_text())
    result = sim.run(timeout=120)
    assert '$readmem file not found' not in result.stdout
    assert 'HADAMARD_CLUSTER_PASS' in result.stdout
    print(result.stdout)


def test_overlap_rotation_runs_under_a_nowait_fetch_on_the_shared_ports():
    """w0[9] on a unit (OVERLAP): the rotation dispatches while the NOWAIT fetch is still
    streaming into another region; both results exact, the list end drains the fetch."""
    dut = SoC(TriXiumConfig(with_hadamard=True, scratchpad_size=32*1024), n=1)
    rng = random.Random(942)
    values = [rng.randrange(-128, 128) for _ in range(1024)]
    signs = [rng.choice([-1, 1]) for _ in values]
    expected = dense(values, signs)
    stream = [rng.randrange(1 << 256) for _ in range(160)]
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, len(stream), 0, 0, 0, 0x4000, 0] + [0]*5
    rotation = scratchpad_rotation_descriptor(src=0x800, signs=0x600, dst=0x2000, stats=0x700, blocks=1)
    rotation[0] |= 1 << 9

    def run():
        for beat, word in enumerate(stream):
            yield dut.models[0].mem[0x100//32 + beat].eq(word)
        for beat in range(32):
            yield dut.scratchpad.mem[0x800//32 + beat].eq(packed(values[beat*32:(beat+1)*32], 8))
        for beat in range(4):
            yield dut.scratchpad.mem[0x600//32 + beat].eq(sum((s < 0) << i for i, s in enumerate(signs[beat*256:(beat+1)*256])))
        yield from install_list(dut, [fetch, rotation])
        overlapped = False
        for cycle in range(20000):
            if (yield dut.hadamard.busy) and (yield dut.engine.sp_busy):
                overlapped = True
            assert not (yield dut.seq.hadamard_error.status)
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('overlapped list did not complete')
        assert overlapped
        assert (yield dut.seq.status.fields.dcount) == 2
        assert (yield dut.engine_csr.done_count.status) == 1
        for beat in range(128):
            word = yield dut.scratchpad.mem[0x2000//32 + beat]
            assert unpacked(word, 32, 8) == expected[beat*8:(beat+1)*8]
        for beat, word in enumerate(stream):
            assert (yield dut.scratchpad.mem[0x4000//32 + beat]) == word, beat
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_without_overlap_the_rotation_waits_for_the_nowait_fetch():
    dut = SoC(TriXiumConfig(with_hadamard=True, scratchpad_size=32*1024), n=1)
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 160, 0, 0, 0, 0x4000, 0] + [0]*5
    rotation = scratchpad_rotation_descriptor(src=0x800, signs=0x600, dst=0x2000, stats=0x700, blocks=1)

    def run():
        yield from install_list(dut, [fetch, rotation])
        for cycle in range(20000):
            assert not ((yield dut.hadamard.busy) and (yield dut.engine.sp_busy))
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('list did not complete')
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})

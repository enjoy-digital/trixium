#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import pytest

from migen import Memory, Module, run_simulation

from trixium.core.core import TriXiumConfig
from trixium.core.sequencer import Sequencer
from trixium.model.activation import scratchpad_restore_descriptor
from trixium.model import asm
from test_projection_restore import expected
from test_projection_restore_sp import pack
from test_hadamard_integration import install_list
from test_integration import SoC, write


# Helpers ------------------------------------------------------------------------------------------

def descriptor(**changes):
    fields = dict(src=0x800, dst=0x1000, exponents=0x700, elements=17, adjustment=-13)
    fields.update(changes)
    return scratchpad_restore_descriptor(**fields)


def fixture(descriptors):
    dut = Module()
    dut.submodules.seq = seq = Sequencer(with_projection_restore=True)
    words = [pack(desc[start:start+8]) for desc in descriptors for start in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    return dut


def start(seq, count):
    yield from write(seq.base, 0)
    yield from write(seq.count, count)
    yield from write(seq.start, 1)


# Tests --------------------------------------------------------------------------------------------

def test_restoration_descriptor_and_configuration():
    assert descriptor() == [8, 0x800, 0x1000, 0x700, 17, 0xfffffff3] + [0]*10
    for changes in ({'src': 1}, {'dst': -32}, {'exponents': 1 << 24}, {'elements': 0},
                    {'elements': 65536}, {'adjustment': -(1 << 31)-1}, {'adjustment': 1 << 31}):
        with pytest.raises(ValueError):
            descriptor(**changes)
    assert descriptor(adjustment=-(1 << 31))[5] == 0x80000000
    for options in ({'with_vecunit': False}, {'scratchpad_size': 512*1024}):
        with pytest.raises(ValueError):
            TriXiumConfig(with_projection_restore=True, **options)
    for size in (0, 31, 33, 256*1024 + 32):
        with pytest.raises(ValueError):
            Sequencer(with_projection_restore=True, scratchpad_size=size)


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (0x3fe0, 1)])
def test_restoration_only_profile_list_bounds_before_fetch(base, count):
    dut = Sequencer(with_projection_restore=True, scratchpad_size=16*1024)
    def run():
        yield from write(dut.base, base)
        yield from write(dut.count, count)
        yield from write(dut.start, 1)
        for _ in range(5):
            assert (yield dut.busy) and (yield dut.projection_restore_error.status) == 1
            assert not (yield dut.sp_active) and not (yield dut.projection_restore_cmd.valid)
            yield
    run_simulation(dut, run())


@pytest.mark.parametrize('field,value', [(0, 8 | 1 << 10), (0, 8 | 1 << 4), (0, 9), (0, 7),
    (1, 1 << 24), (2, 1 << 24), (3, 1 << 24), (4, 65536)] + [(field, 1) for field in range(6, 16)])
def test_restoration_reserved_fields_rejected_before_dispatch(field, value):
    words = descriptor()
    words[field] = value
    dut = fixture([words])
    def run():
        yield from start(dut.seq, 1)
        for _ in range(30):
            yield
            assert not (yield dut.seq.projection_restore_cmd.valid)
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
        assert (yield dut.seq.projection_restore_error.status) == 2
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 8, 16, 64])
def test_restoration_waits_nowait_engine_stable_dispatch_and_failure_wins(failure):
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 3, 0, 0, 0, 0x800] + [0]*6
    dut = fixture([fetch, descriptor()])
    seq, cmd = dut.seq, dut.seq.projection_restore_cmd
    def run():
        assert (yield seq.projection_restore_abi.status) == 0x10000
        yield from start(seq, 2)
        for _ in range(30):
            yield
            if (yield seq.eng_valid):
                break
        else:
            raise AssertionError('NOWAIT fixture did not launch')
        yield seq.eng_ready.eq(1)
        yield
        yield seq.eng_ready.eq(0)
        for _ in range(30):
            yield
            assert not (yield cmd.valid)
        assert (yield seq.status.fields.dcount) == 1 and (yield seq.busy)
        yield seq.eng_done.eq(1)
        yield
        yield seq.eng_done.eq(0)
        for _ in range(10):
            yield
            if (yield cmd.valid):
                break
        else:
            raise AssertionError('restoration did not dispatch after NOWAIT drained')
        for _ in range(5):
            assert (yield cmd.valid) and (yield seq.status.fields.dcount) == 1
            fields = dict(src=0x800, dst=0x1000, exponents=0x700, elements=17,
                          adjustment=0xfffffff3, protect_start=0, protect_end=128)
            for name, value in fields.items():
                assert (yield getattr(cmd, name)) == value
            yield
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        for _ in range(5):
            yield
            assert (yield seq.busy) and (yield seq.status.fields.dcount) == 1
        # Simultaneous failure and success must not retire the descriptor.
        yield seq.projection_restore_failed.eq(failure)
        yield seq.projection_restore_done.eq(1)
        yield
        yield seq.projection_restore_done.eq(0)
        for _ in range(10):
            yield
        assert (yield seq.projection_restore_error.status) == failure << 8
        assert (yield seq.status.fields.dcount) == (1 if failure else 2)
        assert bool((yield seq.busy)) == bool(failure)
    run_simulation(dut, run())


@pytest.mark.parametrize('companions', [False, True])
def test_restoration_fetch_nowait_restore_store_shared_cluster(companions):
    dut = SoC(TriXiumConfig(with_projection_restore=True, with_hadamard=companions,
                                 with_activation_quantize=companions, scratchpad_size=16*1024), n=1)
    words = [(i - 8)*(1 << 24) + (i % 5) for i in range(17)]
    restored, error = expected(words, -31)
    assert not error
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 3, 0, 0, 0, 0x800] + [0]*6
    store = [1, 3, 0, 0, 0x2000, 3, 0, 0, 0x1000, 0] + [0]*6
    def run():
        for beat in range(3):
            yield dut.models[0].mem[0x100//32 + beat].eq(pack((words + [0x7fffffff]*7)[8*beat:8*beat+8]))
            yield dut.scratchpad.mem[0x1000//32 + beat].eq(pack([0xdeadbeef]*8))
        yield dut.scratchpad.mem[0x700//32].eq((-18) & 0xffffffff)
        yield from install_list(dut, [fetch, descriptor(), store])
        seen = False
        for _ in range(1000):
            if (yield dut.projection_restore.cmd.valid) and (yield dut.projection_restore.cmd.ready):
                seen = True
                assert not (yield dut.engine.sp_busy)
                assert (yield dut.engine_csr.done_count.status) == 1
            assert not (yield dut.seq.projection_restore_error.status)
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('fetch/restoration/store list timed out')
        assert seen and (yield dut.seq.status.fields.dcount) == 3
        assert (yield dut.engine_csr.done_count.status) == 2
        for beat in range(3):
            actual = yield dut.scratchpad.mem[0x1000//32 + beat]
            assert (yield dut.models[0].mem[0x2000//32 + beat]) == actual
            for lane in range(8):
                i = 8*beat + lane
                assert (actual >> (32*lane)) & 0xffffffff == (
                    (restored >> (32*i)) & 0xffffffff if i < 17 else 0xdeadbeef)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_restoration_actual_wrapping_vector_reduce_before_float_conversion():
    dut = SoC(TriXiumConfig(with_projection_restore=True, scratchpad_size=16*1024), n=1)
    a = [(1 << 24)+1, 0x7fffffff, 0x80000000, -7, (1 << 24)+3, 0, 1, -1]
    b = [-(1 << 24), 1, -1, 7, -(1 << 24), 0, -1, 1]
    total = [(x+y) & 0xffffffff for x, y in zip(a, b)]
    result, error = expected(total, 0)
    assert not error
    reduce = [2, 0, 0x800, 0x820, 0x840, 1] + [0]*10
    def run():
        yield from write(dut.vecunit.prog_adr, 0)
        for word in asm.assemble(asm.prog_reduce_acc()):
            yield from write(dut.vecunit.prog, word)
        yield dut.scratchpad.mem[0x800//32].eq(pack(a))
        yield dut.scratchpad.mem[0x820//32].eq(pack(b))
        yield dut.scratchpad.mem[0x700//32].eq(0)
        yield from install_list(dut, [reduce, descriptor(src=0x840, dst=0x840, elements=8, adjustment=0)])
        for _ in range(300):
            yield
            assert not (yield dut.seq.projection_restore_error.status)
            if not (yield dut.seq.busy):
                break
        else:
            raise AssertionError('integer reduce/restoration list timed out')
        assert (yield dut.seq.status.fields.dcount) == 2
        assert (yield dut.scratchpad.mem[0x840//32]) == result
        assert (result & 0xffffffff) == 0x3f800000  # 1.0, not separately-rounded 0.0.
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_restoration_shared_reset_and_whole_list_protection():
    dut = SoC(TriXiumConfig(with_projection_restore=True, scratchpad_size=16*1024), n=1)
    def run():
        # Output writes target the second descriptor, not the executing one.
        bad = descriptor(dst=64, elements=1)
        yield from install_list(dut, [bad, [0]*16])
        for _ in range(50):
            yield
            assert not (yield dut.projection_restore.sp_wr_req)
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
        assert (yield dut.seq.projection_restore_error.status) == 64 << 8
        assert (yield dut.scratchpad.mem[2]) == 0
        yield from write(dut.engine_csr.soft_reset, 1)
        for _ in range(100):
            yield
        assert not (yield dut.seq.busy) and not (yield dut.seq.projection_restore_error.status)
        assert not (yield dut.projection_restore.busy) and not (yield dut.engine_csr.rst_busy)
        # Same client and sequencer reuse successfully after common domain reset.
        yield dut.scratchpad.mem[0x800//32].eq(pack([1, -1] + [0x7fffffff]*6))
        yield dut.scratchpad.mem[0x700//32].eq((-149) & 0xffffffff)
        yield from install_list(dut, [descriptor(elements=2, adjustment=0)])
        for _ in range(100):
            yield
            assert not (yield dut.seq.projection_restore_error.status)
            if not (yield dut.seq.busy):
                break
        else:
            raise AssertionError('restoration failed after common reset')
        assert (yield dut.seq.status.fields.dcount) == 1
        assert (yield dut.scratchpad.mem[0x1000//32]) & ((1 << 64)-1) == expected([1, -1], -149)[0]
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})

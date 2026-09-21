# SPDX-License-Identifier: BSD-2-Clause
"""Strict optional embedding descriptor ABI and shared-cluster ownership."""
import pytest
from migen import Memory, Module, run_simulation

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.gateware.sequencer import BonsaiSequencer
from trixium.model.embedding import scratchpad_embedding_descriptor
from test_projection_restore_sp import pack
from test_integration import write


def descriptor(**changes):
    fields = dict(src=0x800, signs=0x900, dst=0x1000, groups=8)
    fields.update(changes)
    return scratchpad_embedding_descriptor(**fields)


def fixture(descriptors):
    dut = Module()
    dut.submodules.seq = seq = BonsaiSequencer(with_embedding_restore=True)
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


def test_embedding_descriptor_and_configuration():
    assert descriptor() == [9, 0x800, 0x900, 0x1000, 8] + [0]*11
    for changes in ({'src': 1}, {'signs': -32}, {'dst': 1 << 24}, {'groups': 0},
                    {'groups': 7}, {'groups': 520}, {'groups': True}, {'src': False}):
        with pytest.raises(ValueError):
            descriptor(**changes)
    assert descriptor(groups=512)[4] == 512  # Actual hardware limit checked separately.
    for options in ({'with_vecunit': False}, {'scratchpad_size': 512*1024}):
        with pytest.raises(ValueError):
            BonsaiClusterConfig(with_embedding_restore=True, **options)
    for size in (0, 31, 33, 256*1024+32):
        with pytest.raises(ValueError):
            BonsaiSequencer(with_embedding_restore=True, scratchpad_size=size)
    assert not hasattr(BonsaiSequencer(), 'embedding_restore_abi')
    assert not BonsaiClusterConfig().with_embedding_restore


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (0x3fe0, 1)])
def test_embedding_only_profile_list_bounds_before_fetch(base, count):
    dut = BonsaiSequencer(with_embedding_restore=True, scratchpad_size=16*1024)
    def run():
        yield from write(dut.base, base)
        yield from write(dut.count, count)
        yield from write(dut.start, 1)
        for _ in range(5):
            assert (yield dut.busy) and (yield dut.embedding_restore_error.status) == 1
            assert not (yield dut.sp_active) and not (yield dut.embedding_restore_cmd.valid)
            yield
    run_simulation(dut, run())


@pytest.mark.parametrize('field,value', [(0, 9 | 1 << 9), (0, 9 | 1 << 4), (0, 8), (0, 7),
    (1, 1 << 24), (2, 1 << 24), (3, 1 << 24), (4, 65536)] + [(field, 1) for field in range(5, 16)])
def test_embedding_reserved_fields_rejected_before_dispatch(field, value):
    words = descriptor()
    words[field] = value
    dut = fixture([words])
    def run():
        yield from start(dut.seq, 1)
        for _ in range(30):
            yield
            assert not (yield dut.seq.embedding_restore_cmd.valid)
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
        assert (yield dut.seq.embedding_restore_error.status) == 2
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 4, 8, 0x1000, 0x2000])
def test_embedding_waits_nowait_engine_stable_dispatch_and_failure_wins(failure):
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 7, 0, 0, 0, 0x800] + [0]*6
    dut = fixture([fetch, descriptor()])
    seq, cmd = dut.seq, dut.seq.embedding_restore_cmd
    def run():
        assert (yield seq.embedding_restore_abi.status) == 0x10000
        assert (yield seq.embedding_restore_limits.status) == 64 | (16 << 16)
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
            raise AssertionError('embedding did not dispatch after NOWAIT drained')
        for _ in range(5):
            assert (yield cmd.valid) and (yield seq.status.fields.dcount) == 1
            for name, value in dict(src=0x800, signs=0x900, dst=0x1000, groups=8,
                                    protect_start=0, protect_end=128).items():
                assert (yield getattr(cmd, name)) == value
            yield
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        for _ in range(5):
            yield
            assert (yield seq.busy) and (yield seq.status.fields.dcount) == 1
        yield seq.embedding_restore_failed.eq(failure)
        yield seq.embedding_restore_done.eq(1)
        yield
        yield seq.embedding_restore_done.eq(0)
        for _ in range(10):
            yield
        assert (yield seq.embedding_restore_error.status) == failure << 8
        assert (yield seq.status.fields.dcount) == (1 if failure else 2)
        assert bool((yield seq.busy)) == bool(failure)
    run_simulation(dut, run())

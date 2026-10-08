#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from migen import C, Cat, Signal, run_simulation
from migen.fhdl import verilog

from trixium.core.core import TriXiumConfig
from trixium.model.hadamard import scratchpad_rotation_descriptor
from test_hadamard import dense, packed, unpacked
from test_integration import SoC, write


# Helpers ------------------------------------------------------------------------------------------

def install_list(dut, descriptors):
    for index, words in enumerate(descriptors):
        assert len(words) == 16
        yield dut.scratchpad.mem[index*2].eq(packed(words[:8], 32))
        yield dut.scratchpad.mem[index*2 + 1].eq(packed(words[8:], 32))
    yield from write(dut.seq.base, 0)
    yield from write(dut.seq.count, len(descriptors))
    yield from write(dut.seq.start, 1)


# Tests --------------------------------------------------------------------------------------------

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
    from trixium.core.sequencer import Sequencer
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
    from trixium.core.sequencer import Sequencer
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


def test_compiled_full_vector_rotation_through_shared_cluster_and_wishbone(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for compiled full-vector scratchpad check')
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
    rtl = tmp_path / 'hadamard_cluster.v'
    converted = verilog.convert(dut, ios=set(signals.values()), name='hadamard_cluster')
    rtl.write_text(converted.main_source)
    for name, content in converted.data_files.items():
        (tmp_path / name).write_text(content)
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/hadamard_cluster.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'hadamard_cluster',
                            str(rtl), str(harness.resolve())], capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vhadamard_cluster')], cwd=tmp_path,
                             capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
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

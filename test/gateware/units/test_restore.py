#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/restore.py: I32ToFp32Scaled, ProjectionRestoreScratchpad and the restoration
lists on the shared cluster."""

import math
import random
import shutil
import struct
import subprocess

import pytest

from migen import *

from litex.gen import LiteXModule

from gateware.units.restore import I32ToFp32Scaled, ProjectionRestoreScratchpad
from gateware.core.scratchpad import Scratchpad
from gateware.core.core import TriXiumConfig
from gateware.core.sequencer import Sequencer
from model.activation import scratchpad_restore_descriptor
from model import asm

from common import COMMON, RTL
from common.cluster import SoC, write, install_list
from common.verilator import Simulator

# I32ToFp32Scaled ----------------------------------------------------------------------------------
# IEEE binary32 scalar oracle, independent of the RTL normalization/shifts.

def bits(value):
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return 0xff800000 if value < 0 else 0x7f800000


def from_bits(value):
    return struct.unpack('<f', struct.pack('<I', value))[0]


def signed(value):
    return struct.unpack('<i', struct.pack('<I', value & 0xffffffff))[0]


def expected(words, exponent):
    bad = not -149 <= exponent <= 127
    error = int(bad) << len(words)
    data = 0
    for lane, word in enumerate(words):
        if bad:
            continue
        converted = from_bits(bits(signed(word)))
        restored = bits(converted * from_bits(bits(math.ldexp(1.0, exponent))))
        if restored & 0x7fffffff == 0x7f800000:
            error |= 1 << lane
            restored = 0
        data |= restored << (32*lane)
    return data, error


def vectors(lanes=8):
    rng = random.Random(271525)
    result = []
    # Every exponent encoding, signed INT_MIN, carry at INT_MAX, zero, odd/even
    # tie conversion and the smallest subnormal / normal result boundaries.
    edges = [0, 1, -1, -(1 << 31), (1 << 31) - 1, (1 << 24) + 1,
             (1 << 24) + 3, -((1 << 24) + 3)]
    for exponent in range(-512, 512):
        for start in range(0, 8, lanes):
            word = [edges[(start + i) % 8] & 0xffffffff for i in range(lanes)]
            result.append((word, exponent, exponent % 17 == 0, exponent % 17 == 16))
    for power in range(24, 31):
        quantum = 1 << (power - 23)
        for base in ((1 << power), (1 << (power + 1)) - 2*quantum):
            for offset in range(-3, 4):
                for parity in range(2):
                    tie = base + parity*quantum + quantum//2
                    value = tie + offset
                    for exponent in (-149, -148, -127, -126, -31, 0, 96, 97, 103, 127):
                        values = [signed(value if i % 2 == 0 else -value) & 0xffffffff
                                  for i in range(lanes)]
                        result.append((values, exponent, offset == -3, offset == 3))
    for i in range(700):
        result.append(([rng.getrandbits(32) for _ in range(lanes)],
                       rng.randrange(-149, 128), i % 11 == 0, i % 11 == 10))
    return result


def exercise(dut, inputs, stalls=True, cancels=()):
    rng = random.Random(317)
    pending, held = [], None
    index, retired, discarded, offering = 0, 0, 0, False
    for cycle in range(20*len(inputs) + 100):
        cancel = cycle in cancels
        yield dut.cancel.eq(cancel)
        yield dut.source.ready.eq(not stalls or rng.randrange(3) != 0)
        if index < len(inputs):
            words, exponent, first, last = inputs[index]
            offering = offering or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offering)
            yield dut.sink.data.eq(sum(word << (32*i) for i, word in enumerate(words)))
            yield dut.sink.exponent.eq(exponent & 1023)
            yield dut.sink.first.eq(first)
            yield dut.sink.last.eq(last)
        else:
            yield dut.sink.valid.eq(0)
        yield
        if cancel:
            assert not (yield dut.sink.ready) and not (yield dut.source.valid)
            discarded += len(pending)
            pending.clear()
            held = None
            continue
        payload = ((yield dut.source.data), (yield dut.source.error),
                   (yield dut.source.exponent), (yield dut.source.first), (yield dut.source.last))
        if held is not None:
            assert (yield dut.source.valid) and payload == held
        held = payload if (yield dut.source.valid) and not (yield dut.source.ready) else None
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            pending.append((*expected(words, exponent), exponent & 1023, first, last))
            index += 1
            offering = False
        if (yield dut.source.valid) and (yield dut.source.ready):
            assert pending, 'stale result after cancellation'
            assert payload == pending.pop(0), (cycle, index, payload)
            retired += 1
        if index == len(inputs) and not pending:
            assert retired + discarded == len(inputs)
            yield dut.sink.valid.eq(0)
            for _ in range(6):
                yield
                assert not (yield dut.source.valid)
            return cycle + 1
    raise AssertionError('projection restoration did not drain')


def test_projection_restore_all_exponents_rounding_boundaries():
    dut = I32ToFp32Scaled()
    run_simulation(dut, exercise(dut, vectors(), stalls=False))


@pytest.mark.parametrize('lanes', [1, 2, 8, 16])
def test_projection_restore_elastic_cancel(lanes):
    rng = random.Random(948)
    inputs = [([rng.getrandbits(32) for _ in range(lanes)], rng.randrange(-160, 140),
               i % 13 == 0, i % 13 == 12) for i in range(400)]
    dut = I32ToFp32Scaled(lanes)
    run_simulation(dut, exercise(dut, inputs, cancels=(1, 2, 11, 78, 143, 301)))


def test_projection_restore_invalid_lanes():
    for lanes in (0, 3, 32):
        with pytest.raises(ValueError):
            I32ToFp32Scaled(lanes)


def test_projection_restore_reference_operation_order():
    # This observable difference forbids independently restoring K-segments
    # and then adding floats. Reduction is first, including int32 wrap.
    a, b = (1 << 24) + 1, -(1 << 24)
    restored, error = expected([a + b], 0)
    assert not error and restored == bits(1.0)
    assert bits(from_bits(bits(a)) + from_bits(bits(b))) == bits(0.0)
    assert expected([signed(0x7fffffff + 1)], 0) == (bits(-(1 << 31)), 0)


def test_projection_restore_compiled_float_oracle():
    dut = I32ToFp32Scaled()
    signals = {'cancel': dut.cancel}
    for direction, endpoint in (('in', dut.sink), ('out', dut.source)):
        for name in ('data', 'exponent', 'first', 'last', 'valid', 'ready'):
            signals[f'{direction}_{name}'] = getattr(endpoint, name)
    signals['out_error'] = dut.source.error
    for name, signal in signals.items():
        signal.name_override = name
    sim = Simulator(dut, ios=signals, name='projection_restore',
                    driver=(RTL / 'projection_restore.cpp').read_text())
    result = sim.run(timeout=60)
    assert 'PROJECTION_RESTORE_PASS' in result.stdout
    print(result.stdout)


@pytest.mark.parametrize('abi', ['ilp32f', 'ilp32'])
def test_projection_restore_reference_against_rv32(tmp_path, abi):
    cc, qemu = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('qemu-riscv32')
    if not cc or not qemu:
        pytest.skip('target restoration check needs RV32 compiler and qemu-riscv32')
    source = COMMON / 'rv32_projection_restore_oracle.c'
    exe = tmp_path / 'restore.elf'
    march = 'rv32imf' if abi == 'ilp32f' else 'rv32im'
    compiled = subprocess.run([cc, f'-march={march}', f'-mabi={abi}', '-O2', '-ffp-contract=off',
                               '-msmall-data-limit=0', '-nostartfiles', '-nostdlib',
                               '-Wl,--no-relax,-e,_start', str(source), '-lm', '-lgcc', '-o', str(exe)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    inputs = [(word, exponent) for words, exponent, _, _ in vectors()
              if -149 <= exponent <= 127 for word in words]
    output = subprocess.run([qemu, str(exe)],
                            input=b''.join(struct.pack('<Ii', word, exponent) for word, exponent in inputs),
                            check=True, capture_output=True, timeout=60).stdout
    assert len(output) == 4*len(inputs)
    for (word, exponent), (actual,) in zip(inputs, struct.iter_unpack('<I', output)):
        value, error = expected([word], exponent)
        if error:
            assert actual == (0xff800000 if word & 0x80000000 else 0x7f800000)
        else:
            assert actual == value, (hex(word), exponent, hex(actual), hex(value))


# ProjectionRestoreScratchpad ----------------------------------------------------------------------

def pack(words):
    return sum((word & 0xffffffff) << (32*i) for i, word in enumerate(words))


class RestoreSP(LiteXModule):
    def __init__(self, size=8192):
        self.sp = Scratchpad(size=size)
        self.unit = ProjectionRestoreScratchpad(size)
        self.grant = Signal(reset=1)
        self.comb += [self.sp.fill.adr.eq(self.unit.sp_rd_adr), self.unit.sp_rd_dat.eq(self.sp.fill.dat_r),
                     self.unit.sp_wr_gnt.eq(self.grant & self.unit.sp_wr_req),
                     self.sp.gather.adr.eq(self.unit.sp_wr_adr), self.sp.gather.dat_w.eq(self.unit.sp_wr_dat),
                     self.sp.gather.we.eq(Mux(self.grant & self.unit.sp_wr_req, self.unit.sp_wr_we, 0))]


def submit(dut, **changed):
    fields = dict(src=0x400, dst=0x1000, exponents=0x1fe0, elements=65, adjustment=0,
                  protect_start=0, protect_end=64)
    fields.update(changed)
    for name, value in fields.items():
        yield getattr(dut.unit.cmd, name).eq(value)
    yield dut.unit.cmd.valid.eq(1)
    yield
    assert (yield dut.unit.cmd.ready)
    yield dut.unit.cmd.valid.eq(0)
    yield


def valid(dut, words, *, exponent=0, adjustment=0, inplace=False, dst=None):
    src = 0x400
    dst = src if inplace else (0x1000 if dst is None else dst)
    beats = (len(words) + 7)//8
    source_tail = [0x7fffffff]*(8*beats - len(words))
    output_sentinel = pack([0xdeadbeef]*8)
    for beat in range(beats):
        yield dut.sp.mem[dst//32 + beat].eq(output_sentinel)
    for beat in range(beats):
        yield dut.sp.mem[src//32 + beat].eq(pack((words + source_tail)[8*beat:8*beat+8]))
    yield dut.sp.mem[0x1fe0//32].eq(pack([exponent] + [0xbadf00d]*7))
    yield dut.sp.mem[dst//32 - 1].eq(output_sentinel)
    if dst + 32*beats != 0x1fe0:
        yield dut.sp.mem[dst//32 + beats].eq(output_sentinel)
    yield from submit(dut, src=src, dst=dst, elements=len(words), adjustment=adjustment)
    rng, held = random.Random(91), None
    for cycle in range(4000):
        yield dut.grant.eq(cycle % 83 not in range(10, 42) and rng.randrange(3) != 0)
        yield dut.unit.cmd.elements.eq(0)
        yield dut.unit.cmd.adjustment.eq(0x7fffffff)
        yield dut.unit.cmd.dst.eq(0xffffff)
        yield
        assert not (yield dut.unit.error), (cycle, (yield dut.unit.error))
        payload = ((yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat))
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        if (yield dut.unit.done):
            break
    else:
        raise AssertionError('restore client timed out')
    result, error = expected(words, exponent + adjustment)
    assert not error
    for beat in range(beats):
        data = yield dut.sp.mem[dst//32 + beat]
        for lane in range(8):
            i = beat*8 + lane
            value = (data >> (32*lane)) & 0xffffffff
            assert value == ((result >> (32*i)) & 0xffffffff if i < len(words)
                             else (0x7fffffff if inplace else 0xdeadbeef)), (i, hex(value))
    assert (yield dut.unit.read_beats) == beats + 1
    assert (yield dut.unit.output_beats) == beats
    assert (yield dut.unit.exponent) == exponent + adjustment
    assert (yield dut.sp.mem[0x1fe0//32]) == pack([exponent] + [0xbadf00d]*7)
    assert (yield dut.sp.mem[dst//32 - 1]) == output_sentinel
    if dst + 32*beats != 0x1fe0:
        assert (yield dut.sp.mem[dst//32 + beats]) == output_sentinel
    yield


@pytest.mark.parametrize('inplace', [False, True])
def test_restore_sp_stalls_tail_scales_and_reuse(inplace):
    dut = RestoreSP()
    rng = random.Random(927)
    def run():
        for count, exponent, adjustment in ((65, 3, -17), (128, -149, 0), (1, 127, 0),
                                            (8, 0, 0), (3, -127, -22), (257, -65, 32)):
            words = [rng.getrandbits(32) for _ in range(count)] if exponent != 127 else [0]
            yield from valid(dut, words, exponent=exponent, adjustment=adjustment, inplace=inplace)
    run_simulation(dut, run())


def test_restore_sp_exact_end_and_readonly_metadata_alias():
    dut = RestoreSP()
    def run():
        yield from valid(dut, [1, -1, 7], exponent=-149, dst=0x1fc0)
        # Read-only source can include its own metadata beat. Only lane zero
        # participates, giving x=-149 and scale 2**-149; no source writes.
        yield dut.sp.mem[0x1fe0//32].eq(pack([-149] + [0x7fffffff]*7))
        yield dut.grant.eq(1)
        yield from submit(dut, src=0x1fe0, elements=1)
        for _ in range(60):
            yield
            assert not (yield dut.unit.error)
            if (yield dut.unit.done):
                break
        else:
            raise AssertionError('read-only alias command failed to finish')
        assert (yield dut.sp.mem[0x1000//32]) & 0xffffffff == expected([0xffffff6b], -149)[0]
    run_simulation(dut, run())


@pytest.mark.parametrize('changed,mask', [
    ({'elements': 0}, 1), ({'src': 1}, 1), ({'dst': 1}, 1), ({'exponents': 1}, 1),
    ({'protect_start': 65, 'protect_end': 64}, 1),
    ({'src': 0xffffe0}, 2), ({'dst': 0xffffe0}, 2), ({'exponents': 0xffffe0}, 2),
    ({'elements': 65535}, 2), ({'protect_end': 8193}, 2),
    ({'src': 0x1fe0, 'elements': 9}, 2), ({'dst': 0x1fe0, 'elements': 9}, 2),
    ({'dst': 0x420}, 4), ({'dst': 0x3e0}, 4), ({'dst': 0x1fe0, 'elements': 1}, 4),
    ({'dst': 0}, 64), ({'protect_start': 0x1100, 'protect_end': 0x1101}, 64),
])
def test_restore_sp_rejects_before_memory_access(changed, mask):
    dut = RestoreSP()
    def run():
        yield from submit(dut, **changed)
        for _ in range(20):
            yield
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            assert not (yield dut.unit.done) and (yield dut.unit.busy)
        assert (yield dut.unit.error) & mask
        assert not (yield dut.unit.read_beats) and not (yield dut.unit.output_beats)
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.error) and not (yield dut.unit.busy)
        yield from valid(dut, [1, -1, 0], exponent=-149)
    run_simulation(dut, run())


@pytest.mark.parametrize('metadata,adjustment', [(-150, 0), (128, 0), (0x40000000, 0),
    (0x7fffffff, 1), (-0x80000000, -1), (0x7fffffff, 0x7fffffff), (-0x80000000, -0x80000000)])
def test_restore_sp_full_width_exponent_validation(metadata, adjustment):
    dut = RestoreSP()
    def run():
        yield dut.sp.mem[0x1fe0//32].eq(pack([metadata]*8))
        yield from submit(dut, adjustment=adjustment)
        for _ in range(30):
            yield
            assert not (yield dut.unit.sp_wr_req) and not (yield dut.unit.done)
        assert (yield dut.unit.error) == 8 and (yield dut.unit.busy)
        assert (yield dut.unit.read_beats) == 1 and not (yield dut.unit.output_beats)
    run_simulation(dut, run())


def test_restore_sp_overflow_invalidates_prefix_and_no_success():
    dut = RestoreSP()
    def run():
        for i in range(16):
            yield dut.sp.mem[0x400//32 + i].eq(pack([0]*8 if i != 14 else [0x7fffffff]*8))
            yield dut.sp.mem[0x1000//32 + i].eq(pack([0xdeadbeef]*8))
        yield dut.sp.mem[0x1fe0//32].eq(127)
        yield from submit(dut, elements=128)
        for _ in range(100):
            yield
            assert not (yield dut.unit.done)
        assert (yield dut.unit.error) == 16 and (yield dut.unit.busy)
        assert 0 < (yield dut.unit.output_beats) < 16
        assert (yield dut.sp.mem[0x1000//32 + 14]) == pack([0xdeadbeef]*8)
        assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
    run_simulation(dut, run())


@pytest.mark.parametrize('delay', [0, 1, 2, 3, 5, 8, 12, 24, 50])
def test_restore_sp_cancel_pipeline_and_restart(delay):
    dut = RestoreSP()
    def run():
        yield dut.sp.mem[0x1fe0//32].eq(0)
        yield dut.grant.eq(0)
        yield from submit(dut, elements=128)
        for _ in range(delay):
            yield
        yield dut.unit.cancel.eq(1)
        yield
        assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.busy) and not (yield dut.unit.error)
        yield from valid(dut, [-(1 << 31), (1 << 31) - 1, 0], exponent=-31)
    run_simulation(dut, run())


def test_restore_sp_invalid_physical_sizes():
    for size in (0, 31, 33, 256*1024 + 32):
        with pytest.raises(ValueError):
            ProjectionRestoreScratchpad(size)


def test_restore_sp_compiled_memory_tails_inplace_reset_and_extremes():
    dut = ProjectionRestoreScratchpad()
    signals = {name: getattr(dut, name) for name in (
        'cancel', 'busy', 'done', 'error', 'read_beats', 'output_beats', 'exponent',
        'sp_rd_req', 'sp_rd_adr', 'sp_rd_dat', 'sp_wr_req', 'sp_wr_adr', 'sp_wr_dat', 'sp_wr_we', 'sp_wr_gnt')}
    for name, _ in dut.cmd.description.payload_layout:
        signals['cmd_' + name] = getattr(dut.cmd, name)
    signals.update(cmd_valid=dut.cmd.valid, cmd_ready=dut.cmd.ready)
    for name, signal in signals.items():
        signal.name_override = name
    sim = Simulator(dut, ios=signals, name='projection_restore_sp',
                    driver=(RTL / 'projection_restore_sp.cpp').read_text())
    result = sim.run(timeout=60)
    assert 'PROJECTION_RESTORE_SP_PASS' in result.stdout
    print(result.stdout)


# Sequencer Lists on the Shared Cluster ------------------------------------------------------------

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

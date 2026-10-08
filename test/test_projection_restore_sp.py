# SPDX-License-Identifier: BSD-2-Clause
import random
from pathlib import Path
import shutil
import subprocess

import pytest
from migen import *
from migen.fhdl import verilog
from litex.gen import LiteXModule

from trixium.units.restore import BonsaiProjectionRestoreScratchpad
from trixium.core.scratchpad import BonsaiScratchpad
from test_projection_restore import expected


def pack(words):
    return sum((word & 0xffffffff) << (32*i) for i, word in enumerate(words))


class RestoreSP(LiteXModule):
    def __init__(self, size=8192):
        self.sp = BonsaiScratchpad(size=size)
        self.unit = BonsaiProjectionRestoreScratchpad(size)
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
            BonsaiProjectionRestoreScratchpad(size)


def test_restore_sp_compiled_memory_tails_inplace_reset_and_extremes(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for standalone compiled scratchpad restoration check')
    dut = BonsaiProjectionRestoreScratchpad()
    signals = {name: getattr(dut, name) for name in (
        'cancel', 'busy', 'done', 'error', 'read_beats', 'output_beats', 'exponent',
        'sp_rd_req', 'sp_rd_adr', 'sp_rd_dat', 'sp_wr_req', 'sp_wr_adr', 'sp_wr_dat', 'sp_wr_we', 'sp_wr_gnt')}
    for name, _ in dut.cmd.description.payload_layout:
        signals['cmd_' + name] = getattr(dut.cmd, name)
    signals.update(cmd_valid=dut.cmd.valid, cmd_ready=dut.cmd.ready)
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / 'projection_restore_sp.v'
    verilog.convert(dut, ios=set(signals.values()), name='projection_restore_sp').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/projection_restore_sp.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'projection_restore_sp',
                            str(rtl), str(harness.resolve())], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vprojection_restore_sp')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PROJECTION_RESTORE_SP_PASS' in result.stdout
    print(result.stdout)

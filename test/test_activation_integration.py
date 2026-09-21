# SPDX-License-Identifier: BSD-2-Clause
import math
from pathlib import Path
import shutil
import subprocess

import pytest
from migen import C, Cat, Memory, Module, Signal, run_simulation
from migen.fhdl import verilog

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.model.activation import scratchpad_quantize_descriptor, scratchpad_requantize_descriptor
from test_activation_sp import bits, pack, reference
from test_hadamard_integration import install_list
from test_integration import SoC, write


def test_activation_descriptor_and_configuration():
    assert scratchpad_quantize_descriptor(src=32, dst=64, exponents=96, elements=1024, blocks=17) == [4, 32, 64, 96, 1024, 17] + [0]*10
    for changes in ({'src': 1}, {'dst': -32}, {'exponents': 1 << 24}, {'elements': 0},
                    {'elements': 40}, {'elements': 65536}, {'blocks': 0}, {'blocks': 65536}):
        args = dict(src=32, dst=64, exponents=96, elements=1024)
        args.update(changes)
        with pytest.raises(ValueError):
            scratchpad_quantize_descriptor(**args)
    for options in ({'with_vecunit': False}, {'scratchpad_size': 512*1024}):
        with pytest.raises(ValueError):
            BonsaiClusterConfig(with_activation_quantize=True, **options)
    assert scratchpad_requantize_descriptor(src=32, dst=64, exponents=96, maxima=128,
        result_exponent=160, blocks=17) == [5, 32, 64, 96, 128, 160, 17] + [0]*9
    for changes in ({'src': 1}, {'dst': -32}, {'exponents': 1 << 24}, {'maxima': 127},
                    {'result_exponent': 1 << 24}, {'blocks': 0}, {'blocks': 65536}):
        args = dict(src=32, dst=64, exponents=96, maxima=128, result_exponent=160, blocks=17)
        args.update(changes)
        with pytest.raises(ValueError):
            scratchpad_requantize_descriptor(**args)


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (0x3fe0, 1)])
def test_quantization_only_profile_validates_full_list_before_fetch(base, count):
    from trixium.gateware.sequencer import BonsaiSequencer
    dut = BonsaiSequencer(with_activation_quantize=True, scratchpad_size=16*1024)
    def run():
        yield from write(dut.base, base)
        yield from write(dut.count, count)
        yield from write(dut.start, 1)
        for _ in range(5):
            assert (yield dut.busy) and (yield dut.activation_quantize_error.status) == 1
            assert not (yield dut.sp_active)
            yield
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 128])
def test_requantization_dispatch_without_hadamard_and_completion_publication(failure):
    from trixium.gateware.sequencer import BonsaiSequencer
    dut = Module()
    dut.submodules.seq = seq = BonsaiSequencer(with_activation_quantize=True)
    words = scratchpad_requantize_descriptor(src=0x10000, dst=0x8000, exponents=0x800,
        maxima=0x900, result_exponent=0xa00, blocks=17)
    mem = Memory(256, 2, init=[pack(words[:8], 32), pack(words[8:], 32)])
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    def run():
        assert (yield seq.activation_quantize_abi.status) == 0x10001
        yield from write(seq.base, 0)
        yield from write(seq.count, 1)
        yield from write(seq.start, 1)
        cmd = seq.activation_quantize_cmd
        for _ in range(50):
            if (yield cmd.valid):
                break
            yield
        else:
            raise AssertionError('unit 5 did not dispatch without Hadamard')
        expected_fields = dict(src=0x10000, dst=0x8000, exponents=0x800, maxima=0x900,
                               result_exponent=0xa00, blocks=17, elements=1024, requantize=1,
                               protect_start=0, protect_end=64)
        for _ in range(5):
            assert (yield cmd.valid) and not (yield seq.status.fields.dcount)
            for name, value in expected_fields.items():
                assert (yield getattr(cmd, name)) == value
            yield
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        for _ in range(5):
            yield
            assert (yield seq.busy) and not (yield seq.status.fields.dcount)
        yield seq.activation_quantize_done.eq(not failure)
        yield seq.activation_quantize_failed.eq(failure)
        for _ in range(10):
            yield
        assert (yield seq.activation_quantize_error.status) == failure << 8
        assert (yield seq.status.fields.dcount) == (0 if failure else 1)
        assert bool((yield seq.busy)) == bool(failure)
    run_simulation(dut, run())


@pytest.mark.parametrize('with_hadamard', [False, True])
def test_sequenced_fetch_nowait_quantize_store(with_hadamard):
    dut = SoC(BonsaiClusterConfig(with_activation_quantize=True, with_hadamard=with_hadamard,
                                 scratchpad_size=16*1024), n=1)
    arrays = [[bits(math.ldexp((i - 32)*0.5, e)) for i in range(64)] for e in (-7, 4)]
    raw = [word for a in arrays for word in a]
    expected = b''.join(reference(a)[0] for a in arrays)
    fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 16, 0, 0, 0, 0x1000, 0] + [0]*5
    quant = scratchpad_quantize_descriptor(src=0x1000, dst=0x800, exponents=0x700, elements=64, blocks=2)
    store = [1, 3, 0, 0, 0x1000, 4, 0, 0, 0x800, 0, 0] + [0]*5

    def run():
        for beat in range(16):
            yield dut.models[0].mem[0x100//32 + beat].eq(pack(raw[8*beat:8*(beat+1)], 32))
        yield from install_list(dut, [fetch, quant, store])
        accepted = False
        for _ in range(3000):
            unit = dut.activation_quantize
            if (yield unit.cmd.valid) and (yield unit.cmd.ready):
                assert not (yield dut.engine.sp_busy)
                assert (yield dut.engine_csr.done_count.status) == 1
                accepted = True
            assert not (yield dut.seq.activation_quantize_error.status)
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('sequenced conversion did not complete')
        assert accepted and (yield dut.seq.status.fields.dcount) == 3
        assert (yield dut.engine_csr.done_count.status) == 2
        for beat in range(4):
            word = int.from_bytes(expected[32*beat:32*(beat+1)], 'little')
            assert (yield dut.scratchpad.mem[0x800//32 + beat]) == word
            assert (yield dut.models[0].mem[0x1000//32 + beat]) == word
        exponents = yield dut.scratchpad.mem[0x700//32]
        assert exponents & ((1 << 64) - 1) == pack([reference(a)[1] for a in arrays], 32)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


@pytest.mark.parametrize('native', [False, True])
def test_compiled_resident_conversion_rotation_full_ffn_and_reset(tmp_path, native):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for full resident conversion/rotation check')
    options = {}
    if native:
        from test_native_reader import region
        options = dict(native_regions=(region(7, size=256*1024),), native_endpoints=(0,))
    dut = SoC(BonsaiClusterConfig(with_hadamard=True, with_activation_quantize=True, **options), n=1)
    if native:
        import numpy as np
        from trixium.model.qformats import ptq1_0_pack
        # Three complete compact rows. The two 68-group segments retain a
        # whole-row stride, and consume the resident Q8 slices without host IO.
        raw = bytearray(256*1024)
        for row in range(3):
            codes = np.array([(i*7 + i//13 + row) % 3 for i in range(17408)], dtype=np.uint8)
            payload = ptq1_0_pack(np.ones(136, dtype=np.float16), codes.reshape(136, 128)).tobytes()
            base = 0x10000 + row*136*28
            raw[base:base + len(payload)] = payload
        dut.models[0].mem.init = [int.from_bytes(raw[i:i+32], 'little') for i in range(0, len(raw), 32)]
    bus = dut.scratchpad.add_wishbone_port()
    status = Signal(24)
    dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
    signals = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
    signals.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
        seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re, seq_status=status,
        seq_error=dut.seq.activation_quantize_error.status, had_error=dut.seq.hadamard_error.status,
        reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
        reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
        quant_blocks=dut.activation_quantize.blocks_done, quant_reads=dut.activation_quantize.read_beats,
        quant_writes=dut.activation_quantize.output_beats)
    if native:
        signals.update(native_error=dut.seq.native_error.status, engine_done=dut.engine_csr.done_count.status,
                       engine_busy=dut.engine.busy)
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / 'activation_cluster.v'
    converted = verilog.convert(dut, ios=set(signals.values()), name='activation_cluster')
    rtl.write_text(converted.main_source)
    for name, content in converted.data_files.items():
        (tmp_path / name).write_text(content)
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/activation_cluster.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'activation_cluster',
                            str(rtl), str(harness.resolve())] + (['-CFLAGS', '-DNATIVE_CHAIN'] if native else []),
                           capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vactivation_cluster')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '$readmem file not found' not in result.stdout
    assert 'ACTIVATION_CLUSTER_PASS' in result.stdout
    print(result.stdout)

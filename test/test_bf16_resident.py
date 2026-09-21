# SPDX-License-Identifier: BSD-2-Clause
"""Prepared BF16 -> real shared quantize/native/restore chain, no CPU between stages."""
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import Cat, C, Signal
from migen.fhdl import verilog

from trixium import layout
from trixium.gateware.integration import BonsaiClusterConfig
from trixium.model import bitexact as bx, bons4
from trixium.model.gguf_reader import GGUFReader
from trixium.model.projection import prepare_bf16_projection, resident_reference_q8_descriptors
from test_activation_sp import reference
from test_integration import SoC
from test_native_reader import region
from test_projection import bf16_bytes, legacy_spec


@pytest.mark.parametrize('local', [False, True])
def test_compiled_resident_beta_alpha_shared_unrotated_activation(tmp_path, local):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for resident beta/alpha qualification')
    gguf_path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    if local and not gguf_path.exists():
        pytest.skip('optional local 27B GGUF is not installed')
    gguf = GGUFReader(gguf_path) if local else None
    rng = np.random.default_rng(382)
    names = ['blk.0.ssm_beta.weight', 'blk.0.ssm_alpha.weight']
    sources, refs = [], []
    for name, role in zip(names, [layout.ROLE_SSM_BETA, layout.ROLE_SSM_ALPHA]):
        raw = (bytes(gguf.tensor_bytes(name)) if local else
               bf16_bytes(rng.standard_normal((48, 5120)).astype(np.float32) * .006))
        sources.append(bons4.TensorSource(name, bons4.Encoding.BF16, (48, 5120), raw, role, 0))
        refs.append(legacy_spec(raw, (48, 5120), name))
    image = tmp_path / 'projection-subset.bons'
    bons4.write(image, sources, {'qualification_subset': True})
    reader = bons4.Reader(image)
    assert reader.verify()
    prepared = [prepare_bf16_projection(reader, name) for name in names]

    dut = SoC(BonsaiClusterConfig(with_activation_quantize=True, with_projection_restore=True,
        native_regions=(region(7, size=256*1024), region(42, base=1 << 32, size=256*1024, endpoint=1)),
        native_endpoints=(0, 1)), n=2)
    descs = []
    for i, p in enumerate(prepared):
        descs += resident_reference_q8_descriptors(p.execution,
            pipe_locations=[(7, i*0x20000+256), (42, i*0x20000+256)],
            fp32_src=None if i else 0x10000, q8_src=0x4000, exponents=0x800,
            int32_dst=0x6000+i*256, fp32_dst=0x7000+i*256)
    assert [d[0] for d in descs] == [4, 6, 6, 7, 8, 6, 6, 7, 8]
    for pipe in range(2):
        rawmem = bytearray(256*1024)
        for i, p in enumerate(prepared):
            payload = p.execution.read_rows(pipe*24, 24)
            offset = i*0x20000+256
            rawmem[offset:offset+len(payload)] = payload
        dut.models[pipe].mem.init = [int.from_bytes(rawmem[i:i+32], 'little') for i in range(0, len(rawmem), 32)]

    # One file contains complete initial writes and independent expected reads
    # for three activations. Checks include unchanged source/list, Q8, metadata,
    # all 96 native int32 results, all 96 FP32 results and destination guards.
    vectors = bytearray(struct.pack('<II', 0x42465231, 3))
    for profile in range(3):
        x = rng.standard_normal(5120).astype(np.float32)
        if profile == 1:
            x *= np.exp2(np.arange(5120, dtype=np.int32) % 9 - 4).astype(np.float32)
        if profile == 2:
            x.fill(0)
        words = x.view(np.uint32).tolist()
        x8, exponent = reference(words)
        writes = {0x10000+4*i: word for i, word in enumerate(words)}
        writes.update({4*i: word for i, word in enumerate(w for d in descs for w in d)})
        checks = writes.copy()
        writes.update({0x800+4*i: 0xdeadbeef for i in range(8)})
        checks.update({0x800+4*i: 0xdeadbeef for i in range(1, 8)})
        checks[0x800] = exponent & 0xffffffff
        checks.update({0x4000+4*i: word for i, word in enumerate(struct.unpack('<1280I', x8))})
        guards = [0x3ffc, 0x5400]
        for i, ref in enumerate(refs):
            result = bx.matvec_q8(ref.scales, ref.values, np.frombuffer(x8, np.int8), ref.bias, ref.rshift)
            scale = np.float32(2.0**(exponent+ref.rshift-ref.bias))
            restored = (result.astype(np.float32) * scale).astype(np.float32).view(np.uint32)
            for base, output in ((0x6000+i*256, result.view(np.uint32)), (0x7000+i*256, restored)):
                guards += [base-4, base+48*4]
                writes.update({base+4*r: 0xdeadbeef for r in range(48)})
                checks.update({base+4*r: int(word) for r, word in enumerate(output)})
        writes.update({address: 0xdeadbeef for address in guards})
        checks.update({address: 0xdeadbeef for address in guards})
        vectors += struct.pack('<III', len(writes), len(checks), len(descs))
        vectors += b''.join(struct.pack('<II', addr, word) for addr, word in writes.items())
        vectors += b''.join(struct.pack('<II', addr, word) for addr, word in checks.items())
    (tmp_path / 'vectors.bin').write_bytes(vectors)

    bus = dut.scratchpad.add_wishbone_port()
    status = Signal(24)
    dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
    signals = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
    signals.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
        seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re, seq_status=status,
        seq_error=dut.seq.activation_quantize_error.status, native_error=dut.seq.native_error.status,
        restore_error=dut.seq.projection_restore_error.status, restore_busy=dut.projection_restore.busy,
        reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
        reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
        engine_busy=dut.engine.busy, engine_done=dut.engine_csr.done_count.status,
        quant_blocks=dut.activation_quantize.blocks_done, quant_reads=dut.activation_quantize.read_beats,
        quant_writes=dut.activation_quantize.output_beats)
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / 'bf16_resident.v'
    converted = verilog.convert(dut, ios=set(signals.values()), name='bf16_resident')
    rtl.write_text(converted.main_source)
    for name, content in converted.data_files.items():
        (tmp_path / name).write_text(content)
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/bf16_resident.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
        '--Mdir', str(obj), '--top-module', 'bf16_resident', str(rtl), str(harness.resolve())],
        capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stdout[-6000:] + built.stderr[-6000:]
    run = subprocess.run([str(obj / 'Vbf16_resident')], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert '$readmem file not found' not in run.stdout
    assert 'BF16_RESIDENT_PASS profiles=3 quantizations_per_pair=1 reset_reuse=2' in run.stdout
    print(('local_27b' if local else 'synthetic') + '\n' + run.stdout)

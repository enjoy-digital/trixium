# SPDX-License-Identifier: BSD-2-Clause
"""Real shared-cluster fetch -> embedding -> quantize -> store/readback list."""
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import Cat, C, Signal
from migen.fhdl import verilog

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.model import qformats
from trixium.model.activation import scratchpad_quantize_descriptor
from trixium.model.embedding import scratchpad_embedding_descriptor
from trixium.model.native_sequence import region_fetch_descriptor
from trixium.model.gguf_reader import GGUFReader
from trixium.model.golden import config_from_gguf
from test_activation_sp import reference
from test_embedding_restore import example, golden
from test_hadamard import packed
from test_integration import SoC
from test_native_reader import region


@pytest.mark.parametrize('companions,region_fetch', [(False, False), (True, False), (True, True)])
def test_compiled_embedding_resident_chain(tmp_path, companions, region_fetch):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for resident embedding chain')
    profiles = [example(40, span=2), example(40, seed=412, span=2)]
    profiles[1][1].fill(1)  # Zero row still traverses every resident stage.
    source = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    local = source.exists()
    if local:
        reader = GGUFReader(source)
        scales, codes = qformats.ptq1_0_unpack(reader.tensor_bytes('token_embd.weight')[11751*1120:11752*1120])
        profiles.append((scales, codes.reshape(-1), np.asarray(config_from_gguf(reader).hadamard_signs[5120])))
    else:
        profiles.append(example(40, seed=943, span=4))
    options = (dict(native_regions=(region(42, size=1 << 33) if region_fetch else region(7, size=256*1024),),
                    native_endpoints=(0,)) if companions else {})
    dut = SoC(BonsaiClusterConfig(with_embedding_restore=True, with_activation_quantize=True,
        with_hadamard=companions, with_projection_restore=companions,
        with_region_fetch=region_fetch, **options), n=1)
    memory = bytearray(256*1024)
    vectors = bytearray(struct.pack('<III', 0x45525331, len(profiles), int(region_fetch)))
    for profile, (scales, codes, signs) in enumerate(profiles):
        raw = qformats.ptq1_0_pack(scales, codes.reshape(-1, 128)).tobytes()
        sign_bytes = b''.join(packed(signs[i:i+128] < 0, 1).to_bytes(16, 'little') for i in range(0, 5120, 128))
        raw_base, sign_base = 0x1000+profile*0x2000, 0x2000+profile*0x2000
        memory[raw_base:raw_base+len(raw)] = raw
        memory[sign_base:sign_base+len(sign_bytes)] = sign_bytes
        _, embedding_e, fp = golden(scales, codes, signs)
        q8, exponent = reference(fp.view(np.uint32).tolist())
        def fetch(base, length, dst, nowait=False):
            if region_fetch:
                return region_fetch_descriptor(region=42, offset=base+((1 << 32) if nowait else 0),
                                               length=length//32, dst=dst)
            return [1 | (int(nowait) << 9), 2, 0, 0, base, length//32, 0, 0, 0, dst] + [0]*6
        def store(base, length, src):
            return [1, 3, 0, 0, base, length//32, 0, 0, src, 0] + [0]*6
        descriptors = [fetch(raw_base, len(raw), 0x4000, True),
            fetch(sign_base, len(sign_bytes), 0x5000, True),
            scratchpad_embedding_descriptor(src=0x4000, signs=0x5000, dst=0x8000, groups=40),
            scratchpad_quantize_descriptor(src=0x8000, dst=0xe000, exponents=0x1000, elements=5120),
            store(0x10000, 20480, 0x8000), store(0x20000, 5120, 0xe000),
            fetch(0x10000, 20480, 0x18000), fetch(0x20000, 5120, 0x20000)]
        writes = {4*i: word for i, word in enumerate(w for d in descriptors for w in d)}
        checks = writes.copy()
        for base, payload in ((0x4000, raw), (0x5000, sign_bytes), (0x8000, fp.astype('<f4').tobytes()),
                              (0xe000, q8), (0x18000, fp.astype('<f4').tobytes()), (0x20000, q8)):
            writes.update({base+4*i: 0xdeadbeef for i in range(len(payload)//4)})
            checks.update({base+4*i: word for i, (word,) in enumerate(struct.iter_unpack('<I', payload))})
            for address in (base-4, base+len(payload)):
                writes[address] = checks[address] = 0xdeadbeef
        writes.update({0x1000+4*i: 0xdeadbeef for i in range(8)})
        checks.update({0x1000+4*i: exponent & 0xffffffff if i == 0 else 0xdeadbeef for i in range(8)})
        vectors += struct.pack('<4I', len(writes), len(checks), len(descriptors), embedding_e & 0xffffffff)
        for entries in (writes, checks):
            vectors += b''.join(struct.pack('<II', a, v) for a, v in entries.items())
    dut.models[0].mem.init = [int.from_bytes(memory[i:i+32], 'little') for i in range(0, len(memory), 32)]
    (tmp_path / 'vectors.bin').write_bytes(vectors)
    bus = dut.scratchpad.add_wishbone_port()
    status = Signal(24)
    dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
    ports = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
    ports.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
        seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re, seq_status=status,
        embedding_error=dut.seq.embedding_restore_error.status,
        embedding_busy=dut.embedding_restore.busy, embedding_reads=dut.embedding_restore.read_beats,
        embedding_writes=dut.embedding_restore.output_beats, embedding_exponent=dut.embedding_restore.exponent,
        quant_error=dut.seq.activation_quantize_error.status, quant_busy=dut.activation_quantize.busy,
        quant_reads=dut.activation_quantize.read_beats, quant_writes=dut.activation_quantize.output_beats,
        quant_blocks=dut.activation_quantize.blocks_done,
        reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
        reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
        engine_done=dut.engine_csr.done_count.status, engine_busy=dut.engine.busy,
        region_error=dut.seq.region_fetch_error.status if region_fetch else Signal(32),
        manual_read_accepted=dut.zzverify.accepted.status if companions else Signal(32),
        manual_read_completed=dut.zzverify.completed.status if companions else Signal(32),
        ar_valid=dut.ports[0].ar.valid, ar_ready=dut.ports[0].ar.ready,
        ar_addr=dut.ports[0].ar.addr, ar_len=dut.ports[0].ar.len)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'embedding_resident.v'
    converted = verilog.convert(dut, ios=set(ports.values()), name='embedding_resident')
    rtl.write_text(converted.main_source)
    for name, content in converted.data_files.items():
        (tmp_path / name).write_text(content)
    obj = tmp_path / 'obj'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
        '--Mdir', str(obj), '--top-module', 'embedding_resident', str(rtl),
        str((Path(__file__).parent / 'rtl/embedding_resident.cpp').resolve())],
        capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stdout[-7000:]+built.stderr[-7000:]
    assert 'UNOPTFLAT' not in built.stderr, built.stderr
    result = subprocess.run([str(obj / 'Vembedding_resident')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '$readmem file not found' not in result.stdout
    assert 'EMBEDDING_RESIDENT_PASS' in result.stdout
    print(f'companions={companions} region_fetch={region_fetch} local_27b={local}\n{result.stdout}')

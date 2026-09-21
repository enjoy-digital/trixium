# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import Memory, Signal
from migen.fhdl import verilog
from litex.gen import LiteXModule

from trixium.gateware.embedding_restore import EmbeddingRestore
from trixium.gateware.native_weights import NativeWeightStream, PTQ1_0
from trixium.model import qformats
from trixium.model.gguf_reader import GGUFReader
from trixium.model.golden import config_from_gguf
from test_embedding_restore import example, golden
from test_hadamard import packed


class NativeEmbedding(LiteXModule):
    def __init__(self, width):
        self.unit = u = EmbeddingRestore(input_width=width)
        self.decoder = dec = NativeWeightStream()
        self.cancel = Signal()
        self.error = Signal(7)
        # Test-only preload port for already resident inverse signs. No host
        # lookup/conversion occurs between start and completion. This is not a
        # production scratchpad adapter or an extra physical memory port claim.
        signs = Memory(128, 64)
        self.sign_write = sw = signs.get_port(write_capable=True)
        sr = signs.get_port(async_read=True)
        self.specials += signs, sw, sr
        self.comb += [u.cancel.eq(self.cancel), dec.cancel.eq(self.cancel),
            dec.start.eq(u.cmd.valid & u.cmd.ready), dec.rows.eq(1), dec.groups.eq(u.cmd.groups),
            dec.encoding.eq(PTQ1_0), sr.adr.eq(dec.source.group),
            u.sink.valid.eq(dec.source.valid), dec.source.ready.eq(u.sink.ready),
            u.sink.codes.eq(dec.source.codes), u.sink.scale.eq(dec.source.scale),
            u.sink.negate.eq(sr.dat_r), u.sink.first.eq(dec.source.first), u.sink.last.eq(dec.source.last),
            self.error.eq(u.error | (dec.error << 6))]


def cases(width):
    out = []
    for i, groups in enumerate((40, 8, 16, 64, 40, 8)):
        out.append((*example(groups, seed=43+i, span=min(width-12, 4)), {}))
    scales, codes, signs = example(40, span=0)
    out.append((scales, np.ones_like(codes), signs, {}))
    scales = np.zeros(40, dtype=np.float16); scales.view(np.uint16)[0] = 1
    codes = np.ones(5120, dtype=np.uint8); codes[0] = 2
    out.append((scales, codes, signs, {}))
    for deficit in (15, 16, 17, 47, 48, 49):
        scales = np.ones(8, dtype=np.float16); scales[-1] = np.float16(2047/2048)
        codes = np.ones(1024, dtype=np.uint8)
        codes[:512-deficit] = 2; codes[896:896+deficit] = 2
        signs = np.ones(1024, dtype=np.int8); signs[0] = -1
        out.append((scales, codes, signs, {}))
    scales, codes, signs = example(8, span=0)
    scales.view(np.uint16)[:] = [0, 0x8000, 1, 0x8001, 1023, 0x83ff, 1024, 0x8400]
    out.append((scales, codes, signs, {}))
    if width == 32:
        scales.view(np.uint16)[:] = [1, 0x8001, 0x57ff, 0xd7ff, 0x13ff, 0x0400, 0, 0x8000]
        out.append((scales.copy(), codes, signs, {}))
    for exponent_field in range(31):
        scales, codes, signs = example(8, seed=exponent_field+43, span=0)
        magnitude = (exponent_field << 10) | 1000
        scales.view(np.uint16)[:] = [magnitude, magnitude | 0x8000]*4
        out.append((scales, codes, signs, {}))
    for groups in (0, 7, 65):
        # Zero groups is invalid to both owners; partial/oversized rows fail
        # the embedding contract before the generic decoder receives data.
        out.append((*example(8), dict(groups=groups, error=65 if groups == 0 else 1)))
    for value in (0x7c00, 0xfc00, 0x7e01, 0xfe00):
        scales, codes, signs = example(8); scales.view(np.uint16)[3] = value
        out.append((scales, codes, signs, dict(error=4)))
    scales, codes, signs = example(8)
    scales.view(np.uint16)[0] = 1; scales.view(np.uint16)[1] = 0x7bff
    out.append((scales, codes, signs, dict(error=8)))
    for mode in (1, 2):
        out.append((*example(8), dict(frame=mode, error=64)))

    path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    local_count = 0
    if path.exists():
        reader = GGUFReader(path)
        cfg = config_from_gguf(reader)
        table = reader.tensors['token_embd.weight']
        source = reader.tensor_bytes(table.name)
        all_scales = qformats.ptq1_0_scales(source).reshape(table.shape[0], -1)
        from trixium.model.bitexact import fp16_decompose
        _, exponents = fp16_decompose(all_scales)
        rare = np.flatnonzero(exponents.max(axis=1)-exponents.min(axis=1) == 2).tolist()
        ids = sorted(set([0, 1, 11751, table.shape[0]-1] + rare +
                         np.random.default_rng(302).integers(0, table.shape[0], 16).tolist()))
        signs = np.asarray(cfg.hadamard_signs[5120], dtype=np.int8)
        for row in ids:
            scales, codes = qformats.ptq1_0_unpack(source[row*1120:(row+1)*1120])
            out.append((scales, codes.reshape(-1), signs, {}))
        local_count = len(ids)
    return out, local_count


@pytest.mark.parametrize('width', [16, 32])
def test_compiled_native_ptq_embedding_exactness_and_cancellation(tmp_path, width):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for compact embedding chain')
    examples, local_count = cases(width)
    vectors = bytearray(struct.pack('<III', 0x454d4231, len(examples), width))
    for scales, codes, signs, options in examples:
        raw = qformats.ptq1_0_pack(scales, codes.reshape(-1, 128)).tobytes()
        error = options.get('error', 0)
        exponent, expected = 0, b''
        if not error:
            _, exponent, fp = golden(scales, codes, signs)
            expected = fp.astype('<f4').tobytes()
        vectors += struct.pack('<6I', options.get('groups', len(scales)), len(raw)//32,
            exponent & 0xffffffff, error, options.get('frame', 0), len(scales))
        vectors += b''.join(packed(signs[g*128:(g+1)*128] < 0, 1).to_bytes(16, 'little') for g in range(len(scales)))
        vectors += raw + expected
    (tmp_path / 'vectors.bin').write_bytes(vectors)
    dut = NativeEmbedding(width)
    u, dec, sw = dut.unit, dut.decoder, dut.sign_write
    ports = dict(cancel=dut.cancel, error=dut.error, busy=u.busy, done=u.done, cancelled=u.cancelled,
        cmd_valid=u.cmd.valid, cmd_ready=u.cmd.ready, cmd_groups=u.cmd.groups,
        sink_valid=dec.sink.valid, sink_ready=dec.sink.ready, sink_data=dec.sink.data, sink_last=dec.sink.last,
        source_valid=u.source.valid, source_ready=u.source.ready, source_data=u.source.data,
        source_exponent=u.source.exponent, source_first=u.source.first, source_last=u.source.last,
        signs_address=sw.adr, signs_data=sw.dat_w, signs_we=sw.we)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'embedding_native.v'
    verilog.convert(dut, ios=set(ports.values()), name='embedding_native').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/embedding_native.cpp'
    build = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
        '--Mdir', str(obj), '--top-module', 'embedding_native', str(rtl), str(harness.resolve())],
        capture_output=True, text=True, timeout=240)
    assert build.returncode == 0, build.stdout[-8000:] + build.stderr[-8000:]
    result = subprocess.run([str(obj / 'Vembedding_native')], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'EMBEDDING_NATIVE_PASS' in result.stdout
    print(f'input_width={width} local_27b_rows={local_count}\n{result.stdout}')

# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import random

import numpy as np
import pytest
from migen import run_simulation

from trixium import layout
from trixium.gateware.embedding_restore import EmbeddingRestore
from trixium.model import bons4, qformats
from trixium.model.bons import restored_bfp_rows
from trixium.model.embedding import embedding_scale_contract, profile_ptq_embedding, restore_embedding_row
from test_hadamard import packed


def example(groups=8, seed=84, span=4):
    rng = np.random.default_rng(seed)
    scale_bits = rng.integers(0, 1024, groups, dtype=np.uint16)
    scale_bits |= (rng.integers(1, span+2, groups, dtype=np.uint16) << 10)
    scale_bits |= (rng.integers(0, 2, groups, dtype=np.uint16) << 15)
    scales = scale_bits.view(np.float16)
    codes = rng.integers(0, 3, groups*128, dtype=np.uint8)
    signs = rng.choice([-1, 1], groups*128)
    return scales, codes, signs


def golden(scales, codes, signs):
    m, e = restored_bfp_rows(scales[None, :], codes[None, :], signs, 1024)
    fp = m[0].astype(np.float32) * np.float32(2.0**int(e[0]))
    return m[0], int(e[0]), fp


@pytest.mark.parametrize('groups,span', [(8, 0), (8, 4), (40, 2), (64, 4)])
def test_integer_embedding_reference_matches_existing_dense_converter(groups, span):
    scales, codes, signs = example(groups, span=span)
    got, expected = restore_embedding_row(scales, codes, signs), golden(scales, codes, signs)
    assert np.array_equal(got[0], expected[0]) and got[1] == expected[1]
    assert np.array_equal(got[2].view(np.uint32), expected[2].view(np.uint32))


def test_embedding_zero_subnormal_signs_and_whole_row_exponent():
    scales, codes, signs = example(16, span=0)
    scales.view(np.uint16)[:] = [0x8000, 0, 1, 0x8001, 1023, 0x83ff, 1024, 0x8400]*2
    scales[8:] *= np.float16(8)
    got, expected = restore_embedding_row(scales, codes, signs), golden(scales, codes, signs)
    assert np.array_equal(got[0], expected[0]) and got[1] == expected[1]
    assert np.array_equal(got[2].view(np.uint32), expected[2].view(np.uint32))
    mant, exponent, floats = restore_embedding_row(scales, np.ones_like(codes), signs)
    assert not mant.any() and exponent == 0 and not floats.view(np.uint32).any()


def test_embedding_wider_exact_input_does_not_cap_large_scale_differences():
    scales, codes, signs = example(8, span=0)
    scales.view(np.uint16)[:] = [1, 0x8001, 0x57ff, 0xd7ff, 0x13ff, 0x0400, 0, 0x8000]
    with pytest.raises(ValueError, match='span'):
        restore_embedding_row(scales, codes, signs)
    got = restore_embedding_row(scales, codes, signs, input_width=32)
    expected = golden(scales, codes, signs)
    assert np.array_equal(got[0], expected[0]) and got[1] == expected[1]
    assert np.array_equal(got[2].view(np.uint32), expected[2].view(np.uint32))


def test_embedding_reference_rejects_lossy_alignment_and_bad_geometry():
    scales, codes, signs = example(span=0)
    for index, bad in [(0, np.inf), (1, np.nan)]:
        broken = scales.copy(); broken[index] = bad
        with pytest.raises(ValueError, match='nonfinite'):
            restore_embedding_row(broken, codes, signs)
    broken = scales.copy(); broken[0] = np.float16(1)
    with pytest.raises(ValueError, match='span'):
        restore_embedding_row(broken, codes, signs)
    with pytest.raises(ValueError):
        restore_embedding_row(scales, codes[:-1], signs[:-1])
    with pytest.raises(ValueError):
        restore_embedding_row(scales, codes.astype(np.float32), signs)
    broken_codes = codes.copy(); broken_codes[0] = 3
    with pytest.raises(ValueError):
        restore_embedding_row(scales, broken_codes, signs)
    with pytest.raises(ValueError):
        restore_embedding_row(scales, codes, signs*0)


@pytest.mark.parametrize('deficit', [15, 16, 17, 47, 48, 49])
def test_embedding_rne_halfway_even_odd_and_symmetric_clip(deficit):
    scales = np.ones(8, dtype=np.float16)
    scales[-1] = np.float16(2047/2048)
    codes = np.ones(1024, dtype=np.uint8)
    codes[:512-deficit] = 2
    codes[896:896+deficit] = 2
    signs = np.ones(1024, dtype=np.int8); signs[0] = -1
    # Raw DC = 2**20-deficit. k=5; deficit=16 is the odd tie that
    # rounds to 32768 and must clip symmetrically, not wrap to -32768.
    got, expected = restore_embedding_row(scales, codes, signs), golden(scales, codes, signs)
    assert np.array_equal(got[0], expected[0]) and got[1] == expected[1]
    assert got[0][0] == -min(32767, round(((1 << 20)-deficit)/32))


def make_image(tmp_path):
    scales, codes, signs = example(40, span=2)
    payload = qformats.ptq1_0_pack(scales, codes.reshape(-1, 128)).tobytes()
    path = tmp_path / 'embedding.bons'
    bons4.write(path, [bons4.TensorSource('token_embd.weight', bons4.Encoding.PTQ1_0,
        (1, 5120), payload, role=layout.ROLE_TOK_EMB)], {'qualification_subset': True})
    return path, bons4.Reader(path)


def test_embedding_profile_verifies_source_and_bound_without_expansion(tmp_path):
    path, reader = make_image(tmp_path)
    saved = path.read_bytes()
    profile = profile_ptq_embedding(reader)
    assert profile.source.shape == (1, 5120) and profile.input_width == 16
    assert sum(profile.span_histogram) == 1 and profile.span_histogram[2] == 1
    assert profile.source_directory_sha256 == reader.directory_sha256
    assert path.read_bytes() == saved and reader.verify()
    with pytest.raises(ValueError, match='geometry'):
        profile_ptq_embedding(reader, max_groups=8)
    broken = bytearray(saved); broken[profile.source.offset] ^= 1
    path.write_bytes(broken)
    with pytest.raises(ValueError, match='SHA256'):
        profile_ptq_embedding(reader)


def start(dut, groups):
    yield dut.cmd.groups.eq(groups)
    yield dut.cmd.valid.eq(1)
    yield
    assert (yield dut.cmd.ready)
    yield dut.cmd.valid.eq(0)
    yield
    assert (yield dut.busy)


def complete(dut, scales, codes, signs, seed=29):
    groups = len(scales)
    _, exponent, fp = golden(scales, codes, signs)
    expected = fp.view(np.uint32).tolist()
    rng = random.Random(seed)
    yield dut.sink.valid.eq(0)
    yield dut.source.ready.eq(0)
    yield from start(dut, groups)
    sent = received = 0
    held, offering = None, False
    for cycle in range(3000*groups):
        offering = offering or rng.randrange(4) != 0
        yield dut.sink.valid.eq(sent < groups and offering)
        yield dut.sink.first.eq(sent == 0)
        yield dut.sink.last.eq(sent == groups-1)
        if sent < groups:
            yield dut.sink.codes.eq(packed(codes[sent*128:(sent+1)*128], 2))
            yield dut.sink.scale.eq(int(scales.view(np.uint16)[sent]))
            yield dut.sink.negate.eq(packed(signs[sent*128:(sent+1)*128] < 0, 1))
        yield dut.source.ready.eq(cycle % 71 not in range(10, 33) and rng.randrange(3) != 0)
        yield dut.cmd.groups.eq(0)  # Geometry must have been snapshotted.
        yield
        assert not (yield dut.error), ((yield dut.error), cycle)
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            sent += 1; offering = False
        fields = ((yield dut.source.data), (yield dut.source.exponent),
                  (yield dut.source.first), (yield dut.source.last))
        if held is not None:
            assert (yield dut.source.valid) and fields == held
        held = fields if (yield dut.source.valid) and not (yield dut.source.ready) else None
        if (yield dut.source.valid) and (yield dut.source.ready):
            assert sent == groups
            assert fields[0] == packed(expected[received:received+8], 32), (received, fields[0])
            assert fields[1] == exponent & 1023
            assert fields[2] == (received == 0) and fields[3] == (received+8 == len(codes))
            received += 8
        if (yield dut.done):
            assert received == len(codes) and not (yield dut.busy)
            yield dut.sink.valid.eq(0)
            yield
            assert not (yield dut.done) and not (yield dut.source.valid)
            return
    raise AssertionError(f'embedding timeout sent={sent} received={received}')


def test_embedding_kernel_stalls_exact_fp32_and_reuse():
    dut = EmbeddingRestore(max_groups=16)
    def run():
        yield from complete(dut, *example())
        scales, codes, signs = example(16, span=2)
        yield from complete(dut, scales, codes, signs)
        yield from complete(dut, scales, np.ones_like(codes), signs)
    run_simulation(dut, run())


@pytest.mark.parametrize('fault', ['groups_zero', 'groups_partial', 'groups_large', 'first',
                                  'last', 'code', 'nan', 'inf', 'span'])
def test_embedding_kernel_rejects_bad_rows_before_output(fault):
    dut = EmbeddingRestore(max_groups=8)
    def run():
        yield from start(dut, {'groups_zero': 0, 'groups_partial': 7, 'groups_large': 16}.get(fault, 8))
        for group in range(8):
            yield dut.sink.valid.eq(1)
            yield dut.sink.first.eq(group == 0 and fault != 'first')
            yield dut.sink.last.eq(group == 7 or (fault == 'last' and group == 0))
            yield dut.sink.codes.eq(3 if fault == 'code' else 0)
            yield dut.sink.scale.eq({'nan': 0x7e00, 'inf': 0x7c00}.get(fault,
                0x3c00 if fault == 'span' and group == 0 else 0x0400))
            yield
            if (yield dut.error):
                break
        yield dut.sink.valid.eq(0)
        for _ in range(8):
            yield
            assert not (yield dut.source.valid) and not (yield dut.done) and (yield dut.busy)
        wanted = 1 if fault.startswith('groups') else 2 if fault in ('first', 'last', 'code') else 4 if fault in ('nan', 'inf') else 8
        assert (yield dut.error) == wanted
        yield dut.cancel.eq(1)
        yield
        assert not (yield dut.source.valid) and not (yield dut.sink.ready)
        yield dut.cancel.eq(0)
        yield
        assert not (yield dut.error) and not (yield dut.busy) and (yield dut.cancelled)
    run_simulation(dut, run())


def test_local_complete_embedding_admission(tmp_path):
    from trixium.model.gguf_reader import GGUFReader
    path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    if not path.exists():
        pytest.skip('optional local 27B GGUF is not installed')
    source = GGUFReader(path).tensors['token_embd.weight']
    image = tmp_path / 'embedding-subset.bons'
    bons4.write(image, [bons4.TensorSource(source.name, bons4.Encoding.PTQ1_0, source.shape,
        bons4.FileSlice(path, source.file_offset, 278_118_400), role=layout.ROLE_TOK_EMB)],
        {'qualification_subset': True})
    reader = bons4.Reader(image)
    profile = profile_ptq_embedding(reader)
    assert profile.source.shape == (248320, 5120)
    assert profile.span_histogram == (100548, 147764, 8, 0, 0)
    assert (profile.min_exponent, profile.max_exponent) == (-24, -16)
    print(f'EMBEDDING_ADMISSION_PASS rows={profile.source.rows} bytes={profile.source.size} '
          f'sha256={profile.source.sha256.hex()} spans={profile.span_histogram}')


@pytest.mark.parametrize('groups,width', [(0, 16), (7, 16), (9, 16), (520, 16), (8, 11), (8, 33), (True, 16)])
def test_embedding_kernel_parameter_validation(groups, width):
    with pytest.raises(ValueError):
        EmbeddingRestore(groups, width)


def test_embedding_synthesis_probe_refuses_overwrite_and_bad_parameters(tmp_path):
    import subprocess
    import sys
    tool = Path(__file__).resolve().parents[1] / 'tools/synth_embedding_restore.py'
    output = tmp_path / 'probe'
    args = [sys.executable, str(tool), '--output-dir', str(output)]
    run = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    assert b'-mode out_of_context' in saved['synth.tcl'] and b'set_false_path' not in saved['synth.xdc']
    assert subprocess.run(args, capture_output=True, timeout=30).returncode != 0
    assert saved == {p.name: p.read_bytes() for p in output.iterdir()}
    for bad in (['--max-groups', '7'], ['--input-width', '33'], ['--part', 'x; error injected']):
        invalid = tmp_path / 'bad'
        assert subprocess.run([sys.executable, str(tool), '--output-dir', str(invalid)] + bad,
            capture_output=True, timeout=30).returncode != 0
        assert not invalid.exists()

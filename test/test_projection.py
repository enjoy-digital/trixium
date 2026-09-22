# SPDX-License-Identifier: BSD-2-Clause
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trixium import layout
from trixium.model import bitexact as bx, bons4, qformats
from trixium.model.convert import _spec_from_gguf
from trixium.model.gguf_reader import GGUFReader
from trixium.model.projection import (
    BF16_Q8_TRANSFORM, ReferenceQ8Projection, prepare_bf16_projection, reference_q8_from_bf16,
    resident_reference_q8_descriptors,
)
from test_native_engine import compiled_case, compiled_native_engine


def bf16_bytes(weights):
    return (np.asarray(weights, dtype=np.float32).view(np.uint32) >> 16).astype('<u2').tobytes()


def legacy_spec(raw, shape, name='blk.0.ssm_beta.weight'):
    # Unmodified production converter is the oracle, not another implementation
    # of the new quantizer and not the new prepared payload's own unpacked bytes.
    reader = SimpleNamespace(tensors={name: SimpleNamespace(type_name='BF16', type_id=30, shape=shape)},
                             tensor_bytes=lambda name: raw)
    return _spec_from_gguf(reader, name, 16, 0)


def assert_legacy_equal(raw, shape, projection):
    ref = legacy_spec(raw, shape)
    expected = qformats.q8_0_pack(ref.scales.reshape(-1), ref.values.reshape(-1, 32)).tobytes()
    assert projection.payload == expected
    assert (projection.bias, projection.rshift) == (ref.bias, ref.rshift)
    return ref


@pytest.mark.parametrize('shape', [(1, 32), (7, 96), (48, 5120), (49, 5120)])
def test_reference_q8_matches_unchanged_converter_and_row_slices(shape):
    rng = np.random.default_rng(174)
    w = rng.standard_normal(shape).astype(np.float32)
    w *= np.exp2(rng.integers(-28, 6, (shape[0], shape[1] // 32, 1))).repeat(32, axis=2).reshape(shape)
    raw = bf16_bytes(w)
    p = reference_q8_from_bf16(raw, shape)
    assert_legacy_equal(raw, shape, p)
    assert p.sha256 == hashlib.sha256(p.payload).digest()
    assert b''.join(p.read_rows(i, 1) for i in range(shape[0])) == p.payload
    with pytest.raises(FrozenInstanceError):
        p.bias = 0
    for start, count in [(-1, 1), (0, 0), (shape[0], 1), (0, shape[0]+1), (True, 1), (0, True)]:
        with pytest.raises(ValueError, match='row range'):
            p.read_rows(start, count)


def test_fp16_scale_first_rne_zero_underflow_and_little_endian():
    w = np.zeros((4, 32), dtype=np.float32)
    w[0, :10] = [127, -127, .5, -.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5]
    w[1] = np.float32(2**-30)  # FP16 scale underflows; do not invent a minimum scale.
    w[2] = -0.0
    w[3] = np.float32(2**-18)  # FP16 subnormal scale, not flushed to zero.
    raw = bf16_bytes(w)
    p = reference_q8_from_bf16(raw, w.shape)
    assert_legacy_equal(raw, w.shape, p)
    assert p.payload[:2] == b'\x00\x3c'  # FP16 1, explicitly LE.
    s, q = qformats.q8_0_unpack(p.payload)
    assert q[0, :10].tolist() == [127, -127, 0, 0, 2, -2, 2, -2, 4, -4]
    assert not q[1:3].any() and not s[1:3].view(np.uint16).any()
    assert s[3].view(np.uint16) == 1
    rng = np.random.default_rng(8)
    raw = bf16_bytes(rng.standard_normal((48, 5120)).astype(np.float32) * .0123)
    w = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
    p = reference_q8_from_bf16(raw, (48, 5120))
    assert p.payload != qformats.q8_0_quant(w).tobytes(), 'fixture must detect scale-order regression'


def test_whole_matrix_fixed_point_must_not_be_rechosen_per_shard():
    w = np.ones((2, 32), dtype=np.float32)
    w[1] *= 2**-20
    p = reference_q8_from_bf16(bf16_bytes(w), w.shape)
    shard = ReferenceQ8Projection((1, 32), p.read_rows(1, 1))
    assert shard.bias != p.bias
    # read_rows intentionally returns only bytes, not a new object with changed B/R.
    assert_legacy_equal(bf16_bytes(w), w.shape, p)


def test_zero_and_unsafe_dense_matrix_accumulator_contract():
    p = reference_q8_from_bf16(bytes(64), (1, 32))
    assert p.payload == bytes(34) and (p.bias, p.rshift, p.accumulator_bound) == (36, 0, 0)
    # Unlike the real sparse-ish quantized matrices, constant maximal weights
    # can overflow int48 with the legacy B. Do not quietly change B or saturate.
    with pytest.raises(ValueError, match='int48'):
        reference_q8_from_bf16(bf16_bytes(np.ones((1, 5120), np.float32)), (1, 5120))


def test_accumulator_bound_covers_signed_floor_and_weight_aligned_activations():
    w = np.zeros((2, 64), dtype=np.float32)
    w[0, :32] = 127
    w[0, 32:] = 2**-18  # Small-scale negative products exercise floor, not truncation.
    w[1] = -w[0]
    p = reference_q8_from_bf16(bf16_bytes(w), w.shape)
    scales, quants = qformats.q8_0_unpack(p.payload)
    scales, quants = scales.reshape(2, 2), quants.reshape(2, 2, 32)
    mantissas, exponents = bx.fp16_decompose(scales)
    for row in range(2):
        for sign in (-1, 1):
            x = np.sign(quants[row].astype(np.int64)) * (sign * 127)
            acc = 0
            for group in range(2):
                product = int((quants[row, group].astype(np.int64) * x[group]).sum()) * int(mantissas[row, group])
                shift = int(exponents[row, group]) + p.bias
                acc += product << shift if shift >= 0 else product >> -shift
            assert abs(acc) <= p.accumulator_bound < 1 << 47


@pytest.mark.parametrize('bits', [0x7f80, 0xff80, 0x7fc1, 0xffa0, 0x7f7f, 0xff7f])
def test_reject_nonfinite_and_scale_overflow(bits):
    raw = np.full(32, bits, dtype='<u2').tobytes()
    with pytest.raises(ValueError, match='nonfinite|overflows'):
        reference_q8_from_bf16(raw, (1, 32))


@pytest.mark.parametrize('shape,payload', [
    ((32,), bytes(64)), ((1, 1, 32), bytes(64)), ((1, 33), bytes(66)),
    ((0, 32), b''), ((True, 32), bytes(64)), ((1, 32), bytes(63)),
    ((1, 32), bytes(65)), ((1, 32), bytearray(64)), ((1 << 63, 32), b''),
])
def test_reject_bad_bf16_geometry(shape, payload):
    with pytest.raises(ValueError):
        reference_q8_from_bf16(payload, shape)


@pytest.mark.parametrize('scale,quant', [(float('nan'), 0), (float('inf'), 0), (-1, 0), (1, -128)])
def test_q8_contract_rejects_unsafe_weights(scale, quant):
    raw = qformats.q8_0_pack(np.array([scale], np.float16), np.full((1, 32), quant, np.int8)).tobytes()
    with pytest.raises(ValueError):
        ReferenceQ8Projection((1, 32), raw)


def make_bons(tmp_path, **changes):
    raw = bf16_bytes(np.arange(96, dtype=np.float32).reshape(3, 32) / 1024)
    params = dict(name='blk.2.ssm_beta.weight', encoding=bons4.Encoding.BF16,
                  shape=(3, 32), payload=raw, role=layout.ROLE_SSM_BETA, layer=2)
    params.update(changes)
    path = tmp_path / 'model.bons'
    bons4.write(path, [bons4.TensorSource(**params)], {'architecture': 'test'})
    return path, bons4.Reader(path), raw


def test_bons4_preparation_preserves_source_and_records_distinct_execution(tmp_path):
    path, reader, raw = make_bons(tmp_path)
    before = path.read_bytes()
    p = prepare_bf16_projection(reader, 'blk.2.ssm_beta.weight')
    assert_legacy_equal(raw, (3, 32), p.execution)
    assert p.source.encoding == bons4.Encoding.BF16
    assert p.source_directory_sha256 == reader.directory_sha256
    m = p.manifest()
    assert m['transform'] == BF16_Q8_TRANSFORM
    assert m['source_sha256'] == hashlib.sha256(raw).hexdigest()
    assert m['execution_sha256'] == hashlib.sha256(p.execution.payload).hexdigest()
    assert (m['source_bytes'], m['execution_bytes']) == (192, 102)
    assert m['source_encoding'] == 30 and m['execution_encoding'] == 8
    assert m['activation_basis'] == 'unrotated'
    assert m['activation_q8_range'] == [-127, 127]
    assert m['source_directory_sha256'] == reader.directory_sha256.hex()
    assert path.read_bytes() == before and reader.verify()
    with pytest.raises(ValueError, match='byte budget'):
        prepare_bf16_projection(reader, p.source.name, max_source_bytes=len(raw)-1)
    # A directory-valid artifact with damaged payload must never produce executable bytes.
    damaged = bytearray(before)
    damaged[p.source.offset] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match='SHA256'):
        prepare_bf16_projection(reader, p.source.name)


@pytest.mark.parametrize('changes', [dict(role=layout.ROLE_ATTN_Q), dict(layer=bons4.UNSPECIFIED),
                                    dict(shape=(96,)), dict(encoding=bons4.Encoding.F16)])
def test_bons4_preparation_rejects_unrecognized_roles_or_shapes(tmp_path, changes):
    _, reader, _ = make_bons(tmp_path, **changes)
    with pytest.raises(ValueError, match='recognized'):
        prepare_bf16_projection(reader, 'blk.2.ssm_beta.weight')


@pytest.mark.parametrize('pipes', [1, 2, 16, 32])
def test_resident_projection_lowering_keeps_whole_contract_and_reuses_activation(pipes):
    p = reference_q8_from_bf16(bytes(3*5120*2), (3, 5120))
    args = dict(pipe_locations=[(i+42, (1 << 32)+i*4096) for i in range(pipes)],
                q8_src=0x4000, exponents=0x800, int32_dst=0x6000, fp32_dst=0x7000)
    seq = resident_reference_q8_descriptors(p, fp32_src=0x10000, **args)
    assert [d[0] for d in seq] == [4] + [6]*pipes + [7, 8]
    assert seq[0][4:6] == [5120, 1]  # Whole unrotated activation, not five independent blocks.
    assert seq[1][3:7] == [0, 1, 5440, 0]
    assert seq[-2][2] == 3//pipes | ((3 % pipes) << 16)
    assert seq[-2][7] == p.bias | (p.rshift << 16)
    assert seq[-1][4:6] == [3, (p.rshift-p.bias) & 0xffffffff]
    assert resident_reference_q8_descriptors(p, **args) == seq[1:]
    for locations in ([], [(0, 0)]*33):
        with pytest.raises(ValueError, match='compute-pipe'):
            resident_reference_q8_descriptors(p, **dict(args, pipe_locations=locations))


def projection_engine_case(projection, reference, seed=38):
    rows, cols = projection.shape
    per_pipe, rem = divmod(rows, 2)
    groups = cols // 32
    rng = np.random.default_rng(seed)
    x = rng.integers(-127, 128, cols, dtype=np.int8)
    expected = bx.matvec_q8(reference.scales, reference.values, x, reference.bias, reference.rshift)
    memory, descriptors, reads = {}, [], []
    for pipe, base in enumerate((0, 1 << 32)):
        start = pipe * per_pipe + min(pipe, rem)
        count = per_pipe + (pipe < rem)
        # Device rows are beat-aligned; a spare beat between rows exercises the stride.
        offset, stride = 0x10020, projection.device_stride + 32
        descriptors.append((7 if pipe == 0 else 42, offset, stride))
        for i in range(count):
            payload = projection.read_rows(start + i, 1)
            address = base + offset + i * stride
            memory.update({address + j: b for j, b in enumerate(payload)})
            reads.append((pipe, address, (len(payload) + 31) // 32))
    cmd = dict(native=True, kind=1, rows=per_pipe, rem=rem, cols=cols, cbeats=groups,
               sbeats=(groups+15)//16, wbase=0, wlen=0, bias=projection.bias, rshift=projection.rshift,
               rshift2=0, xsrc=0, dst=0xc004, qn=False, qstride=rows+3, ilv=False, supw=0, xred=0)
    return dict(cmd=cmd, descriptors=descriptors, memory=memory, reads=reads, expected=[expected], xb=x.tobytes())


def test_compiled_prepared_bf16_full_matrix_and_uneven_shards(tmp_path, compiled_native_engine):
    rng = np.random.default_rng(374)
    shape = (49, 5120)
    raw = bf16_bytes(rng.standard_normal(shape).astype(np.float32) * .007)
    projection = reference_q8_from_bf16(raw, shape)
    ref = assert_legacy_equal(raw, shape, projection)
    for seed in (81, 92, 631):
        case = projection_engine_case(projection, ref, seed)
        compiled_case(tmp_path, compiled_native_engine, case, seed=seed)


def test_local_27b_all_bf16_projections_match_reference(tmp_path, compiled_native_engine):
    path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    if not path.exists():
        pytest.skip('optional local 27B GGUF is not installed')
    reader = GGUFReader(path)
    tensors = [t for t in reader.tensors.values() if t.type_id == 30]
    assert len(tensors) == 96
    source_bytes = execution_bytes = 0
    worst_bound = (0, '')
    for index, tensor in enumerate(tensors):
        assert tensor.shape == (48, 5120)
        raw = bytes(reader.tensor_bytes(tensor.name))
        p = reference_q8_from_bf16(raw, tensor.shape)
        ref = _spec_from_gguf(reader, tensor.name, 16, 0)
        assert p.payload == qformats.q8_0_pack(ref.scales.reshape(-1), ref.values.reshape(-1, 32)).tobytes()
        assert (p.bias, p.rshift) == (ref.bias, ref.rshift)
        source_bytes += len(raw)
        execution_bytes += len(p.payload)
        worst_bound = max(worst_bound, (p.accumulator_bound, tensor.name))
        compiled_case(tmp_path, compiled_native_engine, projection_engine_case(p, ref, 911+index), seed=300+index)
    assert source_bytes == 47_185_920
    assert execution_bytes == 25_067_520
    print(f'BF16_MODEL_PASS tensors={len(tensors)} source_bytes={source_bytes} execution_bytes={execution_bytes} '
          f'worst_bound={worst_bound[0]} worst_tensor={worst_bound[1]} int48_fraction={worst_bound[0]/(1<<47):.12f}')

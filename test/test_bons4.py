# SPDX-License-Identifier: BSD-2-Clause
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from trixium.capabilities import (
    Access, Capabilities, Feature, MemoryKind, MemoryRegion, Reservation, ReservationKind,
)
from trixium.model import bons4 as b4, qformats as qf
from trixium.model.bons import BONSReader
from trixium.model.gguf_reader import GGUFReader
from trixium.model.pack_native import native_sources, pack
from trixium.placement import Placement, plan_weights

ROOT = Path(__file__).resolve().parents[1]


def sources():
    rng = np.random.default_rng(431)
    s = np.array([0.0, -0.0, 0.5, -2, 1e-5, 10], dtype='<f2')
    c = rng.integers(0, 3, (6, 128), dtype=np.uint8)
    return [b4.TensorSource('long.tensor.name.that.must.never.be.truncated', b4.Encoding.PTQ1_0,
                           (3, 256), qf.ptq1_0_pack(s, c).tobytes(), role=4, layer=2),
            b4.TensorSource('q8', b4.Encoding.Q8_0, (6, 32),
                           qf.q8_0_pack(s, rng.integers(-128, 128, (6, 32), dtype=np.int8)).tobytes()),
            b4.TensorSource('pq2', b4.Encoding.PQ2_0, (2, 3, 128), qf.q2_0_pack(s, c).tobytes()),
            b4.TensorSource('q1', b4.Encoding.Q1_0, (6, 128), qf.q1_0_pack(s, c & 1).tobytes()),
            b4.TensorSource('f32', b4.Encoding.F32, (2, 4, 3, 5),
                           rng.standard_normal(120).astype('<f4').tobytes()),
            b4.TensorSource('f16', b4.Encoding.F16, (6,), s.tobytes()),
            b4.TensorSource('bf16', b4.Encoding.BF16, (5,), bytes(range(10)))]


def make_image(tmp_path):
    path = tmp_path / 'model.bons'
    result = b4.write(path, sources(), {'architecture': 'test', 'nested': {'a': [1, True, None]}})
    return path, b4.Reader(path), result


def caps(direct=True, big=False):
    size = (8 << 30) if big else 4096
    return Capabilities(150_000_000, 150_000_000, 2, 32768,
        Feature.REGION_IO if direct else Feature(0), 3,
        tuple(MemoryRegion(i, MemoryKind.HBM, 0, i * size, size,
                           access=(Access.READ | Access.WRITE) if direct else Access(0), locality=i)
              for i in range(3)),
        (Reservation(0, ReservationKind.CPU, 0, (1 << 32) if big else 512),))


def repair(blob):
    b = bytearray(blob)
    poff = struct.unpack_from('<Q', b, 40)[0]
    b[56:88] = bytes(32)
    b[56:88] = hashlib.sha256(b[:poff]).digest()
    return bytes(b)


def mutate(blob, offset, value, fmt='<I'):
    b = bytearray(blob)
    struct.pack_into(fmt, b, offset, value)
    return repair(b)


def test_roundtrip_deterministic_native_payloads(tmp_path):
    path, reader, report = make_image(tmp_path)
    assert b4.HEADER.size == b4.ENTRY.size == 128
    assert reader.verify()
    assert report['file_bytes'] == path.stat().st_size
    assert report['payload_bytes'] == sum(len(s.payload) for s in sources())
    assert report['directory_sha256'] == reader.directory_sha256.hex()
    second = tmp_path / 'second.bons'
    b4.write(second, sources(), reader.model)
    assert second.read_bytes() == path.read_bytes()
    for source, t in zip(sources(), reader.entries):
        assert t.name == source.name and t.shape == source.shape and t.encoding == source.encoding
        assert b''.join(reader.iter_tensor(t.name)) == source.payload
        assert reader.read_rows(t.name, 0, t.rows) == source.payload
        assert t.sha256 == hashlib.sha256(source.payload).digest()
        if t.rows > 1:
            assert reader.read_rows(t.name, 1, 1) == source.payload[t.row_bytes:2*t.row_bytes]
        for start, count in [(-1, 1), (0, 0), (t.rows, 1), (0, t.rows + 1), (True, 1)]:
            with pytest.raises(ValueError):
                reader.read_rows(t.name, start, count)
    ptq = reader.entries[0]
    scales, codes = qf.ptq1_0_unpack(reader.read_rows(ptq.name, 1, 1))
    ref_scales, ref_codes = qf.ptq1_0_unpack(sources()[0].payload)
    assert np.array_equal(scales.view('<u2'), ref_scales[2:4].view('<u2'))
    assert np.array_equal(codes, ref_codes[2:4])


def test_writer_refuses_overwrite_alias_bad_sources_and_metadata(tmp_path):
    path, reader, _ = make_image(tmp_path)
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        b4.write(path, sources(), {})
    assert path.read_bytes() == saved
    for changes in [dict(name=''), dict(name='a\0b'), dict(name='a'*4097),
                    dict(shape=(1, 255)), dict(shape=()), dict(shape=(0, 128)),
                    dict(shape=(1 << 63, 128)), dict(encoding=999), dict(role=-1),
                    dict(layer=1 << 32), dict(payload=b''), dict(payload=bytearray(168))]:
        with pytest.raises(ValueError):
            replace(sources()[0], **changes)
    for src, model in [([], {}), ([sources()[0]]*2, {}), (sources(), []),
                       (sources(), {'a': float('nan')}), (sources(), {'a': 'x'*b4.MAX_METADATA})]:
        with pytest.raises(ValueError):
            b4.write(tmp_path / 'bad', src, model)
        assert not (tmp_path / 'bad').exists()
    s = sources()[0]
    target = tmp_path / 'alias'
    with pytest.raises(ValueError, match='aliases'):
        b4.write(target, [replace(s, payload=b4.FileSlice(target, 0, len(s.payload)))], {})


def test_streaming_file_slice_and_incomplete_output(tmp_path, monkeypatch):
    monkeypatch.setattr(b4, 'CHUNK_BYTES', 17)
    source = sources()[0]
    raw = tmp_path / 'source'
    raw.write_bytes(b'prefix' + source.payload + b'suffix')
    fsource = replace(source, payload=b4.FileSlice(raw, 6, len(source.payload)))
    assert max(map(len, fsource.chunks())) <= 17
    out = tmp_path / 'stream'
    b4.write(out, [fsource], {})
    assert b4.Reader(out).verify()
    raw.write_bytes(b'prefix')
    with pytest.raises(ValueError, match='truncated source'):
        b4.write(tmp_path / 'partial', [fsource], {})
    with pytest.raises(ValueError):
        b4.Reader(tmp_path / 'partial')


def test_reader_rejects_bad_directory_even_with_repaired_digest(tmp_path):
    path, _, _ = make_image(tmp_path)
    blob = path.read_bytes()
    toff = struct.unpack_from('<Q', blob, 32)[0]
    changes = [(4, 3, '<I'), (8, 64, '<I'), (12, 1, '<I'), (16, 64, '<I'),
               (20, 0, '<I'), (20, b4.MAX_TENSORS + 1, '<I'), (24, 0, '<Q'),
               (24, b4.MAX_METADATA + 1, '<Q'), (32, 128, '<Q'), (48, len(blob) + 1, '<Q'),
               (120, 1, '<Q'), (toff, 1, '<I'), (toff + 4, 99, '<I'),
               (toff + 16, 0, '<I'), (toff + 16, 5, '<I'), (toff + 20, 1, '<I'),
               (toff + 24, 0, '<Q'), (toff + 24, 1 << 63, '<Q'),
               (toff + 32, 255, '<Q'), (toff + 40, 128, '<Q'),
               (toff + 56, 0, '<Q'), (toff + 64, 1, '<Q'), (toff + 104, 1, '<Q')]
    for offset, value, fmt in changes:
        path.write_bytes(mutate(blob, offset, value, fmt))
        with pytest.raises(ValueError):
            b4.Reader(path)
    for data in [blob[:64], blob[:-1], blob + b'\0', blob[:130] + b'X' + blob[131:]]:
        path.write_bytes(data)
        with pytest.raises(ValueError):
            b4.Reader(path)


def test_payload_hash_and_padding_separate_from_metadata(tmp_path):
    path, reader, _ = make_image(tmp_path)
    blob = path.read_bytes()
    t = reader.entries[0]
    for pos in [t.offset, t.offset + t.size]:
        b = bytearray(blob)
        b[pos] ^= 1
        path.write_bytes(b)
        opened = b4.Reader(path)  # Directory itself is still valid.
        with pytest.raises(ValueError):
            opened.verify()


def test_mutation_fuzz_never_accepts_changed_bytes(tmp_path):
    path, _, _ = make_image(tmp_path)
    blob = path.read_bytes()
    rng = random.Random(43)
    for _ in range(100):
        b = bytearray(blob)
        pos = rng.randrange(len(b))
        b[pos] ^= 1 << rng.randrange(8)
        path.write_bytes(b)
        with pytest.raises((ValueError, UnicodeError)):
            b4.Reader(path).verify()


def test_logical_v4_rejected_by_legacy_reader_even_optimized_python(tmp_path):
    path, _, _ = make_image(tmp_path)
    with pytest.raises(ValueError, match='v2/v3 only'):
        BONSReader(path)
    p = subprocess.run([sys.executable, '-O', '-c',
                        'from trixium.model.bons import BONSReader; import sys; BONSReader(sys.argv[1])',
                        str(path)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and 'v2/v3 only' in p.stderr


def _string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def synthetic_gguf(path, *, type42=False, unknown=False):
    data = sources()
    data[0] = replace(data[0], name='token_embd.weight')
    if type42:
        # Enough groups to distinguish g128 from g64 despite alignment padding.
        data[2] = replace(data[2], shape=(8, 3, 128), payload=data[2].payload*4)
    metadata = {'general.architecture': 'qwen3', 'general.alignment': 32,
                'qwen3.block_count': 1, 'qwen3.embedding_length': 256,
                'qwen3.attention.head_count': 4, 'qwen3.attention.head_count_kv': 2,
                'qwen3.feed_forward_length': 512, 'qwen3.nextn_predict_layers': 1,
                'tokenizer.test': 'excluded from model metadata'}
    blob = struct.pack('<4sIQQ', b'GGUF', 3, len(data), len(metadata))
    for k, v in metadata.items():
        blob += _string(k)
        blob += struct.pack('<I', 8 if isinstance(v, str) else 4)
        blob += _string(v) if isinstance(v, str) else struct.pack('<I', v)
    payload = b''
    for i, s in enumerate(data):
        encoding = 42 if type42 and s.encoding == b4.Encoding.PQ2_0 else int(s.encoding)
        if unknown and i == 0:
            encoding = 99
        blob += _string(s.name) + struct.pack('<I', len(s.shape))
        blob += b''.join(struct.pack('<Q', d) for d in reversed(s.shape))
        blob += struct.pack('<IQ', encoding, len(payload))
        payload += s.payload
        payload += b'\xa5' * (b4.align(len(payload), 32) - len(payload))
    blob += bytes(b4.align(len(blob), 32) - len(blob))
    path.write_bytes(blob + payload)
    return data


def test_native_pack_preserves_all_tensors_and_metadata(tmp_path):
    gguf, out = tmp_path / 'test.gguf', tmp_path / 'native.bons'
    originals = synthetic_gguf(gguf, type42=True)
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.write_bytes(b'test tokenizer')
    result = pack(gguf, out, tokenizer=tokenizer, check=True)
    reader = b4.Reader(out)
    assert result['payload_bytes'] == sum(len(s.payload) for s in originals)
    assert reader.model['gguf_metadata']['qwen3.nextn_predict_layers'] == 1
    assert 'tokenizer.test' not in reader.model['gguf_metadata']
    assert reader.tokenizer_sha256 == hashlib.sha256(tokenizer.read_bytes()).digest()
    assert reader.model['source']['sha256'] == hashlib.sha256(gguf.read_bytes()).hexdigest()
    assert reader.model['config']['tie_embeddings'] is True
    for s in originals:
        t = reader.by_name[s.name]
        assert b''.join(reader.iter_tensor(t.name)) == s.payload
        assert t.shape == s.shape and t.encoding == s.encoding
    assert reader.by_name['token_embd.weight'].role != b4.UNSPECIFIED
    assert reader.by_name['q8'].role == b4.UNSPECIFIED
    assert 'had.' not in ' '.join(reader.by_name)
    assert '.w16' not in ' '.join(reader.by_name)


def test_native_pack_preflight_unknown_and_ambiguous_type(tmp_path):
    gguf = tmp_path / 'test.gguf'
    synthetic_gguf(gguf, unknown=True)
    with pytest.raises(ValueError):
        pack(gguf, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    synthetic_gguf(gguf, type42=True)
    reader = GGUFReader(gguf)
    # Both 34-byte g128 and 36-byte g64 extents can pad to 64: cannot decide.
    t = reader.tensors['pq2']
    t.shape, t.nbytes = (1, 128), 64
    with pytest.raises(ValueError, match='ambiguous/group-64'):
        native_sources(reader)


def test_placement_separate_from_compute_and_rounding(tmp_path):
    _, image, _ = make_image(tmp_path)
    c = caps()
    runtime = (Reservation(0, ReservationKind.SYSTEM, 768, 128),)
    plan = plan_weights(image, c, runtime_reservations=runtime)
    assert plan.validate(image, c, for_transfer=True)
    assert Placement.from_dict(json.loads(json.dumps(plan.to_dict())), image, c) == plan
    for key, value in [('version', 2), ('allocated_bytes', 0), ('unexpected', 1),
                       ('topology_only', 'false'), ('directory_sha256', 'xx')]:
        doc = plan.to_dict()
        doc[key] = value
        with pytest.raises(ValueError):
            Placement.from_dict(doc, image, c)
    assert len(c.regions) > c.compute_pipes
    assert plan.to_dict()['payload_bytes'] == sum(t.size for t in image.entries)
    assert plan.to_dict()['allocated_bytes'] >= plan.to_dict()['payload_bytes']
    # No projection support is implied by successful raw transport placement.
    assert not c.weight_encodings & (1 << 5)
    for s in plan.spans:
        assert s.offset % 32 == s.allocated_bytes % 32 == 0
        if s.region_id == 0:
            assert s.offset >= 512
            assert s.offset >= 896 or s.offset + s.allocated_bytes <= 768
    no_io = caps(False)
    with pytest.raises(ValueError, match='does not support'):
        plan_weights(image, no_io)
    offline = plan_weights(image, no_io, topology_only=True)
    assert offline.validate(image, no_io)
    with pytest.raises(ValueError, match='cannot authorize'):
        offline.validate(image, no_io, for_transfer=True)
    with pytest.raises(ValueError, match='different image/device'):
        plan.validate(image, replace(c, compute_pipes=16))


def test_placement_rejects_invalid_and_insufficient_capacity(tmp_path):
    _, image, _ = make_image(tmp_path)
    c = caps()
    plan = plan_weights(image, c)
    first = plan.spans[0]
    changes = [dict(spans=plan.spans[:-1]), dict(spans=plan.spans + (first,)),
               dict(spans=(replace(first, offset=0),) + plan.spans[1:]),
               dict(spans=(replace(first, row_start=1),) + plan.spans[1:]),
               dict(spans=(replace(first, allocated_bytes=first.allocated_bytes + 32),) + plan.spans[1:]),
               dict(spans=(replace(first, tensor_id=9999),) + plan.spans[1:])]
    for change in changes:
        with pytest.raises(ValueError):
            replace(plan, **change).validate(image, c)
    full = tuple(Reservation(r.id, ReservationKind.SYSTEM, 512 if r.id == 0 else 0,
                             r.size - (512 if r.id == 0 else 0)) for r in c.regions)
    with pytest.raises(ValueError, match='no whole-row placement'):
        plan_weights(image, c, runtime_reservations=full)
    with pytest.raises(ValueError, match='overlap'):
        plan_weights(image, c, runtime_reservations=c.reservations)


def test_placement_offsets_over_4gib_and_fragmentation(tmp_path):
    _, image, _ = make_image(tmp_path)
    c = caps(big=True)
    plan = plan_weights(image, c)
    assert all(s.offset >= 1 << 32 for s in plan.spans)
    assert plan.validate(image, c, for_transfer=True)
    # Many small regions, fewer pipes; tensors split only between complete rows.
    small = replace(caps(), regions=tuple(MemoryRegion(i, MemoryKind.DDR4, i, 0, 256,
                    access=Access.READ | Access.WRITE) for i in range(16)), reservations=())
    fragmented = plan_weights(image, small)
    assert fragmented.validate(image, small)
    assert len({s.region_id for s in fragmented.spans}) > small.compute_pipes


def test_reader_64bit_file_offsets_without_allocating_gigabytes(tmp_path):
    # Sparse test intentionally does not claim whole-payload verification.
    metadata = b4._json({'schema': 1, 'model': {}, 'tensor_names': ['large', 'tail']})
    toff = b4.align(128 + len(metadata))
    poff = toff + 2*b4.ENTRY.size
    large = b4.Tensor(0, 'large', b4.Encoding.F32, (1 << 30, 1), 0, 0, poff, 1 << 32, bytes(32))
    tail_bytes = bytes(range(28))
    tail = b4.Tensor(1, 'tail', b4.Encoding.PTQ1_0, (1, 128), 0, 0, poff + (1 << 32),
                     28, hashlib.sha256(tail_bytes).digest())
    total = b4.align(tail.offset + tail.size)
    directory = metadata + bytes(toff - 128 - len(metadata)) + large.packed() + tail.packed()
    header = b4.HEADER.pack(b'BONS', 4, 128, 0, 128, 2, len(metadata), toff, poff, total,
                            bytes(32), bytes(32), bytes(8))
    path = tmp_path / 'sparse.bons'
    with path.open('wb') as f:
        f.write(repair(header + directory))
        f.seek(tail.offset)
        f.write(tail_bytes + bytes(total - tail.offset - 28))
    reader = b4.Reader(path)
    assert reader.entries[1].offset > 1 << 32
    assert reader.read_rows('tail', 0, 1) == tail_bytes


def test_cli_pack_verify_place(tmp_path):
    from trixium.cli import main
    gguf = tmp_path / 'tiny.gguf'
    synthetic_gguf(gguf)
    image, placement = tmp_path / 'native.bons', tmp_path / 'placement.json'
    main(['pack', str(gguf), '--out', str(image), '--check'])
    main(['verify', str(image)])
    device = tmp_path / 'caps.bin'
    device.write_bytes(caps(False).to_bytes())
    main(['place', str(image), '--capabilities', str(device), '--out', str(placement), '--topology-only'])
    assert json.loads(placement.read_text())['topology_only'] is True


def test_legacy_c_upload_rejects_v4_before_mmio(tmp_path):
    cc = shutil.which('gcc')
    if not cc:
        pytest.skip('native C compiler required')
    source = ROOT / 'trixium/software/user/bonsai_loader.c'
    names = sorted(set(re.findall(r'\bCSR_[A-Z0-9_]+\b', source.read_text())))
    (tmp_path / 'csr.h').write_text('\n'.join(f'#define {name} {i*4}' for i, name in enumerate(names)))
    (tmp_path / 'soc.h').write_text('#define BONSAI_CHAN_WEIGHT_BASE 0x14000000\n'
                                  '#define BONSAI_CHAN_STRIDE 0x10000000\n#define BONSAI_N_CHANNELS 16\n')
    (tmp_path / 'mem.h').write_text('#define MAIN_RAM_BASE 0x40000000\n')
    harness = tmp_path / 'main.c'
    harness.write_text('#include "bonsai_loader.h"\n'
                       'int main(int argc, char **argv) { return loader_upload_image(argv[1], 0) ? 1 : 0; }\n')
    exe = tmp_path / 'load-test'
    subprocess.run([cc, '-O2', '-I'+str(tmp_path), '-I'+str(source.parent),
                    str(source), str(harness), '-o', str(exe)], check=True, capture_output=True)
    path, _, _ = make_image(tmp_path)
    p = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
    assert p.returncode == 0 and 'legacy loader accepts BONS v2/v3, not v4' in p.stderr
    for version in (0, 1, 2, 3, 4, 5, 0xffffffff):
        # No BAR mapping; any accidental MMIO fails the process. For v2/v3 a
        # mismatched channel count proves the version gate still lets them through.
        raw = bytearray(128)
        struct.pack_into('<II', raw, 0, 0x534e4f42, version)
        path.write_bytes(raw)
        p = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
        assert p.returncode == 0
        assert ('image has 0 channels' if version in (2, 3) else 'legacy loader accepts') in p.stderr


def test_firmware_staging_version_guard_before_copy(tmp_path):
    cc = shutil.which('gcc')
    if not cc:
        pytest.skip('native C compiler required')
    main = (ROOT / 'firmware/main.c').read_text()
    function = main[main.index('static int hdrload_staged(int quiet)'):main.index('\nstatic void hdrload(void)')]
    harness = tmp_path / 'staging.c'
    harness.write_text('''#include <stdio.h>
#include <stdlib.h>
#include "bons.h"
static struct bons_header staged;
static int fetches, resets;
#define SCRATCHPAD_BASE ((uintptr_t)&staged - 0xC000)
#define HDR_STAGING 0
#define BONSAI_HDR_ADDR ((uintptr_t)&staged)
static void engine_fetch(unsigned a, unsigned b, unsigned c, unsigned d) { fetches++; }
static const struct bons_header *model_header(void) { return &staged; }
static void model_cache_reset(void) { resets++; }
''' + function + '''
int main(int argc, char **argv) {
    staged.magic = BONS_MAGIC;
    staged.version = strtoul(argv[1], 0, 0);
    staged.payload_offset = 0xffffffff;
    int valid = staged.version == 2 || staged.version == 3;
    int rc = hdrload_staged(1);
    return !(rc == (valid ? 0 : -1) && fetches == 1 && resets == 0);
}
''')
    exe = tmp_path / 'staging-test'
    subprocess.run([cc, '-O2', '-I'+str(ROOT / 'firmware'), str(harness), '-o', str(exe)],
                   check=True, capture_output=True)
    for version in (0, 1, 2, 3, 4, 5, 0xffffffff):
        subprocess.run([str(exe), str(version)], check=True, capture_output=True)


def test_native_pack_rejects_duplicate_gguf_names(tmp_path):
    path = tmp_path / 'duplicate.gguf'
    synthetic_gguf(path)
    data = path.read_bytes().replace(b'q1', b'q8')
    path.write_bytes(data)
    with pytest.raises(ValueError, match='duplicate GGUF tensor'):
        GGUFReader(path)

# SPDX-License-Identifier: BSD-2-Clause
"""BONS v4: immutable logical tensors, not a board's upload image.

Native GGUF blocks are retained byte-for-byte. No channel count, lane index,
physical address, expanded embedding, or synthesized Hadamard matrix belongs in
this container. See docs/bons-v4.md. Legacy v2/v3 readers intentionally stay separate.
"""
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from pathlib import Path
import struct

MAGIC, VERSION = b'BONS', 4
ALIGNMENT = 64
HEADER = struct.Struct('<4sIIIIIQQQQ32s32s8s')  # 128 bytes
ENTRY = struct.Struct('<IIIIII4QQQ32s24s')      # 128 bytes
DIGEST_OFFSET = 56
MAX_METADATA = 16 * 1024 * 1024
MAX_TENSORS = 65536
CHUNK_BYTES = 1024 * 1024
UNSPECIFIED = 0xffffffff


class Encoding(IntEnum):
    # Native storage IDs follow the unambiguous GGML types. These are NOT the
    # capability ABI's hardware WeightEncoding IDs; storage != supported compute.
    F32 = 0
    F16 = 1
    Q8_0 = 8
    BF16 = 30
    Q1_0 = 41
    PQ2_0 = 142
    PTQ1_0 = 143


# (elements per block, bytes per block); blocks never cross a logical row.
BLOCKS = {Encoding.F32: (1, 4), Encoding.F16: (1, 2), Encoding.BF16: (1, 2),
          Encoding.Q8_0: (32, 34), Encoding.Q1_0: (128, 18),
          Encoding.PQ2_0: (128, 34), Encoding.PTQ1_0: (128, 28)}


def align(n, alignment=ALIGNMENT):
    return (n + alignment - 1) // alignment * alignment


def _uint(n, bits, label):
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < 1 << bits:
        raise ValueError(f'{label}: expected unsigned {bits}-bit integer')


def geometry(encoding, shape):
    """Return (flattened rows, packed bytes per row), with checked u64 sizes."""
    encoding = Encoding(encoding)
    if not 1 <= len(shape) <= 4:
        raise ValueError('tensor rank must be 1..4')
    for dim in shape:
        _uint(dim, 64, 'dimension')
        if not dim:
            raise ValueError('zero tensor dimension')
    group, block_bytes = BLOCKS[encoding]
    if shape[-1] % group:
        raise ValueError('quantization blocks must divide the last dimension')
    rows = math.prod(shape[:-1])
    stride = shape[-1] // group * block_bytes
    if rows * shape[-1] >= 1 << 64 or rows * stride >= 1 << 64:
        raise ValueError('tensor geometry exceeds u64')
    return rows, stride


def _name(name):
    if not isinstance(name, str) or not name or '\0' in name or len(name.encode('utf-8')) > 4096:
        raise ValueError('tensor name must be nonempty UTF-8, at most 4096 bytes, without NUL')


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def _unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('duplicate JSON key')
        out[k] = v
    return out


@dataclass(frozen=True)
class FileSlice:
    path: str | Path
    offset: int
    size: int

    def __post_init__(self):
        _uint(self.offset, 64, 'source offset')
        _uint(self.size, 64, 'source size')
        if not self.size or self.offset + self.size >= 1 << 64:
            raise ValueError('invalid source extent')

    def chunks(self):
        with open(self.path, 'rb') as f:
            f.seek(self.offset)
            left = self.size
            while left:
                data = f.read(min(left, CHUNK_BYTES))
                if not data:
                    raise ValueError('truncated source payload')
                left -= len(data)
                yield data


@dataclass(frozen=True)
class TensorSource:
    name: str
    encoding: Encoding
    shape: tuple[int, ...]
    payload: bytes | FileSlice
    role: int = UNSPECIFIED
    layer: int = UNSPECIFIED

    def __post_init__(self):
        _name(self.name)
        object.__setattr__(self, 'encoding', Encoding(self.encoding))
        object.__setattr__(self, 'shape', tuple(self.shape))
        rows, stride = geometry(self.encoding, self.shape)
        _uint(self.role, 32, 'role')
        _uint(self.layer, 32, 'layer')
        if not isinstance(self.payload, (bytes, FileSlice)):
            raise ValueError('payload must be immutable bytes or a FileSlice')
        size = self.payload.size if isinstance(self.payload, FileSlice) else len(self.payload)
        if size != rows * stride:
            raise ValueError(f'{self.name}: payload size does not match geometry')

    def chunks(self):
        if isinstance(self.payload, FileSlice):
            yield from self.payload.chunks()
        else:
            view = memoryview(self.payload)
            for pos in range(0, len(view), CHUNK_BYTES):
                yield view[pos:pos + CHUNK_BYTES]


@dataclass(frozen=True)
class Tensor:
    id: int
    name: str
    encoding: Encoding
    shape: tuple[int, ...]
    role: int
    layer: int
    offset: int  # File offset only. Never an FPGA address.
    size: int
    sha256: bytes

    @property
    def rows(self):
        return geometry(self.encoding, self.shape)[0]

    @property
    def row_bytes(self):
        return geometry(self.encoding, self.shape)[1]

    def packed(self):
        return ENTRY.pack(self.id, self.encoding, self.role, self.layer, len(self.shape), 0,
                          *(self.shape + (0,) * (4 - len(self.shape))), self.offset,
                          self.size, self.sha256, bytes(24))


def write(path, tensors, model, *, tokenizer_sha256=bytes(32), progress=None):
    """Stream a new file, refusing overwrite. A failed write stays invalid/incomplete.

    Memory use is O(directory + 1 MiB), excluding caller-owned input. Model metadata
    is JSON; inference adapters must validate their required architecture fields.
    This writes a storage artifact, not a claim of engine/model compatibility.
    """
    sources = tuple(tensors)
    if not 1 <= len(sources) <= MAX_TENSORS or len({s.name for s in sources}) != len(sources):
        raise ValueError('missing, duplicate or too many tensors')
    if not isinstance(model, dict) or not isinstance(tokenizer_sha256, bytes) or len(tokenizer_sha256) != 32:
        raise ValueError('model must be an object; tokenizer digest must have 32 bytes')
    metadata = _json({'schema': 1, 'model': model, 'tensor_names': [s.name for s in sources]})
    if len(metadata) > MAX_METADATA:
        raise ValueError('metadata exceeds limit')
    table_offset = align(HEADER.size + len(metadata))
    payload_offset = align(table_offset + ENTRY.size * len(sources))
    entries = []
    # Preflight source/output aliasing even when the output does not exist yet.
    output = Path(path).resolve()
    if any(isinstance(s.payload, FileSlice) and Path(s.payload.path).resolve() == output for s in sources):
        raise ValueError('output aliases an input payload')
    with open(path, 'xb+') as f:
        f.write(bytes(HEADER.size))  # No valid header until every payload is complete.
        f.seek(payload_offset)
        for i, source in enumerate(sources):
            if progress:
                progress(source.name)
            offset, digest, size = f.tell(), hashlib.sha256(), 0
            rows, stride = geometry(source.encoding, source.shape)
            for chunk in source.chunks():
                digest.update(chunk)
                f.write(chunk)
                size += len(chunk)
            if size != rows * stride:
                raise ValueError('source length changed while writing')
            entries.append(Tensor(i, source.name, source.encoding, source.shape, source.role,
                                  source.layer, offset, size, digest.digest()))
            f.write(bytes(align(f.tell()) - f.tell()))
        total = f.tell()
        _uint(total, 64, 'file size')
        table = b''.join(t.packed() for t in entries)
        directory = (metadata + bytes(table_offset - HEADER.size - len(metadata)) + table
                     + bytes(payload_offset - table_offset - len(table)))
        header = HEADER.pack(MAGIC, VERSION, HEADER.size, 0, ENTRY.size, len(entries),
                             len(metadata), table_offset, payload_offset, total,
                             bytes(32), tokenizer_sha256, bytes(8))
        digest = hashlib.sha256(header + directory).digest()
        f.seek(0)
        f.write(header[:DIGEST_OFFSET] + digest + header[DIGEST_OFFSET + 32:])
        f.write(directory)
    return {'version': VERSION, 'directory_sha256': digest.hex(), 'tensors': len(entries),
            'payload_bytes': sum(t.size for t in entries), 'file_bytes': total}


class Reader:
    """Bounded metadata reader; verify() separately checks ALL payload hashes.

    Keep artifacts immutable while using them. Opening alone authenticates internal
    directory integrity, not payload integrity or provenance/signatures. No upload
    interface is exposed here and no v2/v3 channel layout is emulated.
    """
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            raw = f.read(HEADER.size)
            if len(raw) != HEADER.size:
                raise ValueError('truncated BONS v4 header')
            (magic, version, hsize, flags, esize, count, msize, toff, poff, total,
             digest, tokenizer, reserved) = HEADER.unpack(raw)
            if (magic, version, hsize, flags, esize, reserved) != (MAGIC, VERSION, HEADER.size, 0, ENTRY.size, bytes(8)):
                raise ValueError('unsupported BONS v4 header/version/flags')
            if not 1 <= count <= MAX_TENSORS or not 1 <= msize <= MAX_METADATA:
                raise ValueError('invalid BONS v4 directory size')
            if toff != align(HEADER.size + msize) or poff != align(toff + count * ENTRY.size):
                raise ValueError('noncanonical/overlapping directory offsets')
            f.seek(0, 2)
            if f.tell() != total or total < poff:
                raise ValueError('truncated or trailing BONS v4 file bytes')
            f.seek(HEADER.size)
            directory = f.read(poff - HEADER.size)
        if hashlib.sha256(raw[:DIGEST_OFFSET] + bytes(32) + raw[DIGEST_OFFSET + 32:] + directory).digest() != digest:
            raise ValueError('BONS v4 directory SHA256 mismatch')
        metadata = json.loads(directory[:msize], object_pairs_hook=_unique_object)
        if (not isinstance(metadata, dict) or set(metadata) != {'schema', 'model', 'tensor_names'}
                or type(metadata['schema']) is not int or metadata['schema'] != 1
                or not isinstance(metadata['model'], dict) or not isinstance(metadata['tensor_names'], list)
                or len(metadata['tensor_names']) != count or _json(metadata) != directory[:msize]):
            raise ValueError('unsupported/noncanonical BONS v4 metadata')
        names = metadata['tensor_names']
        for name in names:
            _name(name)
        if len(set(names)) != count:
            raise ValueError('duplicate tensor names')
        table = directory[toff - HEADER.size:toff - HEADER.size + count * ENTRY.size]
        if any(directory[msize:toff - HEADER.size]) or any(directory[toff - HEADER.size + len(table):]):
            raise ValueError('nonzero directory padding')
        entries, next_offset = [], poff
        for i in range(count):
            fields = ENTRY.unpack_from(table, i * ENTRY.size)
            tid, encoding, role, layer, rank, flags = fields[:6]
            dims, offset, size, sha, reserved = fields[6:10], *fields[10:]
            if tid != i or not 1 <= rank <= 4 or flags or any(dims[rank:]) or any(reserved):
                raise ValueError('unsupported/noncanonical tensor entry')
            shape = dims[:rank]
            rows, stride = geometry(encoding, shape)
            if offset != next_offset or size != rows * stride or size > total - offset:
                raise ValueError('inconsistent/overlapping tensor payload')
            next_offset = align(offset + size)
            if next_offset > total:
                raise ValueError('truncated payload alignment')
            entries.append(Tensor(i, names[i], Encoding(encoding), shape, role, layer, offset, size, sha))
        if next_offset != total:
            raise ValueError('unclaimed file bytes')
        self.entries = tuple(entries)
        self.by_name = {e.name: e for e in entries}
        self.model = metadata['model']
        self.directory_sha256 = digest
        self.tokenizer_sha256 = tokenizer
        self.file_bytes = total

    def iter_tensor(self, name):
        t = self.by_name[name]
        yield from FileSlice(self.path, t.offset, t.size).chunks()

    def verify(self):
        for t in self.entries:
            digest = hashlib.sha256()
            for chunk in self.iter_tensor(t.name):
                digest.update(chunk)
            if digest.digest() != t.sha256:
                raise ValueError(f'{t.name}: payload SHA256 mismatch')
            with self.path.open('rb') as f:
                f.seek(t.offset + t.size)
                padding = f.read(align(t.offset + t.size) - t.offset - t.size)
                if len(padding) != align(t.offset + t.size) - t.offset - t.size or any(padding):
                    raise ValueError('truncated/nonzero payload padding')
        if self.path.stat().st_size != self.file_bytes:
            raise ValueError('file size changed during verification')
        return True

    def read_rows(self, name, start, count):
        """Logical row access; callers bound count (e.g. one embedding lookup)."""
        t = self.by_name[name]
        _uint(start, 64, 'row start')
        _uint(count, 64, 'row count')
        if not count or start >= t.rows or count > t.rows - start:
            raise ValueError('row range out of bounds')
        with self.path.open('rb') as f:
            f.seek(t.offset + start * t.row_bytes)
            data = f.read(count * t.row_bytes)
        if len(data) != count * t.row_bytes:
            raise ValueError('truncated row payload')
        return data

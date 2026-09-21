# SPDX-License-Identifier: BSD-2-Clause
"""Lossless GGUF -> logical BONS v4. No board selection or quantization changes."""
from dataclasses import asdict
import hashlib
from pathlib import Path

from trixium.model.bons4 import Encoding, FileSlice, TensorSource, geometry, write


def _jsonable(value):
    # GGUF metadata includes numpy scalars/arrays. Keep all non-tokenizer metadata,
    # including unknown architecture/MTP keys, rather than silently dropping it.
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return _jsonable(value.tolist())
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while data := f.read(1024 * 1024):
            digest.update(data)
    return digest.digest()


def native_sources(reader):
    """Keep every tensor, including unrecognized roles; remove only file padding.

    Type 42 is ambiguous in Prism/mainline files. Recognize g128 only if its exact
    or alignment-padded extent matches uniquely; never reinterpret g64 as g128.
    Padding contents do not participate in model semantics and may be nonzero.
    """
    from trixium.model.convert import _name_role
    alignment = reader.metadata.get('general.alignment', 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError('invalid GGUF alignment')
    sources = []
    for t in reader.tensors.values():
        shape = tuple(int(d) for d in t.shape)
        encoding = Encoding.PQ2_0 if t.type_id == 42 else Encoding(t.type_id)
        rows, stride = geometry(encoding, shape)
        size = rows * stride
        if t.type_id == 42:
            size64 = rows * shape[-1] // 64 * 18
            if size64 <= t.nbytes < size64 + alignment:
                raise ValueError(f'{t.name}: ambiguous/group-64 type 42; use explicit PQ2_0 g128')
        if not size <= t.nbytes < size + alignment:
            raise ValueError(f'{t.name}: GGUF extent does not match native block geometry')
        if (t.file_offset < reader.data_start or t.file_offset % alignment
                or t.file_offset + t.nbytes > len(reader._data)):
            raise ValueError(f'{t.name}: GGUF extent out of bounds')
        try:
            layer, role, _ = _name_role(t.name)
        except (KeyError, ValueError):
            layer = role = 0xffffffff
        sources.append(TensorSource(t.name, encoding, shape, FileSlice(reader.path, t.file_offset, size),
                                    role=role, layer=layer))
    return sources


def pack(gguf, out, *, tokenizer=None, check=False, progress=None):
    from trixium.model.gguf_reader import GGUFReader
    from trixium.model.golden import config_from_gguf
    from trixium.model.bons4 import Reader
    reader = GGUFReader(gguf)
    sources = native_sources(reader)  # Preflight unsupported types before creating output.
    cfg = config_from_gguf(reader)
    model = {'config': _jsonable(asdict(cfg)),
             'gguf_metadata': _jsonable({k: v for k, v in reader.metadata.items()
                                         if not k.startswith('tokenizer.')}),
             'source': {'format': 'GGUF', 'name': Path(gguf).name,
                        'bytes': len(reader._data), 'sha256': file_sha256(gguf).hex()}}
    token_sha = file_sha256(tokenizer) if tokenizer else bytes(32)
    result = write(out, sources, model, tokenizer_sha256=token_sha, progress=progress)
    if check:
        Reader(out).verify()
    return result

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# GGUFReader unit test: writes a small synthetic GGUF v3 file (with an unknown quant type to
# exercise the offset-delta size derivation) and checks metadata + tensor parsing.

import struct

import numpy as np

from trixium.model.gguf_reader import GGUFReader

ALIGNMENT = 32

# Helpers ------------------------------------------------------------------------------------------

def _string(s):
    return struct.pack("<Q", len(s)) + s.encode()

def _kv(key, vtype, payload):
    return _string(key) + struct.pack("<I", vtype) + payload

def _tensor_info(name, dims, type_id, offset):
    # dims are written innermost-first (GGUF convention).
    out = _string(name) + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<II", type_id, 0)[0:4]
    out += struct.pack("<Q", offset)
    return out

def write_tiny_gguf(path):
    t0 = np.arange(64, dtype=np.float32)          # F32 tensor, 2 x 32.
    t1 = np.arange(96, dtype=np.uint8)            # "unknown type 99" raw payload.

    header = struct.pack("<IIQQ", 0x46554747, 3, 2, 4)
    meta  = _kv("general.architecture", 8, _string("qwen3"))
    meta += _kv("general.alignment",    4, struct.pack("<I", ALIGNMENT))
    meta += _kv("qwen3.block_count",    4, struct.pack("<I", 2))
    # Array of u32.
    meta += _kv("test.array", 9, struct.pack("<IQ", 4, 3) + struct.pack("<III", 7, 8, 9))

    # Tensor infos: t1 offset must be aligned.
    t1_off = (t0.nbytes + ALIGNMENT - 1)//ALIGNMENT*ALIGNMENT
    infos  = _tensor_info("a.weight", (32, 2), 0,  0)       # F32, numpy shape (2, 32).
    infos += _tensor_info("b.weight", (96,),   99, t1_off)  # Unknown type id.

    blob = header + meta + infos
    data_start = (len(blob) + ALIGNMENT - 1)//ALIGNMENT*ALIGNMENT
    blob += b"\x00"*(data_start - len(blob))
    blob += t0.tobytes()
    blob += b"\x00"*(t1_off - t0.nbytes)
    blob += t1.tobytes()

    with open(path, "wb") as f:
        f.write(blob)

# Tests --------------------------------------------------------------------------------------------

def test_gguf_reader(tmp_path):
    path = tmp_path/"tiny.gguf"
    write_tiny_gguf(path)
    r = GGUFReader(str(path))

    # Metadata.
    assert r.metadata["general.architecture"] == "qwen3"
    assert r.metadata["qwen3.block_count"] == 2
    assert list(r.metadata["test.array"]) == [7, 8, 9]

    # Tensors.
    a = r.tensors["a.weight"]
    assert a.shape == (2, 32)          # Reversed to numpy order.
    assert a.type_name == "F32"
    assert a.nbytes >= 64*4            # Offset-delta includes alignment padding.
    data = np.frombuffer(r.tensor_bytes("a.weight")[:64*4], dtype=np.float32)
    assert np.array_equal(data, np.arange(64, dtype=np.float32))

    b = r.tensors["b.weight"]
    assert b.type_name == "UNK99"
    assert b.nbytes == 96              # Last tensor extends to EOF.
    assert np.array_equal(np.asarray(r.tensor_bytes("b.weight")), np.arange(96, dtype=np.uint8))

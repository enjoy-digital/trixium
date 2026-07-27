#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Minimal pure-numpy GGUF v3 reader.

Reads GGUF metadata and raw tensor bytes without depending on the `gguf` package: the PrismML
llama.cpp fork's Q2_0 (g128) quantization uses a tensor type id unknown to mainline tooling, so
tensor payload sizes are derived from offset deltas rather than from a per-type size table.
Tensor data is exposed as numpy memmap slices (nothing is loaded eagerly).

Run `python -m trixium.model.gguf_reader <file.gguf>` to inspect a file.
"""

import sys

import numpy as np

# GGUF constants -----------------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747 # "GGUF" little-endian.

# Metadata value types.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL, \
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = range(13)

_SCALAR_FMT = {
    _T_UINT8   : np.uint8,   _T_INT8  : np.int8,
    _T_UINT16  : np.uint16,  _T_INT16 : np.int16,
    _T_UINT32  : np.uint32,  _T_INT32 : np.int32,
    _T_UINT64  : np.uint64,  _T_INT64 : np.int64,
    _T_FLOAT32 : np.float32, _T_FLOAT64 : np.float64,
}

# Known ggml tensor type ids (mainline llama.cpp); unknown ids are reported numerically.
GGML_TYPE_NAMES = {
     0: "F32",     1: "F16",     2: "Q4_0",   3: "Q4_1",   6: "Q5_0",  7: "Q5_1",
     8: "Q8_0",    9: "Q8_1",   10: "Q2_K",  11: "Q3_K",  12: "Q4_K", 13: "Q5_K",
    14: "Q6_K",   15: "Q8_K",   24: "I8",    25: "I16",   26: "I32",  27: "I64",
    28: "F64",    30: "BF16",   34: "TQ1_0", 35: "TQ2_0",
}

# GGUFTensor ---------------------------------------------------------------------------------------

class GGUFTensor:
    """One tensor entry: logical shape (numpy order), raw type id and byte extent in the file."""
    def __init__(self, name, shape, type_id, file_offset, nbytes):
        self.name        = name
        self.shape       = shape       # Numpy order: shape[-1] is the contiguous (row) dimension.
        self.type_id     = type_id
        self.type_name   = GGML_TYPE_NAMES.get(type_id, f"UNK{type_id}")
        self.file_offset = file_offset # Absolute offset in the file.
        self.nbytes      = nbytes

    @property
    def n_elements(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bits_per_weight(self):
        return 8*self.nbytes/self.n_elements

    def __repr__(self):
        return f"<GGUFTensor {self.name} {self.type_name} {self.shape} {self.nbytes}B>"

# GGUFReader ---------------------------------------------------------------------------------------

class GGUFReader:
    def __init__(self, path):
        self.path     = path
        self.metadata = {}
        self.tensors  = {}
        self._data    = np.memmap(path, dtype=np.uint8, mode="r")
        self._parse()

    # Low-level cursor reads.
    def _u32(self):
        v = int(np.frombuffer(self._data[self._pos:self._pos+4], dtype=np.uint32)[0])
        self._pos += 4
        return v

    def _u64(self):
        v = int(np.frombuffer(self._data[self._pos:self._pos+8], dtype=np.uint64)[0])
        self._pos += 8
        return v

    def _scalar(self, vtype):
        dtype = _SCALAR_FMT[vtype]
        size  = np.dtype(dtype).itemsize
        v     = np.frombuffer(self._data[self._pos:self._pos+size], dtype=dtype)[0]
        self._pos += size
        if vtype in (_T_FLOAT32, _T_FLOAT64):
            return float(v)
        return int(v)

    def _string(self):
        n = self._u64()
        s = bytes(self._data[self._pos:self._pos+n]).decode("utf-8")
        self._pos += n
        return s

    def _value(self, vtype):
        if vtype == _T_STRING:
            return self._string()
        if vtype == _T_BOOL:
            v = int(self._data[self._pos])
            self._pos += 1
            return bool(v)
        if vtype == _T_ARRAY:
            etype = self._u32()
            count = self._u64()
            # Fast path for scalar arrays (token/merge tables can be huge).
            if etype in _SCALAR_FMT:
                dtype = _SCALAR_FMT[etype]
                size  = np.dtype(dtype).itemsize
                arr   = np.frombuffer(self._data[self._pos:self._pos+size*count], dtype=dtype)
                self._pos += size*count
                return arr
            return [self._value(etype) for _ in range(count)]
        return self._scalar(vtype)

    def _parse(self):
        self._pos = 0
        magic     = self._u32()
        assert magic == GGUF_MAGIC, f"not a GGUF file: magic {magic:#x}"
        version = self._u32()
        assert version >= 2, f"unsupported GGUF version {version}"
        n_tensors = self._u64()
        n_kv      = self._u64()

        # Metadata key/values.
        for _ in range(n_kv):
            key   = self._string()
            vtype = self._u32()
            self.metadata[key] = self._value(vtype)

        # Tensor infos.
        infos = []
        for _ in range(n_tensors):
            name    = self._string()
            n_dims  = self._u32()
            dims    = [self._u64() for _ in range(n_dims)]
            type_id = self._u32()
            offset  = self._u64()
            # GGUF stores dims innermost-first; reverse for numpy convention.
            infos.append((name, tuple(reversed(dims)), type_id, offset))

        # Data section starts aligned after the header/infos.
        alignment       = int(self.metadata.get("general.alignment", 32))
        self.data_start = (self._pos + alignment - 1)//alignment*alignment

        # Payload sizes from offset deltas (robust to unknown quant types).
        infos_by_offset = sorted(infos, key=lambda t: t[3])
        file_size       = len(self._data)
        for i, (name, shape, type_id, offset) in enumerate(infos_by_offset):
            if i + 1 < len(infos_by_offset):
                end = self.data_start + infos_by_offset[i+1][3]
            else:
                end = file_size
            nbytes = end - (self.data_start + offset)
            self.tensors[name] = GGUFTensor(name, shape, type_id, self.data_start + offset, nbytes)

    def tensor_bytes(self, name):
        """Raw payload of a tensor as a numpy uint8 memmap slice (zero-copy)."""
        t = self.tensors[name]
        return self._data[t.file_offset:t.file_offset+t.nbytes]

# Inspect CLI --------------------------------------------------------------------------------------

def inspect(path, show_metadata=True):
    r = GGUFReader(path)
    if show_metadata:
        print(f"== {path}")
        for k, v in r.metadata.items():
            if isinstance(v, np.ndarray):
                v = f"array[{len(v)}] {v[:4]}..."
            elif isinstance(v, list) and len(v) > 4:
                v = f"list[{len(v)}] {v[:4]}..."
            elif isinstance(v, str) and len(v) > 80:
                v = v[:77] + "..."
            print(f"  {k:56s} {v}")
    print(f"  {'tensor':44s} {'type':8s} {'shape':20s} {'bytes':>12s} {'bpw':>6s}")
    types = {}
    for t in r.tensors.values():
        print(f"  {t.name:44s} {t.type_name:8s} {str(t.shape):20s} {t.nbytes:12d} {t.bits_per_weight:6.3f}")
        types.setdefault(t.type_name, [0, 0])
        types[t.type_name][0] += 1
        types[t.type_name][1] += t.nbytes
    print("  -- per-type summary:")
    for name, (count, nbytes) in sorted(types.items()):
        print(f"  {name:10s} {count:4d} tensors {nbytes/1e6:10.1f} MB")
    return r

if __name__ == "__main__":
    for f in sys.argv[1:]:
        inspect(f)

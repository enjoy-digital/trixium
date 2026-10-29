#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Rewrite a PrismML ternary GGUF with its PTQ1_0 / PQ2_0 tensors as Q8_0, for upstream llama.cpp.

Upstream llama.cpp has the Hadamard v1/v2 graph of the Ternary-Bonsai-2 models but not their
PrismML tensor types. A ternary group of 128 with one fp16 scale is exactly four Q8_0 blocks
(d = the scale, qs = code - 1), so the rewrite is lossless: the evaluation tools (perplexity,
KV-cache formats) then run on the same weights. The metadata is copied byte for byte.

    python3 -m software.lab.gguf_to_q8 models/2b-v2/Ternary-Bonsai-2-2B-PTQ1_0.gguf /dev/shm/2b-q8.gguf
"""

import argparse
import struct

import numpy as np

from model import qformats
from model.gguf_reader import GGUFReader

GGML_Q8_0  = 8
PTQ1_0_ID  = 143

def kv_extent(path):
    """Byte range of the metadata key/values (to copy them verbatim) and the tensor order."""
    r = GGUFReader(path)
    # Re-walk the header to find where the KVs end (GGUFReader keeps only the values).
    r._pos = 0
    r._u32(); r._u32(); n_t = r._u64(); n_kv = r._u64()
    start = r._pos
    for _ in range(n_kv):
        r._string(); vt = r._u32(); r._value(vt)
    end = r._pos
    names = []
    for _ in range(n_t):
        names.append(r._string()); nd = r._u32(); [r._u64() for _ in range(nd)]; r._u32(); r._u64()
    return r, start, end, n_kv, names

def to_q8(t, raw):
    if t.type_id == PTQ1_0_ID:
        scales, codes = qformats.ptq1_0_unpack(raw)
    else:
        raise ValueError(f"{t.name}: no lossless Q8_0 rewrite for type {t.type_name}")
    nb = codes.shape[0]*4
    out = np.empty((nb, 34), np.uint8)
    out[:, 0:2] = np.repeat(scales.astype(np.float16), 4).view(np.uint8).reshape(nb, 2)
    out[:, 2:]  = (codes.astype(np.int16) - 1).astype(np.int8).view(np.uint8).reshape(nb, 32)
    return out.reshape(-1)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("dst")
    a = p.parse_args()

    r, kv0, kv1, n_kv, names = kv_extent(a.src)
    align = int(r.metadata.get("general.alignment", 32))
    pad = lambda n: (n + align - 1)//align*align
    plan = []                                    # (name, shape, type, payload callable)
    for n in names:
        t = r.tensors[n]
        if t.type_id == PTQ1_0_ID:
            plan.append((t, GGML_Q8_0, t.n_elements//32*34))
        else:
            plan.append((t, t.type_id, t.nbytes))
    head = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(plan), n_kv))
    head += bytes(r._data[kv0:kv1])
    off = 0
    offsets = []
    for t, ty, nb in plan:
        dims = list(reversed(t.shape))
        head += struct.pack("<Q", len(t.name.encode())) + t.name.encode()
        head += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", ty, off)
        offsets.append(off)
        off = pad(off + nb)
    with open(a.dst, "wb") as f:
        f.write(head); f.write(b"\0"*(pad(len(head)) - len(head)))
        base = pad(len(head))
        for (t, ty, nb), o in zip(plan, offsets):
            f.seek(base + o)
            raw = r.tensor_bytes(t.name)
            f.write(to_q8(t, raw).tobytes() if ty != t.type_id else bytes(raw[:nb]))
        f.truncate(base + off)
    print(f"{a.dst}: {len(plan)} tensors, {sum(1 for t, ty, _ in plan if ty != t.type_id)} rewritten as Q8_0")

if __name__ == "__main__":
    main()

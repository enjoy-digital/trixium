#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BONS weight image writer/reader (format spec in trixium.layout).

The writer takes decoded tensors (scales/codes/values as numpy arrays), lays every tensor out
as hardware row packets, block-shards it across N channels and emits a single image file +
JSON manifest. The reader parses an image back (used by tests, the golden model and the sim
image builder) - what the reader returns is byte-derived from the exact payload the hardware
streams, so golden results computed from it are bit-faithful to what the engine sees.
"""

import json
import zlib

import numpy as np

from trixium import layout
from trixium.layout import (
    BEAT_BYTES, GROUP_TERNARY, GROUP_Q8, SCALES_PER_BEAT,
    KIND_TERNARY, KIND_Q8, KIND_F16, KIND_F32, KIND_NORM16, KIND_NAMES, ROLE_NAMES,
    FLAG_REPLICATED, ceil_div, scale_beats, code_beats, row_beats, shard_rows,
)

# TensorSpec ---------------------------------------------------------------------------------------

class TensorSpec:
    """One tensor to place in the image.

    - KIND_TERNARY: scales (rows, cols//128) float16, codes (rows, cols) uint8 in {0,1,2}.
    - KIND_Q8:      scales (rows, cols//32)  float16, values (rows, cols) int8.
    - KIND_F16/F32: values (rows, cols).
    """
    def __init__(self, name, role, layer, kind, rows, cols, scales=None, values=None, flags=0):
        self.name   = name
        self.role   = role
        self.layer  = layer
        self.kind   = kind
        self.rows   = rows
        self.cols   = cols
        self.scales = scales
        self.values = values
        self.flags  = flags
        if kind in (KIND_TERNARY, KIND_Q8):
            group = layout.KIND_GROUP[kind]
            assert cols % group == 0
            assert scales.shape == (rows, cols//group), (name, scales.shape)
            # Per-tensor accumulator scale bias / result shift (bitexact contract).
            from trixium.model import bitexact as bx
            _, e16      = bx.fp16_decompose(np.asarray(scales, dtype=np.float16))
            gs_max      = bx.GS_MAX if kind == KIND_TERNARY else bx.GS_MAX_Q8
            self.bias   = bx.choose_scale_bias(e16)
            self.rshift = bx.choose_result_shift(scales, self.bias, gs_max=gs_max)
        else:
            self.bias   = 0
            self.rshift = 0
        assert values.shape == (rows, cols), (name, values.shape)

# Row packet encoding ------------------------------------------------------------------------------

def encode_rows(spec, row0, nrows):
    """Encode rows [row0, row0+nrows) of a tensor as row-packet bytes -> (nrows*row_beats*32,)."""
    sb = scale_beats(spec.cols, spec.kind)
    cb = code_beats(spec.cols, spec.kind)
    out = np.zeros((nrows, (sb + cb)*BEAT_BYTES), dtype=np.uint8)

    if spec.kind in (KIND_TERNARY, KIND_Q8):
        scales = np.ascontiguousarray(spec.scales[row0:row0+nrows].astype(np.float16))
        sbytes = scales.view(np.uint8).reshape(nrows, -1)
        out[:, :sbytes.shape[1]] = sbytes

    v = spec.values[row0:row0+nrows]
    if spec.kind == KIND_TERNARY:
        codes = np.ascontiguousarray(v.astype(np.uint8)).reshape(nrows, spec.cols//4, 4)
        packed = (codes[:, :, 0] | (codes[:, :, 1] << 2) |
                  (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)).astype(np.uint8)
        out[:, sb*BEAT_BYTES:sb*BEAT_BYTES + packed.shape[1]] = packed
    elif spec.kind == KIND_Q8:
        vbytes = np.ascontiguousarray(v.astype(np.int8)).view(np.uint8)
        out[:, sb*BEAT_BYTES:sb*BEAT_BYTES + spec.cols] = vbytes
    elif spec.kind == KIND_F16:
        vbytes = np.ascontiguousarray(v.astype(np.float16)).view(np.uint8).reshape(nrows, -1)
        out[:, sb*BEAT_BYTES:sb*BEAT_BYTES + vbytes.shape[1]] = vbytes
    elif spec.kind == KIND_F32:
        vbytes = np.ascontiguousarray(v.astype(np.float32)).view(np.uint8).reshape(nrows, -1)
        out[:, sb*BEAT_BYTES:sb*BEAT_BYTES + vbytes.shape[1]] = vbytes
    elif spec.kind == KIND_NORM16:
        vbytes = np.ascontiguousarray(v.astype("<i2")).view(np.uint8).reshape(nrows, -1)
        out[:, sb*BEAT_BYTES:sb*BEAT_BYTES + vbytes.shape[1]] = vbytes
    else:
        raise ValueError(spec.kind)
    return out.reshape(-1)

def decode_rows(spec_entry, payload):
    """Inverse of encode_rows for one channel's slice: entry dict + bytes -> (scales, values)."""
    kind  = spec_entry["kind"]
    cols  = spec_entry["cols"]
    sb    = scale_beats(cols, kind)
    cb    = code_beats(cols, kind)
    rb    = (sb + cb)*BEAT_BYTES
    data  = np.frombuffer(payload, dtype=np.uint8).reshape(-1, rb)
    nrows = data.shape[0]

    scales = None
    if kind in (KIND_TERNARY, KIND_Q8):
        group  = layout.KIND_GROUP[kind]
        scales = data[:, :2*(cols//group)].copy().view(np.float16)

    body = data[:, sb*BEAT_BYTES:]
    if kind == KIND_TERNARY:
        qs     = body[:, :cols//4]
        values = np.stack([(qs >> (2*j)) & 3 for j in range(4)], axis=-1).reshape(nrows, cols)
    elif kind == KIND_Q8:
        values = body[:, :cols].copy().view(np.int8)
    elif kind == KIND_F16:
        values = body[:, :2*cols].copy().view(np.float16)
    elif kind == KIND_F32:
        values = body[:, :4*cols].copy().view(np.float32)
    elif kind == KIND_NORM16:
        values = body[:, :2*cols].copy().view(np.int16)
    else:
        raise ValueError(kind)
    return scales, values

# Writer -------------------------------------------------------------------------------------------

def write_image(cfg, tensors, n_channels, path, tokenizer_sha256=b"", manifest_path=None):
    """Write a BONS image + JSON manifest. Returns the manifest dict."""
    chan_payloads = [bytearray() for _ in range(n_channels)]
    entries       = []
    manifest_tensors = []

    # Every F32 tensor (norm weights) gets an int16 companion at NORM16_EXP so the vector
    # unit's rmsnorm can fetch it straight into the scratchpad (same role/layer; ".w16"
    # name - the entry name field is 24 bytes; consumers select by kind).
    tensors = list(tensors)
    for spec in [t for t in tensors if t.kind == KIND_F32]:
        q = np.clip(np.round(spec.values.astype(np.float64)*2.0**(-layout.NORM16_EXP)),
                    -32767, 32767).astype(np.int16)
        n16_name = spec.name.replace(".weight", "") + ".w16"
        tensors.append(TensorSpec(n16_name, spec.role, spec.layer,
                                  layout.KIND_NORM16, spec.rows, spec.cols,
                                  values=q, flags=spec.flags))

    for spec in tensors:
        base_beats = len(chan_payloads[0])//BEAT_BYTES
        replicated = bool(spec.flags & FLAG_REPLICATED)
        crc = 0 # CRC over the channel-major concatenation of this tensor's payload.
        for c in range(n_channels):
            if replicated:
                row0, nrows = 0, spec.rows
            else:
                row0, nrows = shard_rows(spec.rows, n_channels, c)
            chunk = encode_rows(spec, row0, nrows)
            crc   = zlib.crc32(chunk.tobytes(), crc) & 0xFFFFFFFF
            chan_payloads[c].extend(chunk.tobytes())
        len_beats = len(chan_payloads[0])//BEAT_BYTES - base_beats
        entries.append(layout.pack_entry(
            spec.role, spec.kind, spec.flags, spec.layer, spec.rows, spec.cols,
            base_beats, len_beats, crc, spec.bias, spec.rshift, spec.name))
        manifest_tensors.append({
            "name": spec.name, "role": ROLE_NAMES[spec.role], "layer": spec.layer,
            "kind": KIND_NAMES[spec.kind], "flags": spec.flags,
            "rows": spec.rows, "cols": spec.cols,
            "chan_base_beats": base_beats, "chan_len_beats": len_beats, "crc32": crc,
            "bias": spec.bias, "rshift": spec.rshift,
        })
        # All channels must stay in lockstep.
        assert all(len(p) == len(chan_payloads[0]) for p in chan_payloads), spec.name

    payload_beats = len(chan_payloads[0])//BEAT_BYTES
    table         = b"".join(entries)
    table_crc     = layout.crc32(table)
    header        = layout.pack_header(cfg, n_channels, len(tensors), payload_beats,
                                       table_crc, tokenizer_sha256)
    table_padded  = table.ljust(
        ceil_div(len(table), layout.BONS_HEADER_SIZE)*layout.BONS_HEADER_SIZE, b"\x00")

    with open(path, "wb") as f:
        f.write(header)
        f.write(table_padded)
        for p in chan_payloads:
            f.write(bytes(p))

    manifest = {
        "model"          : cfg.name,
        "config"         : cfg.__dict__,
        "n_channels"     : n_channels,
        "payload_beats"  : payload_beats,
        "payload_bytes_per_channel": payload_beats*BEAT_BYTES,
        "table_crc32"    : table_crc,
        "tokenizer_sha256": tokenizer_sha256.hex() if tokenizer_sha256 else "",
        "tensors"        : manifest_tensors,
    }
    if manifest_path is not None:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    return manifest

# Reader -------------------------------------------------------------------------------------------

class BONSReader:
    def __init__(self, path):
        self.data   = np.memmap(path, dtype=np.uint8, mode="r")
        self.header = layout.unpack_header(bytes(self.data[:layout.BONS_HEADER_SIZE]))
        self.entries = []
        table_off = self.header["table_offset"]
        for i in range(self.header["n_tensors"]):
            self.entries.append(layout.unpack_entry(
                bytes(self.data[table_off + i*layout.BONS_ENTRY_SIZE:
                                table_off + (i+1)*layout.BONS_ENTRY_SIZE])))
        self.by_name = {e["name"]: e for e in self.entries}

    def channel_payload(self, channel):
        """Full payload bytes of one channel (what gets uploaded to that pseudo-channel)."""
        h     = self.header
        start = h["payload_offset"] + channel*h["payload_beats"]*BEAT_BYTES
        return self.data[start:start + h["payload_beats"]*BEAT_BYTES]

    def tensor_channel_bytes(self, name, channel):
        e  = self.by_name[name]
        ch = self.channel_payload(channel)
        return ch[e["chan_base_beats"]*BEAT_BYTES:
                  (e["chan_base_beats"] + e["chan_len_beats"])*BEAT_BYTES]

    def tensor(self, name):
        """Reassemble a full tensor from its shards -> (scales, values) numpy arrays."""
        e = self.by_name[name]
        h = self.header
        if e["flags"] & FLAG_REPLICATED:
            return decode_rows(e, self.tensor_channel_bytes(name, 0))
        parts = [decode_rows(e, self.tensor_channel_bytes(name, c))
                 for c in range(h["n_channels"])]
        scales = None
        if parts[0][0] is not None:
            scales = np.concatenate([p[0] for p in parts], axis=0)
        values = np.concatenate([p[1] for p in parts], axis=0)
        return scales, values

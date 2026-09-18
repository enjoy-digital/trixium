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
from trixium.layout import BONS_NAME_BYTES
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
    def __init__(self, name, role, layer, kind, rows, cols, scales=None, values=None, flags=0,
                 loader=None, bias=None):
        self.name   = name
        self.role   = role
        self.layer  = layer
        self.kind   = kind
        self.rows   = rows
        self.cols   = cols
        self.scales = scales
        self.values = values
        self.flags  = flags
        # loader: a callable returning `values` on demand, for the streaming writer - the
        # 1.7B's codes are 1.7 GB and the whole image another 1.1, and the old writer held
        # both plus every temporary (14.5 GB peak); the 8B could not convert at all.
        self.loader = loader
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
            # NORM16 norm companions: bias = the tensor's exponent (int8, two's complement).
            self.bias   = (bias & 0xFF) if bias is not None else 0
            self.rshift = 0
        if values is not None:
            assert values.shape == (rows, cols), (name, values.shape)
        else:
            assert loader is not None, name

    def get_values(self):
        """The (rows, cols) array; loaded through `loader` when the spec is lazy (not cached:
        the caller encodes every channel from one load and drops it)."""
        return self.values if self.values is not None else self.loader()

# Row packet encoding ------------------------------------------------------------------------------

def encode_rows(spec, row0, nrows, values=None):
    """Encode rows [row0, row0+nrows) of a tensor as row-packet bytes -> (nrows*row_beats*32,).
    `values` overrides spec.values (the streaming writer passes one load for all channels)."""
    sb = scale_beats(spec.cols, spec.kind)
    cb = code_beats(spec.cols, spec.kind)
    out = np.zeros((nrows, (sb + cb)*BEAT_BYTES), dtype=np.uint8)

    if spec.kind in (KIND_TERNARY, KIND_Q8):
        scales = np.ascontiguousarray(spec.scales[row0:row0+nrows].astype(np.float16))
        sbytes = scales.view(np.uint8).reshape(nrows, -1)
        out[:, :sbytes.shape[1]] = sbytes

    if values is None:
        values = spec.get_values()
    v = values[row0:row0+nrows]
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

def embed_bfp_rows(spec):
    from trixium.model import bitexact as bx
    """Every embedding row in the firmware's int16 BFP form: (mantissas int16 (rows, cols),
    exponents int32 (rows,)). Vectorized transcription of embed_row_bfp (firmware/infer.c)
    and the sim reference embed_bfp_int (test_sim_smoke.py): per group g the fp16 scale
    splits into (m11, e16); every group's (q-1)*m11 is aligned to the row's smallest e16
    (shift capped at 20); the row's absmax over groups with any nonzero code sets
    k2 = bitlen(absmax) - 15 so mantissas land in [2^14, 2^15); net = sh - k2 shifts left
    or arithmetic-right; the result is truncated to int16."""
    m11, e16 = bx.fp16_decompose(spec.scales.astype(np.float16))          # (rows, G)
    m11 = m11.astype(np.int64); e16 = e16.astype(np.int64)
    rows, cols = spec.values.shape
    G = cols//layout.GROUP_TERNARY
    q = spec.values.astype(np.int64).reshape(rows, G, layout.GROUP_TERNARY) - 1
    e16min = e16.min(axis=1, keepdims=True)
    sh = np.minimum(e16 - e16min, 20)                                       # (rows, G)
    nonzero = np.any(q != 0, axis=2)                                         # (rows, G)
    gm = np.where(nonzero, np.abs(m11) << sh, 0)
    amax = gm.max(axis=1)                                                    # (rows,)
    bl = np.zeros(rows, dtype=np.int64)
    nz = amax > 0
    bl[nz] = np.floor(np.log2(amax[nz].astype(np.float64))).astype(np.int64) + 1
    # log2 rounding guard: exact bit_length check for the values near a power of two.
    bl[nz] = np.where((1 << bl[nz]) <= amax[nz], bl[nz] + 1, bl[nz])
    bl[nz] = np.where((1 << (bl[nz] - 1)) > amax[nz], bl[nz] - 1, bl[nz])
    k2 = bl - 15
    net = sh - k2[:, None]                                                    # (rows, G)
    v2 = q*m11[:, :, None]                                                   # (rows, G, 128)
    netc = net[:, :, None]
    out = np.where(netc >= 0, v2 << np.maximum(netc, 0), v2 >> np.maximum(-netc, 0))
    mant = out.reshape(rows, cols).astype(np.int16)                         # truncation, as the firmware
    exps = (e16min[:, 0] + k2).astype(np.int32)
    return mant, exps

def _tensor_beats(spec, n_channels):
    """Per-channel payload beats of a tensor: every channel carries the same count (the
    sharding divides rows exactly; replicated tensors carry all rows everywhere)."""
    nrows = spec.rows if (spec.flags & FLAG_REPLICATED) else shard_rows(spec.rows, n_channels, 0)[1]
    return nrows*(scale_beats(spec.cols, spec.kind) + code_beats(spec.cols, spec.kind))


def restored_bfp_rows(scales, codes, signs, hblock):
    """A rotated embedding block (PrismML's basis) dequantized, restored to the plain basis
    (e = s * (R z), golden.hadamard_restore) and quantized to the firmware's int16 BFP rows:
    per row the exponent puts the absmax in [2^14, 2^15), mantissas rounded to nearest."""
    from trixium.model.golden import hadamard_restore
    rows, cols = codes.shape
    G = cols//layout.GROUP_TERNARY
    z = (codes.astype(np.float32).reshape(rows, G, layout.GROUP_TERNARY) - 1.0) \
        * scales.astype(np.float32)[:, :, None]
    e = hadamard_restore(z.reshape(rows, cols), signs, hblock).astype(np.float64)
    amax = np.abs(e).max(axis=1)
    exps = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30))).astype(np.int64) - 14, 0)
    # A row's absmax lands in [2^14, 2^15) before rounding; rounding can push it to 2^15
    # exactly (one LSB of 2^15 = 3e-5 relative), which the int16 cannot hold: clip.
    mant = np.clip(np.round(e*2.0**(-exps[:, None])), -32767, 32767)
    return mant.astype(np.int16), exps.astype(np.int32)


def _emb16_specs(emb, n_channels, block=8192, restore=None):
    """The token embedding pre-dequantized to int16 BFP rows (ROLE_TOK_EMB16) plus the per-row
    exponent (ROLE_TOK_EMB16_EXP), bit-exact with the firmware's embed_row_bfp. Computed in
    row blocks when loaded (the int64 intermediates of the whole table were 2.5 GB).
    restore = (signs, hadamard_block): the table is in PrismML's rotated basis and the rows
    are restored to the plain one here (restored_bfp_rows) - the firmware cannot decode
    the ternary rows itself for such a model (header flag HYB_EMB16_RESTORED)."""
    rows, cols = emb.rows, emb.cols
    erows = (rows + 15)//16
    state = {}

    def compute():
        if "mant" in state:
            return
        codes  = emb.get_values()
        mant   = np.empty((rows, cols), dtype=np.int16)
        exps   = np.empty(rows, dtype=np.int32)
        for r0 in range(0, rows, block):
            r1 = min(rows, r0 + block)
            if restore is not None:
                m, e = restored_bfp_rows(emb.scales[r0:r1], codes[r0:r1], *restore)
            else:
                sub = type("Sub", (), {})()
                sub.values = codes[r0:r1]
                sub.scales = emb.scales[r0:r1]
                m, e = embed_bfp_rows(sub)
            mant[r0:r1] = m
            exps[r0:r1] = e
        state["mant"] = mant
        etab = np.zeros((erows, 16), dtype=np.int16)
        etab.reshape(-1)[:rows] = exps.astype(np.int16)
        state["etab"] = etab

    def load_mant():
        compute()
        return state.pop("mant")           # Dropped after the writer has encoded it.

    def load_etab():
        compute()
        return state["etab"]

    return [TensorSpec("token_embd.bfp16", layout.ROLE_TOK_EMB16, 0, layout.KIND_NORM16,
                       rows, cols, loader=load_mant, flags=0),
            # 16 exponents per beat; rows padded to a multiple of 16 tokens. Replicated: the
            # firmware fetches one beat from channel 0 per token.
            TensorSpec("token_embd.bfp16_exp", layout.ROLE_TOK_EMB16_EXP, 0, layout.KIND_NORM16,
                       erows, 16, loader=load_etab, flags=FLAG_REPLICATED)]


def write_image(cfg, tensors, n_channels, path, tokenizer_sha256=b"", manifest_path=None,
                emb16=True, progress=None):
    """Write a BONS image + JSON manifest. Returns the manifest dict.

    Streams: the metadata pass sizes every tensor from rows/cols/kind (the scales are
    already in the specs), which fixes every channel's file offset, so the data pass
    loads one tensor (TensorSpec.loader), encodes it for all channels, writes each chunk
    at its place and drops it. Peak memory is one tensor, not the model.
    emb16=False leaves out the 622 MB int16 embedding table (the firmware decodes the
    ternary row itself then: ~2.6% slower per token, 4.5 s instead of 10 to upload)."""
    tensors = list(tensors)

    # Rope tables baked into the image: one NORM16 row per position, [c14 | s14] Q14
    # halves (Q14 absorbs YaRN's mscale > 1), computed with golden.rope_tables (float64
    # numpy) - the image is the single source of truth, so the firmware and every mirror
    # use bit-identical tables with no libm coupling.
    from trixium.model.golden import rope_tables
    if getattr(cfg, "max_ctx", 0) and getattr(cfg, "head_dim", 0):
        cos, sin = rope_tables(cfg, np.arange(cfg.max_ctx), dtype=np.float64)
        tab = np.concatenate([cos, sin], axis=1)
        t14 = np.clip(np.round(tab*(1 << 14)), -32767, 32767).astype(np.int16)
        tensors.append(TensorSpec("rope_tab", layout.ROLE_ROPE_TAB, 0,
                                  layout.KIND_NORM16, cfg.max_ctx, tab.shape[1],   # = rope_dims when partial
                                  values=t14, flags=FLAG_REPLICATED))

    # Every F32 tensor (norm weights) gets an int16 companion at its own exponent so the vector
    # unit's rmsnorm can fetch it straight into the scratchpad (same role/layer; ".w16"
    # name - the entry name field is 24 bytes; consumers select by kind).
    for spec in [t for t in tensors if t.kind == KIND_F32]:
        w = spec.get_values().astype(np.float64)
        e = layout.norm16_exp(w)               # per tensor: the gammas reach 153 in the 1.7B
        q = np.round(w*2.0**(-e))
        assert np.abs(q).max() <= 32767, (spec.name, e)
        n16_name = spec.name.replace(".weight", "") + ".w16"
        tensors.append(TensorSpec(n16_name, spec.role, spec.layer,
                                  layout.KIND_NORM16, spec.rows, spec.cols,
                                  values=q.astype(np.int16), flags=spec.flags, bias=e))

    emb = [t for t in tensors if t.role == layout.ROLE_TOK_EMB and t.kind == KIND_TERNARY]
    if emb16 and emb and not any(t.role == layout.ROLE_TOK_EMB16 for t in tensors):
        restore = None
        if getattr(cfg, "hadamard_block", 0):
            # The rotated table is restored offline: the firmware never sees the rotation.
            restore = (cfg.hadamard_signs[emb[0].cols], cfg.hadamard_block)
        tensors.extend(_emb16_specs(emb[0], n_channels, restore=restore))

    # Metadata pass: offsets from shapes alone.
    bases, base = [], 0
    for spec in tensors:
        n = _tensor_beats(spec, n_channels)
        bases.append((base, n))
        base += n
    payload_beats  = base
    table_bytes    = len(tensors)*layout.BONS_ENTRY_SIZE
    table_padded_n = ceil_div(table_bytes, layout.BONS_HEADER_SIZE)*layout.BONS_HEADER_SIZE
    payload_offset = layout.BONS_HEADER_SIZE + table_padded_n
    chan_bytes     = payload_beats*BEAT_BYTES

    entries, manifest_tensors = [], []
    with open(path, "wb") as f:
        f.truncate(payload_offset + n_channels*chan_bytes)
        for spec, (base_beats, len_beats) in zip(tensors, bases):
            if progress:
                progress(spec.name)
            values = spec.get_values()
            crc = 0 # CRC over the channel-major concatenation of this tensor's payload.
            for c in range(n_channels):
                if spec.flags & FLAG_REPLICATED:
                    row0, nrows = 0, spec.rows
                else:
                    row0, nrows = shard_rows(spec.rows, n_channels, c)
                chunk = encode_rows(spec, row0, nrows, values=values).tobytes()
                assert len(chunk) == len_beats*BEAT_BYTES, spec.name
                crc = zlib.crc32(chunk, crc) & 0xFFFFFFFF
                f.seek(payload_offset + c*chan_bytes + base_beats*BEAT_BYTES)
                f.write(chunk)
            del values
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
        table     = b"".join(entries)
        table_crc = layout.crc32(table)
        header    = layout.pack_header(cfg, n_channels, len(tensors), payload_beats,
                                       table_crc, tokenizer_sha256)
        f.seek(0)
        f.write(header)
        f.write(table.ljust(table_padded_n, b"\x00"))

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

class _TruncTolerantNames(dict):
    """Name -> entry, tolerant of the 24-byte name field.

    Anything longer is stored truncated - "blk.10.attn_q_norm.weight" (25 chars) lands as
    "...weigh" - so a lookup by the real name misses and the tensor looks absent from the
    image. The firmware indexes by (role, layer) and never saw this; every by-name consumer
    did, including the Python mirror of the firmware loop.
    """
    def __missing__(self, name):
        e = dict.get(self, name[:BONS_NAME_BYTES])
        if e is None:
            raise KeyError(name)
        return e


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
        # The entry name field is 24 bytes, so anything longer is stored truncated -
        # "blk.10.attn_q_norm.weight" (25 chars) lands as "...weigh". Index both forms so a
        # lookup by the real name still resolves; the firmware indexes by (role, layer) and
        # never saw this, but every by-name consumer did, and it looks exactly like the
        # tensor being absent from the image.
        self.by_name = _TruncTolerantNames()
        for e in self.entries:
            self.by_name.setdefault(e["name"], e)

    def channel_payload(self, channel):
        """Full payload bytes of one channel (what gets uploaded to that pseudo-channel)."""
        h     = self.header
        start = h["payload_offset"] + channel*h["payload_beats"]*BEAT_BYTES
        return self.data[start:start + h["payload_beats"]*BEAT_BYTES]

    def tensor_channel_bytes(self, name, channel):
        return self.entry_channel_bytes(self._lookup(name), channel)

    def entry_channel_bytes(self, e, channel):
        """By entry, not by name: the 24-byte entry names collide (`post_attention_norm.
        weight` and its `.w16` companion truncate to the same string, 2026-09-18)."""
        ch = self.channel_payload(channel)
        return ch[e["chan_base_beats"]*BEAT_BYTES:
                  (e["chan_base_beats"] + e["chan_len_beats"])*BEAT_BYTES]

    def _lookup(self, name):
        e = self.by_name.get(name)
        if e is None:
            e = self.by_name.get(name[:BONS_NAME_BYTES])
        if e is None:
            raise KeyError(name)
        return e

    def tensor(self, name):
        """Reassemble a full tensor from its shards -> (scales, values) numpy arrays."""
        e = self._lookup(name)
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

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""GGUF quantization formats used by the Bonsai models (pure numpy, vectorized).

Semantics taken from the PrismML llama.cpp fork (ggml-quants.c, prism branch):

- PQ2_0 (ggml type 142, g128): block of 128 weights = 2B fp16 scale `d` + 32B of 2-bit
  codes. Element j lives in byte j//4 at bit offset (j%4)*2 (LSB-first).
  Dequant: w = (q - 1) * d with q in {0,1,2}; q = 3 (+2*d) is reserved/unused for ternary.
  This is the block the fork shipped as "Q2_0" (type 42) until 2026-06; the fork then gave
  type 42 to upstream's group-64 Q2_0 (18-byte blocks, two scales per 128 weights - not
  representable in the engine's g128 image) and moved the g128 block to type 142. A type-42
  payload is told apart by its size (34 vs 18 bytes per block), see q2_0_group().

- PTQ1_0 (ggml type 143, g128): the same ternary weights packed 5 trits per byte (base 3,
  upstream TQ1_0's layout at group 128): 24 bytes of 5-trit bytes + 2 bytes of 4-trit bytes
  + the fp16 scale = 28 bytes per 128 weights (1.75 bpw). Decoded into the same 0..2 codes.

- Q8_0 (mainline): block of 32 weights = 2B fp16 scale `d` + 32 int8 quants. w = q * d.
"""

import numpy as np

# Q2_0 (g128) --------------------------------------------------------------------------------------

QK_Q2_0    = 128
BLOCK_Q2_0 = 2 + QK_Q2_0//4 # 34 bytes.

def q2_0_unpack(raw):
    """Raw Q2_0 bytes -> (scales float16 (nb,), codes uint8 (nb, 128) with values 0..3)."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_Q2_0 == 0, f"Q2_0 payload not a multiple of {BLOCK_Q2_0} bytes"
    blocks = raw.reshape(-1, BLOCK_Q2_0)
    scales = blocks[:, 0:2].copy().view(np.float16).reshape(-1)
    qs     = blocks[:, 2:]                                        # (nb, 32)
    # Element j = (qs[j//4] >> ((j%4)*2)) & 3 -> byte b holds elements 4b..4b+3, LSB-first.
    codes  = np.stack([(qs >> (2*j)) & 3 for j in range(4)], axis=-1).reshape(-1, QK_Q2_0)
    return scales, codes

def q2_0_scales(raw):
    """Raw Q2_0 bytes -> scales float16 (nb,) only - the 2 bytes of every 34, no code
    unpacking. The converter's metadata pass needs the scales (bias/rshift) for every
    tensor while the codes stream one tensor at a time (2026-09-11)."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_Q2_0 == 0, f"Q2_0 payload not a multiple of {BLOCK_Q2_0} bytes"
    return raw.reshape(-1, BLOCK_Q2_0)[:, 0:2].copy().view(np.float16).reshape(-1)

def q2_0_pack(scales, codes):
    """(scales float16 (nb,), codes uint8 (nb, 128)) -> raw Q2_0 bytes."""
    scales = np.asarray(scales, dtype=np.float16)
    codes  = np.asarray(codes,  dtype=np.uint8)
    assert codes.ndim == 2 and codes.shape[1] == QK_Q2_0
    assert scales.shape == (codes.shape[0],)
    assert codes.max(initial=0) <= 3
    nb  = codes.shape[0]
    c   = codes.reshape(nb, QK_Q2_0//4, 4)
    qs  = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((nb, BLOCK_Q2_0), dtype=np.uint8)
    out[:, 0:2] = scales.view(np.uint8).reshape(nb, 2)
    out[:, 2:]  = qs
    return out.reshape(-1)

def q2_0_group(raw, n_elements):
    """128 or 64: which Q2_0 block a type-42 payload holds, from its size. The fork stored
    the g128 block as type 42 until 2026-06 (34 B per 128), upstream's type 42 is g64 (18 B
    per 64 = 36 B per 128)."""
    nbytes = raw.size if isinstance(raw, np.ndarray) else len(raw)
    if nbytes*128 == n_elements*BLOCK_Q2_0:
        return 128
    if nbytes*64 == n_elements*18:
        return 64
    raise ValueError(f"Q2_0 payload of {nbytes} bytes fits neither g128 nor g64 for "
                     f"{n_elements} elements")

def q2_0_dequant(raw):
    """Raw Q2_0 bytes -> float32 weights (nb*128,)."""
    scales, codes = q2_0_unpack(raw)
    return ((codes.astype(np.int32) - 1)*scales.astype(np.float32)[:, None]).reshape(-1)

# PTQ1_0 (g128, dense trits) ----------------------------------------------------------------------

QK_PTQ1_0    = 128
BLOCK_PTQ1_0 = 24 + 2 + 2 # 28 bytes: qs[24] (5 trits each), qh[2] (4 trits each), fp16 scale.
_PTQ1_STAGES = (32, 16, 8) # ggml-quants.c ptq1_0_stages: the qs bytes come in runs of c
                           # bytes, run k of stage c holding elements m + n*c, n = 0..4.
_POW3        = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)

def _ptq1_0_layout():
    """Element index for (byte, trit) of a block, following dequantize_row_ptq1_0: the
    element order is the order the C loop writes y. Returns (qs_idx (24, 5), qh_idx (2, 4))."""
    qs_idx = np.zeros((24, 5), dtype=np.int64)
    qh_idx = np.zeros((2, 4), dtype=np.int64)
    e = 0
    j = 0
    for c in _PTQ1_STAGES:
        while j + c <= 24:
            for n in range(5):
                for m in range(c):
                    qs_idx[j + m, n] = e
                    e += 1
            j += c
    for n in range(4):
        for h in range(2):
            qh_idx[h, n] = e
            e += 1
    assert e == QK_PTQ1_0
    return qs_idx, qh_idx

_PTQ1_QS_IDX, _PTQ1_QH_IDX = _ptq1_0_layout()

def ptq1_0_unpack(raw):
    """Raw PTQ1_0 bytes -> (scales float16 (nb,), codes uint8 (nb, 128) with values 0..2).
    Trit n of a byte q is ((q * 3^n) & 0xFF) * 3 >> 8, exactly the C kernel's arithmetic."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_PTQ1_0 == 0, f"PTQ1_0 payload not a multiple of {BLOCK_PTQ1_0} bytes"
    blocks = raw.reshape(-1, BLOCK_PTQ1_0)
    nb     = blocks.shape[0]
    qs     = blocks[:, 0:24].astype(np.uint16)
    qh     = blocks[:, 24:26].astype(np.uint16)
    scales = blocks[:, 26:28].copy().view(np.float16).reshape(-1)
    codes  = np.empty((nb, QK_PTQ1_0), dtype=np.uint8)
    for n in range(5):
        q = (qs*_POW3[n]) & 0xFF
        codes[:, _PTQ1_QS_IDX[:, n]] = ((q*3) >> 8).astype(np.uint8)
    for n in range(4):
        q = (qh*_POW3[n]) & 0xFF
        codes[:, _PTQ1_QH_IDX[:, n]] = ((q*3) >> 8).astype(np.uint8)
    return scales, codes

def ptq1_0_scales(raw):
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_PTQ1_0 == 0
    return raw.reshape(-1, BLOCK_PTQ1_0)[:, 26:28].copy().view(np.float16).reshape(-1)

def ptq1_0_pack(scales, codes):
    """(scales float16 (nb,), codes uint8 (nb, 128) in 0..2) -> raw PTQ1_0 bytes, as
    quantize_row_ptq1_0_ref packs them (base-3 digits, most significant first, then the
    ceiling scaling q = ceil(q*256/243) that makes the (q*3^n)*3>>8 extraction exact)."""
    scales = np.asarray(scales, dtype=np.float16)
    codes  = np.asarray(codes,  dtype=np.uint16)
    assert codes.ndim == 2 and codes.shape[1] == QK_PTQ1_0 and codes.max(initial=0) <= 2
    nb = codes.shape[0]
    qs = np.zeros((nb, 24), dtype=np.uint16)
    for n in range(5):
        qs = qs*3 + codes[:, _PTQ1_QS_IDX[:, n]]
    qs = (qs*256 + 242)//243
    qh = np.zeros((nb, 2), dtype=np.uint16)
    for n in range(4):
        qh = qh*3 + codes[:, _PTQ1_QH_IDX[:, n]]
    qh = qh*3                                        # the first trit to the top digit
    qh = (qh*256 + 242)//243
    out = np.empty((nb, BLOCK_PTQ1_0), dtype=np.uint8)
    out[:, 0:24]  = qs.astype(np.uint8)
    out[:, 24:26] = qh.astype(np.uint8)
    out[:, 26:28] = scales.view(np.uint8).reshape(nb, 2)
    return out.reshape(-1)

def ptq1_0_dequant(raw):
    scales, codes = ptq1_0_unpack(raw)
    return ((codes.astype(np.float32) - 1.0)*scales.astype(np.float32)[:, None]).reshape(-1)

# Q1_0 (1-bit, PrismML fork) -----------------------------------------------------------------------
# Block of 128 weights = 2B fp16 scale d + 16B of bits (element j: bit j%8 of byte j//8).
# Dequant: w = bit ? +d : -d  (1.125 bits/weight).

QK_Q1_0    = 128
BLOCK_Q1_0 = 2 + QK_Q1_0//8 # 18 bytes.

def q1_0_unpack(raw):
    """Raw Q1_0 bytes -> (scales float16 (nb,), bits uint8 (nb, 128) with values 0/1)."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_Q1_0 == 0, f"Q1_0 payload not a multiple of {BLOCK_Q1_0} bytes"
    blocks = raw.reshape(-1, BLOCK_Q1_0)
    scales = blocks[:, 0:2].copy().view(np.float16).reshape(-1)
    qs     = blocks[:, 2:]
    bits   = np.unpackbits(qs, axis=1, bitorder="little")
    return scales, bits

def q1_0_pack(scales, bits):
    scales = np.asarray(scales, dtype=np.float16)
    bits   = np.asarray(bits, dtype=np.uint8)
    assert bits.ndim == 2 and bits.shape[1] == QK_Q1_0
    nb  = bits.shape[0]
    qs  = np.packbits(bits, axis=1, bitorder="little")
    out = np.empty((nb, BLOCK_Q1_0), dtype=np.uint8)
    out[:, 0:2] = scales.view(np.uint8).reshape(nb, 2)
    out[:, 2:]  = qs
    return out.reshape(-1)

def q1_0_dequant(raw):
    scales, bits = q1_0_unpack(raw)
    w = bits.astype(np.float32)*2.0 - 1.0
    return (w*scales.astype(np.float32)[:, None]).reshape(-1)

def q1_0_to_ternary_codes(raw):
    """Bridge for the existing ternary engine: 1-bit weights as ternary codes (0 -> -1 -> code
    0, 1 -> +1 -> code 2). Stored at 2.125 bpw instead of 1.125 - functional today; a native
    Q1 unpacker mode (phase 2) halves the streamed bytes."""
    scales, bits = q1_0_unpack(raw)
    return scales, (bits.astype(np.uint8)*2)

# Q8_0 ---------------------------------------------------------------------------------------------

QK_Q8_0    = 32
BLOCK_Q8_0 = 2 + QK_Q8_0 # 34 bytes.

def q8_0_unpack(raw):
    """Raw Q8_0 bytes -> (scales float16 (nb,), quants int8 (nb, 32))."""
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    assert raw.size % BLOCK_Q8_0 == 0, f"Q8_0 payload not a multiple of {BLOCK_Q8_0} bytes"
    blocks = raw.reshape(-1, BLOCK_Q8_0)
    scales = blocks[:, 0:2].copy().view(np.float16).reshape(-1)
    quants = blocks[:, 2:].copy().view(np.int8)
    return scales, quants

def q8_0_pack(scales, quants):
    scales = np.asarray(scales, dtype=np.float16)
    quants = np.asarray(quants, dtype=np.int8)
    assert quants.ndim == 2 and quants.shape[1] == QK_Q8_0
    nb  = quants.shape[0]
    out = np.empty((nb, BLOCK_Q8_0), dtype=np.uint8)
    out[:, 0:2] = scales.view(np.uint8).reshape(nb, 2)
    out[:, 2:]  = quants.view(np.uint8)
    return out.reshape(-1)

def q8_0_dequant(raw):
    scales, quants = q8_0_unpack(raw)
    return (quants.astype(np.float32)*scales.astype(np.float32)[:, None]).reshape(-1)

def q8_0_quant(w):
    """Float weights -> Q8_0 (absmax per 32-block), returns raw bytes. Reference-style rounding."""
    w  = np.asarray(w, dtype=np.float32).reshape(-1, QK_Q8_0)
    d  = np.abs(w).max(axis=1)/127.0
    id = np.where(d > 0, 1.0/np.where(d > 0, d, 1.0), 0.0)
    q  = np.clip(np.round(w*id[:, None]), -127, 127).astype(np.int8)
    return q8_0_pack(d.astype(np.float16), q)

# F16/F32 ------------------------------------------------------------------------------------------

def f16_dequant(raw):
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32)

def f32_dequant(raw):
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    return np.ascontiguousarray(raw).view(np.float32).copy()

# GGUF type id -> dequant dispatch -----------------------------------------------------------------

GGML_Q2_0   = 42  # PrismML fork type id: the g128 block until 2026-06, upstream's g64 since.
GGML_Q1_0   = 41  # PrismML fork type id (1-bit builds).
GGML_PQ2_0  = 142 # PrismML fork: the g128 2-bit block, its current name.
GGML_PTQ1_0 = 143 # PrismML fork: g128 dense trits (5 per byte).

DEQUANT = {
    0:           f32_dequant,
    1:           f16_dequant,
    8:           q8_0_dequant,
    GGML_Q1_0:   q1_0_dequant,
    GGML_Q2_0:   q2_0_dequant,   # g128 only - load_gguf_f32 checks q2_0_group() first
    GGML_PQ2_0:  q2_0_dequant,
    GGML_PTQ1_0: ptq1_0_dequant,
}

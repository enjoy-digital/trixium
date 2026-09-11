#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""GGUF quantization formats used by the Bonsai models (pure numpy, vectorized).

Semantics taken from the PrismML llama.cpp fork (ggml-quants.c, prism branch):

- Q2_0 (g128): block of 128 weights = 2B fp16 scale `d` + 32B of 2-bit codes.
  Element j lives in byte j//4 at bit offset (j%4)*2 (LSB-first).
  Dequant: w = (q - 1) * d with q in {0,1,2}; q = 3 (+2*d) is reserved/unused for ternary.

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

def q2_0_dequant(raw):
    """Raw Q2_0 bytes -> float32 weights (nb*128,)."""
    scales, codes = q2_0_unpack(raw)
    return ((codes.astype(np.int32) - 1)*scales.astype(np.float32)[:, None]).reshape(-1)

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

GGML_Q2_0 = 42 # PrismML fork type id.
GGML_Q1_0 = 41 # PrismML fork type id (1-bit builds, future).

DEQUANT = {
    0:         f32_dequant,
    1:         f16_dequant,
    8:         q8_0_dequant,
    GGML_Q1_0: q1_0_dequant,
    GGML_Q2_0: q2_0_dequant,
}

# SPDX-License-Identifier: BSD-2-Clause
"""Exact on-demand inverse rotation of compact PTQ embedding rows.

This module defines/admission-checks the integer execution path. It neither
changes the logical model nor replaces the legacy unrotated BFP embedding path.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from trixium import layout
from trixium.model import bitexact as bx, bons4


def scratchpad_embedding_descriptor(*, src, signs, dst, groups):
    """Encode optional unit 9; check embedding_restore ABI/profile before use.

    Input is groups compact 28-byte PTQ records plus groups*16 inverse-sign
    bytes, all resident in scratchpad. Output is groups*128 FP32 elements.
    Hardware validates its configured capacity, full spans and list protection.
    Entire inputs are cached first, so output may overwrite either input; reload
    overwritten compact/sign data before reusing it. This does not fetch a model
    row from a device-memory region or authorize a model's numerical profile.
    """
    for name, value in (('src', src), ('signs', signs), ('dst', dst)):
        if type(value) is not int or not 0 <= value < 1 << 24 or value % 32:
            raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')
    if type(groups) is not int or not 8 <= groups <= 512 or groups % 8:
        raise ValueError('groups must be a multiple of eight in [8,512]')
    return [9, src, signs, dst, groups] + [0]*11


def _parameters(input_width, max_groups):
    if type(input_width) is not int or not 12 <= input_width <= 32:
        raise ValueError('embedding input width must be in [12,32]')
    if type(max_groups) is not int or not 8 <= max_groups <= 512 or max_groups % 8:
        raise ValueError('maximum embedding groups must be a multiple of 8 in [8,512]')


def embedding_scale_contract(scales, *, input_width=16):
    """Return per-row common exponent and exact signed integer scale mantissas.

    No shift capping or rounding: emax-emin <= input_width-12 guarantees all
    signed FP16 mantissas fit after alignment, including negative/subnormal/zero
    scales. Wider ranges fail; an execution adapter must not silently narrow.
    """
    _parameters(input_width, 64)
    scales = np.asarray(scales, dtype=np.float16)
    if scales.ndim != 2 or not scales.shape[0] or not scales.shape[1]:
        raise ValueError('expected nonempty row/group scales')
    if not np.isfinite(scales).all():
        raise ValueError('nonfinite embedding scale')
    mantissas, exponents = bx.fp16_decompose(scales)
    common = exponents.min(axis=1)
    shifts = exponents - common[:, None]
    if np.any(shifts > input_width - 12):
        raise ValueError('embedding scale span exceeds exact integer input width')
    return common, mantissas << shifts


def restore_embedding_row(scales, codes, signs, *, input_width=16):
    """Integer reference: compact row -> inverse H/32 -> whole-row BFP16/FP32.

    Return (int16 mantissas, signed exponent, binary32 values). The clipping and
    ties-to-even rounding match bons.restored_bfp_rows, not activation Q8 RMM.
    All 1024-element blocks share one alignment exponent and one final BFP
    exponent. Signs apply AFTER the transform. Zero rows use exponent zero.
    """
    codes, signs = np.asarray(codes), np.asarray(signs)
    if (codes.ndim != 1 or not len(codes) or len(codes) % 1024 or signs.shape != codes.shape
            or codes.dtype.kind not in 'iu' or not np.isin(codes, [0, 1, 2]).all()
            or not np.isin(signs, [-1, 1]).all()):
        raise ValueError('expected complete 1024-element blocks of trits and +/-1 signs')
    scales = np.asarray(scales, dtype=np.float16)
    if scales.shape != (len(codes)//128,):
        raise ValueError('embedding scale count does not match row')
    common, aligned = embedding_scale_contract(scales[None, :], input_width=input_width)
    values = ((codes.astype(np.int64).reshape(-1, 128)-1) * aligned.reshape(-1, 1)).reshape(-1, 1024)
    distance = 1
    while distance < 1024:
        chunks = values.reshape(-1, 2*distance)
        lo, hi = chunks[:, :distance].copy(), chunks[:, distance:].copy()
        chunks[:, :distance], chunks[:, distance:] = lo + hi, lo - hi
        distance *= 2
    raw = values.reshape(-1) * signs.astype(np.int64)
    maximum = int(np.abs(raw).max())
    shift = maximum.bit_length() - 15
    exponent = int(common[0]) - 5 + shift if maximum else 0
    magnitude = np.abs(raw)
    if shift > 0:
        rounded = bx.rne_shift(magnitude, shift)
    else:
        rounded = magnitude << -shift
    mant = (np.minimum(rounded, 32767) * np.sign(raw)).astype(np.int16)
    floats = mant.astype(np.float32) * np.float32(2.0**exponent)
    return mant, exponent, floats


@dataclass(frozen=True)
class EmbeddingProfile:
    source: bons4.Tensor
    source_directory_sha256: bytes
    input_width: int
    max_groups: int
    span_histogram: tuple[int, ...]
    min_exponent: int
    max_exponent: int


def profile_ptq_embedding(reader, name='token_embd.weight', *, input_width=16, max_groups=64):
    """Hash-check and admit every compact row without expanding the table.

    This proves format/range compatibility only. The model adapter still owns
    the 1024-point rotation/sign semantics, placement and optional hardware ABI.
    Artifacts must remain immutable between profiling, upload and execution.
    """
    _parameters(input_width, max_groups)
    source = reader.by_name[name]
    if (source.encoding != bons4.Encoding.PTQ1_0 or source.role != layout.ROLE_TOK_EMB
            or len(source.shape) != 2 or source.shape[1] % 1024
            or source.shape[1]//128 > max_groups):
        raise ValueError('unsupported compact embedding geometry/role/encoding')
    digest = hashlib.sha256()
    histogram = np.zeros(input_width-11, dtype=np.int64)
    minimum, maximum = 5, -24
    for start in range(0, source.rows, 256):
        count = min(256, source.rows-start)
        raw = reader.read_rows(name, start, count)
        digest.update(raw)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(count, -1, 28)
        scales = blocks[:, :, 26:28].copy().view('<f2').reshape(count, -1)
        common, _ = embedding_scale_contract(scales, input_width=input_width)
        _, exponents = bx.fp16_decompose(scales)
        spans = exponents.max(axis=1)-common
        histogram += np.bincount(spans, minlength=len(histogram))
        minimum, maximum = min(minimum, int(exponents.min())), max(maximum, int(exponents.max()))
    if digest.digest() != source.sha256:
        raise ValueError('embedding source payload SHA256 mismatch')
    return EmbeddingProfile(source, reader.directory_sha256, input_width, max_groups,
                            tuple(int(n) for n in histogram), minimum, maximum)

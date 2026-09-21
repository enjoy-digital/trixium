# SPDX-License-Identifier: BSD-2-Clause
"""Board-independent, reference-preserving projection preparation.

This is load-time preparation, not CPU inference or an executable model loader.
The immutable BONS4 source is never changed; derived Q8 bytes need their own
allocation and must not be uploaded using the original BF16 placement spans.
"""
from dataclasses import dataclass, field
import hashlib

import numpy as np

from trixium import layout
from trixium.model import bitexact as bx, bons4

BF16_Q8_TRANSFORM = 'bf16-q8-fp16-scale-rne-v1'


@dataclass(frozen=True)
class ReferenceQ8Projection:
    """Whole-matrix fixed-point contract, shared unchanged by every row shard.

    Inputs must be reference activation Q8 in [-127, 127]. This representation
    does not prescribe pipe count, memory region, physical stride or K-segments.
    Native command admission/capabilities remain a separate runtime check.
    """
    shape: tuple[int, int]
    payload: bytes
    bias: int = field(init=False)
    rshift: int = field(init=False)
    accumulator_bound: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        if len(self.shape) != 2:
            raise ValueError('projection must be a rank-two matrix')
        rows, row_bytes = bons4.geometry(bons4.Encoding.Q8_0, self.shape)
        if not isinstance(self.payload, bytes) or len(self.payload) != rows * row_bytes:
            raise ValueError('Q8 payload size does not match projection geometry')
        blocks = np.frombuffer(self.payload, dtype=np.uint8).reshape(-1, 34)
        scales = blocks[:, :2].copy().view('<f2').reshape(rows, -1)
        if not np.isfinite(scales).all() or (scales < 0).any():
            raise ValueError('reference Q8 scales must be finite and nonnegative')
        if (blocks[:, 2:] == 128).any():
            raise ValueError('reference Q8 weights must be in [-127, 127]')
        mantissas, exponents = bx.fp16_decompose(scales)
        bias = bx.choose_scale_bias(exponents)
        # The reference's all-weights=127 bound selects R, but can overestimate
        # a 160-group matrix beyond int48. Prove accumulator safety from the
        # actual immutable weights, for ANY activation in [-127,127], without
        # changing B/R. Negative arithmetic shifts require rounding UP here.
        q = blocks[:, 2:].view(np.int8).astype(np.int64).reshape(rows, -1, 32)
        product = np.abs(q).sum(axis=2) * 127 * mantissas
        shifts = exponents + bias
        right = np.maximum(-shifts, 0)
        magnitudes = np.where(shifts >= 0, product << np.maximum(shifts, 0),
                              (product + (1 << right) - 1) >> right)
        # Positive integer summands are exact in float64 throughout the accepted
        # int48 range; oversized inputs cannot wrap an int64 reduction negative.
        bound = int(magnitudes.sum(axis=1, dtype=np.float64).max())
        if bound >= 1 << (bx.ACC_BITS - 1):
            raise ValueError('projection exceeds the reference int48 accumulator')
        shift = bx.choose_result_shift(scales, bias, gs_max=bx.GS_MAX_Q8)
        object.__setattr__(self, 'bias', bias)
        object.__setattr__(self, 'rshift', shift)
        object.__setattr__(self, 'accumulator_bound', bound)

    @property
    def row_bytes(self):
        return self.shape[1] // 32 * 34

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).digest()

    def read_rows(self, start, count):
        """Return compact rows; keep this object's whole-matrix B/R when using them."""
        if (type(start) is not int or type(count) is not int or start < 0 or count <= 0
                or start >= self.shape[0] or count > self.shape[0] - start):
            raise ValueError('projection row range out of bounds')
        return self.payload[start * self.row_bytes:(start + count) * self.row_bytes]


def reference_q8_from_bf16(payload, shape):
    """Match the legacy beta/alpha conversion, including FP16 scale rounding.

    BF16 input and Q8_0 output are explicitly little-endian. Quantize using the
    *rounded FP16* scale, then RNE and clip; qformats.q8_0_quant uses a different
    order and is intentionally NOT used. Zero/underflowed scales yield zero
    quants. Nonfinite inputs and overflowed scales fail closed.
    """
    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError('projection must be a rank-two matrix')
    rows, row_bytes = bons4.geometry(bons4.Encoding.BF16, shape)
    if shape[1] % 32:
        raise ValueError('Q8 groups must divide the projection width')
    if not isinstance(payload, bytes) or len(payload) != rows * row_bytes:
        raise ValueError('BF16 payload size does not match projection geometry')
    bits = np.frombuffer(payload, dtype='<u2').astype(np.uint32) << 16
    weights = bits.view(np.float32).reshape(rows, shape[1] // 32, 32)
    if not np.isfinite(weights).all():
        raise ValueError('nonfinite BF16 projection weight')
    with np.errstate(over='ignore', under='ignore'):
        scales = (np.abs(weights).max(axis=2) / np.float32(127)).astype(np.float16)
    if not np.isfinite(scales).all():
        raise ValueError('BF16 projection scale overflows FP16')
    sc = scales.astype(np.float32)[:, :, None]
    quants = np.where(sc > 0, np.rint(weights / np.where(sc > 0, sc, np.float32(1))), 0)
    quants = quants.clip(-127, 127).astype(np.int8)
    out = np.empty((rows * (shape[1] // 32), 34), dtype=np.uint8)
    out[:, :2] = np.frombuffer(scales.astype('<f2').tobytes(), dtype=np.uint8).reshape(-1, 2)
    out[:, 2:] = quants.reshape(-1, 32).view(np.uint8)
    return ReferenceQ8Projection(shape, out.tobytes())


@dataclass(frozen=True)
class PreparedBF16Projection:
    """In-memory preparation result, not an untrusted cache/placement parser."""
    source: bons4.Tensor
    source_directory_sha256: bytes
    execution: ReferenceQ8Projection

    def manifest(self):
        """Provenance/report data, never authorization for a device transfer."""
        return dict(transform=BF16_Q8_TRANSFORM,
                    source_directory_sha256=self.source_directory_sha256.hex(),
                    source_tensor_id=self.source.id, source_name=self.source.name,
                    source_encoding=int(self.source.encoding), source_sha256=self.source.sha256.hex(),
                    source_bytes=self.source.size, shape=list(self.source.shape),
                    role=self.source.role, layer=self.source.layer,
                    execution_encoding=int(bons4.Encoding.Q8_0),
                    execution_sha256=self.execution.sha256.hex(), execution_bytes=len(self.execution.payload),
                    bias=self.execution.bias, rshift=self.execution.rshift,
                    accumulator_bound=self.execution.accumulator_bound,
                    activation_basis='unrotated', activation_q8_range=[-127, 127])


def prepare_bf16_projection(reader, name, *, max_source_bytes=64 * 1024 * 1024):
    """Verify one BONS4 BF16 beta/alpha tensor, then prepare its Q8 execution view.

    One bounded matrix is materialized at a time. This validates this payload,
    not every other model payload, model completeness or runtime support.
    """
    source = reader.by_name[name]
    if (source.encoding != bons4.Encoding.BF16 or len(source.shape) != 2
            or source.role not in (layout.ROLE_SSM_BETA, layout.ROLE_SSM_ALPHA)
            or source.layer == bons4.UNSPECIFIED or source.shape[1] % 32):
        raise ValueError('expected a recognized rank-two BF16 beta/alpha projection')
    if type(max_source_bytes) is not int or max_source_bytes <= 0 or source.size > max_source_bytes:
        raise ValueError('BF16 projection exceeds source byte budget')
    payload = reader.read_rows(name, 0, source.rows)
    if hashlib.sha256(payload).digest() != source.sha256:
        raise ValueError('BF16 source payload SHA256 mismatch')
    return PreparedBF16Projection(source, reader.directory_sha256,
                                  reference_q8_from_bf16(payload, source.shape))


def resident_reference_q8_descriptors(projection, *, pipe_locations, q8_src, exponents,
                                      int32_dst, fp32_dst, fp32_src=None):
    """Lower one prepared projection into resident units [4], 6, 7 and 8.

    pipe_locations gives (region ID, byte offset) in compute-pipe order, one
    entry per actual hardware pipe. Upload contiguous compact row shards there:
    pipe i starts at i*(rows//pipes)+min(i, rows%pipes), has rows//pipes+(i<rem)
    rows, and keeps the WHOLE matrix's B/R. Empty trailing pipes are permitted.
    These are derived Q8 allocations, never the original BF16 placement spans.

    fp32_src=None reuses resident Q8/exponent metadata, e.g. alpha after beta.
    Otherwise quantize the entire original, UNROTATED activation once. This
    helper does not allocate memory, upload weights, validate region ownership
    or advertise support. The runtime must check all three optional ABIs,
    actual pipe count/capacity and physical spans before submitting the list.
    """
    from trixium.model.activation import scratchpad_quantize_descriptor, scratchpad_restore_descriptor
    from trixium.model.native_sequence import native_pipe_descriptor, native_projection_descriptor

    locations = tuple(pipe_locations)
    if not 1 <= len(locations) <= 32:
        raise ValueError('projection needs 1..32 physical compute-pipe locations')
    rows, cols = projection.shape
    per_pipe, rem = divmod(rows, len(locations))
    descriptors = []
    if fp32_src is not None:
        descriptors.append(scratchpad_quantize_descriptor(src=fp32_src, dst=q8_src,
                                                          exponents=exponents, elements=cols))
    for pipe, (region, offset) in enumerate(locations):
        descriptors.append(native_pipe_descriptor(pipe=pipe, region=region, offset=offset,
                                                  stride=projection.row_bytes))
    descriptors.append(native_projection_descriptor(kind=1, rows=per_pipe, rem=rem, groups=cols//32,
        bias=projection.bias, rshift=projection.rshift, xsrc=q8_src, dst=int32_dst))
    descriptors.append(scratchpad_restore_descriptor(src=int32_dst, dst=fp32_dst, exponents=exponents,
        elements=rows, adjustment=projection.rshift-projection.bias))
    return descriptors

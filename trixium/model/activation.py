# SPDX-License-Identifier: BSD-2-Clause
"""Portable resident activation conversion and projection restoration descriptors."""


def scratchpad_quantize_descriptor(*, src, dst, exponents, elements, blocks=1):
    """Encode optional sequencer unit 4; check seq.activation_quantize_abi first.

    Hardware validates physical capacity, complete spans and descriptor-list
    protection. One block gives whole-vector quantization; multiple equal blocks
    give independent scales, e.g. 1024 elements each before a Hadamard transform.
    """
    for name, value in (('src', src), ('dst', dst), ('exponents', exponents)):
        if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')
    if not isinstance(elements, int) or not 32 <= elements < 65536 or elements % 32:
        raise ValueError('elements must be a positive multiple of 32 below 65536')
    if not isinstance(blocks, int) or not 1 <= blocks < 65536:
        raise ValueError('blocks must be in [1, 65535]')
    return [4, src, dst, exponents, elements, blocks] + [0]*10


def scratchpad_requantize_descriptor(*, src, dst, exponents, maxima, result_exponent, blocks):
    """Encode unit 5 (activation_quantize ABI major 1, minor >= 1).

    Raw signed int32 Hadamard blocks each contain 1024 elements. Input metadata
    is signed int32 pre-rotation e8 and uint32 raw absmax per block. The result
    uses one exponent across *all* blocks, before any matrix K segmentation.
    Reserve one 32-byte beat at result_exponent; seven tail words are preserved.
    The hardware validates physical spans and every metadata/raw input value.
    """
    for name, value in (('src', src), ('dst', dst), ('exponents', exponents),
                        ('maxima', maxima), ('result_exponent', result_exponent)):
        if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')
    if not isinstance(blocks, int) or not 1 <= blocks < 65536:
        raise ValueError('blocks must be in [1, 65535]')
    return [5, src, dst, exponents, maxima, result_exponent, blocks] + [0]*9


def scratchpad_restore_descriptor(*, src, dst, exponents, elements, adjustment=0):
    """Encode optional unit 8; check seq.projection_restore_abi major 1 first.

    Restore int32 to FP32 after any wrapping int32 K-segment reduction. The
    low lane at exponents holds signed int32 e8; adjustment is normally R-B.
    The client validates the full-width sum, physical spans and list protection.
    src==dst is supported; inactive final destination lanes are preserved.
    """
    for name, value in (('src', src), ('dst', dst), ('exponents', exponents)):
        if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')
    if not isinstance(elements, int) or not 1 <= elements < 65536:
        raise ValueError('elements must be in [1, 65535]')
    if not isinstance(adjustment, int) or not -(1 << 31) <= adjustment < 1 << 31:
        raise ValueError('adjustment must be a signed int32')
    return [8, src, dst, exponents, elements, adjustment & 0xffffffff] + [0]*10

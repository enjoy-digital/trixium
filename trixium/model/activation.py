# SPDX-License-Identifier: BSD-2-Clause
"""Portable resident FP32 -> block/vector Q8 descriptor encoding."""


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

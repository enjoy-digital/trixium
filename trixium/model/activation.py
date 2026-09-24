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


def scratchpad_silu_descriptor(*, src_g, src_m=0, dst=0, fixmap=0, elements=0, load=False,
                               unit_m=False, seg_beats=0, perm_a=0, perm_b=0):
    """Encode optional unit 11; check seq.silu_abi major 1 first.

    RUN: dst[i] = m[i] * silu(g[i]) bit-exact with the firmware, fixup elements keep g in
    dst and set bit i of the fixmap bitmap (ceil(elements/32) words, all written).
    dst == src_g or disjoint; fixmap disjoint from everything. load=True reads the
    1,025-beat table image at src_g (silu_table_image) instead; other fields must be 0.
    """
    names = ('src_g',) if load else ('src_g', 'src_m', 'dst', 'fixmap')
    for name, value in (('src_g', src_g), ('src_m', src_m), ('dst', dst), ('fixmap', fixmap)):
        if name not in names:
            if value:
                raise ValueError(name + ' must be 0 for a table load')
        elif not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')
    if load:
        if elements:
            raise ValueError('a table load has no elements')
    elif not isinstance(elements, int) or not 1 <= elements < 65536:
        raise ValueError('elements must be in [1, 65535]')
    if load and (unit_m or seg_beats or perm_a or perm_b):
        raise ValueError('a table load takes no run options')
    for name, value, bits in (('seg_beats', seg_beats, 16), ('perm_a', perm_a, 8), ('perm_b', perm_b, 8)):
        if not isinstance(value, int) or not 0 <= value < 1 << bits:
            raise ValueError(name + ' out of range')
    if perm_a > 1 and seg_beats*perm_a*perm_b*8 != elements:
        raise ValueError('a permuted output needs elements == seg_beats*8*perm_a*perm_b')
    return [11, src_g, src_m, dst, fixmap, elements, int(load) | int(bool(unit_m)) << 1,
            seg_beats | perm_a << 16 | perm_b << 24] + [0]*8


def _sp_address(name, value):
    if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
        raise ValueError(name + ' must be a 32-byte-aligned unsigned 24-bit address')


def scratchpad_norm_descriptor(*, src, dst, n, nseg=1, eps, weight=0, l2=False, src_stride=0,
                               dst_stride=0):
    """Encode optional unit 12 (seq.norm_abi major 1): RMSNorm (weight) or L2 over nseg
    segments of n FP32 values; eps as FP32 bits. docs/canonical-reductions.md."""
    for name, value in (('src', src), ('dst', dst), ('weight', weight)):
        _sp_address(name, value)
    if not isinstance(n, int) or not 8 <= n <= 8192 or n % 8:
        raise ValueError('n must be a multiple of 8 in [8, 8192]')
    if not isinstance(nseg, int) or not 1 <= nseg < 65536:
        raise ValueError('nseg must be in [1, 65535]')
    if not isinstance(eps, int) or not 0 <= eps < 1 << 32:
        raise ValueError('eps must be FP32 bits')
    if l2 not in (0, 1):
        raise ValueError('l2 must be 0 (RMS) or 1 (L2)')
    for name, value in (('src_stride', src_stride), ('dst_stride', dst_stride)):
        if not isinstance(value, int) or not (value == 0 or n//8 <= value < 65536):
            raise ValueError(name + ' must be 0 or at least n/8 beats')
    return [12, src, dst, weight, n | nseg << 16, eps, int(l2), src_stride | dst_stride << 16] + [0]*8


def scratchpad_conv_descriptor(*, x0, x1, x2, x3, w, dst, channels):
    """Encode optional unit 13 (seq.conv_abi major 1): the DeltaNet 4-tap causal conv."""
    for name, value in (('x0', x0), ('x1', x1), ('x2', x2), ('x3', x3), ('w', w), ('dst', dst)):
        _sp_address(name, value)
    if not isinstance(channels, int) or not 8 <= channels < 65536 or channels % 8:
        raise ValueError('channels must be a multiple of 8 in [8, 65528]')
    return [13, x0, x1, x2, x3, w, dst, channels] + [0]*8


def scratchpad_add_descriptor(*, a, b, dst, n):
    """Unit 13 op 1: dst = a + b over n FP32 values (the residual add); dst == a or b."""
    for name, value in (('a', a), ('b', b), ('dst', dst)):
        _sp_address(name, value)
    if not isinstance(n, int) or not 8 <= n < 65536 or n % 8:
        raise ValueError('n must be a multiple of 8 in [8, 65528]')
    return [13, a, 0, 0, b, 0, dst, n | 1 << 16] + [0]*8


def silu_table_image(lut, dlut):
    """The unit-11 LOAD image as 32-bit words: lut[0..4096] then pad to beat 513, dlut."""
    if len(lut) != 4097 or len(dlut) != 4096:
        raise ValueError('lut must have 4097 entries and dlut 4096')
    return list(lut) + [0]*7 + list(dlut)


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

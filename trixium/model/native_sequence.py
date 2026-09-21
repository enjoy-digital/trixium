# SPDX-License-Identifier: BSD-2-Clause
"""Portable sequencer encoding for native PTQ1_0/Q8_0 projections (ABI 1.0)."""


def _unsigned(name, value, width):
    if not isinstance(value, int) or not 0 <= value < 1 << width:
        raise ValueError(f'{name} must be an unsigned {width}-bit integer')
    return value


def native_pipe_descriptor(*, pipe, region, offset, stride):
    """Configure one pipe using a region ID and full 64-bit byte offset/stride.

    Check seq.native_abi major 1 first. Hardware checks the actual pipe count;
    region/reachability/row-span checks occur before each native memory request.
    Descriptors persist between projections but must be reinstalled after reset.
    """
    for name, value, width in (('pipe', pipe, 5), ('region', region, 32),
                               ('offset', offset, 64), ('stride', stride, 64)):
        _unsigned(name, value, width)
    return [6, pipe, region, offset & 0xffffffff, offset >> 32,
            stride & 0xffffffff, stride >> 32] + [0]*9


def region_fetch_descriptor(*, region, offset, length, dst):
    """Encode optional unit 10; check seq.region_fetch_abi major 1 first.

    offset is region-relative bytes; length is a count of full 32-byte beats.
    The shared reader checks access/reservations, endpoint/address reachability,
    full source/destination spans and AXI completion. The sequencer protects its
    entire active descriptor list. A failed read requires common reset/drain and
    invalidates the whole destination, including any prefix already written.
    This fetch does not establish that an uploaded model allocation is valid.
    """
    for name, value, width in (('region', region, 32), ('offset', offset, 64),
                               ('length', length, 24), ('dst', dst, 24)):
        _unsigned(name, value, width)
    if not length or offset % 32 or dst % 32:
        raise ValueError('region fetch needs positive beats and 32-byte-aligned offset/dst')
    return [10, region, offset & 0xffffffff, offset >> 32, length, dst] + [0]*10


def native_projection_descriptor(*, kind, rows, groups, bias, rshift, xsrc, dst,
                                 rem=0, interleaved=False, suppress_writes=False,
                                 dual=False, qstride=0):
    """Serialized native projection; no implicit activation quantization or B/R.

    kind 0 uses 128-element PTQ1_0 groups; kind 1 uses 32-element Q8_0 groups.
    rows/rem describe the shared engine's ragged per-pipe row distribution.
    Use whole-matrix B/R and one whole-activation exponent across K segments.
    Physical capacity, pipe count, x-buffer limits and output/list protection
    are checked by hardware. Each result segment has its own int32 destination;
    combining segments/restoring FP32 scale is an explicit later operation.
    """
    if kind not in (0, 1):
        raise ValueError('kind must be 0 (PTQ1_0) or 1 (Q8_0)')
    for name, value, width in (('rows', rows, 16), ('rem', rem, 5), ('groups', groups, 8),
            ('bias', bias, 7), ('rshift', rshift, 6), ('xsrc', xsrc, 24), ('dst', dst, 24),
            ('qstride', qstride, 16)):
        _unsigned(name, value, width)
    for name, value in (('interleaved', interleaved), ('suppress_writes', suppress_writes), ('dual', dual)):
        if value not in (False, True):
            raise ValueError(name + ' must be boolean')
    if groups == 0 or rows == rem == 0:
        raise ValueError('projection requires nonzero groups and total rows')
    if xsrc % 32 or dst % 4:
        raise ValueError('xsrc must be 32-byte aligned and dst 4-byte aligned')
    return [7, kind | (int(interleaved) << 8) | (int(suppress_writes) << 9) | (int(dual) << 11),
            rows | (rem << 16), 0, 0, 0, (groups << 8) | (qstride << 16),
            bias | (rshift << 16), xsrc, dst] + [0]*6

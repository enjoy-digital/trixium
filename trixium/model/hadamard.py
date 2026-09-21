# SPDX-License-Identifier: BSD-2-Clause
"""Board-independent sequencer descriptor encoding; not a model execution profile."""


def scratchpad_rotation_descriptor(*, src, signs, dst, stats, blocks, inverse=False):
    """ABI 1.0, unit 3: N=1024 int8 blocks -> raw int32 + per-block raw absmax.

    All addresses are scratchpad-relative bytes, not host/DDR addresses. Actual
    capacity, overlap and descriptor-list protection are checked by gateware.
    Caller must check the target's seq.hadamard_abi before submitting this form.
    """
    for name, value in (("src", src), ("signs", signs), ("dst", dst), ("stats", stats)):
        if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(f"{name} must be a 32-byte-aligned unsigned 24-bit byte address")
    if not isinstance(blocks, int) or not 1 <= blocks < 1 << 16:
        raise ValueError("blocks must be in [1, 65535]")
    if inverse not in (False, True):
        raise ValueError("inverse must be boolean")
    return [3, src, signs, dst, blocks, stats, int(inverse)] + [0]*9

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Sequencer descriptor of the resident Hadamard unit (unit 3), board independent."""

# Hadamard (unit 3) --------------------------------------------------------------------------------

def scratchpad_rotation_descriptor(*, src, signs, dst, stats, blocks, inverse=False, half=False):
    """ABI 1.0, unit 3: N=1024 int8 blocks -> raw int32 + per-block raw absmax. half (ABI 1.1):
    blocks of 512 on the 1,024 unit (the inverse word's bit 1).

    All addresses are scratchpad-relative bytes, not host/DDR addresses. Actual
    capacity, overlap and descriptor-list protection are checked by gateware.
    Caller must check the target's seq.hadamard_abi before submitting this form.
    """
    for name, value in (("src", src), ("signs", signs), ("dst", dst), ("stats", stats)):
        if not isinstance(value, int) or value < 0 or value >= 1 << 24 or value % 32:
            raise ValueError(f"{name} must be a 32-byte-aligned unsigned 24-bit byte address")
    if not isinstance(blocks, int) or not 1 <= blocks < 1 << 16:
        raise ValueError("blocks must be in [1, 65535]")
    if inverse not in (False, True) or half not in (False, True):
        raise ValueError("inverse and half must be boolean")
    return [3, src, signs, dst, blocks, stats, int(inverse) | int(half) << 1] + [0]*9

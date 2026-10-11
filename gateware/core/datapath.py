#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Datapath registers carry no reset (doc/architecture.md, "Resets").

Most registers are marked at their declaration (`Signal(..., reset_less=True)`). For a pure
pipeline, where every register but its valids is datapath, `reset_less_registers` marks them
all after construction.
"""

from migen.fhdl.tools import list_targets

def reset_less_registers(module, keep=()):
    """Mark every register `module` itself assigns synchronously reset-less, except `keep` (its
    valids). Submodules are not touched."""
    keep = {id(signal) for signal in keep} # By identity: Migen's == builds an expression.
    for statements in module._fragment.sync.values():
        for target in list_targets(statements):
            if id(target) not in keep:
                target.reset_less = True

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

from migen import If, Signal
from migen.fhdl import tools as fhdl_tools
from migen.fhdl.namer import build_namespace

from trixium.migen_fast import linear_group_by_targets, linear_topological_sort


# Helpers ------------------------------------------------------------------------------------------

def _component_signature(groups):
    return {
        (frozenset(id(target) for target in targets),
         frozenset(id(statement) for statement in statements))
        for targets, statements in groups
    }


# Tests --------------------------------------------------------------------------------------------

def test_linear_statement_grouping_matches_migen_components():
    a, b, c, d, e = [Signal(name=name) for name in "abcde"]
    statements = [
        a.eq(b),
        c.eq(d),
        If(e, a.eq(c), b.eq(d)),  # Joins the a/b and c/d components transitively.
        e.eq(a),
    ]
    expected = fhdl_tools.group_by_targets(statements)
    actual = linear_group_by_targets(statements)
    assert _component_signature(actual) == _component_signature(expected)
    for _, group in actual:
        # Statements inside an always block retain source order.
        assert [statements.index(statement) for statement in group] == sorted(
            statements.index(statement) for statement in group)


def test_linear_topological_sort_matches_dependency_order_and_finds_cycle():
    a, b, c, d = [Signal(name=name) for name in "abcd"]
    targets = {a, b, c, d}
    deps = {a: set(), b: {a}, c: {a}, d: {b, c}}
    ns = build_namespace(targets)
    ordered = linear_topological_sort(targets, deps, ns)
    positions = {target: index for index, target in enumerate(ordered)}
    assert positions[a] < positions[b] < positions[d]
    assert positions[a] < positions[c] < positions[d]

    deps[a] = {d}
    assert linear_topological_sort(targets, deps, build_namespace(targets)) is None

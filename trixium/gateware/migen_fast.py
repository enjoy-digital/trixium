#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Linear-time replacements for two quadratic Migen/LiteX Verilog-conversion steps.

Migen groups combinational statements (``group_by_targets``) and LiteX sorts their
dependency graph (``_topological_sort_targets``) with repeated full-list scans. On designs
of this size those take many minutes (the Verilator simulation: ~25 min instead of ~1).
These replacements keep the same connected components and a dependency-first order, using
union-find and Kahn's algorithm. The emitted Verilog is equivalent, with the combinational
blocks in a different (deterministic) order.

Call :func:`install` once, before the SoC is converted; the targets and the simulation do.
"""

import collections
import heapq

from litex.gen.fhdl import verilog
from migen.fhdl import tools as fhdl_tools


def linear_topological_sort(targets, deps, ns):
    """Return dependency-first targets, or ``None`` when the graph has a cycle."""
    targets = set(targets)
    indegree = {}
    users = collections.defaultdict(list)
    for target in targets:
        direct = deps.get(target, set()) & targets
        indegree[target] = len(direct)
        for dep in direct:
            users[dep].append(target)

    heap = []
    for target, degree in indegree.items():
        if degree == 0:
            heapq.heappush(heap, (ns.get_name(target), id(target), target))
    ordered = []
    while heap:
        _, _, target = heapq.heappop(heap)
        ordered.append(target)
        for user in users[target]:
            indegree[user] -= 1
            if indegree[user] == 0:
                heapq.heappush(heap, (ns.get_name(user), id(user), user))
    return ordered if len(ordered) == len(targets) else None


def linear_group_by_targets(statements):
    """Group statements into target-sharing connected components in source order."""
    flat = list(fhdl_tools.flat_iteration(statements))
    parent = list(range(len(flat)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    statement_targets = []
    owner = {}
    for index, statement in enumerate(flat):
        targets = set(fhdl_tools.list_targets(statement))
        statement_targets.append(targets)
        for target in targets:
            if target in owner:
                union(index, owner[target])
            else:
                owner[target] = index

    components = collections.defaultdict(list)
    for index in range(len(flat)):
        components[find(index)].append(index)
    result = []
    for indices in sorted(components.values(), key=min):
        targets = set()
        for index in indices:
            targets |= statement_targets[index]
        result.append((targets, [flat[index] for index in sorted(indices)]))
    return result


def install():
    verilog._topological_sort_targets = linear_topological_sort
    verilog.group_by_targets          = linear_group_by_targets
    fhdl_tools.group_by_targets       = linear_group_by_targets

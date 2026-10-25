#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Cheaper ternary lanes (doc/plan-multirow.md, "Cheaper lanes"): the lanes select {~x, 0, x}
and the code-00 corrections (-x = ~x + 1) enter the adder tree as carry-ins. The group sum must
equal the reference sum of {-x, 0, +x, 0} lanes for every input.

Two levels: a pure-Python model of the lane + correction arithmetic (exhaustive over x and the
code), and a Migen simulation of the gateware helpers (_ternary_lanes + _adder_tree with carry-ins)
fed every (x, code) pair at every lane position plus extreme and random 128-lane groups.
"""

import random

from migen import *

from gateware.core.pipe import _ternary_lanes, _adder_tree


def _ref_lane(x, c):
    return {0: -x, 1: 0, 2: x, 3: 0}[c]


def _cheap_lane(x, c):
    return {0: ~x, 1: 0, 2: x, 3: 0}[c], int(c == 0)


def test_lane_model_exhaustive():
    for x in range(-128, 128):
        for c in range(4):
            v, k = _cheap_lane(x, c)
            assert -128 <= v <= 127 # 8-bit signed lane (x = -128: ~x = 127).
            assert v + k == _ref_lane(x, c)


def _pack(xs, cs):
    xword = 0
    codes = 0
    for j, (x, c) in enumerate(zip(xs, cs)):
        xword |= (x & 0xff) << (8*j)
        codes |= c << (2*j)
    return xword, codes


class _Dut(Module):
    def __init__(self):
        self.xword = Signal(1024)
        self.codes = Signal(256)
        en = Signal(reset=1)
        lanes, cins = _ternary_lanes(self, self.xword, self.codes, [en]*8)
        self.sum, self.regs = _adder_tree(self, lanes, cins=cins)


def _groups():
    pairs = [(x, c) for x in range(-128, 128) for c in range(4)] # 1024 (x, code) pairs.
    groups = []
    # Every pair at every lane position: lane j of group g gets pair (g + 8j) mod 1024.
    for g in range(1024):
        groups.append([pairs[(g + 8*j) % 1024] for j in range(128)])
    # Extremes (tree range limits).
    for x in (-128, -127, -1, 0, 1, 127):
        for c in range(4):
            groups.append([(x, c)]*128)
    rng = random.Random(1)
    for _ in range(300):
        groups.append([(rng.randrange(-128, 128), rng.randrange(4)) for _ in range(128)])
    return groups


def test_tree_exact():
    dut    = _Dut()
    groups = _groups()
    lat    = 1 + dut.regs # Lane register + tree stages (inputs applied one yield earlier).
    got    = []

    def gen():
        for k in range(len(groups) + lat + 1):
            if k < len(groups):
                xs, cs = zip(*groups[k])
                xw, cw = _pack(xs, cs)
                yield dut.xword.eq(xw)
                yield dut.codes.eq(cw)
            yield
            if k >= lat and len(got) < len(groups):
                got.append((yield dut.sum))

    run_simulation(dut, gen())
    assert len(got) == len(groups)
    for g, s in zip(groups, got):
        assert s == sum(_ref_lane(x, c) for x, c in g)

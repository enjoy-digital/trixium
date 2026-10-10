#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The fly-by channel of the calibration's SoC simulation (trixium_sim.py --board ypcb
--s7hr-model, gateware/memory/s7hrddrphy.py S7HRChannelModel) against its Python reference.

Compiled: the model's arithmetic must survive the Verilog conversion (a first version with
signed arithmetic read every bit bad at a negative CK phase in the SoC simulation)."""

import random

import pytest

from migen import *

from gateware.memory.s7hrddrphy import S7HRChannelModel

from common.verilator import run

CROSS   = [-30, -25, -20, -15, -10, -5, 0, 5]
BASE_WL = [0, 0, 6, 6, 0, 0, 6, 0]
V0      = [40 + (b % 8) + (24 if b == 27 else 0) for b in range(64)]


def reference(ck, wl, rb, tap, zone=3, half_eye=4):
    drift = ((ck + 128)*55) >> 8
    lanes = 0
    for l in range(8):
        need = (BASE_WL[l] + 2) & 7 if ck >= CROSS[l] else BASE_WL[l]
        if wl[l] != need or abs(ck - CROSS[l]) <= zone:
            lanes |= 1 << l
    bits = 0
    for b in range(64):
        if abs(rb[b]*12 + tap[b] + 27 - (drift + V0[b])) > half_eye:
            bits |= 1 << b
    return lanes, bits


@pytest.mark.parametrize("backend", ["verilator", pytest.param("migen", marks=pytest.mark.slow)])
def test_channel_model_matches_its_reference(backend):
    class _DUT(Module):
        def __init__(self):
            self.submodules.channel = ch = S7HRChannelModel(CROSS, BASE_WL, V0)
            self.ck, self.wl, self.rb, self.tap = ch.ck, ch.wl, ch.rb, ch.tap
            self.lane_bad, self.bit_bad = ch.lane_bad, ch.bit_bad
            tick = Signal(8) # The model is combinational: a clock for the compiled testbench.
            self.sync += tick.eq(tick + 1)
    dut     = _DUT()
    signals = {"ck": dut.ck, "lane_bad": dut.lane_bad, "bit_bad": dut.bit_bad}
    signals.update({f"wl{l}": s for l, s in enumerate(dut.wl)})
    signals.update({f"rb{b}": s for b, s in enumerate(dut.rb)})
    signals.update({f"tap{b}": s for b, s in enumerate(dut.tap)})

    def gen():
        rng = random.Random(1)
        for i in range(200):
            ck  = rng.randrange(-56, 56)
            wl  = [rng.choice((0, 2, 4, 6)) for _ in range(8)]
            rb  = [rng.randrange(8) for _ in range(64)]
            tap = [rng.randrange(32) for _ in range(64)]
            if i % 2:
                # Half the cases near each bit's eye, so both outcomes are exercised.
                drift = ((ck + 128)*55) >> 8
                tap = [max(0, min(31, V0[b] + drift - 27 - 12*rb[b] + rng.randrange(-6, 7)))
                       for b in range(64)]
            yield dut.ck.eq(ck)
            for l in range(8):
                yield dut.wl[l].eq(wl[l])
            for b in range(64):
                yield dut.rb[b].eq(rb[b])
                yield dut.tap[b].eq(tap[b])
            yield
            lanes, bits = reference(ck, wl, rb, tap)
            assert (yield dut.lane_bad) == lanes, (ck, wl)
            assert (yield dut.bit_bad) == bits, ck

    run(dut, [gen()], backend, signals=signals, memories={}, name="s7hr_channel")

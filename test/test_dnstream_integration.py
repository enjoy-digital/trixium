#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Engine op 5 through the shared cluster: channel readers -> stream lanes -> loader writers.

The plumbing test. A small geometry on integer operators (the unit's arithmetic is
qualified bit for bit on the exact operators in test_deltanet_stream.py) through the real
integration: CSR submit, per-channel reader offsets, the unit's scratchpad reads and
writes on the engine's ports, the loader's multi-channel store and the op's completion.
"""

import numpy as np
import pytest

from migen import run_simulation

from litex.gen import LiteXModule

from gateware.core.core import TriXiumConfig
from test_integration import SoC, write

MASK = 0xFFFFFFFF


# Helpers ------------------------------------------------------------------------------------------

class _IntAdd5(LiteXModule):
    latency = 5

    def __init__(self):
        from migen import Signal
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a + self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]


class _IntMul5(LiteXModule):
    latency = 5

    def __init__(self):
        from migen import Signal
        self.a, self.b, self.valid = Signal(32), Signal(32), Signal()
        self.result, self.result_valid = Signal(32), Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a*self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]


def int_reference(m, q, k, v, decay, beta, scale):
    m = [list(r) for r in m]
    out = []
    for j, row in enumerate(m):
        d = 0
        for i in range(len(row)):
            row[i] = (row[i]*decay) & MASK
            d = (d + row[i]*k[i]) & MASK
        delta = ((v[j] - d)*beta) & MASK
        o = 0
        for i in range(len(row)):
            row[i] = (row[i] + delta*k[i]) & MASK
            o = (o + row[i]*q[i]) & MASK
        out.append((o*scale) & MASK)
    return m, out


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('placement', [dict(), dict(chan_stride=0x10000, chan_mod=2, chan_skew=0x1000)])
def test_dnstream_op_through_the_cluster(placement):
    L, rows, cols, heads = 2, 16, 16, 5
    R, G = rows//L, cols//8
    config = TriXiumConfig(with_vecunit=True, scratchpad_size=16*1024, with_dnstream=True,
        dnstream_options=(("rows", rows), ("cols", cols), ("add_factory", _IntAdd5),
                          ("mul_factory", _IntMul5), ("negate", lambda x: -x)),
        **placement)
    dut = SoC(config, n=L)
    rng = np.random.default_rng(1)
    rnd = lambda *shape: rng.integers(0, 1 << 12, shape).tolist()
    states = [[rnd(cols) for _ in range(rows)] for _ in range(heads)]
    q, k = rnd(heads, cols), rnd(heads, cols)
    v = rnd(heads, rows)
    dec, bet, scl = rnd(heads), rnd(heads), rnd(heads)
    Q, V, O, T = 0x0000, 0x1000, 0x1800, 0x2000     # q|k per head, v, out, records
    wbase, wlen = 0x4000, heads*G*R
    offset = [0, 0x10000 + 0x1000] if placement else [0, 0]
    beat = lambda words: sum(w << (32*i) for i, w in enumerate(words))
    sp = {}
    for h in range(heads):
        for g in range(G):
            sp[Q//32 + 2*G*h + g] = beat(q[h][8*g:8*g + 8])
            sp[Q//32 + 2*G*h + G + g] = beat(k[h][8*g:8*g + 8])
        for b in range(rows//8):
            sp[V//32 + (rows//8)*h + b] = beat(v[h][8*b:8*b + 8])
        sp[T//32 + h] = beat([Q + 64*G*h, Q + 64*G*h + 32*G, V + 4*rows*h, O + 4*rows*h,
                              dec[h], bet[h], scl[h], 0])
    want = [int_reference(states[h], q[h], k[h], v[h], dec[h], bet[h], scl[h]) for h in range(heads)]

    def lane_beats(st, l):     # This lane's stream: beat g*R + r = row r*L + l, cols 8g..
        per = [[beat(st[h][r*L + l][8*g:8*g + 8]) for g in range(G) for r in range(R)] for h in range(heads)]
        return [b for h in per for b in h]

    def gen():
        for a, value in sp.items():
            yield dut.scratchpad.mem[a].eq(value)
        for l in range(L):
            for i, value in enumerate(lane_beats(states, l)):
                yield dut.models[l].mem[(offset[l] + wbase)//32 + i].eq(value)
        for name, value in dict(kind=5, rows=heads, wbase=wbase, wlen=wlen, xsrc=T).items():
            yield from write(getattr(dut.engine_csr, name), value)
        yield from write(dut.engine_csr.submit, 1)
        for cycle in range(20000):
            if (yield dut.engine_csr.done_count.status):
                break
            yield
        else:
            raise AssertionError("op 5 did not complete")
        print("op 5:", cycle, "cycles for", heads, "heads")
        yield
        for l in range(L):
            got = []
            for i in range(wlen):
                got.append((yield dut.models[l].mem[(offset[l] + wbase)//32 + i]))
            assert got == lane_beats([w[0] for w in want], l), l
        for h in range(heads):
            got = []
            for b in range(rows//8):
                value = yield dut.scratchpad.mem[O//32 + (rows//8)*h + b]
                got += [(value >> (32*i)) & MASK for i in range(8)]
            assert got == want[h][1], h
        assert not (yield dut.engine.busy)

    run_simulation(dut, [gen()], clocks={'sys': 10, 'eng': 10})
    assert 'TRIXIUM_HAS_DNSTREAM' in dut.constants

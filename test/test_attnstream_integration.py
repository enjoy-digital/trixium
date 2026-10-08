#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Engine op 6 through the shared cluster: position-striped rows -> merge -> attention lanes.

Plumbing, on integer operators (the arithmetic is qualified bit for bit on the exact
operators in test_attn_stream.py): CSR submit, the rows/rem split of T positions over the
channels with per-channel offsets, the round-robin merge back to position order, the unit's
record/operand reads and result writes on the engine's scratchpad ports, completion.
"""

import numpy as np
import pytest

from migen import run_simulation

from trixium.core.core import TriXiumConfig
from test_integration import SoC, write
from test_dnstream_integration import _IntAdd5, _IntMul5

MASK = 0xFFFFFFFF


@pytest.mark.parametrize('mode,T', [(0, 11), (1, 11), (0, 1), (1, 3)])
@pytest.mark.parametrize('placement', [dict(), dict(chan_stride=0x10000, chan_mod=2, chan_skew=0x1000)])
def test_attn_op_through_the_cluster(placement, mode, T):
    n, nq, n_rep, nkv, hd = 2, 4, 2, 2, 32
    hb, rowbeats = hd//8, nkv*hd//8
    config = TriXiumConfig(with_vecunit=True, scratchpad_size=16*1024, with_attnstream=True,
        attnstream_options=(("lanes", 4), ("groups", 2), ("hb_max", 4), ("t_max", 64),
                            ("add_factory", _IntAdd5), ("mul_factory", _IntMul5)),
        **placement)
    dut = SoC(config, n=n)
    rng = np.random.default_rng(T + 10*mode)
    rows = rng.integers(0, 1 << 10, (T, nkv, hd)).tolist()
    SRC, DST, REC = 0x0000, 0x1000, 0x3000
    tb = (T + 7)//8
    beat = lambda words: sum((w & MASK) << (32*i) for i, w in enumerate(words))
    sp = {}
    if mode == 0:
        opnd = rng.integers(0, 1 << 10, (nq, hd)).tolist()
        sstr, dstr, scale = 4*hd, 32*tb, 3
        for l in range(nq):
            for w in range(hb):
                sp[(SRC + l*sstr)//32 + w] = beat(opnd[l][8*w:8*w + 8])
        want = [[(sum(opnd[l][i]*rows[t][l//n_rep][i] for i in range(hd))*scale) & MASK
                 for t in range(T)] for l in range(nq)]
    else:
        opnd = rng.integers(0, 1 << 10, (nq, T)).tolist()
        sstr, dstr, scale = 32*tb, 4*hd, 0
        for l in range(nq):
            padded = opnd[l] + [0]*(8*tb - T)
            for w in range(tb):
                sp[(SRC + l*sstr)//32 + w] = beat(padded[8*w:8*w + 8])
        want = [[sum(opnd[l][t]*rows[t][l//n_rep][i] for t in range(T)) & MASK for i in range(hd)]
                for l in range(nq)]
    sp[REC//32] = beat([mode | nq << 8 | n_rep << 16 | nkv << 24, T, hd, SRC, sstr, DST, dstr, scale])
    offset = [0, 0x10000 + 0x1000] if placement else [0, 0]
    wbase = 0x4000

    def gen():
        for a, value in sp.items():
            yield dut.scratchpad.mem[a].eq(value)
        for t in range(T):          # position t: channel t % n, slot t // n
            c, slot = t % n, t//n
            flat = [x for kv in rows[t] for x in kv]
            for b in range(rowbeats):
                yield dut.models[c].mem[(offset[c] + wbase)//32 + slot*rowbeats + b].eq(beat(flat[8*b:8*b + 8]))
        for name, value in dict(kind=6, rows=T//n, rem=T % n, wbase=wbase, wlen=(T//n)*rowbeats,
                                sbeats=0, cbeats=rowbeats, xsrc=REC).items():
            yield from write(getattr(dut.engine_csr, name), value)
        yield from write(dut.engine_csr.submit, 1)
        for cycle in range(40000):
            if (yield dut.engine_csr.done_count.status):
                break
            yield
        else:
            raise AssertionError("op 6 did not complete")
        yield
        for l in range(nq):
            got = []
            count = T if mode == 0 else hd
            for w in range((count + 7)//8):
                value = yield dut.scratchpad.mem[(DST + l*dstr)//32 + w]
                got += [(value >> (32*i)) & MASK for i in range(8)]
            assert got[:count] == want[l], (l, got[:count], want[l])
        assert not (yield dut.engine.busy)
        assert not (yield dut.engine.error)

    run_simulation(dut, [gen()], clocks={'sys': 10, 'eng': 10})
    assert 'TRIXIUM_HAS_ATTNSTREAM' in dut.constants

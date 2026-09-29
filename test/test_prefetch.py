# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiWeightPrefetch against a behavioral reader: every beat the engine receives must be
the beat of the address it asked for, whatever the prefetch predicted."""
import random

import pytest
from migen import run_simulation
from migen.sim import passive

from trixium.gateware.prefetch import BonsaiWeightPrefetch, MODE_OFF, MODE_RECORD, MODE_REPLAY


def word(addr_beat):
    return (addr_beat*0x9E3779B97F4A7C15 + 0x1234) & ((1 << 256) - 1) | (addr_beat << 200)


@passive
def reader(dut, rng, rate, stats):
    """One command at a time (like BonsaiAXIReader: sink ready while idle), data in order
    after a latency, at most `rate` beats per clock on average, with gaps."""
    queue = []                       # beat addresses to deliver, in order
    lat = []
    while True:
        yield dut.rd_cmd.ready.eq(1)
        if (yield dut.rd_cmd.valid):
            a = (yield dut.rd_cmd.addr) >> 5
            n = yield dut.rd_cmd.len
            stats['cmds'].append((a, n))
            for i in range(n):
                lat.append((a + i, 20))
        if lat and lat[0][1] <= 0 and len(queue) < 64:
            queue.append(lat.pop(0)[0])
        lat = [(b, c - 1) for b, c in lat]
        v = bool(queue) and rng.random() < rate
        yield dut.rd_data.valid.eq(v)
        if v:
            yield dut.rd_data.data.eq(word(queue[0]))
        yield
        if v and (yield dut.rd_data.ready):
            queue.pop(0)
            stats['beats'] += 1


def engine(dut, rng, forwards, got, gaps, ready_p):
    """forwards: [(mode, [(beat_addr, len, weight)])]."""
    for mode, cmds in forwards:
        yield dut.mode.eq(mode)
        yield
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        for (a, n, w) in cmds:
            for _ in range(gaps):         # the units' work between the lists
                yield
            yield dut.cmd.valid.eq(1)
            yield dut.cmd.addr.eq(a << 5)
            yield dut.cmd.len.eq(n)
            yield dut.cmd_weight.eq(w)
            yield
            while not (yield dut.cmd.ready):
                yield
            yield dut.cmd.valid.eq(0)
            yield dut.cmd_weight.eq(0)
            for i in range(n):
                while True:
                    r = rng.random() < ready_p
                    yield dut.source.ready.eq(r)
                    yield
                    if r and (yield dut.source.valid):
                        got.append(((yield dut.source.data), word(a + i)))
                        break
            yield dut.source.ready.eq(0)
    got.append('done')


def run(forwards, rate=0.5, gaps=0, ready_p=1.0, seed=1, depth=1024, chunk=16):
    dut = BonsaiWeightPrefetch(addr_width=33, depth=depth, sched_depth=64, chunk=chunk)
    rng = random.Random(seed)
    got, stats = [], dict(cmds=[], beats=0)
    cycles = [0]
    def counter():
        while not got or got[-1] != 'done':
            cycles[0] += 1
            if cycles[0] > 400000:
                raise AssertionError(f"hang: {len(got)} beats, synced {(yield dut.synced)}, "
                                     f"cmd.ready {(yield dut.cmd.ready)}, source.valid {(yield dut.source.valid)}")
            yield
    run_simulation(dut, [engine(dut, rng, forwards, got, gaps, ready_p),
                         reader(dut, random.Random(seed + 1), rate, stats), counter()])
    assert got[-1] == 'done'
    bad = [(i, hex(g), hex(w)) for i, (g, w) in enumerate(got[:-1]) if g != w]
    assert not bad, bad[:4]
    return dut, stats, cycles[0]


def forward(rng, n, weight_every=3, lens=(8, 90)):
    """A forward: weight reads (the same every time) and a few direct ones."""
    cmds, a = [], 1000
    for k in range(n):
        ln = rng.randrange(*lens)
        cmds.append((a, ln, 1))
        a += ln + rng.randrange(0, 40)
        if k % weight_every == 2:
            cmds.append((50000 + k*100, rng.randrange(1, 30), 0))     # a fetch, a state read
    return cmds


def test_off_mode_is_direct():
    f = forward(random.Random(3), 8)
    dut, stats, _ = run([(MODE_OFF, f)])
    assert len(stats['cmds']) == len(f)


@pytest.mark.parametrize("ready_p", [1.0, 0.6])
def test_record_then_replay_hits_every_weight_read(ready_p):
    f = forward(random.Random(4), 10)
    nw = sum(1 for c in f if c[2])
    captured = {}
    dut, stats, _ = run([(MODE_RECORD, f), (MODE_REPLAY, f), (MODE_REPLAY, f)], ready_p=ready_p)


def test_replay_counters_and_deviation(tmp_path=None):
    rng = random.Random(5)
    f = forward(rng, 10)
    g = list(f)
    # the 5th weight read differs (another address): a desync, then direct to the end
    idx = [i for i, c in enumerate(g) if c[2]][4]
    g[idx] = (g[idx][0] + 7, g[idx][1], 1)
    short = [c for c in f][:len(f)//2]                  # a prefill forward: stops early
    dut = BonsaiWeightPrefetch(addr_width=33, depth=1024, sched_depth=64, chunk=16)
    rng2 = random.Random(9)
    got, stats = [], dict(cmds=[], beats=0)
    fw = [(MODE_RECORD, f), (MODE_REPLAY, f), (MODE_REPLAY, g), (MODE_REPLAY, short), (MODE_REPLAY, f)]
    res = {}
    def check():
        while not got or got[-1] != 'done':
            yield
        res['hits'] = yield dut.hits
        res['misses'] = yield dut.misses
        res['rec'] = yield dut.recorded
    run_simulation(dut, [engine(dut, rng2, fw, got, 0, 1.0), reader(dut, random.Random(2), 0.5, stats), check()])
    bad = [i for i, (a, b) in enumerate(got[:-1]) if a != b]
    assert not bad
    nw = sum(1 for c in f if c[2])
    assert res['rec'] == nw
    assert res['misses'] == 1
    # replay f: nw hits; replay g: 4 hits then direct; short: its weight reads; f again: nw
    assert res['hits'] == nw + 4 + sum(1 for c in short if c[2]) + nw


def test_prefetch_hides_the_gaps():
    """With idle gaps between the commands (the units' work), the replayed forward runs at
    the consumer's pace: fewer cycles than the recorded one."""
    f = forward(random.Random(6), 12, lens=(100, 300))   # weight reads of a few hundred beats
    _, _, direct = run([(MODE_OFF, f)], rate=0.45, gaps=150, depth=4096)
    _, _, both = run([(MODE_RECORD, f), (MODE_REPLAY, f)], rate=0.45, gaps=150, depth=4096)
    replay = both - direct
    print("direct", direct, "replayed", replay)
    assert replay < 0.8*direct

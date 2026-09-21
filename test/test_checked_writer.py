# SPDX-License-Identifier: BSD-2-Clause
from collections import deque
import random

import pytest
from migen import *
from migen.sim import passive
from litex.soc.interconnect.axi import AXIInterface

from trixium.gateware.axi_writer import BonsaiAXIWriter


def write_server(axi, observed, *, delay_b=0, aw_waits_w=False, response=0, bid=0, seed=37):
    """Allow W before AW; correlate responses only after both have completed."""
    rng = random.Random(seed)
    addresses, bursts, current = deque(), deque(), []
    held_aw = held_w = None
    @passive
    def run():
        nonlocal current, held_aw, held_w
        cycle, seen_w, responding = 0, False, False
        while True:
            # Seeing WVALID is enough to release AWREADY, not a W handshake.
            seen_w = seen_w or bool((yield axi.w.valid))
            yield axi.aw.ready.eq((not aw_waits_w or seen_w) and rng.randrange(4) != 0)
            yield axi.w.ready.eq(rng.randrange(4) != 0)
            responding = bool(addresses and bursts and cycle >= delay_b and rng.randrange(3))
            yield axi.b.valid.eq(responding)
            yield axi.b.resp.eq(response)
            yield axi.b.id.eq(bid)
            yield
            aw = ((yield axi.aw.addr), (yield axi.aw.len), (yield axi.aw.id),
                  (yield axi.aw.size), (yield axi.aw.burst))
            if held_aw is not None:
                assert (yield axi.aw.valid) and aw == held_aw
            held_aw = aw if (yield axi.aw.valid) and not (yield axi.aw.ready) else None
            w = ((yield axi.w.data), (yield axi.w.last), (yield axi.w.strb))
            if held_w is not None:
                assert (yield axi.w.valid) and w == held_w
            held_w = w if (yield axi.w.valid) and not (yield axi.w.ready) else None
            if (yield axi.aw.valid) and (yield axi.aw.ready):
                address, length, ident, size, burst = aw
                assert address % 32 == 0 and length < 16 and (address % 4096) + (length+1)*32 <= 4096
                assert ident == 0 and size == 5 and burst == 1
                addresses.append((address, length+1))
                observed.setdefault('aw', []).append((address, length+1))
            if (yield axi.w.valid) and (yield axi.w.ready):
                assert w[2] == 0xffffffff
                current.append(w[0])
                if w[1]:
                    bursts.append(current)
                    current = []
            if responding and (yield axi.b.ready):
                address, length = addresses.popleft()
                words = bursts.popleft()
                assert len(words) == length
                observed.setdefault('data', []).extend((address+32*i, word) for i, word in enumerate(words))
                observed['responses'] = observed.get('responses', 0) + 1
            cycle += 1
    return run()


@pytest.mark.parametrize('length,burst,credits,delay,aw_waits_w', [
    (1, 16, 1, 0, True), (39, 16, 8, 200, False), (300, 1, 4, 500, True),
])
def test_checked_writer_bounds_credits_and_preserves_exact_writes(length, burst, credits, delay, aw_waits_w):
    axi = AXIInterface(data_width=256, address_width=34, id_width=4)
    dut = BonsaiAXIWriter(axi, burst_beats=burst, checked=True, max_pending=credits)
    observed = {}
    address = (1 << 32) + 4096 - 32
    words = [random.Random(19+i).getrandbits(256) for i in range(length)]
    def run():
        yield dut.sink_cmd.addr.eq(address)
        yield dut.sink_cmd.len.eq(length)
        yield dut.sink_cmd.valid.eq(1)
        yield
        assert (yield dut.sink_cmd.ready)
        yield dut.sink_cmd.valid.eq(0)
        index = 0
        for cycle in range(10000):
            yield dut.sink_data.valid.eq(index < length)
            yield dut.sink_data.data.eq(words[index] if index < length else 0)
            yield
            if index < length and (yield dut.sink_data.ready):
                index += 1
            assert not (yield dut.error)
            assert (yield dut.outstanding) <= credits
            if cycle < delay-5:
                assert len(observed.get('aw', [])) <= credits
            if (yield dut.idle):
                assert index == length
                break
        else:
            raise AssertionError('checked writer did not complete')
        for _ in range(3):
            yield
        assert (yield dut.sink_cmd.ready)
    run_simulation(dut, [run(), write_server(axi, observed, delay_b=delay, aw_waits_w=aw_waits_w)])
    assert observed['data'] == [(address+32*i, word) for i, word in enumerate(words)]


@pytest.mark.parametrize('response,bid,expected', [(2, 0, 1), (0, 3, 6)])
def test_checked_writer_reports_bad_responses(response, bid, expected):
    axi = AXIInterface(data_width=256, address_width=33, id_width=4)
    dut = BonsaiAXIWriter(axi, checked=True)
    def run():
        yield dut.sink_cmd.addr.eq(0)
        yield dut.sink_cmd.len.eq(1)
        yield dut.sink_cmd.valid.eq(1)
        yield
        yield dut.sink_cmd.valid.eq(0)
        yield dut.sink_data.valid.eq(1)
        yield dut.sink_data.data.eq(123)
        yield
        for _ in range(100):
            if (yield dut.sink_data.ready):
                break
            yield
        else:
            raise AssertionError('writer did not accept its data')
        yield dut.sink_data.valid.eq(0)
        for _ in range(100):
            yield
        assert (yield dut.error) == expected
        assert not (yield dut.sink_cmd.ready)
        assert bool((yield dut.idle)) == (bid == 0)
    run_simulation(dut, [run(), write_server(axi, {}, response=response, bid=bid)])


@pytest.mark.parametrize('addr,length', [(1, 1), (0, 0)])
def test_checked_writer_rejects_invalid_command(addr, length):
    axi = AXIInterface(data_width=256, address_width=33, id_width=4)
    dut = BonsaiAXIWriter(axi, checked=True)
    def run():
        yield dut.sink_cmd.addr.eq(addr)
        yield dut.sink_cmd.len.eq(length)
        yield dut.sink_cmd.valid.eq(1)
        yield
        yield dut.sink_cmd.valid.eq(0)
        for _ in range(20):
            yield
            assert not (yield axi.aw.valid)
            assert not (yield axi.w.valid)
        assert (yield dut.error) == 8
    run_simulation(dut, run())


def test_checked_writer_rejects_unsolicited_response():
    axi = AXIInterface(data_width=256, address_width=33, id_width=4)
    dut = BonsaiAXIWriter(axi, checked=True)
    def run():
        yield axi.b.valid.eq(1)
        yield
        yield axi.b.valid.eq(0)
        yield
        assert (yield dut.error) == 4
        assert (yield dut.outstanding) == 0
        assert not (yield dut.sink_cmd.ready)
    run_simulation(dut, run())

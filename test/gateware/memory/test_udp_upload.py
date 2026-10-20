#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/udp_upload.py: datagrams in, the loader's stream out - only the expected
sequence number passes, whole; gaps, duplicates, partial beats and oversized datagrams are
dropped and counted; the output is the accepted payloads in order, under back-pressure."""

import random
from types import SimpleNamespace

from migen import *
from migen.sim import run_simulation

from litex.soc.interconnect import stream
from liteeth.common import eth_udp_user_description

from gateware.memory.udp_upload import UDPUpload


def test_only_the_expected_datagram_passes():
    port = SimpleNamespace(source=stream.Endpoint(eth_udp_user_description(32)))
    dut  = UDPUpload(port, max_beats=4, out_dw=128)
    rng  = random.Random(3)
    def payload(beats):
        return [rng.getrandbits(32) for _ in range(8*beats)]
    p0, p1, p2 = payload(2), payload(4), payload(1)
    datagrams = [
        (0, p0),                 # accepted
        (2, payload(1)),         # a gap: dropped
        (1, p1),                 # accepted (4 beats: the staging FIFO's maximum)
        (1, payload(1)),         # a duplicate: dropped
        (2, payload(1)[:5]),     # not a whole beat: dropped
        (2, payload(5)),         # larger than the FIFO: dropped
        (2, []),                 # no payload: dropped
        (2, p2),                 # accepted
    ]
    want = p0 + p1 + p2
    got  = []
    state = {}

    def sender():
        for seq, words in datagrams:
            for k, w in enumerate([seq] + words):
                yield port.source.data.eq(w)
                yield port.source.last.eq(k == len(words))
                yield port.source.valid.eq(1)
                yield
                while not (yield port.source.ready):
                    yield
            yield port.source.valid.eq(0)
            for _ in range(rng.randrange(0, 6)):
                yield

    def receiver():
        prng = random.Random(9)
        for _ in range(6000):
            yield dut.source.ready.eq(prng.randrange(3) != 0)
            yield
            if (yield dut.source.valid) and (yield dut.source.ready):
                v = (yield dut.source.data)
                got.extend((v >> (32*i)) & 0xFFFFFFFF for i in range(4))
        state["expected"] = (yield dut.expected.status)
        state["dropped"]  = (yield dut.dropped.status)
        state["accepted"] = (yield dut.accepted.status)

    run_simulation(dut, [sender(), receiver()])
    assert got == want, f"{len(got)} words out, {len(want)} wanted"
    assert state == {"expected": 3, "dropped": 5, "accepted": 3}, state

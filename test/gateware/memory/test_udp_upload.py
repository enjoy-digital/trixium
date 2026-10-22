#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/udp_upload.py: datagrams in, the loader's stream out - only the expected
sequence number passes, whole; gaps, duplicates, partial beats and oversized datagrams are
dropped and counted; the output is the accepted payloads in order, under back-pressure."""

import random
import struct
import zlib
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
        state["crc"]      = (yield dut.crc.status)

    run_simulation(dut, [sender(), receiver()])
    assert got == want, f"{len(got)} words out, {len(want)} wanted"
    # The CRC covers exactly the forwarded payload, as the host computes it.
    crc = state.pop("crc")
    assert crc == zlib.crc32(struct.pack(f"<{len(want)}I", *want)), f"crc {crc:08x}"
    assert state == {"expected": 3, "dropped": 5, "accepted": 3}, state


def test_crc_restarts_with_the_transfer():
    port = SimpleNamespace(source=stream.Endpoint(eth_udp_user_description(32)))
    dut  = UDPUpload(port, max_beats=2, out_dw=128)
    rng  = random.Random(5)
    a = [rng.getrandbits(32) for _ in range(16)]
    b = [rng.getrandbits(32) for _ in range(8)]
    crcs = []

    def send(seq, words):
        for k, w in enumerate([seq] + words):
            yield port.source.data.eq(w)
            yield port.source.last.eq(k == len(words))
            yield port.source.valid.eq(1)
            yield
        yield port.source.valid.eq(0)

    def gen():
        yield dut.source.ready.eq(1)
        yield from send(0, a)
        for _ in range(60): yield
        crcs.append((yield dut.crc.status))
        yield dut.restart.storage.eq(0)          # A new transfer: the CRC starts over.
        yield dut.restart.re.eq(1)
        yield
        yield dut.restart.re.eq(0)
        yield
        crcs.append((yield dut.crc.status))
        yield from send(0, b)
        for _ in range(60): yield
        crcs.append((yield dut.crc.status))
        crcs.append((yield dut.max_beats.status))

    run_simulation(dut, gen())
    assert crcs == [zlib.crc32(struct.pack("<16I", *a)), 0, zlib.crc32(struct.pack("<8I", *b)), 2], \
        [hex(c) for c in crcs]


def test_never_backpressures_the_udp_rx():
    # The UDP RX path is shared with Etherbone: with the loader stalled, datagrams arriving back
    # to back are dropped (never stalled), and the host's resend from `expected` still delivers
    # every payload once, in order. The debug CSRs describe the last datagram.
    port = SimpleNamespace(source=stream.Endpoint(eth_udp_user_description(32)))
    dut  = UDPUpload(port, max_beats=4, out_dw=128)
    rng  = random.Random(5)
    payloads = [[rng.getrandbits(32) for _ in range(8*rng.choice([1, 2, 4]))] for _ in range(12)]
    want  = [w for p in payloads for w in p]
    got   = []
    state = {"stalled_rx": 0, "sent": 0}

    def sender():
        while True:
            expected = (yield dut.expected.status)
            if expected == len(payloads):
                break
            for seq in range(expected, min(len(payloads), expected + 6)):
                words = [seq] + payloads[seq]
                for k, w in enumerate(words):
                    yield port.source.data.eq(w)
                    yield port.source.last.eq(k == len(words) - 1)
                    yield port.source.valid.eq(1)
                    yield
                    if not (yield port.source.ready):
                        state["stalled_rx"] += 1
                state["sent"] += 1
            yield port.source.valid.eq(0)
            for _ in range(40):
                yield
        state["last"] = ((yield dut.last_seq.status), (yield dut.last_words.status),
                         (yield dut.last_flags.status))

    def receiver():
        prng = random.Random(2)
        for cycle in range(40000):
            yield dut.source.ready.eq(cycle > 600 and prng.randrange(4) != 0)
            yield
            if (yield dut.source.valid) and (yield dut.source.ready):
                v = (yield dut.source.data)
                got.extend((v >> (32*i)) & 0xFFFFFFFF for i in range(4))
        state["accepted"] = (yield dut.accepted.status)
        state["dropped"]  = (yield dut.dropped.status)

    run_simulation(dut, [sender(), receiver()])
    assert state["stalled_rx"] == 0
    assert got == want, f"{len(got)} words out, {len(want)} wanted"
    assert state["accepted"] == len(payloads)
    assert state["dropped"] == state["sent"] - len(payloads) > 0
    assert state["last"] == (11, len(payloads[11]), 0b0011), state["last"]

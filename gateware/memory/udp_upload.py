#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""UDPUpload: weight uploads over UDP for boards without PCIe (LiteEth user port -> loader).

The host arms the loader through its CSRs (chan, base, length, start: Etherbone on the same
link), then sends the transfer's bytes as datagrams of `[seq, payload...]`, 32-bit little-endian
words, the payload a whole number of 256-bit beats. UDP may drop or reorder datagrams and has
no flow control, so the port forwards a datagram to the loader only when its `seq` is the one it
expects - anything else is dropped whole and counted - and the loader's memory only ever sees
the transfer's bytes in order. The host reads `expected` (the next sequence number wanted) back
and resends from there; `dropped` counts the datagrams refused (a wrong sequence number, a
payload not a whole number of beats, an underrun of the length field). `restart` sets the
expected number (to 0 for a new transfer).

The datagram is staged whole in a FIFO before it is released (its sequence number and size are
only known as complete at its end), so the FIFO holds one maximal datagram: `max_beats` beats.
The port never back-pressures the UDP RX path (it is LiteEth's, shared with Etherbone: a stall
there stops the CSR link too): a datagram that does not fit, while the loader drains earlier ones,
is dropped and resent like a lost one. `last_seq`/`last_words`/`last_flags` describe the last
datagram received (bring-up: why datagrams are refused).

`crc` is the CRC-32 (IEEE 802.3, zlib's) of every payload byte forwarded to the loader since the
last `restart`, the words taken little-endian: the host compares it with zlib.crc32 of the bytes
it sent, so a transfer is checked as it lands, without reading the memory back. `max_beats`
tells the host the largest datagram the port stages.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSR

from liteeth.mac.crc import LiteEthMACCRCEngine

# UDPUpload ----------------------------------------------------------------------------------------

class UDPUpload(LiteXModule):
    def __init__(self, udp_port, max_beats=64, out_dw=128):
        # udp_port: a LiteEth UDP user port (crossbar.get_port(PORT, dw=32)); its `source` carries
        # the received datagrams. self.source: the loader's upload stream (out_dw bits).
        assert len(udp_port.source.data) == 32
        self.source   = stream.Endpoint([("data", out_dw)])
        self.restart  = CSRStorage(32, description="Write: the next expected sequence number.")
        self.expected = CSRStatus(32,  description="The next expected sequence number.")
        self.dropped  = CSRStatus(32,  description="Datagrams dropped (sequence, size).")
        self.accepted = CSRStatus(32,  description="Datagrams forwarded to the loader.")
        self.crc      = CSRStatus(32,  description="CRC-32 (zlib's) of the payload forwarded since the last restart.")
        self.max_beats = CSRStatus(16, reset=max_beats, description="The largest datagram payload, in 256-bit beats.")

        # # #

        rx = udp_port.source
        words_per_beat = 8
        max_words      = max_beats*words_per_beat

        # Debug: the last datagram's sequence word, payload words and verdict flags.
        self.last_seq   = CSRStatus(32, description="The last datagram's sequence word.")
        self.last_words = CSRStatus(16, description="The last datagram's payload words.")
        self.last_flags = CSRStatus(4,  description="The last datagram: [0] sequence matched, "
            "[1] whole beats, [2] a word lost (staging full or past max_beats), [3] no verdict slot.")

        # Stage the payload words; mark each datagram's end with its verdict in a side FIFO.
        # The UDP RX path is shared (Etherbone): the port never back-pressures it. A datagram
        # that cannot be staged whole is dropped instead.
        self.fifo = fifo = stream.SyncFIFO([("data", 32)], max_words + 8, buffered=True)
        self.verdicts = verdicts = stream.SyncFIFO([("ok", 1)], 4)
        self.counts   = counts   = stream.SyncFIFO([("n", 16)], 4)
        expected = Signal(32)
        self.comb += self.expected.status.eq(expected)

        first  = Signal(reset=1)  # The next word is a datagram's first (its sequence number).
        good   = Signal()         # The datagram's sequence number matched.
        nwords = Signal(16)       # Payload words of the datagram so far.
        lost   = Signal()         # A payload word of the datagram was not staged.
        skip   = Signal()         # No verdict slot when the datagram began: nothing staged.
        seq_ok = Signal()
        room   = Signal()
        self.comb += [
            seq_ok.eq(rx.data == expected),
            room.eq(verdicts.sink.ready & counts.sink.ready),
            rx.ready.eq(1),
        ]

        word_in  = Signal()
        pushed   = Signal()
        skip_now = Signal()       # This datagram has no verdict slot (decided at its first word).
        lost_now = Signal()       # This datagram lost a word, this one included.
        self.comb += [
            word_in.eq(rx.valid),
            skip_now.eq(Mux(first, ~room, skip)),
            # Payload words go to the staging FIFO; the sequence word does not.
            fifo.sink.valid.eq(rx.valid & ~first & ~skip & ~lost & (nwords < max_words)),
            fifo.sink.data.eq(rx.data),
            pushed.eq(fifo.sink.valid & fifo.sink.ready),
            lost_now.eq(lost | (~first & ~pushed)),
        ]
        staged = Signal(16)       # Words of this datagram in the staging FIFO.
        self.sync += If(word_in,
            If(first, staged.eq(0)).Else(staged.eq(staged + pushed)),
        )
        size_ok = Signal()
        self.comb += size_ok.eq((nwords + 1)[:3] == 0)   # This last word completes a beat.
        self.sync += [
            If(self.restart.re,
                expected.eq(self.restart.storage),
            ),
            If(word_in,
                If(first,
                    good.eq(seq_ok),
                    nwords.eq(0),
                    lost.eq(0),
                    skip.eq(~room),
                    first.eq(rx.last),        # A datagram of one word: no payload.
                ).Else(
                    nwords.eq(nwords + 1),
                    lost.eq(lost_now),
                    first.eq(rx.last),
                ),
            ),
        ]
        # The verdict at the datagram's last word (one with no payload is refused).
        verdict = Signal()
        side    = Signal()
        self.comb += [
            verdict.eq(good & ~first & size_ok & ~lost_now & ~skip_now),
            side.eq(word_in & rx.last & ~skip_now),
            verdicts.sink.valid.eq(side),
            verdicts.sink.ok.eq(verdict),
            counts.sink.valid.eq(side),
            counts.sink.n.eq(Mux(first, 0, staged + pushed)),   # Every staged word drains once.
        ]
        self.sync += If(word_in & rx.last,
            If(verdict,
                expected.eq(expected + 1),
                self.accepted.status.eq(self.accepted.status + 1),
            ).Else(
                self.dropped.status.eq(self.dropped.status + 1),
            ),
            self.last_words.status.eq(Mux(first, 0, nwords + 1)),
            self.last_flags.status.eq(Cat(Mux(first, seq_ok, good), size_ok & ~first,
                lost_now & ~first, skip_now)),
        )
        self.sync += If(word_in & first, self.last_seq.status.eq(rx.data))

        # Release: with a verdict at the head, drain the staged words (to the loader, or away).
        # The staged words of one datagram are exactly those before its verdict was pushed: the
        # FIFO's words are counted per datagram in the counts FIFO.
        left  = Signal(16)
        busy  = Signal()
        keep  = Signal()
        conv  = stream.Converter(32, out_dw)
        self.conv = conv
        drain = Signal()
        self.comb += [
            drain.eq(fifo.source.valid & busy & (left != 0)),
            conv.sink.valid.eq(drain & keep),
            conv.sink.data.eq(fifo.source.data),
            fifo.source.ready.eq(busy & (left != 0) & (conv.sink.ready | ~keep)),
            conv.source.connect(self.source),
            verdicts.source.ready.eq(~busy & verdicts.source.valid & counts.source.valid),
            counts.source.ready.eq(~busy & verdicts.source.valid & counts.source.valid),
        ]
        # The forwarded words' CRC (IEEE 802.3: LiteEth's engine, the MAC's), restarted with the
        # transfer.
        self.crcgen = crcgen = LiteEthMACCRCEngine(data_width=32, width=32, polynom=0x04c11db7)
        crc_reg = Signal(32, reset=0xffffffff)
        self.comb += [
            crcgen.data.eq(fifo.source.data),
            crcgen.crc_prev.eq(crc_reg),
            self.crc.status.eq(crc_reg[::-1] ^ 0xffffffff),
        ]
        self.sync += If(self.restart.re,
            crc_reg.eq(0xffffffff),
        ).Elif(conv.sink.valid & conv.sink.ready,
            crc_reg.eq(crcgen.crc_next),
        )
        self.sync += [
            If(~busy & verdicts.source.valid & counts.source.valid,
                busy.eq(counts.source.n != 0),
                left.eq(counts.source.n),
                keep.eq(verdicts.source.ok),
            ).Elif(busy & fifo.source.valid & fifo.source.ready,
                left.eq(left - 1),
                If(left == 1, busy.eq(0)),
            ),
        ]

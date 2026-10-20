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
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSR

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

        # # #

        rx = udp_port.source
        words_per_beat = 8

        # Stage the payload words; mark each datagram's end with its verdict in a side FIFO.
        self.fifo = fifo = stream.SyncFIFO([("data", 32)], max_beats*words_per_beat + 8, buffered=True)
        self.verdicts = verdicts = stream.SyncFIFO([("ok", 1)], 4)
        counts_ready = Signal()
        expected = Signal(32)
        self.comb += self.expected.status.eq(expected)

        first  = Signal(reset=1)  # The next word is a datagram's first (its sequence number).
        good   = Signal()         # The datagram's sequence number matched.
        nwords = Signal(16)       # Payload words of the datagram so far.
        over   = Signal()         # More payload than the FIFO holds.
        seq_ok = Signal()
        self.comb += seq_ok.eq(rx.data == expected)

        word_in = Signal()
        self.comb += [
            word_in.eq(rx.valid & rx.ready),
            # Payload words go to the staging FIFO; the sequence word does not.
            fifo.sink.valid.eq(rx.valid & ~first & ~over),
            fifo.sink.data.eq(rx.data),
            rx.ready.eq((fifo.sink.ready | first | over) & verdicts.sink.ready & counts_ready),
        ]
        pushed = Signal()
        staged = Signal(16)       # Words of this datagram in the staging FIFO.
        self.comb += pushed.eq(fifo.sink.valid & fifo.sink.ready)
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
                    over.eq(0),
                    first.eq(rx.last),        # A datagram of one word: no payload.
                ).Else(
                    nwords.eq(nwords + 1),
                    If(nwords == max_beats*words_per_beat - 1, over.eq(1)),
                    first.eq(rx.last),
                ),
            ),
        ]
        # The verdict at the datagram's last word (one with no payload is refused).
        verdict = Signal()
        self.comb += [
            verdict.eq(good & ~first & size_ok & ~over),
            verdicts.sink.valid.eq(word_in & rx.last),
            verdicts.sink.ok.eq(Mux(first, 0, verdict)),
        ]
        self.sync += If(word_in & rx.last,
            If(~first & verdict,
                expected.eq(expected + 1),
                self.accepted.status.eq(self.accepted.status + 1),
            ).Else(
                self.dropped.status.eq(self.dropped.status + 1),
            )
        )

        # Release: with a verdict at the head, drain the staged words (to the loader, or away).
        # The staged words of one datagram are exactly those before its verdict was pushed: the
        # FIFO's words are counted per datagram in a third FIFO.
        self.counts = counts = stream.SyncFIFO([("n", 16)], 4)
        self.comb += [
            counts_ready.eq(counts.sink.ready),
            counts.sink.valid.eq(word_in & rx.last),
            counts.sink.n.eq(Mux(first, 0, staged + pushed)),   # Every staged word drains once.
        ]
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

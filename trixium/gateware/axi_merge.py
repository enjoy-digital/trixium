#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""A 256-bit AXI master onto a 512-bit slave with its read bursts MERGED, two beats per word.

LiteX's AXIUpConverter turns every narrow beat into its own wide single-beat transaction:
correct, pipelined, but a 512-bit DRAM word per 256-bit beat - twice the traffic, and on
the XCU1525 the four engine ports of a DDR4 controller are already near the controller's
word rate (v5: 0.74 words/cycle of a possible 1). This module keeps that converter for
the write path (STOREs and uploads, a small share of the traffic, byte strobes make the
half-word writes exact) and merges reads: a 16-beat 256-bit burst becomes one 512-bit
burst of 8 or 9 words (9 when it starts in the upper half of a word), and each word
coming back is split into its two halves, the unused half at the ends dropped. IDs pass
through; the R order is the AR order, which LiteDRAMAXI2Native guarantees.
"""

from migen import *

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface, AXIUpConverter, BURST_INCR


class AXIReadMerge(LiteXModule):
    """The read channels only: axi_from.ar/r (narrow) <-> axi_to.ar/r (wide)."""
    def __init__(self, axi_from, axi_to, depth=16):
        dw_from = len(axi_from.r.data)
        dw_to   = len(axi_to.r.data)
        assert dw_to == 2*dw_from, "two narrow beats per wide word"
        bytes_from = dw_from//8
        lsb = log2_int(bytes_from)          # The narrow beat's address bit selects the half.

        # AR: one wide burst per narrow burst. The wide length covers the words the narrow
        # beats touch: first word = addr >> (lsb+1), last = (addr + (len+1)*bytes - 1) >> (lsb+1).
        info = stream.SyncFIFO([("id", axi_from.id_width), ("nbeats", 9), ("upper", 1)], depth)
        self.submodules += info
        ar = axi_from.ar
        nbeats     = Signal(9)
        first_word = Signal(len(ar.addr))
        last_word  = Signal(len(ar.addr))
        self.comb += [
            nbeats.eq(ar.len + 1),
            first_word.eq(ar.addr >> (lsb + 1)),
            last_word.eq((ar.addr + (nbeats << lsb) - 1) >> (lsb + 1)),
            axi_to.ar.valid.eq(ar.valid & info.sink.ready),
            axi_to.ar.addr.eq(first_word << (lsb + 1)),
            axi_to.ar.len.eq(last_word - first_word),
            axi_to.ar.size.eq(log2_int(dw_to//8)),
            axi_to.ar.burst.eq(BURST_INCR),
            axi_to.ar.id.eq(ar.id),
            axi_to.ar.lock.eq(ar.lock), axi_to.ar.prot.eq(ar.prot), axi_to.ar.cache.eq(ar.cache),
            axi_to.ar.qos.eq(ar.qos), axi_to.ar.region.eq(ar.region),
            ar.ready.eq(axi_to.ar.ready & info.sink.ready),
            info.sink.valid.eq(ar.valid & axi_to.ar.ready),
            info.sink.id.eq(ar.id),
            info.sink.nbeats.eq(nbeats),
            info.sink.upper.eq(ar.addr[lsb]),
        ]

        # R: each wide word yields its lower half then its upper half, except that the first
        # word of a burst starting in the upper half yields only that, and the burst stops
        # after nbeats narrow beats (the last word's upper half may go unused).
        r     = axi_to.r
        half  = Signal()          # 0: lower half of the current word, 1: upper.
        count = Signal(9)         # Narrow beats delivered in this burst.
        first = Signal(reset=1)   # At the first word of the burst.
        cur_half = Signal()
        self.comb += [
            cur_half.eq(Mux(first & info.source.upper, 1, half)),
            axi_from.r.valid.eq(r.valid & info.source.valid),
            axi_from.r.data.eq(Mux(cur_half, r.data[dw_from:], r.data[:dw_from])),
            axi_from.r.resp.eq(r.resp),
            axi_from.r.id.eq(info.source.id),
            axi_from.r.last.eq(count == info.source.nbeats - 1),
        ]
        done_word = Signal()   # This narrow beat is the last one taken from the wide word.
        self.comb += done_word.eq(cur_half | axi_from.r.last)
        self.comb += [
            r.ready.eq(axi_from.r.ready & info.source.valid & done_word),
            info.source.ready.eq(axi_from.r.valid & axi_from.r.ready & axi_from.r.last),
        ]
        self.sync += If(axi_from.r.valid & axi_from.r.ready,
            If(axi_from.r.last,
                count.eq(0), half.eq(0), first.eq(1)
            ).Else(
                count.eq(count + 1),
                first.eq(0),
                half.eq(~cur_half),
            )
        )


class AXIWide(LiteXModule):
    """A 256-bit master onto a 512-bit slave: writes through LiteX's beat-wise up-converter,
    reads merged (AXIReadMerge)."""
    def __init__(self, axi_from, axi_to):
        w_to = AXIInterface(data_width=len(axi_to.r.data), address_width=axi_to.address_width,
                            id_width=axi_to.id_width)
        w_from = AXIInterface(data_width=len(axi_from.r.data), address_width=axi_from.address_width,
                              id_width=axi_from.id_width)
        self.submodules.writes = AXIUpConverter(w_from, w_to)
        self.submodules.reads  = AXIReadMerge(axi_from, axi_to)
        self.comb += [
            axi_from.aw.connect(w_from.aw),
            axi_from.w.connect(w_from.w),
            w_from.b.connect(axi_from.b),
            w_to.aw.connect(axi_to.aw),
            w_to.w.connect(axi_to.w),
            axi_to.b.connect(w_to.b),
            # The converter's own read channels stay idle.
            w_from.ar.valid.eq(0), w_to.r.ready.eq(0),
        ]

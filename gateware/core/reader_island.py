#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""ReaderIsland: a channel's AXI reader in the memory's own clock domain.

A channel's 256-bit AXI runs at the engine's clock: at 100 MHz it caps the weight stream at
3.2 GB/s against a DDR3-1066 controller's ~3.7 GB/s share per channel (the stream measured at
0.95 memory beats per engine clock on the YPCB with the pipes in their own 125 MHz island). In
the memory's domain (`cd`: the controller's clock with the engine's reset, `reng` of
gateware/core/reset.py) the reader issues and receives at the DRAM's rate and the port is no
longer a cap; the pipe island's input FIFO then takes the weight beats straight from this domain
(PipeIsland in_cd) and only the beats the engine itself consumes - FETCH, DeltaNet, attention -
cross back.

The engine-side interface keeps AXIReader's: `sink` (the command, in the engine's domain,
through a small async FIFO), `source` (the engine-consumed beats, through an async FIFO) and the
perf counters (sys-safe copies); plus `wsource`, the weight beats in `cd` (to PipeIsland.sink),
and two steering levels the engine registers from its latched command: `misc` (the op's beats
are not weights) and `consume` (beats may be taken: the prefetch gate, RUN). Both are quasi-static
- set at the op's start, read from the first beat on, >= 20 memory clocks later - and cross by
MultiReg. The reader frees its credits as beats leave it, so the FIFOs after it (the island's
input, `source`'s) bound nothing but latency: they can never overflow its data FIFO.

Commands: the FIFO's `ready` means room, not an idle reader, so the engine asserts a channel's
`sink.valid` for exactly one cycle (every channel ready, then advance; engine.py ISSUE).
"""

from migen import *
from migen.genlib.cdc import MultiReg

from litex.gen import *
from litex.soc.interconnect import stream

from gateware.memory.axi import AXIReader

# ReaderIsland -------------------------------------------------------------------------------------

class ReaderIsland(LiteXModule):
    def __init__(self, cd, axi, cmd_depth=4, misc_depth=16, **reader_kwargs):
        reader = AXIReader(axi, **reader_kwargs)
        # The engine's domain.
        self.sink    = stream.Endpoint(reader.sink.description)     # Commands.
        self.source  = stream.Endpoint([("data", 256)])             # FETCH / DeltaNet / attention beats.
        self.misc    = Signal()   # Level: the latched op's beats go to `source`, not the pipe.
        self.consume = Signal()   # Level: beats may be taken (the engine in RUN).
        self.idle        = Signal()     # The reader's, synchronised.
        self.total_beats = Signal(32)   # Copies of the reader's (valid at rest).
        self.ar_stalls   = Signal(32)
        self.r_stalls    = Signal(32)
        # The memory's domain.
        self.wsource = stream.Endpoint([("data", 256)])             # Weight beats (PipeIsland.sink).
        self.stat    = stream.Endpoint([("data", 256)])             # The reader's output after the gate, for statistics.

        # # #

        self.reader = ClockDomainsRenamer(cd)(reader)
        self.cmds = cmds = ClockDomainsRenamer({"write": "sys", "read": cd})(
            stream.AsyncFIFO(self.sink.description, cmd_depth))
        self.comb += [
            self.sink.connect(cmds.sink),
            cmds.source.connect(reader.sink),
        ]
        # Relay register between the reader's FIFO and the steering, for timing (as the engine's).
        self.rbuf = rbuf = ClockDomainsRenamer(cd)(stream.Buffer([("data", 256)]))
        self.comb += reader.source.connect(rbuf.sink)
        misc_cd, consume_cd = Signal(), Signal()
        self.specials += [
            MultiReg(self.misc,    misc_cd,    odomain=cd),
            MultiReg(self.consume, consume_cd, odomain=cd),
        ]
        self.miscf = miscf = ClockDomainsRenamer({"write": cd, "read": "sys"})(
            stream.AsyncFIFO([("data", 256)], misc_depth, buffered=True))
        gated = self.stat
        self.comb += [
            gated.valid.eq(rbuf.source.valid & consume_cd),
            gated.data.eq(rbuf.source.data),
            rbuf.source.ready.eq(gated.ready & consume_cd),
            self.wsource.valid.eq(gated.valid & ~misc_cd),
            self.wsource.data.eq(gated.data),
            miscf.sink.valid.eq(gated.valid & misc_cd),
            miscf.sink.data.eq(gated.data),
            gated.ready.eq(Mux(misc_cd, miscf.sink.ready, self.wsource.ready)),
            miscf.source.connect(self.source),
        ]
        self.specials += [
            MultiReg(reader.idle,        self.idle,        odomain="sys"),
            MultiReg(reader.total_beats, self.total_beats, odomain="sys"),
            MultiReg(reader.ar_stalls,   self.ar_stalls,   odomain="sys"),
            MultiReg(reader.r_stalls,    self.r_stalls,    odomain="sys"),
        ]

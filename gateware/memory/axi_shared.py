#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause


"""AXIIDSquash: a master's AXI requests under one ID, for a slave with a narrow ID.

The single-controller boards put every engine channel and the CPU's memory on one AXI port through
LiteX's AXIArbiter in its interleaved mode (litex#2689), which carries each master's index in the
upper ID bits: the Efinix controller's IDs are narrow, so the CPU's memory bus (several IDs) goes
through AXIIDSquash first.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

# AXIIDSquash --------------------------------------------------------------------------------------

class AXIIDSquash(LiteXModule):
    """A master's AXI with every request under ID 0, the original IDs restored on the way back
    from FIFOs in request order (AXI returns same-ID responses in order): makes room in a narrow
    slave ID (the Efinix controller's) for the interleaved AXIArbiter's master index, at the cost of the
    master's cross-ID parallelism. `depth`: the requests one direction may have in flight."""
    def __init__(self, master, slave, depth=16):
        assert master.data_width == slave.data_width
        # # #
        self.rq = rq = stream.SyncFIFO([("id", master.id_width)], depth)
        self.bq = bq = stream.SyncFIFO([("id", master.id_width)], depth)
        for ch in ("aw", "w", "ar"):
            self.comb += getattr(master, ch).connect(getattr(slave, ch), omit={"id", "valid", "ready"} if ch != "w" else set())
        self.comb += [
            slave.ar.id.eq(0),
            slave.ar.valid.eq(master.ar.valid & rq.sink.ready),
            master.ar.ready.eq(slave.ar.ready & rq.sink.ready),
            rq.sink.valid.eq(master.ar.valid & master.ar.ready),
            rq.sink.id.eq(master.ar.id),
            slave.aw.id.eq(0),
            slave.aw.valid.eq(master.aw.valid & bq.sink.ready),
            master.aw.ready.eq(slave.aw.ready & bq.sink.ready),
            bq.sink.valid.eq(master.aw.valid & master.aw.ready),
            bq.sink.id.eq(master.aw.id),
        ]
        self.comb += slave.r.connect(master.r, omit={"id"})
        self.comb += [
            master.r.id.eq(rq.source.id),
            rq.source.ready.eq(master.r.valid & master.r.ready & master.r.last),
        ]
        self.comb += slave.b.connect(master.b, omit={"id"})
        self.comb += [
            master.b.id.eq(bq.source.id),
            bq.source.ready.eq(master.b.valid & master.b.ready),
        ]

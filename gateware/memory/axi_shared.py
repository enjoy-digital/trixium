#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""AXISharedPort: several AXI masters onto one AXI slave port, reads interleaved.

A board with one memory controller and one AXI port (the Efinix Titanium's hard DDR controller)
carries every engine channel and the CPU's memory on that port. LiteX's AXIArbiter grants one
master until all of its requests have been answered: four channel readers, each keeping several
bursts in flight, would then stream one master's bursts at a time and leave the port idle
through every switch. Here:

- reads: AR is arbitrated per request (round robin, no lock), the master's index rides in the
  upper ARID bits, R beats are routed back by those bits - each master keeps its own bursts in
  flight and the slave serves them in whatever order its IDs allow (in order per ID, as AXI
  requires);
- writes: AW is arbitrated per burst; the granted masters' indices queue in AW order and W
  beats are taken from the queue head's master until WLAST (AXI4 has no WID: W follows AW
  order); B is routed back by the upper BID bits.

The slave's ID width must hold the masters' IDs and the index: `id_width(masters)`.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface


def _tag(m_id, width, idx, ib):
    """A master's ID padded to `width` bits, its index above."""
    pad = width - len(m_id)
    return Cat(m_id, C(0, pad), C(idx, ib)) if pad else Cat(m_id, C(idx, ib))

# Helpers ------------------------------------------------------------------------------------------

def index_bits(n):
    return max(1, (n - 1).bit_length())

def slave_id_width(masters):
    return max(m.id_width for m in masters) + index_bits(len(masters))

def _rr_next(current, requests, n):
    """Round robin: the first requesting index after `current`, else `current` (combinational)."""
    choice = Signal(max=max(2, n))
    cases = {}
    for c in range(n):
        stmt = choice.eq(current)
        for i in reversed([(c + 1 + k) % n for k in range(n)]):
            stmt = If(requests[i], choice.eq(i)).Else(stmt)
        cases[c] = stmt
    return choice, Case(current, cases)

# AXISharedPort ------------------------------------------------------------------------------------

class AXISharedPort(LiteXModule):
    def __init__(self, masters, slave, w_queue_depth=8):
        n = len(masters)
        ib = index_bits(n)
        mid = max(m.id_width for m in masters)
        assert slave.id_width >= mid + ib, "the slave's ID must hold the masters' IDs and the index"
        for m in masters:
            assert m.data_width == slave.data_width and m.address_width <= slave.address_width

        # # #

        # Read address: per-request round robin.
        ar_sel = Signal(max=max(2, n))
        ar_req = [m.ar.valid for m in masters]
        ar_pick, ar_case = _rr_next(ar_sel, ar_req, n)
        ar_cur = Signal(max=max(2, n))     # The master presenting this cycle.
        ar_hold = Signal()                 # A request was shown and not taken: keep it.
        self.comb += ar_case
        self.comb += If(ar_hold, ar_cur.eq(ar_sel)).Else(ar_cur.eq(ar_pick))
        for i, m in enumerate(masters):
            self.comb += If(ar_cur == i,
                slave.ar.valid.eq(m.ar.valid),
                slave.ar.addr.eq(m.ar.addr),
                slave.ar.len.eq(m.ar.len),
                slave.ar.size.eq(m.ar.size),
                slave.ar.burst.eq(m.ar.burst),
                slave.ar.id.eq(_tag(m.ar.id, mid, i, ib)),
                m.ar.ready.eq(slave.ar.ready),
            )
        self.sync += [
            ar_sel.eq(ar_cur),
            # AXI: a shown request stays until taken - keep the grant while it waits.
            ar_hold.eq(slave.ar.valid & ~slave.ar.ready),
        ]
        # Read data: back by the index bits.
        r_idx = Signal(ib)
        self.comb += r_idx.eq(slave.r.id[mid:mid + ib])
        for i, m in enumerate(masters):
            self.comb += [
                m.r.valid.eq(slave.r.valid & (r_idx == i)),
                m.r.data.eq(slave.r.data),
                m.r.resp.eq(slave.r.resp),
                m.r.last.eq(slave.r.last),
                m.r.id.eq(slave.r.id[:m.id_width]),
            ]
        self.comb += slave.r.ready.eq(Array([m.r.ready for m in masters])[r_idx])

        # Write address: per-burst round robin, the grants queued in AW order for W.
        self.wq = wq = stream.SyncFIFO([("idx", ib)], w_queue_depth)
        aw_sel = Signal(max=max(2, n))
        aw_req = [m.aw.valid for m in masters]
        aw_pick, aw_case = _rr_next(aw_sel, aw_req, n)
        aw_cur  = Signal(max=max(2, n))
        aw_hold = Signal()
        self.comb += aw_case
        self.comb += If(aw_hold, aw_cur.eq(aw_sel)).Else(aw_cur.eq(aw_pick))
        aw_valid = Signal()
        for i, m in enumerate(masters):
            self.comb += If(aw_cur == i,
                aw_valid.eq(m.aw.valid),
                slave.aw.addr.eq(m.aw.addr),
                slave.aw.len.eq(m.aw.len),
                slave.aw.size.eq(m.aw.size),
                slave.aw.burst.eq(m.aw.burst),
                slave.aw.id.eq(_tag(m.aw.id, mid, i, ib)),
                m.aw.ready.eq(slave.aw.ready & wq.sink.ready),
            )
        self.comb += [
            slave.aw.valid.eq(aw_valid & wq.sink.ready),
            wq.sink.valid.eq(slave.aw.valid & slave.aw.ready),
            wq.sink.idx.eq(aw_cur),
        ]
        self.sync += [
            aw_sel.eq(aw_cur),
            aw_hold.eq(slave.aw.valid & ~slave.aw.ready),
        ]
        # Write data: from the queue head's master until WLAST.
        w_idx = wq.source.idx
        for i, m in enumerate(masters):
            self.comb += If(wq.source.valid & (w_idx == i),
                slave.w.valid.eq(m.w.valid),
                slave.w.data.eq(m.w.data),
                slave.w.strb.eq(m.w.strb),
                slave.w.last.eq(m.w.last),
                m.w.ready.eq(slave.w.ready),
            )
        self.comb += wq.source.ready.eq(slave.w.valid & slave.w.ready & slave.w.last)
        # Write responses: back by the index bits.
        b_idx = Signal(ib)
        self.comb += b_idx.eq(slave.b.id[mid:mid + ib])
        for i, m in enumerate(masters):
            self.comb += [
                m.b.valid.eq(slave.b.valid & (b_idx == i)),
                m.b.resp.eq(slave.b.resp),
                m.b.id.eq(slave.b.id[:m.id_width]),
            ]
        self.comb += slave.b.ready.eq(Array([m.b.ready for m in masters])[b_idx])

# SPDX-License-Identifier: BSD-2-Clause
"""Passive response accounting for the shared AXI reader's checked transfers."""
from functools import reduce
from operator import or_

from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

ERROR_RESPONSE = 1
ERROR_ID = 2
ERROR_FRAMING = 4
ERROR_QUEUE = 8


class AXIReadMonitor(LiteXModule):
    """Check each accepted burst, including same-cycle AR/R and interleaved IDs.

    This observes, but never drives, AXI. Enable before issue and keep enabled
    through drain. Errors are sticky until the enclosing transport reset. An
    erroneous transfer must not be reused merely because ``drained`` is true.
    Queue depth is the shared reader's GLOBAL outstanding-burst bound, per ID;
    it does not depend on round-robin ID allocation for correctness.
    """
    def __init__(self, axi, *, n_ids=4, max_bursts=8):
        if not 1 <= n_ids <= (1 << len(axi.ar.id)) or max_bursts < 2:
            raise ValueError('invalid read monitor geometry')
        self.enable = Signal()
        self.error = Signal(4)
        self.drained = Signal()
        ar_fire, r_fire = Signal(), Signal()
        r_valid, r_id, r_last, r_resp = Signal(), Signal.like(axi.r.id), Signal(), Signal(2)
        self.comb += [ar_fire.eq(self.enable & axi.ar.valid & axi.ar.ready),
                      r_fire.eq(self.enable & axi.r.valid & axi.r.ready)]
        # Delay accepted R by one cycle. Even a zero-latency slave's metadata
        # is then in the queue before its first response is examined.
        self.sync += [r_valid.eq(r_fire), r_id.eq(axi.r.id),
                      r_last.eq(axi.r.last), r_resp.eq(axi.r.resp)]
        response_fault, id_fault, frame_fault, queue_fault = (Signal() for _ in range(4))
        unmatched, frames, overflows, pending = [], [], [], []
        for i in range(n_ids):
            fifo = stream.SyncFIFO([('length', 8)], max_bursts)
            setattr(self.submodules, f'burst{i}', fifo)
            beat = Signal(8)
            selected = r_valid & (r_id == i)
            final = beat == fifo.source.length
            self.comb += [
                fifo.sink.valid.eq(ar_fire & (axi.ar.id == i)),
                fifo.sink.length.eq(axi.ar.len),
                fifo.source.ready.eq(selected & final),
            ]
            self.sync += If(selected & fifo.source.valid,
                If(final, beat.eq(0)).Else(beat.eq(beat + 1)))
            unmatched.append(selected & ~fifo.source.valid)
            frames.append(selected & fifo.source.valid & (r_last != final))
            overflows.append(fifo.sink.valid & ~fifo.sink.ready)
            pending.append(fifo.source.valid)
        any_of = lambda values: reduce(or_, values, C(0, 1))
        self.comb += [
            response_fault.eq(r_valid & (r_resp != 0)),
            id_fault.eq((r_valid & (r_id >= n_ids)) | any_of(unmatched)
                        | (ar_fire & (axi.ar.id >= n_ids))),
            frame_fault.eq(any_of(frames)), queue_fault.eq(any_of(overflows)),
            self.drained.eq(~any_of(pending) & ~r_valid & ~ar_fire
                            & ~(self.enable & axi.r.valid)),
        ]
        self.sync += self.error.eq(self.error | Cat(response_fault, id_fault, frame_fault, queue_fault))

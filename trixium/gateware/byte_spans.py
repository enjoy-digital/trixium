# SPDX-License-Identifier: BSD-2-Clause
"""Bounded concatenation of byte slices from aligned 256-bit memory beats."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream


class ByteSpanPacker(LiteXModule):
    """Concatenate sink.data[skip:skip+count] into zero-tail-padded beats.

    skip/count are BYTE quantities; each accepted span has 1..32 valid bytes
    wholly inside its input beat. sink.last terminates the complete command,
    not each memory row. start requires idle; cancel flushes local storage only.
    Any error/cancellation invalidates all previously emitted command bytes.
    """
    def __init__(self):
        self.sink = stream.Endpoint([('data', 256), ('skip', 5), ('count', 6)])
        self.source = stream.Endpoint([('data', 256)])
        self.start, self.cancel = Signal(), Signal()
        self.busy, self.done, self.error = Signal(), Signal(), Signal()
        self.buffered_bytes = Signal(7)

        buffer = Signal(512)
        count = self.buffered_bytes
        ending, first = Signal(), Signal()
        pop, push, valid_span = Signal(), Signal(), Signal()
        span_end = Signal(7)
        remaining = Signal(7)
        shifted = Signal(512)
        input_shift = Signal(8)
        append_shift = Signal(9)
        sliced, masked = Signal(256), Signal(256)
        self.comb += [
            self.source.valid.eq(self.busy & (count != 0) & ((count >= 32) | ending)
                                 & ~self.cancel),
            self.source.data.eq(buffer[:256]), self.source.first.eq(first),
            self.source.last.eq(ending & (count <= 32)),
            pop.eq(self.source.valid & self.source.ready),
            remaining.eq(count - Mux(pop, Mux(count < 32, count, 32), 0)),
            shifted.eq(Mux(pop, buffer >> 256, buffer)),
            self.sink.ready.eq(self.busy & ~ending & (remaining <= 32) & ~self.cancel),
            push.eq(self.sink.valid & self.sink.ready),
            span_end.eq(self.sink.skip + self.sink.count),
            valid_span.eq((self.sink.count != 0) & (self.sink.count <= 32) & (span_end <= 32)),
            input_shift.eq(Cat(C(0, 3), self.sink.skip)),
            append_shift.eq(Cat(C(0, 3), remaining[:6])),
            sliced.eq(self.sink.data >> input_shift),
        ]
        # Mask BEFORE append: read-ahead/padding must not contaminate the next
        # row when another short span is ORed into the reservoir's high bytes.
        for i in range(32):
            self.comb += masked[8*i:8*(i+1)].eq(Mux(i < self.sink.count,
                                                         sliced[8*i:8*(i+1)], 0))
        self.sync += [
            self.done.eq(0),
            If(self.cancel,
                self.done.eq(self.busy), self.busy.eq(0), count.eq(0), buffer.eq(0), ending.eq(0),
            ).Elif(self.start & ~self.busy,
                self.busy.eq(1), self.error.eq(0), count.eq(0), buffer.eq(0),
                ending.eq(0), first.eq(1),
            ).Elif(self.busy,
                If(self.start, self.error.eq(1)),
                If(push & ~valid_span,
                    self.error.eq(1), self.busy.eq(0), self.done.eq(1),
                    count.eq(0), buffer.eq(0), ending.eq(0),
                ).Else(
                    If(push | pop,
                        buffer.eq(shifted | Mux(push, masked << append_shift, 0)),
                        count.eq(remaining + Mux(push, self.sink.count, 0)),
                    ),
                    If(push & self.sink.last, ending.eq(1)),
                    If(pop, first.eq(0)),
                    If(pop & self.source.last, self.busy.eq(0), self.done.eq(1)),
                ),
            ),
        ]

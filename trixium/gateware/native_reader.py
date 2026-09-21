# SPDX-License-Identifier: BSD-2-Clause
"""Region-addressed strided native reads through the existing AXI burst reader.

One physical endpoint per instance; the scheduler assigns commands to a reader
that reaches their region. Region IDs are not compute-pipe IDs. This optional
component does not alter board capabilities or the legacy command interface.
"""
from functools import reduce
from operator import and_

from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.capabilities import Access
from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.byte_spans import ByteSpanPacker
from trixium.gateware.native_weights import NativeWeightStream, GROUP_LAYOUT, PTQ1_0, Q8_0

ERROR_CONFIG = 1
ERROR_BOUNDS = 2
ERROR_FRAMING = 4
ERROR_RESPONSE = 8
ERROR_DECODER = 16
ERROR_PACKER = 32


class NativeRegionReader(LiteXModule):
    """Read rows of native blocks at region offset + row * stride.

    cmd fields are latched on valid/ready. offset/stride are 64-bit BYTE values;
    rows/groups are nonzero 16-bit counts. Group encodings use capability IDs.
    Every aligned physical span is checked against the selected region, its
    reservations and AXI address width BEFORE issuing that row. A later failing
    row invalidates the whole command, including any earlier emitted groups.

    cancel prevents new row commands, flushes packer/decoder and drains accepted
    reader commands. It does not retract ARVALID or reset an outstanding burst.
    Missing/malformed AXI responses may require the enclosing reset/drain shim;
    no timeout is invented here. done requires reader idle, including on failure.
    """
    def __init__(self, axi, regions, reservations=(), *, endpoint=0,
                 fifo_depth=128, n_ids=4, max_bursts=8, row_queue_depth=8):
        regions, reservations = tuple(regions), tuple(reservations)
        if not regions or len({r.id for r in regions}) != len(regions):
            raise ValueError('unique physical regions are required')
        if any(r.alignment != 32 for r in regions):
            raise ValueError('native reader currently requires 32-byte region alignment')
        if not 12 <= axi.address_width <= 64 or (1 << axi.id_width) < n_ids:
            raise ValueError('AXI address/ID width cannot represent this reader')
        if row_queue_depth < 2 or fifo_depth < 16:
            raise ValueError('native reader requires row queue >=2 and FIFO >=16')
        by_id = {r.id: r for r in regions}
        ordered = sorted(regions, key=lambda r: (r.endpoint, r.base))
        if any(a.endpoint == b.endpoint and a.base + a.size > b.base
               for a, b in zip(ordered, ordered[1:])):
            raise ValueError('physical regions on one endpoint overlap')
        for reservation in reservations:
            if (reservation.region_id not in by_id
                    or reservation.offset + reservation.size > by_id[reservation.region_id].size):
                raise ValueError('reservation is outside its region')

        self.cmd = stream.Endpoint([('region', 32), ('offset', 64), ('stride', 64),
                                    ('rows', 16), ('groups', 16), ('encoding', 8)])
        self.source = stream.Endpoint(GROUP_LAYOUT)
        self.cancel = Signal()
        self.busy, self.done, self.cancelled = Signal(), Signal(), Signal()
        self.error = Signal(6)
        self.reader = reader = BonsaiAXIReader(axi, fifo_depth=fifo_depth, n_ids=n_ids,
                                              max_bursts=max_bursts)
        self.packer = packer = ByteSpanPacker()
        self.decoder = decoder = NativeWeightStream()
        self.row_queue = meta = ResetInserter()(stream.SyncFIFO(
            [('skip', 5), ('beats', 24)], row_queue_depth))
        self.comb += packer.source.connect(decoder.sink)

        cfg_row_bytes = Signal(22)
        cfg_found, cfg_ok = Signal(), Signal()
        cfg_base, cfg_size = Signal(64), Signal(64)
        readable = [r for r in regions if r.endpoint == endpoint and r.access & Access.READ]
        self.comb += [
            cfg_row_bytes.eq(Mux(self.cmd.encoding == PTQ1_0, self.cmd.groups*28, self.cmd.groups*34)),
            cfg_found.eq(0), cfg_base.eq(0), cfg_size.eq(0),
            Case(self.cmd.region, {r.id: [cfg_found.eq(1), cfg_base.eq(r.base), cfg_size.eq(r.size)]
                                   for r in readable}),
            cfg_ok.eq(cfg_found & (self.cmd.rows != 0) & (self.cmd.groups != 0)
                      & ((self.cmd.encoding == PTQ1_0) | (self.cmd.encoding == Q8_0))
                      & ((self.cmd.rows == 1) | (self.cmd.stride >= cfg_row_bytes))),
        ]
        region = Signal(32)
        base, size, offset, stride = (Signal(64) for _ in range(4))
        rows_left = Signal(16)
        row_bytes = Signal(22)
        offset_wrapped = Signal()
        next_offset = Signal(65)
        aligned_offset = Signal(64)
        physical_start, physical_end = Signal(66), Signal(66)
        offset_end = Signal(65)
        span_rounded, span_bytes = Signal(23), Signal(23)
        beats = Signal(24)
        row_ok = Signal()
        bounds_fault, framing_fault, response_fault = Signal(), Signal(), Signal()
        abort, draining, decoded_done = Signal(), Signal(), Signal()
        start, issue, consume = Signal(), Signal(), Signal()
        faults = Signal(6)
        self.comb += [
            next_offset.eq(offset + stride),
            aligned_offset.eq(Cat(C(0, 5), offset[5:])),
            span_rounded.eq(offset[:5] + row_bytes + 31),
            beats.eq(span_rounded >> 5), span_bytes.eq(beats << 5),
            offset_end.eq(aligned_offset + span_bytes),
            physical_start.eq(base + aligned_offset), physical_end.eq(physical_start + span_bytes),
            row_ok.eq(~offset_wrapped & (offset_end <= size)
                      & (physical_end <= (1 << axi.address_width))
                      & reduce(and_, [((region != r.region_id) | (offset_end <= r.offset)
                                     | (aligned_offset >= r.offset + r.size))
                                      for r in reservations], C(1, 1))),
            bounds_fault.eq(self.busy & ~draining & (rows_left != 0) & ~row_ok),
            response_fault.eq(self.busy & axi.r.valid & axi.r.ready
                              & ((axi.r.resp != 0) | (axi.r.id >= n_ids))),
            faults.eq(Cat(C(0, 1), bounds_fault, framing_fault, response_fault,
                          decoder.error & self.busy, packer.error & self.busy)),
            abort.eq(self.busy & ~draining & (self.cancel | (faults != 0))),
            self.cmd.ready.eq(~self.busy & reader.idle & ~self.cancel),
            start.eq(self.cmd.valid & self.cmd.ready),
            decoder.start.eq(start & cfg_ok), decoder.cancel.eq(abort),
            decoder.rows.eq(self.cmd.rows), decoder.groups.eq(self.cmd.groups),
            decoder.encoding.eq(self.cmd.encoding),
            packer.start.eq(start & cfg_ok), packer.cancel.eq(abort),
            meta.reset.eq(start | abort),
            reader.sink.valid.eq(self.busy & ~draining & ~abort & (rows_left != 0)
                                 & row_ok & meta.sink.ready),
            reader.sink.addr.eq(physical_start), reader.sink.len.eq(beats),
            issue.eq(reader.sink.valid & reader.sink.ready),
            meta.sink.valid.eq(issue), meta.sink.skip.eq(offset[:5]),
            meta.sink.beats.eq(beats), meta.sink.last.eq(rows_left == 1),
            decoder.source.connect(self.source, omit={'valid', 'ready'}),
            self.source.valid.eq(decoder.source.valid & ~abort & ~draining),
            decoder.source.ready.eq(self.source.ready & ~abort & ~draining),
        ]

        beat_index = Signal(24)
        bytes_left = Signal(22)
        skip = Signal(5)
        available = Signal(6)
        expected_last = Signal()
        self.comb += [
            skip.eq(Mux(beat_index == 0, meta.source.skip, 0)),
            available.eq(32 - skip),
            expected_last.eq(beat_index == meta.source.beats - 1),
            framing_fault.eq(self.busy & ~draining & reader.source.valid & meta.source.valid
                             & (reader.source.last != expected_last)),
            packer.sink.valid.eq(self.busy & ~draining & ~abort
                                 & reader.source.valid & meta.source.valid),
            packer.sink.data.eq(reader.source.data), packer.sink.skip.eq(skip),
            packer.sink.count.eq(Mux(bytes_left < available, bytes_left, available)),
            packer.sink.last.eq(expected_last & meta.source.last),
            reader.source.ready.eq(Mux(draining | abort, 1, packer.sink.ready & meta.source.valid)),
            consume.eq(packer.sink.valid & packer.sink.ready),
            meta.source.ready.eq(consume & expected_last),
        ]
        self.sync += [
            self.done.eq(0),
            If(start,
                self.error.eq(Mux(cfg_ok, 0, ERROR_CONFIG)), self.cancelled.eq(0),
                self.busy.eq(cfg_ok), self.done.eq(~cfg_ok), draining.eq(0), decoded_done.eq(0),
                region.eq(self.cmd.region), base.eq(cfg_base), size.eq(cfg_size),
                offset.eq(self.cmd.offset), stride.eq(self.cmd.stride), rows_left.eq(self.cmd.rows),
                row_bytes.eq(cfg_row_bytes), bytes_left.eq(cfg_row_bytes), beat_index.eq(0),
                offset_wrapped.eq(0),
            ).Elif(self.busy,
                self.error.eq(self.error | faults),
                If(abort,
                    draining.eq(1), self.cancelled.eq(self.cancel),
                ),
                If(issue,
                    rows_left.eq(rows_left - 1), offset.eq(next_offset[:64]),
                    offset_wrapped.eq(next_offset[64]),
                ),
                If(consume,
                    If(expected_last, beat_index.eq(0), bytes_left.eq(row_bytes)
                    ).Else(beat_index.eq(beat_index + 1), bytes_left.eq(bytes_left - packer.sink.count)),
                ),
                If(decoder.done & ~abort & ~draining, decoded_done.eq(1)),
                If(~abort & reader.idle & (draining | decoded_done),
                    self.busy.eq(0), self.done.eq(1), draining.eq(0),
                ),
            ),
        ]

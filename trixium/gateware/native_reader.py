# SPDX-License-Identifier: BSD-2-Clause
"""Region-addressed strided native reads through the existing AXI burst reader.

One physical endpoint per instance; the scheduler assigns commands to a reader
that reaches their region. Region IDs are not compute-pipe IDs. This optional
component does not alter board capabilities or the legacy command interface.

Rows are beat-aligned on the device (offset and stride are multiples of 32
bytes; each row occupies ceil(groups*block/32) beats, see
model/qformats.py::native_row_span). The first implementation accepted any byte
offset through a byte-shifting packer and checked every row's bounds in one
combinational cone. On the v10 VCU build that family alone held all 110k
failing endpoints (28-33 logic levels, two chained 66-bit adders feeding
ARVALID and every cancel/reset) and 95k LUTs for sixteen pipes. Now the bounds
check is a three-stage registered pipeline ahead of issue, faults are latched
before they gate anything, and the packer is gone.
"""
from functools import reduce
from operator import and_

from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.capabilities import Access
from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.native_weights import NativeWeightStream, GROUP_LAYOUT, PTQ1_0, Q8_0

ERROR_CONFIG = 1
ERROR_BOUNDS = 2
ERROR_FRAMING = 4
ERROR_RESPONSE = 8
ERROR_DECODER = 16
ERROR_PACKER = 32  # Retired with the byte packer; the bit stays reserved.


class NativeReadTransport:
    """Port bundle for sharing an existing burst reader, not another RTL module.

    The integrator arbitrates sink/source and supplies idle. AXI responses on
    NativeRegionReader's axi argument must belong to that same transport; ID and
    response validation still observes that bus while a native command is busy.
    """
    def __init__(self, address_width):
        self.address_width = address_width
        self.sink = stream.Endpoint([('addr', address_width), ('len', 24)])
        self.source = stream.Endpoint([('data', 256)])
        self.idle = Signal()


class NativeRegionReader(LiteXModule):
    """Read rows of native blocks at region offset + row * stride.

    cmd fields are latched on valid/ready. offset/stride are 64-bit BYTE values
    and must be multiples of 32; rows/groups are nonzero 16-bit counts. Group
    encodings use capability IDs. Every row's physical span is checked against
    the selected region, its reservations and the AXI address width BEFORE that
    row is issued (three registered stages; a row is at least one beat, so the
    pipeline never starves the reader). A later failing row invalidates the
    whole command, including any earlier emitted groups.

    cancel prevents new row commands, flushes the decoder and drains accepted
    reader commands. It does not retract ARVALID or reset an outstanding burst.
    Missing/malformed AXI responses may require the enclosing reset/drain shim;
    no timeout is invented here. done requires reader idle, including on failure.
    Faults take effect one cycle after detection: at most one more beat enters
    the decoder and one more row command may issue, both discarded/drained.
    """
    def __init__(self, axi, regions, reservations=(), *, endpoint=0,
                 fifo_depth=128, n_ids=4, max_bursts=8, row_queue_depth=8, transport=None):
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
        if transport is not None and (not isinstance(transport, NativeReadTransport)
                                      or transport.address_width != axi.address_width):
            raise ValueError('shared transport must match the AXI address width')
        # A port bundle is not a Module: do not register an external AXI reader
        # a second time (which changes hierarchy and can duplicate its drivers).
        self.reader = reader = (transport if transport is not None else
            BonsaiAXIReader(axi, fifo_depth=fifo_depth, n_ids=n_ids, max_bursts=max_bursts))
        self.decoder = decoder = NativeWeightStream()
        # Rows issued but not yet consumed; only the command's last-row flag travels.
        self.row_queue = meta = ResetInserter()(stream.SyncFIFO([('cmd_last', 1)], row_queue_depth))

        aw = axi.address_width
        readable = [r for r in regions if r.endpoint == endpoint and r.access & Access.READ]
        # Reservation compares only need the widest region's offset range: a row
        # whose end exceeds its region bound fails anyway, and inside the bound
        # every offset fits nb bits, so truncated compares are exact.
        nb = max(r.size for r in regions).bit_length()

        # Command capture ---------------------------------------------------------------
        cfg_row_bytes = Signal(22)
        cfg_found, cfg_ok = Signal(), Signal()
        cfg_base, cfg_bound = Signal(64), Signal(65)
        self.comb += [
            # 28g = 32g - 4g and 34g = 32g + 2g: carry chains, not combinational DSPs.
            cfg_row_bytes.eq(Mux(self.cmd.encoding == PTQ1_0,
                                 (self.cmd.groups << 5) - (self.cmd.groups << 2),
                                 (self.cmd.groups << 5) + (self.cmd.groups << 1))),
            cfg_found.eq(0), cfg_base.eq(0), cfg_bound.eq(0),
            # bound = min(size, 2**aw - base): a row end at or below it lies inside
            # the region AND the AXI address space, so only one compare runs later.
            Case(self.cmd.region, {r.id: [cfg_found.eq(1), cfg_base.eq(r.base),
                                          cfg_bound.eq(min(r.size, max(0, (1 << aw) - r.base)))]
                                   for r in readable}),
            cfg_ok.eq(cfg_found & (self.cmd.rows != 0) & (self.cmd.groups != 0)
                      & ((self.cmd.encoding == PTQ1_0) | (self.cmd.encoding == Q8_0))
                      & (self.cmd.offset[:5] == 0) & (self.cmd.stride[:5] == 0)
                      & ((self.cmd.rows == 1) | (self.cmd.stride >= cfg_row_bytes))),
        ]
        base = Signal(64)
        bound = Signal(65)
        res_active = Signal(max(1, len(reservations)))
        stride = Signal(64)
        row_bytes = Signal(22)
        row_rounded = Signal(23)    # row_bytes + 31: an explicit width, never a nested shift operand
        beats = Signal(18)          # ceil(row_bytes/32) <= 2**17
        span_bytes = Signal(23)
        last_beat = Signal(18)
        prepared = Signal()         # beats/span/last_beat derived from row_bytes
        draining, decoded_done = Signal(), Signal()
        abort = Signal()            # registered one-cycle pulse
        start, issue, consume = Signal(), Signal(), Signal()

        # Row generator -> S1 -> S2 -> S3 -> issue ----------------------------------------
        offset = Signal(64)
        offset_wrapped = Signal()
        next_offset = Signal(65)
        rows_left = Signal(16)
        gen_valid, gen_take = Signal(), Signal()
        s1_valid, s1_take, s1_wrapped, s1_last = Signal(), Signal(), Signal(), Signal()
        s1_offset = Signal(nb)
        s1_offset_end = Signal(65)
        s1_start = Signal(aw)
        s2_valid, s2_take, s2_ok, s2_last = Signal(), Signal(), Signal(), Signal()
        s2_start = Signal(aw)
        s3_valid, s3_take, s3_ok, s3_last = Signal(), Signal(), Signal(), Signal()
        s3_start = Signal(aw)
        in_bound = Signal()
        clear_of = [Signal(name=f'clear_of{i}') for i in range(len(reservations))]
        bounds_fault, framing_fault, response_fault = Signal(), Signal(), Signal()
        faults = Signal(6)
        self.comb += [
            row_rounded.eq(row_bytes + 31),
            next_offset.eq(offset + stride),
            gen_valid.eq(self.busy & ~draining & ~abort & prepared & (rows_left != 0)),
            # Each stage takes when empty or when its successor takes this cycle.
            s3_take.eq(~s3_valid | issue),
            s2_take.eq(~s2_valid | s3_take),
            s1_take.eq(~s1_valid | s2_take),
            gen_take.eq(gen_valid & s1_take),
            # S1 arithmetic: two independent adders, no compare.
            in_bound.eq(s1_offset_end <= bound),
        ] + [clear_of[i].eq(~res_active[i] | (s1_offset_end[:nb] <= r.offset)
                            | (s1_offset >= r.offset + r.size))
             for i, r in enumerate(reservations)] + [
            reader.sink.valid.eq(s3_valid & s3_ok & ~draining & ~abort & meta.sink.ready),
            reader.sink.addr.eq(s3_start), reader.sink.len.eq(beats),
            issue.eq(reader.sink.valid & reader.sink.ready),
            meta.sink.valid.eq(issue), meta.sink.cmd_last.eq(s3_last),
            bounds_fault.eq(s3_valid & ~s3_ok & ~draining),
            response_fault.eq(self.busy & axi.r.valid & axi.r.ready
                              & ((axi.r.resp != 0) | (axi.r.id >= n_ids))),
            faults.eq(Cat(C(0, 1), bounds_fault, framing_fault, response_fault,
                          decoder.error & self.busy, C(0, 1))),
            self.cmd.ready.eq(~self.busy & reader.idle & ~self.cancel),
            start.eq(self.cmd.valid & self.cmd.ready),
            decoder.start.eq(start & cfg_ok), decoder.cancel.eq(abort),
            decoder.rows.eq(self.cmd.rows), decoder.groups.eq(self.cmd.groups),
            decoder.encoding.eq(self.cmd.encoding),
            meta.reset.eq(start | abort),
            decoder.source.connect(self.source, omit={'valid', 'ready'}),
            self.source.valid.eq(decoder.source.valid & ~abort & ~draining),
            decoder.source.ready.eq(self.source.ready & ~abort & ~draining),
        ]
        self.sync += [
            If(start | abort,
                s1_valid.eq(0), s2_valid.eq(0), s3_valid.eq(0),
            ).Else(
                If(s1_take,
                    s1_valid.eq(gen_valid),
                    s1_offset.eq(offset[:nb]), s1_wrapped.eq(offset_wrapped),
                    s1_last.eq(rows_left == 1),
                    s1_offset_end.eq(offset + span_bytes),
                    s1_start.eq((base + offset)[:aw]),
                ),
                If(s2_take,
                    s2_valid.eq(s1_valid), s2_last.eq(s1_last), s2_start.eq(s1_start),
                    s2_ok.eq(~s1_wrapped & in_bound & reduce(and_, clear_of, C(1, 1))),
                ),
                If(s3_take,
                    s3_valid.eq(s2_valid), s3_last.eq(s2_last), s3_start.eq(s2_start),
                    s3_ok.eq(s2_ok),
                ),
            ),
        ]

        # Response side ------------------------------------------------------------------
        beat_index = Signal(18)
        expected_last = Signal()
        self.comb += [
            expected_last.eq(beat_index == last_beat),
            framing_fault.eq(self.busy & ~draining & reader.source.valid & meta.source.valid
                             & (reader.source.last != expected_last)),
            decoder.sink.valid.eq(self.busy & ~draining & ~abort
                                  & reader.source.valid & meta.source.valid),
            decoder.sink.data.eq(reader.source.data),
            decoder.sink.last.eq(expected_last & meta.source.cmd_last),
            reader.source.ready.eq(Mux(draining | abort, 1, decoder.sink.ready & meta.source.valid)),
            consume.eq(decoder.sink.valid & decoder.sink.ready),
            meta.source.ready.eq(consume & expected_last),
        ]
        self.sync += [
            self.done.eq(0),
            abort.eq(0),
            If(start,
                self.error.eq(Mux(cfg_ok, 0, ERROR_CONFIG)), self.cancelled.eq(0),
                self.busy.eq(cfg_ok), self.done.eq(~cfg_ok), draining.eq(0), decoded_done.eq(0),
                base.eq(cfg_base), bound.eq(cfg_bound),
                res_active.eq(Cat(*[self.cmd.region == r.region_id for r in reservations])
                              if reservations else 0),
                offset.eq(self.cmd.offset), stride.eq(self.cmd.stride), rows_left.eq(self.cmd.rows),
                row_bytes.eq(cfg_row_bytes), beat_index.eq(0), offset_wrapped.eq(0), prepared.eq(0),
            ).Elif(self.busy,
                self.error.eq(self.error | faults),
                If(~prepared,
                    beats.eq(row_rounded[5:]), span_bytes.eq(Cat(C(0, 5), row_rounded[5:])),
                    last_beat.eq(row_rounded[5:] - 1), prepared.eq(1),
                ),
                If(~draining & ~abort & (self.cancel | (faults != 0)),
                    abort.eq(1), self.cancelled.eq(self.cancel),
                ),
                If(abort, draining.eq(1)),
                If(gen_take,
                    rows_left.eq(rows_left - 1), offset.eq(next_offset[:64]),
                    offset_wrapped.eq(next_offset[64]),
                ),
                If(consume,
                    If(expected_last, beat_index.eq(0)).Else(beat_index.eq(beat_index + 1)),
                ),
                If(decoder.done & ~abort & ~draining, decoded_done.eq(1)),
                If(~abort & reader.idle & (draining | decoded_done),
                    self.busy.eq(0), self.done.eq(1), draining.eq(0),
                ),
            ),
        ]

# SPDX-License-Identifier: BSD-2-Clause
"""Compact embedding restoration over the existing shared scratchpad ports."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.embedding_restore import EmbeddingRestore
from trixium.gateware.native_weights import GROUP_LAYOUT, NativeWeightStream, PTQ1_0

EMBEDDING_COMMAND = [('src', 24), ('signs', 24), ('dst', 24), ('groups', 16),
                     ('protect_start', 24), ('protect_end', 24)]


class BonsaiEmbeddingScratchpad(LiteXModule):
    """Read one compact PTQ row and inverse signs; write its restored FP32 row.

    Bases are 32-byte aligned. groups is a positive multiple of eight, within
    the configured kernel capacity. Source/sign/destination byte spans are
    groups*28, groups*16 and groups*512. All full spans and the protected output
    interval are checked before reads. Source and signs may alias each other or
    output: the entire input is cached/validated before the first output write.
    Reusing overwritten inputs requires reloading them after any such operation.

    No extra scratchpad port: the owner grants every requested read and returns
    data one clock later; writes require sp_wr_req & sp_wr_gnt. A one-beat sign
    cache serves each adjacent pair of semantic groups. Only one read is in
    flight, tagged as raw data or signs; one raw holding beat absorbs decoder
    backpressure. Inputs must remain immutable until captured, and the owner
    must drain older writes and serialize other memory clients.

    error bits 0 geometry, 1 bounds, 2 protected output, 3 native decoder,
    4 read/output accounting, 5 semantic metadata, 8..13 kernel errors. Failure
    stays busy until cancel/reset; discard the entire output, including any
    written prefix. done follows the last granted output and a drain cycle.
    exponent is the row's BFP exponent, valid only after successful completion.
    """
    def __init__(self, scratchpad_size=256*1024, max_groups=64, input_width=16):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError('scratchpad must be aligned within the 13-bit address space')
        self.cmd = stream.Endpoint(EMBEDDING_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(16)
        self.read_beats, self.output_beats = Signal(32), Signal(32)
        self.exponent = Signal((10, True))
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)
        self.kernel = kernel = EmbeddingRestore(max_groups, input_width)
        self.decoder = decoder = NativeWeightStream()
        # The byte decoder suppresses source.valid combinationally on bad
        # incoming framing; its sink.ready also depends on source.ready. Do not
        # feed that valid through metadata/failure/sign arbitration back into
        # either input valid or output ready. An elastic registered group makes
        # the client's decisions independent of this upstream clear path.
        group = stream.Endpoint(GROUP_LAYOUT)

        IDLE, VALIDATE, START, RUN, FINISH, FAILED = range(6)
        state = Signal(3, reset=IDLE)
        captured = {name: Signal(width) for name, width in EMBEDDING_COMMAND}
        ends = {name: Signal(64) for name in ('src', 'signs', 'dst')}
        geometry_bad, bounds_bad, protected_bad = Signal(), Signal(), Signal()
        raw_count, output_count = Signal(20), Signal(20)
        self.comb += [
            raw_count.eq((captured['groups'] * 28) >> 5), output_count.eq(captured['groups'] << 4),
            ends['src'].eq(captured['src'] + captured['groups']*28),
            ends['signs'].eq(captured['signs'] + (captured['groups'] << 4)),
            ends['dst'].eq(captured['dst'] + (captured['groups'] << 9)),
            geometry_bad.eq((captured['groups'] == 0) | (captured['groups'] > max_groups) |
                (captured['groups'][:3] != 0) | (captured['src'][:5] != 0) |
                (captured['signs'][:5] != 0) | (captured['dst'][:5] != 0) |
                (captured['protect_start'] > captured['protect_end'])),
            bounds_bad.eq((ends['src'] > scratchpad_size) | (ends['signs'] > scratchpad_size) |
                (ends['dst'] > scratchpad_size) | (captured['protect_end'] > scratchpad_size)),
            protected_bad.eq((captured['protect_start'] != captured['protect_end']) &
                (captured['dst'] < captured['protect_end']) & (captured['protect_start'] < ends['dst'])),
        ]
        active, clear = Signal(), Signal()
        pending, pending_sign, pending_last = Signal(), Signal(), Signal()
        pending_tag = Signal(16)
        raw_valid, raw_last = Signal(), Signal()
        raw_data = Signal(256)
        issued = Signal(20)
        sign_valid, sign_match = Signal(), Signal()
        sign_data = Signal(256)
        sign_tag = Signal(16)
        sign_pair = Signal(16)
        input_done, last_written = Signal(), Signal()
        read_sign, read_raw, raw_pop, written = Signal(), Signal(), Signal(), Signal()
        metadata_bad, accounting_bad, failed = Signal(), Signal(), Signal()
        self.comb += [
            active.eq(state == RUN), clear.eq(self.cancel | (state == IDLE) | (state == VALIDATE) | (state == FAILED)),
            self.cmd.ready.eq((state == IDLE) & ~self.cancel), self.busy.eq(state != IDLE),
            kernel.cancel.eq(clear), decoder.cancel.eq(clear),
            kernel.cmd.valid.eq((state == START) & ~self.cancel), kernel.cmd.groups.eq(captured['groups']),
            decoder.start.eq(kernel.cmd.valid & kernel.cmd.ready), decoder.rows.eq(1),
            decoder.groups.eq(captured['groups']), decoder.encoding.eq(PTQ1_0),
            sign_pair.eq(group.group >> 1), sign_match.eq(sign_valid & (sign_tag == sign_pair)),
            metadata_bad.eq(group.valid & ((group.row != 0) |
                group.kind | group.error | (group.group >= captured['groups']))),
            accounting_bad.eq((pending & ~pending_sign & raw_valid) |
                (kernel.source.valid & (~input_done | (kernel.source.first != (self.output_beats == 0)) |
                    (kernel.source.last != (self.output_beats == output_count-1)))) |
                (kernel.done & (~last_written | (self.output_beats != output_count)))),
            failed.eq(active & (decoder.error | (kernel.error != 0) | metadata_bad | accounting_bad)),
            decoder.sink.valid.eq(raw_valid & active & ~failed & ~self.cancel), decoder.sink.data.eq(raw_data),
            decoder.sink.last.eq(raw_last), raw_pop.eq(decoder.sink.valid & decoder.sink.ready),
            kernel.sink.valid.eq(group.valid & sign_match & active & ~failed & ~self.cancel),
            group.ready.eq(kernel.sink.ready & sign_match & active & ~failed & ~self.cancel),
            decoder.source.ready.eq((~group.valid | group.ready) & active & ~failed & ~self.cancel),
            kernel.sink.codes.eq(group.codes), kernel.sink.scale.eq(group.scale),
            kernel.sink.negate.eq(Mux(group.group[0], sign_data[128:], sign_data[:128])),
            kernel.sink.first.eq(group.first), kernel.sink.last.eq(group.last),
            read_sign.eq(active & ~pending & group.valid & ~sign_match & ~metadata_bad),
            read_raw.eq(active & ~pending & ~read_sign & (issued < raw_count) & (~raw_valid | raw_pop)),
            self.sp_rd_req.eq((read_sign | read_raw) & ~failed & ~self.cancel),
            self.sp_rd_adr.eq(Mux(read_sign, (captured['signs'] >> 5) + sign_pair,
                                 (captured['src'] >> 5) + issued)),
            self.sp_wr_req.eq(active & kernel.source.valid & ~failed & ~self.cancel),
            self.sp_wr_adr.eq((captured['dst'] >> 5) + self.output_beats),
            self.sp_wr_dat.eq(kernel.source.data), self.sp_wr_we.eq(255),
            kernel.source.ready.eq(active & self.sp_wr_gnt & ~failed & ~self.cancel),
            written.eq(self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear,
            group.valid.eq(0), pending.eq(0), raw_valid.eq(0), sign_valid.eq(0), input_done.eq(0), last_written.eq(0),
        ).Else(
            If(decoder.source.ready,
                group.valid.eq(decoder.source.valid),
                If(decoder.source.valid, group.payload.eq(decoder.source.payload),
                    group.first.eq(decoder.source.first), group.last.eq(decoder.source.last))),
            pending.eq(self.sp_rd_req),
            If(self.sp_rd_req, pending_sign.eq(read_sign), pending_tag.eq(sign_pair), pending_last.eq(issued == raw_count-1)),
            If(raw_pop, raw_valid.eq(0)),
            If(pending,
                If(pending_sign, sign_valid.eq(1), sign_tag.eq(pending_tag), sign_data.eq(self.sp_rd_dat)).Else(
                    raw_valid.eq(1), raw_data.eq(self.sp_rd_dat), raw_last.eq(pending_last)),
            ),
            If(decoder.done & ~decoder.error, input_done.eq(1)),
            If(written & kernel.source.last, last_written.eq(1)),
        )
        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0), self.exponent.eq(0),
            self.read_beats.eq(0), self.output_beats.eq(0), issued.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, self.read_beats.eq(self.read_beats+1),
                If(read_raw, issued.eq(issued+1))),
            If(written, self.output_beats.eq(self.output_beats+1), self.exponent.eq(kernel.source.exponent)),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    self.error.eq(0), self.exponent.eq(0), issued.eq(0), self.read_beats.eq(0),
                    self.output_beats.eq(0), state.eq(VALIDATE)),
                VALIDATE: If(geometry_bad | bounds_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, protected_bad)), state.eq(FAILED),
                ).Else(state.eq(START)),
                START: If(kernel.cmd.ready, state.eq(RUN)),
                RUN: If(kernel.done, state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
            # Failure wins over same-cycle completion; no invalid publication.
            If(failed, self.error.eq(Cat(C(0, 3), decoder.error, accounting_bad, metadata_bad,
                C(0, 2), kernel.error)), state.eq(FAILED)),
        )

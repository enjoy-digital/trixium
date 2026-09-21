# SPDX-License-Identifier: BSD-2-Clause
"""Bounded native weights -> semantic groups, independent of AXI/DDR/HBM.

PTQ1_0 retains its exact trit order and FP16 bits; Q8_0 retains signed-byte bits.
No arithmetic quantization, row packet expansion or scale RAM is involved. The
optional BonsaiChannelPipe semantic ingress accepts these groups; board command
frontends still use the default legacy interface and do not advertise native I/O.
"""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.capabilities import WeightEncoding

PTQ1_0 = int(WeightEncoding.PTQ1_0)
Q8_0 = int(WeightEncoding.Q8_0)

BLOCK_LAYOUT = [('data', 272), ('encoding', 8), ('row', 16), ('group', 16), ('row_last', 1)]
GROUP_LAYOUT = [('codes', 256), ('scale', 16), ('kind', 1), ('error', 1),
                ('row', 16), ('group', 16), ('row_last', 1)]
# kind=0: 128 two-bit ternary codes (code 1 is zero); kind=1: 32 int8 bit patterns.
# scale is the exact little-endian FP16 bit pattern, NOT a converted value.
# first/last endpoint flags delimit the complete command, not each row.


class NativeBlockDecoder(LiteXModule):
    """Two elastic register stages; up to one semantic group per cycle.

    The low 28/34 bytes of data hold one PTQ1_0/Q8_0 block. Unused high bits are
    ignored. Unsupported encodings produce an error-marked, zero-data group;
    consumers MUST reject it. clear discards both stages (cancellation exception
    to the usual stable-valid-until-ready rule). No downstream value is a result
    until the containing command has passed its framing/error checks.
    """
    def __init__(self):
        self.sink = stream.Endpoint(BLOCK_LAYOUT)
        self.source = stream.Endpoint(GROUP_LAYOUT)
        self.clear = Signal()

        raw = stream.Endpoint(BLOCK_LAYOUT)
        result = stream.Endpoint(GROUP_LAYOUT)
        take_raw, take_result = Signal(), Signal()
        self.comb += [
            take_result.eq(~result.valid | self.source.ready),
            take_raw.eq(~raw.valid | take_result),
            self.sink.ready.eq(take_raw),
            result.connect(self.source, omit={'valid', 'ready'}),
            self.source.valid.eq(result.valid & ~self.clear),
        ]
        self.sync += If(self.clear,
            raw.valid.eq(0), result.valid.eq(0),
        ).Else(
            If(take_raw,
                raw.valid.eq(self.sink.valid),
                If(self.sink.valid,
                    raw.payload.eq(self.sink.payload), raw.first.eq(self.sink.first),
                    raw.last.eq(self.sink.last),
                ),
            ),
            If(take_result,
                result.valid.eq(raw.valid),
                If(raw.valid,
                    result.row.eq(raw.row), result.group.eq(raw.group),
                    result.row_last.eq(raw.row_last), result.first.eq(raw.first),
                    result.last.eq(raw.last),
                    result.kind.eq(raw.encoding == Q8_0),
                    result.error.eq((raw.encoding != PTQ1_0) & (raw.encoding != Q8_0)),
                ),
            ),
        )

        ternary_codes = Signal(256)
        # PTQ1_0: qs[0:16] encodes five interleaved runs of 16 elements,
        # qs[16:24] five runs of 8, then qh[0:2] four runs of 2.
        # Do NOT use q//3**n: the stored byte is ceil(base3*256/243).
        for byte_index in range(26):
            byte = raw.data[8*byte_index:8*(byte_index + 1)]
            for trit in range(5 if byte_index < 24 else 4):
                if byte_index < 16:
                    element = 16*trit + byte_index
                elif byte_index < 24:
                    element = 80 + 8*trit + byte_index - 16
                else:
                    element = 120 + 2*trit + byte_index - 24
                wrapped = Signal(8)
                tripled = Signal(10)
                self.comb += [wrapped.eq(byte * (3**trit)),
                              tripled.eq((wrapped << 1) + wrapped),
                              ternary_codes[2*element:2*(element + 1)].eq(tripled[8:10])]
        self.sync += If(take_result & raw.valid & ~self.clear,
            If(raw.encoding == PTQ1_0,
                result.codes.eq(ternary_codes), result.scale.eq(raw.data[208:224]),
            ).Elif(raw.encoding == Q8_0,
                result.codes.eq(raw.data[16:272]), result.scale.eq(raw.data[:16]),
            ).Else(result.codes.eq(0), result.scale.eq(0)),
        )


class NativeWeightStream(LiteXModule):
    """Frame 256-bit memory beats into native blocks using a 64-byte reservoir.

    start latches rows/groups/encoding; byte zero of the first beat is the first
    block. The last beat may contain up to 31 ignored high padding bytes. sink.last
    must mark exactly ceil(rows*groups*block_bytes/32) beats. Command-level I/O
    supplies these aligned data and drains its own outstanding AXI traffic.

    On bad framing the decoder flushes; incoming beats drain through the next
    last, then done pulses with sticky error. A missing last therefore needs an
    upstream timeout/cancel. cancel locally flushes and sets cancelled; it does
    NOT cancel/drain physical memory transactions. Reuse requires the upstream
    transport to have drained/reset. Previously emitted groups from a failed or
    cancelled command must not be committed as results.
    """
    def __init__(self):
        self.sink = stream.Endpoint([('data', 256)])
        self.source = stream.Endpoint(GROUP_LAYOUT)
        self.start = Signal()
        self.cancel = Signal()
        self.rows = Signal(16)
        self.groups = Signal(16)
        self.encoding = Signal(8)
        self.busy = Signal()
        self.done = Signal()
        self.error = Signal()
        self.cancelled = Signal()

        self.decoder = decoder = NativeBlockDecoder()
        buffer = Signal(512)
        # Both native blocks have an even byte count. Keep halfword occupancy so
        # the append mux needs only 16-bit shifts; consume is a fixed 28/34-byte
        # selection rather than a general barrel shifter.
        count = Signal(6)
        self.buffered_bytes = Signal(7)  # Diagnostic, not a cumulative counter.
        fmt = Signal(8)
        block_units = Signal(5)
        groups_per_row = Signal(16)
        row, group = Signal(16), Signal(16)
        groups_left = Signal(32)
        units_left = Signal(37)
        beats_left = Signal(33)
        first = Signal()
        draining = Signal()
        pop, push, bad_frame = Signal(), Signal(), Signal()
        remaining = Signal(6)
        incoming = Signal(5)
        shifted = Signal(512)
        append_shift = Signal(9)
        total_groups = Signal(32)
        total_units = Signal(37)
        config_ok = Signal()
        self.comb += [
            total_groups.eq(self.rows * self.groups),
            total_units.eq(Mux(self.encoding == PTQ1_0, total_groups * 14, total_groups * 17)),
            self.buffered_bytes.eq(count << 1),
            config_ok.eq((self.rows != 0) & (self.groups != 0)
                         & ((self.encoding == PTQ1_0) | (self.encoding == Q8_0))),
            decoder.sink.valid.eq(self.busy & ~draining & (groups_left != 0)
                                  & (count >= block_units) & ~self.cancel),
            decoder.sink.data.eq(buffer[:272]), decoder.sink.encoding.eq(fmt),
            decoder.sink.row.eq(row), decoder.sink.group.eq(group),
            decoder.sink.row_last.eq(group == groups_per_row - 1),
            decoder.sink.first.eq(first), decoder.sink.last.eq(groups_left == 1),
            pop.eq(decoder.sink.valid & decoder.sink.ready),
            remaining.eq(count - Mux(pop, block_units, 0)),
            shifted.eq(Mux(pop, Mux(fmt == PTQ1_0, buffer >> 224, buffer >> 272), buffer)),
            incoming.eq(Mux(units_left < 16, units_left, 16)),
            # A nested (remaining[:5] << 4) used as a Verilog shift amount is
            # self-sized to FIVE bits, unlike Migen simulation. Explicit concat
            # preserves all nine address bits (caught by compiled RTL tests).
            append_shift.eq(Cat(C(0, 4), remaining[:5])),
            self.sink.ready.eq(self.busy & ~self.cancel
                               & (draining | ((beats_left != 0) & (remaining <= 16)))),
            push.eq(self.sink.valid & self.sink.ready),
            bad_frame.eq(push & ~draining & (self.sink.last != (beats_left == 1))),
            decoder.clear.eq(self.cancel | bad_frame | (self.start & ~self.busy)),
            decoder.source.connect(self.source),
        ]
        self.sync += [
            self.done.eq(0),
            If(self.cancel,
                self.done.eq(self.busy), self.busy.eq(0), self.cancelled.eq(1),
                buffer.eq(0), count.eq(0), draining.eq(0),
            ).Elif(self.start & ~self.busy,
                self.error.eq(~config_ok), self.cancelled.eq(0),
                self.busy.eq(config_ok), self.done.eq(~config_ok),
                fmt.eq(self.encoding), block_units.eq(Mux(self.encoding == PTQ1_0, 14, 17)),
                groups_per_row.eq(self.groups), row.eq(0), group.eq(0), first.eq(1),
                groups_left.eq(total_groups), units_left.eq(total_units),
                beats_left.eq((total_units + 15) >> 4),
                buffer.eq(0), count.eq(0), draining.eq(0),
            ).Elif(self.busy,
                If(self.start, self.error.eq(1)),  # Cannot overwrite a live command.
                If(bad_frame | draining,
                    self.error.eq(1), draining.eq(1), buffer.eq(0), count.eq(0),
                    If(push & self.sink.last,
                        self.busy.eq(0), self.done.eq(1), draining.eq(0),
                    ),
                ).Else(
                    If(push | pop,
                        buffer.eq(shifted | Mux(push, self.sink.data << append_shift, 0)),
                        count.eq(remaining + Mux(push, incoming, 0)),
                    ),
                    If(push,
                        units_left.eq(units_left - incoming), beats_left.eq(beats_left - 1),
                    ),
                    If(pop,
                        first.eq(0), groups_left.eq(groups_left - 1),
                        If(group == groups_per_row - 1,
                            row.eq(row + 1), group.eq(0),
                        ).Else(group.eq(group + 1)),
                    ),
                    If(self.source.valid & self.source.ready & self.source.last,
                        self.done.eq(1), self.busy.eq(0), count.eq(0),
                    ),
                ),
            ),
        ]

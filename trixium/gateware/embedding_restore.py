# SPDX-License-Identifier: BSD-2-Clause
"""Portable exact inverse-rotated PTQ embedding row -> reference BFP16 -> FP32."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.hadamard import IntegerHadamard
from trixium.gateware.projection_restore import I32ToFp32Scaled


class EmbeddingRestore(LiteXModule):
    """One semantic ternary row; no board, AXI or scratchpad dependency.

    cmd.groups is a positive multiple of eight (1024 elements/block), bounded
    by max_groups. sink supplies groups of 128 two-bit codes, exact FP16 scale
    and 128 inverse-sign bits (1=negate); first/last delimit the complete row.
    All input is validated/cached before calculation. Scales align EXACTLY to
    their row-minimum exponent; span > input_width-12 is rejected, not capped.

    An integer FWHT applies signs AFTER each block. Whole-row absmax selects
    the BFP exponent, then RNE/clipping to [-32767,32767] precedes exact FP32
    restoration. Output has eight FP32 elements/word and the common BFP
    exponent. Zero rows have exponent zero. No inter-block/CPU rounding occurs.

    Errors: bit0 command, bit1 framing/codes, bit2 nonfinite FP16, bit3 scale
    span, bit4 FWHT, bit5 output/internal credit. Failure stays busy until
    cancel/reset. Cancellation masks both handshakes and invalidates all output
    prefixes; memories are overwritten, not scrubbed, before reuse. Only done
    publishes success, after the last granted output. Sources must be drained
    externally before reuse; this kernel does not cancel physical memory reads.
    """
    def __init__(self, max_groups=64, input_width=16):
        if type(max_groups) is not int or not 8 <= max_groups <= 512 or max_groups % 8:
            raise ValueError('max_groups must be a multiple of eight in [8,512]')
        if type(input_width) is not int or not 12 <= input_width <= 32:
            raise ValueError('input_width must be in [12,32]')
        self.max_groups, self.input_width = max_groups, input_width
        self.cmd = stream.Endpoint([('groups', 16)])
        self.sink = stream.Endpoint([('codes', 256), ('scale', 16), ('negate', 128)])
        self.source = stream.Endpoint([('data', 256), ('exponent', 10)])
        self.cancel, self.busy, self.done, self.cancelled = Signal(), Signal(), Signal(), Signal()
        self.error = Signal(6)
        self.common_exponent = Signal((7, True))
        self.output_exponent = Signal((10, True))

        self.core = core = IntegerHadamard(1024, 8, input_width)
        self.restore = restore = I32ToFp32Scaled(8)
        width = core.output_width
        IDLE, LOAD, CHECK, HSTART, GROUP_REQ, GROUP_CAP, SEND, HOUT, NORM, OUTPUT, FINISH, FAILED = range(12)
        state = Signal(4, reset=IDLE)
        groups = Signal(16)
        received = Signal(max=max_groups+1)
        group = Signal(max=max_groups)
        word_in_group = Signal(4)
        codes, scale, signs = Signal(256), Signal(16), Signal(128)
        emin, emax = self.common_exponent, Signal((7, True))
        maximum = Signal(width)
        bfp_shift = Signal((8, True))
        write_index = Signal(max=max_groups*16+1)

        cache = Memory(400, max_groups, name='embedding_groups')
        cache_write = cache.get_port(write_capable=True)
        cache_read = cache.get_port(has_re=True)
        results = Memory(width*8, max_groups*16, name='embedding_raw')
        raw_write = results.get_port(write_capable=True)
        raw_read = results.get_port(has_re=True)
        self.specials += cache, cache_write, cache_read, results, raw_write, raw_read

        input_e, input_normal_e = Signal((7, True)), Signal((7, True))
        finite, framing_bad, command_bad = Signal(), Signal(), Signal()
        accept = Signal()
        self.comb += [
            command_bad.eq((self.cmd.groups == 0) | (self.cmd.groups > max_groups) | (self.cmd.groups[:3] != 0)),
            # Widen BEFORE the mux: a nested unsigned five-bit E-25 becomes
            # self-sized inside Migen's Verilog signed cast and wraps by 32.
            input_normal_e.eq(self.sink.scale[10:15] - C(25, 7)),
            input_e.eq(Mux(self.sink.scale[10:15] == 0, C(-24, (7, True)), input_normal_e)),
            finite.eq(self.sink.scale[10:15] != 31),
            framing_bad.eq((self.sink.first != (received == 0)) |
                (self.sink.last != (received == groups-1)) |
                ((self.sink.codes & (self.sink.codes >> 1) & C(int('55'*32, 16), 256)) != 0)),
            self.cmd.ready.eq((state == IDLE) & ~self.cancel),
            self.sink.ready.eq((state == LOAD) & ~self.cancel),
            self.busy.eq(state != IDLE), accept.eq(self.sink.valid & self.sink.ready),
            cache_write.adr.eq(received), cache_write.dat_w.eq(Cat(self.sink.codes, self.sink.scale, self.sink.negate)),
            cache_write.we.eq(accept & finite & ~framing_bad),
            cache_read.adr.eq(group), cache_read.re.eq((state == GROUP_REQ) & ~self.cancel),
            core.cancel.eq(self.cancel | (state == FAILED) | (state == IDLE)),
            core.cmd.valid.eq((state == HSTART) & ~self.cancel), core.cmd.inverse.eq(1),
            core.sink.valid.eq((state == SEND) & ~self.cancel), core.sink.negate.eq(signs[:8]),
            core.sink.last.eq((group[:3] == 7) & (word_in_group == 15)),
            core.source.ready.eq((state == HOUT) & ~self.cancel),
            raw_write.adr.eq(write_index), raw_write.dat_w.eq(core.source.data),
            raw_write.we.eq(core.source.valid & core.source.ready),
        ]
        scale_e, scale_normal_e = Signal((7, True)), Signal((7, True))
        align_shift = Signal(5)
        mantissa = Signal(11)
        aligned = Signal(input_width)
        signed_scale = Signal((input_width, True))
        self.comb += [scale_normal_e.eq(scale[10:15] - C(25, 7)),
            scale_e.eq(Mux(scale[10:15] == 0, C(-24, (7, True)), scale_normal_e)),
            align_shift.eq(scale_e-emin), mantissa.eq(Cat(scale[:10], scale[10:15] != 0)),
            aligned.eq(mantissa << align_shift), signed_scale.eq(Mux(scale[15], -aligned, aligned))]
        for lane in range(8):
            code = codes[2*lane:2*lane+2]
            self.comb += core.sink.data[input_width*lane:input_width*(lane+1)].eq(
                Mux(code == 0, -signed_scale, Mux(code == 2, signed_scale, 0)))

        bit_length = Signal(6)
        span = Signal(7)
        self.comb += [span.eq(emax-emin), bit_length.eq(0)]
        for bit in range(width):
            self.comb += If(maximum[bit], bit_length.eq(bit+1))

        # Two slots reserve space for the one-cycle synchronous RAM response.
        self.raw_fifo = fifo = ResetInserter()(stream.SyncFIFO([('data', width*8)], 2))
        issued = Signal(max=max_groups*16+1)
        pending, pending_first, pending_last = Signal(), Signal(), Signal()
        issue, pop, credit_bad = Signal(), Signal(), Signal()
        total_words = Signal(20)
        self.comb += [
            total_words.eq(groups << 4),
            fifo.reset.eq(self.cancel | (state == IDLE) | (state == FAILED)),
            pop.eq(fifo.source.valid & fifo.source.ready),
            issue.eq((state == OUTPUT) & ~self.cancel & (issued < total_words) &
                     ((fifo.level + pending < 2) | pop)),
            raw_read.adr.eq(issued), raw_read.re.eq(issue),
            fifo.sink.valid.eq(pending & (state == OUTPUT) & ~self.cancel),
            fifo.sink.data.eq(raw_read.dat_r), fifo.sink.first.eq(pending_first), fifo.sink.last.eq(pending_last),
            credit_bad.eq(fifo.sink.valid & ~fifo.sink.ready),
            restore.cancel.eq(self.cancel | (state == IDLE) | (state == FAILED)),
            restore.sink.valid.eq(fifo.source.valid & (state == OUTPUT) & ~self.cancel),
            fifo.source.ready.eq(restore.sink.ready & (state == OUTPUT) & ~self.cancel),
            restore.sink.first.eq(fifo.source.first), restore.sink.last.eq(fifo.source.last),
            restore.sink.exponent.eq(self.output_exponent),
            self.source.valid.eq(restore.source.valid & (restore.source.error == 0) & (state == OUTPUT) & ~self.cancel),
            self.source.data.eq(restore.source.data), self.source.exponent.eq(restore.source.exponent),
            self.source.first.eq(restore.source.first), self.source.last.eq(restore.source.last),
            restore.source.ready.eq(self.source.ready & (state == OUTPUT) & (restore.source.error == 0) & ~self.cancel),
        ]
        right, left, half_index = Signal(6), Signal(5), Signal(6)
        half = Signal(width)
        self.comb += [right.eq(Mux(bfp_shift > 0, bfp_shift, 0)),
            left.eq(Mux(bfp_shift < 0, -bfp_shift, 0)), half_index.eq(Mux(right != 0, right-1, 0)),
            half.eq(C(1, width) << half_index)]
        for lane in range(8):
            raw = fifo.source.data[width*lane:width*(lane+1)]
            magnitude, quotient, remainder = Signal(width), Signal(width), Signal(width)
            rounded = Signal(width+1)
            clipped = Signal(16)
            signed = Signal((32, True))
            self.comb += [magnitude.eq(Mux(raw[width-1], -raw, raw)), quotient.eq(magnitude >> right),
                remainder.eq(magnitude & ((half << 1)-1)),
                rounded.eq(Mux(bfp_shift > 0, quotient + C(0, width+1) +
                    ((remainder > half) | ((remainder == half) & quotient[0])), magnitude << left)),
                clipped.eq(Mux(rounded > 32767, 32767, rounded)),
                signed.eq(Mux(raw[width-1], -clipped, clipped)),
                restore.sink.data[32*lane:32*(lane+1)].eq(signed)]

        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0), self.cancelled.eq(1), pending.eq(0),
        ).Else(
            self.done.eq(0),
            pending.eq(issue),
            If(issue, pending_first.eq(issued == 0), pending_last.eq(issued == total_words-1), issued.eq(issued+1)),
            If(core.error & (state != IDLE), self.error.eq(self.error | 16), state.eq(FAILED)).Elif(
                credit_bad | (restore.source.valid & (restore.source.error != 0)),
                self.error.eq(self.error | 32), state.eq(FAILED),
            ).Else(Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    groups.eq(self.cmd.groups), received.eq(0), emin.eq(31), emax.eq(-32),
                    group.eq(0), word_in_group.eq(0), maximum.eq(0), write_index.eq(0), issued.eq(0),
                    self.error.eq(command_bad), self.cancelled.eq(0), self.output_exponent.eq(0),
                    If(command_bad, state.eq(FAILED)).Else(state.eq(LOAD)),
                ),
                LOAD: If(accept,
                    If(framing_bad | ~finite,
                        self.error.eq(Cat(C(0, 1), framing_bad, ~finite)), state.eq(FAILED),
                    ).Else(
                        If(input_e < emin, emin.eq(input_e)), If(input_e > emax, emax.eq(input_e)),
                        received.eq(received+1), If(self.sink.last, state.eq(CHECK)),
                    ),
                ),
                CHECK: If(span > input_width-12, self.error.eq(8), state.eq(FAILED)).Else(state.eq(HSTART)),
                HSTART: If(core.cmd.ready, state.eq(GROUP_REQ)),
                GROUP_REQ: state.eq(GROUP_CAP),
                GROUP_CAP: [codes.eq(cache_read.dat_r[:256]), scale.eq(cache_read.dat_r[256:272]),
                            signs.eq(cache_read.dat_r[272:400]), state.eq(SEND)],
                SEND: If(core.sink.valid & core.sink.ready,
                    codes.eq(codes >> 16), signs.eq(signs >> 8),
                    If(word_in_group == 15,
                        word_in_group.eq(0),
                        If(group[:3] == 7, state.eq(HOUT)).Else(group.eq(group+1), state.eq(GROUP_REQ)),
                    ).Else(word_in_group.eq(word_in_group+1)),
                ),
                HOUT: [If(core.source.valid & core.source.ready, write_index.eq(write_index+1)),
                    If(core.done,
                        If(core.absmax > maximum, maximum.eq(core.absmax)),
                        If(group == groups-1, state.eq(NORM)).Else(group.eq(group+1), state.eq(HSTART)),
                    )],
                NORM: If(write_index != total_words, self.error.eq(32), state.eq(FAILED)).Else(
                    bfp_shift.eq(bit_length-15),
                    self.output_exponent.eq(Mux(maximum == 0, 0, emin-5+bit_length-15)),
                    state.eq(OUTPUT)),
                OUTPUT: If(self.source.valid & self.source.ready & self.source.last, state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            })),
        )

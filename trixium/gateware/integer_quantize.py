# SPDX-License-Identifier: BSD-2-Clause
"""Exact signed-integer power-of-two quantization, without FP conversion."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream


class IntegerToQ8(LiteXModule):
    """Three elastic stages: q=clamp(round_ties_away(x/2**e8), -127, 127).

    e8 is a signed two's-complement ten-bit exponent on every accepted word.
    Every exponent is defined: large positive shifts give zero; large negative
    shifts saturate nonzero integers. No floating-point cast/rounding is applied.
    This matches an FP32 boundary only when the input integer representation is
    exact there, as for bounded raw 1024-element Hadamard results. It is not an
    implicit replacement for either general int32->FP32 or BFP ties-even modes.

    Metadata and data hold together under backpressure. cancel masks all
    handshakes and flushes validity, not any already consumed output prefix.
    """
    def __init__(self, lanes=8, input_width=19):
        if not isinstance(lanes, int) or lanes < 1 or lanes > 16 or lanes & (lanes - 1):
            raise ValueError('lanes must be a power of two in [1, 16]')
        if not isinstance(input_width, int) or not 2 <= input_width <= 32:
            raise ValueError('input_width must be in [2, 32]')
        self.lanes, self.input_width = lanes, input_width
        self.sink = stream.Endpoint([('data', lanes*input_width), ('e8', 10)])
        self.source = stream.Endpoint([('data', lanes*8), ('e8', 10)])
        self.cancel = Signal()
        valid, ready = [Signal() for _ in range(3)], [Signal() for _ in range(3)]
        self.comb += [ready[2].eq(~valid[2] | self.source.ready),
                      ready[1].eq(~valid[1] | ready[2]), ready[0].eq(~valid[0] | ready[1]),
                      self.sink.ready.eq(ready[0] & ~self.cancel),
                      self.source.valid.eq(valid[2] & ~self.cancel)]
        exponents = [Signal((10, True)) for _ in range(2)]
        first, last = [Signal() for _ in range(2)], [Signal() for _ in range(2)]
        signs = [Signal(lanes) for _ in range(2)]
        magnitudes = [Signal(input_width + 1) for _ in range(lanes)]
        rounded_values = [Signal(9) for _ in range(lanes)]
        right_amount = Signal(max=max(2, input_width))
        left_amount = Signal(3)
        self.comb += [right_amount.eq(exponents[0] - 1), left_amount.eq(-exponents[0])]
        capture = [exponents[0].eq(self.sink.e8), first[0].eq(self.sink.first), last[0].eq(self.sink.last)]
        shift = [exponents[1].eq(exponents[0]), first[1].eq(first[0]), last[1].eq(last[0]), signs[1].eq(signs[0])]
        finish = [self.source.e8.eq(exponents[1]), self.source.first.eq(first[1]), self.source.last.eq(last[1])]
        for lane in range(lanes):
            value = Signal((input_width, True))
            self.comb += value.eq(self.sink.data[lane*input_width:(lane+1)*input_width])
            magnitude = magnitudes[lane]
            capture += [magnitude.eq(Mux(value < 0, -value, value)), signs[0][lane].eq(value < 0)]
            right_result = Signal(input_width + 2)
            left_result = Signal(14)
            rounded = rounded_values[lane]
            self.comb += [
                right_result.eq(((magnitude >> right_amount) + C(1, input_width + 2)) >> 1),
                left_result.eq(magnitude[:min(7, input_width + 1)] << left_amount),
            ]
            shift.append(rounded.eq(Mux(magnitude == 0, 0,
                Mux(exponents[0] > input_width, 0,
                    Mux(exponents[0] > 0, Mux(right_result > 127, 128, right_result),
                        Mux((exponents[0] <= -7) | (magnitude > 127), 128,
                            Mux(left_result > 127, 128, left_result)))))))
            clamped = Signal(8)
            self.comb += clamped.eq(Mux(rounded > 127, 127, rounded))
            finish.append(self.source.data[lane*8:(lane+1)*8].eq(Mux(signs[1][lane], -clamped, clamped)))
        self.sync += If(self.cancel, *[v.eq(0) for v in valid]).Else(
            If(ready[0], valid[0].eq(self.sink.valid), If(self.sink.valid, *capture)),
            If(ready[1], valid[1].eq(valid[0]), If(valid[0], *shift)),
            If(ready[2], valid[2].eq(valid[1]), If(valid[1], *finish)),
        )

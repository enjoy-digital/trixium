# SPDX-License-Identifier: BSD-2-Clause
"""Portable activation conversion; no floating-point IP or board memory mapping."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream


class Fp32AbsmaxWord(LiteXModule):
    """Elastic finite absmax reduction of one word, not a whole vector.

    Magnitudes are positive binary32 bit patterns (31 bits). Nonfinite lanes
    contribute zero and set error. first/last are passed through unchanged.
    The operation owner accumulates these word maxima/errors and chooses e8
    only after the final result. Default latency is three clocks, interval one.
    """
    def __init__(self, lanes=8):
        if not isinstance(lanes, int) or lanes < 1 or lanes > 16 or lanes & (lanes - 1):
            raise ValueError("lanes must be a power of two in [1, 16]")
        self.sink = stream.Endpoint([("data", lanes*32)])
        self.source = stream.Endpoint([("absmax", 31), ("error", 1)])
        self.cancel = Signal()
        levels = max(1, lanes.bit_length() - 1)
        valid, ready = [Signal() for _ in range(levels)], [Signal() for _ in range(levels)]
        first, last = [Signal() for _ in range(levels)], [Signal() for _ in range(levels)]
        values, errors = [], []
        for lane in range(lanes):
            word = self.sink.data[32*lane:32*(lane+1)]
            bad = Signal()
            value = Signal(31)
            self.comb += [bad.eq(word[23:31] == 255), value.eq(Mux(bad, 0, word[:31]))]
            values.append(value)
            errors.append(bad)
        for level in range(levels):
            nxt_values = [Signal(31) for _ in range(max(1, len(values)//2))]
            nxt_errors = [Signal() for _ in nxt_values]
            updates = []
            for index, (value, error) in enumerate(zip(nxt_values, nxt_errors)):
                a, b = 2*index, min(2*index + 1, len(values) - 1)
                updates += [value.eq(Mux(values[a] > values[b], values[a], values[b])),
                            error.eq(errors[a] | errors[b])]
            previous_valid = self.sink.valid if level == 0 else valid[level - 1]
            updates += [first[level].eq(self.sink.first if level == 0 else first[level - 1]),
                        last[level].eq(self.sink.last if level == 0 else last[level - 1])]
            self.comb += ready[level].eq(~valid[level] |
                (self.source.ready if level == levels - 1 else ready[level + 1]))
            self.sync += If(self.cancel, valid[level].eq(0)).Elif(ready[level],
                valid[level].eq(previous_valid), If(previous_valid, *updates))
            values, errors = nxt_values, nxt_errors
        self.comb += [self.sink.ready.eq(ready[0] & ~self.cancel),
                      self.source.valid.eq(valid[-1] & ~self.cancel),
                      self.source.absmax.eq(values[0]), self.source.error.eq(errors[0]),
                      self.source.first.eq(first[-1]), self.source.last.eq(last[-1])]


class Fp32ToQ8(LiteXModule):
    """Three-stage elastic FP32 -> symmetric int8 converter, eight lanes by default.

    Each accepted word carries a signed two's-complement 10-bit e8. The result
    is clamp(round_nearest_ties_away(x * 2**-e8), -127, 127). For finite vectors
    with absmax-selected -127 <= e8 <= 121 this matches firmware/infer.c's
    power-of-two multiply and RMM/lroundf conversion, including subnormal lanes
    and signed zeros. Arbitrary caller-selected exponents have defined saturation
    for large finite values, without depending on out-of-range C lroundf behavior.
    This range keeps the firmware's separate 2**-e8 scale finite and nonzero;
    exponent selection for a tiny all-small vector must reject an unsafe scale.

    source.error bits [lanes-1:0] mark nonfinite input lanes; bit lanes marks an
    unsupported exponent. Invalid lanes (all lanes for a bad exponent) output
    zero, never a plausible successful conversion. The caller accumulates these
    errors across the operation and discards the entire result on any failure.
    e8/first/last are passed through, not interpreted as a command or frame.

    Backpressure holds all output fields stable. cancel immediately masks both
    handshakes and flushes the pipeline; it does not undo an already consumed
    prefix. There is no scratchpad access, absmax scan, exponent selection or
    publication in this primitive. No DSPs or vendor FP operators are used.
    """
    def __init__(self, lanes=8):
        if not isinstance(lanes, int) or lanes < 1 or lanes > 16 or lanes & (lanes - 1):
            raise ValueError("lanes must be a power of two in [1, 16]")
        self.lanes = lanes
        self.sink = stream.Endpoint([("data", lanes*32), ("e8", 10)])
        self.source = stream.Endpoint([("data", lanes*8), ("e8", 10), ("error", lanes + 1)])
        self.cancel = Signal()

        valid = [Signal() for _ in range(3)]
        ready = [Signal() for _ in range(3)]
        self.comb += [
            ready[2].eq(~valid[2] | self.source.ready),
            ready[1].eq(~valid[1] | ready[2]),
            ready[0].eq(~valid[0] | ready[1]),
            self.sink.ready.eq(ready[0] & ~self.cancel),
            self.source.valid.eq(valid[2] & ~self.cancel),
        ]
        metadata = [[Signal(10), Signal(), Signal(), Signal(lanes + 1)] for _ in range(2)]
        signs = [Signal(lanes) for _ in range(2)]
        significands = [Signal(24) for _ in range(lanes)]
        shifts = [Signal((12, True)) for _ in range(lanes)]
        magnitudes = [Signal(9) for _ in range(lanes)]
        exponent = Signal((10, True))
        nonfinite = Signal(lanes)
        bad_scale = Signal()
        self.comb += [exponent.eq(self.sink.e8),
                      bad_scale.eq((exponent < -127) | (exponent > 121))]

        capture = [metadata[0][0].eq(self.sink.e8), metadata[0][1].eq(self.sink.first),
                   metadata[0][2].eq(self.sink.last), metadata[0][3].eq(Cat(nonfinite, bad_scale))]
        shift = [dst.eq(src) for dst, src in zip(metadata[1], metadata[0])]
        shift.append(signs[1].eq(signs[0]))
        finish = [self.source.e8.eq(metadata[1][0]), self.source.first.eq(metadata[1][1]),
                  self.source.last.eq(metadata[1][2]), self.source.error.eq(metadata[1][3])]
        for lane in range(lanes):
            word = self.sink.data[32*lane:32*(lane+1)]
            biased = Signal((10, True))
            self.comb += [biased.eq(word[23:31]), nonfinite[lane].eq(word[23:31] == 255)]
            capture += [
                significands[lane].eq(Cat(word[:23], word[23:31] != 0)),
                signs[0][lane].eq(word[31]),
                shifts[lane].eq(Mux(biased == 0, 149, 150 - biased) + exponent),
            ]
            # In the allowed exponent domain, a subnormal's right shift is >=22.
            # A normal significand has its leading 1 at bit 23: shifts below 17
            # must saturate. Only shifts 17..24 require a barrel shifter, and an
            # extra round bit is enough for ties-away (no sticky/ties-even rule).
            amount, rounded = Signal(5), Signal(25)
            self.comb += [amount.eq(shifts[lane] - 1),
                          rounded.eq(((significands[lane] >> amount) + C(1, 25)) >> 1)]
            shift.append(magnitudes[lane].eq(Mux(significands[lane] == 0, 0,
                Mux(shifts[lane] < 17, 128, Mux(shifts[lane] > 24, 0, rounded)))))
            clamped = Signal(8)
            self.comb += clamped.eq(Mux(magnitudes[lane] > 127, 127, magnitudes[lane]))
            finish.append(self.source.data[8*lane:8*(lane+1)].eq(
                Mux(metadata[1][3][lane] | metadata[1][3][lanes], 0,
                    Mux(signs[1][lane], -clamped, clamped))))

        self.sync += If(self.cancel, *[v.eq(0) for v in valid]).Else(
            If(ready[0], valid[0].eq(self.sink.valid), If(self.sink.valid, *capture)),
            If(ready[1], valid[1].eq(valid[0]), If(valid[0], *shift)),
            If(ready[2], valid[2].eq(valid[1]), If(valid[1], *finish)),
        )

# SPDX-License-Identifier: BSD-2-Clause
"""Portable, bit-exact restoration of integer projection results to binary32."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream


class I32ToFp32Scaled(LiteXModule):
    """Four-stage elastic ``float32(int32(x)) * float32(2**exponent)``.

    The signed ten-bit exponent must be in [-149, 127], so the separately
    constructed binary32 scale is finite and nonzero. Conversion uses nearest,
    ties to even (RNE), not the activation quantizer's ties-away rule. Integer
    segment reduction, if any, must happen BEFORE this conversion, preserving
    the reference vector unit's wrapping int32 addition.

    No second rounding is necessary in this domain: if scaling produces a
    subnormal, the unscaled integer is <2**23 and exactly representable, and the
    scale is an integer multiple of 2**-149. Normal results preserve all the
    converted significand bits. Carry from rounding INT_MAX to 2**31 is handled
    before checking the scaled exponent. Zero always produces positive zero.

    source.error bits [lanes-1:0] mark overflow to infinity, and bit lanes marks
    an unsupported scale exponent. Failed lanes (all lanes for a bad scale)
    output zero; an operation owner MUST invalidate its entire partial result
    on any error. This is a fail-closed finite-result contract, not silent
    clipping, IEEE exception flags, or a general floating-point scale operator.

    first/last/exponent pass through. Sustained interval is one word per clock;
    backpressure holds all output fields stable. cancel masks both handshakes
    immediately and flushes the pipeline, but cannot undo consumed output.
    There is no scratchpad access, exponent selection or result publication.
    No DSP, vendor floating-point operator or board-specific resource is used.
    """
    def __init__(self, lanes=8):
        if not isinstance(lanes, int) or lanes < 1 or lanes > 16 or lanes & (lanes - 1):
            raise ValueError("lanes must be a power of two in [1, 16]")
        self.lanes = lanes
        self.sink = stream.Endpoint([("data", lanes*32), ("exponent", 10)])
        self.source = stream.Endpoint([("data", lanes*32), ("exponent", 10), ("error", lanes + 1)])
        self.cancel = Signal()

        valid = [Signal() for _ in range(4)]
        ready = [Signal() for _ in range(4)]
        for stage in range(4):
            self.comb += ready[stage].eq(~valid[stage] |
                (self.source.ready if stage == 3 else ready[stage + 1]))
        self.comb += [self.sink.ready.eq(ready[0] & ~self.cancel),
                      self.source.valid.eq(valid[3] & ~self.cancel)]

        metadata = [[Signal((10, True)), Signal(), Signal(), Signal()] for _ in range(3)]
        signs = [Signal(lanes) for _ in range(3)]
        zeros = [Signal(lanes) for _ in range(3)]
        exponent = Signal((10, True))
        self.comb += exponent.eq(self.sink.exponent)
        updates = [[], [], [], []]
        updates[0] += [metadata[0][0].eq(exponent), metadata[0][1].eq(self.sink.first),
                       metadata[0][2].eq(self.sink.last),
                       metadata[0][3].eq((exponent < -149) | (exponent > 127))]
        for stage in (1, 2):
            updates[stage] += [dst.eq(src) for dst, src in zip(metadata[stage], metadata[stage - 1])]
            updates[stage] += [signs[stage].eq(signs[stage - 1]), zeros[stage].eq(zeros[stage - 1])]
        updates[3] += [self.source.exponent.eq(metadata[2][0]),
                       self.source.first.eq(metadata[2][1]), self.source.last.eq(metadata[2][2]),
                       self.source.error[lanes].eq(metadata[2][3])]

        for lane in range(lanes):
            word = self.sink.data[32*lane:32*(lane+1)]
            magnitude = Signal(32)
            updates[0] += [magnitude.eq(Mux(word[31], -word, word)),
                           signs[0][lane].eq(word[31]), zeros[0][lane].eq(word == 0)]

            # Highest-set-bit selection, then a left normalize. All discarded
            # conversion bits fit below bit 8 of this 32-bit normalized value.
            index = Signal(5)
            self.comb += index.eq(0)
            for bit in range(1, 32):
                self.comb += If(magnitude[bit], index.eq(bit))
            amount = Signal(5)
            normalized = Signal(32)
            significand = Signal(24)
            increment = Signal()
            power = Signal((12, True))
            self.comb += [amount.eq(31 - index), normalized.eq(magnitude << amount)]
            updates[1] += [significand.eq(normalized[8:]),
                           increment.eq(normalized[7] & ((normalized[:7] != 0) | normalized[8])),
                           power.eq(index + metadata[0][0] + 127)]

            rounded = Signal(25)
            mantissa = Signal(24)
            biased = Signal((12, True))
            self.comb += rounded.eq(significand + C(0, 25) + increment)
            updates[2] += [mantissa.eq(Mux(rounded[24], rounded[1:25], rounded[:24])),
                           biased.eq(power + rounded[24])]

            overflow = Signal()
            fraction = Signal(23)
            encoded_exponent = Signal(8)
            subnormal_shift = Signal(5)
            self.comb += [
                overflow.eq(~metadata[2][3] & ~zeros[2][lane] & (biased >= 255)),
                subnormal_shift.eq(1 - biased),
                fraction.eq(Mux(biased > 0, mantissa[:23], mantissa >> subnormal_shift)),
                encoded_exponent.eq(Mux(biased > 0, biased, 0)),
            ]
            updates[3] += [self.source.error[lane].eq(overflow),
                self.source.data[32*lane:32*(lane+1)].eq(
                    Mux(metadata[2][3] | overflow | zeros[2][lane], 0,
                        Cat(fraction, encoded_exponent, signs[2][lane])))]

        self.sync += If(self.cancel, *[v.eq(0) for v in valid]).Else(*[
            If(ready[stage],
                valid[stage].eq(self.sink.valid if stage == 0 else valid[stage - 1]),
                If(self.sink.valid if stage == 0 else valid[stage - 1], *updates[stage]))
            for stage in range(4)
        ])

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Activation quantization: FP32 or raw integer vectors to Q8 with power-of-two scales.

IntegerToQ8, Fp32AbsmaxWord and Fp32ToQ8 are portable elastic streaming primitives: exact
signed-integer or FP32 quantization without floating-point IP or board memory mapping.

ActivationScratchpad is the resident unit built on them over the shared scratchpad: per
block it scans the FP32 absmax, selects e8, then rereads and packs the block into Q8; in
requantize mode it instead rescales raw 1024- (or 512-) element int32 Hadamard blocks to a
single whole-vector exponent. Its command (ACTIVATION_COMMAND) is the firmware's descriptor layout.
"""

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import LiteXModule

from litex.soc.interconnect import stream

from gateware.core.datapath import reset_less_registers

# Integer To Q8 ------------------------------------------------------------------------------------

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
            raise ValueError("lanes must be a power of two in [1, 16]")
        if not isinstance(input_width, int) or not 2 <= input_width <= 32:
            raise ValueError("input_width must be in [2, 32]")
        self.lanes, self.input_width = lanes, input_width
        self.sink   = stream.Endpoint([("data", lanes*input_width), ("e8", 10)])
        self.source = stream.Endpoint([("data", lanes*8), ("e8", 10)])
        self.cancel = Signal()

        # # #

        # Elastic stage handshake: capture, shift/round, finish.
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

        # Per-lane datapath, split over the three stages.
        capture = [exponents[0].eq(self.sink.e8), first[0].eq(self.sink.first),
                   last[0].eq(self.sink.last)]
        shift = [exponents[1].eq(exponents[0]), first[1].eq(first[0]), last[1].eq(last[0]),
                 signs[1].eq(signs[0])]
        finish = [self.source.e8.eq(exponents[1]), self.source.first.eq(first[1]),
                  self.source.last.eq(last[1])]
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
            finish.append(self.source.data[lane*8:(lane+1)*8].eq(
                Mux(signs[1][lane], -clamped, clamped)))
        self.sync += If(self.cancel, *[v.eq(0) for v in valid]).Else(
            If(ready[0], valid[0].eq(self.sink.valid), If(self.sink.valid, *capture)),
            If(ready[1], valid[1].eq(valid[0]), If(valid[0], *shift)),
            If(ready[2], valid[2].eq(valid[1]), If(valid[1], *finish)),
        )
        # An elastic pipeline: every register but the stage valids is datapath (no reset).
        reset_less_registers(self, keep=valid)

# FP32 Absmax Word ---------------------------------------------------------------------------------

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
        self.sink   = stream.Endpoint([("data", lanes*32)])
        self.source = stream.Endpoint([("absmax", 31), ("error", 1)])
        self.cancel = Signal()

        # # #

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

        # Pairwise maximum tree, one elastic stage per level.
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
        # An elastic pipeline: every register but the stage valids is datapath (no reset).
        reset_less_registers(self, keep=valid)
        self.comb += [self.sink.ready.eq(ready[0] & ~self.cancel),
                      self.source.valid.eq(valid[-1] & ~self.cancel),
                      self.source.absmax.eq(values[0]), self.source.error.eq(errors[0]),
                      self.source.first.eq(first[-1]), self.source.last.eq(last[-1])]

# FP32 To Q8 ---------------------------------------------------------------------------------------

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
        self.lanes  = lanes
        self.sink   = stream.Endpoint([("data", lanes*32), ("e8", 10)])
        self.source = stream.Endpoint([("data", lanes*8), ("e8", 10), ("error", lanes + 1)])
        self.cancel = Signal()

        # # #

        # Elastic stage handshake: capture, shift/round, finish.
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

        # Per-lane datapath, split over the three stages.
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
        # An elastic pipeline: every register but the stage valids is datapath (no reset).
        reset_less_registers(self, keep=valid)

# Activation Scratchpad ----------------------------------------------------------------------------

ACTIVATION_COMMAND = [("src", 24), ("dst", 24), ("exponents", 24),
                      ("elements", 16), ("blocks", 16),
                      ("requantize", 1), ("maxima", 24), ("result_exponent", 24),
                      ("protect_start", 24), ("protect_end", 24)]

class ActivationScratchpad(LiteXModule):
    """Scan, select e8, reread and pack each independent FP32 block into Q8.

    src/dst/exponents are 32-byte-aligned bases. elements is a positive multiple
    of 32; blocks is nonzero. Inputs/outputs are contiguous across blocks. Each
    block gets one signed int32 exponent, preserving unused lanes in the last
    exponent beat. blocks=1 selects one exponent for a complete vector.

    The complete input/output/rounded-exponent spans must fit physical memory
    and be mutually disjoint. An optional half-open protected interval (empty
    by default) additionally protects the sequencer's descriptor list from writes.
    Bounds use wide arithmetic before any scratchpad access. The integrator
    grants every read, returns it one cycle later and gates writes with gnt;
    no other owner may change these buffers while the command is active.

    requantize selects raw 1024-element int32 Hadamard blocks instead of FP32 (or 512-element
    ones: the rotation's normalization is then 2**-4, not 2**-5 - the image folds the remaining
    2**-0.5 into the rotated weights' scales).
    exponents/maxima then hold input signed int32 e8/uint32 raw absmax per block;
    result_exponent receives a single whole-vector signed int32 scale. All
    metadata is validated and cached before writes. Read-only input spans may
    overlap in this mode, but neither output may overlap an input or the other
    output. Reserve a full 32-byte beat for result_exponent (tail preserved).

    error bits: 0 geometry/alignment, 1 capacity, 2 overlap, 3 nonfinite input,
    4 unsafe exponent, 5 internal conversion/credit, 6 protected-list overlap,
    7 invalid rotation metadata or raw input inconsistent with its absmax.
    Failure remains busy until cancel/reset; all partial output is invalid.
    done publishes success only after the last exponent write and drain.
    """
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client "
                             "address space")
        # Command / status.
        self.cmd = stream.Endpoint(ACTIVATION_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error        = Signal(8)
        self.blocks_done  = Signal(16)
        self.read_beats   = Signal(32)
        self.output_beats = Signal(32)
        self.absmax       = Signal(31)
        self.exponent     = Signal((10, True))

        # Scratchpad client ports.
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)

        # # #

        # Submodules: absmax scan, FP32/integer quantizers, read-response FIFO.
        self.scan             = scan = Fp32AbsmaxWord()
        self.quantize         = quantize = Fp32ToQ8()
        self.integer_quantize = integer_quantize = IntegerToQ8()
        self.responses        = responses = ResetInserter()(SyncFIFO(width=258, depth=2))
        # Block buffer for the ping-pong mode (pp, per-block FP32 with <= 1,024 elements and
        # several blocks): the scan pass copies the block into one half, so the convert pass
        # reads it from there while the next block's scan reads the scratchpad into the other
        # half - one scratchpad pass per block instead of two (~135 instead of ~262 clocks).
        blkbuf = Memory(256, 256, name="activation_blkbuf")
        buf_wr = blkbuf.get_port(write_capable=True)
        buf_rd = blkbuf.get_port()
        self.specials += blkbuf, buf_wr, buf_rd

        # Command capture and validation.
        (IDLE, VALIDATE, SCAN, SCAN_DRAIN, CHOOSE, CONVERT, CONVERT_DRAIN, EXPONENT,
         NEXT, FINISH, FAILED, META_E_REQ, META_E_CAPTURE, META_A_REQ, META_A_CAPTURE,
         META_CHECK, GLOBAL_CHOOSE, CACHE_READ, CACHE_CAPTURE, RESULT) = range(20)
        state = Signal(5, reset=IDLE)
        # The command and its sizes are reset-less: loaded on accept, read from VALIDATE on.
        captured = {name: Signal(width, reset_less=True) for name, width in ACTIVATION_COMMAND}
        rotation = captured["requantize"]
        total = Signal(32, reset_less=True)
        ends = {name: Signal(64)
                for name in ("src", "dst", "exponents", "maxima", "result_exponent")}
        cache_depth = max(1, scratchpad_size//4096)
        # Small per-command metadata cache. Entries are all overwritten before
        # use, so cancellation only invalidates control; no RAM scrub is needed.
        cache = Memory(28, max(2, cache_depth))
        cache_wr = cache.get_port(write_capable=True)
        cache_rd = cache.get_port()
        self.specials += cache, cache_wr, cache_rd
        config_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))

        def overlaps(a, b):
            return (captured[a] < ends[b]) & (captured[b] < ends[a])

        def protected_span(a):
            return (captured[a] < captured["protect_end"]) & (captured["protect_start"] < ends[a])
        self.comb += [
            ends["src"].eq(captured["src"] + (total << 2)),
            ends["dst"].eq(captured["dst"] + total),
            ends["exponents"].eq(captured["exponents"] +
                                 (((captured["blocks"] + C(7, 17)) >> 3) << 5)),
            ends["maxima"].eq(captured["maxima"] + (((captured["blocks"] + C(7, 17)) >> 3) << 5)),
            ends["result_exponent"].eq(captured["result_exponent"] + C(32, 25)),
            config_bad.eq((captured["elements"] == 0) | (captured["elements"][:5] != 0) |
                (captured["blocks"] == 0) | (captured["src"][:5] != 0) |
                (captured["dst"][:5] != 0) | (captured["exponents"][:5] != 0) |
                (captured["protect_start"] > captured["protect_end"]) |
                (rotation & (((captured["elements"] != 1024) & (captured["elements"] != 512)) |
                             (captured["maxima"][:5] != 0) |
                             (captured["result_exponent"][:5] != 0)))),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                (ends["exponents"] > scratchpad_size) |
                (captured["protect_end"] > scratchpad_size) |
                (rotation & ((ends["maxima"] > scratchpad_size) |
                             (ends["result_exponent"] > scratchpad_size) |
                             (captured["blocks"] > cache_depth)))),
            overlap_bad.eq(overlaps("src", "dst") | overlaps("dst", "exponents") |
                Mux(rotation, overlaps("dst", "maxima") | overlaps("dst", "result_exponent") |
                    overlaps("src", "result_exponent") | overlaps("exponents", "result_exponent") |
                    overlaps("maxima", "result_exponent"), overlaps("src", "exponents"))),
            protected_bad.eq((captured["protect_start"] != captured["protect_end"]) &
                (protected_span("dst") | Mux(rotation, protected_span("result_exponent"),
                                             protected_span("exponents")))),
        ]

        # Datapath: scratchpad reads -> scan/quantize -> packed output beats and exponents.
        block = Signal(16)
        src_base, dst_base = Signal(13, reset_less=True), Signal(13, reset_less=True)
        issued, output_index = Signal(14), Signal(14)
        read_count = Signal(13)
        read_pending = Signal()
        read_first, read_last = Signal(reset_less=True), Signal(reset_less=True) # (read_pending)
        in_scan, in_convert, clear = Signal(), Signal(), Signal()
        packed_word = Signal(256, reset_less=True)
        packed_index = Signal(2)
        packed_pending, packed_last = Signal(), Signal(reset_less=True) # (packed_pending)
        response_pop, write_word, convert_error = Signal(), Signal(), Signal()
        reserved = Signal(3)
        maximum = Signal(31)
        wide_exponent = Signal((32, True))
        metadata_e = Signal((32, True), reset_less=True)
        metadata_a = Signal(32, reset_less=True)
        metadata_length = Signal(6)
        metadata_candidate = Signal((34, True))
        metadata_bad, global_nonzero = Signal(), Signal()
        input_exponent = Signal((10, True), reset_less=True)
        declared_maximum = Signal(18, reset_less=True)
        seen_equal = Signal()
        raw_bad, raw_equal = Signal(), Signal()
        raw_bad_lanes, raw_equal_lanes = Signal(8), Signal(8)
        metadata_error = Signal()
        selected_valid, selected_ready, selected_last = Signal(), Signal(), Signal()
        selected_data = Signal(64)
        # Ping-pong mode: pp latched at VALIDATE; s_* the next block's scan under a convert.
        pp, s_active, s_done, s_issuing = Signal(), Signal(), Signal(), Signal()
        s_issued = Signal(14)
        s_pend = Signal()
        s_first, s_last = Signal(reset_less=True), Signal(reset_less=True) # (s_pend)
        buf_pend, cnv_issue, issue_main = Signal(), Signal(), Signal()
        buf_wadr = Signal(8, reset_less=True)                             # (write strobe)
        self.comb += metadata_length.eq(0)
        for bit in range(32):
            self.comb += If(metadata_a[bit], metadata_length.eq(bit + 1))
        for lane in range(8):
            raw = Signal((32, True))
            absolute = Signal(33)
            self.comb += [
                raw.eq(responses.dout[32*lane:32*(lane+1)]),
                absolute.eq(Mux(raw < 0, -raw, raw)),
                raw_bad_lanes[lane].eq(absolute > declared_maximum),
                raw_equal_lanes[lane].eq(absolute == declared_maximum),
                integer_quantize.sink.data[19*lane:19*(lane+1)].eq(raw[:19]),
            ]
        self.comb += [
            # int8 (7 magnitude bits) of the block's raw sums, normalized by 2**-5 (1024: 1/32)
            # or 2**-4 (512).
            metadata_candidate.eq(metadata_e + metadata_length - Mux(captured["elements"] == 512, 11, 12)),
            metadata_bad.eq((metadata_e < -127) | (metadata_e > 121) | (metadata_a > 131072)),
            cache_wr.adr.eq(block), cache_wr.dat_w.eq(Cat(metadata_e[:10], metadata_a[:18])),
            cache_wr.we.eq((state == META_CHECK) & ~metadata_bad & ~self.cancel),
            cache_rd.adr.eq(block),
            self.cmd.ready.eq((state == IDLE) & ~self.cancel), self.busy.eq(state != IDLE),
            in_scan.eq((state == SCAN) | (state == SCAN_DRAIN)),
            in_convert.eq((state == CONVERT) | (state == CONVERT_DRAIN)),
            clear.eq(self.cancel | (~in_scan & ~in_convert & ~(s_active & (state != FAILED)))),
            s_issuing.eq(s_active & (s_issued < read_count) & (state != FAILED) & ~self.cancel),
            cnv_issue.eq((state == CONVERT) & ((reserved < 2) | response_pop) & ~self.cancel),
            issue_main.eq(Mux(pp & in_convert, cnv_issue,
                              self.sp_rd_req & ~s_issuing & (in_scan | in_convert))),
            buf_rd.adr.eq(Cat(issued[:7], block[0])),
            buf_wr.adr.eq(buf_wadr), buf_wr.dat_w.eq(self.sp_rd_dat),
            buf_wr.we.eq(pp & ((read_pending & in_scan) | s_pend) & ~self.cancel),
            scan.cancel.eq(clear), quantize.cancel.eq(clear), integer_quantize.cancel.eq(clear),
            responses.reset.eq(clear),
            read_count.eq(captured["elements"] >> 3),
            scan.sink.valid.eq(((read_pending & in_scan) | s_pend) & ~self.cancel),
            scan.sink.data.eq(self.sp_rd_dat),
            scan.sink.first.eq(Mux(s_pend, s_first, read_first)),
            scan.sink.last.eq(Mux(s_pend, s_last, read_last)),
            scan.source.ready.eq(1),
            maximum.eq(Mux(scan.source.absmax > self.absmax, scan.source.absmax, self.absmax)),
            responses.we.eq(Mux(pp, buf_pend, read_pending) & in_convert & ~self.cancel),
            responses.din.eq(Cat(Mux(pp, buf_rd.dat_r, self.sp_rd_dat), read_first, read_last)),
            quantize.sink.valid.eq(responses.readable & in_convert & ~rotation & ~self.cancel),
            quantize.sink.data.eq(responses.dout[:256]),
            quantize.sink.first.eq(responses.dout[256]),
            quantize.sink.last.eq(responses.dout[257]), quantize.sink.e8.eq(self.exponent),
            integer_quantize.sink.valid.eq(responses.readable & in_convert & rotation &
                                           ~self.cancel),
            integer_quantize.sink.first.eq(responses.dout[256]),
            integer_quantize.sink.last.eq(responses.dout[257]),
            integer_quantize.sink.e8.eq(self.exponent - input_exponent +
                                        Mux(captured["elements"] == 512, 4, 5)),
            response_pop.eq(Mux(rotation, integer_quantize.sink.valid & integer_quantize.sink.ready,
                               quantize.sink.valid & quantize.sink.ready)),
            responses.re.eq(response_pop),
            reserved.eq(responses.level + Mux(pp, buf_pend, read_pending)),
            self.sp_rd_req.eq(~self.cancel & ((state == SCAN) |
                ((state == CONVERT) & ~pp & ((reserved < 2) | response_pop)) | s_issuing |
                (state == META_E_REQ) | (state == META_A_REQ))),
            self.sp_rd_adr.eq(Mux(state == META_E_REQ, (captured["exponents"] >> 5) + (block >> 3),
                Mux(state == META_A_REQ, (captured["maxima"] >> 5) + (block >> 3),
                    Mux(s_issuing, src_base + read_count + s_issued, src_base + issued)))),
            convert_error.eq(in_convert & ((quantize.source.valid & (quantize.source.error != 0)) |
                (responses.we & ~responses.writable))),
            raw_bad.eq(raw_bad_lanes != 0), raw_equal.eq(raw_equal_lanes != 0),
            # Do not depend on sink.ready/pop: write grants can feed back into
            # pipeline readiness, so pop->error->write request creates a loop.
            metadata_error.eq(rotation & in_convert & ((responses.readable & raw_bad) |
                (packed_pending & packed_last & ~seen_equal))),
            selected_valid.eq(Mux(rotation, integer_quantize.source.valid, quantize.source.valid)),
            selected_data.eq(Mux(rotation, integer_quantize.source.data, quantize.source.data)),
            selected_last.eq(Mux(rotation, integer_quantize.source.last, quantize.source.last)),
            selected_ready.eq(in_convert & (~packed_pending | self.sp_wr_gnt) & ~self.cancel),
            quantize.source.ready.eq(selected_ready & ~rotation),
            integer_quantize.source.ready.eq(selected_ready & rotation),
            self.sp_wr_req.eq(~self.cancel & ~convert_error & ~metadata_error &
                ((in_convert & packed_pending) | (state == EXPONENT) | (state == RESULT))),
            self.sp_wr_adr.eq(Mux(state == EXPONENT,
                (captured["exponents"] >> 5) + (block >> 3),
                Mux(state == RESULT, captured["result_exponent"] >> 5, dst_base + output_index))),
            self.sp_wr_we.eq(Mux(state == EXPONENT, C(1, 8) << block[:3],
                                 Mux(state == RESULT, 1, 0xff))),
            wide_exponent.eq(self.exponent),
            self.sp_wr_dat.eq(Mux((state == EXPONENT) | (state == RESULT),
                                  Replicate(wide_exponent, 8), packed_word)),
            write_word.eq(in_convert & self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear, read_pending.eq(0), buf_pend.eq(0), s_pend.eq(0)).Else(
            read_pending.eq(self.sp_rd_req & ~s_issuing & (in_scan | in_convert)),
            buf_pend.eq(cnv_issue & pp),
            s_pend.eq(s_issuing),
            If(issue_main, read_first.eq(issued == 0), read_last.eq(issued == read_count - 1)),
        )
        self.sync += [
            If(s_issuing, s_first.eq(s_issued == 0), s_last.eq(s_issued == read_count - 1)),
            If(s_issuing, buf_wadr.eq(Cat(s_issued[:7], ~block[0]))
            ).Elif(self.sp_rd_req, buf_wadr.eq(Cat(issued[:7], block[0]))),
        ]
        packing = [Case(packed_index, {index: packed_word[64*index:64*(index+1)].eq(selected_data)
                                      for index in range(4)}),
                   packed_index.eq(packed_index + 1),
                   If(packed_index == 3, packed_pending.eq(1), packed_last.eq(selected_last))]
        # Control.
        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0), self.blocks_done.eq(0),
            self.read_beats.eq(0), self.output_beats.eq(0), packed_pending.eq(0),
            s_active.eq(0), s_done.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, self.read_beats.eq(self.read_beats + 1)),
            If(issue_main, issued.eq(issued + 1),
                If(issued == read_count - 1,
                    state.eq(Mux(state == SCAN, SCAN_DRAIN, CONVERT_DRAIN)))),
            If(s_issuing, s_issued.eq(s_issued + 1)),
            If(write_word, packed_pending.eq(0), output_index.eq(output_index + 1),
                self.output_beats.eq(self.output_beats + 1), If(packed_last,
                    If(rotation,
                        self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT),
                    ).Else(state.eq(EXPONENT)))),
            If(selected_valid & selected_ready, *packing),
            If(rotation & response_pop & raw_equal, seen_equal.eq(1)),
            If(in_scan & scan.source.valid,
                self.absmax.eq(maximum), If(scan.source.last, state.eq(CHOOSE))),
            # The next block's scan under this block's convert (pp): absmax and a done flag.
            If(s_active & ~in_scan & scan.source.valid,
                self.absmax.eq(maximum), If(scan.source.last, s_done.eq(1))),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    total.eq(self.cmd.elements * self.cmd.blocks),
                    src_base.eq(self.cmd.src >> 5), dst_base.eq(self.cmd.dst >> 5),
                    block.eq(0), issued.eq(0), output_index.eq(0), packed_index.eq(0),
                    packed_pending.eq(0),
                    self.absmax.eq(0), self.exponent.eq(0), self.blocks_done.eq(0),
                    global_nonzero.eq(0), seen_equal.eq(0),
                    self.read_beats.eq(0), self.output_beats.eq(0), self.error.eq(0),
                    s_active.eq(0), s_done.eq(0),
                    state.eq(VALIDATE)),
                VALIDATE: If(config_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(config_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(FAILED),
                ).Else(
                    pp.eq(~rotation & (captured["elements"] <= 1024) & (captured["blocks"] > 1)),
                    state.eq(Mux(rotation, META_E_REQ, SCAN))),
                META_E_REQ: state.eq(META_E_CAPTURE),
                META_E_CAPTURE: [
                    metadata_e.eq(
                        Array(self.sp_rd_dat[32*i:32*(i+1)] for i in range(8))[block[:3]]),
                    state.eq(META_A_REQ)],
                META_A_REQ: state.eq(META_A_CAPTURE),
                META_A_CAPTURE: [
                    metadata_a.eq(
                        Array(self.sp_rd_dat[32*i:32*(i+1)] for i in range(8))[block[:3]]),
                    state.eq(META_CHECK)],
                META_CHECK: If(metadata_bad, self.error.eq(128), state.eq(FAILED)).Else(
                    If(metadata_a != 0, global_nonzero.eq(1),
                        If(~global_nonzero | (metadata_candidate > self.exponent),
                            self.exponent.eq(metadata_candidate))),
                    If(block == captured["blocks"] - 1, state.eq(GLOBAL_CHOOSE)).Else(
                        block.eq(block + 1), state.eq(META_E_REQ))),
                GLOBAL_CHOOSE: If((self.exponent < -127) | (self.exponent > 121),
                    self.error.eq(16), state.eq(FAILED),
                ).Else(block.eq(0), state.eq(CACHE_READ)),
                CACHE_READ: state.eq(CACHE_CAPTURE),
                CACHE_CAPTURE: [input_exponent.eq(cache_rd.dat_r[:10]),
                                declared_maximum.eq(cache_rd.dat_r[10:]),
                                seen_equal.eq(0), state.eq(CONVERT)],
                CHOOSE: If((self.absmax != 0) & (self.absmax[23:] < 6),
                    self.error.eq(16), state.eq(FAILED),
                ).Else(self.exponent.eq(Mux(self.absmax == 0, 0, self.absmax[23:] - 133)),
                       issued.eq(0), state.eq(CONVERT),
                       # pp: the next block's scan starts with this block's convert.
                       If(pp,
                           self.absmax.eq(0),
                           If(block != captured["blocks"] - 1,
                               s_active.eq(1), s_issued.eq(0), s_done.eq(0)))),
                EXPONENT: If(self.sp_wr_gnt,
                    self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT)),
                NEXT: If(block == captured["blocks"] - 1,
                    state.eq(Mux(rotation, RESULT, FINISH)),
                ).Elif(~pp | s_done,           # pp: the next block is scanned (absmax kept)
                    block.eq(block + 1), src_base.eq(src_base + read_count),
                    dst_base.eq(dst_base + (captured["elements"] >> 5)),
                    issued.eq(0), output_index.eq(0), packed_index.eq(0), packed_pending.eq(0),
                    s_active.eq(0),
                    If(~pp, self.absmax.eq(0)),
                    state.eq(Mux(rotation, CACHE_READ, Mux(pp, CHOOSE, SCAN)))),
                RESULT: If(self.sp_wr_gnt, state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
            # Failure wins over state transitions and never publishes success.
            If((in_scan | s_active) & scan.source.valid & scan.source.error,
                self.error.eq(8), state.eq(FAILED)),
            If(convert_error, self.error.eq(32), state.eq(FAILED)),
            If(metadata_error, self.error.eq(128), state.eq(FAILED)),
        )

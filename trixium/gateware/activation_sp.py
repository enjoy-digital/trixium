# SPDX-License-Identifier: BSD-2-Clause
"""Resident FP32 conversion and scaled integer requantization on shared scratchpad."""
from migen import *
from migen.genlib.fifo import SyncFIFO
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.activation_quantize import Fp32AbsmaxWord, Fp32ToQ8
from trixium.gateware.integer_quantize import IntegerToQ8

ACTIVATION_COMMAND = [("src", 24), ("dst", 24), ("exponents", 24),
                      ("elements", 16), ("blocks", 16),
                      ("requantize", 1), ("maxima", 24), ("result_exponent", 24),
                      ("protect_start", 24), ("protect_end", 24)]


class BonsaiActivationScratchpad(LiteXModule):
    """Scan, select e8, reread and pack each independent FP32 block into Q8.

    src/dst/exponents are 32-byte-aligned bases. elements is a positive multiple
    of 32; blocks is nonzero. Inputs/outputs are contiguous across blocks. Each
    block gets one signed int32 exponent, preserving unused lanes in the last
    exponent beat. blocks=1 selects one exponent for a complete vector.

    The complete input/output/rounded-exponent spans must fit physical memory
    and be mutually disjoint. An optional half-open protected interval (empty
    by default) additionally protects a sequencer's descriptor list from writes.
    Bounds use wide arithmetic before any scratchpad access. The integrator
    grants every read, returns it one cycle later and gates writes with gnt;
    no other owner may change these buffers while the command is active.

    requantize selects raw 1024-element int32 Hadamard blocks instead of FP32.
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
            raise ValueError("scratchpad_size must be aligned within the 13-bit client address space")
        self.cmd = stream.Endpoint(ACTIVATION_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.blocks_done = Signal(16)
        self.read_beats = Signal(32)
        self.output_beats = Signal(32)
        self.absmax = Signal(31)
        self.exponent = Signal((10, True))
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)
        self.scan = scan = Fp32AbsmaxWord()
        self.quantize = quantize = Fp32ToQ8()
        self.integer_quantize = integer_quantize = IntegerToQ8()
        self.responses = responses = ResetInserter()(SyncFIFO(width=258, depth=2))

        (IDLE, VALIDATE, SCAN, SCAN_DRAIN, CHOOSE, CONVERT, CONVERT_DRAIN, EXPONENT,
         NEXT, FINISH, FAILED, META_E_REQ, META_E_CAPTURE, META_A_REQ, META_A_CAPTURE,
         META_CHECK, GLOBAL_CHOOSE, CACHE_READ, CACHE_CAPTURE, RESULT) = range(20)
        state = Signal(5, reset=IDLE)
        captured = {name: Signal(width) for name, width in ACTIVATION_COMMAND}
        rotation = captured["requantize"]
        total = Signal(32)
        ends = {name: Signal(64) for name in ("src", "dst", "exponents", "maxima", "result_exponent")}
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
            ends["exponents"].eq(captured["exponents"] + (((captured["blocks"] + C(7, 17)) >> 3) << 5)),
            ends["maxima"].eq(captured["maxima"] + (((captured["blocks"] + C(7, 17)) >> 3) << 5)),
            ends["result_exponent"].eq(captured["result_exponent"] + C(32, 25)),
            config_bad.eq((captured["elements"] == 0) | (captured["elements"][:5] != 0) |
                (captured["blocks"] == 0) | (captured["src"][:5] != 0) |
                (captured["dst"][:5] != 0) | (captured["exponents"][:5] != 0) |
                (captured["protect_start"] > captured["protect_end"]) |
                (rotation & ((captured["elements"] != 1024) | (captured["maxima"][:5] != 0) |
                             (captured["result_exponent"][:5] != 0)))),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                (ends["exponents"] > scratchpad_size) | (captured["protect_end"] > scratchpad_size) |
                (rotation & ((ends["maxima"] > scratchpad_size) | (ends["result_exponent"] > scratchpad_size) |
                             (captured["blocks"] > cache_depth)))),
            overlap_bad.eq(overlaps("src", "dst") | overlaps("dst", "exponents") |
                Mux(rotation, overlaps("dst", "maxima") | overlaps("dst", "result_exponent") |
                    overlaps("src", "result_exponent") | overlaps("exponents", "result_exponent") |
                    overlaps("maxima", "result_exponent"), overlaps("src", "exponents"))),
            protected_bad.eq((captured["protect_start"] != captured["protect_end"]) &
                (protected_span("dst") | Mux(rotation, protected_span("result_exponent"), protected_span("exponents")))),
        ]

        block = Signal(16)
        src_base, dst_base = Signal(13), Signal(13)
        issued, output_index = Signal(14), Signal(14)
        read_count = Signal(13)
        read_pending, read_first, read_last = Signal(), Signal(), Signal()
        in_scan, in_convert, clear = Signal(), Signal(), Signal()
        packed_word = Signal(256)
        packed_index = Signal(2)
        packed_pending, packed_last = Signal(), Signal()
        response_pop, write_word, convert_error = Signal(), Signal(), Signal()
        reserved = Signal(3)
        maximum = Signal(31)
        wide_exponent = Signal((32, True))
        metadata_e = Signal((32, True))
        metadata_a = Signal(32)
        metadata_length = Signal(6)
        metadata_candidate = Signal((34, True))
        metadata_bad, global_nonzero = Signal(), Signal()
        input_exponent = Signal((10, True))
        declared_maximum = Signal(18)
        seen_equal = Signal()
        raw_bad, raw_equal = Signal(), Signal()
        raw_bad_lanes, raw_equal_lanes = Signal(8), Signal(8)
        metadata_error = Signal()
        selected_valid, selected_ready, selected_last = Signal(), Signal(), Signal()
        selected_data = Signal(64)
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
            metadata_candidate.eq(metadata_e + metadata_length - 12),
            metadata_bad.eq((metadata_e < -127) | (metadata_e > 121) | (metadata_a > 131072)),
            cache_wr.adr.eq(block), cache_wr.dat_w.eq(Cat(metadata_e[:10], metadata_a[:18])),
            cache_wr.we.eq((state == META_CHECK) & ~metadata_bad & ~self.cancel),
            cache_rd.adr.eq(block),
            self.cmd.ready.eq((state == IDLE) & ~self.cancel), self.busy.eq(state != IDLE),
            in_scan.eq((state == SCAN) | (state == SCAN_DRAIN)),
            in_convert.eq((state == CONVERT) | (state == CONVERT_DRAIN)),
            clear.eq(self.cancel | (~in_scan & ~in_convert)),
            scan.cancel.eq(clear), quantize.cancel.eq(clear), integer_quantize.cancel.eq(clear), responses.reset.eq(clear),
            read_count.eq(captured["elements"] >> 3),
            scan.sink.valid.eq(read_pending & in_scan & ~self.cancel),
            scan.sink.data.eq(self.sp_rd_dat), scan.sink.first.eq(read_first), scan.sink.last.eq(read_last),
            scan.source.ready.eq(1),
            maximum.eq(Mux(scan.source.absmax > self.absmax, scan.source.absmax, self.absmax)),
            responses.we.eq(read_pending & in_convert & ~self.cancel),
            responses.din.eq(Cat(self.sp_rd_dat, read_first, read_last)),
            quantize.sink.valid.eq(responses.readable & in_convert & ~rotation & ~self.cancel),
            quantize.sink.data.eq(responses.dout[:256]), quantize.sink.first.eq(responses.dout[256]),
            quantize.sink.last.eq(responses.dout[257]), quantize.sink.e8.eq(self.exponent),
            integer_quantize.sink.valid.eq(responses.readable & in_convert & rotation & ~self.cancel),
            integer_quantize.sink.first.eq(responses.dout[256]), integer_quantize.sink.last.eq(responses.dout[257]),
            integer_quantize.sink.e8.eq(self.exponent - input_exponent + 5),
            response_pop.eq(Mux(rotation, integer_quantize.sink.valid & integer_quantize.sink.ready,
                               quantize.sink.valid & quantize.sink.ready)), responses.re.eq(response_pop),
            reserved.eq(responses.level + read_pending),
            self.sp_rd_req.eq(~self.cancel & ((state == SCAN) |
                ((state == CONVERT) & ((reserved < 2) | response_pop)) |
                (state == META_E_REQ) | (state == META_A_REQ))),
            self.sp_rd_adr.eq(Mux(state == META_E_REQ, (captured["exponents"] >> 5) + (block >> 3),
                Mux(state == META_A_REQ, (captured["maxima"] >> 5) + (block >> 3), src_base + issued))),
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
            self.sp_wr_we.eq(Mux(state == EXPONENT, C(1, 8) << block[:3], Mux(state == RESULT, 1, 0xff))),
            wide_exponent.eq(self.exponent),
            self.sp_wr_dat.eq(Mux((state == EXPONENT) | (state == RESULT), Replicate(wide_exponent, 8), packed_word)),
            write_word.eq(in_convert & self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear, read_pending.eq(0)).Else(
            read_pending.eq(self.sp_rd_req & (in_scan | in_convert)),
            If(self.sp_rd_req, read_first.eq(issued == 0), read_last.eq(issued == read_count - 1)),
        )
        packing = [Case(packed_index, {index: packed_word[64*index:64*(index+1)].eq(selected_data)
                                      for index in range(4)}),
                   packed_index.eq(packed_index + 1),
                   If(packed_index == 3, packed_pending.eq(1), packed_last.eq(selected_last))]
        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0), self.blocks_done.eq(0),
            self.read_beats.eq(0), self.output_beats.eq(0), packed_pending.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, self.read_beats.eq(self.read_beats + 1)),
            If(self.sp_rd_req & (in_scan | in_convert), issued.eq(issued + 1),
                If(issued == read_count - 1,
                    state.eq(Mux(state == SCAN, SCAN_DRAIN, CONVERT_DRAIN)))),
            If(write_word, packed_pending.eq(0), output_index.eq(output_index + 1),
                self.output_beats.eq(self.output_beats + 1), If(packed_last,
                    If(rotation, self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT)).Else(state.eq(EXPONENT)))),
            If(selected_valid & selected_ready, *packing),
            If(rotation & response_pop & raw_equal, seen_equal.eq(1)),
            If(in_scan & scan.source.valid,
                self.absmax.eq(maximum), If(scan.source.last, state.eq(CHOOSE))),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    total.eq(self.cmd.elements * self.cmd.blocks),
                    src_base.eq(self.cmd.src >> 5), dst_base.eq(self.cmd.dst >> 5),
                    block.eq(0), issued.eq(0), output_index.eq(0), packed_index.eq(0), packed_pending.eq(0),
                    self.absmax.eq(0), self.exponent.eq(0), self.blocks_done.eq(0),
                    global_nonzero.eq(0), seen_equal.eq(0),
                    self.read_beats.eq(0), self.output_beats.eq(0), self.error.eq(0), state.eq(VALIDATE)),
                VALIDATE: If(config_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(config_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(FAILED),
                ).Else(state.eq(Mux(rotation, META_E_REQ, SCAN))),
                META_E_REQ: state.eq(META_E_CAPTURE),
                META_E_CAPTURE: [metadata_e.eq(Array(self.sp_rd_dat[32*i:32*(i+1)] for i in range(8))[block[:3]]),
                                 state.eq(META_A_REQ)],
                META_A_REQ: state.eq(META_A_CAPTURE),
                META_A_CAPTURE: [metadata_a.eq(Array(self.sp_rd_dat[32*i:32*(i+1)] for i in range(8))[block[:3]]),
                                 state.eq(META_CHECK)],
                META_CHECK: If(metadata_bad, self.error.eq(128), state.eq(FAILED)).Else(
                    If(metadata_a != 0, global_nonzero.eq(1),
                        If(~global_nonzero | (metadata_candidate > self.exponent), self.exponent.eq(metadata_candidate))),
                    If(block == captured["blocks"] - 1, state.eq(GLOBAL_CHOOSE)).Else(
                        block.eq(block + 1), state.eq(META_E_REQ))),
                GLOBAL_CHOOSE: If((self.exponent < -127) | (self.exponent > 121),
                    self.error.eq(16), state.eq(FAILED),
                ).Else(block.eq(0), state.eq(CACHE_READ)),
                CACHE_READ: state.eq(CACHE_CAPTURE),
                CACHE_CAPTURE: [input_exponent.eq(cache_rd.dat_r[:10]), declared_maximum.eq(cache_rd.dat_r[10:]),
                                seen_equal.eq(0), state.eq(CONVERT)],
                CHOOSE: If((self.absmax != 0) & (self.absmax[23:] < 6),
                    self.error.eq(16), state.eq(FAILED),
                ).Else(self.exponent.eq(Mux(self.absmax == 0, 0, self.absmax[23:] - 133)),
                       issued.eq(0), state.eq(CONVERT)),
                EXPONENT: If(self.sp_wr_gnt, self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT)),
                NEXT: If(block == captured["blocks"] - 1, state.eq(Mux(rotation, RESULT, FINISH))).Else(
                    block.eq(block + 1), src_base.eq(src_base + read_count),
                    dst_base.eq(dst_base + (captured["elements"] >> 5)),
                    issued.eq(0), output_index.eq(0), packed_index.eq(0), packed_pending.eq(0),
                    self.absmax.eq(0), state.eq(Mux(rotation, CACHE_READ, SCAN))),
                RESULT: If(self.sp_wr_gnt, state.eq(FINISH)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
            # Failure wins over state transitions and never publishes success.
            If(in_scan & scan.source.valid & scan.source.error,
                self.error.eq(8), state.eq(FAILED)),
            If(convert_error, self.error.eq(32), state.eq(FAILED)),
            If(metadata_error, self.error.eq(128), state.eq(FAILED)),
        )

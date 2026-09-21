# SPDX-License-Identifier: BSD-2-Clause
"""Resident two-pass FP32 block/vector quantization on the shared scratchpad."""
from migen import *
from migen.genlib.fifo import SyncFIFO
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.activation_quantize import Fp32AbsmaxWord, Fp32ToQ8

ACTIVATION_COMMAND = [("src", 24), ("dst", 24), ("exponents", 24),
                      ("elements", 16), ("blocks", 16),
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

    error bits: 0 geometry/alignment, 1 capacity, 2 overlap, 3 nonfinite input,
    4 unsafe exponent, 5 internal conversion/credit, 6 protected-list overlap.
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
        self.responses = responses = ResetInserter()(SyncFIFO(width=258, depth=2))

        IDLE, VALIDATE, SCAN, SCAN_DRAIN, CHOOSE, CONVERT, CONVERT_DRAIN, EXPONENT, NEXT, FINISH, FAILED = range(11)
        state = Signal(4, reset=IDLE)
        captured = {name: Signal(width) for name, width in ACTIVATION_COMMAND}
        total = Signal(32)
        ends = {name: Signal(64) for name in ("src", "dst", "exponents")}
        config_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))
        self.comb += [
            ends["src"].eq(captured["src"] + (total << 2)),
            ends["dst"].eq(captured["dst"] + total),
            ends["exponents"].eq(captured["exponents"] + (((captured["blocks"] + C(7, 17)) >> 3) << 5)),
            config_bad.eq((captured["elements"] == 0) | (captured["elements"][:5] != 0) |
                (captured["blocks"] == 0) | (captured["src"][:5] != 0) |
                (captured["dst"][:5] != 0) | (captured["exponents"][:5] != 0) |
                (captured["protect_start"] > captured["protect_end"])),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                (ends["exponents"] > scratchpad_size) | (captured["protect_end"] > scratchpad_size)),
            overlap_bad.eq(((captured["src"] < ends["dst"]) & (captured["dst"] < ends["src"])) |
                ((captured["src"] < ends["exponents"]) & (captured["exponents"] < ends["src"])) |
                ((captured["dst"] < ends["exponents"]) & (captured["exponents"] < ends["dst"]))),
            protected_bad.eq((captured["protect_start"] != captured["protect_end"]) &
                (((captured["dst"] < captured["protect_end"]) & (captured["protect_start"] < ends["dst"])) |
                 ((captured["exponents"] < captured["protect_end"]) &
                  (captured["protect_start"] < ends["exponents"])))),
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
        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel), self.busy.eq(state != IDLE),
            in_scan.eq((state == SCAN) | (state == SCAN_DRAIN)),
            in_convert.eq((state == CONVERT) | (state == CONVERT_DRAIN)),
            clear.eq(self.cancel | (state == IDLE) | (state == VALIDATE) |
                     (state == CHOOSE) | (state == NEXT) | (state == FAILED)),
            scan.cancel.eq(clear), quantize.cancel.eq(clear), responses.reset.eq(clear),
            read_count.eq(captured["elements"] >> 3),
            scan.sink.valid.eq(read_pending & in_scan & ~self.cancel),
            scan.sink.data.eq(self.sp_rd_dat), scan.sink.first.eq(read_first), scan.sink.last.eq(read_last),
            scan.source.ready.eq(1),
            maximum.eq(Mux(scan.source.absmax > self.absmax, scan.source.absmax, self.absmax)),
            responses.we.eq(read_pending & in_convert & ~self.cancel),
            responses.din.eq(Cat(self.sp_rd_dat, read_first, read_last)),
            quantize.sink.valid.eq(responses.readable & in_convert & ~self.cancel),
            quantize.sink.data.eq(responses.dout[:256]), quantize.sink.first.eq(responses.dout[256]),
            quantize.sink.last.eq(responses.dout[257]), quantize.sink.e8.eq(self.exponent),
            response_pop.eq(quantize.sink.valid & quantize.sink.ready), responses.re.eq(response_pop),
            reserved.eq(responses.level + read_pending),
            self.sp_rd_req.eq(~self.cancel & ((state == SCAN) |
                ((state == CONVERT) & ((reserved < 2) | response_pop)))),
            self.sp_rd_adr.eq(src_base + issued),
            convert_error.eq(in_convert & ((quantize.source.valid & (quantize.source.error != 0)) |
                (responses.we & ~responses.writable))),
            quantize.source.ready.eq(in_convert & (~packed_pending | self.sp_wr_gnt) & ~self.cancel),
            self.sp_wr_req.eq(~self.cancel & ~convert_error &
                ((in_convert & packed_pending) | (state == EXPONENT))),
            self.sp_wr_adr.eq(Mux(state == EXPONENT,
                (captured["exponents"] >> 5) + (block >> 3), dst_base + output_index)),
            self.sp_wr_we.eq(Mux(state == EXPONENT, C(1, 8) << block[:3], 0xff)),
            wide_exponent.eq(self.exponent),
            self.sp_wr_dat.eq(Mux(state == EXPONENT, Replicate(wide_exponent, 8), packed_word)),
            write_word.eq(in_convert & self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(clear, read_pending.eq(0)).Else(
            read_pending.eq(self.sp_rd_req),
            If(self.sp_rd_req, read_first.eq(issued == 0), read_last.eq(issued == read_count - 1)),
        )
        packing = [Case(packed_index, {index: packed_word[64*index:64*(index+1)].eq(quantize.source.data)
                                      for index in range(4)}),
                   packed_index.eq(packed_index + 1),
                   If(packed_index == 3, packed_pending.eq(1), packed_last.eq(quantize.source.last))]
        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0), self.blocks_done.eq(0),
            self.read_beats.eq(0), self.output_beats.eq(0), packed_pending.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, issued.eq(issued + 1), self.read_beats.eq(self.read_beats + 1),
                If(issued == read_count - 1,
                    state.eq(Mux(state == SCAN, SCAN_DRAIN, CONVERT_DRAIN)))),
            If(write_word, packed_pending.eq(0), output_index.eq(output_index + 1),
                self.output_beats.eq(self.output_beats + 1), If(packed_last, state.eq(EXPONENT))),
            If(quantize.source.valid & quantize.source.ready, *packing),
            If(in_scan & scan.source.valid,
                self.absmax.eq(maximum), If(scan.source.last, state.eq(CHOOSE))),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    total.eq(self.cmd.elements * self.cmd.blocks),
                    src_base.eq(self.cmd.src >> 5), dst_base.eq(self.cmd.dst >> 5),
                    block.eq(0), issued.eq(0), output_index.eq(0), packed_index.eq(0), packed_pending.eq(0),
                    self.absmax.eq(0), self.exponent.eq(0), self.blocks_done.eq(0),
                    self.read_beats.eq(0), self.output_beats.eq(0), self.error.eq(0), state.eq(VALIDATE)),
                VALIDATE: If(config_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(config_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(FAILED),
                ).Else(state.eq(SCAN)),
                CHOOSE: If((self.absmax != 0) & (self.absmax[23:] < 6),
                    self.error.eq(16), state.eq(FAILED),
                ).Else(self.exponent.eq(Mux(self.absmax == 0, 0, self.absmax[23:] - 133)),
                       issued.eq(0), state.eq(CONVERT)),
                EXPONENT: If(self.sp_wr_gnt, self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT)),
                NEXT: If(block == captured["blocks"] - 1, state.eq(FINISH)).Else(
                    block.eq(block + 1), src_base.eq(src_base + read_count),
                    dst_base.eq(dst_base + (captured["elements"] >> 5)),
                    issued.eq(0), output_index.eq(0), packed_index.eq(0), packed_pending.eq(0),
                    self.absmax.eq(0), state.eq(SCAN)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
            # Failure wins over state transitions and never publishes success.
            If(in_scan & scan.source.valid & scan.source.error,
                self.error.eq(8), state.eq(FAILED)),
            If(convert_error, self.error.eq(32), state.eq(FAILED)),
        )

# SPDX-License-Identifier: BSD-2-Clause
"""Resident projection-result restoration using the shared two-port scratchpad."""
from migen import *
from migen.genlib.fifo import SyncFIFO
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.projection_restore import I32ToFp32Scaled


RESTORE_COMMAND = [("src", 24), ("dst", 24), ("exponents", 24),
                   ("elements", 16), ("adjustment", 32),
                   ("protect_start", 24), ("protect_end", 24)]


class BonsaiProjectionRestoreScratchpad(LiteXModule):
    """Read resident signed e8, then restore int32 -> FP32 with exponent e8+R-B.

    exponents names a 32-byte aligned beat whose low lane holds signed int32 e8.
    adjustment is a signed int32 (normally whole-matrix R-B). Both are added
    at 33 bits, then checked against [-149,127] BEFORE narrowing. src/dst are
    32-byte aligned and elements is in [1,65535]. The full rounded source and
    destination beat spans and exponent beat must fit physical memory. Partial
    final output beats preserve inactive int32 lanes; inactive input lanes are
    zeroed before conversion and cannot produce an overflow error.

    Exact src==dst in-place conversion is allowed (read always precedes write).
    Other source/output overlap is rejected. Output may not overlap the exponent
    beat or the optional protected interval, even though the exponent is read
    first. Read-only source/exponent overlap is permitted. The integrator must
    provide exclusive ownership, grant every read and return it one clock later;
    writes occur only with sp_wr_req & sp_wr_gnt, with stable stalled payload.

    error bits: 0 geometry/alignment, 1 bounds, 2 forbidden overlap, 3 unsupported
    final exponent, 4 FP32 result overflow, 5 response credit/internal conversion,
    6 protected output. Any failure stays busy until cancel/reset; all partial
    output is invalid. done publishes success only after all accepted writes
    and a drain cycle. This client does not combine K-segments or publish a
    model result. Optional sequencer unit 8 supplies exclusive ownership and
    the protected descriptor-list interval through the shared integration.
    """
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client address space")
        self.cmd = stream.Endpoint(RESTORE_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.read_beats, self.output_beats = Signal(32), Signal(32)
        self.exponent = Signal((10, True))
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)
        self.restore = restore = I32ToFp32Scaled()
        self.responses = responses = ResetInserter()(SyncFIFO(width=258, depth=2))

        IDLE, VALIDATE, META_REQ, META_CAPTURE, META_CHECK, RUN, DRAIN, FINISH, FAILED = range(9)
        state = Signal(4, reset=IDLE)
        captured = {name: Signal(width) for name, width in RESTORE_COMMAND}
        count = Signal(14)
        issued, retired = Signal(14), Signal(14)
        ends = {name: Signal(64) for name in ("src", "dst", "exponents")}
        geometry_bad, bounds_bad, overlap_bad, protected_bad = (Signal() for _ in range(4))
        def overlap(a, b):
            return (captured[a] < ends[b]) & (captured[b] < ends[a])
        self.comb += [
            ends["src"].eq(captured["src"] + (count << 5)),
            ends["dst"].eq(captured["dst"] + (count << 5)),
            ends["exponents"].eq(captured["exponents"] + C(32, 25)),
            geometry_bad.eq((captured["elements"] == 0) | (captured["src"][:5] != 0) |
                (captured["dst"][:5] != 0) | (captured["exponents"][:5] != 0) |
                (captured["protect_start"] > captured["protect_end"])),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["dst"] > scratchpad_size) |
                (ends["exponents"] > scratchpad_size) | (captured["protect_end"] > scratchpad_size)),
            overlap_bad.eq((overlap("src", "dst") & (captured["src"] != captured["dst"])) |
                overlap("dst", "exponents")),
            protected_bad.eq((captured["protect_start"] != captured["protect_end"]) &
                (captured["dst"] < captured["protect_end"]) & (captured["protect_start"] < ends["dst"])),
        ]

        metadata = Signal((32, True))
        adjustment = Signal((32, True))
        combined = Signal((33, True))
        exponent_bad = Signal()
        self.comb += [adjustment.eq(captured["adjustment"]), combined.eq(metadata + adjustment),
                      exponent_bad.eq((combined < -149) | (combined > 127))]

        active, clear, read_pending, read_first, read_last = (Signal() for _ in range(5))
        pop, credit_error, conversion_error, overflow, failed, written = (Signal() for _ in range(6))
        reserved = Signal(3)
        tail = captured["elements"][:3]
        mask = Signal(8)
        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel), self.busy.eq(state != IDLE),
            active.eq((state == RUN) | (state == DRAIN)), clear.eq(self.cancel | ~active),
            restore.cancel.eq(clear), responses.reset.eq(clear),
            responses.we.eq(read_pending & active & ~self.cancel),
            responses.din.eq(Cat(self.sp_rd_dat, read_first, read_last)),
            restore.sink.valid.eq(responses.readable & active & ~self.cancel),
            restore.sink.exponent.eq(self.exponent), restore.sink.first.eq(responses.dout[256]),
            restore.sink.last.eq(responses.dout[257]),
            pop.eq(restore.sink.valid & restore.sink.ready), responses.re.eq(pop),
            reserved.eq(responses.level + read_pending),
            # Error detection does not depend on downstream ready/grant, avoiding
            # a credit/pop -> failure -> write request combinational feedback.
            credit_error.eq(responses.we & ~responses.writable),
            conversion_error.eq(restore.source.valid & restore.source.error[8]),
            overflow.eq(restore.source.valid & (restore.source.error[:8] != 0)),
            failed.eq(active & (credit_error | conversion_error | overflow)),
            self.sp_rd_req.eq(~self.cancel & ~failed & ((state == META_REQ) |
                ((state == RUN) & ((reserved < 2) | pop)))),
            self.sp_rd_adr.eq(Mux(state == META_REQ, captured["exponents"] >> 5,
                                   (captured["src"] >> 5) + issued)),
            mask.eq(Mux(restore.source.last & (tail != 0), (C(1, 9) << tail) - 1, 255)),
            self.sp_wr_req.eq(active & restore.source.valid & ~failed & ~self.cancel),
            self.sp_wr_adr.eq((captured["dst"] >> 5) + retired),
            self.sp_wr_dat.eq(restore.source.data), self.sp_wr_we.eq(mask),
            restore.source.ready.eq(active & self.sp_wr_gnt & ~failed & ~self.cancel),
            written.eq(self.sp_wr_req & self.sp_wr_gnt),
        ]
        for lane in range(8):
            self.comb += restore.sink.data[32*lane:32*(lane+1)].eq(
                Mux(responses.dout[257] & (tail != 0) & (tail <= lane), 0,
                    responses.dout[32*lane:32*(lane+1)]))
        self.sync += If(clear, read_pending.eq(0)).Else(
            read_pending.eq(self.sp_rd_req),
            If(self.sp_rd_req, read_first.eq(issued == 0), read_last.eq(issued == count - 1)))
        self.sync += If(self.cancel,
            state.eq(IDLE), self.error.eq(0), self.done.eq(0),
            self.exponent.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
        ).Else(
            self.done.eq(0),
            If(self.sp_rd_req, self.read_beats.eq(self.read_beats + 1)),
            If(self.sp_rd_req & active, issued.eq(issued + 1),
                If(issued == count - 1, state.eq(DRAIN))),
            If(written, retired.eq(retired + 1), self.output_beats.eq(self.output_beats + 1),
                If(restore.source.last, state.eq(FINISH))),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    count.eq((self.cmd.elements + C(7, 17)) >> 3),
                    issued.eq(0), retired.eq(0), self.exponent.eq(0),
                    self.read_beats.eq(0), self.output_beats.eq(0), self.error.eq(0), state.eq(VALIDATE)),
                VALIDATE: If(geometry_bad | bounds_bad | overlap_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, overlap_bad, C(0, 3), protected_bad)),
                    state.eq(FAILED),
                ).Else(state.eq(META_REQ)),
                META_REQ: state.eq(META_CAPTURE),
                META_CAPTURE: [metadata.eq(self.sp_rd_dat[:32]), state.eq(META_CHECK)],
                META_CHECK: If(exponent_bad, self.error.eq(8), state.eq(FAILED)).Else(
                    self.exponent.eq(combined), state.eq(RUN)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            }),
            # Numerical/credit failure wins over any same-cycle retirement.
            If(failed, self.error.eq(Cat(C(0, 4), overflow, credit_error | conversion_error)), state.eq(FAILED)),
        )

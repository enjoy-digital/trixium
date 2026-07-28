#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiEngineCSR: CSR front-end for the matvec engine (hybrid CPU/host control).

The phase-1 sequencing model: software (VexRiscv firmware over the SoC bus, or the host over
the PCIe BAR0 bridge - the exact same registers) programs one op at a time and polls `done`.
Sequencing cost is ~10 CSR writes + a poll per op, ~500 ops/token: <0.5% of the weight
streaming time, so a hardware descriptor sequencer is a phase-2 optimization, not a need.

Ops: kind 0 = ternary matvec, 1 = q8 matvec, 2 = fetch (HBM row -> scratchpad; also the
upload-verification readback path)."""

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

# BonsaiEngineCSR ----------------------------------------------------------------------------------

class BonsaiEngineCSR(LiteXModule):
    def __init__(self, engine):
        self.kind    = CSRStorage(2,  description="Op kind: 0 ternary, 1 q8, 2 fetch.")
        self.rows    = CSRStorage(16, description="Rows per channel.")
        self.cols    = CSRStorage(14, description="Columns (K).")
        self.wbase   = CSRStorage(32, description="Per-channel weight byte address.")
        self.wlen    = CSRStorage(24, description="Per-channel weight length in beats.")
        self.sbeats  = CSRStorage(4,  description="Scale beats per row.")
        self.cbeats  = CSRStorage(8,  description="Code beats per row.")
        self.bias    = CSRStorage(7,  description="Accumulator scale bias B.")
        self.rshift  = CSRStorage(6,  description="Result shift R.")
        self.xsrc    = CSRStorage(24, description="x8 source scratchpad byte address.")
        self.dst     = CSRStorage(24, description="Result destination scratchpad byte address.")
        self.chan    = CSRStorage(5,  description="Fetch source channel.")
        self.submit  = CSRStorage(1,  description="Write 1 to submit the op.")
        self.status  = CSRStatus(fields=[
            CSRField("busy",     size=1),
            CSRField("error",    size=1, description="Sticky reserved-code error."),
            CSRField("pending",  size=1, description="Submitted op not yet accepted."),
        ])
        self.done_count = CSRStatus(32, description="Completed op counter.")
        self.absmax     = CSRStatus(32, description="|int32| max over the last matvec's results.")

        # Perf counters, muxed by perf_chan.
        self.perf_chan  = CSRStorage(5)
        self.perf_beats = CSRStatus(32, description="Beats delivered by selected reader.")
        self.perf_ar    = CSRStatus(32, description="AR stall cycles of selected reader.")
        self.perf_r     = CSRStatus(32, description="R stall cycles of selected reader.")

        # # #

        pending = Signal()
        # Accept beats submit: when the engine is idle it accepts the command on the submit
        # cycle itself, and latching pending anyway would replay the op once the engine
        # returns to idle (results are rewritten identically, but the replay's command
        # accept resets the absmax tracker under the firmware's feet - and doubles the
        # HBM stream time of every op).
        self.sync += [
            If(engine.cmd.valid & engine.cmd.ready,
                pending.eq(0),
            ).Elif(self.submit.re & self.submit.storage,
                pending.eq(1),
            ),
            If(engine.done,
                self.done_count.status.eq(self.done_count.status + 1),
            ),
        ]
        # `submit.re` fires on CSR write; latch as cmd.valid until accepted.
        self.comb += [
            engine.cmd.valid.eq(pending | (self.submit.re & self.submit.storage)),
            engine.cmd.kind.eq(self.kind.storage),
            engine.cmd.rows.eq(self.rows.storage),
            engine.cmd.cols.eq(self.cols.storage),
            engine.cmd.wbase.eq(self.wbase.storage),
            engine.cmd.wlen.eq(self.wlen.storage),
            engine.cmd.sbeats.eq(self.sbeats.storage),
            engine.cmd.cbeats.eq(self.cbeats.storage),
            engine.cmd.bias.eq(self.bias.storage),
            engine.cmd.rshift.eq(self.rshift.storage),
            engine.cmd.xsrc.eq(self.xsrc.storage),
            engine.cmd.dst.eq(self.dst.storage),
            engine.cmd.chan.eq(self.chan.storage),
            self.status.fields.busy.eq(engine.busy),
            self.status.fields.error.eq(engine.error),
            self.status.fields.pending.eq(pending),
            self.absmax.status.eq(engine.absmax),
        ]

        beats = Array([r.total_beats for r in engine.readers])
        ars   = Array([r.ar_stalls   for r in engine.readers])
        rs    = Array([r.r_stalls    for r in engine.readers])
        self.comb += [
            self.perf_beats.status.eq(beats[self.perf_chan.storage]),
            self.perf_ar.status.eq(ars[self.perf_chan.storage]),
            self.perf_r.status.eq(rs[self.perf_chan.storage]),
        ]

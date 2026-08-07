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
    def __init__(self, engine, seq=None):
        self.kind    = CSRStorage(3,  description="Op kind: 0 ternary, 1 q8, 2 fetch, 3 store, 4 axpy.")
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
        self.chan    = CSRStorage(5,  description="Fetch source / store destination channel.")
        self.rem     = CSRStorage(5,  description="Channels < rem stream one extra row.")
        self.ilv     = CSRStorage(1,  description="Interleaved (position-major) gather.")
        self.supw    = CSRStorage(1,  description="Suppress scratchpad result writes (argmax-only ops).")
        self.qn      = CSRStorage(1,  description="Dual-Q GQA mode (2 heads per K/V stream).")
        self.xred    = CSRStorage(1,  description="AXPY: sum the per-channel partials in hardware.")
        self.qstride = CSRStorage(16, description="Head-B result offset in words.")
        self.rshift2 = CSRStorage(6,  description="Head-B AXPY pre-shift.")
        self.submit  = CSRStorage(1,  description="Write 1 to submit the op.")
        self.status  = CSRStatus(fields=[
            CSRField("busy",     size=1),
            CSRField("error",    size=1, description="Sticky reserved-code error."),
            CSRField("pending",  size=1, description="Submitted op not yet accepted."),
            # A wedged op is only diagnosable if it says where it is wedged. On hardware a KV
            # store left the engine busy forever and all the CSRs could report was "busy": no
            # way to tell ST_CMD (loader will not take the command) from ST_RUN (scratchpad is
            # not delivering beats) from ST_DRAIN (writer's B responses never came), which are
            # three different bugs. The FSM state costs four bits and settles it.
            CSRField("st_cmd",   size=1, offset=4, description="STORE: loader has not taken the command."),
            CSRField("st_run",   size=1, offset=5, description="STORE: streaming beats from the scratchpad."),
            CSRField("st_drain", size=1, offset=6, description="STORE: waiting for the writer's B responses."),
            CSRField("fill",     size=1, offset=7, description="Broadcasting x into the pipes."),
            CSRField("run",      size=1, offset=8, description="Streaming weights."),
            CSRField("sp_grant", size=1, offset=9, description="Engine holds the scratchpad."),
            CSRField("st_busy",  size=1, offset=10, description="Loader is running our store."),
            CSRField("vec_busy", size=1, offset=11, description="Vector unit busy."),
        ])
        self.done_count = CSRStatus(32, description="Completed op counter.")
        self.absmax     = CSRStatus(32, description="|int32| max over the last matvec's results.")
        self.argmax_val = CSRStatus(32, description="Max signed result of the last matvec.")
        self.argmax_idx = CSRStatus(24, description="Global row index of argmax (ties: smallest).")

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
        # `submit.re` fires on CSR write; latch as cmd.valid until accepted. While the
        # sequencer runs it owns the command interface (firmware never mixes the two).
        seq_own = Signal()
        if seq is not None:
            self.comb += [
                seq_own.eq(seq.busy),
                seq.eng_ready.eq(engine.cmd.ready & seq_own),
                seq.eng_done.eq(engine.done),
                seq.absmax.eq(engine.absmax),
            ]
        self.comb += [
            If(seq_own,
                engine.cmd.valid.eq(0 if seq is None else seq.eng_valid),
                engine.cmd.kind.eq(0 if seq is None else seq.eng_kind),
                engine.cmd.rows.eq(0 if seq is None else seq.eng_rows),
                engine.cmd.cols.eq(0 if seq is None else seq.eng_cols),
                engine.cmd.wbase.eq(0 if seq is None else seq.eng_wbase),
                engine.cmd.wlen.eq(0 if seq is None else seq.eng_wlen),
                engine.cmd.sbeats.eq(0 if seq is None else seq.eng_sbeats),
                engine.cmd.cbeats.eq(0 if seq is None else seq.eng_cbeats),
                engine.cmd.bias.eq(0 if seq is None else seq.eng_bias),
                engine.cmd.rshift.eq(0 if seq is None else seq.eng_rshift),
                engine.cmd.xsrc.eq(0 if seq is None else seq.eng_xsrc),
                engine.cmd.dst.eq(0 if seq is None else seq.eng_dst),
                engine.cmd.chan.eq(0 if seq is None else seq.eng_chan),
                engine.cmd.rem.eq(0 if seq is None else seq.eng_rem),
                engine.cmd.ilv.eq(0 if seq is None else seq.eng_ilv),
                engine.cmd.supw.eq(0 if seq is None else seq.eng_supw),
                engine.cmd.qn.eq(0 if seq is None else seq.eng_qn),
                engine.cmd.xred.eq(0 if seq is None else seq.eng_xred),
                engine.cmd.qstride.eq(0 if seq is None else seq.eng_qstride),
                engine.cmd.rshift2.eq(0 if seq is None else seq.eng_rshift2),
            ).Else(
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
            engine.cmd.rem.eq(self.rem.storage),
            engine.cmd.ilv.eq(self.ilv.storage),
            engine.cmd.supw.eq(self.supw.storage),
            engine.cmd.qn.eq(self.qn.storage),
            engine.cmd.xred.eq(self.xred.storage),
            engine.cmd.qstride.eq(self.qstride.storage),
            engine.cmd.rshift2.eq(self.rshift2.storage),
            ),
            self.status.fields.busy.eq(engine.busy),
            self.status.fields.error.eq(engine.error),
            self.status.fields.pending.eq(pending),
            self.status.fields.st_cmd.eq(engine.fsm.ongoing("ST_CMD")),
            self.status.fields.st_run.eq(engine.fsm.ongoing("ST_RUN")),
            self.status.fields.st_drain.eq(engine.fsm.ongoing("ST_DRAIN")),
            self.status.fields.fill.eq(engine.fsm.ongoing("FILL")),
            self.status.fields.run.eq(engine.fsm.ongoing("RUN")),
            self.status.fields.sp_grant.eq(engine.sp_busy),
            self.status.fields.st_busy.eq(engine.store_busy),
            self.status.fields.vec_busy.eq(getattr(engine, "vec_busy_in", 0)),
            self.absmax.status.eq(engine.absmax),
            self.argmax_val.status.eq(engine.argmax_val),
            self.argmax_idx.status.eq(engine.argmax_idx),
        ]

        beats = Array([r.total_beats for r in engine.readers])
        ars   = Array([r.ar_stalls   for r in engine.readers])
        rs    = Array([r.r_stalls    for r in engine.readers])
        self.comb += [
            self.perf_beats.status.eq(beats[self.perf_chan.storage]),
            self.perf_ar.status.eq(ars[self.perf_chan.storage]),
            self.perf_r.status.eq(rs[self.perf_chan.storage]),
        ]

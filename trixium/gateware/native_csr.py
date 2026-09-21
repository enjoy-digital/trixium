# SPDX-License-Identifier: BSD-2-Clause
"""Append-only native controls and exclusive manual/sequencer configuration mux."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRField, CSRStatus, CSRStorage


class BonsaiNativeCSR(LiteXModule):
    """Install as zznative so existing CSR banks retain their addresses.

    mode selects native for manual engine submissions only. Legacy sequencer
    commands remain legacy; sequencer unit 7 selects native independently.
    Native K is derived from cbeats (128/32 elements per
    group) so the existing 14-bit cols CSR is not widened or reinterpreted in
    legacy mode. Program one descriptor per active compute pipe before submit.
    One pending descriptor is snapshotted at commit; a second commit while full
    is rejected and sets sticky queue_error. The common submit waits for it.
    With native sequencing, older queued manual cfg/commands drain before list
    fetch. New commits/submits during list ownership are rejected (queue_error),
    never replayed afterwards. accepted counts manual cfg handshakes only.
    """
    def __init__(self, engine, engine_csr, seq=None):
        self.abi = CSRStatus(32, reset=0x00010000)
        self.mode = CSRStorage(1, description='Native PTQ/Q8 for manual matvec; sequencer selects mode per descriptor.')
        self.chan = CSRStorage(5, description='Compute pipe receiving this region descriptor.')
        self.region = CSRStorage(32)
        self.offset = CSRStorage(64, description='Native first selected block: region-relative byte offset.')
        self.stride = CSRStorage(64, description='Byte distance between selected native rows.')
        self.commit = CSRStorage(1, description='Snapshot and queue one descriptor. Wait for accepted/status.')
        self.clear_error = CSRStorage(1, description='Clear queue error; does not reset a failed engine.')
        self.status = CSRStatus(fields=[
            CSRField('pending', size=1), CSRField('queue_error', size=1),
            CSRField('config_error', size=1), CSRField('failed', size=1),
        ])
        self.accepted = CSRStatus(32, description='Descriptor handshakes since system reset; wraps at 32 bits.')
        self.configured = CSRStatus(32, description='Pipes with committed descriptors; engine reset clears these.')
        self.reader_busy = CSRStatus(32)
        self.reader_error = CSRStatus(32)
        self.pipe_error = CSRStatus(32)
        pending, queue_error = Signal(), Signal()
        captured = {name: Signal.like(getattr(engine.native_cfg, name))
                    for name in ('chan', 'region', 'offset', 'stride')}
        resetting = engine_csr.rst_trigger | engine_csr.rst_busy
        accepted = Signal()
        owned = C(0, 1)
        if seq is not None and hasattr(seq, 'native_cfg'):
            owned = seq.busy | (seq.start.re & seq.start.storage)
            use_seq = seq.busy & ~seq.native_manual_drain
            available = ~resetting & ~engine.sp_busy & ~engine.vec_busy_in
            self.comb += [
                engine.native_cfg.valid.eq(Mux(use_seq, seq.native_cfg.valid & available, pending & ~resetting)),
                seq.native_cfg.ready.eq(engine.native_cfg.ready & use_seq & available),
                *[getattr(engine.native_cfg, name).eq(Mux(use_seq, getattr(seq.native_cfg, name), signal))
                  for name, signal in captured.items()],
                seq.native_cfg_error.eq(engine.native_cfg_error), seq.native_failed.eq(engine.native_failed),
                seq.native_manual_pending.eq(pending | engine_csr.status.fields.pending),
                seq.native_engine_busy.eq(engine.sp_busy | engine.vec_busy_in),
                accepted.eq(engine.native_cfg.valid & engine.native_cfg.ready & ~use_seq),
            ]
        else:
            self.comb += [engine.native_cfg.valid.eq(pending & ~resetting),
                          accepted.eq(engine.native_cfg.valid & engine.native_cfg.ready),
                          *[getattr(engine.native_cfg, name).eq(signal) for name, signal in captured.items()]]
        self.comb += [
            engine_csr.native_enable.eq(self.mode.storage),
            engine_csr.native_config_pending.eq(pending),
            self.status.fields.pending.eq(pending), self.status.fields.queue_error.eq(queue_error),
            self.status.fields.config_error.eq(engine.native_cfg_error),
            self.status.fields.failed.eq(engine.native_failed),
            self.configured.status.eq(engine.native_cfg_valid),
            self.reader_busy.status.eq(Cat(*[r.busy for r in engine.native_readers])),
            self.reader_error.status.eq(Cat(*[r.error != 0 for r in engine.native_readers])),
            self.pipe_error.status.eq(Cat(*[p.error | p.overflow for p in engine.pipes])),
        ]
        self.sync += [
            If(self.clear_error.re & self.clear_error.storage, queue_error.eq(0)),
            If(getattr(engine_csr, 'native_manual_conflict', C(0, 1)), queue_error.eq(1)),
            If(resetting,
                pending.eq(0),
            ).Else(
                If(accepted,
                    pending.eq(0), self.accepted.status.eq(self.accepted.status + 1),
                ),
                If(self.commit.re & self.commit.storage,
                    If(pending | owned,
                        queue_error.eq(1),
                    ).Else(
                        pending.eq(1),
                        *[signal.eq(getattr(self, name).storage) for name, signal in captured.items()],
                    ),
                ),
            ),
        ]

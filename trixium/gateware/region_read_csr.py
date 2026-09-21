# SPDX-License-Identifier: BSD-2-Clause
"""Append-only control of bounded region readback into the shared scratchpad."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRField, CSRStatus, CSRStorage


class BonsaiRegionReadCSR(LiteXModule):
    """Install as zzverify (after zzregion); optional peek status adds ABI 1.1.

    Completion makes a scratchpad span readable, not an uploaded model valid.
    Software must compare/hash it before publishing the allocation. Failure or
    timeout requires the common full-cluster reset/drain, never a local flush.
    """
    def __init__(self, engine, engine_csr, *, scratchpad_csr=None, seq=None):
        self.abi = CSRStatus(32, reset=0x00010001 if scratchpad_csr is not None else 0x00010000)
        self.region = CSRStorage(32)
        self.offset = CSRStorage(64)
        self.length = CSRStorage(24, description='Nonzero number of full 32-byte beats.')
        self.dst = CSRStorage(24, description='32-byte-aligned scratchpad byte destination.')
        self.start = CSRStorage(1, description='Snapshot one readback; busy/failed start is rejected.')
        self.status = CSRStatus(fields=[CSRField('busy', size=1), CSRField('pending', size=1),
                                       CSRField('failed', size=1)])
        self.error = CSRStatus(16, description='Low byte: transport; bit 8: start misuse.')
        self.beats = CSRStatus(24, description='Scratchpad beats written, not transfer success.')
        self.accepted = CSRStatus(32)
        self.completed = CSRStatus(32, description='Successful readbacks only; wraps at 32 bits.')
        active, pending, local_error, failed, fire = (Signal() for _ in range(5))
        resetting = engine_csr.rst_trigger | engine_csr.rst_busy
        if scratchpad_csr is not None:
            self.sp_pending = CSRStatus(1, description='Scratchpad CSR read pending/capturing; serialize peek requests.')
            self.sp_completed = CSRStatus(32, description='Scratchpad CSR reads captured; persists across engine reset.')
            self.comb += [self.sp_pending.status.eq(scratchpad_csr.read_pending),
                          self.sp_completed.status.eq(scratchpad_csr.read_completed),
                          scratchpad_csr.read_reset.eq(resetting)]
        captured = {name: Signal.like(getattr(engine.region_read_cmd, name))
                    for name in ('region', 'offset', 'length', 'dst')}
        start = self.start.re & self.start.storage
        owned, use_seq = C(0, 1), C(0, 1)
        if seq is not None:
            if not hasattr(seq, 'region_fetch_cmd'):
                raise ValueError('sequenced read control needs the region-fetch interface')
            owned = seq.busy | (seq.start.re & seq.start.storage)
            use_seq = seq.busy & ~seq.native_manual_drain
            self.comb += [
                seq.region_fetch_cmd.ready.eq(engine.region_read_cmd.ready & use_seq & ~resetting),
                seq.region_fetch_done.eq(engine.region_read_done & use_seq),
                seq.region_fetch_failed.eq(Cat(engine.region_read_error, engine.region_read_failed)),
                seq.region_fetch_manual_pending.eq(active | pending),
                # A rejected NEW start must not poison the owning sequence.
                # Only a failed older active/queued read blocks SBDRAIN.
                seq.region_fetch_manual_failed.eq(failed & (active | pending)),
            ]
        misuse = start & (active | failed | owned)
        self.comb += [
            failed.eq(local_error | (engine.region_read_error != 0) | engine.region_read_failed),
            engine.region_read_cmd.valid.eq(Mux(use_seq,
                C(0, 1) if seq is None else seq.region_fetch_cmd.valid, pending & ~failed) & ~resetting),
            *[getattr(engine.region_read_cmd, name).eq(Mux(use_seq,
                value if seq is None else getattr(seq.region_fetch_cmd, name), value))
              for name, value in captured.items()],
            fire.eq(engine.region_read_cmd.valid & engine.region_read_cmd.ready & ~use_seq),
            self.status.fields.busy.eq(active | engine.region_read_busy),
            self.status.fields.pending.eq(pending), self.status.fields.failed.eq(failed),
            self.error.status.eq(Cat(engine.region_read_error, local_error)),
            self.beats.status.eq(engine.region_read_beats),
        ]
        self.sync += If(resetting,
            active.eq(0), pending.eq(0), local_error.eq(0),
        ).Else(
            If(fire, pending.eq(0), self.accepted.status.eq(self.accepted.status+1)),
            If(engine.region_read_done & active & ~use_seq & ~failed & ~misuse,
                active.eq(0), self.completed.status.eq(self.completed.status+1)),
            If(start,
                If(active | failed | owned, local_error.eq(1)).Else(
                    active.eq(1), pending.eq(1),
                    *[value.eq(getattr(self, name).storage) for name, value in captured.items()],
                )),
        )

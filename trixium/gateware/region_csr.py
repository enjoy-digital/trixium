# SPDX-License-Identifier: BSD-2-Clause
"""Append-only CSR upload source for the bounded region-write transport."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRField, CSRStatus, CSRStorage


class BonsaiRegionUploadCSR(LiteXModule):
    """Install as zzregion, after zznative. Only full 32-byte beats are uploaded.

    This does not reuse the legacy DMA ring or its sync-marker heuristic. A
    credits register bounds CSR writes, and framing is derived from an exact
    word count. Errors invalidate the whole transfer and require cluster reset.
    Host/runtime must own the device exclusively while loading model weights.
    """
    def __init__(self, loader, engine_csr, *, fifo_depth=32):
        if not 2 <= fifo_depth <= 256:
            raise ValueError('region upload FIFO must have 2..256 beats')
        self.abi = CSRStatus(32, reset=0x00010000)
        self.region = CSRStorage(32)
        self.offset = CSRStorage(64)
        self.length = CSRStorage(24, description='Nonzero count of full 32-byte beats.')
        self.start = CSRStorage(1, description='Snapshot one upload command; reject when busy/failed.')
        self.poke = CSRStorage(32, description='Next little-endian payload word; obey space and exact length.')
        self.status = CSRStatus(fields=[CSRField('busy', size=1), CSRField('pending', size=1),
                                       CSRField('failed', size=1)])
        self.error = CSRStatus(16, description='Low byte: transport; bits 8/9: start misuse / payload overrun.')
        self.space = CSRStatus(16, description='Conservative free payload-word credits, bounded by words remaining.')
        self.words = CSRStatus(27, description='Accepted payload words in the current transfer.')
        self.beats = CSRStatus(24, description='Payload beats accepted by the transport, not memory acknowledgements.')
        self.accepted = CSRStatus(32)
        self.completed = CSRStatus(32, description='Successful transfers only; wraps at 32 bits.')

        self.fifo = fifo = ResetInserter()(stream.SyncFIFO([('data', 256)], fifo_depth))
        resetting = engine_csr.rst_trigger | engine_csr.rst_busy
        active, pending = Signal(), Signal()
        local_error = Signal(2)
        captured = {name: Signal.like(getattr(loader.region_cmd, name)) for name in ('region', 'offset', 'length')}
        word_limit = Signal(27)
        packed, partial = Signal(256), Signal(3)
        available = Signal(16)
        remaining = Signal(27)
        failed, command_fire, poke_ok = Signal(), Signal(), Signal()
        self.comb += [
            fifo.reset.eq(resetting),
            failed.eq((local_error != 0) | (loader.region_error != 0) | loader.region_failed),
            loader.region_abort.eq(local_error != 0),
            loader.region_cmd.valid.eq(pending & ~failed & ~resetting),
            *[getattr(loader.region_cmd, name).eq(value) for name, value in captured.items()],
            command_fire.eq(loader.region_cmd.valid & loader.region_cmd.ready),
            fifo.source.connect(loader.region_data),
            available.eq((fifo_depth-fifo.level-fifo.sink.valid)*8-partial),
            remaining.eq(word_limit-self.words.status),
            self.space.status.eq(Mux(active & ~failed & ~resetting,
                Mux(remaining < available, remaining, available), 0)),
            poke_ok.eq(self.space.status != 0),
            self.status.fields.busy.eq(active | loader.region_busy),
            self.status.fields.pending.eq(pending), self.status.fields.failed.eq(failed),
            self.error.status.eq(Cat(loader.region_error, local_error)),
            self.beats.status.eq(loader.region_beats),
        ]
        self.sync += [
            fifo.sink.valid.eq(0),
            If(resetting,
                active.eq(0), pending.eq(0), partial.eq(0), local_error.eq(0),
                packed.eq(0), word_limit.eq(0), self.words.status.eq(0),
            ).Else(
                If(fifo.sink.valid & ~fifo.sink.ready, local_error[1].eq(1)),
                If(command_fire, pending.eq(0), self.accepted.status.eq(self.accepted.status+1)),
                If(loader.region_done & ~failed,
                    active.eq(0), self.completed.status.eq(self.completed.status+1)),
                If(self.start.re & self.start.storage,
                    If(active | failed,
                        local_error[0].eq(1),
                    ).Else(
                        active.eq(1), pending.eq(1), self.words.status.eq(0),
                        word_limit.eq(self.length.storage << 3), partial.eq(0), packed.eq(0),
                        *[value.eq(getattr(self, name).storage) for name, value in captured.items()],
                    ),
                ),
                If(self.poke.re,
                    If(~poke_ok,
                        local_error[1].eq(1),
                    ).Else(
                        self.words.status.eq(self.words.status+1), partial.eq(partial+1),
                        packed.eq(Cat(packed[32:], self.poke.storage)),
                        If(partial == 7,
                            fifo.sink.valid.eq(1), fifo.sink.data.eq(Cat(packed[32:], self.poke.storage)),
                            fifo.sink.last.eq(self.words.status == word_limit-1),
                        ),
                    ),
                ),
            ),
        ]

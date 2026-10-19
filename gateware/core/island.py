#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""PipeIsland: a channel's unpacker and pipe in their own, faster clock domain.

The channel pipe consumes one 128-trit beat per clock; with trit5 weights (128 trits in 26 of
a 32-byte memory beat) the memory only has to deliver 0.81 memory beats per pipe clock, so at
the engine's clock the pipes are the limit and the memory idles. Clocking the unpacker and the
pipe faster (`cd`, a domain the SoC derives from a PLL, with the engine's reset) moves the
limit back to the memory port: at 1.25x the engine clock a trit5 stream needs 1.02 memory
beats per engine clock, the port's rate. The unpacker must be in the island: it emits one
pipe beat per clock of its own domain, and in the engine's domain it would cap the island's
input at the old rate.

The island keeps the engine-side interface of ChannelPipe (and of Trit5Unpacker as `unp`):
- memory beats in and results out cross in async FIFOs (the engine's beat gating and the
  result gather stay where they are);
- `start` crosses in a pulse synchronizer and starts the unpacker and the pipe together (the
  engine pulses both with the same expression); the op parameters (kind, rows, beats per
  row, bias, shifts, qn, packed) are quasi-static - written at the op's accept, read from
  `start` on - so they are wired through and settle during the synchronizer's latency;
- the x-buffer's write port stays in the engine's domain (a dual-clock memory): the fill
  completes before the first beat is issued;
- the sticky flags come back through MultiRegs.
"""

from migen import *
from migen.genlib.cdc import PulseSynchronizer, MultiReg

from litex.gen import *
from litex.soc.interconnect import stream

from gateware.core.pipe import ChannelPipe
from gateware.core.trit5 import Trit5Unpacker

# PipeIsland ---------------------------------------------------------------------------------------

class _UnpackerPorts:
    """The engine's view of the island's unpacker (Trit5Unpacker's parameter signals)."""
    def __init__(self):
        self.packed      = Signal()
        self.scale_beats = Signal(4)
        self.code_beats  = Signal(8)
        self.start       = Signal()   # Driven by the engine with the pipe's start; the island
                                      # starts its unpacker from the pipe's synchronized pulse.

class PipeIsland(LiteXModule):
    def __init__(self, cd, trit5=False, in_depth=32, out_depth=16, in_cd="sys", **pipe_kwargs):
        # in_cd: the domain the memory beats arrive in (`sink`): the engine's, or the readers'
        # when they run in the memory's clock (ReaderIsland).
        pipe = ChannelPipe(xw_cd="xw", **pipe_kwargs)
        # The engine's view (its own clock domain): the same signals as ChannelPipe's.
        self.kind        = Signal.like(pipe.kind)
        self.rows        = Signal.like(pipe.rows)
        self.scale_beats = Signal.like(pipe.scale_beats)
        self.code_beats  = Signal.like(pipe.code_beats)
        self.bias        = Signal.like(pipe.bias)
        self.rshift      = Signal.like(pipe.rshift)
        self.qn          = Signal.like(pipe.qn)
        self.rshift2     = Signal.like(pipe.rshift2)
        self.start       = Signal()
        self.sink        = stream.Endpoint(pipe.sink.description)   # Memory beats (trit5: packed).
        self.source      = stream.Endpoint(pipe.source.description)
        self.xw_group    = Signal.like(pipe.xw_group)
        self.xw_sub      = Signal.like(pipe.xw_sub)
        self.xw_data     = Signal.like(pipe.xw_data)
        self.xw_we       = Signal()
        self.error       = Signal()
        self.overflow    = Signal()
        self.axpy        = pipe.axpy
        self.trit5       = trit5
        self.unp         = _UnpackerPorts() if trit5 else None

        # # #

        # The pipe in `cd`; its x-buffer write port ("xw") back in the engine's domain.
        self.pipe = ClockDomainsRenamer({"sys": cd, "xw": "sys"})(pipe)
        # Quasi-static parameters and the x-buffer writes: wired through.
        self.comb += [
            pipe.kind.eq(self.kind),
            pipe.rows.eq(self.rows),
            pipe.scale_beats.eq(self.scale_beats),
            pipe.code_beats.eq(self.code_beats),
            pipe.bias.eq(self.bias),
            pipe.rshift.eq(self.rshift),
            pipe.qn.eq(self.qn),
            pipe.rshift2.eq(self.rshift2),
            pipe.xw_group.eq(self.xw_group),
            pipe.xw_sub.eq(self.xw_sub),
            pipe.xw_data.eq(self.xw_data),
            pipe.xw_we.eq(self.xw_we),
        ]
        # start: one pulse per op, synchronized.
        self.start_sync = ps = PulseSynchronizer("sys", cd)
        self.comb += [
            ps.i.eq(self.start),
            pipe.start.eq(ps.o),
        ]
        # Beats in (memory beats), through the unpacker when the build has it, results out.
        self.fifo_in  = fi = ClockDomainsRenamer({"write": in_cd, "read": cd})(
            stream.AsyncFIFO(pipe.sink.description, in_depth))
        self.fifo_out = fo = ClockDomainsRenamer({"write": cd, "read": "sys"})(
            stream.AsyncFIFO(pipe.source.description, out_depth))
        self.comb += self.sink.connect(fi.sink)
        if trit5:
            unp = Trit5Unpacker()
            self.unpacker = ClockDomainsRenamer(cd)(unp)
            self.comb += [
                unp.packed.eq(self.unp.packed),
                unp.scale_beats.eq(self.unp.scale_beats),
                unp.code_beats.eq(self.unp.code_beats),
                unp.start.eq(ps.o),
                fi.source.connect(unp.sink),
                unp.source.connect(pipe.sink),
            ]
        else:
            self.comb += fi.source.connect(pipe.sink)
        self.comb += [
            pipe.source.connect(fo.sink),
            fo.source.connect(self.source),
        ]
        # Sticky flags.
        self.specials += [
            MultiReg(pipe.error,    self.error,    odomain="sys"),
            MultiReg(pipe.overflow, self.overflow, odomain="sys"),
        ]

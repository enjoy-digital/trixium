#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiMatVecEngine: N-channel ternary/Q8 matvec engine.

One BonsaiAXIReader + BonsaiChannelPipe per HBM pseudo-channel. A matvec command:

  1. FILL : broadcast x8 (cols bytes) from the scratchpad into every channel's x-buffer.
  2. ISSUE: give every reader the same {wbase, wlen} (identical relative layout per channel).
  3. RUN  : pipes stream rows; the gather round-robins int32 results into the scratchpad at
            dst + (channel*rows + row)*4 bytes; done when rows*n_channels results landed.

The command uses per-channel row counts (rows = M/n_channels) and byte addresses inside the
per-channel HBM address space. Sequenced by the descriptor engine (or a CSR shim/testbench)."""

from migen import *
from migen.genlib.roundrobin import RoundRobin, SP_CE

from litex.gen import *

from litex.soc.interconnect import stream

from trixium.layout import BEAT_BYTES
from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.ternary_mac import BonsaiChannelPipe

# BonsaiMatVecEngine -------------------------------------------------------------------------------

class BonsaiMatVecEngine(LiteXModule):
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96):
        self.n_channels = n = len(hbm_axis)

        self.cmd = stream.Endpoint([
            ("kind",   2),   # 0: ternary, 1: q8.
            ("rows",   16),  # Rows per channel.
            ("cols",   14),  # K.
            ("wbase",  32),  # Per-channel weight byte address (beat-aligned).
            ("wlen",   24),  # Per-channel weight length in beats.
            ("sbeats", 4),   # Scale beats per row.
            ("cbeats", 8),   # Code beats per row.
            ("bias",   7),   # B.
            ("rshift", 6),   # R.
            ("xsrc",   24),  # x8 source byte address in scratchpad (32B aligned).
            ("dst",    24),  # Result destination byte address in scratchpad (4B aligned).
        ])
        self.busy  = Signal()
        self.done  = Signal() # Pulse.
        self.error = Signal()

        # # #

        # Channels ----------------------------------------------------------------------------
        self.readers = []
        self.pipes   = []
        for i, axi in enumerate(hbm_axis):
            reader = BonsaiAXIReader(axi)
            pipe   = BonsaiChannelPipe(xbuf_groups=xbuf_groups)
            setattr(self.submodules, f"reader{i}", reader)
            setattr(self.submodules, f"pipe{i}",   pipe)
            self.comb += reader.source.connect(pipe.sink, omit={"last"})
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command ---------------------------------------------------------------------
        cmd_r = {}
        for name in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                     "bias", "rshift", "xsrc", "dst"):
            cmd_r[name] = Signal.like(getattr(self.cmd, name))
        self.sync += If(self.cmd.valid & self.cmd.ready,
            *[cmd_r[k].eq(getattr(self.cmd, k)) for k in cmd_r])

        # Broadcast op params to pipes.
        for pipe in self.pipes:
            self.comb += [
                pipe.kind.eq(cmd_r["kind"]),
                pipe.rows.eq(cmd_r["rows"]),
                pipe.scale_beats.eq(cmd_r["sbeats"]),
                pipe.code_beats.eq(cmd_r["cbeats"]),
                pipe.bias.eq(cmd_r["bias"]),
                pipe.rshift.eq(cmd_r["rshift"]),
            ]
        self.comb += self.error.eq(reduce_or([p.error for p in self.pipes]))

        # x-fill ------------------------------------------------------------------------------
        fill_beat  = Signal(10) # Beat index within x fill (cols/32 <= 384).
        fill_total = Signal(10)
        self.comb += fill_total.eq(cmd_r["cols"] >> 5)

        # Registered fill pipeline: address at T, data+write at T+1.
        fill_active_d = Signal()
        fill_beat_d   = Signal(10)
        self.comb += scratchpad.fill.adr.eq((cmd_r["xsrc"] >> 5) + fill_beat)
        for pipe in self.pipes:
            self.comb += [
                pipe.xw_group.eq(fill_beat_d[2:]),
                pipe.xw_sub.eq(fill_beat_d[:2]),
                pipe.xw_data.eq(scratchpad.fill.dat_r),
                pipe.xw_we.eq(fill_active_d),
            ]

        # Gather ------------------------------------------------------------------------------
        self.rr = rr = RoundRobin(n, SP_CE)
        result_cnt  = Signal(32)
        grant_valid = Array([p.source.valid for p in self.pipes])[rr.grant]
        grant_row   = Array([p.source.row   for p in self.pipes])[rr.grant]
        grant_data  = Array([p.source.data  for p in self.pipes])[rr.grant]

        self.comb += [
            rr.request.eq(Cat(*[p.source.valid for p in self.pipes])),
            rr.ce.eq(1),
        ]
        for i, pipe in enumerate(self.pipes):
            self.comb += pipe.source.ready.eq(rr.grant == i)
        # Write granted result: scratchpad int32 slot = dst/4 + grant*rows + row.
        slot = Signal(24)
        self.comb += [
            slot.eq((cmd_r["dst"] >> 2) + rr.grant*cmd_r["rows"] + grant_row),
            scratchpad.gather.adr.eq(slot[3:]),
            scratchpad.gather.dat_w.eq(Replicate(grant_data, 8)),
            scratchpad.gather.we.eq(Mux(grant_valid, 1 << slot[:3], 0)),
        ]

        # Control FSM -------------------------------------------------------------------------
        self.fsm = fsm = FSM(reset_state="IDLE")
        start_pulse = Signal()
        for pipe in self.pipes:
            self.comb += pipe.start.eq(start_pulse)

        fsm.act("IDLE",
            self.cmd.ready.eq(1),
            If(self.cmd.valid,
                NextValue(fill_beat, 0),
                NextValue(result_cnt, 0),
                NextState("START"),
            )
        )
        fsm.act("START",
            start_pulse.eq(1),
            NextState("FILL"),
        )
        fsm.act("FILL",
            self.busy.eq(1),
            NextValue(fill_beat, fill_beat + 1),
            If(fill_beat == fill_total - 1,
                NextState("FILL_LAST"),
            )
        )
        fsm.act("FILL_LAST", # Drain the registered fill write.
            self.busy.eq(1),
            NextState("ISSUE"),
        )
        fsm.act("ISSUE",
            self.busy.eq(1),
            *[r.sink.valid.eq(1) for r in self.readers],
            *[r.sink.addr.eq(cmd_r["wbase"]) for r in self.readers],
            *[r.sink.len.eq(cmd_r["wlen"]) for r in self.readers],
            # All readers are idle here: they accept together.
            If(self.readers[0].sink.ready,
                NextState("RUN"),
            )
        )
        fsm.act("RUN",
            self.busy.eq(1),
            If(result_cnt == (cmd_r["rows"]*self.n_channels),
                self.done.eq(1),
                NextState("IDLE"),
            )
        )

        # Fill write-stage delay + result counting.
        self.sync += [
            fill_active_d.eq(fsm.ongoing("FILL")),
            fill_beat_d.eq(fill_beat),
            If(start_pulse,
                result_cnt.eq(0),
            ).Elif(scratchpad.gather.we != 0,
                result_cnt.eq(result_cnt + 1),
            ),
        ]

def reduce_or(sigs):
    from functools import reduce
    from operator import or_
    return reduce(or_, sigs)

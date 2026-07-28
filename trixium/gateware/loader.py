#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiHBMLoader: host -> HBM weight upload.

Consumes the LitePCIe DMA (host->device) byte stream, width-converts to 256-bit beats and
writes them to the CSR-selected channel/base. Software sequence per transfer: set chan/base/
length, arm `start`, then DMA exactly length*32 bytes; poll `done`. Upload verification uses
the engine's FETCH op (HBM -> scratchpad) + software CRC - no readback datapath needed here.
A CPU fallback path also exists (channel windows / scratchpad), PCIe-less bring-up only."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *

from trixium.gateware.axi_writer import BonsaiAXIWriter

# BonsaiHBMLoader ----------------------------------------------------------------------------------

class BonsaiHBMLoader(LiteXModule):
    def __init__(self, hbm_axis, data_width=128):
        self.sink = stream.Endpoint([("data", data_width)]) # From PCIe DMA (host->device).

        self.chan   = CSRStorage(5,  description="Destination channel.")
        self.base   = CSRStorage(32, description="Destination byte address (beat-aligned).")
        self.length = CSRStorage(24, description="Transfer length in 256-bit beats.")
        self.start  = CSRStorage(1,  description="Write 1 to arm the transfer.")
        self.status = CSRStatus(fields=[
            CSRField("done",  size=1, description="Transfer complete (idle)."),
            CSRField("beats", size=24, offset=8, description="Beats written so far."),
        ])

        # Engine STORE source (scratchpad -> HBM, KV append): granted only from IDLE, so
        # host uploads and engine stores serialize on the shared per-channel writers.
        # Firmware never overlaps them (weight upload happens before generation).
        self.store_cmd  = stream.Endpoint([("chan", 5), ("addr", 32), ("len", 24)])
        self.store_data = stream.Endpoint([("data", 256)])
        self.store_busy = Signal()

        # # #

        n = len(hbm_axis)

        # Width conversion to 256-bit beats.
        self.conv = conv = stream.Converter(data_width, 256)

        # One writer per channel; command + data demuxed by `chan`.
        self.writers = []
        for i, axi in enumerate(hbm_axis):
            writer = BonsaiAXIWriter(axi)
            setattr(self.submodules, f"writer{i}", writer)
            self.writers.append(writer)

        beats = Signal(24)
        chan  = self.chan.storage

        st_chan = Signal(5)
        st_addr = Signal(32)
        st_len  = Signal(24)
        st_cnt  = Signal(24)

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.start.re & self.start.storage,
                NextValue(beats, 0),
                NextState("CMD"),
            ).Elif(self.store_cmd.valid,
                self.store_cmd.ready.eq(1),
                NextValue(st_chan, self.store_cmd.chan),
                NextValue(st_addr, self.store_cmd.addr),
                NextValue(st_len,  self.store_cmd.len),
                NextValue(st_cnt,  0),
                NextState("EST_CMD"),
            )
        )
        fsm.act("EST_CMD",
            *[w.sink_cmd.valid.eq(st_chan == i) for i, w in enumerate(self.writers)],
            *[w.sink_cmd.addr.eq(st_addr) for w in self.writers],
            *[w.sink_cmd.len.eq(st_len) for w in self.writers],
            If(Array([w.sink_cmd.ready for w in self.writers])[st_chan],
                NextState("EST_DATA"),
            )
        )
        fsm.act("EST_DATA",
            *[w.sink_data.valid.eq((st_chan == i) & self.store_data.valid)
              for i, w in enumerate(self.writers)],
            *[w.sink_data.data.eq(self.store_data.data) for w in self.writers],
            self.store_data.ready.eq(Array([w.sink_data.ready for w in self.writers])[st_chan]),
            If(self.store_data.valid & self.store_data.ready,
                NextValue(st_cnt, st_cnt + 1),
                If(st_cnt == st_len - 1,
                    NextState("EST_DRAIN"),
                )
            )
        )
        fsm.act("EST_DRAIN", # B responses: data readable once idle.
            If(Array([w.idle for w in self.writers])[st_chan],
                NextState("IDLE"),
            )
        )
        self.comb += self.store_busy.eq(fsm.ongoing("EST_CMD") | fsm.ongoing("EST_DATA")
                                        | fsm.ongoing("EST_DRAIN"))
        fsm.act("CMD",
            *[w.sink_cmd.valid.eq(chan == i) for i, w in enumerate(self.writers)],
            *[w.sink_cmd.addr.eq(self.base.storage) for w in self.writers],
            *[w.sink_cmd.len.eq(self.length.storage) for w in self.writers],
            If(Array([w.sink_cmd.ready for w in self.writers])[chan],
                NextState("DATA"),
            )
        )
        fsm.act("DATA",
            *[w.sink_data.valid.eq((chan == i) & conv.source.valid)
              for i, w in enumerate(self.writers)],
            *[w.sink_data.data.eq(conv.source.data) for w in self.writers],
            conv.source.ready.eq(Array([w.sink_data.ready for w in self.writers])[chan]),
            If(conv.source.valid & conv.source.ready,
                NextValue(beats, beats + 1),
                If(beats == self.length.storage - 1,
                    NextState("DRAIN"),
                )
            )
        )
        fsm.act("DRAIN", # Wait for B responses.
            If(Array([w.idle for w in self.writers])[chan],
                NextState("IDLE"),
            )
        )

        self.comb += [
            self.sink.connect(conv.sink, omit={"last"}),
            self.status.fields.done.eq(fsm.ongoing("IDLE")),
            self.status.fields.beats.eq(beats),
        ]

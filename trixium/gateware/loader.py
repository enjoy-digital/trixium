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

# One 256-bit beat the host prefixes to a transfer so the loader can find the start of the
# payload in a stream it does not otherwise control. Repeated 32-bit word so it is trivial to
# emit from C and unlikely to occur in packed weights.
SYNC_MAGIC = int("B0FFA5A5"*8, 16)

# BonsaiHBMLoader ----------------------------------------------------------------------------------

class BonsaiHBMLoader(LiteXModule):
    def __init__(self, hbm_axis, data_width=128):
        self.sink = stream.Endpoint([("data", data_width)]) # From PCIe DMA (host->device).

        self.chan   = CSRStorage(5,  description="Destination channel.")
        self.base   = CSRStorage(32, description="Destination byte address (beat-aligned).")
        self.length = CSRStorage(24, description="Transfer length in 256-bit beats.")
        self.start  = CSRStorage(1,  description="Write 1 to arm the transfer.")
        # NB: not `self.sync` - that name is Migen's synchronous-logic proxy on a Module.
        self.resync = CSRStorage(1,  description=
            "Arm with stream resynchronisation: discard beats until the LOADER_SYNC_MAGIC "
            "beat is seen, then take the payload from the beat after it.")
        # CSR upload path: the host poking words through BAR0 instead of streaming over DMA.
        #
        # The DMA is the fast path but it cannot be flow-controlled - see the resync notes
        # below - so this is the one that is simply always right: the host writes each word
        # itself, so nothing can be dropped, reordered or invented. It is slower (a posted
        # PCIe write per 32 bits) but it needs no marker, no pacing and no luck.
        self.src_sel = CSRStorage(1, description="Upload source: 0 = PCIe DMA, 1 = CSR poke.")
        self.poke    = CSRStorage(32, description="Write a 32-bit word into the upload stream.")

        self.status = CSRStatus(fields=[
            CSRField("done",  size=1, description="Transfer complete (idle)."),
            CSRField("sync",  size=1, offset=1, description="Hunting for the sync marker."),
            CSRField("poke_full", size=1, offset=2, description="CSR poke FIFO cannot accept."),
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

        # CSR poke -> stream: pack 32-bit writes into sink-width words, buffer them, and mux
        # that in place of the DMA when src_sel is set.
        assert data_width % 32 == 0
        poke_ratio = data_width//32
        poke_sr    = Signal(data_width)
        poke_cnt   = Signal(max=max(poke_ratio, 2))
        self.poke_fifo = poke_fifo = stream.SyncFIFO([("data", data_width)], 512)
        self.sync += [
            poke_fifo.sink.valid.eq(0),
            If(self.poke.re,
                poke_sr.eq(Cat(poke_sr[32:], self.poke.storage)),
                poke_cnt.eq(poke_cnt + 1),
                If(poke_cnt == poke_ratio - 1,
                    poke_cnt.eq(0),
                    poke_fifo.sink.valid.eq(1),
                    poke_fifo.sink.data.eq(Cat(poke_sr[32:], self.poke.storage)),
                )
            )
        ]

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

        # Stream resynchronisation.
        #
        # The host->device DMA is free-running and cannot be flow-controlled: the driver
        # programs every descriptor and then enables the reader, so the card is fed the ring's
        # contents whether or not software has written them, and litepcie_write() silently
        # drops the caller's data whenever the hardware laps it. Either way the loader cannot
        # tell payload from filler, and a transfer that starts on the wrong beat writes
        # plausible-looking rubbish to HBM - which is exactly how the staged BONS header and
        # the first megabyte of channel 0 were lost.
        #
        # So let the stream say where it starts. Armed with `sync`, the loader discards beats
        # until it sees SYNC_MAGIC and takes the payload from the beat after it; the host
        # prefixes one marker beat per transfer. Anything the DMA invents beforehand is
        # discarded instead of written.
        sync_hunt = Signal()
        sync_seen = Signal()

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            # Deliberately NOT ready here.
            #
            # Back-pressure while idle is load-bearing: it is the only thing throttling the
            # free-running host->device DMA. Letting the loader swallow the stream instead (to
            # allow a sacrificial transfer to be over-fed) means the hardware laps the ring
            # continuously, litepcie_write() sees itself permanently overtaken, and every write
            # is dropped - the card then receives nothing but stale ring contents and no
            # transfer ever completes. Stalling the stream between transfers is what lets
            # software catch up, and after the first lap the drops stop.
            #
            # The garbage this parks at the head of the stream is handled by the resync marker
            # below, not by discarding it here.
            If(self.start.re & self.start.storage,
                NextValue(beats, 0),
                NextValue(sync_hunt, self.resync.storage),
                NextValue(sync_seen, 0),
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
                If(sync_hunt,
                    NextState("SYNC"),
                ).Else(
                    NextState("DATA"),
                )
            )
        )
        fsm.act("SYNC",
            # Discard until a marker is seen, then discard the run of markers; the payload is
            # the first non-marker beat after it, and that beat is left for DATA to consume.
            #
            # Skipping the whole run (rather than exactly one marker) is what lets the host
            # send a long prefix of them. The dropped region at the head of the stream can be
            # a megabyte, so a single marker would simply be lost and the loader would hunt
            # forever; a prefix longer than the drop always leaves one standing.
            If(conv.source.valid & (conv.source.data == SYNC_MAGIC),
                conv.source.ready.eq(1),
                NextValue(sync_seen, 1),
            ).Elif(conv.source.valid & sync_seen,
                NextState("DATA"),          # leave this beat for DATA
            ).Else(
                conv.source.ready.eq(1),    # pre-marker filler
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
            If(self.src_sel.storage,
                poke_fifo.source.connect(conv.sink, omit={"last"}),
            ).Else(
                self.sink.connect(conv.sink, omit={"last"}),
                self.sink.ready.eq(conv.sink.ready | fsm.ongoing("IDLE")),
            ),
            self.status.fields.poke_full.eq(~poke_fifo.sink.ready),
            self.status.fields.done.eq(fsm.ongoing("IDLE")),
            self.status.fields.sync.eq(fsm.ongoing("SYNC")),
            self.status.fields.beats.eq(beats),
        ]

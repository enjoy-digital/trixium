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

from functools import reduce
from operator import and_, or_

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
    def __init__(self, hbm_axis, data_width=128, chan_stride=0, chan_mod=0, chan_skew=0,
                 *, regions=None, reservations=(), endpoints=None, with_mstore=False):
        with_regions = regions is not None
        if with_regions:
            from trixium.capabilities import Access
            from trixium.gateware.region_span import RegionSpanValidator
            endpoints = tuple(endpoints or [0]*len(hbm_axis))
            if len(endpoints) != len(hbm_axis):
                raise ValueError('one endpoint is required per loader transport port')
            self.region_cmd = stream.Endpoint([('region', 32), ('offset', 64), ('length', 24)])
            self.region_data = stream.Endpoint([('data', 256)])
            self.region_abort = Signal()
            self.region_busy, self.region_done, self.region_failed = Signal(), Signal(), Signal()
            self.region_error = Signal(8)
            self.region_beats = Signal(24)
            self.region_validator = validator = RegionSpanValidator(regions, reservations,
                [(endpoint, axi.address_width) for endpoint, axi in zip(endpoints, hbm_axis)], access=Access.WRITE)
            self.comb += [validator.region.eq(self.region_cmd.region), validator.offset.eq(self.region_cmd.offset),
                          validator.length.eq(self.region_cmd.length << 5)]
        # chan_stride: byte distance between channel windows, added IN HARDWARE to host
        # uploads (base + chan*stride, 33 bits) exactly as the engine adds it for reads
        # and stores - so the host writes the channel-relative base and channel 15's
        # window above 4 GB needs no wider CSR. The SoC exports
        # BONSAI_LOADER_ADDS_CHAN_OFFSET so the host tools know not to add it themselves.
        assert chan_stride == 0 or (chan_stride & (chan_stride - 1)) == 0
        chan_shift = chan_stride.bit_length() - 1 if chan_stride else 0
        # chan_mod: channels per memory controller (see matvec.py) - the upload path adds
        # (chan % chan_mod)*stride, the window's offset inside the channel's own controller.
        assert chan_mod == 0 or (chan_mod & (chan_mod - 1)) == 0
        chan_sel = (lambda c: c & (chan_mod - 1)) if chan_mod else (lambda c: c)
        # chan_skew: the per-channel bank skew the engine adds (matvec.py), added here too.
        assert chan_skew == 0 or chan_mod
        chan_skew_of = lambda sel: Array([k*chan_skew for k in range(chan_mod)])[sel]
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
        self.store_cmd  = stream.Endpoint([("chan", 5), ("addr", 33), ("len", 24)])
        self.store_data = stream.Endpoint([("data", 256)])
        self.store_busy = Signal()
        # Multi-channel store (the DeltaNet stream): every writer takes the same span at its
        # own channel offset - the address the engine's reader of that channel uses - and
        # writer i drains mstore_data[i]. Granted from IDLE like the single store.
        if with_mstore:
            self.mstore_cmd  = stream.Endpoint([("addr", 33), ("len", 24)])
            self.mstore_data = [stream.Endpoint([("data", 256)]) for _ in hbm_axis]

        # # #

        n = len(hbm_axis)

        # Width conversion to 256-bit beats.
        self.conv = conv = stream.Converter(data_width, 256)

        # CSR poke -> stream: pack 32-bit writes into sink-width words, buffer them, and mux
        # that in place of the DMA when src_sel is set.
        # Pack straight to 256-bit beats and mux at the *converter output*, not its input:
        # feeding 128-bit words into the upconverter duplicated each one across both halves of
        # the beat on hardware (the sim DUT used data_width=256, so it never saw the
        # converter and the bug did not show). Nothing needs converting here anyway - the
        # host knows it is writing beats.
        poke_ratio = 256//32
        poke_sr    = Signal(256)
        poke_cnt   = Signal(max=poke_ratio)
        self.poke_fifo = poke_fifo = stream.SyncFIFO([("data", 256)], 256)
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
        #
        # Data reaches the writers through register slices at both ends of the crossing
        # (n16 timing #3c: the store data was the scratchpad BRAM output, combinational
        # through the state mux into every writer's W register - 5.8 ns of pure route to
        # the far HBM ports, two logic levels). `sbuf` registers the engine's store beats
        # once at the loader; `wbuf[i]` registers them again next to each writer, so the
        # 16-way fan-out is FF -> FF and Vivado can replicate the source. The DMA upload
        # path shares the per-writer slices. Latency only: +1 cycle per beat at each slice,
        # and the drain states still wait on the writers (w_active holds until the last
        # beat has left the slice).
        self.writers = []
        self.wbufs   = []
        for i, axi in enumerate(hbm_axis):
            writer = BonsaiAXIWriter(axi, checked=with_regions)
            setattr(self.submodules, f"writer{i}", writer)
            self.writers.append(writer)
            wbuf = stream.Buffer([("data", 256)])
            setattr(self.submodules, f"wbuf{i}", wbuf)
            self.comb += wbuf.source.connect(writer.sink_data)
            self.wbufs.append(wbuf)
        self.submodules.sbuf = sbuf = stream.Buffer([("data", 256)])
        self.comb += self.store_data.connect(sbuf.sink)
        st_src = sbuf.source

        beats = Signal(24)
        chan  = self.chan.storage

        st_chan = Signal(5)
        st_addr  = Signal(33)
        st_len  = Signal(24)
        st_cnt  = Signal(24)
        mst_addr = Signal(33)
        mst_len  = Signal(24)

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

        # Upload source seen by the transfer FSM: DMA (via the width converter) or CSR pokes.
        src = stream.Endpoint([("data", 256)])
        self.comb += If(self.src_sel.storage,
            poke_fifo.source.connect(src),
        ).Else(
            conv.source.connect(src),
        )

        self.fsm = fsm = FSM(reset_state="IDLE")
        region_idle = []
        if with_regions:
            region_port, region_address, region_length = Signal(5), Signal(64), Signal(24)
            region_fault = Signal(8)
            writer_error = Signal(4)
            self.comb += [
                writer_error.eq(Array([w.error for w in self.writers])[region_port]),
                self.region_busy.eq(fsm.ongoing('REGION_CMD') | fsm.ongoing('REGION_DATA')
                                    | fsm.ongoing('REGION_DRAIN') | fsm.ongoing('REGION_FAILED')),
                self.region_failed.eq(fsm.ongoing('REGION_FAILED')),
                self.region_error.eq(region_fault | Mux(self.region_busy, writer_error << 3, 0)),
                self.region_cmd.ready.eq(fsm.ongoing('IDLE') & ~(self.start.re & self.start.storage)
                                         & ~self.store_cmd.valid & ~self.region_abort
                                         & (~self.mstore_cmd.valid if with_mstore else 1)),
            ]
            region_idle = [If(self.region_cmd.valid & self.region_cmd.ready,
                NextValue(region_port, validator.port), NextValue(region_address, validator.address),
                NextValue(region_length, self.region_cmd.length), NextValue(self.region_beats, 0),
                NextValue(region_fault, validator.error),
                If(validator.valid, NextState('REGION_CMD')).Else(NextState('REGION_FAILED')),
            )]
            fsm.act('REGION_CMD',
                *[w.sink_cmd.valid.eq((region_port == i) & ~self.region_abort & (writer_error == 0))
                  for i, w in enumerate(self.writers)],
                *[w.sink_cmd.addr.eq(region_address) for w in self.writers],
                *[w.sink_cmd.len.eq(region_length) for w in self.writers],
                If(self.region_abort | (writer_error != 0),
                    NextValue(region_fault, region_fault | (self.region_abort << 7)), NextState('REGION_FAILED'),
                ).Elif(Array([w.sink_cmd.ready for w in self.writers])[region_port], NextState('REGION_DATA')),
            )
            framing_bad = Signal()
            self.comb += framing_bad.eq(self.region_data.last != (self.region_beats == region_length-1))
            fsm.act('REGION_DATA',
                *[w.sink.valid.eq((region_port == i) & self.region_data.valid & ~framing_bad
                                  & ~self.region_abort & (writer_error == 0)) for i, w in enumerate(self.wbufs)],
                *[w.sink.data.eq(self.region_data.data) for w in self.wbufs],
                self.region_data.ready.eq(Array([w.sink.ready for w in self.wbufs])[region_port]
                                          & ~self.region_abort & (writer_error == 0)),
                If(self.region_abort | (writer_error != 0) | (self.region_data.valid & framing_bad),
                    NextValue(region_fault, region_fault | ((self.region_abort
                              | (self.region_data.valid & framing_bad)) << 7)), NextState('REGION_FAILED'),
                ).Elif(self.region_data.valid & self.region_data.ready,
                    NextValue(self.region_beats, self.region_beats+1),
                    If(self.region_beats == region_length-1, NextState('REGION_DRAIN')),
                ),
            )
            fsm.act('REGION_DRAIN',
                If(self.region_abort | (writer_error != 0),
                    NextValue(region_fault, region_fault | (self.region_abort << 7)), NextState('REGION_FAILED'),
                ).Elif(Array([w.idle for w in self.writers])[region_port],
                    self.region_done.eq(1), NextState('IDLE'),
                ),
            )
            # Full cluster reset/drain is required after failure. Never retract
            # the writer's held AW/W here or falsely acknowledge partial writes.
            fsm.act('REGION_FAILED')
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
            ).Elif(self.mstore_cmd.valid if with_mstore else 0,
                *([self.mstore_cmd.ready.eq(1),
                   NextValue(mst_addr, self.mstore_cmd.addr),
                   NextValue(mst_len, self.mstore_cmd.len),
                   NextState("MST_CMD")] if with_mstore else []),
            ).Else(*region_idle)
        )
        mst_states = []
        if with_mstore:
            def chan_off(i): # The engine's per-reader constant (matvec.py _chan_off).
                local = (i % chan_mod) if chan_mod else i
                return local*chan_stride + local*chan_skew
            mst_cmd_done = Signal(n)
            mst_cnt = [Signal(24, name=f"mst_cnt{i}") for i in range(n)]
            mst_left = [Signal(name=f"mst_left{i}") for i in range(n)]
            all_cmd = Signal()
            self.comb += all_cmd.eq(reduce(and_, [mst_cmd_done[i] | (w.sink_cmd.valid & w.sink_cmd.ready)
                                                  for i, w in enumerate(self.writers)]))
            for i in range(n):
                self.comb += mst_left[i].eq(mst_cnt[i] != mst_len)
            fsm.act("MST_CMD",
                *[w.sink_cmd.valid.eq(~mst_cmd_done[i]) for i, w in enumerate(self.writers)],
                *[w.sink_cmd.addr.eq(mst_addr + chan_off(i)) for i, w in enumerate(self.writers)],
                *[w.sink_cmd.len.eq(mst_len) for w in self.writers],
                If(all_cmd, NextState("MST_DATA")),
            )
            fsm.act("MST_DATA",
                *[w.sink.valid.eq(d.valid & mst_left[i])
                  for i, (w, d) in enumerate(zip(self.wbufs, self.mstore_data))],
                *[w.sink.data.eq(d.data) for w, d in zip(self.wbufs, self.mstore_data)],
                *[d.ready.eq(w.sink.ready & mst_left[i])
                  for i, (w, d) in enumerate(zip(self.wbufs, self.mstore_data))],
                If(~reduce(or_, mst_left), NextState("MST_DRAIN")),
            )
            fsm.act("MST_DRAIN",
                If(reduce(and_, [w.idle for w in self.writers]), NextState("IDLE")),
            )
            self.sync += [
                If(fsm.ongoing("IDLE"),
                    mst_cmd_done.eq(0),
                    *[c.eq(0) for c in mst_cnt],
                ),
                If(fsm.ongoing("MST_CMD"),
                    *[If(w.sink_cmd.valid & w.sink_cmd.ready, mst_cmd_done[i].eq(1))
                      for i, w in enumerate(self.writers)],
                ),
                If(fsm.ongoing("MST_DATA"),
                    *[If(d.valid & d.ready, mst_cnt[i].eq(mst_cnt[i] + 1))
                      for i, d in enumerate(self.mstore_data)],
                ),
            ]
            mst_states = [fsm.ongoing(n_) for n_ in ("MST_CMD", "MST_DATA", "MST_DRAIN")]
        fsm.act("EST_CMD",
            *[w.sink_cmd.valid.eq(st_chan == i) for i, w in enumerate(self.writers)],
            *[w.sink_cmd.addr.eq(st_addr) for w in self.writers],
            *[w.sink_cmd.len.eq(st_len) for w in self.writers],
            If(Array([w.sink_cmd.ready for w in self.writers])[st_chan],
                NextState("EST_DATA"),
            )
        )
        fsm.act("EST_DATA",
            *[w.sink.valid.eq((st_chan == i) & st_src.valid)
              for i, w in enumerate(self.wbufs)],
            *[w.sink.data.eq(st_src.data) for w in self.wbufs],
            st_src.ready.eq(Array([w.sink.ready for w in self.wbufs])[st_chan]),
            If(st_src.valid & st_src.ready,
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
                                        | fsm.ongoing("EST_DRAIN") | reduce(or_, mst_states, 0))
        # Built term by term so that chan_skew=0 emits exactly the expression it always did.
        _upload_addr = self.base.storage + ((chan_sel(chan) << chan_shift) if chan_stride else 0)
        if chan_skew:
            _upload_addr = _upload_addr + chan_skew_of(chan_sel(chan))
        fsm.act("CMD",
            *[w.sink_cmd.valid.eq(chan == i) for i, w in enumerate(self.writers)],
            *[w.sink_cmd.addr.eq(_upload_addr) for w in self.writers],
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
            If(src.valid & (src.data == SYNC_MAGIC),
                src.ready.eq(1),
                NextValue(sync_seen, 1),
            ).Elif(src.valid & sync_seen,
                NextState("DATA"),          # leave this beat for DATA
            ).Else(
                src.ready.eq(1),    # pre-marker filler
            )
        )
        fsm.act("DATA",
            *[w.sink.valid.eq((chan == i) & src.valid)
              for i, w in enumerate(self.wbufs)],
            *[w.sink.data.eq(src.data) for w in self.wbufs],
            src.ready.eq(Array([w.sink.ready for w in self.wbufs])[chan]),
            If(src.valid & src.ready,
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
            self.status.fields.poke_full.eq(~poke_fifo.sink.ready),
            self.status.fields.done.eq(fsm.ongoing("IDLE")),
            self.status.fields.sync.eq(fsm.ongoing("SYNC")),
            self.status.fields.beats.eq(beats),
        ]

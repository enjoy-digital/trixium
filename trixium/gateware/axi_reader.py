#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiAXIReader: native AXI burst reader (the weight streamer front-end).

Takes {addr, len} commands (beat-aligned byte address, length in 256-bit beats), splits them
into INCR bursts (16 beats by default, never crossing a 4 KB boundary), keeps several bursts
outstanding with credit-based reservation of the output FIFO (so R data is never
back-pressured by us at the AXI port when credits were granted), and streams beats out with
`last` on the final beat of each command.

**ARIDs.** With n_ids=1 every burst carries ARID 0 and R data is in-order. That is
functionally fine and catastrophically slow on the real HBM: the Xilinx controller strictly
serializes same-ID reads (measured on the FK33: 33.7 cycles/beat from one port, one 16-beat
burst per round trip, while the identical RTL sustains 2.35 in the full-SoC simulation).
With n_ids=N bursts round-robin over N ARIDs so the controller can pipeline them; R data
then interleaves across IDs (in-order only per ID), so beats land in N per-ID FIFOs and a
drain stage replays them in issue order - reordering by construction, no address matching.
The per-ID FIFOs are sized for the outstanding-burst cap (info FIFO depth), so they can
never overflow: a third burst on an ID cannot have been issued while two sit unconsumed.

**burst_beats must not exceed 16.** The Xilinx HBM controller's ports are AXI3
(`AXI_nn_ARLEN[3:0]`, see ip/hbm*/hbm_*_stub.v), so AxLEN is four bits wide and the longest
legal burst is 16 beats. LiteX's AXIInterface carries an 8-bit `len`, and the upper bits are
simply dropped where it meets the IP - asking for 64 beats arrives as 16. Nothing complains:
the reader then waits forever for beats 17..64 of each burst. On hardware that showed up as
`ENGSTALL ... ch0 beats=00000020` (two bursts in flight x 16 beats delivered, then a
permanent stall) on the very first engine fetch from HBM.

Perf counters (free-running Signals, aggregated at the engine level): total beats delivered,
AR stall cycles (command issued but not accepted), R stall cycles (R valid, FIFO not ready -
should stay 0 thanks to crediting).
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import BURST_INCR

from trixium.layout import BEAT_BYTES

# BonsaiAXIReader ----------------------------------------------------------------------------------

class BonsaiAXIReader(LiteXModule):
    def __init__(self, axi, burst_beats=16, fifo_depth=512, n_ids=1, max_bursts=8):
        assert axi.data_width == 256
        assert 1 <= burst_beats <= 16, "AXI3 AxLEN is 4 bits: bursts are capped at 16 beats"
        assert n_ids in (1, 2, 4, 8)
        assert max_bursts % n_ids == 0
        self.sink   = sink   = stream.Endpoint([("addr", axi.address_width), ("len", 24)])
        self.source = source = stream.Endpoint([("data", 256)])

        self.total_beats = Signal(32)
        self.ar_stalls   = Signal(32)
        self.r_stalls    = Signal(32)

        # # #

        beat_shift = log2_int(BEAT_BYTES)

        # Burst splitter ----------------------------------------------------------------------
        cur_addr   = Signal(axi.address_width)
        beats_left = Signal(24)
        active     = Signal()

        self.comb += sink.ready.eq(~active)
        self.sync += [
            If(sink.valid & sink.ready,
                cur_addr.eq(sink.addr),
                beats_left.eq(sink.len),
                active.eq(sink.len != 0),
            )
        ]

        # Burst length: min(burst_beats, beats_left, beats to next 4 KB boundary).
        boundary_full  = Signal(13) # Full-width intermediate: Migen emits narrow-context
        boundary_beats = Signal(8)  # arithmetic inside shifts otherwise. 0 encodes 128.
        b_tmp          = Signal(25)
        burst_len      = Signal(8) # 1..burst_beats, 0 encodes 128 in b_tmp path (never here).
        self.comb += [
            boundary_full.eq(4096 - (cur_addr & 4095)),
            boundary_beats.eq(boundary_full >> beat_shift),
            If(beats_left < burst_beats,
                b_tmp.eq(beats_left),
            ).Else(
                b_tmp.eq(burst_beats),
            ),
            # boundary_beats == 0 means the full 128-beat page is ahead (addr 4 KB-aligned).
            If((boundary_beats != 0) & (boundary_beats < b_tmp),
                burst_len.eq(boundary_beats),
            ).Else(
                burst_len.eq(b_tmp),
            ),
        ]

        # Credit-based FIFO reservation.
        reserved  = Signal(max=2*fifo_depth)
        can_issue = Signal()
        self.comb += can_issue.eq(active & (reserved <= (fifo_depth - burst_len)))

        # Per-burst info in issue order (which ID it went out on, last burst of command?).
        idbits = max(1, log2_int(n_ids)) if n_ids > 1 else 1
        info = stream.SyncFIFO([("cmd_last", 1), ("bid", idbits)], max_bursts)
        self.submodules += info

        id_rr = Signal(idbits)
        self.comb += [
            axi.ar.valid.eq(can_issue & info.sink.ready),
            axi.ar.addr.eq(cur_addr),
            axi.ar.len.eq(burst_len - 1),
            axi.ar.size.eq(beat_shift), # 32 bytes/beat.
            axi.ar.burst.eq(BURST_INCR),
            axi.ar.id.eq(id_rr if n_ids > 1 else 0),
            info.sink.valid.eq(axi.ar.valid & axi.ar.ready),
            info.sink.cmd_last.eq(beats_left == burst_len),
            info.sink.bid.eq(id_rr if n_ids > 1 else 0),
        ]
        if n_ids > 1:
            self.sync += If(axi.ar.valid & axi.ar.ready, id_rr.eq(id_rr + 1))
        self.sync += [
            If(axi.ar.valid & axi.ar.ready,
                cur_addr.eq(cur_addr + (burst_len << beat_shift)),
                beats_left.eq(beats_left - burst_len),
                If(beats_left == burst_len,
                    active.eq(0),
                )
            )
        ]

        # R-channel register slice (skid buffer): the HBM hard macro's AXI interface pins
        # sit at fixed die locations; a comb path from them into the fabric FIFO was the
        # residual timing family at 225MHz. The slice holds beats outside the FIFO, so
        # the AR credit accounting (bounded by fifo_depth) stays conservative.
        r_valid  = Signal()
        r_data   = Signal(256)
        r_last   = Signal()
        r_id     = Signal(idbits)
        r_ready  = Signal()
        rs_valid = Signal()
        rs_data  = Signal(256)
        rs_last  = Signal()
        rs_id    = Signal(idbits)
        r_infire = Signal()
        self.comb += [
            axi.r.ready.eq(~rs_valid),
            r_infire.eq(axi.r.valid & ~rs_valid),
        ]
        self.sync += [
            If(r_ready | ~r_valid,
                If(rs_valid,
                    r_valid.eq(1),
                    r_data.eq(rs_data),
                    r_last.eq(rs_last),
                    r_id.eq(rs_id),
                    rs_valid.eq(0),
                ).Else(
                    r_valid.eq(r_infire),
                    r_data.eq(axi.r.data),
                    r_last.eq(axi.r.last),
                    r_id.eq(axi.r.id[:idbits]),
                )
            ).Elif(r_infire,
                rs_valid.eq(1),
                rs_data.eq(axi.r.data),
                rs_last.eq(axi.r.last),
                rs_id.eq(axi.r.id[:idbits]),
            )
        ]

        # Data FIFO ---------------------------------------------------------------------------
        fifo = stream.SyncFIFO([("data", 256)], fifo_depth, buffered=True)
        self.submodules += fifo

        if n_ids == 1:
            self.comb += [
                fifo.sink.valid.eq(r_valid),
                r_ready.eq(fifo.sink.ready),
                fifo.sink.data.eq(r_data),
                # Command-last: burst-last AND this burst is the command's final burst. The
                # info pop tracks the sliced stream, so command tags stay beat-aligned.
                fifo.sink.last.eq(r_last & info.source.cmd_last),
                info.source.ready.eq(r_valid & r_ready & r_last),
                fifo.source.connect(source),
            ]
        else:
            # Per-ID landing FIFOs: R data is in-order within an ID; depth covers the
            # outstanding cap (max_bursts/n_ids bursts of burst_beats each).
            perid_depth = (max_bursts//n_ids)*burst_beats
            # (`last` is a built-in endpoint flag - SyncFIFO carries it through.)
            perid = [stream.SyncFIFO([("data", 256)], perid_depth)
                     for _ in range(n_ids)]
            self.submodules += perid
            for i, p in enumerate(perid):
                self.comb += [
                    p.sink.valid.eq(r_valid & (r_id == i)),
                    p.sink.data.eq(r_data),
                    p.sink.last.eq(r_last),
                ]
            self.comb += r_ready.eq(Array([p.sink.ready for p in perid])[r_id])

            # Drain in issue order: the head of `info` names the ID whose burst is next;
            # its captured rlast marks the burst end, no beat counter needed.
            sel        = info.source.bid
            sel_valid  = Array([p.source.valid for p in perid])[sel]
            sel_data   = Array([p.source.data  for p in perid])[sel]
            sel_last   = Array([p.source.last  for p in perid])[sel]
            drain_fire = Signal()
            self.comb += [
                fifo.sink.valid.eq(info.source.valid & sel_valid),
                fifo.sink.data.eq(sel_data),
                fifo.sink.last.eq(sel_last & info.source.cmd_last),
                drain_fire.eq(fifo.sink.valid & fifo.sink.ready),
                info.source.ready.eq(drain_fire & sel_last),
                fifo.source.connect(source),
            ]
            for i, p in enumerate(perid):
                self.comb += p.source.ready.eq(drain_fire & (sel == i))

        # Credits: +burst_len on AR fire, -1 per beat delivered to the consumer.
        ar_fire  = Signal()
        out_fire = Signal()
        self.comb += [
            ar_fire.eq(axi.ar.valid & axi.ar.ready),
            out_fire.eq(source.valid & source.ready),
        ]
        self.sync += [
            reserved.eq(reserved + Mux(ar_fire, burst_len, 0) - Mux(out_fire, 1, 0)),
        ]

        # Perf counters -----------------------------------------------------------------------
        self.sync += [
            If(out_fire,                     self.total_beats.eq(self.total_beats + 1)),
            If(axi.ar.valid & ~axi.ar.ready, self.ar_stalls.eq(self.ar_stalls + 1)),
            If(axi.r.valid & ~axi.r.ready,   self.r_stalls.eq(self.r_stalls + 1)),
        ]

        # Idle indication (all issued data consumed, no command in flight).
        self.idle = Signal()
        self.comb += self.idle.eq(~active & (reserved == 0) & ~source.valid)

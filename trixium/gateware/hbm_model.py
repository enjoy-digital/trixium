#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Simulation-only behavioral model of one HBM2 pseudo-channel AXI port.

Matches the USPHBM2 port geometry exactly (AXI4, 256-bit data, 33-bit byte address, 6-bit id)
so the engine RTL is identical in simulation and on hardware. Supports INCR bursts, a
programmable AR-to-R latency and an LFSR-driven R-stall pattern (backpressure jitter) to
stress the streamer's credit logic. One command is processed at a time per direction
(in-order, single-ID traffic, matching the engine's usage); multiple outstanding bursts
simply queue in the AR/AW FIFOs.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface, BURST_INCR

from trixium.layout import BEAT_BYTES

# Helpers ------------------------------------------------------------------------------------------

def words_from_bytes(data):
    """bytes -> list of 256-bit little-endian words (padded with zeros)."""
    data = bytes(data)
    if len(data) % BEAT_BYTES:
        data = data + b"\x00"*(BEAT_BYTES - len(data) % BEAT_BYTES)
    return [int.from_bytes(data[i:i+BEAT_BYTES], "little") for i in range(0, len(data), BEAT_BYTES)]

# BonsaiHBMChannelModel ----------------------------------------------------------------------------

class BonsaiHBMChannelModel(LiteXModule):
    def __init__(self, axi=None, size=64*1024, init=None, latency=16, stall_rate=0):
        """
        axi        : AXIInterface slave (created if None).
        size       : modeled memory bytes (power of two).
        init       : optional list of 256-bit init words (words_from_bytes()).
        latency    : cycles between AR acceptance and first R beat.
        stall_rate : R-stall density, 0..15 (LFSR nibble < stall_rate stalls R).
        """
        if axi is None:
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        self.axi = axi
        depth    = size//BEAT_BYTES

        # # #

        mem = Memory(256, depth, init=init)
        self.specials += mem
        wr_port = mem.get_port(write_capable=True, we_granularity=8)
        rd_port = mem.get_port(async_read=True)
        self.specials += wr_port, rd_port

        # LFSR for R stalls.
        lfsr = Signal(16, reset=0xACE1)
        self.sync += lfsr.eq(Cat(lfsr[1:], lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[5]))
        stall = Signal()
        self.comb += stall.eq(lfsr[:4] < stall_rate)

        # Read path ---------------------------------------------------------------------------
        ar_fifo = stream.SyncFIFO([("addr", 33), ("len", 8), ("id", 6)], 8)
        self.submodules += ar_fifo
        self.comb += [
            ar_fifo.sink.valid.eq(axi.ar.valid),
            axi.ar.ready.eq(ar_fifo.sink.ready),
            ar_fifo.sink.addr.eq(axi.ar.addr),
            ar_fifo.sink.len.eq(axi.ar.len),
            ar_fifo.sink.id.eq(axi.ar.id),
        ]

        r_beat  = Signal(30) # Current beat index.
        r_left  = Signal(9)  # Beats left in burst (len+1 counted down).
        r_id    = Signal(6)
        lat_cnt = Signal(16)

        self.comb += [
            rd_port.adr.eq(r_beat),
            axi.r.data.eq(rd_port.dat_r),
            axi.r.id.eq(r_id),
            axi.r.resp.eq(0),
            axi.r.last.eq(r_left == 1),
        ]

        self.rd_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            ar_fifo.source.ready.eq(1),
            If(ar_fifo.source.valid,
                NextValue(r_beat, ar_fifo.source.addr[log2_int(BEAT_BYTES):]),
                NextValue(r_left, ar_fifo.source.len + 1),
                NextValue(r_id,   ar_fifo.source.id),
                NextValue(lat_cnt, latency),
                NextState("LATENCY"),
            )
        )
        fsm.act("LATENCY",
            NextValue(lat_cnt, lat_cnt - 1),
            If(lat_cnt == 0,
                NextState("STREAM"),
            )
        )
        fsm.act("STREAM",
            axi.r.valid.eq(~stall),
            If(axi.r.valid & axi.r.ready,
                NextValue(r_beat, r_beat + 1),
                NextValue(r_left, r_left - 1),
                If(r_left == 1,
                    # Chain the next queued burst back-to-back (the real HBM pipelines
                    # outstanding bursts; latency only applies from an idle queue).
                    If(ar_fifo.source.valid,
                        ar_fifo.source.ready.eq(1),
                        NextValue(r_beat, ar_fifo.source.addr[log2_int(BEAT_BYTES):]),
                        NextValue(r_left, ar_fifo.source.len + 1),
                        NextValue(r_id,   ar_fifo.source.id),
                    ).Else(
                        NextState("IDLE"),
                    )
                )
            )
        )

        # Write path --------------------------------------------------------------------------
        aw_fifo = stream.SyncFIFO([("addr", 33), ("len", 8), ("id", 6)], 8)
        self.submodules += aw_fifo
        self.comb += [
            aw_fifo.sink.valid.eq(axi.aw.valid),
            axi.aw.ready.eq(aw_fifo.sink.ready),
            aw_fifo.sink.addr.eq(axi.aw.addr),
            aw_fifo.sink.len.eq(axi.aw.len),
            aw_fifo.sink.id.eq(axi.aw.id),
        ]

        w_beat = Signal(30)
        w_id   = Signal(6)

        self.comb += [
            wr_port.adr.eq(w_beat),
            wr_port.dat_w.eq(axi.w.data),
        ]

        self.wr_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            aw_fifo.source.ready.eq(1),
            If(aw_fifo.source.valid,
                NextValue(w_beat, aw_fifo.source.addr[log2_int(BEAT_BYTES):]),
                NextValue(w_id,   aw_fifo.source.id),
                NextState("STREAM"),
            )
        )
        fsm.act("STREAM",
            axi.w.ready.eq(1),
            If(axi.w.valid,
                wr_port.we.eq(axi.w.strb),
                NextValue(w_beat, w_beat + 1),
                If(axi.w.last,
                    NextState("RESP"),
                )
            )
        )
        fsm.act("RESP",
            axi.b.valid.eq(1),
            axi.b.id.eq(w_id),
            axi.b.resp.eq(0),
            If(axi.b.ready,
                NextState("IDLE"),
            )
        )

        self.mem = mem # Exposed for direct inspection in tests.

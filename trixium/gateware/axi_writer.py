#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiAXIWriter: native AXI4 burst writer (KV-cache append + weight loader back-end).

Takes {addr, len} commands and a beat data stream, splits into INCR bursts (4 KB safe),
issues W data only after the corresponding AW was accepted, counts B responses and exposes
`idle` for sequencing. Full-beat writes only (wstrb all-ones). Single AWID."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import BURST_INCR

from trixium.layout import BEAT_BYTES

# BonsaiAXIWriter ----------------------------------------------------------------------------------

class BonsaiAXIWriter(LiteXModule):
    def __init__(self, axi, burst_beats=64):
        assert axi.data_width == 256
        self.sink_cmd  = cmd  = stream.Endpoint([("addr", axi.address_width), ("len", 24)])
        self.sink_data = data = stream.Endpoint([("data", 256)])
        self.idle      = Signal()

        # # #

        beat_shift = log2_int(BEAT_BYTES)

        # Burst splitter (mirror of the reader's).
        cur_addr   = Signal(axi.address_width)
        beats_left = Signal(24)
        active     = Signal()

        self.comb += cmd.ready.eq(~active)
        self.sync += [
            If(cmd.valid & cmd.ready,
                cur_addr.eq(cmd.addr),
                beats_left.eq(cmd.len),
                active.eq(cmd.len != 0),
            )
        ]

        boundary_beats = Signal(8)
        b_tmp          = Signal(25)
        burst_len      = Signal(8)
        self.comb += [
            boundary_beats.eq((4096 - (cur_addr & 4095)) >> beat_shift),
            If(beats_left < burst_beats,
                b_tmp.eq(beats_left),
            ).Else(
                b_tmp.eq(burst_beats),
            ),
            If((boundary_beats != 0) & (boundary_beats < b_tmp),
                burst_len.eq(boundary_beats),
            ).Else(
                burst_len.eq(b_tmp),
            ),
        ]

        # Issued-burst queue: W beats stream only for accepted AWs.
        winfo = stream.SyncFIFO([("len", 8)], 8)
        self.submodules += winfo

        self.comb += [
            axi.aw.valid.eq(active & winfo.sink.ready),
            axi.aw.addr.eq(cur_addr),
            axi.aw.len.eq(burst_len - 1),
            axi.aw.size.eq(beat_shift),
            axi.aw.burst.eq(BURST_INCR),
            axi.aw.id.eq(0),
            winfo.sink.valid.eq(axi.aw.valid & axi.aw.ready),
            winfo.sink.len.eq(burst_len),
        ]
        self.sync += [
            If(axi.aw.valid & axi.aw.ready,
                cur_addr.eq(cur_addr + (burst_len << beat_shift)),
                beats_left.eq(beats_left - burst_len),
                If(beats_left == burst_len,
                    active.eq(0),
                )
            )
        ]

        # W data path: pop winfo, stream len beats with wlast.
        w_left   = Signal(8)
        w_active = Signal()
        self.comb += [
            winfo.source.ready.eq(~w_active),
            axi.w.valid.eq(w_active & data.valid),
            axi.w.data.eq(data.data),
            axi.w.strb.eq(2**len(axi.w.strb) - 1),
            axi.w.last.eq(w_left == 1),
            data.ready.eq(w_active & axi.w.ready),
        ]
        self.sync += [
            If(winfo.source.valid & winfo.source.ready,
                w_left.eq(winfo.source.len),
                w_active.eq(1),
            ),
            If(axi.w.valid & axi.w.ready,
                w_left.eq(w_left - 1),
                If(w_left == 1,
                    w_active.eq(0),
                )
            ),
        ]

        # B responses.
        outstanding = Signal(8)
        self.comb += axi.b.ready.eq(1)
        self.sync += outstanding.eq(outstanding
            + Mux(axi.aw.valid & axi.aw.ready, 1, 0)
            - Mux(axi.b.valid & axi.b.ready, 1, 0))

        self.comb += self.idle.eq(~active & ~w_active & (outstanding == 0) & ~winfo.source.valid)

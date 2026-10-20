#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Simulation-only behavioral model of one HBM2 pseudo-channel AXI port.

Matches the USPHBM2 port geometry exactly (AXI4, 256-bit data, 33-bit byte address, 6-bit id)
so the engine RTL is identical in simulation and on hardware. Supports INCR bursts, a
programmable AR-to-R latency and an LFSR-driven R-stall pattern (backpressure jitter) to
stress the streamer's credit logic. Reads are in order, one command at a time (outstanding
bursts queue in the AR FIFO), or out of order across IDs with reorder_ids; writes are one
command at a time.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface

from model.layout import BEAT_BYTES

# Helpers ------------------------------------------------------------------------------------------

def words_from_bytes(data):
    """bytes -> list of 256-bit little-endian words (padded with zeros)."""
    data = bytes(data)
    if len(data) % BEAT_BYTES:
        data = data + b"\x00"*(BEAT_BYTES - len(data) % BEAT_BYTES)
    return [int.from_bytes(data[i:i+BEAT_BYTES], "little") for i in range(0, len(data), BEAT_BYTES)]

# HBMChannelModel ----------------------------------------------------------------------------------

class HBMChannelModel(LiteXModule):
    """Behavioral HBM2 pseudo-channel port (simulation only).

    axi        : AXIInterface slave (created if None).
    size       : modeled memory bytes (power of two).
    init       : optional list of 256-bit init words (words_from_bytes()).
    latency    : cycles between AR acceptance and first R beat.
    stall_rate : R-stall density, 0..15 (LFSR nibble < stall_rate stalls R).
    reorder_ids: 0 keeps the simple in-order read server. N (power of two) models the
                 real Xilinx HBM controller's read behavior instead: requests with the
                 SAME ARID are strictly serialized - the next one is not even started
                 until the previous one's data has fully returned, plus a `latency` gap -
                 while requests on DIFFERENT IDs overlap (one burst's latency counts down
                 while another streams). Measured on the FK33, that same-ID serialization
                 is exactly why a single-ID reader gets 33.7 cycles/beat out of a port
                 whose RTL-side path sustains 2.35 in the full-SoC sim: one 16-beat burst
                 per round trip. A single-ID reader against this model reproduces the
                 pathology; the multi-ID reader must erase it. Response order across IDs
                 follows service order (rotating scan), not request order.
    write_order: the slave's AW/W handshake discipline - every one is legal AXI and a
                 master must work with all three (see AXIResetShim):
                 "aw_first"   : AW into a FIFO whenever presented; W taken only after
                                its AW (the default).
                 "w_first"    : W beats taken into a FIFO ahead of their AW (LiteDRAM's
                                bridge: a W FIFO independent of its one-deep AW buffer).
                 "aw_after_w" : AWREADY only once WVALID is seen (a slave that waits for
                                data before the address - the HBM IP behaves like this).
    """
    def __init__(self, axi=None, size=64*1024, init=None, latency=16, stall_rate=0,
                 reorder_ids=0, write_order="aw_first", data_width=256, axi3=True):
        # data_width: the port's (256: HBM; 512: the Efinix hard DDR controller's AXI4 port);
        # axi3: the HBM ports' 4-bit AxLEN (an AXI4 port takes 8-bit lengths).
        assert write_order in ("aw_first", "w_first", "aw_after_w")
        if axi is None:
            axi = AXIInterface(data_width=data_width, address_width=33, id_width=6)
        data_width = axi.data_width
        assert reorder_ids == 0 or data_width == 256
        self.beat_bytes = data_width//8
        self.axi3       = axi3
        self.axi = axi
        # Test control: while `freeze` is up the read server delivers nothing (a port that
        # stops answering - the wedge the engine soft reset exists for); released later, the
        # beats of the bursts it still owes come out, which is what the AXI shim must absorb.
        self.freeze = Signal()

        self.write_order     = write_order
        self.autocsr_exclude = {"mem"} # Keep the model memory off the CSR bus (AutoCSR scan).

        # # #

        # Memory.
        depth = size//self.beat_bytes
        mem   = Memory(data_width, depth, init=init)
        self.specials += mem
        wr_port = mem.get_port(write_capable=True, we_granularity=8)
        rd_port = mem.get_port(async_read=True)
        self.specials += wr_port, rd_port

        # LFSR for R stalls.
        lfsr = Signal(16, reset=0xACE1)
        self.sync += lfsr.eq(Cat(lfsr[1:], lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[5]))
        stall = Signal()
        self.comb += stall.eq(lfsr[:4] < stall_rate)

        # AXI3 burst-length truncation -------------------------------------------------------------
        # The real IP's ports are AXI3: AXI_nn_AxLEN is 4 bits, so anything above 16 beats
        # silently loses its upper bits on the way in. Model that exactly, or simulation
        # happily serves 64-beat bursts that hardware never will (a stall on the first fetch).
        # `axlen_overflow` latches the mistake so a test can name it instead of timing out.
        self.axlen_overflow = axlen_overflow = Signal()
        if axi3:
            self.sync += If((axi.ar.valid & (axi.ar.len[4:] != 0)) |
                            (axi.aw.valid & (axi.aw.len[4:] != 0)),
                axlen_overflow.eq(1),
            )

        if reorder_ids:
            self._reorder_read_path(axi, rd_port, latency, stall, reorder_ids)
        else:
            self._inorder_read_path(axi, rd_port, latency, stall)
        self._write_path(axi, wr_port)
        self.mem = mem # Exposed for direct inspection in tests (excluded from AutoCSR above).

    def _inorder_read_path(self, axi, rd_port, latency, stall):
        # Read path --------------------------------------------------------------------------------
        ar_fifo = stream.SyncFIFO([("addr", axi.address_width), ("len", 8), ("id", axi.id_width)], 8)
        self += ar_fifo
        self.comb += [
            ar_fifo.sink.valid.eq(axi.ar.valid),
            axi.ar.ready.eq(ar_fifo.sink.ready),
            ar_fifo.sink.addr.eq(axi.ar.addr),
            ar_fifo.sink.len.eq(axi.ar.len[:4] if self.axi3 else axi.ar.len),
            ar_fifo.sink.id.eq(axi.ar.id),
        ]

        r_beat  = Signal(30) # Current beat index.
        r_left  = Signal(9)  # Beats left in burst (len+1 counted down).
        r_id    = Signal(axi.id_width)
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
                NextValue(r_beat, ar_fifo.source.addr[log2_int(self.beat_bytes):]),
                NextValue(r_left,  ar_fifo.source.len + 1),
                NextValue(r_id,    ar_fifo.source.id),
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
            axi.r.valid.eq(~stall & ~self.freeze),
            If(axi.r.valid & axi.r.ready,
                NextValue(r_beat, r_beat + 1),
                NextValue(r_left, r_left - 1),
                If(r_left == 1,
                    # Chain the next queued burst back-to-back (the real HBM pipelines
                    # outstanding bursts; latency only applies from an idle queue).
                    If(ar_fifo.source.valid,
                        ar_fifo.source.ready.eq(1),
                        NextValue(r_beat, ar_fifo.source.addr[log2_int(self.beat_bytes):]),
                        NextValue(r_left, ar_fifo.source.len + 1),
                        NextValue(r_id,   ar_fifo.source.id),
                    ).Else(
                        NextState("IDLE"),
                    )
                )
            )
        )

    def _reorder_read_path(self, axi, rd_port, latency, stall, nq):
        # Per-ID request queues (ID low bits select the queue).
        qbits = log2_int(nq)
        ar_qs = [stream.SyncFIFO([("addr", axi.address_width), ("len", 8), ("id", axi.id_width)], 4) for _ in range(nq)]
        self.submodules += ar_qs
        for i, q in enumerate(ar_qs):
            self.comb += [
                q.sink.valid.eq(axi.ar.valid & (axi.ar.id[:qbits] == i)),
                q.sink.addr.eq(axi.ar.addr),
                q.sink.len.eq(axi.ar.len[:4]),
                q.sink.id.eq(axi.ar.id),
            ]
        self.comb += axi.ar.ready.eq(Array([q.sink.ready for q in ar_qs])[axi.ar.id[:qbits]])

        # One staged slot per queue, each counting its own latency down IN PARALLEL - that
        # is exactly the pipelining same-ID serialization permits: one in-flight burst per
        # ID, any number of IDs overlapped. A queue refills its slot only `latency` after
        # its previous burst finished streaming (cooldown), so single-ID traffic degrades
        # to one burst per round trip - the measured hardware pathology - while multi-ID
        # chains bursts back-to-back on the data channel.
        st_valid = [Signal(    name=f"st_valid{i}") for i in range(nq)]
        st_addr  = [Signal(33, name=f"st_addr{i}")  for i in range(nq)]
        st_len   = [Signal(8,  name=f"st_len{i}")   for i in range(nq)]
        st_id    = [Signal(axi.id_width, name=f"st_id{i}") for i in range(nq)]
        st_lat   = [Signal(16, name=f"st_lat{i}")   for i in range(nq)]
        cooldown = [Signal(16, name=f"cooldown{i}") for i in range(nq)]

        cur_valid = Signal()
        cur_beat  = Signal(30)
        cur_left  = Signal(9)
        cur_id    = Signal(axi.id_width)

        rr_fire  = Signal()
        cur_done = Signal()
        self.comb += [
            rd_port.adr.eq(cur_beat),
            axi.r.data.eq(rd_port.dat_r),
            axi.r.id.eq(cur_id),
            axi.r.resp.eq(0),
            axi.r.last.eq(cur_left == 1),
            axi.r.valid.eq(cur_valid & ~stall & ~self.freeze),
            rr_fire.eq(axi.r.valid & axi.r.ready),
            cur_done.eq(rr_fire & (cur_left == 1)),
        ]

        # Promote any ready staged burst when the data channel frees up; rotating priority
        # so the response order shuffles across IDs.
        qp    = Signal(qbits)
        ready = Signal(nq)
        for i in range(nq):
            self.comb += ready[i].eq(st_valid[i] & (st_lat[i] == 0))
        # pick_q = first ready queue at/after qp (priority rotates each cycle).
        pick_ok = Signal()
        pick_q  = Signal(qbits)
        cases   = If(0, pick_ok.eq(0))
        for off in range(nq):
            cond = Signal(name=f"pick_cond{off}")
            self.comb += cond.eq(Array(ready)[qp + off])
            cases = cases.Elif(cond, pick_ok.eq(1), pick_q.eq(qp + off))
        self.comb += cases

        promote = Signal()
        self.comb += promote.eq((~cur_valid | cur_done) & pick_ok)

        self.sync += [
            qp.eq(qp + 1),
            If(promote,
                cur_valid.eq(1),
                cur_beat.eq(Array(st_addr)[pick_q][log2_int(BEAT_BYTES):]),
                cur_left.eq(Array(st_len)[pick_q] + 1),
                cur_id.eq(Array(st_id)[pick_q]),
            ).Elif(cur_done,
                cur_valid.eq(0),
            ),
            If(rr_fire & ~promote,
                cur_beat.eq(cur_beat + 1),
                cur_left.eq(cur_left - 1),
            ),
            If(cur_valid & cur_done,
                Array(cooldown)[cur_id[:qbits]].eq(latency),
            ),
        ]
        for i, q in enumerate(ar_qs):
            fill = Signal(name=f"fill{i}")
            self.comb += [
                fill.eq(q.source.valid & ~st_valid[i] & (cooldown[i] == 0)
                        & ~(cur_valid & (cur_id[:qbits] == i))),
                q.source.ready.eq(fill),
            ]
            self.sync += [
                If(cooldown[i] != 0, cooldown[i].eq(cooldown[i] - 1)),
                If(fill,
                    st_valid[i].eq(1),
                    st_addr[i].eq(q.source.addr),
                    st_len[i].eq(q.source.len),
                    st_id[i].eq(q.source.id),
                    st_lat[i].eq(latency),
                ).Elif(st_valid[i] & (st_lat[i] != 0),
                    st_lat[i].eq(st_lat[i] - 1),
                ),
                If(promote & (pick_q == i),
                    st_valid[i].eq(0),
                ),
            ]

    def _write_path(self, axi, wr_port):
        # Write path -------------------------------------------------------------------------------
        aw_fifo = stream.SyncFIFO([("addr", axi.address_width), ("len", 8), ("id", axi.id_width)], 8)
        self += aw_fifo
        self.comb += [
            aw_fifo.sink.addr.eq(axi.aw.addr),
            aw_fifo.sink.len.eq(axi.aw.len[:4] if self.axi3 else axi.aw.len),
            aw_fifo.sink.id.eq(axi.aw.id),
        ]
        if self.write_order == "aw_after_w":
            # The address is taken only once the data has started to show.
            self.comb += [
                aw_fifo.sink.valid.eq(axi.aw.valid & axi.w.valid),
                axi.aw.ready.eq(aw_fifo.sink.ready & axi.w.valid),
            ]
        else:
            self.comb += [
                aw_fifo.sink.valid.eq(axi.aw.valid),
                axi.aw.ready.eq(aw_fifo.sink.ready),
            ]

        # The data source of the write server: the bus, or a FIFO fed ahead of the AW.
        dw = axi.data_width
        w = stream.Endpoint([("data", dw), ("strb", dw//8)]) # `last` is the stream's own.
        if self.write_order == "w_first":
            w_fifo = stream.SyncFIFO([("data", dw), ("strb", dw//8)], 64)
            self += w_fifo
            self.comb += [
                w_fifo.sink.valid.eq(axi.w.valid),
                axi.w.ready.eq(w_fifo.sink.ready),
                w_fifo.sink.data.eq(axi.w.data),
                w_fifo.sink.strb.eq(axi.w.strb),
                w_fifo.sink.last.eq(axi.w.last),
                w_fifo.source.connect(w),
            ]
        else:
            self.comb += [
                w.valid.eq(axi.w.valid),
                axi.w.ready.eq(w.ready),
                w.data.eq(axi.w.data),
                w.strb.eq(axi.w.strb),
                w.last.eq(axi.w.last),
            ]

        w_beat = Signal(30)
        w_id   = Signal(axi.id_width)

        self.comb += [
            wr_port.adr.eq(w_beat),
            wr_port.dat_w.eq(w.data),
        ]

        self.wr_fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            aw_fifo.source.ready.eq(1),
            If(aw_fifo.source.valid,
                NextValue(w_beat, aw_fifo.source.addr[log2_int(self.beat_bytes):]),
                NextValue(w_id,   aw_fifo.source.id),
                NextState("STREAM"),
            )
        )
        fsm.act("STREAM",
            w.ready.eq(1),
            If(w.valid,
                wr_port.we.eq(w.strb),
                NextValue(w_beat, w_beat + 1),
                If(w.last,
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

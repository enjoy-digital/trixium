#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Exact-FP32 DeltaNet recurrence building blocks.

The recurrent matrix is row-major in DDR and in the shared scratchpad.  A 256-bit beat
therefore contains eight adjacent columns of one row.  The FP32 pipeline instead advances
eight independent rows every cycle so the 11-cycle adder dependency is hidden by sixteen
rows per lane.  :class:`BonsaiDeltaNetState` is the transpose buffer between those layouts:

* load/dump are sequential row-major 256-bit streams;
* bank ``b`` owns rows whose index modulo eight is ``b``;
* a compute read returns one selected FP32 column from all eight banks;
* a compute write updates that column in all eight banks.

For the 27B geometry (128 x 128) the eight banks total 64 KiB and infer as 16 RAMB36s.
"""

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import *
from litex.soc.interconnect import stream


class BonsaiDeltaNetState(LiteXModule):
    """Banked row-major state buffer with an eight-row compute port."""

    def __init__(self, rows=128, cols=128, lanes=8):
        assert rows % lanes == 0
        assert cols % 8 == 0
        assert lanes > 0 and lanes & (lanes - 1) == 0
        groups = cols // 8
        assert groups > 0 and groups & (groups - 1) == 0
        rows_per_lane = rows // lanes
        depth = rows_per_lane * groups
        beats = rows * groups
        lane_bits = (lanes - 1).bit_length()
        group_bits = (groups - 1).bit_length()

        self.rows = rows
        self.cols = cols
        self.lanes = lanes
        self.groups = groups
        self.rows_per_lane = rows_per_lane
        self.beats = beats

        # Sequential row-major load stream. load_start restarts at row 0, column 0.
        self.load_start = Signal()
        self.load_valid = Signal()
        self.load_ready = Signal()
        self.load_data  = Signal(256)
        self.load_beat  = Signal(max=beats)
        self.load_done  = Signal()

        # Eight-row compute port. row_phase selects rows
        # {phase*lanes + 0, ..., phase*lanes + lanes-1}.
        self.rd_en        = Signal()
        self.rd_row_phase = Signal(max=rows_per_lane)
        self.rd_col       = Signal(max=cols)
        self.rd_valid     = Signal()
        self.rd_data      = [Signal(32, name=f"dn_state_rd{i}") for i in range(lanes)]

        self.wr_en        = Signal()
        self.wr_row_phase = Signal(max=rows_per_lane)
        self.wr_col       = Signal(max=cols)
        self.wr_data      = [Signal(32, name=f"dn_state_wr{i}") for i in range(lanes)]

        # Sequential row-major dump stream. The source is backpressure-safe.
        self.dump_start = Signal()
        self.source = stream.Endpoint([("data", 256)])
        self.dump_done = Signal()

        # # #

        memories = []
        read_ports = []
        write_ports = []
        for lane in range(lanes):
            mem = Memory(256, depth, name=f"dn_state{lane}")
            rp = mem.get_port(has_re=True)
            wp = mem.get_port(write_capable=True, we_granularity=32)
            self.specials += mem, rp, wp
            memories.append(mem)
            read_ports.append(rp)
            write_ports.append(wp)
        # Test-visible storage, like BonsaiScratchpad.mem.
        self.memories = memories

        load_active = Signal()       # Still accepting stream beats.
        load_writing = Signal()      # FIFO may still contain accepted beats.
        load_row    = Signal(max=rows)
        load_bank   = Signal(max=lanes)
        load_group  = Signal(max=groups)
        load_phase  = Signal(max=rows_per_lane)
        load_addr   = Signal(max=depth)
        self.comb += [
            load_row.eq(self.load_beat >> group_bits),
            load_bank.eq(self.load_beat[group_bits:group_bits + lane_bits]),
            load_group.eq(self.load_beat[:group_bits]),
            load_phase.eq(self.load_beat >> (group_bits + lane_bits)),
            load_addr.eq(Cat(load_group, load_phase)),
        ]
        # Payload and decoded destination travel through the same FIFO word. This is more
        # than bookkeeping: it prevents an independently-updated beat counter from moving
        # the BRAM address one cycle ahead of the wide payload at a clock edge.
        load_fifo_width = 256 + lane_bits + len(load_addr) + 1
        load_fifo = SyncFIFO(load_fifo_width, 4)
        self.submodules.load_fifo = load_fifo
        load_fire = Signal()
        load_push = Signal()
        load_write = Signal()
        load_pending = Signal()
        load_data_r = Signal(256)
        load_bank_r = Signal(max=lanes)
        load_addr_r = Signal(max=depth)
        load_last_r = Signal()
        load_fifo_data = Signal(256)
        load_fifo_bank = Signal(max=lanes)
        load_fifo_addr = Signal(max=depth)
        load_fifo_last = Signal()
        self.comb += [
            # One elastic register makes data and decoded destination a single sampled
            # object before they enter the FIFO. It can push the previous beat and accept
            # the next in the same cycle, so sustained throughput remains one beat/cycle.
            self.load_ready.eq(load_active & (~load_pending | load_fifo.writable)),
            load_fire.eq(self.load_valid & self.load_ready),
            load_push.eq(load_pending & load_fifo.writable),
            load_fifo.we.eq(load_push),
            load_fifo.din.eq(Cat(load_data_r, load_bank_r, load_addr_r, load_last_r)),
            load_write.eq(load_writing & load_fifo.readable),
            load_fifo.re.eq(load_write),
            Cat(load_fifo_data, load_fifo_bank, load_fifo_addr, load_fifo_last).eq(
                load_fifo.dout),
        ]

        # Compute address. cols are grouped eight floats per BRAM word.
        rd_group = Signal(max=groups)
        rd_sub   = Signal(3)
        rd_addr  = Signal(max=depth)
        rd_pending = Signal()
        rd_addr_r  = Signal(max=depth)
        rd_sub_r   = Signal(3)
        wr_group = Signal(max=groups)
        wr_sub   = Signal(3)
        wr_addr  = Signal(max=depth)
        self.comb += [
            rd_group.eq(self.rd_col >> 3),
            rd_sub.eq(self.rd_col[:3]),
            rd_addr.eq(Cat(rd_group, self.rd_row_phase)),
            wr_group.eq(self.wr_col >> 3),
            wr_sub.eq(self.wr_col[:3]),
            wr_addr.eq(Cat(wr_group, self.wr_row_phase)),
        ]
        rd_sub_d = Signal(3)

        # Dump reads one row-major beat at a time. All banks see the same local address;
        # the delayed bank number selects the one whose synchronous output is valid.
        dump_active = Signal()
        dump_count  = Signal(max=beats)
        dump_row    = Signal(max=rows)
        dump_bank   = Signal(max=lanes)
        dump_group  = Signal(max=groups)
        dump_phase  = Signal(max=rows_per_lane)
        dump_addr   = Signal(max=depth)
        self.comb += [
            dump_row.eq(dump_count >> group_bits),
            dump_bank.eq(dump_count[group_bits:group_bits + lane_bits]),
            dump_group.eq(dump_count[:group_bits]),
            dump_phase.eq(dump_count >> (group_bits + lane_bits)),
            dump_addr.eq(Cat(dump_group, dump_phase)),
        ]

        # A small FIFO absorbs downstream stalls and the one already-issued synchronous
        # read. Its payload carries `last`, so dump_done means the final beat was accepted.
        dump_fifo = SyncFIFO(257, 4)
        self.submodules.dump_fifo = dump_fifo
        dump_request = Signal()
        dump_return  = Signal()
        dump_addr_r  = Signal(max=depth)
        dump_bank_r  = Signal(max=lanes)
        dump_last_r  = Signal()
        dump_bank_d  = Signal(max=lanes)
        dump_last_d  = Signal()
        dump_issue   = Signal()
        dump_data    = Signal(256)
        self.comb += [
            # Keep one read in flight for now: the selected bank/address remains stable
            # through the synchronous BRAM read and FIFO capture, which makes the transpose
            # boundary unambiguous under backpressure. The recurrence pipeline can later
            # overlap a tagged sequence of these reads without changing this interface.
            dump_issue.eq(dump_active & ~dump_request & ~dump_return & dump_fifo.writable),
            dump_data.eq(Array([p.dat_r for p in read_ports])[dump_bank_d]),
            dump_fifo.we.eq(dump_return),
            dump_fifo.din.eq(Cat(dump_data, dump_last_d)),
            self.source.valid.eq(dump_fifo.readable),
            self.source.data.eq(dump_fifo.dout[:256]),
            dump_fifo.re.eq(self.source.valid & self.source.ready),
            self.dump_done.eq(dump_fifo.re & dump_fifo.dout[256]),
        ]

        # RAM port muxes. Load owns W; otherwise the compute update does. Compute owns R;
        # dump uses R only after the caller has stopped compute reads.
        for lane, (rp, wp) in enumerate(zip(read_ports, write_ports)):
            self.comb += [
                rp.adr.eq(Mux(dump_active | dump_request | dump_return,
                              dump_addr_r, rd_addr_r)),
                rp.re.eq(Mux(dump_active | dump_request | dump_return,
                             dump_request, rd_pending)),
                wp.adr.eq(Mux(load_write, load_fifo_addr, wr_addr)),
                wp.dat_w.eq(Mux(load_write, load_fifo_data,
                                Replicate(self.wr_data[lane], 8))),
                wp.we.eq(Mux(load_write,
                    Mux(load_fifo_bank == lane, 0xff, 0),
                    Mux(self.wr_en, 1 << wr_sub, 0))),
                self.rd_data[lane].eq(Array([
                    rp.dat_r[32*i:32*(i + 1)] for i in range(8)
                ])[rd_sub_d]),
            ]

        self.sync += [
            If(load_fire,
                load_pending.eq(1),
                load_data_r.eq(self.load_data),
                load_bank_r.eq(load_bank),
                load_addr_r.eq(load_addr),
                load_last_r.eq(self.load_beat == beats - 1),
            ).Elif(load_push,
                load_pending.eq(0),
            ),
            If(self.load_start,
                load_active.eq(1),
                load_writing.eq(1),
                load_pending.eq(0),
                self.load_done.eq(0),
            ).Elif(load_fire,
                If(self.load_beat == beats - 1,
                    load_active.eq(0),
                )
            ),
            If(load_write,
                If(load_fifo_last,
                    load_writing.eq(0),
                    self.load_done.eq(1),
                )
            ),
            rd_pending.eq(self.rd_en),
            If(self.rd_en,
                rd_addr_r.eq(rd_addr),
                rd_sub_r.eq(rd_sub),
            ),
            rd_sub_d.eq(rd_sub_r),
            self.rd_valid.eq(rd_pending & ~dump_active),
            dump_request.eq(dump_issue),
            dump_return.eq(dump_request),
            If(dump_issue,
                dump_addr_r.eq(dump_addr),
                dump_bank_r.eq(dump_bank),
                dump_last_r.eq(dump_count == beats - 1),
            ),
            If(dump_request,
                dump_bank_d.eq(dump_bank_r),
                dump_last_d.eq(dump_last_r),
            ),
            If(dump_issue,
                If(dump_count == beats - 1,
                    dump_active.eq(0),
                ).Else(
                    dump_count.eq(dump_count + 1),
                )
            ),
            If(self.dump_start,
                dump_active.eq(1),
                dump_count.eq(0),
                dump_request.eq(0),
                dump_return.eq(0),
            ),
        ]

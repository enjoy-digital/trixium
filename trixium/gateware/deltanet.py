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

from functools import reduce
from operator import or_

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *

from trixium.gateware.fp32 import BonsaiFP32Add, BonsaiFP32Mul


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
        self.read_latency = 2

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
            # Two reads may be in the registered BRAM/return pipeline. Keep two FIFO slots
            # free for them; with a draining consumer this launches one state beat/cycle.
            dump_issue.eq(dump_active & (dump_fifo.level < 2)),
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


class BonsaiDeltaNetRecurrence(LiteXModule):
    """Eight-lane, operation-order-exact DeltaNet recurrence pipeline.

    The matrix state lives in :attr:`state`; software/integration logic loads and dumps it
    through that object's row-major stream. Parameters are written a scalar at a time before
    ``start``. The datapath computes, for every row, exactly::

        m = round(m * decay)
        d = round(d + round(m * k[i]))
        delta = round(round(v - d) * beta)
        m = round(m + round(delta * k[i]))
        o = round(o + round(m * q[i]))
        out = round(o * scale)

    No FMA or tree reduction is used. Rows are interleaved, so the feedback distance is
    ``rows/lanes`` cycles; the 27B's 16 phases exceed the FP adder's 11-cycle latency.

    ``add_factory``/``mul_factory`` are test hooks. Hardware uses the Xilinx binary32
    operators; a small integer pipeline can exercise the scheduler in the Migen simulator.
    """

    def __init__(self, rows=128, cols=128, lanes=8,
                 add_factory=BonsaiFP32Add, mul_factory=BonsaiFP32Mul,
                 negate=lambda x: x ^ 0x80000000):
        assert rows % lanes == 0
        phases = rows // lanes
        assert phases >= add_factory.latency, \
            "row interleave must cover the accumulator feedback latency"
        issue_cycles = phases * cols

        self.start = Signal()
        self.busy = Signal()
        self.done = Signal()
        self.decay = Signal(32)
        self.beta  = Signal(32)
        self.scale = Signal(32)

        # Parameter write port: kind 0=q, 1=k, 2=v; index is the scalar index.
        self.param_we    = Signal()
        self.param_kind  = Signal(2)
        self.param_index = Signal(max=max(rows, cols))
        self.param_data  = Signal(32)
        # Wide loader used by the scratchpad wrapper: eight adjacent scalars per beat.
        self.param_word_we    = Signal()
        self.param_word_kind  = Signal(2)
        self.param_word_group = Signal(max=max(rows, cols)//8)
        self.param_word_data  = Signal(256)

        self.state = BonsaiDeltaNetState(rows=rows, cols=cols, lanes=lanes)
        self.submodules.state = self.state

        self.out_row_phase = Signal(max=phases)
        self.out_data = Signal(32*lanes)

        # # #

        q = [Signal(32, name=f"dn_q{i}") for i in range(cols)]
        k = [Signal(32, name=f"dn_k{i}") for i in range(cols)]
        v = [[Signal(32, name=f"dn_v{lane}_{p}") for p in range(phases)]
             for lane in range(lanes)]
        acc = [[Signal(32, name=f"dn_acc{lane}_{p}") for p in range(phases)]
               for lane in range(lanes)]
        delta = [[Signal(32, name=f"dn_delta{lane}_{p}") for p in range(phases)]
                 for lane in range(lanes)]
        out = [[Signal(32, name=f"dn_out{lane}_{p}") for p in range(phases)]
               for lane in range(lanes)]
        qa, ka = Array(q), Array(k)
        va = [Array(x) for x in v]
        acca = [Array(x) for x in acc]
        deltaa = [Array(x) for x in delta]
        outa = [Array(x) for x in out]

        self.sync += If(self.param_we,
            If(self.param_kind == 0,
                qa[self.param_index].eq(self.param_data),
            ).Elif(self.param_kind == 1,
                ka[self.param_index].eq(self.param_data),
            ).Else(
                *[If(self.param_index[0:(lanes - 1).bit_length()] == lane,
                    va[lane][self.param_index >> (lanes - 1).bit_length()].eq(
                        self.param_data)) for lane in range(lanes)]
            )
        )
        for i in range(8):
            qi = Signal(max=cols)
            vi = Signal(max=rows)
            self.comb += [
                qi.eq((self.param_word_group << 3) + i),
                vi.eq((self.param_word_group << 3) + i),
            ]
            self.sync += If(self.param_word_we,
                If(self.param_word_kind == 0,
                    qa[qi].eq(self.param_word_data[32*i:32*(i + 1)]),
                ).Elif(self.param_word_kind == 1,
                    ka[qi].eq(self.param_word_data[32*i:32*(i + 1)]),
                ).Else(
                    *[If(vi[0:(lanes - 1).bit_length()] == lane,
                        va[lane][vi >> (lanes - 1).bit_length()].eq(
                            self.param_word_data[32*i:32*(i + 1)]))
                      for lane in range(lanes)]
                )
            )
        self.comb += self.out_data.eq(Cat(*[outa[l][self.out_row_phase]
                                            for l in range(lanes)]))

        add0, add1, mul0, mul1 = [], [], [], []
        for lane in range(lanes):
            for group, factory, name in (
                    (add0, add_factory, f"add0_{lane}"),
                    (add1, add_factory, f"add1_{lane}"),
                    (mul0, mul_factory, f"mul0_{lane}"),
                    (mul1, mul_factory, f"mul1_{lane}")):
                op = factory()
                setattr(self.submodules, name, op)
                group.append(op)
        add_latency = add0[0].latency
        mul_latency = mul0[0].latency

        def delayed(value, latency, name):
            """A metadata delay matching a non-stalling arithmetic pipeline."""
            regs = [Signal.like(value, name=f"{name}_{i}") for i in range(latency)]
            if regs:
                self.sync += regs[0].eq(value)
                for a, b in zip(regs[1:], regs[:-1]):
                    self.sync += a.eq(b)
                return regs[-1]
            return value

        fsm = FSM(reset_state="IDLE")
        self.submodules.fsm = fsm
        p1_mode = Signal()
        delta_mode = Signal()
        p2_mode = Signal()
        scale_mode = Signal()
        self.comb += [
            p1_mode.eq(fsm.ongoing("P1_ISSUE") | fsm.ongoing("P1_DRAIN")),
            delta_mode.eq(fsm.ongoing("DELTA_ISSUE") | fsm.ongoing("DELTA_DRAIN")),
            p2_mode.eq(fsm.ongoing("P2_ISSUE") | fsm.ongoing("P2_DRAIN")),
            scale_mode.eq(fsm.ongoing("SCALE_ISSUE") | fsm.ongoing("SCALE_DRAIN")),
            self.busy.eq(~fsm.ongoing("IDLE")),
        ]

        phase = Signal(max=phases)
        col = Signal(max=cols)
        phase_bits = (phases - 1).bit_length()
        self.comb += [
            self.state.rd_row_phase.eq(phase),
            self.state.rd_col.eq(col),
        ]
        rd_phase = delayed(self.state.rd_row_phase, self.state.read_latency, "rd_phase")
        rd_col   = delayed(self.state.rd_col,       self.state.read_latency, "rd_col")

        # First pass: decay the state, then the sequential k dot.
        p1_m_phase = delayed(rd_phase, mul_latency, "p1_m_phase")
        p1_m_col   = delayed(rd_col,   mul_latency, "p1_m_col")
        p1_m_k     = delayed(ka[rd_col], mul_latency, "p1_m_k")
        p1_k_phase = delayed(p1_m_phase, mul_latency, "p1_k_phase")
        p1_a_phase = delayed(p1_k_phase, add_latency, "p1_a_phase")

        # Delta stage tags.
        delta_add_phase = delayed(phase, add_latency, "delta_add_phase")
        delta_mul_phase = delayed(delta_add_phase, mul_latency, "delta_mul_phase")

        # Second pass: rank-1 update, then the sequential q dot.
        p2_m_phase = delayed(rd_phase, mul_latency, "p2_m_phase")
        p2_m_col   = delayed(rd_col,   mul_latency, "p2_m_col")
        p2_m_state = [delayed(self.state.rd_data[l], mul_latency, f"p2_m_state{l}")
                      for l in range(lanes)]
        p2_m_q     = delayed(qa[rd_col], mul_latency, "p2_m_q")
        p2_u_phase = delayed(p2_m_phase, add_latency, "p2_u_phase")
        p2_u_col   = delayed(p2_m_col,   add_latency, "p2_u_col")
        p2_u_q     = delayed(p2_m_q,     add_latency, "p2_u_q")
        p2_q_phase = delayed(p2_u_phase, mul_latency, "p2_q_phase")
        p2_a_phase = delayed(p2_q_phase, add_latency, "p2_a_phase")

        scale_phase = delayed(phase, mul_latency, "scale_phase")

        # Operator routing. A phase fully drains before the next phase reuses a pipe.
        p1_rd = self.state.rd_valid & p1_mode
        p2_rd = self.state.rd_valid & p2_mode
        for lane in range(lanes):
            self.comb += [
                # mul0: state decay (P1), delta*k (P2), or (v-dot)*beta (DELTA).
                mul0[lane].valid.eq(Mux(delta_mode, add0[lane].result_valid,
                                    Mux(p1_mode, p1_rd, p2_rd))),
                mul0[lane].a.eq(Mux(delta_mode, add0[lane].result,
                                Mux(p1_mode, self.state.rd_data[lane],
                                    deltaa[lane][rd_phase]))),
                mul0[lane].b.eq(Mux(delta_mode, self.beta,
                                Mux(p1_mode, self.decay, ka[rd_col]))),

                # mul1: decayed-state*k (P1), or updated-state*q (P2).
                mul1[lane].valid.eq(Mux(p1_mode, mul0[lane].result_valid,
                                                   add0[lane].result_valid & p2_mode)),
                mul1[lane].a.eq(Mux(p1_mode, mul0[lane].result, add0[lane].result)),
                mul1[lane].b.eq(Mux(p1_mode, p1_m_k, p2_u_q)),

                # add0: k-dot accumulator, v-dot subtraction, or state update.
                add0[lane].valid.eq(Mux(delta_mode,
                    fsm.ongoing("DELTA_ISSUE"),
                    Mux(p1_mode, mul1[lane].result_valid, mul0[lane].result_valid & p2_mode))),
                add0[lane].a.eq(Mux(delta_mode, va[lane][phase],
                    Mux(p1_mode, acca[lane][p1_k_phase], p2_m_state[lane]))),
                add0[lane].b.eq(Mux(delta_mode, negate(acca[lane][phase]),
                    Mux(p1_mode, mul1[lane].result, mul0[lane].result))),

                # add1: q-dot accumulator (P2 only).
                add1[lane].valid.eq(mul1[lane].result_valid & p2_mode),
                add1[lane].a.eq(acca[lane][p2_q_phase]),
                add1[lane].b.eq(mul1[lane].result),
            ]

        # State updates from the two passes.
        self.comb += [
            self.state.wr_en.eq((p1_mode & mul0[0].result_valid) |
                                (p2_mode & add0[0].result_valid)),
            self.state.wr_row_phase.eq(Mux(p1_mode, p1_m_phase, p2_u_phase)),
            self.state.wr_col.eq(Mux(p1_mode, p1_m_col, p2_u_col)),
        ]
        for lane in range(lanes):
            self.comb += self.state.wr_data[lane].eq(
                Mux(p1_mode, mul0[lane].result, add0[lane].result))

        # Group retirement counters (one result_valid represents all eight lanes).
        p1_ret = Signal(max=issue_cycles + 1)
        delta_ret = Signal(max=phases + 1)
        p2_ret = Signal(max=issue_cycles + 1)
        scale_ret = Signal(max=phases + 1)
        delta_done = Signal()
        self.comb += delta_done.eq(delta_mode & mul0[0].result_valid &
                                   (delta_ret == phases - 1))

        # Accumulator/delta/output register writes. Dynamic phase compares infer compact
        # one-hot enables; each lane owns a separate small register bank.
        for lane in range(lanes):
            for p in range(phases):
                self.sync += If(self.start,
                    acc[lane][p].eq(0),
                ).Elif(p1_mode & add0[lane].result_valid & (p1_a_phase == p),
                    acc[lane][p].eq(add0[lane].result),
                ).Elif(delta_done,
                    acc[lane][p].eq(0),
                ).Elif(p2_mode & add1[lane].result_valid & (p2_a_phase == p),
                    acc[lane][p].eq(add1[lane].result),
                )
                self.sync += If(delta_mode & mul0[lane].result_valid &
                                (delta_mul_phase == p),
                    delta[lane][p].eq(mul0[lane].result),
                )
                self.sync += If(scale_mode & mul0[lane].result_valid &
                                (scale_phase == p),
                    out[lane][p].eq(mul0[lane].result),
                )

        self.sync += [
            If(self.start,
                p1_ret.eq(0), delta_ret.eq(0), p2_ret.eq(0), scale_ret.eq(0),
            ).Else(
                If(p1_mode & add0[0].result_valid, p1_ret.eq(p1_ret + 1)),
                If(delta_mode & mul0[0].result_valid, delta_ret.eq(delta_ret + 1)),
                If(p2_mode & add1[0].result_valid, p2_ret.eq(p2_ret + 1)),
                If(scale_mode & mul0[0].result_valid, scale_ret.eq(scale_ret + 1)),
            )
        ]

        def advance_issue(next_state):
            return If(phase == phases - 1,
                NextValue(phase, 0),
                If(col == cols - 1,
                    NextState(next_state),
                ).Else(
                    NextValue(col, col + 1),
                )
            ).Else(
                NextValue(phase, phase + 1),
            )

        fsm.act("IDLE",
            If(self.start,
                NextValue(phase, 0), NextValue(col, 0),
                NextState("P1_ISSUE"),
            )
        )
        fsm.act("P1_ISSUE",
            self.state.rd_en.eq(1),
            advance_issue("P1_DRAIN"),
        )
        fsm.act("P1_DRAIN",
            If(add0[0].result_valid & (p1_ret == issue_cycles - 1),
                NextValue(phase, 0),
                NextState("DELTA_ISSUE"),
            )
        )
        fsm.act("DELTA_ISSUE",
            If(phase == phases - 1,
                NextState("DELTA_DRAIN"),
            ).Else(
                NextValue(phase, phase + 1),
            )
        )
        fsm.act("DELTA_DRAIN",
            If(mul0[0].result_valid & (delta_ret == phases - 1),
                NextValue(phase, 0), NextValue(col, 0),
                NextState("P2_ISSUE"),
            )
        )
        fsm.act("P2_ISSUE",
            self.state.rd_en.eq(1),
            advance_issue("P2_DRAIN"),
        )
        fsm.act("P2_DRAIN",
            If(add1[0].result_valid & (p2_ret == issue_cycles - 1),
                NextValue(phase, 0),
                NextState("SCALE_ISSUE"),
            )
        )
        # Reuse mul0 after P2 has completely drained.
        for lane in range(lanes):
            self.comb += If(scale_mode,
                mul0[lane].valid.eq(fsm.ongoing("SCALE_ISSUE")),
                mul0[lane].a.eq(acca[lane][phase]),
                mul0[lane].b.eq(self.scale),
            )
        fsm.act("SCALE_ISSUE",
            If(phase == phases - 1,
                NextState("SCALE_DRAIN"),
            ).Else(
                NextValue(phase, phase + 1),
            )
        )
        fsm.act("SCALE_DRAIN",
            If(mul0[0].result_valid & (scale_ret == phases - 1),
                self.done.eq(1),
                NextState("IDLE"),
            )
        )


class BonsaiDeltaNetUnit(LiteXModule):
    """CSR/scratchpad front-end for :class:`BonsaiDeltaNetRecurrence`.

    One command consumes q, k, v and the row-major state from scratchpad, runs the exact
    recurrence, then overwrites the state and writes the 128-row result. The interface is
    intentionally the same read/write client shape as ``BonsaiVectorUnit`` so the SoC can
    give this unit priority while it is busy without adding scratchpad ports.
    """

    def __init__(self, core=None, rows=128, cols=128, lanes=8, *, with_dispatch_guard=False):
        if core is None:
            core = BonsaiDeltaNetRecurrence(rows=rows, cols=cols, lanes=lanes)
        self.core = core
        self.submodules.core = core
        rows, cols, lanes = core.state.rows, core.state.cols, core.state.lanes
        qgroups, vgroups = cols//8, rows//8
        phases = rows//lanes
        state_beats = core.state.beats

        self.state_base = CSRStorage(24, description="Scratchpad byte base of row-major M.")
        self.q_base = CSRStorage(24, description="Scratchpad byte base of q[cols].")
        self.k_base = CSRStorage(24, description="Scratchpad byte base of k[cols].")
        self.v_base = CSRStorage(24, description="Scratchpad byte base of v[rows].")
        self.out_base = CSRStorage(24, description="Scratchpad byte base of output[rows].")
        self.decay = CSRStorage(32, description="FP32 decay bits.")
        self.beta  = CSRStorage(32, description="FP32 beta bits.")
        self.scale = CSRStorage(32, description="FP32 1/sqrt(cols) bits.")
        self.submit = CSRStorage(1, description="Write 1 to run one head recurrence.")
        self.status = CSRStatus(fields=[CSRField("busy", size=1)] +
            ([CSRField("rejected", size=1, description="Manual submit while sequencer owns scratchpad; reset required.")]
             if with_dispatch_guard else []))
        self.done_count = CSRStatus(32)

        self.busy = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(13)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(13)
        self.sp_wr_we  = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()
        dispatch_allowed = C(1, 1)
        if with_dispatch_guard:
            self.dispatch_allowed = dispatch_allowed = Signal(reset=1)
            self.sync += If(self.submit.re & ~dispatch_allowed, self.status.fields.rejected.eq(1))

        fsm = FSM(reset_state="IDLE")
        self.submodules.fsm = fsm
        self.comb += [
            self.busy.eq(~fsm.ongoing("IDLE")),
            self.status.fields.busy.eq(self.busy),
            core.decay.eq(self.decay.storage),
            core.beta.eq(self.beta.storage),
            core.scale.eq(self.scale.storage),
        ]

        issue = Signal(12)
        response = Signal(12)
        rd_pending = Signal()
        rd_kind = Signal(2)
        rd_index = Signal(12)
        load_kind = Signal(2)
        load_limit = Signal(12)
        load_base = Signal(13)
        loading = Signal()
        self.comb += [
            loading.eq(fsm.ongoing("LOAD_Q") | fsm.ongoing("LOAD_K") |
                       fsm.ongoing("LOAD_V") | fsm.ongoing("LOAD_STATE")),
            load_kind.eq(Mux(fsm.ongoing("LOAD_Q"), 0,
                         Mux(fsm.ongoing("LOAD_K"), 1,
                         Mux(fsm.ongoing("LOAD_V"), 2, 3)))),
            load_limit.eq(Mux(fsm.ongoing("LOAD_Q") | fsm.ongoing("LOAD_K"), qgroups,
                          Mux(fsm.ongoing("LOAD_V"), vgroups, state_beats))),
            load_base.eq(Mux(fsm.ongoing("LOAD_Q"), self.q_base.storage >> 5,
                         Mux(fsm.ongoing("LOAD_K"), self.k_base.storage >> 5,
                         Mux(fsm.ongoing("LOAD_V"), self.v_base.storage >> 5,
                             self.state_base.storage >> 5)))),
            self.sp_rd_req.eq(loading & (issue < load_limit) &
                              (~fsm.ongoing("LOAD_STATE") | core.state.load_ready)),
            self.sp_rd_adr.eq(load_base + issue),
            core.param_word_we.eq(rd_pending & (rd_kind != 3)),
            core.param_word_kind.eq(rd_kind),
            core.param_word_group.eq(rd_index),
            core.param_word_data.eq(self.sp_rd_dat),
            core.state.load_valid.eq(rd_pending & (rd_kind == 3)),
            core.state.load_beat.eq(rd_index),
            core.state.load_data.eq(self.sp_rd_dat),
        ]
        rd_return = Signal()
        self.comb += rd_return.eq(rd_pending)
        self.sync += [
            rd_pending.eq(self.sp_rd_req),
            If(self.sp_rd_req,
                rd_kind.eq(load_kind),
                rd_index.eq(issue),
                issue.eq(issue + 1),
            ),
            If(rd_return, response.eq(response + 1)),
        ]

        wr_count = Signal(12)
        state_fire = Signal()
        out_fire = Signal()
        self.comb += [
            state_fire.eq(fsm.ongoing("WRITE_STATE") & core.state.source.valid &
                          self.sp_wr_gnt),
            out_fire.eq(fsm.ongoing("WRITE_OUT") & self.sp_wr_gnt),
            core.state.source.ready.eq(fsm.ongoing("WRITE_STATE") & self.sp_wr_gnt),
            core.out_row_phase.eq(wr_count[:max(1, (phases - 1).bit_length())]),
            If(fsm.ongoing("WRITE_STATE"),
                self.sp_wr_req.eq(core.state.source.valid),
                self.sp_wr_adr.eq((self.state_base.storage >> 5) + wr_count),
                self.sp_wr_we.eq(0xff),
                self.sp_wr_dat.eq(core.state.source.data),
            ).Elif(fsm.ongoing("WRITE_OUT"),
                self.sp_wr_req.eq(1),
                self.sp_wr_adr.eq((self.out_base.storage >> 5) + wr_count),
                self.sp_wr_we.eq(0xff),
                self.sp_wr_dat.eq(core.out_data),
            )
        ]
        self.sync += If(state_fire | out_fire, wr_count.eq(wr_count + 1))

        def begin_load(state):
            return [NextValue(issue, 0), NextValue(response, 0), NextState(state)]

        fsm.act("IDLE",
            If(self.submit.re & dispatch_allowed, *begin_load("LOAD_Q"))
        )
        for name, next_name in (("LOAD_Q", "LOAD_K"), ("LOAD_K", "LOAD_V"),
                                ("LOAD_V", "STATE_INIT")):
            fsm.act(name,
                If(rd_return & (response == load_limit - 1), *begin_load(next_name))
            )
        fsm.act("STATE_INIT",
            core.state.load_start.eq(1),
            *begin_load("LOAD_STATE"),
        )
        fsm.act("LOAD_STATE",
            If(rd_return & (response == state_beats - 1),
                NextState("STATE_DRAIN"),
            )
        )
        fsm.act("STATE_DRAIN",
            If(core.state.load_done, NextState("CORE_START"))
        )
        fsm.act("CORE_START",
            core.start.eq(1),
            NextState("RUN"),
        )
        fsm.act("RUN",
            If(core.done, NextState("DUMP_INIT"))
        )
        fsm.act("DUMP_INIT",
            core.state.dump_start.eq(1),
            NextValue(wr_count, 0),
            NextState("WRITE_STATE"),
        )
        fsm.act("WRITE_STATE",
            If(state_fire & (wr_count == state_beats - 1),
                NextValue(wr_count, 0),
                NextState("WRITE_OUT"),
            )
        )
        fsm.act("WRITE_OUT",
            If(out_fire & (wr_count == phases - 1),
                NextValue(self.done_count.status, self.done_count.status + 1),
                NextState("IDLE"),
            )
        )


class BonsaiScratchpadClientMux(LiteXModule):
    """Give an auxiliary unit priority over the vector-unit scratchpad client."""

    def __init__(self, vector, auxiliary):
        self.busy = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal.like(vector.sp_rd_adr)
        self.sp_rd_dat = Signal.like(vector.sp_rd_dat)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal.like(vector.sp_wr_adr)
        self.sp_wr_we  = Signal.like(vector.sp_wr_we)
        self.sp_wr_dat = Signal.like(vector.sp_wr_dat)
        self.sp_wr_gnt = Signal()
        sel = auxiliary.busy
        self.comb += [
            self.busy.eq(vector.busy | auxiliary.busy),
            self.sp_rd_req.eq(Mux(sel, auxiliary.sp_rd_req, vector.sp_rd_req)),
            self.sp_rd_adr.eq(Mux(sel, auxiliary.sp_rd_adr, vector.sp_rd_adr)),
            vector.sp_rd_dat.eq(self.sp_rd_dat),
            auxiliary.sp_rd_dat.eq(self.sp_rd_dat),
            self.sp_wr_req.eq(Mux(sel, auxiliary.sp_wr_req, vector.sp_wr_req)),
            self.sp_wr_adr.eq(Mux(sel, auxiliary.sp_wr_adr, vector.sp_wr_adr)),
            self.sp_wr_we.eq(Mux(sel, auxiliary.sp_wr_we, vector.sp_wr_we)),
            self.sp_wr_dat.eq(Mux(sel, auxiliary.sp_wr_dat, vector.sp_wr_dat)),
            vector.sp_wr_gnt.eq(self.sp_wr_gnt & ~sel),
            auxiliary.sp_wr_gnt.eq(self.sp_wr_gnt & sel),
        ]


class BonsaiScratchpadArbiter(LiteXModule):
    """Flat priority arbiter for the shared scratchpad client port.

    Functionally identical to nesting BonsaiScratchpadClientMux once per auxiliary unit
    (base = the vector unit, clients in the order they were attached, the LAST attached
    busy client wins, else the base), with the same combinational latency: requests, the
    write grant and the broadcast read data behave exactly as through the chain. What
    changes is depth. The chain puts one serial 2:1 mux per client on the BRAM address,
    data and enable pins and one AND per client on the grant back; with seven clients the
    v11 VCU build failed setup at -1.519 ns on exactly those paths (docs/vcu-clock-
    qualification.md). Here each client gets a one-hot select from a priority reduction
    over the busy bits and the outputs are one AND-OR level, so depth grows with log2 of
    the client count instead of linearly.
    """

    def __init__(self, base, clients):
        clients = list(clients)
        self.busy = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal.like(base.sp_rd_adr)
        self.sp_rd_dat = Signal.like(base.sp_rd_dat)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal.like(base.sp_wr_adr)
        self.sp_wr_we  = Signal.like(base.sp_wr_we)
        self.sp_wr_dat = Signal.like(base.sp_wr_dat)
        self.sp_wr_gnt = Signal()
        self.aux_busy  = Signal()  # Any attached (non-base) client busy.

        # One-hot select: client i wins when busy and no LATER client is busy.
        n = len(clients)
        later = [Signal(name=f"arb_later{i}") for i in range(n)]
        sel = [Signal(name=f"arb_sel{i}") for i in range(n)]
        base_sel = Signal()
        for i in range(n):
            self.comb += later[i].eq(reduce(or_, [c.busy for c in clients[i + 1:]], C(0, 1)))
            self.comb += sel[i].eq(clients[i].busy & ~later[i])
        any_aux = reduce(or_, [c.busy for c in clients], C(0, 1))
        self.comb += [
            self.aux_busy.eq(any_aux),
            base_sel.eq(~any_aux),
            self.busy.eq(base.busy | any_aux),
        ]

        def pick(field):
            terms = [Mux(base_sel, getattr(base, field), 0)]
            terms += [Mux(sel[i], getattr(clients[i], field), 0) for i in range(n)]
            return reduce(or_, terms)

        for field in ("sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat"):
            self.comb += getattr(self, field).eq(pick(field))
        self.comb += [base.sp_rd_dat.eq(self.sp_rd_dat), base.sp_wr_gnt.eq(self.sp_wr_gnt & base_sel)]
        for i, c in enumerate(clients):
            self.comb += [c.sp_rd_dat.eq(self.sp_rd_dat), c.sp_wr_gnt.eq(self.sp_wr_gnt & sel[i])]

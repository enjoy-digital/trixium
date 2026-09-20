#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

from migen import *
from migen.fhdl.tools import list_targets
from migen.sim import run_simulation

from litex.gen import *

from trixium.gateware.deltanet import (
    BonsaiDeltaNetState, BonsaiDeltaNetRecurrence, BonsaiDeltaNetUnit,
)


class _IntAdd(LiteXModule):
    """Integer test operator with the real FP adder's pipeline latency."""
    latency = 11

    def __init__(self):
        self.a = Signal(32)
        self.b = Signal(32)
        self.valid = Signal()
        self.result = Signal(32)
        self.result_valid = Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a + self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]


class _IntMul(LiteXModule):
    latency = 8

    def __init__(self):
        self.a = Signal(32)
        self.b = Signal(32)
        self.valid = Signal()
        self.result = Signal(32)
        self.result_valid = Signal()
        data = [Signal(32) for _ in range(self.latency)]
        valid = [Signal() for _ in range(self.latency)]
        self.sync += [data[0].eq(self.a * self.b), valid[0].eq(self.valid)]
        for i in range(1, self.latency):
            self.sync += [data[i].eq(data[i - 1]), valid[i].eq(valid[i - 1])]
        self.comb += [self.result.eq(data[-1]), self.result_valid.eq(valid[-1])]


def test_deltanet_done_has_one_driver_domain():
    """Vivado must not discard the completion pulse as a multi-driven net."""
    dut = BonsaiDeltaNetRecurrence(
        rows=96, cols=16, lanes=8,
        add_factory=_IntAdd, mul_factory=_IntMul,
        negate=lambda x: -x,
    )
    fragment = dut.get_fragment()
    assert dut.done in list_targets(fragment.comb)
    assert all(dut.done not in list_targets(statements)
               for statements in fragment.sync.values())


def test_deltanet_state_transpose_update_and_dump_backpressure():
    rows, cols, lanes = 32, 32, 8
    dut = BonsaiDeltaNetState(rows=rows, cols=cols, lanes=lanes)
    original = [[1000*r + c for c in range(cols)] for r in range(rows)]
    beats = []
    for r in range(rows):
        for c0 in range(0, cols, 8):
            beats.append(sum(original[r][c0 + i] << (32*i) for i in range(8)))
    dumped = []

    def gen():
        # Row-major stream -> banked state.
        yield dut.load_start.eq(1)
        yield
        yield dut.load_start.eq(0)
        for bi, word in enumerate(beats):
            yield dut.load_beat.eq(bi)
            yield dut.load_data.eq(word)
            yield dut.load_valid.eq(1)
            yield
            for _ in range(20):
                if (yield dut.load_ready):
                    break
                yield
            else:
                raise AssertionError(f"load ready timeout at beat {bi}")
        yield dut.load_valid.eq(0)
        for _ in range(1000):
            if (yield dut.load_done):
                break
            yield
        else:
            raise AssertionError("load timeout")
        yield
        for lane in range(lanes):
            expect = sum(original[lane][i] << (32*i) for i in range(8))
            got = (yield dut.memories[lane][0])
            unpack = lambda x: [(x >> (32*i)) & 0xffffffff for i in range(8)]
            assert got == expect, (lane, unpack(got), unpack(expect))
        last = sum(original[-1][-8 + i] << (32*i) for i in range(8))
        got_last = (yield dut.memories[-1][dut.rows_per_lane*dut.groups - 1])
        assert got_last == last, (unpack(got_last), unpack(last))

        # Every compute read returns one column from eight consecutive modulo-banked rows.
        for phase, col in ((0, 0), (1, 7), (2, 18), (3, 31)):
            yield dut.rd_row_phase.eq(phase)
            yield dut.rd_col.eq(col)
            yield dut.rd_en.eq(1)
            yield
            yield dut.rd_en.eq(0)
            while not (yield dut.rd_valid):
                yield
            for lane in range(lanes):
                assert (yield dut.rd_data[lane]) == original[phase*lanes + lane][col]
            yield

        # Update the same selected column in all eight banks.
        phase, col = 2, 13
        yield dut.wr_row_phase.eq(phase)
        yield dut.wr_col.eq(col)
        for lane in range(lanes):
            yield dut.wr_data[lane].eq(0x50000000 + lane)
            original[phase*lanes + lane][col] = 0x50000000 + lane
        yield dut.wr_en.eq(1)
        yield
        yield dut.wr_en.eq(0)

        # Dump after sustained backpressure. The output must return to the original row-major
        # layout and hold data stable whenever ready is low.
        yield dut.dump_start.eq(1)
        yield
        yield dut.dump_start.eq(0)
        yield dut.source.ready.eq(0)
        for _ in range(10):
            yield
        assert (yield dut.source.valid)
        held = (yield dut.source.data)
        for _ in range(4):
            yield
            assert (yield dut.source.valid)
            assert (yield dut.source.data) == held

        yield dut.source.ready.eq(1)
        for _ in range(4):
            yield
            if (yield dut.source.data) != held:
                break
        else:
            raise AssertionError("held dump beat was not consumed")
        dumped.append(held)  # Handshake of the beat held during backpressure.
        cycle = 0
        while len(dumped) < len(beats):
            valid = (yield dut.source.valid)
            word = (yield dut.source.data)
            yield
            if valid:
                dumped.append(word)
            cycle += 1
            assert cycle < 2000

    run_simulation(dut, gen())

    expected = []
    for r in range(rows):
        for c0 in range(0, cols, 8):
            expected.append(sum(original[r][c0 + i] << (32*i) for i in range(8)))
    assert dumped == expected


def test_deltanet_recurrence_scheduler_preserves_row_order():
    # Integer operators test the scheduler independently of the vendor FP simulation model.
    # k=[1,1,0...], q=[1,1,0...] gives a compact closed-form oracle while exercising both
    # sequential dots and both state updates.
    rows, cols, lanes = 96, 16, 8
    dut = BonsaiDeltaNetRecurrence(
        rows=rows, cols=cols, lanes=lanes,
        add_factory=_IntAdd, mul_factory=_IntMul,
        negate=lambda x: -x,
    )
    state = [[1 + (r % 3) + (c % 2) for c in range(cols)] for r in range(rows)]
    kval = [1, 1] + [0]*(cols - 2)
    qval = [1, 1] + [0]*(cols - 2)
    vval = [100 + r for r in range(rows)]
    expected = [row[:] for row in state]
    expected_out = []
    for r, row in enumerate(expected):
        d = vval[r] - (row[0] + row[1])
        row[0] += d
        row[1] += d
        expected_out.append(row[0] + row[1])

    beats = []
    for r in range(rows):
        for c0 in range(0, cols, 8):
            beats.append(sum(state[r][c0 + i] << (32*i) for i in range(8)))
    result = {}

    def write_param(kind, index, value):
        yield dut.param_kind.eq(kind)
        yield dut.param_index.eq(index)
        yield dut.param_data.eq(value & 0xffffffff)
        yield dut.param_we.eq(1)
        yield
        yield dut.param_we.eq(0)
        yield

    def gen():
        yield dut.state.load_start.eq(1)
        yield
        yield dut.state.load_start.eq(0)
        for bi, word in enumerate(beats):
            yield dut.state.load_beat.eq(bi)
            yield dut.state.load_data.eq(word)
            yield dut.state.load_valid.eq(1)
            yield
            while not (yield dut.state.load_ready):
                yield
        yield dut.state.load_valid.eq(0)
        for _ in range(1000):
            if (yield dut.state.load_done):
                break
            yield
        else:
            raise AssertionError("state load timeout")

        for i, x in enumerate(qval):
            yield from write_param(0, i, x)
        for i, x in enumerate(kval):
            yield from write_param(1, i, x)
        for i, x in enumerate(vval):
            yield from write_param(2, i, x)
        yield dut.decay.eq(1)
        yield dut.beta.eq(1)
        yield dut.scale.eq(1)
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        for cycles in range(4000):
            if (yield dut.done):
                result["cycles"] = cycles
                break
            yield
        else:
            raise AssertionError("recurrence timeout")
        yield

        got = [[0]*cols for _ in range(rows)]
        groups = cols//8
        for r in range(rows):
            bank, phase = r % lanes, r//lanes
            for g in range(groups):
                word = (yield dut.state.memories[bank][phase*groups + g])
                for i in range(8):
                    got[r][8*g + i] = (word >> (32*i)) & 0xffffffff
        result["state"] = got
        outs = [0]*rows
        for phase in range(rows//lanes):
            yield dut.out_row_phase.eq(phase)
            yield
            word = (yield dut.out_data)
            for lane in range(lanes):
                outs[phase*lanes + lane] = (word >> (32*lane)) & 0xffffffff
        result["out"] = outs

    run_simulation(dut, gen())
    assert result["state"] == expected
    assert result["out"] == expected_out
    assert result["cycles"] < 2000


def test_deltanet_scratchpad_unit_moves_a_complete_head():
    rows, cols, lanes = 96, 16, 8
    core = BonsaiDeltaNetRecurrence(
        rows=rows, cols=cols, lanes=lanes,
        add_factory=_IntAdd, mul_factory=_IntMul,
        negate=lambda x: -x,
    )
    dut = BonsaiDeltaNetUnit(core=core)
    state = [[1 + (r % 3) + (c % 2) for c in range(cols)] for r in range(rows)]
    kval = [1, 1] + [0]*(cols - 2)
    qval = [1, 1] + [0]*(cols - 2)
    vval = [100 + r for r in range(rows)]
    expected = [row[:] for row in state]
    expected_out = []
    for r, row in enumerate(expected):
        d = vval[r] - row[0] - row[1]
        row[0] += d
        row[1] += d
        expected_out.append(row[0] + row[1])

    qbase, kbase, vbase, sbase, obase = 0x000, 0x040, 0x080, 0x400, 0x2000
    depth = 512
    init = [0]*depth

    def put_words(base, values):
        for g in range((len(values) + 7)//8):
            xs = values[8*g:8*g + 8] + [0]*8
            init[base//32 + g] = sum((xs[i] & 0xffffffff) << (32*i) for i in range(8))

    put_words(qbase, qval)
    put_words(kbase, kval)
    put_words(vbase, vval)
    flat_state = [x for row in state for x in row]
    put_words(sbase, flat_state)

    mem = Memory(256, depth, init=init)
    rp = mem.get_port(has_re=True)
    wp = mem.get_port(write_capable=True, we_granularity=32)
    dut.specials += mem, rp, wp
    dut.comb += [
        rp.adr.eq(dut.sp_rd_adr),
        rp.re.eq(dut.sp_rd_req),
        dut.sp_rd_dat.eq(rp.dat_r),
        dut.sp_wr_gnt.eq(1),
        wp.adr.eq(dut.sp_wr_adr),
        wp.we.eq(dut.sp_wr_we),
        wp.dat_w.eq(dut.sp_wr_dat),
    ]
    result = {}

    def gen():
        for csr, value in ((dut.q_base, qbase), (dut.k_base, kbase),
                           (dut.v_base, vbase), (dut.state_base, sbase),
                           (dut.out_base, obase), (dut.decay, 1),
                           (dut.beta, 1), (dut.scale, 1)):
            yield csr.storage.eq(value)
        yield dut.submit.re.eq(1)
        yield
        yield dut.submit.re.eq(0)
        for cycles in range(6000):
            if (yield dut.done_count.status) == 1:
                result["cycles"] = cycles
                break
            yield
        else:
            raise AssertionError("scratchpad recurrence timeout")
        yield
        got = []
        for i in range(rows*cols):
            word = (yield mem[sbase//32 + i//8])
            got.append((word >> (32*(i % 8))) & 0xffffffff)
        result["state"] = [got[r*cols:(r + 1)*cols] for r in range(rows)]
        outs = []
        for i in range(rows):
            word = (yield mem[obase//32 + i//8])
            outs.append((word >> (32*(i % 8))) & 0xffffffff)
        result["out"] = outs

    run_simulation(dut, gen())
    assert result["state"] == expected
    assert result["out"] == expected_out
    assert result["cycles"] < 3000

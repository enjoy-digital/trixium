#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

from migen.sim import run_simulation

from trixium.gateware.deltanet import BonsaiDeltaNetState


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

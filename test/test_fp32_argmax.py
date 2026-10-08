# SPDX-License-Identifier: BSD-2-Clause
"""FP32ArgmaxTracker against the firmware's greedy argmax loop."""
import numpy as np
import pytest
from migen import run_simulation

from trixium.units.fp32 import FP32ArgmaxTracker

f32 = np.float32


def cpu_argmax(x):
    best, idx = x[0], 0
    for i in range(1, len(x)):
        if x[i] > best:
            best, idx = x[i], i
    return idx


def run_tracker(values, chunks):
    """Feed `values` as consecutive commands of the given element counts (partial last beats)."""
    dut = FP32ArgmaxTracker()
    bits = np.asarray(values, f32).view(np.uint32)
    out = {}
    def gen():
        yield dut.clear.eq(1); yield
        yield dut.clear.eq(0)
        pos = 0
        for n in chunks:
            for b in range(0, n, 8):
                lanes = min(8, n - b)
                word = sum(int(bits[pos + b + j]) << (32*j) for j in range(lanes))
                yield dut.valid.eq(1); yield dut.data.eq(word); yield dut.mask.eq((1 << lanes) - 1)
                yield
                yield dut.valid.eq(0)
                if b % 3 == 0:
                    yield                               # gaps between beats
            pos += n
            # a whole chunk ends on a beat boundary in the unit (one command per chunk)
        for _ in range(8):
            yield
        out["idx"] = (yield dut.best_idx)
        out["valid"] = (yield dut.best_valid)
    run_simulation(dut, gen())
    return out


@pytest.mark.parametrize("case", ["random", "ties", "zeros", "nan_first", "nan_inside", "neg"])
def test_tracker_equals_the_cpu_loop(case):
    rng = np.random.default_rng(hash(case) % 1000)
    x = rng.normal(0, 3, 200).astype(f32)
    if case == "ties":
        x[:] = f32(1.5); x[37] = f32(2.0); x[150] = f32(2.0)
    elif case == "zeros":
        x[:] = f32(-1); x[5] = f32(-0.0); x[9] = f32(0.0); x[100] = f32(-0.0)
    elif case == "nan_first":
        x[0] = np.nan
    elif case == "nan_inside":
        x[17] = np.nan; x[18] = f32(1e30); x[64] = np.nan
    elif case == "neg":
        x = -np.abs(x) - 1
    chunks = [48, 64, 88]                      # element counts of consecutive commands (8-aligned)
    got = run_tracker(x, chunks)
    assert got["valid"] == 1
    assert got["idx"] == cpu_argmax(x), (case, got["idx"], cpu_argmax(x))


def test_tracker_partial_last_beat():
    x = np.full(19, f32(-5)); x[18] = f32(7)
    assert run_tracker(x, [19])["idx"] == 18


def test_tracker_partial_beats_between_commands():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 60).astype(f32); x[45] = f32(9)
    assert run_tracker(x, [13, 21, 26])["idx"] == 45

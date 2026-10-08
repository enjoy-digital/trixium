# SPDX-License-Identifier: BSD-2-Clause
"""Independent dense integer oracle; no floating-point butterfly reference."""
import random
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from migen import run_simulation
from migen.fhdl import verilog

from trixium.gateware.hadamard import IntegerHadamard


def dense(values, signs, inverse=False):
    size = len(values)
    h = np.array([[1 - 2*((i & j).bit_count() & 1) for j in range(size)]
                  for i in range(size)], dtype=np.int64)
    x, s = np.asarray(values, dtype=np.int64), np.asarray(signs, dtype=np.int64)
    return ((h @ x)*s if inverse else h @ (s*x)).tolist()


def packed(values, width):
    return sum((int(value) & ((1 << width) - 1)) << (i*width) for i, value in enumerate(values))


def unpacked(value, width, count):
    words = [(value >> (i*width)) & ((1 << width) - 1) for i in range(count)]
    return [word - (1 << width) if word & (1 << (width-1)) else word for word in words]


def begin(dut, inverse=False):
    yield dut.cmd.inverse.eq(inverse)
    yield dut.cmd.valid.eq(1)
    yield
    assert (yield dut.cmd.ready)
    yield dut.cmd.valid.eq(0)
    yield
    assert (yield dut.busy)


def command(dut, values, signs, inverse=False, stalls=True, seed=7):
    rng = random.Random(seed)
    expected = dense(values, signs, inverse)
    yield dut.sink.valid.eq(0)
    yield dut.source.ready.eq(0)
    yield from begin(dut, inverse)
    sent, received, held = 0, [], None
    lanes, count = dut.lanes, len(values)//dut.lanes
    offered = False
    transfer_cycles = []
    for cycle in range(10*len(values) + 100):
        # Changing the command payload while busy must not affect the latched mode.
        yield dut.cmd.inverse.eq(not inverse)
        # Withdraw the extra offer once the final result is transferred; leaving
        # valid high after returning to IDLE would legitimately start a new block.
        yield dut.cmd.valid.eq(len(received) < len(values))
        if sent < count:
            offered = offered or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offered)
            yield dut.sink.data.eq(packed(values[sent*lanes:(sent+1)*lanes], dut.input_width))
            yield dut.sink.negate.eq(sum((s < 0) << i for i, s in enumerate(signs[sent*lanes:(sent+1)*lanes])))
            yield dut.sink.last.eq(sent == count - 1)
        else:
            yield dut.sink.valid.eq(0)
        yield dut.source.ready.eq(not stalls or rng.randrange(3) != 0)
        yield
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            sent += 1
            offered = False
        valid, ready = (yield dut.source.valid), (yield dut.source.ready)
        current = (yield dut.source.data), (yield dut.source.first), (yield dut.source.last)
        if held is not None:
            assert valid and current == held, "output changed while stalled"
        held = current if valid and not ready else None
        if valid and ready:
            assert current[1] == (not received)
            assert current[2] == (len(received) + lanes == len(values))
            received.extend(unpacked(current[0], dut.output_width, lanes))
            transfer_cycles.append(cycle)
        if (yield dut.done):
            assert not (yield dut.busy)
            assert not (yield dut.error)
            assert not (yield dut.cancelled)
            assert received == expected
            assert (yield dut.absmax) == max(map(abs, expected))
            yield dut.cmd.valid.eq(0)
            yield dut.sink.valid.eq(0)
            yield
            assert not (yield dut.done)
            assert not (yield dut.source.valid)
            return transfer_cycles
        assert not (yield dut.cmd.ready), "second command accepted while busy"
    raise AssertionError(f"stalled: sent {sent} words, received {len(received)} values")


@pytest.mark.parametrize("size,lanes,width", [(4, 2, 8), (4, 2, 32), (16, 2, 8), (16, 8, 8),
                                               (128, 4, 16), (1024, 8, 8)])
@pytest.mark.parametrize("inverse", [False, True])
def test_integer_hadamard_dense_equivalence(size, lanes, width, inverse):
    rng = random.Random(size + width)
    values = [rng.randrange(-(1 << (width-1)), 1 << (width-1)) for _ in range(size)]
    values[0] = -(1 << (width-1))
    signs = [rng.choice([-1, 1]) for _ in range(size)]
    dut = IntegerHadamard(size, lanes, width)
    run_simulation(dut, command(dut, values, signs, inverse))


def test_integer_hadamard_extremes_reuse_and_dense_rate():
    dut = IntegerHadamard()
    size, lanes = dut.block_size, dut.lanes

    def run():
        for values, signs, inverse in [
            ([0]*size, [-1]*size, False),
            ([-128]*size, [-1]*size, False),  # +131072: requires the extra sign bit.
            ([-128]*size, [-1]*size, True),
            ([127]*size, [1 if i % 2 else -1 for i in range(size)], False),
        ]:
            cycles = yield from command(dut, values, signs, inverse, stalls=False)
            assert all(b == a + 1 for a, b in zip(cycles, cycles[1:]))
            # 128 input cycles + 10*(64 issue + 1 drain) + one output-read cycle.
            assert cycles[0] <= size//lanes + 10*(size//(2*lanes) + 1) + 2
    run_simulation(dut, run())


def test_integer_hadamard_matches_dense_engine_rotation_contract():
    from trixium.model import bitexact as bx

    size = 1024
    rng = np.random.default_rng(2714)
    signs = rng.choice([-1, 1], size)
    h = np.array([[1 - 2*((i & j).bit_count() & 1) for j in range(size)]
                  for i in range(size)], dtype=np.int64)
    codes = (h*signs[None, :] + 1).astype(np.uint8)
    scales = np.full((size, size//128), 1/32, dtype=np.float16)
    _, exponent = bx.fp16_decompose(scales)
    bias = bx.choose_scale_bias(exponent)
    shift = bx.choose_result_shift(scales, bias)
    assert (bias, shift) == (27, 8)
    dut = IntegerHadamard()

    def run():
        # The existing quantizer uses [-127,127]. -128 is safe in the new raw
        # core but not necessarily after the legacy h<<14 int32 presentation.
        for values in [np.zeros(size, dtype=np.int8),
                       rng.integers(-127, 128, size, dtype=np.int8),
                       (127*signs).astype(np.int8)]:
            raw = h @ (signs*values.astype(np.int64))
            engine = bx.matvec_ternary(scales, codes, values, bias, shift)
            assert np.array_equal(engine.astype(np.int64), raw << 14)
            yield from command(dut, values.tolist(), signs.tolist(), stalls=False)
    run_simulation(dut, run())


@pytest.mark.parametrize("last_word", [0, None])
def test_integer_hadamard_rejects_early_or_missing_last_and_cancel_recovers(last_word):
    dut = IntegerHadamard(16, 4)

    def run():
        yield from begin(dut)
        for word in range(4):
            yield dut.sink.valid.eq(1)
            yield dut.sink.data.eq(0)
            yield dut.sink.last.eq(word == last_word)
            yield
            if word == last_word:
                break
        yield dut.sink.valid.eq(0)
        for _ in range(5):
            yield
            assert (yield dut.busy) and (yield dut.error)
            assert not (yield dut.cmd.ready)
            assert not (yield dut.source.valid) and not (yield dut.done)
        yield dut.cancel.eq(1)
        yield
        yield dut.cancel.eq(0)
        yield
        assert not (yield dut.busy) and (yield dut.cancelled)
        yield from command(dut, list(range(16)), [-1, 1]*8)
    run_simulation(dut, run())


@pytest.mark.parametrize("stop_cycle", [0, 2, 5, 10, 20, 30])
def test_integer_hadamard_cancel_flushes_loading_compute_and_stalled_output(stop_cycle):
    dut = IntegerHadamard(16, 4)

    def run():
        yield from begin(dut, True)
        for cycle in range(stop_cycle):
            yield dut.sink.valid.eq(cycle < 4)
            yield dut.sink.last.eq(cycle == 3)
            yield dut.sink.data.eq(0x80808080)
            yield dut.sink.negate.eq(15)
            yield dut.source.ready.eq(0)
            yield
        yield dut.cancel.eq(1)
        yield
        assert not (yield dut.source.valid)
        assert not (yield dut.sink.ready) and not (yield dut.cmd.ready)
        yield dut.cancel.eq(0)
        yield dut.sink.valid.eq(0)
        yield
        assert not (yield dut.busy) and (yield dut.cancelled)
        for _ in range(3):
            yield
            assert not (yield dut.source.valid) and not (yield dut.done)
        yield from command(dut, [-128, 127]*8, [1, -1]*8, inverse=False)
    run_simulation(dut, run())


@pytest.mark.parametrize("args", [(3, 2, 8), (16, 3, 8), (16, 16, 8), (16, 2, 1), (16, 2, 33)])
def test_integer_hadamard_parameter_validation(args):
    with pytest.raises(ValueError):
        IntegerHadamard(*args)


def test_compiled_integer_hadamard(tmp_path):
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.skip("Verilator required for compiled arithmetic/memory check")
    dut = IntegerHadamard()
    ios = set(dut.cmd.flatten() + dut.sink.flatten() + dut.source.flatten()) | {
        dut.cancel, dut.busy, dut.done, dut.error, dut.cancelled, dut.absmax}
    rtl = tmp_path / "integer_hadamard.v"
    verilog.convert(dut, ios=ios, name="integer_hadamard").write(str(rtl))
    obj = tmp_path / "obj"
    harness = Path(__file__).parent / "rtl/integer_hadamard.cpp"
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "2", "-Wno-fatal",
                            "--Mdir", str(obj), "--top-module", "integer_hadamard",
                            str(rtl), str(harness.resolve())], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout[-8000:] + built.stderr[-8000:]
    result = subprocess.run([str(obj / "Vinteger_hadamard")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HADAMARD_COMPILED_PASS" in result.stdout
    print(result.stdout)

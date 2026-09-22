# SPDX-License-Identifier: BSD-2-Clause
import random
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from migen import run_simulation
from migen.fhdl import verilog

from trixium.gateware.native_weights import NativeBlockDecoder, NativeWeightStream, PTQ1_0, Q8_0
from trixium.model import qformats

FIELDS = ('codes', 'scale', 'kind', 'error', 'row', 'group', 'row_last', 'first', 'last')


def decoded(block, encoding, row, group, groups, first, last):
    if encoding == PTQ1_0:
        _, values = qformats.ptq1_0_unpack(block)
        codes = sum(int(code) << (2*i) for i, code in enumerate(values[0]))
        scale = int.from_bytes(block[26:28], 'little')
    else:
        codes = int.from_bytes(block[2:34], 'little')
        scale = int.from_bytes(block[:2], 'little')
    return (codes, scale, int(encoding == Q8_0), 0, row, group, int(group == groups - 1), first, last)


def sample(source):
    values = []
    for name in FIELDS:
        values.append((yield getattr(source, name)))
    return tuple(values)


def test_block_decoder_all_byte_values_and_scale_bits_with_stalls():
    rng = random.Random(71)
    blocks, expected = [], []
    special_scales = (0, 0x8000, 1, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e01)
    for i in range(256):
        # Every packed byte sees all 256 possible values, including noncanonical
        # encodings. Compare against the independent NumPy/C-reference mapping.
        ptq = bytes((i+j) % 256 for j in range(26)) + special_scales[i % 8].to_bytes(2, 'little')
        q8 = special_scales[i % 8].to_bytes(2, 'little') + bytes((i+j) % 256 for j in range(32))
        for encoding, block in [(PTQ1_0, ptq), (Q8_0, q8)]:
            blocks.append((encoding, block, i))
            expected.append(decoded(block, encoding, i, 0, 1, i == 0, i == 255))
    blocks.append((255, bytes(34), 0))
    expected.append((0, 0, 0, 1, 0, 0, 1, 1, 0))
    dut = NativeBlockDecoder()

    def run():
        sent, got, held, offered = 0, [], None, False
        for cycle in range(10000):
            if sent < len(blocks):
                offered = offered or rng.randrange(4) != 0
                encoding, block, row = blocks[sent]
                yield dut.sink.valid.eq(offered)
                yield dut.sink.data.eq(int.from_bytes(block, 'little'))
                yield dut.sink.encoding.eq(encoding)
                yield dut.sink.row.eq(row)
                yield dut.sink.group.eq(0)
                yield dut.sink.row_last.eq(1)
                yield dut.sink.first.eq(row == 0)
                yield dut.sink.last.eq(row == 255)
            else:
                yield dut.sink.valid.eq(0)
            yield dut.source.ready.eq(rng.randrange(3) != 0)
            yield
            if (yield dut.sink.valid) and (yield dut.sink.ready):
                sent += 1
                offered = False
            valid, ready = (yield dut.source.valid), (yield dut.source.ready)
            values = yield from sample(dut.source)
            if held is not None:
                assert valid and values == held
            held = values if valid and not ready else None
            if valid and ready:
                got.append(values)
            if len(got) == len(expected):
                assert sent == len(blocks)
                assert got == expected
                return
        raise AssertionError('decoder stalled')
    run_simulation(dut, run())


def command(dut, encoding, rows, groups, seed=1, stalls=True, hold_output=0, start_while_busy=False):
    rng = random.Random(seed)
    size = 28 if encoding == PTQ1_0 else 34
    blocks = [bytes(rng.randrange(256) for _ in range(size)) for _ in range(rows*groups)]
    expected = [decoded(b, encoding, i // groups, i % groups, groups, i == 0, i == len(blocks)-1)
                for i, b in enumerate(blocks)]
    # Every row is padded to a beat boundary on the device; nonzero padding is ignored.
    payload = qformats.pad_native_rows([b''.join(blocks[r*groups:(r+1)*groups]) for r in range(rows)])
    beats = [int.from_bytes(payload[i:i+32], 'little') for i in range(0, len(payload), 32)]
    yield dut.sink.valid.eq(0)
    yield dut.source.ready.eq(0)
    yield dut.encoding.eq(encoding)
    yield dut.rows.eq(rows)
    yield dut.groups.eq(groups)
    yield dut.start.eq(1)
    yield
    yield dut.start.eq(0)
    yield
    assert (yield dut.busy)
    sent, received, held, offered = 0, [], None, False
    cycles = []
    for cycle in range(10000):
        # Inputs are latched; changing the config after start must have no effect.
        yield dut.rows.eq(65535)
        yield dut.groups.eq(65535)
        yield dut.encoding.eq(255)
        yield dut.start.eq(start_while_busy and cycle == 2)
        if sent < len(beats):
            offered = offered or not stalls or rng.randrange(4) != 0
            yield dut.sink.valid.eq(offered)
            yield dut.sink.data.eq(beats[sent])
            yield dut.sink.last.eq(sent == len(beats) - 1)
        else:
            yield dut.sink.valid.eq(0)
        yield dut.source.ready.eq(cycle >= hold_output and (not stalls or rng.randrange(3) != 0))
        yield
        assert 0 <= (yield dut.buffered_bytes) <= 64
        if (yield dut.sink.valid) and (yield dut.sink.ready):
            sent += 1
            offered = False
        valid, ready = (yield dut.source.valid), (yield dut.source.ready)
        values = yield from sample(dut.source)
        if held is not None:
            assert valid and values == held
        held = values if valid and not ready else None
        if valid and ready:
            received.append(values)
            cycles.append(cycle)
        if (yield dut.done):
            assert not (yield dut.busy)
            assert not (yield dut.cancelled)
            assert (yield dut.error) == int(start_while_busy)
            assert sent == len(beats) and received == expected
            yield dut.sink.valid.eq(0)
            for _ in range(4):
                yield
                assert not (yield dut.source.valid)
                assert not (yield dut.done)
            return cycles
    raise AssertionError(f'weight stream stalled: {sent}/{len(beats)} beats, {len(received)} groups')


@pytest.mark.parametrize('encoding', [PTQ1_0, Q8_0])
@pytest.mark.parametrize('rows,groups', [(1, 1), (3, 1), (3, 2), (2, 8), (3, 9), (2, 136)])
def test_native_stream_groups_and_row_tags(encoding, rows, groups):
    dut = NativeWeightStream()
    run_simulation(dut, command(dut, encoding, rows, groups, seed=rows*groups))


def test_reuse_stalls_latched_config_and_streaming_rate():
    dut = NativeWeightStream()
    def run():
        # A long projection: after pipeline fill, PTQ sustains one group/cycle.
        cycles = yield from command(dut, PTQ1_0, 3, 40, stalls=False)
        assert all(b-a == 1 for a, b in zip(cycles, cycles[1:])), cycles
        # Long downstream stall fills the bounded reservoir + two decoder stages.
        yield from command(dut, Q8_0, 3, 9, hold_output=90)
        yield from command(dut, PTQ1_0, 1, 1, stalls=False)
        # Busy start cannot overwrite the live command; it marks the result failed.
        yield from command(dut, PTQ1_0, 2, 9, start_while_busy=True)
        yield from command(dut, Q8_0, 1, 3, stalls=False)
    run_simulation(dut, run())


@pytest.mark.parametrize('framing_encoding', [PTQ1_0, Q8_0])
def test_invalid_config_early_last_missing_last_and_cancel_recover(framing_encoding):
    dut = NativeWeightStream()
    def run():
        for rows, groups, encoding in [(0, 1, PTQ1_0), (1, 0, PTQ1_0), (1, 1, 255)]:
            yield dut.rows.eq(rows)
            yield dut.groups.eq(groups)
            yield dut.encoding.eq(encoding)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield
            assert not (yield dut.busy) and (yield dut.done) and (yield dut.error)
            yield
        for early in [True, False]:
            yield dut.rows.eq(1)
            yield dut.groups.eq(8)
            yield dut.encoding.eq(framing_encoding)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield
            expected_beats = (8*(28 if framing_encoding == PTQ1_0 else 34) + 31)//32
            nbeats = 2 if early else expected_beats + 1
            yield dut.source.ready.eq(1)
            for i in range(nbeats):
                yield dut.sink.valid.eq(1)
                yield dut.sink.data.eq(i)
                yield dut.sink.last.eq(i == nbeats - 1)
                yield
                while not (yield dut.sink.ready):
                    yield
            yield dut.sink.valid.eq(0)
            yield
            assert (yield dut.error) and (yield dut.done) and not (yield dut.busy)
            for _ in range(4):
                yield
                assert not (yield dut.source.valid)
            yield from command(dut, PTQ1_0, 2, 2, stalls=False)
        # Cancel while output is stalled and both internal stages/reservoir are live.
        yield dut.rows.eq(1)
        yield dut.groups.eq(16)
        yield dut.encoding.eq(PTQ1_0)
        yield dut.start.eq(1)
        yield
        yield dut.start.eq(0)
        yield dut.source.ready.eq(0)
        yield dut.sink.valid.eq(1)
        yield dut.sink.last.eq(0)
        for _ in range(10):
            yield
        assert (yield dut.busy) and (yield dut.source.valid)
        yield dut.sink.valid.eq(0)
        yield dut.cancel.eq(1)
        yield
        yield dut.cancel.eq(0)
        yield
        assert not (yield dut.busy) and (yield dut.cancelled) and (yield dut.done)
        for _ in range(4):
            yield
            assert not (yield dut.source.valid)
        yield from command(dut, Q8_0, 2, 3, stalls=False)
    run_simulation(dut, run())


def test_decoder_elaborates_without_vendor_memory():
    dut = NativeWeightStream()
    output = str(verilog.convert(dut, ios={dut.start, dut.cancel, dut.busy, dut.done,
                 dut.sink.data, dut.sink.valid, dut.sink.ready, dut.sink.last,
                 dut.source.codes, dut.source.scale, dut.source.valid, dut.source.ready}))
    assert 'module top' in output
    assert 'RAMB' not in output and 'DSP48' not in output


def test_compiled_verilog_matches_scalar_reference(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled RTL check')
    dut = NativeWeightStream()
    ios = set(dut.sink.flatten() + dut.source.flatten()) | {
        dut.start, dut.cancel, dut.rows, dut.groups, dut.encoding,
        dut.busy, dut.done, dut.error, dut.cancelled, dut.buffered_bytes}
    rtl = tmp_path / 'native_weights.v'
    verilog.convert(dut, ios=ios, name='native_weights').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/native_weights.cpp'
    result = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                             '--Mdir', str(obj), '--top-module', 'native_weights',
                             str(rtl), str(harness.resolve())], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    rng = random.Random(743)
    for encoding in (PTQ1_0, Q8_0):
        for rows, groups in [(1, 1), (3, 9), (2, 136)]:
            raw = tmp_path / 'raw'
            size = (28 if encoding == PTQ1_0 else 34)*rows*groups
            raw.write_bytes(bytes(rng.randrange(256) for _ in range(size)))
            run = subprocess.run([str(obj / 'Vnative_weights'), str(raw), str(encoding),
                                  str(rows), str(groups)], capture_output=True, text=True, timeout=10)
            assert run.returncode == 0, run.stdout + run.stderr

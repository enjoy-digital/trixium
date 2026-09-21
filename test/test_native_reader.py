# SPDX-License-Identifier: BSD-2-Clause
from collections import defaultdict, deque
from pathlib import Path
import random
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import *
from litex.soc.interconnect.axi import AXIInterface

from trixium.capabilities import MemoryRegion, MemoryKind, Reservation, ReservationKind, Access
from trixium.gateware.native_reader import (
    NativeRegionReader, ERROR_CONFIG, ERROR_BOUNDS, ERROR_RESPONSE, ERROR_FRAMING,
)
from trixium.gateware.native_weights import PTQ1_0, Q8_0
from trixium.model import qformats

FIELDS = ('codes', 'scale', 'kind', 'error', 'row', 'group', 'row_last', 'first', 'last')


def region(region_id=7, base=0, size=1 << 33, endpoint=0, access=Access.READ):
    return MemoryRegion(region_id, MemoryKind.SIMULATION, endpoint, base, size, access=access)


class ReaderDUT(Module):
    def __init__(self, regions=None, reservations=(), width=34, **kwargs):
        self.axi = AXIInterface(data_width=256, address_width=width, id_width=4)
        self.submodules.unit = NativeRegionReader(self.axi, regions or [region()], reservations,
                                                  fifo_depth=64, **kwargs)


def row_case(encoding, rows, groups, offset, stride, region_id=7, base=0, seed=5):
    rng = random.Random(seed)
    block_bytes = 28 if encoding == PTQ1_0 else 34
    memory, expected, spans = {}, [], []
    for row in range(rows):
        start = base + offset + row*stride
        spans.append((start & -32, ((start % 32) + groups*block_bytes + 31)//32))
        for group in range(groups):
            block = rng.randbytes(block_bytes)
            memory.update({start+group*block_bytes+i: b for i, b in enumerate(block)})
            if encoding == PTQ1_0:
                _, codes = qformats.ptq1_0_unpack(block)
                bits = sum(int(c) << (2*i) for i, c in enumerate(codes[0]))
                scale = int.from_bytes(block[26:28], 'little')
            else:
                bits, scale = int.from_bytes(block[2:], 'little'), int.from_bytes(block[:2], 'little')
            expected.append((bits, scale, int(encoding == Q8_0), 0, row, group,
                             int(group == groups-1), int(row == group == 0),
                             int(row == rows-1 and group == groups-1)))
    cmd = dict(region=region_id, offset=offset, stride=stride, rows=rows, groups=groups, encoding=encoding)
    return cmd, memory, expected, spans


def memory_server(dut, memory, observed, *, seed=25, latency=8, response_error=False,
                  wrong_id=False, early_last=False):
    axi = dut.axi
    rng = random.Random(seed)
    pending = defaultdict(deque)
    cycle, held, injected_last = 0, None, False
    @passive
    def run():
        nonlocal cycle, held, injected_last
        while True:
            yield axi.ar.ready.eq(rng.randrange(4) != 0)
            if held is None:
                ids = [bid for bid, queue in pending.items() if queue and queue[0][2] <= cycle]
                if ids and rng.randrange(4) != 0:
                    bid = rng.choice(ids)
                    addr, left, ready_at = pending[bid][0]
                    held = bid, addr, left
            yield axi.r.valid.eq(held is not None)
            if held is not None:
                bid, addr, left = held
                yield axi.r.id.eq(bid + 4 if wrong_id else bid)
                yield axi.r.data.eq(int.from_bytes(bytes(memory.get(addr+i, 0xa5) for i in range(32)), 'little'))
                yield axi.r.last.eq(left == 1 or (early_last and not injected_last))
                yield axi.r.resp.eq(2 if response_error else 0)
            yield
            if (yield axi.ar.valid) and (yield axi.ar.ready):
                addr, beats, bid = (yield axi.ar.addr), (yield axi.ar.len)+1, (yield axi.ar.id)
                assert addr % 32 == 0 and 1 <= beats <= 16
                assert (addr & 4095) + beats*32 <= 4096
                observed.append((addr, beats, bid))
                pending[bid].append((addr, beats, cycle+latency))
            if held is not None and (yield axi.r.ready):
                bid, addr, left = held
                pending[bid].popleft()
                if left > 1:
                    pending[bid].appendleft((addr+32, left-1, cycle))
                held = None
                injected_last = True
            cycle += 1
    return run()


def submit(unit, cmd):
    for name, value in cmd.items():
        yield getattr(unit.cmd, name).eq(value)
    yield unit.cmd.valid.eq(1)
    yield
    while not (yield unit.cmd.ready):
        yield
    yield unit.cmd.valid.eq(0)
    # No subsequent address or format input may affect this latched command.
    for name in cmd:
        yield getattr(unit.cmd, name).eq(0)


def collect(unit, cmd, expected, error=0, cancel_after=None):
    yield from submit(unit, cmd)
    got, held = [], None
    rng = random.Random(76)
    for cycle in range(30000):
        yield unit.source.ready.eq(cycle >= 80 and rng.randrange(3) != 0)
        yield unit.cancel.eq(cancel_after == cycle)
        yield
        fields = []
        for name in FIELDS:
            fields.append((yield getattr(unit.source, name)))
        fields = tuple(fields)
        valid, ready = (yield unit.source.valid), (yield unit.source.ready)
        if held is not None and not error and cancel_after is None:
            assert valid and fields == held
        held = fields if valid and not ready else None
        if valid and ready:
            got.append(fields)
        if (yield unit.done):
            assert (yield unit.error) == error
            assert bool((yield unit.cancelled)) == (cancel_after is not None)
            assert not (yield unit.busy) and (yield unit.reader.idle)
            if not error and cancel_after is None:
                assert got == expected
            yield unit.cancel.eq(0)
            return
    raise AssertionError(f'region read timed out: {len(got)}/{len(expected)} groups')


@pytest.mark.parametrize('encoding', [PTQ1_0, Q8_0])
@pytest.mark.parametrize('rows,groups,prefix,gap', [
    (17, 1, 31, 3), (5, 9, 2, 60), (3, 68, 16, 1904), (2, 136, 0, 32),
])
def test_strided_native_reads_above_4g(encoding, rows, groups, prefix, gap):
    block_bytes = 28 if encoding == PTQ1_0 else 34
    cmd, memory, expected, spans = row_case(encoding, rows, groups,
        (1 << 32) + 4064 + prefix, groups*block_bytes + gap)
    dut = ReaderDUT()
    observed = []
    run_simulation(dut, [collect(dut.unit, cmd, expected), memory_server(dut, memory, observed)])
    wanted = [addr + i*32 for addr, beats in spans for i in range(beats)]
    got = [addr + i*32 for addr, beats, _ in observed for i in range(beats)]
    assert got == wanted
    assert {bid for _, _, bid in observed} == {0, 1, 2, 3}


def test_region_base_and_mode_reuse():
    regions = [region(7, base=1 << 32, size=1 << 20), region(42, size=1 << 20)]
    cases = [row_case(PTQ1_0, 3, 1, 14, 30, base=1 << 32),
             row_case(Q8_0, 3, 2, 30, 96, region_id=42, seed=9)]
    dut = ReaderDUT(regions)
    observed = []
    memory = {addr: value for _, data, _, _ in cases for addr, value in data.items()}
    def run():
        for cmd, _, expected, _ in cases:
            yield from collect(dut.unit, cmd, expected)
    run_simulation(dut, [run(), memory_server(dut, memory, observed)])


@pytest.mark.parametrize('change', [dict(region=99), dict(rows=0), dict(groups=0),
                                   dict(encoding=255), dict(stride=27)])
def test_bad_configuration_does_not_issue(change):
    cmd, memory, expected, _ = row_case(PTQ1_0, 2, 1, 0, 32)
    cmd.update(change)
    dut, observed = ReaderDUT(), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_CONFIG), memory_server(dut, memory, observed)])
    assert not observed


@pytest.mark.parametrize('offset,reservations,width', [
    (4090, (), 34),  # Tail read crosses the end of the selected 4 KiB region.
    (1 << 40, (), 34),  # Must not truncate to AXI width.
    (2**64-8, (), 64),  # Must not wrap offset + length at 64 bits.
    (33, (Reservation(7, ReservationKind.SYSTEM, 0, 40),), 34),  # Aligned prefix is reserved.
    (3, (Reservation(7, ReservationKind.SYSTEM, 31, 1),), 34),  # Trailing read-ahead is reserved.
])
def test_aligned_span_bounds_before_io(offset, reservations, width):
    cmd, memory, expected, _ = row_case(PTQ1_0, 1, 1, offset, 28)
    dut, observed = ReaderDUT([region(size=4096)], reservations, width), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_BOUNDS), memory_server(dut, memory, observed)])
    assert not observed


def test_later_invalid_row_drains_prior_requests_and_reuses():
    cmd, memory, _, _ = row_case(PTQ1_0, 5, 1, 4000, 32)
    good, data, expected, _ = row_case(Q8_0, 2, 1, 0, 36)
    memory.update(data)
    dut, observed = ReaderDUT([region(size=4096)]), []
    def run():
        yield from collect(dut.unit, cmd, [], error=ERROR_BOUNDS)
        assert all(addr+beats*32 <= 4096 for addr, beats, _ in observed)
        yield from collect(dut.unit, good, expected)
    run_simulation(dut, [run(), memory_server(dut, memory, observed, latency=40)])


@pytest.mark.parametrize('response_error', [False, True])
def test_cancel_or_response_failure_drains(response_error):
    cmd, memory, _, _ = row_case(PTQ1_0, 9, 68, 16, 3808)
    dut, observed = ReaderDUT(), []
    error = ERROR_RESPONSE if response_error else 0
    def run():
        yield from collect(dut.unit, cmd, [], error=error, cancel_after=None if response_error else 60)
        for _ in range(8):
            yield
            assert not (yield dut.unit.source.valid)
        if not response_error:
            small = dict(cmd, rows=1)
            _, _, expected, _ = row_case(PTQ1_0, 1, 68, 16, 3808)
            yield from collect(dut.unit, small, expected)
    run_simulation(dut, [run(), memory_server(dut, memory, observed, response_error=response_error)])


def test_unknown_response_id_is_not_silently_truncated():
    cmd, memory, _, _ = row_case(Q8_0, 5, 9, 16, 320)
    dut, observed = ReaderDUT(), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_RESPONSE),
                         memory_server(dut, memory, observed, wrong_id=True)])


def test_early_row_last_fails_and_does_not_claim_transport_recovery():
    cmd, memory, _, _ = row_case(PTQ1_0, 1, 9, 0, 252)
    dut, observed = ReaderDUT(), []
    def run():
        yield from submit(dut.unit, cmd)
        yield dut.unit.source.ready.eq(1)
        failed = False
        for _ in range(200):
            yield
            failed |= bool((yield dut.unit.error) & ERROR_FRAMING)
            if failed:
                assert not (yield dut.unit.source.valid)
            if not (yield dut.unit.reader.idle):
                assert not (yield dut.unit.done)
        assert failed
        # Malformed RLAST can strand the underlying reader's reorder queues.
        # A real runtime must use the external reset/drain shim before reuse.
        assert (yield dut.unit.busy)
    run_simulation(dut, [run(), memory_server(dut, memory, observed, early_last=True)])


@pytest.mark.parametrize('regions,width', [
    ([region(endpoint=1)], 34), ([region(access=Access(0))], 34),
])
def test_unreachable_or_disallowed_region_is_rejected(regions, width):
    cmd, memory, _, _ = row_case(PTQ1_0, 1, 1, 0, 28)
    dut, observed = ReaderDUT(regions, width=width), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_CONFIG), memory_server(dut, memory, observed)])
    assert not observed


def test_physical_base_must_fit_axi_width():
    cmd, memory, _, _ = row_case(PTQ1_0, 1, 1, 0, 28, base=1 << 32)
    dut, observed = ReaderDUT([region(base=1 << 32, size=4096)], width=32), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_BOUNDS), memory_server(dut, memory, observed)])
    assert not observed


def test_64_bit_stride_addition_cannot_alias_low_memory():
    cmd, memory, _, _ = row_case(PTQ1_0, 3, 1, 32, 2**64-32)
    dut, observed = ReaderDUT(), []
    run_simulation(dut, [collect(dut.unit, cmd, [], error=ERROR_BOUNDS), memory_server(dut, memory, observed)])
    assert [addr for addr, _, _ in observed] == [32]


def test_region_constructor_rejects_invalid_tables():
    for regions, reservations in [
        ([region(), region()], ()),
        ([region(7, size=4096), region(8, size=4096)], ()),
        ([region(size=4096)], (Reservation(8, ReservationKind.SYSTEM, 0, 32),)),
        ([region(size=4096)], (Reservation(7, ReservationKind.SYSTEM, 4096, 32),)),
    ]:
        with pytest.raises(ValueError):
            ReaderDUT(regions, reservations)


def test_real_27b_ffn_k_segments_through_axi_and_mac():
    from trixium.gateware.ternary_mac import BonsaiChannelPipe
    from trixium.model.gguf_reader import GGUFReader
    from trixium.model import bitexact as bx
    path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    if not path.exists():
        pytest.skip('local 27B GGUF required')
    image = GGUFReader(path)
    tensor = image.tensors['blk.0.ffn_down.weight']
    rows, cols = tensor.shape
    assert cols == 17408 and tensor.type_id == qformats.GGML_PTQ1_0
    raw = image.tensor_bytes(tensor.name)
    all_scales = qformats.ptq1_0_scales(raw).reshape(rows, 136)
    _, exponents = bx.fp16_decompose(all_scales)
    bias = bx.choose_scale_bias(exponents)
    shift = bx.choose_result_shift(all_scales, bias)
    row0, nrows, native_row = 1137, 2, 136*28
    payload = bytes(raw[row0*native_row:(row0+nrows)*native_row])
    physical_base = (1 << 32) + 64
    memory = {physical_base+i: b for i, b in enumerate(payload)}
    # Quantized activations are selected ONCE at whole-tensor width, then sliced.
    x = np.random.default_rng(315).integers(-128, 128, cols, dtype=np.int8)
    dut, observed = ReaderDUT(), []
    dut.submodules.pipe = pipe = BonsaiChannelPipe(xbuf_groups=96, semantic=True, dual=False)
    dut.comb += [dut.unit.source.connect(pipe.sink),
                 pipe.start.eq(dut.unit.cmd.valid & dut.unit.cmd.ready)]
    def run():
        for segment in (0, 1):
            for name, value in dict(kind=0, rows=nrows, code_beats=68, bias=bias, rshift=shift).items():
                yield getattr(pipe, name).eq(value)
            xb = x[segment*8704:(segment+1)*8704].tobytes()
            for pos in range(0, len(xb), 32):
                yield pipe.xw_group.eq(pos//128)
                yield pipe.xw_sub.eq(pos//32 % 4)
                yield pipe.xw_data.eq(int.from_bytes(xb[pos:pos+32], 'little'))
                yield pipe.xw_we.eq(1)
                yield
            yield pipe.xw_we.eq(0)
            scales, codes = qformats.ptq1_0_unpack(b''.join(
                payload[r*native_row+segment*1904:r*native_row+(segment+1)*1904] for r in range(nrows)))
            expected = bx.matvec_ternary(scales.reshape(nrows, 68), codes.reshape(nrows, 8704),
                                        x[segment*8704:(segment+1)*8704], bias, shift)
            yield from submit(dut.unit, dict(region=7, offset=physical_base+segment*1904,
                                             stride=native_row, rows=nrows, groups=68, encoding=PTQ1_0))
            got, decoded = [], False
            for cycle in range(10000):
                yield pipe.source.ready.eq(cycle >= 150 and cycle % 3 != 0)
                yield
                assert not (yield dut.unit.error) and not (yield pipe.error) and not (yield pipe.overflow)
                decoded |= bool((yield dut.unit.done))
                if (yield pipe.source.valid) and (yield pipe.source.ready):
                    assert (yield pipe.source.row) == len(got)
                    assert not (yield pipe.source.head) and not (yield pipe.source.wide)
                    value = (yield pipe.source.data) & 0xffffffff
                    got.append(value - (1 << 32) if value & (1 << 31) else value)
                if decoded and len(got) == nrows:
                    assert np.array_equal(got, expected)
                    break
            else:
                raise AssertionError('real K-segment MAC timeout')
    run_simulation(dut, [run(), memory_server(dut, memory, observed)])
    wanted = [((physical_base+s*1904+r*native_row) & -32) + b*32
              for s in range(2) for r in range(nrows) for b in range(60)]
    actual = [addr+i*32 for addr, beats, _ in observed for i in range(beats)]
    assert actual == wanted


@pytest.fixture(scope='module')
def compiled_native_reader(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled reader check')
    from migen.fhdl import verilog
    tmp_path = tmp_path_factory.mktemp('compiled-native-reader')
    dut = ReaderDUT([region(), region(42, base=1 << 33, size=1 << 20)])
    unit, axi = dut.unit, dut.axi
    ports = {name: getattr(unit, name) for name in ('busy', 'done', 'error', 'cancel', 'cancelled')}
    ports['reader_idle'] = unit.reader.idle
    for prefix, endpoint, fields in [
        ('cmd', unit.cmd, ('valid', 'ready', 'region', 'offset', 'stride', 'rows', 'groups', 'encoding')),
        ('source', unit.source, ('valid', 'ready') + FIELDS),
        ('ar', axi.ar, ('valid', 'ready', 'addr', 'len', 'size', 'burst', 'id')),
        ('r', axi.r, ('valid', 'ready', 'data', 'last', 'id', 'resp')),
    ]:
        ports.update({prefix+'_'+name: getattr(endpoint, name) for name in fields})
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'native_reader.v'
    verilog.convert(dut, ios=set(ports.values()), name='native_reader').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/native_reader.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'native_reader', str(rtl),
                            str(harness.resolve())], capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout[-6000:] + built.stderr[-6000:]
    return obj / 'Vnative_reader'


def test_compiled_native_region_reader(tmp_path, compiled_native_reader):
    cases = []
    for encoding in (PTQ1_0, Q8_0):
        for prefix in range(32):
            block = 28 if encoding == PTQ1_0 else 34
            cases.append((row_case(encoding, 9, 1, (1 << 32)+4064+prefix, block+3), 0, 0))
        cases += [(row_case(encoding, 3, 68, (1 << 32)+1904, 3808 if encoding == PTQ1_0 else 4800), 0, 0),
                  (row_case(encoding, 2, 136, (1 << 32), 136*block), 0, 0)]
    # Each flag is exercised with multiple accepted row commands in flight.
    case = row_case(PTQ1_0, 9, 68, 16, 3808)
    cases += [(case, ERROR_RESPONSE, 1), (case, 0, 2), (case, ERROR_RESPONSE, 4)]
    for offset in (2**64-8, 1 << 40, (1 << 33)-16):
        cases.append((row_case(PTQ1_0, 1, 1, offset, 28), ERROR_BOUNDS, 0))
    cases += [(row_case(PTQ1_0, 3, 1, 32, 2**64-32), ERROR_BOUNDS, 0),
              (row_case(Q8_0, 4, 1, (1 << 33)-192, 64), ERROR_BOUNDS, 0),
              (row_case(Q8_0, 3, 9, 17, 308, base=1 << 33, region_id=42), 0, 0)]
    for index, ((cmd, memory, expected, spans), error, flags) in enumerate(cases):
        reads = [addr+i*32 for addr, beats in spans for i in range(beats)
                 if error != ERROR_BOUNDS or addr+beats*32 <= 1 << 33]
        addresses = sorted(set(reads))
        header = struct.pack('<10I2Q', cmd['encoding'], cmd['rows'], cmd['groups'], cmd['region'],
                             error, flags, len(addresses), len(expected), len(reads), 0,
                             cmd['offset'], cmd['stride'])
        payload = b''.join(struct.pack('<Q', addr) + bytes(memory.get(addr+i, 0xa5) for i in range(32))
                           for addr in addresses)
        for codes, scale, kind, bad, row, group, row_last, first, last in expected:
            payload += codes.to_bytes(32, 'little') + struct.pack('<8I', scale, row, group, row_last,
                                                                 first, last, kind, bad)
        payload += b''.join(struct.pack('<Q', addr) for addr in reads)
        vector = tmp_path / 'region-vector'
        vector.write_bytes(header + payload)
        result = subprocess.run([str(compiled_native_reader), str(vector)], capture_output=True,
                                text=True, timeout=15)
        assert result.returncode == 0, (index, cmd, error, flags, result.stdout, result.stderr)

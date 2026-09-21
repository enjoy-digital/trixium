# SPDX-License-Identifier: BSD-2-Clause
"""Native packed memory beats -> shared semantic MAC, not a replacement dot product."""
import random
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import *

from trixium.gateware.native_weights import NativeWeightStream, PTQ1_0, Q8_0
from trixium.gateware.ternary_mac import (
    BonsaiChannelPipe, KIND_OP_TERNARY, KIND_OP_Q8, KIND_OP_AXPY,
)
from trixium.model import bitexact as bx, qformats


class NativeMAC(Module):
    def __init__(self, xgroups, dual=False, channel=0, channels=1, out_fifo_depth=24):
        self.submodules.decoder = NativeWeightStream()
        self.submodules.pipe = BonsaiChannelPipe(xbuf_groups=xgroups, dual=dual, semantic=True,
                                                  chan_id=channel, n_channels=channels,
                                                  out_fifo_depth=out_fifo_depth)
        self.comb += self.decoder.source.connect(self.pipe.sink)


def make_case(kind, rows, groups, qn=False, seed=1, channel=0, channels=1):
    rng = np.random.default_rng(seed)
    width = (128 if kind == KIND_OP_TERNARY else 32)*groups
    if kind == KIND_OP_AXPY:
        scales = np.repeat(np.exp2(rng.integers(-14, -2, rows)).astype(np.float16)[:, None], groups, axis=1)
    else:
        scales = (rng.uniform(-2, 2, (rows, groups))*np.exp2(rng.integers(-14, 2, (rows, groups)))).astype(np.float16)
        scales.flat[0] = np.float16(0)
        if rows*groups > 2:
            scales.flat[1] = np.float16(-0.0)
            scales.flat[2] = np.float16(2**-24)  # Subnormal, preserved through native FP16 bits.
    values = (rng.integers(0, 3, (rows, width), dtype=np.uint8) if kind == KIND_OP_TERNARY else
              rng.integers(-128, 128, (rows, width), dtype=np.int8))
    xs = [rng.integers(-128, 128, width, dtype=np.int8) for _ in range(2 if qn else 1)]
    _, exp = bx.fp16_decompose(scales)
    bias = bx.choose_scale_bias(exp)
    shift = bx.choose_result_shift(scales, bias, gs_max=bx.GS_MAX if kind == 0 else bx.GS_MAX_Q8)
    shift2 = shift
    if kind == KIND_OP_AXPY:
        probs = [rng.integers(0, 128, rows*channels, dtype=np.int8) for _ in xs]
        shift = bx.axpy_rpre(int(probs[0].astype(np.int64).sum()))
        shift2 = bx.axpy_rpre(int(probs[-1].astype(np.int64).sum()))
        if qn and shift == shift2:
            shift2 += 1
        expected = [bx.axpy_rows(scales, values, p[channel::channels], bias, r)
                    for p, r in zip(probs, [shift, shift2])]
        if qn:
            xb = bytearray(((rows*channels + 31)//32)*64)
            for t in range(rows*channels):
                xb[(t >> 5)*64 + (t & 31)] = int(probs[0][t])
                xb[(t >> 5)*64 + (t & 31) + 32] = int(probs[1][t])
        else:
            xb = bytearray(probs[0].tobytes())
    else:
        fn = bx.matvec_ternary if kind == KIND_OP_TERNARY else bx.matvec_q8
        expected = [fn(scales, values, x, bias, shift) for x in xs]
        if qn:
            granule = 128 if kind == KIND_OP_TERNARY else 32
            xb = bytearray(b''.join(x[g:g+granule].tobytes()
                                   for g in range(0, width, granule) for x in xs))
        else:
            xb = bytearray(xs[0].tobytes())
    xb += bytes((-len(xb)) % 128)
    pack = qformats.ptq1_0_pack if kind == KIND_OP_TERNARY else qformats.q8_0_pack
    payload = pack(scales.reshape(-1), values.reshape(rows*groups, -1)).tobytes()
    payload += bytes([0xa5])*((-len(payload)) % 32)
    return dict(kind=kind, rows=rows, groups=groups, qn=qn, bias=bias, shift=shift, shift2=shift2,
                expected=expected, xb=bytes(xb), payload=payload)


def overflow_case(kind, qn=False, negative=False):
    """Representable inputs but invalid accumulator/result range, including B-only overflow."""
    rows, groups = (5, 1) if kind == KIND_OP_AXPY else (1, (65 if negative else 64) if kind == KIND_OP_Q8 else 1)
    width = groups*(128 if kind == KIND_OP_TERNARY else 32)
    values = np.full((rows, width), 2 if kind == KIND_OP_TERNARY else (-128 if kind == KIND_OP_Q8 else 127),
                     dtype=np.uint8 if kind == KIND_OP_TERNARY else np.int8)
    scales = np.ones((rows, groups), dtype=np.float16)
    if negative:
        if kind == KIND_OP_AXPY:
            values.fill(-128)
        else:
            scales *= -1
    if kind == KIND_OP_AXPY:
        # Global probabilities, channel 2 of 4: one coefficient for every stripe row.
        x = bytes([127])*(rows*4)
        xb = bytearray(128)
        for t, value in enumerate(x):
            if qn:
                xb[(t >> 5)*64 + (t & 31) + 32] = value
            else:
                xb[t] = value
    else:
        x = bytes([127 if kind == KIND_OP_TERNARY else 128])*width
        granule = 128 if kind == KIND_OP_TERNARY else 32
        xb = bytearray(b''.join(bytes(granule) + x[g:g+granule] for g in range(0, width, granule)) if qn else x)
    xb += bytes((-len(xb)) % 128)
    pack = qformats.ptq1_0_pack if kind == KIND_OP_TERNARY else qformats.q8_0_pack
    payload = pack(scales.reshape(-1), values.reshape(rows*groups, -1)).tobytes()
    payload += bytes((-len(payload)) % 32)
    shift = 0 if kind == KIND_OP_TERNARY else (40 if kind == KIND_OP_Q8 else 7)
    return dict(kind=kind, rows=rows, groups=groups, qn=qn, bias=22, shift=shift, shift2=shift,
                xb=bytes(xb), payload=payload, expected=[])


def drive(dut, case, *, gaps=True, output_stall=0, seed=3):
    pipe, decoder = dut.pipe, dut.decoder
    rng = random.Random(seed)
    x = case['xb']
    for offset in range(0, len(x), 32):
        yield pipe.xw_group.eq(offset // 128)
        yield pipe.xw_sub.eq((offset // 32) % 4)
        yield pipe.xw_data.eq(int.from_bytes(x[offset:offset+32], 'little'))
        yield pipe.xw_we.eq(1)
        yield
    yield pipe.xw_we.eq(0)
    for name, value in [('kind', case['kind']), ('rows', case['rows']), ('code_beats', case['groups']),
                        ('bias', case['bias']), ('rshift', case['shift']), ('rshift2', case['shift2']),
                        ('qn', case['qn'])]:
        yield getattr(pipe, name).eq(value)
    yield pipe.start.eq(1)
    yield decoder.start.eq(1)
    yield decoder.rows.eq(case['rows'])
    yield decoder.groups.eq(case['groups'])
    yield decoder.encoding.eq(PTQ1_0 if case['kind'] == KIND_OP_TERNARY else Q8_0)
    yield
    yield pipe.start.eq(0)
    yield decoder.start.eq(0)
    yield
    payload = case['payload']
    beats = [int.from_bytes(payload[i:i+32], 'little') for i in range(0, len(payload), 32)]
    sent, offered, done, held = 0, False, False, None
    result = {}
    wanted = sum(len(e) for e in case['expected'])
    for cycle in range(15000):
        if sent < len(beats):
            offered = offered or not gaps or rng.randrange(3) != 0
            yield decoder.sink.valid.eq(offered)
            yield decoder.sink.data.eq(beats[sent])
            yield decoder.sink.last.eq(sent == len(beats)-1)
        else:
            yield decoder.sink.valid.eq(0)
        yield pipe.source.ready.eq(cycle >= output_stall and (not gaps or rng.randrange(3) != 0))
        yield
        if (yield decoder.sink.valid) and (yield decoder.sink.ready):
            sent += 1
            offered = False
        done |= bool((yield decoder.done))
        assert not (yield decoder.error)
        assert not (yield pipe.error), f'pipe error at cycle {cycle}, sent {sent}'
        assert not (yield pipe.overflow), f'FIFO overflow at cycle {cycle}'
        fields = ((yield pipe.source.row), (yield pipe.source.head), (yield pipe.source.data), (yield pipe.source.wide))
        valid, ready = (yield pipe.source.valid), (yield pipe.source.ready)
        if held is not None:
            assert valid and fields == held
        held = fields if valid and not ready else None
        if valid and ready:
            row, head, data, wide = fields
            assert wide == (case['kind'] == KIND_OP_AXPY)
            for lane in range(8 if wide else 1):
                raw = (data >> (32*lane)) & 0xffffffff
                value = raw - (1 << 32) if raw & (1 << 31) else raw
                key = (head, row+lane)
                assert key not in result, f'duplicate output {key}'
                result[key] = value
        if len(result) == wanted and done:
            assert sent == len(beats)
            for head, expected in enumerate(case['expected']):
                got = np.array([result[head, r] for r in range(len(expected))], dtype=np.int32)
                assert np.array_equal(got, expected), (case['kind'], case['rows'], case['groups'], head, got, expected)
            yield decoder.sink.valid.eq(0)
            for _ in range(16):
                yield
                assert not (yield pipe.source.valid)
            return
    raise AssertionError(f'native MAC timeout: {sent}/{len(beats)} beats, {len(result)}/{wanted} results, done={done}')


@pytest.mark.parametrize('kind', [KIND_OP_TERNARY, KIND_OP_Q8])
@pytest.mark.parametrize('groups,rows,dual,qn,gaps', [
    (1, 50, False, False, False),  # Dense single-head row-lasts exceed old scale-beat spacing.
    (1, 50, True, True, False),    # Dual row-lasts must allow both result FIFO pushes.
    (2, 9, True, True, False),
    (5, 7, False, False, True),
    (9, 3, True, True, True),
    (40, 2, True, False, False),
])
def test_native_matvec(kind, groups, rows, dual, qn, gaps):
    case = make_case(kind, rows, groups, qn, seed=groups*rows + kind)
    dut = NativeMAC(max(8, len(case['xb']) // 128), dual=dual)
    run_simulation(dut, drive(dut, case, gaps=gaps, output_stall=100))


def test_full_ffn_136_groups_without_scale_ram_alias():
    case = make_case(KIND_OP_TERNARY, 2, 136, seed=42)
    dut = NativeMAC(136)
    run_simulation(dut, drive(dut, case, gaps=False))


@pytest.mark.parametrize('qn', [False, True])
def test_small_fifo_dense_rows_backpressure(qn):
    case = make_case(KIND_OP_TERNARY, 50, 1, qn, seed=63)
    dut = NativeMAC(8, dual=qn, out_fifo_depth=10)
    run_simulation(dut, drive(dut, case, gaps=False, output_stall=100))


@pytest.mark.parametrize('groups,qn', [(1, False), (1, True), (2, False), (4, True)])
def test_native_axpy(groups, qn):
    case = make_case(KIND_OP_AXPY, 29, groups, qn, seed=groups*100, channel=2, channels=4)
    dut = NativeMAC(8, dual=qn, channel=2, channels=4)
    run_simulation(dut, drive(dut, case, gaps=False, output_stall=250))


def test_reuse_modes_without_stale_scales():
    dut = NativeMAC(16, dual=True)
    def run():
        for i, kind in enumerate([KIND_OP_TERNARY, KIND_OP_Q8, KIND_OP_AXPY, KIND_OP_TERNARY]):
            case = make_case(kind, 7, 2, qn=bool(i & 1), seed=87+i)
            yield from drive(dut, case)
    run_simulation(dut, run())


def test_invalid_semantic_group_is_not_launched():
    mutations = {'row': 1, 'group': 256, 'row_last': 0, 'first': 0, 'last': 0, 'kind': 1, 'error': 1}
    for field, value in mutations.items():
        dut = BonsaiChannelPipe(xbuf_groups=8, dual=False, semantic=True)
        def run():
            yield dut.rows.eq(1)
            yield dut.code_beats.eq(1)
            yield dut.kind.eq(KIND_OP_TERNARY)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield dut.sink.valid.eq(1)
            for name, v in dict(row=0, group=0, row_last=1, first=1, last=1, kind=0, error=0).items():
                yield getattr(dut.sink, name).eq(value if name == field else v)
            yield
            assert (yield dut.sink.ready)
            yield dut.sink.valid.eq(0)
            yield
            assert (yield dut.error)
            yield dut.source.ready.eq(1)
            for _ in range(30):
                yield
                assert not (yield dut.source.valid)
        run_simulation(dut, run())


def test_semantic_axpy_rejects_non_power2_or_nonconstant_scales():
    for scale, second in [(0, False), (0x8000, False), (0xbc00, False), (0x3c01, False),
                          (0x7c00, False), (0x7e01, False), (1, False), (0x4000, True)]:
        dut = BonsaiChannelPipe(xbuf_groups=8, dual=False, semantic=True)
        def run():
            yield dut.rows.eq(1)
            yield dut.code_beats.eq(2 if second else 1)
            yield dut.kind.eq(KIND_OP_AXPY)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield dut.sink.valid.eq(1)
            yield dut.sink.kind.eq(1)
            yield dut.sink.first.eq(1)
            if second:
                yield dut.sink.scale.eq(0x3c00)
                yield
                assert (yield dut.sink.ready)
                yield dut.sink.group.eq(1)
                yield dut.sink.first.eq(0)
            yield dut.sink.scale.eq(scale)
            yield dut.sink.row_last.eq(1)
            yield dut.sink.last.eq(1)
            yield
            assert (yield dut.sink.ready)
            yield dut.sink.valid.eq(0)
            yield
            assert (yield dut.error)
            yield dut.source.ready.eq(1)
            for _ in range(30):
                yield
                assert not (yield dut.source.valid)
        run_simulation(dut, run())


def test_semantic_pipe_constructor_bounds():
    for options in [dict(out_fifo_depth=9), dict(xbuf_groups=0), dict(n_channels=0), dict(chan_id=1)]:
        with pytest.raises(ValueError):
            BonsaiChannelPipe(semantic=True, **options)


def test_semantic_rejects_nonfinite_and_unsupported_scale_shift_contracts():
    cases = [(0, 0x7c00, 0, 0, 0, 0), (0, 0xfe01, 0, 0, 0, 0),
             (0, 0x3c00, 23, 20, 20, 0), (0, 0, 37, 0, 0, 0),
             (2, 0x3c00, 22, 6, 20, 0), (2, 0x3c00, 22, 8, 6, 1)]
    for kind, scale, bias, shift, shift2, qn in cases:
        dut = BonsaiChannelPipe(xbuf_groups=8, dual=True, semantic=True)
        def run():
            yield dut.rows.eq(1)
            yield dut.code_beats.eq(1)
            yield dut.kind.eq(kind)
            yield dut.qn.eq(qn)
            yield dut.bias.eq(bias)
            yield dut.rshift.eq(shift)
            yield dut.rshift2.eq(shift2)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield dut.sink.valid.eq(1)
            yield dut.sink.kind.eq(kind != 0)
            yield dut.sink.scale.eq(scale)
            yield dut.sink.first.eq(1)
            yield dut.sink.row_last.eq(1)
            yield dut.sink.last.eq(1)
            yield
            yield dut.sink.valid.eq(0)
            yield
            assert (yield dut.error) and not (yield dut.arithmetic_error)
            for _ in range(30):
                yield
                assert not (yield dut.source.valid)
        run_simulation(dut, run())


def test_native_mac_local_reset_discards_pending_groups_and_results():
    dut = ResetInserter()(NativeMAC(16, dual=True))
    pipe, decoder = dut.pipe, dut.decoder
    def run():
        # Fill output FIFO while valid groups remain in the arithmetic pipeline.
        yield pipe.kind.eq(0)
        yield pipe.rows.eq(50)
        yield pipe.code_beats.eq(1)
        yield pipe.bias.eq(0)
        yield pipe.rshift.eq(0)
        yield decoder.rows.eq(50)
        yield decoder.groups.eq(1)
        yield decoder.encoding.eq(PTQ1_0)
        yield pipe.start.eq(1)
        yield decoder.start.eq(1)
        yield
        yield pipe.start.eq(0)
        yield decoder.start.eq(0)
        yield decoder.sink.valid.eq(1)
        yield decoder.sink.data.eq(0)
        yield decoder.sink.last.eq(0)
        yield pipe.source.ready.eq(0)
        for _ in range(50):
            yield
        assert (yield pipe.source.valid) and (yield decoder.busy)
        yield decoder.sink.valid.eq(0)  # Real adapter must separately drain its AXI traffic.
        yield dut.reset.eq(1)
        for _ in range(3):
            yield
        yield dut.reset.eq(0)
        for _ in range(8):
            yield
            assert not (yield pipe.source.valid)
            assert not (yield decoder.busy)
            assert not (yield pipe.error)
        yield from drive(dut, make_case(KIND_OP_Q8, 7, 2, qn=True, seed=221))
    run_simulation(dut, run())


@pytest.mark.parametrize('kind,qn', [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)])
def test_semantic_arithmetic_overflow_is_reported(kind, qn):
    case = overflow_case(kind, qn)
    dut = NativeMAC(32, dual=qn, channel=2, channels=4)
    pipe, decoder = dut.pipe, dut.decoder
    def run():
        for offset in range(0, len(case['xb']), 32):
            yield pipe.xw_group.eq(offset//128)
            yield pipe.xw_sub.eq((offset//32) % 4)
            yield pipe.xw_data.eq(int.from_bytes(case['xb'][offset:offset+32], 'little'))
            yield pipe.xw_we.eq(1)
            yield
        yield pipe.xw_we.eq(0)
        for field, key in [('kind', 'kind'), ('rows', 'rows'), ('code_beats', 'groups'), ('bias', 'bias'),
                           ('rshift', 'shift'), ('rshift2', 'shift2'), ('qn', 'qn')]:
            yield getattr(pipe, field).eq(case[key])
        yield decoder.rows.eq(case['rows'])
        yield decoder.groups.eq(case['groups'])
        yield decoder.encoding.eq(PTQ1_0 if kind == 0 else Q8_0)
        yield pipe.start.eq(1)
        yield decoder.start.eq(1)
        yield
        yield pipe.start.eq(0)
        yield decoder.start.eq(0)
        yield pipe.source.ready.eq(1)
        payload = case['payload']
        for offset in range(0, len(payload), 32):
            yield decoder.sink.valid.eq(1)
            yield decoder.sink.data.eq(int.from_bytes(payload[offset:offset+32], 'little'))
            yield decoder.sink.last.eq(offset+32 == len(payload))
            yield
            while not (yield decoder.sink.ready):
                yield
        yield decoder.sink.valid.eq(0)
        for _ in range(40):
            yield
            assert not (yield decoder.error) and not (yield pipe.overflow)
            if (yield pipe.arithmetic_error):
                assert (yield pipe.error)
                return
        raise AssertionError('arithmetic overflow was not reported')
    run_simulation(dut, run())


@pytest.mark.parametrize('dual', [False, True])
def test_semantic_configuration_bounds_and_empty_axpy(dual):
    dut = BonsaiChannelPipe(xbuf_groups=8, dual=dual, semantic=True, n_channels=4, chan_id=2)
    # (kind, rows, groups, qn, valid). Guard bounds before truncated RAM addressing.
    configs = [(0, 1, 0, 0, False), (0, 0, 1, 0, False), (3, 1, 1, 0, False),
               (0, 1, 8, 0, True), (0, 1, 9, 0, False),
               (0, 1, 4, 1, dual), (0, 1, 5, 1, False),
               (1, 1, 32, 0, True), (1, 1, 33, 0, False),
               (1, 1, 16, 1, dual), (1, 1, 17, 1, False),
               (2, 257, 1, 0, False), (2, 256, 1, 0, True),
               (2, 128, 1, 1, dual), (2, 129, 1, 1, False),
               (2, 0, 5, 0, False), (2, 0, 4, 0, True)]
    def run():
        for kind, rows, groups, qn, valid in configs:
            yield dut.kind.eq(kind)
            yield dut.rows.eq(rows)
            yield dut.code_beats.eq(groups)
            yield dut.qn.eq(qn)
            yield dut.start.eq(1)
            yield
            yield dut.start.eq(0)
            yield
            assert bool((yield dut.error)) != valid, (kind, rows, groups, qn, dual)
            if kind == KIND_OP_AXPY and rows == 0:
                yield dut.source.ready.eq(1)
                got = 0
                for _ in range(32):
                    if (yield dut.source.valid):
                        assert (yield dut.source.wide) and (yield dut.source.data) == 0
                        got += 1
                    yield
                assert got == (groups*4 if valid else 0)
    # Configuration-only checks cannot reset an AX_ACC FSM with a start; independent
    # reset per case is handled by the complete module reset below instead.
    # The empty AXPY cases therefore run in separate freshly reset instances.
    nonempty = configs[:-2]
    configs = nonempty
    run_simulation(dut, run())
    for config in [(2, 0, 5, 0, False), (2, 0, 4, 0, True)]:
        dut = BonsaiChannelPipe(xbuf_groups=8, dual=dual, semantic=True, n_channels=4, chan_id=2)
        configs = [config]
        run_simulation(dut, run())


@pytest.fixture(scope='module')
def compiled_native_mac(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled MAC check')
    tmp_path = tmp_path_factory.mktemp('compiled-native-mac')
    from migen.fhdl import verilog
    dut = NativeMAC(160, dual=True, channel=2, channels=4)
    pipe, decoder = dut.pipe, dut.decoder
    ports = {}
    for prefix, unit, fields in [('p', pipe, ('start', 'kind', 'rows', 'code_beats', 'bias', 'rshift', 'rshift2', 'qn',
                                            'xw_group', 'xw_sub', 'xw_data', 'xw_we', 'error', 'overflow', 'arithmetic_error')),
                                 ('d', decoder, ('start', 'rows', 'groups', 'encoding', 'busy', 'done', 'error'))]:
        for name in fields:
            ports[f'{prefix}_{name}'] = getattr(unit, name)
    for name in ('valid', 'ready', 'last', 'data'):
        ports['sink_' + name] = getattr(decoder.sink, name)
    for name in ('valid', 'ready', 'row', 'head', 'wide', 'data'):
        ports['source_' + name] = getattr(pipe.source, name)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'semantic_mac.v'
    verilog.convert(dut, ios=set(ports.values()), name='semantic_mac').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/semantic_mac.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'semantic_mac', str(rtl),
                            str(harness.resolve())], capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout[-6000:] + built.stderr[-6000:]
    return obj / 'Vsemantic_mac'


def test_compiled_native_mac_matches_python_reference(tmp_path, compiled_native_mac):
    cases = [(kind, rows, groups, qn) for kind in (0, 1) for rows, groups, qn in
             [(50, 1, False), (50, 1, True), (9, 2, True), (3, 9, True)]]
    cases += [(0, 2, 136, False), (2, 29, 1, False), (2, 29, 1, True), (2, 29, 4, True)]
    for kind, rows, groups, qn in cases:
        case = make_case(kind, rows, groups, qn, seed=92+kind+groups, channel=2, channels=4)
        expected = np.concatenate(case['expected']).astype('<i4').tobytes()
        header = struct.pack('<10I', kind, rows, groups, qn, case['bias'], case['shift'], case['shift2'],
                             len(case['xb']), len(case['payload']), len(expected)//4)
        vector = tmp_path / 'vector'
        vector.write_bytes(header + case['xb'] + case['payload'] + expected)
        result = subprocess.run([str(compiled_native_mac), str(vector)], capture_output=True,
                                text=True, timeout=15)
        assert result.returncode == 0, (kind, rows, groups, qn, result.stdout, result.stderr)
    for kind in (0, 1, 2):
        for qn, negative in [(False, False), (True, False), (False, True), (True, True)]:
            case = overflow_case(kind, qn, negative=negative)
            header = struct.pack('<10I', kind, case['rows'], case['groups'], qn, case['bias'],
                                 case['shift'], case['shift2'], len(case['xb']), len(case['payload']), 0)
            vector.write_bytes(header + case['xb'] + case['payload'])
            result = subprocess.run([str(compiled_native_mac), str(vector), 'arithmetic-error'],
                                    capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, (kind, qn, negative, result.stdout, result.stderr)


def test_compiled_signed_limits_are_not_overflow(tmp_path, compiled_native_mac):
    # Exact INT32_MIN and nearby positive output; exact signed-48 accumulator
    # minimum and largest whole-group positive sum below its maximum.
    for kind, xvalue, groups, scale, bias, shift, expected in [
        (0, -128, 1, 1, 17, 0, -(1 << 31)),
        (0, 127, 1, 1, 17, 0, (1 << 31) - (1 << 24)),
        (1, -128, 64, -1, 22, 40, -128),
        (1, -128, 63, 1, 22, 40, 126),
    ]:
        width = groups * (128 if kind == 0 else 32)
        codes = np.full((groups, 128 if kind == 0 else 32), 2 if kind == 0 else -128,
                        dtype=np.uint8 if kind == 0 else np.int8)
        scales = np.full(groups, scale, dtype=np.float16)
        pack = qformats.ptq1_0_pack if kind == 0 else qformats.q8_0_pack
        payload = pack(scales, codes).tobytes()
        payload += bytes((-len(payload)) % 32)
        xb = np.full(width, xvalue, dtype=np.int8).tobytes()
        header = struct.pack('<10I', kind, 1, groups, 0, bias, shift, shift, len(xb), len(payload), 1)
        vector = tmp_path / 'signed-boundary'
        vector.write_bytes(header + xb + payload + struct.pack('<i', expected))
        result = subprocess.run([str(compiled_native_mac), str(vector)], capture_output=True,
                                text=True, timeout=15)
        assert result.returncode == 0, (kind, groups, xvalue, scale, result.stdout, result.stderr)


@pytest.mark.parametrize('name,row0', [('blk.0.ffn_gate.weight', 100), ('blk.0.ffn_down.weight', 1137)])
def test_real_27b_native_projection_slice(tmp_path, compiled_native_mac, name, row0):
    from trixium.model.gguf_reader import GGUFReader
    path = Path(__file__).resolve().parents[1] / 'models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    if not path.exists():
        pytest.skip('local 27B GGUF required')
    image = GGUFReader(path)
    tensor = image.tensors[name]
    assert tensor.type_id == qformats.GGML_PTQ1_0
    rows, cols = tensor.shape
    groups = cols//128
    raw = image.tensor_bytes(name)[:rows*groups*28]
    all_scales = qformats.ptq1_0_scales(raw).reshape(rows, groups)
    assert np.isfinite(all_scales).all()
    _, exponent = bx.fp16_decompose(all_scales)
    bias = bx.choose_scale_bias(exponent)
    shift = bx.choose_result_shift(all_scales, bias)
    # The descriptor uses the WHOLE matrix's B/R, not a newly chosen slice bias.
    payload = bytes(raw[row0*groups*28:(row0+2)*groups*28])
    scales, codes = qformats.ptq1_0_unpack(payload)
    x = np.random.default_rng(1127).integers(-128, 128, cols, dtype=np.int8)
    expected = bx.matvec_ternary(scales.reshape(2, groups), codes.reshape(2, cols), x, bias, shift)
    payload += bytes((-len(payload)) % 32)
    header = struct.pack('<10I', 0, 2, groups, 0, bias, shift, shift, cols, len(payload), 2)
    vector = tmp_path / 'real-vector'
    vector.write_bytes(header + x.tobytes() + payload + expected.astype('<i4').tobytes())
    result = subprocess.run([str(compiled_native_mac), str(vector)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, (name, result.stdout, result.stderr)

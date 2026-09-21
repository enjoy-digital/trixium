# SPDX-License-Identifier: BSD-2-Clause
from types import SimpleNamespace
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from migen import *
from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.model import bitexact as bx, qformats
from trixium.model.bons import TensorSpec, encode_rows
from test_native_reader import memory_server, region


class NativeEngineDUT(Module):
    def __init__(self, n=2, xgroups=96, dual=True):
        self.axis = [AXIInterface(data_width=256, address_width=34, id_width=4) for _ in range(n)]
        self.regions = [region(7, size=1 << 20), region(42, base=1 << 32, size=1 << 20)]
        self.submodules.sp = BonsaiScratchpad(size=64*1024)
        self.submodules.engine = BonsaiMatVecEngine(self.sp, self.axis, xbuf_groups=xgroups, dual=dual,
            chan_stride=65536, native_regions=self.regions)
        self.comb += self.sp.units_idle.eq(~self.engine.sp_busy)


def make_engine_case(dut, kind=0, rows=3, rem=0, groups=2, qn=False, ilv=False, native=True, seed=82):
    n = len(dut.axis)
    total = rows*n + rem
    cols = groups*(128 if kind == 0 else 32)
    rng = np.random.default_rng(seed)
    scales = rng.uniform(-0.03, 0.03, (total, groups)).astype(np.float16)
    values = (rng.integers(0, 3, (total, cols), dtype=np.uint8) if kind == 0 else
              rng.integers(-128, 128, (total, cols), dtype=np.int8))
    xs = [rng.integers(-128, 128, cols, dtype=np.int8) for _ in range(2 if qn else 1)]
    _, exponents = bx.fp16_decompose(scales)
    bias = bx.choose_scale_bias(exponents)
    shift = bx.choose_result_shift(scales, bias, gs_max=bx.GS_MAX if kind == 0 else bx.GS_MAX_Q8)
    expected = [(bx.matvec_ternary if kind == 0 else bx.matvec_q8)(scales, values, x, bias, shift) for x in xs]
    memory, descriptors, reads = {}, [], []
    for i in range(n):
        indices = (list(range(i, total, n)) if ilv else
                   list(range(i*rows+min(i, rem), i*rows+min(i, rem)+rows+(i < rem))))
        region_id, base = (7, 0) if i % 2 == 0 else (42, 1 << 32)
        offset = i*8192 + 16
        block_bytes = 28 if kind == 0 else 34
        stride = groups*block_bytes + 3
        descriptors.append((region_id, offset, stride))
        if native:
            for row, index in enumerate(indices):
                pack = qformats.ptq1_0_pack if kind == 0 else qformats.q8_0_pack
                payload = pack(scales[index], values[index].reshape(groups, -1)).tobytes()
                address = base+offset+row*stride
                memory.update({address+j: v for j, v in enumerate(payload)})
                reads.append((i, address & -32, (address%32 + len(payload)+31)//32))
        else:
            assert rem == 0
            spec = TensorSpec('w', layout.ROLE_ATTN_Q, 0, kind, rows, cols,
                              scales=scales[indices], values=values[indices])
            payload = encode_rows(spec, 0, rows).tobytes()
            address = 0x800 + i*65536
            memory.update({address+j: v for j, v in enumerate(payload)})
            reads.append((i, address, len(payload)//32))
    cmd = dict(native=native, kind=kind, rows=rows, rem=rem, cols=cols, cbeats=groups,
               sbeats=(groups+15)//16, wbase=0x800, wlen=rows*(groups+(groups+15)//16),
               bias=bias, rshift=shift, rshift2=shift, xsrc=0, dst=0xc004, qn=qn,
               qstride=total+3, ilv=ilv, supw=0, xred=0, chan=0, rep=0, dstride=0, sstride=0)
    return dict(cmd=cmd, descriptors=descriptors, memory=memory, reads=reads, expected=expected,
                xb=b''.join(x.tobytes() for x in xs))


def submit_engine(dut, case):
    engine = dut.engine
    xb = case['xb'] + bytes((-len(case['xb'])) % 32)
    for i in range(0, len(xb), 32):
        yield dut.sp.mem[i//32].eq(int.from_bytes(xb[i:i+32], 'little'))
    if case['cmd']['native']:
        for i, (region_id, offset, stride) in enumerate(case['descriptors']):
            yield engine.native_cfg.chan.eq(i)
            yield engine.native_cfg.region.eq(region_id)
            yield engine.native_cfg.offset.eq(offset)
            yield engine.native_cfg.stride.eq(stride)
            yield engine.native_cfg.valid.eq(1)
            yield
            for _ in range(100):
                if (yield engine.native_cfg.ready):
                    break
                yield
            else:
                raise AssertionError('native config handshake timed out')
            assert not (yield engine.native_cfg_error)
        yield engine.native_cfg.valid.eq(0)
    for name, value in case['cmd'].items():
        yield getattr(engine.cmd, name).eq(value)
    yield engine.cmd.valid.eq(1)
    yield
    for _ in range(100):
        if (yield engine.cmd.ready):
            break
        yield
    else:
        raise AssertionError(f'command handshake timed out: busy={(yield engine.busy)}, error={(yield engine.error)}')
    yield engine.cmd.valid.eq(0)
    yield


def finish_engine(dut, case):
    engine = dut.engine
    for cycle in range(4000):
        assert not (yield engine.error), f'engine error at {cycle}, cfg={(yield engine.native_cfg_error)}'
        assert not (yield engine.native_cfg.ready)
        if (yield engine.done):
            break
        yield
    else:
        raise AssertionError(f'engine timeout, results={(yield engine.dbg_result_cnt)}')
    for _ in range(3):
        yield
    for head, expected in enumerate(case['expected']):
        dst = case['cmd']['dst'] + head*case['cmd']['qstride']*4
        got = []
        for row in range(len(expected)):
            slot = dst+row*4
            word = (yield dut.sp.mem[slot//32])
            value = (word >> (8*(slot % 32))) & 0xffffffff
            got.append(value-(1 << 32) if value & (1 << 31) else value)
        assert np.array_equal(got, expected), (case['cmd'], got, expected)
    if not case['cmd']['ilv'] and not case['cmd']['qn']:
        assert (yield engine.argmax_idx) == int(np.argmax(case['expected'][0]))
        assert (yield engine.argmax_val) == int(np.max(case['expected'][0]))
    for reader in engine.readers:
        assert (yield reader.idle)


@pytest.mark.parametrize('n,kind,rows,rem,groups,qn,ilv', [
    (2, 0, 3, 1, 2, False, False),
    (4, 0, 10, 1, 1, False, False),
    (2, 1, 3, 1, 5, True, False),
    (4, 1, 0, 2, 2, False, True),
    (2, 0, 3, 0, 1, True, False),
    (1, 0, 2, 0, 136, False, False),
])
def test_native_engine_shared_fill_compute_and_gather(n, kind, rows, rem, groups, qn, ilv):
    dut = NativeEngineDUT(n, xgroups=max(16, groups), dual=qn)
    case = make_engine_case(dut, kind, rows, rem, groups, qn, ilv)
    observed = [[] for _ in dut.axis]
    def run():
        yield from submit_engine(dut, case)
        yield from finish_engine(dut, case)
    run_simulation(dut, [run()] + [memory_server(SimpleNamespace(axi=axi), case['memory'], observed[i], seed=80+i)
                                  for i, axi in enumerate(dut.axis)])
    for i in range(n):
        wanted = [addr+32*b for channel, addr, beats in case['reads'] if channel == i for b in range(beats)]
        assert [addr+32*b for addr, beats, _ in observed[i] for b in range(beats)] == wanted


def test_native_and_legacy_commands_reuse_same_engine():
    from test_region_read import submit as submit_read
    dut = NativeEngineDUT(n=2)
    cases = [make_engine_case(dut, kind=kind, native=native, seed=120+i, qn=qn)
             for i, (kind, native, qn) in enumerate([(0, False, False), (1, True, True),
                                                    (1, False, False), (0, True, False)])]
    memory, observed = {}, [[], []]
    def run():
        for case in cases:
            memory.clear()
            memory.update(case['memory'])
            yield from submit_engine(dut, case)
            yield from finish_engine(dut, case)
            payload = bytes(range(64))
            memory.update({(1 << 32)+0x100+i: b for i, b in enumerate(payload)})
            yield from submit_read(dut.engine, 42, 0x100, 2, 0xe000)
            for _ in range(300):
                yield
                assert not (yield dut.engine.error)
                if (yield dut.engine.region_read_done):
                    break
            else:
                raise AssertionError('raw read between projections timed out')
            yield
            for i in range(2):
                assert (yield dut.sp.mem[0xe000//32+i]) == int.from_bytes(payload[32*i:32*(i+1)], 'little')
    run_simulation(dut, [run()] + [memory_server(SimpleNamespace(axi=axi), memory, observed[i], seed=90+i)
                                  for i, axi in enumerate(dut.axis)])


def test_native_slim_engine_rejects_dual_before_any_io():
    dut = NativeEngineDUT(n=1, xgroups=16, dual=False)
    case = make_engine_case(dut, qn=True)
    def run():
        yield from submit_engine(dut, case)
        for _ in range(30):
            assert not (yield dut.axis[0].ar.valid)
            assert not (yield dut.engine.store_cmd.valid)
            assert not (yield dut.engine.dbg_gw_we)
            assert not (yield dut.engine.done)
            yield
        assert (yield dut.engine.native_failed)
        assert (yield dut.engine.error)
        assert (yield dut.engine.busy)
        assert not (yield dut.engine.cmd.ready)
    run_simulation(dut, run())


@pytest.fixture(scope='module')
def compiled_native_engine(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independently compiled engine check')
    from migen.fhdl import verilog
    tmp_path = tmp_path_factory.mktemp('compiled-native-engine')
    dut = NativeEngineDUT(n=2, xgroups=160, dual=True)
    engine = dut.engine
    ports = {name: getattr(engine, name) for name in ('busy', 'done', 'error', 'native_failed', 'argmax_idx', 'argmax_val')}
    engine.fsm.finalize()
    ports.update(result_count=engine.dbg_result_cnt, expected_count=engine.dbg_mv_total,
                 state=engine.fsm.state, issued=engine.native_issued, completed=engine.native_completed,
                 gather_write=engine.dbg_gw_we)
    for i, reader in enumerate(engine.native_readers):
        ports[f'native_busy{i}'] = reader.busy
        ports[f'reader_idle{i}'] = engine.readers[i].idle
    for prefix, endpoint, fields in [
        ('cmd', engine.cmd, ('valid', 'ready', 'native', 'kind', 'rows', 'cols', 'rem', 'cbeats',
                            'bias', 'rshift', 'rshift2', 'qn', 'ilv', 'dst', 'qstride', 'wbase', 'wlen',
                            'sbeats', 'xsrc', 'supw', 'xred')),
        ('cfg', engine.native_cfg, ('valid', 'ready', 'chan', 'region', 'offset', 'stride')),
    ]:
        ports.update({prefix+'_'+name: getattr(endpoint, name) for name in fields})
    for i, axi in enumerate(dut.axis):
        for prefix, endpoint, fields in [
            ('ar', axi.ar, ('valid', 'ready', 'addr', 'len', 'size', 'burst', 'id')),
            ('r', axi.r, ('valid', 'ready', 'data', 'last', 'id', 'resp')),
        ]:
            ports.update({f'{prefix}{i}_'+name: getattr(endpoint, name) for name in fields})
    # Test-only memory access port. Production scratchpad still has its two
    # physical ports and existing sideband arbiter; no synthesis inference claim.
    debug = dut.sp.mem.get_port(write_capable=True, async_read=True)
    dut.specials += debug
    ports.update(sp_addr=debug.adr, sp_data=debug.dat_w, sp_we=debug.we, sp_result=debug.dat_r)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'native_engine.v'
    verilog.convert(dut, ios=set(ports.values()), name='native_engine').write(str(rtl))
    obj = tmp_path / 'obj'
    harness = Path(__file__).parent / 'rtl/native_engine.cpp'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
                            '--Mdir', str(obj), '--top-module', 'native_engine', str(rtl),
                            str(harness.resolve())], capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stdout[-6000:] + built.stderr[-6000:]
    return obj / 'Vnative_engine'


def compiled_case(tmp_path, executable, case, *, configured=3, failure=False, seed=87):
    cmd = case['cmd']
    reads = [[addr+32*b for c, addr, beats in case['reads'] if c == i for b in range(beats)] for i in range(2)]
    addresses = sorted(set(reads[0] + reads[1]))
    xb = case['xb'] + bytes((-len(case['xb'])) % 32)
    expected = np.concatenate(case['expected']).astype('<i4').tobytes()
    header = struct.pack('<32I', 0x42534e45, cmd['native'], cmd['kind'], cmd['rows'], cmd['rem'],
                         cmd['cols'], cmd['cbeats'], cmd['bias'], cmd['rshift'], cmd['qn'], cmd['ilv'],
                         cmd['dst'], cmd['qstride'], cmd['wlen'], cmd['sbeats'], len(addresses),
                         len(reads[0]), len(reads[1]), len(expected)//4, len(xb), cmd['rshift2'],
                         cmd['xsrc'], cmd['supw'], cmd['xred'], configured, failure, 0, seed, 0, 0, 0, 0)
    header += b''.join(struct.pack('<IQQ', *desc) for desc in case['descriptors'])
    payload = xb + expected
    payload += b''.join(struct.pack('<Q', addr) + bytes(case['memory'].get(addr+j, 0xa5) for j in range(32))
                        for addr in addresses)
    payload += b''.join(struct.pack('<Q', addr) for port in reads for addr in port)
    vector = tmp_path / 'engine-vector'
    vector.write_bytes(header + payload)
    result = subprocess.run([str(executable), str(vector)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, (cmd, result.stdout, result.stderr)


def test_compiled_native_engine_exact_results_and_physical_reads(tmp_path, compiled_native_engine):
    # make_engine_case only needs the physical-port count; do not elaborate a
    # second DUT just to prepare host-reference vectors.
    model = SimpleNamespace(axis=[None, None])
    cases = [(0, 3, 1, 1, False, False, True), (0, 20, 1, 1, True, False, True),
             (1, 3, 1, 5, True, False, True), (1, 0, 1, 2, False, True, True),
             (0, 2, 0, 136, False, False, True),
             (0, 3, 0, 2, True, False, False), (1, 3, 0, 5, False, False, False)]
    for index, (kind, rows, rem, groups, qn, ilv, native) in enumerate(cases):
        case = make_engine_case(model, kind, rows, rem, groups, qn, ilv, native, seed=400+index)
        for seed in (87, 91, 12345):
            compiled_case(tmp_path, compiled_native_engine, case, seed=seed)
        if native and not qn and not ilv:
            case['cmd']['supw'] = 1
            compiled_case(tmp_path, compiled_native_engine, case)


@pytest.mark.parametrize('patch', [
    {'kind': 2}, {'kind': 3}, {'kind': 4}, {'rows': 0}, {'rem': 2},
    {'rows': 65535, 'rem': 1}, {'cbeats': 0}, {'cbeats': 161, 'cols': 20608},
    {'cols': 256 + (1 << 16)}, {'cols': 255}, {'qn': 1, 'cbeats': 81, 'cols': 10368},
    {'xsrc': 1}, {'xsrc': 65536-128}, {'xsrc': 1 << 23},
    {'dst': 3}, {'dst': 65536-4}, {'dst': 1 << 23},
    {'qn': 1, 'qstride': 5}, {'qn': 1, 'qstride': 65535},
    {'xred': 1}, {'supw': 1, 'qn': 1}, {'supw': 1, 'ilv': 1},
])
def test_compiled_native_engine_rejects_bad_geometry_without_io(tmp_path, compiled_native_engine, patch):
    case = make_engine_case(SimpleNamespace(axis=[None, None]))
    case['cmd'].update(patch)
    compiled_case(tmp_path, compiled_native_engine, case, failure=True)


@pytest.mark.parametrize('configured', [0, 1, 2])
def test_compiled_native_engine_requires_all_active_descriptors(tmp_path, compiled_native_engine, configured):
    case = make_engine_case(SimpleNamespace(axis=[None, None]))
    compiled_case(tmp_path, compiled_native_engine, case, configured=configured, failure=True)

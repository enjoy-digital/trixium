# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from migen import *
from litex.soc.interconnect.axi import AXIInterface
from litex.soc.interconnect import stream

from trixium.capabilities import Access, Reservation, ReservationKind
from trixium.gateware.axi_read_monitor import AXIReadMonitor
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.region_read_csr import BonsaiRegionReadCSR
from test_integration import write
from test_native_reader import region, memory_server


class ReadDUT(Module):
    def __init__(self):
        self.axis = [AXIInterface(data_width=256, address_width=34, id_width=4) for _ in range(2)]
        regions = [region(7, size=1 << 20, endpoint=9),
                   region(42, base=1 << 32, size=1 << 20, endpoint=5),
                   region(43, base=(1 << 34)-32, size=64, endpoint=5),
                   region(44, size=1 << 20, endpoint=6),
                   region(45, base=1 << 20, size=1 << 20, endpoint=5, access=Access.WRITE)]
        self.submodules.sp = BonsaiScratchpad(size=256*1024)
        self.submodules.engine = BonsaiMatVecEngine(self.sp, self.axis, xbuf_groups=4,
            native_regions=regions, native_endpoints=[5, 9],
            native_reservations=[Reservation(7, ReservationKind.CPU, 0, 128)])
        self.comb += self.sp.units_idle.eq(~self.engine.sp_busy)


def submit(engine, region_id=42, offset=4064, length=39, dst=64):
    cmd = engine.region_read_cmd
    for name, value in dict(region=region_id, offset=offset, length=length, dst=dst).items():
        yield getattr(cmd, name).eq(value)
    yield cmd.valid.eq(1)
    yield
    for _ in range(100):
        if (yield cmd.ready):
            break
        yield
    else:
        raise AssertionError('readback command not accepted')
    yield cmd.valid.eq(0)
    for name in ('region', 'offset', 'length', 'dst'):
        yield getattr(cmd, name).eq(0)


@pytest.mark.parametrize('rid,port,offset,base', [(42, 0, 4064, 1 << 32), (7, 1, 128, 0)])
def test_region_read_exact_shared_fetch(rid, port, offset, base):
    dut = ReadDUT()
    payload = bytes((i*37+9) & 255 for i in range(39*32))
    memory = {base+offset+i: byte for i, byte in enumerate(payload)}
    observed = [[], []]
    def run():
        for _ in range(2):
            yield from submit(dut.engine, rid, offset)
            for cycle in range(2000):
                yield
                assert not (yield dut.engine.region_read_error)
                if (yield dut.engine.region_read_done):
                    assert (yield dut.engine.region_read_beats) == 39
                    assert (yield dut.engine.readers[port].idle)
                    assert (yield dut.engine.region_read_monitors[port].drained)
                    break
            else:
                raise AssertionError('region read timed out')
            yield
            for i in range(39):
                assert (yield dut.sp.mem[2+i]) == int.from_bytes(payload[32*i:32*(i+1)], 'little')
            yield
    run_simulation(dut, [run()] + [memory_server(SimpleNamespace(axi=axi), memory, observed[i])
                                  for i, axi in enumerate(dut.axis)])
    assert not observed[1-port]
    assert [addr+32*b for addr, beats, _ in observed[port] for b in range(beats)] == [
        base+offset+32*i for i in range(39)]*2


@pytest.mark.parametrize('fault,expected', [('none', 0), ('resp', 1), ('id', 2),
    ('unmatched', 2), ('early', 4), ('missing', 4)])
def test_response_monitor_zero_latency_interleaved_and_faults(fault, expected):
    dut = Module()
    axi = AXIInterface(data_width=256, address_width=34, id_width=4)
    dut.submodules.monitor = m = AXIReadMonitor(axi)
    def run():
        yield m.enable.eq(1)
        yield axi.ar.ready.eq(1)
        yield axi.r.ready.eq(1)
        # First response arrives in the same cycle as its AR. IDs 0 and 1
        # subsequently interleave; each ID preserves its own burst order.
        events = [(0, 1, 0, False), (1, 2, 1, False), (None, 0, 0, True),
                  (None, 0, 1, False), (None, 0, 1, True)]
        for k, (arid, length, rid, last) in enumerate(events):
            yield axi.ar.valid.eq(arid is not None)
            yield axi.ar.id.eq(arid or 0)
            yield axi.ar.len.eq(length)
            yield axi.r.valid.eq(1)
            yield axi.r.id.eq((7 if fault == 'id' else 3) if k == 0 and fault in ('id', 'unmatched') else rid)
            yield axi.r.last.eq((not last) if (fault == 'early' and k == 0) or (fault == 'missing' and k == 2) else last)
            yield axi.r.resp.eq(2 if fault == 'resp' and k == 0 else 0)
            yield
        yield axi.ar.valid.eq(0)
        yield axi.r.valid.eq(0)
        for _ in range(5):
            yield
        if expected:
            assert (yield m.error) & expected
        else:
            assert not (yield m.error)
            assert (yield m.drained)
    run_simulation(dut, run())


def test_region_read_csr_snapshot_reset_and_completion_race():
    dut = Module()
    engine = SimpleNamespace(region_read_cmd=stream.Endpoint([
        ('region', 32), ('offset', 64), ('length', 24), ('dst', 24)]),
        region_read_error=Signal(8), region_read_failed=Signal(), region_read_done=Signal(),
        region_read_busy=Signal(), region_read_beats=Signal(24))
    reset = SimpleNamespace(rst_trigger=Signal(), rst_busy=Signal())
    dut.submodules.csr = csr = BonsaiRegionReadCSR(engine, reset)
    def run():
        values = dict(region=42, offset=(1 << 32)+128, length=5, dst=64)
        for name, value in values.items():
            yield from write(getattr(csr, name), value)
        yield from write(csr.start, 1)
        for name in values:
            yield from write(getattr(csr, name), 0)
        assert (yield csr.status.fields.pending)
        for name, value in values.items():
            assert (yield getattr(engine.region_read_cmd, name)) == value
        yield engine.region_read_cmd.ready.eq(1)
        yield
        yield
        assert (yield csr.accepted.status) == 1
        assert not (yield csr.status.fields.pending)
        # Duplicate start on exactly the successful-completion cycle must NOT
        # credit success; the misuse invalidates the command until full reset.
        yield engine.region_read_done.eq(1)
        yield from write(csr.start, 1)
        yield engine.region_read_done.eq(0)
        yield
        assert (yield csr.completed.status) == 0
        assert (yield csr.status.fields.failed)
        assert (yield csr.error.status) == 256
        yield reset.rst_trigger.eq(1)
        yield
        yield
        yield reset.rst_trigger.eq(0)
        yield
        assert not (yield csr.status.fields.busy)
        assert not (yield csr.error.status)
        assert (yield csr.accepted.status) == 1
        # Reset a pending command before the engine accepts it.
        yield engine.region_read_cmd.ready.eq(0)
        yield from write(csr.start, 1)
        assert (yield csr.status.fields.pending)
        yield reset.rst_busy.eq(1)
        yield
        yield
        assert not (yield engine.region_read_cmd.valid)
        assert not (yield csr.status.fields.pending)
        assert (yield csr.accepted.status) == 1
    run_simulation(dut, run())


def test_region_read_csr_abi_layout():
    engine = ReadDUT().engine
    csr = BonsaiRegionReadCSR(engine, SimpleNamespace(rst_trigger=Signal(), rst_busy=Signal()))
    offsets, offset = {}, 0
    for register in csr.get_csrs():
        offsets[register.name] = (offset, register.size)
        offset += 4*((register.size+31)//32)
    assert offsets == dict(abi=(0x00,32), region=(0x04,32), offset=(0x08,64),
        length=(0x10,24), dst=(0x14,24), start=(0x18,1), status=(0x1c,3), error=(0x20,16),
        beats=(0x24,24), accepted=(0x28,32), completed=(0x2c,32))


def test_response_monitor_queue_overflow_is_sticky():
    dut = Module()
    axi = AXIInterface(data_width=256, address_width=34, id_width=4)
    dut.submodules.monitor = m = AXIReadMonitor(axi, max_bursts=2)
    def run():
        yield m.enable.eq(1)
        yield axi.ar.valid.eq(1)
        yield axi.ar.ready.eq(1)
        for _ in range(4):
            yield
        yield axi.ar.valid.eq(0)
        yield m.enable.eq(0)
        for _ in range(5):
            yield
            assert (yield m.error) == 8
            assert not (yield m.drained)
    run_simulation(dut, run())


@pytest.fixture(scope='module')
def compiled_region_read(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independently compiled readback check')
    from migen.fhdl import verilog
    tmp = tmp_path_factory.mktemp('compiled-region-read')
    dut = ReadDUT()
    engine = dut.engine
    ports = {name: getattr(engine, 'region_read_'+name) for name in ('busy', 'done', 'error', 'failed', 'beats')}
    ports['gather_write'] = engine.dbg_gw_we
    for name in ('region', 'offset', 'length', 'dst', 'valid', 'ready'):
        ports['cmd_'+name] = getattr(engine.region_read_cmd, name)
    ports['legacy_valid'] = engine.cmd.valid
    ports['cfg_valid'] = engine.native_cfg.valid
    ports['vec_busy'] = engine.vec_busy_in
    for i, axi in enumerate(dut.axis):
        ports[f'idle{i}'] = engine.readers[i].idle
        ports[f'drained{i}'] = engine.region_read_monitors[i].drained
        for prefix, endpoint, fields in [
            ('ar', axi.ar, ('valid', 'ready', 'addr', 'len', 'size', 'burst', 'id')),
            ('r', axi.r, ('valid', 'ready', 'data', 'last', 'id', 'resp')),
        ]:
            ports.update({f'{prefix}{i}_'+name: getattr(endpoint, name) for name in fields})
    debug = dut.sp.mem.get_port(write_capable=True, async_read=True)
    dut.specials += debug
    ports.update(sp_addr=debug.adr, sp_data=debug.dat_w, sp_we=debug.we, sp_result=debug.dat_r)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp / 'region_read.v'
    verilog.convert(dut, ios=set(ports.values()), name='region_read').write(str(rtl))
    obj = tmp / 'obj'
    harness = Path(__file__).parent / 'rtl/region_read.cpp'
    result = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
        '--Mdir', str(obj), '--top-module', 'region_read', str(rtl), str(harness.resolve())],
        capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    return obj / 'Vregion_read'


@pytest.mark.parametrize('rid,offset,length,dst,physical,port,fault', [
    (42, 4064, 39, 64, (1 << 32)+4064, 0, 0),
    (7, 128, 4097, 64, 128, 1, 0),
    (42, (1 << 20)-32, 1, 256*1024-32, (1 << 32)+(1 << 20)-32, 0, 0),
    (43, 0, 1, 0, (1 << 34)-32, 0, 0),
    (99, 0, 1, 0, 0, 0, 1), (44, 0, 1, 0, 0, 0, 1), (45, 0, 1, 0, 0, 0, 1),
    (7, 0, 1, 0, 0, 0, 1), (7, 127, 1, 0, 0, 0, 1), (42, 0, 0, 0, 0, 0, 1),
    (42, (1 << 20)-32, 2, 0, 0, 0, 1), (42, 1 << 32, 1, 0, 0, 0, 1),
    (42, (1 << 64)-32, 2, 0, 0, 0, 1), (43, 0, 2, 0, 0, 0, 1),
    (42, 0, 1, 1, 0, 0, 1), (42, 0, 2, 256*1024-32, 0, 0, 1),
    (42, 0, 1, 1 << 23, 0, 0, 1), (42, 0, 1 << 23, 0, 0, 0, 1),
    *[(42, 4064, 39, 64, (1 << 32)+4064, 0, f) for f in range(2, 7)],
])
def test_compiled_region_read_bounds_data_and_errors(compiled_region_read, rid, offset, length, dst, physical, port, fault):
    result = subprocess.run([str(compiled_region_read), *map(str, (rid, offset, length, dst, physical, port, fault))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

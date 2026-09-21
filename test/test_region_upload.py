# SPDX-License-Identifier: BSD-2-Clause
import random
import re
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect.axi import AXIInterface
from litex.soc.interconnect import stream
from migen.sim import passive

from trixium.capabilities import Access, Reservation, ReservationKind
from trixium.gateware.loader import BonsaiHBMLoader
from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster
from trixium.gateware.hbm_model import BonsaiHBMChannelModel
from trixium.gateware.region_csr import BonsaiRegionUploadCSR
from test_checked_writer import write_server
from test_integration import write
from test_native_reader import region


class UploadDUT(Module):
    def __init__(self):
        self.axis = [AXIInterface(data_width=256, address_width=34, id_width=4) for _ in range(2)]
        regions = [region(7, size=1 << 20, endpoint=9, access=Access.WRITE),
                   region(42, base=1 << 32, size=1 << 20, endpoint=5, access=Access.WRITE)]
        self.submodules.loader = BonsaiHBMLoader(self.axis, regions=regions, endpoints=[5, 9],
            reservations=[Reservation(7, ReservationKind.CPU, 0, 128)])


def submit(loader, rid=42, offset=4064, length=39):
    for name, value in dict(region=rid, offset=offset, length=length).items():
        yield getattr(loader.region_cmd, name).eq(value)
    yield loader.region_cmd.valid.eq(1)
    yield
    for _ in range(100):
        if (yield loader.region_cmd.ready):
            break
        yield
    else:
        raise AssertionError('region command not accepted')
    yield loader.region_cmd.valid.eq(0)
    yield loader.region_cmd.region.eq(99)
    yield loader.region_cmd.offset.eq((1 << 64)-32)
    yield loader.region_cmd.length.eq(0)


@pytest.mark.parametrize('rid,offset,length,port,base', [(42, 4064, 39, 0, 1 << 32), (7, 128, 5, 1, 0)])
def test_region_upload_exact_data_address_and_b_drain(rid, offset, length, port, base):
    dut = UploadDUT()
    words = [random.Random(i+85).getrandbits(256) for i in range(length)]
    observed = [{}, {}]
    def run():
        for _ in range(2):
            yield from submit(dut.loader, rid, offset, length)
            for i, word in enumerate(words):
                yield dut.loader.region_data.data.eq(word)
                yield dut.loader.region_data.last.eq(i == length-1)
                yield dut.loader.region_data.valid.eq(1)
                yield
                for _ in range(2000):
                    assert not (yield dut.loader.region_error)
                    if (yield dut.loader.region_data.ready):
                        break
                    yield
                else:
                    raise AssertionError('upload payload stalled')
            yield dut.loader.region_data.valid.eq(0)
            for _ in range(2000):
                yield
                assert not (yield dut.loader.region_error)
                if (yield dut.loader.region_done):
                    assert (yield dut.loader.region_beats) == length
                    assert (yield dut.loader.writers[port].idle)
                    break
            else:
                raise AssertionError('upload did not drain')
            yield
            yield
    run_simulation(dut, [run()] + [write_server(axi, observed[i], delay_b=200, aw_waits_w=True)
                                    for i, axi in enumerate(dut.axis)])
    assert observed[port]['data'] == [(base+offset+32*i, w) for i, w in enumerate(words)]*2
    assert not observed[1-port]


@pytest.mark.parametrize('rid,offset,length', [(99, 0, 1), (7, 0, 1), (7, 127, 1),
    (7, 128, 0), (7, (1 << 20)-32, 2), (42, 1 << 32, 1), (42, (1 << 64)-32, 2)])
def test_bad_region_upload_is_rejected_before_aw(rid, offset, length):
    dut = UploadDUT()
    def run():
        yield from submit(dut.loader, rid, offset, length)
        for _ in range(20):
            yield
            for axi in dut.axis:
                assert not (yield axi.aw.valid)
                assert not (yield axi.w.valid)
            assert not (yield dut.loader.region_done)
        assert (yield dut.loader.region_failed)
        assert (yield dut.loader.region_error)
        assert not (yield dut.loader.region_cmd.ready)
    run_simulation(dut, run())


@pytest.mark.parametrize('fault', ['last', 'response', 'id', 'abort'])
def test_region_upload_faults_never_report_success(fault):
    dut = UploadDUT()
    def run():
        yield from submit(dut.loader, length=1)
        yield dut.loader.region_data.valid.eq(1)
        yield dut.loader.region_data.last.eq(fault != 'last')
        yield dut.loader.region_abort.eq(fault == 'abort')
        for _ in range(100):
            yield
            assert not (yield dut.loader.region_done)
            if (yield dut.loader.region_failed):
                break
        else:
            raise AssertionError('fault did not enter failure state')
        assert (yield dut.loader.region_error)
    run_simulation(dut, [run(), write_server(dut.axis[0], {}, response=2 if fault == 'response' else 0,
                                            bid=3 if fault == 'id' else 0)])


class UploadSoC(LiteXModule):
    def __init__(self):
        config = BonsaiClusterConfig(with_vecunit=False, scratchpad_size=16*1024,
                    native_regions=(region(size=256*1024, access=Access.READ | Access.WRITE),),
                    native_endpoints=(0,))
        port = AXIInterface(data_width=256, address_width=33, id_width=4)
        model_axi = AXIInterface(data_width=256, address_width=33, id_width=4)
        self.constants = {}
        add_bonsai_cluster(self, [port], config)
        self.model = model = BonsaiHBMChannelModel(model_axi, size=256*1024, latency=7)
        self.models = [model]
        # The stock model's freeze only withholds READ responses. Explicitly
        # hold B here as well, so write-reset tests genuinely have AXI in flight.
        self.comb += [port.ar.connect(model_axi.ar), model_axi.r.connect(port.r),
                      port.aw.connect(model_axi.aw), port.w.connect(model_axi.w),
                      model_axi.b.connect(port.b, omit={'valid', 'ready'}),
                      port.b.valid.eq(model_axi.b.valid & ~model.freeze),
                      model_axi.b.ready.eq(port.b.ready & ~model.freeze)]

    def add_constant(self, name, value=None):
        self.constants[name] = value


def csr_soc():
    return UploadSoC()


def test_region_csr_layout_matches_portable_driver_abi():
    csrs = csr_soc().zzregion.get_csrs()
    actual = {}
    offset = 0
    for csr in csrs:
        if csr.name == 'offset':
            actual['OFFSET_HI'], actual['OFFSET_LO'] = offset, offset+4
        else:
            actual[csr.name.upper()] = offset
        offset += 4*((csr.size+31)//32)
    header = Path(__file__).resolve().parents[1] / 'trixium/software/lib/bonsai_region_upload.h'
    wanted = {name: int(value, 16) for name, value in re.findall(r'BNS_UP_(\w+)=(0x[0-9a-f]+)', header.read_text())}
    assert actual == wanted


@pytest.mark.parametrize('depth', [2, 256])
def test_region_csr_credit_extremes(depth):
    dut = Module()
    loader = SimpleNamespace(region_cmd=stream.Endpoint([('region',32), ('offset',64), ('length',24)]),
        region_data=stream.Endpoint([('data',256)]), region_abort=Signal(), region_error=Signal(8),
        region_failed=Signal(), region_done=Signal(), region_busy=Signal(), region_beats=Signal(24))
    control = SimpleNamespace(rst_trigger=Signal(), rst_busy=Signal())
    dut.submodules.front = front = BonsaiRegionUploadCSR(loader, control, fifo_depth=depth)
    total = depth+2
    received = []
    @passive
    def consume():
        while True:
            yield
            if (yield loader.region_data.valid) and (yield loader.region_data.ready):
                received.append(((yield loader.region_data.data), (yield loader.region_data.last)))
    def run():
        yield loader.region_cmd.ready.eq(1)
        yield from write(front.length, total)
        yield from write(front.start, 1)
        exhausted = False
        for word in range(total*8):
            for _ in range(100):
                if (yield front.space.status):
                    break
                exhausted = True
                yield loader.region_data.ready.eq(1)
                yield
            else:
                raise AssertionError('FIFO credits never replenished')
            yield from write(front.poke, word)
            assert not (yield front.error.status)
        assert exhausted
        yield loader.region_data.ready.eq(1)
        for _ in range(total+20):
            yield
        assert (yield front.words.status) == total*8
        assert not (yield front.space.status)
    run_simulation(dut, [run(), consume()])
    assert received == [(sum((8*i+j) << (32*j) for j in range(8)), int(i==total-1)) for i in range(total)]


def arm(dut, offset, length, rid=7):
    for name, value in dict(region=rid, offset=offset, length=length).items():
        yield from write(getattr(dut.zzregion, name), value)
    yield from write(dut.zzregion.start, 1)


def wait_completed(dut, count):
    for _ in range(2000):
        assert not (yield dut.zzregion.error.status)
        if (yield dut.zzregion.completed.status) == count:
            return
        yield
    raise AssertionError('CSR upload did not complete')


def reset(dut):
    yield from write(dut.engine_csr.soft_reset, 1)
    for _ in range(90):
        yield
    yield dut.models[0].freeze.eq(0)
    for _ in range(1500):
        if not (yield dut.engine_csr.rst_busy):
            break
        yield
    else:
        raise AssertionError('region-write reset did not drain')
    assert not (yield dut.engine_csr.rst_stuck)
    assert not (yield dut.zzregion.status.fields.busy)
    assert not (yield dut.zzregion.error.status)


def test_region_upload_csr_credits_snapshot_full_reset_and_reuse():
    dut = csr_soc()
    words = [0x51000000+i for i in range(40*8)]
    def upload(offset, expected_count):
        yield from arm(dut, offset, 40)
        yield from write(dut.zzregion.offset, 0x123400000000)
        yield from write(dut.zzregion.length, 1)
        for word in words:
            for _ in range(1000):
                if (yield dut.zzregion.space.status):
                    break
                yield
            else:
                raise AssertionError('CSR upload credits never replenished')
            yield from write(dut.zzregion.poke, word)
        yield from wait_completed(dut, expected_count)
        assert (yield dut.zzregion.words.status) == len(words)
        for i in range(40):
            assert (yield dut.models[0].mem[offset//32+i]) == sum(words[8*i+j] << (32*j) for j in range(8))
    def run():
        yield from upload(0x400, 1)
        # Withheld B responses keep the command busy; reset cancels the queued
        # stream and drains accepted writes with the existing outer shim.
        yield dut.models[0].freeze.eq(1)
        yield from arm(dut, 0x4000, 40)
        for word in words[:9]:
            yield from write(dut.zzregion.poke, word)
        assert (yield dut.zzregion.status.fields.busy)
        yield from reset(dut)
        assert (yield dut.zzregion.completed.status) == 1
        yield from upload(0x8000, 2)
        assert (yield dut.zzregion.accepted.status) == 3
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


@pytest.mark.parametrize('fault', ['extra_word', 'duplicate_start', 'idle_word', 'invalid_region'])
def test_region_upload_csr_failures_require_reset(fault):
    dut = csr_soc()
    def run():
        if fault != 'idle_word':
            yield from arm(dut, 0x800, 1, rid=99 if fault == 'invalid_region' else 7)
        if fault == 'extra_word':
            for word in range(9):
                yield from write(dut.zzregion.poke, word)
        elif fault == 'duplicate_start':
            yield from write(dut.zzregion.start, 1)
        elif fault == 'idle_word':
            yield from write(dut.zzregion.poke, 42)
        for _ in range(80):
            yield
        assert (yield dut.zzregion.status.fields.failed)
        assert (yield dut.zzregion.error.status)
        assert not (yield dut.zzregion.completed.status)
        assert not (yield dut.zzregion.space.status)
        yield from reset(dut)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_csr_uploaded_native_weight_is_consumed_by_shared_projection():
    import numpy as np
    from trixium.model import qformats
    from test_native_csr import configure, wait_count
    from test_integration import submit as legacy_submit
    dut = csr_soc()
    block = qformats.ptq1_0_pack(np.ones(1, dtype=np.float16),
                                np.full((1, 128), 2, dtype=np.uint8)).tobytes()
    payload = block + bytes(32-len(block))
    def run():
        yield from arm(dut, 0x1000, 1)
        for i in range(0, 32, 4):
            yield from write(dut.zzregion.poke, int.from_bytes(payload[i:i+4], 'little'))
        yield from wait_completed(dut, 1)
        for i in range(4):
            yield dut.scratchpad.mem[i].eq(int.from_bytes(bytes([1])*32, 'little'))
        yield from configure(dut, offset=0x1000)
        yield from write(dut.zznative.mode, 1)
        for name, value in dict(kind=0, rows=1, cbeats=1, bias=0, rshift=0, xsrc=0, dst=0x800).items():
            yield from write(getattr(dut.engine_csr, name), value)
        yield from write(dut.engine_csr.submit, 1)
        yield from wait_count(dut, 1)
        assert (yield dut.scratchpad.mem[0x800//32]) & 0xffffffff == 128
        # Native upload and projection leave the same checked writer/reader
        # available to legacy STORE/FETCH commands; no duplicate transport.
        yield from write(dut.zznative.mode, 0)
        yield from legacy_submit(dut, kind=3, chan=0, wbase=0x3000, wlen=1, xsrc=0x800)
        yield from legacy_submit(dut, kind=2, chan=0, wbase=0x3000, wlen=1, dst=0x820)
        assert (yield dut.scratchpad.mem[0x820//32]) & 0xffffffff == 128
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


@pytest.fixture(scope='module')
def compiled_upload(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for independent compiled upload checks')
    from migen.fhdl import verilog
    directory = tmp_path_factory.mktemp('compiled-region-upload')
    dut = UploadDUT()
    ports = {name: getattr(dut.loader, 'region_'+name) for name in ('busy', 'done', 'failed', 'error')}
    ports['cancel'] = dut.loader.region_abort
    for prefix, endpoint, fields in [
        ('cmd', dut.loader.region_cmd, ('valid', 'ready', 'region', 'offset', 'length')),
        ('data', dut.loader.region_data, ('valid', 'ready', 'data', 'last')),
    ]:
        ports.update({prefix+'_'+field: getattr(endpoint, field) for field in fields})
    for i, axi in enumerate(dut.axis):
        for prefix, endpoint, fields in [
            ('aw', axi.aw, ('valid', 'ready', 'addr', 'len', 'id', 'size', 'burst')),
            ('w', axi.w, ('valid', 'ready', 'data', 'strb', 'last')),
            ('b', axi.b, ('valid', 'ready', 'resp', 'id')),
        ]:
            ports.update({f'{prefix}{i}_'+field: getattr(endpoint, field) for field in fields})
    for name, signal in ports.items():
        signal.name_override = name
    rtl = directory / 'region_upload.v'
    verilog.convert(dut, ios=set(ports.values()), name='region_upload').write(str(rtl))
    obj = directory / 'obj'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
            '--Mdir', str(obj), '--top-module', 'region_upload', str(rtl),
            str(Path(__file__).parent.resolve() / 'rtl/region_upload.cpp')], capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout[-4000:]+built.stderr[-4000:]
    return obj / 'Vregion_upload'


@pytest.mark.parametrize('rid,offset,length,physical,port,fault', [
    (42, 4064, 39, (1 << 32)+4064, 0, 0),
    (42, 0, 4097, 1 << 32, 0, 0),  # >256 bursts with bounded delayed B responses
    (7, 128, 1, 128, 1, 0),
    (7, (1 << 20)-32, 1, (1 << 20)-32, 1, 0),
    (7, 0, 1, 0, 1, 1), (7, 127, 1, 0, 1, 1), (7, 128, 0, 0, 1, 1),
    (99, 0, 1, 0, 0, 1), (42, (1 << 20)-32, 2, 0, 0, 1),
    (42, 0, 0x800001, 0, 0, 1), (42, 32, 0xffffff, 0, 0, 1),
    (42, (1 << 64)-32, 2, 0, 0, 1), (42, 1 << 32, 1, 0, 0, 1),
    (42, 0, 1, 1 << 32, 0, 2), (42, 0, 39, 1 << 32, 0, 3),
    (42, 0, 39, 1 << 32, 0, 4), (42, 0, 39, 1 << 32, 0, 5),
])
def test_compiled_region_upload_exact_physical_writes_and_failure(compiled_upload, rid, offset, length, physical, port, fault):
    result = subprocess.run([str(compiled_upload), *map(str, (rid, offset, length, physical, port, fault))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr


def test_compiled_region_csr_credit_batches_and_inflight_reset(tmp_path):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for compiled CSR checks')
    from migen.fhdl import verilog
    dut = csr_soc()
    ports = {name: getattr(dut.zzregion, name).status for name in
             ('error', 'space', 'words', 'beats', 'accepted', 'completed')}
    # An undriven constant CSRStatus exported directly becomes a top-level
    # INPUT in Migen's standalone conversion. Observe it through an output alias.
    ports['abi'] = Signal(32)
    dut.comb += ports['abi'].eq(dut.zzregion.abi.status)
    # Standalone conversion has no CSR bus bank to install CSRStatus's field
    # packing module. Pack the same live fields explicitly for this observation.
    ports['status'] = Signal(3)
    dut.comb += ports['status'].eq(Cat(dut.zzregion.status.fields.busy,
        dut.zzregion.status.fields.pending, dut.zzregion.status.fields.failed))
    for name in ('region', 'offset', 'length', 'start', 'poke'):
        for field in ('storage', 're'):
            ports[name+'_'+field] = getattr(getattr(dut.zzregion, name), field)
    ports.update(reset_re=dut.engine_csr.soft_reset.re, reset_storage=dut.engine_csr.soft_reset.storage,
                 reset_busy=dut.engrst.busy, reset_stuck=dut.engrst.stuck, freeze=dut.model.freeze)
    for name in ('error', 'beats', 'accepted', 'completed', 'sp_pending', 'sp_completed'):
        ports['v_'+name] = getattr(dut.zzverify, name).status
    ports['v_abi'] = Signal(32)
    dut.comb += ports['v_abi'].eq(dut.zzverify.abi.status)
    ports['v_status'] = Signal(3)
    dut.comb += ports['v_status'].eq(Cat(dut.zzverify.status.fields.busy,
        dut.zzverify.status.fields.pending, dut.zzverify.status.fields.failed))
    for name in ('region', 'offset', 'length', 'dst', 'start'):
        for field in ('storage', 're'):
            ports['v_'+name+'_'+field] = getattr(getattr(dut.zzverify, name), field)
    spdebug = dut.scratchpad.mem.get_port(async_read=True)
    dut.specials += spdebug
    ports.update(sp_addr=spdebug.adr, sp_data=spdebug.dat_r)
    ports.update(sp_adr_storage=dut.sp_csr.adr.storage, sp_adr_re=dut.sp_csr.adr.re,
                 sp_csr_data=dut.sp_csr.data_r.status)
    # Test a higher-priority CPU read retaining the sideband port. Do not
    # invent a fixed peek latency in either the driver or production hardware.
    ports['hold_peek'] = Signal()
    dut.comb += dut.scratchpad._sb_req[0].eq(ports['hold_peek'])
    debug = dut.model.mem.get_port(async_read=True, write_capable=True)
    dut.specials += debug
    ports.update(debug_addr=debug.adr, debug_data=debug.dat_r, debug_we=debug.we, debug_write=debug.dat_w)
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp_path / 'region_csr.v'
    verilog.convert(dut, ios=set(ports.values()), name='region_csr').write(str(rtl))
    obj = tmp_path / 'obj'
    library = Path(__file__).resolve().parents[1] / 'trixium/software/lib'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
            '--Mdir', str(obj), '--top-module', 'region_csr', str(rtl),
            '-CFLAGS', '-I'+str(library), str(library / 'bonsai_region_upload.c'),
            str(library / 'bonsai_region_read.c'),
            str(Path(__file__).parent.resolve() / 'rtl/region_csr.cpp')], capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout[-4000:]+built.stderr[-4000:]
    result = subprocess.run([str(obj / 'Vregion_csr')], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr

# SPDX-License-Identifier: BSD-2-Clause
"""Checked sequence reads through actual cluster, AXI shims and scratchpad."""
from pathlib import Path
import shutil
import subprocess

import pytest
from migen import C, Cat, Signal
from migen.fhdl import verilog
from litex.gen import LiteXModule
from litex.soc.interconnect.axi import AXIInterface

from trixium.capabilities import Access, Reservation, ReservationKind
from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster
from test_native_reader import region


class FetchSoC(LiteXModule):
    def __init__(self):
        self.ports = [AXIInterface(data_width=256, address_width=34, id_width=4) for _ in range(2)]
        regions = (region(7, size=1 << 20, endpoint=9),
                   region(42, base=1 << 32, size=1 << 20, endpoint=5),
                   region(43, base=(1 << 34)-32, size=64, endpoint=5),
                   region(44, size=1 << 20, endpoint=6),
                   region(45, base=1 << 20, size=1 << 20, endpoint=5, access=Access.WRITE))
        add_bonsai_cluster(self, self.ports, BonsaiClusterConfig(with_region_fetch=True,
            native_regions=regions, native_endpoints=(5, 9),
            native_reservations=(Reservation(7, ReservationKind.CPU, 0, 128),)))

    def add_constant(self, name, value=None):
        pass


@pytest.fixture(scope='module')
def compiled_region_fetch(tmp_path_factory):
    verilator = shutil.which('verilator')
    if not verilator:
        pytest.skip('Verilator required for sequenced region transport qualification')
    tmp = tmp_path_factory.mktemp('compiled-region-fetch')
    dut = FetchSoC()
    status = Signal(24)
    dut.comb += status.eq(Cat(dut.seq.busy, C(0, 7), dut.seq.status.fields.dcount))
    bus = dut.scratchpad.add_wishbone_port()
    ports = {f'host_{name}': getattr(bus, name) for name in ('cyc', 'stb', 'we', 'adr', 'dat_w', 'dat_r', 'sel', 'ack')}
    ports.update(seq_base=dut.seq.base.storage, seq_count=dut.seq.count.storage,
        seq_start=dut.seq.start.storage, seq_start_re=dut.seq.start.re, seq_status=status,
        seq_error=dut.seq.region_fetch_error.status, seq_draining=dut.seq.native_manual_drain,
        read_error=dut.engine.region_read_error, read_busy=dut.engine.region_read_busy,
        read_beats=dut.engine.region_read_beats, engine_done=dut.engine_csr.done_count.status,
        reset_value=dut.engine_csr.soft_reset.storage, reset_re=dut.engine_csr.soft_reset.re,
        reset_busy=dut.engine_csr.rst_busy, reset_stuck=dut.engine_csr.rst_stuck,
        gather_we=dut.scratchpad.gather.we, gather_adr=dut.scratchpad.gather.adr,
        manual_region=dut.zzverify.region.storage, manual_offset=dut.zzverify.offset.storage,
        manual_length=dut.zzverify.length.storage, manual_dst=dut.zzverify.dst.storage,
        manual_start=dut.zzverify.start.storage, manual_start_re=dut.zzverify.start.re,
        manual_accepted=dut.zzverify.accepted.status, manual_completed=dut.zzverify.completed.status,
        manual_error=dut.zzverify.error.status)
    for i, axi in enumerate(dut.ports):
        for prefix, endpoint, fields in [
            ('ar', axi.ar, ('valid', 'ready', 'addr', 'len', 'size', 'burst', 'id')),
            ('r', axi.r, ('valid', 'ready', 'data', 'last', 'id', 'resp'))]:
            ports.update({f'{prefix}{i}_{name}': getattr(endpoint, name) for name in fields})
    for name, signal in ports.items():
        signal.name_override = name
    rtl = tmp / 'region_fetch.v'
    converted = verilog.convert(dut, ios=set(ports.values()), name='region_fetch')
    rtl.write_text(converted.main_source)
    for name, content in converted.data_files.items():
        (tmp / name).write_text(content)
    obj = tmp / 'obj'
    built = subprocess.run([verilator, '--cc', '--exe', '--build', '-j', '2', '-Wno-fatal',
        '--Mdir', str(obj), '--top-module', 'region_fetch', str(rtl),
        str((Path(__file__).parent / 'rtl/region_fetch.cpp').resolve())], capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stdout[-6000:]+built.stderr[-6000:]
    assert 'UNOPTFLAT' not in built.stderr, built.stderr
    return obj / 'Vregion_fetch'


@pytest.mark.parametrize('rid,offset,length,dst,physical,port,fault', [
    (42, 4064, 39, 0x800, (1 << 32)+4064, 0, 0),
    (7, 128, 4097, 0x800, 128, 1, 0),
    (42, (1 << 20)-32, 1, 256*1024-32, (1 << 32)+(1 << 20)-32, 0, 0),
    (43, 0, 1, 0x800, (1 << 34)-32, 0, 0),
    (99, 0, 1, 0x800, 0, 0, 1), (44, 0, 1, 0x800, 0, 0, 1), (45, 0, 1, 0x800, 0, 0, 1),
    (7, 0, 1, 0x800, 0, 0, 1), (7, 127, 1, 0x800, 0, 0, 1), (42, 0, 0, 0x800, 0, 0, 1),
    (42, (1 << 20)-32, 2, 0x800, 0, 0, 1), (42, 1 << 32, 1, 0x800, 0, 0, 1),
    (42, (1 << 64)-32, 2, 0x800, 0, 0, 1), (43, 0, 2, 0x800, 0, 0, 1),
    (42, 0, 1, 0x801, 0, 0, 1), (42, 0, 2, 256*1024-32, 0, 0, 1),
    (42, 0, 1, 1 << 23, 0, 0, 1), (42, 0, 1 << 23, 0x800, 0, 0, 1),
    (42, 0, 1, 64, 0, 0, 1),  # Protected later descriptor, not only current one.
    *[(42, 4064, 39, 0x800, (1 << 32)+4064, 0, fault) for fault in range(2, 12)],
])
def test_compiled_sequence_region_bounds_faults_reset_and_owner(compiled_region_fetch, rid, offset, length, dst, physical, port, fault):
    run = subprocess.run([str(compiled_region_fetch), *map(str, (rid, offset, length, dst, physical, port, fault))],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout+run.stderr
    assert 'REGION_FETCH_PASS' in run.stdout
    print(run.stdout.strip())

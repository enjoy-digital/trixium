# SPDX-License-Identifier: BSD-2-Clause
import ctypes as C
from dataclasses import replace
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
from types import SimpleNamespace
import zlib

import pytest
from migen import run_simulation

from trixium.capabilities import (
    Access, Capabilities, Feature, HEADER, MAGIC, MemoryKind, MemoryRegion,
    REGION, RESERVATION, RegionAddress, Reservation, ReservationKind, read_capabilities,
)
from trixium.gateware.capabilities import BonsaiCapabilitiesCSR, add_capabilities
from trixium.gateware.integration import BonsaiClusterConfig
from litex.gen import LiteXModule

ROOT = Path(__file__).resolve().parents[1]


def example(direct=True):
    return Capabilities(150_000_000, 175_000_000, 16, 262144,
        Feature.SOFT_RESET | (Feature.REGION_IO if direct else Feature(0)), 3,
        (MemoryRegion(7, MemoryKind.HBM, 0, 1 << 32, 1 << 33,
                      access=Access.READ | Access.WRITE if direct else Access(0)),
         MemoryRegion(2, MemoryKind.DDR4, 1, 0, 1 << 32)),
        (Reservation(7, ReservationKind.CPU, 0, 256 << 20),
         Reservation(7, ReservationKind.SYSTEM, 2 << 30, 4096)))


def repair_crc(blob):
    blob = bytearray(blob)
    struct.pack_into('<I', blob, 12, 0)
    struct.pack_into('<I', blob, 12, zlib.crc32(blob))
    return bytes(blob)


def modify(blob, offset, value, fmt='<I'):
    blob = bytearray(blob)
    struct.pack_into(fmt, blob, offset, value)
    return repair_crc(blob)


def test_python_roundtrip_and_artifacts(tmp_path):
    caps = example()
    blob = caps.to_bytes()
    assert len(blob) == HEADER.size + 2*REGION.size + 2*RESERVATION.size
    assert Capabilities.from_bytes(blob) == caps
    assert caps.physical_bytes == 12 << 30
    assert caps.unreserved_bytes == (12 << 30) - (256 << 20) - 4096
    caps.write_artifacts(tmp_path)
    assert (tmp_path / 'capabilities.bin').read_bytes() == blob
    assert '"compute_pipes": 16' in (tmp_path / 'capabilities.json').read_text()


def test_region_addressing_never_truncates():
    caps = example()
    assert caps.resolve(RegionAddress(7, (1 << 32) + 32), 64, Access.WRITE) == (0, (1 << 33) + 32)
    assert caps.resolve(RegionAddress(7, (1 << 33) - 32), 32) == (0, (12 << 30) - 32)
    for address, size in [(RegionAddress(7, 0), 32), (RegionAddress(7, 2 << 30), 32),
                           (RegionAddress(7, (1 << 33) - 32), 64), (RegionAddress(8, 0), 32),
                           (RegionAddress(7, 1 << 32), 0), (RegionAddress(7, (1 << 32) + 1), 32)]:
        with pytest.raises(ValueError):
            caps.resolve(address, size)
    with pytest.raises(ValueError, match='does not support'):
        example(False).resolve(RegionAddress(7, 1 << 32), 32)
    for value in [-1, 1 << 64, True, 0.5]:
        with pytest.raises(ValueError):
            RegionAddress(7, value)


def test_reject_inconsistent_construction():
    caps = example()
    for changes in [dict(compute_pipes=0), dict(sys_hz=0), dict(compute_hz=-1),
                    dict(features=Feature(1 << 60)), dict(weight_encodings=1 << 63),
                    dict(regions=()), dict(regions=(caps.regions[0],)*2),
                    dict(features=Feature.SOFT_RESET),
                    dict(reservations=(Reservation(99, ReservationKind.CPU, 0, 32),)),
                    dict(reservations=(Reservation(7, ReservationKind.CPU, (1 << 33) - 32, 64),)),
                    dict(reservations=(caps.reservations[0],)*2)]:
        with pytest.raises(ValueError):
            replace(caps, **changes)
    for changes in [dict(size=0), dict(size=31), dict(base=1), dict(alignment=3),
                    dict(kind=99), dict(base=(1 << 64) - 32, size=64)]:
        with pytest.raises(ValueError):
            replace(caps.regions[0], **changes)
    with pytest.raises(ValueError, match='overlap'):
        replace(caps, regions=(caps.regions[0], replace(caps.regions[0], id=9)))


def test_real_csr_rom_contents_and_bounds():
    caps = example(False)
    dut = BonsaiCapabilitiesCSR(caps)
    received = bytearray()

    def gen():
        assert (yield dut.magic.status) == int.from_bytes(MAGIC, 'little')
        assert (yield dut.abi.status) == 1
        assert (yield dut.length.status) == len(caps.to_bytes())
        for index in range(len(caps.to_bytes())//4):
            yield dut.index.storage.eq(index)
            yield
            yield
            assert (yield dut.valid.status)
            received.extend((yield dut.word.status).to_bytes(4, 'little'))
        for index in [len(caps.to_bytes())//4, 0xffffffff]:
            yield dut.index.storage.eq(index)
            yield
            yield
            assert not (yield dut.valid.status)
            assert (yield dut.word.status) == 0
    run_simulation(dut, gen())
    assert Capabilities.from_bytes(received) == caps


def test_remote_register_reader_fails_closed():
    blob = example(False).to_bytes()
    current = [0]
    reg = lambda value: SimpleNamespace(read=lambda: value)
    regs = SimpleNamespace(zzcaps_magic=reg(int.from_bytes(MAGIC, 'little')), zzcaps_abi=reg(1),
        zzcaps_length=reg(len(blob)), zzcaps_index=SimpleNamespace(write=lambda x: current.__setitem__(0, x)),
        zzcaps_valid=reg(1), zzcaps_word=SimpleNamespace(read=lambda: struct.unpack_from('<I', blob, 4*current[0])[0]))
    assert read_capabilities(regs) == example(False)
    for name, value in [('zzcaps_magic', 0), ('zzcaps_abi', 2), ('zzcaps_length', 1 << 32),
                        ('zzcaps_valid', 0), ('zzcaps_word', 0)]:
        modified = SimpleNamespace(**vars(regs))
        setattr(modified, name, reg(value))
        with pytest.raises(ValueError):
            read_capabilities(modified)
    with pytest.raises(ValueError, match='no versioned capabilities'):
        read_capabilities(SimpleNamespace())


@pytest.mark.parametrize('config, expected', [
    (BonsaiClusterConfig(with_vecunit=False), Feature.SOFT_RESET | Feature.COHERENT_OVERLAP),
    (BonsaiClusterConfig(with_dual=True, with_dnrec=True), Feature.SOFT_RESET |
     Feature.COHERENT_OVERLAP | Feature.VECTOR | Feature.SEQUENCER | Feature.DUAL_Q | Feature.DELTANET_FP32),
])
def test_shared_feature_derivation(config, expected):
    soc = LiteXModule()
    regions = (MemoryRegion(0, MemoryKind.DDR4, 0, 0, 1 << 32),)
    caps = add_capabilities(soc, config, [object()]*16, regions, sys_hz=149999999)
    assert caps.features == expected
    assert caps.sys_hz == caps.compute_hz == 149999999
    assert len(caps.regions) == 1 and caps.compute_pipes == 16
    assert caps.weight_encodings == 3
    assert not caps.features & Feature.REGION_IO
    with pytest.raises(ValueError, match='already installed'):
        add_capabilities(soc, config, [], regions, sys_hz=150000000)


class CapsC(C.Structure):
    _fields_ = [('data', C.c_void_p)] + [(n, C.c_uint32) for n in
        ('sys_hz', 'compute_hz', 'compute_pipes', 'scratchpad_bytes')] + [(n, C.c_uint64) for n in
        ('features', 'weight_encodings', 'physical_bytes', 'unreserved_bytes')] + [(n, C.c_uint32) for n in
        ('region_count', 'reservation_count')]


@pytest.fixture(scope='module')
def c_api(tmp_path_factory):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('native compiler unavailable')
    path = tmp_path_factory.mktemp('caps') / 'caps.so'
    subprocess.run([cc, '-shared', '-fPIC', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    str(ROOT / 'trixium/software/lib/bonsai_caps.c'), '-o', str(path)],
                   check=True, capture_output=True)
    lib = C.CDLL(str(path))
    lib.bns_caps_open.argtypes = [C.POINTER(CapsC), C.c_void_p, C.c_size_t]
    lib.bns_caps_open.restype = C.c_int
    lib.bns_caps_resolve.argtypes = [C.POINTER(CapsC), C.c_uint32, C.c_uint64, C.c_uint64,
                                    C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_uint64)]
    lib.bns_caps_resolve.restype = C.c_int
    return lib


def test_c_reader_and_resolver_match_python(c_api):
    for caps in [example(), example(False), replace(example(), regions=(MemoryRegion(
            7, MemoryKind.HBM, 0, (1 << 64) - 64, 64, access=Access.READ),), reservations=())]:
        blob = caps.to_bytes()
        # Deliberately unaligned input, including 64-bit fields above 4 GiB.
        storage = C.create_string_buffer(b'!' + blob)
        view = CapsC()
        assert c_api.bns_caps_open(C.byref(view), C.byref(storage, 1), len(blob)) == 0
        for name in ('sys_hz', 'compute_hz', 'compute_pipes', 'scratchpad_bytes',
                     'features', 'weight_encodings', 'physical_bytes', 'unreserved_bytes'):
            assert getattr(view, name) == getattr(caps, name)
        for region_id in [7, 2, 99]:
            for offset in [0, 1, 32, 256 << 20, 2 << 30, 1 << 32, (1 << 33) - 32, (1 << 64) - 32]:
                for size in [0, 1, 32, 64, 1 << 63]:
                    for access in [Access.READ, Access.WRITE, Access.READ | Access.WRITE]:
                        endpoint, address = C.c_uint32(), C.c_uint64()
                        rc = c_api.bns_caps_resolve(C.byref(view), region_id, offset, size, access,
                                                   C.byref(endpoint), C.byref(address))
                        try:
                            expected = caps.resolve(RegionAddress(region_id, offset), size, access)
                        except ValueError:
                            assert rc < 0
                        else:
                            assert rc == 0
                            assert (endpoint.value, address.value) == expected


def test_malformed_and_mutated_blobs_agree(c_api):
    blob = example().to_bytes()
    candidates = [b'', blob[:63], blob[:-1], blob + b'\0', bytes(len(blob))]
    for offset, value in [(0, 0), (4, 2), (8, 1), (16, 0), (20, 0), (24, 0), (28, 0),
                           (32, 1 << 30), (40, 1 << 31), (48, 0xffffffff), (52, 0xffffffff),
                           (56, 32), (60, 32), (64 + 4, 99), (64 + 12, 4), (64 + 32, 3),
                           (64 + 40, 7), (64 + 80, 999), (64 + 84, 0)]:
        candidates.append(modify(blob, offset, value))
    # Corrupted CRC and well-checksummed corrupt descriptors are both covered.
    rng = random.Random(140)
    for _ in range(500):
        altered = bytearray(blob)
        altered[rng.randrange(len(blob))] ^= 1 << rng.randrange(8)
        candidates.append(repair_crc(altered) if rng.randrange(2) else bytes(altered))
    for candidate in candidates:
        storage = C.create_string_buffer(candidate)
        view = CapsC()
        rc = c_api.bns_caps_open(C.byref(view), storage, len(candidate))
        try:
            Capabilities.from_bytes(candidate)
        except (ValueError, TypeError):
            assert rc < 0
            assert not view.data  # Failed views must not remain usable.
        else:
            assert rc == 0


def test_c_parser_cross_compiles_for_rv32(tmp_path):
    cc = shutil.which('riscv64-unknown-elf-gcc')
    if not cc:
        pytest.skip('RV32 cross compiler unavailable')
    subprocess.run([cc, '-march=rv32im', '-mabi=ilp32', '-std=c11', '-O2',
                    '-Wall', '-Wextra', '-Werror', '-ffreestanding', '-c',
                    str(ROOT / 'trixium/software/lib/bonsai_caps.c'),
                    '-o', str(tmp_path / 'caps.o')], check=True, capture_output=True)


def test_host_makefile_rebuilds_on_map_switch(tmp_path):
    if not shutil.which('gcc') or not shutil.which('make'):
        pytest.skip('host build tools unavailable')
    user = tmp_path / 'user'
    user.mkdir()
    (user / 'liblitepcie').mkdir()
    (tmp_path / 'lib').mkdir()
    shutil.copyfile(ROOT / 'trixium/software/user/Makefile', user / 'Makefile')
    (user / 'bonsai_caps.c').write_text('#include "csr.h"\nint main(void) { return TEST_RESULT; }\n')
    (tmp_path / 'lib/bonsai_caps.c').write_text('void parser_fixture(void) {}\n')
    (tmp_path / 'lib/bonsai_caps.h').write_text('')
    for i, stem in enumerate(['dma', 'flash', 'helpers']):
        (user / 'liblitepcie' / f'litepcie_{stem}.c').write_text(f'void fixture{i}(void) {{}}\n')
    for i in [1, 2]:
        inc = tmp_path / f'board{i}/driver/kernel'
        inc.mkdir(parents=True)
        header = inc / 'csr.h'
        header.write_text(f'#define TEST_RESULT {i}\n')
        os.utime(header, (1000, 1000))  # Both maps much older than every object.
    previous = None
    for i in [1, 1, 2, 2, 1]:
        before = (user / 'bonsai_caps.o').stat().st_mtime_ns if previous else 0
        subprocess.run(['make', 'bonsai_caps', f'BUILD_DIR={tmp_path / ("board" + str(i))}'],
                       cwd=user, check=True, capture_output=True)
        after = (user / 'bonsai_caps.o').stat().st_mtime_ns
        assert (after != before) == (i != previous)
        assert subprocess.run([str(user / 'bonsai_caps')]).returncode == i
        previous = i


@pytest.mark.parametrize('present', [False, True])
def test_real_host_inspector_file_and_pcie_backend(tmp_path, present):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('native compiler unavailable')
    header = '#include <stdint.h>\nuint32_t litepcie_readl(int, uint32_t);\n'
    header += 'void litepcie_writel(int, uint32_t, uint32_t);\n'
    if present:
        for name, addr in [('MAGIC', 0), ('ABI', 4), ('LENGTH', 8), ('INDEX', 12), ('WORD', 16), ('VALID', 20)]:
            header += f'#define CSR_ZZCAPS_{name}_ADDR {addr}\n'
    (tmp_path / 'liblitepcie.h').write_text(header)
    stub = tmp_path / 'pcie.c'
    stub.write_text('''
#define _XOPEN_SOURCE 700
#include <stdint.h>
#include <unistd.h>
static uint32_t cursor;
uint32_t litepcie_readl(int fd, uint32_t addr) {
    uint8_t b[4];
    if (addr == 20) return 1;
    if (addr == 16) addr = 4*cursor;
    if (pread(fd, b, 4, addr) != 4) return UINT32_MAX;
    return (uint32_t)b[0] | (uint32_t)b[1]<<8 | (uint32_t)b[2]<<16 | (uint32_t)b[3]<<24;
}
void litepcie_writel(int fd, uint32_t addr, uint32_t value) {
    (void)fd; (void)addr; cursor = value;
}
''')
    exe = tmp_path / 'inspector'
    subprocess.run([cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-I' + str(tmp_path),
                    '-I' + str(ROOT / 'trixium/software/lib'),
                    str(ROOT / 'trixium/software/user/bonsai_caps.c'),
                    str(ROOT / 'trixium/software/lib/bonsai_caps.c'), str(stub),
                    '-o', str(exe)], check=True, capture_output=True)
    path = tmp_path / 'blob.bin'
    path.write_bytes(example(False).to_bytes())
    baseline = subprocess.run([str(exe), '-f', str(path)], check=True, capture_output=True, text=True).stdout
    assert 'physical_bytes=12884901888' in baseline
    assert 'base=4294967296 size=8589934592' in baseline
    live = subprocess.run([str(exe), '-d', str(path)], capture_output=True, text=True)
    if present:
        assert live.returncode == 0 and live.stdout == baseline
    else:
        assert live.returncode != 0 and 'no capability CSR map' in live.stderr
    for bad in [b'', path.read_bytes()[:-1], b'x'*65537]:
        path.write_bytes(bad)
        assert subprocess.run([str(exe), '-f', str(path)], capture_output=True).returncode != 0

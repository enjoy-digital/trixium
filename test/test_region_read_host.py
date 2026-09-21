# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'trixium/software/lib'


def test_portable_c_readback_copy_verify_and_fail_closed(tmp_path):
    compiler = shutil.which('gcc')
    if not compiler:
        pytest.skip('C compiler required')
    binary = tmp_path / 'readback-host'
    result = subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
        '-I'+str(LIB), str(LIB / 'bonsai_region_read.c'), str(ROOT / 'test/region_read_host.c'),
        '-o', str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr


def test_readback_driver_csr_layout_and_optional_peek_status():
    from test_region_upload import csr_soc
    from trixium.gateware.integration import BonsaiClusterConfig
    from test_integration import SoC
    from test_native_reader import region
    dut = csr_soc()
    actual, offset = {}, 0
    for csr in dut.zzverify.get_csrs():
        if csr.name == 'offset':
            actual['OFFSET_HI'], actual['OFFSET_LO'] = offset, offset+4
        else:
            actual[csr.name.upper()] = offset
        offset += 4*((csr.size+31)//32)
    header = (LIB / 'bonsai_region_read.h').read_text().split('enum bns_read_sp_register')[0]
    wanted = {name: int(value, 16) for name, value in re.findall(r'BNS_RB_(\w+)=(0x[0-9a-f]+)', header)}
    assert actual == wanted
    assert dut.zzverify.abi.status.reset.value == 0x10001
    no_peek = SoC(BonsaiClusterConfig(with_vecunit=False, with_sp_csr=False,
        native_regions=(region(size=256*1024),), native_endpoints=(0,)), n=1)
    assert no_peek.zzverify.abi.status.reset.value == 0x10000
    assert not hasattr(no_peek.zzverify, 'sp_completed')
    legacy = SoC(BonsaiClusterConfig(with_vecunit=False), n=1)
    assert not hasattr(legacy.sp_csr, 'read_completed')
    appended = SoC(BonsaiClusterConfig(with_vecunit=False, append_sp_csr=True,
        native_regions=(region(size=256*1024),), native_endpoints=(0,)), n=1)
    assert not hasattr(appended, 'sp_csr')
    assert hasattr(appended.zzxsp, 'read_completed')
    assert appended.zzverify.abi.status.reset.value == 0x10001
    with pytest.raises(ValueError, match='append_sp_csr'):
        BonsaiClusterConfig(with_sp_csr=False, append_sp_csr=True)


def test_readback_driver_cross_compiles_for_rv32(tmp_path):
    cc = shutil.which('riscv64-unknown-elf-gcc')
    if not cc and Path('/opt/riscv/bin/riscv64-unknown-elf-gcc').exists():
        cc = '/opt/riscv/bin/riscv64-unknown-elf-gcc'
    if not cc:
        pytest.skip('RV32 cross compiler unavailable')
    result = subprocess.run([cc, '-march=rv32ima', '-mabi=ilp32', '-std=c11', '-O2',
        '-Wall', '-Wextra', '-Werror', '-ffreestanding', '-c', str(LIB / 'bonsai_region_read.c'),
        '-o', str(tmp_path / 'readback.o')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr

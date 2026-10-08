# SPDX-License-Identifier: BSD-2-Clause
"""The actual firmware linker must enforce stack headroom, not just document it."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_static_scratch_windows(tmp_path):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('native compiler unavailable')
    source = tmp_path / 'windows.c'
    source.write_text('''
#define BONSAI_HDR_ADDR 0x40400000u
#include "static_memory.h"
_Static_assert(BONSAI_CONSOLE_LINE_ADDR >= BONSAI_HDR_ADDR + 0xa0000 + 2048*4,
               "console overlaps prompt buffer");
_Static_assert(BONSAI_CONSOLE_LINE_ADDR + BONSAI_CONSOLE_LINE_SIZE <= BONSAI_CONSOLE_PROMPT_ADDR,
               "line overlaps prompt");
_Static_assert(BONSAI_CONSOLE_PROMPT_ADDR + 64*4 <= BONSAI_FPROF_BUFFER_ADDR,
               "prompt overlaps fprof");
_Static_assert(BONSAI_FPROF_BUFFER_ADDR + 160 <= BONSAI_HDR_ADDR + 0x100000,
               "fprof overlaps heap");
_Static_assert(BONSAI_FPROF_BUFFER_ADDR % 8 == 0, "64-bit alignment");
''')
    subprocess.run([cc, '-std=c11', '-Wall', '-Werror', '-fsyntax-only',
                    '-I' + str(ROOT / 'firmware'), str(source)], check=True, capture_output=True)


@pytest.mark.parametrize('bss_bytes, success', [(4096, True), (4097, False)])
def test_linker_stack_floor(tmp_path, bss_bytes, success):
    cc = shutil.which('riscv64-unknown-elf-gcc')
    if not cc:
        pytest.skip('RISC-V cross compiler unavailable')
    generated = tmp_path / 'generated'
    generated.mkdir()
    (generated / 'output_format.ld').write_text('OUTPUT_ARCH(riscv)\n')
    (generated / 'regions.ld').write_text('''MEMORY {
main_ram : ORIGIN = 0x40000000, LENGTH = 0x100000
sram : ORIGIN = 0x10000000, LENGTH = 0x2000
}
''')
    source = tmp_path / 'stack.c'
    source.write_text(f'volatile char test_bss[{bss_bytes}];\n'
                      'void _start(void) { test_bss[0] = 1; for (;;) {} }\n')
    result = subprocess.run([cc, '-march=rv32im', '-mabi=ilp32', '-nostdlib', '-nostartfiles',
                             '-msmall-data-limit=0', '-T', str(ROOT / 'firmware/linker.ld'),
                             str(source), '-o', str(tmp_path / 'stack.elf')], cwd=tmp_path,
                            capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr
    if not success:
        assert 'firmware must leave at least 4 KiB' in result.stderr

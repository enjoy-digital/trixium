# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import shutil
import subprocess

import pytest


def test_portable_c_upload_chunking_and_fail_closed_checks(tmp_path):
    compiler = shutil.which('gcc')
    if not compiler:
        pytest.skip('C compiler required')
    root = Path(__file__).resolve().parents[1]
    library = root / 'trixium/software/lib'
    binary = tmp_path / 'region-upload-host'
    built = subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I'+str(library), str(library / 'bonsai_region_upload.c'),
            str(root / 'test/region_upload_host.c'), '-o', str(binary)], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout+built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr

# SPDX-License-Identifier: BSD-2-Clause
import subprocess
import sys
from pathlib import Path

import pytest
from migen import Memory, Module

from tools.synth_probe import emit_probe


@pytest.mark.parametrize('extra,message', [({'part': 'bad; exit'}, 'part'),
    ({'top': 'bad; exit'}, 'top'), ({'marker': 'bad; exit'}, 'marker'),
    ({'period': 0}, 'period'), ({'period': float('nan')}, 'period')])
def test_probe_invalid_arguments_never_create_artifacts(tmp_path, extra, message):
    args = dict(top='example', part='xcvu9p-fsgd2104-2L-e')
    args.update(extra)
    with pytest.raises(ValueError, match=message):
        emit_probe(Module(), set(), tmp_path / 'out', **args)
    assert not (tmp_path / 'out').exists()


def test_probe_memory_files_stay_in_output_and_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dut = Module()
    mem = Memory(32, 32, init=[1, 2, 3])
    port = mem.get_port()
    dut.specials += mem, port
    output = emit_probe(dut, {port.adr, port.dat_r}, tmp_path / 'out', top='example', part='xcvu9p-fsgd2104-2L-e')
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    assert any(name.endswith('.init') for name in original)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out']
    with pytest.raises(FileExistsError):
        emit_probe(Module(), set(), output, top='example', part='xcvu9p-fsgd2104-2L-e')
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


def test_activation_probe_cli(tmp_path):
    tool = Path(__file__).resolve().parents[1] / 'tools/synth_activation_quantize.py'
    output = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(tool), '--output-dir', str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name for p in output.iterdir()} == {'activation_quantize.v', 'synth.xdc', 'synth.tcl'}
    assert 'synth_design' in (output / 'synth.tcl').read_text()
    assert 'false_path' not in (output / 'synth.xdc').read_text()
    rejected = subprocess.run([sys.executable, str(tool), '--output-dir', str(tmp_path / 'bad'),
                               '--part', 'part; exit'], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'invalid FPGA part name' in rejected.stderr
    assert not (tmp_path / 'bad').exists()

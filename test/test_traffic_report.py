# SPDX-License-Identifier: BSD-2-Clause
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.traffic_report import METRICS, parse_record, read_windows, summarize_window

ROOT = Path(__file__).resolve().parents[1]


def record(**overrides):
    r = dict(v=1, valid=1, window=1, sources=2, source=0, kind=1, index=0,
             word_bytes=32, mask=31, epoch=7, cycles=500, clock_hz=150000000,
             read_bytes=128, write_bytes=64, cmd_stalls=5, data_stalls=0, errors=0)
    r.update(overrides)
    return 'MEM_TRAFFIC ' + ' '.join(f'{k}={v}' for k, v in r.items())


@pytest.mark.parametrize('overrides', [
    {'v': 2}, {'valid': 0}, {'source': 2}, {'sources': 65}, {'cycles': 0},
    {'clock_hz': 0}, {'epoch': 1 << 64}, {'kind': 3}, {'mask': 7},
    {'read_bytes': 129}, {'read_bytes': 32000}, {'cmd_stalls': 501},
    {'data_stalls': 501}, {'errors': 1001}, {'index': 256}, {'word_bytes': 0},
    {'kind': 2, 'mask': 7, 'errors': 1}, {'extra': 0}, {'read_bytes': '-32'},
])
def test_invalid_record(overrides):
    with pytest.raises(ValueError):
        parse_record(record(**overrides))


def test_missing_duplicate_and_truncated_records(tmp_path):
    good = record() + '\n' + record(source=1, kind=2, word_bytes=64, mask=7)
    for content in ['', record(), good + '\nMEM_TRAFFIC v=1', record() + '\n' + record(),
                    good + '\n' + good,
                    record(sources=1) + '\n' + record(sources=1),
                    good.replace('source=1', 'source=0'), good.replace('kind=2', 'kind=1'),
                    good.replace('mask=7', 'mask=7 mask=7'), good.replace('cycles=500', '', 1)]:
        path = tmp_path / 'log'
        path.write_text(content)
        with pytest.raises(ValueError):
            read_windows(path)
    path.write_text('unrelated console output\n' + good)
    windows = read_windows(path)
    assert len(windows) == 1
    assert [tier['read_bytes'] for tier in windows[0]['tiers']] == [128, 128]
    assert windows[0]['tiers'][1]['errors'] is None
    assert windows[0]['tiers'][0]['read_GB_s'] == pytest.approx(.0384)
    path.write_text(good + '\n' + good.replace('window=1', 'window=2'))
    assert len(read_windows(path)) == 2
    for changed in [dict(source=1, epoch=8), dict(source=1, cycles=501), dict(source=1)]:
        with pytest.raises(ValueError):
            summarize_window([parse_record(record()), parse_record(record(**changed))])


@pytest.mark.parametrize('available', [False, True])
def test_actual_firmware_traffic(tmp_path, available):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('native C compiler unavailable')
    generated = tmp_path / 'generated'
    generated.mkdir()
    (generated / 'soc.h').write_text('#define CONFIG_CLOCK_FREQUENCY 150000000\n')
    header = '''#include <stdint.h>
extern uint32_t test_version, test_count, test_valid, test_select, test_info[2], test_mask[2];
extern uint64_t test_epoch, test_cycles, test_values[2][5];
extern int test_conflict;
'''
    if available:
        header += '''#define CSR_ZTRAFFIC_VERSION_ADDR 1
static inline void ztraffic_snapshot_write(uint32_t v) { (void)v; }
static inline void ztraffic_select_write(uint32_t v) { test_select = v; }
'''
        for name, value, typ in [('version', 'test_version', 'uint32_t'),
                                  ('count', 'test_count', 'uint32_t'),
                                  ('valid', 'test_valid', 'uint32_t'),
                                  ('epoch', 'test_epoch', 'uint64_t'),
                                  ('cycles', 'test_cycles + (test_conflict ? test_conflict++ : 0)', 'uint64_t'),
                                  ('info', 'test_info[test_select]', 'uint32_t'),
                                  ('mask', 'test_mask[test_select]', 'uint32_t')]:
            header += f'static inline {typ} ztraffic_{name}_read(void) {{ return {value}; }}\n'
        for i, name in enumerate(METRICS):
            header += (f'static inline uint64_t ztraffic_{name}_read(void) '
                       f'{{ return test_values[test_select][{i}]; }}\n')
    (generated / 'csr.h').write_text(header)
    exe = tmp_path / 'traffic-test'
    subprocess.run([cc, '-std=c11', '-Wall', '-Wextra', '-Werror',
                    '-DBONSAI_TRAFFIC_HOST_TEST',
                    '-I' + str(tmp_path), '-I' + str(ROOT / 'firmware'),
                    str(ROOT / 'firmware/traffic.c'), str(ROOT / 'test/traffic_host.c'),
                    '-o', str(exe)], check=True, capture_output=True, text=True)
    reasons = {1: 'reset_or_clock', 2: 'counter_reset_or_mask', 3: 'source_count',
               4: 'descriptor_changed', 5: 'reset_or_selector', 6: 'version',
               7: 'reset_or_clock', 8: 'reset_or_clock', 9: 'duplicate',
               10: 'counter_reset_or_mask', 11: 'snapshot_replaced', 12: 'no_end', 14: 'no_start'}
    for mode in range(15):
        output = subprocess.run([str(exe), str(mode)], check=True,
                                capture_output=True, text=True).stdout
        if not available or mode in reasons:
            assert 'valid=0' in output and 'valid=1' not in output
            assert 'reason=' + (reasons[mode] if available else 'absent') in output
        else:
            result = summarize_window([parse_record(line) for line in output.splitlines()])
            assert result['cycles'] == 500
            assert len(result['tiers']) == 2
            assert all(t['read_bytes'] == 128 and t['write_bytes'] == 64 for t in result['tiers'])

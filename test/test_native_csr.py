# SPDX-License-Identifier: BSD-2-Clause
import numpy as np
from migen import run_simulation

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.model import qformats
from test_integration import SoC, write
from test_native_reader import region


def configure(dut, offset=15, region_id=7):
    for name, value in dict(chan=0, region=region_id, offset=offset, stride=31).items():
        yield from write(getattr(dut.zznative, name), value)
    yield from write(dut.zznative.commit, 1)


def wait_count(dut, count):
    for _ in range(1500):
        assert not (yield dut.engine.error)
        if (yield dut.engine_csr.done_count.status) == count:
            for _ in range(3):
                yield
            return
        yield
    raise AssertionError('native CSR operation timed out')


def test_native_csr_snapshot_queue_order_and_full_reset():
    config = BonsaiClusterConfig(with_vecunit=False, scratchpad_size=16*1024,
                                native_regions=(region(size=256*1024),), native_endpoints=(0,))
    dut = SoC(config, n=1)
    positive = qformats.ptq1_0_pack(np.ones(1, dtype=np.float16), np.full((1, 128), 2, dtype=np.uint8)).tobytes()
    zero = qformats.ptq1_0_pack(np.ones(1, dtype=np.float16), np.ones((1, 128), dtype=np.uint8)).tobytes()
    memory = bytearray([0xa5])*(256*1024)
    for offset, payload in [(15, positive), (8192, zero)]:
        for row in range(3):
            memory[offset+row*31:offset+row*31+28] = payload
    def reset():
        yield from write(dut.engine_csr.soft_reset, 1)
        for _ in range(90):
            yield
        yield dut.models[0].freeze.eq(0)
        for _ in range(1000):
            if not (yield dut.engine_csr.rst_busy):
                break
            yield
        else:
            raise AssertionError('native reset/drain did not finish')
        assert not (yield dut.engine_csr.rst_stuck)
        assert (yield dut.zznative.configured.status) == 0
        assert not (yield dut.zznative.status.fields.pending)
        assert not (yield dut.zznative.status.fields.failed)
    def check_results(offset, value):
        for row in range(3):
            data = (yield dut.scratchpad.mem[(offset+row*4)//32])
            assert (data >> (8*((offset+row*4) % 32))) & 0xffffffff == value
    def run():
        for base in (0, 8192):
            for pos in range(base, base+128, 32):
                yield dut.models[0].mem[pos//32].eq(int.from_bytes(memory[pos:pos+32], 'little'))
        for word in range(4):
            yield dut.scratchpad.mem[word].eq(int.from_bytes(bytes([1])*32, 'little'))
        yield from configure(dut)
        yield from write(dut.zznative.mode, 1)
        for name, value in dict(kind=0, rows=3, cols=17, cbeats=1, bias=0, rshift=0,
                                xsrc=0, dst=0x1000).items():
            yield from write(getattr(dut.engine_csr, name), value)
        # Native K derives from cbeats, NOT the deliberately wrong legacy cols CSR.
        yield dut.models[0].freeze.eq(1)
        yield from write(dut.engine_csr.submit, 1)
        for _ in range(35):
            yield
        assert (yield dut.engine.busy)
        yield from configure(dut, offset=8192)
        assert (yield dut.zznative.status.fields.pending)
        yield from write(dut.zznative.offset, 0xdead00000000)
        yield from write(dut.zznative.commit, 1)  # Reject rather than overwrite queued descriptor.
        assert (yield dut.zznative.status.fields.queue_error)
        yield from write(dut.engine_csr.dst, 0x1400)
        yield from write(dut.engine_csr.submit, 1)
        assert (yield dut.engine_csr.status.fields.pending)
        yield dut.models[0].freeze.eq(0)
        yield from wait_count(dut, 2)
        yield from check_results(0x1000, 128)
        yield from check_results(0x1400, 0)
        assert (yield dut.zznative.accepted.status) == 2
        yield from write(dut.zznative.clear_error, 1)
        assert not (yield dut.zznative.status.fields.queue_error)

        # Unknown region: failed, no successful done count, and reset is required.
        yield from configure(dut, region_id=99)
        yield from write(dut.engine_csr.submit, 1)
        for _ in range(150):
            yield
            if (yield dut.zznative.status.fields.failed):
                break
        else:
            raise AssertionError('invalid region did not fail')
        assert (yield dut.engine_csr.done_count.status) == 2
        assert (yield dut.engine_csr.status.fields.error)
        assert (yield dut.zznative.reader_error.status) == 1
        yield from reset()

        # Cancel a native operation with AXI requests in flight; queue a config
        # update too. The common soft reset must discard BOTH, drain, then reuse.
        yield from configure(dut)
        yield dut.models[0].freeze.eq(1)
        yield from write(dut.engine_csr.submit, 1)
        for _ in range(35):
            yield
        yield from configure(dut, offset=8192)
        assert (yield dut.zznative.status.fields.pending)
        yield from reset()
        assert (yield dut.engine_csr.done_count.status) == 2
        yield from configure(dut)
        yield from write(dut.engine_csr.submit, 1)
        yield from wait_count(dut, 3)
        yield from check_results(0x1400, 128)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})
    assert 'BONSAI_HAS_NATIVE_ENGINE' in dut.constants

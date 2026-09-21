# SPDX-License-Identifier: BSD-2-Clause
from migen import *

from trixium.gateware.scratchpad import BonsaiScratchpad
from test_integration import write


def test_checked_peek_waits_for_capture_and_cancels_pending_read():
    dut = Module()
    dut.submodules.sp = sp = BonsaiScratchpad(size=1024)
    dut.submodules.csr = csr = sp.add_csr_port(with_read_status=True)
    def read(address, wanted, count):
        yield from write(csr.adr, address)
        for _ in range(30):
            if (yield csr.read_completed) == count:
                assert not (yield csr.read_pending)
                assert (yield csr.data_r.status) == wanted
                return
            yield
        raise AssertionError('scratchpad peek did not capture')
    def run():
        yield sp.mem[0].eq(sum((0xaabb0000+i) << (32*i) for i in range(8)))
        yield sp.mem[1].eq(sum((0xccdd0000+i) << (32*i) for i in range(8)))
        yield sp.sbr_allow.eq(1)
        yield from read(4, 0xaabb0001, 1)
        yield sp.sbr_allow.eq(0)
        yield from write(csr.adr, 36)
        for _ in range(12):
            yield
            assert (yield csr.read_pending)
            assert (yield csr.data_r.status) == 0xaabb0001
            assert (yield csr.read_completed) == 1
        yield sp.sbr_allow.eq(1)
        for _ in range(10):
            yield
        assert (yield csr.read_completed) == 2
        assert (yield csr.data_r.status) == 0xccdd0001
        yield sp.sbr_allow.eq(0)
        yield from write(csr.adr, 16)
        yield csr.read_reset.eq(1)
        for _ in range(5):
            yield
        assert not (yield csr.read_pending)
        assert (yield csr.read_completed) == 2
        yield csr.read_reset.eq(0)
        yield sp.sbr_allow.eq(1)
        yield from read(28, 0xaabb0007, 3)
        yield from read(32, 0xccdd0000, 4)
    run_simulation(dut, run())

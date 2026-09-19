#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The XCU1525's IDELAYCTRL reset sequence releases the sys domain.

The v2 bitstream held the whole SoC in reset forever (PCIe up, every CSR 0xffffffff): its
reset counter lived in `sys`, which LiteX holds in reset until the IDELAYCTRL is ready,
which needs the counter to release RST. This models the IDELAYCTRL (RDY a few cycles after
RST drops) and checks that RST drops on its own and sys.rst follows.
"""

import os
import sys

from migen import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_idelayctrl_reset_releases_sys():
    from trixium_xcu1525 import _IDELAYCTRL

    class _DUT(Module):
        def __init__(self):
            self.cd_sys = ClockDomain("sys")
            self.cd_ref = ClockDomain("idelay")
            self.submodules.ic = _IDELAYCTRL(None, self.cd_ref, self.cd_sys, reset_cycles=8, ready_cycles=4)
            # The IDELAYCTRL model: RDY rises 5 cycles after RST is released.
            since = Signal(4)
            self.sync.idelayrst += If(self.ic.ic_reset, since.eq(0)).Elif(since != 15, since.eq(since + 1))
            self.comb += self.ic.ic_ready.eq(since >= 5)

    dut = _DUT()
    seen = {"rst_dropped": None, "sys_in_reset": None, "sys_released": None, "rst_back": None}

    def gen():
        # sys.rst starts low (its register's init), goes high once the ic domain runs, and
        # must come back low after RDY - with RST staying low throughout (v2's failure was
        # RST re-asserting when sys reset the counter that drives it).
        for t in range(80):
            rst = (yield dut.ic.ic_reset)
            srst = (yield dut.cd_sys.rst)
            if seen["rst_dropped"] is None and not rst:
                seen["rst_dropped"] = t
            if seen["rst_dropped"] is not None and rst and seen["rst_back"] is None:
                seen["rst_back"] = t
            if seen["sys_in_reset"] is None and srst:
                seen["sys_in_reset"] = t
            if seen["sys_in_reset"] is not None and seen["sys_released"] is None and not srst:
                seen["sys_released"] = t
            yield
        assert seen["rst_dropped"] is not None, "IDELAYCTRL RST never released"
        assert seen["rst_back"] is None, f"RST re-asserted at {seen['rst_back']} (the v2 deadlock)"
        assert seen["sys_in_reset"] is not None and seen["sys_released"] is not None, seen
        assert seen["sys_released"] > seen["rst_dropped"], seen

    run_simulation(dut, [gen()], clocks={"sys": 10, "idelay": 4, "idelayrst": 10, "ic": 10})

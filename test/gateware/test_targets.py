#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The targets (trixium_xcu1525.py, trixium_fk33.py, trixium_sim.py): the XCU1525's CRG, and the
one shared core construction every target uses."""

import ast

from migen import *

from common import ROOT

# XCU1525 CRG --------------------------------------------------------------------------------------
# The XCU1525's IDELAYCTRL reset sequence releases the sys domain.
#
# A reset counter in `sys` would deadlock (the whole SoC held in reset, PCIe up, every CSR
# 0xffffffff): LiteX holds `sys` in reset until the IDELAYCTRL is ready, which needs the counter
# to release RST. This models the IDELAYCTRL (RDY a few cycles after
# RST drops) and checks that RST drops on its own and sys.rst follows.

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
        # must come back low after RDY - with RST staying low throughout (the deadlock: RST
        # re-asserting when sys resets the counter that drives it).
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
        assert seen["rst_back"] is None, f"RST re-asserted at {seen['rst_back']} (the reset deadlock)"
        assert seen["sys_in_reset"] is not None and seen["sys_released"] is not None, seen
        assert seen["sys_released"] > seen["rst_dropped"], seen

    run_simulation(dut, [gen()], clocks={"sys": 10, "idelay": 4, "idelayrst": 10, "ic": 10})


def test_idelayctrl_reset_waits_for_mmcm_lock():
    """RST stays asserted while the MMCM is unlocked, drops reset_cycles after lock, and
    comes back (counter reloaded) if lock is lost - UG571's "reset after REFCLK is stable"."""
    from trixium_xcu1525 import _IDELAYCTRL

    class _DUT(Module):
        def __init__(self):
            self.cd_sys = ClockDomain("sys")
            self.cd_ref = ClockDomain("idelay")
            self.locked = Signal()
            self.submodules.ic = _IDELAYCTRL(None, self.cd_ref, self.cd_sys, reset_cycles=8,
                                             ready_cycles=4, locked=self.locked)
            since = Signal(4)
            self.sync.idelayrst += If(self.ic.ic_reset, since.eq(0)).Elif(since != 15, since.eq(since + 1))
            self.comb += self.ic.ic_ready.eq(since >= 5)

    dut = _DUT()
    trace = []

    def gen():
        # 30 cycles unlocked: RST must not drop. Then lock: RST drops ~8 (+2 sync) cycles
        # later and sys.rst follows. Then a lock loss re-asserts RST at once.
        for t in range(30):
            assert (yield dut.ic.ic_reset) == 1, f"RST released at {t} before MMCM lock"
            yield
        yield dut.locked.eq(1)
        for t in range(30):
            trace.append(((yield dut.ic.ic_reset), (yield dut.cd_sys.rst)))
            yield
        drops = [i for i, (r, _) in enumerate(trace) if r == 0]
        assert drops, "RST never released after lock"
        assert 8 <= drops[0] <= 12, f"RST released {drops[0]} cycles after lock (want ~8+sync)"
        assert any(s == 0 for r, s in trace[drops[0]:]), "sys.rst never released"
        yield dut.locked.eq(0)
        for t in range(6):
            yield
        assert (yield dut.ic.ic_reset) == 1, "RST not re-asserted on lock loss"

    run_simulation(dut, [gen()], clocks={"sys": 10, "idelay": 4, "idelayrst": 10, "ic": 10})


def test_serializer_clock_group_uses_related_buffer_outputs():
    from litex_boards.platforms.sqrl_xcu1525 import Platform
    from trixium_xcu1525 import _CRG
    platform = Platform()
    crg = _CRG(platform, 125e6, ddr_clk_freq=187.5e6)
    groups = [(command, signals) for command, signals in platform.constraint_manager.platform_commands
              if 'CLOCK_DELAY_GROUP' in command]
    assert len(groups) == 1
    command, signals = groups[0]
    assert signals['slow'] is crg.cd_ddr.clk
    assert signals['fast'] is crg.cd_ddr4x.clk
    assert command.format(slow='ddr_clk', fast='ddr4x_clk') == (
        'set_property CLOCK_DELAY_GROUP trixium_ddr4 [get_nets {ddr_clk ddr4x_clk}]')
    assert 'false_path' not in command and 'uncertainty' not in command


def test_serializer_buffers_share_mmcm_output_and_preserve_fabric_clock():
    """Structural test, not a primitive simulation or a routed timing result."""
    from litex_boards.platforms.sqrl_xcu1525 import Platform
    from trixium_xcu1525 import _CRG

    crg = _CRG(Platform(), 125e6, ddr_clk_freq=187.5e6)
    buffers = [special for special in crg._fragment.specials if isinstance(special, Instance)]
    assert len(buffers) == 2
    by_output = {}
    for buffer in buffers:
        pins = {item.name: item.expr for item in buffer.items
                if isinstance(item, (Instance.Input, Instance.Output))}
        params = {item.name: item.value for item in buffer.items
                  if isinstance(item, Instance.Parameter)}
        assert pins["I"] is crg.cd_pll4x.clk
        assert pins["CE"].value == 1
        by_output[id(pins["O"])] = buffer.of, pins, params

    slow_type, slow_pins, slow_params = by_output[id(crg.cd_ddr.clk)]
    assert slow_type == "BUFGCE_DIV"
    assert slow_params["BUFGCE_DIVIDE"].value == 4
    fast_type, fast_pins, fast_params = by_output[id(crg.cd_ddr4x.clk)]
    assert fast_type == "BUFGCE"
    assert not fast_params

# Shared Core Construction -------------------------------------------------------------------------

def test_targets_use_one_cluster_implementation():
    """The former source-copy guard is replaced by a shared construction boundary.

    Functional seam/reset/store tests live in test/system/test_integration.py; these checks
    prevent a target from silently reinstating its own divergent third copy.
    Internal Migen names are allowed to change, but require fresh physical gates.
    """
    root = ROOT
    for fname in ("trixium_fk33.py", "trixium_xcu1525.py", "trixium_sim.py"):
        tree = ast.parse((root / fname).read_text())
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert calls.count("TriXiumCore") == 1, fname
        for constructor in ("MatVecEngine", "VectorUnit", "Sequencer",
                            "EngineCSR", "Loader", "EngineResetDomain"):
            assert constructor not in calls, (fname, constructor)

    portable = (root / "gateware/core/core.py").read_text()
    assert "vecunit.seq_seed7.eq(seq.vec_seed7)" in portable
    assert "platform.add_ip" not in portable
    assert "add_platform_command" not in portable

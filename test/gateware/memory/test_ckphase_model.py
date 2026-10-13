#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The simulation's DDR3 PHY model (gateware/memory/ckphase_model.py) against the PHY it stands for,
LiteDRAM's K7DDRPHY in CK phase mode as trixium_ypcb.py builds it: the latencies the controller
schedules with and the capabilities liblitedram's leveling is compiled against (sdram_phy.h).
"""

from migen.genlib.record import Record

from litedram.common import PHYPadsReducer
from litedram.modules import MT41K256M8
from litedram.phy.model import SDRAMPHYModel
from litedram.phy.s7ddrphy import K7DDRPHY

from gateware.memory.ckphase_model import ckphase_model_settings, CKPhasePHYModel

FREQ = 400e6/3 # The YPCB's ddr domain (DDR3-1066).


def ddr3_pads(lanes=9):
    """A YPCB DDR3 channel: 15 address bits, 8 banks, nine x8 lanes (the 9th is ECC)."""
    return Record([
        ("a", 15), ("ba", 3), ("ras_n", 1), ("cas_n", 1), ("we_n", 1), ("cs_n", 1),
        ("cke", 1), ("odt", 1), ("reset_n", 1),
        ("dq", 8*lanes), ("dqs_p", lanes), ("dqs_n", lanes), ("clk_p", 1), ("clk_n", 1),
    ])


def test_model_settings_match_the_phy():
    phy = K7DDRPHY(PHYPadsReducer(ddr3_pads(), list(range(8))), memtype="DDR3",
        sys_clk_freq=FREQ, cl=7, cwl=6, with_odelay=False, with_ck_phase_shift=True)
    module = MT41K256M8(FREQ, "1:4")
    model  = CKPhasePHYModel(SDRAMPHYModel(module=module,
        settings=ckphase_model_settings(FREQ, 64, cl=7, cwl=6), clk_freq=FREQ))
    p, m = phy.settings, model.settings
    for k in ("memtype", "databits", "dfi_databits", "nphases", "cl", "cwl", "read_latency",
              "write_latency", "read_leveling", "write_leveling", "write_latency_calibration",
              "delays", "bitslips", "ckphase_steps"):
        assert getattr(m, k, False) == getattr(p, k, False), k
    assert m.rdphase == p.rdphase.reset.value
    assert m.wrphase == p.wrphase.reset.value

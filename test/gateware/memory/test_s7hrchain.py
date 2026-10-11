#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The calibration's simulated channel behind LiteDRAM's DFII (trixium_sim.py --s7hr-model):
LiteDRAMCore -> S7HRPHYModel -> SDRAMPHYModel, driven through the DFII and PHY CSRs as
firmware/ddr_s7hr.c does: a DFI write/read round trip reads back right at a calibrated state
(each lane at its write latency, each bit at its eye) and wrong at the reset state."""

import random

from migen import *

from litedram.core import LiteDRAMCore

from gateware.memory.s7hrddrphy import S7HRPHYModel

from common.verilator import run


def test_dfii_round_trip_through_the_channel_model():
    import trixium_sim as ts

    class _DUT(Module):
        def __init__(self):
            phy, module = ts.ddr3_phy(bytearray(64), ts.DDR3_SIM_SIZE)
            self.submodules.phy  = S7HRPHYModel(phy, **ts.S7HR_CHANNEL)
            self.submodules.core = LiteDRAMCore(phy=self.phy, geom_settings=module.geom_settings,
                timing_settings=module.timing_settings, clk_freq=ts.DDR3_SIM_FREQ)
            self.port = self.core.crossbar.get_port() # Idle (the crossbar needs a master).
            # The CSR()s' write strobes: undriven without a CSR bank, inputs the testbench drives.
            self.strobes = {f"pi{p}_issue": getattr(self.core.dfii, f"pi{p}")._command_issue.wr_stb
                for p in range(4)}
            self.strobes.update({n: getattr(self.phy, "_" + n).wr_stb for n in ("rdly_dq_rst",
                "rdly_dq_inc", "rdly_dq_bitslip_rst", "rdly_dq_bitslip", "wdly_dq_bitslip_rst",
                "wdly_dq_bitslip")})
            self.strobes["ck_inc"] = self.phy.ckphase._inc.wr_stb

    dut  = _DUT()
    dfii = dut.core.dfii
    phy  = dut.phy
    ch   = ts.S7HR_CHANNEL
    pis  = [getattr(dfii, f"pi{p}") for p in range(4)]
    # Without a CSR bank the CSRStorages' field assignments are not in the design: the testbench
    # drives the fields themselves.
    signals = {f"control_{f}": getattr(dfii._control.fields, f) for f in ("sel", "cke", "odt", "reset_n")}
    signals.update({"rst": phy._rst.storage,
               "dly_sel": phy._dly_sel.storage, "dq_dly_sel": phy._dq_dly_sel.storage,
               "ck_value": phy.ckphase._status.fields.value})
    signals.update(dut.strobes)
    for p, pi in enumerate(pis):
        signals.update({f"pi{p}_{f}": getattr(pi._command.fields, f)
            for f in ("cs", "we", "cas", "ras", "wren", "rden")})
        signals[f"pi{p}_address"]  = pi._address.storage
        signals[f"pi{p}_baddress"] = pi._baddress.storage
        signals[f"pi{p}_wrdata"]   = pi._wrdata.storage
        signals[f"pi{p}_rddata"]   = pi._rddata.status
    wrphase, rdphase = phy.settings.wrphase, phy.settings.rdphase
    results = {}
    CK = 20 # Outside every lane's crossing zone (the channel's crossings: -30..5).

    def strobe(name, n=1):
        for _ in range(n):
            yield dut.strobes[name].eq(1)
            yield
            yield dut.strobes[name].eq(0)
            for _ in range(16):     # The CK phase step takes 12 cycles.
                yield

    def command(p, cmd, address=0, wait=24):
        yield pis[p]._address.storage.eq(address)
        yield pis[p]._baddress.storage.eq(0)
        for i, f in enumerate(("cs", "we", "cas", "ras", "wren", "rden")):
            yield getattr(pis[p]._command.fields, f).eq((cmd >> i) & 1)
        yield from strobe(f"pi{p}_issue")
        for _ in range(wait):
            yield

    def round_trip(seed):
        rng  = random.Random(seed)
        data = [rng.getrandbits(128) for _ in range(4)]
        yield from command(0, 0b001001)                       # ACT (cs ras).
        for p in range(4):
            yield pis[p]._wrdata.storage.eq(data[p])
        yield from command(wrphase, 0b010111)                 # WR (cs we cas wren).
        yield from command(rdphase, 0b100101, wait=40)        # RD (cs cas rden).
        got = []
        for p in range(4):
            got.append((yield pis[p]._rddata.status))
        yield from command(0, 0b001011, address=0x400)        # PRE all (cs we ras).
        return got == data

    def gen():
        for f, v in (("sel", 0), ("cke", 1), ("odt", 1), ("reset_n", 1)):  # Software control.
            yield getattr(dfii._control.fields, f).eq(v)
        for _ in range(10):
            yield
        yield from strobe("ck_inc", CK)
        assert (yield phy.ckphase._status.fields.value) == CK
        results["reset"] = yield from round_trip(1)
        # Calibrated at CK: each lane at its latency, each bit at its eye's centre
        # (bitslip*12 + tap = v0 + 0.215*CK).
        drift = ((CK + 128)*55 >> 8) - 27
        for l in range(8):
            need = (ch["base_wl"][l] + 2) & 7 if CK >= ch["cross"][l] else ch["base_wl"][l]
            yield phy._dly_sel.storage.eq(1 << l)
            yield from strobe("wdly_dq_bitslip_rst")
            yield from strobe("wdly_dq_bitslip", need)
            for i in range(8):
                v = ch["v0"][8*l + i] + drift
                yield phy._dq_dly_sel.storage.eq(1 << i)
                yield from strobe("rdly_dq_bitslip_rst")
                yield from strobe("rdly_dq_bitslip", v//12 - 1)
                yield from strobe("rdly_dq_rst")
                yield from strobe("rdly_dq_inc", v - 12*(v//12 - 1))
        yield phy._dly_sel.storage.eq(0)
        yield phy._dq_dly_sel.storage.eq(0)
        results["calibrated"]  = yield from round_trip(2)
        results["calibrated2"] = yield from round_trip(3)

    run(dut, [gen()], "verilator", signals=signals, memories={}, name="s7hr_chain")
    assert not results["reset"], "the reset state must not read back right"
    assert results["calibrated"] and results["calibrated2"], results

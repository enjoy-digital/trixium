#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Simulation models of the YPCB's DDR3 PHY: LiteDRAM's K7DDRPHY on HR banks with its CK phase
shift (with_odelay=False, with_ck_phase_shift=True), in front of LiteDRAM's SDRAMPHYModel.

- ckphase_model_settings(): the PHY's latencies, for SDRAMPHYModel.
- CKPhasePHYModel: the PHY's CSRs and a fly-by channel (CKPhaseChannelModel), so liblitedram's
  CK phase leveling (SDRAM_PHY_CKPHASE_STEPS) runs in the SoC simulation.
- NoDMPHYModel: the board's DDR3 without DM pins (whole-word writes).
"""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

from litedram.common import *
from litedram.phy.dfi import Interface
from litedram.phy.s7ddrphy import S7DDRPHYCKPhase

# Latencies ----------------------------------------------------------------------------------------

def ckphase_latencies(sys_clk_freq, cl=None, cwl=None, cmd_latency=0, nphases=4):
    """(cl, cwl, rdphase, wrphase, read_latency, write_latency) of the 1:4 PHY: the S7DDRPHY's
    schedule (read data one memory-clock phase into the ISERDESE2 word)."""
    tck             = 2/(2*nphases*sys_clk_freq)
    cl              = get_default_cl( "DDR3", tck) if cl  is None else cl
    cwl             = get_default_cwl("DDR3", tck) if cwl is None else cwl
    cl_sys_latency  = get_sys_latency(nphases, cl)
    cwl_sys_latency = get_sys_latency(nphases, cwl)
    rdphase         = get_sys_phase(nphases, cl_sys_latency,   cl + cmd_latency)
    wrphase         = get_sys_phase(nphases, cwl_sys_latency, cwl + cmd_latency)
    phase_cycles, rdphase = divmod(rdphase + 1, nphases)
    cl_sys_latency -= phase_cycles
    return cl, cwl, rdphase, wrphase, cl_sys_latency + 6, cwl_sys_latency - 1


def ckphase_model_settings(sys_clk_freq, databits, cl=None, cwl=None):
    """PhySettings for LiteDRAM's SDRAMPHYModel with this PHY's latencies (the simulation)."""
    cl, cwl, rdphase, wrphase, read_latency, write_latency = ckphase_latencies(
        sys_clk_freq, cl, cwl)
    return PhySettings(
        phytype       = "SDRAMPHYModel",
        memtype       = "DDR3",
        databits      = databits,
        dfi_databits  = 2*databits,
        nphases       = 4,
        rdphase       = rdphase,
        wrphase       = wrphase,
        cl            = cl,
        cwl           = cwl,
        read_latency  = read_latency,
        write_latency = write_latency,
    )

# CKPhaseChannelModel ------------------------------------------------------------------------------

class CKPhaseChannelModel(Module):
    """A fly-by DDR3 channel as the calibration sees it (CKPhasePHYModel, the simulation).

    Inputs: the CK phase `ck` (signed fine steps, |ck| < 128), each lane's write bitslip `wl`,
    each bit's read bitslip `rb` and IDELAY tap `tap`. Outputs: `lane_bad` (the lane's writes go
    wrong: wrong latency for its side of the tDQSS crossing, or within +-zone of it) and
    `bit_bad` (the bit reads outside its eye, v0 + 0.215*ck in bitslip*12 + tap units).
    Unsigned arithmetic only (a first version with signed arithmetic read every bit bad at a
    negative CK phase in the SoC simulation; test_ckphase_channel.py checks the compiled model).
    """
    def __init__(self, cross, base_wl, v0, zone=3, half_eye=4):
        lanes, bits = len(cross), len(v0)
        self.ck       = Signal((16, True))
        self.wl       = [Signal(3) for _ in range(lanes)]
        self.rb       = [Signal(3) for _ in range(bits)]
        self.tap      = [Signal(5) for _ in range(bits)]
        self.lane_bad = Signal(lanes)
        self.bit_bad  = Signal(bits)

        # # #

        # The phase offset by 128: u = ck + 128.
        u = Signal(8)
        self.comb += u.eq(self.ck[:8] + 128)
        # The read eye's drift: 16.74 ps per step over 78 ps taps, 55/256 = 0.215; at ck = 0,
        # (128*55) >> 8 = 27.
        drift = Signal(16)
        self.comb += drift.eq((u*55)[8:])
        for l in range(lanes):
            need = Signal(3)
            cu   = cross[l] + 128
            self.comb += [
                need.eq(Mux(u >= cu, (base_wl[l] + 2) & 7, base_wl[l])),
                self.lane_bad[l].eq((self.wl[l] != need) | ((u + zone >= cu) & (u <= cu + zone))),
            ]
        for b in range(bits):
            # Good when |bitslip*12 + tap - (v0 + drift - 27)| <= half_eye.
            pos    = Signal(12)
            centre = Signal(16)
            self.comb += [
                pos.eq(self.rb[b]*12 + self.tap[b] + 27),
                centre.eq(drift + v0[b]),
                self.bit_bad[b].eq((pos + half_eye < centre) | (pos > centre + half_eye)),
            ]

# CKPhasePHYModel ----------------------------------------------------------------------------------

class CKPhasePHYModel(LiteXModule):
    """The K7DDRPHY's CSRs in CK phase mode and a fly-by channel in front of LiteDRAM's
    SDRAMPHYModel, for the simulation of liblitedram's CK phase leveling in the SoC.

    The channel: each lane has a tDQSS crossing `cross[l]` (CK fine steps) where its write latency
    steps from base_wl[l] to base_wl[l] + 2 and around which (+-zone) its writes fail; a lane
    written at the wrong latency or phase stores 0xA5 in its bytes. Each DQ bit reads right when
    bitslip*12 + tap is within +-half_eye of its eye, v0[b] + 0.215*ck (the DRAM drives DQ from
    the shifted CK, |ck| < 128); a bit read outside it comes back inverted. ideal: no channel
    effects (a controller the firmware does not calibrate in the simulation, e.g. the one it runs
    from). ckphase_steps: the CK phase steps over one tCK (the board's PHY: 112).
    """
    def __init__(self, model, cross=None, base_wl=None, v0=None, zone=3, half_eye=4, ideal=False,
                 ckphase_steps=112):
        self.model = model
        settings = model.settings
        # The PHY's capabilities: liblitedram builds its leveling against them (and against the
        # CSRs below), which a plain SDRAMPHYModel does not have.
        settings.read_leveling = True
        settings.delays        = 32
        settings.bitslips      = 8
        settings.ckphase_steps = ckphase_steps
        databits = settings.databits
        strobes  = databits//8
        nphases  = settings.nphases
        assert nphases == 4 and databits % 8 == 0
        self.settings = settings
        cross   = cross   or [0]*strobes
        base_wl = base_wl or [0]*strobes
        v0      = v0      or [0]*databits

        # Registers --------------------------------------------------------------------------------
        self._rst             = CSRStorage()
        self._dly_sel         = CSRStorage(strobes)
        self._dq_dly_sel      = CSRStorage(8)
        self._half_sys8x_taps = CSRStorage(5)
        self._wlevel_en       = CSRStorage()
        self._wlevel_strobe   = CSR()
        self._rdly_dq_rst         = CSR()
        self._rdly_dq_inc         = CSR()
        self._rdly_dq_bitslip_rst = CSR()
        self._rdly_dq_bitslip     = CSR()
        self._wdly_dq_bitslip_rst = CSR()
        self._wdly_dq_bitslip     = CSR()
        self._rdphase = CSRStorage(2, reset=settings.rdphase)
        self._wrphase = CSRStorage(2, reset=settings.wrphase)
        self.ckphase = ckphase = S7DDRPHYCKPhase()

        # DFI Interface ----------------------------------------------------------------------------
        self.dfi = dfi = Interface(len(model.dfi.p0.address), len(model.dfi.p0.bank),
            settings.nranks, 2*databits, nphases)

        # # #

        # The MMCM: locked, PSDONE 12 cycles after PSEN.
        ps_delay = Signal(12)
        self.sync += ps_delay.eq(Cat(ckphase.psen, ps_delay[:-1]))
        self.comb += [
            ckphase.locked.eq(1),
            ckphase.psdone.eq(ps_delay[-1]),
        ]
        ck = Signal((16, True))
        self.comb += ck.eq(ckphase._status.fields.value)

        # PHY state: write bitslip per lane, read bitslip and IDELAY tap per bit.
        wl  = [Signal(3) for _ in range(strobes)]
        rb  = [Signal(3) for _ in range(databits)]
        tap = [Signal(5) for _ in range(databits)]
        dq_sel = Signal(8)
        self.comb += If(self._dq_dly_sel.storage == 0,
            dq_sel.eq(0xff)
        ).Else(
            dq_sel.eq(self._dq_dly_sel.storage)
        )
        for l in range(strobes):
            self.sync += [
                If(self._rst.storage | (self._dly_sel.storage[l] & self._wdly_dq_bitslip_rst.re),
                    wl[l].eq(0)
                ).Elif(self._dly_sel.storage[l] & self._wdly_dq_bitslip.re,
                    wl[l].eq(wl[l] + 1)
                )
            ]
        for b in range(databits):
            sel = self._dly_sel.storage[b//8] & dq_sel[b%8]
            self.sync += [
                If(self._rst.storage | (sel & self._rdly_dq_bitslip_rst.re),
                    rb[b].eq(0)
                ).Elif(sel & self._rdly_dq_bitslip.re,
                    rb[b].eq(rb[b] + 1)
                ),
                If(self._rst.storage | (sel & self._rdly_dq_rst.re),
                    tap[b].eq(0)
                ).Elif(sel & self._rdly_dq_inc.re & (tap[b] != 31),
                    tap[b].eq(tap[b] + 1)
                ),
            ]

        # Channel: lanes that write wrong, bits that read wrong.
        lane_bad = Signal(strobes)
        bit_bad  = Signal(databits)
        if not ideal:
            self.channel = channel = CKPhaseChannelModel(cross, base_wl, v0, zone, half_eye)
            self.comb += [
                channel.ck.eq(ck),
                lane_bad.eq(channel.lane_bad),
                bit_bad.eq(channel.bit_bad),
            ]
            self.comb += [channel.wl[l].eq(wl[l]) for l in range(strobes)]
            self.comb += [channel.rb[b].eq(rb[b]) for b in range(databits)]
            self.comb += [channel.tap[b].eq(tap[b]) for b in range(databits)]

        # DFI pass-through with the channel's effects on the write and read data. The read data
        # valid on every phase, as the PHY drives it (SDRAMPHYModel drives phase 0's only, and
        # the DFII's phase injectors each capture on their own phase's valid).
        # A lane written wrong stores 0xa5 (not an XOR of the data: a fixed XOR and a read
        # inversion would cancel and pass the check).
        wsel  = Cat(*[Replicate(lane_bad[l], 8) for _ in range(2) for l in range(strobes)])
        wbad  = Cat(*[Constant(0xa5, 8) for _ in range(2*strobes)])
        rmask = Cat(bit_bad, bit_bad)
        rddata_valid = Signal()
        self.comb += rddata_valid.eq(reduce(or_, [ph.rddata_valid for ph in model.dfi.phases]))
        for p in range(nphases):
            ours, theirs = dfi.phases[p], model.dfi.phases[p]
            for name in ["address", "bank", "cas_n", "cs_n", "ras_n", "we_n", "cke", "odt",
                         "reset_n", "act_n", "wrdata_en", "rddata_en"]:
                if hasattr(ours, name) and hasattr(theirs, name):
                    self.comb += getattr(theirs, name).eq(getattr(ours, name))
            self.comb += [
                theirs.wrdata_mask.eq(0), # No DM pins: a write stores the whole word.
                theirs.wrdata.eq((ours.wrdata & ~wsel) | (wbad & wsel)),
                ours.rddata.eq(theirs.rddata ^ rmask),
                ours.rddata_valid.eq(rddata_valid),
            ]


class NoDMPHYModel(LiteXModule):
    """LiteDRAM's SDRAMPHYModel as the YPCB's DDR3 has it: no DM pins, so the controller's write
    masks never reach the memory and a partially-strobed write stores the whole word (the
    unstrobed bytes with whatever the data bus carries). The ports that write partial words
    need a read-modify-write (LiteDRAMAXI2Native's with_read_modify_write) on such a memory."""
    def __init__(self, model):
        self.model    = model
        self.settings = model.settings
        self.dfi = dfi = Interface(len(model.dfi.p0.address), len(model.dfi.p0.bank),
            model.settings.nranks, len(model.dfi.p0.wrdata), model.settings.nphases)

        # # #

        for ours, theirs in zip(dfi.phases, model.dfi.phases):
            for name in ["address", "bank", "cas_n", "cs_n", "ras_n", "we_n", "cke", "odt",
                         "reset_n", "act_n", "wrdata_en", "wrdata", "rddata_en"]:
                if hasattr(ours, name) and hasattr(theirs, name):
                    self.comb += getattr(theirs, name).eq(getattr(ours, name))
            self.comb += [
                theirs.wrdata_mask.eq(0),
                ours.rddata.eq(theirs.rddata),
                ours.rddata_valid.eq(theirs.rddata_valid),
            ]

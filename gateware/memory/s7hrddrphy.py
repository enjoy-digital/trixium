#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Series-7 DDR3 PHY for HR banks, with write leveling by CK phase (S7HRDDRPHY).

LiteDRAM's A7DDRPHY structure (1:4, OSERDESE2/ISERDESE2, read leveling with IDELAYE2 + bitslip,
per-lane write latency bitslip) for DDR3 on high-range banks, where there is no ODELAYE2 and
therefore no per-lane write leveling. A fly-by command/clock bus (a DIMM-like layout, e.g. the
nine x8 chips of an Inspur YPCB-00338 channel) puts each chip's CK at a different delay from
its DQS: with a single static DQS phase only part of the lanes meet tDQSS. The technique
(openTPU, FeSens, "WL7DDRPHY", docs/litedram.md; reimplemented here):

- CK and the command/address pins run on a pair of clocks with MMCM fine phase shift; DQ, DQS
  and the read capture run on static clocks of the same MMCM, so DQ keeps its DQS at 90 deg and
  the read capture does not move. Moving CK against DQ/DQS is the relative motion write leveling
  needs, and the MMCM's fine-PS outputs all share one phase (the FINE_PS_FRAC DRC refuses
  fine-PS outputs at different sub-VCO phases, so CK is the one that moves).
- One MMCM per PHY, cascaded from the controller clock (`sys`) with its feedback through a
  BUFG: the static outputs keep the phase of `sys` and Vivado times sys <-> static clocks as
  synchronous. Every OSERDESE2's CLK/CLKDIV pair comes from that one VCO.
- The commands cross from `sys` to the shifted CLKDIV through registers clocked on the falling
  edge of `sys` (FDRE IS_C_INVERTED on the global clock, not a LUT inverter): with the CK phase
  kept within +-1/2 tCK of `sys` (+-0.94 ns at DDR3-1066), the crossing keeps ~2.8 ns of setup
  and hold. add_constraints() adds the phase range as clock uncertainty on that crossing.
- Each lane keeps its own write latency (the A7DDRPHY's write bitslip); the read side adds
  per-bit selection (`dq_dly_sel`, liblitedram's SDRAM_DELAY_PER_DQ CSR name: the IDELAY taps and
  the read bitslip of single DQ bits) since the ISERDESE2 of one lane do not all frame a word
  alike.

The calibration (firmware/ddr_s7hr.c) scans the CK phase over one tCK, runs the per-lane write
latency calibration, the per-bit read leveling and a traffic test (LiteDRAM BIST) at each step,
and settles in the middle of the longest run that every lane passes. A single-burst DFI check
is not enough to pick the phase (a lane can pass it and fail under traffic).

CSRs (on top of the A7DDRPHY's):
- ckphase_inc / ckphase_dec: one fine phase step of CK (1/56 of the VCO period, 16.7 ps at
  DDR3-1066) later / earlier; the MMCM takes ~12 cycles (busy).
- ckphase_status: value (signed fine steps from the reset phase), busy, locked.
- dq_dly_sel: per-bit mask of the read-side delay/bitslip strobes within the lanes `dly_sel`
  selects (0: every bit, the A7DDRPHY behaviour).
"""

import math
from functools import reduce
from operator import or_

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *

from litex.soc.interconnect.csr import *

from litedram.common import *
from litedram.phy.dfi import Interface

# Helpers ------------------------------------------------------------------------------------------

def _negedge(module, signal, clk):
    """`signal` registered on the falling edge of `clk` (one FDRE per bit, IS_C_INVERTED)."""
    q = Signal.like(signal)
    for i in range(len(signal)):
        module.specials += Instance("FDRE",
            p_INIT         = 0,
            p_IS_C_INVERTED = 1,
            i_C  = clk,
            i_CE = 1,
            i_R  = 0,
            i_D  = signal[i],
            o_Q  = q[i],
        )
    return q

# Latencies ----------------------------------------------------------------------------------------

def s7hr_latencies(sys_clk_freq, cl=None, cwl=None, cmd_latency=0, nphases=4):
    """(cl, cwl, rdphase, wrphase, read_latency, write_latency) of the 1:4 PHY: the A7DDRPHY's
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


def s7hr_model_settings(sys_clk_freq, databits, cl=None, cwl=None):
    """PhySettings for LiteDRAM's SDRAMPHYModel with this PHY's latencies (the simulation)."""
    cl, cwl, rdphase, wrphase, read_latency, write_latency = s7hr_latencies(sys_clk_freq, cl, cwl)
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

# CK Phase Shift -----------------------------------------------------------------------------------

class S7HRCKPhase(LiteXModule):
    """The MMCM fine phase shift of CK, one step per CSR write (PSCLK = sys).

    CSRs: inc / dec (one step later / earlier), status (value: signed steps from the MMCM's reset
    phase, busy: a step in progress, the MMCM takes ~12 cycles; locked). A step requested while
    busy is dropped: wait for busy low. The value follows the MMCM: back to 0 when it unlocks.
    """
    def __init__(self):
        self.psen     = Signal()
        self.psincdec = Signal()
        self.psdone   = Signal()
        self.locked   = Signal()

        self._inc    = CSR()
        self._dec    = CSR()
        self._status = CSRStatus(fields=[
            CSRField("value",  size=16, description="CK phase, signed fine steps from reset."),
            CSRField("busy",   size=1,  description="A phase step is in progress."),
            CSRField("locked", size=1,  description="The PHY's MMCM is locked."),
        ])

        # # #

        busy  = Signal()
        value = Signal((16, True))
        self.sync += [
            self.psen.eq(0),
            If(~busy & (self._inc.re | self._dec.re),
                self.psen.eq(1),
                self.psincdec.eq(self._inc.re),
                busy.eq(1),
            ),
            If(self.psdone,
                busy.eq(0),
                If(self.psincdec,
                    value.eq(value + 1)
                ).Else(
                    value.eq(value - 1)
                )
            ),
            If(~self.locked,
                busy.eq(0),
                value.eq(0),
            ),
        ]
        self.comb += [
            self._status.fields.value.eq(value),
            self._status.fields.busy.eq(busy),
            self._status.fields.locked.eq(self.locked),
        ]

# S7HRDDRPHY ---------------------------------------------------------------------------------------

class S7HRDDRPHY(LiteXModule):
    """DDR3 1:4 PHY for Series-7 HR banks, write leveling by CK phase (see the module doc).

    pads:      DDR3 pads (one group; PHYPadsReducer to leave out an ECC lane).
    name:      prefix of the PHY's own clock domains (unique per PHY of a SoC).
    mmcm_loc:  optional MMCME2_ADV site (the CMT of the channel's banks).
    """
    def __init__(self, pads, sys_clk_freq, iodelay_clk_freq=200e6, cl=None, cwl=None,
                 cmd_latency=0, name="ddrphy", mmcm_loc=None):
        memtype     = "DDR3"
        nphases     = 4
        phytype     = self.__class__.__name__
        tck         = 2/(2*nphases*sys_clk_freq)
        addressbits = len(pads.a)
        bankbits    = len(pads.ba)
        nranks      = 1 if not hasattr(pads, "cs_n") else len(pads.cs_n)
        databits    = len(pads.dq)
        strobes     = len(pads.dqs_p)
        assert databits == 8*strobes, "x8 lanes (8 DQ per DQS)"
        assert strobes <= 16
        self.name     = name
        self.mmcm_loc = mmcm_loc

        # Parameters -------------------------------------------------------------------------------
        iodelay_tap_average = {200e6: 78e-12, 300e6: 52e-12, 400e6: 39e-12}
        half_sys8x_taps = math.floor(tck/(4*iodelay_tap_average[iodelay_clk_freq]))

        cl, cwl, rdphase, wrphase, read_latency, write_latency = s7hr_latencies(
            sys_clk_freq, cl, cwl, cmd_latency, nphases)
        cwl_sys_latency = write_latency + 1

        # Registers --------------------------------------------------------------------------------
        self._rst             = CSRStorage()
        self._dly_sel         = CSRStorage(strobes)
        self._dq_dly_sel      = CSRStorage(8, description="Read-side per-bit mask (0: all bits).")
        self._half_sys8x_taps = CSRStorage(5, reset=half_sys8x_taps)

        self._wlevel_en     = CSRStorage()
        self._wlevel_strobe = CSR()

        self._rdly_dq_rst         = CSR()
        self._rdly_dq_inc         = CSR()
        self._rdly_dq_bitslip_rst = CSR()
        self._rdly_dq_bitslip     = CSR()

        self._wdly_dq_bitslip_rst = CSR()
        self._wdly_dq_bitslip     = CSR()

        self._rdphase = CSRStorage(int(math.log2(nphases)), reset=rdphase)
        self._wrphase = CSRStorage(int(math.log2(nphases)), reset=wrphase)

        self.settings = PhySettings(
            phytype                   = phytype,
            memtype                   = memtype,
            databits                  = databits,
            strobes                   = strobes,
            dfi_databits              = 2*databits,
            nranks                    = nranks,
            nphases                   = nphases,
            rdphase                   = self._rdphase.storage,
            wrphase                   = self._wrphase.storage,
            cl                        = cl,
            cwl                       = cwl,
            read_latency              = read_latency,
            write_latency             = write_latency,
            cmd_latency               = cmd_latency,
            write_leveling            = False,
            write_dq_dqs_training     = False,
            # The per-lane write latency is calibrated (firmware/ddr_s7hr.c), but not by
            # liblitedram's sequence: its write latency calibration does not build without
            # write leveling (the write bitslip accessors are behind it).
            write_latency_calibration = False,
            read_leveling             = True,
            delays                    = 32,
            bitslips                  = 8,
            with_dm                   = hasattr(pads, "dm"),
        )

        # DFI Interface ----------------------------------------------------------------------------
        self.dfi = dfi = Interface(addressbits, bankbits, nranks, 2*databits, nphases)

        # # #

        # Clocking ---------------------------------------------------------------------------------
        # One MMCM per PHY, cascaded from sys (VCO = 2*nphases*sys = the DDR rate):
        #   c       sys     fine PS   CLKDIV of CK and the commands
        #   c4x     4x sys  fine PS   CLK of CK and the commands
        #   w       sys     0         CLKDIV of DQ/DQS/DM and of the read ISERDESE2
        #   w4x     4x sys  0         CLK of DQ/DM and of the read ISERDESE2
        #   w4x_dqs 4x sys  90 deg    CLK of DQS
        self.cd_c       = ClockDomain(f"{name}_c")
        self.cd_c4x     = ClockDomain(f"{name}_c4x",     reset_less=True)
        self.cd_w       = ClockDomain(f"{name}_w")
        self.cd_w4x     = ClockDomain(f"{name}_w4x",     reset_less=True)
        self.cd_w4x_dqs = ClockDomain(f"{name}_w4x_dqs", reset_less=True)
        vco_mult  = 2*nphases
        vco_freq  = sys_clk_freq*vco_mult
        assert 600e6 <= vco_freq <= 1440e6, f"MMCM VCO {vco_freq/1e6:.1f} MHz out of range"
        self.ps_step_ps = 1e12/vco_freq/56
        self.ps_steps_per_tck = 56*vco_mult//nphases # One tCK = 2 VCO periods = 112 steps (1:4).
        fb, fb_buf = Signal(), Signal()
        outs       = [Signal() for _ in range(5)]
        self.mmcm_locked = locked = Signal()
        self.ckphase = ckphase = S7HRCKPhase()
        self.comb += ckphase.locked.eq(locked)
        self.mmcm_rst = mmcm_rst = Signal()
        self.comb += mmcm_rst.eq(ResetSignal("sys"))
        self.specials += Instance("MMCME2_ADV", name=f"{name}_mmcm",
            p_BANDWIDTH          = "OPTIMIZED",
            p_CLKIN1_PERIOD      = 1e9/sys_clk_freq,
            p_REF_JITTER1        = 0.01,
            p_DIVCLK_DIVIDE      = 1,
            p_CLKFBOUT_MULT_F    = float(vco_mult),
            p_CLKFBOUT_PHASE     = 0.0,
            p_CLKOUT0_DIVIDE_F   = float(vco_mult),
            p_CLKOUT0_PHASE      = 0.0,
            p_CLKOUT0_USE_FINE_PS = "TRUE",
            p_CLKOUT1_DIVIDE     = 2,
            p_CLKOUT1_PHASE      = 0.0,
            p_CLKOUT1_USE_FINE_PS = "TRUE",
            p_CLKOUT2_DIVIDE     = vco_mult,
            p_CLKOUT2_PHASE      = 0.0,
            p_CLKOUT3_DIVIDE     = 2,
            p_CLKOUT3_PHASE      = 0.0,
            p_CLKOUT4_DIVIDE     = 2,
            p_CLKOUT4_PHASE      = 90.0,
            i_CLKIN1   = ClockSignal("sys"),
            i_CLKIN2   = 0,
            i_CLKINSEL = 1,
            i_CLKFBIN  = fb_buf,
            o_CLKFBOUT = fb,
            i_RST      = mmcm_rst,
            i_PWRDWN   = 0,
            o_LOCKED   = locked,
            i_PSCLK    = ClockSignal("sys"),
            i_PSEN     = ckphase.psen,
            i_PSINCDEC = ckphase.psincdec,
            o_PSDONE   = ckphase.psdone,
            i_DCLK = 0, i_DEN = 0, i_DWE = 0, i_DADDR = 0, i_DI = 0,
            o_CLKOUT0 = outs[0],
            o_CLKOUT1 = outs[1],
            o_CLKOUT2 = outs[2],
            o_CLKOUT3 = outs[3],
            o_CLKOUT4 = outs[4],
        )
        self.specials += Instance("BUFG", i_I=fb, o_O=fb_buf)
        for o, cd in zip(outs, [self.cd_c, self.cd_c4x, self.cd_w, self.cd_w4x, self.cd_w4x_dqs]):
            self.specials += Instance("BUFG", i_I=o, o_O=cd.clk)

        # Serializer resets: held while the MMCM is unlocked, released synchronously to each
        # CLKDIV; one register per lane on the write/read side.
        serdes_rst = Signal()
        self.comb += serdes_rst.eq(ResetSignal("sys") | ~locked | self._rst.storage)
        self.specials += [
            AsyncResetSynchronizer(self.cd_c, serdes_rst),
            AsyncResetSynchronizer(self.cd_w, serdes_rst),
        ]
        lane_rst = [Signal(reset_less=True) for _ in range(strobes)]
        c_rst    = Signal(reset_less=True)
        sync_w   = getattr(self.sync, f"{name}_w")
        sync_c   = getattr(self.sync, f"{name}_c")
        sync_w  += [r.eq(ResetSignal(f"{name}_w")) for r in lane_rst]
        sync_c  += c_rst.eq(ResetSignal(f"{name}_c"))

        # Per-bit selection (read side) ------------------------------------------------------------
        dq_sel = Signal(8)
        self.comb += If(self._dq_dly_sel.storage == 0,
            dq_sel.eq(0xff)
        ).Else(
            dq_sel.eq(self._dq_dly_sel.storage)
        )
        def bit_sel(i):
            return self._dly_sel.storage[i//8] & dq_sel[i%8]

        rdly_dq_rst         = self._rdly_dq_rst.re
        rdly_dq_inc         = self._rdly_dq_inc.re
        rdly_dq_bitslip_rst = self._rdly_dq_bitslip_rst.re
        rdly_dq_bitslip     = self._rdly_dq_bitslip.re
        wdly_dq_bitslip_rst = self._wdly_dq_bitslip_rst.re
        wdly_dq_bitslip     = self._wdly_dq_bitslip.re
        wlevel_strobe       = self._wlevel_strobe.re

        # Clock ------------------------------------------------------------------------------------
        for i in range(len(pads.clk_p)):
            sd_clk_se = Signal()
            self.specials += Instance("OSERDESE2",
                p_SERDES_MODE    = "MASTER",
                p_DATA_WIDTH     = 2*nphases,
                p_TRISTATE_WIDTH = 1,
                p_DATA_RATE_OQ   = "DDR",
                p_DATA_RATE_TQ   = "BUF",
                i_RST    = c_rst,
                i_CLK    = ClockSignal(f"{name}_c4x"),
                i_CLKDIV = ClockSignal(f"{name}_c"),
                **{f"i_D{n+1}": (0b10101010 >> n) & 0b1 for n in range(8)},
                i_OCE    = 1,
                o_OQ     = sd_clk_se,
            )
            self.specials += Instance("OBUFDS",
                i_I  = sd_clk_se,
                o_O  = pads.clk_p[i],
                o_OB = pads.clk_n[i],
            )

        # Commands ---------------------------------------------------------------------------------
        # The DFI phases cross into the shifted CLKDIV through falling-edge registers of sys.
        commands = {
            "reset_n" : ("reset_n", False),
            "cs_n"    : ("cs_n",    False),
            "a"       : ("address", True),
            "ba"      : ("bank",    True),
            "ras_n"   : ("ras_n",   True),
            "cas_n"   : ("cas_n",   True),
            "we_n"    : ("we_n",    True),
            "cke"     : ("cke",     False),
            "odt"     : ("odt",     False),
        }
        for pad_name, (dfi_name, required) in commands.items():
            pad = getattr(pads, pad_name, None)
            if pad is None:
                if required:
                    raise ValueError(f"DRAM pad {pad_name} required but not found in pads.")
                continue
            phases = [_negedge(self, getattr(dfi.phases[p], dfi_name), ClockSignal("sys"))
                for p in range(nphases)]
            for i in range(len(pad)):
                self.specials += Instance("OSERDESE2",
                    p_SERDES_MODE    = "MASTER",
                    p_DATA_WIDTH     = 2*nphases,
                    p_TRISTATE_WIDTH = 1,
                    p_DATA_RATE_OQ   = "DDR",
                    p_DATA_RATE_TQ   = "BUF",
                    i_RST    = c_rst,
                    i_CLK    = ClockSignal(f"{name}_c4x"),
                    i_CLKDIV = ClockSignal(f"{name}_c"),
                    **{f"i_D{n+1}": phases[n//2][i] for n in range(8)},
                    i_OCE    = 1,
                    o_OQ     = pad[i],
                )

        # DQS --------------------------------------------------------------------------------------
        dqs_oe        = Signal()
        dqs_preamble  = Signal()
        dqs_postamble = Signal()
        dqs_oe_delay  = TappedDelayLine(ntaps=2)
        dqs_pattern   = DQSPattern(
            wlevel_en     = self._wlevel_en.storage,
            wlevel_strobe = wlevel_strobe,
            register      = True)
        self.submodules += dqs_oe_delay, dqs_pattern
        self.comb += dqs_oe_delay.input.eq(dqs_preamble | dqs_oe | dqs_postamble)
        for i in range(strobes):
            dqs_o = Signal()
            dqs_t = Signal()
            dqs_bitslip = BitSlip(8,
                i      = dqs_pattern.o,
                rst    = (self._dly_sel.storage[i] & wdly_dq_bitslip_rst) | self._rst.storage,
                slp    = self._dly_sel.storage[i] & wdly_dq_bitslip,
                cycles = 1)
            self.submodules += dqs_bitslip
            self.specials += Instance("OSERDESE2",
                p_SERDES_MODE    = "MASTER",
                p_DATA_WIDTH     = 2*nphases,
                p_TRISTATE_WIDTH = 1,
                p_DATA_RATE_OQ   = "DDR",
                p_DATA_RATE_TQ   = "BUF",
                i_RST    = lane_rst[i],
                i_CLK    = ClockSignal(f"{name}_w4x_dqs"),
                i_CLKDIV = ClockSignal(f"{name}_w"),
                **{f"i_D{n+1}": dqs_bitslip.o[n] for n in range(8)},
                i_OCE    = 1,
                o_OQ     = dqs_o,
                i_TCE    = 1,
                i_T1     = ~dqs_oe_delay.output,
                o_TQ     = dqs_t,
            )
            self.specials += Instance("IOBUFDS",
                i_T    = dqs_t,
                i_I    = dqs_o,
                io_IO  = pads.dqs_p[i],
                io_IOB = pads.dqs_n[i],
            )

        # DM ---------------------------------------------------------------------------------------
        if hasattr(pads, "dm"):
            for i in range(strobes):
                dm_i = Cat(*[dfi.phases[n//2].wrdata_mask[n%2*databits//8+i] for n in range(8)])
                dm_bitslip = BitSlip(8,
                    i      = dm_i,
                    rst    = (self._dly_sel.storage[i] & wdly_dq_bitslip_rst) | self._rst.storage,
                    slp    = self._dly_sel.storage[i] & wdly_dq_bitslip,
                    cycles = 1)
                self.submodules += dm_bitslip
                self.specials += Instance("OSERDESE2",
                    p_SERDES_MODE    = "MASTER",
                    p_DATA_WIDTH     = 2*nphases,
                    p_TRISTATE_WIDTH = 1,
                    p_DATA_RATE_OQ   = "DDR",
                    p_DATA_RATE_TQ   = "BUF",
                    i_RST    = lane_rst[i],
                    i_CLK    = ClockSignal(f"{name}_w4x"),
                    i_CLKDIV = ClockSignal(f"{name}_w"),
                    **{f"i_D{n+1}": dm_bitslip.o[n] for n in range(8)},
                    i_OCE    = 1,
                    o_OQ     = pads.dm[i],
                )

        # DQ ---------------------------------------------------------------------------------------
        dq_oe       = Signal()
        dq_oe_delay = TappedDelayLine(ntaps=2)
        self.submodules += dq_oe_delay
        self.comb += dq_oe_delay.input.eq(dqs_preamble | dq_oe | dqs_postamble)
        for i in range(databits):
            dq_o         = Signal()
            dq_i         = Signal()
            dq_i_delayed = Signal()
            dq_t         = Signal()
            dq_o_bitslip = BitSlip(8,
                i      = Cat(*[dfi.phases[n//2].wrdata[n%2*databits+i] for n in range(8)]),
                rst    = (self._dly_sel.storage[i//8] & wdly_dq_bitslip_rst) | self._rst.storage,
                slp    = self._dly_sel.storage[i//8] & wdly_dq_bitslip,
                cycles = 1)
            self.submodules += dq_o_bitslip
            self.specials += Instance("OSERDESE2",
                p_SERDES_MODE    = "MASTER",
                p_DATA_WIDTH     = 2*nphases,
                p_TRISTATE_WIDTH = 1,
                p_DATA_RATE_OQ   = "DDR",
                p_DATA_RATE_TQ   = "BUF",
                i_RST    = lane_rst[i//8],
                i_CLK    = ClockSignal(f"{name}_w4x"),
                i_CLKDIV = ClockSignal(f"{name}_w"),
                **{f"i_D{n+1}": dq_o_bitslip.o[n] for n in range(8)},
                i_TCE    = 1,
                i_T1     = ~dq_oe_delay.output,
                o_TQ     = dq_t,
                i_OCE    = 1,
                o_OQ     = dq_o,
            )
            dq_i_bitslip = BitSlip(8,
                rst    = (bit_sel(i) & rdly_dq_bitslip_rst) | self._rst.storage,
                slp    = bit_sel(i) & rdly_dq_bitslip,
                cycles = 1)
            self.submodules += dq_i_bitslip
            self.specials += Instance("ISERDESE2",
                p_SERDES_MODE    = "MASTER",
                p_INTERFACE_TYPE = "NETWORKING",
                p_DATA_WIDTH     = 2*nphases,
                p_DATA_RATE      = "DDR",
                p_NUM_CE         = 1,
                p_IOBDELAY       = "IFD",
                i_RST     = lane_rst[i//8],
                i_CLK     = ClockSignal(f"{name}_w4x"),
                i_CLKB    = ~ClockSignal(f"{name}_w4x"),
                i_CLKDIV  = ClockSignal(f"{name}_w"),
                i_BITSLIP = 0,
                i_CE1     = 1,
                i_DDLY    = dq_i_delayed,
                **{f"o_Q{n+1}": dq_i_bitslip.i[8-1-n] for n in range(8)},
            )
            for n in range(8):
                self.comb += dfi.phases[n//2].rddata[n%2*databits+i].eq(dq_i_bitslip.o[n])
            self.specials += Instance("IDELAYE2",
                p_SIGNAL_PATTERN        = "DATA",
                p_DELAY_SRC             = "IDATAIN",
                p_CINVCTRL_SEL          = "FALSE",
                p_HIGH_PERFORMANCE_MODE = "TRUE",
                p_REFCLK_FREQUENCY      = iodelay_clk_freq/1e6,
                p_PIPE_SEL              = "FALSE",
                p_IDELAY_TYPE           = "VARIABLE",
                p_IDELAY_VALUE          = 0,
                i_C        = ClockSignal("sys"),
                i_LD       = (bit_sel(i) & rdly_dq_rst) | self._rst.storage,
                i_LDPIPEEN = 0,
                i_CE       = bit_sel(i) & rdly_dq_inc,
                i_INC      = 1,
                i_IDATAIN  = dq_i,
                o_DATAOUT  = dq_i_delayed,
            )
            self.specials += Instance("IOBUF",
                i_I   = dq_o,
                o_O   = dq_i,
                i_T   = dq_t,
                io_IO = pads.dq[i],
            )

        # Read Control Path ------------------------------------------------------------------------
        rddata_en = TappedDelayLine(
            signal = reduce(or_, [dfi.phases[i].rddata_en for i in range(nphases)]),
            ntaps  = self.settings.read_latency)
        self.submodules += rddata_en
        self.comb += [phase.rddata_valid.eq(rddata_en.output | self._wlevel_en.storage)
            for phase in dfi.phases]

        # Write Control Path -----------------------------------------------------------------------
        wrtap = cwl_sys_latency - 1
        wrdata_en = TappedDelayLine(
            signal = reduce(or_, [dfi.phases[i].wrdata_en for i in range(nphases)]),
            ntaps  = wrtap + 2)
        self.submodules += wrdata_en
        self.comb += dq_oe.eq(wrdata_en.taps[wrtap])
        self.comb += If(self._wlevel_en.storage, dqs_oe.eq(1)).Else(dqs_oe.eq(dq_oe))
        self.comb += dqs_preamble.eq( wrdata_en.taps[wrtap - 1] & ~wrdata_en.taps[wrtap + 0])
        self.comb += dqs_postamble.eq(wrdata_en.taps[wrtap + 1] & ~wrdata_en.taps[wrtap + 0])

    def add_constraints(self, platform):
        """Timing of the phase-shifted crossing and the MMCM placement.

        The commands leave sys on its falling edge and enter the shifted CLKDIV: the CK phase
        range (+-1/2 tCK) goes in as clock uncertainty on that crossing, setup and hold. The
        MMCM reset is asynchronous."""
        sys_clk = f"[get_clocks -of_objects [get_pins -hierarchical -filter {{{{NAME =~ *{self.name}_mmcm/CLKIN1}}}}]]"
        c_clk   = f"[get_clocks -of_objects [get_pins -hierarchical -filter {{{{NAME =~ *{self.name}_mmcm/CLKOUT0}}}}]]"
        for kind in ("setup", "hold"):
            platform.add_platform_command(f"set_clock_uncertainty -{kind} 1.0 -from {sys_clk} -to {c_clk}")
        platform.add_platform_command(
            f"set_false_path -to [get_pins -hierarchical -filter {{{{NAME =~ *{self.name}_mmcm/RST}}}}]")
        if self.mmcm_loc is not None:
            platform.add_platform_command(
                f"set_property LOC {self.mmcm_loc} [get_cells -hierarchical -filter {{{{NAME =~ *{self.name}_mmcm}}}}]")

# S7HRChannelModel ---------------------------------------------------------------------------------

class S7HRChannelModel(Module):
    """A fly-by DDR3 channel as the calibration sees it (S7HRPHYModel, the simulation).

    Inputs: the CK phase `ck` (signed fine steps, |ck| < 128), each lane's write bitslip `wl`,
    each bit's read bitslip `rb` and IDELAY tap `tap`. Outputs: `lane_bad` (the lane's writes go
    wrong: wrong latency for its side of the tDQSS crossing, or within +-zone of it) and
    `bit_bad` (the bit reads outside its eye, v0 + 0.215*ck in bitslip*12 + tap units).
    Unsigned arithmetic only (a first version with signed arithmetic read every bit bad at a
    negative CK phase in the SoC simulation; test_s7hrmodel.py checks the compiled model).
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

# S7HRPHYModel -------------------------------------------------------------------------------------

class S7HRPHYModel(LiteXModule):
    """The S7HRDDRPHY's CSRs and a fly-by channel in front of LiteDRAM's SDRAMPHYModel, for the
    simulation of the calibration (firmware/ddr_s7hr.c) in the SoC.

    The channel (the host test's fake, test/common/s7hr/s7hr_fake.c, in gateware): each lane has
    a tDQSS crossing `cross[l]` (CK fine steps) where its write latency steps from base_wl[l] to
    base_wl[l] + 2 and around which (+-zone) its writes fail; a lane written at the wrong
    latency or phase stores 0xA5 in its bytes. Each DQ bit reads right when
    bitslip*12 + tap is within +-half_eye of its eye, v0[b] + 0.215*ck (the DRAM drives DQ from
    the shifted CK, |ck| < 128); a bit read outside it comes back inverted. ideal: no channel effects (a
    controller the firmware does not calibrate in the simulation, e.g. the one it runs from).
    """
    def __init__(self, model, cross=None, base_wl=None, v0=None, zone=3, half_eye=4, ideal=False):
        self.model = model
        settings = model.settings
        # The PHY's capabilities: liblitedram builds its leveling against them (and against the
        # CSRs below), which a plain SDRAMPHYModel does not have.
        settings.read_leveling = True
        settings.delays        = 32
        settings.bitslips      = 8
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
        self.ckphase = ckphase = S7HRCKPhase()

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
            self.channel = channel = S7HRChannelModel(cross, base_wl, v0, zone, half_eye)
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

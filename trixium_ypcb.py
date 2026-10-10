#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium on the Inspur YPCB-00338-1P1 (XC7K480T-2, two 72-bit DDR3 channels, PCIe Gen2 x8).

The TriXium core streams its weights from the two DDR3 channels (64 data bits each, the ECC
lane unused): engine channel c reads from controller c // 2 through its own crossbar port, at
two windows per controller. The DDR3 sits on HR banks (no ODELAY, a fly-by command bus over
nine x8 chips): the PHY is gateware/memory/s7hrddrphy.py (write leveling by CK phase) and the
calibration runs on the SoC's CPU (firmware/ddr_s7hr.c: the BIOS calibrates controller 0, the
firmware controller 1). The defaults are the shipping recipe: the Qwen3 path's units (engine,
sequencer, vector unit; the Ternary-Bonsai 1.7B and 8B), DDR3-1066 in a 133.33 MHz ddr domain,
sys 150 MHz, the FK33's CPU configuration. Design notes: doc/boards.md.

From the repository root:
  ./trixium_ypcb.py --build                          # -> build/ypcb/ + build/ypcb.verdict
  ./trixium_ypcb.py --load                           # JTAG (openFPGALoader, Digilent HS2 cable)
  trixium deploy --board ypcb --model 1p7b           # the lab board: software/lab/deploy.py

Vivado 2024.1 (--vivado-settings). A K480T route needs ~12 GB of RAM.
"""

import os
import sys

from migen import *

from litex.gen import *

from litex.build.generic_platform import Subsignal, Pins, IOStandard

from litex_boards.platforms import ypcb_00338_1p1

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *
from litex.soc.interconnect.axi import AXIInterface

from litedram.common import PHYPadsReducer
from litedram.modules import MT41K256M8, _SpeedgradeTimings
from litedram.core.controller import ControllerSettings

from litepcie.phy.s7pciephy import S7PCIEPHY
from litepcie.software import generate_litepcie_software

from model.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from gateware.memory.s7hrddrphy import S7HRDDRPHY
from gateware.common import add_cluster_constraints, add_zbb_constant
from gateware.common import set_main_ram_executable, CPUBusMonitor
from gateware.common import VIVADO_PATH, set_vivado_env, write_verdict

# IOs ----------------------------------------------------------------------------------------------

# litex-boards' pcie_x8 has lane 6's RX N on E3; the GTX pair is R4/R3 (MGTXRXN1_115).
_pcie_x8_io = [
    ("pcie_x8", 1,
        Subsignal("clk_p", Pins("J8")),
        Subsignal("clk_n", Pins("J7")),
        Subsignal("rst_n", Pins("Y26"), IOStandard("LVCMOS18")),
        Subsignal("tx_p",  Pins("F2 H2 K2 M2 N4 P2 T2 U4")),
        Subsignal("tx_n",  Pins("F1 H1 K1 M1 N3 P1 T1 U3")),
        Subsignal("rx_p",  Pins("H6 J4 K6 L4 M6 P6 R4 T6")),
        Subsignal("rx_n",  Pins("H5 J3 K5 L3 M5 P5 R3 T5")),
    ),
]

# DDR3 ---------------------------------------------------------------------------------------------

# DDR3-1066 bin (CL 7 / CWL 6, the timings the board's MIG project used); LiteDRAM's defaults at
# 1.875 ns would be CL 10 / CWL 7.
DDR3_CL  = 7
DDR3_CWL = 6

class YPCBDDR3(MT41K256M8):
    """The card's MT41K256M8 (2 Gb x8, DDR3-1600 parts run at DDR3-1066 by the HR banks), with
    the 2 Gb tRFC of 160 ns (LiteDRAM's class has 128 tCK, 240 ns at DDR3-1066)."""
    speedgrade_timings = {
        "1600": _SpeedgradeTimings(tRP=13.75, tRCD=13.75, tWR=15, tRFC=(None, 160), tFAW=(None, 40), tRAS=35),
    }
    speedgrade_timings["default"] = speedgrade_timings["1600"]


def ddr_controller_settings(read_time=256, write_time=128, refresh_postponing=2):
    """The controller settings for weight streaming. LiteDRAM's defaults (32/16, postponing 1)
    switch direction every few dozen cycles and refresh one row at a time; longer read/write
    times and postponed refreshes keep the streams on their rows (trixium_sim.py --board ypcb
    --bench measures them)."""
    return lambda: ControllerSettings(
        read_time          = read_time,
        write_time         = write_time,
        refresh_postponing = refresh_postponing)

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    """Clocks: `sys` (CPU, TriXium core), `ddr` (the two LiteDRAM controllers, DDR3 rate =
    8 x ddr) and the 200 MHz IDELAY reference, from one MMCM on the 50 MHz oscillator (the
    board's reset pin is not wired). Each DDR3 PHY makes its own interface clocks from `ddr`
    (S7HRDDRPHY: one MMCM per channel). The memory ports cross sys <-> ddr at the AXI level
    (memory/ddr.py), the controllers' CSRs behind a wishbone crossing.
    """
    def __init__(self, platform, sys_clk_freq, ddr_clk_freq):
        assert ddr_clk_freq != sys_clk_freq
        self.rst       = Signal()
        self.cd_sys    = ClockDomain()
        self.cd_ddr    = ClockDomain()
        self.cd_idelay = ClockDomain()

        # # #

        self.pll = pll = S7MMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        clk50 = platform.request("clk50")
        platform.add_period_constraint(clk50, 1e9/50e6) # Not the platform's default clock (clk200).
        pll.register_clkin(clk50, 50e6)
        pll.create_clkout(self.cd_sys,    sys_clk_freq)
        pll.create_clkout(self.cd_ddr,    ddr_clk_freq)
        pll.create_clkout(self.cd_idelay, 200e6)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.
        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_ddr.clk)

        # IDELAYCTRL (replicated by Vivado in every bank with IDELAYs).
        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)

# TriXiumYPCBSoC -----------------------------------------------------------------------------------

# The (sdram, ddrphy, sdram_generator, sdram_checker) CSR pages of the two controllers: their
# own CSR bus in the ddr domain, behind a wishbone crossing at DDR_CSR_WINDOW, controller k's
# registers at controller 0's + k*DDR_CSR_STRIDE (firmware/ddr.c).
DDR_CSR_WINDOW = 0xF0020000
DDR_CSR_STRIDE = 0x2000

# The MMCM of each DDR3 PHY in its channel's clock region (channel 0: banks 11-13, the MMCM of
# bank 13's region; channel 1: banks 16-18, bank 17's, the command bank).
DDR_MMCM_LOCS = ["MMCME2_ADV_X0Y2", "MMCME2_ADV_X0Y6"]

class TriXiumYPCBSoC(SoCCore):
    def __init__(self, sys_clk_freq=150e6, ddr_clk_freq=400e6/3, engine_channels=4,
        main_ram_size  = 0x1000_0000,
        chan_stride    = 0x4000_0000,
        chan_skew      = 0x8000,
        read_time      = 256,
        write_time     = 128,
        refresh_postponing = 2,
        **kwargs):
        ddram_channels = 2
        assert engine_channels % ddram_channels == 0, \
            "engine channels shard evenly over the controllers"
        chan_mod = engine_channels//ddram_channels
        assert chan_mod & (chan_mod - 1) == 0, "channels per controller must be a power of two"
        platform = ypcb_00338_1p1.Platform()
        platform.add_extension(_pcie_x8_io)

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq, ddr_clk_freq=ddr_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # Console over PCIe/JTAGBone.
        kwargs["with_jtagbone"] = True
        # The BIOS's DDR3 calibration (firmware/ddr_s7hr.c) keeps its scan maps in SRAM.
        kwargs["integrated_sram_size"] = max(kwargs.get("integrated_sram_size") or 0, 0x4000)
        SoCCore.__init__(self, platform, sys_clk_freq,
            ident = "TriXium on YPCB-00338-1P1", **kwargs)
        self.add_constant("TRIXIUM_VARIANT", ord("b"))

        # DDR3 x2 ----------------------------------------------------------------------------------
        # Controller 0 also carries the CPU's main_ram (256 MB, through the L2 cache);
        # controller 1 has crossbar ports only.
        self.sdrams = []
        self.add_ddr_controllers(platform, ddram_channels, ddr_clk_freq, main_ram_size,
            kwargs.get("l2_size", 8192),
            ddr_controller_settings(read_time, write_time, refresh_postponing))
        set_main_ram_executable(self)

        # TriXium Memory Ports ---------------------------------------------------------------------
        # Engine channel c -> controller c // chan_mod: a 512-bit crossbar port behind an AXI
        # bridge, crossed into sys at the AXI level (memory/ddr.py).
        memory_ports = []
        for c in range(engine_channels):
            axi = AXIInterface(data_width=256, address_width=32, id_width=6)
            for j, m in enumerate(engine_ddr_port(self.sdrams[c//chan_mod], axi, clock_domain="sys_port")):
                setattr(self, f"engport{c}_{j}", m)
            memory_ports.append(axi)

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = S7PCIEPHY(platform, platform.request("pcie_x8", 1),
            data_width = 128,
            bar0_size  = 0x40000) # Covers DDR_CSR_WINDOW (CSR region + 0x20000): PHY status.
        self.pcie_phy.update_config({"PCIe_Blk_Locn": "X0Y0"})
        # The lanes are on GT banks 116 and 115, lane i on GTXE2_CHANNEL_X0Y(23 - i); the
        # core's default sites are a quad lower.
        self.pcie_phy.add_gt_loc_constraints(locs=[f"GTXE2_CHANNEL_X0Y{23 - i}" for i in range(8)],
            gt_type="gtx")
        self.add_pcie(phy=self.pcie_phy, ndmas=1, address_width=32)

        # CPU Bus Monitor (read over PCIe by software/host/cpumon.py) ------------------------------
        self.cpumon = CPUBusMonitor(self.cpu)
        self.csr.add("cpumon", 31) # Pinned: ctrl must stay at 0 (software/host/cpurst.py, console).

        # Leds -------------------------------------------------------------------------------------
        self.leds = LedChaser(
            pads         = platform.request_all("user_led"),
            sys_clk_freq = sys_clk_freq)

        # TriXium ----------------------------------------------------------------------------------
        # Per-channel windows: controller c // chan_mod's local HBM_WEIGHT_BASE (320 MB) +
        # (c % chan_mod)*chan_stride + (c % chan_mod)*chan_skew: 704 MB per channel, the second
        # ending at the controller's 2 GB. LiteDRAM's ROW_BANK_COL mapping on a 64-bit DDR3
        # channel (1024 columns, 8 KB rows) puts the bank in byte address bits 13..15: a stream
        # moves to the next bank every 8 KB and round the 8 banks every 64 KB. Both windows of a
        # controller stream the same relative addresses at the same time; the 32 KB skew keeps
        # them four banks apart, so the two crossbar ports never wait on each other's bank (nor
        # thrash one bank's row). The second window loses 32 KB (the converter's
        # --window-reserve). main_ram is controller 0's 0..256 MB, below every window; the BONS
        # staging area is the MB below channel 0's window.
        ctrl_size = 2 << 30
        assert chan_stride & (chan_stride - 1) == 0 and HBM_WEIGHT_BASE < chan_stride
        assert HBM_WEIGHT_BASE + chan_stride*chan_mod - (HBM_WEIGHT_BASE % chan_stride) <= ctrl_size
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride // 64
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000

        # Core: engine, sequencer and vector unit (the Qwen3 path needs no scratchpad unit).
        config = TriXiumConfig(scratchpad_xpm=True,
            chan_stride = chan_stride,
            chan_mod    = chan_mod,
            chan_skew   = chan_skew)
        self.trixium = TriXiumCore(memory_ports, config)
        for name, value in self.trixium.constants.items():
            self.add_constant(name, value)
        add_cluster_constraints(platform)

        # Scratchpad on the SoC bus (CPU access).
        self.bus.add_slave("scratchpad", self.trixium.bus,
            SoCRegion(origin=0xa000_0000, size=self.trixium.scratchpad.size, cached=False))

        # PCIe DMA0 (host -> FPGA) -> TriXium upload stream.
        self.comb += self.pcie_dma0.source.connect(self.trixium.upload,
            keep={"valid", "ready", "data"})

        # Board memory map, for the firmware.
        self.add_constant("TRIXIUM_LOADER_ADDS_CHAN_OFFSET", 1)
        self.add_constant("TRIXIUM_HDR_ADDR",         0x4000_0000 + HDR_CPU_OFFSET)
        self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", HBM_WEIGHT_BASE)
        self.add_constant("TRIXIUM_CHAN_STRIDE",      chan_stride)
        self.add_constant("TRIXIUM_N_CHANNELS",       engine_channels)
        self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
        self.add_constant("TRIXIUM_CHAN_SKEW",        chan_skew)
        self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
        self.add_constant("TRIXIUM_N_MEMCTRL",        ddram_channels)
        self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   DDR_CSR_STRIDE)

    def add_ddr_controllers(self, platform, n, ddr_clk_freq, main_ram_size, l2_size,
                            controller_settings):
        """n S7HRDDRPHY + MT41K256M8 controllers in the `ddr` domain (memory/ddr.py), each with
        a LiteDRAM BIST for the calibration's traffic test; the calibration replaces
        liblitedram's (CONFIG_SDRAM_CUSTOM_INIT, firmware/ddr_s7hr.c)."""
        phys, modules = [], []
        for i in range(n):
            phy = S7HRDDRPHY(
                pads         = PHYPadsReducer(platform.request("ddram", i), list(range(8))),
                sys_clk_freq = ddr_clk_freq,
                cl           = DDR3_CL,
                cwl          = DDR3_CWL,
                name         = f"ddrphy{i}",
                mmcm_loc     = DDR_MMCM_LOCS[i])
            phy.add_constraints(platform)
            phys.append(phy)
            modules.append(YPCBDDR3(ddr_clk_freq, "1:4"))
        add_ddr_domain_controllers(self, phys, modules, ddr_clk_freq, main_ram_size, l2_size,
            DDR_CSR_WINDOW, phy_renames={"sys": "ddr"}, controller_settings=controller_settings,
            with_bist=True)
        self.add_config("SDRAM_CUSTOM_INIT")
        self.add_constant("TRIXIUM_DDR_S7HR")
        self.add_constant("TRIXIUM_DDR_CKPHASE_STEPS", phys[0].ps_steps_per_tck)

# Build --------------------------------------------------------------------------------------------

# CPU: the FK33's (the Qwen3 path needs no FPU): 32 KB I$ + 32 KB D$, branch prediction, Zcbm
# (cache maintenance) and Zbb (clz).
VEXII_ARGS = ("--fetch-l1-ways=4 --fetch-l1-sets=128 --lsu-l1-ways=4 --lsu-l1-sets=128 "
    "--with-btb --with-ras --with-gshare --with-rvZcbm --with-rvZbb")

# The BIOS side of the DDR3 calibration: a software package the BIOS links (sdram_custom_init()).
BIOS_DDR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firmware", "bios")

def add_bios_ddr(builder):
    builder.add_software_package("libtrixiumddr", BIOS_DDR_DIR)
    builder.add_software_library("libtrixiumddr")

def main():
    # VexiiRiscv's netlist cache key must not depend on set ordering: hash randomization is
    # fixed at interpreter start, so run again with PYTHONHASHSEED=0.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from litex.build.parser import LiteXArgumentParser
    from software.lab.timing import gate
    parser = LiteXArgumentParser(platform=ypcb_00338_1p1.Platform, description="TriXium on the Inspur YPCB-00338-1P1.")
    parser.add_target_argument("--sys-clk-freq",       default=150e6,    type=float,                          help="System clock frequency (CPU, TriXium core).")
    parser.add_target_argument("--ddr-clk-freq",       default=400e6/3,  type=float,                          help="DDR3 controller clock frequency (DDR3 rate = 8x: 133.33e6 = DDR3-1066).")
    parser.add_target_argument("--engine-channels",    default=4,        type=int,                            help="Engine channels (multiple of 2).")
    parser.add_target_argument("--read-time",          default=256,      type=int,                            help="LiteDRAM read time (cycles before a pending write wins).")
    parser.add_target_argument("--write-time",         default=128,      type=int,                            help="LiteDRAM write time.")
    parser.add_target_argument("--refresh-postponing", default=2,        type=int,                            help="LiteDRAM refresh postponing (1-8).")
    parser.add_target_argument("--driver",             action="store_true",                                   help="Generate PCIe driver (always with --build).")
    parser.add_target_argument("--vivado-settings",    default=os.environ.get("LITEX_ENV_VIVADO", VIVADO_PATH), help="Vivado install sourced by the build (settings64.sh).")
    # The shipping recipe (overridable from the command line).
    parser.set_defaults(
        cpu_type                             = "vexiiriscv",
        update_repo                          = "no",
        vexii_args                           = VEXII_ARGS,
        vivado_synth_directive               = "AreaOptimized_medium",
        vivado_opt_directive                 = "ExploreWithRemap",
        vivado_place_directive               = "ExtraNetDelay_high",
        vivado_post_place_phys_opt_directive = "AggressiveExplore",
        vivado_route_directive               = "AggressiveExplore",
        vivado_post_route_phys_opt_directive = "AggressiveExplore",
        vivado_max_threads                   = 4,
    )
    args = parser.parse_args()

    soc = TriXiumYPCBSoC(
        sys_clk_freq       = args.sys_clk_freq,
        ddr_clk_freq       = args.ddr_clk_freq,
        engine_channels    = args.engine_channels,
        read_time          = args.read_time,
        write_time         = args.write_time,
        refresh_postponing = args.refresh_postponing,
        **parser.soc_argdict)
    add_zbb_constant(soc, args)

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", "ypcb")
    if builder_argd.get("csr_csv") is None:
        builder_argd["csr_csv"] = os.path.join(builder_argd["output_dir"], "csr.csv")
    builder = Builder(soc, **builder_argd)
    add_bios_ddr(builder)
    if args.build or args.driver:
        if not args.build:
            builder.compile_software = False
            builder.compile_gateware = False
        set_vivado_env(args.vivado_settings)
        verdict = builder.output_dir.rstrip("/") + ".verdict"
        try:
            builder.build(**parser.toolchain_argdict)
        except OSError:
            if builder.compile_gateware:
                write_verdict(verdict, 1)
            raise
        generate_litepcie_software(soc, os.path.join(builder.output_dir, "driver"), name="trixium")

        # A successful Vivado run is not a qualified bitstream: the timing gate decides
        # (strict: this board has no build that has run on a card yet).
        if builder.compile_gateware:
            report = os.path.join(builder.gateware_dir, f"{soc.platform.name}_timing.rpt")
            print("\nTiming gate:")
            rc = gate(report, strict=True, board="ypcb")
            write_verdict(verdict, rc)
            if rc:
                sys.exit(rc)

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

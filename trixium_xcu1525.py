#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium on the SQRL XCU1525 (XCVU9P, four DDR4 x64 channels, PCIe Gen3 x4).

The TriXium core streams its weights from the four DDR4 controllers: engine channel c reads
from controller c // 4 through its own crossbar port, at per-channel windows inside that
controller. The defaults are the configuration that runs the 27B (every scratchpad unit, the
DeltaNet and attention streams, DDR4-1500 in a 187.5 MHz ddr domain, sys 125 MHz), built with the
CPU, Vivado directives and timing gate it shipped with. Design decisions and board notes:
doc/boards.md.

From the repository root:
  ./trixium_xcu1525.py --build                         # -> build/xcu1525/ + build/xcu1525.verdict
  ./trixium_xcu1525.py --build --sys-clk-freq 137.5e6 --output-dir build/xcu1525_137
  ./trixium_xcu1525.py --load                          # openFPGALoader (JTAG)
  trixium deploy --board xcu1525 --model 27b           # the lab board: software/lab/deploy.py

Vivado 2024.1 (--vivado-settings). A VU9P route needs ~15 GB of disk and ~25 GB of RAM.
"""

import os
import sys

from migen import *
from migen.genlib.cdc import MultiReg
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *

from litex_boards.platforms import sqrl_xcu1525

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *
from litex.soc.interconnect.axi import AXIInterface

from litedram.modules import MT40A512M8
from litedram.phy import usddrphy

from litepcie.phy.usppciephy import USPPCIEPHY
from litepcie.software import generate_litepcie_software

from model.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from gateware.common import add_cluster_constraints, add_zbb_constant
from gateware.common import set_main_ram_executable, CPUBusMonitor
from gateware.common import VIVADO_PATH, set_vivado_env, write_verdict

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    """Clocks: `sys` (CPU, TriXium core) and the DDR4 `ddr`/`ddr4x` domains.

    The four DDR4 controllers and PHYs run in their own `ddr` domain (4:1: the DDR4 data rate is
    8 x ddr_clk_freq). The DDR MMCM (on channel 0's 300 MHz reference) makes ddr/ddr4x through
    parallel global buffers (BUFGCE_DIV /4 and BUFGCE /1 of one output) and the IDELAY reference;
    the PHYs capture on ddr4x through calibrated IDELAYs, so per-bank skew is absorbed by read
    leveling. A second MMCM on the same reference makes sys; the memory ports cross at the AXI
    level (memory/ddr.py), the controllers' CSRs behind a wishbone crossing.
    """
    def __init__(self, platform, sys_clk_freq, ddr_clk_freq, clk_channel=0):
        assert ddr_clk_freq != sys_clk_freq
        self.rst       = Signal()
        self.cd_sys    = ClockDomain()
        self.cd_sys4x  = ClockDomain()
        self.cd_pll4x  = ClockDomain()
        self.cd_idelay = ClockDomain()
        self.cd_ddr    = ClockDomain()
        self.cd_ddr4x  = ClockDomain()

        # # #

        # DDR MMCM: ddr4x (before the buffers) and the IDELAY reference.
        clk300 = platform.request("clk300", clk_channel)
        self.pll = pll = USPMMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(clk300, 300e6)
        pll.create_clkout(self.cd_pll4x, ddr_clk_freq*4, buf=None, with_reset=False)
        # IDELAY reference at 400 MHz: one MMCM cannot make 4*200 = 800 and 500 from 300 without a
        # fractional divide; IDELAYCTRL takes 300-800 MHz and the PHY is told.
        self.idelay_freq = 400e6
        pll.create_clkout(self.cd_idelay, self.idelay_freq)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.

        # Sys MMCM.
        self.syspll = syspll = USPMMCM(speedgrade=-2)
        self.comb += syspll.reset.eq(self.rst)
        syspll.register_clkin(pll.clkin, 300e6)
        syspll.create_clkout(self.cd_sys, sys_clk_freq, with_reset=False)
        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_ddr.clk)

        # DDR clock buffers.
        self.specials += [
            Instance("BUFGCE_DIV",
                p_BUFGCE_DIVIDE=4,
                i_CE=1, i_I=self.cd_pll4x.clk, o_O=self.cd_ddr.clk),
            Instance("BUFGCE",
                i_CE=1, i_I=self.cd_pll4x.clk, o_O=self.cd_ddr4x.clk),
        ]
        # The two buffers share one MMCM output but drive very different loads (ddr fabric vs
        # four PHYs across three SLRs): have the placer/router balance them (UG912
        # CLOCK_DELAY_GROUP, on the buffer OUTPUT nets) for the OSERDESE3 CLK/CLKDIV skew checks
        # (0.270 ns max). Those checks must not be waived here.
        platform.add_platform_command(
            "set_property CLOCK_DELAY_GROUP trixium_ddr4 "
            "[get_nets {{{slow} {fast}}}]",
            slow=self.cd_ddr.clk, fast=self.cd_ddr4x.clk)

        # IDELAYCTRL (releases sys) and the ddr reset, synchronized on sys's.
        locked = Signal()
        self.comb += locked.eq(pll.locked & self.syspll.locked)
        self.idelayctrl = _IDELAYCTRL(platform,
            cd_ref = self.cd_idelay,
            cd_sys = self.cd_sys,
            locked = locked)
        self.specials += AsyncResetSynchronizer(self.cd_ddr, self.cd_sys.rst)

# IDELAYCTRL ---------------------------------------------------------------------------------------

class _IDELAYCTRL(LiteXModule):
    """LiteX's USPIDELAYCTRL with the reset counter on the sys clock in a reset-less domain,
    and a false path to the IDELAYCTRL RST pins.

    LiteX clocks its 64-cycle reset counter with the IDELAY reference, and Vivado replicates
    the IDELAYCTRL once per I/O bank: with four DDR4 channels the replicas sit in all three SLRs
    and the reset net needs more route than one reference period. The RST pin is asynchronous
    on UltraScale+ (UG571: a 52 ns minimum pulse, no clock relation), so the counter runs at the
    sys clock (512 ns of reset) and the RST paths are false paths. The RDY handling is LiteX's.

    The counter's domain must NOT be `sys`: sys is held in reset until the IDELAYCTRL is ready,
    and the IDELAYCTRL is not ready until this counter releases RST, so a counter in sys never
    counts and the SoC never leaves reset. It counts in a reset-less domain on the sys clock
    (test/gateware/test_targets.py).

    The count starts at MMCM lock: UG571 wants the IDELAYCTRL reset applied after REFCLK is
    stable. `locked` is synchronized into the counter's domain and holds RST asserted (counter
    reloaded) while low, since the MMCM outputs toggle during lock acquisition.
    """
    def __init__(self, platform, cd_ref, cd_sys, reset_cycles=64, ready_cycles=64, locked=None):
        self.cd_ic        = ClockDomain()
        self.cd_idelayrst = ClockDomain("idelayrst", reset_less=True)
        self.ic_reset     = ic_reset = Signal(reset=1)
        self.ic_ready     = ic_ready = Signal()

        # # #

        # Reset counter (sys clock, reset-less, restarted on lock loss).
        ic_reset_counter = Signal(max=reset_cycles, reset=reset_cycles-1)
        self.comb += self.cd_idelayrst.clk.eq(cd_sys.clk)
        locked_s = Signal()
        if locked is None:
            self.comb += locked_s.eq(1)
        else:
            self.specials += MultiReg(locked, locked_s, "idelayrst")
        self.sync.idelayrst += [
            If(~locked_s,
                ic_reset_counter.eq(ic_reset_counter.reset),
                ic_reset.eq(1)
            ).Elif(ic_reset_counter != 0,
                ic_reset_counter.eq(ic_reset_counter - 1)
            ).Else(
                ic_reset.eq(0)
            )
        ]

        # Ready counter: sys leaves reset ready_cycles after RDY.
        ic_ready_counter = Signal(max=ready_cycles, reset=ready_cycles-1)
        self.comb += self.cd_ic.clk.eq(cd_sys.clk)
        self.sync.ic += [
            cd_sys.rst.eq(1),
            If(ic_ready,
                If(ic_ready_counter != 0,
                    ic_ready_counter.eq(ic_ready_counter - 1)
                ).Else(
                    cd_sys.rst.eq(0)
                )
            )
        ]

        # IDELAYCTRL primitive.
        if platform is not None:
            self.specials += [
                Instance("IDELAYCTRL",
                    p_SIM_DEVICE = "ULTRASCALE",
                    i_REFCLK     = cd_ref.clk,
                    i_RST        = ic_reset,
                    o_RDY        = ic_ready),
                AsyncResetSynchronizer(self.cd_ic, ic_reset)
            ]
            platform.add_platform_command("set_false_path -to [get_pins -hierarchical "
                "-filter {{REF_NAME == IDELAYCTRL && NAME =~ *RST}}]")
        else:
            # Test: no primitive; the test drives ic_ready and resets cd_ic itself.
            self.specials += AsyncResetSynchronizer(self.cd_ic, ic_reset)

# TriXiumXCU1525SoC --------------------------------------------------------------------------------

# The (sdram, ddrphy) CSR pairs of the four controllers: their own CSR bus in the ddr domain,
# behind a wishbone crossing at DDR_CSR_WINDOW (between the CLINT and the PLIC), controller k's
# registers at controller 0's + k*DDR_CSR_STRIDE, which is how the firmware initializes 1-3 with
# the BIOS's own liblitedram (firmware/ddr.c).
DDR_CSR_WINDOW = 0xF0020000
DDR_CSR_STRIDE = 0x1000

class TriXiumXCU1525SoC(SoCCore):
    def __init__(self, sys_clk_freq=125e6, ddr_clk_freq=187.5e6, engine_channels=16,
        main_ram_size  = 0x1000_0000,
        chan_stride    = 0x4000_0000,
        chan_skew      = 0x8000,
        attn_geometry  = (24, 4, 256, 1024),
        cpu_slr        = 1,
        **kwargs):
        ddram_channels = 4
        assert engine_channels % ddram_channels == 0, \
            "engine channels shard evenly over the controllers"
        chan_mod = engine_channels//ddram_channels
        assert chan_mod & (chan_mod - 1) == 0, "channels per controller must be a power of two"
        platform = sqrl_xcu1525.Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq, ddr_clk_freq=ddr_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # Console over PCIe/JTAGBone.
        kwargs["with_jtagbone"] = True
        SoCCore.__init__(self, platform, sys_clk_freq,
            ident = "TriXium on XCU1525", **kwargs)
        self.add_constant("TRIXIUM_VARIANT", ord("b"))

        # DDR4 x4 ----------------------------------------------------------------------------------
        # Controller 0 also carries the CPU's main_ram (256 MB, through the L2 cache);
        # controllers 1-3 have crossbar ports only.
        self.sdrams = []
        self.add_ddr_controllers(platform, ddram_channels, ddr_clk_freq, main_ram_size,
            kwargs.get("l2_size", 8192))
        set_main_ram_executable(self)
        # Workaround for a Vivado DRC on the PHY's clocking (upstream target).
        platform.add_platform_command("set_property SEVERITY {{Warning}} [get_drc_checks PDCN-2736]")

        # TriXium Memory Ports ---------------------------------------------------------------------
        # Engine channel c -> controller c // chan_mod: a 512-bit crossbar port behind an AXI
        # bridge, crossed into sys at the AXI level (memory/ddr.py).
        memory_ports = []
        for c in range(engine_channels):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            for j, m in enumerate(engine_ddr_port(self.sdrams[c//chan_mod], axi, clock_domain="sys_port")):
                setattr(self, f"engport{c}_{j}", m)
            memory_ports.append(axi)

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = USPPCIEPHY(platform, platform.request("pcie_x4"),
            data_width = 128,
            bar0_size  = 0x40000) # Covers DDR_CSR_WINDOW (CSR region + 0x20000): host DDR trainer.
        self.add_pcie(phy=self.pcie_phy, ndmas=1, address_width=32)

        # CPU Bus Monitor (read over PCIe by software/host/cpumon.py) ------------------------------
        self.cpumon = CPUBusMonitor(self.cpu)
        self.csr.add("cpumon", 31) # Pinned: ctrl must stay at 0 (software/host/cpurst.py, console).
        if cpu_slr is not None:
            # The CPU in one SLR: spread over two, its memory-bus return crosses the boundary
            # at 8 logic levels and misses timing.
            platform.add_platform_command("create_pblock pb_cpu")
            platform.add_platform_command(f"resize_pblock pb_cpu -add SLR{cpu_slr}")
            platform.add_platform_command(
                "add_cells_to_pblock pb_cpu [get_cells -quiet -hierarchical -filter "
                "{{NAME =~ VexiiRiscvLitex_*}}]")

        # Leds -------------------------------------------------------------------------------------
        self.leds = LedChaser(
            pads         = platform.request_all("user_led"),
            sys_clk_freq = sys_clk_freq)

        # TriXium ----------------------------------------------------------------------------------
        # Per-channel windows: controller c // chan_mod's local HBM_WEIGHT_BASE (320 MB) +
        # (c % chan_mod)*chan_stride + (c % chan_mod)*chan_skew: 704 MB per channel, the last
        # ending at the controller's 4 GB (the 27B: 370 MB of ternary weights + its 159 MB int16
        # embedding table + KV and recurrent state per channel). The skew starts the four
        # windows of a controller in four different DDR4 bank groups, so the four streams never
        # thrash one bank's row (nor meet tCCD_L); the last window loses (chan_mod-1)*skew = 96 KB
        # (the converter's --window-reserve). main_ram is controller 0's 0..256 MB, below every
        # window; the BONS staging area is the MB below channel 0's window.
        assert chan_stride & (chan_stride - 1) == 0 and HBM_WEIGHT_BASE < chan_stride
        assert HBM_WEIGHT_BASE + chan_stride*chan_mod - (HBM_WEIGHT_BASE % chan_stride) <= 4 << 30
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride // 64
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000

        # Core: every scratchpad unit, the DeltaNet and attention streams.
        config = TriXiumConfig(scratchpad_xpm=True,
            chan_stride              = chan_stride,
            chan_mod                 = chan_mod,
            chan_skew                = chan_skew,
            with_hadamard            = True,
            with_activation_quantize = True,
            with_projection_restore  = True,
            with_silu                = True,
            with_norm                = True,
            with_conv                = True,
            with_dnstream            = True,
            with_attnstream          = True,
            attnstream_options       = tuple(zip(("lanes", "groups", "hb_max", "t_max"),
                (attn_geometry[0], attn_geometry[1], attn_geometry[2]//8, attn_geometry[3]))))
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

    def add_ddr_controllers(self, platform, n, ddr_clk_freq, main_ram_size, l2_size):
        """n USPDDRPHY + MT40A512M8 controllers in the `ddr` domain (memory/ddr.py)."""
        phys, modules = [], []
        for i in range(n):
            phy = usddrphy.USPDDRPHY(
                pads             = platform.request("ddram", i),
                memtype          = "DDR4",
                sys_clk_freq     = ddr_clk_freq,
                iodelay_clk_freq = self.crg.idelay_freq)
            # liblitedram's write DQ-DQS training (per-module centering of the DQ output delay
            # against the DQS), which the US+ PHY does not declare by default: without it,
            # controller 0's byte lane 7 comes out of write leveling at the edge of its write eye.
            phy.settings.write_dq_dqs_training = True
            phys.append(phy)
            modules.append(MT40A512M8(ddr_clk_freq, "1:4"))
        add_ddr_domain_controllers(self, phys, modules, ddr_clk_freq, main_ram_size, l2_size,
            DDR_CSR_WINDOW, phy_renames={"sys": "ddr", "sys4x": "ddr4x"})

# Build --------------------------------------------------------------------------------------------

# CPU: 32 KB I$ + 32 KB D$ (the per-layer firmware code path does not fit 16 KB), branch
# prediction, the FPU (the 27B's CPU path is ~150 M float ops per token), Zcbm (cache
# maintenance for the resident lists) and Zbb (clz).
VEXII_ARGS = ("--fetch-l1-ways=4 --fetch-l1-sets=128 --lsu-l1-ways=4 --lsu-l1-sets=128 "
    "--with-btb --with-ras --with-gshare --with-rvZcbm --with-rvZbb --with-rvf")

def main():
    # VexiiRiscv's netlist cache key must not depend on set ordering: hash randomization is
    # fixed at interpreter start, so run again with PYTHONHASHSEED=0.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from litex.build.parser import LiteXArgumentParser
    from software.lab.timing import gate
    parser = LiteXArgumentParser(platform=sqrl_xcu1525.Platform, description="TriXium on the SQRL XCU1525.")
    parser.add_target_argument("--sys-clk-freq",    default=125e6,   type=float,                             help="System clock frequency (CPU, TriXium core).")
    parser.add_target_argument("--ddr-clk-freq",    default=187.5e6, type=float,                             help="DDR4 controller clock frequency (DDR4 rate = 8x: 187.5e6 = DDR4-1500).")
    parser.add_target_argument("--engine-channels", default=16,      type=int,                               help="Engine channels (multiple of 4).")
    parser.add_target_argument("--cpu-slr",         default=1,       type=int,                               help="CPU pblock SLR.")
    parser.add_target_argument("--driver",          action="store_true",                                     help="Generate PCIe driver (always with --build).")
    parser.add_target_argument("--vivado-settings", default=os.environ.get("LITEX_ENV_VIVADO", VIVADO_PATH), help="Vivado install sourced by the build (settings64.sh).")
    parser.add_target_argument("--timing-approval", default="ddr4-overclock",                             help="Timing gate approval (software/lab/timing.py APPROVALS); DDR4-1500 overclocks the ISERDES.")
    # The shipping recipe (overridable from the command line).
    parser.set_defaults(
        cpu_type                             = "vexiiriscv",
        update_repo                          = "no", # Never update ext/VexiiRiscv from here.
        vexii_args                           = VEXII_ARGS,
        vivado_synth_directive               = "AreaOptimized_medium",
        vivado_opt_directive                 = "ExploreWithRemap",
        vivado_place_directive               = "SSI_SpreadLogic_high",
        vivado_post_place_phys_opt_directive = "AggressiveExplore",
        vivado_route_directive               = "AggressiveExplore",
        vivado_post_route_phys_opt_directive = "AggressiveExplore",
        vivado_max_threads                   = 4,
    )
    args = parser.parse_args()

    soc = TriXiumXCU1525SoC(
        sys_clk_freq    = args.sys_clk_freq,
        ddr_clk_freq    = args.ddr_clk_freq,
        engine_channels = args.engine_channels,
        cpu_slr         = args.cpu_slr,
        **parser.soc_argdict)
    add_zbb_constant(soc, args)

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", "xcu1525")
    if builder_argd.get("csr_csv") is None:
        builder_argd["csr_csv"] = os.path.join(builder_argd["output_dir"], "csr.csv")
    builder = Builder(soc, **builder_argd)
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

        # A successful Vivado run is not a qualified bitstream: the timing gate decides. The strict
        # release verdict is printed for information; DDR4-1500 runs the USPDDRPHY's ISERDES past
        # its minimum period, an out-of-spec overclock passed under its per-build approval.
        if builder.compile_gateware:
            report = os.path.join(builder.gateware_dir, f"{soc.platform.name}_timing.rpt")
            print("\nTiming gate (strict, for information):")
            gate(report, strict=True, serializer_baseline="v8")
            print(f"\nTiming gate (approval {args.timing_approval}):")
            rc = gate(report, experiment=True, serializer_baseline="v8", approval=args.timing_approval)
            write_verdict(verdict, rc)
            if rc:
                sys.exit(rc)

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

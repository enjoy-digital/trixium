#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# TriXium on the SQRL XCU1525 (XCVU9P, four DDR4 x64 channels, PCIe Gen3 x4).
#
# The TriXium core streams its weights from the four DDR4 controllers: engine channel c reads
# from controller c // 4 through its own crossbar port, at per-channel windows inside that
# controller. The defaults are the configuration that runs the 27B (every scratchpad unit, the
# DeltaNet and attention streams, DDR4-1500 in a 187.5 MHz ddr domain, sys 125 MHz). Design
# decisions and board notes: docs/xcu1525.md.
#
# Build (from the repository root):
#   tools/build_xcu1525.sh <name>                       # Vivado directives + timing gate.
#   python3 -m trixium.targets.xcu1525 --build          # or by hand.

import os

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer
from migen.genlib.cdc import MultiReg

from litex.gen import *

from litex_boards.platforms import sqrl_xcu1525

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *
from litex.soc.interconnect.axi import AXIInterface

import litedram.modules
from litedram.phy import usddrphy

from litepcie.phy.usppciephy import USPPCIEPHY
from litepcie.software import generate_litepcie_software

from trixium.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET
from trixium.core.core import TriXiumConfig, TriXiumCore
from trixium.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from trixium.targets.common import add_cluster_constraints, CPUBusMonitor

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq, ddr_clk_freq, clk_channel=0):
        self.rst       = Signal()
        self.cd_sys    = ClockDomain()
        self.cd_sys4x  = ClockDomain()
        self.cd_pll4x  = ClockDomain()
        self.cd_idelay = ClockDomain()

        # # #

        # The four DDR4 controllers and PHYs run in their own `ddr` domain (4:1: the DDR4 data
        # rate is 8x ddr_clk_freq); the CPU and the TriXium core in `sys`. The DDR MMCM (on
        # channel 0's 300 MHz reference) makes ddr/ddr4x through parallel global buffers
        # (BUFGCE_DIV /4 and BUFGCE /1 of one output) and the IDELAY reference; the PHYs
        # capture on ddr4x through calibrated IDELAYs, so per-bank skew is absorbed by read
        # leveling. A second MMCM on the same reference makes sys; the memory ports cross at
        # the AXI level (memory/ddr.py), the controllers' CSRs behind a wishbone crossing.
        assert ddr_clk_freq != sys_clk_freq
        self.cd_ddr   = ClockDomain()
        self.cd_ddr4x = ClockDomain()
        cd_slow, cd_fast = self.cd_ddr, self.cd_ddr4x
        dram_freq = ddr_clk_freq
        clk300 = platform.request("clk300", clk_channel)
        self.pll = pll = USPMMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(clk300, 300e6)
        pll.create_clkout(self.cd_pll4x, dram_freq*4, buf=None, with_reset=False)
        # The IDELAY reference: 400 MHz (one MMCM cannot make 4*200 = 800 and 500 from 300
        # without a fractional divide; IDELAYCTRL takes 300-800 MHz and the PHY is told).
        self.idelay_freq = 400e6
        pll.create_clkout(self.cd_idelay, self.idelay_freq)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.
        self.syspll = syspll = USPMMCM(speedgrade=-2)
        self.comb += syspll.reset.eq(self.rst)
        syspll.register_clkin(pll.clkin, 300e6)
        syspll.create_clkout(self.cd_sys, sys_clk_freq, with_reset=False)
        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_ddr.clk)

        self.specials += Instance("BUFGCE_DIV",
            p_BUFGCE_DIVIDE=4,
            i_CE=1, i_I=self.cd_pll4x.clk, o_O=cd_slow.clk)
        self.specials += Instance("BUFGCE",
            i_CE=1, i_I=self.cd_pll4x.clk, o_O=cd_fast.clk)

        # These parallel buffers share one MMCM output, but their distribution trees
        # have very different loads (sys fabric vs four PHYs across three SLRs).
        # Ask the placer/router to balance them; do NOT waive OSERDESE3 CLK/CLKDIV
        # checks. v8 missed the 0.270 ns maximum skew by 0.003 ns at 16 endpoints.
        # AMD UG912 CLOCK_DELAY_GROUP: apply to the clock-buffer OUTPUT net segments.
        platform.add_platform_command(
            "set_property CLOCK_DELAY_GROUP trixium_ddr4 "
            "[get_nets {{{slow} {fast}}}]",
            slow=cd_slow.clk, fast=cd_fast.clk)

        locked = Signal()
        self.comb += locked.eq(pll.locked & self.syspll.locked)
        self.idelayctrl = _IDELAYCTRL(platform, cd_ref=self.cd_idelay, cd_sys=self.cd_sys, locked=locked)
        # ddr leaves reset with sys (after the IDELAYCTRL is ready), synchronized.
        self.specials += AsyncResetSynchronizer(self.cd_ddr, self.cd_sys.rst)

# _IDELAYCTRL --------------------------------------------------------------------------------------

class _IDELAYCTRL(LiteXModule):
    """LiteX's USPIDELAYCTRL with the reset counter on the sys clock in a reset-less domain,
    and a false path to the IDELAYCTRL RST pins.

    LiteX clocks its 64-cycle reset counter with the 500 MHz IDELAY reference and Vivado
    replicates the IDELAYCTRL once per I/O bank: with four DDR4 channels the replicas sit in
    all three SLRs, the reset net is 2.1 ns of route against a 2 ns period, and every one of
    the first build's 82 failing endpoints was this net (xcu1525_n16_125_v1: WNS -0.809, the
    engine and controllers at 125 MHz all met). The RST pin is asynchronous on UltraScale+
    (UG571: a 52 ns minimum pulse, no clock relation), so the counter runs at the sys clock
    (512 ns of reset) and the RST paths are false paths. The RDY handling is LiteX's.

    The counter's domain must NOT be `sys`: sys is held in reset until the IDELAYCTRL is
    ready, and the IDELAYCTRL is not ready until this counter releases RST - a counter in
    sys never counts and the SoC never leaves reset (v2 on the board, 2026-09-19: PCIe up,
    every CSR read 0xffffffff). LiteX's original avoided it by counting in the reference
    domain; this one counts in a reset-less domain on the sys clock (test_xcu1525_crg.py).

    The count starts at MMCM lock, not at configuration: UG571 wants the IDELAYCTRL reset
    applied after REFCLK is stable, and LiteX gets that for free because its reference
    domain is reset by ~locked. Here `locked` is synchronised into the counter's domain and
    holds RST asserted (counter reloaded) while it is low - the MMCM outputs toggle for the
    ~100 us of lock acquisition, 64 sys cycles would have released RST into an unstable
    REFCLK, and a RDY that never rises is the v2 symptom again.
    """
    def __init__(self, platform, cd_ref, cd_sys, reset_cycles=64, ready_cycles=64, locked=None):
        self.cd_ic       = ClockDomain()
        self.cd_idelayrst = ClockDomain("idelayrst", reset_less=True)
        self.ic_reset    = ic_reset = Signal(reset=1)
        self.ic_ready    = ic_ready = Signal()
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
        if platform is not None:
            self.specials += [
                Instance("IDELAYCTRL",
                    p_SIM_DEVICE = "ULTRASCALE",
                    i_REFCLK     = cd_ref.clk,
                    i_RST        = ic_reset,
                    o_RDY        = ic_ready),
                AsyncResetSynchronizer(self.cd_ic, ic_reset)
            ]
            platform.add_platform_command(
                "set_false_path -to [get_pins -hierarchical -filter {{REF_NAME == IDELAYCTRL && NAME =~ *RST}}]")
        else:
            # Test: no primitive; the test drives ic_ready and resets cd_ic itself.
            self.specials += AsyncResetSynchronizer(self.cd_ic, ic_reset)

# TriXiumXCU1525SoC -------------------------------------------------------------------------------

# The (sdram, ddrphy) CSR pairs of the four controllers: their own CSR bus in the ddr domain,
# behind a wishbone crossing at DDR_CSR_WINDOW (between the CLINT and the PLIC), controller k's
# registers at controller 0's + k*DDR_CSR_STRIDE - which is how the firmware initialises 1-3
# with the BIOS's own liblitedram (firmware/ddr.c).
DDR_CSR_WINDOW = 0xF0020000
DDR_CSR_STRIDE = 0x1000

class TriXiumXCU1525SoC(SoCCore):
    def _add_ddr_domain_controllers(self, platform, n, ddr_clk_freq, main_ram_size, l2_size):
        phys, modules = [], []
        for i in range(n):
            phy = usddrphy.USPDDRPHY(
                pads             = platform.request("ddram", i),
                memtype          = "DDR4",
                sys_clk_freq     = ddr_clk_freq,
                iodelay_clk_freq = self.crg.idelay_freq)
            # liblitedram's write DQ-DQS training (per-module centring of the DQ output delay
            # against the DQS), which the US+ PHY does not declare by default: controller 0's
            # byte lane 7 comes out of write leveling at the edge of its write eye without it.
            phy.settings.write_dq_dqs_training = True
            phys.append(phy)
            modules.append(litedram.modules.MT40A512M8(ddr_clk_freq, "1:4"))
        add_ddr_domain_controllers(self, phys, modules, ddr_clk_freq, main_ram_size, l2_size,
            DDR_CSR_WINDOW, phy_renames={"sys": "ddr", "sys4x": "ddr4x"})

    def __init__(self, sys_clk_freq=125e6, ddr_clk_freq=187.5e6, engine_channels=16,
        main_ram_size  = 0x1000_0000,
        chan_stride    = 0x4000_0000,
        chan_skew      = 0x8000,
        attn_geometry  = (24, 4, 256, 1024),
        cpu_slr        = 1,
        **kwargs):
        ddram_channels = 4
        assert engine_channels % ddram_channels == 0, "engine channels shard evenly over the controllers"
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
        self._add_ddr_domain_controllers(platform, ddram_channels, ddr_clk_freq, main_ram_size,
            kwargs.get("l2_size", 8192))
        # VexiiRiscv builds its memory map from region.mode and enforces execute permission.
        for _rn, _r in self.bus.regions.items():
            if _r.origin == 0x4000_0000 and not _r.linker:
                _r.mode = "rwx"
        # Workaround for a Vivado DRC on the PHY's clocking (upstream target).
        platform.add_platform_command("set_property SEVERITY {{Warning}} [get_drc_checks PDCN-2736]")

        # Memory ports: engine channel c -> controller c // chan_mod, a 512-bit crossbar port
        # behind an AXI bridge, crossed into sys at the AXI level (memory/ddr.py).
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

        # CPU bus monitor (read over PCIe by tools/cpumon.py) -------------------------------------
        self.cpumon = CPUBusMonitor(self.cpu)
        self.csr.add("cpumon", 31)  # Pinned: ctrl must stay at 0 (fk33_cpurst.py, pcie_console).
        if cpu_slr is not None:
            # The CPU in one SLR: spread over two, its memory-bus return crossed the boundary
            # at 8 logic levels (-0.50 ns).
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
        # windows of a controller in four different DDR4 bank groups, so the four streams
        # never thrash one bank's row (nor meet tCCD_L); the last window loses
        # (chan_mod-1)*skew = 96 KB (the converter's --window-reserve). main_ram is controller
        # 0's 0..256 MB, below every window; the BONS staging area is the MB below channel
        # 0's window.
        assert chan_stride & (chan_stride - 1) == 0 and HBM_WEIGHT_BASE < chan_stride
        assert HBM_WEIGHT_BASE + chan_stride*chan_mod - (HBM_WEIGHT_BASE % chan_stride) <= 4 << 30
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride // 64
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000
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
        self.bus.add_slave("scratchpad", self.trixium.bus,
            SoCRegion(origin=0xa000_0000, size=self.trixium.scratchpad.size, cached=False))
        self.comb += self.pcie_dma0.source.connect(self.trixium.upload, keep={"valid", "ready", "data"})

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

# Build --------------------------------------------------------------------------------------------

def main():
    from trixium.migen_fast import install as install_migen_fast
    install_migen_fast()  # Linear-time Verilog conversion (minutes -> seconds).
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=sqrl_xcu1525.Platform, description="TriXium on the SQRL XCU1525.")
    parser.add_target_argument("--sys-clk-freq",    default=125e6,   type=float, help="System clock frequency (CPU and TriXium core).")
    parser.add_target_argument("--ddr-clk-freq",    default=187.5e6, type=float, help="DDR4 controller clock (4:1: data rate 8x; 187.5e6 = DDR4-1500).")
    parser.add_target_argument("--engine-channels", default=16,      type=int,   help="Engine channels (a multiple of 4: the DDR4 controllers).")
    parser.add_target_argument("--cpu-slr",         default=1,       type=int,   help="SLR of the CPU pblock.")
    parser.add_target_argument("--driver",          action="store_true",         help="Generate LitePCIe driver headers.")
    args = parser.parse_args()

    soc = TriXiumXCU1525SoC(
        sys_clk_freq    = args.sys_clk_freq,
        ddr_clk_freq    = args.ddr_clk_freq,
        engine_channels = args.engine_channels,
        cpu_slr         = args.cpu_slr,
        **parser.soc_argdict)

    # Zbb in the CPU (--vexii-args ... --with-rvZbb): the toolchain (GCC 10.1) predates the
    # extension, so the firmware reaches clz through a .insn encoding behind this.
    if "--with-rvZbb" in (getattr(args, "vexii_args", "") or "") or \
       "zbb" in (getattr(soc.cpu, "isa_map", None) or []):
        soc.add_constant("TRIXIUM_HAS_ZBB")

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", f"xcu1525_n{args.engine_channels}")
    if builder_argd.get("csr_csv") is None:
        builder_argd["csr_csv"] = os.path.join(builder_argd["output_dir"], "csr.csv")
    builder = Builder(soc, **builder_argd)
    if args.build or args.driver:
        if not args.build:
            builder.compile_software = False
            builder.compile_gateware = False
        builder.build(**parser.toolchain_argdict)
    if args.driver:
        generate_litepcie_software(soc, os.path.join(builder.output_dir, "driver"))
    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

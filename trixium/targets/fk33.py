#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# TriXium on the SQRL FK33 (XCVU33P, 8 GB HBM2, PCIe Gen3 x4).
#
# The TriXium core streams its weights from HBM: engine channel c is pseudo-channel c + 1
# (channel 0 carries the CPU's main_ram). The defaults are the configuration that runs the
# Bonsai 1.7B and 8B (16 channels, HBM at 800 MHz, sys = HBM AXI clock = 212.5 MHz). Board
# bring-up and operation: docs/boards.md.
#
# Build:
#   tools/build_fk33.sh <name>                      # Vivado directives + timing gate.
#   python3 -m trixium.targets.fk33 --build         # or by hand.

import os

from migen import *

from litex.gen import *

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.cores.ram.xilinx_usp_hbm2 import USPHBM2, add_usphbm2_pseudochannels
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *

from litepcie.phy.usppciephy import USPHBMPCIEPHY
from litepcie.software import generate_litepcie_software

from trixium.layout import HBM_WEIGHT_BASE, HBM_CHAN_STRIDE, HDR_CPU_OFFSET
from trixium.core.core import TriXiumConfig, TriXiumCore
from trixium.targets.common import add_cluster_constraints, add_zbb_constant
from trixium.targets.common import set_main_ram_executable
from trixium.targets.platform_fk33 import Platform

# Constants ----------------------------------------------------------------------------------------

# The HBM IP configurations, one directory per stack clock (their Vivado support files share
# names, among them the memory files that encode the PLL divider).
IP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ip")

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    """Clocks: sys (CPU, TriXium core, HBM AXI), the HBM reference and the HBM APB clock."""
    def __init__(self, platform, sys_clk_freq):
        self.rst        = Signal()
        self.cd_sys     = ClockDomain()
        self.cd_hbm_ref = ClockDomain()
        self.cd_apb     = ClockDomain()

        # # #

        self.pll = pll = USPMMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(platform.request("clk200"), 200e6)
        pll.create_clkout(self.cd_sys,     sys_clk_freq)
        pll.create_clkout(self.cd_hbm_ref, 100e6)
        pll.create_clkout(self.cd_apb,     100e6)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.

# TriXiumFK33SoC -----------------------------------------------------------------------------------

class TriXiumFK33SoC(SoCCore):
    def __init__(self, sys_clk_freq=212.5e6, hbm_tck=800, engine_channels=16, **kwargs):
        # The HBM AXI clock (sys) must sit in [tck/4, tck/2] (PG276). tck is the stack clock
        # set by the IP's PLL divider: 900 MHz (1800 Mbps) -> 225..450 MHz; 800 MHz (1600 Mbps)
        # -> 200..400 MHz at 11% less bandwidth, which buys timing headroom.
        assert hbm_tck in (800, 900)
        assert (hbm_tck/4)*1e6 <= sys_clk_freq <= (hbm_tck/2)*1e6
        assert 1 <= engine_channels <= 31
        platform = Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # No physical UART: console over JTAGBone/PCIe.
        kwargs["with_jtagbone"]  = True
        kwargs["jtagbone_chain"] = 2 # Chain 1 already used by HBM2 debug probes.
        SoCCore.__init__(self, platform, sys_clk_freq,
            ident = "TriXium B on FK33", **kwargs)
        self.add_constant("TRIXIUM_VARIANT", ord("b"))

        # HBM --------------------------------------------------------------------------------------
        # 32 pseudo-channels: channel 0 is the CPU's main_ram, channels 1..N the TriXium memory
        # ports (their own AXI ports: the SoC bus bridges do not burst).
        hbm_ip = "hbm_0" if hbm_tck == 900 else f"hbm_{hbm_tck}"
        hbm = USPHBM2(platform, hbm_ip_name=hbm_ip)
        # USPHBM2 adds <cwd>/ip/hbm/<name>.xci: use this stack clock's own directory instead.
        hbm.add_sources = lambda platform: platform.add_ip(
            os.path.join(IP_DIR, f"hbm{hbm_tck}", hbm_ip + ".xci"))
        self.hbm = hbm = ClockDomainsRenamer({"axi": "sys"})(hbm)
        add_usphbm2_pseudochannels(soc=self, hbm=hbm, channels=(0,), main_channel=0)
        set_main_ram_executable(self)
        # The apb<->sys crossings (HBM AXI resets driven from the apb domain, apb_complete
        # status into CSRs) are quasi-static: false paths.
        platform.add_false_path_constraints(self.crg.cd_sys.clk, self.crg.cd_apb.clk)
        # The HBM IP embeds HBM monitor debug cores: the auto-inserted debug hub needs its
        # clock connected explicitly (opt_design: "dbg_hub/clk has 1 unconnected channels").
        platform.add_platform_command("connect_debug_port dbg_hub/clk [get_nets apb_clk]")

        # TriXium Memory Ports ---------------------------------------------------------------------
        # Engine channel c -> HBM pseudo-channel c + 1, on an AXI port of its own.
        memory_ports = [hbm.axi[1 + i] for i in range(engine_channels)]

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = USPHBMPCIEPHY(platform, platform.request("pcie_x4"),
            data_width = 128,
            bar0_size  = 0x20000)
        self.add_pcie(phy=self.pcie_phy, ndmas=1, address_width=32)

        # Leds -------------------------------------------------------------------------------------
        self.leds = LedChaser(
            pads         = platform.request_all("user_led"),
            sys_clk_freq = sys_clk_freq)

        # TriXium ----------------------------------------------------------------------------------
        # One flat HBM address space: channel c's window is pseudo-channel c + 1's, at
        # HBM_WEIGHT_BASE + c*HBM_CHAN_STRIDE (256 MB per pseudo-channel, port-matched).

        # Core: engine, sequencer and vector unit (no scratchpad units).
        config = TriXiumConfig(scratchpad_xpm=True,
            chan_stride = HBM_CHAN_STRIDE,
            chan_mod    = 0,
            chan_skew   = 0)
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
        self.add_constant("TRIXIUM_CHAN_STRIDE",      HBM_CHAN_STRIDE)
        self.add_constant("TRIXIUM_N_CHANNELS",       engine_channels)

# Build --------------------------------------------------------------------------------------------

def main():
    from trixium.migen_fast import install as install_migen_fast
    install_migen_fast() # Linear-time Verilog conversion (minutes -> seconds).
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=Platform, description="TriXium on the SQRL FK33.")
    parser.add_target_argument("--sys-clk-freq",    default=212.5e6, type=float,                     help="System clock frequency (= HBM AXI clock).")
    parser.add_target_argument("--hbm-tck",         default=800,     type=int,   choices=[800, 900], help="HBM stack clock (MHz).")
    parser.add_target_argument("--engine-channels", default=16,      type=int,                       help="Engine channels (HBM pseudo-channels 1..N, max 31).")
    parser.add_target_argument("--driver",          action="store_true",                             help="Generate PCIe driver.")
    args = parser.parse_args()

    soc = TriXiumFK33SoC(
        sys_clk_freq    = args.sys_clk_freq,
        hbm_tck         = args.hbm_tck,
        engine_channels = args.engine_channels,
        **parser.soc_argdict)
    add_zbb_constant(soc, args)

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", f"fk33_n{args.engine_channels}")
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

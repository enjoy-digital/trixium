#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium on the SQRL FK33 (XCVU33P, 8 GB HBM2, PCIe Gen3 x4).

The TriXium core streams its weights from HBM: engine channel c is pseudo-channel c + 1
(channel 0 carries the CPU's main_ram). The defaults are the configuration that runs the
Bonsai 1.7B and 8B (16 channels, HBM at 800 MHz, sys = HBM AXI clock = 212.5 MHz), built with
the CPU, Vivado directives and timing gate it shipped with. Board bring-up and operation:
doc/boards.md.

From the repository root:
  ./trixium_fk33.py --build                          # -> build/fk33/ + build/fk33.verdict
  ./trixium_fk33.py --load                           # OpenOCD (JTAG, onboard FT2232)
  trixium deploy --board fk33 --model 1p7b           # the lab board: software/lab/deploy.py

What the recipe's directives and caches are worth:
- fetch-l1 128 sets x 4 ways = 32 KB instruction cache: the per-layer firmware code path
  (~24 KB) does not fit 16 KB (-15% on the token); lsu-l1 the same (-5.5%).
- AreaOptimized_medium: on this ~98%-full die the synthesis area directive closes timing;
  AltSpreadLogic_low is the placer directive that does. Judge a build by TNS and failing
  endpoints against the last shipped one, never by WNS alone (software/lab/timing.py).
- update_repo "no": never let LiteX update ext/VexiiRiscv from here (needs network, and a
  half-done update leaves a CPU that does not compile against the local SpinalHDL).
- If post-route phys_opt crashes, re-route from the placed checkpoint and skip it (worth
  ~0.15 ns of WNS, not correctness).
Vivado 2024.1 (the 2025.2 install lacks the VU33P; --vivado-settings). A route peaks at ~11-15 GB.
"""

import os
import sys
import shutil

from migen import *

from litex.gen import *

from litex_boards.platforms.sqrl_fk33 import Platform

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.cores.ram.xilinx_usp_hbm2 import USPHBM2, add_usphbm2_pseudochannels
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *

from litepcie.phy.usppciephy import USPHBMPCIEPHY
from litepcie.software import generate_litepcie_software

from model.layout import HBM_WEIGHT_BASE, HBM_CHAN_STRIDE, HDR_CPU_OFFSET
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.common import add_cluster_constraints, add_zbb_constant
from gateware.common import set_main_ram_executable
from gateware.common import VIVADO_PATH, set_vivado_env, write_verdict

# Constants ----------------------------------------------------------------------------------------

# The HBM IP: stack clock 800 MHz (PLL divider 8), AXI clock window 200..400 MHz (PG276). Builds
# use a copy in their own directory: Vivado generates the IP's outputs next to the xci.
HBM_XCI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateware", "memory", "hbm_800.xci")

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
    def __init__(self, sys_clk_freq=212.5e6, engine_channels=16, hbm_xci=HBM_XCI, **kwargs):
        # The HBM AXI clock (sys) must sit in [tck/4, tck/2] of the IP's 800 MHz stack clock.
        assert 200e6 <= sys_clk_freq <= 400e6
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
        hbm = USPHBM2(platform, hbm_ip_name="hbm_800", xci_path=hbm_xci)
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

# CPU: 32 KB I$ + 32 KB D$, branch prediction, Zcbm (cache maintenance for the resident lists)
# and Zbb (clz).
VEXII_ARGS = ("--fetch-l1-ways=4 --fetch-l1-sets=128 --lsu-l1-ways=4 --lsu-l1-sets=128 "
    "--with-btb --with-ras --with-gshare --with-rvZcbm --with-rvZbb")

def main():
    # VexiiRiscv's netlist cache key must not depend on set ordering: hash randomization is
    # fixed at interpreter start, so run again with PYTHONHASHSEED=0.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from litex.build.parser import LiteXArgumentParser
    from software.lab.timing import gate
    parser = LiteXArgumentParser(platform=Platform, description="TriXium on the SQRL FK33.")
    parser.add_target_argument("--sys-clk-freq",    default=212.5e6, type=float,                             help="System clock frequency (= HBM AXI clock).")
    parser.add_target_argument("--engine-channels", default=16,      type=int,                               help="Engine channels (HBM pseudo-channels 1..N, max 31).")
    parser.add_target_argument("--driver",          action="store_true",                                     help="Generate PCIe driver (always with --build).")
    parser.add_target_argument("--vivado-settings", default=os.environ.get("LITEX_ENV_VIVADO", VIVADO_PATH), help="Vivado install sourced by the build (settings64.sh).")
    # The shipping recipe (overridable from the command line).
    parser.set_defaults(
        cpu_type                             = "vexiiriscv",
        update_repo                          = "no",
        vexii_args                           = VEXII_ARGS,
        vivado_synth_directive               = "AreaOptimized_medium",
        vivado_opt_directive                 = "ExploreWithRemap",
        vivado_place_directive               = "AltSpreadLogic_low",
        vivado_post_place_phys_opt_directive = "AggressiveExplore",
        vivado_route_directive               = "AggressiveExplore",
        vivado_post_route_phys_opt_directive = "AggressiveExplore",
    )
    args = parser.parse_args()

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", "fk33")
    hbm_xci = os.path.join(builder_argd["output_dir"], "ip", "hbm_800", "hbm_800.xci")
    os.makedirs(os.path.dirname(hbm_xci), exist_ok=True)
    shutil.copyfile(HBM_XCI, hbm_xci)

    soc = TriXiumFK33SoC(
        sys_clk_freq    = args.sys_clk_freq,
        engine_channels = args.engine_channels,
        hbm_xci         = hbm_xci,
        **parser.soc_argdict)
    add_zbb_constant(soc, args)

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

        # The FK33 ships on the empirical gate (no worse than the builds that have run on the
        # card): no FK33 build has closed the strict gate.
        if builder.compile_gateware:
            report = os.path.join(builder.gateware_dir, f"{soc.platform.name}_timing.rpt")
            print("\nTiming gate:")
            rc = gate(report)
            write_verdict(verdict, rc)
            if rc:
                sys.exit(rc)

    # Load: litex-boards' OpenOCD programmer (onboard FT2232, pld load); openFPGALoader reports
    # "done" on this board while the factory flash image wins the race.
    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

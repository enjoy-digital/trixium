#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Build/Use (from the repo root, ip/hbm/hbm_0.xci is resolved relative to CWD):
# ./trixium.py --variant=a --build --load
# litex_server --jtag --jtag-config=prog/openocd_xc7_ft2232.cfg --jtag-chain=2
# litex_term crossover
#
# Variants:
# - a: Infrastructure : HBM2 + PCIe x4 + per-channel bandwidth BIST + JTAGBone + Crossover-UART.
# - b: a + Ternary MatVec Engine (host-driven, N=4 HBM channels).
# - c: Full inference  : Engine + Vector Unit + KV-Cache + Top-K + firmware token loop.

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

from trixium.gateware.platform_fk33 import Platform

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
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

# BonsaiSoC ----------------------------------------------------------------------------------------

class BonsaiSoC(SoCCore):
    def __init__(self, variant="a", sys_clk_freq=225e6, pcie_lanes=4, **kwargs):
        assert variant in ["a", "b", "c"]
        assert 225e6 <= sys_clk_freq <= 450e6 # HBM AXI clock range.
        self.bonsai_variant = variant
        platform = Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # No physical UART: console over JTAGBone/PCIe.
        kwargs["with_jtagbone"]  = True
        kwargs["jtagbone_chain"] = 2 # Chain 1 already used by HBM2 debug probes.
        SoCCore.__init__(self, platform, sys_clk_freq,
            ident = f"TriXium {variant.upper()} on FK33", **kwargs)
        self.add_constant("BONSAI_VARIANT", ord(variant))

        # HBM --------------------------------------------------------------------------------------
        # 32 pseudo-channels; CPU/BIOS window on channel 0 only (main_ram). Channels 1..31 are
        # reserved for the engine/BIST through their native AXI ports (the AXI-Lite SoC bridges
        # don't burst).
        self.hbm = hbm = ClockDomainsRenamer({"axi": "sys"})(USPHBM2(platform))
        add_usphbm2_pseudochannels(soc=self, hbm=hbm, channels=(0,), main_channel=0)

        # The apb<->sys crossings (HBM AXI resets driven from the apb domain, apb_complete
        # status into CSRs) are quasi-static: declare them false paths.
        platform.add_false_path_constraints(self.crg.cd_sys.clk, self.crg.cd_apb.clk)

        # The HBM IP (rev 16) embeds HBM monitor debug cores; the auto-inserted debug hub needs
        # its clock connected explicitly or opt_design errors with "The debug port 'dbg_hub/clk'
        # has 1 unconnected channels".
        platform.add_platform_command("connect_debug_port dbg_hub/clk [get_nets apb_clk]")

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = USPHBMPCIEPHY(platform, platform.request(f"pcie_x{pcie_lanes}"),
            data_width = {4: 128, 8: 256, 16: 512}[pcie_lanes],
            bar0_size  = 0x20000)
        self.add_pcie(phy=self.pcie_phy, ndmas=1, address_width=32)

        # Leds -------------------------------------------------------------------------------------
        self.leds = LedChaser(
            pads         = platform.request_all("user_led"),
            sys_clk_freq = sys_clk_freq)

        # Bonsai -----------------------------------------------------------------------------------
        # Channel budget: ch0 = CPU main_ram (AXI-Lite window); engine/loader own ch1..N;
        # BIST muxes over the remaining channels.
        engine_channels = {"a": 0, "b": 4, "c": 8}[variant]
        engine_axis     = [hbm.axi[1 + i] for i in range(engine_channels)]
        bist_channels   = list(range(1 + engine_channels, 32))

        if variant in ["b", "c"]:
            from trixium.gateware.scratchpad import BonsaiScratchpad
            from trixium.gateware.matvec import BonsaiMatVecEngine
            from trixium.gateware.csr_front import BonsaiEngineCSR
            from trixium.gateware.loader import BonsaiHBMLoader
            from litex.soc.integration.soc import SoCRegion

            self.scratchpad = BonsaiScratchpad(size=256*1024)
            self.engine     = BonsaiMatVecEngine(self.scratchpad, engine_axis)
            self.engine_csr = BonsaiEngineCSR(self.engine)
            self.loader     = BonsaiHBMLoader(engine_axis, data_width=128)

            # Scratchpad on the SoC bus (uncached: shared with the engine).
            self.bus.add_slave("scratchpad", self.scratchpad.add_wishbone_port(),
                SoCRegion(origin=0xa000_0000, size=self.scratchpad.size, cached=False))

            # Weight upload: PCIe DMA0 host->device stream feeds the loader.
            self.comb += [
                self.pcie_dma0.source.connect(self.loader.sink, keep={"valid", "ready", "data"}),
            ]

        if variant == "c":
            raise NotImplementedError("variant c: vector unit/KV/top-k WIP")

        # Per-channel HBM bandwidth BIST (channels not owned by the engine).
        from trixium.gateware.bist import BonsaiHBMBIST
        self.bist = BonsaiHBMBIST(hbm.axi, channels=bist_channels)

# Build --------------------------------------------------------------------------------------------

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=Platform, description="TriXium SoC on FK33.")
    parser.add_target_argument("--variant",      default="a",               help="Bonsai variant (a: infra, b: matvec, c: full).")
    parser.add_target_argument("--sys-clk-freq", default=225e6, type=float, help="System clock frequency (= HBM AXI clock, 225-450 MHz).")
    parser.add_target_argument("--pcie-lanes",   default=4, type=int,       choices=[2, 4, 8, 16], help="PCIe lane count.")
    parser.add_target_argument("--driver",       action="store_true",       help="Generate LitePCIe driver headers.")
    args = parser.parse_args()

    # Builds must run from the repo root: the HBM XCI is resolved as ip/hbm/hbm_0.xci from CWD.
    assert os.path.exists(os.path.join("ip", "hbm", "hbm_0.xci")), \
        "ip/hbm/hbm_0.xci not found; run from the repository root."

    soc = BonsaiSoC(
        variant      = args.variant,
        sys_clk_freq = args.sys_clk_freq,
        pcie_lanes   = args.pcie_lanes,
        **parser.soc_argdict)

    # Per-variant build directory (default Builder output_dir collides across variants).
    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", f"trixium_{args.variant}")
    if builder_argd.get("csr_csv") is None:
        builder_argd["csr_csv"] = "csr.csv"

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

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

from litex_boards.platforms import sqrl_fk33
from litex_boards.targets.sqrl_fk33 import BaseSoC

from litex.soc.integration.builder import *

from litepcie.software import generate_litepcie_software

# Constants ----------------------------------------------------------------------------------------

# CPU/BIOS HBM window: channel 0 only (main_ram); channels 1..31 are reserved for the engine/BIST
# and accessed through their native AXI ports (the AXI-Lite SoC bridges don't burst).
CPU_HBM_CHANNELS = (0,)

# BonsaiSoC ----------------------------------------------------------------------------------------

class BonsaiSoC(BaseSoC):
    def __init__(self, variant="a", sys_clk_freq=250e6, pcie_lanes=4, **kwargs):
        assert variant in ["a", "b", "c"]
        self.bonsai_variant = variant

        BaseSoC.__init__(self,
            sys_clk_freq = sys_clk_freq,
            with_pcie    = True,
            pcie_lanes   = pcie_lanes,
            with_hbm     = True,
            hbm_channels = CPU_HBM_CHANNELS,
            **kwargs)
        self.add_constant("BONSAI_VARIANT", ord(variant))

        # The upgraded HBM IP (rev 16, Vivado 2024.1) embeds HBM monitor debug cores; the
        # auto-inserted debug hub needs its clock connected explicitly or opt_design errors
        # with "The debug port 'dbg_hub/clk' has 1 unconnected channels".
        self.platform.add_platform_command("connect_debug_port dbg_hub/clk [get_nets apb_clk]")

        # Variant a: HBM bandwidth BIST on channels 1..31 (added once gateware/bist.py lands).
        # Variant b: + MatVec engine (host-driven).
        # Variant c: + Vector unit/KV/Top-K + firmware token loop.
        if variant in ["b", "c"]:
            raise NotImplementedError(f"variant {variant} not implemented yet (engine gateware WIP)")

# Build --------------------------------------------------------------------------------------------

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=sqrl_fk33.Platform, description="TriXium SoC on FK33.")
    parser.add_target_argument("--variant",      default="a",               help="Bonsai variant (a: infra, b: matvec, c: full).")
    parser.add_target_argument("--sys-clk-freq", default=250e6, type=float, help="System clock frequency.")
    parser.add_target_argument("--pcie-lanes",   default=4, type=int,       choices=[4, 8, 16], help="PCIe lane count.")
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

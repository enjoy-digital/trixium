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
    def __init__(self, variant="a", sys_clk_freq=225e6, pcie_lanes=4, engine_channels=None,
                 with_nvme=False, with_vecunit=False, with_dual=False, hbm_tck=900,
                 **kwargs):
        # Phase-2 exploration flags (see docs/phase2.md).
        if with_nvme:
            raise NotImplementedError(
                "--with-nvme (LiteNVMe standalone weight loading) is a phase-2 option: the "
                "PCIe lanes switch from endpoint (host) to root (NVMe) role - mutually "
                "exclusive with --with-pcie host access. See docs/phase2.md.")
        assert variant in ["a", "b", "c"]
        # The HBM AXI clock must sit in [tck/4, tck/2] (PG276). tck is the stack clock set
        # by the IP's PLL divider: 900 MHz (1800 Mbps, ip/hbm/hbm_0.xci) -> 225..450 MHz;
        # 800 MHz (1600 Mbps, ip/hbm/hbm_800.xci) -> 200..400 MHz at 11% less bandwidth,
        # which buys timing headroom when the fabric cannot make 225.
        assert hbm_tck in (800, 900)
        assert (hbm_tck/4)*1e6 <= sys_clk_freq <= (hbm_tck/2)*1e6 # HBM AXI clock range.
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
        hbm_ip = "hbm_0" if hbm_tck == 900 else f"hbm_{hbm_tck}"
        self.hbm = hbm = ClockDomainsRenamer({"axi": "sys"})(
            USPHBM2(platform, hbm_ip_name=hbm_ip))
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
        # BIST muxes over the remaining channels. N scales the engine (phase-1 default 4;
        # 8B needs N >= 16, phase-2 throughput target N = 31).
        if engine_channels is None:
            engine_channels = {"a": 0, "b": 4, "c": 8}[variant]
        assert 0 <= engine_channels <= 31
        engine_axis   = [hbm.axi[1 + i] for i in range(engine_channels)]
        bist_channels = list(range(1 + engine_channels, 32))

        if variant in ["b", "c"]:
            from trixium.gateware.scratchpad import BonsaiScratchpad
            from trixium.gateware.matvec import BonsaiMatVecEngine
            from trixium.gateware.csr_front import BonsaiEngineCSR
            from trixium.gateware.loader import BonsaiHBMLoader
            from litex.soc.integration.soc import SoCRegion

            self.scratchpad = BonsaiScratchpad(size=256*1024)
            vecunit = None
            if with_vecunit:
                from trixium.gateware.vector import BonsaiVectorUnit
                vecunit = self.vecunit = BonsaiVectorUnit()
                self.add_constant("BONSAI_HAS_VECUNIT")
                self.add_constant("BONSAI_HAS_VECUNIT2") # M4 program set.
            seq = None
            if with_vecunit:
                from trixium.gateware.sequencer import BonsaiSequencer
                seq = self.seq = BonsaiSequencer()
                self.comb += [
                    vecunit.seq_en.eq(seq.busy),
                    vecunit.seq_start.eq(seq.vec_start),
                    vecunit.seq_pc.eq(seq.vec_pc),
                    seq.vec_done.eq(vecunit.done),
                    seq.vec_res6.eq(vecunit.res6.status),
                    vecunit.seq_seed7.eq(seq.vec_res6), # res6 chaining (S7 seed).
                ]
                self.comb += [vecunit.seq_args[i].eq(seq.vec_args[i]) for i in range(8)]
                self.add_constant("BONSAI_HAS_SEQ")
                # The 1KB vec imem must stay in distributed RAM: as BRAM, Vivado packs the
                # FETCHR register into the BRAM output stage, re-anchoring the decode
                # launch to the BRAM's clock-to-out + placement (the overnight worst path).
                platform.add_platform_command(
                    "set_property RAM_STYLE distributed "
                    "[get_cells -hierarchical -filter {{NAME =~ *imem_reg*}}]")
                # The vecunit EX cone is architecturally multicycle: operands register at
                # EX0, multiplier products at EX1, and every S/V-register effect is CE-
                # gated on EX2 - sources are stable for >= 2 full cycles before capture.
                # Declared as MCP so the tools stop fighting single-cycle closure on it.
                # Scoped from the operand/product registers only: paths with true 1-cycle
                # windows into S/V (e.g. scratchpad dat_r -> LDW writeback) stay default.
                platform.add_platform_command(
                    "set_multicycle_path -setup 2 "
                    "-from [get_cells -hierarchical -regexp "
                    "{{.*(m[0-9]+_reg|va_flat.*_reg|vb_flat.*_reg|sa_val.*_reg|sd_val.*_reg|rom_[0-9]+_dat0.*).*}}] "
                    "-to [get_cells -hierarchical -regexp {{.*((sr|vr)[0-9]_reg|red_(sum|max|smax).*_reg).*}}]")
                platform.add_platform_command(
                    "set_multicycle_path -hold 1 "
                    "-from [get_cells -hierarchical -regexp "
                    "{{.*(m[0-9]+_reg|va_flat.*_reg|vb_flat.*_reg|sa_val.*_reg|sd_val.*_reg|rom_[0-9]+_dat0.*).*}}] "
                    "-to [get_cells -hierarchical -regexp {{.*((sr|vr)[0-9]_reg|red_(sum|max|smax).*_reg).*}}]")
            self.engine     = BonsaiMatVecEngine(self.scratchpad, engine_axis, vecunit=vecunit,
                                                 seq=seq, dual=with_dual)
            if with_dual:
                # S3 dual-Q GQA + S6 speculative pair-verify: ~2x the engine datapath.
                self.add_constant("BONSAI_HAS_DUALQ")
            self.engine_csr = BonsaiEngineCSR(self.engine, seq=seq)
            self.loader     = BonsaiHBMLoader(engine_axis, data_width=128)
            # Engine kind-3 STORE (KV append) rides the loader's writers.
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
            self.add_constant("BONSAI_HAS_KVSTORE")
            self.add_constant("BONSAI_HAS_TOPK") # Argmax tap + suppressed writes.
            # Sideband (wishbone/CSR) scratchpad access only while all units are idle.
            units_idle = ~self.engine.sp_busy
            if vecunit is not None:
                units_idle = units_idle & ~vecunit.busy
            if seq is not None:
                units_idle = units_idle & ~seq.busy
            self.comb += self.scratchpad.units_idle.eq(units_idle)
            self.sp_csr     = self.scratchpad.add_csr_port() # Host peek/poke over BAR0/JTAG.

            # Scratchpad on the SoC bus (uncached: shared with the engine).
            self.bus.add_slave("scratchpad", self.scratchpad.add_wishbone_port(),
                SoCRegion(origin=0xa000_0000, size=self.scratchpad.size, cached=False))

            # Weight upload: PCIe DMA0 host->device stream feeds the loader.
            self.comb += [
                self.pcie_dma0.source.connect(self.loader.sink, keep={"valid", "ready", "data"}),
            ]

            # Firmware contract: BONS header+table live in HBM channel 0 (the CPU window) at
            # HBM_WEIGHT_BASE; engine-channel payloads at the same offset in channels 1..N.
            # Variant c == b gateware: the token loop is firmware + these constants.
            from trixium.layout import HBM_WEIGHT_BASE
            self.add_constant("BONSAI_HDR_ADDR",         0x4000_0000 + HBM_WEIGHT_BASE)
            self.add_constant("BONSAI_CHAN_WEIGHT_BASE", HBM_WEIGHT_BASE)
            self.add_constant("BONSAI_N_CHANNELS",       engine_channels)

        # Per-channel HBM bandwidth BIST (channels not owned by the engine). Skipped on
        # phase-2 inference builds (--with-vecunit): its 20+ channels of unsliced macro-
        # boundary FIFO paths were the residual 225MHz timing family, and the archived
        # phase-1 bitstreams remain the bandwidth-test vehicle.
        if not with_vecunit:
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
    parser.add_target_argument("--engine-channels", default=None, type=int, help="Engine HBM channel count override (default: 4 for b, 8 for c; 8B needs >= 16, max 31).")
    parser.add_target_argument("--with-nvme",    action="store_true",       help="Phase-2: standalone NVMe weight loading (LiteNVMe root, no host PCIe) - not implemented yet.")
    parser.add_target_argument("--hbm-tck",      default=900, type=int, choices=[800, 900], help="HBM stack clock MHz: 900 (AXI 225-450) or 800 (AXI 200-400, -11%% bandwidth).")
    parser.add_target_argument("--with-dual",    action="store_true",       help="Phase-2: dual-Q GQA + speculative pair-verify datapath (~2x engine area; needs headroom).")
    parser.add_target_argument("--with-vecunit", action="store_true",       help="Phase-2: microcoded vector unit + resident attention (all offload milestones sim-gated 2026-07-29; default off so the archived phase-1 bitstreams stay reproducible).")
    args = parser.parse_args()

    # Builds must run from the repo root: the HBM XCI is resolved as ip/hbm/hbm_0.xci from CWD.
    assert os.path.exists(os.path.join("ip", "hbm", "hbm_0.xci")), \
        "ip/hbm/hbm_0.xci not found; run from the repository root."

    soc = BonsaiSoC(
        variant         = args.variant,
        engine_channels = args.engine_channels,
        with_nvme       = args.with_nvme,
        with_vecunit    = args.with_vecunit,
        with_dual       = args.with_dual,
        hbm_tck         = args.hbm_tck,
        sys_clk_freq = args.sys_clk_freq,
        pcie_lanes   = args.pcie_lanes,
        **parser.soc_argdict)

    # Per-variant build directory (default Builder output_dir collides across variants).
    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        suffix = f"_n{args.engine_channels}" if args.engine_channels is not None else ""
        builder_argd["output_dir"] = os.path.join("build", f"trixium_{args.variant}{suffix}")
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

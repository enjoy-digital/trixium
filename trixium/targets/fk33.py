#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Build/Use (from the repo root: USPHBM2 resolves ip/hbm/<name>.xci relative to CWD, and ip/hbm
# is a symlink at the TCK variant you are building - ip/hbm900 or ip/hbm800):
# ./trixium.py --variant=a --build --load
# litex_server --jtag --jtag-config=prog/openocd_xc7_ft2232.cfg --jtag-chain=2
# litex_term crossover
#
# Variants:
# - a: Infrastructure : HBM2 + PCIe x4 + per-channel bandwidth BIST + JTAGBone + Crossover-UART.
# - b: a + Ternary MatVec Engine (host-driven, N=4 HBM channels).
# - c: Full inference  : Engine + Vector Unit + KV-Cache + Top-K + firmware token loop.

import atexit
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
                 with_memory_traffic=False, with_capabilities=True, with_native=False,
                 with_hadamard=False, with_activation_quantize=False, with_projection_restore=False,
                 with_embedding_restore=False, with_silu=False, with_norm=False, with_conv=False,
                 with_region_fetch=False,
                 **kwargs):
        # Phase-2 exploration flags (see docs/phase2.md).
        if with_nvme:
            raise NotImplementedError(
                "--with-nvme (LiteNVMe standalone weight loading) is a phase-2 option: the "
                "PCIe lanes switch from endpoint (host) to root (NVMe) role - mutually "
                "exclusive with --with-pcie host access. See docs/phase2.md.")
        assert variant in ["a", "b", "c"]
        # The HBM AXI clock must sit in [tck/4, tck/2] (PG276). tck is the stack clock set
        # by the IP's PLL divider: 900 MHz (1800 Mbps, ip/hbm900/hbm_0.xci) -> 225..450 MHz;
        # 800 MHz (1600 Mbps, ip/hbm800/hbm_800.xci) -> 200..400 MHz at 11% less bandwidth,
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
        # VexiiRiscv builds its memory map from region.mode and ENFORCES execute permission;
        # the HBM main_ram window is added with LiteX's default mode="rw", so the CPU cannot
        # fetch from it - firmware loads, then every ifetch faults and reads come back 0xFF.
        # VexRiscv ignored the attribute, which is why this never showed before. Mark the
        # real (non-linker) region at the main_ram origin executable.
        for _rn, _r in self.bus.regions.items():
            if _r.origin == 0x4000_0000 and not _r.linker:
                _r.mode = "rwx"


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

        with_bonsai            = variant in ["b", "c"]
        cluster_chan_mod       = 0     # One flat HBM address space.
        cluster_mem_cpu_offset = None  # The CPU window passes through 1:1 (the default).
        cluster_n_memctrl      = None  # The BIOS owns the one memory system.
        from trixium.layout import HBM_WEIGHT_BASE, HBM_CHAN_STRIDE
        cluster_chan_stride    = HBM_CHAN_STRIDE  # 256 MB: one pseudo-channel per engine channel.
        cluster_chan_skew      = 0     # No shared controller, no bank skew.
        cluster_weight_base    = HBM_WEIGHT_BASE  # +64 MB into it (port-matched, CPU-safe).
        cluster_with_dnrec     = False # The XCU1525-only exact-FP32 recurrence engine.
        # One portable wiring implementation; board policy stays here.
        if with_bonsai:
            from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster, native_read_options
            from trixium.gateware.xilinx_constraints import add_cluster_constraints
            from litex.soc.integration.soc import SoCRegion
            def memory_description():
                from trixium.capabilities import MemoryRegion, MemoryKind, Reservation, ReservationKind
                from trixium.layout import HBM_NCHANNELS, HBM_CHANNEL_SIZE, HBM_META_OFFSET
                regions = [MemoryRegion(i, MemoryKind.HBM, 0, i*HBM_CHANNEL_SIZE,
                                        HBM_CHANNEL_SIZE, locality=i) for i in range(HBM_NCHANNELS)]
                # Current CPU addresses reach physical PC4, not the planned F0 PC0.
                cpu = self.bus.regions['main_ram']
                reservations = [Reservation(cpu.origin//HBM_CHANNEL_SIZE, ReservationKind.CPU,
                                            cpu.origin % HBM_CHANNEL_SIZE, cpu.size),
                                Reservation(HBM_META_OFFSET//HBM_CHANNEL_SIZE, ReservationKind.UPLOAD_STAGING,
                                            HBM_META_OFFSET % HBM_CHANNEL_SIZE, 0x100000)]
                return regions, reservations
            native = (native_read_options(*memory_description(), [0]*engine_channels, write=True)
                      if with_native else {})
            config = BonsaiClusterConfig(
                with_vecunit=with_vecunit, with_dual=with_dual,
                with_dnrec=cluster_with_dnrec, chan_stride=cluster_chan_stride,
                chan_mod=cluster_chan_mod, chan_skew=cluster_chan_skew, with_hadamard=with_hadamard,
                with_activation_quantize=with_activation_quantize, with_projection_restore=with_projection_restore,
                with_embedding_restore=with_embedding_restore, with_region_fetch=with_region_fetch,
                with_silu=with_silu, with_norm=with_norm, with_conv=with_conv, **native)
            add_bonsai_cluster(self, engine_axis, config)
            if with_memory_traffic:
                from trixium.gateware.traffic import add_memory_traffic
                add_memory_traffic(self, engine_axis)
            add_cluster_constraints(platform, with_vecunit=with_vecunit)
            if cluster_with_dnrec:
                # Vendor implementations are target-owned, not portable-core sources.
                platform.add_ip(os.path.join("ip", "fp32", "fp32_add", "fp32_add.xci"))
                platform.add_ip(os.path.join("ip", "fp32", "fp32_mul", "fp32_mul.xci"))
            self.bus.add_slave("scratchpad", self.scratchpad.add_wishbone_port(),
                SoCRegion(origin=0xa000_0000, size=self.scratchpad.size, cached=False))
            self.comb += self.pcie_dma0.source.connect(
                self.loader.sink, keep={"valid", "ready", "data"})

            # Board memory map and host-upload convention (legacy v2/v3 images).
            from trixium.layout import HDR_CPU_OFFSET
            self.add_constant("BONSAI_LOADER_ADDS_CHAN_OFFSET", 1)
            self.add_constant("BONSAI_HDR_ADDR",         0x4000_0000 + HDR_CPU_OFFSET)
            self.add_constant("BONSAI_CHAN_WEIGHT_BASE", cluster_weight_base)
            self.add_constant("BONSAI_CHAN_STRIDE",      cluster_chan_stride)
            self.add_constant("BONSAI_N_CHANNELS",       engine_channels)
            if cluster_chan_mod:
                self.add_constant("BONSAI_CHAN_MOD", cluster_chan_mod)
            if cluster_chan_skew:
                self.add_constant("BONSAI_CHAN_SKEW", cluster_chan_skew)
            if cluster_mem_cpu_offset is not None:
                self.add_constant("BONSAI_MEM_CPU_OFFSET", cluster_mem_cpu_offset)
            if cluster_n_memctrl is not None:
                self.add_constant("BONSAI_N_MEMCTRL", cluster_n_memctrl)

            if with_capabilities:
                from trixium.gateware.capabilities import add_capabilities
                regions, reservations = memory_description()
                configured_sys_hz = round(self.crg.pll.compute_config()['clkout0_freq'])
                self.bonsai_capabilities = add_capabilities(self, config, engine_axis, regions, reservations,
                                                           sys_hz=configured_sys_hz)


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
    parser.add_target_argument("--with-memory-traffic", action="store_true", help="Passive coherent accelerator AXI counters (not physical HBM traffic).")
    parser.add_target_argument("--with-native", action="store_true", help="Experimental shared native PTQ/Q8 engine and append-only controls; no FK33 deployment qualification.")
    parser.add_target_argument("--with-hadamard", action="store_true", help="Experimental sequenced scratchpad integer rotations; no FK33 deployment qualification.")
    parser.add_target_argument("--with-activation-quantize", action="store_true", help="Experimental resident FP32 and post-rotation Q8 quantization; no FK33 deployment qualification.")
    parser.add_target_argument("--with-projection-restore", action="store_true", help="Experimental resident int32 projection to FP32 restoration; no FK33 deployment qualification.")
    parser.add_target_argument("--with-embedding-restore", action="store_true", help="Experimental compact PTQ embedding restoration; no FK33 deployment qualification.")
    parser.add_target_argument("--with-silu", action="store_true", help="Exact FP32 SiLU-multiply scratchpad unit (sequencer unit 11).")
    parser.add_target_argument("--with-norm", action="store_true", help="Exact RMSNorm/L2 scratchpad unit (sequencer unit 12).")
    parser.add_target_argument("--with-conv", action="store_true", help="Exact DeltaNet conv scratchpad unit (sequencer unit 13).")
    parser.add_target_argument("--with-region-fetch", action="store_true", help="Experimental sequenced 64-bit checked region reads; requires --with-native.")
    parser.add_target_argument("--no-capabilities", action="store_true", help="Omit the read-only capability ROM (legacy build reproduction).")
    parser.add_target_argument("--with-vecunit", action="store_true",       help="Phase-2: microcoded vector unit + resident attention (all offload milestones sim-gated 2026-07-29; default off so the archived phase-1 bitstreams stay reproducible).")
    args = parser.parse_args()

    # Builds must run from the repo root: USPHBM2 resolves the XCI as <cwd>/ip/hbm/<name>.xci.
    # Each TCK variant keeps its own directory (ip/hbm900, ip/hbm800) because Vivado's per-IP
    # support files collide - 31 identically named files, among them the
    # xpm_internal_config_file_*.mem that encode the PLL divider - so a merged ip/hbm would
    # silently reconfigure whichever stack was generated second. ip/hbm is therefore a symlink
    # at the variant being built (see docs/bringup.md), and is not tracked; header-only runs
    # (--driver) just need the XCI to resolve, so we make a file symlink and remove it at exit.
    _hbm_ip  = "hbm_0" if args.hbm_tck == 900 else f"hbm_{args.hbm_tck}"
    _hbm_xci = os.path.join("ip", "hbm", _hbm_ip + ".xci")
    _hbm_alt = os.path.join("ip", f"hbm{args.hbm_tck}", _hbm_ip + ".xci")
    if not os.path.exists(_hbm_xci):
        assert os.path.exists(_hbm_alt), \
            f"neither {_hbm_xci} nor {_hbm_alt} found; run from the repository root."
        assert not args.build, (
            f"{_hbm_ip} lives in ip/hbm{args.hbm_tck}/ but USPHBM2 reads ip/hbm/; point ip/hbm at "
            f"it first ('rm -f ip/hbm && ln -s hbm{args.hbm_tck} ip/hbm').")
        os.symlink(os.path.join("..", f"hbm{args.hbm_tck}", _hbm_ip + ".xci"), _hbm_xci)
        atexit.register(lambda p=_hbm_xci: os.path.islink(p) and os.unlink(p))

    soc = BonsaiSoC(
        variant         = args.variant,
        engine_channels = args.engine_channels,
        with_nvme       = args.with_nvme,
        with_vecunit    = args.with_vecunit,
        with_dual       = args.with_dual,
        with_memory_traffic = args.with_memory_traffic,
        with_capabilities = not args.no_capabilities,
        with_native       = args.with_native,
        with_hadamard     = args.with_hadamard,
        with_activation_quantize = args.with_activation_quantize,
        with_projection_restore = args.with_projection_restore,
        with_embedding_restore = args.with_embedding_restore,
        with_silu              = args.with_silu,
        with_norm              = args.with_norm,
        with_conv              = args.with_conv,
        with_region_fetch = args.with_region_fetch,
        hbm_tck         = args.hbm_tck,
        sys_clk_freq = args.sys_clk_freq,
        pcie_lanes   = args.pcie_lanes,
        **parser.soc_argdict)

    # Zbb in the CPU (--vexii-args ... --with-rvZbb): the toolchain (GCC 10.1) predates
    # the extension, so the firmware reaches clz through a .insn encoding behind this.
    if "--with-rvZbb" in (getattr(args, "vexii_args", "") or "") or \
       "zbb" in (getattr(soc.cpu, "isa_map", None) or []):
        soc.add_constant("BONSAI_HAS_ZBB")

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
        if hasattr(soc, 'bonsai_capabilities'):
            soc.bonsai_capabilities.write_artifacts(builder.output_dir)

    if args.driver:
        generate_litepcie_software(soc, os.path.join(builder.output_dir, "driver"))

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

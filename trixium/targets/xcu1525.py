#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# TriXium on the SQRL XCU1525: XCVU9P, four DDR4 x64 channels, PCIe. The same engine
# cluster as the FK33 (one shared integration module) over LiteDRAM: engine channel c
# streams from DDR4 controller c // 4 through a 256-bit crossbar port, at the FK33's
# per-channel window constants inside that controller (chan_mod = 4), so the images and the
# firmware constants are the same on both boards. Design decisions, projection and status:
# docs/xcu1525.md.
#
# Build (from the repo root):
#   tools/build_xcu1525.sh <name>          # the recipe that ships; or by hand:
#   ./trixium_xcu1525.py --engine-channels 16 --with-vecunit --build
#
# The whole SoC - CPU, engine cluster, the four controllers - runs in one `sys` domain at
# the DDR4 controller clock (125 MHz with the USPDDRPHY: 4:1, DDR4-1000). No clock-domain
# crossing anywhere; the accelerator has 4.8 ns of margin from its 212.5 MHz closure on the
# FK33. --sys-clk-freq=150e6 selects DDR4-1200; the current ISERDESE3 configuration
# has a 1.600 ns minimum clock period, so >156.25 MHz is not a qualified option.
# See docs/vcu-clock-qualification.md before changing clocks or deploying a build.

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
from litedram.frontend.axi import LiteDRAMAXI2Native

from litepcie.phy.usppciephy import USPPCIEPHY
from litepcie.software import generate_litepcie_software

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq, clk_channel=0, fast_clock_buffer="bufgce"):
        if fast_clock_buffer not in ("bufgce", "bufgce-div"):
            raise ValueError("fast_clock_buffer must be bufgce or bufgce-div")
        self.rst       = Signal()
        self.cd_sys    = ClockDomain()
        self.cd_sys4x  = ClockDomain()
        self.cd_pll4x  = ClockDomain()
        self.cd_idelay = ClockDomain()

        # # #

        # One MMCM (on DDR channel 0's 300 MHz reference) for the whole SoC: sys and sys4x
        # come from the same MMCM output through parallel global buffers (slow /4,
        # fast /1). The four PHYs share them; the USPDDRPHY captures on sys4x
        # through calibrated IDELAYs, so per-bank skew is absorbed by the read leveling.
        self.pll = pll = USPMMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(platform.request("clk300", clk_channel), 300e6)
        pll.create_clkout(self.cd_pll4x, sys_clk_freq*4, buf=None, with_reset=False)
        pll.create_clkout(self.cd_idelay, 500e6)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.

        self.specials += Instance("BUFGCE_DIV",
            p_BUFGCE_DIVIDE=4,
            i_CE=1, i_I=self.cd_pll4x.clk, o_O=self.cd_sys.clk)
        if fast_clock_buffer == "bufgce-div":
            # Opt-in timing experiment: UG949 permits a second BUFGCE_DIV with
            # divide=1 on CLK. Match buffer types while preserving the same MMCM
            # output, fabric/CLKDIV clock and clock-delay group. This is NOT a
            # waiver or a timing-closure claim; route and apply the strict gate.
            self.specials += Instance("BUFGCE_DIV",
                p_BUFGCE_DIVIDE=1,
                i_CE=1, i_CLR=0, i_I=self.cd_pll4x.clk, o_O=self.cd_sys4x.clk)
        else:
            self.specials += Instance("BUFGCE",
                i_CE=1, i_I=self.cd_pll4x.clk, o_O=self.cd_sys4x.clk)

        # These parallel buffers share one MMCM output, but their distribution trees
        # have very different loads (sys fabric vs four PHYs across three SLRs).
        # Ask the placer/router to balance them; do NOT waive OSERDESE3 CLK/CLKDIV
        # checks. v8 missed the 0.270 ns maximum skew by 0.003 ns at 16 endpoints.
        # AMD UG912 CLOCK_DELAY_GROUP: apply to the clock-buffer OUTPUT net segments.
        platform.add_platform_command(
            "set_property CLOCK_DELAY_GROUP bonsai_ddr4 "
            "[get_nets {{{slow} {fast}}}]",
            slow=self.cd_sys.clk, fast=self.cd_sys4x.clk)

        self.idelayctrl = _IDELAYCTRL(platform, cd_ref=self.cd_idelay, cd_sys=self.cd_sys, locked=pll.locked)

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

# BonsaiXCU1525SoC ---------------------------------------------------------------------------------

# The (sdram, ddrphy) CSR pairs sit on pages 8/9, 10/11, 12/13, 14/15: controller k's
# registers are controller 0's shifted by k*DDR_CSR_STRIDE, which is how the firmware
# initialises 1-3 with the BIOS's own liblitedram (firmware/ddr.c).
DDR_CSR_PAGE0  = 8
DDR_CSR_STRIDE = 0x1000

class BonsaiXCU1525SoC(SoCCore):
    def __init__(self, sys_clk_freq=125e6, pcie_lanes=4, engine_channels=16,
                 with_vecunit=True, with_dual=False, ddram_module="MT40A512M8",
                 ddram_channels=4, main_ram_size=0x1000_0000, chan_stride=0x4000_0000, chan_skew=0x8000,
                 port_width=512, with_memory_traffic=False, with_capabilities=True, with_native=False,
                 ddr_fast_clock_buffer="bufgce",
                 with_hadamard=False,
                 **kwargs):
        assert 1 <= ddram_channels <= 4
        assert engine_channels % ddram_channels == 0, \
            "engine channels shard evenly over the DDR4 controllers"
        chan_mod = engine_channels//ddram_channels
        assert chan_mod & (chan_mod - 1) == 0, "channels per controller must be a power of two"
        platform = sqrl_xcu1525.Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq, fast_clock_buffer=ddr_fast_clock_buffer)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # Console over PCIe/JTAGBone, as on the FK33.
        kwargs["with_jtagbone"] = True
        SoCCore.__init__(self, platform, sys_clk_freq,
            ident = "TriXium on XCU1525", **kwargs)
        self.add_constant("BONSAI_VARIANT", ord("b"))

        # DDR4 x4 ----------------------------------------------------------------------------------
        # Controller 0 is the BIOS's: main_ram (256 MB of it: the FK33's size, so the
        # firmware's memory map is the same) through the L2 cache. Controllers 1-3 have
        # crossbar ports only. All four on pinned CSR pages (see DDR_CSR_PAGE0).
        module_cls = getattr(litedram.modules, ddram_module)
        for i in range(ddram_channels):
            sfx = "" if i == 0 else str(i)
            self.csr.add(f"sdram{sfx}",  DDR_CSR_PAGE0 + 2*i)
            self.csr.add(f"ddrphy{sfx}", DDR_CSR_PAGE0 + 2*i + 1)
        self.sdrams = []
        memory_sizes = []
        for i in range(ddram_channels):
            sfx = "" if i == 0 else str(i)
            phy = usddrphy.USPDDRPHY(
                pads             = platform.request("ddram", i),
                memtype          = "DDR4",
                sys_clk_freq     = sys_clk_freq,
                iodelay_clk_freq = 500e6)
            # liblitedram's write DQ-DQS training (per-module centring of the DQ output delay
            # against the DQS), which the US+ PHY does not declare by default. On this board
            # controller 0's byte lane 7 came out of write leveling at the edge of its write
            # eye (three DQ bits lost written 1s, read leveling then found no window at any
            # bitslip, memtest KO); the same scan run from the host (tools/xcu1525_rdlevel.py)
            # moved lane 7's DQ by ~50 taps and the 16 MB memtest passed (2026-09-19). The
            # flag only adds the C define to sdram_phy.h: the BIOS in the ROM and the
            # firmware's ddr.c both run the training, no gateware changes.
            phy.settings.write_dq_dqs_training = True
            setattr(self, f"ddrphy{sfx}", phy)
            module = module_cls(sys_clk_freq, "1:4")
            g = module.geom_settings
            memory_sizes.append((1 << (g.bankbits + g.rowbits + g.colbits))*phy.settings.nranks*phy.settings.databits//8)
            if i == 0:
                self.add_sdram("sdram",
                    phy           = phy,
                    module        = module,
                    size          = main_ram_size,
                    l2_cache_size = kwargs.get("l2_size", 8192))
            else:
                self.add_sdram(f"sdram{sfx}",
                    phy                   = phy,
                    module                = module,
                    with_soc_interconnect = False)
            self.sdrams.append(getattr(self, f"sdram{sfx}"))
        # VexiiRiscv builds its memory map from region.mode and enforces execute
        # permission: mark main_ram executable (the FK33 target's lesson).
        for _rn, _r in self.bus.regions.items():
            if _r.origin == 0x4000_0000 and not _r.linker:
                _r.mode = "rwx"
        # Workaround for a Vivado DRC on the PHY's clocking (upstream target).
        platform.add_platform_command("set_property SEVERITY {{Warning}} [get_drc_checks PDCN-2736]")
        # Engine ports: engine channel c = controller c // chan_mod, a 256-bit crossbar port
        # (the controller is 512 bits wide at 4:1; four ports share its 8 GB/s at 125 MHz)
        # behind an AXI bridge. The read buffer holds the readers' 8 outstanding 16-beat
        # bursts (128 beats), so a channel is never throttled by the bridge.
        # The port itself: trixium/gateware/ddr_port.py (512-bit ports by default since
        # v6 - the 256-bit port's native up-converter costs ~5 cycles per controller word).
        from trixium.gateware.ddr_port import engine_ddr_port
        engine_axis = []
        for c in range(engine_channels):
            core = self.sdrams[c//chan_mod]
            axi  = AXIInterface(data_width=256, address_width=33, id_width=6)
            for j, m in enumerate(engine_ddr_port(core, axi, port_width=port_width)):
                setattr(self, f"engport{c}_{j}", m)
            engine_axis.append(axi)

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = USPPCIEPHY(platform, platform.request(f"pcie_x{pcie_lanes}"),
            data_width = {2: 64, 4: 128, 8: 256, 16: 512}[pcie_lanes],
            bar0_size  = 0x20000)
        self.add_pcie(phy=self.pcie_phy, ndmas=1, address_width=32)

        # Leds -------------------------------------------------------------------------------------
        self.leds = LedChaser(
            pads         = platform.request_all("user_led"),
            sys_clk_freq = sys_clk_freq)

        # Bonsai -----------------------------------------------------------------------------------
        # The window per engine channel: the FK33's base (320 MB) with a 1 GB stride, so
        # channel c's window is controller c // chan_mod's local 320 MB + (c % chan_mod) GB,
        # 704 MB usable (the stride minus the base's offset into it), the last of the four
        # ending exactly at the channel's 4 GB. 704 MB holds the 27B (370 MB of ternary
        # weights + its 159 MB int16 embedding table + KV and recurrent state per channel
        # at N=16), where the FK33's 192 MB window holds only the 8B without its table. The
        # 1.7B image is byte-identical to the FK33's; the 8B's gains its embedding table
        # (convert with --chan-stride 0x40000000). main_ram is controller 0's local
        # 0..256 MB, below every window; the BONS staging area is the megabyte below
        # channel 0's window (319 MB) as on the FK33. BONSAI_MEM_CPU_OFFSET = 0: main_ram's
        # origin is controller 0's local address 0 (the FK33's CPU window passes through
        # 1:1 at 0x4000_0000).
        from trixium.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET
        assert chan_stride & (chan_stride - 1) == 0 and HBM_WEIGHT_BASE < chan_stride
        assert HBM_WEIGHT_BASE + chan_stride*chan_mod - (HBM_WEIGHT_BASE % chan_stride) \
            <= 4 << 30, "the last window must end inside the channel's 4 GB"
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride // 64
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000, "main_ram must end below the staging area"
        with_bonsai            = True
        cluster_chan_mod       = chan_mod
        cluster_mem_cpu_offset = 0
        cluster_n_memctrl      = ddram_channels
        cluster_chan_stride    = chan_stride
        # The four windows of a controller start in four different DDR4 banks: a
        # power-of-two stride is a multiple of the bank period, so without this the four
        # streams sharing a controller thrash one bank's row (the LiteDRAM sim: 6-7.6
        # cycles/beat per port same-bank, 4.3 with a bank skew, 2026-09-19). In LiteDRAM's
        # ROW_BANK_COL mapping the bank index is byte address bits 13..16 (colbits 10, 64 B
        # words), DFI bank[1:0] = BA and bank[3:2] = BG: a 32 KB skew steps the bank GROUP,
        # so the four streams also never meet DDR4's tCCD_L. The last window loses
        # (chan_mod-1)*skew = 96 KB (the converter's --window-reserve).
        cluster_chan_skew      = chan_skew
        cluster_weight_base    = HBM_WEIGHT_BASE
        cluster_with_dnrec     = True  # Exact FP32 recurrence for the 27B fast path.
        # One portable wiring implementation; board policy stays here.
        if with_bonsai:
            from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster, native_read_options
            from trixium.gateware.xilinx_constraints import add_cluster_constraints
            from litex.soc.integration.soc import SoCRegion
            def memory_description():
                from trixium.capabilities import MemoryRegion, MemoryKind, Reservation, ReservationKind
                regions = [MemoryRegion(i, MemoryKind.DDR4, i, 0, size, locality=i)
                           for i, size in enumerate(memory_sizes)]
                reservations = [Reservation(0, ReservationKind.CPU, 0, self.bus.regions['main_ram'].size),
                                Reservation(0, ReservationKind.UPLOAD_STAGING,
                                            cluster_weight_base - 0x100000, 0x100000)]
                test_base = cluster_weight_base + chan_stride - (cluster_weight_base % chan_stride)
                test_base -= (chan_mod - 1)*chan_skew + 0x1000
                reservations += [Reservation(c//chan_mod, ReservationKind.DIAGNOSTIC,
                    test_base + (c % chan_mod)*(chan_stride + chan_skew), 0x1000)
                    for c in range(engine_channels)]
                return regions, reservations
            native = (native_read_options(*memory_description(),
                       [c//chan_mod for c in range(engine_channels)], write=True) if with_native else {})
            config = BonsaiClusterConfig(
                with_vecunit=with_vecunit, with_dual=with_dual,
                with_dnrec=cluster_with_dnrec, chan_stride=cluster_chan_stride,
                chan_mod=cluster_chan_mod, chan_skew=cluster_chan_skew, with_hadamard=with_hadamard, **native)
            add_bonsai_cluster(self, engine_axis, config)
            if with_memory_traffic:
                from trixium.gateware.traffic import add_memory_traffic
                add_memory_traffic(self, engine_axis,
                    [core.controller.interface for core in self.sdrams])
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
                configured_sys_hz = round(self.crg.pll.compute_config()['clkout0_freq']/4)
                self.bonsai_capabilities = add_capabilities(self, config, engine_axis, regions, reservations,
                                                           sys_hz=configured_sys_hz)


        self.add_constant("BONSAI_DDR_CSR_STRIDE", DDR_CSR_STRIDE)

# Build --------------------------------------------------------------------------------------------

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=sqrl_xcu1525.Platform,
                                 description="TriXium SoC on the SQRL XCU1525.")
    parser.add_target_argument("--sys-clk-freq",    default=125e6, type=float, help="System clock = DDR4 controller clock (4:1).")
    parser.add_target_argument("--ddr-fast-clock-buffer", default="bufgce", choices=["bufgce", "bufgce-div"],
                               help="DDR CLK buffer: legacy BUFGCE or experimental BUFGCE_DIV /1; both require strict routed timing qualification.")
    parser.add_target_argument("--pcie-lanes",      default=4, type=int,       choices=[2, 4, 8, 16], help="PCIe lane count.")
    parser.add_target_argument("--driver",          action="store_true",       help="Generate LitePCIe driver headers.")
    parser.add_target_argument("--engine-channels", default=16, type=int,      help="Engine channel count (a multiple of the DDR4 channel count).")
    parser.add_target_argument("--ddram-channels",  default=4, type=int,       help="DDR4 channels to instantiate (1-4).")
    parser.add_target_argument("--ddram-module",    default="MT40A512M8",      help="LiteDRAM module class of the DDR4 (upstream target: MT40A512M8).")
    parser.add_target_argument("--main-ram-size",   default=0x1000_0000, type=lambda x: int(x, 0), help="CPU main_ram size on controller 0 (default 256 MB, the FK33's).")
    parser.add_target_argument("--chan-stride",     default=0x4000_0000, type=lambda x: int(x, 0), help="Per-engine-channel stride inside a controller (default 1 GB: 704 MB windows, the 27B fits; 0x10000000 = the FK33's 256 MB).")
    parser.add_target_argument("--chan-skew",       default=0x8000, type=lambda x: int(x, 0), help="Extra per-channel offset inside a controller so the windows start in different DDR4 bank groups (default 32 KB; 0 = none).")
    parser.add_target_argument("--port-width",      default=512, type=int, help="Crossbar port width per engine channel: 512 (LiteX AXIUpConverter, default) or 256 (LiteDRAM's native up-converter).")
    parser.add_target_argument("--with-dual",       action="store_true",       help="Dual-Q GQA + speculative pair-verify datapath (~2x engine area).")
    parser.add_target_argument("--with-memory-traffic", action="store_true", help="Passive coherent AXI/controller traffic counters (new CSR bank).")
    parser.add_target_argument("--with-native", action="store_true", help="Experimental shared native PTQ/Q8 engine and append-only controls; not resident runtime qualification.")
    parser.add_target_argument("--with-hadamard", action="store_true", help="Experimental sequenced scratchpad integer rotations; not resident runtime qualification.")
    parser.add_target_argument("--no-capabilities", action="store_true", help="Omit the read-only capability ROM (legacy build reproduction).")
    parser.add_target_argument("--no-vecunit",      action="store_true",       help="Build without the vector unit + sequencer (engine only).")
    args = parser.parse_args()

    soc = BonsaiXCU1525SoC(
        sys_clk_freq    = args.sys_clk_freq,
        pcie_lanes      = args.pcie_lanes,
        engine_channels = args.engine_channels,
        with_vecunit    = not args.no_vecunit,
        with_dual       = args.with_dual,
        ddram_module    = args.ddram_module,
        ddram_channels  = args.ddram_channels,
        main_ram_size   = args.main_ram_size,
        chan_stride     = args.chan_stride,
        chan_skew       = args.chan_skew,
        port_width      = args.port_width,
        with_memory_traffic = args.with_memory_traffic,
        with_capabilities = not args.no_capabilities,
        with_native       = args.with_native,
        ddr_fast_clock_buffer = args.ddr_fast_clock_buffer,
        with_hadamard     = args.with_hadamard,
        **parser.soc_argdict)

    # Zbb in the CPU (--vexii-args ... --with-rvZbb): the toolchain (GCC 10.1) predates
    # the extension, so the firmware reaches clz through a .insn encoding behind this.
    if "--with-rvZbb" in (getattr(args, "vexii_args", "") or "") or \
       "zbb" in (getattr(soc.cpu, "isa_map", None) or []):
        soc.add_constant("BONSAI_HAS_ZBB")

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
        if hasattr(soc, 'bonsai_capabilities'):
            soc.bonsai_capabilities.write_artifacts(builder.output_dir)

    if args.driver:
        generate_litepcie_software(soc, os.path.join(builder.output_dir, "driver"))

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

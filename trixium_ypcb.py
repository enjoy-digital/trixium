#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium on the Inspur YPCB-00338-1P1 (XC7K480T-2, two 72-bit DDR3 channels, PCIe Gen2 x8).

The TriXium core streams its weights from the two DDR3 channels (64 data bits each, the ECC
lane unused): engine channel c reads from controller c // (channels/2) through its own crossbar port, at
channels/2 windows per controller. The DDR3 sits on HR banks (no ODELAY, a fly-by command bus over
nine x8 chips): the PHY is LiteDRAM's K7DDRPHY with its CK phase shift (write leveling by CK
phase) and the BIOS calibrates both controllers at boot (liblitedram's CK phase leveling,
SDRAM_CONTROLLERS). The defaults are the shipping recipe: the Qwen3 path's units (engine,
sequencer, vector unit; the Ternary-Bonsai 1.7B and 8B) on eight engine channels with 5-trit
weight codes (--trit5: images from trixium convert --board ypcb), DDR3-1066 in a 133.33 MHz ddr domain,
sys 125 MHz (the clock this K480T design closes timing at), the FK33's CPU configuration. Design notes: doc/boards.md.

From the repository root:
  ./trixium_ypcb.py --build                          # -> build/ypcb/ + build/ypcb.verdict
  ./trixium_ypcb.py --load                           # JTAG (openFPGALoader, Digilent HS2 cable)
  trixium deploy --board ypcb --model 1p7b           # the lab board: software/lab/deploy.py

Vivado 2024.1 (--vivado-settings). A K480T route needs ~12 GB of RAM.
"""

import os
import sys
import argparse

from migen import *

from litex.gen import *


from litex_boards.platforms import ypcb_00338_1p1

from litex.soc.cores.clock import *
from litex.soc.cores.led import LedChaser
from litex.soc.integration.soc import *
from litex.soc.integration.builder import *
from litex.soc.interconnect.axi import AXIInterface

from litedram.common import PHYPadsReducer
from litedram.modules import MT41K256M8, _SpeedgradeTimings
from litedram.core.controller import ControllerSettings
from litedram.phy.s7ddrphy import K7DDRPHY

from litepcie.phy.s7pciephy import S7PCIEPHY
from litepcie.software import generate_litepcie_software

from model.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET, ddr_chan_stride
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from gateware.common import add_cluster_constraints, add_zbb_constant
from gateware.common import set_main_ram_executable, CPUBusMonitor
from gateware.common import VIVADO_PATH, set_vivado_env, write_verdict

# The DDR3 has no DM pins: a write stores whole 512-bit controller words, so the engine ports
# read-modify-write their partially-strobed words (256-bit beats, KV rows of an odd number of
# beats; LiteDRAMAXI2Native's with_read_modify_write). Without it every STORE corrupts the
# other half of its words (the first card run's ddrtest: half the words wrong).
ENGINE_RMW = True

# --hybrid unit sizing for the K7-480T (out-of-context LUTs, K7 -2): unit 12 without its RoPE stage
# (the CPU rotates the 2B's 6 attention layers' q/k; ~half of its 41k), unit 1 at 16 lanes (6.7k
# against 14.3k at 32, ~2x the clocks per block) and unit 11 at 2 ways (4.6k against 7.0k). With
# every unit at the XCU1525's sizes the 4-channel SoC is 325k of 298.6k LUTs.
YPCB_HYBRID_UNIT_OPTIONS = (("norm_rope", False), ("hadamard_lanes", 16), ("silu_ways", 2))

# DDR3 ---------------------------------------------------------------------------------------------

# DDR3-1066 bin (CL 7 / CWL 6, the timings the board's MIG project used); LiteDRAM's defaults at
# 1.875 ns would be CL 10 / CWL 7.
DDR3_CL  = 7
DDR3_CWL = 6

def ddr3_latencies(ddr_clk_freq):
    """(CL, CWL) at a ddr clock (DDR3 rate = 8 x ddr): the DDR3-1066 bin's 7 / 6 up to 133.33 MHz,
    the DDR3-1333 bin's 9 / 7 for the overclocked rates up to 166.67 MHz (DDR3-1200: 150 MHz)."""
    if ddr_clk_freq <= 400e6/3 + 1e3:
        return DDR3_CL, DDR3_CWL
    assert ddr_clk_freq <= 500e6/3 + 1e3, "past the DDR3-1333 bin"
    return 9, 7

class YPCBDDR3(MT41K256M8):
    """The card's MT41K256M8 (2 Gb x8, DDR3-1600 parts run at DDR3-1066 by the HR banks), with
    the 2 Gb tRFC of 160 ns (LiteDRAM's class has 128 tCK, 240 ns at DDR3-1066)."""
    speedgrade_timings = {
        "1600": _SpeedgradeTimings(tRP=13.75, tRCD=13.75, tWR=15, tRFC=(None, 160), tFAW=(None, 40), tRAS=35),
    }
    speedgrade_timings["default"] = speedgrade_timings["1600"]


def ddr_controller_settings(read_time=256, write_time=128, refresh_postponing=1):
    """The controller settings (trixium_sim.py --board ypcb --ddr-settings). Measured on the
    truncated 1.7B (one layer + LM head, decode token): every setting within 0.25 %, the weight
    streams are reads and a token's KV writes are a few beats. Postponing refreshes costs (2:
    +0.1 %, 8: +0.25 %: a burst of refreshes stalls both streams at once); the read/write times
    are neutral here (32/16 to 256/128 within 0.02 %) and 256/128 keeps the writes of longer
    contexts grouped (-0.6 % on a Qwen3.5 decode in openTPU's co-simulation of this card)."""
    return lambda: ControllerSettings(
        read_time          = read_time,
        write_time         = write_time,
        refresh_postponing = refresh_postponing)

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    """Clocks: `sys` (CPU, TriXium core), `ddr` (the two LiteDRAM controllers, DDR3 rate =
    8 x ddr) and the 200 MHz IDELAY reference, from one MMCM on the 50 MHz oscillator (the
    board's reset pin is not wired). Each DDR3 PHY makes its own interface clocks from `ddr`
    (the K7DDRPHY in CK phase mode: one MMCM per channel). The memory ports cross sys <-> ddr at
    the AXI level (memory/ddr.py), the controllers' CSRs behind a wishbone crossing.
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
        # Exact frequencies (the controllers' timings and refresh interval are computed for
        # them); ddr first: CLKOUT0 is the MMCM's fractional output (1000 MHz / 7.5 at 125 MHz).
        pll.create_clkout(self.cd_ddr,    ddr_clk_freq, margin=1e-4)
        pll.create_clkout(self.cd_sys,    sys_clk_freq, margin=1e-4)
        pll.create_clkout(self.cd_idelay, 200e6,        margin=1e-4)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin) # Ignore sys_clk to pll.clkin path created by SoC's rst.
        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_ddr.clk)

        # IDELAYCTRL (replicated by Vivado in every bank with IDELAYs).
        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)

# TriXiumYPCBSoC -----------------------------------------------------------------------------------

# The (sdram, ddrphy, sdram_generator, sdram_checker) CSR pages of the two controllers: their
# own CSR bus in the ddr domain, behind a wishbone crossing at DDR_CSR_WINDOW, controller k's
# registers at controller 0's + k*DDR_CSR_STRIDE (liblitedram's SDRAM_CSR_STRIDE).
DDR_CSR_WINDOW = 0xF0020000
DDR_CSR_STRIDE = 0x2000

# The MMCM of each DDR3 PHY in its channel's clock region (channel 0: banks 11-13, the MMCM of
# bank 13's region; channel 1: banks 16-18, bank 17's, the command bank).
DDR_MMCM_LOCS = ["MMCME2_ADV_X0Y2", "MMCME2_ADV_X0Y6"]

class TriXiumYPCBSoC(SoCCore):
    def __init__(self, sys_clk_freq=125e6, ddr_clk_freq=400e6/3, engine_channels=8,
        main_ram_size  = 0x1000_0000,
        ddr_ibuf_hp    = True,
        ddr_ecc        = True,
        chan_stride    = None,
        chan_skew      = None,
        read_time      = 256,
        write_time     = 128,
        refresh_postponing = 1,
        trit5          = True,
        prefetch       = 2048,
        vec_fused      = True,
        hybrid         = False,
        attn_geometry  = (8, 2, 256, 2048),
        **kwargs):
        ddram_channels = 2
        assert engine_channels % ddram_channels == 0, \
            "engine channels shard evenly over the controllers"
        chan_mod = engine_channels//ddram_channels
        assert chan_mod & (chan_mod - 1) == 0, "channels per controller must be a power of two"
        platform = ypcb_00338_1p1.Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq, ddr_clk_freq=ddr_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        if kwargs.get("uart_name", "serial") == "serial":
            kwargs["uart_name"] = "crossover" # Console over PCIe/JTAGBone.
        kwargs["with_jtagbone"] = True
        # The BIOS's CK phase leveling keeps its scan maps in SRAM (~3.5 KB of .bss).
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
            ddr_controller_settings(read_time, write_time, refresh_postponing), ddr_ibuf_hp, ddr_ecc)
        set_main_ram_executable(self)

        # TriXium Memory Ports ---------------------------------------------------------------------
        # Engine channel c -> controller c // chan_mod: a 512-bit crossbar port behind an AXI
        # bridge, crossed into sys at the AXI level (memory/ddr.py).
        memory_ports = []
        for c in range(engine_channels):
            axi = AXIInterface(data_width=256, address_width=32, id_width=6)
            for j, m in enumerate(engine_ddr_port(self.sdrams[c//chan_mod], axi, clock_domain="sys_port",
                                                  rmw=ENGINE_RMW)):
                setattr(self, f"engport{c}_{j}", m)
            memory_ports.append(axi)

        # PCIe -------------------------------------------------------------------------------------
        self.pcie_phy = S7PCIEPHY(platform, platform.request("pcie_x8"),
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
        # (c % chan_mod)*(chan_stride + chan_skew), the controller above the weight base split
        # evenly (ddr_chan_stride: 864 MB per window at 4 channels, 432 MB at 8; the STORE and
        # upload paths add a constant per window, so no power of two). LiteDRAM's ROW_BANK_COL
        # mapping on a 64-bit DDR3 channel (1024 columns, 8 KB rows) puts the bank in byte
        # address bits 13..15: a stream moves to the next bank every 8 KB and round the 8 banks
        # every 64 KB. A controller's windows stream the same relative addresses at the same
        # time; the skew (64 KB / windows, a 64 KB-multiple stride) spreads them over the banks,
        # so the crossbar ports never wait on each other's bank (nor thrash one bank's row). The
        # last window loses (chan_mod - 1)*chan_skew (the converter's --window-reserve).
        # main_ram is controller 0's 0..256 MB, below every window; the BONS staging area is the
        # MB below channel 0's window.
        ctrl_size = 2 << 30
        if chan_stride is None:
            chan_stride = ddr_chan_stride(ctrl_size, chan_mod)
        if chan_skew is None:
            chan_skew = 0x10000//chan_mod # The 8 banks' 64 KB cycle, spread over the windows.
        chan_window = chan_stride - (chan_mod - 1)*chan_skew # The smallest (last) window.
        assert chan_stride % 0x10000 == 0
        assert HBM_WEIGHT_BASE + chan_mod*chan_stride <= ctrl_size
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride // 64
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000

        # Core: engine, sequencer and vector unit (the Qwen3 path needs no scratchpad unit).
        # --hybrid: also the hybrid models' units (Ternary-Bonsai-2 2B/4B/9B, which fit the 4 GB):
        # every scratchpad unit (sized by YPCB_HYBRID_UNIT_OPTIONS), the DeltaNet stream, and the
        # attention stream at the 2B's geometry (8 query heads, 2 kv heads, head_dim 256; the
        # 4B's 16/4 needs 16 lanes, 33k LUTs more; context up to attn_geometry[3] positions).
        units = {}
        if hybrid:
            units = dict(
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
        if hybrid:
            units.update(YPCB_HYBRID_UNIT_OPTIONS)
        config = TriXiumConfig(scratchpad_xpm=True, **units,
            chan_stride = chan_stride,
            chan_mod    = chan_mod,
            chan_skew   = chan_skew,
            with_trit5  = trit5, # Ternary codes 5 trits per byte (convert --trit5 images).
            # Weight prefetch (engine op 7): a matvec's reads issued under the serial work
            # before it, into 64 KB reader FIFOs per channel (block RAM).
            with_prefetch     = bool(prefetch),
            reader_fifo_depth = prefetch or 512,
            # Fused vector instructions (the clamp/round after an ALU op in one instruction).
            with_vec_fused    = vec_fused)
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
        self.add_constant("TRIXIUM_CHAN_WINDOW",      chan_window)
        self.add_constant("TRIXIUM_N_CHANNELS",       engine_channels)
        self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
        self.add_constant("TRIXIUM_CHAN_SKEW",        chan_skew)
        self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
        self.add_constant("TRIXIUM_N_MEMCTRL",        ddram_channels)
        self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   DDR_CSR_STRIDE)

    def add_ddr_controllers(self, platform, n, ddr_clk_freq, main_ram_size, l2_size,
                            controller_settings, ddr_ibuf_hp=True, ddr_ecc=True):
        """n K7DDRPHY (HR banks, CK phase shift) + MT41K256M8 controllers in the `ddr` domain
        (memory/ddr.py), each with a LiteDRAM BIST for the CK phase leveling's traffic test; the
        BIOS levels them all (SDRAM_CSR_STRIDE, SDRAM_CONTROLLERS)."""
        phys, modules = [], []
        for i in range(n):
            phy = K7DDRPHY(
                # ECC: all nine byte lanes (SECDED on the ninth); without, the ECC lane unused.
                pads                = PHYPadsReducer(platform.request("ddram", i), list(range(9 if ddr_ecc else 8))),
                memtype             = "DDR3",
                sys_clk_freq        = ddr_clk_freq,
                cl                  = ddr3_latencies(ddr_clk_freq)[0],
                cwl                 = ddr3_latencies(ddr_clk_freq)[1],
                with_odelay         = False, # HR banks.
                with_ck_phase_shift = True,
                name                = f"ddrphy{i}",
                mmcm_loc            = DDR_MMCM_LOCS[i])
            phy.add_constraints(platform)
            if ddr_ibuf_hp:
                # Full-performance input buffers on DQ/DQS (Xilinx's MIG sets the same; the
                # low-power default narrows the read eye under sustained reads).
                platform.add_platform_command(f"set_property IBUF_LOW_PWR FALSE [get_ports ddram{i}_dq*]")
            phys.append(phy)
            modules.append(YPCBDDR3(ddr_clk_freq, "1:4"))
        add_ddr_domain_controllers(self, phys, modules, ddr_clk_freq, main_ram_size, l2_size,
            DDR_CSR_WINDOW, phy_renames={"sys": "ddr"}, controller_settings=controller_settings,
            with_bist=True, with_ecc=ddr_ecc)
        self.add_constant("SDRAM_CSR_STRIDE",  DDR_CSR_STRIDE)
        self.add_constant("SDRAM_CONTROLLERS", n)

# Build --------------------------------------------------------------------------------------------

# CPU: the FK33's (the Qwen3 path needs no FPU): 32 KB I$ + 32 KB D$, branch prediction, Zcbm
# (cache maintenance) and Zbb (clz).
VEXII_ARGS = ("--fetch-l1-ways=4 --fetch-l1-sets=128 --lsu-l1-ways=4 --lsu-l1-sets=128 "
    "--with-btb --with-ras --with-gshare --with-rvZcbm --with-rvZbb")
VEXII_ARGS_HYBRID = VEXII_ARGS + " --with-rvf" # The hybrid forward's CPU share is FP32.

def main():
    # VexiiRiscv's netlist cache key must not depend on set ordering: hash randomization is
    # fixed at interpreter start, so run again with PYTHONHASHSEED=0.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from litex.build.parser import LiteXArgumentParser
    from software.lab.timing import gate
    parser = LiteXArgumentParser(platform=ypcb_00338_1p1.Platform, description="TriXium on the Inspur YPCB-00338-1P1.")
    parser.add_target_argument("--sys-clk-freq",       default=125e6,    type=float,                          help="System clock frequency (CPU, TriXium core).")
    parser.add_target_argument("--ddr-clk-freq",       default=400e6/3,  type=float,                          help="DDR3 controller clock frequency (DDR3 rate = 8x: 133.33e6 = DDR3-1066).")
    parser.add_target_argument("--engine-channels",    default=8,        type=int,                            help="Engine channels (multiple of 2).")
    parser.add_target_argument("--read-time",          default=256,      type=int,                            help="LiteDRAM read time (cycles before a pending write wins).")
    parser.add_target_argument("--write-time",         default=128,      type=int,                            help="LiteDRAM write time.")
    parser.add_target_argument("--refresh-postponing", default=1,        type=int,                            help="LiteDRAM refresh postponing (1-8).")
    parser.add_target_argument("--chan-stride",        default=None,     type=lambda x: int(x, 0),            help="Engine window stride (default: each controller above the weight base split evenly).")
    parser.add_target_argument("--trit5",              action=argparse.BooleanOptionalAction, default=True,  help="Unpackers for 5-trits-per-byte weights (the default; --no-trit5: 2-bit images, convert --board ypcb --no-trit5).")
    parser.add_target_argument("--prefetch",           default=2048,     type=int,                            help="Weight prefetch: reader FIFO beats per channel (0: no prefetch).")
    parser.add_target_argument("--vec-fused",          action=argparse.BooleanOptionalAction, default=True,  help="Vector unit with the fused clamp/round instructions.")
    parser.add_target_argument("--ddr-ecc",            action=argparse.BooleanOptionalAction, default=True,  help="DDR3 ECC: SECDED on each channel's ninth byte lane (the default; corrects the rare read errors under full streaming).")
    parser.add_target_argument("--ddr-ibuf-low-power", action="store_true",                                   help="DQ/DQS input buffers in low-power mode (Vivado's default; not for sustained reads).")
    parser.add_target_argument("--hybrid",             action="store_true",                                   help="With the hybrid models' units (Ternary-Bonsai-2 2B/4B/9B: DeltaNet/attention streams, FWHT, ...) and a CPU FPU.")
    parser.add_target_argument("--driver",             action="store_true",                                   help="Generate PCIe driver (always with --build).")
    parser.add_target_argument("--vivado-settings",    default=os.environ.get("LITEX_ENV_VIVADO", VIVADO_PATH), help="Vivado install sourced by the build (settings64.sh).")
    # The shipping recipe (overridable from the command line).
    parser.set_defaults(
        cpu_type                             = "vexiiriscv",
        update_repo                          = "no",
        # LiteX applies the CPU arguments inside parse_args: pick the FPU variant up front.
        vexii_args                           = VEXII_ARGS_HYBRID if "--hybrid" in sys.argv else VEXII_ARGS,
        vivado_synth_directive               = "PerformanceOptimized",
        vivado_opt_directive                 = "ExploreWithRemap",
        vivado_place_directive               = "ExtraTimingOpt",
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
        trit5              = args.trit5,
        prefetch           = args.prefetch,
        vec_fused          = args.vec_fused,
        chan_stride        = args.chan_stride,
        ddr_ibuf_hp        = not args.ddr_ibuf_low_power,
        ddr_ecc            = args.ddr_ecc,
        hybrid             = args.hybrid,
        **parser.soc_argdict)
    add_zbb_constant(soc, args)

    builder_argd = parser.builder_argdict
    if builder_argd.get("output_dir") is None:
        builder_argd["output_dir"] = os.path.join("build", "ypcb")
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

        # A successful Vivado run is not a qualified bitstream: the timing gate decides
        # (strict: this board has no build that has run on a card yet).
        if builder.compile_gateware:
            report = os.path.join(builder.gateware_dir, f"{soc.platform.name}_timing.rpt")
            print("\nTiming gate:")
            rc = gate(report, strict=True, board="ypcb", clocks=[round(1e9/args.sys_clk_freq, 3)])
            write_verdict(verdict, rc)
            if rc:
                sys.exit(rc)

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

if __name__ == "__main__":
    main()

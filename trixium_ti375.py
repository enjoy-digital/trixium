#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium on the Efinix Ti375 C529 development kit (small boards, milestone M5).

The memory system is the simulator's `--board ti375` preset on the real part: the hard LPDDR4x
controller's single 512-bit AXI4 port (1 GB) carries the CPU's main_ram and every engine
channel's window, through AXISharedPort (reads interleaved per request, each master's write
bursts buffered before their AW, per-master read buffers). The engine's 256-bit channel ports
merge two beats per word (AXIWide). No PCIe: the console on the kit's UART, the host link over
Ethernet (RGMII): Etherbone for the CSRs and UDPUpload for the weights (software/host/udpload.py).

First version: everything in sys (100 MHz, the hard controller's AXI included), the units at the
lite profile, 4 channels. Efinity 2025.1: LITEX_ENV_EFINITY=/opt/efinix/efinity/2025.1.

  ./trixium_ti375.py --build
"""

import os
import argparse

from migen import *

from litex.gen import *

from litex_boards.platforms import efinix_ti375_c529_dev_kit

from litex.soc.cores.clock.efinix import TITANIUMPLL
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder
from litex.soc.interconnect.axi import AXIInterface
from litex.soc.cores.ram.efinix_ddr import EfinixDDR

from model.layout import HBM_WEIGHT_BASE, HDR_CPU_OFFSET, ddr_chan_stride
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.axi import AXIWide
from gateware.memory.axi_shared import AXISharedPort, AXIIDSquash, slave_id_width
from gateware.memory.udp_upload import UDPUpload

from trixium_ypcb import YPCB_HYBRID_UNIT_OPTIONS, YPCB_LITE_UNIT_OPTIONS, YPCB_LITE_ATTN, VEXII_ARGS

UPLOAD_PORT = 2345   # software/host/udpload.py

# CRG ----------------------------------------------------------------------------------------------

class _CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq):
        self.rst    = Signal()
        self.cd_sys = ClockDomain()
        self.cd_rst = ClockDomain(reset_less=True)

        # # #

        clk100 = platform.request("clk100")
        rst_n  = platform.request("user_btn_n", 0)
        self.comb += self.cd_rst.clk.eq(clk100)
        self.rst_pulse = Signal()
        last_rst = Signal()
        self.sync.rst += [last_rst.eq(self.rst), self.rst_pulse.eq(~last_rst & self.rst)]

        self.pll = pll = TITANIUMPLL(platform)
        self.comb += pll.reset.eq(~rst_n | self.rst_pulse)
        pll.register_clkin(clk100, platform.default_clk_freq)
        pll.create_clkout(self.cd_sys, sys_clk_freq, with_reset=True)
        # The LPDDR4x controller takes its PHY clock from PLL0's output 3 (the platform's
        # ddr_config clkin_sel): outputs 1 and 2 are placeholders, as in LiteX-Boards' target.
        pll.create_clkout(None, 100e6)
        pll.create_clkout(None, 60e6, margin=0)
        pll.create_clkout(None, 800e6)

# TriXium SoC --------------------------------------------------------------------------------------

class TriXiumTi375SoC(SoCCore):
    def __init__(self, sys_clk_freq=100e6, engine_channels=4, profile="lite", prefetch=512,
                 eth_ip="<board-host>", attn_geometry=YPCB_LITE_ATTN + (256, 1024), dn_sublanes=0,
                 **kwargs):
        platform = efinix_ti375_c529_dev_kit.Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(platform, sys_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        kwargs.setdefault("cpu_type", "vexiiriscv")
        kwargs["integrated_sram_size"] = max(kwargs.get("integrated_sram_size") or 0, 0x4000)
        kwargs["integrated_main_ram_size"] = 0
        SoCCore.__init__(self, platform, sys_clk_freq, ident="TriXium on Efinix Ti375 C529", **kwargs)
        self.add_constant("TRIXIUM_VARIANT", ord("t"))

        # Ethernet: Etherbone (the CSRs) and the weight upload port ---------------------------------
        from liteeth.phy.titaniumrgmii import LiteEthPHYRGMII
        self.ethphy = LiteEthPHYRGMII(platform=platform, clock_pads=platform.request("eth_clocks"),
                                      pads=platform.request("eth"), with_hw_init_reset=False)
        platform.add_false_path_constraints(self.crg.cd_sys.clk,
            self.ethphy.crg.cd_eth_rx.clk, self.ethphy.crg.cd_eth_tx.clk)
        self.add_etherbone(phy=self.ethphy, ip_address=eth_ip, with_timing_constraints=False)
        udp_port = self.ethcore_etherbone.udp.crossbar.get_port(UPLOAD_PORT, dw=32, cd="sys")
        self.udpload = UDPUpload(udp_port, max_beats=32, out_dw=128)

        # Memory: the hard controller's port, shared --------------------------------------------
        self.ddr = ddr = EfinixDDR(platform=platform, clock_domain="sys", **platform.ddr_config)
        ctrl_size  = platform.ddr_size
        chan_mod   = engine_channels
        chan_stride = ddr_chan_stride(ctrl_size, chan_mod)
        main_ram_size = 0x1000_0000
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000
        assert HBM_WEIGHT_BASE + chan_mod*chan_stride <= ctrl_size

        masters, memory_ports = [], []
        for c in range(engine_channels):
            axi  = AXIInterface(data_width=256, address_width=32, id_width=4)
            wide = AXIInterface(data_width=512, address_width=32, id_width=4)
            self.add_module(f"engwide{c}", AXIWide(axi, wide))
            memory_ports.append(axi)
            masters.append(wide)
        base = self.mem_map["main_ram"]
        self.bus.add_region("main_ram", SoCRegion(origin=base, size=main_ram_size, mode="rwx"))
        if hasattr(self.cpu, "add_memory_buses"):
            self.cpu.add_memory_buses(address_width=32, data_width=512)
        for i, mem_bus in enumerate(self.cpu.memory_buses):
            # Its IDs squashed to one (restored in order on the way back): the controller's 8 ID
            # bits must also hold the shared port's master index.
            sq = AXIInterface(data_width=512, address_width=32, id_width=1)
            self.add_module(f"cpusquash{i}", AXIIDSquash(mem_bus, sq))
            mem_bus = sq
            far = AXIInterface(data_width=512, address_width=32, id_width=mem_bus.id_width)
            for ch in ("aw", "w", "ar"):
                self.comb += getattr(mem_bus, ch).connect(getattr(far, ch), omit={"addr"})
            for ch in ("b", "r"):
                self.comb += getattr(far, ch).connect(getattr(mem_bus, ch))
            self.comb += [far.ar.addr.eq(mem_bus.ar.addr - base), far.aw.addr.eq(mem_bus.aw.addr - base)]
            masters.append(far)
        idw = slave_id_width(masters)
        assert idw <= ddr.bus.id_width, f"{idw} ID bits for the port, the controller has {ddr.bus.id_width}"
        shared = AXIInterface(data_width=512, address_width=32, id_width=idw)
        self.port = AXISharedPort(masters, shared, r_buffer=96)
        for ch in ("aw", "w", "ar"):
            self.comb += getattr(shared, ch).connect(getattr(ddr.bus, ch), omit={"id", "addr"})
            self.comb += [getattr(ddr.bus, ch).id.eq(getattr(shared, ch).id)] if ch != "w" else []
        self.comb += [ddr.bus.ar.addr.eq(shared.ar.addr), ddr.bus.aw.addr.eq(shared.aw.addr)]
        for ch in ("b", "r"):
            self.comb += getattr(ddr.bus, ch).connect(getattr(shared, ch), omit={"id"})
            self.comb += getattr(shared, ch).id.eq(getattr(ddr.bus, ch).id[:idw])
        self.comb += ddr.awallstrb.eq(0)

        # TriXium ----------------------------------------------------------------------------------
        units = dict(with_axpy=False, with_hadamard=True, with_activation_quantize=True,
                     with_projection_restore=True, with_silu=True, with_norm=True, with_conv=True,
                     with_softmax=True, with_dnstream=True, with_attnstream=True,
                     attnstream_options=tuple(zip(("lanes", "groups", "hb_max", "t_max"),
                        (attn_geometry[0], attn_geometry[1], attn_geometry[2]//8, attn_geometry[3]))))
        units.update(YPCB_HYBRID_UNIT_OPTIONS)
        if profile == "lite":
            units.update(YPCB_LITE_UNIT_OPTIONS, with_softmax=False)
        # The attention stream's operand memories in block RAM (Efinix has no distributed RAM: an
        # asynchronous read became 131k flip-flops).
        units["attnstream_options"] = units["attnstream_options"] + (("sync_operand", True),)
        if dn_sublanes:   # lanes (= channels) x sub-lanes: the DeltaNet stream's width.
            units["dnstream_options"] = (("sublanes", dn_sublanes),)
        config = TriXiumConfig(scratchpad_xpm=False, efinix_ram=True, **units,
            chan_stride       = chan_stride,
            chan_mod          = chan_mod,
            chan_skew         = 0,
            with_trit5        = True,
            with_prefetch     = bool(prefetch),
            reader_fifo_depth = prefetch or 512)
        self.trixium = TriXiumCore(memory_ports, config)
        for name, value in self.trixium.constants.items():
            self.add_constant(name, value)
        self.bus.add_slave("scratchpad", self.trixium.bus,
            SoCRegion(origin=0xa000_0000, size=self.trixium.scratchpad.size, cached=False))
        self.comb += self.udpload.source.connect(self.trixium.upload, keep={"valid", "ready", "data"})

        # Board memory map, for the firmware (the YPCB's, one controller, no LiteDRAM).
        self.add_constant("TRIXIUM_LOADER_ADDS_CHAN_OFFSET", 1)
        self.add_constant("TRIXIUM_HDR_ADDR",         0x4000_0000 + HDR_CPU_OFFSET)
        self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", HBM_WEIGHT_BASE)
        self.add_constant("TRIXIUM_CHAN_STRIDE",      chan_stride)
        self.add_constant("TRIXIUM_CHAN_WINDOW",      chan_stride)
        self.add_constant("TRIXIUM_N_CHANNELS",       engine_channels)
        self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
        self.add_constant("TRIXIUM_CHAN_SKEW",        0)
        self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)

# Build --------------------------------------------------------------------------------------------

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=efinix_ti375_c529_dev_kit.Platform,
                                 description="TriXium on the Efinix Ti375 C529 dev kit.")
    parser.add_target_argument("--sys-clk-freq",    default=100e6, type=float, help="System clock frequency.")
    parser.add_target_argument("--engine-channels", default=4,     type=int,   help="Engine channels on the shared port.")
    parser.add_target_argument("--profile",         default="lite", choices=["full", "lite"], help="The units' sizing.")
    parser.add_target_argument("--prefetch",        default=256,   type=int,   help="Reader FIFO beats (0: no prefetch).")
    parser.add_target_argument("--dn-sublanes",     default=1,     type=int,   help="DeltaNet stream sub-lanes per channel (1 at 4 channels: width 4).")
    parser.add_target_argument("--attn-t-max",      default=1024,  type=int,   help="Attention stream context positions.")
    parser.add_target_argument("--eth-ip",          default="<board-host>",    help="Etherbone / upload IP address.")
    # LiteX applies the CPU arguments inside parse_args (with the variant's own: mul/div, which the
    # FPU needs); never touch the shared VexiiRiscv checkout.
    parser.set_defaults(cpu_type="vexiiriscv", update_repo="no", vexii_args=VEXII_ARGS + " --with-rvf")
    args = parser.parse_args()
    os.environ.setdefault("LITEX_ENV_EFINITY", "/opt/efinix/efinity/2025.1")
    soc = TriXiumTi375SoC(sys_clk_freq=args.sys_clk_freq, engine_channels=args.engine_channels,
                          profile=args.profile, prefetch=args.prefetch, eth_ip=args.eth_ip,
                          dn_sublanes=args.dn_sublanes, attn_geometry=YPCB_LITE_ATTN + (256, args.attn_t_max),
                          **parser.soc_argdict)
    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        builder.build(**parser.toolchain_argdict)


if __name__ == "__main__":
    main()

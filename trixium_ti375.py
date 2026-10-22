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

Everything in sys (100 MHz, the hard controller's AXI included), the units at the lite profile,
4 channels. One 512-bit port at 100 MHz is 6.4 GB/s, half the LPDDR4x's rate, and the weight
stream runs at ~92% of it: --ddr-targets 2 also enables the controller's second AXI target and
splits the channels between the two (channels 0-1 and the CPU on target 0, 2-3 on target 1;
needs LiteX's EfinixDDR with `targets`, litex#2688). Efinity 2025.1: LITEX_ENV_EFINITY=/opt/efinix/efinity/2025.1.

  ./trixium_ti375.py --build [--ddr-targets 2]
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
from gateware.common import CPUBusMonitor
from litex.soc.interconnect.csr import CSRStatus

from trixium_ypcb import YPCB_HYBRID_UNIT_OPTIONS, YPCB_LITE_UNIT_OPTIONS, YPCB_LITE_ATTN, VEXII_ARGS

UPLOAD_PORT = 2345   # software/host/udpload.py
# The firmware's place in the kit's configuration flash (IS25WP512M, 64 MB): above the bitstream
# (~7 MB), as LiteX's flashboot image (length + CRC + the binary: crcfbigen).
FIRMWARE_FLASH_OFFSET = 0x0200_0000

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
                 ddr_targets=1, with_spi_flash=False, upload_beats=32, chan_skew=0, no_core=False, **kwargs):
        # no_core: the SoC without the TriXium core (the clocks, the CPU on the shared memory port,
        # Ethernet): a ~10-minute build to bring the board and the memory attachment up alone.
        idle_channels = 0
        if no_core:
            # The engine's ports kept as idle masters: the shared port's IDs and arbitration as in
            # the full design.
            idle_channels, engine_channels = engine_channels, 0
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
        # The UDP/IP core at 32 bits in sys (an 8-bit one at 100 MHz is slower than the 1 Gb/s line);
        # the upload port in the SoC's sys clock domain (cd="etherbone").
        self.add_etherbone(phy=self.ethphy, ip_address=eth_ip, data_width=32, with_timing_constraints=False)
        udp_port = self.ethcore_etherbone.udp.crossbar.get_port(UPLOAD_PORT, dw=32, cd="etherbone")
        # upload_beats: the largest datagram (45 beats = 1,440 bytes + the sequence word fit a
        # 1,500-byte MTU's UDP payload; 32 before).
        self.udpload = UDPUpload(udp_port, max_beats=upload_beats, out_dw=128)

        # The configuration flash: the BIOS boots the firmware from it (flashboot) when no host
        # answers its serial boot first - power up and the firmware runs (software/host/ti375.py
        # flash writes bitstream + firmware). The SDR PHY (1:1) at 25 MHz: at 50 MHz DQ3 is sampled
        # late (~0.6% of the words lose a bit, measured on the kit; 33 MHz clean): the firmware's
        # 226 KB copy still takes ~20 ms.
        if with_spi_flash:
            from litespi.modules import IS25WP512M
            from litespi.opcodes import SpiNorFlashOpCodes as Codes
            self.add_spi_flash(mode="4x", clk_freq=25e6, number=0,
                               module=IS25WP512M(Codes.READ_1_1_4_4B), with_master=True, rate="1:1")
            self.add_constant("FLASH_BOOT_ADDRESS",
                              self.bus.regions["spiflash"].origin + FIRMWARE_FLASH_OFFSET)
            # The BIOS's clock calibration reads one block at faster divisors and keeps the fastest
            # whose CRC matches: the sporadic DQ3 errors pass it at 50 MHz. Keep the 25 MHz.
            self.add_constant("SPIFLASH_SKIP_FREQ_INIT")

        # Memory: the hard controller's port, shared --------------------------------------------
        # EfinixDDR's own AXI ID width (the controller's 6 bits): litex-boards' ddr_config pinned 8
        # before litex-boards#931 (merged); the key is dropped for older checkouts.
        ddr_config = {k: v for k, v in platform.ddr_config.items() if k != "id_width"}
        # ddr_targets=2: the controller's second AXI target too (bus1), the channels split between
        # the two (the first half with the CPU on target 0): two 512-bit ports at sys, 12.8 GB/s,
        # instead of one port at half the LPDDR4x's rate.
        self.ddr = ddr = EfinixDDR(platform=platform, clock_domain="sys", targets=ddr_targets, **ddr_config)
        ctrl_size  = platform.ddr_size
        chan_mod   = engine_channels or 4
        chan_stride = ddr_chan_stride(ctrl_size, chan_mod)
        # chan_skew: channel k's window starts k*skew further, so the channels' simultaneous
        # streams fall in different DRAM banks (the YPCB's: the bank cycle / channels); the last
        # window loses (chan_mod - 1)*skew.
        assert chan_skew % 0x1000 == 0 and (chan_mod - 1)*chan_skew < chan_stride//64
        chan_window = chan_stride - (chan_mod - 1)*chan_skew
        main_ram_size = 0x1000_0000
        assert main_ram_size <= HBM_WEIGHT_BASE - 0x100000
        assert HBM_WEIGHT_BASE + chan_mod*chan_stride <= ctrl_size

        masters, memory_ports = [[] for t in range(ddr_targets)], []
        n_ports = engine_channels + idle_channels
        target_of = lambda c: c*ddr_targets//n_ports
        for c in range(engine_channels):
            # 3 ID bits (the readers round-robin 4 IDs): with the master index, the port's IDs fit
            # the controller's 6 ID pins.
            axi  = AXIInterface(data_width=256, address_width=32, id_width=3)
            wide = AXIInterface(data_width=512, address_width=32, id_width=3)
            self.add_module(f"engwide{c}", AXIWide(axi, wide))
            memory_ports.append(axi)
            masters[target_of(c)].append(wide)
        for c in range(idle_channels):
            masters[target_of(c)].append(AXIInterface(data_width=512, address_width=32, id_width=3))
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
            masters[0].append(far)
        for t, ms in enumerate(masters):
            bus = ddr.bus1 if t else ddr.bus
            idw = slave_id_width(ms)
            # The controller's AXI IDs are 6 bits (ARID/AWID/RID/BID[5:0]): the masters' index must fit.
            assert idw <= len(bus.ar.id), f"{idw} ID bits for the port, the controller has {len(bus.ar.id)}"
            shared = AXIInterface(data_width=512, address_width=32, id_width=idw)
            port = AXISharedPort(ms, shared, r_buffer=96)
            setattr(self, f"port{t}" if t else "port", port)
            for ch in ("aw", "w", "ar"):
                self.comb += getattr(shared, ch).connect(getattr(bus, ch), omit={"id", "addr"})
                self.comb += [getattr(bus, ch).id.eq(getattr(shared, ch).id)] if ch != "w" else []
            self.comb += [bus.ar.addr.eq(shared.ar.addr), bus.aw.addr.eq(shared.aw.addr)]
            for ch in ("b", "r"):
                self.comb += getattr(bus, ch).connect(getattr(shared, ch), omit={"id"})
                self.comb += getattr(shared, ch).id.eq(getattr(bus, ch).id[:idw])
        self.comb += ddr.awallstrb.eq(0)

        # Bring-up visibility over Etherbone (the CPU has no debug port here): where the CPU is
        # (bus handshake counts and last addresses, software/host/cpumon.py's CSRs) and the hard
        # controller's initialisation and port state.
        self.cpumon = CPUBusMonitor(self.cpu)
        self.ddrstat = LiteXModule()
        self.ddrstat.init_done = CSRStatus(1, description="The LPDDR4x controller finished its initialisation.")
        self.ddrstat.port = CSRStatus(10, description="The controller AXI handshakes: ar/r/aw/w/b valid and ready.")
        self.comb += [
            self.ddrstat.init_done.status.eq(ddr.init_done),
            self.ddrstat.port.status.eq(Cat(ddr.bus.ar.valid, ddr.bus.ar.ready, ddr.bus.r.valid, ddr.bus.r.ready,
                ddr.bus.aw.valid, ddr.bus.aw.ready, ddr.bus.w.valid, ddr.bus.w.ready, ddr.bus.b.valid, ddr.bus.b.ready)),
        ]

        if no_core:
            return
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
            chan_skew         = chan_skew,
            with_trit5        = True,
            with_prefetch     = bool(prefetch),
            reader_fifo_depth = prefetch or 512,
            # One ARID per reader: AXIWide's read merge needs the R order to be the AR order, and
            # the hard controller may reorder (or interleave) bursts across IDs.
            reader_n_ids      = 1)
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
        self.add_constant("TRIXIUM_CHAN_WINDOW",      chan_window)
        self.add_constant("TRIXIUM_N_CHANNELS",       engine_channels)
        self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
        self.add_constant("TRIXIUM_CHAN_SKEW",        chan_skew)
        self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
        # The memory port's bytes per sys clock (smi's peak: the channels outrun the port here).
        self.add_constant("TRIXIUM_PORT_BYTES",       64*ddr_targets)

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
    parser.add_target_argument("--ddr-targets",     default=1,     type=int, choices=[1, 2], help="The DDR controller's AXI targets the channels use (2: twice the port bandwidth).")
    parser.add_target_argument("--with-spi-flash",  action="store_true",       help="The configuration flash on the bus: the BIOS boots the firmware from it.")
    parser.add_target_argument("--upload-beats",    default=32,    type=int,   help="UDP upload datagram payload in 256-bit beats (45 fills a 1,500 MTU).")
    parser.add_target_argument("--chan-skew",       default=0,     type=lambda x: int(x, 0), help="Per-channel window offset in bytes (DRAM bank spreading; 4 KB multiple).")
    parser.add_target_argument("--no-core",         action="store_true",       help="Without the TriXium core (board bring-up).")
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
                          ddr_targets=args.ddr_targets, with_spi_flash=args.with_spi_flash,
                          upload_beats=args.upload_beats, chan_skew=args.chan_skew, no_core=args.no_core,
                          **parser.soc_argdict)
    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        builder.build(**parser.toolchain_argdict)


if __name__ == "__main__":
    main()

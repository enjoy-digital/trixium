#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium Verilator simulation SoC.

VexRiscv + engine (N channels backed by BonsaiHBMChannelModels initialized from a BONS image)
+ scratchpad + engine CSRs - the same firmware-visible contract as the hardware target. The
firmware binary and the BONS header+table are baked into main_ram at elaboration; channel
payloads into the HBM models (CHAN_WEIGHT_BASE = 0 in sim).

Flow (from the repo root):
  python3 sim/bonsai_sim.py --build-software          # headers + BIOS (no gateware)
  make -C firmware BUILD_DIR=../build/sim             # firmware/bonsai.bin
  python3 sim/bonsai_sim.py --run                     # elaborate + verilate + run
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *
from litex.gen.genlib.misc import *

from litex.build.io import CRG
from litex.build.sim.config import SimConfig
from litex.tools.litex_sim import Platform

from litex.soc.integration.common import *
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder

from litex.soc.interconnect.axi import AXIInterface

from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes
from trixium.model.bons import BONSReader

# Constants ----------------------------------------------------------------------------------------

MAIN_RAM_BASE = 0x4000_0000
MAIN_RAM_SIZE = 0x0080_0000 # 8 MB.
from trixium.layout import HDR_CPU_OFFSET
HDR_OFFSET    = HDR_CPU_OFFSET # BONS header+table at main_ram + 4 MB (shared with hardware).

# BonsaiSimSoC -------------------------------------------------------------------------------------

# The DDR4 flavour of the sim (--mem ddr4): the XCU1525's memory plumbing on LiteDRAM's PHY
# model - main_ram on controller 0 through the BIOS's own sdram_init(), the engine channels
# on 256-bit crossbar ports behind LiteDRAMAXI2Native bridges, chan_mod channels per
# controller at these window constants, and the firmware's ddr_init_all()/ddr_test() for
# controllers 1..N-1 (firmware/ddr.c) - all on the tiny model, tokens against the same
# reference as the HBM-model sim. Sizes: main_ram is 8 MB (0..8 MB of controller 0), the
# windows start at 12 MB so the header staging area (window - 1 MB) and the round-trip test
# (window - 2 MB) stay above main_ram; 2 MB per window holds the tiny payload and its KV.
DDR4_SIM_WEIGHT_BASE = 0x00C0_0000  # 12 MB.
DDR4_SIM_CHAN_STRIDE = 0x0020_0000  # 2 MB.
DDR4_SIM_CHAN_SKEW   = 0xA000       # 40 KB: a bank skew IN THE SIM'S GEOMETRY (ncols 128 puts the
                                    # bank bits at byte bits 10..13, so the board's 32 KB would be
                                    # a row bit here); the address arithmetic is what the gate checks.
DDR4_SIM_CSR_PAGE0   = 8            # (sdram, ddrphy) pairs on pages 8/9, 10/11, ... as the target.
DDR4_SIM_CSR_STRIDE  = 0x1000

class BonsaiSimSoC(SoCCore):
    def __init__(self, image=None, firmware=None, hbm_latency=16, hbm_stall_rate=0,
                 dual=True, cpu_type="vexriscv", icache_kb=0, vexii_extra="",
                 mem="hbm", n_memctrl=4, update_repo="no", with_memory_traffic=False,
                 with_capabilities=True, with_native=False, with_hadamard=False, with_activation_quantize=False,
                 with_projection_restore=False, with_embedding_restore=False, with_region_fetch=False,
                 with_silu=False, with_norm=False, with_conv=False, with_dnrec=False):
        assert mem in ("hbm", "ddr4")
        platform = Platform()
        self.comb += platform.trace.eq(1)

        # main_ram init: firmware at 0, BONS header+table at HDR_OFFSET.
        ram = bytearray(MAIN_RAM_SIZE)
        if firmware is not None and os.path.exists(firmware):
            fw = open(firmware, "rb").read()
            assert len(fw) < HDR_OFFSET
            ram[:len(fw)] = fw
        img = None
        if image is not None:
            img  = BONSReader(image)
            meta = bytes(img.data[:img.header["payload_offset"]])
            ram[HDR_OFFSET:HDR_OFFSET+len(meta)] = meta
        ram_words = [int.from_bytes(ram[i:i+4], "little") for i in range(0, len(ram), 4)]

        # VexiiRiscv is generated from SpinalHDL at build time, so unlike VexRiscv - whose
        # I-cache is a fixed 4 KB baked into every pregenerated variant - the cache is a
        # parameter. That matters here: forward() alone is 12 KB, so a 4 KB I-cache cannot
        # hold the hot loop, and the PC profiler measures ~49% of CPU cycles retiring nothing.
        # 64 sets x 4 ways x 64 B lines = 16 KB, which covers forward() with room to spare.
        if cpu_type == "vexiiriscv":
            import argparse as _ap
            from litex.soc.cores.cpu.vexiiriscv.core import VexiiRiscv
            # VexiiRiscv reads a pile of class attributes (soc_args, soc_keys, update_repo,
            # vexii_args, ...) that LiteX normally populates from its CLI. Constructing
            # SoCCore directly bypasses that, so drive the real API with a defaults-only
            # parser instead of hand-setting attributes - hand-setting works until the next
            # LiteX bump adds a key, and then fails as an AttributeError deep in codegen.
            _p = _ap.ArgumentParser()
            VexiiRiscv.args_fill(_p)
            _va = _p.parse_args([])
            _va.cpu_variant = "standard"
            _va.update_repo = update_repo
            # The extra args go through the real API: args_read runs the generator's
            # PythonArgsGen on them and the ISA map (hence the software's -march) follows,
            # so "--with-rvf" gives an FPU the firmware actually uses.
            _va.vexii_args  = vexii_extra
            VexiiRiscv.args_read(_va)
            # LiteX builds --with-isa from a set, whose process-random iteration order used
            # to give the same CPU a different netlist hash on successive runs and needlessly
            # invoke SBT again.  Canonicalise only that comma list; the option order remains
            # otherwise untouched.
            _canon = []
            for _arg in VexiiRiscv.vexii_args.split():
                if _arg.startswith("--with-isa="):
                    _arg = "--with-isa=" + ",".join(sorted(_arg.split("=", 1)[1].split(",")))
                _canon.append(_arg)
            VexiiRiscv.vexii_args = " ".join(_canon)
            # Now override just the fetch cache. 64 sets x 4 ways x 64 B = 16 KB, against a
            # 12 KB forward() that a 4 KB VexRiscv cache cannot hold.
            sets = max(16, (icache_kb*1024)//(4*64)) if icache_kb else 64
            VexiiRiscv.vexii_args = (VexiiRiscv.vexii_args
                                     .replace("--fetch-l1-ways=2", "--fetch-l1-ways=4")
                                     + f" --fetch-l1-sets={sets}")
            print(f"[bonsai] VexiiRiscv args: {VexiiRiscv.vexii_args}")

        SoCCore.__init__(self, platform,
            clk_freq                 = int(1e6),
            ident                    = "TriXium Sim" + (" DDR4" if mem == "ddr4" else ""),
            cpu_type                 = cpu_type,
            integrated_rom_size      = 0x20000,
            integrated_main_ram_size = MAIN_RAM_SIZE if mem == "hbm" else 0,
            integrated_main_ram_init = ram_words if mem == "hbm" else [],
            uart_name                = "sim")
        self.crg = CRG(platform.request("sys_clk"))

        # Profiler pad: unit busy bits as TOP-LEVEL sim ports (prof_eng_busy, prof_vec_busy,
        # prof_seq_busy on the Vsim). The C++ profiler traces their edges into an event file
        # for the Perfetto timeline. A pad, not __PVT__ names: Verilator's internal names are
        # not a stable contract (__PVT__vecunit_busy is already folded away in this build),
        # while requested pads are emitted verbatim.
        from litex.build.generic_platform import Subsignal, Pins
        platform.add_extension([("prof", 0,
            Subsignal("eng_busy", Pins(1)),
            Subsignal("vec_busy", Pins(1)),
            Subsignal("seq_busy", Pins(1)))])
        self._prof_pads = platform.request("prof")

        # Scratchpad + engine + the memory system.
        axis = []
        memory_sizes = []
        n_channels = img.header["n_channels"] if img is not None else 2
        chan_mod = 0
        if mem == "hbm":
            for c in range(n_channels):
                axi = AXIInterface(data_width=256, address_width=33, id_width=6)
                init = None
                size = 256*1024
                if img is not None:
                    payload = bytes(img.channel_payload(c))
                    # Reserve headroom above the weights for the KV region (kind-3 stores).
                    while size < len(payload) + 128*1024:
                        size *= 2
                    init = words_from_bytes(payload)
                model = BonsaiHBMChannelModel(axi, size=size, init=init,
                    latency=hbm_latency, stall_rate=hbm_stall_rate)
                self.add_module(f"hbm_model{c}", model)
                axis.append(axi)
                memory_sizes.append(size)
        else:
            chan_mod = self._add_ddr4(n_channels, n_memctrl, ram, img, axis)
        # Same integration as both hardware targets; only memory/physical policy differs.
        from trixium.gateware.integration import BonsaiClusterConfig, add_bonsai_cluster, native_read_options
        def memory_description():
            from trixium.capabilities import MemoryRegion, MemoryKind, Reservation, ReservationKind
            if mem == 'hbm':
                regions = [MemoryRegion(i, MemoryKind.SIMULATION, i, 0, size, locality=i)
                           for i, size in enumerate(memory_sizes)]
                # Integrated CPU RAM is distinct from the simulated accelerator memories.
                regions.append(MemoryRegion(0x80000000, MemoryKind.SIMULATION, 0x80000000, 0, MAIN_RAM_SIZE))
                reservations = [Reservation(0x80000000, ReservationKind.CPU, 0, MAIN_RAM_SIZE)]
            else:
                regions = [MemoryRegion(i, MemoryKind.SIMULATION, i, 0, 32 << 20, locality=i)
                           for i in range(n_memctrl)]
                reservations = [Reservation(0, ReservationKind.CPU, 0, MAIN_RAM_SIZE),
                                Reservation(0, ReservationKind.UPLOAD_STAGING,
                                            DDR4_SIM_WEIGHT_BASE - 0x100000, 0x100000)]
                test_base = DDR4_SIM_WEIGHT_BASE + DDR4_SIM_CHAN_STRIDE
                test_base -= DDR4_SIM_WEIGHT_BASE % DDR4_SIM_CHAN_STRIDE
                test_base -= (chan_mod - 1)*DDR4_SIM_CHAN_SKEW + 0x1000
                reservations += [Reservation(c//chan_mod, ReservationKind.DIAGNOSTIC,
                    test_base + (c % chan_mod)*(DDR4_SIM_CHAN_STRIDE + DDR4_SIM_CHAN_SKEW), 0x1000)
                    for c in range(n_channels)]
            return regions, reservations
        native = (native_read_options(*memory_description(),
                   list(range(n_channels)) if mem == 'hbm' else [c//chan_mod for c in range(n_channels)], write=True)
                  if with_native else {})
        config = BonsaiClusterConfig(
            with_vecunit=True, with_dual=dual, with_sp_csr=with_native, append_sp_csr=with_native,
            chan_stride=DDR4_SIM_CHAN_STRIDE if mem == "ddr4" else 0,
            chan_mod=chan_mod, chan_skew=DDR4_SIM_CHAN_SKEW if mem == "ddr4" else 0,
            with_hadamard=with_hadamard, with_activation_quantize=with_activation_quantize,
            with_projection_restore=with_projection_restore, with_embedding_restore=with_embedding_restore,
            with_region_fetch=with_region_fetch, with_silu=with_silu,
            with_norm=with_norm, with_conv=with_conv, with_dnrec=with_dnrec, **native)
        add_bonsai_cluster(self, axis, config)
        if with_memory_traffic:
            from trixium.gateware.traffic import add_memory_traffic
            controllers = [getattr(self, 'sdram' + (str(i) if i else '')).controller.interface
                           for i in range(n_memctrl)] if mem == 'ddr4' else []
            add_memory_traffic(self, axis, controllers)
        if with_capabilities:
            from trixium.gateware.capabilities import add_capabilities
            regions, reservations = memory_description()
            self.bonsai_capabilities = add_capabilities(self, config, axis, regions, reservations,
                                                       sys_hz=int(self.sys_clk_freq))
        self.comb += [
            self._prof_pads.eng_busy.eq(self.engine.busy),
            self._prof_pads.vec_busy.eq(self.vecunit.busy),
            self._prof_pads.seq_busy.eq(self.seq.busy),
        ]

        self.bus.add_slave("scratchpad", self.scratchpad.add_wishbone_port(),
            SoCRegion(origin=0xa000_0000, size=self.scratchpad.size, cached=False))

        # Firmware-visible contract.
        self.add_constant("BONSAI_HDR_ADDR",         MAIN_RAM_BASE + HDR_OFFSET)
        if mem == "hbm":
            self.add_constant("BONSAI_CHAN_WEIGHT_BASE", 0)
            # Sim keeps stride 0: each channel has its own independent HBM model, so the models
            # already give every channel a private address space. On hardware the AXI switch makes
            # the address select the pseudo-channel, hence the non-zero stride there.
            self.add_constant("BONSAI_CHAN_STRIDE",      0)
        else:
            # The XCU1525's geometry, scaled to the tiny model: the same constants the
            # firmware reads on the board (BONSAI_CHAN_MOD, BONSAI_MEM_CPU_OFFSET,
            # BONSAI_N_MEMCTRL, BONSAI_DDR_CSR_STRIDE) with 2 MB windows from 12 MB.
            self.add_constant("BONSAI_CHAN_WEIGHT_BASE", DDR4_SIM_WEIGHT_BASE)
            self.add_constant("BONSAI_CHAN_STRIDE",      DDR4_SIM_CHAN_STRIDE)
            self.add_constant("BONSAI_CHAN_MOD",         chan_mod)
            self.add_constant("BONSAI_CHAN_SKEW",        DDR4_SIM_CHAN_SKEW)
            self.add_constant("BONSAI_MEM_CPU_OFFSET",   0)
            self.add_constant("BONSAI_N_MEMCTRL",        n_memctrl)
            self.add_constant("BONSAI_DDR_CSR_STRIDE",   DDR4_SIM_CSR_STRIDE)
            self.add_constant("SDRAM_TEST_DISABLE") # The BIOS's memtest would erase the init.
        self.add_constant("BONSAI_N_CHANNELS",       n_channels)
        if firmware is not None:
            self.add_constant("ROM_BOOT_ADDRESS", MAIN_RAM_BASE)

    def _add_ddr4(self, n_channels, n_memctrl, ram, img, axis):
        """The XCU1525's memory system on LiteDRAM's PHY model; returns chan_mod."""
        from litedram.modules import MT40A512M8
        from litedram.phy.model import SDRAMPHYModel
        from litedram.frontend.axi import LiteDRAMAXI2Native

        assert n_channels % n_memctrl == 0, "channels shard evenly over the controllers"
        chan_mod = n_channels//n_memctrl
        assert chan_mod & (chan_mod - 1) == 0

        class _SimDDR4(MT40A512M8):
            # The board's part with 2048 rows x 128 columns: 32 MB per x64 controller, enough
            # for main_ram (8 MB) and up to four 2 MB windows from 12 MB, small enough to
            # initialise from Python and hold in the verilated memory. (The PHY model needs
            # at least 11 address bits for the A10 precharge-all decode: rows, not columns.)
            nrows = 2048
            ncols = 128

        # Controller k's contents: controller 0 carries main_ram (firmware + header) at 0
        # and, like the others, its channels' payloads at their windows.
        span = DDR4_SIM_WEIGHT_BASE + chan_mod*DDR4_SIM_CHAN_STRIDE
        for k in range(n_memctrl):
            content = bytearray(span)
            if k == 0:
                content[:len(ram)] = ram
            if img is not None:
                for j in range(chan_mod):
                    c = k*chan_mod + j
                    payload = bytes(img.channel_payload(c))
                    base = DDR4_SIM_WEIGHT_BASE + j*DDR4_SIM_CHAN_STRIDE + j*DDR4_SIM_CHAN_SKEW
                    assert len(payload) + 512*1024 + (chan_mod - 1)*DDR4_SIM_CHAN_SKEW <= DDR4_SIM_CHAN_STRIDE, \
                        f"channel {c}: {len(payload)} B does not fit a sim window"
                    content[base:base + len(payload)] = payload
            words = [int.from_bytes(content[i:i+4], "little") for i in range(0, len(content), 4)]
            sfx = "" if k == 0 else str(k)
            self.csr.add(f"sdram{sfx}",  DDR4_SIM_CSR_PAGE0 + 2*k)
            self.csr.add(f"ddrphy{sfx}", DDR4_SIM_CSR_PAGE0 + 2*k + 1)
            module = _SimDDR4(int(100e6), "1:4") # 100 MHz timings, as litex_sim does.
            phy = SDRAMPHYModel(module=module, data_width=64, clk_freq=int(100e6), init=words)
            setattr(self, f"ddrphy{sfx}", phy)
            if k == 0:
                self.add_sdram("sdram", phy=phy, module=module, size=MAIN_RAM_SIZE,
                               l2_cache_size=8192)
            else:
                self.add_sdram(f"sdram{sfx}", phy=phy, module=module,
                               with_soc_interconnect=False)
            core = getattr(self, f"sdram{sfx}")
            from trixium.gateware.ddr_port import engine_ddr_port
            for j in range(chan_mod):
                c = k*chan_mod + j
                axi  = AXIInterface(data_width=256, address_width=33, id_width=6)
                for jj, m in enumerate(engine_ddr_port(core, axi)):   # the target's default port
                    setattr(self, f"engport{c}_{jj}", m)
                axis.append(axi)
        # VexiiRiscv enforces execute permission from region.mode (the FK33 target's lesson).
        for _rn, _r in self.bus.regions.items():
            if _r.origin == MAIN_RAM_BASE and not _r.linker:
                _r.mode = "rwx"
        return chan_mod

# Build --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriXium Verilator simulation.")
    parser.add_argument("--image",          default="models/tiny/bonsai-tiny-n2.img")
    parser.add_argument("--firmware",       default="firmware/bonsai.bin")
    parser.add_argument("--build-software", action="store_true", help="Generate headers + BIOS only.")
    parser.add_argument("--run",            action="store_true", help="Build and run the simulation.")
    parser.add_argument("--cpu-type",  default="vexriscv",
                        choices=["vexriscv", "vexiiriscv"],
                        help="vexiiriscv is SpinalHDL-generated so its I-cache is sizable; "
                             "vexriscv's is a fixed 4 KB in every variant")
    parser.add_argument("--icache-kb", type=int, default=0,
                        help="vexiiriscv only: I-cache size in KB (default 16)")
    parser.add_argument("--vexii-extra", default="",
                        help="vexiiriscv only: extra SocGen args, e.g. "
                             "'--with-btb --with-ras --with-gshare'")
    parser.add_argument("--update-repo", default="no", choices=["no", "recommended", "latest"],
                        help="VexiiRiscv source update policy (default: no, for reproducible runs)")
    parser.add_argument("--pc-profile",     action="store_true",
                        help="compile the VexRiscv PC profiler in (sim/pc_profiler.cpp); "
                             "writes pcprof-*.txt into build/sim/gateware")
    parser.add_argument("--trace",          action="store_true", help="Enable FST tracing.")
    parser.add_argument("--hbm-latency",    default=16, type=int)
    parser.add_argument("--hbm-stall-rate", default=0, type=int)
    parser.add_argument("--threads",        default=4, type=int)
    parser.add_argument("--dual",           action="store_true",
        help="Build the dual-Q engine (S3/S6 head-B datapath, spec decode). The default is the "
             "slim engine the hardware builds ship (no --with-dual): the dual attention path "
             "(attn_seq2_layer) is not what the card runs, and on 2026-09-11 it was the one "
             "diverging from the mirror while the slim path matched at every checksum.")
    parser.add_argument("--no-dual",        action="store_true", help=argparse.SUPPRESS)  # old default
    parser.add_argument("--mem",            default="hbm", choices=["hbm", "ddr4"],
        help="hbm: one independent channel model per engine channel (the FK33's shape, the "
             "default). ddr4: the XCU1525's memory system on LiteDRAM's PHY model - N "
             "controllers, main_ram on controller 0, chan_mod channels per controller; "
             "builds in build/sim_ddr4 (the firmware must be built against that tree).")
    parser.add_argument("--n-memctrl",      default=4, type=int, help="--mem ddr4: controllers.")
    parser.add_argument("--with-memory-traffic", action="store_true", help="Enable coherent memory traffic counters.")
    parser.add_argument("--with-native", action="store_true", help="Enable the experimental shared native projection engine.")
    parser.add_argument("--with-hadamard", action="store_true", help="Enable experimental sequenced scratchpad integer rotations.")
    parser.add_argument("--with-activation-quantize", action="store_true", help="Enable experimental resident FP32 and post-rotation Q8 quantization.")
    parser.add_argument("--with-projection-restore", action="store_true", help="Enable experimental resident int32 projection to FP32 restoration.")
    parser.add_argument("--with-embedding-restore", action="store_true", help="Enable experimental compact PTQ embedding restoration.")
    parser.add_argument("--with-silu", action="store_true", help="Enable the exact FP32 SiLU-multiply scratchpad unit (sequencer unit 11).")
    parser.add_argument("--with-norm", action="store_true", help="Enable the exact RMSNorm/L2 scratchpad unit (sequencer unit 12).")
    parser.add_argument("--with-conv", action="store_true", help="Enable the exact DeltaNet conv scratchpad unit (sequencer unit 13).")
    parser.add_argument("--with-dnrec", action="store_true", help="DeltaNet recurrence unit on the exact FP32 operators (the board's hardware path).")
    parser.add_argument("--with-region-fetch", action="store_true", help="Enable sequenced checked region reads; requires --with-native.")
    parser.add_argument("--no-capabilities", action="store_true", help="Omit read-only capability ROM.")
    args = parser.parse_args()
    out_dir = "build/sim" if args.mem == "hbm" else "build/sim_ddr4"

    sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_module("serial2console", "serial")

    if args.build_software:
        soc     = BonsaiSimSoC(image=args.image, firmware=None, dual=args.dual,
        cpu_type=args.cpu_type, icache_kb=args.icache_kb, vexii_extra=args.vexii_extra,
            hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate,
            mem=args.mem, n_memctrl=args.n_memctrl, update_repo=args.update_repo,
            with_memory_traffic=args.with_memory_traffic, with_capabilities=not args.no_capabilities,
            with_native=args.with_native, with_hadamard=args.with_hadamard,
            with_activation_quantize=args.with_activation_quantize, with_projection_restore=args.with_projection_restore,
            with_embedding_restore=args.with_embedding_restore, with_region_fetch=args.with_region_fetch,
            with_silu=args.with_silu, with_norm=args.with_norm, with_conv=args.with_conv,
            with_dnrec=args.with_dnrec)
        builder = Builder(soc, output_dir=out_dir, csr_csv=os.path.join(out_dir, "csr.csv"))
        builder.compile_gateware = False
        builder.build(sim_config=sim_config, run=False)
        if hasattr(soc, 'bonsai_capabilities'):
            soc.bonsai_capabilities.write_artifacts(builder.output_dir)
        return

    soc     = BonsaiSimSoC(image=args.image, firmware=args.firmware, dual=args.dual,
        cpu_type=args.cpu_type, icache_kb=args.icache_kb, vexii_extra=args.vexii_extra,
        hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate,
        mem=args.mem, n_memctrl=args.n_memctrl, update_repo=args.update_repo,
        with_memory_traffic=args.with_memory_traffic, with_capabilities=not args.no_capabilities,
        with_native=args.with_native, with_hadamard=args.with_hadamard,
        with_activation_quantize=args.with_activation_quantize, with_projection_restore=args.with_projection_restore,
        with_embedding_restore=args.with_embedding_restore, with_region_fetch=args.with_region_fetch,
            with_silu=args.with_silu, with_norm=args.with_norm, with_conv=args.with_conv,
            with_dnrec=args.with_dnrec)
    builder = Builder(soc, output_dir=out_dir, csr_csv=os.path.join(out_dir, "csr.csv"))
    # --pc-profile compiles sim/pc_profiler.cpp into the verilated binary, where it overrides
    # LiteX's weak litex_sim_user_init / litex_sim_user_post_eval hooks and samples the
    # VexRiscv retired PC every cycle. That is the only way to attribute the ~69% of a token
    # in which neither the engine nor the vector unit is busy; perf_print's buckets are
    # op-class wall time that already contains the CPU glue.
    #
    # Note LiteX's build_sim.sh starts with `rm -rf obj_dir/`, so adding this triggers a full
    # re-verilate the first time.
    extra_sources = []
    if args.pc_profile and args.cpu_type != "vexriscv":
        # pc_profiler.cpp includes Vsim_VexRiscv.h and reads lastStagePc/lastStageIsFiring by
        # name. VexiiRiscv's generated module is VexiiRiscvLitex_<hash> with a different
        # internal structure, so the profiler would not compile, let alone sample. Refuse
        # loudly rather than emit a confusing "No such file: Vsim_VexRiscv.h" from verilator.
        raise SystemExit("--pc-profile currently supports --cpu-type vexriscv only "
                         "(the profiler reads VexRiscv's lastStagePc by name). "
                         "Compare CPUs on leg cycle counts instead.")
    if args.pc_profile:
        extra_sources.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "pc_profiler.cpp"))
    builder.build(sim_config=sim_config, run=args.run, opt_level="O3",
        threads=args.threads, trace=args.trace, trace_fst=args.trace,
        verilator_extra_sources=extra_sources)
    if hasattr(soc, 'bonsai_capabilities'):
        soc.bonsai_capabilities.write_artifacts(builder.output_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium Verilator simulation SoC.

The same TriXium core and firmware-visible contract as the boards, on simulated memories
initialised from a BONS image: --board xcu1525 is the XCU1525's memory system (four
LiteDRAM controllers on LiteDRAM's DDR4 PHY model, in their own ddr domain) with every unit,
--board fk33 one HBM channel model per engine channel and no scratchpad units. The firmware
binary and the BONS header+table are baked into main_ram at elaboration, the channel payloads
into the memory models.

Flow (from the repository root):
  python3 -m trixium.targets.sim --build-software       # headers + BIOS (no gateware)
  make -C firmware BUILD_DIR=../build/sim_xcu1525       # firmware/trixium.bin
  python3 -m trixium.targets.sim --run                  # elaborate + verilate + run
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *
from litex.gen.genlib.misc import *

from litex.build.io import CRG
from litex.build.generic_platform import Pins, Subsignal
from litex.build.sim.config import SimConfig
from litex.tools.litex_sim import Platform

from litex.soc.cores.cpu.vexiiriscv.core import VexiiRiscv
from litex.soc.integration.common import *
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder
from litex.soc.interconnect.axi import AXIInterface

from litedram.modules import MT40A512M8
from litedram.phy.model import SDRAMPHYModel

from trixium.layout import HDR_CPU_OFFSET
from trixium.core.core import TriXiumConfig, TriXiumCore
from trixium.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from trixium.memory.hbm_model import HBMChannelModel, words_from_bytes
from trixium.model.bons import BONSReader

# Constants ----------------------------------------------------------------------------------------

MAIN_RAM_BASE = 0x4000_0000
MAIN_RAM_SIZE = 0x0080_0000    # 8 MB.
HDR_OFFSET    = HDR_CPU_OFFSET # BONS header+table at main_ram + 4 MB (shared with hardware).

# The DDR4 flavour of the sim (--mem ddr4): the XCU1525's memory plumbing on LiteDRAM's PHY
# model - main_ram on controller 0 through the BIOS's own sdram_init(), the engine channels
# on 256-bit crossbar ports behind LiteDRAMAXI2Native bridges, chan_mod channels per
# controller at these window constants, and the firmware's ddr_init_all()/ddr_test() for
# controllers 1..N-1 (firmware/ddr.c) - all on the tiny model, tokens against the same
# reference as the HBM-model sim. Sizes: main_ram is 8 MB (0..8 MB of controller 0), the
# windows start at 12 MB so the header staging area (window - 1 MB) and the round-trip test
# (window - 2 MB) stay above main_ram; 4 MB per window holds the tiny payload and its KV.
# The channel skew is a bank skew IN THE SIM'S GEOMETRY (ncols 128 puts the bank bits at byte
# bits 10..13, so the board's 32 KB would be a row bit here); the address arithmetic is what
# the gate checks.
DDR4_SIM_WEIGHT_BASE = 0x00C0_0000 # 12 MB.
DDR4_SIM_CHAN_STRIDE = 0x0040_0000 # 4 MB.
DDR4_SIM_CHAN_SKEW   = 0xA000      # 40 KB.
DDR4_SIM_CSR_WINDOW  = 0xF0020000  # The target's DDR_CSR_WINDOW.
DDR_SPLIT_CLK_RATIO  = 2           # ddr clock / sys clock in the simulation.
DDR4_SIM_CSR_PAGE0   = 8           # (sdram, ddrphy) pairs on pages 8/9, 10/11, ... as the target.
DDR4_SIM_CSR_STRIDE  = 0x1000

# TriXiumSimSoC ------------------------------------------------------------------------------------

class TriXiumSimSoC(SoCCore):
    def __init__(self, image=None, firmware=None, hbm_latency=16, hbm_stall_rate=0,
                 cpu_type="vexriscv", icache_kb=0, vexii_extra="",
                 mem="hbm", n_memctrl=4, update_repo="no",
                 with_hadamard=False, with_activation_quantize=False,
                 with_projection_restore=False, with_silu=False, with_norm=False, with_conv=False,
                 with_dnstream=False, attn_geometry=None):
        assert mem in ("hbm", "ddr4")
        ddr_split = mem == "ddr4"
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
            # VexiiRiscv reads a pile of class attributes (soc_args, soc_keys, update_repo,
            # vexii_args, ...) that LiteX normally populates from its CLI. Constructing
            # SoCCore directly bypasses that, so drive the real API with a defaults-only
            # parser instead of hand-setting attributes - hand-setting works until the next
            # LiteX bump adds a key, and then fails as an AttributeError deep in codegen.
            _p = argparse.ArgumentParser()
            VexiiRiscv.args_fill(_p)
            _va = _p.parse_args([])
            _va.cpu_variant = "standard"
            _va.update_repo = update_repo
            # The extra args go through the real API: args_read runs the generator's
            # PythonArgsGen on them and the ISA map (hence the software's -march) follows,
            # so "--with-rvf" gives an FPU the firmware actually uses.
            _va.vexii_args  = vexii_extra
            VexiiRiscv.args_read(_va)
            # LiteX builds --with-isa from a set, whose process-random iteration order gives
            # the same CPU a different netlist hash on successive runs and needlessly invokes
            # SBT again. Canonicalise only that comma list; the option order remains otherwise
            # untouched.
            _canon = []
            for _arg in VexiiRiscv.vexii_args.split():
                if _arg.startswith("--with-isa="):
                    _arg = "--with-isa=" + ",".join(sorted(_arg.split("=", 1)[1].split(",")))
                _canon.append(_arg)
            VexiiRiscv.vexii_args = " ".join(_canon)
            # Now override just the fetch cache.
            sets = max(16, (icache_kb*1024)//(4*64)) if icache_kb else 64
            VexiiRiscv.vexii_args = (VexiiRiscv.vexii_args
                                     .replace("--fetch-l1-ways=2", "--fetch-l1-ways=4")
                                     + f" --fetch-l1-sets={sets}")
            print(f"[trixium] VexiiRiscv args: {VexiiRiscv.vexii_args}")

        SoCCore.__init__(self, platform,
            clk_freq                 = int(1e6),
            ident                    = "TriXium Sim" + (" DDR4" if mem == "ddr4" else ""),
            cpu_type                 = cpu_type,
            integrated_rom_size      = 0x20000,
            integrated_main_ram_size = MAIN_RAM_SIZE if mem == "hbm" else 0,
            integrated_main_ram_init = ram_words if mem == "hbm" else [],
            uart_name                = "sim")
        self.crg = CRG(platform.request("sys_clk"))
        if ddr_split:
            # The controllers in their own `ddr` domain on a second sim clock
            # (DDR_SPLIT_CLK_RATIO x sys), as on the XCU1525.
            platform.add_extension([("ddr_clk", 0, Pins(1))])
            self.cd_ddr = ClockDomain()
            self.comb += self.cd_ddr.clk.eq(platform.request("ddr_clk"))
            self.specials += AsyncResetSynchronizer(self.cd_ddr, ResetSignal("sys"))

        # Profiler pad: unit busy bits as TOP-LEVEL sim ports (prof_eng_busy, prof_vec_busy,
        # prof_seq_busy on the Vsim). The C++ profiler traces their edges into an event file
        # for the Perfetto timeline. A pad, not __PVT__ names: Verilator's internal names are
        # not a stable contract (__PVT__vecunit_busy is already folded away in this build),
        # while requested pads are emitted verbatim.
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
                model = HBMChannelModel(axi, size=size, init=init,
                    latency=hbm_latency, stall_rate=hbm_stall_rate)
                self.add_module(f"hbm_model{c}", model)
                axis.append(axi)
                memory_sizes.append(size)
        else:
            chan_mod = self._add_ddr4(n_channels, n_memctrl, ram, img, axis)
        # Same integration as both hardware targets; only memory/physical policy differs.
        config = TriXiumConfig(with_sp_csr=False,
            chan_stride=DDR4_SIM_CHAN_STRIDE if mem == "ddr4" else 0,
            chan_mod=chan_mod, chan_skew=DDR4_SIM_CHAN_SKEW if mem == "ddr4" else 0,
            with_hadamard=with_hadamard, with_activation_quantize=with_activation_quantize,
            with_projection_restore=with_projection_restore, with_silu=with_silu,
            with_norm=with_norm, with_conv=with_conv,
            with_dnstream=bool(with_dnstream),
            with_attnstream=attn_geometry is not None,
            attnstream_options=tuple(zip(("lanes", "groups", "hb_max", "t_max"),
                (attn_geometry[0], attn_geometry[1], attn_geometry[2]//8, attn_geometry[3])))
                if attn_geometry else ())
        self.trixium = TriXiumCore(axis, config)
        for name, value in self.trixium.constants.items():
            self.add_constant(name, value)
        self.comb += [
            self._prof_pads.eng_busy.eq(self.trixium.engine.busy),
            self._prof_pads.vec_busy.eq(self.trixium.vecunit.busy),
            self._prof_pads.seq_busy.eq(self.trixium.seq.busy),
        ]

        self.bus.add_slave("scratchpad", self.trixium.bus,
            SoCRegion(origin=0xa000_0000, size=self.trixium.scratchpad.size, cached=False))

        # Firmware-visible contract.
        self.add_constant("TRIXIUM_HDR_ADDR",         MAIN_RAM_BASE + HDR_OFFSET)
        if mem == "hbm":
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", 0)
            # Sim keeps stride 0: each channel has its own independent HBM model, so the models
            # already give every channel a private address space. On hardware the AXI switch makes
            # the address select the pseudo-channel, hence the non-zero stride there.
            self.add_constant("TRIXIUM_CHAN_STRIDE",      0)
        else:
            # The XCU1525's geometry, scaled to the tiny model: the same constants the
            # firmware reads on the board (TRIXIUM_CHAN_MOD, TRIXIUM_MEM_CPU_OFFSET,
            # TRIXIUM_N_MEMCTRL, TRIXIUM_DDR_CSR_STRIDE) with 4 MB windows from 12 MB.
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", DDR4_SIM_WEIGHT_BASE)
            self.add_constant("TRIXIUM_CHAN_STRIDE",      DDR4_SIM_CHAN_STRIDE)
            self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
            self.add_constant("TRIXIUM_CHAN_SKEW",        DDR4_SIM_CHAN_SKEW)
            self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
            self.add_constant("TRIXIUM_N_MEMCTRL",        n_memctrl)
            self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   DDR4_SIM_CSR_STRIDE)
            self.add_constant("SDRAM_TEST_DISABLE") # The BIOS's memtest would erase the init.
        self.add_constant("TRIXIUM_N_CHANNELS",       n_channels)
        if firmware is not None:
            self.add_constant("ROM_BOOT_ADDRESS", MAIN_RAM_BASE)

    def _add_ddr4(self, n_channels, n_memctrl, ram, img, axis):
        """The XCU1525's memory system on LiteDRAM's PHY model; returns chan_mod."""
        assert n_channels % n_memctrl == 0, "channels shard evenly over the controllers"
        chan_mod = n_channels//n_memctrl
        assert chan_mod & (chan_mod - 1) == 0

        class _SimDDR4(MT40A512M8):
            # The board's part with 2048 rows x 128 columns: 32 MB per x64 controller, enough
            # for main_ram (8 MB) and up to four 4 MB windows from 12 MB, small enough to
            # initialise from Python and hold in the verilated memory. (The PHY model needs
            # at least 11 address bits for the A10 precharge-all decode: rows, not columns.)
            nrows = 2048
            ncols = 128

        # Controller k's contents: controller 0 carries main_ram (firmware + header) at 0
        # and, like the others, its channels' payloads at their windows.
        span = DDR4_SIM_WEIGHT_BASE + chan_mod*DDR4_SIM_CHAN_STRIDE
        self.sdrams, split_phys, split_modules = [], [], []
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
            module = _SimDDR4(int(100e6), "1:4") # 100 MHz timings, as litex_sim does.
            phy = SDRAMPHYModel(module=module, data_width=64, clk_freq=int(100e6), init=words)
            split_phys.append(phy); split_modules.append(module)
        add_ddr_domain_controllers(self, split_phys, split_modules, int(100e6), MAIN_RAM_SIZE,
                                   8192, DDR4_SIM_CSR_WINDOW)
        for k in range(n_memctrl):
            core = self.sdrams[k]
            for j in range(chan_mod):
                c = k*chan_mod + j
                axi  = AXIInterface(data_width=256, address_width=33, id_width=6)
                for jj, m in enumerate(engine_ddr_port(core, axi, clock_domain="sys_port")):
                    setattr(self, f"engport{c}_{jj}", m)
                axis.append(axi)
        # VexiiRiscv builds its memory map from region.mode and enforces execute permission.
        for _rn, _r in self.bus.regions.items():
            if _r.origin == MAIN_RAM_BASE and not _r.linker:
                _r.mode = "rwx"
        return chan_mod

# Build --------------------------------------------------------------------------------------------

# Board presets: the hardware the simulation mirrors.
BOARDS = {
    # The XCU1525's memory system and every unit (the 27B's hybrid model needs them), the
    # attention stream at the tiny models' geometry.
    "xcu1525": dict(mem="ddr4", n_memctrl=4, with_hadamard=True, with_activation_quantize=True,
        with_projection_restore=True, with_silu=True, with_norm=True, with_conv=True,
        with_dnstream=True, attn_geometry=(8, 2, 128, 256),
        cpu_type="vexiiriscv", icache_kb=32, vexii_extra="--with-rvf --with-rvZcbm"),
    # The FK33: HBM channels, the engine/sequencer/vector unit only (the Qwen3 path).
    "fk33": dict(mem="hbm",
        cpu_type="vexiiriscv", icache_kb=32, vexii_extra="--with-rvZcbm"),
}

def main():
    from trixium.migen_fast import install as install_migen_fast
    install_migen_fast()  # Linear-time Verilog conversion (minutes -> seconds).
    parser = argparse.ArgumentParser(description="TriXium Verilator simulation.")
    parser.add_argument("--board",          default="xcu1525", choices=list(BOARDS),              help="Hardware to mirror.")
    parser.add_argument("--image",          default="models/tiny/bonsai-tiny-hybrid-1k-n8.img",     help="BONS image to bake in.")
    parser.add_argument("--firmware",       default="firmware/trixium.bin",                         help="Firmware binary to bake in.")
    parser.add_argument("--build-software", action="store_true",                                    help="Generate headers + BIOS only (to build the firmware against).")
    parser.add_argument("--run",            action="store_true",                                    help="Build and run the simulation.")
    parser.add_argument("--out-dir",        default=None,                                           help="Build directory (default: build/sim_<board>).")
    parser.add_argument("--cpu-type",       default=None, choices=["vexriscv", "vexiiriscv"],      help="Override the board's CPU (vexriscv: the PC profiler).")
    parser.add_argument("--update-repo",    default="no", choices=["no", "recommended", "latest"], help="VexiiRiscv source update policy.")
    parser.add_argument("--pc-profile",     action="store_true",                                    help="Compile the VexRiscv PC profiler in (sim_pc_profiler.cpp).")
    parser.add_argument("--trace",          action="store_true",                                    help="Enable FST tracing.")
    parser.add_argument("--hbm-latency",    default=16, type=int,                                   help="--board fk33: HBM model read latency.")
    parser.add_argument("--hbm-stall-rate", default=0,  type=int,                                   help="--board fk33: HBM model stall rate.")
    parser.add_argument("--threads",        default=4,  type=int,                                   help="Verilator threads.")
    args = parser.parse_args()
    out_dir = args.out_dir or f"build/sim_{args.board}"
    board   = dict(BOARDS[args.board])
    if args.cpu_type is not None:
        board.update(cpu_type=args.cpu_type, icache_kb=0, vexii_extra="")

    sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_module("serial2console", "serial")
    if board["mem"] == "ddr4":
        sim_config.add_clocker("ddr_clk", int(1e6)*DDR_SPLIT_CLK_RATIO)

    soc_kwargs = dict(board, update_repo=args.update_repo,
        hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate)

    if args.build_software:
        soc     = TriXiumSimSoC(image=args.image, firmware=None, **soc_kwargs)
        builder = Builder(soc, output_dir=out_dir, csr_csv=os.path.join(out_dir, "csr.csv"))
        builder.compile_gateware = False
        builder.build(sim_config=sim_config, run=False)
        return

    soc     = TriXiumSimSoC(image=args.image, firmware=args.firmware, **soc_kwargs)
    builder = Builder(soc, output_dir=out_dir, csr_csv=os.path.join(out_dir, "csr.csv"))
    # --pc-profile compiles sim_pc_profiler.cpp into the verilated binary, where it overrides
    # LiteX's weak litex_sim_user_init / litex_sim_user_post_eval hooks and samples the
    # VexRiscv retired PC every cycle. That is the only way to attribute the part of a token
    # in which neither the engine nor the vector unit is busy; perf_print's buckets are
    # op-class wall time that already contains the CPU glue.
    #
    # Note LiteX's build_sim.sh starts with `rm -rf obj_dir/`, so adding this triggers a full
    # re-verilate the first time.
    extra_sources = []
    if args.pc_profile and board["cpu_type"] != "vexriscv":
        # pc_profiler.cpp includes Vsim_VexRiscv.h and reads lastStagePc/lastStageIsFiring by
        # name. VexiiRiscv's generated module is VexiiRiscvLitex_<hash> with a different
        # internal structure, so the profiler would not compile, let alone sample. Refuse
        # loudly rather than emit a confusing "No such file: Vsim_VexRiscv.h" from verilator.
        raise SystemExit("--pc-profile currently supports --cpu-type vexriscv only "
                         "(the profiler reads VexRiscv's lastStagePc by name). "
                         "Compare CPUs on leg cycle counts instead.")
    if args.pc_profile:
        extra_sources.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "sim_pc_profiler.cpp"))
    builder.build(sim_config=sim_config, run=args.run, opt_level="O3",
        threads=args.threads, trace=args.trace, trace_fst=args.trace,
        verilator_extra_sources=extra_sources)

if __name__ == "__main__":
    main()

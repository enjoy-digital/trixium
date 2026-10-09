#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXium Verilator simulation SoC.

The same TriXium core and firmware-visible contract as the boards, on simulated memories
initialized from a BONS image: --board xcu1525 is the XCU1525's memory system (four LiteDRAM
controllers on LiteDRAM's DDR4 PHY model, in their own ddr domain) with every unit, --board fk33
one HBM channel model per engine channel and no scratchpad units. The firmware binary and the
BONS header+table are baked into main_ram at elaboration, the channel payloads into the memory
models.

  ./trixium_sim.py --build      # headers + libraries, firmware, Verilog (~1 min), Verilator (~2.5 min)
  ./trixium_sim.py --run        # run it (console on stdin/stdout)
  ./trixium_sim.py --firmware   # rebuild the firmware and re-bake it into the built sim (seconds)
  ./trixium_sim.py --gate       # traced generation compared with the Python mirror (~10 min)
  ./trixium_sim.py --bench      # resident path off/on, cycles per forward

--board (xcu1525, fk33) and --image select the build (build/sim_<board>); the gate and the bench
must use the image the simulation was built with.
"""

import os
import sys
import glob
import argparse
import subprocess

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *

from litex.build.io import CRG
from litex.build.generic_platform import Pins
from litex.build.sim.config import SimConfig
from litex.tools.litex_sim import Platform

from litex.soc.cores.cpu.vexiiriscv.core import VexiiRiscv
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder
from litex.soc.interconnect.axi import AXIInterface

from litedram.modules import MT40A512M8
from litedram.phy.model import SDRAMPHYModel

from model.layout import HDR_CPU_OFFSET
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.ddr import add_ddr_domain_controllers, engine_ddr_port
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes
from model.bons import BONSReader
from gateware.common import set_main_ram_executable

# Constants ----------------------------------------------------------------------------------------

MAIN_RAM_BASE = 0x4000_0000
MAIN_RAM_SIZE = 0x0080_0000    # 8 MB.
HDR_OFFSET    = HDR_CPU_OFFSET # BONS header+table at main_ram + 4 MB (shared with hardware).

# The DDR4 flavor of the sim: the XCU1525's memory plumbing on LiteDRAM's PHY model - main_ram on
# controller 0, the engine channels on crossbar ports behind AXI bridges, chan_mod channels per
# controller at these window constants (the PHY model needs no init sequence: the firmware skips
# ddr_init_all()/ddr_test() under TRIXIUM_SIM, its console still runs them). Sizes: main_ram is 8 MB
# (0..8 MB of controller 0), the windows start at 12 MB so the header staging area (window - 1 MB)
# and the round-trip test (window - 2 MB) stay above main_ram; 4 MB per window holds the tiny
# payload and its KV. The channel skew is a bank skew IN THE SIM'S GEOMETRY (ncols 128 puts the
# bank bits at byte bits 10..13, so the board's 32 KB would be a row bit here); the address
# arithmetic is what the simulation checks.
DDR4_SIM_WEIGHT_BASE = 0x00C0_0000 # 12 MB.
DDR4_SIM_CHAN_STRIDE = 0x0040_0000 # 4 MB.
DDR4_SIM_CHAN_SKEW   = 0xA000      # 40 KB.
DDR4_SIM_CSR_WINDOW  = 0xF0020000  # The target's DDR_CSR_WINDOW.
DDR_SPLIT_CLK_RATIO  = 2           # ddr clock / sys clock in the simulation.
DDR4_SIM_CSR_STRIDE  = 0x1000

# The ROM holds no BIOS, only a jump to the firmware in main_ram (`lui t0, MAIN_RAM_BASE;
# jr t0`), which the elaboration preloads: the BIOS's banner, ROM CRC, controller-0 init,
# serial-boot timeout and pre-boot flush of the whole 8 MB main_ram (one CBO per line) took
# ~2.4M cycles, ~150 s of Verilator, before the firmware's first instruction. The firmware's
# crt0 needs nothing from it (stack, trap vector, .data/.bss are its own), the PHY model needs
# no init sequence and the DFII starts in hardware control.
ROM_BOOT_STUB = [
    0x0000_0037 | (5 << 7) | (MAIN_RAM_BASE & 0xFFFF_F000), # lui  t0, %hi(MAIN_RAM_BASE)
    0x0000_0067 | (5 << 15),                                # jalr x0, 0(t0)
]

# Memory Contents ----------------------------------------------------------------------------------

class SimDDR4(MT40A512M8):
    # The board's part with 2048 rows x 128 columns: 32 MB per x64 controller, enough for
    # main_ram (8 MB) and up to four 4 MB windows from 12 MB, small enough to initialize from
    # Python and hold in the verilated memory. (The PHY model needs at least 11 address bits for
    # the A10 precharge-all decode: rows, not columns.)
    nrows = 2048
    ncols = 128


def main_ram_content(img, firmware=None):
    """main_ram: the firmware binary at 0, the BONS header+table at HDR_OFFSET."""
    ram = bytearray(MAIN_RAM_SIZE)
    if firmware is not None:
        assert len(firmware) < HDR_OFFSET
        ram[:len(firmware)] = firmware
    if img is not None:
        meta = bytes(img.data[:img.header["payload_offset"]])
        ram[HDR_OFFSET:HDR_OFFSET + len(meta)] = meta
    return ram


def ddr4_content(k, ram, img, chan_mod):
    """DDR4 controller k: main_ram at 0 for controller 0 and, for all, its channels' payloads at
    their windows."""
    content = bytearray(DDR4_SIM_WEIGHT_BASE + chan_mod*DDR4_SIM_CHAN_STRIDE)
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
    return content


def words(content):
    return [int.from_bytes(content[i:i+4], "little") for i in range(0, len(content), 4)]


def ddr4_phy(content):
    module = SimDDR4(int(100e6), "1:4") # 100 MHz timings, as litex_sim does.
    phy    = SDRAMPHYModel(module=module, data_width=64, clk_freq=int(100e6), init=words(content))
    return phy, module

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
        fw = None
        if firmware is not None and os.path.exists(firmware):
            with open(firmware, "rb") as f:
                fw = f.read()
        img = BONSReader(image) if image is not None else None
        ram = main_ram_content(img, fw)

        # CPU --------------------------------------------------------------------------------------
        # VexiiRiscv is generated at build time, so its I-cache is a parameter (VexRiscv's is a
        # fixed 4 KB): forward() alone is 12 KB, and 64 sets x 4 ways x 64 B lines = 16 KB covers
        # it with room to spare.
        if cpu_type == "vexiiriscv":
            # VexiiRiscv reads class attributes (soc_args, update_repo, vexii_args, ...) that LiteX
            # normally fills from its CLI: drive the real API with a defaults-only parser, so a
            # LiteX update adding a key does not break this construction.
            _p = argparse.ArgumentParser()
            VexiiRiscv.args_fill(_p)
            _va = _p.parse_args([])
            _va.cpu_variant = "standard"
            _va.update_repo = update_repo
            # args_read runs the generator's argument parsing on the extra args, so the ISA map
            # (hence the software's -march) follows them ("--with-rvf" gives an FPU the firmware
            # uses).
            _va.vexii_args  = vexii_extra
            VexiiRiscv.args_read(_va)
            # LiteX builds --with-isa from a set, whose iteration order changes between runs and
            # with it the netlist hash (a needless SpinalHDL regeneration): sort that list only.
            _canon = []
            for _arg in VexiiRiscv.vexii_args.split():
                if _arg.startswith("--with-isa="):
                    _arg = "--with-isa=" + ",".join(sorted(_arg.split("=", 1)[1].split(",")))
                _canon.append(_arg)
            VexiiRiscv.vexii_args = " ".join(_canon)
            # Fetch cache geometry.
            sets = max(16, (icache_kb*1024)//(4*64)) if icache_kb else 64
            VexiiRiscv.vexii_args = (VexiiRiscv.vexii_args
                                     .replace("--fetch-l1-ways=2", "--fetch-l1-ways=4")
                                     + f" --fetch-l1-sets={sets}")
            print(f"[trixium] VexiiRiscv args: {VexiiRiscv.vexii_args}")

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(self, platform,
            clk_freq                 = int(1e6),
            ident                    = "TriXium Sim" + (" DDR4" if mem == "ddr4" else ""),
            cpu_type                 = cpu_type,
            integrated_rom_size      = 0x20000,
            integrated_rom_init      = ROM_BOOT_STUB,
            integrated_main_ram_size = MAIN_RAM_SIZE if mem == "hbm" else 0,
            integrated_main_ram_init = words(ram) if mem == "hbm" else [],
            uart_name                = "sim")

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(platform.request("sys_clk"))
        if ddr_split:
            # The controllers in their own `ddr` domain on a second sim clock
            # (DDR_SPLIT_CLK_RATIO x sys), as on the XCU1525.
            platform.add_extension([("ddr_clk", 0, Pins(1))])
            self.cd_ddr = ClockDomain()
            self.comb += self.cd_ddr.clk.eq(platform.request("ddr_clk"))
            self.specials += AsyncResetSynchronizer(self.cd_ddr, ResetSignal("sys"))

        # TriXium Memory Ports ---------------------------------------------------------------------
        memory_ports = []
        n_channels = img.header["n_channels"] if img is not None else 2
        chan_mod = 0
        if mem == "hbm":
            # One HBM channel model per engine channel.
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
                memory_ports.append(axi)
        else:
            chan_mod = self.add_ddr4(n_channels, n_memctrl, ram, img, memory_ports)

        # TriXium ----------------------------------------------------------------------------------
        # Same integration as both hardware targets; only memory/physical policy differs.
        config = TriXiumConfig(with_sp_csr=False,
            chan_stride              = DDR4_SIM_CHAN_STRIDE if mem == "ddr4" else 0,
            chan_mod                 = chan_mod,
            chan_skew                = DDR4_SIM_CHAN_SKEW if mem == "ddr4" else 0,
            with_hadamard            = with_hadamard,
            with_activation_quantize = with_activation_quantize,
            with_projection_restore  = with_projection_restore,
            with_silu                = with_silu,
            with_norm                = with_norm,
            with_conv                = with_conv,
            with_dnstream            = bool(with_dnstream),
            with_attnstream          = attn_geometry is not None,
            attnstream_options       = tuple(zip(("lanes", "groups", "hb_max", "t_max"),
                (attn_geometry[0], attn_geometry[1], attn_geometry[2]//8, attn_geometry[3])))
                if attn_geometry else ())
        self.trixium = TriXiumCore(memory_ports, config)
        for name, value in self.trixium.constants.items():
            self.add_constant(name, value)

        # Scratchpad on the SoC bus (CPU access).
        self.bus.add_slave("scratchpad", self.trixium.bus,
            SoCRegion(origin=0xa000_0000, size=self.trixium.scratchpad.size, cached=False))

        # Firmware-visible contract.
        self.add_constant("TRIXIUM_HDR_ADDR",         MAIN_RAM_BASE + HDR_OFFSET)
        if mem == "hbm":
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", 0)
            # Stride 0: every channel has its own HBM model, hence a private address space. On
            # hardware the AXI switch makes the address select the pseudo-channel.
            self.add_constant("TRIXIUM_CHAN_STRIDE",      0)
        else:
            # The XCU1525's geometry, scaled to the tiny model: the same constants the firmware
            # reads on the board, with 4 MB windows from 12 MB.
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", DDR4_SIM_WEIGHT_BASE)
            self.add_constant("TRIXIUM_CHAN_STRIDE",      DDR4_SIM_CHAN_STRIDE)
            self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
            self.add_constant("TRIXIUM_CHAN_SKEW",        DDR4_SIM_CHAN_SKEW)
            self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
            self.add_constant("TRIXIUM_N_MEMCTRL",        n_memctrl)
            self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   DDR4_SIM_CSR_STRIDE)
            self.add_constant("SDRAM_TEST_DISABLE") # sdram_init()'s memtest would erase the preload.
        self.add_constant("TRIXIUM_N_CHANNELS",       n_channels)
        # Simulation-only firmware paths (boot shortcuts, boot-phase cycle marks): the board
        # targets never define it.
        self.add_constant("TRIXIUM_SIM")

    def add_ddr4(self, n_channels, n_memctrl, ram, img, memory_ports):
        """The XCU1525's memory system on LiteDRAM's PHY model; returns chan_mod."""
        assert n_channels % n_memctrl == 0, "channels shard evenly over the controllers"
        chan_mod = n_channels//n_memctrl
        assert chan_mod & (chan_mod - 1) == 0

        # Controllers.
        self.sdrams, split_phys, split_modules = [], [], []
        for k in range(n_memctrl):
            phy, module = ddr4_phy(ddr4_content(k, ram, img, chan_mod))
            split_phys.append(phy)
            split_modules.append(module)
        add_ddr_domain_controllers(self, split_phys, split_modules, int(100e6), MAIN_RAM_SIZE,
                                   8192, DDR4_SIM_CSR_WINDOW)

        # Engine ports: channel c -> controller c // chan_mod.
        for k in range(n_memctrl):
            core = self.sdrams[k]
            for j in range(chan_mod):
                c = k*chan_mod + j
                axi = AXIInterface(data_width=256, address_width=33, id_width=6)
                for jj, m in enumerate(engine_ddr_port(core, axi, clock_domain="sys_port")):
                    setattr(self, f"engport{c}_{jj}", m)
                memory_ports.append(axi)
        set_main_ram_executable(self, MAIN_RAM_BASE)
        return chan_mod

# Firmware Re-bake --------------------------------------------------------------------------------

def bank_inits(content):
    """The init lines of each bank Memory of a DDR4 PHY model holding `content`."""
    phy, _ = ddr4_phy(content)
    inits  = []
    # BankModels are anonymous submodules: collect their Memory specials in creation order.
    for _, sub in phy._submodules:
        for m in sub._fragment.specials:
            if isinstance(m, Memory) and m.init is not None and len(m.init):
                width = (m.width + 3)//4
                inits.append([f"{v:0{width}x}" for v in m.init])
    return inits


def rebake_firmware(gateware_dir, image, firmware, n_memctrl):
    """Rewrite the bank init files holding the firmware in a built DDR4 simulation (seconds).

    The firmware lives in DDR4 controller 0's memory model: one sim_mem_N.init per bank, read at
    simulation start ($readmemh). Controller 0's contents are rebuilt as the SoC does, each bank's
    file is found by matching every line outside the firmware area, and rewritten."""
    img      = BONSReader(image)
    chan_mod = img.header["n_channels"]//n_memctrl
    with open(firmware, "rb") as f:
        fw = f.read()
    def inits(fw):
        return bank_inits(ddr4_content(0, main_ram_content(img, fw), img, chan_mod))
    new   = inits(fw)
    blank = inits(None)
    # Every line a firmware could occupy (the whole window below the header), so a smaller
    # firmware than the one baked in still matches: the old firmware's tail is replaced.
    full  = inits(b"\xff"*(HDR_OFFSET - 4))
    assert len(new) == len(blank) and new, "no bank memories found"

    files = {}
    for filename in glob.glob(os.path.join(gateware_dir, "sim_mem*.init")):
        with open(filename, encoding="utf-8") as f:
            files[filename] = f.read().split()
    done = 0
    for b, (n, z, f) in enumerate(zip(new, blank, full)):
        fw_lines = {i for i, (x, y) in enumerate(zip(f, z)) if x != y}
        match = [name for name, lines in files.items() if len(lines) == len(n)
                 and all(lines[i] == n[i] for i in range(len(n)) if i not in fw_lines)]
        if len(match) != 1:
            raise SystemExit(f"bank {b}: {len(match)} matching init files (rebuild with --build).")
        if files[match[0]] != n:
            with open(match[0], "w", encoding="utf-8") as fh:
                fh.write("\n".join(n) + "\n")
            done += 1
    print(f"[trixium] {done} bank files rewritten ({len(fw)} B firmware).")

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

FIRMWARE = "firmware/trixium.bin"


def build_firmware(out_dir):
    # BUILD_DIR relative to firmware/, as documented (make -C firmware BUILD_DIR=../build/<name>):
    # the firmware's build stamp then sees the same tree and keeps its objects.
    build_dir = os.path.relpath(os.path.abspath(out_dir), "firmware")
    subprocess.run(["make", "-C", "firmware", f"BUILD_DIR={build_dir}", f"PYTHON={sys.executable}"],
        check=True)


def build_sim(args, board, out_dir):
    sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_module("serial2console", "serial")
    if board["mem"] == "ddr4":
        sim_config.add_clocker("ddr_clk", int(1e6)*DDR_SPLIT_CLK_RATIO)
    soc_kwargs = dict(board, update_repo=args.update_repo,
        hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate)
    csr_csv = os.path.join(out_dir, "csr.csv")

    # Headers + LiteX libraries (to build the firmware against; the ROM is the boot stub, no BIOS).
    soc     = TriXiumSimSoC(image=args.image, firmware=None, **soc_kwargs)
    builder = Builder(soc, output_dir=out_dir, csr_csv=csr_csv)
    builder.build(sim_config=sim_config, build=False, run=False)

    # Firmware.
    build_firmware(out_dir)

    # SoC with the firmware baked in: Verilog, then Verilator.
    soc     = TriXiumSimSoC(image=args.image, firmware=FIRMWARE, **soc_kwargs)
    builder = Builder(soc, output_dir=out_dir, csr_csv=csr_csv)
    builder.build(sim_config=sim_config, run=False, opt_level="O3",
        threads=args.threads, trace=args.trace, trace_fst=args.trace)
    subprocess.run(["bash", "build_sim.sh"], cwd=os.path.join(out_dir, "gateware"), check=True)

# Run ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriXium Verilator simulation.")
    parser.add_argument("--board",          default="xcu1525", choices=list(BOARDS),               help="Hardware to mirror.")
    parser.add_argument("--image",          default="models/tiny/bonsai-tiny-hybrid-1k-n8.img",    help="BONS image to bake in.")
    parser.add_argument("--out-dir",        default=None,                                          help="Build directory (default: build/sim_<board>).")
    parser.add_argument("--build",          action="store_true",                                   help="Build the simulation (headers + libraries, firmware, Verilog, Verilator).")
    parser.add_argument("--run",            action="store_true",                                   help="Run the built simulation (console on stdin/stdout).")
    parser.add_argument("--firmware",       action="store_true",                                   help="Rebuild the firmware and re-bake it into the built simulation (--board xcu1525).")
    parser.add_argument("--gate",           action="store_true",                                   help="Traced generation compared with the Python mirror.")
    parser.add_argument("--bench",          action="store_true",                                   help="Cycles per forward with --bench-toggle off, then on.")
    parser.add_argument("--bench-toggle",   default="resident",                                    help="--bench: console toggle to A/B (resident, silu, dnstream).")
    parser.add_argument("--bench-tokens",   default=2,  type=int,                                  help="--bench: bench N (N+2 forwards).")
    parser.add_argument("--cpu-type",       default=None, choices=["vexriscv", "vexiiriscv"],      help="Override the board's CPU.")
    parser.add_argument("--update-repo",    default="no", choices=["no", "recommended", "latest"], help="VexiiRiscv source update policy.")
    parser.add_argument("--trace",          action="store_true",                                   help="Enable FST tracing.")
    parser.add_argument("--hbm-latency",    default=16, type=int,                                  help="--board fk33: HBM model read latency.")
    parser.add_argument("--hbm-stall-rate", default=0,  type=int,                                  help="--board fk33: HBM model stall rate.")
    parser.add_argument("--threads",        default=4,  type=int,                                  help="Verilator threads.")
    args = parser.parse_args()
    if not (args.build or args.run or args.firmware or args.gate or args.bench):
        parser.error("nothing to do: --build, --run, --firmware, --gate or --bench.")

    # Reproducible builds: the generated sources follow set/dict orders, hence the hash seed.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Everything runs from the repository root (firmware/, models/, build/).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_dir      = args.out_dir or f"build/sim_{args.board}"
    gateware_dir = os.path.join(out_dir, "gateware")
    board        = dict(BOARDS[args.board])
    if args.cpu_type is not None:
        board.update(cpu_type=args.cpu_type, icache_kb=0, vexii_extra="")

    # Build.
    if args.build:
        build_sim(args, board, out_dir)

    # Firmware.
    if args.firmware:
        if board["mem"] != "ddr4":
            raise SystemExit("--firmware re-bakes the DDR4 model memories (--board xcu1525): use --build.")
        build_firmware(out_dir)
        rebake_firmware(gateware_dir, args.image, FIRMWARE, board["n_memctrl"])

    # Run.
    if args.run:
        if not os.path.exists(os.path.join(gateware_dir, "obj_dir", "Vsim")):
            raise SystemExit(f"no simulator in {gateware_dir}: build it first (--build).")
        sys.exit(subprocess.call(["obj_dir/Vsim"], cwd=gateware_dir))

    # Gate / Bench.
    from software.lab import simgate
    if args.gate:
        rc = simgate.gate(gateware_dir, args.image, os.path.join(out_dir, "gate"),
            expect_silu=board.get("with_silu", False))
        if rc:
            sys.exit(rc)
    if args.bench:
        sys.exit(simgate.bench(gateware_dir, os.path.join(out_dir, "bench"),
            toggle=args.bench_toggle, tokens=args.bench_tokens))

if __name__ == "__main__":
    main()

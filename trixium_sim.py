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
one HBM channel model per engine channel and no scratchpad units, --board ypcb the YPCB's two
DDR3 controllers (the board's geometry, timings, PHY latencies and clock ratio) with the FK33's
units. The firmware binary and the
BONS header+table are baked into main_ram at elaboration, the channel payloads into the memory
models.

  ./trixium_sim.py --build      # headers + libraries, firmware, Verilog (~1 min), Verilator (~2.5 min)
  ./trixium_sim.py --run        # run it (console on stdin/stdout)
  ./trixium_sim.py --firmware   # rebuild the firmware and re-bake it into the built sim (seconds)
  ./trixium_sim.py --gate       # traced generation compared with the Python mirror (~10 min)
  ./trixium_sim.py --bench      # resident path off/on, cycles per forward

--board (xcu1525, fk33, ypcb) and --image select the build (build/sim_<board>); the gate and the bench
must use the image the simulation was built with.

--board-timing (xcu1525) builds the board-faithful flavor for the shape-true slices (models from
`python -m model.tinygen --slice <name> --channels 16`): the board's ddr:sys clock ratio, DDR4
timings, bank map, channel skew, CPU and stream geometry; its --bench predicts the board's
cycles per token (doc/simulation.md, software/lab/simcal.py):

  ./trixium_sim.py --board-timing --image models/slices/bonsai-27b-slice-n16.img --build
  ./trixium_sim.py --board-timing --image models/slices/bonsai-27b-slice-n16.img --bench
"""

import os
import sys
import glob
import argparse
import subprocess

import numpy as np

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
from litex.soc.interconnect.csr import CSR, CSRStatus, CSRStorage
from litex.soc.interconnect.axi import AXIInterface

from litedram.modules import MT40A512M8
from litedram.phy.model import SDRAMPHYModel

from trixium_ypcb import YPCBDDR3, DDR3_CL, DDR3_CWL, ddr_controller_settings
from trixium_ypcb import VEXII_ARGS as YPCB_VEXII_ARGS, YPCB_HYBRID_UNIT_OPTIONS, YPCB_HYBRID_ATTN

from model.layout import HDR_CPU_OFFSET, KIND_TERNARY5
from gateware.core.core import TriXiumConfig, TriXiumCore
from gateware.memory.ddr import add_ddr_domain_controllers, engine_ddr_port, EnginePortStats, NativePortStats
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes
from gateware.memory.ckphase_model import ckphase_model_settings, CKPhasePHYModel, NoDMPHYModel
from model.bons import BONSReader
from gateware.common import set_main_ram_executable

# Constants ----------------------------------------------------------------------------------------

MAIN_RAM_BASE = 0x4000_0000
MAIN_RAM_SIZE = 0x0080_0000    # 8 MB (the HBM flavor and the default DDR4 map).
HDR_OFFSET    = HDR_CPU_OFFSET # BONS header+table at main_ram + 4 MB (shared with hardware).

DDR4_SIM_CSR_WINDOW = 0xF0020000 # The target's DDR_CSR_WINDOW.
DDR4_SIM_CSR_STRIDE = 0x1000

# The YPCB's DDR3 flavor: two LiteDRAM controllers on LiteDRAM's PHY model with the K7DDRPHY's
# latencies (CK phase mode), the board's DDR3-1066 timings and controller settings, and the ddr
# clock at 8/9 of sys (the board's 133.33/150 MHz). The geometry is the board's (8 banks x 1024
# columns: 8 KB rows, the bank in byte address bits 13..15) with 2048 rows, 128 MB per controller,
# so the windows' 32 KB bank skew is the board's own. Windows from 12 MB at a power-of-two stride
# sized for the image (4 MB for the tiny models; a truncated 1.7B for the performance runs needs
# 32 MB).
DDR3_SIM_WEIGHT_BASE = 0x00C0_0000 # 12 MB.
DDR3_SIM_BANK_CYCLE  = 0x10000     # 8 banks x 8 KB rows: the windows' skew is 1/chan_mod of it.
DDR3_SIM_DDR_PERIOD  = 9000        # ps: the sim's clocks are integer picoseconds, ddr at 9 ns and
                                   # sys at the board's ratio (8 ns for 150 MHz, ddr3_sim_clocks()).
DDR3_SIM_FREQ        = 400e6/3     # The controllers' timings: the board's DDR3-1066.
YPCB_ENGINE_RMW      = True        # The board's DDR3 has no DM pins (trixium_ypcb.ENGINE_RMW).
DDR3_SIM_SIZE        = 128 << 20   # Per controller, at least (2048 rows; more for big images).

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


class BoardDDR4(MT40A512M8):
    # The board's part with its own banks and columns (bank bits at byte bits 13..16, so the
    # 32 KB channel skew steps the bank group as on the XCU1525), rows reduced to 4096: 512 MB
    # per x64 controller.
    nrows = 4096


class DDR4Map:
    """The DDR4 flavor's memory map and clocks: the XCU1525's memory plumbing on LiteDRAM's PHY
    model - main_ram on controller 0, the engine channels on crossbar ports behind AXI bridges,
    chan_mod channels per controller at these windows (the PHY model needs no init sequence: the
    firmware skips ddr_init_all()/ddr_test() under TRIXIUM_SIM, its console still runs them).

    sim (the default, the gate's): main_ram 8 MB (0..8 MB of controller 0), the windows from
    12 MB so the header staging area (window - 1 MB) and the round-trip test (window - 2 MB) stay
    above main_ram, 4 MB per window for the tiny payload and its KV. The channel skew is a bank
    skew IN THE SIM'S GEOMETRY (ncols 128 puts the bank bits at byte bits 10..13, so the board's
    32 KB would be a row bit here): the address arithmetic is what this map checks. The DDR4
    timings are for 100 MHz, the ddr clock is 2x sys.

    board (--board-timing): what sets the board's cycles - the ddr clock 1.5x sys (187.5 vs
    125 MHz), the DDR4 timings at 187.5 MHz (the target's MT40A512M8 at 1:4), the board's
    bank/column map and 32 KB channel skew; main_ram 32 MB (the shape-true slices' CPU tables),
    64 MB windows from 64 MB for their payloads, KV and DeltaNet state (the firmware sizes a
    window up to the next stride boundary: the base is one)."""
    def __init__(self, module, ddr_clk_freq, sim_clk_freqs, main_ram_size, weight_base,
                 chan_stride, chan_skew, reserve):
        self.module        = module
        self.ddr_clk_freq  = ddr_clk_freq  # The controllers' timing frequency.
        self.sim_clk_freqs = sim_clk_freqs # (sys, ddr) sim clockers: the ratio is what matters.
        self.main_ram_size = main_ram_size
        self.weight_base   = weight_base
        self.chan_stride   = chan_stride
        self.chan_skew     = chan_skew
        self.reserve       = reserve       # Room above a payload (KV, recurrent state).
        assert weight_base % chan_stride == 0, "the firmware ends a window at a stride boundary"

SIM_DDR4_MAP = DDR4Map(SimDDR4, 100e6,
    sim_clk_freqs = (int(1e6), int(2e6)),
    main_ram_size = MAIN_RAM_SIZE,
    weight_base   = 0x00C0_0000, # 12 MB.
    chan_stride   = 0x0040_0000, # 4 MB.
    chan_skew     = 0xA000,      # 40 KB.
    reserve       = 512*1024)

# The sim clockers' periods must be whole picoseconds with a common half-period timebase:
# 1.2 us / 0.8 us is 125:187.5 exactly (6 evaluations per sys cycle, the default's 4).
BOARD_DDR4_MAP = DDR4Map(BoardDDR4, 187.5e6,
    sim_clk_freqs = (833333, 1250000),
    main_ram_size = 0x0200_0000, # 32 MB.
    weight_base   = 0x0400_0000, # 64 MB.
    chan_stride   = 0x0400_0000, # 64 MB.
    chan_skew     = 0x8000,      # 32 KB: the target's chan_skew.
    reserve       = 4 << 20)


def main_ram_content(img, firmware=None, size=MAIN_RAM_SIZE):
    """main_ram: the firmware binary at 0, the BONS header+table at HDR_OFFSET."""
    ram = bytearray(size)
    if firmware is not None:
        assert len(firmware) < HDR_OFFSET
        ram[:len(firmware)] = firmware
    if img is not None:
        meta = bytes(img.data[:img.header["payload_offset"]])
        ram[HDR_OFFSET:HDR_OFFSET + len(meta)] = meta
    return ram


def ddr4_content(k, ram, img, chan_mod, dmap=SIM_DDR4_MAP):
    """DDR4 controller k: main_ram at 0 for controller 0 and, for all, its channels' payloads at
    their windows (up to the last payload byte: the rest of the model reads as zeros)."""
    content = bytearray(dmap.weight_base + chan_mod*dmap.chan_stride)
    end     = len(ram) if k == 0 else 0
    if k == 0:
        content[:len(ram)] = ram
    if img is not None:
        for j in range(chan_mod):
            c = k*chan_mod + j
            payload = bytes(img.channel_payload(c))
            base = dmap.weight_base + j*dmap.chan_stride + j*dmap.chan_skew
            assert len(payload) + dmap.reserve + (chan_mod - 1)*dmap.chan_skew <= dmap.chan_stride, \
                f"channel {c}: {len(payload)} B does not fit a sim window"
            content[base:base + len(payload)] = payload
            end = max(end, base + len(payload))
    return content[:end]


def ddr3_size(chan_mod, stride):
    """The sim's DDR3 per controller: a power of two holding the windows (128 MB at least: the
    PHY model needs 11 row address bits for the A10 precharge-all decode)."""
    size = DDR3_SIM_SIZE
    while size < DDR3_SIM_WEIGHT_BASE + chan_mod*stride:
        size *= 2
    return size


def sim_ddr3(size):
    """The board's part (8 banks x 1024 columns, 8 KB rows on 64 bits) with the rows of `size`."""
    return type("SimDDR3", (YPCBDDR3,), {"nrows": size//(8*1024*8)})


# The fly-by channel of the calibration's simulation (--ckphase-model): crossings ordered along the
# chain, latencies 0 and 6 (-2), one bit of lane 3 framed a CLK late; the common window is
# +9..-34 across the tCK boundary (the scan's stride-28 samples settle at -56).
CKPHASE_CHANNEL = dict(
    cross   = [-30, -25, -20, -15, -10, -5, 0, 5],
    base_wl = [0, 0, 6, 6, 0, 0, 6, 0],
    v0      = [40 + (b % 8) + (24 if b == 27 else 0) for b in range(64)],
)


def ddr3_sim_clocks(sys_clk_freq=125e6):
    """(sys, ddr) clocker frequencies of the YPCB sim: the board's sys/ddr ratio on integer
    picosecond periods (the clocker's timebase is their half periods' GCD)."""
    sys_period = 100*round(DDR3_SIM_DDR_PERIOD*DDR3_SIM_FREQ/sys_clk_freq/100) # 0.1 ns steps.
    return int(1e12//sys_period), int(1e12//DDR3_SIM_DDR_PERIOD)


def ddr3_skew(chan_mod):
    """The board's bank skew: the chan_mod windows of a controller evenly spread over the 8 banks
    (32 KB, four banks apart, for two)."""
    return DDR3_SIM_BANK_CYCLE//chan_mod


def ddr3_stride(img):
    """The YPCB sim's window stride: a power of two holding the largest channel payload, its KV
    cache (the converter's sizing), the firmware's round-trip test page and the bank skew (4 MB
    at least)."""
    stride = 0x40_0000
    if img is not None:
        h    = img.header
        n    = h["n_channels"]
        kv   = h["n_layer"]*h["n_kv_head"]*2*((h["max_ctx"] + n - 1)//n)*(1 + h["head_dim"]//32)*32
        need = max(len(img.channel_payload(c)) for c in range(n)) + kv + 512*1024
        while stride < need + DDR3_SIM_BANK_CYCLE:
            stride *= 2
    return stride


def ddr3_content(k, ram, img, chan_mod, stride, size):
    """DDR3 controller k: main_ram at 0 for controller 0 and, for all, its channels' payloads at
    their windows (the content ends with the last payload: the model's banks start zeroed)."""
    content = bytearray(ram) if k == 0 else bytearray()
    if img is not None:
        for j in range(chan_mod):
            c = k*chan_mod + j
            payload = bytes(img.channel_payload(c))
            base = DDR3_SIM_WEIGHT_BASE + j*stride + j*ddr3_skew(chan_mod)
            assert base + stride - j*ddr3_skew(chan_mod) <= size, \
                f"channel {c}: a {stride >> 20} MB window does not fit the sim's DDR3"
            if len(content) < base + len(payload):
                content.extend(bytes(base + len(payload) - len(content)))
            content[base:base + len(payload)] = payload
    content.extend(bytes(-len(content) % 64))
    return content


def ddr3_phy(content, size, mapping="ROW_BANK_COL"):
    """The YPCB sim's DDR3 PHY model holding `content`, its banks filled as the controller's
    address mapping places the words (a model filled in another mapping than the controller's
    boots garbage: the firmware sits where the other mapping puts it)."""
    module   = sim_ddr3(size)(DDR3_SIM_FREQ, "1:4")
    settings = ckphase_model_settings(DDR3_SIM_FREQ, 64, DDR3_CL, DDR3_CWL)
    phy      = SDRAMPHYModel(module=module, settings=settings, clk_freq=DDR3_SIM_FREQ,
                             init=words(content), address_mapping=mapping)
    return phy, module


def words(content):
    content = bytes(content) + bytes(-len(content) % 4)
    return np.frombuffer(content, dtype="<u4").tolist()


def bank_words(content, module, mapping="ROW_BANK_COL"):
    """The init words of each bank of a DDR4/DDR3 PHY model holding `content`: 512-bit words
    (4 phases x 2 x 64 bits, little-endian) placed as the controller's address mapping does
    (ROW_BANK_COL: a bank's row, ncols x 8 bytes, at a time)."""
    assert mapping == "ROW_BANK_COL", f"{mapping}: the model's init layout is ROW_BANK_COL's"
    g      = module.geom_settings
    nbanks = 2**g.bankbits
    span   = (2**g.colbits)*8
    data   = bytes(content) + bytes(-len(content) % (nbanks*span))
    rows   = np.frombuffer(data, dtype=np.uint8).reshape(-1, nbanks, span)
    inits  = []
    for b in range(nbanks):
        flat = rows[:, b, :].tobytes()
        inits.append([int.from_bytes(flat[i:i+64], "little") for i in range(0, len(flat), 64)])
    return inits


def bank_memories(phy):
    """The bank Memories of a DDR4 PHY model, in bank order (BankModels are anonymous
    submodules: their Memory specials in creation order)."""
    mems = []
    for _, sub in phy._submodules:
        mems += [m for m in sub._fragment.specials if isinstance(m, Memory)]
    return mems


def ddr4_phy(content, dmap=SIM_DDR4_MAP):
    """LiteDRAM's DDR4 PHY model holding `content`, its banks' init computed here (numpy: the
    model's own 32-bit word path takes minutes on the slices' hundreds of MB)."""
    module = dmap.module(int(dmap.ddr_clk_freq), "1:4")
    phy    = SDRAMPHYModel(module=module, data_width=64, clk_freq=int(dmap.ddr_clk_freq))
    if content:
        mems = bank_memories(phy)
        assert len(mems) == 2**module.geom_settings.bankbits
        for m, init in zip(mems, bank_words(content, module)):
            m.init = init
    return phy, module

class ReaderStats(LiteXModule):
    """Per channel reader, at its output to the channel pipe (the sim's instrumentation): the
    beats delivered, the cycles a beat waited on the pipe (valid, not ready), the cycles
    without a beat to offer (the reader's FIFO empty). `snap` latches and restarts, `sel`
    picks the channel (as EnginePortStats)."""
    def __init__(self, readers):
        n = len(readers)
        self.snap    = CSR()
        self.sel     = CSRStorage(max(1, (n - 1).bit_length()))
        self.cycles  = CSRStatus(32)
        self.beats   = CSRStatus(32)
        self.stalled = CSRStatus(32)
        self.starved = CSRStatus(32)
        cycles = Signal(32)
        self.sync += If(self.snap.re, self.cycles.status.eq(cycles), cycles.eq(0)).Else(cycles.eq(cycles + 1))
        holds = {name: [] for name in ("beats", "stalled", "starved")}
        for r in readers:
            for name, expr in (("beats",   r.source.valid & r.source.ready),
                               ("stalled", r.source.valid & ~r.source.ready),
                               ("starved", ~r.source.valid)):
                inc, count, hold = Signal(), Signal(32), Signal(32)
                self.comb += inc.eq(expr)
                self.sync += If(self.snap.re, hold.eq(count), count.eq(0)).Else(count.eq(count + inc))
                holds[name].append(hold)
        for name, regs in holds.items():
            self.comb += getattr(self, name).status.eq(Array(regs)[self.sel.storage])

# TriXiumSimSoC ------------------------------------------------------------------------------------

class TriXiumSimSoC(SoCCore):
    def __init__(self, image=None, firmware=None, hbm_latency=16, hbm_stall_rate=0,
                 cpu_type="vexriscv", icache_kb=0, vexii_extra="", reader_bursts=8, pipe_clk_freq=0,
                 mem="hbm", n_memctrl=4, update_repo="no",
                 with_hadamard=False, with_activation_quantize=False,
                 with_projection_restore=False, with_silu=False, with_norm=False, with_conv=False,
                 with_softmax=False,
                 with_dnstream=False, attn_geometry=None, board_timing=False, contents=True,
                 ddr_settings=None, ckphase_model=False, prefetch=0, vec_fused=False,
                 unit_options=()):
        assert mem in ("hbm", "ddr4", "ddr3")
        assert not board_timing or mem == "ddr4", "board timing is the XCU1525's (DDR4)"
        ddr_split = mem in ("ddr4", "ddr3")
        dmap      = BOARD_DDR4_MAP if board_timing else SIM_DDR4_MAP
        platform  = Platform()
        self.comb += platform.trace.eq(1)

        # main_ram init: firmware at 0, BONS header+table at HDR_OFFSET (contents=False: the
        # constants only, for the headers the firmware is built against).
        fw = None
        if firmware is not None and os.path.exists(firmware):
            with open(firmware, "rb") as f:
                fw = f.read()
        img = BONSReader(image) if image is not None else None
        ram = main_ram_content(img, fw, size=dmap.main_ram_size if ddr_split else MAIN_RAM_SIZE)

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
            # Fetch cache geometry (icache_kb=None: the extra args carry the board's caches).
            if icache_kb is not None:
                sets = max(16, (icache_kb*1024)//(4*64)) if icache_kb else 64
                VexiiRiscv.vexii_args = (VexiiRiscv.vexii_args
                                         .replace("--fetch-l1-ways=2", "--fetch-l1-ways=4")
                                         + f" --fetch-l1-sets={sets}")
            print(f"[trixium] VexiiRiscv args: {VexiiRiscv.vexii_args}")

        # SoCCore ----------------------------------------------------------------------------------
        # Board timing declares the board's sys clock (the firmware's tok/s), the sim clockers
        # only keep the ddr:sys ratio.
        SoCCore.__init__(self, platform,
            clk_freq                 = int(BOARD_SYS_CLK_FREQ) if board_timing else int(1e6),
            ident                    = "TriXium Sim" + {"hbm": "", "ddr4": " DDR4", "ddr3": " DDR3"}[mem],
            cpu_type                 = cpu_type,
            integrated_rom_size      = 0x20000,
            integrated_rom_init      = ROM_BOOT_STUB,
            # The YPCB's 16 KB (trixium_ypcb.py): liblitedram's CK phase leveling state in .bss.
            integrated_sram_size     = 0x4000 if mem == "ddr3" else 0x2000,
            integrated_main_ram_size = MAIN_RAM_SIZE if mem == "hbm" else 0,
            integrated_main_ram_init = words(ram) if mem == "hbm" else [],
            uart_name                = "sim")

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(platform.request("sys_clk"))
        if ddr_split:
            # The controllers in their own `ddr` domain on a second sim clock (the map's ddr:sys
            # ratio: 2 by default, the board's 1.5 with board timing), as on the XCU1525.
            platform.add_extension([("ddr_clk", 0, Pins(1))])
            self.cd_ddr = ClockDomain()
            self.comb += self.cd_ddr.clk.eq(platform.request("ddr_clk"))
            self.specials += AsyncResetSynchronizer(self.cd_ddr, ResetSignal("sys"))
        if pipe_clk_freq:
            # The channel pipes' clock (PipeIsland) on a third sim clock.
            platform.add_extension([("pipeclk", 0, Pins(1))])
            self.cd_pipeclk = ClockDomain("pipeclk")
            self.comb += self.cd_pipeclk.clk.eq(platform.request("pipeclk"))
            self.specials += AsyncResetSynchronizer(self.cd_pipeclk, ResetSignal("sys"))

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
        elif mem == "ddr4":
            chan_mod = self.add_ddr4(n_channels, n_memctrl, ram, img, memory_ports, dmap,
                                     contents)
        else:
            stride   = ddr3_stride(img)
            chan_mod = self.add_ddr3(n_channels, n_memctrl, ram, img, memory_ports, stride,
                ddr_settings or {}, ckphase_model)

        # TriXium ----------------------------------------------------------------------------------
        # Same integration as both hardware targets; only memory/physical policy differs.
        config = TriXiumConfig(with_sp_csr=False,
            chan_stride              = stride if mem == "ddr3" else dmap.chan_stride if mem == "ddr4" else 0,
            chan_mod                 = chan_mod,
            chan_skew                = ddr3_skew(chan_mod) if mem == "ddr3" else dmap.chan_skew if mem == "ddr4" else 0,
            with_hadamard            = with_hadamard,
            with_activation_quantize = with_activation_quantize,
            with_projection_restore  = with_projection_restore,
            with_silu                = with_silu,
            with_norm                = with_norm,
            with_conv                = with_conv,
            with_softmax             = with_softmax,
            with_dnstream            = bool(with_dnstream),
            with_axpy                = not with_dnstream,  # the hybrid path never issues the weighted-V op
            pipe_clk                 = "pipeclk" if pipe_clk_freq else "",
            with_attnstream          = attn_geometry is not None,
            with_prefetch            = bool(prefetch),
            with_vec_fused           = vec_fused,
            reader_fifo_depth        = prefetch or 512,
            reader_max_bursts        = reader_bursts,
            # An image with 5-trits-per-byte weights needs the unpacker in front of the pipes.
            with_trit5               = img is not None and any(
                e["kind"] == KIND_TERNARY5 for e in img.entries),
            attnstream_options       = tuple(zip(("lanes", "groups", "hb_max", "t_max"),
                (attn_geometry[0], attn_geometry[1], attn_geometry[2]//8, attn_geometry[3])))
                if attn_geometry else (),
            **dict(unit_options)) # Unit sizing (TriXiumConfig fields), as the board target's.
        self.trixium = TriXiumCore(memory_ports, config)
        self.rdstat  = ReaderStats(self.trixium.engine.readers)   # console `ddrstat`: the readers' output
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
        elif mem == "ddr3":
            # The YPCB's constants, with the sim's windows.
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", DDR3_SIM_WEIGHT_BASE)
            self.add_constant("TRIXIUM_CHAN_STRIDE",      stride)
            self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
            self.add_constant("TRIXIUM_CHAN_SKEW",        ddr3_skew(chan_mod))
            self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
            self.add_constant("TRIXIUM_N_MEMCTRL",        n_memctrl)
            # Per controller: sdram, ddrphy, (generator, checker with the CK phase model,) stats.
            self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   self.csr.paging*(5 if ckphase_model else 3))
            self.add_constant("SDRAM_TEST_DISABLE")
            if ckphase_model:
                # The board's calibration (liblitedram's CK phase leveling, SDRAM_PHY_CKPHASE_STEPS
                # from the model's settings) on the PHY model: `ddrinit 1` on the console. Scan
                # stride and BIST sizes cut to simulation time.
                self.add_constant("SDRAM_CSR_STRIDE",               self.csr.paging*5)
                self.add_constant("SDRAM_CKPHASE_SCAN_STRIDE",      28)
                self.add_constant("SDRAM_CKPHASE_BIST_STEP_BYTES",  0x10000)
                self.add_constant("SDRAM_CKPHASE_BIST_FINAL_BYTES", 0x40000)
        else:
            # The XCU1525's geometry, scaled to the simulated memories: the same constants the
            # firmware reads on the board, at the map's windows (DDR4Map).
            self.add_constant("TRIXIUM_CHAN_WEIGHT_BASE", dmap.weight_base)
            self.add_constant("TRIXIUM_CHAN_STRIDE",      dmap.chan_stride)
            self.add_constant("TRIXIUM_CHAN_MOD",         chan_mod)
            self.add_constant("TRIXIUM_CHAN_SKEW",        dmap.chan_skew)
            self.add_constant("TRIXIUM_MEM_CPU_OFFSET",   0)
            self.add_constant("TRIXIUM_N_MEMCTRL",        n_memctrl)
            self.add_constant("TRIXIUM_DDR_CSR_STRIDE",   DDR4_SIM_CSR_STRIDE)
            self.add_constant("SDRAM_TEST_DISABLE") # sdram_init()'s memtest would erase the preload.
        self.add_constant("TRIXIUM_N_CHANNELS",       n_channels)
        if cpu_type == "vexiiriscv" and "--with-rvZbb" in vexii_extra:
            self.add_constant("TRIXIUM_HAS_ZBB") # As the board target's add_zbb_constant.
        # Simulation-only firmware paths (boot shortcuts, boot-phase cycle marks): the board
        # targets never define it.
        self.add_constant("TRIXIUM_SIM")

    def add_ddr4(self, n_channels, n_memctrl, ram, img, memory_ports, dmap, contents=True):
        """The XCU1525's memory system on LiteDRAM's PHY model at the map `dmap`; returns
        chan_mod."""
        assert n_channels % n_memctrl == 0, "channels shard evenly over the controllers"
        chan_mod = n_channels//n_memctrl
        assert chan_mod & (chan_mod - 1) == 0

        # Controllers.
        self.sdrams, split_phys, split_modules = [], [], []
        for k in range(n_memctrl):
            content     = ddr4_content(k, ram, img, chan_mod, dmap) if contents else b""
            phy, module = ddr4_phy(content, dmap)
            split_phys.append(phy)
            split_modules.append(module)
        add_ddr_domain_controllers(self, split_phys, split_modules, int(dmap.ddr_clk_freq),
                                   dmap.main_ram_size, 8192, DDR4_SIM_CSR_WINDOW)

        # Engine ports: channel c -> controller c // chan_mod.
        for k in range(n_memctrl):
            core = self.sdrams[k]
            for j in range(chan_mod):
                c = k*chan_mod + j
                axi = AXIInterface(data_width=256, address_width=33, id_width=6)
                for jj, m in enumerate(engine_ddr_port(core, axi, clock_domain="sys_port",
                                                       rmw=YPCB_ENGINE_RMW)):
                    setattr(self, f"engport{c}_{jj}", m)
                memory_ports.append(axi)
        self.engstat = EnginePortStats(memory_ports)   # console `ddrstat`: the ports' waits
        set_main_ram_executable(self, MAIN_RAM_BASE)
        return chan_mod

    def add_ddr3(self, n_channels, n_memctrl, ram, img, memory_ports, stride, ddr_settings,
                 ckphase_model=False):
        """The YPCB's memory system on LiteDRAM's PHY model; returns chan_mod.

        ckphase_model: the PHY models behind the K7DDRPHY's CSRs in CK phase mode (CKPhasePHYModel),
        with a LiteDRAM BIST per controller: controller 0 (main_ram, the firmware) without channel effects,
        controller 1 as a fly-by channel to calibrate (CKPHASE_CHANNEL)."""
        assert n_channels % n_memctrl == 0, "channels shard evenly over the controllers"
        chan_mod = n_channels//n_memctrl
        assert chan_mod & (chan_mod - 1) == 0

        # Controllers (the board's controller settings unless ddr_settings overrides them; the
        # memory models are filled in the controllers' address mapping).
        controller_settings = ddr_controller_settings(**ddr_settings) # A factory (one per core).
        mapping = controller_settings().address_mapping
        self.sdrams, split_phys, split_modules = [], [], []
        for k in range(n_memctrl):
            size = ddr3_size(chan_mod, stride)
            phy, module = ddr3_phy(ddr3_content(k, ram, img, chan_mod, stride, size), size, mapping)
            # The board has no DM pins: neither model lets a write mask reach the memory.
            if ckphase_model:
                phy = CKPhasePHYModel(phy, ideal=(k == 0), **CKPHASE_CHANNEL)
            else:
                phy = NoDMPHYModel(phy)
            split_phys.append(phy)
            split_modules.append(module)
        add_ddr_domain_controllers(self, split_phys, split_modules, DDR3_SIM_FREQ, MAIN_RAM_SIZE,
            8192, DDR4_SIM_CSR_WINDOW, controller_settings=controller_settings,
            with_bist=ckphase_model, with_stats=True)

        # Engine ports: channel c -> controller c // chan_mod.
        taps, native_taps = [], []
        for k in range(n_memctrl):
            core = self.sdrams[k]
            for j in range(chan_mod):
                c = k*chan_mod + j
                axi = AXIInterface(data_width=256, address_width=32, id_width=6)
                for jj, m in enumerate(engine_ddr_port(core, axi, clock_domain="sys_port",
                                                       rmw=YPCB_ENGINE_RMW, taps=taps,
                                                       native_taps=native_taps)):
                    setattr(self, f"engport{c}_{jj}", m)
                memory_ports.append(axi)
        self.engstat  = EnginePortStats(memory_ports)  # console `ddrstat`: the ports' waits
        self.widestat = EnginePortStats(taps)          # ... the 512-bit words the crossing delivers
        self.natstat  = NativePortStats(native_taps)   # ... and the native ports' commands
        set_main_ram_executable(self, MAIN_RAM_BASE)
        return chan_mod

# Firmware Re-bake --------------------------------------------------------------------------------

def bank_inits(content, dmap=SIM_DDR4_MAP):
    """The init lines of each bank Memory of a DDR4 PHY model holding `content`."""
    phy, _ = ddr4_phy(content, dmap)
    inits  = []
    for m in bank_memories(phy):
        if m.init is not None and len(m.init):
            width = (m.width + 3)//4
            inits.append([f"{v:0{width}x}" for v in m.init])
    return inits


def ddr3_bank_inits(content, size, mapping="ROW_BANK_COL"):
    """The init lines of each bank Memory of the YPCB sim's DDR3 PHY model holding `content`
    (the controller's address mapping's layout of 512-bit words, as bank_words)."""
    module = sim_ddr3(size)(DDR3_SIM_FREQ, "1:4")
    nbanks = 2**module.geom_settings.bankbits
    col    = 2**module.geom_settings.colbits*8//64 # 512-bit words per bank row.
    # The model stops at the content's end, padded as SDRAMPHYModel pads its init: to a multiple
    # of data_width_bytes 32-bit words (64 words = 256 bytes = four 512-bit words here).
    n      = -(-len(content)//256)*4
    inits  = []
    for b, words in enumerate(bank_words(content, module, mapping)):
        keep = sum(max(0, min(col, n - (r*nbanks + b)*col)) for r in range(len(words)//col))
        inits.append([f"{v:0128x}" for v in words[:keep]])
    return inits


def rebake_firmware(gateware_dir, image, firmware, n_memctrl, dmap=SIM_DDR4_MAP, mem="ddr4",
                    mapping="ROW_BANK_COL"):
    """Rewrite the bank init files holding the firmware in a built DDR4/DDR3 simulation (seconds).

    The firmware lives in controller 0's memory model: one sim_mem_N.init per bank, read at
    simulation start ($readmemh). Controller 0's contents are rebuilt as the SoC does, each bank's
    file is found by matching every line outside the firmware area, and rewritten."""
    img      = BONSReader(image)
    chan_mod = img.header["n_channels"]//n_memctrl
    with open(firmware, "rb") as f:
        fw = f.read()
    def inits(fw):
        ram = main_ram_content(img, fw, size=dmap.main_ram_size)
        if mem == "ddr3":
            stride = ddr3_stride(img)
            size   = ddr3_size(chan_mod, stride)
            return ddr3_bank_inits(ddr3_content(0, ram, img, chan_mod, stride, size), size,
                                   mapping)
        return bank_inits(ddr4_content(0, ram, img, chan_mod, dmap), dmap)
    new   = inits(fw)
    blank = inits(None)
    # Every line a firmware could occupy (the whole window below the header), so a smaller
    # firmware than the one baked in still matches: the old firmware's tail is replaced.
    full  = inits(b"\xff"*(HDR_OFFSET - 4))
    assert len(new) == len(blank) and new, "no bank memories found"

    files = {}
    sizes = {len(n)*(len(n[0]) + 1) for n in new}
    for filename in glob.glob(os.path.join(gateware_dir, "sim_mem*.init")):
        if os.path.getsize(filename) not in sizes:
            continue # Not a bank of controller 0 (one line per word, newline-terminated).
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
    # The YPCB: two DDR3 controllers, the FK33's units (the Qwen3 path), the board target's CPU
    # (caches, branch prediction, Zbb: the per-op CPU work between the lists is board-true).
    "ypcb": dict(mem="ddr3", n_memctrl=2,
        cpu_type="vexiiriscv", icache_kb=None, vexii_extra=YPCB_VEXII_ARGS),
}

# --hybrid (--board ypcb): the YPCB with the hybrid models' units (Ternary-Bonsai-2 2B/4B/9B fit its
# 4 GB): every scratchpad unit, the DeltaNet and attention streams, and an FPU for the CPU's share
# of the hybrid forward (the XCU1525's set; the attention stream at the tiny models' geometry), sized
# as trixium_ypcb.py's YPCB_HYBRID_UNIT_OPTIONS (the K7-480T's area: narrower Hadamard and SiLU
# units, no vector unit - the hybrid path never runs it - which pays for unit 12's RoPE stage).
HYBRID_UNITS = dict(with_hadamard=True, with_activation_quantize=True, with_projection_restore=True,
    with_silu=True, with_norm=True, with_conv=True, with_softmax=True, with_dnstream=True,
    attn_geometry=YPCB_HYBRID_ATTN + (128, 256), unit_options=YPCB_HYBRID_UNIT_OPTIONS)

# The image each board's simulation bakes in by default (--image).
DEFAULT_IMAGES = {
    "xcu1525": "models/tiny/bonsai-tiny-hybrid-1k-n8.img",
    "fk33":    "models/tiny/bonsai-tiny-hybrid-1k-n8.img",
    "ypcb":    "models/tiny/bonsai-tiny-n4.img", # Four engine channels, the Qwen3 path.
}

BOARD_SYS_CLK_FREQ = 125e6

def board_timing_preset():
    """--board-timing: the xcu1525 preset at the XCU1525 target's own defaults - its CPU
    (caches, branch prediction, Zbb), its attention-stream geometry - on the board DDR4 map
    (BOARD_DDR4_MAP: clocks, DDR4 timings, bank map, channel skew, checked against the target).
    The DeltaNet stream has one lane per engine channel: 16 with a 16-channel image."""
    import inspect
    import trixium_xcu1525 as target
    d = {k: p.default for k, p in inspect.signature(target.TriXiumXCU1525SoC.__init__).parameters.items()}
    dmap = BOARD_DDR4_MAP
    sys_hz, ddr_hz = dmap.sim_clk_freqs
    assert d["sys_clk_freq"] == BOARD_SYS_CLK_FREQ and d["ddr_clk_freq"] == dmap.ddr_clk_freq
    assert int(1e12/sys_hz)*d["sys_clk_freq"] == int(1e12/ddr_hz)*d["ddr_clk_freq"], "ddr:sys ratio"
    assert d["chan_skew"] == dmap.chan_skew
    return dict(BOARDS["xcu1525"], board_timing=True, attn_geometry=d["attn_geometry"],
        icache_kb=None, vexii_extra=target.VEXII_ARGS, engine_channels=d["engine_channels"])

FIRMWARE = "firmware/trixium.bin"


def build_firmware(out_dir):
    # BUILD_DIR relative to firmware/, as documented (make -C firmware BUILD_DIR=../build/<name>):
    # the firmware's build stamp then sees the same tree and keeps its objects.
    build_dir = os.path.relpath(os.path.abspath(out_dir), "firmware")
    subprocess.run(["make", "-C", "firmware", f"BUILD_DIR={build_dir}", f"PYTHON={sys.executable}"],
        check=True)


def ddr_settings_of(arg):
    """--ddr-settings k=v,...: the controller settings overrides (integers where they parse)."""
    if not arg:
        return {}
    return {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in
            (kv.split("=") for kv in arg.split(","))}


def build_sim(args, board, out_dir):
    board = dict(board)
    board.pop("engine_channels", None)
    # LiteX's sim Makefile compiles the verilated C++ with make -j$(JOBS): unset, every file at
    # once (the board-timing model is ~750 files, tens of GB on a shared machine).
    os.environ.setdefault("JOBS", "8")
    if board["mem"] == "ddr4":
        dmap = BOARD_DDR4_MAP if board.get("board_timing") else SIM_DDR4_MAP
        sys_hz, ddr_hz = dmap.sim_clk_freqs
        sim_config = SimConfig(default_clk="sys_clk", default_clk_freq=sys_hz)
        sim_config.add_clocker("ddr_clk", ddr_hz)
    elif board["mem"] == "ddr3":
        sys_hz, ddr_hz = ddr3_sim_clocks(args.sys_clk_freq)
        sim_config = SimConfig(default_clk="sys_clk", default_clk_freq=sys_hz)
        sim_config.add_clocker("ddr_clk", ddr_hz)
        if args.pipe_clk_freq:
            # The pipes' clock at the board's pipe:sys ratio on a 0.1 ns period grid.
            pclk_period = 100*round(1e12/sys_hz*args.sys_clk_freq/args.pipe_clk_freq/100)
            sim_config.add_clocker("pipeclk", int(1e12//pclk_period))
    else:
        sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_module("serial2console", "serial")
    soc_kwargs = dict(board, update_repo=args.update_repo,
        hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate)
    if args.ckphase_model:
        soc_kwargs["ckphase_model"] = True
    if args.prefetch:
        soc_kwargs["prefetch"] = args.prefetch
    if args.reader_bursts:
        soc_kwargs["reader_bursts"] = args.reader_bursts
    if args.pipe_clk_freq:
        soc_kwargs["pipe_clk_freq"] = args.pipe_clk_freq
    if args.dn_sublanes and "unit_options" in soc_kwargs:
        # The DeltaNet stream's sub-lanes per channel, over the board's default.
        soc_kwargs["unit_options"] = tuple((k, v) for k, v in soc_kwargs["unit_options"]
                                           if k != "dnstream_options") \
                                     + (("dnstream_options", (("sublanes", args.dn_sublanes),)),)
    if args.vec_fused:
        soc_kwargs["vec_fused"] = True
    if args.ddr_settings:
        soc_kwargs["ddr_settings"] = ddr_settings_of(args.ddr_settings)
    csr_csv = os.path.join(out_dir, "csr.csv")

    # Headers + LiteX libraries (to build the firmware against; the ROM is the boot stub, no BIOS;
    # no DDR4 contents, the next pass writes them).
    soc     = TriXiumSimSoC(image=args.image, firmware=None, contents=False, **soc_kwargs)
    builder = Builder(soc, output_dir=out_dir, csr_csv=csr_csv)
    builder.build(sim_config=sim_config, build=False, run=False)

    # Firmware.
    build_firmware(out_dir)

    # SoC with the firmware baked in: Verilog, then Verilator.
    soc     = TriXiumSimSoC(image=args.image, firmware=FIRMWARE, **soc_kwargs)
    builder = Builder(soc, output_dir=out_dir, csr_csv=csr_csv)
    builder.build(sim_config=sim_config, run=False, opt_level="O3",
        threads=args.threads, trace=args.trace, trace_fst=args.trace)
    if args.random_init is not None:
        # The never-reset registers start at random values (this seed), not their init.
        from software.lab import randinit
        n, bits = randinit.randomize(os.path.join(out_dir, "gateware", "sim.v"), args.random_init)
        print(f"[trixium] random init (seed {args.random_init}): {n} never-reset registers, "
              f"{bits} bits")
    subprocess.run(["bash", "build_sim.sh"], cwd=os.path.join(out_dir, "gateware"), check=True)

    # Only obj_dir/Vsim is run. The objects (with -ggdb debug info), their archive and the
    # precompiled headers are ~10x its size (4.5 GB for the board-timing model), and
    # build_sim.sh starts from an empty obj_dir anyway.
    obj_dir = os.path.join(out_dir, "gateware", "obj_dir")
    for name in os.listdir(obj_dir):
        if name.endswith((".o", ".a", ".gch")):
            os.remove(os.path.join(obj_dir, name))

# Run ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriXium Verilator simulation.")
    parser.add_argument("--board",          default="xcu1525", choices=list(BOARDS),               help="Hardware to mirror.")
    parser.add_argument("--hybrid",         action="store_true",                                   help="--board ypcb: with the hybrid models' units (DeltaNet/attention streams, FWHT, ...; build/sim_ypcb_hybrid).")
    parser.add_argument("--image",          default=None,                                          help="BONS image to bake in (default: the board's, DEFAULT_IMAGES).")
    parser.add_argument("--out-dir",        default=None,                                          help="Build directory (default: build/sim_<board>, build/sim_<board>_bt_<image> with --board-timing).")
    parser.add_argument("--board-timing",   action="store_true",                                   help="--board xcu1525: the board's clocks, DDR4 timings/map, CPU and stream geometry (shape-true slices, doc/simulation.md).")
    parser.add_argument("--attn-geometry",  default=None,                                          help="Attention stream lanes,groups,head_dim,t_max (default: the board's; e.g. 4,1,128,256: one command per kv-head slice).")
    parser.add_argument("--build",          action="store_true",                                   help="Build the simulation (headers + libraries, firmware, Verilog, Verilator).")
    parser.add_argument("--run",            action="store_true",                                   help="Run the built simulation (console on stdin/stdout).")
    parser.add_argument("--firmware",       action="store_true",                                   help="Rebuild the firmware and re-bake it into the built simulation (--board xcu1525/ypcb).")
    parser.add_argument("--tprof",          action="store_true",                                   help="Token trace of a bench token and where the engine sits idle (software/lab/tprof.py).")
    parser.add_argument("--gate",           action="store_true",                                   help="Traced generation compared with the Python mirror.")
    parser.add_argument("--bench",          action="store_true",                                   help="Cycles per forward with --bench-toggle off, then on (none: one run, W/P split and the board prediction).")
    parser.add_argument("--bench-toggle",   default=None,                                          help="--bench: console toggle to A/B (resident, silu, dnstream, none; default: resident, none with --board-timing).")
    parser.add_argument("--bench-tokens",   default=2,  type=int,                                  help="--bench: bench N (N+2 forwards).")
    parser.add_argument("--cpu-type",       default=None, choices=["vexriscv", "vexiiriscv"],      help="Override the board's CPU.")
    parser.add_argument("--update-repo",    default="no", choices=["no", "recommended", "latest"], help="VexiiRiscv source update policy.")
    parser.add_argument("--trace",          action="store_true",                                   help="Enable FST tracing.")
    parser.add_argument("--hbm-latency",    default=16, type=int,                                  help="--board fk33: HBM model read latency.")
    parser.add_argument("--hbm-stall-rate", default=0,  type=int,                                  help="--board fk33: HBM model stall rate.")
    parser.add_argument("--threads",        default=4,  type=int,                                  help="Verilator threads.")
    parser.add_argument("--sys-clk-freq",   default=125e6, type=float,                             help="--board ypcb: the board's sys clock (the sim keeps the sys/ddr ratio).")
    parser.add_argument("--prefetch",       default=0, type=int, metavar="BEATS",                 help="Engine weight prefetch (op 7) with BEATS-deep reader FIFOs per channel (0: none).")
    parser.add_argument("--pipe-clk-freq",  default=0, type=float,                                 help="--board ypcb: the channel pipes' own clock (PipeIsland), 0: the sys clock.")
    parser.add_argument("--dn-sublanes",    default=0, type=int,                                   help="--hybrid: DeltaNet stream sub-lanes per channel (0: the board's default).")
    parser.add_argument("--reader-bursts",  default=0, type=int, metavar="N",                     help="16-beat bursts a channel reader keeps outstanding (0: the core's default, 8).")
    parser.add_argument("--vec-fused",      action="store_true",                                   help="Vector unit with the fused clamp/round instructions (TRIXIUM_VEC_FUSED).")
    parser.add_argument("--ckphase-model",  action="store_true",                                   help="--board ypcb: the PHY models behind the K7DDRPHY's CK phase CSRs (calibration test: ddrinit 1).")
    parser.add_argument("--ddr-settings",   default=None,                                          help="--board ypcb: controller settings, e.g. read_time=32,write_time=16,refresh_postponing=1 (default: the board's).")
    parser.add_argument("--gate-pre",       default="",                                            help="--gate: console commands before the traced generation, ';'-separated (e.g. 'gen 2 5 6 7;engreset').")
    parser.add_argument("--random-init",    default=None, type=int, metavar="SEED",               help="--build: never-reset registers start at random values from SEED (build/sim_<board>_ri<SEED>; software/lab/randinit.py).")
    args = parser.parse_args()
    if not (args.build or args.run or args.firmware or args.gate or args.bench or args.tprof):
        parser.error("nothing to do: --build, --run, --firmware, --gate or --bench.")

    # Reproducible builds: the generated sources follow set/dict orders, hence the hash seed.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Everything runs from the repository root (firmware/, models/, build/).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.image is None:
        args.image = DEFAULT_IMAGES.get(args.board, DEFAULT_IMAGES["xcu1525"])
    out_dir      = args.out_dir or (f"build/sim_{args.board}" +
                                    ("" if args.random_init is None else f"_ri{args.random_init}"))
    board        = dict(BOARDS[args.board])
    if args.board_timing:
        if args.board != "xcu1525":
            parser.error("--board-timing is the XCU1525's (DDR4).")
        board      = board_timing_preset()
        name       = os.path.splitext(os.path.basename(args.image))[0]
        out_dir    = args.out_dir or f"build/sim_{args.board}_bt_{name}"
        n_channels = BONSReader(args.image).header["n_channels"]
        if n_channels != board["engine_channels"]:
            parser.error(f"--board-timing: the board has {board['engine_channels']} engine "
                         f"channels, the image {n_channels}.")
    gateware_dir = os.path.join(out_dir, "gateware")
    if args.hybrid:
        if args.board != "ypcb":
            parser.error("--hybrid is the YPCB's (the XCU1525 has every unit already).")
        board.update(HYBRID_UNITS, vexii_extra=board["vexii_extra"] + " --with-rvf")
        out_dir      = args.out_dir or out_dir + "_hybrid"
        gateware_dir = os.path.join(out_dir, "gateware")
    if args.attn_geometry is not None:
        board.update(attn_geometry=tuple(int(v) for v in args.attn_geometry.split(",")))
    if args.cpu_type is not None:
        board.update(cpu_type=args.cpu_type, icache_kb=0, vexii_extra="")
    if args.bench_toggle is None:
        args.bench_toggle = "none" if args.board_timing else "resident"

    # Build.
    if args.build:
        build_sim(args, board, out_dir)

    # Firmware.
    if args.firmware:
        if board["mem"] not in ("ddr4", "ddr3"):
            raise SystemExit("--firmware re-bakes the DDR4/DDR3 model memories (--board xcu1525/ypcb): use --build.")
        build_firmware(out_dir)
        rebake_firmware(gateware_dir, args.image, FIRMWARE, board["n_memctrl"],
            BOARD_DDR4_MAP if args.board_timing else SIM_DDR4_MAP, mem=board["mem"],
            mapping=ddr_controller_settings(**ddr_settings_of(args.ddr_settings))().address_mapping
                    if board["mem"] == "ddr3" else "ROW_BANK_COL")

    # Run.
    if args.run:
        if not os.path.exists(os.path.join(gateware_dir, "obj_dir", "Vsim")):
            raise SystemExit(f"no simulator in {gateware_dir}: build it first (--build).")
        sys.exit(subprocess.call(["obj_dir/Vsim"], cwd=gateware_dir))

    # Gate / Bench.
    from software.lab import simgate
    if args.gate:
        rc = simgate.gate(gateware_dir, args.image, os.path.join(out_dir, "gate"),
            expect_silu=board.get("with_silu", False),
            pre=[c.strip() for c in args.gate_pre.split(";") if c.strip()])
        if rc:
            sys.exit(rc)
    if args.tprof:
        rc = simgate.tprof(gateware_dir, os.path.join(out_dir, "tprof"))
        if rc or not args.bench:
            sys.exit(rc)
    if args.bench:
        sys.exit(simgate.bench(gateware_dir, os.path.join(out_dir, "bench"),
            toggle=args.bench_toggle, tokens=args.bench_tokens, image=args.image,
            board=args.board))

if __name__ == "__main__":
    main()

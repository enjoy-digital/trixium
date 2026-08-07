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

from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.csr_front import BonsaiEngineCSR
from trixium.gateware.vector import BonsaiVectorUnit
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes
from trixium.gateware.loader import BonsaiHBMLoader
from trixium.gateware.sequencer import BonsaiSequencer
from trixium.model.bons import BONSReader

# Constants ----------------------------------------------------------------------------------------

MAIN_RAM_BASE = 0x4000_0000
MAIN_RAM_SIZE = 0x0080_0000 # 8 MB.
from trixium.layout import HDR_CPU_OFFSET
HDR_OFFSET    = HDR_CPU_OFFSET # BONS header+table at main_ram + 4 MB (shared with hardware).

# BonsaiSimSoC -------------------------------------------------------------------------------------

class BonsaiSimSoC(SoCCore):
    def __init__(self, image=None, firmware=None, hbm_latency=16, hbm_stall_rate=0,
                 dual=True):
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

        SoCCore.__init__(self, platform,
            clk_freq                 = int(1e6),
            ident                    = "TriXium Sim",
            cpu_type                 = "vexriscv",
            integrated_rom_size      = 0x20000,
            integrated_main_ram_size = MAIN_RAM_SIZE,
            integrated_main_ram_init = ram_words,
            uart_name                = "sim")
        self.crg = CRG(platform.request("sys_clk"))

        # Scratchpad + engine + HBM channel models.
        self.scratchpad = BonsaiScratchpad(size=256*1024)
        axis = []
        n_channels = img.header["n_channels"] if img is not None else 2
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
        self.vecunit    = BonsaiVectorUnit()
        self.seq = BonsaiSequencer()
        self.comb += [
            self.vecunit.seq_en.eq(self.seq.busy),
            self.vecunit.seq_start.eq(self.seq.vec_start),
            self.vecunit.seq_pc.eq(self.seq.vec_pc),
            self.seq.vec_done.eq(self.vecunit.done),
            self.seq.vec_res6.eq(self.vecunit.res6.status),
            # Must match trixium.py: res6 chaining, or the engine absmax when the
            # descriptor sets w0[4]. Wiring this from vec_res6 directly made seed_abs a
            # silent no-op in sim - the spill then persisted the p8 sum instead of the
            # absmax, which is what 'am == P8 exactly' in the XPROBE output showed.
            self.vecunit.seq_seed7.eq(self.seq.vec_seed7),
        ]
        self.comb += [self.vecunit.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
        self.engine     = BonsaiMatVecEngine(self.scratchpad, axis, vecunit=self.vecunit,
                                              seq=self.seq, dual=dual)
        self.engine_csr = BonsaiEngineCSR(self.engine, seq=self.seq)
        self.add_constant("BONSAI_HAS_SEQ")
        # Loader: PCIe sink dangles in sim; its per-channel AXI writers carry the engine's
        # kind-3 STORE ops (KV append).
        self.loader = BonsaiHBMLoader(axis)
        self.comb += [
            self.engine.store_cmd.connect(self.loader.store_cmd),
            self.engine.store_data.connect(self.loader.store_data),
            self.engine.store_busy.eq(self.loader.store_busy),
            # Sideband (wishbone) scratchpad access is granted only while every unit
            # that touches the two memory ports is idle.
            self.scratchpad.units_idle.eq(~self.engine.sp_busy & ~self.vecunit.busy
                                          & ~self.seq.busy),
        ]
        self.add_constant("BONSAI_HAS_KVSTORE")
        self.add_constant("BONSAI_HAS_TOPK") # Argmax tap + suppressed writes.
        if dual: # Matches the hardware --with-dual option.
            self.add_constant("BONSAI_HAS_DUALQ")
        self.add_constant("BONSAI_HAS_VECUNIT")
        self.add_constant("BONSAI_HAS_VECUNIT2") # M4 program set (rmsnorm scalar ops + rsqrt fix).

        self.bus.add_slave("scratchpad", self.scratchpad.add_wishbone_port(),
            SoCRegion(origin=0xa000_0000, size=self.scratchpad.size, cached=False))

        # Firmware-visible contract.
        self.add_constant("BONSAI_HDR_ADDR",         MAIN_RAM_BASE + HDR_OFFSET)
        self.add_constant("BONSAI_CHAN_WEIGHT_BASE", 0)
        # Sim keeps stride 0: each channel has its own independent HBM model, so the models
        # already give every channel a private address space. On hardware the AXI switch makes
        # the address select the pseudo-channel, hence the non-zero stride there.
        self.add_constant("BONSAI_CHAN_STRIDE",      0)
        self.add_constant("BONSAI_N_CHANNELS",       n_channels)
        if firmware is not None:
            self.add_constant("ROM_BOOT_ADDRESS", MAIN_RAM_BASE)

# Build --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriXium Verilator simulation.")
    parser.add_argument("--image",          default="models/tiny/bonsai-tiny-n2.img")
    parser.add_argument("--firmware",       default="firmware/bonsai.bin")
    parser.add_argument("--build-software", action="store_true", help="Generate headers + BIOS only.")
    parser.add_argument("--run",            action="store_true", help="Build and run the simulation.")
    parser.add_argument("--trace",          action="store_true", help="Enable FST tracing.")
    parser.add_argument("--hbm-latency",    default=16, type=int)
    parser.add_argument("--hbm-stall-rate", default=0, type=int)
    parser.add_argument("--threads",        default=4, type=int)
    parser.add_argument("--no-dual",        action="store_true",
        help="Build the slim engine (no S3/S6 head-B datapath), matching a hardware build without --with-dual.")
    args = parser.parse_args()

    sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_module("serial2console", "serial")

    if args.build_software:
        soc     = BonsaiSimSoC(image=args.image, firmware=None, dual=not args.no_dual,
            hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate)
        builder = Builder(soc, output_dir="build/sim", csr_csv="build/sim/csr.csv")
        builder.compile_gateware = False
        builder.build(sim_config=sim_config, run=False)
        return

    soc     = BonsaiSimSoC(image=args.image, firmware=args.firmware, dual=not args.no_dual,
        hbm_latency=args.hbm_latency, hbm_stall_rate=args.hbm_stall_rate)
    builder = Builder(soc, output_dir="build/sim", csr_csv="build/sim/csr.csv")
    builder.build(sim_config=sim_config, run=args.run, opt_level="O3",
        threads=args.threads, trace=args.trace, trace_fst=args.trace)

if __name__ == "__main__":
    main()

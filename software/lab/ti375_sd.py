#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Ti375 kit's microSD slot on its own: a small SoC (VexRiscv, LiteX's BIOS and SD driver,
UARTBone + crossover console on the kit's serial port), sticky per-pad flags and LiteScope on the
SD pads. Builds in ~2 min; tells which data line a 4-bit transfer misses.

  python3 -m software.lab.ti375_sd build                # build/ti375_sd
  python3 -m software.lab.ti375_sd load                 # volatile (JTAG); the flash is untouched
  litex_server --uart --uart-port /dev/ttyUSB2 --uart-baudrate 115200 &
  python3 -m software.lab.ti375_sd probe 4 8            # card init, 4-bit, 12.5 MHz, read sector 0
  python3 -m litex.tools.litex_term crossover --csr-csv build/ti375_sd/csr.csv   # BIOS: sdcard_*

`probe` prints the pads sampled low/high since cleared (bits: DAT0-3, CMD). 2026-10-23 with the
kit's card: a 4-bit read leaves DAT2 never low (low 1b), the BIOS's 4-bit write fails its CRC
status, and DAT2 does read low while the FPGA drives it: open between the A8 ball and the card
(R323, the slot's pin 1, or the card). Efinix's own demo on this kit runs 4-bit.
"""

import os
import sys
import time
import argparse

OUT = "build/ti375_sd"

# SoC ----------------------------------------------------------------------------------------------

def soc(sys_clk_freq=100e6):
    from migen import Signal, ClockDomain, Cat, If
    from litex.gen import LiteXModule
    from litex_boards.platforms import efinix_ti375_c529_dev_kit
    from litex.soc.cores.clock import TITANIUMPLL
    from litex.soc.integration.soc_core import SoCCore
    from litex.soc.interconnect.csr import CSRStorage, CSRStatus
    from litescope import LiteScopeAnalyzer

    class CRG(LiteXModule):
        def __init__(self, platform):
            self.cd_sys = ClockDomain()
            clk100 = platform.request("clk100")
            self.pll = pll = TITANIUMPLL(platform)
            self.comb += pll.reset.eq(~platform.request("user_btn_n", 0))
            pll.register_clkin(clk100, platform.default_clk_freq)
            pll.create_clkout(self.cd_sys, sys_clk_freq, with_reset=True)

    class PadMonitor(LiteXModule):
        def __init__(self, data_i, cmd_i):
            self.clear = CSRStorage()
            self.low   = CSRStatus(5)
            self.high  = CSRStatus(5)
            self.now   = CSRStatus(5)
            pads = Signal(5)
            self.comb += [pads.eq(Cat(data_i, cmd_i)), self.now.status.eq(pads)]
            self.sync += If(self.clear.re,
                self.low.status.eq(0), self.high.status.eq(0),
            ).Else(
                self.low.status.eq(self.low.status | ~pads),
                self.high.status.eq(self.high.status | pads),
            )

    platform = efinix_ti375_c529_dev_kit.Platform()
    s = SoCCore(platform, sys_clk_freq, ident="TriXium SD slot (Ti375)",
        cpu_type="vexriscv", cpu_variant="standard",
        uart_name="crossover+uartbone", uart_baudrate=115200,
        integrated_rom_size=0x20000, integrated_sram_size=0x4000, integrated_main_ram_size=0x8000)
    s.crg = CRG(platform)
    s.add_sdcard()
    sdpads = s.sdcard.phy.sdpads
    s.padmon = PadMonitor(sdpads.data.i, sdpads.cmd.i)
    s.analyzer = LiteScopeAnalyzer([
            sdpads.clk, sdpads.cmd.i, sdpads.cmd.o, sdpads.cmd.oe,
            sdpads.data.i, sdpads.data.o, sdpads.data.oe, sdpads.data_i_ce,
            s.sdcard.phy.datar.fsm, s.sdcard.phy.cmdr.fsm,
        ], depth=4096, clock_domain="sys", samplerate=sys_clk_freq,
        csr_csv=os.path.join(OUT, "analyzer.csv"))
    return s

# Probe --------------------------------------------------------------------------------------------

def probe(width4, div, sector=0):
    from litex import RemoteClient
    b = RemoteClient(csr_csv=os.path.join(OUT, "csr.csv")); b.open(); r = b.regs

    def cmd(arg, c, rsp):
        r.sdcard_core_cmd_argument.write(arg); r.sdcard_core_cmd_command.write((c << 8) | rsp)
        r.sdcard_core_cmd_send.write(1)
        for _ in range(200):
            e = r.sdcard_core_cmd_event.read()
            if e & 1: return e
        return e

    def resp():
        return r.sdcard_core_cmd_response.read() & 0xffffffff

    def flags(tag):
        print(f"{tag:<24} low {r.padmon_low.read():02x} high {r.padmon_high.read():02x} "
              f"now {r.padmon_now.read():02x}  (bits DAT0-3, CMD)")

    r.padmon_clear.write(1); flags("idle")
    r.sdcard_phy_settings.write(0); r.sdcard_phy_clocker_divider.write(256)
    r.sdcard_phy_init_initialize.write(1); time.sleep(0.01)
    cmd(0, 0, 0)
    print(f"CMD8 {cmd(0x1aa, 8, 5):x}")
    for _ in range(200):
        cmd(0, 55, 5)
        if cmd(0x70ff8000, 41, 1) == 1 and resp() & 0x80000000: break
    cmd(0, 2, 6); cmd(0, 3, 5); rca = resp() >> 16
    cmd(rca << 16, 9, 6); cmd(rca << 16, 7, 7)
    cmd(rca << 16, 55, 5)
    print(f"ACMD6 ({4 if width4 else 1}-bit) {cmd(2 if width4 else 0, 6, 5):x}")
    r.sdcard_phy_settings.write(1 if width4 else 0)
    cmd(512, 16, 5)
    r.sdcard_phy_clocker_divider.write(div)
    r.sdcard_core_block_length.write(512); r.sdcard_core_block_count.write(1)
    r.padmon_clear.write(1)
    e = cmd(sector, 17, (1 << 5) | 5)
    for _ in range(2000):
        d = r.sdcard_core_data_event.read()
        if d & 1: break
    print(f"CMD17 sector {sector}: cmd {e:x} data {d:x} "
          f"({'ok' if d == 1 else 'timeout' if d & 4 else 'CRC error' if d & 8 else 'pending'})")
    if d & 4:
        cmd(0, 12, 7)
    flags("during the read")
    b.close()

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("load")
    q = sub.add_parser("probe"); q.add_argument("width", choices=["1", "4"]); q.add_argument("div", type=int)
    q.add_argument("--sector", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "build":
        from litex.soc.integration.builder import Builder
        s = soc()
        Builder(s, output_dir=OUT, csr_csv=os.path.join(OUT, "csr.csv")).build()
    elif args.cmd == "load":
        from software.host import ti375
        ti375.load(os.path.join(OUT, "gateware", "efinix_ti375_c529_dev_kit.bit"))
    else:
        probe(args.width == "4", args.div, args.sector)

if __name__ == "__main__":
    main()

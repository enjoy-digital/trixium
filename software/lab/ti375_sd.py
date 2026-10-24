#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Ti375 kit's microSD slot or eMMC on its own: a small SoC (VexRiscv, LiteX's BIOS and SD
driver with its DMAs, UARTBone + crossover console on the kit's serial port), sticky per-pad flags
and LiteScope on the pads. Builds in ~2 min.

  python3 -m software.lab.ti375_sd build [--emmc]       # build/ti375_sd, build/ti375_emmc
  python3 -m software.lab.ti375_sd load  [--emmc]       # volatile (JTAG); the flash is untouched
  litex_server --uart --uart-port /dev/ttyUSB2 --uart-baudrate 115200 &
  python3 -m software.lab.ti375_sd probe 4 8            # SD card: init, 4-bit, 12.5 MHz, sector 0
  python3 -m software.lab.ti375_sd probe 8 2 --emmc     # eMMC: init, EXT_CSD, 8-bit HS, 50 MHz
  python3 -m litex.tools.litex_term crossover --csr-csv build/ti375_sd/csr.csv   # BIOS: sdcard_*

`probe` prints the pads sampled low/high since cleared (bit i = DATi, the top bit CMD).
2026-10-23, SD slot: a 4-bit read leaves DAT2 never low, the BIOS's 4-bit write fails its CRC
status, and DAT2 does read low while the FPGA drives it: open between the A8 ball and the card
(R323, the slot's pin 1, or the card). Efinix's own demo on this kit runs 4-bit.
2026-10-24, eMMC (blank, 7.28 GiB, EXT_CSD rev 8): 8-bit at 50 MHz, all nine lines toggle.
"""

import os
import time
import argparse

def out_dir(emmc):
    return "build/ti375_emmc" if emmc else "build/ti375_sd"

# SoC ----------------------------------------------------------------------------------------------

def soc(emmc=False, sys_clk_freq=100e6):
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
            n = len(data_i) + 1
            self.clear = CSRStorage()
            self.low   = CSRStatus(n)
            self.high  = CSRStatus(n)
            self.now   = CSRStatus(n)
            pads = Signal(n)
            self.comb += [pads.eq(Cat(data_i, cmd_i)), self.now.status.eq(pads)]
            self.sync += If(self.clear.re,
                self.low.status.eq(0), self.high.status.eq(0),
            ).Else(
                self.low.status.eq(self.low.status | ~pads),
                self.high.status.eq(self.high.status | pads),
            )

    platform = efinix_ti375_c529_dev_kit.Platform()
    s = SoCCore(platform, sys_clk_freq, ident="TriXium " + ("eMMC" if emmc else "SD slot") + " (Ti375)",
        cpu_type="vexriscv", cpu_variant="standard",
        uart_name="crossover+uartbone", uart_baudrate=115200,
        integrated_rom_size=0x20000, integrated_sram_size=0x4000, integrated_main_ram_size=0x8000)
    s.crg = CRG(platform)
    s.add_sdcard(sdcard_name="emmc" if emmc else "sdcard")
    sdpads = s.sdcard.phy.sdpads
    s.padmon = PadMonitor(sdpads.data.i, sdpads.cmd.i)
    s.analyzer = LiteScopeAnalyzer([
            sdpads.clk, sdpads.cmd.i, sdpads.cmd.o, sdpads.cmd.oe,
            sdpads.data.i, sdpads.data.o, sdpads.data.oe, sdpads.data_i_ce,
            s.sdcard.phy.datar.fsm, s.sdcard.phy.cmdr.fsm,
        ], depth=4096, clock_domain="sys", samplerate=sys_clk_freq,
        csr_csv=os.path.join(out_dir(emmc), "analyzer.csv"))
    return s

# Probe --------------------------------------------------------------------------------------------

RSP_NONE, RSP_SHORT, RSP_LONG, RSP_BUSY, RSP_CRC = 0, 1, 2, 3, 4
DATA_READ = 1 << 5

class Card:
    def __init__(self, emmc):
        from litex import RemoteClient
        self.emmc = emmc
        self.bus  = RemoteClient(csr_csv=os.path.join(out_dir(emmc), "csr.csv"))
        self.bus.open()
        self.r = self.bus.regs
        self.ndata = 8 if emmc else 4

    def cmd(self, arg, c, rsp):
        r = self.r
        r.sdcard_core_cmd_argument.write(arg); r.sdcard_core_cmd_command.write((c << 8) | rsp)
        r.sdcard_core_cmd_send.write(1)
        for _ in range(200):
            e = r.sdcard_core_cmd_event.read()
            if e & 1: return e
        return e

    def resp(self):
        return self.r.sdcard_core_cmd_response.read() & 0xffffffff

    def data_wait(self):
        for _ in range(2000):
            d = self.r.sdcard_core_data_event.read()
            if d & 1: return d
        return d

    def flags(self, tag):
        r = self.r
        w = (self.ndata + 4) // 4
        print(f"{tag:<24} low {r.padmon_low.read():0{w}x} high {r.padmon_high.read():0{w}x} "
              f"now {r.padmon_now.read():0{w}x}  (bit i = DATi, bit {self.ndata} = CMD)")

    def read_dma(self, c, arg, length, count=1, addr=None):
        """A data read (block length `length`, `count` blocks) landed by the block2mem DMA in
        main_ram; returns the bytes."""
        r = self.r
        addr = addr or self.bus.mems.main_ram.base
        r.sdcard_core_block_length.write(length); r.sdcard_core_block_count.write(count)
        r.sdcard_block2mem_dma_enable.write(0)
        r.sdcard_block2mem_dma_base.write(addr)
        r.sdcard_block2mem_dma_length.write(length*count)
        r.sdcard_block2mem_dma_enable.write(1)
        e = self.cmd(arg, c, DATA_READ | RSP_SHORT | RSP_CRC)
        d = self.data_wait()
        if count > 1:
            self.cmd(0, 12, RSP_BUSY | RSP_CRC)
        words = [self.bus.read(addr + 4*i) for i in range(length*count//4)]
        return e, d, b"".join(w.to_bytes(4, "little") for w in words)

    def init_sd(self, width, div):
        r = self.r
        r.sdcard_phy_settings.write(0); r.sdcard_phy_clocker_divider.write(256)
        r.sdcard_phy_init_initialize.write(1); time.sleep(0.01)
        self.cmd(0, 0, RSP_NONE)
        print(f"CMD8 {self.cmd(0x1aa, 8, RSP_SHORT | RSP_CRC):x}")
        for _ in range(200):
            self.cmd(0, 55, RSP_SHORT | RSP_CRC)
            if self.cmd(0x70ff8000, 41, RSP_SHORT) == 1 and self.resp() & 0x80000000: break
        self.cmd(0, 2, RSP_LONG | RSP_CRC); self.cmd(0, 3, RSP_SHORT | RSP_CRC); rca = self.resp() >> 16
        self.cmd(rca << 16, 9, RSP_LONG | RSP_CRC); self.cmd(rca << 16, 7, RSP_BUSY | RSP_CRC)
        self.cmd(rca << 16, 55, RSP_SHORT | RSP_CRC)
        print(f"ACMD6 ({width}-bit) {self.cmd(2 if width == 4 else 0, 6, RSP_SHORT | RSP_CRC):x}")
        r.sdcard_phy_settings.write(1 if width == 4 else 0)
        self.cmd(512, 16, RSP_SHORT | RSP_CRC)
        r.sdcard_phy_clocker_divider.write(div)

    def init_mmc(self, width, div):
        r = self.r
        r.sdcard_phy_settings.write(0); r.sdcard_phy_clocker_divider.write(256)
        r.sdcard_phy_init_initialize.write(1); time.sleep(0.01)
        self.cmd(0, 0, RSP_NONE)
        for i in range(500):                                   # CMD1: sector mode, 1.8/3.3 V.
            if self.cmd(0x40ff8080, 1, RSP_SHORT) == 1 and self.resp() & 0x80000000: break
            time.sleep(0.002)
        ocr = self.resp()
        print(f"CMD1 OCR {ocr:08x} ({'ready' if ocr & 0x80000000 else 'busy'}, "
              f"{'sector' if (ocr >> 29) & 3 == 2 else 'byte'} mode) after {i + 1} tries")
        print(f"CMD2 {self.cmd(0, 2, RSP_LONG | RSP_CRC):x}")
        rca = 1                                                # The host assigns it (MMC).
        print(f"CMD3 {self.cmd(rca << 16, 3, RSP_SHORT | RSP_CRC):x}")
        print(f"CMD9 {self.cmd(rca << 16, 9, RSP_LONG | RSP_CRC):x}")
        print(f"CMD7 {self.cmd(rca << 16, 7, RSP_BUSY | RSP_CRC):x}")
        r.sdcard_phy_clocker_divider.write(8)
        e, d, ext = self.read_dma(8, 0, 512)                    # CMD8: EXT_CSD.
        sec = int.from_bytes(ext[212:216], "little")
        print(f"CMD8 EXT_CSD cmd {e:x} data {d:x}: rev {ext[192]}, card type {ext[196]:02x}, "
              f"bus width {ext[183]}, hs timing {ext[185]}, {sec} sectors ({sec*512/2**30:.2f} GiB), "
              f"boot partitions {ext[226]*128} KB")
        bw = {1: 0, 4: 1, 8: 2}[width]
        print(f"CMD6 bus width {width} {self.cmd((3 << 24) | (183 << 16) | (bw << 8), 6, RSP_BUSY | RSP_CRC):x}")
        r.sdcard_phy_settings.write({1: 0, 4: 1, 8: 2}[width])
        print(f"CMD6 hs timing {self.cmd((3 << 24) | (185 << 16) | (1 << 8), 6, RSP_BUSY | RSP_CRC):x}")
        r.sdcard_phy_clocker_divider.write(div)
        e, d, ext2 = self.read_dma(8, 0, 512)
        same = all(a == b for i, (a, b) in enumerate(zip(ext, ext2)) if i not in (183, 185))
        print(f"EXT_CSD again at {width}-bit, {100/div:.1f} MHz: cmd {e:x} data {d:x}, "
              f"bus width {ext2[183]}, hs timing {ext2[185]}, the rest {'identical' if same else 'DIFFERS'}")
        self.cmd(512, 16, RSP_SHORT | RSP_CRC)
        return sec

def probe(width, div, emmc=False, sector=0, blocks=8):
    c = Card(emmc)
    r = c.r
    r.padmon_clear.write(1); c.flags("idle")
    (c.init_mmc if emmc else c.init_sd)(width, div)
    r.padmon_clear.write(1)
    t = time.time()
    e, d, data = c.read_dma(18 if blocks > 1 else 17, sector, 512, blocks)
    state = "ok" if d == 1 else "timeout" if d & 4 else "CRC error" if d & 8 else "pending"
    print(f"CMD{18 if blocks > 1 else 17} sector {sector} x{blocks}: cmd {e:x} data {d:x} ({state}); "
          f"first bytes {data[:16].hex()}")
    c.flags("during the read")
    c.bus.close()

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["build", "load"]:
        q = sub.add_parser(name); q.add_argument("--emmc", action="store_true")
    q = sub.add_parser("probe"); q.add_argument("width", type=int, choices=[1, 4, 8]); q.add_argument("div", type=int)
    q.add_argument("--emmc", action="store_true"); q.add_argument("--sector", type=int, default=0)
    q.add_argument("--blocks", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "build":
        from litex.soc.integration.builder import Builder
        s = soc(args.emmc)
        Builder(s, output_dir=out_dir(args.emmc), csr_csv=os.path.join(out_dir(args.emmc), "csr.csv")).build()
    elif args.cmd == "load":
        from software.host import ti375
        ti375.load(os.path.join(out_dir(args.emmc), "gateware", "efinix_ti375_c529_dev_kit.bit"))
    else:
        probe(args.width, args.div, args.emmc, args.sector, args.blocks)

if __name__ == "__main__":
    main()

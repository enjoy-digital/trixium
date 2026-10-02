#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""An engine channel's 256-bit AXI onto a LiteDRAM controller (the XCU1525's DDR4).

Two ways, measured on the real LiteDRAM core in simulation (2026-09-19):

- a 256-bit crossbar port: LiteDRAM's native up-converter groups two 256-bit commands into
  one 512-bit controller word through a five-state FSM - ~5 cycles per word even with reads
  pipelined (docs/patches/litedram-native-port-upconverter-pipelined-reads.patch; before
  that patch one word per round trip, 23 cycles/beat on the board). 2.55 cycles/beat for a
  lone stream, 4.3 with four ports on one controller;
- a 512-bit crossbar port behind LiteX's AXIUpConverter: every 256-bit beat becomes a
  single 512-bit transaction on the lane its address selects. Twice the DRAM traffic, but
  the bridge issues a controller command per cycle: 1.31 cycles/beat alone, 2.9 with four
  ports. The default here until a converter that merges the two halves of a word exists.
"""

from migen import *

from litex.gen import LiteXModule
from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface, AXIUpConverter
from litedram.frontend.axi import LiteDRAMAXI2Native


class AXICrossing(LiteXModule):
    """An AXI interface from cd_from to cd_to, one async FIFO per channel (LiteX's
    AXIClockDomainCrossing with a depth and output buffers).

    The split DDR domain crosses here, not with the crossbar's native-port CDC: that one's
    read-data FIFO (16 words) has no backpressure - the controller cannot be stalled - so a
    LiteDRAMAXI2Native holding more reads in flight than it (the engine ports: 256) loses
    data once the ddr clock outruns sys. The AXI R channel has a ready; the bridge then sits
    in the ddr domain on a native port with no CDC at all."""
    def __init__(self, master, slave, cd_from, cd_to, depth=16):
        for ch, a, b in (("aw", cd_from, cd_to), ("w", cd_from, cd_to), ("ar", cd_from, cd_to),
                         ("b", cd_to, cd_from), ("r", cd_to, cd_from)):
            src, dst = (master, slave) if a == cd_from else (slave, master)
            cdc = stream.ClockDomainCrossing(getattr(src, ch).description, a, b,
                                             depth=depth, buffered=True)
            self.add_module(f"{ch}_cdc", cdc)
            self.comb += [getattr(src, ch).connect(cdc.sink), cdc.source.connect(getattr(dst, ch))]


def engine_ddr_port(core, axi, port_width=512, merge=True, w_buffer_depth=64, r_buffer_depth=256,
                    clock_domain="sys"):
    """Returns the list of modules (to register as submodules) that connect `axi` (the
    engine channel's 256-bit AXI) to a crossbar port of `core` of `port_width` bits.
    merge: on a wider port, merge the read bursts two beats per word (axi_merge.AXIWide)
    instead of one wide transaction per beat (LiteX's AXIUpConverter, twice the traffic).
    clock_domain: the port's side when the controller runs in its own domain (the crossbar
    inserts the native-port CDC; see the XCU1525 target's --ddr-clk-freq)."""
    split = clock_domain != "sys"
    # Split: a native port on the controller's own side (the core renames its sys to ddr),
    # the bridge there too, and the AXI crossing into it (AXICrossing).
    port = core.crossbar.get_port(data_width=port_width)
    mods = []
    wide = axi
    if port_width != len(axi.r.data):
        wide = AXIInterface(data_width=port_width, address_width=axi.address_width,
                            id_width=axi.id_width)
        if merge:
            from trixium.gateware.axi_merge import AXIWide
            mods.append(AXIWide(axi, wide))
        else:
            mods.append(AXIUpConverter(axi, wide))
    if split:
        far = AXIInterface(data_width=port_width, address_width=axi.address_width,
                           id_width=axi.id_width, version=wide.version)
        mods.append(AXICrossing(wide, far, "sys", "ddr"))
        wide = far
    bridge = LiteDRAMAXI2Native(wide, port, w_buffer_depth=w_buffer_depth,
                                r_buffer_depth=r_buffer_depth)
    mods.append(ClockDomainsRenamer("ddr")(bridge) if split else bridge)
    return mods

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

from litex.soc.interconnect.axi import AXIInterface, AXIUpConverter
from litedram.frontend.axi import LiteDRAMAXI2Native


def engine_ddr_port(core, axi, port_width=512, merge=True, w_buffer_depth=64, r_buffer_depth=256):
    """Returns the list of modules (to register as submodules) that connect `axi` (the
    engine channel's 256-bit AXI) to a crossbar port of `core` of `port_width` bits.
    merge: on a wider port, merge the read bursts two beats per word (axi_merge.AXIWide)
    instead of one wide transaction per beat (LiteX's AXIUpConverter, twice the traffic)."""
    port = core.crossbar.get_port(data_width=port_width)
    if port_width == len(axi.r.data):
        return [LiteDRAMAXI2Native(axi, port, w_buffer_depth=w_buffer_depth,
                                   r_buffer_depth=r_buffer_depth)]
    wide = AXIInterface(data_width=port_width, address_width=axi.address_width,
                        id_width=axi.id_width)
    if merge:
        from trixium.gateware.axi_merge import AXIWide
        conv = AXIWide(axi, wide)
    else:
        conv = AXIUpConverter(axi, wide)
    return [conv, LiteDRAMAXI2Native(wide, port, w_buffer_depth=w_buffer_depth,
                                     r_buffer_depth=r_buffer_depth)]

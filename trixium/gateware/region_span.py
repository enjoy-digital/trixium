# SPDX-License-Identifier: BSD-2-Clause
"""Combinational, full-width validation of a contiguous physical memory transfer."""
from functools import reduce
from operator import and_

from migen import *
from litex.gen import LiteXModule

from trixium.capabilities import Access

ERROR_CONFIG = 1
ERROR_BOUNDS = 2
ERROR_RESERVED = 4


class RegionSpanValidator(LiteXModule):
    """Resolve one aligned span before any physical bus request is allowed.

    Ports are (endpoint ID, AXI address width) pairs in transport order. Multiple
    ports may reach one endpoint; use the first, so callers never infer memory
    capacity or physical placement from compute-pipe indices. Length is bytes.
    Tables are target policy, not mutable public capability advertisements.
    """
    def __init__(self, regions, reservations, ports, *, access):
        regions, reservations, ports = tuple(regions), tuple(reservations), tuple(ports)
        if not regions or len({r.id for r in regions}) != len(regions):
            raise ValueError('unique physical regions are required')
        if not 1 <= len(ports) <= 31 or any(not 12 <= width <= 64 for _, width in ports):
            raise ValueError('span validator requires 1..31 ports with 12..64-bit addresses')
        if access not in (Access.READ, Access.WRITE):
            raise ValueError('choose one direction for span validation')
        if any(r.alignment != 32 for r in regions):
            raise ValueError('span validator currently requires 32-byte region alignment')
        by_id = {r.id: r for r in regions}
        ordered = sorted(regions, key=lambda r: (r.endpoint, r.base))
        if any(a.endpoint == b.endpoint and a.base+a.size > b.base for a, b in zip(ordered, ordered[1:])):
            raise ValueError('physical regions on one endpoint overlap')
        for r in reservations:
            if r.region_id not in by_id or r.offset+r.size > by_id[r.region_id].size:
                raise ValueError('reservation is outside its region')
        routes = {}
        for i, (endpoint, width) in enumerate(ports):
            routes.setdefault(endpoint, (i, width))

        self.region = Signal(32)
        self.offset, self.length = Signal(64), Signal(64)
        self.valid = Signal()
        self.error = Signal(3)
        self.port = Signal(5)
        self.address = Signal(64)
        found = Signal()
        base, size = Signal(64), Signal(64)
        limit, offset_end = Signal(65), Signal(65)
        physical_start, physical_end = Signal(66), Signal(66)
        reserved_ok = Signal()
        choices = {}
        for r in regions:
            if r.endpoint in routes and r.access & access:
                port, width = routes[r.endpoint]
                choices[r.id] = [found.eq(1), base.eq(r.base), size.eq(r.size),
                                 self.port.eq(port), limit.eq(1 << width)]
        self.comb += [
            found.eq(0), base.eq(0), size.eq(0), self.port.eq(0), limit.eq(0),
            Case(self.region, choices),
            offset_end.eq(self.offset + self.length),
            physical_start.eq(base + self.offset), physical_end.eq(physical_start + self.length),
            self.address.eq(physical_start),
            reserved_ok.eq(reduce(and_, [((self.region != r.region_id) | (offset_end <= r.offset)
                             | (self.offset >= r.offset+r.size)) for r in reservations], C(1, 1))),
            self.error.eq(Cat(~found | (self.length == 0) | (self.offset[:5] != 0)
                             | (self.length[:5] != 0),
                             found & ((offset_end > size) | (physical_end > limit)),
                             found & ~reserved_ok)),
            self.valid.eq(self.error == 0),
        ]

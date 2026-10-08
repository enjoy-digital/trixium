# SPDX-License-Identifier: BSD-2-Clause
"""The flat scratchpad arbiter must be the nested 2:1 client-mux chain, bit for bit.

BonsaiScratchpadArbiter replaces one BonsaiScratchpadClientMux per attached unit (the
v11 VCU build failed setup on those serial mux levels). It is only safe to swap in if
every output toward the scratchpad and every return toward every client is identical to
the chain's, for every combination of busy bits - including two units busy at once,
which the sequencer never does but a manual CSR start could. Checked exhaustively over
the busy combinations for one to six attached clients, with random payloads.
"""
import random

import pytest
from migen import Module, Signal, Mux, run_simulation

from litex.gen import LiteXModule

from trixium.core.scratchpad import BonsaiScratchpadArbiter

# Reference: the nested 2:1 client mux the arbiter replaced.
class BonsaiScratchpadClientMux(LiteXModule):
    """Give an auxiliary unit priority over the vector-unit scratchpad client."""

    def __init__(self, vector, auxiliary):
        self.busy = Signal()
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal.like(vector.sp_rd_adr)
        self.sp_rd_dat = Signal.like(vector.sp_rd_dat)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal.like(vector.sp_wr_adr)
        self.sp_wr_we  = Signal.like(vector.sp_wr_we)
        self.sp_wr_dat = Signal.like(vector.sp_wr_dat)
        self.sp_wr_gnt = Signal()
        sel = auxiliary.busy
        self.comb += [
            self.busy.eq(vector.busy | auxiliary.busy),
            self.sp_rd_req.eq(Mux(sel, auxiliary.sp_rd_req, vector.sp_rd_req)),
            self.sp_rd_adr.eq(Mux(sel, auxiliary.sp_rd_adr, vector.sp_rd_adr)),
            vector.sp_rd_dat.eq(self.sp_rd_dat),
            auxiliary.sp_rd_dat.eq(self.sp_rd_dat),
            self.sp_wr_req.eq(Mux(sel, auxiliary.sp_wr_req, vector.sp_wr_req)),
            self.sp_wr_adr.eq(Mux(sel, auxiliary.sp_wr_adr, vector.sp_wr_adr)),
            self.sp_wr_we.eq(Mux(sel, auxiliary.sp_wr_we, vector.sp_wr_we)),
            self.sp_wr_dat.eq(Mux(sel, auxiliary.sp_wr_dat, vector.sp_wr_dat)),
            vector.sp_wr_gnt.eq(self.sp_wr_gnt & ~sel),
            auxiliary.sp_wr_gnt.eq(self.sp_wr_gnt & sel),
        ]


WIDTHS = dict(sp_rd_adr=13, sp_rd_dat=256, sp_wr_adr=13, sp_wr_we=8, sp_wr_dat=256)
DRIVEN = ("busy", "sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat")
TO_SP = ("busy", "sp_rd_req", "sp_rd_adr", "sp_wr_req", "sp_wr_adr", "sp_wr_we", "sp_wr_dat")
RETURNED = ("sp_rd_dat", "sp_wr_gnt")


class Client:
    def __init__(self):
        for name in ("busy", "sp_rd_req", "sp_wr_req", "sp_wr_gnt"):
            setattr(self, name, Signal())
        for name, width in WIDTHS.items():
            setattr(self, name, Signal(width))


class DUT(Module):
    def __init__(self, n):
        self.chain_base, self.chain = Client(), [Client() for _ in range(n)]
        self.flat_base, self.flat = Client(), [Client() for _ in range(n)]
        top = self.chain_base
        for i, client in enumerate(self.chain):
            mux = BonsaiScratchpadClientMux(top, client)
            setattr(self.submodules, f"mux{i}", mux)
            top = mux
        self.chain_top = top
        self.submodules.arb = BonsaiScratchpadArbiter(self.flat_base, self.flat)
        # The scratchpad side feeds both trees the same read data and write grant.
        self.rd_dat, self.wr_gnt = Signal(256), Signal()
        self.comb += [self.chain_top.sp_rd_dat.eq(self.rd_dat), self.chain_top.sp_wr_gnt.eq(self.wr_gnt),
                      self.arb.sp_rd_dat.eq(self.rd_dat), self.arb.sp_wr_gnt.eq(self.wr_gnt)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_flat_arbiter_equals_the_nested_chain(n):
    dut = DUT(n)
    rng = random.Random(1000 + n)

    def drive():
        checked = 0
        for busy_bits in range(1 << (n + 1)):
            for _ in range(3):
                for k, (c_chain, c_flat) in enumerate(zip([dut.chain_base] + dut.chain,
                                                          [dut.flat_base] + dut.flat)):
                    values = {"busy": (busy_bits >> k) & 1}
                    for name in DRIVEN[1:]:
                        width = WIDTHS.get(name, 1)
                        values[name] = rng.getrandbits(width)
                    for name, value in values.items():
                        yield getattr(c_chain, name).eq(value)
                        yield getattr(c_flat, name).eq(value)
                yield dut.rd_dat.eq(rng.getrandbits(256))
                yield dut.wr_gnt.eq(rng.getrandbits(1))
                yield
                for name in TO_SP:
                    a, b = (yield getattr(dut.chain_top, name)), (yield getattr(dut.arb, name))
                    assert a == b, f"n={n} busy={busy_bits:b}: {name} chain {a:#x} flat {b:#x}"
                for k, (c_chain, c_flat) in enumerate(zip([dut.chain_base] + dut.chain,
                                                          [dut.flat_base] + dut.flat)):
                    for name in RETURNED:
                        a, b = (yield getattr(c_chain, name)), (yield getattr(c_flat, name))
                        assert a == b, f"n={n} busy={busy_bits:b} client {k}: {name} {a:#x} != {b:#x}"
                checked += 1
        assert checked == 3 << (n + 1)

    run_simulation(dut, drive())


def test_priority_is_last_attached():
    # Two units busy together: the one attached last owns the port, as in the chain.
    dut = DUT(3)

    def drive():
        for c in [dut.flat_base] + dut.flat:
            yield c.busy.eq(0)
        for k, c in enumerate(dut.flat):
            yield c.sp_rd_adr.eq(0x100 + k)
        yield dut.flat_base.sp_rd_adr.eq(0x0FF)
        yield dut.flat[0].busy.eq(1)
        yield dut.flat[2].busy.eq(1)
        yield dut.wr_gnt.eq(1)
        yield
        assert (yield dut.arb.sp_rd_adr) == 0x102
        assert (yield dut.flat[2].sp_wr_gnt) == 1
        assert (yield dut.flat[0].sp_wr_gnt) == 0 and (yield dut.flat_base.sp_wr_gnt) == 0
        yield dut.flat[0].busy.eq(0)
        yield dut.flat[2].busy.eq(0)
        yield
        assert (yield dut.arb.sp_rd_adr) == 0x0FF and (yield dut.flat_base.sp_wr_gnt) == 1

    run_simulation(dut, drive())

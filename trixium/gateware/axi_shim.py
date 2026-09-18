#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""AXIResetShim: keeps one HBM port's AXI protocol whole while the master behind it is reset.

The engine, the loader, the sequencer and the vector unit live in a soft-resettable clock
domain (the `engreset` CSR) so a wedged op - a stalled sequencer, a fetch of a row that does
not exist - costs a few microseconds instead of an FPGA reconfigure and a host reboot. A
reset in the middle of an AXI transaction, though, would leave the HBM controller with a
burst nobody drains (R beats for a reader that is gone), an accepted AW whose W beats never
come, or a `valid` that vanished before its `ready`; every one of those wedges the port for
good, which is the very thing the reset is for.

So the shim sits between the port and the master, in the never-reset `sys` domain, and owns
the protocol:

- AR and AW go through one-entry skids: a request the master presented is held valid until
  the port takes it, whatever the master does afterwards.
- It counts what the port still owes or is owed: read bursts issued minus R-last beats
  seen; W beats promised by accepted AWs minus W beats delivered; write bursts issued minus
  B responses.
- `hold` (the soft reset, stretched) starts a drain: the master side is blocked (no new AR/
  AW/W, R and B hidden from it), R beats of outstanding bursts are accepted and discarded,
  W beats still owed are padded with wstrb = 0 (a legal write of nothing - HBM keeps its
  contents), B responses are collected. `busy` stays up until every counter is zero.
- A burst the port never answers (the fetch of an unmapped address) never drains: `stuck`
  goes up after ~5 ms so the firmware can say "reconfigure" instead of waiting forever.

Between drains the shim is transparent on R, W and B (no added registers on the streaming
paths); AR/AW gain one register, which the readers already amortize with 8 outstanding
bursts. No W handshake depends on the AW one (AXI's rule for masters): WLAST is the
master's on the normal path, and the accepted-AW length queue only pads during a drain.
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface

# AXIResetShim -------------------------------------------------------------------------------------

class AXIResetShim(LiteXModule):
    def __init__(self, port, stuck_cycles=1 << 20):
        """port: the HBM-side AXIInterface. self.master: the engine-side one (same params)."""
        self.master = master = AXIInterface(data_width=port.data_width,
                                            address_width=port.address_width,
                                            id_width=port.id_width)
        self.hold  = Signal()   # Assert while the master domain is in reset.
        self.busy  = Signal()   # Draining: the master must not be released yet.
        self.stuck = Signal()   # A drain that will not finish (sticky until the next hold).

        # # #

        # Read side ---------------------------------------------------------------------------
        drain_r = Signal()
        out_r   = Signal(16)   # Bursts issued to the port, not yet fully returned.

        # AR skid: one request held valid until the port takes it.
        ar_v = Signal()
        ar_addr  = Signal(port.address_width)
        ar_len   = Signal(8)
        ar_size  = Signal(3)
        ar_burst = Signal(2)
        ar_id    = Signal(port.id_width)
        ar_fire  = Signal()
        self.comb += [
            port.ar.valid.eq(ar_v),
            port.ar.addr.eq(ar_addr),
            port.ar.len.eq(ar_len),
            port.ar.size.eq(ar_size),
            port.ar.burst.eq(ar_burst),
            port.ar.id.eq(ar_id),
            ar_fire.eq(ar_v & port.ar.ready),
            master.ar.ready.eq((~ar_v | port.ar.ready) & ~drain_r & ~self.hold),
        ]
        self.sync += [
            If(master.ar.valid & master.ar.ready,
                ar_v.eq(1),
                ar_addr.eq(master.ar.addr),
                ar_len.eq(master.ar.len),
                ar_size.eq(master.ar.size),
                ar_burst.eq(master.ar.burst),
                ar_id.eq(master.ar.id),
            ).Elif(ar_fire,
                ar_v.eq(0),
            ),
        ]

        # R: transparent, or discarded while draining.
        r_last_fire = Signal()
        self.comb += [
            master.r.valid.eq(port.r.valid & ~drain_r),
            master.r.data.eq(port.r.data),
            master.r.resp.eq(port.r.resp),
            master.r.last.eq(port.r.last),
            master.r.id.eq(port.r.id),
            port.r.ready.eq(Mux(drain_r, 1, master.r.ready)),
            r_last_fire.eq(port.r.valid & port.r.ready & port.r.last),
        ]
        self.sync += [
            out_r.eq(out_r + Mux(ar_fire, 1, 0) - Mux(r_last_fire, 1, 0)),
            If(self.hold,
                drain_r.eq(1),
            ).Elif(drain_r & ~ar_v & (out_r == 0) & ~r_last_fire & ~ar_fire,
                drain_r.eq(0),
            ),
        ]

        # Write side --------------------------------------------------------------------------
        drain_w = Signal()
        out_b   = Signal(16)   # Write bursts issued, B not yet seen.
        w_due   = Signal(24)   # W beats promised by accepted AWs, not yet delivered.

        wq     = stream.SyncFIFO([("len", 8)], 16)   # accepted AWs' lengths, issue order
        self.submodules += wq

        aw_v = Signal()
        aw_addr  = Signal(port.address_width)
        aw_len   = Signal(8)
        aw_size  = Signal(3)
        aw_burst = Signal(2)
        aw_id    = Signal(port.id_width)
        aw_fire  = Signal()
        self.comb += [
            port.aw.valid.eq(aw_v & wq.sink.ready),   # never more AWs than the W queue holds
            port.aw.addr.eq(aw_addr),
            port.aw.len.eq(aw_len),
            port.aw.size.eq(aw_size),
            port.aw.burst.eq(aw_burst),
            port.aw.id.eq(aw_id),
            aw_fire.eq(port.aw.valid & port.aw.ready),
            master.aw.ready.eq((~aw_v | port.aw.ready) & ~drain_w & ~self.hold),
        ]
        self.sync += [
            If(master.aw.valid & master.aw.ready,
                aw_v.eq(1),
                aw_addr.eq(master.aw.addr),
                aw_len.eq(master.aw.len),
                aw_size.eq(master.aw.size),
                aw_burst.eq(master.aw.burst),
                aw_id.eq(master.aw.id),
            ).Elif(aw_fire,
                aw_v.eq(0),
            ),
        ]

        # W skid: a beat the master presented is delivered whatever happens to the master.
        # On the normal path WLAST is the master's own (the writers set it right), and no W
        # handshake depends on the AW one: AXI forbids a master to wait for AWREADY before
        # WVALID, and a slave may wait for WVALID before AWREADY - the HBM IP does something
        # of the kind (2026-09-18: a W gated on the accepted-AW queue cost ~3k cycles per
        # KV-store packet on the card, 7% of the token), while LiteDRAM's bridge takes W
        # into a FIFO ahead of its one-deep AW buffer, so W beats may fire before their AW
        # and whole bursts may complete before it. The queue of accepted AW lengths serves
        # only the drain: padding the rest of a burst the master abandoned, and whole
        # bursts whose AW fired without any W - one B per AW either way. Bursts that
        # completed ahead of their AW (`ahead`) skip the queue when the AW fires.
        w_v    = Signal()
        w_data = Signal(port.data_width)
        w_strb = Signal(port.data_width//8)
        w_last = Signal()
        w_fire = Signal()
        pad    = Signal()
        w_cnt  = Signal(8)
        w_lastc = Signal()
        last_fire = Signal()
        lf_nq     = Signal()   # A last beat fired with no AW entry for it (yet).
        ahead     = Signal(4)  # Bursts complete before their AW fired at the port.
        aw_match  = Signal()   # This AW's burst already completed: no queue entry.
        self.comb += [
            w_lastc.eq(w_cnt == wq.source.len),        # len = beats - 1 (pad beats only)
            pad.eq(drain_w & ~w_v & (w_due != 0) & wq.source.valid),
            port.w.valid.eq(w_v | pad),
            port.w.data.eq(Mux(w_v, w_data, 0)),
            port.w.strb.eq(Mux(w_v, w_strb, 0)),
            port.w.last.eq(Mux(w_v, w_last, w_lastc)),
            w_fire.eq(port.w.valid & port.w.ready),
            last_fire.eq(w_fire & port.w.last),
            lf_nq.eq(last_fire & ~wq.source.valid),
            aw_match.eq(aw_fire & ((ahead != 0) | lf_nq)),
            wq.sink.valid.eq(aw_fire & ~aw_match),
            wq.sink.len.eq(aw_len),
            wq.source.ready.eq(last_fire & wq.source.valid),
            master.w.ready.eq((~w_v | port.w.ready) & ~drain_w & ~self.hold),
        ]
        self.sync += [
            If(master.w.valid & master.w.ready,
                w_v.eq(1),
                w_data.eq(master.w.data),
                w_strb.eq(master.w.strb),
                w_last.eq(master.w.last),
            ).Elif(w_fire,
                w_v.eq(0),
            ),
            If(w_fire,
                If(port.w.last,
                    w_cnt.eq(0),
                ).Else(
                    w_cnt.eq(w_cnt + 1),
                )
            ),
            If(lf_nq & ~aw_match,
                ahead.eq(ahead + 1),
            ).Elif(aw_match & ~lf_nq,
                ahead.eq(ahead - 1),
            ),
        ]

        # B: the port is always accepted (the writers never back-pressure B either); the
        # master sees it unless draining.
        b_fire = Signal()
        self.comb += [
            port.b.ready.eq(1),
            master.b.valid.eq(port.b.valid & ~drain_w),
            master.b.resp.eq(port.b.resp),
            master.b.id.eq(port.b.id),
            b_fire.eq(port.b.valid & port.b.ready),
        ]
        self.sync += [
            out_b.eq(out_b + Mux(aw_fire, 1, 0) - Mux(b_fire, 1, 0)),
            w_due.eq(w_due + Mux(aw_fire, aw_len + 1, 0) - Mux(w_fire, 1, 0)),
            If(self.hold,
                drain_w.eq(1),
            ).Elif(drain_w & ~aw_v & ~w_v & (w_due == 0) & (out_b == 0) & ~aw_fire & ~b_fire,
                drain_w.eq(0),
            ),
        ]

        # Status ------------------------------------------------------------------------------
        timer = Signal(max=stuck_cycles + 1)
        self.comb += self.busy.eq(drain_r | drain_w)
        self.sync += [
            If(self.hold,
                timer.eq(0),
                self.stuck.eq(0),
            ).Elif(self.busy,
                If(timer == stuck_cycles,
                    self.stuck.eq(1),
                ).Else(
                    timer.eq(timer + 1),
                ),
            ),
        ]

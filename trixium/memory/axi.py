#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""AXI building blocks of the engine's memory channels.

- AXIReader:    burst reader (the weight streamer front-end), multi-ID, credit-based.
- AXIWriter:    burst writer (KV-cache append, weight loader back-end).
- AXIResetShim: keeps an HBM port's AXI protocol whole while the master behind it is reset.
- AXIReadMerge / AXIWide: a 256-bit master onto a 512-bit slave, read bursts merged.

Bursts are capped at 16 beats everywhere: the Xilinx HBM controller's ports are AXI3, so
AxLEN is four bits wide (see AXIReader).
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface, AXIUpConverter, BURST_INCR

from trixium.layout import BEAT_BYTES


# AXIReader ----------------------------------------------------------------------------------------

class AXIReader(LiteXModule):
    """AXI burst reader (the weight streamer front-end).

    Takes {addr, len} commands (beat-aligned byte address, length in 256-bit beats), splits
    them into INCR bursts (16 beats by default, never crossing a 4 KB boundary), keeps several
    bursts outstanding with credit-based reservation of the output FIFO (so R data is never
    back-pressured by us at the AXI port when credits were granted), and streams beats out
    with `last` on the final beat of each command.

    **ARIDs.** With n_ids=1 every burst carries ARID 0 and R data is in-order. That is
    functionally fine and very slow on the real HBM: the Xilinx controller strictly
    serializes same-ID reads (measured on the FK33: 33.7 cycles/beat from one port, one
    16-beat burst per round trip, while the identical RTL sustains 2.35 in the full-SoC
    simulation). With n_ids=N bursts round-robin over N ARIDs so the controller can pipeline
    them; R data then interleaves across IDs (in-order only per ID), so beats land in N
    per-ID FIFOs and a drain stage replays them in issue order - reordering by construction,
    no address matching. The per-ID FIFOs are sized for the outstanding-burst cap (info FIFO
    depth), so they can never overflow: a third burst on an ID cannot have been issued while
    two sit unconsumed.

    **burst_beats must not exceed 16.** The Xilinx HBM controller's ports are AXI3
    (`AXI_nn_ARLEN[3:0]`), so AxLEN is four bits wide and the longest legal burst is 16
    beats. LiteX's AXIInterface carries an 8-bit `len`, and the upper bits are simply
    dropped where it meets the IP - asking for 64 beats arrives as 16. Nothing complains:
    the reader then waits forever for beats 17..64 of each burst (a permanent stall after
    the first bursts of the first fetch).

    Perf counters (free-running Signals, aggregated at the engine level): total beats
    delivered, AR stall cycles (command issued but not accepted), R stall cycles (R valid,
    FIFO not ready - should stay 0 thanks to crediting).
    """
    def __init__(self, axi, burst_beats=16, fifo_depth=512, n_ids=1, max_bursts=8):
        assert axi.data_width == 256
        assert 1 <= burst_beats <= 16, "AXI3 AxLEN is 4 bits: bursts are capped at 16 beats"
        assert n_ids in (1, 2, 4, 8)
        assert max_bursts % n_ids == 0
        self.sink   = sink   = stream.Endpoint([("addr", axi.address_width), ("len", 24)])
        self.source = source = stream.Endpoint([("data", 256)])

        self.total_beats = Signal(32)
        self.ar_stalls   = Signal(32)
        self.r_stalls    = Signal(32)

        # # #

        beat_shift = log2_int(BEAT_BYTES)

        # Burst splitter ---------------------------------------------------------------------------
        cur_addr   = Signal(axi.address_width)
        beats_left = Signal(24)
        active     = Signal()

        self.comb += sink.ready.eq(~active)
        self.sync += [
            If(sink.valid & sink.ready,
                cur_addr.eq(sink.addr),
                beats_left.eq(sink.len),
                active.eq(sink.len != 0),
            )
        ]

        # Burst length: min(burst_beats, beats_left, beats to next 4 KB boundary).
        boundary_full  = Signal(13) # Full-width intermediate: Migen emits narrow-context
        boundary_beats = Signal(8)  # arithmetic inside shifts otherwise. 0 encodes 128.
        b_tmp          = Signal(25)
        burst_len      = Signal(8) # 1..burst_beats, 0 encodes 128 in b_tmp path (never here).
        self.comb += [
            boundary_full.eq(4096 - (cur_addr & 4095)),
            boundary_beats.eq(boundary_full >> beat_shift),
            If(beats_left < burst_beats,
                b_tmp.eq(beats_left),
            ).Else(
                b_tmp.eq(burst_beats),
            ),
            # boundary_beats == 0 means the full 128-beat page is ahead (addr 4 KB-aligned).
            If((boundary_beats != 0) & (boundary_beats < b_tmp),
                burst_len.eq(boundary_beats),
            ).Else(
                burst_len.eq(b_tmp),
            ),
        ]

        # Credit-based FIFO reservation.
        reserved  = Signal(max=2*fifo_depth)
        can_issue = Signal()
        self.comb += can_issue.eq(active & (reserved <= (fifo_depth - burst_len)))

        # Per-burst info in issue order (which ID it went out on, last burst of command?).
        idbits = max(1, log2_int(n_ids)) if n_ids > 1 else 1
        info = stream.SyncFIFO([("cmd_last", 1), ("bid", idbits)], max_bursts)
        self.submodules += info

        id_rr = Signal(idbits)
        self.comb += [
            axi.ar.valid.eq(can_issue & info.sink.ready),
            axi.ar.addr.eq(cur_addr),
            axi.ar.len.eq(burst_len - 1),
            axi.ar.size.eq(beat_shift), # 32 bytes/beat.
            axi.ar.burst.eq(BURST_INCR),
            axi.ar.id.eq(id_rr if n_ids > 1 else 0),
            info.sink.valid.eq(axi.ar.valid & axi.ar.ready),
            info.sink.cmd_last.eq(beats_left == burst_len),
            info.sink.bid.eq(id_rr if n_ids > 1 else 0),
        ]
        if n_ids > 1:
            self.sync += If(axi.ar.valid & axi.ar.ready, id_rr.eq(id_rr + 1))
        self.sync += [
            If(axi.ar.valid & axi.ar.ready,
                cur_addr.eq(cur_addr + (burst_len << beat_shift)),
                beats_left.eq(beats_left - burst_len),
                If(beats_left == burst_len,
                    active.eq(0),
                )
            )
        ]

        # R-channel register slice (skid buffer): the HBM hard macro's AXI interface pins
        # sit at fixed die locations; a comb path from them into the fabric FIFO limited
        # timing. The slice holds beats outside the FIFO, so the AR credit accounting
        # (bounded by fifo_depth) stays conservative.
        r_valid  = Signal()
        r_data   = Signal(256)
        r_last   = Signal()
        r_id     = Signal(idbits)
        r_ready  = Signal()
        rs_valid = Signal()
        rs_data  = Signal(256)
        rs_last  = Signal()
        rs_id    = Signal(idbits)
        r_infire = Signal()
        self.comb += [
            axi.r.ready.eq(~rs_valid),
            r_infire.eq(axi.r.valid & ~rs_valid),
        ]
        self.sync += [
            If(r_ready | ~r_valid,
                If(rs_valid,
                    r_valid.eq(1),
                    r_data.eq(rs_data),
                    r_last.eq(rs_last),
                    r_id.eq(rs_id),
                    rs_valid.eq(0),
                ).Else(
                    r_valid.eq(r_infire),
                    r_data.eq(axi.r.data),
                    r_last.eq(axi.r.last),
                    r_id.eq(axi.r.id[:idbits]),
                )
            ).Elif(r_infire,
                rs_valid.eq(1),
                rs_data.eq(axi.r.data),
                rs_last.eq(axi.r.last),
                rs_id.eq(axi.r.id[:idbits]),
            )
        ]

        # Data FIFO --------------------------------------------------------------------------------
        fifo = stream.SyncFIFO([("data", 256)], fifo_depth, buffered=True)
        self.submodules += fifo

        if n_ids == 1:
            self.comb += [
                fifo.sink.valid.eq(r_valid),
                r_ready.eq(fifo.sink.ready),
                fifo.sink.data.eq(r_data),
                # Command-last: burst-last AND this burst is the command's final burst. The
                # info pop tracks the sliced stream, so command tags stay beat-aligned.
                fifo.sink.last.eq(r_last & info.source.cmd_last),
                info.source.ready.eq(r_valid & r_ready & r_last),
                fifo.source.connect(source),
            ]
        else:
            # Per-ID landing FIFOs: R data is in-order within an ID; depth covers the
            # outstanding cap (max_bursts/n_ids bursts of burst_beats each).
            perid_depth = (max_bursts//n_ids)*burst_beats
            # (`last` is a built-in endpoint flag - SyncFIFO carries it through.)
            perid = [stream.SyncFIFO([("data", 256)], perid_depth)
                     for _ in range(n_ids)]
            self.submodules += perid
            for i, p in enumerate(perid):
                self.comb += [
                    p.sink.valid.eq(r_valid & (r_id == i)),
                    p.sink.data.eq(r_data),
                    p.sink.last.eq(r_last),
                ]
            self.comb += r_ready.eq(Array([p.sink.ready for p in perid])[r_id])

            # Drain in issue order: the head of `info` names the ID whose burst is next;
            # its captured rlast marks the burst end, no beat counter needed.
            sel        = info.source.bid
            sel_valid  = Array([p.source.valid for p in perid])[sel]
            sel_data   = Array([p.source.data  for p in perid])[sel]
            sel_last   = Array([p.source.last  for p in perid])[sel]
            drain_fire = Signal()
            self.comb += [
                fifo.sink.valid.eq(info.source.valid & sel_valid),
                fifo.sink.data.eq(sel_data),
                fifo.sink.last.eq(sel_last & info.source.cmd_last),
                drain_fire.eq(fifo.sink.valid & fifo.sink.ready),
                info.source.ready.eq(drain_fire & sel_last),
                fifo.source.connect(source),
            ]
            for i, p in enumerate(perid):
                self.comb += p.source.ready.eq(drain_fire & (sel == i))

        # Credits: +burst_len on AR fire, -1 per beat delivered to the consumer.
        ar_fire  = Signal()
        out_fire = Signal()
        self.comb += [
            ar_fire.eq(axi.ar.valid & axi.ar.ready),
            out_fire.eq(source.valid & source.ready),
        ]
        self.sync += [
            reserved.eq(reserved + Mux(ar_fire, burst_len, 0) - Mux(out_fire, 1, 0)),
        ]

        # Perf counters ----------------------------------------------------------------------------
        self.sync += [
            If(out_fire,                     self.total_beats.eq(self.total_beats + 1)),
            If(axi.ar.valid & ~axi.ar.ready, self.ar_stalls.eq(self.ar_stalls + 1)),
            If(axi.r.valid & ~axi.r.ready,   self.r_stalls.eq(self.r_stalls + 1)),
        ]

        # Idle indication (all issued data consumed, no command in flight).
        self.idle = Signal()
        self.comb += self.idle.eq(~active & (reserved == 0) & ~source.valid)


# AXIWriter ----------------------------------------------------------------------------------------

class AXIWriter(LiteXModule):
    """AXI burst writer (KV-cache append, weight loader back-end).

    Takes {addr, len} commands and a beat data stream, splits them into INCR bursts (4 KB
    safe), issues W data only after the corresponding AW was accepted, counts B responses and
    exposes `idle` for sequencing. Full-beat writes only (wstrb all-ones). Single AWID.

    **burst_beats must not exceed 16**: the HBM controller's ports are AXI3, so AWLEN is four
    bits (see AXIReader). Overlong write bursts fail more quietly than overlong reads - they
    do not stall: the IP ends the burst after 16 beats while we keep driving W data and place
    WLAST 48 beats late, so the surplus beats are attributed by the controller rather than by
    us. A weight upload still reports success, because the count it reports is our own beat
    counter and not an acknowledgement from HBM.
    """
    def __init__(self, axi, burst_beats=16):
        assert axi.data_width == 256
        assert 1 <= burst_beats <= 16, "AXI3 AxLEN is 4 bits: bursts are capped at 16 beats"
        self.sink_cmd  = cmd  = stream.Endpoint([("addr", axi.address_width), ("len", 24)])
        self.sink_data = data = stream.Endpoint([("data", 256)])
        self.idle      = Signal()

        # # #

        beat_shift = log2_int(BEAT_BYTES)

        # Burst splitter (mirror of the reader's).
        cur_addr   = Signal(axi.address_width)
        beats_left = Signal(24)
        active     = Signal()

        self.comb += cmd.ready.eq(~active)
        cmd_ok = C(1, 1) # Constant-true command check (kept: the generated Verilog depends on it).
        self.sync += [
            If(cmd.valid & cmd.ready,
                cur_addr.eq(cmd.addr),
                beats_left.eq(cmd.len),
                active.eq((cmd.len != 0) & cmd_ok),
            )
        ]

        boundary_full  = Signal(13) # Full-width intermediate (see AXIReader).
        boundary_beats = Signal(8)
        b_tmp          = Signal(25)
        burst_len      = Signal(8)
        self.comb += [
            boundary_full.eq(4096 - (cur_addr & 4095)),
            boundary_beats.eq(boundary_full >> beat_shift),
            If(beats_left < burst_beats,
                b_tmp.eq(beats_left),
            ).Else(
                b_tmp.eq(burst_beats),
            ),
            If((boundary_beats != 0) & (boundary_beats < b_tmp),
                burst_len.eq(boundary_beats),
            ).Else(
                burst_len.eq(b_tmp),
            ),
        ]

        # Issued-burst queue: W beats stream only for accepted AWs.
        winfo = stream.SyncFIFO([("len", 8)], 8)
        self.submodules += winfo
        outstanding = Signal(8)
        self.outstanding = outstanding

        self.comb += [axi.aw.valid.eq(active & winfo.sink.ready),
                      winfo.sink.valid.eq(axi.aw.valid & axi.aw.ready)]

        self.comb += [
            axi.aw.addr.eq(cur_addr),
            axi.aw.len.eq(burst_len - 1),
            axi.aw.size.eq(beat_shift),
            axi.aw.burst.eq(BURST_INCR),
            axi.aw.id.eq(0),
            winfo.sink.len.eq(burst_len),
        ]
        self.sync += [
            If(axi.aw.valid & axi.aw.ready,
                cur_addr.eq(cur_addr + (burst_len << beat_shift)),
                beats_left.eq(beats_left - burst_len),
                If(beats_left == burst_len,
                    active.eq(0),
                )
            )
        ]

        # W data path: pop winfo, stream len beats with wlast - through an output skid
        # so the HBM macro's W-READY pin never gates fabric FIFO pops combinationally
        # (macro-boundary register slice, same reasoning as the reader's R slice).
        w_left   = Signal(8)
        w_active = Signal()
        wi_valid = Signal()
        wi_last  = Signal()
        wi_ready = Signal()
        wo_valid = Signal()
        wo_data  = Signal(len(axi.w.data))
        wo_last  = Signal()
        ws_valid = Signal()
        ws_data  = Signal(len(axi.w.data))
        ws_last  = Signal()
        self.comb += [
            winfo.source.ready.eq(~w_active),
            wi_valid.eq(w_active & data.valid),
            wi_last.eq(w_left == 1),
            wi_ready.eq(~ws_valid),
            data.ready.eq(w_active & wi_ready),
            axi.w.valid.eq(wo_valid),
            axi.w.data.eq(wo_data),
            axi.w.strb.eq(2**len(axi.w.strb) - 1),
            axi.w.last.eq(wo_last),
        ]
        self.sync += [
            If(axi.w.ready | ~wo_valid,
                If(ws_valid,
                    wo_valid.eq(1),
                    wo_data.eq(ws_data),
                    wo_last.eq(ws_last),
                    ws_valid.eq(0),
                ).Else(
                    wo_valid.eq(wi_valid & wi_ready),
                    wo_data.eq(data.data),
                    wo_last.eq(wi_last),
                )
            ).Elif(wi_valid & wi_ready,
                ws_valid.eq(1),
                ws_data.eq(data.data),
                ws_last.eq(wi_last),
            ),
            If(winfo.source.valid & winfo.source.ready,
                w_left.eq(winfo.source.len),
                w_active.eq(1),
            ),
            If(wi_valid & wi_ready,
                w_left.eq(w_left - 1),
                If(w_left == 1,
                    w_active.eq(0),
                )
            ),
        ]

        # B responses (valid sampled through a register: no comb path off the macro pin).
        b_valid_r   = Signal()
        self.comb += axi.b.ready.eq(1)
        ack = axi.b.valid
        self.sync += [
            b_valid_r.eq(ack),
            outstanding.eq(outstanding
                + Mux(axi.aw.valid & axi.aw.ready, 1, 0)
                - Mux(b_valid_r, 1, 0)),
        ]

        # C(1, 1): constant-true term (kept: the generated Verilog depends on it).
        self.comb += self.idle.eq(~active & ~w_active & (outstanding == 0) & ~winfo.source.valid
                                 & C(1, 1))


# AXIResetShim -------------------------------------------------------------------------------------

class AXIResetShim(LiteXModule):
    """Keeps one HBM port's AXI protocol whole while the master behind it is reset.

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

    port: the HBM-side AXIInterface. self.master: the engine-side one (same params).
    """
    def __init__(self, port, stuck_cycles=1 << 20):
        self.master = master = AXIInterface(data_width=port.data_width,
                                            address_width=port.address_width,
                                            id_width=port.id_width)
        self.hold  = Signal()   # Assert while the master domain is in reset.
        self.busy  = Signal()   # Draining: the master must not be released yet.
        self.stuck = Signal()   # A drain that will not finish (sticky until the next hold).

        # # #

        # Read side --------------------------------------------------------------------------------
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

        # Write side -------------------------------------------------------------------------------
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
        # of the kind (a W gated on the accepted-AW queue cost ~3k cycles per KV-store
        # packet, 7% of a token), while LiteDRAM's bridge takes W
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

        # Status -----------------------------------------------------------------------------------
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


# AXIReadMerge -------------------------------------------------------------------------------------

class AXIReadMerge(LiteXModule):
    """A 256-bit AXI master onto a 512-bit slave with its read bursts MERGED, two beats per
    word. The read channels only: axi_from.ar/r (narrow) <-> axi_to.ar/r (wide).

    LiteX's AXIUpConverter turns every narrow beat into its own wide single-beat transaction:
    correct, pipelined, but a 512-bit DRAM word per 256-bit beat - twice the traffic, and on
    the XCU1525 the four engine ports of a DDR4 controller are already near the controller's
    word rate (measured 0.74 words/cycle of a possible 1). AXIWide keeps that converter for
    the write path (STOREs and uploads, a small share of the traffic, byte strobes make the
    half-word writes exact) and merges reads here: a 16-beat 256-bit burst becomes one
    512-bit burst of 8 or 9 words (9 when it starts in the upper half of a word), and each
    word coming back is split into its two halves, the unused half at the ends dropped. IDs
    pass through; the R order is the AR order, which LiteDRAMAXI2Native guarantees.
    """
    def __init__(self, axi_from, axi_to, depth=16):
        dw_from = len(axi_from.r.data)
        dw_to   = len(axi_to.r.data)
        assert dw_to == 2*dw_from, "two narrow beats per wide word"
        bytes_from = dw_from//8
        lsb = log2_int(bytes_from)          # The narrow beat's address bit selects the half.

        # AR: one wide burst per narrow burst. The wide length covers the words the narrow
        # beats touch: first word = addr >> (lsb+1),
        # last word = (addr + (len+1)*bytes - 1) >> (lsb+1).
        info = stream.SyncFIFO([("id", axi_from.id_width), ("nbeats", 9), ("upper", 1)], depth)
        self.submodules += info
        ar = axi_from.ar
        nbeats     = Signal(9)
        first_word = Signal(len(ar.addr))
        last_word  = Signal(len(ar.addr))
        self.comb += [
            nbeats.eq(ar.len + 1),
            first_word.eq(ar.addr >> (lsb + 1)),
            last_word.eq((ar.addr + (nbeats << lsb) - 1) >> (lsb + 1)),
            axi_to.ar.valid.eq(ar.valid & info.sink.ready),
            axi_to.ar.addr.eq(first_word << (lsb + 1)),
            axi_to.ar.len.eq(last_word - first_word),
            axi_to.ar.size.eq(log2_int(dw_to//8)),
            axi_to.ar.burst.eq(BURST_INCR),
            axi_to.ar.id.eq(ar.id),
            axi_to.ar.lock.eq(ar.lock), axi_to.ar.prot.eq(ar.prot), axi_to.ar.cache.eq(ar.cache),
            axi_to.ar.qos.eq(ar.qos), axi_to.ar.region.eq(ar.region),
            ar.ready.eq(axi_to.ar.ready & info.sink.ready),
            info.sink.valid.eq(ar.valid & axi_to.ar.ready),
            info.sink.id.eq(ar.id),
            info.sink.nbeats.eq(nbeats),
            info.sink.upper.eq(ar.addr[lsb]),
        ]

        # R: each wide word yields its lower half then its upper half, except that the first
        # word of a burst starting in the upper half yields only that, and the burst stops
        # after nbeats narrow beats (the last word's upper half may go unused).
        r     = axi_to.r
        half  = Signal()          # 0: lower half of the current word, 1: upper.
        count = Signal(9)         # Narrow beats delivered in this burst.
        first = Signal(reset=1)   # At the first word of the burst.
        cur_half = Signal()
        self.comb += [
            cur_half.eq(Mux(first & info.source.upper, 1, half)),
            axi_from.r.valid.eq(r.valid & info.source.valid),
            axi_from.r.data.eq(Mux(cur_half, r.data[dw_from:], r.data[:dw_from])),
            axi_from.r.resp.eq(r.resp),
            axi_from.r.id.eq(info.source.id),
            axi_from.r.last.eq(count == info.source.nbeats - 1),
        ]
        done_word = Signal()   # This narrow beat is the last one taken from the wide word.
        self.comb += done_word.eq(cur_half | axi_from.r.last)
        self.comb += [
            r.ready.eq(axi_from.r.ready & info.source.valid & done_word),
            info.source.ready.eq(axi_from.r.valid & axi_from.r.ready & axi_from.r.last),
        ]
        self.sync += If(axi_from.r.valid & axi_from.r.ready,
            If(axi_from.r.last,
                count.eq(0), half.eq(0), first.eq(1)
            ).Else(
                count.eq(count + 1),
                first.eq(0),
                half.eq(~cur_half),
            )
        )


# AXIWide ------------------------------------------------------------------------------------------

class AXIWide(LiteXModule):
    """A 256-bit master onto a 512-bit slave: writes through LiteX's beat-wise up-converter,
    reads merged (AXIReadMerge)."""
    def __init__(self, axi_from, axi_to):
        w_to = AXIInterface(data_width=len(axi_to.r.data), address_width=axi_to.address_width,
                            id_width=axi_to.id_width)
        w_from = AXIInterface(data_width=len(axi_from.r.data),
                              address_width=axi_from.address_width,
                              id_width=axi_from.id_width)
        self.submodules.writes = AXIUpConverter(w_from, w_to)
        self.submodules.reads  = AXIReadMerge(axi_from, axi_to)
        self.comb += [
            axi_from.aw.connect(w_from.aw),
            axi_from.w.connect(w_from.w),
            w_from.b.connect(axi_from.b),
            w_to.aw.connect(axi_to.aw),
            w_to.w.connect(axi_to.w),
            axi_to.b.connect(w_to.b),
            # The converter's own read channels stay idle.
            w_from.ar.valid.eq(0), w_to.r.ready.eq(0),
        ]

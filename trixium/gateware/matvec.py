#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiMatVecEngine: N-channel ternary/Q8 matvec engine.

One BonsaiAXIReader + BonsaiChannelPipe per HBM pseudo-channel. A matvec command:

  1. FILL : broadcast x8 (cols bytes) from the scratchpad into every channel's x-buffer.
  2. ISSUE: give every reader the same {wbase, wlen} (identical relative layout per channel).
  3. RUN  : pipes stream rows; the gather round-robins int32 results into the scratchpad at
            dst + (channel*rows + row)*4 bytes; done when rows*n_channels results landed.

The command uses per-channel row counts (rows = M/n_channels) and byte addresses inside the
per-channel HBM address space. Sequenced by the descriptor engine (or a CSR shim/testbench)."""

from migen import *
from migen.genlib.roundrobin import RoundRobin, SP_CE

from litex.gen import *

from litex.soc.interconnect import stream

from trixium.layout import BEAT_BYTES
from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.ternary_mac import BonsaiChannelPipe

# BonsaiMatVecEngine -------------------------------------------------------------------------------

class BonsaiMatVecEngine(LiteXModule):
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96, vecunit=None, seq=None):
        # When a BonsaiVectorUnit is passed, the scratchpad fill/gather ports are muxed
        # between the engine and the vector unit (granted to the vector unit while it is
        # busy - firmware serializes engine ops and vector ops, so contention is
        # structurally impossible; never a 5th RAM port).
        self.n_channels = n = len(hbm_axis)

        self.cmd = stream.Endpoint([
            ("kind",   3),   # 0: ternary, 1: q8, 2: fetch (HBM -> SP), 3: store (SP -> HBM),
                             # 4: axpy (weighted-V: per-lane accumulate, cols results/chan).
            ("rows",   16),  # Rows per channel.
            ("cols",   14),  # K.
            ("wbase",  32),  # Per-channel weight byte address (beat-aligned).
            ("wlen",   24),  # Per-channel weight length in beats.
            ("sbeats", 4),   # Scale beats per row.
            ("cbeats", 8),   # Code beats per row.
            ("bias",   7),   # B.
            ("rshift", 6),   # R.
            ("xsrc",   24),  # x8 source byte address in scratchpad (32B aligned).
            ("dst",    24),  # Result destination byte address in scratchpad (4B aligned).
            ("chan",   5),   # Fetch: source channel index. Store: destination channel.
            ("rem",    5),   # Channels < rem stream one extra row (position-striped KV:
                             # rows = T//n_channels, rem = T % n_channels).
            ("ilv",    1),   # Interleaved gather: result slot = row*n_channels + chan
                             # (position-major scores) instead of chan*rows + row.
            ("supw",   1),   # Suppress scratchpad result writes (argmax-tap-only ops:
                             # the LM head needs no 600 KB result spill).
            ("qn",     1),   # Dual-Q GQA mode: one K/V stream serves 2 sibling Q-heads
                             # (Q8 scores and AXPY kinds only; x holds both vectors).
            ("qstride", 16), # Head-B result offset in words (dst + 4*qstride).
            ("rshift2", 6),  # Head-B AXPY pre-shift (Rpre differs per head).
        ])
        self.busy   = Signal()
        self.sp_busy = Signal() # busy + 1 cycle: the registered write stage's last beat
                                # lands the cycle after busy drops; sideband arbitration
                                # (scratchpad.units_idle) must wait for it.
        self.done   = Signal() # Pulse.
        self.error  = Signal()
        self.absmax = Signal(32) # |int32| max over the last matvec's gathered results.
        # Argmax tap over the gathered results (matvec kinds only): value + global row
        # index (grant*rows + row), ties to the smallest index (matches np.argmax across
        # the round-robin arrival order). Reset on command accept.
        self.argmax_val = Signal((32, True), reset=-(1 << 31))
        self.argmax_idx = Signal(24, reset=0xFFFFFF)

        # STORE (kind 3): scratchpad beats -> one HBM channel, through the loader's
        # per-channel AXI writers (wired by the integrator; a kind-3 op hangs if absent -
        # firmware only issues it when BONSAI_HAS_KVSTORE is set). cmd reuse: chan = dest
        # channel, wbase = HBM byte address, wlen = beats, xsrc = scratchpad byte source.
        self.store_cmd  = stream.Endpoint([("chan", 5), ("addr", 32), ("len", 24)])
        self.store_data = stream.Endpoint([("data", 256)])
        self.store_busy = Signal() # From the loader: executing our store (incl. B drain).

        # # #

        # Channels ----------------------------------------------------------------------------
        fetch_mode = Signal() # kind == 2 for the latched command.
        fetch_chan = Signal(5)

        self.readers = []
        self.pipes   = []
        fetch_valids = []
        fetch_datas  = []
        for i, axi in enumerate(hbm_axis):
            reader = BonsaiAXIReader(axi)
            pipe   = BonsaiChannelPipe(xbuf_groups=xbuf_groups, chan_id=i, n_channels=n)
            setattr(self.submodules, f"reader{i}", reader)
            setattr(self.submodules, f"pipe{i}",   pipe)
            is_fetch = Signal()
            self.comb += [
                is_fetch.eq(fetch_mode & (fetch_chan == i)),
                # Normal: beats feed the compute pipe. Fetch: beats go to the scratchpad.
                pipe.sink.valid.eq(reader.source.valid & ~is_fetch),
                pipe.sink.data.eq(reader.source.data),
                # Fetch path always accepts; compute beats honor the pipe's result-FIFO
                # backpressure (drops results past N=2 otherwise - N/row_beats > 1 gather/cycle).
                reader.source.ready.eq(is_fetch | pipe.sink.ready),
            ]
            fetch_valids.append(reader.source.valid & is_fetch)
            fetch_datas.append(reader.source.data)
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command ---------------------------------------------------------------------
        cmd_r = {}
        for name in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                     "bias", "rshift", "xsrc", "dst", "chan", "rem", "ilv", "supw",
                     "qn", "qstride", "rshift2"):
            cmd_r[name] = Signal.like(getattr(self.cmd, name))
        self.sync += If(self.cmd.valid & self.cmd.ready,
            *[cmd_r[k].eq(getattr(self.cmd, k)) for k in cmd_r])
        store_mode = Signal()
        axpy_mode  = Signal()
        self.comb += [
            fetch_mode.eq(cmd_r["kind"] == 2),
            store_mode.eq(cmd_r["kind"] == 3),
            axpy_mode.eq(cmd_r["kind"] == 4),
            fetch_chan.eq(cmd_r["chan"]),
        ]

        # Broadcast op params to pipes (channels < rem run one extra row).
        for i, pipe in enumerate(self.pipes):
            self.comb += [
                pipe.kind.eq(Mux(axpy_mode, 2,
                             Mux(fetch_mode | store_mode, 0, cmd_r["kind"][0:2]))),
                pipe.rows.eq(cmd_r["rows"] + (i < cmd_r["rem"])),
                pipe.scale_beats.eq(cmd_r["sbeats"]),
                pipe.code_beats.eq(cmd_r["cbeats"]),
                pipe.bias.eq(cmd_r["bias"]),
                pipe.rshift.eq(cmd_r["rshift"]),
                pipe.qn.eq(cmd_r["qn"]),
                pipe.rshift2.eq(cmd_r["rshift2"]),
            ]
        # Overflow is included so any future rate mismatch fails loud (sticky error bit)
        # instead of dropping results and hanging the op.
        self.comb += self.error.eq(reduce_or([p.error | p.overflow for p in self.pipes]))

        # x-fill ------------------------------------------------------------------------------
        fill_beat  = Signal(10) # Beat index within x fill (cols/32 <= 384).
        fill_total = Signal(10)
        fill_half  = Signal(10) # Beats per head (dual-Q reads 2x: A then B, contiguous).
        self.comb += [
            fill_half.eq(Mux(axpy_mode,
                (cmd_r["rows"]*self.n_channels + cmd_r["rem"] + 31) >> 5,
                cmd_r["cols"] >> 5)),
            fill_total.eq(Mux(cmd_r["qn"], fill_half << 1, fill_half)),
        ]

        # Registered fill pipeline: address at T, data+write at T+1.
        fill_active_d = Signal()
        fill_beat_d   = Signal(10)
        st_cnt        = Signal(13) # STORE beat counter (address holds until accepted).
        st_rdv        = Signal()   # fill.dat_r valid for the STORE stream.
        eng_fill_adr = Signal(len(scratchpad.fill.adr))
        self.comb += eng_fill_adr.eq((cmd_r["xsrc"] >> 5) + Mux(store_mode, st_cnt, fill_beat))
        self.comb += [
            self.store_data.valid.eq(st_rdv),
            self.store_data.data.eq(scratchpad.fill.dat_r),
        ]
        st_advance = Signal()
        self.comb += st_advance.eq(self.store_data.valid & self.store_data.ready)
        # Dual-Q fill: source beats stay contiguous (head A then head B at xsrc), the
        # x-buffer position interleaves them at 32B granularity (pos = 2g + head) so one
        # xbuf read serves both heads in the pipes.
        fill_headb = Signal()
        fill_g     = Signal(10)
        fill_pos   = Signal(10)
        self.comb += [
            fill_headb.eq(cmd_r["qn"] & (fill_beat_d >= fill_half)),
            fill_g.eq(Mux(fill_headb, fill_beat_d - fill_half, fill_beat_d)),
            # Ternary consumes whole 1024b words per beat: interleave at word granularity
            # (head B at odd words, read via the xbuf's second port). Q8/AXPY interleave
            # at 32B sub-word granularity (one word covers both heads' slices).
            fill_pos.eq(Mux(cmd_r["qn"],
                Mux(cmd_r["kind"] == 0,
                    Cat(fill_g[0:2], fill_headb, fill_g[2:9]),
                    Cat(fill_headb, fill_g[:9])),
                fill_beat_d)),
        ]
        # Second fill stage: the 256-bit scratchpad read data was broadcast COMB from
        # the BRAM output to every pipe's x-buffer write port (the v11 worst path).
        # Data registers once here; position/enable delay a cycle to stay aligned. The
        # last writes land during ISSUE - x-buffer reads only start once beats arrive.
        # Per-pipe copies (dont_touch): a single stage-2 register driving every pipe's
        # x-buffer write port was the v13 route-spread path; each pipe now owns a local
        # copy the placer can pull next to its x-buffer.
        for i, pipe in enumerate(self.pipes):
            fa2 = Signal(name=f"fill_act2_{i}")
            fp2 = Signal(10, name=f"fill_pos2_{i}")
            fd2 = Signal(256, name=f"fill_dat2_{i}")
            for sig in (fa2, fp2, fd2):
                sig.attr.add(("dont_touch", "true"))
            self.sync += [
                fa2.eq(fill_active_d),
                fp2.eq(fill_pos),
                fd2.eq(scratchpad.fill.dat_r),
            ]
            self.comb += [
                pipe.xw_group.eq(fp2[2:]),
                pipe.xw_sub.eq(fp2[:2]),
                pipe.xw_data.eq(fd2),
                pipe.xw_we.eq(fa2),
            ]

        # Gather ------------------------------------------------------------------------------
        self.rr = rr = RoundRobin(n, SP_CE)
        result_cnt  = Signal(32)
        grant_valid = Array([p.source.valid for p in self.pipes])[rr.grant]
        grant_row   = Array([p.source.row   for p in self.pipes])[rr.grant]
        grant_data  = Array([p.source.data  for p in self.pipes])[rr.grant]
        grant_wide  = Array([p.source.wide  for p in self.pipes])[rr.grant]

        # Interleaved (position-major) ops merge every pipe whose head falls in the same
        # scratchpad beat as the granted head: slot = row*N + chan, so within one beat all
        # lanes map to distinct pipes (N power-of-2) and up to 8 results retire per cycle.
        # This is the gather-widen: at N channels the aggregate score rate N/row_beats
        # exceeds 1/cycle from N=4 up, and merged retirement keeps the drain above it.
        ilv_mode = Signal()
        dstw     = Signal(24)
        self.comb += [
            ilv_mode.eq(cmd_r["ilv"] & ~axpy_mode & ~fetch_mode & ~store_mode),
            dstw.eq(cmd_r["dst"] >> 2),
        ]
        slot_p     = [Signal(24, name=f"slot_p{i}") for i in range(n)]
        merge_mask = Signal(n)
        grant_slot = Signal(24)
        for i, pipe in enumerate(self.pipes):
            self.comb += slot_p[i].eq(dstw + Mux(pipe.source.head, cmd_r["qstride"], 0)
                                      + pipe.source.row*n + i)
        self.comb += grant_slot.eq(Array(slot_p)[rr.grant])
        self.comb += merge_mask.eq(Cat(*[
            self.pipes[i].source.valid & ~self.pipes[i].source.wide
            & (slot_p[i][3:] == grant_slot[3:])
            for i in range(n)]))

        self.comb += [
            rr.request.eq(Cat(*[p.source.valid for p in self.pipes])),
            rr.ce.eq(1),
        ]
        for i, pipe in enumerate(self.pipes):
            self.comb += pipe.source.ready.eq(Mux(ilv_mode, merge_mask[i], rr.grant == i))

        # Merged lane build: within one beat each word lane belongs to at most one pipe.
        lane_we  = Signal(8)
        lane_dat = Signal(256)
        for l in range(8):
            sels = [merge_mask[i] & (slot_p[i][0:3] == l) for i in range(n)]
            self.comb += lane_we[l].eq(reduce_or(sels))
            self.comb += lane_dat[32*l:32*l+32].eq(reduce_or(
                [Replicate(sels[i], 32) & self.pipes[i].source.data[0:32]
                 for i in range(n)]))

        is_matvec = Signal() # Kinds 0/1 only (not fetch/store/axpy partials).
        gidx      = Signal(24)
        grant_s   = Signal((32, True))
        self.comb += [
            grant_s.eq(grant_data[0:32]),
            is_matvec.eq((cmd_r["kind"] == 0) | (cmd_r["kind"] == 1)),
            gidx.eq(rr.grant*cmd_r["rows"] + grant_row),
        ]
        # Argmax tap rides the single-grant path only: it is used exclusively by the
        # chan-major LM-head op (never ilv, never AXPY), where grants stay 1/cycle.
        # Argmax compare runs one stage after the grant (same lag argument as absmax).
        am_v_r  = Signal()
        am_s_r  = Signal((32, True))
        am_i_r  = Signal(24)
        self.sync += [
            am_v_r.eq(is_matvec & ~ilv_mode & grant_valid & ~grant_wide),
            am_s_r.eq(grant_s),
            am_i_r.eq(gidx),
        ]
        self.sync += [
            If(self.cmd.valid & self.cmd.ready,
                self.argmax_val.eq(-(1 << 31)),
                self.argmax_idx.eq(0xFFFFFF),
            ).Elif(am_v_r &
                   ((am_s_r > self.argmax_val)
                    | ((am_s_r == self.argmax_val) & (am_i_r < self.argmax_idx))),
                self.argmax_val.eq(am_s_r),
                self.argmax_idx.eq(am_i_r),
            )
        ]

        # Fetch datapath: selected channel's beats -> scratchpad (256-bit writes).
        fetch_valid = Signal()
        fetch_data  = Signal(256)
        fetch_cnt   = Signal(24)
        self.comb += [
            fetch_valid.eq(reduce_or(fetch_valids)),
            fetch_data.eq(Array(fetch_datas)[fetch_chan]),
        ]

        # Scratchpad write port: fetch beats, one merged ilv beat, one wide AXPY beat, or
        # a single gathered int32. The write stage is REGISTERED: FIFO-out -> merge muxes
        # -> RAM write / absmax tree in one cycle was the worst hardware path (38 logic
        # levels); the RAM write, result counting and the trackers run one cycle later
        # off these registers (ops are separated by dispatch - the lag is invisible).
        slot = Signal(24)
        # Fetch write addressing counts at the comb stage (beats arrive every cycle;
        # the registered fetch_cnt lags one cycle and is only the terminator).
        fetch_wcnt   = Signal(24)
        eng_gw_adr_c = Signal(len(scratchpad.gather.adr))
        eng_gw_we_c  = Signal(8)
        eng_gw_dat_c = Signal(256)
        eng_gw_adr = Signal(len(scratchpad.gather.adr))
        eng_gw_we  = Signal(8)
        eng_gw_dat = Signal(256)
        self.sync += [
            eng_gw_adr.eq(eng_gw_adr_c),
            eng_gw_we.eq(eng_gw_we_c),
            eng_gw_dat.eq(eng_gw_dat_c),
        ]
        busy_d = Signal()
        self.sync += busy_d.eq(self.busy)
        self.comb += self.sp_busy.eq(self.busy | busy_d)
        self.sync += If(self.cmd.valid & self.cmd.ready,
            fetch_wcnt.eq(0),
        ).Elif((eng_gw_we_c != 0) & fetch_mode,
            fetch_wcnt.eq(fetch_wcnt + 1),
        )
        grant_head = Array([p.source.head for p in self.pipes])[rr.grant]
        grant_qoff = Signal(16)
        self.comb += grant_qoff.eq(Mux(grant_head, cmd_r["qstride"], 0))
        self.comb += [
            If(axpy_mode,
                # Wide drain rows are 8-lane group starts; dst must be 32B-aligned.
                slot.eq(dstw + grant_qoff + rr.grant*cmd_r["cols"] + grant_row),
            ).Else(
                slot.eq(dstw + grant_qoff + rr.grant*cmd_r["rows"] + grant_row),
            ),
            If(fetch_mode,
                eng_gw_adr_c.eq((cmd_r["dst"] >> 5) + fetch_wcnt),
                eng_gw_dat_c.eq(fetch_data),
                eng_gw_we_c.eq(Mux(fetch_valid, 0xFF, 0)),
            ).Elif(ilv_mode,
                eng_gw_adr_c.eq(grant_slot[3:]),
                eng_gw_dat_c.eq(lane_dat),
                eng_gw_we_c.eq(Mux(grant_valid, lane_we, 0)),
            ).Elif(axpy_mode,
                eng_gw_adr_c.eq(slot[3:]),
                eng_gw_dat_c.eq(grant_data),
                eng_gw_we_c.eq(Mux(grant_valid, 0xFF, 0)),
            ).Else(
                eng_gw_adr_c.eq(slot[3:]),
                eng_gw_dat_c.eq(Replicate(grant_data[0:32], 8)),
                eng_gw_we_c.eq(Mux(grant_valid, 1 << slot[:3], 0)),
            )
        ]

        # Result absmax tracker, write-stage (norm_i32's shift argument; reset on command
        # accept). Runs over every written lane so merged/wide retirement is covered.
        # Per-lane |value| is registered before the max tree: FF(write data) -> abs ->
        # tree -> compare -> CE was v6's worst path. The tracker lags the write by two
        # cycles; every reader (CSR, sequencer k(absmax) at DISPATCH) trails done by more.
        lane_abs = []
        for l in range(8):
            ls = Signal((32, True), name=f"labs_s{l}")
            la = Signal(32, name=f"labs{l}")
            lr = Signal(32, name=f"labsr{l}")
            self.comb += [
                ls.eq(eng_gw_dat[32*l:32*l+32]),
                la.eq(Mux(eng_gw_we[l], Mux(ls < 0, -ls, ls), 0)),
            ]
            self.sync += lr.eq(la)
            lane_abs.append(lr)
        def _maxtree(sigs):
            while len(sigs) > 1:
                nxt = []
                for a, b in zip(sigs[::2], sigs[1::2]):
                    m = Signal(32)
                    self.comb += m.eq(Mux(a > b, a, b))
                    nxt.append(m)
                sigs = nxt
            return sigs[0]
        wr_absmax = _maxtree(lane_abs)
        self.sync += [
            If(self.cmd.valid & self.cmd.ready,
                self.absmax.eq(0),
            ).Elif(~fetch_mode & (wr_absmax > self.absmax),
                self.absmax.eq(wr_absmax),
            )
        ]

        # Scratchpad port wiring (direct, or muxed with the vector unit). supw masks only
        # the memory-facing enables - result counting still runs on eng_gw_we.
        sp_gw_we = Signal(8)
        self.comb += sp_gw_we.eq(Mux(cmd_r["supw"] & ~fetch_mode, 0, eng_gw_we))
        if vecunit is None:
            self.comb += [
                scratchpad.fill.adr.eq(eng_fill_adr),
                scratchpad.gather.adr.eq(eng_gw_adr),
                scratchpad.gather.dat_w.eq(eng_gw_dat),
                scratchpad.gather.we.eq(sp_gw_we),
            ]
        else:
            grant_vec = Signal()
            grant_seq = Signal()
            self.comb += [
                grant_vec.eq(vecunit.busy),
                grant_seq.eq(0 if seq is None else seq.sp_active),
                If(grant_vec,
                    scratchpad.fill.adr.eq(vecunit.sp_rd_adr),
                    scratchpad.gather.adr.eq(vecunit.sp_wr_adr),
                    scratchpad.gather.dat_w.eq(vecunit.sp_wr_dat),
                    scratchpad.gather.we.eq(vecunit.sp_wr_we),
                ).Elif(grant_seq,
                    scratchpad.fill.adr.eq(0 if seq is None else seq.sp_adr),
                    scratchpad.gather.adr.eq(eng_gw_adr),
                    scratchpad.gather.dat_w.eq(eng_gw_dat),
                    scratchpad.gather.we.eq(0), # Descriptor fetches never write.
                ).Else(
                    scratchpad.fill.adr.eq(eng_fill_adr),
                    scratchpad.gather.adr.eq(eng_gw_adr),
                    scratchpad.gather.dat_w.eq(eng_gw_dat),
                    scratchpad.gather.we.eq(sp_gw_we),
                ),
                vecunit.sp_rd_dat.eq(scratchpad.fill.dat_r),
            ]
            if seq is not None:
                self.comb += seq.sp_dat.eq(scratchpad.fill.dat_r)

        # Control FSM -------------------------------------------------------------------------
        self.fsm = fsm = FSM(reset_state="IDLE")
        start_pulse = Signal()
        for pipe in self.pipes:
            self.comb += pipe.start.eq(start_pulse)

        fsm.act("IDLE",
            self.cmd.ready.eq(1),
            If(self.cmd.valid,
                NextValue(fill_beat, 0),
                NextValue(result_cnt, 0),
                NextState("START"),
            )
        )
        fsm.act("START",
            start_pulse.eq(1),
            If(fetch_mode,
                NextState("ISSUE"), # No x-fill for fetches.
            ).Elif(store_mode,
                NextState("ST_CMD"),
            ).Else(
                NextState("FILL"),
            )
        )
        fsm.act("ST_CMD",
            self.busy.eq(1),
            self.store_cmd.valid.eq(1),
            self.store_cmd.chan.eq(cmd_r["chan"]),
            self.store_cmd.addr.eq(cmd_r["wbase"]),
            self.store_cmd.len.eq(cmd_r["wlen"]),
            If(self.store_cmd.ready,
                NextState("ST_RUN"),
            )
        )
        fsm.act("ST_RUN",
            self.busy.eq(1),
            If(st_advance & (st_cnt == cmd_r["wlen"] - 1),
                NextState("ST_DRAIN"),
            )
        )
        fsm.act("ST_DRAIN", # Wait for the loader's B-channel drain: data is then readable.
            self.busy.eq(1),
            If(~self.store_busy,
                self.done.eq(1),
                NextState("IDLE"),
            )
        )
        fsm.act("FILL",
            self.busy.eq(1),
            NextValue(fill_beat, fill_beat + 1),
            If(fill_beat == fill_total - 1,
                NextState("FILL_LAST"),
            )
        )
        fsm.act("FILL_LAST", # Drain the registered fill write.
            self.busy.eq(1),
            NextState("ISSUE"),
        )
        fsm.act("ISSUE",
            self.busy.eq(1),
            *[r.sink.valid.eq(~fetch_mode | (fetch_chan == i))
              for i, r in enumerate(self.readers)],
            *[r.sink.addr.eq(cmd_r["wbase"]) for r in self.readers],
            *[r.sink.len.eq(cmd_r["wlen"]
                + Mux(i < cmd_r["rem"], cmd_r["sbeats"] + cmd_r["cbeats"], 0))
              for i, r in enumerate(self.readers)],
            # All readers are idle here: they accept together.
            If(self.readers[0].sink.ready,
                NextState("RUN"),
            )
        )
        mv_total = Signal(24)
        ax_total = Signal(24)
        self.comb += [
            mv_total.eq(cmd_r["rows"]*self.n_channels + cmd_r["rem"]),
            ax_total.eq(cmd_r["cols"]*self.n_channels),
        ]
        fsm.act("RUN",
            self.busy.eq(1),
            If(Mux(fetch_mode,
                   fetch_cnt == cmd_r["wlen"],
                   # Dual-Q ops retire two heads' results per stream.
                   result_cnt == Mux(cmd_r["qn"],
                       Mux(axpy_mode, ax_total << 1, mv_total << 1),
                       Mux(axpy_mode, ax_total, mv_total))),
                self.done.eq(1),
                NextState("IDLE"),
            )
        )

        # Fill write-stage delay + result counting.
        self.sync += [
            fill_active_d.eq(fsm.ongoing("FILL")),
            fill_beat_d.eq(fill_beat),
            # STORE stream: the read address holds until the beat is accepted; dat_r is
            # valid one stable-address cycle later (one dead cycle after each advance).
            st_rdv.eq(fsm.ongoing("ST_RUN") & ~st_advance),
            If(start_pulse,
                st_cnt.eq(0),
            ).Elif(st_advance,
                st_cnt.eq(st_cnt + 1),
            ),
            If(start_pulse,
                result_cnt.eq(0),
                fetch_cnt.eq(0),
            ).Elif(eng_gw_we != 0,
                If(fetch_mode,
                    fetch_cnt.eq(fetch_cnt + 1),
                ).Else(
                    # Merged ilv beats and wide AXPY beats retire several results at once.
                    result_cnt.eq(result_cnt + sum(eng_gw_we[l] for l in range(8))),
                )
            ),
        ]

def reduce_or(sigs):
    from functools import reduce
    from operator import or_
    return reduce(or_, sigs)

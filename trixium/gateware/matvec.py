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
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96, vecunit=None):
        # When a BonsaiVectorUnit is passed, the scratchpad fill/gather ports are muxed
        # between the engine and the vector unit (granted to the vector unit while it is
        # busy - firmware serializes engine ops and vector ops, so contention is
        # structurally impossible; never a 5th RAM port).
        self.n_channels = n = len(hbm_axis)

        self.cmd = stream.Endpoint([
            ("kind",   2),   # 0: ternary, 1: q8, 2: fetch (HBM -> SP), 3: store (SP -> HBM).
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
        ])
        self.busy   = Signal()
        self.done   = Signal() # Pulse.
        self.error  = Signal()
        self.absmax = Signal(32) # |int32| max over the last matvec's gathered results.

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
            pipe   = BonsaiChannelPipe(xbuf_groups=xbuf_groups)
            setattr(self.submodules, f"reader{i}", reader)
            setattr(self.submodules, f"pipe{i}",   pipe)
            is_fetch = Signal()
            self.comb += [
                is_fetch.eq(fetch_mode & (fetch_chan == i)),
                # Normal: beats feed the compute pipe. Fetch: beats go to the scratchpad.
                pipe.sink.valid.eq(reader.source.valid & ~is_fetch),
                pipe.sink.data.eq(reader.source.data),
                reader.source.ready.eq(1), # Pipe always accepts; fetch path always accepts.
            ]
            fetch_valids.append(reader.source.valid & is_fetch)
            fetch_datas.append(reader.source.data)
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command ---------------------------------------------------------------------
        cmd_r = {}
        for name in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                     "bias", "rshift", "xsrc", "dst", "chan", "rem", "ilv"):
            cmd_r[name] = Signal.like(getattr(self.cmd, name))
        self.sync += If(self.cmd.valid & self.cmd.ready,
            *[cmd_r[k].eq(getattr(self.cmd, k)) for k in cmd_r])
        store_mode = Signal()
        self.comb += [
            fetch_mode.eq(cmd_r["kind"] == 2),
            store_mode.eq(cmd_r["kind"] == 3),
            fetch_chan.eq(cmd_r["chan"]),
        ]

        # Broadcast op params to pipes (channels < rem run one extra row).
        for i, pipe in enumerate(self.pipes):
            self.comb += [
                pipe.kind.eq(cmd_r["kind"]),
                pipe.rows.eq(cmd_r["rows"] + (i < cmd_r["rem"])),
                pipe.scale_beats.eq(cmd_r["sbeats"]),
                pipe.code_beats.eq(cmd_r["cbeats"]),
                pipe.bias.eq(cmd_r["bias"]),
                pipe.rshift.eq(cmd_r["rshift"]),
            ]
        self.comb += self.error.eq(reduce_or([p.error for p in self.pipes]))

        # x-fill ------------------------------------------------------------------------------
        fill_beat  = Signal(10) # Beat index within x fill (cols/32 <= 384).
        fill_total = Signal(10)
        self.comb += fill_total.eq(cmd_r["cols"] >> 5)

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
        for pipe in self.pipes:
            self.comb += [
                pipe.xw_group.eq(fill_beat_d[2:]),
                pipe.xw_sub.eq(fill_beat_d[:2]),
                pipe.xw_data.eq(scratchpad.fill.dat_r),
                pipe.xw_we.eq(fill_active_d),
            ]

        # Gather ------------------------------------------------------------------------------
        self.rr = rr = RoundRobin(n, SP_CE)
        result_cnt  = Signal(32)
        grant_valid = Array([p.source.valid for p in self.pipes])[rr.grant]
        grant_row   = Array([p.source.row   for p in self.pipes])[rr.grant]
        grant_data  = Array([p.source.data  for p in self.pipes])[rr.grant]

        self.comb += [
            rr.request.eq(Cat(*[p.source.valid for p in self.pipes])),
            rr.ce.eq(1),
        ]
        for i, pipe in enumerate(self.pipes):
            self.comb += pipe.source.ready.eq(rr.grant == i)

        # Result absmax tracker (norm_i32's shift argument; reset on command accept).
        grant_abs = Signal(32)
        grant_s   = Signal((32, True))
        self.comb += [
            grant_s.eq(grant_data),
            grant_abs.eq(Mux(grant_s < 0, -grant_s, grant_s)),
        ]
        self.sync += [
            If(self.cmd.valid & self.cmd.ready,
                self.absmax.eq(0),
            ).Elif(~fetch_mode & grant_valid & (grant_abs > self.absmax),
                self.absmax.eq(grant_abs),
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

        # Scratchpad write port: fetch beats or gathered int32 results.
        slot = Signal(24)
        eng_gw_adr = Signal(len(scratchpad.gather.adr))
        eng_gw_we  = Signal(8)
        eng_gw_dat = Signal(256)
        self.comb += [
            If(cmd_r["ilv"],
                slot.eq((cmd_r["dst"] >> 2) + grant_row*self.n_channels + rr.grant),
            ).Else(
                slot.eq((cmd_r["dst"] >> 2) + rr.grant*cmd_r["rows"] + grant_row),
            ),
            If(fetch_mode,
                eng_gw_adr.eq((cmd_r["dst"] >> 5) + fetch_cnt),
                eng_gw_dat.eq(fetch_data),
                eng_gw_we.eq(Mux(fetch_valid, 0xFF, 0)),
            ).Else(
                eng_gw_adr.eq(slot[3:]),
                eng_gw_dat.eq(Replicate(grant_data, 8)),
                eng_gw_we.eq(Mux(grant_valid, 1 << slot[:3], 0)),
            )
        ]

        # Scratchpad port wiring (direct, or muxed with the vector unit).
        if vecunit is None:
            self.comb += [
                scratchpad.fill.adr.eq(eng_fill_adr),
                scratchpad.gather.adr.eq(eng_gw_adr),
                scratchpad.gather.dat_w.eq(eng_gw_dat),
                scratchpad.gather.we.eq(eng_gw_we),
            ]
        else:
            grant_vec = Signal()
            self.comb += [
                grant_vec.eq(vecunit.busy),
                If(grant_vec,
                    scratchpad.fill.adr.eq(vecunit.sp_rd_adr),
                    scratchpad.gather.adr.eq(vecunit.sp_wr_adr),
                    scratchpad.gather.dat_w.eq(vecunit.sp_wr_dat),
                    scratchpad.gather.we.eq(vecunit.sp_wr_we),
                ).Else(
                    scratchpad.fill.adr.eq(eng_fill_adr),
                    scratchpad.gather.adr.eq(eng_gw_adr),
                    scratchpad.gather.dat_w.eq(eng_gw_dat),
                    scratchpad.gather.we.eq(eng_gw_we),
                ),
                vecunit.sp_rd_dat.eq(scratchpad.fill.dat_r),
            ]

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
        fsm.act("RUN",
            self.busy.eq(1),
            If(Mux(fetch_mode,
                   fetch_cnt == cmd_r["wlen"],
                   result_cnt == (cmd_r["rows"]*self.n_channels + cmd_r["rem"])),
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
                    result_cnt.eq(result_cnt + 1),
                )
            ),
        ]

def reduce_or(sigs):
    from functools import reduce
    from operator import or_
    return reduce(or_, sigs)

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

from functools import reduce
from operator import and_

from migen import *
from migen.genlib.fifo import SyncFIFO
from migen.genlib.roundrobin import RoundRobin, SP_CE

from litex.gen import *

from litex.soc.interconnect import stream

from trixium.layout import BEAT_BYTES
from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.ternary_mac import BonsaiChannelPipe

# BonsaiMatVecEngine -------------------------------------------------------------------------------

class BonsaiMatVecEngine(LiteXModule):
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96, vecunit=None, seq=None,
                 dual=True, chan_stride=0):
        # chan_stride: byte distance between consecutive channels' weight regions. The HBM IP
        # is configured with the AXI switch enabled (USER_SWITCH_ENABLE_00/01), so the address
        # - not the port it arrives on - selects the pseudo-channel: with stride 0 every
        # channel reads the same bytes. Confirmed on hardware, where `peek <c> 4000000`
        # returned byte-identical data for all eight channels despite each having been
        # uploaded a different payload. Sim keeps 0 because its per-channel HBM models are
        # independent memories (each channel already has its own address space there).
        # When a BonsaiVectorUnit is passed, the scratchpad fill/gather ports are muxed
        # between the engine and the vector unit (granted to the vector unit while it is
        # busy - firmware serializes engine ops and vector ops, so contention is
        # structurally impossible; never a 5th RAM port).
        assert chan_stride == 0 or (chan_stride & (chan_stride - 1)) == 0, \
            "chan_stride must be 0 or a power of two (the STORE path shifts by it)"
        chan_shift = chan_stride.bit_length() - 1 if chan_stride else 0

        # Keep the vector unit's busy signal, not the module: assigning a Module to an
        # attribute registers it as a submodule, and it is already one of the SoC's. The
        # double registration renames its CSRs (vecunit_* -> vecunit_vecunit_*, plus an
        # engine_vecunit_* alias) and moves them, which silently breaks firmware/vec.c.
        self.vec_busy_in = vecunit.busy if vecunit is not None else Signal()

        # Overlap counters. The MAC array is idle ~92% of a token and the sequencer runs one
        # descriptor at a time, so the question that decides whether dual-issue is worth
        # building is: how much of the token has BOTH units busy already? Free-running from
        # reset; the headroom is total - eng - vec + both, i.e. cycles where NEITHER ran,
        # which is pure serialisation and dispatch.
        #
        # Counted here rather than in the CSR front end because the engine already holds a
        # vec_busy_in reference; adding one in csr_front would mean a new integration signal
        # in BOTH SoC definitions, which is exactly the drift the guard test exists to catch.
        self.cyc_total = Signal(48)
        self.cyc_eng   = Signal(48)
        self.cyc_vec   = Signal(48)
        self.cyc_both  = Signal(48)
        self.busy      = Signal()      # declared early: the counters below need it
        self.sync += [
            self.cyc_total.eq(self.cyc_total + 1),
            If(self.busy,          self.cyc_eng.eq(self.cyc_eng + 1)),
            If(self.vec_busy_in,   self.cyc_vec.eq(self.cyc_vec + 1)),
            If(self.busy & self.vec_busy_in, self.cyc_both.eq(self.cyc_both + 1)),
        ]
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
            ("xred",   1),   # AXPY only: sum the per-channel partials IN HARDWARE and write
                             # one cols-long result instead of n_channels of them. Without
                             # it the firmware pays n_channels-1 reduce_acc calls per head
                             # (7 at N=8), which measured 3 cycles/element and is 9 of the
                             # 13 per-head attention descriptors. int32 add wraps and is
                             # associative, so the tree here is bit-identical to the
                             # sequential vector-unit reduce it replaces.
            ("rep",    5),   # STORE only: number of packets (0/1 = one). Packet r goes to
                             # wbase + r*dstride beats from xsrc + r*sstride beats - the
                             # 16 per-layer KV packets in ONE op (F1-1, 2026-08-29).
            ("dstride", 24), # STORE: HBM beats between consecutive packets.
            ("sstride", 8),  # STORE: scratchpad beats between consecutive source slots.
        ])
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
        self.store_cmd  = stream.Endpoint([("chan", 5), ("addr", 33), ("len", 24)])
        # addr is 33 bits: with port-matched 256 MB windows channel 15 lives at
        # PC16 = 0x1_0400_0000 (stack 1's first pseudo-channel, N=16 - 2026-08-29).
        self.store_data = stream.Endpoint([("data", 256)])
        self.store_busy = Signal() # From the loader: executing our store (incl. B drain).

        # # #

        # Channels ----------------------------------------------------------------------------
        fetch_mode = Signal() # kind == 2 for the latched command.
        fetch_chan = Signal(5)

        self.readers = []
        self.pipes   = []
        # Per-cycle port arbitration state (docs/overlap-design.md). rd_free: the engine
        # may put an address on the scratchpad R port this cycle. fetch_rdy: the gather
        # write FIFO has headroom for a fetch beat. gw_pending: engine writes queued but
        # not yet in RAM (done waits on it). All default to the serial behavior when no
        # vector unit is integrated.
        rd_free    = Signal()
        fetch_rdy  = Signal()
        gw_pending = Signal()

        fetch_valids = []
        fetch_datas  = []
        for i, axi in enumerate(hbm_axis):
            # n_ids=4: the HBM controller serializes same-ARID reads (one 16-beat burst
            # per round trip - 33.7 cycles/beat measured); round-robin ARIDs let it pipeline.
            reader = BonsaiAXIReader(axi, n_ids=4)
            pipe   = BonsaiChannelPipe(xbuf_groups=xbuf_groups, chan_id=i, n_channels=n,
                                       dual=dual)
            setattr(self.submodules, f"reader{i}", reader)
            setattr(self.submodules, f"pipe{i}",   pipe)
            # Relay register between the reader (its FIFO sits at the HBM port, bottom of
            # the die) and the pipe: n16 timing #3c had the FIFO output reaching codes_r
            # with zero logic levels and 4.5 ns of route. Latency only.
            rbuf = stream.Buffer([("data", 256)])
            setattr(self.submodules, f"rbuf{i}", rbuf)
            self.comb += reader.source.connect(rbuf.sink)
            rsrc = rbuf.source
            # Registered per channel (n16 timing #3d: the combinational decode from cmd_r
            # reached a far pipe's scale-RAM write enables with 5.4 ns of route). Stable
            # from the cycle after START; the reader is commanded later than that.
            is_fetch = Signal(name=f"is_fetch{i}")
            self.sync += is_fetch.eq(fetch_mode & (fetch_chan == i))
            self.comb += [
                # Normal: beats feed the compute pipe. Fetch: beats go to the scratchpad.
                pipe.sink.valid.eq(rsrc.valid & ~is_fetch),
                pipe.sink.data.eq(rsrc.data),
                # Fetch beats honor the gather-FIFO headroom (a vec store can steal the
                # W port for a cycle); compute beats honor the pipe's result-FIFO
                # backpressure (drops results past N=2 otherwise - N/row_beats > 1 gather/cycle).
                rsrc.ready.eq((is_fetch & fetch_rdy) | (~is_fetch & pipe.sink.ready)),
            ]
            fetch_valids.append(rsrc.valid & is_fetch)
            fetch_datas.append(rsrc.data)
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command ---------------------------------------------------------------------
        cmd_r = {}
        for name in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                     "bias", "rshift", "xsrc", "dst", "chan", "rem", "ilv", "supw",
                     "rep", "dstride", "sstride",
                     "qn", "qstride", "rshift2", "xred"):
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

        # Broadcast op params to pipes (channels < rem run one extra row). REGISTERED
        # per pipe (n16 timing #3c: the combinational broadcast reached a far pipe's
        # x-buffer address with 3.9 ns of route). They load during START from the new
        # cmd_r and are stable one cycle later, together with the registered start pulse
        # below - the first beat or fill write reaches a pipe later than that.
        for i, pipe in enumerate(self.pipes):
            self.sync += [
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
        # Multi-packet STORE state: packet index, running HBM address, running source
        # beat offset (all reset on accept; advanced between packets).
        st_rep        = Signal(5)
        st_addr_r     = Signal(32)
        st_src_off    = Signal(13)
        st_last_pkt   = Signal()
        st_next       = Signal()   # Pulse: advance to the next packet.
        self.comb += st_last_pkt.eq((st_rep + 1 >= cmd_r["rep"]))
        # Scratchpad read address for the fill and the STORE stream, kept as a COUNTER
        # (n16 timing #3d: xsrc + Mux(store_mode, off + cnt, beat) put two carry chains and
        # the store_mode decode in front of a 64-BRAM address net). Reset to xsrc at START,
        # advanced exactly when fill_beat / st_cnt advance, re-based on each STORE packet.
        eng_fill_adr = Signal(len(scratchpad.fill.adr))
        xsrc_w = Signal(len(scratchpad.fill.adr))
        self.comb += xsrc_w.eq(cmd_r["xsrc"] >> 5)
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
        # Relay stage first (n16 timing #3c): the BRAM output fanned out to sixteen
        # per-pipe copies across the die. One register next to the scratchpad, then the
        # per-pipe copies off a flop the tool may replicate. Adds a cycle to the fill; the
        # last writes still land during ISSUE.
        fill_act1 = Signal()
        fill_pos1 = Signal(10)
        fill_dat1 = Signal(256)
        self.sync += [
            fill_act1.eq(fill_active_d),
            fill_pos1.eq(fill_pos),
            fill_dat1.eq(scratchpad.fill.dat_r),
        ]
        for i, pipe in enumerate(self.pipes):
            fa2 = Signal(name=f"fill_act2_{i}")
            fp2 = Signal(10, name=f"fill_pos2_{i}")
            fd2 = Signal(256, name=f"fill_dat2_{i}")
            for sig in (fa2, fp2, fd2):
                sig.attr.add(("dont_touch", "true"))
            self.sync += [
                fa2.eq(fill_act1),
                fp2.eq(fill_pos1),
                fd2.eq(fill_dat1),
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
        # Interleaved (position-major) ops merge every pipe whose head falls in the same
        # scratchpad beat as the granted head: slot = row*N + chan, so within one beat all
        # lanes map to distinct pipes (N power-of-2) and up to 8 results retire per cycle.
        # This is the gather-widen: at N channels the aggregate score rate N/row_beats
        # exceeds 1/cycle from N=4 up, and merged retirement keeps the drain above it.
        ilv_mode  = Signal()
        xred_mode = Signal()
        dstw      = Signal(24)
        self.comb += [
            ilv_mode.eq(cmd_r["ilv"] & ~axpy_mode & ~fetch_mode & ~store_mode),
            xred_mode.eq(axpy_mode & cmd_r["xred"]),
            dstw.eq(cmd_r["dst"] >> 2),
        ]
        # Per-pipe relay register in front of the gather (n16 timing #3c: the result
        # FIFO's asynchronous read fed the slot arithmetic, the merge compare and the lane
        # muxes in one cycle - 14 levels, 6.4 ns). The scratchpad slot of each result is
        # computed HERE, from the pipe's FIFO output and the (op-static) command fields,
        # and travels registered with the result: the gather then only compares and
        # selects. Every mode's addressing is folded into one field:
        #   ilv      slot = dst + qoff + row*N + chan       (position-major, merged beats)
        #   xred     slot = dst + qoff + row                (one summed vector)
        #   axpy     slot = dst + qoff + chan*cols + row    (wide drain rows)
        #   default  slot = dst + qoff + chan*rows + row    (channel-major)
        # The pipe's stream handshake is unchanged (consumed == counted still holds: the
        # counting stage sits after the relay, on the same signals it always used).
        gsrc = []
        for i, pipe in enumerate(self.pipes):
            gb = stream.Buffer([("row", 16), ("data", 256), ("wide", 1), ("head", 1),
                                ("slot", 24)])
            setattr(self.submodules, f"gbuf{i}", gb)
            self.comb += [
                gb.sink.valid.eq(pipe.source.valid),
                pipe.source.ready.eq(gb.sink.ready),
                gb.sink.row.eq(pipe.source.row),
                gb.sink.data.eq(pipe.source.data),
                gb.sink.wide.eq(pipe.source.wide),
                gb.sink.head.eq(pipe.source.head),
                gb.sink.slot.eq(dstw + Mux(pipe.source.head, cmd_r["qstride"], 0)
                    + Mux(ilv_mode, pipe.source.row*n + i,
                      Mux(xred_mode, pipe.source.row,
                          pipe.source.row + Mux(axpy_mode, i*cmd_r["cols"], i*cmd_r["rows"])))),
            ]
            gsrc.append(gb.source)
        grant_valid = Array([g.valid for g in gsrc])[rr.grant]
        grant_row   = Array([g.row   for g in gsrc])[rr.grant]
        grant_data  = Array([g.data  for g in gsrc])[rr.grant]
        grant_wide  = Array([g.wide  for g in gsrc])[rr.grant]
        grant_slot  = Signal(24)
        merge_mask  = Signal(n)
        self.comb += grant_slot.eq(Array([g.slot for g in gsrc])[rr.grant])
        self.comb += merge_mask.eq(Cat(*[
            gsrc[i].valid & ~gsrc[i].wide & (gsrc[i].slot[3:] == grant_slot[3:])
            for i in range(n)]))

        # Cross-channel reduce (AXPY + cmd.xred). Every pipe drains its partial vector as
        # cols/8 beats in the SAME order - beat k is always lane group k, pushed by the drain
        # FSM's d_cnt - so the channels are in lockstep by construction and no addressing is
        # needed to line them up: wait until all n have a beat, add them lane-wise, write one
        # beat, and consume all n together.
        #
        # Head-of-line blocking is not a hazard here: every channel produces exactly cols/8
        # beats regardless of `rem` (rem varies the number of POSITIONS a channel accumulates,
        # not the length of its output), and the drain runs after the reader has finished, so
        # a stalled axw_fifo cannot deadlock against an in-flight read.
        xred_all  = Signal()
        self.comb += xred_all.eq(xred_mode & reduce(and_, [g.valid for g in gsrc]))
        xred_dat = Signal(256)
        def _addtree(sigs):
            # Balanced: a left fold is a chain of n-1 adders (15 carry chains deep at
            # N=16); the tree is log2(n) deep and, being wrap-add, bit-identical. Exactly
            # ONE level is registered whatever the depth (n16 timing #3d: four carry levels
            # behind a far-placed relay were the eng_gw_dat worst path) - the second level
            # when there is one, else the only level, else the lone input - so the summed
            # beat always lands one cycle later and the xred write below follows it. (An
            # earlier form registered "level 1" only: at N=2 there is no level 1, nothing
            # was registered, and the data ran a cycle ahead of its write enable.)
            depth = (len(sigs) - 1).bit_length()
            if depth == 0:
                r = Signal(32)
                self.sync += r.eq(sigs[0])
                return r
            reg_level = min(1, depth - 1)
            level = 0
            while len(sigs) > 1:
                reg = (level == reg_level)
                nxt = []
                for a, b in zip(sigs[::2], sigs[1::2]):
                    m = Signal(32)
                    if reg:
                        self.sync += m.eq(a + b)
                    else:
                        self.comb += m.eq(a + b)
                    nxt.append(m)
                if len(sigs) % 2:
                    if reg:
                        r = Signal(32)
                        self.sync += r.eq(sigs[-1])
                        nxt.append(r)
                    else:
                        nxt.append(sigs[-1])
                sigs = nxt
                level += 1
            return sigs[0]
        for l in range(8):
            # int32 wrap add: the sum is truncated to 32 bits, which is exactly what the
            # vector unit's VADD does, so the result is bit-identical whatever the order.
            self.comb += xred_dat[32*l:32*l + 32].eq(
                _addtree([g.data[32*l:32*l + 32] for g in gsrc]))
        # xred write follows the registered tree level by one cycle.
        xred_all_d  = Signal()
        xred_slot_d = Signal(len(scratchpad.gather.adr))

        self.comb += [
            rr.request.eq(Cat(*[g.valid for g in gsrc])),
            rr.ce.eq(1),
        ]
        for i in range(n):
            self.comb += gsrc[i].ready.eq(
                Mux(xred_mode, xred_all,
                    Mux(ilv_mode, merge_mask[i], rr.grant == i)))

        # Merged lane build: within one beat each word lane belongs to at most one pipe.
        lane_we  = Signal(8)
        lane_dat = Signal(256)
        for l in range(8):
            sels = [merge_mask[i] & (gsrc[i].slot[0:3] == l) for i in range(n)]
            self.comb += lane_we[l].eq(reduce_or(sels))
            self.comb += lane_dat[32*l:32*l+32].eq(reduce_or(
                [Replicate(sels[i], 32) & gsrc[i].data[0:32]
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
        fetch_take = Signal() # A fetch beat is actually consumed this cycle.
        self.comb += [
            fetch_valid.eq(reduce_or(fetch_valids)),
            fetch_data.eq(Array(fetch_datas)[fetch_chan]),
            fetch_take.eq(fetch_valid & fetch_rdy),
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
        ).Elif(fetch_mode & fetch_take,
            # Equivalent to (eng_gw_we_c != 0) in fetch mode, but off the reader valids
            # directly: the OR-reduce of the muxed write enables dragged the pipes' FIFO
            # state into this CE (v15's worst path; fetch_rdy adds only a small level
            # compare from the gather FIFO).
            fetch_wcnt.eq(fetch_wcnt + 1),
        )
        # The granted result's slot, precomputed per pipe at the relay register (xred:
        # all n are on the same beat when xred_all, so the granted one's row is the row;
        # axpy: wide drain rows are 8-lane group starts, dst must be 32B-aligned).
        self.comb += slot.eq(grant_slot)
        self.sync += [
            xred_all_d.eq(xred_all),
            xred_slot_d.eq(slot[3:]),
        ]
        self.comb += [
            If(fetch_mode,
                eng_gw_adr_c.eq((cmd_r["dst"] >> 5) + fetch_wcnt),
                eng_gw_dat_c.eq(fetch_data),
                eng_gw_we_c.eq(Mux(fetch_take, 0xFF, 0)),
            ).Elif(ilv_mode,
                eng_gw_adr_c.eq(grant_slot[3:]),
                eng_gw_dat_c.eq(lane_dat),
                # NOT gated on grant_valid. A merged beat is consumed on merge_mask, which
                # only sets lanes for pipes that are themselves valid and whose slot is in
                # this beat, so lane_we is already zero when nothing merges. The extra gate
                # only broke the agreement between consuming and counting: with the
                # round-robin still pointing at a pipe that had just gone invalid, its stale
                # slot could match a valid pipe's beat, merge_mask took that pipe's result,
                # and grant_valid held the write enable at zero. The result vanished without
                # incrementing result_cnt, RUN waited for a count it could never reach, and
                # the engine sat busy holding the scratchpad grant (status 0x301).
                #
                # It needs pipes presenting at staggered times, so it only bites from two
                # rows per channel on - with one row all eight merge in a single beat, which
                # is why rows=1 works at every rem. Position 9 at 8 channels on the card.
                #
                # The address is safe without the gate: merge_mask requires
                # gsrc[i].slot[3:] == grant_slot[3:], so every lane written belongs here.
                eng_gw_we_c.eq(lane_we),
            ).Elif(xred_mode,
                eng_gw_adr_c.eq(xred_slot_d),
                eng_gw_dat_c.eq(xred_dat),
                eng_gw_we_c.eq(Mux(xred_all_d, 0xFF, 0)),
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
                rd_free.eq(1),
                fetch_rdy.eq(1),
                # Posted sideband writes drain whenever the engine is not writing.
                scratchpad.sbw_gnt.eq(scratchpad.sbw_pending & (sp_gw_we == 0)),
                scratchpad.sbr_allow.eq(1), # No vec/seq in this configuration.
                # gw_pending stays 0: writes hit the RAM directly, done needs no drain.
                scratchpad.fill.adr.eq(eng_fill_adr),
                scratchpad.gather.adr.eq(eng_gw_adr),
                scratchpad.gather.dat_w.eq(eng_gw_dat),
                scratchpad.gather.we.eq(sp_gw_we),
            ]
        else:
            # Per-cycle arbitration (docs/overlap-design.md): the units may run
            # CONCURRENTLY, so the grant is per cycle, not per op.
            #
            # R port: vec > seq > engine. A vec load claims the port for exactly its
            # E1 (address) cycle since the V1 pipeline (sp_rd_req; it was a 3-cycle EX1..EX2
            # window on the FSM unit); the engine's fill/store address generation simply
            # does not advance on those cycles (rd_free below). Seq descriptor fetches still
            # only happen while both units are idle.
            #
            # W port: engine writes are enqueued into a small FIFO instead of driving the
            # port directly - the merge/counting stage upstream MUST NOT stall (see the
            # consumed-vs-counted note above). Vec stores get the port while the FIFO has
            # headroom; at/above the mark the engine drains continuously and the vec unit
            # holds in EX2 (sp_wr_gnt). Deadlock-free: the engine never waits on the vec
            # unit mid-op, so the FIFO always drains. Depth 8 suffices: input is <= 1
            # write/cycle, and above the mark drain matches input.
            grant_seq = Signal()
            vec_rd    = Signal()
            vec_wr    = Signal()
            sb_headroom_r = Signal() # Registered W-FIFO headroom for the sideband grant.
            gw_w      = len(scratchpad.gather.adr) + 8 + 256
            wr_fifo   = SyncFIFO(width=gw_w, depth=8)
            self.submodules += wr_fifo
            f_adr = Signal(len(scratchpad.gather.adr))
            f_we  = Signal(8)
            f_dat = Signal(256)
            self.comb += Cat(f_adr, f_we, f_dat).eq(wr_fifo.dout)
            self.gw_ovf = Signal() # Sticky: an engine write hit a full FIFO (must never).
            self.sync += If((sp_gw_we != 0) & ~wr_fifo.writable, self.gw_ovf.eq(1))
            self.comb += [
                grant_seq.eq(0 if seq is None else seq.sp_active),
                vec_rd.eq(vecunit.sp_rd_req),
                # The seq's fetch FSM holds its address cycle when a vec load outranks
                # it, and while posted sideband writes are still in flight - a descriptor
                # read must never see a stale word (the ordering interlock that lets
                # firmware patch lists with zero care).
                *([] if seq is None else [seq.sp_gnt.eq(~vec_rd
                                                        & ~scratchpad.sbw_pending)]),
                # Posted sideband writes (see scratchpad.py): the drain takes the W
                # port on any cycle the vector store slot is idle and the gather FIFO
                # has registered headroom - an engine-local grant with no CPU-region
                # term in the cone, and the R port arbitration is untouched. The
                # headroom margin keeps gw_ovf impossible (level(T-1) < 5 bounds
                # level(T) <= 6 < 8 on a non-draining cycle).
                scratchpad.sbw_gnt.eq(scratchpad.sbw_pending & ~vec_wr & sb_headroom_r),
                # Read slot (C2-2): reads take the R port on any free cycle, below vec
                # loads and the seq fetch; the fill/store address just holds (rd_free).
                scratchpad.sbr_allow.eq(~vec_rd & ~grant_seq),
                rd_free.eq(~vec_rd & ~grant_seq & ~scratchpad.sbr_gnt_any),
                vec_wr.eq(vecunit.sp_wr_req & (wr_fifo.level < 6)),
                vecunit.sp_wr_gnt.eq(vec_wr),
                fetch_rdy.eq(wr_fifo.level < 6),
                gw_pending.eq(wr_fifo.readable | (sp_gw_we != 0)),
                wr_fifo.din.eq(Cat(eng_gw_adr, sp_gw_we, eng_gw_dat)),
                wr_fifo.we.eq(sp_gw_we != 0),
                wr_fifo.re.eq(~vec_wr & wr_fifo.readable & ~scratchpad.sbw_gnt),
                If(vec_rd,
                    scratchpad.fill.adr.eq(vecunit.sp_rd_adr),
                ).Elif(grant_seq,
                    scratchpad.fill.adr.eq(0 if seq is None else seq.sp_adr),
                ).Else(
                    scratchpad.fill.adr.eq(eng_fill_adr),
                ),
                If(vec_wr,
                    scratchpad.gather.adr.eq(vecunit.sp_wr_adr),
                    scratchpad.gather.dat_w.eq(vecunit.sp_wr_dat),
                    scratchpad.gather.we.eq(vecunit.sp_wr_we),
                ).Else(
                    scratchpad.gather.adr.eq(f_adr),
                    scratchpad.gather.dat_w.eq(f_dat),
                    scratchpad.gather.we.eq(Mux(wr_fifo.re, f_we, 0)),
                ),
                vecunit.sp_rd_dat.eq(scratchpad.fill.dat_r),
            ]
            self.sync += sb_headroom_r.eq(wr_fifo.level < 5)
            if seq is not None:
                self.comb += seq.sp_dat.eq(scratchpad.fill.dat_r)

        # Control FSM -------------------------------------------------------------------------
        self.fsm = fsm = FSM(reset_state="IDLE")
        start_pulse = Signal()
        for pipe in self.pipes:
            self.sync += pipe.start.eq(start_pulse) # One cycle behind, with the params.

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
            # STORE targets the one channel in cmd.chan, so it needs the same per-channel
            # offset the readers get; a power-of-two stride keeps that a shift, not a multiply.
            self.store_cmd.addr.eq(st_addr_r +
                (cmd_r["chan"] << chan_shift if chan_stride else 0)),
            self.store_cmd.len.eq(cmd_r["wlen"]),
            If(self.store_cmd.ready,
                NextState("ST_RUN"),
            )
        )
        fsm.act("ST_RUN",
            self.busy.eq(1),
            If(st_advance & (st_cnt == cmd_r["wlen"] - 1),
                If(st_last_pkt,
                    NextState("ST_DRAIN"),
                ).Else(
                    # Next packet: the loader accepts the command once it has drained
                    # this one (its store FSM returns to idle), so no extra wait here.
                    st_next.eq(1),
                    NextState("ST_CMD"),
                )
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
            # Advance only when the R port is ours this cycle (a vec load may claim it).
            If(rd_free,
                NextValue(fill_beat, fill_beat + 1),
                If(fill_beat == fill_total - 1,
                    NextState("FILL_LAST"),
                )
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
            *[r.sink.addr.eq(cmd_r["wbase"] + i*chan_stride)
              for i, r in enumerate(self.readers)],
            *[r.sink.len.eq(cmd_r["wlen"]
                + Mux(i < cmd_r["rem"], cmd_r["sbeats"] + cmd_r["cbeats"], 0))
              for i, r in enumerate(self.readers)],
            # NOTE: this waits on readers[0] alone. A reader only accepts while idle
            # (sink.ready = ~active), and readers do not always go idle together - with
            # rem != 0 they get different lengths, and a fetch commands one channel - so in
            # principle a reader still draining would miss its command. Measured on the card
            # it does not happen: at the stall every channel had delivered exactly the beats
            # commanded (ch0=10/10, ch1..7=5/5). Fixing it anyway cost WNS 0.015 -> -0.766 ns
            # on a design with no margin, and changed nothing, so it is left as written and
            # recorded here instead.
            If(self.readers[0].sink.ready,
                NextState("RUN"),
            )
        )
        mv_total = Signal(24)
        ax_total = Signal(24)
        # Test-visible aliases (no hardware: same signals under a name a testbench can find).
        # A stall in RUN is either "results never produced" or "produced but never retired",
        # and result_cnt against the expected total is what separates the two.
        self.dbg_result_cnt = result_cnt
        self.dbg_mv_total   = mv_total
        self.dbg_ax_total   = ax_total
        self.comb += [
            mv_total.eq(cmd_r["rows"]*self.n_channels + cmd_r["rem"]),
            # xred writes ONE cols-long vector, so the retire count drops by n_channels.
            # result_cnt counts written lanes (popcount of the write enables), so this is
            # all the accounting the merged beats need.
            ax_total.eq(Mux(cmd_r["xred"], cmd_r["cols"],
                            cmd_r["cols"]*self.n_channels)),
        ]
        fsm.act("RUN",
            self.busy.eq(1),
            If(Mux(fetch_mode,
                   fetch_cnt == cmd_r["wlen"],
                   # Dual-Q ops retire two heads' results per stream.
                   result_cnt == Mux(cmd_r["qn"],
                       Mux(axpy_mode, ax_total << 1, mv_total << 1),
                       Mux(axpy_mode, ax_total, mv_total)))
               & ~gw_pending, # Queued writes must be in RAM before done.
                self.done.eq(1),
                NextState("IDLE"),
            )
        )

        # Fill write-stage delay + result counting.
        self.sync += [
            fill_active_d.eq(fsm.ongoing("FILL") & rd_free),
            fill_beat_d.eq(fill_beat),
            # STORE stream: the read address holds until the beat is accepted; dat_r is
            # valid one stable-address cycle later (one dead cycle after each advance).
            # rd_free: dat_r next cycle is ours only if our address was on the port.
            st_rdv.eq(fsm.ongoing("ST_RUN") & ~st_advance & rd_free),
            # Fill/STORE read address counter (see eng_fill_adr above).
            If(start_pulse,
                eng_fill_adr.eq(xsrc_w),
            ).Elif(st_next,
                eng_fill_adr.eq(xsrc_w + st_src_off + cmd_r["sstride"]),
            ).Elif(fsm.ongoing("FILL") & rd_free,
                eng_fill_adr.eq(eng_fill_adr + 1),
            ).Elif(st_advance,
                eng_fill_adr.eq(eng_fill_adr + 1),
            ),
            If(start_pulse,
                st_cnt.eq(0),
                st_rep.eq(0),
                st_addr_r.eq(cmd_r["wbase"]),
                st_src_off.eq(0),
            ).Elif(st_next,
                st_cnt.eq(0),
                st_rep.eq(st_rep + 1),
                st_addr_r.eq(st_addr_r + (cmd_r["dstride"] << 5)),
                st_src_off.eq(st_src_off + cmd_r["sstride"]),
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

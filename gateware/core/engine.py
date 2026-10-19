#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""MatVecEngine: N-channel ternary/Q8 matvec engine.

One AXIReader + ChannelPipe per memory channel. A matvec command:

  1. FILL : broadcast x8 (cols bytes) from the scratchpad into every channel's x-buffer.
  2. ISSUE: give every reader the same {wbase, wlen} (identical relative layout per channel).
  3. RUN  : pipes stream rows; the gather round-robins int32 results into the scratchpad at
            dst + (channel*rows + row)*4 bytes; done when rows*n_channels results landed.

The command uses per-channel row counts (rows = M/n_channels) and byte addresses inside the
per-channel address space. Other kinds: fetch (memory row -> scratchpad), store (scratchpad
-> memory, through the loader's writers), axpy (weighted-V accumulate), and the optional
DeltaNet (5) and attention (6) stream units. Driven by the sequencer or the CSR front-end.
"""

from operator import and_, or_
from functools import reduce

from migen import *
from migen.genlib.fifo import SyncFIFO
from migen.genlib.roundrobin import RoundRobin, SP_CE

from litex.gen import *

from litex.soc.interconnect import stream

from gateware.memory.axi import AXIReader
from gateware.core.pipe import ChannelPipe
from gateware.core.trit5 import Trit5Unpacker
from gateware.units.deltanet_stream import DeltaNetStream
from gateware.units.attention_stream import AttentionStream

# Helpers ------------------------------------------------------------------------------------------

def reduce_or(sigs):
    return reduce(or_, sigs)

# Result Packer ------------------------------------------------------------------------------------

class _ResultPacker(LiteXModule):
    """The per-pipe relay register in front of the gather, which in plain-matvec mode also
    coalesces a channel's consecutive results into full 8-lane beats.

    One int32 per W-port beat would hold the scratchpad write port for most of a stream and
    starve the vector unit. Plain-mode rows are channel-major and contiguous per channel
    (slot = dst + chan*rows + row), so a channel's 8 consecutive results are one aligned beat.

    Structure, chosen for timing:
    - This is the relay: its registers feed the gather's N:1 mux and the round-robin.
    - Its input is registered too (the engine puts a stream.Buffer in front), so every cone
      here starts at flops: the pipe FIFO's asynchronous read and the slot arithmetic never
      reach the merge compare.
    - No data mux: the pipe replicates its 32-bit result across all 8 lanes, so lane l's
      register always loads bits [32l:32l+32] of the input and only its enable depends on
      the lane. Bypass modes (ilv/AXPY/xred/fetch) load all 8 lanes (a plain register).
    - Flush is local: the pipe's last row is static per op (rows_last), so the packer
      flushes its own partial beat when it has consumed that row.

    In pack mode a result whose beat matches the held one merges into its lane; otherwise
    the held beat is emitted (displace) and the new result loads. A full beat is presented
    as soon as it is full. Byte-exact: the memory sees the same lanes with the same data,
    only in fewer, wider writes.
    """
    def __init__(self):
        layout = [("row", 16), ("data", 256), ("wide", 1), ("head", 1), ("slot", 24)]
        self.sink   = stream.Endpoint(layout)
        self.source = stream.Endpoint(layout + [("lanes", 8)])
        self.enable    = Signal()   # Plain-matvec mode: pack. Otherwise a plain relay.
        self.clear     = Signal()   # Op start.
        self.rows_last = Signal(16) # This pipe's last row index (static per op).
        self.qn        = Signal()   # Dual-Q: the last result is (rows_last, head 1).

        # # #

        # Held entry (the payload is reset-less: qualified by p_v).
        p_v       = Signal()
        p_row     = Signal(16,  reset_less=True)
        p_dat     = Signal(256, reset_less=True)
        p_wide    = Signal(reset_less=True)
        p_head    = Signal(reset_less=True)
        p_slot    = Signal(24,  reset_less=True)
        p_lanes   = Signal(8,   reset_less=True)
        last_seen = Signal() # The pipe's final result has been consumed.

        # Merge / emit / accept decisions.

        in_v     = Signal()
        lane     = Signal(3)
        merge    = Signal() # Same beat as the held one: add a lane.
        displace = Signal() # A new entry that cannot merge: emit the held one.
        full     = Signal()
        is_last  = Signal()
        accept   = Signal()
        self.comb += [
            in_v.eq(self.sink.valid),
            lane.eq(self.sink.slot[0:3]),
            merge.eq(self.enable & p_v & (p_slot[3:] == self.sink.slot[3:])),
            displace.eq(in_v & p_v & ~merge),
            full.eq(p_lanes == 0xFF),
            is_last.eq((self.sink.row == self.rows_last) & (~self.qn | self.sink.head)),
            # Emit: as a plain relay whenever something is held; packing, when the beat is
            # full, when a new beat needs the slot, or when the pipe is done and this is
            # its partial tail.
            self.source.valid.eq(p_v & (~self.enable | full | displace
                                        | (last_seen & ~in_v))),
            self.source.row.eq(p_row),
            self.source.data.eq(p_dat),
            self.source.wide.eq(p_wide),
            self.source.head.eq(p_head),
            self.source.slot.eq(p_slot),
            self.source.lanes.eq(p_lanes),
            # Take the input when the slot is free, when it merges, or when the held
            # entry leaves this cycle.
            accept.eq(in_v & (~p_v | merge | self.source.ready)),
            self.sink.ready.eq(accept),
        ]

        # Held entry update.
        load = Signal()
        self.comb += load.eq(accept & ~merge)
        self.sync += [
            If(self.clear,
                p_v.eq(0),
                last_seen.eq(0),
            ).Else(
                If(load,
                    p_v.eq(1),
                    p_row.eq(self.sink.row),
                    p_wide.eq(self.sink.wide),
                    p_head.eq(self.sink.head),
                    p_slot.eq(self.sink.slot),
                    p_lanes.eq(Mux(self.enable, 1 << lane, 0xFF)),
                ).Elif(accept & merge,
                    p_lanes.eq(p_lanes | (1 << lane)),
                ).Elif(self.source.valid & self.source.ready,
                    p_v.eq(0),
                ),
                If(accept & is_last,
                    last_seen.eq(1),
                ),
            ),
        ]
        # Lane registers: no data mux - the pipe replicates a plain result into every
        # lane, so lane l always loads its own bits; only the enable is lane-dependent.
        for l in range(8):
            self.sync += If(accept & (~self.enable | (lane == l)),
                            p_dat[32*l:32*l + 32].eq(self.sink.data[32*l:32*l + 32]))

# MatVec Engine ------------------------------------------------------------------------------------

class MatVecEngine(LiteXModule):
    """N-channel matvec engine: one AXIReader + ChannelPipe per memory channel (see module doc)."""
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96, vecunit=None, seq=None,
                 chan_stride=0, chan_mod=0, chan_skew=0, dnstream=False, attnstream=False,
                 trit5=False, prefetch=False, reader_fifo_depth=512, reader_max_bursts=8):
        # chan_stride: byte distance between consecutive channels' weight regions. With the
        # HBM AXI switch enabled the address (not the port) selects the pseudo-channel, so
        # each channel needs its own window; 0 when every channel has its own address space.
        # When a VectorUnit is passed, the scratchpad fill/gather ports are arbitrated per
        # cycle between the engine, the vector unit and the sequencer (see below).
        # Without chan_mod the STORE path shifts the channel index by it (a power of two);
        # with chan_mod it adds a mux of per-window constants, so any multiple of 32 works
        # (e.g. a DDR controller split evenly between its windows).
        assert chan_stride == 0 or chan_mod or (chan_stride & (chan_stride - 1)) == 0, \
            "chan_stride must be 0 or a power of two without chan_mod (the STORE path shifts by it)"
        chan_shift = chan_stride.bit_length() - 1 if chan_stride else 0
        # chan_mod: channels per memory controller, a power of two (0 = one flat address
        # space, e.g. HBM). With several controllers (e.g. four DDR4 channels) engine channel
        # c sits on controller c // chan_mod and its window is that controller's local weight
        # base + (c % chan_mod)*stride. The mask is applied on the read, fetch and STORE paths
        # alike (the loader masks the upload path the same way).
        assert chan_mod == 0 or (chan_mod & (chan_mod - 1)) == 0, \
            "chan_mod must be 0 or a power of two (the STORE path masks by it)"
        # chan_skew: an extra per-channel offset of (c % chan_mod)*chan_skew bytes on top of
        # the stride, so that the channels sharing a DDR4 controller start in different
        # banks: with power-of-two strides they would all stream through the same bank and
        # thrash its row.
        assert chan_skew == 0 or chan_mod, "chan_skew needs chan_mod"

        def _chan_off(i): # Python int: per-reader constant.
            local = (i % chan_mod) if chan_mod else i
            return local*chan_stride + local*chan_skew
        def _chan_sel(c): # Signal: the STORE path's per-op channel.
            return (c & (chan_mod - 1)) if chan_mod else c
        def _chan_off_of(sel): # Signal: the STORE path's window offset, a mux of constants.
            return Array([k*chan_stride + k*chan_skew for k in range(chan_mod)])[sel]

        # Keep the vector unit's busy signal, not the module: assigning a Module to an
        # attribute would register it as a submodule a second time (and rename its CSRs).
        self.vec_busy_in = vecunit.busy if vecunit is not None else Signal()

        # Overlap counters, free-running from reset: total, engine busy, vector unit busy
        # and both busy (exposed by the CSR front-end).
        self.cyc_total = Signal(64)
        self.cyc_eng   = Signal(64)
        self.cyc_vec   = Signal(64)
        self.cyc_both  = Signal(64)
        self.busy      = Signal() # Declared early: the counters below need it.
        self.sync += [
            self.cyc_total.eq(self.cyc_total + 1),
            If(self.busy,          self.cyc_eng.eq(self.cyc_eng + 1)),
            If(self.vec_busy_in,   self.cyc_vec.eq(self.cyc_vec + 1)),
            If(self.busy & self.vec_busy_in, self.cyc_both.eq(self.cyc_both + 1)),
        ]
        self.n_channels = n = len(hbm_axis)

        self.cmd = stream.Endpoint([
            ("kind",   3),   # 0: ternary, 1: q8, 2: fetch (mem -> SP), 3: store (SP -> mem),
                             # 4: axpy (weighted-V: per-lane accumulate, cols results/chan),
                             # 5: DeltaNet stream (dnstream only: rows = heads, xsrc = the
                             # head record table, wbase/wlen = every channel's state span,
                             # read there and written back through the loader's writers at
                             # wbase + dstride beats, dstride signed: 0 rewrites in place),
                             # 6: attention stream (attnstream only: xsrc = the unit's record;
                             # K or V rows position-striped: rows/rem/wbase/wlen as the KV ops,
                             # sbeats + cbeats = beats per row, merged in position order).
                             # 7: weight prefetch (prefetch only: the next matvec's read
                             # commands - wbase/wlen/rem/sbeats/cbeats as that op - issued now,
                             # its beats wait in the readers' FIFOs; done at once).
            ("rows",   16),  # Rows per channel.
            ("cols",   14),  # K.
            ("wbase",  32),  # Per-channel weight byte address (beat-aligned).
            ("wlen",   24),  # Per-channel weight length in beats.
            ("sbeats", 4),   # Scale beats per row.
            ("cbeats", 8),   # Code beats per row.
            ("pk",     1),   # Ternary codes stored 5 trits per byte (trit5: Trit5Unpacker).
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
            ("xred",   1),   # AXPY only: sum the per-channel partials in hardware and write
                             # one cols-long result instead of n_channels of them (int32
                             # wrap add: bit-identical to a sequential vector-unit reduce).
            ("rep",    5),   # STORE only: number of packets (0/1 = one). Packet r goes to
                             # wbase + r*dstride beats from xsrc + r*sstride beats.
            ("dstride", 24), # STORE: memory beats between consecutive packets.
            ("sstride", 8),  # STORE: scratchpad beats between consecutive source slots.
        ])
        # sp_busy: busy + 1 cycle. The registered write stage's last beat lands the cycle after
        # busy drops; sideband arbitration (scratchpad.units_idle) must wait for it.
        self.sp_busy = Signal()
        self.done    = Signal()   # Pulse.
        self.error   = Signal()
        self.absmax  = Signal(32) # |int32| max over the last matvec's gathered results.
        # Argmax tap over the gathered results (matvec kinds only): value + global row
        # index (grant*rows + row), ties to the smallest index (matches np.argmax across
        # the round-robin arrival order). Reset on command accept.
        self.argmax_val = Signal((32, True), reset=-(1 << 31))
        self.argmax_idx = Signal(24, reset=0xFFFFFF)

        # STORE (kind 3): scratchpad beats -> one memory channel, through the loader's
        # per-channel AXI writers (wired by the integrator). cmd reuse: chan = dest channel,
        # wbase = memory byte address, wlen = beats, xsrc = scratchpad byte source.
        # addr is 33 bits: per-channel windows can extend past 4 GB.
        self.store_cmd  = stream.Endpoint([("chan", 5), ("addr", 33), ("len", 24)])
        self.store_data = stream.Endpoint([("data", 256)])
        self.store_busy = Signal() # From the loader: executing our store (incl. B drain).

        # DeltaNet stream (kind 5): channel i's reader feeds lane i of the unit and lane i's
        # updated state goes to channel i's writer through the loader's multi-channel store
        # (same wbase/wlen on every channel, the readers' per-channel offsets).
        self.with_dnstream = dnstream
        if dnstream:
            # dnstream=True: the 27B geometry on the exact operators; a dict overrides the
            # unit's arguments (tests: a small geometry on integer operators).
            opts = dict(dnstream) if isinstance(dnstream, (dict, tuple)) else {}
            self.dnstream = dn = DeltaNetStream(
                lanes=len(hbm_axis), adr_bits=len(scratchpad.fill.adr), **opts)
            self.mstore_cmd  = stream.Endpoint([("addr", 33), ("len", 24)])
            self.mstore_data = dn.sources

        self.with_attnstream = attnstream
        if attnstream:
            opts = dict(attnstream) if isinstance(attnstream, (dict, tuple)) else {}
            self.attnstream = at = AttentionStream(
                adr_bits=len(scratchpad.fill.adr), **opts)

        # # #

        # Channels ---------------------------------------------------------------------------------
        fetch_mode = Signal() # kind == 2 for the latched command.
        fetch_chan = Signal(5)
        run_state  = Signal() # The FSM is in RUN (the gate of prefetched beats).
        self.with_prefetch = prefetch

        self.readers = []
        self.pipes   = []
        self.unpackers = []
        read_cmds = []
        # Per-cycle port arbitration state. rd_free: the engine may put an address on the
        # scratchpad R port this cycle. fetch_rdy: the gather write FIFO has headroom for a
        # fetch beat. gw_pending: engine writes queued but not yet in RAM (done waits on it).
        # All default to the serial behavior when no vector unit is integrated.
        rd_free    = Signal()
        fetch_rdy  = Signal()
        gw_pending = Signal()

        fetch_valids = []
        fetch_datas  = []
        dn_mode = Signal() # kind == 5 for the latched command.
        at_mode = Signal() # kind == 6 for the latched command.
        at_chan = Signal(max=max(2, len(hbm_axis))) # The merge's current channel.
        at_take = Signal() # The merged beat is accepted this cycle.
        rsrc_valids, rsrc_datas = [], []
        for i, axi in enumerate(hbm_axis):
            # n_ids=4: the HBM controller serializes same-ARID reads (one 16-beat burst
            # per round trip); round-robin ARIDs let it pipeline.
            reader = AXIReader(axi, n_ids=4, fifo_depth=reader_fifo_depth,
                               max_bursts=reader_max_bursts)
            pipe   = ChannelPipe(xbuf_groups=xbuf_groups, chan_id=i, n_channels=n)
            setattr(self, f"reader{i}", reader)
            setattr(self, f"pipe{i}",   pipe)
            # trit5: the compute beats go through the unpacker (a pass-through unless the op's
            # codes are packed), the pipe sees 2-bit code beats either way.
            psink = pipe.sink
            if trit5:
                unp = Trit5Unpacker()
                setattr(self, f"unpack{i}", unp)
                self.comb += unp.source.connect(pipe.sink)
                psink = unp.sink
                self.unpackers.append(unp)
            # Relay register between the reader (its FIFO sits at the memory port) and the
            # pipe, for timing (route). Latency only.
            rbuf = stream.Buffer([("data", 256)])
            setattr(self, f"rbuf{i}", rbuf)
            self.comb += reader.source.connect(rbuf.sink)
            read_cmds.append(reader.sink)
            rsrc = rbuf.source
            # Registered per channel for timing (the decode from cmd_r fans out to far
            # pipes). Stable from the cycle after START; the reader is commanded later.
            is_fetch = Signal(name=f"is_fetch{i}")
            self.sync += is_fetch.eq(fetch_mode & (fetch_chan == i))
            is_dn = C(0, 1)
            if dnstream:
                is_dn = Signal(name=f"is_dn{i}")
                self.sync += is_dn.eq(dn_mode)
                self.comb += [
                    dn.sinks[i].valid.eq(rsrc.valid & is_dn),
                    dn.sinks[i].data.eq(rsrc.data),
                ]
            is_at = C(0, 1)
            if attnstream:
                is_at = Signal(name=f"is_at{i}")
                self.sync += is_at.eq(at_mode)
            # Prefetch: beats may arrive before their op runs (a kind-7 op issued the reads), so
            # every consumer takes them only while an op is in RUN (registered per channel).
            # Without prefetch the reads are issued in ISSUE, beats never arrive earlier.
            if prefetch:
                consume = Signal(name=f"consume{i}")
                self.sync += consume.eq(run_state)
                rsrc_gated = stream.Endpoint([("data", 256)])
                self.comb += [
                    rsrc_gated.valid.eq(rbuf.source.valid & consume),
                    rsrc_gated.data.eq(rbuf.source.data),
                    rbuf.source.ready.eq(rsrc_gated.ready & consume),
                ]
                rsrc = rsrc_gated
            self.comb += [
                # Normal: beats feed the compute pipe. Fetch: beats go to the scratchpad.
                psink.valid.eq(rsrc.valid & ~is_fetch & ~is_dn & ~is_at),
                psink.data.eq(rsrc.data),
                # Fetch beats honor the gather-FIFO headroom (a vec store can steal the
                # W port for a cycle); compute beats honor the pipe's result-FIFO
                # backpressure.
                rsrc.ready.eq((is_fetch & fetch_rdy)
                              | (~is_fetch & ~is_dn & ~is_at & psink.ready)
                              | (is_dn & (dn.sinks[i].ready if dnstream else 0))
                              | (is_at & at_take & (at_chan == i))),
            ]
            rsrc_valids.append(rsrc.valid)
            rsrc_datas.append(rsrc.data)
            fetch_valids.append(rsrc.valid & is_fetch)
            fetch_datas.append(rsrc.data)
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command --------------------------------------------------------------------------
        # Reset-less but for the kind (its mode decodes steer the read and gather handshakes):
        # every other field is loaded at accept and only read from START on.
        cmd_r = {}
        for name in (("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats", "pk",
                     "bias", "rshift", "xsrc", "dst", "chan", "rem", "ilv", "supw",
                     "rep", "dstride", "sstride",
                     "qn", "qstride", "rshift2", "xred")):
            cmd_r[name] = Signal.like(getattr(self.cmd, name), reset_less=(name != "kind"))
        self.sync += If(self.cmd.valid & self.cmd.ready,
            *[cmd_r[k].eq(getattr(self.cmd, k)) for k in cmd_r])
        command_accept = self.cmd.valid & self.cmd.ready
        # The result trackers (absmax, argmax) reset on an op's accept, but not on a prefetch's:
        # it slips between a matvec and the firmware's read of its absmax.
        tracker_reset = command_accept & ((self.cmd.kind != 7) if prefetch else 1)
        store_mode = Signal()
        axpy_mode  = Signal()
        pf_mode    = Signal()
        self.comb += [
            pf_mode.eq(cmd_r["kind"] == 7 if prefetch else 0),
            fetch_mode.eq(cmd_r["kind"] == 2),
            store_mode.eq(cmd_r["kind"] == 3),
            axpy_mode.eq(cmd_r["kind"] == 4),
            dn_mode.eq(cmd_r["kind"] == 5 if dnstream else 0),
            at_mode.eq(cmd_r["kind"] == 6 if attnstream else 0),
            fetch_chan.eq(cmd_r["chan"]),
        ]
        # Broadcast op params to pipes (channels < rem run one extra row), registered per
        # pipe for timing. They load during START from the new cmd_r and are stable one
        # cycle later, together with the registered start pulse below.
        for i, pipe in enumerate(self.pipes):
            # dont_touch: the copies are logically identical and would otherwise be merged
            # back into one high-fanout flop.
            for s in (pipe.kind, pipe.scale_beats, pipe.code_beats, pipe.bias,
                      pipe.rshift, pipe.qn, pipe.rshift2):
                s.attr.add(("dont_touch", "true"))
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
        for unp in self.unpackers:
            self.sync += [
                unp.packed.eq(cmd_r["pk"] & (cmd_r["kind"] == 0)),
                unp.scale_beats.eq(cmd_r["sbeats"]),
                unp.code_beats.eq(cmd_r["cbeats"]),
            ]
        # Overflow is included so any future rate mismatch fails loud (sticky error bit)
        # instead of dropping results and hanging the op.
        pipe_errors = [p.error | p.overflow for p in self.pipes]
        self.comb += self.error.eq(reduce_or(pipe_errors)
                                  | ((at_mode & at.error) if attnstream else 0))

        # x-fill -----------------------------------------------------------------------------------
        fill_beat  = Signal(10)
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
        fill_beat_d   = Signal(10, reset_less=True)
        st_cnt        = Signal(13) # STORE beat counter (address holds until accepted).
        # Multi-packet STORE state: packet index, running memory address, running source
        # beat offset (all reset on accept; advanced between packets).
        st_rep        = Signal(5)
        st_addr_r     = Signal(32, reset_less=True)
        st_src_off    = Signal(13, reset_less=True)
        st_last_pkt   = Signal()
        st_next       = Signal()   # Pulse: advance to the next packet.
        self.comb += st_last_pkt.eq((st_rep + 1 >= cmd_r["rep"]))
        # Scratchpad read address for the fill and the STORE stream, kept as a counter (no
        # adder in front of the BRAM address, timing). Reset to xsrc at START, advanced
        # exactly when fill_beat / st_cnt advance, re-based on each STORE packet.
        eng_fill_adr = Signal(len(scratchpad.fill.adr), reset_less=True)
        xsrc_w = Signal(len(scratchpad.fill.adr))
        self.comb += xsrc_w.eq(cmd_r["xsrc"] >> 5)
        # STORE stream through a 4-deep prefetch FIFO: reads are issued ahead (whenever the
        # R port is ours and credit remains), so the stream runs at one beat per clock.
        self.st_fifo = st_fifo = ResetInserter()(stream.SyncFIFO([("data", 256)], 4))
        st_run   = Signal()    # fsm.ongoing("ST_RUN"), assigned once the FSM exists.
        st_iss   = Signal(13)  # Beats read in the current packet.
        st_pend  = Signal()    # A read issued last cycle: its data is on fill.dat_r now.
        st_issue = Signal()
        self.comb += [
            st_issue.eq(st_run & rd_free & (st_iss < cmd_r["wlen"]) &
                        ((st_fifo.level + st_pend) < 3)),
            st_fifo.sink.valid.eq(st_pend),
            st_fifo.sink.data.eq(scratchpad.fill.dat_r),
            self.store_data.valid.eq(st_fifo.source.valid),
            self.store_data.data.eq(st_fifo.source.data),
            st_fifo.source.ready.eq(self.store_data.ready),
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
        # Fill broadcast, registered for timing: one relay register next to the scratchpad,
        # then per-pipe copies (dont_touch) the placer can pull next to each x-buffer;
        # position/enable are delayed to stay aligned. The last writes land during ISSUE
        # (x-buffer reads only start once beats arrive).
        fill_act1 = Signal()
        fill_pos1 = Signal(10,  reset_less=True)
        fill_dat1 = Signal(256, reset_less=True)
        self.sync += [
            fill_act1.eq(fill_active_d),
            fill_pos1.eq(fill_pos),
            fill_dat1.eq(scratchpad.fill.dat_r),
        ]
        for i, pipe in enumerate(self.pipes):
            fa2 = Signal(name=f"fill_act2_{i}")
            fp2 = Signal(10, name=f"fill_pos2_{i}", reset_less=True)
            fd2 = Signal(256, name=f"fill_dat2_{i}", reset_less=True)
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

        # Gather -----------------------------------------------------------------------------------
        self.rr = rr = RoundRobin(n, SP_CE)
        result_cnt = Signal(32)
        # Interleaved (position-major) ops merge every pipe whose head falls in the same
        # scratchpad beat as the granted head: slot = row*N + chan, so within one beat all
        # lanes map to distinct pipes (N power-of-2) and up to 8 results retire per cycle,
        # keeping the drain above the aggregate score rate N/row_beats.
        ilv_mode  = Signal()
        xred_mode = Signal()
        dstw      = Signal(24)
        self.comb += [
            ilv_mode.eq(cmd_r["ilv"] & ~axpy_mode & ~fetch_mode & ~store_mode),
            xred_mode.eq(axpy_mode & cmd_r["xred"]),
            dstw.eq(cmd_r["dst"] >> 2),
        ]
        # Per-pipe relay register in front of the gather (timing). The scratchpad slot of
        # each result is computed here, from the pipe's FIFO output and the (op-static)
        # command fields, and travels registered with the result: the gather then only
        # compares and selects. Every mode's addressing is folded into one field:
        #   ilv      slot = dst + qoff + row*N + chan       (position-major, merged beats)
        #   xred     slot = dst + qoff + row                (one summed vector)
        #   axpy     slot = dst + qoff + chan*cols + row    (wide drain rows)
        #   default  slot = dst + qoff + chan*rows + row    (channel-major)
        # Consumed == counted holds: the counting stage sits after the relay.
        # Plain-matvec results are packed into full beats per channel in the relay stage
        # (_ResultPacker above). pack_en excludes every mode with its own retirement path.
        pack_en = Signal()
        self.comb += pack_en.eq(~fetch_mode & ~store_mode & ~axpy_mode & ~ilv_mode)
        gsrc  = []
        packs = []
        for i, pipe in enumerate(self.pipes):
            # Registered input stage: the slot arithmetic below is computed from the pipe
            # FIFO's asynchronous read and lands in flops here, so the packer's merge
            # compare starts from registers.
            ib = stream.Buffer([("row", 16), ("data", 256), ("wide", 1), ("head", 1),
                                ("slot", 24)])
            setattr(self, f"gin{i}", ib)
            pk = _ResultPacker()
            setattr(self, f"pack{i}", pk)
            rows_last_i = Signal(16, reset_less=True)
            self.sync += rows_last_i.eq(cmd_r["rows"] - 1 + (i < cmd_r["rem"]))
            self.comb += [
                ib.sink.valid.eq(pipe.source.valid),
                pipe.source.ready.eq(ib.sink.ready),
                ib.sink.row.eq(pipe.source.row),
                ib.sink.data.eq(pipe.source.data),
                ib.sink.wide.eq(pipe.source.wide),
                ib.sink.head.eq(pipe.source.head),
                ib.sink.slot.eq(dstw + Mux(pipe.source.head, cmd_r["qstride"], 0)
                    + Mux(ilv_mode, pipe.source.row*n + i,
                      Mux(xred_mode, pipe.source.row,
                          pipe.source.row + Mux(axpy_mode, i*cmd_r["cols"],
                                                                i*cmd_r["rows"])))),
                ib.source.connect(pk.sink),
                pk.enable.eq(pack_en),
                pk.rows_last.eq(rows_last_i),
                pk.qn.eq(cmd_r["qn"]),
            ]
            gsrc.append(pk.source)
            packs.append(pk)
        grant_valid = Array([g.valid for g in gsrc])[rr.grant]
        grant_row   = Array([g.row   for g in gsrc])[rr.grant]
        grant_data  = Array([g.data  for g in gsrc])[rr.grant]
        grant_wide  = Array([g.wide  for g in gsrc])[rr.grant]
        grant_lanes = Array([g.lanes for g in gsrc])[rr.grant]
        grant_slot  = Signal(24)
        merge_mask  = Signal(n)
        self.comb += grant_slot.eq(Array([g.slot for g in gsrc])[rr.grant])
        self.comb += merge_mask.eq(Cat(*[
            gsrc[i].valid & ~gsrc[i].wide & (gsrc[i].slot[3:] == grant_slot[3:])
            for i in range(n)]))

        # Cross-channel reduce (AXPY + cmd.xred). Every pipe drains its partial vector as
        # cols/8 beats in the same order - beat k is always lane group k, pushed by the drain
        # FSM's d_cnt - so the channels are in lockstep by construction and no addressing is
        # needed to line them up: wait until all n have a beat, add them lane-wise, write one
        # beat, and consume all n together.
        #
        # Head-of-line blocking is not a hazard here: every channel produces exactly cols/8
        # beats regardless of `rem` (rem varies the number of positions a channel accumulates,
        # not the length of its output), and the drain runs after the reader has finished, so
        # a stalled axw_fifo cannot deadlock against an in-flight read.
        xred_all = Signal()
        self.comb += xred_all.eq(xred_mode & reduce(and_, [g.valid for g in gsrc]))
        xred_dat = Signal(256)
        def _addtree(sigs):
            # Balanced tree, log2(n) deep (wrap-add: bit-identical to a left fold). Exactly
            # one level is registered whatever the depth (timing) - the second level when
            # there is one, else the only level, else the lone input - so the summed beat
            # always lands one cycle later and the xred write below follows it.
            depth = (len(sigs) - 1).bit_length()
            if depth == 0:
                r = Signal(32, reset_less=True)
                self.sync += r.eq(sigs[0])
                return r
            reg_level = min(1, depth - 1)
            level = 0
            while len(sigs) > 1:
                reg = (level == reg_level)
                nxt = []
                for a, b in zip(sigs[::2], sigs[1::2]):
                    m = Signal(32, reset_less=True)
                    if reg:
                        self.sync += m.eq(a + b)
                    else:
                        self.comb += m.eq(a + b)
                    nxt.append(m)
                if len(sigs) % 2:
                    if reg:
                        r = Signal(32, reset_less=True)
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
        xred_slot_d = Signal(len(scratchpad.gather.adr), reset_less=True)

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
        self.comb += [
            is_matvec.eq((cmd_r["kind"] == 0) | (cmd_r["kind"] == 1)),
            # Logical row index of the packed beat's LANE 0: the packer carries the first
            # result's row and lane (slot[:3]), and lanes are consecutive rows of one
            # channel, so lane l is row0 + (l - lane0). Wraps mod 2^24 for lanes below
            # lane0, which are never valid.
            gidx.eq(rr.grant*cmd_r["rows"] + grant_row - grant_slot[0:3]),
        ]
        # Argmax tap rides the plain-matvec grant path only: it is used exclusively by the
        # chan-major LM-head op (never ilv, never AXPY). A grant is a packed beat of up
        # to 8 results (_ResultPacker), so the tap reduces 8 lanes to one (value, index)
        # with np.argmax's tie-break (smallest index) through a registered tree - three
        # levels, one per cycle - and then the running compare. Invalid lanes enter as
        # (INT32_MIN, 0xFFFFFF), which no valid lane loses to. `done` waits for this
        # pipeline to empty (am_busy), so no reader can see a stale argmax.
        am_v_r = Signal()
        am_l_r = Signal(8,   reset_less=True)
        am_d_r = Signal(256, reset_less=True)
        am_b_r = Signal(24,  reset_less=True)
        self.sync += [
            am_v_r.eq(is_matvec & ~ilv_mode & grant_valid & ~grant_wide),
            am_l_r.eq(grant_lanes),
            am_d_r.eq(grant_data),
            am_b_r.eq(gidx),
        ]
        NEG = -(1 << 31)
        def _pick(va, ia, vb, ib):
            v = Signal((32, True)); i = Signal(24)
            self.comb += If((va > vb) | ((va == vb) & (ia <= ib)),
                             v.eq(va), i.eq(ia)).Else(v.eq(vb), i.eq(ib))
            return v, i
        lv = []
        for l in range(8):
            v = Signal((32, True), name=f"am_lv{l}"); i = Signal(24, name=f"am_li{l}")
            self.comb += [
                v.eq(Mux(am_l_r[l], am_d_r[32*l:32*l + 32], NEG)),
                i.eq(Mux(am_l_r[l], am_b_r + l, 0xFFFFFF)),
            ]
            lv.append((v, i))
        # Level 1: 8 -> 4 (registered).
        am_v2 = Signal(); l2 = []
        for k in range(4):
            v, i = _pick(*lv[2*k], *lv[2*k + 1])
            vr = Signal((32, True), name=f"am_v2_{k}", reset_less=True)
            ir = Signal(24, name=f"am_i2_{k}", reset_less=True)
            self.sync += [vr.eq(v), ir.eq(i)]
            l2.append((vr, ir))
        self.sync += am_v2.eq(am_v_r)
        # Level 2: 4 -> 2 (registered).
        am_v3 = Signal(); l3 = []
        for k in range(2):
            v, i = _pick(*l2[2*k], *l2[2*k + 1])
            vr = Signal((32, True), name=f"am_v3_{k}", reset_less=True)
            ir = Signal(24, name=f"am_i3_{k}", reset_less=True)
            self.sync += [vr.eq(v), ir.eq(i)]
            l3.append((vr, ir))
        self.sync += am_v3.eq(am_v2)
        # Level 3: 2 -> 1 (registered).
        am_v4 = Signal()
        v, i = _pick(*l3[0], *l3[1])
        am_s_r = Signal((32, True), reset_less=True); am_i_r = Signal(24, reset_less=True)
        self.sync += [am_s_r.eq(v), am_i_r.eq(i), am_v4.eq(am_v3)]
        am_busy = Signal()
        self.comb += am_busy.eq(am_v_r | am_v2 | am_v3 | am_v4)
        self.sync += [
            If(tracker_reset,
                self.argmax_val.eq(-(1 << 31)),
                self.argmax_idx.eq(0xFFFFFF),
            ).Elif(am_v4 &
                   ((am_s_r > self.argmax_val)
                    | ((am_s_r == self.argmax_val) & (am_i_r < self.argmax_idx))),
                self.argmax_val.eq(am_s_r),
                self.argmax_idx.eq(am_i_r),
            ),
        ]

        # Fetch datapath: selected channel's beats -> scratchpad (256-bit writes).
        fetch_valid = Signal()
        fetch_data  = Signal(256)
        fetch_cnt   = Signal(24)
        fetch_wcnt  = Signal(24, reset_less=True) # Write address offset, cleared at accept.
        fetch_take  = Signal() # A fetch beat is actually consumed this cycle.
        self.comb += [
            fetch_valid.eq(reduce_or(fetch_valids)),
            fetch_data.eq(Array(fetch_datas)[fetch_chan]),
            fetch_take.eq(fetch_valid & fetch_rdy),
        ]

        # Scratchpad write port: fetch beats, one merged ilv beat, one wide AXPY beat, or
        # a single gathered int32. The write stage is registered (timing): the RAM write,
        # result counting and the trackers run one cycle later off these registers (ops are
        # separated by dispatch - the lag is invisible).
        slot = Signal(24)
        # Fetch write addressing counts at the comb stage (beats arrive every cycle;
        # the registered fetch_cnt lags one cycle and is only the terminator).
        eng_gw_adr_c = Signal(len(scratchpad.gather.adr))
        eng_gw_we_c  = Signal(8)
        eng_gw_dat_c = Signal(256)
        eng_gw_adr   = Signal(len(scratchpad.gather.adr), reset_less=True)
        eng_gw_we    = Signal(8)
        eng_gw_dat   = Signal(256, reset_less=True)
        self.sync += [
            eng_gw_adr.eq(eng_gw_adr_c),
            eng_gw_we.eq(eng_gw_we_c),
            eng_gw_dat.eq(eng_gw_dat_c),
        ]
        busy_d = Signal()
        self.sync += busy_d.eq(self.busy)
        self.comb += self.sp_busy.eq(self.busy | busy_d)
        self.sync += If(command_accept,
            fetch_wcnt.eq(0),
        ).Elif(fetch_mode & fetch_take,
            # Equivalent to (eng_gw_we_c != 0) in fetch mode, but off the reader valids
            # directly: keeps the pipes' FIFO state out of this CE (timing).
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
        dn_wr_take = Signal()
        if dnstream:
            self.comb += [
                dn_wr_take.eq(dn_mode & dn.sp_wr_req & fetch_rdy),
                dn.sp_wr_gnt.eq(dn_wr_take),
            ]
        at_wr_take = Signal()
        if attnstream:
            self.comb += [
                at_wr_take.eq(at_mode & at.sp_wr_req & fetch_rdy),
                at.sp_wr_gnt.eq(at_wr_take),
            ]
        self.comb += [
            If(at_mode,
                eng_gw_adr_c.eq(at.sp_wr_adr if attnstream else 0),
                eng_gw_dat_c.eq(at.sp_wr_dat if attnstream else 0),
                eng_gw_we_c.eq(Mux(at_wr_take, 0xFF, 0)),
            ).Elif(dn_mode,
                # The unit's output beats: granted when the gather FIFO has headroom, the
                # fetch rule (the payload is its FIFO head, held until then).
                eng_gw_adr_c.eq(dn.sp_wr_adr if dnstream else 0),
                eng_gw_dat_c.eq(dn.sp_wr_dat if dnstream else 0),
                eng_gw_we_c.eq(Mux(dn_wr_take, 0xFF, 0)),
            ).Elif(fetch_mode,
                eng_gw_adr_c.eq((cmd_r["dst"] >> 5) + fetch_wcnt),
                eng_gw_dat_c.eq(fetch_data),
                eng_gw_we_c.eq(Mux(fetch_take, 0xFF, 0)),
            ).Elif(ilv_mode,
                eng_gw_adr_c.eq(grant_slot[3:]),
                eng_gw_dat_c.eq(lane_dat),
                # Not gated on grant_valid: a merged beat is consumed on merge_mask, which
                # only sets lanes for pipes that are themselves valid and whose slot is in
                # this beat, so lane_we is already zero when nothing merges. Gating it would
                # break consumed == counted (the round-robin can still point at a pipe that
                # just went invalid while merge_mask takes another pipe's result) and hang
                # RUN. The address is safe: merge_mask requires gsrc[i].slot[3:] ==
                # grant_slot[3:], so every lane written belongs here.
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
                # Plain matvec: a packed beat from the channel's _ResultPacker - up to 8
                # results, lanes as written.
                eng_gw_adr_c.eq(slot[3:]),
                eng_gw_dat_c.eq(grant_data),
                eng_gw_we_c.eq(Mux(grant_valid, grant_lanes, 0)),
            ),
        ]

        # Result absmax tracker, write-stage (norm_i32's shift argument; reset on command
        # accept). Runs over every written lane so merged/wide retirement is covered.
        # Per-lane |value| is registered before the max tree and the tree's result before the
        # compare (timing: the tree and the 32-bit compare were one 24-level cone). The tracker
        # lags the write by three cycles; done waits for it (absmax_busy), so every reader (CSR,
        # sequencer k(absmax) at DISPATCH) sees the op's final value.
        lane_abs = []
        for l in range(8):
            ls = Signal((32, True), name=f"labs_s{l}")
            la = Signal(32, name=f"labs{l}")
            lr = Signal(32, name=f"labsr{l}", reset_less=True) # Rewritten every cycle.
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
        wr_absmax_c = _maxtree(lane_abs)
        wr_absmax   = Signal(32, reset_less=True)
        am_w1, am_w2 = Signal(), Signal() # A write's |value| in the lane / tree registers.
        absmax_busy  = Signal()
        self.sync += [
            # Zeroed with the tracker: a value in flight belongs to the op before the reset.
            If(tracker_reset, wr_absmax.eq(0)).Else(wr_absmax.eq(wr_absmax_c)),
            am_w1.eq(eng_gw_we != 0),
            am_w2.eq(am_w1),
            If(tracker_reset,
                self.absmax.eq(0),
            ).Elif(~fetch_mode & ~dn_mode & ~at_mode & (wr_absmax > self.absmax),
                self.absmax.eq(wr_absmax),
            ),
        ]
        self.comb += absmax_busy.eq(am_w1 | am_w2)

        # The engine's R-port address: the fill/STORE counter, or a stream unit's registered
        # request while its op runs.
        eng_rd_adr = Signal(len(scratchpad.fill.adr))
        self.comb += eng_rd_adr.eq(eng_fill_adr)
        if dnstream:
            self.comb += If(dn_mode, eng_rd_adr.eq(dn.sp_rd_adr))
        if attnstream:
            self.comb += If(at_mode, eng_rd_adr.eq(at.sp_rd_adr))

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
                scratchpad.fill.adr.eq(eng_rd_adr),
                scratchpad.gather.adr.eq(eng_gw_adr),
                scratchpad.gather.dat_w.eq(eng_gw_dat),
                scratchpad.gather.we.eq(sp_gw_we),
            ]
        else:
            # Per-cycle arbitration: the units may run concurrently, so the grant is per
            # cycle, not per op.
            #
            # R port: vec > seq > engine. A vec load claims the port for exactly its address
            # cycle (sp_rd_req); the engine's fill/store address generation simply does not
            # advance on those cycles (rd_free below).
            #
            # W port: engine writes are enqueued into a small FIFO instead of driving the
            # port directly - the merge/counting stage upstream must not stall (see the
            # consumed-vs-counted note above). Vec stores get the port while the FIFO has
            # headroom; at/above the mark the engine drains continuously and the vec unit
            # holds (sp_wr_gnt). Deadlock-free: the engine never waits on the vec unit
            # mid-op, so the FIFO always drains. Depth 8 suffices: input is <= 1 write/cycle,
            # and above the mark drain matches input.
            grant_seq     = Signal()
            vec_rd        = Signal()
            vec_wr        = Signal()
            sb_headroom_r = Signal() # Registered W-FIFO headroom for the sideband grant.
            # Engine-local copy of seq.sbmode (see sequencer.py): a rarely-changing CSR
            # bit re-registered here so the fetch-grant cone stays engine-local.
            sbmode_r = Signal()
            if seq is not None:
                self.sync += sbmode_r.eq(seq.sbmode.storage)
            gw_w    = len(scratchpad.gather.adr) + 8 + 256
            wr_fifo = SyncFIFO(width=gw_w, depth=8)
            self += wr_fifo
            f_adr = Signal(len(scratchpad.gather.adr))
            f_we  = Signal(8)
            f_dat = Signal(256)
            self.comb += Cat(f_adr, f_we, f_dat).eq(wr_fifo.dout)
            self.gw_ovf = Signal() # Sticky: an engine write hit a full FIFO (must never).
            self.sync += If((sp_gw_we != 0) & ~wr_fifo.writable, self.gw_ovf.eq(1))
            self.comb += [
                grant_seq.eq(0 if seq is None else seq.sp_active),
                vec_rd.eq(vecunit.sp_rd_req),
                # The seq's fetch FSM holds its address cycle when a vec load outranks it.
                # Posted sideband writes: either the fetch waits on them (sbmode=0) or the
                # sequencer drains the FIFO once at START (sbmode=1, sequencer.py SBDRAIN).
                # sbmode is used through a local register (sbmode_r) to keep a CPU-side CSR
                # bit out of this port cone (congestion).
                *([] if seq is None else [
                    seq.sp_gnt.eq(~vec_rd & (~scratchpad.sbw_pending | sbmode_r)),
                    seq.sbw_pending.eq(scratchpad.sbw_pending)]),
                # Posted sideband writes (see scratchpad.py): the drain takes the W
                # port on any cycle the vector store slot is idle and the gather FIFO
                # has registered headroom - an engine-local grant. The headroom margin
                # keeps gw_ovf impossible (level(T-1) < 5 bounds level(T) <= 6 < 8 on a
                # non-draining cycle).
                scratchpad.sbw_gnt.eq(scratchpad.sbw_pending & ~vec_wr & sb_headroom_r),
                # Read slot: sideband reads take the R port on any free cycle, below vec
                # loads and the seq fetch; the fill/store address just holds (rd_free).
                scratchpad.sbr_allow.eq(~vec_rd & ~grant_seq),
                rd_free.eq(~vec_rd & ~grant_seq & ~scratchpad.sbr_gnt_any),
                vec_wr.eq(vecunit.sp_wr_req & (wr_fifo.level < 6)),
                vecunit.sp_wr_gnt.eq(vec_wr),
                # A ScratchpadArbiter takes the room instead (per-client grants, scratchpad.py).
                *([vecunit.sp_wr_room.eq(wr_fifo.level < 6)] if hasattr(vecunit, "sp_wr_room") else []),
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
                    scratchpad.fill.adr.eq(eng_rd_adr),
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

        # Control FSM ------------------------------------------------------------------------------
        self.fsm = fsm = FSM(reset_state="IDLE")
        self.comb += st_run.eq(fsm.ongoing("ST_RUN"))
        start_pulse = Signal()
        self.comb += st_fifo.reset.eq(start_pulse | st_next)
        for i, pipe in enumerate(self.pipes):
            self.sync += pipe.start.eq(start_pulse & ~dn_mode & ~at_mode)
        for unp in self.unpackers:
            self.sync += unp.start.eq(start_pulse & ~dn_mode & ~at_mode)
        dn_fin = Signal() # The unit reported done for this op (sticky until the next).
        if dnstream:
            self.comb += [
                dn.start.eq(start_pulse & dn_mode),
                dn.heads.eq(cmd_r["rows"]),
                dn.table.eq(cmd_r["xsrc"] >> 5),
                dn.zero.eq(cmd_r["bias"][0]),      # Op 5: the incoming state is +0.0.
                dn.sp_rd_ok.eq(dn_mode & rd_free),
                dn.sp_rd_dat.eq(scratchpad.fill.dat_r),
            ]
            self.sync += If(start_pulse, dn_fin.eq(0)).Elif(dn.done, dn_fin.eq(1))
        at_fin = Signal()
        if attnstream:
            # The merge: position t is channel t % n's next row, sbeats + cbeats beats each.
            at_rowbeats = Signal(9)
            at_beat     = Signal(9)
            self.comb += [
                at_rowbeats.eq(cmd_r["sbeats"] + cmd_r["cbeats"]),
                at.start.eq(start_pulse & at_mode),
                at.rec.eq(cmd_r["xsrc"] >> 5),
                at.sp_rd_ok.eq(at_mode & rd_free),
                at.sp_rd_dat.eq(scratchpad.fill.dat_r),
                at.sink.valid.eq(at_mode & Array(rsrc_valids)[at_chan]),
                at.sink.data.eq(Array(rsrc_datas)[at_chan]),
                at_take.eq(at.sink.valid & at.sink.ready),
            ]
            self.sync += [
                If(start_pulse, at_fin.eq(0)).Elif(at.done, at_fin.eq(1)),
                If(start_pulse,
                    at_chan.eq(0), at_beat.eq(0),
                ).Elif(at_take,
                    If(at_beat == at_rowbeats - 1,
                        at_beat.eq(0),
                        at_chan.eq(Mux(at_chan == n - 1, 0, at_chan + 1)),
                    ).Else(
                        at_beat.eq(at_beat + 1),
                    )
                ),
            ]
        for pk in packs:
            self.comb += pk.clear.eq(start_pulse) # No partial beat survives an op boundary.
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
            ).Elif(dn_mode,
                NextState("DN_CMD" if dnstream else "IDLE"),
            ).Elif(at_mode,
                NextState("ISSUE" if attnstream else "IDLE"),
            ).Elif(pf_mode,
                NextState("PF_ISSUE" if prefetch else "IDLE"),
            ).Else(
                NextState("FILL"),
            )
        )
        # STORE targets the one channel in cmd.chan, so it needs the same per-channel
        # offset the readers get: with chan_mod a mux of the windows' constant offsets
        # (stride + skew), without it a shift of the channel index (power-of-two stride).
        if chan_mod and (chan_stride or chan_skew):
            _store_addr = st_addr_r + _chan_off_of(_chan_sel(cmd_r["chan"]))
        else:
            _store_addr = st_addr_r + (cmd_r["chan"] << chan_shift if chan_stride else 0)
        fsm.act("ST_CMD",
            self.busy.eq(1),
            self.store_cmd.valid.eq(1),
            self.store_cmd.chan.eq(cmd_r["chan"]),
            self.store_cmd.addr.eq(_store_addr),
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
        fsm.act("ST_DRAIN",
            # Wait for the loader's B-channel drain: data is then readable.
            self.busy.eq(1),
            If(~self.store_busy,
                self.done.eq(1),
                NextState("IDLE"),
            )
        )
        if dnstream:
            # The writers take the command first (they only accept data once commanded),
            # then every reader gets the same span; RUN waits for the unit and the writers.
            fsm.act("DN_CMD",
                self.busy.eq(1),
                self.mstore_cmd.valid.eq(1),
                self.mstore_cmd.addr.eq(cmd_r["wbase"] + Cat(Replicate(0, 5), cmd_r["dstride"],
                    Replicate(cmd_r["dstride"][23], 3))),
                self.mstore_cmd.len.eq(cmd_r["wlen"]),
                If(self.mstore_cmd.ready, NextState("ISSUE")),
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
        fsm.act("FILL_LAST",
            # Drain the registered fill write.
            self.busy.eq(1),
            NextState("ISSUE"),
        )
        read_cmd_fields = [
            *[c.addr.eq(cmd_r["wbase"] + _chan_off(i))
              for i, c in enumerate(read_cmds)],
            *[c.len.eq(cmd_r["wlen"]
                + Mux(i < cmd_r["rem"], cmd_r["sbeats"] + cmd_r["cbeats"], 0))
              for i, c in enumerate(read_cmds)],
        ]
        # Prefetch bookkeeping: a kind-7 op leaves its reads pending for the next matvec, whose
        # ISSUE then only checks the address (pf_mismatch, sticky: a firmware ordering bug) and
        # goes straight to RUN.
        pf_pending  = Signal()
        pf_wbase    = Signal(32, reset_less=True)
        self.pf_hits     = Signal(32) # Matvecs served from a prefetch.
        self.pf_mismatch = Signal()   # Sticky: a matvec (or another op) followed a prefetch
                                      # of a different span.
        pf_skip = Signal()
        if prefetch:
            self.comb += pf_skip.eq(pf_pending)
        fsm.act("ISSUE",
            self.busy.eq(1),
            If(pf_skip,
                NextState("RUN"),
            ).Else(
                *[c.valid.eq(~fetch_mode | (fetch_chan == i))
                  for i, c in enumerate(read_cmds)],
                *read_cmd_fields,
                # Waits on readers[0] alone. A reader only accepts while idle, so a reader
                # still draining could in principle miss its command; not observed on
                # hardware, and a per-reader wait costs timing.
                If(read_cmds[0].ready,
                    NextState("RUN"),
                )
            )
        )
        if prefetch:
            fsm.act("PF_ISSUE",
                self.busy.eq(1),
                *[c.valid.eq(1) for c in read_cmds],
                *read_cmd_fields,
                If(read_cmds[0].ready,
                    self.done.eq(1),
                    NextState("IDLE"),
                )
            )
            self.sync += [
                If(fsm.ongoing("PF_ISSUE") & read_cmds[0].ready,
                    pf_pending.eq(cmd_r["wlen"] != 0), # wlen 0: a no-op (a list's unused slot).
                    pf_wbase.eq(cmd_r["wbase"]),
                ).Elif(fsm.ongoing("ISSUE") & pf_pending,
                    pf_pending.eq(0),
                    self.pf_hits.eq(self.pf_hits + 1),
                    If((cmd_r["wbase"] != pf_wbase) | fetch_mode | at_mode | dn_mode,
                        self.pf_mismatch.eq(1)),
                ),
            ]
        mv_total = Signal(24)
        ax_total = Signal(24)
        # Test-visible aliases (no hardware: same signals under a name a testbench can find).
        # A stall in RUN is either "results never produced" or "produced but never retired",
        # and result_cnt against the expected total is what separates the two.
        self.dbg_result_cnt = result_cnt
        self.dbg_gw_we      = sp_gw_we # W-port write cycles (a testbench counts them).
        self.dbg_mv_total   = mv_total
        self.dbg_ax_total   = ax_total
        self.comb += [
            mv_total.eq(cmd_r["rows"]*self.n_channels + cmd_r["rem"]),
            # xred writes one cols-long vector, so the retire count drops by n_channels.
            # result_cnt counts written lanes (popcount of the write enables), so this is
            # all the accounting the merged beats need.
            ax_total.eq(Mux(cmd_r["xred"], cmd_r["cols"],
                            cmd_r["cols"]*self.n_channels)),
        ]
        self.comb += run_state.eq(fsm.ongoing("RUN"))
        fsm.act("RUN",
            self.busy.eq(1),
            If(Mux(at_mode, at_fin, Mux(dn_mode, dn_fin & ~self.store_busy, Mux(fetch_mode,
                   fetch_cnt == cmd_r["wlen"],
                   # Dual-Q ops retire two heads' results per stream.
                   result_cnt == Mux(cmd_r["qn"],
                       Mux(axpy_mode, ax_total << 1, mv_total << 1),
                       Mux(axpy_mode, ax_total, mv_total)))))
               & ~gw_pending  # Queued writes must be in RAM before done.
               & ~am_busy & ~absmax_busy,
                self.done.eq(1),
                NextState("IDLE"),
            )
        )

        # Fill write-stage delay, STORE counters and result counting.
        self.sync += [
            fill_active_d.eq(fsm.ongoing("FILL") & rd_free),
            fill_beat_d.eq(fill_beat),
            # STORE prefetch: a read issued this cycle (the R port was ours) has its data on
            # fill.dat_r next cycle, when it enters the FIFO.
            st_pend.eq(st_issue & ~(start_pulse | st_next)),
            # Fill/STORE read address counter (see eng_fill_adr above).
            If(start_pulse,
                eng_fill_adr.eq(xsrc_w),
            ).Elif(st_next,
                eng_fill_adr.eq(xsrc_w + st_src_off + cmd_r["sstride"]),
            ).Elif(fsm.ongoing("FILL") & rd_free,
                eng_fill_adr.eq(eng_fill_adr + 1),
            ).Elif(st_issue,
                eng_fill_adr.eq(eng_fill_adr + 1),
            ),
            If(start_pulse | st_next, st_iss.eq(0)).Elif(st_issue, st_iss.eq(st_iss + 1)),
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

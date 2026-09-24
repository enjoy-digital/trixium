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

class _ResultPacker(Module):
    """The per-pipe relay register in front of the gather, which in plain-matvec mode also
    coalesces a channel's consecutive results into full 8-lane beats.

    Why (2026-09-07, docs/performance.md "The vector unit runs 2.9x slower under an engine
    stream"): a plain matvec wrote ONE int32 result per W-port beat, so the up projection's
    6,144 results were 6,144 port cycles in a ~6.7k-cycle stream. The engine's result FIFO
    drains only on cycles the vector unit is not storing, so it pinned at its threshold and
    a vector store was granted about one cycle in twelve - norm_i32 ran at 15.9 cy/beat under
    the stream against 5.6 idle. Plain-mode rows are channel-major and contiguous per channel
    (slot = dst + chan*rows + row), so a channel's 8 consecutive results are one aligned beat.

    Structure, chosen for timing (the first version, v10, failed the gate at 31k endpoints):
    - This IS the relay: its registers feed the gather's 16:1 mux and the round-robin, so
      the grant side sees exactly what the old stream.Buffer relay gave it.
    - Its input is registered too (the engine puts a stream.Buffer in front), so every cone
      here starts at flops: the pipe FIFO's asynchronous read and the slot arithmetic - the
      chain the n16 #3c fix moved off the critical path - never reach the merge compare.
    - No data mux: the pipe replicates its 32-bit result across all 8 lanes, so lane l's
      register always loads bits [32l:32l+32] of the input and only its ENABLE depends on
      the lane. Bypass modes (ilv/AXPY/xred/fetch) load all 8 lanes and behave as the
      plain register they replaced.
    - Flush is LOCAL: the pipe's last row is static per op (rows_last), so the packer
      flushes its own partial beat when it has consumed that row - no cross-die count.

    In pack mode a result whose beat matches the held one merges into its lane; otherwise
    the held beat is emitted (displace) and the new result loads. A full beat is presented
    as soon as it is full. Byte-exact: the memory sees the same lanes with the same data,
    only in fewer, wider writes."""
    def __init__(self):
        layout = [("row", 16), ("data", 256), ("wide", 1), ("head", 1), ("slot", 24)]
        self.sink   = stream.Endpoint(layout)
        self.source = stream.Endpoint(layout + [("lanes", 8)])
        self.enable    = Signal()   # Plain-matvec mode: pack. Otherwise a plain relay.
        self.clear     = Signal()   # Op start.
        self.rows_last = Signal(16) # This pipe's last row index (static per op).
        self.qn        = Signal()   # Dual-Q: the last result is (rows_last, head 1).

        p_v     = Signal()
        p_row   = Signal(16)
        p_dat   = Signal(256)
        p_wide  = Signal()
        p_head  = Signal()
        p_slot  = Signal(24)
        p_lanes = Signal(8)
        last_seen = Signal()        # The pipe's final result has been consumed.

        in_v    = Signal()
        lane    = Signal(3)
        merge   = Signal()          # Same beat as the held one: add a lane.
        displace = Signal()         # A new entry that cannot merge: emit the held one.
        full    = Signal()
        is_last = Signal()
        accept  = Signal()
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


class BonsaiMatVecEngine(LiteXModule):
    def __init__(self, scratchpad, hbm_axis, xbuf_groups=96, vecunit=None, seq=None,
                 dual=True, chan_stride=0, chan_mod=0, chan_skew=0,
                 native_regions=None, native_reservations=(), native_endpoints=None,
                 dnstream=False):
        with_native = native_regions is not None
        if with_native:
            from trixium.gateware.native_reader import NativeRegionReader, NativeReadTransport
            from trixium.gateware.native_weights import GROUP_LAYOUT, PTQ1_0, Q8_0
            native_regions, native_reservations = tuple(native_regions), tuple(native_reservations)
            native_endpoints = tuple(native_endpoints or [0]*len(hbm_axis))
            if len(native_endpoints) != len(hbm_axis):
                raise ValueError('one physical endpoint ID is required per compute pipe')
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
        # chan_mod: channels per memory controller, a power of two (0 = one flat address
        # space, the FK33's HBM). On a board with several controllers (the XCU1525's four
        # DDR4 channels) engine channel c sits on controller c // chan_mod and its window
        # is that controller's local HBM_WEIGHT_BASE + (c % chan_mod)*stride: the FK33's
        # constants and images, per controller. The mask is applied on the read, fetch and
        # STORE paths alike (the loader masks the upload path the same way).
        assert chan_mod == 0 or (chan_mod & (chan_mod - 1)) == 0, \
            "chan_mod must be 0 or a power of two (the STORE path masks by it)"
        # chan_skew: an extra per-channel offset of (c % chan_mod)*chan_skew bytes on top of
        # the stride, so that the channels sharing a DDR4 controller start in different
        # banks. With power-of-two strides the four windows of a controller sit in the same
        # bank (the stride is a multiple of the bank period) and four sequential streams
        # thrash that bank's row: 6-7.6 cycles/beat per port in the LiteDRAM sim against
        # 4.3 with a bank skew (2026-09-19; the XCU1525 uses 32 KB = a bank group step).
        # 0 keeps the FK33's netlist exactly.
        assert chan_skew == 0 or chan_mod, "chan_skew needs chan_mod"
        def _chan_off(i): # Python int: per-reader constant.
            local = (i % chan_mod) if chan_mod else i
            return local*chan_stride + local*chan_skew
        def _chan_sel(c): # Signal: the STORE path's per-op channel.
            return (c & (chan_mod - 1)) if chan_mod else c
        def _chan_skew_of(sel): # Signal: the STORE path's skew, a mux of constants.
            return Array([k*chan_skew for k in range(chan_mod)])[sel]

        # Keep the vector unit's busy signal, not the module: assigning a Module to an
        # attribute registers it as a submodule, and it is already one of the SoC's. The
        # double registration renames its CSRs (vecunit_* -> vecunit_vecunit_*, plus an
        # engine_vecunit_* alias) and moves them, which silently breaks firmware/vec.c.
        self.vec_busy_in = vecunit.busy if vecunit is not None else Signal()

        # Overlap counters. The MAC array is idle ~92% of a token and the sequencer runs one
        # descriptor at a time, so the question that decides whether dual-issue is worth
        # building is: how much of the token has BOTH units busy already? Free-running from
        # reset. Neither includes CPU arithmetic and memory stalls as well as dispatch;
        # it is NOT a prediction of the gain from dual issue. The vector busy input can
        # include the DeltaNet client through the shared scratchpad mux.
        #
        # Counted here rather than in the CSR front end because the engine already holds a
        # vec_busy_in reference; adding one in csr_front would mean a new integration signal
        # in BOTH SoC definitions, which is exactly the drift the guard test exists to catch.
        self.cyc_total = Signal(64)
        self.cyc_eng   = Signal(64)
        self.cyc_vec   = Signal(64)
        self.cyc_both  = Signal(64)
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
                             # 4: axpy (weighted-V: per-lane accumulate, cols results/chan),
                             # 5: DeltaNet stream (dnstream only: rows = heads, xsrc = the
                             # head record table, wbase/wlen = every channel's state span,
                             # read and rewritten in place through the loader's writers).
            ("rows",   16),  # Rows per channel.
            ("cols",   17 if with_native else 14),  # K; legacy CSR remains 14 bits.
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
        ] + ([("native", 1)] if with_native else []))
        native_mode = C(0, 1)
        native_config_error = C(0, 1)
        native_complete = C(1, 1)
        raw_mode, raw_accept, raw_error = C(0, 1), C(0, 1), C(0, 8)
        raw_address, raw_complete = C(0, 64), C(1, 1)
        if with_native:
            from trixium.capabilities import Access
            from trixium.gateware.region_span import RegionSpanValidator
            from trixium.gateware.axi_read_monitor import AXIReadMonitor
            raw_mode, raw_accept, raw_address = Signal(), Signal(), Signal(64)
            self.region_read_cmd = raw = stream.Endpoint([
                ('region', 32), ('offset', 64), ('length', 24), ('dst', 24)])
            self.region_read_error = raw_error = Signal(8)
            self.region_read_done, self.region_read_busy, self.region_read_failed = Signal(), Signal(), Signal()
            self.region_read_beats = Signal(24)
            self.submodules.region_read_span = span = RegionSpanValidator(
                native_regions, native_reservations,
                zip(native_endpoints, [axi.address_width for axi in hbm_axis]), access=Access.READ)
            raw_config = Signal(4)
            raw_sp_end = Signal(30)
            self.comb += [
                span.region.eq(raw.region), span.offset.eq(raw.offset), span.length.eq(raw.length << 5),
                raw_sp_end.eq(raw.dst + (raw.length << 5)),
                raw_accept.eq(raw.valid & raw.ready),
            ]
            self.sync += If(raw_accept,
                raw_address.eq(span.address),
                raw_config.eq(Cat(span.error, (raw.dst[:5] != 0) | (raw_sp_end > scratchpad.size))),
            )
            self.region_read_monitors = []
            self.native_mode = native_mode = Signal()
            self.native_cfg = cfg = stream.Endpoint([
                ("chan", 5), ("region", 32), ("offset", 64), ("stride", 64)])
            self.native_cfg_error = Signal()
            self.native_cfg_valid = cfg_valid = Signal(n)
            native_regions_r = [Signal(32, name=f'native_region{i}') for i in range(n)]
            native_offsets_r = [Signal(64, name=f'native_offset{i}') for i in range(n)]
            native_strides_r = [Signal(64, name=f'native_stride{i}') for i in range(n)]
            self.comb += cfg.ready.eq(self.cmd.ready & ~self.cmd.valid)
            self.sync += If(cfg.valid & cfg.ready,
                self.native_cfg_error.eq(cfg.chan >= n),
                Case(cfg.chan, {i: [native_regions_r[i].eq(cfg.region),
                                   native_offsets_r[i].eq(cfg.offset), native_strides_r[i].eq(cfg.stride),
                                   cfg_valid[i].eq(1)] for i in range(n)}),
            )
            self.native_readers = []
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

        # DeltaNet stream (kind 5): channel i's reader feeds lane i of the unit and lane i's
        # updated state goes to channel i's writer through the loader's multi-channel store
        # (same wbase/wlen on every channel, the readers' per-channel offsets).
        self.with_dnstream = dnstream
        if dnstream:
            from trixium.gateware.deltanet_stream import BonsaiDeltaNetStream
            # dnstream=True: the 27B geometry on the exact operators; a dict overrides the
            # unit's arguments (tests: a small geometry on integer operators).
            opts = dict(dnstream) if isinstance(dnstream, (dict, tuple)) else {}
            self.submodules.dnstream = dn = BonsaiDeltaNetStream(
                lanes=len(hbm_axis), adr_bits=len(scratchpad.fill.adr), **opts)
            self.mstore_cmd  = stream.Endpoint([("addr", 33), ("len", 24)])
            self.mstore_data = dn.sources

        # # #

        # Channels ----------------------------------------------------------------------------
        fetch_mode = Signal() # kind == 2 for the latched command.
        fetch_chan = Signal(5)

        self.readers = []
        self.pipes   = []
        read_cmds = []
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
        dn_mode = Signal() # kind == 5 for the latched command.
        for i, axi in enumerate(hbm_axis):
            # n_ids=4: the HBM controller serializes same-ARID reads (one 16-beat burst
            # per round trip - 33.7 cycles/beat measured); round-robin ARIDs let it pipeline.
            reader = BonsaiAXIReader(axi, n_ids=4)
            pipe   = BonsaiChannelPipe(xbuf_groups=xbuf_groups, chan_id=i, n_channels=n,
                                       dual=dual, with_native=with_native)
            setattr(self.submodules, f"reader{i}", reader)
            setattr(self.submodules, f"pipe{i}",   pipe)
            # Relay register between the reader (its FIFO sits at the HBM port, bottom of
            # the die) and the pipe: n16 timing #3c had the FIFO output reaching codes_r
            # with zero logic levels and 4.5 ns of route. Latency only.
            rbuf = stream.Buffer([("data", 256)])
            setattr(self.submodules, f"rbuf{i}", rbuf)
            if with_native:
                monitor = AXIReadMonitor(axi)
                setattr(self.submodules, f'region_read_monitor{i}', monitor)
                self.region_read_monitors.append(monitor)
                self.comb += monitor.enable.eq(raw_mode & (fetch_chan == i))
                transport = NativeReadTransport(axi.address_width)
                nreader = NativeRegionReader(axi, native_regions, native_reservations,
                                             endpoint=native_endpoints[i], transport=transport)
                setattr(self.submodules, f'native_reader{i}', nreader)
                self.native_readers.append(nreader)
                legacy_cmd = stream.Endpoint([('addr', axi.address_width), ('len', 24)])
                read_cmds.append(legacy_cmd)
                # Exactly ONE physical burst reader and ONE MAC per compute pipe.
                # Native/legacy formatters share both, selected for the entire op.
                self.comb += [
                    reader.sink.valid.eq(Mux(native_mode, transport.sink.valid, legacy_cmd.valid)),
                    reader.sink.addr.eq(Mux(native_mode, transport.sink.addr, legacy_cmd.addr)),
                    reader.sink.len.eq(Mux(native_mode, transport.sink.len, legacy_cmd.len)),
                    transport.sink.ready.eq(reader.sink.ready & native_mode),
                    legacy_cmd.ready.eq(reader.sink.ready & ~native_mode),
                    reader.source.connect(transport.source, omit={'valid', 'ready'}),
                    transport.source.valid.eq(reader.source.valid & native_mode),
                    transport.idle.eq(reader.idle),
                    reader.source.connect(rbuf.sink, omit={'valid', 'ready'}),
                    rbuf.sink.valid.eq(reader.source.valid & ~native_mode),
                    reader.source.ready.eq(Mux(native_mode, transport.source.ready, rbuf.sink.ready)),
                ]
                nbuf = stream.Buffer(GROUP_LAYOUT)
                setattr(self.submodules, f'native_rbuf{i}', nbuf)
                self.comb += [nreader.source.connect(nbuf.sink), nbuf.source.connect(pipe.native_sink)]
            else:
                self.comb += reader.source.connect(rbuf.sink)
                read_cmds.append(reader.sink)
            rsrc = rbuf.source
            # Registered per channel (n16 timing #3d: the combinational decode from cmd_r
            # reached a far pipe's scale-RAM write enables with 5.4 ns of route). Stable
            # from the cycle after START; the reader is commanded later than that.
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
            self.comb += [
                # Normal: beats feed the compute pipe. Fetch: beats go to the scratchpad.
                pipe.sink.valid.eq(rsrc.valid & ~is_fetch & ~is_dn),
                pipe.sink.data.eq(rsrc.data),
                # Fetch beats honor the gather-FIFO headroom (a vec store can steal the
                # W port for a cycle); compute beats honor the pipe's result-FIFO
                # backpressure (drops results past N=2 otherwise - N/row_beats > 1 gather/cycle).
                rsrc.ready.eq((is_fetch & fetch_rdy) | (~is_fetch & ~is_dn & pipe.sink.ready)
                              | (is_dn & (dn.sinks[i].ready if dnstream else 0))),
            ]
            fetch_valids.append(rsrc.valid & is_fetch)
            fetch_datas.append(rsrc.data)
            self.readers.append(reader)
            self.pipes.append(pipe)

        # Latched command ---------------------------------------------------------------------
        cmd_r = {}
        for name in (("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
                     "bias", "rshift", "xsrc", "dst", "chan", "rem", "ilv", "supw",
                     "rep", "dstride", "sstride",
                     "qn", "qstride", "rshift2", "xred") + (("native",) if with_native else ())):
            cmd_r[name] = Signal.like(getattr(self.cmd, name))
        self.sync += If(self.cmd.valid & self.cmd.ready,
            *[cmd_r[k].eq(getattr(self.cmd, k)) for k in cmd_r])
        if with_native:
            self.sync += [
                If(self.cmd.valid & self.cmd.ready, raw_mode.eq(0)),
                If(raw_accept,
                    raw_mode.eq(1),
                    *[cmd_r[k].eq({'kind': 2, 'wlen': raw.length, 'dst': raw.dst,
                                  'chan': span.port}.get(k, 0)) for k in cmd_r]),
            ]
            raw_complete = Signal()
            self.comb += [
                raw_error.eq(Cat(raw_config, Array([m.error for m in self.region_read_monitors])[fetch_chan])),
                raw_complete.eq(Array([r.idle & m.drained for r, m in
                                      zip(self.readers, self.region_read_monitors)])[fetch_chan]),
                self.region_read_done.eq(self.done & raw_mode),
            ]
        command_accept = (self.cmd.valid & self.cmd.ready) | raw_accept
        store_mode = Signal()
        axpy_mode  = Signal()
        self.comb += [
            fetch_mode.eq(cmd_r["kind"] == 2),
            store_mode.eq(cmd_r["kind"] == 3),
            axpy_mode.eq(cmd_r["kind"] == 4),
            dn_mode.eq(cmd_r["kind"] == 5 if dnstream else 0),
            fetch_chan.eq(cmd_r["chan"]),
        ]
        if with_native:
            self.comb += native_mode.eq(cmd_r['native'])
            total_rows = Signal(24)
            input_bytes = Signal(19)
            input_end, output_end = Signal(27), Signal(28)
            groups_limit = Signal(9)
            native_config_error = Signal()
            self.comb += [
                total_rows.eq(cmd_r['rows']*n + cmd_r['rem']),
                input_bytes.eq(cmd_r['cols'] << cmd_r['qn']),
                input_end.eq(cmd_r['xsrc'] + input_bytes),
                output_end.eq(cmd_r['dst'] + ((total_rows + Mux(cmd_r['qn'], cmd_r['qstride'], 0)) << 2)),
                groups_limit.eq(Mux(cmd_r['kind'] == 0,
                    Mux(cmd_r['qn'], min(128, xbuf_groups//2), min(255, xbuf_groups)),
                    Mux(cmd_r['qn'], min(255, xbuf_groups*2), min(255, xbuf_groups*4)))),
                native_config_error.eq(self.native_cfg_error | (cmd_r['kind'] > 1)
                    | (total_rows == 0) | (cmd_r['rem'] >= n)
                    | ((cmd_r['rows'] == 65535) & (cmd_r['rem'] != 0))
                    | (cmd_r['cbeats'] == 0) | (cmd_r['cbeats'] > groups_limit)
                    | (cmd_r['cols'] != Mux(cmd_r['kind'] == 0, cmd_r['cbeats'] << 7, cmd_r['cbeats'] << 5))
                    | (cmd_r['qn'] & C(int(not dual))) | cmd_r['xred']
                    | ((cmd_r['xsrc'] & 31) != 0) | (input_end > scratchpad.size)
                    | ((cmd_r['dst'] & 3) != 0)
                    | (~cmd_r['supw'] & (output_end > scratchpad.size))
                    | (cmd_r['qn'] & (cmd_r['qstride'] < total_rows))
                    | (cmd_r['supw'] & (cmd_r['qn'] | cmd_r['ilv']))
                    | reduce_or([~cfg_valid[i] & ((cmd_r['rows'] != 0) | (i < cmd_r['rem']))
                                 for i in range(n)])),
            ]

        # Broadcast op params to pipes (channels < rem run one extra row). REGISTERED
        # per pipe (n16 timing #3c: the combinational broadcast reached a far pipe's
        # x-buffer address with 3.9 ns of route). They load during START from the new
        # cmd_r and are stable one cycle later, together with the registered start pulse
        # below - the first beat or fill write reaches a pipe later than that.
        for i, pipe in enumerate(self.pipes):
            # The copies are logically identical, so opt_design merged them back into ONE
            # flop (kept under pipe 0's name) with a fanout of 9,756 and ~3.4 ns of route
            # into every pipe's AXPY shift cone - the largest failing-endpoint family on
            # v11 (2026-09-10). dont_touch keeps each pipe's copy next to its consumers,
            # the v1_rep / fill_*2 idiom. (rows differs per pipe and never merged.)
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
            if with_native:
                pipe.native_mode.attr.add(('dont_touch', 'true'))
                self.sync += pipe.native_mode.eq(native_mode)
        # Overflow is included so any future rate mismatch fails loud (sticky error bit)
        # instead of dropping results and hanging the op.
        pipe_errors = [(p.error | p.overflow) & (~native_mode | (p.rows != 0)) for p in self.pipes]
        backend_error = (reduce_or([r.error != 0 for r in self.native_readers]) if with_native else C(0, 1))
        self.comb += self.error.eq(reduce_or(pipe_errors) | (native_mode & (native_config_error | backend_error))
                                  | (raw_mode & (raw_error != 0)))

        # x-fill ------------------------------------------------------------------------------
        fill_width = 11 if with_native else 10
        fill_beat  = Signal(fill_width)
        fill_total = Signal(fill_width)
        fill_half  = Signal(fill_width) # Beats per head (dual-Q reads 2x: A then B, contiguous).
        self.comb += [
            fill_half.eq(Mux(axpy_mode,
                (cmd_r["rows"]*self.n_channels + cmd_r["rem"] + 31) >> 5,
                cmd_r["cols"] >> 5)),
            fill_total.eq(Mux(cmd_r["qn"], fill_half << 1, fill_half)),
        ]

        # Registered fill pipeline: address at T, data+write at T+1.
        fill_active_d = Signal()
        fill_beat_d   = Signal(fill_width)
        st_cnt        = Signal(13) # STORE beat counter (address holds until accepted).
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
        # STORE stream through a 4-deep prefetch FIFO: reads are issued ahead (whenever the
        # R port is ours and credit remains), so the stream runs at one beat per clock. It
        # used to hold each address until its beat was accepted and then wait a cycle for the
        # next read: one beat every two clocks for every STORE (2026-09-24).
        self.submodules.st_fifo = st_fifo = ResetInserter()(stream.SyncFIFO([("data", 256)], 4))
        st_run   = Signal()    # fsm.ongoing("ST_RUN"), assigned once the FSM exists
        st_iss   = Signal(13)  # beats read in the current packet
        st_pend  = Signal()    # a read issued last cycle: its data is on fill.dat_r now
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
        # Plain-matvec results are packed into full beats per channel in the relay stage
        # (_ResultPacker above). pack_en excludes every mode with its own retirement path.
        pack_en = Signal()
        self.comb += pack_en.eq(~fetch_mode & ~store_mode & ~axpy_mode & ~ilv_mode)
        gsrc = []
        packs = []
        for i, pipe in enumerate(self.pipes):
            # Registered input stage: the slot arithmetic below is computed from the pipe
            # FIFO's asynchronous read and lands in flops here, so the packer's merge
            # compare starts from registers (n16 #3c, kept).
            ib = stream.Buffer([("row", 16), ("data", 256), ("wide", 1), ("head", 1),
                                ("slot", 24)])
            setattr(self.submodules, f"gin{i}", ib)
            pk = _ResultPacker()
            setattr(self.submodules, f"pack{i}", pk)
            rows_last_i = Signal(16)
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
                          pipe.source.row + Mux(axpy_mode, i*cmd_r["cols"], i*cmd_r["rows"]
                            + (Mux(native_mode, Mux(i < cmd_r['rem'], i, cmd_r['rem']), 0)
                               if with_native else 0))))),
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
        self.comb += [
            is_matvec.eq((cmd_r["kind"] == 0) | (cmd_r["kind"] == 1)),
            # Logical row index of the packed beat's LANE 0: the packer carries the first
            # result's row and lane (slot[:3]), and lanes are consecutive rows of one
            # channel, so lane l is row0 + (l - lane0). Wraps mod 2^24 for lanes below
            # lane0, which are never valid.
            gidx.eq(rr.grant*cmd_r["rows"] + grant_row - grant_slot[0:3]
                    + (Mux(native_mode, Mux(rr.grant < cmd_r['rem'], rr.grant, cmd_r['rem']), 0)
                       if with_native else 0)),
        ]
        # Argmax tap rides the plain-matvec grant path only: it is used exclusively by the
        # chan-major LM-head op (never ilv, never AXPY). A grant is now a PACKED beat of up
        # to 8 results (_ResultPacker), so the tap reduces 8 lanes to one (value, index)
        # with np.argmax's tie-break (smallest index) through a registered tree - three
        # levels, one per cycle - and then the running compare. Invalid lanes enter as
        # (INT32_MIN, 0xFFFFFF), which no valid lane loses to. `done` waits for this
        # pipeline to empty (am_busy), so no reader can see a stale argmax.
        am_v_r  = Signal()
        am_l_r  = Signal(8)
        am_d_r  = Signal(256)
        am_b_r  = Signal(24)
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
            vr = Signal((32, True), name=f"am_v2_{k}"); ir = Signal(24, name=f"am_i2_{k}")
            self.sync += [vr.eq(v), ir.eq(i)]
            l2.append((vr, ir))
        self.sync += am_v2.eq(am_v_r)
        # Level 2: 4 -> 2 (registered).
        am_v3 = Signal(); l3 = []
        for k in range(2):
            v, i = _pick(*l2[2*k], *l2[2*k + 1])
            vr = Signal((32, True), name=f"am_v3_{k}"); ir = Signal(24, name=f"am_i3_{k}")
            self.sync += [vr.eq(v), ir.eq(i)]
            l3.append((vr, ir))
        self.sync += am_v3.eq(am_v2)
        # Level 3: 2 -> 1 (registered).
        am_v4 = Signal()
        v, i = _pick(*l3[0], *l3[1])
        am_s_r = Signal((32, True)); am_i_r = Signal(24)
        self.sync += [am_s_r.eq(v), am_i_r.eq(i), am_v4.eq(am_v3)]
        am_busy = Signal()
        self.comb += am_busy.eq(am_v_r | am_v2 | am_v3 | am_v4)
        self.sync += [
            If(self.cmd.valid & self.cmd.ready,
                self.argmax_val.eq(-(1 << 31)),
                self.argmax_idx.eq(0xFFFFFF),
            ).Elif(am_v4 &
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
        fetch_wcnt  = Signal(24)
        fetch_take = Signal() # A fetch beat is actually consumed this cycle.
        self.comb += [
            fetch_valid.eq(reduce_or(fetch_valids)),
            fetch_data.eq(Array(fetch_datas)[fetch_chan]),
            fetch_take.eq(fetch_valid & fetch_rdy & (~raw_mode | ((fetch_wcnt < cmd_r['wlen']) & (raw_error == 0)))),
        ]

        # Scratchpad write port: fetch beats, one merged ilv beat, one wide AXPY beat, or
        # a single gathered int32. The write stage is REGISTERED: FIFO-out -> merge muxes
        # -> RAM write / absmax tree in one cycle was the worst hardware path (38 logic
        # levels); the RAM write, result counting and the trackers run one cycle later
        # off these registers (ops are separated by dispatch - the lag is invisible).
        slot = Signal(24)
        # Fetch write addressing counts at the comb stage (beats arrive every cycle;
        # the registered fetch_cnt lags one cycle and is only the terminator).
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
        self.sync += If(command_accept,
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
        dn_wr_take = Signal()
        if dnstream:
            self.comb += [
                dn_wr_take.eq(dn_mode & dn.sp_wr_req & fetch_rdy),
                dn.sp_wr_gnt.eq(dn_wr_take),
            ]
        self.comb += [
            If(dn_mode,
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
                # Plain matvec: a packed beat from the channel's _ResultPacker - up to 8
                # results, lanes as written. Was one lane per write (Replicate + 1 << lane),
                # which held the W port ~92% of a stream and starved the vector unit.
                eng_gw_adr_c.eq(slot[3:]),
                eng_gw_dat_c.eq(grant_data),
                eng_gw_we_c.eq(Mux(grant_valid, grant_lanes, 0)),
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
            If(command_accept,
                self.absmax.eq(0),
            ).Elif(~fetch_mode & ~dn_mode & (wr_absmax > self.absmax),
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
                scratchpad.fill.adr.eq(Mux(dn_mode, dn.sp_rd_adr, eng_fill_adr) if dnstream else eng_fill_adr),
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
            # Engine-local copy of seq.sbmode (see sequencer.py): a rarely-changing CSR
            # bit re-registered here so the fetch-grant cone stays engine-local.
            sbmode_r = Signal()
            if seq is not None:
                self.sync += sbmode_r.eq(seq.sbmode.storage)
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
                # The seq's fetch FSM holds its address cycle when a vec load outranks it.
                # It no longer also waits on posted sideband writes: that interlock was
                # correct but made a CPU patch burst starve a running list (~7.5k cycles
                # per layer for 80 writes, v8 qtrace). The ordering firmware actually
                # needs is "every patch for a list has landed before that list starts",
                # so the sequencer drains the FIFO once at START instead (sequencer.py
                # SBDRAIN). See the invariant recorded there.
                # sbmode picks which of the two orderings is live, so both can be measured
                # in one boot. It is used through a LOCAL register (sbmode_r below): a CSR
                # storage bit routed straight into this cone is a CPU-region signal in a
                # port cone, the family that cost two unroutable builds.
                *([] if seq is None else [
                    seq.sp_gnt.eq(~vec_rd & (~scratchpad.sbw_pending | sbmode_r)),
                    seq.sbw_pending.eq(scratchpad.sbw_pending)]),
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
                    scratchpad.fill.adr.eq(Mux(dn_mode, dn.sp_rd_adr, eng_fill_adr) if dnstream else eng_fill_adr),
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
        self.comb += st_run.eq(fsm.ongoing("ST_RUN"))
        start_pulse = Signal()
        self.comb += st_fifo.reset.eq(start_pulse | st_next)
        for i, pipe in enumerate(self.pipes):
            self.sync += pipe.start.eq(start_pulse & ~raw_mode & ~dn_mode
                                      & (~native_mode | (cmd_r['rows'] != 0) | (i < cmd_r['rem'])))
        dn_fin = Signal() # The unit reported done for this op (sticky until the next).
        if dnstream:
            self.comb += [
                dn.start.eq(start_pulse & dn_mode),
                dn.heads.eq(cmd_r["rows"]),
                dn.table.eq(cmd_r["xsrc"] >> 5),
                dn.sp_rd_ok.eq(dn_mode & rd_free),
                dn.sp_rd_dat.eq(scratchpad.fill.dat_r),
            ]
            self.sync += If(start_pulse, dn_fin.eq(0)).Elif(dn.done, dn_fin.eq(1))
        for pk in packs:
            self.comb += pk.clear.eq(start_pulse)   # No partial beat survives an op boundary.
        if with_native:
            issued, completed = Signal(n), Signal(n)
            self.native_issued = issued
            self.native_completed = completed
            active_mask = Signal(n)
            all_issued, native_complete = Signal(), Signal()
            self.native_failed = Signal()
            raw_owner = C(1, 1) if seq is None else ~seq.busy
            if seq is not None and hasattr(seq, 'region_fetch_cmd'):
                # The shared read-control mux selects the sequencer during
                # list ownership. Older queued manual reads may drain only in
                # SBDRAIN; no new manual command is admitted during the list.
                raw_owner = ~seq.busy | seq.native_manual_drain | seq.region_fetch_cmd.valid
            self.comb += [
                raw.ready.eq(fsm.ongoing('IDLE') & ~self.cmd.valid & ~cfg.valid
                             & ~self.sp_busy & ~self.vec_busy_in & ~scratchpad.sbw_pending
                             & reduce(and_, [r.idle for r in self.readers])
                             & raw_owner),
                self.region_read_busy.eq(raw_mode & ~fsm.ongoing('IDLE')),
                self.region_read_failed.eq(raw_mode & fsm.ongoing('N_FAILED')),
                self.region_read_beats.eq(fetch_cnt),
                active_mask.eq(Cat(*[(cmd_r['rows'] != 0) | (i < cmd_r['rem']) for i in range(n)])),
                all_issued.eq((issued & active_mask) == active_mask),
                native_complete.eq((completed & active_mask) == active_mask),
                self.native_failed.eq(fsm.ongoing('N_FAILED')),
            ]
            for i, nr in enumerate(self.native_readers):
                self.comb += [
                    nr.cmd.valid.eq(fsm.ongoing('N_ISSUE') & active_mask[i] & ~issued[i] & ~self.error),
                    nr.cmd.region.eq(native_regions_r[i]), nr.cmd.offset.eq(native_offsets_r[i]),
                    nr.cmd.stride.eq(native_strides_r[i]), nr.cmd.rows.eq(cmd_r['rows'] + (i < cmd_r['rem'])),
                    nr.cmd.groups.eq(cmd_r['cbeats']), nr.cmd.encoding.eq(Mux(cmd_r['kind'] == 0, PTQ1_0, Q8_0)),
                    nr.cancel.eq(fsm.ongoing('N_FAILED')),
                ]
                self.sync += If(start_pulse, issued[i].eq(0), completed[i].eq(0)).Else(
                    If(nr.cmd.valid & nr.cmd.ready, issued[i].eq(1)),
                    If(nr.done, completed[i].eq(1)),
                )
            fsm.act('N_ISSUE',
                self.busy.eq(1),
                If(self.error, NextState('N_FAILED')).Elif(all_issued, NextState('RUN')),
            )
            # Failed commands never report successful completion. The established
            # engine reset/drain shim must reset ALL pipes, relays and gather state
            # before reuse; reader cancellation alone cannot flush the whole engine.
            fsm.act('N_FAILED', self.busy.eq(1))

        fsm.act("IDLE",
            self.cmd.ready.eq(1),
            If(self.cmd.valid,
                NextValue(fill_beat, 0),
                NextValue(result_cnt, 0),
                NextState("START"),
            ).Elif(raw_accept,
                NextValue(fill_beat, 0),
                NextValue(result_cnt, 0),
                NextState('START'),
            )
        )
        fsm.act("START",
            start_pulse.eq(1),
            If((native_mode & native_config_error) | (raw_mode & (raw_error != 0)),
                NextState('N_FAILED') if with_native else NextState('IDLE'),
            ).Elif(fetch_mode,
                NextState("ISSUE"), # No x-fill for fetches.
            ).Elif(store_mode,
                NextState("ST_CMD"),
            ).Elif(dn_mode,
                NextState("DN_CMD" if dnstream else "IDLE"),
            ).Else(
                NextState("FILL"),
            )
        )
        # STORE targets the one channel in cmd.chan, so it needs the same per-channel
        # offset the readers get; a power-of-two stride keeps that a shift, not a multiply.
        # Built term by term so that chan_skew=0 emits exactly the expression it always did.
        _store_addr = st_addr_r + (_chan_sel(cmd_r["chan"]) << chan_shift if chan_stride else 0)
        if chan_skew:
            _store_addr = _store_addr + _chan_skew_of(_chan_sel(cmd_r["chan"]))
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
        fsm.act("ST_DRAIN", # Wait for the loader's B-channel drain: data is then readable.
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
                self.mstore_cmd.addr.eq(cmd_r["wbase"]),
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
        fsm.act("FILL_LAST", # Drain the registered fill write.
            self.busy.eq(1),
            If(native_mode, NextState('N_ISSUE') if with_native else NextState('ISSUE')
            ).Else(NextState('ISSUE')),
        )
        fsm.act("ISSUE",
            self.busy.eq(1),
            *[c.valid.eq(~fetch_mode | (fetch_chan == i))
              for i, c in enumerate(read_cmds)],
            *[c.addr.eq(Mux(raw_mode, raw_address, cmd_r["wbase"] + _chan_off(i)))
              for i, c in enumerate(read_cmds)],
            *[c.len.eq(cmd_r["wlen"]
                + Mux(i < cmd_r["rem"], cmd_r["sbeats"] + cmd_r["cbeats"], 0))
              for i, c in enumerate(read_cmds)],
            # Legacy NOTE: this waits on readers[0] alone. A reader only accepts while idle
            # (sink.ready = ~active), and readers do not always go idle together - with
            # rem != 0 they get different lengths, and a fetch commands one channel - so in
            # principle a reader still draining would miss its command. Measured on the card
            # it does not happen: at the stall every channel had delivered exactly the beats
            # commanded (ch0=10/10, ch1..7=5/5). Fixing it anyway cost WNS 0.015 -> -0.766 ns
            # on a design with no margin, and changed nothing, so it is left as written and
            # recorded here instead. Bounded raw reads use their selected port's
            # own handshake and require idle readers before command acceptance.
            If(Mux(raw_mode, Array([c.ready for c in read_cmds])[fetch_chan], read_cmds[0].ready),
                NextState("RUN"),
            )
        )
        mv_total = Signal(24)
        ax_total = Signal(24)
        # Test-visible aliases (no hardware: same signals under a name a testbench can find).
        # A stall in RUN is either "results never produced" or "produced but never retired",
        # and result_cnt against the expected total is what separates the two.
        self.dbg_result_cnt = result_cnt
        self.dbg_gw_we      = sp_gw_we  # W-port write cycles (a testbench counts them).
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
            If((native_mode | raw_mode) & self.error,
                NextState('N_FAILED') if with_native else NextState('IDLE'),
            ).Elif(Mux(dn_mode, dn_fin & ~self.store_busy, Mux(fetch_mode,
                   fetch_cnt == cmd_r["wlen"],
                   # Dual-Q ops retire two heads' results per stream.
                   result_cnt == Mux(cmd_r["qn"],
                       Mux(axpy_mode, ax_total << 1, mv_total << 1),
                       Mux(axpy_mode, ax_total, mv_total))))
               & ~gw_pending  # Queued writes must be in RAM before done.
               & ~am_busy & (~native_mode | native_complete) & (~raw_mode | raw_complete),
                self.done.eq(1),
                NextState("IDLE"),
            )
        )

        # Fill write-stage delay + result counting.
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

def reduce_or(sigs):
    from functools import reduce
    from operator import or_
    return reduce(or_, sigs)

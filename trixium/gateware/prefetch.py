# SPDX-License-Identifier: BSD-2-Clause
"""Run-ahead weight prefetch for one engine channel.

Between the lists the DDR sits idle while the units work (~12 M of the 27B's ~42 M cycles per
token at the v22 profile), and a token's weight read order never changes. This block sits
between the engine's read commands and the channel's AXI reader:

- mode 1 (record): the channel's weight commands (`cmd_weight`: the matvec kinds) of one
  forward go to a schedule memory, in order; every command is read directly.
- mode 2 (replay): `start` (each forward) rewinds and resynchronizes. The block walks the
  schedule ahead of the engine, `chunk` beats per reader command, into a deep FIFO while it
  has room. A weight command equal to the schedule's next entry is served from the FIFO (at
  the consumer's pace, not the DDR's); any other command goes to the reader. A weight
  command that does NOT match drops everything prefetched (the FIFO now, the in-flight
  chunks' beats as they arrive) and the channel runs direct until the next start: the
  prediction can cost time, never change what the engine receives.
- mode 0: every command direct.

Reader data returns in command order; a tag FIFO says whose each command's beats are (and
from which generation: a flush or a start moves to the next one, and older prefetch beats
are dropped). Direct commands go to the reader before any further chunk.
"""
from migen import *
from migen.genlib.fifo import SyncFIFO, SyncFIFOBuffered
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

MODE_OFF, MODE_RECORD, MODE_REPLAY = 0, 1, 2


class _PrefetchFifo(Module):
    """migen's SyncFIFOBuffered (registered read, first word fall through) with its memory
    named `pf_fifo`: the Xilinx target puts it in UltraRAM by that name (32k beats x 256 bits
    per channel would not fit the block RAM)."""
    def __init__(self, width, depth):
        self.din, self.we, self.writable = Signal(width), Signal(), Signal()
        self.dout, self.re, self.readable = Signal(width), Signal(), Signal()
        self.level = Signal(max=depth + 2)
        produce, consume = Signal(max=depth), Signal(max=depth)
        inner = Signal(max=depth + 1)
        storage = Memory(width, depth, name="pf_fifo")
        wp = storage.get_port(write_capable=True, mode=READ_FIRST)
        rp = storage.get_port(has_re=True, mode=READ_FIRST)
        self.specials += storage, wp, rp
        do_write, do_read = Signal(), Signal()
        inc = lambda v: If(v == depth - 1, v.eq(0)).Else(v.eq(v + 1))
        self.comb += [
            self.writable.eq(inner != depth),
            do_write.eq(self.we & self.writable),
            wp.adr.eq(produce), wp.dat_w.eq(self.din), wp.we.eq(do_write),
            # the inner FIFO's read feeds the output register
            do_read.eq((inner != 0) & (~self.readable | self.re)),
            rp.adr.eq(consume), rp.re.eq(do_read),
            self.dout.eq(rp.dat_r),
            self.level.eq(inner + self.readable),
        ]
        self.sync += [
            If(do_write, inc(produce)),
            If(do_read, inc(consume)),
            If(do_write & ~do_read, inner.eq(inner + 1)).Elif(do_read & ~do_write, inner.eq(inner - 1)),
            If(do_read, self.readable.eq(1)).Elif(self.re, self.readable.eq(0)),
        ]


class BonsaiWeightPrefetch(LiteXModule):
    def __init__(self, addr_width=33, depth=32768, sched_depth=1024, chunk=64):
        assert depth & (depth - 1) == 0 and 8 <= chunk <= depth//4
        self.depth, self.sched_depth, self.chunk = depth, sched_depth, chunk
        self.cmd        = stream.Endpoint([("addr", addr_width), ("len", 24)])   # from the engine
        self.cmd_weight = Signal()      # with cmd: a matvec weight read
        self.source     = stream.Endpoint([("data", 256)])                       # to the pipe side
        self.rd_cmd     = stream.Endpoint([("addr", addr_width), ("len", 24)])   # to the reader
        self.rd_data    = stream.Endpoint([("data", 256)])                       # from the reader
        self.mode       = Signal(2)     # MODE_*, quasi-static
        self.start      = Signal()      # pulse: a forward begins (no engine op in flight)
        self.hits       = Signal(32)    # weight commands served from the FIFO
        self.misses     = Signal(32)    # desynchronizations
        self.served     = Signal(32)    # beats served from the FIFO
        self.recorded   = Signal(16)    # schedule length
        self.synced     = Signal()

        # # #

        sb = log2_int(sched_depth)
        replay, record = Signal(), Signal()
        self.comb += [replay.eq(self.mode == MODE_REPLAY), record.eq(self.mode == MODE_RECORD)]

        # Schedule: port A writes (record) or reads for the issue side (replay); port B reads
        # for the consumer side. Registered reads: an index moves, the entry lands next cycle.
        sched = Memory(addr_width + 24, sched_depth, name="pf_sched")
        pa = sched.get_port(write_capable=True)
        pb = sched.get_port()
        self.specials += sched, pa, pb
        count = Signal(sb + 1)
        wi, pi, ci = Signal(sb + 1), Signal(sb + 1), Signal(sb + 1)
        po = Signal(24)                                  # beats issued inside entry pi
        # An index moves with *_req set; the next cycle the memory reads it (*_load); the
        # cycle after, the entry is captured.
        i_addr, i_len, i_ok, i_req, i_load = Signal(addr_width), Signal(24), Signal(), Signal(), Signal()
        c_addr, c_len, c_ok, c_req, c_load = Signal(addr_width), Signal(24), Signal(), Signal(), Signal()
        self.comb += [
            self.recorded.eq(count),
            pa.adr.eq(Mux(record, wi, pi)[:sb]),
            pb.adr.eq(ci[:sb]),
        ]
        self.sync += [
            i_load.eq(i_req), c_load.eq(c_req), i_req.eq(0), c_req.eq(0),
            If(i_load, i_addr.eq(pa.dat_r[:addr_width]), i_len.eq(pa.dat_r[addr_width:]),
               i_ok.eq(pi < count)),
            If(c_load, c_addr.eq(pb.dat_r[:addr_width]), c_len.eq(pb.dat_r[addr_width:]),
               c_ok.eq(ci < count)),
        ]

        # Deep FIFO (first-word-fall-through); flushed by a desync or a start.
        flush = Signal()
        self.submodules.fifo = fifo = ResetInserter()(_PrefetchFifo(256, depth))
        self.comb += fifo.reset.eq(flush)
        reserved = Signal(max=depth + chunk + 1)          # prefetch beats issued, not written

        # Tag FIFO: one entry per reader command.
        gen = Signal()
        self.submodules.tags = tags = SyncFIFO(26, 8)     # beats[24] | pf | gen
        t_pf, t_gen, t_n = Signal(), Signal(), Signal(24)
        self.comb += [t_n.eq(tags.dout[:24]), t_pf.eq(tags.dout[24]), t_gen.eq(tags.dout[25])]
        t_left = Signal(24)
        t_first = Signal(reset=1)                         # the head's first beat is next

        # Engine commands ---------------------------------------------------------------
        synced = self.synced
        serving, serve_left = Signal(), Signal(24)
        d_pend, d_addr, d_len = Signal(), Signal(addr_width), Signal(24)
        match, accept = Signal(), Signal()
        self.comb += [
            match.eq(c_ok & (self.cmd.addr == c_addr) & (self.cmd.len == c_len)),
            # One command at a time; a synced weight command waits for its entry to land.
            self.cmd.ready.eq(~d_pend & ~serving & ~self.start &
                              ~(replay & synced & self.cmd_weight & (c_req | c_load | ~c_ok) & (ci < count))),
            accept.eq(self.cmd.valid & self.cmd.ready),
        ]

        # Reader commands: a pending direct command first, else the next chunk.
        chunk_len, room, issue_pf, pf_took, d_took = Signal(24), Signal(), Signal(), Signal(), Signal()
        self.comb += [
            chunk_len.eq(Mux(i_len - po < chunk, i_len - po, chunk)),
            room.eq(fifo.level + reserved + chunk <= depth),
            issue_pf.eq(replay & synced & i_ok & ~i_req & ~i_load & ~d_pend & room & ~flush & ~self.start),
            self.rd_cmd.valid.eq(tags.writable & (d_pend | issue_pf)),
            If(d_pend,
                self.rd_cmd.addr.eq(d_addr), self.rd_cmd.len.eq(d_len),
            ).Else(
                self.rd_cmd.addr.eq(i_addr + (po << 5)), self.rd_cmd.len.eq(chunk_len),
            ),
            tags.we.eq(self.rd_cmd.valid & self.rd_cmd.ready),
            tags.din.eq(Cat(self.rd_cmd.len, ~d_pend, gen)),
            d_took.eq(tags.we & d_pend),
            pf_took.eq(tags.we & ~d_pend),
        ]

        # Reader data: direct beats out, current-generation prefetch beats into the FIFO,
        # older ones dropped.
        head_direct, head_keep = Signal(), Signal()
        take, fifo_in = Signal(), Signal()
        self.comb += [
            head_direct.eq(tags.readable & ~t_pf),
            head_keep.eq(tags.readable & t_pf & (t_gen == gen)),
            If(head_direct,
                self.source.valid.eq(self.rd_data.valid),
                self.source.data.eq(self.rd_data.data),
                self.rd_data.ready.eq(self.source.ready),
            ).Else(
                self.rd_data.ready.eq(tags.readable & (~head_keep | fifo.writable)),
                self.source.valid.eq(serving & fifo.readable),
                self.source.data.eq(fifo.dout),
            ),
            take.eq(self.rd_data.valid & self.rd_data.ready),
            fifo_in.eq(take & head_keep & ~flush),
            fifo.we.eq(fifo_in), fifo.din.eq(self.rd_data.data),
            fifo.re.eq(~head_direct & serving & self.source.ready),
        ]
        t_beats, t_last = Signal(24), Signal()
        self.comb += [
            t_beats.eq(Mux(t_first, t_n, t_left)),
            t_last.eq(take & (t_beats == 1)),
            tags.re.eq(t_last),
        ]
        out_fifo = Signal()
        self.comb += out_fifo.eq(fifo.re & fifo.readable)

        # Sequencing ----------------------------------------------------------------------
        desync = Signal()
        self.comb += [
            desync.eq(accept & replay & synced & self.cmd_weight & ~match),
            flush.eq(self.start | desync),
        ]
        self.sync += [
            # tag bookkeeping
            If(take, If(t_last, t_first.eq(1)).Else(t_first.eq(0), t_left.eq(t_beats - 1))),
            # reservations: + an issued chunk, - a kept beat; a flush forgets the in-flight ones
            If(flush, reserved.eq(0)).Else(
                reserved.eq(reserved + Mux(pf_took, chunk_len, 0) - fifo_in)),
            If(flush, gen.eq(~gen)),
            # the issue side
            If(pf_took,
                If(po + chunk_len == i_len,
                    po.eq(0), pi.eq(pi + 1), i_ok.eq(0), i_req.eq(1),
                ).Else(po.eq(po + chunk_len)),
            ),
            # direct commands
            If(d_took, d_pend.eq(0)),
            # serving
            If(out_fifo,
                self.served.eq(self.served + 1),
                serve_left.eq(serve_left - 1),
                If(serve_left == 1, serving.eq(0)),
            ),
            If(accept,
                If(record & self.cmd_weight & (wi < sched_depth),
                    wi.eq(wi + 1), count.eq(wi + 1),
                ),
                If(replay & synced & self.cmd_weight & match & (self.cmd.len != 0),
                    serving.eq(1), serve_left.eq(self.cmd.len),
                    ci.eq(ci + 1), c_ok.eq(0), c_req.eq(1),
                    self.hits.eq(self.hits + 1),
                ).Elif(self.cmd.len != 0,
                    d_pend.eq(1), d_addr.eq(self.cmd.addr), d_len.eq(self.cmd.len),
                ),
            ),
            If(desync, synced.eq(0), self.misses.eq(self.misses + 1)),
            If(self.start,
                pi.eq(0), ci.eq(0), po.eq(0), i_ok.eq(0), c_ok.eq(0),
                i_req.eq(replay), c_req.eq(replay),
                synced.eq(replay & (count != 0)),
                If(record, wi.eq(0), count.eq(0)),
            ),
        ]
        self.comb += [
            pa.we.eq(accept & record & self.cmd_weight & (wi < sched_depth)),
            pa.dat_w.eq(Cat(self.cmd.addr, self.cmd.len)),
        ]

# SPDX-License-Identifier: BSD-2-Clause
"""Bounded quantized-vector rotation through the existing scratchpad client port."""
from migen import *
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from trixium.gateware.hadamard import IntegerHadamard

HADAMARD_COMMAND = [("src", 24), ("signs", 24), ("dst", 24), ("stats", 24),
                    ("blocks", 16), ("inverse", 1)]


class BonsaiHadamardScratchpad(LiteXModule):
    """Transform complete int8 blocks into raw signed int32 scratchpad vectors.

    All four byte bases are 32-byte aligned. Signs are packed low-bit first,
    1=negate. stats receives one uint32 raw absmax per block, preserving unused
    lanes in its last beat. No normalization, exponent selection or quantization
    is performed. Output and rounded-up stats spans must not overlap each other
    or either input span; the complete command is validated before any access.

    The integrator grants every sp_rd_req (data returns next cycle), and gates
    writes with sp_wr_gnt. It must serialize other clients and drain older writes
    before accepting a command. Failure/cancel invalidates the entire output,
    including already-written prefixes. Only done publishes successful results.
    """
    def __init__(self, scratchpad_size=256*1024, block_size=1024, lanes=32):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be 32-byte aligned and within the 13-bit client address space")
        if block_size < 256 or block_size > 32768 or block_size & (block_size - 1):
            raise ValueError("block_size must be a power of two in [256, 32768]")
        # lanes: core butterflies per clock and values per core word. 32 = one scratchpad
        # input beat per word: ~10*(block/64 + 1) stage clocks against 10*(block/16 + 1) at
        # 8 lanes, the output then bound by the write port (8 int32 per beat, L/8 beats
        # per word). 27B prepares: ~1.5k -> ~0.4k clocks per 1,024-block.
        if lanes not in (8, 16, 32):
            raise ValueError("lanes must be 8, 16 or 32")
        self.cmd = stream.Endpoint(HADAMARD_COMMAND)
        self.cancel = Signal()
        self.busy = Signal()
        self.done = Signal()
        self.error = Signal(8)  # bit0 config, bit1 bounds, bit2 overlap, bit3 core.
        self.blocks_done = Signal(16)
        self.output_beats = Signal(24)
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(13)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(13)
        self.sp_wr_we = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()
        self.core = core = IntegerHadamard(block_size=block_size, lanes=lanes)
        L = lanes
        wpb, wps = 32//L, 256//L      # core words per data beat, per sign beat
        bpw = L//8                    # output beats per core word

        IDLE, START, SIGN_REQ, SIGN_CAP, DATA_REQ, DATA_CAP, SEND, OUTPUT, STATS, NEXT, FINISH, FAILED = range(12)
        state = Signal(4, reset=IDLE)
        captured = {name: Signal(width) for name, width in HADAMARD_COMMAND}
        block = Signal(16)
        in_word = Signal(max=block_size//L)
        out_word = Signal(max=block_size//8)   # output beat
        out_sub = Signal(max=max(2, bpw))      # beat within the core word
        inputs, signs = Signal(256), Signal(256)
        ends = {name: Signal(64) for name in ("src", "signs", "dst", "stats")}
        stats_bytes = Signal(64)
        config_bad, bounds_bad, overlap_bad = Signal(), Signal(), Signal()
        self.comb += [
            stats_bytes.eq(((self.cmd.blocks + C(7, 17)) >> 3) << 5),
            ends["src"].eq(self.cmd.src + self.cmd.blocks*block_size),
            ends["signs"].eq(self.cmd.signs + self.cmd.blocks*(block_size//8)),
            ends["dst"].eq(self.cmd.dst + self.cmd.blocks*(block_size*4)),
            ends["stats"].eq(self.cmd.stats + stats_bytes),
            config_bad.eq((self.cmd.blocks == 0) | (self.cmd.src[:5] != 0) |
                          (self.cmd.signs[:5] != 0) | (self.cmd.dst[:5] != 0) | (self.cmd.stats[:5] != 0)),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["signs"] > scratchpad_size) |
                          (ends["dst"] > scratchpad_size) | (ends["stats"] > scratchpad_size)),
        ]
        overlaps = []
        for a, b in (("dst", "src"), ("dst", "signs"), ("stats", "src"),
                     ("stats", "signs"), ("stats", "dst")):
            overlaps.append((getattr(self.cmd, a) < ends[b]) & (getattr(self.cmd, b) < ends[a]))
        from functools import reduce
        from operator import or_
        self.comb += overlap_bad.eq(reduce(or_, overlaps))

        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel),
            self.busy.eq(state != IDLE),
            core.cancel.eq(self.cancel | (state == FAILED)),
            core.cmd.valid.eq((state == START) & ~self.cancel),
            core.cmd.inverse.eq(captured["inverse"]),
            core.sink.valid.eq((state == SEND) & ~self.cancel),
            core.sink.data.eq(inputs[:8*L]), core.sink.negate.eq(signs[:L]),
            core.sink.last.eq(in_word == block_size//L - 1),
            core.source.ready.eq((state == OUTPUT) & self.sp_wr_gnt & (out_sub == bpw - 1) &
                                 ~self.cancel),
            self.sp_rd_req.eq(((state == SIGN_REQ) | (state == DATA_REQ)) & ~self.cancel),
            self.sp_rd_adr.eq(Mux(state == SIGN_REQ,
                (captured["signs"] >> 5) + block*(block_size//256) + (in_word >> (wps.bit_length() - 1)),
                (captured["src"] >> 5) + block*(block_size//32) + (in_word >> (wpb.bit_length() - 1)))),
            self.sp_wr_req.eq(((state == OUTPUT) & core.source.valid | (state == STATS)) & ~self.cancel),
            self.sp_wr_adr.eq(Mux(state == STATS,
                (captured["stats"] >> 5) + (block >> 3),
                (captured["dst"] >> 5) + block*(block_size//8) + out_word)),
            self.sp_wr_we.eq(Mux(state == STATS, C(1, 8) << block[:3], 0xff)),
        ]
        for lane in range(8):
            value = Signal((core.output_width, True))
            wide = Signal((32, True))
            ow = core.output_width
            self.comb += [value.eq(Array(core.source.data[(8*sub + lane)*ow:(8*sub + lane + 1)*ow]
                                         for sub in range(bpw))[out_sub] if bpw > 1 else
                                   core.source.data[lane*ow:(lane + 1)*ow]),
                          wide.eq(value),
                          self.sp_wr_dat[32*lane:32*(lane+1)].eq(Mux(state == STATS, core.absmax, wide))]
        self.sync += If(self.cancel,
            state.eq(IDLE), self.done.eq(0), self.error.eq(0),
            self.blocks_done.eq(0), self.output_beats.eq(0),
        ).Else(
            self.done.eq(0),
            If(core.error & (state != IDLE),
                self.error.eq(self.error | 8), state.eq(FAILED),
            ).Else(Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    *[value.eq(getattr(self.cmd, name)) for name, value in captured.items()],
                    block.eq(0), in_word.eq(0), out_word.eq(0), out_sub.eq(0),
                    self.blocks_done.eq(0), self.output_beats.eq(0),
                    self.error.eq(Cat(config_bad, bounds_bad, overlap_bad)),
                    If(config_bad | bounds_bad | overlap_bad, state.eq(FAILED)).Else(state.eq(START)),
                ),
                START: If(core.cmd.ready, state.eq(SIGN_REQ)),
                SIGN_REQ: state.eq(SIGN_CAP),
                SIGN_CAP: [signs.eq(self.sp_rd_dat), state.eq(DATA_REQ)],
                DATA_REQ: state.eq(DATA_CAP),
                DATA_CAP: [inputs.eq(self.sp_rd_dat), state.eq(SEND)],
                SEND: If(core.sink.ready,
                    inputs.eq(inputs >> 8*L), signs.eq(signs >> L),
                    If(in_word == block_size//L - 1, state.eq(OUTPUT)).Else(
                        in_word.eq(in_word + 1),
                        If((in_word & (wps - 1)) == wps - 1, state.eq(SIGN_REQ)
                        ).Elif((in_word & (wpb - 1)) == wpb - 1, state.eq(DATA_REQ)),
                    ),
                ),
                OUTPUT: [
                    If(core.source.valid & self.sp_wr_gnt,
                        out_word.eq(out_word + 1), self.output_beats.eq(self.output_beats + 1),
                        out_sub.eq(Mux(out_sub == bpw - 1, 0, out_sub + 1))),
                    If(core.done, state.eq(STATS)),
                ],
                STATS: If(self.sp_wr_gnt,
                    self.blocks_done.eq(self.blocks_done + 1), state.eq(NEXT)),
                NEXT: If(block == captured["blocks"] - 1,
                    state.eq(FINISH),
                ).Else(block.eq(block + 1), in_word.eq(0), out_word.eq(0), out_sub.eq(0),
                       state.eq(START)),
                FINISH: [self.done.eq(1), state.eq(IDLE)],
            })),
        )

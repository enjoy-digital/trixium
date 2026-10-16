#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Exact blockwise integer Hadamard transform and its scratchpad client.

IntegerHadamard is the board- and memory-independent kernel: one fixed-size block in natural
Sylvester order, without normalization or rounding. The caller owns quantization, block
exponents and whole-vector requantization.

HadamardScratchpad runs the kernel over the shared two-port scratchpad: it rotates complete
int8 blocks into raw signed int32 vectors and writes one raw absmax per block. Its command
(HADAMARD_COMMAND, also the firmware's descriptor layout) is: src/signs/dst/stats byte bases
(32-byte aligned), blocks (block count) and inverse.
"""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import LiteXModule

from litex.soc.interconnect import stream

# Integer Hadamard ---------------------------------------------------------------------------------

class IntegerHadamard(LiteXModule):
    """One fixed-size block, natural Sylvester order, no normalization/rounding.

    cmd.inverse=0 computes H*(sign*x); inverse=1 computes sign*(H*x). cmd.half=1 transforms
    a block of block_size/2 instead (the first words/2 words, one stage less: the Sylvester
    sub-transform, e.g. 512-element blocks on a 1,024 kernel).
    sink carries ``lanes`` little-endian signed integers and one negate bit per
    lane (1 means sign=-1). last must mark exactly the final input word. Output
    integers have input_width + log2(block_size) + 1 bits: sign negation of the
    most-negative input and the complete sum both fit without saturation.

    Two ping-pong true-dual-port word memories perform ``lanes`` butterflies
    per cycle. Low stages butterfly within each of two words; high stages pair
    corresponding lanes of two words. A drain cycle separates stages. The input
    and output interfaces transfer ``lanes`` values per cycle, with backpressure.

    done pulses after the last output handshake and log2(lanes) absmax drain
    cycles; absmax then describes the entire block. A framing failure is sticky
    (busy) until cancel. cancel discards all valid state, including stalled
    outputs, but does not clear RAM: a new command must overwrite a complete
    block before any result can be emitted. An external consumer must discard
    any prefix from a cancelled command.
    """
    def __init__(self, block_size=1024, lanes=8, input_width=8):
        if (not isinstance(block_size, int) or block_size < 4 or
                block_size & (block_size - 1)):
            raise ValueError("block_size must be a power of two >= 4")
        if (not isinstance(lanes, int) or lanes < 2 or lanes & (lanes - 1) or
                lanes > block_size // 2):
            raise ValueError("lanes must be a power of two in [2, block_size/2]")
        if not isinstance(input_width, int) or not 2 <= input_width <= 32:
            raise ValueError("input_width must be in [2, 32]")
        stages, lane_bits = block_size.bit_length() - 1, lanes.bit_length() - 1
        words = block_size // lanes
        word_bits = words.bit_length() - 1
        width = self.output_width = input_width + stages + 1
        self.block_size, self.lanes, self.input_width = block_size, lanes, input_width

        # Streams.
        self.cmd    = stream.Endpoint([("inverse", 1), ("half", 1)])
        self.sink   = stream.Endpoint([("data", lanes*input_width), ("negate", lanes)])
        self.source = stream.Endpoint([("data", lanes*width)])

        # Control / status.
        self.cancel    = Signal()
        self.busy      = Signal()
        self.done      = Signal()
        self.error     = Signal()
        self.cancelled = Signal()
        self.absmax    = Signal(width)

        # # #

        IDLE, LOAD, RUN, DRAIN, OUTPUT, FAILED, FINISH = range(7)
        state = Signal(3, reset=IDLE)
        inverse, bank = Signal(reset_less=True), Signal() # inverse: latched with the command.
        half = Signal(reset_less=True)                     # (latched with the command)
        last_word  = Signal(word_bits)                     # words (or words/2) - 1
        last_pair  = Signal(max=max(2, words//2))          # pairs per stage - 1
        last_stage = Signal(max=stages)
        self.comb += [
            last_word.eq(Mux(half, words//2 - 1, words - 1)),
            last_pair.eq(Mux(half, max(words//4, 1) - 1, words//2 - 1)),
            last_stage.eq(Mux(half, stages - 2, stages - 1)),
        ]
        stage = Signal(max=stages)
        load_index = Signal(word_bits)
        pair = Signal(max=max(2, words//2))
        calc_valid = Signal()
        write_a, write_b = Signal(word_bits, reset_less=True), Signal(word_bits, reset_less=True)
        read_a, read_b = Signal(word_bits), Signal(word_bits)
        issued = Signal(max=words + 1)
        output_index = Signal(word_bits)
        output_valid, output_read = Signal(), Signal()

        # Ping-pong banks and the per-word sign bits.
        memories = [Memory(lanes*width, words, name=f"hadamard_bank{i}") for i in range(2)]
        # Explicit registered-data/no-change ports infer true dual-port RAM.
        # Migen's default WRITE_FIRST emits registered addresses + asynchronous
        # data reads, which Vivado rejects for this two-writer memory template.
        # No algorithm phase observes a port's output during its write cycle.
        ports = [[memory.get_port(write_capable=True, has_re=True, mode=NO_CHANGE)
                  for _ in range(2)] for memory in memories]
        signs = Memory(lanes, words, name="hadamard_signs")
        sign_write = signs.get_port(write_capable=True)
        sign_read = signs.get_port(has_re=True, mode=READ_FIRST)
        self.specials += memories + [p for group in ports for p in group] + [
            signs, sign_write, sign_read]

        # Input load (sign applied on the way in for the forward transform).
        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel),
            self.sink.ready.eq((state == LOAD) & ~self.cancel),
            self.source.valid.eq((state == OUTPUT) & output_valid & ~self.cancel),
            self.source.first.eq(output_index == 0),
            self.source.last.eq(output_index == last_word),
            self.busy.eq(state != IDLE),
            output_read.eq((state == OUTPUT) & (~output_valid | self.source.ready) &
                           (issued <= last_word) & ~self.cancel),
            sign_write.adr.eq(load_index), sign_write.dat_w.eq(self.sink.negate),
            sign_write.we.eq(self.sink.valid & self.sink.ready),
            sign_read.adr.eq(issued), sign_read.re.eq(output_read),
        ]
        incoming = []
        for lane in range(lanes):
            value = Signal((input_width, True))
            signed = Signal((width, True))
            self.comb += [value.eq(self.sink.data[lane*input_width:(lane+1)*input_width]),
                          signed.eq(Mux(~inverse & self.sink.negate[lane], -value, value))]
            incoming.append(signed)

        # Butterflies: insert the high-stage butterfly bit into the sequential pair index.
        addresses = {}
        for number in range(stages):
            if number < lane_bits:
                a, b = pair << 1, (pair << 1) | 1
            else:
                bit = number - lane_bits
                a = ((pair >> bit) << (bit + 1)) | (pair & ((1 << bit) - 1))
                b = a | (1 << bit)
            addresses[number] = [read_a.eq(a), read_b.eq(b)]
        self.comb += Case(stage, addresses)

        operands = [[Signal((width, True)) for _ in range(lanes)] for _ in range(2)]
        results = [[Signal((width, True)) for _ in range(lanes)] for _ in range(2)]
        selected = [[(Signal((width, True)), Signal((width, True)), Signal())
                     for _ in range(lanes)] for _ in range(2)]
        for port in range(2):
            for lane in range(lanes):
                low, high = lane*width, (lane+1)*width
                self.comb += operands[port][lane].eq(Mux(bank,
                    ports[1][port].dat_r[low:high], ports[0][port].dat_r[low:high]))
        butterflies = {}
        for number in range(stages):
            assignments = []
            for lane in range(lanes):
                if number < lane_bits:
                    bit = 1 << number
                    for port in range(2):
                        a, b = operands[port][lane & ~bit], operands[port][lane | bit]
                        lhs, rhs, subtract = selected[port][lane]
                        assignments += [lhs.eq(a), rhs.eq(b), subtract.eq(bool(lane & bit))]
                else:
                    a, b = operands[0][lane], operands[1][lane]
                    for port in range(2):
                        lhs, rhs, subtract = selected[port][lane]
                        assignments += [lhs.eq(a), rhs.eq(b), subtract.eq(port)]
            butterflies[number] = assignments
        self.comb += Case(stage, butterflies)
        for port in range(2):
            for lane in range(lanes):
                lhs, rhs, subtract = selected[port][lane]
                # Select operands BEFORE arithmetic; do not duplicate an adder
                # bank for each stage. Fixed-width two's-complement add/subtract.
                self.comb += results[port][lane].eq(
                    lhs + (rhs ^ Replicate(subtract, width)) + subtract)

        for which in range(2):
            for port in range(2):
                ram = ports[which][port]
                self.comb += [ram.adr.eq(0), ram.re.eq(0), ram.we.eq(0), ram.dat_w.eq(0)]
                self.comb += If(~self.cancel,
                    If(state == LOAD,
                        *([ram.adr.eq(load_index), ram.dat_w.eq(Cat(*incoming)),
                           ram.we.eq(self.sink.valid & self.sink.ready)]
                          if which == 0 and port == 0 else []),
                    ).Elif((state == RUN) | (state == DRAIN),
                        If(bank == which,
                            ram.adr.eq(read_b if port else read_a), ram.re.eq(state == RUN),
                        ).Else(
                            ram.adr.eq(write_b if port else write_a),
                            ram.dat_w.eq(Cat(*results[port])), ram.we.eq(calc_valid),
                        ),
                    ).Elif((state == OUTPUT) & (bank == which),
                        *([ram.adr.eq(issued), ram.re.eq(output_read)] if port == 0 else []),
                    ),
                )

        # Output (sign applied on the way out for the inverse) and absmax tree.
        magnitudes = []
        for lane in range(lanes):
            value = Signal((width, True))
            result = Signal((width, True))
            magnitude = Signal(width)
            self.comb += [value.eq(operands[0][lane]),
                          result.eq(Mux(inverse & sign_read.dat_r[lane], -value, value)),
                          self.source.data[lane*width:(lane+1)*width].eq(result),
                          # Sign restoration never changes magnitude. Avoid a
                          # second negate on the absmax path in inverse mode.
                          magnitude.eq(Mux(value < 0, -value, value))]
            magnitudes.append(magnitude)
        maximum_valid = self.source.valid & self.source.ready
        maximum_last = self.source.last
        while len(magnitudes) > 1:
            level = []
            valid, last = Signal(), Signal(reset_less=True) # last: qualified by valid.
            capture = []
            for a, b in zip(magnitudes[::2], magnitudes[1::2]):
                maximum = Signal(width, reset_less=True)
                capture.append(maximum.eq(Mux(a > b, a, b)))
                level.append(maximum)
            # Pipeline each tree level, retaining one accepted word per cycle.
            # Without this, RAM -> abs -> three comparisons -> running maximum
            # missed even the synthesis-estimated 150 MHz budget.
            self.sync += If(self.cancel, valid.eq(0)).Else(
                valid.eq(maximum_valid),
                If(maximum_valid, last.eq(maximum_last), *capture),
            )
            magnitudes = level
            maximum_valid, maximum_last = valid, last

        # Control.
        self.sync += If(self.cancel,
            state.eq(IDLE), calc_valid.eq(0), output_valid.eq(0),
            self.done.eq(0), self.error.eq(0), self.cancelled.eq(1), self.absmax.eq(0),
        ).Else(
            self.done.eq(0),
            If(maximum_valid & ((state == OUTPUT) | (state == FINISH)),
                If(magnitudes[0] > self.absmax, self.absmax.eq(magnitudes[0])),
                If(maximum_last & (state == FINISH), self.done.eq(1), state.eq(IDLE)),
            ),
            Case(state, {
                IDLE: If(self.cmd.valid & self.cmd.ready,
                    # A half block needs two pairs of words (words >= 4); else ignored.
                    inverse.eq(self.cmd.inverse), half.eq(self.cmd.half & (words >= 4)),
                    bank.eq(0), stage.eq(0), load_index.eq(0),
                    pair.eq(0), calc_valid.eq(0), output_valid.eq(0), issued.eq(0),
                    self.error.eq(0), self.cancelled.eq(0), self.absmax.eq(0), state.eq(LOAD),
                ),
                LOAD: If(self.sink.valid & self.sink.ready,
                    If(self.sink.last != (load_index == last_word),
                        self.error.eq(1), state.eq(FAILED),
                    ).Elif(load_index == last_word,
                        state.eq(RUN),
                    ).Else(load_index.eq(load_index + 1)),
                ),
                RUN: [
                    calc_valid.eq(1), write_a.eq(read_a), write_b.eq(read_b),
                    If(pair == last_pair, state.eq(DRAIN)).Else(pair.eq(pair + 1)),
                ],
                DRAIN: [
                    calc_valid.eq(0), bank.eq(~bank), pair.eq(0),
                    If(stage == last_stage,
                        state.eq(OUTPUT),
                    ).Else(stage.eq(stage + 1), state.eq(RUN)),
                ],
                OUTPUT: [
                    If(output_read,
                        output_valid.eq(1), output_index.eq(issued), issued.eq(issued + 1),
                    ).Elif(self.source.ready, output_valid.eq(0)),
                    If(self.source.valid & self.source.ready & self.source.last, state.eq(FINISH)),
                ],
            }),
        )

# Hadamard Scratchpad ------------------------------------------------------------------------------

HADAMARD_COMMAND = [("src", 24), ("signs", 24), ("dst", 24), ("stats", 24),
                    ("blocks", 16), ("inverse", 1), ("half", 1)]

class HadamardScratchpad(LiteXModule):
    """Transform complete int8 blocks into raw signed int32 scratchpad vectors.

    All four byte bases are 32-byte aligned. Signs are packed low-bit first,
    1=negate. half=1: blocks of block_size/2 (ABI 1.1; block_size >= 512, so a block's signs
    stay whole beats), packed back to back like full blocks. stats receives one uint32 raw absmax per block, preserving unused
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
            raise ValueError("scratchpad_size must be 32-byte aligned and within the 13-bit "
                             "client address space")
        if block_size < 256 or block_size > 32768 or block_size & (block_size - 1):
            raise ValueError("block_size must be a power of two in [256, 32768]")
        # lanes: core butterflies per clock and values per core word. 32 = one scratchpad
        # input beat per word: ~10*(block/64 + 1) stage clocks against 10*(block/16 + 1) at
        # 8 lanes, the output then bound by the write port (8 int32 per beat, L/8 beats
        # per word): ~1.5k -> ~0.4k clocks per 1,024-element block.
        if lanes not in (8, 16, 32):
            raise ValueError("lanes must be 8, 16 or 32")

        # Command / status.
        self.cmd          = stream.Endpoint(HADAMARD_COMMAND)
        self.cancel       = Signal()
        self.busy         = Signal()
        self.done         = Signal()
        self.error        = Signal(8) # bit0 config, bit1 bounds, bit2 overlap, bit3 core.
        self.blocks_done  = Signal(16)
        self.output_beats = Signal(24)

        # Scratchpad client ports.
        self.sp_rd_req = Signal()
        self.sp_rd_adr = Signal(13)
        self.sp_rd_dat = Signal(256)
        self.sp_wr_req = Signal()
        self.sp_wr_adr = Signal(13)
        self.sp_wr_we  = Signal(8)
        self.sp_wr_dat = Signal(256)
        self.sp_wr_gnt = Signal()

        # # #

        # Kernel.
        self.core = core = IntegerHadamard(block_size=block_size, lanes=lanes)
        L = lanes
        half_ok = block_size >= 512 and block_size//L >= 4 # Else half=1 is a config error.
        wpb, wps = 32//L, 256//L      # core words per data beat, per sign beat
        bpw = L//8                    # output beats per core word

        # Command capture and validation.
        (IDLE, START, SIGN_REQ, SIGN_CAP, DATA_REQ, DATA_CAP, SEND, OUTPUT, STATS, NEXT, FINISH,
         FAILED, SIGNS, STREAM) = range(14)
        state = Signal(4, reset=IDLE)
        # Streamed input (32 lanes, one core word per scratchpad beat, blocks <= 1,024): the
        # block's sign beats first (one read per clock into a block-wide register), then one
        # data read per clock straight into the kernel, which is always ready while loading.
        # ~block/32 + block/256 + 2 clocks per block instead of three per word (request,
        # capture, send) plus two per sign beat.
        streamed = (L == 32) and (block_size <= 1024)
        nsb      = block_size//256              # sign beats per block
        words    = block_size//L
        captured = {name: Signal(width, reset_less=True) for name, width in HADAMARD_COMMAND}
        block = Signal(16)
        in_word = Signal(max=block_size//L)
        out_word = Signal(max=block_size//8)   # output beat
        out_sub = Signal(max=max(2, bpw))      # beat within the core word
        inputs, signs = Signal(256, reset_less=True), Signal(256, reset_less=True)
        ends = {name: Signal(64) for name in ("src", "signs", "dst", "stats")}
        stats_bytes = Signal(64)
        config_bad, bounds_bad, overlap_bad = Signal(), Signal(), Signal()
        # Per-block strides (bytes, beats) of the command being validated / the captured one.
        cmd_bsz = Signal(max=block_size + 1)
        half = captured["half"] if half_ok else C(0, 1)
        src_beats  = Signal(max=block_size//32 + 1)   # src beats (core words when streamed)
        sign_beats = Signal(max=nsb + 1)
        dst_beats  = Signal(max=block_size//8 + 1)
        last_in    = Signal(max=block_size//L)
        self.comb += [
            cmd_bsz.eq(Mux(self.cmd.half & half_ok, block_size//2, block_size)),
            src_beats.eq(Mux(half, block_size//64, block_size//32)),
            sign_beats.eq(Mux(half, nsb//2, nsb)),
            dst_beats.eq(Mux(half, block_size//16, block_size//8)),
            last_in.eq(Mux(half, block_size//(2*L) - 1, block_size//L - 1)),
        ]
        self.comb += [
            stats_bytes.eq(((self.cmd.blocks + C(7, 17)) >> 3) << 5),
            ends["src"].eq(self.cmd.src + self.cmd.blocks*cmd_bsz),
            ends["signs"].eq(self.cmd.signs + self.cmd.blocks*(cmd_bsz >> 3)),
            ends["dst"].eq(self.cmd.dst + self.cmd.blocks*(cmd_bsz << 2)),
            ends["stats"].eq(self.cmd.stats + stats_bytes),
            config_bad.eq((self.cmd.blocks == 0) | (self.cmd.src[:5] != 0) |
                          (self.cmd.half & (not half_ok)) |
                          (self.cmd.signs[:5] != 0) | (self.cmd.dst[:5] != 0) |
                          (self.cmd.stats[:5] != 0)),
            bounds_bad.eq((ends["src"] > scratchpad_size) | (ends["signs"] > scratchpad_size) |
                          (ends["dst"] > scratchpad_size) | (ends["stats"] > scratchpad_size)),
        ]
        overlaps = []
        for a, b in (("dst", "src"), ("dst", "signs"), ("stats", "src"),
                     ("stats", "signs"), ("stats", "dst")):
            overlaps.append((getattr(self.cmd, a) < ends[b]) & (getattr(self.cmd, b) < ends[a]))
        self.comb += overlap_bad.eq(reduce(or_, overlaps))

        # Datapath: scratchpad reads -> kernel -> int32 output beats and the block absmax.
        self.comb += [
            self.cmd.ready.eq((state == IDLE) & ~self.cancel),
            self.busy.eq(state != IDLE),
            core.cancel.eq(self.cancel | (state == FAILED)),
            core.cmd.valid.eq((state == START) & ~self.cancel),
            core.cmd.inverse.eq(captured["inverse"]),
            core.cmd.half.eq(half),
            core.source.ready.eq((state == OUTPUT) & self.sp_wr_gnt & (out_sub == bpw - 1) &
                                 ~self.cancel),
            self.sp_wr_req.eq(((state == OUTPUT) & core.source.valid | (state == STATS)) &
                              ~self.cancel),
            self.sp_wr_adr.eq(Mux(state == STATS,
                (captured["stats"] >> 5) + (block >> 3),
                (captured["dst"] >> 5) + block*dst_beats + out_word)),
            self.sp_wr_we.eq(Mux(state == STATS, C(1, 8) << block[:3], 0xff)),
        ]
        if not streamed:
            self.comb += [
                core.sink.valid.eq((state == SEND) & ~self.cancel),
                core.sink.data.eq(inputs[:8*L]), core.sink.negate.eq(signs[:L]),
                core.sink.last.eq(in_word == last_in),
                self.sp_rd_req.eq(((state == SIGN_REQ) | (state == DATA_REQ)) & ~self.cancel),
                self.sp_rd_adr.eq(Mux(state == SIGN_REQ,
                    (captured["signs"] >> 5) + block*sign_beats +
                    (in_word >> (wps.bit_length() - 1)),
                    (captured["src"] >> 5) + block*src_beats +
                    (in_word >> (wpb.bit_length() - 1)))),
            ]
        else:
            issue      = Signal(max=max(2, words + 1))  # reads issued in SIGNS/STREAM
            sign_pend  = Signal()                       # a sign read's data is on sp_rd_dat
            sign_shift = Signal(block_size, reset_less=True)
            data_pend  = Signal()                       # a data read's data is on sp_rd_dat
            data_last  = Signal(reset_less=True)        # (data_pend)
            self.comb += [
                core.sink.valid.eq(data_pend & (state == STREAM) & ~self.cancel),
                core.sink.data.eq(self.sp_rd_dat[:8*L]),
                core.sink.negate.eq(sign_shift[:L]),
                core.sink.last.eq(data_last),
                self.sp_rd_req.eq((((state == SIGNS) & (issue < sign_beats)) |
                                   ((state == STREAM) & (issue <= last_in))) & ~self.cancel),
                self.sp_rd_adr.eq(Mux(state == SIGNS,
                    (captured["signs"] >> 5) + block*sign_beats + issue,
                    (captured["src"] >> 5) + block*src_beats + issue)),
            ]
            self.sync += [
                If(self.cancel | (state == FAILED),
                    sign_pend.eq(0), data_pend.eq(0),
                ).Else(
                    sign_pend.eq(self.sp_rd_req & (state == SIGNS)),
                    data_pend.eq(self.sp_rd_req & (state == STREAM)),
                ),
                If(self.sp_rd_req, data_last.eq(issue == last_in)),
                # Sign beats arrive in order: shift each in from the top, so after nsb beats
                # beat 0 sits at the bottom; each accepted data word then consumes L bits.
                # A half block's nsb/2 beats shift in at the middle instead.
                If(sign_pend,
                    If(half,
                        sign_shift.eq(Cat(sign_shift[256:block_size//2], self.sp_rd_dat,
                                          sign_shift[block_size//2:])),
                    ).Else(
                        sign_shift.eq(Cat(sign_shift[256:], self.sp_rd_dat) if nsb > 1 else
                                      self.sp_rd_dat)),
                ).Elif(core.sink.valid & core.sink.ready,
                    sign_shift.eq(sign_shift >> L)),
            ]
        for lane in range(8):
            value = Signal((core.output_width, True))
            wide = Signal((32, True))
            ow = core.output_width
            self.comb += [value.eq(Array(core.source.data[(8*sub + lane)*ow:(8*sub + lane + 1)*ow]
                                         for sub in range(bpw))[out_sub] if bpw > 1 else
                                   core.source.data[lane*ow:(lane + 1)*ow]),
                          wide.eq(value),
                          self.sp_wr_dat[32*lane:32*(lane+1)].eq(
                              Mux(state == STATS, core.absmax, wide))]
        # Control.
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
                    If(config_bad | bounds_bad | overlap_bad,
                        state.eq(FAILED),
                    ).Else(state.eq(START)),
                ),
                START: If(core.cmd.ready, state.eq(SIGNS if streamed else SIGN_REQ),
                          *([issue.eq(0)] if streamed else [])),
                **({
                    SIGNS: If(self.sp_rd_req,
                        issue.eq(issue + 1),
                    ).Elif(~sign_pend,         # the last sign beat has landed
                        issue.eq(0), state.eq(STREAM)),
                    STREAM: [
                        If(self.sp_rd_req, issue.eq(issue + 1)),
                        If(core.sink.valid & core.sink.ready & data_last, state.eq(OUTPUT)),
                    ],
                } if streamed else {}),
                SIGN_REQ: state.eq(SIGN_CAP),
                SIGN_CAP: [signs.eq(self.sp_rd_dat), state.eq(DATA_REQ)],
                DATA_REQ: state.eq(DATA_CAP),
                DATA_CAP: [inputs.eq(self.sp_rd_dat), state.eq(SEND)],
                SEND: If(core.sink.ready,
                    inputs.eq(inputs >> 8*L), signs.eq(signs >> L),
                    If(in_word == last_in, state.eq(OUTPUT)).Else(
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

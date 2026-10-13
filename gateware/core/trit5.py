#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Trit5Unpacker: ternary code beats stored 5 trits per byte, back to the pipe's 2-bit codes.

A ternary row in memory (model.layout, TRIT5): its scale beats (32 bytes, 16 fp16 scales each)
as today, then its code beats at 26 bytes each (128 trits, five per byte in base 3: lane 5i+k is
digit k of byte i, d = trit + 1 in {0, 1, 2}; the 26th byte holds lanes 125-127), the row padded
to a whole 32-byte memory beat. The pipe's 2-bit code of a trit is its digit (0: -x, 1: 0,
2: +x), so a decoded byte is five codes. A byte above 242 decodes to code 3 in every lane, which
the pipe flags as an error, as a reserved 2-bit code is today.

Memory beats in (32 bytes), pipe beats out (256 bits): one beat per clock when the memory keeps
up, a code beat using 26/32 of a memory beat. Scale beats pass through (a row starts on a beat
boundary). The packet counters (scale_beats, code_beats per row) are the pipe's own parameters.
`packed` low: every beat passes through unchanged (q8, AXPY, unpacked images).
"""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

BEAT_BYTES = 32
CODE_BYTES = 26 # ceil(128/5).

# Helpers ------------------------------------------------------------------------------------------

def trit5_decode_table():
    """byte -> 10-bit code word (five 2-bit codes, lane k at bits [2k, 2k+2)); invalid -> all 3."""
    table = []
    for b in range(256):
        if b > 242:
            table.append(0x3ff)
            continue
        w, v = 0, b
        for k in range(5):
            w |= (v % 3) << (2*k)
            v //= 3
        table.append(w)
    return table


def trit5_pack(codes):
    """128 2-bit codes (digits 0..2) -> 26 bytes (the converter's packing, the reference)."""
    assert len(codes) == 128 and all(0 <= c <= 2 for c in codes)
    out = bytearray()
    for i in range(0, 128, 5):
        v = 0
        for k, c in enumerate(codes[i:i + 5]):
            v += c*3**k
        out.append(v)
    assert len(out) == CODE_BYTES
    return bytes(out)

# Trit5 Unpacker -----------------------------------------------------------------------------------

class Trit5Unpacker(LiteXModule):
    def __init__(self):
        # Op parameters (quasi-static while busy, as the pipe's).
        self.packed      = Signal(reset_less=True)
        self.scale_beats = Signal(4, reset_less=True)
        self.code_beats  = Signal(8, reset_less=True)
        self.start       = Signal() # Pulse: a new op (counters, buffer and offset reset).

        self.sink   = sink   = stream.Endpoint([("data", 256)])
        self.source = source = stream.Endpoint([("data", 256)])

        # # #

        # Beat buffer: up to three memory beats; b[0] is the oldest.
        b  = [Signal(256, reset_less=True) for _ in range(3)]
        n  = Signal(2)       # Beats held.
        r  = Signal(5)       # Byte offset of the next code beat in b[0].

        # Packet position (the pipe's parser, one step ahead of it).
        phase = Signal()     # 0: scale beats, 1: code beats.
        s_cnt = Signal(4)
        g_cnt = Signal(8)

        # What the next output beat needs: a scale beat is b[0]; a code beat b[0] from r, plus
        # b[1] when it crosses the beat boundary.
        r_end    = Signal(6)
        crosses  = Signal()
        row_last = Signal()
        need     = Signal(2)
        self.comb += [
            r_end.eq(r + CODE_BYTES),
            crosses.eq(r_end > BEAT_BYTES),
            row_last.eq(phase & (g_cnt == self.code_beats - 1)),
            If(~self.packed | ~phase,
                need.eq(1)
            ).Else(
                need.eq(Mux(crosses, 2, 1))
            ),
        ]

        # Beats consumed by an output beat: scale beat (or pass-through) 1; code beat 1 when it
        # reaches the end of b[0], plus the row padding left after the row's last code beat.
        pops  = Signal(2)
        r_nxt = Signal(6)
        self.comb += [
            If(~self.packed | ~phase,
                pops.eq(1), r_nxt.eq(0),
            ).Elif(row_last,
                # The row ends on a beat boundary: drop what is left of the beat it ends in.
                pops.eq(Mux(r_end > BEAT_BYTES, 2, 1)), r_nxt.eq(0),
            ).Elif(r_end >= BEAT_BYTES,
                pops.eq(1), r_nxt.eq(r_end - BEAT_BYTES),
            ).Else(
                pops.eq(0), r_nxt.eq(r_end),
            ),
        ]

        # Output stage 1: the 26-byte window, registered, then decoded into stage 2.
        win  = Signal(2*256)
        self.comb += win.eq(Cat(b[0], b[1]))
        sel  = Signal(8*CODE_BYTES)
        self.comb += Case(r, {k: sel.eq(win[8*k:8*(k + CODE_BYTES)]) for k in range(BEAT_BYTES)})

        s1_valid = Signal()
        s1_code  = Signal(reset_less=True)
        s1_data  = Signal(256, reset_less=True)
        s1_bytes = Signal(8*CODE_BYTES, reset_less=True)
        s2_ready = Signal()
        s1_ready = Signal()
        self.comb += s1_ready.eq(~s1_valid | s2_ready)

        fire = Signal()
        self.comb += fire.eq(s1_ready & (n >= need) & ~self.start)

        self.sync += [
            If(self.start,
                s1_valid.eq(0),
            ).Elif(s1_ready,
                s1_valid.eq(fire),
                s1_code.eq(self.packed & phase),
                s1_data.eq(b[0]),
                s1_bytes.eq(sel),
            )
        ]

        # Stage 2: decode (a 256 x 10-bit table per byte) and the output register.
        table = trit5_decode_table()
        codes = Signal(256)
        for i in range(CODE_BYTES):
            word  = Signal(10)
            byte  = s1_bytes[8*i:8*i + 8]
            self.comb += Case(byte, {v: word.eq(table[v]) for v in range(256)})
            lanes = min(5, 128 - 5*i)
            self.comb += codes[10*i:10*i + 2*lanes].eq(word[:2*lanes])

        s2_valid = Signal()
        s2_data  = Signal(256, reset_less=True)
        self.comb += s2_ready.eq(~s2_valid | source.ready)
        self.sync += [
            If(self.start,
                s2_valid.eq(0),
            ).Elif(s2_ready,
                s2_valid.eq(s1_valid),
                s2_data.eq(Mux(s1_code, codes, s1_data)),
            )
        ]
        self.comb += [
            source.valid.eq(s2_valid),
            source.data.eq(s2_data),
        ]

        # Packet counters and offset, on each output beat.
        self.sync += [
            If(self.start,
                phase.eq(0), s_cnt.eq(0), g_cnt.eq(0), r.eq(0),
            ).Elif(fire,
                r.eq(r_nxt),
                If(~phase,
                    s_cnt.eq(s_cnt + 1),
                    If(s_cnt == self.scale_beats - 1, phase.eq(1), g_cnt.eq(0)),
                ).Else(
                    g_cnt.eq(g_cnt + 1),
                    If(row_last, phase.eq(0), s_cnt.eq(0)),
                ),
            )
        ]

        # Beat buffer: pop `pops` on an output beat, push the accepted memory beat behind.
        p      = Signal(2)
        n_left = Signal(2)
        self.comb += [
            p.eq(Mux(fire, pops, 0)),
            n_left.eq(n - p),
            sink.ready.eq(~self.start & (n_left < 3)),
        ]
        push = Signal()
        self.comb += push.eq(sink.valid & sink.ready)
        nb = [Signal(256) for _ in range(3)]
        for k in range(3):
            # Slot k after the pops: the beat that was at k + p, or the pushed one at n_left.
            self.comb += [
                nb[k].eq(b[k]),
                If(p == 1, nb[k].eq(b[k + 1] if k + 1 < 3 else b[k])),
                If(p == 2, nb[k].eq(b[k + 2] if k + 2 < 3 else b[k])),
                If(push & (n_left == k), nb[k].eq(sink.data)),
            ]
            self.sync += b[k].eq(nb[k])
        self.sync += [
            If(self.start,
                n.eq(0),
            ).Else(
                n.eq(n_left + push),
            )
        ]

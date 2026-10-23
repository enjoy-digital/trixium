#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Resident attention softmax over the shared scratchpad (sequencer unit 14).

For each of `rows` rows of `n` FP32 scores at src + row*stride*32 (stride in beats, 0 =
ceil(n/8)), in place, bit-exact with hybrid.c's per-head loop (softmax_rows):

  mx = -1e30f;  for t < n: if (s[t] > mx) mx = s[t];                      a NaN never wins
  sum = +0.0f;  for t < n: e[t] = softmax_exp(s[t] - mx); sum += e[t];    in t order
  inv = 1.0f/sum;  for t < n: p[t] = e[t]*inv

softmax_exp is the firmware's canonical exp (fastmath.h trixium_softmax_exp: the bounded exp -
range reduction, six fused multiply-adds, an exact 2^n - on [-80, 0], +0 below, a NaN through
as the canonical NaN). Every operation is an exact IEEE binary32 operation (fp32.py: FP32Add,
FP32Mul, FP32Fma, FP32DivSqrt), so there are no fixups. Arguments above +0 (impossible in a
softmax: s - max <= 0) take the same bounded exp, whose 2^n overflows past +88 as the CPU's.

Three passes over the rows:
  1. the maximum, 8 lanes per clock (one beat per clock), with the sign-magnitude ordering the
     argmax tracker uses (+0 and -0 compare equal, as in C; a zero maximum's sign cannot change
     a result: s - (+-0) = s, and exp(+-0) = 1);
  2. the exp and the sums: one element per clock through a ~85-stage pipeline of exact
     operators, the rows taken eight at a time and interleaved element by element, so each
     row's sequential sum sees one addition every 8 clocks (past the adder's latency) and
     stays in the CPU's order; the exps written back in place;
  3. one reciprocal per row (the sequential divider), then the scale, 8 lanes per clock.
Lanes past n in a row's last beat are not written (byte-enabled writes).

The command (SOFTMAX_COMMAND) is the firmware's descriptor layout. Contract with the integrator
as for the other resident clients (one-cycle reads, writes on sp_wr_req & sp_wr_gnt with the
payload held). error bits: 0 geometry, 1 bounds, 6 protected output; a failure stays busy until
cancel.
"""

from migen import *
from migen.genlib.fifo import SyncFIFO

from litex.gen import LiteXModule

from litex.soc.interconnect import stream

from gateware.units.fp32 import CANONICAL_NAN, FP32Add, FP32DivSqrt, FP32Fma, FP32Mul

# Constants ----------------------------------------------------------------------------------------

SOFTMAX_COMMAND = [("src", 24), ("n", 16), ("rows", 16), ("stride", 16),
                   ("protect_start", 24), ("protect_end", 24)]
NMAX     = 4096          # positions per row (the attention stream's t_max)
ROWS_MAX = 256           # rows per command (query heads)
GROUP    = 8             # rows interleaved in pass 2

F32_ONE   = 0x3F800000
F32_HALF  = 0x3F000000
F32_M80   = 0xC2A00000   # -80.0f
F32_M1E30 = 0xF149F2CA   # -1e30f: the maximum's initial value
LOG2E     = 0x3FB8AA3B   # 0x1.715476p+0
LN2_HI    = 0x3F317200   # 0x1.62e400p-1
LN2_LO    = 0x35BFBE8E   # 0x1.7f7d1cp-20
POLY      = (0x3AB60B61, 0x3C088889, 0x3D2AAAAB, 0x3E2AAAAB, 0x3F000000, 0x3F800000, 0x3F800000)
#            1/720       1/120       1/24        1/6         1/2         1           1

def _delay(module, value, cycles, name, reset_less=True):
    prev = value
    for i in range(cycles):
        r = Signal.like(value, name=f"{name}{i}", reset_less=reset_less)
        module.sync += r.eq(prev)
        prev = r
    return prev

def _key(module, x):
    """The argmax tracker's ordering: a monotonic 32-bit key (+-0 share one key)."""
    k = Signal(32)
    module.comb += k.eq(Mux(x[:31] == 0, 0x80000000, Mux(x[31], ~x, x | 0x80000000)))
    return k

def _nan(x):
    return (x[23:31] == 0xFF) & (x[:23] != 0)

# Canonical Exp Pipeline ---------------------------------------------------------------------------

class SoftmaxExp(LiteXModule):
    """e = trixium_softmax_exp(x - mx), one element per clock, LATENCY clocks later (no reset:
    the caller tags validity). Every stage is an exact operator in the firmware's order."""
    LATENCY = (FP32Add.LATENCY        # d = x - mx
               + FP32Mul.LATENCY      # z = d*log2e
               + FP32Add.LATENCY      # t = z +- 0.5
               + 1                    # n = (int)t, fn, 2^n
               + FP32Mul.LATENCY      # fn*ln2_hi, fn*ln2_lo
               + FP32Add.LATENCY      # d - fn*ln2_hi
               + FP32Add.LATENCY      # ... - fn*ln2_lo
               + 6*FP32Fma.LATENCY    # the polynomial
               + FP32Mul.LATENCY      # p*2^n
               + 1)                   # the special cases

    def __init__(self):
        self.x, self.mx, self.e = Signal(32), Signal(32), Signal(32)

        # # #

        LA, LM, LF = FP32Add.LATENCY, FP32Mul.LATENCY, FP32Fma.LATENCY
        # d = x - mx.
        self.sub = sub = FP32Add()
        self.comb += [sub.a.eq(self.x), sub.b.eq(self.mx ^ 0x80000000)]
        d = sub.r
        # z = d*log2e.
        self.mz = mz = FP32Mul()
        self.comb += [mz.a.eq(d), mz.b.eq(LOG2E)]
        d1 = _delay(self, d, LM, "d1")
        z = mz.r
        # t = z + (z >= 0 ? 0.5 : -0.5): -0 >= 0 holds; a NaN z takes -0.5 (NaN either way).
        self.ah = ah = FP32Add()
        self.comb += [ah.a.eq(z), ah.b.eq(Mux(z[31] & (z[:31] != 0), F32_HALF | 0x80000000, F32_HALF))]
        d2 = _delay(self, d1, LA, "d2")
        t = ah.r
        # n = (int)t (toward zero), fn = (float)n, scale = 2^n. |t| < 128 for every bounded
        # argument (n in [-116, 24]); the exponent is clamped, the result past that is not the
        # firmware's either (its 2^n overflows).
        texp = t[23:31]
        sh = Signal(3)
        big = Signal()
        self.comb += [
            big.eq(texp >= 127),
            sh.eq(Mux(texp > 134, 7, (texp - 127)[:3])),
        ]
        tman = Cat(t[:23], C(1, 1))                         # 24 bits, the hidden one
        mag = Signal(8)
        # |n| = tman >> (23 - sh): the sh + 1 bits from the hidden one down (constant slices).
        self.comb += mag.eq(Mux(big, Array([Cat(tman[23 - k:24], C(0, 8))[:8] for k in range(8)])[sh], 0))
        nneg = Signal()
        self.comb += nneg.eq(t[31] & big)
        # fn from |n| (<= 128): a leading-one count on 8 bits.
        lz = Signal(4)
        self.comb += lz.eq(8)
        for bit in range(8):
            self.comb += If(mag[bit], lz.eq(7 - bit))
        fnman = Signal(8)
        self.comb += fnman.eq(Array([(mag << k)[:8] for k in range(8)] + [C(0, 8)])[lz])  # the leading one at bit 7
        fn = Signal(32)
        self.comb += If(mag == 0, fn.eq(0)).Else(
            fn.eq(Cat(C(0, 16), fnman[:7], (127 + 7) - lz, nneg)))
        nval = Signal((9, True))
        self.comb += nval.eq(Mux(nneg, -mag, mag))
        s3_fn, s3_scale, s3_d = Signal(32), Signal(32), Signal(32)
        self.sync += [s3_fn.eq(fn), s3_scale.eq(Cat(C(0, 23), (nval + 127)[:8], C(0, 1))),
                      s3_d.eq(d2)]
        # m1 = fn*ln2_hi, m2 = fn*ln2_lo.
        self.m1, self.m2 = m1, m2 = FP32Mul(), FP32Mul()
        self.comb += [m1.a.eq(s3_fn), m1.b.eq(LN2_HI), m2.a.eq(s3_fn), m2.b.eq(LN2_LO)]
        d4 = _delay(self, s3_d, LM, "d4")
        scale4 = _delay(self, s3_scale, LM, "scale4")
        # s1 = d - m1.
        self.a1 = a1 = FP32Add()
        self.comb += [a1.a.eq(d4), a1.b.eq(m1.r ^ 0x80000000)]
        m2_5 = _delay(self, m2.r, LA, "m2d")
        d5 = _delay(self, d4, LA, "d5")
        scale5 = _delay(self, scale4, LA, "scale5")
        # r = s1 - m2.
        self.a2 = a2 = FP32Add()
        self.comb += [a2.a.eq(a1.r), a2.b.eq(m2_5 ^ 0x80000000)]
        r = a2.r
        d6 = _delay(self, d5, LA, "d6")
        scale6 = _delay(self, scale5, LA, "scale6")
        # The polynomial: p = c0; p = fma(p, r, c_k).
        p = C(POLY[0], 32)
        rk, dk, sk = r, d6, scale6
        self.fmas = []
        for k, c in enumerate(POLY[1:]):
            f = FP32Fma()
            self.fmas.append(f)
            setattr(self.submodules, f"fma{k}", f)
            self.comb += [f.a.eq(p), f.b.eq(rk), f.c.eq(c)]
            p = f.r
            rk = _delay(self, rk, LF, f"r{k}_")
            dk = _delay(self, dk, LF, f"d{k}_")
            sk = _delay(self, sk, LF, f"s{k}_")
        # e = p*2^n.
        self.ms = ms = FP32Mul()
        self.comb += [ms.a.eq(p), ms.b.eq(sk)]
        dl = _delay(self, dk, LM, "dl")
        # Special cases on the argument: a NaN through (canonical), +0 below -80.
        dkey, m80key = _key(self, dl), _key(self, C(F32_M80, 32))
        self.sync += If(_nan(dl), self.e.eq(CANONICAL_NAN)
                     ).Elif(dkey < m80key, self.e.eq(0)
                     ).Else(self.e.eq(ms.r))

# Softmax Scratchpad -------------------------------------------------------------------------------

class SoftmaxScratchpad(LiteXModule):
    """Resident attention softmax (see the module docstring)."""
    def __init__(self, scratchpad_size=256*1024):
        if scratchpad_size < 32 or scratchpad_size > 256*1024 or scratchpad_size % 32:
            raise ValueError("scratchpad_size must be aligned within the 13-bit client "
                             "address space")
        # Command / status.
        self.cmd = stream.Endpoint(SOFTMAX_COMMAND)
        self.cancel, self.busy, self.done = Signal(), Signal(), Signal()
        self.error = Signal(8)
        self.read_beats, self.output_beats = Signal(32), Signal(32)

        # Scratchpad client ports.
        self.sp_rd_req, self.sp_wr_req, self.sp_wr_gnt = Signal(), Signal(), Signal()
        self.sp_rd_adr, self.sp_wr_adr = Signal(13), Signal(13)
        self.sp_rd_dat, self.sp_wr_dat = Signal(256), Signal(256)
        self.sp_wr_we = Signal(8)

        # # #

        LA, LM = FP32Add.LATENCY, FP32Mul.LATENCY
        LE = SoftmaxExp.LATENCY

        # Command capture and validation (reset-less: loaded on accept, read from VALIDATE on).
        cap = {name: Signal(width, name=f"cap_{name}", reset_less=True)
               for name, width in SOFTMAX_COMMAND}
        nbeats = Signal(13)
        stride_b = Signal(16)
        span = Signal(64)
        end = Signal(64, reset_less=True)
        geometry_bad, bounds_bad, protected_bad = (Signal(reset_less=True) for _ in range(3))
        self.comb += [
            nbeats.eq((cap["n"] + 7)[3:16]),
            stride_b.eq(Mux(cap["stride"] == 0, nbeats, cap["stride"])),
            span.eq(((cap["rows"] - 1)*stride_b << 5) + (nbeats << 5)),
        ]
        self.sync += [
            end.eq(cap["src"] + span),
            geometry_bad.eq((cap["n"] == 0) | (cap["n"] > NMAX) | (cap["rows"] == 0) |
                            (cap["rows"] > ROWS_MAX) | (cap["src"][:5] != 0) |
                            (stride_b < nbeats) | (cap["protect_start"] > cap["protect_end"])),
            bounds_bad.eq((end > scratchpad_size) | (cap["protect_end"] > scratchpad_size)),
            protected_bad.eq((cap["protect_start"] != cap["protect_end"]) &
                             (cap["src"] < cap["protect_end"]) & (cap["protect_start"] < end)),
        ]
        last_mask = Signal(8)                      # lanes of a row's last beat
        self.comb += last_mask.eq(Array([C((1 << k) - 1 if k else 0xFF, 8) for k in range(8)])[cap["n"][:3]])
        def beat_mask(b):
            return Mux(b == nbeats - 1, last_mask, C(0xFF, 8))

        # Per-row scalars.
        mxmem, summem, invmem = (Memory(32, ROWS_MAX, name=f"softmax_{n}") for n in ("mx", "sum", "inv"))
        mx_w, sum_w, inv_w = (m.get_port(write_capable=True) for m in (mxmem, summem, invmem))
        mx_r, sum_r, inv_r = (m.get_port() for m in (mxmem, summem, invmem))
        self.specials += mxmem, summem, invmem, mx_w, sum_w, inv_w, mx_r, sum_r, inv_r

        # Output FIFO with credits for the beats in flight: data, beat address, lane mask.
        FIFO_DEPTH = 32
        self.submodules.ofifo = ofifo = ResetInserter()(SyncFIFO(width=256 + 13 + 8, depth=FIFO_DEPTH))
        inflight = Signal(max=FIFO_DEPTH + 1)
        wr_now = Signal()
        wr_adr, wr_dat, wr_we = Signal(13), Signal(256), Signal(8)
        self.comb += [
            ofifo.we.eq(wr_now),
            ofifo.din.eq(Cat(wr_dat, wr_adr, wr_we)),
            self.sp_wr_req.eq(ofifo.readable & ~self.cancel),
            self.sp_wr_dat.eq(ofifo.dout[:256]),
            self.sp_wr_adr.eq(ofifo.dout[256:269]),
            self.sp_wr_we.eq(ofifo.dout[269:277]),
            ofifo.re.eq(self.sp_wr_req & self.sp_wr_gnt),
        ]
        self.sync += If(ofifo.re, self.output_beats.eq(self.output_beats + 1))

        # Control ----------------------------------------------------------------------------------
        states = ["IDLE", "SPAN", "CHECK", "VALIDATE", "P1", "P1_END", "P2_MX", "P2_RUN",
                  "P2_DRAIN", "P2_SUMS", "DIV", "P3", "FINISH", "FAILED"]
        S = {name: i for i, name in enumerate(states)}
        state = Signal(max=len(states))
        clear = Signal()
        self.comb += [
            clear.eq(self.cancel | (state == S["IDLE"])),
            ofifo.reset.eq(clear),
            self.cmd.ready.eq((state == S["IDLE"]) & ~self.cancel),
            self.busy.eq(state != S["IDLE"]),
        ]
        row = Signal(16)                           # pass 1 / div / pass 3 row
        row_base = Signal(13, reset_less=True)     # its first beat
        beat = Signal(13)
        issue = Signal()
        rd_pending = Signal()
        rd_kind = Signal(2, reset_less=True)       # 1 pass 1, 2 pass 2 (fill), 3 pass 3
        rd_beat = Signal(13, reset_less=True)
        rd_g = Signal(3, reset_less=True)
        rd_half = Signal(reset_less=True)
        self.sync += rd_pending.eq(issue & ~clear)
        self.sync += If(issue, self.read_beats.eq(self.read_beats + 1))

        # Pass 1: the maximum, one beat per clock --------------------------------------------------
        issue1 = Signal()
        self.comb += issue1.eq((state == S["P1"]) & (beat < nbeats) & ~self.cancel)
        # The read data through one register first: the scratchpad's outputs fan out across the
        # die and the wire into the key compares was a Ti375 near-critical group.
        p1_valid, p1_dat = Signal(), Signal(256, reset_less=True)
        self.sync += [p1_valid.eq(rd_pending & (rd_kind == 1) & ~clear), p1_dat.eq(self.sp_rd_dat)]
        p1_mask = _delay(self, beat_mask(beat), 2, "p1_mask")
        # Tree of keys: lane -> 4 -> 2 -> 1 -> the running maximum.
        mx_run = Signal(32, reset_less=True)
        mx_key = _key(self, mx_run)
        ks, xs = [], []
        for j in range(8):
            x = p1_dat[32*j:32*j + 32]
            k = _key(self, x)
            kk = Signal(32, name=f"k{j}")
            self.comb += kk.eq(Mux(p1_mask[j] & ~_nan(x), k, 0))
            ks.append(kk); xs.append(x)
        def level(ks, xs, name):
            nk, nx = [], []
            for i in range(0, len(ks), 2):
                k, x = Signal(32, name=f"{name}k{i}", reset_less=True), Signal(32, name=f"{name}x{i}", reset_less=True)
                self.sync += If(ks[i] > ks[i + 1], k.eq(ks[i]), x.eq(xs[i])).Else(k.eq(ks[i + 1]), x.eq(xs[i + 1]))
                nk.append(k); nx.append(x)
            return nk, nx
        k4, x4 = level(ks, xs, "t1")
        k2, x2 = level(k4, x4, "t2")
        k1, x1 = level(k2, x2, "t3")
        t_valid = _delay(self, p1_valid, 3, "t_valid", reset_less=False)
        p1_inflight = Signal(3)
        self.sync += If(clear, p1_inflight.eq(0)).Else(p1_inflight.eq(p1_inflight + p1_valid - t_valid))
        p1_final = Signal()                        # P1_END: every beat of the row is in mx_run
        self.comb += p1_final.eq((state == S["P1_END"]) & (p1_inflight == 0) & ~t_valid & ~p1_valid &
                                  ~(rd_pending & (rd_kind == 1)))
        self.sync += If((state == S["VALIDATE"]) | p1_final, mx_run.eq(F32_M1E30)
                     ).Elif(t_valid & (k1[0] > mx_key), mx_run.eq(x1[0]))
        self.comb += [mx_w.adr.eq(row), mx_w.dat_w.eq(mx_run), mx_w.we.eq(p1_final)]

        # Pass 2: the exp and the sums, eight rows interleaved -----------------------------------
        group = Signal(16)                         # first row of the group
        grows = Signal(4)                          # rows in the group (1..8)
        self.comb += grows.eq(Mux(cap["rows"] - group >= GROUP, GROUP, (cap["rows"] - group)[:4]))
        gbase = [Signal(13, name=f"gbase{g}", reset_less=True) for g in range(GROUP)]  # row bases
        mx_g = [Signal(32, name=f"mxg{g}", reset_less=True) for g in range(GROUP)]
        sum_g = [Signal(32, name=f"sumg{g}", reset_less=True) for g in range(GROUP)]
        # The group's maxima: 8 reads of mxmem (P2_MX), one row base add per read.
        mxl = Signal(4)
        self.comb += mx_r.adr.eq(group + mxl)
        mxl_d = _delay(self, mxl, 1, "mxl_d")
        mxl_v = _delay(self, (state == S["P2_MX"]) & (mxl < GROUP), 1, "mxl_v", reset_less=False)
        self.sync += [If(mxl_v, Array(mx_g)[mxl_d[:3]].eq(mx_r.dat_r))]
        # Input beats: two halves of 8 (the beat being streamed, the one being filled).
        inbuf = Memory(256, 2*GROUP, name="softmax_inbuf")
        in_w, in_r = inbuf.get_port(write_capable=True), inbuf.get_port()
        self.specials += inbuf, in_w, in_r
        self.comb += [in_w.we.eq(rd_pending & (rd_kind == 2)), in_w.adr.eq(Cat(rd_g, rd_half)),
                      in_w.dat_w.eq(self.sp_rd_dat)]
        # Fill: beat `fbeat` of rows group..group+grows-1 into half `fhalf`.
        fbeat = Signal(13)
        fg = Signal(4)
        fhalf = Signal()
        filled = Signal()                          # the half for `sbeat` is complete
        can_fill = Signal()
        self.comb += can_fill.eq((ofifo.level + inflight) < FIFO_DEPTH - GROUP)
        issue2 = Signal()
        # Slots: 64 per beat (lane l of row g), the input half read one clock ahead.
        sbeat = Signal(13)
        slot = Signal(7)                           # l = slot[3:6], g = slot[:3]
        shalf = Signal()
        running = Signal()                         # slots of `sbeat` in progress
        fill_done = Signal()
        fill_ok = Signal()                         # a fill step may happen (beat sbeat + 1 at most)
        fill_step = Signal()                       # it happens: a read, or a skipped row
        self.comb += [
            fill_done.eq(fbeat > sbeat),           # the beat after the one streaming is in
            fill_ok.eq((state == S["P2_RUN"]) & ~self.cancel & (fbeat < nbeats) & (fbeat <= sbeat + 1)),
            issue2.eq(fill_ok & (fg < grows) & can_fill),
            fill_step.eq(issue2 | (fill_ok & (fg >= grows))),
        ]
        # Slot stream.
        slot_v = Signal()
        self.comb += slot_v.eq((state == S["P2_RUN"]) & running & ~self.cancel)
        sl, sg = slot[3:6], slot[:3]
        self.comb += in_r.adr.eq(Cat(sg, shalf))
        # One clock later: the operand and its tags.
        v1 = Signal()
        g1, l1, b1 = Signal(3, reset_less=True), Signal(3, reset_less=True), Signal(13, reset_less=True)
        valid1 = Signal(reset_less=True)
        self.sync += [v1.eq(slot_v & ~clear), g1.eq(sg), l1.eq(sl), b1.eq(sbeat),
                      valid1.eq((sg < grows) & (Cat(sl, sbeat) < cap["n"]))]
        x1_in = Signal(32)
        self.comb += x1_in.eq(Array([in_r.dat_r[32*j:32*j + 32] for j in range(8)])[l1])
        self.exp = ex = SoftmaxExp()
        self.comb += [ex.x.eq(x1_in), ex.mx.eq(Array(mx_g)[g1])]
        ev = _delay(self, v1 & valid1, LE, "ev", reset_less=False)
        eg, el, eb = _delay(self, g1, LE, "eg"), _delay(self, l1, LE, "el"), _delay(self, b1, LE, "eb")
        e = ex.e
        # Output beats per row, pushed at lane 7 (or the row's last valid lane).
        obeat = [Signal(256, name=f"obeat{g}", reset_less=True) for g in range(GROUP)]
        e_last = Signal()                          # the beat's last valid lane
        emask = Signal(8)
        next_valid = Signal()                      # lane el + 1 is valid
        self.comb += [emask.eq(beat_mask(eb)),
                      next_valid.eq(Array([emask[j + 1] if j < 7 else C(0, 1) for j in range(8)])[el]),
                      e_last.eq(~next_valid)]
        for g in range(GROUP):
            for j in range(8):
                self.sync += If(ev & (eg == g) & (el == j), obeat[g][32*j:32*j + 32].eq(e))
        push2 = Signal()
        self.comb += push2.eq(ev & e_last)
        # Sums: one adder, row g's add every 8 clocks.
        self.sadd = sadd = FP32Add()
        self.comb += [sadd.a.eq(Array(sum_g)[eg]), sadd.b.eq(e)]
        sv = _delay(self, ev, LA, "sv", reset_less=False)
        sg_d = _delay(self, eg, LA, "sg_d")
        self.sync += If(state == S["P2_MX"], *[s.eq(0) for s in sum_g]
                     ).Elif(sv, Array(sum_g)[sg_d].eq(sadd.r))
        p2_inflight = Signal(8)
        self.sync += If(clear, p2_inflight.eq(0)).Else(p2_inflight.eq(p2_inflight + (v1 & valid1) - sv))
        # Group sums to memory (P2_SUMS: 8 writes).
        suml = Signal(4)
        self.comb += [sum_w.adr.eq(group + suml), sum_w.dat_w.eq(Array(sum_g)[suml[:3]]),
                      sum_w.we.eq((state == S["P2_SUMS"]) & (suml < grows))]

        # Reciprocals -----------------------------------------------------------------------------
        self.ds = ds = FP32DivSqrt()
        div_st = Signal(2)                         # 0 read sum, 1 start, 2 wait, 3 write
        self.comb += [sum_r.adr.eq(row), ds.a.eq(F32_ONE), ds.b.eq(sum_r.dat_r), ds.op.eq(0),
                      ds.start.eq((state == S["DIV"]) & (div_st == 1)),
                      inv_w.adr.eq(row), inv_w.dat_w.eq(ds.r),
                      inv_w.we.eq((state == S["DIV"]) & (div_st == 3))]

        # Pass 3: the scale, one beat per clock ---------------------------------------------------
        inv_row = Signal(32, reset_less=True)
        self.comb += inv_r.adr.eq(row)
        issue3 = Signal()
        can_p3 = Signal()
        p3_hold = Signal(2)                        # the row's reciprocal: read, then registered
        self.comb += [can_p3.eq((ofifo.level + inflight) < FIFO_DEPTH),
                      issue3.eq((state == S["P3"]) & (p3_hold == 0) & (beat < nbeats) & can_p3 & ~self.cancel)]
        p3_valid = Signal()
        self.comb += p3_valid.eq(rd_pending & (rd_kind == 3))
        self.muls = muls = [FP32Mul() for _ in range(8)]
        self.submodules += muls
        for j in range(8):
            self.comb += [muls[j].a.eq(self.sp_rd_dat[32*j:32*j + 32]), muls[j].b.eq(inv_row)]
        p3v = _delay(self, p3_valid, LM, "p3v", reset_less=False)
        p3b = _delay(self, rd_beat, LM, "p3b")
        p3m = _delay(self, beat_mask(rd_beat), LM, "p3m")
        p3_inflight = Signal(4)
        self.sync += If(clear, p3_inflight.eq(0)).Else(p3_inflight.eq(p3_inflight + p3_valid - p3v))
        y3 = Cat(*[m.r for m in muls])

        # Scratchpad reads, output pushes ---------------------------------------------------------
        self.comb += [
            issue.eq(issue1 | issue2 | issue3),
            self.sp_rd_req.eq(issue),
            If(issue2,
                self.sp_rd_adr.eq(Array(gbase)[fg[:3]] + fbeat),
            ).Else(
                self.sp_rd_adr.eq(row_base + beat),
            ),
        ]
        self.sync += [
            rd_kind.eq(Mux(issue2, 2, Mux(issue3, 3, 1))),
            rd_beat.eq(Mux(issue2, fbeat, beat)),
            rd_g.eq(fg[:3]), rd_half.eq(fhalf),
        ]
        obeat_now = Signal(256)                    # the pushing row's beat with this clock's lane
        self.comb += obeat_now.eq(Cat(*[Mux(el == j, e, Array(obeat)[eg][32*j:32*j + 32]) for j in range(8)]))
        self.comb += [
            wr_now.eq((push2 | p3v) & ~clear),
            If(push2,
                wr_adr.eq(Array(gbase)[eg] + eb), wr_dat.eq(obeat_now), wr_we.eq(emask),
            ).Else(
                wr_adr.eq(row_base + p3b), wr_dat.eq(y3), wr_we.eq(p3m),
            ),
        ]
        # Beats issued and not yet pushed: pass-2 fills count per beat read (the push comes
        # per output beat, the same number), pass 3 per read.
        self.sync += If(clear, inflight.eq(0)).Else(
            inflight.eq(inflight + issue2 + issue3 - wr_now))
        drained = Signal()
        self.comb += drained.eq((inflight == 0) & ~ofifo.readable & ~rd_pending)

        # Sequencing -------------------------------------------------------------------------------
        self.sync += [
            self.done.eq(0),
            If(self.cancel, state.eq(S["IDLE"]), self.error.eq(0)).Else(
            Case(state, {
                S["IDLE"]: If(self.cmd.valid & self.cmd.ready,
                    *[v.eq(getattr(self.cmd, k)) for k, v in cap.items()],
                    self.error.eq(0), self.read_beats.eq(0), self.output_beats.eq(0),
                    state.eq(S["SPAN"])),
                S["SPAN"]:  state.eq(S["CHECK"]),
                S["CHECK"]: state.eq(S["VALIDATE"]),
                S["VALIDATE"]: If(geometry_bad | bounds_bad | protected_bad,
                    self.error.eq(Cat(geometry_bad, bounds_bad, C(0, 4), protected_bad)),
                    state.eq(S["FAILED"]),
                ).Else(
                    row.eq(0), row_base.eq(cap["src"] >> 5), beat.eq(0), state.eq(S["P1"]),
                ),
                # Pass 1: row by row.
                S["P1"]: If(issue1, beat.eq(beat + 1)).Elif(beat == nbeats, state.eq(S["P1_END"])),
                S["P1_END"]: If(p1_final,                          # mx_run final: written this clock
                    If(row + 1 == cap["rows"],
                        group.eq(0), state.eq(S["P2_MX"]), mxl.eq(0),
                        *[gbase[g].eq((cap["src"] >> 5) + g*stride_b) for g in range(GROUP)],
                    ).Else(
                        row.eq(row + 1), row_base.eq(row_base + stride_b), beat.eq(0), state.eq(S["P1"]),
                    )),
                # Pass 2, per group: the maxima, the beats, the drain, the sums.
                S["P2_MX"]: If(mxl < GROUP + 1,
                    mxl.eq(mxl + 1),
                ).Else(
                    fbeat.eq(0), fg.eq(0), fhalf.eq(0), sbeat.eq(0), slot.eq(0), shalf.eq(0),
                    running.eq(0), state.eq(S["P2_RUN"]),
                ),
                S["P2_RUN"]: [
                    # Fill the half for beat fbeat (rows beyond grows skip their read).
                    If(fill_step,
                        fg.eq(fg + 1),
                        If(fg == GROUP - 1, fg.eq(0), fbeat.eq(fbeat + 1), fhalf.eq(~fhalf)),
                    ),
                    # Stream the slots of sbeat once its half is in (one clock after the last
                    # read of the fill lands: fbeat > sbeat and no read pending into it).
                    If(~running,
                        If((fbeat > sbeat) & ~(rd_pending & (rd_kind == 2) & (rd_beat == sbeat)) & (sbeat < nbeats),
                            running.eq(1), slot.eq(0)),
                    ).Else(
                        slot.eq(slot + 1),
                        If(slot == 63,
                            running.eq(0), sbeat.eq(sbeat + 1), shalf.eq(~shalf),
                        ),
                    ),
                    If(~running & (sbeat == nbeats), state.eq(S["P2_DRAIN"])),
                ],
                S["P2_DRAIN"]: If((p2_inflight == 0) & ~v1, suml.eq(0), state.eq(S["P2_SUMS"])),
                S["P2_SUMS"]: If(suml < GROUP,
                    suml.eq(suml + 1),
                ).Elif(group + GROUP < cap["rows"],
                    group.eq(group + GROUP), mxl.eq(0), state.eq(S["P2_MX"]),
                    *[gbase[g].eq(gbase[g] + (stride_b << 3)) for g in range(GROUP)],
                ).Elif(drained,
                    row.eq(0), div_st.eq(0), state.eq(S["DIV"]),
                ),
                # Reciprocals, row by row.
                S["DIV"]: Case(div_st, {
                    0: div_st.eq(1),                                   # sum_r.dat_r valid next clock
                    1: div_st.eq(2),                                   # start pulses this clock
                    2: If(ds.done, div_st.eq(3)),
                    3: [div_st.eq(0),
                        If(row + 1 == cap["rows"],
                            row.eq(0), row_base.eq(cap["src"] >> 5), beat.eq(0), p3_hold.eq(2),
                            state.eq(S["P3"]),
                        ).Else(row.eq(row + 1))],
                }),
                # Pass 3, row by row (inv_row from invmem: the row's first read waits a clock).
                S["P3"]: [
                    inv_row.eq(inv_r.dat_r),
                    If(p3_hold != 0, p3_hold.eq(p3_hold - 1)),
                    If(issue3, beat.eq(beat + 1)
                    ).Elif((beat == nbeats) & (p3_inflight == 0) & ~p3_valid,
                        If(row + 1 == cap["rows"],
                            state.eq(S["FINISH"]),
                        ).Else(row.eq(row + 1), row_base.eq(row_base + stride_b), beat.eq(0),
                               p3_hold.eq(2)),
                    ),
                ],
                S["FINISH"]: If(drained, self.done.eq(1), state.eq(S["IDLE"])),
            })),
        ]

# SPDX-License-Identifier: BSD-2-Clause
"""Batched matvec through the whole engine (nbatch, docs/speculative-verify-design.md step 2).

One weight stream, nb token vectors: vector b's x at xsrc + b*dstride beats (deliberately
NOT contiguous), its results at dst + b*qstride words, every vector bit-exact against the
single-vector reference. Ternary and Q8, several channel counts, uneven reader latency and
stalls, and a plain (nb=0) op on the same nbatch engine afterwards."""
import numpy as np
import pytest
from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes


class _DUT(Module):
    def __init__(self, n, payloads, sp_bytes, latency, stall, xbuf_groups, nbatch):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(n)]
        self.submodules.scratchpad = BonsaiScratchpad(size=64*1024, init=words_from_bytes(sp_bytes))
        self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=xbuf_groups,
                                                    dual=False, nbatch=nbatch)
        for i, axi in enumerate(axis):
            setattr(self.submodules, f"model{i}", BonsaiHBMChannelModel(
                axi, size=64*1024, init=words_from_bytes(payloads[i]),
                latency=latency[i], stall_rate=stall[i]))


def _run(dut, cmds, reads):
    out = []

    def driver():
        for cmd in cmds:
            for k in ("nb", "qstride", "dstride", "ilv", "qn", "rem", "supw"):
                yield getattr(dut.engine.cmd, k).eq(0)
            for k, v in cmd.items():
                yield getattr(dut.engine.cmd, k).eq(v)
            yield dut.engine.cmd.valid.eq(1)
            yield
            while not (yield dut.engine.cmd.ready):
                yield
            yield dut.engine.cmd.valid.eq(0)
            for _ in range(100000):
                if (yield dut.engine.done):
                    break
                assert not (yield dut.engine.error)
                yield
            else:
                raise AssertionError(f"engine timeout, results {(yield dut.engine.dbg_result_cnt)}")
            yield
            yield
            assert (yield dut.engine.error) == 0
        for dst, count in reads:
            vals = []
            for i in range(count):
                word = (yield dut.scratchpad.mem[(dst + 4*i)//32])
                v = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
                vals.append(v - (1 << 32) if v & (1 << 31) else v)
            out.append(np.array(vals, dtype=np.int64))

    run_simulation(dut, [driver()])
    return out


def _case(kind, n, rows_per_chan, cols, nb, latency, stall, seed, xbuf_groups=32):
    rng = np.random.default_rng(seed)
    M = n*rows_per_chan
    ternary = kind == 0
    lk = layout.KIND_TERNARY if ternary else layout.KIND_Q8
    G = cols//(128 if ternary else 32)
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    vals = (rng.integers(0, 3, (M, cols), dtype=np.uint8) if ternary
            else rng.integers(-127, 128, (M, cols), dtype=np.int8))
    xs = [rng.integers(-127, 128, cols, dtype=np.int8) for _ in range(nb)]
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B, gs_max=bx.GS_MAX if ternary else bx.GS_MAX_Q8)
    fn = bx.matvec_ternary if ternary else bx.matvec_q8
    exp = [fn(scales, vals, x, B, R) for x in xs]
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, lk, M, cols, scales=scales, values=vals)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n, c)).tobytes() for c in range(n)]
    # x vectors at a stride with a gap (cols/32 + 3 beats), results at a stride of M + 13 words.
    xstride = cols//32 + 3
    sp = bytearray(0x4000)
    for b, x in enumerate(xs):
        sp[32*xstride*b:32*xstride*b + cols] = x.tobytes()
    qstride, dst = M + 13, 0x4000
    dut = _DUT(n, payloads, bytes(sp), latency, stall, xbuf_groups, nbatch=4)
    rb = layout.row_beats(cols, lk)
    base = dict(kind=kind, rows=rows_per_chan, cols=cols, wbase=0, wlen=rows_per_chan*rb,
                sbeats=layout.scale_beats(cols, lk), cbeats=layout.code_beats(cols, lk),
                bias=B, rshift=R, xsrc=0)
    # The batched op, then a plain op on vector 1 to a fresh destination (nb back to 0).
    cmds = [dict(base, dst=dst, nb=nb - 1, qstride=qstride, dstride=xstride)]
    reads = [(dst + 4*qstride*b, M) for b in range(nb)]
    if nb > 1:
        cmds.append(dict(base, dst=0x8000, xsrc=32*xstride))
        reads.append((0x8000, M))
    got = _run(dut, cmds, reads)
    for b in range(nb):
        assert np.array_equal(got[b], exp[b]), (kind, n, nb, b, got[b][:6], exp[b][:6])
    if nb > 1:
        assert np.array_equal(got[nb], exp[1]), "plain op after a batched one"


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("n,rows,nb,lat,stall", [
    (1, 6, 2, [8], [0]),
    (2, 5, 3, [8, 2], [0, 2]),
    (4, 4, 4, [8, 3, 5, 2], [2, 0, 1, 0]),
    (4, 24, 2, [2]*4, [0]*4),      # Dense small rows: gather pressure with nb results/row.
])
def test_engine_batched(kind, n, rows, nb, lat, stall):
    cols = 256 if kind == 0 else 160   # Q8: 5 groups -> 2 x-words per vector (rounded up).
    _case(kind, n, rows, cols, nb, lat, stall, seed=17*n + nb + kind)

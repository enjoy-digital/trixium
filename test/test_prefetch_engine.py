# SPDX-License-Identifier: BSD-2-Clause
"""The engine with the run-ahead weight prefetch: a recorded forward, then replayed ones -
bit-exact results, every weight read served from the prefetch FIFO, faster ops."""
import numpy as np
from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes
from trixium.gateware.prefetch import MODE_RECORD, MODE_REPLAY


def tensor(rng, M, cols):
    G = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    return scales, codes


class _DUT(Module):
    def __init__(self, n, payloads, sp_init):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6) for _ in range(n)]
        self.submodules.scratchpad = BonsaiScratchpad(size=64*1024, init=sp_init)
        self.submodules.engine = BonsaiMatVecEngine(self.scratchpad, axis, xbuf_groups=32, dual=False,
                                                    prefetch=dict(depth=2048, chunk=32, sched_depth=64))
        for i, axi in enumerate(axis):
            # A slow DDR: stall_rate 9/16 leaves ~0.44 beats/clock, the XCU1525's 0.46.
            setattr(self.submodules, f"model{i}", BonsaiHBMChannelModel(
                axi, size=256*1024, init=words_from_bytes(payloads[i]), latency=12, stall_rate=9))


def test_prefetch_replay_is_bit_exact_and_faster():
    rng = np.random.default_rng(21)
    n, rpc, cols = 2, 48, 512
    M = n*rpc
    specs, cmds, expected = [], [], []
    x8 = rng.integers(-127, 128, cols, dtype=np.int8)
    rb = layout.row_beats(cols, layout.KIND_TERNARY)
    payloads = [b""]*n
    for t in range(2):
        scales, codes = tensor(rng, M, cols)
        _, e16 = bx.fp16_decompose(scales)
        B = bx.choose_scale_bias(e16)
        R = bx.choose_result_shift(scales, B)
        expected.append(bx.matvec_ternary(scales, codes, x8, B, R))
        spec = TensorSpec(f"w{t}", layout.ROLE_ATTN_Q, t, layout.KIND_TERNARY, M, cols, scales=scales, values=codes)
        wbase = len(payloads[0])
        for c in range(n):
            row0, nrows = layout.shard_rows(M, n, c)
            payloads[c] += encode_rows(spec, row0, nrows).tobytes()
        cmds.append(dict(kind=0, rows=rpc, cols=cols, wbase=wbase, wlen=rpc*rb,
                         sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
                         cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
                         bias=B, rshift=R, xsrc=0, dst=0x4000 + 0x1000*t))
    dut = _DUT(n, payloads, words_from_bytes(x8.tobytes()))
    eng = dut.engine
    log = []

    def op(cmd, t):
        for k, v in cmd.items():
            yield getattr(eng.cmd, k).eq(v)
        yield eng.cmd.valid.eq(1)
        yield
        while not (yield eng.cmd.ready):
            yield
        yield eng.cmd.valid.eq(0)
        n0 = 0
        while not (yield eng.done):
            n0 += 1
            assert n0 < 50000, "engine timeout"
            yield
        yield
        yield
        assert (yield eng.error) == 0
        got = []
        dst = cmd["dst"]
        for i in range(M):
            word = yield dut.scratchpad.mem[(dst + 4*i)//32]
            v = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
            got.append(v - (1 << 32) if v & (1 << 31) else v)
        assert np.array_equal(np.array(got), expected[t]), t
        return n0

    def driver():
        for mode in (MODE_RECORD, MODE_REPLAY, MODE_REPLAY):
            yield eng.pf_mode.eq(mode)
            yield eng.pf_start.eq(1)
            yield
            yield eng.pf_start.eq(0)
            cyc = []
            for t, cmd in enumerate(cmds):
                for _ in range(600):            # the units' work between the lists
                    yield
                cyc.append((yield from op(cmd, t)))
            log.append((mode, cyc))
        u = eng.prefetch_units
        h, m, r = [], [], []
        for p in u:
            h.append((yield p.hits)); m.append((yield p.misses)); r.append((yield p.recorded))
        log.append(("hits", h, m, r))

    run_simulation(dut, [driver()])
    print(log)
    (m0, rec), (m1, rep1), (m2, rep2), (_, hits, misses, recorded) = log
    assert recorded == [2]*n and misses == [0]*n and hits == [4]*n
    assert sum(rep1) < 0.75*sum(rec) and sum(rep2) < 0.75*sum(rec)

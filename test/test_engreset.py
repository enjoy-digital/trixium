"""The engine soft reset: a wedged op is recovered without touching the HBM ports' state.

A ternary matvec runs; the HBM model of channel 1 then freezes mid-stream (a port that
stops answering - what a fetch of a row that does not exist looks like), the engine sits
busy with a half-delivered op, the reset domain is pulsed (the `engreset` CSR path), the
port answers again later, the shims drain the late beats, and the same matvec then runs
again with bit-exact results. The scratchpad and the loader's ports are untouched.
"""

import numpy as np
from migen import *
from migen.sim import run_simulation

from litex.gen import *
from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import TensorSpec, encode_rows
from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.loader import BonsaiHBMLoader
from trixium.gateware.engreset import BonsaiEngineResetDomain
from trixium.gateware.hbm_model import BonsaiHBMChannelModel, words_from_bytes


class _DUT(LiteXModule):
    def __init__(self, n_channels, chan_payloads, sp_init):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.scratchpad = BonsaiScratchpad(size=64*1024, init=sp_init)
        self.engrst = BonsaiEngineResetDomain(axis, hold_cycles=16, stuck_cycles=4000)
        self.engine = ClockDomainsRenamer("eng")(
            BonsaiMatVecEngine(self.scratchpad, self.engrst.masters, dual=False))
        self.loader = ClockDomainsRenamer("eng")(BonsaiHBMLoader(self.engrst.masters))
        self.comb += [
            self.engine.store_cmd.connect(self.loader.store_cmd),
            self.engine.store_data.connect(self.loader.store_data),
            self.engine.store_busy.eq(self.loader.store_busy),
        ]
        self.models = []
        for i, axi in enumerate(axis):
            m = BonsaiHBMChannelModel(axi, size=64*1024, init=words_from_bytes(chan_payloads[i]),
                                      latency=6)
            setattr(self, f"model{i}", m)
            self.models.append(m)


def _submit(dut, cmd):
    for k, v in cmd.items():
        yield getattr(dut.engine.cmd, k).eq(v)
    yield dut.engine.cmd.valid.eq(1)
    yield
    while not (yield dut.engine.cmd.ready):
        yield
    yield dut.engine.cmd.valid.eq(0)


def _wait_done(dut, timeout=20000):
    for _ in range(timeout):
        if (yield dut.engine.done):
            return True
        yield
    return False


def _results(dut, dst, n):
    out = []
    for i in range(n):
        word = (yield dut.scratchpad.mem[(dst + 4*i)//32])
        v = (word >> (32*(((dst + 4*i)//4) % 8))) & 0xFFFFFFFF
        out.append(v - (1 << 32) if v & (1 << 31) else v)
    return np.array(out, dtype=np.int64)


def test_engreset_recovers_a_wedged_matvec():
    n_channels, rows_per_chan, cols = 2, 8, 256
    rng    = np.random.default_rng(7)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    expected = bx.matvec_ternary(scales, codes, x8, B, R)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = []
    for c in range(n_channels):
        row0, nrows = layout.shard_rows(M, n_channels, c)
        payloads.append(encode_rows(spec, row0, nrows).tobytes())
    dst = 0x4000
    dut = _DUT(n_channels, payloads, words_from_bytes(x8.tobytes()))
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=dst)
    out = {}

    def gen():
        # 1. A clean run.
        yield from _submit(dut, cmd)
        assert (yield from _wait_done(dut))
        yield; yield
        out["first"] = (yield from _results(dut, dst, M))
        # Scribble the destination so the second run has to write it again.
        for i in range(M):
            yield dut.scratchpad.mem[(dst + 4*i)//32].eq(0)
        # 2. Channel 1's port stops answering two cycles into the stream: the op wedges.
        yield from _submit(dut, cmd)
        for _ in range(12):
            yield
        yield dut.models[1].freeze.eq(1)
        for _ in range(400):
            yield
        out["wedged_busy"] = (yield dut.engine.busy)
        out["wedged_done"] = (yield dut.engine.done)
        # 3. The soft reset: the engine drops the op at once...
        yield dut.engrst.trigger.eq(1)
        yield
        yield dut.engrst.trigger.eq(0)
        for _ in range(40):
            yield
        out["busy_after_reset"] = (yield dut.engine.busy)
        out["draining"] = (yield dut.engrst.busy)
        # ...the port answers again, the shim eats the late beats, and the drain ends.
        yield dut.models[1].freeze.eq(0)
        for _ in range(3000):
            if not (yield dut.engrst.busy):
                break
            yield
        out["drained"] = not (yield dut.engrst.busy)
        out["stuck"] = (yield dut.engrst.stuck)
        # 4. The same op again is bit-exact: no stale beat reached the pipes.
        yield from _submit(dut, cmd)
        assert (yield from _wait_done(dut))
        yield; yield
        out["second"] = (yield from _results(dut, dst, M))
        out["error"] = (yield dut.engine.error)

    run_simulation(dut, gen(), clocks={"sys": 10, "eng": 10})
    assert np.array_equal(out["first"], expected)
    assert out["wedged_busy"] == 1 and out["wedged_done"] == 0
    assert out["busy_after_reset"] == 0 and out["draining"] == 1
    assert out["drained"] and not out["stuck"]
    assert out["error"] == 0
    assert np.array_equal(out["second"], expected), "stale beats after the reset"


def test_engreset_reports_a_port_that_never_answers():
    n_channels, rows_per_chan, cols = 2, 8, 256
    rng    = np.random.default_rng(8)
    M      = n_channels*rows_per_chan
    G      = cols//layout.GROUP_TERNARY
    scales = rng.uniform(0.01, 0.08, (M, G)).astype(np.float16)
    codes  = rng.integers(0, 3, (M, cols), dtype=np.uint8)
    x8     = rng.integers(-127, 128, cols, dtype=np.int8)
    _, e16 = bx.fp16_decompose(scales)
    B = bx.choose_scale_bias(e16)
    R = bx.choose_result_shift(scales, B)
    spec = TensorSpec("w", layout.ROLE_ATTN_Q, 0, layout.KIND_TERNARY, M, cols,
                      scales=scales, values=codes)
    payloads = [encode_rows(spec, *layout.shard_rows(M, n_channels, c)).tobytes()
                for c in range(n_channels)]
    dut = _DUT(n_channels, payloads, words_from_bytes(x8.tobytes()))
    cmd = dict(kind=0, rows=rows_per_chan, cols=cols, wbase=0,
               wlen=rows_per_chan*layout.row_beats(cols, layout.KIND_TERNARY),
               sbeats=layout.scale_beats(cols, layout.KIND_TERNARY),
               cbeats=layout.code_beats(cols, layout.KIND_TERNARY),
               bias=B, rshift=R, xsrc=0, dst=0x4000)
    out = {}

    def gen():
        yield dut.models[0].freeze.eq(1)          # from the very first burst
        yield from _submit(dut, cmd)
        for _ in range(200):
            yield
        yield dut.engrst.trigger.eq(1)
        yield
        yield dut.engrst.trigger.eq(0)
        for _ in range(6000):
            yield
        out["busy"] = (yield dut.engrst.busy)
        out["stuck"] = (yield dut.engrst.stuck)

    run_simulation(dut, gen(), clocks={"sys": 10, "eng": 10})
    assert out["busy"] == 1 and out["stuck"] == 1

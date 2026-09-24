# SPDX-License-Identifier: BSD-2-Clause
"""BonsaiSiluScratchpad: table LOAD and in-scratchpad SiLU-multiply RUN, against the model.

The kernel itself is proven element-exact in test_silu_mul_rtl.py; here the questions are
the client's: every element lands in its lane, fixup lanes carry g, the bitmap has every
word written and no stale bit, tails and neighbours are untouched, in place works, write
payloads hold under stall, bad commands never touch memory, cancel recovers.
"""
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from migen import run_simulation
from migen.fhdl import verilog
from litex.gen import LiteXModule
from migen import Signal, Mux

from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.silu_sp import BonsaiSiluScratchpad, LOAD_BEATS, OP_LOAD, OP_RUN, SILU_COMMAND
from trixium.model.silu_hw import LUT_N, silu_mul_fixed, tables
from test_silu_mul_rtl import stress_vectors

SENTINEL = 0xDEADBEEF


def table_image(pair=None):
    """The LOAD layout: lut[0..4096] in beats 0..512 (lut[4096] alone in beat 512), dlut after."""
    lut, dlut = pair or tables()
    words = np.full(LOAD_BEATS*8, SENTINEL, dtype=np.uint32)
    words[:LUT_N + 1] = lut
    words[513*8:513*8 + LUT_N] = dlut
    return words


def reference(mem, cmd, pair=None):
    """Apply a RUN command to a uint32 word array; returns the fixup count."""
    lut, dlut = pair or tables()
    n, g0, m0, d0, f0 = cmd["elements"], cmd["src_g"]//4, cmd["src_m"]//4, cmd["dst"]//4, cmd["fixmap"]//4
    gates = [int(x) for x in mem[g0:g0 + n]]
    mults = [0x3F800000]*n if cmd.get("unit_m") else [int(x) for x in mem[m0:m0 + n]]
    bits = [0]*((n + 31)//32)
    a, b, sb = cmd.get("perm_a", 0), cmd.get("perm_b", 0), cmd.get("seg_beats", 0)
    def dst_index(i):
        if a <= 1:
            return i
        seg, within = divmod(i, 8*sb)
        return ((seg % a)*b + seg//a)*8*sb + within
    out = {}
    for i, (g, m) in enumerate(zip(gates, mults)):
        r, fix = silu_mul_fixed(g, m, lut, dlut, table_only=bool(cmd.get("table_only")))
        out[d0 + dst_index(i)] = g if fix else r
        bits[i//32] |= int(fix) << (i % 32)
    for k, v in out.items():
        mem[k] = v
    mem[f0:f0 + len(bits)] = bits
    return sum(bin(b).count("1") for b in bits)


def run_cmd(**changed):
    cmd = dict(op=OP_RUN, src_g=0, src_m=0, dst=0, fixmap=0, elements=0, unit_m=0, seg_beats=0,
               perm_a=0, perm_b=0, protect_start=0, protect_end=0)
    cmd.update(changed)
    return cmd


# Migen -----------------------------------------------------------------------------------------

SIZE = 64*1024


class SiluSP(LiteXModule):
    def __init__(self, size=SIZE, preload=True):
        self.sp = BonsaiScratchpad(size=size)
        self.unit = BonsaiSiluScratchpad(size, tables=tables() if preload else None)
        self.grant = Signal(reset=1)
        self.comb += [self.sp.fill.adr.eq(self.unit.sp_rd_adr), self.unit.sp_rd_dat.eq(self.sp.fill.dat_r),
                      self.unit.sp_wr_gnt.eq(self.grant & self.unit.sp_wr_req),
                      self.sp.gather.adr.eq(self.unit.sp_wr_adr), self.sp.gather.dat_w.eq(self.unit.sp_wr_dat),
                      self.sp.gather.we.eq(Mux(self.grant & self.unit.sp_wr_req, self.unit.sp_wr_we, 0))]


def pack(words):
    return sum((int(w) & 0xFFFFFFFF) << (32*i) for i, w in enumerate(words))


def put(dut, byte_addr, words):
    words = list(words)
    for b in range((len(words) + 7)//8):
        yield dut.sp.mem[byte_addr//32 + b].eq(pack(words[8*b:8*b + 8]))


def get(dut, byte_addr, count):
    out = []
    for b in range((count + 7)//8):
        beat = yield dut.sp.mem[byte_addr//32 + b]
        out += [(beat >> (32*l)) & 0xFFFFFFFF for l in range(8)]
    return out[:count]


def submit(dut, cmd):
    for name, value in cmd.items():
        yield getattr(dut.unit.cmd, name).eq(value)
    yield dut.unit.cmd.valid.eq(1)
    yield
    assert (yield dut.unit.cmd.ready)
    yield dut.unit.cmd.valid.eq(0)
    yield dut.unit.cmd.elements.eq(0)
    yield dut.unit.cmd.dst.eq(0xFFFFFF)
    yield


def wait_done(dut, stalls, limit, seed=5):
    rng, held = random.Random(seed), None
    for cycle in range(limit):
        if stalls:
            yield dut.grant.eq(cycle % 97 not in range(20, 45) and rng.randrange(3) != 0)
        yield
        assert not (yield dut.unit.error), (cycle, (yield dut.unit.error))
        payload = ((yield dut.unit.sp_wr_adr), (yield dut.unit.sp_wr_we), (yield dut.unit.sp_wr_dat))
        if held is not None:
            assert (yield dut.unit.sp_wr_req) and payload == held
        held = payload if (yield dut.unit.sp_wr_req) and not (yield dut.unit.sp_wr_gnt) else None
        if (yield dut.unit.done):
            yield dut.grant.eq(1)
            return cycle
    raise AssertionError("SiLU client timed out")


def load_tables(dut):
    yield from put(dut, 0, table_image())
    yield from submit(dut, run_cmd(op=OP_LOAD, src_g=0))
    yield from wait_done(dut, False, LOAD_BEATS*12)


def check_run(dut, cmd, gates, mults, stalls=True):
    n = cmd["elements"]
    beats, words = (n + 7)//8, (n + 31)//32
    inplace = cmd["dst"] == cmd["src_g"]
    mem = np.full(SIZE//4, SENTINEL, dtype=np.uint32)
    mem[cmd["src_g"]//4:cmd["src_g"]//4 + n] = gates
    mem[cmd["src_m"]//4:cmd["src_m"]//4 + n] = mults
    # Stale bitmap bits must be cleared: pre-fill the bitmap with ones.
    mem[cmd["fixmap"]//4:cmd["fixmap"]//4 + words] = 0xFFFFFFFF
    before = mem.copy()
    fixups = reference(mem, cmd)
    for base in (cmd["src_g"], cmd["src_m"], cmd["dst"], cmd["fixmap"]):
        lo, hi = base//4 - 8, base//4 + 8*(beats + 1)
        yield from put(dut, 4*lo, before[lo:hi])
    yield from submit(dut, cmd)
    yield from wait_done(dut, stalls, 40*n + 2000)
    for base in (cmd["src_g"], cmd["src_m"], cmd["dst"], cmd["fixmap"]):
        lo, hi = base//4 - 8, base//4 + 8*(beats + 1)
        got = yield from get(dut, 4*lo, hi - lo)
        assert got == [int(x) for x in mem[lo:hi]], hex(base)
    assert (yield dut.unit.fixups) == fixups
    assert (yield dut.unit.output_beats) == beats
    assert (yield dut.unit.read_beats) == 2*beats
    assert not (yield dut.unit.busy)
    return fixups


def test_migen_runs_with_stalls_tails_and_inplace():
    dut = SiluSP()
    gates, mults = stress_vectors(seed=31, n_random=300)
    # Put fixups next to exact elements in the first bitmap word.
    gates = [0x7FC00000, 0xC2000000] + gates
    mults = [0x3F800000, 0x3F800000] + mults

    def run():
        seen = 0
        for n, inplace, stalls in ((77, False, True), (8, True, False), (1, False, True),
                                   (300, True, True), (33, False, False)):
            dst = 0x9000 if inplace else 0xB000
            cmd = run_cmd(src_g=0x9000, src_m=0xA000, dst=dst, fixmap=0xC000, elements=n)
            seen += yield from check_run(dut, cmd, gates[:n], mults[:n], stalls)
        assert seen > 0
    run_simulation(dut, run())


REJECTS = [
    (dict(elements=0), 1), (dict(src_g=0x9004), 1), (dict(src_m=0xA010), 1), (dict(dst=0xB001), 1),
    (dict(fixmap=0xC008), 1), (dict(protect_start=0x100, protect_end=0x80), 1),
    (dict(src_m=SIZE - 32), 2), (dict(dst=0xFFFFE0), 2), (dict(fixmap=SIZE), 2),
    (dict(protect_start=SIZE, protect_end=SIZE + 32), 2), (dict(op=OP_LOAD, src_g=SIZE - LOAD_BEATS*32 + 32), 2),
    (dict(dst=0xA000), 4), (dict(dst=0x9020), 4), (dict(fixmap=0x9000), 4),
    (dict(fixmap=0xA3E0), 4), (dict(fixmap=0xB000), 4), (dict(dst=0xA3E0), 4),
    (dict(protect_start=0xB100, protect_end=0xB120), 64),
    (dict(protect_start=0xC000, protect_end=0xC001), 64),
]


@pytest.mark.parametrize("changed,mask", REJECTS)
def test_migen_rejects_before_memory_access(changed, mask):
    dut = SiluSP()
    cmd = run_cmd(src_g=0x9000, src_m=0xA000, dst=0xB000, fixmap=0xC000, elements=250)
    cmd.update(changed)

    def run():
        yield from submit(dut, cmd)
        for _ in range(20):
            yield
            assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
            assert not (yield dut.unit.done) and (yield dut.unit.busy)
        assert (yield dut.unit.error) == mask
        yield dut.unit.cancel.eq(1)
        yield
        yield dut.unit.cancel.eq(0)
        yield
        assert not (yield dut.unit.error) and not (yield dut.unit.busy)
    run_simulation(dut, run())


@pytest.mark.parametrize("delay", [0, 1, 3, 7, 20, 60])
def test_migen_cancel_mid_run_then_clean_run(delay):
    dut = SiluSP()
    gates, mults = stress_vectors(seed=41, n_random=100)

    def run():
        yield dut.grant.eq(0)
        yield from submit(dut, run_cmd(src_g=0x9000, src_m=0xA000, dst=0xB000, fixmap=0xC000, elements=200))
        for _ in range(delay):
            yield
        yield dut.unit.cancel.eq(1)
        yield
        assert not (yield dut.unit.sp_rd_req) and not (yield dut.unit.sp_wr_req)
        yield dut.unit.cancel.eq(0)
        yield dut.grant.eq(1)
        yield
        assert not (yield dut.unit.busy) and not (yield dut.unit.error)
        cmd = run_cmd(src_g=0x9000, src_m=0xA000, dst=0x9000, fixmap=0xC000, elements=45)
        yield from check_run(dut, cmd, gates[:45], mults[:45])
    run_simulation(dut, run())


def test_migen_load_fills_both_tables():
    # Unloaded kernel memories, a real LOAD, then a run that touches both tables.
    dut = SiluSP(preload=False)
    gates, mults = stress_vectors(seed=51, n_random=64)

    def run():
        yield from load_tables(dut)
        assert (yield dut.unit.read_beats) == LOAD_BEATS
        cmd = run_cmd(src_g=0x9000, src_m=0xA000, dst=0xB000, fixmap=0xC000, elements=64)
        yield from check_run(dut, cmd, gates[:64], mults[:64], stalls=False)
    run_simulation(dut, run())


def test_invalid_physical_sizes():
    for size in (0, 31, 33, 256*1024 + 32):
        with pytest.raises(ValueError):
            BonsaiSiluScratchpad(size)


# Compiled ---------------------------------------------------------------------------------------

def test_compiled_full_scratchpad_image(tmp_path):
    from rtl_client import build_client, run_client
    size = 256*1024
    binary = build_client(BonsaiSiluScratchpad(size), SILU_COMMAND, tmp_path, "silu_sp")
    rng = np.random.default_rng(12)
    mem = rng.integers(0, 1 << 32, size//4, dtype=np.uint64).astype(np.uint32)
    tab = 0x37000
    mem[tab//4:tab//4 + LOAD_BEATS*8] = table_image()
    gates, mults = stress_vectors(seed=17, n_random=12000)
    cmds = [(run_cmd(op=OP_LOAD, src_g=tab), 0)]
    n1 = 12345                                   # not a multiple of 8 or 32
    mem[0x00000//4:0x00000//4 + n1] = gates[:n1]
    mem[0x10000//4:0x10000//4 + n1] = mults[:n1]
    cmds.append((run_cmd(src_g=0x0, src_m=0x10000, dst=0x20000, fixmap=0x2F000, elements=n1), 0))
    # In place, over the first result, with the m vector reused.
    cmds.append((run_cmd(src_g=0x20000, src_m=0x10000, dst=0x20000, fixmap=0x2F800, elements=5001), 0))
    # Unit multiplier (the conv SiLU), in place.
    cmds.append((run_cmd(src_g=0x0, src_m=0x3FFE0, dst=0x0, fixmap=0x2FC00, elements=2048, unit_m=1), 0))
    # The DeltaNet gated-norm permutation: 48 heads x 128, grouped order (a = 16, b = 3).
    cmds.append((run_cmd(src_g=0x30000, src_m=0x10000, dst=0x38000, fixmap=0x36000, elements=6144,
                         seg_beats=16, perm_a=16, perm_b=3), 0))
    mem[0x30000//4:0x30000//4 + 6144] = gates[6000:12144]
    cmds.append((run_cmd(src_g=0x30000, src_m=0x30020, dst=0x30040, fixmap=0x30060, elements=9), 4))
    cmds.append((run_cmd(src_g=0x30000, src_m=0x31000, dst=0x32000, fixmap=0x33000, elements=64,
                         protect_start=0x33010, protect_end=0x33014), 64))
    # A permutation must be a full grid, and never in place.
    cmds.append((run_cmd(src_g=0x30000, src_m=0x10000, dst=0x38000, fixmap=0x36000, elements=6144,
                         seg_beats=16, perm_a=16, perm_b=2), 1))
    cmds.append((run_cmd(src_g=0x30000, src_m=0x10000, dst=0x30000, fixmap=0x36000, elements=6144,
                         seg_beats=16, perm_a=16, perm_b=3), 4))
    want = mem.copy()
    for cmd, err in cmds:
        if not err and cmd["op"] == OP_RUN:
            reference(want, cmd)
    got, lines = run_client(binary, SILU_COMMAND, mem, cmds, tmp_path)
    print("\n".join(lines))
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:10]]
    run1 = next(l for l in lines if l.startswith("cmd 1 "))
    assert int(run1.split(" cycles ")[1].split()[0]) < 1.2*n1, run1
    unit = next(l for l in lines if l.startswith("cmd 3 "))
    assert int(unit.split(" reads ")[1].split()[0]) == 2048//8        # g beats only


def sigmoid_tables():
    """The canonical sigmoid table (firmware/fastmath.h) in the unit's lut/dlut form."""
    from trixium.model.rv32_math import sigmoid_table
    lut = np.asarray(sigmoid_table(), dtype=np.float32)
    dlut = (lut[1:] - lut[:-1]).astype(np.float32)
    return [int(x) for x in lut.view(np.uint32)], [int(x) for x in dlut.view(np.uint32)]


def test_compiled_table_only_sigmoid(tmp_path):
    """ABI 1.2: a sigmoid table, table only - the attention output gate o *= sigmoid(g).
    Every non-fixup element equals the firmware's m * bonsai_sigmoid_lookup(g)."""
    from rtl_client import build_client, run_client
    from trixium.model.rv32_math import sigmoid_rv32
    size = 256*1024
    binary = build_client(BonsaiSiluScratchpad(size), SILU_COMMAND, tmp_path, "silu_sp")
    rng = np.random.default_rng(21)
    mem = rng.integers(0, 1 << 32, size//4, dtype=np.uint64).astype(np.uint32)
    tab, pair = 0x37000, sigmoid_tables()
    mem[tab//4:tab//4 + LOAD_BEATS*8] = table_image(pair)
    n = 6144
    g = rng.normal(0, 4, n).astype(np.float32)
    g[:12] = [16, -16, 17.5, 90, -85, -100, 15.999999, -15.999999, 0, -0.0, 79, -79]
    m = rng.normal(0, 1, n).astype(np.float32)
    mem[0:n] = g.view(np.uint32)
    mem[0x10000//4:0x10000//4 + n] = m.view(np.uint32)
    cmds = [(run_cmd(op=OP_LOAD, src_g=tab), 0),
            # In place over the gates: m (the attention outputs) stays for the CPU's fixups.
            (run_cmd(src_g=0x0, src_m=0x10000, dst=0x0, fixmap=0x2F000, elements=n, table_only=1), 0)]
    want = mem.copy()
    fixups = reference(want, cmds[1][0], pair)
    got, lines = run_client(binary, SILU_COMMAND, mem, cmds, tmp_path)
    bad = np.nonzero(got != want)[0]
    assert not len(bad), [(hex(4*i), hex(int(got[i])), hex(int(want[i]))) for i in bad[:10]]
    bits = got[0x2F000//4:0x2F000//4 + (n + 31)//32]
    fix = np.array([(int(bits[i//32]) >> (i % 32)) & 1 for i in range(n)], bool)
    assert fix.sum() == fixups and fix[[0, 1, 2, 3, 4, 5]].all()      # |g| >= 16: CPU
    ok = ~fix
    firmware = (m*sigmoid_rv32(g)).astype(np.float32)
    out = got[0:n].view(np.float32)
    assert np.array_equal(out[ok].view(np.uint32), firmware[ok].view(np.uint32))

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/units/silu.py: SiluMulKernel, SiluScratchpad and sequencer unit 11 on the shared
cluster."""

import random
import struct
import subprocess

import numpy as np
import pytest

from migen import run_simulation, Signal, Mux, Memory, Module

from litex.gen import LiteXModule

from gateware.units.silu import (
    SiluMulKernel, SiluScratchpad, LOAD_BEATS, OP_LOAD, OP_RUN, SILU_COMMAND)
from gateware.core.scratchpad import Scratchpad
from gateware.core.core import TriXiumConfig
from gateware.core.sequencer import Sequencer
from model.silu_hw import LUT_N, f32_bits, silu_mul_firmware, silu_mul_fixed, tables
from model.activation import scratchpad_silu_descriptor, silu_table_image

from common import RTL
from common.cluster import SoC, write, install_list
from common.verilator import Simulator

# SiluMulKernel ------------------------------------------------------------------------------------
# SiluMulKernel (portable RTL) against the bit-exact model, compiled with Verilator.
#
# The chain of evidence: hybrid.c == silu_mul_fast == silu_mul_fixed (test/model/test_silu_hw.py),
# and here RTL == silu_mul_fixed on every element, with random input gaps and output
# stalls. A smaller Migen-simulated run checks the same kernel without Verilator.

def stress_vectors(seed=3, n_random=120000):
    rng = np.random.default_rng(seed)
    gates, mults = [], []
    # Every table node +-2 ulp, each with a few multipliers.
    ms = [f32_bits(np.float32(v)) for v in (1.0, -1.0, 0.7, -3.25, 1e-3, 1234.5, 0.0, -0.0)]
    for i in range(LUT_N + 1):
        b = f32_bits(np.float32(-16 + i/128))
        for dlt in (-2, -1, 0, 1, 2):
            for m in ms[:3] if dlt else ms:
                gates.append((b + dlt) & 0xFFFFFFFF); mults.append(m)
    # Random gates across the table range and across magnitudes down to subnormals.
    gates += [f32_bits(x) for x in rng.uniform(-16, 16, n_random).astype(np.float32)]
    tiny = rng.normal(0, 1, n_random//2) * np.exp2(rng.integers(-150, 4, n_random//2).astype(float))
    gates += [f32_bits(x) for x in tiny.astype(np.float32)]
    # Tails, specials, boundaries.
    for x in (16.0, -16.0, 80.0, -80.0, 16.000002, -16.000002, -80.00001, -79.99999, 17.0, -100.0,
              1e30, -1e30, 3.4e38, -3.4e38, 0.0, -0.0):
        gates.append(f32_bits(np.float32(x)))
    gates += [0x417FFFFF, 0xC17FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80000001]
    while len(mults) < len(gates):
        mults.append(f32_bits(np.float32(rng.normal(0, 3))))
    # Multipliers that push the product out of the normal range, and special multipliers.
    for g in (f32_bits(np.float32(v)) for v in (-10.0, -3.0, 0.5, 5.0, 20.0)):
        for m in (f32_bits(np.float32(v)) for v in (1e-35, -1e-37, 1e36, -3e38)) :
            gates.append(g); mults.append(m)
        for m in (0x7F800000, 0x7FC00000, 0x00000003, 0x80400000, 0x00800000):
            gates.append(g); mults.append(m)
    order = list(range(len(gates)))
    random.Random(seed).shuffle(order)
    return [gates[i] for i in order], [mults[i] for i in order]


def expected(gates, mults):
    lut, dlut = tables()
    out = []
    for g, m in zip(gates, mults):
        r, fix = silu_mul_fixed(g, m, lut, dlut)
        out.append((r, int(fix)))
    return out


@pytest.fixture(scope="module")
def compiled():
    k = SiluMulKernel()
    ports = {"sink_valid": k.sink.valid, "sink_ready": k.sink.ready, "sink_g": k.sink.g,
             "sink_m": k.sink.m, "source_valid": k.source.valid, "source_ready": k.source.ready,
             "source_r": k.source.r, "source_fixup": k.source.fixup,
             "lut_we": k.lut_we, "lut_adr": k.lut_adr, "lut_dat": k.lut_dat,
             "dlut_we": k.dlut_we, "dlut_adr": k.dlut_adr, "dlut_dat": k.dlut_dat}
    for name, sig in ports.items():
        sig.name_override = name
    return Simulator(k, ios=ports, name="silu_mul", driver=(RTL / "silu_mul.cpp").read_text()).binary


def write_vectors(path, gates, mults, want):
    lut, dlut = tables()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(gates)))
        f.write(struct.pack(f"<{len(lut)}I", *lut))
        f.write(struct.pack(f"<{len(dlut)}I", *dlut))
        for g, m, (r, fix) in zip(gates, mults, want):
            f.write(struct.pack("<4I", g, m, r, fix))


def test_compiled_kernel_equals_the_model(tmp_path, compiled):
    gates, mults = stress_vectors()
    want = expected(gates, mults)
    vec = tmp_path / "vectors.bin"
    write_vectors(vec, gates, mults, want)
    run = subprocess.run([str(compiled), str(vec)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)
    fix = sum(f for _, f in want)
    assert f"{len(gates)} elements" in run.stdout and f"{fix} fixups" in run.stdout


def test_model_reference_spot_checked_against_firmware():
    # The bulk reference above is silu_mul_fixed; re-check a sample against the firmware
    # semantics directly so this file stands on its own.
    gates, mults = stress_vectors(seed=9, n_random=2000)
    lut, dlut = tables()
    for g, m in list(zip(gates, mults))[:6000]:
        r, fix = silu_mul_fixed(g, m, lut, dlut)
        if not fix:
            assert r == silu_mul_firmware(g, m), (hex(g), hex(m))


def test_migen_simulation_small():
    """The same kernel in Migen's simulator: a few hundred elements with stalls."""
    k = SiluMulKernel()
    lut, dlut = tables()
    gates, mults = stress_vectors(seed=21, n_random=200)
    gates, mults = gates[:400], mults[:400]
    want = expected(gates, mults)
    rng = random.Random(4)
    got = []

    def drive():
        for i in range(LUT_N + 1):
            yield k.lut_we.eq(1); yield k.lut_adr.eq(i); yield k.lut_dat.eq(lut[i])
            yield k.dlut_we.eq(i < LUT_N); yield k.dlut_adr.eq(i & 0xFFF)
            yield k.dlut_dat.eq(dlut[i] if i < LUT_N else 0)
            yield
        yield k.lut_we.eq(0); yield k.dlut_we.eq(0)
        sent = 0
        for _ in range(8000):
            offer = sent < len(gates) and rng.randrange(6) != 0
            yield k.sink.valid.eq(offer)
            if offer:
                yield k.sink.g.eq(gates[sent]); yield k.sink.m.eq(mults[sent])
            yield k.source.ready.eq(rng.randrange(4) != 0)
            yield
            if (yield k.sink.valid) and (yield k.sink.ready):
                sent += 1
            if (yield k.source.valid) and (yield k.source.ready):
                got.append(((yield k.source.r), (yield k.source.fixup)))
            if len(got) == len(gates):
                return

    run_simulation(k, drive())
    assert len(got) == len(want)
    for idx, ((r, f), (wr, wf)) in enumerate(zip(got, want)):
        assert f == wf, (idx, hex(gates[idx]), hex(mults[idx]))
        if not wf:
            assert r == wr, (idx, hex(gates[idx]), hex(mults[idx]), hex(r), hex(wr))


# SiluScratchpad -----------------------------------------------------------------------------------
# SiluScratchpad: table LOAD and in-scratchpad SiLU-multiply RUN, against the model.
#
# The kernel itself is proven element-exact above (SiluMulKernel); here the questions are
# the client's: every element lands in its lane, fixup lanes carry g, the bitmap has every
# word written and no stale bit, tails and neighbours are untouched, in place works, write
# payloads hold under stall, bad commands never touch memory, cancel recovers.

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


# Migen --------------------------------------------------------------------------------------------

SIZE = 64*1024


class SiluSP(LiteXModule):
    def __init__(self, size=SIZE, preload=True):
        self.sp = Scratchpad(size=size)
        self.unit = SiluScratchpad(size, tables=tables() if preload else None)
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
            SiluScratchpad(size)


# Compiled -----------------------------------------------------------------------------------------

def test_compiled_full_scratchpad_image(tmp_path):
    from common.rtl_client import build_client, run_client
    size = 256*1024
    binary = build_client(SiluScratchpad(size), SILU_COMMAND, tmp_path, "silu_sp")
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
    assert int(run1.split(" cycles ")[1].split()[0]) < 0.4*n1, run1   # 4 ways: ~3 elements/clock
    unit = next(l for l in lines if l.startswith("cmd 3 "))
    assert int(unit.split(" reads ")[1].split()[0]) == 2048//8        # g beats only


def sigmoid_tables():
    """The canonical sigmoid table (firmware/fastmath.h) in the unit's lut/dlut form."""
    from model.rv32_math import sigmoid_table
    lut = np.asarray(sigmoid_table(), dtype=np.float32)
    dlut = (lut[1:] - lut[:-1]).astype(np.float32)
    return [int(x) for x in lut.view(np.uint32)], [int(x) for x in dlut.view(np.uint32)]


def test_compiled_table_only_sigmoid(tmp_path):
    """ABI 1.2: a sigmoid table, table only - the attention output gate o *= sigmoid(g).
    Every non-fixup element equals the firmware's m * trixium_sigmoid_lookup(g)."""
    from common.rtl_client import build_client, run_client
    from model.rv32_math import sigmoid_rv32
    size = 256*1024
    binary = build_client(SiluScratchpad(size), SILU_COMMAND, tmp_path, "silu_sp")
    rng = np.random.default_rng(21)
    mem = rng.integers(0, 1 << 32, size//4, dtype=np.uint64).astype(np.uint32)
    # Both tables resident (ABI 1.3): the SiLU's in set 0, the sigmoid's in set 1.
    tab, pair, tab0 = 0x37000, sigmoid_tables(), 0x26000
    mem[tab//4:tab//4 + LOAD_BEATS*8] = table_image(pair)
    mem[tab0//4:tab0//4 + LOAD_BEATS*8] = table_image()
    n = 6144
    g = rng.normal(0, 4, n).astype(np.float32)
    g[:12] = [16, -16, 17.5, 90, -85, -100, 15.999999, -15.999999, 0, -0.0, 79, -79]
    m = rng.normal(0, 1, n).astype(np.float32)
    mem[0:n] = g.view(np.uint32)
    mem[0x10000//4:0x10000//4 + n] = m.view(np.uint32)
    cmds = [(run_cmd(op=OP_LOAD, src_g=tab, table_set=1), 0),
            (run_cmd(op=OP_LOAD, src_g=tab0), 0),
            # In place over the gates: m (the attention outputs) stays for the CPU's fixups.
            (run_cmd(src_g=0x0, src_m=0x10000, dst=0x0, fixmap=0x2F000, elements=n, table_only=1,
                     table_set=1), 0),
            # The SiLU in set 0 is still intact.
            (run_cmd(src_g=0x20000, src_m=0x10000, dst=0x24000, fixmap=0x2F800, elements=2048), 0)]
    mem[0x20000//4:0x20000//4 + 2048] = g[:2048].view(np.uint32)
    want = mem.copy()
    fixups = reference(want, cmds[2][0], pair)
    reference(want, cmds[3][0])
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


# Sequencer Unit 11 on the Shared Cluster ----------------------------------------------------------
# Sequencer unit 11 (SiLU-multiply): descriptor ABI, dispatch, and a list on the shared cluster.

def descriptor(**changes):
    fields = dict(src_g=0x800, src_m=0x1000, dst=0x800, fixmap=0x1800, elements=17)
    fields.update(changes)
    return scratchpad_silu_descriptor(**fields)


def test_descriptor_encoding_and_configuration():
    assert descriptor() == [11, 0x800, 0x1000, 0x800, 0x1800, 17, 0] + [0]*9
    assert scratchpad_silu_descriptor(src_g=0x40, load=True) == [11, 0x40, 0, 0, 0, 0, 1] + [0]*9
    for changes in ({'src_g': 1}, {'dst': -32}, {'fixmap': 1 << 24}, {'elements': 0}, {'elements': 65536}):
        with pytest.raises(ValueError):
            descriptor(**changes)
    for changes in ({'src_m': 32}, {'elements': 1}):
        with pytest.raises(ValueError):
            scratchpad_silu_descriptor(src_g=0, load=True, **changes)
    lut, dlut = tables()
    image, reference_image = silu_table_image(lut, dlut), [int(w) for w in table_image()]
    assert len(image) == 8*1025 and image[4097:4104] == [0]*7    # padding lanes are ignored
    assert image[:4097] == reference_image[:4097] and image[4104:] == reference_image[4104:]
    for options in ({'with_vecunit': False}, {'scratchpad_size': 512*1024}):
        with pytest.raises(ValueError):
            TriXiumConfig(with_silu=True, **options)


def fixture(descriptors):
    dut = Module()
    dut.submodules.seq = seq = Sequencer(with_silu=True)
    words = [pack(desc[start:start+8]) for desc in descriptors for start in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    return dut


@pytest.mark.parametrize("field,value", [(0, 11 | 1 << 10), (0, 12), (1, 1 << 24), (2, 1 << 24),
    (3, 1 << 24), (4, 1 << 24), (5, 65536), (6, 16)] + [(field, 1) for field in range(8, 16)])
def test_reserved_fields_rejected_before_dispatch(field, value):
    words = descriptor()
    words[field] = value
    dut = fixture([words])
    def run():
        yield from write(dut.seq.base, 0)
        yield from write(dut.seq.count, 1)
        yield from write(dut.seq.start, 1)
        for _ in range(30):
            yield
            assert not (yield dut.seq.silu_cmd.valid)
        assert (yield dut.seq.busy) and (yield dut.seq.silu_error.status) == 2
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 4, 64])
def test_dispatch_fields_fixup_sum_and_failure_wins(failure):
    dut = fixture([scratchpad_silu_descriptor(src_g=0x2000, load=True), descriptor(), descriptor(elements=40)])
    seq, cmd = dut.seq, dut.seq.silu_cmd
    def run():
        assert (yield seq.silu_abi.status) == 0x10003
        yield from write(seq.base, 0)
        yield from write(seq.count, 3)
        yield from write(seq.start, 1)
        for _ in range(30):
            yield
            if (yield cmd.valid):
                break
        assert (yield cmd.op) == 1 and (yield cmd.src_g) == 0x2000 and (yield cmd.protect_end) == 192
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        yield seq.silu_done.eq(1)
        yield
        yield seq.silu_done.eq(0)
        for index, fixups in enumerate((3, 5)):
            for _ in range(30):
                yield
                if (yield cmd.valid):
                    break
            else:
                raise AssertionError('no dispatch')
            fields = dict(op=0, src_g=0x800, src_m=0x1000, dst=0x800, fixmap=0x1800,
                          elements=(17, 40)[index], protect_start=0, protect_end=192)
            for name, value in fields.items():
                assert (yield getattr(cmd, name)) == value, name
            yield cmd.ready.eq(1)
            yield
            yield cmd.ready.eq(0)
            yield seq.silu_unit_fixups.eq(fixups)
            if index == 1 and failure:
                yield seq.silu_failed.eq(failure)
            yield seq.silu_done.eq(1)
            yield
            yield seq.silu_done.eq(0)
            for _ in range(5):
                yield
            if index == 1 and failure:
                assert (yield seq.silu_error.status) == failure << 8 and (yield seq.busy)
                assert (yield seq.silu_fixups.status) == 3
                return
        for _ in range(20):
            yield
        assert not (yield seq.busy) and (yield seq.silu_fixups.status) == 8
        # A new list starts the sum again.
        yield from write(seq.start, 1)
        assert (yield seq.silu_fixups.status) == 0
    run_simulation(dut, run())


def cluster():
    return SoC(TriXiumConfig(with_silu=True, with_projection_restore=True,
                                   scratchpad_size=64*1024), n=1)


def wait_list(dut, limit, error):
    for _ in range(limit):
        yield
        assert not (yield error)
        if not (yield dut.seq.busy):
            return
    raise AssertionError('list timed out')


def preload(dut):
    """Kernel tables straight into the memories (LOAD is covered by the SiluScratchpad tests above)."""
    lut, dlut = tables()
    for kernel in dut.silu.kernels:          # every way has its own table copy
        for i, v in enumerate(lut):
            yield kernel.lut_mem[i].eq(v)
        for i, v in enumerate(dlut):
            yield kernel.dlut_mem[i].eq(v)


def test_cluster_runs_through_the_shared_arbiter():
    """An out-of-place RUN and an in-place RUN on its result, one list, on the cluster."""
    dut = cluster()
    size = 64*1024
    gates, mults = stress_vectors(seed=61, n_random=100)
    gates = [0x7F800000] + gates      # one guaranteed fixup
    mults = [0x3F800000] + mults
    n1, n2 = 70, 45
    mem = np.full(size//4, SENTINEL, dtype=np.uint32)
    g, m, d, f = 0x9000, 0xA000, 0xB000, 0xC000
    mem[g//4:g//4 + n1] = gates[:n1]
    mem[m//4:m//4 + n1] = mults[:n1]
    before = mem.copy()
    cmds = [run_cmd(src_g=g, src_m=m, dst=d, fixmap=f, elements=n1),
            run_cmd(src_g=d, src_m=m, dst=d, fixmap=f + 0x100, elements=n2)]
    fixups = sum(reference(mem, c) for c in cmds)
    descs = [
        scratchpad_silu_descriptor(src_g=c['src_g'], src_m=c['src_m'], dst=c['dst'],
                                   fixmap=c['fixmap'], elements=c['elements']) for c in cmds]

    def run():
        yield from preload(dut)
        for base in (g, m, d, f):
            for beat in range(base//32 - 1, base//32 + 16):
                yield dut.scratchpad.mem[beat].eq(pack(before[8*beat:8*beat + 8]))
        yield from install_list(dut, descs)
        yield from wait_list(dut, 3000, dut.seq.silu_error.status)
        assert (yield dut.seq.status.fields.dcount) == 2
        assert (yield dut.seq.silu_fixups.status) == fixups > 0
        for base, count in ((d, n1), (f, 8*2), (f + 0x100, 8*2)):
            for beat in range((count + 7)//8 + 1):
                got = yield dut.scratchpad.mem[base//32 + beat]
                want = pack(mem[base//4 + 8*beat:base//4 + 8*beat + 8])
                assert got == want, (hex(base), beat)
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_cluster_list_protection_and_reset_recovery():
    dut = cluster()
    def run():
        # The fixmap would overwrite the list itself (second descriptor): rejected, no writes.
        yield from install_list(dut, [descriptor(fixmap=0x20), [0]*16])
        for _ in range(60):
            yield
            assert not (yield dut.silu.sp_wr_req)
        assert (yield dut.seq.busy) and (yield dut.seq.silu_error.status) == 64 << 8
        yield from write(dut.engine_csr.soft_reset, 1)
        for _ in range(100):
            yield
        assert not (yield dut.seq.busy) and not (yield dut.seq.silu_error.status)
        assert not (yield dut.silu.busy)
        yield from preload(dut)
        mem = np.full(64*1024//4, SENTINEL, dtype=np.uint32)
        mem[0x800//4:0x800//4 + 2] = [0x3F800000, 0xC1000000]      # 1.0, -8.0
        mem[0x1000//4:0x1000//4 + 2] = [0x40000000, 0x3F800000]
        yield dut.scratchpad.mem[0x800//32].eq(pack(mem[0x800//4:0x800//4 + 8]))
        yield dut.scratchpad.mem[0x1000//32].eq(pack(mem[0x1000//4:0x1000//4 + 8]))
        yield dut.scratchpad.mem[0x1800//32].eq(pack(mem[0x1800//4:0x1800//4 + 8]))
        reference(mem, run_cmd(src_g=0x800, src_m=0x1000, dst=0x800, fixmap=0x1800, elements=2))
        yield from install_list(dut, [descriptor(elements=2)])
        yield from wait_list(dut, 500, dut.seq.silu_error.status)
        assert (yield dut.scratchpad.mem[0x800//32]) == pack(mem[0x800//4:0x800//4 + 8])
        assert (yield dut.scratchpad.mem[0x1800//32]) == pack(mem[0x1800//4:0x1800//4 + 8])
    run_simulation(dut, run(), clocks={'sys': 10, 'eng': 10})


def test_no_cluster_memory_is_exported_over_the_csr_bus():
    """AutoCSR exports Memory attributes as CSR SRAMs with an extra sys write port: the
    kernel tables became 3-port memories that simulated fine and broke on the board."""
    dut = SoC(TriXiumConfig(with_silu=True, with_projection_restore=True, with_hadamard=True,
                                  with_activation_quantize=True, scratchpad_size=64*1024), n=1)
    assert list(dut.get_memories()) == []

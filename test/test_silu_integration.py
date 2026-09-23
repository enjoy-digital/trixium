# SPDX-License-Identifier: BSD-2-Clause
"""Sequencer unit 11 (SiLU-multiply): descriptor ABI, dispatch, and a list on the shared cluster."""
import pytest
from migen import Memory, Module, run_simulation

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.gateware.sequencer import BonsaiSequencer
from trixium.model.activation import scratchpad_silu_descriptor, silu_table_image
from trixium.model.silu_hw import tables
from test_hadamard_integration import install_list
from test_integration import SoC, write
from test_silu_mul_rtl import stress_vectors
from test_silu_sp import SENTINEL, pack, reference, run_cmd, table_image

import numpy as np


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
            BonsaiClusterConfig(with_silu=True, **options)


def fixture(descriptors):
    dut = Module()
    dut.submodules.seq = seq = BonsaiSequencer(with_silu=True)
    words = [pack(desc[start:start+8]) for desc in descriptors for start in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    return dut


@pytest.mark.parametrize('field,value', [(0, 11 | 1 << 9), (0, 12), (1, 1 << 24), (2, 1 << 24),
    (3, 1 << 24), (4, 1 << 24), (5, 65536), (6, 2)] + [(field, 1) for field in range(7, 16)])
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
        assert (yield seq.silu_abi.status) == 0x10000
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
    return SoC(BonsaiClusterConfig(with_silu=True, with_projection_restore=True,
                                   scratchpad_size=64*1024), n=1)


def wait_list(dut, limit, error):
    for _ in range(limit):
        yield
        assert not (yield error)
        if not (yield dut.seq.busy):
            return
    raise AssertionError('list timed out')


def preload(dut):
    """Kernel tables straight into the memories (LOAD is covered by test_silu_sp.py)."""
    lut, dlut = tables()
    for i, v in enumerate(lut):
        yield dut.silu.kernel.lut_mem[i].eq(v)
    for i, v in enumerate(dlut):
        yield dut.silu.kernel.dlut_mem[i].eq(v)


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
    dut = SoC(BonsaiClusterConfig(with_silu=True, with_projection_restore=True, with_hadamard=True,
                                  with_activation_quantize=True, scratchpad_size=64*1024), n=1)
    assert list(dut.get_memories()) == []

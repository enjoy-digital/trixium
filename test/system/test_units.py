#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Sequencer units 11 (options), 12 (norm) and 13 (conv): decode, and lists on the cluster."""

import numpy as np
import pytest

from migen import Memory, Module, run_simulation

from gateware.core.core import TriXiumConfig
from gateware.core.sequencer import Sequencer
from model.activation import (scratchpad_conv_descriptor, scratchpad_norm_descriptor,
                                           scratchpad_silu_descriptor)

from common import import_test
from common.cluster import BACKEND, SoC, install_list, pack, write
from common.cluster import run as run_cluster

norm_reference = import_test("gateware/units/test_norm.py").reference
conv_reference = import_test("gateware/units/test_conv.py").reference
EPS            = import_test("gateware/units/test_norm.py").EPS

f32 = np.float32


# Helpers ------------------------------------------------------------------------------------------

def fixture(descriptors, **units):
    dut = Module()
    dut.submodules.seq = seq = Sequencer(**units)
    words = [pack(d[i:i + 8]) for d in descriptors for i in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    return dut


def dispatch_fields(dut, cmd, count=1):
    fields = {}
    def run():
        yield from write(dut.seq.base, 0)
        yield from write(dut.seq.count, count)
        yield from write(dut.seq.start, 1)
        for _ in range(40):
            yield
            if (yield cmd.valid):
                break
        else:
            raise AssertionError("no dispatch")
        for name, _ in cmd.description.payload_layout:
            fields[name] = (yield getattr(cmd, name))
    run_simulation(dut, run())
    return fields


# Tests --------------------------------------------------------------------------------------------

def test_silu_options_decode():
    d = scratchpad_silu_descriptor(src_g=0x100, src_m=0x200, dst=0x300, fixmap=0x400, elements=6144,
                                   seg_beats=16, perm_a=16, perm_b=3)
    dut = fixture([d], with_silu=True)
    f = dispatch_fields(dut, dut.seq.silu_cmd)
    assert (f["seg_beats"], f["perm_a"], f["perm_b"], f["unit_m"], f["op"]) == (16, 16, 3, 0, 0)
    d = scratchpad_silu_descriptor(src_g=0x100, dst=0x100, fixmap=0x400, elements=2048, unit_m=True)
    dut = fixture([d], with_silu=True)
    assert dispatch_fields(dut, dut.seq.silu_cmd)["unit_m"] == 1


def test_norm_and_conv_decode_and_reserved_words():
    eps = int(EPS.view(np.uint32))
    d = scratchpad_norm_descriptor(src=0x100, dst=0x200, weight=0x300, n=128, nseg=48, eps=eps, l2=1)
    dut = fixture([d], with_norm=True)
    f = dispatch_fields(dut, dut.seq.norm_cmd)
    assert (f["src"], f["dst"], f["weight"], f["n"], f["nseg"], f["eps"], f["op"]) == (
        0x100, 0x200, 0x300, 128, 48, eps, 1)
    d = scratchpad_conv_descriptor(x0=0, x1=32, x2=64, x3=96, w=128, dst=160, channels=2048)
    dut = fixture([d], with_conv=True)
    f = dispatch_fields(dut, dut.seq.conv_cmd)
    assert [f[k] for k in ("x0", "x1", "x2", "x3", "w", "dst", "channels")] == [0, 32, 64, 96, 128, 160, 2048]
    for units, desc, err in ((dict(with_norm=True), scratchpad_norm_descriptor(
                                  src=0, dst=0, n=8, eps=eps), "norm_error"),
                             (dict(with_conv=True), scratchpad_conv_descriptor(
                                  x0=0, x1=0, x2=0, x3=0, w=0, dst=0, channels=8), "conv_error")):
        bad = list(desc); bad[9] = 1
        dut = fixture([bad], **units)
        def run():
            yield from write(dut.seq.base, 0)
            yield from write(dut.seq.count, 1)
            yield from write(dut.seq.start, 1)
            for _ in range(30):
                yield
            assert (yield getattr(dut.seq, err).status) == 2
        run_simulation(dut, run())


@BACKEND  # Migen simulator: ~3 min
def test_cluster_conv_then_norm_list(backend):
    """conv (16 channels) then an in-place L2 over two 8-element segments, one list."""
    size = 64*1024
    dut = SoC(TriXiumConfig(with_norm=True, with_conv=True, scratchpad_size=size), n=1, backend=backend)
    rng = np.random.default_rng(3)
    mem = np.zeros(size//4, dtype=np.uint32)
    for a in (0x1000, 0x1100, 0x1200, 0x1300):
        mem[a//4:a//4 + 16] = rng.normal(0, 1, 16).astype(f32).view(np.uint32)
    mem[0x1400//4:0x1400//4 + 64] = rng.normal(0, 0.5, 64).astype(f32).view(np.uint32)
    conv = dict(x0=0x1000, x1=0x1100, x2=0x1200, x3=0x1300, w=0x1400, dst=0x2000, channels=16)
    norm = dict(op=1, src=0x2000, dst=0x2000, weight=0, n=8, nseg=2, eps=int(EPS.view(np.uint32)))
    want = mem.copy()
    conv_reference(want, conv)
    norm_reference(want, norm)
    descs = [scratchpad_conv_descriptor(**conv),
             scratchpad_norm_descriptor(src=0x2000, dst=0x2000, n=8, nseg=2, eps=norm["eps"], l2=1)]
    def run():
        for beat in range(0x1000//32, 0x1500//32):
            yield dut.scratchpad.mem[beat].eq(pack(mem[8*beat:8*beat + 8]))
        yield from install_list(dut, descs)
        for _ in range(3000):
            yield
            assert not (yield dut.seq.conv_error.status) and not (yield dut.seq.norm_error.status)
            if not (yield dut.seq.busy):
                break
        else:
            raise AssertionError("list timed out")
        for beat in (0x2000//32, 0x2000//32 + 1):
            assert (yield dut.scratchpad.mem[beat]) == pack(want[8*beat:8*beat + 8]), beat
    run_cluster(dut, run())

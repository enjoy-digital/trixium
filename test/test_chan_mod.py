#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""chan_mod: the per-controller channel offset (the XCU1525's four DDR4 controllers).

With chan_mod = 4 engine channel c addresses controller c // 4 at the FK33's constants for
channel c % 4. The three address paths - the readers (matvec/fetch), the engine STORE and
the host upload - are probed on the bus, since each channel's model is its own memory and
a data round trip would pass with any offset at all. chan_mod = 0 must leave the FK33's
addresses untouched.
"""

import numpy as np
import pytest

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from trixium import layout
from trixium.core.scratchpad import Scratchpad
from trixium.core.engine import MatVecEngine
from trixium.memory.loader import Loader
from trixium.memory.hbm_model import HBMChannelModel, words_from_bytes
from trixium.layout import HBM_CHAN_STRIDE, HBM_WEIGHT_BASE


# Helpers ------------------------------------------------------------------------------------------

class _DUT(Module):
    def __init__(self, n_channels, chan_mod, chan_skew=0):
        axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                for _ in range(n_channels)]
        self.axis = axis
        src = np.arange(1024, dtype=np.uint8)
        self.submodules.scratchpad = Scratchpad(size=64*1024,
                                                      init=words_from_bytes(src.tobytes()))
        self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=32,
                                                    chan_stride=HBM_CHAN_STRIDE,
                                                    chan_mod=chan_mod, chan_skew=chan_skew)
        self.submodules.loader = Loader(axis, data_width=256,
                                                 chan_stride=HBM_CHAN_STRIDE,
                                                 chan_mod=chan_mod, chan_skew=chan_skew)
        self.comb += [
            self.engine.store_cmd.connect(self.loader.store_cmd),
            self.engine.store_data.connect(self.loader.store_data),
            self.engine.store_busy.eq(self.loader.store_busy),
        ]
        for i, axi in enumerate(axis):
            # 64 KB windows addressed modulo their size: the address is checked on the bus.
            model = HBMChannelModel(axi, size=64*1024, latency=4)
            setattr(self.submodules, f"model{i}", model)


def _submit(dut, c):
    for k in ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats",
              "bias", "rshift", "xsrc", "dst", "chan"):
        yield getattr(dut.engine.cmd, k).eq(c.get(k, 0))
    yield dut.engine.cmd.valid.eq(1)
    yield
    while not (yield dut.engine.cmd.ready):
        yield
    yield dut.engine.cmd.valid.eq(0)


def _wait_done(dut):
    for _ in range(8000):
        if (yield dut.engine.done):
            return
        yield
    raise AssertionError("timeout")


def _probe(axi, chan, out, key):
    # Runs until the driver is done (out["done"]): an 8-channel engine is slow to simulate,
    # so the probes must not pin the run to a fixed cycle count.
    while not out.get("done"):
        if (yield axi.aw.valid) and (yield axi.aw.ready):
            out.setdefault(("aw", key), []).append((yield axi.aw.addr))
        if (yield axi.ar.valid) and (yield axi.ar.ready):
            out.setdefault(("ar", key), []).append((yield axi.ar.addr))
        yield


# Tests --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chan_mod, chan_skew", [(0, 0), (4, 0), (4, 0x2000)])
def test_chan_mod_addresses_every_path(chan_mod, chan_skew):
    """Read (FETCH), STORE and host upload each add (c % chan_mod)*stride + (c % chan_mod)*skew
    in hardware - the bank skew (XCU1525: 8 KB) on top of the window stride."""
    n_channels = 8
    dut = _DUT(n_channels, chan_mod, chan_skew)
    out = {}
    wbase = HBM_WEIGHT_BASE + 0x8000

    def local(c):
        return (c % chan_mod) if chan_mod else c

    def off(c):
        return local(c)*HBM_CHAN_STRIDE + local(c)*chan_skew

    def driver():
        # FETCH from channel 6: one reader's AR carries wbase + local(6)*stride.
        yield from _submit(dut, dict(kind=2, chan=6, wbase=wbase, wlen=2, dst=0x4000))
        yield from _wait_done(dut)
        yield
        # STORE to channel 5: the loader's writer on channel 5 sees the masked offset.
        yield from _submit(dut, dict(kind=3, chan=5, wbase=wbase, wlen=2, xsrc=0))
        yield from _wait_done(dut)
        yield
        # Host upload to channel 7 through the loader's CSR path (base + offset in hardware).
        yield dut.loader.chan.storage.eq(7)
        yield dut.loader.base.storage.eq(wbase)
        yield dut.loader.length.storage.eq(2)
        yield dut.loader.start.storage.eq(1)
        yield dut.loader.start.re.eq(1)
        yield
        yield dut.loader.start.re.eq(0)
        for beat in range(2):
            yield dut.loader.sink.data.eq(beat + 1)
            yield dut.loader.sink.valid.eq(1)
            yield
            while not (yield dut.loader.sink.ready):
                yield
        yield dut.loader.sink.valid.eq(0)
        for _ in range(2000):
            if (yield dut.loader.status.fields.done):
                break
            yield
        for _ in range(20):
            yield
        out["done"] = True

        assert out[("ar", 6)][0] == wbase + off(6), \
            [hex(a) for a in out[("ar", 6)][:2]]
        assert out[("aw", 5)][0] == wbase + off(5), \
            [hex(a) for a in out[("aw", 5)][:2]]
        assert out[("aw", 7)][0] == wbase + off(7), \
            [hex(a) for a in out[("aw", 7)][:2]]
        # Nothing else was addressed on the probed channels' write side.
        assert ("aw", 6) not in out

    run_simulation(dut, [driver(),
                         _probe(dut.axis[5], 5, out, 5),
                         _probe(dut.axis[6], 6, out, 6),
                         _probe(dut.axis[7], 7, out, 7)])

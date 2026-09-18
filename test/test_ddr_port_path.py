#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The XCU1525's engine port path: the engine's AXI over LiteDRAM's bridge and converter.

On the FK33 the engine's 256-bit AXI ports meet the HBM controller directly. On the XCU1525
they meet a 512-bit DDR4 controller through LiteDRAMAXI2Native and the crossbar's 256 -> 512
native-port up-converter. This test drives the engine's own reader and the loader's own
writer (16-beat INCR bursts, 8 outstanding) through exactly that chain onto LiteDRAM's
native-port memory model, so a stall or a data mix-up is found here in seconds rather than
in a 30-minute Verilator run (where the first DDR4 sim stalled every op at zero beats,
2026-09-18).
"""

import numpy as np
import pytest

from migen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.axi import AXIInterface

from litedram.common import LiteDRAMNativePort
from litedram.frontend.axi import LiteDRAMAXI2Native
from litedram.frontend.adapter import LiteDRAMNativePortConverter

from trixium.gateware.axi_reader import BonsaiAXIReader
from trixium.gateware.axi_writer import BonsaiAXIWriter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "litex", "litedram"))
from test.common import DRAMMemory  # LiteDRAM's native-port memory model.


class _DUT(Module):
    def __init__(self, port_width=256, ctrl_width=512):
        self.axi = AXIInterface(data_width=256, address_width=33, id_width=6)
        # The crossbar's get_port(data_width=256) on a 512-bit controller: a 256-bit user
        # port, converted onto a 512-bit port (one address bit fewer).
        self.ctrl_port = LiteDRAMNativePort("both", address_width=20, data_width=ctrl_width)
        self.user_port = LiteDRAMNativePort("both", address_width=21, data_width=port_width)
        self.submodules.conv   = LiteDRAMNativePortConverter(self.user_port, self.ctrl_port)
        self.submodules.bridge = LiteDRAMAXI2Native(self.axi, self.user_port,
                                                    w_buffer_depth=64, r_buffer_depth=256)
        self.submodules.reader = BonsaiAXIReader(self.axi)
        self.submodules.writer = BonsaiAXIWriter(self.axi)
        self.comb += self.reader.source.ready.eq(1)   # Drained by the test.
        self.mem = DRAMMemory(ctrl_width, 4096)       # 4096 x 64 B = 256 KB.


def _beats_to_words(data, width):
    return [int.from_bytes(data[i:i+width//8], "little") for i in range(0, len(data), width//8)]


def test_engine_port_path_write_then_read():
    dut  = _DUT()
    rng  = np.random.default_rng(5)
    n_beats = 64
    payload = rng.integers(0, 256, 32*n_beats, dtype=np.uint8).tobytes()
    base = 0x0A0_0000 + 3*0x20_0000     # The sim's channel-3 test address (16 MB).
    got  = []

    def writer():
        yield dut.writer.sink_cmd.addr.eq(base)
        yield dut.writer.sink_cmd.len.eq(n_beats)
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        words = _beats_to_words(payload, 256)
        for w in words:
            yield dut.writer.sink_data.data.eq(w)
            yield dut.writer.sink_data.valid.eq(1)
            yield
            while not (yield dut.writer.sink_data.ready):
                yield
        yield dut.writer.sink_data.valid.eq(0)
        for _ in range(4000):
            if (yield dut.writer.idle):
                break
            yield
        else:
            raise AssertionError("writer never went idle: the write path stalls")
        # Now read it back through the reader.
        yield dut.reader.sink.addr.eq(base)
        yield dut.reader.sink.len.eq(n_beats)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)
        for _ in range(6000):
            if (yield dut.reader.source.valid):
                got.append((yield dut.reader.source.data))
            if len(got) == n_beats:
                break
            yield
        assert len(got) == n_beats, f"read path returned {len(got)} of {n_beats} beats"
        assert got == words, "read data differs from what was written"

    run_simulation(dut, [writer(), dut.mem.write_handler(dut.ctrl_port),
                         dut.mem.read_handler(dut.ctrl_port)])


def test_engine_port_path_through_the_controller():
    """The same chain on the real LiteDRAM core (bank machines, multiplexer, refresher) over
    the PHY model, DFI in hardware mode as after the BIOS's init - the Verilator sim minus
    the CPU. Two engine ports on one controller, so the crossbar arbitrates."""
    from litedram.core import LiteDRAMCore
    from litedram.modules import MT40A512M8
    from litedram.phy.model import SDRAMPHYModel

    class _SimDDR4(MT40A512M8):
        nrows = 2048
        ncols = 128

    class _CtrlDUT(Module):
        def __init__(self):
            module = _SimDDR4(int(100e6), "1:4")
            self.submodules.phy  = SDRAMPHYModel(module=module, data_width=64,
                                                 clk_freq=int(100e6), init=[])
            self.submodules.core = LiteDRAMCore(phy=self.phy, geom_settings=module.geom_settings,
                                                timing_settings=module.timing_settings,
                                                clk_freq=int(100e6))
            self.axis, self.readers, self.writers = [], [], []
            for i in range(2):
                port = self.core.crossbar.get_port(data_width=256)
                axi  = AXIInterface(data_width=256, address_width=33, id_width=6)
                setattr(self.submodules, f"bridge{i}",
                        LiteDRAMAXI2Native(axi, port, w_buffer_depth=64, r_buffer_depth=256))
                rd = BonsaiAXIReader(axi); wr = BonsaiAXIWriter(axi)
                setattr(self.submodules, f"reader{i}", rd)
                setattr(self.submodules, f"writer{i}", wr)
                self.comb += rd.source.ready.eq(1)
                self.axis.append(axi); self.readers.append(rd); self.writers.append(wr)

    dut = _CtrlDUT()
    rng = np.random.default_rng(7)
    n_beats = 48
    payloads = [rng.integers(0, 256, 32*n_beats, dtype=np.uint8).tobytes() for _ in range(2)]
    bases = [0x0A0_0000, 0x0A0_0000 + 0x20_0000]   # Two channels' test addresses.
    got = [[], []]

    def port_driver(i):
        wr, rd = dut.writers[i], dut.readers[i]
        words = _beats_to_words(payloads[i], 256)
        yield wr.sink_cmd.addr.eq(bases[i])
        yield wr.sink_cmd.len.eq(n_beats)
        yield wr.sink_cmd.valid.eq(1)
        yield
        while not (yield wr.sink_cmd.ready):
            yield
        yield wr.sink_cmd.valid.eq(0)
        for w in words:
            yield wr.sink_data.data.eq(w)
            yield wr.sink_data.valid.eq(1)
            yield
            while not (yield wr.sink_data.ready):
                yield
        yield wr.sink_data.valid.eq(0)
        for _ in range(20000):
            if (yield wr.idle):
                break
            yield
        else:
            raise AssertionError(f"port {i}: writer never went idle (write path stalls)")
        yield rd.sink.addr.eq(bases[i])
        yield rd.sink.len.eq(n_beats)
        yield rd.sink.valid.eq(1)
        yield
        while not (yield rd.sink.ready):
            yield
        yield rd.sink.valid.eq(0)
        for _ in range(20000):
            if (yield rd.source.valid):
                got[i].append((yield rd.source.data))
            if len(got[i]) == n_beats:
                break
            yield
        assert len(got[i]) == n_beats, f"port {i}: {len(got[i])} of {n_beats} beats read back"
        assert got[i] == words, f"port {i}: read data differs"

    def control():
        # As after sdram_init(): DFI to the hardware controller, clocks enabled.
        yield dut.core.dfii._control.storage.eq(0b1111)
        yield

    run_simulation(dut, [control(), port_driver(0), port_driver(1)])


def test_engine_port_path_through_the_reset_shim():
    """The chain as the SoC wires it: the engine's reader/writer behind the AXI reset shim
    (engreset.py), the shim's port side into LiteDRAMAXI2Native. The shim derives WLAST from
    its own queue of accepted AW lengths; LiteDRAM's bridge, unlike the HBM IP, accepts W
    beats into a FIFO before their AW has fired, which is where the two can disagree."""
    from trixium.gateware.engreset import BonsaiEngineResetDomain

    class _ShimDUT(Module):
        def __init__(self):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            self.submodules.engrst = BonsaiEngineResetDomain([axi])
            self.ctrl_port = LiteDRAMNativePort("both", address_width=20, data_width=512)
            self.user_port = LiteDRAMNativePort("both", address_width=21, data_width=256)
            self.submodules.conv   = LiteDRAMNativePortConverter(self.user_port, self.ctrl_port)
            self.submodules.bridge = LiteDRAMAXI2Native(axi, self.user_port,
                                                        w_buffer_depth=64, r_buffer_depth=256)
            m = self.engrst.masters[0]
            self.submodules.reader = BonsaiAXIReader(m)
            self.submodules.writer = BonsaiAXIWriter(m)
            self.comb += self.reader.source.ready.eq(1)
            self.mem = DRAMMemory(512, 4096)
            self.port_axi = axi
            self.bcount = Signal(16)
            self.sync += If(axi.b.valid & axi.b.ready, self.bcount.eq(self.bcount + 1))

    dut = _ShimDUT()
    rng = np.random.default_rng(11)
    n_beats = 64
    payload = rng.integers(0, 256, 32*n_beats, dtype=np.uint8).tobytes()
    base = 0x0A0_0000
    got = []

    def driver():
        words = _beats_to_words(payload, 256)
        yield dut.writer.sink_cmd.addr.eq(base)
        yield dut.writer.sink_cmd.len.eq(n_beats)
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        for w in words:
            yield dut.writer.sink_data.data.eq(w)
            yield dut.writer.sink_data.valid.eq(1)
            yield
            while not (yield dut.writer.sink_data.ready):
                yield
        yield dut.writer.sink_data.valid.eq(0)
        for _ in range(6000):
            if (yield dut.writer.idle):
                break
            yield
        else:
            raise AssertionError(f"writer never went idle: {(yield dut.bcount)} B responses "
                                 f"for {n_beats//16} bursts, shim busy={(yield dut.engrst.busy)}")
        assert (yield dut.bcount) == n_beats//16, "one B per burst expected"
        yield dut.reader.sink.addr.eq(base)
        yield dut.reader.sink.len.eq(n_beats)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)
        for _ in range(8000):
            if (yield dut.reader.source.valid):
                got.append((yield dut.reader.source.data))
            if len(got) == n_beats:
                break
            yield
        assert got == words

    run_simulation(dut, [driver(), dut.mem.write_handler(dut.ctrl_port),
                         dut.mem.read_handler(dut.ctrl_port)],
                   clocks={"sys": 10, "eng": 10})


@pytest.mark.parametrize("row_beats", [3, 5])
def test_engine_port_path_short_unaligned_rows(row_beats):
    """KV rows are 1 + head_dim/32 beats (3 on the tiny model, 5 on the 1.7B/8B): consecutive
    rows start at every 32-byte phase of the converter's 64-byte word, and the scores stream
    reads them back row by row. Write 8 rows one STORE each, read them back one FETCH each
    and as one long stream: every byte where it was put."""
    from trixium.gateware.engreset import BonsaiEngineResetDomain

    class _RowDUT(Module):
        def __init__(self):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            self.submodules.engrst = BonsaiEngineResetDomain([axi])
            self.ctrl_port = LiteDRAMNativePort("both", address_width=20, data_width=512)
            self.user_port = LiteDRAMNativePort("both", address_width=21, data_width=256)
            self.submodules.conv   = LiteDRAMNativePortConverter(self.user_port, self.ctrl_port)
            self.submodules.bridge = LiteDRAMAXI2Native(axi, self.user_port,
                                                        w_buffer_depth=64, r_buffer_depth=256)
            m = self.engrst.masters[0]
            self.submodules.reader = BonsaiAXIReader(m)
            self.submodules.writer = BonsaiAXIWriter(m)
            self.comb += self.reader.source.ready.eq(1)
            self.mem = DRAMMemory(512, 4096)

    dut  = _RowDUT()
    rng  = np.random.default_rng(3 + row_beats)
    rows = 8
    base = 0x0A0_0000 + 0x1000
    data = [rng.integers(0, 256, 32*row_beats, dtype=np.uint8).tobytes() for _ in range(rows)]

    def write_row(r):
        words = _beats_to_words(data[r], 256)
        yield dut.writer.sink_cmd.addr.eq(base + r*32*row_beats)
        yield dut.writer.sink_cmd.len.eq(row_beats)
        yield dut.writer.sink_cmd.valid.eq(1)
        yield
        while not (yield dut.writer.sink_cmd.ready):
            yield
        yield dut.writer.sink_cmd.valid.eq(0)
        for w in words:
            yield dut.writer.sink_data.data.eq(w)
            yield dut.writer.sink_data.valid.eq(1)
            yield
            while not (yield dut.writer.sink_data.ready):
                yield
        yield dut.writer.sink_data.valid.eq(0)
        for _ in range(3000):
            if (yield dut.writer.idle):
                return
            yield
        raise AssertionError(f"row {r}: writer never idle")

    def read(addr, n):
        got = []
        yield dut.reader.sink.addr.eq(addr)
        yield dut.reader.sink.len.eq(n)
        yield dut.reader.sink.valid.eq(1)
        yield
        while not (yield dut.reader.sink.ready):
            yield
        yield dut.reader.sink.valid.eq(0)
        for _ in range(6000):
            if (yield dut.reader.source.valid):
                got.append((yield dut.reader.source.data))
            if len(got) == n:
                return got
            yield
        raise AssertionError(f"read of {n} beats at {addr:#x} returned {len(got)}")

    def driver():
        for r in range(rows):
            yield from write_row(r)
        for r in range(rows):
            got = yield from read(base + r*32*row_beats, row_beats)
            assert got == _beats_to_words(data[r], 256), f"row {r} read back wrong"
        got = yield from read(base, rows*row_beats)
        assert got == _beats_to_words(b"".join(data), 256), "the long stream differs"

    run_simulation(dut, [driver(), dut.mem.write_handler(dut.ctrl_port),
                         dut.mem.read_handler(dut.ctrl_port)],
                   clocks={"sys": 10, "eng": 10})


def test_engine_port_path_read_right_after_write():
    """The token path stores a KV row and streams it back as soon as the engine's ST_DRAIN
    saw the writer idle (all B responses in). On the HBM model a read cannot overtake a
    write; on LiteDRAM the bridge answers B when the data left its FIFO, so this pins that
    a read issued the cycle the writer goes idle still returns the new data, row by row,
    with the previous row's read in flight."""
    from trixium.gateware.engreset import BonsaiEngineResetDomain

    class _RAWDUT(Module):
        def __init__(self):
            axi = AXIInterface(data_width=256, address_width=33, id_width=6)
            self.submodules.engrst = BonsaiEngineResetDomain([axi])
            self.ctrl_port = LiteDRAMNativePort("both", address_width=20, data_width=512)
            self.user_port = LiteDRAMNativePort("both", address_width=21, data_width=256)
            self.submodules.conv   = LiteDRAMNativePortConverter(self.user_port, self.ctrl_port)
            self.submodules.bridge = LiteDRAMAXI2Native(axi, self.user_port,
                                                        w_buffer_depth=64, r_buffer_depth=256)
            m = self.engrst.masters[0]
            self.submodules.reader = BonsaiAXIReader(m)
            self.submodules.writer = BonsaiAXIWriter(m)
            self.comb += self.reader.source.ready.eq(1)
            self.mem = DRAMMemory(512, 4096)

    dut  = _RAWDUT()
    rng  = np.random.default_rng(21)
    row_beats, rows = 3, 12
    base = 0x0A0_0000 + 0x2000
    data = [rng.integers(0, 256, 32*row_beats, dtype=np.uint8).tobytes() for _ in range(rows)]
    got  = {}

    def collector():
        r = 0; n = 0
        while r < rows:
            if (yield dut.reader.source.valid):
                got.setdefault(r, []).append((yield dut.reader.source.data))
                n += 1
                if n == row_beats:
                    r += 1; n = 0
            yield

    def driver():
        for r in range(rows):
            # Overwrite first (stale data), then the real row: the read must see the second.
            for payload in (bytes(32*row_beats), data[r]):
                words = _beats_to_words(payload, 256)
                yield dut.writer.sink_cmd.addr.eq(base + r*32*row_beats)
                yield dut.writer.sink_cmd.len.eq(row_beats)
                yield dut.writer.sink_cmd.valid.eq(1)
                yield
                while not (yield dut.writer.sink_cmd.ready):
                    yield
                yield dut.writer.sink_cmd.valid.eq(0)
                for w in words:
                    yield dut.writer.sink_data.data.eq(w)
                    yield dut.writer.sink_data.valid.eq(1)
                    yield
                    while not (yield dut.writer.sink_data.ready):
                        yield
                yield dut.writer.sink_data.valid.eq(0)
                while not (yield dut.writer.idle):
                    yield
            # The read, the very cycle the writer reports idle.
            yield dut.reader.sink.addr.eq(base + r*32*row_beats)
            yield dut.reader.sink.len.eq(row_beats)
            yield dut.reader.sink.valid.eq(1)
            yield
            while not (yield dut.reader.sink.ready):
                yield
            yield dut.reader.sink.valid.eq(0)
        for _ in range(4000):
            if len(got.get(rows - 1, [])) == row_beats:
                break
            yield
        for r in range(rows):
            assert got.get(r) == _beats_to_words(data[r], 256), \
                f"row {r}: read returned {'stale' if got.get(r) == [0]*row_beats else 'wrong'} data"

    run_simulation(dut, [driver(), collector(), dut.mem.write_handler(dut.ctrl_port),
                         dut.mem.read_handler(dut.ctrl_port)],
                   clocks={"sys": 10, "eng": 10})

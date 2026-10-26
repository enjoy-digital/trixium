#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""gateware/memory/sata_upload.py: SATAUpload reads sectors from a LiteSATA user port into
the loader's stream (chunked commands, FIS splitting, failed commands, upload gate, abort,
sectors above 2**32); SATAStore writes the upload stream to the port; both against the port
model with random stalls, and against the real crossbar arbiter when they share the port."""

import random
import struct
import zlib

from migen import *
from migen.fhdl import verilog
from migen.sim import run_simulation

from litex.gen import LiteXModule

from litesata.frontend.arbitration import LiteSATACrossbar

from gateware.memory.sata_upload import LiteSATAPortModel, SATAStore, SATAUpload

PATTERN_MULT = 0x9E3779B1


def pattern_words(sector, nsectors):
    """The model's dwords for the sectors outside its memory."""
    return [((s*128 + i)*PATTERN_MULT) & 0xffffffff for s in range(sector, sector + nsectors)
            for i in range(128)]


def words_of(data):
    return list(struct.unpack(f"<{len(data)//4}I", data))


def csr_write(csr, value):
    yield csr.storage.eq(value)
    yield csr.re.eq(1)
    yield
    yield csr.re.eq(0)
    yield                 # The DUT's registers have reacted when this returns.


def pulse(csr):
    yield csr.re.eq(1)
    yield
    yield csr.re.eq(0)
    yield


def wait_idle(dut, timeout=60000):
    for _ in range(timeout):
        if not (yield dut.status.fields.busy):
            return
        yield
    raise AssertionError("still busy after the timeout")


def read_status(dut):
    return {
        "done":         (yield dut.status.fields.done),
        "error":        (yield dut.status.fields.error),
        "failed":       (yield dut.status.fields.failed),
        "busy":         (yield dut.status.fields.busy),
        "sectors_done": (yield dut.sectors_done.status),
        "dwords":       (yield dut.dwords.status),
        "cmds":         (yield dut.cmds.status),
        "crc":          (yield dut.crc.status),
    }


def upload_transfer(dut, sector, nsectors, chunk=256, upload=1):
    yield from csr_write(dut.upload, upload)
    yield from csr_write(dut.sector, sector)
    yield from csr_write(dut.nsectors, nsectors)
    yield from csr_write(dut.chunk, chunk)
    yield from csr_write(dut.crc_reset, 1)
    yield from pulse(dut.start)
    yield from wait_idle(dut)
    st = yield from read_status(dut)
    for _ in range(300):             # The FIFO and converter drain to the receiver.
        yield
    return st


def receiver(source, got, state, seed=9):
    """The loader: random ready, the 128-bit beats as 4 dwords."""
    prng  = random.Random(seed)
    ready = None
    while not state.get("stop"):
        r = prng.randrange(3) != 0
        if r != ready:            # A generator write costs a propagation pass: only on change.
            yield source.ready.eq(r)
            ready = r
        yield
        if (yield source.valid) and (yield source.ready):
            v = (yield source.data)
            got.extend((v >> (32*k)) & 0xffffffff for k in range(4))


def sender(dut, words128, seed=7):
    """The upload stream into a SATAStore: 128-bit beats with random gaps; gives up when the
    transfer stops (the sink does not accept while idle)."""
    prng = random.Random(seed)
    for w in words128:
        yield dut.sink.data.eq(w)
        yield dut.sink.valid.eq(1)
        yield
        while not (yield dut.sink.ready):
            if not (yield dut.status.fields.busy):
                yield dut.sink.valid.eq(0)
                return
            yield
        yield dut.sink.valid.eq(0)
        for _ in range(prng.randrange(0, 3)):
            yield


def make_image(nsectors, seed):
    return random.Random(seed).randbytes(nsectors*512)


def make_upload(image, **model_kw):
    model = LiteSATAPortModel(sectors=image, fis_dwords=64, stall=0.25, **model_kw)
    dut   = SATAUpload(model, out_dw=128, fifo_depth=32)
    dut.model = model
    return dut


def run_upload(image, transfers, **model_kw):
    """Run `transfers` (a generator taking the dut) against the image; returns the dwords the
    receiver got and the list of statuses the transfers recorded."""
    dut   = make_upload(image, **model_kw)
    got   = []
    state = {"stats": []}

    def driver():
        yield from transfers(dut, state["stats"])
        state["stop"] = True

    run_simulation(dut, [driver(), receiver(dut.source, got, state)])
    return got, state["stats"]


def test_upload_64_sectors_bit_exact():
    image = make_image(64, 1)

    def transfers(dut, stats):
        stats.append((yield from upload_transfer(dut, 0, 64)))

    got, (st,) = run_upload(image, transfers)
    assert got == words_of(image), f"{len(got)} words out, {len(image)//4} wanted"
    assert st["crc"] == zlib.crc32(image), f"crc {st['crc']:08x}"
    assert (st["done"], st["error"], st["failed"], st["busy"]) == (1, 0, 0, 0), st
    assert (st["sectors_done"], st["dwords"], st["cmds"]) == (64, 64*128, 1), st


def test_upload_multi_chunk():
    image = make_image(64, 2)

    def transfers(dut, stats):
        stats.append((yield from upload_transfer(dut, 5, 37, chunk=8)))

    got, (st,) = run_upload(image, transfers)
    want = image[5*512:42*512]
    assert got == words_of(want)
    assert st["crc"] == zlib.crc32(want)
    assert (st["sectors_done"], st["dwords"], st["cmds"]) == (37, 37*128, 5), st
    assert (st["done"], st["error"]) == (1, 0), st


def test_upload_failed_command_stops_the_transfer():
    # Chunks [0,8) and [8,16) succeed, [16,24) holds the bad sector: its data arrives (the
    # core delivers what it got before a late error) and its end beat fails; no more commands.
    image = make_image(64, 3)

    def transfers(dut, stats):
        stats.append((yield from upload_transfer(dut, 0, 37, chunk=8)))

    got, (st,) = run_upload(image, transfers, error_sectors=(20,))
    assert got == words_of(image[:24*512])
    assert (st["done"], st["error"], st["failed"], st["busy"]) == (1, 1, 1, 0), st
    assert (st["sectors_done"], st["dwords"], st["cmds"]) == (16, 24*128, 2), st


def test_upload_gate_drops_the_data():
    image = make_image(16, 4)

    def transfers(dut, stats):
        stats.append((yield from upload_transfer(dut, 0, 16, upload=0)))
        stats.append((yield from upload_transfer(dut, 0, 4, upload=1)))   # Nothing stale ahead.

    got, (st0, st1) = run_upload(image, transfers)
    assert (st0["done"], st0["error"], st0["sectors_done"], st0["dwords"]) == (1, 0, 16, 16*128), st0
    assert st0["crc"] == 0                                              # Nothing was forwarded.
    assert got == words_of(image[:4*512])
    assert st1["crc"] == zlib.crc32(image[:4*512])


def test_abort_then_a_new_transfer():
    image = make_image(64, 5)
    dut   = make_upload(image)
    got   = []
    marks = []
    state = {}

    def driver():
        yield from csr_write(dut.upload, 1)
        yield from csr_write(dut.sector, 0)
        yield from csr_write(dut.nsectors, 64)
        yield from csr_write(dut.chunk, 8)
        yield from pulse(dut.start)
        for _ in range(400):                 # Inside the first command's 1024 dwords.
            yield
        assert (yield dut.status.fields.busy)
        yield from pulse(dut.abort)
        st = yield from read_status(dut)
        assert (st["busy"], st["done"]) == (0, 0), st
        assert 0 < st["dwords"] < 8*128, st   # The command is still in flight: DRAIN next.
        for _ in range(30):                  # The reset data path delivers nothing more...
            yield
        marks.append(len(got))
        for _ in range(30):
            yield
        marks.append(len(got))
        # ...and the next transfer waits for the aborted command's end beat before its own.
        state["st"] = yield from upload_transfer(dut, 40, 12, chunk=8)
        state["stop"] = True

    run_simulation(dut, [driver(), receiver(dut.source, got, state)])
    st = state["st"]
    assert marks[0] == marks[1], marks
    assert got[marks[1]:] == words_of(image[40*512:52*512])
    assert (st["done"], st["error"], st["sectors_done"], st["dwords"], st["cmds"]) == \
        (1, 0, 12, 12*128, 2), st


def test_sector_above_2_32_reads_the_pattern():
    sector = 0x1_2345_6789
    want   = pattern_words(sector, 3)

    def transfers(dut, stats):
        stats.append((yield from upload_transfer(dut, sector, 3, chunk=2)))

    got, (st,) = run_upload(None, transfers)            # A one-sector memory at sector 0.
    assert got == want
    assert st["crc"] == zlib.crc32(struct.pack(f"<{len(want)}I", *want))
    assert (st["sectors_done"], st["dwords"], st["cmds"], st["done"], st["error"]) == (3, 3*128, 2, 1, 0), st


class _SharedPort(LiteXModule):
    """A store and an upload on two user ports of the real crossbar, over one model."""
    def __init__(self, model):
        self.model = model
        self.xbar  = LiteSATACrossbar(model)
        self.store = SATAStore(self.xbar.get_port(32), in_dw=128, fifo_depth=32)
        self.up    = SATAUpload(self.xbar.get_port(32), out_dw=128, fifo_depth=32)


def beats128(data):
    return [int.from_bytes(data[k:k + 16], "little") for k in range(0, len(data), 16)]


def test_store_then_upload_round_trip():
    data  = make_image(40, 6)
    model = LiteSATAPortModel(sectors=None, size_sectors=64, fis_dwords=64, stall=0.25)
    dut   = _SharedPort(model)
    got   = []
    state = {}

    def driver():
        st = dut.store
        yield from csr_write(st.enable, 1)
        yield from csr_write(st.sector, 3)
        yield from csr_write(st.nsectors, 40)
        yield from csr_write(st.chunk, 16)
        yield from csr_write(st.crc_reset, 1)
        yield from pulse(st.start)
        yield from sender(st, beats128(data))
        yield from wait_idle(st)
        state["store"] = yield from read_status(st)
        state["up"]    = yield from upload_transfer(dut.up, 3, 40)
        state["stop"]  = True

    run_simulation(dut, [driver(), receiver(dut.up.source, got, state)])
    st, up = state["store"], state["up"]
    assert got == words_of(data), f"{len(got)} words back, {len(data)//4} wanted"
    assert st["crc"] == up["crc"] == zlib.crc32(data), (hex(st["crc"]), hex(up["crc"]))
    assert (st["done"], st["error"], st["busy"], st["sectors_done"], st["dwords"], st["cmds"]) == \
        (1, 0, 0, 40, 40*128, 3), st
    assert (up["done"], up["error"], up["sectors_done"], up["dwords"]) == (1, 0, 40, 40*128), up


def test_store_failed_command():
    # Chunk [0,8) is written, [8,16) holds the bad sector: its data is sent, its end beat fails,
    # the transfer stops and the sink stops accepting (the sender gives up).
    data  = make_image(20, 8)
    model = LiteSATAPortModel(sectors=None, size_sectors=32, fis_dwords=64, stall=0.25,
                              error_sectors=(10,))
    dut   = _SharedPort(model)
    state = {}

    def driver():
        st = dut.store
        yield from csr_write(st.sector, 0)
        yield from csr_write(st.nsectors, 20)
        yield from csr_write(st.chunk, 8)
        yield from pulse(st.start)
        yield from sender(st, beats128(data))
        yield from wait_idle(st)
        state["store"] = yield from read_status(st)
        # The sectors before the failure are on the device.
        state["up"] = yield from upload_transfer(dut.up, 0, 8)
        state["stop"] = True

    got = []
    run_simulation(dut, [driver(), receiver(dut.up.source, got, state)])
    st = state["store"]
    assert (st["done"], st["error"], st["failed"], st["busy"]) == (1, 1, 1, 0), st
    assert (st["sectors_done"], st["dwords"], st["cmds"]) == (8, 16*128, 1), st
    assert got == words_of(data[:8*512])


def test_modules_convert_to_verilog():
    # The model goes into the Verilator target: nothing Python-only may remain in the netlist.
    model = LiteSATAPortModel(sectors=make_image(2, 1), base_sector=5, error_sectors=(7,), stall=0.1)
    dut   = _SharedPort(model)
    v = str(verilog.convert(dut))
    assert "module top" in v and str(PATTERN_MULT) in v

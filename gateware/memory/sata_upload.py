#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""SATAUpload / SATAStore: the weights from (and to) a SATA SSD, through a LiteSATA user port.

The standalone build streams its model from an SSD: no host DMA, no card. Two stream modules
on two crossbar user ports:

- `SATAUpload` (SSD -> loader, the SoC names it `sataup`): with `upload` set, the sectors read
  come out of `source` as the loader's upload stream (32-bit dwords in SSD order widened to
  `out_dw`, the first dword in the low bits: the UDP port's convention), landing in the
  channel's window with no CPU copy. The firmware arms the loader (chan, base, length) as for
  a UDP transfer, then sets `sector`/`nsectors` here and pulses `start`: the transfer is
  issued to the port as READ commands of at most `chunk` sectors (the command's count field is
  16 bits; a chunk is only how often the command's overhead is paid) and `status.done` says
  when the last one ended. A transfer is a whole number of 512-byte sectors (16 beats of 256
  bits for the loader): the loader is armed for the padded length.
- `SATAStore` (upload stream -> SSD, `satastore`): the SSD is written once, from the UDP
  upload stream: with `enable` set the SoC routes that stream into `sink` instead of the
  loader. The firmware sets `sector`/`nsectors`, pulses `start`, and the host sends exactly
  `nsectors*512` bytes: they go to the port as WRITE commands of `chunk` sectors. While idle
  the module does not accept data (`sink.ready` = 0): the SoC's mux never loses bytes, and the
  UDP port (which never back-pressures its link) drops and the host resends from `expected`.

Both: `status.error`/`failed` when a command's end beat (or, for reads, a data beat) carries
`failed`: the transfer stops there (`sectors_done`, `cmds` say where). `abort` returns the FSM
to IDLE and discards the staged data (an upload's command in flight is drained in IDLE, and
the next `start` waits for its end beat first, so no transfer ever sees another's beats; a
store's command in flight is completed with zero dwords so the device's command ends, a
second `abort` forces IDLE if it never does). `crc` is the CRC-32 (zlib's) of the data bytes
moved since `crc_reset`, the firmware's table check (reads: the bytes handed to the loader;
writes: the bytes handed to the port).

`LiteSATAPortModel` stands in for a crossbar user port in simulation (unit tests and the
Verilator target, with a real image): the command protocol above over a small Memory, and a
deterministic pattern for the sectors outside it.
"""

import struct
from functools import reduce
from operator import or_, xor

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSR, CSRField, CSRStatus, CSRStorage

from liteeth.mac.crc import crc_calc

from litesata.common import command_rx_description, command_tx_description, logical_sector_size

SECTOR_DWORDS = logical_sector_size//4    # 128 32-bit words per sector.

# CRC32Dword ---------------------------------------------------------------------------------------

class CRC32Dword(LiteXModule):
    """CRC-32 (IEEE 802.3: zlib's) of a stream of 32-bit words, the first byte in the low bits.
    `value` is the CRC of the words since `reset` (final XOR applied), the UDP port's convention
    (the loader's tables are checked against it).

    The update is the linear map of LiteEth's CRC engine (the columns come from liteeth's
    `crc_calc`, the engine's Python reference) written as a XOR of constant-masked input bits
    instead of one XOR tree per output bit: the same gates once synthesised, a tenth of the
    expression nodes for the Migen simulator, which re-evaluates every comb statement on every
    propagation pass (the engine alone made these tests minutes long). Its own register, as
    the UDP port's: the MAC's CRC32 module presents the CRC of the word on its input before it
    is accepted, which a CSR readout must not see."""
    def __init__(self):
        self.reset = Signal()
        self.ce    = Signal()
        self.data  = Signal(32)
        self.value = Signal(32)

        # # #

        # The register holds the CRC bit-reversed (zlib's order, the engine's reversed): the
        # reversal is folded into the constants and `value` is one XOR away.
        reg  = Signal(32, reset=0xffffffff)
        poly = 0x04c11db7
        rev  = lambda v: int(f"{v:032b}"[::-1], 2)
        columns = [(self.data, [rev(crc_calc(32, 32, poly, 0, 1 << n)) for n in range(32)]),
                   (reg,       [rev(crc_calc(32, 32, poly, 1 << (31 - n), 0)) for n in range(32)])]
        nxt = reduce(xor, [Replicate(src[n], 32) & col[n] for src, col in columns for n in range(32)])
        self.comb += self.value.eq(reg ^ 0xffffffff)
        self.sync += If(self.reset, reg.eq(0xffffffff)).Elif(self.ce, reg.eq(nxt))

# Transfer CSRs ------------------------------------------------------------------------------------

class _SATATransfer(LiteXModule):
    """The CSRs and the chunk bookkeeping SATAUpload and SATAStore share: `nsectors` sectors
    from `sector`, as commands of at most `chunk` sectors. The subclass's FSM calls
    `start_actions` on `start.re` and `chunk_done_actions` on a command's good end beat."""
    def __init__(self):
        self.sector   = CSRStorage(48, description="The transfer's first sector.")
        self.nsectors = CSRStorage(32, description="Sectors in the transfer (0: done at once).")
        self.chunk    = CSRStorage(16, reset=256, description="Sectors per SATA command (1..65535; 0 acts as 1).")
        self.start    = CSR()   # Write: start the transfer (ignored while busy).
        self.abort    = CSR()   # Write: stop the transfer, discard the staged data.
        self.status   = CSRStatus(fields=[
            CSRField("done",   size=1, offset=0, description="The last transfer started ran to its end "
                                                             "(all sectors, or a failed command)."),
            CSRField("error",  size=1, offset=1, description="The transfer stopped on a failed command."),
            CSRField("failed", size=1, offset=2, description="A beat of the last command carried `failed`."),
            CSRField("busy",   size=1, offset=3, description="A transfer is in progress (abort clears it)."),
        ])
        self.sectors_done = CSRStatus(32, description="Sectors completed in this transfer.")
        self.dwords       = CSRStatus(32, description="Data dwords (32-bit) moved in this transfer.")
        self.cmds         = CSRStatus(16, description="SATA commands completed in this transfer.")
        self.crc_reset    = CSRStorage(1, description="Write 1: restart the data CRC.")
        self.crc          = CSRStatus(32, description="CRC-32 (zlib's) of the data bytes since crc_reset.")

        # # #

        self.cur       = Signal(48)   # The next command's first sector.
        self.remaining = Signal(32)   # Sectors not yet commanded.
        self.n         = Signal(16)   # The current command's sectors.
        self.n_next    = Signal(16)   # The next command's sectors: min(remaining, chunk).
        self.done      = Signal()
        self.error     = Signal()
        self.failed    = Signal()
        self.busy      = Signal()
        chunk = Signal(16)
        self.comb += [
            chunk.eq(Mux(self.chunk.storage == 0, 1, self.chunk.storage)),   # 0 would never end.
            self.n_next.eq(Mux(self.remaining < chunk, self.remaining[:16], chunk)),
            self.status.fields.done.eq(self.done),
            self.status.fields.error.eq(self.error),
            self.status.fields.failed.eq(self.failed),
            self.status.fields.busy.eq(self.busy),
        ]
        self.crc32 = crc32 = CRC32Dword()
        self.comb += [
            crc32.reset.eq(self.crc_reset.re),
            self.crc.status.eq(crc32.value),
        ]

    def start_actions(self):
        return [
            NextValue(self.cur,       self.sector.storage),
            NextValue(self.remaining, self.nsectors.storage),
            NextValue(self.done,   0),
            NextValue(self.error,  0),
            NextValue(self.failed, 0),
            NextValue(self.sectors_done.status, 0),
            NextValue(self.dwords.status,       0),
            NextValue(self.cmds.status,         0),
        ]

    def chunk_done_actions(self):
        return [
            NextValue(self.cur,       self.cur + self.n),
            NextValue(self.remaining, self.remaining - self.n),
            NextValue(self.sectors_done.status, self.sectors_done.status + self.n),
            NextValue(self.cmds.status,         self.cmds.status + 1),
        ]

    def fail_actions(self):
        return [NextValue(self.failed, 1), NextValue(self.error, 1), NextValue(self.done, 1)]

# SATAUpload ---------------------------------------------------------------------------------------

class SATAUpload(_SATATransfer):
    """SSD -> loader: READ commands on a LiteSATA user port, the data as the upload stream."""
    def __init__(self, port, out_dw=128, fifo_depth=128):
        _SATATransfer.__init__(self)
        assert len(port.source.data) == 32
        self.upload = CSRStorage(1, description="1: the sectors read feed the weight loader "
                                                "(0: they are read and dropped).")
        self.source = stream.Endpoint([("data", out_dw)])

        # # #

        cmd, rsp = port.sink, port.source   # Commands out to the port, responses back from it.
        upload   = self.upload.storage

        # Data path: port dwords -> FIFO -> loader beats. The core tolerates a stalled consumer
        # (the link layer buffers a FIS and sends HOLD): the FIFO only decouples the loader's
        # ready. Abort resets both: a partial beat left in the converter would misalign every
        # later transfer.
        self.fifo = fifo = ResetInserter()(stream.SyncFIFO([("data", 32)], fifo_depth))
        self.conv = conv = ResetInserter()(stream.Converter(32, out_dw))
        self.comb += [
            fifo.reset.eq(self.abort.re),
            conv.reset.eq(self.abort.re),
            fifo.sink.data.eq(rsp.data),
            fifo.source.connect(conv.sink, keep={"valid", "ready", "data"}),
            conv.source.connect(self.source, keep={"valid", "ready", "data"}),
            self.crc32.ce.eq(fifo.sink.valid & fifo.sink.ready),
            self.crc32.data.eq(fifo.sink.data),
        ]

        # A command the port accepted and has not ended yet (an abort leaves one in flight).
        pending  = Signal()
        end_beat = Signal()
        self.comb += end_beat.eq(rsp.valid & rsp.ready & rsp.end)
        self.sync += If(end_beat, pending.eq(0)).Elif(cmd.valid & cmd.ready, pending.eq(1))

        self.fsm = fsm = FSM(reset_state="IDLE")
        self.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        fsm.act("IDLE",
            rsp.ready.eq(1),                  # Drain an aborted command's beats.
            If(self.start.re,
                *self.start_actions(),
                If(self.nsectors.storage == 0,
                    NextValue(self.done, 1),
                ).Elif(pending,
                    NextState("DRAIN"),
                ).Else(
                    NextState("CMD"),
                ),
            ),
        )
        fsm.act("DRAIN",
            # The aborted command's data would be counted in this transfer: wait it out.
            rsp.ready.eq(1),
            If(end_beat, NextState("CMD")),
            If(self.abort.re, NextState("IDLE")),
        )
        fsm.act("CMD",
            cmd.valid.eq(1),
            cmd.last.eq(1),
            cmd.read.eq(1),
            cmd.sector.eq(self.cur),
            cmd.count.eq(self.n_next),
            If(cmd.ready,
                NextValue(self.n, self.n_next),
                NextState("DATA"),
            ),
            If(self.abort.re, NextState("IDLE")),
        )
        fsm.act("DATA",
            fifo.sink.valid.eq(rsp.valid & ~rsp.end & upload),
            rsp.ready.eq(fifo.sink.ready | rsp.end | ~upload),
            If(rsp.valid & rsp.ready,
                If(~rsp.end,
                    NextValue(self.dwords.status, self.dwords.status + 1),
                    If(rsp.failed, NextValue(self.failed, 1)),   # A link error on a DATA FIS.
                ).Elif(rsp.failed | self.failed,
                    *self.fail_actions(),
                    NextState("IDLE"),
                ).Else(
                    *self.chunk_done_actions(),
                    If(self.remaining == self.n,
                        NextValue(self.done, 1),
                        NextState("IDLE"),
                    ).Else(
                        NextState("CMD"),
                    ),
                ),
            ),
            If(self.abort.re, NextState("IDLE")),
        )

# SATAStore ----------------------------------------------------------------------------------------

class SATAStore(_SATATransfer):
    """Upload stream -> SSD: WRITE commands on a LiteSATA user port, the data from `sink`."""
    def __init__(self, port, in_dw=128, fifo_depth=128):
        _SATATransfer.__init__(self)
        assert len(port.sink.data) == 32
        self.enable = CSRStorage(1, description="1: the SoC routes the upload stream here "
                                                "(instead of the loader).")
        self.sink = stream.Endpoint([("data", in_dw)])

        # # #

        cmd, rsp = port.sink, port.source

        # Data path: stream beats -> dwords -> FIFO -> the port's write beats. Accepted only
        # while a transfer runs (the host sends exactly its bytes); whatever is staged is
        # discarded at the next start or abort, so a stray beat cannot shift a later transfer.
        flush = Signal()
        self.conv = conv = ResetInserter()(stream.Converter(in_dw, 32))
        self.fifo = fifo = ResetInserter()(stream.SyncFIFO([("data", 32)], fifo_depth))
        self.comb += [
            flush.eq(self.start.re | self.abort.re),
            conv.reset.eq(flush),
            fifo.reset.eq(flush),
            If(self.busy,
                self.sink.connect(conv.sink, keep={"valid", "ready", "data"}),
            ),
            conv.source.connect(fifo.sink, keep={"valid", "ready", "data"}),
            self.crc32.ce.eq(cmd.valid & cmd.ready),
            self.crc32.data.eq(cmd.data),
        ]

        i      = Signal(23)               # Dword index in the command (n*128 < 2**23).
        last_i = Signal(23)
        self.comb += last_i.eq(Cat(C(0, 7), self.n) - 1)   # n*128 - 1.

        self.fsm = fsm = FSM(reset_state="IDLE")
        self.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        fsm.act("IDLE",
            rsp.ready.eq(1),
            If(self.start.re,
                *self.start_actions(),
                If(self.nsectors.storage == 0,
                    NextValue(self.done, 1),
                ).Else(
                    NextState("CMD"),
                ),
            ),
        )
        fsm.act("CMD",
            NextValue(self.n, self.n_next),
            NextValue(i, 0),
            NextState("DATA"),
            If(self.abort.re, NextState("IDLE")),
        )
        fsm.act("DATA",
            # Every beat of a write carries the command; the last dword ends it.
            cmd.valid.eq(fifo.source.valid),
            cmd.write.eq(1),
            cmd.sector.eq(self.cur),
            cmd.count.eq(self.n),
            cmd.data.eq(fifo.source.data),
            cmd.last.eq(i == last_i),
            fifo.source.ready.eq(cmd.ready),
            rsp.ready.eq(1),
            If(cmd.valid & cmd.ready,
                NextValue(i, i + 1),
                NextValue(self.dwords.status, self.dwords.status + 1),
                If(cmd.last, NextState("END")),
            ),
            If(rsp.valid & rsp.end & rsp.failed,   # The device refused the command early.
                *self.fail_actions(),
                NextState("IDLE"),
            ),
            If(self.abort.re, NextState("PAD")),
        )
        fsm.act("PAD",
            # Aborted mid-command: complete the device's command with zeros (the sectors are
            # the transfer's own) so the port is left between commands.
            cmd.valid.eq(1),
            cmd.write.eq(1),
            cmd.sector.eq(self.cur),
            cmd.count.eq(self.n),
            cmd.last.eq(i == last_i),
            rsp.ready.eq(1),
            If(cmd.ready,
                NextValue(i, i + 1),
                If(cmd.last, NextState("IDLE")),
            ),
            If(rsp.valid & rsp.end, NextState("IDLE")),
            If(self.abort.re, NextState("IDLE")),
        )
        fsm.act("END",
            rsp.ready.eq(1),
            If(rsp.valid & rsp.end,
                If(rsp.failed,
                    *self.fail_actions(),
                    NextState("IDLE"),
                ).Else(
                    *self.chunk_done_actions(),
                    If(self.remaining == self.n,
                        NextValue(self.done, 1),
                        NextState("IDLE"),
                    ).Else(
                        NextState("CMD"),
                    ),
                ),
            ),
            If(self.abort.re, NextState("IDLE")),
        )

# LiteSATAPortModel --------------------------------------------------------------------------------

class LiteSATAPortModel(LiteXModule):
    """A LiteSATA crossbar user port's device side, for simulation (synthesizable: the
    Verilator target instantiates it with a real image).

    `sectors` (bytes, padded to whole sectors) fill a Memory from `base_sector`; `size_sectors`
    (default: the image's) sets the memory's size, so sectors past the image are writable too.
    Sectors outside the memory read as `((sector*128 + i)*0x9E3779B1) & 0xffffffff` per dword
    (a test can predict the CRC of any sector, above 2**32 included) and writes to them are
    dropped. A command touching a sector of `error_sectors` ends with `failed` (its data is
    delivered first, as the core does on a late error). READ data is split into FISes of
    `fis_dwords` (`last` on each); an IDENTIFY, or a beat neither read nor write, is answered
    with a failed end beat. `stall` (0..1) is the probability, from an LFSR, of a cycle with
    `source.valid` or `sink.ready` low (0: none, and no LFSR)."""
    def __init__(self, sectors=None, base_sector=0, fis_dwords=2048, error_sectors=(), stall=0.0,
                 size_sectors=None):
        self.dw            = 32
        self.controller_dw = 32
        self.sink   = sink   = stream.Endpoint(command_tx_description(32))
        self.source = source = stream.Endpoint(command_rx_description(32))

        # # #

        image = bytes(sectors or b"")
        image += b"\0"*(-len(image) % logical_sector_size)
        if size_sectors is None:
            size_sectors = max(len(image)//logical_sector_size, 1)
        assert size_sectors*logical_sector_size >= len(image)
        words  = list(struct.unpack(f"<{len(image)//4}I", image))
        words += [0]*(size_sectors*SECTOR_DWORDS - len(words))
        self.mem = mem = Memory(32, size_sectors*SECTOR_DWORDS, init=words)
        self.port = port = mem.get_port(write_capable=True)
        self.specials += mem, port
        base_dw = base_sector*SECTOR_DWORDS
        end_dw  = (base_sector + size_sectors)*SECTOR_DWORDS

        # Random stalls: a 16-bit LFSR, one byte compared per side.
        stall_v = Signal()
        stall_r = Signal()
        if stall:
            th   = min(int(round(stall*256)), 255)
            lfsr = Signal(16, reset=0xace1)
            self.sync += [
                lfsr.eq(Cat(lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10], lfsr[:15])),
                stall_v.eq(lfsr[0:8] < th),
                stall_r.eq(lfsr[8:16] < th),
            ]

        cmd_sector = Signal(48)
        cmd_count  = Signal(16)
        cmd_read   = Signal()
        cmd_write  = Signal()
        dw_addr    = Signal(55)     # sector*128 + i of the current data beat.
        i          = Signal(23)     # Dword index in a read.
        last_i     = Signal(23)
        fis_cnt    = Signal(max=fis_dwords)
        failed     = Signal()
        self.comb += failed.eq(~(cmd_read | cmd_write) | reduce(or_, [
            (cmd_sector <= e) & (e < cmd_sector + cmd_count) for e in error_sectors], 0))

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            sink.ready.eq(~stall_r),
            If(sink.valid & sink.ready,
                NextValue(cmd_sector, sink.sector),
                NextValue(cmd_count,  sink.count),
                NextValue(cmd_read,   sink.read),
                NextValue(cmd_write,  sink.write),
                NextValue(last_i,     Cat(C(0, 7), sink.count) - 1),
                NextValue(i, 0),
                NextValue(fis_cnt, 0),
                If(sink.read,
                    NextValue(dw_addr, Cat(C(0, 7), sink.sector)),
                    If(sink.count == 0, NextState("END")).Else(NextState("READ_DATA")),
                ).Elif(sink.write,
                    # This beat is the first data dword (written below).
                    NextValue(dw_addr, Cat(C(0, 7), sink.sector) + 1),
                    If(sink.last, NextState("END")).Else(NextState("WRITE_DATA")),
                ).Else(
                    NextState("END"),
                ),
            ),
        )
        fsm.act("WRITE_DATA",
            sink.ready.eq(~stall_r),
            If(sink.valid & sink.ready,
                NextValue(dw_addr, dw_addr + 1),
                If(sink.last, NextState("END")),
            ),
        )
        fsm.act("READ_DATA",
            source.valid.eq(~stall_v),
            source.read.eq(1),
            source.last.eq((fis_cnt == fis_dwords - 1) | (i == last_i)),
            If(source.valid & source.ready,
                NextValue(dw_addr, dw_addr + 1),
                NextValue(i, i + 1),
                NextValue(fis_cnt, Mux(source.last, 0, fis_cnt + 1)),
                If(i == last_i, NextState("END")),
            ),
        )
        fsm.act("END",
            source.valid.eq(~stall_v),
            source.last.eq(1),
            source.end.eq(1),
            source.read.eq(cmd_read),
            source.write.eq(cmd_write),
            source.failed.eq(failed),
            If(source.valid & source.ready, NextState("IDLE")),
        )

        # The memory port: a read's address runs one beat ahead (the read is synchronous, so
        # the next dword is there when the current one is taken, stalls included); a write's
        # is the current beat's. A command's first beat in IDLE uses the command's sector.
        rd_accept = Signal()
        wr_beat   = Signal()
        adr_abs   = Signal(55)
        self.comb += [
            rd_accept.eq(fsm.ongoing("READ_DATA") & source.valid & source.ready),
            wr_beat.eq(sink.valid & sink.ready &
                       Mux(fsm.ongoing("IDLE"), sink.write, fsm.ongoing("WRITE_DATA"))),
            If(fsm.ongoing("IDLE"),
                adr_abs.eq(Cat(C(0, 7), sink.sector)),
            ).Else(
                adr_abs.eq(dw_addr + rd_accept),
            ),
            port.adr.eq((adr_abs - base_dw)[:len(port.adr)]),
            port.dat_w.eq(sink.data),
            port.we.eq(wr_beat & (adr_abs >= base_dw) & (adr_abs < end_dw)),
            If((dw_addr >= base_dw) & (dw_addr < end_dw),
                source.data.eq(port.dat_r),
            ).Else(
                source.data.eq((dw_addr[:32]*0x9E3779B1)[:32]),
            ),
        ]

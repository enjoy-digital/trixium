#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/host/udpload.py against a model of the board: the loader's CSRs and the UDP port's
rule (only the expected sequence number passes), over a link that drops and reorders: the
memory ends up holding the image at wbload's placement, and losses are resent."""

import os
import random
import types

import pytest

from model.bons import BONSReader
from software.host import udpload, wbload

IMAGE = "models/tiny/bonsai-tiny-hybrid-1k-t5-n2.img"


class Board:
    def __init__(self, consts, drop=0.0, reorder=0.0, seed=1):
        self.constants = types.SimpleNamespace(d=consts)
        self.rng, self.drop, self.reorder = random.Random(seed), drop, reorder
        self.mem, self.expected, self.armed, self.held = {}, 0, None, []
        board = self
        class Reg:
            def __init__(self, name): self.name = name
            def write(self, v):
                if self.name == "udpload_restart":
                    board.expected = v
                elif self.name == "trixium_loader_start":
                    board.armed = dict(chan=board.chan, base=board.base, length=board.length, data=b"")
                else:
                    setattr(board, self.name.replace("trixium_loader_", ""), v)
            def read(self):
                if self.name == "udpload_expected":
                    return board.expected
                if self.name == "trixium_loader_status":
                    a = board.armed
                    return 1 if a and len(a["data"]) >= 32*a["length"] else 0
                return 0
        names = ["src_sel", "chan", "base", "length", "resync", "start", "status"]
        self.regs = types.SimpleNamespace(**{f"trixium_loader_{n}": Reg(f"trixium_loader_{n}") for n in names},
                                          udpload_restart=Reg("udpload_restart"),
                                          udpload_expected=Reg("udpload_expected"))

    def send(self, seq, payload):                # The link, then the port.
        if self.rng.random() < self.drop:
            return
        if self.rng.random() < self.reorder:
            self.held.append((seq, payload))
            return
        for s, p in [(seq, payload)] + [self.held.pop(0)] if self.held and self.rng.random() < 0.5 else [(seq, payload)]:
            if s == self.expected and len(p) % 32 == 0 and p:
                a = self.armed
                a["data"] += p
                self.expected += 1
                if len(a["data"]) >= 32*a["length"]:
                    self.mem[(a["chan"], a["base"])] = a["data"][:32*a["length"]]


@pytest.mark.skipif(not os.path.exists(IMAGE), reason="tiny image not generated (cli models --tiny)")
@pytest.mark.parametrize("drop,reorder", [(0.0, 0.0), (0.05, 0.05)])
def test_upload_over_a_lossy_link(drop, reorder):
    board = Board(dict(trixium_chan_weight_base=0x00C0_0000, trixium_chan_stride=0x0080_0000,
                       trixium_loader_adds_chan_offset=1, trixium_n_channels=2), drop, reorder)
    total, resent = udpload.upload(board, board, IMAGE, beats_per_datagram=32, window=16)
    img = BONSReader(IMAGE); h = img.header
    meta = bytes(img.data[:h["payload_offset"]])
    staged = board.mem[(0, 0x00C0_0000 - wbload.HDR_STAGING_BELOW)]
    assert staged[:len(meta)] == meta
    for c in range(h["n_channels"]):
        assert board.mem[(c, 0x00C0_0000)] == bytes(img.channel_payload(c))
    assert (resent > 0) == (drop > 0)

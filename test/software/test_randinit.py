#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/lab/randinit.py: which registers get a random power-on value, on Migen's Verilog."""

from migen import *
from migen.fhdl import verilog
from migen.genlib.cdc import MultiReg

from software.lab import randinit

class _DUT(Module):
    def __init__(self):
        self.i = Signal(8)
        self.ctrl = Signal(4)                       # Reset.
        self.data = Signal(8, reset_less=True)      # Never reset: randomized.
        self.comb_only = Signal(8)                  # Combinational: untouched.
        self.synced = Signal(8)
        self.sync += [self.ctrl.eq(self.ctrl + 1), self.data.eq(self.i)]
        self.comb += self.comb_only.eq(self.i)
        self.specials += MultiReg(self.data, self.synced) # Synchronizer stages: untouched.
        self.mem = Memory(8, 4)
        port = self.mem.get_port(write_capable=True)
        self.specials += self.mem, port
        self.comb += [port.adr.eq(self.i[:2]), port.dat_w.eq(self.i), port.we.eq(1)]

def _verilog(tmp_path):
    dut = _DUT()
    path = tmp_path/"dut.v"
    verilog.convert(dut, ios={dut.i, dut.ctrl, dut.comb_only, dut.synced},
                    name="dut").write(str(path))
    return path

def test_classification(tmp_path):
    text = _verilog(tmp_path).read_text()
    clocked, reset = randinit.clocked_and_reset(text)
    assert "ctrl" in reset and "data" in clocked and "data" not in reset
    assert "comb_only" not in clocked

def test_randomize_only_never_reset_registers(tmp_path):
    path = _verilog(tmp_path)
    before = path.read_text()
    n, bits = randinit.randomize(path, seed=1)
    after = path.read_text()
    assert (n, bits) == (1, 8)                      # data only.
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(changed) == 1 and "data" in changed[0][1]

def test_synchronizer_stages_untouched():
    # In a SoC's Verilog the MultiReg stages carry their instance name and no attribute.
    text = ("reg [4:0] builder_multiregimpl3_regs1 = 5'd0;\n"
            "always @(posedge sys_clk) begin\n"
            "\tbuilder_multiregimpl3_regs1 <= x;\n"
            "end\n")
    clocked, reset = randinit.clocked_and_reset(text)
    assert "builder_multiregimpl3_regs1" in clocked
    assert randinit.SYNC.search("builder_multiregimpl3_regs1")

def test_seed_determinism(tmp_path):
    texts = []
    for k in range(2):
        (tmp_path/f"run{k}").mkdir()
        path = _verilog(tmp_path/f"run{k}")
        randinit.randomize(path, seed=5)
        texts.append(path.read_text())
    assert texts[0] == texts[1]

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Random power-on values for the registers no reset reaches (doc/architecture.md, "Resets").

Migen gives every register an initializer (`reg x = 0;`), so a simulation never sees what a
reset-less register holds after a soft reset, or after a reset it is not part of: it starts
at its init value, as the reset would have set it. `randomize(sim.v, seed)` replaces the
initializer of every clocked register that no reset branch assigns with a random value of its
width. A register that needed its reset then shows up as a run that depends on the seed: a
gate mismatch, a hang, a wrong token.

Only those registers change: memories, memory read registers and the registers a reset
assigns keep their values (the FPGA configures BRAMs and flops to their INIT values; what a
reset-less register must survive is a reset that does not touch it). So do two kinds of
registers whose INIT value is their function:
- the reset generators: the power-on reset (the `por` domain's register that starts at 1), the
  reset synchronizers (`resync`) and the reset nets themselves (`*_rst`) - randomized, the
  reset they make never comes;
- the MultiReg synchronizer stages: clock-domain crossings start from INIT by design (an async
  FIFO's gray pointers, a status bit), and a random value there is a phantom crossing at power
  on, not a missing reset.
"""

import random
import re

DECL   = re.compile(r"^(reg\s+(?:signed\s+)?(?:\[(\d+):(\d+)\]\s*)?)(\w+)(\s*=\s*[^;]+);", re.M)
LHS    = re.compile(r"^\s*(\w+)\s*(?:\[|<=)")
CONCAT = re.compile(r"^\s*\{([^}]*)\}\s*<=")
SYNC   = re.compile(r"multiregimpl\d*_regs\d+$") # Migen's MultiReg stages.
RESET  = re.compile(r"(^|_)rst\d*$")                # Reset nets (sys_rst, ddr_rst, eng_rst...).
RESET_DOMAINS = ("por", "resync")                   # Reset generation: power-on, synchronizers.

def _lhs(line):
    m = CONCAT.match(line)
    if m:
        return [n.strip().split("[")[0] for n in m.group(1).split(",")]
    m = LHS.match(line)
    return [m.group(1)] if m and "<=" in line else []

def clocked_and_reset(text):
    """(clocked, reset): the registers assigned in a clocked always block, and those assigned
    in one of its reset branches (`if (<domain>_rst) begin ... end`) or by a reset-generation
    domain (RESET_DOMAINS: they count as reset, their INIT is their function)."""
    clocked, reset = set(), set()
    in_clocked, rst_indent, generator = False, None, False
    for line in text.splitlines():
        if line.startswith("always @(posedge"):
            in_clocked = True
            generator  = any(line.startswith(f"always @(posedge {d}_clk") for d in RESET_DOMAINS)
            continue
        if line.startswith("always"):
            in_clocked = False
            continue
        if not in_clocked:
            continue
        if line == "end":
            in_clocked, rst_indent = False, None
            continue
        stripped = line.lstrip()
        indent   = len(line) - len(stripped)
        if rst_indent is None and re.match(r"if \(\w+_rst\) begin", stripped):
            rst_indent = indent
            continue
        if rst_indent is not None and indent == rst_indent and stripped.startswith("end"):
            rst_indent = None
            continue
        for name in _lhs(line):
            (reset if rst_indent is not None or generator else clocked).add(name)
    return clocked, reset

def randomize(path, seed):
    """Give every clocked, never-reset register of the Verilog at `path` a random initial value
    (`seed`). Returns (randomized registers, randomized bits)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    clocked, reset = clocked_and_reset(text)
    rng   = random.Random(seed)
    count = [0, 0]
    def sub(m):
        name = m.group(4)
        if name not in clocked or name in reset or SYNC.search(name) or RESET.search(name):
            return m.group(0)
        width = int(m.group(2)) - int(m.group(3)) + 1 if m.group(2) else 1
        count[0] += 1
        count[1] += width
        return f"{m.group(1)}{name} = {width}'d{rng.getrandbits(width)};"
    text = DECL.sub(sub, text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return tuple(count)

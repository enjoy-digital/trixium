#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Slow end-to-end check of the YPCB simulation (./trixium_sim.py --board ypcb): the Qwen3 path
on the card's memory system - two LiteDRAM DDR3 controllers with the S7HRDDRPHY's latencies, the
board's timings and controller settings, four engine channels at two bank-skewed windows per
controller, the ddr clock at 8/9 of sys. The firmware's self-test checksums against the golden
model + image bytes, the generated token ids against the hybrid reference (model/mirror.py), and
the engine round trip through every channel's window (ddrtest).

  python3 -m pytest test/system/test_sim_ypcb.py --runslow
"""

import os
import re
import sys
import time
import subprocess

import numpy as np
import pytest

from model import bitexact as bx
from model.bons import BONSReader
from model.mirror import hybrid_generate

IMAGE   = os.environ.get("TRIXIUM_SIM_IMAGE", "models/tiny/bonsai-tiny-n4.img")
OUT_DIR = os.path.join("build", "sim_ypcb_" + os.path.splitext(os.path.basename(IMAGE))[0])
SIM_DIR = os.path.join(OUT_DIR, "gateware")


def fnv1a(data):
    h = 2166136261
    for b in bytes(data):
        h = ((h ^ b)*16777619) & 0xFFFFFFFF
    return h


def x_pattern(n):
    x = np.array([((i*137 + 11) & 0xFF) for i in range(n)], dtype=np.uint8)
    return x.view(np.int8)


def run_console(commands, timeout=40*60):
    """Boot the simulation to TRIXIUM-READY, then send each console command; returns the text."""
    p = subprocess.Popen(["./obj_dir/Vsim"], cwd=SIM_DIR, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, sent, deadline = [], False, time.time() + timeout
    try:
        for line in p.stdout:
            lines.append(line)
            if not sent and "TRIXIUM-READY" in line:
                for c in commands:
                    p.stdin.write(c + "\n")
                p.stdin.flush()
                sent = True
            if sent and sum(l.count("DDRTEST chan") for l in lines) >= 4:
                break
            if time.time() > deadline:
                break
    finally:
        p.kill()
    return "".join(lines)


@pytest.mark.slow
def test_sim_ypcb():
    assert os.path.exists(IMAGE)
    if not os.path.exists(os.path.join(SIM_DIR, "obj_dir/Vsim")):
        subprocess.run([sys.executable, "trixium_sim.py", "--board", "ypcb", "--image", IMAGE,
            "--out-dir", OUT_DIR, "--build"], check=True)
    out = run_console(["ddrtest"])
    assert "SPTEST OK" in out, out[-2000:]
    assert "VECTEST OK" in out, out[-2000:]

    # Self-test checksums: the first tensor's matvec and its first beats through channel 0.
    img = BONSReader(IMAGE)
    e   = img.by_name["blk.0.attn_q.weight"]
    scales, codes = img.tensor("blk.0.attn_q.weight")
    res  = bx.matvec_ternary(scales, codes, x_pattern(e["cols"]), e["bias"], e["rshift"])
    want = f"MATVEC CSUM={fnv1a(res.astype('<i4').tobytes()):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"
    data = bytes(img.tensor_channel_bytes("blk.0.attn_q.weight", 0)[:8*32])
    want = f"FETCH CSUM={fnv1a(data):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"

    # Generated tokens vs the hybrid reference, both legs.
    m = re.search(r"GEN:((?: \d+)+)", out)
    assert m, f"no GEN line in:\n{out[-1500:]}"
    got = [int(t) for t in m.group(1).split()]
    assert got == hybrid_generate(img, [1, 2, 3], len(got))
    mhw = re.search(r"GEN-HW:((?: \d+)+)", out)
    assert mhw, f"no GEN-HW line in:\n{out[-1500:]}"
    hw = [int(t) for t in mhw.group(1).split()]
    assert hw == hybrid_generate(img, [1, 2, 3], len(hw),
        modes={"silu_mul": "hw", "rmsnorm": "hw", "kv_hbm": "hw"})

    # Every channel's window, both controllers (a wrong chan_mod/stride/skew aliases them).
    for c in range(4):
        assert re.search(rf"DDRTEST chan +{c} \(controller {c//2}\) OK", out), out[-1500:]

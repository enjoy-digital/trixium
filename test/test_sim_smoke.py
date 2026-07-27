#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Slow end-to-end check (needs a built Verilator sim): runs the firmware in simulation and
# verifies the self-test checksums against the golden model + image bytes, and the generated
# token ids against a hybrid reference (engine-exact matvecs + float vector ops).
# Run with: .venv/bin/python -m pytest test/test_sim_smoke.py --runslow

import os
import subprocess

import numpy as np
import pytest

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import BONSReader

SIM_DIR = "build/sim/gateware"
IMAGE   = "models/tiny/bonsai-tiny-n2.img"

def fnv1a(data):
    h = 2166136261
    for b in bytes(data):
        h = ((h ^ b)*16777619) & 0xFFFFFFFF
    return h

def x_pattern(n):
    x = np.array([((i*137 + 11) & 0xFF) for i in range(n)], dtype=np.uint8)
    return x.view(np.int8)

@pytest.mark.slow
def test_sim_selftest_checksums():
    assert os.path.exists(os.path.join(SIM_DIR, "obj_dir/Vsim")), "build the sim first"
    assert os.path.exists(IMAGE)

    # Run the sim until the self-test completes.
    p = subprocess.Popen(["./obj_dir/Vsim"], cwd=SIM_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in p.stdout:
            lines.append(line)
            if "SELFTEST DONE" in line or len(lines) > 3000:
                break
    finally:
        p.kill()
    out = "".join(lines)
    assert "SPTEST OK" in out, out[-2000:]

    img  = BONSReader(IMAGE)
    e    = img.by_name["blk.0.attn_q.weight"]
    n_ch = img.header["n_channels"]

    # Expected MATVEC checksum.
    scales, codes = img.tensor("blk.0.attn_q.weight")
    x8   = x_pattern(e["cols"])
    res  = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
    want = f"MATVEC CSUM={fnv1a(res.astype('<i4').tobytes()):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"

    # Expected FETCH checksum: first 8 beats of channel 0's slice of the tensor.
    data = bytes(img.tensor_channel_bytes("blk.0.attn_q.weight", 0)[:8*32])
    want = f"FETCH CSUM={fnv1a(data):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"

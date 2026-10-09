#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Slow end-to-end check of the Qwen3 path in the FK33 simulation: the firmware's self-test
checksums against the golden model + image bytes, and the generated token ids against a hybrid
reference (model/mirror.py: engine-exact matvecs + float vector ops).

  python3 -m pytest test/system/test_sim_smoke.py --runslow
"""

import os
import sys
import subprocess

import numpy as np
import pytest

from model import layout
from model import bitexact as bx
from model.bons import BONSReader
from model.mirror import hybrid_generate

# The Qwen3 path (1.7B/8B) in the FK33's simulation (the gate, ./trixium_sim.py --gate, covers
# the hybrid 27B path on the XCU1525's). Built here when missing (~3 min); another image:
# TRIXIUM_SIM_IMAGE=... (rebuilt when the image changes: the image is baked in).
IMAGE   = os.environ.get("TRIXIUM_SIM_IMAGE", "models/tiny/bonsai-tiny-n2.img")
OUT_DIR = os.path.join("build", "sim_fk33_" + os.path.splitext(os.path.basename(IMAGE))[0])
SIM_DIR = os.path.join(OUT_DIR, "gateware")

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
    assert os.path.exists(IMAGE)
    if not os.path.exists(os.path.join(SIM_DIR, "obj_dir/Vsim")):
        subprocess.run([sys.executable, "trixium_sim.py", "--board", "fk33", "--image", IMAGE,
            "--out-dir", OUT_DIR, "--build"], check=True)

    # Run the sim until the boot generation demo completes (the token gate needs the GEN
    # line, which lands ~50 sim-seconds / ~5 min wall after SELFTEST).
    p = subprocess.Popen(["./obj_dir/Vsim"], cwd=SIM_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    import time as _t
    deadline = _t.time() + 40*60
    try:
        for line in p.stdout:
            lines.append(line)
            if "TRIXIUM-READY" in line or len(lines) > 400000 or _t.time() > deadline:
                break
    finally:
        p.kill()
    out = "".join(lines)
    assert "SPTEST OK" in out, out[-2000:]
    assert "VECTEST OK" in out, "vec unit smoke failed (emission-level datapath break)"

    # Per-program CRC sweep vs the interpreter (named failure per program).
    import re as _re
    from model import asm as _vasm
    sp = bytearray(256*1024)
    seed = 0x12345678
    for i in range(1024):
        seed = (seed*1664525 + 1013904223) & 0xFFFFFFFF
        sp[0x8000 + 4*i:0x8000 + 4*i + 4] = seed.to_bytes(4, "little")
    # The destination is explicit: it is not always A2. rope takes A2 as its *cos table* and
    # writes via A5/A6, so checksumming A2 read an input rather than the result.
    _progs = [
        ("quant8",   "quant8",     [0x8000, 1, 0xA000, 8], 0xA000),   # A1 = 1: measured.
        ("quant8f",  "quant8",     [0x8000, 0, 0xB200, 8], 0xB200),   # A1 = 0: fixed >>8.
        ("norm_i32", "norm_i32",   [0x8000, 0, 0xA200, 4, 13], 0xA200),
        ("resid",    "resid_add",  [0x8000, 0x8400, 0xA400, 8, 1, 2], 0xA400),
        ("silu",     "silu_mul",   [0x8000, 0x8400, 0xA600, 8, -6, 0, -15], 0xA600),
        ("rmsnorm",  "rmsnorm",    [0x8000, 0x8400, 0xA800, 8, -7, 17, -17], 0xA800),
        ("rope",     "rope",       [0x8000, 0x8080, 0x8400, 0x8480, 4, 0xAA00, 0xAA80], 0xAA00),
        ("softmax8", "softmax8",   [0x8000, 739, 0xAC00, 4, -11, 17, 0xB400], 0xAC00),
        ("reduce",   "reduce_acc", [0xA200, 0x8400, 0xAE00, 4], 0xAE00),
        ("scale8",   "scale8",     [0x8000, 0, 0xB000, 8, 19571], 0xB000),
    ]
    exp_crc = {}
    for name, prog, pargs, pdst in _progs:
        # Same clear the firmware does: the CRC window is wider than what any program
        # writes, so without it the checksum folds in scratchpad history.
        sp[pdst:pdst + 512] = bytes(512)
        S, _ = _vasm.VecInterp(sp).run(_vasm.assemble(_vasm.PROGRAMS[prog]()), list(pargs))
        crc = 0
        for i in range(128):
            w = int.from_bytes(sp[pdst + 4*i:pdst + 4*i + 4], "little")
            crc = ((crc ^ w)*16777619) & 0xFFFFFFFF
        exp_crc[name] = (crc, S[6] & 0xFFFFFFFF)
    for m2 in _re.finditer(r"VECCRC (\w+) ([0-9a-f]{8}) ([0-9a-f]{8})", out):
        name, crc, r6 = m2.group(1), int(m2.group(2), 16), int(m2.group(3), 16)
        assert name in exp_crc, name
        assert crc == exp_crc[name][0], (
            f"vec program '{name}' diverged from the interpreter "
            f"(crc {crc:08x} != {exp_crc[name][0]:08x}) - emission-level break")
    assert len(_re.findall(r"VECCRC", out)) == len(_progs), "missing VECCRC lines"

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

    # Generated tokens vs the hybrid reference (engine-exact matvecs + float vector ops).
    import re
    m = re.search(r"GEN:((?: \d+)+)", out)
    assert m, f"no GEN line in:\n{out[-1500:]}"
    got_tokens = [int(t) for t in m.group(1).split()]
    ref_tokens = hybrid_generate(img, [1, 2, 3], len(got_tokens))
    assert got_tokens == ref_tokens, f"sim {got_tokens} != ref {ref_tokens}"

    # HW-ops leg (present when the sim SoC has the vector unit): same greedy tokens. The
    # quantized HW ops are not bit-identical to soft-float, but on every model tested the
    # argmax agrees; a divergence here is a regression signal to investigate either way.
    mhw = re.search(r"GEN-HW:((?: \d+)+)", out)
    assert mhw, f"no GEN-HW line in:\n{out[-1500:]}"
    hw_tokens = [int(t) for t in mhw.group(1).split()]
    ref_hw    = hybrid_generate(img, [1, 2, 3], len(hw_tokens), modes={"silu_mul": "hw", "rmsnorm": "hw", "kv_hbm": "hw"})
    assert hw_tokens == ref_hw, f"HW-ops tokens {hw_tokens} != hw ref {ref_hw}"

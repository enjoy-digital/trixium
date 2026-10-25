#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Slow end-to-end check of the YPCB simulation (./trixium_sim.py --board ypcb): the Qwen3 path
on the card's memory system - two LiteDRAM DDR3 controllers with the K7DDRPHY's latencies, the
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


@pytest.mark.slow
def test_sim_ypcb_calibration():
    """The board's DDR3 calibration (liblitedram's CK phase leveling) in the SoC: the PHY models
    behind the K7DDRPHY's CK phase CSRs (CKPhasePHYModel, --ckphase-model), controller 1 a fly-by
    channel that needs it (trixium_sim.CKPHASE_CHANNEL), calibrated from the console through the
    CSR window and the BIST ports as on the card; then every channel's window and a generation
    must be right."""
    import trixium_sim as ts
    out_dir = os.path.join("build", "sim_ypcb_ckphase")
    sim_dir = os.path.join(out_dir, "gateware")
    if not os.path.exists(os.path.join(sim_dir, "obj_dir/Vsim")):
        subprocess.run([sys.executable, "trixium_sim.py", "--board", "ypcb", "--ckphase-model",
            "--image", IMAGE, "--out-dir", out_dir, "--build"], check=True)
    global SIM_DIR
    saved, SIM_DIR = SIM_DIR, sim_dir
    try:
        out = run_console(["ddrinit 1", "ddrtest"], timeout=90*60)
    finally:
        SIM_DIR = saved
    assert "  OK (BIST" in out and "ddr: controller 1 OK" in out, out[-3000:]

    # The phase: inside the model's common window (every lane outside its crossing zone) and
    # each lane at the write latency the model needs there.
    ch = ts.CKPHASE_CHANNEL
    m  = re.search(r"  CK phase ([+-]\d+) \(run (\d+) steps\), write bitslips((?: \d)+)", out)
    assert m, out[-3000:]
    ck  = int(m.group(1))
    wls = [int(w) for w in m.group(3).split()]
    for l, (cross, base) in enumerate(zip(ch["cross"], ch["base_wl"])):
        assert abs(ck - cross) > 3, (ck, l)
        assert wls[l] == ((base + 2) & 7 if ck >= cross else base), (ck, l, wls)
    for c in range(4):
        assert re.search(rf"DDRTEST chan +{c} \(controller {c//2}\) OK", out), out[-1500:]


# The hybrid path (Ternary-Bonsai-2: the DeltaNet and attention streams, the scratchpad units, the
# resident firmware) on the YPCB simulation, through the gate of software/lab/simgate.py: the
# pre-generation's tokens against the mirror, the traced generation's checkpoints against it, the
# list image cache verified. Parametrised over the board's profile (4 channels, 2 controllers, the
# shipping units) and the small boards' lite profile at 2 and 1 channels on one controller.
HYBRID_CASES = [
    pytest.param("models/tiny/bonsai-tiny-hybrid-1k-t5-n4.img", 2, "full", id="n4-full"),
    pytest.param("models/tiny/bonsai-tiny-hybrid-1k-t5-n2.img", 1, "lite", id="n2-lite-1ctrl"),
    pytest.param("models/tiny/bonsai-tiny-hybrid-1k-t5-n1.img", 1, "lite", id="n1-lite-1ctrl"),
]

@pytest.mark.slow
@pytest.mark.parametrize("image,memctrl,profile", HYBRID_CASES)
def test_sim_ypcb_hybrid_gate(image, memctrl, profile):
    assert os.path.exists(image), image
    stem = os.path.splitext(os.path.basename(image))[0]
    out_dir = os.path.join("build", f"sim_ypcb_{stem}_{profile}_m{memctrl}")
    common = [sys.executable, "trixium_sim.py", "--board", "ypcb", "--hybrid", "--image", image,
              "--sys-clk-freq", "100e6", "--prefetch", "512", "--pipe-clk-freq", "125e6",
              "--memctrl", str(memctrl), "--profile", profile, "--out-dir", out_dir]
    if not os.path.exists(os.path.join(out_dir, "gateware", "obj_dir", "Vsim")):
        subprocess.run(common + ["--build"], check=True)
    r = subprocess.run(common + ["--gate", "--gate-pre", "listcache verify;gen 4 1 2 3;stats"],
                       capture_output=True, text=True, timeout=60*60)
    assert r.returncode == 0 and "GATE PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-1500:]


# Two conversations decoded together (`gen2`, --matvec-rows 2) with the two-row LM head on an LM
# head of real-model chunk sizes: the tiny image's 512-row vocabulary is one 512-row chunk (and
# `lmchunk 128` only shrinks it), where the 2B's head runs 22,016-row chunks and a ragged tail.
# The image here has a 24,000-row vocabulary: an automatic chunk of 22,016 rows (5,504 per
# channel, the 2B's: row 1's int32 staged up to 0x2F800) and a ragged 1,984-row tail, then
# `lmchunk 2048`: twelve chunks with a ragged 1,472-row tail. Each stream must be the mirror's
# tokens for its own prompt (the gate's gen2 check), the per-row heads (`lmhead off`) too.
GEN2_VOCAB = 24000

@pytest.mark.slow
def test_sim_ypcb_gen2_lmhead_chunks():
    from model import tinygen
    image = f"models/tiny/bonsai-tiny-hybrid-1k-v{GEN2_VOCAB}-t5-n4.img"
    if not os.path.exists(image):
        tinygen.generate("models/tiny", 4, cfg=tinygen.BONSAI_TINY_HYBRID_1K, trit5=True,
                         vocab=GEN2_VOCAB)
    out_dir = os.path.join("build", f"sim_ypcb_mr2_v{GEN2_VOCAB}")
    common = [sys.executable, "trixium_sim.py", "--board", "ypcb", "--hybrid", "--image", image,
              "--sys-clk-freq", "100e6", "--prefetch", "512", "--pipe-clk-freq", "125e6",
              "--memctrl", "2", "--matvec-rows", "2", "--out-dir", out_dir]
    if not os.path.exists(os.path.join(out_dir, "gateware", "obj_dir", "Vsim")):
        subprocess.run(common + ["--build"], check=True)
    pre = ("listcache verify;gen2 lmhead on;gen2 4 1 2 3 | 5 6;lmchunk 2048;gen2 4 5 6 | 1 2 3;"
           "gen2 lmhead off;gen2 3 1 2 3 | 5 6;lmchunk 0;stats")
    r = subprocess.run(common + ["--gate", "--gate-pre", pre],
                       capture_output=True, text=True, timeout=3*60*60)
    assert r.returncode == 0 and "GATE PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-1500:]
    assert r.stdout.count("pre: gen2 ") >= 3 and "gen2 stream B" in r.stdout, r.stdout[-3000:]

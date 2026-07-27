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

    # Run the sim until the boot generation demo completes (the token gate needs the GEN
    # line, which lands ~50 sim-seconds / ~5 min wall after SELFTEST).
    p = subprocess.Popen(["./obj_dir/Vsim"], cwd=SIM_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in p.stdout:
            lines.append(line)
            if "GEN DONE" in line or "BONSAI-READY" in line or len(lines) > 20000:
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

    # Generated tokens vs the hybrid reference (engine-exact matvecs + float vector ops).
    import re
    m = re.search(r"GEN:((?: \d+)+)", out)
    assert m, f"no GEN line in:\n{out[-1500:]}"
    got_tokens = [int(t) for t in m.group(1).split()]
    ref_tokens = hybrid_generate(img, [1, 2, 3], len(got_tokens))
    assert got_tokens == ref_tokens, f"sim {got_tokens} != ref {ref_tokens}"

def hybrid_generate(img, prompt, n_new):
    """Python mirror of the firmware loop: engine-bitexact matvecs, float32 vector ops."""
    h    = img.header
    f32  = np.float32
    n_ch = h["n_channels"]

    def entry(name):
        return img.by_name[name]

    def tensor(name):
        scales, values = img.tensor(name)
        return scales, values

    def matvec(name, x):
        e = entry(name)
        scales, codes = tensor(name)
        amax = np.abs(x).max()
        s    = f32(amax/127.0) if amax > 0 else f32(1.0)
        x8   = np.clip(np.round(x/s), -127, 127).astype(np.int8)
        res  = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
        return (res.astype(np.float64)*np.exp2(e["rshift"] - e["bias"])*s).astype(f32)

    def rmsnorm(x, w, eps):
        ss = f32((x.astype(f32)**2).sum())
        r  = f32(1.0)/np.sqrt(ss/f32(len(x)) + f32(eps))
        return (x*r*w).astype(f32)

    def norm_w(name):
        _, v = tensor(name)
        return v.reshape(-1).astype(f32)

    n_layer, n_embd  = h["n_layer"], h["n_embd"]
    n_head, n_kv     = h["n_head"], h["n_kv_head"]
    hd               = h["head_dim"]
    eps              = h["rms_eps"]
    emb_s, emb_c     = tensor("token_embd.weight")

    kc, vc, ks, vs = [], [], [], []  # Per layer lists of per-pos entries.
    for _ in range(n_layer):
        kc.append([]); vc.append([]); ks.append([]); vs.append([])

    def forward(tok, pos, want):
        x = ((emb_c[tok].astype(np.int32) - 1)
             .reshape(-1, 128)*emb_s[tok].astype(f32)[:, None]).reshape(-1).astype(f32)
        for l in range(n_layer):
            p  = f"blk.{l}."
            hh = rmsnorm(x, norm_w(p + "attn_norm.weight"), eps)
            q  = matvec(p + "attn_q.weight", hh)
            k  = matvec(p + "attn_k.weight", hh)
            v  = matvec(p + "attn_v.weight", hh)
            qn, kn = norm_w(p + "attn_q_norm.weight"), norm_w(p + "attn_k_norm.weight")
            q = np.concatenate([rmsnorm(q[i*hd:(i+1)*hd], qn, eps) for i in range(n_head)])
            k = np.concatenate([rmsnorm(k[i*hd:(i+1)*hd], kn, eps) for i in range(n_kv)])
            # Plain rope (tiny model: no yarn), float32 like the firmware.
            for arr, nh in ((q, n_head), (k, n_kv)):
                for i in range(nh):
                    vh = arr[i*hd:(i+1)*hd]
                    for p2 in range(hd//2):
                        theta = f32(pos*np.power(f32(h["rope_theta"]), f32(-2.0*p2/hd)))
                        c, s = np.cos(theta, dtype=f32), np.sin(theta, dtype=f32)
                        x0, x1 = vh[p2], vh[p2 + hd//2]
                        vh[p2]         = x0*c - x1*s
                        vh[p2 + hd//2] = x0*s + x1*c
            # int8 KV append.
            ka = np.abs(k).max(); kscale = f32(ka/127.0) if ka > 0 else f32(1.0)
            va = np.abs(v).max(); vscale = f32(va/127.0) if va > 0 else f32(1.0)
            kc[l].append(np.round(k/kscale).astype(np.int8)); ks[l].append(kscale)
            vc[l].append(np.round(v/vscale).astype(np.int8)); vs[l].append(vscale)
            # Attention.
            n_rep = n_head//n_kv
            attn  = np.zeros(n_head*hd, dtype=f32)
            for i in range(n_head):
                kvh = i//n_rep
                qh  = q[i*hd:(i+1)*hd]
                sc  = np.array([f32((qh*kt[kvh*hd:(kvh+1)*hd]).sum())*kst
                                for kt, kst in zip(kc[l], ks[l])], dtype=f32)
                sc  = sc*f32(1.0/np.sqrt(hd))
                e   = np.exp(sc - sc.max(), dtype=f32)
                pr  = e/e.sum()
                ah  = np.zeros(hd, dtype=f32)
                for t in range(len(vc[l])):
                    ah += pr[t]*vs[l][t]*vc[l][t][kvh*hd:(kvh+1)*hd].astype(f32)
                attn[i*hd:(i+1)*hd] = ah
            x = x + matvec(p + "attn_output.weight", attn)
            hh   = rmsnorm(x, norm_w(p + "ffn_norm.weight"), eps)
            g    = matvec(p + "ffn_gate.weight", hh)
            u    = matvec(p + "ffn_up.weight", hh)
            act  = (g/(f32(1.0) + np.exp(-g, dtype=f32))*u).astype(f32)
            x    = x + matvec(p + "ffn_down.weight", act)
        if not want:
            return -1
        hh   = rmsnorm(x, norm_w("output_norm.weight"), eps)
        head = "output.weight" if "output.weight" in img.by_name else "token_embd.weight"
        e    = entry(head)
        scales, codes = tensor(head)
        amax = np.abs(hh).max()
        s    = f32(amax/127.0) if amax > 0 else f32(1.0)
        x8   = np.clip(np.round(hh/s), -127, 127).astype(np.int8)
        res  = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
        res  = res[:h["vocab"]]
        return int(np.argmax(res)) # Greedy on raw int32 (monotonic in the float logits).

    pos = 0
    tok = -1
    for i, t in enumerate(prompt):
        tok = forward(t, pos, i == len(prompt) - 1)
        pos += 1
    out = []
    for _ in range(n_new):
        out.append(tok)
        tok = forward(tok, pos, True)
        pos += 1
    return out

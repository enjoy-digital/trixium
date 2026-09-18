#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The Python mirror of firmware/hybrid.c - the hybrid models' CPU path.

hybrid27_generate walks a BONS v3 image exactly as hybrid.c does: every matvec through the
engine's bit-exact model (int8 input with a power-of-two scale, int32 results, float
scaling), everything else in float32 in the firmware's operation order (sequential sums).
It is what tools/check_sim_tokens.py compares the sim's GEN legs against when the image is
a hybrid, and the test below pins it to the fp32 golden on the tiny hybrid model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import BONSReader

f32 = np.float32


def _fsum32(v):
    acc = f32(0.0)
    for e in np.asarray(v, dtype=f32):
        acc = f32(acc + e)
    return acc


def _lroundf(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def hybrid27_generate(img, prompt, n_new, exact=False):
    """exact=True: fp32 weights and no activation quantization (the golden's arithmetic),
    which tests the mirror's structure against the golden independently of the int8 noise."""
    h = img.header
    assert h["version"] >= 3 and (any(t == "deltanet" for t in h["layer_types"]) or h["hadamard_block"])
    n_layer, n_embd, ffn_dim = h["n_layer"], h["n_embd"], h["ffn_dim"]
    n_head, n_kv, hd = h["n_head"], h["n_kv_head"], h["head_dim"]
    nk, hk, nv, hv = h["dn_num_k_heads"], h["dn_k_head_dim"], h["dn_num_v_heads"], h["dn_v_head_dim"]
    nq, nvh, K = nk*hk, nv*hv, h["dn_conv_kernel"]
    rope_n = h["rope_dims"] or hd
    eps = f32(h["rms_eps"])
    B = h["hadamard_block"]
    widths = list(h["hadamard_widths"])
    gated = bool(h["attn_gated"])
    q_dim, kv_dim, n_rep = n_head*hd, n_kv*hd, n_head//n_kv

    def entry(role, l, kind=None):
        for e in img.entries:
            if e["role"] == role and e["layer"] == l and (kind is None or e["kind"] == kind):
                return e
        return None

    def f32_rows(role, l):
        e = entry(role, l, layout.KIND_F32)
        assert e is not None, (role, l)
        _, v = img.tensor(e["name"])
        return np.asarray(v, dtype=f32)

    def quantize_f_pow2(x):
        amax = np.abs(x).max()
        if amax == 0:
            return np.zeros(len(x), dtype=np.int8), 0
        e8 = int(np.frexp(f32(amax))[1]) - 7
        return np.clip(_lroundf(x*f32(np.exp2(-e8))), -127, 127).astype(np.int8), e8

    wcache = {}

    def matvec_f(e, x):
        if exact:
            if e["name"] not in wcache:
                scales, codes = img.tensor(e["name"])
                g = 128 if e["kind"] == layout.KIND_TERNARY else 32
                off = 1.0 if e["kind"] == layout.KIND_TERNARY else 0.0
                w = (codes.astype(f32).reshape(e["rows"], -1, g) - f32(off))*scales.astype(f32)[:, :, None]
                wcache[e["name"]] = w.reshape(e["rows"], e["cols"])
            return (wcache[e["name"]] @ np.asarray(x, dtype=f32)).astype(f32)
        x8, e8 = quantize_f_pow2(np.asarray(x, dtype=f32))
        scales, codes = img.tensor(e["name"])
        if e["kind"] == layout.KIND_TERNARY:
            res = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
        else:
            res = bx.matvec_q8(scales, codes, x8, e["bias"], e["rshift"])
        sc = f32(np.exp2(f32(e8 + e["rshift"] - e["bias"])))
        return (res.astype(f32)*sc).astype(f32)

    def rotate(x, width):
        if not B:
            return np.asarray(x, dtype=f32)
        wi = widths.index(width)
        out = np.empty(width, dtype=f32)
        for b in range(width//B):
            e = entry(layout.ROLE_HADAMARD, layout.hadamard_layer(wi, b))
            out[b*B:(b + 1)*B] = matvec_f(e, x[b*B:(b + 1)*B])
        return out

    def proj(role, l, x):
        e = entry(role, l)
        return matvec_f(e, rotate(x, e["cols"]) if B else x)

    def rmsnorm(x, w):
        ss = _fsum32(x*x)
        r = f32(1.0)/f32(np.sqrt(f32(ss/f32(len(x))) + eps))
        return (x*r*w).astype(f32)

    def l2norm(x):
        nrm = f32(np.sqrt(_fsum32(x*x)))
        nrm = max(nrm, eps)
        return (x/nrm).astype(f32)

    def sigmoid(x): return (f32(1.0)/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    def silu(x):    return (x/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x, dtype=f32), dtype=f32)).astype(f32)

    # Tables.
    norm_attn = [f32_rows(layout.ROLE_ATTN_NORM, l).reshape(-1) for l in range(n_layer)]
    norm_ffn  = [f32_rows(layout.ROLE_FFN_NORM, l).reshape(-1) for l in range(n_layer)]
    dn = [t == "deltanet" for t in h["layer_types"]]
    norm_q, norm_k, norm_ssm, ssm_a, ssm_dt, conv_w = {}, {}, {}, {}, {}, {}
    for l in range(n_layer):
        if dn[l]:
            norm_ssm[l] = f32_rows(layout.ROLE_SSM_NORM, l).reshape(-1)
            ssm_a[l]    = f32_rows(layout.ROLE_SSM_A, l).reshape(-1)
            ssm_dt[l]   = f32_rows(layout.ROLE_SSM_DT, l).reshape(-1)
            conv_w[l]   = f32_rows(layout.ROLE_SSM_CONV, l).reshape(-1)   # c*K + i
        else:
            norm_q[l] = f32_rows(layout.ROLE_ATTN_Q_NORM, l).reshape(-1)
            norm_k[l] = f32_rows(layout.ROLE_ATTN_K_NORM, l).reshape(-1)
    out_norm = f32_rows(layout.ROLE_OUT_NORM, 0).reshape(-1)
    emb16  = entry(layout.ROLE_TOK_EMB16, 0)
    emb16x = entry(layout.ROLE_TOK_EMB16_EXP, 0)
    _, mant = img.tensor(emb16["name"])
    _, exps = img.tensor(emb16x["name"])
    exps = np.asarray(exps).reshape(-1)
    rope_freq = np.array([f32(np.float32(h["rope_theta"])**f32(-2.0*i/rope_n)) for i in range(rope_n//2)], dtype=f32)

    # State.
    state = {l: np.zeros((nv, hv, hk), dtype=f32) for l in range(n_layer) if dn[l]}
    win   = {l: np.zeros((K - 1, 2*nq + nvh), dtype=f32) for l in range(n_layer) if dn[l]}
    kc    = {l: [] for l in range(n_layer) if not dn[l]}
    vc    = {l: [] for l in range(n_layer) if not dn[l]}
    scale_dn = f32(1.0/np.sqrt(hk))
    scale_at = f32(1.0/np.sqrt(hd))

    def deltanet(l, hn):
        C = 2*nq + nvh
        qkv = proj(layout.ROLE_ATTN_QKV, l, hn)
        z   = proj(layout.ROLE_ATTN_ZGATE, l, hn)
        beta  = sigmoid(matvec_f(entry(layout.ROLE_SSM_BETA, l), hn))
        alpha = matvec_f(entry(layout.ROLE_SSM_ALPHA, l), hn)
        g = (softplus(alpha + ssm_dt[l])*ssm_a[l]).astype(f32)
        w = conv_w[l]
        conv = np.empty(C, dtype=f32)
        for c in range(C):
            acc = f32(0.0)
            for i in range(K - 1):
                acc = f32(acc + f32(w[c*K + i]*win[l][i, c]))
            acc = f32(acc + f32(w[c*K + K - 1]*qkv[c]))
            conv[c] = acc
        conv = silu(conv)
        win[l][:-1] = win[l][1:]
        win[l][-1] = qkv
        q = conv[:nq].copy(); k = conv[nq:2*nq].copy(); v = conv[2*nq:].copy()
        for hh in range(nk):
            q[hh*hk:(hh + 1)*hk] = l2norm(q[hh*hk:(hh + 1)*hk])
            k[hh*hk:(hh + 1)*hk] = l2norm(k[hh*hk:(hh + 1)*hk])
        rep = nv//nk
        og = np.zeros(nvh, dtype=f32)
        for hh in range(nv):
            kh = hh % nk
            qh, kk, vh = q[kh*hk:(kh + 1)*hk], k[kh*hk:(kh + 1)*hk], v[hh*hv:(hh + 1)*hv]
            M = state[l][hh]
            decay = f32(np.exp(g[hh], dtype=f32))
            delta = np.empty(hv, dtype=f32)
            for j in range(hv):
                Mj = M[j]
                dot = f32(0.0)
                for i in range(hk):
                    Mj[i] = f32(Mj[i]*decay)
                    dot = f32(dot + f32(Mj[i]*kk[i]))
                delta[j] = f32(f32(vh[j] - dot)*beta[hh])
            oh = np.empty(hv, dtype=f32)
            for j in range(hv):
                Mj = M[j]
                dot = f32(0.0)
                for i in range(hk):
                    Mj[i] = f32(Mj[i] + f32(delta[j]*kk[i]))
                    dot = f32(dot + f32(Mj[i]*qh[i]))
                oh[j] = f32(dot*scale_dn)
            nrm = rmsnorm(oh, norm_ssm[l])
            gpos = kh*rep + hh//nk
            og[gpos*hv:(gpos + 1)*hv] = (nrm*silu(z[hh*hv:(hh + 1)*hv])).astype(f32)
        return proj(layout.ROLE_SSM_OUT, l, og)

    def attention(l, hn, pos):
        qf = proj(layout.ROLE_ATTN_Q, l, hn)
        k  = proj(layout.ROLE_ATTN_K, l, hn)
        v  = proj(layout.ROLE_ATTN_V, l, hn)
        if gated:
            qf = qf.reshape(n_head, 2*hd)
            q, gate = qf[:, :hd].copy(), qf[:, hd:].copy()
        else:
            q, gate = qf.reshape(n_head, hd).copy(), None
        k = k.reshape(n_kv, hd).copy(); v = v.reshape(n_kv, hd).copy()
        cos = np.cos((f32(pos)*rope_freq).astype(f32), dtype=f32)
        sin = np.sin((f32(pos)*rope_freq).astype(f32), dtype=f32)
        half = rope_n//2
        def rope(vh):
            x0, x1 = vh[:half].copy(), vh[half:rope_n].copy()
            vh[:half]       = (x0*cos - x1*sin).astype(f32)
            vh[half:rope_n] = (x0*sin + x1*cos).astype(f32)
        for hh in range(n_head):
            q[hh] = rmsnorm(q[hh], norm_q[l]); rope(q[hh])
        for hh in range(n_kv):
            k[hh] = rmsnorm(k[hh], norm_k[l]); rope(k[hh])
        kc[l].append(k); vc[l].append(v)
        T = len(kc[l])
        out = np.zeros((n_head, hd), dtype=f32)
        for hh in range(n_head):
            kvh = hh//n_rep
            sc = np.array([f32(_fsum32(q[hh]*kc[l][t][kvh])*scale_at) for t in range(T)], dtype=f32)
            mx = sc.max()
            ex = np.exp((sc - mx).astype(f32), dtype=f32)
            s = _fsum32(ex)
            o = np.zeros(hd, dtype=f32)
            for t in range(T):
                p = f32(ex[t]/s)
                o = (o + p*vc[l][t][kvh]).astype(f32)
            if gated:
                o = (o*sigmoid(gate[hh])).astype(f32)
            out[hh] = o
        return proj(layout.ROLE_ATTN_OUT, l, out.reshape(-1))

    def forward(tok, pos):
        x = (mant[tok].astype(f32)*f32(np.exp2(f32(exps[tok])))).astype(f32)
        for l in range(n_layer):
            hn = rmsnorm(x, norm_attn[l])
            x = (x + (deltanet(l, hn) if dn[l] else attention(l, hn, pos))).astype(f32)
            hn = rmsnorm(x, norm_ffn[l])
            r = rotate(hn, n_embd) if B else hn
            gate = matvec_f(entry(layout.ROLE_FFN_GATE, l), r)
            up   = matvec_f(entry(layout.ROLE_FFN_UP, l), r)
            act  = (silu(gate)*up).astype(f32)
            x = (x + proj(layout.ROLE_FFN_DOWN, l, act)).astype(f32)
        hn = rmsnorm(x, out_norm)
        head = entry(layout.ROLE_LM_HEAD, 0) or entry(layout.ROLE_TOK_EMB, 0)
        logits = proj(layout.ROLE_LM_HEAD if entry(layout.ROLE_LM_HEAD, 0) else layout.ROLE_TOK_EMB, 0, hn)
        return int(np.argmax(logits[:h["vocab"]]))

    out, pos = [], 0
    tok = None
    for t in prompt:
        tok = forward(t, pos); pos += 1
    for _ in range(n_new):
        out.append(tok)
        tok = forward(tok, pos); pos += 1
    return out


def test_mirror_agrees_with_the_fp32_golden_on_the_tiny_hybrid():
    """The mirror's structure against the fp32 golden: run exact (fp32 weights, no activation
    quantization) it must reproduce the golden's tokens on the tiny hybrid model, which pins
    the recurrence, the head order, the gates and the rotation independently of the int8
    noise. The sim then compares the firmware to the quantized mirror exactly."""
    import tempfile
    from trixium.model.configs import BONSAI_TINY_HYBRID
    from trixium.model.tinygen import generate, make_tiny_hybrid_tensors
    from trixium.model.golden import FloatModel
    cfg = BONSAI_TINY_HYBRID
    with tempfile.TemporaryDirectory() as d:
        img_path, _, _ = generate(d, 2, 1234, cfg=cfg)
        img = BONSReader(img_path)
        got = hybrid27_generate(img, [1, 2, 3], 4, exact=True)
        quant = hybrid27_generate(img, [1, 2, 3], 4)
    specs = make_tiny_hybrid_tensors(cfg, 1234)
    t = {}
    for s in specs:
        if s.kind == layout.KIND_TERNARY:
            w = (s.get_values().astype(np.float32).reshape(s.rows, -1, 128) - 1.0)*s.scales.astype(np.float32)[:, :, None]
            t[s.name] = w.reshape(s.rows, s.cols)
        elif s.kind == layout.KIND_Q8:
            w = s.get_values().astype(np.float32).reshape(s.rows, -1, 32)*s.scales.astype(np.float32)[:, :, None]
            t[s.name] = w.reshape(s.rows, s.cols)
        elif s.kind == layout.KIND_F32:
            v = s.get_values()
            t[s.name] = v.reshape(-1, cfg.dn_conv_kernel) if s.role == layout.ROLE_SSM_CONV else v.reshape(-1)
    m = FloatModel(cfg, t)
    want = m.generate([1, 2, 3], 4)
    assert got == want, (got, want)
    # The quantized mirror (what the firmware computes) agrees on the first token at least;
    # later tokens can flip on this random model's small logit margins (0.03 at step 1).
    assert quant[0] == want[0], (quant, want)

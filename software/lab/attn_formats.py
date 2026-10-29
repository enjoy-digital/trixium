#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Attention number formats against perplexity (A2 of doc/plan-core-knowledge.md).

Runs model/fastref.py's hybrid reference over chunks of a text with the attention computed in
several formats, the rest of the model unchanged, and reports the perplexity of each (all
positions, and the positions past 1,024 where the KV is long):

  fp32    the reference (what the attention stream computes today: FP32 K/V, FP32 MACs)
  kv8     K and V stored as int8 with one scale per (position, kv head) row, dequantized:
          the storage saving alone
  hw8     the integer datapath: q int8 per (position, head), K int8 per row, the score an
          exact int32 dot times sq*sk; p' = p*sv folded and quantized to int16 with one scale
          per (query, head), V int8 per row, the values an exact integer sum times that scale
  hw8g32  hw8 with K and V scaled per group of 32 (one 256-bit beat of int8), q8_0-like

    python3 -m software.lab.attn_formats models/bons/bonsai2-2b-n4-ti375.img \\
        --text wiki.test.raw --tokenizer models/2b-v2/tokenizer.json --ctx 2048 --chunks 4
"""

import argparse
import time

import numpy as np

from model.bons import BONSReader
from model.fastref import HybridRef, attention_fp32

f32 = np.float32

def q_rows(x, bits=8, group=None):
    """Symmetric round-to-nearest int quantization along the last axis (per row, or per group
    of `group`) -> (ints as float32, scales broadcastable to x)."""
    lim = (1 << (bits - 1)) - 1
    sh = x.shape
    xg = x.reshape(*sh[:-1], -1, group) if group else x[..., None, :]
    s = np.abs(xg).max(axis=-1, keepdims=True)/lim
    s = np.where(s == 0, 1, s).astype(f32)
    qi = np.clip(np.rint(xg/s), -lim, lim).astype(f32)
    return qi, s

def causal_softmax(s):
    T = s.shape[1]
    s = s + np.triu(np.full((T, T), -np.inf, f32), 1)[None]
    s = s - s.max(axis=2, keepdims=True)
    p = np.exp(s)
    return (p/p.sum(axis=2, keepdims=True)).astype(f32)

def attention_kv8(q, k, v, layer=None):
    kq, ks = q_rows(k); vq, vs = q_rows(v)
    return attention_fp32(q, (kq*ks).reshape(k.shape), (vq*vs).reshape(v.shape))

def attention_hw8(q, k, v, layer=None, group=None, p_bits=16):
    T, nh, hd = q.shape
    n_rep = nh//k.shape[1]
    qq, qs = q_rows(q)                           # [T, nh, 1, hd], [T, nh, 1, 1]
    kq, ks = q_rows(k, group=group)              # [T, nkv, G, g], [T, nkv, G, 1]
    vq, vs = q_rows(v, group=group)
    kq, ks = np.repeat(kq, n_rep, 1), np.repeat(ks, n_rep, 1)
    vq, vs = np.repeat(vq, n_rep, 1), np.repeat(vs, n_rep, 1)
    G = kq.shape[2]
    qg = qq.reshape(T, nh, G, -1)
    # Exact integer dots per group (float64 holds them exactly), scaled per group.
    dots = np.einsum("thgd,shgd->hgts", qg, kq)
    s = (dots*ks[..., 0].transpose(1, 2, 0)[:, :, None, :]).sum(axis=1)*qs[:, :, 0, 0].T[:, :, None]
    p = causal_softmax(s.astype(f32))            # [nh, T, S]
    # p' = p * sv per (key position, group), int16 with one scale per (query, head, group).
    svt = vs[..., 0].transpose(1, 2, 0)          # [nh, G, S]
    pp = (p[:, None]*svt[:, :, None, :]).astype(f32) # [nh, G, T, S]
    lim = (1 << (p_bits - 1)) - 1
    Sp = pp.max(axis=3, keepdims=True)/lim
    Sp = np.where(Sp == 0, 1, Sp)
    pi = np.rint(pp/Sp)
    o = np.einsum("hgts,shgd->thgd", pi.astype(f32), vq)*Sp.transpose(2, 0, 1, 3)
    return o.reshape(T, nh, hd).astype(f32)

def q_pow2(x):
    """int8 per row (last axis) with a power-of-two exponent, mantissas in [64, 128): the
    engine's int8 convention (mirror._quantize_f_pow2) -> (ints as float32, exponents)."""
    amax = np.abs(x).max(axis=-1, keepdims=True)
    e = np.where(amax > 0, np.frexp(amax)[1] - 7, 0)
    v = x*np.exp2(-e)
    qi = np.clip(np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)), -127, 127)
    return qi.astype(f32), e

def attention_hw8p2(q, k, v, layer=None):
    """The unit's integer arithmetic: q, K, V int8 rows with power-of-two exponents; the score
    f32(dot)*2^(eq+ek) (q carries the 1/sqrt(hd) scale here); p' = p*2^(ev-EVmax)*2^(14-E)
    as int16 (E: p's largest exponent per query and head, EVmax: the largest V exponent of
    the slots), the exact integer sum, then f32(acc)*2^(EVmax+E-14)."""
    T, nh, hd = q.shape
    n_rep = nh//k.shape[1]
    qq, eq = q_pow2(q)                        # [T, nh, hd], [T, nh, 1]
    kq, ek = q_pow2(k); vq, ev = q_pow2(v)    # [T, nkv, hd], [T, nkv, 1]
    kq, ek = np.repeat(kq, n_rep, 1), np.repeat(ek, n_rep, 1)
    vq, ev = np.repeat(vq, n_rep, 1), np.repeat(ev, n_rep, 1)
    dots = np.einsum("thd,shd->hts", qq, kq)  # exact: |dot| < 2^24
    s = (dots*np.exp2(eq[:, :, 0].T[:, :, None] + ek[:, :, 0].T[:, None, :])).astype(f32)
    p = causal_softmax(s)                     # [nh, T, S]
    E = np.frexp(p.max(axis=2, keepdims=True))[1] - 1
    evt = ev[:, :, 0].T[:, None, :]           # [nh, 1, S]
    # Causal: the running maximum over the slots present at each query.
    EVmax = np.maximum.accumulate(ev[:, :, 0].T, axis=1)[:, :, None]    # [nh, T, 1]
    pp = p*np.exp2(evt - EVmax + 14 - E)
    pi = np.minimum(np.floor(pp + 0.5), 32767)*(np.tril(np.ones((T, T), f32))[None])
    acc = np.einsum("hts,shd->thd", pi.astype(np.float64), vq.astype(np.float64))
    return (acc*np.exp2(EVmax + E - 14).transpose(1, 0, 2)).astype(f32)

FORMATS = {
    "hw8p2":  attention_hw8p2,
    "fp32":   attention_fp32,
    "kv8":    attention_kv8,
    "hw8":    attention_hw8,
    "hw8g32": lambda q, k, v, layer=None: attention_hw8(q, k, v, group=32),
}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--text", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--formats", default="fp32,kv8,hw8,hw8g32")
    a = p.parse_args()

    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(a.tokenizer).encode(open(a.text, encoding="utf-8").read()).ids
    ref = HybridRef(BONSReader(a.image))
    fmts = a.formats.split(",")
    res = {f: [] for f in fmts}
    for c in range(a.chunks):
        toks = ids[c*a.ctx:(c + 1)*a.ctx]
        for f in fmts:
            t0 = time.time()
            ref.attn_fn = FORMATS[f]
            nll, _ = ref.score(toks)
            res[f].append(nll)
            print(f"chunk {c} {f:7s} ppl {np.exp(nll.mean()):8.4f}  past 1024 "
                  f"{np.exp(nll[1024:].mean()) if len(nll) > 1024 else float('nan'):8.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    base = np.concatenate(res[fmts[0]])
    print(f"\n{'format':8s} {'ppl':>9s} {'vs ' + fmts[0]:>9s} {'ppl>1024':>9s} {'mean |dNLL|':>11s}")
    for f in fmts:
        n = np.concatenate(res[f])
        late = np.concatenate([x[1024:] for x in res[f]])
        print(f"{f:8s} {np.exp(n.mean()):9.4f} {100*(np.exp(n.mean() - base.mean()) - 1):+8.3f}% "
              f"{np.exp(late.mean()) if len(late) else float('nan'):9.4f} {np.abs(n - base).mean():11.5f}")

if __name__ == "__main__":
    main()

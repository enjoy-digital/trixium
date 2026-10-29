#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""A fast, batched float reference of the hybrid (Qwen3.5 / Ternary-Bonsai-2) BONS images.

The arithmetic of model/mirror.py's hybrid27_generate(exact=True) (fp32 weights, no activation
quantization, the same rotations, gates, convolution and recurrence) but vectorized: the
projections of a whole sequence run as matrix products, the DeltaNet recurrence steps over the
positions with every head at once, the attention is causal over the sequence. Not bit-exact
(the reductions are reordered): it is the tool for quality experiments on long texts
(perplexity, attention number formats, hidden states for retrieval), where the mirror's
scalar loops would take hours.

    from model.bons import BONSReader
    from model.fastref import HybridRef
    ref = HybridRef(BONSReader("models/bons/bonsai2-2b-n4-ti375.img"))
    nll, hidden = ref.score(tokens)          # per-position next-token NLL, final hidden states
    ref.attn_fn = my_attention              # (q, k, v, layer) -> out: try a number format
"""

import numpy as np

from model import layout
from model.bons import restored_row_exact

f32 = np.float32

def attention_fp32(q, k, v, layer=None):
    """Causal attention: q [T, nh, hd], k, v [T, nkv, hd] (already normed and rotated, q
    pre-scaled) -> [T, nh, hd]."""
    T, nh, hd = q.shape
    n_rep = nh//k.shape[1]
    kk = np.repeat(k, n_rep, axis=1)
    vv = np.repeat(v, n_rep, axis=1)
    s = np.einsum("thd,shd->hts", q, kk)
    s += np.triu(np.full((T, T), -np.inf, f32), 1)[None]
    s -= s.max(axis=2, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=2, keepdims=True)
    return np.einsum("hts,shd->thd", p, vv).astype(f32)

class HybridRef:
    def __init__(self, img, wdtype=np.float16):
        self.img = img
        h = self.h = img.header
        assert h["version"] >= 3
        self.n_layer, self.n_embd = h["n_layer"], h["n_embd"]
        self.n_head, self.n_kv, self.hd = h["n_head"], h["n_kv_head"], h["head_dim"]
        self.nk, self.hk = h["dn_num_k_heads"], h["dn_k_head_dim"]
        self.nv, self.hv = h["dn_num_v_heads"], h["dn_v_head_dim"]
        self.K = h["dn_conv_kernel"]
        self.rope_n = h["rope_dims"] or self.hd
        self.eps = f32(h["rms_eps"])
        self.B = h["hadamard_block"]
        self.widths = list(h["hadamard_widths"])
        self.gated = bool(h["attn_gated"])
        self.dn = [t == "deltanet" for t in h["layer_types"]]
        self.wdtype = wdtype
        self.wcache = {}
        self.attn_fn = attention_fp32
        self.rope_freq = np.array([f32(f32(h["rope_theta"])**f32(-2.0*i/self.rope_n))
                                   for i in range(self.rope_n//2)], dtype=f32)

        L = range(self.n_layer)
        self.norm_attn = [self._f32(layout.ROLE_ATTN_NORM, l) for l in L]
        self.norm_ffn  = [self._f32(layout.ROLE_FFN_NORM, l) for l in L]
        self.norm_q, self.norm_k, self.norm_ssm, self.ssm_a, self.ssm_dt, self.conv_w = ({}, {}, {}, {}, {}, {})
        for l in L:
            if self.dn[l]:
                self.norm_ssm[l] = self._f32(layout.ROLE_SSM_NORM, l)
                self.ssm_a[l]    = self._f32(layout.ROLE_SSM_A, l)
                self.ssm_dt[l]   = self._f32(layout.ROLE_SSM_DT, l)
                self.conv_w[l]   = self._f32(layout.ROLE_SSM_CONV, l).reshape(-1, self.K)
            else:
                self.norm_q[l] = self._f32(layout.ROLE_ATTN_Q_NORM, l)
                self.norm_k[l] = self._f32(layout.ROLE_ATTN_K_NORM, l)
        self.out_norm = self._f32(layout.ROLE_OUT_NORM, 0)
        self.head_role = layout.ROLE_LM_HEAD if self._entry(layout.ROLE_LM_HEAD, 0) else layout.ROLE_TOK_EMB
        self._emb = None

    # Tensors ---------------------------------------------------------------------------------------
    def _entry(self, role, l, kind=None):
        for e in self.img.entries:
            if e["role"] == role and e["layer"] == l and (kind is None or e["kind"] == kind):
                return e
        return None

    def _f32(self, role, l):
        e = self._entry(role, l, layout.KIND_F32)
        assert e is not None, (role, l)
        return np.asarray(self.img.tensor(e["name"])[1], dtype=f32).reshape(-1)

    def _ksegs(self, e):
        if not (e["flags"] & layout.FLAG_KSEG):
            return [e]
        segs = [t for t in self.img.entries if t["role"] == e["role"] and t["layer"] == e["layer"]
                and t["kind"] == e["kind"] and (t["flags"] & layout.FLAG_KSEG)]
        return sorted(segs, key=lambda t: t["kseg"])

    def weight(self, e):
        """The dequantized matrix of an entry (its K-segments joined), cached."""
        if e["name"] not in self.wcache:
            ws = []
            for t in self._ksegs(e):
                scales, codes = self.img.tensor(t["name"])
                if t["kind"] == layout.KIND_F32:
                    ws.append(np.asarray(codes, dtype=f32).reshape(t["rows"], t["cols"]))
                    continue
                g = 128 if layout.is_ternary(t["kind"]) else 32
                off = 1.0 if layout.is_ternary(t["kind"]) else 0.0
                w = (codes.astype(f32).reshape(t["rows"], -1, g) - f32(off))*scales.astype(f32)[:, :, None]
                ws.append(w.reshape(t["rows"], t["cols"]))
            self.wcache[e["name"]] = np.concatenate(ws, axis=1).astype(self.wdtype)
        return self.wcache[e["name"]]

    def matmul(self, e, X):
        """X [T, cols] -> [T, rows] (fp32 products of the cached weights)."""
        W = self.weight(e)
        out = np.empty((X.shape[0], W.shape[0]), f32)
        for r0 in range(0, W.shape[0], 8192):              # Bounded fp32 temporaries.
            out[:, r0:r0 + 8192] = X @ W[r0:r0 + 8192].astype(f32).T
        return out

    def rotate(self, X, width):
        if not self.B:
            return X
        wi = self.widths.index(width)
        out = np.empty_like(X)
        for b in range(width//self.B):
            e = self._entry(layout.ROLE_HADAMARD, layout.hadamard_layer(wi, b))
            out[:, b*self.B:(b + 1)*self.B] = self.matmul(e, X[:, b*self.B:(b + 1)*self.B])
        return out

    def proj(self, role, l, X):
        e = self._entry(role, l)
        cols = sum(t["cols"] for t in self._ksegs(e))
        return self.matmul(e, self.rotate(X, cols) if self.B else X)

    # Ops -------------------------------------------------------------------------------------------
    def rmsnorm(self, X, w):
        r = f32(1.0)/np.sqrt((X*X).mean(axis=-1, keepdims=True, dtype=f32) + self.eps)
        return (X*r*w).astype(f32)

    def l2norm(self, X):
        n = np.maximum(np.sqrt((X*X).sum(axis=-1, keepdims=True, dtype=f32)), self.eps)
        return (X/n).astype(f32)

    @staticmethod
    def sigmoid(x): return (1.0/(1.0 + np.exp(-x))).astype(f32)
    @staticmethod
    def silu(x): return (x/(1.0 + np.exp(-x))).astype(f32)
    @staticmethod
    def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(f32)

    def embed(self, tokens):
        h = self.h
        if self._emb is None:
            emb16 = self._entry(layout.ROLE_TOK_EMB16, 0)
            if emb16 is not None:
                _, mant = self.img.tensor(emb16["name"])
                _, exps = self.img.tensor(self._entry(layout.ROLE_TOK_EMB16_EXP, 0)["name"])
                self._emb = ("table", mant, np.asarray(exps).reshape(-1))
            else:
                s, c = self.img.tensor(self._entry(layout.ROLE_TOK_EMB, 0)["name"])
                wi = self.widths.index(self.n_embd)
                signs = np.concatenate([np.asarray(self.img.tensor(self._entry(layout.ROLE_HADAMARD,
                    layout.hadamard_layer(wi, b))["name"])[1])[0].astype(np.int64) - 1
                    for b in range(self.n_embd//self.B)])
                self._emb = ("restore", np.asarray(s), np.asarray(c), signs)
        X = np.empty((len(tokens), self.n_embd), f32)
        for i, t in enumerate(tokens):
            if self._emb[0] == "table":
                m, e = self._emb[1][t], int(self._emb[2][t])
            else:
                m, e = restored_row_exact(self._emb[1][t], self._emb[2][t], self._emb[3], self.B)
            X[i] = m.astype(f32)*f32(np.exp2(e))
        return X

    def deltanet(self, l, Hn):
        T = Hn.shape[0]
        nq, nvh = self.nk*self.hk, self.nv*self.hv
        qkv = self.proj(layout.ROLE_ATTN_QKV, l, Hn)
        z   = self.proj(layout.ROLE_ATTN_ZGATE, l, Hn)
        beta  = self.sigmoid(self.matmul(self._entry(layout.ROLE_SSM_BETA, l), Hn))
        alpha = self.matmul(self._entry(layout.ROLE_SSM_ALPHA, l), Hn)
        decay = np.exp((self.softplus(alpha + self.ssm_dt[l])*self.ssm_a[l]).astype(f32))
        # Causal depthwise convolution over K taps (zero history), then SiLU.
        w = self.conv_w[l]
        pad = np.concatenate([np.zeros((self.K - 1, qkv.shape[1]), f32), qkv])
        conv = sum(pad[i:i + T]*w[:, i] for i in range(self.K)).astype(f32)
        conv = self.silu(conv)
        q = self.l2norm(conv[:, :nq].reshape(T, self.nk, self.hk))
        k = self.l2norm(conv[:, nq:2*nq].reshape(T, self.nk, self.hk))
        v = conv[:, 2*nq:].reshape(T, self.nv, self.hv)
        kh = np.arange(self.nv) % self.nk
        M = np.zeros((self.nv, self.hv, self.hk), f32)
        o = np.empty((T, self.nv, self.hv), f32)
        scale = f32(1.0/np.sqrt(self.hk))
        for t in range(T):
            qt, kt = q[t, kh], k[t, kh]                         # [nv, hk]
            M *= decay[t][:, None, None]
            delta = (v[t] - np.einsum("hjk,hk->hj", M, kt))*beta[t][:, None]
            M += delta[:, :, None]*kt[:, None, :]
            o[t] = np.einsum("hjk,hk->hj", M, qt)*scale
        nrm = self.rmsnorm(o, self.norm_ssm[l])
        gz = nrm*self.silu(z.reshape(T, self.nv, self.hv))
        rep = self.nv//self.nk
        gpos = kh*rep + np.arange(self.nv)//self.nk
        og = np.empty_like(gz)
        og[:, gpos] = gz
        return self.proj(layout.ROLE_SSM_OUT, l, og.reshape(T, -1))

    def attention(self, l, Hn, pos0=0):
        T, hd, nh, nkv = Hn.shape[0], self.hd, self.n_head, self.n_kv
        qf = self.proj(layout.ROLE_ATTN_Q, l, Hn)
        k  = self.proj(layout.ROLE_ATTN_K, l, Hn).reshape(T, nkv, hd)
        v  = self.proj(layout.ROLE_ATTN_V, l, Hn).reshape(T, nkv, hd)
        if self.gated:
            qf = qf.reshape(T, nh, 2*hd)
            q, gate = qf[:, :, :hd], qf[:, :, hd:]
        else:
            q, gate = qf.reshape(T, nh, hd), None
        q = self.rmsnorm(q, self.norm_q[l]); k = self.rmsnorm(k, self.norm_k[l])
        ang = (np.arange(pos0, pos0 + T, dtype=f32)[:, None]*self.rope_freq[None]).astype(f32)
        cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
        half = self.rope_n//2
        def rope(x):
            x = x.copy()
            x0, x1 = x[..., :half].copy(), x[..., half:self.rope_n].copy()
            x[..., :half] = x0*cos - x1*sin
            x[..., half:self.rope_n] = x0*sin + x1*cos
            return x
        q, k = rope(q), rope(k)
        q = (q*f32(1.0/np.sqrt(hd))).astype(f32)
        o = self.attn_fn(q, k, v, l)
        if self.gated:
            o = o*self.sigmoid(gate)
        return self.proj(layout.ROLE_ATTN_OUT, l, o.reshape(T, -1).astype(f32))

    # Sequences -------------------------------------------------------------------------------------
    def hidden(self, tokens, taps=None):
        """Final normed hidden states [T, n_embd] of a sequence (teacher forced from position 0).
        taps: a dict {layer: None} filled with the residual stream after those layers (unnormed)."""
        X = self.embed(tokens)
        for l in range(self.n_layer):
            Hn = self.rmsnorm(X, self.norm_attn[l])
            X = X + (self.deltanet(l, Hn) if self.dn[l] else self.attention(l, Hn))
            Hn = self.rmsnorm(X, self.norm_ffn[l])
            R = self.rotate(Hn, self.n_embd) if self.B else Hn
            g = self.matmul(self._entry(layout.ROLE_FFN_GATE, l), R)
            u = self.matmul(self._entry(layout.ROLE_FFN_UP, l), R)
            X = X + self.proj(layout.ROLE_FFN_DOWN, l, (self.silu(g)*u).astype(f32))
            if taps is not None and l in taps:
                taps[l] = X.copy()
        return self.rmsnorm(X, self.out_norm)

    def logits(self, Hs):
        return self.proj(self.head_role, 0, Hs)[:, :self.h["vocab"]]

    def score(self, tokens, chunk=256):
        """-> (nll [T-1] of tokens[1:] given their prefix, final hidden states [T, n_embd])."""
        Hs = self.hidden(tokens)
        nll = np.empty(len(tokens) - 1, np.float64)
        for c0 in range(0, len(tokens) - 1, chunk):
            lg = self.logits(Hs[c0:min(c0 + chunk, len(tokens) - 1)]).astype(np.float64)
            m = lg.max(axis=1, keepdims=True)
            lse = (m + np.log(np.exp(lg - m).sum(axis=1, keepdims=True)))[:, 0]
            tgt = np.asarray(tokens[c0 + 1:c0 + 1 + lg.shape[0]])
            nll[c0:c0 + lg.shape[0]] = lse - lg[np.arange(lg.shape[0]), tgt]
        return nll, Hs

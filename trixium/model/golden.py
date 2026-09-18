#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Golden models for TriXium.

Two references live here:

- FloatModel: fp32 Qwen3 forward pass (dequantized weights), transliterated from the PrismML
  llama.cpp fork's semantics (neox RoPE with YaRN, per-head Q/K RMSNorm, GQA, SwiGLU). Used to
  validate GGUF/BONS parsing and as the quality baseline.

- BitExactModel (see bitexact.py): the integer/fixed-point specification the RTL implements,
  validated against FloatModel.

CLI smoke test:
  .venv/bin/python -m trixium.model.golden --gguf models/1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf \
      --tokenizer models/1.7b/tokenizer.json --prompt "The capital of France is" -n 8
"""

import argparse
import math

import numpy as np

from trixium.model.gguf_reader import GGUFReader
from trixium.model.configs import ModelConfig, CONFIGS
from trixium.model import qformats

# GGUF loading -------------------------------------------------------------------------------------

# The GGUF architectures the geometry reader understands. qwen3: the Ternary-Bonsai 1.7B and
# 8B. qwen35 (llama.cpp's name for Qwen3.5; qwen3next is the same family): Bonsai 2, the 27B -
# 48 Gated DeltaNet layers and 16 gated-attention layers (every 4th), the DeltaNet keys under
# ssm.* (time_step_rank = value heads, group_count = key heads, state_size = head dim,
# inner_size = value heads x head dim, conv_kernel), rope.dimension_count = the partial RoPE,
# and PrismML's prism.hadamard.* rotation metadata. Read from the file on 2026-09-18:
# 24/4 attention heads x 256 with the output gate, 48/16 DeltaNet heads x 128, hidden 5120,
# FFN 17408, vocab 248320, Hadamard block 1024 with sign vectors for widths 5120/6144/17408.
# Its DeltaNet layers are refused by the converter until the engine has a recurrent unit
# (docs/xcu1525.md).
GGUF_ARCHS = ("qwen3", "qwen35", "qwen3next")

def config_from_gguf(reader):
    """Build a ModelConfig from GGUF metadata."""
    md   = reader.metadata
    arch = md["general.architecture"]
    assert arch in GGUF_ARCHS, f"unsupported architecture {arch} (known: {GGUF_ARCHS})"
    g = lambda k, default=None: md.get(f"{arch}.{k}", default)
    vocab = reader.tensors["token_embd.weight"].shape[0]
    name = md.get("general.basename", arch).lower()
    if name in ("folded", "hf", "") and md.get("general.size_label"):   # the 27B file says "folded"
        name = f"bonsai-{md['general.size_label'].lower()}"
    cfg = ModelConfig(
        name           = name,
        n_layer        = g("block_count"),
        n_embd         = g("embedding_length"),
        n_head         = g("attention.head_count"),
        n_kv_head      = g("attention.head_count_kv"),
        head_dim       = g("attention.key_length", g("embedding_length")//g("attention.head_count")),
        ffn_dim        = g("feed_forward_length"),
        vocab          = vocab,
        rope_theta     = g("rope.freq_base", 10000.0),
        rms_eps        = g("attention.layer_norm_rms_epsilon", 1e-6),
        tie_embeddings = "output.weight" not in reader.tensors,
        yarn_factor    = g("rope.scaling.factor", 1.0) if g("rope.scaling.type") == "yarn" else 1.0,
        yarn_orig_ctx  = g("rope.scaling.original_context_length", 0),
    )
    if arch in ("qwen35", "qwen3next"):
        interval = g("full_attention_interval", 4)
        cfg.layer_types    = tuple("attn" if (l + 1) % interval == 0 else "deltanet"
                                   for l in range(cfg.n_layer))
        cfg.dn_num_v_heads = g("ssm.time_step_rank", 0)
        cfg.dn_num_k_heads = g("ssm.group_count", 0)
        cfg.dn_k_head_dim  = g("ssm.state_size", 0)
        cfg.dn_v_head_dim  = g("ssm.inner_size", 0)//max(1, cfg.dn_num_v_heads)
        cfg.dn_conv_kernel = g("ssm.conv_kernel", 0)
        cfg.rope_dims      = g("rope.dimension_count", 0)
        # The attention layers' q projection is [q | gate] (2 x n_head x head_dim rows):
        # the file says so through its shape, so read it from the first attention layer.
        first_attn = next((l for l in range(cfg.n_layer) if cfg.layer_types[l] == "attn"), None)
        if first_attn is not None and f"blk.{first_attn}.attn_q.weight" in reader.tensors:
            cfg.attn_gated = reader.tensors[f"blk.{first_attn}.attn_q.weight"].shape[0] == 2*cfg.q_dim
    if md.get("prism.hadamard.block_size"):
        assert md.get("prism.hadamard.transform") == "normalized-sylvester-walsh-hadamard", \
            md.get("prism.hadamard.transform")
        assert md.get("prism.hadamard.axis") == "input-last-dimension"
        cfg.hadamard_block = int(md["prism.hadamard.block_size"])
        widths, signs, pos = md["prism.hadamard.sign_widths"], md["prism.hadamard.sign_values"], 0
        cfg.hadamard_signs = {}
        for w in widths:
            cfg.hadamard_signs[int(w)] = tuple(int(s) for s in signs[pos:pos + int(w)]); pos += int(w)
        assert pos == len(signs), (pos, len(signs))
    return cfg

def check_engine_support(cfg):
    """What the engine can run today. Raises SystemExit with the reason otherwise."""
    if cfg.n_dn_layers:
        raise SystemExit(
            f"{cfg.name}: {cfg.n_dn_layers} of {cfg.n_layer} layers are Gated DeltaNet "
            f"({cfg.dn_num_v_heads} value heads x {cfg.dn_v_head_dim}, {cfg.dn_num_k_heads} "
            f"key heads x {cfg.dn_k_head_dim}, conv kernel {cfg.dn_conv_kernel}); the engine "
            f"has no recurrent unit for them yet - the projections are ordinary matvecs, the "
            f"state update is not. See docs/xcu1525.md (the 27B).")
    for name, dim in (("n_embd", cfg.n_embd), ("ffn_dim", cfg.ffn_dim), ("head_dim", cfg.head_dim)):
        if dim % 128:
            raise SystemExit(f"{cfg.name}: {name} = {dim} is not a multiple of the 128-wide "
                             f"ternary group")
    if cfg.hadamard_block:
        raise SystemExit(
            f"{cfg.name}: the weights are stored in PrismML's rotated basis (Hadamard block "
            f"{cfg.hadamard_block}, sign vectors for widths {sorted(cfg.hadamard_signs)}): the "
            f"runtime has to rotate every projection's input activation and un-rotate the "
            f"embedding rows, which the engine does not do yet (docs/xcu1525.md, the 27B).")

class LazyTernary:
    """A ternary matrix kept as its int8 codes and fp16 scales (1 byte per weight instead of
    4): the 27B's 27 GB fits where 108 GB of fp32 do not. `@` and `.T` are what FloatModel
    uses; the product is exact fp32 math per group (codes - 1) * scale, row by row in
    chunks."""
    def __init__(self, scales, codes, shape):
        self.shape  = tuple(shape)
        self.scales = scales.astype(np.float32).reshape(shape[0], -1)     # (rows, G)
        self.codes  = codes.reshape(shape[0], shape[1])                     # uint8 {0,1,2}
        self.T      = _LazyT(self)

    def __getitem__(self, idx):        # embedding lookup: rows -> fp32
        rows = np.atleast_1d(np.asarray(idx))
        c = self.codes[rows].astype(np.float32).reshape(len(rows), -1, 128) - 1.0
        out = (c*self.scales[rows][:, :, None]).reshape(len(rows), self.shape[1])
        return out if np.ndim(idx) else out[0]

class _LazyT:
    def __init__(self, m):
        self.m = m
    def __rmatmul__(self, x):          # x @ W.T  ->  (.., rows)
        m = self.m
        x = np.asarray(x, dtype=np.float32)
        xs = x.reshape(-1, m.shape[1])                                     # (P, cols)
        out = np.empty((xs.shape[0], m.shape[0]), dtype=np.float32)
        step = max(1, (64 << 20)//(m.shape[1]*4))                          # 64 MB of fp32 rows
        for r0 in range(0, m.shape[0], step):
            r1 = min(m.shape[0], r0 + step)
            w = (m.codes[r0:r1].astype(np.float32).reshape(r1 - r0, -1, 128) - 1.0) \
                * m.scales[r0:r1][:, :, None]
            out[:, r0:r1] = xs @ w.reshape(r1 - r0, m.shape[1]).T
        return out.reshape(*x.shape[:-1], m.shape[0])

def load_gguf_f32(path, lazy=False):
    """Load a GGUF file -> (config, dict name -> fp32 ndarray with GGUF logical shape).
    lazy=True keeps the ternary matrices as LazyTernary (codes + scales, 1 byte/weight)."""
    reader  = GGUFReader(path)
    cfg     = config_from_gguf(reader)
    tensors = {}
    for name, t in reader.tensors.items():
        dequant = qformats.DEQUANT.get(t.type_id)
        assert dequant is not None, f"no dequant for {name} type {t.type_id}"
        raw = reader.tensor_bytes(name)
        if t.type_id == qformats.GGML_Q2_0 and qformats.q2_0_group(raw, int(np.prod(t.shape))) != 128:
            raise NotImplementedError(f"{name}: a group-64 Q2_0 file - use the PQ2_0 (g128) file")
        if t.type_id in (qformats.GGML_Q2_0, qformats.GGML_PQ2_0, qformats.GGML_PTQ1_0):
            # In place: the generic dequant's float temporaries (codes - 1, times the
            # broadcast scales) doubled the 8B's 2.5 GB embedding table three times over
            # and the whole 33 GB model was OOM-killed at 57 GB (2026-09-18).
            unpack = qformats.ptq1_0_unpack if t.type_id == qformats.GGML_PTQ1_0 else qformats.q2_0_unpack
            scales, codes = unpack(raw)
            if lazy:
                tensors[name] = LazyTernary(scales, codes, t.shape)
                continue
            w = codes.astype(np.float32).reshape(-1, 128)
            del codes
            w -= 1.0
            w *= scales.astype(np.float32)[:, None]
            tensors[name] = w.reshape(t.shape)
        else:
            tensors[name] = dequant(raw).reshape(t.shape)
    return cfg, tensors

# RoPE with YaRN (exact transliteration of ggml's rope_yarn) ---------------------------------------

def _yarn_corr_dim(n_dims, n_ctx_orig, n_rot, base):
    return n_dims*math.log(n_ctx_orig/(n_rot*2*math.pi))/(2*math.log(base))

def rope_tables(cfg, positions, dtype=np.float32):
    """cos/sin tables -> (len(positions), head_dim//2) each, yarn-corrected like llama.cpp
    (ext_factor=1 when yarn is configured, attn_factor=1, beta_fast=32, beta_slow=1)."""
    n_dims     = cfg.rope_dims if cfg.rope_dims else cfg.head_dim
    freq_scale = 1.0/cfg.yarn_factor
    ext_factor = 1.0 if cfg.yarn_factor != 1.0 else 0.0
    mscale     = 1.0

    p     = np.asarray(positions, dtype=np.float64)[:, None]      # (P, 1)
    d     = np.arange(n_dims//2, dtype=np.float64)[None, :]       # (1, D/2) pair index.
    theta_extrap = p*np.power(cfg.rope_theta, -2.0*d/n_dims)
    theta_interp = freq_scale*theta_extrap
    if ext_factor != 0.0:
        low  = max(0.0, math.floor(_yarn_corr_dim(n_dims, cfg.yarn_orig_ctx, 32.0, cfg.rope_theta)))
        high = min(n_dims - 1.0, math.ceil(_yarn_corr_dim(n_dims, cfg.yarn_orig_ctx, 1.0, cfg.rope_theta)))
        # ggml's ramp is indexed by pair index (i0/2 with i0 stepping by 2).
        y        = (d - low)/max(0.001, high - low)
        ramp     = (1.0 - np.minimum(1.0, np.maximum(0.0, y)))*ext_factor
        theta    = theta_interp*(1 - ramp) + theta_extrap*ramp
        mscale  *= 1.0 + 0.1*math.log(1.0/freq_scale)
    else:
        theta = theta_interp
    return (np.cos(theta)*mscale).astype(dtype), (np.sin(theta)*mscale).astype(dtype)

def apply_rope_neox(x, cos, sin):
    """x: (P, n_head, head_dim); pairs (i, i+D/2) (neox), tables (P, D/2)."""
    D  = x.shape[-1]
    x0 = x[..., :D//2]
    x1 = x[..., D//2:]
    c  = cos[:, None, :]
    s  = sin[:, None, :]
    return np.concatenate([x0*c - x1*s, x0*s + x1*c], axis=-1)

# Float model --------------------------------------------------------------------------------------

def rmsnorm(x, w, eps):
    return x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)*w

def silu(x):
    return x/(1.0 + np.exp(-x))

def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e/e.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def softplus(x):
    return np.logaddexp(0.0, x)

def l2norm(x, eps):
    """ggml_l2_norm: x / max(|x|, eps) over the last axis."""
    n = np.sqrt(np.sum(x*x, axis=-1, keepdims=True))
    return x/np.maximum(n, eps)

# PrismML's rotated weight basis (the 27B). R is the natural-order Sylvester Walsh-Hadamard
# matrix of the block, R[i][j] = (-1)^popcount(i & j) / sqrt(block) - symmetric and its own
# inverse. A folded weight W' = W R diag(s) is applied as y = W' (R (s * x)), blockwise on
# the input dimension; a folded embedding table stores z = R diag(s) e... and is restored as
# e = s * (R z) after the lookup. Exactly llama.cpp's fork (llama-graph.cpp, 2026-09-18).
def hadamard_matrix(block):
    i = np.arange(block)
    par = np.bitwise_and(i[:, None], i[None, :])
    bits = np.zeros_like(par)
    while par.any():
        bits ^= par & 1
        par >>= 1
    return np.where(bits & 1, -1.0, 1.0).astype(np.float32)/np.sqrt(float(block))

def hadamard_rotate(x, signs, block):
    """x' = R (s * x) per block over the last axis (the projection-input transform)."""
    R = hadamard_matrix(block)
    xs = (x*np.asarray(signs, dtype=np.float32)).reshape(*x.shape[:-1], -1, block)
    return (xs @ R.T).reshape(x.shape)

def hadamard_restore(z, signs, block):
    """e = s * (R z) per block (the embedding table's inverse-after-lookup)."""
    R = hadamard_matrix(block)
    zr = z.reshape(*z.shape[:-1], -1, block) @ R.T
    return zr.reshape(z.shape)*np.asarray(signs, dtype=np.float32)

class FloatModel:
    """fp32 forward pass of the Qwen3 models (the 1.7B/8B) and of the Qwen3.5 hybrid (the
    27B: Gated DeltaNet layers, gated attention with partial RoPE, PrismML's rotated weight
    basis). Tensor dict uses GGUF names/shapes."""

    def __init__(self, cfg, tensors):
        self.cfg = cfg
        self.t   = tensors
        self.head_out = tensors.get("output.weight", tensors["token_embd.weight"])
        # A lazy (codes + scales) embedding table: rows are dequantized on lookup.
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.k_cache = [np.zeros((0, cfg.n_kv_head, cfg.head_dim), dtype=np.float32)
                        for _ in range(cfg.n_layer)]
        self.v_cache = [np.zeros((0, cfg.n_kv_head, cfg.head_dim), dtype=np.float32)
                        for _ in range(cfg.n_layer)]
        # DeltaNet: the conv window (last kernel-1 inputs of the qkv channels) and the
        # recurrent state per value head, S[h] of (v dim, k dim) - llama.cpp keeps it
        # transposed (rows j = v index); S[h][j, i] here.
        if cfg.n_dn_layers:
            nq = cfg.dn_num_k_heads*cfg.dn_k_head_dim
            nv = cfg.dn_num_v_heads*cfg.dn_v_head_dim
            self.conv_state = [np.zeros((cfg.dn_conv_kernel - 1, 2*nq + nv), dtype=np.float32)
                               for _ in range(cfg.n_layer)]
            self.dn_state   = [np.zeros((cfg.dn_num_v_heads, cfg.dn_v_head_dim, cfg.dn_k_head_dim),
                                        dtype=np.float32) for _ in range(cfg.n_layer)]
        self.pos = 0

    # The projections: every folded weight sees its input rotated first.
    def _rot(self, x, perm_gdn=False):
        cfg = self.cfg
        if not cfg.hadamard_block:
            return x
        if perm_gdn:
            # The DeltaNet output is tiled [hd, nk, rep] (v head h = k head h % nk); the
            # fold used the grouped order [hd, rep, nk] (the rep value heads of one key head
            # adjacent), so permute before the rotation. The 16-channel firmware writes
            # head outputs in grouped order directly (docs/xcu1525.md).
            hd, nk = cfg.dn_v_head_dim, cfg.dn_num_k_heads
            rep = cfg.dn_num_v_heads//nk
            x = x.reshape(*x.shape[:-1], rep, nk, hd).swapaxes(-3, -2).reshape(x.shape)
        width, B = x.shape[-1], cfg.hadamard_block
        if f"had.{width}.0" in self.t:
            # The image's own block tensors (their fp16 scale is exact only for a power-of-
            # four block: 1/32 for the 27B's 1024, not 1/sqrt(128) for the tiny model), so
            # the golden that gates the firmware multiplies by exactly what the engine does.
            out = np.empty_like(x)
            for b in range(width//B):
                out[..., b*B:(b + 1)*B] = x[..., b*B:(b + 1)*B] @ self.t[f"had.{width}.{b}"].T
            return out
        return hadamard_rotate(x, cfg.hadamard_signs[width], B)

    def _proj(self, x, name, perm_gdn=False):
        return self._rot(x, perm_gdn) @ self.t[name].T

    def _embed(self, tokens):
        e = np.asarray(self.t["token_embd.weight"][np.asarray(tokens)], dtype=np.float32)
        if self.cfg.hadamard_block:
            e = hadamard_restore(e, self.cfg.hadamard_signs[e.shape[-1]], self.cfg.hadamard_block)
        return e

    def _attn(self, l, h, cos, sin, P):
        cfg, p = self.cfg, f"blk.{l}."
        n_rep = cfg.n_head//cfg.n_kv_head
        scale = 1.0/math.sqrt(cfg.head_dim)
        qf = self._proj(h, p+"attn_q.weight")
        if cfg.attn_gated:
            qf   = qf.reshape(P, cfg.n_head, 2*cfg.head_dim)
            q    = qf[:, :, :cfg.head_dim]
            gate = qf[:, :, cfg.head_dim:]
        else:
            q, gate = qf.reshape(P, cfg.n_head, cfg.head_dim), None
        k = self._proj(h, p+"attn_k.weight").reshape(P, cfg.n_kv_head, cfg.head_dim)
        v = self._proj(h, p+"attn_v.weight").reshape(P, cfg.n_kv_head, cfg.head_dim)
        q = rmsnorm(q, self.t[p+"attn_q_norm.weight"], cfg.rms_eps)
        k = rmsnorm(k, self.t[p+"attn_k_norm.weight"], cfg.rms_eps)
        if cfg.rope_dims and cfg.rope_dims < cfg.head_dim:
            # Partial RoPE: the first rope_dims of every head, neox pairs inside them; the
            # rest passes through (Qwen3.5's mrope with equal position components).
            d = cfg.rope_dims
            q = np.concatenate([apply_rope_neox(q[..., :d], cos, sin), q[..., d:]], axis=-1)
            k = np.concatenate([apply_rope_neox(k[..., :d], cos, sin), k[..., d:]], axis=-1)
        else:
            q = apply_rope_neox(q, cos, sin)
            k = apply_rope_neox(k, cos, sin)
        self.k_cache[l] = np.concatenate([self.k_cache[l], k], axis=0)
        self.v_cache[l] = np.concatenate([self.v_cache[l], v], axis=0)
        K, V = self.k_cache[l], self.v_cache[l]
        T = K.shape[0]
        kq = np.repeat(K, n_rep, axis=1)
        vq = np.repeat(V, n_rep, axis=1)
        scores = np.einsum("phd,thd->hpt", q, kq)*scale
        mask   = (np.arange(T)[None, :] > (self.pos + np.arange(P))[:, None])
        scores = np.where(mask[None, :, :], -np.inf, scores)
        probs  = softmax(scores, axis=-1)
        attn   = np.einsum("hpt,thd->phd", probs, vq)
        if gate is not None:
            attn = attn*sigmoid(gate)
        return self._proj(attn.reshape(P, -1), p+"attn_output.weight")

    def _deltanet(self, l, h, P):
        cfg, p = self.cfg, f"blk.{l}."
        nk, hk = cfg.dn_num_k_heads, cfg.dn_k_head_dim
        nv, hv = cfg.dn_num_v_heads, cfg.dn_v_head_dim
        nq = nk*hk
        qkv   = self._proj(h, p+"attn_qkv.weight")                    # (P, 2 nq + nv hv)
        z     = self._proj(h, p+"attn_gate.weight").reshape(P, nv, hv)
        beta  = sigmoid(h @ self.t[p+"ssm_beta.weight"].T)             # (P, nv)  not folded
        alpha = h @ self.t[p+"ssm_alpha.weight"].T                     # (P, nv)
        g     = softplus(alpha + self.t[p+"ssm_dt.bias"])*self.t[p+"ssm_a"]   # ssm_a = -exp(A_log)
        # Causal short conv over the token axis, per channel, kernel taps w[c, 0..K-1]
        # applied to inputs t-K+1..t (ggml_ssm_conv), then SiLU.
        w  = self.t[p+"ssm_conv1d.weight"]                              # (channels, K)
        K_ = w.shape[1]
        win = np.concatenate([self.conv_state[l], qkv], axis=0)         # (K-1+P, C)
        self.conv_state[l] = win[-(K_ - 1):]
        conv = np.stack([np.sum(win[t:t + K_]*w.T, axis=0) for t in range(P)])  # (P, C)
        conv = silu(conv)
        q = l2norm(conv[:, :nq].reshape(P, nk, hk), cfg.rms_eps)
        k = l2norm(conv[:, nq:2*nq].reshape(P, nk, hk), cfg.rms_eps)
        v = conv[:, 2*nq:].reshape(P, nv, hv)
        S = self.dn_state[l]                                            # (nv, hv, hk)
        out = np.zeros((P, nv, hv), dtype=np.float32)
        scale = 1.0/math.sqrt(hk)
        for t in range(P):
            for hh in range(nv):
                kh = hh % nk                                            # tiled head map
                qt, kt, vt = q[t, kh], k[t, kh], v[t, hh]
                M = S[hh]*np.exp(g[t, hh])                               # (hv, hk)
                delta = (vt - M @ kt)*beta[t, hh]                       # (hv,)
                M = M + delta[:, None]*kt[None, :]
                S[hh] = M
                out[t, hh] = (M @ qt)*scale
        o = rmsnorm(out, self.t[p+"ssm_norm.weight"], cfg.rms_eps)*silu(z)
        return self._proj(o.reshape(P, nv*hv), p+"ssm_out.weight", perm_gdn=True)

    def forward(self, tokens):
        """Process `tokens` (list of ids) after the current cache; returns logits of the last
        position (V,)."""
        cfg = self.cfg
        P   = len(tokens)
        pos = np.arange(self.pos, self.pos + P)
        cos, sin = rope_tables(cfg, pos)
        x = self._embed(tokens)                                          # (P, E)
        for l in range(cfg.n_layer):
            p = f"blk.{l}."
            h = rmsnorm(x, self.t[p+"attn_norm.weight"], cfg.rms_eps)
            if cfg.layer_type(l) == "deltanet":
                x = x + self._deltanet(l, h, P)
            else:
                x = x + self._attn(l, h, cos, sin, P)
            ffn_norm = p + ("post_attention_norm.weight" if p+"post_attention_norm.weight" in self.t
                            else "ffn_norm.weight")
            h    = rmsnorm(x, self.t[ffn_norm], cfg.rms_eps)
            gate = self._proj(h, p+"ffn_gate.weight")
            up   = self._proj(h, p+"ffn_up.weight")
            x    = x + self._proj(silu(gate)*up, p+"ffn_down.weight")
        self.pos += P
        h = rmsnorm(x[-1], self.t["output_norm.weight"], cfg.rms_eps)
        return self._rot(h) @ self.head_out.T  # (V,)

    def generate(self, tokens, n, greedy=True):
        """Greedy continuation; returns the generated ids."""
        out    = []
        logits = self.forward(list(tokens))
        for _ in range(n):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            logits = self.forward([nxt])
        return out

# CLI ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriXium float golden model.")
    parser.add_argument("--gguf",      required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt",    default="The capital of France is")
    parser.add_argument("--raw",       action="store_true", help="No chat template.")
    parser.add_argument("-n",          default=8, type=int)
    parser.add_argument("--lazy",      action="store_true", help="Keep ternary matrices as codes (the 27B).")
    args = parser.parse_args()

    print(f"loading {args.gguf}...")
    cfg, tensors = load_gguf_f32(args.gguf, lazy=args.lazy)
    print(f"config: {cfg}")
    model = FloatModel(cfg, tensors)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    if args.raw:
        text = args.prompt
    else:
        text = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    print(f"prompt ids: {ids}")
    out = model.generate(ids, args.n)
    print(f"generated ids: {out}")
    print(f"text: {tok.decode(out)!r}")

if __name__ == "__main__":
    main()

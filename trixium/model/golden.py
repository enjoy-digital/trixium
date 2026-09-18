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
# 8B. qwen3next: Bonsai 2 (the 27B), whose DeltaNet keys follow llama.cpp's convention for
# the architecture (ssm.* for the linear-attention geometry, full_attention_interval,
# rope.dimension_count for the partial RoPE); its DeltaNet layers are read here and refused
# by the converter until the engine has a recurrent unit (docs/xcu1525.md).
GGUF_ARCHS = ("qwen3", "qwen3next")

def config_from_gguf(reader):
    """Build a ModelConfig from GGUF metadata."""
    md   = reader.metadata
    arch = md["general.architecture"]
    assert arch in GGUF_ARCHS, f"unsupported architecture {arch} (known: {GGUF_ARCHS})"
    g = lambda k, default=None: md.get(f"{arch}.{k}", default)
    vocab = reader.tensors["token_embd.weight"].shape[0]
    cfg = ModelConfig(
        name           = md.get("general.basename", arch).lower(),
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
    if arch == "qwen3next":
        interval = g("full_attention_interval", 4)
        cfg.layer_types    = tuple("attn" if (l + 1) % interval == 0 else "deltanet"
                                   for l in range(cfg.n_layer))
        cfg.dn_num_v_heads = g("ssm.time_step_rank", 0)
        cfg.dn_num_k_heads = g("ssm.group_count", 0)
        cfg.dn_k_head_dim  = g("ssm.state_size", 0)
        cfg.dn_v_head_dim  = g("ssm.inner_size", 0)//max(1, cfg.dn_num_v_heads)
        cfg.dn_conv_kernel = g("ssm.conv_kernel", 0)
        cfg.rope_dims      = g("rope.dimension_count", 0)
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

def load_gguf_f32(path):
    """Load a GGUF file -> (config, dict name -> fp32 ndarray with GGUF logical shape)."""
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
    n_dims     = cfg.head_dim
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

class FloatModel:
    """fp32 Qwen3 forward pass. Tensor dict uses GGUF names/shapes:
    token_embd.weight (V, E), blk.N.{attn_norm,attn_q,attn_k,attn_v,attn_q_norm,attn_k_norm,
    attn_output,ffn_norm,ffn_gate,ffn_up,ffn_down}.weight, output_norm.weight, output.weight."""

    def __init__(self, cfg, tensors):
        self.cfg = cfg
        self.t   = tensors
        self.head_out = tensors.get("output.weight", tensors["token_embd.weight"])
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.k_cache = [np.zeros((0, cfg.n_kv_head, cfg.head_dim), dtype=np.float32)
                        for _ in range(cfg.n_layer)]
        self.v_cache = [np.zeros((0, cfg.n_kv_head, cfg.head_dim), dtype=np.float32)
                        for _ in range(cfg.n_layer)]
        self.pos = 0

    def forward(self, tokens):
        """Process `tokens` (list of ids) after the current cache; returns logits of the last
        position (V,)."""
        cfg = self.cfg
        P   = len(tokens)
        pos = np.arange(self.pos, self.pos + P)
        cos, sin = rope_tables(cfg, pos)

        x = self.t["token_embd.weight"][np.asarray(tokens)]  # (P, E)

        n_rep = cfg.n_head//cfg.n_kv_head
        scale = 1.0/math.sqrt(cfg.head_dim)

        for l in range(cfg.n_layer):
            p = f"blk.{l}."
            h = rmsnorm(x, self.t[p+"attn_norm.weight"], cfg.rms_eps)

            q = (h @ self.t[p+"attn_q.weight"].T).reshape(P, cfg.n_head,    cfg.head_dim)
            k = (h @ self.t[p+"attn_k.weight"].T).reshape(P, cfg.n_kv_head, cfg.head_dim)
            v = (h @ self.t[p+"attn_v.weight"].T).reshape(P, cfg.n_kv_head, cfg.head_dim)

            # Qwen3 per-head Q/K RMSNorm (before rope).
            q = rmsnorm(q, self.t[p+"attn_q_norm.weight"], cfg.rms_eps)
            k = rmsnorm(k, self.t[p+"attn_k_norm.weight"], cfg.rms_eps)

            q = apply_rope_neox(q, cos, sin)
            k = apply_rope_neox(k, cos, sin)

            self.k_cache[l] = np.concatenate([self.k_cache[l], k], axis=0)
            self.v_cache[l] = np.concatenate([self.v_cache[l], v], axis=0)
            K = self.k_cache[l] # (T, KV, D)
            V = self.v_cache[l]
            T = K.shape[0]

            # GQA attention with causal mask.
            kq = np.repeat(K, n_rep, axis=1)  # (T, H, D)
            vq = np.repeat(V, n_rep, axis=1)
            scores = np.einsum("phd,thd->hpt", q, kq)*scale  # (H, P, T)
            mask   = (np.arange(T)[None, :] > (self.pos + np.arange(P))[:, None])
            scores = np.where(mask[None, :, :], -np.inf, scores)
            probs  = softmax(scores, axis=-1)
            attn   = np.einsum("hpt,thd->phd", probs, vq).reshape(P, -1)

            x = x + attn @ self.t[p+"attn_output.weight"].T

            h = rmsnorm(x, self.t[p+"ffn_norm.weight"], cfg.rms_eps)
            gate = h @ self.t[p+"ffn_gate.weight"].T
            up   = h @ self.t[p+"ffn_up.weight"].T
            x    = x + (silu(gate)*up) @ self.t[p+"ffn_down.weight"].T

        self.pos += P
        h = rmsnorm(x[-1], self.t["output_norm.weight"], cfg.rms_eps)
        return h @ self.head_out.T  # (V,)

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
    args = parser.parse_args()

    print(f"loading {args.gguf}...")
    cfg, tensors = load_gguf_f32(args.gguf)
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

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""bonsai-tiny generator: a random-weight Qwen3-topology model small enough for Verilator.

Emits a BONS image + manifest directly (no GGUF round-trip). Deterministic per seed, so the
golden model and the RTL simulation regenerate identical images.

Usage: python -m trixium.model.tinygen --out models/tiny --channels 2 [--seed 1234]
"""

import argparse
import os

import numpy as np

from trixium.layout import (
    KIND_TERNARY, KIND_Q8, KIND_F32, FLAG_REPLICATED, GROUP_TERNARY, GROUP_Q8,
    ROLE_TOK_EMB, ROLE_OUT_NORM, ROLE_LM_HEAD,
    ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
    ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
    ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
)
from trixium.model.bons import TensorSpec, write_image
from trixium.model.configs import BONSAI_TINY

def _ternary(rng, name, role, layer, rows, cols):
    scales = rng.uniform(0.02, 0.08, (rows, cols//GROUP_TERNARY)).astype(np.float16)
    codes  = rng.integers(0, 3, (rows, cols), dtype=np.uint8) # {0,1,2} -> {-1,0,+1}.
    return TensorSpec(name, role, layer, KIND_TERNARY, rows, cols, scales=scales, values=codes)

def _q8(rng, name, role, layer, rows, cols):
    scales = rng.uniform(0.005, 0.02, (rows, cols//GROUP_Q8)).astype(np.float16)
    values = rng.integers(-127, 128, (rows, cols), dtype=np.int8)
    return TensorSpec(name, role, layer, KIND_Q8, rows, cols, scales=scales, values=values)

def _norm(rng, name, role, layer, cols):
    # RMSNorm gains around 1.0, replicated in every channel.
    values = rng.uniform(0.8, 1.2, (1, cols)).astype(np.float32)
    return TensorSpec(name, role, layer, KIND_F32, 1, cols, values=values, flags=FLAG_REPLICATED)

def make_tiny_tensors(cfg, seed=1234):
    # Mirrors the real Ternary-Bonsai GGUF inventory: ternary everything (embeddings included,
    # tied - no separate output.weight) + F32 norm vectors.
    rng = np.random.default_rng(seed)
    t   = []
    t.append(_ternary(rng, "token_embd.weight", ROLE_TOK_EMB, 0, cfg.vocab, cfg.n_embd))
    for l in range(cfg.n_layer):
        p = f"blk.{l}."
        t.append(_norm   (rng, p+"attn_norm.weight",   ROLE_ATTN_NORM,   l, cfg.n_embd))
        t.append(_ternary(rng, p+"attn_q.weight",      ROLE_ATTN_Q,      l, cfg.q_dim,   cfg.n_embd))
        t.append(_ternary(rng, p+"attn_k.weight",      ROLE_ATTN_K,      l, cfg.kv_dim,  cfg.n_embd))
        t.append(_ternary(rng, p+"attn_v.weight",      ROLE_ATTN_V,      l, cfg.kv_dim,  cfg.n_embd))
        t.append(_norm   (rng, p+"attn_q_norm.weight", ROLE_ATTN_Q_NORM, l, cfg.head_dim))
        t.append(_norm   (rng, p+"attn_k_norm.weight", ROLE_ATTN_K_NORM, l, cfg.head_dim))
        t.append(_ternary(rng, p+"attn_output.weight", ROLE_ATTN_OUT,    l, cfg.n_embd,  cfg.q_dim))
        t.append(_norm   (rng, p+"ffn_norm.weight",    ROLE_FFN_NORM,    l, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_gate.weight",    ROLE_FFN_GATE,    l, cfg.ffn_dim, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_up.weight",      ROLE_FFN_UP,      l, cfg.ffn_dim, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_down.weight",    ROLE_FFN_DOWN,    l, cfg.n_embd,  cfg.ffn_dim))
    t.append(_norm(rng, "output_norm.weight", ROLE_OUT_NORM, 0, cfg.n_embd))
    # Tied embeddings: the LM head matvec streams token_embd.weight.
    return t

def generate(out_dir, n_channels=2, seed=1234, cfg=BONSAI_TINY):
    os.makedirs(out_dir, exist_ok=True)
    tensors  = make_tiny_tensors(cfg, seed)
    img_path = os.path.join(out_dir, f"{cfg.name}-n{n_channels}.img")
    man_path = os.path.join(out_dir, f"{cfg.name}-n{n_channels}.json")
    manifest = write_image(cfg, tensors, n_channels, img_path, manifest_path=man_path)
    return img_path, man_path, manifest

def main():
    parser = argparse.ArgumentParser(description="Generate the bonsai-tiny random model.")
    parser.add_argument("--out",      default="models/tiny", help="Output directory.")
    parser.add_argument("--channels", default=2, type=int,   help="Channel shard count.")
    parser.add_argument("--seed",     default=1234, type=int)
    args = parser.parse_args()
    img, man, manifest = generate(args.out, args.channels, args.seed)
    print(f"image    : {img} ({os.path.getsize(img)/1e6:.2f} MB)")
    print(f"manifest : {man}")
    print(f"payload  : {manifest['payload_beats']} beats/channel "
          f"({manifest['payload_bytes_per_channel']/1e6:.2f} MB/channel)")

if __name__ == "__main__":
    main()

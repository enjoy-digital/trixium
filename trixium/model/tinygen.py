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

def make_tiny_mtp_tensors(cfg, rng, kind="q8"):
    """An MTP head at layer n_layer, as convert --mtp puts the Qwen3.5 nextn block in: a gated
    attention block (the attention-layer roles), eh_proj [enorm(e) | hnorm(h)] -> n_embd in
    the plain basis, and the three norms. kind: the matrices' kind (q8 or ternary)."""
    from trixium.layout import ROLE_MTP_EH, ROLE_MTP_ENORM, ROLE_MTP_HNORM, ROLE_MTP_HEAD_NORM
    mk = _q8 if kind == "q8" else _ternary
    L = cfg.n_layer
    p = f"blk.{L}."
    return [
        _norm(rng, p+"attn_norm.weight",   ROLE_ATTN_NORM,   L, cfg.n_embd),
        mk   (rng, p+"attn_q.weight",      ROLE_ATTN_Q,      L, 2*cfg.q_dim, cfg.n_embd),
        mk   (rng, p+"attn_k.weight",      ROLE_ATTN_K,      L, cfg.kv_dim,  cfg.n_embd),
        mk   (rng, p+"attn_v.weight",      ROLE_ATTN_V,      L, cfg.kv_dim,  cfg.n_embd),
        _norm(rng, p+"attn_q_norm.weight", ROLE_ATTN_Q_NORM, L, cfg.head_dim),
        _norm(rng, p+"attn_k_norm.weight", ROLE_ATTN_K_NORM, L, cfg.head_dim),
        mk   (rng, p+"attn_output.weight", ROLE_ATTN_OUT,    L, cfg.n_embd,  cfg.q_dim),
        _norm(rng, p+"post_attention_norm.weight", ROLE_FFN_NORM, L, cfg.n_embd),
        mk   (rng, p+"ffn_gate.weight",    ROLE_FFN_GATE,    L, cfg.ffn_dim, cfg.n_embd),
        mk   (rng, p+"ffn_up.weight",      ROLE_FFN_UP,      L, cfg.ffn_dim, cfg.n_embd),
        mk   (rng, p+"ffn_down.weight",    ROLE_FFN_DOWN,    L, cfg.n_embd,  cfg.ffn_dim),
        mk   (rng, p+"nextn.eh_proj.weight", ROLE_MTP_EH,    L, cfg.n_embd,  2*cfg.n_embd),
        _norm(rng, p+"nextn.enorm.weight", ROLE_MTP_ENORM,   L, cfg.n_embd),
        _norm(rng, p+"nextn.hnorm.weight", ROLE_MTP_HNORM,   L, cfg.n_embd),
        _norm(rng, p+"nextn.shared_head_norm.weight", ROLE_MTP_HEAD_NORM, L, cfg.n_embd),
    ]

def make_tiny_hybrid_tensors(cfg, seed=1234, mtp=None):
    """The 27B's tensor inventory at sim size (convert.py's names and kinds): DeltaNet layers
    with the fused qkv, the z gate, Q8 beta/alpha head projections, F32 conv taps (rows of
    1024), A, dt bias and the head norm; gated attention layers with [q | gate]; the
    Hadamard block tensors per input width; post_attention_norm as the FFN norm."""
    import dataclasses
    from trixium.layout import (ROLE_ATTN_QKV, ROLE_ATTN_ZGATE, ROLE_SSM_OUT, ROLE_SSM_BETA,
        ROLE_SSM_ALPHA, ROLE_SSM_CONV, ROLE_SSM_A, ROLE_SSM_DT, ROLE_SSM_NORM, ROLE_HADAMARD)
    from trixium.model.golden import hadamard_matrix
    rng = np.random.default_rng(seed)
    nk, hk, nv, hv = cfg.dn_num_k_heads, cfg.dn_k_head_dim, cfg.dn_num_v_heads, cfg.dn_v_head_dim
    nq, nvh = nk*hk, nv*hv
    widths = sorted({cfg.n_embd, cfg.q_dim, nvh, cfg.ffn_dim})
    cfg.hadamard_signs = {w: tuple(int(x) for x in rng.choice([-1, 1], w)) for w in widths}
    t = []
    t.append(_ternary(rng, "token_embd.weight", ROLE_TOK_EMB, 0, cfg.vocab, cfg.n_embd))
    for l in range(cfg.n_layer):
        p = f"blk.{l}."
        t.append(_norm(rng, p+"attn_norm.weight", ROLE_ATTN_NORM, l, cfg.n_embd))
        if cfg.layer_type(l) == "deltanet":
            t.append(_ternary(rng, p+"attn_qkv.weight",  ROLE_ATTN_QKV,   l, 2*nq + nvh, cfg.n_embd))
            t.append(_ternary(rng, p+"attn_gate.weight", ROLE_ATTN_ZGATE, l, nvh,        cfg.n_embd))
            t.append(_q8     (rng, p+"ssm_beta.weight",  ROLE_SSM_BETA,   l, nv,         cfg.n_embd))
            t.append(_q8     (rng, p+"ssm_alpha.weight", ROLE_SSM_ALPHA,  l, nv,         cfg.n_embd))
            taps = rng.uniform(-0.5, 0.5, (2*nq + nvh)*cfg.dn_conv_kernel).astype(np.float32)
            t.append(TensorSpec(p+"ssm_conv1d.weight", ROLE_SSM_CONV, l, KIND_F32,
                                len(taps)//1024, 1024, values=taps.reshape(-1, 1024), flags=FLAG_REPLICATED))
            t.append(TensorSpec(p+"ssm_a", ROLE_SSM_A, l, KIND_F32, 1, nv,
                                values=-np.exp(rng.uniform(-1.5, 0.5, (1, nv))).astype(np.float32), flags=FLAG_REPLICATED))
            t.append(TensorSpec(p+"ssm_dt.bias", ROLE_SSM_DT, l, KIND_F32, 1, nv,
                                values=rng.uniform(-2.0, 0.5, (1, nv)).astype(np.float32), flags=FLAG_REPLICATED))
            t.append(_norm   (rng, p+"ssm_norm.weight",  ROLE_SSM_NORM,   l, hv))
            t.append(_ternary(rng, p+"ssm_out.weight",   ROLE_SSM_OUT,    l, cfg.n_embd,  nvh))
        else:
            t.append(_ternary(rng, p+"attn_q.weight",      ROLE_ATTN_Q,      l, 2*cfg.q_dim, cfg.n_embd))
            t.append(_ternary(rng, p+"attn_k.weight",      ROLE_ATTN_K,      l, cfg.kv_dim,  cfg.n_embd))
            t.append(_ternary(rng, p+"attn_v.weight",      ROLE_ATTN_V,      l, cfg.kv_dim,  cfg.n_embd))
            t.append(_norm   (rng, p+"attn_q_norm.weight", ROLE_ATTN_Q_NORM, l, cfg.head_dim))
            t.append(_norm   (rng, p+"attn_k_norm.weight", ROLE_ATTN_K_NORM, l, cfg.head_dim))
            t.append(_ternary(rng, p+"attn_output.weight", ROLE_ATTN_OUT,    l, cfg.n_embd,  cfg.q_dim))
        t.append(_norm   (rng, p+"post_attention_norm.weight", ROLE_FFN_NORM, l, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_gate.weight", ROLE_FFN_GATE, l, cfg.ffn_dim, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_up.weight",   ROLE_FFN_UP,   l, cfg.ffn_dim, cfg.n_embd))
        t.append(_ternary(rng, p+"ffn_down.weight", ROLE_FFN_DOWN, l, cfg.n_embd,  cfg.ffn_dim))
    t.append(_norm(rng, "output_norm.weight", ROLE_OUT_NORM, 0, cfg.n_embd))
    t.append(_ternary(rng, "output.weight", ROLE_LM_HEAD, 0, cfg.vocab, cfg.n_embd))
    # The rotation tensors, exactly as convert._hadamard_specs builds them.
    from trixium.model.convert import _hadamard_specs
    t += _hadamard_specs(cfg)
    if mtp:
        t += make_tiny_mtp_tensors(cfg, rng, mtp)     # after everything: the main tensors
    return t                                          # stay those of the image without

def generate(out_dir, n_channels=2, seed=1234, cfg=BONSAI_TINY, kseg_max=None, mtp=None):
    """kseg_max: split matvecs wider than this into K-segments (the 27B's FFN down projection
    path, exercised on the tiny model with e.g. 256); the image is named `-kseg`.
    mtp: add an MTP head of that kind ("q8"/"ternary", hybrid configs); the image is named
    `-mtp`."""
    import dataclasses
    os.makedirs(out_dir, exist_ok=True)
    suffix = ""
    if mtp:
        assert cfg.layer_types, "the MTP head is a Qwen3.5 (hybrid) feature"
        cfg = dataclasses.replace(cfg)
        cfg.mtp_layers = 1
        suffix = "-mtp"
    if cfg.layer_types:
        tensors = make_tiny_hybrid_tensors(cfg, seed, mtp)
    else:
        tensors = make_tiny_tensors(cfg, seed)
    if kseg_max is not None:
        from trixium.model.bons import split_wide
        tensors = [seg for t in tensors for seg in split_wide(t, kseg_max)]
        suffix += "-kseg"
    img_path = os.path.join(out_dir, f"{cfg.name}{suffix}-n{n_channels}.img")
    man_path = os.path.join(out_dir, f"{cfg.name}{suffix}-n{n_channels}.json")
    manifest = write_image(cfg, tensors, n_channels, img_path, manifest_path=man_path)
    return img_path, man_path, manifest

def main():
    parser = argparse.ArgumentParser(description="Generate the bonsai-tiny random model.")
    parser.add_argument("--out",      default="models/tiny", help="Output directory.")
    parser.add_argument("--channels", default=2, type=int,   help="Channel shard count.")
    parser.add_argument("--seed",     default=1234, type=int)
    parser.add_argument("--hybrid",   action="store_true", help="bonsai-tiny-hybrid (the 27B's shape).")
    parser.add_argument("--hybrid-1k", action="store_true",
                        help="bonsai-tiny-hybrid-1k: 1,024-wide rotation blocks (the resident path).")
    parser.add_argument("--kseg-max", default=None, type=int, help="Split matvecs wider than this (K-segments).")
    parser.add_argument("--mtp", default=None, choices=["q8", "ternary"],
                        help="Add an MTP head (speculative decoding) with matrices of this kind.")
    args = parser.parse_args()
    from trixium.model.configs import BONSAI_TINY_HYBRID, BONSAI_TINY_HYBRID_1K
    cfg = BONSAI_TINY_HYBRID_1K if args.hybrid_1k else BONSAI_TINY_HYBRID if args.hybrid else BONSAI_TINY
    img, man, manifest = generate(args.out, args.channels, args.seed, cfg=cfg, kseg_max=args.kseg_max, mtp=args.mtp)
    print(f"image    : {img} ({os.path.getsize(img)/1e6:.2f} MB)")
    print(f"manifest : {man}")
    print(f"payload  : {manifest['payload_beats']} beats/channel "
          f"({manifest['payload_bytes_per_channel']/1e6:.2f} MB/channel)")

if __name__ == "__main__":
    main()

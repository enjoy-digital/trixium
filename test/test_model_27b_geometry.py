#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The 27B's geometry through the GGUF reader: what the converter learns before it refuses.

Bonsai 2 (Qwen3.5-27B; the GGUF architecture is `qwen35`) is a hybrid: 48 Gated DeltaNet
layers and 16 gated-attention layers (every fourth), hidden 5120, FFN 17408, a 248,320
vocabulary; attention 24 query / 4 KV heads at head dim 256 with a per-head output gate
and partial RoPE over 64 dims; DeltaNet 48 value heads over 16 key heads at 128, conv
kernel 4. Its ternary matrices are stored in PrismML's rotated basis (Hadamard block 1024,
one sign vector per input width). The metadata here is what the real file declares
(models/27b); the sign vectors are shortened. What is pinned: the layer pattern,
the dimensions the engine shards (multiples of 128 and of 16 channels), the gate detection
from the q projection's shape, the Hadamard bookkeeping, and that the converter refuses the
model with the reason instead of building a wrong image.
"""

import pytest

from model.configs import BONSAI_8B
from model.golden import config_from_gguf, check_engine_support


# Helpers ------------------------------------------------------------------------------------------

class _Tensor:
    def __init__(self, shape):
        self.shape = shape

class _Reader:
    def __init__(self, metadata, tensors):
        self.metadata, self.tensors = metadata, tensors


def _bonsai2_27b():
    a = "qwen35"
    widths = [5120, 6144, 17408]
    signs = [1 if (i*7919) % 3 else -1 for i in range(sum(widths))]
    md = {
        "general.architecture":            a,
        "general.basename":                "folded",
        "general.size_label":              "27B",
        f"{a}.block_count":                64,
        f"{a}.embedding_length":           5120,
        f"{a}.attention.head_count":       24,
        f"{a}.attention.head_count_kv":    4,
        f"{a}.attention.key_length":       256,
        f"{a}.attention.value_length":     256,
        f"{a}.feed_forward_length":        17408,
        f"{a}.full_attention_interval":    4,
        f"{a}.ssm.time_step_rank":         48,    # DeltaNet value heads.
        f"{a}.ssm.group_count":            16,    # DeltaNet key heads.
        f"{a}.ssm.state_size":             128,   # Key head dim.
        f"{a}.ssm.inner_size":             6144,  # Value heads x value head dim.
        f"{a}.ssm.conv_kernel":            4,
        f"{a}.rope.dimension_count":       64,    # Partial RoPE (0.25 of 256).
        f"{a}.rope.freq_base":             10000000.0,
        f"{a}.attention.layer_norm_rms_epsilon": 1e-6,
        "prism.hadamard.transform":        "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis":             "input-last-dimension",
        "prism.hadamard.block_size":       1024,
        "prism.hadamard.sign_widths":      widths,
        "prism.hadamard.sign_values":      signs,
    }
    tensors = {"token_embd.weight": _Tensor([248320, 5120]),
               "output.weight":     _Tensor([248320, 5120]),
               "blk.3.attn_q.weight": _Tensor([12288, 5120]),   # [q | gate]: 2 x 24 x 256 rows.
               "blk.3.attn_k.weight": _Tensor([1024, 5120])}
    return _Reader(md, tensors)


# Tests --------------------------------------------------------------------------------------------

def test_27b_layer_pattern_and_dimensions():
    cfg = config_from_gguf(_bonsai2_27b())
    assert cfg.name == "bonsai-27b"
    assert cfg.n_layer == 64 and cfg.n_dn_layers == 48 and cfg.n_attn_layers == 16
    assert [cfg.layer_type(l) for l in range(8)] == ["deltanet"]*3 + ["attn"] + ["deltanet"]*3 + ["attn"]
    assert cfg.head_dim == 256 and cfg.n_kv_head == 4 and cfg.kv_dim == 1024 and cfg.q_dim == 6144
    assert cfg.attn_gated
    assert cfg.rope_dims == 64 and cfg.rope_theta == 10000000.0
    assert (cfg.dn_num_v_heads, cfg.dn_v_head_dim, cfg.dn_num_k_heads, cfg.dn_k_head_dim,
            cfg.dn_conv_kernel) == (48, 128, 16, 128, 4)
    assert not cfg.tie_embeddings
    assert cfg.hadamard_block == 1024 and sorted(cfg.hadamard_signs) == [5120, 6144, 17408]
    assert all(len(s) == w and set(s) <= {1, -1} for w, s in cfg.hadamard_signs.items())
    # The engine shards output rows over 16 channels and streams 128-wide ternary groups.
    for rows in (cfg.q_dim, cfg.kv_dim, cfg.n_embd, cfg.ffn_dim):
        assert rows % 16 == 0
    for cols in (cfg.n_embd, cfg.ffn_dim, cfg.head_dim):
        assert cols % 128 == 0
    assert cfg.vocab_padded(16) == 248320
    # 48 DeltaNet layers x 48 value heads = 2304 whole 128x128 FP32 matrices;
    # striped evenly over 16 channels, that is 144 x 64 KiB per channel.
    from model import layout
    assert layout.deltanet_state_bytes_per_channel(cfg, 16) == 144*128*128*4


def test_27b_and_8b_are_supported():
    check_engine_support(config_from_gguf(_bonsai2_27b()))
    check_engine_support(BONSAI_8B)


def test_qwen3_models_have_no_layer_types():
    assert BONSAI_8B.layer_types == () and BONSAI_8B.n_dn_layers == 0
    assert BONSAI_8B.layer_type(0) == "attn"


def test_hadamard_rotation_algebra():
    """The fold's algebra: with R the block Sylvester matrix and s the signs, a weight stored
    as W' = W diag(s) R sees its input as R (s * x) and reproduces W x; the embedding table
    stored as z = R (s * e) is restored by s * (R z)."""
    import numpy as np
    from model.golden import hadamard_matrix, hadamard_rotate, hadamard_restore
    rng   = np.random.default_rng(3)
    block = 16
    width = 3*block
    signs = rng.choice([-1, 1], width)
    R     = hadamard_matrix(block)
    assert np.allclose(R @ R, np.eye(block), atol=1e-6)          # symmetric involution
    assert R[0, 0] == 0.25 and R[1, 1] == -0.25 and R[1, 2] == 0.25
    x = rng.standard_normal(width).astype(np.float32)
    W = rng.standard_normal((5, width)).astype(np.float32)
    Rfull = np.kron(np.eye(width//block), R)
    Wf = W @ np.diag(signs) @ Rfull
    assert np.allclose(Wf @ hadamard_rotate(x, signs, block), W @ x, atol=1e-4)
    e = rng.standard_normal((4, width)).astype(np.float32)
    assert np.allclose(hadamard_restore(hadamard_rotate(e, signs, block), signs, block), e, atol=1e-5)
    assert np.allclose(hadamard_rotate(hadamard_restore(e, signs, block), signs, block), e, atol=1e-5)


def test_hadamard_block_tensors_match_the_golden():
    """The converter's ROLE_HADAMARD block tensors, dequantized and applied blockwise, are
    the golden's hadamard_rotate - so an engine matvec per block rotates like the reference."""
    import numpy as np
    from model import layout
    from model.configs import BONSAI_TINY_HYBRID
    from model.tinygen import make_tiny_hybrid_tensors
    from model.golden import hadamard_rotate
    cfg   = BONSAI_TINY_HYBRID
    specs = {s.name: s for s in make_tiny_hybrid_tensors(cfg, 7)}
    rng   = np.random.default_rng(9)
    for wi, width in enumerate(sorted(cfg.hadamard_signs)):
        x = rng.standard_normal(width).astype(np.float32)
        want = hadamard_rotate(x, cfg.hadamard_signs[width], cfg.hadamard_block)
        got = np.zeros(width, np.float32)
        B = cfg.hadamard_block
        for b in range(width//B):
            s = specs[f"had.{width}.{b}"]
            assert s.layer == layout.hadamard_layer(wi, b)
            T = (s.get_values().astype(np.float32).reshape(B, -1, 128) - 1.0) \
                * s.scales.astype(np.float32)[:, :, None]
            got[b*B:(b + 1)*B] = T.reshape(B, B) @ x[b*B:(b + 1)*B]
        # fp16 holds 1/sqrt(128) to 1e-4 (the 27B's 1/32 exactly).
        assert np.allclose(got, want, rtol=2e-3, atol=1e-3), width

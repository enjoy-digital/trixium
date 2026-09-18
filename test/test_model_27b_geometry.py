#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The 27B's geometry through the GGUF reader: what the converter learns before it refuses.

Bonsai 2 (Qwen3.8-27B) is a hybrid: Gated DeltaNet layers with full attention every fourth
layer, hidden 5120, FFN 17408, head_dim 256, a 248k vocabulary. No 27B file is on this
workstation (5.9 GB; the disk was full on 2026-09-18), so the metadata here is synthetic,
in llama.cpp's key convention for the qwen3next architecture; the DeltaNet head counts are
Qwen3-Next's and stand in for the 27B's until its file is read. What is pinned: the layer
pattern, the dimensions the engine shards (all multiples of 128 and of 16 channels), and
that the converter refuses the model with a reason instead of building a wrong image.
"""

import pytest

from trixium.model.configs import BONSAI_8B
from trixium.model.golden import config_from_gguf, check_engine_support


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

class _Reader:
    def __init__(self, metadata, tensors):
        self.metadata, self.tensors = metadata, tensors


def _bonsai2_27b():
    a = "qwen3next"
    md = {
        "general.architecture":            a,
        "general.basename":                "Bonsai-27B",
        f"{a}.block_count":                64,
        f"{a}.embedding_length":           5120,
        f"{a}.attention.head_count":       16,
        f"{a}.attention.head_count_kv":    2,
        f"{a}.attention.key_length":       256,
        f"{a}.feed_forward_length":        17408,
        f"{a}.full_attention_interval":    4,
        f"{a}.ssm.time_step_rank":         32,    # DeltaNet value heads.
        f"{a}.ssm.group_count":            16,    # DeltaNet key heads.
        f"{a}.ssm.state_size":             128,   # Key head dim.
        f"{a}.ssm.inner_size":             4096,  # Value heads x value head dim.
        f"{a}.ssm.conv_kernel":            4,
        f"{a}.rope.dimension_count":       64,    # Partial RoPE (0.25 of 256).
        f"{a}.rope.freq_base":             10000000.0,
        f"{a}.attention.layer_norm_rms_epsilon": 1e-6,
    }
    tensors = {"token_embd.weight": _Tensor([248320, 5120]),
               "output.weight":     _Tensor([248320, 5120])}
    return _Reader(md, tensors)


def test_27b_layer_pattern_and_dimensions():
    cfg = config_from_gguf(_bonsai2_27b())
    assert cfg.n_layer == 64 and cfg.n_dn_layers == 48 and cfg.n_attn_layers == 16
    assert [cfg.layer_type(l) for l in range(8)] == ["deltanet"]*3 + ["attn"] + ["deltanet"]*3 + ["attn"]
    assert cfg.head_dim == 256 and cfg.n_kv_head == 2 and cfg.kv_dim == 512 and cfg.q_dim == 4096
    assert cfg.rope_dims == 64 and cfg.rope_theta == 10000000.0
    assert (cfg.dn_num_v_heads, cfg.dn_v_head_dim, cfg.dn_num_k_heads, cfg.dn_k_head_dim,
            cfg.dn_conv_kernel) == (32, 128, 16, 128, 4)
    assert not cfg.tie_embeddings
    # The engine shards output rows over 16 channels and streams 128-wide ternary groups.
    for rows in (cfg.q_dim, cfg.kv_dim, cfg.n_embd, cfg.ffn_dim):
        assert rows % 16 == 0
    for cols in (cfg.n_embd, cfg.ffn_dim, cfg.head_dim):
        assert cols % 128 == 0
    assert cfg.vocab_padded(16) == 248320


def test_27b_is_refused_with_the_reason_and_the_8b_is_not():
    cfg = config_from_gguf(_bonsai2_27b())
    with pytest.raises(SystemExit) as e:
        check_engine_support(cfg)
    msg = str(e.value)
    assert "48 of 64 layers are Gated DeltaNet" in msg and "32 value heads x 128" in msg
    check_engine_support(BONSAI_8B)   # The models that run: no exception.


def test_qwen3_models_have_no_layer_types():
    assert BONSAI_8B.layer_types == () and BONSAI_8B.n_dn_layers == 0
    assert BONSAI_8B.layer_type(0) == "attn"

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Model configurations (Qwen3-family geometry used by the Ternary-Bonsai models)."""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    name           : str
    n_layer        : int
    n_embd         : int
    n_head         : int
    n_kv_head      : int
    head_dim       : int
    ffn_dim        : int
    vocab          : int           # Real vocab size (may be odd, e.g. Qwen3's 151669).
    rope_theta     : float = 1000000.0
    rms_eps        : float = 1e-6
    max_ctx        : int   = 2048  # Phase-1 KV budget, not the model's native limit.
    tie_embeddings : bool  = False
    # YaRN rope scaling (Ternary-Bonsai GGUFs ship yarn factor 4 over 8192). Pure table math:
    # host/firmware fold it into the precomputed sin/cos tables, no hardware impact.
    yarn_factor    : float = 1.0
    yarn_orig_ctx  : int   = 0
    # Bonsai 2 / Qwen3.8 (the 27B): a hybrid of Gated DeltaNet layers with full attention
    # every full_attention_interval-th layer. () = every layer is attention (the Qwen3
    # models). The DeltaNet geometry is carried so the converter and the firmware can say
    # exactly what the missing recurrent unit has to compute (docs/xcu1525.md).
    layer_types    : tuple = ()           # "attn" / "deltanet" per layer, or ().
    dn_num_v_heads : int   = 0            # DeltaNet value heads (GGUF ssm.time_step_rank).
    dn_num_k_heads : int   = 0            # DeltaNet key heads (ssm.group_count).
    dn_k_head_dim  : int   = 0            # Key/query head dim (ssm.state_size).
    dn_v_head_dim  : int   = 0            # Value head dim (ssm.inner_size / v heads).
    dn_conv_kernel : int   = 0            # Short causal conv kernel (ssm.conv_kernel).
    rope_dims      : int   = 0            # Partial RoPE: rotated dims per head (0 = all).

    def layer_type(self, l):
        return self.layer_types[l] if self.layer_types else "attn"

    @property
    def n_dn_layers(self):
        return sum(1 for t in self.layer_types if t == "deltanet")

    @property
    def n_attn_layers(self):
        return self.n_layer - self.n_dn_layers

    @property
    def q_dim(self):
        return self.n_head*self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_head*self.head_dim

    def vocab_padded(self, n_channels):
        """Vocab-sharded tensors (embeddings/LM head) are row-padded to divide the channel
        count; the top-k unit masks rows >= vocab so pad rows can never win."""
        return (self.vocab + n_channels - 1)//n_channels*n_channels

# bonsai-tiny: random-weight config for simulation (2 layers, dims chosen so every sharded
# tensor divides by the sim channel count and every ternary K divides the 128 group).
BONSAI_TINY = ModelConfig(
    name      = "bonsai-tiny",
    n_layer   = 2,
    n_embd    = 256,
    n_head    = 4,
    n_kv_head = 2,
    head_dim  = 64,
    ffn_dim   = 512,
    vocab     = 512,
    max_ctx   = 64,
)

# Ternary-Bonsai-1.7B (verified against the GGUF metadata: qwen3, 28 blocks, 2048 embd,
# 16Q/8KV heads x 128, ffn 6144, vocab 151669 (odd - padded at conversion), rope 1e6 with
# yarn x4 over 8192, rms_eps 1e-6, tied ternary embeddings, no output.weight).
BONSAI_1P7B = ModelConfig(
    name           = "bonsai-1.7b",
    n_layer        = 28,
    n_embd         = 2048,
    n_head         = 16,
    n_kv_head      = 8,
    head_dim       = 128,
    ffn_dim        = 6144,
    vocab          = 151669,
    rms_eps        = 1e-6,
    tie_embeddings = True,
    yarn_factor    = 4.0,
    yarn_orig_ctx  = 8192,
)

# Ternary-Bonsai-8B (verified against the GGUF metadata: qwen3, 36 blocks, 4096 embd,
# 32Q/8KV heads x 128, ffn 12288, vocab 151669, yarn x4 over 16384, untied - separate ternary
# output.weight).
BONSAI_8B = ModelConfig(
    name          = "bonsai-8b",
    n_layer       = 36,
    n_embd        = 4096,
    n_head        = 32,
    n_kv_head     = 8,
    head_dim      = 128,
    ffn_dim       = 12288,
    vocab         = 151669,
    yarn_factor   = 4.0,
    yarn_orig_ctx = 16384,
)

CONFIGS = {c.name: c for c in (BONSAI_TINY, BONSAI_1P7B, BONSAI_8B)}

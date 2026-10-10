#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Model configurations (Qwen3-family geometry used by the Ternary-Bonsai models)."""

from dataclasses import dataclass

# Model Config -------------------------------------------------------------------------------------

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
    max_ctx        : int   = 2048  # KV budget, not the model's native limit.
    tie_embeddings : bool  = False
    # YaRN rope scaling (Ternary-Bonsai GGUFs ship yarn factor 4 over 8192). Pure table math:
    # host/firmware fold it into the precomputed sin/cos tables, no hardware impact.
    yarn_factor    : float = 1.0
    yarn_orig_ctx  : int   = 0
    # Bonsai 2 / Qwen3.5 (the 27B): a hybrid of Gated DeltaNet layers with full attention
    # every full_attention_interval-th layer. () = every layer is attention (the Qwen3
    # models). The DeltaNet geometry sizes the recurrent state for the converter and the
    # firmware (doc/boards.md).
    layer_types    : tuple = ()           # "attn" / "deltanet" per layer, or ().
    dn_num_v_heads : int   = 0            # DeltaNet value heads (GGUF ssm.time_step_rank).
    dn_num_k_heads : int   = 0            # DeltaNet key heads (ssm.group_count).
    dn_k_head_dim  : int   = 0            # Key/query head dim (ssm.state_size).
    dn_v_head_dim  : int   = 0            # Value head dim (ssm.inner_size / v heads).
    dn_conv_kernel : int   = 0            # Short causal conv kernel (ssm.conv_kernel).
    rope_dims      : int   = 0            # Partial RoPE: rotated dims per head (0 = all).
    attn_gated     : bool  = False        # attn_q also emits a per-head sigmoid output gate.
    # PrismML's rotated weight basis (the 27B): every ternary matrix W is stored as
    # W·H_s, a blockwise signed Walsh-Hadamard rotation of its input dimension (block
    # hadamard_block, one fixed ±1 sign vector per input width), and the runtime applies
    # H_s to the activation before the matvec; token_embd is stored with the inverse.
    hadamard_block : int   = 0            # 0 = no rotation (the 1.7B/8B files).
    hadamard_signs : dict  = None         # {input width: tuple of ±1}, from the GGUF.

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
        count; the firmware never picks a row >= vocab, so pad rows can never win."""
        return (self.vocab + n_channels - 1)//n_channels*n_channels

# Configs ------------------------------------------------------------------------------------------

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

# bonsai-tiny-hybrid: the 27B's shape at sim size - Gated DeltaNet layers with an attention
# layer every 3rd (layer 2), gated attention with partial RoPE, PrismML's rotated basis with
# a 128-block. Random weights; the golden defines the semantics (the "folded" weights are
# simply what the golden multiplies the rotated activations by).
BONSAI_TINY_HYBRID = ModelConfig(
    name           = "bonsai-tiny-hybrid",
    n_layer        = 3,
    n_embd         = 256,
    n_head         = 4,
    n_kv_head      = 2,
    head_dim       = 64,
    ffn_dim        = 512,
    vocab          = 512,
    max_ctx        = 64,
    rope_theta     = 10000000.0,
    layer_types    = ("deltanet", "deltanet", "attn"),
    dn_num_v_heads = 8,      # 8 rows shard over the n8 sim (the 27B: 48 over 16)
    dn_num_k_heads = 2,
    dn_k_head_dim  = 64,
    dn_v_head_dim  = 64,
    dn_conv_kernel = 4,
    rope_dims      = 16,
    attn_gated     = True,
    hadamard_block = 128,
    hadamard_signs = None,   # filled by tinygen (deterministic from the seed)
)

# bonsai-tiny-hybrid-1k: the same topology with every rotation width a multiple of 1,024,
# the hardware FWHT block (units 3/5 of the resident path are fixed at N=1024; the 27B's
# widths 5120/4096/6144/17408 all are). Sized for the sim: n_embd 1024, ffn 2048, q 1024,
# DeltaNet value width 1024. ~33 M weights; a forward costs minutes of Verilator time.
BONSAI_TINY_HYBRID_1K = ModelConfig(
    name           = "bonsai-tiny-hybrid-1k",
    n_layer        = 3,
    n_embd         = 1024,
    n_head         = 8,
    n_kv_head      = 2,
    head_dim       = 128,
    ffn_dim        = 2048,
    vocab          = 512,
    max_ctx        = 64,
    rope_theta     = 10000000.0,
    layer_types    = ("deltanet", "deltanet", "attn"),
    dn_num_v_heads = 8,
    dn_num_k_heads = 2,
    dn_k_head_dim  = 128,
    dn_v_head_dim  = 128,
    dn_conv_kernel = 4,
    rope_dims      = 32,
    attn_gated     = True,
    hadamard_block = 1024,
    hadamard_signs = None,
)

# Shape-true slices: random weights at the real models' widths and head counts, a few layers and
# a truncated vocabulary, for the board-timing simulation (trixium_sim.py --board-timing): the
# per-layer cycles are the board's, the full model is extrapolated from them (doc/simulation.md).
# max_ctx only sizes the KV/rope tables (the bench runs a few positions).
BONSAI_27B_SLICE = ModelConfig(
    name           = "bonsai-27b-slice",
    n_layer        = 4,
    n_embd         = 5120,
    n_head         = 24,
    n_kv_head      = 4,
    head_dim       = 256,
    ffn_dim        = 17408,
    vocab          = 29440,  # The 27B: 248320 (both: whole LM-head chunks + one of 7552 rows).
    max_ctx        = 256,
    rope_theta     = 10000000.0,
    layer_types    = ("deltanet", "deltanet", "deltanet", "attn"),
    dn_num_v_heads = 48,
    dn_num_k_heads = 16,
    dn_k_head_dim  = 128,
    dn_v_head_dim  = 128,
    dn_conv_kernel = 4,
    rope_dims      = 64,
    attn_gated     = True,
    hadamard_block = 1024,
    hadamard_signs = None,   # filled by tinygen
)

BONSAI_8B_SLICE = ModelConfig(
    name          = "bonsai-8b-slice",
    n_layer       = 4,
    n_embd        = 4096,
    n_head        = 32,
    n_kv_head     = 8,
    head_dim      = 128,
    ffn_dim       = 12288,
    vocab         = 8192,    # The 8B: 151669.
    max_ctx       = 256,
    yarn_factor   = 4.0,
    yarn_orig_ctx = 16384,
)

BONSAI_1P7B_SLICE = ModelConfig(
    name           = "bonsai-1p7b-slice",
    n_layer        = 4,
    n_embd         = 2048,
    n_head         = 16,
    n_kv_head      = 8,
    head_dim       = 128,
    ffn_dim        = 6144,
    vocab          = 8192,   # The 1.7B: 151669.
    max_ctx        = 256,
    tie_embeddings = True,
    yarn_factor    = 4.0,
    yarn_orig_ctx  = 8192,
)

# The full model a slice stands for: per-layer cycles x layers + the LM head scaled by the vocab.
SLICE_OF = {
    "bonsai-27b-slice"  : dict(model="27B",  n_dn=48, n_attn=16, vocab=248320),
    "bonsai-8b-slice"   : dict(model="8B",   n_dn=0,  n_attn=36, vocab=151669),
    "bonsai-1p7b-slice" : dict(model="1.7B", n_dn=0,  n_attn=28, vocab=151669),
}

CONFIGS = {c.name: c for c in (BONSAI_TINY, BONSAI_1P7B, BONSAI_8B, BONSAI_TINY_HYBRID,
                               BONSAI_TINY_HYBRID_1K, BONSAI_27B_SLICE, BONSAI_8B_SLICE,
                               BONSAI_1P7B_SLICE)}

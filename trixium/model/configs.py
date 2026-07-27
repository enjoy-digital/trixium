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
    vocab          : int
    rope_theta     : float = 1000000.0
    rms_eps        : float = 1e-6
    max_ctx        : int   = 2048  # Phase-1 KV budget, not the model's native limit.
    tie_embeddings : bool  = False

    @property
    def q_dim(self):
        return self.n_head*self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_head*self.head_dim

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

# Ternary-Bonsai-1.7B (Qwen3-1.7B geometry; cross-checked against the GGUF metadata by
# convert.py at conversion time).
BONSAI_1P7B = ModelConfig(
    name           = "bonsai-1.7b",
    n_layer        = 28,
    n_embd         = 2048,
    n_head         = 16,
    n_kv_head      = 8,
    head_dim       = 128,
    ffn_dim        = 6144,
    vocab          = 151936,
    tie_embeddings = True,
)

# Ternary-Bonsai-8B (Qwen3-8B geometry).
BONSAI_8B = ModelConfig(
    name      = "bonsai-8b",
    n_layer   = 36,
    n_embd    = 4096,
    n_head    = 32,
    n_kv_head = 8,
    head_dim  = 128,
    ffn_dim   = 12288,
    vocab     = 151936,
)

CONFIGS = {c.name: c for c in (BONSAI_TINY, BONSAI_1P7B, BONSAI_8B)}

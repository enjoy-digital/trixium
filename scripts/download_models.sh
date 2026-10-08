#!/bin/bash
#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# Download the PrismML Ternary-Bonsai models used by TriXium:
# - GGUF PQ2_0 (ggml type 142: 2-bit codes + one fp16 scale per 128 weights) - the block
#   earlier releases shipped as "Q2_0" (those files still convert; the *_Q2_0_g64 files do
#   not: two scales per 128 weights). 1.7B and 8B, both run on the card.
# - Tokenizer + config from the matching "unpacked" repos.
#
#   scripts/download_models.sh        (or: make models)

set -eu
cd "$(dirname "$0")/.."

HF=.venv/bin/hf
MODELS_DIR=models

# Disk guard: refuse to start with less than 8 GB free.
avail=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
if [ "$avail" -lt 8 ]; then
    echo "ERROR: only ${avail}G free, need >= 8G. Run 'make clean-vivado' first."
    exit 1
fi

"$HF" download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-PQ2_0.gguf \
    --local-dir "$MODELS_DIR/1.7b"
"$HF" download prism-ml/Ternary-Bonsai-1.7B-unpacked config.json tokenizer.json tokenizer_config.json \
    --local-dir "$MODELS_DIR/1.7b"

"$HF" download prism-ml/Ternary-Bonsai-8B-gguf Ternary-Bonsai-8B-PQ2_0.gguf \
    --local-dir "$MODELS_DIR/8b"
"$HF" download prism-ml/Ternary-Bonsai-8B-unpacked config.json tokenizer.json tokenizer_config.json \
    --local-dir "$MODELS_DIR/8b"

echo "== Models:"
find "$MODELS_DIR" -type f -name "*.gguf" -o -name "*.json" | sort
df -h . | tail -1

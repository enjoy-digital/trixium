#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Export a HuggingFace byte-level BPE tokenizer to a flat binary for the C host tool.

The C side should not have to know about the GPT-2 byte<->unicode mapping or JSON, so every
token is stored as its RAW BYTES here and merges as (id_a, id_b) -> id_ab rank pairs.

  tokenizer.bin:
    magic "BTOK", version, n_vocab, n_merges
    n_vocab x { u32 len, len bytes }            (token id order)
    n_merges x { u32 a, u32 b, u32 result }     (rank order = merge priority)

Also writes tokenizer_tests.txt: prompts encoded by the reference tokenizer, so `trixium_chat -T`
can prove the C encoder agrees with HuggingFace.

  trixium tokenizer models/1.7b/tokenizer.json   # -> models/1.7b/tokenizer.bin
"""

import os
import json
import struct
import hashlib

# Helpers ------------------------------------------------------------------------------------------

def byte_to_unicode():
    """GPT-2 byte <-> unicode table (the tokenizer stores tokens in this mapped alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
       + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

# Reference Prompts --------------------------------------------------------------------------------

TEST_PROMPTS = [
    "The capital of France is",
    "Hello world!",
    "def fibonacci(n):",
    "1234 + 5678 = ",
    "  leading spaces and\ttabs\n",
    "Emoji: \U0001F600 and accents: café naïve",
    "A" * 40,
    "The quick brown fox jumps over the lazy dog. It was 2026, and the FK33 ran at 200 MHz.",
]

# Export -------------------------------------------------------------------------------------------

def export_tokenizer(tokenizer_json, out, tests=None):
    """tokenizer.json -> tokenizer.bin (+ tokenizer_tests.txt, default: next to `out`)."""
    tests = tests or os.path.join(os.path.dirname(os.path.abspath(out)), "tokenizer_tests.txt")
    with open(tokenizer_json, encoding="utf-8") as f:
        d = json.load(f)
    model = d["model"]
    if model["type"] != "BPE":
        raise ValueError(f"unsupported tokenizer type {model['type']}")

    b2u = byte_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def token_to_bytes(tok):
        """Mapped-alphabet token string -> the raw bytes it represents."""
        raw = bytearray()
        for ch in tok:
            if ch not in u2b:
                return None  # Not in the byte alphabet (added/special token): keep as UTF-8.
            raw.append(u2b[ch])
        return bytes(raw)

    vocab = model["vocab"]  # token string -> id
    added = d.get("added_tokens", [])
    # The added tokens sit ABOVE the BPE vocab (Qwen3: 151643..151668 over a 151643-entry
    # vocab): sizing the table by the vocab alone would drop every one of them, and the C tool
    # would have no <|im_start|>/<|im_end|> to build a chat turn or to stop on.
    n_vocab = max([max(vocab.values())] + [at["id"] for at in added]) + 1
    toks = [b""]*n_vocab
    for tok, tid in vocab.items():
        raw = token_to_bytes(tok)
        toks[tid] = raw if raw is not None else tok.encode("utf-8")
    # Added/special tokens (<|im_start|> etc.) are literal UTF-8.
    for at in added:
        toks[at["id"]] = at["content"].encode("utf-8")

    merges = []
    for m in model.get("merges", []):
        a, b = (m.split(" ", 1) if isinstance(m, str) else (m[0], m[1]))
        ia, ib, iab = vocab.get(a), vocab.get(b), vocab.get(a + b)
        if ia is None or ib is None or iab is None:
            continue
        merges.append((ia, ib, iab))

    # BTOK v2: after the counts, the SHA-256 of the tokenizer.json this blob
    # came from - the same hash the converter writes into every BONS header (--tokenizer),
    # so trixium_chat can tell a model and a tokenizer apart before the first prompt.
    with open(tokenizer_json, "rb") as f:
        src_sha = hashlib.sha256(f.read()).digest()
    with open(out, "wb") as f:
        f.write(b"BTOK")
        f.write(struct.pack("<III", 2, n_vocab, len(merges)))
        f.write(src_sha)
        for t in toks:
            f.write(struct.pack("<I", len(t)))
            f.write(t)
        for a, b, ab in merges:
            f.write(struct.pack("<III", a, b, ab))
    print(f"{out}: {n_vocab} tokens, {len(merges)} merges")

    # Reference encodings so the C encoder can be checked against HuggingFace.
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tokenizer_json)
        with open(tests, "w", encoding="utf-8") as f:
            for p in TEST_PROMPTS:
                ids = tk.encode(p, add_special_tokens=False).ids
                f.write(json.dumps(p) + "\t" + " ".join(str(i) for i in ids) + "\n")
        print(f"{tests}: {len(TEST_PROMPTS)} reference encodings")
    except ImportError:
        print("(tokenizers not installed - skipped reference encodings)")

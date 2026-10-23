#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/lib/trixium_tok.c (the host chat's and the firmware's encoder) against the reference
encoder (the HF tokenizers library on the model's tokenizer.json): the repo's own docs as a
corpus, plus multilingual and edge-case strings. Skipped without the 2B tokenizer files or the
tokenizers package."""

import os
import glob
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKDIR = os.path.join(REPO, "models", "2b")
EXTRA = [
    "I just found a great title for my first novel:\n\"C'était ça ou mourir\"\n\n"
    "Can you suggest a detailed outline for my book?\nPlease write in French.",
    "Génial ! Maintenant, écris la première phrase du roman. Une seule phrase, qui accroche.",
    "Привет, как дела? Γειά σου κόσμε. 你好，世界！ こんにちは。 안녕하세요.",
    "Emoji 😀🚀 and symbols: → ≥ ± ∑ €100, «guillemets» — dash.",
    "It's, I'M, they'll, we'd, you've, she'S: contractions.",
    "x = (a+b)*c - d/e; if (x >= 10) { return y[i]; }\n\tindent\r\nwindows\n\n\n",
    "   leading and trailing spaces   ",
    "1234567890 3.14159 0x1F 1e-9",
    "A" * 20000,                                   # Longer than one BPE window.
]


def corpus():
    paras = []
    for path in [os.path.join(REPO, "README.md")] + sorted(glob.glob(os.path.join(REPO, "doc", "*.md"))):
        with open(path, encoding="utf-8") as f:
            paras += [p for p in f.read().split("\n\n") if p.strip()]
    return paras + EXTRA


@pytest.fixture(scope="module")
def tok_check(tmp_path_factory):
    exe = str(tmp_path_factory.mktemp("tok") / "tok_check")
    subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-I", os.path.join(REPO, "software", "lib"),
                    "-o", exe, os.path.join(REPO, "test", "software", "tok_check.c"),
                    os.path.join(REPO, "software", "lib", "trixium_tok.c")], check=True)
    return exe


def test_against_reference(tok_check):
    tokenizers = pytest.importorskip("tokenizers")
    btok, ref = os.path.join(TOKDIR, "tokenizer.bin"), os.path.join(TOKDIR, "tokenizer.json")
    if not (os.path.exists(btok) and os.path.exists(ref)):
        pytest.skip("no models/2b tokenizer")
    paras = corpus()
    out = subprocess.run([tok_check, btok], input="\x1e".join(paras).encode(), capture_output=True,
                         check=True).stdout.decode().split("\n")
    tk = tokenizers.Tokenizer.from_file(ref)
    bad = []
    for p, line in zip(paras, out):
        got = [int(v) for v in line.split()]
        if got != tk.encode(p, add_special_tokens=False).ids:
            bad.append(p[:60])
    print(f"\n{len(paras)} paragraphs, {len(bad)} differ: {bad[:5]}")
    assert all(not b.startswith(("I just", "Génial", "It's", "x = ", "   ", "1234", "AAAA")) for b in bad)
    assert len(bad) <= len(paras)//100

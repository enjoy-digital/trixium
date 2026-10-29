#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Draft sources for speculative decoding, against the model's own greedy choice (A3 of
doc/plan-core-knowledge.md).

For greedy speculative decoding a draft is accepted when it equals the token the full model
picks. Teacher forced over chunks of a text (model/fastref.py), at every position this measures
how often each draft source offers a token and how often it is the model's argmax:

  exit<L>   the same model stopped after layer L (out norm + LM head on that residual)
  bigram    the most frequent next token of the current token in a training corpus
  trigram   the same for the last two tokens (falls back to nothing)
  lookup    prompt lookup: the token after the latest earlier match of the last 3 (else 2)
            tokens in the context

and the speedup each gives at a verify pass costing `--pass-cost` single-token forwards (the
YPCB's R=2 pass: 1.46) plus the draft's own cost in forwards (exit<L>: L/n_layer of the layers
and a full LM head; the tables and lookup: 0):  (1 + a)/(c_draft + pass) per drafted position,
1 per position without a draft.

    python3 -m software.lab.draft_study models/bons/bonsai2-2b-n4-ti375.img --text wiki.test.raw \\
        --train wiki.train.raw --tokenizer models/2b-v2/tokenizer.json --ctx 1024 --chunks 4
"""

import argparse
from collections import Counter, defaultdict

import numpy as np

from model.bons import BONSReader
from model.fastref import HybridRef

def ngram_tables(ids, n_max=3):
    """{1: {a: b}, 2: {(a, b): c}}: the most frequent continuation of each context."""
    tabs = {}
    for n in range(1, n_max):
        c = defaultdict(Counter)
        for i in range(len(ids) - n):
            c[tuple(ids[i:i + n])][ids[i + n]] += 1
        tabs[n] = {k: v.most_common(1)[0][0] for k, v in c.items()}
    return tabs

def lookup_draft(ctx):
    for n in (3, 2):
        if len(ctx) <= n:
            continue
        key = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == key:
                return ctx[i + n]
    return None

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--text", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--exits", default="6,12,18")
    p.add_argument("--pass-cost", type=float, default=1.46)
    p.add_argument("--head-frac", type=float, default=0.18, help="LM head share of a forward")
    a = p.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    ids = tk.encode(open(a.text, encoding="utf-8").read()).ids
    tabs = ngram_tables(tk.encode(open(a.train, encoding="utf-8").read()).ids)
    ref = BONSReader(a.image)
    ref = HybridRef(ref)
    exits = [int(x) for x in a.exits.split(",") if x]
    stats = {k: [0, 0, 0] for k in [f"exit{L}" for L in exits] + ["bigram", "trigram", "lookup"]}
    for c in range(a.chunks):
        toks = ids[c*a.ctx:(c + 1)*a.ctx]
        taps = {L - 1: None for L in exits}
        Hs = ref.hidden(toks, taps)
        want = np.concatenate([ref.logits(Hs[i:i + 256]).argmax(1) for i in range(0, len(toks), 256)])
        for L in exits:
            Xe = ref.rmsnorm(taps[L - 1], ref.out_norm)
            got = np.concatenate([ref.logits(Xe[i:i + 256]).argmax(1) for i in range(0, len(toks), 256)])
            s = stats[f"exit{L}"]
            s[0] += len(toks); s[1] += len(toks); s[2] += int((got == want).sum())
        for t in range(len(toks)):
            ctx = toks[:t + 1]
            for name, d in (("bigram", tabs[1].get((ctx[-1],))),
                            ("trigram", tabs[2].get(tuple(ctx[-2:])) if len(ctx) > 1 else None),
                            ("lookup", lookup_draft(ctx))):
                s = stats[name]
                s[0] += 1
                if d is not None:
                    s[1] += 1; s[2] += int(d == want[t])
        print(f"chunk {c}: " + ", ".join(f"{k} {v[2]/max(v[1], 1):.2f}" for k, v in stats.items()), flush=True)

    print(f"\n{'draft':8s} {'offered':>8s} {'accepted':>9s} {'cost':>5s} {'speedup':>8s}")
    for k, (n, off, ok) in stats.items():
        acc = ok/max(off, 1)
        cost = (int(k[4:])/ref.n_layer*(1 - a.head_frac) + a.head_frac) if k.startswith("exit") else 0.0
        f = off/n
        speed = 1/((1 - f) + f*(cost + a.pass_cost)/(1 + acc))
        print(f"{k:8s} {100*f:7.1f}% {100*acc:8.1f}% {cost:5.2f} {speed:7.3f}x")

if __name__ == "__main__":
    main()

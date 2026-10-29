#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""kNN-LM on a hybrid image: knowledge outside the weights, fetched per token (B2 of
doc/plan-core-knowledge.md).

The datastore: (final hidden state -> next token) for every position of a knowledge corpus
(model/fastref.py, teacher forced in chunks). At each position of held-out text from the same
domain, the k nearest stored states (L2) vote for their next tokens, p_knn = softmax(-d/temp)
summed per token, and the prediction is lambda*p_knn + (1-lambda)*p_lm (Khandelwal et al.,
2020). Reported: perplexity against lambda, and with the lookup gated on the model's own
uncertainty (only where the LM's top probability is below a threshold): the lookup rate an FPGA
lookup unit would serve, against the gain.

    python3 -m software.lab.knnlm_study models/bons/bonsai2-2b-n4-ti375.img \\
        --tokenizer models/2b-v2/tokenizer.json --corpus doc/*.md README.md --ctx 1024
"""

import argparse
import time

import numpy as np

from model.bons import BONSReader
from model.fastref import HybridRef

def chunks(ids, ctx):
    return [ids[i:i + ctx] for i in range(0, len(ids) - 1, ctx) if len(ids[i:i + ctx]) > 16]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--max-store", type=int, default=0, help="cap on datastore tokens (0: all)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--temp", type=float, default=0.0, help="kNN temperature (0: the median distance)")
    a = p.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    docs = [tk.encode(open(f, encoding="utf-8").read()).ids for f in a.corpus]
    # Held-out text: the last eval_frac of every document (same domain, unseen positions).
    store_ids, eval_ids = [], []
    for d in docs:
        cut = int(len(d)*(1 - a.eval_frac))
        store_ids += d[:cut]; eval_ids += d[cut:]
    if a.max_store:
        store_ids = store_ids[:a.max_store]
    print(f"corpus {len(a.corpus)} files: datastore {len(store_ids)} tokens, eval {len(eval_ids)}", flush=True)

    ref = HybridRef(BONSReader(a.image))
    t0 = time.time()
    keys, vals = [], []
    for c in chunks(store_ids, a.ctx):
        Hs = ref.hidden(c)
        keys.append(Hs[:-1].astype(np.float16)); vals.append(np.asarray(c[1:]))
        print(f"  datastore {sum(len(v) for v in vals)} ({time.time() - t0:.0f} s)", flush=True)
    K = np.concatenate(keys).astype(np.float32); Vt = np.concatenate(vals)
    Kn = (K*K).sum(1)

    lm_lp, knn_p, top = [], [], []
    for c in chunks(eval_ids, a.ctx):
        Hs = ref.hidden(c)
        H = Hs[:-1]
        tgt = np.asarray(c[1:])
        lg = np.concatenate([ref.logits(H[i:i + 256]) for i in range(0, len(H), 256)]).astype(np.float64)
        m = lg.max(1, keepdims=True)
        lse = (m + np.log(np.exp(lg - m).sum(1, keepdims=True)))[:, 0]
        lm_lp.append(lg[np.arange(len(tgt)), tgt] - lse)
        top.append(np.exp(lg.max(1) - lse))
        d = Kn[None, :] - 2*H @ K.T + (H*H).sum(1)[:, None]          # squared L2
        idx = np.argpartition(d, a.k, axis=1)[:, :a.k]
        dk = np.take_along_axis(d, idx, 1)
        temp = a.temp or float(np.median(dk))
        w = np.exp(-(dk - dk.min(1, keepdims=True))/temp)
        w /= w.sum(1, keepdims=True)
        knn_p.append((w*(Vt[idx] == tgt[:, None])).sum(1))
    lm_lp = np.concatenate(lm_lp); knn_p = np.concatenate(knn_p); top = np.concatenate(top)
    lm_p = np.exp(lm_lp)
    print(f"\neval {len(lm_p)} positions; LM ppl {np.exp(-lm_lp.mean()):.3f}")
    print(f"{'lambda':>7s} {'ppl':>9s} {'vs LM':>8s}")
    best = (0.0, np.exp(-lm_lp.mean()))
    for lam in (0.1, 0.2, 0.3, 0.5):
        ppl = np.exp(-np.log(lam*knn_p + (1 - lam)*lm_p + 1e-30).mean())
        best = min(best, (lam, ppl), key=lambda x: x[1])
        print(f"{lam:7.2f} {ppl:9.3f} {100*(ppl/np.exp(-lm_lp.mean()) - 1):+7.2f}%")
    lam = best[0] or 0.25
    print(f"\ngated on the LM's top probability (lambda {lam}):")
    print(f"{'top <':>7s} {'lookups':>8s} {'ppl':>9s} {'vs LM':>8s}")
    for th in (0.3, 0.5, 0.7, 0.9, 1.01):
        g = top < th
        mix = np.where(g, lam*knn_p + (1 - lam)*lm_p, lm_p)
        ppl = np.exp(-np.log(mix + 1e-30).mean())
        print(f"{th:7.2f} {100*g.mean():7.1f}% {ppl:9.3f} {100*(ppl/np.exp(-lm_lp.mean()) - 1):+7.2f}%")

if __name__ == "__main__":
    main()

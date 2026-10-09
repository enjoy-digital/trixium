#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Numerics check against the fp32 golden on the REAL model (the tiny sim gate cannot see it).

Two halves, because the GGUF lives on the workstation and the card on the board host:

  # workstation: fp32 greedy per prompt, top-5 logits at every position (~3 min a prompt)
  python3 -m software.fp32_check golden models/1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf -o build/fp32_ref.json
  # board host: at every position of fp32's path, what does the card pick, and how far below
  # fp32's best is that pick (teacher-forced, so one wrong token does not poison the rest)
  python3 -m software.fp32_check card build/fp32_ref.json --tty /dev/ttyLXU0

Int8 activations cost a logit or so of noise, so a healthy card diverges where fp32 itself
was undecided: gaps under ~1 logit, or picks outside the top-10 only where the top-10 spans
a few logits (poetry). A confident decision overturned is the signature of a numerics bug
(e.g. a 3-4 logit gap at the first token of a plain greeting where fp32 had a 2.4-logit
margin); the verdict line flags gap > 3 with margin > 2.
"""

import os
import sys
import json
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constants ----------------------------------------------------------------------------------------

PROMPTS = ["hello", "What is the capital of France? Answer in one sentence.",
           "Write a haiku about FPGAs.", "Explain what an FPGA is to a child, in two sentences."]
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Golden (workstation) -----------------------------------------------------------------------------

def cmd_golden(a):
    import numpy as np
    from tokenizers import Tokenizer
    from model.golden import load_gguf_f32, FloatModel
    tk = Tokenizer.from_file(a.tokenizer or os.path.join(os.path.dirname(a.gguf), "tokenizer.json"))
    cfg, tensors = load_gguf_f32(a.gguf)
    m = FloatModel(cfg, tensors)
    im_end = tk.token_to_id("<|im_end|>")
    ref = {}
    for text in (a.prompt or PROMPTS):
        ids = tk.encode(TEMPLATE.format(text)).ids
        m.reset()
        logits = m.forward(ids)
        out, steps = [], []
        for _ in range(a.n):
            top = np.argsort(-logits)[:10]
            steps.append([[int(i), round(float(logits[i]), 3)] for i in top])
            nxt = int(top[0])
            out.append(nxt)
            if nxt == im_end:
                break
            logits = m.forward([nxt])
        ref[text] = {"prompt": ids, "out": out, "steps": steps, "text": tk.decode(out)}
        print(f"{text!r} -> {tk.decode(out)!r}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print(a.out)

# Card (board host) --------------------------------------------------------------------------------

def cmd_card(a):
    from software.client import TriXium
    with open(a.ref, encoding="utf-8") as f:
        ref = json.load(f)
    tot = div = 0
    flagged = []
    with TriXium(tty=a.tty, tokenizer=a.tokenizer, chat_template=False, timeout=120) as b:
        for text, r in ref.items():
            lines = []
            for j in range(len(r["out"])):
                b.reset()
                b._send_ids(r["prompt"] + r["out"][:j])
                got = b.command("genb 1", timeout=120)
                toks = [int(t) for t in got.replace("GEN:", " ").split() if t.isdigit()]
                card = toks[0] if toks else -1
                top = r["steps"][j]
                top1, l1 = top[0]
                tot += 1
                if card != top1:
                    div += 1
                    margin = round(l1 - top[1][1], 2)
                    spread = round(l1 - top[-1][1], 2)          # top-1 minus top-10
                    gap = next((round(l1 - lg, 2) for tid, lg in top if tid == card), None)
                    if gap is not None and gap > 3 and margin > 2:
                        flagged.append((text, j, gap, margin))
                    lines.append(f"    pos {j:2d}: fp32 {top1:6d} card {card:6d} gap "
                                 f"{gap if gap is not None else f'>top10 (spread {spread})'}"
                                 f"  fp32 margin {margin}")
            print(f"{text!r}: {len(r['out'])} positions, {len(lines)} divergences")
            for l in lines:
                print(l)
    print(f"TOTAL {div}/{tot} divergences; confident fp32 decisions overturned (gap > 3, margin > 2): "
          f"{len(flagged)}" + ("  <- numerics bug territory" if flagged else ""))
    return 1 if flagged else 0

# Run ----------------------------------------------------------------------------------------------

def main():
    parser     = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Golden.
    p = subparsers.add_parser("golden", help="fp32 greedy reference (workstation).")
    p.add_argument("gguf",                                         help="Ternary-Bonsai GGUF.")
    p.add_argument("-o", "--out",  default="build/fp32_ref.json",  help="Reference JSON.")
    p.add_argument("--tokenizer",                                  help="tokenizer.json (default: next to the GGUF).")
    p.add_argument("-n",           type=int, default=20,           help="Tokens per prompt.")
    p.add_argument("--prompt",     action="append",                help="Prompt instead of the built-in ones (repeatable).")
    p.set_defaults(fn=cmd_golden)

    # Card.
    p = subparsers.add_parser("card", help="Teacher-forced card picks vs the reference (board host).")
    p.add_argument("ref",                                                                     help="Reference JSON.")
    p.add_argument("--tty",       default="/dev/ttyLXU0",                                     help="LiteUART tty.")
    p.add_argument("--tokenizer", default=os.path.join(REPO, "models", "1.7b", "tokenizer.json"), help="tokenizer.json.")
    p.set_defaults(fn=cmd_card)

    a = parser.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()

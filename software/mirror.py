#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Bit-exactness of the card against the Python mirror on the REAL model, one forward at a
time: both checksum the same resident intermediates (embedding, attention input x8, attention
output x8, the residual after attention and after the FFN, the FFN x8s, the argmax) at the
same points, and the first line that differs names the block to open.

  # workstation (the mirror takes ~2 min for a 13-token prompt on the 1.7B)
  python3 -m software.mirror mirror models/bons/bonsai-1p7b-n16.img --prompt "Hello" -o build/mir.txt
  # board host: the firmware's `dbg` checksums (software/fp32_check.py's paced client)
  python3 -m software.mirror card --prompt "Hello" -o card.txt
  # anywhere
  python3 -m software.mirror compare build/mir.txt card.txt

The simulation gate (./trixium_sim.py --gate) runs mirror and compare against the simulated card.

Why: the tiny sim model answers token 0 to everything, so its token gate cannot see a
numerics deviation on a real model, nor say where the card and the mirror part.
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constants ----------------------------------------------------------------------------------------

TEMPLATE     = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INT8_TAGS    = {"attn_in_x8", "attn_x8", "ffn_x8", "silu_x8"}
TRACE_DTYPES = {"i8": "<i1", "i16": "<i2", "i32": "<i4", "u32": "<u4", "f32": "<f4"}

# Checkpoints --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Checkpoint:
    exponent: int
    checksum: str
    dtype: str | None = None
    count: int | None = None
    details: str = ""

    @property
    def signature(self):
        return self.exponent, self.checksum, self.dtype, self.count


def format_typed_trace(pos, layer, tag, dtype, values, exponent=0):
    import numpy as np
    arr = np.asarray(values, dtype=TRACE_DTYPES[dtype]).reshape(-1)
    if not len(arr):
        raise ValueError("empty typed checkpoint")
    return (f"DBG pos={pos} l={layer} {tag} dtype={dtype} n={len(arr)} "
            f"e={exponent} {fnv(arr.tobytes()):08x}")


def fnv(b):
    h = 2166136261
    for x in b:
        h = ((h ^ x)*16777619) & 0xFFFFFFFF
    return h


def prompt_ids(a):
    if a.ids:
        return [int(t) for t in a.ids.split()]
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    return tk.encode(TEMPLATE.format(a.prompt) if not a.raw else a.prompt).ids

# Mirror (workstation) -----------------------------------------------------------------------------

def mirror(image, ids, out, hybrid_silu="rv32f"):
    """Run the Python mirror on `ids` and write its checkpoints to `out`."""
    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "test")) # The mirrors live with their tests.
    from test_sim_smoke import hybrid_generate
    from model.bons import BONSReader
    img = BONSReader(image)
    lines = []

    def trace(pos, key, val):
        if not isinstance(val, tuple):
            return                                  # the older float traces
        l, tag = key
        m, e = val
        arr = np.asarray(m).astype(np.int64)
        b = arr.astype("<i1" if tag in INT8_TAGS else "<i2").tobytes()
        v = list(arr[:4]) + [0]*(4 - min(4, len(arr)))
        lines.append(f"DBG pos={pos} l={l} {tag} e={e} {fnv(b):08x}  amax={int(np.abs(arr).max())} "
                     f"v0..3={v[0]},{v[1]},{v[2]},{v[3]}")
    modes = {"silu_mul": "hw", "rmsnorm": "hw", "kv_hbm": "hw"}
    h = img.header
    is_hybrid = h["version"] >= 3 and ("deltanet" in h["layer_types"] or h["hadamard_block"])
    if is_hybrid:
        from test_hybrid_mirror import hybrid27_generate
        mode = hybrid_silu
        print(f"NOTE: hybrid SiLU profile is {mode}. RV32F means GCC-10.1/newlib LUT/FMA. Other "
              "transcendentals still use NumPy; arbitrary-input bit-exactness and other "
              "target libms require qualification. See doc/simulation.md.",
              file=sys.stderr)
        def typed_trace(pos, layer, tag, dtype, values):
            lines.append(format_typed_trace(pos, layer, tag, dtype, values))
        tokens = hybrid27_generate(img, ids, 1, trace=typed_trace, silu_mode=mode)
    else:
        tokens = hybrid_generate(img, ids, 1, modes=modes, trace=trace)
        lines.append(f"DBG pos={len(ids)-1} l=-1 argmax e=0 {tokens[0]:08x}")
    with open(out, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} checksums, mirror token {tokens}")


def cmd_mirror(a):
    mirror(a.image, prompt_ids(a), a.out, a.hybrid_silu)

# Card (board host) --------------------------------------------------------------------------------

def cmd_card(a):
    from software.client import TriXium
    ids = prompt_ids(a)
    with TriXium(tty=a.tty, tokenizer=a.tokenizer, chat_template=False, timeout=600) as b:
        for c in a.pre:                          # e.g. --pre "silu off": compare a toggle's path
            b.command(c)
        b.reset()
        b._send_ids(ids)
        b.command(f"dbg {len(ids)}")
        out = b.command("genb 1", timeout=600)
        for c in a.post:
            b.command(c)
    # GEN: and emitted token IDs do not end with a newline. The first checkpoint can
    # therefore share a line with either prefix; do not silently discard it.
    lines = [l[l.index("DBG"):].strip() for l in out.splitlines() if "DBG" in l]
    with open(a.out, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    m = re.search(r"GEN:\s*(\d+)", out.replace("\n", " "))
    print(f"{a.out}: {len(lines)} checksums, card token {m.group(1) if m else '?'}")

# Compare ------------------------------------------------------------------------------------------

def parse(path):
    d     = {}
    order = []
    with open(path, encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if "DBG" not in line:
                continue                        # console chatter is not a checkpoint
            line = line[line.index("DBG"):].strip()
            if re.fullmatch(r"DBG armed for \d+ forwards", line):
                continue                        # known console command acknowledgement
            m = re.fullmatch(r"DBG pos=(-?\d+) l=(-?\d+) (\S+) "
                             r"(?:dtype=(i8|i16|i32|u32|f32) n=([1-9]\d*) )?e=(-?\d+) "
                             r"([0-9a-fA-F]{8})(?:\s+(.*))?", line)
            if m is None:
                raise ValueError(f"{path}:{lineno}: malformed DBG checkpoint")
            key = (int(m.group(1)), int(m.group(2)), m.group(3))
            if key in d:
                raise ValueError(f"{path}:{lineno}: duplicate checkpoint {key}")
            d[key] = Checkpoint(int(m.group(6)), m.group(7).lower(), m.group(4),
                                int(m.group(5)) if m.group(5) else None,
                                (m.group(8) or "").strip())
            order.append(key)
    if not order:
        raise ValueError(f"{path}: no DBG checkpoints (empty or unsupported trace)")
    return d, order


def compare(mirror_path, card_path, verbose=False):
    """Compare a card (or simulation) trace with the mirror's; 0 when every point matches."""
    try:
        mir, order = parse(mirror_path)
        card, _    = parse(card_path)
    except (OSError, ValueError) as exc:
        print(f"TRACE ERROR: {exc}")
        return 1
    same = diff = 0
    first = None
    for key in order:
        if key not in card:
            continue
        if mir[key].signature == card[key].signature:
            same += 1
        else:
            diff += 1
            if first is None:
                first = key
                pos, l, tag = key
                print(f"FIRST DIVERGENCE at pos={pos} layer={l} {tag}:\n"
                      f"  mirror {mir[key]}\n"
                      f"  card   {card[key]}")
    missing = [k for k in order if k not in card]
    extra   = [k for k in card if k not in mir]
    print(f"{same} checksums identical, {diff} different, {len(missing)} mirror points the "
          f"card did not report" + (f" (e.g. {missing[0]})" if missing else ""))
    if extra:
        print(f"{len(extra)} unexpected card points (e.g. {extra[0]})")
    if diff and verbose:
        for key in order:
            if key in card and mir[key].signature != card[key].signature:
                print(f"  pos={key[0]} l={key[1]} {key[2]}: mirror {mir[key]} card {card[key]}")
    return 1 if diff or missing or extra else 0


def cmd_compare(a):
    return compare(a.mirror, a.card, a.verbose)

# Run ----------------------------------------------------------------------------------------------

def main():
    parser     = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Mirror / Card.
    for name, fn in (("mirror", cmd_mirror), ("card", cmd_card)):
        p = subparsers.add_parser(name)
        if name == "mirror":
            p.add_argument("image",                                                      help="BONS image.")
            p.add_argument("--hybrid-silu", choices=("rv32f", "mathematical"), default="rv32f", help="Hybrid SiLU profile; rv32f is the qualified VCU baseline libm.")
        else:
            p.add_argument("--tty",  default="/dev/ttyLXU0",           help="LiteUART tty.")
            p.add_argument("--pre",  action="append", default=[],      help="Console command to send first (repeatable).")
            p.add_argument("--post", action="append", default=[],      help="Console command to send after (repeatable).")
        p.add_argument("--prompt",    default="Hello",                                               help="Prompt text.")
        p.add_argument("--ids",                                                                      help="Token ids instead of --prompt.")
        p.add_argument("--raw",       action="store_true",                                           help="No chat template.")
        p.add_argument("--tokenizer", default=os.path.join(REPO, "models", "1.7b", "tokenizer.json"), help="tokenizer.json.")
        p.add_argument("-o", "--out", required=True,                                                 help="Output trace.")
        p.set_defaults(fn=fn)

    # Compare.
    p = subparsers.add_parser("compare")
    p.add_argument("mirror",          help="Mirror trace.")
    p.add_argument("card",            help="Card trace.")
    p.add_argument("-v", "--verbose", action="store_true", help="List every divergence.")
    p.set_defaults(fn=cmd_compare)

    a = parser.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()

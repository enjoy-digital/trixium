#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""How compressible is the weight stream? Trit statistics of a BONS image.

For every ternary tensor (sampled rows of every channel): the {-1, 0, +1} histogram, the order-0
entropy, the entropy conditioned on the previous trit of the row, and the entropy given the
trit at the same column of the previous row. Compares them with what the stream costs today
(trit5: 26 bytes per 128 trits = 1.625 bit/trit, 2-bit codes = 2 bit/trit, plus the fp16 group
scales). An entropy coder on the weight stream only pays if these sit well below 1.625.

    python3 -m software.lab.trit_entropy models/bons/bonsai2-2b-n4-ti375.img [--rows 64]
"""

import argparse
import re

import numpy as np

from model import layout
from model.bons import BONSReader, decode_rows

def entropy(counts):
    p = counts[counts > 0]/counts.sum()
    return float(-(p*np.log2(p)).sum())

def cond_entropy(joint):
    """H(Y|X) from a joint count matrix joint[x, y]."""
    h = 0.0
    n = joint.sum()
    for row in joint:
        if row.sum():
            h += row.sum()/n*entropy(row)
    return h

def tensor_stats(r, e, rows):
    """Sampled codes of one entry across its channels -> (hist[3], joint_prev_col, joint_prev_row)."""
    rb = layout.row_beats(e["cols"], e["kind"])*32
    hist = np.zeros(3, np.int64)
    jc = np.zeros((3, 3), np.int64)
    jr = np.zeros((3, 3), np.int64)
    chans = [0] if e["flags"] & layout.FLAG_REPLICATED else range(r.header["n_channels"])
    for c in chans:
        b = r.entry_channel_bytes(e, c)
        n = len(b)//rb
        if n == 0:
            continue
        take = min(rows, n)
        # One contiguous block (keeps the previous-row context real) from the middle.
        r0 = (n - take)//2
        _, v = decode_rows(e, np.asarray(b[r0*rb:(r0 + take)*rb]))
        v = v.astype(np.int64)
        hist += np.bincount(v.reshape(-1), minlength=3)[:3]
        np.add.at(jc, (v[:, :-1].reshape(-1), v[:, 1:].reshape(-1)), 1)
        if take > 1:
            np.add.at(jr, (v[:-1].reshape(-1), v[1:].reshape(-1)), 1)
    return hist, jc, jr

def group_of(name):
    """Tensor family: the layer index stripped (blk.12.ffn_up -> ffn_up)."""
    return re.sub(r"^blk\.\d+\.", "", name).replace(".weight", "")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--rows", type=int, default=64, help="rows sampled per tensor and channel")
    a = p.parse_args()

    r = BONSReader(a.image)
    tern = [e for e in r.entries if e["kind"] in (layout.KIND_TERNARY, layout.KIND_TERNARY5)]
    fam = {}
    tot_h = np.zeros(3, np.int64); tot_c = np.zeros((3, 3), np.int64); tot_r = np.zeros((3, 3), np.int64)
    trits = 0
    for e in tern:
        h, jc, jr = tensor_stats(r, e, a.rows)
        f = fam.setdefault(group_of(e["name"]), [np.zeros(3, np.int64), np.zeros((3, 3), np.int64),
                                                 np.zeros((3, 3), np.int64), 0])
        f[0] += h; f[1] += jc; f[2] += jr
        f[3] += e["rows"]*e["cols"] if "rows" in e else 0
        tot_h += h; tot_c += jc; tot_r += jr
        trits += e.get("rows", 0)*e["cols"]

    kinds = sorted({layout.KIND_NAMES.get(e["kind"], e["kind"]) for e in tern})
    print(f"{a.image}: {len(tern)} ternary tensors ({', '.join(map(str, kinds))}), "
          f"{a.rows} rows sampled per tensor and channel")
    print(f"{'family':24s} {'-1':>6s} {'0':>6s} {'+1':>6s} {'H0':>6s} {'H|col':>6s} {'H|row':>6s}")
    for name in sorted(fam):
        h, jc, jr, _ = fam[name]
        q = h/h.sum()
        print(f"{name:24s} {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {entropy(h):6.3f} "
              f"{cond_entropy(jc):6.3f} {cond_entropy(jr):6.3f}")
    q = tot_h/tot_h.sum()
    h0 = entropy(tot_h)
    print(f"{'ALL':24s} {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {h0:6.3f} "
          f"{cond_entropy(tot_c):6.3f} {cond_entropy(tot_r):6.3f}")
    print(f"stored today: trit5 1.625 bit/trit (+ fp16 scale per 128 = 0.125): 1.750; "
          f"order-0 entropy + scales: {h0 + 0.125:.3f} -> {100*(1 - (h0 + 0.125)/1.75):.1f}% fewer bytes")

if __name__ == "__main__":
    main()

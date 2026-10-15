#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Where a decode token leaves the DRAM idle: the firmware's token trace (`tprof arm N`, then a
generation or `bench`, then `tprof dump`) broken down by firmware op and sequencer list.

Each TP line is <tag> <timer> <engine-busy cycles> <vector-busy cycles> [<both-busy cycles>]
(firmware/perf.c), the tag's top nibble its kind: 1 op enter, 2 op exit, 3 layer, 4 sequencer
descriptors done, 5 sequencer list start (index, count), 6 begin, 7 end. Between two events the
engine was idle for (timer delta - engine delta) cycles: that is P, the time the weight streams
wait; with the both-busy counter, neither unit busy for (timer - engine - vector + both): the
CPU's own time and the sequencer's dispatch gaps. Each interval is charged to the innermost open
op and to the last list started.

    python3 -m software.lab.tprof console.log
"""

import re
import argparse

OPS = ("embed", "rmsnorm", "qknorm", "rope", "kvquant", "scores", "softmax", "wv", "quant8",
       "norm_i32", "silu_mul", "resid_add", "lmhead", "engine_wait")

M32 = 1 << 32

def parse(text):
    """TP lines -> [(tag, t, eng, vec, both)] (the last dump in the text; both None when the
    firmware did not record it)."""
    events = []
    for line in text.splitlines():
        if line.startswith("TPROF ") and "events" in line:
            events = []
        m = re.match(r"^TP ([0-9a-f]{8}) (\d+) (\d+) (\d+)(?: (\d+))?", line.strip())
        if m:
            both = int(m.group(5)) if m.group(5) is not None else None
            events.append((int(m.group(1), 16), int(m.group(2)), int(m.group(3)),
                           int(m.group(4)), both))
    return events

def analyze(events):
    """-> dict: totals, per-op and per-list [cycles, engine idle, vector busy, neither busy],
    the gaps (neither is 0 without the both-busy counter)."""
    stack, layer, lst = [], -1, "-"
    by_op, by_list, gaps = {}, {}, []
    tot = [0, 0, 0, 0]
    for (tag, t, e, v, b), (tag2, t2, e2, v2, b2) in zip(events, events[1:]):
        kind = tag >> 28
        if kind == 1:
            stack.append((tag >> 8) & 0xFF)
        elif kind == 2:
            op = (tag >> 8) & 0xFF
            if op in stack:
                del stack[len(stack) - 1 - stack[::-1].index(op)]
        elif kind == 3:
            layer = (tag >> 8) & 0xFFFFF
        elif kind == 5:
            lst = f"list@{(tag >> 8) & 0xFF}+{tag & 0xFF}"
        dt   = (t2 - t) % M32
        idle = max(dt - (e2 - e) % M32, 0)
        vb   = min((v2 - v) % M32, dt)
        nb   = 0
        if b is not None and b2 is not None:
            nb = max(dt - (e2 - e) % M32 - (v2 - v) % M32 + (b2 - b) % M32, 0)
        op   = OPS[stack[-1]] if stack and stack[-1] < len(OPS) else "-"
        for d, k in ((by_op, op), (by_list, lst)):
            acc = d.setdefault(k, [0, 0, 0, 0])
            acc[0] += dt; acc[1] += idle; acc[2] += vb; acc[3] += nb
        tot[0] += dt; tot[1] += idle; tot[2] += vb; tot[3] += nb
        gaps.append((idle, dt, layer, op, lst, tag, tag2))
    gaps.sort(reverse=True)
    return dict(total=tot, by_op=by_op, by_list=by_list, gaps=gaps)

def report(res, top=25, out=print):
    tot = res["total"]
    if not tot[0]:
        out("tprof: no trace")
        return
    out(f"token: {tot[0]} cycles, engine idle {tot[1]} ({100*tot[1]/tot[0]:.1f}%), "
        f"vector busy {tot[2]}, neither busy {tot[3]} ({100*tot[3]/tot[0]:.1f}%)")
    for name, d in (("op", res["by_op"]), ("list", res["by_list"])):
        out(f"{name:14s} {'cycles':>10s} {'eng idle':>10s} {'% token':>8s} {'vec busy':>10s} {'neither':>10s}")
        for k, (c, i, v, n) in sorted(d.items(), key=lambda kv: -kv[1][1]):
            out(f"{k:14s} {c:10d} {i:10d} {100*i/tot[0]:7.1f}% {v:10d} {n:10d}")
    out(f"largest engine-idle intervals (layer, op, list, tags):")
    for idle, dt, layer, op, lst, a, b in res["gaps"][:top]:
        out(f"  {idle:8d} of {dt:8d}  L{layer:<3d} {op:12s} {lst:12s} {a:08x} -> {b:08x}")

def main():
    parser = argparse.ArgumentParser(description="Engine-idle breakdown of a tprof dump.")
    parser.add_argument("log", help="Console log holding a `tprof dump`.")
    parser.add_argument("--top", default=25, type=int, help="Largest idle intervals listed.")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8", errors="replace") as f:
        report(analyze(parse(f.read())), top=args.top)

if __name__ == "__main__":
    main()

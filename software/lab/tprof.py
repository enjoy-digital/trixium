#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Where a decode token leaves the DRAM idle: the firmware's token trace (`tprof arm N`, then a
generation or `bench`, then `tprof dump`) broken down by firmware op and sequencer list.

Each TP line is <tag> <timer> <engine-busy cycles> <units-busy cycles> [<both-busy cycles>]
(firmware/perf.c; "units" is the scratchpad arbiter's aggregate client - every scratchpad unit,
the vector unit included - as gateware/core/core.py hands it to the engine's counters), the tag's
top nibble its kind: 1 op enter, 2 op exit, 3 layer, 4 sequencer
descriptors done, 5 sequencer list start (index, count; bits 17-20 its listprof slot, named by
the dump's `TPL <slot> <name>` lines, bit 16 set when the start waited for the previous list),
6 begin, 7 end. Between two events the
engine was idle for (timer delta - engine delta) cycles: that is P, the time the weight streams
wait; with the both-busy counter, neither busy for (timer - engine - units + both): the CPU's
own time and the sequencer's dispatch gaps. Each interval is charged to the innermost open
op and to the last list started.

    python3 -m software.lab.tprof console.log
"""

import re
import argparse

OPS = ("embed", "rmsnorm", "qknorm", "rope", "kvquant", "scores", "softmax", "wv", "quant8",
       "norm_i32", "silu_mul", "resid_add", "lmhead",
       "dn_proj", "dn_act", "dn_conv", "dn_prep", "dn_heads", "dn_gate", "dn_out", "attn",
       "ffn_proj", "ffn_down", "engine_wait")   # firmware/perf.h's enum perf_op

M32 = 1 << 32

def parse_names(text):
    """TPL lines -> {slot: list name}."""
    names = {}
    for m in re.finditer(r"^TPL (\d+) (.+?)\s*$", text, re.M):
        names[int(m.group(1))] = m.group(2)
    return names

def list_name(tag, names):
    """A list start event's label: its listprof name when the dump had one."""
    slot = (tag >> 17) & 0xF
    if slot in names:
        return names[slot]
    return f"list@{(tag >> 8) & 0xFF}+{tag & 0xFF}"

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

def analyze(events, names=None):
    """-> dict: totals, per-op and per-list [cycles, engine idle, units busy, neither busy],
    the gaps (neither is 0 without the both-busy counter), and per list the interval before
    each of its starts [starts, starts that waited, cycles, engine idle, neither busy]: the
    build (and the previous list's post) on the critical path when the start did not wait."""
    names = names or {}
    stack, layer, lst = [], -1, "-"
    by_op, by_list, gaps, pre = {}, {}, [], {}
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
            lst = list_name(tag, names)
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
        if tag2 >> 28 == 5:
            p = pre.setdefault(list_name(tag2, names), [0, 0, 0, 0, 0])
            p[0] += 1; p[1] += (tag2 >> 16) & 1; p[2] += dt; p[3] += idle; p[4] += nb
    gaps.sort(reverse=True)
    return dict(total=tot, by_op=by_op, by_list=by_list, gaps=gaps, pre=pre)

def report(res, top=25, out=print):
    tot = res["total"]
    if not tot[0]:
        out("tprof: no trace")
        return
    out(f"token: {tot[0]} cycles, engine idle {tot[1]} ({100*tot[1]/tot[0]:.1f}%), "
        f"units busy {tot[2]}, neither busy {tot[3]} ({100*tot[3]/tot[0]:.1f}%)")
    for name, d in (("op", res["by_op"]), ("list", res["by_list"])):
        out(f"{name:14s} {'cycles':>10s} {'eng idle':>10s} {'% token':>8s} {'units busy':>10s} {'neither':>10s}")
        for k, (c, i, v, n) in sorted(d.items(), key=lambda kv: -kv[1][1]):
            out(f"{k:14s} {c:10d} {i:10d} {100*i/tot[0]:7.1f}% {v:10d} {n:10d}")
    if res.get("pre"):
        out(f"{'before list start':22s} {'starts':>6s} {'waited':>6s} {'cycles':>10s} {'eng idle':>10s} "
            f"{'neither':>10s}  (the interval from the previous event to each start)")
        for k, (n, w, c, i, nb) in sorted(res["pre"].items(), key=lambda kv: -kv[1][4]):
            out(f"{k:22s} {n:6d} {w:6d} {c:10d} {i:10d} {nb:10d}")
    out(f"largest engine-idle intervals (layer, op, list, tags):")
    for idle, dt, layer, op, lst, a, b in res["gaps"][:top]:
        out(f"  {idle:8d} of {dt:8d}  L{layer:<3d} {op:12s} {lst:12s} {a:08x} -> {b:08x}")

def main():
    parser = argparse.ArgumentParser(description="Engine-idle breakdown of a tprof dump.")
    parser.add_argument("log", help="Console log holding a `tprof dump`.")
    parser.add_argument("--top", default=25, type=int, help="Largest idle intervals listed.")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    report(analyze(parse(text), parse_names(text)), top=args.top)

if __name__ == "__main__":
    main()

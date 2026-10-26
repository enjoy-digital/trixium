#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The video's glow on the diagrams (doc/illustration/trixium.html, GLOW).

Adds, to each diagram SVG of this directory:
- a soft halo of its colour behind every teal, blue and gold block outline (the video's
  coloured halos behind boxes),
- a light bloom on the data arrows and on the titles (the text itself stays crisp on top).

Idempotent: the previous glow (elements of class "glow", the marked defs and style lines) is
removed first, so it can be re-run after editing a diagram. GLOW scales everything, as in the
video (0.4 there and here).

    python3 doc/images/glow.py [file.svg ...]      # default: every diagram here
    python3 doc/images/glow.py --glow 1.0          # twice as strong
"""

import os
import re
import sys
import glob

GLOW   = 0.4
COLORS = ("#35c2ae", "#86a3ec", "#d8b25e") # TriXium teal, LiteX blue, memory gold.

MARK = "glow:"

def strip(svg):
    """The SVG without a previous glow pass."""
    svg = re.sub(r"\n[ \t]*<!-- glow:begin -->.*?<!-- glow:end -->", "", svg, flags=re.S)
    svg = "\n".join(l for l in svg.split("\n") if 'class="glow"' not in l and "/* glow */" not in l)
    return svg

def glow(svg, GLOW=GLOW):
    svg  = strip(svg)
    name = re.search(r'<g id="([^"]+)">', svg).group(1)
    # Filter regions in user space over the whole canvas: a percentage of the bounding box
    # clips a straight (zero-height) arrow to nothing.
    vb   = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    reg  = f'filterUnits="userSpaceOnUse" x="{vb[0]}" y="{vb[1]}" width="{vb[2]}" height="{vb[3]}"'
    halo_sd, bloom_sd = 8*GLOW + 2, 4*GLOW + 1
    defs = f"""
    <!-- glow:begin -->
    <filter id="{name}-halo" {reg}><feGaussianBlur stdDeviation="{halo_sd:g}"/></filter>
    <filter id="{name}-bloom" {reg}><feGaussianBlur in="SourceGraphic" stdDeviation="{bloom_sd:g}" result="b"/><feComponentTransfer in="b" result="b"><feFuncA type="linear" slope="{0.5 + GLOW:g}"/></feComponentTransfer><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="{name}-text" {reg}><feGaussianBlur in="SourceGraphic" stdDeviation="{2 + 2*GLOW:g}" result="b"/><feComponentTransfer in="b" result="b"><feFuncA type="linear" slope="{0.2 + 0.5*GLOW:g}"/></feComponentTransfer><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <!-- glow:end -->"""
    svg = svg.replace("<defs>", "<defs>" + defs, 1)
    css = (f"      #{name} .data {{ filter: url(#{name}-bloom); }} /* glow */\n"
           f"      #{name} .title, #{name} .core-title, #{name} .data-label {{ filter: url(#{name}-text); }} /* glow */\n")
    svg = svg.replace("    </style>", css + "    </style>", 1)
    out = []
    for line in svg.split("\n"):
        m = re.match(r'(\s*)<polygon points="([^"]+)" fill="[^"]*" stroke="(#[0-9a-fA-F]{6})"', line)
        if m and m.group(3).lower() in COLORS:
            indent, pts, col = m.groups()
            out.append(f'{indent}<polygon class="glow" points="{pts}" fill="none" stroke="{col}" '
                       f'stroke-width="{4 + 6*GLOW:g}" opacity="{0.25 + 0.5*GLOW:g}" '
                       f'filter="url(#{name}-halo)"/>')
        out.append(line)
    return "\n".join(out)

def main():
    here  = os.path.dirname(os.path.abspath(__file__))
    args  = sys.argv[1:]
    level = GLOW
    if args[:1] == ["--glow"]:
        level, args = float(args[1]), args[2:]
    files = args or sorted(glob.glob(os.path.join(here, "*.svg")))
    for path in files:
        with open(path, encoding="utf-8") as f:
            svg = f.read()
        if "<!-- glow:skip -->" in svg:   # The classic light-style diagrams (board-ti375.svg).
            print(f"{os.path.basename(path)}: skipped")
            continue
        new = glow(svg, level)
        if new != svg:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
        halos = new.count('class="glow"')
        print(f"{os.path.basename(path)}: {halos} halos")

if __name__ == "__main__":
    main()

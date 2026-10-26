#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Geometry lint for the generated diagram SVGs (board-ti375.py): every <text> is measured with
Carlito (the metric twin of Calibri, the diagrams' first font) and checked against the box
outlines (polygons with a stroke), the arrows (paths) and the other texts. Crossings and near
misses (closer than MARGIN px) are listed; a clean diagram prints "0 issue(s)".

    python3 doc/images/svglint.py doc/images/board-ti375.svg doc/images/board-ti375-simple.svg [-v]

-v also lists the widest texts, to size boxes. Needs Pillow and fonts-crosextra-carlito.
"""
import re, sys, xml.etree.ElementTree as ET
from PIL import ImageFont

FONTS = {(None, None): "Carlito-Regular", ("bold", None): "Carlito-Bold",
         (None, "italic"): "Carlito-Italic", ("bold", "italic"): "Carlito-BoldItalic"}
FDIR = "/usr/share/fonts/truetype/crosextra/"
SVG = "{http://www.w3.org/2000/svg}"
MARGIN = 4.0
cache = {}

def font(w, s, size):
    k = (w, s, size)
    if k not in cache:
        cache[k] = ImageFont.truetype(FDIR + FONTS[(w, s)] + ".ttf", int(round(size)))
    return cache[k]

def text_bbox(el):
    """(x0, y0, x1, y1, text): the ink box of a <text>, in canvas coordinates."""
    x, y = float(el.get("x")), float(el.get("y"))
    size = float(el.get("font-size", 13))
    anchor = el.get("text-anchor", "start")
    w, s = el.get("font-weight"), el.get("font-style")
    t = el.text or ""
    f = font(w, s, size)
    adv = f.getlength(t)
    asc, desc = f.getmetrics()
    ib = f.getbbox(t)
    top, bot = ib[1] - asc, ib[3] - asc        # ink, relative to the baseline
    if anchor == "middle": x0 = x - adv/2
    elif anchor == "end": x0 = x - adv
    else: x0 = x
    if "rotate(-90" in el.get("transform", ""):
        # The text runs upward from (x, y), its ascent to the left.
        return (x + top, y - (x0 + adv - x), x + bot, y - (x0 - x), t)
    return (x0, y + top, x0 + adv, y + bot, t)

def seg_rect(x1, y1, x2, y2, r):
    """Does the segment meet the rectangle (Liang-Barsky)?"""
    rx0, ry0, rx1, ry1 = r
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - rx0), (dx, rx1 - x1), (-dy, y1 - ry0), (dy, ry1 - y1)):
        if p == 0:
            if q < 0: return False
        else:
            t = q / p
            if p < 0:
                if t > t1: return False
                if t > t0: t0 = t
            else:
                if t < t0: return False
                if t < t1: t1 = t
    return True

def inflate(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)

def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

def lint(path, verbose=False):
    root = ET.parse(path).getroot()
    texts = [text_bbox(e) for e in root.iter(SVG + "text")]
    polys = []
    for e in root.iter(SVG + "polygon"):
        if e.get("stroke") is None:   # arrow heads belong to the arrows
            continue
        polys.append([tuple(map(float, p.split(","))) for p in e.get("points").split()])
    paths = [[tuple(map(float, p.split(","))) for p in re.findall(r"[-\d.]+,[-\d.]+", e.get("d"))]
             for e in root.iter(SVG + "path")]
    issues = 0
    for tb in texts:
        r = tb[:4]
        rm = inflate(r, MARGIN)
        where = f"text '{tb[4]}' at {r[0]:.0f},{r[1]:.0f}-{r[2]:.0f},{r[3]:.0f}"
        for pts in polys:
            for a, b in zip(pts, pts[1:] + pts[:1]):
                if seg_rect(a[0], a[1], b[0], b[1], rm):
                    hard = seg_rect(a[0], a[1], b[0], b[1], r)
                    print(f"{'CROSS' if hard else 'near '} box edge ({a[0]:.0f},{a[1]:.0f})-({b[0]:.0f},{b[1]:.0f}): {where}")
                    issues += 1
        for pts in paths:
            for a, b in zip(pts, pts[1:]):
                if seg_rect(a[0], a[1], b[0], b[1], rm):
                    hard = seg_rect(a[0], a[1], b[0], b[1], r)
                    print(f"{'CROSS' if hard else 'near '} arrow ({a[0]:.0f},{a[1]:.0f})-({b[0]:.0f},{b[1]:.0f}): {where}")
                    issues += 1
    for i, a in enumerate(texts):
        for b in texts[i+1:]:
            if overlap(inflate(a[:4], 1.5), b[:4]):
                print(f"OVERLAP text '{a[4]}' / '{b[4]}'")
                issues += 1
    print(f"{path}: {len(texts)} texts, {len(polys)} boxes, {len(paths)} arrows, {issues} issue(s)")
    if verbose:
        for tb in sorted(texts, key=lambda t: t[2]-t[0], reverse=True)[:12]:
            print(f"  {tb[2]-tb[0]:6.1f} px  '{tb[4]}'")
    return issues

if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    sys.exit(1 if sum(lint(p, "-v" in sys.argv) for p in files) else 0)

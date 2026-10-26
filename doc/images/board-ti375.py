#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""board-ti375.svg: TriXium standalone on the Efinix Ti375 C529 kit, in the classic LiteX diagram
style (white canvas, chamfered boxes, grey arrows, the EnjoyDigital and "Built with LiteX" logos
embedded from this directory). Two variants: `full` (every block and path) and `simple` (the
standalone story alone: fewer blocks, one or two lines per box).

    python3 doc/images/board-ti375.py [--variant full|simple|all]   # -> board-ti375[-simple].svg
    inkscape board-ti375.svg --export-type=png --export-filename=board-ti375.png --export-width=2800

The SVGs stay editable by hand (Inkscape or XML): the generator only keeps the geometry consistent.
Layout rules: arrows leave and enter boxes at their centre lines, boxes of one column share their
edges, every label has room (Carlito/Calibri metrics) between the things it sits between.
"""
import argparse, base64, os

FONT = "Calibri, Carlito, 'Liberation Sans', Arial, Helvetica, sans-serif"
STROKE, TEXT, ARROW, THICK = "#7f7f7f", "#1a1a1a", "#595959", "#8c8c8c"
MEM, HARD, GROUP, FPGA, BOARD = "#c9c9c9", "#dcdcdc", "#f3f3f3", "#e4e4e4", "#f8f8f8"
KEY = "#cdcdcd"   # The blocks that make the kit standalone: eMMC path and web server.
out = []
boxes = []        # Every box drawn: (title, x1, y1, x2, y2), for --boxes (the video's highlights).

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def box(x1, y1, x2, y2, fill="#ffffff", c=14, lines=(), shadow=True, dash=False):
    pts = f"{x1},{y1} {x2-c},{y1} {x2},{y1+c} {x2},{y2} {x1+c},{y2} {x1},{y2-c}"
    extra = ' stroke-dasharray="7,4"' if dash else ""
    filt = ' filter="url(#shadow)"' if shadow else ""
    out.append(f'  <polygon points="{pts}" fill="{fill}" stroke="{STROKE}" stroke-width="1.5"{extra}{filt}/>')
    title = (lines[0][0] if isinstance(lines[0], tuple) else lines[0]) if lines else ""
    boxes.append((title, x1, y1, x2, y2))
    if lines:
        text_lines((x1 + x2)/2, (y1 + y2)/2, lines)

def text_lines(cx, cy, lines, anchor="middle"):
    # lines: (text, size, weight, style); weight/style optional. A vertically centred block.
    norm = []
    for l in lines:
        l = (l + (None,)*4)[:4] if isinstance(l, tuple) else (l, 13, None, None)
        norm.append((l[0], l[1] or 13, l[2], l[3]))
    heights = [s*1.3 for _, s, _, _ in norm]
    y = cy - sum(heights)/2
    for (t, s, w, st), h in zip(norm, heights):
        text(cx, y + h*0.5 + s*0.35, t, s, anchor=anchor, weight=w, style=st)
        y += h

def text(x, y, t, size=13, anchor="middle", weight=None, style=None, fill=TEXT, rotate=None):
    attrs = f'x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"'
    if weight: attrs += f' font-weight="{weight}"'
    if style:  attrs += f' font-style="{style}"'
    if rotate: attrs += f' transform="rotate({rotate} {x} {y})"'
    out.append(f'  <text {attrs}>{esc(t)}</text>')

def head(x, y, dx, dy, col, size=9):
    # A filled triangle with its tip at (x, y), pointing along (dx, dy).
    px, py = -dy, dx
    bx, by = x - dx*size, y - dy*size
    hw = size*0.42
    out.append(f'  <polygon points="{x:.1f},{y:.1f} {bx+px*hw:.1f},{by+py*hw:.1f} {bx-px*hw:.1f},{by-py*hw:.1f}" fill="{col}"/>')

def arrow(pts, double=False, thick=False):
    col  = THICK if thick else ARROW
    wdt  = 4 if thick else 1.6
    size = 11 if thick else 9
    p = [list(q) for q in pts]
    def unit(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        n = (dx*dx + dy*dy) ** 0.5
        return dx/n, dy/n
    heads = []
    dx, dy = unit(p[-2], p[-1]); heads.append((p[-1][0], p[-1][1], dx, dy))
    p[-1] = [p[-1][0] - dx*(size-2), p[-1][1] - dy*(size-2)]
    if double:
        dx, dy = unit(p[1], p[0]); heads.append((p[0][0], p[0][1], dx, dy))
        p[0] = [p[0][0] - dx*(size-2), p[0][1] - dy*(size-2)]
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in p)
    out.append(f'  <path d="{d}" fill="none" stroke="{col}" stroke-width="{wdt}" stroke-linejoin="round"/>')
    for x, y, dx, dy in heads:
        head(x, y, dx, dy, col, size)

def image(path, x, y, w, h):
    data = base64.b64encode(open(path, "rb").read()).decode()
    out.append(f'  <image x="{x}" y="{y}" width="{w}" height="{h}" xlink:href="data:image/png;base64,{data}"/>')

def line(x1, y1, x2, y2, col=STROKE, w=1):
    out.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}"/>')

def begin(W, H, name, desc):
    out.clear()
    boxes.clear()
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{name}-title {name}-description">')
    out.append(f'  <title id="{name}-title">TriXium on the Efinix Ti375 C529: a standalone local LLM</title>')
    out.append(f'  <desc id="{name}-description">{esc(desc)}</desc>')
    out.append('  <!-- glow:skip -->')
    out.append('  <!-- Editable SVG, classic LiteX diagram style: chamfered boxes, grey arrows, light fills. Generated by doc/images/board-ti375.py; edit either. -->')
    out.append('  <defs>')
    out.append('    <filter id="shadow" x="-10%" y="-10%" width="125%" height="130%"><feGaussianBlur in="SourceAlpha" stdDeviation="2.4" result="b"/><feOffset in="b" dx="3" dy="3" result="o"/><feComponentTransfer in="o" result="s"><feFuncA type="linear" slope="0.32"/></feComponentTransfer><feMerge><feMergeNode in="s"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    out.append('  </defs>')
    out.append(f'  <g id="{name}" font-family="{FONT}">')
    out.append(f'  <rect width="{W}" height="{H}" fill="#ffffff"/>')

def finish(path):
    out.append('  </g>')
    out.append('</svg>')
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(path, len("\n".join(out)))

DESC = ("The Ti375 kit runs the 2B model with no host: bitstream and firmware boot from the QSPI flash, "
        "the model and its tokenizer load from the eMMC, LiteEth carries a hardware UDP/IP stack (Etherbone, "
        "weight upload) and the CPU's own MAC for the firmware's TCP/IP, DHCP, mDNS and HTTP chat server. "
        "The TriXium core streams ternary weights from 1 GB of LPDDR4x through the hard controller's two "
        "512-bit AXI targets. A browser on the LAN chats at http://trixium.local/.")

# Full variant -------------------------------------------------------------------------------------
#
# Columns: left of the FPGA x 262..392 | peripherals 440..615 | web server + CPU 650..830 (centre 740)
# | TriXium core group 905..1140 (inner boxes 914..1131) | off-FPGA memories 1185..1370.
# Rows: Ethernet y 130 | UART/USB 262 | flash/LiteSPI 370 | memory system 525..648.

def draw_full(logos):
    begin(1400, 880, "board-ti375", DESC)

    # Board and FPGA boundaries.
    box(250, 25, 1388, 742, fill=BOARD, c=34)
    box(415, 55, 1155, 665, fill=FPGA, c=30)
    text(1372, 730, "Efinix Titanium Ti375 C529 development kit · 3-5 W", 17, anchor="end", style="italic")
    text(440, 618, "Ti375 FPGA", 24, anchor="start", weight="bold", style="italic")
    text(440, 644, "LiteX SoC · 100 MHz", 15, anchor="start", style="italic")

    # Outside the board: the user, and the optional workstation.
    box(10, 95, 172, 200, lines=[("Laptop · phone", 18, "bold"), ("any browser", 14), ("http://trixium.local/", 14)])
    arrow([(172, 130), (262, 130)], double=True)
    text(211, 120, "Ethernet", 13, style="italic")
    text(211, 149, "HTTP chat", 13, style="italic")
    box(10, 290, 172, 395, dash=True, lines=[("Workstation", 18, "bold"), ("optional", 13, None, "italic"),
                                              ("JTAG · console", 14), ("deploy · new models", 14)])
    arrow([(172, 332), (217, 332), (217, 262), (262, 262)], double=True)

    # On the board, left of the FPGA.
    box(262, 95, 392, 165, lines=[("RGMII PHY", 17, "bold"), ("RJ45 · 1 Gb/s", 13)])
    box(262, 235, 392, 289, lines=[("USB FT4232H", 16, "bold"), ("JTAG · serial", 13)])
    box(262, 330, 392, 410, fill=MEM, lines=[("QSPI flash", 17, "bold"), ("64 MB", 13), ("bitstream + firmware", 13)])
    arrow([(392, 130), (440, 130)], double=True)
    arrow([(392, 262), (440, 262)], double=True)
    arrow([(392, 370), (440, 370)])

    # LiteX SoC: peripherals (column 1).
    box(440, 95, 615, 205, fill=KEY, lines=[("LiteEth", 18, "bold"), ("RGMII MAC", 13), ("hardware UDP/IP:", 13),
                                             ("Etherbone · upload", 13), ("CPU MAC → firmware", 13)])
    box(440, 240, 560, 284, lines=[("UART", 17, "bold"), ("console", 13)])
    box(440, 330, 615, 410, lines=[("LiteSPI", 18, "bold"), ("QSPI at 25 MHz", 13), ("BIOS flash boot", 13)])

    # Web chat server and CPU (column 2).
    box(650, 95, 830, 205, fill=KEY, lines=[("Web chat server", 17, "bold"), ("firmware TCP/IP · HTTP", 13),
                                              ("GET /  ·  POST /chat", 13), ("DHCP · mDNS", 13)])
    box(650, 235, 830, 415)
    text_lines(740, 282, [("VexiiRiscv", 18, "bold"), ("RV32 + FPU", 13), ("32 KB I$ / D$", 13), ("BIOS ROM · 16 KB SRAM", 13)])
    line(663, 328, 817, 328)
    text_lines(740, 371, [("Firmware", 14, "bold"), ("flash boot · eMMC model", 13), ("tokenizer · chat", 13),
                          ("inference lists · sampling", 13)])
    arrow([(740, 205), (740, 235)], double=True)
    # Peripherals <-> CPU over the SoC bus.
    arrow([(615, 150), (650, 150)], double=True)
    arrow([(560, 262), (650, 262)], double=True)
    arrow([(615, 370), (650, 370)])
    text(632, 316, "Wishbone / CSRs", 13, style="italic", rotate=-90)
    # CPU main RAM through the shared port.
    arrow([(740, 415), (740, 525)], double=True)
    text(732, 464, "main RAM", 13, anchor="end", style="italic")
    text(732, 482, "512-bit AXI", 13, anchor="end", style="italic")

    # TriXium core (column 3).
    box(1010, 92, 1140, 142, fill=KEY, lines=[("SD/MMC core", 17, "bold"), ("8-bit at 50 MHz", 13)])
    box(905, 160, 1140, 500, fill=GROUP, c=18)
    text(1010, 184, "TriXium core", 18, weight="bold")
    box(914, 195, 1131, 241, lines=[("Loader", 16, "bold"), ("eMMC / UDP → memory windows", 13)])
    box(914, 257, 1131, 301, lines=[("Sequencer", 16, "bold"), ("command lists", 13)])
    box(914, 317, 1131, 361, lines=[("Matvec engine", 16, "bold"), ("4 channels · 5-trit weights", 13)])
    box(914, 377, 1131, 439, lines=[("Units", 16, "bold"), ("Hadamard · norm · SiLU · softmax", 13),
                                     ("DeltaNet + attention streams", 13)])
    box(914, 455, 1131, 485, lines=[("Scratchpad 256 KiB", 16, "bold")])
    # Weight paths into the loader.
    arrow([(560, 95), (560, 78), (935, 78), (935, 195)])
    text(748, 70, "weights over UDP · 95 MB/s", 13, style="italic")
    arrow([(1075, 142), (1075, 195)])
    text(1082, 178, "model", 13, anchor="start", style="italic")
    # CPU <-> core.
    arrow([(830, 318), (905, 318)], double=True)
    text(867, 309, "CSRs", 13, style="italic")
    text(867, 335, "scratchpad", 13, style="italic")

    # Memory system.
    box(650, 525, 1140, 570, lines=[("AXI arbiter (interleaved)", 16, "bold"),
                                     ("4 engine channels + CPU → two 512-bit AXI targets", 13)])
    box(650, 598, 1140, 648, fill=HARD, lines=[("Hard LPDDR4x controller", 16, "bold"),
                                                ("Efinix hard IP · 2 × 512-bit AXI at 100 MHz", 13)])
    for x in (975, 1005, 1035, 1065):
        arrow([(x, 500), (x, 525)], double=True)
    text(963, 517, "4 × 256-bit", 13, anchor="end", style="italic")
    for x in (795, 995):
        arrow([(x, 570), (x, 598)], double=True, thick=True)
    box(1185, 525, 1370, 648, fill=MEM, lines=[("LPDDR4x", 18, "bold"), ("1 GB · x32", 14), ("12.8 GB/s", 14),
                                                ("weights 436 MB", 13), ("KV / state · main RAM", 13)])
    arrow([(1140, 623), (1185, 623)], double=True, thick=True)
    box(1185, 85, 1370, 150, fill=MEM, lines=[("eMMC 8 GB", 17, "bold"), ("models + tokenizer", 13), ("7 slots · CRC", 13)])
    arrow([(1185, 117), (1140, 117)], double=True)

    # Caption and logos.
    cx = 690
    text(cx, 782, "TriXium on the Efinix Ti375 C529: a standalone local LLM", 26, weight="bold", style="italic")
    text(cx, 811, "Power on → flash boot → the Bonsai-2 2B + tokenizer from the eMMC → chat at http://trixium.local/ 20 s later", 15, style="italic")
    text(cx, 834, "Ternary weights streamed from 1 GB of LPDDR4x · 14.3 tok/s · 3-5 W · no host, no GPU, no cloud", 15, style="italic")
    image(os.path.join(logos, "enjoy-digital-logo.png"), 18, 750, 110, 110)
    text(1232, 812, "Built with", 19, anchor="end")
    image(os.path.join(logos, "litex-logo.png"), 1240, 762, 148, 85)

# Simple variant -----------------------------------------------------------------------------------
#
# Columns: left of the FPGA x 245..385 | peripherals 430..600 | web server + CPU 640..820 (centre 730)
# | TriXium core group 860..1095 (inner boxes 875..1080) | off-FPGA memories 1135..1310.

def draw_simple(logos):
    begin(1360, 830, "board-ti375-simple", DESC)

    # Board and FPGA boundaries.
    box(230, 25, 1348, 690, fill=BOARD, c=34)
    box(395, 55, 1105, 620, fill=FPGA, c=30)
    text(1332, 678, "Efinix Titanium Ti375 C529 development kit · 3-5 W", 17, anchor="end", style="italic")
    text(420, 575, "Ti375 FPGA", 24, anchor="start", weight="bold", style="italic")
    text(420, 601, "LiteX SoC · 100 MHz", 15, anchor="start", style="italic")

    # The user.
    box(5, 95, 160, 195, lines=[("Laptop · phone", 18, "bold"), ("any browser", 14), ("http://trixium.local/", 14)])
    arrow([(160, 130), (245, 130)], double=True)
    text(195, 120, "Ethernet", 13, style="italic")
    text(195, 149, "HTTP chat", 13, style="italic")

    # On the board, left of the FPGA.
    box(245, 95, 385, 165, lines=[("RGMII PHY", 17, "bold"), ("RJ45 · 1 Gb/s", 14)])
    box(245, 300, 385, 370, fill=MEM, lines=[("QSPI flash", 17, "bold"), ("bitstream + firmware", 13)])
    arrow([(385, 130), (430, 130)], double=True)
    arrow([(385, 335), (430, 335)])

    # LiteX SoC (columns 1 and 2).
    box(430, 95, 600, 185, fill=KEY, lines=[("LiteEth", 18, "bold"), ("RGMII MAC", 14), ("UDP/IP · CPU MAC", 14)])
    box(430, 300, 600, 370, lines=[("LiteSPI", 18, "bold"), ("flash boot", 14)])
    box(640, 95, 820, 185, fill=KEY, lines=[("Web chat server", 17, "bold"), ("firmware TCP/IP · HTTP", 13), ("DHCP · mDNS", 14)])
    box(640, 220, 820, 380)
    text_lines(730, 263, [("VexiiRiscv", 18, "bold"), ("RV32 + FPU", 14)])
    line(653, 296, 807, 296)
    text_lines(730, 333, [("firmware:", 13, None, "italic"), ("tokenizer · chat", 13), ("inference lists · sampling", 13)])
    arrow([(600, 140), (640, 140)], double=True)
    arrow([(730, 185), (730, 220)], double=True)
    arrow([(600, 335), (640, 335)])
    arrow([(730, 380), (730, 530)], double=True)
    text(722, 460, "main RAM", 13, anchor="end", style="italic")

    # TriXium core (column 3).
    box(960, 92, 1095, 142, fill=KEY, lines=[("SD/MMC core", 17, "bold"), ("8-bit, 50 MHz", 13)])
    box(860, 165, 1095, 490, fill=GROUP, c=18)
    text(875, 190, "TriXium core", 18, anchor="start", weight="bold")
    box(875, 205, 1080, 249, lines=[("Loader", 16, "bold"), ("model → memory", 13)])
    box(875, 263, 1080, 307, lines=[("Sequencer", 16, "bold"), ("command lists", 13)])
    box(875, 321, 1080, 365, lines=[("Matvec engine", 16, "bold"), ("4 channels · ternary", 13)])
    box(875, 379, 1080, 423, lines=[("Units", 16, "bold"), ("norm · SiLU · softmax · attention", 13)])
    box(875, 437, 1080, 471, lines=[("Scratchpad 256 KiB", 15, "bold")])
    arrow([(1028, 142), (1028, 205)])
    text(1035, 182, "model", 13, anchor="start", style="italic")
    arrow([(820, 300), (860, 300)], double=True)
    text(840, 290, "CSRs", 13, style="italic")

    # Memory system.
    box(640, 530, 1095, 600, fill=HARD, lines=[("Hard LPDDR4x controller", 17, "bold"),
                                                ("2 × 512-bit AXI at 100 MHz · CPU + 4 engine channels", 13)])
    for x in (930, 960, 990, 1020):
        arrow([(x, 490), (x, 530)], double=True)
    text(918, 515, "4 × 256-bit", 13, anchor="end", style="italic")
    box(1135, 520, 1310, 610, fill=MEM, lines=[("LPDDR4x", 18, "bold"), ("1 GB · 12.8 GB/s", 14), ("weights · KV · RAM", 13)])
    arrow([(1095, 565), (1135, 565)], double=True, thick=True)
    box(1135, 85, 1310, 150, fill=MEM, lines=[("eMMC 8 GB", 17, "bold"), ("models + tokenizer", 13)])
    arrow([(1135, 117), (1095, 117)], double=True)

    # Caption and logos.
    cx = 680
    text(cx, 727, "TriXium on the Efinix Ti375 C529: a standalone local LLM", 26, weight="bold", style="italic")
    text(cx, 757, "Power on → flash boot → the 2B model from the eMMC → chat at http://trixium.local/ 20 s later", 15, style="italic")
    text(cx, 779, "14.3 tok/s · 3-5 W · no host, no GPU, no cloud", 15, style="italic")
    image(os.path.join(logos, "enjoy-digital-logo.png"), 16, 700, 100, 100)
    text(1190, 758, "Built with", 19, anchor="end")
    image(os.path.join(logos, "litex-logo.png"), 1198, 710, 150, 86)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="The Ti375 standalone diagram(s).")
    parser.add_argument("--variant", default="all", choices=["full", "simple", "all"])
    parser.add_argument("--logos",   default=here, help="Directory with enjoy-digital-logo.png and litex-logo.png.")
    parser.add_argument("--out",     default=here, help="Output directory.")
    parser.add_argument("--boxes",   default=None, help="Also write the full variant's box geometry as JSON (the video's highlights).")
    args = parser.parse_args()
    if args.variant in ("full", "all"):
        draw_full(args.logos);   finish(os.path.join(args.out, "board-ti375.svg"))
        if args.boxes:
            import json
            json.dump([dict(title=t, x1=x1, y1=y1, x2=x2, y2=y2) for t, x1, y1, x2, y2 in boxes],
                      open(args.boxes, "w"), indent=1)
    if args.variant in ("simple", "all"):
        draw_simple(args.logos); finish(os.path.join(args.out, "board-ti375-simple.svg"))

if __name__ == "__main__":
    main()

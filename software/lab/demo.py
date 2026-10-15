#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The README's demo GIF: `trixium_chat` on the card (left) and `trixium_smi -l` watching it
(right), recorded from the workstation over ssh, one scene per model.

    trixium demo --board ypcb                         # 8B then 1.7B -> doc/images/demo-ypcb.gif
    trixium demo --board ypcb --scene 8b --no-load    # the model already on the card

Each scene loads its model (`trixium deploy --no-reconfigure`, not recorded), runs both tools in
a local tmux session over ssh, types the prompts and waits for each reply, while both panes are
captured (`tmux capture-pane -e`) about ten times a second. The frames are drawn here with PIL
in the diagrams' palette and written as one GIF: no terminal recorder, no extra dependency, and
a re-recording after every speed-up is one command.
"""

import os
import re
import time
import shlex
import argparse
import subprocess

from PIL import Image, ImageDraw, ImageFont

from software.lab.deploy import REPO, board_config, Remote

# Scenes -------------------------------------------------------------------------------------------

SCENES = {
    "8b": dict(title="Ternary-Bonsai 8B", tokenizer="models/8b/tokenizer.bin", prompts=[
        "What is an FPGA? Two sentences.",
        "Write a haiku about memory bandwidth.",
    ]),
    "1p7b": dict(title="Ternary-Bonsai 1.7B", tokenizer="models/1.7b/tokenizer.bin", prompts=[
        "Give me three fun facts about Paris.",
        "Now a haiku about ternary weights.",
    ]),
}

# Layout: the chat's 72-column box on the left, trixium_smi's 88-column box on the right.
LEFT_COLS, RIGHT_COLS, ROWS = 74, 89, 30

# Capture ------------------------------------------------------------------------------------------

TMUX = ["tmux", "-L", "trixium-demo"]

def tmux(*args, check=True):
    return subprocess.run(TMUX + list(args), check=check, capture_output=True, text=True).stdout

def capture(pane):
    """The pane's visible lines, with their SGR colour sequences."""
    return tmux("capture-pane", "-p", "-e", "-t", f"demo:0.{pane}", check=False).split("\n")[:ROWS]

def capture_both():
    """Both panes in one tmux call (each prints exactly ROWS lines): half the capture time."""
    out = tmux("capture-pane", "-p", "-e", "-t", "demo:0.0", ";",
               "capture-pane", "-p", "-e", "-t", "demo:0.1", check=False).split("\n")
    return out[:ROWS], out[ROWS:2*ROWS]

def plain(lines):
    return "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l) for l in lines)

class Recorder:
    def __init__(self):
        self.frames = [] # (time, left lines, right lines, typed)
        self.typing = False

    def snap(self):
        left, right = capture_both()
        if not self.frames or self.frames[-1][1:3] != (left, right):
            self.frames.append((time.time(), left, right, self.typing))
        return left

    def wait(self, seconds, until=None):
        """Capture for `seconds`, or until `until(plain left pane text)` holds."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            left = self.snap()
            if until and until(plain(left)):
                return True
            time.sleep(0.1)
        return until is None

    def type(self, text, cps=28):
        # Each key's echo crosses ssh twice (~0.35 s here): the GIF plays these frames at the
        # typing rate (write_gif), not at the link's; the replies stay in real time.
        self.typing = True
        for ch in text:
            tmux("send-keys", "-t", "demo:0.0", "-l", ch)
            self.wait(1/cps)
        self.wait(0.4)
        self.typing = False
        tmux("send-keys", "-t", "demo:0.0", "Enter")

def record_scene(conf, scene, rec, ntok):
    remote = Remote(conf)
    ssh    = f"ssh -t {shlex.quote(conf['target'])} "
    d      = shlex.quote(conf["dir"])
    # The console tty can come back root:dialout after a reboot (the driver load opens it).
    remote.ssh(f"{remote.sudo()} chmod 666 /dev/ttyLXU0", cd=False)
    chat = ssh + shlex.quote(f"cd {d} && ./user/trixium_chat -z {scene['tokenizer']} -n {ntok}")
    smi  = ssh + shlex.quote(f"cd {d} && ./user/trixium_smi -l 0.5 -i 0.4")
    tmux("kill-server", check=False)
    tmux("new-session", "-d", "-s", "demo", "-x", str(LEFT_COLS + RIGHT_COLS + 1), "-y", str(ROWS),
         chat)
    tmux("set", "-g", "status", "off")
    tmux("split-window", "-h", "-l", str(RIGHT_COLS), "-t", "demo:0", smi)
    tmux("select-pane", "-t", "demo:0.0")
    if not rec.wait(90, until=lambda t: "you ›" in t):
        raise SystemExit("the chat did not come up:\n" + plain(capture(0)))
    rec.wait(1.5)
    def ready(text): # The reply is done: the last line is a fresh, empty prompt.
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return bool(lines) and lines[-1] == "you ›"
    for prompt in scene["prompts"]:
        rec.type(prompt)
        rec.wait(1.0)
        if not rec.wait(180, until=ready):
            raise SystemExit("no reply:\n" + plain(capture(0)))
        rec.wait(2.5)
    # The smi first: once the chat exits its pane closes and the smi's becomes pane 0.
    tmux("send-keys", "-t", "demo:0.1", "C-c", check=False)
    tmux("send-keys", "-t", "demo:0.0", "C-d", check=False)
    time.sleep(1)
    tmux("kill-server", check=False)

# Render -------------------------------------------------------------------------------------------

BG, FG, DIM, RULE = (7, 13, 12), (231, 238, 233), (140, 158, 152), (53, 194, 174)
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")

class Renderer:
    def __init__(self, size=14):
        self.font = ImageFont.truetype(FONT_PATHS[0], size)
        self.bold = ImageFont.truetype(FONT_PATHS[1], size)
        self.cw   = round(self.font.getlength("M"))
        self.ch   = size + 5
        self.pad  = 14
        self.w    = self.pad*2 + (LEFT_COLS + RIGHT_COLS + 1)*self.cw
        self.h    = self.pad*2 + ROWS*self.ch

    def line(self, draw, x0, y, text):
        fg, bold, x = FG, False, x0
        for part in re.split(r"(\x1b\[[0-9;]*m)", text):
            if part.startswith("\x1b["):
                codes = [int(c) for c in part[2:-1].split(";") if c] or [0]
                i = 0
                while i < len(codes):
                    c = codes[i]
                    if c == 0:          fg, bold = FG, False
                    elif c == 1:        bold = True
                    elif c in (22,):    bold = False
                    elif c == 39:       fg = FG
                    elif c == 38 and codes[i + 1:i + 2] == [2]:
                        fg = tuple(codes[i + 2:i + 5]); i += 4
                    elif c == 38 and codes[i + 1:i + 2] == [5]:
                        i += 2
                    elif 30 <= c <= 37 or 90 <= c <= 97:
                        fg = DIM if c in (90, 37) else FG
                    i += 1
                continue
            for ch in part:
                if ch not in " ":
                    draw.text((x, y), ch, font=self.bold if bold else self.font, fill=fg)
                x += self.cw

    def frame(self, left, right):
        img  = Image.new("RGB", (self.w, self.h), BG)
        draw = ImageDraw.Draw(img)
        for r in range(ROWS):
            y = self.pad + r*self.ch
            if r < len(left):
                self.line(draw, self.pad, y, left[r])
            if r < len(right):
                self.line(draw, self.pad + (LEFT_COLS + 1)*self.cw, y, right[r])
        x = self.pad + LEFT_COLS*self.cw + self.cw//2
        draw.line([(x, self.pad), (x, self.h - self.pad)], fill=(30, 48, 44), width=1)
        return img

    def title(self, text, sub):
        img  = Image.new("RGB", (self.w, self.h), BG)
        draw = ImageDraw.Draw(img)
        big  = ImageFont.truetype(FONT_PATHS[1], 34)
        tw   = draw.textlength(text, font=big)
        draw.text(((self.w - tw)/2, self.h/2 - 40), text, font=big, fill=RULE)
        sw   = draw.textlength(sub, font=self.font)
        draw.text(((self.w - sw)/2, self.h/2 + 12), sub, font=self.font, fill=DIM)
        return img

def write_gif(path, scenes, renderer, max_hold=2.0, key_ms=45):
    """scenes: [(title image, [(time, left, right, typed), ...])]: one GIF, idle stretches
    capped, keystroke frames at key_ms each."""
    images, durations = [], []
    for title, frames in scenes:
        images.append(title); durations.append(1800)
        for (t, l, r, typed), nxt in zip(frames, frames[1:] + [None]):
            dt = (nxt[0] - t) if nxt else 2.5
            images.append(renderer.frame(l, r))
            durations.append(key_ms if typed else int(1000*min(max(dt, 0.06), max_hold)))
    # One palette for all frames, from a strip of frames across the GIF (the first frames
    # alone have no teal or gold yet).
    picks = images[::max(1, len(images)//8)] + images[-1:]
    strip = Image.new("RGB", (renderer.w, renderer.h*len(picks)))
    for k, im in enumerate(picks):
        strip.paste(im, (0, k*renderer.h))
    pal = strip.convert("P", palette=Image.ADAPTIVE, colors=64)
    frames = [im.quantize(palette=pal, dither=Image.NONE) for im in images]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)
    return len(frames)

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Record the README's chat + smi demo GIF.")
    parser.add_argument("--board",  default="ypcb",                     help="Board (boards.ini).")
    parser.add_argument("--build",  default=None,                       help="Build to deploy with (default: the board's).")
    parser.add_argument("--scene",  action="append", choices=sorted(SCENES), help="Scenes in order (default: 8b, 1p7b).")
    parser.add_argument("--no-load", action="store_true",               help="Do not load the models (the scene's model is already up).")
    parser.add_argument("-n",       default=160, type=int,              help="Reply cap in tokens.")
    parser.add_argument("--out",    default=os.path.join(REPO, "doc", "images", "demo-ypcb.gif"), help="GIF to write.")
    args = parser.parse_args(argv)

    conf     = board_config(args.board)
    renderer = Renderer()
    scenes   = []
    for name in args.scene or ["8b", "1p7b"]:
        scene = SCENES[name]
        if not args.no_load:
            from software.lab import deploy
            print(f"[demo] loading {scene['title']} on {args.board} (not recorded)", flush=True)
            rc = deploy.deploy(args.build or args.board, args.board, name, reconfigure=False)
            if rc:
                raise SystemExit(f"deploy {name} failed ({rc})")
        print(f"[demo] recording {scene['title']}", flush=True)
        rec = Recorder()
        record_scene(conf, scene, rec, args.n)
        scenes.append((renderer.title(scene["title"], "trixium_chat on the card · trixium_smi watching it"),
                       rec.frames))
    n = write_gif(args.out, scenes, renderer)
    print(f"[demo] {args.out}: {n} frames, {os.path.getsize(args.out)/1e6:.1f} MB")

if __name__ == "__main__":
    main()

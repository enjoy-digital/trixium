#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Synthesize the TriXium video soundtrack: 120 BPM, A minor, 131 s (the film's length), one
continuous score (no re-cuts).

Section map (bars; 1 bar = 2 s) - the video's chapter cuts use the same grid:
  0-7 opening + intro | 7-11 ternary | 11-16 FK33 design | 16-19 exact (breakdown) | 19-26 climb (drop)
  26-30 upstream | 30-42 8B/27B, XCU1525, YPCB-00338 | 42-46 memory (half-time)
  46-53 every vendor / smaller homes / everywhere | 53-57 what's next | 57-60 open source (breakdown,
  roll into 60) | final hit on bar 60 (the thanks) | 60-63 credits bed | 63-65.5 end card, fade

Usage: python3 music.py music.wav [variant] && ffmpeg -i music.wav -b:a 192k music.mp3
Variants (same tempo, grid, hits and risers; other harmony and instruments): original (default), epic,
original, warm, techno, cinematic, chiptune.
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter
import wave, sys

SR = 44100
BPM = 120
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS = 66
TAIL = 0.0
N = int(131.0 * SR)   # the film is 131 s; the last bar is cut by the fade
FH = 60               # the final hit (into the thanks), bar 60 = 120 s
rng = np.random.default_rng(7)
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "original"
assert VARIANT in ("original", "warm", "techno", "cinematic", "chiptune", "epic"), VARIANT

def buf(): return np.zeros(N)
def t_(dur): return np.arange(int(dur * SR)) / SR
def add(dst, x, at):
    i = int(round(at * SR))
    if i >= N: return
    x = x[: N - i]
    dst[i:i + len(x)] += x
def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)

def sections(b):
    if b < 7: return "intro"
    if b < 11: return "ternary"
    if b < 16: return "arch"
    if b < 19: return "exact"
    if b < 26: return "climb"
    if b < 30: return "litex"     # upstream: the limits removed
    if b < 42: return "big"       # 8B/27B, XCU1525, YPCB-00338
    if b < 46: return "walls"     # the memory wall, the DDR rates
    if b < 53: return "future"    # every vendor, smaller homes, everywhere
    if b < 57: return "litex"     # what's next
    if b < 60: return "opensrc"
    return "outro"

# i - VI - III - VII in A minor
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
ROOTS = [45, 41, 48, 43]
if VARIANT == "warm":        # IV - I - V - vi in C major: brighter
    CHORDS = [[53, 57, 60], [48, 52, 55], [55, 59, 62], [57, 60, 64]]
    ROOTS = [41, 48, 43, 45]
if VARIANT == "cinematic":   # i - v - VI - VII
    CHORDS = [[57, 60, 64], [52, 55, 59], [53, 57, 60], [55, 59, 62]]
    ROOTS = [45, 40, 41, 43]
if VARIANT == "techno":      # i - i - iv - i: a drone that moves little
    CHORDS = [[57, 60, 64], [57, 60, 64], [50, 53, 57], [57, 60, 64]]
    ROOTS = [45, 45, 38, 45]

# ---------------------------------------------------------------- instruments
def kick():
    t = t_(0.42)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 7.5)
    x[:60] += rng.standard_normal(60) * 0.25 * np.linspace(1, 0, 60)
    return x * 0.95

def snare():
    t = t_(0.25)
    n = rng.standard_normal(len(t))
    n = lfilter([1, -1], [1], n) * 0.5
    x = n * np.exp(-t * 16) * 0.55 + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25) * 0.4
    return x

def hat(open_=False):
    t = t_(0.18 if open_ else 0.05)
    n = rng.standard_normal(len(t))
    n = sosfilt(butter(4, 7000, "hp", fs=SR, output="sos"), n)
    return n * np.exp(-t * (14 if open_ else 70)) * 0.22

def saw(freq, dur, detune=(0.0,), phase_rand=True):
    t = t_(dur)
    x = np.zeros(len(t))
    for d in detune:
        f = freq * 2 ** (d / 1200)
        p = rng.random() if phase_rand else 0
        x += 2 * ((t * f + p) % 1.0) - 1
    return x / len(detune)

def pulse(freq, dur, width=0.5):
    t = t_(dur)
    return np.where((t * freq) % 1.0 < width, 1.0, -1.0)

def tri(freq, dur):
    t = t_(dur)
    return 2 * np.abs(2 * ((t * freq) % 1.0) - 1) - 1

def pluck(freq, dur):
    """Piano-ish: a few decaying harmonics."""
    t = t_(dur)
    return sum(np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (3 + 2.5 * k)) / k for k in (1, 2, 3, 4))

def env(n, a, d, s, r, dur):
    """ADSR in seconds, note held for dur (seconds), n samples total."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = t > dur
    e[rel] = e[rel] * np.exp(-(t[rel] - dur) / max(r, 1e-4))
    return e

def lp_sweep(x, cutoffs, block=512, order=2):
    """Time-varying lowpass: cutoffs is a function of time (s) -> Hz."""
    y = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoffs(i / SR), 40, SR * 0.45))
        sos = butter(order, fc, "lp", fs=SR, output="sos")
        if zi is None:
            zi = sosfilt_zi(sos) * 0
        y[i:i + block], zi = sosfilt(sos, x[i:i + block], zi=zi)
    return y

def comb(x, d, fb):
    """y[n] = x[n] + fb*y[n-d], computed a block of d samples at a time."""
    y = x.copy()
    for i in range(d, len(y), d):
        e = min(len(y), i + d)
        y[i:e] += fb * y[i - d:e - d]
    return y

def delay(x, sec, fb, mix):
    d = int(sec * SR)
    dx = np.concatenate([np.zeros(d), x[:-d]])
    return x + comb(dx, d, fb) * mix

def reverb(x, mix=0.25):
    out = np.zeros_like(x)
    for ms, g in [(29.7, .78), (37.1, .77), (41.1, .76), (43.7, .75)]:
        out += comb(x, int(ms * SR / 1000), g)
    out = sosfilt(butter(1, 5000, "lp", fs=SR, output="sos"), out) / 4
    return x + out * mix

# ---------------------------------------------------------------- tracks
drums, bass, pad, arp, lead, fx = buf(), buf(), buf(), buf(), buf(), buf()
K, S = kick(), snare()
Hc, Ho = hat(), hat(True)
kick_times = []

for b in range(NBARS):
    sec = sections(b)
    t0 = b * BAR
    ch = CHORDS[b % 4]
    root = ROOTS[b % 4]

    # --- drums
    four = sec in ("ternary", "arch", "climb", "big", "outro", "bugs", "future", "litex")
    if VARIANT == "cinematic":
        four = sec in ("climb", "big", "outro")
    half = sec == "walls"
    if four and not (sec == "outro" and b >= FH):
        for q in range(4):
            add(drums, K, t0 + q * BEAT); kick_times.append(t0 + q * BEAT)
    if half:
        add(drums, K, t0); kick_times.append(t0)
        if b % 2 == 1: add(drums, K, t0 + 2.5 * BEAT); kick_times.append(t0 + 2.5 * BEAT)
    if (sec in ("arch", "climb", "big", "outro", "litex") or (sec == "future" and b >= 50)) and b < FH:
        add(drums, S, t0 + BEAT); add(drums, S, t0 + 3 * BEAT)
    if half:
        add(drums, S * 0.9, t0 + 2 * BEAT)
    if VARIANT == "techno" and sec not in ("intro", "exact", "opensrc") and b < FH:
        for s16 in range(16):   # 16th hats, accents on the off-beat 8ths
            add(drums, Hc * (0.9 if s16 % 4 == 2 else 0.35), t0 + s16 * BEAT / 4)
    elif VARIANT == "cinematic":
        pass
    elif sec in ("ternary", "arch", "climb", "big", "outro", "walls", "future", "litex") and b < FH:
        for e in range(8):
            h = Ho if (e % 2 == 1 and sec in ("climb", "big", "litex")) else Hc
            add(drums, h * (0.6 if e % 2 == 0 else 1.0), t0 + e * BEAT / 2)
    if sec == "bugs":
        # stuttered 16th hats, snare on 4
        for s16 in range(16):
            add(drums, Hc * (1.0 if s16 % 4 == 2 else 0.5), t0 + s16 * BEAT / 4)
        add(drums, S, t0 + 3 * BEAT)
        if b == 31:
            for s16 in range(8):
                add(drums, S * (0.3 + 0.08 * s16), t0 + 2 * BEAT + s16 * BEAT / 4)
    if (sec == "exact" and b >= 17) or (sec == "opensrc" and b >= 58):
        # snare roll into the drop
        k = b - (16 if sec == "exact" else 57)
        n = 8 if k == 1 else 16
        for i in range(n):
            add(drums, S * (0.25 + 0.6 * (i / n) * k / 2), t0 + i * BAR / n)
    if b in (7, 11, 19, 26, 30, 42, 46, 48, 53, 60):
        # crash-ish noise swell hit
        tt = t_(2.0)
        cr = sosfilt(butter(2, 3500, "hp", fs=SR, output="sos"), rng.standard_normal(len(tt))) * np.exp(-tt * 2.2) * 0.22
        add(fx, cr, t0)

    # --- bass: 8ths, octave jump on offbeats (techno: rolling 16ths; cinematic: held notes;
    # chiptune: pulse)
    if VARIANT == "techno" and sec not in ("intro", "exact", "opensrc") and b < FH:
        for s16 in range(16):
            if s16 % 4 == 0: continue   # leave room for the kick
            m = root - 12 + (12 if s16 % 4 == 3 else 0)
            dur = BEAT / 4 * 0.8
            x = saw(mtof(m), dur + 0.03, detune=(-4, 4)) * env(int((dur + 0.03) * SR), 0.002, 0.06, 0.4, 0.02, dur)
            add(bass, x * 0.55, t0 + s16 * BEAT / 4)
    elif VARIANT == "cinematic" and sec not in ("intro",) and b < FH:
        m = root - 12
        x = (np.sin(2 * np.pi * mtof(m) * t_(BAR + 0.4)) + 0.3 * tri(mtof(m) * 2, BAR + 0.4))
        x *= env(len(x), 0.08, 1.2, 0.8, 0.4, BAR)
        add(bass, x * 0.4, t0)
    elif sec not in ("intro", "exact", "opensrc") and b < FH:
        for e in range(8):
            if sec == "walls" and e % 2: continue
            m = root - 12 + (12 if e % 2 else 0)
            dur = BEAT / 2 * 0.9
            x = pulse(mtof(m), dur + 0.05, 0.25) if VARIANT == "chiptune" else saw(mtof(m), dur + 0.05, detune=(-6, 6))
            x *= env(len(x), 0.003, 0.12, 0.6, 0.03, dur)
            add(bass, x * (0.35 if VARIANT == "chiptune" else 0.5), t0 + e * BEAT / 2)
    if sec in ("exact", "opensrc") or b >= FH:
        m = root - 12
        x = np.sin(2 * np.pi * mtof(m) * t_(BAR)) * env(int(BAR * SR), 0.05, 1.0, 0.7, 0.3, BAR * 0.9)
        add(bass, x * 0.35, t0)

    # --- pad: whole-bar detuned saw chords
    dur = BAR
    for m in ch + [ch[0] + 12]:
        if VARIANT == "warm":
            x = (tri(mtof(m), dur + 0.6) + 0.5 * saw(mtof(m), dur + 0.6, detune=(-7, 7))) * 0.8
        elif VARIANT == "cinematic":
            x = saw(mtof(m), dur + 1.2, detune=(-14, -5, 5, 14))
        elif VARIANT == "chiptune":
            x = pulse(mtof(m), dur + 0.3, 0.5) * 0.5
        else:
            x = saw(mtof(m), dur + 0.6, detune=(-11, 0, 9))
        x *= env(len(x), 0.6 if VARIANT == "cinematic" else 0.25, 1.5, 0.75, 0.9 if VARIANT == "cinematic" else 0.5, dur)
        add(pad, x * (0.17 if VARIANT == "cinematic" else 0.13), t0)

    # --- arp: 16ths over chord tones, 2 octaves
    if sec not in ("bugs",) and b < 65:
        tones = ch + [c + 12 for c in ch]
        patt = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 5, 4, 3, 1]
        for s16 in range(16):
            m = tones[patt[s16] % len(tones)] + 12
            d = BEAT / 4
            if VARIANT in ("warm", "cinematic"):
                if VARIANT == "cinematic" and s16 % 2: continue   # 8ths, piano-like
                x = pluck(mtof(m), d * (6 if VARIANT == "cinematic" else 3))
                g = 0.16 if VARIANT == "cinematic" else 0.12
            elif VARIANT == "techno":
                if s16 % 4 not in (1, 3): continue                 # sparse off-beat blips
                x = pulse(mtof(m + 12), d * 1.2, 0.15)
                x *= env(len(x), 0.001, 0.04, 0.0, 0.02, d)
                g = 0.06
            else:
                x = pulse(mtof(m), d * 1.4, (0.5 if VARIANT == "chiptune" else 0.3) if sec != "walls" else 0.5)
                x *= env(len(x), 0.002, 0.07, 0.0, 0.02, d)
                g = 0.075
            add(arp, x * (g * 1.33 if sec in ("intro", "exact", "walls", "opensrc") else g), t0 + s16 * d)
    if sec == "bugs":
        # detuned stabs on every beat, glitchy
        for q in range(4):
            for m in ch:
                x = saw(mtof(m + 12), 0.14, detune=(-20, 20))
                x *= env(len(x), 0.001, 0.05, 0.0, 0.02, 0.08)
                add(arp, x * 0.14, t0 + q * BEAT + (BEAT / 2 if q == 3 else 0))

# --- lead melody in the drops (climb, big, outro), 4-bar phrase
MEL = [  # (bar offset in beats, midi, beats)
    (0, 76, 1.5), (1.5, 74, 0.5), (2, 72, 1), (3, 74, 1),
    (4, 72, 1.5), (5.5, 71, 0.5), (6, 69, 2),
    (8, 67, 1.5), (9.5, 69, 0.5), (10, 71, 1), (11, 72, 1),
    (12, 74, 2), (14, 71, 1), (15, 67, 1),
]
MEL2 = [(o, m + (3 if m in (69, 72) else 0) if o >= 12 else m, d) for o, m, d in MEL]
if VARIANT == "warm":      # the same contour in C major, a third up at the end
    MEL  = [(o, {71: 72, 69: 67, 67: 67}.get(m, m), d) for o, m, d in MEL]
    MEL2 = [(o, m + 4 if o >= 12 else m, d) for o, m, d in MEL]
if VARIANT == "techno":    # a sparse two-note hook
    MEL = MEL2 = [(0, 76, 0.5), (3, 72, 0.5), (8, 76, 0.5), (11, 74, 1)]
for b0, until, mel in [(21, 26, MEL), (33, 42, MEL2), (53, 57, MEL)]:
    b = b0
    while b + 4 <= until or (b < until):
        for o, m, d in mel:
            at = b * BAR + o * BEAT
            if at >= until * BAR: continue
            dur = d * BEAT
            tt = t_(dur + 0.3)
            f = mtof(m) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt) * np.clip(tt - 0.2, 0, 1))
            ph = np.cumsum(f) / SR
            if VARIANT == "chiptune":
                x = np.where((ph % 1) < 0.25, 1.0, -1.0) * 0.7
            elif VARIANT == "cinematic":
                x = np.sin(2 * np.pi * ph) + 0.3 * np.sin(4 * np.pi * ph)   # soft bell-ish
            else:
                x = (2 * (ph % 1) - 1) * 0.6 + (2 * ((ph * 1.005) % 1) - 1) * 0.4
            x *= env(len(x), 0.01, 0.3, 0.7, 0.15, dur)
            add(lead, x * (0.16 if VARIANT == "cinematic" else 0.13), at)
        b += 4
# final hit
add(fx, np.sin(2 * np.pi * 55 * t_(3.0)) * np.exp(-t_(3.0) * 1.5) * 0.6, FH * BAR)
for m in [57, 64, 69, 72, 76]:
    x = saw(mtof(m), 4.0, detune=(-12, 0, 12)) * env(int(4 * SR), 0.005, 1.2, 0.3, 1.0, 1.5)
    add(fx, x * 0.09, FH * BAR)

# ---------------------------------------------------------------- cinematic fx: risers + sub impacts
def riser(dur):
    tt = t_(dur)
    n = rng.standard_normal(len(tt))
    y = np.zeros_like(n); blk = 1024
    zi = None
    for i in range(0, len(n), blk):
        fc = 300 * (40 ** (i / len(n)))
        sos = butter(2, [fc, min(fc * 2.5, SR * .45)], "bp", fs=SR, output="sos")
        if zi is None: zi = sosfilt_zi(sos) * 0
        y[i:i + blk], zi = sosfilt(sos, n[i:i + blk], zi=zi)
    return y * (tt / dur) ** 2 * 0.5
def impact():
    tt = t_(2.5)
    f = 30 + 60 * np.exp(-tt * 6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6)
    return x * 0.7
for at in (14, 22, 38, 52, 60, 66, 76, 84, 92, 96, 106, 114, 120, 126):
    d = 2.0 if at in (38, 60, 120) else 1.0
    add(fx, riser(d), at - d)
    add(fx, impact(), at)

# ---------------------------------------------------------------- mix
def cut_arp(t):
    b = t / BAR; s = sections(int(min(b, NBARS - 1)))
    if s == "intro": return 500 + 2500 * (b / 7) ** 2
    if s == "exact": return 900 + 3000 * max(0, (b - 16) / 3) ** 2
    if s == "opensrc": return 900 + 3000 * max(0, (b - 57) / 3) ** 2
    if s == "walls": return 1400 - 150 * (b - 42)
    if s == "future": return 1500 + 3700 * (b - 46) / 7
    return 5200
def cut_pad(t):
    b = t / BAR; s = sections(int(min(b, NBARS - 1)))
    if s == "intro": return 400 + 1400 * (b / 7)
    if s in ("climb", "big", "litex"): return 3200
    if s == "future": return 1200 + 2000 * (b - 46) / 7
    if s == "walls": return 900
    return 2200
def cut_bass(t):
    b = t / BAR; s = sections(int(min(b, NBARS - 1)))
    return 900 if s in ("climb", "big", "bugs", "litex") else 600

arp = lp_sweep(arp, cut_arp)
pad = lp_sweep(pad, cut_pad)
bass = lp_sweep(bass, cut_bass, order=4)

# sidechain from kick positions
sc = np.ones(N)
for kt in kick_times:
    i = int(kt * SR); n = int(0.28 * SR)
    seg = 1 - 0.7 * np.exp(-np.arange(n) / SR * 14)
    seg = np.minimum(seg, 1)
    e = min(N, i + n)
    sc[i:e] = np.minimum(sc[i:e], seg[: e - i])

arp = delay(arp, BEAT * 0.75, 0.35, 0.35)
lead = delay(lead, BEAT * 0.75, 0.3, 0.3)
music = drums * 0.9 + bass * sc * 0.9 + reverb(pad * sc, 0.5) + reverb(arp * sc, 0.3) + reverb(lead, 0.35) + reverb(fx, 0.4)

# ---------------------------------------------------------------- variant "epic": its own arrangement
# Hans Zimmer-ish, on the same grid and hit points: a driving low-string 16th ostinato, BRAAM
# brass hits on the chapter impacts and the final hit, taiko/timpani thumps (half-time in the calm
# parts, driving in the climb and the 8B/27B/XCU1525 run), a wide choir/strings pad, a heroic horn
# motif in the climb and the finale, a long swell into the final hit. No hats, no snare.
IMPACTS = (14, 22, 38, 52, 60, 66, 76, 84, 92, 96, 106, 114, 126)
def epic_score():
    CH  = {"calm": [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]],   # Am F C G
           "tense": [[57, 60, 64], [53, 57, 60], [50, 53, 57], [52, 56, 59]]}  # Am F Dm E
    RT  = {"calm": [45, 41, 48, 43], "tense": [45, 41, 38, 40]}
    DRIVE = {"climb": 1.0, "big": 1.0, "litex": .8, "arch": .65, "future": .6, "ternary": .45, "walls": .5}
    strings, brass, drums_e, choir, horn, swell = buf(), buf(), buf(), buf(), buf(), buf()
    def taiko(f0=110, dur=0.6, g=1.0):
        tt = t_(dur); f = 45 + (f0 - 45) * np.exp(-tt * 18)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 5.5)
        n = sosfilt(butter(2, 400, "lp", fs=SR, output="sos"), rng.standard_normal(len(tt))) * np.exp(-tt * 30) * .5
        return (x + n) * g
    def braam(root, dur=4.0, g=1.0):
        tt = t_(dur); x = np.zeros(len(tt))
        for m in (root - 12, root - 5, root, root + 3):
            x += saw(mtof(m), dur, detune=(-14, -5, 0, 6, 15))[:len(tt)] + 0.4 * pulse(mtof(m) / 2, dur, 0.4)[:len(tt)]
        x *= env(len(x), 0.04, 0.8, 0.7, 2.2, dur * 0.55)
        return lp_sweep(x, lambda t: 250 + 2200 * min(1.0, (t / 1.4) ** 1.5), order=2) * g
    for b in range(NBARS):
        sec = sections(b); t0 = b * BAR
        mode = "tense" if sec in ("climb", "big") else "calm"
        ch, root = CH[mode][b % 4], RT[mode][b % 4]
        live = b < FH
        # choir / strings pad: many detuned saws, slow attack
        for m in ch + [ch[0] + 12, ch[0] - 12]:
            x = saw(mtof(m), BAR + 1.2, detune=(-18, -11, -5, 0, 4, 10, 17))
            x *= env(len(x), 1.0 if sec in ("intro", "exact", "opensrc") or not live else 0.5, 1.5, 0.85, 1.2, BAR)
            add(choir, x * (0.07 if live else 0.05), t0)
        # low-string ostinato: staccato 16ths (8ths when calm), root and fifth
        d = DRIVE.get(sec, 0.0) if live else 0.0
        if sec == "opensrc": d = 0.3 + 0.7 * (b - 57) / 3
        if d > 0:
            step = 16 if d >= .6 else 8
            for s16 in range(step):
                m = root - 12 + (7 if s16 % 4 == 3 else 0) + (12 if s16 % 8 == 6 else 0)
                dd = BAR / step
                x = saw(mtof(m), dd * .9, detune=(-9, 0, 8)) * env(int(dd * .9 * SR), 0.004, 0.07, 0.25, 0.03, dd * .6)
                add(strings, x * 0.32 * (0.55 + 0.45 * d) * (1.15 if s16 % 4 == 0 else 1.0), t0 + s16 * dd)
        # taiko / timpani: half-time when calm, driving in the climb and the big run
        if live and sec not in ("intro", "exact"):
            if sec in ("climb", "big"):
                for q, g in [(0, 1.0), (1.5, .6), (2, .85), (3, .7), (3.5, .55)]:
                    add(drums_e, taiko(120 if q == 0 else 95, g=g), t0 + q * BEAT)
            elif sec == "opensrc":
                for q in range(4 * (b - 56)):
                    add(drums_e, taiko(100, 0.4, 0.35 + 0.15 * (b - 57)), t0 + q * BEAT / (b - 56))
            elif b % 2 == 0:
                add(drums_e, taiko(105, g=.8), t0); add(drums_e, taiko(90, g=.5), t0 + 2 * BEAT)
    # BRAAM hits on the chapter impacts, the biggest on the final hit
    for at in IMPACTS:
        b = int(at // BAR); mode = "tense" if sections(min(b, NBARS - 1)) in ("climb", "big") else "calm"
        add(brass, braam(RT[mode][b % 4] - 12, 3.5, 0.55), at)
    add(brass, braam(33, 6.0, 1.0), FH * BAR)
    add(drums_e, taiko(130, 1.2, 1.6), FH * BAR)
    # heroic horn motif: the climb and the finale build
    MOT = [(0, 69, 1.5), (1.5, 72, .5), (2, 76, 2), (4, 74, 1), (5, 72, 1), (6, 71, 2),
           (8, 69, 1.5), (9.5, 72, .5), (10, 76, 1), (11, 79, 1), (12, 81, 4)]
    for b0 in (19, 22, 56):
        for o, m, dd in MOT:
            at = b0 * BAR + o * BEAT
            if at >= FH * BAR: continue
            dur = dd * BEAT; tt = t_(dur + .5)
            x = (saw(mtof(m - 12), dur + .5, detune=(-6, 0, 7))[:len(tt)] + .5 * pulse(mtof(m - 12), dur + .5, .45)[:len(tt)])
            x *= env(len(x), 0.08, 0.3, 0.8, 0.4, dur)
            add(horn, x * 0.16, at)
    horn = lp_sweep(horn, lambda t: 1800, order=2)
    # the swell into the final hit: a rising filtered noise and a choir crescendo
    sw0 = 110.0; tt = t_(FH * BAR - sw0)
    nz = rng.standard_normal(len(tt)) * (tt / tt[-1]) ** 2.5
    add(swell, lp_sweep(nz, lambda t: 300 + 4000 * (t / (FH * BAR - sw0)) ** 2) * .25, sw0)
    strings = lp_sweep(strings, lambda t: 900 + 1600 * DRIVE.get(sections(int(min(t / BAR, NBARS - 1))), .3))
    choir = lp_sweep(choir, lambda t: 2400)
    gain = np.ones(N); i0, i1 = int(sw0 * SR), int(FH * BAR * SR)
    gain[i0:i1] = np.linspace(1, 1.6, i1 - i0)
    return (strings * 0.9 + reverb(brass, 0.45) * 0.8 + drums_e * 0.9 + reverb(choir * gain, 0.65) * 1.1
            + reverb(horn, 0.4) + reverb(swell, 0.5))

if VARIANT == "epic":
    music = epic_score()

# fade the last 3 s
fade = int(3.0 * SR)
music[-fade:] *= np.linspace(1, 0, fade) ** 1.5
music = sosfilt(butter(2, 25, "hp", fs=SR, output="sos"), music)
music /= np.max(np.abs(music)) + 1e-9
music = np.tanh(music * 1.6) / np.tanh(1.6) * 0.89

st = np.stack([music, np.concatenate([np.zeros(485), music[:-485]])], 1)  # mono-compatible stereo
out = sys.argv[1] if len(sys.argv) > 1 else "music.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype("<i2").tobytes())
print(out, N / SR, "s")

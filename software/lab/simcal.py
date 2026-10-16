#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Board prediction from a board-timing simulation (./trixium_sim.py --board-timing --bench).

The board-timing simulation runs a shape-true slice (model/configs.py: the real widths and head
counts, a few layers, a truncated vocabulary) on the XCU1525's clocks, DDR4 timings and maps.
This module parses the firmware's own profile of that run (BENCH, hybprof/dnprof, the op table,
the engine/vector overlap counters), normalizes it per layer and extrapolates the full model:

    token = sum(per-layer phase x the model's layers) + embed + LM head (its chunk lists x the
            vocabulary ratio) + the rest of the measured decode token outside those phases

and compares it with the board's measurements (doc/simulation.md): the XCU1525 v30b, or the
YPCB-00338's 8-channel trit5 build (--board ypcb).

W/P: W is the engine-busy time (weight streams, and the DeltaNet state / K/V streams of ops 5/6),
P the rest (CPU, vector unit, scratchpad units with the engine idle), from the overlap counters.
Stream efficiency: weight bytes over the board's DRAM peak (XCU1525: 4 controllers x 64 B per ddr
clock = 384 B per sys clock at 187.5:125 MHz; YPCB: 2 x 64 B at 133.33:125 MHz, 136.5 B) for the
time they take; per phase where a phase is one weight stream.
"""

import os
import re

from model.bons import BONSReader
from model.configs import SLICE_OF
from model import layout

# Board ---------------------------------------------------------------------------------------------

SYS_CLK_FREQ = 125e6
PEAK_BYTES   = 4*64*187.5/125 # DDR4 bytes per sys clock: four x64 controllers at DDR4-1500.

# XCU1525 v30b (sys 125 MHz, DDR4-1500, doc/timeline.md): decode cycles per token and, for the
# 27B, the per-forward profile of `bench 4` (hybprof/dnprof averaged over 6 forwards, 4 of them
# with the LM head: logits is 0.87 M per forward, 1.305 M per head).
BOARD_V30B = {
    "27B"  : dict(cycles=32.2e6, weights=6.904e9,
                  phases={"fproj": 9.36e6, "fdown": 6.04e6, "dn": 10.7e6, "attn": 2.72e6,
                          "logits": 0.87e6*6/4},
                  dn={"inproj": 1.71e6, "act": 2.13e6, "conv": 1.87e6, "heads": 2.47e6,
                      "outproj": 1.75e6}),
    "8B"   : dict(cycles=SYS_CLK_FREQ/14.9, weights=2.010e9),
    "1.7B" : dict(cycles=SYS_CLK_FREQ/53.8, weights=0.457e9),
}

# YPCB-00338, build/ypcb_t5n8c (8 channels, trit5, sys 125 MHz, DDR3-1066; doc/boards.md): the
# decode cycles per token of `bench`, 3-token prompt.
BOARD_YPCB_T5N8 = {
    "8B"   : dict(cycles=15973678),
    "1.7B" : dict(cycles=3890287),
}

# Per board: DRAM peak bytes per sys clock, its label, the reference build and its measurements.
BOARDS = {
    "xcu1525" : dict(peak=PEAK_BYTES, label="DDR4 peak, 384 B/clock", ref="v30b", meas=BOARD_V30B),
    "ypcb"    : dict(peak=2*64*(400/3)/125, label="DDR3 peak, 136.5 B/clock", ref="ypcb_t5n8c",
                     meas=BOARD_YPCB_T5N8),
}

# Which per-layer phases a layer kind has (hybprof), and the weight roles each stream phase reads.
PHASES_ALL = ("anorm", "resid", "fprep", "fproj", "fact", "fdown")
STREAMS    = {
    "fproj"      : ("ffn_gate", "ffn_up"),
    "fdown"      : ("ffn_down",),
    "dn.inproj"  : ("attn_qkv", "attn_zgate", "ssm_beta", "ssm_alpha"),
    "dn.outproj" : ("ssm_out",),
    "logits"     : ("lm_head",),
}

def lm_chunk_rows(n_ch):
    """The hybrid firmware's LM-head chunk (hybrid.c lm_chunk_rows: the scratchpad room above
    RS_F_ADDR, at most 24576 rows, whole 8 x n_ch units)."""
    unit = 8*n_ch
    return min((0x32000 - 0x1C800)//4, 24576)//unit*unit

# Parse ---------------------------------------------------------------------------------------------

def _pairs(line):
    """`name value` pairs after `cycles/fwd` on a hybprof/dnprof/attnprof line."""
    tail = line.split("cycles/fwd", 1)[1]
    return {k: int(v) for k, v in re.findall(r"([a-z][\w-]*) (\d+)", tail)}

def parse_bench(text):
    """The firmware's bench + stats output -> dict (missing parts absent)."""
    out = {"ops": {}}
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"(\d+) forwards, (\d+) cycles/fwd", line)
        if line.startswith("BENCH:") and m:
            out["forwards"], out["cycles_fwd"] = int(m.group(1)), int(m.group(2))
        elif line.startswith("BENCH_PHASE"):
            out["phase"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("hybprof:"):
            out["hybprof"] = _pairs(line)
        elif line.startswith("dnprof:"):
            out["dnprof"] = _pairs(line)
        elif line.startswith("attnprof:"):
            out["attnprof"] = _pairs(line)
        elif line.startswith("listprof lm head:"):
            m = re.search(r"(\d+) cycles/call", line)
            out["lmhead_call"] = int(m.group(1))
        elif line.startswith("overlap:"):
            # The bench's own (the later `stats` window also counts the idle console).
            m = re.search(r"eng (\d+)%\s+vec/dn (\d+)%\s+both (\d+)%\s+neither (\d+)%\s+\(of (\d+)", line)
            if m and "overlap" not in out:
                out["overlap"] = dict(zip(("eng", "vec", "both", "neither", "total"),
                                          map(int, m.groups())))
        else:
            m = re.match(r"^([a-z_0-9]+)\s+(\d+)\s+(\d+)\s+\d+", line)
            if m:
                out["ops"][m.group(1)] = dict(calls=int(m.group(2)), cycles=int(m.group(3)))
    return out

# Image ---------------------------------------------------------------------------------------------

def stream_bytes(image):
    """Streamed weight bytes per forward of an image, by role, all channels (ternary/Q8 matvec
    weights: the token embedding only as a tied LM head, the Hadamard tensors never)."""
    r     = BONSReader(image)
    n_ch  = r.header["n_channels"]
    roles = {}
    for e in r.entries:
        if e["kind"] not in (layout.KIND_TERNARY, layout.KIND_TERNARY5, layout.KIND_Q8):
            continue
        name = layout.ROLE_NAMES[e["role"]]
        roles[name] = roles.get(name, 0) + e["chan_len_beats"]*32*n_ch
    roles.pop("hadamard", None)
    emb = roles.pop("tok_emb", 0)
    if "lm_head" not in roles:
        roles["lm_head"] = emb # Tied: the head streams the embedding.
    return roles, r.header

# Predict -------------------------------------------------------------------------------------------

def predict(profile, image, board="xcu1525"):
    """Per-layer cycles of the slice run and the full-model prediction (a decode token) -> dict."""
    model     = SLICE_OF.get(_slice_name(image))
    if model is None or "cycles_fwd" not in profile:
        return None
    roles, h  = stream_bytes(image)
    n_ch      = h["n_channels"]
    types     = h["layer_types"] or ("attn",)*h["n_layer"]
    n_dn_s    = sum(t == "deltanet" for t in types)
    n_attn_s  = len(types) - n_dn_s
    n_layer_s = len(types)
    n_dn, n_attn = model["n_dn"], model["n_attn"]
    n_layer   = n_dn + n_attn
    rows_s    = -(-h["vocab"]//n_ch)*n_ch    # Head rows (padded to the channels).
    rows_f    = -(-model["vocab"]//n_ch)*n_ch
    vocab     = rows_f/rows_s
    fwd       = profile["cycles_fwd"]
    nf        = profile.get("forwards", 1)
    ph        = profile.get("phase", {})
    heads     = ph.get("generated_tokens") or 1 # One LM head per generated token.
    # The decode token: measured when the bench decoded (bench >= 2), else a forward.
    dec       = ph["decode_cycles"]/ph["decode_tokens"] if ph.get("decode_tokens") else fwd
    res       = dict(model=model["model"], fwd=fwd, dec=dec, layers={}, phases={}, eff={})
    peak      = BOARDS[board]["peak"]

    hyb = profile.get("hybprof")
    if hyb:
        # Hybrid: hybprof's per-forward phases per layer kind (the same in every forward), the
        # head per call: its chunk lists scale with the rows, the rest (final norm, rotation,
        # argmax) does not; the remainder of the decode token outside hybprof per token.
        per = {"dn": hyb["dn"]/max(n_dn_s, 1), "attn": hyb["attn"]/max(n_attn_s, 1)}
        for p in PHASES_ALL:
            per[p] = hyb[p]/n_layer_s
        head  = hyb["logits"]*nf/heads
        lists = profile.get("lmhead_call", 0)*-(-rows_s//lm_chunk_rows(n_ch))
        lists = min(lists, head)
        full  = {"dn": per["dn"]*n_dn, "attn": per["attn"]*n_attn}
        for p in PHASES_ALL:
            full[p] = per[p]*n_layer
        full["embed"]  = hyb["embed"]
        full["logits"] = head - lists + lists*vocab
        full["other"]  = max(dec - (hyb["total"] - hyb["logits"] + head), 0)
        res["layers"]  = dict(per, head=head)
        res["phases"]  = full
        dn = profile.get("dnprof", {})
        res["dn"] = {k: v/max(n_dn_s, 1) for k, v in dn.items() if k != "total"}
        # Stream efficiency of the one-stream phases.
        cyc = {"fproj": hyb["fproj"], "fdown": hyb["fdown"], "logits": lists,
               "dn.inproj": dn.get("inproj", 0), "dn.outproj": dn.get("outproj", 0)}
        for p, rs in STREAMS.items():
            b = sum(roles.get(r, 0) for r in rs)
            if b and cyc.get(p):
                res["eff"][p] = b/(cyc[p]*peak)
    else:
        # Qwen3: the LM head (one argmax op over every row: scales with them) and the embedding
        # per call from the op table, the rest of the decode token per layer.
        ops    = profile["ops"]
        lm     = ops.get("lmhead", {})
        emb    = ops.get("embed",  {})
        head   = lm.get("cycles", 0)/max(lm.get("calls", 1), 1)
        embed  = emb.get("cycles", 0)/max(emb.get("calls", 1), 1)
        per    = (dec - head - embed)/n_layer_s
        res["layers"] = {"layer": per, "head": head}
        res["phases"] = {"layers": per*n_layer, "embed": embed, "logits": head*vocab}
        if roles.get("lm_head") and head:
            res["eff"]["logits"] = roles["lm_head"]/(head*peak)
    res["predicted"] = sum(res["phases"].values())

    # Weight bytes extrapolated like the cycles; W/P from the overlap counters (the engine-busy
    # share of the bench), extrapolated at the slice's streaming efficiency.
    layer_b = sum(v for k, v in roles.items() if k != "lm_head")
    res["weights"] = layer_b/n_layer_s*n_layer + roles.get("lm_head", 0)*vocab
    ovl = profile.get("overlap")
    if ovl and ovl["eng"]:
        res["w"] = ovl["eng"]/100*dec
        res["p"] = dec - res["w"]
        slice_b  = layer_b + roles.get("lm_head", 0)*(heads/nf)
        eff      = slice_b*nf/(ovl["eng"]/100*ovl["total"]*peak)
        res["eff"]["forward (W)"] = eff
        res["w_full"] = res["weights"]/(eff*peak)
        res["p_full"] = res["predicted"] - res["w_full"]
    return res

def _slice_name(image):
    base = os.path.basename(image)
    return next((name for name in SLICE_OF if base.startswith(name)), None)

# Report --------------------------------------------------------------------------------------------

def _m(v):
    return f"{v/1e6:7.3f} M"

def _err(sim, board):
    return f"{100*(sim - board)/board:+6.1f}%"

def report(profile, image, out=print, board="xcu1525"):
    """Print the prediction and the board comparison; returns the prediction dict (or None)."""
    res = predict(profile, image, board)
    if res is None:
        out("calibration: not a shape-true slice image (model/configs.py SLICE_OF) or no BENCH line")
        return None
    b     = BOARDS[board]
    board = b["meas"].get(res["model"], {})
    out(f"slice: {res['fwd']} cycles/fwd, {res['dec']:.0f} cycles/decode token; per layer: " +
        ", ".join(f"{k} {v:.0f}" for k, v in res["layers"].items()))
    if "w" in res:
        out(f"slice W/P: W (engine busy) {res['w']:.0f} ({100*res['w']/res['dec']:.0f}%), "
            f"P {res['p']:.0f} cycles/token")
    if res["eff"]:
        out(f"stream efficiency (of the {b['label']}): " +
            ", ".join(f"{k} {100*v:.0f}%" for k, v in res["eff"].items()))
    out(f"predicted {res['model']} (board, 125 MHz): {res['predicted']/1e6:.2f} M cycles/token = "
        f"{SYS_CLK_FREQ/res['predicted']:.2f} tok/s, {res['weights']/1e9:.3f} GB/token" +
        (f"; W {res['w_full']/1e6:.2f} M, P {res['p_full']/1e6:.2f} M" if "w_full" in res else ""))
    if not board:
        return res
    out(f"board {b['ref']}: {board['cycles']/1e6:.2f} M cycles/token = {SYS_CLK_FREQ/board['cycles']:.2f} tok/s "
        f"(prediction {_err(res['predicted'], board['cycles'])})")
    if "layers" in res["phases"]:
        # Qwen3: the board's per layer, its token less the predicted head and embedding.
        n     = SLICE_OF[_slice_name(image)]["n_attn"]
        sim   = res["phases"]["layers"]/n
        per_b = (board["cycles"] - res["phases"]["embed"] - res["phases"]["logits"])/n
        out(f"per layer: sim {sim:.0f}, board {per_b:.0f} cycles ({_err(sim, per_b)}; the board's "
            f"token less the predicted LM head {res['phases']['logits']/1e6:.3f} M)")
    if board.get("phases"):
        out(f"{'phase':12s} {'sim':>10s} {'board':>10s} {'error':>7s}   (full model, per token; "
            "logits per LM head)")
    for k, v in board.get("phases", {}).items():
        out(f"{k:12s} {_m(res['phases'].get(k, 0))} {_m(v)} {_err(res['phases'].get(k, 0), v)}")
    for k, v in board.get("dn", {}).items():
        sim = res["dn"].get(k, 0)*SLICE_OF[_slice_name(image)]["n_dn"]
        out(f"{'dn.' + k:12s} {_m(sim)} {_m(v)} {_err(sim, v)}")
    return res

# Run -----------------------------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Board prediction from a board-timing bench log.")
    parser.add_argument("log",   help="bench log (build/sim_xcu1525_bt_<image>/bench/bench-as-built.log).")
    parser.add_argument("image", help="The slice image the simulation was built with.")
    parser.add_argument("--board", default="xcu1525", choices=sorted(BOARDS), help="The board simulated.")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8") as f:
        report(parse_bench(f.read()), args.image, board=args.board)

if __name__ == "__main__":
    main()

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""trixium dashboard: a browser page that watches the board think, next to a chat.

  python3 -m software.host.dashboard --csr-csv build/ti375_v18/csr.csv [--port 8375] [--watts 7.5]
  python3 -m software.host.dashboard --fake                       # synthetic data, no board

A small HTTP server (standard library only) samples the board over Etherbone a few times a second
and streams one JSON object per sample to the page (Server-Sent Events, /events):
- the engine and scratchpad-unit utilization and the per-channel DDR read rate (the counters of
  software/host/ebsmi.py, read-only, next to a running chat);
- where a forward's cycles go (embedding, DeltaNet, attention, FFN, LM head): the firmware's
  per-stage rdcycle totals, published once per forward in a block at the start of sram (struct
  hyb_live, firmware/hybrid.c, linker.ld's .live) - firmwares before it show the hardware counters
  only;
- forwards per second (tokens, prompt included) and the chat tool's record (decode tok/s);
- watts: typed on the page (a USB-C power meter between the supply and the kit) or --watts.
"""

import os
import sys
import json
import math
import time
import random
import struct
import argparse
import collections
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from software.host import ebsmi

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# The firmware's live block (firmware/hybrid.c struct hyb_live) -------------------------------------

LIVE_MAGIC   = 0x4556494c             # "LIVE"
LIVE_STAGES  = ("embed", "anorm", "dn", "attn", "resid", "fprep", "fproj", "fact", "fdown", "logits")
LIVE_WORDS   = (16 + 8 + 8*len(LIVE_STAGES) + 8)//4  # Header, now, cyc[], weight_beats.

# The page's groups of the firmware's stages.
GROUPS = (("embed",     "Embedding", ("embed",)),
          ("deltanet",  "DeltaNet",  ("dn",)),
          ("attention", "Attention", ("attn",)),
          ("ffn",       "FFN",       ("fprep", "fproj", "fact", "fdown")),
          ("lmhead",    "LM head",   ("logits",)),
          ("other",     "Norms + residual", ("anorm", "resid")))

def parse_live(words):
    raw = struct.pack(f"<{len(words)}I", *words)
    magic, version, seq, forwards, now = struct.unpack_from("<IIIIQ", raw, 0)
    cyc = struct.unpack_from(f"<{len(LIVE_STAGES)}Q", raw, 24)
    weight_beats, = struct.unpack_from("<Q", raw, 24 + 8*len(LIVE_STAGES))
    return dict(magic=magic, version=version, seq=seq, forwards=forwards, now=now,
                cyc=dict(zip(LIVE_STAGES, cyc)), weight_beats=weight_beats)

class BoardSource:
    """Samples a board through a litex_server (ebsmi.Card) and the firmware's live block."""
    def __init__(self, csr_csv, host="localhost", port=1234, interval=0.25):
        self.card     = ebsmi.Card(csr_csv, host, port)
        self.interval = interval
        # The start of sram (linker.ld's .live): on the shared bus, unlike main_ram on the Ti375 -
        # never read an address the bus does not decode: each read waits out the bus timeout and
        # holds the bus the firmware's CSR accesses need.
        sram = getattr(self.card.bus.mems, "sram", None)
        self.live_addr = sram.base if sram is not None else None
        self.last = None

    def identity(self):
        return self.card.identifier()

    def live(self):
        if self.live_addr is None:
            return None
        for _ in range(4):
            words = self.card.bus.read(self.live_addr, LIVE_WORDS)
            l = parse_live(words)
            if l["magic"] != LIVE_MAGIC:
                return None
            if l["seq"] % 2 == 0 and self.card.bus.read(self.live_addr + 8) == l["seq"]:
                return l
        return None

    def sample(self):
        s = ebsmi.status(self.card, self.interval)
        s["live"] = self.live()
        return s

    def close(self):
        self.card.close()

class FakeSource:
    """Synthetic samples shaped like the 2B on the Ti375 (v18): a chat that alternates prompts,
    generation and idle time, so the page can be developed and shown without a board."""
    STAGE_CYC = dict(embed=245e3, anorm=12e3, dn=2.53e6, attn=0.8e6, resid=5e3, fprep=0,
                     fproj=1.85e6, fact=0, fdown=18e3, logits=1.37e6)

    def __init__(self, interval=0.25):
        self.interval = interval
        self.t0 = time.monotonic()
        self.cyc = {k: 0.0 for k in LIVE_STAGES}
        self.forwards, self.pos, self.tokens = 0.0, 0, 0
        self.fwd_acc = 0.0

    def identity(self):
        return "TriXium Ti375 (synthetic data)"

    def sample(self):
        time.sleep(self.interval)
        t = time.monotonic() - self.t0
        phase = t % 30.0
        busy = phase < 24.0
        prefill = phase < 3.0
        if phase < 0.3:
            self.pos, self.tokens = 0, 0
        n = 0
        if busy:
            # Attention grows with the context; the rest is per token.
            per = dict(self.STAGE_CYC, attn=self.STAGE_CYC["attn"] + 1.6e3*self.pos)
            fwd_cycles = sum(per.values())
            self.fwd_acc += 100e6*self.interval/fwd_cycles
            n = int(self.fwd_acc)
            self.fwd_acc -= n
            for _ in range(n):
                for k in LIVE_STAGES:
                    self.cyc[k] += per[k]*random.uniform(0.98, 1.02)
                self.pos += 1
                if not prefill:
                    self.tokens += 1
            self.forwards += n
        rate = n/self.interval
        util = 0.93 if busy else 0.0
        ch = [util*random.uniform(0.72, 0.84) for _ in range(4)]
        chat = dict(pid=0, tool="trixium_chat", model="Bonsai2 2B", tokens=self.tokens,
                    state="generating" if busy and not prefill else "prompt" if busy else "idle",
                    tok_s=14.3 if busy and not prefill else 0.0, ttft=1.2)
        return dict(identifier=self.identity(), platform="efinix_ti375_c529_dev_kit", clock_mhz=100.0,
                    cpu="running", bus_errors=0, engine_error=0, engine_run=int(busy), engine_busy=int(busy),
                    seconds=self.interval, engine=util*0.82, units=util*0.35, overlap=util*0.2,
                    ops_s=rate*900, prefetch_s=0.0, channels=4,
                    gbs=sum(ch)/4*6.4, peak_gbs=6.4, request_wait=0.05 if busy else 0.0,
                    channel_util=ch, load=dict(image="bonsai2-2b-n4-ti375.img", bytes=452e6, time=time.time() - 600),
                    chat=chat,
                    live=dict(magic=LIVE_MAGIC, version=1, seq=0, forwards=int(self.forwards),
                              now=int(t*100e6), cyc=dict(self.cyc), weight_beats=3400000))

    def close(self):
        pass

# Sample -> page record ----------------------------------------------------------------------------

class Tracker:
    """Turns successive samples into the page's record: rates and the per-forward cycle split
    over the last window (the split holds its last value while the board idles)."""
    def __init__(self, watts=None, window=1.0):
        self.watts  = watts
        self.window = window
        self.hist   = []                  # (time, forwards, cyc dict) live snapshots
        self.split  = None
        self.split_fwd = 0

    def record(self, s):
        now = time.monotonic()
        rec = {k: s[k] for k in ("identifier", "platform", "clock_mhz", "cpu", "bus_errors",
                                 "engine_error", "engine_run", "engine_busy", "engine", "units",
                                 "overlap", "ops_s", "channels", "gbs", "peak_gbs", "request_wait",
                                 "channel_util")}
        rec["time"] = time.time()
        rec["chat"] = s.get("chat")
        load = s.get("load")
        rec["model"] = os.path.basename(load["image"]) if load and load.get("image") else None
        live = s.get("live")
        rec["live"] = live is not None
        rec["fwd_s"] = None
        if live:
            if self.hist and live["forwards"] < self.hist[-1][1]:
                self.hist = []            # The firmware restarted (hdrload): a new baseline.
            self.hist.append((now, live["forwards"], live["cyc"]))
            while len(self.hist) > 2 and now - self.hist[1][0] >= self.window:
                self.hist.pop(0)
            t0, f0, c0 = self.hist[0]
            df = live["forwards"] - f0
            dt = now - t0
            rec["fwd_s"] = df/dt if dt > 0 else 0.0
            rec["forwards"] = live["forwards"]
            if df > 0:
                per = {k: (live["cyc"][k] - c0[k])/df for k in LIVE_STAGES}
                self.split = {g: sum(per[k] for k in ks) for g, _, ks in GROUPS}
                self.split_fwd = sum(per.values())
            rec["weight_beats"] = live["weight_beats"]
        rec["split"] = self.split
        rec["split_cycles"] = self.split_fwd
        rec["groups"] = [(g, label) for g, label, _ in GROUPS]
        rec["watts"] = self.watts
        return rec

# HTTP ---------------------------------------------------------------------------------------------

class Hub:
    """The latest record and the last two minutes' rates (a page opened later draws them),
    shared between the sampler and the page connections."""
    def __init__(self, history=480):
        self.cond    = threading.Condition()
        self.rec     = None
        self.count   = 0
        self.error   = None
        self.history = collections.deque(maxlen=history)

    def publish(self, rec):
        with self.cond:
            self.rec, self.count = rec, self.count + 1
            self.history.append(dict(tok=rec.get("fwd_s") or 0.0, gbs=rec.get("gbs") or 0.0))
            self.cond.notify_all()

    def wait(self, after, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.count != after, timeout)
            return self.count, self.rec

def sampler(source, tracker, hub, stop):
    while not stop.is_set():
        try:
            hub.publish(tracker.record(source.sample()))
            hub.error = None
        except Exception as e:            # A dropped Etherbone read: report it, keep going.
            hub.error = f"{type(e).__name__}: {e}"
            time.sleep(1.0)

def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/history":
                with hub.cond:
                    body = json.dumps(list(hub.history)).encode()
                self.send(200, body, "application/json")
            elif self.path == "/api/sample":
                self.send(200, json.dumps(dict(hub.rec or {}, error=hub.error)).encode(), "application/json")
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                count = -1
                try:
                    while True:
                        count, rec = hub.wait(count)
                        payload = json.dumps(dict(rec or {}, error=hub.error))
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send(404, b"not found\n", "text/plain")
    return Handler

def serve(source, port=8375, bind="127.0.0.1", watts=None):
    hub, stop = Hub(), threading.Event()
    tracker = Tracker(watts)
    t = threading.Thread(target=sampler, args=(source, tracker, hub, stop), daemon=True)
    t.start()
    httpd = ThreadingHTTPServer((bind, port), make_handler(hub))
    httpd.daemon_threads = True
    print(f"trixium dashboard on http://{bind if bind != '0.0.0.0' else 'localhost'}:{port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        source.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csr-csv", default=None,               help="The loaded build's csr.csv.")
    parser.add_argument("--host",    default="localhost",        help="litex_server host.")
    parser.add_argument("--server-port", default=1234, type=int, help="litex_server port.")
    parser.add_argument("--port",    default=8375, type=int,     help="The page's HTTP port.")
    parser.add_argument("--bind",    default="127.0.0.1",        help="HTTP address (0.0.0.0: other machines too).")
    parser.add_argument("-i",        default=0.25, type=float,   help="Counter sampling interval (s).", dest="interval")
    parser.add_argument("--watts",   default=None, type=float,   help="The board's power, if known (the page also takes it).")
    parser.add_argument("--fake",    action="store_true",        help="Synthetic data, no board.")
    args = parser.parse_args(argv)
    if args.fake:
        source = FakeSource(args.interval)
    else:
        if not args.csr_csv:
            parser.error("--csr-csv (or --fake)")
        source = BoardSource(args.csr_csv, args.host, args.server_port, args.interval)
    serve(source, args.port, args.bind, args.watts)

if __name__ == "__main__":
    main()

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/host/dashboard.py: the firmware's live block layout (struct hyb_live, firmware/hybrid.c)
and the records the page gets (rates, the per-token cycle split, firmware restarts), without a
board."""

import re
import struct
import json
import threading
import urllib.request

from software.host import dashboard as d

REPO_FW = "firmware/hybrid.c"


def live_words(forwards, cyc, seq=2, now=123456789, weight_beats=3400000, magic=d.LIVE_MAGIC):
    raw = struct.pack("<IIIIQ", magic, 1, seq, forwards, now)
    raw += struct.pack(f"<{len(d.LIVE_STAGES)}Q", *[cyc[k] for k in d.LIVE_STAGES])
    raw += struct.pack("<Q", weight_beats)
    return list(struct.unpack(f"<{len(raw)//4}I", raw))


def test_layout_matches_firmware():
    # The stage order, the magic and the block offset are the firmware's.
    src = open(REPO_FW).read()
    assert re.search(r"#define HYB_LIVE_MAGIC\s+0x%08xu" % d.LIVE_MAGIC, src)
    m = re.search(r"uint64_t cyc\[HYB_LIVE_STAGES\];\s*//\s*([a-z ]+)", src)
    assert m and tuple(m.group(1).split()) == d.LIVE_STAGES
    assert re.search(r"#define HYB_LIVE_STAGES\s+%d\b" % len(d.LIVE_STAGES), src)
    ld = open("firmware/linker.ld").read()
    assert ld.index(".live (NOLOAD)") < ld.index("\t.data :")     # The first thing in sram.
    assert re.search(r'section\("\.live"\)', src)
    assert len(live_words(0, dict.fromkeys(d.LIVE_STAGES, 0))) == d.LIVE_WORDS


def test_parse_live():
    cyc = {k: (i + 1)*1000003 + (1 << 33) for i, k in enumerate(d.LIVE_STAGES)}
    l = d.parse_live(live_words(77, cyc))
    assert l["magic"] == d.LIVE_MAGIC and l["forwards"] == 77 and l["cyc"] == cyc
    assert l["weight_beats"] == 3400000 and l["now"] == 123456789


def sample(forwards, cyc):
    s = dict(identifier="t", platform="p", clock_mhz=100.0, cpu="running", bus_errors=0,
             engine_error=0, engine_run=1, engine_busy=1, engine=0.8, units=0.3, overlap=0.1,
             ops_s=1.0, channels=4, gbs=4.5, peak_gbs=6.4, request_wait=0.0,
             channel_util=[0.7]*4, load=None, chat=None)
    s["live"] = d.parse_live(live_words(forwards, cyc))
    return s


def test_tracker_split_and_restart():
    per = dict(embed=245e3, anorm=12e3, dn=2.5e6, attn=0.8e6, resid=5e3, fprep=0, fproj=1.85e6,
               fact=0, fdown=18e3, logits=1.37e6)
    t = d.Tracker(watts=7.5, window=60.0)
    cyc = dict.fromkeys(d.LIVE_STAGES, 0)
    r = t.record(sample(0, cyc))
    assert r["split"] is None and r["live"]
    for n in range(1, 5):
        cyc = {k: int(cyc[k] + 3*per[k]) for k in d.LIVE_STAGES}
        r = t.record(sample(3*n, cyc))
    assert abs(r["split"]["deltanet"] - 2.5e6) < 1
    assert abs(r["split"]["ffn"] - (1.85e6 + 18e3)) < 1
    assert abs(r["split"]["other"] - 17e3) < 1
    assert abs(r["split_cycles"] - sum(per.values())) < 10
    assert r["watts"] == 7.5 and r["forwards"] == 12
    # hdrload restarts the counters: a new baseline, the last split kept until forwards run.
    r = t.record(sample(0, dict.fromkeys(d.LIVE_STAGES, 0)))
    assert r["fwd_s"] == 0.0 and abs(r["split"]["deltanet"] - 2.5e6) < 1


def test_server_fake():
    src = d.FakeSource(interval=0.02)
    hub, stop = d.Hub(), threading.Event()
    th = threading.Thread(target=d.sampler, args=(src, d.Tracker(), hub, stop), daemon=True)
    th.start()
    httpd = d.ThreadingHTTPServer(("127.0.0.1", 0), d.make_handler(hub))
    srv = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv.start()
    try:
        port = httpd.server_address[1]
        hub.wait(0, 2.0)                  # The first sample.
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
        assert b"<title>TriXium Live</title>" in page
        rec = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sample", timeout=5).read())
        assert rec["identifier"] and rec["error"] is None and len(rec["channel_util"]) == 4
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=5) as ev:
            line = ev.readline()
            assert line.startswith(b"data: ") and json.loads(line[6:])["channels"] == 4
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()


def test_sampler_reconnects():
    # An interrupted read leaves the litex_server stream out of step: the sampler reconnects
    # instead of reporting the same error forever.
    class Flaky(d.FakeSource):
        def __init__(self):
            super().__init__(interval=0.01)
            self.broken, self.reconnects = True, 0

        def sample(self):
            if self.broken:
                raise ValueError("unpack_from requires a buffer of at least 24 bytes")
            return super().sample()

        def reconnect(self):
            self.broken, self.reconnects = False, self.reconnects + 1

    src, hub, stop = Flaky(), d.Hub(), threading.Event()
    th = threading.Thread(target=d.sampler, args=(src, d.Tracker(), hub, stop), daemon=True)
    th.start()
    try:
        assert hub.wait(0, 5.0) and src.reconnects == 1 and hub.error is None
    finally:
        stop.set()

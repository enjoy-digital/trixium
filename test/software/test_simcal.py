#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/lab/simcal.py: the board-timing bench parser and the full-model extrapolation."""

import pytest

from software.lab import simcal

# A board-timing bench of the 27B slice (3 DeltaNet + 1 attention layers, vocab 8192), the
# firmware's line formats.
BENCH_27B = """\
op            calls cycles %instrumented (not %wall)
embed        3 30000 1
total (sum of instrumented ops; excludes engine_wait): 30000 cycles; wall 6200000 cycles
hybprof: 4 forwards, avg cycles/fwd embed 10000 anorm 8000 dn 600000 attn 160000 resid 12000 fprep 4000 fproj 600000 fact 0 fdown 380000 logits 30000 total 1804000
dnprof: avg cycles/fwd inproj 90000 act 120000 conv 100000 prep 9000 heads 150000 gate 30000 outproj 100000 total 599000
listprof lm head: 0 calls/fwd, 20000 cycles/call: 985 1325 15000 2690
overlap: eng 60%  vec/dn 20%  both 10%  neither 30%  (of 6200000 cycles; coherent64; neither includes CPU work)
BENCH: 2 tokens, 7600000 cycles total, 3800000 cycles/token (32.895 tok/s @125MHz), 4 forwards, 1900000 cycles/fwd
BENCH_PHASE v=1 valid=1 clock_hz=125000000 requested_tokens=2 generated_tokens=2 prompt_tokens=3 decode_tokens=1 decode_cycles=1890000 total_cycles=7600000
"""

HEADER_27B = dict(n_channels=16, n_layer=4, vocab=8192,
                  layer_types=("deltanet", "deltanet", "deltanet", "attn"))


def test_parse_bench():
    p = simcal.parse_bench(BENCH_27B)
    assert p["forwards"] == 4 and p["cycles_fwd"] == 1900000
    assert p["phase"]["decode_cycles"] == 1890000 and p["lmhead_call"] == 20000
    assert p["hybprof"]["fdown"] == 380000 and p["hybprof"]["total"] == 1804000
    assert p["dnprof"]["heads"] == 150000
    assert p["overlap"]["eng"] == 60 and p["overlap"]["total"] == 6200000
    assert p["ops"]["embed"] == dict(calls=3, cycles=30000)


def test_predict_hybrid(monkeypatch):
    roles = {"ffn_gate": 1e8, "ffn_up": 1e8, "ffn_down": 1e8, "attn_qkv": 4e7, "lm_head": 1e7}
    monkeypatch.setattr(simcal, "stream_bytes", lambda image: (roles, HEADER_27B))
    res = simcal.predict(simcal.parse_bench(BENCH_27B), "models/slices/bonsai-27b-slice-n16.img")
    assert res["model"] == "27B"
    # Per layer: DeltaNet over its 3 layers, FFN phases over all 4; the full model's counts.
    assert res["layers"]["dn"] == pytest.approx(200000)
    assert res["phases"]["dn"] == pytest.approx(200000*48)
    assert res["phases"]["attn"] == pytest.approx(160000*16)
    assert res["phases"]["fproj"] == pytest.approx(150000*64)
    # The head per call (4 forwards, 2 heads): its chunk list scaled by the rows (248320 vs
    # 8192), the rest kept.
    head = 30000*4/2
    assert res["phases"]["logits"] == pytest.approx(head - 20000 + 20000*248320/8192)
    # The decode token outside hybprof's phases.
    assert res["phases"]["other"] == pytest.approx(1890000 - (1804000 - 30000 + head))
    assert res["predicted"] == pytest.approx(sum(res["phases"].values()))
    assert res["w"] == pytest.approx(0.6*1890000)
    assert res["eff"]["fproj"] == pytest.approx(2e8/(600000*simcal.PEAK_BYTES))


def test_predict_not_a_slice():
    assert simcal.predict(simcal.parse_bench(BENCH_27B), "models/tiny/bonsai-tiny-n8.img") is None

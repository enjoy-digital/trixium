#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""trixium smi on a synthetic card: register map parsing, rates, table and JSON output."""

import os
import json

from trixium import smi

CSR = os.path.join(os.path.dirname(__file__), "data", "smi_csr.csv")

# Tests --------------------------------------------------------------------------------------------

def test_csr_map():
    m = smi.CSRMap(CSR)
    assert m.identifier.startswith("TriXium on XCU1525")
    assert m.constant("trixium_n_channels") == 16
    assert m.offset("identifier_mem") == 0x800

def test_query_fake_card():
    m    = smi.CSRMap(CSR)
    card = smi.FakeCard(m)
    d    = smi.query(card, m, interval=0.05)
    assert d["ok"] and d["identifier"] == m.identifier
    assert abs(d["clock_mhz"] - 125.0) < 5
    assert 0.6 < d["util"]["engine"] < 0.8
    assert d["memory"]["channels"] == 16 and d["memory"]["gbs"] > 0
    text = smi.table([d])
    assert "TriXium on XCU1525" in text and "16 channels" in text
    json.dumps(d)

def test_cli_fake(capsys):
    assert smi.main(["--fake", "--csr", CSR, "--interval", "0.02", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["device"] == "/dev/fake0"

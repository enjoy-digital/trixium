#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import json
from pathlib import Path

from migen.fhdl import verilog

from trixium.gateware.fp32 import (
    BonsaiFP32Add, BonsaiFP32Mul, FP32_ADD_LATENCY, FP32_MUL_LATENCY,
)


ROOT = Path(__file__).resolve().parents[1]


def _ip_config(name):
    with (ROOT / "ip" / "fp32" / name / f"{name}.xci").open() as f:
        xci = json.load(f)
    return xci["ip_inst"]["parameters"]["component_parameters"]


def _value(params, key):
    return params[key][0]["value"]


def test_fp32_xci_contract():
    add = _ip_config("fp32_add")
    mul = _ip_config("fp32_mul")
    assert _value(add, "Operation_Type") == "Add_Subtract"
    assert _value(add, "Add_Sub_Value") == "Add"
    assert _value(mul, "Operation_Type") == "Multiply"
    for cfg, latency in ((add, FP32_ADD_LATENCY), (mul, FP32_MUL_LATENCY)):
        assert _value(cfg, "A_Precision_Type") == "Single"
        assert _value(cfg, "Result_Precision_Type") == "Single"
        assert _value(cfg, "Flow_Control") == "NonBlocking"
        assert _value(cfg, "Has_RESULT_TREADY") == "false"
        assert int(_value(cfg, "C_Rate")) == 1
        assert int(_value(cfg, "C_Latency")) == latency


def test_fp32_wrappers_have_paired_valid_and_no_backpressure():
    for dut, primitive in ((BonsaiFP32Add(), "fp32_add"),
                           (BonsaiFP32Mul(), "fp32_mul")):
        text = str(verilog.convert(dut, ios={dut.a, dut.b, dut.valid,
                                            dut.result, dut.result_valid}))
        assert primitive in text
        assert "s_axis_a_tvalid" in text
        assert "s_axis_b_tvalid" in text
        assert "m_axis_result_tvalid" in text
        assert "m_axis_result_tready" not in text

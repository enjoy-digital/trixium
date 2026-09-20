#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Pipelined IEEE-754 binary32 operators for the DeltaNet recurrence engine.

The Xilinx Floating-Point Operator cores are configured by ``tools/gen_fp32_ip.tcl`` for
single precision, round-to-nearest-even, one result per cycle and no stream backpressure.
Their fixed latencies are part of the recurrence scheduler's contract: independent rows
are interleaved far enough apart that an accumulator result is back before that row is
issued again.

These wrappers intentionally expose raw 32-bit payloads.  Migen does no arithmetic on
them, so synthesis and firmware see exactly the IEEE bits produced by the IP.
"""

from migen import *

from litex.gen import *


FP32_ADD_LATENCY = 11
FP32_MUL_LATENCY = 8


class _BonsaiFP32Binary(LiteXModule):
    def __init__(self, primitive, latency):
        self.a = Signal(32)
        self.b = Signal(32)
        self.valid = Signal()
        self.result = Signal(32)
        self.result_valid = Signal()
        self.latency = latency

        self.specials += Instance(primitive,
            i_aclk                = ClockSignal(),
            i_s_axis_a_tvalid     = self.valid,
            i_s_axis_a_tdata      = self.a,
            i_s_axis_b_tvalid     = self.valid,
            i_s_axis_b_tdata      = self.b,
            o_m_axis_result_tvalid= self.result_valid,
            o_m_axis_result_tdata = self.result,
        )


class BonsaiFP32Add(_BonsaiFP32Binary):
    latency = FP32_ADD_LATENCY

    def __init__(self):
        _BonsaiFP32Binary.__init__(self, "fp32_add", self.latency)


class BonsaiFP32Mul(_BonsaiFP32Binary):
    latency = FP32_MUL_LATENCY

    def __init__(self):
        _BonsaiFP32Binary.__init__(self, "fp32_mul", self.latency)


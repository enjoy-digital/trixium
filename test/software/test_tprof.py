#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""software/lab/tprof.py: the token trace's engine-idle breakdown."""

from software.lab import tprof

# begin, layer 0, enter norm_i32 (op 9), list @0+5 started, exit norm_i32, enter resid_add
# (op 11), exit resid_add, end: timer / engine-busy / vector-busy counters.
TRACE = """TPROF 8 events (cap 6000)
TP 60000000 1000 50 7
TP 30000000 1100 50 7
TP 10000900 1200 50 7
TP 50000005 1300 120 7
TP 20000900 2300 1000 207
TP 10000b00 2400 1000 207
TP 20000b00 3400 1000 1007
TP 70000000 3500 1000 1007
"""

def test_tprof_breakdown():
    res = tprof.analyze(tprof.parse(TRACE))
    cycles, idle, vec = res["total"]
    assert cycles == 2500
    assert idle == 2500 - 950
    assert vec == 1000
    # The list interval inside norm_i32: 1000 cycles, 880 of them streaming.
    assert res["by_op"]["norm_i32"] == [1100, 1100 - 950, 200]
    assert res["by_op"]["resid_add"] == [1000, 1000, 800]
    assert res["by_list"]["list@0+5"][0] == 1000 + 100 + 1000 + 100
    assert res["gaps"][0][:2] == (1000, 1000) and res["gaps"][0][3] == "resid_add"


def test_tprof_counter_wrap():
    """32-bit timer and counters wrap between two events."""
    text = "TP 60000000 4294967200 4294967290 0\nTP 70000000 104 10 0\n"
    res  = tprof.analyze(tprof.parse(text))
    assert res["total"][:2] == [200, 200 - 16]

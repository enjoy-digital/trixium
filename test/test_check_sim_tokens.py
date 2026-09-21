# SPDX-License-Identifier: BSD-2-Clause
import pytest

from tools.check_sim_tokens import parse_legs


GEN = "GEN: 143 244\nGEN DONE\n"
HW = "GEN-HW: 143 244\nGEN-HW DONE\n"
SP = "GEN-SP: 143 244 1 2 3 4\nGEN-SP DONE\n"


def test_all_boot_legs_required():
    assert parse_legs("boot chatter\n" + GEN + HW + SP) == {
        "GEN": [143, 244], "GEN-HW": [143, 244], "GEN-SP": [143, 244, 1, 2, 3, 4]}
    assert parse_legs((GEN + HW + SP).replace("\n", "\n\r"))["GEN"] == [143, 244]


def test_explicitly_scoped_diagnostic():
    assert parse_legs(GEN, ["GEN"]) == {"GEN": [143, 244]}


def test_interactive_generations_do_not_replace_boot_legs():
    boot = GEN + HW + SP
    assert parse_legs(boot + "BONSAI-READY\nGEN: 143\nGEN DONE\n") == parse_legs(boot)
    with pytest.raises(ValueError):
        parse_legs(GEN + HW + "BONSAI-READY\n" + SP)


@pytest.mark.parametrize("required", [[], ["GEN", "GEN"], ["unknown"]])
def test_invalid_gate_scope(required):
    with pytest.raises(ValueError):
        parse_legs(GEN, required)


@pytest.mark.parametrize("text", ["", GEN, GEN + HW, GEN + HW + "GEN-SP: 143",
                                 GEN.replace("GEN DONE", "") + HW + SP,
                                 GEN.replace("143 244", "143") + HW + SP,
                                 GEN.replace("143 244", "143 -1") + HW + SP,
                                 GEN.replace("143 244", "[engine wedged]") + HW + SP,
                                 GEN + GEN + HW + SP,
                                 "GEN DONE\nGEN: 143 244\n" + HW + SP])
def test_incomplete_or_ambiguous_simulation_fails(text):
    with pytest.raises(ValueError):
        parse_legs(text)

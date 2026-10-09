#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Trace validation must fail closed, including absent/malformed board output."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from software.lab.mirror import cmd_compare, parse, fnv, format_typed_trace


EMBED = "DBG pos=0 l=-1 embed e=-12 1234abcd amax=42 v0..3=1,2,3,4\n"
ARGMAX = "DBG pos=0 l=-1 argmax e=0 00000003\n"


# Helpers ------------------------------------------------------------------------------------------

def compare(tmp_path, mirror, card):
    a = SimpleNamespace(mirror=tmp_path / "mirror", card=tmp_path / "card", verbose=True)
    a.mirror.write_text(mirror)
    a.card.write_text(card)
    return cmd_compare(a)


# Tests --------------------------------------------------------------------------------------------

def test_matching_trace_with_chatter(tmp_path):
    assert compare(tmp_path, EMBED + ARGMAX, "boot chatter\n" + EMBED + ARGMAX) == 0
    assert compare(tmp_path, EMBED + ARGMAX, "GEN:" + EMBED + " 11751" + ARGMAX) == 0
    assert compare(tmp_path, EMBED, "DBG armed for 1 forwards\nGEN:" + EMBED) == 0


@pytest.mark.parametrize("mirror,card", [
    (EMBED + ARGMAX, EMBED),               # the former false-positive
    (EMBED, EMBED + ARGMAX),              # wrong trace contract
    (EMBED, ""), ("", EMBED), ("", ""),
    (EMBED, "no supported trace points\n"),
    (EMBED, EMBED.replace("e=-12", "e=-11")),
    (EMBED, EMBED.replace("1234abcd", "1234abce")),
    (EMBED, EMBED + EMBED),               # don't silently overwrite duplicate keys
    (EMBED + EMBED, EMBED),
    (EMBED, EMBED.replace("1234abcd", "1234abcd0")),
    (EMBED, "DBG broken or truncated line\n"),
])
def test_incomplete_or_invalid_trace_fails(tmp_path, mirror, card):
    assert compare(tmp_path, mirror, card) == 1


def test_diagnostics_are_not_part_of_checksum(tmp_path):
    assert compare(tmp_path, EMBED, EMBED.replace("amax=42", "amax=43")) == 0
    assert compare(tmp_path, EMBED, EMBED.replace("1234abcd", "1234ABCD")) == 0


def test_parse_reports_line_and_key(tmp_path):
    path = tmp_path / "trace"
    path.write_text(EMBED + EMBED)
    with pytest.raises(ValueError, match=r":2: duplicate checkpoint .*embed"):
        parse(path)


def test_missing_file_fails(tmp_path):
    assert cmd_compare(SimpleNamespace(mirror=tmp_path / "absent", card="", verbose=False)) == 1


def test_typed_float_bits_not_integer_cast(tmp_path):
    import struct
    line = format_typed_trace(0, 2, "ffn_res", "f32", [0.0, -0.0, 1.25, -2.5]) + "\n"
    expected = fnv(struct.pack("<ffff", 0.0, -0.0, 1.25, -2.5))
    assert f"dtype=f32 n=4 e=0 {expected:08x}" in line
    assert compare(tmp_path, line, line) == 0
    # Same hash/exponent isn't sufficient: array shape and interpretation are part of ABI.
    assert compare(tmp_path, line, line.replace("n=4", "n=5")) == 1
    assert compare(tmp_path, line, line.replace("dtype=f32", "dtype=i32")) == 1
    assert compare(tmp_path, line, line.replace("dtype=f32 n=4 ", "")) == 1
    assert compare(tmp_path, line, line.replace("n=4", "n=0")) == 1


def test_typed_argmax_hash():
    import struct
    line = format_typed_trace(2, -1, "argmax", "u32", [11751])
    assert line.endswith(f"{fnv(struct.pack('<I', 11751)):08x}")


def test_ids_do_not_require_tokenizer():
    from software.lab.mirror import prompt_ids
    assert prompt_ids(SimpleNamespace(ids="1 2 3", tokenizer="missing")) == [1, 2, 3]


def test_pinned_fpu_trace_contract():
    path = Path(__file__).parent / "data/v0-tiny-hybrid-fpu-trace.txt"
    points, order = parse(path)
    assert len(order) == len(points) == 87
    assert points[(0, 0, "ffn_act")].checksum == "15069f09"
    assert points[(0, 0, "ffn_act")].dtype == "f32"
    assert points[(0, 0, "ffn_act")].count == 512
    assert [(p, l) for p, l, tag in order if tag == "argmax"] == [(2, -1)]
    assert points[(2, -1, "argmax")].checksum == "c653a7ea"  # FNV-1a of uint32 token 143.

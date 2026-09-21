# SPDX-License-Identifier: BSD-2-Clause
"""Trace validation must fail closed, including absent/malformed board output."""

from types import SimpleNamespace

import pytest

from tools.card_vs_mirror import cmd_compare, parse


EMBED = "DBG pos=0 l=-1 embed e=-12 1234abcd amax=42 v0..3=1,2,3,4\n"
ARGMAX = "DBG pos=0 l=-1 argmax e=0 00000003\n"


def compare(tmp_path, mirror, card):
    a = SimpleNamespace(mirror=tmp_path / "mirror", card=tmp_path / "card", verbose=True)
    a.mirror.write_text(mirror)
    a.card.write_text(card)
    return cmd_compare(a)


def test_matching_trace_with_chatter(tmp_path):
    assert compare(tmp_path, EMBED + ARGMAX, "boot chatter\n" + EMBED + ARGMAX) == 0


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

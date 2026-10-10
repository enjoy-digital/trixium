#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""model/tinygen.py: the tiny random models, including the K-segmented variants (the 27B's FFN
down projection path on a tiny model)."""

import pytest

from model import layout
from model import tinygen
from model.bons import BONSReader


@pytest.mark.parametrize("cfg", [tinygen.BONSAI_TINY, tinygen.BONSAI_TINY_HYBRID_1K])
def test_tinygen_kseg(tmp_path, cfg):
    img, _, _ = tinygen.generate(str(tmp_path), 2, cfg=cfg, kseg_max=256)
    entries = BONSReader(img).entries
    assert any(e["flags"] & layout.FLAG_KSEG for e in entries)
    # The token embedding stays whole (lookup, tied LM head).
    emb = [e for e in entries if e["role"] == layout.ROLE_TOK_EMB]
    assert len(emb) == 1 and not emb[0]["flags"] & layout.FLAG_KSEG

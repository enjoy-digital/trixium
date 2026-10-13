#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Per-board host-side settings: the converter's windows and channel counts, and the deploy
sequence of each board (dry run: every command printed, none run)."""

import io
import contextlib

from model.layout import HBM_WEIGHT_BASE

# Convert ------------------------------------------------------------------------------------------

def test_ypcb_windows():
    """The YPCB-00338's images: 4 channels, two 704 MB windows per 2 GB DDR3 controller at a 1 GB
    stride, the second one short of the 32 KB bank skew (trixium_ypcb.py's constants)."""
    from software.cli import BOARD_WINDOWS, BOARD_CHANNELS
    stride, reserve, suffix = BOARD_WINDOWS["ypcb"]
    assert (stride, reserve, suffix) == (0x4000_0000, 0x8000, "-ypcb")
    assert BOARD_CHANNELS == {"fk33": 16, "xcu1525": 16, "ypcb": 4}
    chan_mod, chan_skew = 2, 0x8000
    assert reserve == (chan_mod - 1)*chan_skew
    # The last window of a controller ends at its 2 GB.
    assert HBM_WEIGHT_BASE + chan_mod*stride - (HBM_WEIGHT_BASE % stride) <= 2 << 30


def test_convert_default_channels(monkeypatch):
    """--channels follows the board when not given (16 for fk33/xcu1525, 4 for ypcb)."""
    from software import cli
    import model.convert
    seen = {}
    def fake_convert(gguf, channels, out, *args, **kwargs):
        seen[kwargs["chan_stride"], kwargs["window_reserve"]] = (channels, out)
    monkeypatch.setattr(model.convert, "convert", fake_convert)
    for board in ["fk33", "xcu1525", "ypcb"]:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["convert", "--board", board, "--out", f"/tmp/x-{board}.img", "Ternary-Bonsai-1.7B-PQ2_0.gguf"])
    assert seen[(None, 0)][0]                == 16
    assert seen[(0x4000_0000, 0x18000)][0]   == 16
    assert seen[(0x4000_0000, 0x8000)][0]    == 4

# Deploy -------------------------------------------------------------------------------------------

def test_ypcb_deploy_dry_run(monkeypatch):
    monkeypatch.setenv("TRIXIUM_YPCB_HOST", "h")
    monkeypatch.setenv("TRIXIUM_YPCB_USER", "lab")
    from software.lab import deploy
    for model, image in [("1p7b", "bonsai-1p7b-n4-ypcb.img"), ("8b", "bonsai-8b-n4-ypcb.img")]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = deploy.deploy("ypcb", "ypcb", model, dry_run=True)
        out = out.getvalue()
        assert rc == 0
        assert "ypcb_00338_1p1.bit" in out and "ypcb_00338_1p1_timing.rpt" in out
        assert "--board ypcb" in out and "--strict" not in out   # The envelope gate.
        assert "reconfigure --board ypcb prog/csr.bit" in out
        assert "boot " in out and "--board ypcb" in out
        assert image in out

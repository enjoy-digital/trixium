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
    """The YPCB-00338's images: each 2 GB DDR3 controller above the weight base split evenly
    between its windows (864 MB at 4 channels, 432 MB at 8), the last one short of the bank skew
    (64 KB / windows per extra window), as trixium_ypcb.py lays them out."""
    from software.cli import board_windows, BOARD_CHANNELS
    assert BOARD_CHANNELS == {"fk33": 16, "xcu1525": 16, "ypcb": 8, "ti375": 4}
    for channels, stride_mb in [(4, 864), (8, 432)]:
        stride, window, reserve = board_windows("ypcb", channels)
        chan_mod, chan_skew = channels//2, 0x10000//(channels//2)
        assert stride == window == stride_mb << 20
        assert reserve == (chan_mod - 1)*chan_skew
        # The last window of a controller ends at its 2 GB.
        last = HBM_WEIGHT_BASE + (chan_mod - 1)*(stride + chan_skew)
        assert last + window - reserve == HBM_WEIGHT_BASE + chan_mod*stride <= 2 << 30


def test_convert_default_channels(monkeypatch):
    """--channels and --trit5 follow the board when not given (16 channels for fk33/xcu1525, 8
    with 5-trit codes for ypcb); --no-trit5 opts out."""
    from software import cli
    import model.convert
    seen = {}
    def fake_convert(gguf, channels, out, *args, **kwargs):
        seen[kwargs["chan_stride"], kwargs["window"], kwargs["window_reserve"]] = (channels, out, kwargs["trit5"])
    monkeypatch.setattr(model.convert, "convert", fake_convert)
    for board in ["fk33", "xcu1525", "ypcb"]:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["convert", "--board", board, "--out", f"/tmp/x-{board}.img", "Ternary-Bonsai-1.7B-PQ2_0.gguf"])
    assert seen[(None, None, 0)]                  == (16, "/tmp/x-fk33.img", False)
    assert seen[(0x4000_0000, None, 0x18000)]     == (16, "/tmp/x-xcu1525.img", False)
    assert seen[(432 << 20, 432 << 20, 0xC000)]   == (8, "/tmp/x-ypcb.img", True)
    seen.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        cli.main(["convert", "--board", "ypcb", "--channels", "4", "--no-trit5", "--out", "/tmp/x.img",
                  "Ternary-Bonsai-1.7B-PQ2_0.gguf"])
    assert seen[(864 << 20, 864 << 20, 0x8000)]   == (4, "/tmp/x.img", False)

# Deploy -------------------------------------------------------------------------------------------

def test_ypcb_deploy_dry_run(monkeypatch):
    monkeypatch.setenv("TRIXIUM_YPCB_HOST", "h")
    monkeypatch.setenv("TRIXIUM_YPCB_USER", "lab")
    from software.lab import deploy
    # The image follows the build's channel count (soc.h); without a build tree, the default 8.
    monkeypatch.setattr(deploy, "soc_constant", lambda soc_h, name, default=None: default)
    for model, image in [("1p7b", "bonsai-1p7b-n8-ypcb.img"), ("8b", "bonsai-8b-n8-ypcb.img")]:
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

# SPDX-License-Identifier: BSD-2-Clause
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from tools.artifact_manifest import REQUIRED, create_manifest, fingerprint


def test_hash_bytes_and_required_roles(tmp_path):
    path = tmp_path / "fixture.bin"
    data = b"test-model\x00\xff" * 100
    path.write_bytes(data)
    got = fingerprint(path)
    assert got["size_bytes"] == len(data)
    assert got["sha256"] == hashlib.sha256(data).hexdigest()
    m = create_manifest([f"{role}={path}" for role in REQUIRED], "revision", "fixture")
    assert set(m["artifacts"]) == REQUIRED
    assert m["qualification"].startswith("unqualified")


@pytest.mark.parametrize("artifacts", [[], ["firmware=file"], ["=path"], ["firmware="],
                                      ["firmware=file", "firmware=other"]])
def test_incomplete_and_duplicate_artifacts_rejected(artifacts):
    with pytest.raises(ValueError):
        create_manifest(artifacts, "revision", "fixture")


def test_manifest_never_overwrites(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"x")
    out = tmp_path / "existing.json"
    out.write_text("KEEP")
    script = Path(__file__).resolve().parents[1] / "tools/artifact_manifest.py"
    args = [sys.executable, str(script), "--revision", "test", "--label", "test", "-o", str(out)]
    for role in REQUIRED:
        args.extend(["--artifact", f"{role}={path}"])
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode == 1 and out.read_text() == "KEEP"

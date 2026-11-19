"""mirror/final/prereg.py: the line-ending-normalised hash, the recorded check and its refusal on a change."""
import json
import subprocess
from pathlib import Path

import pytest

from mirror.final import prereg as P


def test_hash_ignores_crlf(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"one\ntwo\n")
    b.write_bytes(b"one\r\ntwo\r\n")
    assert P.sha256_of(a) == P.sha256_of(b)


def test_record_then_change_is_refused(tmp_path):
    root = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    f = root / P.PREREG_FILE
    f.write_text("| ID | bar |\n|---|---|\n| X | >= 1 |\n", encoding="utf-8")
    subprocess.run(["git", "add", P.PREREG_FILE], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "prereg"], cwd=root, check=True)
    rec = P.record_prereg(root)
    assert json.loads((root / P.RECORD_FILE).read_text())["sha256"] == rec["sha256"]
    assert P.prereg_status(root)["ok"]
    assert P.require_prereg(root=root)["ok"]
    f.write_text("| ID | bar |\n|---|---|\n| X | >= 2 |\n", encoding="utf-8")       # a threshold changed
    st = P.prereg_status(root)
    assert not st["ok"] and "changed" in st["reason"]
    with pytest.raises(P.PreregError):
        P.require_prereg(allow_missing=False, root=root)
    assert P.require_prereg(allow_missing=True, root=root)["allowed_without_check"]

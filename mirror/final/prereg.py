"""Pre-registration guard and run metadata for the final phase (FINAL_PHASE_PLAN.md v2, Sections 7.15 and 10).

PREREG_FINAL_PHASE.md is committed on its own; its sha256 and commit are then recorded in
results/final/prereg_record.json (and quoted in results/STATUS.md). Every Phase G script calls
`require_prereg()` first: it refuses to run when the record is missing or when the file's current sha256
differs from the recorded one (the thresholds must never change after the commit).

CPU tests and laptop dry runs set MIRROR_FINAL_ALLOW_NO_PREREG=1 (or pass allow_missing=True); the result
then says so, and `final_meta` records it, so no GPU result can carry an unchecked pre-registration silently.

`final_meta(...)` returns the meta.json dictionary every final-phase run writes: commit, pre-registration
hash, config, seeds, model revisions, GPU, wall time and peak memory (Section 10).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from ..utils import REPO_ROOT, env_info, git_info

PREREG_FILE = "PREREG_FINAL_PHASE.md"
RECORD_FILE = Path("results") / "final" / "prereg_record.json"
ALLOW_ENV = "MIRROR_FINAL_ALLOW_NO_PREREG"


class PreregError(RuntimeError):
    """The pre-registration is missing, unrecorded or changed since its commit."""


def sha256_of(path) -> str:
    """sha256 of the file's text with line endings normalised to LF (CRLF -> LF), so the laptop's Windows checkout
    (core.autocrlf) and the VM's Linux copy of the same committed file give the same hash; it equals the sha256 of
    the committed blob, `git show <commit>:PREREG_FINAL_PHASE.md | sha256sum`."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def last_commit_of(rel_path: str, root: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel_path], cwd=root,
                             capture_output=True, text=True, timeout=10)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def prereg_status(root: Path = REPO_ROOT) -> dict:
    """{"file", "sha256", "recorded_sha256", "recorded_commit", "commit", "ok", "reason"} for the final phase."""
    root = Path(root)
    path = root / PREREG_FILE
    rec_path = root / RECORD_FILE
    st = {"file": PREREG_FILE, "sha256": None, "recorded_sha256": None, "recorded_commit": None,
          "commit": None, "ok": False, "reason": ""}
    if not path.exists():
        st["reason"] = f"{PREREG_FILE} does not exist"
        return st
    st["sha256"] = sha256_of(path)
    st["commit"] = last_commit_of(PREREG_FILE, root)
    if not rec_path.exists():
        st["reason"] = f"{RECORD_FILE.as_posix()} does not exist (the pre-registration commit is not recorded)"
        return st
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        st["reason"] = f"{RECORD_FILE.as_posix()} is unreadable: {e}"
        return st
    st["recorded_sha256"] = rec.get("sha256")
    st["recorded_commit"] = rec.get("commit")
    if rec.get("sha256") != st["sha256"]:
        st["reason"] = (f"{PREREG_FILE} changed since its pre-registration commit "
                        f"(sha256 {st['sha256'][:12]} against recorded {str(rec.get('sha256'))[:12]})")
        return st
    st["ok"] = True
    st["reason"] = "unchanged since the recorded commit"
    return st


def require_prereg(allow_missing: bool | None = None, root: Path = REPO_ROOT) -> dict:
    """Raise PreregError unless PREREG_FINAL_PHASE.md is recorded and unchanged.

    allow_missing (default: the MIRROR_FINAL_ALLOW_NO_PREREG environment variable) lets CPU tests and dry runs
    continue; the returned status then has "allowed_without_check": True."""
    st = prereg_status(root)
    if st["ok"]:
        return st
    if allow_missing is None:
        allow_missing = os.environ.get(ALLOW_ENV, "") not in ("", "0", "false", "False")
    if allow_missing:
        return {**st, "allowed_without_check": True}
    raise PreregError(f"refusing to run: {st['reason']}. Commit {PREREG_FILE} on its own and record it with "
                      f"scripts/fp_record_prereg.py; set {ALLOW_ENV}=1 only for CPU tests and dry runs.")


def record_prereg(root: Path = REPO_ROOT) -> dict:
    """Write results/final/prereg_record.json from the committed PREREG_FINAL_PHASE.md (run once, right after
    the pre-registration commit). Refuses if the file has uncommitted changes."""
    root = Path(root)
    path = root / PREREG_FILE
    if not path.exists():
        raise PreregError(f"{PREREG_FILE} does not exist")
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG_FILE], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # pragma: no cover - git missing
        raise PreregError(f"git is needed to record the pre-registration: {e}")
    if dirty:
        raise PreregError(f"{PREREG_FILE} has uncommitted changes; commit it on its own first")
    commit = last_commit_of(PREREG_FILE, root)
    if commit is None:
        raise PreregError(f"{PREREG_FILE} is not committed")
    rec = {"file": PREREG_FILE, "sha256": sha256_of(path), "commit": commit,
           "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = root / RECORD_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except (ValueError, TypeError):
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def final_meta(*, experiment: str, config: dict | None = None, seeds: dict | None = None,
               models: dict | None = None, device=None, wall_seconds: float | None = None,
               peak_memory_gb: float | None = None, extra: dict | None = None) -> dict:
    """meta.json content for a final-phase run (Section 10): commit, pre-registration hash, config, seeds,
    model revisions, GPU, wall time and peak memory."""
    import torch

    dev = torch.device(device) if device is not None else torch.device("cpu")
    if peak_memory_gb is None and dev.type == "cuda" and torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    st = prereg_status()
    return {
        "experiment": experiment,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "prereg": {k: st.get(k) for k in ("file", "sha256", "recorded_sha256", "recorded_commit", "ok", "reason")},
        "config": _jsonable(config or {}),
        "seeds": _jsonable(seeds or {}),
        "models": _jsonable(models or {}),
        "env": env_info(dev),
        "wall_seconds": None if wall_seconds is None else round(float(wall_seconds), 2),
        "peak_memory_gb": None if peak_memory_gb is None else round(float(peak_memory_gb), 3),
        "extra": _jsonable(extra or {}),
    }


def write_final_meta(out_dir, name: str = "meta.json", **kwargs) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(final_meta(**kwargs), indent=2), encoding="utf-8")
    return path

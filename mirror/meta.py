"""meta.json for every new-phase run (PROJECT_NEXT_PHASE.md, Section 12).

Every script of the new phase writes `<out>/meta.json` through `write_meta`, with:
    experiment     ID such as "MUST-04"
    git            commit hash and dirtiness of the code that ran (mirror.utils.git_info)
    prereg         PREREG_NEXT_PHASE.md: sha256 of its content and the last commit that touched it
    config         the full argument / config dictionary of the run
    seeds          every seed that affects the result (fit seed, training seed, null seeds, ...)
    models         model IDs and revisions (configs/models/*.yaml entries, or hub IDs)
    env            python, torch, numpy, device, GPU name, CUDA version
    wall_seconds   wall-clock time of the run
    extra          anything else the script wants on record (data split sizes, budgets, ...)

Usage:
    t0 = time.time()
    ...
    write_meta(out_dir, experiment="MUST-04", config=vars(args), seeds={"fit": 0},
               models={"dinov2_vitb14_reg": {...}}, device=device, wall_seconds=time.time() - t0)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .utils import REPO_ROOT, env_info, git_info

PREREG_FILE = "PREREG_NEXT_PHASE.md"


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


def prereg_info(root: Path = REPO_ROOT) -> dict:
    """sha256 of PREREG_NEXT_PHASE.md and the last commit that changed it (None if absent or no git)."""
    path = Path(root) / PREREG_FILE
    if not path.exists():
        return {"file": PREREG_FILE, "sha256": None, "commit": None}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = None
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", PREREG_FILE], cwd=root,
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        commit = None
    return {"file": PREREG_FILE, "sha256": sha, "commit": commit}


def run_meta(*, experiment: str, config: dict | None = None, seeds: dict | None = None,
             models: dict | None = None, device=None, wall_seconds: float | None = None,
             extra: dict | None = None) -> dict:
    import torch

    dev = torch.device(device) if device is not None else torch.device("cpu")
    return {
        "experiment": experiment,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "prereg": prereg_info(),
        "config": _jsonable(config or {}),
        "seeds": _jsonable(seeds or {}),
        "models": _jsonable(models or {}),
        "env": env_info(dev),
        "wall_seconds": None if wall_seconds is None else round(float(wall_seconds), 2),
        "extra": _jsonable(extra or {}),
    }


def write_meta(out_dir, name: str = "meta.json", **kwargs) -> Path:
    """Write run_meta(**kwargs) to <out_dir>/<name>; returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(run_meta(**kwargs), indent=2))
    return path

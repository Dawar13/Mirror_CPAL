"""Shared helpers: config loading, seeding, device setup, provenance and logging."""
from __future__ import annotations

import contextlib
import copy
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, override: dict | None) -> dict:
    """Return a copy of `base` with `override` merged in recursively."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, tiny: bool = False) -> dict:
    """Load a yaml config. With `tiny=True` the `tiny:` section overrides the main settings."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tiny_override = cfg.pop("tiny", {}) or {}
    if tiny:
        cfg = deep_update(cfg, tiny_override)
    cfg["tiny_mode"] = bool(tiny)
    cfg["config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch(cfg: dict) -> torch.device:
    """Pick the device, cap GPU memory (shared VM) and fix matmul precision."""
    comp = cfg["compute"]
    dev = str(comp.get("device", "cuda"))
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config asks for CUDA but no GPU is visible. Use --tiny for CPU runs.")
        torch.cuda.set_per_process_memory_fraction(float(comp.get("gpu_memory_fraction", 0.2)))
    tf32 = bool(comp.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if comp.get("cpu_threads"):
        torch.set_num_threads(int(comp["cpu_threads"]))
    return torch.device(dev)


def git_info(root: Path = REPO_ROOT) -> dict:
    """Commit hash and whether tracked files have uncommitted changes."""

    def run(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD")
    if commit:
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"git_hash": commit, "git_dirty": dirty}
    # No git binary (fresh VM): read the commit straight from .git; dirtiness is then unknown.
    return {"git_hash": _read_git_head(root) or "unknown", "git_dirty": None}


def _read_git_head(root: Path) -> str:
    git_dir = Path(root) / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        if (git_dir / ref).exists():
            return (git_dir / ref).read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
    except OSError:
        pass
    return ""


def env_info(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class StageTimer:
    """Records wall-clock seconds per named stage."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.stages: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        log(f"start: {name}")
        t = time.time()
        try:
            yield
        finally:
            self.stages[name] = round(time.time() - t, 2)
            log(f"done:  {name} ({self.stages[name]:.1f} s)")

    def total(self) -> float:
        return round(time.time() - self.t0, 2)

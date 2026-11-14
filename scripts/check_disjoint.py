#!/usr/bin/env python
"""Data hygiene check (PROJECT_NEXT_PHASE.md, Sections 4.1 and 13): the ImageNet-train roles are
disjoint by image ID (train shard file name + row in that shard), and validation is a separate split.

    python scripts/check_disjoint.py                    # cache/ -> results/data_check/disjoint.json
    python scripts/check_disjoint.py --allow-missing-msel

Reads cache/train_index.npz with cache/train_meta.json (roles fit0, fit1, fit2, bank, healval, heal;
shard index -> file name through "shards_used"), cache/msel_index.npz (role msel; shard index -> file
name through "msel_shards") and cache/meta.json (validation source files). When
cache/train_shard_labels.json is present, every stored label is also compared with the label in its
shard. Records role sizes, per-class counts, every pairwise overlap, the selection fingerprint of
scripts/build_train_cache.py and the git commit. Exit code 1 when any check fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, git_info, log  # noqa: E402

TRAIN_ROLES = [("fit0", "src224"), ("fit1", "src224"), ("fit2", "src224"), ("bank", "src224"),
               ("healval", "src256"), ("heal", "src256")]
EXPECTED = {"fit0": 2000, "fit1": 2000, "fit2": 2000, "bank": 100000, "healval": 5000, "heal": 100000,
            "msel": 10000}


def _btc():
    key = "_build_train_cache_for_check"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / "build_train_cache.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def train_roles(index: dict, shards_used) -> dict:
    """{role: (ids [(shard name, row)], labels)} for the six train roles of train_index.npz."""
    out = {}
    for role, src in TRAIN_ROLES:
        rows = np.asarray(index[f"{role}_rows"], dtype=np.int64)
        pairs = np.asarray(index[src], dtype=np.int64)[rows]
        ids = [(shards_used[int(si)], int(r)) for si, r in pairs]
        out[role] = (ids, np.asarray(index[f"{role}_labels"], dtype=np.int64))
    return out


def msel_role(msel: dict) -> tuple:
    shards = [str(s) for s in msel["msel_shards"]]
    src = np.asarray(msel["msel_source"], dtype=np.int64).reshape(-1, 2)
    return [(shards[int(si)], int(r)) for si, r in src], np.asarray(msel["msel_labels"], dtype=np.int64)


def check_roles(roles: dict) -> dict:
    """Pure check on {role: [(shard name, row), ...]}: duplicates inside each role, overlaps between
    every pair of roles. Returns {"ok", "roles": {role: {n, unique}}, "overlaps": {"a|b": count}}."""
    sets = {k: set(v) for k, v in roles.items()}
    info = {k: {"n": len(v), "unique": len(sets[k])} for k, v in roles.items()}
    overlaps = {f"{a}|{b}": len(sets[a] & sets[b]) for a, b in combinations(sorted(roles), 2)}
    ok = all(i["n"] == i["unique"] for i in info.values()) and not any(overlaps.values())
    return {"ok": bool(ok), "roles": info, "overlaps": overlaps}


def check_labels(roles: dict, labels: dict, shard_labels: dict) -> dict:
    """Stored label against the label in the shard, per role (shards missing from the cache are skipped)."""
    out = {}
    for role, ids in roles.items():
        bad = checked = 0
        for (shard, row), lab in zip(ids, labels[role]):
            if shard in shard_labels:
                checked += 1
                bad += int(shard_labels[shard][row] != int(lab))
        out[role] = {"checked": checked, "mismatches": bad}
    return out


def run(cache_dir, out_path, allow_missing_msel: bool = False) -> dict:
    cache = Path(cache_dir)
    t0 = time.time()
    problems = []
    meta = json.loads((cache / "train_meta.json").read_text())
    shards_used = list(meta["shards_used"])
    with np.load(cache / "train_index.npz") as z:
        index = {k: z[k] for k in z.files}
    parts = train_roles(index, shards_used)
    msel_path = cache / "msel_index.npz"
    msel_info = None
    if msel_path.exists():
        with np.load(msel_path) as z:
            msel = {k: z[k] for k in z.files}
        parts["msel"] = msel_role(msel)
        msel_info = {"shards": [str(s) for s in msel["msel_shards"]],
                     "shards_also_used_by_other_roles": sorted(set(map(str, msel["msel_shards"])) & set(shards_used))}
        if msel_info["shards_also_used_by_other_roles"]:
            problems.append("msel draws from shards that other roles use")
    elif not allow_missing_msel:
        problems.append(f"{msel_path} missing (run scripts/build_train_cache.py --msel)")
    ids = {k: v[0] for k, v in parts.items()}
    labels = {k: v[1] for k, v in parts.items()}
    result = check_roles(ids)
    if not result["ok"]:
        problems.append("roles overlap or contain duplicates")
    sizes = {}
    for role, labs in labels.items():
        per = np.bincount(labs, minlength=1000)
        sizes[role] = {"n": int(len(labs)), "expected": EXPECTED.get(role), "per_class_min": int(per.min()),
                       "per_class_max": int(per.max()), "classes": int((per > 0).sum())}
        if EXPECTED.get(role) is not None and len(labs) != EXPECTED[role]:
            problems.append(f"{role}: {len(labs)} images, expected {EXPECTED[role]}")
    label_check = None
    lab_path = cache / "train_shard_labels.json"
    if lab_path.exists():
        label_check = check_labels(ids, labels, json.loads(lab_path.read_text()))
        if any(v["mismatches"] for v in label_check.values()):
            problems.append("stored labels differ from the shard labels")
    val = None
    if (cache / "meta.json").exists():
        vmeta = json.loads((cache / "meta.json").read_text())
        vsrc = [str(s) for s in vmeta.get("source_files", [])]
        train_names = {Path(s).name for s in shards_used} | {Path(s).name for s in (msel_info or {}).get("shards", [])}
        shared = sorted(set(vsrc) & train_names)
        val = {"n_images": vmeta.get("n_images"), "source_files": vsrc, "source_files_shared_with_train": shared,
               "note": "validation comes from the validation parquet shards, a separate split; used for reporting only"}
        if shared or any(not s.startswith("validation-") for s in vsrc):
            problems.append("validation source files look like train shards")
    else:
        problems.append(f"{cache / 'meta.json'} missing (validation cache not built)")
    report = {
        "ok": not problems, "problems": problems, "id": "train shard file name + row in that shard",
        "cache_dir": str(cache), "roles": sizes, "pairwise_overlaps": result["overlaps"],
        "duplicates_within_role": {k: v["n"] - v["unique"] for k, v in result["roles"].items()},
        "label_check": label_check, "msel": msel_info, "validation": val,
        "train_meta": {k: meta.get(k) for k in ("seed", "repo", "revision", "selection_sha256", "n224", "n256")},
        "shards_used": len(shards_used),
        "selection_sha256_recomputed": _btc().selection_fingerprint(index, shards_used),
        "git": git_info(), "seconds": round(time.time() - t0, 1),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stored = meta.get("selection_sha256")
    if stored and stored != report["selection_sha256_recomputed"]:
        problems.append("train_index.npz does not match the selection fingerprint in train_meta.json")
        report["ok"] = False
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "data_check" / "disjoint.json"))
    p.add_argument("--allow-missing-msel", action="store_true")
    a = p.parse_args(argv)
    rep = run(a.cache_dir, a.out, a.allow_missing_msel)
    for role, s in rep["roles"].items():
        log(f"{role:8s} n={s['n']:6d} per class {s['per_class_min']}..{s['per_class_max']}")
    worst = max(rep["pairwise_overlaps"].values()) if rep["pairwise_overlaps"] else 0
    log(f"largest pairwise overlap: {worst}; selection sha256 {rep['selection_sha256_recomputed'][:16]}")
    log(("PASS" if rep["ok"] else "FAIL: " + "; ".join(rep["problems"])) + f" -> {a.out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

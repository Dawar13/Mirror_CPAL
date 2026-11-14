#!/usr/bin/env python
"""X2 data: ImageNet-1k TRAIN subsets for the paper protocol (PAPER_EXPERIMENTS.md, Section 1;
PROJECT_NEXT_PHASE.md, Section 4.1).

Per class, from train images only (fixed seed 0, disjoint roles):
    fit0, fit1, fit2   2 images each   (3 fit seeds x 2,000 images, operators and twins)
    bank               100 images      (k-NN memory bank, 100k, also the linear-probe training set)
    healval            5 images        (5k, learning-rate selection for healing; never validation)
    heal               100 images      (100k, healing distillation)

Shards are read in file order; only the label column is fetched first (remote, a few KB per shard),
then only the shards that contain the chosen images are downloaded (under $HF_HOME). The shards
read until every class has 211 images are the "used" shards (86 of 294 on the A100 and the 5090).

Writes to cache/:
    train224_uint8.npy   fit0..2 + bank, resize 256 (bicubic) + centre crop 224   [106,000]
    heal256_uint8.npy    heal + healval, resize 256 (bicubic) + centre crop 256    [105,000]
    train_index.npz      per role: row indices into the right memmap, labels, source (shard, row)
    train_meta.json

--msel adds the method-selection set (PROJECT_NEXT_PHASE.md, Section 4.1; read by
mirror.protocol.Protocol.msel_loader): 10 images per class (10k) from the train shards AFTER the used
ones (file order, until every class has 10), chosen with its own generator (seed 1), so it is
disjoint from every other role by construction and the selection of the other roles is untouched:
    msel224_uint8.npy    resize 256 (bicubic) + centre crop 224, as train224           [10,000]
    msel_index.npz       msel_rows, msel_labels (int64), msel_source (int64 [N, 2]: index into
                         msel_shards, row in that shard), msel_shards (shard file names)
    msel_meta.json

    python scripts/build_train_cache.py --workers 14          # fit, bank, healval, heal
    python scripts/build_train_cache.py --msel --workers 14   # method-selection set

The selection is a pure function of the shard labels and the seeds (select_used_shards, select_roles,
select_msel); tests/test_build_train_cache.py checks it against a frozen copy of the original code.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, git_info, log  # noqa: E402

ROLES_224 = [("fit0", 2), ("fit1", 2), ("fit2", 2), ("bank", 100)]
ROLES_256 = [("healval", 5), ("heal", 100)]
REPO = "ILSVRC/imagenet-1k"
# Dataset revision, read from the Hugging Face API on 2026-11-14 (last modified 2025-09-17, so the
# same files the A100 and 5090 caches were built from on 2026-11-08 and 2026-11-09 through "main").
REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
N_CLASSES = 1000
MSEL_PER_CLASS = 10
MSEL_SEED = 1


# ----------------------------------------------------------------------------------------------
# Selection (pure functions of the shard labels and the seeds)
# ----------------------------------------------------------------------------------------------
def need_per_class(roles=None) -> int:
    roles = ROLES_224 + ROLES_256 if roles is None else roles
    return sum(n for _, n in roles)


def select_used_shards(files, labels_of, need: int, n_classes: int = N_CLASSES):
    """Shards in file order until every class has at least `need` images.

    labels_of(f) returns the label list of shard f. Returns (used, counts); `used` is a prefix of
    `files`. Same loop as the original script (it stops right after the shard that completes it)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    used = []
    for f in files:
        labs = labels_of(f)
        used.append(f)
        counts += np.bincount(np.asarray(labs), minlength=n_classes)
        if counts.min() >= need:
            break
    return used, counts


def select_roles(used, shard_labels, seed: int, roles=None, n_classes: int = N_CLASSES) -> dict:
    """Deterministic choice: per class, shuffle its (shard, row) list with the seed, then cut roles.

    Returns {role: [(shard index into `used`, row, class), ...]} in class order. The generator calls
    (one permutation per class, classes in order) are exactly those of the original script."""
    roles = ROLES_224 + ROLES_256 if roles is None else roles
    rng = np.random.default_rng(seed)
    per_class = {c: [] for c in range(n_classes)}
    for si, f in enumerate(used):
        for r, lab in enumerate(shard_labels[f]):
            per_class[lab].append((si, r))
    out = {name: [] for name, _ in roles}
    for c in range(n_classes):
        items = per_class[c]
        order = rng.permutation(len(items))
        pos = 0
        for name, n in roles:
            out[name] += [(items[i][0], items[i][1], c) for i in order[pos:pos + n]]
            pos += n
    return out


def role_groups(roles: dict) -> tuple[list, list]:
    """(group224, group256): the roles concatenated in memmap order."""
    group224 = [x for name, _ in ROLES_224 for x in roles[name]]
    group256 = [x for name, _ in ROLES_256 for x in roles[name]]
    return group224, group256


def build_index(group224, group256) -> dict:
    """train_index.npz contents: role -> rows in its memmap, labels, and the (shard, row) sources."""
    idx, start = {}, 0
    for name, n in ROLES_224:
        k = n * N_CLASSES
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group224[start:start + k]])
        start += k
    start = 0
    for name, n in ROLES_256:
        k = n * N_CLASSES
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group256[start:start + k]])
        start += k
    idx["src224"] = np.array([(si, r) for si, r, _ in group224])
    idx["src256"] = np.array([(si, r) for si, r, _ in group256])
    return idx


def selection_fingerprint(index: dict, shards_used) -> str:
    """sha256 over the used shard names, both source arrays and every role's labels (int64).

    The same value from two machines means the same images in the same roles and order."""
    h = hashlib.sha256()
    h.update("\n".join(shards_used).encode())
    keys = ["src224", "src256"] + [f"{name}_labels" for name, _ in ROLES_224 + ROLES_256]
    for k in keys:
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(index[k], dtype=np.int64)).tobytes())
    return h.hexdigest()


def select_msel(files, n_used: int, labels_of, per_class: int = MSEL_PER_CLASS, seed: int = MSEL_SEED,
                n_classes: int = N_CLASSES):
    """Method-selection images from the shards after the first `n_used` (file order).

    Shards are added until every class has at least `per_class` images; then, with its own generator
    (default_rng(seed)), each class's (shard, row) list is permuted and the first `per_class` taken.
    Returns (msel_shards, picks) with picks = [(index into msel_shards, row, class), ...] in class order."""
    pool, counts = [], np.zeros(n_classes, dtype=np.int64)
    for f in files[n_used:]:
        pool.append(f)
        counts += np.bincount(np.asarray(labels_of(f)), minlength=n_classes)
        if counts.min() >= per_class:
            break
    if counts.min() < per_class:
        raise RuntimeError(f"not enough images per class after the first {n_used} shards "
                           f"(min {int(counts.min())}, need {per_class})")
    rng = np.random.default_rng(seed)
    per = {c: [] for c in range(n_classes)}
    for si, f in enumerate(pool):
        for r, lab in enumerate(labels_of(f)):
            per[lab].append((si, r))
    picks = []
    for c in range(n_classes):
        items = per[c]
        order = rng.permutation(len(items))
        picks += [(items[i][0], items[i][1], c) for i in order[:per_class]]
    return pool, picks


# ----------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------
def _decode(args):
    data, resize, crop = args
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(io.BytesIO(data)).convert("RGB")
    tf = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(crop)])
    return np.asarray(tf(img), dtype=np.uint8)


def decode_group(group, shard_names, crop: int, final: Path, local_path, workers: int, desc: str = "") -> None:
    """Decode the chosen rows into `final` ([len(group), crop, crop, 3] uint8), shard by shard.

    group: [(shard index into shard_names, row, class), ...] in memmap order; local_path(name) returns
    a local parquet file for a shard name. Written to a .partial file, then renamed."""
    import pyarrow.parquet as pq
    from tqdm import tqdm

    final = Path(final)
    if final.exists():
        log(f"{final.name} exists, skipping")
        return
    partial = final.with_name(final.name + ".partial.npy")
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(len(group), crop, crop, 3))
    by_shard = {}
    for row, (si, r, _c) in enumerate(group):
        by_shard.setdefault(si, []).append((r, row))
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(workers)
    try:
        for si in tqdm(sorted(by_shard), desc=desc or final.name):
            local = local_path(shard_names[si])
            rows = sorted(by_shard[si])
            want = {r for r, _ in rows}
            tbl = pq.read_table(local, columns=["image"])
            col = tbl.column("image")
            jobs = [(col[r].as_py()["bytes"], 256, crop) for r, _ in rows]
            arrs = pool.map(_decode, jobs, chunksize=8) if pool is not None else [_decode(j) for j in jobs]
            for (r, row), arr in zip(rows, arrs):
                mm[row] = arr
            assert len(want) == len(rows)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    mm.flush()
    del mm
    os.replace(partial, final)


def msel_index(msel_shards, picks) -> dict:
    """msel_index.npz contents (read by mirror.protocol.Protocol.msel_loader)."""
    return {"msel_rows": np.arange(len(picks), dtype=np.int64),
            "msel_labels": np.array([c for _, _, c in picks], dtype=np.int64),
            "msel_source": np.array([(si, r) for si, r, _ in picks], dtype=np.int64).reshape(-1, 2),
            "msel_shards": np.array(list(msel_shards), dtype=str)}


def write_msel(cache: Path, msel_shards, picks, local_path, workers: int, meta_extra: dict | None = None) -> dict:
    """Decode the method-selection images and write msel224_uint8.npy, msel_index.npz, msel_meta.json."""
    cache = Path(cache)
    decode_group(picks, list(msel_shards), 224, cache / "msel224_uint8.npy", local_path, workers,
                 desc="msel224_uint8.npy")
    np.savez(cache / "msel_index.npz", **msel_index(msel_shards, picks))
    labels = np.array([c for _, _, c in picks], dtype=np.int64)
    per = np.bincount(labels) if len(labels) else np.zeros(1, dtype=np.int64)
    meta = {"complete": True, "n": len(picks), "per_class_min": int(per.min()), "per_class_max": int(per.max()),
            "classes": int(len(np.unique(labels))), "shards": list(msel_shards), "resize": 256, "crop": 224,
            "interpolation": "bicubic",
            "note": "train images only, from shards after those used by fit, bank, healval and heal; "
                    "used as k-NN queries against the bank to choose methods and hyperparameters"}
    meta.update(meta_extra or {})
    (cache / "msel_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------------------------
# Hub access (VM only)
# ----------------------------------------------------------------------------------------------
class ShardLabels:
    """Label lists per shard, fetched remotely (label column only) and cached in a JSON file."""

    def __init__(self, path: Path, repo: str, revision: str | None):
        from huggingface_hub import HfFileSystem

        self.path, self.repo, self.revision = Path(path), repo, revision
        self.fs = HfFileSystem()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __call__(self, f: str) -> list:
        if f not in self.data:
            import pyarrow.parquet as pq

            at = f"@{self.revision}" if self.revision else ""
            with self.fs.open(f"datasets/{self.repo}{at}/{f}", "rb") as fh:
                self.data[f] = pq.read_table(fh, columns=["label"]).column("label").to_pylist()
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data))
            os.replace(tmp, self.path)          # an interrupted write never leaves a broken cache
        return self.data[f]

    def __getitem__(self, f: str) -> list:
        return self(f)


def list_train_shards(repo: str, revision: str | None) -> list:
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
                  if f.startswith("data/train-") and f.endswith(".parquet"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--revision", default=REVISION, help="dataset revision ('main' for the latest)")
    p.add_argument("--msel", action="store_true", help="build the method-selection set only (see above)")
    p.add_argument("--msel-seed", type=int, default=MSEL_SEED)
    p.add_argument("--msel-per-class", type=int, default=MSEL_PER_CLASS)
    a = p.parse_args(argv)
    from huggingface_hub import hf_hub_download

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    need = need_per_class()
    files = list_train_shards(REPO, a.revision)
    log(f"{len(files)} train shards; need {need} images per class")

    def local_path(name: str) -> str:
        return hf_hub_download(REPO, name, repo_type="dataset", revision=a.revision)

    # 1. Labels only, shard by shard, until every class has enough images.
    labels = ShardLabels(cache / "train_shard_labels.json", REPO, a.revision)
    used, counts = select_used_shards(files, labels, need)
    log(f"using {len(used)} shards ({int(counts.sum())} images); min per class {int(counts.min())}")
    if counts.min() < need:
        raise RuntimeError("not enough images per class in the train shards")

    if a.msel:
        meta_path = cache / "train_meta.json"
        if meta_path.exists():
            prev = json.loads(meta_path.read_text()).get("shards_used")
            if prev is not None and list(prev) != used:
                raise RuntimeError("train_meta.json lists other used shards than this selection; "
                                   "refusing to build msel (it must avoid exactly those shards)")
        if (cache / "msel224_uint8.npy").exists() and (cache / "msel_index.npz").exists():
            log("msel224_uint8.npy and msel_index.npz exist, skipping")
            return
        msel_shards, picks = select_msel(files, len(used), labels, a.msel_per_class, a.msel_seed)
        log(f"msel: {len(picks)} images from {len(msel_shards)} shards after the first {len(used)} "
            f"({msel_shards[0]} .. {msel_shards[-1]}), seed {a.msel_seed}")
        write_msel(cache, msel_shards, picks, local_path, a.workers,
                   {"seed": a.msel_seed, "first_shard_index": len(used), "repo": REPO, "revision": a.revision,
                    "shards_used_by_other_roles": len(used), "git": git_info()})
        log(f"msel cache complete: {len(picks)} at 224")
        return

    # 2. Deterministic choice: per class, shuffle its (shard, row) list with the seed, then cut roles.
    roles = select_roles(used, labels, a.seed)
    group224, group256 = role_groups(roles)

    # 3. Download only the shards in use, then decode chosen rows shard by shard.
    decode_group(group224, used, 224, cache / "train224_uint8.npy", local_path, a.workers)
    decode_group(group256, used, 256, cache / "heal256_uint8.npy", local_path, a.workers)

    # 4. Index: role -> rows in its memmap.
    idx = build_index(group224, group256)
    np.savez(cache / "train_index.npz", **idx)
    meta = {"complete": True, "seed": a.seed, "shards_used": used, "roles_224": ROLES_224, "roles_256": ROLES_256,
            "n224": len(group224), "n256": len(group256), "interpolation": "bicubic",
            "note": "train images only; disjoint roles; validation is never used for fitting or healing",
            "repo": REPO, "revision": a.revision, "selection_sha256": selection_fingerprint(idx, used),
            "git": git_info()}
    (cache / "train_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"train cache complete: {len(group224)} at 224, {len(group256)} at 256; "
        f"selection sha256 {meta['selection_sha256'][:16]}")


if __name__ == "__main__":
    main()

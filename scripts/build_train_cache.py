#!/usr/bin/env python
"""X2 data: ImageNet-1k TRAIN subsets for the paper protocol (PAPER_EXPERIMENTS.md, Section 1).

Per class, from train images only (fixed seed 0, disjoint roles):
    fit0, fit1, fit2   2 images each   (3 fit seeds x 2,000 images, operators and twins)
    bank               100 images      (k-NN memory bank, 100k, also the linear-probe training set)
    healval            5 images        (5k, learning-rate selection for healing; never validation)
    heal               100 images      (100k, healing distillation)

Shards are read in file order; only the label column is fetched first (remote, a few KB per shard),
then only the shards that contain the chosen images are downloaded (under $HF_HOME).

Writes to cache/:
    train224_uint8.npy   fit0..2 + bank, resize 256 (bicubic) + centre crop 224   [106,000]
    heal256_uint8.npy    heal + healval, resize 256 (bicubic) + centre crop 256    [105,000]
    train_index.npz      per role: row indices into the right memmap, labels, source (shard, row)
    train_meta.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, log  # noqa: E402

ROLES_224 = [("fit0", 2), ("fit1", 2), ("fit2", 2), ("bank", 100)]
ROLES_256 = [("healval", 5), ("heal", 100)]
REPO = "ILSVRC/imagenet-1k"


def _decode(args):
    data, resize, crop = args
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(io.BytesIO(data)).convert("RGB")
    tf = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(crop)])
    return np.asarray(tf(img), dtype=np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    import multiprocessing as mp

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
    from tqdm import tqdm

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    need = sum(n for _, n in ROLES_224 + ROLES_256)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("data/train-") and f.endswith(".parquet"))
    log(f"{len(files)} train shards; need {need} images per class")

    # 1. Labels only, shard by shard, until every class has enough images.
    fs = HfFileSystem()
    lab_cache = cache / "train_shard_labels.json"
    shard_labels = json.loads(lab_cache.read_text()) if lab_cache.exists() else {}
    counts = np.zeros(1000, dtype=np.int64)
    used = []
    for f in files:
        if f not in shard_labels:
            with fs.open(f"datasets/{REPO}/{f}", "rb") as fh:
                shard_labels[f] = pq.read_table(fh, columns=["label"]).column("label").to_pylist()
            lab_cache.write_text(json.dumps(shard_labels))
        used.append(f)
        counts += np.bincount(np.asarray(shard_labels[f]), minlength=1000)
        if counts.min() >= need:
            break
    log(f"using {len(used)} shards ({int(counts.sum())} images); min per class {int(counts.min())}")
    if counts.min() < need:
        raise RuntimeError("not enough images per class in the train shards")

    # 2. Deterministic choice: per class, shuffle its (shard, row) list with the seed, then cut roles.
    rng = np.random.default_rng(a.seed)
    per_class = {c: [] for c in range(1000)}
    for si, f in enumerate(used):
        for r, lab in enumerate(shard_labels[f]):
            per_class[lab].append((si, r))
    roles = {name: [] for name, _ in ROLES_224 + ROLES_256}
    for c in range(1000):
        items = per_class[c]
        order = rng.permutation(len(items))
        pos = 0
        for name, n in ROLES_224 + ROLES_256:
            roles[name] += [(items[i][0], items[i][1], c) for i in order[pos:pos + n]]
            pos += n
    group224 = [x for name, _ in ROLES_224 for x in roles[name]]
    group256 = [x for name, _ in ROLES_256 for x in roles[name]]

    # 3. Download only the shards in use, then decode chosen rows shard by shard.
    def build(group, crop, out_name):
        final = cache / out_name
        if final.exists():
            log(f"{out_name} exists, skipping")
            return
        partial = cache / (out_name + ".partial.npy")
        mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(len(group), crop, crop, 3))
        by_shard = {}
        for row, (si, r, _c) in enumerate(group):
            by_shard.setdefault(si, []).append((r, row))
        pool = mp.get_context("fork").Pool(a.workers)
        try:
            for si in tqdm(sorted(by_shard), desc=out_name):
                local = hf_hub_download(REPO, used[si], repo_type="dataset")
                rows = sorted(by_shard[si])
                want = {r for r, _ in rows}
                tbl = pq.read_table(local, columns=["image"])
                col = tbl.column("image")
                jobs = [(col[r].as_py()["bytes"], 256, crop) for r, _ in rows]
                arrs = pool.map(_decode, jobs, chunksize=8)
                for (r, row), arr in zip(rows, arrs):
                    mm[row] = arr
                assert len(want) == len(rows)
        finally:
            pool.close()
            pool.join()
        mm.flush()
        del mm
        os.replace(partial, final)

    build(group224, 224, "train224_uint8.npy")
    build(group256, 256, "heal256_uint8.npy")

    # 4. Index: role -> rows in its memmap.
    idx, start = {}, 0
    for name, n in ROLES_224:
        k = n * 1000
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group224[start:start + k]])
        start += k
    start = 0
    for name, n in ROLES_256:
        k = n * 1000
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group256[start:start + k]])
        start += k
    idx["src224"] = np.array([(si, r) for si, r, _ in group224])
    idx["src256"] = np.array([(si, r) for si, r, _ in group256])
    np.savez(cache / "train_index.npz", **idx)
    meta = {"complete": True, "seed": a.seed, "shards_used": used, "roles_224": ROLES_224, "roles_256": ROLES_256,
            "n224": len(group224), "n256": len(group256), "interpolation": "bicubic",
            "note": "train images only; disjoint roles; validation is never used for fitting or healing"}
    (cache / "train_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"train cache complete: {len(group224)} at 224, {len(group256)} at 256")


if __name__ == "__main__":
    main()

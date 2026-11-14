#!/usr/bin/env python
"""ImageNet-100 at 72 x 72 for the controlled-training study (MUST-12, CORE-01; PROJECT_NEXT_PHASE.md 4.1).

Classes: data/imagenet100_classes.txt (the CMC subset, 100 WordNet IDs). Training labels 0..99 are
the rank of each ID in sorted order. The Hugging Face ILSVRC/imagenet-1k label index is the position
in the sorted list of the 1,000 IDs; the build checks that order against the dataset's classes.py.

Images: every train and validation image of the 100 classes, read from the ImageNet-1k parquet
shards on the Hugging Face hub. Each image: convert to RGB, resize the shorter side to 72 (PIL
bicubic, which antialiases when shrinking; the longer side is int(72 x long / short), as
torchvision's Resize(72)), centre crop 72 x 72 (torchvision's CenterCrop rounding), uint8.

Hub listing at the pinned revision (read 2026-11-14): train 294 shards, 146.48 GB; validation
14 shards, 6.69 GB. All train shards are needed, because each shard mixes all classes.

Outputs in --cache-dir (default cache/):
    in100_train72_uint8.npy   uint8 [N_train, 72, 72, 3]   (about 126k images, about 2 GB)
    in100_val72_uint8.npy     uint8 [N_val, 72, 72, 3]     (5,000 images, 78 MB)
    in100_train_labels.npy    int64 [N_train], 0..99
    in100_val_labels.npy      int64 [N_val], 0..99
    in100_index.npz           train_src / val_src int64 [N, 2] (shard index, row in shard),
                              train_hf_label / val_hf_label (ImageNet-1k label), train_shards /
                              val_shards (shard file names), wnids (label order), hf_index
    in100_meta.json           mirror.meta record; extra.complete = true when everything is written
(The name in100_meta.json avoids the validation cache's cache/meta.json.)
Images are in shard order and, within a shard, row order.

Resumable by shard: each processed shard is written to <cache>/in100_parts/<split>/<shard>.npz
(atomically); a rerun skips finished shards, and does not download them again. Parts are deleted
after the split is assembled (--keep-parts keeps them).

Sources (--source):
    hub     (default) hf_hub_download each shard into $HF_HOME, the same cache build_train_cache.py
            uses, so other cache builds reuse the files; shards are fetched --download-workers at a
            time ahead of the decoding
    stream  read the shards over the network (HfFileSystem) without keeping them on disk
    local   parquet files already on disk under --local-dir (searched recursively)
--download-only fetches every needed shard into $HF_HOME (snapshot_download) and stops; run it early
so the files are in place, then run the build.

    export HF_HOME=/big_disk/hf
    python scripts/build_imagenet100.py --download-only --download-workers 16
    python scripts/build_imagenet100.py --workers 16
    python scripts/build_imagenet100.py --tiny          # CPU smoke test: synthetic shards, a few seconds
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import os
import re
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Keep this module light: decoding workers (spawn on Windows) import it, so torch and mirror.* are
# imported inside main() only.
ROOT = Path(__file__).resolve().parents[1]

REPO = "ILSVRC/imagenet-1k"
# Dataset revision; the same pin as scripts/build_train_cache.py (main on 2026-11-14, last modified
# 2025-09-17). The data/ listing at this revision matched main on 2026-11-14.
REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
SPLITS = {"train": "train-", "val": "validation-"}
SIZE = 72
EXPECTED_GB = {"train": 146.48, "val": 6.69}       # hub listing, 2026-11-14


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------
def decode_resize_crop(data: bytes, size: int = SIZE) -> np.ndarray:
    """JPEG/PNG bytes -> RGB -> shorter side to `size` (bicubic, antialiased) -> centre crop -> uint8.

    Same result as torchvision Resize(size, BICUBIC) + CenterCrop(size) on a PIL image, without
    importing torch in the worker processes (a test checks the equality)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        img = im.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = size, int(size * h / w)
    else:
        nw, nh = int(size * w / h), size
    img = img.resize((nw, nh), resample=Image.BICUBIC)
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8)


def _decode_job(args):
    data, size = args
    return decode_resize_crop(data, size)


# ----------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------
def parse_classes_py(text: str) -> list[str]:
    """WordNet IDs in the order they appear in the dataset's classes.py (IMAGENET2012_CLASSES)."""
    return re.findall(r"[\"'](n\d{8})[\"']\s*:", text)


def class_setup(class_file: Path, order_check: list[str] | None) -> dict:
    from mirror.controlled import imagenet1k_wnids, in100_label_map, read_class_list

    in100 = read_class_list(class_file)
    order = imagenet1k_wnids()
    if order_check is not None and order_check != order:
        raise RuntimeError("the dataset's classes.py order differs from the sorted ImageNet-1k list; "
                           "the label mapping would be wrong")
    m = in100_label_map(in100, order)
    m["file_sha256"] = hashlib.sha256(Path(class_file).read_bytes()).hexdigest()
    m["order_checked"] = order_check is not None
    return m


# ----------------------------------------------------------------------------------------------
# Shards
# ----------------------------------------------------------------------------------------------
def list_shards(source: str, split: str, repo: str, revision: str, local_dir: str | None) -> list[str]:
    """Shard names (data/<prefix>NNNNN-of-NNNNN.parquet) in file order; local: paths."""
    prefix = SPLITS[split]
    if source == "local":
        files = sorted(Path(local_dir).rglob(f"{prefix}*.parquet"), key=lambda p: p.name)
        return [str(p) for p in files]
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
                  if f.startswith(f"data/{prefix}") and f.endswith(".parquet"))


def shard_sizes(repo: str, revision: str) -> dict:
    """{shard name: bytes} from the hub listing (no download)."""
    from huggingface_hub import HfApi

    out = {}
    for item in HfApi().list_repo_tree(repo, path_in_repo="data", repo_type="dataset", revision=revision):
        size = getattr(item, "size", None)
        if size is not None:
            out[item.path] = int(size)
    return out


def make_opener(source: str, repo: str, revision: str):
    """name -> something pyarrow.parquet.ParquetFile can read (a local path or an open file)."""
    if source == "local":
        return lambda name: name
    if source == "hub":
        from huggingface_hub import hf_hub_download

        return lambda name: hf_hub_download(repo, name, repo_type="dataset", revision=revision)
    if source == "stream":
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        return lambda name: fs.open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=64 * 2**20)
    raise ValueError(f"unknown source {source!r}")


def read_selected(src, lut: np.ndarray) -> dict:
    """Rows of one shard whose ImageNet-1k label is in the subset: row indices, labels, image bytes.
    Reads one row group at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    rows, labels, data = [], [], []
    offset = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["image", "label"])
        lab = np.asarray(tbl.column("label").to_numpy(zero_copy_only=False), dtype=np.int64)
        valid = (lab >= 0) & (lab < len(lut))
        sel = np.flatnonzero(valid & (lut[np.clip(lab, 0, len(lut) - 1)] >= 0))
        if len(sel):
            imgs = tbl.column("image").take(pa.array(sel)).to_pylist()
            data += [im["bytes"] for im in imgs]
            rows.append(sel + offset)
            labels.append(lab[sel])
        offset += tbl.num_rows
    if hasattr(src, "close"):
        src.close()
    return {"rows": np.concatenate(rows) if rows else np.zeros(0, np.int64),
            "hf_labels": np.concatenate(labels) if labels else np.zeros(0, np.int64),
            "bytes": data, "n_rows": offset}


def _ordered_prefetch(fn, items, depth: int):
    """Yield (item, fn(item)) in order, with up to `depth` calls running ahead in threads."""
    items = iter(items)
    if depth <= 0:
        for it in items:
            yield it, fn(it)
        return
    with ThreadPoolExecutor(max_workers=depth) as ex:
        queue = deque((it, ex.submit(fn, it)) for it in itertools.islice(items, depth))
        while queue:
            it, fut = queue.popleft()
            for nxt in itertools.islice(items, 1):
                queue.append((nxt, ex.submit(fn, nxt)))
            yield it, fut.result()


def _stem(name: str) -> str:
    return Path(name).name.replace(".parquet", "")


def part_path(parts_dir: Path, split: str, name: str) -> Path:
    return Path(parts_dir) / split / f"{_stem(name)}.npz"


def _part_ok(path: Path, size: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return int(z["size"]) == size
    except Exception:
        return False


def process_split(split: str, shards: list[str], opener, lut: np.ndarray, size: int, parts_dir: Path,
                  workers: int, download_workers: int, max_shards: int | None = None) -> dict:
    """Decode the subset's images of every shard into part files. Returns timing and counts."""
    todo = [s for s in shards if not _part_ok(part_path(parts_dir, split, s), size)]
    log(f"{split}: {len(shards)} shards, {len(shards) - len(todo)} already done, {len(todo)} to process")
    if max_shards is not None:
        todo = todo[:int(max_shards)]
    (Path(parts_dir) / split).mkdir(parents=True, exist_ok=True)
    pool = None
    if workers > 1 and todo:
        import multiprocessing as mp

        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() and os.name != "nt" else "spawn")
        pool = ctx.Pool(workers)
    t0, n_img, wait_s = time.time(), 0, 0.0
    try:
        t_wait = time.time()
        for k, (name, got) in enumerate(_ordered_prefetch(lambda s: read_selected(opener(s), lut), todo,
                                                          download_workers)):
            wait_s += time.time() - t_wait
            jobs = [(b, size) for b in got["bytes"]]
            if pool is not None:
                arrs = pool.map(_decode_job, jobs, chunksize=16)
            else:
                arrs = [_decode_job(j) for j in jobs]
            imgs = np.stack(arrs) if arrs else np.zeros((0, size, size, 3), np.uint8)
            out = part_path(parts_dir, split, name)
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, images=imgs, hf_labels=got["hf_labels"], rows=got["rows"], n_rows=got["n_rows"],
                     size=size, shard=np.array(Path(name).name))
            os.replace(tmp, out)
            n_img += len(arrs)
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f"{split}: {k + 1}/{len(todo)} shards, {n_img} images, {el / 60:.1f} min, "
                    f"{n_img / max(el, 1e-9):.0f} img/s, waiting for shards {wait_s / 60:.1f} min")
            t_wait = time.time()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"shards_processed": len(todo), "images_decoded": n_img, "seconds": round(time.time() - t0, 1),
            "seconds_waiting_for_shards": round(wait_s, 1)}


def assemble(split: str, shards: list[str], parts_dir: Path, cache_dir: Path, lut: np.ndarray, size: int,
             keep_parts: bool) -> dict:
    """Concatenate the parts in shard order into the final arrays. Returns the index arrays."""
    paths = [part_path(parts_dir, split, s) for s in shards]
    missing = [p.name for p in paths if not _part_ok(p, size)]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} shards not processed yet (first: {missing[0]}); rerun")
    counts = []
    for p in paths:
        with np.load(p) as z:
            counts.append(int(len(z["rows"])))
    n = int(sum(counts))
    final = Path(cache_dir) / f"in100_{split}{size}_uint8.npy"
    partial = final.with_name(final.name + ".partial.npy")
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    src, hf = [], []
    pos = 0
    for si, p in enumerate(paths):
        with np.load(p) as z:
            k = len(z["rows"])
            if k:
                mm[pos:pos + k] = z["images"]
            src.append(np.stack([np.full(k, si, np.int64), z["rows"].astype(np.int64)], 1))
            hf.append(z["hf_labels"].astype(np.int64))
            pos += k
    mm.flush()
    del mm
    os.replace(partial, final)
    hf_labels = np.concatenate(hf) if hf else np.zeros(0, np.int64)
    labels = lut[hf_labels]
    if (labels < 0).any():
        raise RuntimeError(f"{split}: labels outside the subset")
    np.save(Path(cache_dir) / f"in100_{split}_labels.npy", labels.astype(np.int64))
    index = {f"{split}_src": np.concatenate(src).reshape(-1, 2) if src else np.zeros((0, 2), np.int64),
             f"{split}_hf_label": hf_labels,
             f"{split}_shards": np.array([Path(s).name for s in shards], dtype=str)}
    np.savez(Path(parts_dir) / f"{split}_assembled.npz", **index)
    if not keep_parts:
        shutil.rmtree(Path(parts_dir) / split, ignore_errors=True)
    log(f"{split}: wrote {final.name} [{n}, {size}, {size}, 3]")
    return index


def check_counts(split: str, labels: np.ndarray, n_classes: int, strict: bool) -> dict:
    per = np.bincount(labels, minlength=n_classes)
    info = {"n": int(len(labels)), "per_class_min": int(per.min()), "per_class_max": int(per.max()),
            "classes_present": int((per > 0).sum())}
    problems = []
    if info["classes_present"] != n_classes:
        problems.append(f"{n_classes - info['classes_present']} classes have no images")
    if split == "val" and not (per == 50).all():
        problems.append(f"validation should have 50 images per class, got {info['per_class_min']}..{info['per_class_max']}")
    info["problems"] = problems
    for msg in problems:
        if strict:
            raise RuntimeError(f"{split}: {msg} (use --no-strict to write the cache anyway)")
        log(f"warning: {split}: {msg}")
    return info


# ----------------------------------------------------------------------------------------------
# Synthetic shards (--tiny and tests)
# ----------------------------------------------------------------------------------------------
def make_synthetic_shards(root: Path, hf_in100: np.ndarray, n_train_shards: int = 2, rows_per_shard: int = 48,
                          n_val_rows: int = 40, seed: int = 0) -> Path:
    """Parquet shards in the hub layout (image {bytes, path}, label) under <root>/data/: a mix of
    subset and other labels, JPEG and PNG, colour, greyscale and CMYK, varied sizes and aspect ratios."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    rng = np.random.default_rng(seed)
    data_dir = Path(root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    others = np.setdiff1d(np.arange(1000), hf_in100)
    subset = rng.choice(hf_in100, size=6, replace=False)

    def one_image(i):
        w, h = int(rng.integers(60, 160)), int(rng.integers(60, 160))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        arr[: h // 3, : w // 3] = (255, 0, 0)
        mode, fmt = ["RGB", "L", "CMYK", "RGB"][i % 4], ["JPEG", "JPEG", "JPEG", "PNG"][i % 4]
        img = Image.fromarray(arr).convert(mode)
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=90) if fmt == "JPEG" else img.save(buf, format=fmt)
        return buf.getvalue()

    def write(name, n):
        labels = np.where(rng.random(n) < 0.5, rng.choice(subset, n), rng.choice(others, n)).astype(np.int64)
        imgs = [{"bytes": one_image(i), "path": f"img_{i}.JPEG"} for i in range(n)]
        tbl = pa.table({"image": pa.array(imgs, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                        "label": pa.array(labels, type=pa.int64())})
        pq.write_table(tbl, data_dir / name, row_group_size=16)

    for s in range(n_train_shards):
        write(f"train-{s:05d}-of-{n_train_shards:05d}.parquet", rows_per_shard)
    write("validation-00000-of-00001.parquet", n_val_rows)
    return Path(root)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", default=str(ROOT / "data" / "imagenet100_classes.txt"))
    p.add_argument("--cache-dir", default=None, help="default: cache/ (tiny: tiny_out/in100_cache)")
    p.add_argument("--source", choices=("hub", "stream", "local"), default=None,
                   help="hub (default): shards into $HF_HOME via hf_hub_download, reusable; "
                        "stream: read remotely, nothing kept; local: --local-dir")
    p.add_argument("--local-dir", default=None, help="folder with the parquet shards (--source local)")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1), help="decoding processes")
    p.add_argument("--download-workers", type=int, default=8, help="shards fetched and read ahead in threads")
    p.add_argument("--download-only", action="store_true", help="fetch the shards into $HF_HOME and stop")
    p.add_argument("--max-shards", type=int, default=None, help="process at most n shards per split, then stop "
                                                                "(tests; the build stays resumable)")
    p.add_argument("--keep-parts", action="store_true")
    p.add_argument("--no-strict", action="store_true", help="write the cache even if the counts look wrong")
    p.add_argument("--no-verify-classes", action="store_true",
                   help="skip the check of the label order against the dataset's classes.py")
    p.add_argument("--tiny", action="store_true", help="synthetic shards, CPU, a few seconds")
    a = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from mirror.controlled import labels_fingerprint
    from mirror.meta import write_meta

    t0 = time.time()
    if a.tiny:
        cache = Path(a.cache_dir or ROOT / "tiny_out" / "in100_cache")
        source, strict = "local", False
    else:
        cache = Path(a.cache_dir or ROOT / "cache")
        source, strict = a.source or "hub", not a.no_strict
    cache.mkdir(parents=True, exist_ok=True)
    parts_dir = cache / "in100_parts"
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    for s in splits:
        if s not in SPLITS:
            raise SystemExit(f"unknown split {s!r}; use train and/or val")
    if source != "local" and not os.environ.get("HF_HOME"):
        log("warning: HF_HOME is not set; Hugging Face will download to its default cache")

    # Label order check against the dataset's own class file.
    order_check = None
    if not a.no_verify_classes and not a.tiny:
        if source == "local":
            cands = list(Path(a.local_dir).rglob("classes.py"))
            if cands:
                order_check = parse_classes_py(cands[0].read_text(encoding="utf-8"))
            else:
                log("warning: no classes.py under --local-dir; label order taken from the sorted 1k list")
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(a.repo, "classes.py", repo_type="dataset", revision=a.revision)
            order_check = parse_classes_py(Path(path).read_text(encoding="utf-8"))
    cls = class_setup(Path(a.classes), order_check)
    lut = cls["lut"]
    log(f"classes: 100 from {a.classes}; ImageNet-1k indices {cls['hf_index'][:5].tolist()} ...; "
        f"order checked against classes.py: {cls['order_checked']}")

    if a.tiny:
        src = Path(a.local_dir) if a.local_dir else cache / "in100_src"
        if not list(src.rglob("*.parquet")):
            make_synthetic_shards(src, cls["hf_index"])
        a.local_dir = str(src)

    # Size and disk check (hub listing, no download).
    sizes = {}
    if source != "local":
        sizes = shard_sizes(a.repo, a.revision)
        for s in splits:
            gb = sum(v for k, v in sizes.items() if k.startswith(f"data/{SPLITS[s]}")) / 1e9
            log(f"{s}: {gb:.2f} GB in the hub listing (expected about {EXPECTED_GB[s]} GB)")
        hf_home = os.environ.get("HF_HOME")
        if hf_home and source == "hub":
            free = shutil.disk_usage(hf_home).free / 1e9
            need = sum(sizes.values()) / 1e9
            log(f"free disk under HF_HOME: {free:.0f} GB")
            if free < need * 0.5:
                log(f"warning: little free disk for about {need:.0f} GB of shards (some may already be cached)")

    if a.download_only:
        if source != "hub":
            raise SystemExit("--download-only needs --source hub")
        from huggingface_hub import snapshot_download

        patterns = [f"data/{SPLITS[s]}*.parquet" for s in splits]
        t = time.time()
        snapshot_download(a.repo, repo_type="dataset", revision=a.revision, allow_patterns=patterns,
                          max_workers=a.download_workers)
        el = time.time() - t
        gb = sum(v for k, v in sizes.items() if any(k.startswith(pp.split("*")[0]) for pp in patterns)) / 1e9
        log(f"download done: {gb:.1f} GB in the listing, {el / 60:.1f} min ({gb / max(el, 1e-9):.2f} GB/s if "
            "nothing was cached before)")
        return {"download_seconds": el}

    opener = make_opener(source, a.repo, a.revision)
    timing = {}

    def assembled_ok(split):
        return ((cache / f"in100_{split}{a.size}_uint8.npy").exists() and (parts_dir / f"{split}_assembled.npz").exists()
                and (cache / f"in100_{split}_labels.npy").exists())

    for split in splits:
        if assembled_ok(split):
            log(f"{split}: already assembled, skipping")
            continue
        shards = list_shards(source, split, a.repo, a.revision, a.local_dir)
        if not shards:
            raise SystemExit(f"no {split} shards found")
        timing[split] = process_split(split, shards, opener, lut, a.size, parts_dir, a.workers,
                                      a.download_workers, a.max_shards)
        if a.max_shards is not None and not all(_part_ok(part_path(parts_dir, split, s), a.size) for s in shards):
            log(f"{split}: stopped after --max-shards {a.max_shards}; rerun to continue")
            return {"stopped": True}
        assemble(split, shards, parts_dir, cache, lut, a.size, a.keep_parts)

    # Index and checks over every split that is assembled (a split built in an earlier run counts).
    index, info = {}, {}
    for split in SPLITS:
        if not assembled_ok(split):
            continue
        with np.load(parts_dir / f"{split}_assembled.npz") as z:
            index.update({k: z[k] for k in z.files})
        labels = np.load(cache / f"in100_{split}_labels.npy")
        info[split] = check_counts(split, labels, 100, strict)
        info[split]["labels_sha"] = labels_fingerprint(labels)
    np.savez(cache / "in100_index.npz", **index, wnids=np.array(cls["wnids"], dtype=str), hf_index=cls["hf_index"])
    extra = {
        "complete": all(s in info for s in SPLITS),
        "synthetic": bool(a.tiny),
        "size": a.size,
        "num_classes": 100,
        "n_train": info.get("train", {}).get("n"),
        "n_val": info.get("val", {}).get("n"),
        "train_labels_sha": info.get("train", {}).get("labels_sha"),
        "val_labels_sha": info.get("val", {}).get("labels_sha"),
        "counts": info,
        "repo": a.repo,
        "revision": a.revision,
        "source": source,
        "class_list": {"file": str(Path(a.classes).name), "sha256": cls["file_sha256"],
                       "order_checked_against_classes_py": cls["order_checked"],
                       "wnids_label_order": cls["wnids"], "hf_index": cls["hf_index"]},
        "preprocessing": f"RGB; shorter side to {a.size} (PIL bicubic, antialiased); centre crop {a.size}; uint8",
        "hub_bytes": {s: int(sum(v for k, v in sizes.items() if k.startswith(f"data/{SPLITS[s]}"))) for s in splits}
        if sizes else None,
        "timing": timing,
    }
    write_meta(cache, name="in100_meta.json", experiment="IN100-BUILD", config=vars(a), seeds={},
               models={}, wall_seconds=time.time() - t0, extra=extra)
    state = "complete" if extra["complete"] else "partly built"
    log(f"ImageNet-100 cache {state} in {cache}: " + ", ".join(f"{s} {v['n']}" for s, v in info.items())
        + f" ({(time.time() - t0) / 60:.1f} min)")
    return extra


if __name__ == "__main__":
    main()

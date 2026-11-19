"""Final-phase data roles: existing caches, new roles, full-train stream, role accessor and checks.

FINAL_PHASE_PLAN.md v2, Section 7 item 12; docs/final_phase_design.md; notes/paper/method_spec.md Section 0.

Existing caches, reused unchanged (scripts/build_cache.py, scripts/build_train_cache.py, scripts/h100_data.sh):
    cache/train224_uint8.npy + train_index.npz + train_meta.json   fit0, fit1, fit2 (2,000 each), bank (100k)
    cache/heal256_uint8.npy (same index)                           healval (5k), heal (100k), 256 px
    cache/msel224_uint8.npy + msel_index.npz                       msel (10k)
    cache/val224_uint8.npy + val_labels.npy + meta.json            val (all 50k validation images)

New roles (this module; `python scripts/fp_cache.py build --roles ...`):
    fit3       2 per class (2,000), 224 px             cache/fit3_224_uint8.npy,  fit3_index.npz,  fit3_meta.json
    heal_ext   200 per class (200,000), 256 px          cache/heal_ext256_uint8.npy, heal_ext_index.npz, heal_ext_meta.json
    coco       2,000 COCO train images, 224 px          cache/coco224_uint8.npy,  coco_index.npz,  coco_meta.json
    stream_train  all 1,281,167 ImageNet-1k train images at 224 px, streamed (never stored decoded);
               cache/stream_train_index.npz holds every (shard, row) ID and label in stream order.

Selection (pure functions of the shard labels and fixed seeds, as scripts/build_train_cache.py does). The existing
roles use the train shards in file order: the "used" shards (a prefix, until every class has 211 images) for fit0..2,
bank, healval and heal, then the next shards for msel (until every class has 10; seed 1). The new ImageNet roles
continue the same chain: fit3 takes the next shards until every class has 2 (seed 2), heal_ext the shards after
those until every class has 200 (seed 3). Inside a block, every class's (shard, row) list is permuted with the
block's own generator and the first `per_class` are taken (select_block, the same code as select_msel). Different
shards, so every new role is disjoint by construction from every other role, and the check below verifies it by
(shard, row) ID. heal and heal_ext together give 300,000 distinct images. COCO: the first train shards in file order
until the pool has at least 4,000 rows, then 2,000 rows chosen with default_rng(4), stored in (shard, row) order.
stream_train is the whole train split, so it overlaps every train role by design (full-bank k-NN and the linear
protocol use all train images); for it the check verifies that it covers every train image exactly once and that its
labels agree with every role's labels.

Preprocessing: exactly the existing one, PIL bicubic resize of the shorter side to 256 and a centre crop (224, or 256
for the heal roles), uint8 RGB. `resize_centre_crop` repeats torchvision 0.26's PIL code path call for call
(Resize: _compute_resized_output_size, then Image.resize(size, BICUBIC); CenterCrop: int(round(...)) offsets and
Image.crop), so decoding workers need neither torch nor torchvision; builds compare the first images with the
torchvision transform itself (`--parity`, as scripts/build_train_cache.py applies it) and stop on any difference.

IDs: (shard number, row in that shard), int64 [N, 2]; the shard number is the NNNNN of "train-NNNNN-of-MMMMM" in the
shard file name (ImageNet: data/train-00012-of-00294.parquet -> 12). Validation rows have shard number -1 and their
row in the validation cache. `id_keys(ids)` = shard * 1,000,000 + row.

Accessor: get_role(name) -> Role(images memmap view or None, labels int64, ids int64 [N, 2], ...);
role_arrays(name) -> (images, labels, ids); Role.batches(batch_size) -> (uint8 [B, S, S, 3], labels, ids) for every
role, streamed for stream_train.

Sources checked on 19 November 2026 (web):
  https://huggingface.co/api/datasets/ILSVRC/imagenet-1k/revision/49e2ee26f3810fb5a7536bbf732a7b07389a47b5
      294 shards data/train-00000-of-00294.parquet ... data/train-00293-of-00294.parquet; train 1,281,167 images
  https://huggingface.co/api/datasets/ILSVRC/imagenet-1k/tree/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data
      train shard sizes about 0.48 to 0.53 GB (listing partly read)
  https://huggingface.co/api/datasets/detection-datasets/coco
      sha cf0b22332314a937e9dc8a1957b21725430bb41d, last modified 2023-03-15, not gated, not private; 40 train
      shards data/train-000NN-of-00040-<hash>.parquet and 2 val shards
  https://huggingface.co/datasets/detection-datasets/coco/raw/cf0b22332314a937e9dc8a1957b21725430bb41d/dataset_infos.json
      columns image_id, image, width, height, objects; train 117,266 rows, val 4,952
  https://huggingface.co/api/datasets/detection-datasets/coco/tree/cf0b22332314a937e9dc8a1957b21725430bb41d/data
      train shards 19,159,418,691 bytes in all (about 0.48 GB each)
  The dataset card has no description or licence. 117,266 and 4,952 match the COCO 2017 train and val images that
  have at least one object annotation (verify); COCO image licences are those of the Flickr originals (verify).
  Fallback if COCO is unusable: Places365 (the plan's fallback; no source checked here).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

IMAGENET_REPO = "ILSVRC/imagenet-1k"
IMAGENET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"      # = scripts/build_train_cache.py REVISION
TRAIN_PREFIX = "data/train-"
N_TRAIN = 1_281_167
N_TRAIN_SHARDS = 294
N_CLASSES = 1000
COCO_REPO = "detection-datasets/coco"
COCO_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"
USED_PER_CLASS = 211            # fit0..2 (2 + 2 + 2), bank 100, healval 5, heal 100 (scripts/build_train_cache.py)
MSEL_PER_CLASS, MSEL_SEED = 10, 1
COCO_POOL_FACTOR = 2
ROW_BASE = 1_000_000
IMAGENET_SHARD_BYTES = 0.50e9   # estimate: 0.48 to 0.53 GB per train shard (hub listing above)
COCO_SHARD_BYTES = 19_159_418_691 / 40
SMALLEST_CLASS = 732            # images in the smallest ImageNet-1k train class (verify)


@dataclass(frozen=True)
class RoleSpec:
    name: str
    dataset: str          # imagenet_train or coco_train
    per_class: int        # ImageNet roles: images per class (0 for COCO)
    n: int                # images in the role
    seed: int
    resize: int
    crop: int
    images_file: str
    index_file: str
    meta_file: str


FIT3 = RoleSpec("fit3", "imagenet_train", 2, 2_000, 2, 256, 224, "fit3_224_uint8.npy", "fit3_index.npz",
                "fit3_meta.json")
HEAL_EXT = RoleSpec("heal_ext", "imagenet_train", 200, 200_000, 3, 256, 256, "heal_ext256_uint8.npy",
                    "heal_ext_index.npz", "heal_ext_meta.json")
COCO = RoleSpec("coco", "coco_train", 0, 2_000, 4, 256, 224, "coco224_uint8.npy", "coco_index.npz", "coco_meta.json")
IMAGENET_CHAIN = (FIT3, HEAL_EXT)          # drawn in this order from the shards after msel
NEW_ROLES = {s.name: s for s in (FIT3, HEAL_EXT, COCO)}
STREAM_INDEX = "stream_train_index.npz"
STREAM_META = "stream_train_meta.json"

# Existing roles: (memmap, source column in train_index.npz, crop)
OLD_TRAIN_ROLES = {"fit0": ("train224_uint8.npy", "src224", 224), "fit1": ("train224_uint8.npy", "src224", 224),
                   "fit2": ("train224_uint8.npy", "src224", 224), "bank": ("train224_uint8.npy", "src224", 224),
                   "healval": ("heal256_uint8.npy", "src256", 256), "heal": ("heal256_uint8.npy", "src256", 256)}
ALL_ROLES = (*OLD_TRAIN_ROLES, "msel", "val", "fit3", "heal_ext", "coco", "stream_train")
TRAIN_SET_ROLES = (*OLD_TRAIN_ROLES, "msel", "fit3", "heal_ext")      # must be pairwise disjoint
EXPECTED = {"fit0": 2000, "fit1": 2000, "fit2": 2000, "bank": 100_000, "healval": 5000, "heal": 100_000,
            "msel": 10_000, "fit3": 2000, "heal_ext": 200_000, "coco": 2000, "val": 50_000, "stream_train": N_TRAIN}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_hash(root: Path = REPO_ROOT) -> str | None:
    """Commit hash without importing torch (mirror.utils imports it): git, else .git/HEAD."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        head = (Path(root) / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = Path(root) / ".git" / head.split(" ", 1)[1].strip()
        return ref.read_text().strip() if ref.exists() else None
    except OSError:
        return None


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# IDs
# ----------------------------------------------------------------------------------------------
_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)")


def shard_number(name: str) -> int:
    """NNNNN of '<split>-NNNNN-of-MMMMM[...].parquet' (the global index of the shard in its split)."""
    m = _SHARD_RE.search(Path(str(name)).name)
    if m is None:
        raise ValueError(f"cannot read a shard number from {name!r}")
    return int(m.group(1))


def make_ids(picks, shards) -> np.ndarray:
    nums = np.array([shard_number(s) for s in shards], dtype=np.int64)
    if not len(picks):
        return np.zeros((0, 2), dtype=np.int64)
    src = np.array([(si, r) for si, r, *_ in picks], dtype=np.int64)
    return np.stack([nums[src[:, 0]], src[:, 1]], axis=1)


def id_keys(ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1, 2)
    return ids[:, 0] * ROW_BASE + ids[:, 1]


# ----------------------------------------------------------------------------------------------
# Selection (pure)
# ----------------------------------------------------------------------------------------------
def select_used_shards(files, labels_of, need: int = USED_PER_CLASS, n_classes: int = N_CLASSES):
    """The shards scripts/build_train_cache.py uses for fit0..2, bank, healval and heal (same loop)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    used = []
    for f in files:
        used.append(f)
        counts += np.bincount(np.asarray(labels_of(f)), minlength=n_classes)
        if counts.min() >= need:
            break
    return used, counts


def select_block(files, start: int, labels_of, per_class: int, seed: int, n_classes: int = N_CLASSES):
    """Shards files[start:] in file order until every class has `per_class` images; then per class, in class
    order, permute its (shard, row) list with default_rng(seed) and take the first `per_class`.
    Returns (pool, picks) with picks = [(index into pool, row, class), ...] in class order. Identical to
    scripts/build_train_cache.py select_msel(files, start, labels_of, per_class, seed)."""
    pool, counts = [], np.zeros(n_classes, dtype=np.int64)
    for f in files[start:]:
        pool.append(f)
        counts += np.bincount(np.asarray(labels_of(f)), minlength=n_classes)
        if counts.min() >= per_class:
            break
    if counts.min() < per_class:
        raise RuntimeError(f"not enough images per class after the first {start} shards "
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


def plan_imagenet(files, labels_of, used=None, msel_shards=None, chain=IMAGENET_CHAIN, upto: str | None = None,
                  n_classes: int = N_CLASSES, used_need: int = USED_PER_CLASS, msel_per_class: int = MSEL_PER_CLASS,
                  msel_seed: int = MSEL_SEED) -> dict:
    """{role: {"spec", "shards", "picks", "first_shard_index"}} for the new ImageNet roles, in chain order (up to and
    including `upto`), plus "_used" and "_msel" (the shard lists of the existing roles). `used` and `msel_shards`
    come from the existing caches when given; otherwise they are recomputed exactly as the train cache builder does."""
    files = list(files)
    if used is None:
        used, counts = select_used_shards(files, labels_of, used_need, n_classes)
        if counts.min() < used_need:
            raise RuntimeError("not enough images per class for the existing roles")
    used = list(used)
    if used != files[:len(used)]:
        raise RuntimeError("the shards used by the existing roles are not the first shards in file order")
    n_used = len(used)
    if msel_shards is None:
        msel_shards, _ = select_block(files, n_used, labels_of, msel_per_class, msel_seed, n_classes)
    msel_shards = list(msel_shards)
    if msel_shards != files[n_used:n_used + len(msel_shards)]:
        raise RuntimeError("the msel shards are not the shards right after the used ones")
    out = {"_used": used, "_msel": msel_shards}
    start = n_used + len(msel_shards)
    for spec in chain:
        pool, picks = select_block(files, start, labels_of, spec.per_class, spec.seed, n_classes)
        out[spec.name] = {"spec": spec, "shards": pool, "picks": picks, "first_shard_index": start}
        start += len(pool)
        if spec.name == upto:
            break
    return out


def plan_coco(files, n_rows_of, spec: RoleSpec = COCO, pool_factor: int = COCO_POOL_FACTOR) -> dict:
    """First shards in file order until the pool has pool_factor * n rows; n rows drawn with default_rng(seed),
    returned in (shard, row) order with class -1."""
    pool, sizes = [], []
    for f in files:
        pool.append(f)
        sizes.append(int(n_rows_of(f)))
        if sum(sizes) >= pool_factor * spec.n:
            break
    total = sum(sizes)
    if total < spec.n:
        raise RuntimeError(f"only {total} rows in {len(pool)} shards; need {spec.n}")
    flat = np.sort(np.random.default_rng(spec.seed).permutation(total)[:spec.n])
    bounds = np.cumsum(sizes)
    si = np.searchsorted(bounds, flat, side="right")
    rows = flat - (bounds - np.asarray(sizes))[si]
    picks = [(int(s), int(r), -1) for s, r in zip(si, rows)]
    return {"spec": spec, "shards": pool, "picks": picks, "first_shard_index": 0, "pool_rows": int(total)}


def selection_fingerprint(shards, picks) -> str:
    h = hashlib.sha256()
    h.update("\n".join(shards).encode())
    arr = np.array([(si, r, c) for si, r, c in picks], dtype=np.int64).reshape(-1, 3)
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------
def resize_centre_crop(img, resize: int = 256, crop: int = 224):
    """torchvision Compose([Resize(resize, BICUBIC), CenterCrop(crop)]) on a PIL image, with the same PIL calls."""
    from PIL import Image

    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = resize, int(resize * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    if [h, w] != [new_h, new_w]:
        img = img.resize((new_w, new_h), Image.BICUBIC)
    w, h = img.size
    if crop > w or crop > h:
        raise ValueError(f"crop {crop} larger than the resized image {w} x {h} (needs crop <= resize)")
    top = int(round((h - crop) / 2.0))
    left = int(round((w - crop) / 2.0))
    return img.crop((left, top, left + crop, top + crop))


def decode_image(data: bytes, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Encoded image bytes -> uint8 [crop, crop, 3] (RGB), the cache preprocessing."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(resize_centre_crop(img, resize, crop), dtype=np.uint8)


def decode_reference(data: bytes, resize: int = 256, crop: int = 224) -> np.ndarray:
    """The torchvision transform itself (scripts/build_train_cache.py _decode), for parity checks."""
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(io.BytesIO(data)).convert("RGB")
    tf = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(crop)])
    return np.asarray(tf(img), dtype=np.uint8)


def decode_many(args) -> np.ndarray:
    """Worker: (list of encoded images, resize, crop) -> uint8 [k, crop, crop, 3]."""
    datas, resize, crop = args
    out = np.empty((len(datas), crop, crop, 3), dtype=np.uint8)
    for i, d in enumerate(datas):
        out[i] = decode_image(d, resize, crop)
    return out


def parity_check(datas, decoded, resize: int, crop: int) -> dict:
    """Compare decoded images with the torchvision transform; raise on any difference."""
    try:
        import torchvision  # noqa: F401
    except Exception as exc:          # noqa: BLE001
        return {"checked": 0, "note": f"torchvision unavailable ({type(exc).__name__}); not checked"}
    for i, d in enumerate(datas):
        ref = decode_reference(d, resize, crop)
        if not np.array_equal(ref, decoded[i]):
            raise RuntimeError(f"preprocessing differs from the torchvision transform on image {i}")
    return {"checked": len(datas), "identical": True}


def _pool(workers: int, context: str | None = None):
    if workers is None or workers <= 1:
        return None
    import multiprocessing as mp

    ctx = context or ("forkserver" if sys.platform.startswith("linux") else "spawn")
    return mp.get_context(ctx).Pool(workers)


# ----------------------------------------------------------------------------------------------
# Shard sources
# ----------------------------------------------------------------------------------------------
def _image_bytes(col) -> list:
    import pyarrow as pa

    out = []
    chunks = col.chunks if hasattr(col, "chunks") else [col]
    for ch in chunks:
        if pa.types.is_struct(ch.type):
            vals = ch.field("bytes").to_pylist()
        else:
            vals = ch.to_pylist()
        out.extend(vals)
    if any(v is None for v in out):
        raise ValueError("some images are stored without bytes (paths only); not supported")
    return out


def read_rows(path, rows=None, image_col: str = "image", label_col: str | None = None, extra_cols=()) -> dict:
    """Encoded images (and labels and extra columns) of the given rows of one parquet shard (all rows if None)."""
    import pyarrow.parquet as pq

    cols = [image_col] + ([label_col] if label_col else []) + [c for c in extra_cols if c != label_col]
    tbl = pq.read_table(path, columns=cols)
    if rows is not None:
        tbl = tbl.take(np.asarray(rows, dtype=np.int64))
    out = {"images": _image_bytes(tbl.column(image_col)), "n": tbl.num_rows}
    out["labels"] = tbl.column(label_col).to_pylist() if label_col else None
    for c in extra_cols:
        out[c] = tbl.column(c).to_pylist()
    return out


class LocalParquetSource:
    """Parquet shards in a local folder (tests, dry runs, shards copied by hand). `cached` lists the names that
    count as already present (never deleted by release); the others behave like fresh downloads."""

    def __init__(self, folder, pattern: str = "data/train-*.parquet", dataset: str = "local", image_col: str = "image",
                 label_col: str | None = "label", extra_cols=(), cached=None, delete_on_release: bool = False):
        self.folder, self.pattern, self.dataset = Path(folder), pattern, dataset
        self.image_col, self.label_col, self.extra_cols = image_col, label_col, tuple(extra_cols)
        self.cached = None if cached is None else set(cached)
        self.delete_on_release = delete_on_release
        self.fetched, self.released = [], []

    def files(self) -> list:
        return sorted(p.relative_to(self.folder).as_posix() for p in self.folder.glob(self.pattern))

    def labels(self, f) -> list:
        if self.label_col is None:
            return [-1] * self.n_rows(f)
        import pyarrow.parquet as pq

        return pq.read_table(self.folder / f, columns=[self.label_col]).column(self.label_col).to_pylist()

    def n_rows(self, f) -> int:
        import pyarrow.parquet as pq

        return pq.ParquetFile(self.folder / f).metadata.num_rows

    def fetch(self, f):
        self.fetched.append(f)
        return self.folder / f, (self.cached is None or f in self.cached)

    def release(self, f) -> None:
        self.released.append(f)
        if self.delete_on_release:
            (self.folder / f).unlink(missing_ok=True)


class HubParquetSource:
    """Parquet shards of a Hugging Face dataset at a fixed revision, downloaded under $HF_HOME (VM only).

    Labels are read remotely from the label column only and cached in a JSON file in the same format as
    scripts/build_train_cache.py ShardLabels (cache/train_shard_labels.json), so labels fetched by either are shared."""

    def __init__(self, repo: str, revision: str, prefix: str = TRAIN_PREFIX, dataset: str = "imagenet_train",
                 image_col: str = "image", label_col: str | None = "label", extra_cols=(), labels_cache=None):
        self.repo, self.revision, self.prefix, self.dataset = repo, revision, prefix, dataset
        self.image_col, self.label_col, self.extra_cols = image_col, label_col, tuple(extra_cols)
        self.labels_cache = Path(labels_cache) if labels_cache else None
        self._labels = json.loads(self.labels_cache.read_text()) if self.labels_cache and \
            self.labels_cache.exists() else {}
        self._files = None
        self._lock = threading.Lock()
        self._fs = None

    def files(self) -> list:
        if self._files is None:
            from huggingface_hub import HfApi

            names = HfApi().list_repo_files(self.repo, repo_type="dataset", revision=self.revision)
            self._files = sorted(f for f in names if f.startswith(self.prefix) and f.endswith(".parquet"))
        return list(self._files)

    def _remote(self, f):
        from huggingface_hub import HfFileSystem

        if self._fs is None:
            self._fs = HfFileSystem()
        return self._fs.open(f"datasets/{self.repo}@{self.revision}/{f}", "rb")

    def _save_labels(self) -> None:
        if self.labels_cache is None:
            return
        with self._lock:
            # merge with labels another process may have written meanwhile
            disk = json.loads(self.labels_cache.read_text()) if self.labels_cache.exists() else {}
            disk.update(self._labels)
            self._labels = disk
            tmp = self.labels_cache.with_name(self.labels_cache.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(disk))
            os.replace(tmp, self.labels_cache)

    def _fetch_labels(self, f) -> list:
        import pyarrow.parquet as pq

        with self._remote(f) as fh:
            return pq.read_table(fh, columns=[self.label_col]).column(self.label_col).to_pylist()

    def labels(self, f) -> list:
        if self.label_col is None:
            return [-1] * self.n_rows(f)
        if f not in self._labels:
            labs = self._fetch_labels(f)
            with self._lock:
                self._labels[f] = labs
            self._save_labels()
        return self._labels[f]

    def labels_many(self, files, threads: int = 8) -> dict:
        missing = [f for f in files if f not in self._labels]
        if missing and self.label_col is not None:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max(1, threads)) as ex:
                for f, labs in zip(missing, ex.map(self._fetch_labels, missing)):
                    with self._lock:
                        self._labels[f] = labs
            self._save_labels()
        return {f: self.labels(f) for f in files}

    def n_rows(self, f) -> int:
        if self.label_col is not None:
            return len(self.labels(f))
        import pyarrow.parquet as pq

        with self._remote(f) as fh:
            return pq.ParquetFile(fh).metadata.num_rows

    def _cached_path(self, f):
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(self.repo, f, revision=self.revision, repo_type="dataset")
        return p if isinstance(p, str) and os.path.exists(p) else None

    def fetch(self, f):
        from huggingface_hub import hf_hub_download

        was = self._cached_path(f) is not None
        return Path(hf_hub_download(self.repo, f, repo_type="dataset", revision=self.revision)), was

    def release(self, f) -> None:
        p = self._cached_path(f)
        if p is None:
            return
        link = Path(p)
        real = link.resolve()
        link.unlink(missing_ok=True)
        if real != link:
            real.unlink(missing_ok=True)


def imagenet_source(cache_dir=None) -> HubParquetSource:
    cache = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"
    return HubParquetSource(IMAGENET_REPO, IMAGENET_REVISION, TRAIN_PREFIX, "imagenet_train",
                            labels_cache=cache / "train_shard_labels.json")


def coco_source() -> HubParquetSource:
    return HubParquetSource(COCO_REPO, COCO_REVISION, TRAIN_PREFIX, "coco_train", label_col=None,
                            extra_cols=("image_id",))


# ----------------------------------------------------------------------------------------------
# Decoding into memmaps (resumable shard by shard)
# ----------------------------------------------------------------------------------------------
def decode_into(final, picks, shard_names, source, resize: int, crop: int, workers: int = 8, chunk: int = 64,
                delete_shards: bool = False, parity: int = 0, extra_cols=(), mp_context: str | None = None,
                log=_log) -> dict:
    """Decode the picked rows into `final` (uint8 [len(picks), crop, crop, 3]), shard by shard.

    picks: [(index into shard_names, row, class), ...] in memmap order. Progress is kept in <final>.progress.json
    next to <final>.partial.npy, so an interrupted build resumes after its last finished shard. Shards that the
    source downloaded for this build are deleted after use with delete_shards. parity: compare the first images
    with the torchvision transform. Returns {"n", "shards", "parity", "extras"}."""
    final = Path(final)
    if final.exists():
        log(f"{final.name} exists, skipping")
        return {"n": len(picks), "skipped": True}
    partial = final.with_name(final.name + ".partial.npy")
    prog_path = final.with_name(final.name + ".progress.json")
    n = len(picks)
    shape = [n, crop, crop, 3]
    done, extras = set(), {c: [None] * n for c in extra_cols}
    mm = None
    if partial.exists() and prog_path.exists():
        prog = json.loads(prog_path.read_text())
        if prog.get("shape") == shape:
            mm = np.lib.format.open_memmap(partial, mode="r+")
            if list(mm.shape) == shape and mm.dtype == np.uint8:
                done = set(prog.get("shards_done", []))
                for c in extra_cols:
                    extras[c] = prog.get("extras", {}).get(c, extras[c])
                log(f"{final.name}: resuming after {len(done)} finished shards")
            else:
                del mm
                mm = None
    if mm is None:
        mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=tuple(shape))
        prog_path.unlink(missing_ok=True)
    by_shard = {}
    for pos, (si, r, *_rest) in enumerate(picks):
        by_shard.setdefault(int(si), []).append((int(r), pos))
    todo = [si for si in sorted(by_shard) if shard_names[si] not in done]
    pool = _pool(workers, mp_context)
    parity_report = None
    t0, n_done = time.time(), 0
    from concurrent.futures import ThreadPoolExecutor

    fetcher = ThreadPoolExecutor(1)                 # downloads the next shard while this one decodes
    nxt = fetcher.submit(source.fetch, shard_names[todo[0]]) if todo else None
    try:
        for k, si in enumerate(todo):
            name = shard_names[si]
            path, was_cached = nxt.result()
            nxt = fetcher.submit(source.fetch, shard_names[todo[k + 1]]) if k + 1 < len(todo) else None
            rows = sorted(by_shard[si])
            data = read_rows(path, [r for r, _ in rows], image_col=source.image_col, extra_cols=extra_cols)
            imgs = data["images"]
            jobs = [(imgs[i:i + chunk], resize, crop) for i in range(0, len(imgs), chunk)]
            arrs = pool.map(decode_many, jobs) if pool is not None else [decode_many(j) for j in jobs]
            dec = np.concatenate(arrs) if arrs else np.zeros((0, crop, crop, 3), np.uint8)
            mm[np.array([pos for _, pos in rows], dtype=np.int64)] = dec
            for c in extra_cols:
                for (_, pos), v in zip(rows, data[c]):
                    extras[c][pos] = v
            if parity and parity_report is None:
                parity_report = parity_check(imgs[:parity], dec[:parity], resize, crop)
            mm.flush()
            done.add(name)
            _write_json(prog_path, {"shape": shape, "shards_done": sorted(done), "extras": extras})
            if delete_shards and not was_cached:
                source.release(name)
            n_done += len(rows)
            el = time.time() - t0
            log(f"{final.name}: shard {len(by_shard) - len(todo) + k + 1}/{len(by_shard)} {name} "
                f"({len(rows)} images; {n_done / max(el, 1e-9):.0f} images/s)")
    except BaseException:
        # An interrupted build keeps its finished shards (progress file) for the resume. Drop this frame's reference
        # to the memmap before the exception travels on: its traceback keeps the frame alive, and an open map of the
        # partial file would block the final rename of the resumed build on Windows.
        try:
            mm.flush()
        except Exception:
            pass
        del mm
        raise
    finally:
        if nxt is not None:
            nxt.cancel()
        fetcher.shutdown(wait=True)
        if pool is not None:
            pool.close()
            pool.join()
    mm.flush()
    del mm
    os.replace(partial, final)
    prog_path.unlink(missing_ok=True)
    return {"n": n, "shards": len(by_shard), "parity": parity_report, "extras": extras,
            "seconds": round(time.time() - t0, 1)}


def role_index(picks, shards, extras=None) -> dict:
    out = {"rows": np.arange(len(picks), dtype=np.int64),
           "labels": np.array([c for _, _, c in picks], dtype=np.int64).reshape(-1),
           "source": np.array([(si, r) for si, r, _ in picks], dtype=np.int64).reshape(-1, 2),
           "shards": np.array(list(shards), dtype=str),
           "ids": make_ids(picks, shards)}
    for k, v in (extras or {}).items():
        out[k] = np.array([-1 if x is None else x for x in v], dtype=np.int64)
    return out


def _role_complete(cache: Path, spec: RoleSpec) -> bool:
    try:
        meta = json.loads((cache / spec.meta_file).read_text())
    except (OSError, ValueError):
        return False
    return bool(meta.get("complete")) and (cache / spec.images_file).exists() and (cache / spec.index_file).exists()


def build_role(spec: RoleSpec, plan: dict, source, cache_dir, workers: int = 8, delete_shards: bool = False,
               parity: int = 32, extra_meta: dict | None = None, mp_context: str | None = None, log=_log) -> dict:
    """Decode one planned role into its memmap and write its index and meta (skips a complete role)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if _role_complete(cache, spec):
        log(f"{spec.name}: complete already ({spec.images_file})")
        return json.loads((cache / spec.meta_file).read_text())
    picks, shards = plan["picks"], plan["shards"]
    extra_cols = tuple(getattr(source, "extra_cols", ()))
    log(f"{spec.name}: {len(picks)} images from {len(shards)} shards ({shards[0]} .. {shards[-1]}), "
        f"{spec.crop} px, about {len(picks) * spec.crop * spec.crop * 3 / 1e9:.2f} GB decoded")
    res = decode_into(cache / spec.images_file, picks, shards, source, spec.resize, spec.crop, workers=workers,
                      delete_shards=delete_shards, parity=parity, extra_cols=extra_cols, mp_context=mp_context,
                      log=log)
    idx = role_index(picks, shards, res.get("extras"))
    np.savez(cache / spec.index_file, **idx)
    labels = idx["labels"]
    per = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(1, dtype=np.int64)
    meta = {"complete": True, "role": spec.name, "dataset": spec.dataset, "n": len(picks),
            "per_class": spec.per_class, "per_class_min": int(per.min()), "per_class_max": int(per.max()),
            "classes": int((per > 0).sum()), "seed": spec.seed, "first_shard_index": plan.get("first_shard_index"),
            "n_shards": len(shards), "shards": list(shards), "resize": spec.resize, "crop": spec.crop,
            "interpolation": "bicubic", "preprocessing": "PIL resize of the shorter side (bicubic), centre crop",
            "images_file": spec.images_file, "index_file": spec.index_file,
            "selection_sha256": selection_fingerprint(shards, picks), "parity": res.get("parity"),
            "decode_seconds": res.get("seconds"), "git": _git_hash(),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta.update(extra_meta or {})
    _write_json(cache / spec.meta_file, meta)
    log(f"{spec.name}: complete, selection sha256 {meta['selection_sha256'][:16]}")
    return meta


def existing_shard_lists(cache_dir) -> tuple:
    """(used shards, msel shards) recorded by the existing caches, or None where a cache is missing."""
    cache = Path(cache_dir)
    used = msel = None
    try:
        used = list(json.loads((cache / "train_meta.json").read_text())["shards_used"])
    except (OSError, ValueError, KeyError):
        pass
    if (cache / "msel_index.npz").exists():
        with np.load(cache / "msel_index.npz") as z:
            msel = [str(s) for s in z["msel_shards"]]
    return used, msel


def prefetching_labels(source, files, ahead: int = 16, threads: int = 8):
    """labels_of(f) that, on a miss, fetches the labels of f and the next `ahead` shards at once (when the source
    has labels_many); the selection only ever reads shards in file order."""
    if not hasattr(source, "labels_many"):
        return source.labels
    pos = {f: i for i, f in enumerate(files)}
    seen = set()

    def labels_of(f):
        if f not in seen and f in pos:
            i = pos[f]
            batch = [g for g in files[i:i + ahead] if g not in seen]
            source.labels_many(batch, threads)
            seen.update(batch)
        return source.labels(f)

    return labels_of


def build_imagenet_roles(names, source, cache_dir, workers: int = 8, delete_shards: bool = False, parity: int = 32,
                         chain=IMAGENET_CHAIN, n_classes: int = N_CLASSES, used_need: int = USED_PER_CLASS,
                         msel_per_class: int = MSEL_PER_CLASS, mp_context: str | None = None, log=_log) -> dict:
    cache = Path(cache_dir)
    used, msel = existing_shard_lists(cache)
    want = [s for s in chain if s.name in names]
    if not want:
        return {}
    files = source.files()
    plans = plan_imagenet(files, prefetching_labels(source, files), used, msel, chain=chain, upto=want[-1].name,
                          n_classes=n_classes, used_need=used_need, msel_per_class=msel_per_class)
    extra = {"repo": getattr(source, "repo", None), "revision": getattr(source, "revision", None),
             "shards_used_by_existing_roles": len(plans["_used"]), "msel_shards": len(plans["_msel"]),
             "existing_used_from_cache": used is not None, "existing_msel_from_cache": msel is not None}
    return {s.name: build_role(s, plans[s.name], source, cache, workers, delete_shards, parity, extra, mp_context,
                               log) for s in want}


def build_coco(source, cache_dir, spec: RoleSpec = COCO, workers: int = 8, delete_shards: bool = False,
               parity: int = 32, mp_context: str | None = None, log=_log) -> dict:
    cache = Path(cache_dir)
    if _role_complete(cache, spec):
        log(f"{spec.name}: complete already ({spec.images_file})")
        return json.loads((cache / spec.meta_file).read_text())
    plan = plan_coco(source.files(), source.n_rows, spec)
    extra = {"repo": getattr(source, "repo", None), "revision": getattr(source, "revision", None),
             "pool_rows": plan["pool_rows"], "pool_factor": COCO_POOL_FACTOR,
             "note": "COCO train images (verify: COCO 2017 train images with annotations), no class labels (-1)"}
    return build_role(spec, plan, source, cache, workers, delete_shards, parity, extra, mp_context, log)


def build_stream_index(source, cache_dir, threads: int = 8, log=_log) -> dict:
    """stream_train_index.npz: labels, source (index into shards, row), ids, shards, shard_rows, in stream order."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    files = source.files()
    labels = source.labels_many(files, threads) if hasattr(source, "labels_many") else \
        {f: source.labels(f) for f in files}
    rows = np.array([len(labels[f]) for f in files], dtype=np.int64)
    lab = np.concatenate([np.asarray(labels[f], dtype=np.int64) for f in files]) if files else np.zeros(0, np.int64)
    src = np.concatenate([np.stack([np.full(k, i, np.int64), np.arange(k, dtype=np.int64)], 1)
                          for i, k in enumerate(rows)]) if files else np.zeros((0, 2), np.int64)
    nums = np.array([shard_number(f) for f in files], dtype=np.int64)
    ids = np.stack([nums[src[:, 0]], src[:, 1]], 1) if len(src) else np.zeros((0, 2), np.int64)
    np.savez(cache / STREAM_INDEX, labels=lab, source=src, ids=ids, shards=np.array(files, dtype=str),
             shard_rows=rows)
    per = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(1, dtype=np.int64)
    meta = {"complete": True, "role": "stream_train", "n": int(len(lab)), "n_shards": len(files),
            "per_class_min": int(per.min()), "per_class_max": int(per.max()), "classes": int((per > 0).sum()),
            "repo": getattr(source, "repo", None), "revision": getattr(source, "revision", None),
            "order": "shards in file order, rows in shard order", "resize": 256, "crop": 224,
            "git": _git_hash(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(cache / STREAM_META, meta)
    log(f"stream_train index: {len(lab)} images in {len(files)} shards")
    return meta


# ----------------------------------------------------------------------------------------------
# The full-train stream
# ----------------------------------------------------------------------------------------------
def stream_batches(source, batch_size: int = 256, workers: int = 8, start_shard: int = 0, stop_shard: int | None = None,
                   files=None, resize: int = 256, crop: int = 224, delete_after: bool = False,
                   align_to_shards: bool = False, chunk: int = 64, window: int | None = None, prefetch: int = 1,
                   mp_context: str | None = None, log=None, log_every_s: float = 60.0
                   ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every row of shards [start_shard, stop_shard) in file order, then row order, as batches
    (uint8 [B, crop, crop, 3], labels int64 [B], ids int64 [B, 2] = (shard number, row)).

    One shard is downloaded at a time by a background thread (`prefetch` shards ahead); the encoded images of a
    shard are read into memory and the file is deleted at once with delete_after (only files this stream
    downloaded). Decoding runs in `workers` processes (forkserver on Linux, spawn elsewhere: safe in a process
    that uses CUDA) with at most `window` chunks of `chunk` images in flight; workers=0 decodes in this process.
    Batches span shard boundaries unless align_to_shards. The last batch may be smaller."""
    files = list(files) if files is not None else source.files()
    stop_shard = len(files) if stop_shard is None else min(stop_shard, len(files))
    order = list(range(start_shard, stop_shard))
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def downloader():
        try:
            for si in order:
                if stop.is_set():
                    return
                path, was = source.fetch(files[si])
                if not put(("shard", si, path, was)):
                    return
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            put(("error", exc))
            return
        put(("end",))

    th = threading.Thread(target=downloader, name="stream-download", daemon=True)
    th.start()
    label_col = getattr(source, "label_col", "label")

    def tasks():
        while True:
            item = q.get()
            if item[0] == "end":
                return
            if item[0] == "error":
                raise item[1]
            _, si, path, was = item
            data = read_rows(path, None, image_col=source.image_col, label_col=label_col)
            if delete_after and not was:
                source.release(files[si])
            imgs = data["images"]
            labs = data["labels"] if data["labels"] is not None else [-1] * len(imgs)
            num = shard_number(files[si])
            for r0 in range(0, len(imgs), chunk):
                r1 = min(len(imgs), r0 + chunk)
                yield (si, num, r0, r1, np.asarray(labs[r0:r1], dtype=np.int64), r1 == len(imgs)), imgs[r0:r1]

    pool = _pool(workers, mp_context)
    window = window or (1 if pool is None else max(2, 4 * workers))     # inline decoding gains nothing from more
    pending: deque = deque()
    xs, ys, idss, buffered = [], [], [], 0
    t0, n_out, last_log = time.time(), 0, time.time()

    def take(k):
        nonlocal xs, ys, idss, buffered
        x, y, i = np.concatenate(xs), np.concatenate(ys), np.concatenate(idss)
        xs, ys, idss = [x[k:]], [y[k:]], [i[k:]]
        buffered -= k
        return x[:k], y[:k], i[:k]

    try:
        it = tasks()
        exhausted = False
        while True:
            while not exhausted and len(pending) < window:
                try:
                    meta, payload = next(it)
                except StopIteration:
                    exhausted = True
                    break
                job = (payload, resize, crop)
                pending.append((meta, pool.apply_async(decode_many, (job,)) if pool is not None else decode_many(job)))
            if not pending:
                break
            meta, res = pending.popleft()
            arr = res.get() if pool is not None else res
            si, num, r0, r1, labs, last = meta
            xs.append(arr)
            ys.append(labs)
            idss.append(np.stack([np.full(r1 - r0, num, np.int64), np.arange(r0, r1, dtype=np.int64)], 1))
            buffered += r1 - r0
            while buffered >= batch_size:
                out = take(batch_size)
                n_out += len(out[1])
                yield out
            if align_to_shards and last and buffered:
                out = take(buffered)
                n_out += len(out[1])
                yield out
            if log is not None and time.time() - last_log >= log_every_s:
                el = time.time() - t0
                log(f"stream: {n_out} images in {el:.0f} s ({n_out / max(el, 1e-9):.0f} images/s), shard "
                    f"{si - start_shard + 1}/{len(order)}")
                last_log = time.time()
        if buffered:
            out = take(buffered)
            n_out += len(out[1])
            yield out
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        if pool is not None:
            pool.terminate()
            pool.join()
        th.join(timeout=5)


# ----------------------------------------------------------------------------------------------
# Role accessor
# ----------------------------------------------------------------------------------------------
class RowView:
    """Rows of a memmap that are not contiguous, read on indexing."""

    def __init__(self, mm, rows):
        self.mm, self.rows = mm, np.asarray(rows, dtype=np.int64)
        self.shape = (len(self.rows),) + tuple(mm.shape[1:])
        self.dtype = mm.dtype

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.mm[self.rows[i]]


def _view(mm, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
        return mm[int(rows[0]):int(rows[0]) + len(rows)]
    return RowView(mm, rows)


@dataclass
class Role:
    name: str
    dataset: str
    size: int
    images: object            # uint8 memmap view [N, size, size, 3]; None for stream_train
    labels: np.ndarray        # int64 [N]; -1 where there is no class label (coco)
    ids: np.ndarray           # int64 [N, 2]: (shard number, row); validation (-1, row in the cache)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def as_tuple(self):
        return self.images, self.labels, self.ids

    def batches(self, batch_size: int = 256, start: int = 0, stop: int | None = None, source=None, **stream_kw):
        """(uint8 [B, S, S, 3], labels, ids) in role order; stream_train streams from `source` (default: the hub)."""
        if self.images is None:
            src = source or imagenet_source()
            yield from stream_batches(src, batch_size=batch_size, **stream_kw)
            return
        stop = len(self) if stop is None else min(stop, len(self))
        for a in range(start, stop, batch_size):
            b = min(stop, a + batch_size)
            yield np.ascontiguousarray(self.images[a:b]), self.labels[a:b], self.ids[a:b]


def get_role(name: str, cache_dir=None) -> Role:
    """Any role, old or new (see ALL_ROLES). Raises FileNotFoundError when its files are missing."""
    cache = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"
    if name in OLD_TRAIN_ROLES:
        fname, src_key, size = OLD_TRAIN_ROLES[name]
        meta = json.loads((cache / "train_meta.json").read_text())
        shards = list(meta["shards_used"])
        with np.load(cache / "train_index.npz") as z:
            rows = np.asarray(z[f"{name}_rows"], dtype=np.int64)
            labels = np.asarray(z[f"{name}_labels"], dtype=np.int64)
            src = np.asarray(z[src_key], dtype=np.int64)[rows]
        nums = np.array([shard_number(s) for s in shards], dtype=np.int64)
        ids = np.stack([nums[src[:, 0]], src[:, 1]], 1) if len(src) else np.zeros((0, 2), np.int64)
        mm = np.load(cache / fname, mmap_mode="r")
        return Role(name, "imagenet_train", size, _view(mm, rows), labels, ids,
                    {"file": fname, "rows": [int(rows[0]), int(rows[-1]) + 1] if len(rows) else [0, 0]})
    if name == "msel":
        with np.load(cache / "msel_index.npz") as z:
            rows = np.asarray(z["msel_rows"], dtype=np.int64)
            labels = np.asarray(z["msel_labels"], dtype=np.int64)
            src = np.asarray(z["msel_source"], dtype=np.int64).reshape(-1, 2)
            shards = [str(s) for s in z["msel_shards"]]
        nums = np.array([shard_number(s) for s in shards], dtype=np.int64)
        ids = np.stack([nums[src[:, 0]], src[:, 1]], 1) if len(src) else np.zeros((0, 2), np.int64)
        mm = np.load(cache / "msel224_uint8.npy", mmap_mode="r")
        return Role(name, "imagenet_train", 224, _view(mm, rows), labels, ids, {"file": "msel224_uint8.npy"})
    if name == "val":
        mm = np.load(cache / "val224_uint8.npy", mmap_mode="r")
        labels = np.asarray(np.load(cache / "val_labels.npy"), dtype=np.int64)
        ids = np.stack([np.full(len(labels), -1, np.int64), np.arange(len(labels), dtype=np.int64)], 1)
        return Role(name, "imagenet_val", 224, mm, labels, ids, {"file": "val224_uint8.npy"})
    if name in NEW_ROLES:
        spec = NEW_ROLES[name]
        if not _role_complete(cache, spec):
            raise FileNotFoundError(f"role {name} is not built: run scripts/fp_cache.py build --roles {name}")
        with np.load(cache / spec.index_file) as z:
            rows = np.asarray(z["rows"], dtype=np.int64)
            labels = np.asarray(z["labels"], dtype=np.int64)
            ids = np.asarray(z["ids"], dtype=np.int64).reshape(-1, 2)
            extra = {k: z[k] for k in z.files if k not in ("rows", "labels", "ids", "source", "shards")}
        mm = np.load(cache / spec.images_file, mmap_mode="r")
        meta = json.loads((cache / spec.meta_file).read_text())
        meta.update({f"array_{k}": v for k, v in extra.items()})
        return Role(name, spec.dataset, spec.crop, _view(mm, rows), labels, ids, meta)
    if name == "stream_train":
        with np.load(cache / STREAM_INDEX) as z:
            labels = np.asarray(z["labels"], dtype=np.int64)
            ids = np.asarray(z["ids"], dtype=np.int64).reshape(-1, 2)
            shards = [str(s) for s in z["shards"]]
        return Role(name, "imagenet_train", 224, None, labels, ids, {"shards": shards})
    raise KeyError(f"unknown role {name!r}; roles: {', '.join(ALL_ROLES)}")


def role_arrays(name: str, cache_dir=None):
    """(images memmap or None, labels int64 [N], ids int64 [N, 2]) of any role."""
    return get_role(name, cache_dir).as_tuple()


def list_roles(cache_dir=None) -> list:
    out = []
    for name in ALL_ROLES:
        try:
            r = get_role(name, cache_dir)
            out.append({"role": name, "present": True, "n": len(r), "size": r.size, "dataset": r.dataset,
                        "streamed": r.images is None})
        except (OSError, KeyError, ValueError) as exc:
            out.append({"role": name, "present": False, "why": f"{type(exc).__name__}: {exc}"[:200]})
    return out


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def check_roles(cache_dir=None, out_path=None, expected=None, require=("fit3", "coco"), labels_json=None) -> dict:
    """Disjointness of every train role by (shard, row) ID, duplicates, sizes, label agreement with the shard
    labels (cache/train_shard_labels.json) and with stream_train, and stream_train's coverage. Writes JSON."""
    cache = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"
    expected = EXPECTED if expected is None else expected
    t0 = time.time()
    roles, missing, problems = {}, [], []
    for name in ALL_ROLES:
        try:
            roles[name] = get_role(name, cache)
        except (OSError, KeyError, ValueError):
            missing.append(name)
    for name in require:
        if name in missing:
            problems.append(f"required role {name} missing")
    info, keys = {}, {}
    for name, r in roles.items():
        k = id_keys(r.ids)
        keys[name] = k
        lab = r.labels
        per = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(1, dtype=np.int64)
        info[name] = {"n": len(r), "unique": int(len(np.unique(k))), "dataset": r.dataset,
                      "per_class_min": int(per.min()), "per_class_max": int(per.max()),
                      "expected": expected.get(name)}
        if info[name]["unique"] != info[name]["n"]:
            problems.append(f"{name}: duplicate IDs")
        if expected.get(name) is not None and len(r) != expected[name]:
            problems.append(f"{name}: {len(r)} images, expected {expected[name]}")
    train = [n for n in TRAIN_SET_ROLES if n in roles]
    overlaps = {}
    for i, a in enumerate(train):
        for b in train[i + 1:]:
            c = int(len(np.intersect1d(keys[a], keys[b])))
            overlaps[f"{a}|{b}"] = c
            if c:
                problems.append(f"{a} and {b} share {c} images")
    heal_total = None
    if "heal" in roles and "heal_ext" in roles:
        heal_total = int(len(np.union1d(keys["heal"], keys["heal_ext"])))
    stream = None
    if "stream_train" in roles:
        sk, sl = keys["stream_train"], roles["stream_train"].labels
        order = np.argsort(sk)
        sk_sorted, sl_sorted = sk[order], sl[order]
        stream = {"n": len(sk), "unique": info["stream_train"]["unique"], "roles": {}}
        for name in train:
            pos = np.searchsorted(sk_sorted, keys[name])
            pos = np.minimum(pos, len(sk_sorted) - 1) if len(sk_sorted) else pos
            inside = (sk_sorted[pos] == keys[name]) if len(sk_sorted) else np.zeros(len(keys[name]), bool)
            agree = inside & (sl_sorted[pos] == roles[name].labels) if len(sk_sorted) else inside
            stream["roles"][name] = {"inside": int(inside.sum()), "labels_agree": int(agree.sum())}
            if not inside.all() or not agree.all():
                problems.append(f"{name}: {int((~inside).sum())} IDs outside stream_train, "
                                f"{int((inside & ~agree).sum())} label mismatches")
    label_check = None
    lab_path = Path(labels_json) if labels_json else cache / "train_shard_labels.json"
    if lab_path.exists():
        shard_labels = json.loads(lab_path.read_text())
        by_num = {shard_number(s): v for s, v in shard_labels.items()}
        label_check = {}
        for name in train:
            r = roles[name]
            bad = checked = 0
            for (num, row), lab in zip(r.ids.tolist(), r.labels.tolist()):
                labs = by_num.get(num)
                if labs is not None:
                    checked += 1
                    bad += int(labs[row] != lab)
            label_check[name] = {"checked": checked, "mismatches": bad}
            if bad:
                problems.append(f"{name}: {bad} labels differ from the shard labels")
    report = {"ok": not problems, "problems": problems, "id": "(shard number, row in shard); key shard*1e6+row",
              "cache_dir": str(cache), "roles": info, "missing": missing, "pairwise_overlaps": overlaps,
              "heal_plus_heal_ext_distinct": heal_total, "stream_train": stream, "label_check": label_check,
              "note": "stream_train is the whole train split and overlaps every train role by design",
              "git": _git_hash(), "seconds": round(time.time() - t0, 1),
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        _write_json(Path(out_path), report)
    return report


# ----------------------------------------------------------------------------------------------
# Disk estimates
# ----------------------------------------------------------------------------------------------
def disk_estimates(cache_dir=None) -> list:
    """Per role: images, size, decoded bytes, shards to download and their bytes. Exact shard counts when
    cache/train_shard_labels.json already holds the labels needed (no network); otherwise estimates."""
    cache = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"
    exact = {}
    lab_path = cache / "train_shard_labels.json"
    if lab_path.exists():
        labels = json.loads(lab_path.read_text())
        files = sorted(labels)
        if len(files) >= 2 and all(f.startswith(TRAIN_PREFIX) for f in files):
            try:
                n_total = int(re.search(r"-of-(\d+)", files[0]).group(1))
                full = [f"{TRAIN_PREFIX}{i:05d}-of-{n_total:05d}.parquet" for i in range(n_total)]
                used, msel = existing_shard_lists(cache)
                plans = plan_imagenet(full, labels.__getitem__, used, msel)
                exact = {s.name: len(plans[s.name]["shards"]) for s in IMAGENET_CHAIN}
            except (KeyError, RuntimeError, AttributeError, ValueError):
                exact = {}
    rows = []

    def add(role, n, px, shards, shard_bytes, note, decoded=True):
        rows.append({"role": role, "images": n, "px": px,
                     "decoded_gb": round(n * px * px * 3 / 1e9, 2) if decoded else 0.0,
                     "shards": shards, "download_gb": None if shards is None else round(shards * shard_bytes / 1e9, 1),
                     "note": note})

    per_shard = N_TRAIN / N_TRAIN_SHARDS / N_CLASSES        # images of an average class in one shard
    rare = SMALLEST_CLASS / N_TRAIN_SHARDS                  # images of the smallest class in one shard
    f3 = exact.get("fit3")
    add("fit3", FIT3.n, FIT3.crop, f3 if f3 else 3, IMAGENET_SHARD_BYTES,
        "exact shard count" if f3 else "about 2 to 4 shards (estimate)")
    he = exact.get("heal_ext")
    add("heal_ext", HEAL_EXT.n, HEAL_EXT.crop, he if he else int(np.ceil(HEAL_EXT.per_class / rare)) + 5,
        IMAGENET_SHARD_BYTES, "exact shard count" if he else
        f"estimate: the smallest class has about {rare:.1f} images per shard (average {per_shard:.1f})")
    add("coco", COCO.n, COCO.crop, 2, COCO_SHARD_BYTES, "first 2 COCO train shards (about 2,930 rows each)")
    add("stream_train", N_TRAIN, 224, N_TRAIN_SHARDS, IMAGENET_SHARD_BYTES,
        "streamed, nothing decoded is stored; with delete_after about 2 shards (1 GB) on disk at a time beyond "
        "those already cached; materialised it would take 192.9 GB", decoded=False)
    rows.append({"role": "stream_train_index", "images": N_TRAIN, "px": 0, "decoded_gb": 0.03, "shards": 0,
                 "download_gb": 0.0, "note": "labels and IDs only (label columns read remotely)"})
    for role, n, px in (("fit0..fit2 + bank (existing)", 106_000, 224), ("healval + heal (existing)", 105_000, 256),
                        ("msel (existing)", 10_000, 224), ("val (existing)", 50_000, 224)):
        rows.append({"role": role, "images": n, "px": px, "decoded_gb": round(n * px * px * 3 / 1e9, 2),
                     "shards": None, "download_gb": None, "note": "existing cache, reused"})
    return rows


def format_estimates(rows) -> str:
    lines = [f"{'role':32s} {'images':>9s} {'px':>4s} {'decoded GB':>10s} {'shards':>6s} {'download GB':>11s}  note"]
    for r in rows:
        sh = "-" if r["shards"] is None else str(r["shards"])
        dl = "-" if r["download_gb"] is None else f"{r['download_gb']:.1f}"
        lines.append(f"{r['role']:32s} {r['images']:9d} {r['px']:4d} {r['decoded_gb']:10.2f} {sh:>6s} {dl:>11s}  "
                     f"{r['note']}")
    new = sum(r["decoded_gb"] for r in rows if r["role"] in NEW_ROLES)
    dl = sum(r["download_gb"] or 0 for r in rows if r["role"] in NEW_ROLES)
    lines.append(f"new decoded caches: {new:.1f} GB; their shard downloads: about {dl:.0f} GB "
                 f"(freed with --delete-shards once each role is complete)")
    return "\n".join(lines)


def scaled_spec(spec: RoleSpec, **kw) -> RoleSpec:
    """A copy with other sizes (tests and dry runs)."""
    return replace(spec, **kw)


def write_synthetic_shards(folder, n_shards: int, rows_per_shard: int, n_classes: int, seed: int = 0,
                           size_range=(20, 48), labels: bool = True, image_id: bool = False,
                           prefix: str = TRAIN_PREFIX, fmt: str = "PNG") -> list:
    """Parquet shards in the hub layout (image struct {bytes, path}, label, optional image_id) with random images of
    random sizes, for tests and dry runs. Returns the shard names (relative to `folder`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    folder = Path(folder)
    rng = np.random.default_rng(seed)
    names = []
    for i in range(n_shards):
        name = f"{prefix}{i:05d}-of-{n_shards:05d}.parquet"
        imgs, labs = [], []
        for _ in range(rows_per_shard):
            w, h = (int(v) for v in rng.integers(size_range[0], size_range[1], size=2))
            arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format=fmt)
            imgs.append({"bytes": buf.getvalue(), "path": None})
            labs.append(int(rng.integers(n_classes)))
        cols = {"image": imgs}
        if labels:
            cols["label"] = labs
        if image_id:
            cols["image_id"] = [i * 100_000 + r for r in range(rows_per_shard)]
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), folder / name)
        names.append(name)
    return names

#!/usr/bin/env python
"""Tier 2: ADE20K linear segmentation (PAPER_EXPERIMENTS.md, Section 1, Tier 2: "ADE20K linear
segmentation, short protocol: frozen backbone, linear head on last-layer patch tokens, 448 x 448 crops,
8,000 iterations, batch 16, mIoU. The same protocol for every variant.").

    export HF_HOME=/home/dawar/hf
    python scripts/tier2_ade20k.py --model dinov2_vitb14_reg \
        --variants original,healed_mirror_k9,healed_svd_k9,healed_random_k9,vexact_k6,mirror_mlp_k6,svd_mlp_k6 \
        --out runs/t2_seg
    python scripts/tier2_ade20k.py --tiny --out tiny_out/t2_seg       # CPU smoke test (synthetic parquet)

Data (downloaded on the VM only, under $HF_HOME). ADE20K scene parsing (SceneParsing / ADEChallengeData2016,
150 classes) from the Hugging Face dataset merve/scene_parse_150 at the pinned revision below: a parquet
copy of the official scene_parse_150 "scene_parsing" configuration, not gated, columns image, annotation,
scene_category; train 20,210, validation 2,000 (test 3,352 without labels, not used). Checked on the
Hugging Face API on 2026-11-08: danjacobellis/scene_parse_150 holds byte-identical parquet shards (same LFS
sha256), so either id gives the same data. The dataset card states that annotation values run from 0 to
150 and that 0 ("other objects") is not part of the official evaluation. That this copy equals the
official release image for image has not been checked (verify on VM). The shards are fetched with
huggingface_hub.snapshot_download (repo_type "dataset", revision pinned, only the train and validation
shards, about 0.97 GB), their sha256 is checked against the values recorded below, and they are decoded
once into <cache-dir>/ade20k/:
  train  image resized (Pillow bilinear) so its shorter side is --crop px, label map resized to the same
         size (nearest); both stored back to back in flat uint8 files plus an index (offsets and sizes);
  val    image resized the same way; label map kept at its ORIGINAL resolution (metrics are computed
         there, as in the usual mmsegmentation evaluation).
Rough size at 448 px (estimate, not measured): 20 to 25 GB for train, 2 to 3 GB for validation; the real
sizes are recorded in meta.json. Labels stay raw (0..150) in
the cache; the loaders map 0 -> 255 (ignored) and c -> c - 1 (classes 0..149).

Protocol (one fixed recipe and one seed for every variant; nothing is tuned on validation):
  features   frozen backbone, no gradients, bf16 autocast; the last block's patch tokens after the final
             norm (CLS and register tokens excluded), [B, d, 32, 32] for 448 px at patch 14. Nothing is
             cached: features are recomputed for every batch.
  head       BatchNorm2d(d) followed by a 1 x 1 convolution to 150 classes (a linear map on each patch
             token). The BatchNorm follows the DINOv2 linear segmentation head (BNHead: batch norm, then a
             1 x 1 conv) (verify against the DINOv2 configs); it also makes one learning rate fit every
             variant whatever its feature scale. At evaluation it uses running statistics, so the head is
             an affine map of the features. Conv weight initialised N(0, 0.01^2) with a generator seeded by
             --seed, bias 0, so every variant starts from the same head. --no-bn drops the BatchNorm.
  training   random 448 x 448 crop of the resized train image (padding with ignored pixels only if the image
             is smaller), random horizontal flip (p = 0.5), batch 16, 8,000 iterations (about 6.3 epochs);
             the crop positions, flips and image order come from a schedule drawn once from --seed, so
             every variant sees exactly the same crops in the same order. No scale jitter and no colour
             augmentation (a simplification of the mmsegmentation recipe DINOv2 uses (verify)).
  loss       logits [B, 150, 32, 32] upsampled bilinearly (align_corners False) to 448 x 448, cross-entropy
             averaged over labelled pixels (ignore index 255).
  optimiser  AdamW (betas 0.9, 0.999), lr 1e-3, weight decay 1e-4 on the conv weight only (not on biases or
             BatchNorm), linear warm-up over 300 iterations, then cosine decay to 0. These values are a
             fixed choice for this short protocol, not tuned.
  evaluation all 2,000 validation images; the image (shorter side 448) is covered by 448 x 448 sliding
             windows with stride 298 (about 2/3 of the crop, the same ratio as the mmsegmentation 512 / 341
             slide setting (verify)); window logits are upsampled to 448 x 448 and averaged where windows
             overlap; the full logit map is upsampled bilinearly to the original label resolution and the
             argmax is scored. Single scale, no flip. Metrics from the summed confusion matrix over
             labelled pixels: mIoU (mean over classes with a non-empty union), aAcc (pixel accuracy) and
             mAcc (mean class accuracy), all in %.
  numerics   TF32 off; head, loss and metrics in float32; GPU memory capped at --memory-fraction (0.3).
             CUDA's bilinear-upsampling backward uses atomic adds, so a rerun on the GPU can differ in the
             last digits (CPU runs are bit-reproducible).

Variants: the names and builders of scripts/tier2_linear.py (VariantFactory; nothing reimplemented):
original, mirror_mlp_k<K>, svd_mlp_k<K>, random_k<K>, vlocal_k<K>, vexact_k<K>, svd_attn_k<K>,
random_attn_k<K>, healed_<mirror|svd|random>_k<K>. The adapter is built at the crop size (448 px), so
every variant runs on 32 x 32 patch grids; DINOv2 interpolates its position embedding (stored for 37 x 37)
to the input grid on the fly.

V-exact at 448 px. mirror.m1.build_vexact writes the position embedding interpolated to the adapter's grid
into the model and projects THAT grid onto the mirror-equivariant form (DINOv2's bicubic interpolation with
interpolate_offset = 0.1 is not symmetric under a left-right flip, so the projection must act on the grid in
use). A V-exact model built at 224 px therefore carries a projected 16 x 16 grid, and at 448 px DINOv2 would
re-interpolate it to 32 x 32, which breaks exactness. Here V-exact is rebuilt with the same recipe at the
segmentation resolution: the same span operator A*, neuron pairings, head maps and register pairing (all
resolution-free), with the 32 x 32 grid materialised and projected. Every forward pass is then exactly
448 x 448 (training crops and sliding windows alike), so the model never re-interpolates. The script
checks this: on two training crops it measures the mirrored-token error inside the span (mirror.m1.
stream_equivariance, fp32; M1's bar is a relative error of 1e-4) for the 448 px build, and, for contrast,
for the 224 px build run at 448 px. Both numbers go into the variant's JSON and the summary. V-local, MLP-only,
SVD, random and healed variants change block weights only, so they are the same model at any resolution;
healed students were distilled at 224 px and are evaluated as they are.

Outputs under --out (default runs/t2_seg): t2_ade20k.csv (one row per model and variant: mIoU, aAcc, mAcc,
iterations, seed, git hash, timings and more), T2_ADE20K.md, seg/<model>/<variant>.json (row, settings,
per-class IoU and accuracy, training history, variant provenance, V-exact checks), <variant>_head.pt (the
trained head), <variant>_confusion.npy, <variant>_train_log.csv, and t2_ade20k_meta_<model>.json.
Resume: a variant whose JSON holds a row made with the same settings is skipped; a trained head with the
same settings is reused (evaluation only); an interrupted training run continues from its last checkpoint
(<variant>_ckpt.pt, every 1,000 iterations; the crop schedule makes the continuation identical). --force
recomputes everything.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import normalize_for  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402


def _load_script(name: str):
    """Import scripts/<name>.py once under the key the other Tier-2 scripts use."""
    key = f"_t2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


T2 = _load_script("tier2_linear")        # variant names, VariantFactory, adapters, JSON helpers
T2T = _load_script("tier2_transfer")     # parquet reading helpers
DEFAULT_MEMORY_FRACTION = 0.3
T2.apply_memory_cap(DEFAULT_MEMORY_FRACTION)   # the imports above set their own caps; ours wins

RECIPE = "t2seg-v1"
IGNORE = 255
CACHE_FORMAT = 1
VEXACT_BAR = 1e-4            # M1's exactness bar (max relative error inside the span, fp32)

# Hugging Face ids, revisions and parquet sha256 values read from the Hugging Face API on 2026-11-08.
ADE20K = {
    "name": "ADE20K scene parsing (150 classes)",
    "hf_id": "merve/scene_parse_150",
    "revision": "b83d71b405b381bc5e495f7d497fb844a8de5ffc",
    "splits": {"train": "train", "val": "validation"},
    "image": "image",
    "label": "annotation",
    "num_classes": 150,
    "counts": {"train": 20210, "val": 2000},
    "sha256": {
        "data/train-00000-of-00002.parquet": "ff5f17abd110804cfd461e799b67833f18f88d2bd2e0881d62b2a46c0c63bf9f",
        "data/train-00001-of-00002.parquet": "4ff6104b02fd05a09ebbb91150f86cd2d412a93b1503beb10becc576e2d6c190",
        "data/validation-00000-of-00001.parquet": "9da4450269c4682221f1c1fc993ebc46018875783decae3fbfaf6cac88ed2108",
    },
    "cache_name": "ade20k",
    "note": "parquet copy of the official scene_parse_150 'scene_parsing' configuration; label 0 = other "
            "objects (ignored), 1..150 = classes; identity with the official release not checked (verify on VM)",
}
KNOWN_REVISIONS = {
    "merve/scene_parse_150": "b83d71b405b381bc5e495f7d497fb844a8de5ffc",
    "danjacobellis/scene_parse_150": "bbcc97e31fc582a2fe51f836cd6dccc472232aba",   # same LFS sha256
}

DEFAULTS = {   # normal run / --tiny
    "crop": (448, 56), "iters": (8000, 20), "batch_size": (16, 4), "eval_stride": (298, 42),
    "warmup_iters": (300, 2), "num_workers": (6, 0), "decode_workers": (8, 1), "eval_batch": (16, 4),
    "log_every": (100, 5), "ckpt_every": (1000, 10),
}


# ----------------------------------------------------------------------------------------------
# Labels and metrics
# ----------------------------------------------------------------------------------------------
def label_lut(num_classes: int) -> np.ndarray:
    """Raw ADE20K label -> training label: 0 (other objects) -> 255 (ignored), c -> c - 1 for c in
    1..num_classes, anything above num_classes -> 255."""
    lut = np.full(256, IGNORE, dtype=np.uint8)
    lut[1:num_classes + 1] = np.arange(num_classes, dtype=np.uint8)
    return lut


def seg_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE) -> torch.Tensor:
    """Cross-entropy averaged over labelled pixels (same as the default mean with ignore_index); a batch
    without labelled pixels gives 0 (and zero gradients) instead of NaN."""
    total = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="sum")
    return total / (labels != ignore_index).sum().clamp_min(1)


def confusion_update(conf: torch.Tensor, pred: torch.Tensor, label: torch.Tensor, num_classes: int,
                     ignore_index: int = IGNORE) -> None:
    """conf (flat, num_classes^2, int64; row = label, column = prediction) += counts of labelled pixels."""
    m = label != ignore_index
    idx = label[m].long() * num_classes + pred[m].long()
    conf += torch.bincount(idx, minlength=num_classes * num_classes).to(conf.dtype)


def seg_metrics(conf) -> dict:
    """mIoU, aAcc and mAcc (%) from a confusion matrix [C, C] (row = label, column = prediction).
    IoU_c = TP / (TP + FP + FN); classes with an empty union are left out of the mean (NaN), classes that
    are predicted but absent from the labels score 0. mAcc averages TP / (labelled pixels of c) over
    classes present in the labels."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    area_label, area_pred = conf.sum(1), conf.sum(0)
    union = area_label + area_pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(area_label > 0, tp / area_label, np.nan)
    total = float(area_label.sum())

    def nanmean(v):
        return float(np.nanmean(v)) * 100.0 if np.isfinite(v).any() else float("nan")

    return {"mIoU": nanmean(iou), "aAcc": 100.0 * float(tp.sum()) / total if total > 0 else float("nan"),
            "mAcc": nanmean(acc), "iou": iou, "acc": acc, "n_classes_scored": int((union > 0).sum()),
            "n_classes_in_labels": int((area_label > 0).sum()), "n_pixels": int(total)}


# ----------------------------------------------------------------------------------------------
# Data: download, decode once, ragged uint8 store
# ----------------------------------------------------------------------------------------------
def resized_hw(h: int, w: int, short: int) -> tuple:
    """Size with the shorter side equal to `short` and the aspect ratio kept (rounded)."""
    if h <= w:
        return short, max(short, int(round(w * short / h)))
    return max(short, int(round(h * short / w))), short


def label_array(img) -> np.ndarray:
    """Single-channel label image -> uint8 [H, W]."""
    if img.mode in ("L", "P"):
        return np.asarray(img, dtype=np.uint8)
    if img.mode.startswith("I"):
        arr = np.asarray(img)
        if arr.size and (arr.min() < 0 or arr.max() > 255):
            raise ValueError(f"label values outside 0..255 in a mode {img.mode} annotation")
        return arr.astype(np.uint8)
    raise ValueError(f"annotation mode {img.mode}: expected a single-channel label map (L, P or I)")


def decode_seg_pair(args) -> tuple:
    """(image bytes, label bytes, short side, resize label?) -> (image uint8 [h, w, 3], label uint8,
    original (H, W), label size differed from the image?). The image is resized so its shorter side is
    `short` (Pillow bilinear); the label is resized with nearest neighbour when `resize label` is true and
    kept at the original resolution otherwise."""
    from PIL import Image

    img_b, lab_b, short, resize_label = args
    R = getattr(Image, "Resampling", Image)
    img = Image.open(io.BytesIO(img_b)).convert("RGB")
    lab = Image.open(io.BytesIO(lab_b))
    lab.load()
    W, H = img.size
    mismatch = lab.size != img.size
    if mismatch:
        lab = lab.resize(img.size, R.NEAREST)
    nh, nw = resized_hw(H, W, int(short))
    im = np.asarray(img.resize((nw, nh), R.BILINEAR), dtype=np.uint8)
    lb = label_array(lab.resize((nw, nh), R.NEAREST) if resize_label else lab)
    return np.ascontiguousarray(im), np.ascontiguousarray(lb), (int(H), int(W)), bool(mismatch)


INDEX_KEYS = ("img_off", "img_h", "img_w", "lab_off", "lab_h", "lab_w", "orig_h", "orig_w")


def build_seg_split(batches, n_total: int, out_dir: Path, split: str, short: int, resize_label: bool,
                    workers: int = 8) -> dict:
    """Decode (image, annotation) batches into <split>_images.u8 / <split>_labels.u8 (back to back) and
    <split>_index.npz, written atomically. Returns counts and the raw label histogram."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = {k: out_dir / f"{split}_{k}.u8" for k in ("images", "labels")}
    partial = {k: p.with_name(p.name + ".partial") for k, p in final.items()}
    idx = {k: np.zeros(n_total, dtype=np.int64) for k in INDEX_KEYS}
    mismatch = np.zeros(n_total, dtype=bool)
    hist = np.zeros(256, dtype=np.int64)
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn" if os.name == "nt" else "fork").Pool(workers)
    pos = img_pos = lab_pos = 0
    t0 = time.time()
    try:
        with open(partial["images"], "wb") as fi, open(partial["labels"], "wb") as fl:
            for imgs, labs in batches:
                jobs = [(T2T._image_bytes(a), T2T._image_bytes(b), short, resize_label) for a, b in zip(imgs, labs)]
                res = pool.map(decode_seg_pair, jobs, chunksize=4) if pool else [decode_seg_pair(j) for j in jobs]
                for im, lb, (H, W), mm in res:
                    if pos >= n_total:
                        raise RuntimeError(f"{split}: more rows than the expected {n_total}")
                    fi.write(im.tobytes())
                    fl.write(lb.tobytes())
                    idx["img_off"][pos], (idx["img_h"][pos], idx["img_w"][pos]) = img_pos, im.shape[:2]
                    idx["lab_off"][pos], (idx["lab_h"][pos], idx["lab_w"][pos]) = lab_pos, lb.shape
                    idx["orig_h"][pos], idx["orig_w"][pos] = H, W
                    mismatch[pos] = mm
                    hist += np.bincount(lb.ravel(), minlength=256)
                    img_pos += im.size
                    lab_pos += lb.size
                    pos += 1
                if pos and pos % 2000 < len(res):
                    log(f"  {split}: {pos}/{n_total} decoded, {pos / max(time.time() - t0, 1e-9):.0f} img/s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if pos != n_total:
        raise RuntimeError(f"{split}: decoded {pos} rows, expected {n_total}")
    for k in final:
        os.replace(partial[k], final[k])
    np.savez(out_dir / f"{split}_index.npz", label_mismatch=mismatch, **idx)
    return {"n": int(pos), "image_bytes": int(img_pos), "label_bytes": int(lab_pos),
            "label_histogram": hist.tolist(), "label_size_mismatches": int(mismatch.sum()),
            "label_resolution": "resized (nearest)" if resize_label else "original", "seconds": round(time.time() - t0, 1)}


def download_parquet(entry: dict, verify_sha: bool = True) -> tuple:
    """snapshot_download of the train and validation shards at the pinned revision (under $HF_HOME);
    returns ({hf split: [paths]}, {relative path: sha256})."""
    from huggingface_hub import snapshot_download

    splits = list(entry["splits"].values())
    pats = [f"*{s}-*.parquet" for s in splits] + [f"*{s}.parquet" for s in splits]
    local = Path(snapshot_download(repo_id=entry["hf_id"], repo_type="dataset", revision=entry["revision"],
                                   allow_patterns=pats))
    found = sorted(local.rglob("*.parquet"))
    files = {s: T2T.parquet_files_for_split(found, s) for s in splits}
    for s, fs in files.items():
        if not fs:
            raise FileNotFoundError(f"{entry['hf_id']}@{entry['revision']}: no parquet shard for split {s!r} in {local}")
    hashes = {}
    for f in found:
        rel = Path(f).relative_to(local).as_posix()
        hashes[rel] = T2.sha256(f)
        want = entry.get("sha256", {}).get(rel)
        if want is None:
            log(f"WARNING: no recorded sha256 for {rel}")
        elif hashes[rel] != want:
            msg = f"{rel}: sha256 {hashes[rel]} differs from the recorded {want}"
            if verify_sha:
                raise RuntimeError(msg + " (pass --skip-sha-check to use it anyway)")
            log("WARNING: " + msg)
    return files, hashes


class RaggedStore:
    """Images (uint8 [h, w, 3]) and raw label maps (uint8 [h', w']) of one split, stored back to back in
    flat files and indexed by offsets. The files are opened lazily as read-only memmaps (and dropped when
    the store is pickled), so DataLoader workers map them instead of copying them."""

    def __init__(self, root, split: str):
        self.root, self.split = Path(root), split
        with np.load(self.root / f"{split}_index.npz") as z:
            self.index = {k: np.asarray(z[k]) for k in z.files}
        self._img = self._lab = None
        self._in_memory = False

    @classmethod
    def from_arrays(cls, images, labels) -> "RaggedStore":
        """In-memory store from lists of arrays (tests)."""
        st = cls.__new__(cls)
        st.root, st.split, st._in_memory = None, "memory", True
        n = len(images)
        idx = {k: np.zeros(n, dtype=np.int64) for k in INDEX_KEYS}
        io_, lo = 0, 0
        for i, (im, lb) in enumerate(zip(images, labels)):
            idx["img_off"][i], (idx["img_h"][i], idx["img_w"][i]) = io_, im.shape[:2]
            idx["lab_off"][i], (idx["lab_h"][i], idx["lab_w"][i]) = lo, lb.shape
            idx["orig_h"][i], idx["orig_w"][i] = lb.shape
            io_ += im.size
            lo += lb.size
        st.index = idx
        st._img = np.concatenate([np.asarray(im, np.uint8).ravel() for im in images])
        st._lab = np.concatenate([np.asarray(lb, np.uint8).ravel() for lb in labels])
        return st

    def __len__(self) -> int:
        return int(len(self.index["img_off"]))

    def __getstate__(self):
        d = self.__dict__.copy()
        if not self._in_memory:
            d["_img"] = d["_lab"] = None
        return d

    def _open(self) -> None:
        if self._img is None:
            self._img = np.memmap(self.root / f"{self.split}_images.u8", dtype=np.uint8, mode="r")
            self._lab = np.memmap(self.root / f"{self.split}_labels.u8", dtype=np.uint8, mode="r")

    def image(self, i: int) -> np.ndarray:
        self._open()
        o, h, w = (int(self.index[k][i]) for k in ("img_off", "img_h", "img_w"))
        return self._img[o:o + h * w * 3].reshape(h, w, 3)

    def label(self, i: int) -> np.ndarray:
        self._open()
        o, h, w = (int(self.index[k][i]) for k in ("lab_off", "lab_h", "lab_w"))
        return self._lab[o:o + h * w].reshape(h, w)


def ensure_seg_cache(entry: dict, cache_root, short: int, workers: int = 8, files: dict | None = None,
                     verify_sha: bool = True) -> dict:
    """Build (once) the decoded cache of `entry` and return {"train": RaggedStore, "val": RaggedStore,
    "meta", "dir"}. `files` ({hf split: [parquet paths]}) skips the download (tests, --tiny)."""
    out = Path(cache_root) / entry.get("cache_name", "ade20k")
    meta_path = out / "meta.json"
    want = {"format": CACHE_FORMAT, "hf_id": entry["hf_id"], "revision": entry["revision"], "splits": entry["splits"],
            "image": entry["image"], "label": entry["label"], "short_side": int(short)}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not (meta.get("complete") and all(meta.get(k) == v for k, v in want.items())):
        t0 = time.time()
        hashes = {}
        if files is None:
            files, hashes = download_parquet(entry, verify_sha)
        info, counts = {}, {}
        for key, hf_split in entry["splits"].items():
            n = T2T._count_rows(files[hf_split])
            exp = entry.get("counts", {}).get(key)
            if exp is not None and n != exp:
                raise ValueError(f"{entry['hf_id']} {hf_split}: {n} rows, expected {exp} (verify the dataset)")
            log(f"  {entry['hf_id']} {hf_split}: decoding {n} images (shorter side {short}, "
                f"label {'resized' if key == 'train' else 'at original size'})")
            batches = T2T.iter_parquet(files[hf_split], entry["image"], entry["label"], batch_rows=64)
            info[key] = build_seg_split(batches, n, out, key, short, resize_label=(key == "train"), workers=workers)
            counts[key] = n
            hist = np.asarray(info[key]["label_histogram"])
            if hist[entry["num_classes"] + 1:].any():
                bad = np.flatnonzero(hist[entry["num_classes"] + 1:]) + entry["num_classes"] + 1
                raise ValueError(f"{hf_split}: label values {bad[:10].tolist()} above {entry['num_classes']}")
        meta = {**want, "complete": True, "counts": counts, "decode": info, "parquet_sha256": hashes,
                "sources": {s: [os.path.basename(f) for f in fs] for s, fs in files.items()},
                "num_classes": entry["num_classes"], "interpolation": "image bilinear (Pillow), label nearest",
                "seconds": round(time.time() - t0, 1), "note": entry.get("note", "")}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"train": RaggedStore(out, "train"), "val": RaggedStore(out, "val"), "meta": meta, "dir": str(out)}


# ----------------------------------------------------------------------------------------------
# Training crops (a fixed schedule drawn from the seed)
# ----------------------------------------------------------------------------------------------
def make_schedule(n_images: int, total: int, seed: int) -> dict:
    """Image order (one fresh permutation per epoch), crop positions in [0, 1) and flips for `total`
    samples. One generator per stream (order, rows, columns, flips), so a shorter schedule is a prefix of
    a longer one."""
    rng_o, rng_y, rng_x, rng_f = (np.random.default_rng([int(seed), i]) for i in range(4))
    order, have = [], 0
    while have < total:
        order.append(rng_o.permutation(n_images))
        have += n_images
    return {"index": np.concatenate(order)[:total].astype(np.int64), "uy": rng_y.random(total),
            "ux": rng_x.random(total), "flip": rng_f.random(total) < 0.5}


def crop_flip(img: np.ndarray, lab: np.ndarray, crop: int, uy: float, ux: float, flip: bool, lut: np.ndarray):
    """crop x crop window at relative position (uy, ux) of an image and its raw label map (same size),
    padded (image 0, label raw 0 = ignored) if the image is smaller, optionally mirrored left-right.
    Returns (image uint8 [3, crop, crop], label uint8 [crop, crop] mapped by `lut`)."""
    h, w = lab.shape
    if img.shape[:2] != (h, w):
        raise ValueError(f"image {img.shape[:2]} and label {lab.shape} differ in size")
    my, mx = max(h - crop, 0), max(w - crop, 0)
    y0 = min(int(uy * (my + 1)), my)
    x0 = min(int(ux * (mx + 1)), mx)
    ci, cl = img[y0:y0 + crop, x0:x0 + crop], lab[y0:y0 + crop, x0:x0 + crop]
    if ci.shape[0] < crop or ci.shape[1] < crop:
        pi, pl = np.zeros((crop, crop, 3), np.uint8), np.zeros((crop, crop), np.uint8)
        pi[:ci.shape[0], :ci.shape[1]], pl[:cl.shape[0], :cl.shape[1]] = ci, cl
        ci, cl = pi, pl
    if flip:
        ci, cl = ci[:, ::-1], cl[:, ::-1]
    return torch.from_numpy(np.array(ci, order="C", copy=True)).permute(2, 0, 1), torch.from_numpy(lut[cl])


class TrainCrops(Dataset):
    """Item i = sample (start + i) of the schedule."""

    def __init__(self, store: RaggedStore, schedule: dict, crop: int, lut: np.ndarray, start: int = 0):
        self.store, self.schedule, self.crop, self.lut, self.start = store, schedule, int(crop), lut, int(start)

    def __len__(self) -> int:
        return len(self.schedule["index"]) - self.start

    def __getitem__(self, i: int):
        i = i + self.start
        s = self.schedule
        j = int(s["index"][i])
        return crop_flip(self.store.image(j), self.store.label(j), self.crop, float(s["uy"][i]), float(s["ux"][i]),
                         bool(s["flip"][i]), self.lut)


class ValImages(Dataset):
    """(image uint8 [h, w, 3], label uint8 [H, W] mapped by `lut`) for the first n images."""

    def __init__(self, store: RaggedStore, n: int, lut: np.ndarray):
        self.store, self.n, self.lut = store, int(n), lut

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return torch.from_numpy(np.array(self.store.image(i))), torch.from_numpy(self.lut[self.store.label(i)])


# ----------------------------------------------------------------------------------------------
# Backbone features and head
# ----------------------------------------------------------------------------------------------
def make_feature_fn(adapter, model, device, amp: str | None = "bf16"):
    """uint8 images [B, 3, H, W] on `device` -> float32 patch-token map [B, d, H/p, W/p]: the last block's
    tokens after the final norm, CLS and registers dropped (adapter.n_prefix_tokens), row-major grid.
    bf16 autocast on CUDA only. Call under torch.no_grad()."""
    device = torch.device(device)
    use_amp = amp == "bf16" and device.type == "cuda"
    p, ps = int(adapter.n_prefix_tokens), int(adapter.patch_size)

    def fn(x_uint8: torch.Tensor) -> torch.Tensor:
        x = normalize_for(adapter, x_uint8)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            t = model.norm(adapter.forward_tokens(x, model))
        B, _, H, W = x.shape
        gh, gw = H // ps, W // ps
        pt = t[:, p:]
        if pt.shape[1] != gh * gw:
            raise RuntimeError(f"{pt.shape[1]} patch tokens for a {gh} x {gw} grid")
        return pt.float().transpose(1, 2).reshape(B, pt.shape[-1], gh, gw)

    return fn


class LinearSegHead(nn.Module):
    """BatchNorm2d(d) (optional) then a 1 x 1 convolution to the classes: a linear map per patch token."""

    def __init__(self, dim: int, num_classes: int, bn: bool = True, init_std: float = 0.01, seed: int = 0):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim) if bn else nn.Identity()
        self.conv = nn.Conv2d(dim, num_classes, kernel_size=1)
        g = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * float(init_std))
            self.conv.bias.zero_()

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv(self.bn(f))


@dataclass
class TrainCfg:
    iters: int = 8000
    batch: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 300
    min_lr_ratio: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    log_every: int = 100
    ckpt_every: int = 1000


def lr_at(t: int, cfg: TrainCfg) -> float:
    """Linear warm-up over cfg.warmup iterations, then cosine decay to min_lr_ratio * lr at the end."""
    if t < cfg.warmup:
        return cfg.lr * (t + 1) / cfg.warmup
    p = (t - cfg.warmup) / max(cfg.iters - cfg.warmup, 1)
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0))))


def make_optimizer(head: LinearSegHead, cfg: TrainCfg) -> torch.optim.Optimizer:
    decay = [head.conv.weight]
    rest = [p for n, p in head.named_parameters() if n != "conv.weight"]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": rest, "weight_decay": 0.0}], lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def _atomic_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_head(head: LinearSegHead, feature_fn, store: RaggedStore, crop: int, lut: np.ndarray, cfg: TrainCfg,
               device, seed: int, ckpt_path=None, settings=None, num_workers: int = 0, pin: bool = False) -> dict:
    """Train `head` in place on crops from `store` (schedule from `seed`). With `ckpt_path`, saves the head,
    optimiser and history every cfg.ckpt_every iterations and continues from a checkpoint made with the
    same `settings`. Returns the history (every cfg.log_every iterations) and timings."""
    device = torch.device(device)
    head.to(device).train()
    opt = make_optimizer(head, cfg)
    schedule = make_schedule(len(store), cfg.iters * cfg.batch, seed)
    t0_iter, history = 0, []
    run_sum, run_n = 0.0, 0                     # loss sum and count of the current logging window
    settings_j = T2._jsonable(settings) if settings is not None else None
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ck.get("settings") == settings_j:
            head.load_state_dict(ck["head"])
            opt.load_state_dict(ck["opt"])
            t0_iter, history = int(ck["t"]), list(ck["history"])
            run_sum, run_n = float(ck["run_sum"]), int(ck["run_n"])
            log(f"  resuming training from iteration {t0_iter} ({Path(ckpt_path).name})")
        else:
            log(f"  {Path(ckpt_path).name} was made with other settings; starting again")
    ds = TrainCrops(store, schedule, crop, lut, start=t0_iter * cfg.batch)
    kw = {"num_workers": int(num_workers), "pin_memory": bool(pin), "drop_last": False, "shuffle": False}
    if num_workers:
        kw["prefetch_factor"] = 4
    it = iter(DataLoader(ds, batch_size=cfg.batch, **kw))
    t_start = t_log = time.time()
    run_loss = torch.full((), run_sum, dtype=torch.float64, device=device)
    n_timed = 0                                 # iterations timed in this session's window (for img/s)
    for t in range(t0_iter, cfg.iters):
        img, lab = next(it)
        lr = lr_at(t, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        img = img.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True).long()
        with torch.no_grad():
            f = feature_fn(img)
        logits = F.interpolate(head(f), size=lab.shape[-2:], mode="bilinear", align_corners=False)
        loss = seg_loss(logits, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.detach().double()
        run_n += 1
        n_timed += 1
        last = t + 1 == cfg.iters
        if (t + 1) % cfg.log_every == 0 or last:
            mean_loss = float(run_loss) / run_n
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite training loss at iteration {t + 1}")
            speed = n_timed * cfg.batch / max(time.time() - t_log, 1e-9)
            history.append({"iter": t + 1, "loss": mean_loss, "lr": lr, "img_per_s": speed})
            log(f"  iter {t + 1}/{cfg.iters}: loss {mean_loss:.4f}, lr {lr:.3g}, {speed:.0f} img/s")
            run_loss.zero_()
            run_n, n_timed, t_log = 0, 0, time.time()
        if ckpt_path is not None and (t + 1) % cfg.ckpt_every == 0 and not last:
            _atomic_save({"t": t + 1, "head": head.state_dict(), "opt": opt.state_dict(), "history": history,
                          "run_sum": float(run_loss), "run_n": run_n, "settings": settings_j}, ckpt_path)
    head.eval()
    return {"history": history, "start_iter": t0_iter, "iters": cfg.iters, "seconds": round(time.time() - t_start, 1),
            "final_loss": history[-1]["loss"] if history else None}


# ----------------------------------------------------------------------------------------------
# Sliding-window evaluation
# ----------------------------------------------------------------------------------------------
def sliding_windows(h: int, w: int, crop: int, stride: int) -> list:
    """Top-left corners of crop x crop windows covering an h x w image (h, w >= crop): a grid with step
    `stride` whose last row and column are moved back to touch the bottom and right edges (the
    mmsegmentation slide layout). stride must be in 1..crop, so no pixel is missed."""
    if not 1 <= stride <= crop:
        raise ValueError(f"stride {stride} must be in 1..{crop}")
    if h < crop or w < crop:
        raise ValueError(f"image {h} x {w} is smaller than the {crop} px window (pad it first)")
    nh = (h - crop + stride - 1) // stride + 1
    nw = (w - crop + stride - 1) // stride + 1
    out = [(min(i * stride, h - crop), min(j * stride, w - crop)) for i in range(nh) for j in range(nw)]
    return list(dict.fromkeys(out))


@torch.no_grad()
def slide_logits(logit_fn, x: torch.Tensor, crop: int, stride: int, num_classes: int, batch: int = 16) -> tuple:
    """x [1, 3, h, w] -> (logits [1, C, h, w] float32, count [h, w]). logit_fn maps windows [n, 3, crop, crop]
    to logits [n, C, crop, crop]. Each pixel gets the mean of the logits of the windows that contain it
    (count = how many). An image smaller than the window is zero-padded at the bottom and right first."""
    _, _, h, w = x.shape
    ph, pw = max(h, crop), max(w, crop)
    if (ph, pw) != (h, w):
        x = F.pad(x, (0, pw - w, 0, ph - h))
    wins = sliding_windows(ph, pw, crop, stride)
    acc = torch.zeros(1, num_classes, ph, pw, dtype=torch.float32, device=x.device)
    cnt = torch.zeros(ph, pw, dtype=torch.float32, device=x.device)
    for s in range(0, len(wins), batch):
        chunk = wins[s:s + batch]
        out = logit_fn(torch.cat([x[:, :, y:y + crop, c:c + crop] for y, c in chunk])).float()
        if out.shape[1:] != (num_classes, crop, crop):
            raise RuntimeError(f"logit_fn returned {tuple(out.shape)}, expected (n, {num_classes}, {crop}, {crop})")
        for (y, c), o in zip(chunk, out):
            acc[0, :, y:y + crop, c:c + crop] += o
            cnt[y:y + crop, c:c + crop] += 1
    if (cnt < 1).any():
        raise RuntimeError("sliding windows left pixels uncovered")
    return (acc / cnt)[:, :, :h, :w], cnt[:h, :w]


@torch.no_grad()
def evaluate(head: LinearSegHead, feature_fn, store: RaggedStore, crop: int, stride: int, lut: np.ndarray,
             num_classes: int, device, eval_batch: int = 16, num_workers: int = 0, pin: bool = False,
             max_images: int | None = None, log_every: int = 250) -> dict:
    """Sliding-window evaluation on the first `max_images` (default all) images of `store`; logits are
    upsampled to the stored label resolution before the argmax. Returns metrics and the confusion matrix."""
    device = torch.device(device)
    head.eval()
    n = len(store) if max_images is None else min(int(max_images), len(store))
    loader = DataLoader(ValImages(store, n, lut), batch_size=None, shuffle=False, num_workers=int(num_workers),
                        pin_memory=bool(pin))
    conf = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=device)

    def logit_fn(xb):
        return F.interpolate(head(feature_fn(xb)), size=(crop, crop), mode="bilinear", align_corners=False)

    t0, n_windows = time.time(), 0
    for i, (img, lab) in enumerate(loader):
        x = img.to(device, non_blocking=True).permute(2, 0, 1)[None]
        logits, cnt = slide_logits(logit_fn, x, crop, stride, num_classes, eval_batch)
        n_windows += len(sliding_windows(max(x.shape[-2], crop), max(x.shape[-1], crop), crop, stride))
        lab = lab.to(device, non_blocking=True).long()
        if tuple(logits.shape[-2:]) != tuple(lab.shape):
            logits = F.interpolate(logits, size=tuple(lab.shape), mode="bilinear", align_corners=False)
        confusion_update(conf, logits.argmax(1)[0], lab, num_classes)
        if log_every and (i + 1) % log_every == 0:
            log(f"  eval {i + 1}/{n} images, {(i + 1) / max(time.time() - t0, 1e-9):.1f} img/s")
    cm = conf.view(num_classes, num_classes).cpu().numpy()
    return {"confusion": cm, **seg_metrics(cm), "n_images": int(n), "n_windows": int(n_windows),
            "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------
# V-exact at the segmentation resolution
# ----------------------------------------------------------------------------------------------
def adapter_at(adapter, img_size: int):
    """A shallow copy of `adapter` (same model object) that describes `img_size` inputs."""
    if img_size % adapter.patch_size:
        raise ValueError(f"{img_size} px is not a multiple of the patch size {adapter.patch_size}")
    ad = copy.copy(adapter)
    ad.img_size = int(img_size)
    ad.grid_hw = (ad.img_size // ad.patch_size, ad.img_size // ad.patch_size)
    return ad


@torch.no_grad()
def span_exactness(adapter, model, A, reg_perm, x: torch.Tensor, k: int) -> dict:
    """mirror.m1.stream_equivariance on normalised images x (float32, at the adapter's size): the largest
    relative error inside the span (layers 0..k) and the per-layer rows."""
    from mirror.m1 import register_info, stream_equivariance

    reg = register_info(model, adapter)
    rows = stream_equivariance(adapter, model, x, A, reg_perm=reg_perm, reg_start=reg["start"])
    inside = max(r["rel_max"] for r in rows if r["layer"] <= k)
    return {"max_rel_err_inside_span": float(inside), "bar": VEXACT_BAR, "pass": bool(inside < VEXACT_BAR),
            "layers": [{"layer": r["layer"], "rel_max": r["rel_max"]} for r in rows]}


def vexact_checks(factory, adapter, model, v, x: torch.Tensor, contrast_px: int | None = 224) -> dict:
    """Exactness of the V-exact model built at the adapter's size, run at that size; and, for contrast,
    of the model built at `contrast_px` (the M1 build) run at the adapter's size. The contrast needs a model
    that interpolates its position embedding (DINOv2); it is skipped for fixed-grid models."""
    S, ops, _ = factory.m1_structure(v.k)
    A, rp = S["A_star"][v.k], S["reg_perm"][v.k]
    out = {"built_px": adapter.img_size, "eval_px": adapter.img_size,
           "at_build_resolution": span_exactness(adapter, model, A, rp, x, v.k)}
    if contrast_px and contrast_px != adapter.img_size:
        if hasattr(model, "interpolate_pos_encoding"):
            ad_c = adapter_at(adapter, contrast_px)
            m_c = factory.m1().make_variant("vexact", v.k, ad_c, S, ops)
            res = span_exactness(adapter, m_c, A, rp, x, v.k)
            out["contrast"] = {"built_px": int(contrast_px), "eval_px": adapter.img_size, **res}
            del m_c
        else:
            out["contrast"] = {"built_px": int(contrast_px), "skipped": "the model has a fixed position grid"}
    return out


# ----------------------------------------------------------------------------------------------
# Tiny synthetic data (CPU smoke test and tests)
# ----------------------------------------------------------------------------------------------
TINY_ENTRY = {"name": "tiny synthetic segmentation", "hf_id": "local/tiny_seg", "revision": "none",
              "splits": {"train": "train", "val": "validation"}, "image": "image", "label": "annotation",
              "num_classes": 4, "counts": {}, "sha256": {}, "cache_name": "tiny_seg", "note": "synthetic (tests)"}


def synthetic_seg_dataset(n: int, num_classes: int, sizes, seed: int = 0, block: int = 14, noise: float = 8.0,
                          p_unlabelled: float = 0.15) -> tuple:
    """Lists of images (uint8 [h, w, 3]) and raw label maps (uint8, 0 = unlabelled, 1..num_classes): a grid
    of block x block cells, each one class with its own colour plus noise. Colours are fixed (seed 123)."""
    pal = np.random.default_rng(123).integers(20, 236, size=(num_classes + 1, 3)).astype(np.float64)
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for i in range(n):
        h, w = sizes[i % len(sizes)]
        gh, gw = -(-h // block), -(-w // block)
        cells = rng.integers(1, num_classes + 1, size=(gh, gw))
        cells[rng.random((gh, gw)) < p_unlabelled] = 0
        lab = np.kron(cells, np.ones((block, block), dtype=np.int64))[:h, :w].astype(np.uint8)
        img = np.clip(pal[lab] + noise * rng.standard_normal((h, w, 3)), 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(lab)
    return images, labels


def write_tiny_seg_parquet(root: Path, num_classes: int = 4, n_train: int = 12, n_val: int = 4, seed: int = 0,
                           sizes=((56, 70), (64, 56), (60, 84), (56, 56))) -> dict:
    """Synthetic train / validation parquet shards with PNG-encoded image and annotation columns (the
    layout of the Hugging Face parquet files). Returns {hf split: [path]}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, n, s in (("train", n_train, seed), ("validation", n_val, seed + 1)):
        imgs, labs = synthetic_seg_dataset(n, num_classes, sizes, seed=s)
        enc_i, enc_l = [], []
        for im, lb in zip(imgs, labs):
            for arr, dst in ((im, enc_i), (lb, enc_l)):
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                dst.append({"bytes": buf.getvalue(), "path": None})
        path = root / f"{split}-00000-of-00001.parquet"
        pq.write_table(pa.table({"image": enc_i, "annotation": enc_l, "scene_category": np.zeros(n, np.int64)}), path)
        files[split] = [str(path)]
    return files


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
CSV_COLUMNS = ["model", "variant", "label", "family", "method", "k", "mIoU", "aAcc", "mAcc", "iterations",
               "batch_size", "crop", "seed", "lr", "weight_decay", "warmup_iters", "bn", "eval_stride", "n_train",
               "n_val", "n_val_total", "final_train_loss", "vexact_built_px", "vexact_max_rel_err",
               "vexact_224_build_rel_err_at_seg", "converted_params_kept", "converted_params_dense", "model_params",
               "t_build_s", "t_check_s", "t_train_s", "t_eval_s", "t_total_s", "train_img_per_s",
               "resumed_from_iter", "peak_mem_gb", "amp", "hf_id", "hf_revision", "git_hash", "git_dirty", "date"]


def _f(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _e(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.1e}"


def write_tables(out: Path) -> list:
    """Rebuild t2_ade20k.csv and T2_ADE20K.md from every seg/<model>/<variant>.json under `out`."""
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "seg").glob("*/*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "row" in r:
            rows.append(r["row"])
    if not rows:
        return rows
    rows.sort(key=lambda r: (r["model"], r["k"], r["family"] != "original", r["variant"]))
    df = pd.DataFrame(rows)
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + [c for c in df.columns if c not in CSV_COLUMNS]]
    df.to_csv(Path(out) / "t2_ade20k.csv", index=False)
    write_summary(Path(out) / "T2_ADE20K.md", rows)
    return rows


def write_summary(path: Path, rows: list) -> None:
    r0 = rows[0]
    lines = ["# Tier 2: ADE20K linear segmentation", "",
             f"Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2; script `scripts/tier2_ade20k.py`, recipe {RECIPE}): "
             "frozen backbone (bf16 autocast), last-layer patch tokens after the final norm (CLS and registers "
             f"excluded), head = {'BatchNorm then ' if r0.get('bn') else ''}a 1 x 1 convolution to 150 classes. "
             f"Training: random {r0['crop']} x {r0['crop']} crops of train images resized to a shorter side of "
             f"{r0['crop']} px, random horizontal flip, batch {r0['batch_size']}, {r0['iterations']:,} iterations, "
             f"AdamW (lr {r0['lr']:g}, weight decay {r0['weight_decay']:g}, {r0['warmup_iters']} warm-up iterations, "
             "cosine), cross-entropy on bilinearly upsampled logits, label 0 ignored. The crop schedule and the "
             "head initialisation come from the seed, so every variant sees the same crops. Evaluation: "
             f"validation images resized to a shorter side of {r0['crop']} px, {r0['crop']} px sliding windows "
             f"(stride {r0['eval_stride']}), logits averaged over windows and upsampled to the original label "
             "resolution; single scale, no flip. mIoU, aAcc (pixel accuracy) and mAcc (mean class accuracy) in %.",
             "",
             f"Data: Hugging Face `{r0.get('hf_id')}` at revision `{str(r0.get('hf_revision'))[:12]}` (parquet copy of "
             f"scene_parse_150; train {int(r0['n_train']):,} images, validation {int(r0['n_val_total']):,}) "
             "(verify on VM).", ""]
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        orig = next((r for r in rs if r["family"] == "original"), None)
        lines += [f"## {model}", "",
                  "| Variant | What it is | mIoU | aAcc | mAcc | Change vs original (mIoU) | Notes |",
                  "|---|---|---|---|---|---|---|"]
        for r in rs:
            delta = "-" if orig is None or r is orig else _f(r["mIoU"] - orig["mIoU"])
            notes = []
            if r.get("n_val") is not None and r.get("n_val_total") and r["n_val"] < r["n_val_total"]:
                notes.append(f"partial validation ({r['n_val']} of {r['n_val_total']} images)")
            if r.get("method") == "vexact" and r.get("vexact_built_px"):
                note = (f"rebuilt at {r['vexact_built_px']} px; max relative error inside the span "
                        f"{_e(r.get('vexact_max_rel_err'))}")
                if r.get("vexact_224_build_rel_err_at_seg") is not None:
                    note += f" (224 px build run at {r['crop']} px: {_e(r['vexact_224_build_rel_err_at_seg'])})"
                notes.append(note)
            if r.get("family") == "healed":
                notes.append("healed at 224 px")
            if r.get("resumed_from_iter"):
                notes.append(f"training resumed at iteration {r['resumed_from_iter']}")
            lines.append(f"| {r['variant']} | {r['label']} | {_f(r['mIoU'])} | {_f(r['aAcc'])} | {_f(r['mAcc'])} | "
                         f"{delta} | {'; '.join(notes) or '-'} |")
        lines.append("")
    if any(r.get("method") == "vexact" for r in rows):
        c = r0["crop"]
        lines += ["V-exact: `mirror.m1.build_vexact` projects the position embedding on the grid in use, and "
                  "DINOv2 re-interpolates its position embedding when the input grid changes (bicubic with an "
                  "offset, not flip-symmetric). The model built at 224 px for M1 is therefore not exactly "
                  f"equivariant at {c} px. Here V-exact is rebuilt with the same recipe at the segmentation "
                  "resolution (same A*, pairings, head maps and register pairing; the position grid of "
                  f"{c} px inputs materialised and projected), and every forward pass is exactly one {c} x {c} "
                  "window. The relative errors in the notes are measured on two training crops in fp32 (M1 bar "
                  f"{VEXACT_BAR:g}).", ""]
    git = sorted({str(r["git_hash"])[:10] + (" (dirty)" if r.get("git_dirty") else "") for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    lines += [f"Source: `t2_ade20k.csv` and `seg/<model>/<variant>.json` in this folder. Commit(s): {', '.join(git)}; "
              f"seed(s): {', '.join(map(str, seeds))}. Timings and peak GPU memory are in the CSV.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--variants", default="original", help="variant names as in scripts/tier2_linear.py")
    p.add_argument("--out", default=None, help="output directory (default runs/t2_seg; tiny_out/t2_seg with --tiny)")
    p.add_argument("--cache-dir", default="cache", help="decoded data go to <cache-dir>/ade20k/")
    p.add_argument("--hf-id", default=ADE20K["hf_id"], help=f"one of {sorted(KNOWN_REVISIONS)} or another copy")
    p.add_argument("--hf-revision", default=None, help="dataset revision (default: the pinned one for --hf-id)")
    p.add_argument("--skip-sha-check", action="store_true", help="warn instead of stopping on a parquet sha256 mismatch")
    p.add_argument("--operators", default=None)
    p.add_argument("--m1-structure", default=None)
    p.add_argument("--heal-dir", default=None)
    p.add_argument("--heal-config", default=None)
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="crop schedule and head initialisation (same for every variant)")
    p.add_argument("--crop", type=int, default=None, help="crop and evaluation size, px (448)")
    p.add_argument("--iters", type=int, default=None, help="training iterations (8000)")
    p.add_argument("--batch-size", type=int, default=None, help="training batch (16)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=None, help="linear warm-up iterations (300)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--no-bn", action="store_true", help="no BatchNorm before the 1 x 1 conv")
    p.add_argument("--head-init-std", type=float, default=0.01)
    p.add_argument("--eval-stride", type=int, default=None, help="sliding-window stride, px (298)")
    p.add_argument("--eval-batch", type=int, default=None, help="windows per forward pass at evaluation (16)")
    p.add_argument("--max-val-images", type=int, default=None, help="evaluate on the first N images only (smoke runs)")
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (6)")
    p.add_argument("--decode-workers", type=int, default=None, help="processes for the one-off decoding (8)")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--no-vexact-check", action="store_true", help="skip the V-exact exactness checks")
    p.add_argument("--vexact-contrast-px", type=int, default=224,
                   help="also check the V-exact model built at this size, run at --crop (0 to skip)")
    p.add_argument("--force", action="store_true", help="recompute finished variants and trained heads")
    p.add_argument("--memory-fraction", type=float, default=None, help=f"override the {DEFAULT_MEMORY_FRACTION} GPU cap")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic parquet data, CPU")
    a = p.parse_args(argv)
    for key, (normal, tiny) in DEFAULTS.items():
        if getattr(a, key) is None:
            setattr(a, key, tiny if a.tiny else normal)
    if a.hf_revision is None:
        a.hf_revision = KNOWN_REVISIONS.get(a.hf_id, "main")
    return a


def build_seg_adapter(model_name: str, tiny: bool, device, crop: int):
    """The model adapter at the crop size (tiny: the random timm ViT, which has a fixed grid)."""
    if tiny:
        from mirror.adapters import TimmTinyAdapter

        ad = TimmTinyAdapter(img_size=crop, seed=0, randomise=True, dtype=torch.float32)
        ad.to(device=device)
        ad.model.float().eval()
        return ad
    return T2.build_adapter(model_name, False, device, crop)


def run_settings(a, v, model_name: str, sources: dict, data_meta: dict, entry: dict) -> dict:
    """Everything that changes a variant's result; a finished result is reused only if this matches."""
    return {"recipe": RECIPE, "model": model_name, "variant": v.name, "crop": a.crop, "iters": a.iters,
            "batch": a.batch_size, "lr": a.lr, "weight_decay": a.weight_decay, "warmup": a.warmup_iters,
            "min_lr_ratio": a.min_lr_ratio, "bn": not a.no_bn, "head_init_std": a.head_init_std, "seed": a.seed,
            "eval_stride": a.eval_stride, "max_val_images": a.max_val_images, "amp": a.amp, "ignore_index": IGNORE,
            "dataset": {"hf_id": entry["hf_id"], "revision": entry["revision"], "counts": data_meta.get("counts"),
                        "short_side": data_meta.get("short_side")},
            "random_seed": a.random_seed if v.method == "random" else None, "sources": sources}


def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    variants = T2.parse_variants(a.variants)
    model_name = "tiny" if tiny else a.model
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/t2_seg" if tiny else "runs/t2_seg")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    T2.apply_memory_cap(a.memory_fraction if a.memory_fraction is not None else DEFAULT_MEMORY_FRACTION)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    amp = None if a.amp == "none" else a.amp
    pin = device.type == "cuda"
    if not tiny and not os.environ.get("HF_HOME"):
        log("WARNING: HF_HOME is not set; Hugging Face downloads go to its default cache, not $HF_HOME")
    log(f"Tier 2 ADE20K: model {model_name}, variants {[v.name for v in variants]}, out {out}, device {device}, "
        f"commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("data"):
        if tiny:
            entry = dict(TINY_ENTRY)
            files = write_tiny_seg_parquet(out / "tiny_parquet", num_classes=entry["num_classes"], seed=a.seed)
            data = ensure_seg_cache(entry, out / "cache", a.crop, workers=1, files=files)
        else:
            entry = {**ADE20K, "hf_id": a.hf_id, "revision": a.hf_revision}
            data = ensure_seg_cache(entry, T2._resolve(a.cache_dir), a.crop, workers=a.decode_workers,
                                    verify_sha=not a.skip_sha_check)
    C = entry["num_classes"]
    lut = label_lut(C)
    log(f"data: {data['meta']['counts']} images in {data['dir']}")

    with timer.stage("load"):
        adapter = build_seg_adapter(a.model, tiny, device, a.crop)
        paths = T2.resolve_paths(a)
        if tiny and paths["operators"] is None and any(v.family in ("mlp", "m1") for v in variants):
            from mirror import variants as V

            V.load_or_make_structure(None, True, adapter, V.TinyProtocol(img_size=a.crop), device, out)
            paths["operators"] = str(out / "tiny_fit" / "results" / "e1_operators.npz")
        factory = T2.VariantFactory(adapter, a.model, tiny, paths["operators"], paths["m1_structure"],
                                    paths["heal_dir"], a.heal_config, a.random_seed)
    log(f"adapter: {a.crop} px, grid {adapter.grid_hw}, depth {adapter.depth}, d {adapter.embed_dim}, "
        f"prefix tokens {adapter.n_prefix_tokens}")
    cfg = TrainCfg(iters=a.iters, batch=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, warmup=a.warmup_iters,
                   min_lr_ratio=a.min_lr_ratio, log_every=a.log_every, ckpt_every=a.ckpt_every)
    n_val_total = len(data["val"])

    results, run_timing = [], {}
    for v in variants:
        vt = StageTimer()
        t_var = time.time()
        log(f"=== {v.name}: {v.label}")
        sources = factory.sources(v)                 # fails early when a needed file is missing
        settings = run_settings(a, v, model_name, sources, data["meta"], entry)
        settings_j = T2._jsonable(settings)
        vdir = out / "seg" / model_name
        vdir.mkdir(parents=True, exist_ok=True)
        vjson, head_path, ckpt_path = vdir / f"{v.name}.json", vdir / f"{v.name}_head.pt", vdir / f"{v.name}_ckpt.pt"
        if vjson.exists() and not a.force:
            prev = json.loads(vjson.read_text(encoding="utf-8"))
            if "row" in prev and prev.get("settings") == settings_j:
                log(f"  finished before ({vjson.name}): mIoU {prev['row']['mIoU']:.2f}; skipping")
                results.append(prev["row"])
                continue
            log(f"  {vjson.name} was made with other settings; recomputing")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        with vt.stage(f"build {v.name}"):
            model, info = factory.build(v)
            model.eval()
        t_build = time.time() - t0
        feature_fn = make_feature_fn(adapter, model, device, amp)

        t0, checks = time.time(), None
        if v.method == "vexact" and not a.no_vexact_check:
            with vt.stage(f"V-exact checks {v.name}"):
                sched = make_schedule(len(data["train"]), 2, a.seed)
                xs = [crop_flip(data["train"].image(int(j)), data["train"].label(int(j)), a.crop, float(uy), float(ux),
                                False, lut)[0] for j, uy, ux in zip(sched["index"], sched["uy"], sched["ux"])]
                x = normalize_for(adapter, torch.stack(xs).to(device)).float()
                checks = vexact_checks(factory, adapter, model, v, x, a.vexact_contrast_px or None)
            log(f"  V-exact built at {a.crop} px: max relative error inside the span "
                f"{checks['at_build_resolution']['max_rel_err_inside_span']:.2e} (bar {VEXACT_BAR:g})"
                + (f"; built at {checks['contrast']['built_px']} px and run at {a.crop} px: "
                   f"{checks['contrast']['max_rel_err_inside_span']:.2e}"
                   if checks.get("contrast", {}).get("max_rel_err_inside_span") is not None else ""))
            if not checks["at_build_resolution"]["pass"]:
                log(f"WARNING: {v.name} is not exactly equivariant at {a.crop} px (above the {VEXACT_BAR:g} bar); "
                    "the run continues, the row records the error")
        t_check = time.time() - t0

        head = LinearSegHead(adapter.embed_dim, C, bn=not a.no_bn, init_std=a.head_init_std, seed=a.seed).to(device)
        train_res = None
        if head_path.exists() and not a.force:
            hp = torch.load(head_path, map_location="cpu", weights_only=True)
            if hp.get("settings") == settings_j:
                head.load_state_dict(hp["head"])
                train_res = dict(hp["train"], reused_head=True)
                log(f"  trained head found ({head_path.name}); evaluating only")
        t0 = time.time()
        if train_res is None:
            if a.force and ckpt_path.exists():
                ckpt_path.unlink()
            with vt.stage(f"train {v.name}"):
                train_res = train_head(head, feature_fn, data["train"], a.crop, lut, cfg, device, a.seed, ckpt_path,
                                       settings, a.num_workers, pin)
            _atomic_save({"head": head.state_dict(), "settings": settings_j, "train": T2._jsonable(train_res)},
                         head_path)
            if ckpt_path.exists():
                ckpt_path.unlink()
            import pandas as pd

            pd.DataFrame(train_res["history"]).to_csv(vdir / f"{v.name}_train_log.csv", index=False)
        t_train = time.time() - t0

        t0 = time.time()
        with vt.stage(f"evaluate {v.name}"):
            ev = evaluate(head, feature_fn, data["val"], a.crop, a.eval_stride, lut, C, device, a.eval_batch,
                          a.num_workers, pin, a.max_val_images)
        t_eval = time.time() - t0
        np.save(vdir / f"{v.name}_confusion.npy", ev["confusion"])
        log(f"  {v.name}: mIoU {ev['mIoU']:.2f}, aAcc {ev['aAcc']:.2f}, mAcc {ev['mAcc']:.2f} "
            f"({ev['n_images']} images, {ev['n_windows']} windows)")
        hist = train_res.get("history") or []
        speeds = [h["img_per_s"] for h in hist[1:]] or [h["img_per_s"] for h in hist]
        exact = (checks or {}).get("at_build_resolution", {})
        contrast = (checks or {}).get("contrast", {})
        row = {"model": model_name, "variant": v.name, "label": v.label, "family": v.family, "method": v.method,
               "k": v.k, "mIoU": ev["mIoU"], "aAcc": ev["aAcc"], "mAcc": ev["mAcc"], "iterations": a.iters,
               "batch_size": a.batch_size, "crop": a.crop, "seed": a.seed, "lr": a.lr, "weight_decay": a.weight_decay,
               "warmup_iters": a.warmup_iters, "bn": not a.no_bn, "eval_stride": a.eval_stride,
               "n_train": len(data["train"]), "n_val": ev["n_images"], "n_val_total": n_val_total,
               "final_train_loss": train_res.get("final_loss"),
               "vexact_built_px": a.crop if checks else None,
               "vexact_max_rel_err": exact.get("max_rel_err_inside_span"),
               "vexact_224_build_rel_err_at_seg": contrast.get("max_rel_err_inside_span"),
               "converted_params_kept": info.get("converted_params_kept"),
               "converted_params_dense": info.get("converted_params_dense"), "model_params": info.get("model_params"),
               "t_build_s": round(t_build, 1), "t_check_s": round(t_check, 1), "t_train_s": round(t_train, 1),
               "t_eval_s": round(t_eval, 1), "t_total_s": round(time.time() - t_var, 1),
               "train_img_per_s": round(float(np.median(speeds)), 1) if speeds else None,
               "resumed_from_iter": int(train_res.get("start_iter") or 0),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None,
               "amp": a.amp, "hf_id": entry["hf_id"], "hf_revision": entry["revision"], "git_hash": git["git_hash"],
               "git_dirty": git["git_dirty"], "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        T2.write_json(vjson, {
            "row": row, "settings": settings,
            "metrics": {"mIoU": ev["mIoU"], "aAcc": ev["aAcc"], "mAcc": ev["mAcc"], "per_class_iou": ev["iou"],
                        "per_class_acc": ev["acc"], "n_classes_scored": ev["n_classes_scored"],
                        "n_classes_in_labels": ev["n_classes_in_labels"], "n_pixels": ev["n_pixels"],
                        "n_images": ev["n_images"], "n_windows": ev["n_windows"],
                        "class_ids": "index c = ADE20K label c + 1 (label 0 ignored)"},
            "train": train_res, "train_cfg": asdict(cfg), "vexact_checks": checks, "variant_info": info,
            "sources": sources, "dataset": {k: v_ for k, v_ in data["meta"].items() if k != "decode"},
            "timing": vt.stages, "git": git, "env": env_info(device)})
        results.append(row)
        run_timing[v.name] = vt.stages
        write_tables(out)
        del feature_fn, head
        if model is not adapter.model:
            del model
        T2.free_memory(device)

    rows = write_tables(out)
    T2.write_json(out / f"t2_ade20k_meta_{model_name}.json", {
        "experiment": "Tier 2 ADE20K linear segmentation", "recipe": RECIPE, "model": model_name,
        "variants": [v.name for v in variants], "args": vars(a), "paths": paths, "dataset": entry,
        "cache": {k: v_ for k, v_ in data["meta"].items() if k != "decode"}, "train_cfg": asdict(cfg), "git": git,
        "env": env_info(device), "seed": a.seed,
        "timing": {"stages": timer.stages, "variants": run_timing, "total_s": timer.total()}})
    log(f"done in {timer.total() / 60:.1f} min; {len(rows)} rows in {out / 't2_ade20k.csv'}")
    return {"out": str(out), "rows": results, "all_rows": rows}


if __name__ == "__main__":
    main()

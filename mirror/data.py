"""Data: ImageNet-1k validation cache (VM only), deterministic splits, synthetic data for tests.

Cache layout (written once by scripts/build_cache.py):
    cache/val224_uint8.npy   uint8 [N, 224, 224, 3], resize 256 (bicubic) + centre crop 224
    cache/val_labels.npy     int64 [N]
    cache/splits.npz         index arrays: fit, heldout, bank, query
    cache/meta.json          source files, counts, preprocessing

Images stay uint8 on disk; `normalize` converts on the fly. The mirror is
`torch.flip(img, dims=[-1])`, applied after cropping.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapters import IMAGENET_MEAN, IMAGENET_STD

SPLIT_NAMES = ("fit", "heldout", "bank", "query")


# ----------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------
def make_splits(labels: np.ndarray, per_class: dict[str, int]) -> dict[str, np.ndarray]:
    """Class-balanced split by order of appearance within each class.

    `per_class` is ordered, e.g. {"fit": 2, "heldout": 1, "bank": 39, "query": 8}: the first 2
    images of every class go to fit, the next 1 to heldout, and so on. Indices are returned
    sorted (dataset order) so memmap reads are sequential.
    """
    labels = np.asarray(labels)
    need = sum(per_class.values())
    out: dict[str, list] = {name: [] for name in per_class}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)  # dataset order = order of appearance
        if len(idx) < need:
            raise ValueError(f"class {c} has {len(idx)} images, need {need}")
        start = 0
        for name, n in per_class.items():
            out[name].append(idx[start:start + n])
            start += n
    return {name: np.sort(np.concatenate(parts)).astype(np.int64) for name, parts in out.items()}


# ----------------------------------------------------------------------------------------------
# Tensors
# ----------------------------------------------------------------------------------------------
def normalize(x_uint8: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B, 3, H, W] uint8 -> ImageNet-normalised float."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=dtype, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.to(dtype) / 255.0 - mean) / std


def mirror(x: torch.Tensor) -> torch.Tensor:
    """Left-right mirror of a batch of images [..., H, W]."""
    return torch.flip(x, dims=[-1])


class ArrayImageDataset(Dataset):
    """Wraps a uint8 [N, H, W, 3] array (memmap or in-memory) and a subset of indices."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images = images
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        img = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1)
        return img, int(self.labels[j])


def make_loader(images, labels, indices, batch_size: int, num_workers: int = 0,
                pin_memory: bool = False) -> DataLoader:
    ds = ArrayImageDataset(images, labels, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=False,
                      persistent_workers=False)


# ----------------------------------------------------------------------------------------------
# Synthetic data (tests and --tiny)
# ----------------------------------------------------------------------------------------------
def synthetic_dataset(num_classes: int, per_class_total: int, img_size: int, seed: int = 0):
    """Random uint8 images with class-balanced labels in a shuffled order.

    Returns (images [N, S, S, 3] uint8, labels [N] int64).
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(num_classes), per_class_total)
    rng.shuffle(labels)
    images = rng.integers(0, 256, size=(len(labels), img_size, img_size, 3), dtype=np.uint8)
    # A weak class signal (a colour tint) so k-NN is above chance but not saturated.
    tint = rng.integers(0, 256, size=(num_classes, 1, 1, 3))
    images = ((7 * images.astype(np.int64) + tint[labels]) // 8).astype(np.uint8)
    return images, labels.astype(np.int64)


# ----------------------------------------------------------------------------------------------
# ImageNet validation cache (VM)
# ----------------------------------------------------------------------------------------------
def preprocess_pil(img, resize: int = 256, crop: int = 224) -> np.ndarray:
    """DINOv2 evaluation transform without normalisation: Resize(256, bicubic), CentreCrop(224)."""
    from torchvision import transforms as T

    tf = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(crop)])
    return np.asarray(tf(img.convert("RGB")), dtype=np.uint8)


def _decode_one(args):
    data, resize, crop = args
    from PIL import Image

    return preprocess_pil(Image.open(io.BytesIO(data)), resize, crop)


def find_validation_parquet(repo_id: str, pattern: str) -> list[str]:
    """Download (or reuse from $HF_HOME) only the validation parquet shards; return sorted paths."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[pattern])
    files = sorted(str(p) for p in Path(local).glob(pattern))
    return files


def build_cache_from_parquet(files: list[str], cache_dir: str | Path, per_class: dict[str, int],
                             resize: int = 256, crop: int = 224, workers: int = 8,
                             batch_rows: int = 256) -> dict:
    """Decode parquet shards (columns image{bytes}, label) in file and row order into the cache."""
    import pyarrow.parquet as pq
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    final = cache_dir / f"val{crop}_uint8.npy"
    partial = cache_dir / f"val{crop}_uint8.partial.npy"
    images = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(n_total, crop, crop, 3))
    labels = np.empty(n_total, dtype=np.int64)

    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn" if os.name == "nt" else "fork").Pool(workers)
    pos = 0
    try:
        with tqdm(total=n_total, desc="decode", unit="img") as bar:
            for f in files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=batch_rows, columns=["image", "label"]):
                    imgs = batch.column("image").to_pylist()
                    labs = batch.column("label").to_pylist()
                    jobs = [(im["bytes"], resize, crop) for im in imgs]
                    arrs = pool.map(_decode_one, jobs, chunksize=8) if pool else [_decode_one(j) for j in jobs]
                    n = len(arrs)
                    images[pos:pos + n] = np.stack(arrs)
                    labels[pos:pos + n] = labs
                    pos += n
                    bar.update(n)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    assert pos == n_total, (pos, n_total)
    images.flush()
    del images
    os.replace(partial, final)
    np.save(cache_dir / "val_labels.npy", labels)
    splits = make_splits(labels, per_class)
    np.savez(cache_dir / "splits.npz", **splits)
    meta = {
        "complete": True,
        "n_images": int(n_total),
        "n_classes": int(len(np.unique(labels))),
        "resize": resize,
        "crop": crop,
        "interpolation": "bicubic",
        "per_class": dict(per_class),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "source_files": [os.path.basename(f) for f in files],
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def build_val_cache(cfg: dict) -> dict:
    d = cfg["data"]
    files = find_validation_parquet(d["hf_repo"], d["parquet_pattern"])
    if not files:
        raise FileNotFoundError(f"No files matching {d['parquet_pattern']} in {d['hf_repo']}. "
                                "Check the repo layout on Hugging Face and the gated-access approval.")
    return build_cache_from_parquet(files, d["cache_dir"], d["per_class"], resize=int(d["resize"]),
                                    crop=int(d["crop"]), workers=int(d.get("decode_workers", 8)))


def load_val_cache(cache_dir: str | Path, crop: int = 224):
    """Return (images memmap, labels, splits dict, meta)."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing: run scripts/build_cache.py first")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if not meta.get("complete"):
        raise RuntimeError("cache build did not finish; rerun scripts/build_cache.py")
    images = np.load(cache_dir / f"val{crop}_uint8.npy", mmap_mode="r")
    labels = np.load(cache_dir / "val_labels.npy")
    with np.load(cache_dir / "splits.npz") as z:
        splits = {k: z[k] for k in z.files}
    return images, labels, splits, meta


def get_data(cfg: dict):
    """(images, labels, splits, num_classes) for the configured mode."""
    d = cfg["data"]
    if cfg.get("tiny_mode"):
        total = sum(d["per_class"].values())
        images, labels = synthetic_dataset(int(d["num_classes"]), total, int(d["crop"]), seed=int(cfg["seed"]))
        splits = make_splits(labels, d["per_class"])
        return images, labels, splits, int(d["num_classes"])
    images, labels, splits, meta = load_val_cache(d["cache_dir"], int(d["crop"]))
    return images, labels, splits, int(meta["n_classes"])

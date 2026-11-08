"""Paper evaluation protocol (PAPER_EXPERIMENTS.md, Section 1): fit on train, evaluate on validation.

- Fit sets: ImageNet train, 3 fixed seeds x 2,000 images (cache/train224_uint8.npy, roles fit0..fit2).
- Held-out operator errors: validation, the first image of each class (1,000 images).
- k-NN: bank = 100 train images per class (100k, role "bank"); queries = all 50k validation images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_val_cache, make_loader, make_splits


class Protocol:
    def __init__(self, cache_dir: str | Path, batch_size: int = 128, stats_batch_size: int = 16,
                 num_workers: int = 4, pin: bool = True):
        cache = Path(cache_dir)
        self.train = np.load(cache / "train224_uint8.npy", mmap_mode="r")
        with np.load(cache / "train_index.npz") as z:
            self.index = {k: z[k] for k in z.files}
        self.train_labels = np.full(len(self.train), -1, dtype=np.int64)
        for role in ("fit0", "fit1", "fit2", "bank"):
            self.train_labels[self.index[f"{role}_rows"]] = self.index[f"{role}_labels"]
        self.val, self.val_labels, _, _ = load_val_cache(cache)
        self.heldout_idx = make_splits(self.val_labels, {"heldout": 1})["heldout"]
        self.bs, self.sbs, self.nw, self.pin = batch_size, stats_batch_size, num_workers, pin

    def fit_loader(self, seed: int):
        return make_loader(self.train, self.train_labels, self.index[f"fit{seed}_rows"], self.sbs, self.nw, self.pin)

    def heldout_loader(self):
        return make_loader(self.val, self.val_labels, self.heldout_idx, self.sbs, self.nw, self.pin)

    def bank_loader(self):
        return make_loader(self.train, self.train_labels, self.index["bank_rows"], self.bs, self.nw, self.pin)

    def query_loader(self, n: int | None = None):
        idx = np.arange(len(self.val_labels)) if n is None else make_splits(self.val_labels, {"q": n // 1000})["q"]
        return make_loader(self.val, self.val_labels, idx, self.bs, self.nw, self.pin)

    def sizes(self, seed: int) -> dict:
        return {"fit": int(len(self.index[f"fit{seed}_rows"])), "heldout": int(len(self.heldout_idx)),
                "bank": int(len(self.index["bank_rows"])), "query": int(len(self.val_labels))}

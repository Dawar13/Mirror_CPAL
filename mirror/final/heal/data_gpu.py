"""Heal images for lockstep training (docs/final_phase_design.md, Section 6): a GPU-resident uint8 copy, or a
pinned-host stream with a prefetch thread for sets too large for the GPU (the extended heal set of A11).

Both modes hand the trainer exactly the uint8 values mirror.heal.heal() reads, in the same order:
  training   the rows of mirror.heal._step_rows (a fresh seeded permutation per epoch, rows sorted within a
             batch, last partial batch dropped), one [B, S, S, 3] uint8 tensor on the device per optimiser step;
  centre     calibration and held-out rows in consecutive chunks (mirror.heal._batches order), as uint8 on the
             device; the caller applies mirror.heal.centre_view, as heal() does.
"gpu": the rows a job reads are copied to the device once (in chunks, so host memory stays small); a step is one
index_select on the device, with the index copied from pinned memory without blocking. "stream": every step reads
its rows from the (memory-mapped) array into pinned memory in a background thread (mirror.heal._prefetched and
_read_rows, the reader of heal()) and copies them without blocking. "auto": "gpu" on CUDA when the resident bytes
fit in `max_resident_gb` (default 35% of the device memory), else "stream".

The held-out rows may live in a separate array (`held_images`), for heal sets cached apart from the LR-selection
images; with one array (mirror.heal.HealData) the train and held-out rows must be disjoint, as HealData checks.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from mirror.heal import HealData, _batches, _prefetched, _read_rows, _step_rows

DATA_MODES = ("auto", "gpu", "stream")
AUTO_RESIDENT_FRACTION = 0.35

__all__ = ["DATA_MODES", "ConcatImages", "HealImages", "LockstepData", "resolve_mode"]


def _as_rows(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64).ravel()


class ConcatImages:
    """Several uint8 [N_i, S, S, 3] arrays (memmaps) read as one array: row r of the concatenation is row
    r - offset_i of array i. For a training set spread over caches (the 100k heal set plus the heal_ext role of
    mirror/final/data/cache.py for the extended budget of A11)."""

    def __init__(self, arrays):
        self.arrays = list(arrays)
        if not self.arrays:
            raise ValueError("no arrays")
        a0 = self.arrays[0]
        for a in self.arrays:
            if a.ndim != 4 or tuple(a.shape[1:]) != tuple(a0.shape[1:]) or a.dtype != np.uint8:
                raise ValueError("every array must be uint8 [N, S, S, 3] of the same size")
        self.offsets = np.cumsum([0] + [len(a) for a in self.arrays]).astype(np.int64)
        self.shape = (int(self.offsets[-1]),) + tuple(a0.shape[1:])
        self.dtype = a0.dtype
        self.ndim = 4

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        flat = rows.ravel()
        out = np.empty((len(flat),) + self.shape[1:], dtype=self.dtype)
        src = np.searchsorted(self.offsets, flat, side="right") - 1
        for i in np.unique(src):
            m = src == i
            out[m] = self.arrays[int(i)][flat[m] - self.offsets[int(i)]]
        return out.reshape(rows.shape + self.shape[1:])


class HealImages:
    """uint8 images [N, S, S, 3] and the rows a job reads from them, resident on `device` or read on demand."""

    def __init__(self, images, rows, device, resident: bool, chunk: int = 512):
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[1] != images.shape[2] or images.dtype != np.uint8:
            raise ValueError(f"images must be uint8 [N, S, S, 3], got {images.dtype} {tuple(images.shape)}")
        self.images = images
        self.rows = np.unique(_as_rows(rows))
        if self.rows.size and (self.rows[0] < 0 or self.rows[-1] >= len(images)):
            raise ValueError(f"rows out of range for {len(images)} images")
        self.device = torch.device(device)
        self.pin = self.device.type == "cuda"
        self.resident = bool(resident)
        self.store = None
        if self.resident and self.rows.size:
            S = int(images.shape[1])
            self.store = torch.empty((len(self.rows), S, S, 3), dtype=torch.uint8, device=self.device)
            for lo in range(0, len(self.rows), int(chunk)):
                part = self.rows[lo:lo + int(chunk)]
                host = _read_rows(images, part, self.pin)
                self.store[lo:lo + len(part)].copy_(host, non_blocking=self.pin)
            if self.pin:
                torch.cuda.synchronize(self.device)

    @property
    def image_size(self) -> int:
        return int(self.images.shape[1])

    @property
    def nbytes(self) -> int:
        S = self.image_size
        return int(len(self.rows)) * S * S * 3

    def positions(self, rows) -> np.ndarray:
        rows = _as_rows(rows)
        pos = np.searchsorted(self.rows, rows)
        if pos.size and (pos.max() >= len(self.rows) or not np.array_equal(self.rows[pos], rows)):
            raise KeyError("rows not held by this image set")
        return pos

    def _index(self, rows) -> torch.Tensor:
        idx = torch.from_numpy(self.positions(rows))
        if self.pin:
            return idx.pin_memory().to(self.device, non_blocking=True)
        return idx.to(self.device)

    def gather(self, rows) -> torch.Tensor:
        """uint8 [len(rows), S, S, 3] on the device, in the order of `rows`."""
        if self.store is not None:
            return self.store.index_select(0, self._index(rows))
        return _read_rows(self.images, _as_rows(rows), self.pin).to(self.device, non_blocking=self.pin)


def resolve_mode(mode: str, device, nbytes: int, max_resident_gb: float | None = None) -> str:
    """"gpu" or "stream" for the requested mode (module docstring)."""
    if mode not in DATA_MODES:
        raise ValueError(f"data mode must be one of {DATA_MODES}, not {mode!r}")
    if mode != "auto":
        return mode
    device = torch.device(device)
    if device.type != "cuda":
        return "stream"
    if max_resident_gb is None:
        total = torch.cuda.get_device_properties(device).total_memory
        limit = AUTO_RESIDENT_FRACTION * total
    else:
        limit = float(max_resident_gb) * 2 ** 30
    return "gpu" if nbytes <= limit else "stream"


class LockstepData:
    """Training, calibration and held-out images for one lockstep job (module docstring)."""

    def __init__(self, images, train_rows, device, mode: str = "auto", heldout_rows=None, held_images=None,
                 max_resident_gb: float | None = None, prefetch: int = 2, chunk: int = 512):
        self.device = torch.device(device)
        self.train_rows = _as_rows(train_rows)
        if np.unique(self.train_rows).size != self.train_rows.size:
            raise ValueError("train_rows has duplicates")
        self.heldout_rows = None if heldout_rows is None else _as_rows(heldout_rows)
        if self.heldout_rows is not None and np.unique(self.heldout_rows).size != self.heldout_rows.size:
            raise ValueError("heldout_rows has duplicates")
        same_array = held_images is None
        if same_array and self.heldout_rows is not None and np.intersect1d(self.train_rows, self.heldout_rows).size:
            raise ValueError("train_rows and heldout_rows overlap")
        S = int(images.shape[1])
        held_n = 0 if self.heldout_rows is None else len(self.heldout_rows)
        nbytes = (len(self.train_rows) + held_n) * S * S * 3
        self.mode = resolve_mode(mode, self.device, nbytes, max_resident_gb)
        resident = self.mode == "gpu"
        self.prefetch = int(prefetch)
        self.train = HealImages(images, self.train_rows, self.device, resident, chunk)
        self.held = None
        if self.heldout_rows is not None and held_n:
            src = images if same_array else held_images
            if int(src.shape[1]) != S:
                raise ValueError("held-out images must have the training images' size")
            self.held = HealImages(src, self.heldout_rows, self.device, resident, chunk)

    @classmethod
    def from_heal_data(cls, data: HealData, device, mode: str = "auto", **kw) -> "LockstepData":
        return cls(data.images, data.train_rows, device, mode, heldout_rows=data.heldout_rows, **kw)

    @property
    def image_size(self) -> int:
        return self.train.image_size

    @property
    def resident_bytes(self) -> int:
        return (self.train.nbytes if self.train.resident else 0) + (
            self.held.nbytes if self.held is not None and self.held.resident else 0)

    def describe(self) -> dict:
        return {"mode": self.mode, "n_train": int(len(self.train_rows)),
                "n_heldout": 0 if self.heldout_rows is None else int(len(self.heldout_rows)),
                "image_size": self.image_size, "resident_gb": round(self.resident_bytes / 2 ** 30, 3),
                "prefetch": self.prefetch}

    def step_batches(self, batch_size: int, total_steps: int, seed: int, start: int = 0) -> Iterator[torch.Tensor]:
        """uint8 [B, S, S, 3] on the device for every optimiser step from `start` (heal()'s rows and order)."""
        rows_iter = _step_rows(self.train_rows, int(batch_size), int(total_steps), int(seed), int(start))
        if self.train.resident:
            for rows in rows_iter:
                yield self.train.gather(rows)
            return
        images, pin, dev = self.train.images, self.train.pin, self.device
        for batch in _prefetched(lambda rows: _read_rows(images, rows, pin), rows_iter, self.prefetch):
            yield batch.to(dev, non_blocking=pin)

    def chunks(self, rows, size: int, which: str = "train") -> Iterator[torch.Tensor]:
        """uint8 chunks of `rows` (consecutive, in the given order) on the device: "train" rows (calibration) or
        "heldout" rows."""
        src = self.train if which == "train" else self.held
        if src is None:
            raise ValueError(f"no {which} images")
        for chunk in _batches(_as_rows(rows), int(size)):
            yield src.gather(chunk)

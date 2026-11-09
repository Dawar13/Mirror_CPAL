"""Healing distillation (experiment M3): the converted student learns to match the original teacher.

Recipe (the same for every method; PAPER_EXPERIMENTS.md, M3-pilot):
  teacher    the original model, frozen, eval mode;
  student    the converted model (a deep copy after surgery), kept in eval mode (no dropout or
             drop path; gradients still flow);
  loss       normalised MSE on (a) the final-norm CLS token, (b) the final-norm patch tokens and
             (c) the residual-stream output of every converted block. Each term is
             MSE / (teacher mean square); terms are summed with configurable weights (default 1 each);
  data       uint8 images [N, S, S, 3] (a memmap) plus an array of row indices; on the device:
             RandomResizedCrop to `crop_size`, random horizontal flip, ImageNet normalisation;
  optimiser  AdamW, linear warm-up then cosine decay, bf16 autocast on CUDA (fp32 on CPU),
             gradient accumulation over micro-batches, optional activation checkpointing per block;
  scope      only the converted blocks' parameters train, unless `train_scope="all"`;
  structure  "constraints" keep the converted form exact: callables `c(model)` applied under
             torch.no_grad() once before training and after every optimiser step.

Constraint protocol. Any callable `c(model) -> None` works. Optional methods, used when present:
  c.prepare(model)             one-off structural change before the optimiser is built
                               (LowRankMLP installs its factorised layers here);
  c.frozen_parameters(model)   parameters that must not train (for a mirror-projected MLP, the
                               absorbed norm2 affine and ls2, so the absorbed form is kept);
  c.check(model) -> dict       diagnostics written to the metrics file at the end;
  c.describe() -> dict         a short description for the metrics file.
An attention projector (M1) is added later as one more callable in the same list. The
random-structure control is MirrorMLPConstraint with a random involution and a random pairing.

Teacher targets. By default the targets are the teacher's own final-norm tokens and block outputs.
A teacher adapter with a method `distill_targets(x, plan) -> (feats, start)` supplies its own
targets instead (same keys; `start` may be None); mirror.heal_vexact.VExactTeacher uses this to
map the original model's outputs into the V-exact student's folded, exactly symmetric space.

Normaliser. With `normaliser="fixed"` (default) each term's teacher mean square is measured once,
before training, on `calib_images` evenly spaced training images (centre view) and then held
fixed. The loss is then a fixed weighted MSE, so gradient accumulation is exact (the micro-batch
size only changes floating-point summation order) and held-out losses are comparable between
runs. `normaliser="batch"` divides by the teacher mean square of each micro-batch instead.

Augmentation follows torchvision's RandomResizedCrop parameter sampling (10 attempts, then the
centre-crop fallback) with integer crop boxes, but resamples with bilinear `grid_sample` without
antialiasing. With 256-pixel sources and 224-pixel crops the largest downscale is 256/224, so the
difference from antialiased resizing is small.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import itertools
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .data import normalize
from .surgery import project_mlp_two
from .utils import env_info, git_info, log, set_seed

DEFAULT_LRS = (1e-5, 5e-5, 1e-4)

__all__ = [
    "DEFAULT_LRS", "HealConfig", "HealData", "Constraint", "MirrorMLPConstraint", "LowRankLinear",
    "LowRankMLP", "exact_involution", "configure_trainable", "sample_crop_params", "crop_resize_flip",
    "augment_batch", "centre_view", "distill_terms", "lr_at", "evaluate_heldout", "heal", "select_lr",
    "probe_extra_lrs", "heal_with_lr_selection", "trained_state_dict", "save_trained_blocks", "load_trained_blocks",
]


# ----------------------------------------------------------------------------------------------
# Configuration and data
# ----------------------------------------------------------------------------------------------
@dataclass
class HealConfig:
    """Settings of one healing run. `blocks` are the converted blocks (0-based)."""

    blocks: list
    seed: int = 0
    lr: float = 1e-4
    epochs: float = 2.0
    max_steps: int | None = None          # overrides epochs when set
    batch_size: int = 256                 # images per optimiser step
    micro_batch: int = 64                 # images per forward/backward (gradient accumulation)
    weight_decay: float = 0.05
    decay_1d: bool = False                # also decay biases, norm affines and LayerScale
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    warmup_frac: float = 0.05
    warmup_steps: int | None = None       # overrides warmup_frac when set
    min_lr_ratio: float = 0.0             # cosine floor as a fraction of lr
    grad_clip: float | None = None        # max global grad norm (None: no clipping)
    crop_size: int = 224
    crop_scale: tuple = (0.35, 1.0)
    crop_ratio: tuple = (3.0 / 4.0, 4.0 / 3.0)
    hflip_p: float = 0.5
    w_cls: float = 1.0
    w_patch: float = 1.0
    w_block: float | list = 1.0           # one weight for every block term, or one per block
    normaliser: str = "fixed"             # "fixed" or "batch"
    calib_images: int = 1024              # training images for the fixed normaliser
    train_scope: str = "converted"        # "converted" or "all"
    grad_checkpoint: bool = False
    amp: str = "auto"                     # "auto" (bf16 on CUDA, none on CPU), "bf16" or "none"
    share_prefix: bool = True             # reuse the teacher's residual before the first trained block
    eval_every: int = 0                   # held-out evaluation every n steps (0: start and end only)
    eval_batch: int | None = None         # default: micro_batch
    max_heldout: int | None = None        # cap on held-out images (default: all)
    log_every: int = 20                   # console log every n steps (0: start and end only)
    prefetch: int = 2                     # batches read ahead by a background thread (0: none)
    save_checkpoint: bool = True
    checkpoint_path: str | None = None    # default: <out_dir>/trained_blocks.pt

    @classmethod
    def from_dict(cls, d: dict) -> "HealConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown HealConfig keys: {sorted(unknown)}")
        d = dict(d)
        for k in ("betas", "crop_scale", "crop_ratio"):
            if k in d and d[k] is not None:
                d[k] = tuple(float(v) for v in d[k])
        for k in ("lr", "epochs", "weight_decay", "eps", "warmup_frac", "min_lr_ratio", "hflip_p",
                  "w_cls", "w_patch"):
            if k in d and d[k] is not None:
                d[k] = float(d[k])
        if "blocks" in d:
            d["blocks"] = [int(b) for b in d["blocks"]]
        return cls(**d)


def as_config(cfg) -> HealConfig:
    return cfg if isinstance(cfg, HealConfig) else HealConfig.from_dict(cfg)


@dataclass
class HealData:
    """uint8 images [N, S, S, 3] (memmap or array) and row indices for training and held-out loss.

    Held-out rows are training-set images kept out of the distillation data (never validation);
    the two row sets must be disjoint.
    """

    images: np.ndarray
    train_rows: np.ndarray
    heldout_rows: np.ndarray | None = None

    def __post_init__(self):
        im = self.images
        if im.ndim != 4 or im.shape[-1] != 3 or im.shape[1] != im.shape[2] or im.dtype != np.uint8:
            raise ValueError(f"images must be uint8 [N, S, S, 3], got {im.dtype} {tuple(im.shape)}")
        self.train_rows = self._rows(self.train_rows, "train_rows")
        if self.heldout_rows is not None:
            self.heldout_rows = self._rows(self.heldout_rows, "heldout_rows")
            if np.intersect1d(self.train_rows, self.heldout_rows).size:
                raise ValueError("train_rows and heldout_rows overlap")

    def _rows(self, rows, name: str) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).ravel()
        if rows.size and (rows.min() < 0 or rows.max() >= len(self.images)):
            raise ValueError(f"{name} out of range for {len(self.images)} images")
        if np.unique(rows).size != rows.size:
            raise ValueError(f"{name} has duplicates")
        return rows

    @property
    def image_size(self) -> int:
        return int(self.images.shape[1])


def _read_rows(images: np.ndarray, rows: np.ndarray, pin: bool) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(images[np.asarray(rows)]))
    return t.pin_memory() if pin else t


def _prefetched(fn: Callable, items: Iterable, depth: int):
    """Yield fn(item) in order, computing up to `depth` items ahead in one background thread."""
    if depth <= 0:
        for it in items:
            yield fn(it)
        return
    items = iter(items)
    with ThreadPoolExecutor(max_workers=1) as ex:
        queue = deque(ex.submit(fn, it) for it in itertools.islice(items, depth))
        while queue:
            fut = queue.popleft()
            for it in itertools.islice(items, 1):
                queue.append(ex.submit(fn, it))
            yield fut.result()


def _step_rows(train_rows: np.ndarray, batch_size: int, total_steps: int, seed: int):
    """Rows of every optimiser step: a fresh seeded permutation per epoch, last partial batch dropped.
    Rows are sorted within a batch (sequential memmap reads)."""
    n = len(train_rows)
    per_epoch = n // batch_size
    step, epoch = 0, 0
    while step < total_steps:
        perm = np.random.default_rng([int(seed), 104729, epoch]).permutation(n)
        for k in range(per_epoch):
            if step >= total_steps:
                return
            yield np.sort(train_rows[perm[k * batch_size:(k + 1) * batch_size]])
            step += 1
        epoch += 1


# ----------------------------------------------------------------------------------------------
# Device-side augmentation
# ----------------------------------------------------------------------------------------------
def sample_crop_params(n: int, height: int, width: int, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                       hflip_p: float = 0.5, generator: torch.Generator | None = None) -> torch.Tensor:
    """RandomResizedCrop boxes as torchvision samples them, plus flips: float64 [n, 5] with columns
    (top, left, crop height, crop width, flip). Sampled on the CPU, so the draws do not depend on
    the device."""
    area = float(height * width)
    log_r = torch.log(torch.tensor([float(ratio[0]), float(ratio[1])], dtype=torch.float64))
    u_area = torch.rand(n, 10, generator=generator, dtype=torch.float64)
    u_ratio = torch.rand(n, 10, generator=generator, dtype=torch.float64)
    target = area * (float(scale[0]) + (float(scale[1]) - float(scale[0])) * u_area)
    aspect = torch.exp(log_r[0] + (log_r[1] - log_r[0]) * u_ratio)
    w = torch.round(torch.sqrt(target * aspect))
    h = torch.round(torch.sqrt(target / aspect))
    ok = (w > 0) & (w <= width) & (h > 0) & (h <= height)
    first = torch.argmax(ok.to(torch.uint8), dim=1)          # first valid attempt
    any_ok = ok.any(dim=1)
    idx = torch.arange(n)
    h, w = h[idx, first], w[idx, first]
    # torchvision fallback: whole image with the aspect ratio clamped, centred
    in_ratio = width / height
    if in_ratio < min(ratio):
        fw, fh = width, int(round(width / min(ratio)))
    elif in_ratio > max(ratio):
        fh, fw = height, int(round(height * max(ratio)))
    else:
        fw, fh = width, height
    h = torch.where(any_ok, h, torch.full_like(h, fh))
    w = torch.where(any_ok, w, torch.full_like(w, fw))
    u_i = torch.rand(n, generator=generator, dtype=torch.float64)
    u_j = torch.rand(n, generator=generator, dtype=torch.float64)
    top = torch.where(any_ok, torch.floor(u_i * (height - h + 1)), torch.full_like(h, (height - fh) // 2))
    left = torch.where(any_ok, torch.floor(u_j * (width - w + 1)), torch.full_like(w, (width - fw) // 2))
    flip = (torch.rand(n, generator=generator, dtype=torch.float64) < float(hflip_p)).to(torch.float64)
    return torch.stack([top, left, h, w, flip], dim=1)


def crop_resize_flip(x: torch.Tensor, params: torch.Tensor, out_size: int) -> torch.Tensor:
    """Crop each image of x [B, C, H, W] (float) to its box, resize to out_size (bilinear) and
    mirror it if its flip flag is set. One `grid_sample` call for the whole batch."""
    B, C, H, W = x.shape
    p = params.to(device=x.device, dtype=torch.float64)
    top, left, h, w, flip = p.unbind(1)
    theta = torch.zeros(B, 2, 3, dtype=torch.float64, device=x.device)
    theta[:, 0, 0] = (w / W) * (1.0 - 2.0 * flip)
    theta[:, 0, 2] = (2.0 * left + w) / W - 1.0
    theta[:, 1, 1] = h / H
    theta[:, 1, 2] = (2.0 * top + h) / H - 1.0
    grid = F.affine_grid(theta.to(x.dtype), [B, C, out_size, out_size], align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def augment_batch(x_uint8: torch.Tensor, params: torch.Tensor, out_size: int,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """uint8 [B, S, S, 3] -> crop, resize, flip -> ImageNet-normalised [B, 3, out, out]."""
    x = x_uint8.permute(0, 3, 1, 2).to(dtype)               # 0..255
    return normalize(crop_resize_flip(x, params, out_size), dtype)


def centre_view(x_uint8: torch.Tensor, out_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """uint8 [B, S, S, 3] -> centre crop out_size -> normalised [B, 3, out, out]. For 256-pixel
    sources stored as resize 256 + centre crop 256, this is the usual resize 256 + centre crop 224."""
    S = x_uint8.shape[1]
    if S < out_size:
        raise ValueError(f"image size {S} is smaller than the crop {out_size}")
    o = (S - out_size) // 2
    x = x_uint8[:, o:o + out_size, o:o + out_size].permute(0, 3, 1, 2)
    return normalize(x, dtype)


# ----------------------------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------------------------
class Constraint(Protocol):
    def __call__(self, model: nn.Module) -> None: ...


def exact_involution(A, tol: float = 1e-3) -> np.ndarray:
    """Nearest exact symmetric orthogonal involution V sign(lam) V^T (float64).

    Operators saved in float32 are involutions only to about 1e-7; projecting with them every step
    would drift by W (A^2 - I) / 4 per step, so they are cleaned once here. Results are cached by
    content, so every student build reuses them (eigh of 768 x 768 is slow on a contended CPU)."""
    import hashlib

    A = np.asarray(A, dtype=np.float64)
    key = (A.shape, tol, hashlib.sha1(np.ascontiguousarray(A).tobytes()).hexdigest())
    if key not in _EXACT_CACHE:
        lam, V = np.linalg.eigh(0.5 * (A + A.T))
        if np.max(np.abs(np.abs(lam) - 1.0)) > tol:
            raise ValueError("operator is not close to an orthogonal involution")
        out = (V * np.sign(lam)) @ V.T
        _EXACT_CACHE[key] = 0.5 * (out + out.T)
    return _EXACT_CACHE[key].copy()


_EXACT_CACHE: dict = {}


def _absorbed_parameters(block: nn.Module) -> list:
    out = []
    norm2 = block.norm2
    for name in ("weight", "bias"):
        p = getattr(norm2, name, None)
        if isinstance(p, nn.Parameter):
            out.append(p)
    g = getattr(getattr(block, "ls2", None), "gamma", None)
    if isinstance(g, nn.Parameter):
        out.append(g)
    return out


def _is_absorbed(block: nn.Module) -> bool:
    norm2 = block.norm2
    w, b = getattr(norm2, "weight", None), getattr(norm2, "bias", None)
    g = getattr(getattr(block, "ls2", None), "gamma", None)
    return ((w is None or bool((w == 1).all())) and (b is None or bool((b == 0).all()))
            and (g is None or bool((g == 1).all())))


class MirrorMLPConstraint:
    """Keeps the MLPs of `blocks` exactly mirror-equivariant: after every optimiser step, fc1 and fc2
    weights and biases are re-projected with surgery.project_mlp_two, so MLP(x A_in) = MLP(x) A_out.

    The student's MLPs must be in absorbed form (norm2 affine identity, ls2 ones; see
    surgery.apply_surgery_two); those parameters are frozen so the form stays absorbed. The
    projection runs in `dtype` (float64 by default, so it is exact whatever the TF32 setting) and is
    rounded once to the parameter dtype. The random-structure control is this class with a random
    involution and a random pairing.
    """

    def __init__(self, blocks: Sequence[int], A_in_list: Sequence, A_out_list: Sequence, perm_list: Sequence,
                 dtype: torch.dtype = torch.float64):
        if not (len(blocks) == len(A_in_list) == len(A_out_list) == len(perm_list)):
            raise ValueError("one A_in, A_out and perm per block")
        self.blocks = [int(b) for b in blocks]
        self.A_in = [exact_involution(A) for A in A_in_list]
        self.A_out = [exact_involution(A) for A in A_out_list]
        self.perms = []
        for p in perm_list:
            p = np.asarray(p, dtype=np.int64)
            if not np.array_equal(p[p], np.arange(len(p))):
                raise ValueError("each perm must be an involution")
            self.perms.append(p)
        self.dtype = dtype
        self._cache: dict = {}

    def _ops(self, device: torch.device) -> list:
        key = str(device)
        if key not in self._cache:
            self._cache[key] = [(torch.as_tensor(a, dtype=self.dtype, device=device),
                                 torch.as_tensor(b, dtype=self.dtype, device=device))
                                for a, b in zip(self.A_in, self.A_out)]
        return self._cache[key]

    def prepare(self, model: nn.Module) -> None:
        for i, b in enumerate(self.blocks):
            blk = model.blocks[b]
            fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
            if fc1.bias is None or fc2.bias is None:
                raise ValueError(f"block {b}: fc1 and fc2 need biases")
            d, h = fc1.in_features, fc1.out_features
            if self.A_in[i].shape != (d, d) or self.A_out[i].shape != (d, d) or len(self.perms[i]) != h:
                raise ValueError(f"block {b}: operator or perm shape does not match the MLP ({h} x {d})")
            if not _is_absorbed(blk):
                raise ValueError(f"block {b}: MLP is not in absorbed form; run surgery.apply_surgery_two first")

    def frozen_parameters(self, model: nn.Module) -> list:
        return [p for b in self.blocks for p in _absorbed_parameters(model.blocks[b])]

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> None:
        dt = self.dtype
        for (Ai, Ao), b, perm in zip(self._ops(model.blocks[self.blocks[0]].mlp.fc1.weight.device),
                                     self.blocks, self.perms):
            fc1, fc2 = model.blocks[b].mlp.fc1, model.blocks[b].mlp.fc2
            e = project_mlp_two(fc1.weight.to(dt), fc1.bias.to(dt), fc2.weight.to(dt), fc2.bias.to(dt),
                                A_in=Ai, A_out=Ao, perm=perm)
            fc1.weight.copy_(e["W1"])
            fc1.bias.copy_(e["b1"])
            fc2.weight.copy_(e["W2"])
            fc2.bias.copy_(e["b2"])

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        """Relative violation of P W1 = W1 A_in, P b1 = b1, A_out W2 = W2 P, A_out b2 = b2 per block."""
        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp_min(1e-300))

        per_block = []
        dev = model.blocks[self.blocks[0]].mlp.fc1.weight.device
        for (Ai, Ao), b, perm in zip(self._ops(dev), self.blocks, self.perms):
            fc1, fc2 = model.blocks[b].mlp.fc1, model.blocks[b].mlp.fc2
            p = torch.as_tensor(perm, device=dev)
            W1, b1 = fc1.weight.to(self.dtype), fc1.bias.to(self.dtype)
            W2, b2 = fc2.weight.to(self.dtype), fc2.bias.to(self.dtype)
            per_block.append(max(rel(W1[p], W1 @ Ai), rel(b1[p], b1), rel(Ao @ W2, W2[:, p]), rel(Ao @ b2, b2)))
        return {"max_rel_violation": max(per_block), "per_block": per_block}

    def describe(self) -> dict:
        d_plus = lambda A: int(round((np.trace(A) + A.shape[0]) / 2))  # noqa: E731
        return {"type": "MirrorMLPConstraint", "blocks": self.blocks,
                "n_pairs": [int((p != np.arange(len(p))).sum() // 2) for p in self.perms],
                "n_self": [int((p == np.arange(len(p))).sum()) for p in self.perms],
                "d_plus_in": [d_plus(A) for A in self.A_in], "d_plus_out": [d_plus(A) for A in self.A_out]}


class LowRankLinear(nn.Module):
    """y = x V^T U^T + b with U [out, r] and V [r, in]: rank at most r whatever the training does."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError(f"rank {rank} out of range for {out_features} x {in_features}")
        self.in_features, self.out_features, self.rank = int(in_features), int(out_features), int(rank)
        self.U = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.V = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        """Balanced factors of the rank-r truncated SVD: U = U_r S_r^(1/2), V = S_r^(1/2) Vh_r."""
        W = linear.weight.detach().to(torch.float64)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        root = S[:rank].sqrt()
        m = cls(linear.in_features, linear.out_features, rank, bias=linear.bias is not None,
                device=linear.weight.device, dtype=linear.weight.dtype)
        m.U.copy_(U[:, :rank] * root)
        m.V.copy_(root[:, None] * Vh[:rank])
        if linear.bias is not None:
            m.bias.copy_(linear.bias)
        return m

    @property
    def weight(self) -> torch.Tensor:
        """Dense weight U V (so surgery.fold_mlp and analysis code keep working)."""
        return self.U @ self.V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.V), self.U, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


class LowRankMLP:
    """SVD baseline in training: `prepare` replaces fc1 and fc2 of `blocks` by LowRankLinear layers
    initialised from their truncated SVD, so the rank stays exact while training.

    Call surgery.apply_svd_baseline(student, blocks, ranks) first (it folds norm2 and ls2 and
    truncates); the factorisation then reproduces it exactly. With `freeze_absorbed` (default) the
    absorbed norm2 affine and ls2 stay frozen, as in the mirror arm. `ranks` is an int, one int per
    block or one (r_fc1, r_fc2) per block. Installing is idempotent.
    """

    def __init__(self, blocks: Sequence[int], ranks, freeze_absorbed: bool = True):
        self.blocks = [int(b) for b in blocks]
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.blocks)
        if len(ranks) != len(self.blocks):
            raise ValueError("one rank (or rank pair) per block")
        self.ranks = [tuple(int(v) for v in r) if isinstance(r, (tuple, list)) else (int(r), int(r)) for r in ranks]
        self.freeze_absorbed = bool(freeze_absorbed)

    def install(self, model: nn.Module) -> None:
        for b, (r1, r2) in zip(self.blocks, self.ranks):
            mlp = model.blocks[b].mlp
            for name, r in (("fc1", r1), ("fc2", r2)):
                lin = getattr(mlp, name)
                if isinstance(lin, LowRankLinear):
                    if lin.rank != r:
                        raise ValueError(f"block {b} {name} already has rank {lin.rank}, not {r}")
                    continue
                setattr(mlp, name, LowRankLinear.from_linear(lin, r))

    def prepare(self, model: nn.Module) -> None:
        if self.freeze_absorbed:
            for b in self.blocks:
                if not _is_absorbed(model.blocks[b]):
                    raise ValueError(f"block {b}: MLP is not in absorbed form; run surgery.apply_svd_baseline first")
        self.install(model)

    def frozen_parameters(self, model: nn.Module) -> list:
        if not self.freeze_absorbed:
            return []
        return [p for b in self.blocks for p in _absorbed_parameters(model.blocks[b])]

    def __call__(self, model: nn.Module) -> None:
        return None  # the factorisation keeps the rank exact; nothing to project

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        ranks = []
        for b in self.blocks:
            mlp = model.blocks[b].mlp
            ranks.append([int(torch.linalg.matrix_rank(getattr(mlp, n).weight.to(torch.float64)))
                          for n in ("fc1", "fc2")])
        return {"numerical_ranks": ranks, "target_ranks": [list(r) for r in self.ranks]}

    def describe(self) -> dict:
        return {"type": "LowRankMLP", "blocks": self.blocks, "ranks": [list(r) for r in self.ranks],
                "freeze_absorbed": self.freeze_absorbed}


def _describe(c) -> dict:
    fn = getattr(c, "describe", None)
    if callable(fn):
        return fn()
    return {"type": getattr(c, "__name__", type(c).__name__)}


def configure_trainable(model: nn.Module, blocks: Sequence[int], scope: str = "converted",
                        constraints: Sequence = ()) -> list:
    """Set requires_grad: the converted blocks (or everything with scope "all"), minus the
    parameters each constraint freezes. Returns the trainable (name, parameter) pairs."""
    if scope not in ("converted", "all"):
        raise ValueError(f"train_scope must be 'converted' or 'all', not {scope!r}")
    for p in model.parameters():
        p.requires_grad_(scope == "all")
    if scope == "converted":
        for b in blocks:
            for p in model.blocks[b].parameters():
                p.requires_grad_(True)
    for c in constraints:
        fp = getattr(c, "frozen_parameters", None)
        if callable(fp):
            for p in fp(model):
                p.requires_grad_(False)
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


# ----------------------------------------------------------------------------------------------
# Forward passes and loss
# ----------------------------------------------------------------------------------------------
@dataclass
class _Plan:
    capture: frozenset      # blocks whose outputs enter the loss
    first_grad: int         # first block run with autograd (0 when the prefix trains)
    prefix_grad: bool       # embedding layers train (scope "all")
    share: bool             # student starts from the teacher's residual entering block first_grad
    checkpoint: bool
    n_prefix: int


def term_names(blocks: Sequence[int]) -> list:
    return ["cls", "patch"] + [f"block{b}" for b in blocks]


def _term_weights(cfg: HealConfig) -> dict:
    w = {"cls": float(cfg.w_cls), "patch": float(cfg.w_patch)}
    wb = cfg.w_block
    if isinstance(wb, (list, tuple)):
        if len(wb) != len(cfg.blocks):
            raise ValueError("w_block needs one weight per converted block")
        w.update({f"block{b}": float(x) for b, x in zip(cfg.blocks, wb)})
    else:
        w.update({f"block{b}": float(wb) for b in cfg.blocks})
    return w


def _same_prefix(teacher: nn.Module, student: nn.Module, first: int) -> bool:
    """Is everything the student computes before block `first` bitwise equal to the teacher?"""
    t_sd = teacher.state_dict()
    for k, v in student.state_dict().items():
        if k.startswith("blocks."):
            if int(k.split(".")[1]) >= first:
                continue
        elif k.startswith(("norm.", "head", "fc_norm")):
            continue  # after the blocks
        tv = t_sd.get(k)
        if tv is None or tv.shape != v.shape or tv.dtype != v.dtype or not torch.equal(tv, v):
            return False
    return True


def _make_plan(teacher: nn.Module, student: nn.Module, cfg: HealConfig, n_prefix: int) -> _Plan:
    L = len(student.blocks)
    prefix_grad = any(p.requires_grad for n, p in student.named_parameters()
                      if not n.startswith(("blocks.", "norm.", "head", "fc_norm")))
    if prefix_grad:
        first = 0
    else:
        has = [any(p.requires_grad for p in blk.parameters()) for blk in student.blocks]
        first = next((i for i, f in enumerate(has) if f), L)
    share = bool(cfg.share_prefix) and not prefix_grad and _same_prefix(teacher, student, first)
    return _Plan(capture=frozenset(int(b) for b in cfg.blocks), first_grad=first, prefix_grad=prefix_grad,
                 share=share, checkpoint=bool(cfg.grad_checkpoint), n_prefix=int(n_prefix))


def _final_terms(final: torch.Tensor, n_prefix: int, feats: dict) -> dict:
    feats["cls"] = final[:, 0]
    feats["patch"] = final[:, n_prefix:]
    return feats


@torch.no_grad()
def _teacher_forward(adapter, x: torch.Tensor, plan: _Plan):
    custom = getattr(adapter, "distill_targets", None)
    if callable(custom):  # a teacher that maps its outputs into the student's space (mirror.heal_vexact)
        return custom(x, plan)
    m = adapter.model
    t = adapter.layer0_tokens(x, m)
    start = t if plan.first_grad == 0 else None
    feats = {}
    for i, blk in enumerate(m.blocks):
        t = blk(t)
        if i in plan.capture:
            feats[f"block{i}"] = t
        if i + 1 == plan.first_grad:
            start = t
    return _final_terms(m.norm(t), plan.n_prefix, feats), start


def _student_forward(adapter, model: nn.Module, x: torch.Tensor, plan: _Plan, start) -> dict:
    grad_on = torch.is_grad_enabled()
    if plan.share and start is not None:
        t, begin = start, plan.first_grad
    else:
        with torch.set_grad_enabled(grad_on and plan.prefix_grad):
            t = adapter.layer0_tokens(x, model)
        begin = 0
    feats = {}
    for i in range(begin, len(model.blocks)):
        blk = model.blocks[i]
        if i < plan.first_grad:
            with torch.no_grad():
                t = blk(t)
        elif plan.checkpoint and grad_on:
            t = _checkpoint(blk, t, use_reentrant=False)
        else:
            t = blk(t)
        if i in plan.capture:
            feats[f"block{i}"] = t
    return _final_terms(model.norm(t), plan.n_prefix, feats)


def distill_terms(student_feats: dict, teacher_feats: dict, normalisers: dict | None = None,
                  dtype: torch.dtype = torch.float32) -> dict:
    """Normalised MSE per term: mean((s - t)^2) / teacher mean square (fixed, or of this batch)."""
    out = {}
    for k, t in teacher_feats.items():
        t = t.to(dtype)
        s = student_feats[k].to(dtype)
        denom = normalisers[k] if normalisers is not None else (t.detach() ** 2).mean()
        out[k] = ((s - t) ** 2).mean() / denom
    return out


def _autocast_factory(device: torch.device, amp: str, model_dtype: torch.dtype):
    mode = amp
    if mode == "auto":
        mode = "bf16" if device.type == "cuda" else "none"
    if model_dtype == torch.float64:
        mode = "none"
    if mode == "bf16":
        return (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)), "bf16"
    if mode == "none":
        return contextlib.nullcontext, "none"
    raise ValueError(f"amp must be 'auto', 'bf16' or 'none', not {amp!r}")


def _batches(rows: np.ndarray, size: int):
    for lo in range(0, len(rows), size):
        yield rows[lo:lo + size]


@torch.no_grad()
def _teacher_mean_squares(adapter, images, rows, cfg, plan, device, dtype, autocast) -> dict:
    sums, counts = {}, {}
    for chunk in _batches(rows, int(cfg.eval_batch or cfg.micro_batch)):
        x = centre_view(_read_rows(images, chunk, False).to(device), cfg.crop_size, dtype)
        with autocast():
            feats, _ = _teacher_forward(adapter, x, plan)
        for k, v in feats.items():
            sums[k] = sums.get(k, 0.0) + float((v.double() ** 2).sum())
            counts[k] = counts.get(k, 0) + v.numel()
    return {k: sums[k] / counts[k] for k in sums}


@torch.no_grad()
def _heldout(adapter, student, images, rows, cfg, plan, normalisers, weights, device, dtype, autocast) -> dict:
    sse, tsq, cnt = {}, {}, {}
    for chunk in _batches(rows, int(cfg.eval_batch or cfg.micro_batch)):
        x = centre_view(_read_rows(images, chunk, False).to(device), cfg.crop_size, dtype)
        with autocast():
            tf, start = _teacher_forward(adapter, x, plan)
            sf = _student_forward(adapter, student, x, plan, start)
        for k, t in tf.items():
            t = t.double()
            sse[k] = sse.get(k, 0.0) + float(((sf[k].double() - t) ** 2).sum())
            tsq[k] = tsq.get(k, 0.0) + float((t ** 2).sum())
            cnt[k] = cnt.get(k, 0) + t.numel()
    terms = {k: (sse[k] / cnt[k]) / (normalisers[k] if normalisers is not None else tsq[k] / cnt[k]) for k in sse}
    out = {"total": float(sum(weights[k] * v for k, v in terms.items()))}
    out.update(terms)
    return out


def evaluate_heldout(teacher_adapter, student_model: nn.Module, data: HealData, cfg,
                     normalisers: dict | None = None, rows: np.ndarray | None = None) -> dict:
    """Normalised distillation loss (total and per term) on held-out rows, centre view, no grad.
    Without `normalisers` each term is divided by the teacher mean square over these rows."""
    cfg = as_config(cfg)
    rows = data.heldout_rows if rows is None else np.asarray(rows, dtype=np.int64)
    if rows is None or len(rows) == 0:
        raise ValueError("no held-out rows")
    device = next(student_model.parameters()).device
    model_dtype = next(student_model.parameters()).dtype
    dtype = torch.float64 if model_dtype == torch.float64 else torch.float32
    autocast, _ = _autocast_factory(device, cfg.amp, model_dtype)
    plan = _Plan(capture=frozenset(cfg.blocks), first_grad=0, prefix_grad=False, share=False,
                 checkpoint=False, n_prefix=int(teacher_adapter.n_prefix_tokens))
    was_training = student_model.training
    student_model.eval()
    try:
        return _heldout(teacher_adapter, student_model, data.images, rows, cfg, plan, normalisers,
                        _term_weights(cfg), device, dtype, autocast)
    finally:
        student_model.train(was_training)


# ----------------------------------------------------------------------------------------------
# Schedule, logging, checkpoints
# ----------------------------------------------------------------------------------------------
def lr_at(step: int, base_lr: float, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0) -> float:
    """Learning rate of optimiser step `step` (0-based): linear warm-up, then cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    lo = base_lr * float(min_lr_ratio)
    return lo + (base_lr - lo) * 0.5 * (1.0 + math.cos(math.pi * progress))


class _CsvLog:
    def __init__(self, path: Path | None, columns: list):
        self.columns = columns
        self.fh = None
        if path is not None:
            self.fh = open(path, "w", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.fh, fieldnames=columns)
            self.w.writeheader()
            self.fh.flush()

    def write(self, row: dict) -> None:
        if self.fh is not None:
            self.w.writerow({k: row.get(k) for k in self.columns})
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2)


def trained_state_dict(model: nn.Module, blocks: Sequence[int], scope: str = "converted") -> dict:
    """The trained part of the student: the converted blocks' state (keys "blocks.<b>.<name>"), or
    the whole model with scope "all"."""
    if scope == "all":
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {f"blocks.{b}.{k}": v.detach().cpu().clone() for b in blocks for k, v in model.blocks[b].state_dict().items()}


def save_trained_blocks(model: nn.Module, blocks: Sequence[int], path, scope: str = "converted",
                        meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": trained_state_dict(model, blocks, scope), "blocks": [int(b) for b in blocks],
                "scope": scope, "meta": _jsonable(meta or {})}, path)
    return path


def load_trained_blocks(model: nn.Module, path) -> dict:
    """Load a checkpoint from `save_trained_blocks` into a student of the same structure (for the
    SVD arm, install LowRankMLP first). Returns the checkpoint's meta."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd = payload["state_dict"]
    own = model.state_dict()
    missing = [k for k in sd if k not in own]
    if missing:
        hint = " (install LowRankMLP before loading an SVD-arm checkpoint)" if any(".U" in k for k in missing) else ""
        raise KeyError(f"checkpoint keys not in the model{hint}: {missing[:5]}")
    model.load_state_dict(sd, strict=False)
    return payload.get("meta", {})


# ----------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------
def heal(teacher_adapter, student_model: nn.Module, data: HealData, cfg, constraints: Sequence = (),
         out_dir=None) -> dict:
    """Distil the teacher (teacher_adapter.model) into `student_model` in place.

    Writes to out_dir (if given): train_log.csv (one row per step), heldout_log.csv, metrics.json
    (config, seed, git hash, environment, losses, constraint checks) and the trained blocks'
    state_dict. Returns the metrics plus "history" (per-step training loss).
    """
    cfg = as_config(cfg)
    t0 = time.time()
    teacher = teacher_adapter.model
    student = student_model
    if student is teacher:
        raise ValueError("the student must be a separate copy of the teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.eval()
    device = next(student.parameters()).device
    if next(teacher.parameters()).device != device:
        raise ValueError("teacher and student must be on the same device")
    model_dtype = next(student.parameters()).dtype
    dtype = torch.float64 if model_dtype == torch.float64 else torch.float32
    L = len(student.blocks)
    blocks = [int(b) for b in cfg.blocks]
    if not blocks or len(set(blocks)) != len(blocks) or min(blocks) < 0 or max(blocks) >= L:
        raise ValueError(f"blocks must be distinct indices in [0, {L})")
    if data.image_size < cfg.crop_size:
        raise ValueError(f"images ({data.image_size}) are smaller than crop_size ({cfg.crop_size})")
    if cfg.normaliser not in ("fixed", "batch"):
        raise ValueError("normaliser must be 'fixed' or 'batch'")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    constraints = list(constraints or [])
    for c in constraints:
        prep = getattr(c, "prepare", None)
        if callable(prep):
            prep(student)
    trainable = configure_trainable(student, blocks, cfg.train_scope, constraints)
    if not trainable:
        raise ValueError("no trainable parameters")
    with torch.no_grad():
        for c in constraints:
            c(student)
    plan = _make_plan(teacher, student, cfg, teacher_adapter.n_prefix_tokens)
    weights = _term_weights(cfg)
    names = term_names(blocks)
    autocast, amp_mode = _autocast_factory(device, cfg.amp, model_dtype)

    B = int(cfg.batch_size)
    mb = max(1, min(int(cfg.micro_batch), B))
    n_train = len(data.train_rows)
    per_epoch = n_train // B
    if per_epoch < 1:
        raise ValueError(f"batch_size {B} is larger than the {n_train} training images")
    total = int(cfg.max_steps) if cfg.max_steps is not None else int(math.floor(cfg.epochs * per_epoch))
    if total < 1:
        raise ValueError("the budget gives no optimiser steps")
    warmup = int(cfg.warmup_steps) if cfg.warmup_steps is not None else int(round(cfg.warmup_frac * total))
    warmup = max(0, min(warmup, total))

    decay = [p for _, p in trainable if p.ndim >= 2 or cfg.decay_1d]
    no_decay = [p for _, p in trainable if not (p.ndim >= 2 or cfg.decay_1d)]
    groups = [g for g in ({"params": decay, "weight_decay": float(cfg.weight_decay)},
                          {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
    opt_kw = {"fused": True} if device.type == "cuda" and model_dtype == torch.float32 else {}
    opt = torch.optim.AdamW(groups, lr=float(cfg.lr), betas=tuple(cfg.betas), eps=float(cfg.eps), **opt_kw)
    params = [p for _, p in trainable]

    normalisers = None
    if cfg.normaliser == "fixed":
        sorted_train = np.sort(data.train_rows)
        n_cal = max(1, min(int(cfg.calib_images), n_train))
        calib = sorted_train[np.linspace(0, n_train - 1, n_cal).round().astype(np.int64)]
        normalisers = _teacher_mean_squares(teacher_adapter, data.images, np.unique(calib), cfg, plan,
                                            device, dtype, autocast)
    held_rows = data.heldout_rows
    if held_rows is not None and cfg.max_heldout is not None:
        held_rows = held_rows[:int(cfg.max_heldout)]
    has_held = held_rows is not None and len(held_rows) > 0

    def held_eval():
        return _heldout(teacher_adapter, student, data.images, held_rows, cfg, plan, normalisers, weights,
                        device, dtype, autocast) if has_held else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    held_initial = held_eval()
    train_log = _CsvLog(out / "train_log.csv" if out else None,
                        ["step", "epoch", "lr", "loss", *names, "grad_norm", "seconds"])
    held_log = _CsvLog(out / "heldout_log.csv" if out and has_held else None, ["step", "total", *names])
    if held_initial is not None:
        held_log.write({"step": 0, **held_initial})
    n_trainable = int(sum(p.numel() for p in params))
    log(f"heal: {total} steps x {B} images (micro {mb}), lr {cfg.lr:g}, warm-up {warmup}, amp {amp_mode}, "
        f"{n_trainable:,} trainable parameters, share_prefix {plan.share}, checkpointing {plan.checkpoint}"
        + (f", held-out {held_initial['total']:.5f}" if held_initial else ""))

    gen = torch.Generator().manual_seed(int(cfg.seed) + 7919)  # augmentation draws (CPU)
    pin = device.type == "cuda"
    reader = _prefetched(lambda rows: _read_rows(data.images, rows, pin),
                         _step_rows(data.train_rows, B, total, int(cfg.seed)), int(cfg.prefetch))
    S = data.image_size
    history, diverged, steps_done = [], False, 0
    t_loop = time.time()
    try:
        for step, batch in zip(range(total), reader):
            lr_now = lr_at(step, float(cfg.lr), total, warmup, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr_now
            crop = sample_crop_params(B, S, S, cfg.crop_scale, cfg.crop_ratio, cfg.hflip_p, gen)
            xb = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            acc = {k: torch.zeros((), dtype=torch.float64, device=device) for k in names}
            for lo in range(0, B, mb):
                x = augment_batch(xb[lo:lo + mb], crop[lo:lo + mb], cfg.crop_size, dtype)
                frac = x.shape[0] / B
                with autocast():
                    tf, start = _teacher_forward(teacher_adapter, x, plan)
                    sf = _student_forward(teacher_adapter, student, x, plan, start)
                terms = distill_terms(sf, tf, normalisers, dtype)
                loss = sum(weights[k] * v for k, v in terms.items())
                (loss * frac).backward()
                for k, v in terms.items():
                    acc[k] += v.detach().double() * frac
                del tf, sf, terms, loss, start
            term_vals = {k: float(v) for k, v in acc.items()}
            loss_val = float(sum(weights[k] * v for k, v in term_vals.items()))
            history.append(loss_val)
            if not math.isfinite(loss_val):
                diverged = True
                log(f"heal: non-finite loss at step {step + 1}; stopping")
                break
            gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip) if cfg.grad_clip else float("inf"))
            opt.step()
            with torch.no_grad():
                for c in constraints:
                    c(student)
            steps_done = step + 1
            train_log.write({"step": steps_done, "epoch": round(steps_done / per_epoch, 4), "lr": lr_now,
                             "loss": loss_val, **term_vals, "grad_norm": float(gn),
                             "seconds": round(time.time() - t_loop, 3)})
            if cfg.eval_every and has_held and steps_done % int(cfg.eval_every) == 0 and steps_done < total:
                held_log.write({"step": steps_done, **held_eval()})
            if cfg.log_every and steps_done % int(cfg.log_every) == 0:
                el = time.time() - t_loop
                log(f"heal: step {steps_done}/{total} loss {loss_val:.5f} lr {lr_now:.2e} "
                    f"({el / steps_done:.2f} s/step, ETA {(total - steps_done) * el / steps_done / 60:.1f} min)")
    finally:
        train_log.close()

    held_final = held_eval() if not diverged else None
    if held_final is not None:
        held_log.write({"step": steps_done, **held_final})
    held_log.close()
    checks = {}
    for i, c in enumerate(constraints):
        fn = getattr(c, "check", None)
        if callable(fn):
            checks[f"{i}:{_describe(c)['type']}"] = fn(student)

    ckpt = None
    if cfg.save_checkpoint and (out is not None or cfg.checkpoint_path):
        ckpt = Path(cfg.checkpoint_path) if cfg.checkpoint_path else out / "trained_blocks.pt"
        save_trained_blocks(student, blocks, ckpt, cfg.train_scope,
                            meta={"seed": cfg.seed, "lr": cfg.lr, "steps": steps_done, **git_info()})

    finite = [h for h in history if math.isfinite(h)]
    metrics = {
        "config": asdict(cfg), "seed": int(cfg.seed), **git_info(), "env": env_info(device),
        "constraints": [_describe(c) for c in constraints], "constraint_checks": checks,
        "n_trainable": n_trainable, "n_trainable_tensors": len(trainable),
        "trainable_names": [n for n, _ in trainable],
        "amp": amp_mode, "share_prefix_used": plan.share, "first_grad_block": plan.first_grad,
        "normalisers": normalisers, "weights": weights,
        "total_steps": total, "steps_done": steps_done, "warmup_steps": warmup, "steps_per_epoch": per_epoch,
        "images_seen": steps_done * B, "diverged": diverged,
        "heldout_images": int(len(held_rows)) if has_held else 0,
        "heldout_initial": held_initial, "heldout_final": held_final,
        "train_loss_first": history[0] if history else None,
        "train_loss_last": history[-1] if history else None,
        "train_loss_last10_mean": float(np.mean(finite[-10:])) if finite else None,
        "seconds_total": round(time.time() - t0, 2), "seconds_train": round(time.time() - t_loop, 2),
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else None,
        "checkpoint": str(ckpt) if ckpt else None,
    }
    if out is not None:
        _write_json(out / "metrics.json", metrics)
    log(f"heal: done {steps_done} steps in {metrics['seconds_train']:.1f} s; "
        + (f"held-out {held_initial['total']:.5f} -> {held_final['total']:.5f}" if held_final else "no held-out loss"))
    return {**metrics, "history": history}


# ----------------------------------------------------------------------------------------------
# Learning-rate selection
# ----------------------------------------------------------------------------------------------
def _lr_trial(teacher_adapter, make_student: Callable, data: HealData, cfg: HealConfig, lr: float,
              budget_steps: int, out: Path | None) -> dict:
    """One short selection trial: a fresh student, `budget_steps` steps at `lr` with its own warm-up and
    cosine schedule, the config's seed, no checkpoint; scored by the held-out loss at the end (+inf if it
    diverged). Writes the trial's logs to <out>/lr_<lr>/ when `out` is given."""
    student, cons = make_student()
    tcfg = replace(cfg, lr=float(lr), max_steps=int(budget_steps), save_checkpoint=False,
                   checkpoint_path=None, eval_every=0)
    res = heal(teacher_adapter, student, data, tcfg, cons, out / f"lr_{float(lr):.0e}" if out else None)
    hf = res["heldout_final"]
    score = float(hf["total"]) if (hf is not None and not res["diverged"]) else float("inf")
    if not math.isfinite(score):
        score = float("inf")
    trial = {"lr": float(lr), "heldout_loss": score, "heldout_terms": hf,
             "heldout_initial": res["heldout_initial"]["total"] if res["heldout_initial"] else None,
             "train_loss_last": res["train_loss_last"], "diverged": res["diverged"]}
    del student, cons, res
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trial


def select_lr(teacher_adapter, make_student: Callable, data: HealData, cfg, lrs: Sequence[float] = DEFAULT_LRS,
              budget_steps: int = 100, out_dir=None) -> dict:
    """Short run per learning rate; pick the lowest distillation loss on the held-out train rows.

    `make_student()` must return a fresh (student_model, constraints) each call. Every trial uses
    the same seed, so data order and augmentation are identical; each trial has its own warm-up and
    cosine schedule over `budget_steps`. Diverged trials score +inf; ties go to the earlier rate.
    """
    if data.heldout_rows is None or len(data.heldout_rows) == 0:
        raise ValueError("learning-rate selection needs held-out train rows")
    cfg = as_config(cfg)
    out = Path(out_dir) if out_dir is not None else None
    trials = [_lr_trial(teacher_adapter, make_student, data, cfg, lr, budget_steps, out) for lr in lrs]
    if all(not math.isfinite(t["heldout_loss"]) for t in trials):
        raise RuntimeError("every learning rate diverged")
    best = min(range(len(trials)), key=lambda i: (trials[i]["heldout_loss"], i))
    result = {"best_lr": trials[best]["lr"], "lrs": [float(v) for v in lrs], "budget_steps": int(budget_steps),
              "seed": int(cfg.seed), "trials": trials, **git_info()}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "lr_selection.json", result)
    log("lr selection: " + ", ".join(f"{t['lr']:.0e} -> {t['heldout_loss']:.5f}" for t in trials)
        + f"; best {result['best_lr']:.0e}")
    return result


def probe_extra_lrs(teacher_adapter, make_student: Callable, data: HealData, cfg, selection: dict,
                    extra_lrs: Sequence[float], out_dir=None, extra_info: dict | None = None) -> dict:
    """Secondary learning-rate check (E3, PREREG_E3.md): the short selection trial of select_lr at each rate
    in `extra_lrs` (same seed, so the same data order and augmentation, the same budget and the same held-out
    train rows as the grid in `selection`), compared with the grid's trials. It never changes the chosen
    rate: the result only says whether an extra rate would have won (its held-out loss below every grid
    rate's; ties go to the grid, as select_lr breaks ties towards the earlier rate). A diverged extra trial
    is recorded, not raised. Writes <out_dir>/lr_extra/lr_<lr>/ (trial logs) and <out_dir>/lr_extra.json."""
    if data.heldout_rows is None or len(data.heldout_rows) == 0:
        raise ValueError("learning-rate selection needs held-out train rows")
    cfg = as_config(cfg)
    extra_lrs = [float(v) for v in extra_lrs]
    if not extra_lrs:
        raise ValueError("no extra learning rates")
    out = Path(out_dir) if out_dir is not None else None
    if selection.get("skipped") or not selection.get("trials"):
        raise ValueError("the extra rates are compared with the grid's trials: run the learning-rate selection "
                         "(no --lr) to use them")
    budget = int(selection["budget_steps"])
    if int(selection.get("seed", cfg.seed)) != int(cfg.seed):
        raise ValueError("the grid was selected with another seed")
    grid = [{"lr": float(t["lr"]), "heldout_loss": float(t["heldout_loss"]), "diverged": bool(t["diverged"])}
            for t in selection["trials"]]
    trials = [_lr_trial(teacher_adapter, make_student, data, cfg, lr, budget, out / "lr_extra" if out else None)
              for lr in extra_lrs]
    grid_best = min(t["heldout_loss"] for t in grid)
    winners = [t["lr"] for t in trials if t["heldout_loss"] < grid_best]
    everything = grid + [{"lr": t["lr"], "heldout_loss": t["heldout_loss"]} for t in trials]
    best_all = min(range(len(everything)), key=lambda i: (everything[i]["heldout_loss"], i))
    result = {"check": "secondary learning-rate check (PREREG_E3.md): would an extra rate have won the short selection?",
              "rule": "an extra rate would have won if its held-out loss after the short budget is below every grid "
                      "rate's (ties go to the grid); the chosen rate is never changed",
              "chosen_lr": float(selection["best_lr"]), "grid_lrs": [t["lr"] for t in grid], "grid": grid,
              "extra_lrs": extra_lrs, "extra": trials, "extra_would_win": bool(winners), "winning_extra_lrs": winners,
              "best_lr_with_extra": float(everything[best_all]["lr"]), "budget_steps": budget, "seed": int(cfg.seed),
              "n_heldout": int(len(data.heldout_rows) if cfg.max_heldout is None
                               else min(len(data.heldout_rows), int(cfg.max_heldout))),
              **(extra_info or {}), **git_info()}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "lr_extra.json", result)
    log("lr extra: " + ", ".join(f"{t['lr']:.0e} -> {t['heldout_loss']:.5f}" for t in trials)
        + f"; grid best {selection['best_lr']:.0e} ({grid_best:.5f}); "
        + (f"WOULD HAVE WON: {', '.join(f'{v:.0e}' for v in winners)}" if winners else "no extra rate would have won")
        + " (the chosen rate is unchanged)")
    return result


def heal_with_lr_selection(teacher_adapter, make_student: Callable, data: HealData, cfg,
                           lrs: Sequence[float] = DEFAULT_LRS, budget_steps: int = 100, out_dir=None,
                           extra_lrs: Sequence[float] = ()) -> dict:
    """select_lr, then the full budget with the chosen rate. Returns {"lr_selection", "final",
    "student"} (the healed student model, for evaluation). With `extra_lrs`, probe_extra_lrs runs between
    the selection and the full run (its result under "lr_extra"); the chosen rate is unchanged."""
    cfg = as_config(cfg)
    out = Path(out_dir) if out_dir is not None else None
    sel = select_lr(teacher_adapter, make_student, data, cfg, lrs, budget_steps, out / "lr_selection" if out else None)
    extra = (probe_extra_lrs(teacher_adapter, make_student, data, cfg, sel, extra_lrs, out)
             if extra_lrs else None)
    student, cons = make_student()
    final = heal(teacher_adapter, student, data, replace(cfg, lr=sel["best_lr"]), cons, out / "final" if out else None)
    res = {"lr_selection": sel, "final": final, "student": student}
    if extra is not None:
        res["lr_extra"] = extra
    return res

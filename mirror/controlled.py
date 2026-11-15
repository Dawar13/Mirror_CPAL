"""Controlled training (MUST-12, CORE-01, UP-01, UP-03): small ViTs trained from scratch on ImageNet-100.

The study: train the same ViT on ImageNet-100 at 64 x 64 with different symmetry augmentations and
measure which mirror or rotation structure appears. Every run uses the same recipe; only the
augmentation changes (PROJECT_NEXT_PHASE.md, CORE-01).

Data (written by scripts/build_imagenet100.py, read here):
    cache/in100_train72_uint8.npy   uint8 [N_train, 72, 72, 3]: shorter side to 72 (bicubic, antialias),
                                    centre crop 72
    cache/in100_val72_uint8.npy     uint8 [N_val, 72, 72, 3], same preprocessing (5,000 images)
    cache/in100_train_labels.npy    int64 labels 0..99: the rank of the class's WordNet ID in sorted order
    cache/in100_val_labels.npy
    cache/in100_index.npz           source (shard, row) and Hugging Face label of every image
    cache/in100_meta.json           mirror.meta record; extra.complete is true once the build finished
`write_synthetic_cache` writes the same files with random images (tests and --tiny).

Models (`build_model`): timm VisionTransformer at 64 px, patch 8 (8 x 8 grid plus CLS, 65 tokens),
learned position embeddings (CLS included), no registers, no LayerScale, LayerNorm, qkv bias, MLP
ratio 4, CLS pooling into a linear head.
    vit_s8     dim 384, depth 12, heads 6 (21,432,292 parameters with a 100-class head)
    vit_ti8    dim 192, depth 12, heads 3 (5,407,780 parameters; UP-03)
    vit_test8  dim 48, depth 3, heads 3 (CPU tests and --tiny only)

Symmetry conventions (PROJECT_NEXT_PHASE.md 4.3). Images are [..., H, W]: H runs top to bottom, W
left to right, as the image is displayed. On a G x G patch grid:
    LR    torch.flip(x, dims=[-1])                 token (r, c) -> (r, G-1-c)
    UD    torch.flip(x, dims=[-2])                 token (r, c) -> (G-1-r, c)
    R90   torch.rot90(x, k=1, dims=(-2, -1))       counter-clockwise; token (r, c) -> (G-1-c, r)
    R180, R270: k = 2, 3. LRUD is both flips (the same as R180).
`apply_view` applies one of these to a batch; tests check them with a marker image.

Training augmentation (`augment_train`, all on the device from the uint8 arrays):
    1. random resized crop of the 72 x 72 image to 64 x 64: torchvision's box sampling
       (mirror.heal.sample_crop_params, scale 0.35 to 1, aspect 3/4 to 4/3), bilinear resampling
       (mirror.heal.crop_resize_flip, with its flip switched off);
    2. the condition's symmetry (CONDITIONS): per image, LR flip with probability p_lr, then UD flip
       with probability p_ud, then (R90 condition only) a rotation by k x 90 degrees with k uniform
       over 0..3;
    3. ImageNet mean and std normalisation.
No colour jitter, mixup or cutmix. The random draws are a pure function of (seed, epoch): one
generator for the crop boxes and another for the symmetry draws, and every condition makes the same
draws. So two runs with the same seed see the same images in the same order with the same crops and
start from the same weights; they differ only in the symmetry applied.

Evaluation views: the centre 64 x 64 crop of the stored 72 x 72 image (equal to resize 72 and centre
crop 64), then the view's transform. The centre crop commutes with every view exactly.

Interface for the measurement code (scripts/measure_groups.py imports these lazily):
    load_checkpoint_adapter(path, device)   ControlledAdapter, the mirror.adapters.ViTAdapter interface
    fit_loader(seed, n=2000, batch_size=64) seeded, class-balanced train images, centre view, uint8
    val_loader(transform=None|"LR"|"UD"|"R90", ...)   validation images, centre view, then transform
    list_checkpoints(run_dir)               [(epoch, path)] sorted by epoch
    apply_view(x, view)                     the conventions above, for any [..., H, W] tensor
Loaders yield (uint8 [B, 3, 64, 64], int64 labels), like mirror.protocol.Protocol, so
mirror.data.normalize_for, mirror.knn.extract_features and mirror.stats.run_paired work unchanged.

Training (`TrainConfig`, `train`; CLI in scripts/train_controlled.py): supervised cross-entropy with
label smoothing 0.1; AdamW (fused on CUDA), lr 1e-3 at batch 512 (scaled linearly with the batch),
weight decay 0.05 on weight matrices only, 10 warm-up epochs then cosine decay per step; bf16
autocast and torch.compile on CUDA. The whole training set sits on the GPU as uint8 (about 2 GB).
Checkpoints, logs and the preemption-safe "last" state are described in `train`.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import IMAGENET_MEAN, IMAGENET_STD, ViTAdapter
from .utils import REPO_ROOT, log

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
IMG_SIZE = 64
STORE_SIZE = 72
PATCH_SIZE = 8
NUM_CLASSES = 100
CROP_SCALE = (0.35, 1.0)
CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)
DEFAULT_CKPT_EPOCHS = (0, 2, 5, 10, 25, 50, 100)
CKPT_FORMAT = "mirror-controlled-v1"

CLASS_LIST = REPO_ROOT / "data" / "imagenet100_classes.txt"
TRAIN_FILE = "in100_train72_uint8.npy"
VAL_FILE = "in100_val72_uint8.npy"
TRAIN_LABELS = "in100_train_labels.npy"
VAL_LABELS = "in100_val_labels.npy"
INDEX_FILE = "in100_index.npz"
META_FILE = "in100_meta.json"

ARCHS = {
    "vit_s8": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit_ti8": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "vit_test8": {"embed_dim": 48, "depth": 3, "num_heads": 3},   # CPU tests and --tiny only
    # Patch 16 at 96 x 96 (stored at 108 x 108): the MUST-12 pilot rerun after gate C failed at 64 x 64
    # (PREREG_NEXT_PHASE.md MUST-12, proposed fix; notes/decisions.md, 15 November).
    "vit_s16": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 16},
    "vit_ti16": {"embed_dim": 192, "depth": 12, "num_heads": 3, "patch_size": 16},
    "vit_test16": {"embed_dim": 48, "depth": 3, "num_heads": 3, "patch_size": 16},   # CPU tests only
}


def arch_patch_size(arch: str) -> int:
    """Patch size of an architecture in ARCHS (PATCH_SIZE unless the entry sets its own)."""
    return int(ARCHS[arch].get("patch_size", PATCH_SIZE))


def view_size_for_store(store_size: int) -> int:
    """Model input size for images stored at `store_size`: the same ratio as 64 of 72 (72 -> 64, 108 -> 96)."""
    s = int(store_size)
    if (s * IMG_SIZE) % STORE_SIZE:
        raise ValueError(f"stored size {s} is not a multiple of {STORE_SIZE // math.gcd(IMG_SIZE, STORE_SIZE)}")
    return s * IMG_SIZE // STORE_SIZE

# p_lr, p_ud: per-image probabilities of a left-right and an upside-down flip (independent);
# rot90: per-image rotation by k x 90 degrees counter-clockwise, k uniform over 0..3.
CONDITIONS = {
    "none": {"p_lr": 0.0, "p_ud": 0.0, "rot90": False},
    "LR": {"p_lr": 0.5, "p_ud": 0.0, "rot90": False},
    "UD": {"p_lr": 0.0, "p_ud": 0.5, "rot90": False},
    "LRUD": {"p_lr": 0.5, "p_ud": 0.5, "rot90": False},
    "R90": {"p_lr": 0.0, "p_ud": 0.0, "rot90": True},
    "UD_p0.1": {"p_lr": 0.0, "p_ud": 0.1, "rot90": False},
    "UD_p0.25": {"p_lr": 0.0, "p_ud": 0.25, "rot90": False},
}

VIEWS = ("orig", "LR", "UD", "R90")                       # validation views logged at every checkpoint
_VIEW_ALIASES = {None: "orig", "none": "orig", "orig": "orig", "id": "orig"}
ALL_VIEWS = ("orig", "LR", "UD", "LRUD", "R90", "R180", "R270")


# ----------------------------------------------------------------------------------------------
# Class list and label order
# ----------------------------------------------------------------------------------------------
def read_class_list(path: str | Path = CLASS_LIST) -> list[str]:
    """WordNet IDs from the class-list file (comment lines start with #). Checks 100 unique IDs."""
    wnids = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not re.fullmatch(r"n\d{8}", s):
            raise ValueError(f"not a WordNet ID: {s!r}")
        wnids.append(s)
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise ValueError(f"expected 100 unique WordNet IDs, got {len(wnids)} ({len(set(wnids))} unique)")
    return wnids


def imagenet1k_wnids() -> list[str]:
    """The 1,000 ImageNet-1k WordNet IDs in label order (timm's list; sorted, which is also the
    Hugging Face ILSVRC/imagenet-1k label order per its dataset card)."""
    import timm.data

    path = Path(timm.data.__file__).parent / "_info" / "imagenet_synsets.txt"
    wnids = path.read_text(encoding="ascii").split()
    if len(wnids) != 1000 or wnids != sorted(wnids):
        raise RuntimeError(f"unexpected ImageNet-1k synset list at {path}")
    return wnids


def in100_label_map(in100: list[str], order_1k: list[str]) -> dict:
    """Map the 100 classes to ImageNet-1k label indices and to training labels 0..99.

    Training label = rank of the class's ImageNet-1k index among the 100 (= rank of its WordNet ID
    in sorted order, as the 1k order is sorted). Returns hf_index (sorted, [100]), wnids (in label
    order) and lut (int64 [1000]: 1k index -> 0..99, or -1)."""
    pos = {w: i for i, w in enumerate(order_1k)}
    missing = [w for w in in100 if w not in pos]
    if missing:
        raise ValueError(f"{len(missing)} classes are not in ImageNet-1k: {missing[:5]}")
    hf_index = np.array(sorted(pos[w] for w in in100), dtype=np.int64)
    lut = np.full(len(order_1k), -1, dtype=np.int64)
    lut[hf_index] = np.arange(len(hf_index))
    return {"hf_index": hf_index, "wnids": [order_1k[i] for i in hf_index], "lut": lut}


# ----------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------
def default_cache_dir() -> Path:
    """$IN100_CACHE_DIR if set, else <repo>/cache."""
    env = os.environ.get("IN100_CACHE_DIR")
    return Path(env) if env else REPO_ROOT / "cache"


def _cache_dir(cache_dir) -> Path:
    return default_cache_dir() if cache_dir is None else Path(cache_dir)


def _image_file(d: Path, split: str) -> Path:
    """The cache's image file of a split: in100_<split>72_uint8.npy, else the one in100_<split><S>_uint8.npy
    present (a cache built with build_imagenet100.py --size S, e.g. 108 for 96 x 96 inputs)."""
    f = d / (TRAIN_FILE if split == "train" else VAL_FILE)
    if f.exists():
        return f
    cands = sorted(p for p in d.glob(f"in100_{split}*_uint8.npy") if re.fullmatch(rf"in100_{split}\d+_uint8\.npy", p.name))
    if len(cands) != 1:
        raise FileNotFoundError(f"{d}: expected one in100_{split}<size>_uint8.npy, found {[p.name for p in cands]}")
    return cands[0]


def cache_store_size(cache_dir=None) -> int:
    """Stored image size of the cache (72, or the --size it was built with)."""
    return int(np.load(_image_file(_cache_dir(cache_dir), "train"), mmap_mode="r").shape[1])


def load_in100(cache_dir=None, mmap: bool = True) -> dict:
    """{train_images, train_labels, val_images, val_labels, meta} from the cache (images memmapped)."""
    d = _cache_dir(cache_dir)
    meta_path = d / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing: run scripts/build_imagenet100.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("extra", {}).get("complete"):
        raise RuntimeError(f"{meta_path}: the ImageNet-100 build did not finish; rerun the build")
    mode = "r" if mmap else None
    out = {
        "train_images": np.load(_image_file(d, "train"), mmap_mode=mode),
        "train_labels": np.load(d / TRAIN_LABELS),
        "val_images": np.load(_image_file(d, "val"), mmap_mode=mode),
        "val_labels": np.load(d / VAL_LABELS),
        "meta": meta,
    }
    for split in ("train", "val"):
        im, lab = out[f"{split}_images"], out[f"{split}_labels"]
        if im.ndim != 4 or im.shape[1] != im.shape[2] or im.shape[3] != 3 or im.dtype != np.uint8:
            raise ValueError(f"{split} images must be uint8 [N, S, S, 3], got {im.dtype} {im.shape}")
        if len(im) != len(lab):
            raise ValueError(f"{split}: {len(im)} images but {len(lab)} labels")
    return out


def labels_fingerprint(labels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).hexdigest()[:16]


def write_synthetic_cache(cache_dir, num_classes: int = 10, train_per_class: int = 8, val_per_class: int = 2,
                          size: int = STORE_SIZE, seed: int = 0) -> Path:
    """Random images in the cache layout (tests and --tiny). A weak class signal (a colour tint and
    a bright square whose position depends on the class) makes the task learnable."""
    from .meta import write_meta

    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tint = rng.integers(0, 256, size=(num_classes, 1, 1, 3))      # one tint per class, shared by both splits

    def make(per_class):
        labels = np.repeat(np.arange(num_classes), per_class)
        rng.shuffle(labels)
        imgs = rng.integers(0, 256, size=(len(labels), size, size, 3), dtype=np.uint8)
        imgs = ((3 * imgs.astype(np.int64) + tint[labels]) // 4).astype(np.uint8)
        q = size // 4
        for i, c in enumerate(labels):
            r0, c0 = (c % 3) * q, (c // 3 % 3) * q
            imgs[i, r0:r0 + q, c0:c0 + q] = 255
        return imgs, labels.astype(np.int64)

    tr, trl = make(train_per_class)
    va, val = make(val_per_class)
    np.save(d / f"in100_train{int(size)}_uint8.npy", tr)
    np.save(d / TRAIN_LABELS, trl)
    np.save(d / f"in100_val{int(size)}_uint8.npy", va)
    np.save(d / VAL_LABELS, val)
    np.savez(d / INDEX_FILE, train_src=np.stack([np.zeros(len(trl)), np.arange(len(trl))], 1).astype(np.int64),
             val_src=np.stack([np.zeros(len(val)), np.arange(len(val))], 1).astype(np.int64))
    write_meta(d, name=META_FILE, experiment="IN100-BUILD", config={"synthetic": True, "seed": seed},
               seeds={"synthetic": seed},
               extra={"complete": True, "synthetic": True, "size": size, "num_classes": num_classes,
                      "n_train": int(len(trl)), "n_val": int(len(val)),
                      "train_labels_sha": labels_fingerprint(trl), "val_labels_sha": labels_fingerprint(val)})
    return d


# ----------------------------------------------------------------------------------------------
# Model and adapter
# ----------------------------------------------------------------------------------------------
def build_model(arch: str = "vit_s8", num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE,
                patch_size: int | None = None, drop_path_rate: float = 0.0) -> nn.Module:
    """timm VisionTransformer for the controlled study (random init from the global torch RNG).
    patch_size None: the architecture's own (arch_patch_size)."""
    from timm.models.vision_transformer import VisionTransformer

    from .adapters_timm import check_layout

    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(ARCHS)}")
    spec = ARCHS[arch]
    patch_size = arch_patch_size(arch) if patch_size is None else int(patch_size)
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                              embed_dim=spec["embed_dim"], depth=spec["depth"], num_heads=spec["num_heads"],
                              mlp_ratio=4.0, qkv_bias=True, class_token=True, global_pool="token",
                              init_values=None, drop_path_rate=float(drop_path_rate))
    check_layout(model, img_size)
    g = img_size // patch_size
    if (int(model.num_prefix_tokens) != 1 or model.no_embed_class
            or tuple(model.pos_embed.shape) != (1, 1 + g * g, spec["embed_dim"])):
        raise RuntimeError("unexpected token layout: need CLS only, no registers, CLS in the position table")
    if not all(isinstance(b.ls1, nn.Identity) and isinstance(b.ls2, nn.Identity) for b in model.blocks):
        raise RuntimeError("LayerScale should be off")
    return model


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class ControlledAdapter(ViTAdapter):
    """A controlled-study ViT behind the mirror.adapters.ViTAdapter interface.

    blocks, depth, embed_dim, mlp_hidden, grid_hw (8, 8), n_prefix_tokens (1), n_tokens (65),
    layer0_tokens, forward_tokens, cls_feature (final-norm CLS), pos_embed_grid, plus
    mean / std (ImageNet), interpolation, crop_pct, feature ("cls"), logits (the trained head) and
    `info` (arch, condition, seed, epoch, step, validation accuracy of the checkpoint)."""

    def __init__(self, model: nn.Module, arch: str, img_size: int = IMG_SIZE, info: dict | None = None):
        patch = int(model.patch_embed.patch_size[0])
        super().__init__(model, img_size=img_size, patch_size=patch, n_prefix_tokens=int(model.num_prefix_tokens))
        self.arch = arch
        self.info = dict(info or {})
        self.model_name = f"controlled_{arch}"
        self.feature = "cls"
        self.has_cls_token = True
        self.has_layerscale = False
        self.mean, self.std = tuple(IMAGENET_MEAN), tuple(IMAGENET_STD)
        self.interpolation = "bilinear"
        self.crop_pct = IMG_SIZE / STORE_SIZE     # the same centre-crop ratio at every size (64 of 72, 96 of 108)
        self.num_classes = int(model.num_classes)
        self.condition = self.info.get("condition")
        self.epoch = self.info.get("epoch")

    def layer0_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        t = m.patch_embed(x)
        t = m._pos_embed(t)
        t = m.patch_drop(t)
        return m.norm_pre(t)

    def pos_embed_grid(self, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        H, W = self.grid_hw
        return m.pos_embed[0, self.n_prefix_tokens:].reshape(H, W, -1)

    def logits(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        """Class logits of the trained head (equal to model(x))."""
        m = self._m(model)
        return m.forward_head(m.norm(self.forward_tokens(x, m)))


# ----------------------------------------------------------------------------------------------
# Symmetry actions and augmentation
# ----------------------------------------------------------------------------------------------
def canonical_view(view) -> str:
    v = _VIEW_ALIASES.get(view, view)
    if v not in ALL_VIEWS:
        raise ValueError(f"unknown view {view!r}; known: {ALL_VIEWS}")
    return v


def apply_view(x: torch.Tensor, view) -> torch.Tensor:
    """One symmetry action on a batch [..., H, W] (conventions in the module docstring)."""
    v = canonical_view(view)
    if v == "orig":
        return x
    if v == "LR":
        return torch.flip(x, dims=[-1])
    if v == "UD":
        return torch.flip(x, dims=[-2])
    if v == "LRUD":
        return torch.flip(x, dims=[-2, -1])
    k = {"R90": 1, "R180": 2, "R270": 3}[v]
    return torch.rot90(x, k=k, dims=(-2, -1))


def view_token_permutation(view, grid: int) -> np.ndarray:
    """perm[i] = index of the patch token of the transformed image that shows what token i of the
    original shows (row-major i = r * G + c). R90: (r, c) -> (G-1-c, r) (PROJECT_NEXT_PHASE.md 4.3)."""
    v = canonical_view(view)
    G = int(grid)
    r, c = np.divmod(np.arange(G * G), G)
    m = {"orig": (r, c), "LR": (r, G - 1 - c), "UD": (G - 1 - r, c), "LRUD": (G - 1 - r, G - 1 - c),
         "R90": (G - 1 - c, r), "R180": (G - 1 - r, G - 1 - c), "R270": (c, G - 1 - r)}[v]
    return (m[0] * G + m[1]).astype(np.int64)


def sample_symmetry(n: int, condition: str, generator: torch.Generator | None = None) -> torch.Tensor:
    """int64 [n, 3]: (LR flag, UD flag, k quarter turns). Every condition makes the same draws."""
    spec = CONDITIONS[condition]
    u = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    k = torch.randint(0, 4, (n,), generator=generator, dtype=torch.int64)
    lr = (u[:, 0] < float(spec["p_lr"])).to(torch.int64)
    ud = (u[:, 1] < float(spec["p_ud"])).to(torch.int64)
    if not spec["rot90"]:
        k = torch.zeros_like(k)
    return torch.stack([lr, ud, k], dim=1)


def apply_symmetry(x: torch.Tensor, sym: torch.Tensor, condition: str | None = None) -> torch.Tensor:
    """Per-image symmetry on x [B, C, H, W]: LR flip if sym[:, 0], then UD flip if sym[:, 1], then
    rot90 by sym[:, 2] quarter turns. Vectorised (torch.where), no host synchronisation. With
    `condition`, steps whose probability is zero are skipped (the flags are zero anyway)."""
    spec = CONDITIONS[condition] if condition is not None else {"p_lr": 1, "p_ud": 1, "rot90": True}
    sym = sym.to(x.device)
    if spec["p_lr"] > 0:
        x = torch.where(sym[:, 0].bool().view(-1, 1, 1, 1), torch.flip(x, dims=[-1]), x)
    if spec["p_ud"] > 0:
        x = torch.where(sym[:, 1].bool().view(-1, 1, 1, 1), torch.flip(x, dims=[-2]), x)
    if spec["rot90"]:
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("rotation needs square images")
        k = sym[:, 2].view(-1, 1, 1, 1)
        out = x
        for q in (1, 2, 3):
            out = torch.where(k == q, torch.rot90(x, k=q, dims=(-2, -1)), out)
        x = out
    return x


_NORM_CACHE: dict = {}


def normalise(x: torch.Tensor) -> torch.Tensor:
    """0..255 values (uint8 or float) [B, 3, H, W] -> ImageNet-normalised float32."""
    key = (x.device.type, x.device.index)
    if key not in _NORM_CACHE:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        _NORM_CACHE[key] = (mean * 255.0, std * 255.0)
    mean, std = _NORM_CACHE[key]
    return (x.to(torch.float32) - mean) / std


def _seed_int(*parts: int) -> int:
    state = np.random.SeedSequence([int(p) for p in parts]).generate_state(2, dtype=np.uint32)
    return int((int(state[0]) << 31) ^ int(state[1]))


def epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    """Training order of one epoch (a pure function of seed and epoch)."""
    return np.random.default_rng([int(seed), 7919, int(epoch)]).permutation(n)


def epoch_aug_params(n: int, seed: int, epoch: int, condition: str, store_size: int = STORE_SIZE,
                     crop_scale=CROP_SCALE, crop_ratio=CROP_RATIO) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop boxes (float64 [n, 5], flip column zero) and symmetry draws (int64 [n, 3]) for the n
    image slots of one epoch, on the CPU. Pure function of (seed, epoch, condition)."""
    from .heal import sample_crop_params

    g_crop = torch.Generator().manual_seed(_seed_int(seed, epoch, 1))
    g_sym = torch.Generator().manual_seed(_seed_int(seed, epoch, 2))
    crops = sample_crop_params(n, store_size, store_size, scale=crop_scale, ratio=crop_ratio, hflip_p=0.0,
                               generator=g_crop)
    return crops, sample_symmetry(n, condition, g_sym)


def augment_train(x_u8: torch.Tensor, crops: torch.Tensor, sym: torch.Tensor, out_size: int = IMG_SIZE,
                  condition: str | None = None) -> torch.Tensor:
    """uint8 [B, S, S, 3] -> random resized crop -> symmetry -> normalised float32 [B, 3, out, out]."""
    from .heal import crop_resize_flip

    x = x_u8.permute(0, 3, 1, 2).to(torch.float32)
    x = crop_resize_flip(x, crops, out_size)
    x = apply_symmetry(x, sym, condition)
    return normalise(x)


def centre_crop_u8(x_u8: torch.Tensor, out_size: int = IMG_SIZE) -> torch.Tensor:
    """uint8 [B, S, S, 3] -> centre out x out crop as uint8 [B, 3, out, out]."""
    S = x_u8.shape[1]
    if S < out_size:
        raise ValueError(f"image size {S} is smaller than {out_size}")
    o = (S - out_size) // 2
    return x_u8[:, o:o + out_size, o:o + out_size].permute(0, 3, 1, 2)


def autocast_ctx(device: torch.device, amp: str):
    if amp == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_views(model: nn.Module, images_u8, labels, views=VIEWS, batch_size: int = 1000, amp: str = "none",
                   out_size: int = IMG_SIZE) -> dict:
    """Top-1 (%) of the model's head on the centre view of every image under each view."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    imgs = images_u8 if torch.is_tensor(images_u8) else torch.from_numpy(np.ascontiguousarray(images_u8))
    y = torch.as_tensor(np.asarray(labels) if not torch.is_tensor(labels) else labels).to(device)
    views = [canonical_view(v) for v in views]
    correct = {v: torch.zeros((), dtype=torch.int64, device=device) for v in views}
    try:
        for s in range(0, len(y), int(batch_size)):
            base = centre_crop_u8(imgs[s:s + batch_size].to(device, non_blocking=True), out_size)
            yb = y[s:s + batch_size]
            for v in views:
                with autocast_ctx(device, amp):
                    logits = model(normalise(apply_view(base, v)))
                correct[v] += (logits.argmax(dim=-1) == yb).sum()
    finally:
        model.train(was_training)
    n = int(len(y))
    return {**{f"top1_{v}": 100.0 * int(correct[v]) / max(n, 1) for v in views}, "n_val": n}


# ----------------------------------------------------------------------------------------------
# Loaders for the measurement code
# ----------------------------------------------------------------------------------------------
class _CentreViewDataset(torch.utils.data.Dataset):
    """Centre out x out crop of stored uint8 [N, S, S, 3] images, then a view: uint8 [3, out, out]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, rows: np.ndarray, view=None,
                 out_size: int | None = None):
        self.images, self.labels = images, np.asarray(labels)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.view = canonical_view(view)
        # None: the input size that goes with the stored size (64 for 72, 96 for 108)
        self.out = view_size_for_store(images.shape[1]) if out_size is None else int(out_size)
        self.offset = (int(images.shape[1]) - self.out) // 2

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        j, o = int(self.rows[i]), self.offset
        img = torch.from_numpy(np.ascontiguousarray(self.images[j, o:o + self.out, o:o + self.out]))
        return apply_view(img.permute(2, 0, 1), self.view).contiguous(), int(self.labels[j])


def _loader(images, labels, rows, view, batch_size, num_workers, pin_memory):
    ds = _CentreViewDataset(images, labels, rows, view)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
                                       pin_memory=bool(pin_memory), drop_last=False)


FIT_BASE_SEED = 20261114


def fit_rows(labels: np.ndarray, seed: int, n: int = 2000) -> np.ndarray:
    """Class-balanced train rows for fit seed `seed`: n // C images per class (the first n % C
    classes of a seeded class order get one more). Each class's images are shuffled once with a
    fixed seed and fit seed s takes slot s of that order, so fit sets of different seeds are
    disjoint. Sorted row indices."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    C = len(classes)
    q, r = divmod(int(n), C)
    slot = q + (1 if r else 0)
    extra = set(np.random.default_rng([FIT_BASE_SEED, int(seed)]).permutation(C)[:r].tolist())
    rows = []
    for ci, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        idx = idx[np.random.default_rng([FIT_BASE_SEED, int(c)]).permutation(len(idx))]
        take = q + (1 if ci in extra else 0)
        start = int(seed) * slot
        if start + take > len(idx):
            raise ValueError(f"class {c} has {len(idx)} images; fit seed {seed} needs up to {start + take}")
        rows.append(idx[start:start + take])
    return np.sort(np.concatenate(rows)).astype(np.int64)


def fit_loader(seed: int, n: int = 2000, batch_size: int = 64, cache_dir=None, num_workers: int = 0,
               pin_memory: bool = False, view=None):
    """ImageNet-100 train images (centre 64 x 64 view, uint8 [B, 3, 64, 64]) for fit seed `seed`."""
    d = load_in100(cache_dir)
    rows = fit_rows(d["train_labels"], seed, n)
    return _loader(d["train_images"], d["train_labels"], rows, view, batch_size, num_workers, pin_memory)


def val_loader(transform=None, batch_size: int = 256, cache_dir=None, n: int | None = None, num_workers: int = 0,
               pin_memory: bool = False):
    """ImageNet-100 validation images, centre 64 x 64 view, then `transform` (None, "LR", "UD", "R90",
    or "LRUD", "R180", "R270"). n: a fixed seeded subset of n images (default all), sorted."""
    d = load_in100(cache_dir)
    N = len(d["val_labels"])
    rows = np.arange(N) if n is None else np.sort(np.random.default_rng(0).permutation(N)[:int(n)])
    return _loader(d["val_images"], d["val_labels"], rows, transform, batch_size, num_workers, pin_memory)


# ----------------------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------------------
def checkpoint_name(epoch: int) -> str:
    return f"ckpt_e{int(epoch):03d}.pt"


def list_checkpoints(run_dir) -> list[tuple[int, Path]]:
    out = []
    for p in Path(run_dir).glob("ckpt_e*.pt"):
        m = re.fullmatch(r"ckpt_e(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def ckpt_epochs_for(epochs: int, requested=DEFAULT_CKPT_EPOCHS) -> list[int]:
    """Requested checkpoint epochs that fall within the run, plus the final epoch.
    100 epochs: 0, 2, 5, 10, 25, 50, 100. 60 epochs: 0, 2, 5, 10, 25, 50, 60."""
    return sorted({int(e) for e in requested if 0 <= int(e) <= int(epochs)} | {int(epochs)})


def _model_state(model: nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path} is not a controlled-training checkpoint")
    return ckpt


def load_checkpoint_model(path, device="cpu") -> tuple[nn.Module, dict]:
    """(model in eval mode on `device`, checkpoint dict without the weights)."""
    ckpt = load_checkpoint(path)
    model = build_model(ckpt["arch"], num_classes=int(ckpt["num_classes"]), img_size=int(ckpt["img_size"]),
                        patch_size=int(ckpt["patch_size"]))
    model.load_state_dict(ckpt["model"], strict=True)
    info = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "rng", "epoch_accum")}
    return model.to(device).eval(), info


def load_checkpoint_adapter(path, device="cpu") -> ControlledAdapter:
    """A ControlledAdapter (mirror.adapters.ViTAdapter interface) for a ckpt_e<NNN>.pt or last.pt file."""
    model, info = load_checkpoint_model(path, device)
    info["path"] = str(path)
    return ControlledAdapter(model, info["arch"], img_size=int(info["img_size"]), info=info)


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    exp: str = "CORE-01"
    arch: str = "vit_s8"
    condition: str = "none"
    seed: int = 0
    epochs: int = 100
    ckpt_epochs: tuple = DEFAULT_CKPT_EPOCHS
    batch_size: int = 512
    lr: float = 1e-3                     # at lr_ref_batch; scaled linearly with batch_size
    lr_ref_batch: int = 512
    weight_decay: float = 0.05
    warmup_epochs: float = 10.0
    min_lr_ratio: float = 0.0            # cosine floor as a fraction of the peak lr
    betas: tuple = (0.9, 0.999)
    label_smoothing: float = 0.1
    drop_path: float = 0.0
    grad_clip: float = 0.0               # 0: off (the norm is still logged)
    img_size: int = IMG_SIZE
    crop_scale: tuple = CROP_SCALE
    crop_ratio: tuple = CROP_RATIO
    amp: str = "auto"                    # auto: bf16 on CUDA, none on CPU
    compile: str = "auto"                # auto: on for CUDA, off on CPU; "on" / "off"
    eval_batch: int = 1000
    save_every_min: float = 5.0          # "last" saves at least this often, inside or at the end of epochs (0: off)
    save_every_epochs: int = 0           # also every n epochs (0: off); always at checkpoint epochs and the end
    save_every_steps: int = 0            # mid-epoch "last" saves every n steps (0: off; for tests)
    log_every: int = 50
    cache_dir: str | None = None
    out_root: str = "runs"
    run_dir: str | None = None
    device: str = "cuda"
    stop_after_steps: int | None = None  # save the last state and stop (simulated preemption; tests)
    force_new: bool = False


# Settings that must match for a resume to continue the same run.
RESUME_KEYS = ("arch", "condition", "seed", "epochs", "ckpt_epochs", "batch_size", "lr", "lr_ref_batch",
               "weight_decay", "warmup_epochs", "min_lr_ratio", "betas", "label_smoothing", "drop_path",
               "grad_clip", "img_size", "crop_scale", "crop_ratio", "amp")


def run_dir_for(cfg: TrainConfig) -> Path:
    if cfg.run_dir:
        return Path(cfg.run_dir)
    return Path(cfg.out_root) / cfg.exp / f"{cfg.arch}_{cfg.condition}_s{cfg.seed}"


def _resolve(cfg: TrainConfig, device: torch.device) -> tuple[str, bool]:
    amp = cfg.amp if cfg.amp != "auto" else ("bf16" if device.type == "cuda" else "none")
    comp = cfg.compile if cfg.compile != "auto" else ("on" if device.type == "cuda" else "off")
    return amp, comp == "on"


def _resume_view(cfg: TrainConfig, amp: str, data_fp: dict) -> dict:
    d = {k: getattr(cfg, k) for k in RESUME_KEYS}
    d["amp"] = amp
    d = json.loads(json.dumps(d))       # tuples -> lists, so saved and current compare equal
    d["data"] = data_fp
    return d


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on weight matrices only: not on biases, norm affines, pos_embed or cls_token."""
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name in skip else decay).append(p)
    return [{"params": decay, "weight_decay": float(weight_decay)}, {"params": no_decay, "weight_decay": 0.0}]


def _grad_norm(params: list) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    if hasattr(torch.nn.utils, "get_total_norm"):
        return torch.nn.utils.get_total_norm(grads, 2.0)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


class _Csv:
    """A CSV table that can be cut back to the rows of a resumed state."""

    def __init__(self, path: Path, columns: list):
        self.path, self.columns = Path(path), list(columns)

    def reset(self, keep=None) -> None:
        rows = []
        if keep is not None and self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if keep(r)]
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns)
            w.writeheader()
            w.writerows(rows)

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writerow({k: row.get(k, "") for k in self.columns})

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


TRAIN_LOG_COLUMNS = ["epoch", "step", "lr_end", "train_loss", "train_top1", "grad_norm_mean", "epoch_seconds",
                     "images_per_s", "peak_mem_alloc_gib", "peak_mem_reserved_gib", "steps_timed", "resumed",
                     "wall_seconds"]
VAL_COLUMNS = ["epoch", "step", "top1_orig", "top1_LR", "top1_UD", "top1_R90", "n_val", "eval_seconds"]


def timing_summary(train_rows: list[dict], val_rows: list[dict], wall_seconds: float, epochs: int,
                   n_ckpts: int) -> dict:
    """Timing for the MUST-12 decision (100 against 60 epochs; reduce if a run takes over 25 minutes).

    steady = median over fully timed epochs after the first (the first includes compilation).
    projected_minutes_E = overhead + first epoch + (E - 1) x steady epoch + checkpoints x mean
    evaluation time, where the overhead is the measured wall time not spent in timed epochs or
    evaluations (start-up, data transfer, checkpoint writes)."""
    full = [r for r in train_rows if int(r["resumed"]) == 0]
    ep_s = [float(r["epoch_seconds"]) for r in full]
    steady = ep_s[1:] if len(ep_s) > 1 else ep_s
    steady_epoch = float(np.median(steady)) if steady else float("nan")
    ips = [float(r["images_per_s"]) for r in full[1:]] or [float(r["images_per_s"]) for r in full]
    eval_s = [float(r["eval_seconds"]) for r in val_rows]
    train_s = float(sum(float(r["epoch_seconds"]) for r in train_rows))
    overhead = max(0.0, float(wall_seconds) - train_s - float(sum(eval_s)))
    mean_eval = float(np.mean(eval_s)) if eval_s else 0.0
    out = {
        "wall_seconds": round(float(wall_seconds), 1),
        "wall_minutes": round(float(wall_seconds) / 60.0, 2),
        "train_seconds": round(train_s, 1),
        "eval_seconds": round(float(sum(eval_s)), 1),
        "first_epoch_seconds": round(ep_s[0], 2) if ep_s else None,
        "steady_epoch_seconds": round(steady_epoch, 3),
        "steady_images_per_s": round(float(np.median(ips)), 1) if ips else None,
        "peak_mem_alloc_gib": max((float(r["peak_mem_alloc_gib"]) for r in train_rows), default=None),
        "peak_mem_reserved_gib": max((float(r["peak_mem_reserved_gib"]) for r in train_rows), default=None),
        "overhead_seconds": round(overhead, 1),
        "epochs_run": len(train_rows),
    }
    for e in (100, 60):
        if math.isfinite(steady_epoch) and ep_s:
            n_ck = len(ckpt_epochs_for(e))
            secs = overhead + ep_s[0] + (e - 1) * steady_epoch + n_ck * mean_eval
            out[f"projected_minutes_{e}"] = round(secs / 60.0, 2)
    out["rule"] = "MUST-12: reduce every CORE-01 run to 60 epochs if the pilot run takes over 25 minutes"
    return out


def train(cfg: TrainConfig) -> dict:
    """Train one controlled run. Resumes from <run_dir>/last.pt when present.

    Outputs in run_dir:
      ckpt_e<NNN>.pt  weights (fp32) at each checkpoint epoch (0 = initialisation), with the
                      validation accuracies of that epoch; NNN zero-padded to three digits
      last.pt         model, optimiser, schedule, RNG, step and partial-epoch sums; written at least
                      every save_every_min minutes (inside or at the end of an epoch), at every
                      checkpoint epoch, at the end, and on a requested stop
      train_log.csv   one row per epoch: loss, train top-1 on augmented images, lr, mean gradient
                      norm, epoch seconds (training steps only), images per second, peak memory
                      (GiB, reset each epoch), cumulative wall time
      val_acc.csv     one row per checkpoint: top-1 on original, LR, UD and R90 validation images
      meta.json       mirror.meta record; extra holds the timing summary and the final accuracies
    Resume is exact: data order, crops and flips are pure functions of (seed, epoch), and the
    optimiser state, step and RNG state are restored. On the CPU the resumed run matches the
    uninterrupted run bit for bit (tested); on a GPU some kernels are not bitwise deterministic.
    """
    from .heal import lr_at
    from .meta import write_meta

    session_t0 = time.time()
    if cfg.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {cfg.condition!r}; known: {list(CONDITIONS)}")
    device = torch.device(cfg.device)
    amp, use_compile = _resolve(cfg, device)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- data -----------------------------------------------------------------------------
    data = load_in100(cfg.cache_dir)
    tr_lab_np, va_lab_np = data["train_labels"], data["val_labels"]
    num_classes = int(max(tr_lab_np.max(), va_lab_np.max()) + 1)
    n_train, B = int(len(tr_lab_np)), int(cfg.batch_size)
    spe = n_train // B
    if spe < 1:
        raise ValueError(f"batch {B} is larger than the training set ({n_train})")
    total_steps = spe * int(cfg.epochs)
    warmup_steps = int(round(float(cfg.warmup_epochs) * spe))
    base_lr = float(cfg.lr) * B / float(cfg.lr_ref_batch)
    ckpt_set = ckpt_epochs_for(cfg.epochs, cfg.ckpt_epochs)
    data_fp = {"n_train": n_train, "n_val": int(len(va_lab_np)),
               "train_labels_sha": labels_fingerprint(tr_lab_np), "val_labels_sha": labels_fingerprint(va_lab_np),
               "size": int(data["train_images"].shape[1])}
    if int(cfg.img_size) != view_size_for_store(data_fp["size"]):
        raise ValueError(f"img_size {cfg.img_size} does not go with images stored at {data_fp['size']} "
                         f"(expected {view_size_for_store(data_fp['size'])}): check the cache folder")
    resume_view = _resume_view(cfg, amp, data_fp)

    # ---- model and optimiser ---------------------------------------------------------------
    torch.manual_seed(int(cfg.seed))
    model = build_model(cfg.arch, num_classes=num_classes, img_size=cfg.img_size,
                        drop_path_rate=cfg.drop_path).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt_kw = {"lr": base_lr, "betas": tuple(cfg.betas)}
    if device.type == "cuda":
        opt_kw["fused"] = True
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), **opt_kw)

    ckpt_common = {"format": CKPT_FORMAT, "exp": cfg.exp, "arch": cfg.arch, "condition": cfg.condition,
                   "condition_spec": CONDITIONS[cfg.condition], "seed": int(cfg.seed), "img_size": int(cfg.img_size),
                   "patch_size": arch_patch_size(cfg.arch), "num_classes": num_classes,
                   "drop_path_rate": float(cfg.drop_path), "store_size": data_fp["size"], "config": resume_view}

    # ---- resume or fresh start -------------------------------------------------------------
    last_path = run_dir / "last.pt"
    train_csv, val_csv = _Csv(run_dir / "train_log.csv", TRAIN_LOG_COLUMNS), _Csv(run_dir / "val_acc.csv", VAL_COLUMNS)
    meta_path = run_dir / "meta.json"
    if cfg.force_new:
        for p in [last_path, meta_path, train_csv.path, val_csv.path] + [p for _, p in list_checkpoints(run_dir)]:
            if p.exists():
                p.unlink()
    step, wall_before, accum = 0, 0.0, None
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state.get("format") != CKPT_FORMAT or state.get("kind") != "last":
            raise ValueError(f"{last_path} is not a controlled-training last state")
        if state["config"] != resume_view:
            diff = sorted(k for k in set(state["config"]) | set(resume_view)
                          if state["config"].get(k) != resume_view.get(k))
            raise ValueError(f"{last_path} was written with different settings ({diff}); "
                             "use the same settings, or --force-new to start over")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        step = int(state["step"])
        wall_before = float(state.get("wall_seconds", 0.0))
        accum = state.get("epoch_accum")
        done = step // spe
        train_csv.reset(keep=lambda r: int(r["epoch"]) <= done)
        val_csv.reset(keep=lambda r: int(r["epoch"]) <= done)
        log(f"resumed {run_dir} at step {step} of {total_steps} (epoch {done} done)")
        if step >= total_steps:
            log("run already complete")
            return _finish(cfg, run_dir, train_csv, val_csv, wall_before, device, amp, use_compile, spe,
                           total_steps, ckpt_set, model, data_fp, base_lr, warmup_steps, write=False)
    else:
        if meta_path.exists() and json.loads(meta_path.read_text()).get("extra", {}).get("status") == "complete":
            raise RuntimeError(f"{run_dir} holds a completed run but no last.pt; use --force-new to redo it")
        for _, p in list_checkpoints(run_dir):      # leftovers of a run stopped before its first save
            p.unlink()
        train_csv.reset()
        val_csv.reset()

    write_meta(run_dir, experiment=cfg.exp, config=asdict(cfg), seeds=_seeds(cfg),
               models=_models(cfg, model), device=device, wall_seconds=wall_before,
               extra={"status": "running", "steps_per_epoch": spe, "total_steps": total_steps,
                      "ckpt_epochs": ckpt_set, "data": data_fp, "amp": amp, "compile": use_compile,
                      "base_lr": base_lr, "warmup_steps": warmup_steps})

    # ---- tensors on the device ---------------------------------------------------------------
    t_load = time.time()
    tr_u8 = torch.from_numpy(np.array(data["train_images"], dtype=np.uint8)).to(device)   # memmap -> RAM -> device
    tr_y = torch.from_numpy(np.ascontiguousarray(tr_lab_np, dtype=np.int64)).to(device)
    va_u8 = torch.from_numpy(np.array(data["val_images"], dtype=np.uint8)).to(device)
    va_y = torch.from_numpy(np.ascontiguousarray(va_lab_np, dtype=np.int64)).to(device)
    log(f"data on {device}: train {tuple(tr_u8.shape)}, val {tuple(va_u8.shape)} ({time.time() - t_load:.1f} s); "
        f"{spe} steps per epoch, {total_steps} steps, peak lr {base_lr:g}, warm-up {warmup_steps} steps")
    gpu_info = {}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        gpu_info = {"gpu_free_gib_at_start": round(free / 2**30, 2), "gpu_total_gib": round(total / 2**30, 2),
                    "gpu_frac_cap": "set by the caller (scripts/train_controlled.py --gpu-frac)"}
        log(f"GPU memory free at start (after loading the data): {free / 2**30:.1f} of {total / 2**30:.1f} GiB")

    compile_error = None
    fwd = model
    if use_compile:
        try:
            fwd = torch.compile(model, dynamic=False)
        except Exception as exc:            # noqa: BLE001  (a broken Inductor install fails here)
            compile_error = f"{type(exc).__name__}: {exc}"[:500]
            log(f"torch.compile failed ({compile_error}); training in eager mode")
    compiled_ok = fwd is model             # becomes True after the first compiled step succeeds

    def forward_backward(x, y):
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=float(cfg.label_smoothing))
        loss.backward()
        return logits, loss

    def evaluate_and_save(epoch: int) -> None:
        t = time.time()
        acc = evaluate_views(model, va_u8, va_y, VIEWS, cfg.eval_batch, amp, cfg.img_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ev = time.time() - t
        val_csv.append({"epoch": epoch, "step": step, **{k: f"{v:.4f}" if isinstance(v, float) else v
                                                          for k, v in acc.items()}, "eval_seconds": f"{ev:.2f}"})
        _atomic_save({**ckpt_common, "kind": "checkpoint", "epoch": int(epoch), "step": int(step),
                      "val": acc, "model": _model_state(model)}, run_dir / checkpoint_name(epoch))
        log(f"epoch {epoch}: val top-1 " + ", ".join(f"{v} {acc['top1_' + v]:.2f}" for v in VIEWS)
            + f" ({ev:.1f} s)")

    def save_last(acc_state) -> None:
        rng = {"torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        _atomic_save({**ckpt_common, "kind": "last", "epoch": step // spe, "step": int(step),
                      "model": _model_state(model), "optimizer": opt.state_dict(),
                      "scheduler": {"type": "linear warm-up then cosine, per step", "base_lr": base_lr,
                                    "warmup_steps": warmup_steps, "total_steps": total_steps,
                                    "min_lr_ratio": float(cfg.min_lr_ratio), "step": int(step)},
                      "rng": rng, "epoch_accum": acc_state,
                      "wall_seconds": wall_before + time.time() - session_t0}, last_path)

    if step == 0 and 0 in ckpt_set:
        evaluate_and_save(0)

    # ---- training loop ---------------------------------------------------------------------
    last_save = time.time()
    while step < total_steps:
        epoch = step // spe
        k0 = step % spe
        perm = torch.from_numpy(epoch_permutation(n_train, cfg.seed, epoch)[:spe * B]).to(device)
        crops, sym = epoch_aug_params(spe * B, cfg.seed, epoch, cfg.condition, data_fp["size"],
                                      tuple(cfg.crop_scale), tuple(cfg.crop_ratio))
        crops, sym = crops.to(device), sym.to(device)
        if accum is not None and k0 > 0:
            acc_state = {k: v.to(device) for k, v in accum.items()}
        else:
            acc_state = {"loss_sum": torch.zeros((), dtype=torch.float32, device=device),
                         "correct": torch.zeros((), dtype=torch.int64, device=device),
                         "gn_sum": torch.zeros((), dtype=torch.float32, device=device),
                         "n": torch.zeros((), dtype=torch.int64, device=device)}
        accum = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t_ep = time.time()
        lr = base_lr
        for k in range(k0, spe):
            sl = slice(k * B, (k + 1) * B)
            idx = perm[sl]
            x = augment_train(tr_u8[idx], crops[sl], sym[sl], cfg.img_size, cfg.condition)
            y = tr_y[idx]
            lr = lr_at(step, base_lr, total_steps, warmup_steps, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            if compiled_ok:
                logits, loss = forward_backward(x, y)
            else:                               # first compiled step: fall back to eager if compiling fails
                try:
                    logits, loss = forward_backward(x, y)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:        # noqa: BLE001
                    compile_error = f"{type(exc).__name__}: {exc}"[:500]
                    log(f"torch.compile failed ({compile_error}); continuing in eager mode")
                    fwd = model
                    logits, loss = forward_backward(x, y)
                compiled_ok = True
            gn = (torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip)) if cfg.grad_clip > 0
                  else _grad_norm(params))
            opt.step()
            step += 1
            acc_state["loss_sum"] += loss.detach() * B
            acc_state["correct"] += (logits.detach().argmax(dim=-1) == y).sum()
            acc_state["gn_sum"] += gn.detach().float()
            acc_state["n"] += B
            if cfg.log_every and step % int(cfg.log_every) == 0:
                n_seen = max(int(acc_state["n"]), 1)
                log(f"e {epoch + 1}/{cfg.epochs} step {step}/{total_steps} loss {float(acc_state['loss_sum']) / n_seen:.4f} "
                    f"top1 {100.0 * int(acc_state['correct']) / n_seen:.2f} lr {lr:.3e} gn {float(gn):.3f}")
            stop_now = cfg.stop_after_steps is not None and step >= int(cfg.stop_after_steps)
            due = ((cfg.save_every_steps and step % int(cfg.save_every_steps) == 0)
                   or (cfg.save_every_min and time.time() - last_save >= 60.0 * float(cfg.save_every_min)))
            if (stop_now or due) and step % spe != 0:
                save_last(acc_state)
                last_save = time.time()
            if stop_now:
                if step % spe == 0:
                    break                       # the end-of-epoch work below saves the state, then we stop
                log(f"stopped after step {step} (simulated preemption); last state saved")
                return {"stopped": True, "step": step, "run_dir": str(run_dir)}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ep_s = time.time() - t_ep
        e = step // spe
        n_seen = max(int(acc_state["n"]), 1)
        timed = spe - k0
        peak_a = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        peak_r = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
        if e in ckpt_set:
            evaluate_and_save(e)
        train_csv.append({
            "epoch": e, "step": step, "lr_end": f"{lr:.6e}",
            "train_loss": repr(float(acc_state["loss_sum"]) / n_seen),
            "train_top1": repr(100.0 * int(acc_state["correct"]) / n_seen),
            "grad_norm_mean": repr(float(acc_state["gn_sum"]) / (n_seen // B)),
            "epoch_seconds": f"{ep_s:.3f}", "images_per_s": f"{timed * B / max(ep_s, 1e-9):.1f}",
            "peak_mem_alloc_gib": f"{peak_a:.3f}", "peak_mem_reserved_gib": f"{peak_r:.3f}",
            "steps_timed": timed, "resumed": int(k0 > 0),
            "wall_seconds": f"{wall_before + time.time() - session_t0:.1f}"})
        log(f"epoch {e}/{cfg.epochs} done: {ep_s:.1f} s, {timed * B / max(ep_s, 1e-9):.0f} img/s, "
            f"peak {peak_a:.2f} GiB allocated")
        stop_here = cfg.stop_after_steps is not None and step >= int(cfg.stop_after_steps) and step < total_steps
        # The last state (about 260 MB for vit_s8) is written at checkpoint epochs, at the end, on a
        # requested stop, every save_every_epochs epochs, and when save_every_min minutes have passed;
        # writing it after every short epoch would cost a few per cent of the wall time.
        if (stop_here or e in ckpt_set or e == cfg.epochs
                or (cfg.save_every_epochs and e % int(cfg.save_every_epochs) == 0)
                or (cfg.save_every_min and time.time() - last_save >= 60.0 * float(cfg.save_every_min))):
            save_last(None)
            last_save = time.time()
        if stop_here:
            log(f"stopped after step {step} (simulated preemption); last state saved")
            return {"stopped": True, "step": step, "run_dir": str(run_dir)}

    wall = wall_before + time.time() - session_t0
    return _finish(cfg, run_dir, train_csv, val_csv, wall, device, amp, use_compile, spe, total_steps, ckpt_set,
                   model, data_fp, base_lr, warmup_steps, write=True, compile_error=compile_error,
                   gpu_info=gpu_info)


def _seeds(cfg: TrainConfig) -> dict:
    return {"train": int(cfg.seed), "init": int(cfg.seed), "data_order": int(cfg.seed),
            "augmentation": int(cfg.seed)}


def _models(cfg: TrainConfig, model: nn.Module) -> dict:
    return {cfg.arch: {"builder": "mirror.controlled.build_model (timm VisionTransformer)",
                       **ARCHS[cfg.arch], "patch_size": arch_patch_size(cfg.arch), "img_size": int(cfg.img_size),
                       "mlp_ratio": 4.0, "qkv_bias": True, "registers": 0, "layerscale": False,
                       "num_classes": int(model.num_classes), "n_params": count_params(model)}}


def _finish(cfg, run_dir, train_csv, val_csv, wall, device, amp, use_compile, spe, total_steps, ckpt_set, model,
            data_fp, base_lr, warmup_steps, write: bool, compile_error: str | None = None,
            gpu_info: dict | None = None) -> dict:
    from .meta import write_meta

    train_rows, val_rows = train_csv.rows(), val_csv.rows()
    timing = timing_summary(train_rows, val_rows, wall, cfg.epochs, len(ckpt_set))
    final_val = val_rows[-1] if val_rows else {}
    summary = {"stopped": False, "run_dir": str(run_dir), "timing": timing, "final_val": final_val,
               "checkpoints": [str(p) for _, p in list_checkpoints(run_dir)]}
    if write:
        write_meta(run_dir, experiment=cfg.exp, config=asdict(cfg), seeds=_seeds(cfg),
                   models=_models(cfg, model), device=device, wall_seconds=wall,
                   extra={"status": "complete", "steps_per_epoch": spe, "total_steps": total_steps,
                          "ckpt_epochs": ckpt_set, "data": data_fp, "amp": amp, "compile": use_compile,
                          "base_lr": base_lr, "warmup_steps": warmup_steps, "timing": timing,
                          "final_val": final_val, "condition_spec": CONDITIONS[cfg.condition],
                          "compile_error": compile_error, "compiled": bool(use_compile and compile_error is None),
                          "gpu": gpu_info or {}})
        log(f"done: {run_dir}; wall {timing['wall_minutes']} min; steady {timing['steady_images_per_s']} img/s; "
            f"projected {timing.get('projected_minutes_100')} min at 100 epochs, "
            f"{timing.get('projected_minutes_60')} min at 60")
    return summary


def config_from_dict(d: dict) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
    return TrainConfig(**d)

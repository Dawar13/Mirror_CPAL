#!/usr/bin/env python
"""Tier 2: ImageNet linear probe (PAPER_EXPERIMENTS.md, Section 1, Tier 2).

    python scripts/tier2_linear.py --model dinov2_vitb14_reg \
        --variants original,mirror_mlp_k6,vexact_k6,vlocal_k9,svd_mlp_k6,random_k6,healed_mirror_k9,healed_svd_k9,healed_random_k9 \
        --out runs/t2
    python scripts/tier2_linear.py --model mae_b16 --variants original --out runs/t2
    python scripts/tier2_linear.py --model dinov2_vitb14 --variants original,mirror_mlp_k6,svd_mlp_k6 --out runs/t2
    python scripts/tier2_linear.py --model dinov2_vitb14_reg --variants healed_vexact_k12,healed_vexact_k6 --out runs/t2
    python scripts/tier2_linear.py --tiny --variants original,mirror_mlp_k2 --out tiny_out/t2    # CPU smoke test

Protocol ("logistic regression on [CLS, mean patch token], trained on the 100k bank, tested on
validation"):
  features   for every image, the CLS token after the final norm concatenated with the mean of the
             patch tokens after the final norm (register tokens excluded); 2 x 768 = 1536 numbers for
             ViT-B. Models without a CLS token (SigLIP2) use their own pooled embedding instead of the
             CLS token. Forward passes in bf16 autocast (as every Tier-1 number), batch 256; stored as
             float16 npz per variant and split under <out>/features/<model>/ and reused on a rerun.
  data       train: mirror.protocol.Protocol.bank_loader(), the 100k bank (100 ImageNet train images
             per class, disjoint from the fit and healing images); test: query_loader(), all 50k
             validation images. Same cached 224 px images (resize 256 bicubic, centre crop 224) for
             every model.
  probe      multinomial logistic regression with bias, features standardised with the training
             part's mean and standard deviation; loss = mean cross-entropy + wd / 2 * ||W||^2 (bias
             not penalised); full-batch L-BFGS (strong Wolfe line search) on the GPU, zero start.
  weight     wd is chosen on a fixed, class-stratified 10% split of the bank (seed --seed): fit on
  decay      the other 90%, score top-1 on the 10%; coarse grid --wd-grid (largest first, each solve
             warm-started from the previous one), then the two half-decade neighbours of the best
             value; the best held-out top-1 wins (ties go to the larger wd). The validation set is
             never used for this choice.
  final      refit on the whole bank with the chosen wd; top-1 and top-5 on the 50k validation images.
  check      k-NN top-1 (k = 20, T = 0.07, cosine) on the first half of the cached features (the CLS
             or pooled part), as a sanity check: it should be close to the Tier-1 k-NN number of the
             same variant (small differences come from float16 storage and batch size).

Variants (built with the existing code; nothing is reimplemented here):
  original                    the released model
  mirror_mlp_k<K>             Mirror Surgery on the MLPs of blocks 1..K, per-layer operators (V-local),
                              mirror.variants.build_variant("A1", "per_layer"), as in E1b and X2
  svd_mlp_k<K>                SVD of the same MLPs at matched parameters (build_variant("A1", "svd"))
  random_k<K>, random_mlp_k<K>
                              matched random structure on the MLPs (build_variant("A1",
                              "random_structure"), seed --random-seed; same operators and pairings as
                              the MLP part of the M3 random control)
  vlocal_k<K>, vexact_k<K>    M1 MLP + attention, V-local or V-exact (scripts/m1_attention.make_variant,
                              structure from runs/m1/m1_structure.npz)
  svd_attn_k<K>, random_attn_k<K>
                              the M1 baselines on MLP + attention (make_variant "svd" and "random")
  healed_<mirror|svd|random>_k<K>
                              the M3 healed student: scripts/heal.method_setup with the run's own
                              config (runs/m3_pilot/<method>/summary.json), then
                              mirror.heal.load_trained_blocks(<run>/final/trained_blocks.pt); the
                              constraint checks of the loaded student are recorded
  healed_vexact_k<K>          the healed V-exact student (scripts/heal_vexact.py: one operator A* over
                              blocks 1..K, MLP + attention, exactly mirror-equivariant over the span;
                              K = depth is the full conversion with the even head). Built as
                              scripts/heal_vexact.py builds it (its load_vexact_structure with the run's
                              own config, <vexact-dir>/k<K>/summary.json, then
                              mirror.heal_vexact.build_vexact_student), then
                              mirror.heal.load_trained_blocks(<vexact-dir>/k<K>/final/trained_blocks.pt);
                              --vexact-checkpoint names the file instead. The feature is the probe
                              feature above through the student's own final norm, which at K = depth
                              holds the even head (RMSNorm, y (I + A*)/2, affine): the same forward as
                              the k-NN feature of scripts/heal_vexact.py, so at K = depth an image and
                              its mirror give the same feature (exactly up to fp32 rounding). After
                              loading, heal_vexact's exactness probe (span_exactness on its check
                              images, fp32) and, at K = depth, the probe feature of those images
                              against their mirrors must pass, or the run stops; the values go to the
                              CSV (vexact_* columns and notes) and the log.
                              At another input size than the healing one (scripts/tier2_ade20k.py at
                              448 px), the student is built at that size the same way, every tensor of
                              the healed student except the position grid is copied in, and the healed
                              grid is interpolated with the model's own interpolate_pos_encoding (the
                              rule mirror.m1.materialise_pos_embed uses) and re-projected with A* by
                              mirror.m1.project_embeddings, as build_vexact projects a materialised
                              grid (a deviation, recorded under "position_grid"). The checks then need
                              images at that size (VariantFactory.check_xs)

Outputs under --out: t2_linear.csv (one row per model and variant: top-1, top-5, chosen weight
decay, feature dimension, k-NN check, parameter counts, timings, git hash, seed), T2_LINEAR.md,
probe/<model>/<variant>.json (every weight-decay trial, solver statistics, variant provenance),
probe/<model>/<variant>_probe.npz (W, b, mean, std), features/<model>/<variant>_<bank|val>.npz and
t2_linear_meta.json. A rerun skips cached features and finished probes (--force-features,
--force-probe).
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import make_loader, make_splits, normalize_for, synthetic_dataset  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

FEATURE_RECIPE = "t2v1: [CLS (or pooled) after final norm, mean of patch tokens after final norm]"
DEFAULT_MEMORY_FRACTION = 0.25
DEFAULT_WD_GRID = "1e-7,1e-6,1e-5,1e-4,1e-3,1e-2"
SPLITS = ("bank", "val")

# Default structure files per model (the X2 fits, M1 structure and M3 pilot; all from fit seed 0).
MODEL_DEFAULTS = {
    "dinov2_vitb14_reg": {"operators": "runs/x2/perlayer_reg_s0/results/e1_operators.npz",
                          "m1_structure": "runs/m1/m1_structure.npz", "heal_dir": "runs/m3_pilot",
                          "vexact_dir": "runs/m3_vexact"},
    "dinov2_vitb14": {"operators": "runs/x2/perlayer_plain_s0/results/e1_operators.npz",
                      "m1_structure": None, "heal_dir": None, "vexact_dir": None},
}
DEFAULT_VEXACT_CONFIG = "configs/m3_vexact.yaml"   # fallback when <vexact-dir>/k<K>/summary.json is missing

_MEMORY_FRACTION = DEFAULT_MEMORY_FRACTION


# ----------------------------------------------------------------------------------------------
# Variant names
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantName:
    name: str        # as given on the command line (used in file names and tables)
    family: str      # original | mlp | m1 | healed | healed_vexact
    method: str      # original | mirror | svd | random | vlocal | vexact
    k: int           # converted blocks (0 for the original)

    @property
    def label(self) -> str:
        return VARIANT_LABELS[(self.family, self.method)].format(k=self.k)


VARIANT_LABELS = {
    ("original", "original"): "original",
    ("mlp", "mirror"): "Mirror Surgery, MLP only, V-local, k = {k}",
    ("mlp", "svd"): "SVD at matched parameters, MLP only, k = {k}",
    ("mlp", "random"): "matched random structure, MLP only, k = {k}",
    ("m1", "vlocal"): "Mirror V-local, MLP + attention, k = {k}",
    ("m1", "vexact"): "Mirror V-exact, MLP + attention, k = {k}",
    ("m1", "svd"): "SVD at matched parameters, MLP + attention, k = {k}",
    ("m1", "random"): "matched random structure, MLP + attention, k = {k}",
    ("healed", "mirror"): "healed Mirror V-local, MLP + attention, k = {k} (M3)",
    ("healed", "svd"): "healed SVD, MLP + attention, k = {k} (M3)",
    ("healed", "random"): "healed random structure, MLP + attention, k = {k} (M3)",
    ("healed_vexact", "vexact"): "healed Mirror V-exact, MLP + attention, k = {k} (scripts/heal_vexact.py)",
}

_PATTERNS = [
    (re.compile(r"^original$"), lambda m: ("original", "original", 0)),
    (re.compile(r"^(mirror|svd|random)_mlp_k(\d+)$"), lambda m: ("mlp", m.group(1), int(m.group(2)))),
    (re.compile(r"^random_k(\d+)$"), lambda m: ("mlp", "random", int(m.group(1)))),
    (re.compile(r"^(vlocal|vexact)_k(\d+)$"), lambda m: ("m1", m.group(1), int(m.group(2)))),
    (re.compile(r"^(svd|random)_attn_k(\d+)$"), lambda m: ("m1", m.group(1), int(m.group(2)))),
    (re.compile(r"^healed_(mirror|svd|random)_k(\d+)$"), lambda m: ("healed", m.group(1), int(m.group(2)))),
    (re.compile(r"^healed_vexact_k(\d+)$"), lambda m: ("healed_vexact", "vexact", int(m.group(1)))),
]


def parse_variant(name: str) -> VariantName:
    name = name.strip()
    for pat, fn in _PATTERNS:
        m = pat.match(name)
        if m:
            family, method, k = fn(m)
            if family != "original" and k < 1:
                raise ValueError(f"{name}: k must be at least 1")
            return VariantName(name, family, method, k)
    raise ValueError(f"unknown variant {name!r}; expected original, mirror_mlp_k<K>, svd_mlp_k<K>, random_k<K>, "
                     "random_mlp_k<K>, vlocal_k<K>, vexact_k<K>, svd_attn_k<K>, random_attn_k<K>, "
                     "healed_<mirror|svd|random>_k<K> or healed_vexact_k<K>")


def parse_variants(text: str) -> list:
    out, seen = [], set()
    for part in text.split(","):
        if part.strip() and part.strip() not in seen:
            seen.add(part.strip())
            out.append(parse_variant(part))
    if not out:
        raise ValueError("no variants given")
    return out


def adapter_at(adapter, img_size: int):
    """A shallow copy of `adapter` (same model object) that describes `img_size` inputs."""
    if img_size % adapter.patch_size:
        raise ValueError(f"{img_size} px is not a multiple of the patch size {adapter.patch_size}")
    ad = copy.copy(adapter)
    ad.img_size = int(img_size)
    ad.grid_hw = (ad.img_size // ad.patch_size, ad.img_size // ad.patch_size)
    return ad


def healing_px(cfg: dict) -> int:
    """Input size a scripts/heal_vexact.py run healed at (scripts/heal.py build_teacher's rule)."""
    crop = int(cfg["heal"]["crop_size"])
    if cfg.get("tiny_mode"):
        return crop
    return int((cfg.get("model") or {}).get("img_size", crop))


def check_vexact_checkpoint_arg(variants, vexact_checkpoint) -> None:
    """--vexact-checkpoint names one file, so it can serve one healed V-exact variant only."""
    n = sum(v.family == "healed_vexact" for v in variants)
    if vexact_checkpoint and n != 1:
        raise ValueError(f"--vexact-checkpoint names one checkpoint, but {n} healed_vexact_k<K> variants were "
                         "requested; request exactly one, or use --vexact-dir (<dir>/k<K>/final/trained_blocks.pt)")


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def apply_memory_cap(fraction: float | None = None) -> None:
    """(Re)apply this process's GPU memory cap. Imported scripts set their own cap at import time,
    so this runs again after every import of scripts/heal.py or scripts/m1_attention.py."""
    global _MEMORY_FRACTION
    if fraction is not None:
        _MEMORY_FRACTION = float(fraction)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_MEMORY_FRACTION)


def load_script(name: str):
    """Import scripts/<name>.py once (registered in sys.modules), then restore this process's GPU cap."""
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
    apply_memory_cap()
    return sys.modules[key]


_SHA_CACHE: dict = {}


def sha256(path) -> str:
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2)
    os.replace(tmp, path)


def free_memory(device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
def feature_kind(adapter) -> str:
    """'cls' for models with a CLS token; 'pooled' (the model's own pooled embedding) otherwise."""
    if getattr(adapter, "feature", "cls") == "pooled" or not getattr(adapter, "has_cls_token", True):
        return "pooled"
    return "cls"


def probe_feature(adapter, x: torch.Tensor, model=None) -> torch.Tensor:
    """[CLS (or pooled) after the final norm, mean of the patch tokens after the final norm], [B, 2d]."""
    m = adapter.model if model is None else model
    t = m.norm(adapter.forward_tokens(x, m))
    p = int(adapter.n_prefix_tokens)
    first = m.forward_head(t, pre_logits=True) if feature_kind(adapter) == "pooled" else t[:, 0]
    return torch.cat([first.float(), t[:, p:].float().mean(dim=1)], dim=-1)


@torch.no_grad()
def extract_features(adapter, loader, device, model=None, amp: str | None = "bf16", dtype=np.float16,
                     log_every: int = 50, tag: str = "") -> tuple:
    """Features and labels for every image of `loader`, in loader order: (X [N, D] `dtype`, y [N] int64).
    bf16 autocast on CUDA only (CPU runs stay in float32), as mirror.knn.extract_features."""
    device = torch.device(device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and device.type == "cuda"
           else contextlib.nullcontext())
    m = adapter.model if model is None else model
    m.eval()
    n = len(loader.dataset)
    X, y, pos = None, np.empty(n, dtype=np.int64), 0
    t0 = time.time()
    finfo = np.finfo(dtype) if np.dtype(dtype).kind == "f" else None
    for i, (imgs, lab) in enumerate(loader):
        x = normalize_for(adapter, imgs.to(device, non_blocking=True))
        with ctx:
            f = probe_feature(adapter, x, m)
        f = f.float().cpu().numpy()
        if finfo is not None and np.abs(f).max() > finfo.max:
            raise OverflowError(f"feature values exceed the {np.dtype(dtype).name} range; rerun with "
                                "--feature-dtype float32")
        if X is None:
            X = np.empty((n, f.shape[1]), dtype=dtype)
        b = f.shape[0]
        X[pos:pos + b] = f
        y[pos:pos + b] = np.asarray(lab, dtype=np.int64)
        pos += b
        if log_every and (i + 1) % log_every == 0:
            dt = time.time() - t0
            log(f"  {tag} {pos}/{n} images, {pos / max(dt, 1e-9):.0f} img/s")
    if pos != n:
        raise RuntimeError(f"loader gave {pos} images, dataset has {n}")
    if not np.isfinite(X).all():
        raise FloatingPointError("non-finite features")
    return X, y


def feature_path(out: Path, model_name: str, variant: str, split: str) -> Path:
    return Path(out) / "features" / model_name / f"{variant}_{split}.npz"


def save_features(path: Path, X: np.ndarray, y: np.ndarray, rows: np.ndarray, meta: dict) -> None:
    """Write atomically (a partial file never looks finished)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez(fh, X=X, y=y, rows=np.asarray(rows, dtype=np.int64), meta_json=np.asarray(json.dumps(_jsonable(meta))))
    os.replace(tmp, path)


def cached_features(path: Path, expect: dict):
    """(X, y, meta) if `path` holds features made with the same settings (`expect` keys), else None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta_json"]))
            for key, val in expect.items():
                if meta.get(key) != _jsonable(val):
                    log(f"cached {path.name}: {key} is {meta.get(key)!r}, this run wants {_jsonable(val)!r}; recomputing")
                    return None
            X, y = z["X"], z["y"]
    except Exception as exc:  # unreadable or truncated file: recompute
        log(f"cached {path.name} unreadable ({exc}); recomputing")
        return None
    if X.shape[0] != int(expect.get("n", X.shape[0])) or X.shape[0] != y.shape[0]:
        return None
    return X, y, meta


# ----------------------------------------------------------------------------------------------
# Linear probe
# ----------------------------------------------------------------------------------------------
def stratified_holdout(labels, frac: float = 0.1, seed: int = 0) -> tuple:
    """Fixed class-stratified split of positions 0..N-1: about `frac` of every class held out (at
    least one per class when the class has two or more images). Returns sorted (train, holdout)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    hold = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        n_hold = min(max(int(round(frac * len(idx))), 1), len(idx) - 1)
        hold.append(rng.permutation(idx)[:n_hold])
    hold = np.sort(np.concatenate(hold)) if hold else np.zeros(0, dtype=np.int64)
    train = np.setdiff1d(np.arange(len(labels)), hold)
    return train.astype(np.int64), hold.astype(np.int64)


def standardiser(X: torch.Tensor) -> tuple:
    """Per-feature mean and standard deviation (float64 sums); constant features get std 1."""
    Xd = X.to(torch.float64)
    mean = Xd.mean(dim=0)
    std = Xd.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return mean.to(torch.float32), std.to(torch.float32)


def standardise(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X.to(torch.float32) - mean) / std


def fit_logreg(X: torch.Tensor, y: torch.Tensor, num_classes: int, wd: float, max_iter: int = 1000,
               history_size: int = 20, tol_grad: float = 1e-6, tol_change: float = 1e-9, init=None) -> dict:
    """Multinomial logistic regression, mean cross-entropy + wd / 2 * ||W||^2 (bias not penalised),
    full-batch L-BFGS with a strong Wolfe line search. `init` = (W, b) warm start, else zeros.
    The problem is strictly convex for wd > 0, so a converged solution does not depend on `init`."""
    D = X.shape[1]
    kw = dict(device=X.device, dtype=torch.float32)
    W = (torch.zeros(D, num_classes, **kw) if init is None else init[0].detach().to(**kw).clone()).requires_grad_(True)
    b = (torch.zeros(num_classes, **kw) if init is None else init[1].detach().to(**kw).clone()).requires_grad_(True)
    max_eval = int(max_iter * 1.25)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=int(max_iter), max_eval=max_eval, tolerance_grad=tol_grad,
                            tolerance_change=tol_change, history_size=int(history_size),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=False)
        loss = F.cross_entropy(torch.addmm(b, X, W), y) + 0.5 * float(wd) * (W * W).sum()
        loss.backward()
        return loss

    t0 = time.time()
    with torch.enable_grad():
        opt.step(closure)
        loss = float(closure().detach())
    st = opt.state[opt._params[0]]
    n_iter, n_eval = int(st.get("n_iter", 0)), int(st.get("func_evals", 0))
    gmax = float(max(W.grad.abs().max(), b.grad.abs().max()))
    if X.is_cuda:
        torch.cuda.synchronize(X.device)
    return {"W": W.detach(), "b": b.detach(), "loss": loss, "grad_max": gmax, "n_iter": n_iter, "n_eval": n_eval,
            "hit_limit": bool(n_iter >= max_iter or n_eval >= max_eval), "seconds": time.time() - t0}


@torch.no_grad()
def evaluate_probe(X: torch.Tensor, y: torch.Tensor, W: torch.Tensor, b: torch.Tensor, num_classes: int,
                   chunk: int = 16384) -> dict:
    """Top-1, top-5 (%) and mean per-class top-1 (%) of the probe on (X, y); X already standardised."""
    k5 = min(5, num_classes)
    c1, c5 = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    per_hit = torch.zeros(num_classes, dtype=torch.float64)
    per_n = torch.zeros(num_classes, dtype=torch.float64)
    loss = 0.0
    for s in range(0, X.shape[0], chunk):
        logits = torch.addmm(b, X[s:s + chunk], W)
        yy = y[s:s + chunk]
        loss += float(F.cross_entropy(logits, yy, reduction="sum"))
        top = logits.topk(k5, dim=1).indices
        hit1 = (top[:, 0] == yy)
        c1 += hit1.sum().double().cpu()
        c5 += (top == yy[:, None]).any(dim=1).sum().double().cpu()
        per_hit += torch.bincount(yy[hit1].cpu(), minlength=num_classes).double()
        per_n += torch.bincount(yy.cpu(), minlength=num_classes).double()
    n = X.shape[0]
    present = per_n > 0
    return {"top1": 100.0 * float(c1) / n, "top5": 100.0 * float(c5) / n,
            "mean_per_class_top1": 100.0 * float((per_hit[present] / per_n[present]).mean()),
            "loss": loss / n, "n": int(n)}


def wd_candidates(best: float, tried) -> list:
    """Half-decade neighbours of `best` not tried yet (refinement step)."""
    out = []
    for f in (10 ** 0.5, 10 ** -0.5):
        w = float(f"{best * f:.3g}")
        if all(abs(math.log10(w) - math.log10(t)) > 1e-3 for t in tried):
            out.append(w)
    return out


def select_weight_decay(Xb, yb, num_classes: int, grid, holdout_frac: float = 0.1, seed: int = 0,
                        refine: bool = True, device="cpu", solver: dict | None = None) -> dict:
    """Choose wd on a fixed stratified split of the TRAINING features only (the bank). Never sees the
    test set: it takes no test arguments. Returns the trials, the choice and the split positions."""
    solver = dict(solver or {})
    device = torch.device(device)
    tr, ho = stratified_holdout(np.asarray(yb), holdout_frac, seed)
    if len(ho) == 0:
        raise ValueError("the hold-out split is empty (every class needs at least two training images)")
    Xt = torch.as_tensor(Xb[tr]).to(device)
    mean, std = standardiser(Xt)
    Xt = standardise(Xt, mean, std)
    Xh = standardise(torch.as_tensor(Xb[ho]).to(device), mean, std)
    yt = torch.as_tensor(np.asarray(yb)[tr], dtype=torch.long, device=device)
    yh = torch.as_tensor(np.asarray(yb)[ho], dtype=torch.long, device=device)
    trials, sols = [], {}

    def run(wd, init, stage):
        r = fit_logreg(Xt, yt, num_classes, wd, init=init, **solver)
        ev = evaluate_probe(Xh, yh, r["W"], r["b"], num_classes)
        sols[wd] = (r["W"], r["b"])
        trials.append({"wd": wd, "stage": stage, "holdout_top1": ev["top1"], "holdout_top5": ev["top5"],
                       "holdout_loss": ev["loss"], "train_objective": r["loss"], "grad_max": r["grad_max"],
                       "n_iter": r["n_iter"], "n_eval": r["n_eval"], "hit_limit": r["hit_limit"],
                       "seconds": round(r["seconds"], 2)})
        log(f"    wd {wd:.3g}: hold-out top-1 {ev['top1']:.2f}, {r['n_iter']} iterations"
            f"{' (iteration limit)' if r['hit_limit'] else ''}, {r['seconds']:.1f} s")

    def best_trial():
        # Highest hold-out top-1; ties go to the larger weight decay (more regularised).
        return max(trials, key=lambda t: (round(t["holdout_top1"], 9), t["wd"]))

    init = None
    for wd in sorted({float(w) for w in grid}, reverse=True):   # largest first, warm-started along the path
        run(wd, init, "grid")
        init = sols[wd]
    if refine:
        best = best_trial()["wd"]
        for wd in wd_candidates(best, [t["wd"] for t in trials]):
            run(wd, sols[best], "refine")
    best = best_trial()
    edge = best["wd"] in (min(t["wd"] for t in trials), max(t["wd"] for t in trials))
    out = {"chosen_wd": best["wd"], "holdout_top1_at_chosen": best["holdout_top1"], "wd_at_grid_edge": bool(edge),
           "trials": trials, "n_train": int(len(tr)), "n_holdout": int(len(ho)), "holdout_frac": holdout_frac,
           "seed": seed, "train_pos": tr, "holdout_pos": ho, "init": sols[best["wd"]]}
    del Xt, Xh
    return out


def run_probe(Xb, yb, Xv, yv, num_classes: int, grid, holdout_frac: float = 0.1, seed: int = 0,
              refine: bool = True, device="cpu", solver: dict | None = None) -> dict:
    """Weight decay chosen on the bank split, refit on the whole bank, scored on (Xv, yv)."""
    device = torch.device(device)
    t0 = time.time()
    sel = select_weight_decay(Xb, yb, num_classes, grid, holdout_frac, seed, refine, device, solver)
    t_sel = time.time() - t0
    Xt = torch.as_tensor(Xb).to(device)
    mean, std = standardiser(Xt)
    Xt = standardise(Xt, mean, std)
    yt = torch.as_tensor(np.asarray(yb), dtype=torch.long, device=device)
    fin = fit_logreg(Xt, yt, num_classes, sel["chosen_wd"], init=sel["init"], **dict(solver or {}))
    train_ev = evaluate_probe(Xt, yt, fin["W"], fin["b"], num_classes)
    del Xt
    Xe = standardise(torch.as_tensor(Xv).to(device), mean, std)
    ye = torch.as_tensor(np.asarray(yv), dtype=torch.long, device=device)
    test_ev = evaluate_probe(Xe, ye, fin["W"], fin["b"], num_classes)
    del Xe
    return {"selection": {k: v for k, v in sel.items() if k not in ("train_pos", "holdout_pos", "init")},
            "holdout_positions": sel["holdout_pos"],
            "final": {"wd": sel["chosen_wd"], "train_objective": fin["loss"], "grad_max": fin["grad_max"],
                      "n_iter": fin["n_iter"], "n_eval": fin["n_eval"], "hit_limit": fin["hit_limit"],
                      "seconds": round(fin["seconds"], 2), "train_top1": train_ev["top1"]},
            "test": test_ev, "seconds_selection": round(t_sel, 2), "seconds_total": round(time.time() - t0, 2),
            "W": fin["W"].cpu().numpy(), "b": fin["b"].cpu().numpy(), "mean": mean.cpu().numpy(),
            "std": std.cpu().numpy()}


@torch.no_grad()
def knn_check(Xb, yb, Xv, yv, d_first: int, num_classes: int, device, k: int = 20, T: float = 0.07) -> float:
    """k-NN top-1 (%) on the first `d_first` feature columns (CLS or pooled), cosine, k = 20, T = 0.07."""
    from mirror.knn import knn_predict

    device = torch.device(device)
    bf = F.normalize(torch.as_tensor(Xb[:, :d_first]).to(device).float(), dim=-1)
    qf = F.normalize(torch.as_tensor(Xv[:, :d_first]).to(device).float(), dim=-1)
    by = torch.as_tensor(np.asarray(yb), dtype=torch.long, device=device)
    qy = torch.as_tensor(np.asarray(yv), dtype=torch.long, device=device)
    pred = knn_predict(bf, by, qf, num_classes, k, T)
    return 100.0 * float((pred == qy).float().mean())


# ----------------------------------------------------------------------------------------------
# Model and data
# ----------------------------------------------------------------------------------------------
def build_adapter(model_name: str, tiny: bool, device, img_size: int = 224):
    if tiny:
        from mirror.adapters import TimmTinyAdapter

        adapter = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    else:
        from mirror.adapters_timm import build_model_adapter

        adapter = build_model_adapter(model_name, img_size=img_size)
    adapter.to(device=device)
    adapter.model.float().eval()
    return adapter


class TinyData:
    """Synthetic stand-in for the Protocol (CPU tests): a labelled bank and a disjoint test set of
    random 56 px images with a weak class tint (mirror.data.synthetic_dataset)."""

    def __init__(self, num_classes: int = 4, bank_per_class: int = 20, val_per_class: int = 8, seed: int = 0,
                 batch_size: int = 16):
        images, labels = synthetic_dataset(num_classes, bank_per_class + val_per_class, 56, seed=seed)
        sp = make_splits(labels, {"bank": bank_per_class, "val": val_per_class})
        self.images, self.labels, self.sp, self.bs = images, labels, sp, batch_size
        self.num_classes = num_classes

    def loader(self, split: str):
        return make_loader(self.images, self.labels, self.sp[split], self.bs, 0, False)

    def rows(self, split: str) -> np.ndarray:
        return self.sp[split]


class ProtocolData:
    """The paper protocol: bank = 100k train images (Protocol.bank_loader), test = 50k validation."""

    def __init__(self, cache_dir, batch_size: int, num_workers: int, pin: bool):
        from mirror.protocol import Protocol

        self.proto = Protocol(cache_dir, batch_size=batch_size, num_workers=num_workers, pin=pin)
        self.num_classes = 1000

    def loader(self, split: str):
        return self.proto.bank_loader() if split == "bank" else self.proto.query_loader()

    def rows(self, split: str) -> np.ndarray:
        return (np.asarray(self.proto.index["bank_rows"]) if split == "bank"
                else np.arange(len(self.proto.val_labels)))


# ----------------------------------------------------------------------------------------------
# Variant factory (reuses mirror.variants, scripts/m1_attention.py and scripts/heal.py)
# ----------------------------------------------------------------------------------------------
class VariantFactory:
    def __init__(self, adapter, model_name: str, tiny: bool, operators=None, m1_structure=None, heal_dir=None,
                 heal_config=None, random_seed: int = 1, vexact_dir=None, vexact_checkpoint=None,
                 vexact_config=None):
        self.adapter, self.model_name, self.tiny = adapter, model_name, tiny
        self.operators = _resolve(operators) if operators else None
        self.m1_structure_path = _resolve(m1_structure) if m1_structure else None
        self.heal_dir = _resolve(heal_dir) if heal_dir else None
        self.heal_config = _resolve(heal_config) if heal_config else REPO_ROOT / "configs" / "m3_pilot.yaml"
        self.random_seed = int(random_seed)
        self.vexact_dir = _resolve(vexact_dir) if vexact_dir else None
        self.vexact_checkpoint = _resolve(vexact_checkpoint) if vexact_checkpoint else None
        self.vexact_config = _resolve(vexact_config or DEFAULT_VEXACT_CONFIG)
        self._struct = self._m1 = self._heal = self._heal_vexact = None
        self._m1_S = None
        # Check images for healed V-exact variants (normalised [B, 3, S, S] tensors at the adapter's size and
        # dtype) and how to describe them. None: scripts/heal_vexact.py's own check images (224 px ImageNet).
        self.check_xs = None
        self.check_label = None

    # ----- provenance (cheap; used to validate cached features) -------------------------------
    def _need(self, path, what: str, flag: str) -> Path:
        if path is None:
            raise FileNotFoundError(f"{what} is needed for this variant: pass {flag}")
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found ({what}; pass {flag})")
        return Path(path)

    def heal_run(self, v: VariantName) -> Path:
        return self._need(self.heal_dir, "the M3 healing directory", "--heal-dir") / v.method

    def vexact_run(self, v: VariantName) -> tuple:
        """(run directory, checkpoint) of a healed V-exact variant, in the scripts/heal_vexact.py layout
        <vexact-dir>/k<K>/final/trained_blocks.pt; --vexact-checkpoint names the file directly (its run
        directory is then the folder above final/). Fails with a clear message if the file is missing."""
        if self.vexact_checkpoint is not None:
            ck = self.vexact_checkpoint
            run = ck.parent.parent if ck.parent.name == "final" else ck.parent
        else:
            base = self._need(self.vexact_dir, "the V-exact healing directory (out_dir of scripts/heal_vexact.py)",
                              "--vexact-dir")
            run = base / f"k{v.k}"
            ck = run / "final" / "trained_blocks.pt"
        if not ck.is_file():
            raise FileNotFoundError(
                f"{v.name}: healed V-exact checkpoint {ck} not found. Heal it first (python scripts/heal_vexact.py "
                f"--config configs/m3_vexact.yaml --k {v.k}) or pass --vexact-dir (or --vexact-checkpoint) for a run "
                "that has final/trained_blocks.pt")
        return run, ck

    def sources(self, v: VariantName) -> dict:
        """Files the variant is built from, with their sha256 (stored with the features)."""
        if v.family == "original":
            return {}
        if v.family == "healed_vexact":
            run, ck = self.vexact_run(v)
            return {"vexact_run": str(run), "checkpoint": str(ck), "checkpoint_sha256": sha256(ck)}
        if v.family == "mlp":
            p = self._need(self.operators, "the X2 operator file (e1_operators.npz)", "--operators")
            return {"operators": str(p), "operators_sha256": sha256(p)}
        if v.family == "m1":
            p = self._need(self.operators, "the X2 operator file (e1_operators.npz)", "--operators")
            s = self._need(self.m1_structure_path, "the M1 structure (m1_structure.npz)", "--m1-structure")
            return {"operators": str(p), "operators_sha256": sha256(p), "m1_structure": str(s),
                    "m1_structure_sha256": sha256(s)}
        run = self.heal_run(v)
        ck = run / "final" / "trained_blocks.pt"
        if not ck.exists():
            raise FileNotFoundError(f"{ck} not found (run scripts/heal.py --method {v.method} first)")
        return {"heal_run": str(run), "checkpoint": str(ck), "checkpoint_sha256": sha256(ck)}

    # ----- lazy loaders ------------------------------------------------------------------------
    def structure(self):
        if self._struct is None:
            from mirror import variants as V

            p = self._need(self.operators, "the X2 operator file (e1_operators.npz)", "--operators")
            self._struct = V.load_fit_run(p)
            log(f"structure from {p}: depth {self._struct.depth}, d {self._struct.d}, h {self._struct.h}")
        return self._struct

    def m1(self):
        if self._m1 is None:
            self._m1 = load_script("m1_attention")
        return self._m1

    def heal(self):
        if self._heal is None:
            self._heal = load_script("heal")
        return self._heal

    def heal_vexact(self):
        """scripts/heal_vexact.py as a module (its structure loader and check data are reused as they are)."""
        if self._heal_vexact is None:
            self._heal_vexact = load_script("heal_vexact")
        return self._heal_vexact

    def m1_structure(self, need_k=None):
        """(S, ops) as scripts/m1_attention.py uses them, from the saved m1_structure.npz (its own
        checks: model name and operator sha256), plus the matched random structure."""
        m1 = self.m1()
        if self._m1_S is None:
            ad = self.adapter
            p = self._need(self.operators, "the X2 operator file (e1_operators.npz)", "--operators")
            s = self._need(self.m1_structure_path, "the M1 structure (m1_structure.npz)", "--m1-structure")
            ops = m1.load_operators(p, ad.depth, ad.embed_dim, ad.mlp_hidden)
            with np.load(s) as z:
                saved = {"fit_seed": int(z["fit_seed"]), "random_seed": int(z["random_seed"]),
                         "register_margin": float(z["register_margin"]), "ks": [int(k) for k in z["ks"]]}
            a = SimpleNamespace(model="tiny" if self.tiny else self.model_name, tiny=self.tiny,
                                fit_seed=saved["fit_seed"], random_seed=saved["random_seed"],
                                register_margin=saved["register_margin"])
            S = m1._load_structure(s, a, ops, [])
            if S is None:
                raise RuntimeError(f"could not load {s}")
            S["random"] = m1.matched_random_structure(ops["A_layers"], ops["perms"], saved["random_seed"],
                                                      fix_ones=True, heads=S["maps"])
            self._m1_S = (S, ops, saved)
        S, ops, saved = self._m1_S
        if need_k is not None and int(need_k) not in S["A_star"]:
            raise ValueError(f"{self.m1_structure_path} has V-exact operators for k in {sorted(S['A_star'])}, not "
                             f"{need_k}: rerun scripts/m1_attention.py with --ks including {need_k}")
        return S, ops, saved

    # ----- build -------------------------------------------------------------------------------
    def build(self, v: VariantName):
        """(model, info). The original returns the adapter's own model (not a copy)."""
        ad = self.adapter
        L = ad.depth
        if v.family != "original" and not 1 <= v.k <= L:
            raise ValueError(f"{v.name}: k = {v.k} outside 1..{L}")
        d, h = ad.embed_dim, ad.mlp_hidden
        from mirror.attention import attn_dims

        H, dh = attn_dims(ad.model.blocks[0])
        D = H * dh
        mlp_dense, att_dense = 2 * h * d * v.k, 4 * D * d * v.k
        n_model = int(sum(p.numel() for p in ad.model.parameters()))
        info = {"variant": v.name, "label": v.label, "family": v.family, "method": v.method, "k": v.k,
                "blocks": list(range(v.k)), "model_params_original": n_model}
        if v.family == "original":
            info.update({"converted_params_dense": 0, "converted_params_kept": 0, "model_params": n_model})
            return ad.model, info
        if v.family == "mlp":
            from mirror import variants as V

            struct = self.structure()
            which = {"mirror": "per_layer", "svd": "svd", "random": "random_structure"}[v.method]
            spec = V.build_variant("A1", which, struct, v.k, seed=self.random_seed)
            m = V.variant_copy(ad.model, spec)
            kept = int(spec.params_kept) + att_dense
            info.update({"builder": f"mirror.variants.build_variant('A1', '{which}', k={v.k}, seed={self.random_seed})",
                         "spec_signature": V.spec_signature(spec), "mlp_params_kept": int(spec.params_kept),
                         "mlp_params_dense": int(spec.params_dense), "attn_params_kept": att_dense,
                         "random_seed": self.random_seed if v.method == "random" else None,
                         "fit_run_meta": struct.meta, **({"svd": spec.info} if v.method == "svd" else {})})
        elif v.family == "m1":
            S, ops, saved = self.m1_structure(v.k if v.method == "vexact" else None)
            m1 = self.m1()
            m = m1.make_variant(v.method, v.k, ad, S, ops)
            p = m1.variant_params(v.method, v.k, ad, S, ops, n_model)
            kept = int(p["converted_params_kept"])
            info.update({"builder": f"scripts/m1_attention.make_variant('{v.method}', k={v.k})",
                         "mlp_params_kept": p["mlp_params_kept"], "attn_params_kept": p["attn_params_kept"],
                         "a_out_offset": S["a_out_offset"], "m1_saved": saved,
                         "even_head": bool(v.method == "vexact" and v.k == L)})
        elif v.family == "healed_vexact":
            m, extra = self._build_healed_vexact(v)
            info.update(extra)
            kept = int(extra["mlp_params_kept"]) + int(extra["attn_params_kept"])
        else:
            m, extra = self._build_healed(v)
            info.update(extra)
            kept = int(extra.get("mlp_params_kept", mlp_dense)) + int(extra.get("attn_params_kept", att_dense))
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        dense = mlp_dense + att_dense
        info.update({"converted_params_dense": dense, "converted_params_kept": kept,
                     "model_params": n_model - dense + kept})
        return m, info

    def _build_healed(self, v: VariantName):
        hm = self.heal()
        from mirror.heal import load_trained_blocks
        from mirror.utils import load_config

        run = self.heal_run(v)
        ck = run / "final" / "trained_blocks.pt"
        summ = run / "summary.json"
        if summ.exists():
            cfg = json.loads(summ.read_text(encoding="utf-8"))["config"]
            cfg_src = str(summ)
        else:
            cfg = load_config(self.heal_config, tiny=self.tiny)
            cfg_src = str(self.heal_config)
            log(f"WARNING: {summ} missing; rebuilding the student from {self.heal_config}")
        cfg = dict(cfg)
        cfg["method"] = v.method
        blocks = [int(b) for b in cfg["blocks"]]
        if blocks != list(range(v.k)):
            raise ValueError(f"{v.name}: the healed run {run} converted blocks {blocks}, not 0..{v.k - 1}")
        structure = hm.load_structure(cfg, self.adapter)
        make_student, minfo = hm.method_setup(v.method, self.adapter, structure, blocks, cfg)
        m, cons = make_student()
        for c in cons:
            prep = getattr(c, "prepare", None)
            if callable(prep):
                prep(m)                       # LowRank* install their factorised layers here
        ck_meta = load_trained_blocks(m, ck)
        checks = constraint_checks(m, cons)
        log(f"  healed {v.method}: loaded {ck}; constraint checks {json.dumps(_jsonable(checks))[:300]}")
        extra = {"builder": "scripts/heal.method_setup + mirror.heal.load_trained_blocks", "heal_config_source": cfg_src,
                 "heal_seed": cfg.get("seed"), "method_info": minfo, "checkpoint_meta": ck_meta,
                 "constraint_checks": checks}
        if "mlp_params_kept" in minfo:
            extra["mlp_params_kept"] = int(minfo["mlp_params_kept"])
        if minfo.get("attention") and "attn_params_kept" in minfo:
            extra["attn_params_kept"] = int(minfo["attn_params_kept"])
        return m, extra

    # ----- healed V-exact (scripts/heal_vexact.py) ----------------------------------------------
    def _vexact_config(self, v: VariantName, run: Path) -> tuple:
        """(config, summary or None, config source): the run's own config from <run>/summary.json, as
        the existing healed variants read theirs; configs/m3_vexact.yaml (--vexact-config) otherwise."""
        from mirror.utils import load_config

        summ_path = run / "summary.json"
        if summ_path.exists():
            summ = json.loads(summ_path.read_text(encoding="utf-8"))
            cfg, src = dict(summ["config"]), str(summ_path)
            if int(summ.get("k", v.k)) != v.k:
                raise ValueError(f"{v.name}: {summ_path} is a V-exact run with k = {summ['k']}, not {v.k}")
        else:
            summ = None
            cfg, src = load_config(self.vexact_config, tiny=self.tiny), str(self.vexact_config)
            log(f"WARNING: {summ_path} missing; rebuilding the V-exact student from {self.vexact_config}")
        cfg["k"], cfg["method"] = v.k, "vexact"
        if bool(cfg.get("tiny_mode")) != bool(self.tiny):
            raise ValueError(f"{v.name}: {src} is {'a tiny' if cfg.get('tiny_mode') else 'not a tiny'} run; this run is "
                             f"{'tiny' if self.tiny else 'not tiny'}")
        if not self.tiny and cfg["model"]["name"] != self.model_name:
            raise ValueError(f"{v.name}: {src} healed {cfg['model']['name']}, not {self.model_name}")
        return cfg, summ, src

    def _build_healed_vexact(self, v: VariantName):
        """The healed V-exact student, built exactly as scripts/heal_vexact.py builds it (make_student:
        load_vexact_structure, then build_vexact_student with the run's train_embeddings), the
        constraints prepared as mirror.heal.heal does, then the checkpoint loaded with
        mirror.heal.load_trained_blocks (the reload path heal_vexact.py documents). Then checked."""
        hv = self.heal_vexact()
        from mirror.heal import load_trained_blocks
        from mirror.heal_vexact import build_vexact_student, vexact_param_counts

        ad, L = self.adapter, int(self.adapter.depth)
        run, ck = self.vexact_run(v)
        cfg, summ, cfg_src = self._vexact_config(v, run)
        st = hv.load_vexact_structure(cfg, ad, v.k)
        rec_sha = ((summ or {}).get("structure") or {}).get("span_sha256")
        if rec_sha and st["span_sha256"] != rec_sha:
            raise ValueError(f"{v.name}: {st['span_source']} (sha256 {str(st['span_sha256'])[:12]}) is not the span "
                             f"operator file the run was healed with (sha256 {rec_sha[:12]}, {run / 'summary.json'})")
        A, reg_perm = st["A_star"], st["reg_perm"]
        train_emb = bool((cfg.get("vexact") or {}).get("train_embeddings", True))
        heal_px = healing_px(cfg)
        same_px = heal_px == int(ad.img_size)
        # The healed student at its healing resolution: built as scripts/heal_vexact.py builds it, then loaded.
        ad_h = ad if same_px else adapter_at(ad, heal_px)
        m, cons, _ = build_vexact_student(ad_h, A, v.k, st["perms"], st["heads"], reg_perm, train_embeddings=train_emb)
        for c in cons:
            prep = getattr(c, "prepare", None)
            if callable(prep):
                prep(m)
        ck_meta = load_trained_blocks(m, ck)
        m.eval()
        loaded = checkpoint_matches(m, ck, list(range(v.k)))
        grid = None
        if not same_px:
            m, cons, grid = self._regrid_healed_vexact(v, m, ck, A, reg_perm, st, train_emb, heal_px)
        checks = constraint_checks(m, cons)
        vchecks = self._vexact_checks(v, m, cfg, A, reg_perm, run)
        pc = vexact_param_counts(ad, A, v.k, st["perms"], st["heads"])
        knn = (summ or {}).get("knn") or {}
        on_cuda = str(((summ or {}).get("env") or {}).get("device", "")).startswith("cuda")
        extra = {"builder": "scripts/heal_vexact.load_vexact_structure + mirror.heal_vexact.build_vexact_student "
                            "(as scripts/heal_vexact.py make_student) + mirror.heal.load_trained_blocks",
                 "vexact_run": str(run), "checkpoint": str(ck), "heal_config_source": cfg_src, "heal_seed": cfg.get("seed"),
                 "structure": {kk: st[kk] for kk in ("span_source", "span_sha256", "operators_source", "heads_source")},
                 "reg_perm": [int(x) for x in reg_perm], "d_plus": pc["d_plus"], "d_minus": pc["d_minus"],
                 "train_embeddings": train_emb, "even_head": bool(v.k == L), "feature": vexact_feature_note(v.k, L),
                 "checkpoint_meta": ck_meta, "checkpoint_check": loaded, "constraint_checks": checks,
                 "vexact_checks": vchecks, "param_counts": pc, "mlp_params_kept": int(pc["mlp_params_kept"]),
                 "attn_params_kept": int(pc["attn_params_kept"]),
                 "recorded_knn_healed_top1": (knn.get("healed") or {}).get("top1"),
                 # as scripts/heal_vexact.py's SUMMARY.md: the bf16 autocast applies on CUDA only
                 "recorded_knn_precision": "bf16" if (knn.get("amp") == "bf16" and on_cuda) else "fp32",
                 "healed_px": heal_px, "built_px": int(ad.img_size), "position_grid": grid}
        return m, extra

    @torch.no_grad()
    def _regrid_healed_vexact(self, v: VariantName, healed, ck, A, reg_perm, st, train_emb: bool, heal_px: int):
        """The healed V-exact student `healed` (built at `heal_px` and loaded) moved to the adapter's input size.

        The student is built at the adapter's size exactly as scripts/heal_vexact.py builds it (same structure,
        register pairing and train_embeddings) and receives every tensor of `healed` except the position
        embedding. The healed position embedding (CLS row plus the healed grid) is interpolated to the new grid
        with the model's own interpolate_pos_encoding, the call mirror.m1.materialise_pos_embed makes, and then
        re-projected with A* by mirror.m1.project_embeddings, the projection build_vexact applies to a
        materialised grid; only the position embedding of that projection is kept. This is a deviation from the
        healed model (the healed grid was never trained at this size). Returns (model, constraints, record)."""
        from mirror.heal_vexact import build_vexact_student, span_exactness
        from mirror.m1 import materialise_pos_embed, project_embeddings

        ad = self.adapter
        if not (hasattr(ad.model, "interpolate_pos_encoding") and hasattr(ad.model, "prepare_tokens_with_masks")):
            raise NotImplementedError(f"{v.name}: healed at {heal_px} px, evaluated at {ad.img_size} px, but the model "
                                      "has a fixed position grid; only DINOv2-style models can be moved")
        if self.check_xs is None:
            raise ValueError(f"{v.name}: healed at {heal_px} px, evaluated at {ad.img_size} px: the exactness checks "
                             f"need check images at {ad.img_size} px (set VariantFactory.check_xs)")
        _check_image_sizes(v, self.check_xs, ad)
        m, cons, _ = build_vexact_student(ad, A, v.k, st["perms"], st["heads"], reg_perm, train_embeddings=train_emb)
        for c in cons:
            prep = getattr(c, "prepare", None)
            if callable(prep):
                prep(m)
        src, own = healed.state_dict(), m.state_dict()
        if set(src) != set(own):
            raise RuntimeError(f"{v.name}: the student built at {ad.img_size} px has other tensors than the healed one: "
                               f"{sorted(set(src) ^ set(own))[:5]}")
        keep = {k: t for k, t in src.items() if k != "pos_embed"}
        bad = [k for k, t in keep.items() if t.shape != own[k].shape]
        if bad:
            raise RuntimeError(f"{v.name}: shapes differ between {heal_px} and {ad.img_size} px builds: {bad[:5]}")
        m.load_state_dict(keep, strict=False)
        m.eval()
        # Interpolate the healed position embedding with the model's own rule, on a scratch copy.
        work = copy.deepcopy(m)
        work.pos_embed = torch.nn.Parameter(healed.pos_embed.detach().clone(), requires_grad=False)
        if not materialise_pos_embed(work, ad):
            raise RuntimeError(f"{v.name}: the healed position embedding was not interpolated")
        interp = work.pos_embed.detach().clone()
        before = span_exactness(ad, work, [x.to(interp.device) for x in self.check_xs], A, v.k, reg_perm)
        project_embeddings(work, ad, A, reg_perm, write=True)      # only work.pos_embed is used below
        proj = work.pos_embed.detach().clone()
        del work
        if proj.shape != m.pos_embed.shape:
            raise RuntimeError(f"{v.name}: interpolated grid {tuple(proj.shape)} does not fit {tuple(m.pos_embed.shape)}")
        m.pos_embed.copy_(proj)
        # The model now equals the healed student on every tensor except the position embedding.
        after = m.state_dict()
        differ = [k for k, t in keep.items() if not torch.equal(after[k], t)]
        if differ:
            raise RuntimeError(f"{v.name}: {len(differ)} tensors differ from the healed student, e.g. {differ[:3]}")
        ck_sd = torch.load(ck, map_location="cpu", weights_only=True)["state_dict"]
        ck_differ = [k for k, t in ck_sd.items() if k != "pos_embed" and not torch.equal(after[k].detach().cpu(), t)]
        if ck_differ:
            raise RuntimeError(f"{v.name}: {len(ck_differ)} checkpoint tensors differ after the move, e.g. {ck_differ[:3]}")
        n_pre = int(interp.shape[1]) - ad.grid_hw[0] * ad.grid_hw[1]

        def rel(a, b):
            b64 = b.double()
            den = float(b64.norm())
            return float((a.double() - b64).norm()) / den if den > 0 else float("nan")

        H0 = int(round(math.sqrt(int(healed.pos_embed.shape[1]) - n_pre)))
        off, aa = getattr(ad.model, "interpolate_offset", None), getattr(ad.model, "interpolate_antialias", None)
        rule = (f"bicubic, scale factor (w0 + {off:g}) / M" if off else "bicubic to the target size") + \
               (", antialias" if aa else ", no antialias")
        record = {
            "deviation": (f"healed {H0} x {H0} position grid interpolated to {ad.grid_hw[0]} x {ad.grid_hw[1]} and "
                          "re-projected with A*; every other tensor as healed"),
            "healed_px": int(heal_px), "eval_px": int(ad.img_size), "healed_grid": [H0, H0], "eval_grid": list(ad.grid_hw),
            "interpolation": "model.interpolate_pos_encoding (as mirror.m1.materialise_pos_embed): " + rule,
            "interpolate_offset": off, "interpolate_antialias": aa,
            "projection": "mirror.m1.project_embeddings (as mirror.m1.build_vexact); position embedding only",
            "reprojection_rel_change": rel(proj[0, n_pre:], interp[0, n_pre:]),
            "reprojection_rel_change_with_cls": rel(proj, interp),
            "span_exact_without_reprojection": before["max_rel_inside"],
            "span_exact_without_reprojection_pass": bool(before["pass"]),
            "tensors_copied": len(keep), "checkpoint_tensors_equal_except_pos_embed": len(ck_sd) - int("pos_embed" in ck_sd),
        }
        log(f"  {v.name}: healed at {heal_px} px, moved to {ad.img_size} px: {record['deviation']}; interpolation "
            f"{rule}; re-projection changed the grid by {record['reprojection_rel_change']:.2e} (relative); span "
            f"exactness without the re-projection {before['max_rel_inside']:.2e}")
        return m, cons, record

    @torch.no_grad()
    def _vexact_checks(self, v: VariantName, model, cfg: dict, A, reg_perm, run: Path) -> dict:
        """Checks after loading, so a wrong load cannot pass silently, on scripts/heal_vexact.py's own
        check images (check_data "probe" and probe_batches: 64 ImageNet validation images on the paper
        protocol): its exactness probe (mirror.heal_vexact.span_exactness), the k-NN feature of each
        image against its mirror (mirror.m1.feature_invariance, as flip_check records it), and the same
        for the probe feature. Per batch max |f(mirror x) - f(x)| / max |f(x)|, then the largest over
        batches (flip_check's convention), so the values compare with exactness.json and flipcheck.json
        of the run. At full conversion the probe feature must agree within the exactness tolerance. Run
        in the model's dtype (fp32) without autocast. Raises RuntimeError on failure. When
        VariantFactory.check_xs is set (scripts/tier2_ade20k.py), those images are used instead; the
        images must be at the adapter's input size either way."""
        from mirror.data import mirror as mirror_images
        from mirror.heal_vexact import span_exactness
        from mirror.m1 import feature_invariance

        hv = self.heal_vexact()
        ad, L = self.adapter, int(self.adapter.depth)
        device = next(model.parameters()).device
        cc = cfg.get("checks") or {}
        tol = float(cc.get("exact_tol", 1e-4))
        if self.check_xs is not None:             # images at the adapter's size (scripts/tier2_ade20k.py)
            xs = [x.to(device) for x in self.check_xs]
            protocol = self.check_label or "given check images"
        else:
            cd = hv.check_data(cfg, device)
            xs = hv.probe_batches(ad, cd["probe"], int(cc.get("exact_images", 64)), device)
            protocol = cd["protocol"]
        _check_image_sizes(v, xs, ad)
        ex = span_exactness(ad, model, xs, A, v.k, reg_perm, tol=tol)
        knn_rel = max(feature_invariance(ad, model, x)["rel_max"] for x in xs)

        def rel(a, b):
            r = float(b.abs().max())
            return float(a.abs().max()) / r if r > 0 else float("nan")

        parts = []
        for x in xs:
            f = probe_feature(ad, x, model).double()
            ft = probe_feature(ad, mirror_images(x), model).double()
            s = f.shape[1] - int(ad.embed_dim)                   # [CLS (or pooled) | mean patch token]
            parts.append((rel(ft - f, f), rel(ft[:, :s] - f[:, :s], f[:, :s]), rel(ft[:, s:] - f[:, s:], f[:, s:])))
        flip = {"rel_max": max(p[0] for p in parts), "first_part_rel_max": max(p[1] for p in parts),
                "mean_patch_rel_max": max(p[2] for p in parts), "knn_cls_feature_rel_max": knn_rel,
                "n_images": int(sum(x.shape[0] for x in xs))}
        full = v.k == L
        recorded = {}
        for name, key in (("exactness.json", "max_rel_inside"), ("flipcheck.json", "cls_feature_rel_max")):
            p = run / name
            if p.exists():
                recorded[f"{name}:{key}"] = json.loads(p.read_text(encoding="utf-8")).get(key)
        ok = bool(ex["pass"]) and (not full or flip["rel_max"] < tol)
        out = {"pass": ok, "full_conversion": full, "tol": tol, "protocol": protocol, "n_images": ex["n_images"],
               "dtype": ex["dtype"], "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
               "span_exact_max_rel_inside": ex["max_rel_inside"], "span_exact_pass": bool(ex["pass"]),
               "span_exact_rows": ex["rows"], "probe_feature_flip": flip, "recorded_after_healing": recorded}
        log(f"  {v.name}: checks after loading ({ex['n_images']} {protocol} check images, {ex['dtype']}): span "
            f"exactness max rel {ex['max_rel_inside']:.2e} (tol {tol:g}); image vs mirror max rel: probe feature "
            f"{flip['rel_max']:.2e}, k-NN CLS feature {knn_rel:.2e}"
            f"{' (must be below tol at full conversion)' if full else ' (not invariant: k < depth)'}; "
            f"recorded after healing {json.dumps(_jsonable(recorded))}: {'PASS' if ok else 'FAIL'}")
        if not ex["pass"]:
            raise RuntimeError(f"{v.name}: the loaded student is not exactly mirror-equivariant over blocks 1..{v.k} "
                               f"(max relative error {ex['max_rel_inside']:.2e} >= {tol:g}); wrong checkpoint or "
                               f"structure for {run}. Not probing it.")
        if full and not flip["rel_max"] < tol:
            raise RuntimeError(f"{v.name}: at full conversion the probe feature of an image and its mirror differ by "
                               f"{flip['rel_max']:.2e} (relative, fp32) >= {tol:g}; the even head or the load is wrong. "
                               "Not probing it.")
        return out


def _check_image_sizes(v: VariantName, xs, adapter) -> None:
    """Healed V-exact check images must be at the adapter's input size."""
    sizes = sorted({tuple(int(s) for s in x.shape[-2:]) for x in xs})
    if sizes != [(int(adapter.img_size), int(adapter.img_size))]:
        raise ValueError(f"{v.name}: check images are {sizes} px but the model runs at {adapter.img_size} px; give "
                         "check images at that size (VariantFactory.check_xs)")


@torch.no_grad()
def _rank64(lin) -> int:
    from mirror.heal import LowRankLinear

    W = (lin.U.to(torch.float64) @ lin.V.to(torch.float64) if isinstance(lin, LowRankLinear)
         else lin.weight.to(torch.float64))
    return int(torch.linalg.matrix_rank(W))


def constraint_checks(model, cons) -> dict:
    """Each constraint's own check on the loaded student. For LowRankMLP the ranks are measured on U V
    formed in float64 from the factors (its own check forms U V in float32 first, and the rounding
    noise makes every matrix look full rank)."""
    out = {}
    for i, c in enumerate(cons):
        name = f"{i}:{type(c).__name__}"
        if type(c).__name__ == "LowRankMLP":
            out[name] = {"numerical_ranks": [[_rank64(getattr(model.blocks[b].mlp, n)) for n in ("fc1", "fc2")]
                                             for b in c.blocks],
                         "target_ranks": [list(r) for r in c.ranks],
                         "note": "ranks of U V formed in float64 from the factors"}
        elif callable(getattr(c, "check", None)):
            out[name] = c.check(model)
    return out


@torch.no_grad()
def checkpoint_matches(model, path, blocks) -> dict:
    """After mirror.heal.load_trained_blocks: the checkpoint converted `blocks`, and every tensor in it
    equals the model's state (a load that silently skipped or cast something fails here)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd, own = payload["state_dict"], model.state_dict()
    got = [int(b) for b in payload.get("blocks", [])]
    if got != [int(b) for b in blocks]:
        raise ValueError(f"{path} trained blocks {got}, expected {list(blocks)}")
    differ = [k for k, t in sd.items() if not torch.equal(own[k].detach().cpu(), t)]
    if differ:
        raise RuntimeError(f"{path}: {len(differ)} tensors differ from the model after loading, e.g. {differ[:3]}")
    return {"scope": payload.get("scope"), "blocks": got, "n_tensors": len(sd), "n_model_tensors": len(own),
            "model_tensors_not_in_checkpoint": sorted(set(own) - set(sd))[:20], "all_equal": True}


def vexact_feature_note(k: int, depth: int) -> str:
    """What the probe feature of a healed V-exact variant is (CSV notes)."""
    if int(k) == int(depth):
        return ("feature [CLS, mean patch token] after the student's own final norm (RMSNorm, even head y (I + A*)/2, "
                "affine), the forward of the k-NN feature of scripts/heal_vexact.py; exactly mirror-invariant in fp32 "
                "(bf16 forward passes round the mirrored computation differently)")
    return (f"feature [CLS, mean patch token] after the student's own final norm (RMSNorm, affine; no even head at "
            f"k = {k} < {depth}), the forward of the k-NN feature of scripts/heal_vexact.py; exactly equivariant over "
            f"blocks 1..{k} only, so not mirror-invariant")


def variant_row_extras(info: dict) -> dict:
    """Extra CSV columns for variants with checks after loading (healed V-exact): the feature used and the
    check values. Empty for every other variant."""
    vc = (info or {}).get("vexact_checks")
    if not vc:
        return {}
    fl = vc["probe_feature_flip"]
    rec = vc.get("recorded_after_healing") or {}
    rec_ex = rec.get("exactness.json:max_rel_inside")
    knn = info.get("recorded_knn_healed_top1")
    dt = str(vc.get("dtype", "")).replace("torch.", "")
    note = (f"{info.get('feature', '')}; checks after loading ({vc['n_images']} {vc['protocol']} check images, {dt}): "
            f"span exactness max rel {vc['span_exact_max_rel_inside']:.1e} (tol {vc['tol']:g}"
            + (f"; {rec_ex:.1e} recorded after healing" if isinstance(rec_ex, (int, float)) else "")
            + f"), image vs mirror probe feature max rel {fl['rel_max']:.1e}: {'PASS' if vc['pass'] else 'FAIL'}"
            + (f"; Tier-1 k-NN of the healed run {knn:.2f} ({info.get('recorded_knn_precision', 'fp32')} features)"
               if isinstance(knn, (int, float)) else ""))
    grid = info.get("position_grid")
    if grid:
        note += (f"; deviation: {grid['deviation']} (re-projection changed the interpolated grid by "
                 f"{grid['reprojection_rel_change']:.1e}, relative; without it the span error is "
                 f"{grid['span_exact_without_reprojection']:.1e})")
    return {"notes": note, "vexact_even_head": bool(info.get("even_head")),
            "vexact_span_exact_rel_max": vc["span_exact_max_rel_inside"], "vexact_feature_flip_rel_max": fl["rel_max"],
            "vexact_checks_pass": bool(vc["pass"])}


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
CSV_COLUMNS = ["model", "variant", "label", "family", "method", "k", "top1", "top5", "mean_per_class_top1",
               "knn_check_top1", "chosen_wd", "wd_at_grid_edge", "holdout_top1_at_chosen", "final_hit_iter_limit",
               "final_lbfgs_iters", "feature_dim", "feature_kind", "n_bank", "n_val", "n_wd_train", "n_wd_holdout",
               "converted_params_kept", "converted_params_dense", "model_params", "t_build_s", "t_extract_s",
               "t_probe_s", "amp", "feature_dtype", "wd_grid", "git_hash", "git_dirty", "seed", "date",
               "vexact_even_head", "vexact_span_exact_rel_max", "vexact_feature_flip_rel_max", "vexact_checks_pass",
               "notes"]


def probe_json_path(out: Path, model_name: str, variant: str) -> Path:
    return Path(out) / "probe" / model_name / f"{variant}.json"


def write_tables(out: Path, csv_name: str = "t2_linear.csv", md_name: str = "T2_LINEAR.md") -> list:
    """Rebuild the CSV and the markdown summary from every probe JSON under <out>/probe/."""
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "probe").glob("*/*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "row" in r:
            rows.append(r["row"])
    if not rows:
        return rows
    rows.sort(key=lambda r: (r["model"], r["k"], r["family"] != "original", r["variant"]))
    df = pd.DataFrame(rows)
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + [c for c in df.columns if c not in CSV_COLUMNS]]
    df.to_csv(Path(out) / csv_name, index=False)
    write_summary(Path(out) / md_name, rows, csv_name)
    return rows


def _f(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _int(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "-"


def write_summary(path: Path, rows: list, csv_name: str) -> None:
    lines = ["# Tier 2: ImageNet linear probe", "",
             "Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after "
             "the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for "
             "models without one), features from bf16 forward passes, stored in float16. Trained on the 100k "
             "bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay "
             "chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted "
             "on the whole bank. Full-batch L-BFGS on standardised features.", "",
             "The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same "
             "cached features: it should sit close to the Tier-1 k-NN number of the same variant.", ""]
    if any(r.get("vexact_checks_pass") is not None for r in rows):
        lines += ["Healed V-exact rows (healed_vexact_k<K>, scripts/heal_vexact.py): the same feature through the "
                  "student's own final norm, which at full conversion holds the even head, so the feature of an image "
                  "and of its mirror are equal in fp32 (bf16 forward passes round the mirrored computation differently). "
                  "The loaded student is checked before probing (span exactness and image vs mirror feature, fp32); "
                  "values and the feature statement are in the CSV columns vexact_* and notes.", ""]
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        orig = next((r for r in rs if r["family"] == "original"), None)
        lines += [f"## {model}", "",
                  "| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | "
                  "Model params | Notes |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            delta = "-" if orig is None or r is orig else _f(r["top1"] - orig["top1"])
            notes = []
            if r.get("wd_at_grid_edge"):
                notes.append("wd at grid edge")
            if r.get("final_hit_iter_limit"):
                notes.append("final fit hit the iteration limit")
            if r.get("vexact_checks_pass") is not None:
                notes.append(f"loaded student checked: span exactness {r['vexact_span_exact_rel_max']:.1e}, "
                             f"image vs mirror feature {r['vexact_feature_flip_rel_max']:.1e}, "
                             f"{'PASS' if r['vexact_checks_pass'] else 'FAIL'}; "
                             + ("even head, feature mirror-invariant" if r.get("vexact_even_head") else
                                "no even head (k < depth)"))
            lines.append(f"| {r['variant']} | {r['label']} | {_f(r['top1'])} | {_f(r['top5'])} | {delta} | "
                         f"{_f(r.get('knn_check_top1'))} | {r['chosen_wd']:.3g} | {_int(r.get('model_params'))} | "
                         f"{'; '.join(notes) or '-'} |")
        lines.append("")
    git = sorted({str(r["git_hash"])[:10] + (" (dirty)" if r.get("git_dirty") else "") for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    lines += [f"Source: `{csv_name}` and `probe/<model>/<variant>.json` in this folder. Commit(s): {', '.join(git)}; "
              f"seed(s): {', '.join(map(str, seeds))}. Feature extraction and probe timings are in the CSV.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg",
                   help="registry name (mirror.adapters_timm.MODEL_REGISTRY): dinov2_vitb14_reg, dinov2_vitb14, mae_b16, ...")
    p.add_argument("--variants", default="original", help="comma-separated variant names (see the module docstring)")
    p.add_argument("--out", default=None, help="output directory (default runs/t2; tiny_out/t2 with --tiny)")
    p.add_argument("--operators", default=None, help="X2 e1_operators.npz (default per model, see MODEL_DEFAULTS)")
    p.add_argument("--m1-structure", default=None, help="M1 m1_structure.npz (default per model)")
    p.add_argument("--heal-dir", default=None, help="M3 output directory with <method>/final/trained_blocks.pt")
    p.add_argument("--heal-config", default=None, help="fallback config when <heal-dir>/<method>/summary.json is missing")
    p.add_argument("--vexact-dir", default=None,
                   help="scripts/heal_vexact.py output directory with k<K>/final/trained_blocks.pt (default per model)")
    p.add_argument("--vexact-checkpoint", default=None,
                   help="healed V-exact checkpoint file (overrides --vexact-dir; one healed_vexact variant per run)")
    p.add_argument("--vexact-config", default=None,
                   help=f"fallback config when <vexact-dir>/k<K>/summary.json is missing (default {DEFAULT_VEXACT_CONFIG})")
    p.add_argument("--random-seed", type=int, default=1, help="matched random structure (MLP-only variants)")
    p.add_argument("--seed", type=int, default=0, help="global seed and the weight-decay split")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    p.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--wd-grid", default=DEFAULT_WD_GRID, help="coarse weight-decay grid")
    p.add_argument("--no-wd-refine", action="store_true", help="skip the half-decade refinement around the best wd")
    p.add_argument("--holdout-frac", type=float, default=0.1, help="share of the bank held out to choose wd")
    p.add_argument("--max-iter", type=int, default=1000, help="L-BFGS iterations per solve")
    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--tol-grad", type=float, default=1e-6)
    p.add_argument("--probe-tf32", action="store_true", help="allow TF32 matmuls in the probe (faster, less exact)")
    p.add_argument("--skip-probe", action="store_true", help="extract (and cache) features only")
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--force-probe", action="store_true")
    p.add_argument("--memory-fraction", type=float, default=None, help=f"override the {DEFAULT_MEMORY_FRACTION} GPU cap")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    p.add_argument("--tiny-classes", type=int, default=4)
    p.add_argument("--tiny-bank-per-class", type=int, default=20)
    p.add_argument("--tiny-val-per-class", type=int, default=8)
    return p.parse_args(argv)


def resolve_paths(a) -> dict:
    d = {} if a.tiny else dict(MODEL_DEFAULTS.get(a.model, {}))
    return {"operators": a.operators or d.get("operators"), "m1_structure": a.m1_structure or d.get("m1_structure"),
            "heal_dir": a.heal_dir or d.get("heal_dir"),
            "vexact_dir": getattr(a, "vexact_dir", None) or d.get("vexact_dir"),
            "vexact_checkpoint": getattr(a, "vexact_checkpoint", None),
            "vexact_config": getattr(a, "vexact_config", None)}


def make_factory(adapter, a, tiny: bool, paths: dict) -> "VariantFactory":
    """The VariantFactory of a run, from the parsed arguments and resolve_paths (both Tier-2 scripts)."""
    return VariantFactory(adapter, a.model, tiny, paths["operators"], paths["m1_structure"], paths["heal_dir"],
                          a.heal_config, a.random_seed, vexact_dir=paths.get("vexact_dir"),
                          vexact_checkpoint=paths.get("vexact_checkpoint"), vexact_config=paths.get("vexact_config"))


def main(argv=None) -> dict:
    a = parse_args(argv)
    variants = parse_variants(a.variants)
    check_vexact_checkpoint_arg(variants, a.vexact_checkpoint)
    tiny = a.tiny
    model_name = "tiny" if tiny else a.model
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/t2" if tiny else "runs/t2")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    apply_memory_cap(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    amp = None if a.amp == "none" else a.amp
    fdtype = np.float16 if a.feature_dtype == "float16" else np.float32
    grid = [float(x) for x in a.wd_grid.split(",") if x.strip()]
    solver = {"max_iter": a.max_iter, "history_size": a.history_size, "tol_grad": a.tol_grad}
    paths = resolve_paths(a)
    log(f"Tier 2 linear probe: model {model_name}, variants {[v.name for v in variants]}, out {out}, device {device}, "
        f"commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("load"):
        adapter = build_adapter(a.model, tiny, device, a.img_size)
        if tiny:
            data = TinyData(a.tiny_classes, a.tiny_bank_per_class, a.tiny_val_per_class, seed=a.seed,
                            batch_size=min(a.batch_size, 16))
        else:
            data = ProtocolData(_resolve(a.cache_dir), a.batch_size, a.num_workers, device.type == "cuda")
        if tiny and paths["operators"] is None and any(v.family in ("mlp", "m1") for v in variants):
            # Tiny smoke path: fit operators and twins in-run on synthetic images (run_e1 layout).
            from mirror import variants as V

            V.load_or_make_structure(None, True, adapter, V.TinyProtocol(), device, out)
            paths["operators"] = str(out / "tiny_fit" / "results" / "e1_operators.npz")
        factory = make_factory(adapter, a, tiny, paths)
        kind = feature_kind(adapter)
    log(f"adapter: depth {adapter.depth}, d {adapter.embed_dim}, prefix tokens {adapter.n_prefix_tokens}, "
        f"feature {kind} + mean patch")
    for v in variants:                        # every input file of every variant exists before any work starts
        factory.sources(v)

    results, run_timing = [], {}
    for v in variants:
        vt = StageTimer()
        log(f"=== {v.name}: {v.label}")
        sources = factory.sources(v)          # fails early when a needed file is missing
        expect = {"model": model_name, "variant": v.name, "recipe": FEATURE_RECIPE, "amp": a.amp,
                  "dtype": a.feature_dtype, "sources": sources, "img_size": adapter.img_size}
        feats, model, info = {}, None, None
        t_build = t_extract = 0.0
        for split in SPLITS:
            fp = feature_path(out, model_name, v.name, split)
            hit = None if a.force_features else cached_features(fp, {**expect, "split": split,
                                                                     "n": len(data.loader(split).dataset)})
            if hit is not None:
                feats[split] = hit
                log(f"  {split}: cached features {fp.name} {hit[0].shape}")
                continue
            if model is None:
                t0 = time.time()
                with vt.stage(f"build {v.name}"):
                    model, info = factory.build(v)
                t_build = time.time() - t0
                write_json(out / "probe" / model_name / f"{v.name}_variant.json",
                           {"info": info, "sources": sources, "git": git})
            t0 = time.time()
            with vt.stage(f"extract {v.name} {split}"):
                X, y = extract_features(adapter, data.loader(split), device, model, amp, fdtype, tag=split)
            t_extract += time.time() - t0
            meta = {**expect, "split": split, "n": int(X.shape[0]), "dim": int(X.shape[1]), "feature_kind": kind,
                    "git_hash": git["git_hash"], "seed": a.seed, "seconds": round(time.time() - t0, 2),
                    "batch_size": data.loader(split).batch_size}
            save_features(fp, X, y, data.rows(split), meta)
            feats[split] = (X, y, meta)
            log(f"  {split}: {X.shape} saved to {fp}")
        if model is not None and model is not adapter.model:
            del model
        free_memory(device)
        if info is None:
            vj = out / "probe" / model_name / f"{v.name}_variant.json"
            info = json.loads(vj.read_text(encoding="utf-8"))["info"] if vj.exists() else {
                "variant": v.name, "label": v.label, "family": v.family, "method": v.method, "k": v.k}
        if a.skip_probe:
            run_timing[v.name] = vt.stages
            continue

        pj = probe_json_path(out, model_name, v.name)
        if pj.exists() and not a.force_probe and t_extract == 0.0:
            prev = json.loads(pj.read_text(encoding="utf-8"))
            if prev.get("probe_settings") == _jsonable({"grid": grid, "refine": not a.no_wd_refine,
                                                        "holdout_frac": a.holdout_frac, "seed": a.seed,
                                                        "solver": solver, "tf32": a.probe_tf32}):
                log(f"  probe done before ({pj.name}): top-1 {prev['row']['top1']:.2f}; skipping")
                results.append(prev["row"])
                continue
        (Xb, yb, mb), (Xv, yv, mv) = feats["bank"], feats["val"]
        torch.backends.cuda.matmul.allow_tf32 = bool(a.probe_tf32)
        t0 = time.time()
        with vt.stage(f"probe {v.name}"):
            pr = run_probe(Xb, yb, Xv, yv, data.num_classes, grid, a.holdout_frac, a.seed, not a.no_wd_refine,
                           device, solver)
            d_first = int(adapter.embed_dim) if kind == "cls" else int(Xb.shape[1] - adapter.embed_dim)
            knn = knn_check(Xb, yb, Xv, yv, d_first, data.num_classes, device)
        t_probe = time.time() - t0
        torch.backends.cuda.matmul.allow_tf32 = False
        sel, fin, test = pr["selection"], pr["final"], pr["test"]
        log(f"  {v.name}: top-1 {test['top1']:.2f}, top-5 {test['top5']:.2f} (wd {sel['chosen_wd']:.3g}); "
            f"k-NN check {knn:.2f}")
        row = {"model": model_name, "variant": v.name, "label": v.label, "family": v.family, "method": v.method,
               "k": v.k, "top1": test["top1"], "top5": test["top5"], "mean_per_class_top1": test["mean_per_class_top1"],
               "knn_check_top1": knn, "chosen_wd": sel["chosen_wd"], "wd_at_grid_edge": sel["wd_at_grid_edge"],
               "holdout_top1_at_chosen": sel["holdout_top1_at_chosen"], "final_hit_iter_limit": fin["hit_limit"],
               "final_lbfgs_iters": fin["n_iter"], "feature_dim": int(Xb.shape[1]), "feature_kind": kind,
               "n_bank": int(Xb.shape[0]), "n_val": int(Xv.shape[0]), "n_wd_train": sel["n_train"],
               "n_wd_holdout": sel["n_holdout"], "converted_params_kept": info.get("converted_params_kept"),
               "converted_params_dense": info.get("converted_params_dense"), "model_params": info.get("model_params"),
               "t_build_s": round(t_build, 1), "t_extract_s": round(t_extract, 1), "t_probe_s": round(t_probe, 1),
               "amp": a.amp, "feature_dtype": a.feature_dtype, "wd_grid": a.wd_grid, "git_hash": git["git_hash"],
               "git_dirty": git["git_dirty"], "seed": a.seed, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               **variant_row_extras(info)}
        np.savez(pj.with_name(f"{v.name}_probe.npz"), W=pr["W"], b=pr["b"], mean=pr["mean"], std=pr["std"],
                 holdout_positions=pr["holdout_positions"])
        write_json(pj, {"row": row, "selection": sel, "final": fin, "test": test, "variant_info": info,
                        "sources": sources, "feature_meta": {"bank": mb, "val": mv},
                        "probe_settings": {"grid": grid, "refine": not a.no_wd_refine, "holdout_frac": a.holdout_frac,
                                           "seed": a.seed, "solver": solver, "tf32": a.probe_tf32},
                        "timing": vt.stages, "git": git, "env": env_info(device)})
        results.append(row)
        run_timing[v.name] = vt.stages
        write_tables(out)
        del feats, Xb, Xv, pr
        free_memory(device)

    rows = write_tables(out)
    write_json(out / f"t2_linear_meta_{model_name}.json", {
        "experiment": "Tier 2 ImageNet linear probe", "model": model_name, "variants": [v.name for v in variants],
        "args": vars(a), "paths": paths, "git": git, "env": env_info(device), "seed": a.seed,
        "feature_recipe": FEATURE_RECIPE, "timing": {"stages": timer.stages, "variants": run_timing,
                                                     "total_s": timer.total()}})
    log(f"done in {timer.total() / 60:.1f} min; {len(rows)} rows in {out / 't2_linear.csv'}")
    return {"out": str(out), "rows": results, "all_rows": rows}


if __name__ == "__main__":
    main()

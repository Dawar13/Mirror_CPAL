#!/usr/bin/env python
"""CORE-06: "what" against "which way": orientation is kept, not destroyed (PROJECT_NEXT_PHASE.md CORE-06,
PREREG_NEXT_PHASE.md P6.1 to P6.3).

    python scripts/core06_orientation.py --cub-root $HF_HOME/cub/CUB_200_2011 \
        --healed-run vexact_s0=runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0 \
        --healed-run vexact_s1=runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1 \
        --healed-run vexact_s2=runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2
    python scripts/core06_orientation.py --tiny --out tiny_out/core06          # synthetic CUB, tiny ViT, CPU

Data: CUB-200-2011 (mirror/cub.py; configs/data/cub_parts.yaml), its official train/test split; images resized to
256 (bicubic) and centre-cropped to 224, cached once as uint8 (--cache-dir). Facing direction: the sign of
(x_beak - x_tail), parts 2 and 14, when both are visible and more than 10% of the box width apart (+1 = the bird
faces right; probe class 1 = right, 0 = left); other images have no facing label and are left out of every facing
probe. The paper should say that CUB images overlap with ImageNet (configs/data/cub_parts.yaml warning).

Models (fp32 forward passes, no autocast, so exact invariance is limited only by fp32 rounding):
  original   the released dinov2_vitb14_reg. Even and odd parts via the final-layer operator A_L
             (A_layers[L] of the canonical structure, results/MUST-07/structure/<model>_s0.npz). Convention, matching
             the Mirror-exact even head (RMSNorm, y (I + A*)/2, affine): the final norm's normalisation without its
             affine gives u (for the CLS token, and for the mean of the patch tokens), then
                 even = gamma * (u (I + A)/2) + beta,   odd = gamma * (u (I - A)/2),   full = even + odd,
             so full is the model's own final-norm feature, the bias beta belongs to the even part, and an image and
             its mirror swap the sign of the odd part to the extent that the model is equivariant under A.
  healed     a healed Mirror-exact run (--healed-run NAME=DIR, mirror.heal_structured.load_healed_model; full
             conversion, so model.norm = Sequential(rms, even, affine)). A* from info["structure"]["A_star"]; the even
             part is the model's own final feature (affine(rms(t) P), P = (I + A*)/2, checked against the even head's
             buffer), odd = gamma * (rms(t) - rms(t) P), full = even + odd (the final norm with the even projection
             bypassed). Exactly invariant (even) and exactly sign-flipping (odd) under a mirror, up to fp32 rounding.
  --zeroshot-exact   also the zero-shot full Mirror-exact model built from the structure (reported only).
Feature sets: even, odd and full parts of the CLS token, and of the mean patch token.

Probes: multinomial logistic regression (scripts/tier2_linear.py fit_logreg, by import: L-BFGS, standardised features,
weight decay on W only). The weight decay is chosen by --folds-fold stratified cross-validation on the CUB training
split only (pooled held-out accuracy; ties go to the larger weight decay), then refitted on the whole training split.
Species (200 classes): even, odd and full CLS. Facing (2 classes): every feature set. Test sets: the natural test
images, and the mirror-augmented test set (every test image with a facing label plus its mirror with the facing label
flipped; for species, every test image plus its mirror, same label).

Scoring (PREREG_NEXT_PHASE.md CORE-06; the operational choices below are this script's and are listed in the report):
  P6.1  species top-1 on natural test images: |full CLS - even CLS| <= 1.0 point.
  P6.2  facing accuracy on natural test images from the odd part >= 85%. The odd part is the odd CLS, unless its
        cross-validated facing accuracy on the training split is below 85% ("too small to carry facing direction"),
        decided before any test image is scored: then the odd part of the mean patch token, and the report says so.
  P6.1 and P6.2 are scored on every model probed (original and each healed Mirror-exact seed) and pass only if they
  hold on all of them.
  P6.3  healed Mirror-exact even CLS: facing accuracy <= 60% on natural test images and exactly 50% on the
        mirror-augmented set, on every healed seed. "Exactly" allows fp32 near-ties (PREREG Section 1): every pair whose
        two predictions differ must have a relative logit margin below 1e-4 on one of its two copies.

Outputs (resumable; features and finished probes are reused): <out>/features/<variant>_{natural,mirror}.npz,
<out>/probe/<variant>__<task>__<features>.json, <out>/probes.csv, <out>/metrics.json, <out>/SUMMARY.md, <out>/meta.json.
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.15) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import cub as CUB  # noqa: E402
from mirror.data import make_loader, mirror as mirror_images, normalize_for  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "CORE-06"
DEFAULT_MODEL = "dinov2_vitb14_reg"
DEFAULT_STRUCTURE = "results/MUST-07/structure/{model}_s0.npz"
DEFAULT_WD_GRID = "1e-5,1e-4,1e-3,1e-2,1e-1,1"
P61_MAX_GAP = 1.0
P62_MIN = 85.0
P63_MAX_NATURAL = 60.0
TIE_REL = 1e-4
PARTS = ("cls_even", "cls_odd", "patch_even", "patch_odd")
FEATURE_SETS = {                      # name -> (columns of PARTS summed)
    "even_cls": ("cls_even",), "odd_cls": ("cls_odd",), "full_cls": ("cls_even", "cls_odd"),
    "even_patch": ("patch_even",), "odd_patch": ("patch_odd",), "full_patch": ("patch_even", "patch_odd"),
}
TASK_FEATURES = {"species": ("even_cls", "odd_cls", "full_cls"),
                 "facing": ("even_cls", "odd_cls", "full_cls", "even_patch", "odd_patch", "full_patch")}


def load_tier2():
    """scripts/tier2_linear.py as a module (its probe code is reused, not copied); its import-time GPU cap is
    replaced by this script's own."""
    key = "_core06_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


T2 = load_tier2()


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="registry name of the original model")
    p.add_argument("--structure", default=None, help=f"canonical span structure (default {DEFAULT_STRUCTURE})")
    p.add_argument("--healed-run", action="append", default=None, metavar="[NAME=]DIR",
                   help="healed Mirror-exact run (mirror.heal_structured.load_healed_model); repeatable")
    p.add_argument("--zeroshot-exact", action="store_true",
                   help="also probe the zero-shot full Mirror-exact model built from --structure (reported only)")
    p.add_argument("--no-original", action="store_true", help="skip the original model")
    p.add_argument("--cub-root", default=None, help="CUB_200_2011 folder (default $HF_HOME/cub/CUB_200_2011)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache" / "cub"))
    p.add_argument("--out", default=None, help="output directory (default results/CORE-06; tiny: tiny_out/core06)")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default="none", choices=["none", "bf16"],
                   help="forward precision (default fp32: exact-invariance claims are scored on fp32 features)")
    p.add_argument("--wd-grid", default=DEFAULT_WD_GRID)
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds on the training split")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="fold assignment and global seed")
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--force-probe", action="store_true")
    p.add_argument("--gpu-frac", type=float, default=0.15, help="per-process GPU memory cap (read at import)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="synthetic CUB, tiny random ViT, CPU")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Even and odd parts of the final feature
# ----------------------------------------------------------------------------------------------
class FinalParts:
    """Even and odd parts of the final-norm CLS token and mean patch token (see the module docstring).

    kind "layernorm" (original model): u = LayerNorm(t) without affine, P = (I + A_L)/2.
    kind "even_head" (Mirror-exact at full conversion): u = the model's own RMSNorm(t), P = its even head's buffer.
    Returns [B, 4, d] in PARTS order: cls_even, cls_odd, patch_even, patch_odd."""

    def __init__(self, kind: str, P: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6,
                 rms=None, operator_source: str = ""):
        if kind not in ("layernorm", "even_head"):
            raise ValueError(kind)
        self.kind, self.eps, self.rms, self.operator_source = kind, float(eps), rms, operator_source
        self.P = P.float()
        self.gamma, self.beta = gamma.detach().float(), beta.detach().float()
        self.d_minus = int(round(float(P.shape[0] - torch.trace(P.double()))))

    def to(self, device):
        self.P, self.gamma, self.beta = self.P.to(device), self.gamma.to(device), self.beta.to(device)
        return self

    def normalise(self, t: torch.Tensor) -> torch.Tensor:
        if self.kind == "layernorm":
            return F.layer_norm(t.float(), (t.shape[-1],), None, None, self.eps)
        return self.rms(t.float()).float()

    def __call__(self, tokens: torch.Tensor, n_prefix: int) -> torch.Tensor:
        u = self.normalise(tokens)
        out = []
        for v in (u[:, 0], u[:, int(n_prefix):].mean(dim=1)):
            ve = v @ self.P
            out += [ve * self.gamma + self.beta, (v - ve) * self.gamma]
        return torch.stack(out, dim=1)


def layernorm_parts(model, A, source: str) -> FinalParts:
    norm = model.norm
    if not isinstance(norm, torch.nn.LayerNorm):
        raise ValueError(f"the original model's final norm is {type(norm).__name__}, expected LayerNorm")
    A = np.asarray(A, dtype=np.float64)
    P = 0.5 * (np.eye(A.shape[0]) + A)
    P = 0.5 * (P + P.T)
    dev = norm.weight.device
    return FinalParts("layernorm", torch.as_tensor(P, device=dev), norm.weight, norm.bias, eps=norm.eps,
                      operator_source=source)


def even_head_parts(model, A_star, source: str) -> tuple:
    """FinalParts of a Mirror-exact model with an even head; checks the head's P against (I + A*)/2."""
    norm = model.norm
    if not (isinstance(norm, torch.nn.Sequential) and hasattr(norm, "even") and hasattr(norm, "rms")):
        raise ValueError("CORE-06 needs a Mirror-exact model at full conversion (model.norm = Sequential(rms, even, "
                         "affine)); this model has no even head")
    P = norm.even.P.detach()
    check = None
    if A_star is not None:
        A = np.asarray(A_star, dtype=np.float64)
        P_ref = 0.5 * (np.eye(A.shape[0]) + A)
        check = float(np.abs(P.double().cpu().numpy() - P_ref).max())
        if check > 1e-5:
            raise ValueError(f"the even head's projector differs from (I + A*)/2 by {check:.2e} (A* from {source})")
    return FinalParts("even_head", P, norm.affine.weight, norm.affine.bias, rms=norm.rms,
                      operator_source=source), check


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------
def tiny_runtime(out: Path, device):
    """Tiny adapter, a span structure fitted on synthetic images, and a synthetic CUB folder."""
    from mirror import mirror_family as MF
    from mirror import variants as V
    from mirror.adapters import TimmTinyAdapter

    adapter = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32).to(device=device)
    sp = out / "structure" / "tiny_s0.npz"
    if sp.exists():
        struct = MF.load_span_structure(sp)
    else:
        proto = V.TinyProtocol()
        struct = MF.fit_span_structure(adapter, proto.fit_loader(0), device, model_name="tiny", top_k=3, workers=1)
        MF.save_span_structure(sp, struct)
        struct = MF.load_span_structure(sp)
    root = out / "tiny_cub" / "CUB_200_2011"
    if not (root / "parts" / "part_locs.txt").exists():
        CUB.write_synthetic_cub(root, n_classes=4, per_class=8, size=64, seed=0)
    return adapter, struct, root


def build_variants(a, adapter, struct, struct_path, device) -> list:
    """[(name, role, model, FinalParts, info)]. Roles: original, healed_exact, zeroshot_exact."""
    from mirror import mirror_family as MF

    L = adapter.depth
    out = []
    if not a.no_original:
        parts = layernorm_parts(adapter.model, struct.A_layers[L],
                                f"A_layers[{L}] (final-layer operator) of {struct_path}")
        out.append(("original", "original", adapter.model, parts.to(device),
                    {"operator": parts.operator_source, "d_minus": parts.d_minus}))
    if a.zeroshot_exact or a.tiny:
        m, info = MF.build_mirror_variant(adapter, "mirror_exact", range(L), ("mlp", "attn"), struct)
        m = m.to(device).float().eval()
        parts, chk = even_head_parts(m, struct.A_star[L], f"A_star[{L}] of {struct_path}")
        name = "exact_tiny" if a.tiny else "zeroshot_exact"
        role = "healed_exact" if a.tiny else "zeroshot_exact"   # tiny: the zero-shot model stands in for a healed one
        out.append((name, role, m, parts.to(device), {"operator": parts.operator_source, "d_minus": parts.d_minus,
                                                      "even_head_check": chk, "builder": "build_mirror_variant"}))
    _, dirs = T2.healed_run_variants(a.healed_run)
    for name, run in dirs.items():
        from mirror.heal_structured import load_healed_model

        if not Path(run).is_dir():
            raise FileNotFoundError(f"--healed-run {name}: {run} not found")
        ad_h, m, info = load_healed_model(run, device)
        for attr in ("depth", "embed_dim", "n_prefix_tokens", "patch_size"):
            if int(getattr(ad_h, attr)) != int(getattr(adapter, attr)):
                raise ValueError(f"{name}: healed model {attr} {getattr(ad_h, attr)} != {getattr(adapter, attr)}")
        st = info.get("structure") or {}
        if info.get("family") != "mirror" or st.get("A_star") is None:
            raise ValueError(f"{name}: {run} is not a Mirror-family run with A* (family {info.get('family')})")
        m = m.to(device).float().eval()
        parts, chk = even_head_parts(m, st["A_star"], f"info['structure']['A_star'] of {run}")
        checks = info.get("checks")
        out.append((name, "healed_exact", m, parts.to(device),
                    {"operator": parts.operator_source, "d_minus": parts.d_minus, "even_head_check": chk,
                     "run": str(run), "method": info.get("method"), "seed": info.get("seed"),
                     "constraint_checks": T2.small_info(checks) if checks else None,
                     "sources": T2.VariantFactory(adapter, a.model, False, healed_runs={name: run}).sources(
                         T2.VariantName(name, "healed_run", "run", 0))}))
    return out


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def extract_parts(adapter, model, parts: FinalParts, loader, device, mirrored: bool, amp: str = "none",
                  check_first: bool = True) -> tuple:
    """[N, 4, d] float32 parts for every image of `loader` (mirrored images when `mirrored`), and a check of the
    first batch: the full CLS part against the model's own final-norm CLS feature (with the even projection
    bypassed for a Mirror-exact model)."""
    import contextlib

    device = torch.device(device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and device.type == "cuda"
           else contextlib.nullcontext())
    n = len(loader.dataset)
    out, pos, check = None, 0, None
    for imgs, _ in loader:
        x = normalize_for(adapter, imgs.to(device, non_blocking=True))
        if mirrored:
            x = mirror_images(x)
        with ctx:
            t = adapter.forward_tokens(x, model)
        f = parts(t.float(), adapter.n_prefix_tokens)
        if check_first and check is None:
            own = (T2.final_norm_tokens(model, t, skip_even=True)[:, 0].float() if parts.kind == "even_head"
                   else model.norm(t)[:, 0].float())
            full = f[:, 0] + f[:, 1]
            check = float((full - own).abs().max() / own.abs().max().clamp_min(1e-30))
            if parts.kind == "even_head":
                ev = model.norm(t)[:, 0].float()
                check = max(check, float((f[:, 0] - ev).abs().max() / ev.abs().max().clamp_min(1e-30)))
        if out is None:
            out = np.empty((n, 4, f.shape[-1]), dtype=np.float32)
        out[pos:pos + f.shape[0]] = f.cpu().numpy()
        pos += f.shape[0]
    if pos != n:
        raise RuntimeError(f"loader gave {pos} images, expected {n}")
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite features")
    return out, check


def feature_file(out: Path, variant: str, kind: str) -> Path:
    return out / "features" / f"{variant}_{kind}.npz"


def cached(path: Path, expect: dict):
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta_json"]))
            if any(meta.get(k) != T2._jsonable(v) for k, v in expect.items()):
                log(f"cached {path.name} has other settings; recomputing")
                return None
            return z["X"], meta
    except Exception as exc:  # noqa: BLE001 - unreadable cache: recompute
        log(f"cached {path.name} unreadable ({exc}); recomputing")
        return None


def save_features(path: Path, X: np.ndarray, rows: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez(fh, X=X, rows=np.asarray(rows, dtype=np.int64), meta_json=np.asarray(json.dumps(T2._jsonable(meta))))
    os.replace(tmp, path)


def feature_set(X: np.ndarray, name: str) -> np.ndarray:
    cols = [PARTS.index(c) for c in FEATURE_SETS[name]]
    f = X[:, cols[0]].astype(np.float32)
    for c in cols[1:]:
        f = f + X[:, c]
    return f


def invariance_checks(nat: np.ndarray, mir: np.ndarray) -> dict:
    """Image against mirror on the same rows: even parts should be equal, odd parts opposite (exact for Mirror-exact).
    Largest relative difference over images, ||e(mx) - e(x)|| / ||e(x)|| and ||o(mx) + o(x)|| / ||o(x)||, plus the
    norm share of the odd part."""
    out = {}
    for tag, ie, io in (("cls", 0, 1), ("patch", 2, 3)):
        e, em, o, om = nat[:, ie], mir[:, ie], nat[:, io], mir[:, io]
        ne, no = np.linalg.norm(e, axis=1), np.linalg.norm(o, axis=1)
        out[f"{tag}_even_rel_max"] = float((np.linalg.norm(em - e, axis=1) / np.maximum(ne, 1e-30)).max())
        out[f"{tag}_odd_rel_max"] = float((np.linalg.norm(om + o, axis=1) / np.maximum(no, 1e-30)).max())
        out[f"{tag}_odd_rel_median"] = float(np.median(np.linalg.norm(om + o, axis=1) / np.maximum(no, 1e-30)))
        out[f"{tag}_odd_norm_share_median"] = float(np.median(no ** 2 / np.maximum(ne ** 2 + no ** 2, 1e-30)))
    return out


# ----------------------------------------------------------------------------------------------
# Probes (cross-validation on the training split only)
# ----------------------------------------------------------------------------------------------
def stratified_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id 0..k-1 per sample, stratified by label (a fixed permutation per class; round-robin with a running
    offset so that classes smaller than k spread over the folds)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    offset = 0
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return fold


def cv_select_wd(X: np.ndarray, y: np.ndarray, num_classes: int, grid, folds: int, seed: int, device,
                 solver: dict) -> dict:
    """Weight decay by `folds`-fold stratified cross-validation on (X, y), the training split only: pooled held-out
    top-1 over the folds; the highest wins, ties to the larger weight decay. Each fold's path is warm-started from the
    previous (larger) weight decay. Never sees a test image: it takes none."""
    device = torch.device(device)
    fold = stratified_folds(y, folds, seed)
    inits = [None] * folds
    trials = []
    for wd in sorted({float(w) for w in grid}, reverse=True):
        correct, n = 0.0, 0
        for f in range(folds):
            tr, va = fold != f, fold == f
            if not va.any() or not tr.any():
                continue
            Xt = torch.as_tensor(X[tr], device=device)
            mean, std = T2.standardiser(Xt)
            Xt = T2.standardise(Xt, mean, std)
            yt = torch.as_tensor(y[tr], dtype=torch.long, device=device)
            r = T2.fit_logreg(Xt, yt, num_classes, wd, init=inits[f], **solver)
            inits[f] = (r["W"], r["b"])
            Xv = T2.standardise(torch.as_tensor(X[va], device=device), mean, std)
            ev = T2.evaluate_probe(Xv, torch.as_tensor(y[va], dtype=torch.long, device=device), r["W"], r["b"],
                                   num_classes)
            correct += ev["top1"] * ev["n"] / 100.0
            n += ev["n"]
        trials.append({"wd": wd, "cv_top1": 100.0 * correct / max(n, 1), "n": int(n)})
    best = max(trials, key=lambda t: (round(t["cv_top1"], 9), t["wd"]))
    return {"chosen_wd": best["wd"], "cv_top1": best["cv_top1"], "trials": trials, "folds": int(folds),
            "seed": int(seed), "wd_at_grid_edge": best["wd"] in (min(grid), max(grid))}


def fit_final(X: np.ndarray, y: np.ndarray, num_classes: int, wd: float, device, solver: dict) -> dict:
    device = torch.device(device)
    Xt = torch.as_tensor(X, device=device)
    mean, std = T2.standardiser(Xt)
    r = T2.fit_logreg(T2.standardise(Xt, mean, std), torch.as_tensor(y, dtype=torch.long, device=device),
                      num_classes, wd, **solver)
    return {"W": r["W"], "b": r["b"], "mean": mean, "std": std, "train_objective": r["loss"],
            "hit_limit": r["hit_limit"], "n_iter": r["n_iter"]}


@torch.no_grad()
def logits_of(probe: dict, X: np.ndarray) -> np.ndarray:
    Xs = T2.standardise(torch.as_tensor(X, device=probe["W"].device), probe["mean"], probe["std"])
    return torch.addmm(probe["b"], Xs, probe["W"]).double().cpu().numpy()


def rel_margin(logits: np.ndarray) -> np.ndarray:
    """(top-1 logit - top-2 logit) / max(|top-1|, |top-2|) per row."""
    s = np.sort(logits, axis=1)
    t1, t2 = s[:, -1], s[:, -2]
    return (t1 - t2) / np.maximum(np.maximum(np.abs(t1), np.abs(t2)), 1e-30)


def score_natural(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {"acc": 100.0 * float((pred == y).mean()), "n": int(len(y))}


def score_mirror_augmented(lg_nat: np.ndarray, lg_mir: np.ndarray, y_nat: np.ndarray, y_mir: np.ndarray,
                           tie_rel: float = TIE_REL) -> dict:
    """Accuracy on the mirror-augmented set (each natural image and its mirror, labels y_nat and y_mir), the share of
    pairs predicted alike, and the rounding allowance: every pair whose predictions differ must be a near-tie
    (relative margin below tie_rel on one of its copies)."""
    p_nat, p_mir = lg_nat.argmax(1), lg_mir.argmax(1)
    correct = np.concatenate([p_nat == y_nat, p_mir == y_mir])
    differ = p_nat != p_mir
    m = np.minimum(rel_margin(lg_nat), rel_margin(lg_mir))
    return {"acc": 100.0 * float(correct.mean()), "n": int(len(correct)), "n_pairs": int(len(y_nat)),
            "pairs_same_prediction": 100.0 * float((~differ).mean()), "n_pairs_differ": int(differ.sum()),
            "n_pairs_differ_not_near_tie": int((differ & (m >= tie_rel)).sum()),
            "max_margin_of_differing_pairs": float(m[differ].max()) if differ.any() else None}


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def verdict(ok) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def score_predictions(results: dict, roles: dict) -> dict:
    """P6.1 to P6.3 from results[variant][task][feature] (see run_probes); roles: variant -> role."""
    scored = [v for v, r in roles.items() if r in ("original", "healed_exact") and v in results]
    p61, p62 = {}, {}
    for v in scored:
        sp = results[v].get("species", {})
        if "even_cls" in sp and "full_cls" in sp:
            gap = sp["full_cls"]["natural"]["acc"] - sp["even_cls"]["natural"]["acc"]
            p61[v] = {"even_cls": sp["even_cls"]["natural"]["acc"], "full_cls": sp["full_cls"]["natural"]["acc"],
                      "full_minus_even": gap, "pass": bool(abs(gap) <= P61_MAX_GAP)}
        fc = results[v].get("facing", {})
        if "odd_cls" in fc:
            cv_odd = fc["odd_cls"]["cv"]["cv_top1"]
            fallback = bool(cv_odd < P62_MIN) and "odd_patch" in fc
            feat = "odd_patch" if fallback else "odd_cls"
            p62[v] = {"feature": feat, "fallback_to_mean_patch": fallback, "odd_cls_cv_top1_train": cv_odd,
                      "rule": f"odd CLS unless its training-split cross-validated facing accuracy is below {P62_MIN}%",
                      "facing_acc_natural": fc[feat]["natural"]["acc"],
                      "facing_acc_mirror_augmented": fc[feat]["mirror_augmented"]["acc"],
                      "odd_cls_natural": fc["odd_cls"]["natural"]["acc"],
                      "odd_patch_natural": fc.get("odd_patch", {}).get("natural", {}).get("acc"),
                      "pass": bool(fc[feat]["natural"]["acc"] >= P62_MIN)}
    p63 = {}
    for v in scored:
        if roles[v] != "healed_exact":
            continue
        e = results[v].get("facing", {}).get("even_cls")
        if e is None:
            continue
        ma = e["mirror_augmented"]
        exact50 = ma["n_pairs_differ_not_near_tie"] == 0
        p63[v] = {"natural": e["natural"]["acc"], "mirror_augmented": ma["acc"],
                  "mirror_augmented_raw_exactly_50": bool(ma["acc"] == 50.0),
                  "n_pairs_differ": ma["n_pairs_differ"], "n_pairs_differ_not_near_tie": ma["n_pairs_differ_not_near_tie"],
                  "pass_natural": bool(e["natural"]["acc"] <= P63_MAX_NATURAL), "pass_exact50": bool(exact50),
                  "pass": bool(e["natural"]["acc"] <= P63_MAX_NATURAL and exact50)}

    def all_pass(d):
        return None if not d else bool(all(x["pass"] for x in d.values()))

    have_healed = any(roles[v] == "healed_exact" for v in scored)
    return {
        "P6.1": {"prediction": "species from the even part is within 1 point of the full feature",
                 "operational": f"|full CLS - even CLS| species top-1 on natural test images <= {P61_MAX_GAP}, on every "
                                "model probed (original with A_L, each healed Mirror-exact seed)",
                 "per_model": p61, "verdict": verdict(all_pass(p61)),
                 "complete": bool(p61) and have_healed and "original" in p61},
        "P6.2": {"prediction": "facing direction from the odd part is at least 85%",
                 "operational": f"facing accuracy on natural test images >= {P62_MIN}% from the odd CLS, or from the odd "
                                "mean patch token when the odd CLS's training-split CV accuracy is below "
                                f"{P62_MIN}% (decided before scoring); on every model probed",
                 "per_model": p62, "verdict": verdict(all_pass(p62)),
                 "complete": bool(p62) and have_healed and "original" in p62},
        "P6.3": {"prediction": "facing from healed Mirror-exact's even part is at most 60% on natural test images and "
                               "exactly 50% on the mirror-augmented test set",
                 "operational": f"even CLS, natural <= {P63_MAX_NATURAL}%; mirror-augmented 50% with every differing "
                                f"pair a near-tie (relative logit margin < {TIE_REL:g}); every healed seed",
                 "per_model": p63, "verdict": verdict(all_pass(p63)), "complete": bool(p63)},
    }


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def run_probes(a, out: Path, name: str, nat: np.ndarray, mir: np.ndarray, ann: dict, split: dict, grid, solver,
               device, num_classes: int, rows_out: list) -> dict:
    """Every probe of one variant. nat: features of all images (images.txt order); mir: of the test images
    (split['test'] order). Returns results[task][feature]."""
    train, test = split["train"], split["test"]
    pos_test = {int(r): i for i, r in enumerate(test)}
    sign = split["facing_sign"]
    res = {}
    settings = {"grid": grid, "folds": a.folds, "seed": a.seed, "solver": solver, "num_classes": num_classes}
    for task, feats in TASK_FEATURES.items():
        res[task] = {}
        if task == "species":
            tr_rows, y_tr = train, ann["label"][train]
            te_rows, y_te = test, ann["label"][test]
            aug = {"row": np.concatenate([test, test]), "mirrored": np.r_[np.zeros(len(test), bool), np.ones(len(test), bool)],
                   "label": np.concatenate([y_te, y_te])}
            C = num_classes
        else:
            tr_rows = train[sign[train] != 0]
            y_tr = CUB.facing_class(sign[tr_rows])
            te_rows = test[sign[test] != 0]
            y_te = CUB.facing_class(sign[te_rows])
            ma = CUB.mirror_augmented(te_rows, sign[te_rows])
            aug = {"row": ma["row"], "mirrored": ma["mirrored"], "label": ma["facing"]}
            C = 2
        if len(tr_rows) == 0 or len(te_rows) == 0:
            continue
        for feat in feats:
            pj = out / "probe" / f"{name}__{task}__{feat}.json"
            if pj.exists() and not a.force_probe:
                prev = json.loads(pj.read_text(encoding="utf-8"))
                if prev.get("settings") == T2._jsonable(settings) and prev.get("features_sha") == split["features_sha"][name]:
                    res[task][feat] = prev["result"]
                    rows_out.extend(prev["rows"])
                    continue
            X_tr = feature_set(nat[tr_rows], feat)
            cv = cv_select_wd(X_tr, y_tr, C, grid, a.folds, a.seed, device, solver)
            probe = fit_final(X_tr, y_tr, C, cv["chosen_wd"], device, solver)
            lg_te = logits_of(probe, feature_set(nat[te_rows], feat))
            n_te = len(aug["row"]) // 2
            lg_nat = logits_of(probe, feature_set(nat[aug["row"][:n_te]], feat))
            lg_mir = logits_of(probe, feature_set(mir[[pos_test[int(r)] for r in aug["row"][n_te:]]], feat))
            r = {"cv": cv, "natural": score_natural(lg_te, y_te),
                 "mirror_augmented": score_mirror_augmented(lg_nat, lg_mir, aug["label"][:n_te], aug["label"][n_te:]),
                 "train": {"n": int(len(tr_rows)), "hit_iter_limit": bool(probe["hit_limit"]),
                           "share_class1": float(np.mean(y_tr == 1)) if task == "facing" else None},
                 "test_share_class1": float(np.mean(y_te == 1)) if task == "facing" else None}
            res[task][feat] = r
            rows = []
            for test_set in ("natural", "mirror_augmented"):
                rows.append({"variant": name, "task": task, "features": feat, "test_set": test_set,
                             "acc": r[test_set]["acc"], "n": r[test_set]["n"], "chosen_wd": cv["chosen_wd"],
                             "cv_top1_train": cv["cv_top1"], "wd_at_grid_edge": cv["wd_at_grid_edge"],
                             "n_train": r["train"]["n"],
                             "pairs_same_prediction": r[test_set].get("pairs_same_prediction"),
                             "n_pairs_differ_not_near_tie": r[test_set].get("n_pairs_differ_not_near_tie")})
            rows_out.extend(rows)
            T2.write_json(pj, {"result": r, "rows": rows, "settings": settings, "features_sha": split["features_sha"][name]})
            log(f"  {name} {task} {feat}: natural {r['natural']['acc']:.2f}, mirror-augmented "
                f"{r['mirror_augmented']['acc']:.2f} (wd {cv['chosen_wd']:g}, CV {cv['cv_top1']:.2f})")
    return res


def write_summary(out: Path, metrics: dict, rows: list) -> Path:
    s = metrics["scoring"]
    lines = ["# CORE-06: orientation is kept, not destroyed", "",
             f"Commit {metrics['git']['git_hash'][:10]}{' (dirty)' if metrics['git']['git_dirty'] else ''}; seed "
             f"{metrics['seed']}; PREREG_NEXT_PHASE.md sha256 {str(metrics['prereg'].get('sha256'))[:12]}.", "",
             f"CUB-200-2011: {metrics['data']['n_train']} train and {metrics['data']['n_test']} test images; facing "
             f"labels on {metrics['data']['n_facing_train']} train and {metrics['data']['n_facing_test']} test images "
             f"(share facing right: train {metrics['data']['facing_right_share_train']:.3f}, test "
             f"{metrics['data']['facing_right_share_test']:.3f}). Facing convention: {CUB.FACING_CONVENTION}. CUB "
             "images overlap with ImageNet (Caltech warning); the comparisons are within each model.", "",
             "| Prediction | Verdict | Detail |", "|---|---|---|"]
    for key in ("P6.1", "P6.2", "P6.3"):
        det = "; ".join(f"{v}: " + ", ".join(f"{k} {x:.2f}" if isinstance(x, float) else f"{k} {x}"
                                               for k, x in d.items() if k not in ("rule",))
                        for v, d in s[key]["per_model"].items())
        lines.append(f"| {key} {s[key]['prediction']} | {s[key]['verdict']}"
                     f"{'' if s[key]['complete'] else ' (incomplete)'} | {det or '-'} |")
    lines += ["", "Operational definitions: " + " ".join(f"{k}: {s[k]['operational']}." for k in ("P6.1", "P6.2", "P6.3")),
              "", "## All probes (accuracy %, weight decay chosen by cross-validation on the training split)", "",
              "| Variant | Task | Features | Test set | Accuracy | n | wd | CV (train) | Pairs predicted alike |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ps = r.get("pairs_same_prediction")
        lines.append(f"| {r['variant']} | {r['task']} | {r['features']} | {r['test_set']} | {r['acc']:.2f} | {r['n']} | "
                     f"{r['chosen_wd']:g} | {r['cv_top1_train']:.2f} | {'-' if ps is None else f'{ps:.2f}'} |")
    lines += ["", f"## Invariance checks (test images against their mirrors; forward precision "
                  f"{'fp32' if metrics['amp'] == 'none' else metrics['amp']})", ""]
    for v, c in metrics["invariance"].items():
        lines.append(f"- {v}: " + ", ".join(f"{k} {x:.2e}" for k, x in c.items()))
    lines += ["", "Files: probes.csv, probe/*.json, features/*.npz, metrics.json, meta.json."]
    path = out / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> dict:
    t0 = time.time()
    a = parse_args(argv)
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/core06" if a.tiny else "results/CORE-06")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.tiny else "cpu"))
    T2.apply_memory_cap(a.gpu_frac)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    grid = [float(x) for x in a.wd_grid.split(",") if x.strip()]
    solver = {"max_iter": a.max_iter}
    log(f"CORE-06: out {out}, device {device}, commit {git['git_hash'][:10]}")

    # ---- model, structure and data
    from mirror import mirror_family as MF

    if a.tiny:
        adapter, struct, root = tiny_runtime(out, device)
        struct_path = out / "structure" / "tiny_s0.npz"
        resize, crop, cache_dir = 64, 56, out / "tiny_cache"
        a.folds = min(a.folds, 2)
        model_name = "tiny"
    else:
        from mirror.adapters_timm import build_model_adapter

        adapter = build_model_adapter(a.model, device=device)
        adapter.model.float().eval()
        struct_path = REPO_ROOT / (a.structure or DEFAULT_STRUCTURE.format(model=a.model))
        struct = MF.load_span_structure(struct_path)
        if struct.model != a.model:
            raise SystemExit(f"{struct_path} is a structure of {struct.model}, not {a.model}")
        root = Path(a.cub_root or os.path.join(os.environ.get("HF_HOME", "/home/hf"), "cub", "CUB_200_2011"))
        resize, crop, cache_dir = a.resize, a.crop, Path(a.cache_dir)
        model_name = a.model
    if crop != adapter.img_size:
        raise SystemExit(f"crop {crop} differs from the model's input size {adapter.img_size}")
    ann = CUB.load_annotations(root)
    names_check = CUB.check_part_names(root)
    if names_check.get("ok") is False:
        raise SystemExit(f"parts/parts.txt does not name part 2 beak and part 14 tail: {names_check}")
    cmeta = CUB.build_cub_cache(root, cache_dir, resize, crop, workers=1 if a.tiny else a.decode_workers)
    images = CUB.load_cub_cache(cache_dir, crop)
    sign = CUB.facing_sign(ann["parts"], ann["bbox"][:, 2])
    train, test = np.flatnonzero(ann["is_train"]), np.flatnonzero(~ann["is_train"])
    num_classes = int(ann["label"].max()) + 1
    split = {"train": train, "test": test, "facing_sign": sign, "features_sha": {}}
    data = {"n_images": int(len(ann["name"])), "n_train": int(len(train)), "n_test": int(len(test)),
            "n_classes": num_classes, "n_facing_train": int((sign[train] != 0).sum()),
            "n_facing_test": int((sign[test] != 0).sum()),
            "facing_right_share_train": float((sign[train] > 0).sum() / max((sign[train] != 0).sum(), 1)),
            "facing_right_share_test": float((sign[test] > 0).sum() / max((sign[test] != 0).sum(), 1)),
            "annotation_sha256": ann["files"], "part_names_check": names_check, "cache": cmeta,
            "facing_convention": CUB.FACING_CONVENTION}
    log(f"CUB: {data['n_train']} train, {data['n_test']} test, facing labels {data['n_facing_train']} / "
        f"{data['n_facing_test']}")

    variants = build_variants(a, adapter, struct, struct_path, device)
    roles = {v[0]: v[1] for v in variants}
    struct_sha = T2.sha256(struct_path)
    all_loader = make_loader(images, ann["label"], np.arange(len(ann["label"])), a.batch_size,
                             0 if a.tiny else a.num_workers, device.type == "cuda")
    test_loader = make_loader(images, ann["label"], test, a.batch_size, 0 if a.tiny else a.num_workers,
                              device.type == "cuda")
    results, rows, invariance, var_info = {}, [], {}, {}
    for name, role, model, parts, info in variants:
        log(f"=== {name} ({role}); odd directions d- = {parts.d_minus}; operator: {parts.operator_source}")
        expect = {"variant": name, "model": model_name, "kind": parts.kind, "amp": a.amp, "crop": crop,
                  "structure_sha256": struct_sha, "cache": {k: cmeta.get(k) for k in ("n_images", "annotation_sha256")},
                  "sources": info.get("sources")}
        feats, checks = {}, {}
        for kind, loader, rws, mirrored in (("natural", all_loader, np.arange(len(ann["label"])), False),
                                            ("mirror", test_loader, test, True)):
            fp = feature_file(out, name, kind)
            hit = None if a.force_features else cached(fp, {**expect, "split": kind})
            if hit is not None:
                feats[kind], checks[kind] = hit[0], hit[1].get("first_batch_check")
                continue
            ts = time.time()
            X, chk = extract_parts(adapter, model, parts, loader, device, mirrored, a.amp)
            save_features(fp, X, rws, {**expect, "split": kind, "first_batch_check": chk,
                                       "seconds": round(time.time() - ts, 1), "git_hash": git["git_hash"]})
            feats[kind], checks[kind] = X, chk
            log(f"  {kind}: {X.shape} in {time.time() - ts:.1f} s; full-part check against the model's own feature "
                f"{chk:.1e}")
        split["features_sha"][name] = T2.sha256(feature_file(out, name, "natural"))[:16] + \
            T2.sha256(feature_file(out, name, "mirror"))[:16]
        invariance[name] = invariance_checks(feats["natural"][test], feats["mirror"])
        var_info[name] = {"role": role, **{k: v for k, v in info.items()}, "first_batch_checks": checks}
        results[name] = run_probes(a, out, name, feats["natural"], feats["mirror"], ann, split, grid, solver, device,
                                   num_classes, rows)
        if model is not adapter.model:
            del model
        T2.free_memory(device)

    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "probes.csv", index=False)
    scoring = score_predictions(results, roles)
    from mirror.meta import prereg_info

    metrics = {"experiment": EXPERIMENT, "scoring": scoring, "data": data, "invariance": invariance,
               "variants": var_info, "git": git, "prereg": prereg_info(), "seed": a.seed,
               "probe": {"wd_grid": grid, "folds": a.folds, "solver": solver,
                         "selection": "stratified k-fold CV on the CUB training split only"},
               "amp": a.amp, "structure": str(struct_path), "structure_sha256": struct_sha}
    T2.write_json(out / "metrics.json", metrics)
    write_summary(out, metrics, rows)
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"folds": a.seed, "global": a.seed},
               models={model_name: {"structure": str(struct_path)}, **{n: v.get("run") for n, v in var_info.items()}},
               device=device, wall_seconds=time.time() - t0,
               extra={"verdicts": {k: v["verdict"] for k, v in scoring.items()}, "data": data})
    for k in ("P6.1", "P6.2", "P6.3"):
        log(f"{k}: {scoring[k]['verdict']}")
    return {"out": str(out), "metrics": metrics, "rows": rows}


if __name__ == "__main__":
    main()

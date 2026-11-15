#!/usr/bin/env python
"""CORE-05: why surgery is cheap, and why plain SVD sometimes collapses (PROJECT_NEXT_PHASE.md CORE-05,
PREREG_NEXT_PHASE.md P5.1 to P5.3).

    python scripts/core05_why_cheap.py --stage all                          # every stage below, in a safe order
    python scripts/core05_why_cheap.py --stage measures --models dinov2_vitb14_reg,dinov2_vitl14_reg,deit3_b16,mae_b16
    python scripts/core05_why_cheap.py --stage truth --models dinov2_vitb14_reg
    python scripts/core05_why_cheap.py --stage freeze                       # design model only; writes rule_frozen.json
    python scripts/core05_why_cheap.py --stage truth --models dinov2_vitl14_reg,deit3_b16,mae_b16
    python scripts/core05_why_cheap.py --stage score                        # refuses without rule_frozen.json
    python scripts/core05_why_cheap.py --stage ckpt                         # CORE-01 LR and UD checkpoints
    python scripts/core05_why_cheap.py --stage report
    python scripts/core05_why_cheap.py --tiny --stage all --out tiny_out/core05     # tiny random ViTs, CPU

Structure: the canonical span structure of each model (results/MUST-07/structure/<model>_s0.npz, then
results/CORE-05/structure/, mirror.mirror_family.load_span_structure). Operators: "star" = A*_L, the full-model
Mirror-exact operator, on the input and output of every matrix (the main analysis); "star6" = A*_6 on blocks 1 to 6
(the 6-block span; reported); "perlayer" = A_b for fc1 and A_(b+1) for fc2 (as MUST-04 and F4; reported and used by the
F4-style rule). Hidden pairings: the fitted mirror twins; attention: the fitted head maps (projection in the original
gauge, mirror.attention.project_attn with gauge=False, which is the same orthogonal projection).

Stage measures (one fp32 pass over fit set 0, ImageNet train, 2,000 images; nothing from validation):
  input statistics per block: mean and covariance Sigma (and the uncentred second moment M) of the inputs of the
  folded fc1 (LayerNorm 2 without affine), of fc2 (the hidden activation), of qkv (LayerNorm 1 without affine) and of
  proj (attention output), over every token, float64. Per matrix W (folded: norm affine and LayerScale absorbed) and
  its projection W_eq:
    s_w    weight-energy breaking share ||W - W_eq||^2 / ||W||^2
    s_cov  activation-weighted breaking share ||(W - W_eq) Sigma^(1/2)||^2 / ||W Sigma^(1/2)||^2 (PREREG Section 1)
    s_mom  the same with M (the weighting of MUST-04 and F4)
    E_*    energy plain SVD discards at matched parameters: rank r = ceil(eq / (m + n)) for the matrix's equivariant
           parameter count eq (Section 4.7: round toward giving the baseline more), weight-only (E_w), the
           activation-weighted error of the plain rank-r truncation actually applied (E_cov_plain, E_mom_plain) and the
           activation-weighted energy outside the best rank-r approximation of W Sigma^(1/2) (E_cov_white, E_mom_white)
    bands  the eigenvectors of Sigma sorted by variance, split into the top 10%, the middle 80% and the bottom 10%
           of input directions: per band the breaking energy ||(W - W_eq) V_band||^2 (the three add up to
           ||W - W_eq||^2), its share of the total, the within-band breaking share and the band's variance share
    flatness  effective rank (exp of the entropy of the normalised singular values, mirror.twins.effective_rank,
           Roy and Vetterli 2007 (verify)) of W, of W Sigma^(1/2) and of Sigma^(1/2), divided by the largest possible
  -> measures_<model>.csv, spectra/<model>.npz
Stage truth (single-block surgery, the ground truth of the rule): for each block alone, Mirror (MLP projection with A*_L
and the twin pairing) against plain SVD of the folded fc1 and fc2 at matched parameters (rank as above; biases kept),
k-NN top-1 (paper protocol: 100k train bank, k = 20, T = 0.07, bf16 features) on the method-selection queries (10k
train images) and on validation (50k), plus the mean squared distance of the unit CLS feature from the original's. One
pass per group of variants caches the residual stream of the original model and runs each variant from its block on,
so the cost is about half that of full passes (the arithmetic is the same as a full forward).
  -> truth_<model>.csv, preds/<model>/*.npz
Stage freeze (P5.3 design, dinov2_vitb14_reg only; reads only that model's two files): the rule "Mirror beats SVD at
equal parameters when the activation-weighted breaking share is below the activation-weighted energy SVD discards",
per block, x = E - s (means of the fc1 and fc2 fractions, operator A*_L), Mirror predicted to win when x > 0. The
candidates fix what the plan leaves open: the weighting (Sigma or M) and the SVD energy (plain truncation applied, or
whitened). The candidate with the highest share of correct blocks against the method-selection ground truth
(y = SVD drop - Mirror drop on the 10k method-selection queries; never validation) is frozen, ties to the first in
RULE_CANDIDATES. A case with x = 0 or y = 0 is a miss (the F4 tie rule). Written once to rule_frozen.json, with the
hashes of the design files; a second freeze is refused unless --refreeze (logged as a deviation).
Stage score (P5.3): needs rule_frozen.json; applies it unchanged to dinov2_vitl14_reg, deit3_b16 and mae_b16, every
block, against validation drops (primary); pass if at least 75% of the pooled layer-model cases are correct. Also
reported: the method-selection and feature-distortion ground truths, the other candidates, the F4-style rule on the same
cases and the original F4 result (runs/f4/f4_p2.json when present).
Stage ckpt (P5.2): CORE-01 runs runs/CORE-01/vit_s8_{LR,UD}_s*, every checkpoint (mirror.controlled.load_checkpoint_
adapter). The trained group's per-layer operators and twin pairings come from the CORE-01 measurement cache
(results/CORE-01/<run>/e<NNN>/cache/pass_<G>_fit0.npz) or, if absent, from one paired pass here (mirror.group_stats).
Input covariances from the checkpoint's own fit set (fit seed 0). Band split of fc1 and fc2 in every block.
Stage report: P5.1, P5.2, P5.3 verdicts, the flatness table against plain SVD's zero-shot loss (results/MUST-07),
metrics.json, SUMMARY.md and meta.json.

Scoring (operational choices of this script, listed in the report for sign-off; PREREG_NEXT_PHASE.md gives only the
predictions):
  P5.1  dinov2_vitb14_reg, operator A*_L: mean of s_cov over fc1 and fc2 of blocks 1 to 6 divided by the mean of s_w
        over the same matrices is at most 0.5. Reported per model and for the other operators.
  P5.2  in every CORE-01 LR run (group LR) and UD run (group UD): the within-band breaking share of the top-variance
        band (mean over fc1 and fc2 of blocks 1 to 6) falls by more from the first to the last checkpoint than that of
        the bottom band (absolute fall; relative falls reported).
  P5.3  as stage score.
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.2) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import mirror_family as MF  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.attention import attention_free_params, fold_attn, project_attn  # noqa: E402
from mirror.data import make_loader, normalize_for  # noqa: E402
from mirror.m1 import involution_trace64  # noqa: E402
from mirror.meta import prereg_info, write_meta  # noqa: E402
from mirror.surgery import (equivariant_param_count, fold_mlp, perm_trace, project_mlp_two, svd_truncate,  # noqa: E402
                            write_absorbed_mlp)
from mirror.twins import effective_rank  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "CORE-05"
DESIGN_MODEL = "dinov2_vitb14_reg"
SCORE_MODELS = ("dinov2_vitl14_reg", "deit3_b16", "mae_b16")
ALL_MODELS = (DESIGN_MODEL,) + SCORE_MODELS
STAGES = ("measures", "truth", "freeze", "score", "ckpt", "report", "all")
BAND_FRACS = (0.1, 0.8, 0.1)
BANDS = ("top", "mid", "bottom")
P51_MAX_RATIO = 0.5
P53_MIN = 0.75
EARLY = range(6)                         # blocks 1 to 6 (code indices 0..5)
RULE_CANDIDATES = (
    {"id": "cov_plain", "weighting": "covariance Sigma", "svd_energy": "plain truncation applied",
     "s_col": "s_cov", "E_col": "E_cov_plain"},
    {"id": "cov_white", "weighting": "covariance Sigma", "svd_energy": "whitened (best rank-r of W Sigma^(1/2))",
     "s_col": "s_cov", "E_col": "E_cov_white"},
    {"id": "mom_plain", "weighting": "second moment M", "svd_energy": "plain truncation applied",
     "s_col": "s_mom", "E_col": "E_mom_plain"},
    {"id": "mom_white", "weighting": "second moment M", "svd_energy": "whitened (best rank-r of W M^(1/2))",
     "s_col": "s_mom", "E_col": "E_mom_white"},
)
RULE_COMMON = {"operator": "star", "operator_text": "A*_L (full-model Mirror-exact span operator), input and output",
               "matrices": ["fc1", "fc2"], "block_value": "mean of the fc1 and fc2 fractions",
               "prediction": "Mirror has the smaller single-block drop iff x = E - s > 0",
               "tie": "a case with x == 0 or y == 0 counts as a miss (as F4)",
               "truth_design": "msel", "truth_score": "val"}
F4_STYLE = {"id": "f4_style", "operator": "perlayer", "s_col": "s_mom", "E_col": "E_mom_white",
            "note": "the F4 rule's quantities (per-layer operators, second moments, whitened SVD energy), here with "
                    "fit-set moments, ceil ranks and the new single-block ground truth"}
TRUTH_COLS = {"val": "drop_val", "msel": "drop_msel", "distortion_val": "distortion_val"}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=STAGES)
    p.add_argument("--models", default=None, help=f"comma list (default {','.join(ALL_MODELS)}; tiny: tiny,tiny_s1)")
    p.add_argument("--design-model", default=None, help=f"P5.3 design model (default {DESIGN_MODEL})")
    p.add_argument("--score-models", default=None, help=f"P5.3 scoring models (default {','.join(SCORE_MODELS)})")
    p.add_argument("--structure-dirs", default="results/MUST-07/structure,results/CORE-05/structure",
                   help="searched in order for <model>_s<fit seed>.npz")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--blocks", default="all", help="all, or 1-based ranges such as 1-6 (measures and truth)")
    p.add_argument("--no-attn", action="store_true", help="skip attention statistics and shares")
    p.add_argument("--out", default=None, help="default results/CORE-05 (tiny: tiny_out/core05)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=128, help="truth: images per forward (residuals are cached)")
    p.add_argument("--stats-batch", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8, help="truth: single-block variants per data pass")
    p.add_argument("--bank-per-class", type=int, default=None, help="truth: bank subset (default: the 100k bank)")
    p.add_argument("--queries", type=int, default=None, help="truth: validation subset, a multiple of 1000")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"], help="truth: forward precision (paper: bf16)")
    p.add_argument("--refreeze", action="store_true", help="overwrite rule_frozen.json (a logged deviation)")
    p.add_argument("--ckpt-runs", default="runs/CORE-01/vit_s8_LR_s*,runs/CORE-01/vit_s8_UD_s*",
                   help="comma list of run folders (globs allowed)")
    p.add_argument("--ckpt-measure-dir", default="results/CORE-01", help="measure_groups output root of CORE-01")
    p.add_argument("--ckpt-fit-images", type=int, default=2000)
    p.add_argument("--ckpt-cache-dir", default=None, help="ImageNet-100 cache (default mirror.controlled's)")
    p.add_argument("--top-k", type=int, default=10, help="matching candidates when a checkpoint pass is run here")
    p.add_argument("--workers", type=int, default=8, help="processes for those matchings")
    p.add_argument("--must07-dir", default="results/MUST-07", help="rows_<model>.csv with plain SVD's zero-shot loss")
    p.add_argument("--f4-json", default="runs/f4/f4_p2.json", help="the earlier F4 result, reported next to P5.3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-frac", type=float, default=0.2, help="per-process GPU memory cap (read at import)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViTs, synthetic data, CPU")
    return p.parse_args(argv)


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


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
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def free(device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def parse_blocks(spec: str, L: int) -> list:
    if str(spec).strip().lower() == "all":
        return list(range(L))
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(b - 1 for b in out if 1 <= b <= L)


def svd_rank(eq: int, m: int, n: int) -> int:
    """Rank whose factorisation r (m + n) first reaches eq parameters (rounded toward giving SVD more, Section 4.7)."""
    return int(min(math.ceil(eq / (m + n)), min(m, n)))


# ----------------------------------------------------------------------------------------------
# Pure measures (unit-tested)
# ----------------------------------------------------------------------------------------------
def band_slices(n: int, fracs=BAND_FRACS) -> dict:
    """Input directions 0..n-1 sorted by variance (descending) split into top, middle and bottom bands."""
    n_top = max(1, int(round(fracs[0] * n)))
    n_bot = max(1, int(round(fracs[2] * n)))
    if n_top + n_bot >= n:
        n_top, n_bot = max(1, n // 3), max(1, n // 3)
    return {"top": slice(0, n_top), "mid": slice(n_top, n - n_bot), "bottom": slice(n - n_bot, n)}


def sorted_eig(Sigma: torch.Tensor) -> tuple:
    """Eigenvalues (descending, clamped at 0) and eigenvectors (columns) of a symmetric matrix, float64."""
    S = Sigma.to(torch.float64)
    lam, V_ = torch.linalg.eigh(0.5 * (S + S.T))
    return lam.flip(0).clamp_min(0), V_.flip(1)


def band_split(W: torch.Tensor, W_eq: torch.Tensor, lam: torch.Tensor, Vec: torch.Tensor, fracs=BAND_FRACS) -> dict:
    """Breaking energy of W - W_eq along the input directions Vec (columns, sorted by variance lam, descending),
    split into the top, middle and bottom bands. The three breaking energies add up to ||W - W_eq||^2."""
    W = W.to(torch.float64)
    D = W - W_eq.to(W)
    e = ((D @ Vec) ** 2).sum(0)
    w = ((W @ Vec) ** 2).sum(0)
    tot_e, tot_w, tot_l = float(e.sum()), float(w.sum()), float(lam.sum())
    out = {"breaking_energy_total": tot_e, "weight_energy_total": tot_w,
           "s_cov_eig": float((lam * e).sum() / (lam * w).sum()) if float((lam * w).sum()) > 0 else float("nan")}
    for name, sl in band_slices(int(lam.numel()), fracs).items():
        eb, wb, lb = float(e[sl].sum()), float(w[sl].sum()), float(lam[sl].sum())
        out.update({f"{name}_n": int(sl.stop - sl.start), f"{name}_breaking_energy": eb,
                    f"{name}_share_of_breaking": eb / tot_e if tot_e > 0 else float("nan"),
                    f"{name}_breaking_share": eb / wb if wb > 0 else float("nan"),
                    f"{name}_breaking_over_weight": eb / tot_w if tot_w > 0 else float("nan"),
                    f"{name}_variance_share": lb / tot_l if tot_l > 0 else float("nan")})
    return out


def matrix_measures(W: torch.Tensor, W_eq: torch.Tensor, Sigma: torch.Tensor, M: torch.Tensor, eq: int,
                    bands: bool = True, spectra: bool = True) -> dict:
    """Breaking shares, SVD energies at matched parameters, bands and effective ranks of one matrix W [m, n] whose
    input has covariance Sigma and second moment M (n x n)."""
    W = W.to(torch.float64)
    W_eq = W_eq.to(W)
    Sigma, M = Sigma.to(W), M.to(W)
    m, n = W.shape
    r = svd_rank(int(eq), m, n)
    Wr = svd_truncate(W, r)
    out = {"m": m, "n": n, "eq_params": int(eq), "svd_rank": r, "svd_params": int(r * (m + n)),
           "svd_gap_rel": (r * (m + n) - eq) / eq if eq else float("nan"),
           "s_w": V.weighted_share(W, W_eq), "s_cov": V.weighted_share(W, W_eq, Sigma),
           "s_mom": V.weighted_share(W, W_eq, M),
           "E_w": V.svd_discarded_energy(W, r), "E_cov_white": V.svd_discarded_energy(W, r, Sigma),
           "E_mom_white": V.svd_discarded_energy(W, r, M), "E_cov_plain": V.weighted_share(W, Wr, Sigma),
           "E_mom_plain": V.weighted_share(W, Wr, M)}
    if bands or spectra:
        lam, Vec = sorted_eig(Sigma)
        if bands:
            out.update(band_split(W, W_eq, lam, Vec))
        if spectra:
            sw = torch.linalg.svdvals(W).cpu().numpy()
            sa = V.weighted_spectrum(W, Sigma).clamp_min(0).sqrt().cpu().numpy()
            si = lam.sqrt().cpu().numpy()
            k = min(m, n)
            out.update({"erank_w": effective_rank(sw) / k, "erank_w_energy": effective_rank(sw, "energy") / k,
                        "erank_act": effective_rank(sa) / k, "erank_input": effective_rank(si) / n,
                        "_spectrum_w": sw, "_spectrum_input": lam.cpu().numpy()})
    return out


def rule_hits(x, y) -> np.ndarray:
    """F4's convention: correct when sign(x) == sign(y) and both are non-zero."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (np.sign(x) == np.sign(y)) & (x != 0) & (y != 0)


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):                      # average ties
            idx = v == u
            r[idx] = r[idx].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------------------------
# Input statistics
# ----------------------------------------------------------------------------------------------
class InputStats:
    """Per block: sum and sum of outer products (float64) of the inputs of fc1 (LayerNorm 2 without affine, the
    folded fc1's input), fc2 (hidden activation), qkv (LayerNorm 1 without affine) and proj (attention output), over
    every token of every image."""

    def __init__(self, model: nn.Module, blocks, device, attn: bool = True):
        self.blocks = list(blocks)
        self.slots = ("fc1", "fc2", "qkv", "proj") if attn else ("fc1", "fc2")
        self.S1, self.S2, self.n = {}, {}, {}
        self.handles = []
        kw = dict(dtype=torch.float64, device=device)
        for b in self.blocks:
            blk = model.blocks[b]
            dims = {"fc1": blk.mlp.fc1.in_features, "fc2": blk.mlp.fc2.in_features}
            if attn:
                dims.update({"qkv": blk.attn.qkv.in_features, "proj": blk.attn.proj.in_features})
            for s in self.slots:
                self.S1[(s, b)] = torch.zeros(dims[s], **kw)
                self.S2[(s, b)] = torch.zeros(dims[s], dims[s], **kw)
                self.n[(s, b)] = 0
            for norm, slot in ((blk.norm2, "fc1"), (blk.norm1, "qkv")):
                if slot not in self.slots:
                    continue
                if not isinstance(norm, nn.LayerNorm):
                    raise ValueError(f"block {b}: {type(norm).__name__} before {slot}; expected nn.LayerNorm")
                self.handles.append(norm.register_forward_pre_hook(self._ln_hook(slot, b, norm.eps)))
            self.handles.append(blk.mlp.fc2.register_forward_pre_hook(self._hook("fc2", b)))
            if attn:
                self.handles.append(blk.attn.proj.register_forward_pre_hook(self._hook("proj", b)))

    def _add(self, key, X: torch.Tensor) -> None:
        X = X.reshape(-1, X.shape[-1]).to(torch.float64)
        self.S1[key] += X.sum(0)
        self.S2[key] += X.T @ X
        self.n[key] += int(X.shape[0])

    def _ln_hook(self, slot, b, eps):
        def fn(mod, args):
            x = args[0]
            self._add((slot, b), F.layer_norm(x.to(torch.float64), (x.shape[-1],), None, None, eps))
        return fn

    def _hook(self, slot, b):
        def fn(mod, args):
            self._add((slot, b), args[0])
        return fn

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self, slot: str, b: int) -> dict:
        """mean, covariance Sigma (centred) and second moment M, all divided by the token count."""
        n = self.n[(slot, b)]
        mu = self.S1[(slot, b)] / n
        M = self.S2[(slot, b)] / n
        M = 0.5 * (M + M.T)
        return {"mean": mu, "Sigma": M - torch.outer(mu, mu), "M": M, "n": n}


@torch.no_grad()
def collect_stats(adapter, loader, device, blocks, attn: bool = True) -> tuple:
    """One fp32 pass over `loader`; returns (InputStats, number of images)."""
    st = InputStats(adapter.model, blocks, device, attn)
    n_img = 0
    try:
        for imgs, _ in loader:
            x = normalize_for(adapter, imgs.to(device, non_blocking=True))
            adapter.forward_tokens(x)
            n_img += int(x.shape[0])
    finally:
        st.remove()
    return st, n_img


# ----------------------------------------------------------------------------------------------
# Runtime and structure
# ----------------------------------------------------------------------------------------------
def runtime(model_name: str, a, device) -> tuple:
    """(adapter, protocol, num_classes). Tiny names: "tiny" (seed 0) and "tiny_s<k>" (seed k)."""
    if a.tiny:
        from mirror.adapters import TimmTinyAdapter

        m = re.fullmatch(r"tiny(?:_s(\d+))?", model_name)
        if not m:
            raise ValueError(f"tiny model names are tiny or tiny_s<k>, not {model_name!r}")
        adapter = TimmTinyAdapter(img_size=56, seed=int(m.group(1) or 0), randomise=True).to(device=device)
        proto = V.TinyProtocol()
        return adapter, proto, proto.num_classes
    adapter, proto, C = V.build_runtime(model_name, False, device, a.cache_dir, a.eval_batch, a.stats_batch,
                                        a.num_workers)
    adapter.model.float().eval()
    return adapter, proto, C


def get_structure(model_name: str, adapter, proto, device, a, out: Path):
    for d in [x.strip() for x in a.structure_dirs.split(",") if x.strip()] + [str(out / "structure")]:
        p = _resolve(d) / f"{model_name}_s{a.fit_seed}.npz"
        if p.exists():
            st = MF.load_span_structure(p)
            if not a.tiny and st.model != model_name:
                raise SystemExit(f"{p} is a structure of {st.model}, not {model_name}")
            if st.depth != adapter.depth:
                raise SystemExit(f"{p} has {st.depth} blocks; the model has {adapter.depth}")
            return st, p
    if not a.tiny:
        raise SystemExit(f"no structure {model_name}_s{a.fit_seed}.npz in {a.structure_dirs}: run "
                         f"python scripts/must07_compressors.py --model {model_name} --fit-seeds {a.fit_seed} "
                         f"--structure-only --structure-dir results/CORE-05/structure")
    p = out / "structure" / f"{model_name}_s{a.fit_seed}.npz"
    st = MF.fit_span_structure(adapter, proto.fit_loader(a.fit_seed), device, model_name=model_name,
                               fit_seed=a.fit_seed, top_k=3, workers=1)
    MF.save_span_structure(p, st)
    return MF.load_span_structure(p), p


def ops_for(struct, L: int) -> dict:
    """operator name -> (blocks it applies to, function b -> (A_in, A_out))."""
    out = {"star": (list(range(L)), lambda b: (struct.A_star[L], struct.A_star[L])),
           "perlayer": (list(range(L)), lambda b: (struct.A_layers[b], struct.A_layers[b + 1]))}
    if L > 6:
        out["star6"] = (list(range(6)), lambda b: (struct.A_star[6], struct.A_star[6]))
    return out


def models_of(a) -> list:
    if a.models:
        return [m.strip() for m in a.models.split(",") if m.strip()]
    return ["tiny", "tiny_s1"] if a.tiny else list(ALL_MODELS)


def design_of(a) -> str:
    return a.design_model or ("tiny" if a.tiny else DESIGN_MODEL)


def scoring_of(a) -> list:
    if a.score_models:
        return [m.strip() for m in a.score_models.split(",") if m.strip()]
    return ["tiny_s1"] if a.tiny else list(SCORE_MODELS)


# ----------------------------------------------------------------------------------------------
# Stage: measures
# ----------------------------------------------------------------------------------------------
def stage_measures(a, out: Path, device, git: dict) -> None:
    for model_name in models_of(a):
        path = out / f"measures_{model_name}.csv"
        adapter, proto, _ = runtime(model_name, a, device)
        L = adapter.depth
        blocks = parse_blocks(a.blocks, L)
        struct, spath = get_structure(model_name, adapter, proto, device, a, out)
        ops = ops_for(struct, L)
        want = sum(len([b for b in ob if b in blocks]) for ob, _ in ops.values()) * 2 + (0 if a.no_attn else 4 * len(blocks))
        if path.exists():
            import pandas as pd

            if len(pd.read_csv(path)) >= want:
                log(f"{model_name}: measures done ({path})")
                del adapter
                free(device)
                continue
        t0 = time.time()
        st, n_img = collect_stats(adapter, proto.fit_loader(a.fit_seed), device, blocks, attn=not a.no_attn)
        log(f"{model_name}: input statistics over {n_img} fit images in {time.time() - t0:.1f} s")
        rows, spectra = [], {}
        d = adapter.embed_dim
        for op, (oblocks, fn) in ops.items():
            for b in [x for x in oblocks if x in blocks]:
                A_in, A_out = fn(b)
                blk = adapter.model.blocks[b]
                f = fold_mlp(blk)
                e = project_mlp_two(**f, A_in=A_in, A_out=A_out, perm=struct.perms[b])
                h = f["W1"].shape[0]
                trP = perm_trace(struct.perms[b])
                eq1 = equivariant_param_count(h, d, trP, involution_trace64(A_in))
                eq2 = equivariant_param_count(h, d, trP, involution_trace64(A_out))
                mats = [("fc1", f["W1"], e["W1"], st.get("fc1", b), eq1), ("fc2", f["W2"], e["W2"], st.get("fc2", b), eq2)]
                if op == "star" and not a.no_attn:
                    fa = fold_attn(blk)
                    ea = project_attn(**fa, A_in=A_in, A_out=A_out, head_maps=struct.heads[b], gauge=False)
                    fr = attention_free_params(d, struct.heads[b], A_in, A_out)
                    q = st.get("qkv", b)
                    mats += [("q", fa["Wq"], ea["Wq"], q, fr["Wq"]), ("k", fa["Wk"], ea["Wk"], q, fr["Wk"]),
                             ("v", fa["Wv"], ea["Wv"], q, fr["Wv"]), ("o", fa["Wo"], ea["Wo"], st.get("proj", b), fr["Wo"])]
                for name, W, We, s, eq in mats:
                    full = op == "star"
                    mm = matrix_measures(W, We, s["Sigma"], s["M"], eq, bands=full, spectra=full)
                    if full and name in ("fc1", "fc2"):
                        spectra[f"b{b + 1}_{name}_w"] = mm["_spectrum_w"]
                        spectra[f"b{b + 1}_{name}_input"] = mm["_spectrum_input"]
                    rows.append({"model": model_name, "op": op, "block": b + 1, "matrix": name,
                                 "d_minus": int((A_in.shape[0] - involution_trace64(A_in)) // 2),
                                 **{k: v for k, v in mm.items() if not k.startswith("_")},
                                 "n_tokens": s["n"], "fit_seed": a.fit_seed, "structure": str(spath),
                                 "git_hash": git["git_hash"]})
            log(f"{model_name}: operator {op} measured")
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)
        sp = out / "spectra" / f"{model_name}.npz"
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sp, **spectra)
        log(f"{model_name}: {len(rows)} rows -> {path} ({time.time() - t0:.1f} s)")
        del st, adapter
        free(device)


# ----------------------------------------------------------------------------------------------
# Stage: truth (single-block surgery)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def single_block_variants(adapter, struct, blocks) -> dict:
    """name -> (block, converted copy of that block, record): Mirror (MLP projection with A*_L, A*_L and the twin
    pairing) and plain SVD of the folded fc1 and fc2 at matched parameters, biases kept."""
    L, d = adapter.depth, adapter.embed_dim
    A = struct.A_star[L]
    out = {}
    for b in blocks:
        blk = adapter.model.blocks[b]
        f = fold_mlp(blk)
        h = f["W1"].shape[0]
        eq = equivariant_param_count(h, d, perm_trace(struct.perms[b]), involution_trace64(A))
        r = svd_rank(eq, h, d)
        mb = copy.deepcopy(blk)
        write_absorbed_mlp(mb, **project_mlp_two(**fold_mlp(mb), A_in=A, A_out=A, perm=struct.perms[b]))
        sb = copy.deepcopy(blk)
        fs = fold_mlp(sb)
        write_absorbed_mlp(sb, svd_truncate(fs["W1"], r), fs["b1"], svd_truncate(fs["W2"], r), fs["b2"])
        out[f"b{b + 1}_mirror"] = (b, mb.eval(), {"block": b + 1, "method": "mirror", "params": 2 * eq, "rank": None})
        out[f"b{b + 1}_svd"] = (b, sb.eval(), {"block": b + 1, "method": "svd", "params": 2 * r * (h + d), "rank": r,
                                               "mirror_params": 2 * eq})
    return out


class ResidualEvaluator:
    """Unit CLS features of the original model and of single-block variants from one forward of the original:
    the residual stream entering each block is kept, and each variant runs its converted block and the original
    blocks after it from there. The same operations as a full forward of the variant."""

    def __init__(self, adapter, variants: dict):
        self.ad, self.variants = adapter, variants

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        m = self.ad.model
        t = self.ad.layer0_tokens(x, m)
        res = []
        for blk in m.blocks:
            res.append(t)
            t = blk(t)
        out = {"original": F.normalize(m.norm(t)[:, 0].float(), dim=-1)}
        for name, (b, mod, _) in self.variants.items():
            u = mod(res[b])
            for j in range(b + 1, len(m.blocks)):
                u = m.blocks[j](u)
            out[name] = F.normalize(m.norm(u)[:, 0].float(), dim=-1)
        return out


def bank_loader(proto, per_class, a):
    if per_class is None:
        return proto.bank_loader()
    rows = np.asarray(proto.index["bank_rows"])
    lab = proto.train_labels[rows]
    keep = np.sort(np.concatenate([rows[lab == c][:per_class] for c in np.unique(lab)]))
    return make_loader(proto.train, proto.train_labels, keep, a.eval_batch, a.num_workers, False)


@torch.no_grad()
def eval_group(adapter, group: dict, loaders: dict, device, amp: str) -> tuple:
    """Features of every image of every loader for the original and the group's variants: ({name: {role: X}},
    {role: labels})."""
    dev = torch.device(device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and dev.type == "cuda"
           else contextlib.nullcontext())
    ev = ResidualEvaluator(adapter, group)
    feats = {n: {} for n in ["original"] + list(group)}
    labels = {}
    d = adapter.embed_dim
    for role, loader in loaders.items():
        N = len(loader.dataset)
        for n in feats:
            feats[n][role] = np.empty((N, d), dtype=np.float32)
        labels[role] = np.empty(N, dtype=np.int64)
        pos = 0
        for imgs, y in loader:
            x = normalize_for(adapter, imgs.to(dev, non_blocking=True))
            with ctx:
                f = ev(x)
            B = x.shape[0]
            for n, v in f.items():
                feats[n][role][pos:pos + B] = v.cpu().numpy()
            labels[role][pos:pos + B] = np.asarray(y)
            pos += B
    return feats, labels


@torch.no_grad()
def knn_scores(feats: dict, labels: dict, num_classes: int, device, k: int, T: float, ref: dict | None) -> dict:
    from mirror.knn import knn_predict

    dev = torch.device(device)
    bank = torch.as_tensor(feats["bank"], device=dev)
    by = torch.as_tensor(labels["bank"], device=dev)
    out = {}
    for q in ("val", "msel"):
        qf = torch.as_tensor(feats[q], device=dev)
        pred = knn_predict(bank, by, qf, num_classes, k, T).cpu().numpy()
        out[f"top1_{q}"] = 100.0 * float((pred == labels[q]).mean())
        out[f"pred_{q}"] = pred
        if ref is not None:
            out[f"distortion_{q}"] = float(((feats[q] - ref[q]) ** 2).sum(1).mean())
    return out


def stage_truth(a, out: Path, device, git: dict, models=None) -> None:
    import pandas as pd

    for model_name in models or models_of(a):
        path = out / f"truth_{model_name}.csv"
        csv = V.ResultsCsv(path, ["model", "block", "method"])
        adapter, proto, C = runtime(model_name, a, device)
        L = adapter.depth
        blocks = parse_blocks(a.blocks, L)
        todo = [b for b in blocks for mth in ("mirror", "svd") if not csv.done((model_name, b + 1, mth))]
        if not todo and csv.done((model_name, 0, "original")):
            log(f"{model_name}: single-block truth done ({path})")
            continue
        struct, spath = get_structure(model_name, adapter, proto, device, a, out)
        todo_blocks = sorted(set(todo))
        loaders = {"bank": bank_loader(proto, a.bank_per_class, a), "val": proto.query_loader(a.queries),
                   "msel": proto.msel_loader()}
        orig = None
        if csv.done((model_name, 0, "original")):
            r0 = [r for r in csv.rows if int(r["block"]) == 0 and r["method"] == "original"][0]
            orig = {"top1_val": float(r0["top1_val"]), "top1_msel": float(r0["top1_msel"])}
        groups = [todo_blocks[i:i + max(1, a.group_size // 2)] for i in range(0, len(todo_blocks), max(1, a.group_size // 2))] \
            or [[]]
        for gb in groups:
            t0 = time.time()
            group = single_block_variants(adapter, struct, gb)
            group = {n: v for n, v in group.items() if not csv.done((model_name, v[2]["block"], v[2]["method"]))}
            for n, (b, mod, _) in group.items():
                mod.to(device)
            feats, labels = eval_group(adapter, group, loaders, device, a.amp)
            ks = knn_scores(feats["original"], labels, C, device, a.knn_k, a.knn_temperature, None)
            if orig is None:
                orig = ks
                save_preds(out, model_name, "original", ks)
                csv.add({"model": model_name, "block": 0, "method": "original", "params": None, "rank": None,
                         "top1_val": ks["top1_val"], "top1_msel": ks["top1_msel"], "drop_val": 0.0, "drop_msel": 0.0,
                         "distortion_val": 0.0, "distortion_msel": 0.0, "n_bank": len(labels["bank"]),
                         "n_val": len(labels["val"]), "n_msel": len(labels["msel"]), "amp": a.amp,
                         "structure": str(spath), "git_hash": git["git_hash"], "seed": a.seed})
            for n, (b, mod, rec) in group.items():
                s = knn_scores(feats[n], labels, C, device, a.knn_k, a.knn_temperature, feats["original"])
                save_preds(out, model_name, n, s)
                csv.add({"model": model_name, "block": rec["block"], "method": rec["method"], "params": rec["params"],
                         "rank": rec["rank"], "mirror_params": rec.get("mirror_params", rec["params"]),
                         "top1_val": s["top1_val"], "top1_msel": s["top1_msel"],
                         "drop_val": orig["top1_val"] - s["top1_val"], "drop_msel": orig["top1_msel"] - s["top1_msel"],
                         "distortion_val": s["distortion_val"], "distortion_msel": s["distortion_msel"],
                         "n_bank": len(labels["bank"]), "n_val": len(labels["val"]), "n_msel": len(labels["msel"]),
                         "amp": a.amp, "structure": str(spath), "git_hash": git["git_hash"], "seed": a.seed})
            log(f"{model_name}: blocks {[b + 1 for b in gb]} done in {time.time() - t0:.1f} s")
            del group, feats
            free(device)
        del adapter
        free(device)
        log(f"{model_name}: truth rows in {path} ({len(pd.read_csv(path))})")


def save_preds(out: Path, model_name: str, name: str, s: dict) -> None:
    p = out / "preds" / model_name / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, val=s["pred_val"].astype(np.int16), msel=s["pred_msel"].astype(np.int16))


# ----------------------------------------------------------------------------------------------
# Rule: cases, freeze and score
# ----------------------------------------------------------------------------------------------
def rule_cases(measures, truth, rule: dict, truth_key: str):
    """Per block: x = E - s (rule's columns, mean over fc1 and fc2, rule's operator), y = SVD minus Mirror
    (drop or distortion: positive means Mirror did better), hit."""
    import pandas as pd

    op = rule.get("operator", RULE_COMMON["operator"])
    m = measures[(measures["op"] == op) & (measures["matrix"].isin(["fc1", "fc2"]))]
    g = m.groupby(["model", "block"]).agg(s=(rule["s_col"], "mean"), E=(rule["E_col"], "mean")).reset_index()
    col = TRUTH_COLS[truth_key]
    t = truth[truth["method"].isin(["mirror", "svd"])].pivot_table(index=["model", "block"], columns="method",
                                                                  values=col).reset_index()
    if "mirror" not in t or "svd" not in t:
        return pd.DataFrame(columns=["model", "block", "s", "E", "x", "y", "hit"])
    t["y"] = t["svd"] - t["mirror"]
    df = g.merge(t[["model", "block", "y"]], on=["model", "block"], how="inner")
    df["x"] = df["E"] - df["s"]
    df["hit"] = rule_hits(df["x"], df["y"])
    return df


def _frac(df) -> dict:
    n = int(len(df))
    return {"n_cases": n, "n_hits": int(df["hit"].sum()) if n else 0,
            "fraction": float(df["hit"].mean()) if n else float("nan")}


def load_model_tables(out: Path, models) -> tuple:
    import pandas as pd

    ms, ts, files = [], [], {}
    for m in models:
        for kind, lst in (("measures", ms), ("truth", ts)):
            p = out / f"{kind}_{m}.csv"
            if p.exists():
                lst.append(pd.read_csv(p))
                files[str(p.relative_to(out))] = sha256_file(p)
    cat = (lambda x: pd.concat(x, ignore_index=True) if x else None)
    return cat(ms), cat(ts), files


def stage_freeze(a, out: Path, git: dict) -> dict:
    """Choose and freeze the rule on the design model only (reads only its measures and truth files)."""
    path = out / "rule_frozen.json"
    if path.exists() and not a.refreeze:
        log(f"{path} exists: the rule is frozen; not rewritten (use --refreeze, a logged deviation)")
        return json.loads(path.read_text(encoding="utf-8"))
    design = design_of(a)
    measures, truth, files = load_model_tables(out, [design])
    if measures is None or truth is None:
        raise SystemExit(f"freeze needs measures_{design}.csv and truth_{design}.csv in {out}")
    L = int(measures["block"].max())
    cands = []
    for c in RULE_CANDIDATES:
        df = rule_cases(measures, truth, c, RULE_COMMON["truth_design"])
        cands.append({**c, "design_" + RULE_COMMON["truth_design"]: _frac(df),
                      "design_cases": df[["block", "s", "E", "x", "y", "hit"]].to_dict("records")})
    complete = all(c["design_msel"]["n_cases"] == L for c in cands)
    if not all(c["design_msel"]["n_cases"] > 0 for c in cands):
        raise SystemExit(f"no design cases for {design}: the measures and truth files share no block")
    if not complete:
        raise SystemExit(f"the design model has {cands[0]['design_msel']['n_cases']} of {L} blocks with both measures "
                         "and single-block truth; finish them before freezing the rule")
    best = max(range(len(cands)), key=lambda i: (round(cands[i]["design_msel"]["fraction"], 12), -i))
    if path.exists():
        old = path.with_name(f"rule_frozen.replaced_{time.strftime('%Y%m%dT%H%M%S')}.json")
        os.replace(path, old)
        log(f"--refreeze: the previous rule is kept as {old}; record this deviation in notes/decisions.md")
    rule = {**RULE_COMMON, **{k: cands[best][k] for k in ("id", "weighting", "svd_energy", "s_col", "E_col")}}
    rec = {"experiment": EXPERIMENT, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": rule, "rule_sha256": hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest(),
           "selection": "highest share of correct blocks on the design model against the method-selection ground "
                        "truth (y = SVD drop - Mirror drop, 10k method-selection queries); ties to the first candidate",
           "design_model": design, "design_complete": complete, "design_files_sha256": files,
           "candidates": cands, "scoring": {"models": scoring_of(a), "truth": "validation drops (primary)",
                                            "threshold": P53_MIN, "aggregation": "pooled over every layer-model case"},
           "git": git, "prereg": prereg_info(), "refreeze": bool(a.refreeze)}
    write_json(path, rec)
    log(f"rule frozen: {rule['id']} ({cands[best]['design_msel']['n_hits']} of {cands[best]['design_msel']['n_cases']} "
        f"on {design}) -> {path}")
    return rec


def stage_score(a, out: Path, git: dict) -> dict:
    path = out / "rule_frozen.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --stage freeze on the design model first (the rule must be "
                         "frozen before scoring)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    rule = frozen["rule"]
    if hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest() != frozen["rule_sha256"]:
        raise SystemExit(f"{path}: the rule does not match its recorded sha256 (edited after freezing?)")
    models = frozen["scoring"]["models"]
    measures, truth, files = load_model_tables(out, models)
    res = {"experiment": EXPERIMENT, "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule_frozen_sha256": sha256_file(path), "rule": rule, "rule_frozen_utc": frozen["frozen_utc"],
           "scoring_files_sha256": files, "threshold": P53_MIN, "git": git}
    if measures is None or truth is None:
        res.update({"verdict": "INCOMPLETE (no scoring data)"})
        write_json(out / "rule_score.json", res)
        return res
    depth = measures.groupby("model")["block"].max().to_dict()
    df = rule_cases(measures, truth, rule, RULE_COMMON["truth_score"])
    complete = all(m in depth and len(df[df["model"] == m]) == depth[m] for m in models)
    primary = _frac(df)
    verdict = ("PASS" if primary["fraction"] >= P53_MIN else "FAIL") if complete else \
        f"INCOMPLETE ({sum(m in depth and len(df[df['model'] == m]) == depth[m] for m in models)} of {len(models)} models)"
    res.update({"primary": primary, "verdict": verdict, "complete": complete,
                "per_model": {m: _frac(df[df["model"] == m]) for m in models},
                "cases": df.to_dict("records"),
                "secondary": {
                    "msel_truth": _frac(rule_cases(measures, truth, rule, "msel")),
                    "feature_distortion_truth": _frac(rule_cases(measures, truth, rule, "distortion_val")),
                    "other_candidates_val": {c["id"]: _frac(rule_cases(measures, truth, c, "val"))
                                             for c in RULE_CANDIDATES if c["id"] != rule["id"]},
                    "f4_style_rule_val": {**_frac(rule_cases(measures, truth, F4_STYLE, "val")), "note": F4_STYLE["note"]},
                }})
    f4 = _resolve(a.f4_json)
    if f4.exists():
        j = json.loads(f4.read_text(encoding="utf-8"))
        res["f4_original"] = {"file": str(f4), "verdict": j.get("verdict"),
                              "primary_prereg_models": j.get("primary_prereg_models"), "threshold": j.get("threshold"),
                              "hypothesis": j.get("hypothesis")}
    else:
        res["f4_original"] = {"file": str(f4), "note": "not found here; the F4 result is in results/STATUS.md (P2)"}
    write_json(out / "rule_score.json", res)
    log(f"P5.3: {verdict}: {primary['n_hits']} of {primary['n_cases']} ({primary['fraction']:.3f})")
    return res


# ----------------------------------------------------------------------------------------------
# Stage: ckpt (CORE-01 checkpoints, P5.2)
# ----------------------------------------------------------------------------------------------
RUN_RE = re.compile(r".*_(none|LR|UD|R90|LRUD)_s(\d+)$")


def ckpt_operators(a, run: Path, ep: str, adapter, group: str, device, out: Path, fit_loader_fn) -> dict:
    """{"A": [A_0..A_L], "perm": [L perms], "source"} for one checkpoint and group (LR or UD)."""
    from mirror import groups as GR

    mc = _resolve(a.ckpt_measure_dir) / run.name / ep / "cache" / f"pass_{group}_fit{a.fit_seed}.npz"
    own = out / "ckpt_cache" / run.name / f"{ep}_{group}.npz"
    L = adapter.depth
    if mc.exists():
        with np.load(mc) as z:
            meta = json.loads(str(z["meta_json"]))
            if meta.get("element") != group:
                raise ValueError(f"{mc}: element {meta.get('element')}, expected {group}")
            layers, blocks = [int(x) for x in z["layers"]], [int(x) for x in z["blocks"]]
            C, perms = z["res_C"], z["perm"]
        src = str(mc)
    elif own.exists():
        with np.load(own) as z:
            layers, blocks, C, perms = list(z["layers"]), list(z["blocks"]), z["C"], z["perm"]
        src = str(own)
    else:
        from mirror.group_stats import collect_group_pass
        from mirror.matching import match_many

        gp = collect_group_pass(adapter, fit_loader_fn(), device, group, list(range(L)), residual=True, mlp=True,
                                attn=False)
        ms = match_many([gp.R[b] for b in range(L)], top_k=a.top_k, workers=1 if a.tiny else a.workers)
        layers, blocks, C = list(gp.layers), list(range(L)), gp.residual["C"]
        perms = np.stack([m.perm for m in ms])
        own.parent.mkdir(parents=True, exist_ok=True)
        np.savez(own, layers=np.asarray(layers), blocks=np.asarray(blocks), C=C, perm=perms)
        src = f"pass run here ({own})"
    if layers != list(range(L + 1)) or blocks != list(range(L)):
        raise ValueError(f"{src}: needs every layer 0..{L} and block 1..{L}")
    return {"A": [GR.fit_order2(C[l])[0] for l in range(L + 1)], "perm": [np.asarray(p) for p in perms], "source": src}


def tiny_ckpt_runs(out: Path) -> tuple:
    """Two tiny controlled runs (LR and UD) with two checkpoints each and a synthetic ImageNet-100 cache."""
    import mirror.controlled as ctl

    root = out / "tiny_ctrl"
    cache = root / "cache"
    if not (cache / "in100_meta.json").exists():
        ctl.write_synthetic_cache(cache, num_classes=10, train_per_class=8, val_per_class=2, seed=0)
    runs = []
    for cond in ("LR", "UD"):
        rd = root / "runs" / f"vit_test8_{cond}_s0"
        rd.mkdir(parents=True, exist_ok=True)
        for ep in (0, 1):
            p = rd / ctl.checkpoint_name(ep)
            if not p.exists():
                torch.manual_seed(ep)
                m = ctl.build_model("vit_test8", num_classes=10)
                torch.save({"format": ctl.CKPT_FORMAT, "arch": "vit_test8", "condition": cond, "seed": 0, "epoch": ep,
                            "img_size": ctl.IMG_SIZE, "patch_size": ctl.PATCH_SIZE, "num_classes": 10,
                            "model": {k: v.detach().clone() for k, v in m.state_dict().items()}}, p)
        runs.append(rd)
    return runs, cache


def stage_ckpt(a, out: Path, device, git: dict) -> None:
    import mirror.controlled as ctl

    if a.tiny:
        runs, cache = tiny_ckpt_runs(out)
        n_fit = 20
    else:
        runs = sorted({Path(p) for spec in a.ckpt_runs.split(",") if spec.strip()
                       for p in glob.glob(str(_resolve(spec.strip())))})
        cache, n_fit = a.ckpt_cache_dir, a.ckpt_fit_images
    if not runs:
        log(f"no CORE-01 runs match {a.ckpt_runs}; ckpt stage skipped")
        return
    csv = V.ResultsCsv(out / "ckpt_bands.csv", ["run", "epoch", "block", "matrix"])
    for run in runs:
        mt = RUN_RE.match(run.name)
        if not mt or mt.group(1) not in ("LR", "UD"):
            log(f"{run.name}: not an LR or UD run; skipped")
            continue
        if not a.tiny and not (run / "last_done.flag").exists():
            log(f"{run.name}: training not finished (no last_done.flag); skipped, rerun this stage later")
            continue
        group, seed = mt.group(1), int(mt.group(2))
        for epoch, ck in ctl.list_checkpoints(run):
            if csv.done((run.name, epoch, 1, "fc1")):
                continue
            t0 = time.time()
            adapter = ctl.load_checkpoint_adapter(ck, device)
            adapter.model.float().eval()
            L = adapter.depth

            def fit_loader_fn():
                return ctl.fit_loader(a.fit_seed, n_fit, batch_size=64, cache_dir=cache)

            ops = ckpt_operators(a, run, f"e{epoch:03d}", adapter, group, device, out, fit_loader_fn)
            st, n_img = collect_stats(adapter, fit_loader_fn(), device, range(L), attn=False)
            rows = []
            for b in range(L):
                f = fold_mlp(adapter.model.blocks[b])
                e = project_mlp_two(**f, A_in=ops["A"][b], A_out=ops["A"][b + 1], perm=ops["perm"][b])
                for name, W, We, slot in (("fc1", f["W1"], e["W1"], "fc1"), ("fc2", f["W2"], e["W2"], "fc2")):
                    s = st.get(slot, b)
                    lam, Vec = sorted_eig(s["Sigma"])
                    bs = band_split(W, We, lam, Vec)
                    rows.append({"run": run.name, "condition": group, "seed": seed, "group": group, "epoch": epoch,
                                 "block": b + 1, "matrix": name, "s_w": V.weighted_share(W, We),
                                 "s_cov": V.weighted_share(W, We, s["Sigma"]), **bs, "n_tokens": s["n"],
                                 "operators": ops["source"], "git_hash": git["git_hash"]})
            for r in rows:
                csv.add(r)
            log(f"{run.name} epoch {epoch}: bands measured ({n_img} fit images, {time.time() - t0:.1f} s)")
            del adapter, st
            free(device)


# ----------------------------------------------------------------------------------------------
# Stage: report
# ----------------------------------------------------------------------------------------------
def p51(out: Path, models) -> dict:
    import pandas as pd

    per = {}
    for m in models:
        p = out / f"measures_{m}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        per[m] = {}
        for op in sorted(df["op"].unique()):
            sub = df[(df["op"] == op) & df["matrix"].isin(["fc1", "fc2"]) & df["block"].isin([b + 1 for b in EARLY])]
            if sub.empty:
                continue
            sw, sc, sm = sub["s_w"].mean(), sub["s_cov"].mean(), sub["s_mom"].mean()
            per[m][op] = {"s_w": sw, "s_cov": sc, "s_mom": sm, "ratio_cov": sc / sw, "ratio_mom": sm / sw,
                          "n_matrices": int(len(sub)), "pass": bool(sc / sw <= P51_MAX_RATIO)}
            attn = df[(df["op"] == op) & df["matrix"].isin(["q", "k", "v", "o"]) & df["block"].isin([b + 1 for b in EARLY])]
            if not attn.empty:
                per[m][op]["attention_reported"] = {"s_w": attn["s_w"].mean(), "s_cov": attn["s_cov"].mean()}
    main = (per.get(DESIGN_MODEL) or per.get("tiny") or {}).get("star")
    return {"prediction": "the activation-weighted breaking share is at most half the weight-energy breaking share in "
                          "blocks 1 to 6",
            "operational": "dinov2_vitb14_reg, operator A*_L, mean s_cov over fc1 and fc2 of blocks 1 to 6 divided by "
                           f"the mean s_w, at most {P51_MAX_RATIO}; other models and operators reported",
            "per_model": per, "verdict": "n/a" if main is None else ("PASS" if main["pass"] else "FAIL"),
            "all_models_star": None if not per else bool(all(v.get("star", {}).get("pass", False) for v in per.values()))}


def p52(out: Path) -> dict:
    import pandas as pd

    p = out / "ckpt_bands.csv"
    res = {"prediction": "over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster "
                         "than in the bottom band",
           "operational": "per run, within-band breaking share (mean over fc1 and fc2 of blocks 1 to 6) of the top band "
                          "falls by more than that of the bottom band from the first to the last checkpoint; every LR and "
                          "UD run", "verdict": "n/a", "runs": {}}
    if not p.exists():
        return res
    df = pd.read_csv(p)
    df = df[df["block"].isin([b + 1 for b in EARLY]) & df["matrix"].isin(["fc1", "fc2"])]
    for run, g in df.groupby("run"):
        curve = g.groupby("epoch")[["top_breaking_share", "mid_breaking_share", "bottom_breaking_share",
                                    "top_breaking_over_weight", "bottom_breaking_over_weight", "s_w", "s_cov"]].mean()
        if len(curve) < 2:
            res["runs"][run] = {"note": "fewer than two checkpoints measured", "pass": None}
            continue
        e0, e1 = curve.index.min(), curve.index.max()
        ft = curve.loc[e0, "top_breaking_share"] - curve.loc[e1, "top_breaking_share"]
        fb = curve.loc[e0, "bottom_breaking_share"] - curve.loc[e1, "bottom_breaking_share"]
        res["runs"][run] = {"group": str(g["group"].iloc[0]), "first_epoch": int(e0), "last_epoch": int(e1),
                            "fall_top": ft, "fall_bottom": fb,
                            "rel_fall_top": ft / curve.loc[e0, "top_breaking_share"],
                            "rel_fall_bottom": fb / curve.loc[e0, "bottom_breaking_share"],
                            "curve": curve.reset_index().to_dict("records"), "pass": bool(ft > fb)}
    judged = [r for r in res["runs"].values() if r.get("pass") is not None]
    n_lr = sum(r["group"] == "LR" for r in judged)
    n_ud = sum(r["group"] == "UD" for r in judged)
    res["complete"] = bool(n_lr >= 3 and n_ud >= 3)
    if judged:
        ok = all(r["pass"] for r in judged)
        res["verdict"] = ("PASS" if ok else "FAIL") + ("" if res["complete"] else f" (incomplete: {n_lr} LR, {n_ud} UD runs)")
    return res


def svd_losses(a, models) -> dict:
    import pandas as pd

    out = {}
    for m in models:
        p = _resolve(a.must07_dir) / f"rows_{m}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ok = df[df["status"] == "ok"] if "status" in df else df
        o = ok[ok["method"] == "original"]
        if o.empty:
            continue
        top0 = float(o["val_top1"].iloc[0])
        out[m] = {"original_val_top1": top0, "source": str(p)}
        for size in ("6mlp", "6both", "full"):
            s = ok[(ok["method"] == "svd") & (ok["size"] == size) & (ok["fit_seed"] == 0)]
            if not s.empty:
                out[m][f"svd_loss_{size}"] = top0 - float(s["val_top1"].iloc[0])
    return out


def flatness(a, out: Path, models) -> dict:
    import pandas as pd

    table = {}
    for m in models:
        p = out / f"measures_{m}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[(df["op"] == "star") & df["matrix"].isin(["fc1", "fc2"])]
        table[m] = {}
        for tag, rng in (("blocks_1_6", [b + 1 for b in EARLY]), ("all_blocks", sorted(sub["block"].unique()))):
            s = sub[sub["block"].isin(rng)]
            table[m][tag] = {k: float(s[k].mean()) for k in ("erank_w", "erank_w_energy", "erank_act", "erank_input",
                                                            "E_w", "E_cov_plain", "s_w", "s_cov")}
    losses = svd_losses(a, models)
    rel = {}
    for key in ("erank_w", "erank_act", "erank_input"):
        for size in ("6mlp", "full"):
            ms = [m for m in table if f"svd_loss_{size}" in losses.get(m, {})]
            rel[f"{key}_vs_svd_loss_{size}"] = {
                "models": ms, "spearman": spearman([table[m]["blocks_1_6" if size == "6mlp" else "all_blocks"][key]
                                                    for m in ms], [losses[m][f"svd_loss_{size}"] for m in ms])}
    per_block = {}
    meas, truth, _ = load_model_tables(out, models)
    if meas is not None and truth is not None:
        w = meas[(meas["op"] == "star") & meas["matrix"].isin(["fc1", "fc2"])].groupby(["model", "block"])[
            ["erank_w", "erank_act", "E_w", "E_cov_plain"]].mean().reset_index()
        t = truth[truth["method"] == "svd"][["model", "block", "drop_val"]]
        j = w.merge(t, on=["model", "block"])
        for k in ("erank_w", "erank_act", "E_w", "E_cov_plain"):
            per_block[f"{k}_vs_single_block_svd_drop"] = {"n": int(len(j)), "spearman": spearman(j[k], j["drop_val"])}
    return {"table": table, "svd_zero_shot_loss": losses, "across_models": rel, "across_blocks": per_block,
            "note": "reported, not tested; effective ranks divided by the largest possible value"}


def band_summary(out: Path, models) -> dict:
    import pandas as pd

    res = {}
    for m in models:
        p = out / f"measures_{m}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[(df["op"] == "star") & df["matrix"].isin(["fc1", "fc2"])]
        res[m] = {}
        for tag, rng in (("blocks_1_6", [b + 1 for b in EARLY]), ("all_blocks", sorted(sub["block"].unique()))):
            s = sub[sub["block"].isin(rng)]
            res[m][tag] = {f"{band}_{k}": float(s[f"{band}_{k}"].mean()) for band in BANDS
                           for k in ("share_of_breaking", "breaking_share", "variance_share")}
    return res


def stage_report(a, out: Path, device, git: dict, t0: float) -> dict:
    models = models_of(a)
    score = json.loads((out / "rule_score.json").read_text(encoding="utf-8")) if (out / "rule_score.json").exists() else None
    frozen = json.loads((out / "rule_frozen.json").read_text(encoding="utf-8")) if (out / "rule_frozen.json").exists() else None
    metrics = {"experiment": EXPERIMENT, "P5.1": p51(out, models), "P5.2": p52(out),
               "P5.3": {"prediction": "the out-of-sample rule, designed on dinov2_vitb14_reg only and frozen before "
                                      "scoring, is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III "
                                      "and MAE",
                        "frozen": None if frozen is None else {k: frozen[k] for k in ("frozen_utc", "rule", "rule_sha256",
                                                                                      "design_model", "design_complete")},
                        "design_candidates": None if frozen is None else
                        [{"id": c["id"], **c["design_msel"]} for c in frozen["candidates"]],
                        "score": None if score is None else {k: score.get(k) for k in
                                                             ("verdict", "primary", "per_model", "secondary",
                                                              "f4_original", "complete")},
                        "verdict": "n/a" if score is None else score["verdict"]},
               "bands_released": band_summary(out, models), "flatness": flatness(a, out, models),
               "git": git, "prereg": prereg_info()}
    write_json(out / "metrics.json", metrics)
    write_summary(out, metrics)
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"fit": a.fit_seed, "global": a.seed},
               models={m: {"structure_dirs": a.structure_dirs} for m in models}, device=device,
               wall_seconds=time.time() - t0,
               extra={"verdicts": {k: metrics[k]["verdict"] for k in ("P5.1", "P5.2", "P5.3")}})
    for k in ("P5.1", "P5.2", "P5.3"):
        log(f"{k}: {metrics[k]['verdict']}")
    return metrics


def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def write_summary(out: Path, m: dict) -> Path:
    L = ["# CORE-05: why surgery is cheap", "",
         f"Commit {m['git']['git_hash'][:10]}{' (dirty)' if m['git']['git_dirty'] else ''}; PREREG_NEXT_PHASE.md sha256 "
         f"{str(m['prereg'].get('sha256'))[:12]}. Statistics on fit set 0 (ImageNet train); single-block k-NN on the "
         "method-selection queries (design) and validation (scoring).", "",
         "| Prediction | Verdict |", "|---|---|"]
    for k in ("P5.1", "P5.2", "P5.3"):
        L.append(f"| {k}: {m[k]['prediction']} | {m[k]['verdict']} |")
    L += ["", "## P5.1: activation-weighted against weight-energy breaking share, blocks 1 to 6 (fc1, fc2)", "",
          "| Model | Operator | s_w | s_cov | s_cov / s_w | s_mom / s_w | Pass |", "|---|---|---|---|---|---|---|"]
    for model, ops in m["P5.1"]["per_model"].items():
        for op, r in ops.items():
            L.append(f"| {model} | {op} | {_f(r['s_w'])} | {_f(r['s_cov'])} | {_f(r['ratio_cov'])} | "
                     f"{_f(r['ratio_mom'])} | {'yes' if r['pass'] else 'no'} |")
    L += ["", "## P5.2: band split over training (CORE-01)", "", "| Run | Group | Epochs | Fall top | Fall bottom | Pass |",
          "|---|---|---|---|---|---|"]
    for run, r in m["P5.2"]["runs"].items():
        if r.get("pass") is None:
            L.append(f"| {run} | - | - | - | - | {r.get('note')} |")
        else:
            L.append(f"| {run} | {r['group']} | {r['first_epoch']} to {r['last_epoch']} | {_f(r['fall_top'])} | "
                     f"{_f(r['fall_bottom'])} | {'yes' if r['pass'] else 'no'} |")
    p3 = m["P5.3"]
    L += ["", "## P5.3: the out-of-sample rule", ""]
    if p3["frozen"]:
        fr = p3["frozen"]
        L.append(f"Frozen {fr['frozen_utc']} on {fr['design_model']}: rule {fr['rule']['id']} ({fr['rule']['weighting']}, "
                 f"SVD energy {fr['rule']['svd_energy']}), rule sha256 {fr['rule_sha256'][:12]}.")
        L.append("Design candidates (method-selection truth): " + "; ".join(
            f"{c['id']} {c['n_hits']} of {c['n_cases']}" for c in p3["design_candidates"]) + ".")
    if p3["score"]:
        s = p3["score"]
        pr = s.get("primary") or {}
        L.append(f"Scored: {pr.get('n_hits')} of {pr.get('n_cases')} correct ({_f(pr.get('fraction'))}), bar {P53_MIN}: "
                 f"{s['verdict']}. Per model: " + "; ".join(f"{k} {v['n_hits']} of {v['n_cases']}"
                                                           for k, v in (s.get("per_model") or {}).items()) + ".")
        sec = s.get("secondary") or {}
        if sec:
            L.append(f"Secondary: method-selection truth {sec['msel_truth']['n_hits']} of {sec['msel_truth']['n_cases']}; "
                     f"feature-distortion truth {sec['feature_distortion_truth']['n_hits']} of "
                     f"{sec['feature_distortion_truth']['n_cases']}; F4-style rule on the same cases "
                     f"{sec['f4_style_rule_val']['n_hits']} of {sec['f4_style_rule_val']['n_cases']}.")
        f4 = s.get("f4_original") or {}
        if f4.get("primary_prereg_models"):
            pp = f4["primary_prereg_models"]
            L.append(f"Earlier F4 rule (PREREG_E2.md P2, {f4['file']}): {pp.get('n_hits')} of {pp.get('n_cases')}, "
                     f"{f4.get('verdict')}.")
    L += ["", "## Flatness and plain SVD (reported)", "", "| Model | erank W (1-6) | erank W Sigma^1/2 (1-6) | erank input (1-6) | "
          "SVD loss 6mlp | SVD loss full |", "|---|---|---|---|---|---|"]
    fl = m["flatness"]
    for model, t in fl["table"].items():
        lo = fl["svd_zero_shot_loss"].get(model, {})
        b = t["blocks_1_6"]
        L.append(f"| {model} | {_f(b['erank_w'])} | {_f(b['erank_act'])} | {_f(b['erank_input'])} | "
                 f"{_f(lo.get('svd_loss_6mlp'), 2)} | {_f(lo.get('svd_loss_full'), 2)} |")
    L += ["", "Files: measures_<model>.csv, truth_<model>.csv, rule_frozen.json, rule_score.json, ckpt_bands.csv, "
          "spectra/, preds/, metrics.json, meta.json."]
    path = out / "SUMMARY.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    t0 = time.time()
    a = parse_args(argv)
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/core05" if a.tiny else "results/CORE-05")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.tiny else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    log(f"CORE-05 stage {a.stage}: out {out}, device {device}, commit {git['git_hash'][:10]}")
    st = a.stage
    if st in ("measures", "all"):
        stage_measures(a, out, device, git)
    if st == "truth":
        stage_truth(a, out, device, git)
    if st == "all":
        # Safe order: the design model's truth, then the freeze, and only then the scoring models' truth.
        design = design_of(a)
        stage_truth(a, out, device, git, [design])
        stage_freeze(a, out, git)
        stage_truth(a, out, device, git, [m for m in models_of(a) if m != design])
    if st == "freeze":
        stage_freeze(a, out, git)
    if st in ("score", "all"):
        if (out / "rule_frozen.json").exists() or st == "score":
            stage_score(a, out, git)
    if st in ("ckpt", "all"):
        stage_ckpt(a, out, device, git)
    metrics = None
    if st in ("report", "all"):
        metrics = stage_report(a, out, device, git, t0)
    else:
        write_meta(out, name=f"meta_{st}.json", experiment=EXPERIMENT, config=vars(a),
                   seeds={"fit": a.fit_seed, "global": a.seed}, device=device, wall_seconds=time.time() - t0)
    return {"out": str(out), "metrics": metrics}


if __name__ == "__main__":
    main()

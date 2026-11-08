"""Shared helpers for the findings track (F1 to F3 in PAPER_EXPERIMENTS.md).

- operator cleaning: stored operators are float32, so each is snapped back to an exact orthogonal
  involution (eigen-decompose the symmetrised matrix and keep the signs of the eigenvalues);
- subspace comparisons: agreement tr(A B)/d, agreement on a subspace, energy-weighted agreement,
  top-energy subspaces of a Gram matrix and cosines of principal angles;
- the symmetry-tax formula of PROJECT_CONTEXT.md Section 5.3;
- the P1 evaluator and gate G3, with the thresholds copied from PREREG_E2.md (never edit them).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .report import C_AQUA, C_BLUE, C_NULL, C_ORANGE

# ----------------------------------------------------------------------------------------------
# Models: display names and fixed colours (colour follows the model, never its rank)
# ----------------------------------------------------------------------------------------------
MODEL_LABELS = {"dinov2_vitb14_reg": "DINOv2 ViT-B/14 with registers", "dinov2_vitb14": "DINOv2 ViT-B/14"}
MODEL_SHORT = {"dinov2_vitb14_reg": "DINOv2-reg", "dinov2_vitb14": "DINOv2"}
_MODEL_COLOURS = {"dinov2_vitb14_reg": C_BLUE, "dinov2_vitb14": C_ORANGE}
_SPARE_COLOURS = (C_AQUA, C_NULL)


def model_label(name: str, short: bool = False) -> str:
    return (MODEL_SHORT if short else MODEL_LABELS).get(name, name)


def sort_models(names) -> list[str]:
    """Registers model first (main model of the paper), then plain DINOv2, then the rest by name."""
    order = {"dinov2_vitb14_reg": 0, "dinov2_vitb14": 1}
    return sorted(set(names), key=lambda n: (order.get(n, 2), n))


def model_colours(names) -> dict[str, str]:
    out, spare = {}, list(_SPARE_COLOURS)
    for n in sort_models(names):
        out[n] = _MODEL_COLOURS.get(n) or (spare.pop(0) if spare else C_NULL)
    return out


# ----------------------------------------------------------------------------------------------
# Operators and subspaces
# ----------------------------------------------------------------------------------------------
def clean_involution(A: np.ndarray, return_info: bool = False):
    """Nearest exact symmetric orthogonal involution: A_c = V sign(lam) V^T with sym(A) = V lam V^T.

    With `return_info`, also returns V, the signs (+1 even, -1 odd, aligned with the columns of V)
    and a dict with d_plus, d_minus, the largest | |lam| - 1 | and ||A - A_c||_F / ||A_c||_F.
    """
    A = np.asarray(A, dtype=np.float64)
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    signs = np.where(lam >= 0, 1.0, -1.0)
    Ac = (V * signs) @ V.T
    Ac = 0.5 * (Ac + Ac.T)
    if not return_info:
        return Ac
    info = {"d_plus": int((signs > 0).sum()), "d_minus": int((signs < 0).sum()),
            "max_eig_dev": float(np.max(np.abs(np.abs(lam) - 1.0))) if lam.size else 0.0,
            "rel_change": float(np.linalg.norm(A - Ac) / max(np.linalg.norm(Ac), 1e-300))}
    return Ac, V, signs, info


def even_basis(V: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the +1 eigenspace (even subspace)."""
    return V[:, signs > 0]


def agreement(A: np.ndarray, B: np.ndarray) -> float:
    """tr(A B) / d (1 for identical involutions)."""
    return float(np.sum(A * B.T) / A.shape[0])


def subspace_agreement(A: np.ndarray, B: np.ndarray, U: np.ndarray) -> float:
    """tr(U^T A B U) / k for orthonormal U [d, k] and symmetric A, B: 1 iff A and B agree on span(U)."""
    k = U.shape[1]
    if k == 0:
        return float("nan")
    return float(np.sum((A @ U) * (B @ U)) / k)


def energy_weighted_agreement(A: np.ndarray, B: np.ndarray, G: np.ndarray) -> float:
    """tr(G A B) / tr(G) = sum_x (A x).(B x) / sum_x ||x||^2 for the Gram matrix G = sum_x x x^T."""
    tr = float(np.trace(G))
    if tr <= 0:
        return float("nan")
    return float(np.sum(G * (A @ B).T) / tr)


def energy_subspace(G: np.ndarray, frac: float = 0.95):
    """Top eigenvectors of the Gram matrix G covering `frac` of the activation energy tr(G).

    Returns (U [d, k], k, captured energy fraction, eigenvalues in descending order).
    """
    G = np.asarray(G, dtype=np.float64)
    lam, V = np.linalg.eigh(0.5 * (G + G.T))
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    V = V[:, order]
    total = float(lam.sum())
    if total <= 0:
        raise ValueError("Gram matrix has no energy")
    cum = np.cumsum(lam) / total
    k = int(min(np.searchsorted(cum, frac - 1e-12) + 1, len(lam)))
    return V[:, :k], k, float(cum[k - 1]), lam


def random_subspace(d: int, p: int, seed: int) -> np.ndarray:
    """Orthonormal basis [d, p] of a Haar-random p-dimensional subspace (QR of a Gaussian matrix).

    This is the distribution of the even subspace of mirror.fit.random_involution(d, p, .), drawn
    without building and eigen-decomposing the d x d involution (whose matrix is 2 E E^T - I)."""
    if p == 0:
        return np.zeros((d, 0))
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, p)))
    return Q


def principal_cosines(E1: np.ndarray, E2: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between span(E1) and span(E2) (orthonormal columns),
    min(dim) values in descending order."""
    if E1.shape[1] == 0 or E2.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(E1.T @ E2, compute_uv=False)
    return np.clip(np.sort(s)[::-1], 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# F1: the symmetry tax
# ----------------------------------------------------------------------------------------------
def tax_formula(h: int, d: int, n_self: int, tr_A: int) -> float:
    """Redundant share of an h x d mirror-equivariant map: 1/2 - f (d+ - d-) / (2 h d), f = tr(P).

    The equivariant maps form a subspace of dimension (h d + tr(P) tr(A)) / 2 (Section 5.3).
    """
    return 0.5 - n_self * tr_A / (2.0 * h * d)


# ----------------------------------------------------------------------------------------------
# P1 and gate G3 (copied from PREREG_E2.md; fixed once that file is committed)
# ----------------------------------------------------------------------------------------------
P1_LATE_LAYERS = (9, 10, 11)
P1_EARLY_LAYERS = (1, 2, 3, 4, 5, 6)
P1_TOP_FRAC = 0.01            # "the top 1% highest-norm tokens"
P1_PLAIN_MIN_CUT = 2.0        # plain DINOv2: masking cuts mean e over 9 to 11 by at least 2x
P1_REG_MAX_RATIO = 2.0        # registers: mean e over 9 to 11 at most 2x the mean over 1 to 6
P1_TEXT = {
    "plain": "Plain DINOv2: excluding the top 1% highest-norm tokens cuts the mean e(A_l) over "
             "layers 9 to 11 by at least 2x.",
    "registers": "Registers model: the mean e(A_l) over layers 9 to 11 is at most 2x its mean over "
                 "layers 1 to 6.",
}


def _mean_over(values: dict, layers) -> float:
    missing = [l for l in layers if l not in values]
    if missing:
        raise KeyError(f"layers {missing} missing")
    vals = [float(values[l]) for l in layers]
    return float(np.mean(vals)) if all(math.isfinite(v) for v in vals) else float("nan")


def _ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return a / b if b != 0 else float("inf")


def p1_quantities(e_all: dict, e_excl: dict, late=P1_LATE_LAYERS, early=P1_EARLY_LAYERS) -> dict:
    """Both P1 quantities for one model. `e_all[l]` and `e_excl[l]` are relative held-out errors
    of layer l with all tokens and without the top-1% tokens (`e_excl` may be empty when only the
    registers bullet is needed; the masking quantities are then NaN)."""
    late_all, early_all = _mean_over(e_all, late), _mean_over(e_all, early)
    late_excl = _mean_over(e_excl, late) if e_excl and all(l in e_excl for l in late) else float("nan")
    cut, ratio = _ratio(late_all, late_excl), _ratio(late_all, early_all)
    return {"late_layers": list(late), "early_layers": list(early), "mean_e_all_late": late_all,
            "mean_e_excl_top1_late": late_excl, "mean_e_all_early": early_all,
            "masking_cut_late": cut, "late_over_early": ratio}


def evaluate_p1(e_all: dict, e_excl: dict, bullet: str, late=P1_LATE_LAYERS, early=P1_EARLY_LAYERS) -> dict:
    """Evaluate one P1 bullet ("plain" or "registers") exactly as pre-registered.

    plain:     mean_l e_all / mean_l e_excl over the late layers >= 2   (inclusive)
    registers: mean_l e_all (late) / mean_l e_all (early) <= 2          (inclusive)
    `pass` is None when a needed number is not finite.
    """
    if bullet == "plain":
        _mean_over(e_excl, late)  # raises KeyError if a late layer is missing
    q = p1_quantities(e_all, e_excl, late, early)
    if bullet == "plain":
        value, thr, op = q["masking_cut_late"], P1_PLAIN_MIN_CUT, ">="
        ok = value >= thr if math.isfinite(q["mean_e_all_late"]) and math.isfinite(q["mean_e_excl_top1_late"]) else None
    elif bullet == "registers":
        value, thr, op = q["late_over_early"], P1_REG_MAX_RATIO, "<="
        ok = value <= thr if math.isfinite(q["mean_e_all_late"]) and math.isfinite(q["mean_e_all_early"]) else None
    else:
        raise ValueError(f"unknown P1 bullet {bullet}")
    return {"bullet": bullet, "statement": P1_TEXT[bullet], "value": value, "threshold": thr,
            "comparison": op, "pass": None if ok is None else bool(ok),
            "result": "MISSING" if ok is None else ("PASS" if ok else "FAIL"), **q}


def gate_g3(plain_pass, registers_pass) -> dict:
    """G3: claim C3 is dropped if neither masking (plain bullet) nor registers restores the late layers."""
    if plain_pass is True or registers_pass is True:
        status, keep = "PASS", True
    elif plain_pass is False and registers_pass is False:
        status, keep = "FAIL", False
    else:
        status, keep = "INCOMPLETE", None
    return {"gate": "G3", "status": status, "keep_claim_C3": keep,
            "plain_masking_pass": plain_pass, "registers_pass": registers_pass,
            "rule": "C3 is dropped if neither registers nor masking restores the late layers "
                    "(neither P1 bullet passes)."}


# ----------------------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

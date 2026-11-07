"""Closed-form fit of the residual mirror operator (orthogonal involution) and its nulls.

min_A ||Xt - X A||_F over A = A^T, A^2 = I  <=>  max tr(A C), C = X^T Xt.
With S = (C + C^T)/2 = V diag(lam) V^T, the optimum is A = V diag(sign(lam)) V^T.
"""
from __future__ import annotations

import numpy as np


def fit_involution(C: np.ndarray, zero_tol: float = 1e-10, return_info: bool = False):
    """Return (A, d_plus, d_minus) for the cross-moment C.

    Eigenvalues with |lam| <= zero_tol * max|lam| carry no information (for example the
    all-ones direction after channel centring) and are assigned +1, so A stays an involution.
    """
    C = np.asarray(C, dtype=np.float64)
    S = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(S)
    scale = np.max(np.abs(lam)) if lam.size else 0.0
    zero = np.abs(lam) <= zero_tol * scale
    signs = np.where(lam > 0, 1.0, -1.0)
    signs[zero] = 1.0
    A = (V * signs) @ V.T
    A = 0.5 * (A + A.T)
    d_plus = int((signs > 0).sum())
    d_minus = int((signs < 0).sum())
    if return_info:
        return A, d_plus, d_minus, {"eigvals": lam, "n_zero": int(zero.sum())}
    return A, d_plus, d_minus


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal matrix (QR of a Gaussian with the sign fix)."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def random_involution(d: int, d_plus: int, seed: int) -> np.ndarray:
    """Random orthogonal involution with the given number of +1 eigenvalues."""
    V = random_orthogonal(d, np.random.default_rng(seed))
    signs = np.concatenate([np.ones(d_plus), -np.ones(d - d_plus)])
    A = (V * signs) @ V.T
    return 0.5 * (A + A.T)


def agreement(A1: np.ndarray, A2: np.ndarray) -> float:
    """tr(A1 A2) / d: 1 for identical involutions."""
    return float(np.trace(A1 @ A2) / A1.shape[0])


def involution_trace(A: np.ndarray) -> int:
    return int(round(float(np.trace(A))))


def fit_layers(C: np.ndarray, shared_layers: tuple[int, int], sq_x=None, sq_xt=None) -> dict:
    """Per-layer A_l plus the shared operator from summed C over `shared_layers` (inclusive).

    Also returns a scale-normalised shared operator (each C_l divided by its sqrt(||X||^2 ||Xt||^2))
    as a diagnostic; the decision uses the plain sum, as specified.
    """
    n_layers = C.shape[0]
    per_layer = [fit_involution(C[l], return_info=True) for l in range(n_layers)]
    lo, hi = shared_layers
    sel = list(range(lo, hi + 1))
    A_s, dp_s, dm_s, info_s = fit_involution(C[sel].sum(0), return_info=True)
    out = {
        "A_layers": [p[0] for p in per_layer],
        "d_plus": [p[1] for p in per_layer],
        "d_minus": [p[2] for p in per_layer],
        "n_zero": [p[3]["n_zero"] for p in per_layer],
        "eigvals": [p[3]["eigvals"] for p in per_layer],
        "A_shared": A_s, "d_plus_shared": dp_s, "d_minus_shared": dm_s,
        "n_zero_shared": info_s["n_zero"], "eigvals_shared": info_s["eigvals"],
        "shared_layers": [lo, hi],
    }
    if sq_x is not None and sq_xt is not None:
        Cn = sum(C[l] / np.sqrt(max(sq_x[l] * sq_xt[l], 1e-300)) for l in sel)
        A_n, dp_n, dm_n = fit_involution(Cn)
        out.update({"A_shared_norm": A_n, "d_plus_shared_norm": dp_n, "d_minus_shared_norm": dm_n})
    return out

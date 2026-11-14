"""Finite symmetry groups acting on ViT tokens and weights (PROJECT_NEXT_PHASE.md 4.3 and 4.4, MUST-03).

Groups: LR, UD, R180 (order 2), R90 (cyclic of order 4) and Klein = {e, LR, UD, LR.UD}. This module
holds the maths only (actions, operator fits, hidden permutations, the group projection, the character
formula, nulls and surgery); mirror/group_stats.py collects the paired statistics.

Token actions. The patch grid is G x G, patches in row-major order after the prefix tokens (CLS and
registers), which keep their index. An image action g maps patch (r, c) to g(r, c):
    LR   (r, c) -> (r, G-1-c)        torch.flip(x, [-1])
    UD   (r, c) -> (G-1-r, c)        torch.flip(x, [-2])
    R90  (r, c) -> (G-1-c, r)        torch.rot90(x, 1, (-2, -1)), counter-clockwise
    R180 (r, c) -> (G-1-r, G-1-c)    torch.rot90(x, 2, (-2, -1)) (also LR.UD)
    R270 (r, c) -> (c, G-1-r)        torch.rot90(x, 3, (-2, -1))
`token_index(element, G)[p]` is the flat index of g(p); the tokens of g(x) aligned back to the
positions of x are t_gx[idx] (`align_tokens`). tests/test_groups.py checks every convention with a
marker patch.

Operators (column convention). rho(g) is a d x d orthogonal matrix with aligned token of g(x) ~ rho t(x)
for column vectors; in the row form of the statistics, Xt ~ X rho^T with C = X^T Xt. For symmetric
involutions rho = rho^T is the A of mirror/fit.py. The fit error of any orthogonal rho follows from the
moments: e(rho) = sum ||X||^2 + sum ||Xt||^2 - 2 tr(rho C).
  * order 2: variants.fit_layer_operator (centre_moment then fit.fit_involution), as in all earlier runs;
  * R90: orthogonal Procrustes on the channel-centred subspace (the all-ones direction is fixed, so
    rho 1 = 1 as for the fitted involutions), then rho^4 = I is enforced by rounding the rotation angles
    of the real Schur form to multiples of 90 degrees (`round_to_order`);
  * Klein: two commuting involutions (A_LR, A_UD), fitted by coordinate ascent on
    tr(A_LR S_LR) + tr(A_UD S_UD): each step is the exact best involution that commutes with the other
    (sign of the eigenvalues of the compressed moment in each of the other's eigenspaces).

Hidden permutations (index arrays). perm[i] = j means: unit i of the transformed image matches unit j
of the original, so the permutation matrix P (P[i, perm[i]] = 1) acts as P M = M[perm], exactly as in
mirror/surgery.py. Mutual matching (mirror.matching) for involutions; Hungarian on the directional
correlation R[i, j] = corr(zt_i, z_j) for R90, keeping cycles of length 1, 2 and 4 only; for Klein the
two involution matchings, made to commute by keeping, on each orbit where they do not commute, the
matching with the larger gain and setting the other to the identity there.

Group projection. With elements g of G represented by rho_in(g), rho_out(g) and P(g),
    fc1: W1_eq = (1/|G|) sum_g P(g)^T W1 rho_in(g),      b1_eq = (1/|G|) sum_g P(g)^T b1,
    fc2: W2_eq = (1/|G|) sum_g rho_out(g)^T W2 P(g),     b2_eq = (1/|G|) sum_g rho_out(g)^T b2,
the orthogonal projection onto {P(g) W1 = W1 rho_in(g)} and {rho_out(g) W2 = W2 P(g)}. Summing over g
or g^-1 gives the same average, so this is the plan's (1/|G|) sum_g P(g) W rho(g)^T. For |G| = 2 it is
surgery.project_mlp_two (equal to the last bit; tested).

Character formula. dim{W : P(g) W = W rho(g)} = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)); the bias b1 has
one free parameter per hidden orbit (Burnside), b2 has (1/|G|) sum_g tr(rho_out(g)). For LR this is
(h d + f (d+ - d-)) / 2 = surgery.equivariant_param_count. tax = 1 - free / (d h).

Attention. q~ = P_qk q with P_qk the block permutation of heads carrying a dh x dh orthogonal map per
head (block (sigma(h), h) = B[h], the convention of mirror.attention.HeadMaps), v~ = P_v v likewise with
C[h]. Then Wq_eq = avg P_qk(g)^T Wq rho_in(g) (Wk alike, Wv with P_v) and Wo_eq = avg rho_out(g)^T Wo P_v(g).
Involutions use mirror.attention head pairing unchanged. R90: Hungarian on the directional head scores
(attention.directional_head_scores), the same cycle rule as for neurons, per-head maps by Procrustes
(polar of the cross-moment) and, on each cycle, the closing map adjusted so that the product of the maps
around the cycle has the order the group needs (`fit_heads_c4`). Klein: projection onto the
intersection of the LR and UD equivariant subspaces by alternating projections (`project_attn_klein`),
because two independently fitted head structures need not commute; its free-parameter count is not
given by the character formula and is reported as missing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

GROUPS = ("LR", "UD", "R90", "R180", "Klein")
GROUP_ORDER = {"LR": 2, "UD": 2, "R180": 2, "R90": 4, "Klein": 4}
GROUP_GENERATORS = {"LR": ("LR",), "UD": ("UD",), "R180": ("R180",), "R90": ("R90",), "Klein": ("LR", "UD")}
ELEMENTS = ("e", "LR", "UD", "R90", "R180", "R270", "LRUD")
INVOLUTION_ELEMENTS = ("LR", "UD", "R180", "LRUD")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def kind(group: str) -> str:
    """'c2' (one involution), 'c4' (R90) or 'klein'."""
    if group in ("LR", "UD", "R180"):
        return "c2"
    if group == "R90":
        return "c4"
    if group == "Klein":
        return "klein"
    raise ValueError(f"unknown group {group!r}; choose from {GROUPS}")


def parse_groups(spec: str) -> list:
    out = [g.strip() for g in str(spec).split(",") if g.strip()]
    norm = {g.lower(): g for g in GROUPS}
    bad = [g for g in out if g.lower() not in norm]
    if bad:
        raise ValueError(f"unknown group(s) {bad}; choose from {GROUPS}")
    return list(dict.fromkeys(norm[g.lower()] for g in out))


def generators_needed(groups) -> list:
    """Generator passes needed for `groups` (Klein reuses the LR and UD passes)."""
    out = []
    for g in groups:
        for e in GROUP_GENERATORS[g]:
            if e not in out:
                out.append(e)
    return out


# ----------------------------------------------------------------------------------------------
# Image and token actions
# ----------------------------------------------------------------------------------------------
def image_action(x: torch.Tensor, element: str) -> torch.Tensor:
    """Apply a group element to images [..., H, W] (square images for rotations)."""
    if element == "e":
        return x
    if element == "LR":
        return torch.flip(x, dims=[-1])
    if element == "UD":
        return torch.flip(x, dims=[-2])
    if element in ("R180", "LRUD"):
        return torch.rot90(x, 2, dims=(-2, -1))
    if element == "R90":
        return torch.rot90(x, 1, dims=(-2, -1))
    if element == "R270":
        return torch.rot90(x, 3, dims=(-2, -1))
    raise ValueError(f"unknown element {element!r}")


def grid_map(element: str, r: int, c: int, G: int) -> tuple:
    """Patch (r, c) of the original goes to this patch of g(image)."""
    if element == "e":
        return r, c
    if element == "LR":
        return r, G - 1 - c
    if element == "UD":
        return G - 1 - r, c
    if element in ("R180", "LRUD"):
        return G - 1 - r, G - 1 - c
    if element == "R90":
        return G - 1 - c, r
    if element == "R270":
        return c, G - 1 - r
    raise ValueError(f"unknown element {element!r}")


def token_index(element: str, grid_hw) -> np.ndarray:
    """idx[p] = flat patch index of g(p) for p = r * G + c; aligned tokens of g(x) are t_gx[idx]."""
    H, W = (grid_hw, grid_hw) if isinstance(grid_hw, int) else tuple(grid_hw)
    if H != W and element not in ("e", "LR", "UD"):
        raise ValueError(f"{element} needs a square patch grid, got {H} x {W}")
    if H != W:
        idx = np.arange(H * W).reshape(H, W)
        return (idx[:, ::-1] if element == "LR" else idx[::-1] if element == "UD" else idx).reshape(-1).copy()
    G = H
    out = np.empty(G * G, dtype=np.int64)
    for r in range(G):
        for c in range(G):
            rr, cc = grid_map(element, r, c, G)
            out[r * G + c] = rr * G + cc
    return out


def align_tokens(t: torch.Tensor, idx, n_prefix: int) -> torch.Tensor:
    """Tokens of the transformed image [B, n_prefix + G*G, C] re-indexed to the original positions.

    For LR this equals mirror.stats.unflip_tokens."""
    idx_t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(np.asarray(idx), dtype=torch.long)
    idx_t = idx_t.to(t.device)
    if t.shape[1] != n_prefix + idx_t.numel():
        raise ValueError(f"expected {n_prefix + idx_t.numel()} tokens, got {t.shape[1]}")
    return torch.cat([t[:, :n_prefix], t[:, n_prefix:].index_select(1, idx_t)], dim=1)


def cycle_lengths(perm) -> np.ndarray:
    from .variants import cycle_index

    return cycle_index(np.asarray(perm, dtype=np.int64))[2]


def random_permutation_like(perm, rng: np.random.Generator) -> np.ndarray:
    """Random permutation with the cycle type of `perm` (positions shuffled, cycles cut in order)."""
    lens = np.sort(cycle_lengths(perm))[::-1]
    n = int(lens.sum())
    order = rng.permutation(n)
    out = np.arange(n)
    pos = 0
    for L in lens:
        cyc = order[pos:pos + int(L)]
        out[cyc] = np.roll(cyc, -1)
        pos += int(L)
    return out


# ----------------------------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------------------------
_BASIS_CACHE: dict = {}


def ones_complement_basis(d: int) -> np.ndarray:
    """Orthonormal basis [d, d - 1] of the subspace orthogonal to the all-ones vector (Householder)."""
    if d not in _BASIS_CACHE:
        u = np.full(d, 1.0 / math.sqrt(d))
        v = u.copy()
        v[0] -= 1.0
        H = np.eye(d) - 2.0 * np.outer(v, v) / float(v @ v)
        _BASIS_CACHE[d] = np.ascontiguousarray(H[:, 1:])
    return _BASIS_CACHE[d]


def _compress(C: np.ndarray, center: bool):
    C = np.asarray(C, dtype=np.float64)
    if not center:
        return C, None
    Q = ones_complement_basis(C.shape[0])
    return Q.T @ C @ Q, Q


def _lift(M: np.ndarray, Q) -> np.ndarray:
    if Q is None:
        return M
    d = Q.shape[0]
    return Q @ M @ Q.T + np.full((d, d), 1.0 / d)


def polar(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    return U @ Vt


def fit_order2(C: np.ndarray, center: bool = True):
    """(A, d_plus, d_minus): the existing per-layer involution fit (variants.fit_layer_operator)."""
    from .variants import fit_layer_operator

    return fit_layer_operator(C, center)


def round_to_order(M: np.ndarray, m: int) -> np.ndarray:
    """Orthogonal matrix of order dividing m nearest in angle to the orthogonal M: the rotation angles of
    the real Schur form are rounded to multiples of 2 pi / m (real eigenvalue -1 kept only if m is even)."""
    from scipy.linalg import schur

    M = np.asarray(M, dtype=np.float64)
    d = M.shape[0]
    if m == 1:
        return np.eye(d)
    T, Z = schur(M, output="real")
    Tn = np.zeros_like(T)
    step = 2.0 * math.pi / m
    i = 0
    while i < d:
        if i + 1 < d and T[i + 1, i] != 0.0:
            a, b, c, e = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
            theta = math.atan2(0.5 * (c - b), 0.5 * (a + e))
            th = round(theta / step) * step
            co, si = math.cos(th), math.sin(th)
            Tn[i:i + 2, i:i + 2] = [[co, -si], [si, co]]
            i += 2
        else:
            Tn[i, i] = -1.0 if (m % 2 == 0 and T[i, i] < 0) else 1.0
            i += 1
    out = Z @ Tn @ Z.T
    return out


def fit_c4(C: np.ndarray, center: bool = True) -> dict:
    """R90 operator: Procrustes (raw) and the order-4 operator (enforced), column convention."""
    Cs, Q = _compress(C, center)
    M_raw = polar(Cs)                       # rows: Xt ~ X M
    M_enf = round_to_order(M_raw, 4)
    rho_raw, rho = _lift(M_raw, Q).T, _lift(M_enf, Q).T
    return {"rho": rho, "rho_raw": rho_raw, **c4_multiplicities(rho)}


def c4_multiplicities(rho: np.ndarray) -> dict:
    """Eigenvalue multiplicities of an orthogonal rho with rho^4 = I: +1, -1 and pairs +-i."""
    d = rho.shape[0]
    t1 = float(np.trace(rho))
    t2 = float(np.trace(rho @ rho))
    m_i = int(round((d - t2) / 4.0))
    m_p = int(round((d - 2 * m_i + t1) / 2.0))
    return {"m_plus": m_p, "m_minus": d - 2 * m_i - m_p, "m_i_pairs": m_i}


def _sign_eig(S: np.ndarray, zero_tol: float = 1e-10):
    lam, V = np.linalg.eigh(0.5 * (S + S.T))
    scale = np.max(np.abs(lam)) if lam.size else 0.0
    s = np.where(lam > 0, 1.0, -1.0)
    s[np.abs(lam) <= zero_tol * scale] = 1.0
    return s, V


def fit_klein(C_a: np.ndarray, C_b: np.ndarray, center: bool = True, max_iter: int = 20) -> dict:
    """Two commuting symmetric involutions (A_a, A_b) maximising tr(A_a S_a) + tr(A_b S_b), S = sym(C),
    by coordinate ascent from the independent fit of A_a. Both fix the all-ones vector when `center`."""
    Sa, Q = _compress(C_a, center)
    Sb, _ = _compress(C_b, center)
    Sa, Sb = 0.5 * (Sa + Sa.T), 0.5 * (Sb + Sb.T)
    n = Sa.shape[0]
    sa, W = _sign_eig(Sa)
    sb = np.ones(n)
    history = []

    def refine(W, fixed, S):
        """Best involution commuting with the one given by (W, fixed): sign-eig within each eigenspace."""
        Wn, f_new, s_new = np.empty_like(W), np.empty(n), np.empty(n)
        pos = 0
        for val in (1.0, -1.0):
            cols = np.flatnonzero(fixed == val)
            if cols.size == 0:
                continue
            V = W[:, cols]
            s, U = _sign_eig(V.T @ S @ V)
            k = cols.size
            Wn[:, pos:pos + k] = V @ U
            f_new[pos:pos + k] = val
            s_new[pos:pos + k] = s
            pos += k
        return Wn, f_new, s_new

    def objective(W, sa, sb):
        return float(np.einsum("ij,ij,j->", W, Sa @ W, sa) + np.einsum("ij,ij,j->", W, Sb @ W, sb))

    W, sa, sb = refine(W, sa, Sb)            # A_b given A_a
    obj = objective(W, sa, sb)
    history.append(obj)
    for _ in range(max_iter):
        W2, sb2, sa2 = refine(W, sb, Sa)     # A_a given A_b
        W2, sa2, sb2 = refine(W2, sa2, Sb)   # A_b given A_a
        new = objective(W2, sa2, sb2)
        if new <= obj + 1e-12 * max(abs(obj), 1.0):
            break
        W, sa, sb, obj = W2, sa2, sb2, new
        history.append(obj)
    A_a = _lift((W * sa) @ W.T, Q)
    A_b = _lift((W * sb) @ W.T, Q)
    A_a, A_b = 0.5 * (A_a + A_a.T), 0.5 * (A_b + A_b.T)
    return {"A_a": A_a, "A_b": A_b, "objective": obj, "iterations": len(history), **klein_multiplicities(A_a, A_b)}


def klein_multiplicities(A_a: np.ndarray, A_b: np.ndarray) -> dict:
    """Dimensions of the four joint eigenspaces (++, +-, -+, --) of two commuting involutions."""
    d = A_a.shape[0]
    ta, tb, tab = float(np.trace(A_a)), float(np.trace(A_b)), float(np.trace(A_a @ A_b))
    m = {}
    for name, (xa, xb) in {"m_pp": (1, 1), "m_pm": (1, -1), "m_mp": (-1, 1), "m_mm": (-1, -1)}.items():
        m[name] = int(round((d + xa * ta + xb * tb + xa * xb * tab) / 4.0))
    return m


def moment_error(C: np.ndarray, sq_x: float, sq_xt: float, rho: np.ndarray) -> float:
    """sum ||Xt - X rho^T||^2 from the moments (rho orthogonal)."""
    return float(sq_x + sq_xt - 2.0 * np.trace(np.asarray(rho) @ np.asarray(C, dtype=np.float64)))


def _haar_with_ones(d: int, rng: np.random.Generator, fix_ones: bool) -> np.ndarray:
    if fix_ones:
        Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
        return Q
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def random_c2(d: int, d_plus: int, seed: int, fix_ones: bool = True) -> np.ndarray:
    """Random involution with d_plus eigenvalues +1 (A 1 = 1 by default): the construction of
    variants.random_operator without its final eigendecomposition (QR only, so 20 draws per layer stay
    cheap for d = 1024)."""
    if fix_ones and d_plus < 1:
        raise ValueError("fix_ones needs d_plus >= 1")
    Q = _haar_with_ones(d, np.random.default_rng(seed), fix_ones)
    s = np.concatenate([np.ones(int(d_plus)), -np.ones(d - int(d_plus))])
    A = (Q * s) @ Q.T
    return 0.5 * (A + A.T)


def random_c4(d: int, m_plus: int, m_minus: int, m_i_pairs: int, seed: int, fix_ones: bool = True) -> np.ndarray:
    """Random orthogonal rho with rho^4 = I and the given multiplicities (the all-ones vector in the +1
    eigenspace when fix_ones)."""
    if m_plus + m_minus + 2 * m_i_pairs != d:
        raise ValueError("multiplicities do not add up to d")
    if fix_ones and m_plus < 1:
        raise ValueError("fix_ones needs m_plus >= 1")
    Q = _haar_with_ones(d, np.random.default_rng(seed), fix_ones)
    T = np.zeros((d, d))
    i = 0
    for _ in range(m_plus):
        T[i, i] = 1.0
        i += 1
    for _ in range(m_minus):
        T[i, i] = -1.0
        i += 1
    for _ in range(m_i_pairs):
        T[i:i + 2, i:i + 2] = [[0.0, -1.0], [1.0, 0.0]]
        i += 2
    return Q @ T @ Q.T


def random_klein(d: int, m_pp: int, m_pm: int, m_mp: int, m_mm: int, seed: int, fix_ones: bool = True):
    """Random commuting involutions with the given joint multiplicities (all-ones vector in ++)."""
    if m_pp + m_pm + m_mp + m_mm != d:
        raise ValueError("multiplicities do not add up to d")
    if fix_ones and m_pp < 1:
        raise ValueError("fix_ones needs m_pp >= 1")
    Q = _haar_with_ones(d, np.random.default_rng(seed), fix_ones)
    sa = np.concatenate([np.ones(m_pp), np.ones(m_pm), -np.ones(m_mp), -np.ones(m_mm)])
    sb = np.concatenate([np.ones(m_pp), -np.ones(m_pm), np.ones(m_mp), -np.ones(m_mm)])
    A_a, A_b = (Q * sa) @ Q.T, (Q * sb) @ Q.T
    return 0.5 * (A_a + A_a.T), 0.5 * (A_b + A_b.T)


# ----------------------------------------------------------------------------------------------
# Hidden permutations, orbits and fractions
# ----------------------------------------------------------------------------------------------
def clean_cycles(perm, allowed=(1, 2, 4)) -> tuple:
    """Turn cycles whose length is not in `allowed` into fixed points: (perm, units turned, cycles turned)."""
    from .variants import cycle_index

    perm = np.asarray(perm, dtype=np.int64).copy()
    cid, _, lens = cycle_index(perm)
    bad = ~np.isin(lens[cid], np.asarray(allowed))
    perm[bad] = np.flatnonzero(bad)
    return perm, int(bad.sum()), int((~np.isin(lens, np.asarray(allowed))).sum())


def hungarian_c4(R: np.ndarray) -> dict:
    """R90 neuron matching: Hungarian on R (maximise sum_i R[i, perm[i]]), cycles of 1, 2, 4 kept."""
    from .variants import hungarian_permutation

    raw = hungarian_permutation(R)
    perm, n_turned, n_cyc = clean_cycles(raw)
    R = np.asarray(R)
    idx = np.arange(len(perm))
    return {"perm": perm, "perm_raw": raw, "score": R[idx, perm].astype(np.float64),
            "self_score": np.diag(R).astype(np.float64), "n_turned": n_turned, "n_cycles_turned": n_cyc}


def perm_power(perm, k: int) -> np.ndarray:
    """Index array of P^k (P M = M[perm])."""
    p = np.asarray(perm, dtype=np.int64)
    out = np.arange(len(p))
    for _ in range(int(k)):
        out = out[p]
    return out


def perm_compose(p_a, p_b) -> np.ndarray:
    """Index array of P_a P_b (P_a P_b M = M[p_b][p_a] = M[p_b[p_a]])."""
    return np.asarray(p_b)[np.asarray(p_a)]


def orbits(perms) -> tuple:
    """(orbit id per unit, orbit sizes) of the group generated by the permutations."""
    n = len(perms[0])
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in perms:
        p = np.asarray(p)
        for i in np.flatnonzero(p != np.arange(n)):
            a, b = find(i), find(int(p[i]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n)])
    _, oid = np.unique(roots, return_inverse=True)
    sizes = np.bincount(oid)
    return oid, sizes


def commute_share(p_a, p_b) -> float:
    """Share of units where sigma_a sigma_b = sigma_b sigma_a."""
    return float(np.mean(perm_compose(p_a, p_b) == perm_compose(p_b, p_a)))


def klein_commuting(p_a, p_b, gain_a, gain_b) -> dict:
    """Make two involutions commute: on every orbit of <p_a, p_b> where they do not commute, keep the one
    with the larger summed gain (score at the partner minus self-score) and set the other to the identity."""
    p_a, p_b = np.asarray(p_a, dtype=np.int64).copy(), np.asarray(p_b, dtype=np.int64).copy()
    before = commute_share(p_a, p_b)
    ok = perm_compose(p_a, p_b) == perm_compose(p_b, p_a)
    oid, _ = orbits([p_a, p_b])
    bad_orbits = np.unique(oid[~ok])
    changed = 0
    ga, gb = np.asarray(gain_a, dtype=np.float64), np.asarray(gain_b, dtype=np.float64)
    for o in bad_orbits:
        units = np.flatnonzero(oid == o)
        if ga[units].sum() >= gb[units].sum():
            changed += int((p_b[units] != units).sum())
            p_b[units] = units
        else:
            changed += int((p_a[units] != units).sum())
            p_a[units] = units
    assert commute_share(p_a, p_b) == 1.0
    return {"perm_a": p_a, "perm_b": p_b, "commute_share_before": before,
            "n_orbits_fixed": int(len(bad_orbits)), "n_units_changed": changed}


def orbit_fractions(perms, scores, thresholds=THRESHOLDS) -> dict:
    """Shares of hidden units by orbit size, unthresholded and at each threshold.

    perms: generator permutations; scores: per generator and unit, the correlation at the matched
    partner (the self-correlation for a fixed unit). An orbit passes threshold t when every score of
    every unit in it is at least t. For an involution the 2-cycle share at t is the twin fraction of
    PREREG_NEXT_PHASE.md Section 1 (twin correlation = the symmetrised correlation of the pair)."""
    h = len(perms[0])
    oid, sizes = orbits(perms)
    size_u = sizes[oid]
    smin_u = np.min(np.stack([np.asarray(s, dtype=np.float64) for s in scores]), axis=0)
    omin = np.full(len(sizes), np.inf)
    np.minimum.at(omin, oid, smin_u)
    pass_u = omin[oid]
    out = {"frac_fixed": float(np.mean(size_u == 1)), "frac_2": float(np.mean(size_u == 2)),
           "frac_4": float(np.mean(size_u == 4)), "frac_other": float(np.mean(~np.isin(size_u, (1, 2, 4)))),
           "frac_nontrivial": float(np.mean(size_u > 1)), "n_orbits": int(len(sizes)), "h": int(h)}
    for t in thresholds:
        tag = f"t{int(round(100 * t))}"
        ok = pass_u >= t
        out[f"frac_fixed_{tag}"] = float(np.mean(ok & (size_u == 1)))
        out[f"frac_2_{tag}"] = float(np.mean(ok & (size_u == 2)))
        out[f"frac_4_{tag}"] = float(np.mean(ok & (size_u == 4)))
        out[f"frac_nontrivial_{tag}"] = float(np.mean(ok & (size_u > 1)))
    return out


def random_klein_action_like(p_a, p_b, rng: np.random.Generator) -> tuple:
    """Random commuting involutions with the orbit types of (p_a, p_b): same numbers of fixed units,
    2-orbits swapped by a only, by b only, by both, and regular 4-orbits."""
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    h = len(p_a)
    oid, sizes = orbits([p_a, p_b])
    types = []
    for o in range(len(sizes)):
        u = int(np.flatnonzero(oid == o)[0])
        if sizes[o] == 1:
            types.append("1")
        elif sizes[o] == 2:
            types.append(("a" if p_a[u] != u else "") + ("b" if p_b[u] != u else ""))
        elif sizes[o] == 4:
            types.append("4")
        else:
            raise ValueError("orbit of a commuting pair of involutions must have size 1, 2 or 4")
    order = rng.permutation(h)
    na, nb = np.arange(h), np.arange(h)
    pos = 0
    for t in sorted(types):
        if t == "1":
            pos += 1
        elif t in ("a", "b", "ab"):
            u, v = order[pos], order[pos + 1]
            if "a" in t:
                na[u], na[v] = v, u
            if "b" in t:
                nb[u], nb[v] = v, u
            pos += 2
        else:
            u0, u1, u2, u3 = order[pos:pos + 4]
            na[u0], na[u1], na[u2], na[u3] = u1, u0, u3, u2
            nb[u0], nb[u2], nb[u1], nb[u3] = u2, u0, u3, u1
            pos += 4
    return na, nb


# ----------------------------------------------------------------------------------------------
# Group elements of one block and the projection
# ----------------------------------------------------------------------------------------------
@dataclass
class BlockAction:
    """All |G| elements of a group acting on one MLP: rho_in(g), rho_out(g) (d x d, float64 numpy;
    None = identity) and P(g) (index arrays; None = identity). Element 0 is the identity."""

    group: str
    rho_in: list
    rho_out: list
    perms: list
    info: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return len(self.perms)


def _powers(M, n):
    out = [None]
    cur = None
    for _ in range(1, n):
        cur = np.asarray(M, dtype=np.float64) if cur is None else cur @ np.asarray(M, dtype=np.float64)
        out.append(cur)
    return out


def _perm_powers(p, n):
    return [None] + [perm_power(p, k) for k in range(1, n)]


def block_action(group: str, gen_in: list, gen_out: list, gen_perms: list, info: dict | None = None) -> BlockAction:
    """BlockAction from the generators: c2 [g], c4 [r], klein [a, b] (they must commute)."""
    k = kind(group)
    if k == "c2":
        return BlockAction(group, [None, gen_in[0]], [None, gen_out[0]], [None, np.asarray(gen_perms[0])], dict(info or {}))
    if k == "c4":
        return BlockAction(group, _powers(gen_in[0], 4), _powers(gen_out[0], 4), _perm_powers(gen_perms[0], 4),
                           dict(info or {}))
    a_in, b_in = (np.asarray(x, dtype=np.float64) for x in gen_in)
    a_out, b_out = (np.asarray(x, dtype=np.float64) for x in gen_out)
    pa, pb = (np.asarray(p, dtype=np.int64) for p in gen_perms)
    return BlockAction(group, [None, a_in, b_in, a_in @ b_in], [None, a_out, b_out, a_out @ b_out],
                       [None, pa, pb, perm_compose(pa, pb)], dict(info or {}))


def _dense(M, like: torch.Tensor):
    return None if M is None else torch.as_tensor(np.asarray(M), dtype=like.dtype, device=like.device)


def _inv_index(p, device):
    return torch.as_tensor(np.argsort(np.asarray(p)), dtype=torch.long, device=device)


def _rows_T(W: torch.Tensor, P):
    """P^T W for an index array P (rows of W)."""
    return W if P is None else W.index_select(0, _inv_index(P, W.device))


def _cols(W: torch.Tensor, P):
    """W P for an index array P (columns of W)."""
    return W if P is None else W.index_select(1, _inv_index(P, W.device))


def _mean(terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc * (1.0 / len(terms)) if len(terms) == 2 else acc / len(terms)


def project_mlp_group(W1, b1, W2, b2, action: BlockAction) -> dict:
    """Group projection of folded MLP weights (torch tensors, W1 [h, d], W2 [d, h])."""
    like = W1
    b1, W2, b2 = (torch.as_tensor(x, dtype=like.dtype, device=like.device) for x in (b1, W2, b2))
    t1, tb1, t2, tb2 = [], [], [], []
    for R_in, R_out, P in zip(action.rho_in, action.rho_out, action.perms):
        Ri, Ro = _dense(R_in, like), _dense(R_out, like)
        RoT = None if Ro is None else Ro.T.contiguous()
        A = _rows_T(W1, P)
        t1.append(A if Ri is None else A @ Ri)
        tb1.append(_rows_T(b1[:, None], P)[:, 0])
        B = W2 if RoT is None else RoT @ W2
        t2.append(_cols(B, P))
        tb2.append(b2 if RoT is None else RoT @ b2)
    return {"W1": _mean(t1), "b1": _mean(tb1), "W2": _mean(t2), "b2": _mean(tb2)}


def mlp_group_shares(f: dict, action: BlockAction, M1=None, M2=None) -> dict:
    """Breaking shares of folded MLP weights `f` (surgery.fold_mlp) and, with the input second moments
    M1 (fc1 input, d x d) and M2 (fc2 input, h x h), the activation-weighted shares."""
    from .surgery import breaking_share
    from .variants import weighted_share

    e = project_mlp_group(**f, action=action)
    out = {"s_fc1": breaking_share(f["W1"], e["W1"]), "s_fc2": breaking_share(f["W2"], e["W2"]),
           "s_b1": breaking_share(f["b1"], e["b1"]), "s_b2": breaking_share(f["b2"], e["b2"])}
    out["s_mlp"] = 0.5 * (out["s_fc1"] + out["s_fc2"])
    if M1 is not None:
        out["s_fc1_act"] = weighted_share(f["W1"], e["W1"], torch.as_tensor(M1, device=f["W1"].device))
    if M2 is not None:
        out["s_fc2_act"] = weighted_share(f["W2"], e["W2"], torch.as_tensor(M2, device=f["W1"].device))
    return out


# ----------------------------------------------------------------------------------------------
# Character formula
# ----------------------------------------------------------------------------------------------
def _trace(M, d: int) -> float:
    return float(d) if M is None else float(np.trace(np.asarray(M, dtype=np.float64)))


def _fix(P, h: int) -> int:
    return int(h) if P is None else int((np.asarray(P) == np.arange(h)).sum())


def _char_avg(values) -> int:
    v = float(np.mean(values))
    r = int(round(v))
    if abs(v - r) > 1e-6:
        raise ValueError(f"character average {v} is not an integer: the action is not a representation")
    return r


def character_free(traces_rho, fixes) -> int:
    """(1/|G|) sum_g tr(rho(g)) fix(sigma(g))."""
    return _char_avg([float(t) * float(f) for t, f in zip(traces_rho, fixes)])


def mlp_free_params(action: BlockAction, h: int, d: int) -> dict:
    """Free parameters by the character formula: fc1, fc2 (weights), b1 (hidden orbits), b2, and the tax."""
    fix = [_fix(P, h) for P in action.perms]
    tin = [_trace(R, d) for R in action.rho_in]
    tout = [_trace(R, d) for R in action.rho_out]
    f1, f2 = character_free(tin, fix), character_free(tout, fix)
    out = {"free_fc1": f1, "free_fc2": f2, "free_b1": _char_avg(fix), "free_b2": _char_avg(tout),
           "dense_fc": h * d, "tax_fc1": 1.0 - f1 / (h * d), "tax_fc2": 1.0 - f2 / (h * d),
           "tax_mlp": 1.0 - (f1 + f2) / (2.0 * h * d)}
    return out


def projector_matrix(action: BlockAction, h: int, d: int, which: str = "fc1") -> np.ndarray:
    """The projection as an (h d) x (h d) matrix on vec(W) (brute force, small sizes only)."""
    n = h * d
    shape = (h, d) if which == "fc1" else (d, h)
    out = np.zeros((n, n))
    z1, z2 = torch.zeros(h, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    for k in range(n):
        E = torch.zeros(n, dtype=torch.float64)
        E[k] = 1.0
        E = E.reshape(shape)
        if which == "fc1":
            e = project_mlp_group(E, z1, torch.zeros(d, h, dtype=torch.float64), z2, action)["W1"]
        else:
            e = project_mlp_group(torch.zeros(h, d, dtype=torch.float64), z1, E, z2, action)["W2"]
        out[:, k] = e.reshape(-1).numpy()
    return out


def constraint_nullity(action: BlockAction, h: int, d: int, which: str = "fc1") -> int:
    """Brute-force count: dimension of {W : P(g) W = W rho_in(g)} (fc1) or {rho_out(g) W = W P(g)} (fc2)
    over every group element, from the null space of the stacked linear constraints."""
    rows = []
    for R_in, R_out, P in zip(action.rho_in, action.rho_out, action.perms):
        Pm = np.eye(h) if P is None else np.eye(h)[np.asarray(P)]
        if which == "fc1":
            R = np.eye(d) if R_in is None else np.asarray(R_in)
            # P W - W R = 0  ->  (P kron I - I kron R^T) vec_row(W) = 0
            rows.append(np.kron(Pm, np.eye(d)) - np.kron(np.eye(h), R.T))
        else:
            R = np.eye(d) if R_out is None else np.asarray(R_out)
            rows.append(np.kron(R, np.eye(h)) - np.kron(np.eye(d), Pm.T))
    M = np.concatenate(rows, 0)
    return int(M.shape[1] - np.linalg.matrix_rank(M, tol=1e-9))


# ----------------------------------------------------------------------------------------------
# Nulls for one block
# ----------------------------------------------------------------------------------------------
def random_perm_like(group: str, perms_gen, rng: np.random.Generator) -> list:
    """Random hidden permutation(s) with the cycle type (orbit type for Klein) of the fitted ones."""
    k = kind(group)
    if k in ("c2", "c4"):
        return [random_permutation_like(perms_gen[0], rng)]
    return list(random_klein_action_like(perms_gen[0], perms_gen[1], rng))


def random_operator_like(group: str, op: dict, d: int, seed: int, fix_ones: bool = True) -> list:
    """Random generator operator(s) with the same character as the fitted layer operator `op`."""
    k = kind(group)
    if k == "c2":
        return [random_c2(d, op["d_plus"], seed, fix_ones)]
    if k == "c4":
        return [random_c4(d, op["m_plus"], op["m_minus"], op["m_i_pairs"], seed, fix_ones)]
    return list(random_klein(d, op["m_pp"], op["m_pm"], op["m_mp"], op["m_mm"], seed, fix_ones))


def null_summary(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = np.quantile(v, [0.025, 0.975])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


# ----------------------------------------------------------------------------------------------
# Surgery
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def apply_group_surgery(model, blocks, rho_in, rho_out, sigmas, group: str | None = None) -> list:
    """Zero-shot group-projection surgery of the MLPs of `blocks` (0-based), in place, in folded form
    (norm2 affine and LayerScale absorbed, as surgery.apply_surgery_two).

    Per block i: rho_in[i], rho_out[i] and sigmas[i] are either the full element lists (length |G|,
    identity first) or, with `group`, the generator lists (c2: [g]; R90: [r]; Klein: [a, b]).
    Returns the BlockActions used."""
    from .surgery import fold_mlp, write_absorbed_mlp

    blocks = [int(b) for b in blocks]
    if not (len(blocks) == len(rho_in) == len(rho_out) == len(sigmas)):
        raise ValueError("one rho_in, rho_out and sigma list per block")
    used = []
    for b, Ri, Ro, S in zip(blocks, rho_in, rho_out, sigmas):
        act = (block_action(group, Ri, Ro, S) if group is not None
               else BlockAction("custom", list(Ri), list(Ro), list(S)))
        blk = model.blocks[b]
        write_absorbed_mlp(blk, **project_mlp_group(**fold_mlp(blk), action=act))
        used.append(act)
    return used


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------
def head_rep(sigma, maps) -> np.ndarray:
    """Dense [H dh, H dh] representation: block (sigma[h], h) = maps[h]."""
    sigma = np.asarray(sigma)
    H, dh = len(sigma), np.asarray(maps).shape[-1]
    out = np.zeros((H * dh, H * dh))
    for h in range(H):
        g = int(sigma[h])
        out[g * dh:(g + 1) * dh, h * dh:(h + 1) * dh] = maps[h]
    return out


@dataclass
class AttnAction:
    """Group elements on one attention layer: rho_in, rho_out (d x d), P_qk, P_v ([H dh, H dh]); None =
    identity; element 0 is the identity."""

    group: str
    rho_in: list
    rho_out: list
    P_qk: list
    P_v: list
    info: dict = field(default_factory=dict)


def attn_action_from_generators(group: str, gen_in, gen_out, gen_qk, gen_v, info=None) -> AttnAction:
    k = kind(group)
    n = GROUP_ORDER[group]
    if k == "klein":
        raise ValueError("Klein attention uses project_attn_klein (alternating projections)")
    return AttnAction(group, _powers(gen_in[0], n), _powers(gen_out[0], n), _powers(gen_qk[0], n),
                      _powers(gen_v[0], n), dict(info or {}))


def attn_action_c2(hm, A_in, A_out) -> AttnAction:
    """AttnAction of an involution head structure (mirror.attention.HeadMaps)."""
    return AttnAction("c2", [None, A_in], [None, A_out], [None, head_rep(hm.sigma, hm.B)],
                      [None, head_rep(hm.sigma, hm.C)])


def project_attn_group(f: dict, action: AttnAction) -> dict:
    """Group projection of folded attention weights (attention.fold_attn layout)."""
    like = f["Wq"]
    f = {k: torch.as_tensor(v, dtype=like.dtype, device=like.device) for k, v in f.items()}
    acc = {k: [] for k in ("Wq", "Wk", "Wv", "bq", "bk", "bv", "Wo", "bo")}
    for Ri, Ro, Pq, Pv in zip(action.rho_in, action.rho_out, action.P_qk, action.P_v):
        Ri, Ro, Pq, Pv = (_dense(x, like) for x in (Ri, Ro, Pq, Pv))
        for key in ("Wq", "Wk"):
            W = f[key] if Pq is None else Pq.T @ f[key]
            acc[key].append(W if Ri is None else W @ Ri)
        for key in ("bq", "bk"):
            acc[key].append(f[key] if Pq is None else Pq.T @ f[key])
        W = f["Wv"] if Pv is None else Pv.T @ f["Wv"]
        acc["Wv"].append(W if Ri is None else W @ Ri)
        acc["bv"].append(f["bv"] if Pv is None else Pv.T @ f["bv"])
        W = f["Wo"] if Ro is None else Ro.T @ f["Wo"]
        acc["Wo"].append(W if Pv is None else W @ Pv)
        acc["bo"].append(f["bo"] if Ro is None else Ro.T @ f["bo"])
    return {k: _mean(v) for k, v in acc.items()}


def project_attn_klein(f: dict, act_a: AttnAction, act_b: AttnAction, max_sweeps: int = 100,
                       tol: float = 1e-10) -> tuple:
    """Projection onto the intersection of two equivariant subspaces by alternating projections
    (von Neumann). Returns (weights, info with sweeps and the final relative change)."""
    cur = {k: torch.as_tensor(v) for k, v in f.items()}
    norm = math.sqrt(sum(float((cur[k] ** 2).sum()) for k in ("Wq", "Wk", "Wv", "Wo"))) or 1.0
    change = float("inf")
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        nxt = project_attn_group(project_attn_group(cur, act_a), act_b)
        change = math.sqrt(sum(float(((nxt[k] - cur[k]) ** 2).sum()) for k in ("Wq", "Wk", "Wv", "Wo"))) / norm
        cur = nxt
        if change < tol:
            break
    return cur, {"sweeps": sweeps, "last_change": change}


def attn_free_params(action: AttnAction, d: int, D: int) -> dict:
    """Character-formula free parameters of the attention weights (matrices and biases)."""
    tin = [_trace(R, d) for R in action.rho_in]
    tout = [_trace(R, d) for R in action.rho_out]
    tq = [_trace(P, D) for P in action.P_qk]
    tv = [_trace(P, D) for P in action.P_v]
    n = {"Wq": character_free(tq, tin), "Wk": character_free(tq, tin), "Wv": character_free(tv, tin),
         "Wo": character_free(tout, tv), "bq": _char_avg(tq), "bk": _char_avg(tq), "bv": _char_avg(tv),
         "bo": _char_avg(tout)}
    n["matrices"] = n["Wq"] + n["Wk"] + n["Wv"] + n["Wo"]
    n["matrices_original"] = 4 * D * d
    return n


def attn_shares(f: dict, e: dict) -> dict:
    from .attention import breaking_shares_from

    s = breaking_shares_from(f, e)
    return {"s_attn": s["s_all"], "s_q": s["s_q"], "s_k": s["s_k"], "s_v": s["s_v"], "s_o": s["s_o"]}


def fit_heads_c4(mom: dict) -> dict:
    """R90 head structure from AttnStats.moments: sigma (Hungarian on the directional head scores, cycles
    of 1, 2, 4 kept), per-head maps B (q and k) and C (v) by Procrustes, closing map of each cycle
    adjusted so the product of the maps around a cycle of length k has order dividing 4 / k."""
    from .attention import _qk_moments, directional_head_scores

    Mqk, _, _ = _qk_moments(mom)
    Mv = mom["M"]["v"]
    H, dh = Mqk.shape[0], Mqk.shape[-1]
    R = directional_head_scores(mom)                 # R[g, h]: transformed head g vs original head h
    from .variants import hungarian_permutation

    match = hungarian_permutation(R)                 # transformed head g ~ original head match[g]
    sigma_raw = np.argsort(match)                    # sigma[h] = image of original head h
    sigma, n_turned, _ = clean_cycles(sigma_raw)
    B, C = np.zeros((H, dh, dh)), np.zeros((H, dh, dh))
    for h in range(H):
        g = int(sigma[h])
        B[h], C[h] = polar(Mqk[g, h]), polar(Mv[g, h])
    closing = {}
    for maps, tag in ((B, "B"), (C, "C")):
        for cyc in _cycles(sigma):
            k = len(cyc)
            prod = np.eye(dh)
            for h in cyc[:-1]:
                prod = maps[h] @ prod
            Z = maps[cyc[-1]] @ prod
            Zr = round_to_order(Z, 4 // k)
            maps[cyc[-1]] = Zr @ prod.T
            closing[(tag, cyc[0])] = Zr
    return {"sigma": sigma, "B": B, "C": C, "n_turned": n_turned, "scores": R, "closing": closing}


def _cycles(sigma) -> list:
    """Cycles of sigma, each starting at its smallest index and following h -> sigma[h]."""
    sigma = np.asarray(sigma)
    seen = np.zeros(len(sigma), dtype=bool)
    out = []
    for h in range(len(sigma)):
        if seen[h]:
            continue
        cyc, j = [], h
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = int(sigma[j])
        out.append(cyc)
    return out


def attn_action_c4(hs: dict, rho_in, rho_out) -> AttnAction:
    Pq, Pv = head_rep(hs["sigma"], hs["B"]), head_rep(hs["sigma"], hs["C"])
    return attn_action_from_generators("R90", [rho_in], [rho_out], [Pq], [Pv])


def random_heads_c4_like(hs: dict, seed: int) -> dict:
    """Random R90 head structure with the cycle type of hs["sigma"], Haar-random maps along each cycle and
    closing products with the multiplicities of the fitted ones (cycles matched by length, in order)."""
    from .fit import random_orthogonal

    rng = np.random.default_rng(seed)
    sigma = random_permutation_like(hs["sigma"], rng)
    dh = hs["B"].shape[-1]
    H = len(sigma)
    fitted = {}
    for (tag, h0), Z in sorted(hs["closing"].items(), key=lambda kv: (kv[0][0], kv[0][1])):
        k = len(next(c for c in _cycles(hs["sigma"]) if c[0] == h0))
        fitted.setdefault((tag, k), []).append(Z)
    out = {"sigma": sigma, "B": np.zeros((H, dh, dh)), "C": np.zeros((H, dh, dh))}
    used: dict = {}
    for tag, key in (("B", "B"), ("C", "C")):
        maps = out[key]
        for cyc in _cycles(sigma):
            k = len(cyc)
            i = used.get((tag, k), 0)
            used[(tag, k)] = i + 1
            Zf = fitted[(tag, k)][i]
            if k == 4:
                Zr = np.eye(dh)
            elif k == 2:
                lam = np.linalg.eigvalsh(0.5 * (Zf + Zf.T))
                Zr = random_c2(dh, int((lam > 0).sum()), int(rng.integers(1 << 31)), fix_ones=False)
            else:
                m = c4_multiplicities(Zf)
                Zr = random_c4(dh, m["m_plus"], m["m_minus"], m["m_i_pairs"], int(rng.integers(1 << 31)),
                               fix_ones=False)
            prod = np.eye(dh)
            for h in cyc[:-1]:
                maps[h] = random_orthogonal(dh, rng)
                prod = maps[h] @ prod
            maps[cyc[-1]] = Zr @ prod.T
    return out

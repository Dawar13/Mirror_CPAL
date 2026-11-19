"""A02: agreement between independent structure fits (operators, twins, head maps) and the structure sidecar.

FINAL_PHASE_PLAN.md A02 and PREREG_FINAL_PHASE.md Section 2.3. Two fits a and b of the same model (different
fit images) are compared with these operational definitions, coded exactly as written:

Operators (stream operator A*_L of the full model, and per layer for A_layers):
  operator distance   ||A_a - A_b||_F^2 / (2 d): 0 for identical operators; for involutions it equals
                      (d-_a + d-_b - 2 tr(Pi_a Pi_b)) * 2 / d, so 2 (d-_a + d-_b) / d when the odd subspaces are
                      orthogonal and 2 when A_b = -A_a;
  d+ and d-           numbers of +1 and -1 eigenvalues; A02-1 passes when |d-(fit k) - d-(fit 0)| <= 0.15 d-(fit 0);
  odd-subspace agreement, energy weighted: S_a is the symmetrised, centred, energy-normalised cross-moment sum that
                      defines A*_L of fit a (m1.fit_span_operator: sym(sum_l Pc C_l Pc / tr(G_l)), layers 0..L),
                      Pi_a = (I - A_a)/2 the projector on A_a's odd subspace, E_a = Pi_a |S_a| Pi_a (|S| the matrix
                      absolute value), rho(a->b) = tr(Pi_b E_a) / tr(E_a); the agreement is the symmetric mean
                      (rho(a->b) + rho(b->a)) / 2 (A02-2: at least 0.8 for every pair of ImageNet fits). When A_a is
                      the sign rule of S_a, E_a = sum over a's odd eigenvectors v_i of |lam_i| v_i v_i^T, so rho is
                      the |lam|-weighted mean of ||Pi_b v_i||^2, the cos^2 of v_i's angle to b's odd subspace;
  unweighted          the mean cos^2 of the principal angles: sum of squared singular values of U_a^T U_b divided by
                      min(d-_a, d-_b) (U: orthonormal bases of the odd subspaces).
Twins (blocks 1 to 6, code 0 to 5): a strong pair is a matched pair whose symmetrised twin correlation
  S_ij = (R_ij + R_ji) / 2 is at least 0.5; agreement(a->b) is the share of a's strong pairs (pooled over the blocks)
  that b also matches as a pair, of any strength; reported as the mean of both directions (A02-3: at least 0.70 for
  every pair of ImageNet fits) together with the Jaccard index of the two strong-pair sets.
Head maps: the share of heads over all blocks with the same assignment in both fits (self-mapped, or swapped with
  the same partner) (A02-4: at least 0.90 for every pair of ImageNet fits); for heads with the same assignment the
  relative Frobenius distances ||B_a - B_b|| / ||B_a|| and ||C_a - C_b|| / ||C_a|| (fitted maps, before the twin gauge).

The sidecar. SpanStructure files do not store the pair correlations or the span matrix, so
`fit_with_sidecar` runs the canonical pipeline (variants.fit_structure as pass 1, then
mirror_family.fit_span_structure with that pass 1, exactly what fit_span_structure does by itself) and keeps, next to
the structure: per block the partner score S[i, perm[i]] of every neuron (`twin_score`; for a pair (i, j) both entries
are S_ij; for a self-twin S_ii), the span matrix S (d x d float64) and its eigenvalues, the per-layer spectra
eig(sym(Pc C_l Pc)) / tr(G_l), the layer energies tr(G_l) and the per-layer and per-span d-. `structure_sidecar` checks
that the sign rule of S reproduces the structure's A*_L.

Sources: no external sources; definitions from FINAL_PHASE_PLAN.md A02 and the coordinator's operational definitions
(PREREG_FINAL_PHASE.md Section 2.3); the statistics come from mirror/variants.py, mirror/mirror_family.py, mirror/m1.py.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

SIDECAR_FORMAT = 1
STRONG = 0.5
TWIN_BLOCKS = tuple(range(6))
THRESHOLDS = {"A02-1": 0.15, "A02-2": 0.8, "A02-3": 0.70, "A02-4": 0.90}

__all__ = ["SIDECAR_FORMAT", "STRONG", "TWIN_BLOCKS", "THRESHOLDS", "as_operator", "signature", "operator_distance",
           "odd_projector", "odd_basis", "matrix_abs", "odd_energy", "rho", "principal_cosines", "odd_agreement",
           "d_minus_within", "matched_pairs", "strong_pairs", "twin_agreement", "head_agreement", "span_matrix",
           "layer_spectra", "structure_sidecar", "save_sidecar", "load_sidecar", "fit_with_sidecar",
           "compare_structures", "compare_fits", "a02_gates", "all_pairs", "load_image_rows", "image_loader"]


# ----------------------------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------------------------
def as_operator(A) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")
    return 0.5 * (A + A.T)


def signature(A) -> tuple:
    """(d+, d-) of a symmetric involution: numbers of positive and negative eigenvalues."""
    lam = np.linalg.eigvalsh(as_operator(A))
    return int((lam > 0).sum()), int((lam < 0).sum())


def operator_distance(A_a, A_b) -> float:
    """||A_a - A_b||_F^2 / (2 d)."""
    A_a, A_b = as_operator(A_a), as_operator(A_b)
    return float(((A_a - A_b) ** 2).sum() / (2.0 * A_a.shape[0]))


def odd_projector(A) -> np.ndarray:
    """Pi = (I - A) / 2."""
    A = as_operator(A)
    return 0.5 * (np.eye(A.shape[0]) - A)


def odd_basis(A) -> np.ndarray:
    """Orthonormal basis (d x d-) of the odd (-1) eigenspace."""
    lam, V = np.linalg.eigh(as_operator(A))
    return V[:, lam < 0]


def matrix_abs(S) -> np.ndarray:
    """|S| = V |lam| V^T for a symmetric matrix (symmetrised first)."""
    lam, V = np.linalg.eigh(as_operator(S))
    return (V * np.abs(lam)) @ V.T


def odd_energy(A, S) -> np.ndarray:
    """E = Pi |S| Pi."""
    P = odd_projector(A)
    return P @ matrix_abs(S) @ P


def rho(A_a, S_a, A_b) -> float:
    """rho(a->b) = tr(Pi_b E_a) / tr(E_a), E_a = Pi_a |S_a| Pi_a (nan when tr(E_a) = 0, for example d-_a = 0)."""
    E = odd_energy(A_a, S_a)
    den = float(np.trace(E))
    if not den > 0:
        return float("nan")
    return float(np.sum(odd_projector(A_b) * E) / den)


def principal_cosines(A_a, A_b) -> np.ndarray:
    """Cosines of the principal angles between the odd subspaces (singular values of U_a^T U_b, descending)."""
    Ua, Ub = odd_basis(A_a), odd_basis(A_b)
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return np.zeros(0)
    return np.linalg.svd(Ua.T @ Ub, compute_uv=False)


def odd_agreement(A_a, A_b, S_a=None, S_b=None) -> dict:
    """Energy-weighted agreement (both directions and their mean, when S_a and S_b are given) and the unweighted mean
    cos^2 of the principal angles."""
    da, db = signature(A_a)[1], signature(A_b)[1]
    cos = principal_cosines(A_a, A_b)
    m = min(da, db)
    out = {"d_minus_a": da, "d_minus_b": db,
           "unweighted_cos2": float((cos ** 2).sum() / m) if m > 0 else float("nan"),
           "rho_ab": rho(A_a, S_a, A_b) if S_a is not None else None,
           "rho_ba": rho(A_b, S_b, A_a) if S_b is not None else None}
    if out["rho_ab"] is not None and out["rho_ba"] is not None:
        out["weighted"] = 0.5 * (out["rho_ab"] + out["rho_ba"])
    else:
        out["weighted"] = None
    return out


def d_minus_within(d_fit: int, d_ref: int, tol: float = THRESHOLDS["A02-1"]) -> bool:
    """A02-1 rule: |d-(fit) - d-(fit 0)| <= tol d-(fit 0)."""
    return bool(abs(int(d_fit) - int(d_ref)) <= tol * int(d_ref))


# ----------------------------------------------------------------------------------------------
# Twins
# ----------------------------------------------------------------------------------------------
def matched_pairs(perm) -> set:
    """{(i, j)}, i < j, of an involution."""
    p = np.asarray(perm, dtype=np.int64)
    i = np.flatnonzero(p > np.arange(len(p)))
    return {(int(a), int(p[a])) for a in i}


def strong_pairs(perm, score, threshold: float = STRONG) -> set:
    """Matched pairs (i, j), i < j, whose symmetrised twin correlation score[i] (= S_ij) is at least `threshold`."""
    s = np.asarray(score, dtype=np.float64)
    return {(i, j) for i, j in matched_pairs(perm) if s[i] >= threshold}


def twin_agreement(perms_a, score_a, perms_b, score_b=None, blocks=TWIN_BLOCKS, threshold: float = STRONG) -> dict:
    """Share of a's strong pairs in `blocks` (pooled) that b matches as a pair; with score_b also b->a, their mean
    and the Jaccard index of the strong-pair sets. Per-block numbers in "per_block"."""
    blocks = [int(b) for b in blocks]
    Sa, Sb, Ma, Mb, per_block = set(), set(), set(), set(), []
    for b in blocks:
        sa = {(b, i, j) for i, j in strong_pairs(perms_a[b], score_a[b], threshold)}
        ma = {(b, i, j) for i, j in matched_pairs(perms_a[b])}
        mb = {(b, i, j) for i, j in matched_pairs(perms_b[b])}
        sb = ({(b, i, j) for i, j in strong_pairs(perms_b[b], score_b[b], threshold)} if score_b is not None else set())
        Sa |= sa
        Sb |= sb
        Ma |= ma
        Mb |= mb
        per_block.append({"block": b, "n_strong_a": len(sa), "n_strong_b": len(sb) if score_b is not None else None,
                          "a_to_b": len(sa & mb) / len(sa) if sa else float("nan"),
                          "b_to_a": (len(sb & ma) / len(sb) if sb else float("nan")) if score_b is not None else None})
    a_to_b = len(Sa & Mb) / len(Sa) if Sa else float("nan")
    out = {"blocks": blocks, "threshold": float(threshold), "n_strong_a": len(Sa), "a_to_b": a_to_b,
           "n_pairs_a": len(Ma), "n_pairs_b": len(Mb), "pairs_shared": len(Ma & Mb), "per_block": per_block}
    if score_b is not None:
        b_to_a = len(Sb & Ma) / len(Sb) if Sb else float("nan")
        union = Sa | Sb
        out.update({"n_strong_b": len(Sb), "b_to_a": b_to_a, "mean": 0.5 * (a_to_b + b_to_a),
                    "jaccard": len(Sa & Sb) / len(union) if union else float("nan"), "n_strong_shared": len(Sa & Sb)})
    else:
        out.update({"n_strong_b": None, "b_to_a": None, "mean": None, "jaccard": None, "n_strong_shared": None})
    return out


# ----------------------------------------------------------------------------------------------
# Head maps
# ----------------------------------------------------------------------------------------------
def _rel(X, Y) -> float:
    den = float(np.linalg.norm(X))
    return float(np.linalg.norm(X - Y) / den) if den > 0 else float("nan")


def head_agreement(heads_a, heads_b) -> dict:
    """Share of heads (all blocks) with the same assignment sigma; B and C relative distances on those heads."""
    if len(heads_a) != len(heads_b):
        raise ValueError("the fits have different depths")
    n = same = 0
    dB, dC, per_block = [], [], []
    for b, (ha, hb) in enumerate(zip(heads_a, heads_b)):
        sa, sb = np.asarray(ha.sigma), np.asarray(hb.sigma)
        eq = sa == sb
        n += len(sa)
        same += int(eq.sum())
        for h in np.flatnonzero(eq):
            dB.append(_rel(np.asarray(ha.B[h]), np.asarray(hb.B[h])))
            dC.append(_rel(np.asarray(ha.C[h]), np.asarray(hb.C[h])))
        per_block.append({"block": b, "n_heads": int(len(sa)), "n_same": int(eq.sum()),
                          "pairs_a": [list(p) for p in ha.pairs], "pairs_b": [list(p) for p in hb.pairs]})

    def stats(v):
        v = np.asarray(v, dtype=np.float64)
        return (float(v.mean()), float(np.median(v)), float(v.max())) if len(v) else (float("nan"),) * 3

    bm, bmed, bmax = stats(dB)
    cm, cmed, cmax = stats(dC)
    return {"share_same": same / n if n else float("nan"), "n_heads": n, "n_same": same, "B_rel_mean": bm,
            "B_rel_median": bmed, "B_rel_max": bmax, "C_rel_mean": cm, "C_rel_median": cmed, "C_rel_max": cmax,
            "per_block": per_block}


# ----------------------------------------------------------------------------------------------
# Sidecar
# ----------------------------------------------------------------------------------------------
def _centre_proj(d: int) -> np.ndarray:
    return np.eye(d) - np.full((d, d), 1.0 / d)


def span_matrix(C, G, layers, centre: bool = True) -> np.ndarray:
    """sym(sum_l Pc C_l Pc / tr(G_l)): the matrix whose sign rule is A* (same arithmetic as m1.fit_span_operator)."""
    C = np.asarray(C)
    d = C.shape[-1]
    Pc = _centre_proj(d) if centre else None
    total = np.zeros((d, d))
    for l in layers:
        Cl = np.asarray(C[l], dtype=np.float64)
        Cl = Cl if Pc is None else Pc @ Cl @ Pc
        e = float(np.trace(np.asarray(G[l], dtype=np.float64)))
        total += Cl / max(e, 1e-300)
    return 0.5 * (total + total.T)


def layer_spectra(C, G, centre: bool = True) -> np.ndarray:
    """[L + 1, d]: ascending eigenvalues of sym(Pc C_l Pc) / tr(G_l) per stream layer."""
    C = np.asarray(C)
    out = np.zeros((C.shape[0], C.shape[-1]))
    for l in range(C.shape[0]):
        out[l] = np.linalg.eigvalsh(span_matrix(C, G, [l], centre))
    return out


def structure_sidecar(base, st, *, role: str = "", extra: dict | None = None) -> dict:
    """Sidecar arrays for a fitted SpanStructure `st` whose pass 1 was the variants.Structure `base`."""
    from .fit import fit_involution
    from .heal import exact_involution

    L = st.depth
    if base.C is None or base.G is None or base.match_scores is None:
        raise ValueError("pass 1 needs C, G and match_scores")
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in st.perms])
    if not np.array_equal(perms, np.stack([np.asarray(p, dtype=np.int64) for p in base.perms])):
        raise ValueError("the structure's pairings differ from pass 1's")
    S = span_matrix(base.C, base.G, range(L + 1))
    A_S = exact_involution(exact_involution(fit_involution(S)[0]))
    diff = float(np.abs(A_S - np.asarray(st.A_star[L], dtype=np.float64)).max())
    lam = np.linalg.eigvalsh(S)
    energies = np.einsum("lii->l", np.asarray(base.G, dtype=np.float64))
    meta = {"role": role, "model": st.model, "fit_seed": int(st.fit_seed), "n_tokens": base.meta.get("n_tokens"),
            "pass1": st.meta.get("pass1"), "A_from_S_max_abs_diff": diff, "A_from_S_matches": bool(diff < 1e-8),
            "twin_score": "S[i, perm[i]] with S = (R + R^T)/2, R[i, j] = corr(mirrored z_i, original z_j)",
            "S_span": "sym(sum over layers 0..L of Pc C_l Pc / tr(G_l))", **(extra or {})}
    return {"format": SIDECAR_FORMAT, "model": st.model, "role": role, "fit_seed": int(st.fit_seed), "perms": perms,
            "twin_score": np.asarray(base.match_scores, dtype=np.float64), "S_span": S, "S_eig": lam,
            "layer_eig": layer_spectra(base.C, base.G), "energies": energies,
            "d_minus_layers": np.asarray(base.d_minus, dtype=np.int64),
            "d_plus_layers": np.asarray(base.d_plus, dtype=np.int64),
            "d_minus_span": np.asarray([signature(st.A_star[k])[1] for k in range(1, L + 1)], dtype=np.int64),
            "meta": meta}


def save_sidecar(path, sc: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in sc.items() if k != "meta"}
    arrays["meta_json"] = np.array(json.dumps(sc.get("meta", {}), default=str))
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def load_sidecar(path) -> dict:
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    if int(f["format"]) != SIDECAR_FORMAT:
        raise ValueError(f"{path}: sidecar format {int(f['format'])}, expected {SIDECAR_FORMAT}")
    out = {k: v for k, v in f.items() if k != "meta_json"}
    for k in ("model", "role"):
        out[k] = str(out[k])
    out["fit_seed"] = int(out["fit_seed"])
    out["meta"] = json.loads(str(f["meta_json"]))
    return out


def fit_with_sidecar(adapter, loader, device, *, model_name: str, role: str, fit_seed: int, top_k: int = 10,
                     workers: int = 1, register_margin: float = 0.0, operators=None, control_pairings: bool = False):
    """Fit a SpanStructure on `loader` with the canonical pipeline and build its sidecar. Pass 1 is
    variants.fit_structure on the same loader (what fit_span_structure runs when no pass 1 is given), or an existing
    run_e1.py operator file (`operators`, as MUST-07's --operators-template). `control_pairings` also fits the
    pair_actcorr and pair_wcos pairings (slow; not needed for A02). Returns (structure, sidecar)."""
    from .mirror_family import fit_span_structure
    from .variants import fit_structure, load_fit_run

    t0 = time.time()
    if operators is not None:
        base = load_fit_run(operators)
        pass1 = f"operator file {operators}"
    else:
        base = fit_structure(adapter, loader, device, top_k=top_k, workers=workers, center=True)
        pass1 = "variants.fit_structure (pass 1 on the fit set)"
    st = fit_span_structure(adapter, loader, device, model_name=model_name, fit_seed=fit_seed, top_k=top_k,
                            workers=workers, register_margin=register_margin, base=base, pairings=control_pairings)
    st.meta["pass1"] = pass1
    st.meta["role"] = role
    st.meta["fit_seconds_total"] = round(time.time() - t0, 1)
    sc = structure_sidecar(base, st, role=role)
    return st, sc


# ----------------------------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------------------------
def compare_structures(a, b) -> dict:
    """Does structure a reproduce structure b? Bitwise equality and largest differences of A_star (every span),
    A_layers, pairings, head maps and register pairings, plus the A02 agreement numbers that need no sidecar."""
    L = a.depth
    if b.depth != L:
        raise ValueError("different depths")
    ks = sorted(set(a.A_star) & set(b.A_star))
    a_eq = all(np.array_equal(np.asarray(a.A_star[k]), np.asarray(b.A_star[k])) for k in ks)
    a_diff = max(float(np.abs(np.asarray(a.A_star[k]) - np.asarray(b.A_star[k])).max()) for k in ks)
    sig_eq = all(signature(a.A_star[k]) == signature(b.A_star[k]) for k in ks)
    al_diff = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(a.A_layers, b.A_layers))
    p_eq = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a.perms, b.perms))
    p_diff = int(sum(int((np.asarray(x) != np.asarray(y)).sum()) for x, y in zip(a.perms, b.perms)))
    s_eq = all(np.array_equal(np.asarray(x.sigma), np.asarray(y.sigma)) for x, y in zip(a.heads, b.heads))
    b_diff = max(float(np.abs(np.asarray(x.B) - np.asarray(y.B)).max()) for x, y in zip(a.heads, b.heads))
    c_diff = max(float(np.abs(np.asarray(x.C) - np.asarray(y.C)).max()) for x, y in zip(a.heads, b.heads))
    h_eq = s_eq and all(np.array_equal(np.asarray(x.B), np.asarray(y.B)) and np.array_equal(np.asarray(x.C),
                                                                                           np.asarray(y.C))
                        for x, y in zip(a.heads, b.heads))
    r_eq = all(np.array_equal(np.asarray(a.reg_perm[k]), np.asarray(b.reg_perm[k])) for k in ks)
    oa = odd_agreement(a.A_star[L], b.A_star[L])
    return {"reproduces_exactly": bool(a_eq and p_eq and h_eq), "A_star_equal": bool(a_eq),
            "A_star_max_abs_diff": a_diff, "A_star_signatures_equal": bool(sig_eq), "A_layers_max_abs_diff": al_diff,
            "perms_equal": bool(p_eq), "perm_entries_differing": p_diff, "sigma_equal": bool(s_eq),
            "B_max_abs_diff": b_diff, "C_max_abs_diff": c_diff, "heads_equal": bool(h_eq), "reg_perm_equal": bool(r_eq),
            "reproduces_to_1e-9": bool(p_eq and s_eq and r_eq and max(a_diff, b_diff, c_diff) <= 1e-9),
            "operator_distance_L": operator_distance(a.A_star[L], b.A_star[L]),
            "unweighted_cos2_L": oa["unweighted_cos2"], "d_minus_a": oa["d_minus_a"], "d_minus_b": oa["d_minus_b"],
            "head_share_same": head_agreement(a.heads, b.heads)["share_same"],
            "pairs_shared_share": (float(np.mean([len(matched_pairs(x) & matched_pairs(y)) / max(len(matched_pairs(x)), 1)
                                                  for x, y in zip(a.perms, b.perms)])))}


def _check_sidecar(st, sc, name: str) -> None:
    if sc is None:
        return
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in st.perms])
    if not np.array_equal(perms, np.asarray(sc["perms"])):
        raise ValueError(f"{name}: the sidecar's pairings are not the structure's (the sidecar belongs to another fit)")


def compare_fits(st_a, sc_a, st_b, sc_b, *, twin_blocks=TWIN_BLOCKS, threshold: float = STRONG) -> dict:
    """Every A02 metric between fits a and b (SpanStructures with their sidecars; a sidecar may be None, then the
    numbers that need it are None)."""
    _check_sidecar(st_a, sc_a, "fit a")
    _check_sidecar(st_b, sc_b, "fit b")
    L = st_a.depth
    if st_b.depth != L:
        raise ValueError("the fits have different depths")
    twin_blocks = [int(b) for b in twin_blocks if int(b) < L]          # blocks 1 to 6, or fewer in a shallower model
    A_a, A_b = st_a.A_star[L], st_b.A_star[L]
    oa = odd_agreement(A_a, A_b, None if sc_a is None else sc_a["S_span"], None if sc_b is None else sc_b["S_span"])
    layer_dist = [operator_distance(x, y) for x, y in zip(st_a.A_layers, st_b.A_layers)]
    if sc_a is not None:
        ta = twin_agreement(st_a.perms, sc_a["twin_score"], st_b.perms,
                            None if sc_b is None else sc_b["twin_score"], twin_blocks, threshold)
    else:
        ta = None
    ha = head_agreement(st_a.heads, st_b.heads)
    out = {"operator_distance": operator_distance(A_a, A_b), "d_minus_a": oa["d_minus_a"], "d_minus_b": oa["d_minus_b"],
           "rho_ab": oa["rho_ab"], "rho_ba": oa["rho_ba"], "weighted_agreement": oa["weighted"],
           "unweighted_cos2": oa["unweighted_cos2"], "layer_operator_distance": layer_dist,
           "layer_operator_distance_mean": float(np.mean(layer_dist)),
           "head_share_same": ha["share_same"], "head_n_same": ha["n_same"], "head_n": ha["n_heads"],
           "B_rel_mean": ha["B_rel_mean"], "B_rel_max": ha["B_rel_max"], "C_rel_mean": ha["C_rel_mean"],
           "C_rel_max": ha["C_rel_max"]}
    if ta is not None:
        out.update({"twin_a_to_b": ta["a_to_b"], "twin_b_to_a": ta["b_to_a"], "twin_mean": ta["mean"],
                    "twin_jaccard": ta["jaccard"], "n_strong_a": ta["n_strong_a"], "n_strong_b": ta["n_strong_b"],
                    "n_strong_shared": ta["n_strong_shared"], "twin_per_block": ta["per_block"]})
    else:
        out.update({k: None for k in ("twin_a_to_b", "twin_b_to_a", "twin_mean", "twin_jaccard", "n_strong_a",
                                      "n_strong_b", "n_strong_shared", "twin_per_block")})
    return out


def a02_gates(fit_rows: list, pair_rows: list, *, ref: str = "fit0", group: str = "imagenet",
              thresholds: dict | None = None) -> dict:
    """A02-1 to A02-4 from per-fit rows (name, group, d_minus) and pair rows (a, b, weighted_agreement, twin_mean,
    head_share_same). Pairs count only when both fits are in `group`. A gate without data is reported as None."""
    th = {**THRESHOLDS, **(thresholds or {})}
    fits = {r["name"]: r for r in fit_rows}
    members = [r["name"] for r in fit_rows if r.get("group") == group]
    out = {}
    if ref in fits and len(members) > 1:
        d0 = int(fits[ref]["d_minus"])
        vals = {n: int(fits[n]["d_minus"]) for n in members if n != ref}
        fails = [n for n, v in vals.items() if not d_minus_within(v, d0, th["A02-1"])]
        out["A02-1"] = {"pass": not fails, "ref": ref, "d_minus_ref": d0, "d_minus": vals,
                        "rel_change": {n: (v - d0) / d0 if d0 else float("nan") for n, v in vals.items()},
                        "threshold": th["A02-1"], "failing": fails}
    else:
        out["A02-1"] = {"pass": None, "reason": f"needs {ref} and another {group} fit"}
    inside = [p for p in pair_rows if p["a"] in members and p["b"] in members]
    for gate, key in (("A02-2", "weighted_agreement"), ("A02-3", "twin_mean"), ("A02-4", "head_share_same")):
        vals = {f"{p['a']}-{p['b']}": p.get(key) for p in inside}
        if not vals or any(v is None or not np.isfinite(float(v)) for v in vals.values()):
            out[gate] = {"pass": None, "values": vals, "threshold": th[gate],
                         "reason": "missing values" if vals else f"no pair of {group} fits"}
            continue
        fails = [k for k, v in vals.items() if float(v) < th[gate]]
        out[gate] = {"pass": not fails, "values": vals, "min": float(min(vals.values())), "threshold": th[gate],
                     "failing": fails}
    out["note"] = "A02-5 and A02-6 are decided from the healed runs (not computed here)."
    return out


def all_pairs(names) -> list:
    """Unordered pairs in the given order."""
    return list(itertools.combinations(list(names), 2))


# ----------------------------------------------------------------------------------------------
# Image sets without labels (fit sets of any domain)
# ----------------------------------------------------------------------------------------------
def load_image_rows(images_path, index_path=None, rows_key: str | None = None) -> tuple:
    """(uint8 array [N, S, S, 3] as a read-only memmap, rows to use). With index_path and rows_key the rows are
    index[rows_key] (for example cache/train_index.npz, "fit0_rows"); otherwise every row."""
    images = np.load(images_path, mmap_mode="r")
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"{images_path}: expected uint8 [N, S, S, 3], got {images.dtype} {images.shape}")
    if index_path is not None:
        if not rows_key:
            raise ValueError("--index needs --rows-key")
        with np.load(index_path) as z:
            rows = np.asarray(z[rows_key], dtype=np.int64)
    else:
        rows = np.arange(len(images), dtype=np.int64)
    if len(rows) and (rows.min() < 0 or rows.max() >= len(images)):
        raise ValueError("row index out of range")
    return images, rows


def image_loader(images, rows, batch_size: int = 16, num_workers: int = 0, pin: bool = False):
    """mirror.data.make_loader over `rows` with dummy labels (the fit passes need none)."""
    from .data import make_loader

    return make_loader(images, np.zeros(len(images), dtype=np.int64), np.asarray(rows, dtype=np.int64),
                       int(batch_size), int(num_workers), bool(pin))

"""Zero-shot ablation variants (Track A, A1 to A5) and symmetry-versus-low-rank measures (F4).

Conventions (as in mirror/surgery.py): fc1.weight W1 is [h, d], fc2.weight W2 is [d, h]; A is a
d x d residual involution; a neuron permutation is an index array `perm` with P[i, perm[i]] = 1, so
P M = M[perm] and M P^T = M[:, perm]. Block b (0-based) reads layer b and writes layer b + 1, so in
the per-layer design fc1 is projected with A_in = A_b and fc2 with A_out = A_(b+1), exactly as
scripts/run_e1.py does. Parameter counts are MLP weight counts (biases excluded), as in run_e1.py.

Contents
  Involutions  clean_involution (float32 storage -> exact involution, +1/-1 counts kept) and
               random_operator (matched d+, by default with A 1 = 1 like the fitted operators).
  Structure    load_fit_run (results/e1_operators.npz of a run_e1.py run), fit_structure (refit from
               any loader, for A3), fit_shared_operator (plain or energy-normalised sum of C).
  Pairings     random with matched counts, all self-twins, generic cosine matching, plain Hungarian.
  Projection   project_mlp_general with three modes:
                 "exact"  involution P: (W + P W A) / 2, identical to surgery.project_mlp_two;
                 "approx" any permutation: W_eq = (2 W + P W A + P^T W A) / 4, the Track A
                          definition for the plain Hungarian. NOT an exact projection when P has
                          cycles longer than 2 (see project_mlp_general);
                 "cycle"  any permutation: the exact orthogonal projection onto {W : P W A = W}.
  Variants     VariantSpec, apply_variant, variant_copy, build_variant (A1 to A5 and baselines).
  F4 measures  weighted_share, svd_discarded_energy (weight-only or activation-weighted),
               collect_mlp_input_moments, block_rule_measures.
  Runtime      TinyProtocol, build_runtime, fit_subset_rows, original_knn, ResultsCsv: helpers
               shared by scripts/stage1_ablate.py and scripts/f4_rule.py.
"""
from __future__ import annotations

import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fit import agreement, fit_involution, involution_trace, random_involution
from .matching import random_pairing, solve_matching, symmetrise
from .surgery import (apply_svd_baseline, equivariant_param_count, fold_mlp, perm_trace,
                      project_mlp_two, svd_truncate, write_absorbed_mlp)

# ----------------------------------------------------------------------------------------------
# Variant names (Track A)
# ----------------------------------------------------------------------------------------------
A1_VARIANTS = ("per_layer", "shared_span", "shared_all", "identity", "random_operator",
               "svd", "random_structure")          # the last two are the Track A baselines
A2_VARIANTS = ("mirror", "random_pairing", "self_twins", "generic", "hungarian", "hungarian_cycle")
A4_VARIANTS = ("first", "last", "lowest_share")
A5_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
A5_VARIANTS = tuple(f"alpha{a:.2f}" for a in A5_ALPHAS)
A3_SIZES = (100, 500, 2000, 6000)


def default_variants(ablation: str, a3_sizes=A3_SIZES) -> tuple:
    return {"A1": A1_VARIANTS, "A2": A2_VARIANTS, "A3": tuple(f"n{n}" for n in a3_sizes),
            "A4": A4_VARIANTS, "A5": A5_VARIANTS}[ablation]


# ----------------------------------------------------------------------------------------------
# Involutions
# ----------------------------------------------------------------------------------------------
def clean_involution(A, tol: float = 1e-3):
    """Nearest exact symmetric orthogonal involution V sign(lam) V^T of a stored operator.

    Operators saved in float32 are involutions only to about 1e-7. Returns (A, d_plus, d_minus) in
    float64. Raises if an eigenvalue of the symmetrised input is further than `tol` from +-1 (the
    input was not an involution), so the +1/-1 counts cannot change silently.
    """
    A = np.asarray(A, dtype=np.float64)
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    if lam.size and float(np.max(np.abs(np.abs(lam) - 1.0))) > tol:
        raise ValueError("operator is not close to an orthogonal involution")
    signs = np.where(lam > 0, 1.0, -1.0)
    out = (V * signs) @ V.T
    return 0.5 * (out + out.T), int((signs > 0).sum()), int((signs < 0).sum())


def d_plus_of(A) -> int:
    """Number of +1 eigenvalues of an involution, (d + tr A) / 2."""
    return (int(np.asarray(A).shape[0]) + involution_trace(np.asarray(A))) // 2


def random_involution_fixing_ones(d: int, d_plus: int, seed: int) -> np.ndarray:
    """Random orthogonal involution with d_plus eigenvalues +1 and A 1 = 1 (commutes with LayerNorm
    centring, like a fitted operator; same construction as scripts/heal.py). Needs d_plus >= 1."""
    if not 1 <= d_plus <= d:
        raise ValueError("d_plus must be in [1, d]")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return clean_involution(0.5 * (A + A.T))[0]


def random_operator(d: int, d_plus: int, seed: int, fix_ones: bool = True) -> np.ndarray:
    """Matched random involution (same d+ and d-). fix_ones=True also keeps A 1 = 1, as the fitted
    operators do (the all-ones direction is a null direction of the centred C and is assigned +1)."""
    if fix_ones:
        return random_involution_fixing_ones(d, d_plus, seed)
    return clean_involution(random_involution(d, d_plus, seed))[0]


# ----------------------------------------------------------------------------------------------
# Structure (operators and pairings)
# ----------------------------------------------------------------------------------------------
@dataclass
class Structure:
    A_layers: list                 # L + 1 exact involutions (float64); layer 0 = tokens entering block 0
    d_plus: list
    d_minus: list
    perms: list                    # L neuron involutions (mirror matching), int64
    C: np.ndarray | None = None    # [L + 1, d, d] cross-moments sum X^T Xt (float64)
    G: np.ndarray | None = None    # [L + 1, d, d] Gram sum X^T X (float64)
    center: bool = True            # statistics were channel-centred (the all-ones direction is null)
    match_scores: np.ndarray | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.perms)

    @property
    def d(self) -> int:
        return int(self.A_layers[0].shape[0])

    @property
    def h(self) -> int:
        return int(len(self.perms[0]))

    def n_pairs(self, b: int) -> int:
        return int((self.perms[b] != np.arange(self.h)).sum() // 2)

    def n_self(self, b: int) -> int:
        return perm_trace(self.perms[b])


def centre_moment(C: np.ndarray) -> np.ndarray:
    """(I - 11^T/d) C (I - 11^T/d). For channel-centred statistics the all-ones direction is an exact
    null direction of C; float32 storage leaves it at about 1e-7 relative, where its eigenvalue sign
    would be noise. Centring restores the exact null, so fit_involution assigns it +1 (A 1 = 1)."""
    C = np.asarray(C, dtype=np.float64)
    return C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()


def fit_layer_operator(C_l: np.ndarray, center: bool = True):
    """(A, d_plus, d_minus) from one cross-moment (Section 5.1 closed form)."""
    return fit_involution(centre_moment(C_l) if center else C_l)


def span_layers(blocks) -> list:
    """Residual layers read or written by `blocks` (block b reads layer b, writes layer b + 1)."""
    return sorted({int(b) for b in blocks} | {int(b) + 1 for b in blocks})


def layer_energies(G: np.ndarray) -> np.ndarray:
    """Per-layer activation energy tr(G_l) = sum ||X||^2 over fit tokens (channel-centred)."""
    return np.einsum("lii->l", np.asarray(G, dtype=np.float64))


def fit_shared_operator(C: np.ndarray, layers, energies=None, center: bool = True):
    """One operator for several layers from sum_l C_l (plain, the E1 design) or sum_l C_l / e_l
    (energy-normalised, the V-exact design of Section 5.5) with fit.fit_involution.

    Energy normalisation uses e_l = tr(G_l) = sum ||X_l||^2 on the fit set, because the stored
    operator file has G but not sum ||Xt||^2; run_e1.py's diagnostic A_shared_norm divided by
    sqrt(sum ||X||^2 sum ||Xt||^2), which differs only by the small energy difference between an
    image set and its mirrors.
    """
    acc = None
    for l in layers:
        term = np.asarray(C[l], dtype=np.float64)
        if energies is not None:
            term = term / max(float(energies[l]), 1e-300)
        acc = term if acc is None else acc + term
    return fit_layer_operator(acc, center)


def _is_involution(perm) -> bool:
    perm = np.asarray(perm)
    return bool(np.array_equal(perm[perm], np.arange(len(perm))))


def load_fit_run(run, tol: float = 1e-3) -> Structure:
    """Load the fitted structure of a run_e1.py run (protocol v2: runs/x2/<arm>_s<seed>).

    `run` is the run directory (with results/e1_operators.npz) or the npz itself. Operators are
    cleaned to exact involutions; the +1/-1 counts are checked against results/e1_layers.csv and the
    numbers of pairs against results/e1_mlp.csv when those files exist.
    """
    run = Path(run)
    npz = run if run.suffix == ".npz" else run / "results" / "e1_operators.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found (run scripts/run_e1.py first)")
    res_dir = npz.parent
    meta_path = res_dir / "e1_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cfg = meta.get("config", {}) or {}
    center = bool((cfg.get("stats", {}) or {}).get("center", True))
    with np.load(npz) as z:
        files = set(z.files)
        A_raw = np.asarray(z["A_layers"])
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        C = np.asarray(z["C"], dtype=np.float64) if "C" in files else None
        G = np.asarray(z["G"], dtype=np.float64) if "G" in files else None
        scores = np.asarray(z["match_scores"], dtype=np.float64) if "match_scores" in files else None
        A_shared = np.asarray(z["A_shared"], dtype=np.float64) if "A_shared" in files else None
    cleaned = [clean_involution(a, tol) for a in A_raw]
    A_layers = [c[0] for c in cleaned]
    d_plus = [c[1] for c in cleaned]
    d_minus = [c[2] for c in cleaned]
    max_change = float(max(np.abs(a.astype(np.float64) - c).max() for a, c in zip(A_raw, A_layers)))
    for b, p in enumerate(perms):
        if not _is_involution(p):
            raise ValueError(f"{npz}: perms[{b}] is not an involution")
    checks = {}
    lay_csv = res_dir / "e1_layers.csv"
    if lay_csv.exists():
        import pandas as pd

        lay = pd.read_csv(lay_csv)
        stored = lay["d_plus"].astype(int).tolist()
        if stored != d_plus:
            raise ValueError(f"cleaning changed the +1 counts: stored {stored}, cleaned {d_plus}")
        checks["d_plus_matches_e1_layers_csv"] = True
    mlp_csv = res_dir / "e1_mlp.csv"
    if mlp_csv.exists():
        import pandas as pd

        mlp = pd.read_csv(mlp_csv)
        n_pairs = [int((p != np.arange(len(p))).sum() // 2) for p in perms]
        if mlp["n_pairs"].astype(int).tolist() != n_pairs:
            raise ValueError(f"{npz}: pair counts differ from e1_mlp.csv")
        checks["n_pairs_matches_e1_mlp_csv"] = True
    info = {
        "fit_seed": (meta.get("fit_seed") if meta.get("fit_seed") is not None
                     else (cfg.get("data", {}) or {}).get("fit_seed")),
        "model": (cfg.get("model", {}) or {}).get("name"),
        "protocol": meta.get("protocol"),
        "run_git": meta.get("git"),
        "operator_mode": meta.get("operator_mode"),
        "shared_layers": (meta.get("shared", {}) or {}).get("layers"),
        "center": center,
        "max_abs_change_from_cleaning": max_change,
        "checks": checks,
    }
    struct = Structure(A_layers=A_layers, d_plus=d_plus, d_minus=d_minus, perms=perms, C=C, G=G,
                       center=center, match_scores=scores, source=str(npz), meta=info)
    if A_shared is not None and C is not None and info["shared_layers"] == [1, struct.depth]:
        # Refitting the E1 shared operator from the float32 C should reproduce the stored one.
        A_ref = fit_shared_operator(C, range(1, struct.depth + 1), None, center)[0]
        info["shared_all_refit_agreement_with_stored"] = agreement(A_ref, clean_involution(A_shared, tol)[0])
    return struct


def save_structure(path, struct: Structure, with_moments: bool = True, extra: dict | None = None) -> None:
    """Write a structure in the e1_operators.npz layout (A_layers float32, perms int32, C/G float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"A_layers": np.stack(struct.A_layers).astype(np.float32),
              "perms": np.stack(struct.perms).astype(np.int32),
              "d_plus": np.asarray(struct.d_plus, dtype=np.int64)}
    if with_moments and struct.C is not None:
        arrays["C"] = struct.C.astype(np.float32)
    if with_moments and struct.G is not None:
        arrays["G"] = struct.G.astype(np.float32)
    if struct.match_scores is not None:
        arrays["match_scores"] = np.asarray(struct.match_scores, dtype=np.float32)
    arrays["meta_json"] = np.asarray(json.dumps({**struct.meta, **(extra or {})}, default=str))
    np.savez_compressed(path, **arrays)


def load_structure_npz(path, tol: float = 1e-3) -> Structure:
    """Reload a file written by save_structure (A3 cache)."""
    with np.load(path) as z:
        cleaned = [clean_involution(a, tol) for a in z["A_layers"]]
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        C = np.asarray(z["C"], dtype=np.float64) if "C" in z.files else None
        G = np.asarray(z["G"], dtype=np.float64) if "G" in z.files else None
    return Structure([c[0] for c in cleaned], [c[1] for c in cleaned], [c[2] for c in cleaned], perms,
                     C=C, G=G, center=bool(meta.get("center", True)), source=str(path), meta=meta)


def fit_structure(adapter, loader, device, match_blocks=None, top_k: int = 10, workers: int = 1,
                  center: bool = True) -> Structure:
    """Fit per-layer operators and mirror twin matchings from `loader` (A3; protocol of run_e1.py).

    One paired pass (images and their mirrors) accumulates C_l, G_l and the fc1 cross-correlations;
    A_l = fit_involution(C_l) per layer; twins = max-weight involution matching per block
    (mirror.matching, top_k candidates). Only `match_blocks` are matched (default all); the other
    blocks get the identity pairing and are listed in meta["matched_blocks"].
    """
    from .matching import match_many
    from .stats import collect_fit_stats

    fs = collect_fit_stats(adapter, loader, device, center=center, mlp=True)
    res = fs.residual.numpy()
    L, h = adapter.depth, adapter.mlp_hidden
    ops = [fit_layer_operator(res["C"][l], center) for l in range(L + 1)]
    blocks = list(range(L)) if match_blocks is None else sorted({int(b) for b in match_blocks})
    R_list = [fs.mlp.correlation(b) for b in blocks]
    n_tokens = int(res["n"][0])
    del fs
    matches = match_many(R_list, top_k=top_k, workers=workers)
    perms = [np.arange(h) for _ in range(L)]
    scores = np.zeros((L, h))
    for b, m in zip(blocks, matches):
        perms[b] = m.perm
        scores[b] = m.score
    return Structure(A_layers=[o[0] for o in ops], d_plus=[o[1] for o in ops], d_minus=[o[2] for o in ops],
                     perms=perms, C=res["C"], G=res["G"], center=center, match_scores=scores,
                     source="fit_structure", meta={"matched_blocks": blocks, "n_tokens": n_tokens,
                                                   "top_k": top_k, "center": center})


@torch.no_grad()
def collect_mirror_scores(adapter, loader, device, blocks) -> dict:
    """Symmetrised mirror correlation S_b = (R_b + R_b^T) / 2 per block (R[i, j] = corr(zt_i, z_j)),
    the score the twin matching maximises; one paired pass over `loader`."""
    from .stats import MLPStats, run_paired

    blocks = sorted({int(b) for b in blocks})
    slot = {b: i for i, b in enumerate(blocks)}
    ms = MLPStats(len(blocks), adapter.mlp_hidden, device)

    def cb(b, z, zt):
        if b in slot:
            ms.update(slot[b], z, zt)

    run_paired(adapter, loader, device, fc1_cb=cb, desc="mirror scores")
    return {b: symmetrise(ms.correlation(slot[b])) for b in blocks}


def structure_agreement(new: Structure, ref: Structure, blocks) -> dict:
    """How close a refit is to a reference fit: mean tr(A A_ref)/d over the span layers of `blocks`
    and the share of reference twin pairs that the refit also finds."""
    layers = span_layers(blocks)
    agree = float(np.mean([agreement(new.A_layers[l], ref.A_layers[l]) for l in layers]))
    kept = []
    for b in blocks:
        pr, pn = ref.perms[b], new.perms[b]
        paired = pr != np.arange(len(pr))
        if paired.any():
            kept.append(float((pn[paired] == pr[paired]).mean()))
    return {"mean_operator_agreement": agree,
            "mean_pair_recovery": float(np.mean(kept)) if kept else float("nan")}


# ----------------------------------------------------------------------------------------------
# Pairings
# ----------------------------------------------------------------------------------------------
def self_pairing(h: int) -> np.ndarray:
    return np.arange(h)


def perm_from_pairs(h: int, pairs) -> np.ndarray:
    perm = np.arange(h)
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


def _unit_rows(W) -> np.ndarray:
    W = np.asarray(W.detach().cpu() if isinstance(W, torch.Tensor) else W, dtype=np.float64)
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-300)


def cosine_edges(Wn: np.ndarray, top_k: int = 10):
    """Undirected positive-cosine edges (i < j) among each row's top-k most similar rows."""
    h = Wn.shape[0]
    Cs = Wn @ Wn.T
    np.fill_diagonal(Cs, -np.inf)
    k = min(int(top_k), h - 1)
    top = np.argpartition(-Cs, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(h), k)
    cols = top.ravel()
    w = Cs[rows, cols]
    keep = w > 0
    i, j, w = np.minimum(rows, cols)[keep], np.maximum(rows, cols)[keep], w[keep]
    _, first = np.unique(i * h + j, return_index=True)
    return i[first], j[first], w[first]


def _solve(args):
    return solve_matching(*args)


def _exact_pair_count(pairs, Wn: np.ndarray, n_pairs: int) -> list:
    """Keep the n strongest pairs of a matching; if it has fewer, add pairs greedily by cosine among
    the unmatched rows. Returns exactly n_pairs pairs."""
    pairs = [tuple(map(int, p)) for p in pairs]
    w = np.array([float(Wn[a] @ Wn[b]) for a, b in pairs]) if pairs else np.zeros(0)
    order = np.argsort(-w, kind="stable")
    if len(pairs) >= n_pairs:
        return sorted(pairs[i] for i in order[:n_pairs])
    out = list(pairs)
    used = {x for p in pairs for x in p}
    free = np.array([i for i in range(Wn.shape[0]) if i not in used])
    sub = Wn[free] @ Wn[free].T
    iu, ju = np.triu_indices(len(free), 1)
    for t in np.argsort(-sub[iu, ju], kind="stable"):
        if len(out) >= n_pairs:
            break
        a, b = int(free[iu[t]]), int(free[ju[t]])
        if a in used or b in used:
            continue
        out.append((a, b))
        used.update((a, b))
    return sorted(out)


def generic_pairings(W1_list: list, n_pairs_list: list, top_k: int = 10, workers: int = 1) -> list:
    """Generic-similarity pairing (A2): pair neurons whose fc1 rows have high plain cosine, ignoring
    the mirror. Max-weight matching (networkx blossom, as for the mirror twins) on each neuron's top_k
    positive-cosine candidates; then the n strongest pairs are kept, n = the mirror matching's pair
    count, so the parameter count matches. (Keeping the n strongest pairs of the maximum matching is
    a greedy truncation, not the exact maximum-weight matching of size n.) Returns involutions."""
    unit = [_unit_rows(W) for W in W1_list]
    jobs = [cosine_edges(Wn, top_k) for Wn in unit]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            pair_lists = list(ex.map(_solve, jobs))
    else:
        pair_lists = [_solve(j) for j in jobs]
    out = []
    for Wn, pairs, n in zip(unit, pair_lists, n_pairs_list):
        out.append(perm_from_pairs(Wn.shape[0], _exact_pair_count(pairs, Wn, int(n))))
    return out


def hungarian_permutation(S: np.ndarray) -> np.ndarray:
    """Plain assignment on the mirror score without the involution constraint (A2):
    maximise sum_i S[i, perm[i]] over all permutations (scipy linear_sum_assignment)."""
    from scipy.optimize import linear_sum_assignment

    S = np.asarray(S, dtype=np.float64)
    r, c = linear_sum_assignment(S, maximize=True)
    perm = np.empty(S.shape[0], dtype=np.int64)
    perm[r] = c
    return perm


def cycle_index(perm):
    """(cycle id, position in cycle, cycle lengths) with cycles followed i -> perm[i] -> ..."""
    perm = np.asarray(perm, dtype=np.int64)
    h = len(perm)
    cid = np.full(h, -1, dtype=np.int64)
    pos = np.zeros(h, dtype=np.int64)
    lens = []
    for i in range(h):
        if cid[i] >= 0:
            continue
        j, k, c = i, 0, len(lens)
        while cid[j] < 0:
            cid[j], pos[j] = c, k
            k += 1
            j = int(perm[j])
        lens.append(k)
    return cid, pos, np.asarray(lens, dtype=np.int64)


def cycle_stats(perm) -> dict:
    lens = cycle_index(perm)[2]
    return {"n_fixed": int((lens == 1).sum()), "n_2cycles": int((lens == 2).sum()),
            "n_longer_cycles": int((lens > 2).sum()), "neurons_in_longer_cycles": int(lens[lens > 2].sum()),
            "max_cycle_length": int(lens.max()) if lens.size else 0}


def fixed_space_count(perm, d: int, d_plus: int) -> int:
    """Dimension of {W in R^{h x d} : P W A = W} for a permutation P and an involution A with d_plus
    eigenvalues +1: d per even cycle, d_plus per odd cycle. For an involution P this is the Section
    5.3 count (h d + tr(P) tr(A)) / 2 (d per twin pair, d+ per self-twin)."""
    lens = cycle_index(perm)[2]
    return int(np.where(lens % 2 == 0, d, d_plus).sum())


# ----------------------------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------------------------
def _t(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _cycle_project_rows(W: torch.Tensor, A: torch.Tensor, cid, par, lens) -> torch.Tensor:
    """Exact projection of the rows of W onto {W : W[perm[i]] = W[i] A}: along a cycle the fixed rows
    are v, v A, v, v A, ...; v = mean_k W[i_k] A^k, and v must be even (v A = v) on odd cycles."""
    nc = len(lens)
    even = torch.as_tensor(par == 0, device=W.device)
    cid_t = torch.as_tensor(cid, device=W.device)
    E = W.new_zeros(nc, W.shape[1]).index_add_(0, cid_t[even], W[even])
    O = W.new_zeros(nc, W.shape[1]).index_add_(0, cid_t[~even], W[~even])
    lens_t = torch.as_tensor(lens, dtype=W.dtype, device=W.device)
    Vv = (E + O @ A) / lens_t[:, None]
    odd = torch.as_tensor(lens % 2 == 1, device=W.device)
    Vv[odd] = 0.5 * (Vv[odd] + Vv[odd] @ A)
    out = Vv[cid_t]
    out[~even] = out[~even] @ A
    return out


def project_mlp_general(W1, b1, W2, b2, A_in, A_out, perm, mode: str = "exact") -> dict:
    """Mirror projection of folded MLP weights for any neuron permutation.

    mode "exact": P must be an involution; returns surgery.project_mlp_two (W + P W A) / 2, the
        orthogonal projection onto the equivariant subspace.
    mode "approx" (Track A definition for the plain Hungarian, P any permutation):
        W1_eq = (2 W1 + P W1 A_in + P^T W1 A_in) / 4,   b1_eq = (2 b1 + P b1 + P^T b1) / 4,
        W2_eq = (2 W2 + A_out W2 P^T + A_out W2 P) / 4,  b2_eq = (b2 + A_out b2) / 2.
        This is NOT an exact projection when P has cycles longer than 2. It equals the exact
        projection on fixed points and 2-cycles; on a longer cycle it is (2 I + T + T^T) / 4 with
        T(W) = P W A orthogonal, which scales each eigen-component of T with angle theta by
        (1 + cos theta) / 2: the non-equivariant part is shrunk, not removed, the map is not
        idempotent and the result is generally not in the counted subspace.
    mode "cycle" (P any permutation): the exact orthogonal projection onto {W : P W A = W}, the
        average over the cyclic group generated by T (see _cycle_project_rows); b1 is averaged
        over each cycle. Free parameters are given by fixed_space_count.
    """
    if mode == "exact":
        if not _is_involution(perm):
            raise ValueError("mode 'exact' needs an involution; use 'approx' or 'cycle'")
        return project_mlp_two(W1, b1, W2, b2, A_in=A_in, A_out=A_out, perm=perm)
    Ai, Ao = _t(A_in, W1), _t(A_out, W1)
    b1, W2, b2 = _t(b1, W1), _t(W2, W1), _t(b2, W1)
    perm = np.asarray(perm, dtype=np.int64)
    p = torch.as_tensor(perm, device=W1.device)
    inv = torch.as_tensor(np.argsort(perm), device=W1.device)
    if mode == "approx":
        AW2 = Ao @ W2
        return {"W1": 0.25 * (2 * W1 + W1[p] @ Ai + W1[inv] @ Ai),
                "b1": 0.25 * (2 * b1 + b1[p] + b1[inv]),
                "W2": 0.25 * (2 * W2 + AW2[:, p] + AW2[:, inv]),
                "b2": 0.5 * (b2 + Ao @ b2)}
    if mode == "cycle":
        cid, pos, lens = cycle_index(perm)
        par = pos % 2
        cid_t = torch.as_tensor(cid, device=W1.device)
        mean_b1 = b1.new_zeros(len(lens)).index_add_(0, cid_t, b1) / _t(lens, b1)
        return {"W1": _cycle_project_rows(W1, Ai, cid, par, lens),
                "b1": mean_b1[cid_t],
                "W2": _cycle_project_rows(W2.T, Ao, cid, par, lens).T,
                "b2": 0.5 * (b2 + Ao @ b2)}
    raise ValueError(f"unknown projection mode {mode!r}")


# ----------------------------------------------------------------------------------------------
# Variant specification and application
# ----------------------------------------------------------------------------------------------
@dataclass
class VariantSpec:
    name: str
    blocks: list                    # 0-based block indices converted
    method: str                     # "mirror" or "svd"
    A_in: list | None = None        # per converted block (mirror)
    A_out: list | None = None
    perms: list | None = None
    projection: str = "exact"       # exact | approx | cycle (mirror)
    alpha: float = 1.0              # projection strength (A5): W <- W - alpha (W - W_eq)
    ranks: list | None = None       # [(r_fc1, r_fc2)] per converted block (svd)
    params_kept: int = 0            # MLP weight parameters of the converted blocks after conversion
    params_dense: int = 0           # the same blocks before conversion (2 h d per block)
    params_exact: bool = True       # False when the weights are not exactly in the counted form
    info: dict = field(default_factory=dict)


@torch.no_grad()
def apply_variant(model: nn.Module, spec: VariantSpec) -> None:
    """Apply `spec` to `model` in place (folded weights; biases kept for SVD, projected for mirror)."""
    if spec.method == "svd":
        apply_svd_baseline(model, list(spec.blocks), [tuple(r) for r in spec.ranks])
        return
    if spec.method != "mirror":
        raise ValueError(f"unknown method {spec.method!r}")
    alpha = float(spec.alpha)
    for b, Ai, Ao, perm in zip(spec.blocks, spec.A_in, spec.A_out, spec.perms):
        blk = model.blocks[b]
        f = fold_mlp(blk)
        e = project_mlp_general(**f, A_in=Ai, A_out=Ao, perm=perm, mode=spec.projection)
        if alpha != 1.0:
            e = {k: f[k] - alpha * (f[k] - e[k]) for k in f}
        write_absorbed_mlp(blk, **e)


def variant_copy(model: nn.Module, spec: VariantSpec) -> nn.Module:
    m = copy.deepcopy(model)
    apply_variant(m, spec)
    return m


def spec_signature(spec: VariantSpec) -> str:
    """Content hash of what a variant does to the model (method, blocks, projection, alpha, ranks and
    the exact operators and pairings). Two specs with the same signature produce the same model, so
    the Stage 1 driver evaluates it once (for example A1 per_layer = A2 mirror = A4 first = A5
    alpha1.00 at the same k)."""
    import hashlib

    hs = hashlib.sha1()
    hs.update(json.dumps([spec.method, [int(b) for b in spec.blocks], spec.projection, float(spec.alpha),
                          [list(map(int, r)) for r in (spec.ranks or [])]]).encode())
    for arr in list(spec.A_in or []) + list(spec.A_out or []):
        hs.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    for p in spec.perms or []:
        hs.update(np.ascontiguousarray(p, dtype=np.int64).tobytes())
    return hs.hexdigest()[:20]


def mirror_spec(name: str, blocks, A_in, A_out, perms, h: int, d: int, projection: str = "exact",
                alpha: float = 1.0, info: dict | None = None) -> VariantSpec:
    """Mirror variant with its parameter count: fixed-subspace dimension of each matrix at alpha = 1
    (Section 5.3 count for involutions); dense count for 0 <= alpha < 1 (an interpolated matrix has no
    exact structure)."""
    blocks = [int(b) for b in blocks]
    dense = 2 * h * d * len(blocks)
    if alpha == 1.0:
        kept = sum(fixed_space_count(p, d, d_plus_of(a)) + fixed_space_count(p, d, d_plus_of(o))
                   for a, o, p in zip(A_in, A_out, perms))
    else:
        kept = dense
    return VariantSpec(name=name, blocks=blocks, method="mirror", A_in=list(A_in), A_out=list(A_out),
                       perms=[np.asarray(p, dtype=np.int64) for p in perms], projection=projection,
                       alpha=float(alpha), params_kept=int(kept), params_dense=int(dense),
                       params_exact=(projection != "approx" or alpha != 1.0), info=dict(info or {}))


def per_layer_ops(struct: Structure, blocks):
    return [struct.A_layers[b] for b in blocks], [struct.A_layers[b + 1] for b in blocks]


def mirror_param_counts(struct: Structure, blocks) -> list:
    """[(eq_fc1, eq_fc2)] of the per-layer mirror structure (A_in = A_b, A_out = A_(b+1))."""
    h, d = struct.h, struct.d
    return [(equivariant_param_count(h, d, struct.n_self(b), 2 * struct.d_plus[b] - d),
             equivariant_param_count(h, d, struct.n_self(b), 2 * struct.d_plus[b + 1] - d)) for b in blocks]


def svd_spec(name: str, struct: Structure, blocks, info: dict | None = None) -> VariantSpec:
    """SVD baseline matched per matrix to the per-layer mirror structure: r = floor(eq / (h + d)) for
    each matrix's equivariant parameter count eq (surgery.svd_rank_for)."""
    from .surgery import svd_rank_for

    h, d = struct.h, struct.d
    blocks = [int(b) for b in blocks]
    eq = mirror_param_counts(struct, blocks)
    ranks = [(svd_rank_for(h, d, e1), svd_rank_for(h, d, e2)) for e1, e2 in eq]
    kept = sum((r1 + r2) * (h + d) for r1, r2 in ranks)
    target = sum(e1 + e2 for e1, e2 in eq)
    return VariantSpec(name=name, blocks=blocks, method="svd", ranks=ranks, params_kept=int(kept),
                       params_dense=2 * h * d * len(blocks),
                       info={**(info or {}), "ranks": [list(r) for r in ranks],
                             "matched_to_mirror_params": int(target),
                             "relative_shortfall": float(1 - kept / target)})


# ----------------------------------------------------------------------------------------------
# Variant builders (A1 to A5)
# ----------------------------------------------------------------------------------------------
def _random_layer_ops(struct: Structure, layers, seed: int, fix_ones: bool) -> dict:
    return {l: random_operator(struct.d, struct.d_plus[l], seed * 1000 + l, fix_ones) for l in layers}


def _random_pairings(struct: Structure, blocks, seed: int) -> list:
    return [random_pairing(struct.h, struct.n_pairs(b), seed=seed * 1000 + 500 + b) for b in blocks]


def operator_variant(name: str, struct: Structure, blocks, seed: int = 0, fix_ones: bool = True):
    """A1 operators for `blocks`: returns (A_in list, A_out list, info)."""
    blocks = [int(b) for b in blocks]
    d, L = struct.d, struct.depth
    if name == "per_layer":
        A_in, A_out = per_layer_ops(struct, blocks)
        return A_in, A_out, {"operator": "per-layer A_b (fc1) and A_(b+1) (fc2)"}
    if name in ("shared_span", "shared_all"):
        if struct.C is None or (name == "shared_span" and struct.G is None):
            raise ValueError(f"{name} needs C (and G) in the operator file")
        if name == "shared_span":
            layers = span_layers(blocks)
            A, dp, dm = fit_shared_operator(struct.C, layers, layer_energies(struct.G), struct.center)
            how = "sum of C_l / tr(G_l) over the span layers (energy-normalised, V-exact)"
        else:
            layers = list(range(1, L + 1))
            A, dp, dm = fit_shared_operator(struct.C, layers, None, struct.center)
            how = "plain sum of C_l over layers 1..L (E1 design)"
        return [A] * len(blocks), [A] * len(blocks), {"operator": how, "layers": layers,
                                                      "d_plus": dp, "d_minus": dm}
    if name == "identity":
        I = np.eye(d)
        return [I] * len(blocks), [I] * len(blocks), {"operator": "A = I", "d_plus": d, "d_minus": 0}
    if name == "random_operator":
        R = _random_layer_ops(struct, span_layers(blocks), seed, fix_ones)
        return ([R[b] for b in blocks], [R[b + 1] for b in blocks],
                {"operator": "random involution per layer, matched d+ and d-", "fix_ones": fix_ones,
                 "seed": seed})
    raise ValueError(f"unknown operator variant {name!r}")


def pairing_variant(name: str, struct: Structure, blocks, seed: int = 0, precomputed: dict | None = None,
                    scores: dict | None = None, model: nn.Module | None = None, top_k: int = 10,
                    workers: int = 1):
    """A2 pairings for `blocks`: returns (perm list, projection mode, info).

    `precomputed` maps block -> permutation (cached generic or Hungarian results); otherwise generic
    needs `model` and Hungarian needs `scores` (block -> symmetrised mirror score S)."""
    blocks = [int(b) for b in blocks]
    h = struct.h
    if name == "mirror":
        return [struct.perms[b] for b in blocks], "exact", {"pairing": "mirror matching (stored)"}
    if name == "random_pairing":
        return (_random_pairings(struct, blocks, seed), "exact",
                {"pairing": "random, matched numbers of pairs and self-twins", "seed": seed})
    if name == "self_twins":
        return [self_pairing(h) for _ in blocks], "exact", {"pairing": "all self-twins"}
    if name == "generic":
        if precomputed is None:
            if model is None:
                raise ValueError("generic pairing needs the model (fc1 rows)")
            W1 = [fold_mlp(model.blocks[b])["W1"] for b in blocks]
            perms = generic_pairings(W1, [struct.n_pairs(b) for b in blocks], top_k, workers)
            precomputed = dict(zip(blocks, perms))
        return ([precomputed[b] for b in blocks], "exact",
                {"pairing": "plain cosine of folded fc1 rows, max-weight matching, matched pair count",
                 "top_k": top_k})
    if name in ("hungarian", "hungarian_cycle"):
        if precomputed is None:
            if scores is None:
                raise ValueError("Hungarian pairing needs the mirror scores")
            precomputed = {b: hungarian_permutation(scores[b]) for b in blocks}
        perms = [precomputed[b] for b in blocks]
        mode = "approx" if name == "hungarian" else "cycle"
        st = [cycle_stats(p) for p in perms]
        return perms, mode, {
            "pairing": "plain Hungarian on the mirror score (no involution constraint)",
            "projection": ("(2W + PWA + P^T W A)/4, NOT an exact projection for cycles longer than 2"
                           if mode == "approx" else "exact orthogonal projection (cycle average)"),
            "neurons_in_longer_cycles": [s["neurons_in_longer_cycles"] for s in st],
            "n_2cycles": [s["n_2cycles"] for s in st], "n_fixed": [s["n_fixed"] for s in st],
            "max_cycle_length": [s["max_cycle_length"] for s in st]}
    raise ValueError(f"unknown pairing variant {name!r}")


def mlp_block_shares(model: nn.Module, struct: Structure) -> list:
    """Per-block breaking share of the per-layer mirror structure, (s1 + s2) / 2 (A4 ordering)."""
    from .surgery import mlp_breaking_shares_two

    out = []
    for b in range(struct.depth):
        sh = mlp_breaking_shares_two(model.blocks[b], struct.A_layers[b], struct.A_layers[b + 1], struct.perms[b])
        out.append(0.5 * (sh["s1"] + sh["s2"]))
    return out


def block_order(strategy: str, L: int, k: int, shares=None) -> list:
    """A4: which k blocks to convert (0-based, ascending)."""
    if strategy == "first":
        return list(range(k))
    if strategy == "last":
        return list(range(L - k, L))
    if strategy == "lowest_share":
        if shares is None:
            raise ValueError("lowest_share needs the per-block breaking shares")
        return sorted(int(b) for b in np.argsort(np.asarray(shares), kind="stable")[:k])
    raise ValueError(f"unknown block order {strategy!r}")


def parse_alpha(variant: str) -> float:
    if not variant.startswith("alpha"):
        raise ValueError(f"A5 variant must look like alpha0.25, got {variant!r}")
    return float(variant[len("alpha"):])


def build_variant(ablation: str, variant: str, struct: Structure, k: int, seed: int = 0, *,
                  model: nn.Module | None = None, precomputed: dict | None = None,
                  scores: dict | None = None, shares=None, fix_ones: bool = True, top_k: int = 10,
                  workers: int = 1) -> VariantSpec:
    """VariantSpec for one Track A cell. Unless the ablation says otherwise the variant converts
    blocks 0..k-1 with per-layer operators, the stored mirror matching and alpha = 1.

    A1 operator (+ baselines svd and random_structure), A2 pairing, A3 (struct is the refit from n
    images; variant "n<n>"), A4 block order, A5 alpha ("alpha0.25"). `seed` drives the random
    variants (matched random operator and pairing use the same seeds, so random_structure combines
    exactly the random_operator operators and the random_pairing pairings).
    """
    h, d, L = struct.h, struct.d, struct.depth
    if not 1 <= k <= L:
        raise ValueError(f"k = {k} outside 1..{L}")
    first = list(range(k))
    A_pl_in, A_pl_out = per_layer_ops(struct, first)
    mirror_perms = [struct.perms[b] for b in first]
    if ablation == "A1":
        if variant == "svd":
            return svd_spec(variant, struct, first, {"baseline": "SVD, rank matched per matrix"})
        if variant == "random_structure":
            A_in, A_out, info = operator_variant("random_operator", struct, first, seed, fix_ones)
            perms = _random_pairings(struct, first, seed)
            info = {**info, "pairing": "random, matched counts", "baseline": "matched random structure"}
            return mirror_spec(variant, first, A_in, A_out, perms, h, d, info=info)
        A_in, A_out, info = operator_variant(variant, struct, first, seed, fix_ones)
        return mirror_spec(variant, first, A_in, A_out, mirror_perms, h, d, info=info)
    if ablation == "A2":
        perms, mode, info = pairing_variant(variant, struct, first, seed, precomputed, scores, model, top_k, workers)
        return mirror_spec(variant, first, A_pl_in, A_pl_out, perms, h, d, projection=mode, info=info)
    if ablation == "A3":
        return mirror_spec(variant, first, A_pl_in, A_pl_out, mirror_perms, h, d,
                           info={"refit": struct.meta.get("n_images"), "fit_rows": struct.meta.get("fit_rows")})
    if ablation == "A4":
        blocks = block_order(variant, L, k, shares)
        A_in, A_out = per_layer_ops(struct, blocks)
        info = {"order": variant}
        if shares is not None:
            info["block_shares"] = [float(shares[b]) for b in blocks]
        return mirror_spec(variant, blocks, A_in, A_out, [struct.perms[b] for b in blocks], h, d, info=info)
    if ablation == "A5":
        alpha = parse_alpha(variant)
        return mirror_spec(variant, first, A_pl_in, A_pl_out, mirror_perms, h, d, alpha=alpha,
                           info={"alpha": alpha})
    raise ValueError(f"unknown ablation {ablation!r}")


# ----------------------------------------------------------------------------------------------
# F4: symmetry versus low rank (Section 5.6)
# ----------------------------------------------------------------------------------------------
def psd_sqrt(M: torch.Tensor) -> torch.Tensor:
    lam, V = torch.linalg.eigh(0.5 * (M + M.T))
    return (V * lam.clamp_min(0).sqrt()) @ V.T


def weighted_spectrum(W: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    """Squared singular values of W M^(1/2), descending (W [m, n], M the n x n input second moment)."""
    W = W.to(torch.float64)
    if M is None:
        return torch.linalg.svdvals(W) ** 2
    M = M.to(W)
    if W.shape[0] <= W.shape[1]:
        ev = torch.linalg.eigvalsh(W @ M @ W.T)
    else:
        K = psd_sqrt(M)
        ev = torch.linalg.eigvalsh(K @ (W.T @ W) @ K)
    return ev.clamp_min(0).flip(0)


def svd_discarded_energy(W: torch.Tensor, r: int, M: torch.Tensor | None = None) -> float:
    """E_svd = 1 - sum_{i <= r} sigma_i^2 / sum_i sigma_i^2 of W (M = None) or of W M^(1/2)."""
    s = weighted_spectrum(W, M)
    tot = float(s.sum())
    return float(1.0 - s[:int(r)].sum() / tot) if tot > 0 else float("nan")


def weighted_share(W: torch.Tensor, W_eq: torch.Tensor, M: torch.Tensor | None = None) -> float:
    """||(W - W_eq) M^(1/2)||^2 / ||W M^(1/2)||^2 (the breaking share s when M = None)."""
    W = W.to(torch.float64)
    D = W - W_eq.to(W)
    if M is None:
        num, den = float((D * D).sum()), float((W * W).sum())
    else:
        M = M.to(W)
        num, den = float(((D @ M) * D).sum()), float(((W @ M) * W).sum())
    return num / den if den > 0 else float("nan")


def applied_svd_error(W: torch.Tensor, r: int, M: torch.Tensor | None = None) -> float:
    """Weighted relative error of the plain rank-r truncation that the SVD baseline actually applies."""
    W = W.to(torch.float64)
    return weighted_share(W, svd_truncate(W, int(r)), M)


@torch.no_grad()
def collect_mlp_input_moments(adapter, loader, device, blocks=None, model: nn.Module | None = None) -> dict:
    """Second moments of the inputs of the folded fc1 and of fc2, per block, from one forward pass.

    fc1 (folded, so norm2's affine sits in W1): u = LayerNorm(x) without affine, computed from the
    input of norm2 with norm2's eps. fc2: its input, the GELU hidden activation. Sums over every
    token (prefix and patch tokens) of every image, in float64; M1[b] is d x d, M2[b] is h x h.
    """
    from .data import normalize_for

    m = adapter.model if model is None else model
    L = len(m.blocks)
    blocks = list(range(L)) if blocks is None else sorted({int(b) for b in blocks})
    d, h = adapter.embed_dim, adapter.mlp_hidden
    kw = dict(dtype=torch.float64, device=device)
    M1 = {b: torch.zeros(d, d, **kw) for b in blocks}
    M2 = {b: torch.zeros(h, h, **kw) for b in blocks}
    handles = []
    for b in blocks:
        blk = m.blocks[b]
        if not isinstance(blk.norm2, nn.LayerNorm):
            raise ValueError(f"block {b}: norm2 is {type(blk.norm2).__name__}, expected nn.LayerNorm")

        def pre_norm(mod, args, b=b):
            X = args[0].reshape(-1, d).to(torch.float64)
            U = F.layer_norm(X, (d,), None, None, mod.eps)
            M1[b] += U.T @ U

        def pre_fc2(mod, args, b=b):
            H = args[0].reshape(-1, h).to(torch.float64)
            M2[b] += H.T @ H

        handles.append(blk.norm2.register_forward_pre_hook(pre_norm))
        handles.append(blk.mlp.fc2.register_forward_pre_hook(pre_fc2))
    n_img = 0
    try:
        for imgs, _ in loader:
            x = normalize_for(adapter, imgs.to(device, non_blocking=True))
            adapter.forward_tokens(x, m)
            n_img += int(x.shape[0])
    finally:
        for hd in handles:
            hd.remove()
    return {"M1": M1, "M2": M2, "n_images": n_img, "n_tokens": n_img * adapter.n_tokens}


def block_rule_measures(blk: nn.Module, A_in, A_out, perm, M1=None, M2=None, M1_G=None) -> dict:
    """s and E_svd at matched parameters for one block, per matrix and as block means.

    Matched parameters: r = floor(eq / (h + d)) per matrix, eq the equivariant parameter count with
    the per-layer operators. Measures for W1 (folded fc1) and W2 (folded fc2):
      *_weight       plain weight energy;
      *_act          weights times the measured input second moment to the power 1/2 (M1, M2);
      E_*_act_plainsvd  activation-weighted error of the plain rank-r truncation actually applied;
      *_actG (fc1)   M1 replaced by the residual Gram G of the block's input layer (centred over
                     channels; LayerNorm's per-token scaling is ignored, so G stands in for the
                     second moment of LayerNorm(x) up to a global scale, which the ratios ignore).
    Block-level values are the means of the fc1 and fc2 fractions (both are unit-free).
    """
    from .surgery import svd_rank_for

    f = fold_mlp(blk)
    W1, W2 = f["W1"], f["W2"]
    h, d = W1.shape
    e = project_mlp_two(**f, A_in=A_in, A_out=A_out, perm=perm)
    trP = perm_trace(perm)
    eq1 = equivariant_param_count(h, d, trP, involution_trace(np.asarray(A_in)))
    eq2 = equivariant_param_count(h, d, trP, involution_trace(np.asarray(A_out)))
    r1, r2 = svd_rank_for(h, d, eq1), svd_rank_for(h, d, eq2)
    out = {"h": h, "d": d, "tr_P": trP, "tr_A_in": involution_trace(np.asarray(A_in)),
           "tr_A_out": involution_trace(np.asarray(A_out)), "eq_params_fc1": eq1, "eq_params_fc2": eq2,
           "svd_rank_fc1": r1, "svd_rank_fc2": r2}
    for tag, W, We, r, M, MG in (("fc1", W1, e["W1"], r1, M1, M1_G), ("fc2", W2, e["W2"], r2, M2, None)):
        out[f"s_{tag}_weight"] = weighted_share(W, We)
        out[f"E_{tag}_weight"] = svd_discarded_energy(W, r)
        if M is not None:
            Mt = M.to(device=W.device)
            out[f"s_{tag}_act"] = weighted_share(W, We, Mt)
            out[f"E_{tag}_act"] = svd_discarded_energy(W, r, Mt)
            out[f"E_{tag}_act_plainsvd"] = applied_svd_error(W, r, Mt)
        if MG is not None:
            Mg = torch.as_tensor(MG, dtype=torch.float64, device=W.device)
            out[f"s_{tag}_actG"] = weighted_share(W, We, Mg)
            out[f"E_{tag}_actG"] = svd_discarded_energy(W, r, Mg)
    out["s_weight"] = 0.5 * (out["s_fc1_weight"] + out["s_fc2_weight"])
    out["E_weight"] = 0.5 * (out["E_fc1_weight"] + out["E_fc2_weight"])
    return out


# ----------------------------------------------------------------------------------------------
# Runtime helpers shared by scripts/stage1_ablate.py and scripts/f4_rule.py
# ----------------------------------------------------------------------------------------------
class TinyProtocol:
    """Synthetic stand-in for mirror.protocol.Protocol (CPU tests): same loaders, same `index` layout
    (fit0..fit2, bank) and the same attributes used by fit_subset_rows."""

    ROLES = {"fit0": 2, "fit1": 2, "fit2": 2, "heldout": 2, "bank": 8, "query": 4}

    def __init__(self, img_size: int = 56, num_classes: int = 5, seed: int = 0, batch_size: int = 32,
                 stats_batch_size: int = 8):
        from .data import make_splits, synthetic_dataset

        images, labels = synthetic_dataset(num_classes, sum(self.ROLES.values()), img_size, seed=seed)
        sp = make_splits(labels, self.ROLES)
        self.train = self.val = images
        self.train_labels = self.val_labels = labels
        self.index = {f"{r}_rows": sp[r] for r in ("fit0", "fit1", "fit2", "bank")}
        self.heldout_idx, self.query_idx = sp["heldout"], sp["query"]
        self.num_classes = num_classes
        self.bs, self.sbs, self.nw, self.pin = batch_size, stats_batch_size, 0, False

    def _loader(self, rows, bs):
        from .data import make_loader

        return make_loader(self.train, self.train_labels, rows, bs, self.nw, self.pin)

    def fit_loader(self, seed: int):
        return self._loader(self.index[f"fit{seed}_rows"], self.sbs)

    def heldout_loader(self):
        return self._loader(self.heldout_idx, self.sbs)

    def bank_loader(self):
        return self._loader(self.index["bank_rows"], self.bs)

    def query_loader(self, n: int | None = None):
        rows = self.query_idx if n is None else self.query_idx[:int(n)]
        return self._loader(rows, self.bs)


def build_runtime(model_name: str, tiny: bool, device, cache_dir, batch_size: int = 128,
                  stats_batch_size: int = 16, num_workers: int = 4):
    """(adapter, protocol, num_classes). Tiny: random 4-block timm ViT and synthetic data on CPU."""
    if tiny:
        from .adapters import TimmTinyAdapter

        proto = TinyProtocol()
        return TimmTinyAdapter(img_size=56, seed=0).to(device=device), proto, proto.num_classes
    from .adapters_timm import build_model_adapter
    from .protocol import Protocol

    adapter = build_model_adapter(model_name, device=device)
    pin = torch.device(device).type == "cuda"
    return adapter, Protocol(cache_dir, batch_size, stats_batch_size, num_workers, pin), 1000


def fit_subset_rows(proto, n: int, fit_seed: int, order: str = "spread"):
    """Rows of the A3 fit subset of size n: (rows, fit_seed recorded, description).

    n <= one fit set: rows from fit<fit_seed>. order "first" takes the literal first n rows; these
    are class-sorted (2 per class), so n = 100 would cover only 50 classes. order "spread" (default)
    takes the first n of a fixed permutation (seed 1000 + fit_seed) instead, so the subsets are
    nested and spread over classes. n larger than one fit set: rows from fit0 + fit1 + fit2 pooled
    (6,000 images; the plan's 10,000 does not exist in the cache); recorded with fit seed -1.
    """
    per = np.asarray(proto.index[f"fit{fit_seed}_rows"])
    if n <= len(per):
        rows = per if order == "first" else np.random.default_rng(1000 + fit_seed).permutation(per)
        return np.sort(rows[:n]), int(fit_seed), f"fit{fit_seed} ({order}, first {n})"
    pooled = np.concatenate([np.asarray(proto.index[f"fit{s}_rows"]) for s in (0, 1, 2)])
    if n > len(pooled):
        raise ValueError(f"n = {n} exceeds the {len(pooled)} pooled fit images")
    rows = pooled if order == "first" else np.random.default_rng(1003).permutation(pooled)
    return np.sort(rows[:n]), -1, f"fit0+fit1+fit2 ({order}, first {n})"


def original_knn(adapter, bank_loader, query_loader, device, num_classes: int, model_name: str, amp,
                 cache_dir, n_query, tiny: bool, git_hash: str, k: int = 20, temperature: float = 0.07) -> dict:
    """Original-model k-NN, reusing cache/knn_original/<model>_v2_<amp>.json (as run_e1.py) for the
    full query set, or <model>_v2_<amp>_q<n>.json for a subsample. Tiny runs never use the cache."""
    from .knn import evaluate_knn

    name = f"{model_name}_v2_{amp}" + ("" if n_query is None else f"_q{int(n_query)}") + ".json"
    path = Path(cache_dir) / "knn_original" / name
    if not tiny and path.exists():
        c = json.loads(path.read_text())
        return {"top1": float(c["top1"]), "flip_consistency": float(c.get("flip_consistency", float("nan"))),
                "source": str(path)}
    r = evaluate_knn(adapter, bank_loader, query_loader, device, num_classes, k, temperature, amp=amp)
    out = {"top1": r["top1"], "flip_consistency": r["flip_consistency"]}
    if not tiny:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**out, "git": git_hash}))
    return {**out, "source": "computed" + ("" if tiny else f", cached to {path}")}


def load_or_make_structure(fit_run, tiny: bool, adapter, proto, device, out_dir, fit_seed=None,
                           top_k: int = 10) -> Structure:
    """The fit run's structure; with --tiny and no fit run, fit one on the synthetic fit set, save it
    in the run_e1.py layout under <out_dir>/tiny_fit/results/ and load it back through load_fit_run."""
    if fit_run:
        return load_fit_run(fit_run)
    if not tiny:
        raise SystemExit("--fit-run is required (a run_e1.py output directory, protocol v2)")
    seed = 0 if fit_seed is None else int(fit_seed)
    res_dir = Path(out_dir) / "tiny_fit" / "results"
    npz = res_dir / "e1_operators.npz"
    if not npz.exists():
        st = fit_structure(adapter, proto.fit_loader(seed), device, top_k=top_k, workers=1)
        save_structure(npz, st, with_moments=True)
        (res_dir / "e1_meta.json").write_text(json.dumps({
            "fit_seed": seed, "protocol": "tiny", "note": "tiny synthetic fit (CPU test only)",
            "config": {"stats": {"center": True}, "model": {"name": "tiny"}}}))
    return load_fit_run(res_dir.parent)


def resolve_fit_seed(cli_seed, struct: Structure) -> int:
    """Fit seed of the run; --fit-seed must agree with the run's own record when both exist."""
    run_seed = struct.meta.get("fit_seed")
    if cli_seed is not None and run_seed is not None and int(cli_seed) != int(run_seed):
        raise SystemExit(f"--fit-seed {cli_seed} disagrees with the fit run's fit seed {run_seed}")
    seed = cli_seed if cli_seed is not None else run_seed
    if seed is None:
        raise SystemExit("the fit run does not record its fit seed: pass --fit-seed")
    return int(seed)


@torch.no_grad()
def knn_top1(adapter, bank_loader, query_loader, device, num_classes: int, k: int = 20,
             temperature: float = 0.07, model=None, amp=None) -> float:
    """k-NN top-1 (%) without the mirrored-query pass (same features and vote as evaluate_knn)."""
    from .knn import extract_features, knn_predict

    bf, by = extract_features(adapter, bank_loader, device, model, amp=amp)
    qf, qy = extract_features(adapter, query_loader, device, model, amp=amp)
    return 100.0 * float((knn_predict(bf, by, qf, num_classes, k, temperature) == qy).float().mean())


class ResultsCsv:
    """CSV of result rows, rewritten after every row; `done(key)` makes runs resumable."""

    def __init__(self, path, key: list):
        import pandas as pd

        self.path, self.key = Path(path), list(key)
        self.rows = pd.read_csv(self.path).to_dict("records") if self.path.exists() else []
        self._done = {self._k(r) for r in self.rows}

    def _k(self, row) -> tuple:
        return tuple(_norm_key(row[c]) for c in self.key)

    def done(self, key: tuple) -> bool:
        return tuple(_norm_key(x) for x in key) in self._done

    def add(self, row: dict) -> None:
        import pandas as pd

        self.rows.append(row)
        self._done.add(self._k(row))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def _norm_key(x) -> str:
    if isinstance(x, (float, np.floating)) and math.isfinite(float(x)) and float(x) == int(x):
        return str(int(x))
    return str(x)

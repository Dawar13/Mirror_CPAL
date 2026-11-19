"""A01: the damage-minimising stream operator A*_w (weight-space fit, alternating with re-pairing).

FINAL_PHASE_PLAN.md A01 and PREREG_FINAL_PHASE.md Section 2.2. The current span operator A* (mirror/m1.py
fit_span_operator) is fitted to activations. What costs accuracy is the weight change that the projection of
Mirror-exact makes (mirror/m1.py build_vexact). This module fits the operator that makes that change smallest.

Conventions as in mirror.surgery, mirror.attention and mirror.m1: fc1.weight W1 is [h, d], fc2.weight W2 is [d, h],
tokens are rows, A is a symmetric orthogonal involution acting as x -> x A, the neuron involution P is stored as an
int64 permutation `perm` (P W1 = W1[perm], W2 P = W2[:, perm]). Blocks are 0-based. The span is blocks 0..k-1
(k = L: the full model), so the embeddings always belong to it.

Damage of the projection (every tensor as the fully folded model holds it, after materialise_pos_embed and
fold_layernorm_full(final_norm_affine=False), exactly the state build_vexact projects). Each projection is
X <- (X + T(X))/2 with T an orthogonal involution on that parameter, so the Frobenius change is
||X - T(X)||^2 / 4 = (||X||^2 - <X, T(X)>) / 2, and <X, T(X)> is linear in A: <X, T(X)> = tr(A M_X) with a
symmetric d x d matrix M_X. Read off the code:
  fc1 (surgery.project_mlp)        W1 <- (W1 + P W1 A)/2               M = W1^T P W1
  fc2                              W2 <- (W2 + A W2 P)/2               M = W2 P W2^T
  patch kernel (m1.project_embeddings)  W <- (W + A flip_w(W))/2       M = Wm Wf^T  (Wm = W as [d, c kh kw],
                                                                                     Wf = the width-flipped kernel)
  patch bias                       b <- (b + A b)/2                    M = b b^T
  position grid                    pos(r, c) <- (pos(r, c) + pos(r, W-1-c) A)/2
                                                                       M = sum_(r,c) pos(r, c)^T pos(r, W-1-c)
  CLS token and its position row   c <- (c + c A)/2                    M = c^T c
  registers (and their position rows, if any)  r_j <- (r_j + r_pi(j) A)/2   M = R^T R[pi]
Attention (optional, flag `attn_terms`, default OFF; mirror/attention.py project_attn, head maps sigma, B, C fixed):
  self-head h      Wq^h <- (Wq^h + B Wq^h A)/2 (k with B, v with C)    M = Wq^h^T B Wq^h (+ k, v alike)
                   Wo^h <- (Wo^h + A Wo^h C)/2                         M = Wo^h C Wo^h^T
  twin pair (h, g) the pair moves onto {W^g = B W^h A}; the change is ||W^g - B W^h A||^2 / 2, so with
                   E = ||W^h||^2 + ||W^g||^2 the form (E - tr(A M))/2 holds with
                   M = X + X^T, X = W^g^T B W^h (q, k; C for v), and for the output X = Wo^h C^T Wo^g^T.
  The twin gauge of build_vexact is an orthogonal change of the weights, so it does not change the damage.
Biases of the MLP and attention are left out, as in the plan's formula, so the objective covers the weight matrices
and the embeddings only (b1 and the q, k, v biases do not depend on A; b2 and bo do, and are not counted).
fc1 is not centred on its input side by the fold, so the fc1 term also counts the change of W1 1, which the centred
stream never reads; it depends only on the pairing (A 1 = 1), so it does not move A, only the re-pairing.

Objective. Terms t: one per converted block (fc1 and fc2 together), one for the patch embedding (kernel and bias),
one for the position grid, one for the class token (token and its position row), one for the registers (tokens and
their position rows), and optionally one per block for attention. With e_t the energy (sum of squared entries of the
term's tensors, folded model) and w_t a weight per kind (default 1 for every kind),
    damage(A, P) = sum_t w_t ||X_t - X_t,eq||^2 / e_t = sum_t w_t (1 - tr(A M_t) / e_t) / 2,
the sum of the terms' breaking shares (mirror.surgery.breaking_share). Minimising it is maximising
    J(A, P) = tr(A M(P)),   M(P) = sum_t w_t M_t / e_t,
so damage = (sum_t w_t - J) / 2. Every term matrix is symmetric.

Closed form for fixed pairings (`closed_form_operator`). The folded stream is centred, so every operator must fix
the all-ones vector (A 1 = 1). With u = 1/sqrt(d) and A u = u, tr(A M) = u^T M u + tr(A Mc) with Mc = Pc M Pc,
Pc = I - u u^T; the first part does not depend on A. On the complement of u (an exact orthonormal basis from a
Householder reflection) Mc = V diag(lam) V^T and A = u u^T + V diag(s) V^T:
  "free" (diagnostic): s = sign(lam) (0 counts as +1), the maximiser over all involutions with A 1 = 1;
  "matched" (pre-registered main mode): the m most negative lam get -1, m = d- of the source operator A*_k. This is
           the maximiser among involutions with exactly m odd directions: tr(A Mc) = tr(Mc) - 2 tr(Pi Mc) with Pi
           the odd projector, and the smallest tr(Pi Mc) over rank-m projectors is the sum of the m smallest
           eigenvalues (Ky Fan's minimum principle).

Re-pairing for a fixed A (`repair_block`). The block term is tr(A M_l(P)) / e_l = tr(P G_l) / e_l with
G_l = W1 A W1^T + W2^T A W2 (h x h); tr(P G) = sum_i G_ii + sum over pairs of (2 G_ij - G_ii - G_jj). This is the
existing maximum-weight involution matching (mirror.matching.candidate_edges and solve_matching: each neuron's top_k
candidates, positive gains, exact blossom on integer weights); G is rescaled so that the largest candidate gain maps to
1e7 before the integer rounding. In matched mode the pairs are cut (keep the largest gains) or filled greedily (largest
gain among unused neurons) to the block's source pair count, as mirror_family.top_pairs does. A block's new pairing
is accepted only if its term increases (blocks are independent for a fixed A), so J never decreases.

Alternation (`fit_weightspace`): round 0 fits A on the structure's data-driven pairings; each later round re-pairs
every block for the current A, then refits A; at most `max_rounds` (3) re-pairing rounds, stopping after the first
round in which fewer than `tol` (1%) of the span's neurons change partner. Every round logs J, the damage, d+ and d-,
the neurons that changed partner and the blocks accepted or rejected. In matched mode d- and every block's pair count
equal the source's, so the converted model has exactly the source's size (66,203,360 parameters for the main model).

Registers: `rematch_registers` repeats the register matching of mirror_family.fit_span_structure (m1.RegisterScores
over the stream layers 0..k of a paired pass of the fit images, m1.match_registers) under the new operator. The fit
itself uses the source's register pairing in the register term.

Head maps (sigma, B, C) come from activation moments and do not depend on A, so the structure's fitted maps are kept;
build_mirror_variant projects attention with the new A ("refitted given A, as now").

Output (`weightspace_structure`): the source SpanStructure with A_star[k] replaced by A*_w, perms of blocks 0..k-1
replaced by the final pairings, reg_perm[k] re-matched, heads, A_layers, the other A_star entries and the control
pairings unchanged, and meta["operator_fit"] recording the fit (mode, rounds, J per round, d+ and d-, the source
structure's sha256). Saved with mirror_family.save_span_structure, so build_mirror_variant and the healing code read
it unchanged. Note: the A_star entries for other span sizes stay the source's activation-fitted operators while the
pairings of blocks 0..k-1 are the weight-space ones; fit each span size it is used at with its own k.

Sources: no external sources; every formula is derived from the projection code in this repository
(mirror/m1.py project_embeddings and build_vexact, mirror/surgery.py project_mlp, mirror/attention.py project_attn,
mirror/fold.py) and checked numerically in tests/test_fp_weightspace.py.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .attention import fold_attn
from .fold import fold_layernorm_full
from .heal import exact_involution
from .m1 import RegisterScores, match_registers, materialise_pos_embed, register_info
from .matching import candidate_edges, solve_matching

KINDS = ("mlp", "patch", "pos", "cls", "reg", "attn")
DEFAULT_WEIGHTS = {k: 1.0 for k in KINDS}
MODES = ("matched", "free")
INT_SCALE = 1e7          # largest candidate gain after rescaling, before solve_matching rounds to integers

__all__ = ["KINDS", "DEFAULT_WEIGHTS", "MODES", "Term", "WeightTerms", "WeightSpaceFit", "folded_copy",
           "embedding_terms", "attention_terms", "collect_terms", "with_register_perm", "mlp_matrix",
           "objective_matrix", "term_values",
           "objective", "damage_of", "centre_matrix", "perp_basis", "closed_form_operator", "block_gram",
           "pair_value", "repair_block", "fit_weightspace", "rematch_registers", "weightspace_structure",
           "sha256_file", "d_minus_of", "n_pairs_of", "perm_from_pairs", "full_conversion_knn", "dry_run_adapter"]


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def _sym(M):
    return 0.5 * (M + M.T)


def d_minus_of(A) -> int:
    """Number of -1 eigenvalues of an orthogonal involution, from its trace."""
    A = np.asarray(A, dtype=np.float64)
    return int(round((A.shape[0] - float(np.trace(A))) / 2))


def n_pairs_of(perm) -> int:
    p = np.asarray(perm)
    return int((p != np.arange(len(p))).sum() // 2)


def perm_from_pairs(h: int, pairs) -> np.ndarray:
    perm = np.arange(h, dtype=np.int64)
    for a, b in pairs:
        perm[int(a)], perm[int(b)] = int(b), int(a)
    return perm


def _check_perm(perm, n: int) -> np.ndarray:
    perm = np.arange(n, dtype=np.int64) if perm is None else np.asarray(perm, dtype=np.int64)
    if perm.shape != (n,) or not np.array_equal(perm[perm], np.arange(n)):
        raise ValueError(f"expected an involution on {n} elements")
    return perm


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


# ----------------------------------------------------------------------------------------------
# Damage terms
# ----------------------------------------------------------------------------------------------
@dataclass
class Term:
    """One damage term: damage = (energy - tr(A M)) / 2 for fixed terms; MLP terms carry M(P) implicitly."""

    name: str
    kind: str
    energy: float
    M: np.ndarray | None = None          # d x d symmetric (None for MLP terms: M depends on the pairing)
    block: int | None = None
    tensors: tuple = ()                  # the parameters the term covers (for reports and tests)
    data: dict | None = None             # register term: the [R, d] rows, so M can be rebuilt for another pairing


@dataclass
class WeightTerms:
    """Everything the fit needs from the folded model: MLP weights of blocks 0..k-1 and the fixed terms."""

    k: int
    d: int
    h: int
    W1: list                             # float64 tensors [h, d] (folded fc1.weight), blocks 0..k-1
    W2: list                             # float64 tensors [d, h] (folded fc2.weight)
    mlp: list                            # one Term per block (M depends on the pairing)
    fixed: list                          # embedding (and attention) Terms
    weights: dict
    device: torch.device

    def w(self, term: Term) -> float:
        return float(self.weights.get(term.kind, 1.0))

    @property
    def terms(self) -> list:
        return list(self.mlp) + list(self.fixed)

    def weight_total(self) -> float:
        return float(sum(self.w(t) for t in self.terms if t.energy > 0))


@torch.no_grad()
def folded_copy(adapter, model: nn.Module | None = None) -> nn.Module:
    """Deep copy with the evaluation grid's positions materialised and every LayerNorm folded (the state that
    m1.build_vexact projects)."""
    m = copy.deepcopy(adapter.model if model is None else model)
    materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    return m.eval()


@torch.no_grad()
def embedding_terms(model: nn.Module, adapter, reg_perm=None) -> list:
    """Damage terms of m1.project_embeddings on a folded (materialised) model: "patch", "pos", "cls", "reg".
    The code paths (prefix rows of the position table, registers, CLS) follow project_embeddings line by line."""
    out = []
    pe = model.patch_embed.proj
    if not isinstance(pe, nn.Conv2d):
        raise NotImplementedError("patch_embed.proj must be Conv2d")
    W = pe.weight.detach().double().cpu()
    d = int(W.shape[0])
    Wm, Wf = W.reshape(d, -1), W.flip(-1).reshape(d, -1)
    M = Wm @ Wf.T
    e = float((W ** 2).sum())
    names = ["patch_embed.proj.weight"]
    if pe.bias is not None:
        b = pe.bias.detach().double().cpu()
        M = M + torch.outer(b, b)
        e += float((b ** 2).sum())
        names.append("patch_embed.proj.bias")
    out.append(Term("patch", "patch", e, _sym(M.numpy()), tensors=tuple(names)))

    H, Wg = adapter.grid_hw
    reg = register_info(model, adapter)
    R = int(reg["R"])
    perm = _check_perm(reg_perm, R)
    pidx = torch.as_tensor(perm)
    has_cls = getattr(model, "cls_token", None) is not None
    cls_M, cls_e, cls_names = torch.zeros(d, d, dtype=torch.float64), 0.0, []
    reg_M, reg_e, reg_names, reg_rows = torch.zeros(d, d, dtype=torch.float64), 0.0, [], []
    pos = getattr(model, "pos_embed", None)
    if isinstance(pos, torch.Tensor):
        P = pos.detach().double().cpu()[0]
        n_pre = P.shape[0] - H * Wg
        allowed = {0, int(has_cls)} | ({int(has_cls) + R} if (has_cls and R) else set())
        if n_pre not in allowed:
            raise NotImplementedError(f"position embedding has {n_pre} prefix rows; expected one of {sorted(allowed)}")
        grid = P[n_pre:].reshape(H, Wg, -1)
        G0, Gf = grid.reshape(-1, d), grid.flip(1).reshape(-1, d)
        out.append(Term("pos", "pos", float((G0 ** 2).sum()), _sym((G0.T @ Gf).numpy()),
                        tensors=("pos_embed (patch grid)",)))
        if n_pre >= 1 and has_cls:
            p0 = P[0]
            cls_M += torch.outer(p0, p0)
            cls_e += float((p0 ** 2).sum())
            cls_names.append("pos_embed (CLS row)")
        if n_pre > 1:
            Rp = P[1:1 + R]
            reg_M += Rp.T @ Rp[pidx]
            reg_e += float((Rp ** 2).sum())
            reg_names.append("pos_embed (register rows)")
            reg_rows.append(Rp.numpy().copy())
    if has_cls:
        c = model.cls_token.detach().double().cpu().reshape(-1, d)
        cls_M += c.T @ c
        cls_e += float((c ** 2).sum())
        cls_names.append("cls_token")
    if R:
        r = getattr(model, reg["name"]).detach().double().cpu()[0]
        reg_M += r.T @ r[pidx]
        reg_e += float((r ** 2).sum())
        reg_names.append(str(reg["name"]))
        reg_rows.append(r.numpy().copy())
    if cls_names:
        out.append(Term("cls", "cls", cls_e, _sym(cls_M.numpy()), tensors=tuple(cls_names)))
    if reg_names:
        out.append(Term("reg", "reg", reg_e, _sym(reg_M.numpy()), tensors=tuple(reg_names),
                        data={"R": R, "rows": reg_rows, "perm": perm}))
    return out


def with_register_perm(terms: "WeightTerms", reg_perm) -> "WeightTerms":
    """The same terms with the register term rebuilt for another register pairing (M = sum_X X^T X[pi])."""
    fixed = []
    for t in terms.fixed:
        if t.kind == "reg" and t.data is not None:
            perm = _check_perm(reg_perm, int(t.data["R"]))
            M = sum(X.T @ X[perm] for X in t.data["rows"])
            t = dataclasses.replace(t, M=_sym(M), data={**t.data, "perm": perm})
        fixed.append(t)
    return dataclasses.replace(terms, fixed=fixed)


@torch.no_grad()
def attention_terms(model: nn.Module, blocks, heads) -> list:
    """Attention damage terms (one per block) for the fitted head maps (see the module docstring); folded model."""
    out = []
    for b in blocks:
        b = int(b)
        f = fold_attn(model.blocks[b])
        hm = heads[b]
        H, dh = hm.H, hm.dh
        d = int(f["Wq"].shape[1])
        s = {key: f[key].detach().double().cpu().reshape(H, dh, d) for key in ("Wq", "Wk", "Wv")}
        Wo = f["Wo"].detach().double().cpu().reshape(d, H, dh)
        e = float(sum((s[key] ** 2).sum() for key in s) + (Wo ** 2).sum())
        M = torch.zeros(d, d, dtype=torch.float64)
        sig = np.asarray(hm.sigma)
        for h in range(H):
            g = int(sig[h])
            Bm = torch.as_tensor(np.asarray(hm.B[h], dtype=np.float64))
            Cm = torch.as_tensor(np.asarray(hm.C[h], dtype=np.float64))
            if g == h:
                for key, X in (("Wq", Bm), ("Wk", Bm), ("Wv", Cm)):
                    Wh = s[key][h]
                    M += Wh.T @ X @ Wh
                M += Wo[:, h] @ Cm @ Wo[:, h].T
            elif g > h:
                for key, X in (("Wq", Bm), ("Wk", Bm), ("Wv", Cm)):
                    Xm = s[key][g].T @ X @ s[key][h]
                    M += Xm + Xm.T
                Xo = Wo[:, h] @ Cm.T @ Wo[:, g].T
                M += Xo + Xo.T
        out.append(Term(f"attn.{b}", "attn", e, _sym(M.numpy()), block=b,
                        tensors=(f"blocks.{b}.attn.qkv.weight", f"blocks.{b}.attn.proj.weight")))
    return out


@torch.no_grad()
def collect_terms(adapter, k: int | None = None, reg_perm=None, heads=None, attn_terms: bool = False,
                  weights: dict | None = None, device="cpu", model: nn.Module | None = None) -> WeightTerms:
    """Fold a copy of the model and collect the damage terms of a span of blocks 0..k-1 (default: every block)."""
    m = folded_copy(adapter, model)
    L = len(m.blocks)
    k = L if k is None else int(k)
    if not 1 <= k <= L:
        raise ValueError(f"span size k must be in 1..{L}")
    dev = torch.device(device)
    W1, W2, mlp = [], [], []
    for b in range(k):
        blk = m.blocks[b]
        w1 = blk.mlp.fc1.weight.detach().to(device=dev, dtype=torch.float64).clone()
        w2 = blk.mlp.fc2.weight.detach().to(device=dev, dtype=torch.float64).clone()
        W1.append(w1)
        W2.append(w2)
        mlp.append(Term(f"mlp.{b}", "mlp", float((w1 ** 2).sum() + (w2 ** 2).sum()), None, block=b,
                        tensors=(f"blocks.{b}.mlp.fc1.weight", f"blocks.{b}.mlp.fc2.weight")))
    fixed = embedding_terms(m, adapter, reg_perm)
    if attn_terms:
        if heads is None:
            raise ValueError("attention terms need the head maps")
        fixed += attention_terms(m, range(k), heads)
    h, d = (int(W1[0].shape[0]), int(W1[0].shape[1]))
    wts = dict(DEFAULT_WEIGHTS)
    if weights:
        bad = set(weights) - set(KINDS)
        if bad:
            raise ValueError(f"unknown term kinds {sorted(bad)}; choose from {KINDS}")
        wts.update({str(key): float(v) for key, v in weights.items()})
    del m
    return WeightTerms(k=k, d=d, h=h, W1=W1, W2=W2, mlp=mlp, fixed=fixed, weights=wts, device=dev)


def mlp_matrix(terms: WeightTerms, b: int, perm) -> torch.Tensor:
    """M_l(P) = W1^T P W1 + W2 P W2^T (d x d, float64, on the terms' device)."""
    p = torch.as_tensor(np.asarray(perm, dtype=np.int64), device=terms.device)
    W1, W2 = terms.W1[b], terms.W2[b]
    M = W1.T @ W1[p] + W2[:, p] @ W2.T
    return 0.5 * (M + M.T)


def objective_matrix(terms: WeightTerms, perms) -> np.ndarray:
    """M(P) = sum_t w_t M_t / e_t (d x d, float64, not centred)."""
    M = torch.zeros(terms.d, terms.d, dtype=torch.float64, device=terms.device)
    for b, t in enumerate(terms.mlp):
        w = terms.w(t)
        if w and t.energy > 0:
            M += (w / t.energy) * mlp_matrix(terms, b, perms[b])
    out = M.cpu().numpy()
    for t in terms.fixed:
        w = terms.w(t)
        if w and t.energy > 0:
            out = out + (w / t.energy) * t.M
    return _sym(out)


def term_values(terms: WeightTerms, A, perms) -> list:
    """Per term: energy, damage (Frobenius change squared), breaking share = damage / energy, weight and the term's
    contribution w tr(A M_t) / e_t to J."""
    A64 = np.asarray(A, dtype=np.float64)
    At = torch.as_tensor(A64, device=terms.device)
    rows = []

    def row(t, tr):
        w = terms.w(t)
        dmg = 0.5 * (t.energy - tr)
        share = dmg / t.energy if t.energy > 0 else float("nan")
        return {"name": t.name, "kind": t.kind, "block": t.block, "energy": float(t.energy), "damage": float(dmg),
                "share": float(share), "weight": w,
                "value": float(w * tr / t.energy) if (w and t.energy > 0) else 0.0}

    for b, t in enumerate(terms.mlp):
        rows.append(row(t, float((At * mlp_matrix(terms, b, perms[b])).sum())))
    for t in terms.fixed:
        rows.append(row(t, float(np.sum(A64 * t.M))))
    return rows


def objective(terms: WeightTerms, A, perms) -> float:
    """J(A, P) = tr(A M(P))."""
    return float(sum(r["value"] for r in term_values(terms, A, perms)))


def damage_of(terms: WeightTerms, A, perms) -> float:
    """Weighted sum of the terms' breaking shares, (sum_t w_t - J) / 2."""
    return float(sum(r["weight"] * r["share"] for r in term_values(terms, A, perms)
                     if r["weight"] and np.isfinite(r["share"])))


# ----------------------------------------------------------------------------------------------
# Closed form for fixed pairings
# ----------------------------------------------------------------------------------------------
def centre_matrix(M) -> np.ndarray:
    """Pc M Pc with Pc = I - 11^T / d."""
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def perp_basis(d: int) -> np.ndarray:
    """Orthonormal basis (d x (d-1)) of the complement of the all-ones vector: columns 2..d of the Householder
    reflection that maps e_1 to u = 1/sqrt(d)."""
    if d < 2:
        raise ValueError("d must be at least 2")
    u = np.full(d, 1.0 / math.sqrt(d))
    v = -u.copy()
    v[0] += 1.0
    Hh = np.eye(d) - 2.0 * np.outer(v, v) / float(v @ v)
    return Hh[:, 1:]


def closed_form_operator(M, mode: str = "free", d_minus: int | None = None, centre: bool = True,
                         zero_tol: float = 1e-10) -> tuple:
    """argmax of tr(A M) over symmetric orthogonal involutions (with A 1 = 1 when `centre`).

    mode "free": s = sign of the eigenvalue (|lam| <= zero_tol max|lam| counts as +1);
    mode "matched": exactly `d_minus` odd directions, the most negative eigenvalues.
    Returns (A [d, d] float64, info) with info: mode, d_plus, d_minus, J = tr(A M), eigvals (ascending; of Mc on
    the complement of 1 when centred) and margin (matched: gap between the last odd and the first even eigenvalue;
    free: the smallest |eigenvalue|; 0 means the maximiser is not unique)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    M = _sym(np.asarray(M, dtype=np.float64))
    d = M.shape[0]
    if centre:
        Q = perp_basis(d)
        lam, U = np.linalg.eigh(_sym(Q.T @ M @ Q))
        V = Q @ U
    else:
        lam, V = np.linalg.eigh(M)
    n = len(lam)
    if mode == "free":
        scale = float(np.max(np.abs(lam))) if n else 0.0
        s = np.where(lam < -zero_tol * scale, -1.0, 1.0)
        margin = float(np.min(np.abs(lam))) if n else float("nan")
    else:
        if d_minus is None or not 0 <= int(d_minus) <= n:
            raise ValueError(f"matched mode needs 0 <= d_minus <= {n}, got {d_minus}")
        m = int(d_minus)
        s = np.ones(n)
        s[np.argsort(lam, kind="stable")[:m]] = -1.0
        srt = np.sort(lam)
        margin = float(srt[m] - srt[m - 1]) if 0 < m < n else float("inf")
    A = (V * s) @ V.T
    if centre:
        u = np.full(d, 1.0 / math.sqrt(d))
        A = A + np.outer(u, u)
    A = _sym(A)
    dm = int((s < 0).sum())
    info = {"mode": mode, "d_plus": int(d - dm), "d_minus": dm, "J": float(np.sum(A * M)), "eigvals": lam,
            "margin": margin, "centred": bool(centre)}
    return A, info


# ----------------------------------------------------------------------------------------------
# Re-pairing for a fixed operator
# ----------------------------------------------------------------------------------------------
def block_gram(terms: WeightTerms, b: int, A) -> np.ndarray:
    """G_l = W1 A W1^T + W2^T A W2 (h x h, float64): tr(P G_l) = tr(A M_l(P))."""
    At = torch.as_tensor(np.asarray(A, dtype=np.float64), device=terms.device)
    W1, W2 = terms.W1[b], terms.W2[b]
    G = (W1 @ At) @ W1.T + W2.T @ (At @ W2)
    return _sym(G.cpu().numpy())


def pair_value(G: np.ndarray, perm) -> float:
    """tr(P G) = sum_i G[i, perm[i]]."""
    perm = np.asarray(perm, dtype=np.int64)
    return float(G[np.arange(len(perm)), perm].sum())


def _edges(G: np.ndarray, top_k: int):
    i, j, w = candidate_edges(G, top_k)
    scale = INT_SCALE / float(w.max()) if len(w) and float(w.max()) > 0 else 1.0
    return i, j, w, scale


def _solve(args):
    """Worker job: exact matching on one block's candidate edges (empty edge set: no pairs)."""
    i, j, w, scale = args
    return solve_matching(i, j, w, scale) if len(w) else []


def _cut_or_fill(G: np.ndarray, pairs, n_pairs: int) -> list:
    """Exactly n_pairs pairs: the largest gains 2 G_ij - G_ii - G_jj of `pairs`, or `pairs` filled greedily by the
    largest gain among unused neurons (as mirror_family.top_pairs, on gains)."""
    from .mirror_family import top_pairs

    dg = np.diag(G)
    gains = 2.0 * G - dg[:, None] - dg[None, :]
    return top_pairs(pairs, gains, int(n_pairs))


def repair_block(G: np.ndarray, n_pairs: int | None = None, top_k: int = 10) -> np.ndarray:
    """Involution maximising tr(P G) by the max-weight matching on each neuron's top_k candidates (exact blossom),
    cut or filled to n_pairs pairs when given. Returns the permutation."""
    G = _sym(np.asarray(G, dtype=np.float64))
    pairs = _solve(_edges(G, top_k))
    if n_pairs is not None:
        pairs = _cut_or_fill(G, pairs, n_pairs)
    return perm_from_pairs(G.shape[0], pairs)


def _repair_round(terms: WeightTerms, A, perms, n_pairs, top_k: int, workers: int) -> tuple:
    """One re-pairing step for every block for the fixed operator A. A block's new pairing is accepted only if its
    term tr(P G_l) increases. Returns (new perms, per-block records). G_l is formed twice per block (candidates, then
    evaluation) so that only one h x h matrix is held at a time."""
    k = terms.k
    jobs = [_edges(block_gram(terms, b, A), top_k) for b in range(k)]
    if workers > 1 and k > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), k)) as ex:
            pair_lists = list(ex.map(_solve, jobs))
    else:
        pair_lists = [_solve(job) for job in jobs]
    del jobs
    new_perms, recs = [], []
    for b in range(k):
        G = block_gram(terms, b, A)
        pairs = pair_lists[b]
        if n_pairs is not None:
            pairs = _cut_or_fill(G, pairs, n_pairs[b])
        cand = perm_from_pairs(terms.h, pairs)
        old = np.asarray(perms[b], dtype=np.int64)
        v_old, v_new = pair_value(G, old), pair_value(G, cand)
        accept = bool((v_new - v_old) > 1e-12 * max(1.0, abs(v_old) + abs(v_new)) and not np.array_equal(cand, old))
        new = cand if accept else old.copy()
        new_perms.append(new)
        recs.append({"block": b, "accepted": accept, "value_old": v_old, "value_new": v_new,
                     "neurons_changed": int((new != old).sum()), "n_pairs": n_pairs_of(new)})
        del G
    return new_perms, recs


# ----------------------------------------------------------------------------------------------
# The alternation
# ----------------------------------------------------------------------------------------------
@dataclass
class WeightSpaceFit:
    A: np.ndarray                     # A*_w (exact involution, A 1 = 1)
    perms: list                       # final pairings of blocks 0..k-1
    mode: str
    k: int
    d_plus: int
    d_minus: int
    J: float
    J_source: float                   # J of the source operator with the source pairings
    damage: float                     # weighted sum of breaking shares at the result
    damage_source: float
    rounds: list
    stopped: str
    terms_source: list                # term_values rows, source operator and pairings
    terms_final: list
    M: np.ndarray                     # objective matrix at the final pairing (not centred)
    eigvals: np.ndarray               # eigenvalues of the centred M on the complement of 1 (ascending)
    config: dict = field(default_factory=dict)

    @property
    def S(self) -> np.ndarray:
        """The centred objective matrix that defines A*_w (for the energy-weighted agreement of mirror.stability)."""
        return centre_matrix(self.M)


def fit_weightspace(adapter, structure, mode: str = "matched", k: int | None = None, *, max_rounds: int = 3,
                    tol: float = 0.01, top_k: int = 10, workers: int = 1, attn_terms: bool = False,
                    weights: dict | None = None, device="cpu", terms: WeightTerms | None = None,
                    log=None) -> WeightSpaceFit:
    """Fit A*_w for the span blocks 0..k-1 (default the full model) of `structure` (a mirror_family.SpanStructure),
    starting from its data-driven pairings (see the module docstring)."""
    say = log if log is not None else (lambda msg: None)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    L = structure.depth
    k = L if k is None else int(k)
    if not 1 <= k <= L:
        raise ValueError(f"span size k must be in 1..{L}")
    t0 = time.time()
    A_src = exact_involution(structure.A_star[k])
    reg_src = np.asarray(structure.reg_perm[k], dtype=np.int64)
    if terms is None:
        terms = collect_terms(adapter, k, reg_perm=reg_src, heads=structure.heads, attn_terms=attn_terms,
                              weights=weights, device=device)
    if terms.k != k:
        raise ValueError(f"terms are for k = {terms.k}, not {k}")
    perms = [np.asarray(structure.perms[b], dtype=np.int64).copy() for b in range(k)]
    if any(len(p) != terms.h for p in perms):
        raise ValueError("structure pairings do not match the model's hidden width")
    n_pairs_src = [n_pairs_of(p) for p in perms]
    dm_src = d_minus_of(A_src)
    target = dm_src if mode == "matched" else None
    n_neurons = k * terms.h

    terms_source = term_values(terms, A_src, perms)
    J_src = float(sum(r["value"] for r in terms_source))
    M = objective_matrix(terms, perms)
    A, info = closed_form_operator(M, mode, target)
    A = exact_involution(A)
    J = float(np.sum(A * M))
    wt = terms.weight_total()
    rounds = [{"round": 0, "J_after_pairing": J_src, "J": J, "damage": 0.5 * (wt - J), "d_plus": info["d_plus"],
               "d_minus": info["d_minus"], "neurons_changed": 0, "share_changed": 0.0, "blocks_accepted": [],
               "blocks_rejected": [], "n_pairs": list(n_pairs_src), "margin": info["margin"],
               "note": "A fitted on the source (data-driven) pairings; J_after_pairing is J of the source operator"}]
    say(f"A*_w ({mode}, k = {k}) round 0: J {J_src:.6f} (source A*) -> {J:.6f}, d- {info['d_minus']}")
    stopped = f"max_rounds ({max_rounds})"
    for r in range(1, int(max_rounds) + 1):
        new_perms, recs = _repair_round(terms, A, perms, n_pairs_src if mode == "matched" else None, top_k, workers)
        changed = int(sum(rec["neurons_changed"] for rec in recs))
        J_pair = objective(terms, A, new_perms)
        if J_pair < J - 1e-9 * max(1.0, abs(J)):
            raise AssertionError(f"re-pairing lowered J ({J} -> {J_pair})")
        perms = new_perms
        M = objective_matrix(terms, perms)
        A_new, info = closed_form_operator(M, mode, target)
        A_new = exact_involution(A_new)
        J_new = float(np.sum(A_new * M))
        if J_new < J_pair - 1e-9 * max(1.0, abs(J_pair)):
            raise AssertionError(f"refitting A lowered J ({J_pair} -> {J_new})")
        A, J = A_new, J_new
        share = changed / n_neurons
        rounds.append({"round": r, "J_after_pairing": J_pair, "J": J, "damage": 0.5 * (wt - J),
                       "d_plus": info["d_plus"], "d_minus": info["d_minus"], "neurons_changed": changed,
                       "share_changed": share, "blocks_accepted": [rec["block"] for rec in recs if rec["accepted"]],
                       "blocks_rejected": [rec["block"] for rec in recs if not rec["accepted"]],
                       "n_pairs": [n_pairs_of(p) for p in perms], "margin": info["margin"]})
        say(f"A*_w ({mode}, k = {k}) round {r}: J {J_pair:.6f} after re-pairing, {J:.6f} after refitting; "
            f"{changed} neurons changed partner ({100 * share:.2f}%), d- {info['d_minus']}")
        if share < tol:
            stopped = f"converged (fewer than {100 * tol:g}% of neurons changed partner in round {r})"
            break
    terms_final = term_values(terms, A, perms)
    lam = np.asarray(info["eigvals"])
    dm = d_minus_of(A)
    if mode == "matched" and (dm != dm_src or [n_pairs_of(p) for p in perms] != n_pairs_src):
        raise AssertionError("matched mode changed d- or a block's pair count")
    cfg = {"mode": mode, "k": k, "max_rounds": int(max_rounds), "tol": float(tol), "top_k": int(top_k),
           "attn_terms": any(t.kind == "attn" for t in terms.fixed), "weights": dict(terms.weights),
           "int_scale": INT_SCALE,
           "terms": [{"name": t.name, "kind": t.kind, "tensors": list(t.tensors)} for t in terms.terms],
           "source_d_minus": dm_src, "source_n_pairs": n_pairs_src, "source_reg_perm": reg_src.tolist(),
           "seconds": round(time.time() - t0, 1)}
    return WeightSpaceFit(A=A, perms=perms, mode=mode, k=k, d_plus=int(A.shape[0] - dm), d_minus=dm, J=J,
                          J_source=J_src, damage=0.5 * (wt - J), damage_source=0.5 * (wt - J_src), rounds=rounds,
                          stopped=stopped, terms_source=terms_source, terms_final=terms_final, M=M, eigvals=lam,
                          config=cfg)


# ----------------------------------------------------------------------------------------------
# Registers, output structure
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def rematch_registers(adapter, loader, device, A, k: int | None = None, margin: float = 0.0) -> tuple:
    """Register involution under A for the span 0..k: m1.RegisterScores over stream layers 0..k of one paired pass
    of `loader` through the original model, matched with m1.match_registers (as mirror_family.fit_span_structure).
    Returns (reg_perm int64 [R], cosine matrix [R, R] or None without registers)."""
    from .stats import run_paired

    L = adapter.depth
    k = L if k is None else int(k)
    reg = register_info(adapter.model, adapter)
    if not reg["R"]:
        return np.zeros(0, dtype=np.int64), None
    rs = RegisterScores({k: (np.asarray(A, dtype=np.float64), k)}, L + 1, reg["R"], reg["start"], device)
    run_paired(adapter, loader, device, residual_cb=rs.update, desc="register scores")
    S = rs.cosines(k, range(k + 1))
    return np.asarray(match_registers(S, margin), dtype=np.int64), S


def weightspace_structure(src, fit: WeightSpaceFit, reg_perm=None, *, source_path=None, source_sha256=None,
                          register_source: str = "", extra: dict | None = None):
    """The source SpanStructure with A_star[k], the pairings of blocks 0..k-1 and reg_perm[k] replaced (see the
    module docstring); meta["operator_fit"] records the fit."""
    k = fit.k
    A = exact_involution(fit.A)
    reg = (np.asarray(src.reg_perm[k], dtype=np.int64) if reg_perm is None
           else np.asarray(reg_perm, dtype=np.int64))
    if len(reg) != len(src.reg_perm[k]):
        raise ValueError("register pairing length differs from the source's")
    A_star = {int(key): np.asarray(v, dtype=np.float64) for key, v in src.A_star.items()}
    A_star[k] = A
    perms = [np.asarray(p, dtype=np.int64).copy() for p in src.perms]
    for b in range(k):
        perms[b] = np.asarray(fit.perms[b], dtype=np.int64).copy()
    reg_all = {int(key): np.asarray(v, dtype=np.int64) for key, v in src.reg_perm.items()}
    reg_all[k] = reg
    meta = json.loads(json.dumps(src.meta, default=str))
    if isinstance(meta.get("n_pairs"), list):
        meta["n_pairs"] = [n_pairs_of(p) for p in perms]
    meta.setdefault("A_star", {})[str(k)] = {"d_plus": int(fit.d_plus), "d_minus": int(fit.d_minus), "n_zero": 0}
    meta.setdefault("register_pairs", {})[str(k)] = n_pairs_of(reg) if len(reg) else 0
    meta["operator_fit"] = _jsonable({
        "method": "weight-space damage minimisation (A01, mirror/fit_weightspace.py)",
        "mode": fit.mode, "k": k, "d_plus": fit.d_plus, "d_minus": fit.d_minus,
        "J": fit.J, "J_source": fit.J_source, "damage": fit.damage, "damage_source": fit.damage_source,
        "J_rounds": [r["J"] for r in fit.rounds], "rounds": fit.rounds, "stopped": fit.stopped,
        "n_pairs_source": fit.config.get("source_n_pairs"), "n_pairs": [n_pairs_of(p) for p in fit.perms],
        "source_structure": None if source_path is None else str(source_path), "source_sha256": source_sha256,
        "source_d_minus": fit.config.get("source_d_minus"), "register_pairing": register_source or
        ("kept from the source structure" if reg_perm is None else "re-matched under A*_w"),
        "register_perm": reg.tolist(), "config": fit.config,
        "breaking_shares_source": {r["name"]: r["share"] for r in fit.terms_source},
        "breaking_shares": {r["name"]: r["share"] for r in fit.terms_final},
        "note": (f"only A_star[{k}], the pairings of blocks 0..{k - 1} and reg_perm[{k}] are replaced; other span "
                 "sizes keep the source's activation-fitted operators"),
        **(extra or {})})
    return dataclasses.replace(src, A_star=A_star, perms=perms, reg_perm=reg_all, meta=meta)


# ----------------------------------------------------------------------------------------------
# Dry runs and tests
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def dry_run_adapter(seed: int = 0, embed_dim: int = 32, depth: int = 3, num_heads: int = 4, reg_tokens: int = 2,
                    img_size: int = 56, dtype=torch.float32):
    """Tiny random timm ViT with register tokens for CPU dry runs and tests (no download): LayerNorm affines,
    LayerScale, biases, CLS, registers and positions are randomised, as in tests/test_m1_run.py, so folding and
    projection are not trivially exact. Uses mirror.adapters.TimmTinyAdapter's token preparation."""
    from timm.models.vision_transformer import VisionTransformer

    from .adapters import TimmTinyAdapter, ViTAdapter, _randomise_affine

    torch.manual_seed(seed)
    model = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              init_values=1e-5, reg_tokens=reg_tokens)
    _randomise_affine(model, seed)
    g = torch.Generator().manual_seed(seed + 1)
    for name in ("cls_token", "reg_token", "pos_embed"):
        p = getattr(model, name, None)
        if isinstance(p, torch.Tensor):
            p.copy_(0.5 * torch.randn(p.shape, generator=g) + 0.2)
    model = model.to(dtype)
    ad = TimmTinyAdapter.__new__(TimmTinyAdapter)
    ViTAdapter.__init__(ad, model, img_size=img_size, patch_size=14, n_prefix_tokens=int(model.num_prefix_tokens))
    return ad


# ----------------------------------------------------------------------------------------------
# Zero-shot evaluation of a conversion
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def full_conversion_knn(adapter, structure, bank_loader, query_loader, device, num_classes: int, *, k: int | None = None,
                        knn_k: int = 20, temperature: float = 0.07, amp=None) -> dict:
    """Mirror-exact conversion of blocks 0..k-1 (default the full model; MLP and attention) with
    mirror_family.build_mirror_variant, then mirror.knn.evaluate_knn (weighted k-NN, k = 20, T = 0.07; bf16 autocast
    when amp="bf16" on CUDA). Returns top-1, flip consistency, per-image predictions and parameter counts."""
    from . import mirror_family as MF
    from .knn import evaluate_knn

    L = adapter.depth
    span = L if k is None else int(k)
    model, info = MF.build_mirror_variant(adapter, "mirror_exact", range(span), MF.COMPONENTS, structure)
    r = evaluate_knn(adapter, bank_loader, query_loader, device, num_classes, knn_k, temperature, model=model, amp=amp)
    ds = query_loader.dataset
    labels = np.asarray(ds.labels)[np.asarray(ds.indices)]
    n_orig = int(sum(p.numel() for p in adapter.model.parameters()))
    params_model = n_orig - int(info["span_dense"]) + int(info["params_matrices"]) + int(info["params_basis_change"])
    del model
    return {"top1": float(r["top1"]), "flip_consistency": float(r["flip_consistency"]), "n_bank": int(r["n_bank"]),
            "n_query": int(r["n_query"]), "pred": r["pred"].numpy(), "pred_flip": r["pred_flip"].numpy(),
            "labels": labels, "params_matrices": int(info["params_matrices"]),
            "params_basis_change": int(info["params_basis_change"]), "params_model": int(params_model),
            "params_original": n_orig, "params_frac": params_model / n_orig, "span": span,
            "d_plus": int(info["build"].get("d_plus", -1)) if info.get("build") else None, "amp": amp,
            "knn_k": int(knn_k), "temperature": float(temperature)}

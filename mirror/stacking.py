"""CORE-03: two axes of parsimony. Low rank inside Mirror-exact (PROJECT_NEXT_PHASE.md, Section 6, CORE-03;
PREREG_NEXT_PHASE.md, CORE-03, P3.1).

Mirror-exact (mirror/mirror_family.py, method "mirror_exact") projects every converted matrix onto the subspace that
commutes with the mirror. This module then applies a truncated SVD separately to the even block and to the odd block of
each projected matrix, with ranks allocated so that the converted span holds a given number of parameters. The result
is still exactly mirror-equivariant (the full model stays exactly flip-invariant), so the two kinds of saving stack.

Conventions: torch Linear weights W are [out, in]; blocks are 0-based in code (the plan's "block 1" is index 0);
matrices are the folded weights of surgery.fold_mlp and attention.fold_attn (on the fully folded Mirror-exact model
these are the stored weights themselves).

Block structure (from mirror/efficient.py, build_mirror_mlp and build_mirror_attention)
----------------------------------------------------------------------------------------
Every projected matrix W satisfies P_out W = W P_in for two orthogonal symmetric involutions:

    matrix  shape   P_out                          P_in
    fc1     [h, d]  P  (neuron pairing, h x h)     A* (span operator, d x d)
    fc2     [d, h]  A*                             P
    wq, wk  [D, d]  P_B (head involution, D x D)   A*
    wv      [D, d]  P_C                            A*
    wo      [d, D]  A*                             P_C

P is the permutation matrix of the twin pairing ((P W1)[i] = W1[perm[i]]). P_B and P_C act on the D = H dh query, key
and value channels: block (sigma(h), h) of P_B is B[h] (C[h] for P_C), with the head maps valid after surgery (the
twin gauge sets B = C = I on twin heads, so P_B swaps the two heads of a twin pair; on a self-head s it is the
symmetric involution B[s]). The relations follow from attention.project_attn: W^g = B[h] W^h A* on twin heads,
W^s = B[s] W^s A* on self-heads, Wo^g = A* Wo^h C[h]^T and Wo^s = A* Wo^s C[s].

Each involution splits its space into a +1 part (even) and a -1 part (odd), with orthonormal bases E+ and E-:
    A*    the eigenbasis Q = [Q+, Q-] of mirror/efficient.py (eig_basis): d+ even and d- odd stream directions;
    P     E+ = (e_i + e_j)/sqrt(2) for every twin pair (i, j) and e_s for every self-twin s (h+ = pairs + self-twins),
          E- = (e_i - e_j)/sqrt(2) for every twin pair (h- = pairs);
    P_B   E+ = (e_h + e_g)/sqrt(2) (x) R^dh for twin heads and the +1 eigenvectors of B[s] for self-heads,
          D_B+ = (D + sum_s tr B[s]) / 2; E- likewise (P_C with C).
Because P_out W = W P_in, W maps even to even and odd to odd: in these bases W is block diagonal,

    E_out+^T W E_in+  (even block, m+ x n+)      E_out-^T W E_in-  (odd block, m- x n-)
    E_out+^T W E_in- = 0                         E_out-^T W E_in+ = 0,

with m+ n+ + m- n- = the Mirror-exact count of the matrix (for fc1 (h+ d+ + h- d-) = (h d + tr P tr A*) / 2,
surgery.equivariant_param_count; for attention, attention.attention_free_params). In the notation of
efficient.py's MirrorMLP the fc1 even block is [sqrt(2) a+ ; c+] (twin-pair rows u, self-twin rows) and the odd block
sqrt(2) a- (twin-pair rows v); the fc2 even block is [sqrt(2) p+, q+] and the odd block sqrt(2) p-. The sqrt(2) appears
because E+ and E- are orthonormal, while efficient.py works with the unnormalised sums g_i + g_j and g_i - g_j. In
MirrorAttention the even block is the GEMM We (twin-pair rows u, the even rows of the self-heads in their B or C
eigenbasis) and the odd block is Wodd; the output projection's Oe and Oo are the wo blocks. split_blocks measures the
two off-diagonal blocks and refuses weights whose relative residual exceeds `tol` (1e-5, as efficient.py).

Low rank inside the blocks
--------------------------
Each block F is replaced by its rank-r truncation T_r(F); the matrix becomes

    W' = E_out+ T_r+(F+) E_in+^T + E_out- T_r-(F-) E_in-^T.

W' still satisfies P_out W' = W' P_in, so the stacked model keeps Mirror-exact's exact equivariance (and, at full
conversion with the even head, exact flip invariance of the k-NN feature). The SVD is taken in the orthonormal block
coordinates, so plain truncation is the best approximation of W in Frobenius norm among equivariant matrices with
block ranks (r+, r-) (Eckart-Young per block; the two blocks are orthogonal pieces of W). Inner methods:
    svd     plain truncated SVD of each block (baselines.lowrank_factors "svd").
    svdllm  SVD-LLM-style whitening of each block (baselines.lowrank_factors "svdllm", same ridge `damp`) with the
            block's own input moment E_in,s^T M E_in,s, where M is the matrix's uncentred input moment on the fit
            images through the folded original model (baselines.collect_stats, as the svdllm baseline uses: M1 for
            fc1, M2 for fc2, Nin for wq, wk, wv, Mctx for wo). The context moment of wo is rotated into the twin
            gauge first (G M G^T, G = C[h]^T on the second head of every twin pair, attention.gauge_twin_heads). The
            ridge lam = damp x mean(diag) is taken over the block's moment, not the whole matrix's.

Counting (PROJECT_NEXT_PHASE.md 4.5 and 4.7; mirror/budget.py): a block of shape m x n at rank r stores r (m + n)
weights (two factors); when r (m + n) >= m n it is stored dense (rank min(m, n), m n weights), as
budget._lowrank_alloc does. Multiply-adds per token equal the stored count (the butterfly additions of efficient.py
are not counted, as there). Basis changes as Mirror-exact (mirror/budget.py mirror_exact_basis_change: d^2 at each
span end that is not a model boundary; none at full conversion). Biases, norms and embeddings are dense and not part
of the budget, as everywhere in MUST-07.

Allocation rule (allocate_stack): the rule of mirror/budget.py applied to the blocks. The matrix budget T_m (the
point's budget minus Mirror-exact's basis changes) is spread over the blocks in proportion to their Mirror-exact
counts m n, target_i = ceil(m_i n_i T_m / S) with S = Mirror-exact's matrix total; each block starts at
rank floor(target_i / (m_i + n_i)) and then budget._fill adds one rank at a time to the block with the largest
remaining deficit until the total reaches T_m (so it ends at most one step, the largest m + n, above T_m). A block
whose factors would hold at least m n weights is stored dense. When T_m >= S every block keeps full rank: the point
is Mirror-exact itself, below the budget (`full_rank` in the info).

The low-rank-alone curve uses mirror/baselines.py compress (svd, svdllm) at a budget built by fraction_budget: the
Mirror-exact budget's matrices with `total` replaced by the point's budget, so budget.matrix_targets spreads it in
proportion to Mirror-exact's per-matrix counts (weights "mirror", MUST-07's rule; at the Mirror-exact total this is
exactly the MUST-07 matched baseline) or to the dense counts (weights "dense", a uniform ratio; optional).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .attention import HeadMaps, fold_attn, write_absorbed_attn
from .efficient import eig_basis
from .heal import exact_involution
from .m1 import involution_trace64
from .surgery import fold_mlp, write_absorbed_mlp

INNER_METHODS = ("svd", "svdllm")
COMPONENTS = ("mlp", "attn")
COMPONENT_MATRICES = {"mlp": ("fc1", "fc2"), "attn": ("wq", "wk", "wv", "wo")}
SIGNS = ("even", "odd")
DEFAULT_DAMP = 1e-6
DEFAULT_TOL = 1e-5

__all__ = ["INNER_METHODS", "PermSignBasis", "DenseSignBasis", "head_involution", "twin_gauge_matrix",
           "matrix_bases", "split_blocks", "merge_blocks", "sign_moments", "stack_layout", "allocate_stack",
           "fraction_budget", "truncate_block", "build_stacked", "check_stacked", "block_ranks", "budget_for_fraction",
           "StackedLinear", "StackedTrainable", "make_trainable_stacked"]


# ----------------------------------------------------------------------------------------------
# Even/odd bases
# ----------------------------------------------------------------------------------------------
class PermSignBasis:
    """Orthonormal even/odd basis of a neuron involution `perm` (h entries), without forming an h x h matrix.
    Even coordinates: [(x_i + x_j)/sqrt(2) for each pair i < j (in order of i), x_s for each self-twin s];
    odd coordinates: [(x_i - x_j)/sqrt(2) for each pair]."""

    def __init__(self, perm, device=None):
        perm = np.asarray(perm, dtype=np.int64)
        h = len(perm)
        if not np.array_equal(perm[perm], np.arange(h)):
            raise ValueError("perm must be an involution")
        idx = np.arange(h)
        pi = idx[perm > idx]
        self.n = int(h)
        self.n_pairs = int(len(pi))
        self.n_plus = int(h - self.n_pairs)
        self.n_minus = self.n_pairs
        self.trace = int((perm == idx).sum())
        self._idx = {"i": pi, "j": perm[pi], "s": idx[perm == idx]}
        self._dev = {}
        self.device = device

    def _t(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in self._idx.items()}
        return self._dev[key]

    def down(self, X: torch.Tensor):
        """(E+^T X, E-^T X) along dim 0."""
        t = self._t(X.device)
        r = 1.0 / math.sqrt(2.0)
        xi, xj = X[t["i"]], X[t["j"]]
        return torch.cat([(xi + xj) * r, X[t["s"]]], 0), (xi - xj) * r

    def up(self, Xp: torch.Tensor, Xm: torch.Tensor) -> torch.Tensor:
        """E+ Xp + E- Xm along dim 0."""
        t = self._t(Xp.device)
        r = 1.0 / math.sqrt(2.0)
        out = Xp.new_zeros((self.n,) + tuple(Xp.shape[1:]))
        a, m = Xp[:self.n_pairs], Xm
        out[t["i"]] = (a + m) * r
        out[t["j"]] = (a - m) * r
        out[t["s"]] = Xp[self.n_pairs:]
        return out

    def dense(self) -> np.ndarray:
        """[E+, E-] as an h x h matrix (tests)."""
        E = torch.eye(self.n, dtype=torch.float64)
        p, m = self.down(E)
        return torch.cat([p, m], 0).T.numpy()


class DenseSignBasis:
    """Orthonormal even/odd basis of an orthogonal symmetric involution given as a matrix (A* or P_B, P_C):
    Q = [Q+, Q-] from efficient.eig_basis (even columns first)."""

    def __init__(self, P):
        P = np.asarray(P, dtype=np.float64)
        Q, dp = eig_basis(P)
        self.Q = Q
        self.n = int(P.shape[0])
        self.n_plus = int(dp)
        self.n_minus = int(self.n - dp)
        self.trace = int(2 * dp - self.n)
        self._dev = {}

    def _t(self, device):
        key = str(device)
        if key not in self._dev:
            Q = torch.as_tensor(self.Q, dtype=torch.float64, device=device)
            self._dev[key] = (Q[:, :self.n_plus].contiguous(), Q[:, self.n_plus:].contiguous())
        return self._dev[key]

    def down(self, X: torch.Tensor):
        Qp, Qm = self._t(X.device)
        X = X.to(torch.float64)
        flat = X.reshape(X.shape[0], -1)
        return ((Qp.T @ flat).reshape((self.n_plus,) + tuple(X.shape[1:])),
                (Qm.T @ flat).reshape((self.n_minus,) + tuple(X.shape[1:])))

    def up(self, Xp: torch.Tensor, Xm: torch.Tensor) -> torch.Tensor:
        Qp, Qm = self._t(Xp.device)
        fp, fm = Xp.reshape(Xp.shape[0], -1), Xm.reshape(Xm.shape[0], -1)
        return (Qp @ fp + Qm @ fm).reshape((self.n,) + tuple(Xp.shape[1:]))

    def dense(self) -> np.ndarray:
        return self.Q.copy()


def head_involution(hm: HeadMaps, which: str = "B") -> np.ndarray:
    """D x D involution of the query/key (which "B") or value (which "C") channels: block (sigma(h), h) = X[h]."""
    X = hm.B if which == "B" else hm.C
    H, dh = hm.H, hm.dh
    P = np.zeros((H * dh, H * dh))
    for h in range(H):
        g = int(hm.sigma[h])
        P[g * dh:(g + 1) * dh, h * dh:(h + 1) * dh] = X[h]
    return P


def twin_gauge_matrix(hm_pre: HeadMaps, which: str = "C") -> np.ndarray:
    """D x D map G of attention.gauge_twin_heads on value (or query/key) channels: G = C[h]^T on the second head g of
    every twin pair (h, g), identity elsewhere. A context moment of the original model becomes G M G^T in the gauge
    of the Mirror-exact weights."""
    X = hm_pre.C if which == "C" else hm_pre.B
    H, dh = hm_pre.H, hm_pre.dh
    G = np.eye(H * dh)
    for h, g in hm_pre.pairs:
        G[g * dh:(g + 1) * dh, g * dh:(g + 1) * dh] = np.asarray(X[h]).T
    return G


class _BasisCache:
    """Bases of one span: A* once, a neuron basis per block, head bases per block (B and C)."""

    def __init__(self, A, perms, heads_after, blocks):
        self.A = DenseSignBasis(exact_involution(A))
        self.blocks = [int(b) for b in blocks]
        self.perms = None if perms is None else list(perms)
        self.heads = None if heads_after is None else list(heads_after)
        self._perm, self._head = {}, {}

    def perm(self, i: int) -> PermSignBasis:
        if i not in self._perm:
            self._perm[i] = PermSignBasis(self.perms[i])
        return self._perm[i]

    def head(self, i: int, which: str) -> DenseSignBasis:
        if (i, which) not in self._head:
            self._head[(i, which)] = DenseSignBasis(head_involution(self.heads[i], which))
        return self._head[(i, which)]


def matrix_bases(name: str, cache: _BasisCache, i: int):
    """(output basis, input basis) of matrix `name` of the i-th converted block (i indexes `blocks`)."""
    if name == "fc1":
        return cache.perm(i), cache.A
    if name == "fc2":
        return cache.A, cache.perm(i)
    if name in ("wq", "wk"):
        return cache.head(i, "B"), cache.A
    if name == "wv":
        return cache.head(i, "C"), cache.A
    if name == "wo":
        return cache.A, cache.head(i, "C")
    raise ValueError(f"unknown matrix {name!r}")


# ----------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------
def _sq(x: torch.Tensor) -> float:
    return float((x.double() ** 2).sum())


def split_blocks(W: torch.Tensor, out_b, in_b) -> tuple:
    """(even block [m+, n+], odd block [m-, n-], relative off-diagonal residual) of W [out, in] in float64."""
    W = W.to(torch.float64)
    Lp, Lm = out_b.down(W)                       # [m+, in], [m-, in]
    Bpp, Bpm = in_b.down(Lp.T)                   # [n+, m+], [n-, m+]
    Bmp, Bmm = in_b.down(Lm.T)                   # [n+, m-], [n-, m-]
    tot = _sq(W)
    res = math.sqrt((_sq(Bpm) + _sq(Bmp)) / tot) if tot > 0 else 0.0
    return Bpp.T.contiguous(), Bmm.T.contiguous(), res


def merge_blocks(even: torch.Tensor, odd: torch.Tensor, out_b, in_b) -> torch.Tensor:
    """E_out+ even E_in+^T + E_out- odd E_in-^T (float64)."""
    mp, np_ = even.shape
    mm, nm = odd.shape
    zp = even.new_zeros(nm, mp)
    zm = even.new_zeros(np_, mm)
    Xp = in_b.up(even.T, zp).T                   # [m+, in]
    Xm = in_b.up(zm, odd.T).T                    # [m-, in]
    return out_b.up(Xp, Xm)


def sign_moments(in_b, M: torch.Tensor) -> tuple:
    """(E+^T M E+, E-^T M E-) of an input moment M [n, n]."""
    M = M.to(torch.float64)
    Lp, Lm = in_b.down(M)
    Mpp, _ = in_b.down(Lp.T)
    _, Mmm = in_b.down(Lm.T)
    return Mpp.T.contiguous(), Mmm.T.contiguous()


# ----------------------------------------------------------------------------------------------
# Layout, budgets and allocation
# ----------------------------------------------------------------------------------------------
def _matrix_names(components) -> list:
    comps = [c for c in COMPONENTS if c in tuple(components)]
    return [n for c in comps for n in COMPONENT_MATRICES[c]]


def stack_layout(d: int, h: int, D: int, A, perms, heads_after, blocks, components) -> list:
    """One entry per (block, matrix, sign): shapes of the even and odd blocks of every converted matrix, from the
    traces alone (d+ = (d + tr A*)/2, h+ = h - pairs, D_B+ = (D + sum_s tr B[s])/2). `perms` and `heads_after` are
    aligned with `blocks`. Entry fields: block, name, sign, shape [m, n], dense (m n), matrix_dense, mirror (the
    matrix's Mirror-exact count, m+ n+ + m- n-)."""
    trA = involution_trace64(A)
    dp = (d + trA) // 2
    rows = []
    for i, b in enumerate(blocks):
        for name in _matrix_names(components):
            if name in ("fc1", "fc2"):
                perm = np.asarray(perms[i])
                pairs = int((perm != np.arange(len(perm))).sum() // 2)
                hp, hm_ = h - pairs, pairs
                sh = {"fc1": ((hp, dp), (hm_, d - dp)), "fc2": ((dp, hp), (d - dp, hm_))}[name]
                mshape = (h, d) if name == "fc1" else (d, h)
            else:
                hm = heads_after[i]
                X = hm.C if name in ("wv", "wo") else hm.B
                tr = sum(int(round(float(np.trace(X[s])))) for s in hm.self_heads)
                Dp = (D + tr) // 2
                sh = ((dp, Dp), (d - dp, D - Dp)) if name == "wo" else ((Dp, dp), (D - Dp, d - dp))
                mshape = (d, D) if name == "wo" else (D, d)
            mirror = sh[0][0] * sh[0][1] + sh[1][0] * sh[1][1]
            for sign, s in zip(SIGNS, sh):
                rows.append({"block": int(b), "name": name, "sign": sign, "shape": [int(s[0]), int(s[1])],
                             "dense": int(s[0] * s[1]), "matrix_shape": [int(x) for x in mshape],
                             "matrix_dense": int(mshape[0] * mshape[1]), "mirror": int(mirror)})
    return rows


def allocate_stack(layout: list, total: int) -> list:
    """Ranks for every block of `layout` so that the blocks hold `total` matrix weights (see the module docstring,
    "Allocation rule"; budget._fill does the filling, so the rule is literally mirror/budget.py's). Returns a copy of
    the layout entries with target, rank, full (min(m, n)), dense_storage and params."""
    from .budget import _ceil_div, _fill

    S = int(sum(e["dense"] for e in layout))
    T = int(total)
    if S <= 0:
        raise ValueError("layout has no parameters")
    units = []
    for e in layout:
        m, n = e["shape"]
        units.append({"target": _ceil_div(e["dense"] * T, S), "step": m + n, "max": min(m, n), "cap": m * n})
    counts = _fill(units, T)
    out = []
    for e, u, c in zip(layout, units, counts):
        m, n = e["shape"]
        full = min(m, n)
        dense_store = c * (m + n) >= m * n
        r = full if dense_store else int(c)
        out.append({**e, "target": int(u["target"]), "rank": int(r), "full": int(full),
                    "dense_storage": bool(dense_store), "params": int(m * n if dense_store else r * (m + n))})
    return out


def fraction_budget(base, total: int, weights: str = "mirror"):
    """A copy of a mirror/budget.py SpanBudget (Mirror-exact's) with `total` replaced by the point's budget, for
    baselines.compress. weights "mirror": per-matrix shares in proportion to Mirror-exact's counts (budget.py's rule);
    "dense": in proportion to the dense counts (one compression ratio for every matrix)."""
    if weights not in ("mirror", "dense"):
        raise ValueError("weights must be 'mirror' or 'dense'")
    mats = list(base.matrices)
    if weights == "dense":
        mats = [dataclasses.replace(m, mirror_exact=int(m.dense)) for m in mats]
    note = (f"CORE-03 point: total {int(total):,} = budget of the point (Mirror-exact's own total is {int(base.total):,}); "
            f"per-matrix shares in proportion to the {'Mirror-exact' if weights == 'mirror' else 'dense'} counts")
    return dataclasses.replace(base, matrices=mats, total=int(total), basis_change=0, notes=note,
                               variant=f"core03_{weights}")


def budget_for_fraction(span_dense: int, fraction: float) -> int:
    """The point's budget: ceil(fraction x dense count of the converted span) (integer arithmetic on a 1e-6 grid)."""
    num = int(round(float(fraction) * 1_000_000))
    return int(-(-int(span_dense) * num // 1_000_000))


# ----------------------------------------------------------------------------------------------
# Truncation
# ----------------------------------------------------------------------------------------------
def truncate_block(F: torch.Tensor, r: int, inner: str = "svd", M: torch.Tensor | None = None,
                   damp: float = DEFAULT_DAMP, force: bool = False) -> tuple:
    """(F', info): the rank-r truncation of block F (float64) by `inner`. At full rank F is returned unchanged
    unless `force` (tests: the SVD at full rank must reproduce F)."""
    from .baselines import lowrank_factors

    m, n = F.shape
    full = min(m, n)
    if m == 0 or n == 0:
        return F.clone(), {"rank": 0, "discarded_energy": 0.0}
    if r >= full and not force:
        return F.clone(), {"rank": int(full), "discarded_energy": 0.0}
    if inner == "svd":
        Fn, _, _, inf = lowrank_factors("svd", F, int(r))
    elif inner == "svdllm":
        if M is None:
            raise ValueError("svdllm needs the block's input moment")
        Fn, _, _, inf = lowrank_factors("svdllm", F, int(r), M=M, damp=damp)
    else:
        raise ValueError(f"inner method must be one of {INNER_METHODS}, not {inner!r}")
    return Fn, inf


def _read(blk: nn.Module, comps) -> dict:
    out = {}
    if "mlp" in comps:
        f = fold_mlp(blk)
        out.update({"fc1": f["W1"], "fc2": f["W2"], "_mlp": f})
    if "attn" in comps:
        f = fold_attn(blk)
        out.update({"wq": f["Wq"], "wk": f["Wk"], "wv": f["Wv"], "wo": f["Wo"], "_attn": f})
    return out


def _write(blk: nn.Module, comps, mats: dict) -> None:
    if "mlp" in comps:
        f = mats["_mlp"]
        write_absorbed_mlp(blk, mats["fc1"], f["b1"], mats["fc2"], f["b2"])
    if "attn" in comps:
        f = dict(mats["_attn"])
        f.update({"Wq": mats["wq"], "Wk": mats["wk"], "Wv": mats["wv"], "Wo": mats["wo"]})
        write_absorbed_attn(blk, **f)


def _moment(stats, b: int, name: str, hm_pre: HeadMaps | None) -> torch.Tensor:
    """Input moment of matrix `name` in block b from baselines.collect_stats (original folded model)."""
    from .baselines import _stat

    key = {"fc1": ("M1", b), "fc2": ("M2", b), "wq": ("Nin", b), "wk": ("Nin", b), "wv": ("Nin", b),
           "wo": ("Mctx", b)}[name]
    M = _stat(stats, *key).to(torch.float64)
    if name == "wo" and hm_pre is not None:
        G = torch.as_tensor(twin_gauge_matrix(hm_pre, "C"), dtype=torch.float64, device=M.device)
        M = G @ M @ G.T
    return M


# ----------------------------------------------------------------------------------------------
# Stacked model
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_stacked(adapter, struct, total: int, components=COMPONENTS, k: int | None = None, inner: str = "svd",
                  stats=None, damp: float = DEFAULT_DAMP, base=None, tol: float = DEFAULT_TOL, force: bool = False,
                  keep_blocks: bool = False):
    """Mirror-exact on blocks 0..k-1 (default: the full model), then low rank inside the even and odd blocks of every
    converted matrix, with ranks from allocate_stack so that matrices plus basis changes reach `total`.

    struct: mirror_family.SpanStructure. base: (model, info) of mirror_family.build_mirror_variant("mirror_exact")
    for the same span and components (deep-copied; built here when None). stats: baselines.collect_stats output
    (needed for inner "svdllm"). Returns (model, info): info has the fields of the MUST-07 runner (method, hparams,
    components, blocks, k, full, params_matrices, params_basis_change, params_total, budget_total, gap, gap_rel,
    span_dense, per_matrix, structure) plus mirror_exact_total, full_rank, max_offdiag_residual and the per-block
    allocation (`blocks_alloc`). keep_blocks keeps the truncated blocks (float64, CPU) in info["tensors"]."""
    from . import mirror_family as MF

    comps = MF.normalise_components(components)
    L = adapter.depth
    k = L if k is None else int(k)
    blocks = list(range(k))
    if inner not in INNER_METHODS:
        raise ValueError(f"inner method must be one of {INNER_METHODS}")
    if inner == "svdllm" and stats is None:
        raise ValueError("inner svdllm needs baselines.collect_stats statistics")
    if base is None:
        base = MF.build_mirror_variant(adapter, "mirror_exact", blocks, comps, struct)
    model = copy.deepcopy(base[0])
    binfo = base[1]
    if [int(b) for b in binfo["blocks"]] != blocks or tuple(binfo["components"]) != tuple(comps):
        raise ValueError("base Mirror-exact model is for another span or other components")
    s = binfo["structure"]
    A = exact_involution(s["A"])
    perms = s["perms"] if "mlp" in comps else None
    heads_after = s["heads_after"] if "attn" in comps else None
    heads_pre = s["heads"] if "attn" in comps else None
    blk0 = model.blocks[0]
    d, h = int(blk0.mlp.fc1.in_features), int(blk0.mlp.fc1.out_features)
    D = int(blk0.attn.qkv.out_features // 3)
    layout = stack_layout(d, h, D, A, perms, heads_after, blocks, comps)
    S = int(sum(e["dense"] for e in layout))
    if S != int(binfo["params_matrices"]):
        raise AssertionError(f"block layout holds {S} weights, Mirror-exact counts {binfo['params_matrices']}")
    basis = int(binfo["params_basis_change"])
    T = int(total)
    alloc = allocate_stack(layout, T - basis)
    by = {(e["block"], e["name"], e["sign"]): e for e in alloc}
    cache = _BasisCache(A, perms, heads_after, blocks)
    per_matrix, worst, tensors = [], 0.0, {}
    for i, b in enumerate(blocks):
        blk = model.blocks[b]
        mats = _read(blk, comps)
        for name in _matrix_names(comps):
            out_b, in_b = matrix_bases(name, cache, i)
            F = {}
            Fp, Fm, res = split_blocks(mats[name], out_b, in_b)
            worst = max(worst, res)
            if res > tol:
                raise ValueError(f"block {b} {name}: weights are not equivariant for this structure (relative "
                                 f"off-diagonal residual {res:.2e}, tol {tol:g})")
            Mp = Mm = None
            if inner == "svdllm":
                Mp, Mm = sign_moments(in_b, _moment(stats, b, name, heads_pre[i] if heads_pre else None).to(Fp.device))
            entry = {"block": int(b), "name": name, "kind": "stacked", "inner": inner,
                     "shape": list(mats[name].shape), "dense": int(mats[name].numel()), "residual": float(res)}
            params = 0
            for sign, Fs, Ms in (("even", Fp, Mp), ("odd", Fm, Mm)):
                e = by[(b, name, sign)]
                if list(Fs.shape) != e["shape"]:
                    raise AssertionError(f"block {b} {name} {sign}: shape {list(Fs.shape)}, layout {e['shape']}")
                Fn, inf = truncate_block(Fs, e["rank"], inner, Ms, damp, force=force)
                F[sign] = Fn
                params += e["params"]
                entry.update({f"shape_{sign}": e["shape"], f"rank_{sign}": e["rank"], f"full_{sign}": e["full"],
                              f"dense_storage_{sign}": e["dense_storage"], f"params_{sign}": e["params"],
                              f"target_{sign}": e["target"],
                              f"discarded_energy_{sign}": float(inf.get("discarded_energy", 0.0))})
                if keep_blocks:
                    tensors[f"{b}.{name}.{sign}"] = Fn.detach().cpu()
            mats[name] = merge_blocks(F["even"], F["odd"], out_b, in_b).to(mats[name].dtype)
            entry.update({"mirror": int(e["mirror"]), "params": int(params)})
            per_matrix.append(entry)
        _write(blk, comps, mats)
    model.eval()
    pm = int(sum(e["params"] for e in per_matrix))
    full_rank = all(e["dense_storage"] for e in alloc)
    pt = pm + basis
    step = max((e["shape"][0] + e["shape"][1]) for e in alloc)
    info = {"method": f"mirror_{inner}", "hparams": {"inner": inner, **({"damp": float(damp)} if inner == "svdllm" else {})},
            "components": list(comps), "blocks": blocks, "k": k, "full": k == L,
            "params_matrices": pm, "params_basis_change": basis, "params_total": int(pt), "budget_total": T,
            "gap": int(pt - T), "gap_rel": (pt - T) / T if T else float("nan"), "gap_step": int(step),
            "span_dense": int(binfo["span_dense"]), "mirror_exact_total": int(binfo["params_total"]),
            "mirror_exact_matrices": S, "full_rank": bool(full_rank),
            "below_budget": bool(pt < T), "max_offdiag_residual": float(worst), "per_matrix": per_matrix,
            "blocks_alloc": alloc, "structure": s, "build": binfo.get("build", {}),
            "allocation": "budget._fill over the even and odd blocks, targets proportional to Mirror-exact block counts",
            "tensors": tensors}
    return model, info


@torch.no_grad()
def block_ranks(model: nn.Module, info: dict, rtol: float | None = None) -> list:
    """Numerical ranks (float64 singular values above max(m, n) eps(storage) sigma_max) of the even and odd blocks of
    every converted matrix as stored in `model`, with the allocated ranks."""
    s = info["structure"]
    comps = tuple(info["components"])
    blocks = [int(b) for b in info["blocks"]]
    cache = _BasisCache(s["A"], s["perms"] if "mlp" in comps else None,
                        s["heads_after"] if "attn" in comps else None, blocks)
    out = []
    by = {(e["block"], e["name"]): e for e in info["per_matrix"]}
    for i, b in enumerate(blocks):
        mats = _read(model.blocks[b], comps)
        for name in _matrix_names(comps):
            W = mats[name]
            store = (model.blocks[b].mlp.fc1.weight if name in ("fc1", "fc2") else model.blocks[b].attn.qkv.weight).dtype
            out_b, in_b = matrix_bases(name, cache, i)
            Fp, Fm, res = split_blocks(W, out_b, in_b)
            e = by[(b, name)]
            row = {"block": b, "name": name, "residual": res}
            for sign, Fs in (("even", Fp), ("odd", Fm)):
                if Fs.numel() == 0:
                    row[f"numerical_rank_{sign}"] = 0
                else:
                    sv = torch.linalg.svdvals(Fs)
                    t = (rtol if rtol is not None else max(Fs.shape) * torch.finfo(store).eps) * float(sv.max())
                    row[f"numerical_rank_{sign}"] = int((sv > t).sum())
                row[f"rank_{sign}"] = int(e[f"rank_{sign}"])
            out.append(row)
    return out


@torch.no_grad()
def check_stacked(model: nn.Module, info: dict, tol: float | None = None) -> dict:
    """The stored model has the counted structure: every converted matrix is block diagonal in the even/odd bases
    (relative residual below tol: 1e-10 in float64, 1e-5 otherwise), each truncated block's numerical rank equals its
    allocated rank in float64 storage (at most that rank otherwise), every block stored dense has at most full rank
    (the folded model centres every write into the stream, so the even blocks of fc2 and wo lose the all-ones
    direction and are one short of full rank), and the realised count r (m + n) (m n when stored dense) adds up to
    params_matrices."""
    ranks = block_ranks(model, info)
    ref = next(model.parameters())
    t = tol if tol is not None else (1e-10 if ref.dtype == torch.float64 else 1e-5)
    exact = ref.dtype == torch.float64
    by = {(e["block"], e["name"]): e for e in info["per_matrix"]}
    ok, realised, bad = True, 0, []
    for r in ranks:
        e = by[(r["block"], r["name"])]
        good = r["residual"] <= t
        for sign in SIGNS:
            nr, want = r[f"numerical_rank_{sign}"], r[f"rank_{sign}"]
            if e[f"dense_storage_{sign}"]:
                good = good and nr <= want
            else:
                good = good and (nr == want if exact else nr <= want)
            m, n = e[f"shape_{sign}"]
            realised += m * n if e[f"dense_storage_{sign}"] else want * (m + n)
        if not good:
            bad.append(r)
        ok = ok and good
    ok_params = realised == int(info["params_matrices"])
    return {"ok": bool(ok and ok_params), "params_realised": int(realised), "params_matrices": int(info["params_matrices"]),
            "max_residual": max((r["residual"] for r in ranks), default=0.0), "failures": bad[:10],
            "n_matrices": len(ranks)}


# ----------------------------------------------------------------------------------------------
# Healing support (structure-preserving trainable form; integration with mirror/heal_structured.py is follow-up)
# ----------------------------------------------------------------------------------------------
class StackedLinear(nn.Module):
    """y = E_out+ (U+ (V+ (E_in+^T x))) + E_out- (U- (V- (E_in-^T x))) + E_out+ c.

    The bases are fixed (buffers; a neuron basis is stored as index vectors, a dense one as its two halves) and the
    factors U [m, r], V [r, n] of each block train (a block stored dense trains as one m x n matrix U with V = None).
    The bias lives in the even output subspace (coefficients c), so every gradient step keeps P_out W = W P_in and the
    bias equivariant: the stacked model stays exactly mirror-equivariant during healing without a projection step,
    and no block can exceed its rank."""

    def __init__(self, out_b, in_b, blocks: dict, bias_even: torch.Tensor | None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_features, self.out_features = int(in_b.n), int(out_b.n)
        self._store_basis("out", out_b, dtype, device)
        self._store_basis("in", in_b, dtype, device)
        self.ranks = {}
        for sign in SIGNS:
            U, V = blocks[sign]
            self.ranks[sign] = None if V is None else int(V.shape[0])
            setattr(self, f"U_{sign}", nn.Parameter(U.to(device=device, dtype=dtype).contiguous()))
            if V is None:
                setattr(self, f"V_{sign}", None)
            else:
                setattr(self, f"V_{sign}", nn.Parameter(V.to(device=device, dtype=dtype).contiguous()))
        self.c = None if bias_even is None else nn.Parameter(bias_even.to(device=device, dtype=dtype).contiguous())

    def _store_basis(self, side: str, basis, dtype, device):
        if isinstance(basis, PermSignBasis):
            setattr(self, f"{side}_kind", "perm")
            for k, v in basis._idx.items():
                self.register_buffer(f"{side}_{k}", torch.as_tensor(v, dtype=torch.long, device=device))
            setattr(self, f"{side}_np", basis.n_pairs)
        else:
            setattr(self, f"{side}_kind", "dense")
            Q = torch.as_tensor(basis.Q, dtype=dtype, device=device)
            self.register_buffer(f"{side}_Qp", Q[:, :basis.n_plus].contiguous())
            self.register_buffer(f"{side}_Qm", Q[:, basis.n_plus:].contiguous())
        setattr(self, f"{side}_n", basis.n)

    def _down_in(self, x: torch.Tensor):
        """Input coordinates: x [..., n] -> (x+ [..., n+], x- [..., n-])."""
        if self.in_kind == "dense":
            return x @ self.in_Qp, x @ self.in_Qm
        r = 1.0 / math.sqrt(2.0)
        xi, xj = x[..., self.in_i], x[..., self.in_j]
        return torch.cat([(xi + xj) * r, x[..., self.in_s]], -1), (xi - xj) * r

    def _up_out(self, yp: torch.Tensor, ym: torch.Tensor) -> torch.Tensor:
        if self.out_kind == "dense":
            return yp @ self.out_Qp.T + ym @ self.out_Qm.T
        r = 1.0 / math.sqrt(2.0)
        out = yp.new_zeros(yp.shape[:-1] + (self.out_n,))
        a = yp[..., :self.out_np]
        out[..., self.out_i] = (a + ym) * r
        out[..., self.out_j] = (a - ym) * r
        out[..., self.out_s] = yp[..., self.out_np:]
        return out

    def _block(self, x: torch.Tensor, sign: str) -> torch.Tensor:
        U, V = getattr(self, f"U_{sign}"), getattr(self, f"V_{sign}")
        if V is None:
            return x @ U.T
        return (x @ V.T) @ U.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp, xm = self._down_in(x)
        yp = self._block(xp, "even")
        if self.c is not None:
            yp = yp + self.c
        return self._up_out(yp, self._block(xm, "odd"))

    @torch.no_grad()
    def dense(self) -> tuple:
        """(W [out, in], b [out]) in float64."""
        eye = torch.eye(self.in_features, dtype=self.U_even.dtype, device=self.U_even.device)
        c = self.c
        self.c = None
        try:
            W = self.forward(eye).T
        finally:
            self.c = c
        b = None
        if c is not None:
            z = torch.zeros(1, self.in_features, dtype=self.U_even.dtype, device=self.U_even.device)
            b = self.forward(z)[0]
        return W.double(), (None if b is None else b.double())


class _FusedQKV(nn.Module):
    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.out_features = q.out_features * 3
        self.in_features = q.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q(x), self.k(x), self.v(x)], -1)


class StackedTrainable:
    """Constraint (mirror/heal.py protocol) of a stacked student: freezes everything outside the converted blocks and
    the embeddings, and inside the converted blocks the absorbed norm affines and LayerScale (the trainable set of
    heal_structured.StructuredTrainable and E3); projects nothing, because StackedLinear keeps the structure; `check`
    confirms that no factor has grown beyond its allocated rank."""

    def __init__(self, blocks: Sequence[int], ranks: Sequence[dict]):
        self.blocks = [int(b) for b in blocks]
        self.ranks = [dict(r) for r in ranks]

    def frozen_parameters(self, model: nn.Module) -> list:
        from .heal import _absorbed_parameters
        from .heal_attn import attn_absorbed_parameters
        from .heal_vexact import span_frozen_parameters

        out = list(span_frozen_parameters(model, self.blocks, True))
        for b in self.blocks:
            out += list(_absorbed_parameters(model.blocks[b])) + list(attn_absorbed_parameters(model.blocks[b]))
        return out

    def __call__(self, model: nn.Module) -> None:
        return None

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        ok = True
        for r in self.ranks:
            blk = model.blocks[r["block"]]
            mod = {"fc1": blk.mlp.fc1, "fc2": blk.mlp.fc2, "wo": blk.attn.proj}.get(r["name"])
            if mod is None:
                mod = getattr(blk.attn.qkv, r["name"][1])
            for sign in SIGNS:
                V = getattr(mod, f"V_{sign}")
                ok = ok and (V is None if r[f"dense_storage_{sign}"] else int(V.shape[0]) == int(r[f"rank_{sign}"]))
        return {"ok": bool(ok), "n_matrices": len(self.ranks)}

    def describe(self) -> dict:
        return {"type": "StackedTrainable", "blocks": self.blocks,
                "frozen": "everything outside the converted blocks and the embeddings; norm affines and LayerScale "
                          "inside the converted blocks"}


def _factors(F: torch.Tensor, r: int, dense: bool):
    if dense or min(F.shape) == 0:
        return F, None
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)
    sq = S[:r].sqrt()
    return U[:, :r] * sq, sq[:, None] * Vh[:r]


@torch.no_grad()
def make_trainable_stacked(model: nn.Module, info: dict, adapter=None, tol: float = 1e-4) -> tuple:
    """Turn a stacked model (build_stacked, changed in place) into its trainable form: every converted Linear becomes
    a StackedLinear (fc1, fc2, attn.proj; attn.qkv becomes three of them, fused). Returns (model, constraints, spec).

    constraints: StackedTrainable (the trainable set) and, when an adapter is given, mirror.heal_vexact.
    VExactEmbeddings (embeddings re-projected after every step, as Mirror-exact in MUST-09). The weights need no
    projection: StackedLinear keeps them exactly equivariant and at their ranks by construction. Checked: each new
    module reproduces its dense matrix and bias (relative error below tol). spec: a JSON description (blocks, ranks
    per block) for the record; rebuilding a healed stacked model from it (heal_structured.build_skeleton) is not
    implemented yet."""
    s = info["structure"]
    comps = tuple(info["components"])
    if set(comps) != set(COMPONENTS):
        raise NotImplementedError("the trainable stacked form converts MLP and attention together")
    blocks = [int(b) for b in info["blocks"]]
    cache = _BasisCache(s["A"], s["perms"], s["heads_after"], blocks)
    by = {(e["block"], e["name"]): e for e in info["per_matrix"]}
    ref = next(model.parameters())
    dtype, dev = ref.dtype, ref.device
    worst = 0.0
    for i, b in enumerate(blocks):
        blk = model.blocks[b]
        mats = _read(blk, comps)
        biases = {"fc1": mats["_mlp"]["b1"], "fc2": mats["_mlp"]["b2"], "wq": mats["_attn"]["bq"],
                  "wk": mats["_attn"]["bk"], "wv": mats["_attn"]["bv"], "wo": mats["_attn"]["bo"]}
        mods = {}
        for name in _matrix_names(comps):
            out_b, in_b = matrix_bases(name, cache, i)
            Fp, Fm, _ = split_blocks(mats[name], out_b, in_b)
            e = by[(b, name)]
            fac = {sign: _factors(Fs, e[f"rank_{sign}"], e[f"dense_storage_{sign}"])
                   for sign, Fs in (("even", Fp), ("odd", Fm))}
            bp, bm = out_b.down(biases[name].to(torch.float64)[:, None])
            mod = StackedLinear(out_b, in_b, fac, bp[:, 0], dtype=dtype, device=dev)
            W, bb = mod.dense()
            err = max(float((W - mats[name]).norm() / mats[name].norm().clamp_min(1e-300)),
                      float((bb - biases[name]).norm() / biases[name].norm().clamp_min(1e-300))
                      if float(biases[name].norm()) > 0 else float(bb.norm()))
            worst = max(worst, err)
            if err > tol:
                raise ValueError(f"block {b} {name}: trainable form differs from the stacked weights ({err:.2e})")
            mods[name] = mod
        blk.mlp.fc1, blk.mlp.fc2 = mods["fc1"], mods["fc2"]
        blk.attn.qkv = _FusedQKV(mods["wq"], mods["wk"], mods["wv"])
        blk.attn.proj = mods["wo"]
    ranks = [{k: e[k] for k in ("block", "name", "rank_even", "rank_odd", "dense_storage_even", "dense_storage_odd")}
             for e in info["per_matrix"]]
    cons = [StackedTrainable(blocks, ranks)]
    if adapter is not None:
        from .heal_vexact import VExactEmbeddings

        cons.append(VExactEmbeddings(adapter, exact_involution(s["A"]), blocks, s.get("reg_perm"), train=True))
    spec = {"family": "stacked", "method": info.get("method"), "blocks": blocks, "components": list(comps),
            "ranks": ranks,
            "max_conversion_error": worst,
            "trainable": "StackedLinear factors and even-subspace biases of the converted blocks (equivariant by "
                         "construction) and the embeddings (re-projected by VExactEmbeddings); norms frozen"}
    return model, cons, spec

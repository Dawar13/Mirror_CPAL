"""Strong compression baselines at the Mirror budget (PAPER_EXPERIMENTS.md, M3-grid; zero-shot part).

Three methods compress the MLPs of the converted blocks (blocks 1..k, 0-based 0..k-1) of a pre-norm
ViT. Each is matched to the budget of Mirror V-local, MLP only, at the same k: the MLP weight count of
the converted blocks, sum over blocks of eq_fc1 + eq_fc2 (variants.mirror_param_counts; biases are
excluded, as in run_e1.py and the Track A ablations). Every method starts from the folded MLP weights of
surgery.fold_mlp (norm2 affine in fc1, LayerScale in fc2) and writes the result back in that absorbed
form (surgery.write_absorbed_mlp), exactly as the mirror and plain SVD variants are written.

Conventions (as mirror/surgery.py): fc1.weight W1 is [h, d], fc2.weight W2 is [d, h], column vectors.

asvd: activation-aware SVD in the style of SVD-LLM (arXiv 2403.07378 (verify))
    For each matrix W with input second moment M (sum over fit tokens of x x^T, uncentred, float64;
    fc1: x = norm2 output without affine, d x d; fc2: x = GELU output, h x h):
        L = cholesky(M + lam I),  lam = damp * mean(diag M)
        W L = U S V^T,  W' = U_r S_r V_r^T L^-1          (whiten, truncate, un-whiten)
    W' minimises ||(W - W') L||_F = the error of W x on the fit activations (with the ridge) among
    rank-r matrices; it equals U_r U_r^T W. The ridge is needed: the norm2 output is mean-zero over
    channels, so M_fc1 has the all-ones direction as an exact null direction and no Cholesky factor.
    Ranks: those of the plain SVD baseline (variants.svd_spec: r = floor(eq / (h + d)) per matrix),
    so the stored factors U_r S_r and V_r^T L^-1 hold r (h + d) weights per matrix. Biases kept.

slicegpt: SliceGPT-style PCA slicing (SliceGPT, arXiv 2401.15024), MLP only
    Statistics come from the folded model of mirror/fold.py (every LayerNorm becomes RMSNorm on a
    mean-zero stream, so the network commutes with any orthogonal change of the stream basis, the
    property SliceGPT relies on). For converted block b, Q_b holds the eigenvectors of
    S_b = sum over fit tokens of x x^T, x the folded residual stream entering the MLP (the input of
    norm2), in descending order of eigenvalue. In the basis Q_b the MLP is sliced:
        fc1 reads only the first d_in coordinates of the normalised stream:  (W1 Q_b)[:, :d_in], h x d_in;
        fc2 writes only into the first d_out coordinates:                  (Q_b^T W2)[:d_out], d_out x h,
        and its bias likewise, (Q_b^T b2)[:d_out].
    Sliced: the input dimension of fc1 and the output dimension of fc2 of every converted block.
    Not sliced: the residual stream, the norms (RMSNorm still normalises the full stream), attention,
    the skip connections, the hidden width h and the fc1 bias. In the original basis the sliced MLP is
    W1 P_in, P_out W2, P_out b2 with P_n = Q_b[:, :n] Q_b[:, :n]^T; that is what is written into the
    model (absorbed form, unfolded model; the same function as slicing the folded model, which
    slice_folded_block does and the tests compare in float64).
    Budget: h (d_in + d_out) MLP weights per block, d_in = floor(eq_fc1 / h), d_out = floor(eq_fc2 / h).
    Not counted, in the same way as Mirror V-local's per-block basis rotations are not counted in its
    MLP budget: the change of stream basis between consecutive converted blocks, which an
    implementation holding the stream in basis Q_b around block b stores as one d x d matrix per block
    (the role of SliceGPT's shortcut matrices (verify); d x d here because the stream itself is not
    sliced). Recorded as basis_change_params = k d^2.

prune_act: structured MLP width pruning with activation-weighted importance
    Neuron j of block b scores mean over fit tokens of |g_j| times ||W2[:, j]||_2 (g the GELU output,
    W2 the folded fc2, LayerScale included): the average size of what the neuron writes into the
    stream. The n highest-scoring neurons are kept (ties go to the lower index) and the rest removed:
    fc1 and fc2 are replaced by nn.Linear layers of width n holding the kept rows of W1 and b1 and the
    kept columns of W2; b2 is kept. n = floor((eq_fc1 + eq_fc2) / (2 d)), so 2 n d MLP weights per block.

Budget rounding: every allocation rounds down, as the plain SVD baseline does, so no baseline gets
more weights than Mirror. Shortfall: under (h + d) per matrix (asvd), under h per matrix (slicegpt),
under 2 d per block (prune_act). For DINOv2-B (d = 768, h = 3072, eq above 1.1M per matrix) that is
under 0.4% (tests/test_strong_baselines.py checks the fitted counts of all three fit seeds).

Statistics (collect_baseline_stats): one pass of the fit images (no mirrored copies) through the folded
copy of the original model, every token (CLS, registers, patches), fp32 forward, float64 sums. All
blocks use the original model's activations (not those of a model whose earlier blocks are already
compressed). The fc1 input of the folded model is RMSNorm of the folded stream, which equals the
original norm2 LayerNorm without affine; the fc2 input is the same GELU output as in the original.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .surgery import fold_mlp, svd_rank_for, write_absorbed_mlp

METHODS = ("asvd", "slicegpt", "prune_act")
NEEDS = {"asvd": ("M1", "M2"), "slicegpt": ("S",), "prune_act": ("abs_g",)}
DEFAULT_DAMP = 1e-6
PARAM_TOL = 0.01   # PAPER_EXPERIMENTS.md, Section 1: matched to the converted blocks' count within 1%


# ----------------------------------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------------------------------
def allocate(method: str, h: int, d: int, eq) -> tuple[list, int]:
    """(per-block allocation, MLP weights kept) for `method` given the per-block Mirror counts
    eq = [(eq_fc1, eq_fc2), ...]. Allocations: asvd [(r_fc1, r_fc2)], slicegpt [(d_in, d_out)],
    prune_act [n]. Every one rounds down (never above the Mirror budget)."""
    eq = [(int(a), int(b)) for a, b in eq]
    if method == "asvd":
        alloc = [(min(svd_rank_for(h, d, e1), h, d), min(svd_rank_for(h, d, e2), h, d)) for e1, e2 in eq]
        return alloc, sum((r1 + r2) * (h + d) for r1, r2 in alloc)
    if method == "slicegpt":
        alloc = [(min(e1 // h, d), min(e2 // h, d)) for e1, e2 in eq]
        return alloc, sum(h * (a + b) for a, b in alloc)
    if method == "prune_act":
        alloc = [min((e1 + e2) // (2 * d), h) for e1, e2 in eq]
        return alloc, sum(2 * n * d for n in alloc)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def method_params(method: str, h: int, d: int, alloc) -> int:
    """MLP weights kept by an allocation (the count `allocate` returns)."""
    if method == "asvd":
        return sum((int(r1) + int(r2)) * (h + d) for r1, r2 in alloc)
    if method == "slicegpt":
        return sum(h * (int(a) + int(b)) for a, b in alloc)
    if method == "prune_act":
        return sum(2 * int(n) * d for n in alloc)
    raise ValueError(f"unknown method {method!r}")


def bias_params(method: str, h: int, d: int, alloc) -> int:
    """Bias entries kept (reported, not part of the budget): fc1 bias plus fc2 bias per block."""
    if method == "asvd":
        return len(alloc) * (h + d)
    if method == "slicegpt":
        return sum(h + int(b) for _, b in alloc)
    if method == "prune_act":
        return sum(int(n) + d for n in alloc)
    raise ValueError(f"unknown method {method!r}")


@dataclass
class Plan:
    method: str
    blocks: list
    alloc: list
    params_kept: int
    params_dense: int
    mirror_budget: int
    eq: list

    @property
    def ratio(self) -> float:
        return self.params_kept / self.mirror_budget

    @property
    def within_tol(self) -> bool:
        return abs(self.ratio - 1.0) <= PARAM_TOL

    def as_dict(self) -> dict:
        return {"method": self.method, "blocks": [b + 1 for b in self.blocks],
                "allocation": [list(a) if isinstance(a, (tuple, list)) else int(a) for a in self.alloc],
                "params_kept": int(self.params_kept), "mirror_budget": int(self.mirror_budget),
                "ratio_to_mirror": float(self.ratio), "within_1pct": bool(self.within_tol),
                "params_dense": int(self.params_dense), "mirror_eq_per_block": [list(e) for e in self.eq]}


def plan_for(method: str, struct, k: int) -> Plan:
    """Allocation for blocks 0..k-1 matched to Mirror V-local, MLP only, at the same k."""
    from .variants import mirror_param_counts

    blocks = list(range(int(k)))
    h, d = struct.h, struct.d
    eq = mirror_param_counts(struct, blocks)
    alloc, kept = allocate(method, h, d, eq)
    return Plan(method, blocks, alloc, int(kept), 2 * h * d * len(blocks), int(sum(a + b for a, b in eq)),
                [tuple(map(int, e)) for e in eq])


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------
@dataclass
class BaselineStats:
    blocks: list
    M1: dict = field(default_factory=dict)      # b -> [d, d] sum u u^T, u = fc1 input (norm2 output)
    M2: dict = field(default_factory=dict)      # b -> [h, h] sum g g^T, g = fc2 input (GELU output)
    abs_g: dict = field(default_factory=dict)   # b -> [h]    sum |g|
    S: dict = field(default_factory=dict)       # b -> [d, d] sum x x^T, x = folded stream entering the MLP
    n_tokens: int = 0
    n_images: int = 0
    info: dict = field(default_factory=dict)

    def mean_abs_g(self, b: int) -> torch.Tensor:
        return self.abs_g[b] / max(self.n_tokens, 1)


def needs_for(methods) -> set:
    out = set()
    for m in methods:
        if m not in NEEDS:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS}")
        out.update(NEEDS[m])
    return out


@torch.no_grad()
def collect_baseline_stats(adapter, loader, device, blocks, need=("M1", "M2", "abs_g", "S"),
                           to_cpu: bool = True) -> BaselineStats:
    """One pass of `loader` through the folded copy (mirror/fold.py) of adapter.model; float64 sums of
    the requested statistics for `blocks` (see BaselineStats). The original model is not modified."""
    from .data import normalize_for
    from .fold import fold_copy

    need = set(need)
    unknown = need - {"M1", "M2", "abs_g", "S"}
    if unknown:
        raise ValueError(f"unknown statistics {sorted(unknown)}")
    blocks = sorted({int(b) for b in blocks})
    d, h = adapter.embed_dim, adapter.mlp_hidden
    folded, finfo = fold_copy(adapter.model, final_norm_affine=False)
    folded.eval()
    kw = dict(dtype=torch.float64, device=device)
    st = BaselineStats(blocks=blocks)
    for b in blocks:
        if "M1" in need:
            st.M1[b] = torch.zeros(d, d, **kw)
        if "M2" in need:
            st.M2[b] = torch.zeros(h, h, **kw)
        if "abs_g" in need:
            st.abs_g[b] = torch.zeros(h, **kw)
        if "S" in need:
            st.S[b] = torch.zeros(d, d, **kw)
    worst_mean = [0.0]
    counter = {"tokens": 0}
    handles = []

    def count(_mod, args):
        t = args[0]
        counter["tokens"] += int(t.shape[0]) * int(t.shape[1])

    handles.append(folded.blocks[blocks[0]].register_forward_pre_hook(count))
    for b in blocks:
        blk = folded.blocks[b]
        if "S" in need:
            def pre_norm2(_mod, args, b=b):
                X = args[0].reshape(-1, d).to(torch.float64)
                st.S[b] += X.T @ X
                rel = float((X.mean(dim=1).abs() / X.pow(2).mean(dim=1).sqrt().clamp_min(1e-300)).max())
                worst_mean[0] = max(worst_mean[0], rel)

            handles.append(blk.norm2.register_forward_pre_hook(pre_norm2))
        if "M1" in need:
            def pre_fc1(_mod, args, b=b):
                U = args[0].reshape(-1, d).to(torch.float64)
                st.M1[b] += U.T @ U

            handles.append(blk.mlp.fc1.register_forward_pre_hook(pre_fc1))
        if "M2" in need or "abs_g" in need:
            def pre_fc2(_mod, args, b=b):
                G = args[0].reshape(-1, h).to(torch.float64)
                if "M2" in need:
                    st.M2[b] += G.T @ G
                if "abs_g" in need:
                    st.abs_g[b] += G.abs().sum(0)

            handles.append(blk.mlp.fc2.register_forward_pre_hook(pre_fc2))
    n_img = 0
    try:
        for imgs, _ in loader:
            x = normalize_for(adapter, imgs.to(device, non_blocking=True))
            adapter.forward_tokens(x, folded)
            n_img += int(x.shape[0])
    finally:
        for hd in handles:
            hd.remove()
    del folded
    st.n_tokens, st.n_images = counter["tokens"], n_img
    st.info = {"statistics": sorted(need), "blocks": [b + 1 for b in blocks], "n_images": n_img,
               "n_tokens": st.n_tokens, "model_form": "folded copy (mirror/fold.py, final affine kept in model)",
               "fold_eps": finfo.get("eps"), "tokens": "all (CLS, registers, patches), fit images only, no mirrors",
               "precision": "fp32 forward, float64 sums"}
    if "S" in need:
        st.info["max_rel_channel_mean_of_folded_stream"] = worst_mean[0]
    if to_cpu:
        for name in ("M1", "M2", "abs_g", "S"):
            setattr(st, name, {b: t.cpu() for b, t in getattr(st, name).items()})
    return st


# ----------------------------------------------------------------------------------------------
# Methods (pure functions on folded weights, float64)
# ----------------------------------------------------------------------------------------------
def asvd_truncate(W: torch.Tensor, M: torch.Tensor, r: int, damp: float = DEFAULT_DAMP):
    """Whitened rank-r truncation of W ([out, in]) for inputs with second moment M ([in, in]).

    L = cholesky(M + lam I) with lam = damp * mean(diag M); W L = U S V^T; returns
    (W' = U_r S_r V_r^T L^-1, info). info: rank, the share of whitened energy discarded and lam."""
    W = W.to(torch.float64)
    M = M.to(W)
    n = M.shape[0]
    if W.shape[1] != n:
        raise ValueError(f"W {tuple(W.shape)} does not take inputs of size {n}")
    Ms = 0.5 * (M + M.T)
    lam = float(damp) * float(torch.diagonal(Ms).mean())
    L, err = torch.linalg.cholesky_ex(Ms + lam * torch.eye(n, dtype=W.dtype, device=W.device))
    if int(err) != 0:
        raise RuntimeError(f"Cholesky failed (damp {damp}); increase --asvd-damp")
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    r = int(max(0, min(int(r), S.numel())))
    B = (U[:, :r] * S[:r]) @ Vh[:r]
    W_new = torch.linalg.solve_triangular(L, B, upper=False, left=False)     # X L = B
    e = S * S
    tot = float(e.sum())
    return W_new, {"rank": r, "whitened_discarded_energy": float(1.0 - float(e[:r].sum()) / tot) if tot > 0 else 0.0,
                   "ridge": lam}


def centre_moment_t(S: torch.Tensor) -> torch.Tensor:
    """(I - 11^T/d) S (I - 11^T/d): the all-ones direction becomes an exact null direction."""
    return S - S.mean(0, keepdim=True) - S.mean(1, keepdim=True) + S.mean()


def pca_basis(S: torch.Tensor):
    """(Q, eigenvalues): eigenvectors of the centred stream moment, descending eigenvalue order."""
    S = centre_moment_t(S.to(torch.float64))
    lam, Q = torch.linalg.eigh(0.5 * (S + S.T))
    return Q.flip(1), lam.flip(0)


def slice_projectors(Q: torch.Tensor, d_in: int, d_out: int):
    Qi, Qo = Q[:, :int(d_in)], Q[:, :int(d_out)]
    return Qi @ Qi.T, Qo @ Qo.T


def prune_scores(W2: torch.Tensor, mean_abs_g: torch.Tensor) -> torch.Tensor:
    """Importance of each hidden neuron: mean |g_j| times ||W2[:, j]|| (W2 folded, [d, h])."""
    return mean_abs_g.to(torch.float64) * W2.to(torch.float64).norm(dim=0)


def prune_keep(scores: torch.Tensor, n: int) -> torch.Tensor:
    """Indices (ascending) of the n highest scores; ties go to the lower index."""
    order = torch.argsort(-scores, stable=True)
    return torch.sort(order[:int(n)]).values


# ----------------------------------------------------------------------------------------------
# Application to a model
# ----------------------------------------------------------------------------------------------
def _dev(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float64, device=like.device)


@torch.no_grad()
def _shrink_mlp(blk: nn.Module, n: int) -> None:
    """Replace fc1 and fc2 by nn.Linear layers of hidden width n (weights are written afterwards)."""
    fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
    kw = dict(device=fc1.weight.device, dtype=fc1.weight.dtype)
    new1 = nn.Linear(fc1.in_features, int(n), bias=True, **kw)
    new2 = nn.Linear(int(n), fc2.out_features, bias=True, **kw)
    for p in list(new1.parameters()) + list(new2.parameters()):
        p.requires_grad_(fc1.weight.requires_grad)
    blk.mlp.fc1, blk.mlp.fc2 = new1, new2


@torch.no_grad()
def apply_baseline(model: nn.Module, method: str, blocks, alloc, stats: BaselineStats,
                   damp: float = DEFAULT_DAMP) -> list:
    """Apply `method` to the MLPs of `blocks` of `model` (in place). Returns per-block info."""
    blocks = [int(b) for b in blocks]
    if len(alloc) != len(blocks):
        raise ValueError("one allocation per block")
    out = []
    for b, a in zip(blocks, alloc):
        blk = model.blocks[b]
        f = fold_mlp(blk)
        W1, b1, W2, b2 = f["W1"], f["b1"], f["W2"], f["b2"]
        h, d = W1.shape
        if method == "asvd":
            r1, r2 = (int(x) for x in a)
            W1n, i1 = asvd_truncate(W1, _dev(stats.M1[b], W1), r1, damp)
            W2n, i2 = asvd_truncate(W2, _dev(stats.M2[b], W1), r2, damp)
            write_absorbed_mlp(blk, W1n, b1, W2n, b2)
            out.append({"block": b + 1, "rank_fc1": r1, "rank_fc2": r2,
                        "whitened_discarded_fc1": i1["whitened_discarded_energy"],
                        "whitened_discarded_fc2": i2["whitened_discarded_energy"],
                        "ridge_fc1": i1["ridge"], "ridge_fc2": i2["ridge"]})
        elif method == "slicegpt":
            d_in, d_out = (int(x) for x in a)
            Q, lam = pca_basis(_dev(stats.S[b], W1))
            P_in, P_out = slice_projectors(Q, d_in, d_out)
            write_absorbed_mlp(blk, W1 @ P_in, b1, P_out @ W2, P_out @ b2)
            tot = float(lam.clamp_min(0).sum())
            out.append({"block": b + 1, "d_in": d_in, "d_out": d_out,
                        "pca_energy_kept_in": float(lam[:d_in].clamp_min(0).sum()) / tot if tot > 0 else 1.0,
                        "pca_energy_kept_out": float(lam[:d_out].clamp_min(0).sum()) / tot if tot > 0 else 1.0})
        elif method == "prune_act":
            n = int(a)
            scores = prune_scores(W2, _dev(stats.mean_abs_g(b), W1))
            keep = prune_keep(scores, n)
            _shrink_mlp(blk, n)
            write_absorbed_mlp(blk, W1[keep], b1[keep], W2[:, keep], b2)
            tot = float(scores.sum())
            out.append({"block": b + 1, "n_kept": n, "h": h,
                        "importance_share_kept": float(scores[keep].sum()) / tot if tot > 0 else 1.0})
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def baseline_copy(model: nn.Module, method: str, blocks, alloc, stats: BaselineStats,
                  damp: float = DEFAULT_DAMP):
    """(compressed deep copy, per-block info); `model` is left untouched."""
    m = copy.deepcopy(model)
    info = apply_baseline(m, method, blocks, alloc, stats, damp)
    return m, info


@torch.no_grad()
def slice_folded_block(folded_block: nn.Module, Q: torch.Tensor, d_in: int, d_out: int) -> None:
    """Reference form: slice the MLP of a block of a folded model (mirror/fold.py) in place, fc1 to the
    first d_in and fc2 to the first d_out coordinates of the basis Q (dense equivalent W1 P_in, P_out W2,
    P_out b2). Used by the tests to check that the absorbed form written by apply_baseline computes the
    same function."""
    fc1, fc2 = folded_block.mlp.fc1, folded_block.mlp.fc2
    P_in, P_out = slice_projectors(Q.to(torch.float64), d_in, d_out)
    W1 = fc1.weight.detach().to(torch.float64)
    W2 = fc2.weight.detach().to(torch.float64)
    b2 = fc2.bias.detach().to(torch.float64)
    fc1.weight.copy_((W1 @ P_in.to(W1.device)).to(fc1.weight))
    fc2.weight.copy_((P_out.to(W2.device) @ W2).to(fc2.weight))
    fc2.bias.copy_((P_out.to(b2.device) @ b2).to(fc2.bias))


def realised_params(model: nn.Module, method: str, blocks, alloc) -> int:
    """MLP weights of the converted blocks as stored in the realised form, measured on `model`:
    asvd, the rank of each written matrix times (rows + columns); slicegpt, h times the rank of each
    written matrix (the sliced dimensions); prune_act, the fc1 and fc2 weight entries. Used by the tests
    to confirm that the analytic counts describe the model."""
    total = 0
    for b, a in zip(blocks, alloc):
        blk = model.blocks[int(b)]
        W1 = blk.mlp.fc1.weight.detach().to(torch.float64)
        W2 = blk.mlp.fc2.weight.detach().to(torch.float64)
        if method == "prune_act":
            total += W1.numel() + W2.numel()
            continue
        ranks = [int(torch.linalg.matrix_rank(W)) for W in (W1, W2)]
        if method == "asvd":
            total += sum(r * (W.shape[0] + W.shape[1]) for r, W in zip(ranks, (W1, W2)))
        else:
            total += W1.shape[0] * ranks[0] + W2.shape[1] * ranks[1]
    return int(total)


def weights_signature(model: nn.Module, method: str, blocks, alloc) -> str:
    """Content hash of the converted MLPs (method, blocks, allocation and the written weights)."""
    hs = hashlib.sha1()
    hs.update(json.dumps([method, [int(b) for b in blocks],
                          [list(map(int, a)) if isinstance(a, (tuple, list)) else int(a) for a in alloc]]).encode())
    for b in blocks:
        mlp = model.blocks[int(b)].mlp
        for t in (mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias):
            hs.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return hs.hexdigest()[:20]

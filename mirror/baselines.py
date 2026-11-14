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

MUST-07 and MUST-09 (PROJECT_NEXT_PHASE.md) add a second family at the Mirror-exact budget of mirror/budget.py,
for the MLP and the attention: METHODS_V2, HPARAM_GRIDS, collect_stats, compress and structure_check, documented
in the section "Methods at the Mirror-exact budget" below. The three methods above and their outputs are unchanged.
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
    # collect_stats (methods at the Mirror-exact budget); shapes listed next to V2_STATS below
    abs_u: dict = field(default_factory=dict)
    sum_g: dict = field(default_factory=dict)
    sq_g: dict = field(default_factory=dict)
    Nin: dict = field(default_factory=dict)     # keyed by stream layer l (l = depth: final RMSNorm output)
    Sin: dict = field(default_factory=dict)     # keyed by stream layer l
    abs_n: dict = field(default_factory=dict)
    Mctx: dict = field(default_factory=dict)
    abs_ctx: dict = field(default_factory=dict)
    sum_ctx: dict = field(default_factory=dict)
    sq_ctx: dict = field(default_factory=dict)
    Mq_h: dict = field(default_factory=dict)
    sum_q: dict = field(default_factory=dict)
    Mctx_h: dict = field(default_factory=dict)
    dims: dict = field(default_factory=dict)

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


# ==============================================================================================
# Methods at the Mirror-exact budget (MUST-07 zero-shot, MUST-09 healed)
# ==============================================================================================
# Every method below compresses the folded weights (norm affine and LayerScale folded, surgery.fold_mlp and
# attention.fold_attn) of the chosen components in the chosen blocks, at a budget from mirror/budget.py
# (SpanBudget; allocation rules in budget.allocate). Input statistics come from collect_stats: one pass of the fit
# images (no mirrored copies) through the folded copy of the original model, all tokens, float64 sums, every
# block's statistics from the original model's activations (as collect_baseline_stats). The conventions of the
# existing asvd docstring hold: fc1 reads u = norm2 output without affine; fc2 reads g = GELU output; wq, wk, wv
# read n = norm1 output without affine; wo reads the attention context c (all heads concatenated, [H dh]).
#
# svd         plain truncated SVD of each matrix, W' = U_r S_r V_r^T (surgery.svd_truncate's arithmetic).
# svdllm      SVD-LLM (Wang et al., arXiv 2403.07378), truncation-aware data whitening only. Checked in the paper:
#             S is the Cholesky factor of X X^T (uncentred second moment of the matrix's input over the
#             calibration tokens), W S = U Sigma V^T, W'_u = U Trunc(Sigma)^(1/2), W'_v = Trunc(Sigma)^(1/2) V^T S^-1
#             (stored that way in info tensors), ratio r (m + n) / (m n). The MLP is exactly the existing asvd
#             (asvd_truncate's arithmetic, same ridge); attention adds wq, wk, wv (whitened by the norm1 output
#             moment) and wo (by the context moment). Departures: (1) a ridge lam = damp * mean(diag M) is added
#             before the Cholesky factorisation (the folded norm outputs are mean-zero over channels, so their
#             moment is singular; the paper does not mention a ridge); (2) no "parameter update with sequential
#             low-rank approximation" (the paper's second component, which is a fine-tuning step; healing in
#             MUST-09 plays that role for every method); (3) ranks follow our matched budget per matrix instead of a
#             uniform compression ratio.
# asvd_scale  ASVD (Yuan et al., arXiv 2312.05821), activation-aware scaling. Checked in the paper: diagonal S with
#             S_ii = (mean_j |X_ij|)^alpha over the input channel's activations (Eq. 8), W S = U Sigma V^T, truncate,
#             W' = U_k Sigma_k V_k^T S^-1; alpha = 0.5 by default; the appendix scans alpha in {0.1, 0.25, 0.5, 1, 2}
#             (0.5 best on OPT-125m). Grid here: {0.25, 0.5, 1.0}. Departures: no sensitivity-based truncation rank
#             search (STRS; ranks follow the matched budget per matrix); a channel with mean |x| = 0 gets
#             S_ii = 1e-12 max S (S must be invertible).
# slicegpt    SliceGPT (Ashkboos et al., arXiv 2401.15024). Checked in the paper: LayerNorm converted to RMSNorm by
#             folding the mean subtraction into every write into the stream and diag(alpha) into the next reads
#             (mirror/fold.py does exactly this, plus LayerScale into the writes); per block an orthogonal Q_l from
#             PCA of C_l = sum X^T X with X "the output of the l-th RMSNorm block" (eigenvectors, descending);
#             reads W_in <- Q^T W_in, writes W_out <- W_out Q; the residual connection gets Q_(l-1)^T Q_l; then the
#             minor components are deleted: rows of W_in, columns of W_out and both sides of the shortcut matrices;
#             Figure 4 shows two rotations per transformer layer (attention input Q1, MLP input Q2, next Q3) and two
#             shortcuts (Q1^T Q2, Q2^T Q3); PCA in double precision; one slicing fraction for all layers by default.
#             The RMSNorm of the paper is x / ||x|| (a constant sqrt(D) folded elsewhere), so after slicing the norm
#             is taken over the kept coordinates with the original scale; in our dense form this holds exactly
#             because the sliced stream is held as a d-vector in the span of the kept directions.
#             What is implemented:
#               both components: the residual stream is sliced over the span. Points a{b} (stream entering block b;
#               PCA of the norm1 output, i.e. the RMSNorm output as in the paper, pca "normed"), m{b} (stream entering
#               the MLP; norm2 output) and e (stream leaving the span; norm1 output of the next block, or the final
#               RMSNorm output). All of wq, wk, wv, fc1 read and wo, fc2 write sliced coordinates; one d_s for all
#               points (budget._slice_alloc). The model returned is a folded copy (mirror/fold.py, final affine kept
#               in the model) whose span blocks are SlicedBlock: the same function as the sliced network, written in
#               dense form (each sliced stream held as x = x_s Q_s^T, a d-vector; reads W P, writes P W, shortcuts
#               x -> P x with P = Q_s Q_s^T). Basis changes counted: the shortcut matrices Q_p,s^T Q_(p+1),s
#               (d_s x d_s, two per block), an entry rotation d x d_s if the span does not start at block 0 (at block 0
#               it is folded into the embeddings, as SliceGPT does) and an exit map d_s x d if the span does not end
#               at the last block (at the model end the final features are read through the head, not counted, as
#               for Mirror-exact).
#               one component: as the existing slicegpt (fc1 reads and fc2 writes only the first d_s coordinates of
#               Q_b; the stream is not sliced), extended to attention-only (wq, wk, wv read and wo writes in Q_b).
#               Basis changes: one d x d rotation of the full stream between consecutive converted blocks, plus a
#               d x d entry or exit that is not a model boundary (budget._slice_layout). Written in absorbed form into
#               the unfolded model, as the existing slicegpt.
#             Hyperparameters: basis "per_block" (SliceGPT) or "shared" (our extension: one PCA basis for the whole
#             span from sum_p C_p / tr C_p, so the shortcuts are identities and cost nothing); pca "normed" (the
#             paper) or "stream" (PCA of the unnormalised stream, as the existing slicegpt).
#             Departures: statistics come from the original model (SliceGPT propagates the calibration data through
#             the already sliced earlier layers); one slice dimension for every point (the paper's default; its
#             appendix also varies it by layer).
# flap        FLAP (An et al., arXiv 2312.11983) and its code (github.com/CASIA-IVA-Lab/FLAP, lib/prune.py and
#             lib/layerwrapper.py, read for this). Fluctuation metric (WIFV, Eq. 5): for each input channel j of an
#             output projection (fc2 for the MLP, wo for attention), S_j = sample variance of x_j over the calibration
#             tokens times ||W[:, j]||^2 (the code treats every token as a sample); here the exact token variance with
#             Bessel's correction, from float64 sums. MLP: hidden neurons are removed (rows of fc1 and b1, columns of
#             fc2). Attention: whole heads are removed (rows of wq, wk, wv and their biases, columns of wo); the code
#             scores a head by the sum of its channel scores (uniform mode) or, after standardising the channel
#             scores of the layer, by their mean (adaptive mode). Bias compensation (Eq. 4): the output bias gets
#             W[:, removed] times the mean of the removed inputs. Allocation (hp allocation): "uniform" (FLAP's
#             UL-UM: the same share per block) and "global" (FLAP's default AL-AM: scores standardised to mean 0 and
#             standard deviation 1 per block and module, Eq. 6, then one global ranking in which a head weighs its
#             4 dh d weights and a neuron its 2 d, as the code's compression_weight = 4 * 128 / 3 does for LLaMA).
#             Departures: the code squares the attention metric before standardising; we follow Eq. 5 without the
#             square; heads are removed in dense form (their rows and columns set to zero; the function equals that
#             of the smaller network), MLP neurons by smaller nn.Linear layers.
# zipit       ZipIt! (Stoica et al., arXiv 2305.03053), used within one model. Checked in the paper: features are
#             matched by the Pearson correlation of their activations over a small image set, computed after the
#             nonlinearity; matching is greedy (repeatedly take the most correlated pair, without replacement); the
#             merge matrix M averages the matched features and the unmerge matrix U = 2 M^T duplicates them, so the
#             layer producing the features gets M W (rows averaged, biases averaged) and the consuming layer W U
#             (columns summed); "repeated matching" (alpha) keeps a merged feature in the pool with similarity alpha
#             times the minimum of its members' similarities ("a small value of alpha typically works best").
#             MLP: hidden neurons are merged by the correlation of their GELU outputs (fc2 inputs): fc1 rows and b1
#             averaged over the group, fc2 columns summed, until the width meets the budget. Attention analogue
#             (ours): within each head, (i) query channels are merged by the correlation of q, with wq rows and bq
#             averaged and wk rows and bk summed (the logits sum_c q_c k_c are then exact when the merged q channels
#             are equal), and (ii) value channels are merged by the correlation of the attention context, with wv
#             rows and bv averaged and wo columns summed (exact when equal, since attention mixes value channels
#             linearly with the same weights). Heads keep their number and their softmax scale. Departures: groups
#             of more than two use the plain group mean (the paper's repeated pairwise averaging weights later
#             members more); alpha = 0 (merged features score 0, so they are re-merged only after every positive
#             correlation is used) stands in for matching without replacement; attention is merged in dense form
#             (the group's smallest index holds the merged row or column, the others are zero).
#
# compress returns (compressed deep copy, info). The input model and the statistics are not modified.
# ==============================================================================================
METHODS_V2 = ("svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit")
HPARAM_GRIDS = {
    "svd": [{}],
    "svdllm": [{"damp": DEFAULT_DAMP}],
    "asvd_scale": [{"alpha": 0.25}, {"alpha": 0.5}, {"alpha": 1.0}],
    "slicegpt": [{"basis": "per_block", "pca": "normed"}, {"basis": "shared", "pca": "normed"},
                 {"basis": "per_block", "pca": "stream"}],
    "flap": [{"allocation": "global"}, {"allocation": "uniform"}],
    "zipit": [{"alpha": 0.0}, {"alpha": 0.1}, {"alpha": 0.5}],
}
DEFAULT_HPARAMS = {"svd": {}, "svdllm": {"damp": DEFAULT_DAMP}, "asvd_scale": {"alpha": 0.5, "damp": DEFAULT_DAMP},
                   "slicegpt": {"basis": "per_block", "pca": "normed"}, "flap": {"allocation": "global"},
                   "zipit": {"alpha": 0.0}}
_HP_KEYS = {"svd": set(), "svdllm": {"damp"}, "asvd_scale": {"alpha", "damp"}, "slicegpt": {"basis", "pca"},
            "flap": {"allocation"}, "zipit": {"alpha"}}
V2_STATS = ("M1", "M2", "abs_g", "S", "abs_u", "sum_g", "sq_g", "Nin", "Sin", "abs_n", "Mctx", "abs_ctx", "sum_ctx",
            "sq_ctx", "Mq_h", "sum_q", "Mctx_h")
# Shapes (d stream width, h MLP width, H heads, dh head width, D = H dh). Keyed by block b unless stated.
#   M1 [d, d] sum u u^T (fc1 input = norm2 output; also the PCA moment at point m{b})    abs_u [d] sum |u|
#   M2 [h, h] sum g g^T (fc2 input = GELU output)       abs_g, sum_g, sq_g [h]  sum |g|, sum g, sum g^2
#   S  [d, d] sum x x^T, x the folded stream entering the MLP (unnormalised)
#   Nin[l] [d, d] sum n n^T, n the RMSNorm output at stream layer l (l < L: norm1 output of block l = input of wq,
#        wk, wv; l = L: final RMSNorm output); Sin[l] the same for the unnormalised stream; keyed by layer l
#   abs_n [d] sum |n| (norm1 output)
#   Mctx [D, D] sum c c^T, c the attention context (wo input); abs_ctx, sum_ctx, sq_ctx [D]
#   Mq_h [H, dh, dh] per-head sum q_h q_h^T; sum_q [D];  Mctx_h [H, dh, dh] per-head sum c_h c_h^T


def needs_v2(methods, components) -> set:
    """Statistics (names in V2_STATS) that `methods` need for `components`."""
    from .budget import normalise_components

    comps = normalise_components(components)
    mlp, att = "mlp" in comps, "attn" in comps
    out = set()
    for m in methods:
        if m not in METHODS_V2:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS_V2}")
        if m == "svdllm":
            out |= ({"M1", "M2"} if mlp else set()) | ({"Nin", "Mctx"} if att else set())
        elif m == "asvd_scale":
            out |= ({"abs_u", "abs_g"} if mlp else set()) | ({"abs_n", "abs_ctx"} if att else set())
        elif m == "slicegpt":
            if mlp and att:
                out |= {"Nin", "M1", "Sin", "S"}
            elif mlp:
                out |= {"M1", "S"}
            else:
                out |= {"Nin", "Sin"}
        elif m == "flap":
            out |= ({"sum_g", "sq_g"} if mlp else set()) | ({"sum_ctx", "sq_ctx"} if att else set())
        elif m == "zipit":
            out |= ({"M2", "sum_g"} if mlp else set()) | ({"Mq_h", "sum_q", "Mctx_h", "sum_ctx"} if att else set())
    return out


def stats_memory_bytes(dims: dict, n_blocks: int, methods=METHODS_V2, components=("mlp", "attn")) -> int:
    """float64 bytes held by collect_stats for `n_blocks` blocks (Nin and Sin hold n_blocks + 1 layers)."""
    d, h, H, dh, D = dims["d"], dims["h"], dims["H"], dims["dh"], dims["D"]
    size = {"M1": d * d, "M2": h * h, "abs_g": h, "S": d * d, "abs_u": d, "sum_g": h, "sq_g": h, "Nin": d * d,
            "Sin": d * d, "abs_n": d, "Mctx": D * D, "abs_ctx": D, "sum_ctx": D, "sq_ctx": D, "Mq_h": H * dh * dh,
            "sum_q": D, "Mctx_h": H * dh * dh}
    total = 0
    for name in needs_v2(methods, components):
        total += size[name] * (n_blocks + 1 if name in ("Nin", "Sin") else n_blocks)
    return 8 * total


@torch.no_grad()
def collect_stats(adapter, loader, device, blocks, methods=METHODS_V2, components=("mlp", "attn"),
                  stats_device=None, to_cpu: bool = True) -> BaselineStats:
    """One pass of `loader` (fit images, no mirrored copies) through the folded copy (mirror/fold.py, final affine
    kept in the model) of adapter.model, collecting the float64 sums `methods` need for `components` in `blocks`
    (V2_STATS; Nin and Sin at stream layers blocks and blocks + 1, layer L meaning the final norm). Same conventions
    as collect_baseline_stats (its M1, M2, abs_g and S are computed the same way). Sums are held on `stats_device`
    (default: `device`) and moved to the CPU at the end with `to_cpu`. stats_memory_bytes gives the memory needed.
    The original model is not modified."""
    from .budget import model_dims, normalise_components
    from .data import normalize_for
    from .fold import fold_copy

    comps = normalise_components(components)
    need = needs_v2(methods, comps)
    blocks = sorted({int(b) for b in blocks})
    dims = model_dims(adapter.model)
    d, h, H, dh, D, L = dims["d"], dims["h"], dims["H"], dims["dh"], dims["D"], dims["depth"]
    if not blocks or blocks[0] < 0 or blocks[-1] >= L:
        raise ValueError(f"blocks must lie in [0, {L - 1}]")
    layers = sorted(set(blocks) | {b + 1 for b in blocks}) if need & {"Nin", "Sin"} else []
    folded, finfo = fold_copy(adapter.model, final_norm_affine=False)
    folded.eval()
    sdev = device if stats_device is None else stats_device
    kw = dict(dtype=torch.float64, device=sdev)
    st = BaselineStats(blocks=blocks)
    shapes = {"M1": (d, d), "M2": (h, h), "abs_g": (h,), "S": (d, d), "abs_u": (d,), "sum_g": (h,), "sq_g": (h,),
              "abs_n": (d,), "Mctx": (D, D), "abs_ctx": (D,), "sum_ctx": (D,), "sq_ctx": (D,), "Mq_h": (H, dh, dh),
              "sum_q": (D,), "Mctx_h": (H, dh, dh)}
    for name in need - {"Nin", "Sin"}:
        getattr(st, name).update({b: torch.zeros(*shapes[name], **kw) for b in blocks})
    for name in need & {"Nin", "Sin"}:
        getattr(st, name).update({l: torch.zeros(d, d, **kw) for l in layers})
    worst_mean = [0.0]
    counter = {"tokens": 0}
    handles = []

    def f64(t, n):
        return t.reshape(-1, n).to(device=sdev, dtype=torch.float64)

    def count(_mod, args):
        t = args[0]
        counter["tokens"] += int(t.shape[0]) * int(t.shape[1])

    handles.append(folded.blocks[blocks[0]].register_forward_pre_hook(count))
    for l in layers:
        if l >= L:
            continue
        blk = folded.blocks[l]
        if "Sin" in need:
            def pre_block(_mod, args, l=l):
                X = f64(args[0], d)
                st.Sin[l] += X.T @ X

            handles.append(blk.register_forward_pre_hook(pre_block))
        if "Nin" in need:
            def post_norm1(_mod, _args, out, l=l):
                N = f64(out, d)
                st.Nin[l] += N.T @ N

            handles.append(blk.norm1.register_forward_hook(post_norm1))
    for b in blocks:
        blk = folded.blocks[b]
        if "abs_n" in need:
            def post_norm1_abs(_mod, _args, out, b=b):
                st.abs_n[b] += f64(out, d).abs().sum(0)

            handles.append(blk.norm1.register_forward_hook(post_norm1_abs))
        if "S" in need:
            def pre_norm2(_mod, args, b=b):
                X = f64(args[0], d)
                st.S[b] += X.T @ X
                rel = float((X.mean(dim=1).abs() / X.pow(2).mean(dim=1).sqrt().clamp_min(1e-300)).max())
                worst_mean[0] = max(worst_mean[0], rel)

            handles.append(blk.norm2.register_forward_pre_hook(pre_norm2))
        if need & {"M1", "abs_u"}:
            def pre_fc1(_mod, args, b=b):
                U = f64(args[0], d)
                if "M1" in need:
                    st.M1[b] += U.T @ U
                if "abs_u" in need:
                    st.abs_u[b] += U.abs().sum(0)

            handles.append(blk.mlp.fc1.register_forward_pre_hook(pre_fc1))
        if need & {"M2", "abs_g", "sum_g", "sq_g"}:
            def pre_fc2(_mod, args, b=b):
                G = f64(args[0], h)
                if "M2" in need:
                    st.M2[b] += G.T @ G
                if "abs_g" in need:
                    st.abs_g[b] += G.abs().sum(0)
                if "sum_g" in need:
                    st.sum_g[b] += G.sum(0)
                if "sq_g" in need:
                    st.sq_g[b] += (G * G).sum(0)

            handles.append(blk.mlp.fc2.register_forward_pre_hook(pre_fc2))
        if need & {"Mq_h", "sum_q"}:
            def post_qkv(_mod, _args, out, b=b):
                Q = f64(out[..., :D], D)
                if "sum_q" in need:
                    st.sum_q[b] += Q.sum(0)
                if "Mq_h" in need:
                    Qh = Q.reshape(-1, H, dh)
                    st.Mq_h[b] += torch.einsum("nhi,nhj->hij", Qh, Qh)

            handles.append(blk.attn.qkv.register_forward_hook(post_qkv))
        if need & {"Mctx", "abs_ctx", "sum_ctx", "sq_ctx", "Mctx_h"}:
            def pre_proj(_mod, args, b=b):
                Cx = f64(args[0], D)
                if "Mctx" in need:
                    st.Mctx[b] += Cx.T @ Cx
                if "abs_ctx" in need:
                    st.abs_ctx[b] += Cx.abs().sum(0)
                if "sum_ctx" in need:
                    st.sum_ctx[b] += Cx.sum(0)
                if "sq_ctx" in need:
                    st.sq_ctx[b] += (Cx * Cx).sum(0)
                if "Mctx_h" in need:
                    Ch = Cx.reshape(-1, H, dh)
                    st.Mctx_h[b] += torch.einsum("nhi,nhj->hij", Ch, Ch)

            handles.append(blk.attn.proj.register_forward_pre_hook(pre_proj))
    final_rms = folded.norm.rms if isinstance(folded.norm, nn.Sequential) else folded.norm
    n_img = 0
    try:
        for imgs, _ in loader:
            x = normalize_for(adapter, imgs.to(device, non_blocking=True))
            t = adapter.forward_tokens(x, folded)
            if L in layers:
                if "Sin" in need:
                    X = f64(t, d)
                    st.Sin[L] += X.T @ X
                if "Nin" in need:
                    N = f64(final_rms(t), d)
                    st.Nin[L] += N.T @ N
            n_img += int(x.shape[0])
    finally:
        for hd in handles:
            hd.remove()
    del folded
    st.n_tokens, st.n_images = counter["tokens"], n_img
    st.dims = dict(dims)
    st.info = {"statistics": sorted(need), "methods": list(methods), "components": list(comps),
               "blocks": [b + 1 for b in blocks], "stream_layers": layers, "n_images": n_img,
               "n_tokens": st.n_tokens, "model_form": "folded copy (mirror/fold.py, final affine kept in model)",
               "fold_eps": finfo.get("eps"), "tokens": "all (CLS, registers, patches), fit images only, no mirrors",
               "precision": "model dtype forward, float64 sums", "stats_device": str(sdev)}
    if "S" in need:
        st.info["max_rel_channel_mean_of_folded_stream"] = worst_mean[0]
    if to_cpu:
        for name in V2_STATS:
            setattr(st, name, {k: t.cpu() for k, t in getattr(st, name).items()})
    return st


# ----------------------------------------------------------------------------------------------
# Matrix helpers (float64)
# ----------------------------------------------------------------------------------------------
_ATTN_KEYS = {"wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo"}


def _stat(stats: BaselineStats, name: str, key: int) -> torch.Tensor:
    table = getattr(stats, name, None) or {}
    if key not in table:
        raise KeyError(f"statistic {name}[{key}] missing; collect it with collect_stats (methods/components/blocks)")
    return table[key]


def _input_stats(name: str, b: int):
    """(second-moment statistic, abs-sum statistic, key) of the input of matrix `name` in block b."""
    return {"fc1": ("M1", "abs_u", b), "fc2": ("M2", "abs_g", b), "wq": ("Nin", "abs_n", b),
            "wk": ("Nin", "abs_n", b), "wv": ("Nin", "abs_n", b), "wo": ("Mctx", "abs_ctx", b)}[name]


def lowrank_factors(method: str, W: torch.Tensor, r: int, M: torch.Tensor | None = None,
                    mean_abs: torch.Tensor | None = None, alpha: float = 0.5, damp: float = DEFAULT_DAMP):
    """(W', U_f, V_f, info) with W' = U_f V_f of rank r for W [out, in] (float64).

    svd: W = U S V^T, W' = U_r S_r V_r^T (surgery.svd_truncate's arithmetic); U_f = U_r S_r^(1/2),
        V_f = S_r^(1/2) V_r^T.
    svdllm: the arithmetic of asvd_truncate (L = cholesky(M + lam I), W L = U S V^T, W' = U_r S_r V_r^T L^-1);
        U_f = U_r S_r^(1/2), V_f = S_r^(1/2) V_r^T L^-1 (SVD-LLM's W'_u, W'_v).
    asvd_scale: s = mean_abs^alpha (zeros raised to 1e-12 max s), W diag(s) = U S V^T, W' = U_r S_r V_r^T diag(s)^-1;
        U_f = U_r S_r^(1/2), V_f = S_r^(1/2) V_r^T diag(s)^-1.
    info: rank and the share of (whitened or scaled) energy discarded."""
    W = W.to(torch.float64)
    r = int(r)
    if method == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = max(0, min(r, S.numel()))
        W_new = (U[:, :r] * S[:r]) @ Vh[:r]
        sq = S[:r].sqrt()
        U_f, V_f = U[:, :r] * sq, sq[:, None] * Vh[:r]
        extra = {}
    elif method == "svdllm":
        M = M.to(W)
        n = M.shape[0]
        if W.shape[1] != n:
            raise ValueError(f"W {tuple(W.shape)} does not take inputs of size {n}")
        Ms = 0.5 * (M + M.T)
        lam = float(damp) * float(torch.diagonal(Ms).mean())
        L, err = torch.linalg.cholesky_ex(Ms + lam * torch.eye(n, dtype=W.dtype, device=W.device))
        if int(err) != 0:
            raise RuntimeError(f"Cholesky failed (damp {damp}); increase damp")
        U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
        r = max(0, min(r, S.numel()))
        B = (U[:, :r] * S[:r]) @ Vh[:r]
        W_new = torch.linalg.solve_triangular(L, B, upper=False, left=False)     # X L = B
        sq = S[:r].sqrt()
        U_f = U[:, :r] * sq
        V_f = torch.linalg.solve_triangular(L, sq[:, None] * Vh[:r], upper=False, left=False)
        extra = {"ridge": lam}
    elif method == "asvd_scale":
        s = mean_abs.to(W).clamp_min(0).pow(float(alpha))
        s = torch.where(s > 0, s, s.max().clamp_min(1e-300) * 1e-12)
        U, S, Vh = torch.linalg.svd(W * s[None, :], full_matrices=False)
        r = max(0, min(r, S.numel()))
        W_new = ((U[:, :r] * S[:r]) @ Vh[:r]) / s[None, :]
        sq = S[:r].sqrt()
        U_f, V_f = U[:, :r] * sq, (sq[:, None] * Vh[:r]) / s[None, :]
        extra = {"alpha": float(alpha)}
    else:
        raise ValueError(f"not a low-rank method: {method!r}")
    e = S * S
    tot = float(e.sum())
    info = {"rank": r, "discarded_energy": float(1.0 - float(e[:r].sum()) / tot) if tot > 0 else 0.0, **extra}
    return W_new, U_f, V_f, info


def corr_from_moments(M: torch.Tensor, s: torch.Tensor, n: int) -> torch.Tensor:
    """Pearson correlation matrix from sum x x^T, sum x and the count n (zero-variance channels get 0)."""
    M, s = M.to(torch.float64), s.to(torch.float64)
    mu = s / max(int(n), 1)
    cov = M / max(int(n), 1) - torch.outer(mu, mu)
    sd = torch.diagonal(cov).clamp_min(0).sqrt()
    den = torch.outer(sd, sd)
    return torch.where(den > 0, cov / den.clamp_min(1e-300), torch.zeros_like(cov))


@torch.no_grad()
def zipit_groups(C: torch.Tensor, n_keep: int, alpha: float = 0.0, device=None) -> list:
    """Greedy ZipIt matching within one feature set: repeatedly merge the most similar pair of current features until
    `n_keep` remain. A merged feature stays in the pool with similarity alpha * min(similarities of the two) to every
    other feature (ZipIt's repeated matching). float64 on `device` (default CPU); ties go to the lowest row, then the
    lowest column (torch.max and torch.argmax return the first maximal index). Row maxima are kept up to date
    incrementally, or recomputed in full when many rows are affected by a merge. Returns the groups (sorted lists of
    original indices), ordered by their smallest index."""
    S = C.detach().to(device="cpu" if device is None else device, dtype=torch.float64)
    n = int(S.shape[0])
    n_keep = int(n_keep)
    if not 1 <= n_keep <= n:
        raise ValueError(f"n_keep must be in [1, {n}]")
    groups = [[i] for i in range(n)]
    if n_keep == n:
        return groups
    neg = float("-inf")
    sim = 0.5 * (S + S.T)
    sim.fill_diagonal_(neg)
    alive = torch.ones(n, dtype=torch.bool, device=sim.device)
    rowmax, rowarg = sim.max(dim=1)
    many = max(8, n // 32)
    for _ in range(n - n_keep):
        r = int(torch.argmax(rowmax))
        c = int(rowarg[r])
        i, j = min(r, c), max(r, c)
        groups[i] = groups[i] + groups[j]
        groups[j] = []
        alive[j] = False
        new = torch.minimum(sim[i], sim[j])
        fin = torch.isfinite(new)
        new[fin] = new[fin] * float(alpha)
        new[~alive] = neg
        new[i] = neg
        sim[i, :] = new
        sim[:, i] = new
        sim[j, :] = neg
        sim[:, j] = neg
        rowmax[j] = neg
        aff = (rowarg == i) | (rowarg == j)
        aff[i] = True
        aff[j] = False
        aff &= alive
        idx = aff.nonzero().flatten()
        if idx.numel() > many:
            rowmax, rowarg = sim.max(dim=1)
            continue
        if idx.numel():
            mx, am = sim[idx].max(dim=1)
            rowmax[idx], rowarg[idx] = mx, am
        upd = alive & ~aff & (new > rowmax)
        rowmax[upd] = new[upd]
        rowarg[upd] = i
    return sorted((sorted(g) for g in groups if g), key=lambda g: g[0])


def flap_variance(sq: torch.Tensor, s: torch.Tensor, n: int) -> torch.Tensor:
    """Sample variance over n tokens with Bessel's correction, from sum x^2 and sum x (clamped at 0)."""
    n = int(n)
    return ((sq.to(torch.float64) - s.to(torch.float64) ** 2 / n) / max(n - 1, 1)).clamp_min(0)


def _standardise(x: torch.Tensor) -> torch.Tensor:
    sd = x.std()
    return (x - x.mean()) / sd if float(sd) > 0 else torch.zeros_like(x)


# ----------------------------------------------------------------------------------------------
# SliceGPT: residual-stream slicing in dense form
# ----------------------------------------------------------------------------------------------
class SlicedBlock(nn.Module):
    """One block of a folded model (mirror/fold.py) with SliceGPT's residual slicing written in dense form.

    The weights of `block` are already projected (reads W P_in, writes P_out W). The forward pass is
        x <- x Q_e Q_e^T                 (entry: only in the first block of the span)
        x <- x Q_m Q_m^T + attn(norm1(x)) (attention shortcut; skipped when Q_m is None, the shared basis)
        x <- x Q_n Q_n^T + mlp(norm2(x))  (MLP shortcut; skipped when Q_n is None)
    with sliced orthonormal bases Q [d, d_s] held as fixed buffers. Every stream it produces lies in the span of the
    current basis, so it is the sliced network of SliceGPT written as d-vectors (x = x_s Q^T): the shortcut
    Q_n Q_n^T x equals the sliced shortcut Q_n^T Q_m applied to x_s, and the RMSNorm of the d-vector equals the
    paper's RMSNorm of the sliced vector with the original scale."""

    def __init__(self, block: nn.Module, entry=None, attn_shortcut=None, mlp_shortcut=None):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = block.norm1, block.attn, block.norm2, block.mlp
        self.ls1 = getattr(block, "ls1", None) or nn.Identity()
        self.ls2 = getattr(block, "ls2", None) or nn.Identity()
        ref = block.mlp.fc1.weight
        for name, Q in (("entry", entry), ("attn_shortcut", attn_shortcut), ("mlp_shortcut", mlp_shortcut)):
            self.register_buffer(name, None if Q is None else Q.detach().to(dtype=ref.dtype, device=ref.device)
                                 .contiguous())

    @staticmethod
    def _project(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        Qx = Q.to(x.dtype)
        return (x @ Qx) @ Qx.T

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.entry is not None:
            x = self._project(x, self.entry)
        a = self.ls1(self.attn(self.norm1(x)))
        x = (x if self.attn_shortcut is None else self._project(x, self.attn_shortcut)) + a
        m = self.ls2(self.mlp(self.norm2(x)))
        return (x if self.mlp_shortcut is None else self._project(x, self.mlp_shortcut)) + m


def _point_moment(stats: BaselineStats, point: str, pca: str, blocks: list, depth: int) -> torch.Tensor:
    """PCA moment of a stream point: a{b} stream entering block b, m{b} stream entering its MLP, e stream leaving
    the span. pca "normed": the RMSNorm output there (SliceGPT); "stream": the unnormalised stream."""
    normed = pca == "normed"
    if pca not in ("normed", "stream"):
        raise ValueError(f"slicegpt pca must be 'normed' or 'stream', not {pca!r}")
    if point == "e":
        layer = blocks[-1] + 1
        return _stat(stats, "Nin" if normed else "Sin", layer)
    b = int(point[1:])
    if point[0] == "a":
        return _stat(stats, "Nin" if normed else "Sin", b)
    return _stat(stats, "M1" if normed else "S", b)


def _slice_bases(stats, alloc, pca: str, blocks: list, depth: int) -> dict:
    """basis_key -> sliced orthonormal basis [d, d_s] (float64, CPU), from pca_basis of the point moments; the
    shared basis is the PCA of sum_p C_p / tr(C_p) over every point of the span."""
    ds = next(a["d_s"] for a in alloc if a["kind"] == "slice")
    points = sorted({a["point"] for a in alloc if a["kind"] == "slice"})
    keys = {a["basis_key"] for a in alloc if a["kind"] == "slice"}
    out = {}
    if keys == {"Q.span"}:
        tot = None
        for p in points:
            C = centre_moment_t(_point_moment(stats, p, pca, blocks, depth).to(torch.float64))
            tr = float(torch.diagonal(C).sum())
            C = C / tr if tr > 0 else C
            tot = C if tot is None else tot + C
        out["Q.span"] = pca_basis(tot)[0][:, :ds].contiguous()
    else:
        for p in points:
            out[f"Q.{p}"] = pca_basis(_point_moment(stats, p, pca, blocks, depth))[0][:, :ds].contiguous()
    return out


def _proj(Q: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    Q = Q.to(dtype=torch.float64, device=like.device)
    return Q @ Q.T


@torch.no_grad()
def _compress_slicegpt(model, blocks, comps, budget, stats, hp):
    from .budget import allocate

    alloc = allocate("slicegpt", budget, basis=hp["basis"])
    depth = budget.dims["depth"]
    Qs = _slice_bases(stats, alloc, hp["pca"], blocks, depth)
    per = [dict(a) for a in alloc if a["kind"] == "slice"]
    basis = [dict(a) for a in alloc if a["kind"] == "basis"]
    key = {(a["block"], a["name"]): a["basis_key"] for a in per}
    if set(comps) == {"mlp", "attn"}:
        from .fold import fold_copy

        m, _ = fold_copy(model, final_norm_affine=False)
        shared = hp["basis"] == "shared"
        for i, b in enumerate(blocks):
            blk = m.blocks[b]
            ref = blk.mlp.fc1.weight
            Pa, Pm = _proj(Qs[key[(b, "wq")]], ref), _proj(Qs[key[(b, "wo")]], ref)
            Pn = _proj(Qs[key[(b, "fc2")]], ref)
            qkv, proj, fc1, fc2 = blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2
            qkv.weight.copy_((qkv.weight.to(torch.float64) @ Pa).to(qkv.weight))
            proj.weight.copy_((Pm @ proj.weight.to(torch.float64)).to(proj.weight))
            if proj.bias is not None:
                proj.bias.copy_((Pm @ proj.bias.to(torch.float64)).to(proj.bias))
            fc1.weight.copy_((fc1.weight.to(torch.float64) @ Pm).to(fc1.weight))
            fc2.weight.copy_((Pn @ fc2.weight.to(torch.float64)).to(fc2.weight))
            if fc2.bias is not None:
                fc2.bias.copy_((Pn @ fc2.bias.to(torch.float64)).to(fc2.bias))
            m.blocks[b] = SlicedBlock(blk, entry=Qs[key[(b, "wq")]] if i == 0 else None,
                                      attn_shortcut=None if shared else Qs[key[(b, "wo")]],
                                      mlp_shortcut=None if shared else Qs[key[(b, "fc2")]])
        form = ("folded copy (mirror/fold.py, final affine in the model); span blocks are SlicedBlock (dense form of "
                "the sliced residual stream)")
    else:
        m = copy.deepcopy(model)
        for b in blocks:
            blk = m.blocks[b]
            if comps == ("mlp",):
                f = fold_mlp(blk)
                P = _proj(Qs[key[(b, "fc1")]], f["W1"])
                write_absorbed_mlp(blk, f["W1"] @ P, f["b1"], P @ f["W2"], P @ f["b2"])
            else:
                from .attention import fold_attn, write_absorbed_attn

                f = fold_attn(blk)
                P = _proj(Qs[key[(b, "wq")]], f["Wq"])
                for k_ in ("Wq", "Wk", "Wv"):
                    f[k_] = f[k_] @ P
                f["Wo"], f["bo"] = P @ f["Wo"], P @ f["bo"]
                write_absorbed_attn(blk, **f)
        form = "absorbed weights in the unfolded model (stream not sliced), as the existing slicegpt"
    tensors = {k_: v.to(torch.float64).cpu() for k_, v in Qs.items()}
    return m, per, basis, tensors, form


# ----------------------------------------------------------------------------------------------
# Low-rank, FLAP and ZipIt application
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _compress_lowrank(model, method, blocks, comps, budget, stats, hp, keep_tensors):
    from .attention import fold_attn, write_absorbed_attn
    from .budget import allocate

    alloc = allocate(method, budget)
    by = {(a["block"], a["name"]): dict(a) for a in alloc}
    m = copy.deepcopy(model)
    n_tok = max(int(stats.n_tokens), 1) if stats is not None else 1
    tensors = {}

    def one(b, name, W):
        a = by[(b, name)]
        if a["dense_storage"]:
            a.update({"discarded_energy": 0.0})
            return W
        mom, absn, key = _input_stats(name, b)
        M = _dev(_stat(stats, mom, key), W) if method == "svdllm" else None
        mean_abs = _dev(_stat(stats, absn, key), W) / n_tok if method == "asvd_scale" else None
        W_new, U_f, V_f, inf = lowrank_factors(method, W, a["rank"], M=M, mean_abs=mean_abs,
                                               alpha=hp.get("alpha", 0.5), damp=hp.get("damp", DEFAULT_DAMP))
        a.update({k_: v for k_, v in inf.items() if k_ != "rank"})
        if keep_tensors:
            tensors[f"{b}.{name}.U"] = U_f.to(torch.float32).cpu()
            tensors[f"{b}.{name}.V"] = V_f.to(torch.float32).cpu()
        return W_new

    for b in blocks:
        blk = m.blocks[b]
        if "mlp" in comps:
            f = fold_mlp(blk)
            write_absorbed_mlp(blk, one(b, "fc1", f["W1"]), f["b1"], one(b, "fc2", f["W2"]), f["b2"])
        if "attn" in comps:
            f = fold_attn(blk)
            for name, k_ in _ATTN_KEYS.items():
                f[k_] = one(b, name, f[k_])
            write_absorbed_attn(blk, **f)
    per = [by[(a["block"], a["name"])] for a in alloc]
    return m, per, [], tensors, "absorbed weights in the unfolded model (norm affine and LayerScale folded)"


def flap_scores(model, blocks, comps, stats, standardise: bool) -> dict:
    """Per block: "mlp" [h] neuron scores and "attn" [H] head scores (FLAP's WIFV on fc2 and wo inputs).
    standardise=False: raw scores, heads by the sum of their channel scores (FLAP code, uniform mode);
    True: z-scores within the block and module, heads by the mean of their standardised channel scores (AL-AM)."""
    from .attention import attn_dims, fold_attn

    n = int(stats.n_tokens)
    out = {}
    for b in blocks:
        blk = model.blocks[b]
        out[b] = {}
        if "mlp" in comps:
            W2 = fold_mlp(blk)["W2"]
            s = flap_variance(_stat(stats, "sq_g", b), _stat(stats, "sum_g", b), n).to(W2) * W2.pow(2).sum(0)
            out[b]["mlp"] = (_standardise(s) if standardise else s).cpu()
        if "attn" in comps:
            H, dh = attn_dims(blk)
            Wo = fold_attn(blk)["Wo"]
            s = flap_variance(_stat(stats, "sq_ctx", b), _stat(stats, "sum_ctx", b), n).to(Wo) * Wo.pow(2).sum(0)
            out[b]["attn_channels"] = s.cpu()
            out[b]["attn"] = (_standardise(s).reshape(H, dh).mean(1) if standardise else s.reshape(H, dh).sum(1)).cpu()
    return out


def _top(scores: torch.Tensor, n: int) -> list:
    order = torch.argsort(-scores.to(torch.float64), stable=True)
    return sorted(int(i) for i in order[:int(n)])


@torch.no_grad()
def _compress_flap(model, blocks, comps, budget, stats, hp):
    from .attention import attn_dims, fold_attn, write_absorbed_attn
    from .budget import allocate

    mode = hp["allocation"]
    scores = flap_scores(model, blocks, comps, stats, standardise=(mode == "global"))
    alloc = allocate("flap", budget, allocation=mode, scores=scores)
    by = {(a["block"], a["name"]): dict(a) for a in alloc}
    m = copy.deepcopy(model)
    n = max(int(stats.n_tokens), 1)
    for b in blocks:
        blk = m.blocks[b]
        if "mlp" in comps:
            a = by[(b, "fc1")]
            keep = a["kept"] if "kept" in a else _top(scores[b]["mlp"], a["n"])
            f = fold_mlp(blk)
            h = f["W1"].shape[0]
            removed = sorted(set(range(h)) - set(keep))
            mean_g = _dev(_stat(stats, "sum_g", b), f["W2"]) / n
            kt = torch.as_tensor(keep, dtype=torch.long, device=f["W1"].device)
            rt = torch.as_tensor(removed, dtype=torch.long, device=f["W1"].device)
            b2 = f["b2"] + (f["W2"][:, rt] @ mean_g[rt] if removed else 0.0)
            _shrink_mlp(blk, len(keep))
            write_absorbed_mlp(blk, f["W1"][kt], f["b1"][kt], f["W2"][:, kt], b2)
            for name in ("fc1", "fc2"):
                by[(b, name)].update({"kept": [int(x) for x in keep], "n": len(keep),
                                      "bias_compensation": name == "fc2" and bool(removed)})
        if "attn" in comps:
            H, dh = attn_dims(blk)
            a = by[(b, "wq")]
            heads = a["heads_kept"] if "heads_kept" in a else _top(scores[b]["attn"], a["heads"])
            f = fold_attn(blk)
            gone = sorted(set(range(H)) - set(heads))
            ch_gone = [hh * dh + c for hh in gone for c in range(dh)]
            if ch_gone:
                ct = torch.as_tensor(ch_gone, dtype=torch.long, device=f["Wo"].device)
                mean_c = _dev(_stat(stats, "sum_ctx", b), f["Wo"]) / n
                f["bo"] = f["bo"] + f["Wo"][:, ct] @ mean_c[ct]
                for k_ in ("Wq", "Wk", "Wv"):
                    f[k_][ct] = 0.0
                for k_ in ("bq", "bk", "bv"):
                    f[k_][ct] = 0.0
                f["Wo"][:, ct] = 0.0
            write_absorbed_attn(blk, **f)
            kept = [hh * dh + c for hh in heads for c in range(dh)]
            for name in _ATTN_KEYS:
                by[(b, name)].update({"heads_kept": [int(x) for x in heads], "heads": len(heads), "kept": kept,
                                      "n": len(kept), "bias_compensation": name == "wo" and bool(gone),
                                      "dense_form": "removed heads: rows of wq, wk, wv (and biases) and columns of wo "
                                                    "set to zero"})
    per = [by[(a["block"], a["name"])] for a in alloc]
    return m, per, [], {}, ("absorbed weights in the unfolded model; MLP neurons removed (smaller nn.Linear); "
                            "heads removed in dense form (zero rows and columns)")


@torch.no_grad()
def _compress_zipit(model, blocks, comps, budget, stats, hp):
    from .attention import attn_dims, fold_attn, write_absorbed_attn
    from .budget import allocate

    alloc = allocate("zipit", budget)
    by = {(a["block"], a["name"]): dict(a) for a in alloc}
    m = copy.deepcopy(model)
    n = int(stats.n_tokens)
    alpha = float(hp["alpha"])
    for b in blocks:
        blk = m.blocks[b]
        if "mlp" in comps:
            nk = by[(b, "fc1")]["n"]
            C = corr_from_moments(_stat(stats, "M2", b), _stat(stats, "sum_g", b), n)
            groups = zipit_groups(C, nk, alpha, device=blk.mlp.fc1.weight.device)
            f = fold_mlp(blk)
            W1 = torch.stack([f["W1"][g].mean(0) for g in groups])
            b1 = torch.stack([f["b1"][g].mean() for g in groups])
            W2 = torch.stack([f["W2"][:, g].sum(1) for g in groups], dim=1)
            _shrink_mlp(blk, len(groups))
            write_absorbed_mlp(blk, W1, b1, W2, f["b2"])
            by[(b, "fc1")].update({"groups": groups, "rule": "rows of fc1 and b1 averaged over each group (unmerge: "
                                                             "the merged neuron feeds every member's fc2 column)"})
            by[(b, "fc2")].update({"groups": groups, "rule": "columns of fc2 summed over each group; b2 unchanged"})
        if "attn" in comps:
            H, dh = attn_dims(blk)
            wqk, wv = by[(b, "wq")]["per_head"], by[(b, "wv")]["per_head"]
            Mq, sq_ = _stat(stats, "Mq_h", b), _stat(stats, "sum_q", b)
            Mc, sc_ = _stat(stats, "Mctx_h", b), _stat(stats, "sum_ctx", b)
            gq, gv = [], []
            for hh in range(H):
                sl = slice(hh * dh, (hh + 1) * dh)
                cq, cv = corr_from_moments(Mq[hh], sq_[sl], n), corr_from_moments(Mc[hh], sc_[sl], n)
                gq += [[hh * dh + i for i in g] for g in zipit_groups(cq, wqk, alpha)]
                gv += [[hh * dh + i for i in g] for g in zipit_groups(cv, wv, alpha)]
            f = fold_attn(blk)
            o = {k_: torch.zeros_like(v) for k_, v in f.items() if k_ != "bo"}
            o["bo"] = f["bo"]
            for g in gq:
                r = g[0]
                o["Wq"][r], o["bq"][r] = f["Wq"][g].mean(0), f["bq"][g].mean()
                o["Wk"][r], o["bk"][r] = f["Wk"][g].sum(0), f["bk"][g].sum()
            for g in gv:
                r = g[0]
                o["Wv"][r], o["bv"][r] = f["Wv"][g].mean(0), f["bv"][g].mean()
                o["Wo"][:, r] = f["Wo"][:, g].sum(1)
            write_absorbed_attn(blk, **o)
            dense = "the group's smallest index holds the merged row (column for wo); the other rows (columns) are zero"
            by[(b, "wq")].update({"groups": gq, "rule": "rows of wq and bq averaged over each query group",
                                  "dense_form": dense})
            by[(b, "wk")].update({"groups": gq, "rule": "rows of wk and bk summed over each query group",
                                  "dense_form": dense})
            by[(b, "wv")].update({"groups": gv, "rule": "rows of wv and bv averaged over each value group",
                                  "dense_form": dense})
            by[(b, "wo")].update({"groups": gv, "rule": "columns of wo summed over each value group; bo unchanged",
                                  "dense_form": dense})
    per = [by[(a["block"], a["name"])] for a in alloc]
    return m, per, [], {}, ("absorbed weights in the unfolded model; MLP neurons merged (smaller nn.Linear); attention "
                            "channels merged within heads in dense form")


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------
def resolve_hparams(method: str, hparams: dict | None) -> dict:
    if method not in METHODS_V2:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS_V2}")
    hp = {**DEFAULT_HPARAMS[method], **(hparams or {})}
    bad = set(hp) - _HP_KEYS[method]
    if bad:
        raise ValueError(f"{method}: unknown hyperparameters {sorted(bad)} (allowed {sorted(_HP_KEYS[method])})")
    return hp


def compress(model: nn.Module, method: str, blocks, components, budget, stats: BaselineStats | None,
             hparams: dict | None = None, *, keep_tensors: bool = True, verbose: bool = True):
    """Compress `components` of `blocks` of `model` with `method` at `budget` (mirror/budget.SpanBudget, built for
    the same blocks and components). Returns (compressed deep copy, info); `model` and `stats` are not modified.

    info: method, hparams (resolved), components, blocks (0-based), params_matrices, params_basis_change,
    params_total, budget_total, gap = (params_total - budget_total) / budget_total (0 <= gap <= 0.01 where
    reachable, that is where the method's smallest step gap_step is at most 1% of the budget, gap_reachable;
    otherwise within one step; gap_ok and gap_note say which), params_biases (dense count, the same for every
    method, never in the totals), per_matrix (one dict per converted matrix: block, name, kind "lowrank" |
    "width" | "slice" | "merge", params and the kind's fields: rank; kept; d_in, d_out, basis_key; groups and rule), basis (SliceGPT's basis
    and shortcut matrices), tensors (low-rank factors "<b>.<name>.U" [out, r] and ".V" [r, in] with W' = U V;
    SliceGPT bases "Q.<point>" or "Q.span" [d, d_s]) and model_form (how the returned model holds the result).
    See the section comment above for each method, its checks against the paper and its departures."""
    from .budget import allocation_step, allocation_totals, gap_of, gap_ok, normalise_components

    comps = normalise_components(components)
    blocks = [int(b) for b in blocks]
    if list(budget.blocks) != blocks or tuple(budget.components) != comps:
        raise ValueError(f"budget is for blocks {budget.blocks} and {budget.components}, not {blocks} and {comps}")
    hp = resolve_hparams(method, hparams)
    if method in ("svd", "svdllm", "asvd_scale"):
        m, per, basis, tensors, form = _compress_lowrank(model, method, blocks, comps, budget, stats, hp, keep_tensors)
    elif method == "slicegpt":
        m, per, basis, tensors, form = _compress_slicegpt(model, blocks, comps, budget, stats, hp)
    elif method == "flap":
        m, per, basis, tensors, form = _compress_flap(model, blocks, comps, budget, stats, hp)
    else:
        m, per, basis, tensors, form = _compress_zipit(model, blocks, comps, budget, stats, hp)
    m.eval()
    tot = allocation_totals(per + basis)
    gap = gap_of(tot["total"], budget.total)
    step = allocation_step(method, budget, per + basis, **hp)
    step_rel = step / int(budget.total)
    ok = gap_ok(gap, step_rel)
    if ok and gap <= 0.01:
        note = "within the rule (0 <= gap <= 1%)"
    elif ok:
        note = (f"not reachable at 1%: the method's smallest step ({step:,} weights, {100 * step_rel:.2f}% of the "
                f"budget) is larger; gap within one step")
    elif gap < 0:
        note = "shortfall: the method cannot hold the budget (it keeps every weight it can, e.g. at full size)"
    else:
        note = "above the rule: overshoot larger than one step (unexpected; see per_matrix)"
    info = {"method": method, "hparams": dict(hp), "components": comps, "blocks": blocks,
            "params_matrices": tot["matrices"], "params_basis_change": tot["basis_change"],
            "params_total": tot["total"], "budget_total": int(budget.total), "gap": float(gap), "gap_ok": bool(ok),
            "gap_step": int(step), "gap_reachable": bool(step_rel <= 0.01), "gap_note": note,
            "params_biases": int(budget.biases_dense),
            "budget_matrices": int(budget.matrix_total), "budget_basis_change": int(budget.basis_change),
            "budget_variant": budget.variant, "per_matrix": per, "basis": basis,
            "tensors": tensors if keep_tensors else {}, "model_form": form}
    if verbose:
        from .utils import log

        log(f"{method} {hp}: matrices {tot['matrices']:,} + basis {tot['basis_change']:,} = {tot['total']:,} "
            f"against budget {budget.total:,} (gap {100 * gap:+.3f}%; {note})")
    return m, info


# ----------------------------------------------------------------------------------------------
# Structure check
# ----------------------------------------------------------------------------------------------
def _read_matrix(model: nn.Module, b: int, name: str) -> torch.Tensor:
    blk = model.blocks[int(b)]
    if name == "fc1":
        return blk.mlp.fc1.weight
    if name == "fc2":
        return blk.mlp.fc2.weight
    W = blk.attn.qkv.weight
    D = W.shape[0] // 3
    if name == "wq":
        return W[:D]
    if name == "wk":
        return W[D:2 * D]
    if name == "wv":
        return W[2 * D:]
    return blk.attn.proj.weight


def _read_bias(model: nn.Module, b: int, name: str):
    blk = model.blocks[int(b)]
    if name == "fc2":
        return blk.mlp.fc2.bias
    if name == "wo":
        return blk.attn.proj.bias
    return None


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    den = float(ref.norm())
    return float(x.norm()) / den if den > 0 else float(x.norm())


@torch.no_grad()
def structure_check(model: nn.Module, info: dict, tol: float | None = None) -> dict:
    """Verify the realised structure of a compressed model against its info (from compress).

    lowrank  the factors in info tensors reproduce the written weight (relative error below tol), have rank r, and
             the weight's numerical rank in float64 (singular values above max(m, n) eps(storage dtype) sigma_max)
             equals r for float64 storage and is at most r for lower precision (where rounding hides nothing above
             that tolerance); dense-storage matrices are skipped.
    width    MLP: fc1 and fc2 are nn.Linear of width n holding exactly n neurons; attention: the removed heads' rows of
             wq, wk, wv (and biases) and columns of wo are exactly zero and the kept heads are not all zero.
    slice    reads satisfy W = W Q Q^T and writes W = Q Q^T W (and their biases b = Q Q^T b) to tol; every basis has
             orthonormal columns; SlicedBlock buffers have the counted shapes.
    merge    MLP width equals the number of groups; attention: within each head at most per_head non-zero query and
             value rows, and rows outside the groups' representatives are exactly zero.
    tol defaults to 1e-5 for float32 (or lower) storage and 1e-10 for float64.
    Returns {"ok", "checks": [...], "params_realised", "params_matrices"}."""
    checks = []
    realised = 0
    tensors = info.get("tensors", {}) or {}

    def add(ok, **kw):
        checks.append({"ok": bool(ok), **kw})

    for pm in info["per_matrix"]:
        b, name, kind = int(pm["block"]), pm["name"], pm["kind"]
        Wm = _read_matrix(model, b, name)
        t = tol if tol is not None else (1e-10 if Wm.dtype == torch.float64 else 1e-5)
        W = Wm.detach().to(torch.float64)
        if kind == "lowrank":
            o, i = W.shape
            if pm.get("dense_storage"):
                add(tuple(W.shape) == tuple(pm["shape"]), block=b, name=name, check="dense storage shape")
                realised += o * i
                continue
            r = int(pm["rank"])
            Sv = torch.linalg.svdvals(W)
            thr = max(o, i) * torch.finfo(Wm.dtype).eps * float(Sv.max()) if Sv.numel() else 0.0
            nr = int((Sv > thr).sum())
            ok_rank = nr == r if Wm.dtype == torch.float64 else nr <= r
            add(ok_rank, block=b, name=name, check="numerical rank", rank=r, numerical_rank=nr)
            U, V = tensors.get(f"{b}.{name}.U"), tensors.get(f"{b}.{name}.V")
            if U is not None and V is not None:
                UV = U.to(torch.float64) @ V.to(torch.float64)
                err = _rel(UV.to(W.device) - W, W)
                ok_f = U.shape[1] == r and V.shape[0] == r and err <= max(t, 1e-6 if U.dtype == torch.float32 else t)
                add(ok_f, block=b, name=name, check="factors", rel_err=err)
            realised += r * (o + i)
        elif kind == "width":
            blk = model.blocks[b]
            if name in ("fc1", "fc2"):
                fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
                n = int(pm["n"])
                ok = fc1.out_features == n == fc2.in_features and len(pm.get("kept", range(n))) == n
                add(ok, block=b, name=name, check="width", n=n, fc1_out=fc1.out_features, fc2_in=fc2.in_features)
                realised += n * (fc1.in_features if name == "fc1" else fc2.out_features)
            else:
                from .attention import attn_dims

                H, dh = attn_dims(blk)
                kept = set(int(x) for x in pm["heads_kept"])
                Wh = W.reshape(H, dh, -1) if name != "wo" else W.T.reshape(H, dh, -1)
                zero_ok = all(float(Wh[hh].abs().max()) == 0.0 for hh in range(H) if hh not in kept)
                if name in ("wq", "wk", "wv"):
                    bias = blk.attn.qkv.bias
                    D = H * dh
                    off = {"wq": 0, "wk": D, "wv": 2 * D}[name]
                    bh = bias[off:off + D].reshape(H, dh)
                    zero_ok = zero_ok and all(float(bh[hh].abs().max()) == 0.0 for hh in range(H) if hh not in kept)
                add(zero_ok and len(kept) == int(pm["heads"]), block=b, name=name, check="removed heads are zero",
                    heads=len(kept))
                realised += len(kept) * dh * (W.shape[1] if name != "wo" else W.shape[0])
        elif kind == "slice":
            Q = tensors.get(pm["basis_key"])
            if Q is None:
                add(False, block=b, name=name, check="slice basis missing from info tensors")
                continue
            P = _proj(Q, W)
            if pm["side"] == "in":
                err = _rel(W - W @ P, W)
            else:
                err = _rel(W - P @ W, W)
                bias = _read_bias(model, b, name)
                if bias is not None:
                    bv = bias.detach().to(torch.float64)
                    err = max(err, _rel(bv - P @ bv, bv) if float(bv.norm()) > 0 else 0.0)
            add(err <= t, block=b, name=name, check=f"slice ({pm['side']})", rel_err=err, d_s=int(pm["d_s"]))
            realised += int(pm["d_in"]) * int(pm["d_out"])
        elif kind == "merge":
            blk = model.blocks[b]
            groups = pm["groups"]
            if name in ("fc1", "fc2"):
                n = len(groups)
                ok = blk.mlp.fc1.out_features == n == blk.mlp.fc2.in_features == int(pm["n"])
                add(ok, block=b, name=name, check="merged width", n=n)
                realised += n * (blk.mlp.fc1.in_features if name == "fc1" else blk.mlp.fc2.out_features)
            else:
                reps = {g[0] for g in groups}
                rows = W if name != "wo" else W.T
                nz = (rows.abs().amax(dim=1) > 0).nonzero().flatten().tolist()
                ok = set(nz) <= reps and len(groups) == int(pm["n"])
                from .attention import attn_dims

                H, dh = attn_dims(blk)
                per_head = [sum(1 for g in groups if g[0] // dh == hh) for hh in range(H)]
                ok = ok and max(per_head) <= int(pm["per_head"])
                add(ok, block=b, name=name, check="merged channels", n=len(groups), nonzero_rows=len(nz))
                realised += len(groups) * rows.shape[1]
        else:
            add(False, block=b, name=name, check=f"unknown kind {kind!r}")
    for key_, Q in tensors.items():
        if key_.startswith("Q."):
            Qd = Q.to(torch.float64)
            err = float((Qd.T @ Qd - torch.eye(Qd.shape[1], dtype=torch.float64)).abs().max())
            add(err <= 1e-10, check=f"orthonormal basis {key_}", err=err)
    if info["method"] == "slicegpt" and set(info["components"]) == {"mlp", "attn"}:
        blocks = info["blocks"]
        for i, b in enumerate(blocks):
            blk = model.blocks[b]
            ok = isinstance(blk, SlicedBlock) and ((blk.entry is not None) == (i == 0))
            if info["hparams"]["basis"] == "per_block":
                ok = ok and blk.attn_shortcut is not None and blk.mlp_shortcut is not None
            add(ok, block=b, check="SlicedBlock buffers")
    ok_params = realised == int(info["params_matrices"])
    add(ok_params, check="realised parameters", realised=realised, counted=int(info["params_matrices"]))
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "params_realised": int(realised),
            "params_matrices": int(info["params_matrices"])}

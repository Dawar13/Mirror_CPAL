"""M1: mirror-equivariant attention (folding, statistics, head pairing, head maps, projection).

Conventions (torch, column vectors): attn.qkv.weight is [3 H dh, d] with output rows ordered as
[3, H, dh] (q block, k block, v block; head-major inside each block), attn.proj.weight Wo is
[d, H dh] with columns ordered [H, dh]. A_in is the involution of the stream entering the block
(it acts on q, k, v through the input side) and A_out the involution of the stream the branch
writes into (it acts on Wo, bo). Both are symmetric orthogonal involutions.

Head maps (PROJECT_CONTEXT 5.4). sigma is an involution on heads (twin pairs plus self-heads).
For every head h the maps B[h], C[h] (dh x dh, orthogonal) say how the mirrored image's head
sigma(h) relates to the original head h:
    q~_{sigma(h)} = B[h] q_h,   k~_{sigma(h)} = B[h] k_h,   v~_{sigma(h)} = C[h] v_h.
In weights, with g = sigma(h):
    Wq^g A_in = B[h] Wq^h,  Wk^g A_in = B[h] Wk^h,  bq^g = B[h] bq^h,  bk^g = B[h] bk^h,
    Wv^g A_in = C[h] Wv^h,  bv^g = C[h] bv^h,       A_out Wo^h = Wo^g C[h],  A_out bo = bo.
Then logits of head g on the mirror equal logits of head h on the original (B is orthogonal),
and attn(x A_in) = attn(x) A_out for token rows x. For a twin pair B[g] = B[h]^T and
C[g] = C[h]^T; for a self-head B[h] and C[h] are symmetric involutions.

Gauge. A twin pair's head-internal basis is free: rotating head g by R on q, k (rows) and by S on
v (rows, with Wo^g -> Wo^g S^T on columns) changes neither the logits nor the output. With
R = B[h]^T and S = C[h]^T the pair has B = C = I. `project_attn(..., gauge=True)` writes the
pair in that aligned basis; `gauge=False` projects in the original basis. Both give the same
function (the gauge is orthogonal on weight space and commutes with the projection).

Folding (exact): Wqkv' = Wqkv diag(gamma1), bqkv' = bqkv + Wqkv beta1 (norm1 affine);
Wo' = diag(ls1) Wo, bo' = ls1 * bo (LayerScale). The absorbed block has norm1 affine set to the
identity and ls1 set to ones. LayerNorm without affine commutes with A when A 1 = 1 (fitted
operators satisfy this); an RMSNorm on a mean-zero stream (mirror.fold) commutes with any A.

Statistics are float64 cross-moments of the qkv outputs between the original batch half and the
mirrored, un-flipped half (rows refer to the same image location). Raw activations are never
stored. ViT attention has no positional term inside the block, so token-wise relations hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .fit import fit_involution
from .stats import _flat64, run_paired, unflip_tokens

QKV = ("q", "k", "v")
WEIGHT_KEYS = ("Wq", "Wk", "Wv", "bq", "bk", "bv", "Wo", "bo")


# ----------------------------------------------------------------------------------------------
# Folding
# ----------------------------------------------------------------------------------------------
def _ls1_gamma(block: nn.Module):
    g = getattr(getattr(block, "ls1", None), "gamma", None)
    return g if isinstance(g, torch.Tensor) else None


def _is_identity(mod) -> bool:
    return mod is None or isinstance(mod, nn.Identity)


def check_attn_supported(block: nn.Module) -> None:
    """Raise if the attention has parts this module does not model (qk-norm, gate, scale-norm)."""
    attn = block.attn
    for name in ("q_norm", "k_norm", "norm"):
        if not _is_identity(getattr(attn, name, None)):
            raise ValueError(f"attn.{name} is not supported (must be Identity)")
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported")
    if attn.qkv.out_features % 3 != 0:
        raise ValueError("attn.qkv must output [q; k; v]")


def attn_dims(block: nn.Module) -> tuple[int, int]:
    """(number of heads H, head dimension dh) of one block."""
    attn = block.attn
    H = int(attn.num_heads)
    D = attn.qkv.out_features // 3
    if D % H != 0:
        raise ValueError(f"qkv width {D} not divisible by {H} heads")
    return H, D // H


@torch.no_grad()
def fold_attn(block: nn.Module, dtype: torch.dtype = torch.float64) -> dict:
    """Folded Wq, Wk, Wv [H dh, d], bq, bk, bv, Wo [d, H dh], bo (norm1 and ls1 folded in)."""
    check_attn_supported(block)
    qkv, proj, norm1 = block.attn.qkv, block.attn.proj, block.norm1
    W = qkv.weight.to(dtype)
    D3, d = W.shape
    D = D3 // 3
    b = qkv.bias.to(dtype) if qkv.bias is not None else W.new_zeros(D3)
    gamma = getattr(norm1, "weight", None)
    beta = getattr(norm1, "bias", None)
    gamma = gamma.to(dtype) if isinstance(gamma, torch.Tensor) else W.new_ones(d)
    beta = beta.to(dtype) if isinstance(beta, torch.Tensor) else W.new_zeros(d)
    Wf = W * gamma[None, :]
    bf = b + W @ beta
    Wo = proj.weight.to(dtype)
    bo = proj.bias.to(dtype) if proj.bias is not None else Wo.new_zeros(Wo.shape[0])
    ls = _ls1_gamma(block)
    ls = ls.to(dtype) if ls is not None else Wo.new_ones(Wo.shape[0])
    return {
        "Wq": Wf[:D], "Wk": Wf[D:2 * D], "Wv": Wf[2 * D:],
        "bq": bf[:D], "bk": bf[D:2 * D], "bv": bf[2 * D:],
        "Wo": ls[:, None] * Wo, "bo": ls * bo,
    }


@torch.no_grad()
def write_absorbed_attn(block: nn.Module, Wq, Wk, Wv, bq, bk, bv, Wo, bo) -> None:
    """Write folded attention weights and reset norm1 affine and ls1 to the identity."""
    qkv, proj, norm1 = block.attn.qkv, block.attn.proj, block.norm1
    if qkv.bias is None or proj.bias is None:
        raise ValueError("absorbed form needs qkv and proj biases")
    qkv.weight.copy_(torch.cat([Wq, Wk, Wv], 0).to(qkv.weight))
    qkv.bias.copy_(torch.cat([bq, bk, bv], 0).to(qkv.bias))
    proj.weight.copy_(Wo.to(proj.weight))
    proj.bias.copy_(bo.to(proj.bias))
    if isinstance(getattr(norm1, "weight", None), torch.Tensor):
        norm1.weight.fill_(1.0)
    if isinstance(getattr(norm1, "bias", None), torch.Tensor):
        norm1.bias.zero_()
    g = _ls1_gamma(block)
    if g is not None:
        g.fill_(1.0)


@torch.no_grad()
def absorb_attn(block: nn.Module) -> None:
    write_absorbed_attn(block, **fold_attn(block))


def attn_branch(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The attention half of a pre-norm block: ls1(attn(norm1(x)))."""
    ls1 = getattr(block, "ls1", None)
    y = block.attn(block.norm1(x))
    return y if ls1 is None else ls1(y)


# ----------------------------------------------------------------------------------------------
# Streaming statistics
# ----------------------------------------------------------------------------------------------
class AttnStats:
    """Per block, for t in (q, k, v): cross[b, t] = sum yt^T y ([H dh, H dh], mirror rows against
    original columns), plus per-coordinate sums and squared sums of both halves and the token count.

    The H x H blocks of dh x dh are cross[b, t, g*dh:(g+1)*dh, h*dh:(h+1)*dh] = sum yt_g y_h^T.
    """

    def __init__(self, n_blocks: int, num_heads: int, head_dim: int, device="cpu"):
        self.H, self.dh = int(num_heads), int(head_dim)
        D = self.H * self.dh
        kw = dict(dtype=torch.float64, device=device)
        self.cross = torch.zeros(n_blocks, 3, D, D, **kw)
        self.s = torch.zeros(n_blocks, 3, D, **kw)
        self.s_t = torch.zeros(n_blocks, 3, D, **kw)
        self.sq = torch.zeros(n_blocks, 3, D, **kw)
        self.sq_t = torch.zeros(n_blocks, 3, D, **kw)
        self.n = torch.zeros(n_blocks, **kw)

    @classmethod
    def for_model(cls, model: nn.Module, device=None) -> "AttnStats":
        H, dh = attn_dims(model.blocks[0])
        dev = next(model.parameters()).device if device is None else device
        return cls(len(model.blocks), H, dh, dev)

    def update(self, block: int, y: torch.Tensor, yt: torch.Tensor) -> None:
        """y, yt: qkv outputs [..., 3 H dh] of the original and the un-flipped mirror."""
        D = self.H * self.dh
        Y = _flat64(y).to(self.cross.device).reshape(-1, 3, D)
        Yt = _flat64(yt).to(self.cross.device).reshape(-1, 3, D)
        if Y.shape != Yt.shape:
            raise ValueError("original and mirror halves differ in shape")
        self.n[block] += Y.shape[0]
        self.s[block] += Y.sum(0)
        self.s_t[block] += Yt.sum(0)
        self.sq[block] += (Y * Y).sum(0)
        self.sq_t[block] += (Yt * Yt).sum(0)
        for t in range(3):
            self.cross[block, t] += Yt[:, t].T @ Y[:, t]

    def moments(self, block: int, center: bool = False) -> dict:
        """Per-head moments of one block (numpy, float64).

        M[t][g, h] = sum yt_g y_h^T  ([H, H, dh, dh]), E[t][h] = sum ||y_h||^2,
        Et[t][g] = sum ||yt_g||^2. With `center` the token means are removed first.
        """
        H, dh = self.H, self.dh
        n = float(self.n[block])
        out = {"M": {}, "E": {}, "Et": {}, "n": n, "H": H, "dh": dh}
        for i, t in enumerate(QKV):
            C = self.cross[block, i].clone()
            sq, sq_t = self.sq[block, i].clone(), self.sq_t[block, i].clone()
            if center and n > 0:
                s, s_t = self.s[block, i], self.s_t[block, i]
                C -= torch.outer(s_t, s) / n
                sq -= s * s / n
                sq_t -= s_t * s_t / n
            out["M"][t] = C.reshape(H, dh, H, dh).permute(0, 2, 1, 3).cpu().numpy()
            out["E"][t] = sq.clamp_min(0).reshape(H, dh).sum(1).cpu().numpy()
            out["Et"][t] = sq_t.clamp_min(0).reshape(H, dh).sum(1).cpu().numpy()
        return out

    def numpy(self) -> dict:
        return {"cross": self.cross.cpu().numpy(), "s": self.s.cpu().numpy(),
                "s_t": self.s_t.cpu().numpy(), "sq": self.sq.cpu().numpy(),
                "sq_t": self.sq_t.cpu().numpy(), "n": self.n.cpu().numpy(),
                "H": self.H, "dh": self.dh}


class AttnPairedHooks:
    """Forward hooks on attn.qkv handing (original, un-flipped mirror) pairs to qkv_cb(block, y, yt).

    The model must be run on [x; mirror(x)] as one batch (as stats.run_paired does). Can be used
    alongside stats.PairedHooks, so residual, fc1 and qkv statistics come from one pass.
    """

    def __init__(self, adapter, model: nn.Module, qkv_cb, blocks=None):
        self.handles = []
        grid, npre = adapter.grid_hw, adapter.n_prefix_tokens
        sel = range(len(model.blocks)) if blocks is None else blocks
        for i in sel:
            def hook(_mod, _args, out, i=i):
                b = out.shape[0] // 2
                qkv_cb(i, out[:b], unflip_tokens(out[b:], grid, npre))

            self.handles.append(model.blocks[i].attn.qkv.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def collect_attn_stats(adapter, loader, device, model: nn.Module | None = None, blocks=None,
                       stats_device=None, residual_cb=None, fc1_cb=None) -> AttnStats:
    """One paired pass over `loader` collecting AttnStats (and optionally residual/fc1 callbacks)."""
    m = adapter.model if model is None else model
    st = AttnStats.for_model(m, device if stats_device is None else stats_device)
    with AttnPairedHooks(adapter, m, st.update, blocks):
        run_paired(adapter, loader, device, residual_cb=residual_cb, fc1_cb=fc1_cb, model=m,
                   desc="attn stats")
    return st


# ----------------------------------------------------------------------------------------------
# Head maps, scores and pairing
# ----------------------------------------------------------------------------------------------
def polar(M: np.ndarray) -> np.ndarray:
    """Orthogonal factor U V^T of M = U S V^T (the maximiser of tr(B^T M) over orthogonal B)."""
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def _nuclear(M: np.ndarray) -> float:
    return float(np.linalg.svd(M, compute_uv=False).sum())


def _self_fit_value(M: np.ndarray) -> float:
    """max tr(B M) over symmetric orthogonal involutions B = sum |eig((M + M^T)/2)|."""
    return float(np.abs(np.linalg.eigvalsh(0.5 * (M + M.T))).sum())


@dataclass
class HeadMaps:
    """Head involution sigma and per-head maps B (q, k) and C (v); see the module docstring."""

    sigma: np.ndarray                       # sigma[h] = partner of head h (sigma[h] = h: self-head)
    B: np.ndarray                           # [H, dh, dh]
    C: np.ndarray                           # [H, dh, dh]
    scores: np.ndarray | None = None        # [H, H] pairing scores used to choose sigma
    info: dict = field(default_factory=dict)

    @property
    def H(self) -> int:
        return int(len(self.sigma))

    @property
    def dh(self) -> int:
        return int(self.B.shape[-1])

    @property
    def pairs(self) -> list:
        return [(h, int(g)) for h, g in enumerate(self.sigma) if g > h]

    @property
    def self_heads(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.sigma) == np.arange(self.H))

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_self(self) -> int:
        return int(len(self.self_heads))

    @classmethod
    def identity(cls, H: int, dh: int) -> "HeadMaps":
        eye = np.broadcast_to(np.eye(dh), (H, dh, dh)).copy()
        return cls(sigma=np.arange(H), B=eye, C=eye.copy())

    def check(self, tol: float = 1e-8) -> None:
        """Raise unless sigma is an involution, maps are orthogonal, B[g] = B[h]^T on pairs and
        self-head maps are symmetric involutions."""
        sig = np.asarray(self.sigma)
        H, dh = self.H, self.dh
        if not np.array_equal(sig[sig], np.arange(H)):
            raise ValueError("sigma is not an involution")
        eye = np.eye(dh)
        for X, name in ((self.B, "B"), (self.C, "C")):
            for h in range(H):
                if np.abs(X[h] @ X[h].T - eye).max() > tol:
                    raise ValueError(f"{name}[{h}] is not orthogonal")
                g = sig[h]
                if np.abs(X[g] - X[h].T).max() > tol:
                    raise ValueError(f"{name}[{g}] != {name}[{h}]^T (self-heads need symmetric maps)")

    def gauged(self) -> "HeadMaps":
        """Maps after the twin gauge (B = C = I on every pair; self-heads unchanged)."""
        B, C = self.B.copy(), self.C.copy()
        eye = np.eye(self.dh)
        for h, g in self.pairs:
            B[h] = B[g] = eye
            C[h] = C[g] = eye
        return HeadMaps(sigma=np.asarray(self.sigma).copy(), B=B, C=C, scores=self.scores,
                        info=dict(self.info))


def _qk_moments(mom: dict):
    M = mom["M"]["q"] + mom["M"]["k"]
    E = mom["E"]["q"] + mom["E"]["k"]
    Et = mom["Et"]["q"] + mom["Et"]["k"]
    return M, E, Et


def head_pair_scores(mom: dict, normalise: bool = True) -> np.ndarray:
    """Symmetric H x H score matrix from [q; k] moments of one block.

    Diagonal: best self-head fit max_B tr(B M[h, h]) over involutions B.
    Off-diagonal: joint orbit fit of the pair (h, g), max over orthogonal B of
    tr(B^T M[g, h]) + tr(B M[h, g]) = ||M[g, h] + M[h, g]^T||_* (both directions at once, so the
    matrix is symmetric by construction). With `normalise` each value is divided by
    sqrt(Et E) summed over the orbit, a cosine in [0, 1] (1: exact map up to an orthogonal B);
    without it the raw values are the exact least-squares objective (larger heads weigh more).
    """
    M, E, Et = _qk_moments(mom)
    H = M.shape[0]
    S = np.zeros((H, H))
    for h in range(H):
        v = _self_fit_value(M[h, h])
        den = np.sqrt(max(Et[h] * E[h], 1e-300)) if normalise else 1.0
        S[h, h] = v / den
        for g in range(h + 1, H):
            v = _nuclear(M[g, h] + M[h, g].T)
            den = np.sqrt(max((Et[g] + Et[h]) * (E[h] + E[g]), 1e-300)) if normalise else 2.0
            S[h, g] = S[g, h] = v / den
    return S


def directional_head_scores(mom: dict) -> np.ndarray:
    """R[g, h] = ||M[g, h]||_* / sqrt(Et_g E_h): mirror head g against original head h (diagnostic)."""
    M, E, Et = _qk_moments(mom)
    H = M.shape[0]
    R = np.zeros((H, H))
    for g in range(H):
        for h in range(H):
            R[g, h] = _nuclear(M[g, h]) / np.sqrt(max(Et[g] * E[h], 1e-300))
    return R


def _match_dp(W: np.ndarray) -> list:
    """Exact max-weight matching by dynamic programming over subsets (H <= 16)."""
    H = W.shape[0]
    full = (1 << H) - 1
    best = np.full(1 << H, -np.inf)
    choice = np.full(1 << H, -1, dtype=np.int64)
    best[full] = 0.0
    for mask in range(full - 1, -1, -1):
        i = 0
        while mask >> i & 1:
            i += 1
        m1 = mask | (1 << i)
        val, ch = best[m1], i                      # i stays a self-head
        for j in range(i + 1, H):
            if not mask >> j & 1:
                v = W[i, j] + best[m1 | (1 << j)]
                if v > val:
                    val, ch = v, j
        best[mask], choice[mask] = val, ch
    pairs, mask = [], 0
    while mask != full:
        i = 0
        while mask >> i & 1:
            i += 1
        j = int(choice[mask])
        mask |= (1 << i) | (1 << j)
        if j != i:
            pairs.append((i, j))
    return pairs


def match_heads(S: np.ndarray, method: str = "auto", tol: float = 1e-12) -> np.ndarray:
    """Involution sigma maximising sum_h S[h, sigma(h)] (a pair counts twice, a self-head once).

    Equivalent to a maximum-weight matching with gains w = 2 S[h, g] - S[h, h] - S[g, g].
    method: "dp" (exact subset DP, H <= 16), "networkx" (exact blossom) or "auto".
    """
    S = 0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
    H = S.shape[0]
    diag = np.diag(S)
    W = 2.0 * S - diag[:, None] - diag[None, :]
    if method == "auto":
        method = "dp" if H <= 16 else "networkx"
    if method == "dp":
        if H > 16:
            raise ValueError("dp matching is for H <= 16")
        pairs = _match_dp(W)
    elif method == "networkx":
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(H))
        for i in range(H):
            for j in range(i + 1, H):
                if W[i, j] > tol:
                    G.add_edge(i, j, weight=float(W[i, j]))
        pairs = list(nx.max_weight_matching(G, maxcardinality=False, weight="weight"))
    else:
        raise ValueError(method)
    sigma = np.arange(H)
    for a, b in pairs:
        if W[a, b] > tol:
            sigma[a], sigma[b] = b, a
    return sigma


def fit_head_maps(mom: dict, sigma) -> HeadMaps:
    """Fit B (on [q; k]) and C (on v) for a given head involution sigma.

    Twin pair (h, g), h < g: B[h] = polar(M[g, h] + M[h, g]^T) (joint fit of q~_g = B q_h and
    q~_h = B^T q_g), B[g] = B[h]^T; C likewise on v. Self-head h: involutions from
    fit.fit_involution (sign-of-eigenvalue rule) on M[h, h].
    """
    sigma = np.asarray(sigma, dtype=np.int64)
    Mqk, _, _ = _qk_moments(mom)
    Mv = mom["M"]["v"]
    H, dh = Mqk.shape[0], Mqk.shape[-1]
    B = np.zeros((H, dh, dh))
    C = np.zeros((H, dh, dh))
    for h in range(H):
        g = int(sigma[h])
        if g == h:
            B[h] = fit_involution(Mqk[h, h])[0]
            C[h] = fit_involution(Mv[h, h])[0]
        elif g > h:
            B[h] = polar(Mqk[g, h] + Mqk[h, g].T)
            C[h] = polar(Mv[g, h] + Mv[h, g].T)
            B[g], C[g] = B[h].T, C[h].T
    return HeadMaps(sigma=sigma, B=B, C=C)


def pair_and_fit_heads(mom: dict, method: str = "auto", normalise: bool = True) -> HeadMaps:
    """Score head pairs on [q; k], choose sigma, fit B and C. Scores are kept in `.scores`."""
    S = head_pair_scores(mom, normalise=normalise)
    hm = fit_head_maps(mom, match_heads(S, method=method))
    hm.scores = S
    return hm


def head_map_errors(mom: dict, hm: HeadMaps) -> dict:
    """Relative fit errors from moments, per head h (indexed by the original head):
    ||y~_{sigma h} - B[h] y_h||^2 / ||y~_{sigma h}||^2 for qk (B) and v (C), and the same for the
    identity map (sigma = id, B = C = I) as a baseline. Exact for uncentred moments."""
    Mqk, Eqk, Etqk = _qk_moments(mom)
    Mv, Ev, Etv = mom["M"]["v"], mom["E"]["v"], mom["Et"]["v"]
    H = hm.H
    out = {k: np.zeros(H) for k in ("qk", "v", "qk_identity", "v_identity")}
    for h in range(H):
        g = int(hm.sigma[h])
        out["qk"][h] = (Etqk[g] + Eqk[h] - 2 * np.trace(hm.B[h].T @ Mqk[g, h])) / max(Etqk[g], 1e-300)
        out["v"][h] = (Etv[g] + Ev[h] - 2 * np.trace(hm.C[h].T @ Mv[g, h])) / max(Etv[g], 1e-300)
        out["qk_identity"][h] = (Etqk[h] + Eqk[h] - 2 * np.trace(Mqk[h, h])) / max(Etqk[h], 1e-300)
        out["v_identity"][h] = (Etv[h] + Ev[h] - 2 * np.trace(Mv[h, h])) / max(Etv[h], 1e-300)
    return out


# ----------------------------------------------------------------------------------------------
# Projection onto the equivariant form
# ----------------------------------------------------------------------------------------------
def _t(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x,
                           dtype=like.dtype, device=like.device)


def _split(f: dict, H: int, dh: int) -> dict:
    d = f["Wq"].shape[1]
    out = {k: f[k].clone().reshape(H, dh, d) for k in ("Wq", "Wk", "Wv")}
    out.update({k: f[k].clone().reshape(H, dh) for k in ("bq", "bk", "bv")})
    out["Wo"] = f["Wo"].clone().reshape(f["Wo"].shape[0], H, dh)
    out["bo"] = f["bo"].clone()
    return out


def _merge(s: dict) -> dict:
    H, dh, d = s["Wq"].shape
    out = {k: s[k].reshape(H * dh, d) for k in ("Wq", "Wk", "Wv")}
    out.update({k: s[k].reshape(H * dh) for k in ("bq", "bk", "bv")})
    out["Wo"] = s["Wo"].reshape(s["Wo"].shape[0], H * dh)
    out["bo"] = s["bo"]
    return out


def gauge_twin_heads(f: dict, hm: HeadMaps) -> tuple[dict, HeadMaps]:
    """Rotate the second head g of every twin pair (h, g) so that B = C = I.

    Wq^g, Wk^g, bq^g, bk^g <- B[h]^T (.), Wv^g, bv^g <- C[h]^T (.), Wo^g <- Wo^g C[h].
    The attention function is unchanged. Returns (weights, hm.gauged()).
    """
    s = _split(f, hm.H, hm.dh)
    like = s["Wq"]
    for h, g in hm.pairs:
        R = _t(hm.B[h], like).T
        Sv = _t(hm.C[h], like).T
        for k in ("Wq", "Wk"):
            s[k][g] = R @ s[k][g]
        for k in ("bq", "bk"):
            s[k][g] = R @ s[k][g]
        s["Wv"][g] = Sv @ s["Wv"][g]
        s["bv"][g] = Sv @ s["bv"][g]
        s["Wo"][:, g] = s["Wo"][:, g] @ Sv.T
    return _merge(s), hm.gauged()


def project_attn(Wq, Wk, Wv, bq, bk, bv, Wo, bo, A_in, A_out, head_maps: HeadMaps,
                 gauge: bool = True) -> dict:
    """Nearest (Frobenius, per weight) exactly equivariant attention weights, by orbit averaging.

    Self-head h:  W <- (W + B W A_in)/2 for q, k (and C for v), b <- (b + B b)/2,
                  Wo^h <- (Wo^h + A_out Wo^h C)/2.
    Twin (h, g):  W^h <- (W^h + B[h]^T W^g A_in)/2, W^g <- B[h] W^h A_in (q, k; C for v),
                  b^h <- (b^h + B[h]^T b^g)/2, b^g <- B[h] b^h,
                  Wo^h <- (Wo^h + A_out Wo^g C[h])/2, Wo^g <- A_out Wo^h C[h]^T.
    Output bias:  bo <- (bo + A_out bo)/2.
    With `gauge` the pairs are first rotated so that B = C = I (see gauge_twin_heads); the
    maps valid for the result are then `head_maps.gauged()`.
    """
    hm = head_maps
    f = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "bq": bq, "bk": bk, "bv": bv, "Wo": Wo, "bo": bo}
    like = Wq
    f = {k: _t(v, like) for k, v in f.items()}
    if gauge:
        f, hm = gauge_twin_heads(f, hm)
    Ai, Ao = _t(A_in, like), _t(A_out, like)
    s = _split(f, hm.H, hm.dh)
    o = {k: v.clone() for k, v in s.items()}
    for h in range(hm.H):
        g = int(hm.sigma[h])
        B, C = _t(hm.B[h], like), _t(hm.C[h], like)
        if g == h:
            for k in ("Wq", "Wk"):
                o[k][h] = 0.5 * (s[k][h] + B @ s[k][h] @ Ai)
            for k in ("bq", "bk"):
                o[k][h] = 0.5 * (s[k][h] + B @ s[k][h])
            o["Wv"][h] = 0.5 * (s["Wv"][h] + C @ s["Wv"][h] @ Ai)
            o["bv"][h] = 0.5 * (s["bv"][h] + C @ s["bv"][h])
            o["Wo"][:, h] = 0.5 * (s["Wo"][:, h] + Ao @ s["Wo"][:, h] @ C)
        elif g > h:
            for k, M in (("Wq", B), ("Wk", B), ("Wv", C)):
                Wh = 0.5 * (s[k][h] + M.T @ s[k][g] @ Ai)
                o[k][h], o[k][g] = Wh, M @ Wh @ Ai
            for k, M in (("bq", B), ("bk", B), ("bv", C)):
                bh = 0.5 * (s[k][h] + M.T @ s[k][g])
                o[k][h], o[k][g] = bh, M @ bh
            Woh = 0.5 * (s["Wo"][:, h] + Ao @ s["Wo"][:, g] @ C)
            o["Wo"][:, h], o["Wo"][:, g] = Woh, Ao @ Woh @ C.T
    o["bo"] = 0.5 * (s["bo"] + Ao @ s["bo"])
    return _merge(o)


def _share(W: torch.Tensor, W_eq: torch.Tensor) -> float:
    den = float((W ** 2).sum())
    return float(((W - W_eq) ** 2).sum()) / den if den > 0 else float("nan")


def breaking_shares_from(f: dict, e: dict) -> dict:
    """||W - W_eq||^2 / ||W||^2 for each weight, plus "s_all" over Wq, Wk, Wv, Wo together.

    `f` and `e` must be in the same gauge (the gauge is orthogonal, so shares do not depend on it).
    """
    out = {f"s_{k[1:] if k.startswith('W') else k}": _share(f[k], e[k]) for k in WEIGHT_KEYS}
    num = sum(float(((f[k] - e[k]) ** 2).sum()) for k in ("Wq", "Wk", "Wv", "Wo"))
    den = sum(float((f[k] ** 2).sum()) for k in ("Wq", "Wk", "Wv", "Wo"))
    out["s_all"] = num / den if den > 0 else float("nan")
    return out


def attention_breaking_shares(block: nn.Module, A_in, A_out, head_maps: HeadMaps) -> dict:
    """Breaking shares of one block's folded attention: keys s_q, s_k, s_v, s_o (matrices),
    s_bq, s_bk, s_bv, s_bo (biases) and s_all (the four matrices together)."""
    f, hm = gauge_twin_heads(fold_attn(block), head_maps)
    e = project_attn(**f, A_in=A_in, A_out=A_out, head_maps=hm, gauge=False)
    return breaking_shares_from(f, e)


def _tr(M) -> int:
    return int(round(float(np.trace(np.asarray(M)))))


def attention_free_params(d: int, head_maps: HeadMaps, A_in, A_out) -> dict:
    """Dimension of the equivariant subspace for each attention weight, and totals.

    A self-head contributes (dh d + tr(B) tr(A_in))/2 to Wq and Wk (C for Wv),
    (d dh + tr(A_out) tr(C))/2 to Wo, (dh + tr B)/2 to bq and bk ((dh + tr C)/2 to bv); a twin
    pair contributes dh d per matrix and dh per bias (half of its original count).
    bo has (d + tr A_out)/2. "original" counts the unconstrained parameters.
    """
    hm = head_maps
    dh, H = hm.dh, hm.H
    ti, to = _tr(A_in), _tr(A_out)
    n = {k: 0 for k in WEIGHT_KEYS}
    for h in hm.self_heads:
        tB, tC = _tr(hm.B[h]), _tr(hm.C[h])
        n["Wq"] += (dh * d + tB * ti) // 2
        n["Wk"] += (dh * d + tB * ti) // 2
        n["Wv"] += (dh * d + tC * ti) // 2
        n["Wo"] += (d * dh + to * tC) // 2
        n["bq"] += (dh + tB) // 2
        n["bk"] += (dh + tB) // 2
        n["bv"] += (dh + tC) // 2
    for _ in hm.pairs:
        for k in ("Wq", "Wk", "Wv", "Wo"):
            n[k] += dh * d
        for k in ("bq", "bk", "bv"):
            n[k] += dh
    n["bo"] = (d + to) // 2
    n["total"] = sum(n[k] for k in WEIGHT_KEYS)
    n["original"] = 4 * H * dh * d + 3 * H * dh + d
    n["matrices"] = n["Wq"] + n["Wk"] + n["Wv"] + n["Wo"]
    n["matrices_original"] = 4 * H * dh * d
    return n


# ----------------------------------------------------------------------------------------------
# Surgery
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def apply_attention_surgery(model: nn.Module, blocks: list[int], A_in_list: list, A_out_list: list,
                            head_maps_list: list, gauge: bool = True) -> list[HeadMaps]:
    """Replace the attention of `blocks` by its projected, absorbed equivariant version (in place).

    Returns the head maps that hold for the new weights (gauged when `gauge`).
    """
    if not (len(blocks) == len(A_in_list) == len(A_out_list) == len(head_maps_list)):
        raise ValueError("one A_in, A_out and HeadMaps per block")
    out = []
    for b, A_in, A_out, hm in zip(blocks, A_in_list, A_out_list, head_maps_list):
        blk = model.blocks[b]
        e = project_attn(**fold_attn(blk), A_in=A_in, A_out=A_out, head_maps=hm, gauge=gauge)
        write_absorbed_attn(blk, **e)
        out.append(hm.gauged() if gauge else hm)
    return out

"""M1 conversion helpers: V-exact (one operator over a span), registers, even head, matched random
structure (operators, neuron pairings, head maps), the attention SVD baseline, exactness probes.

Conventions as in mirror.attention and mirror.surgery: tokens are rows and the residual involution
A acts as x -> x A (A is symmetric, so it also acts on column vectors). Blocks are 0-based in code.
Layer l is the stream entering block l (layer 0: the embeddings), layer l + 1 its output.

V-exact (PROJECT_CONTEXT 5.5). One operator A* for the whole converted span. Exact equivariance
over the span means: for the mirrored image, the tokens at mirrored positions (patch grid flipped
left-right, register j replaced by register pi(j), CLS unchanged) equal the original tokens times
A*, at every layer from the span's input to its last output. `build_vexact` constructs it:
  1. DINOv2 only: the interpolated position embedding of the evaluation grid is written into the
     model (`materialise_pos_embed`). DINOv2's bicubic interpolation with interpolate_offset = 0.1
     samples at (i + 0.5) M / (w0 + 0.1) - 0.5, which is not symmetric under a left-right flip, so
     the projection must act on the grid actually used. The function at that resolution is
     unchanged (the hub code returns pos_embed directly when its size matches the grid);
  2. full LayerNorm folding (mirror.fold.fold_layernorm_full, final affine kept in the model) so
     every norm is an RMSNorm that commutes with any orthogonal A*;
  3. if the span starts at block 0, the embeddings are projected (`project_embeddings`):
       patch embedding  W <- (W + A* flip(W)) / 2,  b <- (b + A* b) / 2  (flip: kernel width axis),
       positions        pos(r, c) <- (pos(r, c) + pos(r, W-1-c) A*) / 2,
       CLS              token and its position <- (c + c A*) / 2 (even),
       registers        reg_j <- (reg_j + reg_pi(j) A*) / 2 (pairs swap, self-registers even);
     each is the orthogonal projection onto the equivariant subspace (the map X -> X~ A* is an
     orthogonal involution on that parameter);
  4. MLP and attention of the span projected with A_in = A_out = A* (surgery.apply_surgery_two,
     attention.apply_attention_surgery);
  5. optionally (full conversion) the final normalised features are projected onto the even
     subspace, y <- y (I + A*) / 2, between the RMSNorm and the final affine (`EvenProject`), so the
     features of the mirrored image equal the original's exactly (flip consistency 100%).
DINOv2's mask_token is not used at evaluation and is left as folded.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .attention import HeadMaps, apply_attention_surgery, fold_attn, match_heads, write_absorbed_attn
from .data import mirror
from .fit import fit_involution, random_involution, random_orthogonal
from .fold import fold_layernorm_full
from .heal import exact_involution
from .matching import random_pairing
from .stats import center_channels, unflip_tokens
from .surgery import apply_surgery_two, equivariant_param_count, perm_trace, svd_truncate

__all__ = [
    "centre_projector", "fit_span_operator", "span_operator_report", "EvenProject", "materialise_pos_embed",
    "project_embeddings", "install_even_head", "build_vexact", "register_info", "RegisterScores",
    "match_registers", "align_mirror_tokens", "stream_equivariance", "feature_invariance",
    "random_involution_fixing_ones", "random_head_structure", "random_head_maps_like",
    "matched_random_structure", "attention_svd_ranks", "apply_attention_svd", "mlp_mirror_params",
    "involution_trace64",
]


def involution_trace64(A) -> int:
    return int(round(float(np.trace(np.asarray(A, dtype=np.float64)))))


# ----------------------------------------------------------------------------------------------
# Span operator (V-exact)
# ----------------------------------------------------------------------------------------------
def centre_projector(d: int) -> np.ndarray:
    return np.eye(d) - np.full((d, d), 1.0 / d)


def _centred(C: np.ndarray, Pc: np.ndarray | None) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    return C if Pc is None else Pc @ C @ Pc


def fit_span_operator(C, layers, G=None, centre: bool = True):
    """One involution for the stream layers `layers`: fit.fit_involution on sum_l C_l / e_l.

    e_l is the energy of the original tokens at layer l, tr(G_l) (G = sum X^T X), so every layer
    weighs the same. Without G, e_l = sum |eig((C_l + C_l^T)/2)| (the best per-layer fit value).
    With `centre` each C_l is first projected onto the channel-centred subspace in float64,
    C_l <- Pc C_l Pc, Pc = I - 11^T/d: the stored float32 cross-moments have the all-ones direction
    in their null space only to float32 precision, and this makes A* 1 = 1 exactly (the all-ones
    direction then has eigenvalue 0 and fit_involution assigns it +1), as for the per-layer fits.
    Returns (A [d, d] float64, info).
    """
    C = np.asarray(C)
    d = C.shape[-1]
    Pc = centre_projector(d) if centre else None
    total = np.zeros((d, d))
    energies = []
    for l in layers:
        Cl = _centred(C[l], Pc)
        if G is not None:
            e = float(np.trace(np.asarray(G[l], dtype=np.float64)))
        else:
            e = float(np.abs(np.linalg.eigvalsh(0.5 * (Cl + Cl.T))).sum())
        energies.append(e)
        total += Cl / max(e, 1e-300)
    A, dp, dm, info = fit_involution(total, return_info=True)
    A = exact_involution(A)
    return A, {"layers": [int(l) for l in layers], "d_plus": int(dp), "d_minus": int(dm),
               "n_zero": int(info["n_zero"]), "energies": energies,
               "normalisation": "trace(G_l)" if G is not None else "sum |eig(sym C_l)|",
               "centred": bool(centre)}


def span_operator_report(A_star, A_layers, C, layers, centre: bool = True) -> list:
    """Per layer: agreement tr(A* A_l)/d and alignment tr(A* C_l) / tr(A_l C_l) (fit data, in-sample;
    1 means A* captures as much mirror cross-moment at layer l as the layer's own operator)."""
    d = A_star.shape[0]
    Pc = centre_projector(d) if centre else None
    rows = []
    for l in layers:
        Cl = _centred(C[l], Pc)
        Al = np.asarray(A_layers[l], dtype=np.float64)
        own = float(np.trace(Al @ Cl))
        rows.append({"layer": int(l), "agreement": float(np.trace(A_star @ Al) / d),
                     "alignment": float(np.trace(A_star @ Cl)) / own if own != 0 else float("nan")})
    return rows


# ----------------------------------------------------------------------------------------------
# V-exact model
# ----------------------------------------------------------------------------------------------
class EvenProject(nn.Module):
    """y -> y (I + A)/2: projection onto the even (+1) subspace of A. Runs outside autocast, in the
    buffer's dtype, so the invariance is limited only by that dtype."""

    def __init__(self, A, dtype=torch.float32, device=None):
        super().__init__()
        A = np.asarray(A, dtype=np.float64)
        P = 0.5 * (np.eye(A.shape[0]) + A)
        self.register_buffer("P", torch.as_tensor(0.5 * (P + P.T), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (x.to(self.P.dtype) @ self.P).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.P.shape[0]}, rank={int(round(float(torch.trace(self.P.double()))))}"


def _has_dinov2_interp(model: nn.Module) -> bool:
    return hasattr(model, "interpolate_pos_encoding") and hasattr(model, "prepare_tokens_with_masks")


@torch.no_grad()
def materialise_pos_embed(model: nn.Module, adapter) -> bool:
    """DINOv2: write the position embedding interpolated to the adapter's grid into `model.pos_embed`
    (in place). The function at this resolution is unchanged. Returns True if it changed the shape."""
    if not _has_dinov2_interp(model):
        if getattr(model, "dynamic_img_size", False):
            raise NotImplementedError("timm dynamic_img_size resamples positions on the fly; not supported")
        return False
    H, W = adapter.grid_hw
    if H != W:
        raise NotImplementedError("non-square grids are not supported")
    pe = model.pos_embed
    if pe.shape[1] == 1 + H * W:
        return False
    dummy = torch.zeros(1, 1 + H * W, pe.shape[-1], dtype=pe.dtype, device=pe.device)
    new = model.interpolate_pos_encoding(dummy, adapter.img_size, adapter.img_size)
    model.pos_embed = nn.Parameter(new.detach().clone().to(pe.dtype), requires_grad=pe.requires_grad)
    return True


def register_info(model: nn.Module, adapter) -> dict:
    """Where the register tokens live: the parameter name, count R and first token index (after CLS)."""
    for name in ("register_tokens", "reg_token"):
        p = getattr(model, name, None)
        if isinstance(p, torch.Tensor):
            has_cls = getattr(model, "cls_token", None) is not None
            return {"name": name, "R": int(p.shape[1]), "start": 1 if has_cls else 0}
    return {"name": None, "R": 0, "start": 1}


def _share(old: torch.Tensor, new: torch.Tensor) -> float:
    den = float((old.double() ** 2).sum())
    return float(((old.double() - new.double()) ** 2).sum()) / den if den > 0 else float("nan")


def _check_perm(perm, R: int) -> np.ndarray:
    perm = np.arange(R) if perm is None else np.asarray(perm, dtype=np.int64)
    if perm.shape != (R,) or not np.array_equal(perm[perm], np.arange(R)):
        raise ValueError(f"register permutation must be an involution on {R} registers")
    return perm


@torch.no_grad()
def project_embeddings(model: nn.Module, adapter, A, reg_perm=None, write: bool = True) -> dict:
    """Project the patch embedding, position embedding, CLS token and registers onto the exactly
    equivariant form for A (see the module docstring). Returns the breaking share of each part
    (||X - X_eq||^2 / ||X||^2). With write=False nothing is changed."""
    shares = {}
    A64 = np.asarray(A, dtype=np.float64)

    def At(like):
        return torch.as_tensor(A64, dtype=torch.float64, device=like.device)

    def put(p: torch.Tensor, value: torch.Tensor):
        if write:
            p.copy_(value.to(dtype=p.dtype, device=p.device))

    pe = model.patch_embed.proj
    if not isinstance(pe, nn.Conv2d):
        raise NotImplementedError("patch_embed.proj must be Conv2d")
    W = pe.weight.double()
    Wn = 0.5 * (W + torch.einsum("oi,ichw->ochw", At(W), W.flip(-1)))
    shares["patch_embed.weight"] = _share(W, Wn)
    put(pe.weight, Wn)
    if pe.bias is not None:
        b = pe.bias.double()
        bn = 0.5 * (b + b @ At(b))
        shares["patch_embed.bias"] = _share(b, bn)
        put(pe.bias, bn)

    H, Wg = adapter.grid_hw
    reg = register_info(model, adapter)
    R = reg["R"]
    perm = _check_perm(reg_perm, R)
    has_cls = getattr(model, "cls_token", None) is not None
    pos = getattr(model, "pos_embed", None)
    if isinstance(pos, torch.Tensor):
        P = pos.double()[0].clone()
        n_pre = P.shape[0] - H * Wg
        allowed = {0, int(has_cls)} | ({int(has_cls) + R} if (has_cls and R) else set())
        if n_pre not in allowed:
            raise NotImplementedError(f"position embedding has {n_pre} prefix rows; expected one of {sorted(allowed)}")
        A_ = At(P)
        grid = P[n_pre:].reshape(H, Wg, -1)
        new_grid = 0.5 * (grid + grid.flip(1) @ A_)
        shares["pos_embed.patches"] = _share(grid, new_grid)
        Pn = P.clone()
        Pn[n_pre:] = new_grid.reshape(H * Wg, -1)
        if n_pre >= 1 and has_cls:
            Pn[0] = 0.5 * (P[0] + P[0] @ A_)
            shares["pos_embed.cls"] = _share(P[0], Pn[0])
        if n_pre > 1:
            Rp = P[1:1 + R]
            Pn[1:1 + R] = 0.5 * (Rp + Rp[torch.as_tensor(perm, device=Rp.device)] @ A_)
            shares["pos_embed.registers"] = _share(Rp, Pn[1:1 + R])
        put(pos, Pn[None])
    if has_cls:
        c = model.cls_token.double()
        cn = 0.5 * (c + c @ At(c))
        shares["cls_token"] = _share(c, cn)
        put(model.cls_token, cn)
    if R:
        p = getattr(model, reg["name"])
        r = p.double()[0]
        rn = 0.5 * (r + r[torch.as_tensor(perm, device=r.device)] @ At(r))
        shares["registers"] = _share(r, rn)
        put(p, rn[None])
    return shares


def install_even_head(model: nn.Module, A) -> None:
    """Insert EvenProject(A) between the final RMSNorm and the final affine (model folded with
    final_norm_affine=False, so model.norm is Sequential(rms, affine))."""
    norm = model.norm
    if not (isinstance(norm, nn.Sequential) and hasattr(norm, "rms") and hasattr(norm, "affine")):
        raise ValueError("model.norm must be Sequential(rms, affine) from fold_layernorm_full(final_norm_affine=False)")
    ref = norm.affine.weight
    model.norm = nn.Sequential(OrderedDict([("rms", norm.rms), ("even", EvenProject(A, ref.dtype, ref.device)),
                                            ("affine", norm.affine)]))


@torch.no_grad()
def build_vexact(adapter, A, blocks, perms, head_maps, reg_perm=None, even_head: bool = False,
                 model: nn.Module | None = None, gauge: bool = True):
    """V-exact conversion of `blocks` (contiguous) with one operator A (a deep copy; the original is
    untouched). `perms` and `head_maps` are aligned with `blocks`. Returns (model, info); info holds
    the head maps valid after surgery, the embedding breaking shares (span starting at block 0) and
    whether the position embedding was materialised."""
    blocks = [int(b) for b in blocks]
    if not blocks or blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        raise ValueError("V-exact needs a contiguous, non-empty span of blocks")
    if len(perms) != len(blocks) or len(head_maps) != len(blocks):
        raise ValueError("one perm and one HeadMaps per block")
    A = exact_involution(A)
    m = copy.deepcopy(adapter.model if model is None else model)
    materialised = materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    emb = project_embeddings(m, adapter, A, reg_perm) if blocks[0] == 0 else None
    k = len(blocks)
    apply_surgery_two(m, blocks, [A] * k, [A] * k, list(perms))
    maps_after = apply_attention_surgery(m, blocks, [A] * k, [A] * k, list(head_maps), gauge=gauge)
    if even_head:
        install_even_head(m, A)
    m.eval()
    return m, {"maps_after": maps_after, "embedding_shares": emb, "pos_materialised": materialised,
               "even_head": bool(even_head), "blocks": blocks, "d_plus": int((involution_trace64(A) + A.shape[0]) // 2)}


# ----------------------------------------------------------------------------------------------
# Registers
# ----------------------------------------------------------------------------------------------
class RegisterScores:
    """Streaming alignment scores of register tokens, for register matching under an operator.

    For each named operator (A, last layer) and every layer l <= last layer:
        S[name][l, i, j] = sum over images <xt_i, x_j A>   (xt_i: mirrored image's register i),
    plus per-register energies E[l, j] = sum ||x_j||^2 and Et[l, i] = sum ||xt_i||^2. Tokens are
    channel-centred first (the folded model's stream). Use as residual_cb of stats.run_paired.
    """

    def __init__(self, operators: dict, n_layers: int, n_reg: int, start: int = 1, device="cpu"):
        self.R, self.start, self.device = int(n_reg), int(start), device
        kw = dict(dtype=torch.float64, device=device)
        self.ops = {k: (torch.as_tensor(np.asarray(A, dtype=np.float64), **kw), int(last))
                    for k, (A, last) in operators.items()}
        self.S = {k: torch.zeros(n_layers, self.R, self.R, **kw) for k in self.ops}
        self.E = torch.zeros(n_layers, self.R, **kw)
        self.Et = torch.zeros(n_layers, self.R, **kw)
        self.n = torch.zeros(n_layers, **kw)

    def update(self, layer: int, x: torch.Tensor, xt: torch.Tensor) -> None:
        if self.R == 0:
            return
        sl = slice(self.start, self.start + self.R)
        X = center_channels(x[:, sl].to(device=self.device, dtype=torch.float64))
        Xt = center_channels(xt[:, sl].to(device=self.device, dtype=torch.float64))
        self.E[layer] += (X * X).sum((0, 2))
        self.Et[layer] += (Xt * Xt).sum((0, 2))
        self.n[layer] += X.shape[0]
        for k, (A, last) in self.ops.items():
            if layer <= last:
                self.S[k][layer] += torch.einsum("bid,bjd->ij", Xt, X @ A)

    def cosines(self, key, layers) -> np.ndarray:
        """Mean over `layers` of S[l, i, j] / sqrt(Et[l, i] E[l, j]), symmetrised ([R, R], in [-1, 1])."""
        layers = [int(l) for l in layers]
        out = np.zeros((self.R, self.R))
        for l in layers:
            den = torch.sqrt(torch.outer(self.Et[l], self.E[l])).clamp_min(1e-300)
            out += (self.S[key][l] / den).cpu().numpy()
        out /= max(len(layers), 1)
        return 0.5 * (out + out.T)


def match_registers(S: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Register involution maximising sum_i S[i, pi(i)] (exact; attention.match_heads). A pair is
    kept only if its gain 2 S_ij - S_ii - S_jj exceeds `margin`; otherwise both stay even."""
    S = np.asarray(S, dtype=np.float64)
    R = S.shape[0]
    if R == 0:
        return np.zeros(0, dtype=np.int64)
    S = 0.5 * (S + S.T)
    if margin:
        S = S - 0.5 * float(margin) * (1.0 - np.eye(R))
    return np.asarray(match_heads(S), dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# Exactness probes
# ----------------------------------------------------------------------------------------------
def align_mirror_tokens(t: torch.Tensor, grid_hw, n_prefix: int, reg_perm=None, reg_start: int = 1) -> torch.Tensor:
    """Mirrored image's tokens aligned with the original's: patch grid un-flipped and register j
    replaced by register pi(j) (pi an involution)."""
    u = unflip_tokens(t, grid_hw, n_prefix)
    if reg_perm is not None and len(reg_perm):
        R = len(reg_perm)
        idx = torch.arange(u.shape[1], device=u.device)
        idx[reg_start:reg_start + R] = reg_start + torch.as_tensor(np.asarray(reg_perm), device=u.device)
        u = u[:, idx]
    return u


@torch.no_grad()
def stream_equivariance(adapter, model: nn.Module, x: torch.Tensor, A, reg_perm=None, reg_start: int = 1) -> list:
    """For layers 0..L: error of (aligned mirrored tokens) - (original tokens) A, in float64.

    Rows: layer, max_abs_err, max_abs_ref (max |original token entry|), rel_max = ratio of the two,
    err_sq and ref_sq (sums, for a relative Frobenius error over batches)."""
    B = x.shape[0]
    t = adapter.layer0_tokens(torch.cat([x, mirror(x)], dim=0), model)
    A64 = torch.as_tensor(np.asarray(A, dtype=np.float64), dtype=torch.float64, device=t.device)
    rows = []

    def rec(layer, t):
        X = t[:B].double()
        Xt = align_mirror_tokens(t[B:], adapter.grid_hw, adapter.n_prefix_tokens, reg_perm, reg_start).double()
        diff = Xt - X @ A64
        ma, mr = float(diff.abs().max()), float(X.abs().max())
        rows.append({"layer": layer, "max_abs_err": ma, "max_abs_ref": mr, "rel_max": ma / mr if mr > 0 else float("nan"),
                     "err_sq": float((diff ** 2).sum()), "ref_sq": float((X ** 2).sum())})

    rec(0, t)
    for i, blk in enumerate(model.blocks):
        t = blk(t)
        rec(i + 1, t)
    return rows


@torch.no_grad()
def feature_invariance(adapter, model: nn.Module, x: torch.Tensor) -> dict:
    """max |f(mirror x) - f(x)| / max |f(x)| for the k-NN feature (final-norm CLS)."""
    f = adapter.cls_feature(x, model).double()
    ft = adapter.cls_feature(mirror(x), model).double()
    ma, mr = float((ft - f).abs().max()), float(f.abs().max())
    return {"max_abs_diff": ma, "max_abs_ref": mr, "rel_max": ma / mr if mr > 0 else float("nan")}


# ----------------------------------------------------------------------------------------------
# Matched random structure (the random-structure control)
# ----------------------------------------------------------------------------------------------
def random_involution_fixing_ones(d: int, d_plus: int, seed: int) -> np.ndarray:
    """Random orthogonal involution with d_plus eigenvalues +1 and A 1 = 1 (commutes with LayerNorm
    centring, like a fitted operator). Same construction as scripts/heal.py."""
    if not 1 <= d_plus <= d:
        raise ValueError("d_plus must be in [1, d]")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return exact_involution(0.5 * (A + A.T))


def random_head_structure(H: int, dh: int, n_pairs: int, seed: int, self_traces=None) -> HeadMaps:
    """Random head involution with `n_pairs` twin pairs and random maps: Haar-orthogonal B and C on
    pairs (B[g] = B[h]^T), random symmetric involutions on self-heads with the given (tr B, tr C)
    per self-head (in index order), or random traces if None."""
    rng = np.random.default_rng(seed)
    sigma = random_pairing(H, int(n_pairs), seed=int(rng.integers(1 << 31)))
    selfs = np.flatnonzero(sigma == np.arange(H))
    if self_traces is None:
        self_traces = [(2 * int(rng.integers(1, dh)) - dh, 2 * int(rng.integers(1, dh)) - dh) for _ in selfs]
    if len(self_traces) != len(selfs):
        raise ValueError("one (tr B, tr C) per self-head")
    B, C = np.zeros((H, dh, dh)), np.zeros((H, dh, dh))
    for h in range(H):
        g = int(sigma[h])
        if g > h:
            B[h], C[h] = random_orthogonal(dh, rng), random_orthogonal(dh, rng)
            B[g], C[g] = B[h].T, C[h].T
    for h, (tB, tC) in zip(selfs, self_traces):
        B[h] = random_involution(dh, (dh + int(tB)) // 2, seed=int(rng.integers(1 << 31)))
        C[h] = random_involution(dh, (dh + int(tC)) // 2, seed=int(rng.integers(1 << 31)))
    return HeadMaps(sigma=sigma, B=B, C=C, info={"random_seed": int(seed)})


def random_head_maps_like(hm: HeadMaps, seed: int) -> HeadMaps:
    """Random head structure with the same numbers of twin pairs and self-heads and the same self-head
    traces as `hm`, so it has the same number of free parameters."""
    traces = [(involution_trace64(hm.B[h]), involution_trace64(hm.C[h])) for h in hm.self_heads]
    return random_head_structure(hm.H, hm.dh, hm.n_pairs, seed, traces)


def matched_random_structure(A_layers, perms, seed: int, fix_ones: bool = True, mode: str = "per_layer",
                             heads=None) -> dict:
    """Random-structure control matched to a fitted structure: per layer a random involution with the
    fitted d+ (seed*1000 + l), per block a random neuron pairing with the fitted numbers of pairs
    (seed*1000 + 500 + b) and, if `heads` is given, random head maps matched per block
    (seed*1000 + 900 + b). Operators and pairings are the same as scripts/heal.py produces."""
    d = int(np.asarray(A_layers[0]).shape[0])

    def rand_A(A, s):
        dp = int(round((float(np.trace(A)) + d) / 2))
        return random_involution_fixing_ones(d, dp, s) if fix_ones else exact_involution(random_involution(d, dp, s))

    if mode == "shared":
        A_r = [rand_A(A_layers[0], seed * 1000)] * len(A_layers)
    else:
        A_r = [rand_A(A, seed * 1000 + l) for l, A in enumerate(A_layers)]
    perms_r = []
    for b, p in enumerate(perms):
        p = np.asarray(p)
        h = len(p)
        perms_r.append(random_pairing(h, int((p != np.arange(h)).sum() // 2), seed=seed * 1000 + 500 + b))
    heads_r = None if heads is None else [random_head_maps_like(hm, seed * 1000 + 900 + b) for b, hm in enumerate(heads)]
    return {"A_layers": A_r, "perms": perms_r, "heads": heads_r}


# ----------------------------------------------------------------------------------------------
# Attention SVD baseline and parameter counts
# ----------------------------------------------------------------------------------------------
def attention_svd_ranks(free: dict, d: int, D: int) -> tuple:
    """Ranks (r_q, r_k, r_v, r_o) whose factorisations r (D + d) do not exceed the mirror free
    parameter counts of Wq, Wk, Wv, Wo (attention.attention_free_params)."""
    return tuple(max(1, int(free[k] // (D + d))) for k in ("Wq", "Wk", "Wv", "Wo"))


@torch.no_grad()
def apply_attention_svd(model: nn.Module, blocks, ranks) -> None:
    """Rank truncation of the folded Wq, Wk, Wv and Wo of `blocks` (biases kept; norm1 affine and
    ls1 absorbed), in place. `ranks` holds one (r_q, r_k, r_v, r_o) per block."""
    if len(blocks) != len(ranks):
        raise ValueError("one rank tuple per block")
    for b, r in zip(blocks, ranks):
        blk = model.blocks[int(b)]
        f = fold_attn(blk)
        for key, rr in zip(("Wq", "Wk", "Wv", "Wo"), r):
            f[key] = svd_truncate(f[key], int(rr))
        write_absorbed_attn(blk, **f)


def mlp_mirror_params(h: int, d: int, perm, A_in, A_out) -> int:
    """Free parameters of the equivariant fc1 and fc2 matrices (surgery.equivariant_param_count)."""
    tp = perm_trace(perm)
    return (equivariant_param_count(h, d, tp, involution_trace64(A_in))
            + equivariant_param_count(h, d, tp, involution_trace64(A_out)))

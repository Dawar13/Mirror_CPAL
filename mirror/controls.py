"""A03 discovery null controls: does the discovery pipeline invent structure? (FINAL_PHASE_PLAN.md v2, Section 4
A03 and rule A03-1 in Section 6; docs/final_phase_design.md; notes/paper/method_spec.md Sections 1, 2 and 8.)

The discovery pipeline runs unchanged on inputs whose pairing carries no mirror relation. Only the pairing of the
two halves of every paired batch changes; every statistic, fit, matching, breaking share and null is existing code:

  pass 1    one paired pass [x; T(x')] per batch through the frozen model. The tokens of T(x') are re-indexed by a
            token alignment map (prefix tokens keep their index, as in every earlier run) and folded into
            stats.ResidualStats (channel-centred C_l = sum X^T Xt and G_l = sum X^T X per stream layer) and
            stats.MLPStats (fc1 outputs; R[i, j] = corr(zt_i, z_j) per block) through group_stats.GroupPairedHooks.
  fits      per-layer involutions (variants.fit_layer_operator, the same as groups.fit_order2), the span operators
            A*_k (m1.fit_span_operator over stream layers 0..k, each layer divided by tr(G_l)) and the twin matching
            (matching.match_many, top_k candidates).
  pass 2    the same pairing again for the head statistics (attention.AttnStats) and the register scores under
            every A*_k (m1.RegisterScores): head maps (attention.pair_and_fit_heads) and the register pairing
            (m1.match_registers). The result is a full mirror_family.SpanStructure, the object
            mirror_family.fit_span_structure returns (without its pair_actcorr and pair_wcos pairings), so
            mirror_family.build_mirror_variant converts a model with it unchanged.
  measures  as MUST-04 (scripts/measure_groups.py, group LR). Per block: the twin fraction (groups.orbit_fractions:
            share of hidden units in a matched pair whose symmetrised correlation S_ij is at least 0.5) and the
            breaking shares of the real folded MLP weights, fc1 against A_b and fc2 against A_(b+1),
            s_mlp = (s_fc1 + s_fc2) / 2 (groups.mlp_group_shares). The null: per draw, random involutions with the
            fitted d+ per layer (A 1 = 1; groups.random_operator_like) and random pairings with the fitted cycle
            type per block (groups.random_perm_like), seeded exactly as measure_groups.seed_for(base, "LR", draw,
            slot). Ratio to null = mean over the range's blocks of s_mlp divided by the mean over the same blocks
            of the null-mean s_mlp.

Rows (the pairing of the two halves):
  mirror   reference: x' = x, T = left-right flip, alignment = un-flip the patch grid (groups.token_index("LR")).
           Identical to stats.collect_fit_stats (tested: the structure equals fit_span_structure's).
  C1       shuffled correspondence: x' = the fit image at derangement[i] (a fixed seeded derangement of the fit
           set), T = flip, alignment = un-flip as in the reference.
  C2       no re-alignment: x' = x, T = flip, alignment = identity: token (r, c) of x meets token (r, c) of
           mirror(x).
  C3       non-symmetry transformation: x' = x, T = a fixed seeded random involutive permutation of the pixel
           positions inside every patch, the same in every patch and colour channel, with as many fixed pixels as
           the flip has inside one patch (p for an odd patch width p, none for an even one). The patch grid does
           not move, so the alignment is the identity. It keeps every patch's pixel values but is not a symmetry
           of natural images.
  C4       random-operator null: no pass. The reference structure's own null draws (random involutions with the
           fitted d+ and d-, A 1 = 1, and random pairings with the fitted cycle type), reported as a row: each
           draw's ratio to the mean of the other draws (leave one out), and the twin fraction of its random
           pairing scored with the reference's correlations.

Rule A03-1 (pre-registered): for each of C1, C2 and C3, twin fraction in blocks 1 to 6 at most 0.05 and ratio to
null in blocks 1 to 6 at least 0.9 (`score_a03`). PREREG_FINAL_PHASE.md Section 2.4 fixes the ratio as the
split-half ratio below; scripts/fp_a03_controls.py scores it that way and reports the same-sample ratio as a
diagnostic.

Split-half ratio (the statistic A03-1 scores; `split_half_fit`, `split_half_summary`). The matching
and the operator fit read the same cross-moments, so a structure fitted on any pairing, even one carrying only
sampling noise, agrees with itself through the weights and its breaking share falls below the MUST-04 null (fitted
operator with a random pairing, or the reverse, sits at the null; both fitted together do not). On the exactly
equivariant tiny network of tests/test_fp_controls.py, C1, C2 and C3 have a twin fraction near 0 but a ratio near
0.88. Fitting the operators on one half of the fit images and the pairings on the other removes this: C1 and C2
return to about 1.0; C3 stays near 0.9 because a fixed pixel permutation leaves a fixed linear relation between the
patch embeddings of x and T(x) (systematic, so it holds in both halves).

What the controls keep (read before interpreting a failure). Prefix tokens (CLS, registers) keep their index in
every row, so C2 pairs them exactly as the reference does. C1 keeps the true re-alignment, so the parts of the
tokens that do not depend on the image (position embeddings, the mean image, mean prefix tokens) stay mirror
paired. In C2 and C3 both halves come from the same image, so features that do not depend on position (C2), or
on the pixel order inside a patch (C3), stay paired. Structure carried only by those parts is real structure
that survives the broken correspondence. The spectrum summaries and the agreement of each control's operators
and pairings with the reference's are reported to show where any structure a control finds comes from.

Also here, for the CPU tests and the --dry-run of scripts/fp_a03_controls.py: `tiny_adapter` (a tiny random timm
ViT), `make_equivariant_network` (that ViT converted to an exactly mirror-equivariant network with the existing
Mirror-exact conversion) and `unfold_rmsnorm`.

Sources: this repository only (method_spec.md, scripts/measure_groups.py, mirror/mirror_family.py); no external
source was needed. Blocks are 0-based in code and 1-based in every output ("block 1" is code index 0).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import groups as GR
from .adapters import TimmTinyAdapter, ViTAdapter

CONTROL_ROWS = ("mirror", "C1", "C2", "C3")         # rows with their own paired passes and fitted structure
NULL_ROW = "C4"
ROWS = CONTROL_ROWS + (NULL_ROW,)
TESTED_CONTROLS = ("C1", "C2", "C3")                 # the controls rule A03-1 scores
A03_TWIN_MAX = 0.05                                  # pre-registered (FINAL_PHASE_PLAN.md Section 6, A03-1)
A03_RATIO_MIN = 0.9
TWIN_THRESHOLD = 0.5                                 # twin fraction: S_ij >= 0.5 (groups.orbit_fractions frac_2_t50)
NULL_GROUP = "LR"                                    # seed table and null construction of measure_groups, group LR
PRIMARY_RANGE = "1-6"

CONTROL_SPECS = {
    "mirror": {"transform": "flip", "align": "unflip", "derange": False,
               "label": "real left-right mirror (reference)"},
    "C1": {"transform": "flip", "align": "unflip", "derange": True,
           "label": "shuffled correspondence: the mirror of a different image (seeded derangement), re-aligned"},
    "C2": {"transform": "flip", "align": "identity", "derange": False,
           "label": "no re-alignment: token (r, c) of x against token (r, c) of mirror(x)"},
    "C3": {"transform": "patch_perm", "align": "identity", "derange": False,
           "label": "random involutive pixel permutation inside every patch (same in every patch and channel)"},
}
NULL_LABEL = ("random-operator null: random involutions with the fitted d+ and d- (A 1 = 1) and random pairings "
              "with the fitted cycle type (the reference's null draws)")

__all__ = [
    "CONTROL_ROWS", "NULL_ROW", "ROWS", "TESTED_CONTROLS", "A03_TWIN_MAX", "A03_RATIO_MIN", "TWIN_THRESHOLD",
    "CONTROL_SPECS", "NULL_LABEL", "derangement", "flip_patch_index", "patch_involution", "patch_pixel_index",
    "permute_pixels", "Control", "make_control", "collect_control_pass", "cross_moment_spectra", "spectrum_summary",
    "ControlFit", "fit_control", "null_seed", "null_pairings", "null_twin_fractions", "measure_structure",
    "range_blocks", "range_summary", "null_row_summary", "split_half_fit", "split_half_summary", "agreement_with",
    "score_a03", "knn_after_surgery",
    "TinyTimmAdapter", "tiny_adapter", "with_model", "unfold_rmsnorm", "make_equivariant_network",
]


# ----------------------------------------------------------------------------------------------
# Input actions: image derangement, within-patch pixel involution, token alignment
# ----------------------------------------------------------------------------------------------
def derangement(n: int, seed: int) -> np.ndarray:
    """A uniformly random permutation of range(n) without fixed points (rejection sampling with the seed).

    p[i] is the image whose mirror is paired with image i (C1). About 1 in e draws is a derangement, so the
    loop ends after a few tries."""
    n = int(n)
    if n < 2:
        raise ValueError("a derangement needs at least 2 images")
    rng = np.random.default_rng(int(seed))
    ar = np.arange(n)
    for _ in range(10_000):
        p = rng.permutation(n)
        if not np.any(p == ar):
            return p.astype(np.int64)
    raise RuntimeError("no derangement found")       # probability about (1 - 1/e) ** 10000


def flip_patch_index(patch: int) -> np.ndarray:
    """The left-right flip inside one patch as an index over its patch * patch pixels (row-major)."""
    patch = int(patch)
    i, j = np.divmod(np.arange(patch * patch), patch)
    return (i * patch + (patch - 1 - j)).astype(np.int64)


def patch_involution(patch: int, seed: int, n_fixed: int | None = None) -> np.ndarray:
    """sigma over the patch * patch pixel positions of one patch (row-major) with sigma[sigma] = identity.

    n_fixed fixed pixels (default: as many as the left-right flip has inside a patch, `patch` for an odd width and
    0 for an even one); the other pixels are paired at random with the seed. The flip itself and the identity are
    never returned."""
    patch = int(patch)
    n = patch * patch
    flip = flip_patch_index(patch)
    if n_fixed is None:
        n_fixed = int((flip == np.arange(n)).sum())
    n_fixed = int(n_fixed)
    if not 0 <= n_fixed <= n - 2 or (n - n_fixed) % 2:
        raise ValueError(f"cannot pair {n - n_fixed} of {n} pixels")
    rng = np.random.default_rng(int(seed))
    ident = np.arange(n)
    for _ in range(1000):
        moved = rng.permutation(n)[n_fixed:]
        sigma = ident.copy()
        a, b = moved[0::2], moved[1::2]
        sigma[a], sigma[b] = b, a
        if not np.array_equal(sigma, flip):
            return sigma.astype(np.int64)
    raise RuntimeError("could not draw an involution different from the flip")


def patch_pixel_index(height: int, width: int, patch: int, sigma) -> np.ndarray:
    """Flat index over height * width pixels that applies sigma inside every patch: out[q] = x[index[q]].

    The same sigma acts in every patch (and, through permute_pixels, in every colour channel); patches stay in
    place, so the patch grid, and hence the token grid, is unchanged."""
    height, width, patch = int(height), int(width), int(patch)
    if height % patch or width % patch:
        raise ValueError(f"image {height} x {width} does not divide into {patch} x {patch} patches")
    sigma = np.asarray(sigma, dtype=np.int64)
    if sigma.shape != (patch * patch,) or not np.array_equal(np.sort(sigma), np.arange(patch * patch)):
        raise ValueError("sigma must be a permutation of the patch pixels")
    r = np.arange(height)[:, None]
    c = np.arange(width)[None, :]
    src = sigma[(r % patch) * patch + (c % patch)]
    sr = (r // patch) * patch + src // patch
    sc = (c // patch) * patch + src % patch
    return (sr * width + sc).reshape(-1).astype(np.int64)


def permute_pixels(x: torch.Tensor, index) -> torch.Tensor:
    """Images [..., H, W] with their pixels re-indexed by a flat index over H * W (patch_pixel_index)."""
    H, W = int(x.shape[-2]), int(x.shape[-1])
    idx = index if isinstance(index, torch.Tensor) else torch.as_tensor(np.asarray(index), dtype=torch.long)
    idx = idx.to(device=x.device, dtype=torch.long)
    if idx.numel() != H * W:
        raise ValueError(f"index has {idx.numel()} entries for {H} x {W} pixels")
    return x.reshape(*x.shape[:-2], H * W).index_select(-1, idx).reshape(x.shape)


@dataclass(frozen=True)
class Control:
    """One pairing of the paired pass: the partner image x' of x, the transform T and the token alignment."""

    name: str
    transform: str          # "flip" or "patch_perm"
    align: str              # "unflip" (groups.token_index("LR")) or "identity"
    derange: bool           # x' = a different fit image (seeded derangement) instead of x
    seed: int = 0           # seed of the derangement (C1) and of the pixel involution (C3)
    label: str = ""

    def partner(self, n: int) -> np.ndarray:
        """partner[i]: position in the fit set of the image transformed and paired with image i."""
        return derangement(n, self.seed) if self.derange else np.arange(int(n), dtype=np.int64)

    def token_index(self, grid_hw) -> np.ndarray:
        """Alignment of the transformed half's patch tokens (groups.align_tokens convention)."""
        H, W = (grid_hw, grid_hw) if isinstance(grid_hw, int) else tuple(int(v) for v in grid_hw)
        if self.align == "unflip":
            return np.asarray(GR.token_index("LR", (H, W)), dtype=np.int64)
        if self.align == "identity":
            return np.arange(H * W, dtype=np.int64)
        raise ValueError(f"unknown alignment {self.align!r}")

    def pixel_sigma(self, patch: int):
        return patch_involution(patch, self.seed) if self.transform == "patch_perm" else None

    def pixel_index(self, img_size: int, patch: int):
        sigma = self.pixel_sigma(patch)
        return None if sigma is None else patch_pixel_index(img_size, img_size, patch, sigma)

    def apply(self, x: torch.Tensor, pixel_index=None) -> torch.Tensor:
        """T(x) for normalised images [B, 3, H, W]."""
        if self.transform == "flip":
            return torch.flip(x, dims=[-1])
        if self.transform == "patch_perm":
            if pixel_index is None:
                raise ValueError("patch_perm needs the pixel index (Control.pixel_index)")
            return permute_pixels(x, pixel_index)
        raise ValueError(f"unknown transform {self.transform!r}")

    def describe(self) -> dict:
        return {"name": self.name, "transform": self.transform, "align": self.align, "derange": bool(self.derange),
                "seed": int(self.seed), "label": self.label}


def make_control(name: str, seed: int = 0) -> Control:
    if name not in CONTROL_SPECS:
        raise ValueError(f"unknown control {name!r}; choose from {tuple(CONTROL_SPECS)}")
    s = CONTROL_SPECS[name]
    return Control(name=name, transform=s["transform"], align=s["align"], derange=bool(s["derange"]),
                   seed=int(seed), label=s["label"])


# ----------------------------------------------------------------------------------------------
# Paired pass
# ----------------------------------------------------------------------------------------------
def _image_batch(adapter, images, pos, device) -> torch.Tensor:
    """Normalised images [B, 3, H, W] (adapter dtype) for positions `pos` of a uint8 [N, H, W, 3] array."""
    from .data import normalize_for

    arr = np.ascontiguousarray(np.asarray(images[np.asarray(pos, dtype=np.int64)]))
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()     # NCHW like the DataLoader collate (same conv path)
    return normalize_for(adapter, t.to(device))


def _check_images(adapter, images) -> int:
    shp = tuple(np.shape(images))
    if len(shp) != 4 or shp[-1] != 3 or np.asarray(images[:1]).dtype != np.uint8:
        raise ValueError(f"images must be uint8 [N, H, W, 3], got {shp}")
    if shp[1] != adapter.img_size or shp[2] != adapter.img_size:
        raise ValueError(f"images are {shp[1]} x {shp[2]}, the model takes {adapter.img_size} x {adapter.img_size}")
    return int(shp[0])


@torch.no_grad()
def collect_control_pass(adapter, images, device, control: Control, *, positions=None, blocks=None, layers=None,
                         batch_size: int = 16, residual: bool = True, mlp: bool = True, attn: bool = False,
                         register_ops: dict | None = None, model: nn.Module | None = None) -> dict:
    """One paired pass over the fit images (uint8 [N, H, W, 3]) with the control's pairing.

    Every batch of fit positions p (`positions`, default all, in order) runs as [x_p; T(x_partner[p])], with
    partner = control.partner(N) over the whole fit set; group_stats.GroupPairedHooks re-indexes the second half's
    patch tokens with control.token_index (prefix tokens keep their index) and hands both halves to the
    accumulators: stats.ResidualStats for the stream `layers` (residual; default 0..L), stats.MLPStats and
    attention.AttnStats for `blocks` (mlp, attn; default all) and m1.RegisterScores for `register_ops`
    ({k: (A, last layer)}; models with registers only, every layer). The forward stops after the last block any
    statistic needs. Returns "layers" and "residual" (C, G, sq_x, sq_xt, n, indexed by position in "layers"),
    "blocks" and "R" ({block: [h, h] correlation}), "attn" (AttnStats, slots in the order of "blocks"), "registers"
    (RegisterScores), "partner", "n_images"."""
    from .attention import AttnStats, attn_dims
    from .group_stats import GroupPairedHooks, _Stop
    from .m1 import RegisterScores, register_info
    from .stats import MLPStats, ResidualStats

    m = adapter.model if model is None else model
    L, d, h = len(m.blocks), adapter.embed_dim, adapter.mlp_hidden
    n_full = _check_images(adapter, images)
    partner = control.partner(n_full)
    pos_all = np.arange(n_full) if positions is None else np.asarray(positions, dtype=np.int64)
    blocks = list(range(L)) if blocks is None else sorted({int(b) for b in blocks})
    bslot = {b: i for i, b in enumerate(blocks)}
    rs = None
    if register_ops:
        reg = register_info(m, adapter)
        if reg["R"]:
            rs = RegisterScores(register_ops, L + 1, reg["R"], reg["start"], device)
    layers = list(range(L + 1)) if (layers is None or rs is not None) else sorted({int(x) for x in layers})
    lslot = {l: i for i, l in enumerate(layers)}
    idx = control.token_index(adapter.grid_hw)
    pix = control.pixel_index(adapter.img_size, adapter.patch_size)
    pix_t = None if pix is None else torch.as_tensor(pix, dtype=torch.long, device=device)
    res = ResidualStats(len(layers), d, device, center=True) if residual else None
    ms = MLPStats(len(blocks), h, device) if mlp else None
    ast = None
    if attn:
        H, dh = attn_dims(m.blocks[0])
        ast = AttnStats(len(blocks), H, dh, device)
    need_res = res is not None or rs is not None

    def residual_cb(layer, x, xt):
        if res is not None:
            res.update(lslot[layer], x, xt)
        if rs is not None:
            rs.update(layer, x, xt)

    last = max([b for b in blocks if mlp or attn] + ([max(layers) - 1] if need_res else []) + [0])
    hooks = GroupPairedHooks(adapter, m, idx, blocks=blocks, layers=layers if need_res else [],
                             residual_cb=residual_cb if need_res else None,
                             fc1_cb=(lambda b, z, zt: ms.update(bslot[b], z, zt)) if ms is not None else None,
                             qkv_cb=(lambda b, y, yt: ast.update(bslot[b], y, yt)) if ast is not None else None,
                             stop_after=last)
    t0 = time.time()
    bs = max(1, int(batch_size))
    with hooks:
        for s in range(0, len(pos_all), bs):
            pos = pos_all[s:s + bs]
            x = _image_batch(adapter, images, pos, device)
            xp = _image_batch(adapter, images, partner[pos], device) if control.derange else x
            try:
                adapter.forward_tokens(torch.cat([x, control.apply(xp, pix_t)], dim=0), m)
            except _Stop:
                pass
    out = {"control": control.name, "n_images": int(len(pos_all)), "partner": partner, "layers": layers,
           "blocks": blocks, "seconds": round(time.time() - t0, 2)}
    if res is not None:
        out["residual"] = res.numpy()
    if ms is not None:
        out["R"] = {b: ms.correlation(bslot[b]) for b in blocks}
    if ast is not None:
        out["attn"] = ast
    if rs is not None:
        out["registers"] = rs
    return out


# ----------------------------------------------------------------------------------------------
# Spectra
# ----------------------------------------------------------------------------------------------
def cross_moment_spectra(C, G) -> tuple:
    """Sorted eigenvalues of the symmetrised normalised cross-moments: per stream layer l,
    sym(Pc C_l Pc) / tr(G_l) (the matrix whose signs give the per-layer operator), and for the span, the mean over
    all layers of those matrices (the matrix of m1.fit_span_operator over layers 0..L divided by L + 1, so the
    signs are those of A*_L). Pc = I - 11^T / d. With an exact symmetry (Xt = X A) the absolute values of each
    layer's eigenvalues add up to 1. Returns (eig_layers [L + 1, d], eig_span [d]), ascending."""
    from .m1 import centre_projector

    C = np.asarray(C, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    n_layers, d = C.shape[0], C.shape[-1]
    Pc = centre_projector(d)
    eig = np.zeros((n_layers, d))
    acc = np.zeros((d, d))
    for l in range(n_layers):
        M = Pc @ C[l] @ Pc / max(float(np.trace(G[l])), 1e-300)
        M = 0.5 * (M + M.T)
        eig[l] = np.linalg.eigvalsh(M)
        acc += M
    return eig, np.linalg.eigvalsh(acc / n_layers)


def spectrum_summary(eigs, top: int = 10) -> dict:
    """Shape of one spectrum: abs_sum (sum |lam|: the share of the cross-moment the fitted involution explains, 1
    for an exact symmetry), signed_sum (sum lam: closeness of the pairing to the identity), neg_mass_share (share
    of sum |lam| on negative eigenvalues: the odd share), participation ((sum |lam|)^2 / sum lam^2: effective
    number of directions), n_negative, top_abs (the `top` eigenvalues of largest |lam|, signed) and top_negative
    (the `top` most negative)."""
    lam = np.asarray(eigs, dtype=np.float64).ravel()
    a = np.abs(lam)
    tot = float(a.sum())
    sq = float((lam * lam).sum())
    order = np.argsort(-a, kind="stable")
    return {"abs_sum": tot, "signed_sum": float(lam.sum()),
            "neg_mass_share": float(a[lam < 0].sum()) / tot if tot > 0 else float("nan"),
            "participation": tot * tot / sq if sq > 0 else float("nan"),
            "n_negative": int((lam < 0).sum()),
            "top_abs": [float(x) for x in lam[order[:top]]],
            "top_negative": [float(x) for x in np.sort(lam[lam < 0])[:top]]}


# ----------------------------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------------------------
def null_seed(base: int, draw: int, slot: int) -> int:
    """scripts/measure_groups.py seed_for(base, "LR", draw, slot): layer operators use slot = layer, hidden
    permutations slot = 3000 + block (tested against the script)."""
    return int(base) * 10_000_000 + GR.GROUPS.index(NULL_GROUP) * 1_000_000 + int(draw) * 10_000 + int(slot)


def null_pairings(perm, block: int, n_draws: int, seed: int) -> list:
    """The MUST-04 null's random pairings of one block: the fitted pairing's cycle type, one per draw."""
    return [GR.random_perm_like(NULL_GROUP, [perm], np.random.default_rng(null_seed(seed, k, 3000 + block)))[0]
            for k in range(int(n_draws))]


def null_twin_fractions(perms, R_list, n_draws: int, seed: int) -> np.ndarray:
    """[n_draws, L]: twin fraction (S_ij >= 0.5) of each null pairing scored with this row's own symmetrised
    correlations S = (R + R^T) / 2 (the C4 row uses the reference's)."""
    from .matching import symmetrise

    L = len(perms)
    out = np.zeros((int(n_draws), L))
    for b in range(L):
        S = symmetrise(R_list[b])
        h = S.shape[0]
        for k, p in enumerate(null_pairings(perms[b], b, n_draws, seed)):
            out[k, b] = GR.orbit_fractions([p], [S[np.arange(h), p]])["frac_2_t50"]
    return out


@dataclass
class ControlFit:
    """The fitted structure of one control row and what the measures need besides it."""

    control: Control
    structure: object                   # mirror_family.SpanStructure
    d_plus: list                        # per stream layer 0..L (per-layer operators)
    d_minus: list
    eig_layers: np.ndarray              # [L + 1, d] (cross_moment_spectra)
    eig_span: np.ndarray                # [d]
    scores: np.ndarray                  # [L, h]: S[i, perm[i]] (the self-score for a self-twin)
    self_scores: np.ndarray             # [L, h]: diag R
    null_twin: np.ndarray | None = None  # [n_draws, L] (null_twin_fractions)
    info: dict = field(default_factory=dict)

    @property
    def span_d_plus(self) -> int:
        """d+ of the full-span operator A*_L (the operator of the full conversion)."""
        return _d_plus(self.structure.A_star[self.structure.depth])

    @property
    def span_d_minus(self) -> int:
        return int(self.structure.d - self.span_d_plus)


def _d_plus(A) -> int:
    A = np.asarray(A, dtype=np.float64)
    return int(round((A.shape[0] + float(np.trace(A))) / 2.0))


@torch.no_grad()
def fit_control(adapter, images, device, control: Control, *, model_name: str = "", fit_seed: int = 0,
                top_k: int = 10, workers: int = 1, batch_size: int = 16, register_margin: float = 0.0,
                n_null: int = 20, null_seed_base: int = 0, model: nn.Module | None = None) -> ControlFit:
    """The discovery pipeline on the control's pairing: pass 1 (C_l, G_l, fc1 correlations), the per-layer and span
    operators, the twin matching, pass 2 (head and register statistics), the SpanStructure. For control "mirror"
    the structure equals mirror_family.fit_span_structure(..., pairings=False) on the same images and batches."""
    from .attention import pair_and_fit_heads
    from .heal import exact_involution
    from .m1 import fit_span_operator, match_registers, register_info
    from .matching import match_many
    from .mirror_family import SpanStructure
    from .variants import fit_layer_operator

    m = adapter.model if model is None else model
    L, h = len(m.blocks), adapter.mlp_hidden
    t0 = time.time()
    p1 = collect_control_pass(adapter, images, device, control, batch_size=batch_size, residual=True, mlp=True,
                              model=m)
    res = p1["residual"]
    C, G = res["C"], res["G"]
    ops = [fit_layer_operator(C[l], True) for l in range(L + 1)]
    eig_layers, eig_span = cross_moment_spectra(C, G)
    A_star, star_info = {}, {}
    for k in range(1, L + 1):
        A, inf = fit_span_operator(C, range(k + 1), G)
        A_star[k] = exact_involution(A)
        star_info[k] = {"d_plus": int(inf["d_plus"]), "d_minus": int(inf["d_minus"]), "n_zero": int(inf["n_zero"])}
    R_dict = p1.pop("R")
    R_list = [R_dict[b] for b in range(L)]
    del R_dict
    t_pass1 = time.time() - t0
    t1 = time.time()
    matches = match_many(R_list, top_k=top_k, workers=workers)
    perms = [np.asarray(mm.perm, dtype=np.int64) for mm in matches]
    scores = np.stack([np.asarray(mm.score, dtype=np.float64) for mm in matches])
    self_scores = np.stack([np.diag(R).astype(np.float64) for R in R_list])
    null_twin = null_twin_fractions(perms, R_list, n_null, null_seed_base) if n_null else None
    del R_list, matches
    t_match = time.time() - t1
    t2 = time.time()
    reg = register_info(m, adapter)
    reg_ops = {k: (A_star[k], k) for k in A_star} if reg["R"] else None
    p2 = collect_control_pass(adapter, images, device, control, batch_size=batch_size, residual=False, mlp=False,
                              attn=True, register_ops=reg_ops, model=m)
    ast = p2["attn"]
    heads = []
    for b in range(L):
        hm = pair_and_fit_heads(ast.moments(b))
        hm.check(tol=1e-8)
        heads.append(hm)
    rs = p2.get("registers")
    reg_perm = {k: (match_registers(rs.cosines(k, range(k + 1)), register_margin) if rs is not None
                    else np.zeros(0, dtype=np.int64)) for k in A_star}
    del p2, ast
    t_pass2 = time.time() - t2
    n_pairs = [int((p != np.arange(h)).sum() // 2) for p in perms]
    meta = {"model": model_name, "fit_seed": int(fit_seed), "top_k": int(top_k),
            "register_margin": float(register_margin), "pass1": f"mirror.controls paired pass, control {control.name}",
            "n_pairs": n_pairs, "n_self_heads": [hm.n_self for hm in heads],
            "n_head_pairs": [hm.n_pairs for hm in heads], "n_registers": int(reg["R"]),
            "register_pairs": {int(k): int((v != np.arange(len(v))).sum() // 2) for k, v in reg_perm.items()},
            "A_star": {int(k): v for k, v in star_info.items()}, "d_plus_layers": [int(o[1]) for o in ops],
            "base_meta": {"n_tokens": int(res["n"][0]), "top_k": int(top_k), "center": True, "fit_seed": int(fit_seed)},
            "pairings_fitted": False, "fit_seconds": round(time.time() - t0, 1),
            "control": control.describe(), "n_images": int(p1["n_images"]),
            "operator_fit": "activation cross-moments of the control pairing (m1.fit_span_operator, energy normalised)"}
    st = SpanStructure(model=model_name, fit_seed=int(fit_seed), A_layers=[exact_involution(o[0]) for o in ops],
                       perms=perms, heads=heads, A_star=A_star, reg_perm=reg_perm, actcorr_perms=None,
                       wcos_perms=None, meta=meta)
    info = {"n_images": int(p1["n_images"]), "n_tokens": int(res["n"][0]), "seconds_pass1": round(t_pass1, 1),
            "seconds_match": round(t_match, 1), "seconds_pass2": round(t_pass2, 1),
            "partner": np.asarray(p1["partner"]), "pixel_sigma": control.pixel_sigma(adapter.patch_size)}
    return ControlFit(control=control, structure=st, d_plus=[int(o[1]) for o in ops],
                      d_minus=[int(o[2]) for o in ops], eig_layers=eig_layers, eig_span=eig_span, scores=scores,
                      self_scores=self_scores, null_twin=null_twin, info=info)


# ----------------------------------------------------------------------------------------------
# Measures (MUST-04, group LR)
# ----------------------------------------------------------------------------------------------
def _null_cols(values) -> tuple:
    ns = GR.null_summary(values)
    return ns["mean"], ns["lo"], ns["hi"]


def measure_structure(model: nn.Module, A_layers, perms, scores, blocks=None, *, n_draws: int = 20,
                      seed: int = 0) -> dict:
    """Twin fractions and breaking shares of the real (folded) MLP weights of `model` under the per-layer
    structure (A_layers, perms), and the MUST-04 null, exactly as scripts/measure_groups.py analyse_group does for
    group LR (fc1 against A_b, fc2 against A_(b+1); random involutions with the fitted d+ per layer, seed slot =
    layer; random pairings with the fitted cycle type per block, slot 3000 + block).
    A_layers, perms and scores may be lists or dicts; only the entries of `blocks` (and their layers b, b + 1) are
    read. Returns {"blocks": rows (1-based block, pairs, twin fractions, s_fc1, s_fc2, s_mlp with null mean, lo,
    hi), "draws": rows (block, draw, s_fc1, s_fc2, s_mlp), "n_draws", "seed"}."""
    from .surgery import fold_mlp

    L = len(model.blocks)
    blocks = list(range(L)) if blocks is None else sorted({int(b) for b in blocks})
    d = int(np.asarray(A_layers[blocks[0]]).shape[0])
    h = int(len(perms[blocks[0]]))
    f = {b: fold_mlp(model.blocks[b]) for b in blocks}
    layers = sorted(set(blocks) | {b + 1 for b in blocks})
    block_rows, draw_rows = [], []
    for b in blocks:
        act = GR.block_action(NULL_GROUP, [A_layers[b]], [A_layers[b + 1]], [np.asarray(perms[b])])
        sh = GR.mlp_group_shares(f[b], act)
        fr = GR.orbit_fractions([np.asarray(perms[b])], [np.asarray(scores[b])])
        block_rows.append({"block": b + 1, "h": h, "n_pairs": int((np.asarray(perms[b]) != np.arange(h)).sum() // 2),
                           "frac_2": fr["frac_2"], "frac_2_t50": fr["frac_2_t50"], "frac_2_t60": fr["frac_2_t60"],
                           "frac_2_t70": fr["frac_2_t70"], "frac_2_t80": fr["frac_2_t80"],
                           "s_fc1": sh["s_fc1"], "s_fc2": sh["s_fc2"], "s_mlp": sh["s_mlp"]})
    d_plus = {l: _d_plus(A_layers[l]) for l in layers}
    for k in range(int(n_draws)):
        ops = {l: GR.random_operator_like(NULL_GROUP, {"d_plus": d_plus[l]}, d, null_seed(seed, k, l))[0]
               for l in layers}
        for b in blocks:
            pk = GR.random_perm_like(NULL_GROUP, [np.asarray(perms[b])],
                                     np.random.default_rng(null_seed(seed, k, 3000 + b)))[0]
            sh = GR.mlp_group_shares(f[b], GR.block_action(NULL_GROUP, [ops[b]], [ops[b + 1]], [pk]))
            draw_rows.append({"block": b + 1, "draw": k, "s_fc1": sh["s_fc1"], "s_fc2": sh["s_fc2"],
                              "s_mlp": sh["s_mlp"]})
    for row in block_rows:
        vals = [r for r in draw_rows if r["block"] == row["block"]]
        for key in ("s_fc1", "s_fc2", "s_mlp"):
            row[f"{key}_null_mean"], row[f"{key}_null_lo"], row[f"{key}_null_hi"] = _null_cols([r[key] for r in vals])
            nm = row[f"{key}_null_mean"]
            row[f"{key}_over_null"] = row[key] / nm if vals and nm > 0 else float("nan")
    return {"blocks": block_rows, "draws": draw_rows, "n_draws": int(n_draws), "seed": int(seed)}


def range_blocks(name: str, depth: int) -> list:
    """0-based blocks of a 1-based range such as "1-6", clipped to the model depth (a 4-block tiny model reads
    "1-6" as blocks 1 to 4)."""
    a, _, b = str(name).partition("-")
    lo, hi = int(a), int(b or a)
    return [x - 1 for x in range(lo, hi + 1) if 1 <= x <= int(depth)]


def range_summary(measure: dict, blocks) -> dict:
    """MUST-04 range numbers (scripts/measure_groups.py build_metrics) for 0-based `blocks`: twin fraction (mean
    over blocks of frac_2_t50), ratio to null (mean s_mlp / mean null-mean s_mlp) and the null's 2.5 and 97.5
    percentiles over draws of the range mean; plus the fc1 and fc2 ratios. Empty if a block is missing."""
    want = [int(b) + 1 for b in blocks]
    rows = {r["block"]: r for r in measure["blocks"]}
    if not want or any(b not in rows for b in want):
        return {}
    sub = [rows[b] for b in want]
    out = {"blocks": want, "n_blocks": len(sub)}
    for key in ("s_fc1", "s_fc2", "s_mlp"):
        s = float(np.mean([r[key] for r in sub]))
        nm = float(np.mean([r[f"{key}_null_mean"] for r in sub]))
        out[f"{key}_mean"], out[f"{key}_null_mean"] = s, nm
        out[f"{key}_ratio_to_null"] = s / nm if nm > 0 and math.isfinite(nm) else float("nan")
        per = {}
        for r in measure["draws"]:
            if r["block"] in want:
                per.setdefault(r["draw"], []).append(r[key])
        ns = GR.null_summary([float(np.mean(v)) for v in per.values() if len(v) == len(want)])
        out[f"{key}_null_lo"], out[f"{key}_null_hi"], out[f"{key}_null_n"] = ns["lo"], ns["hi"], ns["n"]
    out["ratio_to_null"] = out["s_mlp_ratio_to_null"]
    for key in ("frac_2", "frac_2_t50", "frac_2_t60", "frac_2_t70", "frac_2_t80"):
        out[key] = float(np.mean([r[key] for r in sub]))
    out["twin_fraction_t50"] = out["frac_2_t50"]
    out["n_pairs_mean"] = float(np.mean([r["n_pairs"] for r in sub]))
    return out


def null_row_summary(measure: dict, null_twin, blocks) -> dict:
    """The C4 row from the reference's null draws over 0-based `blocks`: per draw k, the ratio of its range-mean s_mlp
    to the mean of the other draws' range means (leave one out, so a draw is not compared with itself), and the
    twin fraction of its random pairings scored with the reference's correlations. Means and 2.5 and 97.5
    percentiles over draws."""
    want = [int(b) + 1 for b in blocks]
    per = {}
    for r in measure["draws"]:
        if r["block"] in want:
            per.setdefault(r["draw"], []).append(r["s_mlp"])
    draws = sorted(k for k, v in per.items() if len(v) == len(want))
    if len(draws) < 2:
        return {}
    s = np.array([np.mean(per[k]) for k in draws])
    loo = (s.sum() - s) / (len(s) - 1)
    ratios = s / loo
    out = {"blocks": want, "n_draws": len(draws), "ratio_to_null": float(ratios.mean()),
           "ratio_lo": float(np.quantile(ratios, 0.025)), "ratio_hi": float(np.quantile(ratios, 0.975)),
           "s_mlp_mean": float(s.mean()), "s_mlp_null_mean": float(s.mean())}
    if null_twin is not None:
        nt = np.asarray(null_twin)[np.asarray(draws)][:, [b - 1 for b in want]].mean(axis=1)
        out.update({"twin_fraction_t50": float(nt.mean()), "twin_lo": float(np.quantile(nt, 0.025)),
                    "twin_hi": float(np.quantile(nt, 0.975))})
    return out


@torch.no_grad()
def split_half_fit(adapter, images, device, control: Control, blocks, *, batch_size: int = 16, top_k: int = 10,
                   workers: int = 1, model: nn.Module | None = None) -> list:
    """Per-layer operators and twin matchings fitted separately on the two halves of the fit set (even and odd
    positions; C1 keeps its whole-set derangement), for `blocks` and their stream layers. One list entry per half:
    {"A": {layer: A}, "perms": {block: perm}, "scores": {block: S[i, perm[i]]}, "n_images"}.

    Why (a diagnostic, not part of rule A03-1): the twin matching maximises sum_i S[i, pi(i)], which for fc1 is
    about tr(W1^T P W1 K) with K the paired cross-moment, and the operator is the involution closest to that same
    K. Fitted on the same images, the two agree with each other through the weights even when K is pure sampling
    noise, so the breaking share of any fitted structure sits below the MUST-04 null (about 0.88 instead of 1 on
    the synthetic network of tests/test_fp_controls.py). Fitting the operator on one half and the pairing on the
    other removes the shared noise; structure that holds in both halves stays."""
    from .heal import exact_involution
    from .matching import match_many
    from .variants import fit_layer_operator, span_layers

    m = adapter.model if model is None else model
    blocks = sorted({int(b) for b in blocks})
    layers = span_layers(blocks)
    n = _check_images(adapter, images)
    out = []
    for pos in (np.arange(0, n, 2), np.arange(1, n, 2)):
        p = collect_control_pass(adapter, images, device, control, positions=pos, blocks=blocks, layers=layers,
                                 batch_size=batch_size, residual=True, mlp=True, model=m)
        C = p["residual"]["C"]
        A = {l: exact_involution(fit_layer_operator(C[i], True)[0]) for i, l in enumerate(p["layers"])}
        R = p.pop("R")
        matches = match_many([R[b] for b in blocks], top_k=top_k, workers=workers)
        out.append({"A": A, "perms": {b: np.asarray(mm.perm, dtype=np.int64) for b, mm in zip(blocks, matches)},
                    "scores": {b: np.asarray(mm.score, dtype=np.float64) for b, mm in zip(blocks, matches)},
                    "n_images": int(len(pos))})
        del R, p
    return out


def split_half_summary(model: nn.Module, halves: list, blocks, *, n_draws: int = 20, seed: int = 0) -> dict:
    """Split-half ratio to null over 0-based `blocks`: the MUST-04 measure of the structure (operators of half a,
    pairings of half b) for (a, b) = (0, 1) and (1, 0), each against its own null (d+ of half a's operators,
    cycle type of half b's pairings), averaged. twin_fraction_t50 is the mean of the two halves' own values."""
    out = {"blocks": [int(b) + 1 for b in blocks]}
    ratios, twins = [], []
    for a, b in ((0, 1), (1, 0)):
        ms = measure_structure(model, halves[a]["A"], halves[b]["perms"], halves[b]["scores"], blocks,
                               n_draws=n_draws, seed=seed)
        rs = range_summary(ms, blocks)
        out[f"ratio_A{a}_P{b}"] = rs["ratio_to_null"]
        out[f"s_fc1_ratio_A{a}_P{b}"] = rs["s_fc1_ratio_to_null"]
        out[f"s_fc2_ratio_A{a}_P{b}"] = rs["s_fc2_ratio_to_null"]
        ratios.append(rs["ratio_to_null"])
        twins.append(rs["twin_fraction_t50"])
    out["ratio_to_null"] = float(np.mean(ratios))
    out["twin_fraction_t50"] = float(np.mean(twins))
    out["n_images_halves"] = [int(hf["n_images"]) for hf in halves]
    return out


def agreement_with(st, ref, blocks, ref_scores=None) -> dict:
    """How far a control's structure follows the reference's: mean tr(A_l A_l^ref) / d over the stream layers of
    `blocks`, tr(A*_L A*_L^ref) / d of the span operators, the share of the reference's paired units that the
    control pairs the same way (pair_recovery) and, with ref_scores, the same over the reference's twins
    (S_ij >= 0.5; twin_recovery). Means over `blocks`."""
    layers = sorted({int(b) for b in blocks} | {int(b) + 1 for b in blocks})
    d = st.d
    op = float(np.mean([np.trace(np.asarray(st.A_layers[l]) @ np.asarray(ref.A_layers[l])) / d for l in layers]))
    L = st.depth
    span = float(np.trace(np.asarray(st.A_star[L]) @ np.asarray(ref.A_star[L])) / d)
    pr, tr = [], []
    for b in blocks:
        p, q = np.asarray(st.perms[b]), np.asarray(ref.perms[b])
        paired = q != np.arange(len(q))
        if paired.any():
            pr.append(float((p[paired] == q[paired]).mean()))
        if ref_scores is not None:
            twin = paired & (np.asarray(ref_scores[b]) >= TWIN_THRESHOLD)
            if twin.any():
                tr.append(float((p[twin] == q[twin]).mean()))
    return {"operator_agreement": op, "span_agreement": span,
            "pair_recovery": float(np.mean(pr)) if pr else float("nan"),
            "twin_recovery": float(np.mean(tr)) if tr else float("nan")}


def score_a03(summaries: dict, range_name: str = PRIMARY_RANGE) -> dict:
    """Rule A03-1 on the range summaries {row: range_summary(...)}: each of C1, C2 and C3 passes when its twin
    fraction is at most 0.05 and its ratio to null at least 0.9 (compared with a 1e-12 allowance, so float rounding
    at a bar cannot flip the verdict). A control without both numbers is PENDING; the rule passes only when all
    three pass."""
    def ok(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

    controls = {}
    for c in TESTED_CONTROLS:
        r = summaries.get(c) or {}
        tw, ra = r.get("twin_fraction_t50"), r.get("ratio_to_null")
        if ok(tw) and ok(ra):
            t_ok, r_ok = float(tw) <= A03_TWIN_MAX + 1e-12, float(ra) >= A03_RATIO_MIN - 1e-12
            status = "PASS" if (t_ok and r_ok) else "FAIL"
        else:
            t_ok = r_ok = None
            status = "PENDING"
        controls[c] = {"twin_fraction_t50": tw, "ratio_to_null": ra, "twin_ok": t_ok, "ratio_ok": r_ok,
                       "status": status}
    states = [v["status"] for v in controls.values()]
    overall = "FAIL" if "FAIL" in states else ("PENDING" if "PENDING" in states else "PASS")
    return {"id": "A03-1", "range": range_name, "twin_max": A03_TWIN_MAX, "ratio_min": A03_RATIO_MIN,
            "rule": (f"for each of C1, C2 and C3: twin fraction in blocks {range_name} <= {A03_TWIN_MAX} and ratio "
                     f"to null in blocks {range_name} >= {A03_RATIO_MIN}"),
            "controls": controls, "status": overall}


# ----------------------------------------------------------------------------------------------
# Zero-shot k-NN after full surgery
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def knn_after_surgery(adapter, structure, bank_loader, query_loader, device, num_classes: int, *,
                      method: str = "mirror_exact", seed: int = 0, amp: str | None = None,
                      components=("mlp", "attn")) -> tuple:
    """Full conversion with mirror_family.build_mirror_variant(method, all blocks, components, structure, seed)
    and k-NN (mirror.knn.evaluate_knn, k = 20, T = 0.07). The conversion is always the left-right Mirror-exact
    construction; only the structure (A*, pairings, head maps, register pairing) comes from the row.
    Returns (row dict, predictions)."""
    from .knn import evaluate_knn
    from .mirror_family import build_mirror_variant

    L = adapter.depth
    model, info = build_mirror_variant(adapter, method, list(range(L)), components, structure, seed=seed)
    r = evaluate_knn(adapter, bank_loader, query_loader, device, num_classes, k=20, temperature=0.07, model=model,
                     amp=amp)
    row = {"method": method, "seed": int(seed), "top1": float(r["top1"]),
           "flip_consistency": float(r["flip_consistency"]), "n_bank": int(r["n_bank"]),
           "n_query": int(r["n_query"]), "params_matrices": int(info["params_matrices"]),
           "params_total": int(info["params_total"]), "budget_total": int(info["budget_total"]),
           "d_plus": info.get("build", {}).get("d_plus"), "amp": amp or "none"}
    del model
    return row, r["pred"].numpy()


# ----------------------------------------------------------------------------------------------
# Synthetic equivariant network (CPU tests and --dry-run)
# ----------------------------------------------------------------------------------------------
class TinyTimmAdapter(TimmTinyAdapter):
    """A tiny random timm VisionTransformer (patch 14) with chosen width, depth, heads and registers, behind the
    TimmTinyAdapter interface (affines, biases and LayerScale randomised as there)."""

    def __init__(self, img_size: int = 56, embed_dim: int = 32, depth: int = 3, num_heads: int = 2,
                 reg_tokens: int = 0, seed: int = 0, randomise: bool = True, dtype: torch.dtype = torch.float64):
        from timm.models.vision_transformer import VisionTransformer

        from .adapters import _randomise_affine

        torch.manual_seed(seed)
        model = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=embed_dim, depth=depth,
                                  num_heads=num_heads, init_values=1e-5, reg_tokens=reg_tokens)
        if randomise:
            _randomise_affine(model, seed)
        ViTAdapter.__init__(self, model.to(dtype), img_size=img_size, patch_size=14,
                            n_prefix_tokens=int(model.num_prefix_tokens))


def tiny_adapter(img_size: int = 56, embed_dim: int = 32, depth: int = 3, num_heads: int = 2, reg_tokens: int = 2,
                 seed: int = 0, dtype: torch.dtype = torch.float64) -> TinyTimmAdapter:
    return TinyTimmAdapter(img_size=img_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                           reg_tokens=reg_tokens, seed=seed, dtype=dtype)


def with_model(adapter, model: nn.Module):
    """A shallow copy of `adapter` that wraps `model` (frozen, eval mode)."""
    new = copy.copy(adapter)
    new.model = model.eval()
    for p in new.model.parameters():
        p.requires_grad_(False)
    return new


@torch.no_grad()
def unfold_rmsnorm(model: nn.Module) -> nn.Module:
    """In place: every fold.RMSNorm of norm1 and norm2 becomes nn.LayerNorm(weight 1, bias 0, same eps), and a final
    norm Sequential(rms, [even], affine) becomes nn.LayerNorm with the affine as its weight and bias (an even
    head is dropped). Exact on the folded model's channel-centred stream, where RMSNorm equals LayerNorm without
    affine. The result is a standard pre-norm ViT again, so fold_layernorm_full and build_mirror_variant accept it.
    A stream operator A with A 1 = 1 commutes with LayerNorm without affine, so stream equivariance is kept."""
    from .fold import RMSNorm

    def ln(dim, eps, like):
        out = nn.LayerNorm(int(dim), eps=float(eps))
        return out.to(dtype=like.dtype, device=like.device)

    ref = next(model.parameters())
    for blk in model.blocks:
        for name in ("norm1", "norm2"):
            mod = getattr(blk, name)
            if isinstance(mod, RMSNorm):
                setattr(blk, name, ln(mod.dim, mod.eps, ref))
    norm = getattr(model, "norm", None)
    if isinstance(norm, nn.Sequential) and hasattr(norm, "rms") and hasattr(norm, "affine"):
        new = ln(norm.rms.dim, norm.rms.eps, norm.affine.weight)
        new.weight.copy_(norm.affine.weight)
        new.bias.copy_(norm.affine.bias)
        model.norm = new
    elif isinstance(norm, RMSNorm):
        model.norm = ln(norm.dim, norm.eps, ref)
    return model


@torch.no_grad()
def make_equivariant_network(adapter, images, device, *, batch_size: int = 16, top_k: int = 10,
                             register_margin: float = 1e9, model_name: str = "tiny") -> tuple:
    """A copy of the adapter's network made exactly mirror-equivariant with the existing conversion: the structure
    is fitted on `images` (mirror_family.fit_span_structure, real mirror), the model converted with
    build_mirror_variant("mirror_exact", every block, MLP and attention) and unfolded back to LayerNorm
    (unfold_rmsnorm), so the aligned tokens of mirror(x) equal the tokens of x times A* = A_star[L] at every stream
    layer and every MLP satisfies P W1 = W1 A*, P b1 = b1. The large register margin keeps every register even,
    because the paired statistics never swap registers. Returns (adapter of the equivariant network, the planted
    SpanStructure, the conversion info)."""
    from .data import make_loader
    from .mirror_family import build_mirror_variant, fit_span_structure

    n = int(len(images))
    loader = make_loader(images, np.zeros(n, dtype=np.int64), np.arange(n), int(batch_size))
    st = fit_span_structure(adapter, loader, device, model_name=model_name, fit_seed=0, top_k=top_k, workers=1,
                            register_margin=register_margin, pairings=False)
    L = adapter.depth
    model, info = build_mirror_variant(adapter, "mirror_exact", list(range(L)), ("mlp", "attn"), st)
    return with_model(adapter, unfold_rmsnorm(model)), st, info

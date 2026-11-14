"""Mirror-family methods for MUST-07 and MUST-08 (PROJECT_NEXT_PHASE.md, Sections 4.5, 4.7 and 5).

One entry point, `build_mirror_variant(adapter, method, blocks, components, span_struct, seed)`, returns a
converted deep copy of the model and an `info` dict with the same parameter fields as the compressors of
mirror/baselines.py (method, hparams, params_matrices, params_basis_change, params_total, budget_total, gap,
per_matrix, tensors), so the runner (scripts/must07_compressors.py) treats every method alike.

Blocks are 0-based code indices ("block 1" of the plan is index 0). Spans are contiguous and, in MUST-07 and
MUST-08, always start at index 0, so the embeddings belong to the span. "Full" means every block; then the
even head y -> y (I + A*)/2 is installed and the model is exactly flip-invariant (mirror.m1.build_vexact).

Methods (all except mirror_local share the Mirror-exact machinery: one span operator A*, the model fully
LayerNorm-folded, embeddings projected, MLP and attention projected with A_in = A_out = A*):
  mirror_exact      the fitted A* (layers 0..k, each normalised to equal energy; m1.fit_span_operator), the
                    fitted mirror twin pairing, the fitted head maps and register pairing.
  mirror_local      per-layer operators (the V-local ablation): fc1 with A_b, fc2 with A_(b+1), attention with
                    A_in = A_out = A_b. The model is not folded and the embeddings are not projected (as
                    scripts/m1_attention.py "vlocal"). Attention uses A_out = A_b (M1 chose A_(b+1) by a
                    0.001 lower breaking share) so that each block needs exactly one stream rotation, the
                    accounting of Section 3 ("one d x d basis change per block"): k d^2 parameters counted.
  pair_actcorr      the MLP pairing from same-image activation correlation: corr(z_i, z_j) of fc1 outputs on the
                    same tokens of the fit images, diagonal set to 0 (otherwise every gain 2 S_ij - S_ii - S_jj
                    is negative), matched with mirror.matching (max-weight involution, top_k candidates), then
                    cut or filled greedily to the mirror pairing's number of pairs per block. Tied with A*.
  pair_wcos         the MLP pairing from weight cosine of the folded fc1 rows of the original model
                    (variants.generic_pairings), the mirror pairing's number of pairs per block. Tied with A*.
  pair_random       a random MLP pairing with the mirror pairing's numbers of pairs and self-twins (the same
                    draw as random_structure's pairing, m1.matched_random_span_structure). Tied with A*.
  random_structure  m1.matched_random_span_structure: a random involution with the d+ and d- of A* (A 1 = 1),
                    random pairing, random head maps like the fitted ones and a random register pairing.
For the three pairing controls only the MLP neuron pairing changes; the attention (when converted) keeps the
fitted head maps under A*, so the controls isolate the neuron pairing. All five A*-type methods keep exactly
the same free-parameter counts per matrix as mirror_exact (the counts depend only on tr(A*) and on the numbers
of pairs, self-heads and self-head traces), so their gap to the budget is 0.

Parameter accounting (Section 4.7; PREREG_NEXT_PHASE.md Section 9 item 7):
  params_matrices      free parameters of the converted matrices (fc1, fc2 per block for "mlp"; Wq, Wk, Wv, Wo
                       for "attn"): surgery.equivariant_param_count and attention.attention_free_params, the
                       multiply-adds of the even/odd form (mirror/efficient.py).
  params_basis_change  A*-type: d^2 for each end of the span that is not a model boundary (span starting at
                       block 0: one end below the full model, none at the full model); mirror_local: one d^2
                       rotation per converted block (the last one rotates straight back to the standard basis).
  params_total         params_matrices + params_basis_change; params_total_absorbed: the same with Mirror-exact's
                       span-end change absorbed into the next dense block (reported, PREREG item 7).
  budget_total         Mirror-exact's own params_total for the same model, span and components: the budget every
                       compressor matches (mirror.budget.mirror_exact_budget, or the local shim below).
  gap                  params_total - budget_total (absolute); gap_rel = gap / budget_total.
Biases, norms and embeddings are counted as dense everywhere (they are not part of the budget).

Structure (`SpanStructure`, fitted once per model and fit seed by `fit_span_structure`, cached by the runner):
  pass 1  variants.fit_structure: per-layer operators A_l and the mirror twin pairings from one paired pass
          (images and their mirrors), plus C_l and G_l; or an existing run_e1.py operator file.
  A*      m1.fit_span_operator(C, layers 0..k, G) for every k = 1..L.
  pass 2  attention.collect_attn_stats on the fit images: head maps (attention.pair_and_fit_heads, as M1), the
          register alignment scores for every A* (m1.RegisterScores, matched with m1.match_registers) and the
          same-image fc1 correlation (SameImageStats) for pair_actcorr.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .attention import HeadMaps, apply_attention_surgery, attention_free_params, attn_dims, collect_attn_stats, \
    pair_and_fit_heads
from .heal import exact_involution
from .m1 import (RegisterScores, build_vexact, fit_span_operator, involution_trace64, match_registers,
                 matched_random_span_structure, register_info, vexact_components)
from .matching import match_many, symmetrise
from .stats import _flat64
from .surgery import apply_surgery_two, equivariant_param_count, fold_mlp, perm_trace

MIRROR_METHODS = ("mirror_exact", "mirror_local", "pair_actcorr", "pair_wcos", "pair_random", "random_structure")
EXACT_FAMILY = ("mirror_exact", "pair_actcorr", "pair_wcos", "pair_random", "random_structure")
COMPONENTS = ("mlp", "attn")
STRUCTURE_FORMAT = 1
ATTN_MATRICES = ("Wq", "Wk", "Wv", "Wo")

__all__ = ["MIRROR_METHODS", "EXACT_FAMILY", "COMPONENTS", "normalise_components", "SpanStructure", "SameImageStats",
           "fit_span_structure", "save_span_structure", "load_span_structure", "top_pairs", "exact_per_matrix",
           "local_per_matrix", "n_span_ends", "budget_api", "span_budget", "budget_fields", "build_mirror_variant",
           "ShimMatrixBudget", "ShimSpanBudget", "shim_mirror_exact_budget"]


def normalise_components(components) -> tuple:
    """("mlp",), ("attn",) or ("mlp", "attn") from a sequence or a string ("mlp", "attn", "both", "mlp+attn")."""
    if isinstance(components, str) and components.strip() == "both":
        return COMPONENTS
    return vexact_components(components)


def _log(msg: str) -> None:
    from .utils import log

    log(msg)


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------
@dataclass
class SpanStructure:
    """Everything the Mirror family needs for one model and fit seed (blocks indexed 0..L-1)."""

    model: str
    fit_seed: int
    A_layers: list                 # L + 1 exact involutions; layer 0 = tokens entering block 0
    perms: list                    # L mirror twin pairings (involutions)
    heads: list                    # L HeadMaps (fitted, before the twin gauge)
    A_star: dict                   # k -> span operator for blocks 0..k-1 (stream layers 0..k), k = 1..L
    reg_perm: dict                 # k -> register involution under A_star[k] (length R, empty without registers)
    actcorr_perms: list | None = None
    wcos_perms: list | None = None
    meta: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.perms)

    @property
    def d(self) -> int:
        return int(np.asarray(self.A_layers[0]).shape[0])

    @property
    def h(self) -> int:
        return int(len(self.perms[0]))

    def n_pairs(self, b: int) -> int:
        p = np.asarray(self.perms[b])
        return int((p != np.arange(len(p))).sum() // 2)

    def exact_parts(self, method: str, k: int, seed: int = 0) -> dict:
        """Operator, pairings (blocks 0..k-1), head maps (blocks 0..k-1) and register pairing of an A*-type
        method at span size k."""
        if method not in EXACT_FAMILY:
            raise ValueError(f"{method!r} is not an A*-type method ({EXACT_FAMILY})")
        k = int(k)
        A, reg = self.A_star[k], np.asarray(self.reg_perm[k], dtype=np.int64)
        perms, heads = list(self.perms), list(self.heads)
        info = {"operator": f"fitted span operator A* over stream layers 0..{k}", "pairing": "fitted mirror twins",
                "head_maps": "fitted", "register_pairing": "fitted under A*"}
        if method == "pair_actcorr":
            if self.actcorr_perms is None:
                raise ValueError("this structure has no same-image correlation pairing (fit with pairings=True)")
            perms = list(self.actcorr_perms)
            info["pairing"] = "same-image fc1 correlation, matched pair count"
        elif method == "pair_wcos":
            if self.wcos_perms is None:
                raise ValueError("this structure has no weight-cosine pairing (fit with pairings=True)")
            perms = list(self.wcos_perms)
            info["pairing"] = "weight cosine of folded fc1 rows, matched pair count"
        elif method in ("pair_random", "random_structure"):
            r = matched_random_span_structure(A, self.perms, self.heads, reg, int(seed))
            perms = list(r["perms"])
            info["pairing"] = f"random, matched numbers of pairs and self-twins (seed {int(seed)})"
            if method == "random_structure":
                A, heads, reg = r["A_star"], list(r["heads"]), np.asarray(r["reg_perm"], dtype=np.int64)
                info.update({"operator": f"random involution with the d+ of A*, A 1 = 1 (seed {int(seed)})",
                             "head_maps": "random, matched pairs, self-heads and self-head traces",
                             "register_pairing": "random, matched number of pairs"})
        return {"A": exact_involution(A), "perms": perms[:k], "heads": heads[:k], "reg_perm": reg, "info": info}


class SameImageStats:
    """Per block: sums, squared sums and the same-image cross-moment sum Z^T Z (h x h) of fc1 outputs over the
    tokens of the original images (the first half of a paired batch). Use as fc1_cb of a paired pass."""

    def __init__(self, n_blocks: int, h: int, device):
        kw = dict(dtype=torch.float64, device=device)
        self.n = torch.zeros(n_blocks, **kw)
        self.s = torch.zeros(n_blocks, h, **kw)
        self.s2 = torch.zeros(n_blocks, h, **kw)
        self.cross = torch.zeros(n_blocks, h, h, **kw)

    def update(self, block: int, z: torch.Tensor, zt: torch.Tensor | None = None) -> None:
        Z = _flat64(z).to(self.cross.device)
        self.n[block] += Z.shape[0]
        self.s[block] += Z.sum(0)
        self.s2[block] += (Z * Z).sum(0)
        self.cross[block] += Z.T @ Z

    def correlation(self, block: int, zero_diagonal: bool = True, eps: float = 1e-12) -> np.ndarray:
        """R[i, j] = corr(z_i, z_j) on the same tokens; the diagonal is set to 0 by default (see the module
        docstring, pair_actcorr)."""
        n = self.n[block]
        m = self.s[block] / n
        v = (self.s2[block] / n - m * m).clamp_min(0)
        cov = self.cross[block] / n - torch.outer(m, m)
        den = torch.sqrt(torch.outer(v, v))
        R = torch.where(den > eps, cov / den.clamp_min(eps), torch.zeros_like(cov)).clamp(-1.0, 1.0)
        R = R.cpu().numpy()
        if zero_diagonal:
            np.fill_diagonal(R, 0.0)
        return R


def top_pairs(pairs, S: np.ndarray, n_pairs: int) -> list:
    """Exactly n_pairs disjoint pairs: the n strongest of `pairs` by S, filled greedily (largest S among unused
    neurons) when the matching has fewer."""
    S = np.asarray(S, dtype=np.float64)
    pairs = [tuple(sorted(map(int, p))) for p in pairs]
    w = np.array([S[a, b] for a, b in pairs]) if pairs else np.zeros(0)
    order = np.argsort(-w, kind="stable")
    if len(pairs) >= n_pairs:
        return sorted(pairs[i] for i in order[:n_pairs])
    out = list(pairs)
    used = {x for p in pairs for x in p}
    free = np.array([i for i in range(S.shape[0]) if i not in used], dtype=np.int64)
    if len(free) >= 2:
        sub = S[np.ix_(free, free)]
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


def _perm_from_pairs(h: int, pairs) -> np.ndarray:
    perm = np.arange(h)
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


@torch.no_grad()
def fit_span_structure(adapter, loader, device, *, model_name: str = "", fit_seed: int = 0, top_k: int = 10,
                       workers: int = 1, register_margin: float = 0.0, base=None, pairings: bool = True) -> SpanStructure:
    """Fit the structure of one model on one fit set (see the module docstring). `base` is an existing
    variants.Structure (per-layer operators, twin pairings, C and G) that replaces pass 1."""
    from .variants import fit_structure, generic_pairings

    t0 = time.time()
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    source = "variants.fit_structure (pass 1 on the fit set)"
    if base is None:
        base = fit_structure(adapter, loader, device, top_k=top_k, workers=workers, center=True)
    else:
        source = f"operator file {base.source}"
    if base.C is None or base.G is None:
        raise ValueError("the structure needs C and G to fit the span operators")
    if (base.depth, base.d, base.h) != (L, d, h):
        raise ValueError(f"structure shape {(base.depth, base.d, base.h)} does not match the model {(L, d, h)}")
    A_star, star_info = {}, {}
    for k in range(1, L + 1):
        A, inf = fit_span_operator(base.C, range(k + 1), base.G)
        A_star[k] = exact_involution(A)
        star_info[k] = {"d_plus": inf["d_plus"], "d_minus": inf["d_minus"], "n_zero": inf["n_zero"]}
    reg = register_info(adapter.model, adapter)
    rs = RegisterScores({k: (A_star[k], k) for k in A_star}, L + 1, reg["R"], reg["start"], device) if reg["R"] else None
    same = SameImageStats(L, h, device) if pairings else None
    ast = collect_attn_stats(adapter, loader, device, residual_cb=rs.update if rs is not None else None,
                             fc1_cb=same.update if same is not None else None)
    heads = []
    for b in range(L):
        hm = pair_and_fit_heads(ast.moments(b))
        hm.check(tol=1e-8)
        heads.append(hm)
    del ast
    reg_perm = {k: (match_registers(rs.cosines(k, range(k + 1)), register_margin) if rs is not None
                    else np.zeros(0, dtype=np.int64)) for k in A_star}
    perms = [np.asarray(p, dtype=np.int64) for p in base.perms]
    n_pairs = [int((p != np.arange(h)).sum() // 2) for p in perms]
    actcorr = wcos = None
    if pairings:
        R_list = [same.correlation(b) for b in range(L)]
        del same                                   # free the GPU accumulators before the CPU matchings
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        matches = match_many(R_list, top_k=top_k, workers=workers)
        actcorr = [_perm_from_pairs(h, top_pairs(m.pairs, symmetrise(R), n)) for m, R, n in zip(matches, R_list, n_pairs)]
        del R_list
        W1 = [fold_mlp(adapter.model.blocks[b])["W1"] for b in range(L)]
        wcos = [np.asarray(p, dtype=np.int64) for p in generic_pairings(W1, n_pairs, top_k=top_k, workers=workers)]
    meta = {"model": model_name, "fit_seed": int(fit_seed), "top_k": int(top_k), "register_margin": float(register_margin),
            "pass1": source, "n_pairs": n_pairs, "n_self_heads": [hm.n_self for hm in heads],
            "n_head_pairs": [hm.n_pairs for hm in heads], "n_registers": int(reg["R"]),
            "register_pairs": {int(k): int((v != np.arange(len(v))).sum() // 2) for k, v in reg_perm.items()},
            "A_star": {int(k): v for k, v in star_info.items()}, "d_plus_layers": [int(x) for x in base.d_plus],
            "base_meta": {k: v for k, v in (base.meta or {}).items() if k in ("n_tokens", "top_k", "center", "fit_seed")},
            "pairings_fitted": bool(pairings), "fit_seconds": round(time.time() - t0, 1)}
    return SpanStructure(model=model_name, fit_seed=int(fit_seed), A_layers=[exact_involution(A) for A in base.A_layers],
                         perms=perms, heads=heads, A_star=A_star, reg_perm=reg_perm,
                         actcorr_perms=actcorr, wcos_perms=wcos, meta=meta)


def save_span_structure(path, st: SpanStructure) -> Path:
    """npz with every array in float64 / int64 and the meta as JSON. Load with load_span_structure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    L = st.depth
    R = len(st.reg_perm[L])
    arrays = {
        "format": np.array(STRUCTURE_FORMAT), "model": np.array(st.model), "fit_seed": np.array(st.fit_seed),
        "A_layers": np.stack([np.asarray(A, dtype=np.float64) for A in st.A_layers]),
        "perms": np.stack(st.perms).astype(np.int64),
        "sigma": np.stack([hm.sigma for hm in st.heads]).astype(np.int64),
        "B": np.stack([hm.B for hm in st.heads]), "C": np.stack([hm.C for hm in st.heads]),
        "head_scores": np.stack([hm.scores if hm.scores is not None else np.zeros((hm.H, hm.H)) for hm in st.heads]),
        "A_star": np.stack([np.asarray(st.A_star[k], dtype=np.float64) for k in range(1, L + 1)]),
        "reg_perm": (np.stack([np.asarray(st.reg_perm[k], dtype=np.int64) for k in range(1, L + 1)])
                     if R else np.zeros((L, 0), dtype=np.int64)),
        "meta_json": np.array(json.dumps(st.meta, default=str)),
    }
    if st.actcorr_perms is not None:
        arrays["actcorr"] = np.stack(st.actcorr_perms).astype(np.int64)
    if st.wcos_perms is not None:
        arrays["wcos"] = np.stack(st.wcos_perms).astype(np.int64)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def load_span_structure(path) -> SpanStructure:
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    if int(f["format"]) != STRUCTURE_FORMAT:
        raise ValueError(f"{path}: structure format {int(f['format'])}, expected {STRUCTURE_FORMAT}")
    L = len(f["perms"])
    heads = [HeadMaps(sigma=f["sigma"][b], B=f["B"][b], C=f["C"][b], scores=f["head_scores"][b]) for b in range(L)]
    return SpanStructure(
        model=str(f["model"]), fit_seed=int(f["fit_seed"]), A_layers=list(f["A_layers"]),
        perms=[np.asarray(p, dtype=np.int64) for p in f["perms"]], heads=heads,
        A_star={k: f["A_star"][k - 1] for k in range(1, L + 1)},
        reg_perm={k: np.asarray(f["reg_perm"][k - 1], dtype=np.int64) for k in range(1, L + 1)},
        actcorr_perms=[np.asarray(p, dtype=np.int64) for p in f["actcorr"]] if "actcorr" in f else None,
        wcos_perms=[np.asarray(p, dtype=np.int64) for p in f["wcos"]] if "wcos" in f else None,
        meta=json.loads(str(f["meta_json"])))


# ----------------------------------------------------------------------------------------------
# Parameter accounting
# ----------------------------------------------------------------------------------------------
def _dims(model: nn.Module) -> tuple:
    blk = model.blocks[0]
    d, h = int(blk.mlp.fc1.in_features), int(blk.mlp.fc1.out_features)
    H, dh = attn_dims(blk)
    return d, h, H * dh, len(model.blocks)


def _row(block, name, shape, kept) -> dict:
    return {"block": int(block), "name": name, "shape": [int(s) for s in shape], "dense": int(shape[0] * shape[1]),
            "kept": int(kept)}


def exact_per_matrix(d: int, h: int, D: int, A, blocks, perms, heads, components) -> list:
    """Per converted matrix of an A*-type method: free parameters with A_in = A_out = A (the accounting of
    heal_vexact.vexact_block_params, split by component)."""
    comps = normalise_components(components)
    trA = involution_trace64(A)
    rows = []
    for i, b in enumerate(blocks):
        if "mlp" in comps:
            fc = equivariant_param_count(h, d, perm_trace(perms[i]), trA)
            rows += [_row(b, "fc1", (h, d), fc), _row(b, "fc2", (d, h), fc)]
        if "attn" in comps:
            att = attention_free_params(d, heads[i], A, A)
            rows += [_row(b, n, (D, d) if n != "Wo" else (d, D), att[n]) for n in ATTN_MATRICES]
    return rows


def local_per_matrix(d: int, h: int, D: int, A_layers, blocks, perms, heads, components) -> list:
    """Per converted matrix of Mirror-local: fc1 with A_b, fc2 with A_(b+1), attention with A_in = A_out = A_b."""
    comps = normalise_components(components)
    rows = []
    for b in blocks:
        if "mlp" in comps:
            tp = perm_trace(perms[b])
            rows += [_row(b, "fc1", (h, d), equivariant_param_count(h, d, tp, involution_trace64(A_layers[b]))),
                     _row(b, "fc2", (d, h), equivariant_param_count(h, d, tp, involution_trace64(A_layers[b + 1])))]
        if "attn" in comps:
            att = attention_free_params(d, heads[b], A_layers[b], A_layers[b])
            rows += [_row(b, n, (D, d) if n != "Wo" else (d, D), att[n]) for n in ATTN_MATRICES]
    return rows


def n_span_ends(blocks, depth: int) -> int:
    """Ends of a contiguous span that are not a model boundary (block 0 starts at the embeddings)."""
    blocks = [int(b) for b in blocks]
    return int(blocks[0] > 0) + int(blocks[-1] < int(depth) - 1)


# ---- SHIM (remove at integration): stands in for mirror/budget.py until it exists -----------------
@dataclass
class ShimMatrixBudget:
    block: int
    name: str
    shape: tuple
    dense: int
    mirror_exact: int


@dataclass
class ShimSpanBudget:
    model: str
    blocks: list
    components: tuple
    matrices: list
    basis_change: int
    total: int
    biases_dense: int
    notes: str = ""


def shim_mirror_exact_budget(model, A_star, blocks, perms, head_maps, components=COMPONENTS,
                             model_name: str = "") -> ShimSpanBudget:
    """SHIM of mirror.budget.mirror_exact_budget (same fields): Mirror-exact's matrix counts plus d^2 per span
    end that is not a model boundary. `perms` and `head_maps` are aligned with `blocks`."""
    comps = normalise_components(components)
    d, h, D, L = _dims(model)
    blocks = [int(b) for b in blocks]
    rows = exact_per_matrix(d, h, D, A_star, blocks, perms, head_maps, comps)
    mats = [ShimMatrixBudget(r["block"], r["name"], tuple(r["shape"]), r["dense"], r["kept"]) for r in rows]
    basis = n_span_ends(blocks, L) * d * d
    biases = len(blocks) * ((h + d if "mlp" in comps else 0) + (3 * D + d if "attn" in comps else 0))
    return ShimSpanBudget(model=model_name, blocks=blocks, components=comps, matrices=mats, basis_change=int(basis),
                          total=int(sum(m.mirror_exact for m in mats) + basis), biases_dense=int(biases),
                          notes="SHIM (mirror/mirror_family.py) until mirror/budget.py exists")
# ---- end SHIM -------------------------------------------------------------------------------------


def budget_api():
    """("real", mirror.budget.mirror_exact_budget) when mirror/budget.py exists, else ("shim", the local shim)."""
    try:
        from .budget import mirror_exact_budget
    except ImportError:
        return "shim", shim_mirror_exact_budget
    return "real", mirror_exact_budget


def span_budget(model, struct: SpanStructure, blocks, components, model_name: str = ""):
    """Mirror-exact's budget for `blocks` (span from block 0) and `components` with the fitted structure."""
    blocks = [int(b) for b in blocks]
    comps = normalise_components(components)
    k = len(blocks)
    if blocks != list(range(k)):
        raise ValueError("MUST-07 spans start at block 0")
    parts = struct.exact_parts("mirror_exact", k)
    _, fn = budget_api()
    return fn(model, parts["A"], blocks, parts["perms"], parts["heads"], components=comps, model_name=model_name)


def budget_fields(budget) -> dict:
    """The numbers the runner needs from a SpanBudget (real or shim)."""
    mats = list(budget.matrices)
    return {"budget_total": int(budget.total), "budget_basis_change": int(budget.basis_change),
            "budget_matrices": int(sum(int(m.mirror_exact) for m in mats)),
            "span_dense": int(sum(int(m.dense) for m in mats)), "n_matrices": len(mats)}


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_mirror_variant(adapter, method: str, blocks, components, span_struct: SpanStructure, seed: int = 0,
                         budget=None):
    """Converted deep copy of adapter.model and its info (see the module docstring)."""
    if method not in MIRROR_METHODS:
        raise ValueError(f"unknown Mirror-family method {method!r}; choose from {MIRROR_METHODS}")
    comps = normalise_components(components)
    blocks = [int(b) for b in blocks]
    L = adapter.depth
    k = len(blocks)
    if blocks != list(range(k)) or not 1 <= k <= L:
        raise ValueError("spans start at block 0 and are contiguous")
    if span_struct.depth != L:
        raise ValueError(f"structure has {span_struct.depth} blocks, the model {L}")
    d, h, D, _ = _dims(adapter.model)
    full = k == L
    if budget is None:
        budget = span_budget(adapter.model, span_struct, blocks, comps)
    bf = budget_fields(budget)
    build = {}
    if method == "mirror_local":
        A_l, perms, heads = span_struct.A_layers, span_struct.perms, span_struct.heads
        model = copy.deepcopy(adapter.model)
        if "mlp" in comps:
            apply_surgery_two(model, blocks, [A_l[b] for b in blocks], [A_l[b + 1] for b in blocks],
                              [perms[b] for b in blocks])
        maps_after = None
        if "attn" in comps:
            maps_after = apply_attention_surgery(model, blocks, [A_l[b] for b in blocks], [A_l[b] for b in blocks],
                                                 [heads[b] for b in blocks])
        model.eval()
        rows = local_per_matrix(d, h, D, A_l, blocks, perms, heads, comps)
        basis = k * d * d
        basis_absorbed = basis
        structure = {"A_layers": A_l, "perms": [perms[b] for b in blocks], "heads": [heads[b] for b in blocks],
                     "heads_after": maps_after, "attn_A_out": "A_b"}
        sinfo = {"operator": "per-layer A_l (fc1 A_b, fc2 A_(b+1), attention A_b)", "pairing": "fitted mirror twins",
                 "head_maps": "fitted", "embeddings": "not projected", "even_head": False}
    else:
        parts = span_struct.exact_parts(method, k, seed)
        model, binfo = build_vexact(adapter, parts["A"], blocks, parts["perms"], parts["heads"],
                                    reg_perm=parts["reg_perm"], even_head=full, components=comps)
        rows = exact_per_matrix(d, h, D, parts["A"], blocks, parts["perms"], parts["heads"], comps)
        own_rule = n_span_ends(blocks, L) * d * d
        basis = bf["budget_basis_change"]              # Mirror-exact's own count is the budget (same span ends)
        basis_absorbed = 0
        build = {"embedding_shares": binfo["embedding_shares"], "pos_materialised": binfo["pos_materialised"],
                 "even_head": binfo["even_head"], "d_plus": binfo["d_plus"],
                 "basis_change_local_rule": int(own_rule), "basis_change_from_budget": int(basis)}
        structure = {"A": parts["A"], "perms": parts["perms"], "heads": parts["heads"], "reg_perm": parts["reg_perm"],
                     "heads_after": binfo["maps_after"]}
        sinfo = dict(parts["info"])
    params_matrices = int(sum(r["kept"] for r in rows))
    params_total = params_matrices + int(basis)
    span_dense = int(sum(r["dense"] for r in rows))
    if method == "mirror_exact" and params_matrices != bf["budget_matrices"]:
        raise AssertionError(f"Mirror-exact matrices {params_matrices} differ from the budget's {bf['budget_matrices']}")
    gap = params_total - bf["budget_total"]
    info = {
        "method": method, "hparams": {}, "components": list(comps), "blocks": blocks, "k": k, "full": full,
        "params_matrices": params_matrices, "params_basis_change": int(basis), "params_total": int(params_total),
        "params_total_absorbed": int(params_matrices + basis_absorbed), "budget_total": bf["budget_total"],
        "gap": int(gap), "gap_rel": gap / bf["budget_total"] if bf["budget_total"] else float("nan"),
        "span_dense": span_dense, "per_matrix": rows, "tensors": {}, "structure": structure,
        "structure_info": sinfo, "build": build, "seed": int(seed),
        "budget_source": budget_api()[0],
    }
    return model, info


def describe_info(info: dict) -> dict:
    """JSON-friendly copy of a variant info (drops tensors, structure objects and per-matrix rows)."""
    skip = {"tensors", "structure", "per_matrix"}
    out = {}
    for k, v in info.items():
        if k in skip:
            continue
        try:
            json.dumps(v, default=float)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


def as_namespace(budget) -> SimpleNamespace:
    """A plain namespace view of a SpanBudget (debugging and tests)."""
    return SimpleNamespace(**{k: getattr(budget, k) for k in ("model", "blocks", "components", "matrices",
                                                              "basis_change", "total", "biases_dense", "notes")
                              if hasattr(budget, k)})

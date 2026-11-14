"""Matched parameter budgets for the strong-compressor comparison (PROJECT_NEXT_PHASE.md, Sections 4.5 and 4.7;
MUST-07 zero-shot, MUST-09 healed).

Rule (Section 4.7). Every compressor matches Mirror-exact's parameter count over the same converted matrices to
within 1%, with all basis changes counted. Where a method cannot hit the budget exactly it rounds toward giving the
baseline more parameters, and the gap is printed:
    gap = (params_total - budget_total) / budget_total,   0 <= gap <= 0.01 where reachable,
with params_total = the method's matrix parameters + its own basis-change parameters, and budget_total = Mirror-exact's
matrix parameters + Mirror-exact's basis-change parameters (SpanBudget.total).

Conventions. Blocks are 0-based in code (the plan's "block 1" is index 0). Matrices are torch Linear weights
[out, in] on folded weights (norm affine and LayerScale folded, surgery.fold_mlp and attention.fold_attn):
    fc1 [h, d], fc2 [d, h], wq, wk, wv [D, d], wo [d, D]   (D = heads x head dim).

What is counted.
  * Matrices: the converted matrices of the chosen components ("mlp": fc1, fc2; "attn": wq, wk, wv, wo) in the chosen
    blocks. Mirror-exact's count per matrix is the dimension of its equivariant subspace with one operator A* for the
    whole span (A_in = A_out = A*): surgery.equivariant_param_count for fc1 and fc2 and
    attention.attention_free_params for wq, wk, wv, wo. These are the numbers heal_vexact.vexact_block_params gives,
    so the span sum equals heal_vexact.vexact_param_counts (tests/test_budget.py checks it).
  * Basis changes (d x d each):
      Mirror-exact  one at a span end that is not the model boundary. A span starting at block 0 has none at the start
                    (the patch embedding is in the span, so the rotation is folded into it); a span ending at the last
                    block has none at the end (the final features are read through the even head, which is not
                    counted, as decided in the plan); the full model has none.
      Mirror-local  one per converted block (per-layer operators; the "counting error" row of Section 3).
      Baselines     whatever the method itself needs; see `allocate` (only SliceGPT has any).
  * Biases: dense and counted the same way for every method (the original bias count of the converted matrices),
    reported separately in SpanBudget.biases_dense and never part of `total`.
  * Embeddings: Mirror-exact also projects the patch embedding, position embedding, CLS token and registers when the
    span starts at block 0. That saving is reported separately (SpanBudget.embedding) and is NOT part of the matched
    matrix budget. Decision: the plan's budget is "over the same converted matrices", and no baseline compresses the
    embeddings, so including them would only shift the budget by a number the baselines cannot use. The embedding
    counts are the equivariant-subspace dimensions of m1.project_embeddings' projections on the evaluation grid
    (verify against a direct count on DINOv2: its stored position table is larger than the grid it is interpolated to).

Budgets for the baselines. A baseline must reach `total`, which includes Mirror-exact's basis changes, so methods
without basis changes receive those parameters in their matrices: `matrix_targets` spreads `total` over the matrices
in proportion to Mirror-exact's per-matrix counts (integer ceilings, so the targets sum to at least `total`).
`allocate` then turns the targets into ranks, widths or slice dimensions for each method (see its docstring).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch.nn as nn

MATRIX_NAMES = ("fc1", "fc2", "wq", "wk", "wv", "wo")
COMPONENT_MATRICES = {"mlp": ("fc1", "fc2"), "attn": ("wq", "wk", "wv", "wo")}
COMPONENTS = ("mlp", "attn")
BUDGET_TOL = 0.01

LOWRANK_METHODS = ("svd", "svdllm", "asvd_scale")
ALL_METHODS = LOWRANK_METHODS + ("slicegpt", "flap", "zipit")

__all__ = ["MATRIX_NAMES", "COMPONENT_MATRICES", "BUDGET_TOL", "MatrixBudget", "SpanBudget", "normalise_components",
           "model_dims", "matrix_shapes", "mirror_exact_basis_change", "mirror_exact_budget", "mirror_local_budget",
           "dense_budget", "mirror_exact_embedding_params", "matrix_targets", "allocate", "allocation_totals",
           "allocation_step", "gap_of", "gap_ok"]


# ----------------------------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------------------------
@dataclass
class MatrixBudget:
    block: int
    name: str              # 'fc1', 'fc2', 'wq', 'wk', 'wv', 'wo'
    shape: tuple           # (out, in)
    dense: int
    mirror_exact: int      # the Mirror count of this matrix (Mirror-local's count in a mirror_local_budget)


@dataclass
class SpanBudget:
    model: str
    blocks: list
    components: tuple      # ('mlp',) | ('attn',) | ('mlp', 'attn')
    matrices: list         # [MatrixBudget]
    basis_change: int
    total: int             # sum of mirror_exact over matrices + basis_change (biases and embeddings excluded)
    biases_dense: int
    notes: str
    # additions beyond the agreed contract (all have defaults)
    variant: str = "mirror_exact"            # "mirror_exact", "mirror_local" or "dense"
    dims: dict = field(default_factory=dict)  # d, h, H, dh, D, depth
    embedding: dict = field(default_factory=dict)

    @property
    def matrix_total(self) -> int:
        return int(sum(m.mirror_exact for m in self.matrices))

    @property
    def dense_total(self) -> int:
        return int(sum(m.dense for m in self.matrices))

    def matrix(self, block: int, name: str) -> MatrixBudget:
        for m in self.matrices:
            if m.block == int(block) and m.name == name:
                return m
        raise KeyError((block, name))

    def as_dict(self) -> dict:
        return {"model": self.model, "variant": self.variant, "blocks_0based": [int(b) for b in self.blocks],
                "blocks_1based": [int(b) + 1 for b in self.blocks], "components": list(self.components),
                "matrix_total": self.matrix_total, "dense_total": self.dense_total,
                "basis_change": int(self.basis_change), "total": int(self.total),
                "biases_dense": int(self.biases_dense), "dims": dict(self.dims), "embedding": dict(self.embedding),
                "matrices": [{"block": m.block, "name": m.name, "shape": list(m.shape), "dense": m.dense,
                              "mirror": m.mirror_exact} for m in self.matrices],
                "notes": self.notes}


# ----------------------------------------------------------------------------------------------
# Shapes and helpers
# ----------------------------------------------------------------------------------------------
def normalise_components(components) -> tuple:
    """A non-empty subset of ("mlp", "attn") in that order; strings such as "mlp+attn" are accepted."""
    if isinstance(components, str):
        components = [c for c in components.replace(",", "+").split("+") if c.strip()]
    comps = {str(c).strip() for c in components}
    bad = comps - set(COMPONENTS)
    if bad or not comps:
        raise ValueError(f"components must be a non-empty subset of {COMPONENTS}, got {sorted(comps)}")
    return tuple(c for c in COMPONENTS if c in comps)


def _module(model) -> nn.Module:
    """The network itself, for an adapter (mirror/adapters.py) or a bare nn.Module."""
    inner = getattr(model, "model", None)
    return inner if isinstance(inner, nn.Module) and hasattr(inner, "blocks") else model


def model_dims(model) -> dict:
    """d, h, H, dh, D (= H dh) and depth of a pre-norm ViT (adapter or nn.Module). A dict with keys d, h, H, dh and
    depth is accepted as well (budgets without weights, e.g. DINOv2-B from its known sizes)."""
    from .attention import attn_dims

    if isinstance(model, dict):
        out = {k: int(model[k]) for k in ("d", "h", "H", "dh", "depth")}
        out["D"] = out["H"] * out["dh"]
        return out
    m = _module(model)
    blk = m.blocks[0]
    H, dh = attn_dims(blk)
    return {"d": int(blk.mlp.fc1.in_features), "h": int(blk.mlp.fc1.out_features), "H": int(H), "dh": int(dh),
            "D": int(H * dh), "depth": int(len(m.blocks))}


def matrix_shapes(dims: dict) -> dict:
    d, h, D = dims["d"], dims["h"], dims["D"]
    return {"fc1": (h, d), "fc2": (d, h), "wq": (D, d), "wk": (D, d), "wv": (D, d), "wo": (d, D)}


def _bias_count(dims: dict, components) -> int:
    """Bias entries of the converted matrices of one block (fc1 h, fc2 d, q/k/v D each, o d)."""
    n = 0
    if "mlp" in components:
        n += dims["h"] + dims["d"]
    if "attn" in components:
        n += 3 * dims["D"] + dims["d"]
    return n


def _blocks(blocks) -> list:
    out = [int(b) for b in blocks]
    if not out or len(set(out)) != len(out) or out != sorted(out):
        raise ValueError(f"blocks must be a non-empty, increasing list without repeats, got {out}")
    return out


def _contiguous(blocks) -> bool:
    return blocks == list(range(blocks[0], blocks[0] + len(blocks)))


def _pick(seq, b: int, blocks: list, depth: int, what: str):
    """Per-block structure: indexed by block number (len == depth) or aligned with `blocks`."""
    if seq is None:
        raise ValueError(f"{what} is required for this component")
    if len(seq) == depth:
        return seq[b]
    if len(seq) == len(blocks):
        return seq[blocks.index(b)]
    raise ValueError(f"{what}: expected {depth} entries (indexed by block) or {len(blocks)} (aligned with blocks)")


def mirror_exact_basis_change(d: int, blocks, depth: int) -> int:
    """d^2 at each end of the span that is not a model boundary (span start > 0; span end < depth - 1)."""
    blocks = _blocks(blocks)
    return int(d * d * (int(blocks[0] > 0) + int(blocks[-1] < depth - 1)))


def _tr(A) -> int:
    return int(round(float(np.trace(np.asarray(A, dtype=np.float64)))))


def _make_budget(variant: str, model_name: str, dims: dict, blocks: list, comps: tuple, counts: dict,
                 basis: int, notes: str, embedding: dict | None = None) -> SpanBudget:
    shapes = matrix_shapes(dims)
    mats = []
    for b in blocks:
        for c in comps:
            for name in COMPONENT_MATRICES[c]:
                o, i = shapes[name]
                mats.append(MatrixBudget(block=int(b), name=name, shape=(int(o), int(i)), dense=int(o * i),
                                         mirror_exact=int(counts[(b, name)])))
    total = int(sum(m.mirror_exact for m in mats) + int(basis))
    return SpanBudget(model=model_name, blocks=list(blocks), components=comps, matrices=mats,
                      basis_change=int(basis), total=total, biases_dense=int(len(blocks) * _bias_count(dims, comps)),
                      notes=notes, variant=variant, dims=dict(dims), embedding=dict(embedding or {}))


# ----------------------------------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------------------------------
def mirror_exact_budget(model, A_star, blocks, perms, head_maps, components=("mlp", "attn"), model_name: str = "",
                        grid_hw=None, reg_perm=None) -> SpanBudget:
    """Mirror-exact budget (Section 4.5): one shared operator A_star over the contiguous span `blocks`.

    perms and head_maps: the fitted neuron pairings and head maps (attention.HeadMaps), indexed by block number
    (length = depth) or aligned with `blocks`; perms may be None without "mlp", head_maps None without "attn".
    Per-matrix counts: fc1 = fc2 = surgery.equivariant_param_count(h, d, self-twins, tr A*), wq..wo =
    attention.attention_free_params(d, head_maps[b], A*, A*), as heal_vexact.vexact_block_params.
    Basis change: mirror_exact_basis_change. With grid_hw (and reg_perm for register pairs) and a span starting at
    block 0, the embedding saving is added to `embedding` (reported separately, not in `total`)."""
    from .attention import attention_free_params
    from .heal import exact_involution
    from .surgery import equivariant_param_count, perm_trace

    comps = normalise_components(components)
    dims = model_dims(model)
    blocks = _blocks(blocks)
    if not _contiguous(blocks):
        raise ValueError("Mirror-exact needs a contiguous span of blocks")
    if blocks[-1] >= dims["depth"]:
        raise ValueError(f"block {blocks[-1]} beyond the model depth {dims['depth']}")
    A = exact_involution(A_star)
    d, h = dims["d"], dims["h"]
    tr_a = _tr(A)
    counts = {}
    for b in blocks:
        if "mlp" in comps:
            perm = np.asarray(_pick(perms, b, blocks, dims["depth"], "perms"))
            fc = equivariant_param_count(h, d, perm_trace(perm), tr_a)
            counts[(b, "fc1")] = counts[(b, "fc2")] = int(fc)
        if "attn" in comps:
            att = attention_free_params(d, _pick(head_maps, b, blocks, dims["depth"], "head_maps"), A, A)
            for name, key in (("wq", "Wq"), ("wk", "Wk"), ("wv", "Wv"), ("wo", "Wo")):
                counts[(b, name)] = int(att[key])
    basis = mirror_exact_basis_change(d, blocks, dims["depth"])
    emb = {}
    if blocks[0] == 0:
        emb = (mirror_exact_embedding_params(model, A, grid_hw, reg_perm)
               if grid_hw is not None and not isinstance(model, dict) else
               {"note": "not computed (pass the model and grid_hw); reported separately, not part of the matched "
                        "budget"})
    dp = (d + tr_a) // 2
    notes = (f"Mirror-exact, one operator A* (d+ {dp}, d- {d - dp}) over blocks {blocks[0] + 1}..{blocks[-1] + 1} "
             f"(1-based) of {dims['depth']}; components {'+'.join(comps)}; basis changes: d^2 at each span end that "
             f"is not a model boundary = {basis:,}; biases dense, reported separately; embedding saving reported "
             f"separately, not matched.")
    return _make_budget("mirror_exact", model_name, dims, blocks, comps, counts, basis, notes, emb)


def mirror_local_budget(model, struct, blocks, components=("mlp", "attn"), head_maps=None, a_out_offset: int = 1,
                        model_name: str = "") -> SpanBudget:
    """Mirror-local budget (ablation): per-layer operators from `struct` (variants.Structure).

    MLP of block b: fc1 with A_in = A_b, fc2 with A_out = A_(b+1) (variants.mirror_param_counts). Attention of block b
    (needs head_maps, indexed by block or aligned with blocks): attention.attention_free_params(d, maps[b], A_b,
    A_(b + a_out_offset)), as scripts/m1_attention.variant_params("vlocal"). Basis change: one d x d per converted
    block."""
    from .attention import attention_free_params
    from .surgery import equivariant_param_count

    comps = normalise_components(components)
    dims = model_dims(model)
    blocks = _blocks(blocks)
    d, h = dims["d"], dims["h"]
    if (struct.d, struct.h) != (d, h):
        raise ValueError(f"structure dims {(struct.d, struct.h)} differ from the model's {(d, h)}")
    counts = {}
    for b in blocks:
        if "mlp" in comps:
            ns = struct.n_self(b)
            counts[(b, "fc1")] = int(equivariant_param_count(h, d, ns, 2 * int(struct.d_plus[b]) - d))
            counts[(b, "fc2")] = int(equivariant_param_count(h, d, ns, 2 * int(struct.d_plus[b + 1]) - d))
        if "attn" in comps:
            hm = _pick(head_maps, b, blocks, dims["depth"], "head_maps")
            att = attention_free_params(d, hm, struct.A_layers[b], struct.A_layers[b + int(a_out_offset)])
            for name, key in (("wq", "Wq"), ("wk", "Wk"), ("wv", "Wv"), ("wo", "Wo")):
                counts[(b, name)] = int(att[key])
    basis = int(d * d * len(blocks))
    notes = (f"Mirror-local (per-layer operators), blocks {[b + 1 for b in blocks]} (1-based); components "
             f"{'+'.join(comps)}; one d x d basis change per converted block = {basis:,}; biases dense, separate.")
    return _make_budget("mirror_local", model_name, dims, blocks, comps, counts, basis, notes)


def dense_budget(model, blocks, components=("mlp", "attn"), model_name: str = "") -> SpanBudget:
    """The budget at full size: every matrix keeps its dense count, no basis change (tests: each method must then
    reproduce the original model)."""
    comps = normalise_components(components)
    dims = model_dims(model)
    blocks = _blocks(blocks)
    shapes = matrix_shapes(dims)
    counts = {(b, n): shapes[n][0] * shapes[n][1] for b in blocks for c in comps for n in COMPONENT_MATRICES[c]}
    return _make_budget("dense", model_name, dims, blocks, comps, counts, 0, "dense (full size), no basis change")


def mirror_exact_embedding_params(model, A, grid_hw, reg_perm=None) -> dict:
    """Embedding parameters before and after the Mirror-exact projection (m1.project_embeddings), on the grid
    `grid_hw` actually used. Each projection X -> (X + T X)/2 with T an orthogonal involution keeps (n + tr T)/2
    parameters: patch kernel T = A (x) width flip (tr = tr A * C * p_h * (p_w mod 2)), patch bias and CLS token and
    the CLS position d+ = (d + tr A)/2, position grid T = column flip (x) A (tr = G_h (G_w mod 2) tr A), registers
    and their positions (R d + s tr A)/2 with s self-paired registers. Reported separately, never matched."""
    from .m1 import register_info

    m = _module(model)
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[0]
    tr_a = _tr(A)
    dp = (d + tr_a) // 2
    Gh, Gw = (int(v) for v in grid_hw)
    W = m.patch_embed.proj.weight
    C, ph, pw = int(W.shape[1]), int(W.shape[2]), int(W.shape[3])
    out = {"patch_weight": (d * C * ph * pw, (d * C * ph * pw + tr_a * C * ph * (pw % 2)) // 2)}
    if m.patch_embed.proj.bias is not None:
        out["patch_bias"] = (d, dp)
    out["pos_grid"] = (Gh * Gw * d, (Gh * Gw * d + Gh * (Gw % 2) * tr_a) // 2)
    has_cls = getattr(m, "cls_token", None) is not None
    reg = register_info(m, None) if hasattr(m, "blocks") else {"R": 0}
    R = int(reg["R"])
    perm = np.arange(R) if reg_perm is None or not len(reg_perm) else np.asarray(reg_perm, dtype=np.int64)
    s = int((perm == np.arange(R)).sum()) if R else 0
    pos = getattr(m, "pos_embed", None)
    if has_cls:
        out["cls_token"] = (d, dp)
        if pos is not None and not getattr(m, "no_embed_class", False):
            out["pos_cls"] = (d, dp)
    if R:
        out["registers"] = (R * d, (R * d + s * tr_a) // 2)
    dense = int(sum(v[0] for v in out.values()))
    kept = int(sum(v[1] for v in out.values()))
    return {"parts": {k: {"dense": int(v[0]), "mirror_exact": int(v[1])} for k, v in out.items()},
            "dense": dense, "mirror_exact": kept, "saving": dense - kept, "grid_hw": [Gh, Gw],
            "note": "reported separately, not part of the matched matrix budget; position table counted on the "
                    "evaluation grid (verify against m1.project_embeddings on DINOv2)"}


# ----------------------------------------------------------------------------------------------
# Allocation
# ----------------------------------------------------------------------------------------------
def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def matrix_targets(budget: SpanBudget) -> dict:
    """(block, name) -> integer target: budget.total spread over the matrices in proportion to the Mirror counts,
    each rounded up, so the targets sum to at least budget.total (basis changes of Mirror-exact are thereby given to
    the baseline's matrices)."""
    S = budget.matrix_total
    T = int(budget.total)
    if S <= 0:
        raise ValueError("budget has no matrix parameters")
    return {(m.block, m.name): _ceil_div(m.mirror_exact * T, S) for m in budget.matrices}


def _block_targets(budget: SpanBudget) -> dict:
    """block -> {"mlp": target, "attn": target} summed from matrix_targets."""
    t = matrix_targets(budget)
    out = {}
    for (b, name), v in t.items():
        comp = "mlp" if name in COMPONENT_MATRICES["mlp"] else "attn"
        out.setdefault(b, {"mlp": 0, "attn": 0})[comp] += int(v)
    return out


def _fill(units: list, T: int) -> list:
    """Round a set of integer counts toward the total T from below, then add single steps until the total reaches T.

    units: one dict per item with "target" (its share of T), "step" (weights per unit), "max" (largest count) and
    optionally "cap" (an item never holds more weights than this: a low-rank matrix is stored dense once its factors
    would be larger) and "fixed" (weights already spent elsewhere for this item). An item with count c holds
    min(c step, cap) weights. Each count starts at floor(target / step), capped at max; then, while the total is below
    T, the item with the largest remaining deficit (target - weights; ties to the earlier item) that can still grow
    gets one more unit. Because the targets sum to at least T, the total ends at most one step above T (or below T
    only when every item is full). Returns the counts."""
    def w(c, u):
        cap = u.get("cap")
        return c * int(u["step"]) if cap is None else min(c * int(u["step"]), int(cap))

    def full(c, u):
        return c >= int(u["max"]) or (u.get("cap") is not None and w(c, u) >= int(u["cap"]))

    counts = [min(int(u["target"]) // int(u["step"]), int(u["max"])) if u["step"] > 0 else 0 for u in units]
    total = sum(w(c, u) for c, u in zip(counts, units)) + sum(int(u.get("fixed", 0)) for u in units)
    while total < T:
        best, best_def = -1, None
        for idx, (c, u) in enumerate(zip(counts, units)):
            if full(c, u):
                continue
            deficit = int(u["target"]) - w(c, u)
            if best_def is None or deficit > best_def:
                best, best_def = idx, deficit
        if best < 0:
            break
        u = units[best]
        total += w(counts[best] + 1, u) - w(counts[best], u)
        counts[best] += 1
    return counts


def _lowrank_alloc(budget: SpanBudget) -> list:
    """Per matrix: rank r from _fill with step (m + n) per rank, starting at floor(target / (m + n)); the factors hold
    r (m + n) weights. Where r (m + n) is at least m n the matrix is stored dense instead (rank min(m, n), m n
    weights, which is fewer than or equal to what the factors would hold)."""
    t = matrix_targets(budget)
    units = []
    for mb in budget.matrices:
        o, i = mb.shape
        units.append({"target": t[(mb.block, mb.name)], "step": o + i, "max": min(o, i), "cap": o * i})
    ranks = _fill(units, int(budget.total))
    out = []
    for mb, r in zip(budget.matrices, ranks):
        o, i = mb.shape
        full = min(o, i)
        dense_store = r * (o + i) >= o * i
        if dense_store:
            r = full
        out.append({"block": mb.block, "name": mb.name, "kind": "lowrank", "shape": [o, i], "rank": int(r),
                    "dense_storage": bool(dense_store), "params": int(o * i if dense_store else r * (o + i)),
                    "target": int(t[(mb.block, mb.name)])})
    return out


def _slice_layout(budget: SpanBudget, basis: str) -> dict:
    """Points of the stream that SliceGPT slices and the basis-change parameters as a function of d_s.

    Both components: the residual stream is sliced over the span. Points per block b: "a{b}" (stream entering block b,
    read by wq, wk, wv; written by the previous fc2) and "m{b}" (stream entering the MLP, written by wo, read by fc1);
    "e" is the stream leaving the span (written by the last fc2). All points share one slice dimension d_s.
      per_block  each point has its own PCA basis Q_p (SliceGPT); the residual connections carry the shortcut matrices
                 Q_p^T Q_(p+1) sliced on both sides, two per block (attention and MLP), d_s x d_s each.
      shared     one basis for the whole span (our extension): the shortcuts are identities and cost nothing.
      Entry (span start > 0): d x d_s (the full stream rotated and sliced); free at block 0, where SliceGPT folds the
      rotation into the embeddings. Exit (span end < last block): d_s x d back to the full stream; free at the model
      end, where the final features are read through the head (the same convention as Mirror-exact's even head).
    One component only ("mlp" or "attn"): the stream itself is not sliced; that module reads only the first d_s and
    writes only the first d_s coordinates of a basis Q_b (the PCA basis of its input point, used for reading and
    writing, as the existing mirror/baselines.py slicegpt). An implementation holds the stream in basis Q_b around
    block b, rewriting the other (dense) module in that basis for free, and rotates the full stream between
    consecutive converted blocks: d x d per transition (per_block), plus a full d x d at an entry or exit that is not
    a model boundary (both variants)."""
    dims = budget.dims
    d, h, D, L = dims["d"], dims["h"], dims["D"], dims["depth"]
    blocks = budget.blocks
    k = len(blocks)
    comps = budget.components
    if basis not in ("per_block", "shared"):
        raise ValueError(f"slicegpt basis must be 'per_block' or 'shared', not {basis!r}")
    if not _contiguous(blocks):
        raise ValueError("slicegpt needs a contiguous span of blocks")
    enter, leave = int(blocks[0] > 0), int(blocks[-1] < L - 1)
    both = set(comps) == {"mlp", "attn"}
    per_d = k * ((4 * D + 2 * h) if both else (2 * h if comps == ("mlp",) else 4 * D))

    def basis_params(ds: int) -> int:
        if both:
            sc = 2 * k * ds * ds if basis == "per_block" else 0
            return sc + enter * d * ds + leave * ds * d
        return d * d * ((k - 1 if basis == "per_block" else 0) + enter + leave)

    return {"both": both, "per_d": per_d, "basis_params": basis_params, "enter": enter, "leave": leave, "d": d}


def _slice_alloc(budget: SpanBudget, basis: str = "per_block") -> list:
    lay = _slice_layout(budget, basis)
    d = lay["d"]
    T = int(budget.total)
    ds = d
    for cand in range(1, d + 1):
        if lay["per_d"] * cand + lay["basis_params"](cand) >= T:
            ds = cand
            break
    dims = budget.dims
    h, D = dims["h"], dims["D"]
    blocks = budget.blocks
    shared = basis == "shared"
    out = []

    def key(p: str) -> str:
        return "Q.span" if shared else f"Q.{p}"

    for i, b in enumerate(blocks):
        nxt = "e" if i == len(blocks) - 1 else f"a{blocks[i + 1]}"
        if lay["both"]:
            entries = [("wq", ds, D, f"a{b}", "in"), ("wk", ds, D, f"a{b}", "in"), ("wv", ds, D, f"a{b}", "in"),
                       ("wo", D, ds, f"m{b}", "out"), ("fc1", ds, h, f"m{b}", "in"), ("fc2", h, ds, nxt, "out")]
        else:
            pt = f"m{b}" if budget.components == ("mlp",) else f"a{b}"
            if budget.components == ("mlp",):
                entries = [("fc1", ds, h, pt, "in"), ("fc2", h, ds, pt, "out")]
            else:
                entries = [("wq", ds, D, pt, "in"), ("wk", ds, D, pt, "in"), ("wv", ds, D, pt, "in"),
                           ("wo", D, ds, pt, "out")]
        for name, di, do, pt, side in entries:
            out.append({"block": b, "name": name, "kind": "slice", "d_in": int(di), "d_out": int(do),
                        "shape": [int(do), int(di)], "params": int(di * do), "point": pt, "basis_key": key(pt),
                        "side": side, "d_s": int(ds)})
    # basis changes
    k = len(blocks)
    if lay["both"]:
        if not shared:
            for i, b in enumerate(blocks):
                nxt = "e" if i == k - 1 else f"a{blocks[i + 1]}"
                out.append({"block": b, "name": "shortcut_attn", "kind": "basis", "shape": [ds, ds],
                            "params": ds * ds, "from": key(f"a{b}"), "to": key(f"m{b}")})
                out.append({"block": b, "name": "shortcut_mlp", "kind": "basis", "shape": [ds, ds],
                            "params": ds * ds, "from": key(f"m{b}"), "to": key(nxt)})
        if lay["enter"]:
            out.append({"block": blocks[0], "name": "entry", "kind": "basis", "shape": [d, ds], "params": d * ds,
                        "to": key(f"a{blocks[0]}")})
        if lay["leave"]:
            out.append({"block": blocks[-1], "name": "exit", "kind": "basis", "shape": [ds, d], "params": ds * d,
                        "from": key("e")})
    else:
        pfx = "m" if budget.components == ("mlp",) else "a"
        if not shared:
            for i in range(k - 1):
                out.append({"block": blocks[i], "name": "rotation", "kind": "basis", "shape": [d, d],
                            "params": d * d, "from": key(f"{pfx}{blocks[i]}"), "to": key(f"{pfx}{blocks[i + 1]}")})
        if lay["enter"]:
            out.append({"block": blocks[0], "name": "entry", "kind": "basis", "shape": [d, d], "params": d * d,
                        "to": key(f"{pfx}{blocks[0]}")})
        if lay["leave"]:
            out.append({"block": blocks[-1], "name": "exit", "kind": "basis", "shape": [d, d], "params": d * d,
                        "from": key(f"{pfx}{blocks[-1]}")})
    return out


def _mlp_fill(budget: SpanBudget, spent: dict) -> dict:
    """block -> MLP width n (at least 1, at most h): the rest of each block's share of the budget after `spent`
    (attention weights already allocated in that block), floor(rest / 2 d) and then single neurons where the deficit is
    largest until the span total reaches budget.total (_fill)."""
    dims = budget.dims
    d, h = dims["d"], dims["h"]
    bt = _block_targets(budget)
    units = [{"target": max(bt[b]["mlp"] + bt[b]["attn"] - spent.get(b, 0), 0), "step": 2 * d, "max": h,
              "fixed": spent.get(b, 0)} for b in budget.blocks]
    ns = _fill(units, int(budget.total))
    return {b: max(1, n) for b, n in zip(budget.blocks, ns)}


def _flap_uniform(budget: SpanBudget) -> list:
    """FLAP "UL-UM" (uniform across layers and modules): per block, attention keeps
    H' = round(H x attention share / dense attention) whole heads (nearest, between 0 and H) and the MLP keeps the
    rest of the block's share as hidden neurons (_mlp_fill), so the span lands at most one neuron (2 d weights) above
    budget.total. With attention only, the heads are filled instead (_fill with one head, 4 dh d weights, per step),
    at most one head above the total."""
    dims = budget.dims
    d, H, dh = dims["d"], dims["H"], dims["dh"]
    head = 4 * dh * d
    bt = _block_targets(budget)
    comps = budget.components
    heads = {}
    if "attn" in comps:
        if "mlp" in comps:
            heads = {b: max(0, min(H, int(math.floor(bt[b]["attn"] / head + 0.5)))) for b in budget.blocks}
        else:
            hs = _fill([{"target": bt[b]["attn"], "step": head, "max": H} for b in budget.blocks], int(budget.total))
            heads = dict(zip(budget.blocks, hs))
    widths = _mlp_fill(budget, {b: heads.get(b, 0) * head for b in budget.blocks}) if "mlp" in comps else {}
    out = []
    for b in budget.blocks:
        if "attn" in comps:
            hk = heads[b]
            for name in COMPONENT_MATRICES["attn"]:
                out.append({"block": b, "name": name, "kind": "width", "unit": "head", "heads": int(hk),
                            "n": int(hk * dh), "params": int(hk * dh * d)})
        if "mlp" in comps:
            n = widths[b]
            for name in COMPONENT_MATRICES["mlp"]:
                out.append({"block": b, "name": name, "kind": "width", "unit": "neuron", "n": int(n),
                            "params": int(n * d)})
    return out


def _flap_global(budget: SpanBudget, scores: dict) -> list:
    """FLAP "AL-AM" (adaptive across layers and modules, FLAP's default): `scores[b]` holds standardised scores,
    "mlp" [h] per hidden neuron and "attn" [H] per head (computed by mirror/baselines.py). All items of the span are
    ranked together by score (descending; ties to the earlier block, MLP before attention, lower index) and kept
    in that order until the kept weights (2 d per neuron, 4 dh d per head) reach budget.total. If the item that
    crosses the total is a head whose weights overshoot by more than 1%, heads are skipped from then on and the total
    is reached with neurons (rounding up stays within 2 d). Every block keeps at least one neuron."""
    import torch

    dims = budget.dims
    d, h, H, dh = dims["d"], dims["h"], dims["H"], dims["dh"]
    comps = budget.components
    items = []   # (score, block, module order, index, weight)
    for b in budget.blocks:
        if "mlp" in comps:
            s = torch.as_tensor(scores[b]["mlp"], dtype=torch.float64).cpu().numpy()
            items += [(float(s[j]), b, 0, j, 2 * d) for j in range(h)]
        if "attn" in comps:
            s = torch.as_tensor(scores[b]["attn"], dtype=torch.float64).cpu().numpy()
            items += [(float(s[j]), b, 1, j, 4 * dh * d) for j in range(H)]
    items.sort(key=lambda x: (-x[0], budget.blocks.index(x[1]), x[2], x[3]))
    T = int(budget.total)
    kept = {(b, m): [] for b in budget.blocks for m in (0, 1)}
    total = 0
    # at least one neuron per block (the best one)
    if "mlp" in comps:
        for b in budget.blocks:
            best = next(it for it in items if it[1] == b and it[2] == 0)
            kept[(b, 0)].append(best[3])
            total += best[4]
    chosen = {(b, 0, j) for (b, m), js in kept.items() if m == 0 for j in js}
    heads_open = True
    for sc, b, m, j, w in items:
        if total >= T:
            break
        if (b, m, j) in chosen:
            continue
        if m == 1:
            if not heads_open:
                continue
            if total + w > T and (total + w - T) > BUDGET_TOL * T and "mlp" in comps:
                heads_open = False
                continue
        kept[(b, m)].append(j)
        chosen.add((b, m, j))
        total += w
    out = []
    for b in budget.blocks:
        if "attn" in comps:
            hk = sorted(kept[(b, 1)])
            chans = [int(hh * dh + c) for hh in hk for c in range(dh)]
            for name in COMPONENT_MATRICES["attn"]:
                out.append({"block": b, "name": name, "kind": "width", "unit": "head", "heads": len(hk),
                            "heads_kept": [int(x) for x in hk], "kept": chans, "n": len(chans),
                            "params": int(len(chans) * d)})
        if "mlp" in comps:
            nk = sorted(kept[(b, 0)])
            for name in COMPONENT_MATRICES["mlp"]:
                out.append({"block": b, "name": name, "kind": "width", "unit": "neuron", "kept": [int(x) for x in nk],
                            "n": len(nk), "params": int(len(nk) * d)})
    return out


def _zipit_alloc(budget: SpanBudget) -> list:
    """ZipIt-style merging: per block, every attention head keeps w_qk merged q/k channels and w_v merged v/o
    channels, s = w_qk + w_v = round(attention share / (2 H d)) (nearest, 2 <= s <= 2 dh), w_qk = ceil(s / 2);
    attention keeps H d (2 w_qk + 2 w_v) weights. The MLP keeps the rest of the block's share as merged neurons
    (_mlp_fill), at most one neuron (2 d weights) above budget.total over the span. With attention only, s is filled
    instead (_fill, 2 H d weights per step)."""
    dims = budget.dims
    d, H, dh = dims["d"], dims["H"], dims["dh"]
    bt = _block_targets(budget)
    comps = budget.components
    unit = 2 * H * d
    ss = {}
    if "attn" in comps:
        if "mlp" in comps:
            ss = {b: max(2, min(2 * dh, int(math.floor(bt[b]["attn"] / unit + 0.5)))) for b in budget.blocks}
        else:
            got = _fill([{"target": bt[b]["attn"], "step": unit, "max": 2 * dh} for b in budget.blocks],
                        int(budget.total))
            ss = {b: max(2, s) for b, s in zip(budget.blocks, got)}
    widths = _mlp_fill(budget, {b: ss.get(b, 0) * unit for b in budget.blocks}) if "mlp" in comps else {}
    out = []
    for b in budget.blocks:
        if "attn" in comps:
            s = ss[b]
            wqk, wv = _ceil_div(s, 2), s - _ceil_div(s, 2)
            for name, w in (("wq", wqk), ("wk", wqk), ("wv", wv), ("wo", wv)):
                out.append({"block": b, "name": name, "kind": "merge", "per_head": int(w), "n": int(w * H),
                            "params": int(w * H * d)})
        if "mlp" in comps:
            n = widths[b]
            for name in COMPONENT_MATRICES["mlp"]:
                out.append({"block": b, "name": name, "kind": "merge", "n": int(n), "params": int(n * d)})
    return out


def allocate(method: str, budget: SpanBudget, **hp) -> list:
    """Per-matrix allocation of `method` at `budget`, rounding toward giving the baseline more parameters.

    Returns a list of dicts, one per converted matrix ({"block", "name", "kind", "params", ...}) plus, for SliceGPT,
    entries of kind "basis" for its shortcut, rotation, entry and exit matrices. allocation_totals sums them.
      svd, svdllm, asvd_scale  kind "lowrank": "rank" per matrix (see _lowrank_alloc); per-matrix targets from
                               matrix_targets. Rounding: up, at most (m + n) weights per matrix above its target.
      slicegpt (hp basis = "per_block" | "shared")  kind "slice": one slice dimension d_s for every sliced point,
                               the smallest d_s whose matrices plus basis changes reach budget.total (_slice_layout).
                               Rounding: up, at most one unit of d_s (k (4 D + 2 h) + O(k d_s) weights).
      flap (hp allocation = "uniform" | "global"; "global" needs hp scores)  kind "width" (whole heads for attention,
                               hidden neurons for the MLP); see _flap_uniform and _flap_global.
      zipit                    kind "merge": per-head merged channel counts and merged MLP width (_zipit_alloc);
                               the groups themselves are chosen by mirror/baselines.py from the statistics.
    A method that cannot reach budget.total (the target exceeds the dense size) keeps everything; allocation_totals
    then gives a negative gap, which compress reports as a shortfall. Every rule ends at most one step above the
    total (allocation_step; for FLAP's global ranking one head, accepted only while the overshoot is within 1%), so
    the gap is within 1% whenever step / total is (true at DINOv2-B sizes for all four MUST-07 sizes; on the tiny
    test model some steps are larger than 1% of the budget)."""
    if method in LOWRANK_METHODS:
        return _lowrank_alloc(budget)
    if method == "slicegpt":
        return _slice_alloc(budget, hp.get("basis", "per_block"))
    if method == "flap":
        mode = hp.get("allocation", "global")
        if mode == "uniform":
            return _flap_uniform(budget)
        if mode == "global":
            if hp.get("scores") is None:
                raise ValueError("flap global allocation needs scores (computed by mirror.baselines.compress)")
            return _flap_global(budget, hp["scores"])
        raise ValueError(f"flap allocation must be 'uniform' or 'global', not {mode!r}")
    if method == "zipit":
        return _zipit_alloc(budget)
    raise ValueError(f"unknown method {method!r}; choose from {ALL_METHODS}")


def allocation_totals(alloc: list) -> dict:
    """{"matrices": sum of params over matrix entries, "basis_change": sum over basis entries, "total": both}."""
    mats = int(sum(a["params"] for a in alloc if a["kind"] != "basis"))
    basis = int(sum(a["params"] for a in alloc if a["kind"] == "basis"))
    return {"matrices": mats, "basis_change": basis, "total": mats + basis}


def allocation_step(method: str, budget: SpanBudget, alloc: list, **hp) -> int:
    """The method's smallest parameter step at this budget: the most its rounding can overshoot budget.total.
    lowrank: the largest (m + n) of a matrix; slicegpt: the increase of matrices plus basis changes from d_s - 1 to
    d_s; flap and zipit with the MLP: one neuron (2 d); attention only: one head (flap, 4 dh d) or one channel per
    head in two matrices (zipit, 2 H d). The budget is "reachable" in the 1% sense when step / total <= 1%."""
    dims = budget.dims
    d, H, dh = dims["d"], dims["H"], dims["dh"]
    if method in LOWRANK_METHODS:
        return int(max(sum(m.shape) for m in budget.matrices))
    if method == "slicegpt":
        lay = _slice_layout(budget, hp.get("basis", "per_block"))
        ds = next(a["d_s"] for a in alloc if a["kind"] == "slice")
        f = lambda x: lay["per_d"] * x + lay["basis_params"](x)   # noqa: E731
        return int(f(ds) - f(ds - 1))
    if method == "flap" and "attn" in budget.components and (
            hp.get("allocation", "global") == "global" or "mlp" not in budget.components):
        return 4 * dh * d          # global ranking: one head may cross the total (accepted while within 1%)
    if "mlp" in budget.components:
        return 2 * d
    return 4 * dh * d if method == "flap" else 2 * H * d


def gap_of(params_total: int, budget_total: int) -> float:
    return (int(params_total) - int(budget_total)) / int(budget_total)


def gap_ok(gap: float, step_rel: float = 0.0, tol: float = BUDGET_TOL) -> bool:
    """0 <= gap <= 1%, or, when the budget is not reachable at 1% (step_rel = step / total above 1%),
    0 <= gap <= step_rel."""
    return 0.0 <= gap <= max(tol, step_rel) + 1e-12

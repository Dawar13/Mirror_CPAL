"""Model sizes, budgets at a fraction of the full model, and the pre-registered A06 stacking prediction
(FINAL_PHASE_PLAN.md v2, Section 4 A06; docs/final_phase_design.md Section 4.2).

Counting (the MUST-09 accounting, so every A06 point is counted like the 0.765 point of the paper):
  full count N        every parameter of the released model, sum(p.numel() for p in adapter.model.parameters())
                      (scripts/must09_heal.py cost_of, mirror/heal_structured.param_counts). 86,583,552 for
                      dinov2_vitb14_reg: it includes the hub's 37 x 37 position table, the LayerNorm affines, the
                      LayerScale gammas, the mask token and the registers (notes/paper/method_spec.md Section 5).
  converted dense     the dense entries of the converted matrices: fc1 [h, d], fc2 [d, h], qkv [3 D, d] and
                      proj [d, D] of the chosen blocks and components (84,934,656 for all 12 blocks of DINOv2-B; the
                      same number as scripts/collect_results.analytic_cost's k (4 d^2 + 2 h d) when D = d).
  student count       N - converted dense + the student's kept matrix count + its basis changes. Biases, norms,
                      embeddings (including the stored position table) and everything outside the converted matrices
                      keep their dense count, exactly as for Mirror-exact's 66,203,360 (0.7646).

budget_for_fraction(model, f) builds the budget of an A06 point:
  target            ceil(f N), rounded up so the student gets the extra parameter (the plan's rule "rounding towards
                    more parameters for the baseline");
  matrix budget T   target - (N - converted dense). The returned FractionBudget is a mirror/budget.py SpanBudget with
                    total = T and no basis change, so mirror.baselines.compress("svdllm", ...) allocates it with its own
                    rule (budget._lowrank_alloc: ranks filled until the matrices reach T, at most one rank step above).
                    Its gap (params_matrices - T) / T is then in [0, 1%] whenever one rank step, the largest m + n of a
                    matrix, is at most 1% of T (true at DINOv2-B sizes: 3,840 against T > 30M at f = 0.40).
  per-matrix shares MatrixBudget.mirror_exact holds each matrix's share of T (largest-remainder rounding, summing to
                    T exactly), which budget.matrix_targets uses as the per-matrix target. proportions "dense"
                    (default): in proportion to the dense counts, one compression ratio for every matrix, which is
                    SVD-LLM's own setting as mirror/baselines.py records it ("a uniform compression ratio"; not
                    re-checked against arXiv 2403.07378 here, verify); the MUST-07 svdllm departed from it only to
                    match Mirror's per-matrix counts.
                    proportions=<SpanBudget> (for example mirror.budget.mirror_exact_budget at full conversion): in
                    proportion to that budget's per-matrix counts, the rule of the 0.765 point (MUST-07). Which one A06
                    uses is a coordinator decision; the 0.765 point itself always comes from A05.
student_size(budget, params_matrices, params_basis_change) turns a compressor's matrix count into the model-level
numbers every final-phase builder reports: params_model, params_frac, target, gap (relative to the target, at model
level), gap_matrices (relative to T, the compress convention) and eligible (0 <= gap_matrices <= 1%; this implies
0 <= gap <= 1% because T < target).

A06 prediction (pre-registered; FINAL_PHASE_PLAN.md A06 and Section 6, A06-2):
    acc(Mirror + SVD, B) = acc_SVD(B / 0.765) - delta,   delta = (distilled original - healed Mirror at 0.765),
with acc_SVD linear in log(size) between the measured SVD-LLM points (sizes 0.40, 0.50, 0.65, 0.765) and the
distilled original at size 1.0. acc_pred is a pure function of these numbers; predicted_gain and gain_within score
A06-2 ("measured gain within 50% of the prediction"). The prediction reads the losses as additive: Mirror costs delta
at its own size, and low rank applied on top of it costs what SVD-LLM costs at the relative size B / 0.765.

Sources: FINAL_PHASE_PLAN.md (A06, Section 6), notes/paper/method_spec.md (Section 5), scripts/must09_heal.py
(cost_of), mirror/heal_structured.py (param_counts), scripts/collect_results.py (analytic_cost), mirror/budget.py,
mirror/baselines.py (svdllm). No web sources were needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

import torch.nn as nn

from ...budget import (BUDGET_TOL, COMPONENT_MATRICES, MatrixBudget, SpanBudget, gap_ok, matrix_shapes, model_dims,
                       normalise_components)

A06_FRACTIONS = (0.65, 0.50, 0.40)
SVD_SIZES = (0.40, 0.50, 0.65, 0.765)
MIRROR_SIZE = 0.765
ORIGINAL_SIZE = 1.0
DINOV2_B_REG_PARAMS = 86_583_552        # notes/paper/method_spec.md Section 5 (every parameter of the hub model)

__all__ = ["A06_FRACTIONS", "SVD_SIZES", "MIRROR_SIZE", "ORIGINAL_SIZE", "DINOV2_B_REG_PARAMS", "FractionBudget",
           "full_model_params", "converted_dense_params", "target_params", "budget_for_fraction", "student_size",
           "acc_svd_interp", "acc_pred", "predicted_gain", "gain_within"]


# ----------------------------------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------------------------------
def _module(model_or_adapter):
    """The network for an adapter (mirror/adapters.py) or a bare nn.Module; dicts are passed through."""
    if isinstance(model_or_adapter, dict):
        return model_or_adapter
    inner = getattr(model_or_adapter, "model", None)
    return inner if isinstance(inner, nn.Module) and hasattr(inner, "blocks") else model_or_adapter


def full_model_params(model_or_adapter) -> int:
    """Every parameter of the original model: sum(p.numel() for p in model.parameters()), the count behind 0.7646
    (86,583,552 for dinov2_vitb14_reg). Pass the original model or its adapter, not a converted copy: folding removes
    the block norm affines and DINOv2's materialised position grid is smaller than the stored table. A dict with
    "full_params" (and d, h, H, dh, depth) stands for a model that is not loaded (budget arithmetic only)."""
    m = _module(model_or_adapter)
    if isinstance(m, dict):
        if "full_params" not in m:
            raise ValueError("a dict model needs 'full_params' (the original model's parameter count)")
        return int(m["full_params"])
    blocks = getattr(m, "blocks", None)
    if blocks is not None and len(blocks):
        from ...fold import RMSNorm

        if isinstance(getattr(blocks[0], "norm1", None), RMSNorm):
            raise ValueError("this model is LayerNorm-folded (a converted copy); pass the original model or adapter")
    return int(sum(p.numel() for p in m.parameters()))


def _blocks_of(model_or_adapter, blocks) -> list:
    depth = model_dims(_module(model_or_adapter))["depth"]
    out = list(range(depth)) if blocks is None else [int(b) for b in blocks]
    if not out or out != sorted(set(out)) or out[0] < 0 or out[-1] >= depth:
        raise ValueError(f"blocks must be increasing, without repeats, in [0, {depth - 1}]; got {out}")
    return out


def converted_dense_params(model_or_adapter, components=("mlp", "attn"), blocks=None) -> int:
    """Dense entries of the converted matrices (fc1, fc2 for "mlp"; qkv and proj for "attn") of `blocks` (default:
    all blocks)."""
    comps = normalise_components(components)
    dims = model_dims(_module(model_or_adapter))
    shapes = matrix_shapes(dims)
    per_block = sum(shapes[n][0] * shapes[n][1] for c in comps for n in COMPONENT_MATRICES[c])
    return int(per_block * len(_blocks_of(model_or_adapter, blocks)))


def target_params(model_or_adapter, frac) -> int:
    """ceil(frac x full count), in exact decimal arithmetic (0.65 means 13/20, not the nearest binary float)."""
    f = Fraction(str(float(frac)))
    if not 0 < f <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {frac}")
    return int(math.ceil(f * full_model_params(model_or_adapter)))


# ----------------------------------------------------------------------------------------------
# Budget at a fraction of the full model
# ----------------------------------------------------------------------------------------------
@dataclass
class FractionBudget(SpanBudget):
    """A SpanBudget (mirror/budget.py) for a point at a fraction of the full model (see the module docstring)."""

    frac: float = 0.0
    target_model: int = 0          # ceil(frac N)
    full_params: int = 0           # N
    outside_params: int = 0        # N - converted dense (biases, norms, embeddings, unconverted matrices)
    proportions: str = "dense"


def _shares(weights: Sequence[int], total: int) -> list:
    """Integers in proportion to `weights` that sum to `total` exactly (largest remainder; ties to the earlier
    item)."""
    W = int(sum(int(w) for w in weights))
    if W <= 0:
        raise ValueError("weights must have a positive sum")
    raw = [Fraction(int(w) * int(total), W) for w in weights]
    base = [int(math.floor(x)) for x in raw]
    rest = int(total) - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base


def budget_for_fraction(model_or_adapter, frac, components=("mlp", "attn"), blocks=None, proportions="dense",
                        model_name: str = "") -> FractionBudget:
    """Budget whose student has frac x the full model's parameters (see the module docstring).

    model_or_adapter: the original model or its adapter (or a dict with d, h, H, dh, depth and full_params).
    components, blocks: the converted matrices (default: MLP and attention of every block).
    proportions: "dense" (one compression ratio for every matrix) or a SpanBudget for the same blocks and components
    whose per-matrix counts set the shares (the MUST-07 rule when it is Mirror-exact's budget).
    Use with mirror.baselines.compress(model, "svdllm", budget.blocks, budget.components, budget, stats)."""
    comps = normalise_components(components)
    m = _module(model_or_adapter)
    dims = model_dims(m)
    blks = _blocks_of(m, blocks)
    full = full_model_params(m)
    dense = converted_dense_params(m, comps, blks)
    outside = full - dense
    target = target_params(m, frac)
    T = int(target - outside)
    if T <= 0:
        raise ValueError(f"fraction {frac}: the target {target:,} does not exceed the parameters outside the converted "
                         f"matrices ({outside:,})")
    if T > dense:
        raise ValueError(f"fraction {frac}: the matrix budget {T:,} exceeds the dense converted matrices ({dense:,}); "
                         "nothing would be compressed")
    shapes = matrix_shapes(dims)
    keys = [(b, n) for b in blks for c in comps for n in COMPONENT_MATRICES[c]]
    if isinstance(proportions, str):
        if proportions != "dense":
            raise ValueError("proportions must be 'dense' or a SpanBudget")
        weights = [shapes[n][0] * shapes[n][1] for _, n in keys]
        prop = "dense"
    else:
        ref = {(int(mb.block), mb.name): int(mb.mirror_exact) for mb in proportions.matrices}
        if set(ref) != set(keys):
            raise ValueError("the proportions budget covers other matrices than (blocks, components)")
        weights = [ref[k] for k in keys]
        prop = f"per-matrix counts of a {getattr(proportions, 'variant', 'reference')} budget"
    shares = _shares(weights, T)
    mats = [MatrixBudget(block=int(b), name=n, shape=tuple(int(s) for s in shapes[n]),
                         dense=int(shapes[n][0] * shapes[n][1]), mirror_exact=int(s))
            for (b, n), s in zip(keys, shares)]
    per_block_bias = ((dims["h"] + dims["d"] if "mlp" in comps else 0)
                      + (3 * dims["D"] + dims["d"] if "attn" in comps else 0))
    notes = (f"A06 point: {float(frac):g} of the full model ({full:,} parameters): target {target:,} (rounded up), "
             f"matrix budget {T:,} = target - {outside:,} outside the converted matrices; per-matrix shares in "
             f"proportion to the {prop} counts (field mirror_exact holds each matrix's share); no basis change")
    return FractionBudget(model=model_name, blocks=blks, components=comps, matrices=mats, basis_change=0, total=T,
                          biases_dense=int(per_block_bias * len(blks)), notes=notes,
                          variant=f"fraction_{prop.split()[0]}",
                          dims=dict(dims), embedding={}, frac=float(frac), target_model=int(target),
                          full_params=int(full), outside_params=int(outside), proportions=prop)


def student_size(budget: FractionBudget, params_matrices: int, params_basis_change: int = 0,
                 tol: float = BUDGET_TOL) -> dict:
    """Model-level size of a student built at `budget` (from a compressor's params_matrices and its own basis
    changes): params_model, params_frac, target, gap = (params_model - target) / target, gap_matrices =
    (matrices + basis - T) / T and eligible (0 <= gap_matrices <= tol)."""
    if not isinstance(budget, FractionBudget):
        raise TypeError("student_size needs a FractionBudget (budget_for_fraction)")
    kept = int(params_matrices) + int(params_basis_change)
    pm = int(budget.outside_params) + kept
    T = int(budget.total)
    gm = (kept - T) / T
    return {"params_model": pm, "params_frac": pm / budget.full_params, "target": int(budget.target_model),
            "gap": (pm - budget.target_model) / budget.target_model, "gap_matrices": gm,
            "eligible": bool(gap_ok(gm, 0.0, tol)), "frac": float(budget.frac), "full_params": int(budget.full_params),
            "budget_matrices": T}


# ----------------------------------------------------------------------------------------------
# A06 prediction (pre-registered)
# ----------------------------------------------------------------------------------------------
def _points(svd_points) -> list:
    pts = dict(svd_points.items()) if isinstance(svd_points, Mapping) else dict(svd_points)
    out = sorted((float(s), float(a)) for s, a in pts.items())
    if len(out) < 1 or any(s <= 0 for s, _ in out):
        raise ValueError("SVD-LLM points need positive sizes")
    return out


def acc_svd_interp(size: float, svd_points, distilled_original: float, original_size: float = ORIGINAL_SIZE) -> float:
    """acc_SVD(size): linear in log(size) between the measured SVD-LLM points ({size: accuracy}, sizes as fractions of
    the full model) and the distilled original at `original_size`. Sizes outside the measured range raise (the
    pre-registered rule does not extrapolate)."""
    pts = _points(svd_points)
    if any(abs(s - original_size) < 1e-12 for s, _ in pts):
        raise ValueError("the size of the distilled original must not be among the SVD-LLM points")
    pts = sorted(pts + [(float(original_size), float(distilled_original))])
    s = float(size)
    lo, hi = pts[0][0], pts[-1][0]
    if not lo - 1e-12 <= s <= hi + 1e-12:
        raise ValueError(f"size {s:.4f} is outside the interpolation range [{lo}, {hi}]")
    x = math.log(min(max(s, lo), hi))
    for (s0, a0), (s1, a1) in zip(pts[:-1], pts[1:]):
        x0, x1 = math.log(s0), math.log(s1)
        if x0 - 1e-15 <= x <= x1 + 1e-15:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return float(a0 + t * (a1 - a0))
    raise AssertionError("unreachable")


def acc_pred(B: float, svd_points, distilled_original: float, mirror_healed: float, mirror_size: float = MIRROR_SIZE,
             original_size: float = ORIGINAL_SIZE) -> float:
    """Pre-registered A06 prediction for Mirror + SVD-LLM at budget B (a fraction of the full model):
    acc_SVD(B / mirror_size) - delta, delta = distilled_original - mirror_healed (the healed Mirror at mirror_size)."""
    delta = float(distilled_original) - float(mirror_healed)
    return acc_svd_interp(float(B) / float(mirror_size), svd_points, distilled_original, original_size) - delta


def predicted_gain(B: float, svd_points, distilled_original: float, mirror_healed: float,
                   mirror_size: float = MIRROR_SIZE, original_size: float = ORIGINAL_SIZE) -> float:
    """Predicted stacking gain at B: acc_pred(B) minus the measured SVD-LLM accuracy at B (B must be one of the
    measured SVD-LLM sizes)."""
    pts = dict(_points(svd_points))
    match = [s for s in pts if abs(s - float(B)) < 1e-9]
    if not match:
        raise ValueError(f"no measured SVD-LLM point at B = {B}")
    return acc_pred(B, svd_points, distilled_original, mirror_healed, mirror_size, original_size) - pts[match[0]]


def gain_within(measured_gain: float, predicted: float, rel_tol: float = 0.5) -> bool:
    """A06-2 at one budget: |measured - predicted| <= rel_tol |predicted| ("within 50% of the prediction")."""
    return bool(abs(float(measured_gain) - float(predicted)) <= float(rel_tol) * abs(float(predicted)) + 1e-12)

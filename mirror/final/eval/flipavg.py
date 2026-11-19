"""Two-pass flip averaging for any model (A07, FINAL_PHASE_PLAN.md v2), with MUST-10's definitions.

Why: exact mirror invariance can also be had at test time by averaging the features of an image and of its mirror,
at twice the compute. A07 compares Mirror (one pass) with flip-averaged models at equal compute.

Features (scripts/flip_avg_baseline.py and scripts/must10_flipavg.py, MUST-10; notes/paper/method_spec.md Section 7):
- k-NN: f_avg(x) = (f(x) + f(mirror x)) / 2 with f = adapter.cls_feature, the CLS token after the final norm,
  averaged BEFORE the L2 normalisation of the k-NN, which then acts on the average as it acts on every other feature.
  Two separate forward passes of the same batch, the first on x itself, averaged in float32 (FlipAveragedAdapter
  below is scripts/flip_avg_baseline.FlipAveragedAdapter). The bank and the queries are averaged alike. knn_fast runs
  this as the passes "bf16_flipavg" and "fp32_flipavg".
- linear probe (the recipe of scripts/tier2_linear.py): p_avg(x) = (p(x) + p(mirror x)) / 2 with p = [CLS after the
  final norm, mean patch token after the final norm] (tier2_linear.probe_feature), averaged as a whole
  (scripts/flip_avg_baseline.flip_averaged_probe_feature).
- DINOv2 linear protocol (linear_protocol.py, fullbank.py): the whole protocol feature [class tokens of the last 4
  blocks, mean patch token of the last block, each after the final norm] averaged the same way.
- flip consistency: measured by knn_fast; for the averaged feature it is 100% by construction (addition is
  commutative, so the feature of mirror x is the same sum of the same two forward passes), and is recorded as such.
- analytic FLOPs: the MUST-09 cost model (scripts/must09_heal.cost_of, which calls scripts/collect_results.
  analytic_cost with the per-block counts of mirror.heal_structured.block_costs): the matrices of every block at the
  method's own efficient form (the mirror family at its free parameters, compressors at their own counts), token
  mixing 2 T^2 d per block and the patch embedding; GFLOPs = 2 x MACs per image; norms, softmax, GELU, biases, the
  flip and the average are not counted. A flip-averaged model costs twice its single pass. analytic_cost reimplements
  cost_of's few lines here because importing scripts/must09_heal.py would set that script's GPU memory cap.

Sources: this repository only (scripts/flip_avg_baseline.py, scripts/must10_flipavg.py, scripts/must09_heal.py
cost_of, scripts/collect_results.py analytic_cost, mirror/heal_structured.py block_costs).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from mirror.data import mirror
from mirror.utils import REPO_ROOT

FORWARD_PASSES = {"plain": 1, "flipavg": 2}


class FlipAveragedAdapter:
    """The adapter with the flip-averaged k-NN feature; everything else is delegated to `base`
    (scripts/flip_avg_baseline.FlipAveragedAdapter)."""

    def __init__(self, base):
        self.__dict__["base"] = base

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def cls_feature(self, x: torch.Tensor, model=None) -> torch.Tensor:
        base = self.__dict__["base"]
        return 0.5 * (base.cls_feature(x, model).float() + base.cls_feature(mirror(x), model).float())


def flip_average(fn):
    """feature_fn(adapter, x, model) -> its flip average (fn(x) + fn(mirror x)) / 2 in float32, two forward passes."""

    def averaged(adapter, x, model=None):
        return 0.5 * (fn(adapter, x, model).float() + fn(adapter, mirror(x), model).float())

    averaged.__name__ = f"flipavg_{getattr(fn, '__name__', 'feature')}"
    return averaged


def cls_feature(adapter, x, model=None):
    """The k-NN feature: CLS after the final norm."""
    return adapter.cls_feature(x, model)


def probe_feature(adapter, x, model=None):
    """[CLS (or pooled) after the final norm, mean of the patch tokens after the final norm] in float32, [B, 2d]:
    scripts/tier2_linear.probe_feature (reimplemented: importing that script sets its own GPU memory cap)."""
    m = adapter.model if model is None else model
    t = m.norm(adapter.forward_tokens(x, m))
    p = int(adapter.n_prefix_tokens)
    pooled = getattr(adapter, "feature", "cls") == "pooled" or not getattr(adapter, "has_cls_token", True)
    first = m.forward_head(t, pre_logits=True) if pooled else t[:, 0]
    return torch.cat([first.float(), t[:, p:].float().mean(dim=1)], dim=-1)


def linear_feature(adapter, x, model=None):
    """The DINOv2 linear-protocol feature (linear_protocol.linear_feature)."""
    from .linear_protocol import linear_feature as lf

    return lf(adapter, x, model)


cls_feature_flipavg = flip_average(cls_feature)
probe_feature_flipavg = flip_average(probe_feature)
linear_feature_flipavg = flip_average(linear_feature)


# ----------------------------------------------------------------------------------------------
# Analytic cost (the MUST-09 cost model)
# ----------------------------------------------------------------------------------------------
def _collect_results():
    key = "_fp_eval_collect_results"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "collect_results.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def cost_arch(adapter) -> dict:
    """Inputs of scripts/collect_results.analytic_cost for the adapter at its evaluation size (as
    scripts/must09_heal.cost_of)."""
    H, W = adapter.grid_hw
    proj = getattr(getattr(adapter.model, "patch_embed", None), "proj", None)
    return {"d": int(adapter.embed_dim), "h": int(adapter.mlp_hidden), "depth": int(adapter.depth),
            "T": int(adapter.n_tokens), "n_patches": int(H * W), "patch": int(adapter.patch_size),
            "in_chans": int(getattr(proj, "in_channels", 3)),
            "model_params_dense": int(sum(p.numel() for p in adapter.model.parameters()))}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None


def analytic_cost(adapter, model=None, *, spec: dict | None = None, structure: dict | None = None,
                  info: dict | None = None, run_dir=None) -> dict:
    """Analytic GFLOPs and parameters of ONE forward pass per image. In order: a cost the loader or the run recorded
    (info["cost"], <run_dir>/metrics.json "cost", as scripts/must09_heal.py writes it), else the MUST-09 cost model
    from the student's spec and structure (mirror.heal_structured.block_costs into collect_results.analytic_cost),
    else the dense model when `model` is the adapter's own model (or None). Returns gflops_per_forward None with a
    reason when nothing applies. Multiply gflops_per_forward by the forward passes (2 for flip averaging)."""
    info = info if isinstance(info, dict) else {}
    arch = cost_arch(adapter)
    for src, c in (("loader info cost", info.get("cost")),
                   ("metrics.json cost", (_read_json(Path(run_dir) / "metrics.json") or {}).get("cost")
                    if run_dir else None)):
        if isinstance(c, dict) and c.get("gflops") is not None:
            return {"gflops_per_forward": float(c["gflops"]), "gflops_dense": c.get("gflops_dense"),
                    "params_model": c.get("params_model"), "params_dense": arch["model_params_dense"],
                    "source": src, "arch": arch}
    cr = _collect_results()
    dense = model is None or model is adapter.model
    spec = spec if spec is not None else (info.get("spec") if isinstance(info.get("spec"), dict) else None)
    if spec is None and run_dir and not dense:
        spec = _read_json(Path(run_dir) / "spec.json")
    structure = structure if structure is not None else info.get("structure")
    conv, basis, source = [], 0, "dense model (no converted blocks)"
    if not dense:
        if not isinstance(spec, dict) or "family" not in spec or "blocks" not in spec:
            return {"gflops_per_forward": None, "params_model": None, "params_dense": arch["model_params_dense"],
                    "source": "no recorded cost and no MUST-09-style spec (family, blocks) to count from", "arch": arch}
        from mirror.heal_structured import block_costs

        rows = block_costs(model, spec, structure, arch["d"], arch["h"])
        conv = [{k: r[k] for k in ("mlp", "attn", "rot")} for r in rows]
        basis = int(sum(r["basis_params"] for r in rows))
        source = "mirror.heal_structured.block_costs + scripts/collect_results.analytic_cost (" + \
                 (rows[0]["source"] if rows else "no blocks") + ")"
    c = cr.analytic_cost(arch, conv)
    params = None if c["model_params"] is None else int(c["model_params"]) + basis
    return {"gflops_per_forward": float(c["gflops_model"]), "gflops_dense": float(c["gflops_model_dense"]),
            "params_model": params, "params_dense": arch["model_params_dense"],
            "params_frac": None if params is None else params / arch["model_params_dense"],
            "source": source, "arch": arch}


def flipavg_cost(single: dict) -> dict:
    """The cost of the flip-averaged model: two forward passes of `single` (analytic_cost)."""
    g = single.get("gflops_per_forward")
    return {**single, "forward_passes_per_image": FORWARD_PASSES["flipavg"],
            "gflops_per_image": None if g is None else FORWARD_PASSES["flipavg"] * float(g)}

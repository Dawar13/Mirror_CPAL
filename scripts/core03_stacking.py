#!/usr/bin/env python
"""CORE-03: two axes of parsimony, Mirror and low rank stack (PROJECT_NEXT_PHASE.md Section 6, CORE-03;
PREREG_NEXT_PHASE.md CORE-03, P3.1). Zero-shot, full model, one model per invocation.

    python scripts/core03_stacking.py --model dinov2_vitb14_reg
    python scripts/core03_stacking.py --model dinov2_vitl14_reg
    python scripts/core03_stacking.py --model deit3_b16
    python scripts/core03_stacking.py --summary-only                 # rescore P3.1 and rewrite the summary
    python scripts/core03_stacking.py --tiny --out tiny_out/core03   # tiny random ViT, synthetic data, CPU

Curves (accuracy against parameters of the converted span, the whole model, MLP and attention):
  original         the dense model (fraction 1).
  mirror           Mirror alone: Mirror-exact (mirror_family "mirror_exact") at its own count, one point (its tax).
  lowrank          low rank alone at each budget point: plain SVD ("svd") and SVD-LLM ("svdllm") from
                   mirror/baselines.py compress, at mirror.stacking.fraction_budget (per-matrix shares in proportion to
                   Mirror-exact's counts, MUST-07's rule; --lowrank-weights dense adds a uniform-ratio variant,
                   method names "svd@dense", "svdllm@dense"). The curve takes, at each point, the method with the
                   higher method-selection top-1 (ties: the earlier in the order svd, svdllm, svd@dense, svdllm@dense).
  mirror_lowrank   Mirror plus low rank: Mirror-exact, then a truncated SVD inside the even and odd blocks of every
                   projected matrix (mirror.stacking.build_stacked), ranks allocated by budget.py's proportional rule.
                   Inner methods "mirror_svd" (plain truncation, the plan's method) and "mirror_svdllm" (whitened per
                   block). --score-inner says which one the P3.1 score uses: svd (default, the plan's text), svdllm,
                   or select (the higher method-selection top-1 at each point, as the low-rank-alone curve). It is
                   recorded in meta.json and must be fixed before the run.
Budget points: ceil(f x dense count of the converted span), f in --fractions (default 0.85, 0.70, 0.55, 0.40, the
pre-registered points). Where a point's budget is at or above Mirror-exact's own count, every block keeps full rank:
the Mirror-plus-low-rank point is Mirror-exact itself, below the budget (column below_budget; it is not re-evaluated,
the Mirror-alone row's predictions are copied and the row says same_as mirror_exact).

Data roles (PROJECT_NEXT_PHASE.md 4.1): structure from results/MUST-07/structure/<model>_s<fit seed>.npz (the canonical
structure; this script never fits one except in --tiny mode); SVD-LLM statistics on fit set <fit seed> (train);
k-NN bank Protocol.bank_loader (train); method-selection queries Protocol.msel_loader (train) for every choice;
validation Protocol.query_loader (50k) for reporting only. k-NN, flip consistency, parameter and FLOP counts reuse the
MUST-07 runner (scripts/must07_compressors.py: Evaluator, inventory, variant_costs, save_preds): CLS after the final
norm, cosine, k = 20, T = 0.07, bf16 features; the Mirror-family rows also get an fp32 flip pass (subset by default).

P3.1 (scored by score_p31, exactly as PREREG_NEXT_PHASE.md CORE-03 states): on a model, Mirror plus low rank lies
above low rank alone when its validation top-1 is strictly higher at a budget point; the model passes with at least 3
of the 4 points; the prediction passes when at least 2 of the 3 models (dinov2_vitb14_reg, dinov2_vitl14_reg,
deit3_b16) pass, at least one of them DINOv2-L or DeiT-III. Paired bootstrap intervals (validation, 10,000
resamples) are reported beside each point; they are not part of the test.

Outputs (resumable; one process per output directory at a time):
  <out>/rows_<model>.csv      one row per (model, fit seed, point, method)
  <out>/curves.csv            the curve points (scored and secondary), rewritten by every run
  <out>/p31.json              the P3.1 score;  <out>/bootstrap.csv  paired bootstrap per point
  <out>/preds/<model>/<point>/<method>_s<seed>.npz   per-image predictions;  <out>/info/...json  full row info
  <out>/done_<model>_s<seed>.flag   written when every row of the model is done (for the job scheduler)
  <out>/SUMMARY.md, <out>/meta.json (mirror/meta.py), <out>/meta/<model>_<time>.json
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.25) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import baselines as B  # noqa: E402
from mirror import mirror_family as MF  # noqa: E402
from mirror import stacking as ST  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.bootstrap import paired_bootstrap  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402


def _load_must07():
    name = "must07_compressors_for_core03"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "must07_compressors.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                   # dataclasses in the script look their module up
    spec.loader.exec_module(mod)
    return mod


M07 = _load_must07()

EXPERIMENT = "CORE-03"
PREREG_FRACTIONS = (0.85, 0.70, 0.55, 0.40)
PREREG_MODELS = ("dinov2_vitb14_reg", "dinov2_vitl14_reg", "deit3_b16")
PREREG_ONE_OF = ("dinov2_vitl14_reg", "deit3_b16")
LOWRANK_BASE = ("svd", "svdllm")
KEY = ["model", "fit_seed", "point", "method"]
LABEL = {"original": "original model", "mirror_exact": "Mirror-exact (Mirror alone)", "svd": "plain truncated SVD",
         "svdllm": "SVD-LLM-style whitened SVD", "svd@dense": "plain SVD, uniform ratio",
         "svdllm@dense": "SVD-LLM, uniform ratio", "mirror_svd": "Mirror-exact + per-block SVD",
         "mirror_svdllm": "Mirror-exact + per-block whitened SVD"}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-seed", type=int, default=0, help="fit set of the structure and the SVD-LLM statistics")
    p.add_argument("--fractions", default=",".join(f"{f:.2f}" for f in PREREG_FRACTIONS),
                   help="budget points as fractions of the converted span's dense count")
    p.add_argument("--lowrank-methods", default="svd,svdllm", help="low-rank-alone methods (mirror/baselines.py)")
    p.add_argument("--lowrank-weights", default="mirror",
                   help="per-matrix shares of the low-rank-alone budget: mirror (budget.py's rule), dense, or both "
                        "as 'mirror,dense' (each extra one adds svd@dense and svdllm@dense to the selection)")
    p.add_argument("--inner", default="svd,svdllm", help="inner methods of Mirror plus low rank")
    p.add_argument("--score-inner", default="svd", choices=["svd", "svdllm", "select"],
                   help="which Mirror-plus-low-rank curve P3.1 scores (fix before the run)")
    p.add_argument("--damp", type=float, default=B.DEFAULT_DAMP, help="SVD-LLM ridge (baselines.DEFAULT_DAMP)")
    p.add_argument("--flip-rows", default="mirror", choices=["mirror", "all", "none"],
                   help="rows with the bf16 mirrored validation pass (Mirror-family rows and the original by default)")
    p.add_argument("--fp32-flip", default="exact", choices=["exact", "none"],
                   help="fp32 flip-consistency pass on the rows that claim exact invariance")
    p.add_argument("--fp32-queries", type=int, default=None, help="validation queries of the fp32 pass (default 5000; tiny: all)")
    p.add_argument("--fp32-bank-per-class", type=int, default=None, help="bank images per class of the fp32 pass (default 10; tiny: all)")
    p.add_argument("--out", default=None, help="output directory (default results/CORE-03; tiny: results/CORE-03/tiny)")
    p.add_argument("--structure-dir", default=None, help="canonical structures (default results/MUST-07/structure)")
    p.add_argument("--structure", default=None, help="explicit structure file (overrides --structure-dir)")
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=None, help="feature-extraction batch (default 512; tiny 32)")
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--preload", default="cpu", choices=["cpu", "gpu", "loader"])
    p.add_argument("--queries", type=int, default=None, help="subsample the validation queries (debugging only)")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--gpu-frac", type=float, default=0.25, help="per-process GPU memory cap (set at import)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    p.add_argument("--summary-only", action="store_true", help="only rescore P3.1 and rewrite the summary")
    return p.parse_args(argv)


def _list(s) -> list:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def point_name(f: float) -> str:
    return f"f{float(f):.2f}"


def lowrank_methods(a) -> list:
    base = [m for m in _list(a.lowrank_methods)]
    bad = [m for m in base if m not in LOWRANK_BASE]
    if bad:
        raise SystemExit(f"--lowrank-methods: {bad} not in {LOWRANK_BASE}")
    out = []
    for w in _list(a.lowrank_weights):
        if w not in ("mirror", "dense"):
            raise SystemExit("--lowrank-weights takes mirror and/or dense")
        out += [m if w == "mirror" else f"{m}@dense" for m in base]
    return out


def inner_methods(a) -> list:
    inn = _list(a.inner)
    bad = [m for m in inn if m not in ST.INNER_METHODS]
    if bad:
        raise SystemExit(f"--inner: {bad} not in {ST.INNER_METHODS}")
    if a.score_inner != "select" and a.score_inner not in inn:
        raise SystemExit(f"--score-inner {a.score_inner} needs it in --inner")
    return inn


def _split_lowrank(method: str) -> tuple:
    m, _, w = method.partition("@")
    return m, (w or "mirror")


# ----------------------------------------------------------------------------------------------
# Paths and small helpers
# ----------------------------------------------------------------------------------------------
def preds_path(out: Path, model: str, point: str, method: str, seed: int) -> Path:
    return out / "preds" / model / point / f"{method}_s{seed}.npz"


def info_path(out: Path, model: str, point: str, method: str, seed: int) -> Path:
    return out / "info" / model / point / f"{method}_s{seed}.json"


def write_info(out: Path, model: str, point: str, method: str, seed: int, info: dict, extra: dict) -> None:
    path = info_path(out, model, point, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in info.items() if k not in ("tensors", "structure")}
    path.write_text(json.dumps(M07._jsonable(M07._strip_arrays({**keep, **extra})), indent=1), encoding="utf-8")


def load_structure(a, adapter, proto, device, model_name: str, out: Path):
    """The canonical structure file (never fitted here, except for the tiny model)."""
    if a.structure:
        path = Path(a.structure)
    elif a.tiny:
        path = out / "structure" / f"tiny_s{a.fit_seed}.npz"
        if not path.exists():
            st = MF.fit_span_structure(adapter, proto.fit_loader(a.fit_seed), device, model_name="tiny",
                                       fit_seed=a.fit_seed, top_k=3, workers=1)
            MF.save_span_structure(path, st)
    else:
        sdir = Path(a.structure_dir) if a.structure_dir else REPO_ROOT / "results" / "MUST-07" / "structure"
        path = sdir / f"{model_name}_s{a.fit_seed}.npz"
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"{path} missing: CORE-03 uses the canonical MUST-07 structure (run "
                         f"scripts/must07_compressors.py --model {model_name} --fit-seeds {a.fit_seed} --structure-only)")
    st = MF.load_span_structure(path)
    if not a.tiny and (st.model != model_name or st.fit_seed != a.fit_seed):
        raise SystemExit(f"{path} is for {st.model} fit seed {st.fit_seed}, not {model_name} fit seed {a.fit_seed}")
    if st.depth != adapter.depth:
        raise SystemExit(f"{path} has {st.depth} blocks, the model {adapter.depth}")
    return st, path


def evaluate(ctx, model, flip: bool, fp32: bool) -> tuple:
    ev = ctx["ev"]
    n0, te = ev.images_done, time.time()
    bm = ev.bank_msel(model)
    vv = ev.val(model, bm, flip=flip)
    fp = ev.flip_fp32(model) if fp32 else None
    dt = time.time() - te
    return bm, vv, fp, dt, (ev.images_done - n0) / max(dt, 1e-9)


def base_row(ctx, point: str, fraction, curve: str, method: str) -> dict:
    return {"model": ctx["model_name"], "fit_seed": ctx["seed"], "point": point, "fraction": fraction, "curve": curve,
            "method": method, "git_hash": ctx["git"]["git_hash"], "git_dirty": ctx["git"]["git_dirty"],
            "structure_sha": ctx["struct_sha"]}


def finish_row(ctx, row: dict, info: dict, model, bm, vv, fp, t0: float, t_eval: float, speed: float) -> dict:
    out, name, seed = ctx["out"], ctx["model_name"], ctx["seed"]
    M07.fill_eval(row, bm, vv, fp)
    pt, T, sd = int(info["params_total"]), int(row["budget_total"]), int(ctx["span_dense"])
    row.update({"params_matrices": int(info["params_matrices"]), "params_basis_change": int(info["params_basis_change"]),
                "params_total": pt, "span_dense": sd, "span_kept_frac": pt / sd, "gap": pt - T,
                "gap_rel": (pt - T) / T if T else float("nan"), "below_budget": bool(pt < T)})
    row.update(M07.variant_costs(ctx["inv"], model, info["params_matrices"], sd, info["params_basis_change"],
                                 info.get("per_matrix")))
    sha = M07.save_preds(preds_path(out, name, row["point"], row["method"], seed), {**vv, **bm, **(fp or {})})
    M07.save_labels(out, name, vv["val_y"], bm["msel_y"])
    row.update({"status": "ok", "preds_sha": sha, "time_s": round(time.time() - t0, 2), "eval_s": round(t_eval, 2),
                "img_per_s": round(speed, 1), "peak_mem_gb": M07._peak(ctx["device"])})
    write_info(out, name, row["point"], row["method"], seed, info, {"row": row})
    return row


def _flip(a, curve: str) -> bool:
    return a.flip_rows == "all" or (a.flip_rows == "mirror" and curve in ("original", "mirror", "mirror_lowrank"))


# ----------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------
def run_original(a, ctx) -> dict:
    t0 = time.time()
    M07._reset_peak(ctx["device"])
    m = ctx["adapter"].model
    bm, vv, fp, dt, sp = evaluate(ctx, m, flip=_flip(a, "original"), fp32=False)
    inv = ctx["inv"]
    row = base_row(ctx, "original", 1.0, "original", "original")
    row.update({"budget_total": inv["dense_all"], "hparams": "{}", "claims_exact_invariance": False,
                "same_as": "", "full_rank": True, "mirror_exact_total": ctx["mirror_total"]})
    info = {"method": "original", "params_matrices": ctx["span_dense"], "params_basis_change": 0,
            "params_total": ctx["span_dense"], "per_matrix": []}
    return finish_row(ctx, row, info, m, bm, vv, fp, t0, dt, sp)


def run_mirror(a, ctx) -> dict:
    t0 = time.time()
    M07._reset_peak(ctx["device"])
    model, info = ctx["mirror_base"]
    bm, vv, fp, dt, sp = evaluate(ctx, model, flip=_flip(a, "mirror"), fp32=a.fp32_flip == "exact")
    row = base_row(ctx, "mirror", info["params_total"] / ctx["span_dense"], "mirror", "mirror_exact")
    row.update({"budget_total": int(info["budget_total"]), "hparams": "{}", "claims_exact_invariance": True,
                "same_as": "", "full_rank": True, "mirror_exact_total": ctx["mirror_total"],
                "build_s": round(ctx["mirror_build_s"], 2)})
    return finish_row(ctx, row, info, model, bm, vv, fp, t0, dt, sp)


def run_lowrank(a, ctx, f: float, T: int, method: str, stats) -> dict:
    t0 = time.time()
    M07._reset_peak(ctx["device"])
    base, weights = _split_lowrank(method)
    budget = ST.fraction_budget(ctx["budget"], T, weights)
    hp = dict(B.HPARAM_GRIDS[base][0])
    if base == "svdllm":
        hp["damp"] = float(a.damp)
    tb = time.time()
    model, info = B.compress(ctx["adapter"].model, base, list(range(ctx["L"])), MF.COMPONENTS, budget, stats, hp,
                             keep_tensors=False, verbose=False)
    chk = B.structure_check(model, info)
    tb = time.time() - tb
    bm, vv, fp, dt, sp = evaluate(ctx, model, flip=_flip(a, "lowrank"), fp32=False)
    row = base_row(ctx, point_name(f), f, "lowrank", method)
    row.update({"budget_total": T, "hparams": json.dumps(info["hparams"], sort_keys=True), "alloc_weights": weights,
                "claims_exact_invariance": False, "same_as": "", "full_rank": False,
                "mirror_exact_total": ctx["mirror_total"], "gap_step": int(info.get("gap_step", 0)),
                "gap_note": info.get("gap_note", ""), "structure_check_ok": bool(chk["ok"]), "build_s": round(tb, 2)})
    row = finish_row(ctx, row, info, model, bm, vv, fp, t0, dt, sp)
    del model
    M07._free(ctx["device"])
    return row


def run_stacked(a, ctx, f: float, T: int, inner: str, stats, rows) -> dict:
    t0 = time.time()
    M07._reset_peak(ctx["device"])
    name, seed, out = ctx["model_name"], ctx["seed"], ctx["out"]
    method = f"mirror_{inner}"
    point = point_name(f)
    binfo = ctx["mirror_base"][1]
    row = base_row(ctx, point, f, "mirror_lowrank", method)
    row.update({"budget_total": T, "hparams": json.dumps({"inner": inner, **({"damp": a.damp} if inner == "svdllm" else {})},
                                                         sort_keys=True),
                "alloc_weights": "mirror (blocks)", "claims_exact_invariance": True,
                "mirror_exact_total": ctx["mirror_total"]})
    alloc = ST.allocate_stack(ctx["layout"], T - int(binfo["params_basis_change"]))
    if all(e["dense_storage"] for e in alloc):
        src = next((r for r in rows.rows if r["model"] == name and int(r["fit_seed"]) == seed and r["point"] == "mirror"
                    and r["method"] == "mirror_exact" and r.get("status") == "ok"), None)
        if src is None:
            raise RuntimeError("the Mirror-alone row must be evaluated before a full-rank stacked point")
        keep = {k: v for k, v in src.items() if k not in ("point", "fraction", "curve", "method")}
        row.update({**keep, "budget_total": T, "hparams": row["hparams"], "same_as": "mirror_exact",
                    "full_rank": True, "gap": int(src["params_total"]) - T,
                    "gap_rel": (int(src["params_total"]) - T) / T, "below_budget": int(src["params_total"]) < T,
                    "gap_note": "budget at or above Mirror-exact's own count: every block keeps full rank, so this "
                                "point is Mirror-exact itself (fewer parameters than the budget)",
                    "time_s": 0.0, "eval_s": 0.0, "build_s": 0.0})
        dst = preds_path(out, name, point, method, seed)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preds_path(out, name, "mirror", "mirror_exact", seed), dst)
        write_info(out, name, point, method, seed, {"method": method, "full_rank": True}, {"row": row})
        return row
    tb = time.time()
    model, info = ST.build_stacked(ctx["adapter"], ctx["struct"], T, MF.COMPONENTS, inner=inner, stats=stats,
                                   damp=a.damp, base=ctx["mirror_base"])
    chk = ST.check_stacked(model, info)
    tb = time.time() - tb
    bm, vv, fp, dt, sp = evaluate(ctx, model, flip=_flip(a, "mirror_lowrank"), fp32=a.fp32_flip == "exact")
    row.update({"same_as": "", "full_rank": bool(info["full_rank"]), "gap_step": int(info["gap_step"]),
                "gap_note": "within one block step above the budget (budget._fill)" if info["gap"] >= 0 else
                            "below the budget", "structure_check_ok": bool(chk["ok"]),
                "max_offdiag_residual": float(chk["max_residual"]), "build_s": round(tb, 2)})
    row = finish_row(ctx, row, info, model, bm, vv, fp, t0, dt, sp)
    del model
    M07._free(ctx["device"])
    return row


# ----------------------------------------------------------------------------------------------
# Scoring and summary
# ----------------------------------------------------------------------------------------------
def load_rows(out: Path):
    import pandas as pd

    frames = [pd.read_csv(f) for f in sorted(out.glob("rows_*.csv"))]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["fit_seed"] = df["fit_seed"].astype(int)
    return df[df["status"] == "ok"].copy()


def _pick(g, order: list):
    """Row with the highest method-selection top-1 (ties: earlier in `order`); validation is not read."""
    if g.empty:
        return None
    recs = sorted(g.to_dict("records"), key=lambda r: (-float(r["msel_top1"]),
                                                       order.index(r["method"]) if r["method"] in order else 99))
    return recs[0]


def build_curves(df, fit_seed: int, score_inner: str) -> list:
    """Long table of curve points with the scored choice at each budget point (method-selection data only)."""
    lr_order = ["svd", "svdllm", "svd@dense", "svdllm@dense"]
    in_order = ["mirror_svd", "mirror_svdllm"]
    out = []
    for model in sorted(df["model"].unique()):
        d = df[(df["model"] == model) & (df["fit_seed"] == fit_seed)]
        for _, r in d.iterrows():
            out.append({**{k: r.get(k) for k in ("model", "fit_seed", "point", "fraction", "curve", "method",
                                                 "budget_total", "params_total", "span_kept_frac", "val_top1",
                                                 "msel_top1", "flip_consistency_fp32", "gflops", "gflops_frac",
                                                 "below_budget", "same_as")}, "scored": False, "role": "row"})
        for point in sorted(d[d["curve"] == "lowrank"]["point"].unique(), reverse=True):
            g = d[d["point"] == point]
            lr = _pick(g[g["curve"] == "lowrank"], lr_order)
            ml = g[g["curve"] == "mirror_lowrank"]
            if score_inner == "select":
                mlr = _pick(ml, in_order)
            else:
                hit = ml[ml["method"] == f"mirror_{score_inner}"]
                mlr = hit.iloc[0].to_dict() if not hit.empty else None
            for role, r in (("lowrank_scored", lr), ("mirror_lowrank_scored", mlr)):
                if r is not None:
                    for o in out:
                        if (o["model"] == model and o["point"] == point and o["method"] == r["method"]
                                and o["role"] == "row"):
                            o.update({"scored": True, "role": role})
    return out


def score_p31(per_model: dict, models=PREREG_MODELS, one_of=PREREG_ONE_OF, need_models: int = 2,
              need_points: int = 3, n_points: int = 4) -> dict:
    """PREREG_NEXT_PHASE.md CORE-03 P3.1. per_model[model] = list of points, each {"point", "mirror_lowrank" (val
    top-1), "lowrank" (val top-1)}. A point counts when Mirror plus low rank is strictly higher; a model passes with
    at least need_points of its n_points points (all must be present to be scored); the prediction passes when at
    least need_models of `models` pass, one of them in `one_of`. Verdict "incomplete" while it is still open."""
    res, passed, open_ = {}, [], []
    for m in models:
        pts = per_model.get(m, [])
        complete = [p for p in pts if p.get("mirror_lowrank") is not None and p.get("lowrank") is not None
                    and math.isfinite(float(p["mirror_lowrank"])) and math.isfinite(float(p["lowrank"]))]
        above = [p["point"] for p in complete if float(p["mirror_lowrank"]) > float(p["lowrank"])]
        if len(complete) < n_points:
            status = "incomplete"
            # decided early only if the outcome no longer depends on the missing points
            missing = n_points - len(complete)
            if len(above) >= need_points:
                status = "pass"
            elif len(above) + missing < need_points:
                status = "fail"
        else:
            status = "pass" if len(above) >= need_points else "fail"
        res[m] = {"n_points": len(complete), "n_above": len(above), "points_above": above, "status": status}
        if status == "pass":
            passed.append(m)
        elif status == "incomplete":
            open_.append(m)
    ok_one = any(m in one_of for m in passed)
    can_one = ok_one or any(m in one_of for m in open_)
    if len(passed) >= need_models and ok_one:
        verdict = "PASS"
    elif len(passed) + len(open_) < need_models or not can_one:
        verdict = "FAIL"
    elif open_:
        verdict = "incomplete"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "models": res, "passed": passed, "open": open_,
            "rule": f"Mirror plus low rank strictly above low rank alone (validation top-1) at >= {need_points} of "
                    f"{n_points} budget points, on >= {need_models} of {list(models)}, including one of {list(one_of)}"}


def bootstrap_points(out: Path, curves: list, fit_seed: int, n_resamples: int):
    """Paired bootstrap (validation) of the scored Mirror-plus-low-rank point minus the scored low-rank point, per
    model and budget point (resumable, keyed by the prediction hashes). Reporting only."""
    import pandas as pd

    csv = V.ResultsCsv(out / "bootstrap.csv", ["model", "point", "method_a", "method_b", "sha_a", "sha_b"])
    rows = pd.DataFrame(curves) if curves else None
    if rows is None or rows.empty:
        return None
    sc = rows[rows["scored"]]
    for (model, point), g in sc.groupby(["model", "point"]):
        a = g[g["role"] == "mirror_lowrank_scored"]
        b = g[g["role"] == "lowrank_scored"]
        if a.empty or b.empty:
            continue
        ma, mb = a.iloc[0]["method"], b.iloc[0]["method"]
        pa, pb = preds_path(out, model, point, ma, fit_seed), preds_path(out, model, point, mb, fit_seed)
        lab = out / "preds" / model / "labels.npz"
        if not (pa.exists() and pb.exists() and lab.exists()):
            continue
        key = (model, point, ma, mb, M07.sha1(pa), M07.sha1(pb))
        if csv.done(key):
            continue
        with np.load(lab) as z:
            y = z["val"].astype(np.int64)
        with np.load(pa) as z:
            ca = z["val_pred"].astype(np.int64) == y
        with np.load(pb) as z:
            cb = z["val_pred"].astype(np.int64) == y
        res = paired_bootstrap(ca, cb, n_resamples=n_resamples, seed=0)
        csv.add({"model": model, "point": point, "method_a": ma, "method_b": mb, "sha_a": key[4], "sha_b": key[5], **res})
    return pd.read_csv(out / "bootstrap.csv") if (out / "bootstrap.csv").exists() else None


def _f(x, nd=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"


def postprocess(a, out: Path, git) -> dict:
    import pandas as pd

    df = load_rows(out)
    if df is None or df.empty:
        (out / "SUMMARY.md").write_text("# CORE-03\n\nNo rows yet.\n", encoding="utf-8")
        return {"summary": str(out / "SUMMARY.md"), "p31": None}
    curves = build_curves(df, a.fit_seed, a.score_inner)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    boot = bootstrap_points(out, curves, a.fit_seed, a.bootstrap_resamples)
    fr = [float(x) for x in _list(a.fractions)]
    prereg_points = sorted(round(x, 2) for x in fr) == sorted(PREREG_FRACTIONS)
    per_model = {}
    cdf = pd.DataFrame(curves)
    for model, g in cdf[cdf["scored"]].groupby("model"):
        pts = []
        for point, gp in g.groupby("point"):
            a_ = gp[gp["role"] == "mirror_lowrank_scored"]
            b_ = gp[gp["role"] == "lowrank_scored"]
            pts.append({"point": point, "mirror_lowrank": float(a_.iloc[0]["val_top1"]) if not a_.empty else None,
                        "lowrank": float(b_.iloc[0]["val_top1"]) if not b_.empty else None,
                        "mirror_lowrank_method": a_.iloc[0]["method"] if not a_.empty else None,
                        "lowrank_method": b_.iloc[0]["method"] if not b_.empty else None})
        per_model[model] = pts
    if a.tiny:
        p31 = score_p31(per_model, models=("tiny",), one_of=("tiny",), need_models=1)
        p31["note"] = "TINY TEST: random weights and synthetic data; the scoring code is exercised, nothing is tested"
    else:
        p31 = score_p31(per_model)
        if not prereg_points:
            p31 = {**p31, "verdict": "not scored", "note": f"budget points {fr} are not the pre-registered {PREREG_FRACTIONS}"}
    p31.update({"score_inner": a.score_inner, "fit_seed": a.fit_seed, "per_model_points": per_model,
                "git_hash": git["git_hash"], "git_dirty": git["git_dirty"], "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                "selection": "low rank alone: higher method-selection top-1 of its methods at each point"
                             + ("; Mirror plus low rank: likewise over the inner methods" if a.score_inner == "select"
                                else f"; Mirror plus low rank: inner method {a.score_inner} (fixed)")
                             + "; validation is read only for the comparison itself"})
    (out / "p31.json").write_text(json.dumps(M07._jsonable(p31), indent=2), encoding="utf-8")
    path = write_summary(out, a, df, cdf, p31, boot, git)
    return {"summary": str(path), "p31": p31, "curves": cdf, "bootstrap": boot}


def write_summary(out: Path, a, df, cdf, p31: dict, boot, git) -> Path:
    L = [f"# CORE-03: Mirror and low rank stack (zero-shot, full model)"
         + (" (TINY TEST: random weights, synthetic data)" if a.tiny else ""), "",
         f"Written by `scripts/core03_stacking.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}, commit "
         f"`{git['git_hash'][:10]}`{' (uncommitted changes)' if git['git_dirty'] else ''}; fit seed {a.fit_seed}.", "",
         "Protocol: k-NN top-1 with the 100k-image train bank and the 50k validation images (CLS after the final norm, "
         "cosine, k = 20, T = 0.07, bf16 features); every choice uses the 10k method-selection train images only. "
         "Parameters: the converted matrices of every block (MLP and attention) with basis changes (none at full "
         "conversion); fraction = parameters / dense count of the same matrices. Budget points: ceil(f x dense count). "
         "GFLOPs: analytic, per image, 2 x multiply-adds (MUST-07 runner).", "",
         f"## P3.1: {p31['verdict']}", "", f"Rule: {p31['rule']}.", "",
         f"Scored Mirror plus low rank: inner method **{p31['score_inner']}**. Low rank alone: the better of its "
         "methods on the method-selection set at each point.", ""]
    if p31.get("note"):
        L += [f"Note: {p31['note']}.", ""]
    L += ["| model | points scored | points above | status |", "|---|---|---|---|"]
    for m, r in p31["models"].items():
        L.append(f"| {m} | {r['n_points']} | {r['n_above']} ({', '.join(r['points_above']) or 'none'}) | {r['status']} |")
    L.append("")
    bmap = {}
    if boot is not None and not boot.empty:
        for _, r in boot.iterrows():
            bmap[(r["model"], r["point"], r["method_a"], r["method_b"])] = r
    for model in sorted(df["model"].unique()):
        d = df[(df["model"] == model) & (df["fit_seed"] == a.fit_seed)]
        L += [f"## {model}", ""]
        o = d[d["method"] == "original"]
        mr = d[d["method"] == "mirror_exact"]
        if not o.empty:
            r = o.iloc[0]
            L.append(f"Original: top-1 {_f(r['val_top1'])}, method-selection {_f(r['msel_top1'])}, span "
                     f"{int(r['span_dense']):,} weights, {_f(r['gflops'], 2)} GFLOPs.")
        if not mr.empty:
            r = mr.iloc[0]
            L.append(f"Mirror alone (its tax): {int(r['params_total']):,} weights = {_f(100 * float(r['span_kept_frac']))}% of "
                     f"the span, top-1 {_f(r['val_top1'])}, method-selection {_f(r['msel_top1'])}, flip fp32 "
                     f"{_f(r.get('flip_consistency_fp32'))}, {_f(r['gflops'], 2)} GFLOPs.")
        L += ["", "| point | budget | Mirror + low rank (scored) | weights | top-1 | msel | low rank alone (chosen) | "
                  "weights | top-1 | msel | difference | 95% interval | above |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        g = cdf[(cdf["model"] == model) & cdf["scored"]]
        for point in sorted(g["point"].unique(), reverse=True):
            gp = g[g["point"] == point]
            x = gp[gp["role"] == "mirror_lowrank_scored"]
            y = gp[gp["role"] == "lowrank_scored"]
            if x.empty or y.empty:
                L.append(f"| {point} | | {'missing' if x.empty else x.iloc[0]['method']} | | | | "
                         f"{'missing' if y.empty else y.iloc[0]['method']} | | | | | | |")
                continue
            x, y = x.iloc[0], y.iloc[0]
            diff = float(x["val_top1"]) - float(y["val_top1"])
            b = bmap.get((model, point, x["method"], y["method"]))
            ci = f"[{_f(b['ci_low'])}, {_f(b['ci_high'])}]" if b is not None else "n/a"
            note = " (Mirror-exact itself, below budget)" if str(x.get("same_as")) == "mirror_exact" else ""
            L.append(f"| {point} | {int(x['budget_total']):,} | {x['method']}{note} | {int(x['params_total']):,} | "
                     f"{_f(x['val_top1'])} | {_f(x['msel_top1'])} | {y['method']} | {int(y['params_total']):,} | "
                     f"{_f(y['val_top1'])} | {_f(y['msel_top1'])} | {_f(diff)} | {ci} | {'yes' if diff > 0 else 'no'} |")
        L += ["", "Every row (secondary curves included):", "",
              "| point | curve | method | weights | fraction | gap | top-1 | msel | flip | flip fp32 | GFLOPs | check |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        order = {"original": 0, "mirror": 1}
        d2 = d.assign(_o=d["point"].map(lambda p: order.get(p, 2)), _f=-d["fraction"].astype(float))
        for _, r in d2.sort_values(["_o", "_f", "curve", "method"]).iterrows():
            gap = r.get("gap_rel")
            L.append(f"| {r['point']} | {r['curve']} | {r['method']} | {int(r['params_total']):,} | "
                     f"{_f(r['span_kept_frac'], 3)} | {_f(100 * float(gap), 3) + '%' if gap == gap else ''} | "
                     f"{_f(r['val_top1'])} | {_f(r['msel_top1'])} | {_f(r.get('flip_consistency'))} | "
                     f"{_f(r.get('flip_consistency_fp32'))} | {_f(r['gflops'], 2)} | "
                     f"{'' if str(r.get('structure_check_ok', '')) in ('', 'nan') else r.get('structure_check_ok')} |")
        L.append("")
    L += ["## Notes", "",
          "- Mirror plus low rank keeps Mirror-exact's exact flip invariance (fp32 flip column); a point whose budget is "
          "at or above Mirror-exact's own count is Mirror-exact itself, with fewer weights than the budget.",
          "- Low rank alone spreads each budget over the matrices in proportion to Mirror-exact's per-matrix counts "
          "(mirror/budget.py, as MUST-07); Mirror plus low rank spreads it over the even and odd blocks in proportion "
          "to their Mirror-exact counts (mirror/stacking.py allocate_stack).",
          "- The bootstrap intervals are reported beside each point; P3.1 itself compares the point estimates.",
          "", "Files: rows_<model>.csv, curves.csv, p31.json, bootstrap.csv, preds/, info/, meta.json."]
    text = "\n".join(L) + "\n"
    assert chr(0x2014) not in text
    path = out / "SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    out = Path(a.out).resolve() if a.out else REPO_ROOT / "results" / EXPERIMENT / ("tiny" if a.tiny else "")
    out.mkdir(parents=True, exist_ok=True)
    git = git_info()
    fractions = [float(x) for x in _list(a.fractions)]
    lr_methods, inners = lowrank_methods(a), inner_methods(a)
    if a.summary_only:
        res = postprocess(a, out, git)
        return {"out": str(out), "rows_added": 0, **res}
    if a.tiny:
        device = torch.device("cpu")
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise SystemExit("no GPU visible; use --tiny for CPU runs")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 flip pass needs full fp32
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(0)
    model_name = "tiny" if a.tiny else a.model
    adapter, proto, num_classes = V.build_runtime(a.model, a.tiny, device, a.cache_dir,
                                                  a.eval_batch or (32 if a.tiny else 512), a.stats_batch_size,
                                                  0 if a.tiny else a.num_workers)
    adapter.model.float().eval()
    L = adapter.depth
    struct, spath = load_structure(a, adapter, proto, device, model_name, out)
    amp = "bf16" if device.type == "cuda" else None
    ev = M07.Evaluator(adapter, proto, device, amp, a.eval_batch or (32 if a.tiny else 512), a.preload, num_classes,
                       a.knn_k, a.knn_temperature, a.queries)
    if not a.tiny:
        ev.fp32_queries = a.fp32_queries if a.fp32_queries is not None else 5000
        ev.fp32_bank_per_class = a.fp32_bank_per_class if a.fp32_bank_per_class is not None else 10
    else:
        ev.fp32_queries, ev.fp32_bank_per_class = a.fp32_queries, a.fp32_bank_per_class
    rows = V.ResultsCsv(out / f"rows_{model_name}.csv", KEY)
    n_before = len(rows.rows)
    seed = int(a.fit_seed)
    budget = MF.span_budget(adapter.model, struct, range(L), MF.COMPONENTS, model_name)
    bf = MF.budget_fields(budget)
    ctx = {"adapter": adapter, "device": device, "ev": ev, "inv": M07.inventory(adapter), "git": git, "out": out,
           "model_name": model_name, "seed": seed, "struct": struct, "struct_sha": M07.sha1(spath), "L": L,
           "budget": budget, "span_dense": int(bf["span_dense"]), "mirror_total": int(bf["budget_total"])}
    log(f"{EXPERIMENT} {'TINY ' if a.tiny else ''}on {device}: model {model_name}, fit seed {seed}, points {fractions}, "
        f"low rank {lr_methods}, inner {inners} (scored: {a.score_inner}); span {ctx['span_dense']:,} weights, "
        f"Mirror-exact {ctx['mirror_total']:,} ({100 * ctx['mirror_total'] / ctx['span_dense']:.2f}%); commit "
        f"{git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")
    plan = [("original", None, "original", "original"), ("mirror", None, "mirror", "mirror_exact")]
    for f in fractions:
        plan += [(point_name(f), f, "lowrank", m) for m in lr_methods]
        plan += [(point_name(f), f, "mirror_lowrank", f"mirror_{i}") for i in inners]
    pending = [p for p in plan if not rows.done((model_name, seed, p[0], p[3]))]
    if pending:
        tb = time.time()
        ctx["mirror_base"] = MF.build_mirror_variant(adapter, "mirror_exact", range(L), MF.COMPONENTS, struct,
                                                     budget=budget)
        ctx["mirror_build_s"] = time.time() - tb
        s = ctx["mirror_base"][1]["structure"]
        blk0 = adapter.model.blocks[0]
        ctx["layout"] = ST.stack_layout(int(blk0.mlp.fc1.in_features), int(blk0.mlp.fc1.out_features),
                                        int(blk0.attn.qkv.out_features // 3), s["A"], s["perms"], s["heads_after"],
                                        list(range(L)), MF.COMPONENTS)
        basis = int(ctx["mirror_base"][1]["params_basis_change"])

        def needs_stats(p) -> bool:
            if p[2] == "lowrank":
                return _split_lowrank(p[3])[0] == "svdllm"
            if p[2] == "mirror_lowrank" and p[3] == "mirror_svdllm":
                T = ST.budget_for_fraction(ctx["span_dense"], p[1])
                return not all(e["dense_storage"] for e in ST.allocate_stack(ctx["layout"], T - basis))
            return False

        stats = None
        if any(needs_stats(p) for p in pending):
            t = time.time()
            stats = B.collect_stats(adapter, proto.fit_loader(seed), device, list(range(L)), ["svdllm"], MF.COMPONENTS)
            log(f"SVD-LLM statistics (fit seed {seed}): {time.time() - t:.1f} s")
        for point, f, curve, method in plan:
            if rows.done((model_name, seed, point, method)):
                continue
            if curve == "original":
                row = run_original(a, ctx)
            elif curve == "mirror":
                row = run_mirror(a, ctx)
            elif curve == "lowrank":
                row = run_lowrank(a, ctx, f, ST.budget_for_fraction(ctx["span_dense"], f), method, stats)
            else:
                row = run_stacked(a, ctx, f, ST.budget_for_fraction(ctx["span_dense"], f), method.split("_", 1)[1],
                                  stats, rows)
            rows.add(row)
            log(f"{model_name} {point} {method}: top-1 {float(row['val_top1']):.2f}, msel {float(row['msel_top1']):.2f}, "
                f"weights {int(row['params_total']):,} ({100 * float(row['span_kept_frac']):.2f}% of the span; budget "
                f"{int(row['budget_total']):,}){' same as Mirror-exact' if row.get('same_as') else ''}, "
                f"{float(row.get('time_s', 0.0)):.1f} s")
        del stats
        M07._free(device)
    if all(rows.done((model_name, seed, p[0], p[3])) for p in plan):
        (out / f"done_{model_name}_s{seed}.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    res = postprocess(a, out, git)
    wall = time.time() - t_start
    from mirror.adapters_timm import resolve_name

    try:
        model_meta = {model_name: (resolve_name(a.model) if not a.tiny else "tiny random timm ViT (CPU test)")}
    except KeyError:
        model_meta = {model_name: a.model}
    extra = {"fractions": fractions, "budgets": {point_name(f): ST.budget_for_fraction(ctx["span_dense"], f)
                                                 for f in fractions},
             "span_dense": ctx["span_dense"], "mirror_exact_total": ctx["mirror_total"],
             "mirror_fraction": ctx["mirror_total"] / ctx["span_dense"], "lowrank_methods": lr_methods,
             "inner_methods": inners, "score_inner": a.score_inner, "structure": {"path": str(spath),
                                                                                "sha": ctx["struct_sha"]},
             "rows_added": len(rows.rows) - n_before, "rows_total": len(rows.rows),
             "images_featurised": ev.images_done, "peak_mem_gb_process": M07._peak(device),
             "data_sizes": {r: s.n for r, s in ev._sets.items()}, "amp": amp,
             "fp32_flip": {"mode": a.fp32_flip, "queries": ev.fp32_queries, "bank_per_class": ev.fp32_bank_per_class},
             "p31_verdict": (res.get("p31") or {}).get("verdict")}
    kw = dict(experiment=EXPERIMENT, config=vars(a), seeds={"fit_seed": seed, "bootstrap_seed": 0, "global": 0},
              models=model_meta, device=device, wall_seconds=wall, extra=extra)
    write_meta(out, **kw)
    write_meta(out / "meta", name=f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json", **kw)
    log(f"{len(rows.rows) - n_before} new rows ({len(rows.rows)} in rows_{model_name}.csv); P3.1 "
        f"{(res.get('p31') or {}).get('verdict')}; summary {res['summary']}; {wall / 60:.1f} min")
    return {"out": str(out), "rows_added": len(rows.rows) - n_before, "rows_total": len(rows.rows), **res}


if __name__ == "__main__":
    main()

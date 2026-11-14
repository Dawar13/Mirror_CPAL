#!/usr/bin/env python
"""MUST-01: the law (character formula) against the parameter saving surgery achieves, on existing runs.

    python scripts/must01_law.py                         # CPU, reads runs/ and artifacts/, writes results/MUST-01/
    python scripts/must01_law.py --tiny --out tiny_out/must01

For every measured model and fit seed (protocol v2, per-layer runs of scripts/run_e1.py: runs/x2/*, runs/g/*)
and every block b (block 1 = code index 0):

  predicted  free parameters of the block's fc1 and fc2 from the character formula of mirror/groups.py,
             free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)) for G = {e, LR}, with rho_in = A_b (fc1),
             rho_out = A_(b+1) (fc2) and sigma the twin matching; tax = 1 - free / (h d);
  achieved   the weight count of the compressed MLP that surgery produces: random weights (seeded) are
             projected with surgery.project_mlp_two onto the fitted structure, and the even/odd module
             mirror.efficient.build_mirror_mlp is built from them (it also checks that the projected
             weights are exactly equivariant); the count is the number of entries of its weight tensors,
             so it comes from tensor shapes, not from the formula;
  recorded   the parameter saving the run itself recorded (e1_mlp.csv param_saving_frac) and, for
             blocks 1..k, the surgery rows of e1_surgery.csv (mlp_params_kept_frac).

Where the fitted operators and pairings exist locally (e1_operators.npz under the run or under
artifacts/a100/runs/), the structure is the fitted one. Where they do not (DINOv2-S and DINOv2-L), a
synthetic structure with the recorded counts (d+ per layer, twin pairs and self-twins per block) is used:
then both numbers derive from the same recorded counts and the check covers only the arithmetic. The
summary says which applies to each model.

Outputs: <out>/law_per_layer.csv (one row per model, fit seed and block), <out>/law_surgery_k.csv,
<out>/summary.md (maximum absolute per-block difference in percentage points, per model) and meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import groups as GR  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log  # noqa: E402

DEFAULT_RUNS = "runs/x2/perlayer_*_s*,runs/g/*_s*"
NPZ_ROOTS = ("artifacts/a100/runs",)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=DEFAULT_RUNS, help="comma-separated globs of run_e1.py run folders")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "MUST-01"))
    p.add_argument("--seed", type=int, default=0, help="seed of the random weights that are projected")
    p.add_argument("--tol-pp", type=float, default=1.0, help="pre-registered bar, percentage points per block")
    p.add_argument("--tiny", action="store_true", help="synthetic run folders (tiny model sizes) for the test")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
def _rel(p) -> str:
    """Path relative to the repository when it lies inside it (else as given)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def find_npz(run: Path) -> Path | None:
    own = run / "results" / "e1_operators.npz"
    if own.exists():
        return own
    try:
        rel = run.resolve().relative_to((REPO_ROOT / "runs").resolve())
    except ValueError:
        return None
    for root in NPZ_ROOTS:
        cand = REPO_ROOT / root / rel / "results" / "e1_operators.npz"
        if cand.exists():
            return cand
    return None


def discover(globs: str) -> list:
    runs = []
    for g in globs.split(","):
        g = g.strip()
        if not g:
            continue
        base = Path(g) if Path(g).is_absolute() else REPO_ROOT / g
        for run in sorted(base.parent.glob(base.name)):
            res = run / "results"
            if not (res / "e1_mlp.csv").exists() or not (res / "e1_meta.json").exists():
                continue
            meta = json.loads((res / "e1_meta.json").read_text(encoding="utf-8"))
            if meta.get("operator_mode") != "per_layer":
                continue
            cfg = meta.get("config", {}) or {}
            runs.append({"run": run, "model": (cfg.get("model", {}) or {}).get("name"),
                         "fit_seed": meta.get("fit_seed"), "git": meta.get("git"), "npz": find_npz(run)})
    return runs


def load_structure(info: dict, mlp: pd.DataFrame, lay: pd.DataFrame, seed: int) -> dict:
    """Operators (per layer) and pairings (per block): fitted from the npz, or synthetic with the
    recorded counts."""
    from mirror.variants import clean_involution

    h, d = int(mlp["h"].iloc[0]), int(mlp["d"].iloc[0])
    if info["npz"] is not None:
        with np.load(info["npz"]) as z:
            A_raw = np.asarray(z["A_layers"])
            perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        A = [clean_involution(a)[0] for a in A_raw]
        return {"A": A, "perms": perms, "source": "fitted operators and pairings", "npz": _rel(info["npz"])}
    rng_seed = 10_000 + seed
    A = [clean_involution(GR.random_c2(d, int(dp), rng_seed + l, fix_ones=False))[0]
         for l, dp in enumerate(lay.sort_values("layer")["d_plus"].astype(int))]
    perms = [random_pairing(h, int(n), seed=rng_seed + 500 + b)
             for b, n in enumerate(mlp.sort_values("block")["n_pairs"].astype(int))]
    return {"A": A, "perms": perms, "source": "synthetic structure with the recorded counts", "npz": ""}


# ----------------------------------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------------------------------
def achieved_counts(A_in, A_out, perm, h: int, d: int, seed: int) -> dict:
    """Entries of the weight tensors of the even/odd MLP built from projected random weights."""
    from mirror.efficient import Basis, build_mirror_mlp
    from mirror.surgery import project_mlp_two

    g = torch.Generator().manual_seed(seed)
    W1, b1 = torch.randn(h, d, generator=g, dtype=torch.float64), torch.randn(h, generator=g, dtype=torch.float64)
    W2, b2 = torch.randn(d, h, generator=g, dtype=torch.float64), torch.randn(d, generator=g, dtype=torch.float64)
    e = project_mlp_two(W1, b1, W2, b2, A_in=A_in, A_out=A_out, perm=perm)
    mod = build_mirror_mlp(e["W1"], e["b1"], e["W2"], e["b2"], torch.nn.GELU(), Basis.of(A_in), Basis.of(A_out),
                           A_in, A_out, perm, align=1)
    return {"params_fc1": int(mod.E1.numel() + mod.O1.numel()), "params_fc2": int(mod.E2.numel() + mod.O2.numel()),
            "params_b1": int(mod.b1.numel()), "params_b2": int(mod.b2.numel()),
            "projection_residual": float(mod.cost["residual"])}


def analyse_run(info: dict, seed: int) -> tuple:
    res = info["run"] / "results"
    mlp = pd.read_csv(res / "e1_mlp.csv").sort_values("block")
    lay = pd.read_csv(res / "e1_layers.csv").sort_values("layer")
    st = load_structure(info, mlp, lay, seed)
    h, d = int(mlp["h"].iloc[0]), int(mlp["d"].iloc[0])
    rows = []
    for _, r in mlp.iterrows():
        b = int(r["block"]) - 1
        A_in, A_out, perm = st["A"][b], st["A"][b + 1], st["perms"][b]
        act = GR.block_action("LR", [A_in], [A_out], [perm])
        pred = GR.mlp_free_params(act, h, d)
        ach = achieved_counts(A_in, A_out, perm, h, d, seed * 1000 + b)
        n_self = int((perm == np.arange(h)).sum())
        tr_in, tr_out = int(round(np.trace(A_in))), int(round(np.trace(A_out)))
        sav1 = 1.0 - ach["params_fc1"] / (h * d)
        sav2 = 1.0 - ach["params_fc2"] / (h * d)
        savm = 1.0 - (ach["params_fc1"] + ach["params_fc2"]) / (2.0 * h * d)
        rows.append({
            "model": info["model"], "fit_seed": info["fit_seed"], "run": _rel(info["run"]),
            "block": b + 1, "h": h, "d": d, "n_pairs": int((perm != np.arange(h)).sum() // 2), "n_self": n_self,
            "d_plus_in": (d + tr_in) // 2, "d_plus_out": (d + tr_out) // 2, "structure": st["source"],
            "npz": st["npz"],
            "free_fc1_pred": pred["free_fc1"], "free_fc2_pred": pred["free_fc2"], "free_b1_pred": pred["free_b1"],
            "free_b2_pred": pred["free_b2"], "tax_fc1_pred": pred["tax_fc1"], "tax_fc2_pred": pred["tax_fc2"],
            "tax_mlp_pred": pred["tax_mlp"],
            "params_fc1_achieved": ach["params_fc1"], "params_fc2_achieved": ach["params_fc2"],
            "params_b1_achieved": ach["params_b1"], "params_b2_achieved": ach["params_b2"],
            "saving_fc1_achieved": sav1, "saving_fc2_achieved": sav2, "saving_mlp_achieved": savm,
            "diff_pp_fc1": 100.0 * (pred["tax_fc1"] - sav1), "diff_pp_fc2": 100.0 * (pred["tax_fc2"] - sav2),
            "diff_pp_mlp": 100.0 * (pred["tax_mlp"] - savm),
            "bias_match": bool(pred["free_b1"] == ach["params_b1"] and pred["free_b2"] == ach["params_b2"]),
            "saving_recorded": float(r["param_saving_frac"]),
            "diff_pp_recorded": 100.0 * (pred["tax_mlp"] - float(r["param_saving_frac"])),
            "counts_match_csv": bool(n_self == int(r["n_self"]) and tr_in == int(r["tr_A_in"])
                                     and tr_out == int(r["tr_A_out"])),
            "projection_residual": ach["projection_residual"],
        })
    sk = []
    sp = res / "e1_surgery.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        for _, r in sdf[sdf["variant"] == "surgery"].iterrows():
            k = int(r["k"])
            sub = rows[:k]
            kept = sum(x["params_fc1_achieved"] + x["params_fc2_achieved"] for x in sub) / (2.0 * h * d * k)
            sk.append({"model": info["model"], "fit_seed": info["fit_seed"], "k": k,
                       "kept_frac_recorded": float(r["mlp_params_kept_frac"]), "kept_frac_achieved": kept,
                       "diff_pp": 100.0 * (kept - float(r["mlp_params_kept_frac"]))})
    return rows, sk


def _clear_caches() -> None:
    """mirror.efficient and mirror.heal cache every eigenbasis by content (d x d each); a run's operators
    are not reused by the next run, so the caches are emptied to keep memory flat."""
    import gc

    from mirror import efficient, heal

    for name in ("_EIG_CACHE",):
        getattr(efficient, name, {}).clear()
    for name in ("_EXACT_CACHE",):
        getattr(heal, name, {}).clear()
    gc.collect()


# ----------------------------------------------------------------------------------------------
# Tiny inputs (test only)
# ----------------------------------------------------------------------------------------------
def make_tiny_runs(root: Path) -> str:
    """Two synthetic run folders in the run_e1.py layout (one with an operators npz, one without)."""
    from mirror.surgery import equivariant_param_count

    d, h, L = 12, 40, 3
    for name, with_npz in (("tinya_s0", True), ("tinyb_s0", False)):
        res = root / name / "results"
        res.mkdir(parents=True, exist_ok=True)
        A = [GR.random_c2(d, 5 + l, 100 + l, fix_ones=False) for l in range(L + 1)]
        perms = [random_pairing(h, 8 + 3 * b, seed=200 + b) for b in range(L)]
        lay = pd.DataFrame({"layer": range(L + 1), "d": d, "d_plus": [5 + l for l in range(L + 1)]})
        mrows = []
        for b in range(L):
            f = int((perms[b] == np.arange(h)).sum())
            ti, to = 2 * (5 + b) - d, 2 * (6 + b) - d
            e1, e2 = equivariant_param_count(h, d, f, ti), equivariant_param_count(h, d, f, to)
            mrows.append({"block": b + 1, "h": h, "d": d, "n_pairs": (h - f) // 2, "n_self": f, "tr_P": f,
                          "tr_A_in": ti, "tr_A_out": to, "eq_params_fc1": e1, "eq_params_fc2": e2,
                          "param_saving_frac": 1 - (e1 + e2) / (2 * h * d)})
        lay.to_csv(res / "e1_layers.csv", index=False)
        pd.DataFrame(mrows).to_csv(res / "e1_mlp.csv", index=False)
        (res / "e1_meta.json").write_text(json.dumps({"operator_mode": "per_layer", "fit_seed": 0,
                                                      "config": {"model": {"name": name[:-3]}}}))
        if with_npz:
            np.savez(res / "e1_operators.npz", A_layers=np.stack(A).astype(np.float32),
                     perms=np.stack(perms).astype(np.int32))
    return str(root / "tiny*_s0")


# ----------------------------------------------------------------------------------------------
def write_summary(out: Path, df: pd.DataFrame, sk: pd.DataFrame, tol: float, git: dict) -> str:
    lines = ["# MUST-01: the law against the achieved saving", "",
             f"Commit {git['git_hash'][:10]}{' (uncommitted changes present)' if git['git_dirty'] else ''}. "
             "Written by scripts/must01_law.py; every number below is read from law_per_layer.csv and "
             "law_surgery_k.csv in this folder.", "",
             "Predicted: free parameters of fc1 and fc2 from the character formula (mirror/groups.py) with the "
             "block's operators and twin matching. Achieved: entries of the weight tensors of the even/odd MLP "
             "(mirror/efficient.py) built from random weights projected onto the same structure. The difference "
             f"is in percentage points of the dense MLP weights. Pre-registered bar: {tol:g} point per block.", ""]
    worst = 0.0
    lines += ["| Model | Fit seeds | Blocks | Structure | Max abs diff, MLP (pp) | Max abs diff, fc1 / fc2 (pp) | "
              "Max abs diff vs recorded (pp) | Biases match | Counts match the run's CSV |",
              "|---|---|---|---|---|---|---|---|---|"]
    for model, g in df.groupby("model", sort=False):
        m = float(g["diff_pp_mlp"].abs().max())
        m1, m2 = float(g["diff_pp_fc1"].abs().max()), float(g["diff_pp_fc2"].abs().max())
        mr = float(g["diff_pp_recorded"].abs().max())
        worst = max(worst, m, m1, m2)
        seeds = ", ".join(str(int(s)) for s in sorted(g["fit_seed"].unique()))
        lines.append(f"| {model} | {seeds} | {len(g)} | {'; '.join(sorted(g['structure'].unique()))} | {m:.2e} | "
                     f"{m1:.2e} / {m2:.2e} | {mr:.2e} | {'yes' if g['bias_match'].all() else 'NO'} | "
                     f"{'yes' if g['counts_match_csv'].all() else 'NO'} |")
    lines += ["", f"Largest absolute per-block difference over every model: {worst:.2e} percentage points "
                  f"({'within' if worst <= tol else 'OUTSIDE'} the {tol:g}-point bar).", ""]
    synth = sorted(df[df["structure"].str.startswith("synthetic")]["model"].unique())
    if synth:
        lines += [f"For {', '.join(synth)} the fitted operators and pairings are not on this machine, so the "
                  "structure is synthetic with the recorded counts (d+ per layer, twin pairs and self-twins per "
                  "block). For these models both numbers derive from the same recorded counts: the check covers "
                  "the arithmetic of the two counting routes, not the stored structure.", ""]
    if len(sk):
        lines += ["Surgery rows recorded by the runs (blocks 1 to k): MLP parameters kept, recorded against "
                  "achieved here.", "", "| Model | Fit seed | k | Recorded | Achieved | Diff (pp) |",
                  "|---|---|---|---|---|---|"]
        for _, r in sk.iterrows():
            lines.append(f"| {r['model']} | {int(r['fit_seed'])} | {int(r['k'])} | {r['kept_frac_recorded']:.6f} | "
                         f"{r['kept_frac_achieved']:.6f} | {r['diff_pp']:.2e} |")
        lines.append("")
    lines += ["The formula and the module count agree by construction when the code is right, so this is a "
              "correctness check, not a finding (PROJECT_NEXT_PHASE.md, MUST-01).", ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    git = git_info()
    globs = make_tiny_runs(out / "tiny_runs") if a.tiny else a.runs
    runs = discover(globs)
    if not runs:
        raise SystemExit(f"no per-layer run_e1.py runs found under {globs}")
    rows, sks = [], []
    for info in runs:
        log(f"{info['model']} fit seed {info['fit_seed']}: {info['run'].name}, "
            f"operators {'from ' + str(info['npz']) if info['npz'] else 'not available (synthetic)'}")
        r, s = analyse_run(info, a.seed)
        rows += r
        sks += s
        _clear_caches()
    df = pd.DataFrame(rows)
    sk = pd.DataFrame(sks)
    df.to_csv(out / "law_per_layer.csv", index=False)
    sk.to_csv(out / "law_surgery_k.csv", index=False)
    write_summary(out, df, sk, a.tol_pp, git)
    per_model = {m: {"max_abs_diff_pp_mlp": float(g["diff_pp_mlp"].abs().max()),
                     "max_abs_diff_pp_fc1": float(g["diff_pp_fc1"].abs().max()),
                     "max_abs_diff_pp_fc2": float(g["diff_pp_fc2"].abs().max()),
                     "max_abs_diff_pp_recorded": float(g["diff_pp_recorded"].abs().max()),
                     "n_blocks": int(len(g)), "fit_seeds": sorted(int(s) for s in g["fit_seed"].unique()),
                     "structure": sorted(g["structure"].unique())}
                 for m, g in df.groupby("model", sort=False)}
    metrics = {"per_model": per_model, "max_abs_diff_pp": float(df[["diff_pp_mlp", "diff_pp_fc1", "diff_pp_fc2"]]
                                                                .abs().max().max()),
               "tol_pp": a.tol_pp, "n_rows": int(len(df))}
    metrics["pass"] = bool(metrics["max_abs_diff_pp"] <= a.tol_pp and df["bias_match"].all())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_meta(out, experiment="MUST-01", config=vars(a), seeds={"projected_weights": a.seed},
               models={m: {"runs": sorted({r["run"] for r in rows if r["model"] == m})} for m in per_model},
               device="cpu", wall_seconds=time.time() - t0,
               extra={"runs": [{"run": _rel(i["run"]), "npz": _rel(i["npz"]) if i["npz"] else None,
                                "run_git": i["git"]} for i in runs]})
    log(f"MUST-01: max abs diff {metrics['max_abs_diff_pp']:.2e} pp over {len(df)} blocks; written to {out}")
    return metrics


if __name__ == "__main__":
    main()

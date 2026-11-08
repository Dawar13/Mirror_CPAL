#!/usr/bin/env python
"""F1, the symmetry tax (PAPER_EXPERIMENTS.md Track F; PROJECT_CONTEXT.md Section 5.3).

Reads protocol v2 runs of scripts/run_e1.py at <runs>/<arm>_s<fit seed>/results/ and reports, for
each model and every available fit seed:
  per layer   d+, d-, d-/d and the held-out errors of A_l, identity and the matched null;
  per block   twin pairs and self-twins, breaking shares s1 (fc1) and s2 (fc2) against the matched
              null, the tax predicted by the formula 1/2 - f (d+ - d-) / (2 h d) (f = self-twins;
              fc1 uses the block's input layer, fc2 its output layer) and the parameter saving
              recorded in e1_mlp.csv;
then the mean and sample standard deviation (n - 1) over fit seeds.

    python scripts/f1_tax.py --runs runs/x2 --out runs/f1
    python scripts/f1_tax.py --runs runs/x2 --arms perlayer_reg,perlayer_plain --hist-blocks 1-6

Writes <out>/f1_layers_by_seed.csv, f1_blocks_by_seed.csv, f1_layers_summary.csv,
f1_blocks_summary.csv, f1_ranges.csv, f1_meta.json, F1_SUMMARY.md and plots/fig2_tax.png.
CPU only; reads CSVs, e1_meta.json and e1_operators.npz (perms, match_scores, self_corr).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.findings import model_colours, model_label, sort_models, tax_formula  # noqa: E402
from mirror.utils import git_info, log  # noqa: E402

DEFAULT_ARMS = "perlayer_reg,perlayer_plain"
LAYER_COLS = ["d_plus", "d_minus", "d_zero", "dminus_frac", "rel_e_A_layer", "rel_e_I", "rel_e_null_layer"]
BLOCK_COLS = ["n_pairs", "n_self", "twin_frac", "mean_score_pairs", "mean_score_self", "s1", "s2", "s1_null",
              "s2_null", "s1_over_null", "s2_over_null", "sb1", "sb2"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/x2", help="directory holding <arm>_s<seed>/ run folders")
    p.add_argument("--out", default="runs/f1")
    p.add_argument("--arms", default=DEFAULT_ARMS, help="comma-separated arm names (default: both per-layer arms)")
    p.add_argument("--hist-blocks", default=None,
                   help="blocks pooled in the twin-score histograms, e.g. 1-6 (default: first half)")
    p.add_argument("--hist-seed", type=int, default=None,
                   help="fit seed whose operators npz feeds the histograms (default: lowest available)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
def discover_runs(root: Path, arms: list[str]) -> list[dict]:
    runs = []
    for arm in arms:
        for d in sorted(root.glob(f"{arm}_s*")):
            m = re.fullmatch(rf"{re.escape(arm)}_s(\d+)", d.name)
            if not m or not d.is_dir():
                continue
            res = d / "results"
            if not ((res / "e1_layers.csv").exists() and (res / "e1_mlp.csv").exists()):
                log(f"skip {d}: e1_layers.csv or e1_mlp.csv missing")
                continue
            meta = json.loads((res / "e1_meta.json").read_text(encoding="utf-8")) if (res / "e1_meta.json").exists() else {}
            dir_seed = int(m.group(1))
            fit_seed = meta.get("fit_seed")
            note = ""
            if fit_seed is None:
                fit_seed, note = dir_seed, "fit seed taken from the folder name (e1_meta.json has none)"
            elif int(fit_seed) != dir_seed:
                note = f"folder says seed {dir_seed} but e1_meta.json says fit_seed {fit_seed}; using e1_meta.json"
                log(f"WARNING {d}: {note}")
            model = ((meta.get("config") or {}).get("model") or {}).get("name") or arm
            runs.append({"arm": arm, "dir": d, "results": res, "model": model, "fit_seed": int(fit_seed),
                         "operator_mode": meta.get("operator_mode", "per_layer"), "meta": meta, "note": note,
                         "input_git_hash": (meta.get("git") or {}).get("git_hash"), "tiny": bool(meta.get("tiny", False))})
    return runs


def _block_operator_traces(run: dict, layers: pd.DataFrame, block: int) -> tuple[int, int, int, int]:
    """(d+, d-) of the operators fc1 and fc2 of `block` (1-based) were tested against in run_e1."""
    if run["operator_mode"] == "shared":
        sh = run["meta"].get("shared") or {}
        dp, dm = int(sh["d_plus"]), int(sh["d_minus"])
        return dp, dm, dp, dm
    li = layers.set_index("layer")
    return (int(li.loc[block - 1, "d_plus"]), int(li.loc[block - 1, "d_minus"]),
            int(li.loc[block, "d_plus"]), int(li.loc[block, "d_minus"]))


def tables_for_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    layers = pd.read_csv(run["results"] / "e1_layers.csv")
    mlp = pd.read_csv(run["results"] / "e1_mlp.csv")
    key = {"model": run["model"], "arm": run["arm"], "fit_seed": run["fit_seed"], "input_git_hash": run["input_git_hash"]}

    lrows = []
    for _, r in layers.iterrows():
        row = {**key, "layer": int(r["layer"]), "d": int(r["d"])}
        row.update({c: r[c] for c in LAYER_COLS if c in r})
        row["null_over_A"] = row["rel_e_null_layer"] / row["rel_e_A_layer"] if row.get("rel_e_A_layer", 0) > 0 else float("nan")
        lrows.append(row)

    brows = []
    for _, r in mlp.iterrows():
        b, h, d, f = int(r["block"]), int(r["h"]), int(r["d"]), int(r["n_self"])
        dp_in, dm_in, dp_out, dm_out = _block_operator_traces(run, layers, b)
        tr_in, tr_out = dp_in - dm_in, dp_out - dm_out
        tax1, tax2 = tax_formula(h, d, f, tr_in), tax_formula(h, d, f, tr_out)
        eq1 = float(r["eq_params_fc1"] if "eq_params_fc1" in r else r["eq_params"])
        eq2 = float(r["eq_params_fc2"] if "eq_params_fc2" in r else r["eq_params"])
        csv_tr_in = int(r["tr_A_in"] if "tr_A_in" in r else r["tr_A"])
        csv_tr_out = int(r["tr_A_out"] if "tr_A_out" in r else r["tr_A"])
        row = {**key, "block": b, "h": h, "d": d}
        row.update({c: r[c] for c in BLOCK_COLS if c in r})
        row.update({
            "d_plus_in": dp_in, "d_minus_in": dm_in, "d_plus_out": dp_out, "d_minus_out": dm_out,
            "tax_formula_fc1": tax1, "tax_formula_fc2": tax2, "tax_formula": 0.5 * (tax1 + tax2),
            "saving_realised_fc1": 1.0 - eq1 / (h * d), "saving_realised_fc2": 1.0 - eq2 / (h * d),
            "saving_realised": float(r["param_saving_frac"]),
        })
        row["tax_minus_saving"] = row["tax_formula"] - row["saving_realised"]
        row["trace_matches_csv"] = bool(tr_in == csv_tr_in and tr_out == csv_tr_out)
        brows.append(row)
    return pd.DataFrame(lrows), pd.DataFrame(brows)


def summarise(df: pd.DataFrame, index: str, skip=("fit_seed", "d", "h")) -> pd.DataFrame:
    """Mean, sample std (n - 1) and number of seeds for every numeric column, per model and index."""
    num = [c for c in df.columns if c not in (index, "model", *skip) and pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])]
    g = df.groupby(["model", index])
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out["n_seeds"] = g["fit_seed"].nunique()
    out["fit_seeds"] = g["fit_seed"].apply(lambda s: ",".join(map(str, sorted(set(s)))))
    return out.reset_index()


def block_ranges(n_blocks: int) -> list[tuple[int, int]]:
    half = n_blocks // 2
    return [(1, half), (half + 1, n_blocks)] if half >= 1 else [(1, n_blocks)]


def range_table(layers: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Per model and fit seed: means over the first and second half of the blocks (layers with the
    same numbers for d-/d); then mean and std over seeds."""
    rows = []
    for (model, seed), bdf in blocks.groupby(["model", "fit_seed"]):
        ldf = layers[(layers["model"] == model) & (layers["fit_seed"] == seed)]
        for lo, hi in block_ranges(int(bdf["block"].max())):
            sel = bdf[(bdf["block"] >= lo) & (bdf["block"] <= hi)]
            lsel = ldf[(ldf["layer"] >= lo) & (ldf["layer"] <= hi)]
            rows.append({"model": model, "fit_seed": seed, "blocks": f"{lo}-{hi}",
                         "tax_formula": sel["tax_formula"].mean(), "saving_realised": sel["saving_realised"].mean(),
                         "twin_frac": sel["twin_frac"].mean(), "s1": sel["s1"].mean(), "s2": sel["s2"].mean(),
                         "s1_null": sel["s1_null"].mean(), "s2_null": sel["s2_null"].mean(),
                         "dminus_frac": lsel["dminus_frac"].mean(),
                         "rel_e_A_layer": lsel["rel_e_A_layer"].mean(), "rel_e_null_layer": lsel["rel_e_null_layer"].mean()})
    per_seed = pd.DataFrame(rows)
    g = per_seed.groupby(["model", "blocks"], sort=False)
    vals = [c for c in per_seed.columns if c not in ("model", "fit_seed", "blocks")]
    agg = g[vals].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = g["fit_seed"].nunique()
    agg["fit_seeds"] = g["fit_seed"].apply(lambda s: ",".join(map(str, sorted(set(s)))))
    return agg.reset_index(), per_seed


# ----------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------
def _parse_range(text: str | None, n: int) -> list[int]:
    if not text:
        return list(range(1, max(1, n // 2) + 1))
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def load_twin_scores(run: dict, blocks: list[int]) -> dict | None:
    path = run["results"] / "e1_operators.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        perms, scores, selfc = z["perms"], z["match_scores"], z["self_corr"]
    pair, self_, pair_self = [], [], []
    for b in blocks:
        if b - 1 >= perms.shape[0]:
            continue
        p, s, sc = perms[b - 1], scores[b - 1], selfc[b - 1]
        idx = np.arange(len(p))
        first = p > idx                       # one entry per pair (S is symmetric)
        pair.append(s[first])
        pair_self.append(sc[p != idx])
        self_.append(s[p == idx])
    return {"pair": np.concatenate(pair), "self": np.concatenate(self_), "pair_self": np.concatenate(pair_self),
            "blocks": blocks, "fit_seed": run["fit_seed"]}


def plot_fig2(blocks_sum: pd.DataFrame, hists: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mirror.report import C_AQUA, C_BLUE, C_INK, C_NULL, _style

    bg = "#fcfcfb"
    models = sort_models(blocks_sum["model"])
    colours = model_colours(models)
    hist_models = [m for m in models if hists.get(m) is not None]
    fig, axes = plt.subplots(1, 1 + len(hist_models), figsize=(5.2 + 4.0 * len(hist_models), 3.9), facecolor=bg)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(bg)

    ax = axes[0]
    for m in models:
        d = blocks_sum[blocks_sum["model"] == m].sort_values("block")
        n = int(d["n_seeds"].max())
        yerr = d["tax_formula_std"].fillna(0).to_numpy() if n > 1 else None
        ax.errorbar(d["block"], d["tax_formula_mean"], yerr=yerr, color=colours[m], marker="o", markersize=6,
                    linewidth=2, capsize=3, label=f"{model_label(m, short=True)}: formula ({n} seed{'s' if n > 1 else ''})")
        ax.plot(d["block"], d["saving_realised_mean"], linestyle="none", marker="o", markersize=10,
                markerfacecolor="none", markeredgecolor=colours[m], markeredgewidth=1.2,
                label=f"{model_label(m, short=True)}: realised saving (CSV)")
    ax.axhline(0.5, color=C_NULL, linestyle="--", linewidth=1.2, label="upper bound 1/2")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("block")
    ax.set_ylabel("redundant share of MLP parameters")
    ax.set_title("Symmetry tax against depth", fontsize=10, color=C_INK, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    rank = [0 if "formula" in t else 1 if "realised" in t else 2 for t in labels]
    order = sorted(range(len(labels)), key=lambda i: (rank[i], i))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=7, loc="lower left")
    _style(ax)

    bins = np.linspace(-1, 1, 51)
    for ax, m in zip(axes[1:], hist_models):
        hs = hists[m]
        ax.hist(hs["pair"], bins=bins, color=C_BLUE, alpha=0.85, label=f"twin pairs ({len(hs['pair'])})")
        ax.hist(hs["self"], bins=bins, color=C_AQUA, alpha=0.85, label=f"self-twins ({len(hs['self'])})")
        ax.hist(hs["pair_self"], bins=bins, histtype="step", color=C_NULL, linewidth=1.5,
                label="paired neurons: own self-score")
        b = hs["blocks"]
        ax.set_title(f"{model_label(m, short=True)}, blocks {b[0]} to {b[-1]}, fit seed {hs['fit_seed']}",
                     fontsize=10, color=C_INK, loc="left")
        ax.set_xlabel("twin score  corr(mirror neuron, matched neuron)")
        ax.set_ylabel("count")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------
def _pm(mean, std, spec=".3f") -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:{spec}}" + (f" ± {std:{spec}}" if np.isfinite(std) else "")


def write_summary(path: Path, ranges: pd.DataFrame, blocks: pd.DataFrame, runs: list[dict], git: dict,
                  args, hist_info: dict) -> None:
    lines = ["# F1: the symmetry tax", ""]
    lines.append(f"Generated by `scripts/f1_tax.py` on {time.strftime('%Y-%m-%d %H:%M')}, commit "
                 f"{git['git_hash'][:10]}{' (uncommitted changes)' if git.get('git_dirty') else ''}. "
                 f"Inputs: `{args.runs}`, arms {', '.join(sorted({r['arm'] for r in runs}))}.")
    hashes = sorted({str(r["input_git_hash"])[:10] for r in runs})
    lines.append(f"The input runs were made at commit(s) {', '.join(hashes)}.")
    lines.append("Plus or minus values are the sample standard deviation over fit seeds (n - 1 in the "
                 "denominator); with one seed no spread is shown. Every number below is read from the CSV "
                 "files in this folder.")
    lines.append("")
    lines.append("## Means over blocks (layers with the same numbers for d-/d)")
    lines.append("")
    lines.append("| model | blocks | fit seeds | tax (formula) | realised saving | twin fraction | s1 | s1 null | "
                 "s2 | s2 null | d-/d |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in sort_models(ranges["model"]):
        for _, r in ranges[ranges["model"] == m].iterrows():
            lines.append(
                f"| {model_label(m)} | {r['blocks']} | {r['fit_seeds']} | {_pm(r['tax_formula_mean'], r['tax_formula_std'])} | "
                f"{_pm(r['saving_realised_mean'], r['saving_realised_std'])} | {_pm(r['twin_frac_mean'], r['twin_frac_std'])} | "
                f"{_pm(r['s1_mean'], r['s1_std'])} | {_pm(r['s1_null_mean'], r['s1_null_std'])} | "
                f"{_pm(r['s2_mean'], r['s2_std'])} | {_pm(r['s2_null_mean'], r['s2_null_std'])} | "
                f"{_pm(r['dminus_frac_mean'], r['dminus_frac_std'])} |")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    diff = float(blocks["tax_minus_saving"].abs().max())
    n_ok, n_all = int(blocks["trace_matches_csv"].sum()), len(blocks)
    lines.append(f"- Formula against realised saving: the largest absolute difference over all blocks, models and "
                 f"seeds is {diff:.2e}. The saving in e1_mlp.csv is computed by the surgery code from the same "
                 "counts (self-twins and operator traces), so this checks the bookkeeping, not the theory.")
    lines.append(f"- Operator traces: d+ - d- from e1_layers.csv equals the trace recorded in e1_mlp.csv for "
                 f"{n_ok} of {n_all} block rows.")
    seeds = {m: sorted({r['fit_seed'] for r in runs if r['model'] == m}) for m in sort_models(r['model'] for r in runs)}
    lines.append("- Fit seeds found: " + "; ".join(f"{model_label(m)}: {', '.join(map(str, s))}" for m, s in seeds.items()) + ".")
    for r in runs:
        if r["note"]:
            lines.append(f"- {r['dir'].name}: {r['note']}.")
    if hist_info:
        lines.append("- Twin-score histograms pool blocks " + "; ".join(
            f"{model_label(m)}: {v['blocks'][0]} to {v['blocks'][-1]} (fit seed {v['fit_seed']}, {len(v['pair'])} pairs, "
            f"{len(v['self'])} self-twins)" for m, v in hist_info.items()) + ".")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("- Tax (formula): 1/2 - f (d+ - d-) / (2 h d) per weight matrix (PROJECT_CONTEXT.md Section 5.3), "
                 "with f the number of self-twins; fc1 uses the operator of the block's input layer, fc2 that of its "
                 "output layer; the block value is the mean of the two (both matrices have h d entries).")
    lines.append("- Realised saving: `param_saving_frac` in e1_mlp.csv, 1 - (equivariant parameters of fc1 and fc2) "
                 "/ (2 h d).")
    lines.append("- s1, s2: breaking shares of fc1 and fc2; null: the matched random involution and random pairing "
                 "(same d+, d-, pairs and self-twins), averaged over the null seeds of run_e1.")
    lines.append("")
    lines.append("Files: `f1_layers_by_seed.csv`, `f1_blocks_by_seed.csv`, `f1_layers_summary.csv`, "
                 "`f1_blocks_summary.csv`, `f1_ranges.csv`, `f1_meta.json`, `plots/fig2_tax.png`.")
    text = "\n".join(lines) + "\n"
    assert "\u2014" not in text
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    root, out = Path(args.runs), Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    git = git_info()
    runs = discover_runs(root, [a.strip() for a in args.arms.split(",") if a.strip()])
    if not runs:
        raise SystemExit(f"no runs found under {root} for arms {args.arms}")
    log(f"F1: {len(runs)} runs: " + ", ".join(f"{r['dir'].name} ({r['model']}, fit seed {r['fit_seed']})" for r in runs))
    dup = pd.Series([(r["model"], r["fit_seed"]) for r in runs]).duplicated()
    if dup.any():
        raise SystemExit("two runs share a model and fit seed; pass --arms to choose one arm per model")

    lt, bt = zip(*(tables_for_run(r) for r in runs))
    layers, blocks = pd.concat(lt, ignore_index=True), pd.concat(bt, ignore_index=True)
    for df in (layers, blocks):
        df["git_hash"] = git["git_hash"]
    layers_sum, blocks_sum = summarise(layers, "layer"), summarise(blocks, "block")
    ranges, ranges_seed = range_table(layers, blocks)
    for df in (layers_sum, blocks_sum, ranges):
        df["git_hash"] = git["git_hash"]

    layers.to_csv(out / "f1_layers_by_seed.csv", index=False)
    blocks.to_csv(out / "f1_blocks_by_seed.csv", index=False)
    layers_sum.to_csv(out / "f1_layers_summary.csv", index=False)
    blocks_sum.to_csv(out / "f1_blocks_summary.csv", index=False)
    ranges.to_csv(out / "f1_ranges.csv", index=False)
    ranges_seed.to_csv(out / "f1_ranges_by_seed.csv", index=False)

    hists = {}
    n_blocks = int(blocks["block"].max())
    hb = _parse_range(args.hist_blocks, n_blocks)
    for m in sort_models(r["model"] for r in runs):
        cands = sorted((r for r in runs if r["model"] == m), key=lambda r: r["fit_seed"])
        if args.hist_seed is not None:
            cands = [r for r in cands if r["fit_seed"] == args.hist_seed]
        for r in cands:
            hs = load_twin_scores(r, hb)
            if hs is not None:
                hists[m] = hs
                break
    plot_fig2(blocks_sum, hists, out / "plots" / "fig2_tax.png")
    write_summary(out / "F1_SUMMARY.md", ranges, blocks, runs, git, args, hists)

    meta = {"experiment": "F1", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "seed": None,
            "seed_note": "no randomness in F1; inputs carry their own seeds (fit seeds listed per run)",
            "args": vars(args), "std": "sample standard deviation over fit seeds (ddof = 1)",
            "runs": [{"dir": str(r["dir"]), "arm": r["arm"], "model": r["model"], "fit_seed": r["fit_seed"],
                      "operator_mode": r["operator_mode"], "input_git_hash": r["input_git_hash"],
                      "run_seed": r["meta"].get("seed"), "tiny": r["tiny"], "note": r["note"]} for r in runs],
            "max_abs_tax_minus_saving": float(blocks["tax_minus_saving"].abs().max()),
            "trace_rows_matching": int(blocks["trace_matches_csv"].sum()), "trace_rows": int(len(blocks)),
            "hist_blocks": hb}
    (out / "f1_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"F1 wrote {out} (max |tax - saving| {meta['max_abs_tax_minus_saving']:.2e})")
    return meta


if __name__ == "__main__":
    main()

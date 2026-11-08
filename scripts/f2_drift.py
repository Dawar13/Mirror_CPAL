#!/usr/bin/env python
"""F2, operator drift with depth (PAPER_EXPERIMENTS.md Track F; claim C2).

For every run of scripts/run_e1.py given with --run (its results/e1_operators.npz and e1_meta.json):
  1. clean each stored float32 A_l to an exact involution (eigen-decompose sym(A_l), keep the signs);
  2. agreement matrix tr(A_l A_m) / d for all layer pairs;
  3. agreement on each layer's top-energy subspace: U_l = eigenvectors of G_l = sum X^T X covering
     95% of the activation energy (k_l of them), agree_energy[l, m] = tr(U_l^T A_l A_m U_l) / k_l
     (row l uses layer l's subspace, so the matrix is not symmetric); also the energy-weighted
     agreement tr(G_l A_l A_m) / tr(G_l);
  4. principal angles between the even subspaces (+1 eigenspaces) of neighbouring layers: mean,
     median and minimum cosine, with a matched null (random involutions with the same d+ per layer,
     seeded). When p + q > d two subspaces of dimensions p and q share at least p + q - d
     directions, so the null mean cosine is far from 0; compare against it.
With several runs of one model (fit seeds) it also reports agreement between seeds at the same
layer, the ceiling that drift should be read against.

    python scripts/f2_drift.py --run runs/x2/perlayer_reg_s0 runs/x2/perlayer_plain_s0 --out runs/f2
    python scripts/f2_drift.py --run runs/x2/perlayer_reg_s{0,1,2} runs/x2/perlayer_plain_s{0,1,2} --out runs/f2

Writes <out>/f2_layers.csv, f2_agreement.csv, f2_neighbours.csv, f2_seed_agreement.csv,
f2_meta.json, plots/fig3a_drift.png and plots/fig3a_drift_energy.png. CPU only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.findings import (agreement, clean_involution, energy_subspace, even_basis,  # noqa: E402
                             model_label, principal_cosines, random_subspace, sha256_file, sort_models,
                             subspace_agreement)
from mirror.utils import git_info, log  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", action="extend", required=True,
                   help="run folders (or e1_operators.npz paths); several allowed")
    p.add_argument("--out", default="runs/f2")
    p.add_argument("--energy", type=float, default=0.95, help="activation energy covered by the top subspace")
    p.add_argument("--seed", type=int, default=0, help="seed of the random-involution nulls")
    p.add_argument("--n-null", type=int, default=3, help="null draws per layer")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
def resolve_run(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".npz":
        npz = p
    elif (p / "results" / "e1_operators.npz").exists():
        npz = p / "results" / "e1_operators.npz"
    else:
        npz = p / "e1_operators.npz"
    if not npz.exists():
        raise FileNotFoundError(f"no e1_operators.npz for {path}")
    meta_p = npz.parent / "e1_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    model = ((meta.get("config") or {}).get("model") or {}).get("name") or npz.parent.parent.name
    fit_seed = meta.get("fit_seed")
    return {"path": str(path), "npz": npz, "model": model, "fit_seed": None if fit_seed is None else int(fit_seed),
            "name": npz.parent.parent.name if npz.parent.name == "results" else npz.parent.name,
            "input_git_hash": (meta.get("git") or {}).get("git_hash"), "tiny": bool(meta.get("tiny", False)),
            "layers_csv": npz.parent / "e1_layers.csv"}


def analyse_run(run: dict, energy: float, seed: int, n_null: int, run_index: int = 0) -> dict:
    with np.load(run["npz"]) as z:
        A_raw = z["A_layers"]
        G = z["G"].astype(np.float64) if "G" in z.files else None
    n, d = A_raw.shape[0], A_raw.shape[1]
    A, E, info = [], [], []
    for l in range(n):
        Ac, V, s, inf = clean_involution(A_raw[l], return_info=True)
        A.append(Ac)
        E.append(even_basis(V, s))
        info.append(inf)
    U, k, cap = [None] * n, [np.nan] * n, [np.nan] * n
    if G is not None:
        for l in range(n):
            U[l], k[l], cap[l], _ = energy_subspace(G[l], energy)

    csv_dp = {}
    if run["layers_csv"].exists():
        lc = pd.read_csv(run["layers_csv"])
        csv_dp = dict(zip(lc["layer"].astype(int), lc["d_plus"].astype(int)))
    key = {"model": run["model"], "run": run["name"], "fit_seed": run["fit_seed"]}
    layers = pd.DataFrame([{**key, "layer": l, "d": d, "d_plus_clean": info[l]["d_plus"], "d_minus_clean": info[l]["d_minus"],
                            "d_plus_csv": csv_dp.get(l, np.nan), "max_eig_dev": info[l]["max_eig_dev"],
                            "rel_change_clean": info[l]["rel_change"], "k_energy": k[l], "energy_captured": cap[l]}
                           for l in range(n)])

    full = np.array([[agreement(A[l], A[m]) for m in range(n)] for l in range(n)])
    if G is not None:
        # tr(U^T A_l A_m U) = tr((U U^T A_l) A_m) and tr(G A_l A_m) = tr((G A_l) A_m): precompute per l.
        PA = [U[l] @ (U[l].T @ A[l]) for l in range(n)]
        GA = [G[l] @ A[l] for l in range(n)]
        trG = [float(np.trace(G[l])) for l in range(n)]
        en = np.array([[np.sum(PA[l] * A[m].T) / k[l] for m in range(n)] for l in range(n)])
        ew = np.array([[np.sum(GA[l] * A[m].T) / trG[l] for m in range(n)] for l in range(n)])
    else:
        en = ew = np.full((n, n), np.nan)
    agree = pd.DataFrame([{**key, "l": l, "m": m, "agree_full": full[l, m], "agree_energy": en[l, m],
                           "agree_energy_weighted": ew[l, m]} for l in range(n) for m in range(n)])

    # Matched nulls: random involutions with the same d+ per layer, seeded, independent across layers
    # and runs. The even subspace of such an involution is a Haar-random d+-dimensional subspace, so it
    # is drawn directly (QR of a Gaussian d x d+ matrix) and the involution is 2 E E^T - I.
    null_E = [[random_subspace(d, info[l]["d_plus"], seed=seed * 100003 + 7919 * run_index + 1009 * l + s)
               for s in range(n_null)] for l in range(n)]
    nulls = [[2.0 * (E0 @ E0.T) - np.eye(d) for E0 in row] for row in null_E]
    nb = []
    for l in range(n - 1):
        m = l + 1
        c = principal_cosines(E[l], E[m])
        c_null = [principal_cosines(null_E[l][s], null_E[m][s]).mean() for s in range(n_null)]
        row = {**key, "l": l, "m": m, "d_plus_l": info[l]["d_plus"], "d_plus_m": info[m]["d_plus"],
               "agree_full": full[l, m], "agree_full_null": float(np.mean([agreement(nulls[l][s], nulls[m][s])
                                                                           for s in range(n_null)])),
               "agree_energy_l": en[l, m], "agree_energy_m": en[m, l],
               "agree_energy_weighted_l": ew[l, m], "agree_energy_weighted_m": ew[m, l],
               "pa_n_angles": int(c.size), "pa_mean_cos": float(c.mean()) if c.size else np.nan,
               "pa_median_cos": float(np.median(c)) if c.size else np.nan,
               "pa_min_cos": float(c.min()) if c.size else np.nan,
               "pa_frac_cos_above_0p9": float((c > 0.9).mean()) if c.size else np.nan,
               "pa_null_mean_cos": float(np.mean(c_null)),
               "pa_trivial_shared_dims": max(0, info[l]["d_plus"] + info[m]["d_plus"] - d)}
        if G is not None:
            row["agree_energy_l_null"] = float(np.mean([subspace_agreement(nulls[l][s], nulls[m][s], U[l])
                                                        for s in range(n_null)]))
        nb.append(row)
    return {"run": run, "A": A, "E": E, "U": U, "G_present": G is not None, "layers": layers, "agree": agree,
            "full": full, "energy": en, "neighbours": pd.DataFrame(nb), "nulls_E": null_E, "d": d, "n": n}


def seed_agreement(res_list: list[dict]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(res_list, 2):
        if a["run"]["model"] != b["run"]["model"]:
            continue
        for l in range(min(a["n"], b["n"])):
            c = principal_cosines(a["E"][l], b["E"][l])
            cn = principal_cosines(a["nulls_E"][l][0], b["nulls_E"][l][0])
            rows.append({"model": a["run"]["model"], "run_a": a["run"]["name"], "run_b": b["run"]["name"],
                         "fit_seed_a": a["run"]["fit_seed"], "fit_seed_b": b["run"]["fit_seed"], "layer": l,
                         "agree_full": agreement(a["A"][l], b["A"][l]),
                         "agree_energy_a": subspace_agreement(a["A"][l], b["A"][l], a["U"][l]) if a["G_present"] else np.nan,
                         "pa_mean_cos": float(c.mean()) if c.size else np.nan,
                         "pa_null_mean_cos": float(cn.mean()) if cn.size else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential ramp


def _heatmaps(fig, axes, res_list, models, key, title):
    from matplotlib.colors import LinearSegmentedColormap

    from mirror.report import C_INK, C_MUTED

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    mats = {}
    for m in models:
        stack = np.stack([r[key] for r in res_list if r["run"]["model"] == m])
        mats[m] = (np.nanmean(stack, axis=0), stack.shape[0])
    finite = [M[np.isfinite(M)] for M, _ in mats.values()]
    vmin = min(0.0, min((float(f.min()) for f in finite if f.size), default=0.0))
    im = None
    for ax, m in zip(axes, models):
        M, ns = mats[m]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1.0, origin="upper", interpolation="nearest")
        ax.set_title(f"{model_label(m, short=True)}: {title} ({ns} seed{'s' if ns > 1 else ''})",
                     fontsize=9, color=C_INK, loc="left")
        ax.set_xlabel("layer m")
        ax.set_ylabel("layer l")
        ticks = np.arange(M.shape[0])
        ax.set_xticks(ticks[::2])
        ax.set_yticks(ticks[::2])
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    return im


def plot_fig3a(res_list: list[dict], nb_all: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mirror.report import C_AQUA, C_BLUE, C_INK, C_NULL, C_ORANGE, _style

    bg = "#fcfcfb"
    models = sort_models(r["run"]["model"] for r in res_list)
    nm = len(models)

    # Figure 3a: full-space agreement heatmaps + neighbour line plot.
    # Own colourbar column (with a spacer) so it never collides with the line plot's axis label.
    fig, all_axes = plt.subplots(1, nm + 3, figsize=(3.9 * nm + 6.4, 4.0), facecolor=bg,
                                 gridspec_kw={"width_ratios": [1] * nm + [0.05, 0.28, 1.45], "wspace": 0.3})
    axes, cax, spacer = list(all_axes[:nm]) + [all_axes[nm + 2]], all_axes[nm], all_axes[nm + 1]
    spacer.axis("off")
    for ax in axes:
        ax.set_facecolor(bg)
    im = _heatmaps(fig, axes[:nm], res_list, models, "full", "tr(A_l A_m) / d")
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8, colors=C_INK)
    cb.outline.set_visible(False)

    ax = axes[nm]
    styles = {m: (ls, mk) for m, ls, mk in zip(models, ("-", "--", ":"), ("o", "s", "^"))}
    metrics = [("agree_full", "tr(A_l A_l+1) / d", C_BLUE),
               ("agree_energy_l", "same, on layer l's 95% energy subspace", C_ORANGE),
               ("pa_mean_cos", "even subspaces: mean principal cosine", C_AQUA),
               ("pa_null_mean_cos", "matched random null of the cosine", C_NULL)]
    for m in models:
        d = nb_all[nb_all["model"] == m]
        g = d.groupby("l")
        x = np.array(sorted(d["l"].unique())) + 0.5
        ls, mk = styles[m]
        for col, label, colour in metrics:
            if d[col].isna().all():
                continue
            mean, std = g[col].mean().to_numpy(), g[col].std().fillna(0).to_numpy()
            ns = int(g["fit_seed"].nunique().max()) if d["fit_seed"].notna().any() else 1
            ax.errorbar(x, mean, yerr=std if ns > 1 else None, linestyle=ls, marker=mk, markersize=5, linewidth=2,
                        capsize=2, color=colour, label=f"{model_label(m, short=True)}: {label}")
    ax.set_xlabel("layer (pair l, l+1 drawn between them)")
    ax.set_ylabel("agreement or cosine")
    ax.set_title("Neighbouring layers", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    _style(ax)
    fig.savefig(out_dir / "fig3a_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Companion: agreement restricted to each row layer's top-energy subspace.
    with_g = [r for r in res_list if r["G_present"]]
    models_g = sort_models(r["run"]["model"] for r in with_g)
    if with_g:
        fig, axes = plt.subplots(1, len(models_g), figsize=(4.4 * len(models_g) + 0.8, 4.0), facecolor=bg)
        axes = np.atleast_1d(axes)
        for ax in axes:
            ax.set_facecolor(bg)
        im = _heatmaps(fig, axes, with_g, models_g, "energy", "tr(U_l' A_l A_m U_l) / k_l")
        fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
        fig.savefig(out_dir / "fig3a_drift_energy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    git = git_info()
    runs = [resolve_run(p) for p in args.run]
    res_list = []
    for i, r in enumerate(runs):
        log(f"F2: {r['name']} ({r['model']}, fit seed {r['fit_seed']})")
        res_list.append(analyse_run(r, args.energy, args.seed, args.n_null, run_index=i))
        if not res_list[-1]["G_present"]:
            log(f"WARNING {r['name']}: no G in the npz (run predates ResidualStats.G); energy columns are NaN")

    layers = pd.concat([x["layers"] for x in res_list], ignore_index=True)
    agree = pd.concat([x["agree"] for x in res_list], ignore_index=True)
    nb = pd.concat([x["neighbours"] for x in res_list], ignore_index=True)
    seeds = seed_agreement(res_list)
    for df in (layers, agree, nb, seeds):
        df["git_hash"] = git["git_hash"]
        df["seed"] = args.seed
    layers.to_csv(out / "f2_layers.csv", index=False)
    agree.to_csv(out / "f2_agreement.csv", index=False)
    nb.to_csv(out / "f2_neighbours.csv", index=False)
    seeds.to_csv(out / "f2_seed_agreement.csv", index=False)
    plot_fig3a(res_list, nb, out / "plots")

    meta = {"experiment": "F2", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "seed": args.seed,
            "args": vars(args),
            "inputs": [{"path": r["path"], "npz": str(r["npz"]), "npz_sha256": sha256_file(r["npz"]),
                        "model": r["model"], "fit_seed": r["fit_seed"], "input_git_hash": r["input_git_hash"],
                        "tiny": r["tiny"], "G_present": x["G_present"]} for r, x in zip(runs, res_list)],
            "cleaning": {"max_eig_dev": float(layers["max_eig_dev"].max()),
                         "max_rel_change": float(layers["rel_change_clean"].max()),
                         "d_plus_mismatch_rows": int((layers["d_plus_csv"].notna()
                                                      & (layers["d_plus_csv"] != layers["d_plus_clean"])).sum())},
            "definitions": {
                "agree_full": "tr(A_l A_m) / d, operators cleaned to exact involutions",
                "agree_energy": f"tr(U_l^T A_l A_m U_l) / k_l, U_l = top eigenvectors of G_l covering {args.energy:g} of tr(G_l)",
                "agree_energy_weighted": "tr(G_l A_l A_m) / tr(G_l)",
                "pa_mean_cos": "mean cosine of the principal angles between the +1 eigenspaces of A_l and A_l+1",
                "nulls": f"{args.n_null} random involutions per layer and run with the same d+: even subspace "
                         f"Haar-random (mirror.findings.random_subspace), operator 2 E E^T - I; base seed {args.seed}"}}
    (out / "f2_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"F2 wrote {out}; cleaning: max |eig| deviation {meta['cleaning']['max_eig_dev']:.2e}")
    return meta


if __name__ == "__main__":
    main()

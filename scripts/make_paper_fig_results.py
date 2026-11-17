#!/usr/bin/env python
"""Paper results figure: healing curves (a) and accuracy against size (b) for DINOv2-B/14 with registers.

    .venv/Scripts/python.exe scripts/make_paper_fig_results.py

Reads only these files (every plotted value comes from them; nothing is typed in by hand):
  results/MUST-09/curves.csv       validation k-NN top-1 per healing step, method and seed (panel a); cross-checked
                                   point by point against runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/metrics.json
                                   ("curve") when those files are present
  results/MUST-09/summary.csv      final k-NN top-1 (step 780, bf16, all 50,000 validation images) and params_model
                                   (the paper's parameter cost model) of each healed run (panel b)
  results/CORE-04/scale/knn.csv    k-NN top-1 and parameter count of the released DINOv2 models with registers, S, B,
                                   L and g (panel b, grey line)
  results/MUST-10/flipavg.csv      the flip-averaged original (variant original, mode flipavg, two forward passes per
                                   image) (panel b, open star)

Panel (a): mean over seeds as a line with markers (Mirror Surgery and SVD-LLM have 3 seeds, the others 1). The left
axis shows the whole range (0 to 80) so the low starting points of plain SVD and random structure stay visible; the
narrow axis on the right zooms into steps 100 to 780 where the methods end close together, and shows the single seeds
as faint thin lines. Steps 0 and 780 are evaluated on all 50,000 validation images and steps 100 and 300 on a
10,000-image validation subset (n_query and full_validation in curves.csv); the script prints both per step.

Panel (b): parameters (millions, log axis) against k-NN top-1. Released models: params_total and top1 of
CORE-04. Healed models: params_model (one value per method, the same for every seed) and the mean of knn_top1 over
seeds. Filled markers are models that are exactly flip-invariant in one forward pass (Mirror Surgery and random
structure); open markers are not (the flip-averaged original is invariant only with two passes, so its star is
open). A point below the y range (random structure) is shown as a small down arrow on the bottom edge with its value.
Both panels use the same colour and marker per method; every series is named by a direct label.

Colours are the shared muted palette of scripts/paper_style.py (ROLE). They fail the dataviz validator's chroma and
colour-vision checks (muted by design; the closest pair is plain SVD sage against random structure grey), so identity
never rests on colour alone: each method also has its own marker shape and a direct label.

Writes paper/figures/fig_results.pdf (vector) and fig_results.png (300 dpi), 6.0 x 2.3 inches, and prints every
plotted value with its source path, plus the git hash of this checkout. No randomness.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, ROLE, apply  # noqa: E402

MODEL = "dinov2_vitb14_reg"
CURVES = ROOT / "results" / "MUST-09" / "curves.csv"
SUMMARY = ROOT / "results" / "MUST-09" / "summary.csv"
RUNS = ROOT / "runs" / "MUST-09" / MODEL
FAMILY = ROOT / "results" / "CORE-04" / "scale" / "knn.csv"
FLIPAVG = ROOT / "results" / "MUST-10" / "flipavg.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_results"

# method key in the result files: (ROLE key, label, marker, exactly flip-invariant in one pass); order = drawing order
METHODS = {
    "mirror_exact": ("mirror", "Mirror Surgery", "o", True),
    "svdllm": ("svdllm", "SVD-LLM", "s", False),
    "slicegpt": ("slicegpt", "SliceGPT-style", "D", False),
    "svd": ("svd", "plain SVD", "^", False),
    "random_structure": ("random", "random structure", "v", True),
}
FAMILY_MODELS = {"dinov2_vits14_reg": "S", "dinov2_vitb14_reg": "B", "dinov2_vitl14_reg": "L",
                 "dinov2_vitg14_reg": "g"}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.3
AX_BOTTOM, AX_TOP = 0.43, 2.04                       # inches from the bottom of the figure
A_MAIN = (0.43, 1.37)                                # (left, width) in inches
A_ZOOM = (2.03, 0.45)
B_AX = (3.72, 2.23)
A_YLIM, A_XLIM = (0, 80), (-20, 800)
ZOOM_FROM = 100                                      # first step shown in the zoom
Z_YLIM, Z_XLIM = (76.7, 78.6), (70, 810)
B_YLIM, B_XLIM = (73.0, 81.0), (9.0, 1700.0)
# label offsets in points: family (dx, dy, ha, va); healed (dx, dy, ha); note (x, y, row step) in axes fraction
FAM_LABEL = {"S": (4, -2, "left", "top"), "B": (6, -2, "left", "center"), "L": (0, 5, "center", "bottom"),
             "g": (0, 5, "center", "bottom")}
HEALED_LABEL = {"mirror_exact": (-5, 0, "right"), "svdllm": (16, 0.5, "left"), "slicegpt": (16, -6.5, "left"),
                "svd": (16, -2.5, "left")}
NOTE_XY = (0.44, 0.355, 0.10)
FS = 8                                               # labels and ticks
FS_TITLE = 9
MS = 4.0                                             # marker size (points)
LW = 1.1                                             # mean line width
LW_SEED = 0.5


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def marker_style(key: str) -> dict:
    role, _, marker, invariant = METHODS[key]
    col = ROLE[role]
    if invariant:
        return dict(marker=marker, markersize=MS, markerfacecolor=col, markeredgecolor=C["paper"],
                    markeredgewidth=0.5)
    return dict(marker=marker, markersize=MS - 0.4, markerfacecolor=C["paper"], markeredgecolor=col,
                markeredgewidth=0.9)


# ---------------------------------------------------------------------------------------------- data
def load_curves() -> pd.DataFrame:
    df = pd.read_csv(CURVES)
    df = df[(df["model"] == MODEL) & (df["method"].isin(METHODS))].copy()
    df["step"] = df["step"].astype(int)
    df["seed"] = df["seed"].astype(int)
    # cross-check every point against the run's own metrics.json, when the run folder is on this machine
    n_checked = 0
    for (method, seed), sub in df.groupby(["method", "seed"]):
        mj = RUNS / method / f"s{seed}" / "metrics.json"
        if not mj.exists():
            continue
        curve = {int(r["step"]): float(r["knn_top1"]) for r in json.loads(mj.read_text())["curve"]}
        for _, r in sub.iterrows():
            assert abs(curve[int(r["step"])] - float(r["knn_top1"])) < 1e-9, (method, seed, r["step"])
            n_checked += 1
    print(f"curves.csv cross-checked against runs/.../metrics.json: {n_checked} of {len(df)} points agree")
    # evaluation set per step
    for step, sub in df.groupby("step"):
        nq = sorted(set(sub["n_query"].astype(int)))
        full = sorted(set(sub["full_validation"].astype(str)))
        print(f"  step {step:>3}: n_query {nq}, full_validation {full}")
    return df


def load_summary() -> pd.DataFrame:
    s = pd.read_csv(SUMMARY)
    return s[(s["model"] == MODEL) & (s["method"].isin(METHODS))].copy()


# ---------------------------------------------------------------------------------------------- panel (a)
def axes_at(fig, left: float, width: float):
    return fig.add_axes([left / FIG_W, AX_BOTTOM / FIG_H, width / FIG_W, (AX_TOP - AX_BOTTOM) / FIG_H])


def panel_a(fig, curves: pd.DataFrame) -> None:
    ax = axes_at(fig, *A_MAIN)
    az = axes_at(fig, *A_ZOOM)
    print("\nPanel (a): validation k-NN top-1 (%) against healing step; source", rel(CURVES))
    means = {}
    for key in METHODS:
        role, label, _, _ = METHODS[key]
        sub = curves[curves["method"] == key]
        m = sub.groupby("step")["knn_top1"].mean().sort_index()
        seeds = sorted(sub["seed"].unique())
        means[key] = m
        col = ROLE[role]
        for s in seeds:
            ss = sub[sub["seed"] == s].sort_values("step")
            print(f"  {label:<17} seed {s}: " + ", ".join(f"step {int(a)} {b:.2f}" for a, b in
                                                          zip(ss["step"], ss["knn_top1"])))
            if len(seeds) > 1:  # single seeds only in the zoom, where they differ visibly
                zs = ss[ss["step"] >= ZOOM_FROM]
                az.plot(zs["step"], zs["knn_top1"], color=col, lw=LW_SEED, alpha=0.4, zorder=2,
                        solid_capstyle="round", solid_joinstyle="round")
        print(f"  {label:<17} mean of {len(seeds)}: " + ", ".join(f"step {a} {b:.2f}" for a, b in m.items()))
        ax.plot(m.index, m.values, color=col, lw=LW, zorder=3, solid_joinstyle="round", solid_capstyle="round",
                **marker_style(key))
        mz = m[m.index >= ZOOM_FROM]
        az.plot(mz.index, mz.values, color=col, lw=LW, zorder=3, solid_joinstyle="round",
                solid_capstyle="round", **marker_style(key))

    steps = sorted(curves["step"].unique())
    end = max(steps)
    # main axes: the whole range
    ax.set_xlim(*A_XLIM)
    ax.set_ylim(*A_YLIM)
    ax.xaxis.set_major_locator(FixedLocator(steps))
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80]))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("healing step")
    ax.set_ylabel("k-NN top-1 (%)")
    ax.set_title("(a) Healing", loc="left", fontsize=FS_TITLE, pad=6)
    y_r = float(means["random_structure"].loc[end])
    ax.annotate("random\nstructure", (end, y_r), xytext=(-1, -15), textcoords="offset points", ha="right",
                va="top", fontsize=FS, color=C["ink2"], linespacing=1.0)

    # zoom axes: steps from ZOOM_FROM on, where the methods end close together
    az.set_xlim(*Z_XLIM)
    az.set_ylim(*Z_YLIM)
    az.xaxis.set_major_locator(FixedLocator([s for s in steps if s >= ZOOM_FROM and s != 300]))
    az.xaxis.set_minor_locator(FixedLocator([300]))
    az.yaxis.set_major_locator(FixedLocator([77, 78]))
    az.grid(axis="x", visible=False)
    az.tick_params(axis="x", which="minor", length=2, width=0.6, color=C["ink2"])
    az.set_title("zoom", loc="left", fontsize=FS, color=C["ink2"], pad=6)
    for key in ("mirror_exact", "svdllm", "slicegpt", "svd"):
        y = float(means[key].loc[end])
        az.annotate(METHODS[key][1], (end, y), xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=FS, color=C["ink2"], annotation_clip=False)


# ---------------------------------------------------------------------------------------------- panel (b)
def panel_b(fig, summary: pd.DataFrame) -> None:
    ax = axes_at(fig, *B_AX)
    ax.set_xscale("log")
    ax.set_xlim(*B_XLIM)
    ax.set_ylim(*B_YLIM)
    print("\nPanel (b): parameters (millions) against validation k-NN top-1 (%)")

    # released family: grey line, open points (not flip-invariant)
    fam = pd.read_csv(FAMILY)
    fam = fam[fam["model"].isin(FAMILY_MODELS)].copy()
    fam["params_m"] = fam["params_total"].astype(float) / 1e6
    fam = fam.sort_values("params_m")
    ax.plot(fam["params_m"], fam["top1"], color=C["grey"], lw=0.8, zorder=2, solid_joinstyle="round")
    ax.plot(fam["params_m"], fam["top1"], ls="none", marker="o", markersize=MS - 0.6,
            markerfacecolor=C["paper"], markeredgecolor=ROLE["original"], markeredgewidth=0.8, zorder=3)
    for _, r in fam.iterrows():
        tag = FAMILY_MODELS[r["model"]]
        print(f"  original {tag} ({r['model']}): params {r['params_m']:.2f}M, top-1 {r['top1']:.2f}  "
              f"[{rel(FAMILY)}]")
        dx, dy, ha, va = FAM_LABEL[tag]
        ax.annotate(tag, (r["params_m"], r["top1"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=FS, color=C["ink2"])
    top = fam.iloc[-2:]  # the two largest: name the line under the segment between them
    mid_x = float(np.sqrt(top["params_m"].iloc[0] * top["params_m"].iloc[1]))
    mid_y = float(top["top1"].mean())
    ax.annotate("released models", (mid_x, mid_y), xytext=(0, -8), textcoords="offset points", ha="center",
                va="top", fontsize=FS, color=C["ink2"])

    # flip-averaged original: open star
    fa = pd.read_csv(FLIPAVG)
    fa = fa[(fa["model"] == MODEL) & (fa["variant"] == "original") & (fa["mode"] == "flipavg")].iloc[0]
    fa_p, fa_y = float(fa["params"]) / 1e6, float(fa["knn_top1"])
    print(f"  flip-averaged original: params {fa_p:.2f}M, top-1 {fa_y:.2f}, forwards per image "
          f"{int(fa['forwards_per_image'])}  [{rel(FLIPAVG)}]")
    ax.plot([fa_p], [fa_y], ls="none", marker="*", markersize=MS + 3.2, markerfacecolor="none",
            markeredgecolor=ROLE["original"], markeredgewidth=0.6, zorder=4)
    ax.annotate(f"flip-averaged\n({int(fa['forwards_per_image'])} passes)", (fa_p, fa_y), xytext=(-16, 9),
                textcoords="offset points", ha="right", va="bottom", fontsize=FS, color=C["ink2"],
                linespacing=1.0, arrowprops=dict(arrowstyle="-", lw=0.4, color=C["grey"], shrinkA=0.5,
                                                 shrinkB=4.5))

    # healed models of DINOv2-B/14 with registers
    for key in METHODS:
        role, label, marker, invariant = METHODS[key]
        sub = summary[summary["method"] == key]
        assert sub["params_model"].nunique() == 1, key
        y = float(sub["knn_top1"].mean())
        p = float(sub["params_model"].iloc[0]) / 1e6
        print(f"  healed {label}: params {p:.3f}M, top-1 {y:.2f} (mean of {len(sub)} seed(s): "
              f"{', '.join(f'{v:.2f}' for v in sub['knn_top1'])})  [{rel(SUMMARY)}]")
        st = marker_style(key)
        if y < B_YLIM[0]:  # below the y range: a down-pointing arrow on the bottom edge, with the value
            ax.annotate("", (p, B_YLIM[0]), xytext=(0, 4.5), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.6,
                                        color=ROLE[role], shrinkA=0, shrinkB=0))
            ax.annotate(f"{label} ({y:.1f})", (p, B_YLIM[0]), xytext=(5, 2.5), textcoords="offset points",
                        ha="left", va="center", fontsize=FS, color=C["ink2"])
            continue
        ax.plot([p], [y], ls="none", zorder=5, **st)
        dx, dy, ha = HEALED_LABEL[key]
        leader = None if abs(dy) < 0.5 and abs(dx) < 7 else dict(arrowstyle="-", lw=0.4, color=C["grey"],
                                                                  shrinkA=0.5, shrinkB=2.5)
        ax.annotate(label, (p, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=FS, color=C["ink2"], arrowprops=leader)

    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([74, 76, 78, 80]))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("k-NN top-1 (%)")
    ax.set_title("(b) Accuracy against size", loc="left", fontsize=FS_TITLE, pad=6)

    # filled / open note (bottom right); "in one forward pass" because the open star is invariant only with two
    x0, y0, dy_row = NOTE_XY
    ax.annotate("in one forward pass:", (x0, y0), xycoords="axes fraction", xytext=(-2.5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=FS, color=C["ink2"])
    for i, (filled, text) in enumerate(((True, "exactly flip-invariant"), (False, "not"))):
        yy = y0 - (i + 1) * dy_row
        ax.plot([x0], [yy], transform=ax.transAxes, ls="none", marker="o", markersize=MS - 0.6,
                markerfacecolor=C["ink2"] if filled else C["paper"], markeredgecolor=C["ink2"],
                markeredgewidth=0.5 if filled else 0.8, clip_on=False)
        ax.annotate(text, (x0, yy), xycoords="axes fraction", xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=FS, color=C["ink2"])


def main() -> None:
    apply()
    plt.rcParams.update({"xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS,
                         "savefig.bbox": None})
    print("git hash:", git_hash())
    curves = load_curves()
    summary = load_summary()
    # the final curve point and the summary row are the same evaluation
    for _, r in summary.iterrows():
        c = curves[(curves["method"] == r["method"]) & (curves["seed"] == r["seed"]) &
                   (curves["step"] == curves["step"].max())]
        assert abs(float(c["knn_top1"].iloc[0]) - float(r["knn_top1"])) < 1e-9, r["method"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, curves)
    panel_b(fig, summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches=None)
    print("\nwrote", rel(OUT_DIR / f"{OUT_STEM}.pdf"), "and", rel(OUT_DIR / f"{OUT_STEM}.png"))


if __name__ == "__main__":
    main()

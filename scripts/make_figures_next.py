#!/usr/bin/env python
"""Paper figures and tables of the next phase, built only from the collector's outputs
(PROJECT_NEXT_PHASE.md Section 12; scripts/collect_next_phase.py writes the inputs).

    python scripts/collect_next_phase.py          # first: FINAL_RESULTS.md, PAPER_NUMBERS.json, results/paper_next/data/
    python scripts/make_figures_next.py           # then: results/paper_next/figures/ and tables/

Inputs: <paper-dir>/data/*.csv (and fig1_teaser.npz) and PAPER_NUMBERS.json. Nothing is computed here beyond what a
figure shows: means over seeds and the differences a panel plots. No randomness. A figure, panel or table whose input
is missing is skipped and listed with the reason in FIGURES.md; no placeholder data is ever drawn.

Outputs under <paper-dir>:
  figures/  fig1_twins, fig2_tax_depth, fig3_controlled, fig4_visibility_stacking, fig5_accuracy_params
            (PNG at 300 dpi and PDF, two-column paper sizes: single column 3.3 in, double column 6.9 in)
  tables/   table1_healed, table2_downstream, table3a_compressors, table3b_components, table3c_earlier_ablations,
            table4_generality, appendix_a1_exactness, appendix_a2_law, appendix_a3_prereg, appendix_a4_failed
            (Markdown, CSV and LaTeX with booktabs)
  FIGURES.md (what was made, from which data file, what was skipped and why) and figures_meta.json.

Style (STYLE below): one sans font (Arial, then Liberation Sans, then DejaVu Sans), 7 pt; recessive hairline grid and
axes; the validated categorical palette of the dataviz reference (the same hex values as mirror/report.py and
scripts/make_figures.py), assigned in fixed order so a colour follows its entity (model, group, method) in every
figure; every colour also carries a marker shape, so the figures read in greyscale and under colour-vision deficiency.
Zero-shot results are hollow markers, healed results filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------- style
# Categorical slots 1 to 8 (light surface), validated with the dataviz skill's validator (adjacent pairs pass the CVD
# and normal-vision floors; slots 3 to 5 sit below 3:1 contrast, so every series also has a marker and a legend).
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_NULL, C_OTHER, SURFACE = "#8a8985", "#c3c2b7", "#ffffff"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE, DOUBLE = 3.3, 6.9
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": C_AXIS, "axes.labelcolor": C_INK, "axes.titlecolor": C_INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": C_GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "xtick.color": C_INK2, "ytick.color": C_INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": C_INK,
}
MODELS = {   # colour and marker follow the model in every figure (as scripts/collect_results.py)
    "dinov2_vitb14_reg": ("DINOv2-B reg", C_BLUE, "o"), "dinov2_vitb14": ("DINOv2-B", C_ORANGE, "s"),
    "deit3_b16": ("DeiT-III B", C_AQUA, "^"), "mae_b16": ("MAE B", C_YELLOW, "D"), "siglip2_b16": ("SigLIP2 B", C_MAGENTA, "v"),
    "dinov2_vits14_reg": ("DINOv2-S reg", C_GREEN, "P"), "dinov2_vitl14_reg": ("DINOv2-L reg", C_VIOLET, "X"),
    "dinov2_vitg14_reg": ("DINOv2-g reg", C_RED, "*"),
}
OTHER_MARKERS = ["h", "p", "8", "d", "<", ">"]
GROUPS = {"LR": ("left-right (LR)", C_BLUE, "o"), "UD": ("upside-down (UD)", C_ORANGE, "s"),
          "R90": ("rotation by 90 (R90)", C_AQUA, "^"), "R180": ("rotation by 180 (R180)", C_YELLOW, "D"),
          "Klein": ("LR and UD (Klein)", C_VIOLET, "X")}
CONDITIONS = {"none": ("none", C_NULL, "o"), "LR": ("LR flips", C_BLUE, "o"), "UD": ("UD flips", C_ORANGE, "s"),
              "LRUD": ("LR and UD flips", C_VIOLET, "X"), "R90": ("90-degree rotations", C_AQUA, "^")}
METHOD_LABEL = {"original": "Original model", "mirror_exact": "Mirror-exact", "mirror_local": "Mirror-local (ablation)",
                "svd": "Plain SVD", "svdllm": "SVD-LLM-style whitening", "asvd_scale": "ASVD-style scaling",
                "slicegpt": "SliceGPT-style slicing", "flap": "FLAP-style pruning", "zipit": "ZipIt-style merging",
                "pair_actcorr": "Pairing by same-image correlation", "pair_wcos": "Pairing by weight cosine",
                "pair_random": "Random pairing", "random_structure": "Random structure"}
METHOD_ORDER = list(METHOD_LABEL)


def use_style() -> None:
    plt.rcParams.update(STYLE)


def log(msg: str) -> None:
    print(f"[figures_next {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_style(model: str, others: dict) -> tuple:
    if model in MODELS:
        return MODELS[model]
    if model not in others:
        others[model] = (model, C_OTHER, OTHER_MARKERS[len(others) % len(OTHER_MARKERS)])
    return others[model]


# ---------------------------------------------------------------------------------------------- bookkeeping
class Book:
    def __init__(self, paper_dir: Path, numbers: dict | None):
        self.dir = Path(paper_dir)
        self.data_dir, self.fig_dir, self.tab_dir = self.dir / "data", self.dir / "figures", self.dir / "tables"
        self.numbers = numbers or {}
        self.entries: list[dict] = []
        self.inputs: dict[str, str] = {}

    def data(self, name: str) -> pd.DataFrame | None:
        p = self.data_dir / f"{name}.csv"
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            return None
        self.inputs[p.name] = sha256(p)
        return df if len(df) else None

    def number(self, key: str):
        e = (self.numbers.get("numbers") or {}).get(key)
        return None if e is None else e.get("value")

    def save_fig(self, fig, name: str, sources: list, caption: str, skipped: list | None = None) -> None:
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        for ext in ("png", "pdf"):
            p = self.fig_dir / f"{name}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            outs.append(p.name)
        plt.close(fig)
        self.entries.append({"kind": "figure", "name": name, "status": "made", "files": outs, "sources": sources,
                             "caption": caption, "skipped_panels": skipped or []})
        log(f"wrote {name}")

    def skip(self, kind: str, name: str, reason: str) -> None:
        self.entries.append({"kind": kind, "name": name, "status": "skipped", "reason": reason})
        log(f"skipped {name}: {reason}")

    def save_table(self, name: str, title: str, df: pd.DataFrame, sources: list, notes: str = "",
                   align: str | None = None) -> None:
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df.to_csv(self.tab_dir / f"{name}.csv", index=False)
        cells = df.astype(object).where(df.notna(), "")
        md = [f"# {title}", ""] + ([notes, ""] if notes else [])
        md.append("| " + " | ".join(str(c) for c in df.columns) + " |")
        md.append("|" + "|".join("---" for _ in df.columns) + "|")
        for _, r in cells.iterrows():
            md.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |")
        md += ["", "Sources: " + ", ".join(f"`{s}`" for s in sources), ""]
        (self.tab_dir / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
        cols = align or ("l" + "r" * (len(df.columns) - 1))
        tex = ["% " + title, "% generated by scripts/make_figures_next.py from " + ", ".join(sources),
               "\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{latex(title)}}}",
               f"\\label{{tab:{name}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
               " & ".join(latex(str(c)) for c in df.columns) + " \\\\", "\\midrule"]
        for _, r in cells.iterrows():
            tex.append(" & ".join(latex(str(v)) for v in r) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        if notes:
            tex += [f"\\par\\smallskip\\footnotesize {latex(notes)}"]
        tex += ["\\end{table}", ""]
        (self.tab_dir / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
        self.entries.append({"kind": "table", "name": name, "status": "made", "files": [f"{name}.{e}" for e in
                                                                                        ("md", "csv", "tex")],
                             "sources": sources, "caption": title})
        log(f"wrote table {name}")


def latex(s: str) -> str:
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{",
           "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "±": "$\\pm$", "<=": "$\\le$",
           ">=": "$\\ge$"}
    out = s
    for k in ("\\", "&", "%", "$", "#", "_", "{", "}", "~", "^"):
        out = out.replace(k, rep[k])
    for k in ("±", "<=", ">="):
        out = out.replace(k, rep[k])
    return out


def f2(x, nd=2) -> str:
    v = num(x)
    return "" if v is None else f"{v:.{nd}f}"


def pm(mu, sd, n=None, nd=2) -> str:
    if num(mu) is None:
        return ""
    s = f"{num(mu):.{nd}f}"
    if num(sd) is not None:
        s += f" ± {num(sd):.{nd}f}"
    if n is not None and num(n) is not None:
        s += f" (n = {int(num(n))})"
    return s


def epoch_axis(ax, epochs) -> None:
    """Symmetric-log epoch axis (early checkpoints readable) with ticks at the measured epochs."""
    ax.set_xscale("symlog", linthresh=5)
    ticks = sorted({int(e) for e in epochs if num(e) is not None})
    if ticks:
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()


def width(n_panels: int, per_panel: float = 2.4) -> float:
    return min(DOUBLE, per_panel * n_panels + 0.3)


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------------------------- figures
def fig1_twins(B: Book) -> None:
    p = B.data_dir / "fig1_teaser.npz"
    if not p.is_file():
        B.skip("figure", "fig1_twins", "no MUST-05 teaser (data/fig1_teaser.npz)")
        return
    B.inputs[p.name] = sha256(p)
    z = np.load(p)
    blocks = [int(b) for b in z["blocks"]]
    n_pairs, n_patch = 4, 6
    blocks = [b for b in blocks if f"b{b}_crops" in z.files]
    if not blocks:
        B.skip("figure", "fig1_twins", "teaser.npz holds no crops")
        return
    rows = []
    for b in blocks:
        crops, valid = z[f"b{b}_crops"], z[f"b{b}_valid"]
        for pi in range(min(n_pairs, crops.shape[0])):
            rows.append((b, pi, crops[pi], valid[pi]))
    k = min(n_patch, rows[0][2].shape[1])
    fig, axes = plt.subplots(len(rows), 2 * k + 1, figsize=(DOUBLE, 0.42 * len(rows) + 0.4),
                             gridspec_kw={"width_ratios": [1] * k + [0.25] + [1] * k, "wspace": 0.04, "hspace": 0.08})
    axes = np.atleast_2d(axes)
    for r, (b, pi, cr, va) in enumerate(rows):
        for side in range(2):
            for j in range(k):
                ax = axes[r, j if side == 0 else k + 1 + j]
                ax.axis("off")
                if va[side, j]:
                    ax.imshow(cr[side, j], interpolation="nearest")
        axes[r, k].axis("off")
        axes[r, 0].text(-0.15, 0.5, f"block {b}, pair {pi + 1}", transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=6, color=C_INK2)
    axes[0, k // 2].set_title("neuron i: top patches", fontsize=7, loc="center")
    axes[0, k + 1 + k // 2].set_title("its twin j: top patches", fontsize=7, loc="center")
    tax = B.number("must04.dinov2_vitb14_reg.LR.tax_1-6")
    if num(tax) is not None:
        fig.suptitle(f"Mirror twins in DINOv2-B reg: the LR tax is {100 * num(tax):.0f}% of MLP weights in blocks 1 to 6",
                     fontsize=7.5, y=1.06)
    B.save_fig(fig, "fig1_twins", ["data/fig1_teaser.npz"] + (["PAPER_NUMBERS.json must04.dinov2_vitb14_reg.LR.tax_1-6"]
                                                               if tax is not None else []),
               "Top activating patches (50k ImageNet-train images) of the strongest twin pairs in blocks "
               + ", ".join(map(str, blocks)) + ".")


def fig2_tax_depth(B: Book) -> None:
    blk, gen = B.data("fig2_blocks"), B.data("table4_generality")
    if blk is None and gen is None:
        B.skip("figure", "fig2_tax_depth", "no MUST-04 or CORE-04 measurement (data/fig2_blocks.csv)")
        return
    panels = []
    if blk is not None and {"group", "tax_mlp", "rel_depth"} <= set(blk.columns) and (blk["group"] == "LR").any():
        panels.append("a")
    if blk is not None and "ratio_to_null" in blk and (blk["model"] == "dinov2_vitb14_reg").any():
        panels.append("b")
    if gen is not None and "ratio_to_null" in gen and gen["ratio_to_null"].notna().any():
        panels.append("c")
    if not panels:
        B.skip("figure", "fig2_tax_depth", "inputs lack the plotted columns")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(width(len(panels)), 2.3), squeeze=False, layout="constrained")
    axes = axes[0]
    others = {}
    i = 0
    if "a" in panels:
        ax = axes[i]
        i += 1
        d = blk[blk["group"] == "LR"].drop_duplicates(["model", "block"], keep="first")
        for model, g in d.groupby("model", sort=False):
            lab, col, mk = model_style(model, others)
            g = g.sort_values("rel_depth")
            ax.plot(g["rel_depth"], g["tax_mlp"], color=col, marker=mk, markersize=3.2, linewidth=1.2, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=0.5)
        ax.set_xlabel("relative depth of the block")
        ax.set_ylabel("LR tax (share of MLP weights)")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        panel_label(ax, "a  Tax by depth")
    if "b" in panels:
        ax = axes[i]
        i += 1
        d = blk[blk["model"] == "dinov2_vitb14_reg"].drop_duplicates(["group", "block"], keep="first")
        for grp in [g for g in GROUPS if g in set(d["group"])]:
            g = d[d["group"] == grp].sort_values("block")
            lab, col, mk = GROUPS[grp]
            ax.plot(g["block"], g["ratio_to_null"], color=col, marker=mk, markersize=3.2, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=0.5)
            if grp == "LR" and {"s_mlp_null_lo", "s_mlp_null_hi", "s_mlp_null_mean"} <= set(g.columns):
                ax.fill_between(g["block"], g["s_mlp_null_lo"] / g["s_mlp_null_mean"],
                                g["s_mlp_null_hi"] / g["s_mlp_null_mean"], color=C_GRID, linewidth=0, zorder=0,
                                label="null 95% interval")
        ax.axhline(1.0, color=C_INK2, linewidth=0.6)
        ax.set_xlabel("block")
        ax.set_ylabel("breaking share / null mean")
        ax.legend(loc="center right", fontsize=6)
        panel_label(ax, "b  Wrong groups, DINOv2-B reg")
    if "c" in panels:
        ax = axes[i]
        d = gen[gen["experiment"] == "MUST-04"] if "experiment" in gen else gen
        d = d[d["range"] == "1-6"] if "range" in d else d
        grps = [g for g in GROUPS if g in set(d["group"])]
        xs = {g: j for j, g in enumerate(grps)}
        models = list(dict.fromkeys(d["model"]))
        for mi, model in enumerate(models):
            lab, col, mk = model_style(model, others)
            g = d[d["model"] == model]
            off = (mi - (len(models) - 1) / 2) * 0.12
            ax.scatter([xs[x] + off for x in g["group"]], g["ratio_to_null"], color=col, marker=mk, s=16, label=lab,
                       edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.axhline(0.9, color=C_INK2, linewidth=0.6)
        ax.text(-0.45, 0.91, "0.9 (P4.1 bar)", ha="left", va="bottom", fontsize=6, color=C_INK2)
        ax.set_xticks(range(len(grps)), [GROUPS[g][0].split(" (")[1].rstrip(")") for g in grps])
        ax.set_ylabel("ratio to null, blocks 1 to 6")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right", fontsize=6)
        panel_label(ax, "c  Released models")
    skipped = [f"panel {p} (inputs missing)" for p in ("a", "b", "c") if p not in panels]
    skipped.append("artefact-token inset (earlier-phase result, results/paper)")
    B.save_fig(fig, "fig2_tax_depth", ["data/fig2_blocks.csv", "data/table4_generality.csv"],
               "Tax by depth across models (a); breaking share relative to the matched null for the trained (LR) and "
               "wrong groups (b, c).", skipped)


def fig3_controlled(B: Book) -> None:
    mat, emg = B.data("fig3_matrix"), B.data("fig3_emergence")
    if mat is None and emg is None:
        B.skip("figure", "fig3_controlled", "no CORE-01 measurement (data/fig3_matrix.csv)")
        return
    panels = [p for p, d in (("a", mat), ("b", emg)) if d is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(width(len(panels), 3.4), 2.5), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    i = 0
    if mat is not None:
        ax = axes[i]
        i += 1
        d = mat[mat["final_ok"].astype(str) == "True"] if "final_ok" in mat else mat
        cols = [("LR", "twin_fraction_t50", "LR twins"), ("UD", "twin_fraction_t50", "UD twins"),
                ("R90", "share_4cycles", "R90 orbits of 4"), ("Klein", "share_4cycles", "Klein orbits of 4")]
        cols = [c for c in cols if ((d["group"] == c[0]) & (d["measure"] == c[1])).any()]
        conds = [c for c in CONDITIONS if c in set(d["condition"])]
        M = np.full((len(conds), len(cols)), np.nan)
        for r, cd in enumerate(conds):
            for k, (grp, meas, _) in enumerate(cols):
                v = d[(d["condition"] == cd) & (d["group"] == grp) & (d["measure"] == meas)]["value"]
                if len(v.dropna()):
                    M[r, k] = float(v.mean())
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.6, float(np.nanmax(M)) if np.isfinite(M).any() else 1),
                       aspect="auto")
        ax.grid(False)
        for r in range(M.shape[0]):
            for k in range(M.shape[1]):
                if np.isfinite(M[r, k]):
                    v = M[r, k]
                    ax.text(k, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="#ffffff" if v > 0.35 else C_INK)
        ax.set_xticks(range(len(cols)), [c[2] for c in cols], rotation=20, ha="right")
        ax.set_yticks(range(len(conds)), [CONDITIONS[c][0] for c in conds])
        ax.set_xlabel("structure measured (blocks 1 to 6, final checkpoint)")
        ax.set_ylabel("trained with")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
        panel_label(ax, "a  Structure matrix (mean over seeds)")
    if emg is not None:
        ax = axes[i]
        trained = {"LR": ("LR", "twin_fraction_t50"), "UD": ("UD", "twin_fraction_t50"),
                   "LRUD": ("Klein", "share_4cycles"), "R90": ("R90", "share_4cycles"), "none": ("LR", "twin_fraction_t50")}
        for cond, (grp, meas) in trained.items():
            g = emg[(emg["condition"] == cond) & (emg["group"] == grp)].dropna(subset=[meas])
            if not len(g):
                continue
            lab, col, mk = CONDITIONS[cond]
            for _, gs in g.groupby("seed"):
                gs = gs.sort_values("epoch")
                ax.plot(gs["epoch"], gs[meas], color=col, linewidth=0.5, alpha=0.5)
            mean = g.groupby("epoch")[meas].mean().sort_index()
            ylab = {"twin_fraction_t50": "twins", "share_4cycles": "orbits of 4"}[meas]
            ax.plot(mean.index, mean.values, color=col, marker=mk, markersize=3, label=f"{lab}: {grp} {ylab}",
                    markeredgecolor=SURFACE, markeredgewidth=0.5)
        epoch_axis(ax, emg["epoch"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("share of MLP units (blocks 1 to 6)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
        panel_label(ax, "b  Emergence over training")
    B.save_fig(fig, "fig3_controlled", ["data/fig3_matrix.csv", "data/fig3_emergence.csv"],
               "Controlled training (ViT-S/8, ImageNet-100): structure by training symmetry (a) and its emergence (b); "
               "thin lines are single seeds.", [f"panel {p}" for p in ("a", "b") if p not in panels])


def fig4_visibility_stacking(B: Book) -> None:
    pairs, stack, bands = B.data("fig4a_pairs"), B.data("fig4b_stacking"), B.data("fig4c_bands")
    panels = [p for p, d in (("a", pairs), ("b", stack), ("c", bands)) if d is not None]
    if not panels:
        B.skip("figure", "fig4_visibility_stacking", "no MUST-06, CORE-03 or CORE-05 data")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(width(len(panels)), 2.3), squeeze=False, layout="constrained")
    axes = axes[0]
    i = 0
    if pairs is not None:
        ax = axes[i]
        i += 1
        d = pairs[pairs["model"] == "dinov2_vitb14_reg"] if (pairs["model"] == "dinov2_vitb14_reg").any() else pairs
        bins = np.linspace(-1, 1, 41)
        series = [(d[d["pair_type"] == "twin"]["mirror_corr"], "twins, mirrored image", C_BLUE),
                  (d[d["pair_type"] == "twin"]["same_corr"], "twins, same image", C_ORANGE),
                  (d[d["pair_type"] == "random"]["same_corr"], "random pairs, same image", C_NULL)]
        for vals, lab, col in series:
            vals = vals.dropna()
            if len(vals):
                ax.hist(vals, bins=bins, histtype="step", linewidth=1.3, color=col, label=lab, density=True)
        ax.set_xlabel("Pearson correlation of the pair")
        ax.set_ylabel("density")
        ax.legend(loc="upper left", fontsize=6)
        panel_label(ax, "a  Visibility (blocks 1 to 6)")
    if stack is not None:
        ax = axes[i]
        i += 1
        xcol = "span_kept_frac" if "span_kept_frac" in stack and stack["span_kept_frac"].notna().any() else "fraction"
        rstyle = {"lowrank_scored": ("low rank alone (better of SVD, SVD-LLM)", C_ORANGE, "s"),
                  "mirror_lowrank_scored": ("Mirror plus low rank", C_BLUE, "o")}
        models = list(dict.fromkeys(stack["model"]))
        lstyles = ["-", (0, (4, 1.5)), (0, (1, 1.2))]
        for mi, model in enumerate(models[:3]):
            g0 = stack[stack["model"] == model]
            for role, (lab, col, mk) in rstyle.items():
                g = g0[g0["role"] == role].dropna(subset=[xcol, "val_top1"]).sort_values(xcol) if "role" in g0 else g0[:0]
                if len(g):
                    ax.plot(g[xcol], g["val_top1"], color=col, marker=mk, markersize=3, linestyle=lstyles[mi],
                            label=lab if mi == 0 else None, markeredgecolor=SURFACE, markeredgewidth=0.5)
            mo = g0[g0["curve"] == "mirror"].dropna(subset=[xcol, "val_top1"]) if "curve" in g0 else g0[:0]
            if len(mo):
                ax.scatter(mo[xcol].iloc[:1], mo["val_top1"].iloc[:1], marker="*", s=36, color=C_INK, zorder=4,
                           label="Mirror alone" if mi == 0 else None)
        if len(models) > 1:
            for mi, model in enumerate(models[:3]):
                ax.plot([], [], color=C_INK2, linestyle=lstyles[mi], label=MODELS.get(model, (model,))[0])
        ax.set_xlabel("parameters kept (share of the converted span)")
        ax.set_ylabel("k-NN top-1 (%)")
        ax.invert_xaxis()
        ax.legend(loc="lower left", fontsize=6)
        panel_label(ax, "b  Stacking with low rank (zero-shot)")
    if bands is not None:
        ax = axes[i]
        cols = {"top": ("top_breaking_share", SEQ_BLUE[6], "o"), "middle": ("mid_breaking_share", SEQ_BLUE[4], "s"),
                "bottom": ("bottom_breaking_share", SEQ_BLUE[2], "^")}
        groups = list(dict.fromkeys(bands["group"])) if "group" in bands else [None]
        lstyles = {"LR": "-", "UD": (0, (4, 1.5))}
        for grp in groups:
            g = bands[bands["group"] == grp] if grp is not None else bands
            for band, (col_name, col, mk) in cols.items():
                if col_name not in g:
                    continue
                gm = g.groupby("epoch")[col_name].mean().sort_index()
                ax.plot(gm.index, gm.values, color=col, marker=mk, markersize=3, linestyle=lstyles.get(grp, "-"),
                        label=f"{band} variance band" + (f", {grp} runs" if grp else ""),
                        markeredgecolor=SURFACE, markeredgewidth=0.5)
        epoch_axis(ax, bands["epoch"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("within-band breaking share")
        ax.legend(fontsize=6)
        panel_label(ax, "c  Where the asymmetry sits")
    B.save_fig(fig, "fig4_visibility_stacking", ["data/fig4a_pairs.csv", "data/fig4b_stacking.csv",
                                                 "data/fig4c_bands.csv"],
               "(a) Twins correlate on the mirrored image but not on the same image; (b) Mirror stacks with low rank; "
               "(c) breaking energy by input-variance band over training.",
               [f"panel {p} (inputs missing)" for p in ("a", "b", "c") if p not in panels])


def fig5_accuracy_params(B: Book) -> None:
    pts = B.data("fig5_points")
    if pts is None:
        B.skip("figure", "fig5_accuracy_params", "no MUST-07, MUST-09 or CORE-04 data (data/fig5_points.csv)")
        return
    best = B.number("gate_b.best")
    role = {"mirror_exact": ("Mirror-exact", C_BLUE, "o", 1.8)}
    if isinstance(best, str) and best in METHOD_LABEL and best != "mirror_exact":
        role[best] = (f"{METHOD_LABEL[best]} (best baseline)", C_ORANGE, "s", 1.4)
    role.setdefault("svd", ("Plain SVD", C_AQUA, "^", 1.2))
    role.setdefault("random_structure", ("Random structure", C_NULL, "v", 1.0))
    zs = pts[(pts["regime"] == "zero-shot") & (pts["model"] == "dinov2_vitb14_reg")]
    heal = pts[pts["regime"].isin(["healed", "released family"])]
    panels = [p for p, d in (("a", zs), ("b", heal)) if len(d)]
    if not panels:
        B.skip("figure", "fig5_accuracy_params", "no zero-shot DINOv2-B reg rows and no healed rows")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(width(len(panels), 3.3), 2.9), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    i = 0
    if len(zs):
        ax = axes[i]
        i += 1
        orig = zs[zs["method"] == "original"]
        other_drawn = False
        for meth in [m for m in METHOD_ORDER if m in set(zs["method"]) and m != "original"]:
            g = zs[zs["method"] == meth].dropna(subset=["params", "top1"]).sort_values("params")
            if not len(g):
                continue
            if meth in role:
                lab, col, mk, lw = role[meth]
                ax.plot(g["params"] / 1e6, g["top1"], color=col, marker=mk, markersize=4, linewidth=lw, label=lab,
                        markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.0, zorder=3)
            else:
                ax.plot(g["params"] / 1e6, g["top1"], color=C_OTHER, linewidth=0.7, marker=".", markersize=2,
                        label=None if other_drawn else "other baselines and controls", zorder=2)
                other_drawn = True
        if len(orig) and num(orig["params"].iloc[0]) is not None:
            ax.scatter([orig["params"].iloc[0] / 1e6], [orig["top1"].iloc[0]], marker="*", s=40, color=C_INK,
                       zorder=4, label="original model")
        ax.set_xlabel("parameters (millions)")
        ax.set_ylabel("validation k-NN top-1 (%)")
        panel_label(ax, "a  Zero-shot, DINOv2-B reg")
    if len(heal):
        ax = axes[i]
        h = heal[heal["regime"] == "healed"].dropna(subset=["top1"])
        for _, r in h.iterrows():
            if num(r["params"]) is None:
                continue
            lab, col, mk, _ = role.get(r["method"], (METHOD_LABEL.get(r["method"], r["method"]), C_OTHER, "D", 1))
            ax.errorbar([r["params"] / 1e6], [r["top1"]], yerr=[[num(r["top1_std"]) or 0], [num(r["top1_std"]) or 0]],
                        color=col, marker=mk, markersize=5, linestyle="none", label=f"{lab}, healed",
                        markeredgecolor=SURFACE, markeredgewidth=0.6, elinewidth=0.8, zorder=3)
        fam = heal[heal["regime"] == "released family"].dropna(subset=["params", "top1"]).sort_values("params")
        if len(fam):
            ax.plot(fam["params"] / 1e6, fam["top1"], color=C_INK, marker="D", markersize=3.5, linewidth=0.8,
                    label="released DINOv2 family", zorder=2)
            for _, r in fam.iterrows():
                ax.annotate(MODELS.get(r["model"], (r["model"],))[0], (r["params"] / 1e6, r["top1"]),
                            textcoords="offset points", xytext=(3, -8), fontsize=5.5, color=C_INK2)
            ax.set_xscale("log")
        ax.set_xlabel("parameters (millions)")
        ax.set_ylabel("validation k-NN top-1 (%)")
        panel_label(ax, "b  Healed full model and the released family")
    handles, labels = [], []
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6)
    B.save_fig(fig, "fig5_accuracy_params", ["data/fig5_points.csv", "PAPER_NUMBERS.json gate_b.best"],
               "Accuracy against parameters: every method at matched parameters, zero-shot (hollow) and healed (filled, "
               "mean and standard deviation over seeds), with the released DINOv2 family.",
               [f"panel {p} (inputs missing)" for p in ("a", "b") if p not in panels])


# ---------------------------------------------------------------------------------------------- tables
def table1(B: Book) -> None:
    h, fa = B.data("table1_healed"), B.data("table1_flipavg")
    if h is None and fa is None:
        B.skip("table", "table1_healed", "no MUST-09 or MUST-10 data")
        return
    rows = []
    if h is not None:
        steps = sorted([c for c in h.columns if c.startswith("knn_step")], key=lambda c: int(c[8:]))
        for _, r in h.sort_values("method", key=lambda s: s.map(lambda m: METHOD_ORDER.index(m)
                                                                  if m in METHOD_ORDER else 99)).iterrows():
            lab = METHOD_LABEL.get(r["method"], r["method"]) + (f" ({r['role']})" if isinstance(r.get("role"), str)
                                                                 and r["role"] else "")
            if r.get("targets") not in (None, "sym") and isinstance(r.get("targets"), str):
                lab += f", {r['targets']} targets"
            row = {"Method": lab, "Seeds": int(r["n_seeds"]), "k-NN top-1": pm(r["knn_mean"], r["knn_std"])}
            for s in steps:
                row[f"step {s[8:]}"] = f2(r[s])
            row.update({"Flip consistency fp32 (min)": f2(r["flip_fp32_min"], 3),
                        "Parameters (share)": f2(r["params_frac"], 3), "GFLOPs (share)": f2(r["gflops_frac"], 3)})
            if "latency_ms_b256" in r:
                row["Latency b256 (ms)"] = f2(r["latency_ms_b256"], 1)
            rows.append(row)
    if fa is not None:
        for _, r in fa.iterrows():
            rows.append({"Method": f"{r['variant']} model, {r['mode']}" + (" (two passes)" if r["mode"] == "flipavg"
                                                                           else ""),
                         "Seeds": 1, "k-NN top-1": f2(r["knn_top1"]),
                         "Flip consistency fp32 (min)": f2(r["flip_fp32"], 3),
                         "Parameters (share)": f2(r["params_frac"], 3)})
    df = pd.DataFrame(rows)
    B.save_table("table1_healed", "Healed full-model comparison (identical recipe) and test-time flip averaging",
                 df, ["data/table1_healed.csv", "data/table1_flipavg.csv"],
                 "k-NN: 100k-image train bank, 50k validation queries; mean ± std over healing seeds. Flip consistency "
                 "with fp32 features on all validation images.")


def table2(B: Book) -> None:
    d, sh = B.data("table2_downstream"), B.data("table2_shift")
    if d is None and sh is None:
        B.skip("table", "table2_downstream", "no MUST-11 or CORE-07 data")
        return
    frames = [] if d is None else [d[["task", "variant", "value"]]]
    if sh is not None:
        for col, task in (("val_top1", "ImageNet val, probe (CORE-07)"), ("v2_top1", "ImageNet-V2"),
                          ("v2_drop", "drop to ImageNet-V2"), ("sketch_top1", "ImageNet-Sketch")):
            if col in sh:
                frames.append(pd.DataFrame({"task": task, "variant": sh["variant"], "value": sh[col]}))
    d = pd.concat(frames, ignore_index=True)
    t = d.pivot_table(index="variant", columns="task", values="value", aggfunc="first")
    t = t.map(lambda v: f2(v)) if hasattr(t, "map") else t.applymap(lambda v: f2(v))
    t = t.reset_index().rename(columns={"variant": "Model"})
    B.save_table("table2_downstream", "Downstream tasks, robustness and mask equivariance of the healed full models",
                 t, ["data/table2_downstream.csv", "data/table2_shift.csv"])


def table3(B: Book) -> None:
    m7 = B.data("must07_rows")
    if m7 is None:
        B.skip("table", "table3a_compressors", "no MUST-07 rows")
    else:
        d = m7[m7["model"] == "dinov2_vitb14_reg"] if (m7["model"] == "dinov2_vitb14_reg").any() else m7
        sizes = [s for s in ("6mlp", "6both", "9both", "full") if s in set(d["size"])]
        rows = []
        for meth in [m for m in METHOD_ORDER if m in set(d["method"])]:
            row = {"Method": METHOD_LABEL.get(meth, meth)}
            for s in sizes:
                g = d[(d["method"] == meth) & (d["size"] == s)]
                row[s] = pm(g["top1_mean"].iloc[0], g["top1_std"].iloc[0], g["n_seeds"].iloc[0]) if len(g) else ""
            rows.append(row)
        p7 = B.data("table3_must07_p7")
        notes = ("Validation k-NN top-1, mean ± std over fit seeds (n). Sizes: 6mlp = blocks 1 to 6, MLP only; "
                 "6both = blocks 1 to 6, MLP and attention; 9both; full = every block.")
        if p7 is not None:
            for _, r in p7.iterrows():
                for row in rows:
                    if row["Method"] == METHOD_LABEL.get(r["method"]):
                        row[f"diff {r['size']} (CI s0)"] = (f"{f2(r['diff_mean'])} [{f2(r['ci_low'])}, "
                                                            f"{f2(r['ci_high'])}]") if num(r["diff_mean"]) is not None else ""
        B.save_table("table3a_compressors", "Zero-shot comparison at matched parameters (MUST-07), DINOv2-B reg",
                     pd.DataFrame(rows), ["data/must07_rows.csv", "data/table3_must07_p7.csv"], notes)
    m8 = B.data("table3_must08")
    if m8 is None:
        B.skip("table", "table3b_components", "no MUST-08 rows")
    else:
        rows = [{"Size": r["size"], "k-NN top-1": pm(r["top1_mean"], r["top1_std"], r["n_seeds"]),
                 "Parameters (share)": f2(r["model_params_frac"], 3), "GFLOPs (share)": f2(r["gflops_frac"], 3),
                 "Flip bf16": f2(r["flip_bf16"], 2), "Flip fp32": f2(r["flip_fp32"], 3)} for _, r in m8.iterrows()]
        B.save_table("table3b_components", "Matched component ablation of Mirror-exact (MUST-08)", pd.DataFrame(rows),
                     ["data/table3_must08.csv"])
    old = B.data("table3_earlier_ablations")
    if old is None:
        B.skip("table", "table3c_earlier_ablations", "results/paper/tables/T_ablations.csv not found by the collector")
    else:
        keys = [c for c in ("model_label", "ablation", "variant", "k") if c in old]
        if "top1" in old and keys:
            rows = []
            for kv, g in old.groupby(keys, sort=False):
                kv = kv if isinstance(kv, tuple) else (kv,)
                vals = pd.to_numeric(g["top1"], errors="coerce").dropna()
                row = dict(zip(["Model", "Ablation", "Variant", "Blocks k"][:len(keys)], kv))
                row["k-NN top-1"] = pm(vals.mean(), vals.std(ddof=1) if len(vals) > 1 else None, len(vals))
                if "params_vs_mirror_first_k" in g:
                    row["Parameters vs Mirror"] = f2(pd.to_numeric(g["params_vs_mirror_first_k"], errors="coerce").mean(), 2)
                if "params_matched" in g:
                    row["Matched"] = "yes" if g["params_matched"].astype(str).eq("True").all() else "no"
                rows.append(row)
            t = pd.DataFrame(rows)
        else:
            t = old
        B.save_table("table3c_earlier_ablations", "Earlier ablations: operator, pairing, fit size, block choice and "
                     "projection strength (mean ± std over fit seeds)", t,
                     ["data/table3_earlier_ablations.csv (from results/paper/tables/T_ablations.csv)"],
                     "Earlier-phase protocol (Mirror-local operators); see results/paper/INDEX.md for each setting.")


def table4(B: Book) -> None:
    d = B.data("table4_generality")
    if d is None or not len(d):
        B.skip("table", "table4_generality", "no MUST-04 or CORE-04 measurement")
        return
    rows = []
    for _, r in d.iterrows():
        rows.append({"Model": MODELS.get(r["model"], (r["model"],))[0], "Group": r["group"], "Range": r.get("range", ""),
                     "Ratio to null": f2(r.get("ratio_to_null"), 3), "Twin fraction": f2(r.get("twin_fraction_t50"), 3),
                     "Orbits of 4": f2(r.get("share_4cycles"), 3), "Tax, first half": f2(r.get("tax_mlp_first_half"), 3),
                     "Surgery drop (points)": f2(r.get("surgery_drop"))})
    law = B.data("appendix_law")
    notes = "Ratio to null and twin fraction (threshold 0.5) over the range; tax from the character formula."
    if law is not None:
        notes += (" The law check (MUST-01): largest |predicted - achieved| saving "
                  f"{f2(law['max_abs_diff_pp'].max(), 2)} pp over {int(law['n_blocks'].sum())} blocks.")
    B.save_table("table4_generality", "Generality and the law across models and groups", pd.DataFrame(rows),
                 ["data/table4_generality.csv", "data/appendix_law.csv"], notes, align="lllrrrrr")
    c2 = B.data("table4_core02")
    if c2 is None:
        B.skip("table", "table4b_refund_new_groups", "no CORE-02 surgery rows")
    else:
        rows = [{"Model": r.get("run"), "Group": r.get("group"), "Relation": r.get("relation"),
                 "Tax predicted": f2(r.get("tax_predicted"), 4), "Tax achieved": f2(r.get("tax_achieved"), 4),
                 "Top-1 before": f2(r.get("top1_original")), "Drop (points)": f2(r.get("drop_points"))}
                for _, r in c2.iterrows()]
        B.save_table("table4b_refund_new_groups", "The law predicts the refund for new groups (CORE-02, controlled "
                     "models, blocks 1 to 6)", pd.DataFrame(rows), ["data/table4_core02.csv"],
                     "Predicted tax written before any surgery; achieved tax from the compressed weights.",
                     align="lllrrrr")


def appendix(B: Book) -> None:
    ex = B.data("appendix_exactness")
    if ex is None:
        B.skip("table", "appendix_a1_exactness", "no MUST-02 per-component audit")
    else:
        keep = [c for c in ("variant", "dtype", "stage", "component", "space", "rel_tensor_max", "rel_token_max",
                            "n_blocks") if c in ex]
        t = ex[keep].copy()
        for c in ("rel_tensor_max", "rel_token_max"):
            if c in t:
                t[c] = t[c].map(lambda v: "" if num(v) is None else f"{num(v):.2e}")
        B.save_table("appendix_a1_exactness", "Exactness audit of Mirror-exact, per component (largest over blocks)",
                     t, ["data/appendix_exactness.csv"], "Relative equivariance error ||T out(x) - out(mirror x)|| / "
                     "||out(mirror x)||, maximum over images; the prediction is below 1e-5 in fp32.")
    law = B.data("appendix_law")
    if law is None:
        B.skip("table", "appendix_a2_law", "no MUST-01 output")
    else:
        t = law[[c for c in ("model", "n_blocks", "max_abs_diff_pp", "structure", "verdict") if c in law]].copy()
        B.save_table("appendix_a2_law", "The law: predicted against achieved parameter saving (MUST-01)", t,
                     ["data/appendix_law.csv"])
    preds = B.numbers.get("predictions") or []
    if not preds:
        B.skip("table", "appendix_a3_prereg", "PAPER_NUMBERS.json has no predictions")
    else:
        rows = [{"ID": p["id"], "Prediction": p["statement"], "Value": p["value"] if isinstance(p["value"], str)
                 else f2(p["value"], 4), "Bar": p["bar"], "Verdict": p["verdict"]} for p in preds]
        B.save_table("appendix_a3_prereg", "Pre-registration: every prediction with its verdict", pd.DataFrame(rows),
                     ["PAPER_NUMBERS.json predictions"], f"Pre-registration commit "
                     f"{str((B.numbers.get('prereg') or {}).get('signoff_commit'))[:10]}.", align="lllll")
    fc = B.numbers.get("failed_checks") or {}
    rows = [{"Phase": "new", "Check": f["id"] + ": " + f["statement"], "Value": f["value"] if isinstance(f["value"], str)
             else f2(f["value"], 4), "Bar": f["bar"]} for f in fc.get("new_phase") or []]
    rows += [{"Phase": "earlier", "Check": f["name"], "Value": f2(f["value"], 4), "Bar": f["bar"]}
             for f in fc.get("earlier_phase") or []]
    if not rows:
        B.skip("table", "appendix_a4_failed", "no failed check recorded")
    else:
        B.save_table("appendix_a4_failed", "Failed checks, reported as failures", pd.DataFrame(rows),
                     ["PAPER_NUMBERS.json failed_checks"], align="llll")


# ---------------------------------------------------------------------------------------------- main
def rel_repo(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def write_index(B: Book, git_hash: str | None) -> None:
    L = ["# Figures and tables of the next phase", "",
         f"Written by `scripts/make_figures_next.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} (code "
         f"commit `{(git_hash or 'not recorded')[:10]}`) from `{rel_repo(B.data_dir)}` and PAPER_NUMBERS.json only. "
         "Skipped items had no input; no placeholder data is drawn.", ""]
    for kind in ("figure", "table"):
        L += [f"## {kind.capitalize()}s", ""]
        for e in [x for x in B.entries if x["kind"] == kind]:
            if e["status"] == "made":
                L.append(f"- **{e['name']}**: {', '.join(e['files'])}. {e.get('caption', '')} Sources: "
                         + ", ".join(e["sources"]) + "."
                         + (" Not drawn: " + "; ".join(e["skipped_panels"]) + "." if e.get("skipped_panels") else ""))
            else:
                L.append(f"- **{e['name']}**: skipped ({e['reason']}).")
        L.append("")
    (B.dir / "FIGURES.md").write_text("\n".join(L), encoding="utf-8")
    meta = {"written_by": "scripts/make_figures_next.py", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_hash": git_hash, "inputs_sha256": B.inputs, "entries": B.entries,
            "versions": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__}}
    (B.dir / "figures_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT), help="tree the collector wrote into (default: the repository)")
    p.add_argument("--paper-dir", default=None, help="default <root>/results/paper_next")
    p.add_argument("--numbers-json", default=None, help="default <root>/PAPER_NUMBERS.json")
    return p.parse_args(argv)


def main(argv=None) -> list:
    a = parse_args(argv)
    root = Path(a.root).resolve()
    paper_dir = Path(a.paper_dir) if a.paper_dir else root / "results" / "paper_next"
    nj = Path(a.numbers_json) if a.numbers_json else root / "PAPER_NUMBERS.json"
    numbers = json.loads(nj.read_text(encoding="utf-8")) if nj.is_file() else {}
    B = Book(paper_dir, numbers)
    for d in (B.fig_dir, B.tab_dir):
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
    use_style()
    for fn in (fig1_twins, fig2_tax_depth, fig3_controlled, fig4_visibility_stacking, fig5_accuracy_params,
               table1, table2, table3, table4, appendix):
        try:
            fn(B)
        except Exception as e:  # noqa: BLE001  one broken item must not stop the others
            plt.close("all")
            B.skip("figure" if fn.__name__.startswith("fig") else "table", fn.__name__,
                   f"error {type(e).__name__}: {e}")
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        gh = out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        gh = None
    B.dir.mkdir(parents=True, exist_ok=True)
    write_index(B, gh)
    made = [e["name"] for e in B.entries if e["status"] == "made"]
    log(f"{len(made)} items made, {len(B.entries) - len(made)} skipped; index in {B.dir / 'FIGURES.md'}")
    return B.entries


if __name__ == "__main__":
    main()

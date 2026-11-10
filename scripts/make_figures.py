#!/usr/bin/env python
"""Paper figures for Mirror Surgery, built only from the collector's tables.

Reads the CSV tables in results/paper/tables/ (written by scripts/collect_results.py) and writes each figure as
vector PDF and 300 dpi PNG into results/paper/figures_paper/, sized for a two-column paper (single column 3.3 in,
double column 6.9 in). It also writes FIGURES.md (one entry per figure: files, draft caption, source tables and
columns, and what was skipped) and figures_meta.json (git hash, inputs with SHA-256, outputs, versions).

No number is computed here beyond what a figure shows: means and standard deviations come from the tables;
the only arithmetic is a difference of two table values where a panel plots a drop (stated in FIGURES.md).
There is no randomness (no seed is needed). A missing table, or missing rows, prints "not available" and
skips that figure or panel; rerun after scripts/collect_results.py has refreshed the tables. The F7 teaser figures
are drawn by scripts/f7_visuals.py into the same folder; this script only lists them in FIGURES.md.

Style: one sans font (Arial, falling back to Liberation Sans or DejaVu Sans), 7 pt text; the palette of
mirror/report.py (slots 1 to 3 of the reference palette) plus slots 4 to 7 as in scripts/collect_results.py;
every colour also carries a marker shape or a hatch so the figures read in greyscale and under colour-vision
deficiency. Zero-shot results are drawn light (hollow markers, tinted bars) and healed results solid.

Usage: python scripts/make_figures.py [--tables DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirror.report import C_AQUA, C_BLUE, C_GRID, C_INK, C_MUTED, C_NULL, C_ORANGE  # noqa: E402

# Slots 4 to 7 of the same reference palette (values as in scripts/collect_results.py).
C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET = "#eda100", "#e87ba4", "#008300", "#6250d6"

SINGLE, DOUBLE = 3.3, 6.9          # column widths in inches
FONT = 7.0
TINT = 0.28                        # zero-shot bars: this much of the colour over white

# Methods: colour follows the method in every figure; marker and hatch carry it in greyscale.
METHODS = {
    "mirror_mlp": {"label": "Mirror V-local, MLP only", "color": C_BLUE, "marker": "o", "hatch": ""},
    "vlocal": {"label": "Mirror V-local, MLP + attention", "color": C_AQUA, "marker": "^", "hatch": "||"},
    "vexact": {"label": "Mirror V-exact, MLP + attention", "color": C_VIOLET, "marker": "D", "hatch": "\\\\\\"},
    "svd": {"label": "SVD at matched parameters", "color": C_ORANGE, "marker": "s", "hatch": "///"},
    "random": {"label": "Random structure", "color": C_NULL, "marker": "v", "hatch": "..."},
    "prune": {"label": "MLP width pruning at matched parameters", "color": C_MUTED, "marker": "P", "hatch": "xxx"},
}
METHOD_ORDER = list(METHODS)
MS = {"o": 4.0, "s": 3.6, "^": 4.4, "v": 4.4, "D": 3.3, "P": 4.4, "X": 4.4}
ORIG_KW = {"color": C_INK, "linewidth": 0.8, "linestyle": (0, (4, 2))}

# Models: labels as in the tables; colours and markers as in scripts/collect_results.py (colour follows the model).
MODELS = {
    "dinov2_vits14_reg": {"label": "DINOv2-S reg", "tick": "DINOv2-S\nreg", "color": C_GREEN, "marker": "P"},
    "dinov2_vitb14_reg": {"label": "DINOv2-reg", "tick": "DINOv2\nreg", "color": C_BLUE, "marker": "o"},
    "dinov2_vitl14_reg": {"label": "DINOv2-L reg", "tick": "DINOv2-L\nreg", "color": C_VIOLET, "marker": "X"},
    "dinov2_vitb14": {"label": "DINOv2", "tick": "DINOv2", "color": C_ORANGE, "marker": "s"},
    "deit3_b16": {"label": "DeiT-III", "tick": "DeiT-III", "color": C_AQUA, "marker": "^"},
    "mae_b16": {"label": "MAE", "tick": "MAE", "color": C_YELLOW, "marker": "D"},
    "siglip2_b16": {"label": "SigLIP2", "tick": "SigLIP2", "color": C_MAGENTA, "marker": "v"},
}
MODEL_ORDER = list(MODELS)
MAIN_MODEL = "dinov2_vitb14_reg"

PROTOCOL = ("k-NN top-1 on ImageNet validation (50k queries, bank of 100 train images per class); operators "
            "fitted on ImageNet train images")


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def num(x):
    """Float from a CSV cell, or None for empty, non-numeric or non-finite cells."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def inum(x):
    v = num(x)
    return None if v is None else int(round(v))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tint(color: str, frac: float = TINT) -> str:
    """The colour laid over white at the given strength (an opaque hex, so PDFs carry no transparency)."""
    c = matplotlib.colors.to_rgb(color)
    return matplotlib.colors.to_hex(tuple(1 - frac * (1 - x) for x in c))


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    h = run("rev-parse", "HEAD")
    st = run("status", "--porcelain")
    return {"git_hash": h, "git_dirty": bool(st) if st is not None else None}


class Tables:
    """CSV tables of results/paper/tables, read as lists of dicts; records what was read and what was missing."""

    def __init__(self, d: Path):
        self.dir = d
        self.used: dict[str, str] = {}
        self.missing: list[str] = []

    def get(self, name: str):
        p = self.dir / name if name.endswith(".csv") else self.dir / f"{name}.csv"
        if not p.is_file():
            if p.name not in self.missing:
                self.missing.append(p.name)
            return None
        self.used.setdefault(p.name, sha256(p))
        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def glob(self, pattern: str) -> list[str]:
        return [p.name for p in sorted(self.dir.glob(pattern))]


class Registry:
    """What was drawn and what was skipped (printed, and written to FIGURES.md)."""

    def __init__(self):
        self.figures: list[dict] = []
        self.skipped: list[tuple[str, str]] = []

    def na(self, what: str, why: str):
        print(f"not available: {what} ({why})")
        self.skipped.append((what, why))


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 0.5,
        "ytick.labelsize": FONT - 0.5, "legend.fontsize": FONT - 0.5, "legend.title_fontsize": FONT - 0.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
        "axes.linewidth": 0.6, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "axes.titlecolor": C_INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "xtick.labelcolor": C_INK, "ytick.labelcolor": C_INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5, "axes.titlepad": 4.0,
        "grid.color": C_GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "lines.linewidth": 1.2, "lines.markersize": 4.0, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "patch.linewidth": 0.6, "hatch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.3, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.35,
        "errorbar.capsize": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
    })


def panel_title(ax, letter: str, text: str = ""):
    ax.set_title(rf"$\mathbf{{{letter}}}$   {text}" if text else rf"$\mathbf{{{letter}}}$", loc="left")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis)


def orig_handle(label="Original model"):
    return Line2D([], [], label=label, **ORIG_KW)


def method_line_handle(key, label=None):
    m = METHODS[key]
    return Line2D([], [], color=m["color"], marker=m["marker"], ms=MS[m["marker"]], mec="white", mew=0.5,
                  lw=1.2, label=label or m["label"])


def draw_bar(ax, x, h, w, key, healed: bool, err=None):
    """One bar: zero-shot as a tint with a coloured edge and coloured hatch, healed as a solid fill with white hatch.
    A bar of height 0 still shows its edge at the baseline."""
    m = METHODS[key]
    if healed:
        ax.bar(x, h, w, facecolor=m["color"], edgecolor=m["color"], linewidth=0.6, zorder=2)
        if m["hatch"]:
            ax.bar(x, h, w, facecolor="none", edgecolor="white", hatch=m["hatch"], linewidth=0.0, zorder=2.1)
    else:
        ax.bar(x, h, w, facecolor=tint(m["color"]), edgecolor=m["color"], hatch=m["hatch"], linewidth=0.7, zorder=2)
    if err is not None and err > 0:
        ax.errorbar(x, h, yerr=err, fmt="none", ecolor=C_INK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=3)


def rel(p: Path) -> str:
    """Path relative to the repository when inside it, else as given."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def save(fig, out: Path, stem: str) -> list[str]:
    files = []
    pdf, png = out / f"{stem}.pdf", out / f"{stem}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Creator": "scripts/make_figures.py"})
    fig.savefig(png, dpi=300, metadata={"Software": "scripts/make_figures.py"})
    files += [pdf.name, png.name]
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {rel(pdf)} and .png ({w:.2f} x {h:.2f} in)")
    return files


def seeds_phrase(ns: list[int], what: str = "fit seeds") -> str:
    ns = sorted(set(n for n in ns if n))
    if not ns:
        return "number of seeds not recorded"
    if ns == [1]:
        return "One run each (n = 1, no error bars)"
    multi = [n for n in ns if n > 1]
    s = f"Error bars are the sample standard deviation over n = {', '.join(map(str, multi))} {what}"
    if 1 in ns:
        s += "; points without error bars come from one run (n = 1)"
    return s


def fmt_list(xs) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


# ----------------------------------------------------------------------------------------------
# Figure 1: zero-shot k-NN top-1 against k
# ----------------------------------------------------------------------------------------------
FIG1_PANELS = [
    ("dinov2_vitb14_reg", "MLP only", [("mirror_mlp", "mirror_mlp"), ("svd_mlp", "svd"), ("random_mlp", "random")]),
    ("dinov2_vitb14_reg", "MLP + attention",
     [("vlocal", "vlocal"), ("vexact", "vexact"), ("svd_attn", "svd"), ("random_attn", "random")]),
    ("dinov2_vitb14", "MLP only", [("mirror_mlp", "mirror_mlp"), ("svd_mlp", "svd"), ("random_mlp", "random")]),
    ("dinov2_vitb14", "MLP + attention",
     [("vlocal", "vlocal"), ("vexact", "vexact"), ("svd_attn", "svd"), ("random_attn", "random")]),
]


def fig1_main(T: Tables, R: Registry, out: Path):
    name = "Figure 1 (main zero-shot result)"
    rows = T.get("T_main_zeroshot")
    if rows is None:
        return R.na(name, "T_main_zeroshot.csv not found")

    def ok(r):
        return r.get("status") == "ok" and num(r.get("top1_mean")) is not None

    panels, ns = [], []
    for model, budget, spec in FIG1_PANELS:
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            R.na(f"{name}, {MODELS.get(model, {}).get('label', model)} {budget}", "model not in T_main_zeroshot")
            continue
        label = mrows[0].get("model_label") or model
        series = []
        for var, key in spec:
            pts = sorted((inum(r["k"]), num(r["top1_mean"]), num(r["top1_std"]), inum(r["n_seeds"]) or 0)
                         for r in mrows if r["variant"] == var and ok(r) and (inum(r["k"]) or 0) > 0)
            if pts:
                series.append((key, var, pts))
                ns += [p[3] for p in pts]
        orig = next((num(r["top1_mean"]) for r in mrows if r["variant"] == "original" and ok(r)), None)
        if not series:
            if budget == "MLP + attention" and model != MAIN_MODEL:
                print(f"note: {name}: no {budget} rows for {label} (not run); panel left out")
            else:
                R.na(f"{name}, panel {label} {budget}", "no rows with status ok")
            continue
        panels.append({"model": model, "label": label, "budget": budget, "series": series, "orig": orig})
    if not panels:
        return R.na(name, "no plottable rows")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE if n > 1 else SINGLE, 2.45), sharey=True, layout="constrained",
                             squeeze=False)
    used = []
    # one k axis for every panel, so panels compare directly (a k with no point in a panel was not run there)
    ks = sorted({pt[0] for p in panels for _, _, pts in p["series"] for pt in pts})
    for i, (ax, p) in enumerate(zip(axes[0], panels)):
        grid(ax)
        for key, var, pts in p["series"]:
            m = METHODS[key]
            x, y = [q[0] for q in pts], [q[1] for q in pts]
            ax.plot(x, y, color=m["color"], marker=m["marker"], ms=MS[m["marker"]], mec="white", mew=0.5, lw=1.2,
                    zorder=3, clip_on=False)
            for k, mean, sd, nn in pts:
                if nn > 1 and sd is not None:
                    ax.errorbar(k, mean, yerr=sd, fmt="none", ecolor=m["color"], elinewidth=0.7, capsize=1.6,
                                capthick=0.7, zorder=4)
            if key not in used:
                used.append(key)
        if p["orig"] is not None:
            ax.axhline(p["orig"], zorder=1.5, **ORIG_KW)
            tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(0.99, p["orig"] + 1.2, f"original {p['orig']:.1f}", transform=tr, ha="right", va="bottom",
                    fontsize=FONT - 1, color=C_MUTED)
        ax.set_xticks(ks)
        ax.set_xlim(min(ks) - 0.8, max(ks) + 0.8)
        ax.set_ylim(0, 86)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Converted blocks, k")
        if i == 0:
            ax.set_ylabel("k-NN top-1 (%)")
        panel_title(ax, "abcd"[i], f"{p['label']}, {p['budget']}")
    handles = [method_line_handle(k) for k in METHOD_ORDER if k in used] + [orig_handle()]
    fig.legend(handles=handles, loc="outside upper center", ncol=3 if len(handles) > 4 else len(handles))
    files = save(fig, out, "fig1_main_knn_vs_k")

    letters = "abcd"
    panel_txt = "; ".join(f"({letters[i]}) {p['label']}, {p['budget']}" for i, p in enumerate(panels))
    multi = sorted({(pt[0], pt[3]) for p in panels for _, _, pts in p["series"] for pt in pts if pt[3] > 1})
    multi_txt = (f"Error bars (k = {fmt_list(sorted({k for k, _ in multi}))}, MLP only): sample standard deviation over "
                 f"n = {fmt_list(sorted({nn for _, nn in multi}))} fit seeds; they are mostly smaller than the "
                 f"markers. All other points come from one fit seed (n = 1)." if multi else
                 "Every point comes from one fit seed (n = 1).")
    cap = (f"Zero-shot accuracy after converting the first k transformer blocks, without any retraining. "
           f"Each line is k-NN top-1 (%) against k for Mirror Surgery (weights projected onto the learned left-right "
           f"symmetry), SVD truncation keeping the same number of parameters, and a random-structure control (a "
           f"random involution with the same d+ and d- and a random pairing with the same counts). Dashed line: the "
           f"original model. Panels: {panel_txt}. In the MLP + attention panel, SVD and random structure are matched "
           f"to the V-local budget. {multi_txt} Protocol: {PROTOCOL}.")
    R.figures.append({"title": "Figure 1. Main result: zero-shot k-NN top-1 against converted blocks", "files": files,
                      "caption": cap, "sources": [("T_main_zeroshot.csv",
                                                  "model, model_label, k, variant, status, n_seeds, top1_mean, top1_std "
                                                  "(rows variant = original give the dashed line)")],
                      "notes": ["Rows `mirror_mlp_m1` (the M1 run's MLP-only Mirror at k = 6, 9, 12) are not drawn; "
                                "the MLP-only panel uses the X2 rows, which have the matched SVD and random rows.",
                                "Rows with status `not run` (for example k = 12 MLP only, DINOv2 MLP + attention) are "
                                "left out."]})


# ----------------------------------------------------------------------------------------------
# Figure 2: healing
# ----------------------------------------------------------------------------------------------
def fig2_healing(T: Tables, R: Registry, out: Path):
    name = "Figure 2 (healing)"
    rows = T.get("T_healing")
    if rows is None:
        return R.na(name, "T_healing.csv not found")
    style = {"mirror": "vlocal", "svd": "svd", "random": "random"}
    tick = {"mirror": "Mirror\nV-local", "svd": "SVD", "random": "Random\nstructure"}
    cats = []
    for meth in ("mirror", "svd", "random"):
        r = next((r for r in rows if r.get("method") == meth and r.get("status") == "complete"), None)
        if r is None or num(r.get("healed_top1")) is None:
            R.na(f"{name}, {meth}", "no complete row in T_healing")
            continue
        cats.append({"key": style[meth], "tick": f"{tick[meth]}\nk = {inum(r['k'])}", "zs": num(r["zero_shot_top1"]),
                     "healed": num(r["healed_top1"]), "orig": num(r.get("original_top1")), "k": inum(r["k"]),
                     "seed": r.get("seed"), "steps": inum(r.get("steps_done"))})
    vx = T.get("T_healing_vexact")
    vx_note = None
    if vx is None:
        R.na(f"{name}, V-exact full conversion healed point", "T_healing_vexact.csv not in the tables yet "
             "(rerun scripts/collect_results.py once runs/m3_vexact/ is collected)")
        vx_note = "missing"
    else:
        done = [r for r in vx if num(r.get("healed_top1")) is not None and not str(r.get("tiny", "")).lower() == "true"]
        if not done:
            R.na(f"{name}, V-exact healed point", "T_healing_vexact.csv has no healed_top1 values")
            vx_note = "missing"
        for r in sorted(done, key=lambda r: inum(r["k"]) or 0):
            full = str(r.get("full_conversion", "")).lower() == "true"
            cats.append({"key": "vexact", "tick": f"Mirror\nV-exact\nk = {inum(r['k'])}" + (" (all)" if full else ""),
                         "zs": num(r.get("zero_shot_top1")) if num(r.get("zero_shot_top1")) is not None
                         else num(r.get("m1_zero_shot_top1")),
                         "healed": num(r["healed_top1"]), "orig": num(r.get("original_top1")), "k": inum(r["k"]),
                         "seed": r.get("seed"), "steps": inum(r.get("steps_done"))})
    if not cats:
        return R.na(name, "no healed rows")

    fig, ax = plt.subplots(figsize=(SINGLE, 2.5), layout="constrained")
    grid(ax)
    orig = next((c["orig"] for c in cats if c["orig"] is not None), None)
    if orig is not None:
        ax.axhline(orig, zorder=1.5, **ORIG_KW)
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.99, orig + 1.2, f"original {orig:.1f}", transform=tr, ha="right", va="bottom",
                fontsize=FONT - 1, color=C_MUTED)
    for i, c in enumerate(cats):
        m = METHODS[c["key"]]
        ms = MS[m["marker"]] + 0.8
        if c["zs"] is not None:
            ax.annotate("", xy=(i, c["healed"]), xytext=(i, c["zs"]),
                        arrowprops={"arrowstyle": "-|>", "color": m["color"], "lw": 0.9, "shrinkA": 3.5,
                                    "shrinkB": 3.5, "mutation_scale": 6}, zorder=2)
            ax.plot(i, c["zs"], marker=m["marker"], ms=ms, mfc="white", mec=m["color"], mew=1.0, ls="none", zorder=3)
            # zero-shot value to the left of its marker, healed value to the right: no collision however close
            ax.text(i - 0.12, c["zs"], f"{c['zs']:.1f}", ha="right", va="center", fontsize=FONT - 1, color=C_MUTED)
        ax.plot(i, c["healed"], marker=m["marker"], ms=ms, mfc=m["color"], mec="white", mew=0.5, ls="none", zorder=4)
        # keep the healed value off the dashed original line when the two are close
        yl = c["healed"]
        if orig is not None and abs(yl - orig) < 3.0:
            yl = min(yl, orig) - 3.0
        ax.text(i + 0.12, yl, f"{c['healed']:.1f}", ha="left", va="center", fontsize=FONT - 1, color=C_INK)
    ax.set_xticks(range(len(cats)), [c["tick"] for c in cats])
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 86)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("k-NN top-1 (%)")
    ax.tick_params(axis="x", length=0)
    handles = [Line2D([], [], marker="o", ms=4.5, mfc="white", mec=C_MUTED, mew=1.0, ls="none", label="zero-shot"),
               Line2D([], [], marker="o", ms=4.5, mfc=C_MUTED, mec="white", mew=0.5, ls="none", label="healed"),
               orig_handle("original model")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    files = save(fig, out, "fig2_healing")

    steps = sorted({c["steps"] for c in cats if c["steps"]})
    pilot = [c for c in cats if c["key"] != "vexact"]
    cap = (f"Accuracy before and after healing. For each method, the hollow marker is the zero-shot k-NN top-1 (%) "
           f"right after conversion and the filled marker (arrow head) the k-NN top-1 after healing by distillation "
           f"from the original model on ImageNet train images, with the converted blocks kept in their constrained "
           f"form"
           + (f" ({fmt_list(steps)} steps)" if steps else "") + ". "
           f"Mirror V-local, SVD and random structure convert the first {pilot[0]['k'] if pilot else 'k'} blocks (MLP "
           f"and attention) of DINOv2-reg at matched parameters (the M3 pilot). ")
    if any(c["key"] == "vexact" for c in cats):
        cap += ("Mirror V-exact is exactly mirror-equivariant over the converted span; its point is healed with the "
                "same recipe and has no matched healed baseline. ")
    cap += (f"Dashed line: the original model. One healing run per method (seed 0, n = 1, no error bars). "
            f"Protocol: {PROTOCOL}; healing data from train images only.")
    notes = []
    for r in (vx or []):
        if r.get("note") and num(r.get("healed_top1")) is not None:
            notes.append(f"V-exact k = {inum(r['k'])}: {r['note']} (T_healing_vexact.csv, column gpu).")
    if vx_note:
        notes.append("The V-exact full-conversion healed point (k = 12) is not drawn: T_healing_vexact.csv is not in "
                     "results/paper/tables/ yet. STATUS.md reports it from runs/m3_vexact/k12/; the figure picks it up "
                     "automatically once scripts/collect_results.py writes that table.")
    R.figures.append({"title": "Figure 2. Healing: zero-shot and healed k-NN top-1", "files": files, "caption": cap,
                      "sources": [("T_healing.csv", "method, status, k, zero_shot_top1, healed_top1, original_top1, "
                                   "seed, steps_done"),
                                  ("T_healing_vexact.csv (when present)", "k, zero_shot_top1 (else m1_zero_shot_top1), "
                                   "healed_top1, original_top1, full_conversion")],
                      "notes": notes})


# ----------------------------------------------------------------------------------------------
# Figure 3: ablations
# ----------------------------------------------------------------------------------------------
ABL = [
    ("A1", "Operator (A1)", "per_layer",
     [("per_layer", "per-layer operator"), ("shared_span", "shared over the span"),
      ("shared_all", "shared over all layers"), ("identity", "identity, A = I"),
      ("random_operator", "random involution"), ("svd", "SVD, matched parameters"),
      ("random_structure", "random structure")]),
    ("A2", "Pairing (A2)", "mirror",
     [("mirror", "mirror matching"), ("hungarian_cycle", "Hungarian, exact projection"),
      ("hungarian", "Hungarian, approximate projection"), ("generic", "weight-cosine pairing"),
      ("random_pairing", "random pairing"), ("self_twins", "all self-twins")]),
    ("A4", "Block selection (A4)", "first",
     [("first", "first k blocks"), ("last", "last k blocks"), ("lowest_share", "lowest breaking share first")]),
    ("A5", "Projection strength (A5)", "alpha1.00", None),
]
K_MARK = [("o", True), ("s", False), ("D", True), ("^", False)]


def _abl_data(rows, model):
    data = defaultdict(dict)
    for r in rows:
        if r.get("model") != model or num(r.get("top1_mean")) is None:
            continue
        data[r["ablation"]][(r["variant"], inum(r["k"]))] = {
            "mean": num(r["top1_mean"]), "std": num(r.get("top1_std")), "n": inum(r.get("n_seeds")) or 0,
            "seeds": r.get("fit_seeds", ""), "note": r.get("note") or ""}
    return data


def fig3_ablations(T: Tables, R: Registry, out: Path):
    name = "Figure 3 (ablations)"
    rows = T.get("T_ablations_seeds")
    if rows is None:
        return R.na(name, "T_ablations_seeds.csv not found")
    per = T.get("T_ablations") or []
    main = T.get("T_main_zeroshot") or []
    model = MAIN_MODEL
    data = _abl_data(rows, model)
    orig = next((num(r["top1_mean"]) for r in main if r["model"] == model and r["variant"] == "original"
                 and r.get("status") == "ok"), None)
    if orig is None:
        print(f"note: {name}: original top-1 not found in T_main_zeroshot; no reference line")
    present = [a for a in ABL if data.get(a[0])]
    for a in ABL:
        if not data.get(a[0]):
            R.na(f"{name}, {a[1]}", f"no {a[0]} rows for {model} in T_ablations_seeds")
    if not present:
        return R.na(name, "no ablation rows")
    all_ks = sorted({k for a in present for (_, k) in data[a[0]]})
    kmark = {k: K_MARK[i % len(K_MARK)] for i, k in enumerate(all_ks)}

    fig = plt.figure(figsize=(DOUBLE, 4.15), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    slots = {"A1": gs[0, 0], "A2": gs[0, 1], "A4": gs[1, 0], "A5": gs[1, 1]}
    letters = iter("abcd")
    ns, info = [], []
    for abl, title, default, order in ABL:
        if not data.get(abl):
            continue
        d = data[abl]
        ax = fig.add_subplot(slots[abl])
        letter = next(letters)
        panel_title(ax, letter, title)
        ks = sorted({k for (_, k) in d})
        ns += [v["n"] for v in d.values()]
        info.append((letter, title, ks, sorted({v["n"] for v in d.values()})))
        if order is None:  # A5: line against alpha
            grid(ax, "y")
            for k in ks:
                pts = []
                for (v, kk), val in d.items():
                    if kk != k or not v.startswith("alpha"):
                        continue
                    a = num(v[len("alpha"):])
                    if a is not None:
                        pts.append((a, val["mean"], val["std"], val["n"]))
                pts.sort()
                mk, filled = kmark[k]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C_BLUE, lw=1.2, marker=mk, ms=MS[mk],
                        mfc=C_BLUE if filled else "white", mec=C_BLUE if not filled else "white",
                        mew=1.0 if not filled else 0.5, zorder=3, clip_on=False,
                        ls="-" if filled else (0, (3, 1.5)))
                for a, mean, sd, n in pts:
                    if n > 1 and sd:
                        ax.errorbar(a, mean, yerr=sd, fmt="none", ecolor=C_BLUE, elinewidth=0.7, capsize=1.6, zorder=4)
            vals = [v["mean"] for v in d.values()] + ([orig] if orig is not None else [])
            lo, hi = min(vals), max(vals)
            pad = max(0.25, 0.12 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            if orig is not None:
                ax.axhline(orig, zorder=1.5, **ORIG_KW)
            ax.set_xlim(-0.06, 1.06)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0\n(original)", "0.25", "0.5", "0.75", "1\n(default)"])
            ax.set_xlabel(r"Projection strength $\alpha$")
            ax.set_ylabel("k-NN top-1 (%)")
            continue
        # dot plot: variants down the y axis
        grid(ax, "y")
        known = [v for v, _ in order]
        variants = [v for v in known if any((v, k) in d for k in ks)]
        variants += sorted({v for (v, _) in d if v not in known})
        lab = dict(order)
        ylabels = []
        kept = defaultdict(list)
        for r in per:
            if r.get("model") == model and r.get("ablation") == abl and num(r.get("mlp_params_kept_frac")) is not None:
                kept[r["variant"]].append(num(r["mlp_params_kept_frac"]))
        for v in variants:
            t = lab.get(v, v)
            if abl == "A4" and len(ks) == 1:
                t = t.replace("k blocks", f"{ks[0]} blocks")
            if v == default:
                t += " (default)"
            notes = " ".join(d[(v, k)]["note"] for k in ks if (v, k) in d)
            if abl == "A4" and "A4/last" in notes:
                t += "\n(same blocks as last)"
            elif abl == "A4" and len(ks) == 1 and kept.get(v):
                t += f"\nMLP weights kept {np.mean(kept[v]):.2f}"
            ylabels.append(t)
        allv = [val["mean"] - (val["std"] or 0) for val in d.values()] + \
               [val["mean"] + (val["std"] or 0) for val in d.values()]
        if orig is not None:
            allv.append(orig)
        if min(allv) > 50:     # zoomed axis (block selection): differences are small
            lo, hi = min(allv), max(allv)
            pad = max(0.3, 0.15 * (hi - lo))
            xlim = (lo - pad, hi + pad)
        else:
            xlim = (0, 86)
        for j, k in enumerate(ks):
            off = (j - (len(ks) - 1) / 2) * 0.26
            mk, filled = kmark[k]
            for i, v in enumerate(variants):
                if (v, k) not in d:
                    continue
                val = d[(v, k)]
                col = C_BLUE if v == default else C_MUTED
                xerr = val["std"] if (val["n"] > 1 and val["std"]) else None
                ax.errorbar(val["mean"], i + off, xerr=xerr, fmt=mk, ms=MS[mk] + 0.4, color=col,
                            mfc=col if filled else "white", mec="white" if filled else col,
                            mew=0.5 if filled else 1.0, elinewidth=0.7, capsize=1.6, zorder=3)
                if len(ks) == 1:
                    span = xlim[1] - xlim[0]
                    right = (val["mean"] - xlim[0]) / span < 0.72
                    dx = (0.025 * span + (xerr or 0)) * (1 if right else -1)
                    ax.text(val["mean"] + dx, i, f"{val['mean']:.1f}", ha="left" if right else "right",
                            va="center", fontsize=FONT - 1, color=C_MUTED)
        if orig is not None:
            ax.axvline(orig, zorder=1.5, **ORIG_KW)
        ax.set_yticks(range(len(variants)), ylabels)
        ax.set_ylim(len(variants) - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*xlim)
        ax.set_xlabel("k-NN top-1 (%)")
        ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ms=4.4, color=C_BLUE, mec="white", mew=0.5, ls="none", label="default setting"),
               Line2D([], [], marker="o", ms=4.4, color=C_MUTED, mec="white", mew=0.5, ls="none", label="alternative"),
               orig_handle("original model")]
    if len(all_ks) > 1:
        for k in all_ks:
            mk, filled = kmark[k]
            handles.append(Line2D([], [], marker=mk, ms=MS[mk] + 0.4, ls="none", color=C_MUTED,
                                  mfc=C_MUTED if filled else "white", mec="white" if filled else C_MUTED,
                                  mew=0.5 if filled else 1.0, label=f"k = {k}"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    files = save(fig, out, "fig3_ablations")

    kk = sorted({k for _, _, ks, _ in info for k in ks})
    parts = []
    for letter, title, ks, nset in info:
        nn = [n for n in nset if n]
        parts.append(f"({letter}) {title}, k = {fmt_list(ks)}, n = {fmt_list(nn) if nn else 'not recorded'}")
    cap = (f"Zero-shot ablations on DINOv2-reg (MLP only, first k blocks converted, k = {fmt_list(kk)}). Each marker is "
           f"k-NN top-1 (%); the blue marker is the default Mirror Surgery setting and grey markers are the "
           f"alternatives; the dashed line is the original model. (a) Operator: per-layer operators against one "
           f"operator shared over the converted span, one shared over all layers, the identity (A = I, which zeroes "
           f"the odd weights), a random involution with the same d+ and d-, SVD at matched parameters and random "
           f"structure. (b) Pairing of mirror-twin neurons: mirror matching against plain Hungarian matching without "
           f"the involution constraint (with an exact cycle projection, or an approximate one), pairing by plain "
           f"weight cosine, random pairing with the same counts, and no pairs at all (every neuron its own twin). "
           f"(c) Which blocks are converted: the first k, the last k, or the blocks with the lowest breaking share "
           f"first; the labels give the fraction of MLP weights kept, so these settings are not at matched "
           f"parameters. (d) Projection strength alpha, interpolating between the original weights (alpha = 0) "
           f"and the full projection (alpha = 1). Panels and seeds: {'; '.join(parts)}. "
           f"{seeds_phrase(ns)}. Protocol: {PROTOCOL}.")
    notes = ["A3 (fit-set size) is in T_ablations_seeds but is not drawn (not requested for this figure)."]
    unm = [r for r in rows if r.get("model") == model and r.get("ablation") in ("A1", "A2")
           and str(r.get("params_matched", "")).lower() == "false" and num(r.get("params_vs_first_k_mean")) is not None]
    if unm:
        notes.append("Alternatives in (a) and (b) do not all keep the default's parameter count (params_vs_first_k_mean "
                     "in T_ablations_seeds.csv): " + ", ".join(
                         f"{r['variant']} {num(r['params_vs_first_k_mean']):.2f}" for r in
                         sorted(unm, key=lambda r: -num(r["params_vs_first_k_mean"]))) + ".")
    a4 = [r for r in rows if r.get("model") == model and r.get("ablation") == "A4"
          and str(r.get("params_matched", "")).lower() == "false"]
    if a4:
        notes.append("Panel (c) is not parameter-matched: " + ", ".join(
            f"{r['variant']} keeps {num(r['params_vs_first_k_mean']):.2f} times the default's MLP weights" for r in a4)
            + ", so its gain over the first k blocks partly reflects more weights kept.")
    dflt = next((r for r in rows if r.get("model") == model and r.get("ablation") == "A1"
                 and r.get("variant") == "per_layer" and num(r.get("canonical_top1_mean")) is not None), None)
    if dflt:
        notes.append(f"The default marker is the ablation script's own value ({num(dflt['top1_mean']):.2f} at k = "
                     f"{inum(dflt['k'])}); the canonical Mirror V-local, MLP only number is the X2 value in "
                     f"T_main_zeroshot ({num(dflt['canonical_top1_mean']):.2f}, Figure 1). The two are the same setting; "
                     "the ablation script reloads the operators from their float32 file, a numerical difference.")
    R.figures.append({"title": "Figure 3. Zero-shot ablations (A1, A2, A4, A5)", "files": files, "caption": cap,
                      "sources": [("T_ablations_seeds.csv", "model, ablation, variant, k, n_seeds, fit_seeds, top1_mean, "
                                   "top1_std, note"),
                                  ("T_ablations.csv", "mlp_params_kept_frac (labels of panel c)"),
                                  ("T_main_zeroshot.csv", "top1_mean of model = dinov2_vitb14_reg, variant = original "
                                   "(dashed line)")],
                      "notes": notes})


# ----------------------------------------------------------------------------------------------
# Figure 4: symmetry tax (F1)
# ----------------------------------------------------------------------------------------------
def fig4_tax(T: Tables, R: Registry, out: Path):
    name = "Figure 4 (symmetry tax, F1)"
    summ_files = T.glob("ext_f1_*_f1_blocks_summary.csv")
    seed_files = T.glob("ext_f1_*_f1_blocks_by_seed.csv")
    if not summ_files:
        return R.na(name, "no ext_f1_*_f1_blocks_summary.csv in the tables")
    summ = defaultdict(list)
    for f in summ_files:
        for r in T.get(f) or []:
            if num(r.get("tax_formula_mean")) is not None:
                summ[r["model"]].append(r)
    by_seed = defaultdict(list)
    for f in seed_files:
        for r in T.get(f) or []:
            if num(r.get("tax_formula")) is not None and num(r.get("saving_realised")) is not None:
                by_seed[r["model"]].append(r)
    models = [m for m in MODEL_ORDER if m in summ] + sorted(m for m in summ if m not in MODELS)
    if not models:
        return R.na(name, "F1 tables have no rows")

    fig = plt.figure(figsize=(DOUBLE, 2.55), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    grid(ax)
    nmax = 0
    ns = []
    for mdl in models:
        st = MODELS.get(mdl, {"label": mdl, "color": C_MUTED, "marker": "o"})
        rows = sorted(summ[mdl], key=lambda r: inum(r["block"]))
        b = [inum(r["block"]) for r in rows]
        tf = [num(r["tax_formula_mean"]) for r in rows]
        sr = [num(r["saving_realised_mean"]) for r in rows]
        sd = [num(r.get("saving_realised_std")) for r in rows]
        n = [inum(r.get("n_seeds")) or 0 for r in rows]
        ns += n
        nmax = max(nmax, max(b))
        ax.plot(b, tf, color=st["color"], lw=1.1, zorder=2)
        ax.plot(b, sr, ls="none", marker=st["marker"], ms=MS[st["marker"]], mfc="white", mec=st["color"], mew=0.9,
                zorder=3, clip_on=False)
        for bb, s, d, nn in zip(b, sr, sd, n):
            if nn > 1 and d:
                ax.errorbar(bb, s, yerr=d, fmt="none", ecolor=st["color"], elinewidth=0.6, capsize=1.2, zorder=3)
        if by_seed.get(mdl):
            x = [num(r["tax_formula"]) for r in by_seed[mdl]]
            y = [num(r["saving_realised"]) for r in by_seed[mdl]]
            bx.plot(x, y, ls="none", marker=st["marker"], ms=MS[st["marker"]], mfc="white", mec=st["color"], mew=0.9,
                    zorder=3, clip_on=False)
    ax.set_xlim(0.5, nmax + 0.5)
    ax.set_xticks(range(1, nmax + 1))
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction of MLP weights saved")
    panel_title(ax, "a", "Symmetry tax by block: predicted (line) and realised (markers)")
    if any(by_seed.values()):
        grid(bx, "both")
        lim = (0, 0.52)
        bx.plot(lim, lim, color=C_MUTED, lw=0.7, zorder=1)
        bx.text(0.245, 0.19, "y = x", ha="left", va="top", fontsize=FONT - 1, color=C_MUTED)
        bx.set_xlim(*lim)
        bx.set_ylim(*lim)
        bx.set_aspect("equal")
        bx.set_xlabel("Predicted saving (formula)")
        bx.set_ylabel("Realised saving (converted weights)")
        panel_title(bx, "b", "Predicted against realised, every block")
        tm = [abs(num(r.get("tax_minus_saving")) or 0) for v in by_seed.values() for r in v]
        if tm:
            bx.text(0.03, 0.97, f"largest |difference| {max(tm):.1e}", transform=bx.transAxes, ha="left", va="top",
                    fontsize=FONT - 1, color=C_MUTED)
    else:
        bx.set_visible(False)
        R.na(f"{name}, panel b", "no ext_f1_*_f1_blocks_by_seed.csv")
    handles = [Line2D([], [], color=MODELS.get(m, {}).get("color", C_MUTED), lw=1.1,
                      marker=MODELS.get(m, {}).get("marker", "o"), ms=MS[MODELS.get(m, {}).get("marker", "o")],
                      mfc="white", mew=0.9, label=MODELS.get(m, {}).get("label", m)) for m in models]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    files = save(fig, out, "fig4_symmetry_tax")

    nseed = {m: sorted({inum(r.get("n_seeds")) or 0 for r in summ[m]}) for m in models}
    nt = "; ".join(f"{MODELS.get(m, {}).get('label', m)} n = {fmt_list(v)}" for m, v in nseed.items())
    cap = ("The symmetry tax: the share of each MLP's weights that Mirror Surgery removes, block by block. "
           "(a) Predicted saving from the formula 1/2 - f (d+ - d-) / (2 h d) per weight matrix, averaged over fc1 "
           "and fc2 (line; f is the number of self-twin neurons, d+ and d- the dimensions of the even and odd "
           "subspaces, h the hidden width, d the model width), and the saving realised by the converted weights "
           "(hollow markers). (b) The same two quantities for every block, fit seed and model, with the line y = x. "
           "Realised saving is counted by the surgery code from the same twin and operator counts, so the "
           "agreement (largest difference at rounding level) checks the bookkeeping rather than the theory. "
           f"Seeds: {nt} (fit seeds; error bars, the sample standard deviation over fit seeds, are smaller than "
           "the markers). Operators fitted on ImageNet train images.")
    R.figures.append({"title": "Figure 4. Symmetry tax (F1): predicted against realised", "files": files,
                      "caption": cap,
                      "sources": [(", ".join(summ_files), "model, block, tax_formula_mean, saving_realised_mean, "
                                   "saving_realised_std, n_seeds"),
                                  (", ".join(seed_files), "model, fit_seed, block, tax_formula, saving_realised, "
                                   "tax_minus_saving")],
                      "notes": ["The F1 outputs under results/paper/scripts/f1_*/ are read through their copies in "
                                "results/paper/tables/ (ext_f1_*), as the collector writes them."]})


# ----------------------------------------------------------------------------------------------
# Figure 5: generality
# ----------------------------------------------------------------------------------------------
def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    if len(xs) == 1:
        return xs[0], None, 1
    return float(np.mean(xs)), float(np.std(xs, ddof=1)), len(xs)


def fig5_generality(T: Tables, R: Registry, out: Path):
    name = "Figure 5 (generality)"
    rows = T.get("T_generality")
    if rows is None:
        return R.na(name, "T_generality.csv not found")
    p4 = T.get("T_generality_p4") or []
    by = defaultdict(list)
    for r in rows:
        if num(r.get("row1_value")) is not None:
            by[r["model"]].append(r)
    models = [m for m in MODEL_ORDER if m in by] + sorted(m for m in by if m not in MODELS)
    for m in ("dinov2_vits14_reg", "dinov2_vitl14_reg"):
        if m not in by:
            R.na(f"{name}, {MODELS[m]['label']}", "no row in T_generality yet")
    if not models:
        return R.na(name, "no rows")
    ratio_bar = next((num(r["threshold"]) for r in p4 if str(r.get("check", "")).endswith(".ratio")), None)
    share_bar = next((num(r["threshold"]) for r in p4 if str(r.get("check", "")).endswith(".share")), None)

    st = {}
    for m in models:
        rs = by[m]
        ratio = _mean_sd([num(r["row1_value"]) for r in rs])
        shares, nulls = [], []
        for r in rs:
            s4, s1, s2 = num(r.get("row4_value")), num(r.get("s1")), num(r.get("s2"))
            if s4 is None:
                continue
            which = "s1" if (s1 is not None and (s2 is None or abs(s4 - s1) <= abs(s4 - s2))) else "s2"
            shares.append(s4)
            nulls.append(num(r.get(f"{which}_null")))
        drops = {}
        for key, col in (("mirror_mlp", "top1_surgery_half"), ("svd", "top1_svd_half"), ("random", "top1_random_half")):
            drops[key] = _mean_sd([num(r["top1_original"]) - num(r[col]) for r in rs
                                   if num(r.get("top1_original")) is not None and num(r.get(col)) is not None])
        st[m] = {"ratio": ratio, "share": _mean_sd(shares), "null": _mean_sd(nulls), "drops": drops,
                 "k": sorted({inum(r.get("k_half")) for r in rs if inum(r.get("k_half"))}),
                 "orig": _mean_sd([num(r.get("top1_original")) for r in rs]),
                 "outcome": sorted({r.get("outcome", "") for r in rs}), "n": len(rs),
                 "status": sorted({r.get("status", "") for r in rs})}

    weak = [m for m in models if m == "mae_b16"]
    ticks = [MODELS.get(m, {}).get("label", m) + ("$^{\\dagger}$" if m in weak else "") for m in models]
    fig = plt.figure(figsize=(DOUBLE, 2.75), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4])
    ax, bx, cx = (fig.add_subplot(gs[0, i]) for i in range(3))
    x = np.arange(len(models))
    bar_kw = {"color": C_MUTED, "lw": 0.7, "ls": (0, (1.5, 1.5)), "zorder": 1.5}

    def model_ticks(a):
        a.set_xticks(x, ticks, rotation=35, ha="right", rotation_mode="anchor")
        a.set_xlim(-0.6, len(models) - 0.4)
        a.tick_params(axis="x", length=0)

    # (a) fit ratio
    grid(ax)
    for i, m in enumerate(models):
        mean, sd, n = st[m]["ratio"]
        ax.errorbar(i, mean, yerr=sd if (n > 1 and sd) else None, fmt="o", ms=4.4, color=C_INK, mec="white", mew=0.5,
                    elinewidth=0.7, capsize=1.6, zorder=3)
        ax.text(i, mean * 1.12, f"{mean:.1f}", ha="center", va="bottom", fontsize=FONT - 1, color=C_MUTED)
    ax.set_yscale("log")
    if ratio_bar is not None:
        ax.axhline(ratio_bar, **bar_kw)
    vals = [st[m]["ratio"][0] for m in models]
    lo, hi = min(vals + [ratio_bar or vals[0]]) / 1.5, max(vals) * 1.6
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    model_ticks(ax)
    ax.set_ylabel("Operator fit, e(null) / e(A)")
    panel_title(ax, "a", "Structure: operator fit")
    # (b) breaking share against matched null
    grid(bx)
    for i, m in enumerate(models):
        s, ssd, sn = st[m]["share"]
        nl, _, _ = st[m]["null"]
        if s is None:
            continue
        if nl is not None:
            bx.plot([i, i], [s, nl], color=C_GRID, lw=1.6, zorder=1.8, solid_capstyle="butt")
            bx.plot(i, nl, marker="o", ms=4.4, mfc="white", mec=C_NULL, mew=1.0, ls="none", zorder=3)
        bx.errorbar(i, s, yerr=ssd if (sn > 1 and ssd) else None, fmt="o", ms=4.4, color=C_INK, mec="white",
                    mew=0.5, elinewidth=0.7, capsize=1.6, zorder=4)
    if share_bar is not None:
        bx.axhline(share_bar, **bar_kw)
    bx.set_ylim(0, 0.5)
    model_ticks(bx)
    bx.set_ylabel("MLP breaking share")
    panel_title(bx, "b", "Structure: breaking share")
    # (c) zero-shot drop
    grid(cx)
    keys = ["mirror_mlp", "svd", "random"]
    w = 0.26
    ymax = 0
    for i, m in enumerate(models):
        for j, key in enumerate(keys):
            mean, sd, n = st[m]["drops"][key]
            if mean is None:
                continue
            draw_bar(cx, i + (j - 1) * w, mean, w * 0.92, key, healed=False, err=sd if n > 1 else None)
            ymax = max(ymax, mean + (sd or 0))
    model_ticks(cx)
    cx.set_ylim(0, max(10, ymax * 1.05))
    kh = sorted({k for m in models for k in st[m]["k"]})
    cx.set_ylabel("Drop in k-NN top-1 (points)")
    panel_title(cx, "c", f"Zero-shot drop, first {fmt_list(kh)} blocks, MLP only")
    handles = [Line2D([], [], marker="o", ms=4.4, color=C_INK, mec="white", mew=0.5, ls="none", label="model value"),
               Line2D([], [], marker="o", ms=4.4, mfc="white", mec=C_NULL, mew=1.0, ls="none",
                      label="matched random null"),
               Line2D([], [], label="pre-registered P4 bar", **{k: v for k, v in bar_kw.items() if k != "zorder"})]
    handles += [Patch(facecolor=tint(METHODS[k]["color"]), edgecolor=METHODS[k]["color"], hatch=METHODS[k]["hatch"],
                      linewidth=0.7, label=lab) for k, lab in
                (("mirror_mlp", METHODS["mirror_mlp"]["label"]), ("svd", "SVD, matched parameters"),
                 ("random", "random structure"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    files = save(fig, out, "fig5_generality")

    nn = sorted({st[m]["n"] for m in models})
    labs = ", ".join(MODELS.get(m, {}).get("label", m) for m in models)
    multi = [MODELS.get(m, {}).get("label", m) for m in models if st[m]["n"] > 1]
    cap = (f"Does the mirror structure carry over to other vision transformers? Models: {labs} (B/16 or B/14 "
           f"backbones). (a) Operator fit over layers 1 to 6: held-out error of a matched random involution over that "
           f"of the fitted mirror operator (higher means a cleaner symmetry; log scale; decision row 1). (b) MLP "
           f"symmetry-breaking share over blocks 1 to 6 (decision row 4: the worse of fc1 and fc2; lower means more "
           f"structure), with the matched random null for the same weight matrix (hollow). Dotted lines: the "
           f"pre-registered P4 bars. (c) Zero-shot drop in k-NN top-1 (original minus converted, points) after "
           f"converting the first {fmt_list(kh)} blocks (MLP only), for Mirror, SVD at matched parameters and random "
           f"structure. "
           + (f"{fmt_list(multi)}: mean over n = {max(nn)} fit seeds, error bars the sample standard deviation "
              f"(mostly smaller than the markers); other models one fit seed (n = 1). " if multi else
              "One fit seed per model (n = 1). ")
           + ("Dagger: MAE's CLS token gives weak k-NN accuracy (original "
              f"{st['mae_b16']['orig'][0]:.1f}%), so its drop is not comparable with the others; the plan measures MAE "
              f"with the Tier-2 linear probe instead. " if "mae_b16" in models else "")
           + f"Protocol: {PROTOCOL}.")
    notes = ["Panel c plots differences of table values: top1_original minus top1_surgery_half, top1_svd_half or "
             "top1_random_half, averaged over the fit-seed rows of a model.",
             "MAE has no random-structure value (top1_random_half is empty) and no e1_meta.json (status partial).",
             "Panel b: the null is s1_null or s2_null, whichever matches the matrix that decision row 4 picked "
             "(fc1 for every model here except MAE, where row 4 is fc2)."]
    R.figures.append({"title": "Figure 5. Generality across vision transformers", "files": files, "caption": cap,
                      "sources": [("T_generality.csv", "model, fit_seed, status, row1_value, row4_value, s1, s2, "
                                   "s1_null, s2_null, top1_original, top1_surgery_half, top1_svd_half, "
                                   "top1_random_half, k_half"),
                                  ("T_generality_p4.csv", "check, threshold (dotted P4 bars)")],
                      "notes": notes})


# ----------------------------------------------------------------------------------------------
# Figure 6: cost
# ----------------------------------------------------------------------------------------------
K_FILL = ["none", "bottom", "full", "left"]


def fig6_cost(T: Tables, R: Registry, out: Path):
    name = "Figure 6 (cost, M2)"
    rows = T.get("T_latency")
    if rows is None:
        return R.na(name, "T_latency.csv not found")
    keymap = {"mlp_only": "mirror_mlp", "vlocal": "vlocal", "vexact": "vexact", "svd": "svd", "prune": "prune"}
    pts = [r for r in rows if r.get("variant") in keymap and r.get("mode_used") == "compile"
           and num(r.get("speedup_vs_dense")) is not None and num(r.get("flop_speedup_vs_dense")) is not None]
    if not pts:
        return R.na(name, "no compiled rows with measured and analytic speed-ups")
    batches = sorted({inum(r["batch"]) for r in pts})
    ks = sorted({inum(r["k"]) for r in pts})
    kfill = {k: K_FILL[i % len(K_FILL)] for i, k in enumerate(ks)}
    xs = [num(r["flop_speedup_vs_dense"]) for r in pts]
    ys = [num(r["speedup_vs_dense"]) for r in pts]
    xlim = (min(xs + [1.0]) - 0.03, max(xs) + 0.03)
    ylim = (min(ys) - 0.06, max(ys + [1.0]) + 0.08)

    fig, axes = plt.subplots(1, len(batches), figsize=(DOUBLE, 2.9), sharex=True, sharey=True, layout="constrained",
                             squeeze=False)
    present = []
    for i, (ax, b) in enumerate(zip(axes[0], batches)):
        grid(ax, "both")
        lo, hi = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], color=C_MUTED, lw=0.7, zorder=1)
        ax.axhline(1.0, zorder=1, **ORIG_KW)
        for v in ("mlp_only", "vlocal", "vexact", "svd", "prune"):
            key = keymap[v]
            m = METHODS[key]
            vr = sorted((r for r in pts if r["variant"] == v and inum(r["batch"]) == b), key=lambda r: inum(r["k"]))
            if not vr:
                continue
            if key not in present:
                present.append(key)
            ax.plot([num(r["flop_speedup_vs_dense"]) for r in vr], [num(r["speedup_vs_dense"]) for r in vr],
                    color=m["color"], lw=0.6, zorder=2)
            for r in vr:
                ax.plot(num(r["flop_speedup_vs_dense"]), num(r["speedup_vs_dense"]), marker=m["marker"],
                        ms=MS[m["marker"]] + 0.6, color=m["color"], fillstyle=kfill[inum(r["k"])],
                        mfc=m["color"], mfcalt="white", mec=m["color"], mew=0.9, ls="none", zorder=3)
                if str(r.get("timing_flag", "")).lower() == "true":
                    ax.annotate("outlier?", (num(r["flop_speedup_vs_dense"]), num(r["speedup_vs_dense"])),
                                xytext=(5, -2), textcoords="offset points", fontsize=FONT - 1.5, color=C_MUTED,
                                ha="left", va="top", zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Analytic speed-up (FLOPs)")
        if i == 0:
            ax.set_ylabel("Measured speed-up")
        panel_title(ax, "abcdef"[i], f"batch {b}")
    handles = [Line2D([], [], color=METHODS[k]["color"], marker=METHODS[k]["marker"], ms=MS[METHODS[k]["marker"]] + 0.6,
                      mec=METHODS[k]["color"], lw=0.6,
                      label={"mirror_mlp": "Mirror V-local, MLP only", "vlocal": "Mirror V-local, MLP + attention",
                             "vexact": "Mirror V-exact, MLP + attention", "svd": "SVD", "prune": "MLP width pruning"}[k])
               for k in METHOD_ORDER if k in present]
    row2 = [Line2D([], [], color=C_MUTED, marker="o", ms=4.6, ls="none", fillstyle=kfill[k], mfc=C_MUTED,
                   mfcalt="white", mec=C_MUTED, mew=0.9, label=f"k = {k}") for k in ks]
    row2 += [Line2D([], [], color=C_MUTED, lw=0.7, label="y = x (all of the analytic saving)"),
             Line2D([], [], label="dense model (speed-up 1)", **ORIG_KW)]
    # three columns: the methods fill the first rows and the keys the rows below; the legend fills column by column,
    # so the row-major list is reordered
    ncol = 3
    blank = Line2D([], [], ls="none", label=" ")
    handles = handles + [blank] * (-len(handles) % ncol)
    row2 = row2 + [blank] * (-len(row2) % ncol)
    items = handles + row2
    nrow = len(items) // ncol
    fig.legend(handles=[items[r * ncol + c] for c in range(ncol) for r in range(nrow)], loc="outside upper center",
               ncol=ncol)
    files = save(fig, out, "fig6_cost")

    cap = (f"Measured against analytic speed-up of the converted DINOv2-reg models, relative to the dense model in "
           f"the same implementation (the folded original), for batch sizes {fmt_list(batches)} (panels). Each "
           f"marker is one method at one k (k = {fmt_list(ks)} converted blocks; marker fill shows k, and thin lines "
           f"join the same method). x: dense multiply-adds over the variant's (counted by the timing script); y: dense "
           f"latency over the variant's (median of timed forward passes, bf16, torch.compile, one A100). Points on "
           f"the grey y = x line would realise the whole analytic saving; points below the dashed line are slower "
           f"than the dense model. SVD and width pruning keep the same parameter count as the matched Mirror "
           f"variant. One timing run per point (seed 0, n = 1)."
           + (" A point marked 'outlier?' measures faster than its FLOPs allow and is flagged as a probable timing "
              "outlier." if any(str(r.get("timing_flag", "")).lower() == "true" for r in pts) else ""))
    R.figures.append({"title": "Figure 6. Cost: measured against analytic speed-up (M2)", "files": files,
                      "caption": cap,
                      "sources": [("T_latency.csv", "variant, k, batch, mode_used = compile, speedup_vs_dense, "
                                   "flop_speedup_vs_dense")],
                      "notes": fig6_notes(pts)})


def fig6_notes(pts) -> list[str]:
    notes = ["Eager-mode rows and the hub original are not drawn."]
    vl = sorted((r for r in pts if r.get("variant") == "vlocal" and num(r.get("collector_flop_speedup")) is not None
                 and abs(num(r["collector_flop_speedup"]) - num(r["flop_speedup_vs_dense"])) > 5e-4),
                key=lambda r: (inum(r["k"]), inum(r["batch"])))
    if vl:
        r = vl[0]
        notes.append(f"x is the timing script's own count (flop_speedup_vs_dense, 'analytic (M2)'), the count of the "
                     f"implementation that was timed. The collector's structured count (collector_flop_speedup) differs "
                     f"only for V-local ({num(r['flop_speedup_vs_dense']):.3f} against "
                     f"{num(r['collector_flop_speedup']):.3f} at k = {inum(r['k'])}), because the timed V-local stores "
                     "Wo dense (T_latency notes); it is empty for pruning.")
    flagged = [r for r in pts if str(r.get("timing_flag", "")).lower() == "true"]
    for r in flagged:
        notes.append(f"Marked 'outlier?': {r.get('label') or r['variant']} k = {inum(r['k'])}, batch {inum(r['batch'])} "
                     f"({num(r['speedup_vs_dense']):.3f} measured against {num(r['flop_speedup_vs_dense']):.3f} analytic), "
                     "flagged in T_latency as a probable timing outlier; it is drawn but not used for any claim.")
    return notes


# ----------------------------------------------------------------------------------------------
# Figure 7: Tier-2 linear probes and transfer
# ----------------------------------------------------------------------------------------------
DATASET_LABEL = {"cifar100": "CIFAR-100", "dtd": "DTD", "flowers102": "Flowers-102", "pets": "Pets",
                 "ade20k": "ADE20K"}


def _t2_style(r) -> tuple[str | None, bool]:
    fam, meth, lab = (r.get("family") or "").lower(), (r.get("method") or "").lower(), r.get("label") or ""
    healed = fam.startswith("healed")
    if meth in ("vlocal", "vexact", "svd", "random", "prune"):
        return meth, healed
    if meth == "mirror":
        if "V-exact" in lab:
            return "vexact", healed
        if fam == "mlp" or "MLP only" in lab:
            return "mirror_mlp", healed
        return "vlocal", healed
    return None, healed


def _bar_order(item):
    key, healed, k = item
    return (k or 0, healed, METHOD_ORDER.index(key))


def fig7_tier2(T: Tables, R: Registry, out: Path):
    name = "Figure 7 (Tier-2 probes)"
    lin = T.get("T_tier2")
    tr = T.get("T_tier2_transfer")
    seg = T.get("T_tier2_seg")
    if lin is None:
        R.na(f"{name}, ImageNet linear probe", "T_tier2.csv not found")
    if tr is None:
        R.na(f"{name}, transfer", "T_tier2_transfer.csv not found")
    if seg is None:
        R.na(f"{name}, ADE20K segmentation panel", "T_tier2_seg.csv not in the tables yet")
    if seg and not any(_t2_style(r)[0] for r in seg if num(r.get("mIoU")) is not None):
        R.na(f"{name}, ADE20K segmentation panel", "T_tier2_seg.csv has only the original model so far")
        seg = None
    panels = [p for p, rows in (("lin", lin), ("tr", tr), ("seg", seg)) if rows]
    if not panels:
        return R.na(name, "no Tier-2 tables")

    # (a) ImageNet linear probe: groups by model and k
    groups_a, origs_a, mae_only = [], {}, []
    if lin:
        for mdl in MODEL_ORDER:
            mr = [r for r in lin if r["model"] == mdl and num(r.get("top1")) is not None]
            if not mr:
                continue
            o = next((num(r["top1"]) for r in mr if r.get("family") == "original"), None)
            origs_a[mdl] = o
            byk = defaultdict(list)
            for r in mr:
                key, healed = _t2_style(r)
                if key is None:
                    continue
                byk[inum(r["k"])].append((key, healed, num(r["top1"]), r))
            if not byk:
                mae_only.append(MODELS[mdl]["label"])
            for k in sorted(byk):
                bars = sorted(byk[k], key=lambda t: (t[1], METHOD_ORDER.index(t[0])))
                groups_a.append({"model": mdl, "k": k, "bars": bars, "orig": o})
    n_seg_models = len({r["model"] for r in seg}) if seg else 0
    widths = {"lin": 1.25, "tr": 1.0, "seg": 0.25 + 0.25 * max(n_seg_models, 1)}
    use = [p for p in ("lin", "tr", "seg") if (p == "lin" and groups_a) or (p == "tr" and tr) or (p == "seg" and seg)]
    if not use:
        return R.na(name, "no plottable Tier-2 rows")
    fig = plt.figure(figsize=(DOUBLE, 2.75), layout="constrained")
    gs = fig.add_gridspec(1, len(use), width_ratios=[widths[p] for p in use])
    letters = iter("abc")
    present = set()
    any_healed = any_zs = False
    cap_parts, sources = [], []
    for pi, p in enumerate(use):
        ax = fig.add_subplot(gs[0, pi])
        grid(ax)
        ax.tick_params(axis="x", length=0)
        letter = next(letters)
        if p == "lin":
            w = 0.19
            x0 = 0.0
            centres, labels = [], []
            for g in groups_a:
                nb = len(g["bars"])
                xs = [x0 + (j - (nb - 1) / 2) * w for j in range(nb)]
                for xx, (key, healed, v, _) in zip(xs, g["bars"]):
                    draw_bar(ax, xx, v, w * 0.9, key, healed)
                    present.add(key)
                    any_healed |= healed
                    any_zs |= not healed
                    # value inside the top of a tall bar (clear of the original line), above a short one;
                    # a white box keeps it legible over hatching
                    inside = v > 30
                    ax.text(xx, v - 1.5 if inside else v + 1.5, f"{v:.1f}", ha="center",
                            va="top" if inside else "bottom", fontsize=FONT - 1.5, color=C_INK, rotation=90,
                            zorder=5, bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"})
                if g["orig"] is not None:
                    ax.plot([xs[0] - w * 0.6, xs[-1] + w * 0.6], [g["orig"]] * 2, zorder=4, **ORIG_KW)
                centres.append(x0)
                labels.append(f"{MODELS[g['model']]['label']}\nk = {g['k']}")
                x0 += nb * w + 0.32
            ax.set_xticks(centres, labels)
            ax.set_xlim(centres[0] - 0.5, centres[-1] + 0.5)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_ylabel("Linear-probe top-1 (%)")
            panel_title(ax, letter, "ImageNet linear probe")
            ks_by_key = defaultdict(set)
            for g in groups_a:
                for key, healed, _, _ in g["bars"]:
                    ks_by_key[(key, healed)].add(g["k"])
            cap_parts.append(
                f"({letter}) ImageNet linear-probe top-1 (%) of the frozen features, grouped by model and number of "
                f"converted blocks k (dashed segments: each model's original). Bars: "
                + "; ".join(f"{METHODS[key]['label']}{' healed' if healed else ''} (k = {fmt_list(sorted(v))})"
                            for (key, healed), v in sorted(ks_by_key.items(),
                                                           key=lambda t: (t[0][1], METHOD_ORDER.index(t[0][0]))))
                + ".")
            sources.append(("T_tier2.csv", "model, variant, label, family, method, k, top1 (family = original gives "
                            "the dashed segments)"))
        elif p == "tr":
            ds = [d for d in ("cifar100", "dtd", "flowers102", "pets") if any(r["dataset"] == d for r in tr)]
            ds += sorted({r["dataset"] for r in tr} - set(ds))
            items = sorted({(_t2_style(r)[0], _t2_style(r)[1], inum(r["k"])) for r in tr
                            if r.get("family") != "original" and _t2_style(r)[0]}, key=_bar_order)
            nb = len(items)
            w = min(0.8 / max(nb, 1), 0.2)
            labels, ymax = [], 0.0
            for i, d in enumerate(ds):
                dr = [r for r in tr if r["dataset"] == d]
                o = next((num(r["top1"]) for r in dr if r.get("family") == "original"), None)
                for j, it in enumerate(items):
                    r = next((r for r in dr if (_t2_style(r)[0], _t2_style(r)[1], inum(r["k"])) == it), None)
                    if r is None or num(r.get("drop_vs_original")) is None:
                        continue
                    v = num(r["drop_vs_original"])
                    xx = i + (j - (nb - 1) / 2) * w
                    draw_bar(ax, xx, v, w * 0.9, it[0], it[1])
                    if abs(v) < 0.005:   # a bar of height 0 is invisible: say so
                        ax.text(xx, 0.08, "0", ha="center", va="bottom", fontsize=FONT - 1.5, color=C_MUTED)
                    present.add(it[0])
                    any_healed |= it[1]
                    any_zs |= not it[1]
                    ymax = max(ymax, v)
                # with a third panel there is less room: drop the word "original" (the caption says it)
                otxt = f"({o:.1f})" if "seg" in use else f"(original {o:.1f})"
                labels.append(f"{DATASET_LABEL.get(d, d)}\n{otxt}" if o is not None else DATASET_LABEL.get(d, d))
            ax.set_xticks(range(len(ds)), labels)
            ax.set_xlim(-0.55, len(ds) - 0.45)
            ax.set_ylim(0, max(2.0, ymax * 1.12))
            ax.set_ylabel("Drop in linear-probe top-1 (points)")
            model_lab = sorted({r.get("model_label") or r["model"] for r in tr})
            panel_title(ax, letter, f"Transfer, {', '.join(model_lab)}")
            cap_parts.append(
                f"({letter}) Linear-probe transfer of {', '.join(model_lab)} to {fmt_list([DATASET_LABEL.get(d, d) for d in ds])}: "
                f"drop in top-1 against the original model (points; the original's top-1 is in brackets under each name; '0' "
                f"marks a drop of zero). Bars, left to right: "
                + "; ".join(f"{METHODS[k]['label']}{' healed' if h else ''}, k = {kk}" for k, h, kk in items) + ".")
            sources.append(("T_tier2_transfer.csv", "model_label, dataset, variant, label, family, method, k, top1, "
                            "drop_vs_original"))
        else:  # ADE20K segmentation (when the collector has the table)
            mods = [m for m in MODEL_ORDER if any(r["model"] == m for r in seg)]
            items = sorted({(_t2_style(r)[0], _t2_style(r)[1], inum(r.get("k"))) for r in seg
                            if _t2_style(r)[0] and num(r.get("mIoU")) is not None}, key=_bar_order)
            nb = len(items)
            w = min(0.8 / max(nb, 1), 0.3)
            for i, mdl in enumerate(mods):
                mr = [r for r in seg if r["model"] == mdl]
                o = next((num(r["mIoU"]) for r in mr if (r.get("family") or "").lower() == "original"
                          or (r.get("variant") or "") == "original"), None)
                for j, it in enumerate(items):
                    r = next((r for r in mr if (_t2_style(r)[0], _t2_style(r)[1], inum(r.get("k"))) == it), None)
                    if r is None or num(r.get("mIoU")) is None:
                        continue
                    draw_bar(ax, i + (j - (nb - 1) / 2) * w, num(r["mIoU"]), w * 0.9, it[0], it[1])
                    present.add(it[0])
                if o is not None:
                    ax.plot([i - 0.42, i + 0.42], [o, o], zorder=4, **ORIG_KW)
            ax.set_xticks(range(len(mods)), [MODELS[m]["label"] for m in mods])
            ax.set_ylabel("ADE20K mIoU (%)")
            panel_title(ax, letter, "ADE20K")
            cap_parts.append(f"({letter}) ADE20K linear segmentation, mIoU (%) on validation (dashed: original).")
            sources.append(("T_tier2_seg.csv", "model, variant, label, family, method, k, mIoU"))
    handles = [Patch(facecolor=tint(METHODS[k]["color"]), edgecolor=METHODS[k]["color"], hatch=METHODS[k]["hatch"],
                     linewidth=0.7, label={"mirror_mlp": "Mirror V-local, MLP only",
                                           "vlocal": "Mirror V-local, MLP + attention",
                                           "vexact": "Mirror V-exact, MLP + attention", "svd": "SVD, matched parameters",
                                           "random": "random structure", "prune": "width pruning"}[k])
               for k in METHOD_ORDER if k in present]
    if any_zs:
        handles.append(Patch(facecolor=tint(C_MUTED), edgecolor=C_MUTED, linewidth=0.7, label="light fill: zero-shot"))
    if any_healed:
        handles.append(Patch(facecolor=C_MUTED, edgecolor=C_MUTED, linewidth=0.7, label="solid fill: healed"))
    handles.append(orig_handle("original model"))
    fig.legend(handles=handles, loc="outside upper center", ncol=math.ceil(len(handles) / 2))
    files = save(fig, out, "fig7_tier2_probes")

    cap = ("Tier-2 evaluation with linear probes on frozen features. " + " ".join(cap_parts) +
           " Light bars: zero-shot conversion; solid bars: after healing. One probe per model and variant (seed 0, "
           "n = 1, no error bars); the probe's weight decay is chosen on a held-out part of each training set. "
           "Converted models use operators fitted on ImageNet train images.")
    ceil = sorted({DATASET_LABEL.get(r["dataset"], r["dataset"]) for r in (tr or [])
                   if str(r.get("dataset_at_ceiling", "")).lower() == "true"})
    if ceil:
        cap += (f" {fmt_list(ceil)} is at ceiling (several variants tie with the original exactly), so it does not "
                "separate the methods.")
    notes = []
    if ceil:
        notes.append(f"{fmt_list(ceil)}: flagged at ceiling by the collector (dataset_at_ceiling in "
                     "T_tier2_transfer.csv; see its notes).")
    if mae_only:
        notes.append(f"T_tier2 rows with only the original model ({fmt_list(mae_only)}) are not drawn.")
    if tr:
        edge = sorted({DATASET_LABEL.get(r['dataset'], r['dataset']) for r in tr
                       if str(r.get("wd_at_grid_edge", "")).lower() == "true"})
        if edge:
            notes.append(f"The chosen weight decay is at the edge of the grid for some rows of {fmt_list(edge)} "
                         "(column wd_at_grid_edge).")
    R.figures.append({"title": "Figure 7. Tier-2 linear probes: ImageNet and transfer", "files": files, "caption": cap,
                      "sources": sources, "notes": notes})


# ----------------------------------------------------------------------------------------------
# Figure 0 (teaser): drawn by scripts/f7_visuals.py, indexed here
# ----------------------------------------------------------------------------------------------
F7_STEMS = ("fig0_teaser", "figF7_twins_full", "figF7_layer0_filters")


def fig0_teaser_entry(T: Tables, R: Registry, out: Path):
    """F7 figures are drawn by scripts/f7_visuals.py (into this folder by default, or with --replot runs/f7); this
    script only lists them in FIGURES.md with their source tables."""
    name = "Figure 0 (teaser, F7)"
    rows = T.get("T_f7")
    files = [f"{s}.{x}" for s in F7_STEMS for x in ("pdf", "png") if (out / f"{s}.{x}").is_file()]
    if not files:
        return R.na(name, "figure files not in this folder; `python scripts/f7_visuals.py --replot runs/f7` draws them"
                    + ("" if rows else " (T_f7.csv not found either: runs/f7 not collected)"))
    notes = ["Drawn by scripts/f7_visuals.py from runs/f7/, not by this script; its draft captions (with the check to "
             "make on the figure before keeping them) are in runs/f7/F7_SUMMARY.md."]
    if rows is None:
        notes.append("T_f7.csv is not in the tables: the numbers behind the figure are not collected yet.")
    R.figures.append({"title": "Figure 0 (teaser, F7). Mirror twins and layer-0 filters", "files": files,
                      "caption": "See runs/f7/F7_SUMMARY.md (draft captions written by scripts/f7_visuals.py).",
                      "sources": [("T_f7.csv", "block, twin_r_median, rand_r_median, frac_twin_gt_rand, n_pairs"),
                                  ("T_f7_layer0.csv", "operator, parity, energy_share, sym_fraction")],
                      "notes": notes})


# ----------------------------------------------------------------------------------------------
# Figure 8: E3, healed k-NN at full conversion against the pre-registered bars
# ----------------------------------------------------------------------------------------------
E3_POINTS = (("v0", "vexact", "V-exact\nseed 0"), ("v1", "vexact", "V-exact\nseed 1"),
             ("svd_raw", "svd", "SVD\nraw targets"), ("svd_sym", "svd", "SVD\nV-exact targets"),
             ("random", "random", "random\nstructure"))


def fig8_e3(T: Tables, R: Registry, out: Path):
    name = "Figure 8 (E3, matched healed baselines at full conversion)"
    rows = T.get("T_e3")
    if rows is None:
        return R.na(name, "T_e3.csv not found")
    checks = {r["id"]: r for r in (T.get("T_e3_checks") or [])}
    by = {r.get("key"): r for r in rows}
    pts = []
    for key, style, tick in E3_POINTS:
        r = by.get(key)
        if r is None or num(r.get("healed_top1")) is None:
            R.na(f"{name}, arm {r.get('arm') if r else key}", "no healed k-NN in T_e3")
            continue
        if str(r.get("tiny", "")).lower() == "true":
            R.na(f"{name}, arm {r.get('arm')}", "TINY smoke-test row, not a result")
            continue
        pts.append({"key": key, "style": style, "tick": f"{tick}\n(arm {r.get('arm')})", "y": num(r["healed_top1"]),
                    "orig": num(r.get("original_top1"))})
    if not any(p["key"] == "v0" for p in pts) or len(pts) < 2:
        return R.na(name, "needs arm 1 (V0) and at least one other arm with a healed k-NN")
    ca, cb, cs, cg = (checks.get(i, {}) for i in ("E3.a", "E3.b", "E3.seed", "G5"))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6), layout="constrained")
    grid(ax)
    xs = {p["key"]: i for i, p in enumerate(pts)}
    v0 = next(p for p in pts if p["key"] == "v0")
    if v0["orig"] is not None:
        ax.axhline(v0["orig"], zorder=1.5, **ORIG_KW)
    # The pre-registered bars, drawn over the arms they apply to: R at most V0 - bar(E3.a); S at most V0 - bar(E3.b).
    bar_lines = []
    for cid, keys in (("E3.a", ("random",)), ("E3.b", ("svd_raw", "svd_sym"))):
        y = num(checks.get(cid, {}).get("max_baseline"))
        at = [xs[k] for k in keys if k in xs]
        if y is None or not at:
            continue
        ax.hlines(y, min(at) - 0.42, max(at) + 0.42, colors=C_INK, linewidth=1.0, zorder=2.5)
        ax.text(max(at) + 0.42, y, f" {cid} bar", ha="left", va="center", fontsize=FONT - 1.5, color=C_MUTED)
        bar_lines.append((cid, y))
    for i, p in enumerate(pts):
        m = METHODS[p["style"]]
        ax.plot(i, p["y"], marker=m["marker"], ms=MS[m["marker"]] + 1.0, mfc=m["color"], mec="white", mew=0.5,
                ls="none", zorder=4)
        ax.text(i, p["y"] + 0.25, f"{p['y']:.2f}", ha="center", va="bottom", fontsize=FONT - 1, color=C_INK)
    ys = [p["y"] for p in pts] + [y for _, y in bar_lines] + ([v0["orig"]] if v0["orig"] is not None else [])
    lo, hi = math.floor(min(ys) - 1.0), math.ceil(max(ys) + 1.0)
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(pts)), [p["tick"] for p in pts], fontsize=FONT - 1)
    ax.set_xlim(-0.6, len(pts) - 0.1)
    ax.set_ylabel("healed k-NN top-1 (%)")
    ax.tick_params(axis="x", length=0)
    handles = [orig_handle("original model"), Line2D([], [], color=C_INK, lw=1.0, label="pre-registered bar")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    files = save(fig, out, "fig8_e3")

    def vtxt(c, spec="+.2f"):
        v = num(c.get("value"))
        return (f"{v:{spec}}, " if v is not None else "") + str(c.get("verdict") or "not available")

    s = num(cs.get("value"))
    cap = ("E3: healed k-NN top-1 (%) at full conversion (all 12 blocks, MLP and attention) of V-exact (seeds 0 and 1) "
           "and of its matched healed baselines (SVD at matched parameters with raw and with V-exact final targets, and "
           "the random-structure control), one recipe for all. Each horizontal bar marks the largest baseline value the "
           "pre-registered bar allows (V0 minus the bar); each check also needs its difference to be larger than the seed "
           "difference s = |V0 - V1|" + (f" ({s:.2f})" if s is not None else "") + ". "
           f"E3.a (V0 minus random): {vtxt(ca)}; E3.b (V0 minus the better SVD arm): {vtxt(cb)}; gate G5: "
           f"{cg.get('verdict') or 'not available'}. The y-axis does not start at zero. One healing run per arm (two for "
           f"V-exact). Protocol: {PROTOCOL}; healing data from train images only.")
    notes = [f"Bars: V0 minus the E3.a and E3.b bars, column max_baseline of T_e3_checks ({', '.join(f'{c} {y:.2f}' for c, y in bar_lines)})."]
    if any("pending" in str(c.get("verdict", "")) for c in (ca, cb, cg)):
        notes.append("Some checks are pending (arms not run or a validity condition not yet met); see T_e3_checks.")
    R.figures.append({"title": "Figure 8. E3: matched healed baselines at full conversion", "files": files, "caption": cap,
                      "sources": [("T_e3.csv", "key, arm, healed_top1, original_top1, tiny"),
                                  ("T_e3_checks.csv", "id, value, max_baseline, verdict (E3.a, E3.b, E3.seed, G5)")],
                      "notes": notes})


# ----------------------------------------------------------------------------------------------
# FIGURES.md and meta
# ----------------------------------------------------------------------------------------------
def write_index(R: Registry, T: Tables, out: Path, gi: dict):
    L = ["# Paper figures", "",
         f"Generated by `scripts/make_figures.py` on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} at commit "
         f"{(gi.get('git_hash') or 'unknown')[:10]}{' (uncommitted changes)' if gi.get('git_dirty') else ''}, "
         f"from the CSV tables in `results/paper/tables/` only (written by `scripts/collect_results.py`). "
         f"Each figure is a vector PDF and a 300 dpi PNG, sized for a two-column paper (single column "
         f"{SINGLE} in, double column {DOUBLE} in). Captions are drafts; the numbers behind them are in the tables "
         f"named, and `results/paper/numbers.json` gives each table value's source file, seed and commit. "
         f"Rerun this script after the collector refreshes the tables.", "",
         "Encoding used throughout: colour follows the method, and a marker shape (lines, points) or hatch (bars) "
         "carries it in greyscale: Mirror V-local, MLP only: blue circle, no hatch; Mirror V-local (MLP + attention): "
         "aqua upward triangle, "
         "vertical hatch; Mirror V-exact: violet diamond, back-slash hatch; SVD at matched parameters: orange "
         "square, slash hatch; random structure: grey downward triangle, dotted hatch; MLP width pruning: dark grey "
         "plus. In Figure 4 colour and marker follow the model (as in the collector's figures). Zero-shot results "
         "are light (hollow markers or tinted bars), healed results solid. A dashed black line is the original "
         "model (in Figure 6, the dense model).", ""]
    for f in R.figures:
        L += [f"## {f['title']}", "",
              "Files: " + ", ".join(f"`{x}`" for x in f["files"]), "",
              f"Caption (draft): {f['caption']}", "", "Source tables and columns:"]
        L += [f"- `{t}`: {c}" for t, c in f["sources"]]
        if f.get("notes"):
            L += ["", "Notes:"] + [f"- {n}" for n in f["notes"]]
        L.append("")
    L += ["## Skipped or incomplete", ""]
    if R.skipped:
        L += [f"- {w}: not available ({why})." for w, why in R.skipped]
    else:
        L.append("- Nothing skipped.")
    L += ["", "## Inputs read", ""] + [f"- `{n}` (SHA-256 {h[:12]})" for n, h in sorted(T.used.items())]
    if T.missing:
        L += ["", "Looked for but not found: " + ", ".join(f"`{m}`" for m in T.missing) + "."]
    (out / "FIGURES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {rel(out / 'FIGURES.md')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default=str(ROOT / "results/paper/tables"))
    ap.add_argument("--out", default=str(ROOT / "results/paper/figures_paper"))
    a = ap.parse_args(argv)
    tables, out = Path(a.tables), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    T, R = Tables(tables), Registry()
    gi = git_info()
    for f in (fig0_teaser_entry, fig1_main, fig2_healing, fig3_ablations, fig4_tax, fig5_generality, fig6_cost, fig7_tier2,
              fig8_e3):
        f(T, R, out)
    write_index(R, T, out, gi)
    meta = {"script": "scripts/make_figures.py", "date": dt.datetime.now().isoformat(timespec="seconds"), **gi,
            "seed": None, "seed_note": "no randomness: the figures depend only on the input tables",
            "tables_dir": rel(tables),
            "inputs_sha256": dict(sorted(T.used.items())), "inputs_missing": T.missing,
            "figures": [{"title": f["title"], "files": f["files"]} for f in R.figures],
            "skipped": [{"what": w, "why": why} for w, why in R.skipped],
            "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                         "numpy": np.__version__}}
    (out / "figures_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(out / 'figures_meta.json')}; {len(R.figures)} figures, "
          f"{len(R.skipped)} items not available")


if __name__ == "__main__":
    main()

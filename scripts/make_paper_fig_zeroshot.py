#!/usr/bin/env python
"""Appendix figure: accuracy before healing against how much of the model is converted, DINOv2-B/14 with registers.

    .venv/Scripts/python.exe scripts/make_paper_fig_zeroshot.py

Reads only results/MUST-07/rows_dinov2_vitb14_reg.csv (written by scripts/must07_compressors.py and
scripts/must07_shards.py); every plotted value comes from it, nothing is typed in by hand. Columns used: model, size,
method, fit_seed, val_top1 (validation k-NN top-1 in percent: 100k-image train bank, 50k validation queries, k = 20,
see results/MUST-07/SUMMARY.md), status, git_hash. Only rows with status "ok" are used.

x: the converted span, as categories in the order 6mlp (the MLPs of blocks 1 to 6), 6both (blocks 1 to 6, MLP and
attention), 9both (blocks 1 to 9) and full (all 12 blocks); the span definition is in the file (columns k and
components) and is printed. y: the mean of val_top1 over the fit seeds; where a cell has more than one seed the
range (lowest to highest seed) is drawn as a faint bar behind the marker (most ranges are smaller than the marker).
The original model (method and size "original") is the dotted horizontal line.

Methods drawn (others in the file, such as mirror_local and pair_random, are left out and listed in the printout):
Mirror Surgery (mirror_exact, ROLE["mirror"], thick), SVD-LLM, ASVD-style, SliceGPT-style, FLAP-style, ZipIt-style
and plain SVD in solid colour lines, and the three controls (pairing by correlation, pairing by weight similarity,
random structure) in grey dashed lines. Colours: ROLE for mirror, svdllm, slicegpt, svd and random; C["lavender"],
C["sky"] and C["ink"] (Okabe-Ito black) for ASVD-style, FLAP-style and ZipIt-style. The six chromatic Okabe-Ito
colours pass the dataviz validator for all pairs except a colour-vision warning (reddish purple against bluish green,
delta E 7.6, legal only with a second channel) and a contrast warning for orange and sky blue (needs visible labels);
black and the greys read as grey to the validator. So identity never rests on colour alone: every series has its own
marker, the controls are dashed, and every series is named by a direct label at its right end.

Direct labels: each series is named in a column right of the plot, at the height of its last point; labels that
would overlap are spread apart by a fixed rule (overlapping labels are merged into a group centred on the mean of
their targets, with one line of text per label) and joined to their line ends by thin leader lines. The rule only
moves text; it never moves a data point.

Writes paper/figures/fig_zeroshot.pdf (vector) and fig_zeroshot.png (300 dpi), 6.0 x 2.6 inches (savefig.bbox kept
at "standard" so the page has exactly that size), and prints every plotted value with its source rows and the git
hash. No randomness.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, ROLE, apply  # noqa: E402

MODEL = "dinov2_vitb14_reg"
SRC = ROOT / "results" / "MUST-07" / f"rows_{MODEL}.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_zeroshot"

SIZES = {"6mlp": "6 MLPs", "6both": "6 blocks", "9both": "9 blocks", "full": "all 12"}  # order = x order
DASH_A, DASH_B, DASH_C = (0, (5.5, 1.8)), (0, (2.8, 1.4)), (0, (5.0, 1.5, 1.2, 1.5))
# method: (label, colour, line width, line style, marker, control); dict order = drawing order (last on top)
METHODS = {
    "random_structure": ("random structure", ROLE["random"], 0.9, DASH_C, "v", True),
    "pair_actcorr": ("pairing by correlation", ROLE["random"], 0.9, DASH_A, "<", True),
    "pair_wcos": ("pairing by weight similarity", ROLE["random"], 0.9, DASH_B, ">", True),
    "svd": ("plain SVD", ROLE["svd"], 1.0, "-", "^", False),
    "zipit": ("ZipIt-style", C["ink"], 0.9, "-", "h", False),
    "flap": ("FLAP-style", C["sky"], 1.0, "-", "X", False),
    "slicegpt": ("SliceGPT-style", ROLE["slicegpt"], 1.0, "-", "D", False),
    "asvd_scale": ("ASVD-style", C["lavender"], 1.0, "-", "P", False),
    "svdllm": ("SVD-LLM", ROLE["svdllm"], 1.1, "-", "s", False),
    "mirror_exact": ("Mirror Surgery", ROLE["mirror"], 2.0, "-", "o", False),
}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.6
AX_LEFT, AX_BOTTOM, AX_W, AX_H = 0.47, 0.40, 3.62, 2.08   # inches
XLIM = (-0.22, 3.12)
YLIM = (0, 84)
FS = 8
MS = 3.8                     # marker size (points)
MS_MIRROR = 4.6
LEADER_X = (3.17, 3.36, 3.42)   # data x of the leader: elbow start, elbow end, text start
LABEL_GAP_PT = 9.2              # minimum spacing of the right-hand labels (points)


def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def spread(targets: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label heights near `targets` with at least `gap` between neighbours, inside [lo, hi]. Overlapping labels are
    merged into groups; each group is centred on the mean of its targets (classic 1-D label placement)."""
    order = np.argsort(targets)[::-1]                     # top to bottom
    groups = [[i] for i in order]

    def place(g):
        centre = float(np.mean([targets[i] for i in g]))
        top = centre + gap * (len(g) - 1) / 2
        top = min(top, hi)
        top = max(top, lo + gap * (len(g) - 1))
        return [top - gap * k for k in range(len(g))]

    merged = True
    while merged:
        merged = False
        for a in range(len(groups) - 1):
            ya, yb = place(groups[a]), place(groups[a + 1])
            if ya[-1] - yb[0] < gap - 1e-9:
                groups[a] = groups[a] + groups[a + 1]
                del groups[a + 1]
                merged = True
                break
    out = [0.0] * len(targets)
    for g in groups:
        for i, y in zip(g, place(g)):
            out[i] = y
    return out


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SRC)
    df = df[(df["model"] == MODEL) & (df["status"] == "ok")].copy()
    orig = df[(df["method"] == "original") & (df["size"] == "original")]
    sel = df[df["method"].isin(METHODS) & df["size"].isin(SIZES)]
    agg = (sel.groupby(["method", "size"])["val_top1"]
           .agg(mean="mean", lo="min", hi="max", n="count").reset_index())
    agg["seeds"] = [sorted(sel[(sel["method"] == m) & (sel["size"] == s)]["fit_seed"].astype(int))
                    for m, s in zip(agg["method"], agg["size"])]
    return df, orig, agg


def main() -> int:
    df, orig, agg = load()
    if orig.empty or agg.empty:
        print("nothing to plot")
        return 1
    orig_top1 = float(orig["val_top1"].mean())
    xpos = {s: i for i, s in enumerate(SIZES)}

    apply()
    plt.rcParams["savefig.bbox"] = "standard"   # keep the page exactly FIG_W x FIG_H
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([AX_LEFT / FIG_W, AX_BOTTOM / FIG_H, AX_W / FIG_W, AX_H / FIG_H])

    ax.axhline(orig_top1, color=C["ink2"], lw=0.9, ls=(0, (1.0, 1.6)), zorder=2)
    ends = []   # (label, colour, line style, y of the last point)
    for m, (label, colour, lw, ls, marker, control) in METHODS.items():
        g = agg[agg["method"] == m].copy()
        if g.empty:
            continue
        g["x"] = g["size"].map(xpos)
        g = g.sort_values("x")
        multi = g[g["n"] > 1]
        ax.vlines(multi["x"], multi["lo"], multi["hi"], color=colour, lw=1.4, alpha=0.5, zorder=3)
        ms = MS_MIRROR if m == "mirror_exact" else MS
        ax.plot(g["x"], g["mean"], color=colour, lw=lw, ls=ls, marker=marker, ms=ms, mfc=colour,
                mec=C["paper"], mew=0.5, solid_joinstyle="round", dash_capstyle="butt",
                zorder=6 if m == "mirror_exact" else (4 if control else 5))
        ends.append((label, colour, ls, float(g["mean"].iloc[-1]), float(g["x"].iloc[-1])))
    ends.append(("original", C["ink2"], (0, (1.0, 1.6)), orig_top1, XLIM[1]))

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(len(SIZES)), list(SIZES.values()))
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis="both", labelsize=FS)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_ylabel("k-NN top-1 before healing (%)", fontsize=FS)
    ax.set_xlabel("converted part of the model, counted from block 1", fontsize=FS)

    # direct labels in a column right of the plot, spread apart where they would overlap
    unit_per_pt = (YLIM[1] - YLIM[0]) / (AX_H * 72)
    gap = LABEL_GAP_PT * unit_per_pt
    y_lab = spread([e[3] for e in ends], gap, YLIM[0] - 0.02 * (YLIM[1] - YLIM[0]), YLIM[1])
    x0, x1, xt = LEADER_X
    for (label, colour, ls, y_end, x_end), y in zip(ends, y_lab):
        start = x_end + 0.06 if label != "original" else x_end
        ax.plot([start, x0, x1], [y_end, y_end, y], color=colour, lw=0.6,
                ls="-" if label != "original" else ls, clip_on=False, zorder=2)
        ax.text(xt, y, label, ha="left", va="center", fontsize=FS, color=C["ink"], clip_on=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{OUT_STEM}.pdf", OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ log what was plotted
    src = SRC.relative_to(ROOT).as_posix()
    print(f"repo git HEAD: {git_head()}")
    print(f"source: {src} (status == ok, model == {MODEL})")
    print(f"  git hashes in the plotted rows: {sorted({str(h)[:10] for h in df['git_hash']})}")
    spans = df[df["size"].isin(SIZES)].groupby("size")[["k", "components"]].first()
    print("  spans: " + "; ".join(f"{s} = {SIZES[s]}: k={int(spans.loc[s, 'k'])}, {spans.loc[s, 'components']}"
                                 for s in SIZES if s in spans.index))
    print(f"\noriginal: val_top1 {orig_top1:.2f} (rows: {len(orig)})")
    print(f"\n  {'method':18s} {'label':30s} " + " ".join(f"{SIZES[s]:>22s}" for s in SIZES))
    for m, (label, *_rest) in METHODS.items():
        cells = []
        for s in SIZES:
            r = agg[(agg["method"] == m) & (agg["size"] == s)]
            if r.empty:
                cells.append(f"{'n/a':>22s}")
                continue
            r = r.iloc[0]
            txt = (f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] n={int(r['n'])}" if r["n"] > 1
                   else f"{r['mean']:.2f} n=1")
            cells.append(f"{txt:>22s}")
        print(f"  {m:18s} {label:30s} " + " ".join(cells))
    print("  (mean over fit seeds [lowest, highest seed]; n = number of fit seeds)")
    left_out = sorted(set(df["method"]) - set(METHODS) - {"original"})
    print(f"\nmethods in the file not drawn: {left_out}")
    print(f"\nwrote {pdf.relative_to(ROOT).as_posix()} and {png.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

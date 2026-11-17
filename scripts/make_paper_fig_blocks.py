#!/usr/bin/env python
"""Appendix figure: mirror structure block by block for the four MUST-04 models.

    .venv/Scripts/python.exe scripts/make_paper_fig_blocks.py

Reads only these files, for each model folder results/MUST-04/<model>/ (deit3_b16, dinov2_vitb14, dinov2_vitb14_reg,
dinov2_vitl14_reg); every plotted value comes from them, nothing is typed in by hand:
  blocks.csv   groups LR and UD: frac_2_t50 per block, the share of MLP hidden units in a 2-cycle (twin pair) of the
               matched permutation whose twin correlation is at least 0.5 (metrics.json "definitions",
               twin_fraction_t50) (panel a)
  layers.csv   groups LR and UD: d_minus per layer, the number of odd directions of the action on the residual stream,
               and d, the stream width (panel b)

Relative depth: panel (a) block b of n at b / n (block b writes stream layer b); panel (b) stream layer l of L at
l / L, where layer 0 is the input of block 1 and layer L the output of the last block. n and L are read from the
files (the largest block and layer index of each group).

Encoding: LR in ROLE["LR"] with solid lines, UD in ROLE["UD"] with dashed lines (dash = second channel, so the two
groups never rest on colour alone); one marker shape per model, with a key. The two Okabe-Ito colours pass the
dataviz palette validator (scripts/validate_palette.js of the dataviz skill: lightness, chroma, colour-vision and
contrast checks all pass for blue against vermilion on white).

Writes paper/figures/fig_blocks.pdf (vector) and fig_blocks.png (300 dpi), 6.0 x 2.3 inches (savefig.bbox kept at
"standard" so the page has exactly that size), and prints every plotted value with its source, fit seed and git
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
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, ROLE, apply  # noqa: E402

MUST04 = ROOT / "results" / "MUST-04"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_blocks"

MODELS = {  # folder: (label, marker); dict order = key order
    "deit3_b16": ("DeiT III-B/16", "o"),
    "dinov2_vitb14": ("DINOv2-B/14", "s"),
    "dinov2_vitb14_reg": ("DINOv2-B/14 reg", "^"),
    "dinov2_vitl14_reg": ("DINOv2-L/14 reg", "D"),
}
GROUPS = {  # group: (label, colour, line style)
    "LR": ("left-right (LR)", ROLE["LR"], "-"),
    "UD": ("up-down (UD)", ROLE["UD"], (0, (3.2, 1.6))),
}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.3
FS, FS_TITLE = 8, 9
LW = 0.9              # thin lines, one per model and group
MS = 2.7              # marker size (points)
MEW = 0.4             # white ring around markers


def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load() -> dict:
    data = {}
    for m in MODELS:
        b = pd.read_csv(MUST04 / m / "blocks.csv")
        l = pd.read_csv(MUST04 / m / "layers.csv")  # noqa: E741
        per = {}
        for g in GROUPS:
            bg = b[b["group"] == g].sort_values("block")
            lg = l[l["group"] == g].sort_values("layer")
            if bg.empty and lg.empty:
                continue
            n_blocks = int(bg["block"].max()) if len(bg) else None
            n_layers = int(lg["layer"].max()) if len(lg) else None
            per[g] = dict(
                blocks=bg[["block", "frac_2_t50"]].assign(x=bg["block"] / n_blocks) if len(bg) else None,
                layers=lg[["layer", "d", "d_minus"]].assign(x=lg["layer"] / n_layers) if len(lg) else None,
                n_blocks=n_blocks, n_layers=n_layers,
                seed=sorted(set(bg["fit_seed"]) | set(lg["fit_seed"])),
                git=sorted({str(h)[:10] for h in set(bg["git_hash"]) | set(lg["git_hash"])}))
        data[m] = per
    return data


def style_axis(ax) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.tick_params(labelsize=FS)
    ax.set_xlabel("relative depth", fontsize=FS)


def draw(ax, data: dict, part: str, ycol: str) -> None:
    for g, (_, colour, ls) in GROUPS.items():
        for m, (_, marker) in MODELS.items():
            if g not in data[m] or data[m][g][part] is None:
                continue
            s = data[m][g][part]
            ax.plot(s["x"], s[ycol], color=colour, ls=ls, lw=LW, marker=marker, ms=MS, mfc=colour,
                    mec=C["paper"], mew=MEW, solid_joinstyle="round", dash_joinstyle="round", zorder=3)


def main() -> int:
    data = load()
    apply()
    plt.rcParams["savefig.bbox"] = "standard"   # keep the page exactly FIG_W x FIG_H
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.06)

    # (a) twin pairs per block
    draw(ax_a, data, "blocks", "frac_2_t50")
    style_axis(ax_a)
    ax_a.set_ylim(0, 0.9)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_a.set_ylabel("share of hidden units in twins", fontsize=FS)
    ax_a.set_title("(a)  twin pairs (correlation at least 0.5)", loc="left", fontsize=FS_TITLE, pad=5)
    present = [g for g in GROUPS if any(g in data[m] for m in MODELS)]
    ax_a.legend(handles=[Line2D([], [], color=GROUPS[g][1], ls=GROUPS[g][2], lw=1.3, label=GROUPS[g][0])
                         for g in present],
                loc="upper right", fontsize=FS, handlelength=2.2, handletextpad=0.5, borderaxespad=0.3,
                labelspacing=0.35)

    # (b) odd directions of the stream per layer
    draw(ax_b, data, "layers", "d_minus")
    style_axis(ax_b)
    top = max(float(data[m][g]["layers"]["d_minus"].max()) for m in MODELS for g in data[m])
    ax_b.set_ylim(0, 50 * np.ceil(top / 50) + 10)
    ax_b.set_ylabel(r"odd directions $d_-$", fontsize=FS)
    ax_b.set_title(r"(b)  odd directions of the stream", loc="left", fontsize=FS_TITLE, pad=5)
    ax_b.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=MS + 0.6, mfc=C["ink2"], mec=C["paper"], mew=MEW,
                                label=lab) for lab, mk in MODELS.values()],
                loc="upper right", fontsize=FS, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{OUT_STEM}.pdf", OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ log what was plotted
    print(f"repo git HEAD: {git_head()}\n")
    for m, (lab, _) in MODELS.items():
        for g in GROUPS:
            if g not in data[m]:
                print(f"{m} {g}: not in the files, not drawn")
                continue
            d = data[m][g]
            print(f"{lab} ({m}), group {g}: fit_seed {d['seed']}, git {d['git']}, "
                  f"{d['n_blocks']} blocks, layers 0..{d['n_layers']}, stream width d = "
                  f"{sorted(set(int(v) for v in d['layers']['d']))}")
            bl = d["blocks"]
            print(f"  (a) results/MUST-04/{m}/blocks.csv frac_2_t50 by block:")
            print("      " + " ".join(f"{int(b)}:{v:.3f}" for b, v in zip(bl["block"], bl["frac_2_t50"])))
            ly = d["layers"]
            print(f"  (b) results/MUST-04/{m}/layers.csv d_minus by layer:")
            print("      " + " ".join(f"{int(i)}:{int(v)}" for i, v in zip(ly["layer"], ly["d_minus"])))

    # plain check of the shape the caption may describe (printed, not drawn)
    print("\nLR versus UD (UD minus LR), first and second half of the depth:")
    for m, (lab, _) in MODELS.items():
        if not {"LR", "UD"} <= set(data[m]):
            continue
        out = []
        for part, col in (("blocks", "frac_2_t50"), ("layers", "d_minus")):
            a = data[m]["LR"][part].set_index("x")[col]
            u = data[m]["UD"][part].set_index("x")[col]
            diff = (u - a).dropna()
            early, late = diff[diff.index <= 0.5], diff[diff.index > 0.5]
            fmt = "{:+.3f}" if part == "blocks" else "{:+.1f}"
            out.append(f"{col}: mean diff depth <= 0.5 " + fmt.format(early.mean()) + ", > 0.5 "
                       + fmt.format(late.mean()))
        print(f"  {lab:16s} " + "; ".join(out))
    print(f"\nwrote {pdf.relative_to(ROOT).as_posix()} and {png.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

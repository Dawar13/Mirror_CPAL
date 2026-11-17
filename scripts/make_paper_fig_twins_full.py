#!/usr/bin/env python
"""Appendix figure: the full mirror-twin grid (four twin pairs from each of blocks 1, 3 and 6).

    .venv/Scripts/python.exe scripts/make_paper_fig_twins_full.py

Reads only results/paper_next/data/fig1_teaser.npz (written by scripts/make_figures_next.py from the MUST-05 teaser
of DINOv2-B/14 with registers, results/MUST-05/dinov2_vitb14_reg/teaser.npz; provenance in
results/paper_next/data/fig1_meta.csv). For each of the blocks in `blocks` the file holds the 8 strongest twin pairs
(ranked by the matching weight w = 2 S_ij - S_ii - S_jj, `rank_by`), their neuron indices (`b<b>_pairs`), the
symmetrised fit-set mirror correlation S_ij of each pair (`b<b>_fit_S`, the twin correlation), and for each neuron the
16 top-activating patches over 50,000 ImageNet train images (`b<b>_crops`, 42 x 42 pixel crops around the token, with
`b<b>_valid` marking filled slots).

Each row is one pair: the first 6 patches of neuron i on the left, the first 6 of its twin j on the right, and the
twin correlation S_ij at the right edge. The first 4 pairs of each block are shown, in the file's order. Nothing is
typed in by hand; the script prints every plotted pair, its neuron indices and correlation, and the git hash.

Style: the shared paper style (scripts/paper_style.py, apply()), text in ink colours only; images are drawn with
nearest-neighbour interpolation and no frames. Axes are placed in inches, and savefig.bbox is set to "standard" so the
page is exactly 6.0 inches wide. Writes paper/figures/fig_twins_full.pdf (vector page, embedded images) and .png
(300 dpi preview). No randomness.
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
from paper_style import C, apply  # noqa: E402

SRC = ROOT / "results" / "paper_next" / "data" / "fig1_teaser.npz"
META = ROOT / "results" / "paper_next" / "data" / "fig1_meta.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_twins_full"

# ---------------------------------------------------------------------------------------------- layout only (no data)
N_PAIRS = 4            # pairs per block (the first N_PAIRS in the file's ranking)
N_PATCH = 6            # patches per neuron
FIG_W = 6.0            # inches
LEFT = 0.55            # room for the block labels
RIGHT = 0.42           # room for the correlation column
TOP = 0.20             # room for the column titles
BOTTOM = 0.02
PATCH_GAP = 0.025      # gap between patches in a row (inches)
SIDE_GAP = 0.22        # gap between neuron i and its twin j
ROW_GAP = 0.025        # gap between rows of one block
BLOCK_GAP = 0.13       # extra gap between blocks
FS = 8                 # all text
FS_TITLE = 8.5


def git_hash() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    z = np.load(SRC)
    blocks = [int(b) for b in z["blocks"]]
    rows = []
    for b in blocks:
        crops, valid = z[f"b{b}_crops"], z[f"b{b}_valid"]
        pairs, s = z[f"b{b}_pairs"], z[f"b{b}_fit_S"]
        for p in range(min(N_PAIRS, crops.shape[0])):
            rows.append(dict(block=b, rank=p + 1, crops=crops[p], valid=valid[p], i=int(pairs[p, 0]),
                             j=int(pairs[p, 1]), S=float(s[p])))
    k = min(N_PATCH, rows[0]["crops"].shape[1])

    # patch size from the width: LEFT + 2k patches + (2k - 2) patch gaps + side gap + RIGHT = FIG_W
    patch = (FIG_W - LEFT - RIGHT - SIDE_GAP - (2 * k - 2) * PATCH_GAP) / (2 * k)
    n_rows = len(rows)
    n_blocks = len(blocks)
    grid_h = n_rows * patch + (n_rows - n_blocks) * ROW_GAP + (n_blocks - 1) * BLOCK_GAP
    fig_h = TOP + grid_h + BOTTOM

    apply()
    plt.rcParams["savefig.bbox"] = "standard"   # keep the page exactly FIG_W x fig_h
    fig = plt.figure(figsize=(FIG_W, fig_h))

    def ax_at(x_in: float, y_top_in: float):
        """Axes of one patch, its top-left corner x_in inches from the left, y_top_in inches from the top."""
        return fig.add_axes([x_in / FIG_W, (fig_h - y_top_in - patch) / fig_h, patch / FIG_W, patch / fig_h])

    x_left = [LEFT + c * (patch + PATCH_GAP) for c in range(k)]
    x_twin0 = LEFT + k * patch + (k - 1) * PATCH_GAP + SIDE_GAP
    x_right = [x_twin0 + c * (patch + PATCH_GAP) for c in range(k)]
    x_end = x_right[-1] + patch

    y = TOP
    prev_block = None
    block_span = {}
    for r in rows:
        if prev_block is not None:
            y += BLOCK_GAP + ROW_GAP if r["block"] != prev_block else ROW_GAP
        prev_block = r["block"]
        for side, xs in ((0, x_left), (1, x_right)):
            for c, x in enumerate(xs[:k]):
                ax = ax_at(x, y)
                ax.set_axis_off()
                if r["valid"][side, c]:
                    ax.imshow(r["crops"][side, c], interpolation="nearest")
        # twin correlation at the right edge, centred on the row
        fig.text((x_end + 0.08) / FIG_W, (fig_h - y - patch / 2) / fig_h, f"{r['S']:.2f}", ha="left",
                 va="center", fontsize=FS, color=C["ink2"])
        lo, hi = block_span.get(r["block"], (y, y))
        block_span[r["block"]] = (min(lo, y), max(hi, y + patch))
        y += patch

    # block labels, centred on each block's rows
    for b, (lo, hi) in block_span.items():
        fig.text((LEFT - 0.08) / FIG_W, (fig_h - (lo + hi) / 2) / fig_h, f"block {b}", ha="right", va="center",
                 fontsize=FS, color=C["ink"])

    # column titles
    y_title = (fig_h - TOP + 0.05) / fig_h
    fig.text((LEFT + (k * patch + (k - 1) * PATCH_GAP) / 2) / FIG_W, y_title, "neuron $i$", ha="center",
             va="bottom", fontsize=FS_TITLE, color=C["ink"])
    fig.text((x_twin0 + (k * patch + (k - 1) * PATCH_GAP) / 2) / FIG_W, y_title, "its mirror twin $j$",
             ha="center", va="bottom", fontsize=FS_TITLE, color=C["ink"])
    fig.text((x_end + 0.08) / FIG_W, y_title, "corr.", ha="left", va="bottom", fontsize=FS, color=C["ink2"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{OUT_STEM}.pdf", OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf, dpi=300)   # dpi sets the resolution of the embedded crops (42 px each, so no loss)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ log what was plotted
    print(f"repo git HEAD: {git_hash()}")
    print(f"source: {SRC.relative_to(ROOT).as_posix()}")
    if META.exists():
        m = pd.read_csv(META).iloc[0]
        print(f"  built from {m['source_file']} at git {str(m['git_hash'])[:10]}")
    rank_by = str(z["rank_by"]) if "rank_by" in z.files else "?"
    print(f"  blocks {blocks}; pairs ranked by {rank_by}; crop {int(z['crop'])} px; "
          f"{len(z['vis_rows'])} train images searched for top patches")
    print(f"page {FIG_W:.2f} x {fig_h:.2f} in; patch {patch:.3f} in; {k} patches per neuron\n")
    print(f"  {'row':>3s} {'block':>5s} {'rank':>4s} {'neuron i':>8s} {'twin j':>7s} {'S_ij':>6s}  "
          f"valid patches shown (i, j)   key")
    for n, r in enumerate(rows, 1):
        vi, vj = int(r["valid"][0, :k].sum()), int(r["valid"][1, :k].sum())
        print(f"  {n:3d} {r['block']:5d} {r['rank']:4d} {r['i']:8d} {r['j']:7d} {r['S']:6.3f}  "
              f"{vi}, {vj}{'':20s} b{r['block']}_fit_S[{r['rank'] - 1}], b{r['block']}_pairs[{r['rank'] - 1}]")
    print(f"\nwrote {pdf.relative_to(ROOT).as_posix()} and {png.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

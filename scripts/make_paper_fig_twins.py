"""Compact mirror-twin figure for the paper (Figure 1).

Reads results/paper_next/data/fig1_teaser.npz (written by scripts/make_figures_next.py from the MUST-05 teaser of
DINOv2-B/14 with registers: top-activating patches over 50k ImageNet train images for the strongest twin pairs of
blocks 1, 3 and 6) and draws a few rows of "neuron i | its twin j" without a title, sized for the CPAL text width.
Writes paper/figures/fig_twins.pdf and .png.

    .venv/Scripts/python.exe scripts/make_paper_fig_twins.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "paper_next" / "data" / "fig1_teaser.npz"
OUT = ROOT / "paper" / "figures"
ROWS = [(1, 0), (1, 1), (3, 0), (6, 0)]        # (block, pair index): strongest pairs, as ranked in the teaser file
N_PATCH = 6

plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
                     "font.size": 8, "pdf.fonttype": 42})


def main() -> None:
    z = np.load(SRC)
    fig, axes = plt.subplots(len(ROWS), 2 * N_PATCH + 1, figsize=(6.0, 0.47 * len(ROWS) + 0.25),
                             gridspec_kw={"width_ratios": [1] * N_PATCH + [0.35] + [1] * N_PATCH,
                                          "wspace": 0.05, "hspace": 0.1})
    for r, (b, pi) in enumerate(ROWS):
        crops, valid = z[f"b{b}_crops"][pi], z[f"b{b}_valid"][pi]
        for side in range(2):
            for j in range(N_PATCH):
                ax = axes[r, j if side == 0 else N_PATCH + 1 + j]
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if valid[side, j]:
                    ax.imshow(crops[side, j], interpolation="nearest")
        axes[r, N_PATCH].axis("off")
        axes[r, 0].text(-0.12, 0.5, f"block {b}", transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=8, color="#444444")
        print(f"row {r}: block {b}, pair {tuple(int(x) for x in z[f'b{b}_pairs'][pi])}, "
              f"twin correlation {float(z[f'b{b}_fit_S'][pi]):.3f}")
    axes[0, N_PATCH // 2].set_title("neuron $i$: strongest patches", fontsize=8, loc="center", x=0.0)
    axes[0, N_PATCH + 1 + N_PATCH // 2].set_title("its mirror twin $j$: strongest patches", fontsize=8, loc="center",
                                                  x=0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_twins.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / "fig_twins.pdf")


if __name__ == "__main__":
    main()

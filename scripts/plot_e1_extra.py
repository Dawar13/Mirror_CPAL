#!/usr/bin/env python
"""Extra E1 figures from saved results (no GPU, no model):

    plots/spectrum_by_layer.png   odd-energy share of the mirror cross-moment by layer, and the
                                  sorted eigenvalue curves for three layers
    plots/twin_scores.png         match-score histograms (pairs vs self-twins) for three blocks
    plots/surgery_knn.png         k-NN top-1 and flip consistency vs k for each variant

    python scripts/plot_e1_extra.py [--out-root .]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirror.report import C_AQUA, C_BLUE, C_INK, C_NULL, C_ORANGE, _style  # noqa: E402
from mirror.utils import REPO_ROOT  # noqa: E402

BG = "#fcfcfb"


def _fig(ncols, width=10.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 3.8), facecolor=BG)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(BG)
    return fig, axes


def _pick(n, k=3):
    """Three representative indices: first, middle, last (1-based layers/blocks)."""
    return sorted({1, max(1, n // 2), n})[:k]


def spectrum(ops, layers_df, path):
    eig = ops["eigvals"]  # [L+1, d], eigenvalues of sym(C_l)
    odd = np.array([np.abs(e[e < 0]).sum() / max(np.abs(e).sum(), 1e-300) for e in eig])
    fig, axes = _fig(2)
    ax = axes[0]
    x = np.arange(len(odd))
    ax.plot(x, odd, "-", color=C_BLUE, marker="o", markersize=5, linewidth=2,
            label="odd energy share  sum|lam-| / sum|lam|")
    ax.plot(x, layers_df["dminus_frac"], "-", color=C_ORANGE, marker="s", markersize=5, linewidth=2,
            label="d- / d (count of odd directions)")
    ax.set_xlabel("layer")
    ax.set_ylim(-0.02, max(0.6, float(np.nanmax(odd)) * 1.1))
    ax.set_title("How much of the mirror cross-moment is odd", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    for l, color in zip(_pick(len(eig) - 1), (C_BLUE, C_ORANGE, C_AQUA)):
        e = np.sort(eig[l])[::-1]
        ax.plot(np.arange(len(e)), e / max(np.abs(e).max(), 1e-300), "-", color=color, linewidth=2,
                label=f"layer {l}")
    ax.axhline(0, color=C_NULL, linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1e-4)  # one huge even eigenvalue would otherwise flatten the rest
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue / max |eigenvalue| (symlog)")
    ax.set_title("Spectrum of sym(X^T X~): + even, - odd", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def twin_scores(ops, path):
    perms, scores, selfc = ops["perms"], ops["match_scores"], ops["self_corr"]
    blocks = _pick(perms.shape[0])
    fig, axes = _fig(len(blocks), width=4.0 * len(blocks))
    bins = np.linspace(-1, 1, 51)
    for ax, b in zip(axes, blocks):
        p, s, sc = perms[b - 1], scores[b - 1], selfc[b - 1]
        paired = p != np.arange(len(p))
        ax.hist(s[~paired], bins=bins, color=C_ORANGE, alpha=0.85, label=f"self-twins ({(~paired).sum()})")
        if paired.any():
            ax.hist(s[paired], bins=bins, color=C_BLUE, alpha=0.85, label=f"paired ({paired.sum()})")
            ax.hist(sc[paired], bins=bins, histtype="step", color=C_NULL, linewidth=1.5,
                    label="paired: own self-corr")
        ax.set_xlabel("corr(mirror neuron, matched neuron)")
        ax.set_title(f"Block {b}", fontsize=10, color=C_INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("neurons")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def surgery(df, path):
    orig = df[df["variant"] == "original"].iloc[0]
    fig, axes = _fig(2)
    for ax, col, title in ((axes[0], "top1", "k-NN top-1 (%)"), (axes[1], "flip_consistency", "flip consistency (%)")):
        for name, color, marker in (("surgery", C_BLUE, "o"), ("svd", C_ORANGE, "s"), ("random", C_AQUA, "^")):
            d = df[df["variant"] == name].sort_values("k")
            ax.plot(d["k"], d[col], "-", color=color, marker=marker, markersize=6, linewidth=2,
                    label={"surgery": "mirror surgery", "svd": "rank-matched SVD",
                           "random": "random-involution surgery"}[name])
        ax.axhline(orig[col], color=C_NULL, linestyle="--", linewidth=1.5, label="original model")
        ax.set_xlabel("blocks operated on (first k)")
        ax.set_title(title, fontsize=10, color=C_INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(REPO_ROOT))
    a = p.parse_args(argv)
    root = Path(a.out_root)
    res, plots = root / "results", root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    with np.load(res / "e1_operators.npz") as z:
        ops = {k: z[k] for k in z.files}
    layers = pd.read_csv(res / "e1_layers.csv")
    spectrum(ops, layers, plots / "spectrum_by_layer.png")
    twin_scores(ops, plots / "twin_scores.png")
    surgery(pd.read_csv(res / "e1_surgery.csv"), plots / "surgery_knn.png")
    print("wrote", ", ".join(str(plots / n) for n in ("spectrum_by_layer.png", "twin_scores.png", "surgery_knn.png")))


if __name__ == "__main__":
    main()

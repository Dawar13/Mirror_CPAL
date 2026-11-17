"""Shared figure style for the paper: muted colours, the paper's serif font, thin lines.

Import in every paper figure script:  from paper_style import apply, C, ROLE
"""
from __future__ import annotations

import matplotlib

# Muted, print-friendly palette (soft, not saturated). Names are by hue; ROLE maps meanings to colours.
C = {
    "ink": "#3B3B3B",        # text and axes
    "ink2": "#6B6B6B",       # secondary text
    "grid": "#E4E1DC",       # grid lines, light rules
    "paper": "#FFFFFF",
    "blue": "#5F7FA8",       # slate blue
    "terracotta": "#C07A5A",
    "sage": "#4F9A8C",       # muted teal (Okabe-Ito bluish green, softened); key kept for compatibility
    "teal": "#4F9A8C",
    "lavender": "#9C88B5",
    "sand": "#C9A961",
    "grey": "#9A9A9A",
    "blue_lt": "#C9D5E5",
    "terracotta_lt": "#EBCFC2",
    "sage_lt": "#CDE3DE",
    "teal_lt": "#CDE3DE",
    "lavender_lt": "#DDD4E8",
    "sand_lt": "#EDE1C2",
    "grey_lt": "#E2E2E2",
}

ROLE = {
    "mirror": C["blue"],          # Mirror Surgery
    "svdllm": C["terracotta"],    # SVD-LLM (best baseline)
    "slicegpt": C["sand"],
    "svd": C["sage"],
    "random": C["grey"],
    "original": C["ink"],
    "even": C["blue"],
    "odd": C["terracotta"],
    "LR": C["blue"], "UD": C["terracotta"], "R90": C["sage"], "R180": C["sand"],
    "vision": C["blue"], "imagetext": C["terracotta"], "cnn": C["sage"],
}

FONT = ["TeX Gyre Pagella", "Palatino Linotype", "Palatino", "DejaVu Serif"]


def apply() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": FONT, "mathtext.fontset": "custom",
        "mathtext.rm": FONT[1], "mathtext.it": FONT[1] + ":italic", "mathtext.bf": FONT[1] + ":bold",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

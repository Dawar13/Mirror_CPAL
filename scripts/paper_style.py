"""Shared figure style for the paper: bright colour-blind-safe colours (Okabe-Ito), the paper's serif font, thin lines.

Import in every paper figure script:  from paper_style import apply, C, ROLE
Key names are kept from the earlier muted palette so older scripts keep working:
blue = Okabe-Ito blue, terracotta = vermilion, sage/teal = bluish green, sand = orange, lavender = reddish purple.
"""
from __future__ import annotations

import matplotlib


def _tint(hex_colour: str, w: float = 0.55) -> str:
    """Mix a colour with white (w = share of white)."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * w)  # noqa: E731
    return "#{:02X}{:02X}{:02X}".format(mix(r), mix(g), mix(b))


_BASE = {
    "blue": "#0072B2",        # Okabe-Ito blue
    "terracotta": "#D55E00",  # Okabe-Ito vermilion
    "sage": "#009E73",        # Okabe-Ito bluish green
    "teal": "#009E73",
    "sand": "#E69F00",        # Okabe-Ito orange
    "lavender": "#CC79A7",    # Okabe-Ito reddish purple
    "sky": "#56B4E9",         # Okabe-Ito sky blue
    "grey": "#8C8C8C",
}

C = {"ink": "#2B2B2B", "ink2": "#5E5E5E", "grid": "#E6E6E6", "paper": "#FFFFFF", **_BASE}
C.update({f"{k}_lt": _tint(v) for k, v in _BASE.items()})

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
    "vision": C["blue"], "imagetext": C["lavender"], "cnn": C["sage"],
}

FONT = ["TeX Gyre Pagella", "Palatino Linotype", "Palatino", "DejaVu Serif"]


def apply() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": FONT, "mathtext.fontset": "custom",
        "mathtext.rm": FONT[1], "mathtext.it": FONT[1] + ":italic", "mathtext.bf": FONT[1] + ":bold",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

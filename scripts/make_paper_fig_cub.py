#!/usr/bin/env python
"""Paper figure: linear probes on CUB-200-2011 from the class-token feature split into its even part, its odd part
and the full feature, for the original DINOv2-B/14 with registers and for healed Mirror Surgery (CORE-06, table T6).

    .venv/Scripts/python.exe scripts/make_paper_fig_cub.py

Reads only results/CORE-06/probes.csv (every plotted value comes from it; nothing is typed in by hand). Rows used:
  variant   original, vexact_s0, vexact_s1, vexact_s2 (Mirror Surgery, exact, seeds 0 to 2)
  task      species (200 classes) for panel (a), facing (left or right) for panel (b)
  features  full_cls, even_cls, odd_cls (the class-token feature and its even and odd parts)
  test_set  natural; plus mirror_augmented (each test image and its mirror image) for even_cls in panel (b)
  acc       test accuracy in per cent

Panel (a) "Species": two bars per feature group, original and Mirror Surgery, natural test images.
Panel (b) "Facing direction": the same, plus a hatched third bar in the even group for Mirror Surgery on the
mirror-augmented test set, and a dashed line at chance (50%). The original's mirror-augmented results are printed
(not drawn) so the caption can quote them.

Mirror Surgery bars show the mean over the three seeds. The seed spread is printed for every bar; it is at most
about half a percentage point, well under one printed point at this size, so no error bars are drawn.

Encoding: colour = feature part (paper_style ROLE: even blue, odd vermilion; full grey), fill = model (original a
light tint with an outline in the feature colour, Mirror Surgery solid). The x axis names each group as well, so
identity never rests on colour alone, and every bar carries its value.

Writes paper/figures/fig_cub.pdf (vector) and fig_cub.png (300 dpi), 6.0 x 1.85 inches, and prints every plotted
value with its source line in probes.csv, plus the git hash of this checkout. No randomness.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, ROLE, apply  # noqa: E402

PROBES = ROOT / "results" / "CORE-06" / "probes.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_cub"

ORIGINAL = "original"
MIRROR_SEEDS = ("vexact_s0", "vexact_s1", "vexact_s2")
# feature key in probes.csv: (tick label, solid colour, light tint); order = left to right
FEATURES = {
    "full_cls": ("full", C["grey"], C["grey_lt"]),
    "even_cls": ("even", ROLE["even"], C["blue_lt"]),
    "odd_cls": ("odd", ROLE["odd"], C["terracotta_lt"]),
}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 1.85
AX_BOTTOM, AX_TOP = 0.20, 1.63                       # inches from the bottom of the figure
A_XLIM, B_XLIM = (-0.48, 3.08), (-0.48, 2.86)       # room on the right: legend in (a), "chance" label in (b)
UNIT = 0.742                                         # inches per group unit, the same in both panels
A_AX = (0.42, UNIT * (A_XLIM[1] - A_XLIM[0]))        # (left, width) in inches
B_AX = (A_AX[0] + A_AX[1] + 0.42, UNIT * (B_XLIM[1] - B_XLIM[0]))
YLIM, YTICKS = (0, 109), (0, 25, 50, 75, 100)        # axis drawn 0 to 100; room above for value labels
BAR_W, BAR_GAP = 0.30, 0.035                         # in group units; the gap is the white space between bars
FS, FS_TITLE, FS_VALUE = 8, 9, 8
EDGE_LW = 0.7                                        # outline of the light (original) bars
HATCH = "//////"


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def load() -> pd.DataFrame:
    df = pd.read_csv(PROBES)
    df["line"] = df.index + 2                        # line number in the file (line 1 is the header)
    return df


def row(df: pd.DataFrame, variant: str, task: str, features: str, test_set: str) -> pd.Series:
    sub = df[(df["variant"] == variant) & (df["task"] == task) & (df["features"] == features) &
             (df["test_set"] == test_set)]
    assert len(sub) == 1, (variant, task, features, test_set, len(sub))
    return sub.iloc[0]


def describe(r: pd.Series) -> str:
    return (f"{rel(PROBES)} line {int(r['line'])} ({r['variant']},{r['task']},{r['features']},{r['test_set']}) "
            f"acc {r['acc']:.4f}, n {int(r['n'])}")


def original_value(df, task, features, test_set) -> float:
    r = row(df, ORIGINAL, task, features, test_set)
    print(f"    original        {r['acc']:6.2f}  <- {describe(r)}")
    return float(r["acc"])


def mirror_value(df, task, features, test_set) -> float:
    rows = [row(df, v, task, features, test_set) for v in MIRROR_SEEDS]
    accs = [float(r["acc"]) for r in rows]
    mean = sum(accs) / len(accs)
    for r in rows:
        print(f"      seed          {r['acc']:6.2f}  <- {describe(r)}")
    print(f"    Mirror Surgery  {mean:6.2f}  (mean of {len(accs)} seeds; range {min(accs):.2f} to {max(accs):.2f}, "
          f"spread {max(accs) - min(accs):.2f} points)")
    return mean


def bar(ax, x, h, solid, tint, original: bool, hatch: bool = False) -> None:
    if original:
        ax.bar(x, h, BAR_W, color=tint, edgecolor=solid, linewidth=EDGE_LW, zorder=2)
    else:
        ax.bar(x, h, BAR_W, color=solid, edgecolor=C["paper"] if hatch else solid, linewidth=0,
               hatch=HATCH if hatch else None, zorder=2)
    ax.annotate(f"{h:.1f}", (x, h), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_VALUE, color=C["ink"], zorder=4)


def style_axes(ax, title: str, xlim: tuple, ylabel: bool) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.spines["left"].set_bounds(0, 100)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([v[0] for v in FEATURES.values()])
    ax.tick_params(axis="x", length=0, pad=3)
    if ylabel:
        ax.set_ylabel("test accuracy (%)", labelpad=2)
    ax.set_title(title, loc="left", fontsize=FS_TITLE, pad=3)


def panel_a(fig, df) -> None:
    ax = fig.add_axes([A_AX[0] / FIG_W, AX_BOTTOM / FIG_H, A_AX[1] / FIG_W, (AX_TOP - AX_BOTTOM) / FIG_H])
    print("\nPanel (a): species (200 classes), natural test images, accuracy (%)")
    off = (BAR_W + BAR_GAP) / 2
    for g, (feat, (label, solid, tint)) in enumerate(FEATURES.items()):
        print(f"  {label} ({feat}):")
        bar(ax, g - off, original_value(df, "species", feat, "natural"), solid, tint, original=True)
        bar(ax, g + off, mirror_value(df, "species", feat, "natural"), solid, tint, original=False)
    style_axes(ax, "(a) Species (200 classes)", A_XLIM, ylabel=True)

    # legend: one swatch per feature colour, light = original, solid = Mirror Surgery, hatched = image + its mirror
    light = tuple(Patch(facecolor=t, edgecolor=s, linewidth=EDGE_LW) for _, s, t in FEATURES.values())
    solid = tuple(Patch(facecolor=s, edgecolor=s, linewidth=0) for _, s, t in FEATURES.values())
    hatched = Patch(facecolor=ROLE["even"], edgecolor=C["paper"], linewidth=0, hatch=HATCH)
    ax.legend([light, solid, hatched], ["original", "Mirror Surgery", "Mirror Surgery,\nimage + its mirror"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.12)}, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), fontsize=FS, handlelength=2.1, handleheight=0.9, handletextpad=0.45,
              labelspacing=0.45, borderaxespad=0.1, borderpad=0.2, frameon=True,
              facecolor=C["paper"], edgecolor="none", framealpha=1.0, fancybox=False)


def panel_b(fig, df) -> None:
    ax = fig.add_axes([B_AX[0] / FIG_W, AX_BOTTOM / FIG_H, B_AX[1] / FIG_W, (AX_TOP - AX_BOTTOM) / FIG_H])
    print("\nPanel (b): facing direction (left or right), accuracy (%)")
    step = BAR_W + BAR_GAP
    for g, (feat, (label, solid, tint)) in enumerate(FEATURES.items()):
        print(f"  {label} ({feat}), natural test images:")
        if feat == "even_cls":                       # three bars: original, Mirror Surgery, Mirror Surgery mirrored
            bar(ax, g - step, original_value(df, "facing", feat, "natural"), solid, tint, original=True)
            bar(ax, g, mirror_value(df, "facing", feat, "natural"), solid, tint, original=False)
            print(f"  {label} ({feat}), mirror-augmented test set (each image and its mirror image):")
            bar(ax, g + step, mirror_value(df, "facing", feat, "mirror_augmented"), solid, tint, original=False,
                hatch=True)
            print("    (not drawn) original on the mirror-augmented set, for the caption:")
            original_value(df, "facing", feat, "mirror_augmented")
        else:
            bar(ax, g - step / 2, original_value(df, "facing", feat, "natural"), solid, tint, original=True)
            bar(ax, g + step / 2, mirror_value(df, "facing", feat, "natural"), solid, tint, original=False)
    ax.axhline(50, color=C["ink"], linewidth=0.8, linestyle=(0, (3.5, 2)), zorder=3)
    ax.annotate("chance", (B_XLIM[1], 50), xytext=(0, 1.5), textcoords="offset points", ha="right", va="bottom",
                fontsize=FS, color=C["ink"])
    style_axes(ax, "(b) Facing direction (left or right)", B_XLIM, ylabel=False)


def main() -> None:
    apply()
    plt.rcParams.update({"xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS,
                         "savefig.bbox": None, "hatch.linewidth": 0.6})
    print("git hash:", git_hash())
    print("source:", rel(PROBES))
    df = load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, df)
    panel_b(fig, df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches=None)
    print("\nwrote", rel(OUT_DIR / f"{OUT_STEM}.pdf"), "and", rel(OUT_DIR / f"{OUT_STEM}.png"))


if __name__ == "__main__":
    main()

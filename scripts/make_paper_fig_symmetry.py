#!/usr/bin/env python
"""Paper figure: pairing by symmetry in the early blocks (a) and how much mirror structure is left with depth (b).

    .venv/Scripts/python.exe scripts/make_paper_fig_symmetry.py

Reads only these files (every plotted value comes from them; nothing is typed in by hand). For each model the folder
is results/MUST-04/<model>/ if it exists there, else results/CORE-04/structure/<model>/:
  metrics.json  groups.<G>.ranges.first_half.ratio_to_null (panel a); the first and second half block lists and,
                for CNN and engine-built models, the `units` list that maps blocks to measured spaces (panel b)
  layers.csv    group == "LR": d and d_minus per measured space (panel b)
  blocks.csv    group == "LR": frac_2_t50 per block (printed only: mean twin fraction per half)

Panel (a): the MUST-04 models with all four groups (LR, UD, R90, R180). ratio_to_null is the mean over the first
half of the blocks of the pairing score (s_fc1 + s_fc2) / 2, divided by the mean of its null over the same blocks
(metrics.json "definitions"); lower means stronger pairing, 1.0 means no structure beyond the null. One marker per
model, a bar at the mean of the models.

Panel (b): every model; the share of odd directions d_minus / d of the left-right action in the residual stream,
against relative depth (0 = input of the first block, 1 = output of the last block). Where the stream spaces chain
(block b reads the space block b-1 wrote: all ViTs and ResNet-50), the stream is layer 0..L and the depth is
layer / L. ConvNeXt-B also measures the depthwise-conv output inside each block (spaces dw0..dw35, the MLP input,
not the stream); those are left out, and its stream points are the block outputs out0..out35 at depth b / 36 (the
stem output, depth 0, is not measured).

Writes paper/figures/fig_symmetry_depth.pdf (vector) and fig_symmetry_depth.png (300 dpi preview), and prints every
plotted value with its source, fit seed and git hash. No randomness: the small horizontal offsets in panel (a) are
fixed by model order.

Style: the shared paper style (scripts/paper_style.py: apply() for fonts, spines, grid and line widths; the muted
colours C and ROLE). Panel (a): ROLE["LR"], ROLE["UD"], ROLE["R90"], ROLE["R180"] (slate blue, terracotta, sage,
sand) with the marker shapes of scripts/make_figures_next.py (circle, square, triangle, diamond); each model is a
light-tint marker ringed in the group colour, the mean bar is the full colour, and each group is named on the axis.
Panel (b): ROLE["vision"], ROLE["imagetext"], ROLE["cnn"]; the eleven unnamed lines at reduced opacity, the three
named lines in full colour. The dataviz colour checker flags the muted colours (chroma below its floor; sage and
terracotta close under deuteranopia and for normal vision), so identity never rests on colour alone: shapes and
axis names in (a), and in (b) a legend, direct labels and a dashed line style for the CNNs. Fixed size 6 x 2.3 in
(savefig.bbox kept at "standard" so the page size does not shrink).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import C, ROLE, apply  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MUST04 = ROOT / "results" / "MUST-04"
CORE04 = ROOT / "results" / "CORE-04" / "structure"
OUT_DIR = ROOT / "paper" / "figures"
OUT_STEM = "fig_symmetry_depth"

# ---------------------------------------------------------------------------------------------- style
C_INK, C_INK2, SURFACE = C["ink"], C["ink2"], C["paper"]
_HUE = {v: k for k, v in C.items() if f"{k}_lt" in C}   # full colour -> hue name, for its light tint


def tint(colour: str) -> str:
    return C[_HUE[colour] + "_lt"]


GROUPS = {  # key: (label, colour, marker); dict order = slot order in panel (a)
    "LR": ("LR", ROLE["LR"], "o"),
    "UD": ("UD", ROLE["UD"], "s"),
    "R90": ("R90", ROLE["R90"], "^"),
    "R180": ("R180", ROLE["R180"], "D"),
}
HALF = "first_half"
MODEL_TYPES = {  # key: (legend label, colour, members); dict order = legend order
    "vision": ("vision-only ViTs", ROLE["vision"],
               ("dinov2_vits14_reg", "dinov2_vitb14", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
                "deit3_b16", "dino_b16", "dinov3_b16", "eva02_b14", "mae_b16")),
    "image_text": ("image-text ViTs", ROLE["imagetext"], ("clip_b16", "siglip2_b16")),
    "cnn": ("CNNs", ROLE["cnn"], ("convnext_b", "resnet50")),
}
LINESTYLE = {"vision": "-", "image_text": "-", "cnn": (0, (3.5, 1.5))}  # dashes: second channel for CNN vs image-text
LABELLED = ("dinov2_vitb14_reg", "clip_b16", "mae_b16")  # lines named directly at their end in panel (b)
# Layout only (no data): lines that end in the bunch near zero get their name in the free space above the bunch,
# with a vertical leader down to the line at x (just before the end); value = (x of the leader, y of the label).
LEADER_LABELS = {"dinov2_vitb14_reg": (0.955, 0.165), "clip_b16": (0.885, 0.105)}
SHORT_NAMES = {
    "dinov2_vitb14_reg": "DINOv2-B/14 reg", "deit3_b16": "DeiT III-B/16", "clip_b16": "CLIP-B/16",
    "siglip2_b16": "SigLIP 2-B/16", "mae_b16": "MAE-B/16", "dino_b16": "DINO-B/16", "dinov3_b16": "DINOv3-B/16",
    "eva02_b14": "EVA-02-B/14", "convnext_b": "ConvNeXt-B", "resnet50": "ResNet-50",
    "dinov2_vits14_reg": "DINOv2-S/14 reg", "dinov2_vitl14_reg": "DINOv2-L/14 reg",
    "dinov2_vitg14_reg": "DINOv2-g/14 reg", "dinov2_vitb14": "DINOv2-B/14",
}
FIG_W, FIG_H = 6.0, 2.3


# ---------------------------------------------------------------------------------------------- data
def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # the files contain NaN, which json.load accepts


def ratio(metrics: dict, group: str, half: str) -> float | None:
    try:
        v = float(metrics["groups"][group]["ranges"][half]["ratio_to_null"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def blocks_of(metrics: dict, group: str, half: str) -> list[int]:
    try:
        return list(metrics["groups"][group]["ranges"][half]["blocks"])
    except (KeyError, TypeError):
        return []


def provenance(metrics: dict) -> str:
    git = metrics.get("git") or {}
    h = git.get("git_hash", "?") if isinstance(git, dict) else str(git)
    dirty = " dirty" if isinstance(git, dict) and git.get("git_dirty") else ""
    return f"fit_seed={metrics.get('fit_seed')} n_fit_images={metrics.get('n_fit_images')} git={str(h)[:10]}{dirty}"


def model_dirs(base: Path) -> list[str]:
    return sorted(p.parent.name for p in base.glob("*/metrics.json"))


def stream_of(model_dir: Path, metrics: dict) -> tuple[pd.DataFrame, str]:
    """LR rows of layers.csv on the residual stream, with relative depth x and odd share."""
    df = pd.read_csv(model_dir / "layers.csv")
    lr = df[df["group"] == "LR"].sort_values("layer").copy()
    units = metrics.get("units") or []
    if "space" in lr.columns and units:
        chained = all(units[i]["in_space"] == units[i - 1]["out_space"] for i in range(1, len(units)))
        n = len(units)
        if chained:  # stream = input of block 1, then every block output
            pos = {units[0]["in_space"]: 0.0} | {u["out_space"]: u["block"] / n for u in units}
            how = f"spaces chain; layer 0..{n}, depth = layer / {n}"
        else:  # e.g. ConvNeXt: in_space is inside the block, so only the block outputs are on the stream
            pos = {u["out_space"]: u["block"] / n for u in units}
            dropped = sorted({u["in_space"] for u in units} - set(pos), key=lambda s: (len(s), s))
            how = (f"in-block spaces {dropped[0]}..{dropped[-1]} left out; block outputs at depth b / {n}"
                   if dropped else f"block outputs at depth b / {n}")
        lr = lr[lr["space"].isin(pos)].copy()
        lr["x"] = lr["space"].map(pos)
    else:
        top = lr["layer"].max()
        lr["x"] = lr["layer"] / top
        how = f"layer 0..{top}, depth = layer / {top}"
    lr["odd"] = lr["d_minus"] / lr["d"]
    return lr.sort_values("x").reset_index(drop=True), how


def twin_means(model_dir: Path, metrics: dict) -> tuple[float | None, float | None, list[int], list[int]]:
    b = pd.read_csv(model_dir / "blocks.csv")
    lr = b[b["group"] == "LR"]
    out = []
    for half in ("first_half", "second_half"):
        sel = lr[lr["block"].isin(blocks_of(metrics, "LR", half))]["frac_2_t50"]
        out.append(float(sel.mean()) if len(sel) else None)
    return out[0], out[1], blocks_of(metrics, "LR", "first_half"), blocks_of(metrics, "LR", "second_half")


def collect():
    must = model_dirs(MUST04)
    core = model_dirs(CORE04)
    both = sorted(set(must) & set(core))
    dirs = {m: (MUST04 if m in must else CORE04) / m for m in sorted(set(must) | set(core))}
    metrics = {m: load_metrics(p / "metrics.json") for m, p in dirs.items()}

    # panel (a): MUST-04 models with all four groups in the first half
    a_models = [m for m in must if all(ratio(metrics[m], g, HALF) is not None for g in GROUPS)]
    a_skipped = [m for m in must if m not in a_models]

    # panel (b): LR odd share along the stream, every model with a known type
    type_of = {m: t for t, (_, _, members) in MODEL_TYPES.items() for m in members}
    b_rows, b_missing = [], []
    for m, p in dirs.items():
        if m not in type_of:
            b_missing.append((m, "no model type"))
            continue
        if not (p / "layers.csv").exists():
            b_missing.append((m, "no layers.csv"))
            continue
        stream, how = stream_of(p, metrics[m])
        tw1, tw2, blk1, blk2 = twin_means(p, metrics[m]) if (p / "blocks.csv").exists() else (None, None, [], [])
        b_rows.append(dict(model=m, type=type_of[m], stream=stream, how=how, tw1=tw1, tw2=tw2, blk1=blk1, blk2=blk2))
    return dict(must=must, core=core, both=both, dirs=dirs, metrics=metrics, a_models=a_models,
                a_skipped=a_skipped, b_rows=b_rows, b_missing=b_missing)


# ---------------------------------------------------------------------------------------------- panels
def panel_a(ax, d) -> list[dict]:
    models, metrics = d["a_models"], d["metrics"]
    n_m = len(models)
    m_off = (np.arange(n_m) - (n_m - 1) / 2) * 0.13 if n_m > 1 else np.zeros(1)
    half_w = 0.3                                  # half width of the mean bar
    rows = []
    for gi, (g, (label, colour, marker)) in enumerate(GROUPS.items()):
        vals = [ratio(metrics[m], g, HALF) for m in models]
        ax.scatter(gi + m_off, vals, s=17, marker=marker, facecolors=tint(colour), edgecolors=colour,
                   linewidths=0.7, zorder=3)
        mean = float(np.mean(vals))
        ax.plot([gi - half_w, gi + half_w], [mean, mean], color=colour, lw=2.0, solid_capstyle="butt", zorder=4)
        for m, v in zip(models, vals):
            rows.append(dict(model=m, group=g, value=v, blocks=blocks_of(metrics[m], g, HALF)))
        rows.append(dict(model=f"mean of {n_m}", group=g, value=mean, blocks=[]))

    ax.axhline(1.0, color=C_INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)  # above the grid line at 1.0
    ax.text(len(GROUPS) - 1 + 0.5, 1.0 + 0.012, "no structure", ha="right", va="bottom", fontsize=8, color=C_INK2)
    ax.set_xticks(range(len(GROUPS)), [v[0] for v in GROUPS.values()])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.55, len(GROUPS) - 1 + 0.55)
    ax.set_ylim(0.3, 1.07)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_ylabel("ratio to null (lower = stronger pairing)")
    ax.set_title("(a)  pairing by symmetry, early blocks", loc="left", fontsize=9, pad=6)

    key = [Line2D([], [], ls="none", marker="o", ms=3.9, mfc=tint(C["grey"]), mec=C_INK2, mew=0.7,
                  label=f"one model (n = {n_m})"),
           Line2D([], [], color=C_INK2, lw=2.0, solid_capstyle="butt", label="mean")]
    ax.legend(handles=key, loc="upper left", bbox_to_anchor=(-0.55, 0.965), bbox_transform=ax.transData,
              handlelength=1.2, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    return rows


def panel_b(ax, d) -> None:
    rows = d["b_rows"]
    order = list(MODEL_TYPES)
    # muted lines first, then the named ones; within each, types in legend order
    draw = sorted(rows, key=lambda r: (r["model"] in LABELLED, order.index(r["type"]), r["model"]))
    for r in draw:
        colour = MODEL_TYPES[r["type"]][1]
        named = r["model"] in LABELLED
        s = r["stream"]
        ax.plot(s["x"], s["odd"], color=colour, lw=1.3 if named else 0.8, alpha=1.0 if named else 0.6,
                ls=LINESTYLE[r["type"]], solid_joinstyle="round", solid_capstyle="round", zorder=4 if named else 3)

    # direct labels: at the line end, or (for ends in the bunch near zero) above it with a leader
    for r in rows:
        if r["model"] not in LABELLED:
            continue
        s, colour, name = r["stream"], MODEL_TYPES[r["type"]][1], SHORT_NAMES[r["model"]]
        if r["model"] in LEADER_LABELS:
            x_l, y_l = LEADER_LABELS[r["model"]]
            y_line = float(np.interp(x_l, s["x"], s["odd"]))
            ax.plot([x_l, x_l], [y_line, y_l - 0.004], color=colour, lw=0.7, zorder=5)
            ax.scatter([x_l], [y_line], s=7, color=colour, edgecolors=SURFACE, linewidths=0.4, zorder=6)
            ax.text(x_l + 0.008, y_l, name, ha="right", va="bottom", fontsize=8, color=C_INK, zorder=6)
        else:
            ax.text(s["x"].iloc[-1], s["odd"].iloc[-1] + 0.012, name, ha="right", va="bottom", fontsize=8,
                    color=C_INK, zorder=6)

    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    top = max(float(r["stream"]["odd"].max()) for r in rows)
    ax.set_ylim(0, math.ceil(top * 10) / 10)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_xlabel("relative depth")
    ax.set_ylabel(r"odd share of the stream, $d_-/d$")
    ax.set_title("(b)  how much mirror structure is left with depth", loc="left", fontsize=9, pad=6)
    n_type = {t: sum(r["type"] == t for r in rows) for t in order}
    handles = [Line2D([], [], color=MODEL_TYPES[t][1], lw=1.3, ls=LINESTYLE[t],
                      label=f"{MODEL_TYPES[t][0]} ({n_type[t]})")
               for t in order if n_type[t]]
    ax.legend(handles=handles, loc="upper right", handlelength=1.4, handletextpad=0.5, borderaxespad=0.2,
              labelspacing=0.3)


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    d = collect()
    if not d["a_models"] or not d["b_rows"]:
        print("nothing to plot: panel (a) models", d["a_models"], "panel (b) rows", len(d["b_rows"]))
        return 1
    apply()                                    # shared paper style: fonts, spines, grid, line widths
    plt.rcParams["savefig.bbox"] = "standard"  # keep the page exactly FIG_W x FIG_H (the style default trims it)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.0, 1.75]})
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.08)
    a_rows = panel_a(ax_a, d)
    panel_b(ax_b, d)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{OUT_STEM}.pdf", OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ log what was plotted
    print(f"repo git HEAD: {git_head()}")
    print(f"MUST-04 models: {d['must']}")
    print(f"CORE-04 structure models: {d['core']}")
    print(f"in both (MUST-04 used): {d['both'] or 'none'}")
    print("\nsources:")
    for m, p in d["dirs"].items():
        print(f"  {m:20s} {p.relative_to(ROOT).as_posix():42s} {provenance(d['metrics'][m])}")

    print("\npanel (a): first-half ratio_to_null by group (models with all four groups)")
    if d["a_skipped"]:
        print(f"  MUST-04 models left out (a group missing): {d['a_skipped']}")
    print(f"  {'model':20s} " + " ".join(f"{g:>6s}" for g in GROUPS) + "  blocks")
    for m in d["a_models"] + [f"mean of {len(d['a_models'])}"]:
        vals = {r["group"]: r for r in a_rows if r["model"] == m}
        blk = vals["LR"]["blocks"]
        print(f"  {m:20s} " + " ".join(f"{vals[g]['value']:6.3f}" for g in GROUPS)
              + (f"  {blk[0]}-{blk[-1]}" if blk else ""))

    print("\npanel (b): LR odd share d_minus / d along the stream (first and last plotted point), "
          "and mean LR twin fraction (blocks.csv frac_2_t50) per half")
    print(f"  {'model':20s} {'type':10s} {'pts':>3s} {'x0':>5s} {'d0':>5s} {'d-0':>5s} {'odd0':>5s}  "
          f"{'dL':>5s} {'d-L':>5s} {'oddL':>5s}  {'twin1':>5s} {'twin2':>5s}  halves      stream")
    order = list(MODEL_TYPES)
    for r in sorted(d["b_rows"], key=lambda r: (order.index(r["type"]), r["model"])):
        s = r["stream"]
        f0, fl = s.iloc[0], s.iloc[-1]
        tw = lambda v: f"{v:5.3f}" if v is not None else "  n/a"  # noqa: E731
        halves = f"{r['blk1'][0]}-{r['blk1'][-1]} / {r['blk2'][0]}-{r['blk2'][-1]}" if r["blk1"] else ""
        print(f"  {r['model']:20s} {r['type']:10s} {len(s):3d} {f0['x']:5.3f} {int(f0['d']):5d} "
              f"{int(f0['d_minus']):5d} {f0['odd']:5.3f}  {int(fl['d']):5d} {int(fl['d_minus']):5d} "
              f"{fl['odd']:5.3f}  {tw(r['tw1'])} {tw(r['tw2'])}  {halves:11s} {r['how']}")
    if d["b_missing"]:
        print(f"  left out of panel (b): {d['b_missing']}")
    print(f"\nwrote {pdf.relative_to(ROOT).as_posix()} and {png.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""F3 (paper figure): the discovery null controls of A03 against the real mirror, blocks 1 to 6.

    .venv/Scripts/python.exe scripts/fp_fig_controls.py                      # reads results/A03
    .venv/Scripts/python.exe scripts/fp_fig_controls.py --dry-run            # synthetic A03 files, temporary folder

Reads only these files (every plotted value comes from them; nothing is typed in by hand):
  <results>/A03/controls.csv   one row per control, written by scripts/fp_a03_controls.py:
                                 twin_fraction_t50_<range>   twin fraction (panel a)
                                 ratio_to_null_split_<range> split-half ratio to null, the statistic rule A03-1 scores
                                 ratio_to_null_<range>       same-sample ratio to null (the MUST-04 diagnostic); for C4
                                                             the null's own leave-one-out ratio, with ratio_lo_<range>
                                                             and ratio_hi_<range> (2.5 and 97.5 percentiles of the draws)
  <results>/A03/metrics.json   optional: the C4 twin-fraction interval (summaries.C4.<range>.twin_lo and twin_hi), the
                               two split-half directions (split_half.<row>.ratio_A0_P1 and ratio_A1_P0), the bars and
                               the A03-1 verdict (scores.A03-1), the null draw count and the model name. Anything
                               missing is skipped with a printed note.

Panel (a): twin fraction in blocks 1 to 6 (share of MLP hidden units in a matched pair whose symmetrised mirror
correlation is at least 0.5) of the real mirror, C1, C2, C3 and the C4 null (its mean over draws, with the 2.5 to
97.5 percentile range when metrics.json has it). Panel (b): ratio to null of the breaking share. Filled markers are
the split-half ratio (operators fitted on one half of the fit images, pairings on the other; PREREG_FINAL_PHASE.md
Section 2.4), the statistic A03-1 scores; a thin line behind a filled marker spans its two split directions. The
same-sample ratio is drawn lighter, as an open marker: a diagnostic that sits below 1 for any fitted structure. C4
has no split half: its row is the null itself (mean and percentile range of the leave-one-out ratios of its draws).
Dashed lines are the pre-registered bars (controls pass with twin fraction at most 0.05 and ratio at least 0.9; read
from metrics.json, with the PREREG_FINAL_PHASE.md values as fallback); the thin solid line at 1 marks the null.

Style: scripts/paper_style.py (apply(), C, ROLE): the real mirror in ROLE["mirror"], the controls in vermilion, the
null in grey; text in ink colours only. Identity never rests on colour: every row is named on the axis. Fixed page
of 6.0 x 2.2 inches. Writes <out>/fig_controls.pdf (vector), fig_controls.png (300 dpi) and fig_controls.json (git
hash, inputs with sha256, plotted values, notes), prints every plotted value and a caption. No randomness.

--dry-run writes synthetic files in the real formats to a temporary folder (or --work) and draws them there, marked
"synthetic data"; it never writes to paper/figures unless --out says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
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

STEM = "fig_controls"
TWIN_MAX, RATIO_MIN = 0.05, 0.9          # fallback only: PREREG_FINAL_PHASE.md Section 1, rule A03-1
ROWS = {  # key: (axis label, colour); dict order = top to bottom
    "mirror": ("real mirror", ROLE["mirror"]),
    "C1": ("C1 shuffled correspondence", C["terracotta"]),
    "C2": ("C2 no re-alignment", C["terracotta"]),
    "C3": ("C3 pixel permutation", C["terracotta"]),
    "C4": ("C4 random-operator null", C["grey"]),
}
ROW_Y = {"mirror": 0.0, "C1": 1.3, "C2": 2.3, "C3": 3.3, "C4": 4.6}   # gap after the reference and before the null
_HUE = {v: k for k, v in C.items() if f"{k}_lt" in C}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.2
AX_BOTTOM, AX_TOP = 0.66, 1.80          # inches from the bottom
A_AX = (1.47, 1.70)                     # (left, width) in inches
B_AX = (3.60, 2.30)
FS, FS_TITLE, MS = 8, 9, 4.6


def tint(colour: str) -> str:
    return C[_HUE[colour] + "_lt"] if colour in _HUE else colour


# ---------------------------------------------------------------------------------------------- small helpers
def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return head + (" (uncommitted changes)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ---------------------------------------------------------------------------------------------- data
def load(results: Path, rng: str, notes: list) -> dict:
    a03 = Path(results) / "A03"
    csv = a03 / "controls.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found (run scripts/fp_a03_controls.py first, or use --dry-run)")
    df = pd.read_csv(csv)
    mpath = a03 / "metrics.json"
    metrics = {}
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        notes.append(f"{rel(mpath)} not found: no C4 twin interval, no split directions, bars from the fallback")
    score = dig(metrics, "scores", "A03-1") or {}
    twin_max = finite(score.get("twin_max"))
    ratio_min = finite(score.get("ratio_min"))
    if twin_max is None or ratio_min is None:
        notes.append(f"bars not in metrics.json: fallback {TWIN_MAX} and {RATIO_MIN} (PREREG_FINAL_PHASE.md)")
    rows = []
    for key in ROWS:
        sub = df[df["control"].astype(str) == key]
        if sub.empty:
            notes.append(f"control {key} not in controls.csv: skipped")
            continue
        r = sub.iloc[0]
        get = (lambda col: finite(r[col]) if col in r.index else None)  # noqa: E731
        row = {"control": key, "label": ROWS[key][0],
               "twin": get(f"twin_fraction_t50_{rng}"),
               "split": get(f"ratio_to_null_split_{rng}"),
               "same": get(f"ratio_to_null_{rng}"),
               "same_lo": get(f"ratio_lo_{rng}"), "same_hi": get(f"ratio_hi_{rng}"),
               "twin_lo": finite(dig(metrics, "summaries", key, rng, "twin_lo")),
               "twin_hi": finite(dig(metrics, "summaries", key, rng, "twin_hi")),
               "split_a": finite(dig(metrics, "split_half", key, "ratio_A0_P1")),
               "split_b": finite(dig(metrics, "split_half", key, "ratio_A1_P0")),
               "status": dig(score, "controls", key, "status")}
        if key != "C4" and row["split"] is None:
            notes.append(f"{key}: no split-half ratio in controls.csv (run without --no-split-half): only the "
                         "same-sample ratio is drawn")
        if row["twin"] is None:
            notes.append(f"{key}: no twin fraction for range {rng}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{csv} has none of the rows {list(ROWS)}")
    return {"rows": rows, "metrics": metrics, "twin_max": twin_max if twin_max is not None else TWIN_MAX,
            "ratio_min": ratio_min if ratio_min is not None else RATIO_MIN, "status": score.get("status"),
            "model": metrics.get("model"), "n_draws": metrics.get("null_draws"),
            "inputs": {rel(p): sha256_of(p) for p in (csv, mpath) if p.exists()}}


# ---------------------------------------------------------------------------------------------- drawing
def axes_at(fig, left: float, width: float):
    return fig.add_axes([left / FIG_W, AX_BOTTOM / FIG_H, width / FIG_W, (AX_TOP - AX_BOTTOM) / FIG_H])


def line_label(ax, x: float, text: str, ha: str = "center") -> None:
    """A label just above the axes at data x (left of the line for ha right, right of it for ha left)."""
    dx = {"right": -2, "left": 2}.get(ha, 0)
    ax.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(dx, 2), textcoords="offset points",
                ha=ha, va="bottom", fontsize=FS - 1, color=C["ink2"], annotation_clip=False)


def bar_line(ax, x: float, text: str, ls, ha: str = "center") -> None:
    ax.axvline(x, color=C["ink2"], lw=0.8, ls=ls, zorder=1)
    line_label(ax, x, text, ha)


def style_rows(ax, ys: list, labels: list | None) -> None:
    ax.set_yticks(ys, labels if labels is not None else [""] * len(ys))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(max(ys) + 0.7, min(ys) - 0.7)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)


def draw(d: dict, out: Path, rng: str, synthetic: bool) -> tuple:
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "axes.labelsize": FS})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a, ax_b = axes_at(fig, *A_AX), axes_at(fig, *B_AX)
    rows = d["rows"]
    ys = [ROW_Y[r["control"]] for r in rows]

    # (a) twin fraction
    hi = [d["twin_max"]]
    for r in rows:
        col, y = ROWS[r["control"]][1], ROW_Y[r["control"]]
        if r["twin"] is None:
            continue
        if r["twin_lo"] is not None and r["twin_hi"] is not None:
            ax_a.plot([r["twin_lo"], r["twin_hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            hi.append(r["twin_hi"])
        ax_a.plot([r["twin"]], [y], ls="none", marker="o", ms=MS, mfc=col, mec=C["paper"], mew=0.5, zorder=3,
                  clip_on=False)
        hi.append(r["twin"])
    bar_line(ax_a, d["twin_max"], f"bar {d['twin_max']:g}", (0, (3, 2)))
    ax_a.set_xlim(0, max(1.6 * d["twin_max"], 1.12 * max(hi)))
    style_rows(ax_a, ys, [r["label"] for r in rows])
    ax_a.set_xlabel("twin fraction")
    ax_a.set_title("(a) twin fraction, blocks 1 to 6", loc="left", fontsize=FS_TITLE, pad=13)

    # (b) ratio to null
    xs = [d["ratio_min"], 1.0]
    for r in rows:
        col, y = ROWS[r["control"]][1], ROW_Y[r["control"]]
        if r["control"] == "C4":
            if r["same_lo"] is not None and r["same_hi"] is not None:
                ax_b.plot([r["same_lo"], r["same_hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
                xs += [r["same_lo"], r["same_hi"]]
            if r["same"] is not None:
                ax_b.plot([r["same"]], [y], ls="none", marker="D", ms=MS - 0.8, mfc=col, mec=C["paper"], mew=0.5,
                          zorder=3)
                xs.append(r["same"])
            continue
        if r["same"] is not None:
            ax_b.plot([r["same"]], [y], ls="none", marker="o", ms=MS, mfc=C["paper"], mec=tint(col), mew=1.0,
                      zorder=3)
            xs.append(r["same"])
        if r["split_a"] is not None and r["split_b"] is not None:
            ax_b.plot([r["split_a"], r["split_b"]], [y, y], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
            xs += [r["split_a"], r["split_b"]]
        if r["split"] is not None:
            ax_b.plot([r["split"]], [y], ls="none", marker="o", ms=MS, mfc=col, mec=C["paper"], mew=0.5, zorder=4)
            xs.append(r["split"])
    ax_b.axvline(1.0, color=C["ink2"], lw=0.6, zorder=1)
    line_label(ax_b, 1.0, "null", "left")
    bar_line(ax_b, d["ratio_min"], f"bar {d['ratio_min']:g}", (0, (3, 2)), "right")
    lo, hi_b = min(xs), max(xs)
    pad = 0.06 * (hi_b - lo) if hi_b > lo else 0.05
    ax_b.set_xlim(lo - pad, hi_b + pad)
    style_rows(ax_b, ys, None)
    ax_b.set_xlabel("ratio to null (1 = no structure)")
    ax_b.set_title("(b) ratio to null, blocks 1 to 6", loc="left", fontsize=FS_TITLE, pad=13)
    handles = [Line2D([], [], ls="none", marker="o", ms=MS, mfc=C["ink2"], mec=C["paper"], mew=0.5,
                      label="split half (rule A03-1)"),
               Line2D([], [], ls="none", marker="o", ms=MS, mfc=C["paper"], mec=tint(C["grey"]), mew=1.0,
                      label="same sample (diagnostic)")]
    if any(r["control"] == "C4" for r in rows):
        handles.append(Line2D([], [], color=C["grey"], lw=1.0, marker="D", ms=MS - 0.8, mfc=C["grey"],
                              mec=C["paper"], mew=0.5, label="null: mean, 95% of draws"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), fontsize=FS - 1, handlelength=1.3, handletextpad=0.4, columnspacing=1.0,
               borderaxespad=0.1)
    if synthetic:
        fig.text(0.995, 0.995, "synthetic data (dry run)", ha="right", va="top", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def caption(d: dict) -> str:
    n = d.get("n_draws")
    status = d.get("status") or "not in metrics.json"
    return (f"Discovery null controls (A03), blocks 1 to 6{' of ' + d['model'] if d.get('model') else ''}. "
            "(a) Twin fraction: the share of MLP hidden units in a matched pair whose symmetrised mirror "
            "correlation is at least 0.5. (b) Ratio to null of the share of weight energy the structure's projection "
            "removes: filled markers are the split-half ratio (operators fitted on one half of the fit images, "
            "pairings on the other), the statistic rule A03-1 scores, with a thin line across its two split "
            "directions; light open markers are the same-sample ratio, a diagnostic that sits below 1 for any fitted "
            f"structure. C4 is the null itself (mean and 2.5 to 97.5 percentiles over its {n if n else ''} draws). "
            f"Dashed lines: the pre-registered bars (each control passes with a twin fraction of at most "
            f"{d['twin_max']:g} and a ratio of at least {d['ratio_min']:g}). Rule A03-1: {status}.")


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, rng: str = "1-6", seed: int = 0) -> Path:
    """Synthetic results/A03/controls.csv and metrics.json in the formats of scripts/fp_a03_controls.py (values
    from a seeded generator; they mean nothing)."""
    g = np.random.default_rng(seed)
    res = Path(work) / "results"
    a03 = res / "A03"
    a03.mkdir(parents=True, exist_ok=True)
    rows, summaries, split = [], {}, {}
    for key in ROWS:
        real = key == "mirror"
        twin = float(g.uniform(0.25, 0.4)) if real else float(g.uniform(0.0, 0.02))
        same = float(g.uniform(0.4, 0.55)) if real else float(g.uniform(0.85, 0.92))
        row = {"control": key, "label": ROWS[key][0], "n_images": 2000, "d_minus_span": int(g.integers(300, 380)),
               f"twin_fraction_t50_{rng}": twin, f"ratio_to_null_{rng}": same,
               "twin_fraction_t50_7-12": twin / 4, "ratio_to_null_7-12": same}
        if key == "C4":
            row[f"twin_fraction_t50_{rng}"] = float(g.uniform(0.0, 0.01))
            row[f"ratio_to_null_{rng}"] = 1.0 + float(g.normal(0, 0.003))
            row[f"ratio_lo_{rng}"], row[f"ratio_hi_{rng}"] = 0.97, 1.03
            summaries["C4"] = {rng: {"twin_fraction_t50": row[f"twin_fraction_t50_{rng}"], "twin_lo": 0.0,
                                     "twin_hi": 0.015, "ratio_to_null": row[f"ratio_to_null_{rng}"]}}
        else:
            sp = float(g.uniform(0.45, 0.6)) if real else float(g.uniform(0.93, 1.01))
            row[f"ratio_to_null_split_{rng}"] = sp
            row[f"twin_fraction_t50_split_{rng}"] = twin
            split[key] = {"ratio_to_null": sp, "ratio_A0_P1": sp - 0.01, "ratio_A1_P0": sp + 0.012}
        rows.append(row)
    pd.DataFrame(rows).to_csv(a03 / "controls.csv", index=False)
    statuses = {c: {"status": "PASS"} for c in ("C1", "C2", "C3")}
    metrics = {"experiment": "A03", "model": "synthetic", "null_draws": 20, "summaries": summaries,
               "split_half": split, "scores": {"A03-1": {"twin_max": TWIN_MAX, "ratio_min": RATIO_MIN,
                                                         "status": "PASS", "controls": statuses}},
               "dry_run": True}
    (a03 / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (reads <results>/A03)")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--range", default="1-6", help="block range of the A03 columns (1-based)")
    p.add_argument("--dry-run", action="store_true", help="synthetic A03 files in a temporary folder")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    notes: list = []
    results = Path(a.results)
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    if a.dry_run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_controls_"))
        results = make_dry_inputs(work, a.range)
        out = Path(a.out) if a.out else work / "figures"
        print(f"dry run: synthetic inputs in {work}")
    d = load(results, a.range, notes)
    pdf, png = draw(d, out, a.range, a.dry_run)

    print(f"git: {git_head()}")
    for path, sha in d["inputs"].items():
        print(f"source: {path} (sha256 {str(sha)[:12]})")
    print(f"bars: twin fraction <= {d['twin_max']:g}, ratio to null >= {d['ratio_min']:g}; range {a.range}")
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    print(f"  {'control':8s} {'twin':>7s} {'(lo, hi)':>18s} {'split':>7s} {'(A0P1, A1P0)':>18s} {'same':>7s} "
          f"{'(lo, hi)':>18s}  A03-1")
    for r in d["rows"]:
        print(f"  {r['control']:8s} {fmt(r['twin']):>7s} {'(' + fmt(r['twin_lo']) + ', ' + fmt(r['twin_hi']) + ')':>18s} "
              f"{fmt(r['split']):>7s} {'(' + fmt(r['split_a']) + ', ' + fmt(r['split_b']) + ')':>18s} "
              f"{fmt(r['same']):>7s} {'(' + fmt(r['same_lo']) + ', ' + fmt(r['same_hi']) + ')':>18s}  "
              f"{r['status'] or ''}")
    for n in notes:
        print(f"note: {n}")
    cap = caption(d)
    print(f"\nCaption: {cap}")
    meta = {"figure": STEM, "git": git_head(), "argv": list(sys.argv[1:] if argv is None else argv),
            "dry_run": bool(a.dry_run), "inputs": d["inputs"], "range": a.range, "bars": {
                "twin_max": d["twin_max"], "ratio_min": d["ratio_min"]}, "rows": d["rows"], "verdict": d["status"],
            "notes": notes, "caption": cap}
    mpath = out / f"{STEM}.json"
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {rel(pdf)}, {rel(png)} and {rel(mpath)}")
    return {"pdf": pdf, "png": png, "meta": mpath, "notes": notes, "rows": d["rows"]}


if __name__ == "__main__":
    main()

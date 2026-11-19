#!/usr/bin/env python
"""F1 (paper figure): the accuracy-size frontier of A06, with stacking and the flip-averaged models of A07.

    .venv/Scripts/python.exe scripts/fp_fig_frontier.py                           # runs/final and results/
    .venv/Scripts/python.exe scripts/fp_fig_frontier.py --with-prediction         # also the A06-2 prediction
    .venv/Scripts/python.exe scripts/fp_fig_frontier.py --from-values paper/figures/fig_frontier.json   # re-draw
    .venv/Scripts/python.exe scripts/fp_fig_frontier.py --dry-run                 # synthetic run tree, temp folder

Reads only these files (every plotted value comes from them; nothing is typed in by hand). Run directories follow
docs/final_phase_design.md Sections 4.4 and 4.5a: <runs>/<group>/<model>/<method>[_<variant>]/s<seed>/.
  <run>/eval/step<S>.json      k-NN records of mirror/final/eval/knn_fast.py: passes.bf16.top1 (all 50,000 validation
                               queries, the full 100k bank) at the last step that has it (or --step); for flip
                               averaging passes.bf16_flipavg.top1 and its gflops_per_image, and the record's "cost"
                               (mirror/final/eval/flipavg.analytic_cost: gflops_per_forward, gflops_dense, params_frac)
  <run>/metrics.json           "params" (written by scripts/fp_heal.py from the student builder): model_params_frac or
                               params_frac, the fraction of the original's 86,583,552 parameters; else spec.json
                               fp_heal.params, else the eval record's cost.params_frac
  <runs>/A07/<name>/eval/*.json  flip-averaged released models (scripts/fp_eval.py --original <name> --out
                               runs/final/A07/<name> --flip-average --cost): DINOv2-S/14 reg and the original itself
  <results>/A01/decision.json  {"adopted": true|false}: the main Mirror arm is mirror_exact_wspace when adopted, else
                               mirror_exact_astar (design doc 4.5a); --mirror-variant overrides
  --compute-csv FILE           optional columns key, compute_frac: places two-pass points at a measured compute (for
                               example A12's batch-256 time per image relative to the original); keys are printed

Points (DINOv2-B/14 reg unless --model): SVD-LLM at 0.40, 0.50, 0.65 (group a06, svdllm_f040 ...) and 0.765 (group
main, svdllm); Mirror Surgery at 0.765 (the main arm) and Mirror + SVD-LLM at 0.40, 0.50, 0.65 (a06,
mirror_svdllm_f0xx), drawn as one exactly invariant curve; Mirror on the first 9 blocks (a06, mirror_partial_k9,
0.766); the original distilled with the same recipe (main, original) at 1.0; flip-averaged models (any
bf16_flipavg pass of a non-invariant arm above, and every A07 record) as stars at their compute: two forward passes
times the analytic GFLOPs of one pass, divided by the original's single-pass GFLOPs (the A07 record of the original,
or any record's cost.gflops_dense of the same model); without a cost record, twice the parameter fraction (printed).
Each point is the mean over the seeds found, with an error bar of one standard deviation (ddof 1) where there are
at least two seeds. Optional (--with-prediction): the pre-registered A06-2 prediction for Mirror + SVD-LLM,
mirror.final.baselines.sizes.acc_pred on the plotted means, at the nominal sizes of PREREG_FINAL_PHASE.md.

Marker key: filled markers are exactly mirror-invariant in one forward pass (Mirror Surgery and Mirror + SVD-LLM, by
construction; the measured fp32 flip consistency is printed as a check); open markers are not; stars are invariant
only with two passes. Missing inputs (an arm without runs, no A07 records yet) are skipped with a printed note.

Style: scripts/paper_style.py (apply(), C, ROLE): Mirror in ROLE["mirror"], SVD-LLM in ROLE["svdllm"], the 9-block
model sky blue, the original in ink; text in ink colours. Identity also rests on marker shape and the legend. Log
size axis (the A06 prediction interpolates in log size). Fixed page 6.0 x 2.6 inches. Writes <out>/fig_frontier.pdf,
.png (300 dpi) and fig_frontier.json (git hash, every input with sha256, every plotted value; --from-values re-draws
from it). No randomness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT))                      # the mirror package (only --with-prediction imports it)
from paper_style import C, ROLE, apply  # noqa: E402

STEM = "fig_frontier"
MODEL = "dinov2_vitb14_reg"
SERIES = {  # key: (legend label, colour, marker, filled = exactly invariant in one pass, line)
    "mirror": ("Mirror Surgery\n(+ SVD-LLM below 0.765)", ROLE["mirror"], "o", True, True),
    "svdllm": ("SVD-LLM", ROLE["svdllm"], "s", False, True),
    "partial": ("Mirror, first 9 blocks only", C["sky"], "D", False, False),
    "original": ("original, distilled", ROLE["original"], "o", False, False),
}
# (group, method directory, series, point label, nominal size of the plan (fallback only, printed when used))
def frontier_arms(mv: str) -> list:
    return [("main", f"mirror_exact_{mv}", "mirror", "Mirror Surgery", 0.765),
            ("a06", "mirror_svdllm_f065", "mirror", "Mirror + SVD-LLM 0.65", 0.65),
            ("a06", "mirror_svdllm_f050", "mirror", "Mirror + SVD-LLM 0.50", 0.50),
            ("a06", "mirror_svdllm_f040", "mirror", "Mirror + SVD-LLM 0.40", 0.40),
            ("main", "svdllm", "svdllm", "SVD-LLM 0.765", 0.765),
            ("a06", "svdllm_f065", "svdllm", "SVD-LLM 0.65", 0.65),
            ("a06", "svdllm_f050", "svdllm", "SVD-LLM 0.50", 0.50),
            ("a06", "svdllm_f040", "svdllm", "SVD-LLM 0.40", 0.40),
            ("a06", "mirror_partial_k9", "partial", "Mirror, first 9 blocks", 0.766),
            ("main", "original", "original", "original, distilled", 1.0)]


SHORT_NAMES = {"dinov2_vits14_reg": "DINOv2-S/14 reg", "dinov2_vitb14_reg": "original",
               "dinov2_vitl14_reg": "DINOv2-L/14 reg", "deit3_b16": "DeiT III-B/16"}
STAR_COLOUR = {"svdllm": ROLE["svdllm"], "partial": C["sky"], "original": ROLE["original"], "released": C["grey"]}

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.6
AX = (0.52, 0.42, 3.50, 2.00)            # left, bottom, width, height (inches)
LEGEND_X = 4.18                          # inches
X_TICKS = (0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5, 2.0, 3.0)
FS, MS = 8, 4.4


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


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ---------------------------------------------------------------------------------------------- run tree
def seed_dirs(arm: Path) -> list:
    out = []
    if arm.is_dir():
        for p in arm.iterdir():
            m = re.fullmatch(r"s(\d+)", p.name)
            if m and p.is_dir():
                out.append((int(m.group(1)), p))
    return sorted(out)


def eval_files(run: Path) -> dict:
    out = {}
    ev = run / "eval"
    if ev.is_dir():
        for p in ev.glob("step*.json"):
            m = re.fullmatch(r"step(\d+)\.json", p.name)
            if m:
                out[int(m.group(1))] = p
    return out


def final_record(run: Path, family: str, step: int | None):
    """(step, path, record) of the last step (or `step`) whose record has `family` on all queries."""
    files = eval_files(run)
    steps = [step] if step is not None else sorted(files, reverse=True)
    for s in steps:
        if s in files:
            rec = read_json(files[s])
            if rec and finite(dig(rec, "passes", family, "top1")) is not None:
                return s, files[s], rec
    return None


def params_frac(run: Path, rec: dict) -> tuple:
    """(fraction of the original's parameters, source) from metrics.json params, spec.json fp_heal.params or the
    eval record's cost."""
    cands = (("metrics.json params", dig(read_json(run / "metrics.json"), "params")),
             ("spec.json fp_heal.params", dig(read_json(run / "spec.json"), "fp_heal", "params")),
             ("eval cost", (rec or {}).get("cost")))
    for src, d in cands:
        if not isinstance(d, dict):
            continue
        for k in ("model_params_frac", "params_frac"):
            v = finite(d.get(k))
            if v is not None:
                return v, f"{src}.{k}"
        a, b = finite(d.get("model_params_structured")), finite(d.get("model_params_original"))
        if a is not None and b:
            return a / b, f"{src}.model_params_structured / model_params_original"
    return None, None


def main_variant(results: Path, override: str | None, notes: list) -> str:
    if override:
        return override
    dec = read_json(Path(results) / "A01" / "decision.json")
    if dec is None:
        notes.append("results/A01/decision.json not found: the main Mirror arm is mirror_exact_astar (current A*)")
        return "astar"
    return "wspace" if dec.get("adopted") else "astar"


def reference_gflops(records: list, model: str) -> tuple:
    """The original model's analytic GFLOPs of one forward pass: its A07 record, else any record's
    cost.gflops_dense for the same model."""
    for rec, path in records:
        if rec.get("original") and rec.get("model") == model and finite(dig(rec, "cost", "gflops_per_forward")):
            return float(rec["cost"]["gflops_per_forward"]), f"{rel(path)} cost.gflops_per_forward"
    for rec, path in records:
        if rec.get("model") in (model, None) and finite(dig(rec, "cost", "gflops_dense")):
            return float(rec["cost"]["gflops_dense"]), f"{rel(path)} cost.gflops_dense"
    return None, None


def two_pass_x(rec: dict, pass_key: str, ref: float | None, frac: float | None) -> tuple:
    g = finite(dig(rec, "passes", pass_key, "gflops_per_image"))
    passes = finite(dig(rec, "passes", pass_key, "forward_passes_per_image")) or 2.0
    if g is None and finite(dig(rec, "cost", "gflops_per_forward")) is not None:
        g = passes * float(rec["cost"]["gflops_per_forward"])
    if g is not None and ref:
        return g / ref, "analytic GFLOPs per image / the original's single pass"
    if frac is not None:
        return passes * frac, "no cost record: passes x parameter fraction"
    return None, None


def summarise(values: list) -> tuple:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)


def collect(runs: Path, model: str, family: str, step: int | None, mv: str, a07_dirs: list,
            compute_csv: dict, notes: list) -> dict:
    points, inputs, flip_records, all_records = [], {}, [], []
    for group, method, series, label, nominal in frontier_arms(mv):
        arm = runs / group / model / method
        seeds = seed_dirs(arm)
        if not seeds:
            notes.append(f"{group}/{model}/{method}: no run directory: {label} skipped")
            continue
        vals, sds, xs, xsrc, srcs, flip = [], [], [], set(), [], []
        for seed, run in seeds:
            fr = final_record(run, family, step)
            if fr is None:
                notes.append(f"{rel(run)}: no {family} pass on all queries{'' if step is None else f' at step {step}'}"
                             ": seed skipped")
                continue
            s, path, rec = fr
            all_records.append((rec, path))
            inputs[rel(path)] = sha256_of(path)
            x, src = params_frac(run, rec)
            vals.append(float(rec["passes"][family]["top1"]))
            sds.append(seed)
            if x is not None:
                xs.append(x)
                xsrc.add(src)
            srcs.append(f"{rel(path)} passes.{family}.top1")
            fc = finite(dig(rec, "passes", "fp32", "flip_consistency"))
            if fc is not None and series == "mirror" and fc < 100.0:
                notes.append(f"{rel(path)}: fp32 flip consistency {fc:.3f}% for an exactly invariant method (check)")
            if f"{family}_flipavg" in (rec.get("passes") or {}):
                flip.append((seed, rec, path, x))
        if not vals:
            continue
        if xs:
            x_val, x_kind = float(np.mean(xs)), "; ".join(sorted(xsrc))
        else:
            x_val, x_kind = nominal, "nominal size of the plan (no parameter count found)"
            notes.append(f"{label}: no parameter count in the run files; drawn at the nominal size {nominal}")
        y, sd = summarise(vals)
        points.append({"key": f"{group}/{method}", "series": series, "label": label, "x": x_val, "x_kind": x_kind,
                       "y": y, "sd": sd, "n": len(vals), "seeds": sds, "values": vals, "sources": srcs})
        if flip and series == "mirror":
            notes.append(f"{label}: flip-averaged passes ignored (the model is already exactly invariant)")
        elif flip:
            flip_records.append((series, label, flip))
    # released models evaluated with flip averaging (A07)
    released, seen_dirs = [], set()
    for d in a07_dirs:
        key = str(d.resolve()).lower() if d.is_dir() else None     # A07 and a07 are one folder on Windows
        if key is None or key in seen_dirs:
            continue
        seen_dirs.add(key)
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            for path in sorted((sub / "eval").glob("*.json")) if (sub / "eval").is_dir() else []:
                if path.name.endswith("_meta.json"):
                    continue
                rec = read_json(path)
                if rec and f"{family}_flipavg" in (rec.get("passes") or {}):
                    released.append((sub.name, rec, path))
                    all_records.append((rec, path))
    if not released and not flip_records:
        notes.append("no flip-averaged results (A07) yet: no stars drawn")
    ref, ref_src = reference_gflops(all_records, model)
    if (released or flip_records) and ref is None:
        notes.append("no analytic GFLOPs of the original found: two-pass points use passes x parameter fraction")
    key_pass = f"{family}_flipavg"
    for series, label, flip in flip_records:
        vals, xs, kinds, srcs = [], [], set(), []
        for seed, rec, path, frac in flip:
            x, kind = two_pass_x(rec, key_pass, ref, frac)
            vals.append(float(rec["passes"][key_pass]["top1"]))
            if x is not None:
                xs.append(x)
                kinds.add(kind)
            srcs.append(f"{rel(path)} passes.{key_pass}")
        if not xs:
            notes.append(f"{label}, flip-averaged: no compute or size found: skipped")
            continue
        y, sd = summarise(vals)
        points.append({"key": f"{label} flipavg", "series": "flipavg", "base": series, "label": label,
                       "x": float(np.mean(xs)), "x_kind": "; ".join(sorted(kinds)), "y": y, "sd": sd,
                       "n": len(vals), "values": vals, "sources": srcs})
    for name, rec, path in released:
        inputs[rel(path)] = sha256_of(path)
        frac = finite(dig(rec, "cost", "params_frac"))
        pd_, pm = finite(dig(rec, "cost", "params_dense")), finite(dig(rec, "cost", "params_model"))
        if frac is None and name == model:
            frac = 1.0
        x, kind = two_pass_x(rec, key_pass, ref, frac)
        if x is None:
            notes.append(f"{rel(path)}: no compute or size: skipped")
            continue
        label = SHORT_NAMES.get(name, name)
        points.append({"key": f"A07/{name} flipavg", "series": "flipavg",
                       "base": "original" if name == model else "released", "label": label, "x": x, "x_kind": kind,
                       "y": float(rec["passes"][key_pass]["top1"]), "sd": None, "n": 1,
                       "values": [float(rec["passes"][key_pass]["top1"])],
                       "sources": [f"{rel(path)} passes.{key_pass}"], "params": pm if pm is not None else pd_})
    for p in points:
        if p["key"] in compute_csv:
            p["x"], p["x_kind"] = compute_csv[p["key"]], "--compute-csv"
    return {"points": points, "inputs": inputs, "ref_gflops": ref, "ref_source": ref_src}


def prediction(points: list, notes: list) -> list:
    """The pre-registered A06-2 prediction (mirror.final.baselines.sizes.acc_pred) on the plotted means."""
    from mirror.final.baselines import sizes as SZ

    by = {p["key"]: p for p in points}
    svd = {0.40: "a06/svdllm_f040", 0.50: "a06/svdllm_f050", 0.65: "a06/svdllm_f065", 0.765: "main/svdllm"}
    mirror = next((p for p in points if p["series"] == "mirror" and p["key"].startswith("main/")), None)
    orig = by.get("main/original")
    if mirror is None or orig is None or any(k not in by for k in svd.values()):
        notes.append("--with-prediction: needs every SVD-LLM point, the main Mirror arm and the distilled original")
        return []
    pts = {s: by[k]["y"] for s, k in svd.items()}
    out = []
    for B in SZ.A06_FRACTIONS:
        y = SZ.acc_pred(B, pts, orig["y"], mirror["y"])
        out.append({"key": f"prediction {B:g}", "series": "prediction", "label": f"A06-2 prediction {B:g}",
                    "x": float(B), "x_kind": "nominal size (PREREG_FINAL_PHASE.md)", "y": float(y), "sd": None,
                    "n": 0, "sources": ["mirror.final.baselines.sizes.acc_pred on the plotted means"]})
    return out


# ---------------------------------------------------------------------------------------------- drawing
def draw(points: list, out: Path, synthetic: bool, notes: list) -> tuple:
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "axes.labelsize": FS})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([AX[0] / FIG_W, AX[1] / FIG_H, AX[2] / FIG_W, AX[3] / FIG_H])
    ax.set_xscale("log")
    drawn = []
    main_x = next((p["x"] for p in points if p["series"] == "mirror" and p["key"].startswith("main/")), None)
    for key, (label, col, marker, filled, line) in SERIES.items():
        pts = sorted((p for p in points if p["series"] == key), key=lambda p: p["x"])
        if not pts:
            continue
        xs = np.array([p["x"] for p in pts])
        if key == "partial" and main_x is not None:
            close = np.abs(xs / main_x - 1) < 0.01
            if close.any():
                xs = np.where(close, xs * 1.018, xs)
                notes.append("the 9-block point is drawn 1.8% to the right of its size so it does not hide the full "
                             "conversion (its size is printed)")
        ys = np.array([p["y"] for p in pts])
        st = (dict(marker=marker, ms=MS, mfc=col, mec=C["paper"], mew=0.5) if filled else
              dict(marker=marker, ms=MS - 0.3, mfc=C["paper"], mec=col, mew=0.9))
        ax.plot(xs, ys, color=col, lw=1.1 if line else 0, zorder=3, solid_joinstyle="round", **st)
        for x, p in zip(xs, pts):
            if p.get("sd"):
                ax.errorbar([x], [p["y"]], yerr=[p["sd"]], fmt="none", ecolor=col, elinewidth=0.8, capsize=1.6,
                            capthick=0.8, zorder=2)
        drawn += pts
    stars = [p for p in points if p["series"] == "flipavg"]
    preds = [p for p in points if p["series"] == "prediction"]
    x_all = [p["x"] for p in drawn + stars + preds]
    x_lim = (min(x_all) / 1.1, max(x_all) * 1.1)
    for p in stars:
        col = STAR_COLOUR.get(p.get("base"), C["grey"])
        ax.plot([p["x"]], [p["y"]], ls="none", marker="*", ms=MS + 3.0, mfc=C["paper"], mec=col, mew=0.8, zorder=4)
        if p.get("sd"):
            ax.errorbar([p["x"]], [p["y"]], yerr=[p["sd"]], fmt="none", ecolor=col, elinewidth=0.8, capsize=1.6,
                        zorder=2)
        right = math.log(p["x"]) > math.log(x_lim[0]) + 0.82 * (math.log(x_lim[1]) - math.log(x_lim[0]))
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=(-7, 0) if right else (7, 0), textcoords="offset points",
                    ha="right" if right else "left", va="center", fontsize=FS - 1.5, color=C["ink2"])
    if preds:
        ax.plot([p["x"] for p in preds], [p["y"] for p in preds], ls="none", marker="x", ms=MS, mec=ROLE["mirror"],
                mew=0.9, zorder=5)
    allp = drawn + stars + preds
    lo_y = min(p["y"] - (p.get("sd") or 0) for p in allp)
    hi_y = max(p["y"] + (p.get("sd") or 0) for p in allp)
    pad = max(0.04 * (hi_y - lo_y), 0.15)
    ax.set_ylim(lo_y - pad, hi_y + 2.5 * pad)
    x0, x1 = x_lim
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(FixedLocator([t for t in X_TICKS if x0 <= t <= x1]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("fraction of the original's parameters (two-pass models: of its compute)")
    ax.set_ylabel("k-NN top-1 (%)")

    handles = []
    for key, (label, col, marker, filled, line) in SERIES.items():
        if any(p["series"] == key for p in points):
            st = (dict(mfc=col, mec=C["paper"], mew=0.5) if filled else dict(mfc=C["paper"], mec=col, mew=0.9))
            handles.append(Line2D([], [], color=col, lw=1.1 if line else 0, marker=marker, ms=MS, label=label, **st))
    if stars:
        handles.append(Line2D([], [], ls="none", marker="*", ms=MS + 3.0, mfc=C["paper"], mec=C["ink2"], mew=0.8,
                              label="flip-averaged, two passes\n(placed at their compute)"))
    if preds:
        handles.append(Line2D([], [], ls="none", marker="x", ms=MS, mec=ROLE["mirror"], mew=0.9,
                              label="pre-registered prediction"))
    handles += [Line2D([], [], ls="none", label=" "),
                Line2D([], [], ls="none", marker="o", ms=MS, mfc=C["ink2"], mec=C["paper"], mew=0.5,
                       label="exactly mirror-invariant\nin one forward pass"),
                Line2D([], [], ls="none", marker="o", ms=MS - 0.3, mfc=C["paper"], mec=C["ink2"], mew=0.9,
                       label="not exactly invariant")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(LEGEND_X / FIG_W, (AX[1] + AX[3]) / FIG_H),
               fontsize=FS - 0.5, handlelength=1.8, handletextpad=0.5, labelspacing=0.55, borderaxespad=0.0)
    if synthetic:
        fig.text(0.995, 0.995, "synthetic data (dry run)", ha="right", va="top", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def caption(points: list, model: str) -> str:
    n_seeds = sorted({p["n"] for p in points if p["series"] in SERIES and p["n"] > 1})
    has_stars = any(p["series"] == "flipavg" for p in points)
    has_pred = any(p["series"] == "prediction" for p in points)
    return ("Accuracy against size for " + model + ": validation k-NN top-1 (all 50,000 images) against the "
            "fraction of the original's parameters. SVD-LLM alone (open squares) against Mirror Surgery, extended "
            "below its own size by SVD-LLM inside the even and odd blocks (filled circles: exactly mirror-invariant "
            "in one forward pass); Mirror Surgery on the first 9 blocks only shows the price of exactness; the "
            "original distilled with the same recipe anchors the top. "
            + ("Stars are flip-averaged models (two forward passes), placed at their analytic compute relative to the "
               "original's single pass. " if has_stars else "")
            + ("Crosses are the pre-registered A06-2 prediction. " if has_pred else "")
            + (f"Error bars: one standard deviation over healing seeds (n = {', '.join(map(str, n_seeds))})."
               if n_seeds else "One seed per point."))


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, seed: int = 0, with_a07: bool = True, model: str = MODEL) -> tuple:
    """A synthetic run tree in the formats of scripts/fp_heal.py, mirror/final/eval/knn_fast.py and
    scripts/fp_eval.py (values from a seeded generator; they mean nothing). Returns (runs, results)."""
    g = np.random.default_rng(seed)
    runs, results = Path(work) / "runs" / "final", Path(work) / "results"
    (results / "A01").mkdir(parents=True, exist_ok=True)
    (results / "A01" / "decision.json").write_text(json.dumps({"adopted": False}), encoding="utf-8")
    G = 23.2                                                     # synthetic GFLOPs of the dense model

    def write_run(group, method, seeds, frac, base, flipavg=False):
        exact = method.startswith(("mirror_exact", "mirror_svdllm"))
        for s in seeds:
            run = runs / group / model / method / f"s{s}"
            (run / "eval").mkdir(parents=True, exist_ok=True)
            params = {"model_params_frac": frac, "model_params_structured": int(frac * 86583552)}
            (run / "metrics.json").write_text(json.dumps({"params": params, "seed": s}), encoding="utf-8")
            (run / "spec.json").write_text(json.dumps({"method": method, "fp_heal": {"params": params}}),
                                           encoding="utf-8")
            for step in (0, 100, 300, 780):
                full = step in (0, 780)
                y = base - (25.0 if step == 0 else 1.5 if step == 100 else 0.4 if step == 300 else 0.0)
                y += float(g.normal(0, 0.05))
                key = "bf16" if full else "bf16_q10000"
                rec = {"format": "fp-knn-v1", "tag": f"step{step}", "step": step, "model": model,
                       "run_dir": str(run), "original": False,
                       "passes": {key: {"top1": y, "n_query": 50000 if full else 10000, "pass": "bf16",
                                        "flip_consistency": 100.0 if exact else 97.5,
                                        "forward_passes_per_image": 1}}}
                if step == 780:
                    rec["passes"]["fp32"] = {"top1": y, "n_query": 50000,
                                             "flip_consistency": 100.0 if exact else 97.4}
                    if flipavg:
                        rec["cost"] = {"gflops_per_forward": G * frac * 1.02, "gflops_dense": G, "params_frac": frac}
                        rec["passes"]["bf16_flipavg"] = {"top1": y + 0.3, "n_query": 50000, "forward_passes_per_image": 2,
                                                         "gflops_per_image": 2 * G * frac * 1.02}
                (run / "eval" / f"step{step}.json").write_text(json.dumps(rec), encoding="utf-8")

    write_run("main", "mirror_exact_astar", (0, 1, 2), 0.7646, 78.3)
    write_run("main", "svdllm", (0, 1, 2), 0.7650, 78.0)
    write_run("main", "original", (0,), 1.0, 78.9)
    write_run("a06", "mirror_partial_k9", (0,), 0.7660, 78.5)
    for f, d_ in ((0.65, 0.9), (0.50, 2.3), (0.40, 4.0)):
        tag = f"f{int(round(f * 100)):03d}"
        seeds = (0, 1) if f == 0.50 else (0,)
        write_run("a06", f"svdllm_{tag}", seeds, f, 78.0 - d_, flipavg=with_a07 and f == 0.40)
        write_run("a06", f"mirror_svdllm_{tag}", seeds, f, 78.3 - 0.6 * d_)
    if with_a07:
        for name, top, gf, frac in (("dinov2_vits14_reg", 75.0, 6.1, None), (model, 79.0, G, 1.0)):
            ev = runs / "A07" / name / "eval"
            ev.mkdir(parents=True, exist_ok=True)
            rec = {"format": "fp-knn-v1", "tag": "original", "original": True, "model": name,
                   "cost": {"gflops_per_forward": gf, "gflops_dense": gf, "params_model": None,
                            "params_dense": None, "params_frac": frac},
                   "passes": {"bf16": {"top1": top - 0.1, "n_query": 50000, "forward_passes_per_image": 1},
                              "bf16_flipavg": {"top1": top, "n_query": 50000, "forward_passes_per_image": 2,
                                               "gflops_per_image": 2 * gf}}}
            (ev / "original.json").write_text(json.dumps(rec), encoding="utf-8")
    return runs, results


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(ROOT / "runs" / "final"), help="run root (docs/final_phase_design.md 4.5a)")
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (reads <results>/A01)")
    p.add_argument("--a07", action="append", default=None,
                   help="folder of A07 released-model evaluations (default <runs>/A07 and <runs>/a07)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--pass", dest="family", default="bf16", help="k-NN pass (bf16 or fp32)")
    p.add_argument("--step", type=int, default=None, help="healing step (default: the last with a full pass)")
    p.add_argument("--mirror-variant", default=None, help="astar or wspace (default: results/A01/decision.json)")
    p.add_argument("--compute-csv", default=None, help="CSV with columns key, compute_frac (two-pass placement)")
    p.add_argument("--with-prediction", action="store_true", help="also draw the pre-registered A06-2 prediction")
    p.add_argument("--from-values", default=None, help="re-draw from a fig_frontier.json written earlier")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--dry-run", action="store_true", help="synthetic run tree in a temporary folder")
    p.add_argument("--dry-no-a07", action="store_true", help="dry run without the optional A07 results")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    notes: list = []
    runs, results = Path(a.runs), Path(a.results)
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    if a.dry_run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_frontier_"))
        runs, results = make_dry_inputs(work, with_a07=not a.dry_no_a07, model=a.model)
        out = Path(a.out) if a.out else work / "figures"
        print(f"dry run: synthetic inputs in {work}")
    if a.from_values:
        prev = read_json(Path(a.from_values))
        if not prev or "points" not in prev:
            raise SystemExit(f"{a.from_values}: not a fig_frontier.json")
        d = {"points": prev["points"], "inputs": prev.get("inputs", {}), "ref_gflops": prev.get("ref_gflops"),
             "ref_source": prev.get("ref_source")}
        notes.append(f"re-drawn from {a.from_values} (written at git {prev.get('git')})")
    else:
        compute = {}
        if a.compute_csv:
            import pandas as pd

            cdf = pd.read_csv(a.compute_csv)
            compute = {str(r["key"]): float(r["compute_frac"]) for _, r in cdf.iterrows()}
        mv = main_variant(results, a.mirror_variant, notes)
        a07 = [Path(x) for x in a.a07] if a.a07 else [runs / "A07", runs / "a07"]
        d = collect(runs, a.model, a.family, a.step, mv, a07, compute, notes)
        if a.with_prediction:
            d["points"] += prediction(d["points"], notes)
    if not [p for p in d["points"] if p["series"] in SERIES]:
        raise SystemExit("nothing to plot: no frontier run found under " + str(runs))
    pdf, png = draw(d["points"], out, a.dry_run, notes)

    print(f"git: {git_head()}")
    print(f"runs: {rel(runs)}; model {a.model}; pass {a.family}; original's GFLOPs per pass: "
          f"{d['ref_gflops'] if d['ref_gflops'] is not None else 'not found'} ({d['ref_source'] or ''})")
    print(f"\n  {'point':34s} {'x':>7s} {'k-NN':>7s} {'sd':>6s} {'n':>2s}  x from")
    for p in sorted(d["points"], key=lambda p: (p["series"], p["x"])):
        sd = "" if p.get("sd") is None else f"{p['sd']:.3f}"
        print(f"  {p['label'][:34]:34s} {p['x']:7.4f} {p['y']:7.3f} {sd:>6s} {p['n']:2d}  {p['x_kind']}")
        for s in p.get("sources", []):
            print(f"      {s}")
    for n in notes:
        print(f"note: {n}")
    cap = caption(d["points"], a.model)
    print(f"\nCaption: {cap}")
    meta = {"figure": STEM, "git": git_head(), "argv": list(sys.argv[1:] if argv is None else argv),
            "dry_run": bool(a.dry_run), "model": a.model, "pass": a.family, "inputs": d["inputs"],
            "ref_gflops": d["ref_gflops"], "ref_source": d["ref_source"], "points": d["points"], "notes": notes,
            "caption": cap}
    mpath = out / f"{STEM}.json"
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {rel(pdf)}, {rel(png)} and {rel(mpath)}")
    return {"pdf": pdf, "png": png, "meta": mpath, "notes": notes, "points": d["points"]}


if __name__ == "__main__":
    main()

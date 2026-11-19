#!/usr/bin/env python
"""F4 (paper figure): healing curves of every method of the main comparison (A05 with the A04 rows), DINOv2-B/14 reg.

    .venv/Scripts/python.exe scripts/fp_fig_curves.py                          # runs/final and results/
    .venv/Scripts/python.exe scripts/fp_fig_curves.py --from-values paper/figures/fig_curves.json   # re-draw
    .venv/Scripts/python.exe scripts/fp_fig_curves.py --dry-run                # synthetic run tree, temp folder

Reads only these files (every plotted value comes from them; nothing is typed in by hand). Run directories follow
docs/final_phase_design.md Sections 4.4 and 4.5a: <runs>/main/<model>/<method>[_<variant>]/s<seed>/.
  <run>/eval/step<S>.json      k-NN records of mirror/final/eval/knn_fast.py, one per checkpoint step (0, 100, 300 and
                               780 by default, the hooks of mirror/final/heal/multistudent.py). Per step the bf16 pass:
                               passes.bf16 (all 50,000 validation queries) or, where only a subset was evaluated,
                               passes.bf16_q<n> (the MUST-09 subset of n queries, 10,000 at steps 100 and 300 by
                               default); its n_query is read and printed
  <results>/A01/decision.json  {"adopted": true|false}: the main Mirror arm (and H1's operator) is wspace when adopted,
                               else astar (design doc 4.5a); --mirror-variant overrides

Methods (directory under main/<model>/, legend label): mirror_exact_<v> Mirror Surgery; svdllm SVD-LLM; svdllm_full
full SVD-LLM; iosvd IO-SVD; slicegpt SliceGPT-style; svd plain SVD; flap width-pruned student; blockdrop block-dropped
student; original original, distilled; random_structure random structure; pair_random_h1 true operator, random
pairing (A03 H1). The A02 heals (mirror_exact_fit1, _fit2, _coco) are not methods and are not drawn. A method without runs is skipped with a printed note; --methods limits the list.

Panel (a): every method over all steps (the starting points are the converted models before healing). Panel (b):
steps from 100 on, the methods that end within --zoom-drop points of the best final mean (the others are named in
the panel). Lines are the means over seeds at each step; the band is the range (minimum to maximum) over seeds where
there are at least two. Steps evaluated on a query subset are marked with an asterisk on the axis and explained in
the panel, with the query counts read from the records. Filled markers: exactly mirror-invariant in one forward
pass (Mirror Surgery, random structure, true operator with random pairing); open markers: not.

Style: scripts/paper_style.py (apply(), C, ROLE). The six compression methods take the six chromatic categorical
colours in a fixed order (Mirror blue, SVD-LLM vermilion, then orange, reddish purple, sky blue, bluish green); the
dense students are dark grey (dashed), the distilled original ink, the random controls grey. Identity always also
rests on marker shape, line style and the legend. Fixed page 6.0 x 2.9 inches. Writes <out>/fig_curves.pdf, .png
(300 dpi) and fig_curves.json (git hash, inputs with sha256, every plotted value; --from-values re-draws from it).
No randomness.
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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, ROLE, apply  # noqa: E402

STEM = "fig_curves"
MODEL = "dinov2_vitb14_reg"
DASH, DOT = (0, (4, 1.6)), (0, (1.4, 1.2))


def arms(mv: str) -> list:
    """(method directory, key, legend label, colour, marker, line style, exactly invariant in one pass)."""
    return [(f"mirror_exact_{mv}", "mirror", "Mirror Surgery", ROLE["mirror"], "o", "-", True),
            ("svdllm", "svdllm", "SVD-LLM", ROLE["svdllm"], "s", "-", False),
            ("svdllm_full", "svdllm_full", "full SVD-LLM", C["sand"], "D", "-", False),
            ("iosvd", "iosvd", "IO-SVD", C["lavender"], "^", "-", False),
            ("slicegpt", "slicegpt", "SliceGPT-style", C["sky"], "p", "-", False),
            ("svd", "svd", "plain SVD", C["sage"], "v", "-", False),
            ("flap", "flap", "width-pruned student", C["ink2"], "<", DASH, False),
            ("blockdrop", "blockdrop", "block-dropped student", C["ink2"], ">", DOT, False),
            ("original", "original", "original, distilled", ROLE["original"], "h", "-", False),
            ("random_structure", "random_structure", "random structure", ROLE["random"], "X", "-", True),
            ("pair_random_h1", "pair_random", "true operator, random pairing", ROLE["random"], "P", DASH, True)]


# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.9
AX_BOTTOM, AX_TOP = 1.02, 2.62           # inches
A_AX, B_AX = (0.48, 2.30), (3.22, 2.66)  # (left, width)
FS, FS_TITLE, MS = 8, 9, 3.8


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


def seed_dirs(arm: Path) -> list:
    out = []
    if arm.is_dir():
        for p in arm.iterdir():
            m = re.fullmatch(r"s(\d+)", p.name)
            if m and p.is_dir():
                out.append((int(m.group(1)), p))
    return sorted(out)


def step_pass(rec: dict, family: str) -> tuple | None:
    """(top1, n_query, key): the pass on all queries if present, else the largest query subset (family_q<n>)."""
    passes = rec.get("passes") or {}
    if finite((passes.get(family) or {}).get("top1")) is not None:
        p = passes[family]
        return float(p["top1"]), int(p.get("n_query") or 0), family
    subs = [k for k in passes if re.fullmatch(rf"{re.escape(family)}_q(\d+)", k)
            and finite((passes[k] or {}).get("top1")) is not None]
    if not subs:
        return None
    k = max(subs, key=lambda k: int(k.rsplit("_q", 1)[1]))
    return float(passes[k]["top1"]), int(passes[k].get("n_query") or int(k.rsplit("_q", 1)[1])), k


def main_variant(results: Path, override: str | None, notes: list) -> str:
    if override:
        return override
    dec = read_json(Path(results) / "A01" / "decision.json")
    if dec is None:
        notes.append("results/A01/decision.json not found: the main Mirror arm is mirror_exact_astar (current A*)")
        return "astar"
    return "wspace" if dec.get("adopted") else "astar"


# ---------------------------------------------------------------------------------------------- data
def collect(runs: Path, model: str, family: str, mv: str, only: set | None, notes: list) -> dict:
    series, inputs = [], {}
    for method, key, label, col, marker, ls, inv in arms(mv):
        if only and key not in only and method not in only:
            continue
        arm = runs / "main" / model / method
        seeds = seed_dirs(arm)
        if not seeds:
            notes.append(f"main/{model}/{method}: no run directory: {label} skipped")
            continue
        per_step: dict = {}
        for seed, run in seeds:
            ev = run / "eval"
            for path in sorted(ev.glob("step*.json")) if ev.is_dir() else []:
                m = re.fullmatch(r"step(\d+)\.json", path.name)
                rec = read_json(path) if m else None
                got = step_pass(rec, family) if rec else None
                if got is None:
                    continue
                inputs[rel(path)] = sha256_of(path)
                per_step.setdefault(int(m.group(1)), []).append({"seed": seed, "top1": got[0], "n_query": got[1],
                                                                 "key": got[2], "source": rel(path)})
        if not per_step:
            notes.append(f"main/{model}/{method}: no {family} k-NN records: {label} skipped")
            continue
        steps = sorted(per_step)
        rows = []
        for s in steps:
            v = np.array([r["top1"] for r in per_step[s]])
            rows.append({"step": s, "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                         "n": len(v), "seeds": [r["seed"] for r in per_step[s]], "values": v.tolist(),
                         "n_query": sorted({r["n_query"] for r in per_step[s]}),
                         "keys": sorted({r["key"] for r in per_step[s]}),
                         "sources": [r["source"] for r in per_step[s]]})
        n_seeds = sorted({len(r["seeds"]) for r in rows})
        if len(n_seeds) > 1:
            notes.append(f"{label}: the number of seeds differs between steps ({n_seeds})")
        series.append({"key": key, "method": method, "label": label, "colour": col, "marker": marker,
                       "ls": ls if isinstance(ls, str) else list(ls), "invariant": inv, "rows": rows})
    return {"series": series, "inputs": inputs}


def subset_steps(series: list) -> tuple:
    """(steps evaluated on fewer queries than the largest count, {step: counts}, full count)."""
    counts = {}
    for s in series:
        for r in s["rows"]:
            counts.setdefault(r["step"], set()).update(r["n_query"])
    full = max((max(c) for c in counts.values() if c), default=0)
    sub = sorted(s for s, c in counts.items() if c and min(c) < full)
    return sub, {s: sorted(c) for s, c in counts.items()}, full


# ---------------------------------------------------------------------------------------------- drawing
def axes_at(fig, left: float, width: float):
    return fig.add_axes([left / FIG_W, AX_BOTTOM / FIG_H, width / FIG_W, (AX_TOP - AX_BOTTOM) / FIG_H])


def ls_of(s):
    return s["ls"] if isinstance(s["ls"], str) else (s["ls"][0], tuple(s["ls"][1]))


def marker_kw(s) -> dict:
    col = s["colour"]
    if s["invariant"]:
        return dict(marker=s["marker"], ms=MS, mfc=col, mec=C["paper"], mew=0.5)
    return dict(marker=s["marker"], ms=MS - 0.3, mfc=C["paper"], mec=col, mew=0.9)


def tint(col: str, w: float = 0.6) -> str:
    h = col.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(*(round(c + (255 - c) * w) for c in (r, g, b)))


def draw_series(ax, s: dict, steps_from: int, band: bool) -> None:
    rows = [r for r in s["rows"] if r["step"] >= steps_from]
    if not rows:
        return
    x = [r["step"] for r in rows]
    if band and any(r["n"] > 1 for r in rows):
        ax.fill_between(x, [r["min"] for r in rows], [r["max"] for r in rows], color=tint(s["colour"]), lw=0,
                        alpha=0.55, zorder=1)
    ax.plot(x, [r["mean"] for r in rows], color=s["colour"], lw=1.0, ls=ls_of(s), zorder=3,
            solid_joinstyle="round", dash_capstyle="round", **marker_kw(s))


def draw(d: dict, out: Path, synthetic: bool, zoom_drop: float, notes: list) -> tuple:
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "axes.labelsize": FS})
    series = d["series"]
    sub, counts, full = subset_steps(series)
    steps = sorted({r["step"] for s in series for r in s["rows"]})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_at(fig, *A_AX)
    for s in series:
        draw_series(ax, s, steps[0], band=True)
    lo = min(r["min"] for s in series for r in s["rows"])
    hi = max(r["max"] for s in series for r in s["rows"])
    ax.set_ylim(max(0.0, math.floor(lo / 10) * 10), math.ceil((hi + 0.5) / 5) * 5)
    span = steps[-1] - steps[0]
    ax.set_xlim(steps[0] - 0.03 * span, steps[-1] + 0.03 * span)
    labels = [f"{s}*" if s in sub else str(s) for s in steps]
    ax.xaxis.set_major_locator(FixedLocator(steps))
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("healing step")
    ax.set_ylabel("k-NN top-1 (%)")
    ax.set_title("(a) all steps", loc="left", fontsize=FS_TITLE, pad=6)
    if sub:
        sub_counts = sorted({c for s_ in sub for c in counts[s_] if c < full})
        txt = (f"* {', '.join(f'{c:,}' for c in sub_counts)} validation queries;\n"
               f"other steps all {full:,}")
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=FS - 1.5,
                color=C["ink2"], linespacing=1.1)

    zoom_from = steps[1] if len(steps) > 2 else None
    shown = []
    if zoom_from is not None:
        az = axes_at(fig, *B_AX)
        finals = {s["key"]: s["rows"][-1]["mean"] for s in series}
        best = max(finals.values())
        shown = [s for s in series if finals[s["key"]] >= best - zoom_drop]
        hidden = [s for s in series if s not in shown]
        for s in shown:
            draw_series(az, s, zoom_from, band=True)
        zr = [r for s in shown for r in s["rows"] if r["step"] >= zoom_from]
        zlo, zhi = min(r["min"] for r in zr), max(r["max"] for r in zr)
        pad = max(0.06 * (zhi - zlo), 0.1)
        az.set_ylim(zlo - pad, zhi + pad)
        zsteps = [s_ for s_ in steps if s_ >= zoom_from]
        zspan = max(zsteps[-1] - zsteps[0], 1)
        az.set_xlim(zsteps[0] - 0.04 * zspan, zsteps[-1] + 0.04 * zspan)
        az.xaxis.set_major_locator(FixedLocator(zsteps))
        az.set_xticklabels([f"{s_}*" if s_ in sub else str(s_) for s_ in zsteps])
        az.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        az.grid(axis="x", visible=False)
        az.set_xlabel("healing step")
        az.set_title("(b) end of healing", loc="left", fontsize=FS_TITLE, pad=6)
        if hidden:
            az.text(0.98, 0.04, "below this range at the end:\n" + "\n".join(
                f"{s['label']} ({s['rows'][-1]['mean']:.1f})" for s in hidden), transform=az.transAxes, ha="right",
                va="bottom", fontsize=FS - 1.5, color=C["ink2"], linespacing=1.1)
    handles = [Line2D([], [], color=s["colour"], lw=1.0, ls=ls_of(s), label=s["label"], **marker_kw(s))
               for s in series]
    handles += [Line2D([], [], ls="none", marker="o", ms=MS, mfc=C["ink2"], mec=C["paper"], mew=0.5,
                       label="exactly mirror-invariant in one pass"),
                Line2D([], [], ls="none", marker="o", ms=MS - 0.3, mfc=C["paper"], mec=C["ink2"], mew=0.9,
                       label="not exactly invariant")]
    if any(r["n"] > 1 for s in series for r in s["rows"]):
        handles.append(Patch(facecolor=tint(C["grey"]), edgecolor="none", label="range over healing seeds"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=FS - 1,
               handlelength=2.2, handletextpad=0.5, columnspacing=1.2, labelspacing=0.35, borderaxespad=0.1)
    if synthetic:
        fig.text(0.995, 0.995, "synthetic data (dry run)", ha="right", va="top", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png, {"subset_steps": sub, "query_counts": counts, "full_queries": full,
                      "zoom_from": zoom_from, "zoom_drop": zoom_drop, "zoom_shown": [s["key"] for s in shown]}


def caption(d: dict, info: dict, model: str) -> str:
    sub, full = info["subset_steps"], info["full_queries"]
    sub_counts = sorted({c for s in sub for c in info["query_counts"][s] if c < full})
    seeds = sorted({r["n"] for s in d["series"] for r in s["rows"]})
    return (f"Healing curves of {model}: validation k-NN top-1 against the healing step for every method of the main "
            "comparison, all healed with the identical recipe on the same batches. Step 0 is the converted model "
            "before healing. Lines are means over healing seeds"
            + (f" ({', '.join(map(str, seeds))} seeds)" if seeds else "") + ", bands the range over seeds. "
            + (f"Steps marked * are evaluated on {', '.join(f'{c:,}' for c in sub_counts)} validation images, the "
               f"others on all {full:,}. " if sub else "")
            + "Filled markers: exactly mirror-invariant in one forward pass. (b) enlarges the steps from "
            + f"{info['zoom_from']} on for the methods that end within {info['zoom_drop']:g} points of the best.")


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, seed: int = 0, model: str = MODEL, skip: tuple = ()) -> tuple:
    """A synthetic run tree in the formats of scripts/fp_heal.py and mirror/final/eval/knn_fast.py (values from a
    seeded generator; they mean nothing). Returns (runs, results)."""
    g = np.random.default_rng(seed)
    runs, results = Path(work) / "runs" / "final", Path(work) / "results"
    (results / "A01").mkdir(parents=True, exist_ok=True)
    (results / "A01" / "decision.json").write_text(json.dumps({"adopted": False}), encoding="utf-8")
    plan = {"mirror_exact_astar": (53.8, 78.3, 3), "svdllm": (75.8, 78.1, 3), "svdllm_full": (76.2, 78.0, 3),
            "iosvd": (76.0, 77.9, 3), "slicegpt": (70.0, 77.9, 2), "svd": (40.0, 77.0, 1), "flap": (60.0, 77.4, 1),
            "blockdrop": (55.0, 76.6, 1), "original": (78.9, 78.9, 1), "random_structure": (0.5, 69.9, 1),
            "pair_random_h1": (5.0, 74.0, 1)}
    for method, (start, end, n) in plan.items():
        if method in skip:
            continue
        for s in range(n):
            run = runs / "main" / model / method / f"s{s}"
            (run / "eval").mkdir(parents=True, exist_ok=True)
            for step, frac in ((0, 0.0), (100, 0.9), (300, 0.98), (780, 1.0)):
                full = step in (0, 780)
                y = start + (end - start) * frac + float(g.normal(0, 0.06))
                key = "bf16" if full else "bf16_q10000"
                rec = {"format": "fp-knn-v1", "tag": f"step{step}", "step": step, "model": model, "original": False,
                       "passes": {key: {"top1": y, "n_query": 50000 if full else 10000, "pass": "bf16"}}}
                (run / "eval" / f"step{step}.json").write_text(json.dumps(rec), encoding="utf-8")
    return runs, results


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(ROOT / "runs" / "final"), help="run root (docs/final_phase_design.md 4.5a)")
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (reads <results>/A01)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--pass", dest="family", default="bf16", help="k-NN pass (bf16 or fp32)")
    p.add_argument("--methods", default=None, help="comma list of method keys or directories (default: all)")
    p.add_argument("--mirror-variant", default=None, help="astar or wspace (default: results/A01/decision.json)")
    p.add_argument("--zoom-drop", type=float, default=3.0, help="panel (b) shows methods within this of the best")
    p.add_argument("--from-values", default=None, help="re-draw from a fig_curves.json written earlier")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--dry-run", action="store_true", help="synthetic run tree in a temporary folder")
    p.add_argument("--dry-skip", default="", help="dry run: methods left out (tests the missing-input path)")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    notes: list = []
    runs, results = Path(a.runs), Path(a.results)
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    if a.dry_run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_curves_"))
        runs, results = make_dry_inputs(work, model=a.model,
                                        skip=tuple(x for x in a.dry_skip.split(",") if x))
        out = Path(a.out) if a.out else work / "figures"
        print(f"dry run: synthetic inputs in {work}")
    if a.from_values:
        prev = read_json(Path(a.from_values))
        if not prev or "series" not in prev:
            raise SystemExit(f"{a.from_values}: not a fig_curves.json")
        d = {"series": prev["series"], "inputs": prev.get("inputs", {})}
        notes.append(f"re-drawn from {a.from_values} (written at git {prev.get('git')})")
    else:
        mv = main_variant(results, a.mirror_variant, notes)
        only = {x.strip() for x in a.methods.split(",")} if a.methods else None
        d = collect(runs, a.model, a.family, mv, only, notes)
    if not d["series"]:
        raise SystemExit("nothing to plot: no k-NN records under " + str(runs / "main" / a.model))
    pdf, png, info = draw(d, out, a.dry_run, a.zoom_drop, notes)

    print(f"git: {git_head()}")
    print(f"runs: {rel(runs)}; model {a.model}; pass {a.family}; query counts per step {info['query_counts']}")
    for s in d["series"]:
        print(f"  {s['label']} ({s['method']}):")
        for r in s["rows"]:
            vals = ", ".join(f"s{sd} {v:.2f}" for sd, v in zip(r["seeds"], r["values"]))
            print(f"    step {r['step']:>4}: mean {r['mean']:.3f} (range {r['min']:.3f} to {r['max']:.3f}; {vals}; "
                  f"{'/'.join(r['keys'])}, {r['n_query']} queries)")
    for n in notes:
        print(f"note: {n}")
    cap = caption(d, info, a.model)
    print(f"\nCaption: {cap}")
    meta = {"figure": STEM, "git": git_head(), "argv": list(sys.argv[1:] if argv is None else argv),
            "dry_run": bool(a.dry_run), "model": a.model, "pass": a.family, "inputs": d["inputs"],
            "series": d["series"], "info": {k: v for k, v in info.items() if k != "query_counts"},
            "query_counts": {str(k): v for k, v in info["query_counts"].items()}, "notes": notes, "caption": cap}
    mpath = out / f"{STEM}.json"
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {rel(pdf)}, {rel(png)} and {rel(mpath)}")
    return {"pdf": pdf, "png": png, "meta": mpath, "notes": notes, "series": d["series"], "info": info}


if __name__ == "__main__":
    main()

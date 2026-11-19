#!/usr/bin/env python
"""F2 (paper figure): stability of the operator and the twins across five independent fits (A02).

    .venv/Scripts/python.exe scripts/fp_fig_stability.py                     # reads results/A02
    .venv/Scripts/python.exe scripts/fp_fig_stability.py --dry-run           # synthetic A02 files, temporary folder

Reads only these files (every plotted value comes from them; nothing is typed in by hand), all written by
scripts/fp_a02_stability.py (definitions in mirror/stability.py):
  <results>/A02/spectra.npz    names (fits with a sidecar) and S_eig [fits, d]: the ascending eigenvalues of the span
                               matrix S = sym(sum over stream layers of Pc C_l Pc / tr(G_l)), whose signs give the
                               stream operator A*_L (panel a)
  <results>/A02/pairs.csv      one row per pair of fits (a, b): weighted_agreement (energy-weighted agreement of the
                               odd subspaces, the mean of rho(a->b) and rho(b->a); rule A02-2) and twin_mean (share of
                               the strong twin pairs of blocks 1 to 6 that the other fit also pairs, mean of both
                               directions; rule A02-3), group_a, group_b (panels b and c)
  <results>/A02/stability.csv  optional: d_minus and group per fit (the d- of A*_L printed in panel a); without it d-
                               is the number of negative eigenvalues of S (the same number when the sidecar's sign rule
                               reproduces A*_L, which fp_a02_stability.py checks)
  <results>/A02/metrics.json   optional: the A02-2 and A02-3 verdicts and bars (gates.A02-2.threshold ...)

Panel (a): the spectrum of S for every fit, split by sign: even directions (positive eigenvalues) and odd
directions (negative eigenvalues), each sorted by magnitude, |eigenvalue| against its rank on log axes. Values below
1e-5 of the largest magnitude are not drawn (their number is printed). Panels (b) and (c): heat maps of the
symmetric agreement statistics between every pair of fits (lower triangle; the diagonal is a fit with itself and is
left out), the value printed in each cell. One sequential blue scale for both maps. Cells of two ImageNet fits that
fall below the pre-registered bar (0.8 for odd subspaces, 0.70 for twins; read from metrics.json, with the
PREREG_FINAL_PHASE.md values as fallback) are outlined in vermilion; pairs with the COCO fit are not part of the
rule and are never outlined.

Style: scripts/paper_style.py (apply(), C). Fits take the categorical colours in a fixed order (fit sets 0 to 3: blue,
vermilion, orange, reddish purple; COCO: sky blue, dashed), so identity also rests on the legend and the line style.
Fixed page of 6.0 x 2.05 inches. Writes <out>/fig_stability.pdf, .png (300 dpi) and fig_stability.json (git hash,
inputs with sha256, plotted values, notes), prints every plotted value and a caption. No randomness.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C, apply  # noqa: E402

STEM = "fig_stability"
BARS = {"A02-2": 0.8, "A02-3": 0.70}     # fallback only: PREREG_FINAL_PHASE.md Section 1 (metrics.json wins)
FIT_ORDER = ("fit0", "fit1", "fit2", "fit3", "coco")
FIT_LABEL = {"fit0": "fit set 0", "fit1": "fit set 1", "fit2": "fit set 2", "fit3": "fit set 3", "coco": "COCO"}
FIT_TICK = {"fit0": "0", "fit1": "1", "fit2": "2", "fit3": "3", "coco": "COCO"}
CATEGORICAL = (C["blue"], C["terracotta"], C["sand"], C["lavender"], C["sky"], C["sage"])   # fixed order
FLOOR_REL = 1e-5
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#EEF4F9", C["blue"], "#00304D"])

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 2.05
AX_BOTTOM, AX_H = 0.45, 1.25             # inches
EVEN_AX, ODD_AX = (0.50, 1.10), (1.67, 1.10)
B_AX, C_AX = (3.22, 1.15), (4.45, 1.15)
CB_AX = (5.68, 0.07)
FS, FS_TITLE = 8, 9


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


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def fit_label(name: str) -> str:
    return FIT_LABEL.get(name, name)


def fit_style(names: list) -> dict:
    """Colour and line style per fit, in the fixed categorical order of the fits as listed."""
    out = {}
    for i, n in enumerate(names):
        out[n] = {"color": CATEGORICAL[i % len(CATEGORICAL)], "ls": (0, (3, 1.5)) if n == "coco" else "-"}
    return out


# ---------------------------------------------------------------------------------------------- data
def load(results: Path, notes: list) -> dict:
    a02 = Path(results) / "A02"
    spath, ppath, fpath, mpath = (a02 / "spectra.npz", a02 / "pairs.csv", a02 / "stability.csv",
                                  a02 / "metrics.json")
    for p in (spath, ppath):
        if not p.exists():
            raise SystemExit(f"{p} not found (run scripts/fp_a02_stability.py first, or use --dry-run)")
    with np.load(spath) as z:
        names = [str(x) for x in np.asarray(z["names"]).tolist()]
        S_eig = np.asarray(z["S_eig"], dtype=np.float64)
    if S_eig.ndim != 2 or S_eig.shape[0] != len(names):
        raise SystemExit(f"{spath}: S_eig has shape {S_eig.shape} for {len(names)} names")
    pairs = pd.read_csv(ppath)
    fits = pd.read_csv(fpath) if fpath.exists() else None
    if fits is None:
        notes.append(f"{rel(fpath)} not found: d- counted from the eigenvalues of S")
    metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    bars, verdicts = {}, {}
    for gate in BARS:
        g = (metrics.get("gates") or {}).get(gate) or {}
        bars[gate] = finite(g.get("threshold")) if finite(g.get("threshold")) is not None else BARS[gate]
        verdicts[gate] = None if not g else ("not decided" if g.get("pass") is None else
                                             ("PASS" if g["pass"] else "FAIL"))
        if finite(g.get("threshold")) is None:
            notes.append(f"{gate} bar not in metrics.json: fallback {BARS[gate]} (PREREG_FINAL_PHASE.md)")

    # every fit named in the spectra or the pairs, in the canonical order first
    seen = list(dict.fromkeys(names + [str(x) for x in pairs["a"]] + [str(x) for x in pairs["b"]]))
    order = [n for n in FIT_ORDER if n in seen] + [n for n in seen if n not in FIT_ORDER]
    group = {}
    if fits is not None and "group" in fits.columns:
        group = {str(r["name"]): str(r["group"]) for _, r in fits.iterrows()}
    for _, r in pairs.iterrows():
        group.setdefault(str(r["a"]), str(r.get("group_a", "")))
        group.setdefault(str(r["b"]), str(r.get("group_b", "")))
    spectra = {}
    for i, n in enumerate(names):
        lam = S_eig[i]
        dm = None
        if fits is not None and "d_minus" in fits.columns:
            sub = fits[fits["name"].astype(str) == n]
            dm = int(sub["d_minus"].iloc[0]) if len(sub) and finite(sub["d_minus"].iloc[0]) is not None else None
        spectra[n] = {"even": np.sort(lam[lam > 0])[::-1], "odd": np.sort(-lam[lam < 0])[::-1],
                      "d_minus": dm if dm is not None else int((lam < 0).sum()), "n_neg": int((lam < 0).sum()),
                      "n_pos": int((lam > 0).sum()), "max_abs": float(np.abs(lam).max())}
    missing = [n for n in order if n not in spectra]
    if missing:
        notes.append(f"no spectrum (no sidecar) for {missing}: left out of panel (a)")
    mats = {}
    for key in ("weighted_agreement", "twin_mean"):
        M = np.full((len(order), len(order)), np.nan)
        if key not in pairs.columns:
            notes.append(f"pairs.csv has no {key}")
        else:
            for _, r in pairs.iterrows():
                i, j = order.index(str(r["a"])), order.index(str(r["b"]))
                v = finite(r[key])
                if v is not None:
                    M[max(i, j), min(i, j)] = v
        mats[key] = M
    inputs = {rel(p): sha256_of(p) for p in (spath, ppath, fpath, mpath) if p.exists()}
    return {"order": order, "spectra": spectra, "mats": mats, "group": group, "bars": bars, "verdicts": verdicts,
            "model": metrics.get("model"), "inputs": inputs}


# ---------------------------------------------------------------------------------------------- drawing
def axes_at(fig, left: float, width: float, height: float = AX_H, bottom: float = AX_BOTTOM):
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def panel_spectra(fig, d: dict, style: dict, notes: list) -> tuple:
    ax_e, ax_o = axes_at(fig, *EVEN_AX), axes_at(fig, *ODD_AX)
    top = max(s["max_abs"] for s in d["spectra"].values())
    floor = FLOOR_REL * top
    max_rank = 1
    for n in d["order"]:
        if n not in d["spectra"]:
            continue
        s = d["spectra"][n]
        for ax, v in ((ax_e, s["even"]), (ax_o, s["odd"])):
            keep = v[v >= floor]
            if len(keep) < len(v):
                notes.append(f"{n}: {len(v) - len(keep)} {'even' if ax is ax_e else 'odd'} eigenvalues below "
                             f"{FLOOR_REL:g} of the largest are not drawn")
            if len(keep):
                ax.plot(np.arange(1, len(keep) + 1), keep, color=style[n]["color"], ls=style[n]["ls"], lw=0.9,
                        alpha=0.9, solid_joinstyle="round")
                max_rank = max(max_rank, len(keep))
    for ax, title in ((ax_e, "even ($\\lambda > 0$)"), (ax_o, "odd ($\\lambda < 0$)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, max_rank * 1.25)
        ax.set_ylim(floor, top * 2.0)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("rank")
        ax.text(0.97, 0.97, title, transform=ax.transAxes, ha="right", va="top", fontsize=FS, color=C["ink"])
    ax_o.tick_params(axis="y", labelleft=False)
    ax_e.set_ylabel("|eigenvalue| of $S$")
    ax_e.set_title("(a) spectrum of the span matrix", loc="left", fontsize=FS_TITLE, pad=6)
    dms = [d["spectra"][n]["d_minus"] for n in d["order"] if n in d["spectra"]]
    if dms:
        txt = f"$d_-$ = {min(dms)}" if min(dms) == max(dms) else f"$d_-$ = {min(dms)} to {max(dms)}"
        ax_o.text(0.04, 0.04, txt, transform=ax_o.transAxes, ha="left", va="bottom", fontsize=FS - 1,
                  color=C["ink2"])
    handles = [Line2D([], [], color=style[n]["color"], ls=style[n]["ls"], lw=1.0, label=fit_label(n))
               for n in d["order"] if n in d["spectra"]]
    ax_e.legend(handles=handles, loc="lower left", fontsize=FS - 1.5, handlelength=1.4, handletextpad=0.4,
                borderaxespad=0.2, labelspacing=0.15)
    return ax_e, ax_o


def lum(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def panel_map(fig, box, M: np.ndarray, d: dict, title: str, bar: float, norm, row_labels: bool) -> None:
    ax = axes_at(fig, box[0], box[1], height=box[1], bottom=AX_BOTTOM + AX_H - box[1])   # tops aligned with (a)
    order = d["order"]
    n = len(order)
    for i in range(n):
        for j in range(i):
            v = M[i, j]
            if math.isfinite(v):
                col = SEQ(norm(v))
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=col, edgecolor=C["paper"], lw=1.2))
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=FS - 1.5,
                        color=C["paper"] if lum(col) < 0.45 else C["ink"])
                imagenet = d["group"].get(order[i]) == "imagenet" and d["group"].get(order[j]) == "imagenet"
                if imagenet and v < bar - 1e-12:
                    ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False,
                                           edgecolor=C["terracotta"], lw=1.1))
            else:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=C["grid"], edgecolor=C["paper"], lw=1.2))
                ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=FS - 2, color=C["ink2"])
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n, 1)
    ax.set_xticks(np.arange(n - 1) + 0.5, [FIT_TICK.get(x, x) for x in order[:-1]])
    ax.set_yticks(np.arange(1, n) + 0.5, [FIT_TICK.get(x, x) for x in order[1:]] if row_labels else [])
    ax.tick_params(length=0, pad=2, labelsize=FS - 1)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=FS_TITLE, pad=6)
    ax.text(0.98, 0.97, f"bar {bar:g}", transform=ax.transAxes, ha="right", va="top", fontsize=FS - 1,
            color=C["ink2"])
    ax.set_xlabel("fit set", labelpad=2)


def draw(d: dict, out: Path, synthetic: bool, notes: list) -> tuple:
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
                         "axes.labelsize": FS})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    style = fit_style([n for n in d["order"]])
    panel_spectra(fig, d, style, notes)
    vals = np.concatenate([M[np.isfinite(M)] for M in d["mats"].values()] + [np.array(list(d["bars"].values()))])
    vmin = math.floor(20 * (float(vals.min()) - 0.05)) / 20 if len(vals) else 0.5     # data-driven, 0.05 steps
    norm = Normalize(vmin=vmin, vmax=1.0)
    panel_map(fig, B_AX, d["mats"]["weighted_agreement"], d, "(b) odd subspaces", d["bars"]["A02-2"], norm, True)
    panel_map(fig, C_AX, d["mats"]["twin_mean"], d, "(c) twins", d["bars"]["A02-3"], norm, False)
    cax = axes_at(fig, CB_AX[0], CB_AX[1], height=B_AX[1], bottom=AX_BOTTOM + AX_H - B_AX[1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS - 1.5, length=2, width=0.5)
    if synthetic:
        fig.text(0.995, 0.995, "synthetic data (dry run)", ha="right", va="top", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def caption(d: dict) -> str:
    present = [n for n in d["order"] if n in d["spectra"]]
    dms = ", ".join(f"{fit_label(n)} {d['spectra'][n]['d_minus']}" for n in present)
    v = d["verdicts"]
    return (f"Stability of the operator and the twins across {len(d['order'])} independent fits"
            f"{' of ' + d['model'] if d.get('model') else ''} (disjoint ImageNet-train fit sets and a COCO fit). "
            "(a) Eigenvalues of the span matrix S, whose signs give the stream operator, sorted by magnitude: positive "
            f"eigenvalues are even directions, negative ones odd directions ($d_-$: {dms}). (b) Energy-weighted "
            "agreement of the odd subspaces and (c) share of the strong twin pairs in blocks 1 to 6 that the other fit "
            "also pairs, each the mean of both directions. Outlined cells fall below the pre-registered bar "
            f"({d['bars']['A02-2']:g} and {d['bars']['A02-3']:g}; the rules cover pairs of ImageNet fits only). "
            f"Rule A02-2: {v.get('A02-2') or 'not in metrics.json'}; rule A02-3: {v.get('A02-3') or 'not in metrics.json'}.")


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, seed: int = 0, d: int = 96) -> Path:
    """Synthetic results/A02 files in the formats of scripts/fp_a02_stability.py (values from a seeded generator;
    they mean nothing)."""
    g = np.random.default_rng(seed)
    res = Path(work) / "results"
    a02 = res / "A02"
    a02.mkdir(parents=True, exist_ok=True)
    names = list(FIT_ORDER)
    base = 0.6 * np.arange(1, d + 1, dtype=np.float64) ** -1.4
    sign = np.where(np.arange(d) % 9 < 4, -1.0, 1.0)
    S = []
    for i, n in enumerate(names):
        lam = sign * base * np.exp(g.normal(0, 0.05, d))
        lam[-1] = 0.0                                                   # the centred null direction
        S.append(np.sort(lam))
    S = np.stack(S)
    L = 4
    np.savez(a02 / "spectra.npz", names=np.array(names), S_eig=S, layer_eig=np.repeat(S[:, None, :], L + 1, axis=1),
             d_minus_layers=np.full((len(names), L + 1), int((S[0] < 0).sum())),
             energies=np.ones((len(names), L + 1)))
    rows = []
    for i, n in enumerate(names):
        rows.append({"name": n, "group": "imagenet" if n != "coco" else "coco", "fit_seed": i,
                     "d_plus": int((S[i] >= 0).sum()), "d_minus": int((S[i] < 0).sum()),
                     "n_pairs_total": 100, "n_strong_blocks_1_6": 40, "twin_fraction_blocks_1_6": 0.3})
    pd.DataFrame(rows).to_csv(a02 / "stability.csv", index=False)
    prs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ga, gb = rows[i]["group"], rows[j]["group"]
            w = float(g.uniform(0.82, 0.97)) - (0.05 if "coco" in (names[i], names[j]) else 0.0)
            t = float(g.uniform(0.72, 0.9)) if (names[i], names[j]) != ("fit2", "fit3") else 0.66   # one below
            prs.append({"a": names[i], "b": names[j], "group_a": ga, "group_b": gb,
                        "both_imagenet": ga == gb == "imagenet", "operator_distance": float(g.uniform(0.01, 0.05)),
                        "rho_ab": w + 0.01, "rho_ba": w - 0.01, "weighted_agreement": w,
                        "unweighted_cos2": w - 0.1, "twin_a_to_b": t + 0.02, "twin_b_to_a": t - 0.02,
                        "twin_mean": t, "twin_jaccard": t - 0.2, "head_share_same": 0.95})
    pd.DataFrame(prs).to_csv(a02 / "pairs.csv", index=False)
    metrics = {"experiment": "A02", "model": "synthetic", "dry_run": True,
               "gates": {"A02-2": {"pass": True, "threshold": BARS["A02-2"]},
                         "A02-3": {"pass": False, "threshold": BARS["A02-3"]}}}
    (a02 / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (reads <results>/A02)")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--dry-run", action="store_true", help="synthetic A02 files in a temporary folder")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    notes: list = []
    results = Path(a.results)
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    if a.dry_run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_stability_"))
        results = make_dry_inputs(work)
        out = Path(a.out) if a.out else work / "figures"
        print(f"dry run: synthetic inputs in {work}")
    d = load(results, notes)
    pdf, png = draw(d, out, a.dry_run, notes)

    print(f"git: {git_head()}")
    for path, sha in d["inputs"].items():
        print(f"source: {path} (sha256 {str(sha)[:12]})")
    print("\npanel (a): span-matrix spectrum per fit")
    for n in d["order"]:
        if n in d["spectra"]:
            s = d["spectra"][n]
            print(f"  {n:6s} d+ {s['n_pos']:4d}  d- {s['d_minus']:4d} (negative eigenvalues {s['n_neg']})  "
                  f"largest |eigenvalue| {s['max_abs']:.4g}  top even {s['even'][:3].round(5).tolist()}  "
                  f"top odd {s['odd'][:3].round(5).tolist()}")
    for key, title in (("weighted_agreement", "panel (b): odd-subspace agreement (A02-2)"),
                       ("twin_mean", "panel (c): twin agreement (A02-3)")):
        print(f"\n{title}")
        M = d["mats"][key]
        for i, a_ in enumerate(d["order"]):
            for j in range(i):
                v = M[i, j]
                print(f"  {d['order'][j]:6s} - {a_:6s}: {'n/a' if not math.isfinite(v) else f'{v:.4f}'}")
    for n in notes:
        print(f"note: {n}")
    cap = caption(d)
    print(f"\nCaption: {cap}")
    vals = {"spectra": {n: {"d_minus": s["d_minus"], "n_pos": s["n_pos"], "n_neg": s["n_neg"],
                            "max_abs": s["max_abs"]} for n, s in d["spectra"].items()},
            "order": d["order"], "group": d["group"],
            "weighted_agreement": np.where(np.isfinite(d["mats"]["weighted_agreement"]),
                                           d["mats"]["weighted_agreement"], None).tolist(),
            "twin_mean": np.where(np.isfinite(d["mats"]["twin_mean"]), d["mats"]["twin_mean"], None).tolist()}
    meta = {"figure": STEM, "git": git_head(), "argv": list(sys.argv[1:] if argv is None else argv),
            "dry_run": bool(a.dry_run), "inputs": d["inputs"], "bars": d["bars"], "verdicts": d["verdicts"],
            "values": vals, "floor_rel": FLOOR_REL, "notes": notes, "caption": cap}
    mpath = out / f"{STEM}.json"
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {rel(pdf)}, {rel(png)} and {rel(mpath)}")
    return {"pdf": pdf, "png": png, "meta": mpath, "notes": notes, "data": d}


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Paper data package: collect every result file of the repo into tables, figures, INDEX.md and
numbers.json (PAPER_EXPERIMENTS.md Stage 4).

CPU only. Reads the small CSV and JSON files under runs/ and results/ and, when present, operator
npz files under artifacts/a100/ (for scripts/f1_tax.py and scripts/f2_drift.py, which are imported
and called, never edited). Nothing is typed by hand: every number in the outputs is read from a
file written by a script of this repo, and every number cited in INDEX.md is recorded in
numbers.json with its source file, seed and git hash (and, for numbers read directly, a locator
that tests/test_collect_results.py uses to re-read the source). Missing inputs are marked
"not run" (or "pending" for a check whose runs have not all finished), so the script can be rerun
whenever new results arrive:

    python scripts/collect_results.py                          # writes results/paper/
    python scripts/collect_results.py --out <dir> --skip-scripts

Outputs under --out:
  tables/      T_main_zeroshot, T_healing (with T_healing_vexact), T_e3 (E3 arms; T_e3_checks: E3.a, E3.b,
               seed condition, gate G5, validity, learning-rate check), T_strong_baselines, T_flipavg,
               T_generality, T_structure_per_layer, T_prereg, T_x_checks, T_e1_history, T_attention,
               T_ablations, T_tier2 (with T_tier2_transfer, T_tier2_seg and T_tier2_seg_rerun; per-class
               IoU in T_tier2_seg_classes.csv), T_latency (with T_latency_5090 and T_latency_devices for
               the second device), T_latency_profile, T_f4, T_f7 (each as .csv and .md; some .md files
               hold several related CSVs), plus the outputs of f1_tax.py and f2_drift.py (prefix ext_);
  figures/     PNG figures regenerated from the tables (and the f1/f2 figures, prefix ext_);
  scripts/     raw output folders of f1_tax.py and f2_drift.py;
  INDEX.md     one section per experiment ID; numbers.json (flat); collector_meta.json.

The analytic cost model (parameters and multiply-adds at 224 px) is `analytic_cost` below; its
assumptions are written to tables/cost_model.md. These are analytic counts, not measured latency.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mirror.findings import tax_formula  # noqa: E402
from mirror.report import C_AQUA, C_BLUE, C_GRID, C_INK, C_MUTED, C_NULL, C_ORANGE, _style, decide  # noqa: E402
from mirror.utils import git_info, load_config  # noqa: E402

NOT_RUN, PENDING = "not run", "pending"
PASS, FAIL = "PASS", "FAIL"

# Models: display names, fixed colours (colour follows the model) and markers. Slots 1 to 7 of the
# reference categorical palette (light surface); slots 1 to 3 are the ones in mirror/report.py.
C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET = "#eda100", "#e87ba4", "#008300", "#6250d6"
MODEL_LABEL = {"dinov2_vitb14_reg": "DINOv2-reg", "dinov2_vitb14": "DINOv2", "deit3_b16": "DeiT-III",
               "mae_b16": "MAE", "siglip2_b16": "SigLIP2", "dinov2_vits14_reg": "DINOv2-S/14 reg",
               "dinov2_vitl14_reg": "DINOv2-L/14 reg"}
MODEL_SHORT = {"dinov2_vitb14_reg": "reg", "dinov2_vitb14": "plain", "deit3_b16": "deit3", "mae_b16": "mae",
               "siglip2_b16": "siglip2", "dinov2_vits14_reg": "reg_s", "dinov2_vitl14_reg": "reg_l"}
MODEL_COLOUR = {"dinov2_vitb14_reg": C_BLUE, "dinov2_vitb14": C_ORANGE, "deit3_b16": C_AQUA, "mae_b16": C_YELLOW,
                "siglip2_b16": C_MAGENTA, "dinov2_vits14_reg": C_GREEN, "dinov2_vitl14_reg": C_VIOLET}
MODEL_MARKER = {"dinov2_vitb14_reg": "o", "dinov2_vitb14": "s", "deit3_b16": "^", "mae_b16": "D", "siglip2_b16": "v",
                "dinov2_vits14_reg": "P", "dinov2_vitl14_reg": "X"}
MODEL_ORDER = list(MODEL_LABEL)
MAIN_MODELS = ["dinov2_vitb14_reg", "dinov2_vitb14"]
# G models with their own run folders under runs/g/ (generality); the DINOv2 size pair (S and L, the plan's
# nice-to-have) is kept apart from DeiT-III, MAE and SigLIP2 in the F1 and F2 groups (f2_drift.py draws at
# most three models per figure). DINOv2-L has 24 blocks, so its L/2 is k = 12.
G_MODELS = ["deit3_b16", "mae_b16", "siglip2_b16", "dinov2_vits14_reg", "dinov2_vitl14_reg"]
G_SIZE_PAIR = {"dinov2_vits14_reg": "dinov2_s_reg", "dinov2_vitl14_reg": "dinov2_l_reg"}   # model -> runs/g arm
MAIN_KS = [3, 6, 9, 12]
BG = "#fcfcfb"

# One display label per method, used in every table and in INDEX.md (the producing scripts record their own wording,
# kept in label_recorded where a table carries it). The MLP-only Mirror variant uses per-layer operators, so it is
# the V-local design applied to the MLPs alone.
LABEL_MIRROR_MLP = "Mirror V-local, MLP only"
LABEL_VLOCAL = "Mirror V-local, MLP + attention"
LABEL_VEXACT = "Mirror V-exact, MLP + attention"
LABEL_SVD_MLP = "SVD at matched parameters, MLP only"
LABEL_SVD_ATTN = "SVD at matched parameters, MLP + attention"
LABEL_RANDOM_MLP = "Random structure, MLP only"
LABEL_RANDOM_ATTN = "Random structure, MLP + attention"
METHOD_LABEL = {   # (family, method) of scripts/tier2_*.py rows -> display label
    ("original", "original"): "Original model",
    ("mlp", "mirror"): LABEL_MIRROR_MLP, ("mlp", "svd"): LABEL_SVD_MLP, ("mlp", "random"): LABEL_RANDOM_MLP,
    ("m1", "mlp_only"): LABEL_MIRROR_MLP, ("m1", "vlocal"): LABEL_VLOCAL, ("m1", "vexact"): LABEL_VEXACT,
    ("m1", "svd"): LABEL_SVD_ATTN, ("m1", "random"): LABEL_RANDOM_ATTN,
    ("healed", "mirror"): LABEL_VLOCAL + ", healed", ("healed", "svd"): LABEL_SVD_ATTN + ", healed",
    ("healed", "random"): LABEL_RANDOM_ATTN + ", healed", ("healed_vexact", "vexact"): LABEL_VEXACT + ", healed",
}
M2_LABEL = {"original": "Original model (hub)", "dense": "Dense reference (folded original, same implementation)",
            "vlocal": LABEL_VLOCAL, "vexact": LABEL_VEXACT, "mlp_only": LABEL_MIRROR_MLP, "svd": LABEL_SVD_ATTN,
            "random": LABEL_RANDOM_ATTN, "prune": "MLP width pruning at matched parameters",
            "vlocal_r2": LABEL_VLOCAL + " (Wo structured, second rotation)"}


def method_label(family, method, recorded=None) -> str:
    """Display label of a Tier-2 row from its (family, method); the recorded label when the pair is unknown."""
    return METHOD_LABEL.get((str(family), str(method)), recorded if isinstance(recorded, str) else f"{family} {method}")


# Zero-shot variants of T_main_zeroshot: key, label, budget, experiment, variant name in the source CSV.
VARIANTS = [
    ("mirror_mlp", LABEL_MIRROR_MLP, "MLP only", "X2", "surgery"),
    ("svd_mlp", LABEL_SVD_MLP, "MLP only", "X2", "svd"),
    ("random_mlp", LABEL_RANDOM_MLP, "MLP only", "X2", "random"),
    ("mirror_mlp_m1", LABEL_MIRROR_MLP, "MLP only", "M1", "mlp_only"),
    ("vlocal", LABEL_VLOCAL, "MLP + attention", "M1", "vlocal"),
    ("vexact", LABEL_VEXACT, "MLP + attention", "M1", "vexact"),
    ("svd_attn", LABEL_SVD_ATTN, "MLP + attention", "M1", "svd"),
    ("random_attn", LABEL_RANDOM_ATTN, "MLP + attention", "M1", "random"),
]
VARIANT_INFO = {v[0]: v for v in VARIANTS}

# Canonical source of every MLP-only number (the paper's zero-shot headline is the k = 6 row): the main-protocol
# runs X2 (scripts/run_e1.py), which project with the operators as fitted, in float64 memory. Every other script
# (stage1_ablate.py for A1 to A5, m1_attention.py, tier2_linear.py and the healing scripts) reloads the operators
# from e1_operators.npz, where they are stored in float32 and cleaned back to exact involutions (a change of about
# 3e-8). Same pairing, same parameter count, same evaluation code; the projected weights differ at the 1e-8 level,
# and that moves a handful of the 50,000 bf16 k-NN queries (0.02 to 0.07 points). Rows of the same setting from
# those scripts are kept, marked, and compared with the X2 value of the same fit seed.
CANONICAL_RUN = "X2"
CANONICAL_KEYS = {"mirror": "mirror_mlp", "svd": "svd_mlp", "random": "random_mlp"}   # method -> T_main_zeroshot key
# Ablation cells that are the canonical setting itself (per-layer operators, mirror pairing, blocks 1..k, alpha 1,
# the run's own 2,000 fit images); A3 rows qualify when their fit set is the default one (checked per row).
SAME_AS_CANONICAL = {("A1", "per_layer"): "mirror", ("A2", "mirror"): "mirror", ("A4", "first"): "mirror",
                     ("A5", "alpha1.00"): "mirror", ("A1", "svd"): "svd"}
OPERATOR_ROUNDTRIP = ("projects with the operators reloaded from the float32 operator file (cleaned to exact "
                      "involutions), whereas X2 projects with the float64 operators as fitted; same pairing, parameter "
                      "count and evaluation code, so the difference is numerical")
PARAM_MATCH_TOL = 0.01      # the plan's matched-budget rule: within 1%
BASELINES = {"mirror_mlp": ("svd_mlp", "random_mlp"), "vlocal": ("svd_attn", "random_attn"),
             "vexact": ("svd_attn", "random_attn")}
ROTATING = {"mirror_mlp", "random_mlp", "mirror_mlp_m1", "vlocal", "random_attn"}   # per-layer operators


def log(msg: str) -> None:
    print(f"[collect {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================
# Small helpers
# ================================================================================================
def py(v):
    """numpy scalar -> python; NaN -> None."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if math.isnan(f) else f
    return v


def isnum(v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            and math.isfinite(float(v)))


def fmt(v, spec: str = "") -> str:
    v = py(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if spec:
        if spec.endswith("d") and isinstance(v, float) and v.is_integer():
            v = int(v)
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def pm(mean, std, n=None, spec=".2f") -> str:
    if py(mean) is None:
        return ""
    s = fmt(mean, spec)
    if py(std) is not None:
        s += f" ± {fmt(std, spec)}"
    return s


def short(h) -> str:
    return str(h)[:10] if h else "not recorded"


def agg(vals):
    vals = [float(v) for v in vals if isnum(v)]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def read_json(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log(f"WARNING cannot read {p}: {e}")
        return None


def read_csv(p):
    p = Path(p)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        log(f"WARNING cannot read {p}: {e}")
        return None
    return df if len(df) else None


def write_text(path: Path, lines) -> None:
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    if "—" in text:
        raise ValueError(f"em-dash in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def csv_loc(file: str, where: dict, column: str) -> dict:
    return {"type": "csv", "file": file, "where": where, "column": column}


def json_loc(file: str, path: list) -> dict:
    return {"type": "json", "file": file, "path": path}


def resolve_locator(root, loc: dict):
    """Re-read the value a locator points at (used by the tests)."""
    p = Path(loc["file"])
    p = p if p.is_absolute() else Path(root) / p
    if loc["type"] == "csv":
        df = pd.read_csv(p)
        sel = df
        for c, v in loc["where"].items():
            sel = sel[sel[c] == v]
        if len(sel) != 1:
            raise KeyError(f"{loc}: {len(sel)} matching rows")
        return py(sel.iloc[0][loc["column"]])
    if loc["type"] == "json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        for k in loc["path"]:
            obj = obj[k]
        return py(obj)
    raise ValueError(f"unknown locator type {loc['type']}")


# ================================================================================================
# Paths, tables and the number registry
# ================================================================================================
class Paths:
    def __init__(self, root, artifacts=None):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs"
        # M2: the clean (uncontended) timing run supersedes the first, contended one in runs/m2
        self.m2 = "m2_clean" if (self.runs / "m2_clean" / "m2_latency.csv").is_file() else "m2"
        self.results = self.root / "results"
        self.artifacts = Path(artifacts).resolve() if artifacts else self.root / "artifacts" / "a100"
        self.configs = self.root / "configs"
        self.jobs = self.root / "jobs"

    def rel(self, p) -> str:
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return p.resolve().as_posix()


class Table:
    """Rows are dicts. Keys starting with "_" are provenance and are not written:
    _src (source file or files), _seed (run seed), _fit_seed, _git (git hash of the run that wrote
    the source), _loc {column: locator of a value read directly}, _derived {column: how it was
    computed}."""

    def __init__(self, name, title, key_cols, md_cols, notes=()):
        self.name, self.title, self.key_cols, self.md_cols = name, title, list(key_cols), list(md_cols)
        self.notes = list(notes)
        self.rows: list[dict] = []

    def add(self, row: dict) -> dict:
        row = dict(row)
        row.setdefault("_loc", {})
        row.setdefault("_derived", {})
        row["_key"] = ", ".join(f"{c}={row.get(c)}" for c in self.key_cols)
        self.rows.append(row)
        return row

    def find(self, **where):
        hits = [r for r in self.rows if all(r.get(k) == v for k, v in where.items())]
        if len(hits) > 1:
            raise ValueError(f"{self.name}: {len(hits)} rows match {where}")
        return hits[0] if hits else None

    def public(self) -> pd.DataFrame:
        cols = []
        for r in self.rows:
            for c in r:
                if not c.startswith("_") and c not in cols:
                    cols.append(c)
        return pd.DataFrame([{c: py(r.get(c)) for c in cols} for r in self.rows], columns=cols)

    def md_lines(self, rows=None, heading_level="##") -> list[str]:
        rows = self.rows if rows is None else rows
        out = [f"{heading_level} {self.title}", ""]
        out += [n for n in self.notes] + ([""] if self.notes else [])
        if not rows:
            return out + ["No rows (inputs not run).", ""]
        out.append("| " + " | ".join(h for h, _, _ in self.md_cols) + " |")
        out.append("|" + "|".join("---" for _ in self.md_cols) + "|")
        for r in rows:
            cells = []
            for _, col, spec in self.md_cols:
                v = col(r) if callable(col) else r.get(col)
                cells.append(fmt(v, spec).replace("|", "/").replace("\n", " "))
            out.append("| " + " | ".join(cells) + " |")
        return out + [""]


class Registry:
    """Every number cited in INDEX.md, with provenance (written to numbers.json)."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.headlines: list[tuple[str, str]] = []

    def cite(self, key: str, table: Table, row, col: str, spec: str = ".2f", headline: str | None = None) -> str:
        if row is None:
            return NOT_RUN
        v = py(row.get(col))
        if v is None:
            return "n/a"
        entry = {"value": v, "table": table.name, "row": row["_key"], "column": col,
                 "source_file": row.get("_src"), "seed": py(row.get("_seed")), "fit_seed": py(row.get("_fit_seed")),
                 "git_hash": row.get("_git")}
        loc = row["_loc"].get(col)
        if loc:
            entry["locator"] = loc
        else:
            entry["derived"] = row["_derived"].get(col, "computed by scripts/collect_results.py from the source file(s)")
        if key in self.items and self.items[key]["value"] != v:
            raise ValueError(f"number {key} registered twice with different values")
        self.items[key] = entry
        if headline and key not in [k for k, _ in self.headlines]:
            self.headlines.append((key, headline))
        return fmt(v, spec)

    def put(self, key: str, value, spec: str, *, source, table=None, row=None, git_hash=None, seed=None, fit_seed=None,
            locator=None, derived=None) -> str:
        """A number computed while writing INDEX.md (e.g. a sum over table rows)."""
        v = py(value)
        if v is None:
            return "n/a"
        entry = {"value": v, "table": table, "row": row, "source_file": source, "seed": seed, "fit_seed": fit_seed,
                 "git_hash": git_hash}
        if locator:
            entry["locator"] = locator
        else:
            entry["derived"] = derived or "computed by scripts/collect_results.py"
        if key in self.items and self.items[key]["value"] != v:
            raise ValueError(f"number {key} registered twice with different values")
        self.items[key] = entry
        return fmt(v, spec)


# ================================================================================================
# Loading
# ================================================================================================
JOB_RE = re.compile(r"scripts/run_e1\.py\s+--config\s+(?P<config>\S+)\s+--fit-seed\s+(?P<seed>\d+)"
                    r"\s+--ks\s+(?P<ks>\S+)\s+--out-root\s+(?P<out>[^\s']+)")


def parse_jobs(P: Paths) -> dict:
    jobs: dict[str, list] = {}
    if not P.jobs.is_dir():
        return jobs
    for f in sorted(P.jobs.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = JOB_RE.search(line)
            if m:
                jobs.setdefault(m["out"].rstrip("/"), []).append(
                    {"config": m["config"], "fit_seed": int(m["seed"]), "ks": [int(x) for x in m["ks"].split(",")],
                     "job_file": P.rel(f)})
    return jobs


def load_cfg(P: Paths, path) -> dict:
    if not path:
        return {}
    p = P.root / path
    if not p.is_file():
        return {}
    try:
        return load_config(p)
    except Exception as e:  # noqa: BLE001
        log(f"WARNING cannot load config {p}: {e}")
        return {}


def load_e1_run(P: Paths, run_dir: Path, jobs: dict, res_dir: Path | None = None) -> dict:
    """One output folder of scripts/run_e1.py (E1, E1b, X2, G)."""
    run_dir = Path(run_dir)
    res = Path(res_dir) if res_dir else run_dir / "results"
    r = {"dir": P.rel(run_dir), "name": run_dir.name, "res": res, "res_rel": P.rel(res)}
    meta = read_json(res / "e1_meta.json") or {}
    r["meta"] = meta
    for n in ("layers", "mlp", "surgery"):
        r[n] = read_csv(res / f"e1_{n}.csv")
        r[f"{n}_file"] = P.rel(res / f"e1_{n}.csv")
    r["meta_file"] = P.rel(res / "e1_meta.json")
    present = [n for n in ("layers", "mlp", "surgery") if r[n] is not None]
    if meta and len(present) == 3:
        r["status"] = "complete"
    elif present:
        missing = [f"e1_{n}.csv" for n in ("layers", "mlp", "surgery") if r[n] is None]
        if not meta:
            missing.append("e1_meta.json")
        r["status"] = "partial (missing " + ", ".join(missing) + ")"
    else:
        r["status"] = NOT_RUN
    job = (jobs.get(r["dir"]) or [None])[0]
    r["job"] = job
    cfg_path = meta.get("config_path") or (job or {}).get("config")
    cfg = meta.get("config") or load_cfg(P, cfg_path)
    r["config_path"], r["cfg"] = cfg_path, cfg
    r["model"] = (cfg.get("model") or {}).get("name")
    protocol = meta.get("protocol") or (cfg.get("data") or {}).get("protocol") or "v1"
    r["protocol"] = protocol
    fs, fs_src = meta.get("fit_seed"), "e1_meta.json"
    if protocol == "v2":
        if fs is None and job:
            fs, fs_src = job["fit_seed"], f"{job['job_file']} (no e1_meta.json)"
        if fs is None:
            m = re.search(r"_s(\d+)$", run_dir.name)
            fs, fs_src = (int(m[1]), "folder name") if m else (None, None)
    else:
        fs, fs_src = None, None
    r["fit_seed"], r["fit_seed_source"] = fs, fs_src
    git = (meta.get("git") or {}).get("git_hash")
    git_src = "e1_meta.json" if git else None
    if not git:
        hashes = set()
        for n in present:
            if "git_hash" in r[n]:
                hashes |= set(r[n]["git_hash"].dropna().astype(str))
        git = ";".join(sorted(hashes)) if hashes else None
        git_src = "CSV git_hash column" if git else None
    r["git"], r["git_source"], r["git_dirty"] = git, git_src, (meta.get("git") or {}).get("git_dirty")
    seed = meta.get("seed")
    if seed is None:
        for n in present:
            if "seed" in r[n]:
                seed = int(r[n]["seed"].iloc[0])
                break
    r["seed"] = seed if seed is not None else cfg.get("seed")
    r["operator_mode"] = meta.get("operator_mode") or (cfg.get("stats") or {}).get("operator_mode") or "shared"
    r["ks_planned"] = sorted(set((cfg.get("surgery") or {}).get("ks") or (job or {}).get("ks") or []))
    r["date"] = meta.get("date")
    r["split_sizes"] = meta.get("split_sizes")
    r["arm"] = re.sub(r"_s\d+$", "", run_dir.name)
    # Decision table: from e1_meta.json; recomputed with mirror.report.decide for runs without one.
    r["decision"], r["decision_source"], r["decision_matches_meta"] = meta.get("decision"), "e1_meta.json", None
    dcfg = cfg.get("decision")
    rec = None
    if dcfg and r["layers"] is not None and r["mlp"] is not None and r["surgery"] is not None:
        try:
            if r["operator_mode"] == "per_layer":
                lo, hi = int(dcfg["blocks"][0]), int(dcfg["blocks"][1])
                L = r["layers"][(r["layers"]["layer"] >= lo) & (r["layers"]["layer"] <= hi)]
                dmf = float(np.mean(L["d_minus"] / L["d"]))
            else:
                sh = meta.get("shared") or {}
                dmf = sh["d_minus"] / int(r["layers"]["d"].iloc[0]) if "d_minus" in sh else None
            if dmf is not None:
                rec = decide(r["layers"], r["mlp"], r["surgery"], dmf, dcfg, mode=r["operator_mode"])
        except Exception as e:  # noqa: BLE001
            log(f"WARNING decision recompute failed for {r['dir']}: {e}")
    if r["decision"] is None and rec is not None:
        r["decision"], r["decision_source"] = rec, "recomputed by mirror.report.decide (no e1_meta.json)"
    elif r["decision"] is not None and rec is not None:
        same = rec["outcome"] == r["decision"]["outcome"] and all(
            a["cell"] == b["cell"] and (py(a["value"]) is None or abs(float(a["value"]) - float(b["value"]))
                                        <= 1e-9 * max(1.0, abs(float(b["value"]))))
            for a, b in zip(rec["rows"], r["decision"]["rows"]))
        r["decision_matches_meta"] = bool(same)
    if r["decision"] is None:
        r["decision_source"] = None
    r["tokens_per_image"] = None
    if r["layers"] is not None and r["split_sizes"] and "n_tokens" in r["layers"]:
        n_tok = r["layers"]["n_tokens"].unique()
        held = r["split_sizes"].get("heldout")
        if len(n_tok) == 1 and held and float(n_tok[0]) % held == 0:
            r["tokens_per_image"] = int(n_tok[0] // held)
    return r


def discover_e1_runs(P: Paths, jobs: dict, sub: str) -> dict:
    base = P.runs / sub
    out = {}
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                out[d.name] = load_e1_run(P, d, jobs)
    for job_out in jobs:
        if job_out.startswith(f"runs/{sub}/"):
            name = job_out.split("/")[-1]
            if name not in out:
                out[name] = load_e1_run(P, P.root / job_out, jobs)
    return out


def load_m1(P: Paths) -> dict:
    d = P.runs / "m1"
    m = {"dir": P.rel(d)}
    for n in ("knn", "ranks", "heads", "heads_detail", "vexact", "vexact_blocks", "vexact_layers", "exactness"):
        m[n] = read_csv(d / f"m1_{n}.csv")
        m[f"{n}_file"] = P.rel(d / f"m1_{n}.csv")
    m["meta"] = read_json(d / "m1_meta.json") or {}
    m["flip"] = read_json(d / "m1_flipcheck.json")
    m["a_out"] = read_json(d / "m1_a_out_choice.json")
    m["meta_file"], m["flip_file"], m["a_out_file"] = (P.rel(d / "m1_meta.json"), P.rel(d / "m1_flipcheck.json"),
                                                        P.rel(d / "m1_a_out_choice.json"))
    m["model"] = m["meta"].get("model")
    m["git"] = (m["meta"].get("git") or {}).get("git_hash")
    if not m["git"] and m["knn"] is not None and "git_hash" in m["knn"]:
        m["git"] = ";".join(sorted(set(m["knn"]["git_hash"].astype(str))))
    m["ks"] = m["meta"].get("ks") or (sorted(set(m["knn"]["k"]) - {0}) if m["knn"] is not None else [])
    have = [n for n in ("knn", "ranks", "heads", "vexact") if m[n] is not None]
    m["status"] = "complete" if (m["meta"] and len(have) == 4) else ("partial" if have else NOT_RUN)
    return m


def load_m3(P: Paths) -> dict:
    base = P.runs / "m3_pilot"
    out = {}
    for method in ("mirror", "svd", "random"):
        d = base / method
        r = {"method": method, "dir": P.rel(d)}
        r["summary"] = read_json(d / "summary.json")
        r["knn"] = read_json(d / "knn.json") or ((r["summary"] or {}).get("knn"))
        r["knn_file"] = P.rel(d / "knn.json") if (d / "knn.json").is_file() else (
            P.rel(d / "summary.json") if r["knn"] else None)
        r["knn_prefix"] = [] if (d / "knn.json").is_file() else ["knn"]
        r["metrics"] = read_json(d / "final" / "metrics.json")
        r["metrics_file"] = P.rel(d / "final" / "metrics.json")
        r["lrsel"] = read_json(d / "lr_selection" / "lr_selection.json")
        r["lrsel_file"] = P.rel(d / "lr_selection" / "lr_selection.json")
        r["train_log"] = read_csv(d / "final" / "train_log.csv")
        r["heldout_log"] = read_csv(d / "final" / "heldout_log.csv")
        healed = (r["knn"] or {}).get("healed")
        if healed:
            r["status"] = "complete"
        elif r["metrics"]:
            r["status"] = "healed; final k-NN not yet available"
        elif r["lrsel"]:
            r["status"] = "learning-rate selection only"
        else:
            r["status"] = NOT_RUN
        out[method] = r
    return out


_RUN_FIT_SEEDS: dict[str, int] = {}   # run folder (relative) -> fit seed; filled by main()


def fit_seed_from_path(path) -> int | None:
    """Fit seed of the run whose folder holds an operators file (e.g. runs/x2/perlayer_reg_s0/results/...)."""
    s = str(path or "").replace("\\", "/")
    for d, fs in _RUN_FIT_SEEDS.items():
        if d and (s == d or s.startswith(d + "/") or f"/{d}/" in s):
            return fs
    return None


def git_added_commit(P: Paths, file: str) -> str | None:
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", file], cwd=P.root,
                             capture_output=True, text=True, timeout=20)
        lines = [x for x in out.stdout.split() if x]
        return lines[-1] if out.returncode == 0 and lines else None
    except Exception:  # noqa: BLE001
        return None


def parse_status(P: Paths) -> dict:
    """Sections and bullets of results/STATUS.md (used for reasons and deviations)."""
    p = P.results / "STATUS.md"
    out = {"exists": p.is_file(), "sections": {}, "lines": []}
    if not p.is_file():
        return out
    cur = "preamble"
    for line in p.read_text(encoding="utf-8").splitlines():
        out["lines"].append(line)
        if line.startswith("## "):
            cur = line[3:].strip()
            out["sections"][cur] = []
        else:
            out["sections"].setdefault(cur, []).append(line)
    return out


def status_mentions(st: dict, tokens) -> list[str]:
    """Items of STATUS.md that mention any token: '<section>: <item>' (semicolon lists split)."""
    hits = []
    for sec, lines in st["sections"].items():
        for line in lines:
            text = line.strip().lstrip("-").strip()
            if not text or text.startswith("|---"):
                continue
            items = [x.strip() for x in text.split(";")] if ";" in text and not text.startswith("|") else [text]
            for it in items:
                if any(re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![0-9])", it) for t in tokens):
                    s = f"STATUS.md, {sec}: {it.strip(' .')}"
                    if s not in hits:
                        hits.append(s)
    return hits


def parse_prereg(P: Paths) -> dict:
    """Thresholds of PREREG_E2.md (read from the text, never typed)."""
    p = P.root / "PREREG_E2.md"
    th = {"file": "PREREG_E2.md", "commit": git_added_commit(P, "PREREG_E2.md")}
    if not p.is_file():
        return th
    t = " ".join(p.read_text(encoding="utf-8").split())
    pats = {
        "p1_plain_cut": r"layers 9 to 11 by at least ([\d.]+)x",
        "p1_reg_ratio": r"at most ([\d.]+)x its mean over layers 1 to 6",
        "p2_frac_pct": r"in at least ([\d.]+)% of block and model cases",
        "p3_random": r"at least ([\d.]+) k-NN points above the healed random",
        "p3_svd": r"at least ([\d.]+) points above healed SVD",
        "p3_orig": r"within ([\d.]+) points of the original",
        "p4_share": r"breaking share at most ([\d.]+) over their first half",
        "p4_ratio": r"e\(null\)/e\(A\) at least ([\d.]+)",
        "p5_cos": r"principal angles at least ([\d.]+)",
        "p5_pairs_pct": r"recovers at least ([\d.]+)% of the data-driven pairs",
    }
    for k, pat in pats.items():
        m = re.search(pat, t)
        th[k] = float(m[1]) if m else None
    return th


# ================================================================================================
# Analytic cost model
# ================================================================================================
COST_ASSUMPTIONS = [
    "Analytic counts at 224 px, not measured latency (M2 measures latency; not run).",
    "MACs = multiply-accumulates per image; GFLOPs per image = 2 x MACs / 1e9.",
    "Counted: patch embedding n_patches x (in_chans p^2) x d; per block and token: attention "
    "projections 4 d^2 (qkv 3 d^2 plus proj d^2) and MLP 2 h d; per block and image: attention token "
    "mixing 2 T^2 d (q k^T and attention times v), which surgery does not change.",
    "Not counted: biases, norms, softmax, GELU, LayerScale, residual additions, the twin butterfly "
    "additions (u + v, u - v) and the k-NN head.",
    "Mirror V-local, MLP only (and the MLPs of V-local and V-exact): each matrix costs its equivariant free-parameter count (h d + tr(P) tr(A)) / 2 per token "
    "(eq_params_fc1, eq_params_fc2 of e1_mlp.csv; mlp_eq_fc1, mlp_eq_fc2 of m1_ranks.csv), i.e. every "
    "free parameter is used once per token in even/odd coordinates.",
    "Mirror attention (M1): the free-parameter counts of Wq, Wk, Wv, Wo per block (attn_free_q/k/v/o of "
    "m1_ranks.csv for V-local; attn_free_matrices and mlp_free of m1_vexact_blocks.csv for V-exact, "
    "which depend on the span k).",
    "Basis rotations: every variant with per-layer operators (Mirror V-local MLP only, V-local, and the random "
    "controls, which use random per-layer involutions) pays one d x d rotation per converted block "
    "(d^2 MACs per token): the stream is rotated from the basis of A_(l-1) to that of A_l at each "
    "converted block, the rotation into the first basis is folded into the linear patch embedding and the "
    "one after the last converted block into the next dense block. V-exact (one operator for the span) "
    "and SVD need none. Rotation matrices are d^2 stored parameters each; they are reported in "
    "rotation_params and are not included in model_params (which follows m1_knn.csv).",
    "V-local attention is counted with Wo at its free-parameter count (structured). The implementation timed in M2 "
    "stores Wo dense in the block's basis instead (M1 projected Wo with A_(l+1), not the block's input operator, and "
    "keeping it structured would need a second rotation per block), so the M2 count of the same variant is higher by "
    "T (d H dh - attn_free_o) multiply-adds per converted block (T_latency, columns analytic (M2) and "
    "wo_dense_extra_gflops). The two counts agree for every other variant up to one d x d rotation of the CLS "
    "feature that M2 also counts.",
    "SVD: r (in + out) per matrix (ranks svd_rank and svd_rank_fc2 of e1_mlp.csv; svd_r_* of m1_ranks.csv).",
    "Whole-model parameters: the dense model's total minus the dense matrix parameters of the converted "
    "matrices plus the kept ones (biases stay dense), as in m1_knn.csv.",
]


def analytic_cost(arch: dict, conv_blocks: list[dict]) -> dict:
    """Analytic parameters and multiply-adds of one variant (see COST_ASSUMPTIONS).

    arch: d, h, depth, T (tokens per image), n_patches, patch, in_chans, model_params_dense (or None).
    conv_blocks: one dict per converted block with per-token MACs of its converted matrices:
      {"mlp": fc1 + fc2, "attn": Wq + Wk + Wv + Wo, "rot": number of d x d basis rotations}.
    Returns parameter counts (matrix parameters of the converted blocks, whole model) and MACs and
    GFLOPs for the whole model and for the converted blocks, with the percentages saved.
    """
    d, h, T, depth = arch["d"], arch["h"], arch["T"], arch["depth"]
    k = len(conv_blocks)
    dense_mat = 4 * d * d + 2 * h * d
    mix = 2 * T * T * d
    patch_macs = arch["n_patches"] * arch["in_chans"] * arch["patch"] ** 2 * d
    dense_block = T * dense_mat + mix
    kept = sum(b["mlp"] + b["attn"] for b in conv_blocks)
    rot = sum(b["rot"] for b in conv_blocks)
    conv_macs = sum(T * (b["mlp"] + b["attn"]) + mix + b["rot"] * T * d * d for b in conv_blocks)
    conv_dense = k * dense_block
    model_dense_macs = patch_macs + depth * dense_block
    model_macs = model_dense_macs - conv_dense + conv_macs
    mp_dense = arch.get("model_params_dense")
    out = {
        "matrix_params_converted_dense": k * dense_mat, "matrix_params_kept": kept,
        "kept_frac_converted": kept / (k * dense_mat) if k else 1.0,
        "mlp_kept_frac": sum(b["mlp"] for b in conv_blocks) / (k * 2 * h * d) if k else 1.0,
        "rotation_params": rot * d * d,
        "model_params": (mp_dense - k * dense_mat + kept) if mp_dense else None,
        "macs_model": model_macs, "macs_model_dense": model_dense_macs,
        "gflops_model": 2 * model_macs / 1e9, "gflops_model_dense": 2 * model_dense_macs / 1e9,
        "gflops_saved_model_pct": 100.0 * (1 - model_macs / model_dense_macs),
        "macs_converted": conv_macs, "gflops_converted": 2 * conv_macs / 1e9,
        "gflops_converted_dense": 2 * conv_dense / 1e9,
        "gflops_saved_converted_pct": 100.0 * (1 - conv_macs / conv_dense) if k else 0.0,
    }
    out["model_params_frac"] = out["model_params"] / mp_dense if mp_dense else None
    return out


def model_arch(P: Paths, model: str, run: dict | None, m1: dict, f3: dict) -> dict | None:
    """Inputs of the cost model for one model, each read from a result file (sources recorded)."""
    if run is None or run["mlp"] is None or run["layers"] is None:
        return None
    mlp = run["mlp"]
    a = {"model": model, "d": int(mlp["d"].iloc[0]), "h": int(mlp["h"].iloc[0]), "depth": int(mlp["block"].max()),
         "in_chans": 3, "sources": {}}
    a["sources"]["d, h, depth"] = run["mlp_file"]
    a["sources"]["in_chans"] = "assumption: RGB input"
    T = run["tokens_per_image"]
    if T is None:
        return None
    a["T"] = T
    a["sources"]["T"] = f"n_tokens / split_sizes.heldout ({run['layers_file']}, {run['meta_file']})"
    crop = (run["cfg"].get("data") or {}).get("crop")
    grid = None
    fj = (f3.get(model) or {}).get("json")
    if fj and (fj.get("data") or {}).get("grid_hw"):
        grid, a["sources"]["grid"] = fj["data"]["grid_hw"], f3[model]["file"] + " data.grid_hw"
    elif m1.get("model") == model and m1["meta"].get("grid"):
        grid, a["sources"]["grid"] = m1["meta"]["grid"], m1["meta_file"] + " grid"
    if not grid or not crop:
        return None
    a["n_patches"] = int(grid[0]) * int(grid[1])
    a["patch"] = int(crop) // int(grid[1])
    a["sources"]["patch"] = f"crop {crop} ({run['meta_file']} config.data.crop) / grid width"
    a["n_prefix"] = T - a["n_patches"]
    a["model_params_dense"] = None
    knn = m1.get("knn")
    if knn is not None and m1.get("model"):
        orig = knn[(knn["k"] == 0) & (knn["variant"] == "original")]
        if len(orig):
            mp = int(orig["model_params"].iloc[0])
            if m1["model"] == model:
                a["model_params_dense"] = mp
                a["sources"]["model_params_dense"] = f"{m1['knn_file']} (k = 0, original, model_params)"
            elif m1["model"] == f"{model}_reg" and m1["meta"].get("n_registers") is not None:
                nreg = int(m1["meta"]["n_registers"])
                a["model_params_dense"] = mp - nreg * a["d"]
                a["sources"]["model_params_dense"] = (
                    f"{m1['knn_file']} original model_params of {m1['model']} minus n_registers x d "
                    f"({m1['meta_file']}: n_registers {nreg}); the two models differ only by the register "
                    "tokens (verify against the hub model)")
    return a


def x2_blocks(run: dict, variant: str, k: int) -> list[dict] | None:
    mlp = run["mlp"]
    if mlp is None or int(mlp["block"].max()) < k:
        return None
    d, h = int(mlp["d"].iloc[0]), int(mlp["h"].iloc[0])
    out = []
    for b in range(1, k + 1):
        r = mlp[mlp["block"] == b].iloc[0]
        if variant in ("mirror_mlp", "random_mlp"):
            out.append({"mlp": int(r["eq_params_fc1"] + r["eq_params_fc2"]), "attn": 4 * d * d, "rot": 1})
        elif variant == "svd_mlp":
            out.append({"mlp": int((r["svd_rank"] + r["svd_rank_fc2"]) * (h + d)), "attn": 4 * d * d, "rot": 0})
        else:
            raise ValueError(variant)
    return out


def m1_blocks(m1: dict, variant: str, k: int, d: int, h: int) -> list[dict] | None:
    rk = m1["ranks"]
    if rk is None or int(rk["block"].max()) < k:
        return None
    out = []
    for b in range(1, k + 1):
        r = rk[rk["block"] == b].iloc[0]
        mlp_eq = int(r["mlp_eq_fc1"] + r["mlp_eq_fc2"])
        attn_free = int(r["attn_free_q"] + r["attn_free_k"] + r["attn_free_v"] + r["attn_free_o"])
        if variant in ("vlocal", "random_attn"):
            out.append({"mlp": mlp_eq, "attn": attn_free, "rot": 1})
        elif variant == "mirror_mlp_m1":
            out.append({"mlp": mlp_eq, "attn": 4 * d * d, "rot": 1})
        elif variant == "svd_attn":
            out.append({"mlp": int((r["svd_r_fc1"] + r["svd_r_fc2"]) * (h + d)),
                        "attn": int((r["svd_r_q"] + r["svd_r_k"] + r["svd_r_v"] + r["svd_r_o"]) * 2 * d), "rot": 0})
        elif variant == "vexact":
            vb = m1["vexact_blocks"]
            if vb is None:
                return None
            sel = vb[(vb["k"] == k) & (vb["block"] == b)]
            if len(sel) != 1:
                return None
            out.append({"mlp": int(sel["mlp_free"].iloc[0]), "attn": int(sel["attn_free_matrices"].iloc[0]), "rot": 0})
        else:
            raise ValueError(variant)
    return out


# ================================================================================================
# T_main_zeroshot
# ================================================================================================
def build_main(P, runs_x2, m1, archs, checks) -> Table:
    T = Table("T_main_zeroshot", "Zero-shot k-NN (protocol v2): registers model and plain model",
              ["model", "k", "variant"],
              [("model", "model_label", ""), ("k", "k", "d"), ("variant", "variant_label", ""),
               ("budget", "budget", ""), ("run", "experiment", ""), ("canonical", "canonical", ""), ("n", "n_seeds", "d"),
               ("top-1", lambda r: pm(r.get("top1_mean"), r.get("top1_std")), ""),
               ("drop", lambda r: pm(r.get("drop_mean"), r.get("drop_std")), ""),
               ("vs SVD", lambda r: pm(r.get("margin_svd_mean"), r.get("margin_svd_std")), ""),
               ("vs random", lambda r: pm(r.get("margin_random_mean"), r.get("margin_random_std")), ""),
               ("flip consistency", lambda r: pm(r.get("flip_mean"), r.get("flip_std")), ""),
               ("kept (converted)", "kept_frac_converted", ".3f"), ("MLP kept", "mlp_kept_frac", ".3f"),
               ("model params", "model_params", ",.0f"), ("GFLOPs", "gflops_model", ".2f"),
               ("saved, model %", "gflops_saved_model_pct", ".1f"),
               ("saved, converted %", "gflops_saved_converted_pct", ".1f"),
               ("status", "status", ""), ("row note", lambda r: f"[{r['remark_id']}]" if r.get("remark_id") else "", "")])
    T.notes = [
        "Top-1 and flip consistency in percent (k-NN, k = 20, T = 0.07, bf16 features; bank of 100 train images "
        "per class, all 50k validation queries). Plus or minus is the sample standard deviation over fit seeds "
        "(n - 1); with one seed none is shown. Drop = original minus variant; vs SVD and vs random = variant "
        "minus the baseline at the same k, budget and fit seed.",
        "Kept: matrix parameters (MLP and attention) of the converted blocks over dense; MLP kept: MLP matrices "
        "only (the mlp_params_kept_frac of the E1b reports). GFLOPs and savings come from the analytic "
        "cost model (tables/cost_model.md): analytic counts, not measured latency.",
        "Budgets: X2 variants convert the MLP only (SVD and random matched to the Mirror MLP count); M1 variants "
        "convert MLP and attention (SVD and random matched to V-local). budget_gap_vs_svd_pct in the CSV shows "
        "how far a Mirror variant's kept parameters are from its SVD baseline (the plan asks for within 1%). "
        "Model params: a mean over fit seeds where the seeds keep slightly different counts.",
        f"Canonical: the one number to quote for a setting. Every MLP-only row comes from the {CANONICAL_RUN} runs "
        "(scripts/run_e1.py, the main protocol, 3 fit seeds) where they exist. The M1 re-run of Mirror V-local, MLP only "
        "(run M1) is the same setting at fit seed 0 and is marked not canonical wherever an X2 row exists: it "
        + OPERATOR_ROUNDTRIP + ". Row notes [n] are listed under the table; top1_per_seed in the CSV gives the "
        "per-seed values.",
    ]
    recs: dict = {}

    def add_rec(model, k, variant, rec):
        recs.setdefault((model, k, variant), []).append(rec)

    x2_pl = [r for r in runs_x2.values() if r["operator_mode"] == "per_layer" and r["protocol"] == "v2"
             and r["surgery"] is not None and r["model"] in MAIN_MODELS]
    for r in x2_pl:
        arch = archs.get(r["model"])
        for _, s in r["surgery"].iterrows():
            k, v = int(s["k"]), str(s["variant"])
            key = {"surgery": "mirror_mlp", "svd": "svd_mlp", "random": "random_mlp", "original": "original"}.get(v)
            if key is None:
                continue
            cost = None
            if arch:
                blocks = [] if key == "original" else x2_blocks(r, key, k)
                if blocks is not None:
                    cost = analytic_cost(arch, blocks)
                    if key != "original" and "mlp_params_kept_frac" in s:
                        mlp_kept = sum(b["mlp"] for b in blocks) / (k * 2 * arch["h"] * arch["d"])
                        checks.append({"check": f"{r['dir']} k={k} {v}: MLP kept fraction", "collector": mlp_kept,
                                       "recorded": float(s["mlp_params_kept_frac"]),
                                       "source": r["surgery_file"]})
            add_rec(r["model"], k, key, {"fit_seed": r["fit_seed"], "seed": r["seed"], "git": r["git"],
                                          "src": r["surgery_file"], "where": {"k": k, "variant": v},
                                          "top1": float(s["top1"]), "flip": float(s["flip_consistency"]),
                                          "drop": float(s["drop_points"]), "cost": cost})
    if m1["knn"] is not None and m1["model"] in MAIN_MODELS:
        arch = archs.get(m1["model"])
        m1map = {"original": "original_m1", "folded": "folded", "vlocal": "vlocal", "vexact": "vexact",
                 "mlp_only": "mirror_mlp_m1", "svd": "svd_attn", "random": "random_attn"}
        for _, s in m1["knn"].iterrows():
            k, v = int(s["k"]), str(s["variant"])
            key = m1map.get(v)
            if key is None:
                continue
            cost = None
            if arch:
                blocks = [] if k == 0 else m1_blocks(m1, key, k, arch["d"], arch["h"])
                if blocks is not None:
                    cost = analytic_cost(arch, blocks)
                    if k:
                        mlpk = sum(b["mlp"] for b in blocks)
                        attk = sum(b["attn"] for b in blocks)
                        for name, mine, col in (("MLP params kept", mlpk, "mlp_params_kept"),
                                                ("attention params kept", attk, "attn_params_kept"),
                                                ("model params", cost["model_params"], "model_params")):
                            checks.append({"check": f"m1 k={k} {v}: {name}", "collector": mine,
                                           "recorded": float(s[col]), "source": m1["knn_file"]})
            add_rec(m1["model"], k, key, {"fit_seed": int(s["fit_seed"]), "seed": int(s["seed"]),
                                          "git": str(s["git_hash"]), "src": m1["knn_file"],
                                          "where": {"k": k, "variant": v}, "top1": float(s["top1"]),
                                          "flip": float(s["flip_consistency"]), "drop": float(s["drop_points"]),
                                          "cost": cost})

    def reason(model, k, key):
        exp = VARIANT_INFO[key][3] if key in VARIANT_INFO else "X2"
        if exp == "X2":
            mine = [r for r in runs_x2.values() if r["model"] == model and r["operator_mode"] == "per_layer"]
            if not mine:
                return "no X2 per-layer run for this model"
            ks = sorted({kk for r in mine for kk in r["ks_planned"]})
            done = sorted({int(x) for r in mine if r["surgery"] is not None for x in r["surgery"]["k"] if int(x)})
            return (f"X2 surgery planned at k = {', '.join(map(str, ks))} and run at k = "
                    f"{', '.join(map(str, done)) or 'none'} (config surgery.ks, e1_surgery.csv)")
        if m1["status"] == NOT_RUN:
            return "M1 not run (no runs/m1/m1_knn.csv)"
        if m1["model"] != model:
            return f"M1 was run on {MODEL_LABEL.get(m1['model'], m1['model'])} only ({m1['meta_file']})"
        return f"M1 ran k = {', '.join(map(str, m1['ks']))} ({m1['meta_file']})"

    for model in MAIN_MODELS:
        base_rows = [("original", "Original model", "dense", "X2"), ("folded", "Folded original (X4 sanity, M1)", "dense", "M1")]
        for k in [0] + MAIN_KS:
            keys = base_rows if k == 0 else [(v[0], v[1], v[2], v[3]) for v in VARIANTS]
            for key, label, budget, exp in keys:
                rs = recs.get((model, k, key), [])
                row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "k": k, "variant": key,
                       "variant_label": label, "budget": budget, "experiment": exp, "_loc": {}, "_derived": {}}
                if not rs:
                    if key == "folded" and model != m1.get("model"):
                        continue
                    row.update({"status": NOT_RUN, "note": reason(model, k, key), "n_seeds": 0})
                    T.add(row)
                    continue
                rs = sorted(rs, key=lambda x: (x["fit_seed"] is None, x["fit_seed"]))
                t_m, t_s, n = agg([x["top1"] for x in rs])
                d_m, d_s, _ = agg([x["drop"] for x in rs])
                f_m, f_s, _ = agg([x["flip"] for x in rs])
                row.update({"status": "ok", "n_seeds": n, "fit_seeds": ",".join(str(x["fit_seed"]) for x in rs),
                            "top1_mean": t_m, "top1_std": t_s, "drop_mean": d_m, "drop_std": d_s,
                            "flip_mean": f_m, "flip_std": f_s,
                            "top1_per_seed": "; ".join(f"s{x['fit_seed']}: {x['top1']:.3f}" for x in rs)})
                row["_per_seed"] = {x["fit_seed"]: x for x in rs}
                if key == "original" and n > 1 and len({x["top1"] for x in rs}) == 1 and len({x["flip"] for x in rs}) == 1:
                    # one evaluation of the unmodified model, repeated in every fit-seed run's CSV (run_e1.py reads it
                    # from cache/knn_original when present): no spread to report
                    row.update({"top1_std": None, "drop_std": None, "flip_std": None,
                                "remark": f"the original model's k-NN is identical in all {n} runs' CSVs (one evaluation; "
                                          "run_e1.py reuses cache/knn_original), so no spread is shown"})
                for bname, bkey in zip(("svd", "random"), BASELINES.get(key, (None, None))):
                    if bkey is None:
                        continue
                    bl = {x["fit_seed"]: x for x in recs.get((model, k, bkey), [])}
                    diffs = [x["top1"] - bl[x["fit_seed"]]["top1"] for x in rs if x["fit_seed"] in bl]
                    mm, ms, mn = agg(diffs)
                    row.update({f"margin_{bname}_mean": mm, f"margin_{bname}_std": ms, f"margin_{bname}_n": mn})
                    row["_derived"][f"margin_{bname}_mean"] = (
                        f"top1 of {key} minus top1 of {bkey} at k = {k} and the same fit seed, mean over seeds")
                    if bname == "svd" and mn:
                        gaps = [100.0 * (x["cost"]["matrix_params_kept"] / bl[x["fit_seed"]]["cost"]["matrix_params_kept"] - 1)
                                for x in rs if x["fit_seed"] in bl and x["cost"] and bl[x["fit_seed"]]["cost"]]
                        g_m, _, _ = agg(gaps)
                        row["budget_gap_vs_svd_pct"] = g_m
                        row["budget_within_1pct"] = (abs(g_m) <= 1.0) if g_m is not None else None
                costs = [x["cost"] for x in rs if x["cost"]]
                if costs:
                    for c in ("matrix_params_kept", "kept_frac_converted", "mlp_kept_frac", "model_params", "model_params_frac",
                              "rotation_params", "gflops_model", "gflops_saved_model_pct", "gflops_converted",
                              "gflops_saved_converted_pct"):
                        vals = [cc[c] for cc in costs if cc[c] is not None]
                        row[c] = float(np.mean(vals)) if vals else None
                        if c in ("matrix_params_kept", "model_params", "rotation_params") and row[c] is not None \
                                and float(row[c]).is_integer():
                            row[c] = int(row[c])
                        row["_derived"][c] = "analytic_cost() in scripts/collect_results.py (tables/cost_model.md)"
                row["source_files"] = ";".join(sorted({x["src"] for x in rs}))
                row["git_hashes"] = ";".join(sorted({str(x["git"]) for x in rs}))
                row["_src"], row["_git"] = row["source_files"], row["git_hashes"]
                row["_seed"] = rs[0]["seed"] if len({x["seed"] for x in rs}) == 1 else None
                row["_fit_seed"] = rs[0]["fit_seed"] if n == 1 else row["fit_seeds"]
                if n == 1:
                    x = rs[0]
                    row["_loc"] = {"top1_mean": csv_loc(x["src"], x["where"], "top1"),
                                   "flip_mean": csv_loc(x["src"], x["where"], "flip_consistency"),
                                   "drop_mean": csv_loc(x["src"], x["where"], "drop_points")}
                else:
                    for c in ("top1_mean", "flip_mean", "drop_mean"):
                        row["_derived"][c] = f"mean over fit seeds {row['fit_seeds']} of the per-run values"
                T.add(row)
    mark_canonical_main(T)
    return T


def mark_canonical_main(T: Table) -> None:
    """Canonical flag and row remarks of T_main_zeroshot (rule: CANONICAL_RUN for every MLP-only setting it covers)."""
    for r in T.rows:
        if r["status"] != "ok":
            r["canonical"] = ""
            continue
        r["canonical"] = "yes"
        if r["variant"] == "mirror_mlp_m1":
            x2 = T.find(model=r["model"], k=r["k"], variant="mirror_mlp")
            if x2 is not None and x2["status"] == "ok":
                r["canonical"] = "no"
                same = [(fs, x) for fs, x in r["_per_seed"].items() if fs in x2["_per_seed"]]
                diffs = "; ".join(f"fit seed {fs}: {x['top1']:.3f} against {x2['_per_seed'][fs]['top1']:.3f} "
                                  f"({x['top1'] - x2['_per_seed'][fs]['top1']:+.3f})" for fs, x in same)
                r["minus_canonical"] = (float(np.mean([x["top1"] - x2["_per_seed"][fs]["top1"] for fs, x in same]))
                                        if same else None)
                r["_derived"]["minus_canonical"] = f"top1 minus the {CANONICAL_RUN} mirror_mlp top1 at the same k and fit seed"
                r["remark"] = (f"same setting as the {CANONICAL_RUN} row {LABEL_MIRROR_MLP} at k = {r['k']} (canonical, "
                               f"n = {x2['n_seeds']}, {x2['top1_mean']:.2f}); {diffs}. This M1 run " + OPERATOR_ROUNDTRIP + ".")
            else:
                r["remark"] = (f"only source of {LABEL_MIRROR_MLP} at k = {r['k']} (the {CANONICAL_RUN} runs did not "
                               "convert this many blocks), so canonical here; one fit seed")
        elif r["variant"] == "random_mlp" and r["n_seeds"] > 1:
            # run_e1.py draws its random control with the run seed (0 in every fit-seed run): the random eigenbasis is
            # the same draw in every run, only the d+ counts and pair counts follow the fit seed
            seeds = sorted({x["seed"] for x in r["_per_seed"].values()})
            if len(seeds) == 1:
                r["remark"] = (f"the {r['n_seeds']} runs draw the random control with the same run seed ({seeds[0]}): the "
                               "random operators share one random eigenbasis and the pairings one seed, only the d+ and "
                               "pair counts follow the fit seed, so the spread is not a spread over independent random "
                               "draws (A1 random_structure in T_ablations uses three independent draws)")
    n = 0
    for r in T.rows:
        if r.get("remark"):
            n += 1
            r["remark_id"] = n


# ================================================================================================
# T_healing
# ================================================================================================
def build_healing(P, m3, m1, th, prereg_commit):
    T = Table("T_healing", "M3 pilot: healing distillation (k = 9, registers model, MLP plus attention, V-local)",
              ["method"],
              [("method", "method", ""), ("status", "status", ""), ("original", "original_top1", ".2f"),
               ("zero-shot", "zero_shot_top1", ".2f"), ("healed", "healed_top1", ".2f"),
               ("flip: zero-shot", "zero_shot_flip", ".2f"), ("flip: healed", "healed_flip", ".2f"),
               ("M1 zero-shot (same k)", "m1_zero_shot_top1", ".2f"), ("chosen LR", "chosen_lr", "g"),
               ("held-out loss before", "heldout_loss_before", ".3f"), ("after", "heldout_loss_after", ".3f"),
               ("steps", lambda r: f"{fmt(r.get('steps_done'), 'd')} / {fmt(r.get('total_steps'), 'd')}"
                if r.get("steps_done") is not None else "", ""),
               ("trainable params", "n_trainable", ",d"), ("free matrix params", "free_matrix_params", ",d")])
    T.notes = ["Top-1 and flip consistency in percent (protocol v2 k-NN). Held-out loss: the distillation loss "
               "on the 5k held-out train images (normalised by fixed teacher mean squares, the same for every "
               "method). Chosen LR: lowest held-out loss after the short selection budget, from "
               "{1e-5, 5e-5, 1e-4}. M1 zero-shot: the matching variant of runs/m1/m1_knn.csv at the same k, as a "
               "cross-check of the zero-shot column."]
    trainable_note = ("Trainable params: entries of the tensors the optimiser updates (biases included). Mirror and random "
                      "train the dense weight tensors of the converted blocks and re-project them onto their structured "
                      "form after every step (mirror/heal.py), while SVD trains its factors, so this column is not the "
                      "budget. Free matrix params (summary.json method_info, MLP plus attention kept) is the matched "
                      "quantity.")
    m1map = {"mirror": "vlocal", "svd": "svd", "random": "random"}
    default_cfg = load_cfg(P, "configs/m3_pilot.yaml")
    lrs = next((m3[m]["lrsel"].get("lrs") for m in m3 if m3[m]["lrsel"]), None) or \
        ((default_cfg.get("lr_selection") or {}).get("lrs"))
    T.notes = [T.notes[0].replace("{1e-5, 5e-5, 1e-4}", "{" + ", ".join(f"{x:g}" for x in (lrs or [])) + "} "
                                  "(lr_selection.json)"), trainable_note]
    for method in ("mirror", "svd", "random"):
        r = m3[method]
        cfg = (r["summary"] or {}).get("config")
        row = {"method": method, "status": r["status"], "_src": r["dir"], "_derived": {}, "_loc": {},
               "model": ((cfg or default_cfg or {}).get("model") or {}).get("name")}
        knn, kf, pre = r["knn"] or {}, r["knn_file"], r["knn_prefix"]
        for part in ("original", "zero_shot", "healed"):
            if part in knn:
                row[f"{part}_top1"] = py(knn[part].get("top1"))
                row[f"{part}_flip"] = py(knn[part].get("flip_consistency"))
                row["_loc"][f"{part}_top1"] = json_loc(kf, pre + [part, "top1"])
                row["_loc"][f"{part}_flip"] = json_loc(kf, pre + [part, "flip_consistency"])
        met = r["metrics"] or {}
        blocks = ((met.get("config") or {}).get("blocks")) or ((cfg or {}).get("blocks"))
        row["k"] = len(blocks) if blocks else None
        if r["lrsel"]:
            row["chosen_lr"] = py(r["lrsel"].get("best_lr"))
            row["_loc"]["chosen_lr"] = json_loc(r["lrsel_file"], ["best_lr"])
            row["lr_trials"] = "; ".join(f"{t['lr']:g}: {t['heldout_loss']:.4f}" for t in r["lrsel"].get("trials", []))
            row["lr_budget_steps"] = r["lrsel"].get("budget_steps")
        if met:
            row["heldout_loss_before"] = py((met.get("heldout_initial") or {}).get("total"))
            row["heldout_loss_after"] = py((met.get("heldout_final") or {}).get("total"))
            row["_loc"]["heldout_loss_before"] = json_loc(r["metrics_file"], ["heldout_initial", "total"])
            row["_loc"]["heldout_loss_after"] = json_loc(r["metrics_file"], ["heldout_final", "total"])
            for c in ("steps_done", "total_steps", "images_seen", "n_trainable", "train_loss_first",
                      "train_loss_last10_mean", "seconds_train", "diverged"):
                row[c] = py(met.get(c))
                if c != "diverged":
                    row["_loc"][c] = json_loc(r["metrics_file"], [c])
            viol = [v.get("max_rel_violation") for v in (met.get("constraint_checks") or {}).values()
                    if isinstance(v, dict) and "max_rel_violation" in v]
            row["max_constraint_violation"] = max(viol) if viol else None
            row["lr_final"] = py((met.get("config") or {}).get("lr"))
        row["_seed"] = (met or r["lrsel"] or {}).get("seed")
        row["_git"] = (met or r["lrsel"] or {}).get("git_hash") or (knn.get("git_hash") if knn else None)
        row["seed"], row["git_hash"] = row["_seed"], row["_git"]
        mi = (r["summary"] or {}).get("method_info") or {}
        if mi.get("mlp_params_kept") is not None and mi.get("attn_params_kept") is not None:
            row["free_matrix_params"] = int(mi["mlp_params_kept"]) + int(mi["attn_params_kept"])
            row["_derived"]["free_matrix_params"] = (f"method_info mlp_params_kept + attn_params_kept "
                                                     f"({r['dir']}/summary.json)")
        ops = ((cfg or default_cfg or {}).get("structure") or {}).get("operators")
        row["operators"] = ops
        row["fit_seed"] = row["_fit_seed"] = fit_seed_from_path(ops) if r["status"] != NOT_RUN else None
        row["source"] = "; ".join(x for x in (kf, r["metrics_file"] if met else None,
                                               r["lrsel_file"] if r["lrsel"] else None) if x)
        row["_src"] = row["source"] or r["dir"]
        if row.get("k") and m1["knn"] is not None:
            sel = m1["knn"][(m1["knn"]["k"] == row["k"]) & (m1["knn"]["variant"] == m1map[method])]
            if len(sel) == 1:
                row["m1_zero_shot_top1"] = float(sel["top1"].iloc[0])
                row["_loc"]["m1_zero_shot_top1"] = csv_loc(m1["knn_file"], {"k": row["k"], "variant": m1map[method]}, "top1")
        T.add(row)
    # P3 and gate G2
    G = Table("T_healing_p3", "P3 and gate G2 (pre-registered, PREREG_E2.md)", ["check"],
              [("check", "check", ""), ("statement", "statement", ""), ("value", "value", ".2f"),
               ("threshold", "threshold_text", ""), ("verdict", "verdict", ""), ("note", "note", "")])
    mir, svd, rnd = (T.find(method=m) for m in ("mirror", "svd", "random"))

    def healed(r):
        return r.get("healed_top1") if r else None

    def verdict(val, ok, missing):
        if val is None:
            return PENDING, "waiting for: " + ", ".join(missing)
        return (PASS if ok else FAIL), ""

    checks = []
    gits = ";".join(sorted({str(r["_git"]) for r in T.rows if r.get("_git")})) or None
    seeds = {r["_seed"] for r in T.rows if r.get("_seed") is not None}
    seed = seeds.pop() if len(seeds) == 1 else None
    hm, hs, hr = healed(mir), healed(svd), healed(rnd)
    orig = mir.get("original_top1") if mir else None
    miss = [f"{n} healed k-NN ({T.find(method=n)['status']})" for n, v in (("mirror", hm), ("svd", hs), ("random", hr))
            if v is None]
    a_val = hm - hr if hm is not None and hr is not None else None
    b_val = hm - hs if hm is not None and hs is not None else None
    c_val = abs(orig - hm) if hm is not None and orig is not None else None
    for cid, stmt, val, op, thr, need in (
            ("P3.a", "healed Mirror minus healed random-structure control", a_val, ">=", th.get("p3_random"),
             [m for m in miss if m.startswith(("mirror", "random"))]),
            ("P3.b", "healed Mirror minus healed SVD", b_val, ">=", th.get("p3_svd"),
             [m for m in miss if m.startswith(("mirror", "svd"))]),
            ("P3.c", "absolute difference between the original and healed Mirror", c_val, "<=", th.get("p3_orig"),
             [m for m in miss if m.startswith("mirror")])):
        ok = None if val is None or thr is None else (val >= thr if op == ">=" else val <= thr)
        v, note = verdict(val, ok, need)
        checks.append(G.add({"check": cid, "statement": stmt, "value": val, "comparison": op, "threshold": thr,
                             "threshold_text": f"{op} {fmt(thr, 'g')}", "verdict": v, "note": note,
                             "prereg_commit": prereg_commit, "_src": "T_healing", "_git": gits, "_seed": seed,
                             "_derived": {"value": "from T_healing healed_top1 (and original_top1)"}}))
    def combine(vs):  # all must pass: any failure fails the whole, otherwise pending until every value exists
        return PASS if all(v == PASS for v in vs) else (FAIL if FAIL in vs else PENDING)

    p3 = combine([c["verdict"] for c in checks])
    g2 = combine([checks[0]["verdict"], checks[1]["verdict"]])
    G.add({"check": "P3", "statement": "all three P3 bullets", "verdict": p3, "threshold_text": "all pass",
           "prereg_commit": prereg_commit, "_src": "T_healing_p3", "_git": gits, "_seed": seed})
    G.add({"check": "G2", "statement": "gate G2: P3.a and P3.b (else stop and discuss the framing)", "verdict": g2,
           "threshold_text": "both pass", "prereg_commit": prereg_commit, "_src": "T_healing_p3", "_git": gits,
           "_seed": seed})
    return T, G


def add_linear_to_healing(Th: Table, T2: Table) -> None:
    """Second measure next to k-NN: the Tier-2 linear probe of each healed student (information only; the P3
    verdict stays on k-NN, as pre-registered)."""
    Th.md_cols += [("linear: healed", "linear_top1", ".2f"), ("linear: original", "linear_original_top1", ".2f"),
                   ("Mirror minus this: k-NN", "mirror_minus_knn", ".2f"),
                   ("Mirror minus this: linear", "mirror_minus_linear", ".2f")]
    Th.notes.append("Linear: Tier-2 ImageNet linear probe top-1 (%) of the same healed student (T_tier2), a second "
                    "measure next to k-NN. Information only: the P3 verdict below stays on k-NN, as pre-registered.")
    for r in Th.rows:
        model, k = r.get("model"), r.get("k")
        if not model or not k:
            continue
        t = T2.find(model=model, variant=f"healed_{r['method']}_k{k}")
        o = T2.find(model=model, variant="original")
        if t is not None:
            for src, dst in (("top1", "linear_top1"), ("top5", "linear_top5"), ("chosen_wd", "linear_chosen_wd"),
                             ("knn_check_top1", "linear_knn_check")):
                if t.get(src) is not None:
                    r[dst] = t[src]
                    if t["_loc"].get(src):
                        r["_loc"][dst] = t["_loc"][src]
            r["linear_source"] = t["_src"]
        if o is not None and o.get("top1") is not None:
            r["linear_original_top1"] = o["top1"]
            if o["_loc"].get("top1"):
                r["_loc"]["linear_original_top1"] = o["_loc"]["top1"]
            if r.get("linear_top1") is not None:
                r["linear_drop"] = o["top1"] - r["linear_top1"]
                r["_derived"]["linear_drop"] = "linear top1 of the original minus that of the healed student (T_tier2)"
    mir = Th.find(method="mirror")
    if mir is None:
        return
    for r in Th.rows:
        if r is mir:
            continue
        if r.get("healed_top1") is not None and mir.get("healed_top1") is not None:
            r["mirror_minus_knn"] = mir["healed_top1"] - r["healed_top1"]
            r["_derived"]["mirror_minus_knn"] = "healed k-NN top1 of mirror minus this method's"
        if r.get("linear_top1") is not None and mir.get("linear_top1") is not None:
            r["mirror_minus_linear"] = mir["linear_top1"] - r["linear_top1"]
            r["_derived"]["mirror_minus_linear"] = "healed linear top1 of mirror minus this method's (T_tier2)"


VEXACT_JOB_RE = re.compile(r"scripts/heal_vexact\.py[^\n']*?--k\s+(\d+)")


def vexact_planned_ks(P: Paths) -> tuple[list[int], str]:
    """k values of the V-exact healing jobs (jobs/*.txt), else the k of configs/m3_vexact.yaml."""
    ks, src = [], ""
    if P.jobs.is_dir():
        for f in sorted(P.jobs.glob("*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for m in VEXACT_JOB_RE.finditer(line):
                    if int(m[1]) not in ks:
                        ks.append(int(m[1]))
                        src = P.rel(f)
    if not ks:
        cfg = load_cfg(P, "configs/m3_vexact.yaml")
        if cfg.get("k") is not None:
            ks, src = [int(cfg["k"])], "configs/m3_vexact.yaml"
    return ks, src


def build_healing_vexact(P, m1: dict, depths: dict | None = None) -> tuple[Table, dict]:
    """V-exact healing (scripts/heal_vexact.py): one row per k, from runs/m3_vexact/k<k>/ (knn.json,
    exactness.json, flipcheck.json for k = depth, final/metrics.json, lr_selection/lr_selection.json, summary.json;
    each part read from its own file, else from summary.json). Planned k without files: "not run"."""
    T = Table("T_healing_vexact", "V-exact healing (runs/m3_vexact/k<k>/; companion of T_healing)", ["k"],
              [("k", "k", "d"), ("status", "status", ""), ("original", "original_top1", ".2f"),
               ("zero-shot", "zero_shot_top1", ".2f"), ("healed", "healed_top1", ".2f"),
               ("flip: zero-shot (k-NN)", "zero_shot_flip", ".2f"), ("flip: healed (k-NN)", "healed_flip", ".2f"),
               ("flip: fp32 check", lambda r: (f"{fmt(r.get('flip_fp32'), '.2f')} ({'pass' if r.get('flip_fp32_pass') else 'FAIL'})"
                                               if r.get("flip_fp32") is not None else ""), ""),
               ("exactness (tol)", lambda r: (f"{fmt(r.get('exact_max_rel'), '.1e')} ({fmt(r.get('exact_tol'), 'g')}, "
                                              f"{'pass' if r.get('exact_pass') else 'FAIL'})"
                                              if r.get("exact_max_rel") is not None else ""), ""),
               ("chosen LR", "chosen_lr", "g"), ("held-out loss before", "heldout_loss_before", ".4f"),
               ("after", "heldout_loss_after", ".4f"), ("kept (converted)", "kept_frac_converted", ".3f"),
               ("model params kept", "model_params_frac", ".3f"), ("M1 zero-shot (same k)", "m1_zero_shot_top1", ".2f"),
               ("GPU", "gpu", ""), ("git", lambda r: short(r.get("git_hash")), "")])
    T.notes = ["V-exact student: one operator A* over blocks 1 to k, MLP and attention, exactly mirror-equivariant over "
               "the span (k = depth is the full conversion, with the even head), healed by distillation with the M3 "
               "pilot's recipe (scripts/heal_vexact.py). Top-1 and flip consistency in percent from the k-NN after "
               "healing (protocol v2; feature precision in the knn_precision column). fp32 check: flip consistency of the "
               "fp32 flip check after healing (k = depth only; it must be 100%). Exactness: largest relative error "
               "inside the span after healing (exactness.json), with the tolerance. Kept: matrix parameters of the "
               "converted blocks over dense; model params kept: whole model over the original (summary.json "
               "method_info). M1 zero-shot: the vexact row of runs/m1/m1_knn.csv at the same k (a cross-check).",
               "Held-out losses: the targets here are symmetrised (teacher block outputs centred and "
               "mirror-symmetrised with A*; at k = depth the final features mirror-averaged; columns block_targets and "
               "final_target), so these held-out losses are not comparable with the M3 pilot's in T_healing."]
    base = P.runs / "m3_vexact"
    planned, plan_src = vexact_planned_ks(P)
    found = {}
    for d in sorted(base.glob("k*")) if base.is_dir() else []:
        m = re.fullmatch(r"k(\d+)", d.name)
        if m and d.is_dir() and any(d.iterdir()):
            found[int(m[1])] = d
    info = {"dir": P.rel(base), "planned": planned, "plan_source": plan_src, "found": sorted(found)}
    cfg_model = ((load_cfg(P, "configs/m3_vexact.yaml").get("model") or {}).get("name"))
    for k in sorted(set(planned) | set(found)):
        d = found.get(k, base / f"k{k}")
        row = {"k": k, "dir": P.rel(d), "_loc": {}, "_derived": {}}
        if k not in found:
            row.update({"status": NOT_RUN, "note": f"no files under `{P.rel(d)}/`; planned in `{plan_src}`", "_src": P.rel(d),
                        "model": cfg_model, "depth": (depths or {}).get(cfg_model)})
            if row["depth"] is not None:
                row["full_conversion"] = k == row["depth"]
            T.add(row)
            continue
        sf = P.rel(d / "summary.json")
        summ = read_json(d / "summary.json")

        def part(name, key):
            j = read_json(d / name)
            if j is not None:
                return j, P.rel(d / name), []
            if summ and isinstance(summ.get(key), dict):
                return summ[key], sf, [key]
            return None, None, []

        knn, kf, kp = part("knn.json", "knn")
        ex, ef, ep = part("exactness.json", "exactness")
        fl, ff, fp = part("flipcheck.json", "flipcheck")
        met, mf, mp = part("final/metrics.json", "final")
        sel, lf, lp = part("lr_selection/lr_selection.json", "lr_selection")
        cfg = (summ or {}).get("config") or (met or {}).get("config") or {}
        tiny = bool(cfg.get("tiny_mode")) or (knn or {}).get("protocol") == "tiny synthetic"
        row["tiny"] = tiny
        row["model"] = "tiny random ViT (smoke test)" if tiny else (cfg.get("model") or {}).get("name")
        row["depth"] = py((summ or {}).get("depth"))
        if row["depth"] is not None:
            row["full_conversion"] = k == row["depth"]
        knn = knn or {}
        for part_, col in (("original", "original"), ("zero_shot", "zero_shot"), ("healed", "healed")):
            if isinstance(knn.get(part_), dict):
                for c, s in (("top1", "top1"), ("flip", "flip_consistency"), ("drop", "drop_points")):
                    if knn[part_].get(s) is not None:
                        row[f"{col}_{c}"] = py(knn[part_][s])
                        row["_loc"][f"{col}_{c}"] = json_loc(kf, kp + [part_, s])
        row["knn_amp"], row["knn_protocol"] = knn.get("amp"), knn.get("protocol")
        row["gpu"] = (((summ or {}).get("env") or {}).get("gpu")) or (((met or {}).get("env") or {}).get("gpu"))
        dev = str(((summ or {}).get("env") or {}).get("device") or "")
        # as scripts/heal_vexact.py's SUMMARY.md: the bf16 autocast applies on CUDA only
        row["knn_precision"] = (knn.get("amp") or "fp32") if (dev.startswith("cuda") or not dev) else "fp32"
        if ex:
            for c, s in (("exact_max_rel", "max_rel_inside"), ("exact_tol", "tol"), ("exact_pass", "pass"),
                         ("exact_n_images", "n_images")):
                if s in ex:
                    row[c] = py(ex[s])
                    row["_loc"][c] = json_loc(ef, ep + [s])
            row["exact_dtype"] = str(ex.get("dtype", "")).replace("torch.", "")
        if fl:
            for c, s in (("flip_fp32", "flip_consistency"), ("flip_fp32_pass", "pass"),
                         ("flip_fp32_cls_rel_max", "cls_feature_rel_max"), ("flip_fp32_n_query", "n_query")):
                if s in fl:
                    row[c] = py(fl[s])
                    row["_loc"][c] = json_loc(ff, fp + [s])
            row["flip_fp32_dtype"] = str(fl.get("dtype", "")).replace("torch.", "")
        if sel:
            row["chosen_lr"] = py(sel.get("best_lr"))
            row["_loc"]["chosen_lr"] = json_loc(lf, lp + ["best_lr"])
            row["lr_trials"] = "; ".join(f"{t['lr']:g}: {t['heldout_loss']:.4f}" for t in sel.get("trials", []))
            row["lr_budget_steps"] = sel.get("budget_steps")
            row["lr_skipped"] = bool(sel.get("skipped"))
        if met:
            for c, path in (("heldout_loss_before", ["heldout_initial", "total"]),
                            ("heldout_loss_after", ["heldout_final", "total"])):
                v = met
                for p_ in path:
                    v = (v or {}).get(p_)
                if v is not None:
                    row[c] = py(v)
                    row["_loc"][c] = json_loc(mf, mp + path)
            for c in ("steps_done", "total_steps", "images_seen", "n_trainable", "heldout_images", "seconds_train"):
                if met.get(c) is not None:
                    row[c] = py(met[c])
                    row["_loc"][c] = json_loc(mf, mp + [c])
            row["diverged"] = py(met.get("diverged"))
            viol = [v.get("max_rel_violation") for v in (met.get("constraint_checks") or {}).values()
                    if isinstance(v, dict) and "max_rel_violation" in v]
            row["max_constraint_violation"] = max(viol) if viol else None
        mi = (summ or {}).get("method_info") or {}
        for c, s in (("kept_frac_converted", "converted_kept_frac"), ("model_params_frac", "model_params_frac"),
                     ("model_params", "model_params"), ("converted_params_kept", "converted_params_kept"),
                     ("converted_params_dense", "converted_params_dense"), ("d_plus", "d_plus"), ("d_minus", "d_minus")):
            if mi.get(s) is not None:
                row[c] = py(mi[s])
                row["_loc"][c] = json_loc(sf, ["method_info", s])
        tch = (summ or {}).get("teacher") or {}
        row["block_targets"], row["final_target"] = tch.get("block_targets"), tch.get("final_target")
        row["embeddings"] = (summ or {}).get("embeddings")
        st = (summ or {}).get("structure") or {}
        row["span_source"], row["operators_source"] = st.get("span_source"), st.get("operators_source")
        ops = ((cfg.get("structure") or {}).get("operators"))
        row["fit_seed"] = None if tiny else fit_seed_from_path(ops)
        row["_fit_seed"] = row["fit_seed"]
        row["seed"] = py(next((x.get("seed") for x in (summ, knn, met, sel) if x and x.get("seed") is not None), None))
        row["git_hash"] = next((x.get("git_hash") for x in (knn, summ, met, sel, ex) if x and x.get("git_hash")), None)
        row["git_dirty"] = py((summ or met or sel or {}).get("git_dirty"))
        row["_seed"], row["_git"] = row["seed"], row["git_hash"]
        if row.get("healed_top1") is not None and ex:
            need_flip = row.get("full_conversion") and (cfg.get("checks") or {}).get("flip_check", True)
            row["status"] = "complete" if (fl or not need_flip) else "complete except the fp32 flip check"
        elif met:
            row["status"] = "healed; final k-NN not yet available" if row.get("healed_top1") is None else "healed; checks pending"
        elif sel:
            row["status"] = "learning-rate selection only"
        else:
            row["status"] = "started (no result files yet)"
        if tiny:
            row["status"] += " (TINY smoke test, not a result)"
        row["source"] = "; ".join(x for x in dict.fromkeys((kf, ef, ff, mf, lf, sf if summ else None)) if x)
        row["_src"] = row["source"] or P.rel(d)
        if m1.get("knn") is not None and m1.get("model") == ((cfg.get("model") or {}).get("name")) and not tiny:
            sel_ = m1["knn"][(m1["knn"]["k"] == k) & (m1["knn"]["variant"] == "vexact")]
            if len(sel_) == 1:
                row["m1_zero_shot_top1"] = float(sel_["top1"].iloc[0])
                row["_loc"]["m1_zero_shot_top1"] = csv_loc(m1["knn_file"], {"k": k, "variant": "vexact"}, "top1")
        T.add(row)
    return T, info


# ================================================================================================
# T_generality and P4
# ================================================================================================
def vexact_device_notes(Tv: Table, Tm: Table) -> None:
    """A V-exact healing row run on another GPU than the main runs measures its own original; compare within the row."""
    for r in Tv.rows:
        o = Tm.find(model=r.get("model"), k=0, variant="original") if r.get("model") else None
        if o is None or o.get("status") != "ok" or r.get("original_top1") is None or o.get("top1_mean") is None:
            continue
        if abs(r["original_top1"] - o["top1_mean"]) > 1e-9:
            r["note"] = (f"run on {r.get('gpu') or 'another GPU'}: its original measures {r['original_top1']:.3f} there "
                         f"against {o['top1_mean']:.3f} in the main runs, so compare its healed and zero-shot values with "
                         "its own original")
            Tv.notes.append(f"k = {r['k']}: {r['note']} (run-to-run spread between GPUs; zero-shot "
                            + (f"{r['zero_shot_top1']:.3f} against M1 {r['m1_zero_shot_top1']:.3f}"
                               if r.get("zero_shot_top1") is not None and r.get("m1_zero_shot_top1") is not None else "")
                            + ").")


def first_half(run):
    L = int(run["mlp"]["block"].max())
    return 1, L // 2


def build_generality(P, gen_runs, th, prereg_commit, plan_text):
    T = Table("T_generality", "G: generality across models (protocol v2, per-layer operators, zero-shot)",
              ["model", "fit_seed"],
              [("model", "model_label", ""), ("fit seed", "fit_seed", "d"), ("blocks", "depth", "d"),
               ("k = L/2", "k_half", "d"), ("outcome", "outcome", ""),
               ("1: e(null)/e(A)", lambda r: f"{fmt(r.get('row1_value'), '.2f')} {r.get('row1_cell', '')}", ""),
               ("2: e(A)/e(I)", lambda r: f"{fmt(r.get('row2_value'), '.3f')} {r.get('row2_cell', '')}", ""),
               ("3: d-/d", lambda r: f"{fmt(r.get('row3_value'), '.3f')} {r.get('row3_cell', '')}", ""),
               ("4: share", lambda r: f"{fmt(r.get('row4_value'), '.3f')} {r.get('row4_cell', '')}", ""),
               ("5: drop k=L/2", lambda r: f"{fmt(r.get('row5_value'), '.2f')} {r.get('row5_cell', '')}", ""),
               ("d+ / d-", lambda r: f"{fmt(r.get('d_plus_mean'), '.0f')} / {fmt(r.get('d_minus_mean'), '.0f')}"
                if r.get("d_plus_mean") is not None else "", ""),
               ("twin fraction", "twin_frac", ".3f"),
               ("s1 (null)", lambda r: f"{fmt(r.get('s1'), '.3f')} ({fmt(r.get('s1_null'), '.3f')})", ""),
               ("s2 (null)", lambda r: f"{fmt(r.get('s2'), '.3f')} ({fmt(r.get('s2_null'), '.3f')})", ""),
               ("MLP saving", "mlp_saving_first_half", ".3f"),
               ("original", "top1_original", ".2f"), ("surgery", "top1_surgery_half", ".2f"),
               ("SVD", "top1_svd_half", ".2f"), ("margin", "margin_svd_half", ".2f"), ("note", "note", "")])
    T.notes = ["Rows 1 to 5 are the E1b decision table (blocks 1 to L/2; GO limits: row 1 >= 5, row 2 <= 0.5, row 3 "
               ">= 0.2, row 4 <= 0.2 with evidence, row 5 drop <= 10 and margin over SVD >= 5). d+ / d-: mean over "
               "layers 1 to L/2. Twin fraction, s1, s2 (matched nulls) and MLP saving (param_saving_frac): means "
               "over blocks 1 to L/2. Surgery, SVD and margin: zero-shot k-NN top-1 (%) at k = L/2. L is each "
               "model's own depth (blocks column), so k = L/2 is 12 for a 24-block model."]
    mae_note = ""
    m = re.search(r"\*\*MAE:\*\*\s*([^\n]+)", plan_text or "")
    if m:
        mae_note = "Plan (PAPER_EXPERIMENTS.md G): " + m[1].strip()
    for run in sorted(gen_runs, key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                                               r["fit_seed"] if r["fit_seed"] is not None else 99)):
        if run["layers"] is None or run["mlp"] is None:
            continue
        lo, hi = first_half(run)
        L = run["layers"][(run["layers"]["layer"] >= lo) & (run["layers"]["layer"] <= hi)]
        M = run["mlp"][(run["mlp"]["block"] >= lo) & (run["mlp"]["block"] <= hi)]
        row = {"model": run["model"], "model_label": MODEL_LABEL.get(run["model"], run["model"]),
               "fit_seed": run["fit_seed"], "run": run["dir"], "status": run["status"], "protocol": run["protocol"],
               "blocks": f"{lo}-{hi}", "_src": run["dir"], "_seed": run["seed"], "_fit_seed": run["fit_seed"],
               "_git": run["git"], "git_hash": run["git"], "decision_source": run["decision_source"],
               "decision_matches_meta": run["decision_matches_meta"], "note": mae_note if run["model"] == "mae_b16" else "",
               "_loc": {}, "_derived": {}}
        dec = run["decision"]
        if dec:
            row["outcome"] = dec["outcome"]
            from_meta = run["decision_source"] == "e1_meta.json"
            if from_meta:
                row["_loc"]["outcome"] = json_loc(run["meta_file"], ["decision", "outcome"])
            for i, rr in enumerate(dec["rows"]):
                n = rr["row"]
                row[f"row{n}_value"], row[f"row{n}_cell"] = py(rr["value"]), rr["cell"]
                if from_meta:
                    row["_loc"][f"row{n}_value"] = json_loc(run["meta_file"], ["decision", "rows", i, "value"])
                    row["_loc"][f"row{n}_cell"] = json_loc(run["meta_file"], ["decision", "rows", i, "cell"])
                else:
                    row["_derived"][f"row{n}_value"] = "mirror.report.decide on the run's CSVs (no e1_meta.json)"
            w = (dec.get("worst") or {}).get("row1")
            if w:
                row["row1_worst_layer"], row["row1_worst_value"] = w["layer"], py(w["r1"])
        row["d_plus_mean"], row["d_minus_mean"] = float(L["d_plus"].mean()), float(L["d_minus"].mean())
        row["dminus_frac"] = float(L["dminus_frac"].mean())
        for c in ("twin_frac", "s1", "s2", "s1_null", "s2_null"):
            row[c] = float(M[c].mean())
        row["mlp_saving_first_half"] = float(M["param_saving_frac"].mean())
        row["p4_share"] = max(row["s1"], row["s2"])
        row["p4_ratio"] = float(L["rel_e_null_used"].mean() / L["rel_e_A_used"].mean())
        for c in ("d_plus_mean", "d_minus_mean", "dminus_frac", "twin_frac", "s1", "s2", "s1_null", "s2_null",
                  "mlp_saving_first_half"):
            row["_derived"][c] = f"mean over layers or blocks {lo} to {hi} of {run['layers_file'] if c.startswith(('d_', 'dminus')) else run['mlp_file']}"
        row["_derived"]["p4_share"] = f"max(mean s1, mean s2) over blocks {lo} to {hi} ({run['mlp_file']})"
        row["_derived"]["p4_ratio"] = (f"mean rel_e_null_used / mean rel_e_A_used over layers {lo} to {hi} "
                                       f"({run['layers_file']})")
        kh = hi                     # k = L/2 for every model (6 for the 12-block models, 12 for DINOv2-L)
        row["depth"] = int(run["mlp"]["block"].max())
        if run["surgery"] is not None:
            s = run["surgery"]
            for name, var, kk in (("original", "original", 0), ("surgery_half", "surgery", kh),
                                  ("svd_half", "svd", kh), ("random_half", "random", kh)):
                sel = s[(s["k"] == kk) & (s["variant"] == var)]
                if len(sel) == 1:
                    row[f"top1_{name}"] = float(sel["top1"].iloc[0])
                    row["_loc"][f"top1_{name}"] = csv_loc(run["surgery_file"], {"k": kk, "variant": var}, "top1")
                    if name == "surgery_half":
                        row["flip_surgery_half"] = float(sel["flip_consistency"].iloc[0])
                        row["mlp_params_kept_frac_half"] = float(sel["mlp_params_kept_frac"].iloc[0])
                        row["_loc"]["mlp_params_kept_frac_half"] = csv_loc(run["surgery_file"], {"k": kk, "variant": var},
                                                                           "mlp_params_kept_frac")
            if row.get("top1_surgery_half") is not None and row.get("top1_svd_half") is not None:
                row["margin_svd_half"] = row["top1_surgery_half"] - row["top1_svd_half"]
        row["k_half"] = kh
        T.add(row)

    G = Table("T_generality_p4", "P4 (pre-registered, PREREG_E2.md): fit seed 0 runs", ["check"],
              [("check", "check", ""), ("statement", "statement", ""), ("value", "value", ".3f"),
               ("threshold", "threshold_text", ""), ("verdict", "verdict", ""), ("note", "note", "")])
    s0 = {r["model"]: r for r in T.rows if r["fit_seed"] == 0}
    for mid in ("deit3_b16", "mae_b16"):
        r = s0.get(mid)
        lab = MODEL_LABEL[mid]
        if r is None:
            for part in ("share", "ratio", "all"):
                G.add({"check": f"P4.{MODEL_SHORT[mid]}.{part}", "statement": f"{lab}: {part}", "verdict": NOT_RUN,
                       "note": "no G run with fit seed 0", "prereg_commit": prereg_commit, "_src": ""})
            continue
        a = r["p4_share"] <= th["p4_share"] if th.get("p4_share") is not None else None
        b = r["p4_ratio"] >= th["p4_ratio"] if th.get("p4_ratio") is not None else None
        same = (r.get("row4_value") is not None and abs(r["row4_value"] - r["p4_share"]) < 1e-12 and
                r.get("row1_value") is not None and abs(r["row1_value"] - r["p4_ratio"]) < 1e-9 * r["p4_ratio"])
        note = "equals decision rows 4 and 1" if same else "differs from decision rows 4 and 1 (check aggregation)"
        G.add({"check": f"P4.{MODEL_SHORT[mid]}.share", "statement": f"{lab}: mean MLP breaking share (max of mean s1, "
               f"mean s2) over blocks {r['blocks']}", "value": r["p4_share"], "threshold": th.get("p4_share"),
               "threshold_text": f"<= {fmt(th.get('p4_share'), 'g')}", "verdict": PASS if a else FAIL, "note": note,
               "prereg_commit": prereg_commit, "_src": r["run"], "_git": r["_git"], "_seed": r["_seed"],
               "_fit_seed": 0, "_derived": {"value": r["_derived"]["p4_share"]}})
        G.add({"check": f"P4.{MODEL_SHORT[mid]}.ratio", "statement": f"{lab}: e(null)/e(A) over layers {r['blocks']}",
               "value": r["p4_ratio"], "threshold": th.get("p4_ratio"),
               "threshold_text": f">= {fmt(th.get('p4_ratio'), 'g')}", "verdict": PASS if b else FAIL, "note": note,
               "prereg_commit": prereg_commit, "_src": r["run"], "_git": r["_git"], "_seed": r["_seed"],
               "_fit_seed": 0, "_derived": {"value": r["_derived"]["p4_ratio"]}})
        G.add({"check": f"P4.{MODEL_SHORT[mid]}.all", "statement": f"{lab}: both bullets",
               "verdict": PASS if (a and b) else FAIL, "threshold_text": "both pass", "prereg_commit": prereg_commit,
               "_src": r["run"], "_git": r["_git"], "_seed": r["_seed"], "_fit_seed": 0})
    sg, rg = s0.get("siglip2_b16"), s0.get("dinov2_vitb14_reg")
    if sg and rg:
        G.add({"check": "P4.siglip2", "statement": "SigLIP2 breaking share minus DINOv2-reg breaking share "
               "(row 4 definition, fit seed 0)", "value": sg["p4_share"] - rg["p4_share"], "threshold": 0.0,
               "threshold_text": "> 0", "verdict": PASS if sg["p4_share"] > rg["p4_share"] else FAIL,
               "note": f"SigLIP2 {sg['p4_share']:.3f}, DINOv2-reg {rg['p4_share']:.3f}", "prereg_commit": prereg_commit,
               "_src": f"{sg['run']};{rg['run']}", "_git": f"{sg['_git']};{rg['_git']}",
               "_seed": sg["_seed"] if sg["_seed"] == rg["_seed"] else None, "_fit_seed": 0,
               "_derived": {"value": "p4_share(SigLIP2) - p4_share(DINOv2-reg)"}})
    else:
        G.add({"check": "P4.siglip2", "statement": "SigLIP2 breaking share higher than DINOv2-reg's",
               "verdict": NOT_RUN, "note": "needs both fit seed 0 runs", "prereg_commit": prereg_commit, "_src": ""})
    return T, G


# ================================================================================================
# T_structure_per_layer (F1 style)
# ================================================================================================
def build_structure(P, struct_runs):
    T = Table("T_structure_per_layer", "Structure per layer and block (F1 style; protocol v2, per-layer operators)",
              ["model", "fit_seed", "layer"],
              [("layer", "layer", "d"), ("d+", "d_plus", "d"), ("d-", "d_minus", "d"), ("e(A_l)", "e_A", ".4f"),
               ("e(I)", "e_I", ".4f"), ("e(null)", "e_null", ".4f"), ("e(null)/e(A_l)", "fit_ratio", ".2f"),
               ("block", "block", "d"), ("pairs", "n_pairs", "d"), ("self", "n_self", "d"),
               ("twin frac", "twin_frac", ".3f"), ("twin corr", "mean_twin_corr", ".3f"),
               ("s1 (null)", lambda r: f"{fmt(r.get('s1'), '.3f')} ({fmt(r.get('s1_null'), '.3f')})"
                if r.get("s1") is not None else "", ""),
               ("s2 (null)", lambda r: f"{fmt(r.get('s2'), '.3f')} ({fmt(r.get('s2_null'), '.3f')})"
                if r.get("s2") is not None else "", ""),
               ("tax (formula)", "tax_formula", ".3f"), ("saving", "saving_realised", ".3f")])
    T.notes = ["Layer l is the output of block l (layer 0 enters block 1); block columns refer to block l (from layer "
               "l - 1 to layer l). e(.): relative held-out error e / ||X~||^2 of A_l, the identity and the matched "
               "null (validation images). Twin corr: mean twin score of paired neurons. Tax (formula): mean over fc1 "
               "and fc2 of 1/2 - f (d+ - d-) / (2 h d) with f the self-twins, fc1 using the input layer's operator "
               "and fc2 the output layer's (PROJECT_CONTEXT.md 5.3). Saving: param_saving_frac of e1_mlp.csv."]
    for run in struct_runs:
        lay, mlp = run["layers"], run["mlp"]
        li = lay.set_index("layer")
        for _, lr in lay.iterrows():
            l = int(lr["layer"])
            row = {"model": run["model"], "model_label": MODEL_LABEL.get(run["model"], run["model"]),
                   "fit_seed": run["fit_seed"], "run": run["dir"], "layer": l, "d": int(lr["d"]),
                   "d_plus": int(lr["d_plus"]), "d_minus": int(lr["d_minus"]), "dminus_frac": float(lr["dminus_frac"]),
                   "e_A": float(lr["rel_e_A_layer"]), "e_I": float(lr["rel_e_I"]), "e_null": float(lr["rel_e_null_layer"]),
                   "_src": run["layers_file"], "_seed": run["seed"], "_fit_seed": run["fit_seed"], "_git": run["git"],
                   "git_hash": run["git"], "_loc": {}, "_derived": {}}
            row["fit_ratio"] = row["e_null"] / row["e_A"] if row["e_A"] > 0 else None
            row["ratio_A_over_I"] = row["e_A"] / row["e_I"] if row["e_I"] > 0 else None
            for c, s in (("d_plus", "d_plus"), ("d_minus", "d_minus"), ("e_A", "rel_e_A_layer"), ("e_I", "rel_e_I"),
                         ("e_null", "rel_e_null_layer"), ("dminus_frac", "dminus_frac")):
                row["_loc"][c] = csv_loc(run["layers_file"], {"layer": l}, s)
            sel = mlp[mlp["block"] == l]
            if l >= 1 and len(sel) == 1:
                br = sel.iloc[0]
                h, d, f = int(br["h"]), int(br["d"]), int(br["n_self"])
                tr_in = int(li.loc[l - 1, "d_plus"] - li.loc[l - 1, "d_minus"])
                tr_out = int(li.loc[l, "d_plus"] - li.loc[l, "d_minus"])
                t1, t2 = tax_formula(h, d, f, tr_in), tax_formula(h, d, f, tr_out)
                row.update({"block": l, "h": h, "n_pairs": int(br["n_pairs"]), "n_self": f,
                            "twin_frac": float(br["twin_frac"]), "mean_twin_corr": float(br["mean_score_pairs"]),
                            "mean_self_corr": float(br["mean_score_self"]), "s1": float(br["s1"]),
                            "s1_null": float(br["s1_null"]), "s2": float(br["s2"]), "s2_null": float(br["s2_null"]),
                            "tax_fc1": t1, "tax_fc2": t2, "tax_formula": 0.5 * (t1 + t2),
                            "saving_realised": float(br["param_saving_frac"]),
                            "eq_params_fc1": int(br["eq_params_fc1"]), "eq_params_fc2": int(br["eq_params_fc2"]),
                            "trace_matches_csv": bool(tr_in == int(br["tr_A_in"]) and tr_out == int(br["tr_A_out"]))})
                row["tax_minus_saving"] = row["tax_formula"] - row["saving_realised"]
                for c, s in (("n_pairs", "n_pairs"), ("n_self", "n_self"), ("twin_frac", "twin_frac"),
                             ("mean_twin_corr", "mean_score_pairs"), ("s1", "s1"), ("s1_null", "s1_null"),
                             ("s2", "s2"), ("s2_null", "s2_null"), ("saving_realised", "param_saving_frac")):
                    row["_loc"][c] = csv_loc(run["mlp_file"], {"block": l}, s)
                row["_src"] = f"{run['layers_file']};{run['mlp_file']}"
                row["_derived"]["tax_formula"] = "mirror.findings.tax_formula, mean of fc1 and fc2"
            T.add(row)
    # summary over first and second half, mean and std over fit seeds
    S = Table("T_structure_summary", "Structure summary: means over the first and second half of the blocks",
              ["model", "blocks"],
              [("model", "model_label", ""), ("blocks", "blocks", ""), ("n seeds", "n_seeds", "d"),
               ("tax (formula)", lambda r: pm(r.get("tax_formula_mean"), r.get("tax_formula_std"), spec=".3f"), ""),
               ("saving", lambda r: pm(r.get("saving_realised_mean"), r.get("saving_realised_std"), spec=".3f"), ""),
               ("twin frac", lambda r: pm(r.get("twin_frac_mean"), r.get("twin_frac_std"), spec=".3f"), ""),
               ("s1", lambda r: pm(r.get("s1_mean"), r.get("s1_std"), spec=".3f"), ""),
               ("s1 null", lambda r: pm(r.get("s1_null_mean"), r.get("s1_null_std"), spec=".3f"), ""),
               ("s2", lambda r: pm(r.get("s2_mean"), r.get("s2_std"), spec=".3f"), ""),
               ("d-/d", lambda r: pm(r.get("dminus_frac_mean"), r.get("dminus_frac_std"), spec=".3f"), ""),
               ("e(A_l)", lambda r: pm(r.get("e_A_mean"), r.get("e_A_std"), spec=".4f"), "")])
    S.notes = ["Per fit seed: mean over the blocks (and layers with the same numbers) of the range; then mean and "
               "sample standard deviation over fit seeds."]
    for model in MODEL_ORDER:
        rows = [r for r in T.rows if r["model"] == model]
        if not rows:
            continue
        depth = max(r.get("block") or 0 for r in rows)
        for lo, hi in ((1, depth // 2), (depth // 2 + 1, depth)):
            seeds = sorted({r["fit_seed"] for r in rows}, key=lambda x: (x is None, x))
            per = {c: [] for c in ("tax_formula", "saving_realised", "twin_frac", "s1", "s1_null", "s2", "s2_null",
                                   "dminus_frac", "e_A")}
            for s in seeds:
                rr = [r for r in rows if r["fit_seed"] == s and lo <= r["layer"] <= hi]
                for c in per:
                    per[c].append(float(np.mean([r[c] for r in rr])))
            out = {"model": model, "model_label": MODEL_LABEL.get(model, model), "blocks": f"{lo}-{hi}",
                   "n_seeds": len(seeds), "fit_seeds": ",".join(map(str, seeds)),
                   "_src": ";".join(sorted({r["_src"] for r in rows})), "_fit_seed": ",".join(map(str, seeds)),
                   "_seed": (lambda ss: ss.pop() if len(ss) == 1 else None)({r["_seed"] for r in rows}),
                   "_git": ";".join(sorted({str(r["_git"]) for r in rows}))}
            for c, vals in per.items():
                m_, s_, _ = agg(vals)
                out[f"{c}_mean"], out[f"{c}_std"] = m_, s_
                out["_derived"] = out.get("_derived", {})
                out["_derived"][f"{c}_mean"] = (f"per fit seed the mean over blocks {lo} to {hi} of {c} "
                                                "(T_structure_per_layer), then the mean over seeds")
            S.add(out)
    return T, S


# ================================================================================================
# T_e1_history (protocol v1)
# ================================================================================================
def build_e1_history(P, e1, e1b, diag, commits):
    T = Table("T_e1_history", "E1 and E1b (protocol v1: earlier protocol, all splits from ImageNet validation)",
              ["run_id"],
              [("run", "run_id", ""), ("role", "role", ""), ("model", "model", ""), ("operator", "operator_mode", ""),
               ("outcome", "outcome", ""), ("1", lambda r: f"{fmt(r.get('row1_value'), '.2f')} {r.get('row1_cell', '')}", ""),
               ("2", lambda r: f"{fmt(r.get('row2_value'), '.3f')} {r.get('row2_cell', '')}", ""),
               ("3", lambda r: f"{fmt(r.get('row3_value'), '.3f')} {r.get('row3_cell', '')}", ""),
               ("4", lambda r: f"{fmt(r.get('row4_value'), '.3f')} {r.get('row4_cell', '')}", ""),
               ("5", lambda r: f"{fmt(r.get('row5_value'), '.2f')} {r.get('row5_cell', '')}", ""),
               ("original", "original_top1", ".2f"),
               ("surgery k=3/6/9", lambda r: " / ".join(fmt(r.get(f"surgery_k{k}"), ".2f") or "-" for k in (3, 6, 9)), ""),
               ("SVD k=3/6/9", lambda r: " / ".join(fmt(r.get(f"svd_k{k}"), ".2f") or "-" for k in (3, 6, 9)), ""),
               ("random k=3/6/9", lambda r: " / ".join(fmt(r.get(f"random_k{k}"), ".2f") or "-" for k in (3, 6, 9)), ""),
               ("git", lambda r: short(r.get("git_hash")), "")])
    T.notes = ["Kept unchanged from the E1 and E1b reports; protocol v1 used ImageNet validation images for the fit, "
               "held-out, bank and query splits (fit 2,000, held-out 1,000, bank 39,000, queries 8,000, from "
               "split_sizes in e1_meta.json), so these numbers are not comparable with the protocol v2 tables.",
               "The per-layer diagnostic was chosen after seeing E1 (post hoc); its decision rows are recomputed with "
               "mirror.report.decide from results/e1_layers.csv and the diagnostic CSVs, and carry no verdict."]
    entries = []
    if e1 and e1["status"] != NOT_RUN:
        entries.append(("E1", "pre-registered (E1_INSTRUCTIONS.md)", e1, {"surgery": "surgery", "svd": "svd",
                                                                          "random": "random"}))
    for name, r in sorted(e1b.items()):
        if r["status"] == NOT_RUN:
            continue
        first = ""
        if r["config_path"] and (P.root / r["config_path"]).is_file():
            first = (P.root / r["config_path"]).read_text(encoding="utf-8").splitlines()[0]
        m = re.search(r"E1b (PRIMARY|SECONDARY|EXPLORATORY)", first)
        role = (m[1].lower() if m else "arm") + " (E1B_PREREG.md)"
        entries.append((f"E1b {name.replace('e1b_', '')}", role, r, {"surgery": "surgery", "svd": "svd", "random": "random"}))
    for run_id, role, r, vmap in entries:
        row = {"run_id": run_id, "role": role, "protocol": "v1", "model": r["model"], "operator_mode": r["operator_mode"],
               "date": r["date"], "git_hash": r["git"], "git_dirty": r["git_dirty"], "seed": r["seed"],
               "split_seed": (r["cfg"].get("data") or {}).get("split_seed"),
               "split_sizes": json.dumps(r["split_sizes"]) if r["split_sizes"] else None,
               "_src": r["dir"], "_seed": r["seed"], "_git": r["git"], "_loc": {}, "_derived": {}}
        dec = r["decision"]
        if dec:
            row["outcome"] = dec["outcome"]
            row["_loc"]["outcome"] = json_loc(r["meta_file"], ["decision", "outcome"])
            for i, rr in enumerate(dec["rows"]):
                row[f"row{rr['row']}_value"], row[f"row{rr['row']}_cell"] = py(rr["value"]), rr["cell"]
                row["_loc"][f"row{rr['row']}_value"] = json_loc(r["meta_file"], ["decision", "rows", i, "value"])
                row["_loc"][f"row{rr['row']}_cell"] = json_loc(r["meta_file"], ["decision", "rows", i, "cell"])
        s = r["surgery"]
        if s is not None:
            o = s[(s["k"] == 0) & (s["variant"] == "original")]
            if len(o):
                row["original_top1"] = float(o["top1"].iloc[0])
                row["_loc"]["original_top1"] = csv_loc(r["surgery_file"], {"k": 0, "variant": "original"}, "top1")
            for k in (3, 6, 9):
                for var, src in vmap.items():
                    sel = s[(s["k"] == k) & (s["variant"] == src)]
                    if len(sel) == 1:
                        row[f"{var}_k{k}"] = float(sel["top1"].iloc[0])
                        row["_loc"][f"{var}_k{k}"] = csv_loc(r["surgery_file"], {"k": k, "variant": src}, "top1")
                        if var == "surgery":
                            row[f"flip_surgery_k{k}"] = float(sel["flip_consistency"].iloc[0])
        if r["mlp"] is not None:
            M = r["mlp"][(r["mlp"]["block"] >= 1) & (r["mlp"]["block"] <= 6)]
            row["mlp_saving_first_half"] = float(M["param_saving_frac"].mean())
        row["prereg_file"] = "E1_INSTRUCTIONS.md" if run_id == "E1" else "E1B_PREREG.md"
        row["prereg_commit"] = commits.get(row["prereg_file"])
        T.add(row)
    # E1 per-layer diagnostic (post hoc)
    if diag and diag.get("surgery") is not None and e1 and e1["layers"] is not None and diag.get("blocks") is not None:
        dm = diag["meta"] or {}
        row = {"run_id": "E1 per-layer diagnostic", "role": "post hoc diagnostic (no verdict)", "protocol": "v1",
               "model": e1["model"], "operator_mode": "per_layer", "date": dm.get("date"),
               "git_hash": (dm.get("git") or {}).get("git_hash"), "seed": dm.get("seed"),
               "_src": f"{diag['surgery_file']};{diag['blocks_file']};{e1['layers_file']}", "_seed": dm.get("seed"),
               "_git": (dm.get("git") or {}).get("git_hash"), "_loc": {}, "_derived": {}}
        lay = e1["layers"].copy()
        lay["rel_e_A_used"], lay["rel_e_null_used"] = lay["rel_e_A_layer"], lay["rel_e_null_layer"]
        sur = diag["surgery"].copy()
        sur["variant"] = sur["variant"].replace({"surgery_perlayer": "surgery", "svd_perlayer_matched": "svd",
                                                 "random_perlayer": "random"})
        dcfg = e1["cfg"].get("decision")
        if dcfg:
            lo, hi = dcfg["blocks"]
            Ls = lay[(lay["layer"] >= lo) & (lay["layer"] <= hi)]
            dmf = float(np.mean(Ls["d_minus"] / Ls["d"]))
            try:
                dec = decide(lay, diag["blocks"], sur, dmf, dcfg, mode="per_layer")
                row["outcome"] = "post hoc (no verdict)"
                for rr in dec["rows"]:
                    row[f"row{rr['row']}_value"], row[f"row{rr['row']}_cell"] = py(rr["value"]), rr["cell"]
                    row["_derived"][f"row{rr['row']}_value"] = "mirror.report.decide on the diagnostic CSVs"
            except Exception as e:  # noqa: BLE001
                log(f"WARNING diagnostic decision failed: {e}")
        o = sur[(sur["k"] == 0) & (sur["variant"] == "original")]
        if len(o):
            row["original_top1"] = float(o["top1"].iloc[0])
            row["_loc"]["original_top1"] = csv_loc(diag["surgery_file"], {"k": 0, "variant": "original"}, "top1")
        src_names = {"surgery": "surgery_perlayer", "svd": "svd_perlayer_matched", "random": "random_perlayer"}
        for k in (3, 6, 9):
            for var, src in src_names.items():
                sel = diag["surgery"][(diag["surgery"]["k"] == k) & (diag["surgery"]["variant"] == src)]
                if len(sel) == 1:
                    row[f"{var}_k{k}"] = float(sel["top1"].iloc[0])
                    row["_loc"][f"{var}_k{k}"] = csv_loc(diag["surgery_file"], {"k": k, "variant": src}, "top1")
        T.add(row)
    return T


# ================================================================================================
# T_x_checks (X1, X1b, X4)
# ================================================================================================
def build_x_checks(P, x1, x1b, x4, m1, plan_text):
    T = Table("T_x_checks", "Stage 0 checks: X1 (SVD collapse), X1b (compound SVD), X4 (full LayerNorm folding)",
              ["check_id"],
              [("check", "check_id", ""), ("experiment", "experiment", ""), ("model", "model", ""),
               ("quantity", "quantity", ""), ("value", "value", ".4g"), ("bar", "bar", ""), ("verdict", "verdict", ""),
               ("git", lambda r: short(r.get("git_hash")), "")])
    T.notes = ["X1 and X1b use a small validation protocol (10 bank and 2 query images per class, split seed 1 in the "
               "script), so their top-1 values are not comparable with the main tables."]
    m = re.search(r"to within (\d+)%", plan_text or "")
    within = float(m[1]) / 100 if m else None
    if x1:
        f, j = x1["file"], x1["json"]
        g = (j.get("git") or {}).get("git_hash")
        for model, d in (j.get("models") or {}).items():
            ms = MODEL_SHORT.get(model, model)
            base = {"experiment": "X1", "model": model, "git_hash": g, "_src": f, "_git": g, "_seed": x1.get("seed")}

            def add(cid, q, v, bar="", verdict="", loc=None, derived=None):
                r = T.add({**base, "check_id": f"x1.{ms}.{cid}", "quantity": q, "value": v, "bar": bar, "verdict": verdict})
                if loc:
                    r["_loc"]["value"] = loc
                if derived:
                    r["_derived"]["value"] = derived
            add("full_rank_diff", "full-rank SVD on all blocks: max abs CLS difference", py(d.get("full_rank_max_abs_diff")),
                "identical outputs (no numeric bar)", "", json_loc(f, ["models", model, "full_rank_max_abs_diff"]))
            fr = d.get("fold_max_rel_diff_per_block") or []
            if fr:
                add("fold_max_rel", "folding: max relative difference over blocks", float(max(fr)), "exact folding (no numeric bar)",
                    "", derived="max of fold_max_rel_diff_per_block")
            for mat in ("fc1", "fc2"):
                rr = d.get(f"rank_param_ratio_{mat}") or []
                if rr:
                    ok = within is not None and min(rr) >= 1 - within and max(rr) <= 1.0
                    add(f"rank_ratio_{mat}_min", f"SVD rank parameters / matched count, {mat}, min over blocks", float(min(rr)),
                        f"within {fmt(within * 100 if within else None, 'g')}% and not above (PAPER_EXPERIMENTS.md Section 1)",
                        PASS if ok else FAIL, derived=f"min of rank_param_ratio_{mat}")
            if "svd_k6_features_finite" in d:
                add("finite", "features finite after SVD at k = 6", py(d["svd_k6_features_finite"]), "true",
                    PASS if d["svd_k6_features_finite"] else FAIL, json_loc(f, ["models", model, "svd_k6_features_finite"]))
            if "dtype" in d:
                add("dtype", "precision", d["dtype"], "no fp16", PASS if "16" not in str(d["dtype"]) else FAIL,
                    json_loc(f, ["models", model, "dtype"]))
            if "small_knn_top1_original" in d:
                add("small_top1", "original top-1, small protocol", py(d["small_knn_top1_original"]), "", "",
                    json_loc(f, ["models", model, "small_knn_top1_original"]))
            sb = d.get("single_block_svd_drop") or []
            if sb:
                add("single_block_max_drop", "single-block SVD: largest top-1 drop (small protocol)", float(max(sb)), "", "",
                    derived=f"max of single_block_svd_drop (block {int(np.argmax(sb)) + 1})")
            for mat in ("fc1", "fc2"):
                en = d.get(f"svd_discarded_energy_{mat}") or []
                if en:
                    add(f"svd_energy_{mat}", f"weight energy discarded by SVD, {mat}, mean over blocks 1 to 6",
                        float(np.mean(en[:6])), "", "", derived=f"mean of svd_discarded_energy_{mat}[0:6]")
    if x1b:
        f, j = x1b["file"], x1b["json"]
        g = (j.get("git") or {}).get("git_hash")
        for model, rows in (j.get("models") or {}).items():
            ms = MODEL_SHORT.get(model, model)
            for i, rr in enumerate(rows):
                r = T.add({"check_id": f"x1b.{ms}.k{rr['k']}.top1", "experiment": "X1b", "model": model,
                           "quantity": f"SVD on the first {rr['k']} blocks: top-1 (small protocol)", "value": py(rr["top1"]),
                           "git_hash": g, "_src": f, "_git": g, "_seed": x1b.get("seed")})
                r["_loc"]["value"] = json_loc(f, ["models", model, i, "top1"])
    for key, rec in sorted(x4.items()):
        f, j = rec["file"], rec["json"]
        model = (j.get("model") or {}).get("name") or key
        ms = MODEL_SHORT.get(model, model)
        dt = j.get("dtype")
        g = (j.get("git") or {}).get("git_hash")
        base = {"experiment": "X4", "model": model, "git_hash": g, "_src": f, "_git": g, "_seed": j.get("seed")}
        tol = j.get("tol")
        r = T.add({**base, "check_id": f"x4.{ms}.{dt}.max_abs", "quantity": f"folded vs original, max abs difference ({dt}, "
                   f"{j.get('config', {}).get('n_images')} images)", "value": py(j.get("max_abs_diff")),
                   "bar": f"<= {fmt(tol, 'g')} (the plan's bar, set for fp32)", "verdict": PASS if j.get("pass") else FAIL})
        r["_loc"]["value"] = json_loc(f, ["max_abs_diff"])
        r["_loc"]["verdict_raw"] = json_loc(f, ["pass"])
        r = T.add({**base, "check_id": f"x4.{ms}.{dt}.max_rel", "quantity": f"folded vs original, max relative difference ({dt})",
                   "value": py(j.get("max_rel_diff"))})
        r["_loc"]["value"] = json_loc(f, ["max_rel_diff"])
    if m1["knn"] is not None:
        k = m1["knn"]
        for v in ("original", "folded"):
            sel = k[(k["k"] == 0) & (k["variant"] == v)]
            if len(sel) == 1:
                r = T.add({"check_id": f"x4.m1.{v}.top1", "experiment": "X4", "model": m1["model"],
                           "quantity": f"k-NN top-1, {v} model (protocol v2, M1 run)", "value": float(sel["top1"].iloc[0]),
                           "git_hash": str(sel["git_hash"].iloc[0]), "_src": m1["knn_file"],
                           "_git": str(sel["git_hash"].iloc[0]), "_seed": int(sel["seed"].iloc[0])})
                r["_loc"]["value"] = csv_loc(m1["knn_file"], {"k": 0, "variant": v}, "top1")
    return T


# ================================================================================================
# T_attention (M1)
# ================================================================================================
def build_attention(P, m1):
    H = Table("T_attention", "M1 heads per block (V-local, registers model)", ["block"],
              [("block", "block", "d"), ("self-heads", "n_self", "d"), ("twin pairs", "n_pairs", "d"),
               ("pair score", "mean_pair_score", ".3f"), ("err q,k (identity)", lambda r: f"{fmt(r.get('err_qk'), '.3f')} "
               f"({fmt(r.get('err_qk_identity'), '.3f')})", ""), ("err v (identity)", lambda r: f"{fmt(r.get('err_v'), '.3f')} "
               f"({fmt(r.get('err_v_identity'), '.3f')})", ""), ("s_q", "s_q", ".3f"), ("s_k", "s_k", ".3f"),
               ("s_v", "s_v", ".3f"), ("s_o", "s_o", ".3f"), ("s_all", "s_all", ".3f"), ("random", "s_all_random", ".3f"),
               ("kept", "attn_kept_frac", ".3f")])
    H.notes = ["Pairing score: cosine of the best orthogonal map between head h on the original and head sigma(h) on the "
               "mirror (q and k together). Errors: relative error of the head maps on q, k and v, identity map in "
               "brackets. s_*: attention breaking shares; random: matched random structure. Kept: free attention "
               "matrix parameters over dense."]
    V = Table("T_attention_vexact", "M1 V-exact: one operator per span", ["k"],
              [("k", "k", "d"), ("layers", "layers", ""), ("d+", "d_plus", "d"), ("d-", "d_minus", "d"),
               ("agreement", "mean_agreement", ".3f"), ("alignment (min)", lambda r: f"{fmt(r.get('mean_alignment'), '.3f')} "
               f"({fmt(r.get('min_alignment'), '.3f')})", ""), ("attn s_all", "mean_s_all", ".3f"),
               ("MLP s1 / s2", lambda r: f"{fmt(r.get('mean_mlp_s1'), '.3f')} / {fmt(r.get('mean_mlp_s2'), '.3f')}", ""),
               ("max rel. error in span", "exact_max_rel_in_span", ".2g"), ("exactness pass", "exact_pass", "")])
    V.notes = ["Agreement: tr(A* A_l)/d averaged over the span. Alignment: tr(A* C_l)/tr(A_l C_l). Exactness: largest "
               "relative error between mirrored-image features and A* times the original features over the span's "
               "layers (fp32, m1_exactness.csv); bar 1e-4 (m1_meta.json exact_tol)."]
    C = Table("T_attention_conversion", "M1 full conversion (k = 12, V-exact, fp32) and the A_out choice", ["quantity"],
              [("quantity", "quantity", ""), ("value", "value", ".6g"), ("note", "note", "")])
    if m1["status"] == NOT_RUN:
        return H, V, C
    g = m1["git"]
    if m1["heads"] is not None:
        for _, r in m1["heads"].iterrows():
            b = int(r["block"])
            row = {"block": b, "_src": m1["heads_file"], "_git": g, "_seed": int(r["seed"]), "_fit_seed": int(r["fit_seed"]),
                   "_loc": {}, "_derived": {}}
            for c in ("n_self", "n_pairs", "mean_pair_score", "err_qk", "err_qk_identity", "err_v", "err_v_identity", "s_q",
                      "s_k", "s_v", "s_o", "s_all", "s_all_random", "s_all_over_random", "attn_kept_frac", "mlp_s1", "mlp_s2",
                      "attn_free_total", "attn_dense_total"):
                if c in r:
                    row[c] = py(r[c])
                    row["_loc"][c] = csv_loc(m1["heads_file"], {"block": b}, c)
            row["sigma"] = str(r.get("sigma"))
            H.add(row)
    ex = m1["exactness"]
    if m1["vexact"] is not None:
        for _, r in m1["vexact"].iterrows():
            k = int(r["k"])
            row = {"k": k, "layers": str(r["layers"]), "_src": m1["vexact_file"], "_git": g, "_seed": int(r["seed"]),
                   "_fit_seed": int(r["fit_seed"]), "_loc": {}, "_derived": {}}
            for c in ("d_plus", "d_minus", "mean_agreement", "min_agreement", "mean_alignment", "min_alignment",
                      "mean_s_all", "mean_mlp_s1", "mean_mlp_s2", "register_pairs"):
                row[c] = py(r[c])
                row["_loc"][c] = csv_loc(m1["vexact_file"], {"k": k}, c)
            row["dminus_frac"] = row["d_minus"] / (row["d_plus"] + row["d_minus"])
            if ex is not None:
                sel = ex[(ex["k"] == k) & (ex["in_span"].astype(str) == "True")]
                if len(sel):
                    row["exact_max_rel_in_span"] = float(sel["rel_max"].max())
                    row["exact_pass"] = bool(sel["pass"].astype(str).eq("True").all())
                    row["_derived"]["exact_max_rel_in_span"] = f"max rel_max over in-span layers of {m1['exactness_file']}"
            V.add(row)
    fl = m1["flip"]
    if fl:
        for part, label in (("even_head", "with the even projection of the final features"),
                            ("no_even_head", "without the even projection")):
            if part in fl:
                r = C.add({"quantity": f"flip consistency {label} (%)", "value": py(fl[part]["flip_consistency"]),
                           "note": f"{fl[part].get('n_query')} validation queries, bank {fl[part].get('n_bank')} train images",
                           "_src": m1["flip_file"], "_git": fl.get("git_hash"), "_seed": fl.get("seed"),
                           "_fit_seed": fl.get("fit_seed")})
                r["_loc"]["value"] = json_loc(m1["flip_file"], [part, "flip_consistency"])
                r = C.add({"quantity": f"largest relative CLS feature difference, image vs mirror, {label}",
                           "value": py(fl[part]["cls_feature_rel_max"]), "_src": m1["flip_file"], "_git": fl.get("git_hash"),
                           "_seed": fl.get("seed")})
                r["_loc"]["value"] = json_loc(m1["flip_file"], [part, "cls_feature_rel_max"])
        if "even_head" in fl and "pass" in fl["even_head"]:
            r = C.add({"quantity": "full conversion flip check pass (must be 100%)", "value": bool(fl["even_head"]["pass"]),
                       "_src": m1["flip_file"], "_git": fl.get("git_hash"), "_seed": fl.get("seed")})
            r["_loc"]["value"] = json_loc(m1["flip_file"], ["even_head", "pass"])
    ao = m1["a_out"]
    if ao:
        for key in ("mean_s_all_A_l", "mean_s_all_A_l+1"):
            if key in ao:
                r = C.add({"quantity": f"A_out choice: block-average attention s_all with {key.replace('mean_s_all_', '')}",
                           "value": py(ao[key]), "note": f"kept {ao.get('kept')}", "_src": m1["a_out_file"],
                           "_git": ao.get("git_hash"), "_seed": ao.get("seed")})
                r["_loc"]["value"] = json_loc(m1["a_out_file"], [key])
    return H, V, C


# ================================================================================================
# T_prereg
# ================================================================================================
def build_prereg(P, e1, e1b, f3, d_free, Tp3, Tp4, x1, status, th, commits, f4, Te3c: Table | None = None,
                 th3: dict | None = None):
    T = Table("T_prereg", "Every pre-registered check", ["id"],
              [("id", "id", ""), ("experiment", "experiment", ""), ("check", "statement", ""), ("value", "value", ".4g"),
               ("threshold", "threshold", ""), ("verdict", "verdict", ""), ("source", "source_file", ""),
               ("pre-registration", lambda r: f"{r.get('prereg_file', '')} {short(r.get('prereg_commit'))}"
                if r.get("prereg_file") else "", ""), ("note", "note", "")])
    T.notes = ["Thresholds are read from the pre-registration files (or from the run outputs, which copy them); the "
               "commit is the one that added the file (git log --diff-filter=A), except for PREREG_E3.md, whose record is "
               "the commit that signed it off (the first commit marking it ACTIVE). Pending: a run is still missing."]

    def dec_rows(prefix, exp, r, pfile):
        if not r or not r.get("decision"):
            T.add({"id": f"{prefix}.outcome", "experiment": exp, "statement": "decision table outcome",
                   "verdict": NOT_RUN, "prereg_file": pfile, "prereg_commit": commits.get(pfile), "_src": ""})
            return
        for i, rr in enumerate(r["decision"]["rows"]):
            row = T.add({"id": f"{prefix}.row{rr['row']}", "experiment": exp, "statement": rr["check"],
                         "value": py(rr["value"]), "threshold": f"GO {rr['go']}; KILL {rr['kill']}", "verdict": rr["cell"],
                         "source_file": r["meta_file"], "prereg_file": pfile, "prereg_commit": commits.get(pfile),
                         "run_git_hash": r["git"], "_src": r["meta_file"], "_git": r["git"], "_seed": r["seed"]})
            row["_loc"]["value"] = json_loc(r["meta_file"], ["decision", "rows", i, "value"])
            row["_loc"]["verdict"] = json_loc(r["meta_file"], ["decision", "rows", i, "cell"])
        row = T.add({"id": f"{prefix}.outcome", "experiment": exp, "statement": "decision table outcome",
                     "verdict": r["decision"]["outcome"], "threshold": "rule of the decision table",
                     "source_file": r["meta_file"], "prereg_file": pfile, "prereg_commit": commits.get(pfile),
                     "run_git_hash": r["git"], "note": r["decision"].get("reason"), "_src": r["meta_file"], "_git": r["git"],
                     "_seed": r["seed"]})
        row["_loc"]["verdict"] = json_loc(r["meta_file"], ["decision", "outcome"])

    dec_rows("E1", "E1", e1, "E1_INSTRUCTIONS.md")
    for name, r in sorted(e1b.items()):
        dec_rows(f"E1b.{name.replace('e1b_', '')}", "E1b", r, "E1B_PREREG.md")
    # P1 and G3 (F3 outputs copy the thresholds from PREREG_E2.md via mirror.findings)
    for model, bullet in (("dinov2_vitb14", "plain"), ("dinov2_vitb14_reg", "registers")):
        rec = f3.get(model)
        cid = f"P1.{bullet}"
        if not rec:
            T.add({"id": cid, "experiment": "F3", "statement": f"P1 {bullet} bullet", "verdict": NOT_RUN,
                   "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"), "_src": ""})
            continue
        p = rec["json"]["p1"]
        g = (rec["json"].get("git") or {}).get("git_hash")
        row = T.add({"id": cid, "experiment": "F3", "statement": p["statement"], "value": py(p["value"]),
                     "threshold": f"{p['comparison']} {p['threshold']:g}", "verdict": p["result"], "source_file": rec["file"],
                     "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"),
                     "commit_recorded_by_run": (rec["json"].get("prereg") or {}).get("commit"), "run_git_hash": g,
                     "operators": (rec["json"].get("operators") or {}).get("path"),
                     "_src": rec["file"], "_git": g, "_seed": rec["json"].get("seed"),
                     "_fit_seed": fit_seed_from_path((rec["json"].get("operators") or {}).get("path"))})
        row["_loc"]["value"] = json_loc(rec["file"], ["p1", "value"])
        row["_loc"]["verdict"] = json_loc(rec["file"], ["p1", "result"])
        expect = th.get("p1_plain_cut") if bullet == "plain" else th.get("p1_reg_ratio")
        row["threshold_matches_prereg"] = expect is not None and abs(float(p["threshold"]) - expect) < 1e-12
    g3 = f3.get("_gate")
    if g3:
        g = (g3["json"].get("git") or {}).get("git_hash")
        f3_seeds = {v["json"].get("seed") for k, v in f3.items() if not k.startswith("_")}
        row = T.add({"id": "G3", "_seed": f3_seeds.pop() if len(f3_seeds) == 1 else None, "experiment": "F3", "statement": g3["json"].get("rule"), "verdict": g3["json"].get("status"),
                     "threshold": "at least one P1 bullet passes", "source_file": g3["file"], "prereg_file": "PREREG_E2.md",
                     "prereg_commit": th.get("commit"), "run_git_hash": g, "_src": g3["file"], "_git": g})
        row["_loc"]["verdict"] = json_loc(g3["file"], ["status"])
    else:
        T.add({"id": "G3", "experiment": "F3", "statement": "gate G3", "verdict": NOT_RUN, "prereg_file": "PREREG_E2.md",
               "prereg_commit": th.get("commit"), "_src": ""})
    # P2 (F4)
    if f4:
        j = f4["json"]
        prim = j.get("primary_prereg_models") or {}
        wb = j.get("written_by") or {}
        g4 = (wb.get("git") or {}).get("git_hash") or ";".join(j.get("git_hashes") or [])
        exp = th.get("p2_frac_pct")
        row = T.add({"id": "P2", "experiment": "F4", "statement": j.get("hypothesis") or "P2 rule",
                     "value": py(prim.get("fraction")), "threshold": f">= {fmt(j.get('threshold'), 'g')}",
                     "verdict": j.get("verdict") or NOT_RUN, "source_file": f4["file"],
                     "note": (f"{prim.get('n_hits')} of {prim.get('n_cases')} cases; models complete: "
                              f"{j.get('complete')}" if prim else str(j.get("status", ""))),
                     "threshold_matches_prereg": (exp is not None and j.get("threshold") is not None
                                                  and abs(float(j["threshold"]) - exp / 100) < 1e-12),
                     "commit_recorded_by_run": ";".join(j.get("prereg_commits") or []) or j.get("prereg_commit_this_run"),
                     "run_git_hash": g4, "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"),
                     "_src": f4["file"], "_git": g4, "_seed": wb.get("seed")})
        row["_loc"]["verdict"] = json_loc(f4["file"], ["verdict"]) if "verdict" in j else None
        if "fraction" in prim:
            row["_loc"]["value"] = json_loc(f4["file"], ["primary_prereg_models", "fraction"])
    else:
        T.add({"id": "P2", "experiment": "F4", "statement": "sign(activation-weighted E_svd - s) predicts the smaller "
               "single-block drop", "threshold": f">= {fmt(th.get('p2_frac_pct'), 'g')}% of cases", "verdict": NOT_RUN,
               "note": " | ".join(status_mentions(status, ["F4"])) or "no runs/f4/f4_p2.json", "prereg_file": "PREREG_E2.md",
               "prereg_commit": th.get("commit"), "_src": ""})
    # P3 and G2
    for r in Tp3.rows:
        T.add({"id": r["check"], "experiment": "M3", "statement": r["statement"], "value": r.get("value"),
               "threshold": r.get("threshold_text"), "verdict": r["verdict"], "source_file": "tables/T_healing_p3.csv",
               "note": r.get("note"), "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"),
               "_src": r.get("_src"), "_derived": {"value": "T_healing_p3"}})
    # E3 and gate G5 (PREREG_E3.md; next to G2, Section 12 decision 3)
    for r in (Te3c.rows if Te3c is not None else []):
        if r["id"] not in ("E3.a", "E3.b", "E3.seed", "G5", "E3.lr"):
            continue
        T.add({"id": r["id"], "experiment": "E3", "statement": r["statement"], "value": r.get("value"),
               "threshold": r.get("threshold"), "verdict": r["verdict"], "source_file": "tables/T_e3_checks.csv",
               "note": r.get("note"), "prereg_file": "PREREG_E3.md", "prereg_commit": (th3 or {}).get("commit"),
               "_src": r.get("_src"), "_git": r.get("_git"), "_seed": r.get("_seed"),
               "_derived": {"value": (r.get("_derived") or {}).get("value", "T_e3_checks")}})
    for r in Tp4.rows:
        T.add({"id": r["check"], "experiment": "G", "statement": r["statement"], "value": r.get("value"),
               "threshold": r.get("threshold_text"), "verdict": r["verdict"], "source_file": "tables/T_generality_p4.csv",
               "note": r.get("note"), "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"),
               "_src": r.get("_src"), "_git": r.get("_git"), "_seed": r.get("_seed"),
               "_derived": {"value": (r.get("_derived") or {}).get("value", "T_generality_p4")}})
    # P5 and G4
    if d_free:
        j, f = d_free["json"], d_free["file"]
        p5 = j.get("p5") or {}
        g = ((j.get("provenance") or {}).get("git") or {}).get("git_hash")
        seed = (j.get("provenance") or {}).get("seed")
        ops = ((j.get("provenance") or {}).get("operators") or {}).get("path")
        thr = p5.get("thresholds") or {}
        for cid, part, stmt, thr_k in (("P5.part1", "part1", "layer-0 even subspace: mean cosine of principal angles", "mean_cos_even"),
                                       ("P5.part2", "part2", "block-1 pairing from weights given the data-driven A_0: "
                                        "recovered fraction of data-driven pairs", "pair_recovery"),
                                       ("P5.part2_strict", "part2_strict", "block-1 pairing, fully data-free chain "
                                        "(given A_0^w): recovered fraction", "pair_recovery")):
            if part not in p5:
                continue
            row = T.add({"id": cid, "experiment": "D", "statement": stmt, "value": py(p5[part].get("value")),
                         "threshold": f">= {fmt(thr.get(thr_k), 'g')}", "verdict": PASS if p5[part].get("pass") else FAIL,
                         "source_file": f, "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"),
                         "commit_recorded_by_run": (j.get("provenance", {}).get("prereg") or {}).get("commit"),
                         "run_git_hash": g, "operators": ops, "_src": f, "_git": g, "_seed": seed,
                         "_fit_seed": fit_seed_from_path(ops)})
            row["_loc"]["value"] = json_loc(f, ["p5", part, "value"])
        row = T.add({"id": "P5", "experiment": "D", "statement": "P5 (parts 1 and 2 as D2 defines them)",
                     "verdict": PASS if p5.get("pass") else FAIL, "threshold": "both parts pass", "source_file": f,
                     "note": f"strict reading: {PASS if p5.get('pass_strict') else FAIL}", "prereg_file": "PREREG_E2.md",
                     "prereg_commit": th.get("commit"), "run_git_hash": g, "_src": f, "_git": g, "_seed": seed})
        row = T.add({"id": "G4", "experiment": "D", "statement": "gate G4: below the bar, claim C6 is dropped",
                     "verdict": str(p5.get("gate_G4", "")).upper(), "threshold": "P5 passes", "source_file": f,
                     "prereg_file": "PREREG_E2.md", "prereg_commit": th.get("commit"), "run_git_hash": g, "_src": f,
                     "_git": g, "_seed": seed})
        row["_loc"]["verdict_raw"] = json_loc(f, ["p5", "gate_G4"])
    else:
        for cid in ("P5", "G4"):
            T.add({"id": cid, "experiment": "D", "statement": cid, "verdict": NOT_RUN, "prereg_file": "PREREG_E2.md",
                   "prereg_commit": th.get("commit"), "_src": ""})
    # G1 (gate in the plan; no numeric pre-registered bar)
    quote = [s for s in status_mentions(status, ["X1 SVD-collapse check"])]
    cell, g1 = "", None
    for line in status["lines"]:
        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| X1"):          # older STATUS layout: | X1 ... | passed ... |
            cell = parts[1] if len(parts) > 1 else ""
            g1 = PASS if cell.lower().startswith("passed") else None
        elif line.startswith("| G1"):        # current layout: | G1 <check> | <value> | <bar> | <verdict> |
            cell = " ".join(x for x in parts[:2] if x)
            g1 = PASS if parts[-1].upper().startswith("PASS") else (FAIL if parts[-1].upper().startswith("FAIL") else None)
    T.add({"id": "G1", "experiment": "X1", "statement": "gate G1: if the SVD collapse on the registers model is a bug, fix "
           "it and rerun every SVD number", "verdict": g1 if g1 else ("see note" if cell else NOT_RUN),
           "threshold": "no bug found (X1 checks 1 to 4; no numeric bar)",
           "source_file": (x1 or {}).get("file", ""), "note": ("STATUS.md: " + cell) if cell else " | ".join(quote),
           "prereg_file": "PAPER_EXPERIMENTS.md", "prereg_commit": commits.get("PAPER_EXPERIMENTS.md"),
           "_src": (x1 or {}).get("file", ""), "_git": (((x1 or {}).get("json") or {}).get("git") or {}).get("git_hash"),
           "_seed": (x1 or {}).get("seed")})
    return T


# ================================================================================================
# A (ablations), Tier 2 (linear and transfer probes), M2 (latency), F4 (P2)
# ================================================================================================
A2_EXTRA = ("hungarian", "hungarian_cycle", "generic")


def _info(text) -> dict:
    try:
        v = json.loads(text) if isinstance(text, str) and text.strip().startswith("{") else {}
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def build_ablations(P, Tm: Table | None = None):
    """Per-seed rows of runs/a/<model>/A*_s*.csv, the mean over fit seeds, and the A2 extra pairings. With Tm
    (T_main_zeroshot), cells that are the canonical setting are compared with the canonical X2 value of the same fit
    seed (ablation_vs_canonical)."""
    T = Table("T_ablations", "A1 to A5: zero-shot ablations, one row per fit seed",
              ["model", "ablation", "variant", "k", "fit_seed"],
              [("model", "model_label", ""), ("ablation", "ablation", ""), ("variant", "variant", ""), ("k", "k", "d"),
               ("fit seed", "fit_seed", "d"), ("blocks", "blocks", ""), ("top-1", "top1", ".3f"),
               ("drop", "drop_points", ".2f"), ("flip consistency", "flip_consistency", ".2f"),
               ("MLP kept", "mlp_params_kept_frac", ".3f"), ("params vs default", "params_vs_mirror_first_k", ".3f"),
               ("matched (1%)", "params_matched", ""), ("exact projection", "params_exact", ""),
               ("X2 (canonical)", "canonical_top1", ".3f"), ("minus X2", "minus_canonical", "+.3f"),
               ("reused from", "reused_from", "")])
    T.notes = ["Zero-shot k-NN top-1 and flip consistency (%), paper protocol; drop = original minus variant. "
               "Reused from: the variant is the same model as the named cell (identical content signature in "
               "scripts/stage1_ablate.py), so its result was reused, not evaluated again. A fit seed of -1 marks a row "
               "fitted on pooled fit images (see the row's info).",
               "MLP kept: MLP weights kept in the converted blocks over dense. Params vs default "
               "(params_vs_mirror_first_k of the CSV): MLP weights kept by the variant over those kept by the default "
               "setting (per-layer operators, mirror pairing, blocks 1 to k, alpha 1) at the same k and fit seed. "
               "Matched: within 1% of the default, the plan's matched-budget rule; rows marked no are not "
               "parameter-matched comparisons."]
    S = Table("T_ablations_seeds", "A1 to A5: mean over fit seeds, every k present", ["model", "ablation", "variant", "k"],
              [("model", "model_label", ""), ("ablation", "ablation", ""), ("variant", "variant", ""), ("k", "k", "d"),
               ("n", "n_seeds", "d"), ("fit seeds", "fit_seeds", ""),
               ("top-1", lambda r: pm(r.get("top1_mean"), r.get("top1_std")), ""),
               ("top-1 per seed", "top1_per_seed", ""),
               ("drop", lambda r: pm(r.get("drop_mean"), r.get("drop_std")), ""),
               ("flip consistency", lambda r: pm(r.get("flip_mean"), r.get("flip_std")), ""),
               ("MLP kept", "mlp_kept_frac_mean", ".3f"), ("params vs default", "params_vs_first_k_mean", ".3f"),
               ("matched (1%)", "params_matched", ""), ("X2 (canonical)", "canonical_top1_mean", ".2f"),
               ("note", "note", "")])
    S.notes = ["One row per model, ablation, variant and k (every k present in the CSVs). Mean and sample standard "
               "deviation (n - 1) over the fit seeds present, with n the number of seeds; with one seed no deviation "
               "is shown. Top-1 per seed lists the per-seed values (also one row each in T_ablations). MLP kept and "
               "params vs default: means over the same fit seeds (see T_ablations); matched: within 1% of the default "
               "for every seed."]
    X = Table("T_ablations_a2_extra", "A2 pairings hungarian, hungarian_cycle and generic: pairing and projection as recorded",
              ["model", "variant", "k", "fit_seed"],
              [("variant", "variant", ""), ("k", "k", "d"), ("fit seed", "fit_seed", "d"), ("top-1", "top1", ".2f"),
               ("exact projection", "params_exact", ""), ("pairing", "pairing", ""), ("projection", "projection", "")])
    X.notes = ["Pairing and projection are the descriptions each row records in its info column (scripts/stage1_ablate.py)."]
    base = P.runs / "a"
    files = sorted(base.glob("*/A*_s*.csv")) if base.is_dir() else []
    seen: dict = {}
    dups: list[str] = []
    for f in files:
        df = read_csv(f)
        if df is None:
            continue
        model = str(df["model"].iloc[0]) if "model" in df else f.parent.name
        for _, r in df.iterrows():
            # the CSVs are keyed by (ablation, variant, k, fit seed); a key met twice (in two files) is kept once
            key = (model, str(r["ablation"]), str(r["variant"]), int(r["k"]), int(r["fit_seed"]))
            if key in seen:
                dups.append(f"{'/'.join(map(str, key[1:]))} in {P.rel(f)} (kept {seen[key]})")
                continue
            seen[key] = P.rel(f)
            info = _info(r.get("info"))
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "ablation": r.get("ablation"),
                   "variant": r.get("variant"), "k": py(r.get("k")), "fit_seed": py(r.get("fit_seed")),
                   "reused_from": "" if pd.isna(r.get("reused_from")) else str(r.get("reused_from")),
                   "params_exact": py(r.get("params_exact")), "n_fit_images": py(r.get("n_fit_images")),
                   "info": "" if pd.isna(r.get("info")) else str(r.get("info")), "source": P.rel(f),
                   "git_hash": r.get("git_hash"), "_src": P.rel(f), "_git": r.get("git_hash"), "_seed": py(r.get("seed")),
                   "_fit_seed": py(r.get("fit_seed")), "_loc": {}, "_derived": {}}
            where = {"ablation": r["ablation"], "variant": r["variant"], "k": int(r["k"]), "fit_seed": int(r["fit_seed"])}
            for c in ("top1", "drop_points", "flip_consistency", "mlp_params_kept_frac", "params_vs_mirror_first_k"):
                if c in r and py(r[c]) is not None:
                    row[c] = py(r[c])
                    row["_loc"][c] = csv_loc(P.rel(f), where, c)
            if row.get("params_vs_mirror_first_k") is not None:
                row["params_matched"] = abs(row["params_vs_mirror_first_k"] - 1.0) <= PARAM_MATCH_TOL
            row["blocks"] = "" if pd.isna(r.get("blocks")) else str(r.get("blocks"))
            row["pairing"], row["projection"] = info.get("pairing", ""), info.get("projection", "")
            row["fit_rows"] = info.get("fit_rows", "")
            row["agreement_with_fit_run"] = info.get("agreement_with_fit_run")
            row["variant_seed"] = py(r.get("variant_seed")) if "variant_seed" in r else None
            row["n_query"] = py(r.get("n_query")) if "n_query" in r else None
            T.add(row)
    # Stable order for reading: model, ablation, k, the variant order of the CSVs, fit seed.
    vorder: dict = {}
    for r in T.rows:
        vorder.setdefault((r["model"], r["ablation"]), [])
        if r["variant"] not in vorder[(r["model"], r["ablation"])]:
            vorder[(r["model"], r["ablation"])].append(r["variant"])

    def order(r):
        return (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99, str(r["model"]), str(r["ablation"]),
                r["k"] if r["k"] is not None else -1, vorder[(r["model"], r["ablation"])].index(r["variant"]),
                r["fit_seed"] if r["fit_seed"] is not None else 99)

    T.rows.sort(key=order)
    for r in T.rows:
        if r["ablation"] == "A2" and r["variant"] in A2_EXTRA:
            X.add({**{k: v for k, v in r.items() if not k.startswith("_")}, "_src": r["_src"], "_git": r["_git"],
                   "_seed": r["_seed"], "_fit_seed": r["_fit_seed"], "_loc": dict(r["_loc"]), "_derived": {}})
    groups: dict = {}
    for r in T.rows:
        groups.setdefault((r["model"], r["ablation"], r["variant"], r["k"]), []).append(r)
    for (model, abl, var, k), rs in groups.items():
        rs = sorted(rs, key=lambda x: x["fit_seed"])
        out = {"model": model, "model_label": MODEL_LABEL.get(model, model), "ablation": abl, "variant": var, "k": k,
               "n_seeds": len(rs), "fit_seeds": ",".join(str(x["fit_seed"]) for x in rs),
               "top1_per_seed": "; ".join(f"s{x['fit_seed']}: {fmt(x.get('top1'), '.3f')}" for x in rs),
               "_src": ";".join(sorted({x["_src"] for x in rs})), "_git": ";".join(sorted({str(x["_git"]) for x in rs})),
               "_seed": rs[0]["_seed"] if len({x["_seed"] for x in rs}) == 1 else None,
               "_fit_seed": ",".join(str(x["fit_seed"]) for x in rs), "_loc": {}, "_derived": {}}
        for name, col in (("top1", "top1"), ("drop", "drop_points"), ("flip", "flip_consistency"),
                          ("mlp_kept_frac", "mlp_params_kept_frac"), ("params_vs_first_k", "params_vs_mirror_first_k")):
            m_, s_, n_ = agg([x.get(col) for x in rs])
            out[f"{name}_mean"] = m_
            if name in ("top1", "drop", "flip"):
                out[f"{name}_std"] = s_
            if len(rs) == 1 and col in rs[0]["_loc"]:
                out["_loc"][f"{name}_mean"] = rs[0]["_loc"][col]
            elif m_ is not None:
                out["_derived"][f"{name}_mean"] = f"mean of {col} over fit seeds {out['fit_seeds']}"
        pm_ = [x.get("params_matched") for x in rs if x.get("params_matched") is not None]
        out["params_matched"] = all(pm_) if pm_ else None
        notes = []
        if any(x["fit_seed"] == -1 for x in rs):
            notes.append("fit seed -1: " + (rs[0].get("fit_rows") or "pooled fit images"))
        reused = sorted({x["reused_from"] for x in rs if x["reused_from"]})
        if reused:
            notes.append("same model as " + ", ".join(reused) + " (result reused)")
        if any(x.get("params_exact") is False for x in rs):
            notes.append("projection not exact (params_exact = False)")
        if out["params_matched"] is False and out.get("params_vs_first_k_mean") is not None:
            notes.append(f"not parameter-matched ({out['params_vs_first_k_mean']:.2f} of the default's MLP weights)")
        out["_notes"] = notes
        out["note"] = "; ".join(notes)
        S.add(out)
    if Tm is not None:
        ablation_vs_canonical(T, S, Tm)
    ablation_budget_notes(T, S)
    for r in S.rows:
        r["note"] = "; ".join(r.pop("_notes", []))
    return T, S, X, files, dups


def canonical_seed_value(Tm: Table, model: str, k: int, method: str, fit_seed):
    """(top1, record) of the canonical X2 row (T_main_zeroshot) for `method` at `model`, k and fit seed, else (None, None)."""
    key = CANONICAL_KEYS.get(method)
    r = Tm.find(model=model, k=k, variant=key) if key else None
    if r is None or r.get("status") != "ok" or r.get("experiment") != CANONICAL_RUN:
        return None, None
    x = (r.get("_per_seed") or {}).get(fit_seed)
    return (x["top1"], x) if x else (None, None)


def ablation_vs_canonical(T: Table, S: Table, Tm: Table) -> None:
    """Every ablation cell that is the canonical setting itself gets the canonical X2 value of the same fit seed, the
    difference and a note; T_main_zeroshot stays the number to quote."""
    default_n = {}
    for r in T.rows:
        if (r["ablation"], r["variant"]) == ("A1", "per_layer") and r.get("n_fit_images") is not None:
            default_n[(r["model"], r["fit_seed"])] = r["n_fit_images"]
    for r in T.rows:
        method = SAME_AS_CANONICAL.get((r["ablation"], r["variant"]))
        if method is None and r["ablation"] == "A3" and r["fit_seed"] is not None and r["fit_seed"] >= 0 \
                and r.get("n_fit_images") is not None and r["n_fit_images"] == default_n.get((r["model"], r["fit_seed"])):
            method = "mirror"           # refit from the run's own fit images: the canonical setting, refitted
        if method is None or r.get("top1") is None:
            continue
        v, x = canonical_seed_value(Tm, r["model"], r["k"], method, r["fit_seed"])
        if v is None:
            continue
        r["canonical_top1"], r["minus_canonical"] = v, r["top1"] - v
        r["canonical_source"] = x["src"]
        r["_loc"]["canonical_top1"] = csv_loc(x["src"], x["where"], "top1")
        r["_derived"]["minus_canonical"] = f"top1 minus canonical_top1 ({x['src']})"
        if method == "svd":
            r["canonical_note"] = ("identical model to the X2 SVD row (SVD does not use the operators)" if r["top1"] == v
                                   else "same SVD setting as X2 but a different value")
        elif r["ablation"] == "A3":
            ag = r.get("agreement_with_fit_run") or {}
            r["canonical_note"] = (f"refit from the run's own {r['n_fit_images']} fit images (operator agreement "
                                   f"{fmt(ag.get('mean_operator_agreement'), '.6f')}, pair recovery "
                                   f"{fmt(ag.get('mean_pair_recovery'), '.3f')} with the fit run): the canonical setting "
                                   "refitted, so the difference to X2 is numerical")
        else:
            r["canonical_note"] = "the canonical setting; this script " + OPERATOR_ROUNDTRIP
    for r in S.rows:
        rs = [x for x in T.rows if (x["model"], x["ablation"], x["variant"], x["k"]) ==
              (r["model"], r["ablation"], r["variant"], r["k"]) and x.get("canonical_top1") is not None]
        if not rs:
            continue
        r["canonical_top1_mean"] = float(np.mean([x["canonical_top1"] for x in rs]))
        r["minus_canonical_mean"] = float(np.mean([x["minus_canonical"] for x in rs]))
        r["_derived"]["canonical_top1_mean"] = (f"mean of the {CANONICAL_RUN} values (T_main_zeroshot per seed) over fit "
                                                f"seeds {', '.join(str(x['fit_seed']) for x in rs)}")
        r["_derived"]["minus_canonical_mean"] = "mean of minus_canonical over the same fit seeds"
        diffs = ", ".join(f"s{x['fit_seed']} {x['minus_canonical']:+.3f}" for x in rs)
        if all(x["minus_canonical"] == 0 for x in rs):
            r["_notes"].append(f"identical to {CANONICAL_RUN} at every fit seed")
        elif r["ablation"] == "A3":
            r["_notes"].append(f"{rs[0]['canonical_note']}; quote {CANONICAL_RUN} ({r['canonical_top1_mean']:.2f}); "
                               f"minus X2: {diffs}")
        else:
            r["_notes"].append(f"canonical setting; quote {CANONICAL_RUN} ({r['canonical_top1_mean']:.2f}); minus X2: {diffs} "
                               "(numerical, see the notes)")
    same = [x for x in T.rows if x.get("minus_canonical") is not None and x.get("canonical_note", "").startswith("the canonical")]
    svd_rows = [x for x in T.rows if x.get("minus_canonical") is not None and (x["ablation"], x["variant"]) == ("A1", "svd")]
    svd_same = bool(svd_rows) and all(x["minus_canonical"] == 0 for x in svd_rows)
    if same:
        lo, hi = min(abs(x["minus_canonical"]) for x in same), max(abs(x["minus_canonical"]) for x in same)
        nq = int(same[0].get("n_query") or 0)
        note = (f"Canonical number: the default cell (A1 per_layer = A2 mirror = A4 first = A5 alpha1.00, and A3 at the "
                f"run's own fit set) is the same setting as {LABEL_MIRROR_MLP} in T_main_zeroshot, whose {CANONICAL_RUN} "
                f"value is the one to quote (column X2 (canonical)). stage1_ablate.py " + OPERATOR_ROUNDTRIP +
                f": the two differ by {lo:.3f} to {hi:.3f} points per fit seed"
                + (f" ({round(lo * nq / 100)} to {round(hi * nq / 100)} of the {nq:,} queries). " if nq else ". ")
                + ("A1 svd, which does not use the operators, is the identical model in both scripts and gives the "
                   "identical number at every fit seed. " if svd_same else "")
                + "Ablation rows are compared with each other, so they keep their own values.")
        T.notes.append(note)
        S.notes.append(note)


def ablation_budget_notes(T: Table, S: Table) -> None:
    """Data-driven notes on unmatched parameter budgets (A1, A2 alternatives; A4 block choices; A5 interpolation)."""
    for model in sorted({r["model"] for r in S.rows}):
        lab = MODEL_LABEL.get(model, model)
        for k in sorted({r["k"] for r in S.rows if r["model"] == model}):
            rows = {(r["ablation"], r["variant"]): r for r in S.rows if r["model"] == model and r["k"] == k}
            first = rows.get(("A4", "first")) or rows.get(("A1", "per_layer"))
            a4 = [rows[("A4", v)] for v in ("last", "lowest_share") if ("A4", v) in rows]
            if a4 and first is not None:
                per = {(x["variant"]): x for x in T.rows if x["model"] == model and x["k"] == k and x["ablation"] == "A4"}
                bl = {v: per[v]["blocks"] for v in per}
                last = rows.get(("A4", "last"))
                bits = []
                if last is not None and last.get("params_vs_first_k_mean") is not None:
                    bits.append(f"last {k} keeps {last['mlp_kept_frac_mean']:.2f} of the MLP weights of its blocks against "
                                f"{first['mlp_kept_frac_mean']:.2f} for first {k} ({last['params_vs_first_k_mean']:.2f} "
                                f"times the first-k count), so \"last k beats first k\" ({last['top1_mean']:.2f} against "
                                f"{first['top1_mean']:.2f}) partly reflects more weights kept, as well as the late blocks "
                                "being closer to equivariant")
                if "lowest_share" in bl and "last" in bl and bl["lowest_share"] == bl["last"]:
                    bits.append(f"lowest breaking share selects the same blocks as last ({bl['last']}), so it is the "
                                "same model")
                if bits:
                    note = f"A4 ({lab}, k = {k}) is not parameter-matched: " + "; ".join(bits) + "."
                    T.notes.append(note)
                    S.notes.append(note)
            more = [r for (abl, v), r in rows.items() if abl in ("A1", "A2") and r.get("params_vs_first_k_mean") is not None
                    and r["params_vs_first_k_mean"] > 1 + PARAM_MATCH_TOL]
            less = [r for (abl, v), r in rows.items() if abl in ("A1", "A2") and r.get("params_vs_first_k_mean") is not None
                    and r["params_vs_first_k_mean"] < 1 - PARAM_MATCH_TOL]
            if more or less:
                parts = []
                if more:
                    parts.append("alternatives that keep more MLP weights than the default and still collapse: " + ", ".join(
                        f"{r['ablation']} {r['variant']} {r['params_vs_first_k_mean']:.2f}x ({r['top1_mean']:.2f})"
                        for r in sorted(more, key=lambda r: -r["params_vs_first_k_mean"])))
                if less:
                    parts.append("fewer: " + ", ".join(
                        f"{r['ablation']} {r['variant']} {r['params_vs_first_k_mean']:.2f}x ({r['top1_mean']:.2f})"
                        for r in sorted(less, key=lambda r: r["params_vs_first_k_mean"])))
                d = rows.get(("A1", "per_layer"))
                note = (f"A1 and A2 ({lab}, k = {k}): not every alternative keeps the default's parameter count "
                        f"(default {d['top1_mean']:.2f} at 1.00x); " if d else
                        f"A1 and A2 ({lab}, k = {k}): not every alternative keeps the default's parameter count; ") + \
                    "; ".join(parts) + ". Extra weights do not rescue a wrong operator or pairing."
                T.notes.append(note)
                S.notes.append(note)
            a5 = [r for (abl, v), r in rows.items() if abl == "A5" and r.get("params_vs_first_k_mean") is not None
                  and r["params_vs_first_k_mean"] > 1 + PARAM_MATCH_TOL]
            if a5:
                note = (f"A5 ({lab}, k = {k}): alpha below 1 interpolates between the original and the projected weights, "
                        "which has no exact structure, so those rows keep the dense count "
                        f"({a5[0]['params_vs_first_k_mean']:.2f}x the default); only alpha = 1 is at the Mirror budget.")
                T.notes.append(note)
                S.notes.append(note)


def local_run_dir(P, path) -> Path | None:
    """A run folder named in a result file (often an absolute path on the GPU machine, e.g. /workspace/<repo>/runs/...)
    as a folder of this repository: the path itself when it exists here, else its part from the last runs/ on."""
    if not path:
        return None
    s_ = str(path).replace(chr(92), "/")
    p_ = Path(s_)
    if p_.is_absolute() and p_.is_dir():
        return p_
    if not s_.startswith("/") and (P.root / s_).is_dir():
        return P.root / s_
    m_ = list(re.finditer(r"(?:^|/)(runs/.+)$", s_))
    if m_ and (P.root / m_[-1][1]).is_dir():
        return P.root / m_[-1][1]
    return None


def _runs_path_in(obj):
    """First string in a nested JSON naming a run folder known to the collector, and its fit seed."""
    stack = [obj]
    while stack:
        o = stack.pop(0)
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
        elif isinstance(o, str):
            fs = fit_seed_from_path(o)
            if fs is not None:
                return o, fs
    return None, None


T2_MIRROR = {("mlp", "mirror"), ("m1", "vlocal"), ("m1", "vexact"), ("healed", "mirror")}


def _t2_margins(T: Table, keys: tuple) -> None:
    """Drop against the original and margins of Mirror rows over SVD and random of the same family and k."""
    for r in T.rows:
        same = [x for x in T.rows if all(x.get(k) == r.get(k) for k in keys)]
        orig = next((x for x in same if x["family"] == "original"), None)
        if orig is not None and r.get("top1") is not None:
            r["drop_vs_original"] = orig["top1"] - r["top1"]
            r["_derived"]["drop_vs_original"] = f"top1 of original minus top1 of {r['variant']} ({r['_src']})"
        if (r.get("family"), r.get("method")) in T2_MIRROR:
            for base in ("svd", "random"):
                b = next((x for x in same if x["family"] == r["family"] and x["method"] == base and x["k"] == r["k"]), None)
                if b is not None:
                    r[f"margin_{base}"] = r["top1"] - b["top1"]
                    r["_derived"][f"margin_{base}"] = f"top1 of {r['variant']} minus top1 of {b['variant']} ({r['_src']})"


def build_tier2(P, Tm: Table, Th: Table, m1: dict, x2: dict, Ta: Table | None = None, Tv: Table | None = None):
    T = Table("T_tier2", "Tier 2: ImageNet linear probe (runs/t2/t2_linear.csv)", ["model", "variant"],
              [("model", "model_label", ""), ("variant", "variant", ""), ("method", "label", ""), ("k", "k", "d"),
               ("top-1", "top1", ".2f"),
               ("top-5", "top5", ".2f"), ("drop", "drop_vs_original", ".2f"), ("vs SVD", "margin_svd", ".2f"),
               ("vs random", "margin_random", ".2f"), ("chosen wd", "chosen_wd", "g"), ("wd at grid edge", "wd_at_grid_edge", ""),
               ("k-NN check", "knn_check_top1", ".2f"), ("Tier-1 k-NN", "tier1_knn_top1", ".2f"),
               ("same model as Tier 1", "tier1_same_model", ""), ("model params", "model_params", ",d")])
    T.notes = ["Logistic regression on [CLS, mean patch token] trained on the 100k bank, tested on all 50k validation "
               "images; weight decay chosen on a 10% split of the bank (scripts/tier2_linear.py). Drop: original minus "
               "variant; vs SVD and vs random: Mirror variant minus the SVD or random variant of the same family and k "
               "(empty where that baseline was not probed). k-NN check: k-NN on the probe's own cached CLS features; "
               "Tier-1 k-NN: the same setting's canonical number from the k-NN tables (source in tier1_source). Same "
               "model as Tier 1: whether the probe built the model behind the Tier-1 number, decided from how each was "
               "built (tier1_note in the CSV and the notes below say why not). Method: one label per method in every "
               "table (the label the probe script recorded is in label_recorded)."]
    X = Table("T_tier2_transfer", "Tier 2: transfer linear probes (runs/t2/t2_transfer.csv)", ["model", "dataset", "variant"],
              [("model", "model_label", ""), ("dataset", "dataset", ""), ("variant", "variant", ""), ("method", "label", ""),
               ("k", "k", "d"),
               ("top-1", "top1", ".2f"), ("top-5", "top5", ".2f"), ("mean per-class", "mean_per_class_top1", ".2f"),
               ("drop", "drop_vs_original", ".2f"), ("vs SVD", "margin_svd", ".2f"), ("vs random", "margin_random", ".2f"),
               ("chosen wd", "chosen_wd", "g"), ("train / test", lambda r: f"{fmt(r.get('n_train'), ',d')} / "
                                                                         f"{fmt(r.get('n_test'), ',d')}", ""),
               ("at ceiling", lambda r: "yes" if r.get("dataset_at_ceiling") else "", "")])
    X.notes = ["Same probe recipe as the ImageNet probe, trained on each dataset's training split and scored once on "
               "its test split (scripts/tier2_transfer.py; dataset splits marked (verify on VM) in that script)."]
    f = P.runs / "t2" / "t2_linear.csv"
    df = read_csv(f)
    if df is not None:
        for _, r in df.iterrows():
            model, var = str(r["model"]), str(r["variant"])
            vj = read_json(P.runs / "t2" / "probe" / model / f"{var}_variant.json") or \
                read_json(P.runs / "t2" / "probe" / model / f"{var}.json") or {}
            ops, fs = _runs_path_in(vj)
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": var,
                   "label": method_label(r.get("family"), r.get("method"), r.get("label")), "label_recorded": r.get("label"),
                   "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")),
                   "builder": ((vj.get("info") or {}).get("builder")), "variant_seed": (vj.get("info") or {}).get("random_seed"),
                   "fit_seed": fs, "structure_source": ops, "source": P.rel(f), "git_hash": r.get("git_hash"),
                   "seed": py(r.get("seed")), "date": r.get("date"), "_src": P.rel(f), "_git": r.get("git_hash"),
                   "_seed": py(r.get("seed")), "_fit_seed": fs, "_loc": {}, "_derived": {}}
            for c in ("top1", "top5", "mean_per_class_top1", "knn_check_top1", "chosen_wd", "wd_at_grid_edge",
                      "holdout_top1_at_chosen", "final_hit_iter_limit", "model_params", "converted_params_kept",
                      "converted_params_dense", "n_bank", "n_val", "feature_dim", "feature_kind", "feature_dtype", "amp"):
                if c in r:
                    row[c] = py(r[c])
                    if c not in ("wd_at_grid_edge", "final_hit_iter_limit", "feature_kind", "feature_dtype", "amp"):
                        row["_loc"][c] = csv_loc(P.rel(f), {"model": model, "variant": var}, c)
            # the same variant's Tier-1 k-NN number
            fam, meth, k = row["family"], row["method"], row["k"]
            t1, t1src, t1loc = None, "", None
            if fam == "original":
                o = Tm.find(model=model, k=0, variant="original")
                if o is not None and o.get("top1_mean") is not None:
                    t1, t1src = o["top1_mean"], f"T_main_zeroshot original ({o.get('source_files')})"
                    t1loc = o["_loc"].get("top1_mean")
            elif fam == "healed":
                h = Th.find(method=meth)
                if h is not None and h.get("k") == k and h.get("healed_top1") is not None:
                    t1, t1src, t1loc = h["healed_top1"], f"T_healing healed ({h.get('source')})", h["_loc"].get("healed_top1")
            elif fam == "healed_vexact":
                # the healing run the probe rebuilt (variant JSON), when it is on this machine: runs/m3_vexact (A100) and
                # runs/m3_vexact_5090 hold different checkpoints of the same k
                run = ((vj.get("info") or {}).get("vexact_run") or (vj.get("sources") or {}).get("vexact_run"))
                rp = local_run_dir(P, run)
                kj = read_json(rp / "knn.json") if rp is not None else None
                if kj and isinstance(kj.get("healed"), dict) and kj["healed"].get("top1") is not None:
                    t1, t1src = py(kj["healed"]["top1"]), f"knn.json of the healing run the probe rebuilt ({P.rel(rp)})"
                    t1loc = json_loc(P.rel(rp / "knn.json"), ["healed", "top1"])
                    row["healing_run"] = P.rel(rp)
                elif Tv is not None:
                    h = Tv.find(k=k)
                    if h is not None and h.get("healed_top1") is not None and not h.get("tiny"):
                        t1, t1src = h["healed_top1"], f"T_healing_vexact healed ({h.get('source')})"
                        t1loc = h["_loc"].get("healed_top1")
                        row["healing_run"] = (f"not recorded on this machine ({str(run).replace(chr(92), '/')})" if run
                                              else "not recorded on this machine (no variant JSON)")
            elif fam == "m1" and m1.get("knn") is not None and m1.get("model") == model:
                sel = m1["knn"][(m1["knn"]["k"] == k) & (m1["knn"]["variant"] == meth)]
                if len(sel) == 1:
                    t1, t1src = float(sel["top1"].iloc[0]), m1["knn_file"]
                    t1loc = csv_loc(m1["knn_file"], {"k": k, "variant": meth}, "top1")
            elif fam == "mlp" and fs is not None:
                run = next((x for x in x2.values() if x["model"] == model and x["fit_seed"] == fs
                            and x["operator_mode"] == "per_layer" and x["surgery"] is not None), None)
                sv = {"mirror": "surgery", "svd": "svd", "random": "random"}.get(meth)
                if run is not None and sv:
                    s = run["surgery"]
                    sel = s[(s["k"] == k) & (s["variant"] == sv)]
                    if len(sel) == 1:
                        t1, t1src = float(sel["top1"].iloc[0]), run["surgery_file"]
                        t1loc = csv_loc(run["surgery_file"], {"k": k, "variant": sv}, "top1")
            row["tier1_knn_top1"], row["tier1_source"] = t1, t1src
            if t1loc:
                row["_loc"]["tier1_knn_top1"] = t1loc
            elif t1 is not None:
                row["_derived"]["tier1_knn_top1"] = t1src
            tier1_note(row, Ta)
            T.add(row)
        _t2_margins(T, ("model",))
    # each t2_*_meta_<model>.json describes the last run for that model only; earlier rows keep their own git hash and date
    for mf in sorted((P.runs / "t2").glob("t2_linear_meta_*.json")):
        mj = read_json(mf) or {}
        mm, mv = mj.get("model"), list(mj.get("variants") or [])
        rows_m = [r for r in T.rows if r["model"] == mm]
        if not rows_m or not mv:
            continue
        gpu = (mj.get("env") or {}).get("gpu")
        for r in rows_m:
            if r["variant"] in mv:
                r["run_gpu"] = gpu
        older = sorted({r["variant"] for r in rows_m if r["variant"] not in mv})
        if older:
            T.notes.append(
                f"{MODEL_LABEL.get(mm, mm)}: `{P.rel(mf)}` records only its last run ({', '.join(mv)}; git "
                f"{short((mj.get('git') or {}).get('git_hash'))}, {gpu or 'GPU not recorded'}). The other rows "
                f"({', '.join(older)}) come from earlier runs: their git hash and date are in the CSV, their run meta in "
                "git history. Comparisons across runs (for example a drop against the original) include the run-to-run "
                "spread between GPUs.")
    # once, after the per-model run notes (these were repeated for every run meta file before 2026-11-10)
    for r in [r for r in T.rows if r.get("tier1_note")]:
        T.notes.append(f"{r['model_label']} {r['variant']} ({r['label']}): {r['tier1_note']}")
    gaps = [r for r in T.rows if r.get("tier1_check_gap") is not None]
    if gaps:
        g = max(gaps, key=lambda r: abs(r["tier1_check_gap"]))
        T.notes.append("Where the probe and Tier 1 use the same model, the k-NN check is computed on the probe's own "
                       f"stored features (feature_dtype {g.get('feature_dtype') or 'as recorded'}) and differs from the "
                       f"Tier-1 value by at most {abs(g['tier1_check_gap']):.3f} points ({g['model_label']} "
                       f"{g['variant']}); {len(gaps)} of {sum(1 for r in T.rows if r.get('tier1_same_model') == 'yes')} "
                       "such rows differ at all. Quote the Tier-1 value for k-NN.")
    f2 = P.runs / "t2" / "t2_transfer.csv"
    dx = read_csv(f2)
    if dx is not None:
        for _, r in dx.iterrows():
            model, var, ds = str(r["model"]), str(r["variant"]), str(r["dataset"])
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "dataset": ds, "variant": var,
                   "label": method_label(r.get("family"), r.get("method"), r.get("label")), "label_recorded": r.get("label"),
                   "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")),
                   "source": P.rel(f2), "git_hash": r.get("git_hash"), "seed": py(r.get("seed")), "_src": P.rel(f2),
                   "_git": r.get("git_hash"), "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
            for c in ("top1", "top5", "mean_per_class_top1", "chosen_wd", "wd_at_grid_edge", "n_train", "n_test",
                      "num_classes", "hf_id"):
                if c in r:
                    row[c] = py(r[c])
                    if c in ("top1", "top5", "mean_per_class_top1", "chosen_wd"):
                        row["_loc"][c] = csv_loc(P.rel(f2), {"model": model, "dataset": ds, "variant": var}, c)
            X.add(row)
        _t2_margins(X, ("model", "dataset"))
        transfer_ceiling(X)
    # each t2_*_meta_<model>.json describes the last run for that model only; earlier rows keep their own git hash and date
    for mf in sorted((P.runs / "t2").glob("t2_transfer_meta_*.json")):
        mj = read_json(mf) or {}
        mm, mv = mj.get("model"), list(mj.get("variants") or [])
        rows_m = [r for r in X.rows if r["model"] == mm]
        if not rows_m or not mv:
            continue
        gpu = (mj.get("env") or {}).get("gpu")
        for r in rows_m:
            if r["variant"] in mv:
                r["run_gpu"] = gpu
        older = sorted({r["variant"] for r in rows_m if r["variant"] not in mv})
        if older:
            X.notes.append(
                f"{MODEL_LABEL.get(mm, mm)}: `{P.rel(mf)}` records only its last run ({', '.join(mv)}; git "
                f"{short((mj.get('git') or {}).get('git_hash'))}, {gpu or 'GPU not recorded'}). The other rows "
                f"({', '.join(older)}) come from earlier runs: their git hash and date are in the CSV, their run meta in "
                "git history. Comparisons across runs (for example a drop against the original) include the run-to-run "
                "spread between GPUs.")
    return T, X


def tier1_note(row: dict, Ta: Table | None) -> None:
    """Whether the probed model is the model behind the Tier-1 number, decided from how each was built (not from the
    values): the probe script rebuilds the MLP-only variants with mirror.variants from the float32 operator file (the
    Mirror row is then numerically different from X2's, the random row is another draw); every other row probes the
    same model (original, M1 structures, healed checkpoints). The k-NN check itself runs on the probe's own stored
    features, so it can differ from the Tier-1 value by a few queries even for the same model."""
    chk, t1 = row.get("knn_check_top1"), row.get("tier1_knn_top1")
    if chk is None or t1 is None:
        return
    fam, meth, fs, k = row.get("family"), row.get("method"), row.get("fit_seed"), row.get("k")
    built = f"built by {row['builder']}" if row.get("builder") else "built by mirror.variants"
    if fam == "mlp" and meth == "mirror":
        row["tier1_same_model"] = "no (numerical)"
        a1 = Ta.find(model=row["model"], ablation="A1", variant="per_layer", k=k, fit_seed=fs) if Ta is not None else None
        same_a1 = a1 is not None and a1.get("top1") is not None and abs(a1["top1"] - chk) <= 1e-9
        row["tier1_note"] = (f"the probed model was {built} from the float32 operator file"
                             + (f", the same model as A1 per_layer at fit seed {fs} (k-NN {a1['top1']:.3f})" if same_a1 else "")
                             + f"; the Tier-1 column is the canonical {CANONICAL_RUN} value ({t1:.3f}), {chk - t1:+.3f} "
                             "points away. The difference is numerical: the probe script projects with the operators "
                             f"reloaded from the float32 operator file ({CANONICAL_RUN} uses the float64 operators as "
                             "fitted; same pairing and parameter count), and its k-NN check runs on its own stored "
                             "features.")
    elif fam == "mlp" and meth == "random":
        row["tier1_same_model"] = "no (other draw)"
        sd = f" (variant seed {row['variant_seed']})" if row.get("variant_seed") is not None else ""
        row["tier1_note"] = (f"different random draw: the probed control was {built}{sd}, with random involutions that fix "
                             f"the all-ones direction; the Tier-1 column is the {CANONICAL_RUN} control of run_e1.py, another "
                             f"draw ({t1:.2f} against {chk:.2f}; both at chance). The probe's margins over random use "
                             "its own draw.")
    elif fam == "healed_vexact" and str(row.get("healing_run", "")).startswith("not recorded"):
        row["tier1_same_model"] = "unverified"
        row["tier1_note"] = (f"the healing run this probe rebuilt is {row['healing_run']}, so the Tier-1 column is the "
                             f"V-exact healing table's value at k = {k} ({t1:.2f}) and may come from another healing run of "
                             f"the same k (runs/m3_vexact and runs/m3_vexact_5090 hold different checkpoints); the probe's "
                             f"own k-NN check is {chk:.2f}.")
    else:
        row["tier1_same_model"] = "yes"
        if abs(chk - t1) > 1e-9:
            row["tier1_check_gap"] = chk - t1
            row["_derived"]["tier1_check_gap"] = "knn_check_top1 minus tier1_knn_top1"


def transfer_ceiling(X: Table) -> None:
    """Flag transfer datasets where the original is at ceiling and variants tie with it, so they do not separate the
    methods (rule: original top-1 >= 99%, or at least two other variants with exactly the original's top-1)."""
    for ds in sorted({r["dataset"] for r in X.rows}):
        for model in sorted({r["model"] for r in X.rows if r["dataset"] == ds}):
            rows = [r for r in X.rows if r["dataset"] == ds and r["model"] == model and r.get("top1") is not None]
            o = next((r for r in rows if r.get("family") == "original"), None)
            if o is None:
                continue
            others = [r for r in rows if r is not o]
            ties = [r for r in others if r["top1"] == o["top1"]]
            if not (o["top1"] >= 99.0 or len(ties) >= 2):
                continue
            span = max([abs(r["top1"] - o["top1"]) for r in others] + [0.0])
            n = o.get("n_test")
            imgs = f" ({round(span * n / 100):,} of {n:,} test images)" if isinstance(n, (int, float)) and n else ""
            for r in rows:
                r["dataset_at_ceiling"] = True
            X.notes.append(f"{ds} ({o['model_label']}) does not separate the methods: the original is at "
                           f"{o['top1']:.2f}%, " + (f"{', '.join(r['variant'] for r in ties)} give exactly the same top-1, "
                                                     if ties else "")
                           + f"and no variant differs from the original by more than {span:.2f} points{imgs}. Its "
                           "margins are not evidence for or against any method.")


def build_tier2_seg(P, plan_text: str = ""):
    """Tier 2 ADE20K linear segmentation: runs/t2_seg/ written by scripts/tier2_ade20k.py (t2_ade20k.csv, one row
    per model and variant; seg/<model>/<variant>.json with per-class IoU; t2_ade20k_meta_<model>.json), and
    runs/t2_seg_rerun/ (the same script run again on some variants: the run-to-run spread, T_tier2_seg_rerun). Smoke-test
    folders such as runs/t2_seg_smoke are never read."""
    T = Table("T_tier2_seg", "Tier 2: ADE20K linear segmentation (runs/t2_seg/t2_ade20k.csv)", ["model", "variant"],
              [("model", "model_label", ""), ("variant", "variant", ""), ("k", "k", "d"), ("mIoU", "mIoU", ".2f"),
               ("aAcc", "aAcc", ".2f"), ("mAcc", "mAcc", ".2f"),
               ("change vs original (mIoU)", "delta_miou_vs_original", "+.2f"),
               ("vs SVD (mIoU)", "margin_svd_miou", "+.2f"), ("vs random (mIoU)", "margin_random_miou", "+.2f"),
               ("params", "model_params", ",d"), ("kept (converted)", "kept_frac_converted", ".3f"),
               ("iterations", "iterations", ",d"),
               ("val images", lambda r: f"{fmt(r.get('n_val'), ',d')} / {fmt(r.get('n_val_total'), ',d')}"
                if r.get("n_val") is not None else "", ""),
               ("V-exact error (bar)", lambda r: f"{fmt(r.get('vexact_max_rel_err'), '.1e')} ({fmt(r.get('vexact_bar'), 'g')})"
                if r.get("vexact_max_rel_err") is not None else "", ""),
               ("git", lambda r: short(r.get("git_hash")), ""), ("note", "note", "")])
    T.notes = ["Frozen backbone, linear head (BatchNorm when the bn column is true, then a 1 x 1 convolution) on the "
               "last-layer patch tokens; one recipe and seed for every variant (scripts/tier2_ade20k.py; crop, "
               "iterations, batch and optimiser settings are columns of the CSV and of t2_ade20k_meta_<model>.json). "
               "mIoU, aAcc (pixel accuracy) and mAcc (mean class accuracy) in % on ADE20K validation.",
               "Change vs original: mIoU of the variant minus that of the original model (points; negative is worse). "
               "vs SVD and vs random: mIoU of a Mirror variant minus that of the SVD or random variant of the same "
               "family and k (empty where that baseline was not run). Params: whole-model parameters as recorded; "
               "kept: kept over dense matrix parameters of the converted blocks.",
               "V-exact rows: the model is rebuilt at the segmentation crop; the error is the largest relative error "
               "inside the span measured by the script on training crops (fp32), with its bar in brackets. Per-class "
               "IoU and accuracy: T_tier2_seg_classes.csv."]
    C = Table("T_tier2_seg_classes", "Tier 2 ADE20K: per-class IoU and accuracy (seg/<model>/<variant>.json)",
              ["model", "variant", "class_index"],
              [("model", "model", ""), ("variant", "variant", ""), ("class", "class_index", "d"),
               ("ADE20K label", "ade20k_label", "d"), ("IoU", "iou", ".4f"), ("accuracy", "acc", ".4f")])
    C.notes = ["IoU and accuracy as fractions (0 to 1), as stored in metrics.per_class_iou and per_class_acc; empty "
               "where the class was not scored. Class index c is ADE20K label c + 1 (label 0, other objects, is "
               "ignored); class names are not in the result files."]
    base = P.runs / "t2_seg"
    info = {"dir": P.rel(base), "csv": P.rel(base / "t2_ade20k.csv"), "meta": {}, "missing": {}, "plan_iters": None}
    m = re.search(r"ADE20K linear segmentation[^\n]*?([\d,]+) iterations", plan_text or "")
    if m:
        info["plan_iters"] = int(m[1].replace(",", ""))
    for mf in sorted(base.glob("t2_ade20k_meta_*.json")) if base.is_dir() else []:
        mj = read_json(mf)
        if mj:
            info["meta"][str(mj.get("model") or mf.stem.replace("t2_ade20k_meta_", ""))] = {"file": P.rel(mf), "json": mj}
    R = Table("T_tier2_seg_rerun", "Tier 2 ADE20K: the same variant run twice (runs/t2_seg and runs/t2_seg_rerun)",
              ["model", "variant"],
              [("model", "model_label", ""), ("variant", "variant", ""), ("mIoU: first", "first_mIoU", ".3f"),
               ("mIoU: rerun", "rerun_mIoU", ".3f"), ("rerun minus first", "diff_mIoU", "+.3f"),
               ("aAcc: first / rerun", lambda r: f"{fmt(r.get('first_aAcc'), '.3f')} / {fmt(r.get('rerun_aAcc'), '.3f')}", ""),
               ("mAcc: first / rerun", lambda r: f"{fmt(r.get('first_mAcc'), '.3f')} / {fmt(r.get('rerun_mAcc'), '.3f')}", ""),
               ("GPU: first / rerun", lambda r: f"{r.get('first_gpu') or 'not recorded'} / {r.get('rerun_gpu') or 'not recorded'}", ""),
               ("git: first / rerun", lambda r: f"{short(r.get('first_git'))} / {short(r.get('rerun_git'))}", "")])
    R.notes = ["The same variant trained and scored again with the same script, recipe and seed. scripts/tier2_ade20k.py "
               "notes that CUDA's bilinear-upsampling backward uses atomic adds, so GPU reruns can differ; the difference "
               "is one measured run-to-run spread (one pair of runs, a rough estimate). GPU: env.gpu of the folder's "
               "t2_ade20k_meta_<model>.json when that file lists the variant (it describes the folder's last run only)."]
    info["rerun_csv"] = P.rel(P.runs / "t2_seg_rerun" / "t2_ade20k.csv")
    df = read_csv(base / "t2_ade20k.csv")
    if df is None:
        return T, C, R, info
    f = P.rel(base / "t2_ade20k.csv")
    for _, r in df.iterrows():
        model, var = str(r["model"]), str(r["variant"])
        where = {"model": model, "variant": var}
        vpath = base / "seg" / model / f"{var}.json"
        vj = read_json(vpath) or {}
        ops, fs = _runs_path_in(vj.get("sources") or {})
        row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": var,
               "label": method_label(r.get("family"), r.get("method"), r.get("label")), "label_recorded": r.get("label"),
               "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")), "fit_seed": fs,
               "structure_source": ops, "source": f, "per_class_source": P.rel(vpath) if vj else None,
               "git_hash": r.get("git_hash"), "git_dirty": py(r.get("git_dirty")), "seed": py(r.get("seed")),
               "date": r.get("date"), "_src": f, "_git": r.get("git_hash"), "_seed": py(r.get("seed")), "_fit_seed": fs,
               "_loc": {}, "_derived": {}}
        for c in ("mIoU", "aAcc", "mAcc", "iterations", "batch_size", "crop", "lr", "weight_decay", "warmup_iters", "bn",
                  "eval_stride", "n_train", "n_val", "n_val_total", "final_train_loss", "vexact_built_px",
                  "vexact_max_rel_err", "vexact_224_build_rel_err_at_seg", "converted_params_kept",
                  "converted_params_dense", "model_params", "t_build_s", "t_check_s", "t_train_s", "t_eval_s",
                  "t_total_s", "train_img_per_s", "resumed_from_iter", "peak_mem_gb", "amp", "hf_id", "hf_revision"):
            if c in r:
                row[c] = py(r[c])
                if c not in ("bn", "amp", "hf_id", "hf_revision") and row[c] is not None:
                    row["_loc"][c] = csv_loc(f, where, c)
        dense = row.get("converted_params_dense")
        if dense and row.get("converted_params_kept") is not None:
            row["kept_frac_converted"] = row["converted_params_kept"] / dense
            row["_derived"]["kept_frac_converted"] = "converted_params_kept / converted_params_dense (t2_ade20k.csv)"
        vc = (vj.get("vexact_checks") or {}).get("at_build_resolution") or {}
        if vc:
            row["vexact_bar"], row["vexact_pass"] = py(vc.get("bar")), py(vc.get("pass"))
            row["_loc"]["vexact_bar"] = json_loc(P.rel(vpath), ["vexact_checks", "at_build_resolution", "bar"])
        met = vj.get("metrics") or {}
        iou = met.get("per_class_iou")
        if isinstance(iou, list):
            row["n_classes_scored"] = py(met.get("n_classes_scored"))
            row["_loc"]["n_classes_scored"] = json_loc(P.rel(vpath), ["metrics", "n_classes_scored"])
            vals = [float(x) for x in iou if x is not None]
            if vals:
                row["miou_from_per_class"] = 100.0 * float(np.mean(vals))
                row["_derived"]["miou_from_per_class"] = "100 x mean of the non-empty per_class_iou values (variant JSON)"
                if row.get("mIoU") is not None:
                    row["per_class_matches_miou"] = abs(row["miou_from_per_class"] - row["mIoU"]) <= 1e-6 * max(1.0, row["mIoU"])
            acc = met.get("per_class_acc") if isinstance(met.get("per_class_acc"), list) else []
            for i, x in enumerate(iou):
                cr = C.add({"model": model, "variant": var, "class_index": i, "ade20k_label": i + 1, "iou": py(x),
                            "acc": py(acc[i]) if i < len(acc) else None, "_src": P.rel(vpath), "_git": row["_git"],
                            "_seed": row["_seed"], "_fit_seed": fs})
                cr["_loc"]["iou"] = json_loc(P.rel(vpath), ["metrics", "per_class_iou", i])
                if i < len(acc):
                    cr["_loc"]["acc"] = json_loc(P.rel(vpath), ["metrics", "per_class_acc", i])
        notes = []
        if row.get("n_val") is not None and row.get("n_val_total") and row["n_val"] < row["n_val_total"]:
            notes.append(f"partial validation ({fmt(row['n_val'], ',d')} of {fmt(row['n_val_total'], ',d')} images)")
        if info["plan_iters"] and row.get("iterations") is not None and int(row["iterations"]) != info["plan_iters"]:
            notes.append(f"{fmt(row['iterations'], ',d')} iterations (plan {info['plan_iters']:,})")
        if row.get("vexact_pass") is False:
            notes.append("V-exact error above its bar")
        if row.get("family") == "healed":
            notes.append("healed at 224 px")
        if row.get("resumed_from_iter"):
            notes.append(f"training resumed at iteration {fmt(row['resumed_from_iter'], 'd')}")
        row["note"] = "; ".join(notes)
        T.add(row)
    # change against the original and margins of Mirror variants over SVD and random (same model, family and k)
    for r in T.rows:
        same = [x for x in T.rows if x["model"] == r["model"]]
        orig = next((x for x in same if x.get("family") == "original"), None)
        if orig is not None and r is not orig and r.get("mIoU") is not None and orig.get("mIoU") is not None:
            r["delta_miou_vs_original"] = r["mIoU"] - orig["mIoU"]
            r["_derived"]["delta_miou_vs_original"] = f"mIoU of {r['variant']} minus mIoU of {orig['variant']} ({r['_src']})"
        if (r.get("family"), r.get("method")) in T2_MIRROR and r.get("mIoU") is not None:
            for b in ("svd", "random"):
                x = next((x for x in same if x.get("family") == r["family"] and x.get("method") == b
                          and x.get("k") == r.get("k") and x.get("mIoU") is not None), None)
                if x is not None:
                    r[f"margin_{b}_miou"] = r["mIoU"] - x["mIoU"]
                    r["_derived"][f"margin_{b}_miou"] = f"mIoU of {r['variant']} minus mIoU of {x['variant']} ({r['_src']})"
    for model, mm in info["meta"].items():
        have = {r["variant"] for r in T.rows if r["model"] == model}
        miss = [v for v in (mm["json"].get("variants") or []) if v not in have]
        if miss:
            info["missing"][model] = miss
    seg_rerun(P, T, R, info)
    return T, C, R, info


def seg_rerun(P, T: Table, R: Table, info: dict) -> None:
    """runs/t2_seg_rerun/t2_ade20k.csv: each rerun variant next to its first run (rerun_* columns of T_tier2_seg, one
    row of T_tier2_seg_rerun) and a data-driven note on the run-to-run spread."""
    rb = P.runs / "t2_seg_rerun"
    dr = read_csv(rb / "t2_ade20k.csv")
    if dr is None:
        return
    fr = P.rel(rb / "t2_ade20k.csv")

    def gpu_of(folder: Path, model: str, var: str):
        mj = read_json(folder / f"t2_ade20k_meta_{model}.json") or {}
        return (mj.get("env") or {}).get("gpu") if var in (mj.get("variants") or []) else None

    for _, r in dr.iterrows():
        model, var = str(r["model"]), str(r["variant"])
        where = {"model": model, "variant": var}
        first = T.find(model=model, variant=var)
        row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": var, "rerun_source": fr,
               "rerun_git": r.get("git_hash"), "rerun_date": r.get("date"), "rerun_gpu": gpu_of(rb, model, var),
               "_src": fr, "_git": r.get("git_hash"), "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
        for c in ("mIoU", "aAcc", "mAcc"):
            if c in r and py(r[c]) is not None:
                row[f"rerun_{c}"] = py(r[c])
                row["_loc"][f"rerun_{c}"] = csv_loc(fr, where, c)
        if first is not None:
            row.update({"first_source": first["_src"], "first_git": first.get("git_hash"), "first_date": first.get("date"),
                        "first_gpu": gpu_of(P.runs / "t2_seg", model, var)})
            row["_src"] = f"{first['_src']};{fr}"
            row["_git"] = f"{first.get('_git')};{r.get('git_hash')}"
            for c in ("mIoU", "aAcc", "mAcc"):
                if first.get(c) is not None:
                    row[f"first_{c}"] = first[c]
                    if first["_loc"].get(c):
                        row["_loc"][f"first_{c}"] = first["_loc"][c]
                if row.get(f"first_{c}") is not None and row.get(f"rerun_{c}") is not None:
                    row[f"diff_{c}"] = row[f"rerun_{c}"] - row[f"first_{c}"]
                    row["_derived"][f"diff_{c}"] = f"{c} of the rerun minus that of the first run"
                    first[f"rerun_{c}"] = row[f"rerun_{c}"]
                    first["_loc"][f"rerun_{c}"] = row["_loc"][f"rerun_{c}"]
                    first[f"rerun_minus_first_{c}"] = row[f"diff_{c}"]
                    first["_derived"][f"rerun_minus_first_{c}"] = f"{c} of runs/t2_seg_rerun minus that of runs/t2_seg"
        R.add(row)
    spread = [r for r in R.rows if r.get("diff_mIoU") is not None]
    if spread:
        big = max(spread, key=lambda r: abs(r["diff_mIoU"]))
        T.notes.append("Run-to-run spread (T_tier2_seg_rerun): " + "; ".join(
            f"{r['variant']} measured {r['first_mIoU']:.2f} and {r['rerun_mIoU']:.2f} mIoU ({r['diff_mIoU']:+.2f})" for r in spread)
            + f". Differences between variants of {abs(big['diff_mIoU']):.2f} mIoU or less are within this one measured "
              "rerun difference; columns rerun_* hold the rerun values.")
    for r in R.rows:
        if r.get("first_mIoU") is None:
            T.notes.append(f"{r['variant']} ({r['model_label']}) has a rerun in `{fr}` but no first run in `runs/t2_seg`.")


M2_SECOND = "m2_5090"          # M2 latency on a second device (RTX 5090), same arguments as the primary run
M2_TO_MAIN ={"vlocal": "vlocal", "vexact": "vexact", "mlp_only": "mirror_mlp_m1", "svd": "svd_attn",
              "random": "random_attn", "dense": "original", "original": "original"}


def build_latency(P, Tm: Table, archs: dict, m1: dict | None = None, sub: str | None = None, suffix: str = ""):
    """M2 latency of one run folder (runs/<sub>/, default the primary run P.m2): T_latency<suffix> and
    T_latency_checks<suffix>; a second device's run (runs/m2_5090) uses suffix "_5090"."""
    sub = sub or P.m2
    L = Table(f"T_latency{suffix}", f"M2: measured latency against analytic cost (runs/{sub}/m2_latency.csv)",
              ["model", "variant", "k", "batch", "mode"],
              [("variant", "variant", ""), ("method", "label", ""), ("k", "k", "d"), ("batch", "batch", "d"),
               ("mode", "mode_used", ""),
               ("median ms", "latency_ms_median", ".3f"), ("img/s", "throughput_img_s", ",.0f"),
               ("speed-up vs dense", "speedup_vs_dense", ".3f"), ("vs original", "speedup_vs_original", ".3f"),
               ("analytic (M2)", "flop_speedup_vs_dense", ".3f"), ("analytic (collector)", "collector_flop_speedup", ".3f"),
               ("measured / analytic", "measured_over_analytic", ".3f"), ("GFLOPs", "gflops", ".2f"),
               ("params", "params_model", ",d"), ("checks", "checks", ""),
               ("timing flag", lambda r: "probable outlier" if r.get("timing_flag") else "", "")])
    L.notes = ["Median of the timed forwards (bf16, CUDA events; scripts/m2_latency.py). Speed-up vs dense: dense "
               "latency over the variant's, same batch and mode; dense is the folded original in the same efficient "
               "implementation, original the hub model as released. Analytic (M2): dense multiply-adds over the "
               "variant's, counted by m2_latency.py; analytic (collector): the same ratio from tables/cost_model.md "
               "(whole model); measured / analytic: measured speed-up over the M2 analytic one. Checks: float64, float32 "
               "and bfloat16 agreement of the efficient model with the dense model it was built from "
               "(T_latency_checks).",
               "Which analytic speed-up: measured / analytic uses analytic (M2), the timing script's count of the "
               "implementation it timed, so numerator and denominator describe the same model. Analytic (collector) is the "
               "structured count of tables/cost_model.md that T_main_zeroshot reports. The two agree for every variant "
               "except V-local (M2 also counts one d x d rotation of the CLS feature, below 0.01%). For V-local the timed "
               "implementation stores Wo dense in the block's basis, because M1 projected Wo with A_(l+1) rather than the "
               "block's input operator (keeping it structured would need a second rotation per block, vlocal_r2, not run): "
               "Wo then costs d x H dh multiply-adds per token instead of its free-parameter count. Column "
               "wo_dense_extra_gflops (CSV) recomputes that extra cost from runs/m1/m1_ranks.csv; with the CLS rotation it "
               "accounts for the whole gap between the two GFLOPs counts (column analytic_gap_unexplained_gflops). The "
               "same choice is why V-local's params here (the stored dense Wo) exceed its count in T_main_zeroshot. "
               "Basis rotations are counted the same way in both (one d x d per converted block).",
               "Timing flag: a converted variant whose measured speed-up exceeds its analytic one (measured / analytic "
               "above 1) cannot be explained by its FLOPs; such rows are kept but marked as probable timing outliers "
               "(listed below with the same variant's other batch sizes)."]
    C = Table(f"T_latency_checks{suffix}", f"M2 checks: efficient model against the dense model it was built from "
              f"(runs/{sub}/m2_checks.csv)", ["variant", "k", "dtype"],
              [("variant", "variant", ""), ("k", "k", "d"), ("dtype", "dtype", ""), ("criterion", "criterion", ""),
               ("value", "value", ".2e"), ("pass", "pass", ""), ("noise floor", "noise_floor_unit", ".2e")])
    base = P.runs / sub
    lat, chk, meta = read_csv(base / "m2_latency.csv"), read_csv(base / "m2_checks.csv"), read_json(base / "m2_meta.json")
    mgit = ((meta or {}).get("git") or {}).get("git_hash")
    if meta:
        dev = (meta.get("env") or {}).get("gpu") or (meta.get("env") or {}).get("device")
        ci = str(meta.get("check_images") or "not recorded")
        L.notes.insert(0, f"Device: {dev or 'not recorded'} (runs/{sub}/m2_meta.json, {meta.get('date')}). Timed inputs are "
                          "random normal images in every run (scripts/m2_latency.py time_model); equivalence check images: "
                          f"{ci}.")
        if ci.startswith("random"):
            note = ("The equivalence checks of this run used random normal images: scripts/m2_latency.py falls back to them "
                    "when the validation image cache is not available on the machine. They compare the efficient model with "
                    "the dense model it was built from, so they test the implementation on these inputs, not on validation "
                    "images.")
            L.notes.insert(1, note)
            C.notes.append(note)
    status: dict = {}
    if chk is not None:
        fchk = P.rel(base / "m2_checks.csv")
        for _, r in chk.iterrows():
            key = str(r.get("criterion", "")).split(" ")[0]
            row = {"variant": r["variant"], "k": py(r["k"]), "dtype": r["dtype"], "criterion": r.get("criterion"),
                   "value": py(r.get(key)) if key in r else None, "pass": py(r.get("pass")),
                   "noise_floor_unit": py(r.get("noise_floor_unit")), "_src": fchk, "_git": r.get("git_hash") or mgit,
                   "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
            if key in r:
                row["_loc"]["value"] = csv_loc(fchk, {"variant": r["variant"], "k": int(r["k"]), "dtype": r["dtype"]}, key)
            C.add(row)
            status.setdefault((r["variant"], int(r["k"])), []).append((r["dtype"], bool(py(r.get("pass")))))
    if lat is not None:
        flat = P.rel(base / "m2_latency.csv")
        for _, r in lat.iterrows():
            model, var, k = str(r.get("model")), str(r["variant"]), int(r["k"])
            where = {"variant": var, "k": k, "batch": int(r["batch"]), "mode": r["mode"]}
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": var, "label": r.get("label"),
                   "k": k, "batch": int(r["batch"]), "mode": r["mode"], "mode_used": r.get("mode_used"),
                   "source": flat, "git_hash": r.get("git_hash") or mgit, "seed": py(r.get("seed")),
                   "_src": flat, "_git": r.get("git_hash") or mgit, "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
            for c in ("latency_ms_median", "latency_ms_p10", "latency_ms_p90", "throughput_img_s", "speedup_vs_dense",
                      "speedup_vs_original", "flop_speedup_vs_dense", "gflops", "gflops_exec", "params_model",
                      "params_stored", "kept_frac_converted", "rotation_params", "n_rotations", "macs_per_image",
                      "compile_ok", "error"):
                if c in r:
                    row[c] = py(r[c])
                    if c not in ("compile_ok", "error"):
                        row["_loc"][c] = csv_loc(flat, where, c)
            if row.get("speedup_vs_dense") is not None and row.get("flop_speedup_vs_dense"):
                row["measured_over_analytic"] = row["speedup_vs_dense"] / row["flop_speedup_vs_dense"]
                row["_derived"]["measured_over_analytic"] = "speedup_vs_dense / flop_speedup_vs_dense (m2_latency.csv)"
            row["label_recorded"] = row.get("label")
            row["label"] = M2_LABEL.get(var, row.get("label"))
            mk = M2_TO_MAIN.get(var)
            arch = archs.get(model)
            tm = Tm.find(model=model, k=0 if var in ("dense", "original") else k, variant=mk) if mk else None
            if arch and tm is not None and tm.get("gflops_model"):
                row["collector_flop_speedup"] = analytic_cost(arch, [])["gflops_model_dense"] / tm["gflops_model"]
                row["_derived"]["collector_flop_speedup"] = (f"dense GFLOPs / GFLOPs of T_main_zeroshot {mk} at k = {tm['k']} "
                                                             "(analytic_cost)")
                row["gflops_collector"] = tm["gflops_model"]
                row["_derived"]["gflops_collector"] = f"gflops_model of T_main_zeroshot {mk} at k = {tm['k']}"
                rk = (m1 or {}).get("ranks")
                d_, T_ = arch["d"], arch["T"]
                if var.startswith("vlocal") and rk is not None and k and int(rk["block"].max()) >= k \
                        and str(r.get("wo_mode", "dense")) != "structured" and row.get("gflops") is not None:
                    sub = rk[rk["block"] <= k]
                    row["wo_dense_extra_gflops"] = 2.0 * T_ * float((d_ * d_ - sub["attn_free_o"]).sum()) / 1e9
                    row["_derived"]["wo_dense_extra_gflops"] = ("2 T sum over converted blocks of (d^2 - attn_free_o) / 1e9 "
                                                                "(runs/m1/m1_ranks.csv; Wo stored dense)")
                if row.get("gflops") is not None:
                    # M2's rotation work beyond the per-token rotations the collector also counts: the CLS feature
                    cls_rot = float(py(r.get("rotation_macs")) or 0) - int(py(r.get("n_rotations")) or 0) * T_ * d_ * d_
                    row["cls_rotation_gflops"] = 2.0 * cls_rot / 1e9
                    row["_derived"]["cls_rotation_gflops"] = "2 (rotation_macs - n_rotations T d^2) / 1e9 (m2_latency.csv)"
                    row["analytic_gap_unexplained_gflops"] = (row["gflops"] - tm["gflops_model"]
                                                              - row.get("wo_dense_extra_gflops", 0.0) - row["cls_rotation_gflops"])
                    row["_derived"]["analytic_gap_unexplained_gflops"] = (
                        "gflops (M2) minus gflops_collector minus wo_dense_extra_gflops minus cls_rotation_gflops")
            if "wo_mode" in r:
                row["wo_mode"] = r.get("wo_mode")
            st = status.get((var, k))
            if st:
                row["checks_pass"] = all(p for _, p in st)
                row["checks"] = " / ".join(f"{d}: {'pass' if p else 'FAIL'}" for d, p in st)
            L.add(row)
    # probable timing outliers: a converted variant measured faster than its FLOPs allow
    for r in L.rows:
        if r["k"] and r.get("measured_over_analytic") is not None and r["measured_over_analytic"] > 1.0:
            others = sorted((x for x in L.rows if x["variant"] == r["variant"] and x["k"] == r["k"]
                             and x["mode"] == r["mode"] and x is not r and x.get("speedup_vs_dense") is not None),
                            key=lambda x: x["batch"])
            r["timing_flag"] = True
            r["timing_note"] = (f"{r['label']} k = {r['k']}, batch {r['batch']}, {r['mode']}: measured "
                                f"{r['speedup_vs_dense']:.3f}x against an analytic {r['flop_speedup_vs_dense']:.3f}x; the "
                                "same variant measures " + ", ".join(f"{x['speedup_vs_dense']:.3f}x at batch {x['batch']}"
                                                                     for x in others)
                                + f" ({r['mode']}). Probable timing outlier: kept, not used for any claim.")
            L.notes.append("Flagged: " + r["timing_note"])
    return L, C, meta


PROFILE_CATS = ("gemm", "attention", "triton", "elementwise", "norm", "memory", "other")   # scripts/m2_profile.py
PROFILE_CAT_LABEL = {"gemm": "GEMM", "attention": "attention", "triton": "fused Triton", "elementwise": "elementwise",
                     "norm": "normalisation", "memory": "memory movement", "other": "other"}
PROFILE_PARTS = ("mlp", "attention", "rotation")


def build_latency_profile(P):
    """M2 profile (scripts/m2_profile.py, runs/m2_profile/): one row per model, variant, k, mode and batch with
    the kernel-time breakdown by category (ms per forward and share of kernel time), the latency gap to dense
    split into category changes plus the idle change, and the sub-module split (MLP, attention, rotations) summed
    over the timed converted blocks; plus the fused Triton time by operation (m2_triton.csv)."""
    cat_cols = []
    for c in PROFILE_CATS:
        cat_cols.append((PROFILE_CAT_LABEL[c], lambda r, c=c: (f"{fmt(r.get('ms_' + c), '.3f')} "
                                                              f"({fmt(100 * r['share_' + c] if r.get('share_' + c) is not None else None, '.0f')}%)"
                                                              if r.get("ms_" + c) is not None else ""), ""))
    T = Table("T_latency_profile", "M2 profile: where the time goes (runs/m2_profile/m2_profile.csv, m2_submodules.csv)",
              ["model", "variant", "k", "mode", "batch"],
              [("variant", "variant", ""), ("k", "k", "d"), ("mode", "mode_used", ""), ("median ms", "latency_ms_median", ".3f"),
               ("kernel ms", "kernel_ms", ".3f"), ("idle ms", "idle_ms", ".3f"), ("GEMM TFLOP/s", "gemm_tflops", ".0f")]
              + cat_cols)
    T.notes = ["Kernel time per forward by category, ms and share of the kernel time (scripts/m2_profile.py sorts every "
               "GPU activity into a category by its name; time_source says whether these are CUDA kernel times or, in "
               "the CPU smoke test, operator self times). Idle = median latency minus kernel time. Gap = median latency "
               "minus the dense model's (same model, mode and batch; dense is the folded original in the same "
               "implementation). The category changes plus the idle change add up to the gap (gap_check in the CSV). "
               "Sub-modules: each converted block's MLP, attention and stream rotations timed alone against the same "
               "block of dense, summed over the timed blocks (sub_blocks, 0-based block indices as in "
               "m2_submodules.csv); remainder = gap minus the summed sub-module changes (norms, residual adds, launch "
               "overheads)."]
    X = Table("T_latency_profile_triton", "M2 profile: fused Triton kernel time by operation (runs/m2_profile/m2_triton.csv)",
              ["variant", "k", "mode", "triton_tag"],
              [("variant", "variant", ""), ("k", "k", "d"), ("mode", "mode", ""), ("operations", "triton_tag", ""),
               ("ms per forward", "ms_per_forward", ".3f"), ("kernels", "kernel_count", ".0f")])
    base = P.runs / "m2_profile"
    prof, sub, tri = read_csv(base / "m2_profile.csv"), read_csv(base / "m2_submodules.csv"), read_csv(base / "m2_triton.csv")
    meta = read_json(base / "m2_profile_meta.json")
    if meta:
        env = meta.get("env") or {}
        T.notes.insert(0, f"Device: {env.get('gpu') or env.get('device') or meta.get('device_type') or 'not recorded'} "
                          f"(runs/m2_profile/m2_profile_meta.json, {meta.get('date')}); batch {meta.get('batch')}. GEMM "
                          "TFLOP/s: the GEMM FLOPs per forward (2 x multiply-adds, gemm_gflop) over the GEMM kernel time "
                          "(gemm_tflops of m2_profile.csv).")
    if prof is None:
        return T, X, meta
    f = P.rel(base / "m2_profile.csv")
    keys = ["model", "variant", "k", "mode", "batch"]
    for key, g in prof.groupby(keys, sort=False):
        model, v, k, mode, b = str(key[0]), str(key[1]), int(key[2]), str(key[3]), int(key[4])
        w = {"model": model, "variant": v, "k": k, "mode": mode, "batch": b}
        r0 = g.iloc[0]
        row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": v, "label": r0.get("label"),
               "k": k, "mode": mode, "mode_used": r0.get("mode_used"), "batch": b, "time_source": r0.get("time_source"),
               "device": r0.get("device"), "source": f, "git_hash": r0.get("git_hash"), "seed": py(r0.get("seed")),
               "_src": f, "_git": r0.get("git_hash"), "_seed": py(r0.get("seed")), "_loc": {}, "_derived": {}}
        for c in ("latency_ms_median", "kernel_ms", "idle_ms", "launches", "speedup_vs_dense", "flop_speedup_vs_dense",
                  "gemm_tflops", "macs_per_image", "n_rotations", "ref_m2_clean_ms"):
            if c in g and py(r0.get(c)) is not None:
                row[c] = py(r0[c])
                row["_loc"][c] = csv_loc(f, {**w, "category": str(r0["category"])}, c)
        for _, cr in g.iterrows():
            c = str(cr["category"])
            for col, src in (("ms", "ms_per_forward"), ("share", "share"), ("n", "kernel_count"), ("d", "delta_ms_vs_dense")):
                if src in cr and py(cr[src]) is not None:
                    row[f"{col}_{c}"] = py(cr[src])
                    row["_loc"][f"{col}_{c}"] = csv_loc(f, {**w, "category": c}, src)
        T.add(row)
    for r in T.rows:
        d = T.find(model=r["model"], variant="dense", k=0, mode=r["mode"], batch=r["batch"])
        if d is None or r.get("latency_ms_median") is None or d.get("latency_ms_median") is None:
            continue
        r["gap_ms"] = r["latency_ms_median"] - d["latency_ms_median"]
        r["_derived"]["gap_ms"] = "latency_ms_median minus that of dense (same model, mode and batch; m2_profile.csv)"
        if r.get("idle_ms") is not None and d.get("idle_ms") is not None:
            r["d_idle"] = r["idle_ms"] - d["idle_ms"]
            r["_derived"]["d_idle"] = "idle_ms minus that of dense (m2_profile.csv)"
            ds = [r.get(f"d_{c}") for c in PROFILE_CATS]
            if all(x is not None for x in ds):
                r["gap_check"] = r["gap_ms"] - (sum(ds) + r["d_idle"])
                r["_derived"]["gap_check"] = "gap minus (sum of category changes + idle change); 0 up to rounding"
    if sub is not None:
        fs = P.rel(base / "m2_submodules.csv")
        sub = sub.copy()
        if "model" not in sub:
            sub["model"] = T.rows[0]["model"] if T.rows else ""
        for key, g in sub.groupby(["model", "variant", "k", "mode"], sort=False):
            hits = [r for r in T.rows if r["model"] == str(key[0]) and r["variant"] == str(key[1]) and r["k"] == int(key[2])
                    and r["mode"] == str(key[3])]
            if len(hits) != 1:
                continue
            r = hits[0]
            r["sub_blocks"] = ",".join(str(int(x)) for x in sorted(g["block"].unique()))
            r["sub_source"] = fs
            tot = 0.0
            for p_ in PROFILE_PARTS:
                gp = g[g["part"] == p_]
                if not len(gp):
                    continue
                r[f"{p_}_ms"] = float(gp["ms_median"].sum())
                r[f"{p_}_dense_ms"] = float(gp["dense_ms_median"].sum())
                r[f"{p_}_delta_ms"] = float(gp["delta_ms"].sum())
                r[f"{p_}_n"] = int(len(gp))
                tot += r[f"{p_}_delta_ms"]
                for c, s in (("ms", "ms_median"), ("dense_ms", "dense_ms_median"), ("delta_ms", "delta_ms")):
                    r["_derived"][f"{p_}_{c}"] = f"sum over blocks {r['sub_blocks']} of {s} for part {p_} ({fs})"
            r["sub_sum_delta_ms"] = tot
            r["_derived"]["sub_sum_delta_ms"] = f"sum of the MLP, attention and rotation changes ({fs})"
            if r.get("gap_ms") is not None:
                r["sub_remainder_ms"] = r["gap_ms"] - tot
                r["_derived"]["sub_remainder_ms"] = "gap_ms minus sub_sum_delta_ms"
    if tri is not None:
        ft = P.rel(base / "m2_triton.csv")
        for _, r in tri.iterrows():
            w = {"variant": r["variant"], "k": int(r["k"]), "mode": r["mode"], "triton_tag": r["triton_tag"]}
            row = {**w, "ms_per_forward": py(r.get("ms_per_forward")), "kernel_count": py(r.get("kernel_count")),
                   "_src": ft, "_git": r.get("git_hash"), "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
            for c in ("ms_per_forward", "kernel_count"):
                row["_loc"][c] = csv_loc(ft, w, c)
            X.add(row)
    return T, X, meta


def profile_views(T: Table) -> list[Table]:
    """Three Markdown views of T_latency_profile's rows: the gap split into categories, and the sub-module split."""
    gap = Table("x", "Gap to dense split into categories (ms per forward; the columns add up to the gap)", [],
                [("variant", "variant", ""), ("k", "k", "d"), ("mode", "mode_used", ""), ("gap", "gap_ms", "+.3f")]
                + [(PROFILE_CAT_LABEL[c], f"d_{c}", "+.3f") for c in PROFILE_CATS] + [("idle", "d_idle", "+.3f")])
    sub = Table("x", "Sub-module split: change against the same dense blocks, summed over the timed converted blocks (ms)", [],
                [("variant", "variant", ""), ("k", "k", "d"), ("mode", "mode_used", ""), ("blocks (0-based)", "sub_blocks", ""),
                 ("MLP", "mlp_delta_ms", "+.3f"), ("attention", "attention_delta_ms", "+.3f"),
                 ("rotations", "rotation_delta_ms", "+.3f"), ("sum", "sub_sum_delta_ms", "+.3f"),
                 ("end-to-end gap", "gap_ms", "+.3f"), ("remainder", "sub_remainder_ms", "+.3f")])
    return [gap, sub]


def build_f4(P):
    T = Table("T_f4", "F4: single-block surgery, symmetry against low rank (runs/f4/f4_<model>.csv)", ["model", "block"],
              [("model", "model_label", ""), ("block", "block", "d"), ("s (act.)", "s_act", ".3f"),
               ("E_svd (act.)", "E_act", ".3f"), ("E_svd - s", "rule_x", ".3f"), ("drop Mirror", "drop_mirror", ".2f"),
               ("drop SVD", "drop_svd", ".2f"), ("SVD - Mirror drop", "rule_y", ".2f"), ("rule right", "rule_correct", ""),
               ("s (weight)", "s_weight", ".3f"), ("E_svd (weight)", "E_weight", ".3f")])
    T.notes = ["Activation-weighted breaking share s and SVD discarded energy E_svd at matched parameters, and the k-NN "
               "drops of single-block Mirror surgery and SVD (scripts/f4_rule.py). The P2 rule is right when "
               "E_svd - s and SVD drop - Mirror drop have the same sign (a zero counts as a miss)."]
    S = Table("T_f4_p2", "P2 (pre-registered): fraction of cases the rule gets right (runs/f4/f4_p2.json)", ["scope"],
              [("scope", "scope", ""), ("cases", "n_cases", "d"), ("hits", "n_hits", "d"), ("fraction", "fraction", ".3f"),
               ("note", "note", "")])
    base = P.runs / "f4"
    files = sorted(p for p in base.glob("f4_*.csv")) if base.is_dir() else []
    for f in files:
        df = read_csv(f)
        if df is None:
            continue
        for _, r in df.iterrows():
            model = str(r.get("model"))
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "block": py(r.get("block")),
                   "depth": py(r.get("depth")), "fit_seed": py(r.get("fit_seed")), "prereg_commit": r.get("prereg_commit"),
                   "source": P.rel(f), "git_hash": r.get("git_hash"), "_src": P.rel(f), "_git": r.get("git_hash"),
                   "_seed": py(r.get("seed")), "_fit_seed": py(r.get("fit_seed")), "_loc": {}, "_derived": {}}
            for c in ("s_act", "E_act", "rule_x", "drop_mirror", "drop_svd", "rule_y", "rule_correct", "s_weight",
                      "E_weight", "rule_x_weight", "top1_original", "top1_mirror", "top1_svd", "mlp_params_mirror",
                      "mlp_params_svd", "act_source"):
                if c in r:
                    row[c] = py(r[c])
                    if c not in ("rule_correct", "act_source"):
                        row["_loc"][c] = csv_loc(P.rel(f), {"model": model, "block": int(r["block"])}, c)
            T.add(row)
    pj = read_json(base / "f4_p2.json")
    if pj:
        fj = P.rel(base / "f4_p2.json")
        wb = pj.get("written_by") or {}
        g = (wb.get("git") or {}).get("git_hash") or ";".join(pj.get("git_hashes") or [])
        scopes = [("pre-registered models (primary)", ["primary_prereg_models"]),
                  ("all models run", ["primary_all_models_run"])]
        scopes += [(f"{MODEL_LABEL.get(m, m)}", ["per_model", m]) for m in (pj.get("per_model") or {})]
        scopes += [(f"secondary: {k}", ["secondary", k]) for k in (pj.get("secondary") or {})]
        for name, path in scopes:
            d = pj
            for k in path:
                d = (d or {}).get(k)
            if not isinstance(d, dict):
                continue
            row = {"scope": name, "n_cases": py(d.get("n_cases")), "n_hits": py(d.get("n_hits")),
                   "fraction": py(d.get("fraction")), "_src": fj, "_git": g, "_seed": wb.get("seed"), "_loc": {},
                   "_derived": {}}
            for c in ("n_cases", "n_hits", "fraction"):
                row["_loc"][c] = json_loc(fj, path + [c])
            if path == ["primary_prereg_models"]:
                row["note"] = f"verdict {pj.get('verdict')} (threshold {fmt(pj.get('threshold'), 'g')})"
            S.add(row)
    return T, S, files, pj


# ================================================================================================
# E3 (PREREG_E3.md): matched healed baselines at full conversion, gate G5
# ================================================================================================
# Comparisons of k-NN differences allow this many points for the float32 rounding of the recorded top-1 values (their
# exact values are multiples of 100 / n_query, 0.002 points for the 50,000 validation queries).
E3_TOL = 1e-4
# key, arm (PREREG_E3.md Section 3), display label, kind (writer: heal_vexact.py k<K>/ folders or heal.py <method>/),
# run folder under runs/, Tier-2 folder under runs/ (None: the healed V-exact row of runs/t2, jobs/stage22_korea.txt)
E3_ARMS = (
    ("v0", "1", LABEL_VEXACT + ", healed, seed 0 (RTX 5090, V0)", "vexact", "m3_vexact_5090", None),
    ("v0_a100", "1 (A100)", LABEL_VEXACT + ", healed, seed 0 (A100 copy, not used in the checks)", "vexact",
     "m3_vexact", None),
    ("v1", "2", LABEL_VEXACT + ", healed, seed 1 (V1)", "vexact", "e3/vexact_s1", "e3/t2_vexact_s1"),
    ("svd_raw", "3a", LABEL_SVD_ATTN + ", healed, raw targets", "svd", "e3/svd_raw/svd", "e3/t2_svd_raw"),
    ("svd_sym", "3b", LABEL_SVD_ATTN + ", healed, V-exact final targets", "svd", "e3/svd_sym/svd", "e3/t2_svd_sym"),
    ("random", "4", "Random structure (V-exact construction), MLP + attention, healed (R)", "vexact", "e3/random",
     "e3/t2_random"),
)
E3_LR_PROBE = "e3/lr_probe_vexact_s0"       # Section 8: arm 1's grid rerun next to 2e-4 (lr_extra.json only)
E3_LR_ARMS = ("v0", "v1", "svd_raw", "svd_sym", "random")
E3_USES = {"E3.a": ("v0", "v1", "random"), "E3.b": ("v0", "v1", "svd_raw", "svd_sym")}


def prereg_signoff_commit(P: Paths, file: str, marker: str = "ACTIVE: signed off") -> tuple[str | None, str]:
    """The commit that made a pre-registration file ACTIVE (its record: the first commit whose text holds `marker`),
    else the commit that added the file."""
    try:
        out = subprocess.run(["git", "log", "--format=%H", f"-S{marker}", "--", file], cwd=P.root,
                             capture_output=True, text=True, timeout=20)
        lines = [x for x in out.stdout.split() if x]
        if out.returncode == 0 and lines:
            return lines[-1], f"the commit that signed {file} off (first commit with \"{marker}\")"
    except Exception:  # noqa: BLE001
        pass
    c = git_added_commit(P, file)
    return c, (f"the commit that added {file} (no signed-off commit found)" if c else "not recorded")


def parse_prereg_e3(P: Paths) -> dict:
    """Bars and validity limits of PREREG_E3.md (read from the text, never typed) and its record commit."""
    p = P.root / "PREREG_E3.md"
    th = {"file": "PREREG_E3.md", "exists": p.is_file(), "active": False, "commit": None, "commit_source": "no file"}
    if not p.is_file():
        return th
    th["commit"], th["commit_source"] = prereg_signoff_commit(P, "PREREG_E3.md")
    raw = p.read_text(encoding="utf-8")
    th["active"] = raw.lstrip().startswith("# ACTIVE")
    t = " ".join(raw.split())
    pats = {"bar_a": r"\|\s*E3\.a\s*\|\s*V0 - R\s*\|\s*>=\s*([\d.]+)",
            "bar_b": r"\|\s*E3\.b\s*\|\s*V0 - S\s*\|\s*>=\s*([\d.]+)",
            "steps": r"completes (\d+) of \d+ steps",
            "tol": r"exactness check \(tolerance ([\d.]+e-?\d+|[\d.]+)\)",
            "flip_pct": r"fp32 flip check \(([\d.]+)%\)",
            "block_pct": r"every block is matched within ([\d.]+)%",
            "extra_lr": r"selection trial at ([\d.]+e-?\d+|[\d.]+)",
            "k": r"full conversion \(k = (\d+)\)"}
    for k, pat in pats.items():
        m = re.search(pat, t)
        th[k] = float(m[1]) if m else None
    if th.get("k") is not None:
        th["k"] = int(th["k"])
    return th


def e3_check(diff, bar, s, tol: float = E3_TOL) -> tuple[str, bool | None, bool | None]:
    """PREREG_E3.md Sections 7 and 12: a check passes when its difference meets the bar AND is larger than the seed
    difference s (not the stricter difference >= bar + s). Returns (verdict, meets the bar, larger than s). A missed
    bar fails whatever s is; a met bar waits for s."""
    if diff is None or bar is None:
        return PENDING, None, None
    meets = diff >= bar - tol
    above = None if s is None else bool(diff - s > tol)
    if not meets:
        return FAIL, False, above
    if s is None:
        return PENDING, True, None
    return (PASS if above else FAIL), True, above


def _combine(vs) -> str:
    """All must pass: any failure fails the whole, otherwise pending until every part passes."""
    return PASS if vs and all(v == PASS for v in vs) else (FAIL if FAIL in vs else PENDING)


def lr_txt(x) -> str:
    """A learning rate as the pre-registration writes it (2e-4, 1e-5)."""
    if not isnum(x):
        return "the extra rate"
    m, e = f"{float(x):.6e}".split("e")
    return f"{float(m):g}e{int(e)}"


def _path_tail(path, rel) -> bool:
    a, b = str(path).replace("\\", "/").rstrip("/"), str(rel).replace("\\", "/").rstrip("/")
    return a == b or a.endswith("/" + b)


def read_lr_extra(lx: dict, src: str, prefix: list) -> dict:
    """Section 8 of PREREG_E3.md from an lr_extra.json (mirror.heal.probe_extra_lrs): an extra rate would have won if
    its held-out loss after the short budget is below every grid rate's (ties go to the grid). Recomputed here and
    compared with the run's own flag; a diverged or non-finite loss never wins, a diverged grid rate counts as
    infinite."""
    out = {"_loc": {}, "_derived": {}, "lr_extra_source": src}
    grid, extra = lx.get("grid"), lx.get("extra")
    out["extra_would_win_recorded"] = py(lx.get("extra_would_win"))
    out["extra_lrs"] = "; ".join(f"{float(x):g}" for x in (lx.get("extra_lrs") or [e.get("lr") for e in extra or []]))
    if not isinstance(grid, list) or not isinstance(extra, list) or not extra:
        out["extra_would_win"] = out["extra_would_win_recorded"]
        out["_derived"]["extra_would_win"] = f"recorded by the run ({src}; no per-rate losses to recompute)"
        return out

    def loss(x):
        v = x.get("heldout_loss")
        return float(v) if isnum(v) and not x.get("diverged") else math.inf

    best = min((loss(g) for g in grid), default=math.inf)
    out["grid_lrs"] = "; ".join(f"{float(g['lr']):g}" for g in grid)
    out["grid_best_heldout_loss"] = best if math.isfinite(best) else None
    out["_derived"]["grid_best_heldout_loss"] = f"lowest held-out loss of the grid trials in {src}"
    out["grid_trials"] = "; ".join(f"{float(g['lr']):g}: {fmt(g.get('heldout_loss'), '.5f')}" for g in grid)
    e0 = extra[0]
    out["extra_lr"] = py(e0.get("lr"))
    out["extra_heldout_loss"] = py(e0.get("heldout_loss"))
    out["_loc"]["extra_heldout_loss"] = json_loc(src, prefix + ["extra", 0, "heldout_loss"])
    out["extra_diverged"] = py(e0.get("diverged"))
    wins = [e for e in extra if loss(e) < best]
    out["extra_would_win"] = bool(wins)
    out["_derived"]["extra_would_win"] = (f"recomputed from {src}: an extra rate's held-out loss below every grid "
                                          "rate's (ties go to the grid)")
    if out["extra_would_win_recorded"] is not None:
        out["extra_win_matches_recorded"] = bool(out["extra_would_win_recorded"]) == out["extra_would_win"]
    ref = lx.get("reference") or {}
    if ref:
        out["probe_reference_run"] = ref.get("run")
        out["probe_same_best_lr"] = py(ref.get("same_best_lr"))
        out["probe_max_rel_diff_heldout_loss"] = py(ref.get("max_rel_diff_heldout_loss"))
    return out


def read_heal_run(P: Paths, d: Path) -> dict | None:
    """One healing output folder: scripts/heal_vexact.py (<out>/k<k>/) or scripts/heal.py (<out>/<method>/). Each part
    from its own file (knn.json, exactness.json, flipcheck.json, final/metrics.json, lr_selection/lr_selection.json,
    lr_extra.json), else from summary.json. None when the folder is missing or empty."""
    d = Path(d)
    if not d.is_dir() or not any(d.iterdir()):
        return None
    summ = read_json(d / "summary.json")
    sf = P.rel(d / "summary.json")

    def part(name, key):
        j = read_json(d / name)
        if j is not None:
            return j, P.rel(d / name), []
        if summ and isinstance(summ.get(key), dict):
            return summ[key], sf, [key]
        return None, None, []

    knn, kf, kp = part("knn.json", "knn")
    ex, ef, ep = part("exactness.json", "exactness")
    fl, ff, fp = part("flipcheck.json", "flipcheck")
    met, mf, mp = part("final/metrics.json", "final")
    sel, lf, lp = part("lr_selection/lr_selection.json", "lr_selection")
    lx, xf, xp = part("lr_extra.json", "lr_extra")
    cfg = (summ or {}).get("config") or (met or {}).get("config") or {}
    r = {"dir": P.rel(d), "_loc": {}, "_derived": {}}
    r["tiny"] = bool(cfg.get("tiny_mode")) or (knn or {}).get("protocol") == "tiny synthetic"
    r["model"] = "tiny" if r["tiny"] else (cfg.get("model") or {}).get("name")
    r["method_recorded"] = (summ or {}).get("method")
    blocks = (summ or {}).get("blocks") or cfg.get("blocks")
    r["k"] = py((summ or {}).get("k")) or (len(blocks) if blocks else None)
    r["depth"] = py((summ or {}).get("depth"))
    knn = knn or {}
    for part_ in ("original", "zero_shot", "healed"):
        if isinstance(knn.get(part_), dict):
            for c, s in (("top1", "top1"), ("flip", "flip_consistency"), ("n_query", "n_query")):
                if knn[part_].get(s) is not None:
                    r[f"{part_}_{c}"] = py(knn[part_][s])
                    r["_loc"][f"{part_}_{c}"] = json_loc(kf, kp + [part_, s])
    r["knn_amp"] = knn.get("amp")
    if ex:
        for c, s in (("exact_max_rel", "max_rel_inside"), ("exact_tol", "tol"), ("exact_pass", "pass")):
            if s in ex:
                r[c] = py(ex[s])
                r["_loc"][c] = json_loc(ef, ep + [s])
    if fl:
        for c, s in (("flip_fp32", "flip_consistency"), ("flip_fp32_pass", "pass"), ("flip_fp32_n_query", "n_query")):
            if s in fl:
                r[c] = py(fl[s])
                r["_loc"][c] = json_loc(ff, fp + [s])
    if sel:
        r["chosen_lr"] = py(sel.get("best_lr"))
        r["_loc"]["chosen_lr"] = json_loc(lf, lp + ["best_lr"])
        r["lr_trials"] = "; ".join(f"{float(t['lr']):g}: {fmt(t.get('heldout_loss'), '.5f')}" for t in sel.get("trials", []))
    if lx:
        x = read_lr_extra(lx, xf, xp)
        r["_loc"].update(x.pop("_loc"))
        r["_derived"].update(x.pop("_derived"))
        r.update(x)
    if met:
        for c, path in (("heldout_loss_before", ["heldout_initial", "total"]), ("heldout_loss_after", ["heldout_final", "total"])):
            v = met
            for p_ in path:
                v = (v or {}).get(p_)
            if v is not None:
                r[c] = py(v)
                r["_loc"][c] = json_loc(mf, mp + path)
        for c in ("steps_done", "total_steps", "n_trainable", "seconds_train"):
            if met.get(c) is not None:
                r[c] = py(met[c])
                r["_loc"][c] = json_loc(mf, mp + [c])
        r["diverged"] = py(met.get("diverged"))
        for name, v in (met.get("constraint_checks") or {}).items():
            if not (isinstance(v, dict) and "numerical_ranks" in v):
                continue
            same = v["numerical_ranks"] == v.get("target_ranks")
            if "LowRankAttention" in name:
                r["attn_ranks_equal"] = same
            elif "LowRankMLP" in name:
                r["mlp_ranks_equal_recorded"] = same
    mi = (summ or {}).get("method_info") or {}
    for c in ("mlp_params_kept", "attn_params_kept", "converted_params_kept", "converted_params_dense",
              "converted_kept_frac", "model_params", "model_params_original", "model_params_frac", "d_plus", "d_minus"):
        if mi.get(c) is not None:
            r[c] = py(mi[c])
            r["_loc"][c] = json_loc(sf, ["method_info", c])
    if r.get("converted_params_kept") is None and r.get("mlp_params_kept") is not None and r.get("attn_params_kept") is not None:
        r["converted_params_kept"] = int(r["mlp_params_kept"]) + int(r["attn_params_kept"])
        r["_derived"]["converted_params_kept"] = f"method_info mlp_params_kept + attn_params_kept ({sf})"
    if r.get("converted_params_dense") is None and mi.get("mlp_params_dense") is not None and mi.get("attn_params_dense") is not None:
        r["converted_params_dense"] = int(mi["mlp_params_dense"]) + int(mi["attn_params_dense"])
        r["_derived"]["converted_params_dense"] = f"method_info mlp_params_dense + attn_params_dense ({sf})"
    if r.get("converted_kept_frac") is None and r.get("converted_params_kept") and r.get("converted_params_dense"):
        r["converted_kept_frac"] = r["converted_params_kept"] / r["converted_params_dense"]
        r["_derived"]["converted_kept_frac"] = "converted_params_kept / converted_params_dense"
    vm = mi.get("vexact_match") or {}
    gaps = [float(vm[c]) for c in ("max_block_rel_gap_mlp", "max_block_rel_gap_attn") if isnum(vm.get(c))]
    if gaps:
        r["block_gap_max"] = max(gaps)
        r["_derived"]["block_gap_max"] = f"max of method_info.vexact_match max_block_rel_gap_mlp and _attn ({sf})"
    tg = (summ or {}).get("targets") or (summ or {}).get("teacher") or {}
    r["final_target"], r["block_targets"] = tg.get("final_target"), tg.get("block_targets")
    rs = (summ or {}).get("random_structure") or {}
    if rs:
        r["random_structure_seed"] = py(rs.get("seed"))
        r["random_agreement_with_fitted"] = py(rs.get("agreement_with_fitted_A"))
    r["seed"] = py(next((x.get("seed") for x in (summ, knn, met, sel) if x and x.get("seed") is not None), None))
    r["git_hash"] = next((x.get("git_hash") for x in (knn, summ, met, sel, ex) if x and x.get("git_hash")), None)
    r["git_dirty"] = py((summ or met or sel or {}).get("git_dirty"))
    r["_seed"], r["_git"] = r["seed"], r["git_hash"]
    r["operators"] = (cfg.get("structure") or {}).get("operators")
    r["fit_seed"] = r["_fit_seed"] = None if r["tiny"] else fit_seed_from_path(r["operators"])
    r["gpu"] = ((summ or {}).get("env") or {}).get("gpu") or ((met or {}).get("env") or {}).get("gpu")
    full = r.get("k") is not None and r.get("depth") is not None and r["k"] == r["depth"]
    if r.get("healed_top1") is not None:
        r["status"] = "complete" if (fl or not (full and r["method_recorded"] != "svd")) else \
            "complete except the fp32 flip check"
    elif met:
        r["status"] = "healed; final k-NN not yet available"
    elif sel:
        r["status"] = "learning-rate selection only"
    else:
        r["status"] = "started (no result files yet)"
    if r["tiny"]:
        r["status"] += " (TINY smoke test, not a result)"
    r["source"] = "; ".join(x for x in dict.fromkeys((kf, ef, ff, mf, lf, xf, sf if summ else None)) if x)
    r["_src"] = r["source"] or P.rel(d)
    return r


def e3_linear(P: Paths, csv_path: Path, fam: str, meth: str, k, model=None, run_dir: str | None = None) -> dict:
    """The Tier-2 ImageNet linear probe of one healed student (scripts/tier2_linear.py: t2_linear.csv and
    probe/<model>/<variant>_variant.json, which names the healing run it rebuilt and, for SVD, the float64 rank check
    of the rebuilt factors)."""
    df = read_csv(csv_path)
    if df is None or "family" not in df:
        return {}
    sel = df[(df["family"].astype(str) == fam) & (df["method"].astype(str) == meth)]
    if model is not None and "model" in sel:
        sel = sel[sel["model"].astype(str) == str(model)]
    if k is not None and "k" in sel:
        sel = sel[sel["k"].astype(int) == int(k)]
    f = P.rel(csv_path)
    if len(sel) != 1:
        return {"linear_note": f"{len(sel)} matching rows in `{f}`; none used"} if len(sel) > 1 else {}
    r = sel.iloc[0]
    where = {"model": str(r["model"]), "variant": str(r["variant"])}
    out = {"linear_top1": py(r["top1"]), "linear_variant": where["variant"], "linear_source": f,
           "linear_git": r.get("git_hash"), "linear_model_params": py(r.get("model_params")),
           "linear_wd_at_grid_edge": py(r.get("wd_at_grid_edge")), "_loc": {"linear_top1": csv_loc(f, where, "top1")}}
    for c, s in (("linear_knn_check", "knn_check_top1"), ("linear_chosen_wd", "chosen_wd")):
        if s in r and py(r[s]) is not None:
            out[c] = py(r[s])
            out["_loc"][c] = csv_loc(f, where, s)
    vj = read_json(Path(csv_path).parent / "probe" / where["model"] / f"{where['variant']}_variant.json") or {}
    info, srcs = vj.get("info") or {}, vj.get("sources") or {}
    path = info.get("vexact_run") or srcs.get("vexact_run") or srcs.get("heal_run")
    out["linear_run_path"] = path
    out["linear_same_run"] = _path_tail(path, run_dir) if (path and run_dir) else None
    for name, v in (info.get("constraint_checks") or {}).items():
        if isinstance(v, dict) and "numerical_ranks" in v:
            same = v["numerical_ranks"] == v.get("target_ranks")
            if "LowRankMLP" in name and "float64" in str(v.get("note", "")):
                out["rebuild_mlp_ranks_equal"] = same
            elif "LowRankAttention" in name:
                out["rebuild_attn_ranks_equal"] = same
    vc = info.get("vexact_checks")
    if isinstance(vc, dict):
        out["linear_vexact_checks_pass"] = py(vc.get("pass"))
    return out


def e3_k(P: Paths, th3: dict) -> tuple[int | None, str]:
    """Number of converted blocks of the E3 arms: the arm config, else the k<K> folders written, else PREREG_E3.md."""
    k = load_cfg(P, "configs/e3_vexact_s1.yaml").get("k")
    if k:
        return int(k), "configs/e3_vexact_s1.yaml"
    for sub in ("e3/vexact_s1", "e3/random", E3_LR_PROBE, "m3_vexact_5090"):
        base = P.runs / sub
        ks = sorted(int(m[1]) for d in (base.glob("k*") if base.is_dir() else []) if (m := re.fullmatch(r"k(\d+)", d.name)))
        if ks:
            return ks[-1], f"runs/{sub}/"
    if th3.get("k"):
        return int(th3["k"]), "PREREG_E3.md"
    return None, "not found"


def e3_validity(r: dict, th3: dict, v0: dict | None) -> tuple[str, str]:
    """Validity conditions of PREREG_E3.md Section 7 for one arm: (PASS, FAIL or pending; the conditions, one by one)."""
    conds = []
    steps = th3.get("steps")
    if r.get("steps_done") is None:
        conds.append((PENDING, "steps: healing metrics not available"))
    else:
        ok = (r["steps_done"] == r.get("total_steps") and not r.get("diverged")
              and (steps is None or int(r["steps_done"]) == int(steps)))
        conds.append((PASS if ok else FAIL, f"steps {fmt(r['steps_done'], 'd')} of {fmt(r.get('total_steps'), 'd')} (bar "
                      f"{fmt(steps, '.0f')} of {fmt(steps, '.0f')}), diverged {fmt(bool(r.get('diverged')))}"))
    if r["kind"] == "vexact":
        tol = th3.get("tol")
        if r.get("exact_max_rel") is None:
            conds.append((PENDING, "exactness: exactness.json not available"))
        else:
            ok = bool(r.get("exact_pass")) and (tol is None or r["exact_max_rel"] <= tol)
            conds.append((PASS if ok else FAIL, f"exactness {r['exact_max_rel']:.2e} (tolerance {fmt(tol, 'g')})"))
        fp = th3.get("flip_pct")
        if r.get("flip_fp32") is None:
            conds.append((PENDING, "fp32 flip check: flipcheck.json not available"))
        else:
            ok = bool(r.get("flip_fp32_pass")) and (fp is None or r["flip_fp32"] >= fp - 1e-9)
            conds.append((PASS if ok else FAIL, f"fp32 flip check {r['flip_fp32']:.2f}% (bar {fmt(fp, 'g')}%)"))
    if r["kind"] == "svd":
        a = r.get("attn_ranks_equal", r.get("rebuild_attn_ranks_equal"))
        m_ = r.get("rebuild_mlp_ranks_equal")
        if a is None:
            conds.append((PENDING, "attention ranks: no rank check recorded"))
        else:
            conds.append((PASS if a else FAIL, f"attention numerical ranks {'equal' if a else 'differ from'} the targets "
                          "(float64 check)"))
        if m_ is None:
            conds.append((PENDING, "MLP ranks: not verifiable yet; the healing run's own check forms U V in float32 "
                          "(mirror/heal.py LowRankMLP.check), so it reads full rank whatever the factors are; the Tier-2 "
                          "rebuild records the float64 check (probe/<model>/<variant>_variant.json), not available"))
        else:
            conds.append((PASS if m_ else FAIL, f"MLP numerical ranks {'equal' if m_ else 'differ from'} the targets "
                          "(float64 check of the Tier-2 rebuild)"))
        bp = th3.get("block_pct")
        if r.get("block_gap_max") is None:
            conds.append((PENDING, "budget: per-block match not recorded"))
        else:
            ok = bp is None or r["block_gap_max"] <= bp / 100 + 1e-12
            conds.append((PASS if ok else FAIL, f"largest per-block shortfall against V-exact {100 * r['block_gap_max']:.2f}% "
                          f"(bar {fmt(bp, 'g')}%)"))
    if r["key"] == "random":
        if not v0 or v0.get("mlp_params_kept") is None or r.get("mlp_params_kept") is None:
            conds.append((PENDING, "budget: needs the parameter counts of the random arm and of arm 1"))
        else:
            ok = (r["mlp_params_kept"], r.get("attn_params_kept")) == (v0["mlp_params_kept"], v0.get("attn_params_kept"))
            conds.append((PASS if ok else FAIL, f"MLP and attention counts {'equal' if ok else 'differ from'} arm 1's "
                          f"({fmt(r['mlp_params_kept'], ',d')} and {fmt(r.get('attn_params_kept'), ',d')}; the run records "
                          "totals, not per-block counts)"))
    verdicts = [c[0] for c in conds]
    v = FAIL if FAIL in verdicts else (PENDING if PENDING in verdicts else PASS)
    return v, "; ".join(f"{t} [{c}]" for c, t in conds)


def build_e3(P: Paths, T2: Table, th3: dict):
    """E3 (PREREG_E3.md): one row per arm (T_e3) and the pre-registered checks E3.a, E3.b, the seed condition, gate G5,
    the reporting variants with V1 and with the mean, the secondary learning-rate check and the validity conditions
    (T_e3_checks). Missing arms are "not run" and every check that needs them "pending"."""
    K, k_src = e3_k(P, th3)
    xl = lr_txt(th3.get("extra_lr"))
    T = Table("T_e3", f"E3: matched healed baselines at full conversion (k = {fmt(K, 'd')}), one row per arm", ["arm"],
              [("arm", "arm", ""), ("student", "label", ""), ("status", "status", ""),
               ("zero-shot", "zero_shot_top1", ".2f"), ("healed", "healed_top1", ".2f"), ("original", "original_top1", ".2f"),
               ("linear probe", "linear_top1", ".2f"), ("flip: k-NN", "healed_flip", ".2f"),
               ("flip: fp32 check", lambda r: (f"{fmt(r.get('flip_fp32'), '.2f')} ({'pass' if r.get('flip_fp32_pass') else 'FAIL'})"
                                               if r.get("flip_fp32") is not None else ""), ""),
               ("matrix params kept", "converted_params_kept", ",d"), ("vs arm 1", "kept_vs_vexact", ".4f"),
               ("model params kept", "model_params_frac", ".3f"), ("chosen LR", "chosen_lr", "g"),
               (f"{xl} would have won", lambda r: fmt(r.get("extra_would_win")) if r.get("extra_would_win") is not None else "", ""),
               ("steps", lambda r: f"{fmt(r.get('steps_done'), 'd')} / {fmt(r.get('total_steps'), 'd')}"
                if r.get("steps_done") is not None else "", ""),
               ("valid", "validity", ""), ("seed", "seed", "d"), ("GPU", "gpu", ""), ("git", lambda r: short(r.get("git_hash")), "")])
    C = Table("T_e3_checks", "E3 checks and gate G5 (PREREG_E3.md Sections 7, 8 and 12)", ["id"],
              [("id", "id", ""), ("check", "statement", ""), ("value", "value", ".3f"), ("threshold", "threshold", ""),
               ("meets bar", "meets_bar", ""), ("larger than s", "exceeds_s", ""), ("verdict", "verdict", ""), ("note", "note", "")])
    info = {"k": K, "k_source": k_src, "prereg": th3, "rows": {}, "planned": None}
    jf = P.jobs / "e3_k12.txt"
    plan = P.rel(jf) if jf.is_file() else None
    info["planned"] = plan
    rows: dict = {}
    for key, arm, label, kind, sub, t2sub in E3_ARMS:
        base = P.runs / sub
        d = base / f"k{K}" if (kind == "vexact" and K is not None) else base
        run = read_heal_run(P, d)
        row = {"arm": arm, "key": key, "label": label, "kind": kind, "_loc": {}, "_derived": {}}
        if run is None:
            row.update({"status": NOT_RUN, "dir": P.rel(d), "_src": P.rel(d),
                        "note": f"no files under `{P.rel(d)}/`" + (f"; planned in `{plan}`" if plan and key != "v0_a100" else "")})
        else:
            loc, der = run.pop("_loc"), run.pop("_derived")
            row.update(run)
            row["_loc"].update(loc)
            row["_derived"].update(der)
        lin = {}
        if t2sub is None:
            if T2 is not None and key in ("v0", "v0_a100"):
                lin = e3_linear(P, P.runs / "t2" / "t2_linear.csv", "healed_vexact", "vexact", K, row.get("model"), P.rel(d))
        else:
            fam, meth = ("healed", "svd") if kind == "svd" else ("healed_vexact", "vexact")
            lin = e3_linear(P, P.runs / t2sub / "t2_linear.csv", fam, meth, K, None, P.rel(d))
        if lin:
            lloc = lin.pop("_loc", {})
            if lin.get("linear_same_run") is False:
                row["linear_note"] = (f"the probe in `{lin['linear_source']}` rebuilt `{lin['linear_run_path']}`, not this "
                                      "run; not used here")
                row["linear_other_run"] = lin.get("linear_run_path")
            else:
                row.update(lin)
                row["_loc"].update(lloc)
                if lin.get("linear_same_run") is None and lin.get("linear_top1") is not None:
                    row["linear_note"] = ("the probe's variant JSON (with the healing run it rebuilt) is not on this machine; "
                                          + ("jobs/stage22_korea.txt probes runs/m3_vexact_5090" if key == "v0" else
                                             "run folder not checked"))
                    if key == "v0_a100":            # the runs/t2 probe of arm 1 is the RTX 5090 run (jobs/stage22_korea.txt)
                        for c in list(lin):
                            row.pop(c, None)
                        row["_loc"].pop("linear_top1", None)
                        row.pop("linear_note", None)
        rows[key] = row
    if K is not None:                         # Section 8 for arm 1: the probe run reruns the grid next to the extra rate
        pf = P.runs / E3_LR_PROBE / f"k{K}" / "lr_extra.json"
        lx = read_json(pf)
        if lx is not None:
            x = read_lr_extra(lx, P.rel(pf), [])
            rows["v0"]["_loc"].update(x.pop("_loc"))
            rows["v0"]["_derived"].update(x.pop("_derived"))
            rows["v0"].update(x)
    v0 = rows["v0"] if rows["v0"]["status"] != NOT_RUN else None
    orig_params = (v0 or {}).get("model_params_original")
    o2 = T2.find(model=(v0 or {}).get("model"), variant="original") if (T2 is not None and v0) else None
    for key, r in rows.items():
        if r.get("model_params") is None and r.get("linear_model_params") is not None:
            r["model_params"] = r["linear_model_params"]
            r["_derived"]["model_params"] = f"model_params of the Tier-2 rebuild ({r.get('linear_source')})"
        if r.get("model_params_frac") is None and r.get("model_params") and orig_params:
            r["model_params_frac"] = r["model_params"] / orig_params
            r["_derived"]["model_params_frac"] = "model_params / arm 1's method_info.model_params_original"
        if v0 and key != "v0" and r.get("converted_params_kept") and v0.get("converted_params_kept"):
            r["kept_vs_vexact"] = r["converted_params_kept"] / v0["converted_params_kept"]
            r["_derived"]["kept_vs_vexact"] = "converted_params_kept / that of arm 1"
        if o2 is not None and o2.get("top1") is not None:
            r["linear_original_top1"] = o2["top1"]
            if o2["_loc"].get("top1"):
                r["_loc"]["linear_original_top1"] = o2["_loc"]["top1"]
        if r["status"] != NOT_RUN and key != "v0_a100":
            r["validity"], r["validity_detail"] = e3_validity(r, th3, v0)
        elif key != "v0_a100":
            r["validity"], r["validity_detail"] = PENDING, "not run"
        r.setdefault("_fit_seed", None)
        T.add(r)
    info["rows"] = rows
    T.notes = [
        f"Arms of PREREG_E3.md Section 3 (k = {fmt(K, 'd')}, from {k_src}): every block converted, MLP and attention, one "
        "recipe. Arm 1 (V0) is the seed 0 V-exact run on the RTX 5090, the primary value (Section 6); its A100 copy is "
        "reported next to it and used in no check. Top-1 and flip consistency in percent from each run's knn.json "
        "(protocol v2 k-NN, k = 20, T = 0.07, bf16 features); fp32 check: flip consistency of the fp32 flip check "
        "(flipcheck.json, the V-exact construction only).",
        "Linear probe: Tier-2 ImageNet linear probe of the same healed student (scripts/tier2_linear.py: runs/t2 for arm 1, "
        "runs/e3/t2_<arm> for the others); a probe whose variant JSON names another healing run is not used (column "
        f"linear_note). The original model's probe: {fmt(o2.get('top1') if o2 else None, '.2f') or 'not run'} (T_tier2).",
        "Matrix params kept: MLP and attention matrix parameters of the converted blocks (summary.json method_info; "
        "biases and embeddings not counted, as in PREREG_E3.md Section 4); vs arm 1: over arm 1's count. Model params "
        "kept: whole model over the original (for the SVD arms from the Tier-2 rebuild's count).",
        f"Chosen LR: the grid's choice (lr_selection.json). {xl} would have won: Section 8; the extra trial's held-out "
        "loss after the short budget is below every grid rate's (ties go to the grid), recomputed from lr_extra.json and "
        "compared with the run's own flag (extra_would_win_recorded); it never changes the chosen rate. For arm 1 the "
        f"trials are those of the probe run `runs/{E3_LR_PROBE}/k<K>/` (grid rerun next to {xl}).",
        "Valid: the validity conditions of Section 7 (detail in T_e3_checks and in column validity_detail). Targets: "
        "columns final_target and block_targets (3a raw; 3b the flip-averaged final targets with the teacher's own block "
        "outputs; V-exact and the random control symmetrised).",
    ]
    # ---------------------------------------------------------------- checks
    bars = {"E3.a": th3.get("bar_a"), "E3.b": th3.get("bar_b")}
    srcs = ";".join(sorted({r["_src"] for r in rows.values() if r["status"] != NOT_RUN and r["key"] != "v0_a100"})) or ""
    gits = ";".join(sorted({str(r["_git"]) for r in rows.values() if r.get("_git") and r["key"] != "v0_a100"})) or None

    def val(key):
        r = rows[key]
        return r.get("healed_top1") if r["status"] != NOT_RUN else None

    V0, V1, R = val("v0"), val("v1"), val("random")
    S3a, S3b = val("svd_raw"), val("svd_sym")
    S = max(S3a, S3b) if (S3a is not None and S3b is not None) else None
    s_arm = ("3a" if S3a >= S3b else "3b") if S is not None else None
    s = abs(V0 - V1) if (V0 is not None and V1 is not None) else None
    tiny = [rows[k]["arm"] for k in ("v0", "v1", "random", "svd_raw", "svd_sym") if rows[k].get("tiny")]
    invalid = {k: rows[k] for k in E3_LR_ARMS if rows[k].get("validity") == FAIL}
    unver = {k: rows[k] for k in E3_LR_ARMS if rows[k]["status"] != NOT_RUN and rows[k].get("validity") == PENDING}

    def base_note(cid, diff, verdict):
        missing = [rows[k]["arm"] for k in E3_USES[cid] if val(k) is None]
        bits = []
        if missing:
            bits.append("waiting for arm(s) " + ", ".join(missing))
        if cid == "E3.b" and S is None and V0 is not None and bars["E3.b"] is not None:
            for sv, nm in ((S3a, "3a"), (S3b, "3b")):
                if sv is not None and V0 - sv < bars["E3.b"] - E3_TOL:
                    bits.append(f"V0 minus {nm} alone is {V0 - sv:.3f}, below the bar; S >= {nm}, so E3.b cannot pass")
        return bits

    def check_row(cid, stmt, diff, use_v="V0", info_only=False):
        bar = bars[cid.split(" ")[0]]
        verdict, meets, above = e3_check(diff, bar, s)
        note = base_note(cid.split(" ")[0], diff, verdict) if not info_only else []
        bad = [rows[k]["arm"] for k in E3_USES[cid.split(" ")[0]] if k in invalid]
        if tiny:
            note.append(f"TINY smoke-test inputs (arms {', '.join(tiny)}): not a result")
        if bad:
            note.append(f"arm(s) {', '.join(bad)} fail a validity condition (Section 7: invalid, fix and rerun that arm)")
        if (tiny or bad) and verdict != PENDING:
            note.append(f"computed value {verdict}, not counted")
            verdict = PENDING
        if info_only and verdict in (PASS, FAIL):
            verdict = f"would {verdict}"
        return {"id": cid, "statement": stmt, "value": diff, "threshold": f">= {fmt(bar, '.1f')} and > s" if bar is not None
                else "bar not found in PREREG_E3.md", "bar": bar, "s": s, "meets_bar": meets, "exceeds_s": above,
                "verdict": verdict, "note": "; ".join(note), "_src": srcs, "_git": gits, "_seed": None,
                "_derived": {"value": f"from T_e3 healed_top1 ({stmt})"}}

    a_row = C.add({**check_row("E3.a", "V0 minus R: healed k-NN of V-exact seed 0 minus the random-structure control",
                               None if V0 is None or R is None else V0 - R),
                   "max_baseline": None if V0 is None or bars["E3.a"] is None else V0 - bars["E3.a"]})
    b_row = C.add({**check_row("E3.b", "V0 minus S, S = max(3a, 3b)" + (f" (here {s_arm})" if s_arm else ""),
                               None if V0 is None or S is None else V0 - S),
                   "max_baseline": None if V0 is None or bars["E3.b"] is None else V0 - bars["E3.b"], "S_arm": s_arm, "S": S})
    for rr, nm in ((a_row, "E3.a"), (b_row, "E3.b")):
        rr["_derived"]["max_baseline"] = f"V0 minus the {nm} bar: the largest baseline value the bar allows"
    C.add({"id": "E3.seed", "statement": "s = |V0 - V1|, the seed difference of V-exact (arms 1 and 2)", "value": s,
           "threshold": "a check passes only if its difference is also larger than s", "verdict": "reported" if s is not None
           else PENDING, "note": "" if s is not None else "waiting for arm 2" if V1 is None else "waiting for arm 1",
           "_src": srcs, "_git": gits, "_derived": {"value": "|V0 - V1| from T_e3 healed_top1"}})
    g5 = _combine([a_row["verdict"], b_row["verdict"]])
    alt = {}
    for nm, vv in (("V1", V1), ("mean", None if V0 is None or V1 is None else (V0 + V1) / 2)):
        ra = check_row(f"E3.a ({nm})", f"as E3.a with {'V1' if nm == 'V1' else 'the mean of V0 and V1'} in place of V0 "
                       "(reported, does not change the verdict)", None if vv is None or R is None else vv - R, info_only=True)
        rb = check_row(f"E3.b ({nm})", f"as E3.b with {'V1' if nm == 'V1' else 'the mean of V0 and V1'} in place of V0 "
                       "(reported, does not change the verdict)", None if vv is None or S is None else vv - S, info_only=True)
        alt[nm] = (ra, rb)
    g5_note = []
    if V1 is not None:
        g5_v1 = _combine([alt["V1"][0]["verdict"].replace("would ", ""), alt["V1"][1]["verdict"].replace("would ", "")])
        if g5 in (PASS, FAIL) and g5_v1 in (PASS, FAIL) and g5_v1 != g5:
            g5_note.append(f"the verdict differs between seeds: with V1 in place of V0, G5 would be {g5_v1}")
    if unver and g5 in (PASS, FAIL):
        g5_note.append("validity not yet verified for arm(s) " + ", ".join(r["arm"] for r in unver.values())
                       + " (T_e3_checks E3.valid.*)")
    if g5 == PASS:
        g5_note.append("PREREG_E3.md: at full conversion healing does not close the gap to a matched structure-free "
                       "baseline; the k = 12 headline can be credited to the mirror structure")
    elif g5 == FAIL:
        g5_note.append("PREREG_E3.md: report as is, stop, and discuss the framing with Dawar; no new arms, rates, seeds or "
                       "budgets inside E3")
    C.add({"id": "G5", "statement": "gate G5: E3.a and E3.b both pass (with the seed condition)", "verdict": g5,
           "threshold": "both pass", "note": "; ".join(g5_note), "_src": "T_e3_checks", "_git": gits})
    for ra, rb in alt.values():
        C.add(ra)
        C.add(rb)
    # Section 8: the secondary learning-rate check
    have = {k: rows[k] for k in E3_LR_ARMS if rows[k].get("extra_would_win") is not None}
    winners = [rows[k]["arm"] for k in have if rows[k]["extra_would_win"]]
    mism = [rows[k]["arm"] for k in have if rows[k].get("extra_win_matches_recorded") is False]
    lr_note = [f"arm {rows[k]['arm']}: {xl} {fmt(rows[k].get('extra_heldout_loss'), '.5f')} against best grid "
               f"{fmt(rows[k].get('grid_best_heldout_loss'), '.5f')}" for k in have]
    pr = rows["v0"]
    if pr.get("probe_same_best_lr") is not None:
        lr_note.append(f"arm 1's probe reruns the grid: same best rate as the recorded grid {fmt(pr['probe_same_best_lr'])}, "
                       f"largest relative difference of the held-out losses {fmt(pr.get('probe_max_rel_diff_heldout_loss'), '.2e')}")
    if mism:
        lr_note.append(f"the recomputed flag differs from the run's own for arm(s) {', '.join(mism)}")
    if winners:
        verdict = "yes: " + ", ".join(winners)
        lr_note.append("the main comparison is flagged as possibly under-tuned (Section 8); the full budget is not rerun "
                       f"at {xl} inside E3")
    elif len(have) == len(E3_LR_ARMS):
        verdict = "no"
    else:
        verdict = PENDING
        lr_note.append(f"{len(have)} of {len(E3_LR_ARMS)} arms have their {xl} trial")
    C.add({"id": "E3.lr", "statement": f"secondary learning-rate check: would {xl} have won the short selection for any "
           "arm (never changes a chosen rate)", "value": len(winners) if have else None, "threshold": "reported",
           "verdict": verdict, "note": "; ".join(lr_note), "_src": srcs, "_git": gits,
           "_derived": {"value": "number of arms whose extra trial would have won (T_e3 extra_would_win)"}})
    for key in E3_LR_ARMS:
        r = rows[key]
        C.add({"id": f"E3.valid.{r['arm']}", "statement": f"validity conditions of arm {r['arm']} ({r['label']})",
               "verdict": r.get("validity", PENDING), "threshold": "every condition passes", "note": r.get("validity_detail", ""),
               "_src": r["_src"], "_git": r.get("_git"), "_seed": r.get("_seed")})
    C.notes = [
        f"Bars read from PREREG_E3.md (record commit {short(th3.get('commit'))}, {th3.get('commit_source')}"
        f"{'' if th3.get('active') else '; the file is not marked ACTIVE'}): E3.a V0 - R >= {fmt(bars['E3.a'], '.1f')}, "
        f"E3.b V0 - S >= {fmt(bars['E3.b'], '.1f')}, with V0 and V1 the healed k-NN of V-exact seeds 0 and 1, R the random-"
        "structure control and S the better of the two SVD arms. Seed condition (Section 12, decision 1): a check counts "
        "as passed only if its difference is also larger than s = |V0 - V1| (not the stricter difference >= bar + s). G5: "
        "E3.a and E3.b both pass. A missed bar fails whatever s is; a met bar waits for s.",
        f"Comparisons allow {E3_TOL:g} points for the float32 rounding of the recorded top-1 values (exact values are "
        "multiples of 100 / n_query, 0.002 points for 50,000 queries). Max baseline: V0 minus the bar, the largest R (E3.a) "
        "or S (E3.b) the bar allows.",
        "Rows marked (V1) and (mean) repeat both differences with V1, and with the mean of V0 and V1, in place of V0; they "
        "are reported and never change the verdict. A check that uses an arm failing a validity condition stays pending "
        "(Section 7: that arm is invalid, fixed and rerun). Validity (Section 7): every run completes the pre-registered "
        f"steps ({fmt(th3.get('steps'), '.0f')}) without divergence; the V-exact construction passes the exactness check "
        f"(tolerance {fmt(th3.get('tol'), 'g')}) and the fp32 flip check ({fmt(th3.get('flip_pct'), 'g')}%); the SVD arms' "
        f"numerical ranks equal their targets; every block is matched within {fmt(th3.get('block_pct'), 'g')}%.",
        "MLP ranks of the SVD arms: the healing run's own check (mirror/heal.py LowRankMLP.check) forms U V in float32 and "
        "so reads full rank; the float64 check of the Tier-2 rebuild (scripts/tier2_linear.py constraint_checks) is used. "
        "The attention check of the healing run is already float64 (mirror/heal_attn.py).",
    ]
    info["values"] = {"V0": V0, "V1": V1, "R": R, "S3a": S3a, "S3b": S3b, "S": S, "S_arm": s_arm, "s": s, "G5": g5}
    return T, C, info


# ================================================================================================
# Strong baselines (scripts/strong_baselines.py), flip averaging (scripts/flip_avg_baseline.py), F7 (scripts/f7_visuals.py)
# ================================================================================================
SB_METHODS = ("asvd", "slicegpt", "prune_act")       # mirror/baselines.py METHODS
SB_LABEL = {"asvd": "Activation-aware SVD (SVD-LLM style), MLP only",
            "slicegpt": "SliceGPT-style PCA slicing, MLP only",
            "prune_act": "MLP width pruning, activation-weighted, MLP only"}


def build_strong_baselines(P: Paths, Tm: Table, archs: dict):
    """Zero-shot strong compression baselines at the Mirror V-local MLP-only budget (runs/sb/SB_s<fit seed>.csv, the
    Track A CSV format with ablation "SB"), per model, k and fit seed, next to the canonical X2 numbers of Mirror V-local
    MLP only and plain SVD at the same k and fit seed (T_main_zeroshot)."""
    T = Table("T_strong_baselines", "Strong compression baselines, zero-shot, at the Mirror V-local MLP-only budget "
              "(runs/sb/SB_s<fit seed>.csv)", ["model", "k", "fit_seed", "method"],
              [("model", "model_label", ""), ("k", "k", "d"), ("fit seed", "fit_seed", "d"), ("method", "label", ""),
               ("source", "experiment", ""), ("top-1", "top1", ".2f"), ("drop", "drop_points", ".2f"),
               ("flip consistency", "flip_consistency", ".2f"), ("MLP weights kept", "mlp_params_kept", ",d"),
               ("vs Mirror budget", "ratio_to_mirror", ".4f"), ("within 1%", "within_1pct", ""),
               ("not counted: stream basis", "basis_params_not_counted", ",d"),
               ("Mirror minus this", "mirror_minus_this", "+.2f"), ("this minus SVD", "this_minus_svd", "+.2f"),
               ("original (same run)", "original_top1", ".2f"), ("status", "status", ""),
               ("git", lambda r: short(r.get("git_hash")), "")])
    T.notes = [
        "Each strong baseline compresses the MLPs (fc1, fc2) of blocks 1 to k and nothing else, matched to the MLP weight "
        "count of Mirror V-local, MLP only, at the same k and fit run (biases excluded); statistics from the fit run's "
        "train fit images (scripts/strong_baselines.py, mirror/baselines.py). Top-1, drop and flip consistency in percent "
        "(protocol v2 k-NN, bf16 features). Drop: the run's own original minus the method.",
        f"Mirror V-local, MLP only, and SVD at matched parameters, MLP only: the canonical {CANONICAL_RUN} values at the same "
        "k and fit seed (T_main_zeroshot, per-seed values). Mirror's MLP weights are the budget the strong baselines were "
        "matched to (mirror_budget recorded by strong_baselines.py); SVD's are the analytic count of tables/cost_model.md.",
        "Not counted, stream basis: d x d per converted block, the change of stream basis between blocks that an "
        "implementation of SliceGPT-style slicing stores (basis_change_params recorded by the run), and Mirror V-local's "
        "basis rotations (rotation_params of the cost model); neither is in the MLP budget.",
    ]
    base = P.runs / "sb"
    files = sorted(base.glob("SB_s*.csv")) if base.is_dir() else []
    groups: dict = {}
    for f in files:
        df = read_csv(f)
        if df is None:
            continue
        for _, r in df.iterrows():
            groups.setdefault((str(r["model"]), int(r["k"]), int(r["fit_seed"])), []).append((f, r))
    order = {m: i for i, m in enumerate(SB_METHODS)}
    for (model, k, fs) in sorted(groups, key=lambda x: (MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 99, x[1], x[2])):
        items = sorted(groups[(model, k, fs)], key=lambda x: order.get(str(x[1]["variant"]), 99))
        budgets = {int(_info(r.get("info")).get("mirror_budget")) for _, r in items
                   if _info(r.get("info")).get("mirror_budget") is not None}
        budget = budgets.pop() if len(budgets) == 1 else None
        arch = archs.get(model)
        o = Tm.find(model=model, k=0, variant="original")
        o_seed = ((o or {}).get("_per_seed") or {}).get(fs)
        grp = []
        for method, lab in (("mirror", LABEL_MIRROR_MLP), ("svd", LABEL_SVD_MLP)):
            v, x = canonical_seed_value(Tm, model, k, method, fs)
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "k": k, "fit_seed": fs, "method": method,
                   "label": lab, "experiment": f"{CANONICAL_RUN} (canonical)", "_loc": {}, "_derived": {}, "_fit_seed": fs}
            if v is None:
                row.update({"status": NOT_RUN, "note": f"no {CANONICAL_RUN} row at k = {k}, fit seed {fs}", "_src": ""})
                grp.append(T.add(row))
                continue
            row.update({"status": "ok", "top1": v, "flip_consistency": x["flip"], "drop_points": x["drop"],
                        "source": x["src"], "git_hash": x["git"], "_src": x["src"], "_git": x["git"], "_seed": x["seed"]})
            for c, s in (("top1", "top1"), ("flip_consistency", "flip_consistency"), ("drop_points", "drop_points")):
                row["_loc"][c] = csv_loc(x["src"], x["where"], s)
            if o_seed:
                row["original_top1"] = o_seed["top1"]
                row["_loc"]["original_top1"] = csv_loc(o_seed["src"], o_seed["where"], "top1")
            cost = x.get("cost")
            if method == "mirror" and budget is not None:
                row["mlp_params_kept"] = budget
                row["_derived"]["mlp_params_kept"] = ("mirror_budget recorded in the info column of the strong-baseline "
                                                      "CSV (variants.mirror_param_counts of the same fit run)")
                if cost and arch:
                    row["mlp_params_kept_cost_model"] = int(round(cost["matrix_params_kept"] - k * 4 * arch["d"] ** 2))
                    row["_derived"]["mlp_params_kept_cost_model"] = "analytic_cost matrix_params_kept minus k 4 d^2"
            elif cost and arch:
                row["mlp_params_kept"] = int(round(cost["matrix_params_kept"] - k * 4 * arch["d"] ** 2))
                row["_derived"]["mlp_params_kept"] = ("analytic_cost matrix_params_kept minus the dense attention, k 4 d^2 "
                                                      "(tables/cost_model.md)")
            if cost and method == "mirror":
                row["basis_params_not_counted"] = cost.get("rotation_params")
                row["_derived"]["basis_params_not_counted"] = "rotation_params of analytic_cost (one d x d per block)"
            if method == "svd":
                row["basis_params_not_counted"] = 0
            if budget and row.get("mlp_params_kept") is not None:
                row["ratio_to_mirror"] = row["mlp_params_kept"] / budget
                row["_derived"]["ratio_to_mirror"] = "mlp_params_kept / mirror_budget"
                row["within_1pct"] = abs(row["ratio_to_mirror"] - 1) <= PARAM_MATCH_TOL
            grp.append(T.add(row))
        for f, r in items:
            m = str(r["variant"])
            inf = _info(r.get("info"))
            fr = P.rel(f)
            where = {"model": model, "ablation": str(r["ablation"]), "variant": m, "k": k, "fit_seed": fs}
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "k": k, "fit_seed": fs, "method": m,
                   "label": SB_LABEL.get(m, m), "label_recorded": inf.get("baseline"), "experiment": "SB",
                   "status": "ok", "source": fr, "git_hash": r.get("git_hash"), "git_dirty": py(r.get("git_dirty")),
                   "_src": fr, "_git": r.get("git_hash"), "_seed": py(r.get("seed")), "_fit_seed": fs, "_loc": {},
                   "_derived": {}}
            for c, s in (("top1", "top1"), ("flip_consistency", "flip_consistency"), ("drop_points", "drop_points"),
                         ("original_top1", "top1_original"), ("mlp_params_kept", "mlp_params_kept"),
                         ("mlp_params_dense", "mlp_params_dense"), ("ratio_to_mirror", "params_vs_mirror_first_k"),
                         ("n_query", "n_query"), ("n_fit_images", "n_fit_images")):
                if s in r and py(r[s]) is not None:
                    row[c] = py(r[s])
                    row["_loc"][c] = csv_loc(fr, where, s)
            row["within_1pct"] = (bool(inf["within_1pct"]) if inf.get("within_1pct") is not None else
                                  (abs(row["ratio_to_mirror"] - 1) <= PARAM_MATCH_TOL if row.get("ratio_to_mirror") else None))
            row["mirror_budget"], row["bias_params"] = inf.get("mirror_budget"), inf.get("bias_params")
            row["basis_params_not_counted"] = inf.get("basis_change_params", 0)
            row["allocation"] = json.dumps(inf.get("allocation")) if inf.get("allocation") is not None else None
            if m == "asvd":
                row["asvd_ridge_damp"] = inf.get("ridge_damp")
            grp.append(T.add(row))
        mir = next((x for x in grp if x["method"] == "mirror" and x.get("top1") is not None), None)
        svd = next((x for x in grp if x["method"] == "svd" and x.get("top1") is not None), None)
        for x in grp:
            if x.get("top1") is None:
                continue
            if mir is not None and x is not mir:
                x["mirror_minus_this"] = mir["top1"] - x["top1"]
                x["_derived"]["mirror_minus_this"] = f"top1 of {CANONICAL_RUN} Mirror V-local, MLP only (same k, fit seed) minus this"
            if svd is not None and x is not svd and x["method"] != "mirror":
                x["this_minus_svd"] = x["top1"] - svd["top1"]
                x["_derived"]["this_minus_svd"] = f"top1 of this minus {CANONICAL_RUN} SVD (same k, fit seed)"
        sbo = {x.get("original_top1") for x in grp if x["experiment"] == "SB" and x.get("original_top1") is not None}
        if o_seed and sbo and any(abs(v - o_seed["top1"]) > 1e-9 for v in sbo):
            T.notes.append(f"{MODEL_LABEL.get(model, model)}, k = {k}, fit seed {fs}: the strong-baseline run measured the "
                           f"original at {', '.join(f'{v:.3f}' for v in sorted(sbo))} against {o_seed['top1']:.3f} in "
                           f"{CANONICAL_RUN}; compare drops within a run.")
    return T, files


FA_MODES = {"plain": 0, "plain_fp32": 1, "flipavg": 2}


def build_flipavg(P: Paths):
    """Flip-averaging baseline (scripts/flip_avg_baseline.py, runs/t1_flipavg/flipavg.csv): exact invariance by averaging
    an image and its mirror at test time (two forward passes) against the healed V-exact model at full conversion (one
    pass, exactly equivariant). Relative cost: the run's analytic GFLOPs per image over the original's (one pass)."""
    T = Table("T_flipavg", "Flip averaging at test time (two passes) against healed V-exact (one pass) "
              "(runs/t1_flipavg/flipavg.csv)", ["model", "variant", "mode"],
              [("model", "model_label", ""), ("variant", "variant", ""), ("method", "label", ""), ("mode", "mode", ""),
               ("k-NN top-1", "knn_top1", ".2f"), ("flip consistency", "flip_consistency", ".2f"),
               ("identical features %", "feature_identical_pct", ".2f"), ("linear probe", "probe_top1", ".2f"),
               ("passes", "forwards_per_image", "d"), ("GFLOPs per image", "gflops", ".2f"),
               ("cost vs original", "rel_cost", ".3f"), ("params", "params", ",d"), ("params vs original", "params_frac", ".3f"),
               ("exactly invariant", "exactly_invariant", ""),
               ("minus flip-averaged original: k-NN", "minus_flipavg_knn", "+.2f"),
               ("minus flip-averaged original: linear", "minus_flipavg_probe", "+.2f"), ("note", "note", "")])
    T.notes = [
        "Modes (scripts/flip_avg_baseline.py): plain is the paper's feature; flipavg averages the features of an image and "
        "of its mirror (CLS after the final norm, before the L2 normalisation, for k-NN; the whole probe feature for the "
        "linear probe), for bank and queries alike, so it is exactly mirror-invariant at two forward passes per image; "
        "plain_fp32 is the plain k-NN with fp32 forward passes (healed V-exact at full conversion only). k-NN top-1 and "
        "flip consistency in percent (protocol v2); identical features: share of queries whose unit feature equals that of "
        "their mirror bit for bit; linear probe: the Tier-2 recipe.",
        "Cost vs original: GFLOPs per image (forward passes times the analytic cost of one pass, "
        "scripts/collect_results.analytic_cost) over the original model's one pass; where GFLOPs are not recorded, forward "
        "passes times parameters (column rel_cost_basis). Params vs original: whole-model parameters over the original's. "
        "Minus flip-averaged original: the row's top-1 minus that of the flip-averaged original (healed V-exact rows).",
    ]
    f = P.runs / "t1_flipavg" / "flipavg.csv"
    df = read_csv(f)
    if df is None:
        return T
    fr = P.rel(f)
    for _, r in df.iterrows():
        model, var, mode = str(r["model"]), str(r["variant"]), str(r["mode"])
        where = {"model": model, "variant": var, "mode": mode}
        row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "variant": var, "mode": mode,
               "label": method_label(r.get("family"), r.get("method"), r.get("label")), "label_recorded": r.get("label"),
               "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")), "mode_label": r.get("mode_label"),
               "source": fr, "git_hash": r.get("git_hash"), "seed": py(r.get("seed")), "_src": fr, "_git": r.get("git_hash"),
               "_seed": py(r.get("seed")), "_loc": {}, "_derived": {}}
        for c in ("knn_top1", "flip_consistency", "feature_identical_pct", "feature_flip_max_abs", "probe_top1", "probe_top5",
                  "forwards_per_image", "gflops", "gflops_per_forward", "params", "params_frac", "knn_check_top1",
                  "chosen_wd", "healing_flip_fp32", "vexact_span_exact_rel_max", "vexact_checks_pass", "vexact_even_head",
                  "knn_amp", "wd_at_grid_edge", "notes", "date"):
            if c in r and py(r[c]) is not None:
                row[c] = py(r[c])
                if c not in ("vexact_checks_pass", "vexact_even_head", "knn_amp", "wd_at_grid_edge", "notes", "date"):
                    row["_loc"][c] = csv_loc(fr, where, c)
        row["exactly_invariant"] = mode == "flipavg" or (row.get("family") == "healed_vexact" and bool(row.get("vexact_even_head")))
        if row.get("family") == "healed_vexact":
            vj = read_json(P.runs / "t1_flipavg" / "rows" / model / f"{var}_variant.json") or {}
            run = (vj.get("info") or {}).get("vexact_run") or (vj.get("sources") or {}).get("vexact_run")
            if run:
                rp = local_run_dir(P, run)
                row["healing_run"] = P.rel(rp) if rp is not None else str(run).replace(chr(92), "/")
                kj = read_json(rp / "knn.json") if rp is not None else None
                if kj and isinstance(kj.get("healed"), dict) and kj["healed"].get("top1") is not None:
                    row["healing_run_knn_top1"] = py(kj["healed"]["top1"])
                    row["_loc"]["healing_run_knn_top1"] = json_loc(P.rel(rp / "knn.json"), ["healed", "top1"])
        T.add(row)
    T.rows.sort(key=lambda x: (x["model"], {"original": 0, "healed_vexact": 1}.get(x.get("family"), 2), x.get("k") or 0,
                               x["variant"], FA_MODES.get(x["mode"], 9)))
    for model in sorted({x["model"] for x in T.rows}):
        rs = [x for x in T.rows if x["model"] == model]
        o = next((x for x in rs if x.get("family") == "original" and x["mode"] == "plain"), None)
        fa = next((x for x in rs if x.get("family") == "original" and x["mode"] == "flipavg"), None)
        for x in rs:
            if o is not None:
                if x.get("gflops") is not None and o.get("gflops"):
                    x["rel_cost"], x["rel_cost_basis"] = x["gflops"] / o["gflops"], "GFLOPs (analytic)"
                    x["_derived"]["rel_cost"] = "gflops / gflops of the original, plain (flipavg.csv)"
                elif x.get("params") and o.get("params") and x.get("forwards_per_image"):
                    x["rel_cost"], x["rel_cost_basis"] = x["forwards_per_image"] * x["params"] / o["params"], "passes x parameters"
                    x["_derived"]["rel_cost"] = "forwards_per_image x params / params of the original (flipavg.csv)"
            if fa is not None and x is not fa and x.get("family") == "healed_vexact":
                for c, s in (("minus_flipavg_knn", "knn_top1"), ("minus_flipavg_probe", "probe_top1")):
                    if x.get(s) is not None and fa.get(s) is not None:
                        x[c] = x[s] - fa[s]
                        x["_derived"][c] = f"{s} of {x['variant']} ({x['mode']}) minus that of the flip-averaged original"
                if x.get("gflops") and fa.get("gflops"):
                    x["cost_vs_flipavg"] = x["gflops"] / fa["gflops"]
                    x["_derived"]["cost_vs_flipavg"] = "gflops / gflops of the flip-averaged original"
                if x.get("params") and fa.get("params"):
                    x["params_vs_flipavg"] = x["params"] / fa["params"]
                    x["_derived"]["params_vs_flipavg"] = "params / params of the flip-averaged original"
            notes = []
            if x.get("healing_run"):
                notes.append(f"healed model from `{x['healing_run']}`"
                             + (f" (its own k-NN {x['healing_run_knn_top1']:.2f})" if x.get("healing_run_knn_top1") is not None else ""))
            if x["model"] == "tiny":
                notes.append("TINY smoke test, not a result")
            x["note"] = "; ".join(notes)
    return T


def build_latency_devices(L1: Table, L2: Table, meta1: dict | None, meta2: dict | None) -> Table:
    """M2 on two GPUs, the same arguments: measured speed-up against dense side by side (one row per variant, k, batch
    and mode present in both runs)."""
    g1 = ((meta1 or {}).get("env") or {}).get("gpu") or "first GPU"
    g2 = ((meta2 or {}).get("env") or {}).get("gpu") or "second GPU"
    D = Table("T_latency_devices", "M2 on two GPUs: measured speed-up against dense, same arguments",
              ["model", "variant", "k", "batch", "mode"],
              [("variant", "variant", ""), ("method", "label", ""), ("k", "k", "d"), ("batch", "batch", "d"),
               ("mode", "mode", ""), (f"measured: {g1}", "speedup_1", ".3f"), (f"measured: {g2}", "speedup_2", ".3f"),
               ("second minus first", "speedup_diff", "+.3f"), ("analytic (M2)", "flop_speedup", ".3f"),
               ("timing flag", lambda r: ", ".join(x for x in (r.get("flag_1"), r.get("flag_2")) if x), "")])
    D.notes = [f"Speed-up against the dense reference (dense latency over the variant's, same batch and mode) measured "
               f"on {g1} ({L1.title.split('(')[-1].rstrip(')')}) and on {g2} ({L2.title.split('(')[-1].rstrip(')')}), "
               "with the same script and arguments. Analytic (M2): the timing script's multiply-add ratio, the same on "
               "both devices. Timing flag: a probable timing outlier on that device (measured above analytic)."]
    for r2 in L2.rows:
        r1 = L1.find(model=r2["model"], variant=r2["variant"], k=r2["k"], batch=r2["batch"], mode=r2["mode"])
        if r1 is None or r1.get("speedup_vs_dense") is None or r2.get("speedup_vs_dense") is None:
            continue
        row = {"model": r2["model"], "variant": r2["variant"], "label": r2.get("label"), "k": r2["k"], "batch": r2["batch"],
               "mode": r2["mode"], "speedup_1": r1["speedup_vs_dense"], "speedup_2": r2["speedup_vs_dense"],
               "device_1": g1, "device_2": g2, "flop_speedup": r2.get("flop_speedup_vs_dense"),
               "flag_1": f"probable outlier ({g1})" if r1.get("timing_flag") else "",
               "flag_2": f"probable outlier ({g2})" if r2.get("timing_flag") else "",
               "_src": f"{r1['_src']};{r2['_src']}", "_git": f"{r1['_git']};{r2['_git']}", "_seed": r2.get("_seed"),
               "_loc": {}, "_derived": {}}
        if r1["_loc"].get("speedup_vs_dense"):
            row["_loc"]["speedup_1"] = r1["_loc"]["speedup_vs_dense"]
        if r2["_loc"].get("speedup_vs_dense"):
            row["_loc"]["speedup_2"] = r2["_loc"]["speedup_vs_dense"]
        if r2["_loc"].get("flop_speedup_vs_dense"):
            row["_loc"]["flop_speedup"] = r2["_loc"]["flop_speedup_vs_dense"]
        row["speedup_diff"] = row["speedup_2"] - row["speedup_1"]
        row["_derived"]["speedup_diff"] = "speed-up on the second GPU minus that on the first"
        D.add(row)
    return D


def build_f7(P: Paths):
    """F7 visuals (scripts/f7_visuals.py, runs/f7/): per block, the twin correlation of the mirror pairing against a
    random partner on held-out train images (f7_blocks.csv), and the layer-0 parity summary (f7_meta.json)."""
    B = Table("T_f7", "F7 (teaser): mirror twins per block on held-out train images (runs/f7/f7_blocks.csv)", ["model", "block"],
              [("model", "model_label", ""), ("block", "block", "d"), ("pairs", "n_pairs", "d"), ("self-twins", "n_self", "d"),
               ("twin r median (q10, q90)", lambda r: f"{fmt(r.get('twin_r_median'), '.3f')} ({fmt(r.get('twin_r_q10'), '.3f')}, "
                                                     f"{fmt(r.get('twin_r_q90'), '.3f')})" if r.get("twin_r_median") is not None else "", ""),
               ("random r median (q10, q90)", lambda r: f"{fmt(r.get('rand_r_median'), '.3f')} ({fmt(r.get('rand_r_q10'), '.3f')}, "
                                                       f"{fmt(r.get('rand_r_q90'), '.3f')})" if r.get("rand_r_median") is not None else "", ""),
               ("twin above random", "frac_twin_gt_rand", ".3f"), ("chosen pairs", "n_chosen", "d"),
               ("chosen: twin r mean", "chosen_twin_r_mean", ".3f"), ("images", "n_images", ",d"), ("fit seed", "fit_seed", "d")])
    B.notes = ["Twin r: correlation over every patch token of the visualisation images (ImageNet train images not used to "
               "fit the pairing, unless the run's images.source says fit) between a neuron's post-GELU activation and its "
               "twin's on the mirrored image at the mirrored position, over every twin pair of the block (median and 10th "
               "and 90th percentiles); random r: the same with a random partner. Twin above random: share of paired neurons "
               "whose twin correlates better than the random partner. Chosen pairs: the top pairs of the fit run's ranking "
               "drawn in the teaser (not hand-picked)."]
    L0 = Table("T_f7_layer0", "F7: layer-0 patch-embedding filters by parity of A_0 (runs/f7/f7_meta.json)", ["operator", "parity"],
               [("operator", "operator", ""), ("parity", "parity", ""), ("dimension", "dim", "d"),
                ("energy share", "energy_share", ".3f"), ("symmetric-energy fraction", "sym_fraction", ".3f")])
    L0.notes = ["Patch-embedding filters projected on the even (A_0 = +1) and odd (A_0 = -1) subspaces of the layer-0 operator; "
                "energy share of each subspace and the symmetric-energy fraction of its filters (1 for an exactly symmetric "
                "even part, 0 for an exactly antisymmetric odd part); null: a random involution with the same d+."]
    base = P.runs / "f7"
    info = {"dir": P.rel(base), "meta": read_json(base / "f7_meta.json"), "figures": {}}
    df = read_csv(base / "f7_blocks.csv")
    meta = info["meta"] or {}
    g = (meta.get("git") or {}).get("git_hash")
    if df is not None:
        fb = P.rel(base / "f7_blocks.csv")
        for _, r in df.iterrows():
            model, b = str(r["model"]), int(r["block"])
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "block": b, "_src": fb,
                   "_git": r.get("git_hash") or g, "_seed": py(r.get("seed")), "_fit_seed": py(r.get("fit_seed")),
                   "fit_seed": py(r.get("fit_seed")), "_loc": {}, "_derived": {}}
            for c, s in (("n_pairs", "n_pairs"), ("n_self", "n_self"), ("twin_r_median", "pairs_twin_corr_median"),
                         ("twin_r_q10", "pairs_twin_corr_q10"), ("twin_r_q90", "pairs_twin_corr_q90"),
                         ("rand_r_median", "pairs_rand_corr_median"), ("rand_r_q10", "pairs_rand_corr_q10"),
                         ("rand_r_q90", "pairs_rand_corr_q90"), ("frac_twin_gt_rand", "frac_paired_neurons_twin_gt_rand"),
                         ("n_chosen", "n_chosen"), ("chosen_twin_r_mean", "chosen_twin_corr_mean"),
                         ("chosen_rand_r_mean", "chosen_rand_corr_mean"), ("n_images", "n_images"), ("n_tokens", "n_tokens")):
                if s in r and py(r[s]) is not None:
                    row[c] = py(r[s])
                    row["_loc"][c] = csv_loc(fb, {"model": model, "block": b}, s)
            B.add(row)
    l0 = meta.get("layer0") or {}
    if l0:
        fm = P.rel(base / "f7_meta.json")
        for op, opl in (("fitted", "A_0 (fitted)"), ("null", "random involution, same d+")):
            for par in ("even", "odd"):
                row = {"operator": opl, "parity": par, "_src": fm, "_git": g, "_seed": meta.get("seed"), "_loc": {}, "_derived": {}}
                for q in ("dim", "energy_share", "sym_fraction"):
                    kk = f"{op}_{par}_{q}"
                    if l0.get(kk) is not None:
                        row[q] = py(l0[kk])
                        row["_loc"][q] = json_loc(fm, ["layer0", kk])
                L0.add(row)
    for stem in ("fig0_teaser", "figF7_twins_full", "figF7_layer0_filters"):
        for where in (P.results / "paper" / "figures_paper", base / "figures"):
            if (where / f"{stem}.pdf").is_file() or (where / f"{stem}.png").is_file():
                info["figures"][stem] = P.rel(where)
                break
    return B, L0, info


# ================================================================================================
# External scripts: F1 and F2 (imported and called, not edited)
# ================================================================================================
def npz_for(P: Paths, run: dict) -> Path | None:
    for p in (run["res"] / "e1_operators.npz", P.artifacts / run["res_rel"] / "e1_operators.npz"):
        if p.is_file():
            return p
    return None


def run_f1(P: Paths, out: Path, runs_dir: Path, arms: list[str]) -> dict:
    res = {"status": NOT_RUN, "out": out, "arms": arms}
    try:
        mod = import_script(P.root / "scripts" / "f1_tax.py", "f1_tax_wrapped")
        orig = mod.load_twin_scores

        def patched(run, blocks):  # look for the operators npz under artifacts/ when it is not in runs/
            path = Path(run["results"]) / "e1_operators.npz"
            if not path.exists():
                alt = P.artifacts / P.rel(run["results"]) / "e1_operators.npz"
                if alt.exists():
                    run = {**run, "results": alt.parent}
            return orig(run, blocks)

        mod.load_twin_scores = patched
        meta = mod.main(["--runs", str(runs_dir), "--out", str(out), "--arms", ",".join(arms)])
        res.update({"status": "ok", "meta": meta})
    except SystemExit as e:
        res.update({"status": NOT_RUN, "error": str(e)})
    except Exception as e:  # noqa: BLE001
        res.update({"status": "failed", "error": f"{e}\n{traceback.format_exc()}"})
    return res


def run_f2(P: Paths, out: Path, runs: list[dict]) -> dict:
    res = {"status": NOT_RUN, "out": out}
    items = [(r, npz_for(P, r)) for r in runs]
    items = [(r, p) for r, p in items if p is not None]
    res["inputs"] = [P.rel(p) for _, p in items]
    if not items:
        res["error"] = "no e1_operators.npz found in runs/ or artifacts/a100/ for the per-layer v2 runs"
        return res
    try:
        mod = import_script(P.root / "scripts" / "f2_drift.py", "f2_drift_wrapped")
        base = mod.resolve_run
        by_npz = {str(p.resolve()): r for r, p in items}

        def patched(path):  # npz copies under artifacts/: provenance from the matching runs/ folder
            rr = base(path)
            r = by_npz.get(str(Path(rr["npz"]).resolve()))
            if r is not None:
                rr.update({"model": r["model"], "fit_seed": r["fit_seed"], "name": r["name"],
                           "input_git_hash": r["git"], "layers_csv": r["res"] / "e1_layers.csv"})
            return rr

        mod.resolve_run = patched
        meta = mod.main(["--run", *[str(p) for _, p in items], "--out", str(out)])
        res.update({"status": "ok", "meta": meta})
    except Exception as e:  # noqa: BLE001
        res.update({"status": "failed", "error": f"{e}\n{traceback.format_exc()}"})
    return res


def build_f1_table(P, f1res: list[dict]) -> Table:
    T = Table("T_f1_ranges", "F1 (scripts/f1_tax.py): means over the first and second half of the blocks",
              ["model", "blocks"],
              [("model", "model", ""), ("blocks", "blocks", ""), ("fit seeds", "fit_seeds", ""),
               ("tax", "tax_formula_mean", ".3f"), ("saving", "saving_realised_mean", ".3f"),
               ("twin frac", "twin_frac_mean", ".3f"), ("s1", "s1_mean", ".3f"), ("s1 null", "s1_null_mean", ".3f"),
               ("d-/d", "dminus_frac_mean", ".3f")])
    for fr in f1res:
        if fr["status"] not in ("ok", "reused"):
            continue
        f = Path(fr["out"]) / "f1_ranges.csv"
        df = read_csv(f)
        if df is None:
            continue
        g = ((fr.get("meta") or {}).get("git") or {}).get("git_hash")
        for _, r in df.iterrows():
            row = {c: py(r[c]) for c in df.columns}
            row.update({"_src": P.rel(f), "_git": g, "_seed": None, "_loc": {}, "_derived": {}})
            for c in df.columns:
                if c.endswith("_mean"):
                    row["_loc"][c] = csv_loc(P.rel(f), {"model": r["model"], "blocks": r["blocks"]}, c)
            T.add(row)
    return T


def build_f2_table(P, f2list: list[dict]) -> Table:
    T = Table("T_f2_drift", "F2 (scripts/f2_drift.py): neighbouring-layer agreement, means over layer pairs", ["run"],
              [("run", "run", ""), ("model", "model", ""), ("fit seed", "fit_seed", "d"),
               ("tr(A_l A_l+1)/d", "agree_full", ".3f"), ("null", "agree_full_null", ".3f"),
               ("energy-weighted", "agree_energy_weighted", ".3f"), ("PA mean cos", "pa_mean_cos", ".3f"),
               ("PA null", "pa_null_mean_cos", ".3f")])
    for f2res in f2list:
        f = Path(f2res["out"]) / "f2_neighbours.csv"
        df = read_csv(f) if f2res.get("status") in ("ok", "failed", "reused") else None
        if df is None:
            continue
        g = ";".join(sorted(set(df["git_hash"].astype(str)))) if "git_hash" in df else None
        for run, d in df.groupby("run", sort=False):
            row = {"run": run, "model": d["model"].iloc[0], "fit_seed": py(d["fit_seed"].iloc[0]), "n_pairs": len(d),
                   "group": f2res.get("tag"), "f2_status": f2res.get("status"),
                   "_src": P.rel(f), "_git": g, "_seed": py(d["seed"].iloc[0]) if "seed" in d else None,
                   "_fit_seed": py(d["fit_seed"].iloc[0]), "_loc": {}, "_derived": {}}
            for c, s in (("agree_full", "agree_full"), ("agree_full_null", "agree_full_null"),
                         ("agree_energy_weighted", "agree_energy_weighted_l"), ("pa_mean_cos", "pa_mean_cos"),
                         ("pa_null_mean_cos", "pa_null_mean_cos")):
                if s in d:
                    row[c] = py(d[s].mean())
                    row["_derived"][c] = f"mean of {s} over neighbouring layer pairs in {P.rel(f)}"
            T.add(row)
    return T


# ================================================================================================
# Figures
# ================================================================================================
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_accuracy_vs_k(Tm: Table, path: Path) -> bool:
    plt = _plt()
    series = [("mirror_mlp", C_BLUE, "-", "o"), ("mirror_mlp_m1", C_BLUE, ":", "^"), ("vlocal", C_BLUE, "--", "s"),
              ("vexact", C_AQUA, "--", "D"), ("svd_mlp", C_ORANGE, "-", "o"), ("svd_attn", C_ORANGE, "--", "s"),
              ("random_mlp", C_NULL, "-", "o"), ("random_attn", C_NULL, "--", "s")]
    models = [m for m in MAIN_MODELS if any(r["model"] == m and r["status"] == "ok" for r in Tm.rows)]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.4), facecolor=BG, squeeze=False)
    for ax, model in zip(axes[0], models):
        ax.set_facecolor(BG)
        o = Tm.find(model=model, k=0, variant="original")
        o_top = o.get("top1_mean") if o and o["status"] == "ok" else None
        for key, col, ls, mk in series:
            rows = [r for r in Tm.rows if r["model"] == model and r["variant"] == key and r["status"] == "ok"]
            if not rows:
                continue
            xs = ([0] if o_top is not None else []) + [r["k"] for r in rows]
            ys = ([o_top] if o_top is not None else []) + [r["top1_mean"] for r in rows]
            es = ([0.0] if o_top is not None else []) + [r.get("top1_std") or 0.0 for r in rows]
            ax.errorbar(xs, ys, yerr=es if any(es) else None, color=col, linestyle=ls, marker=mk, markersize=6,
                        linewidth=2, capsize=3,
                        label=VARIANT_INFO[key][1] + (" (M1 run, not canonical where X2 exists)" if key == "mirror_mlp_m1" else ""))
        if o_top is not None:
            ax.axhline(o_top, color=C_INK, linestyle=":", linewidth=1, label="original")
        ax.set_xticks([0] + MAIN_KS)
        ax.set_ylim(0, 85)
        ax.set_xlabel("blocks converted (k)")
        ax.set_ylabel("k-NN top-1 (%)")
        ax.set_title(f"{MODEL_LABEL[model]}: zero-shot top-1 against k", fontsize=10, color=C_INK, loc="left")
        _style(ax)
        ax.set_xticks([0] + MAIN_KS)
    handles, labels = {}, []
    for ax in axes[0]:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in handles:
                handles[lb] = hd
                labels.append(lb)
    fig.legend([handles[lb] for lb in labels], labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_healing(m3: dict, path: Path) -> bool:
    plt = _plt()
    cols = {"mirror": C_BLUE, "svd": C_ORANGE, "random": C_NULL}
    have = [m for m in ("mirror", "svd", "random") if m3[m]["train_log"] is not None or m3[m]["heldout_log"] is not None]
    if not have:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
    for m in have:
        tl, hl = m3[m]["train_log"], m3[m]["heldout_log"]
        if tl is not None:
            axes[0].plot(tl["step"], tl["loss"], color=cols[m], linewidth=1.2, alpha=0.9, label=m)
        if hl is not None:
            axes[1].plot(hl["step"], hl["total"], color=cols[m], linewidth=2, marker="o", markersize=6, label=m)
    for ax, t, yl in ((axes[0], "Training distillation loss", "loss (per step)"),
                      (axes[1], "Held-out distillation loss (5k train images)", "loss")):
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10, color=C_INK, loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_tax(Ts: Table, path: Path) -> bool:
    plt = _plt()
    models = [m for m in MAIN_MODELS if any(r["model"] == m and r.get("block") for r in Ts.rows)]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for m in models:
        rows = [r for r in Ts.rows if r["model"] == m and r.get("block")]
        blocks = sorted({r["block"] for r in rows})
        mean, std, sav = [], [], []
        for b in blocks:
            mm, ss, _ = agg([r["tax_formula"] for r in rows if r["block"] == b])
            mean.append(mm)
            std.append(ss or 0.0)
            sav.append(np.mean([r["saving_realised"] for r in rows if r["block"] == b]))
        n = len({r["fit_seed"] for r in rows})
        ax.errorbar(blocks, mean, yerr=std if any(std) else None, color=MODEL_COLOUR[m], marker=MODEL_MARKER[m],
                    markersize=6, linewidth=2, capsize=3, label=f"{MODEL_LABEL[m]}: formula ({n} seed{'s' if n > 1 else ''})")
        ax.plot(blocks, sav, linestyle="none", marker="o", markersize=10, markerfacecolor="none",
                markeredgecolor=MODEL_COLOUR[m], markeredgewidth=1.2, label=f"{MODEL_LABEL[m]}: realised saving")
    ax.axhline(0.5, color=C_NULL, linestyle="--", linewidth=1.2, label="upper bound 1/2")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("block")
    ax.set_ylabel("redundant share of MLP parameters")
    ax.set_title("Symmetry tax against depth", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_eA(Ts: Table, path: Path) -> bool:
    plt = _plt()
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in Ts.rows)]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for m in models:
        seeds = sorted({r["fit_seed"] for r in Ts.rows if r["model"] == m}, key=lambda x: (x is None, x))
        rows = sorted([r for r in Ts.rows if r["model"] == m and r["fit_seed"] == seeds[0]], key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in rows], [r["e_A"] for r in rows], color=MODEL_COLOUR[m], marker=MODEL_MARKER[m],
                markersize=6, linewidth=2, label=f"{MODEL_LABEL[m]} (fit seed {seeds[0]})")
    ax.set_yscale("log")
    ax.set_xlabel("layer (output of block l)")
    ax.set_ylabel("relative held-out error e(A_l)")
    ax.set_title("Per-layer operator fit against depth", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_structure_summary(Tg: Table, path: Path) -> bool:
    plt = _plt()
    rows = [r for r in Tg.rows if r["fit_seed"] == 0]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99)
    labels = [r["model_label"] for r in rows]
    x = np.arange(len(rows))
    panels = [("row1_value", "e(null) / e(A), layers 1 to L/2", True), ("dminus_frac", "d- / d, layers 1 to L/2", False),
              ("twin_frac", "twin fraction, blocks 1 to L/2", False), ("share", "breaking share vs matched null", False),
              ("mlp_saving_first_half", "MLP parameter saving, blocks 1 to L/2", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2), facecolor=BG)
    for ax, (col, title, logy) in zip(axes, panels):
        ax.set_facecolor(BG)
        cols = [MODEL_COLOUR.get(r["model"], C_NULL) for r in rows]
        if col == "share":
            w = 0.38
            ax.bar(x - w / 2, [r["p4_share"] for r in rows], width=w, color=cols, label="max(s1, s2)")
            ax.bar(x + w / 2, [max(r["s1_null"], r["s2_null"]) for r in rows], width=w, color=C_NULL, alpha=0.6,
                   label="matched null")
            ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        else:
            ax.bar(x, [r.get(col) if r.get(col) is not None else 0 for r in rows], width=0.6, color=cols)
        if logy:
            from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9, color=C_INK, loc="left")
        ax.grid(True, axis="y", color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ================================================================================================
# INDEX.md
# ================================================================================================
def build_index(P, ctx, R: Registry) -> list[str]:
    T = ctx["tables"]
    st = ctx["status"]
    L = ["# Paper data package: index", ""]
    L.append(f"Generated by `scripts/collect_results.py` on {ctx['date']} at commit {short(ctx['git']['git_hash'])}"
             f"{' (uncommitted changes)' if ctx['git'].get('git_dirty') else ''}. Every number below is read from a table "
             "in `tables/`, which is built from the result files named; `numbers.json` holds each one with its source "
             "file, seed and git hash. Rerun the collector after new results arrive; missing inputs show as "
             f"\"{NOT_RUN}\" (or \"{PENDING}\" for a check waiting for a run).")
    L.append("")
    L.append("Protocol v2 (the paper protocol): operators fitted on ImageNet train images, held-out errors and k-NN "
             "queries on validation, k-NN bank of 100 train images per class. Protocol v1 (E1, E1b): all splits from "
             "validation; kept unchanged and marked as the earlier protocol.")
    L.append("")

    def c(key, tname, where, col, spec=".2f", headline=None):
        t = T[tname]
        return R.cite(key, t, t.find(**where), col, spec, headline)

    def runline(r):
        return (f"`{r['dir']}`: config `{r['config_path']}`, model {r['model']}, protocol {r['protocol']}, "
                f"fit seed {r['fit_seed'] if r['fit_seed'] is not None else 'n/a'}, run seed {r['seed']}, git "
                f"{short(r['git'])}{' (dirty)' if r.get('git_dirty') else ''}"
                f"{'' if r['git_source'] in (None, 'e1_meta.json') else ' (' + r['git_source'] + ')'}, "
                f"{r['date'] or 'date not recorded'}; status {r['status']}")

    def reasons(tokens):
        hits = status_mentions(st, tokens)
        return hits

    e1, e1b = ctx["e1"], ctx["e1b"]
    # ---------------------------------------------------------------- E1
    L += ["## E1: one shared operator, DINOv2 ViT-B/14 (protocol v1)", ""]
    if e1 and e1["status"] != NOT_RUN:
        L.append(f"- Run: {runline(e1)}.")
        L.append(f"- Raw files: `{e1['res_rel']}/` (`e1_meta.json`, `e1_layers.csv`, `e1_mlp.csv`, `e1_surgery.csv`, "
                 "`E1_REPORT.md`, `E1_DIAGNOSTIC.md`).")
        L.append(f"- Pre-registered outcome (E1_INSTRUCTIONS.md, commit {short(ctx['commits'].get('E1_INSTRUCTIONS.md'))}): "
                 f"**{c('e1.outcome', 'T_e1_history', {'run_id': 'E1'}, 'outcome')}**.")
        L.append(f"- Decision rows 1 to 5: {c('e1.row1', 'T_e1_history', {'run_id': 'E1'}, 'row1_value', '.3f')} "
                 f"({c('e1.row1.cell', 'T_e1_history', {'run_id': 'E1'}, 'row1_cell')}), "
                 f"{c('e1.row2', 'T_e1_history', {'run_id': 'E1'}, 'row2_value', '.3f')}, "
                 f"{c('e1.row3', 'T_e1_history', {'run_id': 'E1'}, 'row3_value', '.4f')}, "
                 f"{c('e1.row4', 'T_e1_history', {'run_id': 'E1'}, 'row4_value', '.3f')}, "
                 f"{c('e1.row5', 'T_e1_history', {'run_id': 'E1'}, 'row5_value', '.2f')} (drop at k = 6).")
        L.append(f"- k-NN top-1 (%): original {c('e1.original', 'T_e1_history', {'run_id': 'E1'}, 'original_top1')}; "
                 f"k = 6 surgery {c('e1.k6.surgery', 'T_e1_history', {'run_id': 'E1'}, 'surgery_k6')}, "
                 f"SVD {c('e1.k6.svd', 'T_e1_history', {'run_id': 'E1'}, 'svd_k6')}.")
        dg = T["T_e1_history"].find(run_id="E1 per-layer diagnostic")
        if dg:
            L.append(f"- Post hoc per-layer diagnostic (no verdict): k = 6 surgery "
                     f"{c('e1diag.k6.surgery', 'T_e1_history', {'run_id': 'E1 per-layer diagnostic'}, 'surgery_k6')}, SVD "
                     f"{c('e1diag.k6.svd', 'T_e1_history', {'run_id': 'E1 per-layer diagnostic'}, 'svd_k6')}; row 1 "
                     f"{c('e1diag.row1', 'T_e1_history', {'run_id': 'E1 per-layer diagnostic'}, 'row1_value')}.")
        L.append("- Tables: `T_e1_history`, `T_prereg` (ids E1.*).")
    else:
        L.append(f"- {NOT_RUN}: no `results/e1_meta.json`.")
    L.append("")
    # ---------------------------------------------------------------- E1b
    L += ["## E1b: per-layer operators, pre-registered (protocol v1)", ""]
    rows = [r for r in T["T_e1_history"].rows if r["run_id"].startswith("E1b")]
    if rows:
        L.append(f"- Pre-registration: E1B_PREREG.md, commit {short(ctx['commits'].get('E1B_PREREG.md'))}.")
        for r in rows:
            rid = r["run_id"]
            kk = rid.replace("E1b ", "").replace(" ", "_")
            run = e1b.get("e1b_" + rid.replace("E1b ", ""))
            L.append(f"- {rid} ({r['role']}): outcome **{c(f'e1b.{kk}.outcome', 'T_e1_history', {'run_id': rid}, 'outcome')}**; "
                     f"rows 1 to 5: {c(f'e1b.{kk}.row1', 'T_e1_history', {'run_id': rid}, 'row1_value')}, "
                     f"{c(f'e1b.{kk}.row2', 'T_e1_history', {'run_id': rid}, 'row2_value', '.3f')}, "
                     f"{c(f'e1b.{kk}.row3', 'T_e1_history', {'run_id': rid}, 'row3_value', '.3f')}, "
                     f"{c(f'e1b.{kk}.row4', 'T_e1_history', {'run_id': rid}, 'row4_value', '.3f')}, "
                     f"{c(f'e1b.{kk}.row5', 'T_e1_history', {'run_id': rid}, 'row5_value')}; surgery k = 3 / 6 / 9: "
                     f"{c(f'e1b.{kk}.k3.surgery', 'T_e1_history', {'run_id': rid}, 'surgery_k3')} / "
                     f"{c(f'e1b.{kk}.k6.surgery', 'T_e1_history', {'run_id': rid}, 'surgery_k6')} / "
                     f"{c(f'e1b.{kk}.k9.surgery', 'T_e1_history', {'run_id': rid}, 'surgery_k9')}, SVD "
                     f"{c(f'e1b.{kk}.k3.svd', 'T_e1_history', {'run_id': rid}, 'svd_k3')} / "
                     f"{c(f'e1b.{kk}.k6.svd', 'T_e1_history', {'run_id': rid}, 'svd_k6')} / "
                     f"{c(f'e1b.{kk}.k9.svd', 'T_e1_history', {'run_id': rid}, 'svd_k9')}.")
            if run:
                L.append(f"  - Run: {runline(run)}.")
        L.append("- Tables: `T_e1_history`, `T_prereg` (ids E1b.*). Summary: `results/E1B_SUMMARY.md`.")
    else:
        L.append(f"- {NOT_RUN}.")
    L.append("")
    # ---------------------------------------------------------------- X1, X1b
    tx = T["T_x_checks"]
    L += ["## X1: SVD-collapse check (gate G1)", ""]
    if ctx["x1"]:
        L.append(f"- Run: `{ctx['x1']['file']}` (scripts/x1_svd_check.py), git {short((ctx['x1']['json'].get('git') or {}).get('git_hash'))}; "
                 f"seed {ctx['x1'].get('seed')} ({ctx['x1'].get('seed_source')}).")
        for ms in ("reg", "plain"):
            if tx.find(check_id=f"x1.{ms}.full_rank_diff"):
                L.append(f"- {MODEL_LABEL['dinov2_vitb14_reg' if ms == 'reg' else 'dinov2_vitb14']}: full-rank max abs "
                         f"difference {c(f'x1.{ms}.full_rank_diff', 'T_x_checks', {'check_id': f'x1.{ms}.full_rank_diff'}, 'value', '.2e')}; "
                         f"folding max relative {c(f'x1.{ms}.fold_max_rel', 'T_x_checks', {'check_id': f'x1.{ms}.fold_max_rel'}, 'value', '.2e')}; "
                         f"rank ratio fc1 min {c(f'x1.{ms}.rank_ratio_fc1_min', 'T_x_checks', {'check_id': f'x1.{ms}.rank_ratio_fc1_min'}, 'value', '.4f')} "
                         f"({c(f'x1.{ms}.rank_ratio_fc1.verdict', 'T_x_checks', {'check_id': f'x1.{ms}.rank_ratio_fc1_min'}, 'verdict')}); "
                         f"finite features {c(f'x1.{ms}.finite.verdict', 'T_x_checks', {'check_id': f'x1.{ms}.finite'}, 'verdict')}.")
        L.append(f"- Gate G1: **{c('g1.verdict', 'T_prereg', {'id': 'G1'}, 'verdict')}** "
                 f"({T['T_prereg'].find(id='G1').get('note', '')}).")
    else:
        L.append(f"- {NOT_RUN}: no `runs/x1/x1_svd_check.json`.")
    L.append("")
    L += ["## X1b: SVD compounding over blocks", ""]
    if ctx["x1b"]:
        for ms in ("reg", "plain"):
            ks = [r for r in tx.rows if r["check_id"].startswith(f"x1b.{ms}.")]
            if ks:
                vals = ", ".join(f"k = {r['check_id'].split('.')[2][1:]}: "
                                 f"{R.cite(r['check_id'], tx, r, 'value')}" for r in ks)
                L.append(f"- {MODEL_LABEL['dinov2_vitb14_reg' if ms == 'reg' else 'dinov2_vitb14']}, top-1 on the small "
                         f"protocol with SVD on the first k blocks: {vals}.")
        L.append(f"- Run: `{ctx['x1b']['file']}` (scripts/x1b_svd_compound.py). Table `T_x_checks`.")
    else:
        L.append(f"- {NOT_RUN}.")
    L.append("")
    # ---------------------------------------------------------------- X2
    L += ["## X2: E1b arms under the paper protocol (v2)", ""]
    for name, r in sorted(ctx["x2"].items()):
        L.append(f"- {runline(r)}.")
        g = T["T_generality"].find(model=r["model"], fit_seed=r["fit_seed"]) if r["operator_mode"] == "per_layer" else None
        if g and g.get("outcome"):
            ms, fs = MODEL_SHORT.get(r["model"], r["model"]), r["fit_seed"]
            wh = {"model": r["model"], "fit_seed": fs}
            L.append(f"  - Decision outcome **{c(f'x2.{ms}.s{fs}.outcome', 'T_generality', wh, 'outcome')}**"
                     f" ({r['decision_source']}); row 4 {c(f'x2.{ms}.s{fs}.row4', 'T_generality', wh, 'row4_value', '.3f')}.")
        if r["operator_mode"] != "per_layer" and r["surgery"] is not None:
            ks = sorted(int(x) for x in set(r["surgery"]["k"]) if int(x))
            L.append(f"  - Shared operator arm (exploratory, not in the main table); surgery rows present for k = "
                     f"{', '.join(map(str, ks))}; see `{r['surgery_file']}`.")
        if r["status"] == NOT_RUN:
            L.append(f"  - {NOT_RUN} (planned in {(r.get('job') or {}).get('job_file', 'runs/x2/')}; see the reasons below).")
    if any(r["status"] == NOT_RUN for r in ctx["x2"].values()):
        rs = reasons(["X3 seed 1", "seed 1", "plain-model seeds"])
        if rs:
            L.append("- Reasons for the runs not run: " + " | ".join(rs) + ".")
    tm = T["T_main_zeroshot"]
    for model in MAIN_MODELS:
        ms = MODEL_SHORT[model]
        if tm.find(model=model, k=6, variant="mirror_mlp") and tm.find(model=model, k=6, variant="mirror_mlp")["status"] == "ok":
            parts = []
            for k in MAIN_KS:
                rr = tm.find(model=model, k=k, variant="mirror_mlp")
                if rr and rr["status"] == "ok":
                    parts.append(f"k = {k}: {c(f'main.{ms}.k{k}.mirror_mlp.top1', 'T_main_zeroshot', {'model': model, 'k': k, 'variant': 'mirror_mlp'}, 'top1_mean')} "
                                 f"(SVD {c(f'main.{ms}.k{k}.svd_mlp.top1', 'T_main_zeroshot', {'model': model, 'k': k, 'variant': 'svd_mlp'}, 'top1_mean')}, "
                                 f"random {c(f'main.{ms}.k{k}.random_mlp.top1', 'T_main_zeroshot', {'model': model, 'k': k, 'variant': 'random_mlp'}, 'top1_mean')}; "
                                 f"n = {rr['n_seeds']})")
            L.append(f"- {MODEL_LABEL[model]}, {LABEL_MIRROR_MLP} (canonical source of every MLP-only number), top-1 (%): original "
                     f"{c(f'main.{ms}.original.top1', 'T_main_zeroshot', {'model': model, 'k': 0, 'variant': 'original'}, 'top1_mean', headline=f'{MODEL_LABEL[model]} original k-NN top-1')}; "
                     + "; ".join(parts) + ".")
    L.append("- Tables: `T_main_zeroshot` (MLP-only rows), `T_generality` (decision rows), `T_structure_per_layer`.")
    L.append("")
    # ---------------------------------------------------------------- X3
    L += ["## X3: fit seeds", ""]
    for model in MAIN_MODELS:
        have = sorted({r["fit_seed"] for r in ctx["x2"].values() if r["model"] == model and r["status"] != NOT_RUN
                       and r["operator_mode"] == "per_layer"})
        want = sorted({r["fit_seed"] for r in ctx["x2"].values() if r["model"] == model and r["operator_mode"] == "per_layer"})
        L.append(f"- {MODEL_LABEL[model]}: fit seeds with results {', '.join(map(str, have)) or 'none'}; planned "
                 f"{', '.join(map(str, want))} (jobs/*.txt and runs/x2/).")
    rr = tm.find(model="dinov2_vitb14_reg", k=6, variant="mirror_mlp")
    if rr and rr["status"] == "ok":
        L.append(f"- DINOv2-reg, {LABEL_MIRROR_MLP} at k = 6: {c('main.reg.k6.mirror_mlp.top1', 'T_main_zeroshot', {'model': 'dinov2_vitb14_reg', 'k': 6, 'variant': 'mirror_mlp'}, 'top1_mean')}"
                 f" ± {c('main.reg.k6.mirror_mlp.top1_std', 'T_main_zeroshot', {'model': 'dinov2_vitb14_reg', 'k': 6, 'variant': 'mirror_mlp'}, 'top1_std', '.3f')}"
                 f" (n = {c('main.reg.k6.mirror_mlp.n', 'T_main_zeroshot', {'model': 'dinov2_vitb14_reg', 'k': 6, 'variant': 'mirror_mlp'}, 'n_seeds', 'd')}).")
    rs = reasons(["X3"])
    if rs:
        L.append("- Not yet run: " + " | ".join(rs) + ".")
    L.append("- The plan needs 3 fit seeds for every headline number; headline numbers below with n < 3 are not final.")
    L.append("")
    # ---------------------------------------------------------------- X4
    L += ["## X4: full LayerNorm folding", ""]
    for ms, model in (("reg", "dinov2_vitb14_reg"), ("plain", "dinov2_vitb14")):
        bits = []
        for dt in ("float32", "float64"):
            r = tx.find(check_id=f"x4.{ms}.{dt}.max_abs")
            if r:
                bits.append(f"{dt}: max abs difference {c(f'x4.{ms}.{dt}.max_abs', 'T_x_checks', {'check_id': f'x4.{ms}.{dt}.max_abs'}, 'value', '.2e')} "
                            f"({c(f'x4.{ms}.{dt}.verdict', 'T_x_checks', {'check_id': f'x4.{ms}.{dt}.max_abs'}, 'verdict')} against {r['bar']})")
        if bits:
            L.append(f"- {MODEL_LABEL[model]}: " + "; ".join(bits) + ".")
    if tx.find(check_id="x4.m1.folded.top1"):
        L.append(f"- k-NN top-1 of the folded model {c('x4.folded.top1', 'T_x_checks', {'check_id': 'x4.m1.folded.top1'}, 'value')} "
                 f"against {c('x4.original.top1', 'T_x_checks', {'check_id': 'x4.m1.original.top1'}, 'value')} for the original "
                 "(M1 run, protocol v2).")
    L.append(f"- Raw files: {', '.join('`' + v['file'] + '`' for v in ctx['x4'].values()) or NOT_RUN}. Table `T_x_checks`.")
    L.append("")
    # ---------------------------------------------------------------- F1
    L += ["## F1: symmetry tax", ""]
    ss = T["T_structure_summary"]
    for model in MAIN_MODELS:
        ms = MODEL_SHORT[model]
        r = ss.find(model=model, blocks="1-6")
        if r:
            L.append(f"- {MODEL_LABEL[model]} (fit seeds {r['fit_seeds']}), blocks 1 to 6: tax (formula) "
                     f"{c(f'f1.{ms}.first_half.tax', 'T_structure_summary', {'model': model, 'blocks': '1-6'}, 'tax_formula_mean', '.3f', headline=f'{MODEL_LABEL[model]} symmetry tax, blocks 1 to 6 (formula)')}, "
                     f"realised saving {c(f'f1.{ms}.first_half.saving', 'T_structure_summary', {'model': model, 'blocks': '1-6'}, 'saving_realised_mean', '.3f')}, "
                     f"twin fraction {c(f'f1.{ms}.first_half.twin', 'T_structure_summary', {'model': model, 'blocks': '1-6'}, 'twin_frac_mean', '.3f')}, "
                     f"s1 {c(f'f1.{ms}.first_half.s1', 'T_structure_summary', {'model': model, 'blocks': '1-6'}, 's1_mean', '.3f')} "
                     f"(null {c(f'f1.{ms}.first_half.s1_null', 'T_structure_summary', {'model': model, 'blocks': '1-6'}, 's1_null_mean', '.3f')}).")
    if not ctx["f1"]:
        L.append("- `scripts/f1_tax.py` was not run in this collection (skipped, and no earlier outputs to reuse).")
    for fr in ctx["f1"]:
        if fr["status"] in ("ok", "reused"):
            fm = Path(fr["out"]) / "f1_meta.json"
            mx = R.put(f"f1.{fr['tag']}.max_abs_tax_minus_saving", (fr.get("meta") or {}).get("max_abs_tax_minus_saving"),
                       ".2e", source=P.rel(fm), git_hash=((fr.get("meta") or {}).get("git") or {}).get("git_hash"),
                       locator=json_loc(P.rel(fm), ["max_abs_tax_minus_saving"]))
            verb = "ran on" if fr["status"] == "ok" else "outputs (reused) for"
            L.append(f"- `scripts/f1_tax.py` {verb} `{P.rel(fr['runs_dir'])}` (arms {', '.join(fr['arms'])}); outputs in "
                     f"`{P.rel(fr['out'])}/` and copied as `tables/ext_{fr['tag']}_*`, `figures/ext_{fr['tag']}_fig2_tax.png`; "
                     f"largest |tax - saving| recorded in its f1_meta.json: {mx} (bookkeeping check).")
            if fr["status"] == "reused":
                L.append(f"  - Reused from an earlier collection (f1_meta.json: {(fr.get('meta') or {}).get('date')}, commit "
                         f"{short(((fr.get('meta') or {}).get('git') or {}).get('git_hash'))}); not refreshed in this run "
                         "(skipped with --skip-scripts)." + (f" Stale: runs present now that it did not read: {', '.join(fr['stale'])}."
                                                if fr.get("stale") else ""))
            if not fr.get("hist"):
                L.append("  - Twin-score histograms: not drawn (operator npz files for these runs are not on this "
                         "machine; they are expected under `artifacts/a100/`).")
        else:
            L.append(f"- `scripts/f1_tax.py` on `{P.rel(fr['runs_dir'])}`: {fr['status']} ({str(fr.get('error', ''))[:200]}).")
    L.append("- Tables: `T_structure_per_layer`, `T_structure_summary`; figure `figures/fig_tax_vs_depth.png`.")
    L.append("")
    # ---------------------------------------------------------------- F2
    L += ["## F2: operator drift", ""]
    f2list = ctx["f2"]
    if not f2list:
        L.append("- `scripts/f2_drift.py` was not run in this collection (skipped, and no earlier outputs to reuse).")
    for f2 in f2list:
        if f2.get("status") == "ok":
            L.append(f"- `scripts/f2_drift.py` ({f2['tag']}) ran on {', '.join('`' + x + '`' for x in f2['inputs'])}; "
                     f"outputs in `{P.rel(f2['out'])}/` (copied as `tables/ext_{f2['tag']}_*`, `figures/ext_{f2['tag']}_*`).")
        elif f2.get("status") == "reused":
            L.append(f"- `scripts/f2_drift.py` ({f2['tag']}) outputs reused from an earlier collection (f2_meta.json: "
                     f"{(f2.get('meta') or {}).get('date')}, commit {short(((f2.get('meta') or {}).get('git') or {}).get('git_hash'))}), "
                     f"read from {', '.join('`' + x + '`' for x in f2.get('inputs', []))}; not refreshed in this run "
                     "(skipped with --skip-scripts or --skip-f2)." + (f" Stale: operator files present now that it did not read: "
                                            f"{', '.join('`' + x + '`' for x in f2['stale'])}." if f2.get("stale") else ""))
        elif f2.get("status") == "failed":
            L.append(f"- `scripts/f2_drift.py` ({f2['tag']}) failed after reading {', '.join('`' + x + '`' for x in f2.get('inputs', []))}: "
                     f"{str(f2.get('error', '')).splitlines()[0][:160]}; its CSVs, if written, are still summarised below.")
        else:
            L.append(f"- `scripts/f2_drift.py` ({f2['tag']}): {NOT_RUN} ({f2.get('error', '')}).")
    if f2list:
        for r in T["T_f2_drift"].rows:
            rn = r["run"]
            wh = {"run": rn}
            L.append(f"- {rn} ({r['model']}): mean neighbour agreement tr(A_l A_l+1)/d "
                     f"{c(f'f2.{rn}.agree', 'T_f2_drift', wh, 'agree_full', '.3f')} "
                     f"(null {c(f'f2.{rn}.agree_null', 'T_f2_drift', wh, 'agree_full_null', '.3f')}); "
                     f"principal-angle mean cosine {c(f'f2.{rn}.pa', 'T_f2_drift', wh, 'pa_mean_cos', '.3f')} "
                     f"(null {c(f'f2.{rn}.pa_null', 'T_f2_drift', wh, 'pa_null_mean_cos', '.3f')}).")
        missing = [r["dir"] for r in list(ctx["x2"].values()) + list(ctx["g"].values())
                   if r["operator_mode"] == "per_layer" and r["status"] != NOT_RUN and npz_for(P, r) is None]
        if missing:
            L.append(f"- Not run for {', '.join('`' + m + '`' for m in missing)}: their `e1_operators.npz` files are not on "
                     "this machine (expected under `artifacts/a100/<run>/results/`).")
        L.append("- Table `T_f2_drift` (means over neighbouring layer pairs); full outputs `tables/ext_f2_*`.")
    L.append("")
    # ---------------------------------------------------------------- F3
    L += ["## F3: artefact tokens (P1, gate G3)", ""]
    for cid in ("P1.plain", "P1.registers"):
        r = T["T_prereg"].find(id=cid)
        if r and r["verdict"] != NOT_RUN:
            L.append(f"- {cid}: {r['statement']} Value {c(cid.lower() + '.value', 'T_prereg', {'id': cid}, 'value', '.3f', headline=f'{cid} value')} "
                     f"({r['threshold']}): **{c(cid.lower() + '.verdict', 'T_prereg', {'id': cid}, 'verdict')}**. Source `{r['source_file']}`, "
                     f"run git {short(r.get('run_git_hash'))}, pre-registration commit recorded by the run "
                     f"{short(r.get('commit_recorded_by_run'))}.")
        else:
            L.append(f"- {cid}: {NOT_RUN}.")
    L.append(f"- Gate G3: **{c('g3.verdict', 'T_prereg', {'id': 'G3'}, 'verdict')}**. Raw files: `runs/f3/` (per-layer CSVs "
             "and `plots/fig3b_artefacts.png`).")
    L.append("")
    # ---------------------------------------------------------------- G
    L += ["## G: generality (P4)", ""]
    tg = T["T_generality"]
    for r in tg.rows:
        ms = MODEL_SHORT.get(r["model"], r["model"])
        key = f"g.{ms}.s{r['fit_seed']}"
        L.append(f"- {r['model_label']} (fit seed {r['fit_seed']}, `{r['run']}`, git {short(r['git_hash'])}): outcome "
                 f"**{c(key + '.outcome', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'outcome')}** "
                 f"({r['decision_source']}); row 1 {c(key + '.row1', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'row1_value')}, "
                 f"row 4 {c(key + '.row4', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'row4_value', '.3f')}; "
                 f"k = {r['k_half']}: surgery {c(key + '.surgery', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'top1_surgery_half')} "
                 f"vs SVD {c(key + '.svd', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'top1_svd_half')} "
                 f"(original {c(key + '.original', 'T_generality', {'model': r['model'], 'fit_seed': r['fit_seed']}, 'top1_original')}).")
        if r.get("note"):
            L.append(f"  - {r['note']}")
    for r in T["T_generality_p4"].rows:
        val = (f"value {c('p4.' + r['check'].lower().replace('p4.', '') + '.value', 'T_generality_p4', {'check': r['check']}, 'value', '.3f')} ({r.get('threshold_text', '')}): "
               if r.get("value") is not None else "")
        L.append(f"- {r['check']}: {r['statement']}; {val}**{c('p4.' + r['check'].lower().replace('p4.', '') + '.verdict', 'T_generality_p4', {'check': r['check']}, 'verdict')}**"
                 f"{(' (' + r['note'] + ')') if r.get('note') else ''}.")
    gm = [m for m in MAIN_MODELS + G_MODELS if not any(r["model"] == m for r in tg.rows)]
    if gm:
        planned = {r["model"]: r["dir"] for r in ctx["g"].values() if r["model"]}
        L.append(f"- {NOT_RUN}: " + ", ".join(MODEL_LABEL[m] + (f" (planned: `{planned[m]}`)" if m in planned else "")
                                              for m in gm) + ".")
    if any(m in gm for m in G_SIZE_PAIR):
        rs = reasons(["DINOv2-S/L", "DINOv2-S", "DINOv2-L"])
        if rs:
            L.append("- DINOv2-S and L (nice-to-have): " + " | ".join(rs) + ".")
    sl = [r for r in tg.rows if r["model"] in G_SIZE_PAIR]
    if sl:
        L.append("- DINOv2 size pair (registers models, configs `configs/g_dinov2_s_reg.yaml` and `configs/g_dinov2_l_reg.yaml`): "
                 "every column is taken over the model's own first half, blocks 1 to L/2, with zero-shot surgery at "
                 "k = L/2 ("
                 + "; ".join(f"{r['model_label']}: {r['depth']} blocks, k = {r['k_half']}" for r in sl) + ").")
    L.append("- Tables: `T_generality`, `T_generality_p4`, `T_structure_per_layer`, `T_structure_summary`; figures "
             "`fig_eA_vs_depth.png`, `fig_structure_summary.png`.")
    L.append("")
    # ---------------------------------------------------------------- M1
    L += ["## M1: attention conversion (V-local, V-exact)", ""]
    m1 = ctx["m1"]
    if m1["status"] != NOT_RUN:
        L.append(f"- Run: `{m1['dir']}` (scripts/m1_attention.py), model {m1['model']}, fit seed "
                 f"{m1['meta'].get('fit_seed')}, run seed {m1['meta'].get('seed')}, random-structure seed "
                 f"{m1['meta'].get('random_seed')}, git {short(m1['git'])}{' (dirty)' if (m1['meta'].get('git') or {}).get('git_dirty') else ''}, "
                 f"{m1['meta'].get('date')}; operators `{(m1['meta'].get('args') or {}).get('operators')}`; protocol "
                 f"{m1['meta'].get('protocol')}.")
        for k in m1["ks"]:
            bits = []
            for v in ("vlocal", "vexact", "mirror_mlp_m1", "svd_attn", "random_attn"):
                rr = tm.find(model=m1["model"], k=k, variant=v)
                if rr and rr["status"] == "ok":
                    hl = f"M1 k = {k} {VARIANT_INFO[v][1]} top-1" if v in ("vlocal", "vexact") else None
                    extra = ""
                    if v == "mirror_mlp_m1":
                        x2r = tm.find(model=m1["model"], k=k, variant="mirror_mlp")
                        if rr.get("canonical") == "no" and x2r is not None:
                            ms_ = MODEL_SHORT.get(m1["model"], m1["model"])
                            extra = (f"; M1 re-run, not canonical: quote {CANONICAL_RUN} "
                                     f"{c(f'main.{ms_}.k{k}.mirror_mlp.top1', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': 'mirror_mlp'}, 'top1_mean')}"
                                     f" (n = {x2r['n_seeds']}); the M1 run reloads the float32 operator file, a numerical difference")
                        else:
                            extra = f"; the only source at this k ({CANONICAL_RUN} did not run it)"
                    bits.append(f"{VARIANT_INFO[v][1]} {c(f'm1.k{k}.{v}.top1', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'top1_mean', headline=hl)} "
                                f"(flip {c(f'm1.k{k}.{v}.flip', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'flip_mean')}{extra})")
            L.append(f"- k = {k}, top-1 (%): " + "; ".join(bits) + ".")
            for v in ("vlocal", "vexact"):
                rr = tm.find(model=m1["model"], k=k, variant=v)
                if rr and rr.get("gflops_saved_model_pct") is not None:
                    L.append(f"  - {VARIANT_INFO[v][1]}: kept {c(f'm1.k{k}.{v}.kept', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'kept_frac_converted', '.3f')} "
                             f"of the converted matrix parameters; analytic GFLOPs saved {c(f'm1.k{k}.{v}.gflops_saved_model', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'gflops_saved_model_pct', '.1f', headline=f'M1 k = {k} {VARIANT_INFO[v][1]}: analytic GFLOPs saved, whole model (%)')}% "
                             f"of the whole model and {c(f'm1.k{k}.{v}.gflops_saved_conv', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'gflops_saved_converted_pct', '.1f')}% "
                             f"of the converted blocks; kept parameters relative to its SVD baseline "
                             f"{c(f'm1.k{k}.{v}.budget_gap_pct', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'budget_gap_vs_svd_pct', '+.2f')}%.")
        va = T["T_attention_vexact"]
        for r in va.rows:
            k = r["k"]
            L.append(f"- V-exact span k = {k}: d+ = {c(f'm1.vexact.k{k}.dplus', 'T_attention_vexact', {'k': k}, 'd_plus', 'd')}, "
                     f"d- = {c(f'm1.vexact.k{k}.dminus', 'T_attention_vexact', {'k': k}, 'd_minus', 'd')}; exactness max relative error in the span "
                     f"{c(f'm1.vexact.k{k}.exact', 'T_attention_vexact', {'k': k}, 'exact_max_rel_in_span', '.1e')} "
                     f"(pass: {c(f'm1.vexact.k{k}.exact_pass', 'T_attention_vexact', {'k': k}, 'exact_pass', '')}).")
        tc = T["T_attention_conversion"]
        q = "flip consistency with the even projection of the final features (%)"
        if tc.find(quantity=q):
            L.append(f"- Full conversion (k = 12, fp32): flip consistency {c('m1.full.flip', 'T_attention_conversion', {'quantity': q}, 'value', '.2f', headline='V-exact full conversion flip consistency (fp32)')}% "
                     f"(check pass: {c('m1.full.pass', 'T_attention_conversion', {'quantity': 'full conversion flip check pass (must be 100%)'}, 'value', '')}).")
        th_ = T["T_attention"]
        if th_.rows:
            src = th_.rows[0]["_src"]
            tot_self = R.put("m1.heads.n_self_total", sum(r["n_self"] for r in th_.rows), "d", source=src, table="T_attention",
                             git_hash=m1["git"], seed=th_.rows[0]["_seed"], fit_seed=th_.rows[0]["_fit_seed"],
                             derived="sum of n_self over the blocks of m1_heads.csv")
            tot_pairs = R.put("m1.heads.n_pairs_total", sum(r["n_pairs"] for r in th_.rows), "d", source=src,
                              table="T_attention", git_hash=m1["git"], seed=th_.rows[0]["_seed"],
                              fit_seed=th_.rows[0]["_fit_seed"], derived="sum of n_pairs over the blocks of m1_heads.csv")
            L.append(f"- Heads: {tot_self} self-heads and {tot_pairs} twin pairs over {len(th_.rows)} blocks (sums of "
                     "`T_attention`); block 1 attention breaking share "
                     f"{c('m1.b1.s_all', 'T_attention', {'block': 1}, 's_all', '.3f')} against "
                     f"{c('m1.b1.s_all_random', 'T_attention', {'block': 1}, 's_all_random', '.3f')} for the matched random structure.")
        if m1["model"] in MAIN_MODELS:
            other = [m for m in MAIN_MODELS if m != m1["model"]]
            L.append(f"- {NOT_RUN} on {', '.join(MODEL_LABEL[m] for m in other)}; k = 3 not run (M1 ran k = "
                     f"{', '.join(map(str, m1['ks']))}).")
        L.append("- A6 (MLP only vs MLP plus attention) and A7 (V-local vs V-exact), zero-shot part: the M1 rows of "
                 "`T_main_zeroshot`. Tables: `T_attention`, `T_attention_vexact`, `T_attention_conversion`.")
    else:
        L.append(f"- {NOT_RUN}.")
    L.append("")
    # ---------------------------------------------------------------- M3
    L += ["## M3: healing pilot (P3, gate G2)", ""]
    th_ = T["T_healing"]
    for r in th_.rows:
        m = r["method"]
        if r["status"] == NOT_RUN:
            rs = reasons(["M3"])
            L.append(f"- {m}: {NOT_RUN} (no files under `runs/m3_pilot/{m}/`)" + (" | " + " | ".join(rs) if rs else "") + ".")
            continue
        bits = [f"status {r['status']}"]
        if r.get("chosen_lr") is not None:
            bits.append(f"chosen LR {c(f'm3.{m}.lr', 'T_healing', {'method': m}, 'chosen_lr', 'g')}")
        if r.get("heldout_loss_before") is not None:
            bits.append(f"held-out loss {c(f'm3.{m}.heldout_before', 'T_healing', {'method': m}, 'heldout_loss_before', '.3f')} to "
                        f"{c(f'm3.{m}.heldout_after', 'T_healing', {'method': m}, 'heldout_loss_after', '.3f')} over "
                        f"{c(f'm3.{m}.steps', 'T_healing', {'method': m}, 'steps_done', 'd')} steps")
        if r.get("healed_top1") is not None:
            bits.append(f"top-1 zero-shot {c(f'm3.{m}.zero_shot', 'T_healing', {'method': m}, 'zero_shot_top1')} to healed "
                        f"{c(f'm3.{m}.healed', 'T_healing', {'method': m}, 'healed_top1', headline=f'M3 pilot healed {m} top-1 (k = 9)')} "
                        f"(original {c(f'm3.{m}.original', 'T_healing', {'method': m}, 'original_top1')}); flip consistency healed "
                        f"{c(f'm3.{m}.healed_flip', 'T_healing', {'method': m}, 'healed_flip')}")
        else:
            bits.append("healed k-NN: " + PENDING)
        L.append(f"- {m}: " + "; ".join(bits) + f". Seed {r.get('seed')}, git {short(r.get('git_hash'))}; files `{r['source']}`.")
    for r in T["T_healing_p3"].rows:
        L.append(f"- {r['check']}: {r['statement']}"
                 + (f", value {c('m3.' + r['check'].lower() + '.value', 'T_healing_p3', {'check': r['check']}, 'value', headline=r['check'] + ' value: ' + r['statement'])}" if r.get("value") is not None else "")
                 + f" ({r.get('threshold_text', '')}): **{c('m3.' + r['check'].lower() + '.verdict', 'T_healing_p3', {'check': r['check']}, 'verdict', headline=r['check'] + ' verdict' if r['check'] in ('P3', 'G2') else None)}**"
                 + (f"; {r['note']}" if r.get("note") else "") + ".")
    t2 = T.get("T_tier2")
    if t2 is not None and any(r["family"] == "healed" for r in t2.rows):
        bits = []
        for r in [r for r in t2.rows if r["family"] == "healed"]:
            wh = {"model": r["model"], "variant": r["variant"]}
            ms = MODEL_SHORT.get(r["model"], r["model"])
            bits.append(f"{r['method']} {c(f't2.{ms}.' + r['variant'] + '.top1', 'T_tier2', wh, 'top1')}")
        mrow = next((r for r in t2.rows if r["family"] == "healed" and r["method"] == "mirror"), None)
        extra = ""
        if mrow is not None:
            wh = {"model": mrow["model"], "variant": mrow["variant"]}
            ms = MODEL_SHORT.get(mrow["model"], mrow["model"])
            parts = []
            for base in ("svd", "random"):
                if mrow.get(f"margin_{base}") is not None:
                    parts.append(f"over {base} {c(f't2.{ms}.' + mrow['variant'] + '.margin_' + base, 'T_tier2', wh, 'margin_' + base)}")
            extra = "; Mirror's margin " + ", ".join(parts) if parts else ""
        L.append("- Second measure, Tier-2 ImageNet linear probe of the healed students (information only; the P3 "
                 "verdict above stays on k-NN, as pre-registered): top-1 " + ", ".join(bits) + extra
                 + ". Columns `linear_*` and `mirror_minus_*` in `T_healing`; rows in `T_tier2`.")
    # V-exact healing (scripts/heal_vexact.py), companion table T_healing_vexact
    tv, vxi = T["T_healing_vexact"], ctx["vx_info"]
    if tv.rows:
        L.append(f"- V-exact healing (scripts/heal_vexact.py, configs/m3_vexact.yaml; `{vxi['dir']}/k<k>/`; planned k = "
                 f"{', '.join(map(str, vxi['planned'])) or 'none found'}"
                 + (f" in `{vxi['plan_source']}`" if vxi["plan_source"] else "") + "). Table `T_healing_vexact`. Its "
                 "targets are symmetrised, so its held-out losses are not comparable with the pilot's.")
    for r in tv.rows:
        k = r["k"]
        wh = {"k": k}
        key = f"m3vx.k{k}"
        lab = f"k = {k}" + (" (full conversion)" if r.get("full_conversion") else "")
        if r["status"] == NOT_RUN:
            rs = reasons(["V-exact healing"])
            L.append(f"  - {lab}: {NOT_RUN} ({r.get('note', '')})" + (" | " + " | ".join(rs) if rs else "") + ".")
            continue
        bits = [f"status {r['status']}"]
        if r.get("healed_top1") is not None:
            hl = None if r.get("tiny") else f"V-exact healed top-1, k = {k}"
            bits.append(f"top-1 zero-shot {c(key + '.zero_shot', 'T_healing_vexact', wh, 'zero_shot_top1')} to healed "
                        f"{c(key + '.healed', 'T_healing_vexact', wh, 'healed_top1', headline=hl)} (original "
                        f"{c(key + '.original', 'T_healing_vexact', wh, 'original_top1')}); flip consistency (k-NN, "
                        f"{r.get('knn_precision') or 'fp32'}) {c(key + '.healed_flip', 'T_healing_vexact', wh, 'healed_flip')}")
        if r.get("flip_fp32") is not None:
            hl = None if r.get("tiny") else f"V-exact healed fp32 flip check, k = {k}"
            bits.append(f"fp32 flip check {c(key + '.flip_fp32', 'T_healing_vexact', wh, 'flip_fp32', headline=hl)}% "
                        f"(pass: {c(key + '.flip_fp32_pass', 'T_healing_vexact', wh, 'flip_fp32_pass', '')})")
        elif r.get("full_conversion"):
            bits.append("fp32 flip check: " + PENDING)
        if r.get("exact_max_rel") is not None:
            bits.append(f"exactness after healing {c(key + '.exact', 'T_healing_vexact', wh, 'exact_max_rel', '.1e')} "
                        f"(tolerance {c(key + '.exact_tol', 'T_healing_vexact', wh, 'exact_tol', 'g')}; pass: "
                        f"{c(key + '.exact_pass', 'T_healing_vexact', wh, 'exact_pass', '')})")
        if r.get("chosen_lr") is not None:
            bits.append(f"chosen LR {c(key + '.lr', 'T_healing_vexact', wh, 'chosen_lr', 'g')}")
        if r.get("heldout_loss_before") is not None:
            bits.append(f"held-out loss (symmetrised targets) {c(key + '.heldout_before', 'T_healing_vexact', wh, 'heldout_loss_before', '.4f')} "
                        f"to {c(key + '.heldout_after', 'T_healing_vexact', wh, 'heldout_loss_after', '.4f')}")
        if r.get("kept_frac_converted") is not None:
            bits.append(f"kept {c(key + '.kept', 'T_healing_vexact', wh, 'kept_frac_converted', '.3f')} of the converted "
                        f"matrix parameters ({c(key + '.model_params_frac', 'T_healing_vexact', wh, 'model_params_frac', '.3f')} "
                        "of the whole model)")
        if r.get("m1_zero_shot_top1") is not None:
            bits.append(f"M1 zero-shot at the same k {c(key + '.m1_zero_shot', 'T_healing_vexact', wh, 'm1_zero_shot_top1')}")
        L.append(f"  - {lab}: " + "; ".join(bits) + f". Seed {r.get('seed')}, git {short(r.get('git_hash'))}"
                 f"{' (dirty)' if r.get('git_dirty') else ''}; files `{r['source']}`.")
    if not tv.rows:
        rs = reasons(["V-exact healing"])
        L.append(f"- V-exact healing: {NOT_RUN} (no `runs/m3_vexact/k<k>/` and no planned job found)"
                 + (" | " + " | ".join(rs) if rs else "") + ".")
    rs = reasons(["M3 grid"])
    L.append("- M3 grid: " + NOT_RUN + (" (" + " | ".join(rs) + ")" if rs else "") + ". Tables `T_healing`, `T_healing_p3`, "
             "`T_healing_vexact`; figure `fig_healing_curves.png`.")
    L.append("")
    # ---------------------------------------------------------------- E3
    L += ["## E3: matched healed baselines at full conversion (PREREG_E3.md, gate G5)", ""]
    e3i = ctx["e3_info"]
    th3 = e3i["prereg"]
    te3, tec = T["T_e3"], T["T_e3_checks"]
    xl = lr_txt(th3.get("extra_lr"))
    if not th3.get("exists"):
        L.append(f"- {NOT_RUN}: no `PREREG_E3.md`.")
    else:
        L.append(f"- Pre-registration: PREREG_E3.md{' (ACTIVE, signed off)' if th3.get('active') else ' (not marked ACTIVE)'}, "
                 f"record commit {short(th3.get('commit'))} ({th3.get('commit_source')}). Bars read from its text: E3.a "
                 f"V0 - R >= {fmt(th3.get('bar_a'), '.1f')}, E3.b V0 - S >= {fmt(th3.get('bar_b'), '.1f')} with S the better "
                 "SVD arm, each difference also larger than the seed difference s = |V0 - V1|; gate G5 needs both. "
                 f"k = {fmt(e3i['k'], 'd')} ({e3i['k_source']}). Tables `T_e3` (arms), `T_e3_checks` (checks, validity, "
                 "learning-rate check); rows E3.* and G5 also in `T_prereg`.")
        for r in te3.rows:
            key, wh = f"e3.{r['key']}", {"arm": r["arm"]}
            if r["status"] == NOT_RUN:
                L.append(f"- Arm {r['arm']}, {r['label']}: {NOT_RUN} ({r.get('note', '')}).")
                continue
            bits = [f"status {r['status']}"]
            if r.get("healed_top1") is not None:
                hl = (None if r.get("tiny") or r["key"] == "v0_a100"
                      else f"E3 arm {r['arm']}: healed k-NN top-1 (k = {fmt(e3i['k'], 'd')})")
                bits.append(f"k-NN zero-shot {c(key + '.zero_shot', 'T_e3', wh, 'zero_shot_top1')} to healed "
                            f"{c(key + '.healed', 'T_e3', wh, 'healed_top1', headline=hl)} (original "
                            f"{c(key + '.original', 'T_e3', wh, 'original_top1')}); flip consistency (k-NN) "
                            f"{c(key + '.flip', 'T_e3', wh, 'healed_flip')}")
            if r.get("flip_fp32") is not None:
                bits.append(f"fp32 flip check {c(key + '.flip_fp32', 'T_e3', wh, 'flip_fp32')}%")
            if r.get("linear_top1") is not None:
                bits.append(f"linear probe {c(key + '.linear', 'T_e3', wh, 'linear_top1')}")
            elif r.get("linear_note"):
                bits.append(f"linear probe: {r['linear_note']}")
            if r.get("converted_params_kept") is not None:
                bits.append(f"matrix parameters kept {c(key + '.kept', 'T_e3', wh, 'converted_params_kept', ',d')}"
                            + (f" ({c(key + '.kept_vs_v0', 'T_e3', wh, 'kept_vs_vexact', '.4f')} of arm 1's)"
                               if r.get("kept_vs_vexact") is not None else ""))
            if r.get("chosen_lr") is not None:
                bits.append(f"chosen LR {c(key + '.lr', 'T_e3', wh, 'chosen_lr', 'g')}")
            if r.get("extra_would_win") is not None:
                bits.append(f"{xl} would have won: {fmt(r['extra_would_win'])}")
            if r.get("validity"):
                bits.append(f"validity {r['validity']}")
            L.append(f"- Arm {r['arm']}, {r['label']}: " + "; ".join(bits) + f". Seed {r.get('seed')}, git "
                     f"{short(r.get('git_hash'))}{' (dirty)' if r.get('git_dirty') else ''}; files `{r.get('source')}`.")
        for cid, spec in (("E3.a", "+.3f"), ("E3.b", "+.3f"), ("E3.seed", ".3f"), ("G5", "")):
            r = tec.find(id=cid)
            if r is None:
                continue
            k_ = "e3." + cid.lower().replace("e3.", "")
            hv = f"E3 {cid}: {r['statement']}" if cid in ("E3.a", "E3.b") else None
            val = (f", value {c(k_ + '.value', 'T_e3_checks', {'id': cid}, 'value', spec, headline=hv)}"
                   if r.get("value") is not None else "")
            hd = "Gate G5 verdict (E3)" if cid == "G5" else None
            L.append(f"- {cid}: {r['statement']}{val} ({r.get('threshold', '')}): **"
                     f"{c(k_ + '.verdict', 'T_e3_checks', {'id': cid}, 'verdict', '', headline=hd)}**"
                     + (f"; {r['note']}" if r.get("note") else "") + ".")
        alt = [r for r in tec.rows if r["id"].endswith(("(V1)", "(mean)"))]
        if alt and any(r.get("value") is not None for r in alt):
            bits = []
            for r in alt:
                if r.get("value") is None:
                    continue
                kk = "e3." + r["id"].lower().replace("e3.", "").replace(" (", ".").rstrip(")") + ".value"
                bits.append(f"{r['id']} {c(kk, 'T_e3_checks', {'id': r['id']}, 'value', '+.3f')} ({r['verdict']})")
            L.append("- Reported without changing the verdict: " + "; ".join(bits) + ".")
        r = tec.find(id="E3.lr")
        if r is not None:
            L.append(f"- E3.lr (Section 8): {r['statement']}: **{r['verdict']}**" + (f"; {r['note']}" if r.get("note") else "") + ".")
        vr = [r for r in tec.rows if r["id"].startswith("E3.valid.")]
        if vr:
            L.append("- Validity (Section 7): " + "; ".join(f"arm {r['id'].replace('E3.valid.', '')} {r['verdict']}" for r in vr)
                     + " (conditions in `T_e3_checks`).")
        rs = reasons(["E3"])
        if rs and any(r["status"] == NOT_RUN for r in te3.rows):
            L.append("- " + " | ".join(rs) + ".")
    L.append("")
    # ---------------------------------------------------------------- D
    L += ["## D: data-free readout (P5, gate G4)", ""]
    if ctx["d"]:
        j = ctx["d"]["json"]
        pv = j.get("provenance") or {}
        L.append(f"- Run: `{ctx['d']['file']}` (scripts/d_datafree.py), model {pv.get('model')}, seed {pv.get('seed')}, git "
                 f"{short((pv.get('git') or {}).get('git_hash'))}, {pv.get('date')}; operators "
                 f"`{(pv.get('operators') or {}).get('path')}`. Report `runs/d/D_DATAFREE_REPORT.md`.")
        for cid in ("P5.part1", "P5.part2", "P5.part2_strict"):
            r = T["T_prereg"].find(id=cid)
            if r:
                L.append(f"- {cid}: {r['statement']}: {c(cid.lower().replace('.', '_') + '.value', 'T_prereg', {'id': cid}, 'value', '.3f', headline=f'{cid} value')} "
                         f"({r['threshold']}): **{c(cid.lower().replace('.', '_') + '.verdict', 'T_prereg', {'id': cid}, 'verdict')}**.")
        L.append(f"- P5: **{c('p5.verdict', 'T_prereg', {'id': 'P5'}, 'verdict')}**; gate G4: **{c('g4.verdict', 'T_prereg', {'id': 'G4'}, 'verdict')}** "
                 "(claim C6 is dropped unless Dawar decides otherwise, as the run report says).")
    else:
        L.append(f"- {NOT_RUN}: no `runs/d/d_datafree.json`.")
    L.append("")
    # ---------------------------------------------------------------- A, F4, M2, Tier 2
    plan_text = ctx.get("plan_text") or ""
    L += ["## A: zero-shot ablations (A1 to A5)", ""]
    ta, tas, tax = T["T_ablations"], T["T_ablations_seeds"], T["T_ablations_a2_extra"]
    if ta.rows:
        hashes = sorted({short(r["_git"]) for r in ta.rows})
        L.append(f"- Runs: {', '.join('`' + P.rel(f) + '`' for f in ctx['a_files'])} (scripts/stage1_ablate.py); "
                 f"git {', '.join(hashes)}; run seed(s) {', '.join(sorted({str(r['_seed']) for r in ta.rows}))}. "
                 "Tables `T_ablations_seeds` (mean over fit seeds), `T_ablations` (per seed), `T_ablations_a2_extra`.")
        for abl in sorted({r["ablation"] for r in tas.rows}):
            for model in sorted({r["model"] for r in tas.rows if r["ablation"] == abl},
                                key=lambda m_: MODEL_ORDER.index(m_) if m_ in MODEL_ORDER else 99):
                for k in sorted({r["k"] for r in tas.rows if r["ablation"] == abl and r["model"] == model}):
                    rows = [r for r in tas.rows if r["ablation"] == abl and r["model"] == model and r["k"] == k]
                    bits = []
                    for r in rows:
                        ms = MODEL_SHORT.get(r["model"], r["model"])
                        key = f"a.{ms}.{abl}.{r['variant']}.k{r['k']}"
                        wh = {"model": r["model"], "ablation": abl, "variant": r["variant"], "k": r["k"]}
                        b = f"{r['variant']} {c(key + '.top1', 'T_ablations_seeds', wh, 'top1_mean')}"
                        if r.get("top1_std") is not None:
                            b += f" ± {c(key + '.top1_std', 'T_ablations_seeds', wh, 'top1_std')}"
                        par = ""
                        if r.get("params_vs_first_k_mean") is not None:
                            par = (f"; params {c(key + '.params_vs_default', 'T_ablations_seeds', wh, 'params_vs_first_k_mean', '.2f')}x"
                                   + ("" if r.get("params_matched") else ", not matched"))
                        bits.append(b + f" (n = {r['n_seeds']}, fit seeds {r['fit_seeds']}{par})")
                    L.append(f"- {abl} ({MODEL_LABEL.get(model, model)}, k = {k}), k-NN top-1 (%), mean ± standard "
                             "deviation over fit seeds: " + "; ".join(bits) + ".")
        if tax.rows:
            bits = []
            for r in tax.rows:
                rv, rf, rk = r["variant"], r["fit_seed"], r["k"]
                ms = MODEL_SHORT.get(r["model"], r["model"])
                wh = {"model": r["model"], "variant": r["variant"], "k": r["k"], "fit_seed": r["fit_seed"]}
                bits.append(f"{r['variant']} (k = {rk}, fit seed {r['fit_seed']}) "
                            f"{c(f'a.{ms}.A2x.{rv}.k{rk}.s{rf}.top1', 'T_ablations_a2_extra', wh, 'top1')}"
                            f" [{r['pairing']}{'; projection: ' + r['projection'] if r.get('projection') else ''}; exact "
                            f"projection: {fmt(r.get('params_exact'))}]")
            L.append("- A2 pairings hungarian, hungarian_cycle and generic, with the pairing and projection each row records: " + "; ".join(bits) + ".")
        L.append("- Params: MLP weights kept over those of the default setting at the same k and fit seed "
                 "(params_vs_mirror_first_k); \"not matched\" means outside the plan's 1% rule. See the notes of "
                 "`T_ablations_seeds` for what each unmatched comparison means.")
        for r in [r for r in tas.rows if (r["ablation"], r["variant"]) == ("A1", "per_layer") and r.get("canonical_top1_mean") is not None]:
            ms, kk = MODEL_SHORT.get(r["model"], r["model"]), r["k"]
            wh = {"model": r["model"], "ablation": "A1", "variant": "per_layer", "k": r["k"]}
            L.append(f"- Canonical number ({MODEL_LABEL.get(r['model'], r['model'])}, k = {r['k']}): the default cell (A1 "
                     f"per_layer = A2 mirror = A4 first = A5 alpha1.00) is the setting of {LABEL_MIRROR_MLP} in "
                     f"`T_main_zeroshot`; quote {CANONICAL_RUN} "
                     f"{c(f'a.{ms}.A1.per_layer.k{kk}.canonical', 'T_ablations_seeds', wh, 'canonical_top1_mean')} "
                     f"over fit seeds {r['fit_seeds']}, not the ablation script's "
                     f"{c(f'a.{ms}.A1.per_layer.k{kk}.top1', 'T_ablations_seeds', wh, 'top1_mean')} "
                     f"(mean difference {c(f'a.{ms}.A1.per_layer.k{kk}.minus_canonical', 'T_ablations_seeds', wh, 'minus_canonical_mean', '+.3f')}). "
                     "The ablation script reloads the operators from their float32 file and re-cleans them; X2 projects "
                     "with the float64 operators as fitted. Same pairing, parameter count and evaluation code (A1 svd, "
                     "which uses no operator, is identical in both), so the difference is numerical.")
        for r in [r for r in tas.rows if r["ablation"] == "A4" and r["variant"] == "last" and r.get("params_matched") is False]:
            ms, kk = MODEL_SHORT.get(r["model"], r["model"]), r["k"]
            wh = {"model": r["model"], "ablation": "A4", "variant": "last", "k": r["k"]}
            wf = {"model": r["model"], "ablation": "A4", "variant": "first", "k": r["k"]}
            same_bl = "lowest breaking share" if any("A4/last" in str(x.get("note")) for x in tas.rows
                                                      if x["ablation"] == "A4" and x["variant"] == "lowest_share"
                                                      and x["k"] == r["k"] and x["model"] == r["model"]) else None
            L.append(f"- A4 is not parameter-matched ({MODEL_LABEL.get(r['model'], r['model'])}, k = {r['k']}): last k keeps "
                     f"{c(f'a.{ms}.A4.last.k{kk}.mlp_kept', 'T_ablations_seeds', wh, 'mlp_kept_frac_mean', '.2f')} of "
                     f"its MLP weights against "
                     + (f"{c(f'a.{ms}.A4.first.k{kk}.mlp_kept', 'T_ablations_seeds', wf, 'mlp_kept_frac_mean', '.2f')} "
                        if tas.find(**wf) is not None else "fewer ")
                     + "for first k, so \"last k beats first k\" partly reflects more weights kept"
                     + (f"; {same_bl} selects the same blocks as last (the same model)" if same_bl else "") + ".")
        more = [r for r in tas.rows if r["ablation"] in ("A1", "A2") and r.get("params_matched") is False
                and (r.get("params_vs_first_k_mean") or 0) > 1]
        if more:
            bits = []
            for r in more:
                ms = MODEL_SHORT.get(r["model"], r["model"])
                key = f"a.{ms}.{r['ablation']}.{r['variant']}.k{r['k']}"
                wh = {"model": r["model"], "ablation": r["ablation"], "variant": r["variant"], "k": r["k"]}
                bits.append(f"{r['ablation']} {r['variant']} (k = {r['k']}) "
                            f"{c(key + '.params_vs_default', 'T_ablations_seeds', wh, 'params_vs_first_k_mean', '.2f')}x, "
                            f"top-1 {c(key + '.top1', 'T_ablations_seeds', wh, 'top1_mean')}")
            L.append("- A1 and A2 alternatives that keep more MLP weights than the default and still collapse: "
                     + "; ".join(bits) + ". Extra weights do not rescue a wrong operator or pairing.")
        reused = [r for r in ta.rows if r["reused_from"]]
        if reused:
            L.append("- Reused cells (same model as another cell, so its result was reused rather than evaluated again): "
                     + "; ".join(f"{r['ablation']}/{r['variant']}/k{r['k']}/s{r['fit_seed']} = {r['reused_from']}"
                                 for r in reused) + ".")
        pooled = [r for r in ta.rows if r["fit_seed"] == -1]
        for r in pooled:
            L.append(f"- {r['ablation']} {r['variant']}: fit seed -1 means pooled fit images ({r.get('fit_rows') or 'see info'}).")
        m = re.search(r"Track A: zero-shot ablations \(k-NN, k = (\d+) and (\d+), (\d+) seeds", plan_text)
        if m:
            want_k, want_n = {int(m[1]), int(m[2])}, int(m[3])
            gaps = []
            for abl in sorted({r["ablation"] for r in ta.rows}):
                rr = [r for r in ta.rows if r["ablation"] == abl]
                ks = {r["k"] for r in rr}
                per_k = {k: sorted({r["fit_seed"] for r in rr if r["k"] == k and r["fit_seed"] is not None
                                    and r["fit_seed"] >= 0}) for k in sorted(ks | want_k)}
                if not want_k <= ks or any(len(per_k[k]) < want_n for k in want_k):
                    gaps.append(f"{abl} has " + "; ".join(f"k = {k}: fit seeds {', '.join(map(str, s)) or 'none'}"
                                                          for k, s in per_k.items()))
            if gaps:
                L.append(f"- Not yet run against the plan (k = {m[1]} and {m[2]}, {m[3]} seeds): " + "; ".join(gaps) + ".")
    else:
        rs = reasons(["A1 to A5", "A1/A2", "A1"])
        L.append(f"- {NOT_RUN}: no `runs/a/*/A*_s*.csv`." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("- A6 and A7: zero-shot from the M1 rows of `T_main_zeroshot`; latency from M2 (`T_latency`).")
    L.append("")
    L += ["## SB: strong compression baselines, zero-shot", ""]
    tsb = T["T_strong_baselines"]
    if tsb.rows:
        sbg = sorted({short(r["_git"]) for r in tsb.rows if r.get("experiment") == "SB"})
        L.append(f"- Runs: {', '.join('`' + P.rel(f) + '`' for f in ctx['sb_files'])} (scripts/strong_baselines.py, "
                 f"mirror/baselines.py), git {', '.join(sbg)}. Each method compresses the MLPs of blocks 1 to k at the MLP "
                 f"weight count of {LABEL_MIRROR_MLP} at the same k and fit seed; {LABEL_MIRROR_MLP} and SVD are the "
                 f"canonical {CANONICAL_RUN} values. Table `T_strong_baselines`.")
        for (model, k, fs) in dict.fromkeys((r["model"], r["k"], r["fit_seed"]) for r in tsb.rows):
            ms = MODEL_SHORT.get(model, model)
            bits = []
            for r in [r for r in tsb.rows if (r["model"], r["k"], r["fit_seed"]) == (model, k, fs)]:
                wh = {"model": model, "k": k, "fit_seed": fs, "method": r["method"]}
                key = f"sb.{ms}.k{k}.s{fs}.{r['method']}"
                if r.get("top1") is None:
                    bits.append(f"{r['label']}: {r.get('status', NOT_RUN)}")
                    continue
                hl = f"Strong baseline {r['label']}, k = {k}: k-NN top-1" if r.get("experiment") == "SB" else None
                b = f"{r['label']} ({r['experiment']}) {c(key + '.top1', 'T_strong_baselines', wh, 'top1', headline=hl)}"
                if r.get("experiment") == "SB" and r.get("ratio_to_mirror") is not None:
                    b += (f" at {c(key + '.ratio', 'T_strong_baselines', wh, 'ratio_to_mirror', '.4f')} of the Mirror budget"
                          + ("" if r.get("within_1pct") else ", NOT within 1%"))
                if r.get("mirror_minus_this") is not None:
                    b += f", Mirror minus this {c(key + '.mirror_minus', 'T_strong_baselines', wh, 'mirror_minus_this', '+.2f')}"
                bits.append(b)
            L.append(f"- {MODEL_LABEL.get(model, model)}, k = {k}, fit seed {fs}, k-NN top-1 (%): " + "; ".join(bits) + ".")
        sl = [r for r in tsb.rows if r["method"] == "slicegpt" and r.get("basis_params_not_counted")]
        if sl:
            L.append("- Not counted in any MLP budget: SliceGPT-style slicing's change of stream basis between blocks ("
                     + ", ".join(f"k = {r['k']}: {fmt(r['basis_params_not_counted'], ',d')}" for r in sl)
                     + " parameters, d x d per block), as Mirror V-local's basis rotations are not counted.")
    else:
        rs = reasons(["Strong baselines", "strong baselines"])
        L.append(f"- {NOT_RUN}: no `runs/sb/SB_s*.csv`." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    L += ["## F4: symmetry versus low rank (P2)", ""]
    tf, tp = T["T_f4"], T["T_f4_p2"]
    if tf.rows or ctx["f4"]:
        models = sorted({r["model"] for r in tf.rows})
        hashes = sorted({short(r["_git"]) for r in tf.rows})
        L.append(f"- Runs: {', '.join('`' + P.rel(f) + '`' for f in ctx['f4_files']) or 'no per-block CSV'} "
                 f"(scripts/f4_rule.py), git {', '.join(hashes) or 'n/a'}; blocks per model: "
                 + ", ".join(f"{MODEL_LABEL.get(m_, m_)} {sum(1 for r in tf.rows if r['model'] == m_)}" for m_ in models) + ".")
        r = T["T_prereg"].find(id="P2")
        if r is not None and r["verdict"] != NOT_RUN:
            L.append(f"- P2 (pre-registered, {r['threshold']}): value {c('p2.value', 'T_prereg', {'id': 'P2'}, 'value', '.3f', headline='P2 fraction of cases the rule gets right')}: "
                     f"**{c('p2.verdict', 'T_prereg', {'id': 'P2'}, 'verdict', headline='P2 verdict')}** ({r.get('note', '')}).")
        for row in tp.rows:
            sc = row["scope"]
            key = "f4." + re.sub(r"[^a-z0-9]+", "_", sc.lower()).strip("_")
            L.append(f"- {sc}: fraction {c(key + '.fraction', 'T_f4_p2', {'scope': sc}, 'fraction', '.3f')} "
                     f"({c(key + '.hits', 'T_f4_p2', {'scope': sc}, 'n_hits', 'd')} of "
                     f"{c(key + '.cases', 'T_f4_p2', {'scope': sc}, 'n_cases', 'd')} cases).")
        if (Path(ctx["figs_dir"]) / "ext_f4_fig4_rule.png").is_file():
            L.append("- Figure `figures/ext_f4_fig4_rule.png` (copied from `runs/f4/plots/`). Tables `T_f4`, `T_f4_p2`.")
        else:
            L.append("- Tables `T_f4`, `T_f4_p2`.")
    else:
        rs = reasons(["F4"])
        L.append(f"- {NOT_RUN}: no `runs/f4/f4_*.csv` or `runs/f4/f4_p2.json`; P2 cannot be evaluated."
                 + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    L += ["## M2: efficient implementation and latency", ""]
    tl, tck = T["T_latency"], T["T_latency_checks"]
    if tl.rows:
        mm = ctx.get("m2_meta") or {}
        L.append(f"- Run: `runs/{P.m2}/` (scripts/m2_latency.py): model {mm.get('model')}, seed {mm.get('seed')}, git "
                 f"{short((mm.get('git') or {}).get('git_hash'))}, {mm.get('date')}, device "
                 f"{(mm.get('env') or {}).get('gpu') or (mm.get('env') or {}).get('device')}; timing settings in "
                 f"`runs/{P.m2}/m2_meta.json` (warm-up and timed runs, compile mode). Summary `runs/{P.m2}/M2_SUMMARY.md`."
                 + (" The first M2 run (`runs/m2/`) was timed while other jobs shared the GPU and is kept for the "
                    "record only." if P.m2 != "m2" else ""))
        if tck.rows:
            n_ok = R.put("m2.checks.n_pass", sum(1 for r in tck.rows if r.get("pass")), "d", source=f"runs/{P.m2}/m2_checks.csv",
                         git_hash=tck.rows[0].get("_git"), seed=tck.rows[0].get("_seed"),
                         derived="number of rows with pass = True in m2_checks.csv")
            fails = [f"{r['variant']} k = {r['k']} {r['dtype']}" for r in tck.rows if not r.get("pass")]
            L.append(f"- Checks (efficient model against the dense model it was built from): {n_ok} of {len(tck.rows)} pass"
                     + (f"; failing: {', '.join(fails)}" if fails else "") + ". Table `T_latency_checks`.")
        modes = sorted({r["mode"] for r in tl.rows})
        mode = "compile" if "compile" in modes else modes[0]
        rows_m = [r for r in tl.rows if r["mode"] == mode]
        batches = sorted({r["batch"] for r in rows_m})
        show = sorted({batches[0], batches[-1]}) if batches else []
        for b in show:
            if b == batches[0] and len(show) > 1 and b != 1:
                continue
            for k in sorted({r["k"] for r in rows_m if r["k"]}):
                bits = []
                for r in [r for r in rows_m if r["batch"] == b and r["k"] == k]:
                    v = r["variant"]
                    wh = {"model": r["model"], "variant": v, "k": k, "batch": b, "mode": mode}
                    key = f"m2.{v}.k{k}.b{b}.{mode}"
                    hl = (f"M2 {v} k = {k}, batch {b}: measured speed-up vs dense"
                          if v in ("vlocal", "vexact") and b == batches[-1] else None)
                    bits.append(f"{v} measured {c(key + '.speedup', 'T_latency', wh, 'speedup_vs_dense', '.3f', headline=hl)}x, "
                                f"analytic {c(key + '.flop', 'T_latency', wh, 'flop_speedup_vs_dense', '.3f')}x (M2 count)"
                                + (f", {c(key + '.flop_collector', 'T_latency', wh, 'collector_flop_speedup', '.3f')}x "
                                   "(collector)" if r.get("collector_flop_speedup") is not None else ""))
                if bits:
                    L.append(f"- k = {k}, batch {b}, {mode}: speed-up against dense, measured vs analytic: "
                             + "; ".join(bits) + ".")
        vl = [r for r in rows_m if r["variant"] == "vlocal" and r["batch"] == (batches[-1] if batches else None)
              and r.get("wo_dense_extra_gflops") is not None]
        if vl:
            bits = []
            for r in sorted(vl, key=lambda r: r["k"]):
                wh = {"model": r["model"], "variant": "vlocal", "k": r["k"], "batch": r["batch"], "mode": mode}
                key = f"m2.vlocal.k{r['k']}.b{r['batch']}.{mode}"
                bits.append(f"k = {r['k']}: {c(key + '.gflops', 'T_latency', wh, 'gflops', '.2f')} against "
                            f"{c(key + '.gflops_collector', 'T_latency', wh, 'gflops_collector', '.2f')} GFLOPs, of which "
                            f"{c(key + '.wo_extra', 'T_latency', wh, 'wo_dense_extra_gflops', '.3f')} is the dense Wo")
            L.append("- Two analytic speed-ups: measured / analytic uses the M2 count (the implementation that was timed); "
                     "the collector count (`tables/cost_model.md`, used in `T_main_zeroshot`) is the structured one. They "
                     "agree except for V-local, whose timed implementation stores Wo dense in the block's basis (M1 projected "
                     "Wo with A_(l+1); a structured Wo would need a second rotation per block, not run). Basis rotations are "
                     "counted alike in both. " + "; ".join(bits) + " (the rest is one rotation of the CLS feature).")
        flagged = [r for r in tl.rows if r.get("timing_flag")]
        for r in flagged:
            wh = {"model": r["model"], "variant": r["variant"], "k": r["k"], "batch": r["batch"], "mode": r["mode"]}
            key = f"m2.{r['variant']}.k{r['k']}.b{r['batch']}.{r['mode']}"
            others = sorted((x for x in tl.rows if x["variant"] == r["variant"] and x["k"] == r["k"]
                             and x["mode"] == r["mode"] and x is not r), key=lambda x: x["batch"])
            ob = []
            for x in others:
                whx = {"model": x["model"], "variant": x["variant"], "k": x["k"], "batch": x["batch"], "mode": x["mode"]}
                kx = f"m2.{x['variant']}.k{x['k']}.b{x['batch']}.{x['mode']}.speedup"
                ob.append(f"{c(kx, 'T_latency', whx, 'speedup_vs_dense', '.3f')}x at batch {x['batch']}")
            L.append(f"- Probable timing outlier (kept, flagged in `T_latency`): {r['label']} k = {r['k']}, batch {r['batch']}, "
                     f"{r['mode']} measures {c(key + '.speedup', 'T_latency', wh, 'speedup_vs_dense', '.3f')}x, above its "
                     f"analytic {c(key + '.flop', 'T_latency', wh, 'flop_speedup_vs_dense', '.3f')}x; the same variant "
                     f"measures {', '.join(ob)}.")
        o = [r for r in rows_m if r["variant"] == "original" and r["batch"] == (batches[-1] if batches else None)]
        if o:
            r = o[0]
            wh = {"model": r["model"], "variant": "original", "k": 0, "batch": r["batch"], "mode": mode}
            L.append(f"- Hub original against the dense reference (batch {r['batch']}, {mode}): speed-up "
                     f"{c('m2.original.speedup_vs_dense', 'T_latency', wh, 'speedup_vs_dense', '.3f')}x "
                     "(below 1 means the efficient implementation's dense model is faster than the hub model).")
        L.append("- Table `T_latency` (every variant, k, batch and mode, with measured / analytic).")
    else:
        rs = reasons(["M2"])
        L.append(f"- {NOT_RUN}: no `runs/{P.m2}/m2_latency.csv`." + (" " + " | ".join(rs) + "." if rs else "")
                 + " The cost columns of `T_main_zeroshot` are analytic counts from `tables/cost_model.md`, not measured "
                 "latency.")
    tl2, tdv = T["T_latency_5090"], T["T_latency_devices"]
    if tl2.rows:
        mb = ctx.get("m2b_meta") or {}
        ci = str(mb.get("check_images") or "not recorded")
        L.append(f"- Second device: `runs/{M2_SECOND}/` (scripts/m2_latency.py, same arguments as the primary run): model "
                 f"{mb.get('model')}, seed {mb.get('seed')}, git {short((mb.get('git') or {}).get('git_hash'))}, "
                 f"{mb.get('date')}, device {(mb.get('env') or {}).get('gpu') or (mb.get('env') or {}).get('device')}; "
                 f"equivalence check images: {ci}"
                 + (" (the validation cache could not be loaded on that machine, so scripts/m2_latency.py fell back to "
                    "random images; the checks test the implementation, not accuracy)" if ci.startswith("random") else "")
                 + ". Tables `T_latency_5090`, `T_latency_checks_5090`, `T_latency_devices`.")
        ck2 = T["T_latency_checks_5090"]
        if ck2.rows:
            n2 = R.put("m2_5090.checks.n_pass", sum(1 for r in ck2.rows if r.get("pass")), "d",
                       source=f"runs/{M2_SECOND}/m2_checks.csv", git_hash=ck2.rows[0].get("_git"), seed=ck2.rows[0].get("_seed"),
                       derived="number of rows with pass = True in m2_checks.csv")
            L.append(f"- Second device checks: {n2} of {len(ck2.rows)} pass (`T_latency_checks_5090`).")
        modes2 = sorted({r["mode"] for r in tdv.rows})
        if modes2:
            mode2 = "compile" if "compile" in modes2 else modes2[0]
            bmax = max(r["batch"] for r in tdv.rows if r["mode"] == mode2)
            for k in sorted({r["k"] for r in tdv.rows if r["k"] and r["mode"] == mode2 and r["batch"] == bmax}):
                bits = []
                for r in [r for r in tdv.rows if r["k"] == k and r["mode"] == mode2 and r["batch"] == bmax]:
                    wh = {"model": r["model"], "variant": r["variant"], "k": k, "batch": bmax, "mode": mode2}
                    key = f"m2dev.{r['variant']}.k{k}.b{bmax}.{mode2}"
                    bits.append(f"{r['variant']} {c(key + '.first', 'T_latency_devices', wh, 'speedup_1', '.3f')}x / "
                                f"{c(key + '.second', 'T_latency_devices', wh, 'speedup_2', '.3f')}x")
                L.append(f"- k = {k}, batch {bmax}, {mode2}: measured speed-up against dense, "
                         f"{tdv.rows[0]['device_1']} / {tdv.rows[0]['device_2']}: " + "; ".join(bits) + ".")
    else:
        L.append(f"- Second device (`runs/{M2_SECOND}/`): {NOT_RUN}.")
    # M2 profile (scripts/m2_profile.py): where the time goes
    tpf = T["T_latency_profile"]
    pm_ = ctx.get("prof_meta") or {}
    if tpf.rows:
        tiny = bool(pm_.get("tiny")) or pm_.get("device_type") == "cpu"
        src = sorted({str(r.get("time_source")) for r in tpf.rows if r.get("time_source")})
        L.append(f"- Profile (why the converted models are slower than dense): `runs/m2_profile/` (scripts/m2_profile.py), "
                 f"model {pm_.get('model')}, seed {pm_.get('seed')}, git {short((pm_.get('git') or {}).get('git_hash'))}"
                 f"{' (dirty)' if (pm_.get('git') or {}).get('git_dirty') else ''}, {pm_.get('date')}, device "
                 f"{(pm_.get('env') or {}).get('gpu') or (pm_.get('env') or {}).get('device') or pm_.get('device_type')}; "
                 f"batch {pm_.get('batch')}, modes {', '.join(pm_.get('modes') or [])}; time source: {'; '.join(src)}. "
                 "Report `runs/m2_profile/M2_PROFILE.md`; tables `T_latency_profile` (category breakdown, gap to dense by "
                 "category, sub-module split) and `T_latency_profile_triton`."
                 + (" TINY smoke test on CPU: operator self times, not a measurement." if tiny else ""))
        modes = sorted({r["mode"] for r in tpf.rows})
        mode = "compile" if "compile" in modes else modes[0]
        for r in [r for r in tpf.rows if r["mode"] == mode and r["variant"] not in ("dense", "original")]:
            if r.get("gap_ms") is None:
                continue
            ms = MODEL_SHORT.get(r["model"], r["model"])
            key = f"m2p.{ms}.{r['variant']}.k{r['k']}.b{r['batch']}.{mode}"
            wh = {"model": r["model"], "variant": r["variant"], "k": r["k"], "mode": mode, "batch": r["batch"]}
            parts = [f"{PROFILE_CAT_LABEL[cc]} {c(key + '.d_' + cc, 'T_latency_profile', wh, 'd_' + cc, '+.3f')}"
                     for cc in PROFILE_CATS if r.get(f"d_{cc}") is not None]
            if r.get("d_idle") is not None:
                parts.append(f"idle {c(key + '.d_idle', 'T_latency_profile', wh, 'd_idle', '+.3f')}")
            hl = None if tiny else f"M2 profile {r['variant']} k = {r['k']} ({mode}): latency gap to dense (ms)"
            b = (f"- {r['variant']} k = {r['k']} ({mode}): median {c(key + '.latency', 'T_latency_profile', wh, 'latency_ms_median', '.3f')} ms, "
                 f"gap to dense {c(key + '.gap', 'T_latency_profile', wh, 'gap_ms', '+.3f', headline=hl)} ms = "
                 + ", ".join(parts))
            if r.get("sub_sum_delta_ms") is not None:
                sp = [f"{lab} {c(key + '.' + p_ + '_delta', 'T_latency_profile', wh, p_ + '_delta_ms', '+.3f')}"
                      for p_, lab in (("mlp", "MLP"), ("attention", "attention"), ("rotation", "rotations"))
                      if r.get(f"{p_}_delta_ms") is not None]
                b += (f"; sub-modules (0-based blocks {r.get('sub_blocks')}): " + ", ".join(sp)
                      + f" (remainder {c(key + '.sub_remainder', 'T_latency_profile', wh, 'sub_remainder_ms', '+.3f')})")
            L.append(b + ".")
    else:
        rs = reasons(["M2 profile", "m2_profile"])
        L.append(f"- Profile (scripts/m2_profile.py): {NOT_RUN} (no `runs/m2_profile/m2_profile.csv`)."
                 + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    L += ["## Tier 2: linear probes, transfer, segmentation", ""]
    t2, t2x = T["T_tier2"], T["T_tier2_transfer"]
    if t2.rows:
        L.append(f"- ImageNet linear probe: `runs/t2/t2_linear.csv` and `runs/t2/probe/` (scripts/tier2_linear.py), git "
                 f"{', '.join(sorted({short(r['_git']) for r in t2.rows}))}, seed(s) "
                 f"{', '.join(sorted({str(r['_seed']) for r in t2.rows}))}. Table `T_tier2`.")
        for r in t2.rows:
            rv = r["variant"]
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "variant": r["variant"]}
            key = f"t2.{ms}.{r['variant']}"
            hl = f"Tier 2 linear probe top-1, {r['model_label']} {r['variant']}" if r["family"] in ("original", "healed") else None
            b = (f"- {r['model_label']} {r['variant']}: top-1 {c(key + '.top1', 'T_tier2', wh, 'top1', headline=hl)}, "
                 f"top-5 {c(key + '.top5', 'T_tier2', wh, 'top5')}")
            if r.get("drop_vs_original") is not None and r["family"] != "original":
                b += f", drop {c(key + '.drop', 'T_tier2', wh, 'drop_vs_original')}"
            for base in ("svd", "random"):
                if r.get(f"margin_{base}") is not None:
                    b += f", vs {base.upper() if base == 'svd' else base} {c(key + '.margin_' + base, 'T_tier2', wh, 'margin_' + base, headline=(f'Tier 2 linear: {rv} margin over {base}' if r['family'] == 'healed' else None))}"
            b += (f"; chosen wd {c(key + '.wd', 'T_tier2', wh, 'chosen_wd', 'g')}"
                  + (" (at the grid edge)" if r.get("wd_at_grid_edge") else "")
                  + f"; k-NN check {c(key + '.knn_check', 'T_tier2', wh, 'knn_check_top1')}")
            if r.get("tier1_knn_top1") is not None:
                b += f" (Tier-1 k-NN {c(key + '.tier1_knn', 'T_tier2', wh, 'tier1_knn_top1')}"
                if str(r.get("tier1_same_model", "")).startswith("no"):
                    b += ("; not the same model: " + ("a different random draw" if r.get("method") == "random"
                                                      else "built from the float32 operator file, a numerical difference")
                          + ", see the `T_tier2` notes")
                b += ")"
            L.append(b + ".")
    else:
        rs = reasons(["Tier-2", "Tier 2"])
        L.append(f"- ImageNet linear probe: {NOT_RUN} (no `runs/t2/t2_linear.csv`)." + (" " + " | ".join(rs) + "." if rs else ""))
    if t2x.rows:
        L.append(f"- Transfer probes: `runs/t2/t2_transfer.csv` (scripts/tier2_transfer.py). Table `T_tier2_transfer`.")
        for r in t2x.rows:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "dataset": r["dataset"], "variant": r["variant"]}
            key = f"t2x.{ms}.{r['dataset']}.{r['variant']}"
            b = (f"- {r['dataset']}, {r['model_label']} {r['variant']}: top-1 {c(key + '.top1', 'T_tier2_transfer', wh, 'top1')}, "
                 f"mean per-class {c(key + '.mpc', 'T_tier2_transfer', wh, 'mean_per_class_top1')}")
            if r.get("drop_vs_original") is not None and r["family"] != "original":
                b += f", drop {c(key + '.drop', 'T_tier2_transfer', wh, 'drop_vs_original')}"
            L.append(b + ".")
        for ds in sorted({r["dataset"] for r in t2x.rows if r.get("dataset_at_ceiling")}):
            o = next((r for r in t2x.rows if r["dataset"] == ds and r.get("family") == "original"), None)
            if o is not None:
                ms = MODEL_SHORT.get(o["model"], o["model"])
                L.append(f"- {ds} is at ceiling (original {c(f't2x.{ms}.{ds}.original.top1', 'T_tier2_transfer', {'model': o['model'], 'dataset': ds, 'variant': 'original'}, 'top1')}%; "
                         "several variants tie with it exactly and none differs by more than a few test images), so it "
                         "does not separate the methods; see the `T_tier2_transfer` notes.")
    else:
        L.append(f"- Transfer probes (CIFAR-100, Flowers-102, Pets, DTD): {NOT_RUN} (no `runs/t2/t2_transfer.csv`).")
    tsg, si = T["T_tier2_seg"], ctx["seg_info"]
    if tsg.rows:
        rr0 = tsg.rows[0]
        L.append(f"- ADE20K linear segmentation: `{si['csv']}` and `{si['dir']}/seg/<model>/<variant>.json` "
                 f"(scripts/tier2_ade20k.py), git {', '.join(sorted({short(r['_git']) for r in tsg.rows}))}"
                 f"{' (dirty)' if any(r.get('git_dirty') for r in tsg.rows) else ''}, seed(s) "
                 f"{', '.join(sorted({str(r['_seed']) for r in tsg.rows}))}; crop {fmt(rr0.get('crop'), 'd')} px, "
                 f"{', '.join(sorted({fmt(r.get('iterations'), ',d') for r in tsg.rows}))} iterations, batch "
                 f"{fmt(rr0.get('batch_size'), 'd')}; run meta "
                 + (", ".join(f"`{mm['file']}`" for mm in si["meta"].values()) or "not found") + ". Tables "
                 "`T_tier2_seg` and `T_tier2_seg_classes` (per-class IoU).")
        for r in tsg.rows:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "variant": r["variant"]}
            key = f"t2seg.{ms}.{r['variant']}"
            hl = (f"ADE20K mIoU, {r['model_label']} {r['variant']}"
                  if r.get("family") == "original" or (r.get("family"), r.get("method")) in T2_MIRROR else None)
            b = (f"- ADE20K, {r['model_label']} {r['variant']}: mIoU {c(key + '.miou', 'T_tier2_seg', wh, 'mIoU', headline=hl)}, "
                 f"aAcc {c(key + '.aacc', 'T_tier2_seg', wh, 'aAcc')}, mAcc {c(key + '.macc', 'T_tier2_seg', wh, 'mAcc')}")
            if r.get("delta_miou_vs_original") is not None:
                b += f", change vs original {c(key + '.delta', 'T_tier2_seg', wh, 'delta_miou_vs_original', '+.2f')}"
            for base in ("svd", "random"):
                if r.get(f"margin_{base}_miou") is not None:
                    b += f", vs {base.upper() if base == 'svd' else base} {c(key + '.margin_' + base, 'T_tier2_seg', wh, f'margin_{base}_miou', '+.2f')}"
            if r.get("model_params") is not None:
                b += f"; params {c(key + '.params', 'T_tier2_seg', wh, 'model_params', ',d')}"
            if r.get("vexact_max_rel_err") is not None:
                b += (f"; V-exact rebuilt at {fmt(r.get('vexact_built_px'), 'd')} px, largest relative error in the span "
                      f"{c(key + '.vexact_err', 'T_tier2_seg', wh, 'vexact_max_rel_err', '.1e')} (bar "
                      f"{c(key + '.vexact_bar', 'T_tier2_seg', wh, 'vexact_bar', 'g')})")
            if r.get("note"):
                b += f" ({r['note']})"
            L.append(b + ".")
        for model, miss in si["missing"].items():
            L.append(f"- ADE20K, {MODEL_LABEL.get(model, model)}: variants named in `{si['meta'][model]['file']}` without a "
                     f"row: {', '.join(miss)} ({NOT_RUN}).")
        for r in T["T_tier2_seg_rerun"].rows:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "variant": r["variant"]}
            key = f"t2segrerun.{ms}.{r['variant']}"
            if r.get("diff_mIoU") is not None:
                hl = f"ADE20K run-to-run difference, {r['variant']} (mIoU)"
                L.append(f"- ADE20K rerun (run-to-run spread), {r['model_label']} {r['variant']}: mIoU "
                         f"{c(key + '.first', 'T_tier2_seg_rerun', wh, 'first_mIoU', '.3f')} in `runs/t2_seg` and "
                         f"{c(key + '.rerun', 'T_tier2_seg_rerun', wh, 'rerun_mIoU', '.3f')} in `runs/t2_seg_rerun` "
                         f"(difference {c(key + '.diff', 'T_tier2_seg_rerun', wh, 'diff_mIoU', '+.3f', headline=hl)}). "
                         "Table `T_tier2_seg_rerun`.")
            else:
                L.append(f"- ADE20K rerun of {r['variant']} in `runs/t2_seg_rerun` has no first run in `runs/t2_seg`.")
    else:
        rs = reasons(["ADE20K"])
        L.append(f"- ADE20K linear segmentation: {NOT_RUN} (no `{si['csv']}`)." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    # ---------------------------------------------------------------- flip averaging
    L += ["## Flip averaging: exact invariance at test time against V-exact", ""]
    tfa = T["T_flipavg"]
    if tfa.rows:
        L.append(f"- Run: `runs/t1_flipavg/flipavg.csv` (scripts/flip_avg_baseline.py), git "
                 f"{', '.join(sorted({short(r['_git']) for r in tfa.rows}))}, seed(s) "
                 f"{', '.join(sorted({str(r['_seed']) for r in tfa.rows}))}. Table `T_flipavg`.")
        for r in tfa.rows:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "variant": r["variant"], "mode": r["mode"]}
            key = f"fa.{ms}.{r['variant']}.{r['mode']}"
            hl = (f"Flip averaging: {r['variant']} ({r['mode']}) k-NN top-1"
                  if r["model"] != "tiny" and r.get("exactly_invariant") else None)
            b = (f"- {r['model_label']} {r['variant']} ({r['mode']}): k-NN {c(key + '.knn', 'T_flipavg', wh, 'knn_top1', headline=hl)}, "
                 f"flip consistency {c(key + '.flip', 'T_flipavg', wh, 'flip_consistency')}%")
            if r.get("probe_top1") is not None:
                b += f", linear probe {c(key + '.probe', 'T_flipavg', wh, 'probe_top1')}"
            b += f"; {fmt(r.get('forwards_per_image'), 'd')} forward pass(es) per image"
            if r.get("gflops") is not None:
                b += f", {c(key + '.gflops', 'T_flipavg', wh, 'gflops', '.2f')} GFLOPs per image"
            if r.get("rel_cost") is not None:
                b += f" ({c(key + '.rel_cost', 'T_flipavg', wh, 'rel_cost', '.3f')} of the original's, {r.get('rel_cost_basis')})"
            if r.get("params") is not None:
                b += f"; {fmt(r['params'], ',d')} parameters ({fmt(r.get('params_frac'), '.3f')} of the original)"
            if r.get("note"):
                b += f" ({r['note']})"
            L.append(b + ".")
        for r in [r for r in tfa.rows if r.get("cost_vs_flipavg") is not None]:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "variant": r["variant"], "mode": r["mode"]}
            key = f"fa.{ms}.{r['variant']}.{r['mode']}"
            L.append(f"- {r['variant']} ({r['mode']}) against the flip-averaged original: "
                     f"{c(key + '.cost_vs_fa', 'T_flipavg', wh, 'cost_vs_flipavg', '.3f')} of its GFLOPs per image and "
                     f"{c(key + '.params_vs_fa', 'T_flipavg', wh, 'params_vs_flipavg', '.3f')} of its parameters; top-1 "
                     f"difference k-NN {c(key + '.minus_fa_knn', 'T_flipavg', wh, 'minus_flipavg_knn', '+.2f')}"
                     + (f", linear probe {c(key + '.minus_fa_probe', 'T_flipavg', wh, 'minus_flipavg_probe', '+.2f')}"
                        if r.get("minus_flipavg_probe") is not None else "") + " points.")
    else:
        rs = reasons(["flip-averaging", "flip averaging"])
        L.append(f"- {NOT_RUN}: no `runs/t1_flipavg/flipavg.csv`." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    # ---------------------------------------------------------------- F7
    L += ["## F7: visuals (teaser)", ""]
    tf7, f7i = T["T_f7"], ctx["f7_info"]
    fm = f7i.get("meta") or {}
    if tf7.rows or fm:
        env = fm.get("env") or {}
        im = fm.get("images") or {}
        L.append(f"- Run: `{f7i['dir']}/` (scripts/f7_visuals.py), model {fm.get('model')}, fit seed "
                 f"{(fm.get('fit_run') or {}).get('fit_seed')}, seed {fm.get('seed')}, git "
                 f"{short((fm.get('git') or {}).get('git_hash'))}, {fm.get('date')}, device {env.get('gpu') or env.get('device')}; "
                 f"{fmt(im.get('n'), ',d')} images ({im.get('description')}). Tables `T_f7`, `T_f7_layer0`; draft "
                 "captions in `runs/f7/F7_SUMMARY.md`." + (" TINY smoke test, not a result." if fm.get("tiny") else ""))
        for r in tf7.rows:
            ms = MODEL_SHORT.get(r["model"], r["model"])
            wh = {"model": r["model"], "block": r["block"]}
            key = f"f7.{ms}.b{r['block']}"
            L.append(f"- Block {r['block']}: twin correlation median {c(key + '.twin_r', 'T_f7', wh, 'twin_r_median', '.3f')} "
                     f"against {c(key + '.rand_r', 'T_f7', wh, 'rand_r_median', '.3f')} for a random partner, over "
                     f"{fmt(r.get('n_pairs'), ',d')} twin pairs; twin above random for "
                     f"{c(key + '.frac', 'T_f7', wh, 'frac_twin_gt_rand', '.3f')} of the paired neurons.")
        l0 = T["T_f7_layer0"]
        if l0.rows:
            bits = []
            for r in l0.rows:
                wh = {"operator": r["operator"], "parity": r["parity"]}
                key = f"f7.l0.{'fit' if 'fitted' in r['operator'] else 'null'}.{r['parity']}"
                bits.append(f"{r['operator']}, {r['parity']}: energy {c(key + '.energy', 'T_f7_layer0', wh, 'energy_share', '.3f')}, "
                            f"symmetric fraction {c(key + '.sym', 'T_f7_layer0', wh, 'sym_fraction', '.3f')}")
            L.append("- Layer-0 filters by parity: " + "; ".join(bits) + ".")
        figs = f7i.get("figures") or {}
        if figs:
            L.append("- Figures (drawn by scripts/f7_visuals.py, not by the collector): "
                     + ", ".join(f"`{d}/{stem}.pdf`" for stem, d in figs.items()) + ".")
        else:
            L.append("- Figures fig0_teaser, figF7_twins_full and figF7_layer0_filters are not on this machine; "
                     "`python scripts/f7_visuals.py --replot runs/f7` draws them into `results/paper/figures_paper/`.")
    else:
        rs = reasons(["F7"])
        L.append(f"- {NOT_RUN}: no `runs/f7/f7_blocks.csv`." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("")
    # ---------------------------------------------------------------- inputs
    L += ["## Inputs found and missing", ""]
    for item in ctx["inventory"]:
        L.append(f"- [{'x' if item['present'] else ' '}] `{item['path']}`: {item['what']}")
    L.append("")
    # ---------------------------------------------------------------- deviations
    L += ["## Deviations from the plan", ""]
    L.append("From `results/STATUS.md` (section \"Deviations from the plan\"):")
    for line in st["sections"].get("Deviations from the plan", []):
        if line.strip().startswith("-"):
            L.append(line.rstrip())
    L.append("")
    L.append("Detected by the collector from the result files:")
    for dv in ctx["deviations"]:
        L.append(f"- {dv}")
    L.append("")
    # ---------------------------------------------------------------- headline numbers
    L += ["## Numbers for the paper", ""]
    L.append("Headline numbers with their source table and row (all values also in `numbers.json`). Seeds: n is the "
             "number of fit seeds behind a value; the plan needs n = 3 for every headline.")
    L.append("")
    extra = [("main.reg.k9.vlocal.margin_svd", "T_main_zeroshot", {"model": "dinov2_vitb14_reg", "k": 9, "variant": "vlocal"},
              "margin_svd_mean", "DINOv2-reg k = 9 V-local margin over SVD (points)"),
             ("main.reg.k6.mirror_mlp.margin_svd", "T_main_zeroshot", {"model": "dinov2_vitb14_reg", "k": 6, "variant": "mirror_mlp"},
              "margin_svd_mean", f"DINOv2-reg k = 6 {LABEL_MIRROR_MLP} margin over SVD (points)"),
             ("main.plain.k6.mirror_mlp.margin_svd", "T_main_zeroshot", {"model": "dinov2_vitb14", "k": 6, "variant": "mirror_mlp"},
              "margin_svd_mean", f"DINOv2 k = 6 {LABEL_MIRROR_MLP} margin over SVD (points)"),
             ("main.reg.k6.mirror_mlp.top1_mean", "T_main_zeroshot", {"model": "dinov2_vitb14_reg", "k": 6, "variant": "mirror_mlp"},
              "top1_mean", f"DINOv2-reg k = 6 {LABEL_MIRROR_MLP} top-1 (headline; canonical source {CANONICAL_RUN})"),
             ("main.reg.k12.vexact.flip", "T_main_zeroshot", {"model": "dinov2_vitb14_reg", "k": 12, "variant": "vexact"},
              "flip_mean", "DINOv2-reg k = 12 V-exact flip consistency, bf16 (%)"),
             ("g.reg.s0.row4", "T_generality", {"model": "dinov2_vitb14_reg", "fit_seed": 0}, "row4_value",
              "DINOv2-reg breaking share (row 4, fit seed 0)"),
             ("g.plain.s0.row1", "T_generality", {"model": "dinov2_vitb14", "fit_seed": 0}, "row1_value",
              "DINOv2 e(null)/e(A), layers 1 to 6"),
             ("g.reg.s0.row1", "T_generality", {"model": "dinov2_vitb14_reg", "fit_seed": 0}, "row1_value",
              "DINOv2-reg e(null)/e(A), layers 1 to 6")]
    for key, tn, where, col, desc in extra:
        rr = T[tn].find(**where)
        if rr is not None and py(rr.get(col)) is not None:
            c(key, tn, where, col, ".3f", headline=desc)
    L.append("| key | number | value | table | row | n seeds |")
    L.append("|---|---|---|---|---|---|")
    for key, desc in R.headlines:
        e = R.items[key]
        fs = e.get("fit_seed")
        n = len(str(fs).split(",")) if fs is not None else ""
        L.append(f"| `{key}` | {desc} | {fmt(e['value'], '.4g')} | `{e['table']}` | {e['row']} | {n} |")
    L.append("")
    return L


# ================================================================================================
# Main
# ================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT), help="repository root holding runs/, results/, configs/, jobs/")
    p.add_argument("--out", default=None, help="output folder (default <root>/results/paper)")
    p.add_argument("--artifacts", default=None, help="large VM artefacts (default <root>/artifacts/a100)")
    p.add_argument("--skip-scripts", action="store_true",
                   help="do not run scripts/f1_tax.py or scripts/f2_drift.py; reuse their earlier outputs under <out>/scripts/")
    p.add_argument("--skip-f2", action="store_true",
                   help="run scripts/f1_tax.py (seconds) but reuse the earlier scripts/f2_drift.py outputs (minutes)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    P = Paths(a.root, a.artifacts)
    out = Path(a.out).resolve() if a.out else P.results / "paper"
    tdir, fdir, sdir = out / "tables", out / "figures", out / "scripts"
    for d in (tdir, fdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    git = git_info(P.root)
    log(f"collecting from {P.root} into {out}")

    jobs = parse_jobs(P)
    status = parse_status(P)
    th = parse_prereg(P)
    commits = {f: git_added_commit(P, f) for f in ("E1_INSTRUCTIONS.md", "E1B_PREREG.md", "PREREG_E2.md",
                                                     "PAPER_EXPERIMENTS.md")}
    plan_path = P.root / "PAPER_EXPERIMENTS.md"
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""

    e1 = load_e1_run(P, P.results, jobs, res_dir=P.results) if (P.results / "e1_meta.json").is_file() else None
    e1b = {d.name: load_e1_run(P, d, jobs) for d in sorted(P.runs.glob("e1b_*")) if d.is_dir()} if P.runs.is_dir() else {}
    diag = None
    if (P.results / "e1_diag_perlayer_surgery.csv").is_file():
        diag = {"surgery": read_csv(P.results / "e1_diag_perlayer_surgery.csv"),
                "blocks": read_csv(P.results / "e1_diag_perlayer.csv"),
                "meta": read_json(P.results / "e1_diag_perlayer_meta.json"),
                "surgery_file": P.rel(P.results / "e1_diag_perlayer_surgery.csv"),
                "blocks_file": P.rel(P.results / "e1_diag_perlayer.csv")}
    x2 = discover_e1_runs(P, jobs, "x2")
    g = discover_e1_runs(P, jobs, "g")
    _RUN_FIT_SEEDS.clear()
    _RUN_FIT_SEEDS.update({r["dir"]: r["fit_seed"] for r in list(x2.values()) + list(g.values())
                           if r["fit_seed"] is not None})
    m1 = load_m1(P)
    m3 = load_m3(P)
    f3 = {}
    for model in ("dinov2_vitb14", "dinov2_vitb14_reg"):
        fp = P.runs / "f3" / f"f3_{model}.json"
        j = read_json(fp)
        if j:
            f3[model] = {"file": P.rel(fp), "json": j}
    gj = read_json(P.runs / "f3" / "f3_gate_G3.json")
    if gj:
        f3["_gate"] = {"file": P.rel(P.runs / "f3" / "f3_gate_G3.json"), "json": gj}
    d_free = None
    if read_json(P.runs / "d" / "d_datafree.json"):
        d_free = {"file": P.rel(P.runs / "d" / "d_datafree.json"), "json": read_json(P.runs / "d" / "d_datafree.json")}
    f4 = None
    if read_json(P.runs / "f4" / "f4_p2.json"):
        f4 = {"file": P.rel(P.runs / "f4" / "f4_p2.json"), "json": read_json(P.runs / "f4" / "f4_p2.json")}

    def script_seed(name):
        sp = P.root / "scripts" / name
        if sp.is_file():
            m = re.search(r"set_seed\((\d+)\)", sp.read_text(encoding="utf-8"))
            if m:
                return int(m[1]), f"set_seed in scripts/{name}; not stored in the output"
        return None, "not recorded"

    x1 = x1b = None
    if read_json(P.runs / "x1" / "x1_svd_check.json"):
        s, ss = script_seed("x1_svd_check.py")
        x1 = {"file": P.rel(P.runs / "x1" / "x1_svd_check.json"), "json": read_json(P.runs / "x1" / "x1_svd_check.json"),
              "seed": s, "seed_source": ss}
    if read_json(P.runs / "x1" / "x1b_svd_compound.json"):
        s, ss = script_seed("x1b_svd_compound.py")
        x1b = {"file": P.rel(P.runs / "x1" / "x1b_svd_compound.json"),
               "json": read_json(P.runs / "x1" / "x1b_svd_compound.json"), "seed": s, "seed_source": ss}
    x4 = {}
    for fp in sorted(P.results.glob("x4_fold_*.json")) if P.results.is_dir() else []:
        x4[fp.stem] = {"file": P.rel(fp), "json": read_json(fp)}

    # ------------------------------------------------------------- architecture for the cost model
    archs, arch_rows = {}, []
    for model in MAIN_MODELS:
        cand = sorted([r for r in x2.values() if r["model"] == model and r["operator_mode"] == "per_layer"
                       and r["layers"] is not None and r["mlp"] is not None and r["meta"]],
                      key=lambda r: (r["fit_seed"] is None, r["fit_seed"]))
        a_ = model_arch(P, model, cand[0] if cand else None, m1, f3)
        if a_:
            archs[model] = a_
            arch_rows.append(a_)
    cost_checks: list[dict] = []

    # ------------------------------------------------------------- tables
    T: dict[str, Table] = {}
    T["T_main_zeroshot"] = build_main(P, x2, m1, archs, cost_checks)
    T["T_healing"], T["T_healing_p3"] = build_healing(P, m3, m1, th, th.get("commit"))
    gen_runs = [r for r in list(x2.values()) + list(g.values())
                if r["operator_mode"] == "per_layer" and r["protocol"] == "v2" and r["status"] != NOT_RUN]
    T["T_generality"], T["T_generality_p4"] = build_generality(P, gen_runs, th, th.get("commit"), plan_text)
    struct_runs = sorted([r for r in gen_runs if r["layers"] is not None and r["mlp"] is not None],
                         key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                                        r["fit_seed"] if r["fit_seed"] is not None else 99))
    T["T_structure_per_layer"], T["T_structure_summary"] = build_structure(P, struct_runs)
    T["T_e1_history"] = build_e1_history(P, e1, e1b, diag, commits)
    T["T_x_checks"] = build_x_checks(P, x1, x1b, x4, m1, plan_text)
    T["T_attention"], T["T_attention_vexact"], T["T_attention_conversion"] = build_attention(P, m1)
    T["T_ablations"], T["T_ablations_seeds"], T["T_ablations_a2_extra"], a_files, a_dups = build_ablations(
        P, T["T_main_zeroshot"])
    T["T_healing_vexact"], vx_info = build_healing_vexact(P, m1, {m_: a_["depth"] for m_, a_ in archs.items()})
    T["T_tier2"], T["T_tier2_transfer"] = build_tier2(P, T["T_main_zeroshot"], T["T_healing"], m1, x2, T["T_ablations"],
                                                      T["T_healing_vexact"])
    T["T_tier2_seg"], T["T_tier2_seg_classes"], T["T_tier2_seg_rerun"], seg_info = build_tier2_seg(P, plan_text)
    add_linear_to_healing(T["T_healing"], T["T_tier2"])
    vexact_device_notes(T["T_healing_vexact"], T["T_main_zeroshot"])
    T["T_latency"], T["T_latency_checks"], m2_meta = build_latency(P, T["T_main_zeroshot"], archs, m1)
    T["T_latency_5090"], T["T_latency_checks_5090"], m2b_meta = build_latency(P, T["T_main_zeroshot"], archs, m1,
                                                                             sub=M2_SECOND, suffix="_5090")
    T["T_latency_devices"] = build_latency_devices(T["T_latency"], T["T_latency_5090"], m2_meta, m2b_meta)
    T["T_latency_profile"], T["T_latency_profile_triton"], prof_meta = build_latency_profile(P)
    T["T_f4"], T["T_f4_p2"], f4_files, _ = build_f4(P)
    th3 = parse_prereg_e3(P)
    commits["PREREG_E3.md"] = th3.get("commit")
    T["T_e3"], T["T_e3_checks"], e3_info = build_e3(P, T["T_tier2"], th3)
    T["T_strong_baselines"], sb_files = build_strong_baselines(P, T["T_main_zeroshot"], archs)
    T["T_flipavg"] = build_flipavg(P)
    T["T_f7"], T["T_f7_layer0"], f7_info = build_f7(P)
    T["T_prereg"] = build_prereg(P, e1, e1b, f3, d_free, T["T_healing_p3"], T["T_generality_p4"], x1, status, th,
                                 commits, f4, T["T_e3_checks"], th3)

    # ------------------------------------------------------------- external scripts (F1, F2)
    f1res, f2res = [], []
    # The DINOv2 size pair has one F1 group per model, so f1_tax.py's default histogram blocks (first half of the
    # deepest model in the group) are each model's own first half (DINOv2-L has 24 blocks).
    f1_groups = (("f1_dinov2", "x2", ["perlayer_reg", "perlayer_plain"]),
                 ("f1_g", "g", ["deit3_b16", "mae_b16", "siglip2_b16"]),
                 ("f1_g_dinov2_s", "g", [G_SIZE_PAIR["dinov2_vits14_reg"]]),
                 ("f1_g_dinov2_l", "g", [G_SIZE_PAIR["dinov2_vitl14_reg"]]))
    # f2_drift.py draws at most three models per figure: the paper pair (both DINOv2 models), then G (DeiT-III, MAE,
    # SigLIP2), then the DINOv2 size pair (S and L).
    f2_groups = (("f2_dinov2", [r for r in gen_runs if r["model"] in MAIN_MODELS]),
                 ("f2_g", [r for r in gen_runs if r["model"] not in MAIN_MODELS and r["model"] not in G_SIZE_PAIR]),
                 ("f2_g_dinov2_sl", [r for r in gen_runs if r["model"] in G_SIZE_PAIR]))
    # Skipped scripts: reuse outputs of an earlier collection (marked "reused"), so the package keeps F1 and F2,
    # and flag them as stale when runs or operator files are present now that they did not read.
    if not a.skip_scripts:
        for tag, sub, arms in f1_groups:
            runs_dir = P.runs / sub
            have = [r for r in (x2 if sub == "x2" else g).values() if r["arm"] in arms
                    and r["layers"] is not None and r["mlp"] is not None]
            if not runs_dir.is_dir():
                fr = {"status": NOT_RUN, "arms": arms, "error": f"no {runs_dir}"}
            elif not have:
                fr = {"status": NOT_RUN, "arms": arms, "error": f"no completed runs for arm(s) {', '.join(arms)}"}
            else:
                fr = run_f1(P, sdir / tag, runs_dir, arms)
            fr.update({"tag": tag, "runs_dir": runs_dir, "out": sdir / tag})
            fr["hist"] = bool(fr.get("status") == "ok" and any(
                npz_for(P, r) is not None for r in (x2 if sub == "x2" else g).values() if r["arm"] in arms))
            f1res.append(fr)
            log(f"F1 {tag}: {fr['status']}")
    else:
        for tag, sub, arms in f1_groups:
            fm = read_json(sdir / tag / "f1_meta.json")
            if not fm:
                continue
            read_names = {Path(str(x.get("dir", ""))).name for x in fm.get("runs", [])}
            now = {r["name"] for r in (x2 if sub == "x2" else g).values() if r["arm"] in arms
                   and r["layers"] is not None and r["mlp"] is not None}
            summ = sdir / tag / "F1_SUMMARY.md"
            f1res.append({"status": "reused", "meta": fm, "tag": tag, "runs_dir": P.runs / sub, "out": sdir / tag,
                          "arms": arms, "stale": sorted(now - read_names),
                          "hist": summ.is_file() and "Twin-score histograms" in summ.read_text(encoding="utf-8")})
    if not (a.skip_scripts or a.skip_f2):
        for tag, group in f2_groups:
            fr = run_f2(P, sdir / tag, [r for r in group if r["layers"] is not None])
            fr["tag"] = tag
            f2res.append(fr)
            log(f"F2 {tag}: {fr['status']} ({', '.join(fr.get('inputs', []))})")
    else:
        for tag, group in f2_groups:
            fm = read_json(sdir / tag / "f2_meta.json")
            if not fm:
                continue
            used = [P.rel(Path(str(x.get("npz")))) for x in fm.get("inputs", [])]
            now = [P.rel(npz_for(P, r)) for r in group if r["layers"] is not None and npz_for(P, r) is not None]
            f2res.append({"status": "reused", "meta": fm, "tag": tag, "out": sdir / tag, "inputs": used,
                          "stale": sorted(set(now) - set(used))})
    T["T_f1_ranges"] = build_f1_table(P, f1res)
    T["T_f2_drift"] = build_f2_table(P, f2res)
    for fr in f1res:
        if fr["status"] in ("ok", "reused"):
            for f in Path(fr["out"]).glob("*"):
                if f.is_file():
                    shutil.copy2(f, tdir / f"ext_{fr['tag']}_{f.name}")
            for f in (Path(fr["out"]) / "plots").glob("*.png"):
                shutil.copy2(f, fdir / f"ext_{fr['tag']}_{f.name}")
    for fr in f2res:
        if fr.get("status") in ("ok", "failed", "reused") and Path(fr["out"]).is_dir():
            for f in Path(fr["out"]).glob("*"):
                if f.is_file():
                    shutil.copy2(f, tdir / f"ext_{fr['tag']}_{f.name}")
            for f in (Path(fr["out"]) / "plots").glob("*.png"):
                shutil.copy2(f, fdir / f"ext_{fr['tag']}_{f.name}")

    for sub, tag in (("f4", "f4"), (P.m2, "m2"), (M2_SECOND, "m2_5090")):
        for f in sorted((P.runs / sub / "plots").glob("*.png")) if (P.runs / sub / "plots").is_dir() else []:
            shutil.copy2(f, fdir / f"ext_{tag}_{f.name}")

    # ------------------------------------------------------------- figures
    figs = {}
    figs["fig_accuracy_vs_k.png"] = fig_accuracy_vs_k(T["T_main_zeroshot"], fdir / "fig_accuracy_vs_k.png")
    figs["fig_healing_curves.png"] = fig_healing(m3, fdir / "fig_healing_curves.png")
    figs["fig_tax_vs_depth.png"] = fig_tax(T["T_structure_per_layer"], fdir / "fig_tax_vs_depth.png")
    figs["fig_eA_vs_depth.png"] = fig_eA(T["T_structure_per_layer"], fdir / "fig_eA_vs_depth.png")
    figs["fig_structure_summary.png"] = fig_structure_summary(T["T_generality"], fdir / "fig_structure_summary.png")

    # ------------------------------------------------------------- write tables
    groups = [
        ("T_main_zeroshot", "Main zero-shot table", ["T_main_zeroshot"]),
        ("T_healing", "Healing (M3 pilot and V-exact healing)", ["T_healing", "T_healing_p3", "T_healing_vexact"]),
        ("T_e3", "E3: matched healed baselines at full conversion (PREREG_E3.md, gate G5)", ["T_e3_checks", "T_e3"]),
        ("T_strong_baselines", "Strong compression baselines, zero-shot (scripts/strong_baselines.py)",
         ["T_strong_baselines"]),
        ("T_flipavg", "Flip averaging against exact equivariance (scripts/flip_avg_baseline.py)", ["T_flipavg"]),
        ("T_f7", "F7 visuals: mirror twins and layer-0 filters (scripts/f7_visuals.py)", ["T_f7", "T_f7_layer0"]),
        ("T_generality", "Generality (G)", ["T_generality", "T_generality_p4"]),
        ("T_structure_per_layer", "Structure per layer (F1 style)", ["T_structure_summary"]),
        ("T_prereg", "Pre-registered checks", ["T_prereg"]),
        ("T_x_checks", "Stage 0 checks", ["T_x_checks"]),
        ("T_e1_history", "E1 and E1b (earlier protocol)", ["T_e1_history"]),
        ("T_attention", "Attention conversion (M1)", ["T_attention", "T_attention_vexact", "T_attention_conversion"]),
        ("T_ablations", "Ablations (A1 to A5)", ["T_ablations_seeds", "T_ablations", "T_ablations_a2_extra"]),
        ("T_tier2", "Tier 2: linear probes, transfer and ADE20K segmentation",
         ["T_tier2", "T_tier2_transfer", "T_tier2_seg", "T_tier2_seg_rerun"]),
        ("T_latency", "M2: latency and analytic cost", ["T_latency", "T_latency_checks", "T_latency_devices",
                                                        "T_latency_5090", "T_latency_checks_5090"]),
        ("T_latency_profile", "M2 profile: where the time goes", ["T_latency_profile", "T_latency_profile_triton"]),
        ("T_f4", "F4: symmetry against low rank (P2)", ["T_f4_p2", "T_f4"]),
        ("T_f1_ranges", "F1 ranges (scripts/f1_tax.py)", ["T_f1_ranges"]),
        ("T_f2_drift", "F2 drift summary (scripts/f2_drift.py)", ["T_f2_drift"]),
    ]
    for t in T.values():
        t.public().to_csv(tdir / f"{t.name}.csv", index=False)
    head = [f"Generated by `scripts/collect_results.py` ({time.strftime('%Y-%m-%d %H:%M')}, commit "
            f"{short(git['git_hash'])}). Every value is read from the result files named in the CSV columns "
            "source_files, run, source or source_file. Empty cells: not available.", ""]
    for name, title, members in groups:
        lines = [f"# {title}", ""] + head
        if name == "T_structure_per_layer":
            lines += T["T_structure_summary"].md_lines()
            ts = T["T_structure_per_layer"]
            lines += [f"## {ts.title}", ""] + ts.notes + [""]
            for run in struct_runs:
                rr = [r for r in ts.rows if r["run"] == run["dir"]]
                sub = Table("x", f"{MODEL_LABEL.get(run['model'], run['model'])}, fit seed {run['fit_seed']} (`{run['dir']}`, "
                            f"git {short(run['git'])})", [], ts.md_cols)
                lines += sub.md_lines(rr, heading_level="###")
            if not struct_runs:
                lines += ["No rows (inputs not run).", ""]
        elif name == "T_latency_profile":
            tp, tt = T["T_latency_profile"], T["T_latency_profile_triton"]
            lines += tp.md_lines()
            conv = [r for r in tp.rows if r["variant"] not in ("dense", "original")]
            if tp.rows:
                for view in profile_views(tp):
                    lines += view.md_lines(conv, heading_level="###")
            if tp.rows and not tt.rows and (P.runs / "m2_profile" / "m2_triton.csv").is_file():
                lines += [f"## {tt.title}", "", "No fused Triton kernels in this run (they appear only in compiled "
                          "runs; m2_triton.csv has no rows).", ""]
            else:
                lines += tt.md_lines()
        elif name == "T_main_zeroshot":
            lines += T["T_main_zeroshot"].md_lines()
            noted = [r for r in T["T_main_zeroshot"].rows if r.get("remark_id")]
            if noted:
                lines += ["## Row notes", ""]
                lines += [f"- [{r['remark_id']}] {r['model_label']}, k = {r['k']}, {r['variant_label']} (run "
                          f"{r['experiment']}): {r['remark']}" for r in noted]
                lines.append("")
            notrun = [r for r in T["T_main_zeroshot"].rows if r["status"] == NOT_RUN]
            if notrun:
                lines += ["## Rows not run and why", ""]
                seen = {}
                for r in notrun:
                    seen.setdefault((r["model_label"], r["note"]), []).append(f"k = {r['k']} {r['variant']}")
                for (ml, note), items in seen.items():
                    lines.append(f"- {ml}: {', '.join(items)}: {note}.")
                lines.append("")
        else:
            for mname in members:
                lines += T[mname].md_lines()
        write_text(tdir / f"{name}.md", lines)

    # cost model
    cm = ["# Analytic cost model", "", "Function `analytic_cost` in `scripts/collect_results.py`. These are analytic "
          "counts, not measured latency.", "", "## Assumptions", ""] + [f"- {x}" for x in COST_ASSUMPTIONS] + ["", "## Inputs per model", ""]
    for a_ in arch_rows:
        cm.append(f"- {MODEL_LABEL[a_['model']]}: d {a_['d']}, h {a_['h']}, depth {a_['depth']}, T {a_['T']} tokens "
                  f"({a_['n_patches']} patches plus {a_['n_prefix']} prefix), patch {a_['patch']}, in_chans {a_['in_chans']}, "
                  f"dense parameters {fmt(a_['model_params_dense'], ',d')}.")
        for k_, v_ in a_["sources"].items():
            cm.append(f"  - {k_}: {v_}")
    for m_ in MAIN_MODELS:
        if m_ not in archs:
            cm.append(f"- {MODEL_LABEL[m_]}: inputs not available (cost columns empty).")
    cm += ["", "## Checks against counts recorded by the runs", ""]
    worst = 0.0
    for ck in cost_checks:
        rel = abs(ck["collector"] - ck["recorded"]) / max(1e-12, abs(ck["recorded"]))
        ck["rel_diff"] = rel
        worst = max(worst, rel)
    pd.DataFrame(cost_checks).to_csv(tdir / "cost_model_checks.csv", index=False)
    cm.append(f"- {len(cost_checks)} parameter counts recomputed from the per-block CSVs and compared with the totals the "
              f"runs recorded (mlp_params_kept_frac of e1_surgery.csv; mlp_params_kept, attn_params_kept, model_params of "
              f"m1_knn.csv): largest relative difference {worst:.2e} (`cost_model_checks.csv`).")
    dense = {m_: analytic_cost(archs[m_], []) for m_ in archs}
    for m_, dd in dense.items():
        cm.append(f"- {MODEL_LABEL[m_]} dense: {dd['gflops_model_dense']:.2f} GFLOPs per image (analytic).")
    write_text(tdir / "cost_model.md", cm)
    (tdir / "cost_model.json").write_text(json.dumps({"assumptions": COST_ASSUMPTIONS, "architectures": arch_rows,
                                                      "dense": dense, "max_rel_check_diff": worst}, indent=2, default=str),
                                          encoding="utf-8")

    # ------------------------------------------------------------- deviations and inventory
    dev = []
    for model in MAIN_MODELS:
        have = sorted({r["fit_seed"] for r in x2.values() if r["model"] == model and r["status"] != NOT_RUN
                       and r["operator_mode"] == "per_layer"})
        if len(have) < 3:
            dev.append(f"X3: {MODEL_LABEL[model]} has fit seeds {', '.join(map(str, have)) or 'none'} (plan: 3 seeds for "
                       "every headline number).")
    for r in T["T_main_zeroshot"].rows:
        if r.get("budget_within_1pct") is False:
            dev.append(f"Budget: {r['model_label']} k = {r['k']} {r['variant']} keeps {r['budget_gap_vs_svd_pct']:+.2f}% "
                       "parameters relative to its SVD baseline (plan: within 1%; SVD and random are matched to V-local).")
    for r in T["T_x_checks"].rows:
        if r["experiment"] == "X4" and r.get("verdict") == FAIL:
            dev.append(f"X4: {r['model']} {r['quantity']} = {fmt(r['value'], '.2e')} misses {r['bar']}.")
    for r in T["T_generality_p4"].rows:
        if r["verdict"] == FAIL:
            dev.append(f"P4: {r['check']} fails as worded ({r['statement']}).")
    for r in T["T_prereg"].rows:
        if r["id"] in ("P5", "G4") and r["verdict"] in (FAIL, "FAIL"):
            dev.append(f"{r['id']}: {r['verdict']} ({r['source_file']}).")
    if not any(r["model"] == "mae_b16" and r.get("decision_source") == "e1_meta.json" for r in T["T_generality"].rows):
        if any(r["model"] == "mae_b16" for r in T["T_generality"].rows):
            dev.append("G: the MAE run has no e1_meta.json (its k-NN run was stopped); its decision rows are recomputed "
                       "with mirror.report.decide and its git hash is not recorded.")
    for r in x2.values():
        if r["status"].startswith("partial"):
            dev.append(f"X2: `{r['dir']}` is {r['status']}.")
    for r in T["T_tier2"].rows + T["T_tier2_transfer"].rows:
        if r.get("wd_at_grid_edge"):
            dev.append(f"Tier 2: {r['model_label']} {r.get('dataset', 'ImageNet')} {r['variant']}: the chosen weight decay "
                       f"{fmt(r.get('chosen_wd'), 'g')} is at the edge of the grid.")
        if r.get("final_hit_iter_limit"):
            dev.append(f"Tier 2: {r['model_label']} {r['variant']}: the final L-BFGS fit hit its iteration limit.")
    for r in T["T_latency_checks"].rows:
        if r.get("pass") is False:
            dev.append(f"M2 check failed: {r['variant']} k = {r['k']} {r['dtype']} ({r.get('criterion')}, value "
                       f"{fmt(r.get('value'), '.2e')}).")
    p2r = T["T_prereg"].find(id="P2")
    if p2r is not None and str(p2r.get("verdict", "")).startswith("INCOMPLETE"):
        dev.append(f"P2: {p2r['verdict']} (runs/f4/f4_p2.json).")
    for d_ in a_dups:
        dev.append(f"A: the key ablation/variant/k/fit seed {d_} appears in more than one CSV; the first was kept.")
    for r in T["T_main_zeroshot"].rows:
        if r.get("canonical") == "no":
            dev.append(f"Duplicate setting: {r['model_label']} k = {r['k']} {r['variant_label']} has an M1 row next to the "
                       f"canonical {CANONICAL_RUN} row; the M1 row is kept and marked not canonical (operator file round "
                       "trip, a numerical difference; T_main_zeroshot row notes).")
    unmatched = [r for r in T["T_ablations_seeds"].rows if r.get("params_matched") is False and r["ablation"] in ("A1", "A2", "A4")]
    for model in sorted({r["model"] for r in unmatched}):
        for k in sorted({r["k"] for r in unmatched if r["model"] == model}):
            names = [f"{r['ablation']} {r['variant']}" for r in unmatched if r["model"] == model and r["k"] == k]
            dev.append(f"A: not parameter-matched to the default ({MODEL_LABEL.get(model, model)}, k = {k}): "
                       f"{', '.join(names)}; ratios in T_ablations_seeds (params vs default).")
    for r in T["T_latency"].rows:
        if r.get("timing_flag"):
            dev.append(f"M2: {r['label']} k = {r['k']}, batch {r['batch']}, {r['mode']} is a probable timing outlier "
                       "(measured speed-up above its analytic one); kept and flagged in T_latency.")
    for ds in sorted({r["dataset"] for r in T["T_tier2_transfer"].rows if r.get("dataset_at_ceiling")}):
        dev.append(f"Tier 2: {ds} is at ceiling and does not separate the methods (T_tier2_transfer notes).")
    for r in T["T_tier2"].rows:
        if str(r.get("tier1_same_model", "")).startswith("no"):
            dev.append(f"Tier 2: {r['model_label']} {r['variant']}: the probed model is not the model behind its Tier-1 "
                       f"number ({'a different random draw' if r.get('method') == 'random' else 'rebuilt from the float32 operator file, a numerical difference'}; "
                       "T_tier2 notes).")
    for r in T["T_healing_vexact"].rows:
        if r.get("note") and r.get("gpu"):
            dev.append(f"V-exact healing k = {r['k']}: run on {r['gpu']}, which measures its own original; compare within "
                       "the row (T_healing_vexact notes).")
    for r in T["T_tier2_seg"].rows:
        if r.get("note"):
            dev.append(f"Tier 2 ADE20K: {r['model_label']} {r['variant']}: {r['note']}.")
    for r in T["T_healing_vexact"].rows:
        if r.get("tiny"):
            dev.append(f"V-exact healing k = {r['k']}: `{r['dir']}` holds a TINY smoke-test run, not a result.")
        if r.get("exact_pass") is False:
            dev.append(f"V-exact healing k = {r['k']}: exactness after healing fails its tolerance "
                       f"({fmt(r.get('exact_max_rel'), '.2e')} against {fmt(r.get('exact_tol'), 'g')}).")
        if r.get("flip_fp32_pass") is False:
            dev.append(f"V-exact healing k = {r['k']}: the fp32 flip check is not 100% ({fmt(r.get('flip_fp32'), '.2f')}%).")
    if prof_meta and (prof_meta.get("tiny") or prof_meta.get("device_type") == "cpu"):
        dev.append("M2 profile: `runs/m2_profile/` holds a CPU smoke test (operator self times), not a GPU measurement.")
    for meta_, sub_ in ((m2_meta, P.m2), (m2b_meta, M2_SECOND)):
        if meta_ and str(meta_.get("check_images", "")).startswith("random"):
            dev.append(f"M2 (`runs/{sub_}/`): the equivalence checks used {meta_['check_images']}, not validation images "
                       "(scripts/m2_latency.py falls back to them when the validation cache cannot be loaded); timings are "
                       "unaffected (random inputs in every run).")
    for r in T["T_strong_baselines"].rows:
        if r.get("experiment") == "SB" and r.get("within_1pct") is False:
            dev.append(f"Strong baselines: {r['label']} k = {r['k']} keeps {fmt(r.get('ratio_to_mirror'), '.4f')} of the "
                       "Mirror budget (plan: within 1%).")
    if th3.get("exists") and not th3.get("active"):
        dev.append("E3: PREREG_E3.md is not marked ACTIVE (signed off); its checks are not pre-registered verdicts yet.")
    for k_, r in e3_info["rows"].items():
        if r.get("tiny"):
            dev.append(f"E3 arm {r['arm']}: `{r['dir']}` holds a TINY smoke-test run, not a result.")
        if r.get("validity") == FAIL:
            dev.append(f"E3 arm {r['arm']} fails a validity condition (PREREG_E3.md Section 7: invalid, fix and rerun that "
                       f"arm): {r.get('validity_detail')}.")
        if r.get("linear_note") and k_ != "v0_a100":
            dev.append(f"E3 arm {r['arm']}, linear probe: {r['linear_note']}.")
        if r.get("extra_win_matches_recorded") is False:
            dev.append(f"E3 arm {r['arm']}: the run's own flag for the extra learning rate differs from the collector's "
                       "recomputation from lr_extra.json (T_e3).")
    lr_ = T["T_e3_checks"].find(id="E3.lr")
    if lr_ is not None and str(lr_["verdict"]).startswith("yes"):
        dev.append(f"E3 (PREREG_E3.md Section 8): the extra learning rate would have won for arm(s) "
                   f"{lr_['verdict'].replace('yes: ', '')}; the main comparison is flagged as possibly under-tuned.")
    for r in T["T_flipavg"].rows:
        if r["model"] == "tiny":
            dev.append("Flip averaging: `runs/t1_flipavg/` holds TINY smoke-test rows, not results.")
            break
    if (f7_info.get("meta") or {}).get("tiny"):
        dev.append("F7: `runs/f7/` holds a TINY smoke-test run, not a result.")
    inv = []

    def inv_add(path, what):
        p = P.root / path
        inv.append({"path": path, "what": what, "present": p.exists() and (p.is_file() or any(p.iterdir()))})

    inv_add("results/e1_meta.json", "E1")
    for n in sorted(e1b):
        inv_add(f"runs/{n}/results/e1_meta.json", "E1b arm")
    inv_add("runs/x1/x1_svd_check.json", "X1")
    inv_add("runs/x1/x1b_svd_compound.json", "X1b")
    for n in sorted(x2):
        inv_add(f"runs/x2/{n}/results/e1_meta.json", "X2/X3 run (meta)")
        inv_add(f"runs/x2/{n}/results/e1_surgery.csv", "X2/X3 run (surgery)")
    for n in sorted(set(g) | {f"{arm}_s0" for arm in G_SIZE_PAIR.values()}):
        what = "G run, DINOv2 size pair" if re.sub(r"_s\d+$", "", n) in G_SIZE_PAIR.values() else "G run"
        inv_add(f"runs/g/{n}/results/e1_meta.json", f"{what} (meta)")
        inv_add(f"runs/g/{n}/results/e1_surgery.csv", f"{what} (surgery)")
    for f in ("m1_knn.csv", "m1_meta.json", "m1_ranks.csv", "m1_heads.csv", "m1_vexact.csv", "m1_vexact_blocks.csv",
              "m1_exactness.csv", "m1_flipcheck.json"):
        inv_add(f"runs/m1/{f}", "M1")
    for mth in ("mirror", "svd", "random"):
        inv_add(f"runs/m3_pilot/{mth}/knn.json", f"M3 pilot {mth}: final k-NN")
        inv_add(f"runs/m3_pilot/{mth}/final/metrics.json", f"M3 pilot {mth}: healing metrics")
    for r in T["T_healing_vexact"].rows:
        kk = r["k"]
        files = ["summary.json", "knn.json", "exactness.json", "final/metrics.json", "lr_selection/lr_selection.json",
                 "SUMMARY.md"]
        if r.get("full_conversion"):
            files.insert(3, "flipcheck.json")
        for fn in files:
            inv_add(f"runs/m3_vexact/k{kk}/{fn}", f"V-exact healing k = {kk}")
    inv_add("runs/d/d_datafree.json", "D (P5)")
    inv_add("runs/f3/f3_gate_G3.json", "F3 (P1, G3)")
    inv_add("runs/f4/f4_p2.json", "F4 (P2)")
    fsrc = P.root / "scripts" / "f4_rule.py"
    mm_ = re.search(r"PREREG_MODELS = \(([^)]*)\)", fsrc.read_text(encoding="utf-8")) if fsrc.is_file() else None
    for m_ in re.findall(r'"([^"]+)"', mm_[1]) if mm_ else []:
        inv_add(f"runs/f4/f4_{m_}.csv", "F4 per-block rows (P2 model)")
    inv_add("runs/a", "A1 to A5 ablations")
    for f in a_files:
        inv_add(P.rel(f), "ablation rows (every k and fit seed in the file)")
    for f in ("m2_latency.csv", "m2_checks.csv", "m2_meta.json"):
        inv_add(f"runs/{P.m2}/{f}", "M2 latency")
    for f in ("m2_profile.csv", "m2_submodules.csv", "m2_triton.csv", "m2_top_kernels.csv", "M2_PROFILE.md",
              "m2_profile_meta.json"):
        inv_add(f"runs/m2_profile/{f}", "M2 profile (scripts/m2_profile.py)")
    inv_add("PREREG_E3.md", "E3 pre-registration (ACTIVE once signed off)")
    for key_, arm_, _lab, kind_, sub_, t2sub_ in E3_ARMS:
        d_ = e3_info["rows"][key_]["dir"]
        fs_ = ["summary.json", "knn.json", "final/metrics.json", "lr_selection/lr_selection.json"]
        if kind_ == "vexact":
            fs_ += ["exactness.json", "flipcheck.json"]
        if key_ not in ("v0", "v0_a100"):
            fs_.append("lr_extra.json")
        for fn in fs_:
            inv_add(f"{d_}/{fn}", f"E3 arm {arm_}")
        if t2sub_:
            inv_add(f"runs/{t2sub_}/t2_linear.csv", f"E3 arm {arm_}: Tier-2 linear probe")
    if e3_info.get("k") is not None:
        inv_add(f"runs/{E3_LR_PROBE}/k{e3_info['k']}/lr_extra.json", "E3 Section 8: learning-rate probe of arm 1")
    for f in sb_files or [P.runs / "sb" / "SB_s0.csv"]:
        inv_add(P.rel(f), "strong baselines, zero-shot (scripts/strong_baselines.py)")
    inv_add("runs/t1_flipavg/flipavg.csv", "flip-averaging baseline (scripts/flip_avg_baseline.py)")
    for f in ("m2_latency.csv", "m2_checks.csv", "m2_meta.json"):
        inv_add(f"runs/{M2_SECOND}/{f}", "M2 latency, second device")
    inv_add("runs/t2_seg_rerun/t2_ade20k.csv", "Tier 2 ADE20K rerun (run-to-run spread)")
    for f in ("f7_blocks.csv", "f7_meta.json", "F7_SUMMARY.md"):
        inv_add(f"runs/f7/{f}", "F7 visuals (scripts/f7_visuals.py)")
    inv_add("results/paper/figures_paper/fig0_teaser.pdf", "F7 teaser figure (drawn by scripts/f7_visuals.py)")
    inv_add("runs/t2/t2_linear.csv", "Tier 2 ImageNet linear probe")
    inv_add("runs/t2/t2_transfer.csv", "Tier 2 transfer probes")
    inv_add("runs/t2_seg/t2_ade20k.csv", "Tier 2 ADE20K linear segmentation (scripts/tier2_ade20k.py)")
    seg_models = sorted({r["model"] for r in T["T_tier2_seg"].rows} | set(seg_info["meta"]))
    for m_ in seg_models:
        inv_add(f"runs/t2_seg/seg/{m_}", f"Tier 2 ADE20K per-variant JSON with per-class IoU ({m_})")
        inv_add(f"runs/t2_seg/t2_ade20k_meta_{m_}.json", f"Tier 2 ADE20K run meta ({m_})")
    if not seg_models:
        inv_add("runs/t2_seg/seg", "Tier 2 ADE20K per-variant JSON with per-class IoU")
    for r in list(x2.values()) + list(g.values()):
        if r["operator_mode"] == "per_layer" and r["status"] != NOT_RUN:
            inv_add(P.rel(P.artifacts / r["res_rel"] / "e1_operators.npz") if npz_for(P, r) is None or
                    not (r["res"] / "e1_operators.npz").is_file() else P.rel(r["res"] / "e1_operators.npz"),
                    "operators npz (F1 histograms, F2)")
    inv_add(P.rel(P.artifacts / "runs" / "m1" / "m1_structure.npz"), "M1 structure npz (not needed by the tables)")
    ctx = {"tables": T, "status": status, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "commits": commits,
           "e1": e1, "e1b": e1b, "x1": x1, "x1b": x1b, "x2": x2, "g": g, "x4": x4, "m1": m1, "m3": m3, "d": d_free,
           "f4": f4, "f1": f1res, "f2": f2res, "a_files": a_files, "inventory": inv, "deviations": dev,
           "plan_text": plan_text, "m2_meta": m2_meta, "f4_files": f4_files, "figs_dir": fdir, "seg_info": seg_info,
           "vx_info": vx_info, "prof_meta": prof_meta, "e3_info": e3_info, "sb_files": sb_files, "f7_info": f7_info,
           "m2b_meta": m2b_meta}

    # ------------------------------------------------------------- INDEX and numbers
    R = Registry()
    index = build_index(P, ctx, R)
    write_text(out / "INDEX.md", index)
    (out / "numbers.json").write_text(json.dumps(R.items, indent=2, default=str), encoding="utf-8")
    meta = {"written_by": "scripts/collect_results.py", "date": ctx["date"], "git": git, "args": vars(a),
            "root": str(P.root), "artifacts": str(P.artifacts), "seed": None,
            "seed_note": "no randomness in the collector; F2 null draws use f2_drift.py's own --seed (default 0)",
            "prereg_thresholds": th, "prereg_e3": th3, "e3_values": e3_info.get("values"), "prereg_commits": commits,
            "figures": figs,
            "tables": {k: len(v.rows) for k, v in T.items()},
            "f1": [{k: str(v) if isinstance(v, Path) else v for k, v in fr.items() if k != "meta"} for fr in f1res],
            "f2": [{k: str(v) if isinstance(v, Path) else v for k, v in fr.items() if k != "meta"} for fr in f2res],
            "inventory": inv, "deviations": dev, "n_numbers": len(R.items), "elapsed_s": round(time.time() - t0, 1)}
    (out / "collector_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"wrote {len(T)} tables, {sum(bool(v) for v in figs.values())} figures, {len(R.items)} numbers in "
        f"{time.time() - t0:.1f} s")
    return {"out": out, "tables": T, "registry": R, "meta": meta, "ctx": ctx}


if __name__ == "__main__":
    main()

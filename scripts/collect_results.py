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
  tables/      T_main_zeroshot, T_healing, T_generality, T_structure_per_layer, T_prereg,
               T_x_checks, T_e1_history, T_attention (each as .csv and .md; some .md files hold
               several related CSVs), plus the outputs of f1_tax.py and f2_drift.py (prefix ext_);
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

# Models: display names, fixed colours (colour follows the model) and markers. Slots 1 to 5 of the
# reference categorical palette (light surface); slots 1 to 3 are the ones in mirror/report.py.
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
MODEL_LABEL = {"dinov2_vitb14_reg": "DINOv2-reg", "dinov2_vitb14": "DINOv2", "deit3_b16": "DeiT-III",
               "mae_b16": "MAE", "siglip2_b16": "SigLIP2"}
MODEL_SHORT = {"dinov2_vitb14_reg": "reg", "dinov2_vitb14": "plain", "deit3_b16": "deit3", "mae_b16": "mae",
               "siglip2_b16": "siglip2"}
MODEL_COLOUR = {"dinov2_vitb14_reg": C_BLUE, "dinov2_vitb14": C_ORANGE, "deit3_b16": C_AQUA, "mae_b16": C_YELLOW,
                "siglip2_b16": C_MAGENTA}
MODEL_MARKER = {"dinov2_vitb14_reg": "o", "dinov2_vitb14": "s", "deit3_b16": "^", "mae_b16": "D", "siglip2_b16": "v"}
MODEL_ORDER = list(MODEL_LABEL)
MAIN_MODELS = ["dinov2_vitb14_reg", "dinov2_vitb14"]
MAIN_KS = [3, 6, 9, 12]
BG = "#fcfcfb"

# Zero-shot variants of T_main_zeroshot: key, label, budget, experiment, variant name in the source CSV.
VARIANTS = [
    ("mirror_mlp", "Mirror per-layer, MLP only", "MLP only", "X2", "surgery"),
    ("svd_mlp", "SVD at the MLP-only budget", "MLP only", "X2", "svd"),
    ("random_mlp", "Random structure, MLP only", "MLP only", "X2", "random"),
    ("mirror_mlp_m1", "Mirror per-layer, MLP only (M1 run, A6)", "MLP only", "M1", "mlp_only"),
    ("vlocal", "Mirror V-local, MLP + attention", "MLP + attention", "M1", "vlocal"),
    ("vexact", "Mirror V-exact, MLP + attention", "MLP + attention", "M1", "vexact"),
    ("svd_attn", "SVD at the V-local budget", "MLP + attention", "M1", "svd"),
    ("random_attn", "Random structure, MLP + attention", "MLP + attention", "M1", "random"),
]
VARIANT_INFO = {v[0]: v for v in VARIANTS}
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
    "Mirror MLP: each matrix costs its equivariant free-parameter count (h d + tr(P) tr(A)) / 2 per token "
    "(eq_params_fc1, eq_params_fc2 of e1_mlp.csv; mlp_eq_fc1, mlp_eq_fc2 of m1_ranks.csv), i.e. every "
    "free parameter is used once per token in even/odd coordinates.",
    "Mirror attention (M1): the free-parameter counts of Wq, Wk, Wv, Wo per block (attn_free_q/k/v/o of "
    "m1_ranks.csv for V-local; attn_free_matrices and mlp_free of m1_vexact_blocks.csv for V-exact, "
    "which depend on the span k).",
    "Basis rotations: every variant with per-layer operators (Mirror MLP only, V-local, and the random "
    "controls, which use random per-layer involutions) pays one d x d rotation per converted block "
    "(d^2 MACs per token): the stream is rotated from the basis of A_(l-1) to that of A_l at each "
    "converted block, the rotation into the first basis is folded into the linear patch embedding and the "
    "one after the last converted block into the next dense block. V-exact (one operator for the span) "
    "and SVD need none. Rotation matrices are d^2 stored parameters each; they are reported in "
    "rotation_params and are not included in model_params (which follows m1_knn.csv).",
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
               ("budget", "budget", ""), ("n", "n_seeds", "d"),
               ("top-1", lambda r: pm(r.get("top1_mean"), r.get("top1_std")), ""),
               ("drop", lambda r: pm(r.get("drop_mean"), r.get("drop_std")), ""),
               ("vs SVD", lambda r: pm(r.get("margin_svd_mean"), r.get("margin_svd_std")), ""),
               ("vs random", lambda r: pm(r.get("margin_random_mean"), r.get("margin_random_std")), ""),
               ("flip consistency", lambda r: pm(r.get("flip_mean"), r.get("flip_std")), ""),
               ("kept (converted)", "kept_frac_converted", ".3f"), ("MLP kept", "mlp_kept_frac", ".3f"),
               ("model params", "model_params", ",d"), ("GFLOPs", "gflops_model", ".2f"),
               ("saved, model %", "gflops_saved_model_pct", ".1f"),
               ("saved, converted %", "gflops_saved_converted_pct", ".1f"),
               ("status", "status", "")])
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
        "how far a Mirror variant's kept parameters are from its SVD baseline (the plan asks for within 1%).",
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
                            "flip_mean": f_m, "flip_std": f_s})
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
    return T


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
               ("trainable params", "n_trainable", ",d")])
    T.notes = ["Top-1 and flip consistency in percent (protocol v2 k-NN). Held-out loss: the distillation loss "
               "on the 5k held-out train images (normalised by fixed teacher mean squares, the same for every "
               "method). Chosen LR: lowest held-out loss after the short selection budget, from "
               "{1e-5, 5e-5, 1e-4}. M1 zero-shot: the matching variant of runs/m1/m1_knn.csv at the same k, as a "
               "cross-check of the zero-shot column."]
    m1map = {"mirror": "vlocal", "svd": "svd", "random": "random"}
    default_cfg = load_cfg(P, "configs/m3_pilot.yaml")
    lrs = next((m3[m]["lrsel"].get("lrs") for m in m3 if m3[m]["lrsel"]), None) or \
        ((default_cfg.get("lr_selection") or {}).get("lrs"))
    T.notes = [T.notes[0].replace("{1e-5, 5e-5, 1e-4}", "{" + ", ".join(f"{x:g}" for x in (lrs or [])) + "} "
                                  "(lr_selection.json)")]
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


# ================================================================================================
# T_generality and P4
# ================================================================================================
def first_half(run):
    L = int(run["mlp"]["block"].max())
    return 1, L // 2


def build_generality(P, gen_runs, th, prereg_commit, plan_text):
    T = Table("T_generality", "G: generality across models (protocol v2, per-layer operators, zero-shot)",
              ["model", "fit_seed"],
              [("model", "model_label", ""), ("fit seed", "fit_seed", "d"), ("outcome", "outcome", ""),
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
               "over blocks 1 to L/2. Surgery, SVD and margin: zero-shot k-NN top-1 (%) at k = L/2."]
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
        kh = hi
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
def build_prereg(P, e1, e1b, f3, d_free, Tp3, Tp4, x1, status, th, commits, f4):
    T = Table("T_prereg", "Every pre-registered check", ["id"],
              [("id", "id", ""), ("experiment", "experiment", ""), ("check", "statement", ""), ("value", "value", ".4g"),
               ("threshold", "threshold", ""), ("verdict", "verdict", ""), ("source", "source_file", ""),
               ("pre-registration", lambda r: f"{r.get('prereg_file', '')} {short(r.get('prereg_commit'))}"
                if r.get("prereg_file") else "", ""), ("note", "note", "")])
    T.notes = ["Thresholds are read from the pre-registration files (or from the run outputs, which copy them); the "
               "commit is the one that added the file (git log --diff-filter=A). Pending: a run is still missing."]

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
    cell = ""
    for line in status["lines"]:
        if line.startswith("| X1"):
            parts = [x.strip() for x in line.strip("|").split("|")]
            cell = parts[1] if len(parts) > 1 else ""
    T.add({"id": "G1", "experiment": "X1", "statement": "gate G1: if the SVD collapse on the registers model is a bug, fix "
           "it and rerun every SVD number", "verdict": PASS if cell.lower().startswith("passed") else
           ("see note" if cell else NOT_RUN), "threshold": "no bug found (X1 checks 1 to 4; no numeric bar)",
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


def build_ablations(P):
    """Per-seed rows of runs/a/<model>/A*_s*.csv, the mean over fit seeds, and the A2 extra pairings."""
    T = Table("T_ablations", "A1 to A5: zero-shot ablations, one row per fit seed",
              ["model", "ablation", "variant", "k", "fit_seed"],
              [("model", "model_label", ""), ("ablation", "ablation", ""), ("variant", "variant", ""), ("k", "k", "d"),
               ("fit seed", "fit_seed", "d"), ("top-1", "top1", ".2f"), ("drop", "drop_points", ".2f"),
               ("flip consistency", "flip_consistency", ".2f"), ("MLP kept", "mlp_params_kept_frac", ".3f"),
               ("exact projection", "params_exact", ""), ("reused from", "reused_from", "")])
    T.notes = ["Zero-shot k-NN top-1 and flip consistency (%), paper protocol; drop = original minus variant. "
               "Reused from: the variant is the same model as the named cell (identical content signature in "
               "scripts/stage1_ablate.py), so its result was reused, not evaluated again. A fit seed of -1 marks a row "
               "fitted on pooled fit images (see the row's info)."]
    S = Table("T_ablations_seeds", "A1 to A5: mean over fit seeds", ["model", "ablation", "variant", "k"],
              [("model", "model_label", ""), ("ablation", "ablation", ""), ("variant", "variant", ""), ("k", "k", "d"),
               ("n", "n_seeds", "d"), ("fit seeds", "fit_seeds", ""),
               ("top-1", lambda r: pm(r.get("top1_mean"), r.get("top1_std")), ""),
               ("drop", lambda r: pm(r.get("drop_mean"), r.get("drop_std")), ""),
               ("flip consistency", lambda r: pm(r.get("flip_mean"), r.get("flip_std")), ""), ("note", "note", "")])
    S.notes = ["Mean and sample standard deviation (n - 1) over the fit seeds present; with one seed none is shown."]
    X = Table("T_ablations_a2_extra", "A2 pairings hungarian, hungarian_cycle and generic: pairing and projection as recorded",
              ["model", "variant", "k", "fit_seed"],
              [("variant", "variant", ""), ("k", "k", "d"), ("fit seed", "fit_seed", "d"), ("top-1", "top1", ".2f"),
               ("exact projection", "params_exact", ""), ("pairing", "pairing", ""), ("projection", "projection", "")])
    X.notes = ["Pairing and projection are the descriptions each row records in its info column (scripts/stage1_ablate.py)."]
    base = P.runs / "a"
    files = sorted(base.glob("*/A*_s*.csv")) if base.is_dir() else []
    for f in files:
        df = read_csv(f)
        if df is None:
            continue
        model = str(df["model"].iloc[0]) if "model" in df else f.parent.name
        for _, r in df.iterrows():
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
            row["pairing"], row["projection"] = info.get("pairing", ""), info.get("projection", "")
            row["fit_rows"] = info.get("fit_rows", "")
            T.add(row)
            if row["ablation"] == "A2" and row["variant"] in A2_EXTRA:
                X.add({**{k: v for k, v in row.items() if not k.startswith("_")}, "_src": row["_src"], "_git": row["_git"],
                       "_seed": row["_seed"], "_fit_seed": row["_fit_seed"], "_loc": dict(row["_loc"]), "_derived": {}})
    groups: dict = {}
    for r in T.rows:
        groups.setdefault((r["model"], r["ablation"], r["variant"], r["k"]), []).append(r)
    for (model, abl, var, k), rs in groups.items():
        rs = sorted(rs, key=lambda x: x["fit_seed"])
        out = {"model": model, "model_label": MODEL_LABEL.get(model, model), "ablation": abl, "variant": var, "k": k,
               "n_seeds": len(rs), "fit_seeds": ",".join(str(x["fit_seed"]) for x in rs),
               "_src": ";".join(sorted({x["_src"] for x in rs})), "_git": ";".join(sorted({str(x["_git"]) for x in rs})),
               "_seed": rs[0]["_seed"] if len({x["_seed"] for x in rs}) == 1 else None,
               "_fit_seed": ",".join(str(x["fit_seed"]) for x in rs), "_loc": {}, "_derived": {}}
        for name, col in (("top1", "top1"), ("drop", "drop_points"), ("flip", "flip_consistency")):
            m_, s_, n_ = agg([x.get(col) for x in rs])
            out[f"{name}_mean"], out[f"{name}_std"] = m_, s_
            if len(rs) == 1 and col in rs[0]["_loc"]:
                out["_loc"][f"{name}_mean"] = rs[0]["_loc"][col]
            else:
                out["_derived"][f"{name}_mean"] = f"mean of {col} over fit seeds {out['fit_seeds']}"
        notes = []
        if any(x["fit_seed"] == -1 for x in rs):
            notes.append("fit seed -1: " + (rs[0].get("fit_rows") or "pooled fit images"))
        reused = sorted({x["reused_from"] for x in rs if x["reused_from"]})
        if reused:
            notes.append("same model as " + ", ".join(reused) + " (result reused)")
        if any(x.get("params_exact") is False for x in rs):
            notes.append("projection not exact (params_exact = False)")
        out["note"] = "; ".join(notes)
        S.add(out)
    return T, S, X, files


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


def build_tier2(P, Tm: Table, Th: Table, m1: dict, x2: dict):
    T = Table("T_tier2", "Tier 2: ImageNet linear probe (runs/t2/t2_linear.csv)", ["model", "variant"],
              [("model", "model_label", ""), ("variant", "variant", ""), ("k", "k", "d"), ("top-1", "top1", ".2f"),
               ("top-5", "top5", ".2f"), ("drop", "drop_vs_original", ".2f"), ("vs SVD", "margin_svd", ".2f"),
               ("vs random", "margin_random", ".2f"), ("chosen wd", "chosen_wd", "g"), ("wd at grid edge", "wd_at_grid_edge", ""),
               ("k-NN check", "knn_check_top1", ".2f"), ("Tier-1 k-NN", "tier1_knn_top1", ".2f"),
               ("model params", "model_params", ",d")])
    T.notes = ["Logistic regression on [CLS, mean patch token] trained on the 100k bank, tested on all 50k validation "
               "images; weight decay chosen on a 10% split of the bank (scripts/tier2_linear.py). Drop: original minus "
               "variant; vs SVD and vs random: Mirror variant minus the SVD or random variant of the same family and k "
               "(empty where that baseline was not probed). k-NN check: k-NN on the probe's own cached CLS features; "
               "Tier-1 k-NN: the same variant's number from the k-NN tables (source in tier1_source)."]
    X = Table("T_tier2_transfer", "Tier 2: transfer linear probes (runs/t2/t2_transfer.csv)", ["model", "dataset", "variant"],
              [("model", "model_label", ""), ("dataset", "dataset", ""), ("variant", "variant", ""), ("k", "k", "d"),
               ("top-1", "top1", ".2f"), ("top-5", "top5", ".2f"), ("mean per-class", "mean_per_class_top1", ".2f"),
               ("drop", "drop_vs_original", ".2f"), ("vs SVD", "margin_svd", ".2f"), ("vs random", "margin_random", ".2f"),
               ("chosen wd", "chosen_wd", "g"), ("train / test", lambda r: f"{fmt(r.get('n_train'), ',d')} / "
                                                                         f"{fmt(r.get('n_test'), ',d')}", "")])
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
                   "label": r.get("label"), "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")),
                   "fit_seed": fs, "structure_source": ops, "source": P.rel(f), "git_hash": r.get("git_hash"),
                   "seed": py(r.get("seed")), "date": r.get("date"), "_src": P.rel(f), "_git": r.get("git_hash"),
                   "_seed": py(r.get("seed")), "_fit_seed": fs, "_loc": {}, "_derived": {}}
            for c in ("top1", "top5", "mean_per_class_top1", "knn_check_top1", "chosen_wd", "wd_at_grid_edge",
                      "holdout_top1_at_chosen", "final_hit_iter_limit", "model_params", "converted_params_kept",
                      "converted_params_dense", "n_bank", "n_val", "feature_dim", "feature_kind"):
                if c in r:
                    row[c] = py(r[c])
                    if c not in ("wd_at_grid_edge", "final_hit_iter_limit", "feature_kind"):
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
            T.add(row)
        _t2_margins(T, ("model",))
    f2 = P.runs / "t2" / "t2_transfer.csv"
    dx = read_csv(f2)
    if dx is not None:
        for _, r in dx.iterrows():
            model, var, ds = str(r["model"]), str(r["variant"]), str(r["dataset"])
            row = {"model": model, "model_label": MODEL_LABEL.get(model, model), "dataset": ds, "variant": var,
                   "label": r.get("label"), "family": r.get("family"), "method": r.get("method"), "k": py(r.get("k")),
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
    return T, X


M2_TO_MAIN = {"vlocal": "vlocal", "vexact": "vexact", "mlp_only": "mirror_mlp_m1", "svd": "svd_attn",
              "random": "random_attn", "dense": "original", "original": "original"}


def build_latency(P, Tm: Table, archs: dict):
    L = Table("T_latency", f"M2: measured latency against analytic cost (runs/{P.m2}/m2_latency.csv)",
              ["model", "variant", "k", "batch", "mode"],
              [("variant", "variant", ""), ("k", "k", "d"), ("batch", "batch", "d"), ("mode", "mode_used", ""),
               ("median ms", "latency_ms_median", ".3f"), ("img/s", "throughput_img_s", ",.0f"),
               ("speed-up vs dense", "speedup_vs_dense", ".3f"), ("vs original", "speedup_vs_original", ".3f"),
               ("analytic (M2)", "flop_speedup_vs_dense", ".3f"), ("analytic (collector)", "collector_flop_speedup", ".3f"),
               ("measured / analytic", "measured_over_analytic", ".3f"), ("GFLOPs", "gflops", ".2f"),
               ("params", "params_model", ",d"), ("checks", "checks", "")])
    L.notes = ["Median of the timed forwards (bf16, CUDA events; scripts/m2_latency.py). Speed-up vs dense: dense "
               "latency over the variant's, same batch and mode; dense is the folded original in the same efficient "
               "implementation, original the hub model as released. Analytic (M2): dense multiply-adds over the "
               "variant's, counted by m2_latency.py; analytic (collector): the same ratio from tables/cost_model.md "
               "(whole model); measured / analytic: measured speed-up over the M2 analytic one. Checks: float64, float32 "
               "and bfloat16 agreement of the efficient model with the dense model it was built from "
               "(T_latency_checks)."]
    C = Table("T_latency_checks", "M2 checks: efficient model against the dense model it was built from",
              ["variant", "k", "dtype"],
              [("variant", "variant", ""), ("k", "k", "d"), ("dtype", "dtype", ""), ("criterion", "criterion", ""),
               ("value", "value", ".2e"), ("pass", "pass", ""), ("noise floor", "noise_floor_unit", ".2e")])
    base = P.runs / P.m2
    lat, chk, meta = read_csv(base / "m2_latency.csv"), read_csv(base / "m2_checks.csv"), read_json(base / "m2_meta.json")
    mgit = ((meta or {}).get("git") or {}).get("git_hash")
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
            mk = M2_TO_MAIN.get(var)
            arch = archs.get(model)
            tm = Tm.find(model=model, k=0 if var in ("dense", "original") else k, variant=mk) if mk else None
            if arch and tm is not None and tm.get("gflops_model"):
                row["collector_flop_speedup"] = analytic_cost(arch, [])["gflops_model_dense"] / tm["gflops_model"]
                row["_derived"]["collector_flop_speedup"] = (f"dense GFLOPs / GFLOPs of T_main_zeroshot {mk} at k = {tm['k']} "
                                                             "(analytic_cost)")
            st = status.get((var, k))
            if st:
                row["checks_pass"] = all(p for _, p in st)
                row["checks"] = " / ".join(f"{d}: {'pass' if p else 'FAIL'}" for d, p in st)
            L.add(row)
    return L, C, meta


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
                        linewidth=2, capsize=3, label=VARIANT_INFO[key][1])
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
            L.append(f"- {MODEL_LABEL[model]}, Mirror MLP only, top-1 (%): original "
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
        L.append(f"- DINOv2-reg, Mirror MLP only at k = 6: {c('main.reg.k6.mirror_mlp.top1', 'T_main_zeroshot', {'model': 'dinov2_vitb14_reg', 'k': 6, 'variant': 'mirror_mlp'}, 'top1_mean')}"
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
    gm = [m for m in ("dinov2_vitb14_reg", "dinov2_vitb14", "deit3_b16", "mae_b16", "siglip2_b16")
          if not any(r["model"] == m for r in tg.rows)]
    if gm:
        L.append(f"- {NOT_RUN}: {', '.join(MODEL_LABEL[m] for m in gm)}.")
    rs = reasons(["DINOv2-S/L"])
    if rs:
        L.append("- DINOv2-S and L (nice-to-have): " + " | ".join(rs) + ".")
    L.append("- Tables: `T_generality`, `T_generality_p4`; figures `fig_eA_vs_depth.png`, `fig_structure_summary.png`.")
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
                    bits.append(f"{VARIANT_INFO[v][1]} {c(f'm1.k{k}.{v}.top1', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'top1_mean', headline=hl)} "
                                f"(flip {c(f'm1.k{k}.{v}.flip', 'T_main_zeroshot', {'model': m1['model'], 'k': k, 'variant': v}, 'flip_mean')})")
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
    rs = reasons(["M3 grid"])
    L.append("- M3 grid: " + NOT_RUN + (" (" + " | ".join(rs) + ")" if rs else "") + ". Tables `T_healing`, `T_healing_p3`; "
             "figure `fig_healing_curves.png`.")
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
            rows = [r for r in tas.rows if r["ablation"] == abl]
            bits = []
            for r in rows:
                ms = MODEL_SHORT.get(r["model"], r["model"])
                key = f"a.{ms}.{abl}.{r['variant']}.k{r['k']}"
                wh = {"model": r["model"], "ablation": abl, "variant": r["variant"], "k": r["k"]}
                b = f"{r['variant']} {c(key + '.top1', 'T_ablations_seeds', wh, 'top1_mean')}"
                if r.get("top1_std") is not None:
                    b += f" ± {c(key + '.top1_std', 'T_ablations_seeds', wh, 'top1_std')}"
                bits.append(b + f" (fit seeds {r['fit_seeds']})")
            ks = sorted({r["k"] for r in rows})
            models = sorted({r["model_label"] for r in rows})
            L.append(f"- {abl} ({', '.join(models)}, k = {', '.join(map(str, ks))}), k-NN top-1 (%), mean over fit seeds: "
                     + "; ".join(bits) + ".")
        if tax.rows:
            bits = []
            for r in tax.rows:
                rv, rf = r["variant"], r["fit_seed"]
                ms = MODEL_SHORT.get(r["model"], r["model"])
                wh = {"model": r["model"], "variant": r["variant"], "k": r["k"], "fit_seed": r["fit_seed"]}
                bits.append(f"{r['variant']} (fit seed {r['fit_seed']}) "
                            f"{c(f'a.{ms}.A2x.{rv}.s{rf}.top1', 'T_ablations_a2_extra', wh, 'top1')}"
                            f" [{r['pairing']}{'; projection: ' + r['projection'] if r.get('projection') else ''}; exact "
                            f"projection: {fmt(r.get('params_exact'))}]")
            L.append("- A2 pairings hungarian, hungarian_cycle and generic, with the pairing and projection each row records: " + "; ".join(bits) + ".")
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
                seeds = {r["fit_seed"] for r in rr if r["fit_seed"] is not None and r["fit_seed"] >= 0}
                if not want_k <= ks or len(seeds) < want_n:
                    gaps.append(f"{abl} has k = {', '.join(map(str, sorted(ks)))} and fit seeds "
                                f"{', '.join(map(str, sorted(seeds)))}")
            if gaps:
                L.append(f"- Not yet run against the plan (k = {m[1]} and {m[2]}, {m[3]} seeds): " + "; ".join(gaps) + ".")
    else:
        rs = reasons(["A1 to A5", "A1/A2", "A1"])
        L.append(f"- {NOT_RUN}: no `runs/a/*/A*_s*.csv`." + (" " + " | ".join(rs) + "." if rs else ""))
    L.append("- A6 and A7: zero-shot from the M1 rows of `T_main_zeroshot`; latency from M2 (`T_latency`).")
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
                b += f" (Tier-1 k-NN {c(key + '.tier1_knn', 'T_tier2', wh, 'tier1_knn_top1')})"
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
    else:
        L.append(f"- Transfer probes (CIFAR-100, Flowers-102, Pets, DTD): {NOT_RUN} (no `runs/t2/t2_transfer.csv`).")
    L.append(f"- ADE20K linear segmentation: {NOT_RUN} (no result files in the repo).")
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
              "margin_svd_mean", "DINOv2-reg k = 6 Mirror MLP-only margin over SVD (points)"),
             ("main.plain.k6.mirror_mlp.margin_svd", "T_main_zeroshot", {"model": "dinov2_vitb14", "k": 6, "variant": "mirror_mlp"},
              "margin_svd_mean", "DINOv2 k = 6 Mirror MLP-only margin over SVD (points)"),
             ("main.reg.k6.mirror_mlp.top1_mean", "T_main_zeroshot", {"model": "dinov2_vitb14_reg", "k": 6, "variant": "mirror_mlp"},
              "top1_mean", "DINOv2-reg k = 6 Mirror MLP-only top-1"),
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
    T["T_ablations"], T["T_ablations_seeds"], T["T_ablations_a2_extra"], a_files = build_ablations(P)
    T["T_tier2"], T["T_tier2_transfer"] = build_tier2(P, T["T_main_zeroshot"], T["T_healing"], m1, x2)
    add_linear_to_healing(T["T_healing"], T["T_tier2"])
    T["T_latency"], T["T_latency_checks"], m2_meta = build_latency(P, T["T_main_zeroshot"], archs)
    T["T_f4"], T["T_f4_p2"], f4_files, _ = build_f4(P)
    T["T_prereg"] = build_prereg(P, e1, e1b, f3, d_free, T["T_healing_p3"], T["T_generality_p4"], x1, status, th,
                                 commits, f4)

    # ------------------------------------------------------------- external scripts (F1, F2)
    f1res, f2res = [], []
    f1_groups = (("f1_dinov2", "x2", ["perlayer_reg", "perlayer_plain"]),
                 ("f1_g", "g", ["deit3_b16", "mae_b16", "siglip2_b16"]))
    # f2_drift.py draws at most three models per figure: the paper pair (both DINOv2 models), then G.
    f2_groups = (("f2_dinov2", [r for r in gen_runs if r["model"] in MAIN_MODELS]),
                 ("f2_g", [r for r in gen_runs if r["model"] not in MAIN_MODELS]))
    # Skipped scripts: reuse outputs of an earlier collection (marked "reused"), so the package keeps F1 and F2,
    # and flag them as stale when runs or operator files are present now that they did not read.
    if not a.skip_scripts:
        for tag, sub, arms in f1_groups:
            runs_dir = P.runs / sub
            fr = run_f1(P, sdir / tag, runs_dir, arms) if runs_dir.is_dir() else {"status": NOT_RUN, "arms": arms,
                                                                                  "error": f"no {runs_dir}"}
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

    for sub, tag in (("f4", "f4"), (P.m2, "m2")):
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
        ("T_healing", "Healing pilot (M3)", ["T_healing", "T_healing_p3"]),
        ("T_generality", "Generality (G)", ["T_generality", "T_generality_p4"]),
        ("T_structure_per_layer", "Structure per layer (F1 style)", ["T_structure_summary"]),
        ("T_prereg", "Pre-registered checks", ["T_prereg"]),
        ("T_x_checks", "Stage 0 checks", ["T_x_checks"]),
        ("T_e1_history", "E1 and E1b (earlier protocol)", ["T_e1_history"]),
        ("T_attention", "Attention conversion (M1)", ["T_attention", "T_attention_vexact", "T_attention_conversion"]),
        ("T_ablations", "Ablations (A1 to A5)", ["T_ablations_seeds", "T_ablations", "T_ablations_a2_extra"]),
        ("T_tier2", "Tier 2: linear probes", ["T_tier2", "T_tier2_transfer"]),
        ("T_latency", "M2: latency and analytic cost", ["T_latency", "T_latency_checks"]),
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
        elif name == "T_main_zeroshot":
            lines += T["T_main_zeroshot"].md_lines()
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
    for n in sorted(g):
        inv_add(f"runs/g/{n}/results/e1_meta.json", "G run (meta)")
        inv_add(f"runs/g/{n}/results/e1_surgery.csv", "G run (surgery)")
    for f in ("m1_knn.csv", "m1_meta.json", "m1_ranks.csv", "m1_heads.csv", "m1_vexact.csv", "m1_vexact_blocks.csv",
              "m1_exactness.csv", "m1_flipcheck.json"):
        inv_add(f"runs/m1/{f}", "M1")
    for mth in ("mirror", "svd", "random"):
        inv_add(f"runs/m3_pilot/{mth}/knn.json", f"M3 pilot {mth}: final k-NN")
        inv_add(f"runs/m3_pilot/{mth}/final/metrics.json", f"M3 pilot {mth}: healing metrics")
    inv_add("runs/d/d_datafree.json", "D (P5)")
    inv_add("runs/f3/f3_gate_G3.json", "F3 (P1, G3)")
    inv_add("runs/f4/f4_p2.json", "F4 (P2)")
    fsrc = P.root / "scripts" / "f4_rule.py"
    mm_ = re.search(r"PREREG_MODELS = \(([^)]*)\)", fsrc.read_text(encoding="utf-8")) if fsrc.is_file() else None
    for m_ in re.findall(r'"([^"]+)"', mm_[1]) if mm_ else []:
        inv_add(f"runs/f4/f4_{m_}.csv", "F4 per-block rows (P2 model)")
    inv_add("runs/a", "A1 to A5 ablations")
    for f in ("m2_latency.csv", "m2_checks.csv", "m2_meta.json"):
        inv_add(f"runs/{P.m2}/{f}", "M2 latency")
    inv_add("runs/t2/t2_linear.csv", "Tier 2 ImageNet linear probe")
    inv_add("runs/t2/t2_transfer.csv", "Tier 2 transfer probes")
    for r in list(x2.values()) + list(g.values()):
        if r["operator_mode"] == "per_layer" and r["status"] != NOT_RUN:
            inv_add(P.rel(P.artifacts / r["res_rel"] / "e1_operators.npz") if npz_for(P, r) is None or
                    not (r["res"] / "e1_operators.npz").is_file() else P.rel(r["res"] / "e1_operators.npz"),
                    "operators npz (F1 histograms, F2)")
    inv_add(P.rel(P.artifacts / "runs" / "m1" / "m1_structure.npz"), "M1 structure npz (not needed by the tables)")
    ctx = {"tables": T, "status": status, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "commits": commits,
           "e1": e1, "e1b": e1b, "x1": x1, "x1b": x1b, "x2": x2, "g": g, "x4": x4, "m1": m1, "m3": m3, "d": d_free,
           "f4": f4, "f1": f1res, "f2": f2res, "a_files": a_files, "inventory": inv, "deviations": dev,
           "plan_text": plan_text, "m2_meta": m2_meta, "f4_files": f4_files, "figs_dir": fdir}

    # ------------------------------------------------------------- INDEX and numbers
    R = Registry()
    index = build_index(P, ctx, R)
    write_text(out / "INDEX.md", index)
    (out / "numbers.json").write_text(json.dumps(R.items, indent=2, default=str), encoding="utf-8")
    meta = {"written_by": "scripts/collect_results.py", "date": ctx["date"], "git": git, "args": vars(a),
            "root": str(P.root), "artifacts": str(P.artifacts), "seed": None,
            "seed_note": "no randomness in the collector; F2 null draws use f2_drift.py's own --seed (default 0)",
            "prereg_thresholds": th, "prereg_commits": commits, "figures": figs,
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

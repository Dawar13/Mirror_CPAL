#!/usr/bin/env python
"""Next-phase results collector: every pre-registered prediction scored mechanically, every cited number with its
source (PROJECT_NEXT_PHASE.md Section 12; PREREG_NEXT_PHASE.md, signed off and committed).

    python scripts/collect_next_phase.py                      # reads results/ and runs/ of the repository
    python scripts/collect_next_phase.py --root <dir>         # another tree with the same layout (tests)

CPU only, no randomness, nothing typed by hand. The script reads the output folders of the new-phase scripts, computes
each scored quantity with the operational definition of PREREG_NEXT_PHASE.md, and records PASS, FAIL or PENDING
(inputs missing or incomplete) with the value, the bar, n, the source files and the git commit of the run that wrote
them (from its meta.json), and the pre-registration commit (the commit that signed PREREG_NEXT_PHASE.md off). Upside
items are REPORTED, never scored. Each prediction carries a quotation of its prereg wording; the collector checks that
the quotation still appears in PREREG_NEXT_PHASE.md and flags it when it does not.

Outputs:
  <root>/FINAL_RESULTS.md      gates A to E; every prediction with verdict and number; details per model, seed or seed
                               pair; every number the paper will cite with its source; failed checks (new phase, and
                               the earlier phase from results/paper/numbers.json, including F4 and the weights-only
                               pairing); earlier-phase numbers as audited by MUST-00
  <root>/PAPER_NUMBERS.json    the same content, machine-readable
  <root>/results/paper_next/data/   tidy CSV (and copied npz) inputs of scripts/make_figures_next.py, one file per
                               figure or table, each row with its source file and git hash

Inputs (missing ones make the dependent predictions PENDING; see each scorer below for the exact files and keys):
  MUST-01 results/MUST-01/{metrics.json,law_per_layer.csv,meta.json}           (scripts/must01_law.py)
  MUST-02 results/MUST-02/<model>/<variant>/metrics.json, must02_components.csv  (scripts/must02_exactness.py)
  MUST-03 results/MUST-04/dinov2_vitb14_reg/metrics.json reference_check       (scripts/measure_groups.py)
  MUST-04 results/MUST-04/<model>/{metrics.json,blocks.csv,meta.json}          (scripts/measure_groups.py)
  MUST-05 results/MUST-05/<model>/{metrics.json,switch_pairs.csv,teaser.npz}   (scripts/must05_twins.py)
  MUST-06 results/MUST-06/<model>/{metrics.json,pairs.csv,spectra.csv}         (scripts/must06_visibility.py)
  MUST-07 results/MUST-07/{rows_<model>.csv,bootstrap.csv,gate_b.json}         (scripts/must07_compressors.py)
  MUST-08 results/MUST-08/rows_<model>.csv                                     (scripts/must08_components.py)
  MUST-09 runs/MUST-09/<model>/<method>/s<seed>/metrics.json (summary.csv, latency.csv in results/MUST-09/)
  MUST-10 results/MUST-10/flipavg.csv                                          (scripts/must10_flipavg.py)
  MUST-11 results/MUST-11/t2/{t2_linear.csv,t2_transfer.csv}, t2_seg/t2_ade20k.csv (+ seg/<model>/*_maskeq.json)
  MUST-12 results/MUST-12/<run>/e<NNN>/metrics.json, gate_c.json; runs/MUST-12/<run>/meta.json
  CORE-01 results/CORE-01/<run>/e<NNN>/metrics.json; runs/CORE-01/<run>/{meta.json,val_acc.csv}
  CORE-02 results/CORE-02/{predictions.json,surgery.csv,law_per_block.csv}  (scripts/core02_refund.py)
  CORE-03 results/CORE-03/{curves.csv,p31.json}                           (scripts/core03_stacking.py)
  CORE-04 results/CORE-04/structure/<model>/metrics.json (MUST-04 reused), scale/knn.csv (scripts/core04_generality.py)
  CORE-05 results/CORE-05/{measures_<model>.csv,ckpt_bands.csv,rule_score.json}  (scripts/core05_why_cheap.py)
  CORE-06 results/CORE-06/probes.csv                                      (scripts/core06_orientation.py)
  CORE-07 results/CORE-07/shift_rows.csv                                  (scripts/core07_shift.py)
  UP-01, UP-03 results/UP-0x/<run>/e<NNN>/metrics.json; UP-02, UP-04 results/UP-0x/<run>/summary.json
          (scripts/train_controlled_sym.py); UP-06 results/UP-06/comparison.json; UP-07 results/UP-07/up07.json;
          UP-05 results/UP-05/metrics.json (no script yet). Reported, not scored.
  Earlier phase: results/paper/numbers.json, results/paper/tables/T_prereg.csv, results/MUST-00/audit.json.

Conventions fixed here (all listed in FINAL_RESULTS.md under "Scoring conventions"): "within x points" is read
literally as |difference| <= x; an exact-invariance claim passes at a raw 100% or, under the rounding allowance of
PREREG_NEXT_PHASE.md Section 1, when every disagreement has a recorded relative margin below 1e-4 (margins not
recorded: PENDING); a claim that must hold "on all" models, runs or seed pairs is FAIL as soon as one available unit
fails, PASS when every unit is present and passes, else PENDING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, PENDING, REPORTED = "PASS", "FAIL", "PENDING", "REPORTED"
MET, NOT_MET = "MET", "NOT MET"
PREREG_FILE = "PREREG_NEXT_PHASE.md"
SIGNOFF_MARKER = "ACTIVE: signed off"
MAIN = "dinov2_vitb14_reg"
MUST04_MODELS = ("dinov2_vitb14_reg", "dinov2_vitb14", "dinov2_vitl14_reg", "deit3_b16")
MUST06_MODELS = ("dinov2_vitb14_reg", "dinov2_vitl14_reg", "deit3_b16")
MUST07_MODELS = ("dinov2_vitb14_reg", "dinov2_vitl14_reg", "deit3_b16")
BASELINES = ("svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit", "pair_actcorr", "pair_wcos", "pair_random",
             "random_structure")                     # MUST-07 operational: "every baseline" = methods 3 to 12
METHOD_ORDER = ("mirror_exact", "mirror_local") + BASELINES
METHOD_LABEL = {"original": "Original model", "mirror_exact": "Mirror-exact", "mirror_local": "Mirror-local (ablation)",
                "svd": "Plain SVD", "svdllm": "SVD-LLM-style whitening", "asvd_scale": "ASVD-style scaling",
                "slicegpt": "SliceGPT-style slicing", "flap": "FLAP-style pruning", "zipit": "ZipIt-style merging",
                "pair_actcorr": "Pairing by same-image correlation", "pair_wcos": "Pairing by weight cosine",
                "pair_random": "Random pairing", "random_structure": "Random structure"}
SIX_SIZES = ("6mlp", "6both")
SEEDS3 = (0, 1, 2)
CORE01_RUNS = {"none": (0, 1, 2), "LR": (0, 1, 2), "UD": (0, 1, 2), "LRUD": (0, 1), "R90": (0, 1)}
CORE_ARCH = "vit_s8"
# Group elements (for "wrong group" in CORE-02): a surgery group is wrong for a model unless every one of its elements
# is in the group the model was trained with.
GROUP_ELEMENTS = {"none": {"e"}, "LR": {"e", "lr"}, "UD": {"e", "ud"}, "R180": {"e", "r2"},
                  "R90": {"e", "r1", "r2", "r3"}, "Klein": {"e", "lr", "ud", "r2"}}
TRAINED_GROUP = {"none": "none", "LR": "LR", "UD": "UD", "LRUD": "Klein", "R90": "R90"}
MARGIN_TOL = 1e-4                                   # PREREG Section 1, rounding allowance
MARGIN_KEYS = ("max_margin_rel", "max_vote_margin_rel", "max_rel_margin", "disagree_margin_rel_max",
               "max_rel_margin_of_disagreements", "max_margin_rel_of_differing_pixels",
               "max_rel_margin_of_differing_pixels")
MARGIN_LIST_KEYS = ("margins_rel", "disagree_margins_rel", "vote_margins_rel")
N_VAL = 50000
_YAML_CACHE: dict = {}


def log(msg: str) -> None:
    print(f"[collect_next {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================
# Small helpers
# ================================================================================================
def fnum(v):
    """float, or None for missing, non-numeric or non-finite values."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return o.as_posix()
    return o


def fmt(v, spec: str | None = None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(int(v)) >= 10000 else str(int(v))
    if isinstance(v, str):
        return v
    f = fnum(v)
    if f is None:
        return "n/a"
    if spec:
        return format(f, spec)
    if f == 0:
        return "0"
    if abs(f) < 1e-3:
        return f"{f:.2e}"
    if abs(f) >= 1:
        return f"{f:.2f}"
    return f"{f:.4f}"


def short(h) -> str:
    return str(h)[:10] if h else "not recorded"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def git(root: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def combine(verdicts) -> str:
    """All-of rule: FAIL if any unit fails; PASS if every unit is present and passes; else PENDING."""
    vs = list(verdicts)
    if any(v == FAIL for v in vs):
        return FAIL
    if vs and all(v == PASS for v in vs):
        return PASS
    return PENDING


def cmp(value, op: str, bar: float) -> str:
    v = fnum(value)
    if v is None:
        return PENDING
    ok = {">=": v >= bar, ">": v > bar, "<=": v <= bar, "<": v < bar}[op]
    return PASS if ok else FAIL


def within(a, b, tol: float) -> str:
    a, b = fnum(a), fnum(b)
    if a is None or b is None:
        return PENDING
    return PASS if abs(a - b) <= tol else FAIL


def find_margin(d) -> float | None:
    """Largest relative top-two margin among disagreements, if a producing script recorded it."""
    if not isinstance(d, dict):
        return None
    for k in MARGIN_KEYS:
        v = fnum(d.get(k))
        if v is not None:
            return v
    for k in MARGIN_LIST_KEYS:
        v = d.get(k)
        if isinstance(v, list) and v:
            vals = [fnum(x) for x in v]
            vals = [x for x in vals if x is not None]
            if vals:
                return max(vals)
    return None


def exact_invariance(share, n_disagree, src) -> tuple[str, str, float | None]:
    """PREREG Section 1: 100% raw, or every disagreement a near-tie (relative margin < 1e-4)."""
    s = fnum(share)
    if s is None:
        return PENDING, "share not recorded", None
    nd = None if n_disagree is None else int(n_disagree)
    if s >= 100.0 or nd == 0:
        return PASS, "100% raw (no disagreement)", None
    m = find_margin(src)
    nd_txt = f"{nd} disagreement(s)" if nd is not None else "disagreements"
    if m is None:
        return PENDING, (f"{fmt(s, '.4f')}% raw, {nd_txt}; their top-two margins are not recorded, so the rounding "
                         "allowance (PREREG Section 1) cannot be applied"), None
    if m < MARGIN_TOL:
        return PASS, f"{fmt(s, '.4f')}% raw, {nd_txt}, all near-ties (largest relative margin {m:.2e} < 1e-4)", m
    return FAIL, f"{fmt(s, '.4f')}% raw, {nd_txt}; largest relative margin {m:.2e} is not a near-tie (>= 1e-4)", m


def mean_std(vals):
    v = [fnum(x) for x in vals]
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def latest_epoch_dir(d: Path) -> tuple[int | None, Path | None]:
    eps = []
    if d.is_dir():
        for x in d.iterdir():
            m = re.fullmatch(r"e(\d+)", x.name)
            if m and (x / "metrics.json").is_file():
                eps.append((int(m.group(1)), x))
    if not eps:
        return None, None
    return max(eps)


# ================================================================================================
# Records
# ================================================================================================
@dataclass
class Pred:
    id: str
    experiment: str
    phase: str
    statement: str
    quote: str | None
    quantity: str
    bar: str
    verdict: str = PENDING
    value: object = None
    n: object = None
    reason: str = ""
    details: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    quote_found: bool | None = None
    prereg_commit: str | None = None


@dataclass
class Gate:
    gate: str
    when: str
    condition: str
    status: str = PENDING
    value: str = ""
    evidence: str = ""
    sources: list = field(default_factory=list)


class Collector:
    def __init__(self, root: Path, paper_dir: Path):
        self.root = Path(root).resolve()
        self.paper_dir = Path(paper_dir)
        self.inputs: dict[str, str] = {}
        self.numbers: dict[str, dict] = {}
        self.data: dict[str, pd.DataFrame] = {}
        self.copies: dict[str, Path] = {}
        self.preds: list[Pred] = []
        self.gates: list[Gate] = []
        self.notes: list[str] = []
        self._jcache: dict = {}
        self.prereg = self._prereg_info()

    # ---------------------------------------------------------------- paths and reading
    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def rel(self, p) -> str:
        p = Path(p)
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            pass
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def _seen(self, p: Path) -> None:
        r = self.rel(p)
        if r not in self.inputs:
            try:
                self.inputs[r] = sha256_file(p)
            except OSError:
                self.inputs[r] = "unreadable"

    def jload(self, p) -> dict | None:
        p = Path(p)
        if p in self._jcache:
            return self._jcache[p]
        if not p.is_file():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            self.notes.append(f"cannot read {self.rel(p)}: {e}")
            obj = None
        else:
            self._seen(p)
        self._jcache[p] = obj
        return obj

    def cload(self, p) -> pd.DataFrame | None:
        p = Path(p)
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"cannot read {self.rel(p)}: {e}")
            return None
        self._seen(p)
        return df if len(df) else None

    def meta_git(self, p) -> str | None:
        """Git hash of the run that wrote `p`: meta.json in its folder or a parent folder (up to results/ or runs/),
        else a git hash inside the file itself."""
        p = Path(p)
        d = p if p.is_dir() else p.parent
        stop = {self.p("results"), self.p("runs"), self.root}
        while True:
            m = d / "meta.json"
            if m.is_file():
                j = self.jload(m) or {}
                note = self.prereg_hash_check(m)
                if note and note not in self.notes:
                    self.notes.append(note)
                g = (j.get("git") or {}).get("git_hash") if isinstance(j.get("git"), dict) else j.get("git_hash")
                if g:
                    return g
            if d in stop or d.parent == d:
                break
            d = d.parent
        if p.is_file() and p.suffix == ".json":
            j = self.jload(p) or {}
            g = j.get("git")
            if isinstance(g, dict) and g.get("git_hash"):
                return g["git_hash"]
            if j.get("git_hash"):
                return j["git_hash"]
        return None

    def src(self, p, git_hash=None) -> dict:
        return {"file": self.rel(p), "git_hash": git_hash if git_hash is not None else self.meta_git(p)}

    def num(self, key: str, value, *, desc: str, source, n=None, unit: str = "", seed=None) -> None:
        if isinstance(value, (bool, np.bool_)) or isinstance(value, str):
            v = value
        elif isinstance(value, (int, np.integer)):
            v = int(value)
        else:
            v = fnum(value)
            if v is None:
                return
        srcs = source if isinstance(source, list) else [source]
        self.numbers[key] = {"value": jsonable(v), "description": desc, "unit": unit, "n": jsonable(n),
                             "seed": jsonable(seed), "sources": srcs}

    def gates_append(self, g: Gate) -> Gate:
        self.gates.append(g)
        return g

    def add(self, pr: Pred) -> Pred:
        pr.prereg_commit = self.prereg["signoff_commit"]
        if pr.quote:
            pr.quote_found = norm_ws(pr.quote) in self.prereg["text_norm"] if self.prereg["text_norm"] else None
        self.preds.append(pr)
        return pr

    # ---------------------------------------------------------------- pre-registration
    def _prereg_info(self) -> dict:
        path = self.p(PREREG_FILE)
        base = self.root
        if not path.is_file():
            path, base = REPO_ROOT / PREREG_FILE, REPO_ROOT
        if not path.is_file():
            return {"file": PREREG_FILE, "sha256": None, "signoff_commit": None, "last_commit": None, "text_norm": "",
                    "signed_off": False, "read_from": None}
        text = path.read_text(encoding="utf-8")
        log_out = git(base, "log", "--format=%H", f"-S{SIGNOFF_MARKER}", "--", PREREG_FILE)
        commits = [x for x in (log_out or "").split() if x]
        return {"file": PREREG_FILE, "sha256": sha256_file(path), "signoff_commit": commits[-1] if commits else None,
                "last_commit": git(base, "log", "-1", "--format=%H", "--", PREREG_FILE),
                "text_norm": norm_ws(text), "signed_off": text.lstrip().startswith("# ACTIVE: signed off"),
                "read_from": path.as_posix()}

    def prereg_hash_check(self, meta_path) -> str | None:
        """Note when a run's meta.json records another PREREG_NEXT_PHASE.md text (sha256) than the current file."""
        j = self.jload(meta_path) or {}
        pr = j.get("prereg") or {}
        sha = pr.get("sha256")
        if sha and self.prereg["sha256"] and sha != self.prereg["sha256"]:
            return (f"{self.rel(meta_path)} records a PREREG_NEXT_PHASE.md text with sha256 {sha[:12]}, not the current "
                    f"{self.prereg['sha256'][:12]}")
        return None


# ================================================================================================
# MUST-01: the law
# ================================================================================================
def score_must01(c: Collector) -> None:
    pr = c.add(Pred(
        "MUST-01", "MUST-01", "Must",
        "Predicted and achieved parameter savings agree to within 1 percentage point per layer, for all 7 measured models",
        "predicted and achieved parameter savings agree to within 1 percentage point per layer, for all 7 measured models",
        "largest |predicted - achieved| saving over every layer (fc1, fc2) and block (MLP), in percentage points",
        "<= 1.0 pp, 7 models"))
    d = c.p("results", "MUST-01")
    df = c.cload(d / "law_per_layer.csv")
    met = c.jload(d / "metrics.json") or {}
    if df is None:
        pr.reason = "results/MUST-01/law_per_layer.csv missing"
        return
    cols = [x for x in ("diff_pp_fc1", "diff_pp_fc2", "diff_pp_mlp") if x in df]
    if not cols:
        pr.reason = "no diff_pp_* column in law_per_layer.csv"
        return
    g = c.meta_git(d / "meta.json")
    pr.sources = [c.src(d / "law_per_layer.csv", g), c.src(d / "metrics.json", g)]
    rows = []
    for model, sub in df.groupby("model", sort=True):
        mx = float(sub[cols].abs().to_numpy().max())
        struct = sorted(set(map(str, sub["structure"]))) if "structure" in sub else []
        rows.append({"model": model, "n_blocks": int(len(sub)), "fit_seeds": sorted(set(int(s) for s in sub["fit_seed"]))
                     if "fit_seed" in sub else [], "max_abs_diff_pp": mx, "structure": "; ".join(struct),
                     "verdict": cmp(mx, "<=", 1.0)})
    pr.details = rows
    value = max(r["max_abs_diff_pp"] for r in rows)
    n_models = len(rows)
    pr.value, pr.n = value, f"{len(df)} blocks, {n_models} models"
    if value > 1.0:
        pr.verdict = FAIL
    elif n_models >= 7:
        pr.verdict = PASS
    else:
        pr.verdict = PENDING
        pr.reason = f"only {n_models} of 7 models in the file"
    synth = [r["model"] for r in rows if "synthetic" in r["structure"]]
    if synth:
        pr.reason = (pr.reason + "; " if pr.reason else "") + (
            "structure synthetic with the recorded counts (arithmetic check only) for " + ", ".join(synth))
    if met.get("tol_pp") is not None and fnum(met.get("tol_pp")) != 1.0:
        pr.reason += f"; the run used tol_pp {met.get('tol_pp')}, the bar here is the prereg's 1.0"
    c.num("must01.max_abs_diff_pp", value, desc="MUST-01: largest |predicted - achieved| saving, any layer, any model",
          source=pr.sources[0], n=len(df), unit="percentage points")
    c.num("must01.n_blocks", int(len(df)), desc="MUST-01: blocks compared (all models and fit seeds)",
          source=pr.sources[0])
    c.data["appendix_law"] = pd.DataFrame(rows).assign(source_file=pr.sources[0]["file"], git_hash=g)


# ================================================================================================
# MUST-02: exactness
# ================================================================================================
def must02_variants(c: Collector, model: str) -> list[tuple[Path, dict]]:
    out = []
    base = c.p("results", "MUST-02", model)
    if base.is_dir():
        for d in sorted(base.iterdir()):
            m = c.jload(d / "metrics.json")
            if m:
                out.append((d, m))
    return out


def score_must02(c: Collector) -> None:
    pr = c.add(Pred(
        "MUST-02", "MUST-02", "Must",
        "On 1,000 validation images in fp32, every component of Mirror-exact (full model) has maximum relative "
        "equivariance error below 1e-5, and flip consistency is 100%",
        "every component of Mirror-exact (full model) has maximum relative equivariance error below 1e-5, and flip "
        "consistency is 100%",
        "max over images and components of ||T out(x) - out(mirror x)|| / ||out(mirror x)|| (fp32, zero-shot, fit seed "
        "0); k-NN flip consistency with fp32 features on all 50,000 validation images (rounding allowance)",
        "< 1e-5 and 100%"))
    vs = must02_variants(c, MAIN)
    if not vs:
        pr.reason = f"results/MUST-02/{MAIN}/<variant>/metrics.json missing"
        return
    rows, comp_frames = [], []
    for d, m in vs:
        s = m.get("summary") or {}
        fl = m.get("flip_check") or {}
        dtype = str(m.get("dtype", ""))
        mx = fnum(s.get("max_rel_tensor"))
        comp_v = cmp(mx, "<", 1e-5) if mx is not None else PENDING
        fv, freason, margin = exact_invariance(fl.get("flip_consistency"), fl.get("n_disagree"), fl) if fl else \
            (PENDING, "no flip check in this run", None)
        nq = fl.get("n_query")
        if fl and "bf16" in str(fl.get("precision", "")):
            fv, freason = PENDING, "flip check not in fp32"
        if fl and nq is not None and int(nq) < N_VAL and fv == PASS:
            fv, freason = PENDING, f"flip check on {nq} queries, not all {N_VAL:,} validation images"
        g = c.meta_git(d / "metrics.json") or m.get("git_hash")
        rows.append({"variant": d.name, "dtype": dtype, "max_rel_tensor": mx, "worst_component": s.get("worst_component"),
                     "max_rel_token": fnum(s.get("max_rel_token")), "n_components": s.get("n_components"),
                     "components_verdict": comp_v, "flip_consistency_fp32": fnum(fl.get("flip_consistency")),
                     "n_disagree": fl.get("n_disagree"), "n_query": nq, "flip_verdict": fv, "flip_note": freason,
                     "flip_consistency_bf16": fnum((m.get("flip_check_bf16") or {}).get("flip_consistency")),
                     "n_images": m.get("n_images"), "file": c.rel(d / "metrics.json"), "git_hash": g})
        cdf = c.cload(d / "must02_components.csv")
        if cdf is not None:
            comp_frames.append(cdf.assign(source_file=c.rel(d / "must02_components.csv"), git_hash=g))
    pr.details = rows
    zs = [r for r in rows if r["variant"].startswith("zeroshot") and r["dtype"] == "float32"]
    if not zs:
        pr.reason = "no zero-shot float32 variant (results/MUST-02/<model>/zeroshot)"
        return
    z = zs[0]
    pr.sources = [{"file": z["file"], "git_hash": z["git_hash"]}]
    pr.verdict = combine([z["components_verdict"], z["flip_verdict"]])
    pr.value = f"max rel. error {fmt(z['max_rel_tensor'])} ({z['worst_component']}); flip {fmt(z['flip_consistency_fp32'], '.4f')}%"
    pr.n = f"{z['n_images']} images, {z['n_components']} components; flip on {z['n_query']} queries"
    pr.reason = f"flip check: {z['flip_note']}"
    if z["n_images"] is not None and int(z["n_images"]) != 1000:
        pr.reason += f"; the audit used {z['n_images']} images, the prereg says 1,000"
    healed = [r for r in rows if r["variant"].startswith("healed") and r["dtype"] == "float32"]
    if healed:
        h = healed[0]
        pr.reason += (f"; healed seed-0 model against the same bar (reported): components {h['components_verdict']}, "
                      f"flip {h['flip_verdict']}")
    c.num("must02.zeroshot.max_rel_tensor", z["max_rel_tensor"], desc="MUST-02: largest relative equivariance error, "
          "any component, zero-shot Mirror-exact, fp32", source=pr.sources[0], n=z["n_images"])
    c.num("must02.zeroshot.flip_fp32", z["flip_consistency_fp32"], desc="MUST-02: k-NN flip consistency, fp32 features",
          source=pr.sources[0], n=z["n_query"], unit="%")
    c.num("must02.zeroshot.flip_bf16", z["flip_consistency_bf16"], desc="MUST-02: k-NN flip consistency, bf16 (reported "
          "alongside)", source=pr.sources[0], unit="%")
    if comp_frames:
        cdf = pd.concat(comp_frames, ignore_index=True)
        keys = [k for k in ("variant", "dtype", "stage", "component", "space") if k in cdf]
        if "rel_tensor_max" in cdf and keys:
            agg = cdf[cdf["in_span"].astype(str) == "True"] if "in_span" in cdf else cdf
            agg = agg.groupby(keys, dropna=False).agg(
                rel_tensor_max=("rel_tensor_max", "max"), rel_token_max=("rel_token_max", "max"),
                n_blocks=("rel_tensor_max", "size"), source_file=("source_file", "first"),
                git_hash=("git_hash", "first")).reset_index()
            c.data["appendix_exactness"] = agg


# ================================================================================================
# MUST-03 (acceptance check) and MUST-04
# ================================================================================================
def must04_metrics(c: Collector, model: str, exp: str = "MUST-04"):
    p = c.p("results", exp, model, "metrics.json")
    return p, c.jload(p)


def score_must03(c: Collector) -> None:
    pr = c.add(Pred(
        "MUST-03", "MUST-03", "Must",
        "Acceptance check of the generalised measurement code on dinov2_vitb14_reg with LR: the LR numbers of earlier "
        "runs are reproduced (twin pairs and breaking shares of block 1 within rounding of runs/x2/perlayer_reg_s0)",
        "the LR numbers of earlier runs are reproduced (twin pairs and breaking shares of block 1 within rounding of "
        "`runs/x2/perlayer_reg_s0`)",
        "block 1: twin pairs equal; |s_fc1 - s1_ref| and |s_fc2 - s2_ref| below 0.005 (half a unit of the 2-decimal "
        "values quoted, 0.27 and 0.49); the synthetic-network tests are the unit tests, not scored here",
        "pairs equal; |delta share| < 0.005"))
    p, m = must04_metrics(c, MAIN)
    rc = (m or {}).get("reference_check") or {}
    b1 = rc.get("block1")
    if not b1:
        pr.reason = f"no reference_check.block1 in results/MUST-04/{MAIN}/metrics.json"
        return
    pr.sources = [c.src(p)]
    dp = abs(int(b1["n_pairs"]) - int(b1["n_pairs_ref"]))
    d1, d2 = abs(float(b1["s_fc1"]) - float(b1["s1_ref"])), abs(float(b1["s_fc2"]) - float(b1["s2_ref"]))
    pr.value = f"pairs {b1['n_pairs']} vs {b1['n_pairs_ref']}; |ds_fc1| {d1:.2e}, |ds_fc2| {d2:.2e}"
    pr.n = f"reference {rc.get('reference')}"
    pr.verdict = PASS if (dp == 0 and d1 < 0.005 and d2 < 0.005) else FAIL
    pr.details = [{k: v for k, v in rc.items() if k != "block1"} | {"block1": b1}]
    c.num("must03.block1.n_pairs", int(b1["n_pairs"]), desc="MUST-03: block 1 LR twin pairs, new code (reference "
          f"{b1['n_pairs_ref']})", source=pr.sources[0])


def score_must04(c: Collector) -> None:
    common = "per model, fit seed 0; claim passes only if it holds on all four models (PREREG Section 9, item 2)"
    p1 = c.add(Pred("MUST-04 P4.1", "MUST-04", "Must",
                    "UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45)",
                    "P4.1: UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45).",
                    "ratio to null, blocks 1 to 6 (mean breaking share of fc1 and fc2 over the blocks / mean of the "
                    "null means); " + common, ">= 0.9 (UD and R90, every model)"))
    p2 = c.add(Pred("MUST-04 P4.2", "MUST-04", "Must",
                    "The UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval",
                    "P4.2: the UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval.",
                    "UD twin fraction (threshold 0.5), blocks 1 to 6, against the 2.5th and 97.5th percentiles of the "
                    "twin-fraction null (random patch permutation with the same cycle type); " + common,
                    "inside [null 2.5%, null 97.5%]"))
    p3 = c.add(Pred("MUST-04 P4.3", "MUST-04", "Must",
                    "Zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 "
                    "points, against 1.85 for LR",
                    "P4.3: zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 "
                    "points, against 1.85 for LR (78.94 minus 77.09).",
                    "original top-1 minus top-1 after UD surgery (per-layer operators, MLP only, blocks 1 to 6), "
                    "validation k-NN; " + common, "> 30 points"))
    rows, v1, v2, v3, blocks = [], [], [], [], []
    for model in MUST04_MODELS:
        p, m = must04_metrics(c, model)
        if not m:
            v1.append(PENDING)
            v2.append(PENDING)
            v3.append(PENDING)
            rows.append({"model": model, "status": "missing"})
            continue
        s = c.src(p)
        for pr in (p1, p2, p3):
            pr.sources.append(s)
        sc = m.get("scores") or {}
        r = {"model": model}
        for g in ("LR", "UD", "R90", "R180"):
            r[f"ratio_{g}"] = fnum((sc.get(g) or {}).get("ratio_to_null_1-6"))
            r[f"twin_{g}"] = fnum((sc.get(g) or {}).get("twin_fraction_t50_1-6"))
        r["tax_LR_1-6"] = fnum((sc.get("LR") or {}).get("tax_mlp_1-6"))
        a = combine([cmp(r["ratio_UD"], ">=", 0.9), cmp(r["ratio_R90"], ">=", 0.9)])
        tn = (sc.get("UD") or {}).get("twin_null_twin_fraction_t50_1-6") or {}
        lo, hi, obs = fnum(tn.get("null_lo")), fnum(tn.get("null_hi")), fnum(tn.get("observed"))
        r.update({"ud_twin_observed": obs, "ud_twin_null_mean": fnum(tn.get("null_mean")), "ud_twin_null_lo": lo,
                  "ud_twin_null_hi": hi, "ud_twin_null_draws": tn.get("n_draws")})
        b = PENDING if None in (lo, hi, obs) else (PASS if lo <= obs <= hi else FAIL)
        sg = m.get("surgery") or {}
        ud, lr = sg.get("UD_1-6") or {}, sg.get("LR_1-6") or {}
        r.update({"ud_drop": fnum(ud.get("drop_points")), "lr_drop": fnum(lr.get("drop_points")),
                  "ud_top1": fnum(ud.get("top1")), "original_top1": fnum(ud.get("top1_original")),
                  "surgery_n_query": ud.get("n_query"), "ud_vs_lr_ci": (fnum(ud.get("diff_vs_LR_ci_low")),
                                                                         fnum(ud.get("diff_vs_LR_ci_high")))})
        cc = cmp(r["ud_drop"], ">", 30.0)
        r.update({"P4.1": a, "P4.2": b, "P4.3": cc, "git_hash": s["git_hash"]})
        v1.append(a)
        v2.append(b)
        v3.append(cc)
        rows.append(r)
        for g in ("LR", "UD", "R90", "R180"):
            c.num(f"must04.{model}.{g}.ratio_to_null_1-6", r[f"ratio_{g}"], desc=f"MUST-04: {g} ratio to null, blocks "
                  f"1 to 6, {model}", source=s)
        c.num(f"must04.{model}.UD.twin_t50_1-6", obs, desc=f"MUST-04: UD twin fraction (t 0.5), blocks 1 to 6, {model}",
              source=s, n=tn.get("n_draws"))
        c.num(f"must04.{model}.UD.surgery_drop", r["ud_drop"], desc=f"MUST-04: k-NN drop after UD surgery, blocks 1 to "
              f"6, {model}", source=s, unit="points", n=ud.get("n_query"))
        c.num(f"must04.{model}.LR.surgery_drop", r["lr_drop"], desc=f"MUST-04: k-NN drop after LR surgery, blocks 1 to "
              f"6, {model}", source=s, unit="points", n=lr.get("n_query"))
        c.num(f"must04.{model}.LR.tax_1-6", r["tax_LR_1-6"], desc=f"MUST-04: LR tax (character formula), MLP, blocks 1 "
              f"to 6, {model}", source=s)
        bdf = c.cload(p.parent / "blocks.csv")
        if bdf is not None:
            blocks.append(blocks_frame(bdf, model, "MUST-04", m, c.rel(p.parent / "blocks.csv"), s["git_hash"]))
    for pr, vs, key in ((p1, v1, "P4.1"), (p2, v2, "P4.2"), (p3, v3, "P4.3")):
        pr.verdict = combine(vs)
        pr.details = rows
        present = [r for r in rows if r.get("status") != "missing"]
        pr.n = f"{len(present)} of 4 models"
        if pr.verdict == PENDING and len(present) < 4:
            pr.reason = "missing: " + ", ".join(r["model"] for r in rows if r.get("status") == "missing")
    present = [r for r in rows if r.get("status") != "missing"]
    if present:
        p1.value = "min UD {} / min R90 {}".format(
            fmt(min((r["ratio_UD"] for r in present if r["ratio_UD"] is not None), default=None)),
            fmt(min((r["ratio_R90"] for r in present if r["ratio_R90"] is not None), default=None)))
        p2.value = "; ".join(f"{r['model']}: {fmt(r['ud_twin_observed'])} in [{fmt(r['ud_twin_null_lo'])}, "
                             f"{fmt(r['ud_twin_null_hi'])}]" for r in present)
        p3.value = "min UD drop {}".format(fmt(min((r["ud_drop"] for r in present if r["ud_drop"] is not None),
                                                   default=None)))
    if blocks:
        c.data.setdefault("_blocks", []).extend(blocks)


def blocks_frame(bdf: pd.DataFrame, model: str, exp: str, metrics: dict, file: str, git_hash) -> pd.DataFrame:
    depth = metrics.get("depth") or (int(bdf["block"].max()) if "block" in bdf else None)
    keep = [k for k in ("group", "block", "tax_mlp", "s_mlp", "s_mlp_null_mean", "s_mlp_null_lo", "s_mlp_null_hi",
                        "frac_2_t50", "frac_4", "s_fc1", "s_fc2", "s_fc1_act", "s_fc2_act", "n_pairs", "h", "d")
            if k in bdf]
    out = bdf[keep].copy()
    out.insert(0, "model", model)
    out.insert(1, "experiment", exp)
    out["depth"] = depth
    if depth:
        out["rel_depth"] = (out["block"] - 0.5) / float(depth)
    if "s_mlp" in out and "s_mlp_null_mean" in out:
        out["ratio_to_null"] = out["s_mlp"] / out["s_mlp_null_mean"]
    out["source_file"], out["git_hash"] = file, git_hash
    return out


# ================================================================================================
# MUST-05 and MUST-06
# ================================================================================================
def score_must05(c: Collector) -> None:
    p1 = c.add(Pred("MUST-05 P5.1", "MUST-05", "Must",
                    "Switch-off median cosine at least 0.7 for twin pairs and at most 0.2 for random pairs",
                    "P5.1: switch-off median cosine (primary measure, output of block l+1, after re-aligning tokens) at "
                    "least 0.7 for twin pairs and at most 0.2 for random pairs.",
                    "median over pairs of the switch-off cosine at the output of block l+1 (per-pair cosine pooled over "
                    "images and tokens), pooled over the 200 strongest twin pairs (and 200 random pairs) of each of "
                    "blocks 1 to 6, dinov2_vitb14_reg, 500 validation images", ">= 0.7 twins and <= 0.2 random"))
    p2 = c.add(Pred("MUST-05 P5.2", "MUST-05", "Must",
                    "The norm ratio is between 0.8 and 1.25 for at least 70% of twin pairs",
                    "P5.2: the norm ratio is between 0.8 and 1.25 for at least 70% of twin pairs.",
                    "share of twin pairs (blocks 1 to 6 pooled) with ||Delta_j|| / ||A Delta_i|| in [0.8, 1.25]",
                    ">= 0.70"))
    d = c.p("results", "MUST-05", MAIN)
    m = c.jload(d / "metrics.json")
    df = c.cload(d / "switch_pairs.csv")
    if df is None and not m:
        for pr in (p1, p2):
            pr.reason = f"results/MUST-05/{MAIN}/ missing"
        return
    blocks6 = list(range(1, 7))
    if df is not None and {"block", "pair_type", "mid_cos", "mid_norm_ratio"} <= set(df.columns):
        src = c.src(d / "switch_pairs.csv")
        sub = df[df["block"].isin(blocks6)]
        tw, rd = sub[sub["pair_type"] == "twin"], sub[sub["pair_type"] == "random"]
        tw_med = float(tw["mid_cos"].median()) if len(tw) else None
        rd_med = float(rd["mid_cos"].median()) if len(rd) else None
        r = tw["mid_norm_ratio"].dropna()
        share = float(((r >= 0.8) & (r <= 1.25)).mean()) if len(r) else None
        n_tw, n_rd = len(tw), len(rd)
        blocks_present = sorted(set(int(b) for b in sub["block"]))
        per_block = []
        for b, g in sub.groupby("block"):
            t, rr = g[g.pair_type == "twin"], g[g.pair_type == "random"]
            rb = t["mid_norm_ratio"].dropna()
            per_block.append({"block": int(b), "twin_median_cos": float(t["mid_cos"].median()) if len(t) else None,
                              "random_median_cos": float(rr["mid_cos"].median()) if len(rr) else None,
                              "twin_share_ratio_in_band": float(((rb >= 0.8) & (rb <= 1.25)).mean()) if len(rb) else None,
                              "n_twin": len(t), "n_random": len(rr)})
        n_img = int(df["n_images"].iloc[0]) if "n_images" in df else None
    else:
        src = c.src(d / "metrics.json")
        so = (m or {}).get("switch_off") or {}
        pooled = so.get("pooled") or {}
        tw_med = fnum(((pooled.get("twin") or {}).get("mid") or {}).get("cos_median"))
        rd_med = fnum(((pooled.get("random") or {}).get("mid") or {}).get("cos_median"))
        share = fnum(((pooled.get("twin") or {}).get("mid") or {}).get("share_ratio_in_band"))
        n_tw = ((pooled.get("twin") or {}).get("mid") or {}).get("n_pairs")
        n_rd = ((pooled.get("random") or {}).get("mid") or {}).get("n_pairs")
        blocks_present = so.get("blocks") or []
        per_block, n_img = [], so.get("n_images")
    complete = blocks_present == blocks6
    for pr in (p1, p2):
        pr.sources = [src]
        pr.n = f"{n_tw} twin and {n_rd} random pairs, blocks {blocks_present}, {n_img} images"
        pr.details = per_block
    p1.value = f"twins {fmt(tw_med)}, random {fmt(rd_med)}"
    p1.verdict = combine([cmp(tw_med, ">=", 0.7), cmp(rd_med, "<=", 0.2)])
    p2.value, p2.verdict = share, cmp(share, ">=", 0.70)
    if not complete:      # pooled medians and shares: a subset of the blocks decides nothing
        for pr in (p1, p2):
            pr.verdict = PENDING
            pr.reason = f"blocks measured {blocks_present}, not 1 to 6"
    c.num("must05.twin_median_cos", tw_med, desc="MUST-05: switch-off median cosine, twin pairs, blocks 1 to 6", source=src,
          n=n_tw)
    c.num("must05.random_median_cos", rd_med, desc="MUST-05: switch-off median cosine, random pairs", source=src, n=n_rd)
    c.num("must05.twin_share_norm_in_band", share, desc="MUST-05: share of twin pairs with norm ratio in [0.8, 1.25]",
          source=src, n=n_tw)
    if (d / "teaser.npz").is_file():
        c._seen(d / "teaser.npz")
        c.copies["fig1_teaser.npz"] = d / "teaser.npz"
        tp = c.cload(d / "teaser_top_patches.csv")
        meta = {"source_file": c.rel(d / "teaser.npz"), "git_hash": c.meta_git(d / "metrics.json")}
        c.data["fig1_meta"] = pd.DataFrame([meta | {"n_patch_rows": None if tp is None else len(tp)}])


def score_must06(c: Collector) -> None:
    quant_tail = "blocks 1 to 6 pooled; scored on dinov2_vitb14_reg (the claim), reported per model for the other two"
    p1 = c.add(Pred("MUST-06 P6.1", "MUST-06", "Must",
                    "Median same-token Pearson correlation of twins on the same image at most 0.3",
                    "P6.1: median same-token Pearson correlation of twins on the same image at most 0.3.",
                    "median over twin pairs of the same-token Pearson correlation (post-GELU); " + quant_tail, "<= 0.3"))
    p2 = c.add(Pred("MUST-06 P6.2", "MUST-06", "Must",
                    "A twin is in its partner's top 10 same-image correlated neurons for at most 20% of pairs",
                    "P6.2: a twin is in its partner's top 10 same-image correlated neurons for at most 20% of pairs.",
                    "share of twin pairs where the twin is in the top 10 |r| partners of either neuron; " + quant_tail,
                    "<= 0.20"))
    p3 = c.add(Pred("MUST-06 P6.3", "MUST-06", "Must",
                    "Effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection",
                    "P6.3: effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection.",
                    "largest |relative change| of exp(entropy of s / sum s) over the 12 matrices (fc1, fc2 of blocks 1 "
                    "to 6); holds if every matrix satisfies it", "<= 0.02 for all 12"))
    rows, vis_frames, erank_frames = [], [], []
    for model in MUST06_MODELS:
        d = c.p("results", "MUST-06", model)
        m = c.jload(d / "metrics.json")
        pdf, sdf = c.cload(d / "pairs.csv"), c.cload(d / "spectra.csv")
        if not m and pdf is None:
            rows.append({"model": model, "status": "missing"})
            continue
        g = c.meta_git(d / "metrics.json")
        act = (m or {}).get("act", "post")
        det = (m or {}).get("detector") or {}
        k, score = int(det.get("top_partners", 10)), det.get("score", "abs")
        r = {"model": model, "git_hash": g}
        if pdf is not None and f"same_corr_{act}" in pdf:
            sub = pdf[pdf["block"].between(1, 6)]
            tw = sub[sub["pair_type"] == "twin"]
            ri, rj = tw[f"rank_i_{score}_{act}"], tw[f"rank_j_{score}_{act}"]
            r.update({"median_same_corr": float(tw[f"same_corr_{act}"].median()) if len(tw) else None,
                      "median_mirror_corr": float(tw[f"mirror_corr_{act}"].median()) if len(tw) else None,
                      "share_seen": float(((ri <= k) | (rj <= k)).mean()) if len(tw) else None,
                      "n_twin_pairs": int(len(tw)), "blocks": sorted(set(int(b) for b in sub["block"])),
                      "file": c.rel(d / "pairs.csv")})
            keep = sub[["block", "pair_type", f"same_corr_{act}", f"mirror_corr_{act}"]].rename(
                columns={f"same_corr_{act}": "same_corr", f"mirror_corr_{act}": "mirror_corr"})
            vis_frames.append(keep.assign(model=model, source_file=c.rel(d / "pairs.csv"), git_hash=g))
        else:
            tw = ((m or {}).get("pooled") or {}).get("twin") or {}
            r.update({"median_same_corr": fnum(tw.get("same_corr_median")),
                      "median_mirror_corr": fnum(tw.get("mirror_corr_median")),
                      "share_seen": fnum(tw.get("share_seen_either")), "n_twin_pairs": tw.get("n_pairs"),
                      "blocks": (m or {}).get("blocks"), "file": c.rel(d / "metrics.json")})
        if sdf is not None and "erank_l1_rel_change" in sdf:
            mir = sdf[(sdf["structure"] == "mirror") & sdf["block"].between(1, 6)]
            r["max_abs_erank_change"] = float(mir["erank_l1_rel_change"].abs().max()) if len(mir) else None
            r["n_matrices"] = int(len(mir))
            erank_frames.append(sdf[sdf["block"].between(1, 6)].assign(model=model, source_file=c.rel(d / "spectra.csv"),
                                                                        git_hash=g))
        else:
            er = (m or {}).get("effective_rank_l1") or {}
            ch = [abs(fnum(v.get("rel_change"))) for kk, v in er.items()
                  if fnum(v.get("rel_change")) is not None and 1 <= int(kk.split("_")[0]) <= 6]
            r["max_abs_erank_change"] = max(ch) if ch else None
            r["n_matrices"] = len(ch)
        r["P6.1"] = cmp(r["median_same_corr"], "<=", 0.3)
        r["P6.2"] = cmp(r["share_seen"], "<=", 0.20)
        r["P6.3"] = cmp(r["max_abs_erank_change"], "<=", 0.02)
        if r.get("n_matrices") is not None and r["n_matrices"] < 12 and r["P6.3"] == PASS:
            r["P6.3"] = PENDING
        if r.get("blocks") and list(r["blocks"]) != list(range(1, 7)):
            for key in ("P6.1", "P6.2"):      # pooled over blocks 1 to 6: a subset decides nothing
                r[key] = PENDING
        rows.append(r)
        for key, desc in (("median_same_corr", "median same-token twin correlation"),
                          ("share_seen", "share of twins in the top-10 same-image partners"),
                          ("max_abs_erank_change", "largest |relative change| of effective rank under projection")):
            c.num(f"must06.{model}.{key}", r[key], desc=f"MUST-06: {desc}, blocks 1 to 6, {model}",
                  source={"file": r["file"] if key != "max_abs_erank_change" else c.rel(d / "spectra.csv"),
                          "git_hash": g}, n=r.get("n_twin_pairs"))
    main = next((r for r in rows if r["model"] == MAIN and r.get("status") != "missing"), None)
    for pr, key, vk in ((p1, "P6.1", "median_same_corr"), (p2, "P6.2", "share_seen"), (p3, "P6.3", "max_abs_erank_change")):
        pr.details = rows
        if main is None:
            pr.reason = f"results/MUST-06/{MAIN}/ missing"
            continue
        pr.verdict, pr.value = main[key], main[vk]
        pr.n = f"{main.get('n_twin_pairs')} twin pairs" if key != "P6.3" else f"{main.get('n_matrices')} matrices"
        pr.sources = [{"file": main["file"], "git_hash": main["git_hash"]}]
        others = [f"{r['model']}: {r[key]}" for r in rows if r["model"] != MAIN and r.get("status") != "missing"]
        if others:
            pr.reason = "other models (reported): " + "; ".join(others)
    if vis_frames:
        c.data["fig4a_pairs"] = pd.concat(vis_frames, ignore_index=True)
    if erank_frames:
        c.data["fig4a_erank"] = pd.concat(erank_frames, ignore_index=True)
    if main is not None:
        inv = combine([main["P6.1"], main["P6.2"], main["P6.3"]])
        c.num("must06.invisible_claim", inv, desc="MUST-06: the claim 'invisible' (all three predictions on "
              "dinov2_vitb14_reg)", source={"file": main["file"], "git_hash": main["git_hash"]})


# ================================================================================================
# MUST-07 and MUST-08
# ================================================================================================
def rows_frame(c: Collector, exp: str, model: str) -> tuple[pd.DataFrame | None, Path]:
    p = c.p("results", exp, f"rows_{model}.csv")
    df = c.cload(p)
    if df is not None:
        df = df.copy()
        if "status" in df:
            df = df[df["status"].astype(str) == "ok"]
        df["fit_seed"] = pd.to_numeric(df["fit_seed"], errors="coerce").astype("Int64")
    return df, p


def seed_values(df: pd.DataFrame, size: str, method: str, col: str = "val_top1") -> dict:
    g = df[(df["size"] == size) & (df["method"] == method)]
    out = {}
    for _, r in g.iterrows():
        v = fnum(r.get(col))
        if v is not None and pd.notna(r["fit_seed"]):
            out[int(r["fit_seed"])] = v
    return out


def score_must07(c: Collector) -> None:
    p1 = c.add(Pred("MUST-07 P7.1", "MUST-07", "Must",
                    "On dinov2_vitb14_reg at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least "
                    "2 points, with the paired bootstrap interval excluding 0",
                    "P7.1: on `dinov2_vitb14_reg` at 6 blocks, Mirror-exact beats every baseline on validation k-NN by "
                    "at least 2 points, with the paired bootstrap interval excluding 0.",
                    "for each of methods 3 to 12 and each 6-block size (MLP only; MLP plus attention): mean over fit "
                    "seeds 0, 1, 2 of Mirror-exact minus mean of the method; 95% paired bootstrap interval on the "
                    "fit-seed-0 per-image predictions (10,000 resamples)", ">= 2.0 points and CI low > 0, every method, "
                    "both sizes"))
    p2 = c.add(Pred("MUST-07 P7.2", "MUST-07", "Must",
                    "ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 points at "
                    "6 blocks",
                    "P7.2: ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 "
                    "points at 6 blocks.",
                    "mean over fit seeds 0, 1, 2 of Mirror-exact minus mean of zipit and of pair_actcorr, both 6-block "
                    "sizes", ">= 10.0 points"))
    df, path = rows_frame(c, "MUST-07", MAIN)
    boot = c.cload(c.p("results", "MUST-07", "bootstrap.csv"))
    if df is None:
        for pr in (p1, p2):
            pr.reason = f"results/MUST-07/rows_{MAIN}.csv missing"
    else:
        src = c.src(path, str(df["git_hash"].iloc[0]) if "git_hash" in df and len(df) else None)
        bsrc = c.src(c.p("results", "MUST-07", "bootstrap.csv"))
        rows, v1, v2 = [], [], []
        for size in SIX_SIZES:
            me = seed_values(df, size, "mirror_exact")
            for m in BASELINES:
                st = seed_values(df, size, m)
                full = set(SEEDS3) <= set(me) and set(SEEDS3) <= set(st)
                diff = (np.mean([me[s] for s in SEEDS3]) - np.mean([st[s] for s in SEEDS3])) if full else None
                b = None
                if boot is not None:
                    bb = boot[(boot["model"] == MAIN) & (boot["size"] == size) & (boot["method"] == m)
                              & (pd.to_numeric(boot["fit_seed"], errors="coerce") == 0)]
                    if "a" in bb:
                        bb = bb[bb["a"].astype(str) == "mirror_exact"]
                    if len(bb):
                        b = bb.iloc[-1]
                lo = fnum(b["ci_low"]) if b is not None else None
                hi = fnum(b["ci_high"]) if b is not None else None
                v = combine([cmp(diff, ">=", 2.0), cmp(lo, ">", 0.0)]) if (diff is not None or lo is not None) else PENDING
                if diff is None and v != FAIL:
                    v = PENDING
                rows.append({"size": size, "method": m, "seeds_mirror": sorted(me), "seeds_method": sorted(st),
                             "mirror_mean": float(np.mean([me[s] for s in SEEDS3])) if full else None,
                             "method_mean": float(np.mean([st[s] for s in SEEDS3])) if full else None,
                             "diff_mean": diff, "boot_diff_s0": fnum(b["diff"]) if b is not None else None,
                             "ci_low": lo, "ci_high": hi, "boot_n": int(b["n"]) if b is not None and "n" in b else None,
                             "P7.1": v, "P7.2": cmp(diff, ">=", 10.0) if m in ("zipit", "pair_actcorr") else ""})
                v1.append(v)
                if m in ("zipit", "pair_actcorr"):
                    v2.append(cmp(diff, ">=", 10.0))
                c.num(f"must07.{MAIN}.{size}.mirror_minus_{m}", diff, desc=f"MUST-07: Mirror-exact minus {m}, "
                      f"{size}, mean over fit seeds 0 to 2", source=[src, bsrc], n=3 if full else None, unit="points")
        p1.verdict, p2.verdict = combine(v1), combine(v2)
        p1.details = p2.details = rows
        p1.sources, p2.sources = [src, bsrc], [src]
        dd = [r["diff_mean"] for r in rows if r["diff_mean"] is not None]
        p1.value = f"smallest margin {fmt(min(dd))}" if dd else None
        dd2 = [r["diff_mean"] for r in rows if r["method"] in ("zipit", "pair_actcorr") and r["diff_mean"] is not None]
        p2.value = f"smallest margin {fmt(min(dd2))}" if dd2 else None
        p1.n = f"{sum(1 for r in rows if r['diff_mean'] is not None)} of {len(rows)} comparisons with 3 seeds"
        p2.n = f"{len(dd2)} of 4 comparisons with 3 seeds"
        miss = sorted({r["method"] for r in rows if r["P7.1"] == PENDING})
        if miss and p1.verdict == PENDING:
            p1.reason = "incomplete (3 fit seeds and the seed-0 bootstrap needed): " + ", ".join(miss)
        c.data["table3_must07_p7"] = pd.DataFrame(rows).assign(source_file=src["file"], git_hash=src["git_hash"])
    # all MUST-07 rows (every model) as a tidy table
    frames = []
    for model in MUST07_MODELS:
        d, pth = rows_frame(c, "MUST-07", model)
        if d is None:
            continue
        g = str(d["git_hash"].iloc[0]) if "git_hash" in d and len(d) else c.meta_git(pth)
        for (size, method), sub in d.groupby(["size", "method"], sort=False):
            mu, sd, n = mean_std(sub["val_top1"])
            s0 = sub[sub["fit_seed"] == 0]
            frames.append({"experiment": "MUST-07", "model": model, "size": size, "method": method, "top1_mean": mu,
                           "top1_std": sd, "n_seeds": n,
                           "fit_seeds": ",".join(str(int(x)) for x in sorted(sub["fit_seed"].dropna())),
                           "msel_top1_s0": fnum(s0["msel_top1"].iloc[0]) if len(s0) and "msel_top1" in s0 else None,
                           "flip_fp32": mean_std(sub["flip_consistency_fp32"])[0] if "flip_consistency_fp32" in sub else None,
                           "flip_bf16": mean_std(sub["flip_consistency"])[0] if "flip_consistency" in sub else None,
                           "params_total": mean_std(sub["params_total"])[0] if "params_total" in sub else None,
                           "budget_total": mean_std(sub["budget_total"])[0] if "budget_total" in sub else None,
                           "gap_rel_max": float(pd.to_numeric(sub["gap_rel"], errors="coerce").max())
                           if "gap_rel" in sub else None,
                           "model_params": mean_std(sub["model_params"])[0] if "model_params" in sub else None,
                           "model_params_frac": mean_std(sub["model_params_frac"])[0] if "model_params_frac" in sub else None,
                           "gflops_frac": mean_std(sub["gflops_frac"])[0] if "gflops_frac" in sub else None,
                           "source_file": c.rel(pth), "git_hash": g})
            if method in ("mirror_exact", "original") or (size in SIX_SIZES and model == MAIN):
                c.num(f"must07.{model}.{size}.{method}.top1", mu, desc=f"MUST-07: validation k-NN, {method}, {size}, "
                      f"{model} (mean over fit seeds)", source={"file": c.rel(pth), "git_hash": g}, n=n, unit="%")
    if frames:
        c.data["must07_rows"] = pd.DataFrame(frames)


def score_gate_b(c: Collector) -> None:
    gt = c.gates_append(Gate("B", "After MUST-07", "Best baseline chosen on the method-selection set and recorded "
                             "(results/MUST-07/gate_b.json) before MUST-09 starts"))
    p = c.p("results", "MUST-07", "gate_b.json")
    g = c.jload(p)
    if not g:
        gt.evidence = "results/MUST-07/gate_b.json missing"
        return
    gt.sources = [c.src(p)]
    sel = str(g.get("selection_data", ""))
    ok_sel = "method-selection" in sel and "validation is not read" in sel
    if g.get("best") and ok_sel:
        gt.status = MET
    elif g.get("best"):
        gt.status = NOT_MET
    gt.value = f"best {g.get('best')}, second {g.get('second_best')}"
    gt.evidence = (f"ranking by method-selection k-NN at size {g.get('size')}, fit seed {g.get('fit_seed')}; complete "
                   f"{g.get('complete')}" + (f"; missing {g.get('missing_methods')}" if g.get("missing_methods") else "")
                   + (f"; ties {g.get('ties')}" if g.get("ties") else ""))
    c.num("gate_b.best", str(g.get("best")), desc="Gate B: best baseline (method-selection k-NN, full model)",
          source=gt.sources[0])
    c.num("gate_b.second_best", str(g.get("second_best")), desc="Gate B: second-best baseline", source=gt.sources[0])


def score_must08(c: Collector) -> None:
    p1 = c.add(Pred("MUST-08 P8.1", "MUST-08", "Must",
                    "MLP-only saving is more than half of the MLP-plus-attention saving, at 6 blocks and at the full model",
                    "P8.1: the MLP carries most of the saving: MLP-only saving is more than half of the "
                    "MLP-plus-attention saving, at 6 blocks and at the full model.",
                    "saving = original model parameters minus Mirror-exact model parameters (mean over fit seeds); "
                    "ratio MLP-only saving / MLP-plus-attention saving", "> 0.5 at 6 blocks and at the full model"))
    p2 = c.add(Pred("MUST-08 P8.2", "MUST-08", "Must",
                    "k-NN(MLP only) minus k-NN(MLP plus attention) is at most 0.5 at 6 blocks (mean over fit seeds)",
                    "P8.2: adding attention costs at most 0.5 points at 6 blocks: k-NN(MLP only) minus k-NN(MLP plus "
                    "attention) is at most 0.5 (mean over fit seeds).",
                    "mean over fit seeds 0, 1, 2 of validation k-NN, Mirror-exact 6mlp minus 6both", "<= 0.5 points"))
    df, path = rows_frame(c, "MUST-08", MAIN)
    if df is None:
        for pr in (p1, p2):
            pr.reason = f"results/MUST-08/rows_{MAIN}.csv missing"
        return
    src = c.src(path, str(df["git_hash"].iloc[0]) if "git_hash" in df and len(df) else None)
    p1.sources = p2.sources = [src]
    orig = df[df["method"] == "original"]
    n_model = fnum(orig["model_params"].iloc[0]) if len(orig) and "model_params" in orig else None
    rows, vs = [], []
    for span, (sm, sb) in (("6 blocks", ("6mlp", "6both")), ("full model", ("fullmlp", "full"))):
        pm, pb = seed_values(df, sm, "mirror_exact", "model_params"), seed_values(df, sb, "mirror_exact", "model_params")
        if n_model is None or not pm or not pb:
            vs.append(PENDING)
            rows.append({"span": span, "status": "missing"})
            continue
        sav_m, sav_b = n_model - float(np.mean(list(pm.values()))), n_model - float(np.mean(list(pb.values())))
        ratio = sav_m / sav_b if sav_b else None
        v = cmp(ratio, ">", 0.5)
        vs.append(v)
        rows.append({"span": span, "saving_mlp_only": sav_m, "saving_mlp_attn": sav_b, "ratio": ratio,
                     "seeds": f"{sorted(pm)} / {sorted(pb)}", "P8.1": v})
        c.num(f"must08.{span.replace(' ', '_')}.saving_ratio", ratio, desc=f"MUST-08: MLP-only saving over "
              f"MLP-plus-attention saving, {span}", source=src)
    p1.verdict, p1.details = combine(vs), rows
    p1.value = "; ".join(f"{r['span']}: {fmt(r.get('ratio'))}" for r in rows)
    p1.n = "original model parameters from the original row"
    am, ab = seed_values(df, "6mlp", "mirror_exact"), seed_values(df, "6both", "mirror_exact")
    if set(SEEDS3) <= set(am) and set(SEEDS3) <= set(ab):
        diff = float(np.mean([am[s] for s in SEEDS3]) - np.mean([ab[s] for s in SEEDS3]))
        p2.value, p2.verdict, p2.n = diff, cmp(diff, "<=", 0.5), "3 fit seeds"
        c.num("must08.6blocks.mlp_minus_both", diff, desc="MUST-08: k-NN Mirror-exact 6mlp minus 6both, mean over fit "
              "seeds 0 to 2", source=src, n=3, unit="points")
    else:
        p2.reason = f"fit seeds present: 6mlp {sorted(am)}, 6both {sorted(ab)} (need 0, 1, 2)"
    p2.details = [{"size": s, **{f"seed{k}": v for k, v in seed_values(df, s, "mirror_exact").items()}}
                  for s in ("6mlp", "6attn", "6both", "fullmlp", "fullattn", "full")]
    table = []
    for size in ("6mlp", "6attn", "6both", "fullmlp", "fullattn", "full"):
        sub = df[(df["size"] == size) & (df["method"] == "mirror_exact")]
        if not len(sub):
            continue
        mu, sd, n = mean_std(sub["val_top1"])
        table.append({"size": size, "top1_mean": mu, "top1_std": sd, "n_seeds": n,
                      "model_params": mean_std(sub["model_params"])[0] if "model_params" in sub else None,
                      "model_params_frac": mean_std(sub["model_params_frac"])[0] if "model_params_frac" in sub else None,
                      "gflops_frac": mean_std(sub["gflops_frac"])[0] if "gflops_frac" in sub else None,
                      "flip_bf16": mean_std(sub["flip_consistency"])[0] if "flip_consistency" in sub else None,
                      "flip_fp32": mean_std(sub["flip_consistency_fp32"])[0] if "flip_consistency_fp32" in sub else None,
                      "source_file": src["file"], "git_hash": src["git_hash"]})
    if table:
        c.data["table3_must08"] = pd.DataFrame(table)


# ================================================================================================
# MUST-09 and MUST-10
# ================================================================================================
def must09_runs(c: Collector, model: str = MAIN) -> pd.DataFrame | None:
    """One row per finished heal (targets sym and raw), from runs/MUST-09/<model>/<folder>/s<seed>/metrics.json, with
    results/MUST-09/summary.csv as a fallback."""
    base = c.p("runs", "MUST-09", model)
    rows = []
    if base.is_dir():
        for mdir in sorted(base.iterdir()):
            if not mdir.is_dir():
                continue
            for sdir in sorted(mdir.glob("s*")):
                m = c.jload(sdir / "metrics.json")
                if not m or not m.get("complete"):
                    continue
                kb, kf, cost = m.get("knn_bf16") or {}, m.get("knn_fp32") or {}, m.get("cost") or {}
                n_q = kf.get("n_query")
                share = fnum(kf.get("flip_consistency"))
                nd = None if share is None or n_q is None else int(round((1 - share / 100.0) * int(n_q)))
                inv = exact_invariance(share, nd, kf)
                r = {"method": m.get("method"), "targets": m.get("targets", "sym"), "seed": int(m.get("seed", -1)),
                     "knn_top1": fnum(kb.get("knn_top1")), "n_query": kb.get("n_query"),
                     "flip_bf16": fnum(kb.get("flip_consistency")), "flip_fp32": share, "n_query_fp32": n_q,
                     "n_disagree_fp32": nd, "invariance": inv[0], "invariance_note": inv[1],
                     "knn_top1_fp32": fnum(kf.get("top1")), "lr": m.get("lr"),
                     "params_frac": fnum(cost.get("params_frac")), "params_model": cost.get("params_model"),
                     "gflops": fnum(cost.get("gflops")), "gflops_frac": fnum(cost.get("gflops_frac")),
                     "knn_original": fnum((m.get("knn_original") or {}).get("top1")),
                     "cls_flip_max_abs_fp32": fnum((m.get("cls_flip_fp32_probe") or {}).get("max_abs_diff")),
                     "peak_train_memory_gb": fnum(m.get("peak_train_memory_gb")),
                     "file": c.rel(sdir / "metrics.json"), "git_hash": m.get("git_hash") or c.meta_git(sdir)}
                for cr in m.get("curve") or []:
                    r[f"step{int(cr['step'])}"] = fnum(cr.get("knn_top1"))
                rows.append(r)
    if not rows:
        sm = c.cload(c.p("results", "MUST-09", "summary.csv"))
        if sm is None:
            return None
        sm = sm[sm["model"] == model] if "model" in sm else sm
        for _, s in sm.iterrows():
            share, n_q = fnum(s.get("flip_consistency_fp32")), s.get("n_query")
            nd = None if share is None or n_q is None or pd.isna(n_q) else int(round((1 - share / 100.0) * int(n_q)))
            inv = exact_invariance(share, nd, None)
            r = {"method": s.get("method"), "targets": s.get("targets", "sym"), "seed": int(s.get("seed")),
                 "knn_top1": fnum(s.get("knn_top1")), "n_query": s.get("n_query"), "flip_bf16": fnum(s.get("flip_consistency")),
                 "flip_fp32": share, "n_query_fp32": n_q, "n_disagree_fp32": nd, "invariance": inv[0],
                 "invariance_note": inv[1], "knn_top1_fp32": fnum(s.get("knn_top1_fp32")), "lr": s.get("lr"),
                 "params_frac": fnum(s.get("params_frac")), "params_model": s.get("params_model"),
                 "gflops": fnum(s.get("gflops")), "gflops_frac": fnum(s.get("gflops_frac")),
                 "knn_original": fnum(s.get("knn_original")), "cls_flip_max_abs_fp32": fnum(s.get("cls_flip_max_abs_fp32")),
                 "peak_train_memory_gb": fnum(s.get("peak_train_memory_gb")),
                 "file": c.rel(c.p("results", "MUST-09", "summary.csv")), "git_hash": s.get("git_hash")}
            for col in s.index:
                mm = re.fullmatch(r"knn_top1_step(\d+)", str(col))
                if mm:
                    r[f"step{mm.group(1)}"] = fnum(s[col])
            rows.append(r)
    return pd.DataFrame(rows) if rows else None


def heal_seeds(df: pd.DataFrame | None, method: str | None, targets: str = "sym") -> dict:
    if df is None or not method:
        return {}
    sub = df[(df["method"] == method) & (df["targets"] == targets)]
    return {int(r["seed"]): r for _, r in sub.iterrows()}


def score_must09(c: Collector) -> None:
    p1 = c.add(Pred("MUST-09 P9.1", "MUST-09", "Must",
                    "Mirror-exact is non-inferior to the best baseline: mean over seeds of Mirror-exact minus mean of "
                    "the best baseline is above -0.5 points (validation k-NN)",
                    "P9.1: Mirror-exact is non-inferior to the best baseline: mean over seeds of Mirror-exact minus "
                    "mean of the best baseline is above -0.5 points (validation k-NN).",
                    "mean over healing seeds 0, 1, 2 of final validation k-NN (bf16, 50,000 queries, step 780), "
                    "Mirror-exact minus the gate-B best baseline (flip-symmetrised targets)", "> -0.5 points"))
    p2 = c.add(Pred("MUST-09 P9.2", "MUST-09", "Must",
                    "Only Mirror-exact has 100% flip consistency (scored as: Mirror-exact and the random-structure "
                    "control reach 100% (fp32, rounding allowance); no compressor (best, second-best, plain SVD) does)",
                    "P9.2: only Mirror-exact has 100% flip consistency.",
                    "fp32 k-NN flip consistency on all 50,000 validation images, every heal; the rounding allowance "
                    "applies to the exact-invariance side; a compressor fails the prediction if its raw fp32 share is "
                    "100%", "Mirror-exact and random structure 100%; compressors < 100%"))
    p3 = c.add(Pred("MUST-09 P9.3", "MUST-09", "Must", "Random structure trails Mirror-exact by at least 3 points",
                    "P9.3: random structure trails Mirror-exact by at least 3 points.",
                    "mean over seeds 0, 1, 2 of Mirror-exact minus random structure (seed 0), final validation k-NN",
                    ">= 3.0 points"))
    gd = c.gates_append(Gate("D", "After MUST-09", "Mirror-exact mean at least best-baseline mean minus 1.0"))
    df = must09_runs(c)
    gb = c.jload(c.p("results", "MUST-07", "gate_b.json")) or {}
    best, second = gb.get("best"), gb.get("second_best")
    if df is None:
        for pr in (p1, p2, p3):
            pr.reason = "no finished MUST-09 heal (runs/MUST-09/ or results/MUST-09/summary.csv)"
        gd.evidence = p1.reason
        return
    me = heal_seeds(df, "mirror_exact")
    bb = heal_seeds(df, best)
    rs = heal_seeds(df, "random_structure")
    srcs = [{"file": r["file"], "git_hash": r["git_hash"]} for _, r in df.iterrows()]
    me_mean = float(np.mean([me[s]["knn_top1"] for s in SEEDS3])) if set(SEEDS3) <= set(me) else None
    bb_mean = float(np.mean([bb[s]["knn_top1"] for s in SEEDS3])) if set(SEEDS3) <= set(bb) else None
    for pr in (p1, p2, p3):
        pr.sources = srcs
        pr.details = df.drop(columns=[x for x in ("invariance_note",) if x in df]).to_dict("records")
    # P9.1 and gate D
    if not best:
        p1.reason = gd.evidence = "gate B not recorded (results/MUST-07/gate_b.json)"
    elif me_mean is None or bb_mean is None:
        p1.reason = gd.evidence = (f"seeds present: mirror_exact {sorted(me)}, {best} {sorted(bb)} (need 0, 1, 2)")
    else:
        diff = me_mean - bb_mean
        p1.value, p1.verdict, p1.n = diff, cmp(diff, ">", -0.5), f"3 + 3 seeds (best baseline: {best})"
        gd.status = MET if diff >= -1.0 else NOT_MET
        gd.value = f"Mirror-exact {me_mean:.2f} vs {best} {bb_mean:.2f} (difference {diff:+.2f}; bar -1.0)"
        gd.sources = srcs
        c.num("must09.mirror_exact.knn_mean", me_mean, desc="MUST-09: healed Mirror-exact, validation k-NN, mean of "
              "seeds 0 to 2", source=srcs, n=3, unit="%")
        c.num(f"must09.{best}.knn_mean", bb_mean, desc=f"MUST-09: healed best baseline ({best}), mean of seeds 0 to 2",
              source=srcs, n=3, unit="%")
        c.num("must09.mirror_minus_best", diff, desc="MUST-09: Mirror-exact minus best baseline (seed means)",
              source=srcs, n=3, unit="points")
    if me:
        mu, sd, n = mean_std([r["knn_top1"] for r in me.values()])
        c.num("must09.mirror_exact.knn_std", sd, desc="MUST-09: healed Mirror-exact, k-NN standard deviation over seeds",
              source=srcs, n=n, unit="points")
        pf = [r["params_frac"] for r in me.values() if r["params_frac"] is not None]
        c.num("must09.mirror_exact.params_frac", pf[0] if pf else None, desc="MUST-09: healed Mirror-exact, share of the "
              "original's parameters (all basis changes counted)", source=srcs)
    # P9.2
    parts, notes = [], []
    for name, d_, need in (("mirror_exact", me, SEEDS3), ("random_structure", rs, (0,))):
        for s in need:
            if s not in d_:
                parts.append(PENDING)
                notes.append(f"{name} s{s} missing")
                continue
            r = d_[s]
            v = r["invariance"]
            if r["n_query_fp32"] is not None and int(r["n_query_fp32"]) < N_VAL and v == PASS:
                v = PENDING
                notes.append(f"{name} s{s}: fp32 pass on {r['n_query_fp32']} queries")
            parts.append(v)
            notes.append(f"{name} s{s}: {r['invariance_note']}")
    comp_methods = [m for m in (best, second, "svd") if m and m not in ("random_structure",)]
    fam = [m for m in comp_methods if m in ("pair_actcorr", "pair_wcos", "pair_random", "mirror_local")]
    if fam:
        notes.append(f"{', '.join(fam)} belong(s) to the Mirror family (A* construction, exactly invariant by design) "
                     "but are scored here as the compressors the prereg names (best, second-best)")
    for m in dict.fromkeys(comp_methods):
        d_ = heal_seeds(df, m)
        if not d_:
            parts.append(PENDING)
            notes.append(f"{m} missing")
            continue
        for s, r in sorted(d_.items()):
            sh = r["flip_fp32"]
            v = PENDING if sh is None else (FAIL if sh >= 100.0 else PASS)
            parts.append(v)
            notes.append(f"{m} s{s}: fp32 {fmt(sh, '.3f')}%")
    p2.verdict = combine(parts)
    p2.value = "; ".join(f"{r['method']} s{r['seed']}: {fmt(r['flip_fp32'], '.3f')}%" for _, r in df.iterrows()
                         if r["targets"] == "sym")
    p2.reason = "; ".join(notes)
    p2.n = "fp32 pass per heal (50,000 queries each)"
    # P9.3
    if me_mean is not None and 0 in rs:
        diff = me_mean - rs[0]["knn_top1"]
        p3.value, p3.verdict, p3.n = diff, cmp(diff, ">=", 3.0), "3 seeds against 1"
        c.num("must09.mirror_minus_random", diff, desc="MUST-09: Mirror-exact (seed mean) minus random structure (seed 0)",
              source=srcs, unit="points")
    else:
        p3.reason = f"seeds present: mirror_exact {sorted(me)}, random_structure {sorted(rs)}"
    # tables
    lat = c.cload(c.p("results", "MUST-09", "latency.csv"))
    t1 = []
    for (method, targets), sub in df.groupby(["method", "targets"], sort=False):
        mu, sd, n = mean_std(sub["knn_top1"])
        row = {"method": method, "targets": targets, "n_seeds": n, "seeds": ",".join(str(s) for s in sorted(sub["seed"])),
               "knn_mean": mu, "knn_std": sd, "flip_fp32_min": fnum(pd.to_numeric(sub["flip_fp32"]).min()),
               "flip_bf16_mean": mean_std(sub["flip_bf16"])[0], "params_frac": mean_std(sub["params_frac"])[0],
               "gflops_frac": mean_std(sub["gflops_frac"])[0], "params_model": mean_std(sub["params_model"])[0],
               "lr": sub["lr"].iloc[0],
               "peak_train_memory_gb": mean_std(sub["peak_train_memory_gb"])[0],
               "role": ("best baseline" if method == best else "second-best baseline" if method == second else ""),
               "source_file": "; ".join(sub["file"]), "git_hash": "; ".join(sorted(set(map(str, sub["git_hash"]))))}
        for col in sorted([x for x in sub.columns if re.fullmatch(r"step\d+", str(x))], key=lambda x: int(x[4:])):
            row[f"knn_{col}"] = mean_std(sub[col])[0]
        if lat is not None and {"method", "batch"} <= set(lat.columns):
            lt = lat[(lat["method"] == method) & (lat.get("variant", "") != "original")]
            for b in (1, 64, 256):
                x = lt[lt["batch"] == b]
                col = next((k for k in ("median_ms", "latency_ms", "ms") if k in x), None)
                if len(x) and col:
                    row[f"latency_ms_b{b}"] = fnum(x[col].iloc[0])
        t1.append(row)
    c.data["table1_healed"] = pd.DataFrame(t1)
    curves = c.cload(c.p("results", "MUST-09", "curves.csv"))
    if curves is not None:
        c.data["must09_curves"] = curves.assign(source_file="results/MUST-09/curves.csv")


def score_must10(c: Collector) -> None:
    pr = c.add(Pred("MUST-10 P10.1", "MUST-10", "Must",
                    "Healed Mirror-exact (mean of 3 seeds) is within 1 point of flip averaging on validation k-NN, at "
                    "under 45% of its FLOPs (analytic, one pass against two)",
                    "P10.1: healed Mirror-exact (mean of 3 seeds) is within 1 point of flip averaging on validation k-NN, "
                    "at under 45% of its FLOPs (analytic, one pass against two).",
                    "|mean over seeds 0 to 2 of healed Mirror-exact k-NN (MUST-09) minus flip-averaged original k-NN "
                    "(MUST-10)|, and healed GFLOPs per image / flip-averaging GFLOPs per image",
                    "|difference| <= 1.0 and FLOPs ratio < 0.45"))
    fa = c.cload(c.p("results", "MUST-10", "flipavg.csv"))
    if fa is None:
        pr.reason = "results/MUST-10/flipavg.csv missing"
        return
    fsrc = c.src(c.p("results", "MUST-10", "flipavg.csv"), str(fa["git_hash"].iloc[0]) if "git_hash" in fa else None)
    fa_m = fa[fa["model"] == MAIN] if "model" in fa and (fa["model"] == MAIN).any() else fa
    f = fa_m[(fa_m["variant"] == "original") & (fa_m["mode"] == "flipavg")]
    h = fa_m[(fa_m["variant"] == "healed") & (fa_m["mode"] == "plain")]
    o = fa_m[(fa_m["variant"] == "original") & (fa_m["mode"] == "plain")]
    df = must09_runs(c)
    me = heal_seeds(df, "mirror_exact")
    pr.sources = [fsrc] + ([{"file": r["file"], "git_hash": r["git_hash"]} for r in me.values()] if me else [])
    ratio = (fnum(h["gflops_per_image"].iloc[0]) / fnum(f["gflops_per_image"].iloc[0])) if len(h) and len(f) else None
    fa_knn = fnum(f["knn_top1"].iloc[0]) if len(f) else None
    if set(SEEDS3) <= set(me) and fa_knn is not None:
        me_mean = float(np.mean([me[s]["knn_top1"] for s in SEEDS3]))
        diff = me_mean - fa_knn
        pr.verdict = combine([within(me_mean, fa_knn, 1.0), cmp(ratio, "<", 0.45)])
        pr.value = f"difference {diff:+.2f} points; FLOPs ratio {fmt(ratio, '.3f')}"
        pr.n = "3 healed seeds against 1 flip-averaged original"
        c.num("must10.mirror_minus_flipavg", diff, desc="MUST-10: healed Mirror-exact (mean of 3 seeds) minus "
              "flip-averaged original, validation k-NN", source=pr.sources, unit="points")
    else:
        pr.reason = f"need MUST-09 mirror_exact seeds 0 to 2 (present {sorted(me)}) and the flip-averaged row"
        pr.value = f"FLOPs ratio {fmt(ratio, '.3f')}" if ratio is not None else None
    c.num("must10.flipavg.knn", fa_knn, desc="MUST-10: flip-averaged original, validation k-NN (bf16)", source=fsrc,
          unit="%")
    c.num("must10.gflops_ratio", ratio, desc="MUST-10: healed Mirror-exact GFLOPs over flip-averaging GFLOPs (analytic)",
          source=fsrc)
    pr.details = fa_m.to_dict("records")
    rows = []
    for _, r in fa_m.iterrows():
        rows.append({"variant": r.get("variant"), "mode": r.get("mode"), "knn_top1": fnum(r.get("knn_top1")),
                     "flip_fp32": fnum(r.get("flip_consistency_fp32")), "probe_top1": fnum(r.get("probe_top1")),
                     "gflops_per_image": fnum(r.get("gflops_per_image")), "params_frac": fnum(r.get("params_frac")),
                     "source_file": fsrc["file"], "git_hash": fsrc["git_hash"]})
    c.data["table1_flipavg"] = pd.DataFrame(rows)
    if len(o):
        c.num("must10.original.knn", fnum(o["knn_top1"].iloc[0]), desc="MUST-10: original model, validation k-NN (bf16)",
              source=fsrc, unit="%")


# ================================================================================================
# MUST-11
# ================================================================================================
def score_must11(c: Collector) -> None:
    p1 = c.add(Pred("MUST-11 P11.1", "MUST-11", "Must",
                    "Healed Mirror-exact is within 1.0 point of the original on the ImageNet linear probe (mean of 3 "
                    "seeds) and on ADE20K mIoU (seed 0)",
                    "P11.1: healed Mirror-exact is within 1.0 point of the original on the ImageNet linear probe (mean of "
                    "3 seeds) and on ADE20K mIoU (seed 0).",
                    "|mean of vexact_s0..s2 linear-probe top-1 minus the original's| and |vexact_s0 mIoU minus the "
                    "original's|", "both <= 1.0"))
    p2 = c.add(Pred("MUST-11 P11.2", "MUST-11", "Must",
                    "Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays "
                    "below 100%",
                    "P11.2: Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model "
                    "stays below 100%.",
                    "share of pixels with mask(mirror x) = mirror(mask x), fp32, ADE20K validation; Mirror-exact "
                    "(vexact_s0) symmetrised head with the rounding allowance; the original below 100% with every "
                    "recorded head", "Mirror-exact 100%; original < 100%"))
    lin_p = c.p("results", "MUST-11", "t2", "t2_linear.csv")
    seg_p = c.p("results", "MUST-11", "t2_seg", "t2_ade20k.csv")
    lin, seg = c.cload(lin_p), c.cload(seg_p)
    parts, val = [], []
    if lin is not None:
        lsrc = c.src(lin_p, str(lin["git_hash"].iloc[0]) if "git_hash" in lin else None)
        l = lin[lin["model"] == MAIN] if "model" in lin else lin
        orig = l[l["variant"] == "original"]
        vx = {s: l[l["variant"] == f"vexact_s{s}"] for s in SEEDS3}
        if len(orig) and all(len(v) for v in vx.values()):
            mu = float(np.mean([fnum(vx[s]["top1"].iloc[0]) for s in SEEDS3]))
            o = fnum(orig["top1"].iloc[0])
            parts.append(within(mu, o, 1.0))
            val.append(f"probe {mu - o:+.2f}")
            p1.sources.append(lsrc)
            c.num("must11.probe.vexact_mean", mu, desc="MUST-11: ImageNet linear probe, healed Mirror-exact, mean of 3 "
                  "seeds", source=lsrc, n=3, unit="%")
            c.num("must11.probe.original", o, desc="MUST-11: ImageNet linear probe, original", source=lsrc, unit="%")
        else:
            parts.append(PENDING)
            p1.reason = "linear probe rows missing (original and vexact_s0..s2 needed)"
    else:
        parts.append(PENDING)
        p1.reason = "results/MUST-11/t2/t2_linear.csv missing"
    p2parts = []
    if seg is not None:
        ssrc = c.src(seg_p, str(seg["git_hash"].iloc[0]) if "git_hash" in seg else None)
        s_ = seg[seg["model"] == MAIN] if "model" in seg else seg
        orig, v0 = s_[s_["variant"] == "original"], s_[s_["variant"] == "vexact_s0"]
        if len(orig) and len(v0):
            parts.append(within(fnum(v0["mIoU"].iloc[0]), fnum(orig["mIoU"].iloc[0]), 1.0))
            val.append(f"ADE20K {fnum(v0['mIoU'].iloc[0]) - fnum(orig['mIoU'].iloc[0]):+.2f}")
            p1.sources.append(ssrc)
            c.num("must11.ade20k.vexact_s0", fnum(v0["mIoU"].iloc[0]), desc="MUST-11: ADE20K mIoU, healed Mirror-exact "
                  "seed 0", source=ssrc, unit="%")
            c.num("must11.ade20k.original", fnum(orig["mIoU"].iloc[0]), desc="MUST-11: ADE20K mIoU, original",
                  source=ssrc, unit="%")
            p2.sources.append(ssrc)
            if "mask_eq_sym" in v0 and fnum(v0["mask_eq_sym"].iloc[0]) is not None:
                mj = c.jload(c.p("results", "MUST-11", "t2_seg", "seg", MAIN, "vexact_s0_maskeq.json")) or {}
                sym_src = (mj.get("result") or mj).get("sym") if isinstance((mj.get("result") or mj), dict) else None
                v, note, _ = exact_invariance(fnum(v0["mask_eq_sym"].iloc[0]), v0["mask_eq_sym_n_differ"].iloc[0]
                                              if "mask_eq_sym_n_differ" in v0 else None, sym_src)
                p2parts.append(v)
                p2.reason = f"Mirror-exact symmetrised head: {note}"
                c.num("must11.maskeq.vexact_s0.sym", fnum(v0["mask_eq_sym"].iloc[0]), desc="MUST-11: mask equivariance, "
                      "healed Mirror-exact seed 0, symmetrised head (fp32)", source=ssrc, unit="%")
            else:
                p2parts.append(PENDING)
                p2.reason = "mask equivariance of vexact_s0 not recorded"
            oshares = [fnum(orig[col].iloc[0]) for col in ("mask_eq_raw", "mask_eq_sym") if col in orig
                       and fnum(orig[col].iloc[0]) is not None]
            if oshares:
                p2parts.append(PASS if all(x < 100.0 for x in oshares) else FAIL)
                c.num("must11.maskeq.original.raw", fnum(orig["mask_eq_raw"].iloc[0]) if "mask_eq_raw" in orig else None,
                      desc="MUST-11: mask equivariance, original, raw head (fp32)", source=ssrc, unit="%")
            else:
                p2parts.append(PENDING)
        else:
            parts.append(PENDING)
            p2parts.append(PENDING)
    else:
        parts.append(PENDING)
        p2parts.append(PENDING)
        p2.reason = "results/MUST-11/t2_seg/t2_ade20k.csv missing"
    p1.verdict, p1.value = combine(parts), "; ".join(val) or None
    p1.n = "linear probe 3 seeds; ADE20K seed 0"
    p2.verdict = combine(p2parts)
    if seg is not None and len(p2.sources):
        s_ = seg[seg["model"] == MAIN] if "model" in seg else seg
        p2.value = "; ".join(f"{r['variant']}: raw {fmt(fnum(r.get('mask_eq_raw')), '.4f')}, sym "
                             f"{fmt(fnum(r.get('mask_eq_sym')), '.4f')}" for _, r in s_.iterrows()
                             if fnum(r.get("mask_eq_raw")) is not None)
    # Table 2 (downstream): linear, transfer, ADE20K
    t2 = []
    if lin is not None:
        for _, r in lin.iterrows():
            t2.append({"task": "ImageNet linear probe", "variant": r["variant"], "value": fnum(r["top1"]),
                       "source_file": c.rel(lin_p), "git_hash": r.get("git_hash")})
    tr_p = c.p("results", "MUST-11", "t2", "t2_transfer.csv")
    tr = c.cload(tr_p)
    if tr is not None:
        for _, r in tr.iterrows():
            t2.append({"task": f"transfer {r['dataset']}", "variant": r["variant"], "value": fnum(r["top1"]),
                       "source_file": c.rel(tr_p), "git_hash": r.get("git_hash")})
    if seg is not None:
        for _, r in seg.iterrows():
            t2.append({"task": "ADE20K mIoU", "variant": r["variant"], "value": fnum(r["mIoU"]),
                       "source_file": c.rel(seg_p), "git_hash": r.get("git_hash")})
            for col, name in (("mask_eq_raw", "mask equivariance, raw head"),
                              ("mask_eq_sym", "mask equivariance, symmetrised head")):
                if col in r and fnum(r[col]) is not None:
                    t2.append({"task": name, "variant": r["variant"], "value": fnum(r[col]),
                               "source_file": c.rel(seg_p), "git_hash": r.get("git_hash")})
    if t2:
        c.data["table2_downstream"] = pd.DataFrame(t2)


# ================================================================================================
# MUST-12 and gate C; CORE-01
# ================================================================================================
def run_final(c: Collector, exp: str, run: str) -> dict:
    """Latest measured checkpoint of a controlled run, with the run's planned epoch count from its meta.json."""
    rdir = c.p("results", exp, run)
    ep, d = latest_epoch_dir(rdir)
    meta = c.jload(c.p("runs", exp, run, "meta.json")) or {}
    planned = (meta.get("config") or {}).get("epochs")
    out = {"run": run, "experiment": exp, "epoch": ep, "dir": d, "planned_epochs": planned,
           "complete_run": (meta.get("extra") or {}).get("status") == "complete",
           "metrics": c.jload(d / "metrics.json") if d is not None else None,
           "meta": meta}
    return out


def scores_of(m: dict | None, group: str) -> dict:
    return ((m or {}).get("scores") or {}).get(group) or {}


def score_gate_c(c: Collector) -> None:
    gt = c.gates_append(Gate("C", "After MUST-12", "LR model: LR ratio to null in blocks 1 to 6 at most 0.8 and LR twin "
                             "fraction in blocks 1 to 6 at least 0.2"))
    pr = c.add(Pred("MUST-12 gate C", "MUST-12", "Must",
                    "Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin "
                    "fraction in blocks 1 to 6 is at least 0.2",
                    "Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin "
                    "fraction in blocks 1 to 6 is at least 0.2.",
                    "MUST-12 ViT-S/8 LR run, seed 0, final checkpoint: LR ratio to null and LR twin fraction (t 0.5), "
                    "blocks 1 to 6", "ratio <= 0.8 and twin >= 0.2"))
    f = run_final(c, "MUST-12", f"{CORE_ARCH}_LR_s0")
    gj_p = c.p("results", "MUST-12", "gate_c.json")
    gj = c.jload(gj_p)
    none_meta = c.jload(c.p("runs", "MUST-12", f"{CORE_ARCH}_none_s0", "meta.json")) or {}
    minutes = fnum(((none_meta.get("extra") or {}).get("timing") or {}).get("wall_minutes"))
    if minutes is not None:
        c.num("must12.none_wall_minutes", minutes, desc="MUST-12: wall time of the none pilot run (epoch rule: over 25 "
              "minutes means 60 epochs)", source=c.src(c.p("runs", "MUST-12", f"{CORE_ARCH}_none_s0", "meta.json")),
              unit="minutes")
        c.num("must12.core01_epochs", 60 if minutes > 25.0 else 100, desc="CORE-01 epochs by the MUST-12 rule",
              source=c.src(c.p("runs", "MUST-12", f"{CORE_ARCH}_none_s0", "meta.json")))
    if f["metrics"] is None:
        pr.reason = gt.evidence = f"results/MUST-12/{CORE_ARCH}_LR_s0/e<NNN>/metrics.json missing"
        return
    mp = f["dir"] / "metrics.json"
    s = c.src(mp)
    pr.sources = gt.sources = [s] + ([c.src(gj_p)] if gj else [])
    sc = scores_of(f["metrics"], "LR")
    ratio, twin = fnum(sc.get("ratio_to_null_1-6")), fnum(sc.get("twin_fraction_t50_1-6"))
    pr.verdict = combine([cmp(ratio, "<=", 0.8), cmp(twin, ">=", 0.2)])
    if f["planned_epochs"] is not None and f["epoch"] != int(f["planned_epochs"]) and pr.verdict != FAIL:
        pr.verdict = PENDING
        pr.reason = f"latest measured epoch {f['epoch']}, run planned for {f['planned_epochs']}"
    pr.value = f"ratio {fmt(ratio)}, twin {fmt(twin)}"
    pr.n = f"epoch {f['epoch']}"
    gt.status = {PASS: MET, FAIL: NOT_MET}.get(pr.verdict, PENDING)
    gt.value = pr.value
    if gj:
        same = (fnum(gj.get("lr_ratio_to_null_blocks_1_6")) == ratio and fnum(gj.get("lr_twin_fraction_t50_blocks_1_6"))
                == twin and bool(gj.get("pass")) == (pr.verdict == PASS))
        gt.evidence = ("scripts/h100_gate_c.py recorded the same values and verdict" if same else
                       f"gate_c.json differs: pass {gj.get('pass')}, ratio {gj.get('lr_ratio_to_null_blocks_1_6')}, "
                       f"twin {gj.get('lr_twin_fraction_t50_blocks_1_6')}")
        gt.evidence += f"; CORE-01 epochs {gj.get('core01_epochs')}"
    c.num("must12.LR.ratio_to_null_1-6", ratio, desc="MUST-12: LR ratio to null, blocks 1 to 6, LR pilot", source=s)
    c.num("must12.LR.twin_t50_1-6", twin, desc="MUST-12: LR twin fraction (t 0.5), blocks 1 to 6, LR pilot", source=s)


def core01_runs(c: Collector) -> dict:
    """Final measurement of every CORE-01 run (falls back to the MUST-12 pilot for none and LR seed 0 when CORE-01 has
    no measurement of its own and the pilot has the CORE-01 epoch count)."""
    out = {}
    planned = []
    for cond, seeds in CORE01_RUNS.items():
        for s in seeds:
            run = f"{CORE_ARCH}_{cond}_s{s}"
            f = run_final(c, "CORE-01", run)
            if f["metrics"] is None and cond in ("none", "LR") and s == 0:
                g = run_final(c, "MUST-12", run)
                if g["metrics"] is not None:
                    g["note"] = "MUST-12 pilot measurement (same recipe)"
                    f = g
            if f["planned_epochs"] is not None:
                planned.append(int(f["planned_epochs"]))
            out[(cond, s)] = f
    core_epochs = max(set(planned), key=planned.count) if planned else None
    for k, f in out.items():
        f["final_ok"] = (f["metrics"] is not None and (
            (f["planned_epochs"] is not None and f["epoch"] == int(f["planned_epochs"])) or
            (f["planned_epochs"] is None and core_epochs is not None and f["epoch"] == core_epochs) or
            (f["planned_epochs"] is None and core_epochs is None)))
        if f["experiment"] == "MUST-12" and core_epochs is not None and f["epoch"] != core_epochs:
            f["final_ok"] = False
    return out


def epoch_metrics(c: Collector, f: dict, epoch: int) -> dict | None:
    if f.get("dir") is None:
        return None
    return c.jload(f["dir"].parent / f"e{epoch:03d}" / "metrics.json")


def score_core01(c: Collector) -> None:
    t1 = c.add(Pred("CORE-01 T1", "CORE-01", "Core",
                    "Dissociation: in every seed pair s = 0, 1, 2, the UD twin fraction (blocks 1 to 6, final) of UD run "
                    "s minus that of LR run s is at least 0.2 (absolute)",
                    "T1 Dissociation: in every seed pair s = 0, 1, 2, the UD twin fraction (blocks 1 to 6, final) of UD "
                    "run s minus that of LR run s is at least 0.2 (absolute).",
                    "UD twin fraction (t 0.5), blocks 1 to 6, last checkpoint: UD run s minus LR run s, pairs s = 0, 1, 2",
                    ">= 0.2 in every pair"))
    t2 = c.add(Pred("CORE-01 T2", "CORE-01", "Core",
                    "Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none run "
                    "and every LR run (blocks 1 to 6, final)",
                    "T2 Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none "
                    "run and every LR run (blocks 1 to 6, final).",
                    "UD ratio to null, blocks 1 to 6, last checkpoint", "UD runs <= 0.7; none and LR runs >= 0.9"))
    t3 = c.add(Pred("CORE-01 T3", "CORE-01", "Core",
                    "Rotation: in every R90 run, the share in 4-cycles (blocks 1 to 6, final) is at least 0.3, and at "
                    "least 3 times the mean share in the none runs",
                    "T3 Rotation: in every R90 run, the share in 4-cycles (blocks 1 to 6, final) is at least 0.3, and at "
                    "least 3 times the mean share in the none runs.",
                    "R90 share of hidden units in 4-cycles, blocks 1 to 6, last checkpoint; against 3 x the mean over "
                    "the three none runs", ">= 0.3 and >= 3 x none mean, every R90 run"))
    t4 = c.add(Pred("CORE-01 T4", "CORE-01", "Core",
                    "Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained "
                    "group is higher at the final epoch than at epoch 5",
                    "T4 Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained "
                    "group is higher at the final epoch than at epoch 5.",
                    "twin fraction (t 0.5) of the trained group, blocks 1 to 6 (the range of T1 to T3), last checkpoint "
                    "against epoch 5", "final > epoch 5, every LR and UD run"))
    runs = core01_runs(c)

    def val(cond, s, group, key):
        f = runs[(cond, s)]
        if f["metrics"] is None or not f["final_ok"]:
            return None
        return fnum(scores_of(f["metrics"], group).get(key))

    def srcs(keys):
        out = []
        for k in keys:
            f = runs[k]
            if f["metrics"] is not None:
                out.append(c.src(f["dir"] / "metrics.json"))
        return out

    # T1
    rows, vs = [], []
    for s in (0, 1, 2):
        u, l = val("UD", s, "UD", "twin_fraction_t50_1-6"), val("LR", s, "UD", "twin_fraction_t50_1-6")
        d = None if u is None or l is None else u - l
        v = cmp(d, ">=", 0.2)
        vs.append(v)
        rows.append({"seed_pair": s, "UD_run_UD_twin": u, "LR_run_UD_twin": l, "difference": d, "verdict": v,
                     "UD_run_epoch": runs[("UD", s)]["epoch"], "LR_run_epoch": runs[("LR", s)]["epoch"]})
        c.num(f"core01.T1.pair{s}", d, desc=f"CORE-01 T1: UD twin fraction, UD run s{s} minus LR run s{s}",
              source=srcs([("UD", s), ("LR", s)]))
    t1.verdict, t1.details = combine(vs), rows
    t1.value = "min difference " + fmt(min((r["difference"] for r in rows if r["difference"] is not None), default=None))
    t1.n = f"{sum(r['difference'] is not None for r in rows)} of 3 seed pairs"
    t1.sources = srcs([(cn, s) for cn in ("UD", "LR") for s in (0, 1, 2)])
    # T2
    rows, vs = [], []
    for cond, op, bar in (("UD", "<=", 0.7), ("none", ">=", 0.9), ("LR", ">=", 0.9)):
        for s in CORE01_RUNS[cond]:
            x = val(cond, s, "UD", "ratio_to_null_1-6")
            v = cmp(x, op, bar)
            vs.append(v)
            rows.append({"run": f"{cond} s{s}", "UD_ratio_to_null": x, "bar": f"{op} {bar}", "verdict": v,
                         "epoch": runs[(cond, s)]["epoch"]})
    t2.verdict, t2.details = combine(vs), rows
    t2.value = "; ".join(f"{r['run']}: {fmt(r['UD_ratio_to_null'])}" for r in rows)
    t2.n = f"{sum(r['UD_ratio_to_null'] is not None for r in rows)} of 9 runs"
    t2.sources = srcs([(cn, s) for cn in ("UD", "none", "LR") for s in CORE01_RUNS[cn]])
    # T3
    none_sh = [val("none", s, "R90", "share_4cycles_1-6") for s in CORE01_RUNS["none"]]
    none_mean = float(np.mean(none_sh)) if all(x is not None for x in none_sh) else None
    rows, vs = [], []
    for s in CORE01_RUNS["R90"]:
        x = val("R90", s, "R90", "share_4cycles_1-6")
        v = combine([cmp(x, ">=", 0.3), cmp(x, ">=", 3.0 * none_mean) if none_mean is not None else PENDING])
        if x is not None and x < 0.3:
            v = FAIL
        vs.append(v)
        rows.append({"run": f"R90 s{s}", "share_4cycles": x, "none_mean": none_mean,
                     "ratio_to_none": (x / none_mean) if (x is not None and none_mean) else None, "verdict": v})
    t3.verdict, t3.details = combine(vs), rows
    t3.value = "; ".join(f"{r['run']}: {fmt(r['share_4cycles'])}" for r in rows) + f"; none mean {fmt(none_mean)}"
    t3.n = "2 R90 runs, 3 none runs"
    t3.sources = srcs([("R90", s) for s in (0, 1)] + [("none", s) for s in (0, 1, 2)])
    # T4
    rows, vs = [], []
    for cond in ("LR", "UD"):
        for s in CORE01_RUNS[cond]:
            f = runs[(cond, s)]
            fin = val(cond, s, cond, "twin_fraction_t50_1-6")
            e5 = fnum(scores_of(epoch_metrics(c, f, 5), cond).get("twin_fraction_t50_1-6")) if f["metrics"] else None
            v = PENDING if fin is None or e5 is None else (PASS if fin > e5 else FAIL)
            vs.append(v)
            rows.append({"run": f"{cond} s{s}", "group": cond, "epoch5": e5, "final": fin, "final_epoch": f["epoch"],
                         "verdict": v})
    t4.verdict, t4.details = combine(vs), rows
    t4.value = "; ".join(f"{r['run']}: {fmt(r['epoch5'])} -> {fmt(r['final'])}" for r in rows)
    t4.n = f"{sum(r['verdict'] != PENDING for r in rows)} of 6 runs"
    t4.sources = srcs([(cn, s) for cn in ("LR", "UD") for s in CORE01_RUNS[cn]])
    # Structure matrix (final, blocks 1 to 6), emergence curves, validation accuracy
    mat, curves, accs = [], [], []
    for (cond, s), f in runs.items():
        if f["metrics"] is None:
            continue
        for grp, key in (("LR", "twin_fraction_t50_1-6"), ("UD", "twin_fraction_t50_1-6"), ("R90", "share_4cycles_1-6"),
                         ("Klein", "share_4cycles_1-6")):
            sc = scores_of(f["metrics"], grp)
            mat.append({"condition": cond, "seed": s, "group": grp, "measure": key.rsplit("_", 1)[0],
                        "value": fnum(sc.get(key)), "ratio_to_null": fnum(sc.get("ratio_to_null_1-6")),
                        "tax_mlp": fnum(sc.get("tax_mlp_1-6")), "epoch": f["epoch"], "final_ok": f["final_ok"],
                        "source_file": c.rel(f["dir"] / "metrics.json"), "git_hash": c.meta_git(f["dir"] / "metrics.json")})
        for ed in sorted(f["dir"].parent.glob("e*")):
            mm = re.fullmatch(r"e(\d+)", ed.name)
            em = c.jload(ed / "metrics.json") if mm else None
            if not em:
                continue
            for grp in ("LR", "UD", "R90", "Klein"):
                sc = scores_of(em, grp)
                if not sc:
                    continue
                curves.append({"condition": cond, "seed": s, "epoch": int(mm.group(1)), "group": grp,
                               "twin_fraction_t50": fnum(sc.get("twin_fraction_t50_1-6")),
                               "share_4cycles": fnum(sc.get("share_4cycles_1-6")),
                               "ratio_to_null": fnum(sc.get("ratio_to_null_1-6")), "tax_mlp": fnum(sc.get("tax_mlp_1-6")),
                               "source_file": c.rel(ed / "metrics.json")})
        va = c.cload(c.p("runs", f["experiment"], f["run"], "val_acc.csv"))
        if va is not None:
            accs.append(va.assign(condition=cond, seed=s, source_file=c.rel(c.p("runs", f["experiment"], f["run"],
                                                                                  "val_acc.csv"))))
    if mat:
        c.data["fig3_matrix"] = pd.DataFrame(mat)
    if curves:
        c.data["fig3_emergence"] = pd.DataFrame(curves)
    if accs:
        c.data["fig3_val_acc"] = pd.concat(accs, ignore_index=True)


# ================================================================================================
# CORE-02 to CORE-07 (files and keys of scripts/core0*_*.py, named in each scorer)
# ================================================================================================
def score_core02(c: Collector) -> None:
    """scripts/core02_refund.py: results/CORE-02/predictions.json (written and committed before any surgery),
    surgery.csv (run, group, condition, relation, scored, top1_original, top1_surgery, drop_points, free_predicted,
    free_achieved, predicted_utc, surgery_utc) and law_per_block.csv (free_fc1/fc2 predicted and achieved per block,
    recon_max). Scored rows: the LR+UD and R90 models (the script's `scored` column, as CORE-02's "new groups")."""
    p1 = c.add(Pred("CORE-02 P2.1", "CORE-02", "Core", "The achieved saving equals the prediction (identity check)",
                    "P2.1: the achieved saving equals the prediction (identity check).",
                    "per block 1 to 6 of every scored model (LR+UD, R90) and group: achieved free-parameter counts of "
                    "fc1 and fc2 (compressed form) equal to the character-formula counts; reconstruction error below "
                    "1e-4; prediction timestamp before the surgery timestamp", "equal in every block"))
    p2 = c.add(Pred("CORE-02 P2.2", "CORE-02", "Core",
                    "Trained-group surgery on blocks 1 to 6 keeps validation top-1 within 3 points",
                    "P2.2: trained-group surgery on blocks 1 to 6 keeps validation top-1 within 3 points.",
                    "|top-1 drop| after surgery with the model's training group (Klein for LR+UD, R90), ImageNet-100 "
                    "validation, every scored model", "<= 3.0 points"))
    p3 = c.add(Pred("CORE-02 P2.3", "CORE-02", "Core", "Every wrong-group surgery loses at least 20 points",
                    "P2.3: every wrong-group surgery loses at least 20 points.",
                    "top-1 drop after surgery with a group that is neither the training group nor one of its subgroups, "
                    "every scored model", ">= 20 points, every row"))
    d = c.p("results", "CORE-02")
    pj, df, law = c.jload(d / "predictions.json"), c.cload(d / "surgery.csv"), c.cload(d / "law_per_block.csv")
    if df is None:
        for pr in (p1, p2, p3):
            pr.reason = "results/CORE-02/surgery.csv missing" + ("" if pj else " (predictions.json missing too)")
        return
    src = c.src(d / "surgery.csv", str(df["git_hash"].iloc[0]) if "git_hash" in df else None)
    srcs = [src] + ([c.src(d / "predictions.json")] if pj else [])
    for pr in (p1, p2, p3):
        pr.sources = list(srcs)
    if "drop_points" not in df and {"top1", "top1_original"} <= set(df.columns):
        df = df.assign(drop_points=df["top1_original"] - df["top1"])
    if "relation" not in df:
        df = df.assign(relation=[("trained" if g == TRAINED_GROUP.get(str(cd)) else
                                  "subgroup" if GROUP_ELEMENTS.get(g, {"x"}) <= GROUP_ELEMENTS.get(
                                      TRAINED_GROUP.get(str(cd), "none"), set()) else "wrong")
                                 for g, cd in zip(df["group"], df["condition"])])
    sc = df[df["scored"].astype(str) == "True"] if "scored" in df else df[df["condition"].isin(["LRUD", "R90"])]
    rows = [{"run": r.get("run"), "condition": r.get("condition"), "group": r["group"], "relation": r["relation"],
             "scored": r.get("scored"), "top1_original": fnum(r.get("top1_original")), "drop": fnum(r["drop_points"]),
             "tax_predicted": fnum(r.get("tax_predicted")), "tax_achieved": fnum(r.get("tax_achieved"))}
            for _, r in df.iterrows()]
    t, w = sc[sc["relation"] == "trained"], sc[sc["relation"] == "wrong"]
    v2 = [within(x, 0.0, 3.0) for x in t["drop_points"]]
    v3 = [cmp(x, ">=", 20.0) for x in w["drop_points"]]
    runs = set(sc["run"]) if "run" in sc else set()
    full = {"LRUD", "R90"} <= set(sc["condition"]) if len(sc) else False
    for pr, vs, sub in ((p2, v2, t), (p3, v3, w)):
        pr.details = rows
        pr.n = f"{len(vs)} rows on {len(runs)} scored models"
        pr.verdict = combine(vs) if vs else PENDING
        if pr.verdict == PASS and not full:
            pr.verdict, pr.reason = PENDING, "the LR+UD and R90 models are not both present"
    p2.value = f"largest |drop| {fmt(t['drop_points'].abs().max())}" if len(t) else None
    p3.value = f"smallest drop {fmt(w['drop_points'].min())}" if len(w) else None
    # P2.1: per-block identity from law_per_block.csv (surgery.csv totals as a fallback)
    parts, notes = [], []
    if law is not None and {"free_fc1_predicted", "free_fc1_achieved", "free_fc2_predicted", "free_fc2_achieved"} \
            <= set(law.columns):
        lw = law[law["run"].isin(runs)] if "run" in law else law
        eq = (lw["free_fc1_predicted"] == lw["free_fc1_achieved"]) & (lw["free_fc2_predicted"] == lw["free_fc2_achieved"])
        parts.append(PASS if bool(eq.all()) and len(lw) else (FAIL if len(lw) else PENDING))
        rmax = fnum(pd.to_numeric(lw["recon_max"], errors="coerce").max()) if "recon_max" in lw else None
        if rmax is not None:
            parts.append(cmp(rmax, "<", 1e-4))
        dpp = fnum((100.0 * (lw["tax_predicted"] - lw["tax_achieved"])).abs().max()) if {"tax_predicted",
                                                                                           "tax_achieved"} <= set(lw) else None
        p1.value = f"{int((~eq).sum())} of {len(lw)} block rows differ; largest |tax difference| {fmt(dpp)} pp; " \
                   f"largest reconstruction error {fmt(rmax)}"
        p1.n = f"{len(lw)} block rows"
        p1.sources.append(c.src(d / "law_per_block.csv"))
    elif {"free_predicted", "free_achieved"} <= set(sc.columns):
        eq = sc["free_predicted"] == sc["free_achieved"]
        parts.append(PASS if bool(eq.all()) and len(sc) else (FAIL if len(sc) else PENDING))
        p1.value = f"{int((~eq).sum())} of {len(sc)} model-group rows differ (totals over blocks 1 to 6)"
    else:
        parts.append(PENDING)
        notes.append("law_per_block.csv missing")
    if {"predicted_utc", "surgery_utc"} <= set(sc.columns) and len(sc):
        late = [f"{r['run']} {r['group']}" for _, r in sc.iterrows()
                if str(r["predicted_utc"]) >= str(r["surgery_utc"])]
        parts.append(FAIL if late else PASS)
        notes.append("every prediction timestamp precedes its surgery" if not late else
                     "prediction written after surgery for " + ", ".join(late))
    committed = git(c.root, "log", "--diff-filter=A", "--format=%cI", "--", "results/CORE-02/predictions.json")
    notes.append(f"predictions.json first committed {committed.splitlines()[-1]}" if committed else
                 "predictions.json not found in git history (commit it before the surgery, PREREG CORE-02)")
    p1.verdict = combine(parts)
    if p1.verdict == PASS and not full:
        p1.verdict = PENDING
        notes.append("the LR+UD and R90 models are not both present")
    p1.reason = "; ".join(notes)
    p1.details = rows
    c.data["table4_core02"] = df.assign(source_file=src["file"])


def score_core03(c: Collector) -> None:
    """scripts/core03_stacking.py: results/CORE-03/curves.csv, one row per curve point with the scored choice at each
    budget point (role lowrank_scored: plain SVD or SVD-LLM, chosen on the method-selection set; role
    mirror_lowrank_scored), plus p31.json."""
    pr = c.add(Pred("CORE-03 P3.1", "CORE-03", "Core",
                    "Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at least 3 "
                    "of 4 budget points on at least 2 of 3 models, including DINOv2-L or DeiT-III (zero-shot)",
                    "P3.1: Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at "
                    "least 3 of 4 budget points (about 85%, 70%, 55% and 40% of the converted span) on at least 2 of 3 "
                    "models, including DINOv2-L or DeiT-III.",
                    "per model and budget point: validation top-1 of the scored Mirror-plus-low-rank point strictly "
                    "above the scored low-rank-alone point (the better of plain SVD and SVD-LLM by method-selection "
                    "k-NN, as curves.csv marks it); zero-shot",
                    ">= 3 of 4 points on >= 2 of 3 models, one of them DINOv2-L or DeiT-III"))
    p = c.p("results", "CORE-03", "curves.csv")
    df = c.cload(p)
    if df is None or not {"model", "point", "role", "val_top1"} <= set(df.columns):
        pr.reason = "results/CORE-03/curves.csv missing"
        return
    pr.sources = [c.src(p, str(df["git_hash"].iloc[0]) if "git_hash" in df else None)]
    c.data["fig4b_stacking"] = df.assign(source_file=c.rel(p))
    models = ("dinov2_vitb14_reg", "dinov2_vitl14_reg", "deit3_b16")
    rows, status = [], {}
    for model in models:
        g = df[df["model"] == model]
        n, hits = 0, 0
        for point, gp in g.groupby("point"):
            a_ = gp[gp["role"] == "mirror_lowrank_scored"]
            b_ = gp[gp["role"] == "lowrank_scored"]
            if not len(a_) or not len(b_):
                continue
            va, vb = fnum(a_["val_top1"].iloc[0]), fnum(b_["val_top1"].iloc[0])
            if va is None or vb is None:
                continue
            n += 1
            hits += int(va > vb)
            rows.append({"model": model, "point": point, "mirror_lowrank": va, "lowrank": vb,
                         "lowrank_method": b_["method"].iloc[0], "above": va > vb})
        status[model] = PASS if hits >= 3 else (FAIL if hits + (4 - n) < 3 else PENDING)
    passed = [m for m, v in status.items() if v == PASS]
    open_ = [m for m, v in status.items() if v == PENDING]
    one_of = ("dinov2_vitl14_reg", "deit3_b16")
    if len(passed) >= 2 and any(m in one_of for m in passed):
        pr.verdict = PASS
    elif len(passed) + len(open_) < 2 or not any(m in one_of for m in passed + open_):
        pr.verdict = FAIL
    pr.details = rows + [{"model": m, "model_verdict": v} for m, v in status.items()]
    pr.value = "; ".join(f"{m}: {sum(r['above'] for r in rows if r['model'] == m)} of "
                         f"{sum(1 for r in rows if r['model'] == m)} points above" for m in models)
    pr.n = f"{len(rows)} budget points"
    pj = c.jload(c.p("results", "CORE-03", "p31.json"))
    if pj and str(pj.get("verdict", "")).upper() not in (pr.verdict, "INCOMPLETE" if pr.verdict == PENDING else ""):
        pr.reason = f"p31.json records {pj.get('verdict')}"


def core04_metrics(c: Collector, model: str):
    for p in (c.p("results", "CORE-04", "structure", model, "metrics.json"), c.p("results", "CORE-04", model,
                                                                                  "metrics.json"),
              c.p("results", "MUST-04", model, "metrics.json")):
        m = c.jload(p)
        if m and scores_of(m, "LR"):
            return p, m
    return None, None


def score_core04(c: Collector) -> None:
    """scripts/core04_generality.py: results/CORE-04/structure/<model>/metrics.json (scripts/measure_groups.py format,
    or mirror/cnn_groups.py with the same keys; MUST-04 metrics reused for its models), scale/knn.csv (original k-NN
    and parameters of DINOv2 S, B, L, g). Recipes from configs/models/<model>.yaml."""
    p1 = c.add(Pred("CORE-04 P4.1", "CORE-04", "Core",
                    "Scale: the tax in blocks 1 to L/2 increases from DINOv2-S to g (either direction is reported)",
                    "P4.1 (scale): the tax in blocks 1 to L/2 increases from DINOv2-S to g.",
                    "LR tax (character formula, MLP), blocks 1 to L/2, DINOv2 S, B, L, g with registers; increases means "
                    "strictly increasing at every step", "S < B < L < g"))
    p2 = c.add(Pred("CORE-04 P4.2", "CORE-04", "Core",
                    "Models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model "
                    "of the same size",
                    "P4.2 (recipe): models whose recipe is verified to have no flips have a lower LR tax than every "
                    "flip-trained model of the same size.",
                    "LR tax, blocks 1 to L/2; verified no-flip: recipe_flip 'no' without a 'probable' or '(verify)' "
                    "confidence in configs/models; same size: same width and depth",
                    "below every flip-trained model of its size"))
    p3 = c.add(Pred("CORE-04 P4.3", "CORE-04", "Core",
                    "CNNs: twin structure appears in ConvNeXt's pointwise MLPs (LR ratio to null at most 0.8 and twin "
                    "fraction at least 0.2 in the first half of the stages' pointwise MLPs)",
                    "P4.3 (CNNs): twin structure appears in ConvNeXt's pointwise MLPs.",
                    "LR ratio to null and LR twin fraction (t 0.5), first half of ConvNeXt-B's pointwise MLPs",
                    "ratio <= 0.8 and twin >= 0.2"))
    import yaml  # noqa: PLC0415

    def cfg(model):
        p = c.p("configs", "models", f"{model}.yaml")
        if not p.is_file():
            p = REPO_ROOT / "configs" / "models" / f"{model}.yaml"
        if not p.is_file():
            return {}
        key = (str(p), p.stat().st_mtime_ns)
        if key not in _YAML_CACHE:
            try:
                _YAML_CACHE[key] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                _YAML_CACHE[key] = {}
        return _YAML_CACHE[key]

    sizes = ("dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg")
    rows, taxes = [], []
    for model in sizes:
        p, m = core04_metrics(c, model)
        t = fnum(scores_of(m, "LR").get("tax_mlp_first_half")) if m else None
        taxes.append(t)
        rows.append({"model": model, "tax_first_half": t, "file": c.rel(p) if p else None})
        if p:
            s = c.src(p)
            p1.sources.append(s)
            c.num(f"core04.{model}.LR.tax_first_half", t, desc=f"CORE-04: LR tax, blocks 1 to L/2, {model}", source=s)
    p1.details = rows
    p1.n = f"{sum(t is not None for t in taxes)} of 4 sizes"
    if all(t is not None for t in taxes):
        inc = all(b > a for a, b in zip(taxes, taxes[1:]))
        dec = all(b < a for a, b in zip(taxes, taxes[1:]))
        p1.verdict = PASS if inc else FAIL
        p1.value = ", ".join(f"{m.split('_')[1][3:]} {fmt(t)}" for m, t in zip(sizes, taxes))
        p1.reason = ("increasing at every step" if inc else "the tax falls with size at every step (reported either "
                     "way, PREREG Section 6)" if dec else "mixed direction (reported either way, PREREG Section 6)")
    else:
        p1.reason = "missing: " + ", ".join(r["model"] for r in rows if r["tax_first_half"] is None)
    cfg_dir = c.p("configs", "models") if c.p("configs", "models").is_dir() else REPO_ROOT / "configs" / "models"
    entries = []
    for y in sorted(cfg_dir.glob("*.yaml")) if cfg_dir.is_dir() else []:
        cf = cfg(y.stem)
        flip = str(cf.get("recipe_flip", "")).strip().lower()
        conf = str(cf.get("recipe_flip_confidence", "")).lower()
        arch = cf.get("architecture") or {}
        p, m = core04_metrics(c, y.stem)
        entries.append({"model": y.stem, "recipe_flip": flip, "verified_no_flip": flip == "no" and "probable" not in conf
                        and "verify" not in conf, "size": f"{arch.get('width')} x {arch.get('blocks')}",
                        "tax_first_half": fnum(scores_of(m, "LR").get("tax_mlp_first_half")) if m else None,
                        "file": c.rel(p) if p else None})
    nof = [e for e in entries if e["verified_no_flip"]]
    vs = []
    for e in nof:
        peers = [x for x in entries if x["recipe_flip"] == "yes" and x["size"] == e["size"]]
        known = [x for x in peers if x["tax_first_half"] is not None]
        if e["tax_first_half"] is None or not known:
            vs.append(PENDING)
            continue
        if any(e["tax_first_half"] >= x["tax_first_half"] for x in known):
            vs.append(FAIL)
        else:
            vs.append(PASS if len(known) == len(peers) else PENDING)
        p2.sources += [{"file": x["file"], "git_hash": None} for x in [e] + known if x["file"]]
    p2.verdict = combine(vs) if vs else PENDING
    p2.details = entries
    p2.n = f"{len(nof)} verified no-flip model(s): " + ", ".join(e["model"] for e in nof)
    p2.value = "; ".join(f"{e['model']}: {fmt(e['tax_first_half'])}" for e in entries if e["tax_first_half"] is not None)
    if not nof:
        p2.reason = "no model in configs/models with a verified no-flip recipe"
    elif p2.verdict == PENDING:
        p2.reason = "flip-trained models of the same size not all measured yet"
    p_, m = core04_metrics(c, "convnext_b")
    sc = scores_of(m, "LR")
    if sc:
        r_, t_ = fnum(sc.get("ratio_to_null_first_half")), fnum(sc.get("twin_fraction_t50_first_half"))
        p3.verdict, p3.value = combine([cmp(r_, "<=", 0.8), cmp(t_, ">=", 0.2)]), f"ratio {fmt(r_)}, twin {fmt(t_)}"
        p3.sources = [c.src(p_)]
    else:
        p3.reason = "results/CORE-04/structure/convnext_b/metrics.json missing"
    frames = []
    sdir = c.p("results", "CORE-04", "structure")
    for model_dir in sorted(sdir.glob("*")) if sdir.is_dir() else []:
        bdf, m = c.cload(model_dir / "blocks.csv"), c.jload(model_dir / "metrics.json")
        if bdf is not None and m:
            frames.append(blocks_frame(bdf, model_dir.name, "CORE-04", m, c.rel(model_dir / "blocks.csv"),
                                       c.meta_git(model_dir / "metrics.json")))
    if frames:
        c.data.setdefault("_blocks", []).extend(frames)
    kp = c.p("results", "CORE-04", "scale", "knn.csv")
    kn = c.cload(kp)
    if kn is not None and {"model", "top1"} <= set(kn.columns):
        for _, r in kn.iterrows():
            c.data.setdefault("_family", []).append({"model": r["model"], "top1": fnum(r["top1"]),
                                                     "params": fnum(r.get("params_total")), "source_file": c.rel(kp)})
            c.num(f"core04.{r['model']}.original_knn", fnum(r["top1"]), desc=f"CORE-04: original k-NN top-1, "
                  f"{r['model']} (paper protocol)", source=c.src(kp, r.get("git_hash")), unit="%")


def score_core05(c: Collector) -> None:
    """scripts/core05_why_cheap.py: results/CORE-05/measures_<model>.csv (op, block, matrix, s_w, s_cov),
    ckpt_bands.csv (run, group, epoch, block, matrix, top_breaking_share, bottom_breaking_share) and rule_score.json
    (cases with hit, complete). The operational choices are that script's (A*_L operator for P5.1; absolute fall of
    the within-band breaking share for P5.2; pooled cases for P5.3)."""
    p1 = c.add(Pred("CORE-05 P5.1", "CORE-05", "Core",
                    "The activation-weighted breaking share is at most half the weight-energy breaking share in blocks "
                    "1 to 6",
                    "P5.1: the activation-weighted breaking share is at most half the weight-energy breaking share in "
                    "blocks 1 to 6.",
                    "dinov2_vitb14_reg, operator A*_L: mean of s_cov over fc1 and fc2 of blocks 1 to 6 over the mean of "
                    "s_w (other models reported)", "<= 0.5"))
    p2 = c.add(Pred("CORE-05 P5.2", "CORE-05", "Core",
                    "Over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster than "
                    "in the bottom band",
                    "P5.2: over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster "
                    "than in the bottom band.",
                    "per run, within-band breaking share (mean over fc1 and fc2 of blocks 1 to 6): fall from the first to "
                    "the last checkpoint, top band against bottom band; every LR and UD run", "top fall > bottom fall"))
    p3 = c.add(Pred("CORE-05 P5.3", "CORE-05", "Core",
                    "The out-of-sample rule is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III "
                    "and MAE",
                    "P5.3: the out-of-sample rule, designed on `dinov2_vitb14_reg` only and frozen before scoring, is "
                    "correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE.",
                    "share of correct cases (every block of the three models, pooled) of the frozen rule against the "
                    "validation ground truth", ">= 0.75"))
    d = c.p("results", "CORE-05")
    rows = []
    for model in ("dinov2_vitb14_reg", "dinov2_vitl14_reg", "deit3_b16", "mae_b16"):
        mp = d / f"measures_{model}.csv"
        m = c.cload(mp)
        if m is None or not {"op", "block", "matrix", "s_w", "s_cov"} <= set(m.columns):
            continue
        sub = m[(m["op"] == "star") & m["matrix"].isin(["fc1", "fc2"]) & m["block"].between(1, 6)]
        ratio = float(sub["s_cov"].mean() / sub["s_w"].mean()) if len(sub) and sub["s_w"].mean() > 0 else None
        rows.append({"model": model, "s_w": float(sub["s_w"].mean()) if len(sub) else None,
                     "s_cov": float(sub["s_cov"].mean()) if len(sub) else None, "ratio": ratio,
                     "n_matrices": int(len(sub)), "verdict": cmp(ratio, "<=", 0.5), "file": c.rel(mp)})
        c.num(f"core05.{model}.act_over_weight_1-6", ratio, desc=f"CORE-05: activation-weighted over weight-energy "
              f"breaking share (A*_L), blocks 1 to 6, {model}", source=c.src(mp), n=int(len(sub)))
    p1.details = rows
    main = next((r for r in rows if r["model"] == MAIN), None)
    if main is None:
        p1.reason = f"results/CORE-05/measures_{MAIN}.csv missing"
    else:
        p1.verdict, p1.value, p1.n = main["verdict"], main["ratio"], f"{main['n_matrices']} matrices"
        p1.sources = [c.src(c.p(main["file"]))]
        if main["n_matrices"] < 12 and p1.verdict == PASS:
            p1.verdict, p1.reason = PENDING, "fewer than 12 matrices"
    bp = d / "ckpt_bands.csv"
    bands = c.cload(bp)
    need = {"run", "epoch", "block", "matrix", "top_breaking_share", "bottom_breaking_share"}
    if bands is not None and need <= set(bands.columns):
        sub = bands[bands["block"].between(1, 6) & bands["matrix"].isin(["fc1", "fc2"])]
        cols = [x for x in ("top_breaking_share", "mid_breaking_share", "bottom_breaking_share") if x in sub]
        curve = sub.groupby(["run", "epoch"])[cols].mean().reset_index()
        if "group" in sub:
            curve = curve.merge(sub.groupby("run")["group"].first().reset_index(), on="run")
        c.data["fig4c_bands"] = curve.assign(source_file=c.rel(bp))
        res, vs = [], []
        for run, g in curve.groupby("run"):
            g = g.sort_values("epoch")
            if len(g) < 2:
                vs.append(PENDING)
                continue
            ft = g["top_breaking_share"].iloc[0] - g["top_breaking_share"].iloc[-1]
            fb = g["bottom_breaking_share"].iloc[0] - g["bottom_breaking_share"].iloc[-1]
            v = PASS if ft > fb else FAIL
            vs.append(v)
            res.append({"run": run, "first_epoch": int(g["epoch"].iloc[0]), "last_epoch": int(g["epoch"].iloc[-1]),
                        "fall_top": float(ft), "fall_bottom": float(fb), "verdict": v})
        n_lr = sum(1 for r in res if "_LR_" in str(r["run"]))
        n_ud = sum(1 for r in res if "_UD_" in str(r["run"]))
        p2.verdict = combine(vs) if vs else PENDING
        if p2.verdict == PASS and (n_lr < 3 or n_ud < 3):
            p2.verdict, p2.reason = PENDING, f"{n_lr} LR and {n_ud} UD runs measured (3 and 3 needed)"
        p2.details, p2.sources, p2.n = res, [c.src(bp)], f"{n_lr} LR and {n_ud} UD runs"
        p2.value = "; ".join(f"{r['run']}: {r['fall_top']:+.3f} vs {r['fall_bottom']:+.3f}" for r in res)
    else:
        p2.reason = "results/CORE-05/ckpt_bands.csv missing"
    rp = d / "rule_score.json"
    rs = c.jload(rp)
    if rs and isinstance(rs.get("cases"), list):
        cases = pd.DataFrame(rs["cases"])
        held = cases[cases["model"].isin(["dinov2_vitl14_reg", "deit3_b16", "mae_b16"])] if "model" in cases else cases
        hits = held["hit"].astype(str).str.lower().isin(["true", "1", "1.0"]) if "hit" in held else None
        share = float(hits.mean()) if hits is not None and len(held) else None
        p3.value, p3.n, p3.sources = share, f"{len(held)} layer-model cases", [c.src(rp)]
        p3.verdict = cmp(share, ">=", 0.75)
        if not rs.get("complete", False) and p3.verdict == PASS:
            p3.verdict, p3.reason = PENDING, f"rule_score.json: {rs.get('verdict')}"
        frozen = c.jload(d / "rule_frozen.json") or {}
        p3.reason = (p3.reason + "; " if p3.reason else "") + (
            f"rule frozen {frozen.get('frozen_utc', 'n/a')}, scored {rs.get('scored_utc', 'n/a')}")
        c.num("core05.p53.share_correct", share, desc="CORE-05 P5.3: share of correct layer-model cases (frozen rule)",
              source=c.src(rp), n=len(held))
    else:
        p3.reason = "results/CORE-05/rule_score.json missing"


def score_core06(c: Collector) -> None:
    """scripts/core06_orientation.py: results/CORE-06/probes.csv (variant, task, features, test_set, acc,
    cv_top1_train, n_pairs_differ_not_near_tie). Scored, as that script defines: P6.1 and P6.2 on every model probed
    (the original and each healed Mirror-exact seed); P6.2 from the odd CLS, or from the odd mean patch token when
    the odd CLS's training-split cross-validated facing accuracy is below 85%; P6.3 on every healed seed."""
    p1 = c.add(Pred("CORE-06 P6.1", "CORE-06", "Core", "Species from the even part is within 1 point of the full feature",
                    "P6.1: species from the even part is within 1 point of the full feature.",
                    "|species top-1 of the full CLS minus the even CLS|, natural CUB test images, original and every "
                    "healed Mirror-exact seed", "<= 1.0 points, every model"))
    p2 = c.add(Pred("CORE-06 P6.2", "CORE-06", "Core", "Facing direction from the odd part is at least 85%",
                    "P6.2: facing direction from the odd part is at least 85%.",
                    "facing accuracy from the odd part (CLS, or mean patch token when the odd CLS's training-split CV "
                    "accuracy is below 85%), natural test images, every model probed", ">= 85%, every model"))
    p3 = c.add(Pred("CORE-06 P6.3", "CORE-06", "Core",
                    "Facing direction from healed Mirror-exact's even part is at most 60% on natural test images, and "
                    "exactly 50% on the mirror-augmented test set",
                    "P6.3: facing direction from healed Mirror-exact's even part is at most 60% on natural test images, "
                    "and exactly 50% on the mirror-augmented test set.",
                    "facing accuracy from the even CLS of every healed Mirror-exact seed; mirror-augmented: 50% with "
                    "every differing pair a near-tie (relative logit margin < 1e-4)", "<= 60% and = 50%"))
    p = c.p("results", "CORE-06", "probes.csv")
    df = c.cload(p)
    if df is None or not {"variant", "task", "features", "test_set", "acc"} <= set(df.columns):
        for pr in (p1, p2, p3):
            pr.reason = "results/CORE-06/probes.csv missing"
        return
    s = c.src(p, str(df["git_hash"].iloc[0]) if "git_hash" in df else None)
    for pr in (p1, p2, p3):
        pr.sources = [s]

    def get(v, task, feat, test="natural", col="acc"):
        sub = df[(df["variant"] == v) & (df["task"] == task) & (df["features"] == feat) & (df["test_set"] == test)]
        return fnum(sub[col].iloc[0]) if len(sub) and col in sub else None

    variants = list(dict.fromkeys(df["variant"]))
    healed = [v for v in variants if str(v).startswith(("vexact", "healed"))]
    scored = (["original"] if "original" in variants else []) + healed
    r1, r2, r3, v1, v2, v3 = [], [], [], [], [], []
    for v in scored:
        ev, fu = get(v, "species", "even_cls"), get(v, "species", "full_cls")
        x = within(ev, fu, 1.0)
        v1.append(x)
        r1.append({"variant": v, "species_even": ev, "species_full": fu, "verdict": x})
        cv = get(v, "facing", "odd_cls", col="cv_top1_train")
        feat = "odd_patch" if (cv is not None and cv < 85.0 and get(v, "facing", "odd_patch") is not None) else "odd_cls"
        od = get(v, "facing", feat)
        x = cmp(od, ">=", 85.0)
        v2.append(x)
        r2.append({"variant": v, "feature": feat, "odd_cls_cv_train": cv, "facing_odd": od, "verdict": x})
    for v in healed:
        nat, aug = get(v, "facing", "even_cls"), get(v, "facing", "even_cls", "mirror_augmented")
        bad = get(v, "facing", "even_cls", "mirror_augmented", "n_pairs_differ_not_near_tie")
        exact = PENDING if aug is None else (PASS if aug == 50.0 or (bad is not None and bad == 0) else
                                             (FAIL if bad is not None else PENDING))
        x = combine([cmp(nat, "<=", 60.0), exact])
        v3.append(x)
        r3.append({"variant": v, "facing_even_natural": nat, "facing_even_mirror_augmented": aug,
                   "pairs_differ_not_near_tie": bad, "verdict": x})
    for pr, vs, rr in ((p1, v1, r1), (p2, v2, r2), (p3, v3, r3)):
        pr.details = rr
        pr.verdict = combine(vs) if vs else PENDING
        pr.n = f"{len(rr)} model(s)"
        if pr.verdict == PASS and (not healed or (pr is not p3 and "original" not in variants)):
            pr.verdict, pr.reason = PENDING, "the original and the healed Mirror-exact seeds are not all probed"
    p1.value = "; ".join(f"{r['variant']}: {fmt(r['species_full'])} vs {fmt(r['species_even'])}" for r in r1)
    p2.value = "; ".join(f"{r['variant']}: {fmt(r['facing_odd'])}" for r in r2)
    p3.value = "; ".join(f"{r['variant']}: {fmt(r['facing_even_natural'])} / {fmt(r['facing_even_mirror_augmented'])}"
                         for r in r3)


def score_core07(c: Collector) -> None:
    """scripts/core07_shift.py: results/CORE-07/shift_rows.csv (variant, role, val_top1, v2_top1, v2_drop)."""
    pr = c.add(Pred("CORE-07 P7.1", "CORE-07", "Core",
                    "Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within 1 point "
                    "of the original's drop",
                    "P7.1: Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within "
                    "1 point of the original's drop.",
                    "|mean over healed Mirror-exact seeds of (ImageNet top-1 - ImageNet-V2 top-1) minus the original's|, "
                    "same linear probe", "<= 1.0 points"))
    p = c.p("results", "CORE-07", "shift_rows.csv")
    df = c.cload(p)
    if df is None or "v2_drop" not in df:
        pr.reason = "results/CORE-07/shift_rows.csv missing (or no ImageNet-V2 column)"
        return
    pr.sources = [c.src(p, str(df["git_hash"].iloc[0]) if "git_hash" in df else None)]
    role = df["role"] if "role" in df else df["variant"].map(lambda v: "original" if v == "original" else
                                                              "mirror_exact" if str(v).startswith("vexact") else "other")
    o, m = df[role == "original"], df[role == "mirror_exact"]
    do = fnum(o["v2_drop"].iloc[0]) if len(o) else None
    dm = float(m["v2_drop"].mean()) if len(m) else None
    pr.verdict = within(dm, do, 1.0)
    if pr.verdict == PASS and len(m) < 3:
        pr.verdict, pr.reason = PENDING, f"{len(m)} healed Mirror-exact seed(s); the headline uses 3"
    pr.value = None if dm is None or do is None else f"drops: Mirror-exact {dm:.2f} (mean of {len(m)}), original {do:.2f}"
    pr.n = f"{len(m)} Mirror-exact seed(s)"
    pr.details = df.drop(columns=[x for x in df.columns if x.startswith("probe_")]).to_dict("records")
    c.data["table2_shift"] = df.assign(source_file=c.rel(p))
    if dm is not None and do is not None:
        c.num("core07.drop_difference", dm - do, desc="CORE-07: Mirror-exact minus original drop to ImageNet-V2",
              source=pr.sources[0], n=len(m), unit="points")


# ================================================================================================
# Upside: reported, never scored
# ================================================================================================
UP_TEXT = {
    "UP-01": "UD tax rises with p (0.1, 0.25, 0.5)",
    "UP-02": "breaking share stays below 0.05 throughout training from a UD-equivariant start, at accuracy within 1 "
             "point of the UD runs",
    "UP-03": "the trained-group tax differs between ViT-S and ViT-Ti; either direction is reported",
    "UP-04": "tied-from-scratch reaches about the same accuracy at the reduced parameter count",
    "UP-05": "a speed claim needs more than 1.1x the original's speed; otherwise it is a limitation",
    "UP-06": "the extra drop from INT8 weight-only quantisation and from ToMe (r = 8) is the same for both models within "
             "0.5 points",
    "UP-07": "pairing from weights alone agrees with the data-driven pairing on at least 70% of pairs",
}
UP_FILES = {"UP-05": ("metrics.json",), "UP-06": ("comparison.json",), "UP-07": ("up07.json",)}


def flat_scalars(obj, prefix: str = "", depth: int = 2, limit: int = 30) -> dict:
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, (bool, int, float, str)):
            if isinstance(v, str) and len(v) > 80:
                continue
            out[key] = v
        elif isinstance(v, dict) and depth > 1:
            out.update(flat_scalars(v, key + ".", depth - 1, limit))
        if len(out) >= limit:
            break
    return out


def score_upside(c: Collector) -> None:
    runs = core01_runs(c)
    for up, text in UP_TEXT.items():
        pr = c.add(Pred(up, up, "Upside", text[0].upper() + text[1:], f"{up}: {text}.", "reported as measured, not "
                        "scored (PREREG Section 4)", "none (reported)"))
        rows = []
        if up == "UP-01":
            for p_ in ("0.1", "0.25"):
                f = run_final(c, "UP-01", f"{CORE_ARCH}_UD_p{p_}_s0")
                rows.append({"p": float(p_), "run": f["run"], "UD_tax_1-6": fnum(scores_of(f["metrics"], "UD").get(
                    "tax_mlp_1-6")), "epoch": f["epoch"]})
                if f["metrics"] is not None:
                    pr.sources.append(c.src(f["dir"] / "metrics.json"))
            ud = [fnum(scores_of(runs[("UD", s)]["metrics"], "UD").get("tax_mlp_1-6")) for s in (0, 1, 2)
                  if runs[("UD", s)]["metrics"] is not None]
            rows.append({"p": 0.5, "run": "CORE-01 UD runs (mean)", "UD_tax_1-6": float(np.mean(ud)) if ud else None})
            vals = [r["UD_tax_1-6"] for r in rows]
            if all(v is not None for v in vals):
                pr.verdict = REPORTED
                pr.value = " -> ".join(fmt(v) for v in vals) + (
                    " (rises)" if all(b > a for a, b in zip(vals, vals[1:])) else " (does not rise at every step)")
        elif up == "UP-03":
            for cond in ("none", "LR", "UD"):
                f = run_final(c, "UP-03", f"vit_ti8_{cond}_s0")
                grp = cond if cond != "none" else "LR"
                s_ = [fnum(scores_of(runs[(cond, s)]["metrics"], grp).get("tax_mlp_1-6")) for s in (0, 1, 2)
                      if runs[(cond, s)]["metrics"] is not None]
                rows.append({"condition": cond, "group": grp, "ViT-Ti_tax_1-6": fnum(scores_of(f["metrics"], grp).get(
                    "tax_mlp_1-6")), "ViT-S_tax_1-6_mean": float(np.mean(s_)) if s_ else None})
                if f["metrics"] is not None:
                    pr.sources.append(c.src(f["dir"] / "metrics.json"))
            if any(r["ViT-Ti_tax_1-6"] is not None for r in rows):
                pr.verdict = REPORTED
                pr.value = "; ".join(f"{r['condition']}: Ti {fmt(r['ViT-Ti_tax_1-6'])}, S {fmt(r['ViT-S_tax_1-6_mean'])}"
                                     for r in rows)
        elif up in ("UP-02", "UP-04"):
            base = c.p("results", up)
            for d in sorted(base.iterdir()) if base.is_dir() else []:
                sj = c.jload(d / "summary.json") if d.is_dir() else None
                if sj:
                    rows.append({"run": d.name, **flat_scalars(sj)})
                    pr.sources.append(c.src(d / "summary.json"))
            if rows:
                pr.verdict, pr.value = REPORTED, f"{len(rows)} run(s); see details"
        else:
            for name in UP_FILES.get(up, ("metrics.json",)):
                p = c.p("results", up, name)
                m = c.jload(p)
                if m:
                    rows.append(flat_scalars(m))
                    pr.sources.append(c.src(p))
            if rows:
                pr.verdict, pr.value = REPORTED, "see details"
        pr.details = rows
        if pr.verdict != REPORTED:
            pr.reason = f"results/{up}/ has no finished output" + (" (no script yet)" if up == "UP-05" else "")


# ================================================================================================
# Gates A and E; earlier phase
# ================================================================================================
def score_gate_a(c: Collector) -> None:
    g = Gate("A", "Before any new GPU job", "PREREG_NEXT_PHASE.md committed after sign-off; MUST-00 clean")
    c.gates.insert(0, g)
    au_p = c.p("results", "MUST-00", "audit.json")
    au = c.jload(au_p)
    pre = c.prereg
    g.sources = [{"file": PREREG_FILE, "git_hash": pre["signoff_commit"]}] + ([c.src(au_p)] if au else [])
    counts = (au or {}).get("counts") or {}
    g.value = (f"sign-off commit {short(pre['signoff_commit'])}; MUST-00: " +
               (", ".join(f"{k} {v}" for k, v in counts.items()) if counts else "audit missing"))
    if pre["signed_off"] and pre["signoff_commit"] and au:
        g.status = MET
        g.evidence = ("PREREG_NEXT_PHASE.md carries the sign-off line and was committed; MUST-00 audit present "
                      "(mismatches reported in results/MUST-00/audit.md, accepted per results/STATUS.md)")
    elif not pre["signed_off"]:
        g.status, g.evidence = NOT_MET, "PREREG_NEXT_PHASE.md lacks the sign-off line"
    else:
        g.evidence = "sign-off commit or MUST-00 audit not found"


def score_gate_e(c: Collector) -> None:
    g = c.gates_append(Gate("E", "Before abstract registration (23 Nov)", "MUST-04, MUST-05, MUST-07, MUST-12 done; "
                            "CORE-01 running"))
    done = {}
    done["MUST-04"] = all(c.p("results", "MUST-04", m, "metrics.json").is_file() for m in MUST04_MODELS)
    done["MUST-05"] = c.p("results", "MUST-05", MAIN, "metrics.json").is_file()
    r7 = rows_frame(c, "MUST-07", MAIN)[0]
    done["MUST-07"] = r7 is not None and all(set(SEEDS3) <= set(seed_values(r7, s, "mirror_exact")) for s in SIX_SIZES)
    done["MUST-12"] = latest_epoch_dir(c.p("results", "MUST-12", f"{CORE_ARCH}_LR_s0"))[0] is not None
    core = c.p("runs", "CORE-01")
    running = core.is_dir() and any(core.iterdir())
    g.value = "; ".join(f"{k} {'done' if v else 'not done'}" for k, v in done.items()) + \
        f"; CORE-01 {'started' if running else 'not started'}"
    g.status = MET if all(done.values()) and running else PENDING
    g.evidence = "files present (done = the experiment's scored outputs exist)"


OLD_NAMES = {"p2": "F4 energy rule (P2): predicts the single-block winner", "p5_part2": "Weights-only twin pairing "
             "(P5 part 2): agreement with the data-driven pairing", "p5_part2_strict": "Weights-only twin pairing, fully "
             "data-free (P5 part 2, strict)"}


def earlier_failures(c: Collector) -> list[dict]:
    nj_p = c.p("results", "paper", "numbers.json")
    nj = c.jload(nj_p)
    if not nj:
        c.notes.append("results/paper/numbers.json missing: earlier-phase failed checks not listed")
        return []
    tp = c.cload(c.p("results", "paper", "tables", "T_prereg.csv"))
    stmt = {}
    if tp is not None and "id" in tp:
        for _, r in tp.iterrows():
            stmt[re.sub(r"[^a-z0-9]", "", str(r["id"]).lower())] = r
    out = []
    for key, e in nj.items():
        if not key.endswith(".verdict") or str(e.get("value")).upper() != "FAIL":
            continue
        base = key[: -len(".verdict")]
        val = nj.get(base + ".value", {}).get("value")
        norm = re.sub(r"[^a-z0-9]", "", re.sub(r"^m3\.", "", base).lower())
        r = stmt.get(norm)
        x4 = re.fullmatch(r"x4\.(\w+)\.(float\d+)", base)
        name = OLD_NAMES.get(base) or (str(r["statement"]) if r is not None else None) or (
            f"X4 folding check ({x4.group(1)}, {x4.group(2)})" if x4 else base)
        out.append({"key": base, "name": name,
                    "value": val if val is not None else (fnum(r["value"]) if r is not None and "value" in r else None),
                    "bar": (str(r["threshold"]) if r is not None and "threshold" in r and pd.notna(r["threshold"]) else ""),
                    "verdict": "FAIL", "source_file": e.get("source_file"), "git_hash": e.get("git_hash"),
                    "registry": "results/paper/numbers.json"})
    order = {"p2": 0, "p5_part2": 1, "p5_part2_strict": 2}
    out.sort(key=lambda x: (order.get(x["key"], 9), x["key"]))
    return out


def earlier_numbers(c: Collector) -> list[dict]:
    au = c.jload(c.p("results", "MUST-00", "audit.json"))
    if not au:
        return []
    return [{"id": r.get("id"), "topic": r.get("topic"), "claim": r.get("claim"), "value_found": r.get("value_found"),
             "source": r.get("source"), "audit_verdict": r.get("verdict")} for r in au.get("rows") or []]


# ================================================================================================
# Figure data that combine experiments
# ================================================================================================
def finish_data(c: Collector) -> None:
    if c.data.get("_blocks"):
        c.data["fig2_blocks"] = pd.concat(c.data.pop("_blocks"), ignore_index=True)
    else:
        c.data.pop("_blocks", None)
    fam = c.data.pop("_family", [])
    pts = []
    m7 = c.data.get("must07_rows")
    if m7 is not None:
        for _, r in m7.iterrows():
            pts.append({"model": r["model"], "method": r["method"], "regime": "zero-shot", "size": r["size"],
                        "params_frac": r["model_params_frac"], "params": r["model_params"], "top1": r["top1_mean"],
                        "top1_std": r["top1_std"], "n": r["n_seeds"], "source_file": r["source_file"],
                        "git_hash": r["git_hash"]})
    t1 = c.data.get("table1_healed")
    if t1 is not None:
        for _, r in t1.iterrows():
            if r["targets"] != "sym":
                continue
            pts.append({"model": MAIN, "method": r["method"], "regime": "healed", "size": "full",
                        "params_frac": r["params_frac"], "params": r.get("params_model"), "top1": r["knn_mean"],
                        "top1_std": r["knn_std"],
                        "n": r["n_seeds"], "source_file": r["source_file"], "git_hash": r["git_hash"]})
    for f in fam:
        pts.append({"model": f["model"], "method": "original", "regime": "released family", "size": "original",
                    "params_frac": None, "params": f["params"], "top1": f["top1"], "top1_std": None, "n": 1,
                    "source_file": f["source_file"], "git_hash": None})
    if pts:
        c.data["fig5_points"] = pd.DataFrame(pts)
    # Table 4: generality and the law across models and groups
    rows = []
    for exp, base in (("MUST-04", c.p("results", "MUST-04")), ("CORE-04", c.p("results", "CORE-04", "structure"))):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            m = c.jload(d / "metrics.json")
            if not m or not m.get("scores"):
                continue
            for grp, sc in m["scores"].items():
                rng = "1-6" if "ratio_to_null_1-6" in sc else "first_half"
                rows.append({"experiment": exp, "model": d.name, "group": grp, "range": rng,
                             "ratio_to_null": fnum(sc.get(f"ratio_to_null_{rng}")),
                             "twin_fraction_t50": fnum(sc.get(f"twin_fraction_t50_{rng}")),
                             "share_4cycles": fnum(sc.get(f"share_4cycles_{rng}")),
                             "tax_mlp_first_half": fnum(sc.get("tax_mlp_first_half")),
                             "surgery_drop": fnum(((m.get("surgery") or {}).get(f"{grp}_1-6") or {}).get("drop_points")),
                             "source_file": c.rel(d / "metrics.json"), "git_hash": c.meta_git(d / "metrics.json")})
    law = c.data.get("appendix_law")
    if rows or law is not None:
        c.data["table4_generality"] = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["experiment", "model", "group"])
    old = c.cload(c.p("results", "paper", "tables", "T_ablations.csv"))
    if old is not None:
        c.data["table3_earlier_ablations"] = old.assign(source_file="results/paper/tables/T_ablations.csv")


# ================================================================================================
# Writing
# ================================================================================================
def md_table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return out + [""]


def src_txt(sources, limit: int = 3) -> str:
    if not sources:
        return "n/a"
    seen, parts = set(), []
    for s in sources:
        key = (s.get("file"), s.get("git_hash"))
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"`{s.get('file')}` ({short(s.get('git_hash'))})")
    more = f" and {len(parts) - limit} more" if len(parts) > limit else ""
    return "; ".join(parts[:limit]) + more


def detail_lines(pr: Pred) -> list[str]:
    head = [f"### {pr.id}: {pr.verdict}", "", f"- Prediction: {pr.statement}.", f"- Scored quantity: {pr.quantity}.",
            f"- Bar: {pr.bar}. Value: {pr.value if isinstance(pr.value, str) else fmt(pr.value)}. n: "
            f"{pr.n if isinstance(pr.n, str) else fmt(pr.n)}.",
            f"- Sources: {src_txt(pr.sources, 6)}; pre-registration commit {short(pr.prereg_commit)}"
            + ("" if pr.quote_found is not False else " (quoted wording NOT found in the current prereg)") + "."]
    if pr.reason:
        head.append(f"- Notes: {pr.reason}.")
    head.append("")
    recs = [d for d in (pr.details or []) if isinstance(d, dict)]
    if not recs:
        return head
    return head + _detail_table(recs)


def _detail_table(recs: list) -> list[str]:
    cols = []
    for d in recs:
        for k in d:
            if k not in cols and k not in ("file", "invariance_note", "flip_note") and not k.startswith("_"):
                cols.append(k)
    cols = cols[:14]
    rows = [[fmt(d.get(k)) if not isinstance(d.get(k), (list, tuple, dict)) else
             json.dumps(jsonable(d.get(k)))[:60] for k in cols] for d in recs[:40]]
    out = md_table(cols, rows)
    if len(recs) > 40:
        out.append(f"({len(recs) - 40} more rows in PAPER_NUMBERS.json)")
        out.append("")
    return out


def write_outputs(c: Collector, md_path: Path, json_path: Path, old_fail: list, old_nums: list, git_self: dict) -> dict:
    counts = {k: sum(1 for p in c.preds if p.verdict == k) for k in (PASS, FAIL, PENDING, REPORTED)}
    scored = [p for p in c.preds if p.phase != "Upside"]
    pre = c.prereg
    L = ["# FINAL_RESULTS: pre-registered predictions and cited numbers (next phase)", "",
         f"Written by `scripts/collect_next_phase.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, code "
         f"commit `{short(git_self.get('git_hash'))}`{' (uncommitted changes)' if git_self.get('git_dirty') else ''}. "
         "Every number below is read from a result file written by a script of this repository and is listed with "
         "that file and the commit recorded in the run's meta.json. Nothing here is typed by hand; rerun the collector "
         "when new results arrive.", "",
         f"Pre-registration: `{PREREG_FILE}`, signed-off commit `{short(pre['signoff_commit'])}` (sha256 of the current "
         f"text `{(pre['sha256'] or 'n/a')[:12]}`). Scored predictions: {len(scored)}; PASS {counts[PASS]}, FAIL "
         f"{counts[FAIL]}, PENDING {sum(1 for p in scored if p.verdict == PENDING)}. Upside items reported: "
         f"{counts[REPORTED]} of {len(UP_TEXT)}.", ""]
    bad_quotes = [p.id for p in c.preds if p.quote_found is False]
    if bad_quotes:
        L += ["**Warning:** the prereg wording quoted for these items was not found in the current "
              f"{PREREG_FILE}: " + ", ".join(bad_quotes) + ".", ""]
    L += ["## Gates", ""]
    L += md_table(["Gate", "When", "Condition", "Status", "Value", "Evidence", "Source"],
                  [[g.gate, g.when, g.condition, g.status, g.value or "n/a", g.evidence or "", src_txt(g.sources, 2)]
                   for g in sorted(c.gates, key=lambda x: x.gate)])
    L += ["## Pre-registered predictions", "",
          "PENDING means an input is missing or incomplete (the reason is given). Failed predictions are failures and "
          "are reported as such.", ""]
    L += md_table(["ID", "Prediction", "Value", "Bar", "n", "Verdict", "Source (commit)"],
                  [[p.id, p.statement, fmt(p.value) if not isinstance(p.value, str) else p.value, p.bar,
                    fmt(p.n) if not isinstance(p.n, str) else p.n, f"**{p.verdict}**", src_txt(p.sources, 2)]
                   for p in c.preds])
    L += ["## Scoring details", "",
          "For each prediction: the scored quantity as computed, the reason for a PENDING verdict, and the values per "
          "model, run, seed or seed pair.", ""]
    for p in c.preds:
        L += detail_lines(p)
    L += ["## Numbers the paper will cite (new phase)", "",
          "Key, value and source file (with the commit recorded by the run). Values are unrounded in PAPER_NUMBERS.json.",
          ""]
    L += md_table(["Key", "Value", "Unit", "n", "Description", "Source"],
                  [[k, fmt(e["value"]) if not isinstance(e["value"], str) else e["value"], e["unit"] or "",
                    fmt(e["n"]) if e["n"] is not None else "", e["description"], src_txt(e["sources"], 2)]
                   for k, e in sorted(c.numbers.items())])
    L += ["## Failed checks (reported as failures)", "", "### New phase", ""]
    nf = [p for p in c.preds if p.verdict == FAIL]
    L += (md_table(["ID", "Prediction", "Value", "Bar", "Source"],
                   [[p.id, p.statement, fmt(p.value) if not isinstance(p.value, str) else p.value, p.bar,
                     src_txt(p.sources)] for p in nf]) if nf else ["None so far.", ""])
    L += ["### Earlier phase (from results/paper/numbers.json)", ""]
    L += (md_table(["Check", "Key", "Value", "Bar", "Verdict", "Source"],
                   [[o["name"], o["key"], fmt(o["value"]), o["bar"], o["verdict"],
                     f"`{o['source_file']}` ({short(o['git_hash'])})"] for o in old_fail]) if old_fail else
          ["results/paper/numbers.json not found.", ""])
    L += ["## Earlier-phase numbers the paper cites (MUST-00 audit of PROJECT_NEXT_PHASE.md Section 3)", "",
          "As audited in `results/MUST-00/audit.json`; CONTEXT and MISMATCH rows need the corrected wording of "
          "`results/MUST-00/audit.md` before they are cited.", ""]
    L += (md_table(["#", "Topic", "Claim", "Value found", "Source", "Audit"],
                   [[o["id"], o["topic"], o["claim"], o["value_found"], o["source"], o["audit_verdict"]]
                    for o in old_nums]) if old_nums else ["results/MUST-00/audit.json not found.", ""])
    L += ["## Scoring conventions", "",
          "- A claim over several models, runs or seed pairs is FAIL as soon as one available unit fails, PASS when "
          "every unit is present and passes, otherwise PENDING.",
          "- \"Within x points\" is read literally: |difference| <= x.",
          "- Exact invariance or equivariance (MUST-02, MUST-09 P9.2, MUST-11 P11.2): PASS at a raw 100%, or when every "
          "disagreement has a recorded relative top-two margin below 1e-4 (PREREG Section 1); if disagreements exist "
          "and their margins are not recorded the item is PENDING. In P9.2 the allowance applies to the exactly "
          "invariant models only; a compressor fails the prediction only at a raw fp32 share of 100%.",
          "- \"Final\" (CORE-01, MUST-12) is the latest measured checkpoint, and must equal the run's planned epoch count "
          "(runs/<exp>/<run>/meta.json config.epochs).",
          "- MUST-03: twin pairs must be equal and breaking shares within 0.005 (half a unit of the quoted 2-decimal "
          "values).",
          "- CORE-02: a surgery group is wrong unless all its elements are in the trained group (LR+UD trains the Klein "
          "group, R90 the rotations); \"equals\" in P2.1 uses the MUST-01 tolerance of 1 percentage point.",
          "- CORE-04 P4.1: \"increases from S to g\" means strictly increasing across S, B, L and g.", ""]
    if c.notes:
        L += ["## Collector notes", ""] + [f"- {n}" for n in c.notes] + [""]
    L += ["## Inputs read", "", f"{len(c.inputs)} files; their sha256 are in PAPER_NUMBERS.json (`inputs`).", ""]
    text = "\n".join(L) + "\n"
    if "\u2014" in text:
        text = text.replace("\u2014", "-")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    payload = {
        "schema": "paper_numbers/next_phase/v1",
        "written_by": "scripts/collect_next_phase.py",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collector_git": git_self,
        "root": "." if c.root == REPO_ROOT.resolve() else c.root.as_posix(),
        "prereg": {k: v for k, v in pre.items() if k != "text_norm"},
        "counts": {"scored": len(scored), PASS: counts[PASS], FAIL: counts[FAIL],
                   PENDING: sum(1 for p in scored if p.verdict == PENDING), REPORTED: counts[REPORTED]},
        "gates": [asdict(g) for g in sorted(c.gates, key=lambda x: x.gate)],
        "predictions": [asdict(p) for p in c.preds],
        "numbers": c.numbers,
        "failed_checks": {"new_phase": [{"id": p.id, "statement": p.statement, "value": p.value, "bar": p.bar,
                                         "sources": p.sources} for p in nf], "earlier_phase": old_fail},
        "earlier_numbers_audit": old_nums,
        "data_files": sorted(f"{k}.csv" for k in c.data) + sorted(c.copies),
        "notes": c.notes,
        "inputs": c.inputs,
    }
    json_path.write_text(json.dumps(jsonable(payload), indent=1, default=str), encoding="utf-8")
    return payload


def write_data(c: Collector) -> None:
    ddir = c.paper_dir / "data"
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True, exist_ok=True)
    for name, df in c.data.items():
        if isinstance(df, pd.DataFrame):
            out = df.copy()
            for col in out.columns:
                if out[col].map(lambda x: isinstance(x, (list, tuple, dict))).any():
                    out[col] = out[col].map(lambda x: json.dumps(jsonable(x)) if isinstance(x, (list, tuple, dict)) else x)
            out.to_csv(ddir / f"{name}.csv", index=False)
    for name, src in c.copies.items():
        shutil.copyfile(src, ddir / name)


# ================================================================================================
# Main
# ================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT), help="tree holding results/ and runs/ (default: the repository)")
    p.add_argument("--final-md", default=None, help="default <root>/FINAL_RESULTS.md")
    p.add_argument("--numbers-json", default=None, help="default <root>/PAPER_NUMBERS.json")
    p.add_argument("--paper-dir", default=None, help="default <root>/results/paper_next (data/ is written there)")
    return p.parse_args(argv)


SCORERS = (score_gate_a, score_must01, score_must02, score_must03, score_must04, score_must05, score_must06,
           score_must07, score_gate_b, score_must08, score_must09, score_must10, score_must11, score_gate_c,
           score_core01, score_core02, score_core03, score_core04, score_core05, score_core06, score_core07,
           score_upside, score_gate_e)


def main(argv=None) -> dict:
    a = parse_args(argv)
    root = Path(a.root).resolve()
    paper_dir = Path(a.paper_dir) if a.paper_dir else root / "results" / "paper_next"
    c = Collector(root, paper_dir)
    log(f"collecting from {root}; prereg sign-off commit {short(c.prereg['signoff_commit'])}")
    for fn in SCORERS:
        try:
            fn(c)
        except Exception as e:  # noqa: BLE001  a broken input must not hide the other results
            c.notes.append(f"{fn.__name__} raised {type(e).__name__}: {e}")
            log(f"WARNING {fn.__name__}: {type(e).__name__}: {e}")
    finish_data(c)
    gs = git(REPO_ROOT, "rev-parse", "HEAD")
    git_self = {"git_hash": gs, "git_dirty": bool(git(REPO_ROOT, "status", "--porcelain", "--untracked-files=no"))
                if gs else None}
    old_fail, old_nums = earlier_failures(c), earlier_numbers(c)
    md = Path(a.final_md) if a.final_md else root / "FINAL_RESULTS.md"
    js = Path(a.numbers_json) if a.numbers_json else root / "PAPER_NUMBERS.json"
    write_data(c)
    payload = write_outputs(c, md, js, old_fail, old_nums, git_self)
    cnt = payload["counts"]
    log(f"{cnt['scored']} scored predictions: PASS {cnt[PASS]}, FAIL {cnt[FAIL]}, PENDING {cnt[PENDING]}; "
        f"{len(c.numbers)} numbers; wrote {md}, {js} and {paper_dir / 'data'}")
    return payload


if __name__ == "__main__":
    main()

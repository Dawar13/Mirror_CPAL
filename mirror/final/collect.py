"""Final-phase collector: every rule of PREREG_FINAL_PHASE.md scored mechanically, Gate P and the A01 decision for
the runner, and the deliverable tables T1 to T8 (FINAL_PHASE_PLAN.md v2, Sections 6, 8 (End) and 10).

    python scripts/collect_final_phase.py      # results/FINAL_PHASE_REPORT.md, results/final/numbers.json,
                                               # results/final/tables/, the "final_phase" section of PAPER_NUMBERS.json
    python scripts/fp_gate_p.py                # results/A01/gate_p.json (the runner's stop_when reads "pass")
    python scripts/fp_a01_decide.py            # results/A01/decision.json (the runner's when reads "adopted")

CPU only: numpy and pandas, mirror.final.stats.hbootstrap for every interval, and mirror.final.baselines.sizes (with
it torch) for A06-2 only. Nothing is typed by hand: every value is read from a result file written by a script of
this repository and is listed with that file and the commit recorded in its meta.json (eval/step<S>_meta.json for an
evaluation file). Style and conventions follow scripts/collect_next_phase.py, the previous phase's collector.

Verdicts (PREREG_FINAL_PHASE.md Section 3): PASS, FAIL, PENDING (an input is missing or incomplete; the reason is
given), NOT RUN (an input was cut by the Section 9 order or the budget governor, or skipped by hand; never counted as
passed) and REPORTED (report-only rules: A04-2, A06-3, A09-3, A11 and A15a). Units combine as Section 3 says: FAIL as
soon as one available unit fails, then NOT RUN if a needed unit was cut, then PENDING; PASS only when every unit is
present and passes. "At least k of n" rules (A06-1, A06-2) pass once k units pass and fail once k can no longer be
reached. A value within 1e-9 of a bar counts as equal to the bar (binary rounding of decimal percentages): >= and <=
then hold, > and < do not. "Within x" is |difference| <= x. Exact invariance ("100%") passes at a raw 100%, or when
every disagreement is a near-tie in both copies (relative top-two margin below 1e-4 in the image and in its mirror,
PREREG Section 2.1); when only the one-copy margin is recorded the strict reading cannot be applied and the unit is
PENDING unless a one-copy margin already reaches 1e-4 (then it fails under either reading).

Run directories (docs/final_phase_design.md 4.4 and 4.5a): R(g, m, f, s) = runs/final/<g>/<m>/<f>/s<s>/ holding
metrics.json (complete, total_steps, steps_done, lr, params), meta.json (git.git_hash), eval/step<S>.json
(passes.<key>: top1, n_query, n_bank, flip_consistency, n_disagree, max_vote_margin_rel, n_disagree_not_near_tie,
predictions; original_reference), eval/step<S>_meta.json and knn/step<S>_<key>.npz (pred, pred_flip, labels). Folder
names are mirror.final.heal.students.student_folder(method, variant). A healed value is passes.bf16.top1 at the final
step S = total_steps, which must be 780 (the MUST-09 recipe), on all 50,000 validation queries. M = dinov2_vitb14_reg.
"The main Mirror arm" is main/M/mirror_exact_wspace when results/A01/decision.json says "adopted": true and
main/M/mirror_exact_astar otherwise; when that file is missing the A01 rule is recomputed here; "its 3-seed mean" is
the mean of its seeds 0, 1 and 2 (PREREG Section 2.5).

Inputs per rule:
  Gate P  R(main, M, mirror_exact_astar, 0..2) and R(main, M, svdllm, 0..2): final bf16 k-NN.
  A01     zero-shot: eval/step0.json passes.bf16 of R(main, M, mirror_exact_wspace, 0) (seed 1 if seed 0 is missing);
          healed: final bf16 k-NN of mirror_exact_wspace and mirror_exact_astar, seeds 0 and 1. results/A01/
          metrics.json (d+, d-, the fit's own zero-shot preview) and the A*_w structure file are reported.
  A02-1..4 results/A02/metrics.json: fits[] (name, group, d_minus) and pairs[] (a, b, weighted_agreement, twin_mean,
          head_share_same) over the ImageNet fits fit0 to fit3 (the recorded gates are compared, not used).
  A02-5,6 the main arm's seed 0 and R(main, M, mirror_exact_<fit1|fit2|coco>, 0) (the adopted operator type; the
          structure each run used, from its metrics.json or spec.json, is listed beside its value).
  A03-1   results/A03/metrics.json: summaries.<C>."1-6".twin_fraction_t50 and split_half.<C>.ratio_to_null for C1 to
          C3 (controls.csv as a fallback); the in-sample ratio is reported beside it.
  A03-2   R(main, M, pair_random_h1, 0) and the main arm.
  A04-1,2 R(main, M, flap, 0), R(main, M, blockdrop, 0), R(main, M, original, 0) and the main arm.
  A05-1   the main arm (seeds 0 to 2), the A02 heals from fit sets 1 and 2, and R(main, M, <baseline>, seeds) for
          svdllm, svdllm_full, iosvd (0 to 2), slicegpt (0, 1), svd (0); knn/step780_bf16.npz of each for the
          bootstrap (hbootstrap.hierarchical_bootstrap, 10,000 resamples, seed 0, 95% percentile, images paired).
  A06-1..3 R(a06, M, svdllm_f065|f050|f040, 0 (and 1 at f050)), R(a06, M, mirror_svdllm_<same>, ...),
          R(a06, M, mirror_partial_k9, 0); A06-2 also R(main, M, svdllm, 0..2), R(main, M, original, 0) and
          mirror.final.baselines.sizes (predicted_gain, gain_within).
  A07     results/A12/M/systems_table.csv (throughput_b256) with results/A12/M/meta.json "models" (each entry's
          run_dir, model or base: Mirror's entries are those of the main arm, the candidates the seed-0 A06 SVD-LLM
          runs, kind "original" entries with model dinov2_vits14_reg and M, and any flip_average entry on them,
          else twice the single pass); bf16_flipavg passes of R(a06, M, svdllm_<f065|f050|f040>, 0) and of the
          released models (runs/final/a07/<model>/eval/original.json, "model"; the original falls back to
          results/MUST-10/flipavg.csv).
  A08-1   R(a08, deit3_b16, mirror_exact, 0..2) and R(a08, deit3_b16, svdllm, 0..2): eval/top1_step780.json of
          scripts/fp_a08_top1.py ("bf16": top-1 of the model's own classifier on 50,000 images; "fp32": flip
          consistency of the predicted class, n_disagree, disagree_max_margin); the released model's
          runs/final/a08/deit3_b16/_original/eval/top1_original.json is reported; parameters: results/A08/deit3_b16/
          params_check.json (max_abs_diff_pp, or params_stored and params_formula) or law_per_layer.csv (MUST-01
          format; no producer yet).
  A08-2   R(a08, dinov2_vitl14_reg, mirror_exact, 0) and R(a08, dinov2_vitl14_reg, svdllm, 0): bf16 and fp32.
  A09-1..3 results/A09/<model>/<variant>/{components.csv, metrics.json}: M's healed main-arm audit
          (<folder>_s<seed>) is required; every other audited variant present (zero-shot, DeiT-III) is a unit too.
  A10     fp_fullbank summary.json files (rows with fullbank_knn_top1, linear_val_top1) under runs/final/A10,
          results/A10, runs/final/A11 or results/A11, matched to arms by run_dir; the original is the row with no
          run_dir and model M.
  A11     R(a11, M, mirror_exact_ext, 0) and R(a11, M, svdllm_ext, 0) with their full-bank rows.
  A12b    results/A12b/metrics.json (speed_vs_original_b64 and _b256) or a fused entry of results/A12/M/
          systems_table.csv (no producer yet).
  A15a    results/A15a/metrics.json (comparison per size).
  A15b    results/A15b/metrics.json (original.ratio_to_null, finetuned_numbers.ratio_to_null).
  P11.2   runs/final/A10/p112/t2_ade20k.csv (scripts/tier2_ade20k.py --out runs/final/A10/p112; a copy under
          results/ is accepted) row vexact_s0 of M with the mask_eq_sym_* margin columns, else the same fields of
          seg/M/vexact_s0_maskeq.json beside it; the original's row from the same file or results/MUST-11/t2_seg/.
  NOT RUN logs/final/state/{status.json, cuts.json, runner.json, <id>.skip}, the manifest it names (default
          jobs/final_phase.yaml) and the heal job specs (--spec) of its jobs, mapped to run directories.
  Deviations notes/decisions.md against its text at the pre-registration commit (results/final/prereg_record.json).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mirror.final.stats import hbootstrap as HB

REPO_ROOT = Path(__file__).resolve().parents[2]

PASS, FAIL, PENDING, NOT_RUN, REPORTED = "PASS", "FAIL", "PENDING", "NOT RUN", "REPORTED"
PREREG_FILE = "PREREG_FINAL_PHASE.md"
PREREG_RECORD = "results/final/prereg_record.json"
DECISIONS_FILE = "notes/decisions.md"
MAIN, DEIT, LARGE, SMALL = "dinov2_vitb14_reg", "deit3_b16", "dinov2_vitl14_reg", "dinov2_vits14_reg"
N_VAL = 50_000
HEAL_STEPS = 780
SEEDS3 = (0, 1, 2)
EPS = 1e-9                                 # values within EPS of a bar count as equal to the bar
NEAR_TIE = 1e-4                            # PREREG Section 2.1
BOOT_SEED, BOOT_ALPHA, BOOT_RESAMPLES = 0, 0.05, 10_000      # PREREG Section 2.5 (A13)

# Bars (PREREG_FINAL_PHASE.md Section 1; the quoted bar text is checked against the file at run time)
GATE_P_TARGETS = {"mirror_exact_astar": 78.34, "svdllm": 78.06}
GATE_P_TOL = 0.15
A01_ZERO_SHOT_BAR, A01_MARGIN = 58.84, 0.15
A02_DMINUS_TOL, A02_ODD_MIN, A02_TWIN_MIN, A02_HEAD_MIN = 0.15, 0.8, 0.70, 0.90
A02_SPREAD_MAX, A02_COCO_TOL = 0.2, 0.3
A03_TWIN_MAX, A03_RATIO_MIN, A03_H1_GAP = 0.05, 0.9, 3.0
A04_GAP = 0.5
A05_LOWER = -0.3
A06_GAIN, A06_K, A06_REL = 0.5, 2, 0.5
A06_BUDGETS = ((0.65, "f065"), (0.50, "f050"), (0.40, "f040"))
A07_MARGIN = 0.5
A08_GAP = 0.5
A09_1_TOL, A09_2_FACTOR = 1e-10, 2.0
A10_TOL = 1.0
A11_TARGET = 0.2
A12B_SPEED = 1.1
A15B_DROP = 0.1

STRUCTURES = {"wspace": f"results/A01/structure/{MAIN}_wspace_matched.npz",
              "astar": f"results/MUST-07/structure/{MAIN}_s0.npz"}
ITEMS = {"Gate P": ("Gate P", "decision", ("78.34", "78.06", "0.15")),
         "A01": ("A01", "decision", ("58.84", "0.15")),
         "A02-1": ("A02", "scored", ("15%",)), "A02-2": ("A02", "scored", ("0.8",)),
         "A02-3": ("A02", "scored", ("70%",)), "A02-4": ("A02", "scored", ("90%",)),
         "A02-5": ("A02", "scored", ("0.2",)), "A02-6": ("A02", "scored", ("0.3",)),
         "A03-1": ("A03", "scored", ("0.05", "0.9")), "A03-2": ("A03", "scored", ("3.0",)),
         "A04-1": ("A04", "scored", ("0.5",)), "A04-2": ("A04", "report", ()),
         "A05-1": ("A05", "scored", ("0.3",)), "A06-1": ("A06", "scored", ("0.5", "2 of 3", "0.50")),
         "A06-2": ("A06", "scored", ("50%", "2 of 3")), "A06-3": ("A06", "report", ()),
         "A07": ("A07", "scored", ("0.5",)), "A08-1": ("A08", "scored", ("0.5", "100%")),
         "A08-2": ("A08", "scored", ("0.5", "100%")), "A09-1": ("A09", "scored", ("1e-10",)),
         "A09-2": ("A09", "scored", ("2x", "100%")), "A09-3": ("A09", "report", ()),
         "A10": ("A10", "scored", ("1.0",)), "A11": ("A11", "report", ("0.2",)),
         "A12b": ("A12b", "scored", ("1.1x", "64", "256")), "A15a": ("A15a", "report", ()),
         "A15b": ("A15b", "scored", ("0.1",))}
P11_2_TEXT = ("P11.2 (MUST-11, previous phase, re-measured in A10): Mirror-exact with the symmetrised head gives 100% "
              "mask equivariance; the original model stays below 100%")
P11_2_BAR = ("Mirror-exact 100% (every differing pixel a near-tie in both copies, PREREG_NEXT_PHASE.md Section 1); "
             "original < 100%")
NOT_RUN_STATES = {"cut": "cut by the Section 9 cut order", "budget": "not started: budget governor (13.5 VM-hours)",
                  "terminated": "stopped at the 15 VM-hour ceiling", "skipped": "skipped by hand (.skip file)"}
FAILED_STATES = ("failed", "blocked", "gave_up")


def log(msg: str) -> None:
    print(f"[collect_final {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================
# Small helpers
# ================================================================================================
def fnum(v):
    """float, or None for missing, boolean, non-numeric or non-finite values."""
    if v is None or isinstance(v, (bool, np.bool_)):
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
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
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


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def git(root: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_self(root: Path = REPO_ROOT) -> dict:
    h = git(root, "rev-parse", "HEAD")
    return {"git_hash": h, "git_dirty": bool(git(root, "status", "--porcelain", "--untracked-files=no")) if h else None}


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def append_decision(path, decision: str, reason: str, effect: str,
                    by: str = "Runner job (pre-registered rule)") -> bool:
    """Append one row to the decisions table (notes/decisions.md: date, decision, reason, effect, approved by) unless
    a row with the same decision text is already there. Returns True when a row was added."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    cells = [str(x).replace("|", "/").replace("\n", " ") for x in (decision, reason, effect, by)]
    if f"| {cells[0]} |" in text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write(f"| {time.strftime('%Y-%m-%d', time.gmtime())} | " + " | ".join(cells) + " |\n")
    return True


def folder_name(method: str, variant=None) -> str:
    """mirror.final.heal.students.student_folder: <method>[_<variant>] (characters outside [A-Za-z0-9.-] -> '-')."""
    return method if not variant else f"{method}_{re.sub(r'[^A-Za-z0-9.-]+', '-', str(variant))}"


def cmp(value, op: str, bar: float) -> str:
    """PASS or FAIL of `value op bar` (values within EPS of the bar count as equal to it); PENDING without a value."""
    v = fnum(value)
    if v is None:
        return PENDING
    ok = {">=": v >= bar - EPS, ">": v > bar + EPS, "<=": v <= bar + EPS, "<": v < bar - EPS}[op]
    return PASS if ok else FAIL


def within(a, b, tol: float) -> str:
    a, b = fnum(a), fnum(b)
    if a is None or b is None:
        return PENDING
    return PASS if abs(a - b) <= tol + EPS else FAIL


def combine(verdicts) -> str:
    """Section 3: FAIL if any unit fails; NOT RUN if a needed unit was cut; PENDING if one is missing; PASS when every
    unit is present and passes (no units: PENDING)."""
    vs = list(verdicts)
    if any(v == FAIL for v in vs):
        return FAIL
    if any(v == NOT_RUN for v in vs):
        return NOT_RUN
    if not vs or any(v != PASS for v in vs):
        return PENDING
    return PASS


def k_of_n(verdicts, k: int) -> str:
    """'At least k of n units pass': PASS once k pass; FAIL once k can no longer be reached; PENDING while pending
    units could still reach k; NOT RUN when only cut units could."""
    vs = list(verdicts)
    n_pass = sum(v == PASS for v in vs)
    n_pend = sum(v == PENDING for v in vs)
    n_cut = sum(v == NOT_RUN for v in vs)
    if n_pass >= k:
        return PASS
    if n_pass + n_pend + n_cut < k:
        return FAIL
    if n_pass + n_pend >= k:
        return PENDING
    return NOT_RUN


def mean_std(vals):
    v = [fnum(x) for x in vals]
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def truthy(x) -> bool | None:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


# ================================================================================================
# Values and arms
# ================================================================================================
@dataclass
class Val:
    """One scored quantity: a value with its sources, or the state of the missing input ('pending' or 'not_run')."""

    value: float | None = None
    state: str = "pending"            # ok, pending, not_run
    reason: str = ""
    sources: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == "ok" and self.value is not None

    @property
    def missing_verdict(self) -> str:
        return NOT_RUN if self.state == "not_run" else PENDING

    def verdict(self, op: str, bar: float) -> str:
        return cmp(self.value, op, bar) if self.ok else self.missing_verdict


def mean_val(vals, what: str = "") -> Val:
    """Mean of seed-level values (PREREG Section 2.1); the missing state of the first missing seed otherwise."""
    vals = list(vals)
    srcs = [s for v in vals for s in v.sources]
    if vals and all(v.ok for v in vals):
        xs = [v.value for v in vals]
        return Val(float(np.mean(xs)), "ok", sources=srcs, info={"values": xs, "n": len(xs)})
    bad = [v for v in vals if not v.ok]
    state = "not_run" if any(v.state == "not_run" for v in bad) else "pending"
    return Val(None, state, reason=(f"{what}: " if what else "") + "; ".join(v.reason for v in bad if v.reason),
               sources=srcs)


def diff_val(a: Val, b: Val, what: str = "") -> Val:
    if a.ok and b.ok:
        return Val(a.value - b.value, "ok", sources=a.sources + b.sources)
    bad = [v for v in (a, b) if not v.ok]
    state = "not_run" if any(v.state == "not_run" for v in bad) else "pending"
    return Val(None, state, reason=(f"{what}: " if what else "") + "; ".join(v.reason for v in bad if v.reason),
               sources=a.sources + b.sources)


@dataclass(frozen=True)
class Arm:
    """A healed arm: runs/final/<group>/<model>/<folder>/s<seed>/ (folders: accepted names, the first preferred)."""

    group: str
    model: str
    folders: tuple
    label: str = ""

    def path(self, seed, folder: str | None = None) -> str:
        return f"runs/final/{self.group}/{self.model}/{folder or self.folders[0]}/s{int(seed)}"

    @property
    def name(self) -> str:
        return self.label or self.folders[0]


def arm(group: str, model: str, *folders, label: str = "") -> Arm:
    return Arm(group, model, tuple(folders), label)


# Arms whose folder does not depend on the A01 decision (docs/final_phase_design.md 4.5a: the operator they use is
# the adopted one, recorded in each run's spec.json)
H1_ARM = Arm("main", MAIN, ("pair_random_h1",), "H1: main operator, random pairing")
A11_MIRROR = Arm("a11", MAIN, ("mirror_exact_ext",), "Mirror, extended healing")
A11_SVDLLM = Arm("a11", MAIN, ("svdllm_ext",), "SVD-LLM, extended healing")


# ================================================================================================
# Records
# ================================================================================================
@dataclass
class Rule:
    id: str
    experiment: str
    kind: str                         # decision, scored, report
    title: str
    bar: str
    quantity: str
    verdict: str = PENDING
    value: object = None
    n: object = None
    reason: str = ""
    details: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    quote_found: bool | None = None
    bar_tokens_missing: list = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.kind != "report"


def parse_prereg_rows(text: str) -> dict:
    """{ID: (rule text, PASS bar)} from the table of PREREG_FINAL_PHASE.md Section 1."""
    rows, inside = {}, False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.startswith("## 1.")
            continue
        if not inside or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("ID", "") or set(cells[0]) <= {"-", ":"}:
            continue
        rows[cells[0]] = (cells[1], cells[2])
    return rows


def prereg_info(root: Path) -> dict:
    """The pre-registration file (sha256), its record (results/final/prereg_record.json: sha256 and commit) and
    whether the file is unchanged since the record; read from `root`, else from the repository."""
    out = {"file": PREREG_FILE, "read_from": None, "sha256": None, "recorded_sha256": None, "commit": None,
           "recorded_utc": None, "unchanged": None, "status": "", "rows": {}, "text_norm": ""}
    path, base = root / PREREG_FILE, root
    if not path.is_file():
        path, base = REPO_ROOT / PREREG_FILE, REPO_ROOT
    if not path.is_file():
        out["status"] = f"{PREREG_FILE} not found"
        return out
    text = path.read_text(encoding="utf-8")
    out.update(read_from=path.as_posix(), sha256=sha256_file(path), rows=parse_prereg_rows(text),
               text_norm=norm_ws(text))
    rec = read_json(base / PREREG_RECORD)
    if isinstance(rec, dict):
        out.update(recorded_sha256=rec.get("sha256"), commit=rec.get("commit"), recorded_utc=rec.get("recorded_utc"))
        out["unchanged"] = rec.get("sha256") == out["sha256"]
        out["status"] = ("unchanged since the recorded pre-registration commit" if out["unchanged"] else
                         "CHANGED since the recorded pre-registration commit (sha256 differs)")
    else:
        out["commit"] = git(base, "log", "-1", "--format=%H", "--", PREREG_FILE)
        out["status"] = (f"no record ({PREREG_RECORD}); last commit of the file "
                         f"{short(out['commit'])}" if out["commit"] else f"not committed and not recorded")
    return out


# ================================================================================================
# Runner state: which jobs were cut (NOT RUN) and which run directories they would have written
# ================================================================================================
def _load_any(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except ValueError:
            return None
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return None


class CutIndex:
    """Job states of the runner (logs/final/state/status.json, cuts.json, <id>.skip) and the outputs of every job of
    the manifest: heal jobs map to run directories through their --spec file (group, model, seed, students) or their
    done marker; other jobs to result directories (their done marker's folder and --out)."""

    def __init__(self):
        self.jobs: dict = {}
        self.manifest: str | None = None
        self.state_dir: str | None = None

    @classmethod
    def load(cls, c: "Final") -> "CutIndex":
        ci = cls()
        state_dir = c.p("logs", "final", "state")
        runner = c.jload(state_dir / "runner.json") or {}
        mp = Path(runner["manifest"]) if runner.get("manifest") else c.p("jobs", "final_phase.yaml")
        if not mp.is_absolute():
            mp = c.root / mp
        man = _load_any(mp) if mp.is_file() else None
        if isinstance(man, dict):
            ci.manifest = c.rel(mp)
            c.seen(mp)
            sd = (man.get("settings") or {}).get("state_dir")
            if sd:
                state_dir = c.root / sd
        ci.state_dir = c.rel(state_dir)
        status = c.jload(state_dir / "status.json") or {}
        cuts = c.jload(state_dir / "cuts.json") or {}
        states = {str(k): str(v) for k, v in (status.get("states") or {}).items()}
        reasons = {str(k): str(v) for k, v in (status.get("reasons") or {}).items()}
        rank = {}
        for e in cuts.get("cuts") or []:
            for j in e.get("jobs") or []:
                rank[str(j)] = e.get("rank")
        for j in cuts.get("jobs") or []:
            if states.get(str(j)) not in ("finished", "done", "running"):
                states[str(j)] = "cut"
        if state_dir.is_dir():
            for p in state_dir.glob("*.skip"):
                if states.get(p.stem) not in ("finished", "done", "running"):
                    states[p.stem] = "skipped"
        jobs = (man or {}).get("jobs") if isinstance(man, dict) else None
        listed = {}
        for j in jobs or []:
            if isinstance(j, dict) and j.get("id"):
                listed[str(j["id"])] = j
        for jid in set(states) | set(listed):
            arms, prefixes = cls._outputs(c, listed.get(jid) or {})
            st = states.get(jid)
            why = NOT_RUN_STATES.get(st, reasons.get(jid, ""))
            if st == "cut" and rank.get(jid) is not None:
                why = f"cut by the Section 9 cut order (rank {rank[jid]})"
            ci.jobs[jid] = {"state": st, "reason": why, "arms": arms, "prefixes": prefixes}
        return ci

    @staticmethod
    def _outputs(c: "Final", job: dict) -> tuple:
        arms, prefixes = set(), []
        done = str(job.get("done") or "").replace("\\", "/").lstrip("./")
        m = re.match(r"^runs/final/([^/]+)/([^/]+)/([^/]+)/s(\d+)/DONE$", done)
        if m and m.group(3) != "_jobs":
            arms.add((m.group(1), m.group(2), m.group(3), int(m.group(4))))
        elif done.startswith("results/") or (done.startswith("runs/final/") and "/_jobs/" not in done):
            prefixes.append(done.rsplit("/", 1)[0])
        try:
            toks = shlex.split(str(job.get("command") or ""))
        except ValueError:
            toks = str(job.get("command") or "").split()
        for i, t in enumerate(toks[:-1]):
            val = toks[i + 1].replace("\\", "/").rstrip("/")
            if t == "--out" and (val.startswith("results/") or val.startswith("runs/final/")):
                prefixes.append(val)
            if t == "--spec":
                sp = Path(toks[i + 1])
                spec = _load_any(sp if sp.is_absolute() else c.root / sp)
                if isinstance(spec, dict):
                    g = str(spec.get("group") or spec.get("experiment") or "")
                    mdl = str(spec.get("model") or MAIN)
                    seed = int(spec.get("seed", 0) or 0)
                    for s in spec.get("students") or []:
                        if isinstance(s, dict) and s.get("method"):
                            arms.add((g, mdl, folder_name(s["method"], s.get("variant")), seed))
        # a folder must name an experiment (results/<ID>/..., runs/final/<group>/...): a top-level file such as the
        # report's results/FINAL_PHASE_REPORT.md says nothing about the result folders below it
        prefixes = [p for p in dict.fromkeys(prefixes)
                    if len(p.split("/")) >= (3 if p.startswith("runs/") else 2) and p != "results/final"]
        return arms, prefixes

    def arm_state(self, a: Arm, seed) -> tuple | None:
        """(state, job id, reason) of the job that writes this arm's seed, or None when no job is known."""
        hit = None
        for jid, j in sorted(self.jobs.items()):
            if any((a.group, a.model, f, int(seed)) in j["arms"] for f in a.folders):
                if hit is None or (j["state"] in NOT_RUN_STATES and hit[0] not in NOT_RUN_STATES):
                    hit = (j["state"], jid, j["reason"])
        return hit

    def prefix_state(self, rel_dir: str) -> tuple | None:
        """(state, job id, reason) of a job that was not run and writes into (or above) the result folder rel_dir."""
        q = rel_dir.rstrip("/")
        for jid, j in sorted(self.jobs.items()):
            if j["state"] in NOT_RUN_STATES and any(p == q or p.startswith(q + "/") or q.startswith(p + "/")
                                                    for p in j["prefixes"]):
                return j["state"], jid, j["reason"]
        return None

    def not_run_jobs(self) -> list:
        return sorted((jid, j["state"], j["reason"]) for jid, j in self.jobs.items() if j["state"] in NOT_RUN_STATES)


# ================================================================================================
# The collector's context
# ================================================================================================
class Final:
    def __init__(self, root=REPO_ROOT, *, n_val: int = N_VAL, n_resamples: int = BOOT_RESAMPLES,
                 t1_intervals: bool = True):
        self.root = Path(root).resolve()
        self.n_val = int(n_val)
        self.n_resamples = int(n_resamples)
        self.t1_intervals = bool(t1_intervals)
        self.inputs: dict = {}
        self.numbers: dict = {}
        self.rules: list = []
        self.notes: list = []
        self.tables: dict = {}
        self.boot: dict = {}
        self._j: dict = {}
        self._csv: dict = {}
        self._main = None
        self._a01 = None
        self.prereg = prereg_info(self.root)
        self.cuts = CutIndex.load(self)

    # ---------------------------------------------------------------- paths and reading
    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def rel(self, p) -> str:
        p = Path(p)
        for q in (p, p.resolve() if p.exists() else p):
            try:
                return q.relative_to(self.root).as_posix()
            except ValueError:
                continue
        return p.as_posix()

    def seen(self, p) -> None:
        r = self.rel(p)
        if r not in self.inputs:
            try:
                self.inputs[r] = sha256_file(p)
            except OSError:
                self.inputs[r] = "unreadable"

    def jload(self, p):
        p = Path(p)
        if p in self._j:
            return self._j[p]
        obj = None
        if p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError, UnicodeDecodeError) as e:
                self.notes.append(f"cannot read {self.rel(p)}: {e}")
            else:
                self.seen(p)
        self._j[p] = obj
        return obj

    def cload(self, p):
        p = Path(p)
        if p in self._csv:
            return self._csv[p]
        df = None
        if p.is_file() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"cannot read {self.rel(p)}: {e}")
            else:
                self.seen(p)
                df = df if len(df) else None
        self._csv[p] = df
        return df

    def meta_git(self, p) -> str | None:
        """Commit of the run that wrote `p`: <stem>_meta.json beside it (evaluation files), else meta.json in its
        folder or a parent folder (up to results/ or runs/), else a commit recorded inside the file."""
        p = Path(p)
        cands = []
        if p.suffix == ".json":
            cands = [p.with_name(f"{p.stem}_meta.json"), p.with_name(f"meta_{p.stem}.json")]
            if p.stem.startswith("top1_"):        # scripts/fp_a08_top1.py: eval/meta_top1_step<S>.json, <out>/meta_top1.json
                cands += [p.with_name("meta_top1.json"), p.parent.parent / "meta_top1.json"]
        d = p if p.is_dir() else p.parent
        stop = {self.p("results"), self.p("runs"), self.root}
        while True:
            cands.append(d / "meta.json")
            if d in stop or d.parent == d:
                break
            d = d.parent
        for m in cands:
            if m == p or not m.is_file():
                continue
            j = self.jload(m) or {}
            self._prereg_check(m, j)
            g = (j.get("git") or {}).get("git_hash") if isinstance(j.get("git"), dict) else j.get("git_hash")
            if g:
                return g
        if p.is_file() and p.suffix == ".json":
            j = self.jload(p) or {}
            g = j.get("git")
            if isinstance(g, dict) and g.get("git_hash"):
                return g["git_hash"]
            if isinstance(g, str) and g:
                return g
            if j.get("git_hash"):
                return j["git_hash"]
        return None

    def _prereg_check(self, meta_path: Path, j: dict) -> None:
        pr = j.get("prereg") if isinstance(j, dict) else None
        sha = pr.get("sha256") if isinstance(pr, dict) else None
        if sha and self.prereg["sha256"] and sha != self.prereg["sha256"]:
            note = (f"{self.rel(meta_path)} records a {PREREG_FILE} text with sha256 {sha[:12]}, not the current "
                    f"{self.prereg['sha256'][:12]}")
            if note not in self.notes:
                self.notes.append(note)

    def src(self, p, git_hash=None) -> dict:
        return {"file": self.rel(p), "git_hash": git_hash if git_hash is not None else self.meta_git(p)}

    def num(self, key: str, value, *, desc: str, sources, n=None, unit: str = "", seed=None) -> None:
        if isinstance(value, Val):
            if not value.ok:
                return
            value = value.value
        if isinstance(value, (bool, np.bool_, str)):
            v = value if not isinstance(value, np.bool_) else bool(value)
        elif isinstance(value, (int, np.integer)):
            v = int(value)
        else:
            v = fnum(value)
            if v is None:
                return
        srcs = sources if isinstance(sources, list) else [sources]
        srcs = dedupe_sources(srcs)
        self.numbers[key] = {"value": jsonable(v), "unit": unit, "n": jsonable(n), "description": desc,
                             "seed": jsonable(seed), "sources": srcs,
                             "source": srcs[0]["file"] if srcs else None,
                             "commit": srcs[0]["git_hash"] if srcs else None}

    def add(self, rid: str, quantity: str, *, title: str | None = None, bar: str | None = None) -> Rule:
        exp, kind, tokens = ITEMS.get(rid, (rid, "scored", ()))
        row = self.prereg["rows"].get(rid)
        if rid == "P11.2":
            exp, kind = "MUST-11 (A10)", "scored"
            t, b, found = P11_2_TEXT, P11_2_BAR, None
        elif row:
            t, b, found = row[0], row[1], True
        else:
            t, b, found = title or rid, bar or "", (False if self.prereg["rows"] else None)
        r = Rule(rid, exp, kind, t, b, quantity, quote_found=found)
        if row:
            r.bar_tokens_missing = [x for x in tokens if x not in b and x.replace("-", "\u2212") not in b]
        self.rules.append(r)
        return r

    # ---------------------------------------------------------------- runs
    def find(self, a: Arm, seed) -> tuple:
        """(run directory, folder) of the first accepted folder that exists for this seed, else (None, None)."""
        for f in a.folders:
            d = self.p("runs", "final", a.group, a.model, f, f"s{int(seed)}")
            if d.is_dir():
                return d, f
        return None, None

    def missing(self, a: Arm, seed) -> Val:
        st = self.cuts.arm_state(a, seed)
        where = a.path(seed)
        if st and st[0] in NOT_RUN_STATES:
            return Val(state="not_run", reason=f"{where}: {st[2]} (job {st[1]})")
        if st and st[0] in FAILED_STATES:
            return Val(state="pending", reason=f"{where}: job {st[1]} {st[0]} (see logs/final/{st[1]}.log)")
        return Val(state="pending", reason=f"{where} missing (not run or not finished)")

    def run_metrics(self, a: Arm, seed, steps: int | None = HEAL_STEPS) -> tuple:
        """(run directory, metrics.json, final step, Val with the problem or None)."""
        d, folder = self.find(a, seed)
        if d is None:
            return None, None, None, self.missing(a, seed)
        m = self.jload(d / "metrics.json")
        where = self.rel(d)
        if not isinstance(m, dict):
            return d, None, None, Val(state="pending", reason=f"{where}: not finished (no metrics.json)")
        if m.get("diverged"):
            return d, m, None, Val(state="pending", reason=f"{where}: diverged after step {m.get('steps_done')}")
        if not m.get("complete"):
            return d, m, None, Val(state="pending", reason=f"{where}: not finished (metrics.json not complete)")
        S = m.get("total_steps", m.get("steps_done"))
        if S is None:
            return d, m, None, Val(state="pending", reason=f"{where}: metrics.json records no step count")
        S = int(S)
        if int(m.get("steps_done", S)) != S:
            return d, m, None, Val(state="pending", reason=f"{where}: {m.get('steps_done')} of {S} steps done")
        if steps is not None and S != int(steps):
            return d, m, S, Val(state="pending", reason=f"{where}: healed for {S} steps, not the recipe's {steps}")
        return d, m, S, None

    def eval_at(self, d: Path, step: int, key: str = "bf16", need_full: bool = True,
                extra_sources: list | None = None) -> Val:
        p = d / "eval" / f"step{int(step)}.json"
        rec = self.jload(p)
        if not isinstance(rec, dict):
            return Val(state="pending", reason=f"{self.rel(p)} missing (step {step} not evaluated yet)")
        pr = (rec.get("passes") or {}).get(key)
        if not isinstance(pr, dict):
            return Val(state="pending", reason=f"{self.rel(p)}: no '{key}' pass yet")
        nq = pr.get("n_query")
        if need_full and (nq is None or int(nq) != self.n_val):
            return Val(state="pending", reason=f"{self.rel(p)} [{key}]: {nq} queries, not all {self.n_val:,}")
        v = fnum(pr.get("top1"))
        if v is None:
            return Val(state="pending", reason=f"{self.rel(p)} [{key}]: no top-1")
        g = self.meta_git(p) or pr.get("git_hash")
        srcs = list(extra_sources or []) + [{"file": self.rel(p), "git_hash": g}]
        return Val(v, "ok", sources=srcs, info={"record": pr, "eval": rec, "eval_file": self.rel(p),
                                                 "run_dir": d, "step": int(step), "key": key, "n_query": nq})

    def final_eval(self, a: Arm, seed, key: str = "bf16", steps: int | None = HEAL_STEPS,
                   need_full: bool = True) -> Val:
        """The pass `key` at the final step of a finished run (the healed value for key bf16)."""
        d, m, S, bad = self.run_metrics(a, seed, steps)
        if bad is not None:
            return bad
        v = self.eval_at(d, S, key, need_full, [self.src(d / "metrics.json")])
        v.info.update(metrics=m, arm=a, seed=int(seed))
        return v

    def step_eval(self, a: Arm, seed, step: int, key: str = "bf16") -> Val:
        """The pass `key` at checkpoint `step` (no completeness check: step 0 exists once training started)."""
        d, _ = self.find(a, seed)
        if d is None:
            return self.missing(a, seed)
        v = self.eval_at(d, step, key, True, [self.src(d / "meta.json")] if (d / "meta.json").is_file() else [])
        v.info.update(arm=a, seed=int(seed))
        return v

    def correct(self, v: Val) -> tuple:
        """(per-image correctness, labels) from the prediction file of an evaluated pass (MUST-09 format)."""
        rec, d = v.info.get("record") or {}, v.info.get("run_dir")
        rel = rec.get("predictions") or f"knn/step{v.info.get('step')}_{v.info.get('key', 'bf16')}.npz"
        p = Path(d) / rel
        if not p.is_file():
            raise FileNotFoundError(f"{self.rel(p)} missing (per-image predictions)")
        correct, labels = HB.load_knn_correct(p, "pred")
        self.seen(p)
        return correct, labels

    # ---------------------------------------------------------------- the A01 decision and the main arm
    def a01(self) -> dict:
        if self._a01 is None:
            self._a01 = a01_record(self)
        return self._a01

    def main_variant(self) -> tuple:
        """('wspace' | 'astar' | None, how it was decided)."""
        if self._main is not None:
            return self._main
        rec = self.a01()
        p = self.p("results", "A01", "decision.json")
        d = self.jload(p)
        if isinstance(d, dict) and isinstance(d.get("adopted"), bool):
            v, how = ("wspace" if d["adopted"] else "astar"), f"{self.rel(p)} (\"adopted\": {str(d['adopted']).lower()})"
            if rec["adopted"] is not None and rec["adopted"] != d["adopted"]:
                self.notes.append(f"WARNING: {self.rel(p)} says adopted {d['adopted']}, but the A01 rule recomputed "
                                  f"from the run files gives {rec['adopted']} ({rec['status']}); the file is used")
        elif rec["adopted"] is not None:
            v = "wspace" if rec["adopted"] else "astar"
            how = "the A01 rule recomputed here (results/A01/decision.json missing)"
        else:
            v, how = None, f"the A01 decision is pending ({rec['reason']})"
        self._main = (v, how)
        return self._main

    def main_arm(self) -> Arm | None:
        v, _ = self.main_variant()
        return None if v is None else arm("main", MAIN, f"mirror_exact_{v}", label=f"Mirror (main arm, {v})")

    def main_mean(self) -> Val:
        a = self.main_arm()
        if a is None:
            return Val(state="pending", reason=self.main_variant()[1])
        mv = mean_val([self.final_eval(a, s) for s in SEEDS3], "main Mirror arm")
        mv.info["arm"] = a
        return mv


def missing_result(c: Final, rel_dirs, what: str) -> tuple:
    """(NOT RUN or PENDING, reason) for a result that is missing: NOT RUN when a job writing into one of the folders
    rel_dirs (a path or a list) was cut, stopped by the budget governor or skipped."""
    for rel_dir in ([rel_dirs] if isinstance(rel_dirs, str) else rel_dirs):
        st = c.cuts.prefix_state(rel_dir)
        if st:
            return NOT_RUN, f"{what} missing: {st[2]} (job {st[1]})"
    return PENDING, f"{what} missing"


def dedupe_sources(srcs) -> list:
    out, seen = [], set()
    for s in srcs or []:
        if not isinstance(s, dict):
            continue
        key = (s.get("file"), s.get("git_hash"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": s.get("file"), "git_hash": s.get("git_hash")})
    return out


def exact_flip(rec: dict | None, n_val: int, *, strict_key: str = "max_vote_margin_rel",
               not_tie_key: str = "n_disagree_not_near_tie", one_copy_keys=(), one_copy_list: str | None = None,
               n_key: str = "n_query") -> tuple:
    """Exact invariance ('100% flip consistency', PREREG Section 2.1): (verdict, note)."""
    if not isinstance(rec, dict):
        return PENDING, "flip check not recorded"
    share, nd = fnum(rec.get("flip_consistency")), rec.get("n_disagree")
    nq = rec.get(n_key)
    if share is None:
        return PENDING, "flip consistency not recorded"
    if nd is None:
        nd = 0 if share >= 100.0 else None
    if nd == 0:
        v, note = PASS, "100% raw (no disagreement)"
    else:
        nnt, strict = rec.get(not_tie_key), fnum(rec.get(strict_key))
        if nnt is not None:
            v = PASS if int(nnt) == 0 else FAIL
            note = (f"{fmt(share, '.4f')}% raw, {nd} disagreement(s), "
                    + ("all near-ties in both copies" if v == PASS else f"{int(nnt)} not near-ties"))
            if strict is not None:
                note += f" (largest relative margin {strict:.2e})"
        elif strict is not None:
            v = PASS if strict < NEAR_TIE else FAIL
            note = f"{fmt(share, '.4f')}% raw, {nd} disagreement(s), largest relative margin (both copies) {strict:.2e}"
        else:
            one = [fnum(rec.get(k)) for k in one_copy_keys]
            if one_copy_list and isinstance(rec.get(one_copy_list), list):
                one += [fnum(x) for x in rec[one_copy_list]]
            one = [x for x in one if x is not None]
            if one and max(one) >= NEAR_TIE:
                v, note = FAIL, (f"{fmt(share, '.4f')}% raw, {nd} disagreement(s); a one-copy margin of {max(one):.2e} "
                                 "is not a near-tie under either reading")
            else:
                v, note = PENDING, (f"{fmt(share, '.4f')}% raw, {nd} disagreement(s); only the one-copy margin is "
                                    "recorded, so the near-tie rule in both copies (PREREG Section 2.1) cannot be "
                                    "applied")
    if v == PASS and nq is not None and int(nq) < n_val:
        v, note = PENDING, note + f"; on {nq} images, not all {n_val:,}"
    return v, note


# ================================================================================================
# Gate P and the A01 rule (shared with scripts/fp_gate_p.py and scripts/fp_a01_decide.py)
# ================================================================================================
GATE_P_ARMS = {"mirror_exact_astar": arm("main", MAIN, "mirror_exact_astar", label="Mirror, current A*"),
               "svdllm": arm("main", MAIN, "svdllm", label="SVD-LLM (whitening only)")}


def gate_p_record(c: Final) -> dict:
    """Gate P (PREREG Section 2.2): {"pass": true | false | null, "status", per-arm values, means and bars,
    "sources"}. A Gate P run that was cut counts as not reproduced (pass false, status NOT RUN)."""
    out = {"gate": "Gate P", "rule": c.prereg["rows"].get("Gate P", ("", ""))[0],
           "bar_text": c.prereg["rows"].get("Gate P", ("", ""))[1],
           "definition": "final-step bf16 k-NN on all 50,000 validation queries; means over healing seeds 0, 1 "
                         "and 2 within +-0.15 of 78.34 (Mirror, current A*) and 78.06 (SVD-LLM); both must hold",
           "arms": {}, "bars": {}, "sources": []}
    verdicts, reasons = [], []
    for name, a in GATE_P_ARMS.items():
        vals = {s: c.final_eval(a, s) for s in SEEDS3}
        mv = mean_val(vals.values(), name)
        target = GATE_P_TARGETS[name]
        v = within(mv.value, target, GATE_P_TOL) if mv.ok else mv.missing_verdict
        verdicts.append(v)
        if not mv.ok:
            reasons.append(mv.reason)
        out["bars"][name] = {"target": target, "tolerance": GATE_P_TOL, "low": round(target - GATE_P_TOL, 6),
                             "high": round(target + GATE_P_TOL, 6)}
        out["arms"][name] = {"run_dirs": [a.path(s) for s in SEEDS3],
                             "seeds": {str(s): (x.value if x.ok else None) for s, x in vals.items()},
                             "missing": {str(s): x.reason for s, x in vals.items() if not x.ok},
                             "mean": mv.value, "n": sum(x.ok for x in vals.values()),
                             "difference_from_target": None if not mv.ok else mv.value - target,
                             "verdict": v, "sources": dedupe_sources(mv.sources)}
        out["sources"] += mv.sources
    status = combine(verdicts)
    out["sources"] = dedupe_sources(out["sources"])
    out["status"] = status
    out["pass"] = True if status == PASS else (False if status in (FAIL, NOT_RUN) else None)
    out["reason"] = ("; ".join(r for r in reasons if r) if status in (PENDING, NOT_RUN) else
                     "; ".join(f"{k}: mean {fmt(v['mean'], '.3f')} against {GATE_P_TARGETS[k]} +- {GATE_P_TOL} "
                               f"({v['verdict']})" for k, v in out["arms"].items()))
    return out


def a01_record(c: Final) -> dict:
    """The A01 rule (PREREG Section 2.2): adopt A*_w when its zero-shot k-NN (step 0 of the A*_w heal) is at least
    58.84 and the mean of its healed seeds 0 and 1 is at least the current A*'s mean over the same seeds plus 0.15.
    "adopted" is true (both hold), false (one fails, or an input was cut) or None (pending)."""
    w = arm("main", MAIN, "mirror_exact_wspace", label="Mirror, A*_w")
    a = arm("main", MAIN, "mirror_exact_astar", label="Mirror, current A*")
    zs, zs_seed = c.step_eval(w, 0, 0), 0
    if not zs.ok:
        alt = c.step_eval(w, 1, 0)
        if alt.ok:
            zs, zs_seed = alt, 1
    hw = {s: c.final_eval(w, s) for s in (0, 1)}
    ha = {s: c.final_eval(a, s) for s in (0, 1)}
    mw, ma = mean_val(hw.values(), "A*_w seeds 0 and 1"), mean_val(ha.values(), "current A* seeds 0 and 1")
    dv = diff_val(mw, ma)
    v_zs = zs.verdict(">=", A01_ZERO_SHOT_BAR)
    v_h = dv.verdict(">=", A01_MARGIN)
    status = combine([v_zs, v_h])
    adopted = True if status == PASS else (False if status in (FAIL, NOT_RUN) else None)
    variant = None if adopted is None else ("wspace" if adopted else "astar")
    sfile = None if variant is None else STRUCTURES[variant]
    sp = c.p(sfile) if sfile else None
    reasons = [x.reason for x in (zs, mw, ma) if not x.ok and x.reason]
    rec = {"rule": "A01", "status": status, "adopted": adopted, "decided": adopted is not None,
           "main_variant": variant, "main_arm": None if variant is None else f"mirror_exact_{variant}",
           "main_arm_dir": None if variant is None else f"runs/final/main/{MAIN}/mirror_exact_{variant}",
           "operator": None if variant is None else ("A*_w (weight-space fit, matched mode)" if adopted else
                                                     "A* (activation fit, the current operator)"),
           "structure": sfile,
           "structure_exists": None if sp is None else sp.is_file(),
           "structure_sha256": sha256_file(sp) if (sp is not None and sp.is_file()) else None,
           "structures": dict(STRUCTURES),
           "zero_shot": {"value": zs.value, "bar": A01_ZERO_SHOT_BAR, "verdict": v_zs, "seed": zs_seed,
                         "step": 0, "run_dir": w.path(zs_seed), "reason": zs.reason if not zs.ok else ""},
           "healed": {"wspace": {"seeds": {str(s): x.value if x.ok else None for s, x in hw.items()},
                                 "mean": mw.value},
                      "astar": {"seeds": {str(s): x.value if x.ok else None for s, x in ha.items()},
                                "mean": ma.value},
                      "per_seed_difference": {str(s): (hw[s].value - ha[s].value) if (hw[s].ok and ha[s].ok)
                                              else None for s in (0, 1)},
                      "difference": dv.value, "margin": A01_MARGIN, "verdict": v_h},
           "bars": {"zero_shot_min": A01_ZERO_SHOT_BAR, "healed_margin": A01_MARGIN,
                    "text": c.prereg["rows"].get("A01", ("", ""))[1]},
           "reason": "; ".join(reasons) if status in (PENDING, NOT_RUN) else
           (f"zero-shot {fmt(zs.value, '.2f')} ({v_zs}); healed A*_w minus A* {fmt(dv.value, '+.3f')} ({v_h})"),
           "sources": dedupe_sources(zs.sources + mw.sources + ma.sources)}
    return rec


# ================================================================================================
# Rule scorers
# ================================================================================================
def _vals_detail(name: str, vals: dict) -> list:
    return [{"arm": name, "seed": s, "value": x.value if x.ok else None,
             "state": x.state if not x.ok else "ok", "note": x.reason} for s, x in vals.items()]


def score_gate_p(c: Final) -> None:
    r = c.add("Gate P", "mean over healing seeds 0, 1, 2 of the final bf16 k-NN (50,000 queries), Mirror with the "
                        "current A* and SVD-LLM, against 78.34 and 78.06")
    g = gate_p_record(c)
    r.verdict = g["status"]
    m, s = g["arms"]["mirror_exact_astar"], g["arms"]["svdllm"]
    r.value = f"Mirror (current A*) {fmt(m['mean'], '.3f')}; SVD-LLM {fmt(s['mean'], '.3f')}"
    r.n = f"{m['n']} + {s['n']} seeds"
    r.sources = g["sources"]
    r.reason = g["reason"]
    for k, v in g["arms"].items():
        for sd, x in v["seeds"].items():
            r.details.append({"arm": k, "seed": int(sd), "knn_top1": x, "mean": v["mean"],
                              "target": GATE_P_TARGETS[k], "verdict": v["verdict"]})
    rec = c.jload(c.p("results", "A01", "gate_p.json"))
    if isinstance(rec, dict) and "pass" in rec and rec.get("pass") != g["pass"]:
        r.reason += f"; results/A01/gate_p.json records pass {rec.get('pass')} (recomputed here: {g['pass']})"
    for k, v in g["arms"].items():
        c.num(f"gate_p.{k}.knn_mean", v["mean"], desc=f"Gate P: final bf16 k-NN, mean of seeds 0 to 2, {k}",
              sources=v["sources"], n=v["n"], unit="%")


def score_a01(c: Final) -> None:
    r = c.add("A01", "zero-shot k-NN of the A*_w full conversion (step 0 of its heal) >= 58.84, and the mean final "
                     "k-NN of A*_w over healing seeds 0 and 1 >= the current A*'s mean over the same seeds + 0.15")
    rec = c.a01()
    r.verdict = rec["status"]
    h = rec["healed"]
    r.value = (f"zero-shot {fmt(rec['zero_shot']['value'], '.2f')}; healed A*_w {fmt(h['wspace']['mean'], '.3f')} "
               f"vs A* {fmt(h['astar']['mean'], '.3f')} (difference {fmt(h['difference'], '+.3f')})")
    r.n = "zero-shot 1 run; 2 + 2 healed seeds"
    r.sources = rec["sources"]
    r.reason = rec["reason"] + ("" if rec["adopted"] is None else
                                f"; adopted: {'yes' if rec['adopted'] else 'no'} (main arm {rec['main_arm']})")
    v, how = c.main_variant()
    r.reason += f"; main arm used for the later rules: {('mirror_exact_' + v) if v else 'unknown'}, from {how}"
    r.details = [{"part": "zero-shot (step 0)", "value": rec["zero_shot"]["value"], "bar": ">= 58.84",
                  "verdict": rec["zero_shot"]["verdict"]},
                 {"part": "healed A*_w minus A* (seeds 0 and 1)", "value": h["difference"], "bar": ">= 0.15",
                  "verdict": h["verdict"]}]
    met = c.jload(c.p("results", "A01", "metrics.json"))
    if isinstance(met, dict):
        msrc = c.src(c.p("results", "A01", "metrics.json"))
        for row in met.get("operators") or []:
            if isinstance(row, dict):
                r.details.append({"part": f"operator {row.get('operator')}", "d_plus": row.get("d_plus"),
                                  "d_minus": row.get("d_minus"), "zero_shot_fit_run": fnum(row.get("knn_top1")),
                                  "weighted_agreement": fnum(row.get("weighted_agreement"))})
                if row.get("operator") == "wspace_matched":
                    c.num("a01.wspace.d_minus", row.get("d_minus"), desc="A01: d- of A*_w (matched mode)",
                          sources=msrc)
    c.num("a01.zero_shot", rec["zero_shot"]["value"], desc="A01: zero-shot k-NN of the A*_w full conversion (step 0 "
          "of its heal, bf16, 50,000 queries)", sources=rec["sources"][:2], unit="%")
    c.num("a01.healed.wspace_mean_s01", h["wspace"]["mean"], desc="A01: healed A*_w, mean of seeds 0 and 1",
          sources=rec["sources"], n=2, unit="%")
    c.num("a01.healed.astar_mean_s01", h["astar"]["mean"], desc="A01: healed current A*, mean of seeds 0 and 1",
          sources=rec["sources"], n=2, unit="%")
    if rec["adopted"] is not None:
        c.num("a01.adopted", bool(rec["adopted"]), desc="A01: A*_w adopted as the main operator",
              sources=rec["sources"])


# ---------------------------------------------------------------- A02
def _a02(c: Final):
    p = c.p("results", "A02", "metrics.json")
    m = c.jload(p)
    return p, (m if isinstance(m, dict) else None)


IMAGENET_FITS = ("fit0", "fit1", "fit2", "fit3")


def score_a02_structure(c: Final) -> None:
    rules = {"A02-1": c.add("A02-1", "|d-(fit k) - d-(fit 0)| / d-(fit 0) of the full-model operator A*_12, k = 1, 2, "
                                     "3"),
             "A02-2": c.add("A02-2", "energy-weighted odd-subspace agreement (rho(a->b) + rho(b->a)) / 2, every pair "
                                     "of ImageNet fits 0 to 3"),
             "A02-3": c.add("A02-3", "share of strong pairs (blocks 1 to 6) shared, mean of both directions, every "
                                     "pair of ImageNet fits"),
             "A02-4": c.add("A02-4", "share of heads with the same assignment (self or swap), every pair of ImageNet "
                                     "fits")}
    p, m = _a02(c)
    if m is None:
        v_, why = missing_result(c, "results/A02", "results/A02/metrics.json")
        for r in rules.values():
            r.verdict, r.reason = v_, why
        return
    s = c.src(p)
    for r in rules.values():
        r.sources = [s]
    fits = {str(f.get("name")): f for f in m.get("fits") or [] if isinstance(f, dict)}
    # A02-1
    r = rules["A02-1"]
    if "fit0" not in fits or fnum(fits["fit0"].get("d_minus")) is None:
        r.reason = "fit0 missing from results/A02/metrics.json"
    else:
        d0 = float(fits["fit0"]["d_minus"])
        vs, worst = [], None
        for k in IMAGENET_FITS[1:]:
            dk = fnum((fits.get(k) or {}).get("d_minus"))
            if dk is None:
                vs.append(PENDING)
                r.details.append({"fit": k, "d_minus": None, "note": "missing"})
                continue
            rel = abs(dk - d0) / d0 if d0 else None
            v = cmp(rel, "<=", A02_DMINUS_TOL) if rel is not None else PENDING
            vs.append(v)
            worst = rel if worst is None or (rel is not None and rel > worst) else worst
            r.details.append({"fit": k, "d_minus": dk, "d_minus_fit0": d0, "rel_change": rel, "verdict": v})
        r.verdict, r.value, r.n = combine(vs), worst, f"{sum(x != PENDING for x in vs)} of 3 fits against fit 0"
        if r.verdict == PENDING:
            r.reason = "missing: " + ", ".join(d["fit"] for d in r.details if d.get("note") == "missing")
        c.num("a02.d_minus.fit0", int(d0) if d0.is_integer() else d0, desc="A02: d- of A*_12, fit set 0", sources=s)
    # A02-2 to A02-4 over the pairs of ImageNet fits
    pairs = {}
    for q in m.get("pairs") or []:
        if isinstance(q, dict):
            pairs[frozenset((str(q.get("a")), str(q.get("b"))))] = q
    for rid, key, bar in (("A02-2", "weighted_agreement", A02_ODD_MIN), ("A02-3", "twin_mean", A02_TWIN_MIN),
                          ("A02-4", "head_share_same", A02_HEAD_MIN)):
        r = rules[rid]
        vs, vals = [], []
        for a_, b_ in combinations(IMAGENET_FITS, 2):
            q = pairs.get(frozenset((a_, b_)))
            x = fnum((q or {}).get(key))
            v = cmp(x, ">=", bar)
            vs.append(v)
            if x is not None:
                vals.append(x)
            r.details.append({"pair": f"{a_}-{b_}", key: x, "verdict": v if q else PENDING,
                              "note": "" if q else "pair missing"})
        r.verdict = combine(vs)
        r.value = min(vals) if vals else None
        r.n = f"{len(vals)} of 6 pairs"
        if r.verdict == PENDING:
            r.reason = "missing pairs: " + ", ".join(d["pair"] for d in r.details if d.get(key) is None)
        c.num(f"a02.{key}.min", r.value, desc=f"A02: smallest {key} over the pairs of ImageNet fits", sources=s,
              n=len(vals))
    gates = m.get("gates") or {}
    for rid, r in rules.items():
        g = gates.get(rid) if isinstance(gates, dict) else None
        if isinstance(g, dict) and g.get("pass") is not None:
            rec_v = PASS if g["pass"] else FAIL
            if r.verdict in (PASS, FAIL) and rec_v != r.verdict:
                r.reason = (r.reason + "; " if r.reason else "") + f"the producer's gate says {rec_v}"


def a02_heal_arms() -> dict:
    """The A02 heals: main/M/mirror_exact_<fit1|fit2|coco>/s0, each with the adopted operator type fitted on that fit
    set (the file used is in the run's spec.json)."""
    return {fit: arm("main", MAIN, f"mirror_exact_{fit}", label=f"Mirror from {fit}") for fit in ("fit1", "fit2", "coco")}


def structure_used(c: "Final", v: Val) -> str | None:
    """The structure a mirror-family run was built from, as its records name it (metrics.json build.structure_source,
    else spec.json fp_heal.builder_options.structure)."""
    d = v.info.get("run_dir")
    if d is None:
        return None
    m = v.info.get("metrics") or c.jload(Path(d) / "metrics.json") or {}
    s = (m.get("build") or {}).get("structure_source") if isinstance(m.get("build"), dict) else None
    if not s:
        sp = c.jload(Path(d) / "spec.json") or {}
        s = ((sp.get("fp_heal") or {}).get("builder_options") or {}).get("structure")
    return None if s is None else (s if isinstance(s, str) else json.dumps(jsonable(s)))[:200]


def score_a02_heals(c: Final) -> None:
    r5 = c.add("A02-5", "max - min of the final k-NN of Mirror-exact (adopted operator) healed with seed 0 from fit "
                        "sets 0, 1 and 2")
    r6 = c.add("A02-6", "|COCO-fit heal (seed 0) - mean of the three ImageNet-fit heals of A02-5|")
    v, how = c.main_variant()
    if v is None:
        for r in (r5, r6):
            r.reason = how
        return
    arms_ = a02_heal_arms()
    vals = {"fit0": c.final_eval(c.main_arm(), 0), "fit1": c.final_eval(arms_["fit1"], 0),
            "fit2": c.final_eval(arms_["fit2"], 0)}
    coco = c.final_eval(arms_["coco"], 0)
    r5.details = [{"fit_set": k, "knn_top1": x.value, "structure": structure_used(c, x) if x.ok else None,
                   "note": x.reason if not x.ok else ""} for k, x in vals.items()]
    srcs = [s for x in vals.values() for s in x.sources]
    r5.sources = dedupe_sources(srcs)
    if all(x.ok for x in vals.values()):
        xs = [x.value for x in vals.values()]
        spread = max(xs) - min(xs)
        r5.value, r5.verdict, r5.n = spread, cmp(spread, "<=", A02_SPREAD_MAX), "3 heals (seed 0)"
        c.num("a02.heal_spread", spread, desc="A02-5: spread (max - min) of the healed k-NN over fit sets 0 to 2",
              sources=r5.sources, n=3, unit="points")
        mean3 = float(np.mean(xs))
        r6.details = r5.details + [{"fit_set": "coco", "knn_top1": coco.value,
                                    "structure": structure_used(c, coco) if coco.ok else None,
                                    "note": coco.reason if not coco.ok else ""}]
        r6.sources = dedupe_sources(srcs + coco.sources)
        if coco.ok:
            d = coco.value - mean3
            r6.value, r6.verdict, r6.n = d, within(coco.value, mean3, A02_COCO_TOL), "1 COCO heal against 3"
            c.num("a02.coco_minus_imagenet", d, desc="A02-6: COCO-fit heal minus the mean of the ImageNet-fit heals",
                  sources=r6.sources, unit="points")
        else:
            r6.verdict, r6.reason = coco.missing_verdict, coco.reason
    else:
        mv = mean_val(vals.values(), "A02-5 heals")
        r5.verdict, r5.reason = mv.missing_verdict, mv.reason
        bad = [x for x in list(vals.values()) + [coco] if not x.ok]
        r6.verdict = NOT_RUN if any(x.state == "not_run" for x in bad) else PENDING
        r6.reason = "; ".join(x.reason for x in bad)


# ---------------------------------------------------------------- A03
def score_a03(c: Final) -> None:
    r1 = c.add("A03-1", "for each of C1, C2 and C3: twin fraction in blocks 1 to 6 (MUST-04 statistic) <= 0.05 and "
                        "split-half ratio to null (blocks 1 to 6, 20 null draws) >= 0.9")
    p = c.p("results", "A03", "metrics.json")
    m = c.jload(p)
    df = c.cload(c.p("results", "A03", "controls.csv"))
    if not isinstance(m, dict) and df is None:
        r1.verdict, r1.reason = missing_result(c, "results/A03", "results/A03/metrics.json")
    else:
        r1.sources = [c.src(p if isinstance(m, dict) else c.p("results", "A03", "controls.csv"))]
        summ = (m or {}).get("summaries") or {}
        split = (m or {}).get("split_half") or {}
        vs = []
        for ctl in ("mirror", "C1", "C2", "C3"):
            tw = fnum(((summ.get(ctl) or {}).get("1-6") or {}).get("twin_fraction_t50"))
            rin = fnum(((summ.get(ctl) or {}).get("1-6") or {}).get("ratio_to_null"))
            rsp = fnum((split.get(ctl) or {}).get("ratio_to_null"))
            if df is not None and "control" in df:
                row = df[df["control"].astype(str) == ctl]
                if len(row):
                    tw = tw if tw is not None else fnum(row.iloc[0].get("twin_fraction_t50_1-6"))
                    rin = rin if rin is not None else fnum(row.iloc[0].get("ratio_to_null_1-6"))
                    rsp = rsp if rsp is not None else fnum(row.iloc[0].get("ratio_to_null_split_1-6"))
            if ctl == "mirror":
                r1.details.append({"control": ctl, "twin_fraction_1_6": tw, "ratio_in_sample": rin,
                                   "ratio_split_half": rsp, "verdict": "reference (not scored)"})
                continue
            v = combine([cmp(tw, "<=", A03_TWIN_MAX), cmp(rsp, ">=", A03_RATIO_MIN)])
            vs.append(v)
            r1.details.append({"control": ctl, "twin_fraction_1_6": tw, "ratio_in_sample": rin,
                               "ratio_split_half": rsp, "verdict": v})
            c.num(f"a03.{ctl}.twin_fraction_1_6", tw, desc=f"A03: {ctl} twin fraction, blocks 1 to 6", sources=r1.sources)
            c.num(f"a03.{ctl}.ratio_split_half", rsp, desc=f"A03: {ctl} split-half ratio to null, blocks 1 to 6",
                  sources=r1.sources)
            c.num(f"a03.{ctl}.ratio_in_sample", rin, desc=f"A03: {ctl} in-sample ratio to null, blocks 1 to 6 "
                  "(reported beside the scored split-half ratio)", sources=r1.sources)
        r1.verdict = combine(vs)
        scored = [d for d in r1.details if d["control"] != "mirror"]
        r1.value = "; ".join(f"{d['control']}: twins {fmt(d['twin_fraction_1_6'])}, split-half "
                             f"{fmt(d['ratio_split_half'])}" for d in scored)
        r1.n = "3 controls"
        if r1.verdict == PENDING:
            r1.reason = "missing: " + ", ".join(f"{d['control']}" for d in scored
                                                if d["twin_fraction_1_6"] is None or d["ratio_split_half"] is None)
        rec = (((m or {}).get("scores") or {}).get("A03-1") or {}).get("status")
        if rec in (PASS, FAIL) and r1.verdict in (PASS, FAIL) and rec != r1.verdict:
            r1.reason = (r1.reason + "; " if r1.reason else "") + f"the producer's score says {rec}"
    r2 = c.add("A03-2", "final k-NN of H1 (main operator, random pairing, healing seed 0) <= the main Mirror arm's "
                        "3-seed mean - 3.0")
    v, how = c.main_variant()
    if v is None:
        r2.reason = how
        return
    h1 = c.final_eval(H1_ARM, 0)
    mm = c.main_mean()
    d = diff_val(h1, mm)
    r2.verdict = d.verdict("<=", -A03_H1_GAP)
    r2.value, r2.n = d.value, "1 heal against the 3-seed mean"
    r2.sources = dedupe_sources(d.sources)
    r2.reason = d.reason
    r2.details = [{"arm": "H1", "knn_top1": h1.value}, {"arm": "main Mirror (mean)", "knn_top1": mm.value}]
    c.num("a03.h1_minus_mirror", d, desc="A03-2: H1 (true operator, random pairing) minus the main Mirror arm's mean",
          sources=d.sources, unit="points")


# ---------------------------------------------------------------- A04
def score_a04(c: Final) -> None:
    r1 = c.add("A04-1", "final k-NN of the width-pruned (FLAP-style) and the block-dropped dense students, seed 0, "
                        "each <= the main Mirror arm's 3-seed mean - 0.5")
    r2 = c.add("A04-2", "final k-NN of the original distilled with the same recipe (healing without surgery), seed 0")
    mm = c.main_mean()
    vs = []
    for name, folder in (("width-pruned (flap)", "flap"), ("block-dropped (blockdrop)", "blockdrop")):
        x = c.final_eval(arm("main", MAIN, folder), 0)
        d = diff_val(x, mm)
        v = d.verdict("<=", -A04_GAP)
        vs.append(v)
        size = None
        if x.ok:
            pr = (x.info.get("metrics") or {}).get("params") or {}
            size = fnum(pr.get("model_params_frac")) or fnum(pr.get("params_frac"))
        r1.details.append({"student": name, "knn_top1": x.value, "minus_mirror": d.value, "params_frac": size,
                           "verdict": v, "note": d.reason})
        r1.sources += d.sources
        c.num(f"a04.{folder}.knn", x, desc=f"A04: {name} dense student, final k-NN (seed 0)", sources=x.sources,
              unit="%")
        if size is not None:
            c.num(f"a04.{folder}.params_frac", size, desc=f"A04: {name} student, share of the original's parameters",
                  sources=x.sources)
    r1.verdict = combine(vs)
    r1.value = "; ".join(f"{d['student']} {fmt(d['minus_mirror'], '+.2f')}" for d in r1.details)
    r1.n = "2 students (seed 0) against the 3-seed mean"
    r1.sources = dedupe_sources(r1.sources)
    if r1.verdict in (PENDING, NOT_RUN):
        r1.reason = "; ".join(d["note"] for d in r1.details if d["note"])
    o = c.final_eval(arm("main", MAIN, "original"), 0)
    r2.sources = dedupe_sources(o.sources)
    if o.ok:
        r2.verdict, r2.value, r2.n = REPORTED, o.value, "1 heal"
        orig = (o.info.get("eval") or {}).get("original_reference", {}).get("bf16") or {}
        ov = fnum(orig.get("top1"))
        r2.details = [{"row": "original distilled", "knn_top1": o.value},
                      {"row": "original (released, same evaluation)", "knn_top1": ov},
                      {"row": "main Mirror arm (mean)", "knn_top1": mm.value}]
        r2.reason = (f"distilled minus released original {fmt(o.value - ov, '+.2f')}" if ov is not None else "")
        c.num("a04.original_distilled.knn", o, desc="A04-2: original distilled with the healing recipe, final k-NN",
              sources=o.sources, unit="%")
    else:
        r2.verdict, r2.reason = o.missing_verdict, o.reason


# ---------------------------------------------------------------- A05
A05_BASELINES = (("svdllm", (0, 1, 2), "SVD-LLM (whitening only)"), ("svdllm_full", (0, 1, 2), "full SVD-LLM"),
                 ("iosvd", (0, 1, 2), "IO-SVD"), ("slicegpt", (0, 1), "SliceGPT-style"), ("svd", (0,), "plain SVD"))


def a05_mirror_runs(c: Final, variant: str) -> list:
    """(fit set, healing seed, Val): the main arm (fit set 0, seeds 0 to 2) and the A02 heals from fit sets 1 and 2."""
    a = arm("main", MAIN, f"mirror_exact_{variant}")
    h = a02_heal_arms()
    return ([(0, s, c.final_eval(a, s)) for s in SEEDS3]
            + [(1, 0, c.final_eval(h["fit1"], 0)), (2, 0, c.final_eval(h["fit2"], 0))])


def mirror_point(runs: list) -> float:
    """PREREG 2.5: the mean over fit sets of the mean over each fit set's healing seeds."""
    by = {}
    for f, _, v in runs:
        by.setdefault(f, []).append(v.value)
    return float(np.mean([np.mean(x) for x in by.values()]))


def boot_pair(c: Final, a_name: str, a_runs: list, b_name: str, b_runs: list) -> dict:
    """The A13 hierarchical paired bootstrap of (a - b): 10,000 resamples, bootstrap seed 0, 95% percentile interval,
    images drawn once per resample and shared (paired), each method's seeds drawn independently (PREREG 2.5).
    a_runs, b_runs: [(fit set, healing seed, Val)]."""
    runs, labels = [], []
    for name, rs in ((a_name, a_runs), (b_name, b_runs)):
        for f, s, v in rs:
            corr, lab = c.correct(v)
            runs.append(HB.Run(name, f, s, corr))
            labels.append(lab)
    res = HB.hierarchical_bootstrap(runs, n_resamples=c.n_resamples, seed=BOOT_SEED, alpha=BOOT_ALPHA,
                                    pairs=[(a_name, b_name)], paired_seeds=False, labels=labels)
    return res


def score_a05(c: Final) -> None:
    r = c.add("A05-1", "lower end of the 95% interval of (Mirror - best baseline): A13 hierarchical paired bootstrap; "
                       "Mirror = main arm (fit set 0, seeds 0 to 2) and the A02 heals from fit sets 1 and 2; best "
                       "baseline = highest mean healed k-NN among SVD-LLM, full SVD-LLM, IO-SVD, SliceGPT and plain SVD")
    v, how = c.main_variant()
    if v is None:
        r.reason = how
        return
    mir = a05_mirror_runs(c, v)
    base = {}
    for m, seeds, label in A05_BASELINES:
        vals = {s: c.final_eval(arm("main", MAIN, m), s) for s in seeds}
        ok = {s: x for s, x in vals.items() if x.ok}
        base[m] = {"label": label, "vals": vals, "ok": ok,
                   "pending": [s for s, x in vals.items() if x.state == "pending"],
                   "cut": [s for s, x in vals.items() if x.state == "not_run"],
                   "mean": float(np.mean([x.value for x in ok.values()])) if ok else None}
        for s, x in vals.items():
            r.details.append({"method": m, "seed": s, "knn_top1": x.value, "state": "ok" if x.ok else x.state,
                              "note": x.reason if not x.ok else ""})
    for f, s, x in reversed(mir):
        r.details.insert(0, {"method": f"Mirror ({v}) fit set {f}", "seed": s, "knn_top1": x.value,
                             "state": "ok" if x.ok else x.state,
                             "note": (structure_used(c, x) or "") if x.ok else x.reason})
    r.sources = dedupe_sources([s_ for _, _, x in mir for s_ in x.sources]
                               + [s_ for b in base.values() for x in b["vals"].values() for s_ in x.sources])
    c.tables["_a05"] = {"variant": v, "mirror": mir, "base": base}
    notes = [f"{b['label']}: seed(s) {b['cut']} not run (cut)" for b in base.values() if b["cut"]]
    bad_m = [x for _, _, x in mir if not x.ok]
    if bad_m:
        r.verdict = NOT_RUN if any(x.state == "not_run" for x in bad_m) else PENDING
        r.reason = "; ".join(x.reason for x in bad_m)
        return
    pend = [m for m, b in base.items() if b["pending"] or not b["ok"]]
    if pend:
        r.verdict = PENDING
        r.reason = "best baseline not known yet: " + "; ".join(
            f"{m} seeds {base[m]['pending'] or sorted(base[m]['vals'])} missing" for m in pend)
        return
    best = max(base, key=lambda m: base[m]["mean"])
    mp = mirror_point(mir)
    bp = base[best]["mean"]
    try:
        res = boot_pair(c, "mirror", mir, best, [(0, s, x) for s, x in sorted(base[best]["ok"].items())])
    except (FileNotFoundError, KeyError, ValueError) as e:
        r.verdict, r.reason = PENDING, f"bootstrap inputs: {e}"
        return
    pr = res["pairs"][0]
    lo, hi = pr["ci_low"], pr["ci_high"]
    c.boot["A05-1"] = {k: v_ for k, v_ in res.items() if k != "draws"}
    r.verdict = cmp(lo, ">", A05_LOWER)
    sup = cmp(lo, ">", 0.0) == PASS
    r.value = (f"Mirror - {best} {mp - bp:+.3f} (point estimate), 95% interval [{lo:+.3f}, {hi:+.3f}]; "
               f"superiority {'claimed' if sup else 'not claimed'} (lower end {'>' if sup else '<='} 0)")
    r.n = (f"Mirror {len(mir)} runs over 3 fit sets; {best} {len(base[best]['ok'])} seeds; "
           f"{res['n_images']:,} images; {res['n_resamples']:,} resamples")
    r.reason = "; ".join(notes + [f"bootstrap (run-weighted) mean difference {pr['diff']:+.3f}, effect size "
                                  f"{fmt(pr['effect_size'])}"])
    c.num("a05.best_baseline", best, desc="A05-1: best baseline (highest mean healed k-NN)", sources=r.sources)
    c.num("a05.mirror_point", mp, desc="A05-1: Mirror point estimate (mean over fit sets 0 to 2 of the fit-set means)",
          sources=r.sources, n=len(mir), unit="%")
    c.num("a05.mirror_minus_best.point", mp - bp, desc="A05-1: Mirror minus best baseline, point estimate",
          sources=r.sources, unit="points")
    c.num("a05.mirror_minus_best.ci_low", lo, desc="A05-1: lower end of the 95% interval (A13 bootstrap, 10,000 "
          "resamples, seed 0)", sources=r.sources, unit="points")
    c.num("a05.mirror_minus_best.ci_high", hi, desc="A05-1: upper end of the 95% interval", sources=r.sources,
          unit="points")
    c.num("a05.superiority", bool(sup), desc="A05-1: superiority claimed (lower end above 0)", sources=r.sources)
    for m, b in base.items():
        c.num(f"a05.{m}.mean", b["mean"], desc=f"A05: {b['label']}, mean healed k-NN over its seeds",
              sources=[s_ for x in b["ok"].values() for s_ in x.sources], n=len(b["ok"]), unit="%")


# ---------------------------------------------------------------- A06
def a06_point(c: Final, method: str, tag: str) -> Val:
    seeds = (0, 1) if tag == "f050" else (0,)
    vals = [c.final_eval(arm("a06", MAIN, f"{method}_{tag}"), s) for s in seeds]
    v = mean_val(vals, f"{method}_{tag}")
    v.info["seeds"] = seeds
    v.info["vals"] = vals
    return v


def score_a06(c: Final) -> None:
    r1 = c.add("A06-1", "Mirror + SVD-LLM minus SVD-LLM at 0.65, 0.50 and 0.40 of the full size (seed 0; mean of "
                        "seeds 0 and 1 at 0.50) >= 0.5 at 2 or more budgets, and the 95% interval at 0.50 (A13 "
                        "bootstrap over images and seeds 0 and 1) above 0")
    r2 = c.add("A06-2", "measured gain acc(Mirror + SVD, B) - acc(SVD-LLM, B) within 50% of the pre-registered "
                        "prediction acc_pred(B) - acc_SVD(B) (mirror.final.baselines.sizes) at 2 or more budgets")
    r3 = c.add("A06-3", "final k-NN of Mirror on the first 9 blocks (seed 0) minus the full conversion (main arm)")
    pts = {}
    vs1 = []
    for B, tag in A06_BUDGETS:
        ms, sv = a06_point(c, "mirror_svdllm", tag), a06_point(c, "svdllm", tag)
        d = diff_val(ms, sv, f"budget {B}")
        v = d.verdict(">=", A06_GAIN)
        pts[B] = {"tag": tag, "mirror_svd": ms, "svd": sv, "diff": d}
        vs1.append(v)
        r1.details.append({"budget": f"{B:.2f}", "mirror_svdllm": ms.value, "svdllm": sv.value, "difference": d.value,
                           "verdict": v, "note": d.reason})
        r1.sources += d.sources
        c.num(f"a06.{tag}.mirror_svdllm", ms, desc=f"A06: Mirror + SVD-LLM at {B} of the full size, final k-NN",
              sources=ms.sources, n=len(ms.info["seeds"]), unit="%")
        c.num(f"a06.{tag}.svdllm", sv, desc=f"A06: SVD-LLM at {B} of the full size, final k-NN", sources=sv.sources,
              n=len(sv.info["seeds"]), unit="%")
    count = k_of_n(vs1, A06_K)
    # the interval at 0.50
    p50 = pts[0.50]
    iv = PENDING
    note = ""
    if p50["mirror_svd"].ok and p50["svd"].ok:
        try:
            res = boot_pair(c, "mirror_svdllm", [(0, s, x) for s, x in zip((0, 1), p50["mirror_svd"].info["vals"])],
                            "svdllm", [(0, s, x) for s, x in zip((0, 1), p50["svd"].info["vals"])])
            pr = res["pairs"][0]
            c.boot["A06-1"] = {k: v_ for k, v_ in res.items() if k != "draws"}
            iv = cmp(pr["ci_low"], ">", 0.0)
            note = f"interval at 0.50 [{pr['ci_low']:+.3f}, {pr['ci_high']:+.3f}] ({iv})"
            c.num("a06.f050.diff_ci_low", pr["ci_low"], desc="A06-1: lower end of the 95% interval of Mirror + SVD-LLM "
                  "minus SVD-LLM at 0.50", sources=r1.sources, unit="points")
            c.num("a06.f050.diff_ci_high", pr["ci_high"], desc="A06-1: upper end of the same interval",
                  sources=r1.sources, unit="points")
        except (FileNotFoundError, KeyError, ValueError) as e:
            iv, note = PENDING, f"interval at 0.50: {e}"
    else:
        bad = [x for x in (p50["mirror_svd"], p50["svd"]) if not x.ok]
        iv = NOT_RUN if any(x.state == "not_run" for x in bad) else PENDING
        note = "interval at 0.50: " + "; ".join(x.reason for x in bad)
    r1.verdict = combine([count, iv])
    r1.value = "; ".join(f"{d['budget']}: {fmt(d['difference'], '+.2f')}" for d in r1.details) + f"; {note}"
    r1.n = f"{sum(x in (PASS, FAIL) for x in vs1)} of 3 budgets scored; at least {A06_K} must pass ({count})"
    r1.sources = dedupe_sources(r1.sources)
    if r1.verdict in (PENDING, NOT_RUN):
        r1.reason = "; ".join(d["note"] for d in r1.details if d["note"]) or note
    # A06-2
    mm = c.main_mean()
    svd765 = mean_val([c.final_eval(arm("main", MAIN, "svdllm"), s) for s in SEEDS3], "SVD-LLM at 0.765")
    orig = c.final_eval(arm("main", MAIN, "original"), 0)
    svd_pts = {B: pts[B]["svd"] for B, _ in A06_BUDGETS}
    svd_pts[0.765] = svd765
    grid = sorted(list(svd_pts) + [1.0])
    vs2 = []
    try:
        from mirror.final.baselines import sizes as SZ
    except Exception as e:  # noqa: BLE001
        SZ = None
        r2.reason = f"mirror.final.baselines.sizes not importable: {e}"
    for B, tag in A06_BUDGETS:
        x = B / SZ.MIRROR_SIZE if SZ is not None else B / 0.765
        lo_ = max(g for g in grid if g <= x + 1e-12)
        hi_ = min(g for g in grid if g >= x - 1e-12)
        need = [svd_pts[g] for g in {lo_, hi_, B} if g in svd_pts] + [orig, mm]
        meas = pts[B]["diff"]
        if SZ is None or not all(v_.ok for v_ in need) or not meas.ok:
            bad = [v_ for v_ in need + [meas] if not v_.ok]
            v = NOT_RUN if any(v_.state == "not_run" for v_ in bad) else PENDING
            vs2.append(v)
            r2.details.append({"budget": f"{B:.2f}", "predicted_gain": None, "measured_gain": meas.value, "verdict": v,
                               "note": "; ".join(v_.reason for v_ in bad if v_.reason)})
            continue
        avail = {g: svd_pts[g].value for g in svd_pts if svd_pts[g].ok}
        pred = SZ.predicted_gain(B, avail, orig.value, mm.value)
        ok = SZ.gain_within(meas.value, pred, A06_REL)
        v = PASS if ok else FAIL
        vs2.append(v)
        r2.details.append({"budget": f"{B:.2f}", "predicted_gain": pred, "measured_gain": meas.value,
                           "abs_error": abs(meas.value - pred), "allowed": A06_REL * abs(pred), "verdict": v})
        c.num(f"a06.{tag}.predicted_gain", pred, desc=f"A06-2: pre-registered predicted stacking gain at {B}",
              sources=dedupe_sources(meas.sources + orig.sources + mm.sources + svd765.sources), unit="points")
    r2.verdict = k_of_n(vs2, A06_K)
    r2.value = "; ".join(f"{d['budget']}: measured {fmt(d['measured_gain'], '+.2f')} vs predicted "
                         f"{fmt(d['predicted_gain'], '+.2f')}" for d in r2.details)
    r2.n = f"{sum(x in (PASS, FAIL) for x in vs2)} of 3 budgets scored; at least {A06_K} must pass"
    r2.sources = dedupe_sources(r1.sources + orig.sources + mm.sources + svd765.sources)
    if r2.verdict in (PENDING, NOT_RUN):
        r2.reason = "; ".join(d["note"] for d in r2.details if d.get("note")) or r2.reason
    r2.details.append({"budget": "inputs", "note": f"distilled original {fmt(orig.value)}, healed Mirror at 0.765 "
                       f"{fmt(mm.value)}, SVD-LLM at 0.765 {fmt(svd765.value)} (3-seed mean)"})
    # A06-3
    k9 = c.final_eval(arm("a06", MAIN, "mirror_partial_k9"), 0)
    d = diff_val(k9, mm)
    r3.sources = dedupe_sources(d.sources)
    if d.ok:
        r3.verdict, r3.value, r3.n = REPORTED, d.value, "1 heal against the 3-seed mean"
        m0 = c.final_eval(c.main_arm(), 0) if c.main_arm() else Val()
        r3.details = [{"row": "9-block Mirror (seed 0)", "knn_top1": k9.value},
                      {"row": "full conversion (main arm mean)", "knn_top1": mm.value},
                      {"row": "full conversion (main arm seed 0)", "knn_top1": m0.value}]
        if m0.ok:
            r3.reason = f"paired by healing seed 0: {k9.value - m0.value:+.3f}"
        c.num("a06.price_of_exactness", d, desc="A06-3: 9-block Mirror minus the full conversion (main arm mean)",
              sources=d.sources, unit="points")
    else:
        r3.verdict, r3.reason = d.missing_verdict, d.reason


# ---------------------------------------------------------------- A07
def a12_entries(c: Final, model: str = MAIN) -> tuple:
    """({name: row of systems_table.csv + its meta}, source) from results/A12/<model>/."""
    d = c.p("results", "A12", model)
    df = c.cload(d / "systems_table.csv")
    if df is None:
        return {}, None
    meta = c.jload(d / "meta.json") or {}
    models = meta.get("models") or {}
    g = c.meta_git(d / "systems_table.csv")
    out = {}
    for _, row in df.iterrows():
        name = str(row.get("name"))
        e = {k: row.get(k) for k in df.columns}
        e["meta"] = models.get(name) if isinstance(models.get(name), dict) else {}
        tp = fnum(row.get("throughput_b256"))
        e["ms_per_image_b256"] = (1000.0 / tp) if tp else None
        out[name] = e
    return out, {"file": c.rel(d / "systems_table.csv"), "git_hash": g}


def _run_key(path) -> tuple | None:
    s = str(path or "").replace("\\", "/")
    i = s.find("runs/final/")
    if i < 0:
        return None
    parts = s[i + len("runs/final/"):].strip("/").split("/")
    if len(parts) < 4 or not re.fullmatch(r"s\d+", parts[3]):
        return None
    return parts[0], parts[1], parts[2], int(parts[3][1:])


def original_evals(c: Final) -> dict:
    """{model: (eval record, path)} of released-model evaluations (eval/original.json with "original": true) under
    runs/final/A07, results/A07 (and runs/final/a07)."""
    out = {}
    for base in (c.p("runs", "final", "A07"), c.p("runs", "final", "a07"), c.p("results", "A07")):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("original.json")):
            if p.parent.name != "eval":
                continue
            rec = c.jload(p)
            if isinstance(rec, dict) and rec.get("model"):
                out.setdefault(str(rec["model"]), (rec, p))
    return out


def score_a07(c: Final) -> None:
    r = c.add("A07", "main Mirror arm's 3-seed mean k-NN >= the best k-NN among flip-averaged candidates (A06 SVD-LLM "
                     "models, DINOv2-S/14 reg, the original) whose measured time per image at batch 256 (two passes) is "
                     "at most Mirror's faster stored form, + 0.5")
    mm = c.main_mean()
    v, how = c.main_variant()
    ents, a12src = a12_entries(c)
    if v is None:
        r.reason = how
        return
    if not ents:
        r.verdict, r.reason = missing_result(c, f"results/A12/{MAIN}", f"results/A12/{MAIN}/systems_table.csv")
        return
    main_folder = f"mirror_exact_{v}"
    mirror_t = [(n, e["ms_per_image_b256"]) for n, e in ents.items()
                if (_run_key((e["meta"] or {}).get("run_dir")) or ("",) * 4)[:3] == ("main", MAIN, main_folder)
                and e["ms_per_image_b256"] is not None]
    if not mirror_t:
        r.reason = f"no timed entry of the main Mirror arm ({main_folder}) in results/A12/{MAIN}/systems_table.csv"
        return
    m_name, m_ms = min(mirror_t, key=lambda t: t[1])

    def compute_for(match, fallback=None) -> tuple:
        """(ms per image of the flip-averaged model, how): a flip_average entry on the base, else 2 x the base. A
        healing seed does not change a model's shapes, so `fallback` (another seed of the same arm) may stand in."""
        base = [n for n, e in ents.items() if match(e) and str(e.get("kind")) != "flip_average"]
        if not base and fallback is not None:
            base = [n for n, e in ents.items() if fallback(e) and str(e.get("kind")) != "flip_average"]
        fa = [n for n, e in ents.items() if str(e.get("kind")) == "flip_average"
              and str((e["meta"] or {}).get("base")) in base and ents[n]["ms_per_image_b256"] is not None]
        if fa:
            return ents[fa[0]]["ms_per_image_b256"], f"measured ({fa[0]})"
        bt = [ents[n]["ms_per_image_b256"] for n in base if ents[n]["ms_per_image_b256"] is not None]
        if bt:
            return 2.0 * min(bt), f"2 x single pass ({base[0]})"
        return None, "not timed"

    cands = []
    for B, tag in A06_BUDGETS:          # the seed-0 model of every A06 budget (the flip-averaged evaluations of A07)
        s = 0
        a = arm("a06", MAIN, f"svdllm_{tag}")
        kv = c.final_eval(a, s, key="bf16_flipavg")
        if kv.state == "not_run":
            r.details.append({"candidate": f"SVD-LLM {B:.2f} (seed {s})", "note": kv.reason, "eligible": None})
            continue
        ms, how_t = compute_for(lambda e, a=a, s=s: _run_key((e["meta"] or {}).get("run_dir")) ==
                                ("a06", MAIN, a.folders[0], s),
                                lambda e, a=a: (_run_key((e["meta"] or {}).get("run_dir")) or ("",) * 4)[:3] ==
                                ("a06", MAIN, a.folders[0]))
        cands.append((f"SVD-LLM {B:.2f} (seed {s}), flip-averaged", kv, ms, how_t))
    oe = original_evals(c)
    for model, label in ((SMALL, "DINOv2-S/14 reg, flip-averaged"), (MAIN, "original DINOv2-B/14 reg, flip-averaged")):
        kv = Val(state="pending", reason=f"no bf16_flipavg evaluation of {model} (eval/original.json)")
        if model in oe:
            rec, p = oe[model]
            pr = (rec.get("passes") or {}).get("bf16_flipavg")
            if isinstance(pr, dict) and fnum(pr.get("top1")) is not None and int(pr.get("n_query") or 0) == c.n_val:
                kv = Val(fnum(pr["top1"]), "ok", sources=[c.src(p)])
        if not kv.ok and model == MAIN:
            fa = c.cload(c.p("results", "MUST-10", "flipavg.csv"))
            if fa is not None and {"variant", "mode", "knn_top1"} <= set(fa.columns):
                sub = fa[(fa["variant"] == "original") & (fa["mode"] == "flipavg")]
                if "model" in fa:
                    sub = sub[sub["model"] == MAIN] if (sub["model"] == MAIN).any() else sub
                if len(sub) and fnum(sub["knn_top1"].iloc[0]) is not None:
                    g = str(sub["git_hash"].iloc[0]) if "git_hash" in sub else None
                    kv = Val(fnum(sub["knn_top1"].iloc[0]), "ok",
                             sources=[c.src(c.p("results", "MUST-10", "flipavg.csv"), g)],
                             info={"note": "MUST-10 (previous phase)"})
        ms, how_t = compute_for(lambda e, model=model: str(e.get("kind")) == "original"
                                and str((e["meta"] or {}).get("model")) == model)
        cands.append((label, kv, ms, how_t))
    known, pend = [], []
    for label, kv, ms, how_t in cands:
        elig = None if ms is None else ms <= m_ms + 1e-12
        r.details.append({"candidate": label, "knn_top1": kv.value, "ms_per_image_b256": ms, "timing": how_t,
                          "eligible": elig, "note": kv.reason if not kv.ok else (kv.info.get("note") or "")})
        r.sources += kv.sources
        if ms is None or (elig and not kv.ok):
            pend.append(label)
        elif elig:
            known.append((label, kv.value))
    r.details.insert(0, {"candidate": f"Mirror ({main_folder}, {m_name})", "knn_top1": mm.value,
                         "ms_per_image_b256": m_ms, "timing": "measured (faster stored form)", "eligible": None})
    r.sources = dedupe_sources(r.sources + mm.sources + [a12src])
    best = max(known, key=lambda t: t[1]) if known else None
    if not mm.ok:
        r.verdict, r.reason = mm.missing_verdict, mm.reason
    elif best is not None and cmp(mm.value, ">=", best[1] + A07_MARGIN) == FAIL:
        r.verdict = FAIL
    elif pend:
        r.verdict, r.reason = PENDING, "candidates not yet measured (k-NN or time): " + "; ".join(pend)
    elif best is None:
        r.verdict = PENDING
        r.reason = ("no flip-averaged candidate has a measured time at or below Mirror's: the comparison of the rule "
                    "is undefined and is not scored")
    else:
        r.verdict = cmp(mm.value, ">=", best[1] + A07_MARGIN)
    if best is not None and mm.ok:
        r.value = f"Mirror {mm.value:.3f} vs best eligible {best[0]} {best[1]:.3f} (difference {mm.value - best[1]:+.3f})"
        c.num("a07.mirror_minus_best_flipavg", mm.value - best[1], desc="A07: Mirror (3-seed mean) minus the best "
              "flip-averaged model at or below Mirror's measured compute", sources=r.sources, unit="points")
    r.n = f"{len(cands)} candidates, {len(known)} eligible with a value"
    c.num("a07.mirror_ms_per_image_b256", m_ms, desc="A07: Mirror's faster stored form, time per image at batch 256 "
          "(bf16, compiled)", sources=[a12src], unit="ms")


# ---------------------------------------------------------------- A08
def classifier_eval(c: Final, a: Arm, seed) -> tuple:
    """DeiT-III classifier evaluation of the final checkpoint (scripts/fp_a08_top1.py): <run>/eval/top1_step<S>.json
    with one mirror.final.adapters.top1_eval result per pass, "bf16" (the headline top-1) and "fp32" (float32, TF32
    off: the flip consistency of the predicted class). Returns (bf16 top-1 Val, fp32 record or None, source)."""
    d, m, S, bad = c.run_metrics(a, seed)
    if bad is not None:
        return bad, None, None
    p = d / "eval" / f"top1_step{S}.json"
    rec = c.jload(p)
    if not isinstance(rec, dict):
        return Val(state="pending", reason=f"{c.rel(p)} missing (classifier top-1 not evaluated yet)"), None, None
    src = c.src(p)
    top = rec.get("bf16")
    f32 = rec.get("fp32") if isinstance(rec.get("fp32"), dict) else None
    if not isinstance(top, dict) or fnum(top.get("top1")) is None:
        return Val(state="pending", reason=f"{c.rel(p)}: no bf16 pass yet"), f32, src
    n = top.get("n")
    if n is None or int(n) != c.n_val:
        return Val(state="pending", reason=f"{c.rel(p)} [bf16]: top-1 on {n} images, not {c.n_val:,}"), f32, src
    return Val(fnum(top["top1"]), "ok", sources=[c.src(d / "metrics.json"), src], info={"record": top}), f32, src


def released_top1(c: Final, model: str = DEIT) -> Val:
    """Top-1 of the released model's own classifier: runs/final/a08/<model>/_original/eval/top1_original.json."""
    p = c.p("runs", "final", "a08", model, "_original", "eval", "top1_original.json")
    rec = c.jload(p)
    top = rec.get("bf16") if isinstance(rec, dict) else None
    if not isinstance(top, dict) or fnum(top.get("top1")) is None:
        return Val(state="pending", reason=f"{c.rel(p)} missing")
    return Val(fnum(top["top1"]), "ok", sources=[c.src(p)])


def a09_variants(c: Final, model: str) -> list:
    base = c.p("results", "A09", model)
    out = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            m = c.jload(d / "metrics.json")
            if isinstance(m, dict):
                out.append((d, m))
    return out


def score_a08(c: Final) -> None:
    r1 = c.add("A08-1", "DeiT-III-B: mean top-1 (own classifier, 50,000 images, seeds 0 to 2) of Mirror >= SVD-LLM - "
                        "0.5; 100% float32 flip consistency (predicted class) of Mirror; stored parameters equal the "
                        "count formula (0 percentage points)")
    r2 = c.add("A08-2", "DINOv2-L/14 reg, seed 0: Mirror k-NN >= SVD-LLM k-NN - 0.5; 100% float32 flip consistency")
    # A08-1 (docs/final_phase_design.md 4.5a: a08/<model>/mirror_exact, svdllm, random_structure; the operator
    # type follows the A01 decision and is recorded in each run's spec.json)
    ma, sa = arm("a08", DEIT, "mirror_exact", label="Mirror"), arm("a08", DEIT, "svdllm", label="SVD-LLM")
    mres = {s: classifier_eval(c, ma, s) for s in SEEDS3}
    sres = {s: classifier_eval(c, sa, s) for s in SEEDS3}
    mt = mean_val([x[0] for x in mres.values()], "DeiT-III Mirror top-1")
    st_ = mean_val([x[0] for x in sres.values()], "DeiT-III SVD-LLM top-1")
    d = diff_val(mt, st_)
    part1 = d.verdict(">=", -A08_GAP)
    flips, fnotes = [], []
    for s, (tv, frec, fsrc) in mres.items():
        if tv.state == "not_run":
            flips.append(NOT_RUN)
            fnotes.append(f"s{s}: {tv.reason}")
            continue
        if frec is None:
            flips.append(PENDING)
            fnotes.append(f"s{s}: no fp32 pass" + (f" ({tv.reason})" if tv.reason else ""))
            continue
        fv, fn = exact_flip(frec, c.n_val, strict_key="max_margin_rel", not_tie_key="n_disagree_not_near_tie",
                            one_copy_keys=("disagree_max_margin",), one_copy_list="disagree_margins", n_key="n")
        flips.append(fv)
        fnotes.append(f"s{s}: {fn}")
        r1.sources.append(fsrc)
    part2 = combine(flips)
    # parameter formula
    part3, pnote = PENDING, "no parameter check (results/A08/deit3_b16/params_check.json or law_per_layer.csv)"
    pc = c.jload(c.p("results", "A08", DEIT, "params_check.json"))
    law = c.cload(c.p("results", "A08", DEIT, "law_per_layer.csv"))
    if isinstance(pc, dict):
        dpp = fnum(pc.get("max_abs_diff_pp"))
        if dpp is None and fnum(pc.get("params_stored")) is not None and fnum(pc.get("params_formula")) is not None:
            dpp = abs(fnum(pc["params_stored"]) - fnum(pc["params_formula"]))
        if dpp is not None:
            part3, pnote = cmp(dpp, "<=", 0.0), f"largest |stored - formula| {dpp:g}"
            r1.sources.append(c.src(c.p("results", "A08", DEIT, "params_check.json")))
    elif law is not None:
        cols = [x for x in ("diff_pp_fc1", "diff_pp_fc2", "diff_pp_mlp") if x in law]
        if cols:
            dpp = float(law[cols].abs().to_numpy().max())
            part3, pnote = cmp(dpp, "<=", 0.0), f"largest |predicted - achieved| {dpp:g} pp over {len(law)} rows"
            r1.sources.append(c.src(c.p("results", "A08", DEIT, "law_per_layer.csv")))
    if part3 == PENDING and not mt.ok and mt.state == "not_run":
        part3 = NOT_RUN
    r1.verdict = combine([part1, part2, part3])
    r1.value = (f"top-1 Mirror {fmt(mt.value, '.3f')} vs SVD-LLM {fmt(st_.value, '.3f')} (difference "
                f"{fmt(d.value, '+.3f')}); flip: {', '.join(flips) or 'n/a'}; parameters: {pnote}")
    r1.n = f"{sum(x[0].ok for x in mres.values())} + {sum(x[0].ok for x in sres.values())} seeds"
    r1.sources = dedupe_sources(r1.sources + d.sources)
    r1.details = ([{"part": "top-1 difference", "value": d.value, "verdict": part1, "note": d.reason}]
                  + [{"part": "fp32 flip", "value": None, "verdict": f, "note": n} for f, n in zip(flips, fnotes)]
                  + [{"part": "parameters", "value": None, "verdict": part3, "note": pnote}])
    if r1.verdict in (PENDING, NOT_RUN):
        r1.reason = "; ".join(x["note"] for x in r1.details if x["verdict"] in (PENDING, NOT_RUN) and x["note"])
    c.num("a08.deit3.mirror_top1_mean", mt, desc="A08-1: DeiT-III-B Mirror, classifier top-1, mean of seeds 0 to 2",
          sources=mt.sources, n=3, unit="%")
    c.num("a08.deit3.svdllm_top1_mean", st_, desc="A08-1: DeiT-III-B SVD-LLM, classifier top-1, mean of seeds 0 to 2",
          sources=st_.sources, n=3, unit="%")
    orig_top1 = released_top1(c, DEIT)
    if orig_top1.ok:
        r1.details.append({"part": "released DeiT-III-B top-1 (reported)", "value": orig_top1.value, "verdict": "",
                           "note": orig_top1.sources[0]["file"]})
        c.num("a08.deit3.original_top1", orig_top1, desc="A08: released DeiT-III-B/16, classifier top-1 (bf16)",
              sources=orig_top1.sources, unit="%")
    # A08-2
    ml, sl = arm("a08", LARGE, "mirror_exact", label="Mirror"), arm("a08", LARGE, "svdllm", label="SVD-LLM")
    mk, sk = c.final_eval(ml, 0), c.final_eval(sl, 0)
    d2 = diff_val(mk, sk)
    p1 = d2.verdict(">=", -A08_GAP)
    f32 = c.final_eval(ml, 0, key="fp32")
    if f32.ok:
        p2, fnote = exact_flip(f32.info["record"], c.n_val)
    else:
        p2, fnote = f32.missing_verdict, f32.reason
    r2.verdict = combine([p1, p2])
    r2.value = (f"Mirror {fmt(mk.value, '.3f')} vs SVD-LLM {fmt(sk.value, '.3f')} (difference "
                f"{fmt(d2.value, '+.3f')}); fp32 flip: {fnote}")
    r2.n = "1 + 1 heals (indicative)"
    r2.sources = dedupe_sources(d2.sources + f32.sources)
    if r2.verdict in (PENDING, NOT_RUN):
        r2.reason = "; ".join(x for x in (d2.reason, fnote if p2 in (PENDING, NOT_RUN) else "") if x)
    c.num("a08.dinov2l.mirror_knn", mk, desc="A08-2: DINOv2-L/14 reg Mirror, final k-NN (seed 0)", sources=mk.sources,
          unit="%")
    c.num("a08.dinov2l.svdllm_knn", sk, desc="A08-2: DINOv2-L/14 reg SVD-LLM, final k-NN (seed 0)", sources=sk.sources,
          unit="%")


# ---------------------------------------------------------------- A09
def _a09_rows(c: Final, d: Path) -> list | None:
    df = c.cload(d / "components.csv")
    if df is None:
        return None
    rows = []
    for _, x in df.iterrows():
        rows.append({k: x.get(k) for k in df.columns})
    return rows


def a09_unit(c: Final, d: Path, m: dict) -> dict:
    """A09-1, A09-2 and A09-3 of one audited model (PREREG Sections 1 and 2.6) from components.csv (recomputed), else
    from metrics.json's scores."""
    rows = _a09_rows(c, d)
    srcs = [c.src(d / "metrics.json")] + ([c.src(d / "components.csv")] if rows is not None else [])
    sc = m.get("scores") or {}
    out = {"variant": d.name, "model": d.parent.name, "sources": srcs}
    if rows is not None and rows and "eq_rel_tensor_max" in rows[0]:
        span = [r for r in rows if truthy(r.get("in_span")) and str(r.get("form")) != "original"
                and str(r.get("space")) != "none"]
        e1 = [fnum(r.get("eq_rel_tensor_max")) for r in span if str(r.get("precision")) == "float64_exact"]
        e1 = [x for x in e1 if x is not None]
        out["a09_1_max"] = max(e1) if e1 else None
        e2 = [r for r in span if str(r.get("precision")) == "float32"]
        oks = []
        for r in e2:
            flag = truthy(r.get("a09_2_pass_exact_floor"))
            if flag is None:
                eq, dev = fnum(r.get("eq_rel_tensor_max")), fnum(r.get("dev_exact_rel_tensor_max"))
                flag = None if eq is None or dev is None else eq <= A09_2_FACTOR * dev
            oks.append(flag)
        out["a09_2_tensors"] = (None if not e2 or any(x is None for x in oks) else all(oks))
        out["a09_2_n_fail"] = sum(1 for x in oks if x is False)
        out["a09_2_n"] = len(e2)
    else:
        s1, s2 = sc.get("A09-1") or {}, sc.get("A09-2") or {}
        out["a09_1_max"] = fnum(s1.get("max_rel_error"))
        out["a09_2_tensors"] = s2.get("tensors_pass") if isinstance(s2.get("tensors_pass"), bool) else None
        out["a09_2_n_fail"] = s2.get("n_tensors_fail")
        out["a09_2_n"] = s2.get("n_tensors")
    flip = m.get("flip") or {}
    out["flip"] = {}
    for form in ("dense", "even_odd"):
        f = flip.get(f"{form}/float32")
        out["flip"][form] = exact_flip(f, c.n_val, strict_key="max_margin_rel", one_copy_list="disagree_margins",
                                       n_key="n_images") if isinstance(f, dict) else (PENDING, "not measured")
        out[f"flip_fp32_{form}"] = fnum((f or {}).get("flip_consistency"))
        for prec in ("bf16_autocast", "bf16"):
            g = flip.get(f"{form}/{prec}")
            if isinstance(g, dict):
                out[f"flip_bf16_{form}"] = fnum(g.get("flip_consistency"))
    rep = ((sc.get("A09-3") or {}).get("by_form") or {})
    out["a09_3"] = {k: {"max_rel_error": fnum((v or {}).get("max_rel_error")),
                        "flip_consistency": fnum((v or {}).get("flip_consistency"))} for k, v in rep.items()}
    return out


def score_a09(c: Final) -> None:
    r1 = c.add("A09-1", "largest relative equivariance error at any tensor, float64 (re-projected exact weights), both "
                        "forms, 1,000 validation images")
    r2 = c.add("A09-2", "every float32 tensor: equivariance error <= 2 x the model's own float32-versus-float64 "
                        "deviation (exact floor); and float32 flip consistency 100% for the dense and even/odd forms")
    r3 = c.add("A09-3", "bfloat16: largest error and flip consistency for both forms (target 100% for even/odd)")
    v, how = c.main_variant()
    units = []
    main_vars = a09_variants(c, MAIN)
    want = None if v is None else f"mirror_exact_{v}_s"
    req = [x for x in main_vars if want and x[0].name.startswith(want)]
    for dd, m in main_vars + a09_variants(c, DEIT):
        units.append(a09_unit(c, dd, m))
    if not req:
        if want:
            v_, miss = missing_result(c, f"results/A09/{MAIN}", f"results/A09/{MAIN}/{want}<seed>/ (the healed main "
                                                                 "arm's audit)")
        else:
            v_, miss = PENDING, f"the main arm is not known ({how})"
        for r in (r1, r2, r3):
            r.verdict, r.reason = v_, miss
    for u in units:
        for r in (r1, r2, r3):
            r.sources += u["sources"]
    v1 = [cmp(u["a09_1_max"], "<=", A09_1_TOL) for u in units]
    v2 = []
    for u in units:
        t = PENDING if u["a09_2_tensors"] is None else (PASS if u["a09_2_tensors"] else FAIL)
        v2.append(combine([t, u["flip"]["dense"][0], u["flip"]["even_odd"][0]]))
    for u, a, b in zip(units, v1, v2):
        r1.details.append({"model": u["model"], "variant": u["variant"], "max_rel_error_fp64": u["a09_1_max"],
                           "verdict": a})
        r2.details.append({"model": u["model"], "variant": u["variant"], "tensors_pass": u["a09_2_tensors"],
                           "tensors_fail": u["a09_2_n_fail"], "flip_dense": u["flip"]["dense"][1],
                           "flip_even_odd": u["flip"]["even_odd"][1], "verdict": b})
        r3.details.append({"model": u["model"], "variant": u["variant"], "flip_bf16_dense": u.get("flip_bf16_dense"),
                           "flip_bf16_even_odd": u.get("flip_bf16_even_odd"),
                           "max_rel_error": {k: x["max_rel_error"] for k, x in u["a09_3"].items()}})
    if req:
        has_deit = any(u["model"] == DEIT for u in units)
        extra = []
        if not has_deit:
            v_, note = missing_result(c, f"results/A09/{DEIT}", f"results/A09/{DEIT}/ (the DeiT-III-B audit)")
            extra = [v_]
            r1.reason = r2.reason = r3.reason = note
        r1.verdict = combine(v1 + extra)
        r2.verdict = combine(v2 + extra)
        r3.verdict = REPORTED if units else PENDING
        mx = [u["a09_1_max"] for u in units if u["a09_1_max"] is not None]
        r1.value = max(mx) if mx else None
        r1.n = f"{len(units)} audited model(s)"
        r2.value = "; ".join(f"{u['model']}/{u['variant']}: tensors {fmt(u['a09_2_tensors'])}, flip dense "
                             f"{u['flip']['dense'][0]}, even/odd {u['flip']['even_odd'][0]}" for u in units)
        r2.n = r1.n
        r3.value = "; ".join(f"{u['model']}/{u['variant']}: bf16 flip dense {fmt(u.get('flip_bf16_dense'), '.3f')}, "
                             f"even/odd {fmt(u.get('flip_bf16_even_odd'), '.3f')}" for u in units)
        r3.n = r1.n
        if r1.value is not None:
            c.num("a09.max_rel_error_fp64", r1.value, desc="A09-1: largest relative equivariance error in float64, any "
                  "tensor, any audited model", sources=r1.sources)
        for u in units:
            for form in ("dense", "even_odd"):
                c.num(f"a09.{u['model']}.{u['variant']}.flip_bf16_{form}", u.get(f"flip_bf16_{form}"),
                      desc=f"A09-3: bf16 flip consistency, {form} form", sources=u["sources"], unit="%")
    for r in (r1, r2, r3):
        r.sources = dedupe_sources(r.sources)


# ---------------------------------------------------------------- A10 and A11
def fullbank_rows(c: Final) -> list:
    """Rows of every fp_fullbank summary.json (runs/final/A10, results/A10, runs/final/A11, results/A11, searched),
    each with its source and its arm key (group, model, folder, seed) when it is a run."""
    out, seen = [], set()
    for base in (c.p("runs", "final", "A10"), c.p("results", "A10"), c.p("runs", "final", "A11"), c.p("results", "A11"),
                 c.p("runs", "final", "a10"), c.p("runs", "final", "a11")):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("summary.json")):
            if p in seen:
                continue
            seen.add(p)
            rec = c.jload(p)
            rows = rec.get("rows") if isinstance(rec, dict) else None
            if not isinstance(rows, list) or not rows or not any(isinstance(x, dict) and "fullbank_knn_top1" in x
                                                                 for x in rows):
                continue
            s = c.src(p)
            for x in rows:
                if isinstance(x, dict):
                    out.append({**x, "_key": _run_key(x.get("run_dir")), "_source": s})
    return out


def score_a10_a11(c: Final) -> None:
    r = c.add("A10", "|main Mirror arm's 3-seed mean - original| on the full-bank k-NN (1,281,167 train images) and on "
                     "the DINOv2 linear protocol (validation top-1 of the classifier chosen on the held-out 10%)")
    r11 = c.add("A11", "extended healing (about 3x the images): Mirror and SVD-LLM k-NN, full-bank k-NN and linear "
                       "protocol, against the original (target: Mirror within 0.2)")
    rows = fullbank_rows(c)
    v, how = c.main_variant()
    orig = [x for x in rows if not x["_key"] and str(x.get("model")) == MAIN and not truthy(x.get("flip_average"))]
    if v is None:
        r.reason = how
    elif not rows:
        r.verdict, r.reason = missing_result(c, ["runs/final/A10", "results/A10"],
                                             "the full-bank summary (runs/final/A10/**/summary.json)")
    else:
        mf = f"mirror_exact_{v}"
        parts = []
        for col, label in (("fullbank_knn_top1", "full-bank k-NN"), ("linear_val_top1", "linear protocol")):
            mv = {}
            for x in rows:
                k = x["_key"]
                if k and k[:3] == ("main", MAIN, mf) and k[3] in SEEDS3 and fnum(x.get(col)) is not None \
                        and not truthy(x.get("flip_average")):
                    mv.setdefault(k[3], (fnum(x[col]), x["_source"]))
            ov = next(((fnum(x.get(col)), x["_source"]) for x in orig if fnum(x.get(col)) is not None), None)
            for s in SEEDS3:
                r.details.append({"metric": label, "row": f"Mirror s{s}", "value": mv.get(s, (None,))[0]})
            r.details.append({"metric": label, "row": "original", "value": None if ov is None else ov[0]})
            if len(mv) == 3 and ov is not None:
                mean = float(np.mean([mv[s][0] for s in SEEDS3]))
                d = mean - ov[0]
                parts.append(within(mean, ov[0], A10_TOL))
                r.sources += [mv[s][1] for s in SEEDS3] + [ov[1]]
                c.num(f"a10.{col}.mirror_mean", mean, desc=f"A10: {label}, main Mirror arm, mean of 3 seeds",
                      sources=[mv[s][1] for s in SEEDS3], n=3, unit="%")
                c.num(f"a10.{col}.original", ov[0], desc=f"A10: {label}, original", sources=[ov[1]], unit="%")
                r.value = (r.value + "; " if r.value else "") + f"{label} {d:+.3f}"
            else:
                parts.append(PENDING)
                r.reason = (r.reason + "; " if r.reason else "") + (
                    f"{label}: Mirror seeds {sorted(mv)} of 0 to 2, original {'present' if ov else 'missing'}")
        r.verdict = combine(parts)
        r.n = "3 Mirror seeds against the original, two metrics"
        r.sources = dedupe_sources(r.sources)
    # A11 (report)
    mk, sk = c.final_eval(A11_MIRROR, 0, steps=None), c.final_eval(A11_SVDLLM, 0, steps=None)
    if mk.ok or sk.ok:
        r11.verdict = REPORTED
        ref = None
        for x in (mk, sk):
            if x.ok:
                o = ((x.info.get("eval") or {}).get("original_reference") or {}).get("bf16") or {}
                ref = ref if ref is not None else fnum(o.get("top1"))
        for name, x in (("Mirror", mk), ("SVD-LLM", sk)):
            row = {"arm": name, "knn_top1": x.value, "steps": (x.info.get("metrics") or {}).get("total_steps")
                   if x.ok else None, "note": x.reason if not x.ok else ""}
            for fb in rows:
                if x.ok and fb["_key"] and fb["_key"][:3] == ("a11", MAIN, x.info.get("run_dir").parent.name) \
                        and fb["_key"][3] == 0:
                    row.update(fullbank=fnum(fb.get("fullbank_knn_top1")), linear=fnum(fb.get("linear_val_top1")))
                    r11.sources.append(fb["_source"])
            r11.details.append(row)
            r11.sources += x.sources
        r11.details.append({"arm": "original (same evaluation)", "knn_top1": ref})
        if mk.ok and ref is not None:
            d = mk.value - ref
            r11.value = f"Mirror minus original {d:+.3f} (target within {A11_TARGET}: {'met' if abs(d) <= A11_TARGET + EPS else 'not met'})"
            c.num("a11.mirror_minus_original", d, desc="A11: extended-healing Mirror minus the original, k-NN",
                  sources=mk.sources, unit="points")
        r11.n = "1 heal per arm"
        if not (mk.ok and sk.ok):
            r11.reason = "; ".join(x.reason for x in (mk, sk) if not x.ok)
    else:
        r11.verdict = NOT_RUN if (mk.state == "not_run" and sk.state == "not_run") else (
            NOT_RUN if mk.state == "not_run" else PENDING)
        r11.reason = "; ".join(x.reason for x in (mk, sk) if x.reason)
    r11.sources = dedupe_sources(r11.sources)


# ---------------------------------------------------------------- A12b, A15a, A15b
def score_a12b(c: Final) -> None:
    r = c.add("A12b", "throughput of the fused even/odd kernel over the original's at batch 64 and 256 (bf16, "
                      "compiled, same device); >= 1.1 at both earns a speed claim, otherwise a limitation")
    p = c.p("results", "A12b", "metrics.json")
    m = c.jload(p)
    s64 = s256 = None
    src = None
    if isinstance(m, dict):
        s64, s256 = fnum(m.get("speed_vs_original_b64")), fnum(m.get("speed_vs_original_b256"))
        src = c.src(p)
    if s64 is None or s256 is None:
        ents, a12src = a12_entries(c)
        fused = [e for n, e in ents.items() if "fused" in n.lower() or "fused" in str(e.get("kind")).lower()]
        if fused:
            s64, s256 = fnum(fused[0].get("speed_vs_original_b64")), fnum(fused[0].get("speed_vs_original_b256"))
            src = a12src
    if s64 is None or s256 is None:
        r.verdict, r.reason = missing_result(c, "results/A12b", "the fused-kernel timing (results/A12b/metrics.json or "
                                                                f"a 'fused' entry of results/A12/{MAIN}/systems_table.csv)")
        return
    r.sources = [src]
    r.verdict = combine([cmp(s64, ">=", A12B_SPEED), cmp(s256, ">=", A12B_SPEED)])
    r.value = f"batch 64: {s64:.3f}x; batch 256: {s256:.3f}x"
    r.n = "2 batch sizes"
    r.reason = "speed claim earned" if r.verdict == PASS else "reported as a limitation (no speed claim)"
    c.num("a12b.speed_b64", s64, desc="A12b: fused kernel throughput over the original's, batch 64", sources=[src])
    c.num("a12b.speed_b256", s256, desc="A12b: fused kernel throughput over the original's, batch 256", sources=[src])


def score_a15(c: Final) -> None:
    ra = c.add("A15a", "SigLIP 2 B/16 zero-shot k-NN with the whitened projection against the plain projection (the "
                       "CORE-04 collapse), per size")
    rb = c.add("A15b", "R90 ratio to null in blocks 1 to 6 after the rot90 fine-tune minus before <= -0.1")
    p = c.p("results", "A15a", "metrics.json")
    m = c.jload(p)
    comp = (m or {}).get("comparison") if isinstance(m, dict) else None
    if isinstance(comp, dict) and comp:
        s = c.src(p)
        ra.sources = [s]
        ra.verdict = REPORTED
        for size, x in comp.items():
            ra.details.append({"size": size, "plain": fnum(x.get("plain_val_top1")),
                               "whitened": fnum(x.get("whitened_val_top1")), "difference": fnum(x.get("difference")),
                               "original": fnum(x.get("original_val_top1")),
                               "core04_collapse": fnum(x.get("core04_collapse_val_top1"))})
            c.num(f"a15a.{size}.whitened", x.get("whitened_val_top1"), desc=f"A15a: SigLIP 2 whitened projection, "
                  f"zero-shot k-NN, {size}", sources=s, unit="%")
            c.num(f"a15a.{size}.plain", x.get("plain_val_top1"), desc=f"A15a: SigLIP 2 plain projection, zero-shot "
                  f"k-NN, {size}", sources=s, unit="%")
        ra.value = "; ".join(f"{d['size']}: whitened {fmt(d['whitened'])} vs plain {fmt(d['plain'])}"
                             for d in ra.details)
        ra.n = f"{len(comp)} size(s)"
    else:
        ra.verdict, ra.reason = missing_result(c, "results/A15a", "results/A15a/metrics.json")
    p = c.p("results", "A15b", "metrics.json")
    m = c.jload(p)
    if isinstance(m, dict):
        before = fnum((m.get("original") or {}).get("ratio_to_null"))
        after = fnum((m.get("finetuned_numbers") or {}).get("ratio_to_null"))
        s = c.src(p)
        rb.sources = [s]
        if before is None or after is None:
            rb.reason = "ratio to null missing in results/A15b/metrics.json"
        else:
            d = after - before
            rb.verdict, rb.value, rb.n = cmp(d, "<=", -A15B_DROP), d, "1 fine-tune"
            rb.details = [{"model": "original", "ratio_to_null_1_6": before},
                          {"model": "rot90 fine-tune", "ratio_to_null_1_6": after}]
            rb.reason = "reported either way (PREREG Section 1)"
            rec = ((m.get("scores") or {}).get("A15b") or {}).get("status")
            if rec in (PASS, FAIL) and rec != rb.verdict:
                rb.reason += f"; the producer's score says {rec}"
            c.num("a15b.ratio_change", d, desc="A15b: R90 ratio to null (blocks 1 to 6) after minus before the "
                  "fine-tune", sources=s)
    else:
        rb.verdict, rb.reason = missing_result(c, "results/A15b", "results/A15b/metrics.json")


# ---------------------------------------------------------------- P11.2 (previous phase, re-measured)
P112_DIR = ("runs", "final", "A10", "p112")      # scripts/tier2_ade20k.py --out runs/final/A10/p112 (the re-measurement)


def ade20k_files(c: Final) -> list:
    """The re-measurement's t2_ade20k.csv (runs/final/A10/p112) first, then any other t2_ade20k.csv under results/ and
    runs/final/A10/ (a copy pulled to the laptop, for example)."""
    out = [c.p(*P112_DIR, "t2_ade20k.csv")]
    for base in (c.p("results"), c.p("runs", "final", "A10")):
        if base.is_dir():
            out += sorted(base.rglob("t2_ade20k.csv"))
    return [p for p in dict.fromkeys(out) if p.is_file()]


def _maskeq_sym(c: Final, csv_path: Path) -> tuple:
    """(result.sym of seg/<M>/vexact_s0_maskeq.json beside the csv, source) or (None, None)."""
    p = csv_path.parent / "seg" / MAIN / "vexact_s0_maskeq.json"
    rec = c.jload(p)
    sym = ((rec or {}).get("result") or {}).get("sym") if isinstance(rec, dict) else None
    return (sym, c.src(p)) if isinstance(sym, dict) else (None, None)


def score_p11_2(c: Final) -> None:
    r = c.add("P11.2", "share of pixels with mask(mirror x) = mirror(mask x), fp32, ADE20K validation: Mirror-exact "
                       "(healed seed 0 of MUST-09, symmetrised head) with the near-tie rule read in both copies; the "
                       "original below 100% with every recorded head")
    cands = []
    for p in ade20k_files(c):
        df = c.cload(p)
        if df is None or "variant" not in df:
            continue
        sub = df[df["variant"].astype(str) == "vexact_s0"]
        if "model" in sub:
            sub = sub[sub["model"].astype(str) == MAIN]
        if not len(sub):
            continue
        sym, ssrc = _maskeq_sym(c, p)
        has_csv = "mask_eq_sym_max_rel_margin" in df and (
            fnum(sub.iloc[-1].get("mask_eq_sym_max_rel_margin")) is not None
            or fnum(sub.iloc[-1].get("mask_eq_sym_n_differ_not_near_tie")) is not None
            or fnum(sub.iloc[-1].get("mask_eq_sym_n_differ")) == 0)
        has_json = sym is not None and "max_rel_margin_of_differing_pixels" in sym
        if has_csv or has_json:
            cands.append((p, df, sub.iloc[-1], sym if has_json else None, ssrc if has_json else None))
    if not cands:
        r.verdict, r.reason = missing_result(c, ["/".join(P112_DIR)], (
            f"the re-measured ADE20K result with top-two margins ({'/'.join(P112_DIR)}/t2_ade20k.csv and "
            f"seg/{MAIN}/vexact_s0_maskeq.json)"))
        return
    cands.sort(key=lambda t: ("/".join(P112_DIR) in t[0].as_posix(), "A10" in t[0].as_posix(),
                              str(t[2].get("date") or "")))
    p, df, row, sym, ssrc = cands[-1]
    g = str(row.get("git_hash")) if "git_hash" in row and str(row.get("git_hash")) not in ("nan", "") else None
    s = c.src(p, g)
    r.sources = [s] + ([ssrc] if ssrc else [])
    share, nd = fnum(row.get("mask_eq_sym")), fnum(row.get("mask_eq_sym_n_differ"))
    rec = {"flip_consistency": share, "n_disagree": None if nd is None else int(nd),
           "max_margin_rel": fnum(row.get("mask_eq_sym_max_rel_margin")),
           "n_not_tie": fnum(row.get("mask_eq_sym_n_differ_not_near_tie"))}
    if sym is not None:                       # the JSON holds the same fields for the symmetrised head
        rec["flip_consistency"] = share = share if share is not None else fnum(sym.get("share"))
        if rec["n_disagree"] is None and fnum(sym.get("n_differ")) is not None:
            rec["n_disagree"] = int(sym["n_differ"])
        if rec["max_margin_rel"] is None:
            rec["max_margin_rel"] = fnum(sym.get("max_rel_margin_of_differing_pixels"))
        if rec["n_not_tie"] is None:
            rec["n_not_tie"] = fnum(sym.get("n_differ_not_near_tie"))
    if rec["n_not_tie"] is not None:
        rec["n_not_tie"] = int(rec["n_not_tie"])
    v1, note = exact_flip(rec, 0, strict_key="max_margin_rel", not_tie_key="n_not_tie")
    prec = str(row.get("mask_eq_precision") or "")
    if prec and "fp32" not in prec:
        v1, note = PENDING, note + f"; precision recorded as {prec!r}, not fp32"
    orow = df[df["variant"].astype(str) == "original"]
    if "model" in orow:
        orow = orow[orow["model"].astype(str) == MAIN]
    osrc = s
    if not len(orow):
        op = c.p("results", "MUST-11", "t2_seg", "t2_ade20k.csv")
        odf = c.cload(op)
        if odf is not None and "variant" in odf:
            orow = odf[(odf["variant"].astype(str) == "original")
                       & ((odf["model"].astype(str) == MAIN) if "model" in odf else True)]
            if len(orow):
                og = str(orow["git_hash"].iloc[0]) if "git_hash" in orow else None
                osrc = c.src(op, og)
                r.sources.append(osrc)
    oshares = []
    if len(orow):
        o = orow.iloc[-1]
        oshares = [fnum(o.get(k)) for k in ("mask_eq_raw", "mask_eq_sym") if fnum(o.get(k)) is not None]
    v2 = (PASS if all(x < 100.0 - EPS for x in oshares) else FAIL) if oshares else PENDING
    r.verdict = combine([v1, v2])
    r.value = (f"Mirror-exact (symmetrised head) {fmt(share, '.5f')}%: {note}; original "
               + (", ".join(f"{x:.4f}%" for x in oshares) if oshares else "n/a"))
    r.n = f"{int(row['mask_eq_n_pixels']):,} pixels" if fnum(row.get("mask_eq_n_pixels")) is not None else None
    r.details = [{"model": "vexact_s0", "mask_eq_sym": share, "n_differ": rec["n_disagree"],
                  "max_rel_margin_both_copies": rec["max_margin_rel"], "n_not_near_tie": rec["n_not_tie"],
                  "verdict": v1, "file": c.rel(p)},
                 {"model": "original", "shares": oshares, "verdict": v2, "file": osrc["file"]}]
    if v2 == PENDING:
        r.reason = "the original's mask equivariance is not recorded"
    c.num("p11_2.vexact_s0.mask_eq_sym", share, desc="P11.2 (re-measured): mask equivariance, healed Mirror-exact "
          "seed 0, symmetrised head, fp32", sources=[s], unit="%")
    c.num("p11_2.vexact_s0.max_rel_margin", rec["max_margin_rel"], desc="P11.2: largest relative top-two margin of "
          "the differing pixels (larger of the two copies)", sources=[s])


SCORERS = (score_gate_p, score_a01, score_a02_structure, score_a02_heals, score_a03, score_a04, score_a05, score_a06,
           score_a07, score_a08, score_a09, score_a10_a11, score_a12b, score_a15, score_p11_2)
RULE_ORDER = ("Gate P", "A01", "A02-1", "A02-2", "A02-3", "A02-4", "A02-5", "A02-6", "A03-1", "A03-2", "A04-1",
              "A04-2", "A05-1", "A06-1", "A06-2", "A06-3", "A07", "A08-1", "A08-2", "A09-1", "A09-2", "A09-3", "A10",
              "A11", "A12b", "A15a", "A15b", "P11.2")


# ================================================================================================
# Deliverable tables (FINAL_PHASE_PLAN.md Section 10)
# ================================================================================================
def _src_cols(srcs) -> dict:
    s = dedupe_sources(srcs)
    return {"source_file": "; ".join(x["file"] for x in s),
            "git_hash": "; ".join(sorted({str(x["git_hash"]) for x in s if x["git_hash"]}))}


def _row_from_vals(name: str, folder: str, role: str, vals: dict, extra: dict | None = None) -> dict:
    ok = {s: x for s, x in vals.items() if x.ok}
    mu, sd, n = mean_std([x.value for x in ok.values()])
    row = {"arm": name, "folder": folder, "role": role, "seeds": ",".join(str(s) for s in sorted(ok)), "n": n}
    for s in SEEDS3:
        row[f"s{s}"] = vals[s].value if s in vals and vals[s].ok else None
    row.update(mean=mu, sd=sd)
    f32, params, frac, lr = [], None, None, None
    for x in ok.values():
        rec = (x.info.get("eval") or {}).get("passes", {}).get("fp32") or {}
        if fnum(rec.get("flip_consistency")) is not None:
            f32.append(fnum(rec["flip_consistency"]))
        m = x.info.get("metrics") or {}
        pr = m.get("params") or {}
        params = params or pr.get("model_params_structured") or pr.get("params_model")
        frac = frac or fnum(pr.get("model_params_frac")) or fnum(pr.get("params_frac"))
        lr = lr if lr is not None else m.get("lr")
    row.update(flip_fp32_min=min(f32) if f32 else None, params_model=params, params_frac=frac, lr=lr)
    row.update(extra or {})
    row.update(_src_cols([s_ for x in ok.values() for s_ in x.sources]))
    return row


def build_tables(c: Final) -> None:
    v, _ = c.main_variant()
    t = {}
    # T1: main healed comparison
    rows = []
    a05 = c.tables.pop("_a05", None)
    mir_runs = a05["mirror"] if a05 else (a05_mirror_runs(c, v) if v else [])
    mir_ok = bool(mir_runs) and all(x.ok for _, _, x in mir_runs)
    mp = mirror_point(mir_runs) if mir_ok else None
    for name, folder, role, seeds in (
            ("Mirror, A*_w", "mirror_exact_wspace", "main arm" if v == "wspace" else "A01 ablation", (0, 1, 2)),
            ("Mirror, current A*", "mirror_exact_astar", "main arm" if v == "astar" else "Gate P", (0, 1, 2))):
        vals = {s: c.final_eval(arm("main", MAIN, folder), s) for s in seeds}
        rows.append(_row_from_vals(name, folder, role, vals))
    for fit, a in a02_heal_arms().items():
        rows.append(_row_from_vals(f"Mirror from {fit}", a.folders[0], "A02 heal", {0: c.final_eval(a, 0)}))
    for m, seeds, label in A05_BASELINES:
        vals = {s: c.final_eval(arm("main", MAIN, m), s) for s in seeds}
        extra = {}
        ok = [(0, s, x) for s, x in vals.items() if x.ok]
        if mir_ok and ok:
            extra["mirror_minus_arm"] = mp - float(np.mean([x.value for _, _, x in ok]))
            if c.t1_intervals and all(x.ok or x.state == "not_run" for x in vals.values()):
                try:
                    res = boot_pair(c, "mirror", mir_runs, m, ok)
                    pr = res["pairs"][0]
                    extra.update(ci_low=pr["ci_low"], ci_high=pr["ci_high"], p_le_0=pr["p_le_0"],
                                 effect_size=pr["effect_size"])
                except (FileNotFoundError, KeyError, ValueError) as e:
                    extra["interval_note"] = str(e)[:120]
        rows.append(_row_from_vals(label, m, "baseline (A05)", vals, extra))
    for label, folders, role in (("Random structure", ("random_structure",), "control (A04)"),
                                 ("H1: true operator, random pairing", H1_ARM.folders, "control (A03)"),
                                 ("Width-pruned dense student", ("flap",), "dense student (A04)"),
                                 ("Block-dropped dense student", ("blockdrop",), "dense student (A04)"),
                                 ("Original distilled", ("original",), "healing without surgery (A04)")):
        rows.append(_row_from_vals(label, folders[0], role, {0: c.final_eval(arm("main", MAIN, *folders), 0)}))
    t["T1_main_comparison"] = (pd.DataFrame(rows), "T1: main healed comparison, DINOv2-B/14 reg (final bf16 k-NN, "
                                                   "50,000 validation queries; interval: A13 bootstrap of Mirror "
                                                   "minus the arm, 10,000 resamples, seed 0)")
    # T2: stage ablation
    mm = c.main_mean()
    ma = c.main_arm()
    rows = []
    ref = None
    if ma is not None:
        x0 = c.final_eval(ma, 0)
        if x0.ok:
            o = ((x0.info.get("eval") or {}).get("original_reference") or {}).get("bf16") or {}
            ref = (fnum(o.get("top1")), x0.sources[-1:])
    z = c.step_eval(ma, 0, 0) if ma is not None else Val()
    stage = [("Original", "no", "no", "no", Val(ref[0], "ok", sources=ref[1]) if ref and ref[0] is not None else Val()),
             ("Original distilled (healing without surgery)", "no", "no", "yes",
              c.final_eval(arm("main", MAIN, "original"), 0)),
             ("Surgery, before healing", "yes", "yes", "no", z),
             ("Surgery + healing (main arm, 3-seed mean)", "yes", "yes", "yes", mm),
             ("Random structure + healing", "random", "yes", "yes",
              c.final_eval(arm("main", MAIN, "random_structure"), 0)),
             ("True operator + random pairing + healing (H1)", "partial", "yes", "yes", c.final_eval(H1_ARM, 0))]
    for m, seeds, label in A05_BASELINES:
        stage.append((f"Compression without symmetry + healing: {label}", "no", "yes", "yes",
                      mean_val([c.final_eval(arm("main", MAIN, m), s) for s in seeds])))
    stage += [("Width-pruned dense student + healing", "no", "yes", "yes", c.final_eval(arm("main", MAIN, "flap"), 0)),
              ("Block-dropped dense student + healing", "no", "yes", "yes",
               c.final_eval(arm("main", MAIN, "blockdrop"), 0))]
    for name, sym, comp, healed, x in stage:
        frac = None
        m_ = x.info.get("metrics") if x.ok else None
        if m_:
            pr = m_.get("params") or {}
            frac = fnum(pr.get("model_params_frac")) or fnum(pr.get("params_frac"))
        rows.append({"row": name, "symmetry": sym, "compression": comp, "healed": healed, "knn_top1": x.value,
                     "minus_main": (x.value - mm.value) if (x.ok and mm.ok) else None, "params_frac": frac,
                     "note": x.reason if not x.ok else "", **_src_cols(x.sources)})
    t["T2_stage_ablation"] = (pd.DataFrame(rows), "T2: stage ablation (A04), DINOv2-B/14 reg, final bf16 k-NN")
    # T3: other models
    rows = []
    mf = ("mirror_exact",)
    for model, methods in ((DEIT, (("Mirror", mf, SEEDS3), ("SVD-LLM", ("svdllm",), SEEDS3),
                                   ("Random structure", ("random_structure",), (0,)))),
                           (LARGE, (("Mirror", mf, (0,)), ("SVD-LLM", ("svdllm",), (0,))))):
        for label, folders, seeds in methods:
            a = arm("a08", model, *folders)
            kv = {s: c.final_eval(a, s) for s in seeds}
            row = _row_from_vals(label, folders[0], model, kv)
            if model == DEIT:
                tops = [classifier_eval(c, a, s)[0] for s in seeds]
                tv = [x.value for x in tops if x.ok]
                row["classifier_top1_mean"] = float(np.mean(tv)) if tv else None
                row["classifier_n"] = len(tv)
            row["model"] = model
            rows.append(row)
    ot = released_top1(c, DEIT)
    if ot.ok:
        rows.append({"arm": "Original (released)", "folder": "_original", "role": DEIT, "model": DEIT,
                     "classifier_top1_mean": ot.value, "classifier_n": 1, **_src_cols(ot.sources)})
    t["T3_other_models"] = (pd.DataFrame(rows), "T3: other models (A08): DeiT-III-B/16 (classifier top-1 and k-NN) "
                                                "and DINOv2-L/14 reg (k-NN, indicative)")
    # T4: operator stability
    p, m = _a02(c)
    if m is not None:
        s = c.src(p)
        heals = {}
        for fit, a in ([("fit0", c.main_arm())] if v else []) + list(a02_heal_arms().items()):
            x = c.final_eval(a, 0)
            heals[fit] = x.value if x.ok else None
        rows = []
        for f in m.get("fits") or []:
            if isinstance(f, dict):
                rows.append({"fit": f.get("name"), "group": f.get("group"), "d_plus": f.get("d_plus"),
                             "d_minus": f.get("d_minus"), "pairs_total": f.get("n_pairs_total"),
                             "strong_pairs_1_6": f.get("n_strong_blocks_1_6"),
                             "twin_fraction_1_6": f.get("twin_fraction_blocks_1_6"), "head_pairs": f.get("n_head_pairs"),
                             "zero_shot_knn": f.get("knn_top1"), "healed_knn_s0": heals.get(str(f.get("name"))),
                             "source_file": s["file"], "git_hash": s["git_hash"]})
        t["T4_operator_stability"] = (pd.DataFrame(rows), "T4: operator stability (A02), per fit")
        rows = []
        for q in m.get("pairs") or []:
            if isinstance(q, dict):
                rows.append({k: q.get(k) for k in ("a", "b", "operator_distance", "rho_ab", "rho_ba",
                                                   "weighted_agreement", "unweighted_cos2", "twin_mean", "twin_jaccard",
                                                   "head_share_same", "B_rel_mean", "C_rel_mean")}
                            | {"source_file": s["file"], "git_hash": s["git_hash"]})
        t["T4_operator_stability_pairs"] = (pd.DataFrame(rows), "T4 (continued): agreement between pairs of fits")
    # T5: discovery controls
    p = c.p("results", "A03", "metrics.json")
    m = c.jload(p)
    df = c.cload(c.p("results", "A03", "controls.csv"))
    rows = []
    if df is not None:
        s = c.src(c.p("results", "A03", "controls.csv"))
        for _, x in df.iterrows():
            rows.append({"control": x.get("control"), "label": x.get("label"), "d_minus": x.get("d_minus_span"),
                         "twin_fraction_1_6": fnum(x.get("twin_fraction_t50_1-6")),
                         "ratio_in_sample_1_6": fnum(x.get("ratio_to_null_1-6")),
                         "ratio_split_half_1_6": fnum(x.get("ratio_to_null_split_1-6")),
                         "spectrum_abs_sum": fnum(x.get("span_abs_sum")), "odd_share": fnum(x.get("span_neg_mass_share")),
                         "zero_shot_knn": fnum(x.get("knn_top1")), "source_file": s["file"], "git_hash": s["git_hash"]})
    h1 = c.final_eval(H1_ARM, 0)
    if h1.ok:
        rows.append({"control": "H1", "label": "true operator, random pairing, healed (seed 0)",
                     "healed_knn": h1.value, **_src_cols(h1.sources)})
    if rows:
        t["T5_discovery_controls"] = (pd.DataFrame(rows), "T5: discovery controls (A03); A03-1 scores the split-half "
                                                          "ratio, the in-sample ratio is shown beside it")
    # T6: systems
    rows = []
    a12 = c.p("results", "A12")
    if a12.is_dir():
        for d in sorted(a12.iterdir()):
            df = c.cload(d / "systems_table.csv")
            if df is not None:
                s = c.src(d / "systems_table.csv")
                for _, x in df.iterrows():
                    rows.append({"model": d.name, **{k: x.get(k) for k in df.columns}, "source_file": s["file"],
                                 "git_hash": s["git_hash"]})
    if rows:
        t["T6_systems"] = (pd.DataFrame(rows), "T6: systems (A12): parameters, bf16 memory, analytic GFLOPs, "
                                               "throughput and latency (bf16, torch.compile)")
    # T7: exactness audit
    rows = []
    for model in (MAIN, DEIT):
        for d, m in a09_variants(c, model):
            u = a09_unit(c, d, m)
            rows.append({"model": model, "variant": d.name, "a09_1_max_rel_error_fp64": u["a09_1_max"],
                         "a09_2_tensors_pass": u["a09_2_tensors"], "a09_2_tensors_fail": u["a09_2_n_fail"],
                         "flip_fp32_dense": u.get("flip_fp32_dense"), "flip_fp32_even_odd": u.get("flip_fp32_even_odd"),
                         "flip_bf16_dense": u.get("flip_bf16_dense"), "flip_bf16_even_odd": u.get("flip_bf16_even_odd"),
                         **_src_cols(u["sources"])})
    if rows:
        t["T7_exactness_audit"] = (pd.DataFrame(rows), "T7: exactness audit (A09): E1 float64, E2 float32 against "
                                                       "the model's own rounding, flip consistency")
    # T8: standard metrics
    rows = []
    for x in fullbank_rows(c):
        k = x["_key"]
        rows.append({"name": x.get("name"), "model": x.get("model"), "group": k[0] if k else None,
                     "folder": k[2] if k else "(released model)", "seed": k[3] if k else None,
                     "flip_average": x.get("flip_average"), "fullbank_knn_top1": fnum(x.get("fullbank_knn_top1")),
                     "linear_val_top1": fnum(x.get("linear_val_top1")),
                     "linear_val_top1_dinov2_rule": fnum(x.get("linear_val_top1_dinov2_rule")),
                     "linear_chosen": x.get("linear_chosen"), "source_file": x["_source"]["file"],
                     "git_hash": x["_source"]["git_hash"]})
    if rows:
        t["T8_standard_metrics"] = (pd.DataFrame(rows), "T8: standard metrics (A10, A11): full-bank k-NN and the DINOv2 "
                                                        "linear protocol (the DINOv2 selection rule is shown for "
                                                        "comparison only)")
    c.tables.update(t)


# ================================================================================================
# Writing
# ================================================================================================
def md_table(headers, rows) -> list:
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return out + [""]


def src_txt(sources, limit: int = 3) -> str:
    s = dedupe_sources(sources)
    if not s:
        return "n/a"
    parts = [f"`{x['file']}` ({short(x['git_hash'])})" for x in s]
    more = f" and {len(parts) - limit} more" if len(parts) > limit else ""
    return "; ".join(parts[:limit]) + more


def _cell(v) -> str:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(jsonable(v))[:70]
    if isinstance(v, str):
        return v[:160]
    return fmt(v)


def _detail_table(recs: list) -> list:
    cols = []
    for d in recs:
        for k in d:
            if k not in cols and not str(k).startswith("_"):
                cols.append(k)
    cols = cols[:12]
    out = md_table(cols, [[_cell(d.get(k)) for k in cols] for d in recs[:40]])
    if len(recs) > 40:
        out += [f"({len(recs) - 40} more rows in results/final/numbers.json)", ""]
    return out


def deviations(c: Final) -> tuple:
    """(rows of notes/decisions.md added after the pre-registration commit, how they were found)."""
    p = c.p(DECISIONS_FILE)
    if not p.is_file():
        return [], f"{DECISIONS_FILE} not found"
    c.seen(p)
    rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if re.match(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|", ln)]
    commit = c.prereg.get("commit")
    if commit:
        old = git(c.root, "show", f"{commit}:{DECISIONS_FILE}")
        if old is None and c.root != REPO_ROOT:
            old = git(REPO_ROOT, "show", f"{commit}:{DECISIONS_FILE}")
        if old is not None:
            before = {norm_ws(x) for x in old.splitlines()}
            return [x for x in rows if norm_ws(x) not in before], f"rows absent at the pre-registration commit {short(commit)}"
    day = str(c.prereg.get("recorded_utc") or "")[:10]
    if day:
        return [x for x in rows if x.split("|")[1].strip() >= day], f"rows dated on or after {day} (git not available)"
    return [], "the pre-registration commit is not recorded yet, so no row can be placed after it"


def rule_dict(r: Rule) -> dict:
    d = asdict(r)
    d["scored"] = r.scored
    return jsonable(d)


def write_tables(c: Final, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, item in sorted(c.tables.items()):
        if name.startswith("_") or not isinstance(item, tuple) or not isinstance(item[0], pd.DataFrame):
            continue
        df, title = item
        df = df.copy()
        for col in df.columns:
            if df[col].map(lambda x: isinstance(x, (list, tuple, dict))).any():
                df[col] = df[col].map(lambda x: json.dumps(jsonable(x)) if isinstance(x, (list, tuple, dict)) else x)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        cols = [x for x in df.columns if x not in ("source_file", "git_hash")]
        lines = [f"# {title}", "", f"Written by `scripts/collect_final_phase.py`; every row's source file and commit "
                 f"are in `{name}.csv` (columns source_file and git_hash).", ""]
        lines += md_table(cols, [[_cell(x) for x in row] for row in df[cols].itertuples(index=False)]) if len(df) \
            else ["No rows yet.", ""]
        (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files += [out_dir / f"{name}.csv", out_dir / f"{name}.md"]
    return files


def write_outputs(c: Final, report: Path, numbers_json: Path, paper_numbers: Path, tables_dir: Path) -> dict:
    me = git_self()
    order = {k: i for i, k in enumerate(RULE_ORDER)}
    c.rules.sort(key=lambda r: order.get(r.id, 999))
    for r in c.rules:
        r.sources = dedupe_sources(r.sources)
    scored = [r for r in c.rules if r.scored]
    counts = {k: sum(1 for r in scored if r.verdict == k) for k in (PASS, FAIL, PENDING, NOT_RUN)}
    counts["scored"] = len(scored)
    counts[REPORTED] = sum(1 for r in c.rules if not r.scored and r.verdict == REPORTED)
    counts["report_only"] = sum(1 for r in c.rules if not r.scored)
    devs, how = deviations(c)
    tfiles = write_tables(c, tables_dir)
    pre = c.prereg
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    L = ["# FINAL_PHASE_REPORT: pre-registered rules of the final phase, scored", "",
         f"Written by `scripts/collect_final_phase.py` (`mirror/final/collect.py`) on {now}, code commit "
         f"`{short(me.get('git_hash'))}`{' (uncommitted changes)' if me.get('git_dirty') else ''}. Every number below "
         "is read from a result file written by a script of this repository and is listed with that file and the "
         "commit recorded in the run's meta.json. Nothing here is typed by hand; rerun the collector when new results "
         "arrive.", "",
         f"Pre-registration: `{PREREG_FILE}`, commit `{short(pre.get('commit'))}`, sha256 of the current text "
         f"`{(pre.get('sha256') or 'n/a')[:12]}`: {pre.get('status')}.", "",
         f"Scored rules: {counts['scored']}; PASS {counts[PASS]}, FAIL {counts[FAIL]}, PENDING {counts[PENDING]}, "
         f"NOT RUN {counts[NOT_RUN]}. Report-only rules: {counts['report_only']} ({counts[REPORTED]} reported).", ""]
    if pre.get("unchanged") is False:
        L += [f"**Warning:** {PREREG_FILE} has changed since its recorded pre-registration commit.", ""]
    bad = [r.id for r in c.rules if r.quote_found is False]
    if bad:
        L += [f"**Warning:** no row for {', '.join(bad)} in Section 1 of {PREREG_FILE}.", ""]
    tok = [f"{r.id} ({', '.join(r.bar_tokens_missing)})" for r in c.rules if r.bar_tokens_missing]
    if tok:
        L += ["**Warning:** the quoted bar text lacks the numbers the collector uses: " + "; ".join(tok) + ".", ""]
    v, how_main = c.main_variant()
    L += [f"Main Mirror arm: `{('main/' + MAIN + '/mirror_exact_' + v) if v else 'not decided'}` (from {how_main}).",
          ""]
    nr = c.cuts.not_run_jobs()
    if nr:
        L += ["Jobs not run (runner state " + f"`{c.cuts.state_dir}`): "
              + "; ".join(f"{j} ({why or st})" for j, st, why in nr) + ".", ""]
    L += ["## Rules", "",
          "PENDING means an input is missing or incomplete (the reason is given). NOT RUN means an input was cut by "
          "the Section 9 order or the budget governor; it is never counted as passed. Failed rules are failures and "
          "are reported as such.", ""]
    L += md_table(["ID", "Rule", "Value", "Bar", "n", "Verdict", "Sources (commit)"],
                  [[r.id, r.title, fmt(r.value) if not isinstance(r.value, str) else r.value, r.bar,
                    fmt(r.n) if not isinstance(r.n, str) else r.n, f"**{r.verdict}**", src_txt(r.sources, 2)]
                   for r in c.rules])
    L += ["## Scoring details", ""]
    for r in c.rules:
        L += [f"### {r.id}: {r.verdict}", "", f"- Rule: {r.title}.", f"- Bar: {r.bar}.",
              f"- Scored quantity: {r.quantity}.",
              f"- Value: {r.value if isinstance(r.value, str) else fmt(r.value)}. n: "
              f"{r.n if isinstance(r.n, str) else fmt(r.n)}.",
              f"- Sources: {src_txt(r.sources, 8)}."]
        if r.reason:
            L.append(f"- Notes: {r.reason}.")
        L.append("")
        recs = [d for d in r.details if isinstance(d, dict)]
        if recs:
            L += _detail_table(recs)
    L += ["## Headline numbers", "",
          "Key, value and source file with the commit recorded by the run; unrounded in `results/final/numbers.json` "
          "and in the `final_phase` section of `PAPER_NUMBERS.json`.", ""]
    L += md_table(["Key", "Value", "Unit", "n", "Description", "Source"],
                  [[k, fmt(e["value"]) if not isinstance(e["value"], str) else e["value"], e["unit"] or "",
                    fmt(e["n"]) if e["n"] is not None else "", e["description"], src_txt(e["sources"], 2)]
                   for k, e in sorted(c.numbers.items())])
    L += ["## Deviations logged after the pre-registration commit", "",
          f"From `{DECISIONS_FILE}` ({how}).", ""]
    L += (devs + [""]) if devs else ["None.", ""]
    L += ["## Tables", "", "Under `results/final/tables/` (CSV with source columns, and markdown): "
          + ", ".join(sorted({p.stem for p in tfiles})) + ".", ""]
    L += ["## Scoring conventions", "",
          "- Section 3 of the pre-registration: a rule over several units fails as soon as one available unit fails, "
          "passes only when every unit is present and passes, and is otherwise PENDING with the reason; a rule whose "
          "needed input was cut is NOT RUN, never passed.",
          "- \"At least k of n\" rules (A06-1, A06-2) pass once k units pass and fail once k can no longer be reached.",
          "- \"Within x\" is |difference| <= x; values within 1e-9 of a bar count as equal to the bar.",
          "- Healed values: final-step bf16 k-NN (step 780), all 50,000 validation queries, 100k bank.",
          "- 100% flip consistency: no disagreement, or every disagreement a near-tie in both copies (relative top-two "
          "margin below 1e-4 in the image and its mirror). When only the one-copy margin is recorded, the unit is "
          "PENDING unless a one-copy margin already reaches 1e-4.",
          "- A05-1: point estimate = mean over fit sets of the fit-set means; interval = mirror.final.stats.hbootstrap "
          "(10,000 resamples, seed 0, 95% percentile, images paired, seeds drawn independently per method).",
          "- A07: candidates whose measured time per image at batch 256 (flip averaging: two passes) is at most "
          "Mirror's faster stored form; Mirror must be at least 0.5 above the best of them.", ""]
    if c.notes:
        L += ["## Collector notes", ""] + [f"- {n}" for n in c.notes] + [""]
    L += ["## Inputs read", "", f"{len(c.inputs)} files; their sha256 are in `results/final/numbers.json` (`inputs`).",
          ""]
    text = "\n".join(L) + "\n"
    text = text.replace("\u2014", "-")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    payload = {"schema": "final_phase_numbers/v1", "written_by": "scripts/collect_final_phase.py",
               "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "collector_git": me,
               "root": "." if c.root == REPO_ROOT.resolve() else c.root.as_posix(),
               "prereg": {k: v_ for k, v_ in pre.items() if k not in ("text_norm", "rows")},
               "main_arm": {"variant": v, "how": how_main}, "counts": counts,
               "rules": [rule_dict(r) for r in c.rules], "numbers": c.numbers, "bootstrap": c.boot,
               "deviations": devs, "not_run_jobs": nr, "tables": [c.rel(p) for p in tfiles], "notes": c.notes,
               "inputs": c.inputs}
    write_json_atomic(numbers_json, payload)
    section = {"schema": "paper_numbers/final_phase/v1", "written_by": "scripts/collect_final_phase.py",
               "date_utc": payload["date_utc"], "collector_git": me, "prereg": payload["prereg"],
               "main_arm": payload["main_arm"], "counts": counts,
               "rules": [{k: d[k] for k in ("id", "verdict", "value", "bar", "n", "reason", "sources")}
                         for d in payload["rules"]],
               "numbers": c.numbers, "report": c.rel(report), "numbers_file": c.rel(numbers_json)}
    merge_paper_numbers(paper_numbers, section)
    return payload


def merge_paper_numbers(path: Path, section: dict) -> None:
    """Add (or replace) the "final_phase" key of PAPER_NUMBERS.json; every other key is kept as it is."""
    path = Path(path)
    data = read_json(path) if path.is_file() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object; not touched")
    data["final_phase"] = jsonable(section)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_all(c: Final) -> None:
    for fn in SCORERS:
        try:
            fn(c)
        except Exception as e:  # noqa: BLE001  a broken input must not hide the other results
            c.notes.append(f"{fn.__name__} raised {type(e).__name__}: {e}")
            log(f"WARNING {fn.__name__}: {type(e).__name__}: {e}")
    try:
        build_tables(c)
    except Exception as e:  # noqa: BLE001
        c.notes.append(f"build_tables raised {type(e).__name__}: {e}")
        log(f"WARNING build_tables: {type(e).__name__}: {e}")

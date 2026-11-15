#!/usr/bin/env python
"""UP-07 (PROJECT_NEXT_PHASE.md, Section 7): pairing from weights alone, second attempt (CPU only).

    python scripts/up07_weights_pairing.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0 --blocks 1-12
    python scripts/up07_weights_pairing.py --tiny        # planted mirror-symmetric tiny ViT, CPU, under a minute

Bar (PREREG_NEXT_PHASE.md, Section 4, reported, not gating): pairing from weights alone agrees with the data-driven
pairing on at least 70% of pairs (the bar of the first attempt, P5 part 2 of PREREG_E2.md).

First attempt (scripts/d_datafree.py, runs/d/D_DATAFREE_REPORT.md): MLP twins from the fc1 rows only,
S[i, j] = cos(w_i, A w_j), in block 1 recovered 0.466 of the data-driven pairs given the data-driven A_0 and 0.441
with the operator read from the weights (A_0^w).

Second attempt (this script): the fc2 columns are scored jointly with the fc1 rows. An exactly mirror-equivariant MLP
satisfies P W1 = W1 A_in and A_out W2 = W2 P (mirror/surgery.py), so twins i and j have W1[j] = W1[i] A_in (rows) and
W2[:, j] = A_out W2[:, i] (columns). The score is
    S = (1 - alpha) S_fc1 + alpha S_fc2,  S_fc1[i, j] = cos(W1[i] A_in, W1[j]),  S_fc2[i, j] = cos(A_out W2[:, i], W2[:, j]),
on the folded weights (norm2 affine in fc1, LayerScale in fc2), both symmetric, with alpha = 0.5 (equal weight, fixed
in code before any run on real weights). The pairing is the max-weight involution matching of the twins
(mirror/matching.py: edge weight 2 S_ij - S_ii - S_jj, top-10 candidates per neuron, exact blossom), as in E1b and the
first attempt.

Operators from the weights alone (mirror/datafree.py):
  layer 0   A_0^w from the patch and position embeddings (D1, isotropic N(0, I) prior on normalised pixels, sigma^2 = 1:
            the first attempt's primary choice);
  layer b+1 A_(b+1)^w from the D3 moment model: the layer-b moments are propagated through block b with the block's
            weight-only fc1 pairing (the first attempt's fully data-free chain), then fit_involution.
Block b (code index b - 1 for "block b" in the plan) uses A_in = A_(b-1)^w for fc1 and A_out = A_b^w for fc2, as the
per-layer data-driven protocol uses A_in = A_(b-1), A_out = A_b (scripts/run_e1.py). With --chain-pairing joint the
chain is propagated with the joint pairing instead (one extra sequential matching per block).

Variants (all scored on the same blocks; recovery = data-driven pairs found / data-driven pairs, exactly the first
attempt's measure, mirror.datafree.pairing_agreement):
  weights_joint  PRIMARY: weights-only operators, fc1 and fc2 jointly;
  weights_fc1    weights-only operators, fc1 rows only (the first attempt's rule; block 1 reproduces its 0.441);
  weights_fc2    weights-only operators, fc2 columns only;
  shared_joint   A_0^w for fc1 and fc2 in every block (one shared operator, as Mirror-exact), jointly;
  data_joint     data-driven A_(b-1), A_b of the fit run (the "given A" reading of the first attempt), jointly;
  data_fc1       data-driven A_(b-1), fc1 rows only (the first attempt's D2 rows; block 1 reproduces its 0.466).
Verdict: weights_joint in block 1 against the 0.70 bar (block 1 is where the first attempt was scored). Also reported:
data_joint in block 1 (the reading of the first attempt's 46.6%), and the recovery pooled over the requested blocks.

Outputs under --out (default results/UP-07; tiny: tiny_out/results/UP-07): pairing_by_block.csv (one row per block
and variant), operators_by_layer.csv, pairings.npz (every pairing found, the weights-only operators), up07.json,
summary.md, meta.json (mirror/meta.py). Everything runs on the CPU in float64; --gpu-frac only caps a GPU if one is
visible (none is used).
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.05


def _early_gpu_frac(argv, default: float = DEFAULT_GPU_FRAC) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap (nothing is allocated)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future, ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.datafree import (compare_operators, fit_layer0, layer0_parts, pair_score_summary,  # noqa: E402
                             pairing_agreement, propagate_block, twin_scores)
from mirror.fit import fit_involution  # noqa: E402
from mirror.matching import build_result, candidate_edges, solve_matching  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.surgery import fold_mlp  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, git_info, log, set_seed  # noqa: E402
from mirror.variants import load_fit_run  # noqa: E402

EXPERIMENT = "UP-07"
BAR = 0.70            # PREREG_NEXT_PHASE.md Section 4, UP-07 (same bar as P5 part 2 of PREREG_E2.md)
ALPHA = 0.5           # weight of the fc2 term; fixed before any run on real weights
SIGMA2 = 1.0          # D1 weighting: the first attempt's primary choice (scripts/d_datafree.py SIGMA2_PRIMARY)
TOP_K = 10            # matching candidates per neuron (E1b and the first attempt)
PRIMARY = "weights_joint"
VARIANTS = {
    "weights_joint": "weights-only operators, fc1 and fc2 jointly (primary)",
    "weights_fc1": "weights-only operators, fc1 rows only (first attempt's rule)",
    "weights_fc2": "weights-only operators, fc2 columns only",
    "shared_joint": "A_0^w on both sides in every block, jointly",
    "data_joint": "data-driven operators, fc1 and fc2 jointly",
    "data_fc1": "data-driven A_in, fc1 rows only (first attempt's D2)",
}
# First attempt, block 1 (runs/d/d_datafree.json, d2 rows, block 1; P5 part 2 and its strict reading).
FIRST_ATTEMPT = {
    "data_fc1": {"n_common": 544, "n_pairs_ref": 1168, "recovery": 544 / 1168,
                 "label": "fc1 rows only, given the data-driven A_0 (P5 part 2)"},
    "weights_fc1": {"n_common": 515, "n_pairs_ref": 1168, "recovery": 515 / 1168,
                    "label": "fc1 rows only, operator from the weights A_0^w (strict reading)"},
}
FIRST_ATTEMPT_JSON = REPO_ROOT / "runs" / "d" / "d_datafree.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="torch.hub entry")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--img-size", type=int, default=224, help="crop size of the operator run")
    p.add_argument("--fit-run", default="runs/x2/perlayer_reg_s0",
                   help="run_e1.py run folder (results/e1_operators.npz) or the npz itself: the data-driven pairing")
    p.add_argument("--blocks", default=None, help="1-based blocks, e.g. 1-12 or 1,2,3 or all (default 1-12; tiny all)")
    p.add_argument("--alpha", type=float, default=ALPHA, help="weight of the fc2 term in the joint score")
    p.add_argument("--chain-pairing", default="fc1", choices=["fc1", "joint"],
                   help="pairing used to propagate the weight-only moments (fc1: the first attempt's chain)")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--workers", default="auto", help="processes for the blossom matchings (auto: min(8, cores - 1))")
    p.add_argument("--n-nodes", type=int, default=64, help="Gauss-Hermite nodes of the moment model (as before)")
    p.add_argument("--out", default=None, help="default results/UP-07 (tiny: tiny_out/results/UP-07)")
    p.add_argument("--seed", type=int, default=0, help="global seed (the matching itself is deterministic)")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC, help="GPU memory cap if a GPU is visible")
    p.add_argument("--allow-model-mismatch", action="store_true")
    p.add_argument("--tiny", action="store_true", help="planted mirror-symmetric tiny timm ViT, synthetic data")
    return p.parse_args(argv)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(name: str):
    key = f"_up07_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def parse_blocks(spec: str, L: int) -> list:
    """1-based block list ("1-12", "1,3,5", "all") -> sorted 0-based indices."""
    s = spec.strip().lower()
    if s == "all":
        return list(range(L))
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            out.update(range(lo - 1, hi))
        else:
            out.add(int(part) - 1)
    out = sorted(out)
    if not out or out[0] < 0 or out[-1] >= L:
        raise ValueError(f"blocks must be in 1..{L}, got {spec!r}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------------------------
def column_scores(W2, A_out) -> np.ndarray:
    """S_fc2[i, j] = cos(A_out W2[:, i], W2[:, j]) for W2 [d, h] (symmetric because A_out is)."""
    return twin_scores(np.asarray(W2, dtype=np.float64).T, A_out)


def joint_scores(W1, W2, A_in, A_out, alpha: float = ALPHA) -> np.ndarray:
    """(1 - alpha) cos(W1[i] A_in, W1[j]) + alpha cos(A_out W2[:, i], W2[:, j]) (alpha 0: fc1 only; 1: fc2 only)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return twin_scores(W1, A_in)
    if alpha == 1.0:
        return column_scores(W2, A_out)
    return (1.0 - alpha) * twin_scores(W1, A_in) + alpha * column_scores(W2, A_out)


def match_scores(S: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    """The twins' max-weight involution matching (mirror.matching) on a symmetric score; returns the permutation."""
    return build_result(S, solve_matching(*candidate_edges(S, top_k))).perm


# ----------------------------------------------------------------------------------------------
# Matching jobs (blossom in worker processes; the moment chain runs in the main process)
# ----------------------------------------------------------------------------------------------
class Matcher:
    """Blossom matchings in worker processes (or inline with one worker). Each job keeps its score matrix (float32)
    only until its pairing is back; then its row against the data-driven pairing is computed and the matrix dropped.
    `seconds` runs from submission to collection (queue time included)."""

    def __init__(self, workers: int, top_k: int, ref_perms):
        self.top_k = int(top_k)
        self.ref = ref_perms
        self.ex = ProcessPoolExecutor(max_workers=int(workers)) if workers > 1 else None
        self.jobs = []

    def submit(self, block: int, variant: str, S: np.ndarray, operators: str, inline: bool = False) -> dict:
        """inline: solve in this process now (the sequential chain), never queued behind the pool's jobs."""
        S = np.asarray(S, dtype=np.float64)
        edges = candidate_edges(S, self.top_k)
        if self.ex is None or inline:
            fut = Future()
            fut.set_result(solve_matching(*edges))
        else:
            fut = self.ex.submit(solve_matching, *edges)
        job = {"block": int(block), "variant": variant, "S": S.astype(np.float32), "future": fut,
               "operators": operators, "t0": time.time(), "result": None, "row": None}
        self.jobs.append(job)
        return job

    def finish(self, job: dict) -> np.ndarray:
        if job["result"] is None:
            pairs = job["future"].result()
            S = job.pop("S").astype(np.float64)
            job["result"] = build_result(S, pairs).perm
            job["row"] = job_row(job, S, self.ref[job["block"]], round(time.time() - job["t0"], 1))
        return job["result"]

    def close(self) -> None:
        for job in self.jobs:
            self.finish(job)
        if self.ex is not None:
            self.ex.shutdown(wait=True)


def job_row(job: dict, S: np.ndarray, ref_perm: np.ndarray, seconds: float | None = None) -> dict:
    perm = job["result"]
    ag = pairing_agreement(ref_perm, perm)
    sd = pair_score_summary(S, ref_perm)
    st = pair_score_summary(S, perm)
    return {"block": job["block"] + 1, "variant": job["variant"], "operators": job["operators"], **ag,
            "mean_S_data_pairs": sd["mean_S_pairs"], "mean_S_data_self": sd["mean_S_self"],
            "mean_S_found_pairs": st["mean_S_pairs"], "mean_S_found_self": st["mean_S_self"],
            "seconds": seconds}


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
def load_inputs(a, out: Path) -> dict:
    planted = None
    if a.tiny:
        DD = _load("d_datafree")
        adapter, planted = DD.build_tiny(a.seed)
        npz, _ = DD.tiny_operators(adapter, out, a.seed, a.top_k)
        model_name = "tiny_planted"
        struct = load_fit_run(npz)
    else:
        from mirror.adapters import Dinov2Adapter

        adapter = Dinov2Adapter(hub_repo=a.hub_repo, name=a.model, img_size=a.img_size)
        adapter.to(device=torch.device("cpu"), dtype=torch.float64)
        model_name = a.model
        struct = load_fit_run(resolve(a.fit_run))
        run_model = struct.meta.get("model")
        if run_model is not None and run_model != model_name and not a.allow_model_mismatch:
            raise SystemExit(f"the fit run's operators were fitted on {run_model}, not {model_name}")
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    if len(struct.A_layers) != L + 1 or struct.d != d or struct.h != h or struct.depth != L:
        raise SystemExit(f"fit run does not fit the model: {len(struct.A_layers)} operators of size {struct.d}, "
                         f"{struct.depth} pairings of {struct.h}; model L {L}, d {d}, h {h}")
    return {"adapter": adapter, "struct": struct, "model_name": model_name, "planted": planted}


def first_attempt_check() -> dict | None:
    """The first attempt's stored rows (runs/d/d_datafree.json), if present, for the reproduction check."""
    if not FIRST_ATTEMPT_JSON.exists():
        return None
    R = json.loads(FIRST_ATTEMPT_JSON.read_text(encoding="utf-8"))
    out = {"data_fc1": {}, "weights_fc1": {}}
    for r in R.get("d2", {}).get("rows", []):
        if r.get("operator") == "data A_b":
            out["data_fc1"][int(r["block"])] = r["recovery"]
        elif r.get("operator") == "data-free A_0^w":
            out["weights_fc1"][int(r["block"])] = r["recovery"]
    for r in (R.get("d3") or {}).get("pairing_rows", []):
        out["weights_fc1"][int(r["block"])] = r["recovery"]
    return {"path": str(FIRST_ATTEMPT_JSON.relative_to(REPO_ROOT)), "sha256": sha256(FIRST_ATTEMPT_JSON),
            "recovery": out, "operators_file": R.get("provenance", {}).get("operators", {}).get("sha256")}


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------
def write_csv(path: Path, rows: list) -> None:
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    os.replace(tmp, path)


def _f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(xf) else f"{xf:.{nd}f}"


def pooled(rows: list, variant: str, blocks: list) -> dict:
    sel = [r for r in rows if r["variant"] == variant and r["block"] - 1 in blocks]
    ref = sum(r["n_pairs_ref"] for r in sel)
    com = sum(r["n_common"] for r in sel)
    return {"variant": variant, "n_blocks": len(sel), "n_common": com, "n_pairs_ref": ref,
            "recovery_pooled": com / ref if ref else float("nan"),
            "recovery_mean_over_blocks": float(np.mean([r["recovery"] for r in sel])) if sel else float("nan")}


def write_summary(path: Path, R: dict) -> None:
    v = R["verdict"]
    L = []
    L.append(f"# UP-07: pairing from weights alone, second attempt"
             f"{' (TINY TEST: planted tiny model, not a result)' if R['tiny'] else ''}\n")
    L.append("## Verdict\n")
    L.append(f"- Primary (weights only; fc1 rows and fc2 columns scored jointly; A_0^w for fc1 and the propagated "
             f"A_1^w for fc2): block 1 recovers {_f(v['primary']['recovery'])} of the data-driven pairs "
             f"({v['primary']['n_common']} of {v['primary']['n_pairs_ref']}; bar {BAR}): "
             f"**{'PASS' if v['primary']['pass'] else 'FAIL'}**.")
    L.append(f"- First attempt (fc1 rows only, runs/d): {_f(FIRST_ATTEMPT['data_fc1']['recovery'])} given the "
             f"data-driven A_0 (the 46.6% of P5 part 2) and {_f(FIRST_ATTEMPT['weights_fc1']['recovery'])} with A_0^w.")
    if v.get("given_data"):
        g = v["given_data"]
        L.append(f"- The same joint rule given the data-driven operators (A_0 for fc1, A_1 for fc2; the reading of the "
                 f"first attempt's 46.6%): {_f(g['recovery'])} ({g['n_common']} of {g['n_pairs_ref']}): "
                 f"**{'PASS' if g['pass'] else 'FAIL'}** against the same bar.")
    pr = v.get("pooled_primary")
    if pr:
        L.append(f"- Primary rule pooled over blocks {R['blocks_label']}: {_f(pr['recovery_pooled'])} of the "
                 f"data-driven pairs ({pr['n_common']} of {pr['n_pairs_ref']}); mean over blocks "
                 f"{_f(pr['recovery_mean_over_blocks'])}. Reported, not scored.")
    rep = R.get("reproduction")
    if rep:
        L.append(f"- Reproduction of the first attempt in block 1 (same code path, fc1 rows only): data-driven A_0 "
                 f"{_f(rep['data_fc1'])} (stored {_f(rep['stored_data_fc1'])}), A_0^w {_f(rep['weights_fc1'])} "
                 f"(stored {_f(rep['stored_weights_fc1'])}); layer-0 operator agreement (P5 part 1) "
                 f"{_f(R['operators'][0]['mean_cos_even'])} (first attempt 0.900).")
        pb = rep.get("per_block") or {}
        if pb:
            worst = max(abs(x["now"] - x["stored"]) for x in pb.values())
            L.append(f"- Against every stored fc1-only row of the first attempt ({len(pb)} rows, {rep['stored_file']}; "
                     f"same operators file: {'yes' if rep.get('same_operators_file') else 'no'}), the largest "
                     f"difference in recovery is {worst:.4f}.")
    L.append("")
    L.append("## Recovery by block\n")
    L.append("Recovery = data-driven pairs found / data-driven pairs (mirror.datafree.pairing_agreement, the first "
             "attempt's measure). Chance = expected recovery of a random pairing with the primary rule's number of "
             "pairs.\n")
    names = [n for n in VARIANTS if any(r["variant"] == n for r in R["rows"])]
    L.append("| block | pairs (data) | " + " | ".join(names) + " | pairs found (primary) | chance |")
    L.append("|---|---|" + "---|" * len(names) + "---|---|")
    by = {(r["block"], r["variant"]): r for r in R["rows"]}
    for b in sorted({r["block"] for r in R["rows"]}):
        ref = next(r for r in R["rows"] if r["block"] == b)
        prim = by.get((b, PRIMARY), {})
        L.append(f"| {b} | {ref['n_pairs_ref']} | " + " | ".join(_f(by.get((b, n), {}).get("recovery")) for n in names)
                 + f" | {prim.get('n_pairs_test', 'n/a')} | {_f(prim.get('chance_recovery'), 5)} |")
    L.append("")
    L.append("Variants: " + "; ".join(f"{k}: {VARIANTS[k]}" for k in names) + ".\n")
    L.append("Pooled over the requested blocks:\n")
    L.append("| variant | pairs found | data-driven pairs | pooled recovery | mean over blocks |")
    L.append("|---|---|---|---|---|")
    for p in R["pooled"]:
        L.append(f"| {p['variant']} | {p['n_common']} | {p['n_pairs_ref']} | {_f(p['recovery_pooled'])} | "
                 f"{_f(p['recovery_mean_over_blocks'])} |")
    L.append("")
    L.append("## Weight-only operators against the data-driven ones\n")
    L.append("Layer l = the stream after block l (layer 0 = block 1's input). Mean cosine of the principal angles "
             "between the +1 eigenspaces and tr(A A^w)/d (mirror.datafree.compare_operators); the copied row uses "
             "A_0^w at every layer.\n")
    L.append("| layer | d+ (data) | d+ (weights) | mean cos even | mean cos odd | tr(A A^w)/d | tr, A_0^w copied |")
    L.append("|---|---|---|---|---|---|---|")
    for o in R["operators"]:
        L.append(f"| {o['layer']} | {o['d_plus_ref']} | {o['d_plus_test']} | {_f(o['mean_cos_even'])} | "
                 f"{_f(o['mean_cos_odd'])} | {_f(o['trace_agreement'])} | {_f(o['copied_trace_agreement'])} |")
    L.append("")
    pv = R["provenance"]
    L.append("## Provenance\n")
    L.append(f"- Model `{pv['model']}`; data-driven pairing `{pv['fit_run']}` (sha256 {pv['npz_sha256'][:16]}, fit seed "
             f"{pv['fit_seed']}); git {pv['git']['git_hash']}{' (dirty)' if pv['git'].get('git_dirty') else ''}; "
             f"seed {pv['seed']}; alpha {pv['alpha']}; top-k {pv['top_k']}; chain pairing {pv['chain_pairing']}; "
             f"{pv['workers']} matching worker{'s' if pv['workers'] != 1 else ''}; {pv['minutes']:.1f} min on the CPU.")
    if R.get("planted"):
        pl = R["planted"]
        L.append(f"- Planted check (tiny): the primary rule recovers {_f(pl['primary_vs_planted']['recovery'])} of the "
                 f"planted block-1 pairs; A_0^w against the planted operator: tr(A A^w)/d "
                 f"{_f(pl['A0w_vs_planted']['trace_agreement'])}.")
    L.append("")
    L.append("## Caveats\n")
    for c in R["caveats"]:
        L.append(f"- {c}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    seed = int(a.seed)
    set_seed(seed)
    git = git_info()
    timer = StageTimer()
    out = resolve(a.out or ("tiny_out/results/UP-07" if a.tiny else "results/UP-07"))
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(8, (os.cpu_count() or 2) - 1)) if a.workers == "auto" else int(a.workers)
    log(f"UP-07 {'TINY ' if a.tiny else ''}run, commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}, "
        f"seed {seed}, workers {workers}, alpha {a.alpha}, chain pairing {a.chain_pairing}")

    with timer.stage("load"):
        inp = load_inputs(a, out)
    adapter, struct, model_name = inp["adapter"], inp["struct"], inp["model_name"]
    L = adapter.depth
    blocks = parse_blocks(a.blocks or ("all" if a.tiny else "1-12"), L)
    n_chain = max(blocks) + 1
    A_data = struct.A_layers
    log(f"model {model_name}: L {L}, d {adapter.embed_dim}, h {adapter.mlp_hidden}; fit run {struct.source}; "
        f"blocks {[b + 1 for b in blocks]}")

    with timer.stage("d1"):
        parts = layer0_parts(adapter, check_seed=seed)
        if parts["affine_rel_err"] > 1e-8:
            raise SystemExit(f"layer-0 map is not the patch embedding plus a constant (rel err {parts['affine_rel_err']:.2e})")
        d1 = fit_layer0(parts, sigma2=SIGMA2)
        A0w, state = d1["A"], d1["moments"]
        folded = [fold_mlp(adapter.blocks[b]) for b in range(n_chain)]
        W1 = [f["W1"].cpu().numpy() for f in folded]
        W2 = [f["W2"].cpu().numpy() for f in folded]
        del folded

    M = Matcher(workers, a.top_k, struct.perms)
    jobs = []
    with timer.stage("matchings"):
        # Independent of the chain: submitted first so they run while the chain proceeds.
        for b in blocks:
            jobs.append(M.submit(b, "shared_joint", joint_scores(W1[b], W2[b], A0w, A0w, a.alpha), "A_0^w, A_0^w"))
            jobs.append(M.submit(b, "data_joint", joint_scores(W1[b], W2[b], A_data[b], A_data[b + 1], a.alpha),
                                 f"data A_{b}, A_{b + 1}"))
            jobs.append(M.submit(b, "data_fc1", twin_scores(W1[b], A_data[b]), f"data A_{b}"))
        ops_w, A_cur = [A0w], A0w
        for b in range(n_chain):
            t0 = time.time()
            S1 = twin_scores(W1[b], A_cur)
            j1 = M.submit(b, "weights_fc1", S1, f"A_{b}^w", inline=True)
            p_fc1 = M.finish(j1)
            st = propagate_block(adapter.blocks[b], state, p_fc1, n_nodes=a.n_nodes)
            A_next = fit_involution(st["C"])[0]
            S2 = column_scores(W2[b], A_next)
            jj = M.submit(b, "weights_joint", (1.0 - a.alpha) * S1 + a.alpha * S2, f"A_{b}^w, A_{b + 1}^w",
                          inline=a.chain_pairing == "joint")
            j2 = M.submit(b, "weights_fc2", S2, f"A_{b + 1}^w")
            jobs += [j1, jj, j2]
            if a.chain_pairing == "joint":
                st = propagate_block(adapter.blocks[b], state, M.finish(jj), n_nodes=a.n_nodes)
                A_next = fit_involution(st["C"])[0]
            state, A_cur = st, A_next
            ops_w.append(A_next)
            del S1, S2
            log(f"chain block {b + 1}: fc1 pairing {int((p_fc1 != np.arange(len(p_fc1))).sum() // 2)} pairs, "
                f"operator A_{b + 1}^w d+ {int(round((np.trace(A_next) + A_next.shape[0]) / 2))} "
                f"({time.time() - t0:.1f} s)")
        M.close()

    rows = [job["row"] for job in jobs if job["block"] in blocks]
    order = {n: i for i, n in enumerate(VARIANTS)}
    rows.sort(key=lambda r: (r["block"], order[r["variant"]]))
    for r in rows:
        if r["variant"] in (PRIMARY, "data_joint", "data_fc1", "weights_fc1"):
            log(f"block {r['block']} {r['variant']}: recovery {r['recovery']:.4f} ({r['n_common']} of {r['n_pairs_ref']}), "
                f"precision {r['precision']:.4f}")

    op_rows = []
    for l in range(n_chain + 1):
        c = compare_operators(A_data[l], ops_w[l])
        op_rows.append({"layer": l, **c, "copied_trace_agreement": compare_operators(A_data[l], A0w)["trace_agreement"],
                        "copied_mean_cos_even": compare_operators(A_data[l], A0w)["mean_cos_even"]})

    by = {(r["block"], r["variant"]): r for r in rows}
    verdict = {"bar": BAR, "scored_block": 1}
    if 0 in blocks:
        p = by[(1, PRIMARY)]
        verdict["primary"] = {"variant": PRIMARY, "recovery": p["recovery"], "n_common": p["n_common"],
                              "n_pairs_ref": p["n_pairs_ref"], "pass": bool(p["recovery"] >= BAR)}
        g = by[(1, "data_joint")]
        verdict["given_data"] = {"variant": "data_joint", "recovery": g["recovery"], "n_common": g["n_common"],
                                 "n_pairs_ref": g["n_pairs_ref"], "pass": bool(g["recovery"] >= BAR)}
    else:
        verdict["primary"] = {"variant": PRIMARY, "recovery": float("nan"), "n_common": 0, "n_pairs_ref": 0,
                              "pass": False, "note": "block 1 not requested: not scored"}
    pooled_rows = [pooled(rows, n, blocks) for n in VARIANTS]
    verdict["pooled_primary"] = next(p for p in pooled_rows if p["variant"] == PRIMARY)

    # Reproduction of the first attempt (its stored values belong to the real model, so not in tiny mode).
    stored = None if a.tiny else first_attempt_check()
    reproduction = None
    if 0 in blocks and not a.tiny:
        reproduction = {"data_fc1": by[(1, "data_fc1")]["recovery"], "weights_fc1": by[(1, "weights_fc1")]["recovery"],
                        "stored_data_fc1": FIRST_ATTEMPT["data_fc1"]["recovery"],
                        "stored_weights_fc1": FIRST_ATTEMPT["weights_fc1"]["recovery"]}
        if stored:
            reproduction["per_block"] = {f"{var}_block{blk}": {"now": by[(blk, var)]["recovery"], "stored": val}
                                         for var in ("data_fc1", "weights_fc1")
                                         for blk, val in stored["recovery"][var].items() if (blk, var) in by}
            reproduction["stored_file"] = stored["path"]
            reproduction["same_operators_file"] = stored.get("operators_file") == sha256(Path(struct.source))

    planted = None
    if inp["planted"] is not None:
        pl = inp["planted"]
        j = next(job for job in jobs if job["block"] == 0 and job["variant"] == PRIMARY)
        planted = {"primary_vs_planted": pairing_agreement(pl["perm"], j["result"]),
                   "A0w_vs_planted": compare_operators(pl["A"], A0w)}

    found = {(job["block"], job["variant"]): job["result"] for job in jobs}
    np.savez_compressed(out / "pairings.npz", A_w=np.stack(ops_w).astype(np.float32),
                        blocks=np.asarray([b + 1 for b in blocks], dtype=np.int64),
                        **{f"perm_{n}": np.stack([found[(b, n)] for b in blocks]).astype(np.int32) for n in VARIANTS})
    write_csv(out / "pairing_by_block.csv", [{**r, "model": model_name, "alpha": a.alpha, "top_k": a.top_k,
                                              "chain_pairing": a.chain_pairing, "seed": seed,
                                              "git_hash": git["git_hash"]} for r in rows])
    write_csv(out / "operators_by_layer.csv", [{**o, "model": model_name, "git_hash": git["git_hash"]} for o in op_rows])

    caveats = [
        f"alpha = {a.alpha} (equal weight of the fc1 and fc2 cosines) was fixed in code before any run on real weights; "
        "no other alpha is scored.",
        "The fc2 operator A_(b)^w comes from the first attempt's moment model (attention replaced by its value-output "
        "map, independent Gaussian pre-activations), propagated with the weight-only fc1 pairing; its error grows with "
        "depth (see the operator table).",
        f"The data-driven pairing is itself an estimate from one fit set (fit seed {struct.meta.get('fit_seed')}); it "
        "is the reference here, not the truth, so 100% is not the ceiling of a perfect weight-only rule.",
        "Block 1 is scored, as in the first attempt; the other blocks and the pooled recovery are reported, not scored.",
        "PREREG_NEXT_PHASE.md does not say whether 'weights alone' may use the data-driven operators; the primary row "
        "uses none, the data_joint row is shown for comparison with the first attempt's 46.6%.",
    ]
    if a.tiny:
        caveats.insert(0, "TINY TEST: a random tiny ViT with a planted exact mirror symmetry in layer 0 and block 1, "
                          "data-driven pairing from synthetic images; every number checks the code, none is a result.")
    minutes = (time.time() - t_start) / 60
    provenance = {"model": model_name, "fit_run": str(struct.source), "npz_sha256": sha256(Path(struct.source)),
                  "fit_seed": struct.meta.get("fit_seed"), "git": git, "seed": seed, "alpha": a.alpha,
                  "top_k": a.top_k, "chain_pairing": a.chain_pairing, "workers": workers, "minutes": minutes}
    lab = [b + 1 for b in blocks]
    blocks_label = f"{lab[0]} to {lab[-1]}" if lab == list(range(lab[0], lab[-1] + 1)) else ", ".join(map(str, lab))
    R = {"tiny": bool(a.tiny), "verdict": verdict, "rows": rows, "pooled": pooled_rows, "operators": op_rows,
         "reproduction": reproduction, "planted": planted, "first_attempt": FIRST_ATTEMPT, "caveats": caveats,
         "provenance": provenance, "blocks_label": blocks_label, "timing": timer.stages}
    with open(out / "up07.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    write_summary(out / "summary.md", R)
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": seed, "fit_seed": struct.meta.get("fit_seed")},
               models={model_name: {"hub_repo": None if a.tiny else a.hub_repo, "img_size": adapter.img_size}},
               device="cpu", wall_seconds=time.time() - t_start,
               extra={"fit_run": str(struct.source), "npz_sha256": provenance["npz_sha256"], "bar": BAR,
                      "first_attempt": FIRST_ATTEMPT, "first_attempt_file": stored, "timing": timer.stages,
                      "workers": workers, "verdict": verdict})
    pv = verdict["primary"]
    log(f"UP-07 done in {minutes:.1f} min: block-1 recovery {pv['recovery']:.4f} (bar {BAR}) "
        f"{'PASS' if pv['pass'] else 'FAIL'}; results in {out}")
    return R


if __name__ == "__main__":
    main()

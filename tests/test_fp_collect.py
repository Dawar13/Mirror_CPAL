"""Tests of the final-phase collector (mirror/final/collect.py, scripts/collect_final_phase.py), Gate P
(scripts/fp_gate_p.py) and the A01 decision (scripts/fp_a01_decide.py).

A synthetic tree in a temporary folder holds run directories and result files in the formats the producers write
(scripts/fp_heal.py metrics.json and meta.json, mirror/final/eval/knn_fast.py eval/step<S>.json and knn/*.npz,
scripts/fp_a08_top1.py eval/top1_step<S>.json, the A02, A03, A09, A12, A15 metrics files, fp_fullbank summary.json,
scripts/tier2_ade20k.py t2_ade20k.csv, the runner's state files). Two versions of the tree make every scored rule
pass or fail; an empty tree leaves every rule pending; a tree with cut jobs gives NOT RUN. The numbers are made up
for the tests and mean nothing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

from mirror.final import collect as C  # noqa: E402
from mirror.final.stats import hbootstrap as HB  # noqa: E402

M = C.MAIN
N = 240                                     # validation images of the synthetic tree (the real value is 50,000)
LABELS = (np.arange(N) % 10).astype(np.int64)
ORDER = np.random.default_rng(123).permutation(N)
GIT = "a" * 40
REPORT_ONLY = {"A04-2", "A06-3", "A09-3", "A11", "A15a"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wj(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def correct(k: int) -> np.ndarray:
    """k correct images; the sets are nested (correct(a) is inside correct(b) for a < b)."""
    out = np.zeros(N, dtype=bool)
    out[ORDER[:k]] = True
    return out


def run(root: Path, group: str, model: str, folder: str, seed: int, top1: float | None, *, steps: int = 780,
        step0: float | None = None, fp32: dict | None = None, flipavg: float | None = None, k: int | None = None,
        complete: bool = True, top1_cls: tuple | None = None, structure: str | None = None) -> Path:
    """One run directory as scripts/fp_heal.py and the evaluation server leave it."""
    d = root / "runs" / "final" / group / model / folder / f"s{seed}"
    m = {"complete": complete, "steps_done": steps, "total_steps": steps, "diverged": False, "seed": seed,
         "method": folder.split("_")[0], "lr": 1e-4, "git_hash": GIT,
         "params": {"model_params_structured": 66203360, "model_params_frac": 0.7646}}
    if structure:
        m["build"] = {"structure_source": structure}
    wj(d / "metrics.json", m)
    wj(d / "meta.json", {"experiment": "test", "git": {"git_hash": GIT, "git_dirty": False}})
    if top1 is not None:
        passes = {"bf16": {"top1": top1, "n_query": N, "n_bank": 100000, "flip_consistency": 99.7, "n_disagree": 1,
                           "predictions": f"knn/step{steps}_bf16.npz"}}
        if fp32 is not None:
            passes["fp32"] = {"top1": top1, "n_query": N, **fp32}
        if flipavg is not None:
            passes["bf16_flipavg"] = {"top1": flipavg, "n_query": N, "flip_consistency": 100.0, "n_disagree": 0}
        wj(d / "eval" / f"step{steps}.json", {"format": "fp-knn-v1", "tag": f"step{steps}", "step": steps,
                                              "passes": passes, "original_reference": {"bf16": {"top1": 78.95}}})
        wj(d / "eval" / f"step{steps}_meta.json", {"git": {"git_hash": GIT}})
    if step0 is not None:
        wj(d / "eval" / "step0.json", {"step": 0, "passes": {"bf16": {"top1": step0, "n_query": N}}})
    if k is not None:
        (d / "knn").mkdir(parents=True, exist_ok=True)
        c = correct(k)
        pred = np.where(c, LABELS, (LABELS + 1) % 10).astype(np.int16)
        np.savez(d / "knn" / f"step{steps}_bf16.npz", pred=pred, pred_flip=pred, labels=LABELS.astype(np.int16))
    if top1_cls is not None:
        b, nd = top1_cls
        wj(d / "eval" / f"top1_step{steps}.json",
           {"bf16": {"top1": b, "n": N, "flip_consistency": 99.9, "n_disagree": 1},
            "fp32": {"top1": b, "n": N, "flip_consistency": 100.0 if nd == 0 else 99.0, "n_disagree": nd,
                     "disagree_max_margin": None if nd == 0 else 5e-5, "pass_with_near_ties": True}})
        wj(d / "eval" / f"meta_top1_step{steps}.json", {"git": {"git_hash": GIT}})
    return d


FP32_EXACT = {"flip_consistency": 100.0, "n_disagree": 0, "max_vote_margin_rel": None, "n_disagree_not_near_tie": 0}


def a06_values() -> dict:
    """SVD-LLM points and the Mirror + SVD-LLM values of the passing tree: at 0.65 and 0.50 the measured gain equals
    the pre-registered prediction (mirror.final.baselines.sizes), at 0.40 it is 0.2."""
    from mirror.final.baselines import sizes as SZ

    svd = {0.65: 75.0, 0.50: 73.1, 0.40: 70.8, 0.765: float(np.mean([78.06, 78.05, 78.07]))}
    orig, mirror = 79.2, float(np.mean([78.60, 78.55, 78.50]))
    ms = {B: svd[B] + SZ.predicted_gain(B, svd, orig, mirror) for B in (0.65, 0.50)}
    ms[0.40] = svd[0.40] + 0.2
    return {"svd": svd, "ms": ms}


def build(root: Path, fail: bool = False) -> Path:
    """Every input of every rule: all scored rules pass (fail=False) or all fail (fail=True)."""
    v = "astar" if fail else "wspace"                      # the main arm follows the A01 outcome
    # Gate P and A01
    astar = [77.0, 77.1, 77.2] if fail else [78.40, 78.30, 78.32]
    for s, x in enumerate(astar):
        run(root, "main", M, "mirror_exact_astar", s, x, fp32=FP32_EXACT, k=(150 + s) if fail else None)
    for s, x in enumerate([78.06, 78.05, 78.07]):
        run(root, "main", M, "svdllm", s, x, k=180 + s, fp32={"flip_consistency": 95.8, "n_disagree": 2100,
                                                               "max_vote_margin_rel": 0.4, "n_disagree_not_near_tie": 2000})
    zs = 55.0 if fail else 60.0
    for s, x in enumerate([78.60, 78.55] + ([] if fail else [78.50])):
        run(root, "main", M, "mirror_exact_wspace", s, x, step0=zs, fp32=FP32_EXACT, k=None if fail else 186 + s)
    if fail:
        wj(root / "results" / "A01" / "metrics.json", {"operators": [{"operator": "wspace_matched", "d_plus": 700,
                                                                      "d_minus": 68, "knn_top1": 55.0}]})
    # A02 structure and heals
    dm = {"fit0": 68, "fit1": 100 if fail else 70, "fit2": 66, "fit3": 68}
    fits = [{"name": n, "group": "imagenet", "d_plus": 768 - d, "d_minus": d} for n, d in dm.items()]
    fits.append({"name": "coco", "group": "coco", "d_plus": 700, "d_minus": 68})
    pairs = [{"a": a, "b": b, "weighted_agreement": 0.5 if fail else 0.9, "twin_mean": 0.4 if fail else 0.8,
              "head_share_same": 0.5 if fail else 0.95, "both_imagenet": True}
             for i, a in enumerate(dm) for b in list(dm)[i + 1:]]
    wj(root / "results" / "A02" / "metrics.json", {"fits": fits, "pairs": pairs, "gates": {}})
    wj(root / "results" / "A02" / "meta.json", {"git": {"git_hash": GIT}})
    heals = {"fit1": 75.0, "fit2": 77.0, "coco": 70.0} if fail else {"fit1": 78.55, "fit2": 78.50, "coco": 78.60}
    for fit, x in heals.items():
        kk = (150 if fail else 186) if fit != "coco" else None
        run(root, "main", M, f"mirror_exact_{fit}", 0, x, k=kk, structure=f"results/A02/fits/{M}_{fit}.npz")
    # A03
    ctl = {c: {"1-6": {"twin_fraction_t50": 0.2 if (fail and c == "C1") else 0.01, "ratio_to_null": 0.88}}
           for c in ("mirror", "C1", "C2", "C3")}
    split = {c: {"ratio_to_null": 0.0 if c == "mirror" else 0.95} for c in ctl}
    wj(root / "results" / "A03" / "metrics.json", {"summaries": ctl, "split_half": split, "scores": {}})
    run(root, "main", M, "pair_random_h1", 0, 77.0 if fail else 70.0)
    # A04
    run(root, "main", M, "flap", 0, 77.5 if fail else 75.0)
    run(root, "main", M, "blockdrop", 0, 74.0)
    run(root, "main", M, "original", 0, 79.2)
    # A05 baselines (svdllm above; the others without prediction files)
    for name, vals in (("svdllm_full", [77.9] * 3), ("iosvd", [77.8] * 3), ("slicegpt", [77.5, 77.6]),
                       ("svd", [77.0]), ("random_structure", [69.9])):
        for s, x in enumerate(vals):
            run(root, "main", M, name, s, x)
    # A06
    a6 = a06_values()
    for B, tag in C.A06_BUDGETS:
        for s in ((0, 1) if tag == "f050" else (0,)):
            msv = (a6["svd"][B] + 0.1) if fail else a6["ms"][B]
            run(root, "a06", M, f"svdllm_{tag}", s, a6["svd"][B], k=190 + s if tag == "f050" else None,
                flipavg=71.5 if tag == "f040" else None)
            run(root, "a06", M, f"mirror_svdllm_{tag}", s, msv, k=200 + s if tag == "f050" else None)
    run(root, "a06", M, "mirror_partial_k9", 0, 77.5)
    # A07: systems table (main arm faster form 1 ms per image at batch 256) and released-model evaluations
    entries = [("mirror_even_odd", "mirror_even_odd", 1000.0, {"run_dir": f"runs/final/main/{M}/mirror_exact_{v}/s0"}),
               ("mirror_dense", "mirror_exact", 800.0, {"run_dir": f"runs/final/main/{M}/mirror_exact_{v}/s0"}),
               ("original", "original", 1400.0, {"model": M}),
               ("dinov2_s", "original", 5000.0, {"model": C.SMALL})]
    for tag, tp in (("f065", 1500.0), ("f050", 1800.0), ("f040", 2200.0)):
        entries.append((f"svdllm_{tag}", "svdllm", tp, {"run_dir": f"runs/final/a06/{M}/svdllm_{tag}/s0"}))
    a12 = root / "results" / "A12" / M
    a12.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"name": n, "kind": k, "passes": 1, "throughput_b64": tp, "throughput_b256": tp,
                   "latency_ms_b256": 256000.0 / tp} for n, k, tp, _ in entries]).to_csv(a12 / "systems_table.csv",
                                                                                         index=False)
    wj(a12 / "meta.json", {"git": {"git_hash": GIT}, "models": {n: meta for n, _, _, meta in entries}})
    for model, x in ((C.SMALL, 79.0 if fail else 76.0), (M, 79.0)):
        wj(root / "runs" / "final" / "a07" / model / "eval" / "original.json",
           {"original": True, "model": model, "passes": {"bf16_flipavg": {"top1": x, "n_query": N}}})
    # A08
    for s in range(3):
        run(root, "a08", C.DEIT, "mirror_exact", s, 70.0, top1_cls=(79.0 if fail else 81.0 + 0.1 * s, 0))
        run(root, "a08", C.DEIT, "svdllm", s, 70.0, top1_cls=(81.0, 5))
    wj(root / "results" / "A08" / C.DEIT / "params_check.json", {"max_abs_diff_pp": 0.0})
    wj(root / "runs" / "final" / "a08" / C.DEIT / "_original" / "eval" / "top1_original.json",
       {"bf16": {"top1": 81.8, "n": N}})
    run(root, "a08", C.LARGE, "mirror_exact", 0, 79.0 if fail else 80.0, fp32=FP32_EXACT)
    run(root, "a08", C.LARGE, "svdllm", 0, 80.2)
    # A09
    for model, var in ((M, f"mirror_exact_{v}_s0"), (C.DEIT, "mirror_exact_s0")):
        d = root / "results" / "A09" / model / var
        rows = []
        for form in ("dense", "even_odd"):
            for b in range(2):
                rows.append({"form": form, "stage": "block", "block": b, "component": "mlp", "precision": "float64_exact",
                             "in_span": True, "space": "stream", "eq_rel_tensor_max": 1e-8 if fail else 1e-12,
                             "dev_exact_rel_tensor_max": 1e-15, "a09_2_pass_exact_floor": None})
                rows.append({"form": form, "stage": "block", "block": b, "component": "mlp", "precision": "float32",
                             "in_span": True, "space": "stream", "eq_rel_tensor_max": 1e-7,
                             "dev_exact_rel_tensor_max": 1e-7, "a09_2_pass_exact_floor": not (fail and b == 1)})
        rows.append({"form": "original", "stage": "block", "block": 0, "component": "mlp", "precision": "float32",
                     "in_span": True, "space": "stream", "eq_rel_tensor_max": float("nan"),
                     "dev_exact_rel_tensor_max": 1e-7, "a09_2_pass_exact_floor": None})
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / "components.csv", index=False)
        flip = {f"{f}/float32": {"flip_consistency": 100.0, "n_disagree": 0, "n_images": N} for f in ("dense", "even_odd")}
        flip.update({f"{f}/bf16_autocast": {"flip_consistency": 99.8, "n_disagree": 3, "n_images": N}
                     for f in ("dense", "even_odd")})
        wj(d / "metrics.json", {"scores": {"A09-3": {"verdict": "REPORT", "by_form": {}}}, "flip": flip,
                                "classifier_head": model == C.DEIT})
        wj(d / "meta.json", {"git": {"git_hash": GIT}})
    # A10 and A11
    rows = [{"name": "original", "model": M, "run_dir": None, "step": None, "flip_average": False,
             "fullbank_knn_top1": 80.0, "linear_val_top1": 84.0}]
    for s in range(3):
        rows.append({"name": f"mirror_s{s}", "model": M, "run_dir": f"runs/final/main/{M}/mirror_exact_{v}/s{s}",
                     "step": 780, "flip_average": False, "fullbank_knn_top1": 77.0 if fail else 79.5,
                     "linear_val_top1": 83.5})
    rows.append({"name": "mirror_ext", "model": M, "run_dir": f"runs/final/a11/{M}/mirror_exact_ext/s0", "step": 2340,
                 "flip_average": False, "fullbank_knn_top1": 79.9, "linear_val_top1": 83.9})
    wj(root / "runs" / "final" / "A10" / "fullbank" / "summary.json", {"rows": rows})
    wj(root / "runs" / "final" / "A10" / "fullbank" / "meta.json", {"git": {"git_hash": GIT}})
    run(root, "a11", M, "mirror_exact_ext", 0, 78.9, steps=2340)
    run(root, "a11", M, "svdllm_ext", 0, 78.5, steps=2340)
    # A12b, A15a, A15b
    wj(root / "results" / "A12b" / "metrics.json", {"speed_vs_original_b64": 1.05 if fail else 1.2,
                                                    "speed_vs_original_b256": 1.15})
    wj(root / "results" / "A15a" / "metrics.json", {"comparison": {"6mlp": {"plain_val_top1": 7.84,
                                                                             "whitened_val_top1": 40.0,
                                                                             "difference": 32.16}}})
    wj(root / "results" / "A15b" / "metrics.json", {"original": {"ratio_to_null": 0.85},
                                                    "finetuned_numbers": {"ratio_to_null": 0.80 if fail else 0.70},
                                                    "scores": {}})
    # P11.2 (re-measured, runs/final/A10/p112)
    p = root / "runs" / "final" / "A10" / "p112"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": M, "variant": "vexact_s0", "mask_eq_raw": 98.98, "mask_eq_sym": 99.9999,
                   "mask_eq_sym_n_differ": 422, "mask_eq_n_pixels": 487666546,
                   "mask_eq_sym_max_rel_margin": 2e-3 if fail else 5e-5,
                   "mask_eq_sym_n_differ_not_near_tie": 3 if fail else 0,
                   "mask_eq_precision": "fp32 (no autocast, TF32 off)", "git_hash": GIT, "date": "2026-11-24"},
                  {"model": M, "variant": "original", "mask_eq_raw": 95.03, "mask_eq_sym": 94.74,
                   "mask_eq_sym_max_rel_margin": None, "git_hash": GIT, "date": "2026-11-24"}]).to_csv(
        p / "t2_ade20k.csv", index=False)
    return root


def collect(root: Path, **kw) -> tuple:
    mod = load_script("collect_final_phase")
    payload = mod.main(["--root", str(root), "--n-val", str(N), *kw.get("args", [])])
    return payload, {r["id"]: r for r in payload["rules"]}


# ------------------------------------------------------------------------------------------------
# Small rules
# ------------------------------------------------------------------------------------------------
def test_combine_k_of_n_and_bars():
    assert C.combine([C.PASS, C.PASS]) == C.PASS
    assert C.combine([C.PASS, C.PENDING]) == C.PENDING
    assert C.combine([C.PENDING, C.NOT_RUN]) == C.NOT_RUN
    assert C.combine([C.NOT_RUN, C.FAIL]) == C.FAIL
    assert C.combine([]) == C.PENDING
    assert C.k_of_n([C.PASS, C.PASS, C.FAIL], 2) == C.PASS
    assert C.k_of_n([C.PASS, C.FAIL, C.FAIL], 2) == C.FAIL
    assert C.k_of_n([C.PASS, C.PENDING, C.FAIL], 2) == C.PENDING
    assert C.k_of_n([C.PASS, C.NOT_RUN, C.FAIL], 2) == C.NOT_RUN
    # at a bar: >= and <= hold, > and < do not (binary rounding of decimals is absorbed)
    assert C.within(78.49, 78.34, 0.15) == C.PASS and C.within(78.4901, 78.34, 0.15) == C.FAIL
    assert C.cmp(78.49 - 78.34, ">=", 0.15) == C.PASS and C.cmp(-0.3, ">", -0.3) == C.FAIL
    assert C.cmp(-0.2999, ">", -0.3) == C.PASS and C.cmp(None, ">=", 1.0) == C.PENDING


def test_exact_flip_near_tie_readings():
    ok = {"flip_consistency": 99.99, "n_disagree": 2, "n_query": N, "max_vote_margin_rel": 5e-5,
          "n_disagree_not_near_tie": 0}
    assert C.exact_flip(ok, N)[0] == C.PASS
    assert C.exact_flip({**ok, "n_disagree_not_near_tie": 1, "max_vote_margin_rel": 2e-4}, N)[0] == C.FAIL
    assert C.exact_flip({**ok, "n_query": N - 1}, N)[0] == C.PENDING          # not every validation image
    assert C.exact_flip({"flip_consistency": 100.0, "n_disagree": 0, "n_query": N}, N)[0] == C.PASS
    one = {"flip_consistency": 99.9, "n_disagree": 3, "n": N, "disagree_max_margin": 5e-5}
    kw = dict(strict_key="max_margin_rel", one_copy_keys=("disagree_max_margin",), n_key="n")
    assert C.exact_flip(one, N, **kw)[0] == C.PENDING                          # only the one-copy margin
    assert C.exact_flip({**one, "disagree_max_margin": 3e-4}, N, **kw)[0] == C.FAIL


# ------------------------------------------------------------------------------------------------
# Gate P and A01 at their bars, and the two decision scripts
# ------------------------------------------------------------------------------------------------
def gate_runs(root: Path, mirror: list, svd: list) -> None:
    for s, x in enumerate(mirror):
        run(root, "main", M, "mirror_exact_astar", s, x)
    for s, x in enumerate(svd):
        run(root, "main", M, "svdllm", s, x)


@pytest.mark.parametrize("mirror,svd,expect", [
    ([78.49, 78.49, 78.49], [77.91, 77.91, 77.91], True),         # both means exactly at the edge of their bars
    ([78.19, 78.19, 78.19], [78.21, 78.21, 78.21], True),
    ([78.4901, 78.4901, 78.4901], [78.06, 78.06, 78.06], False),  # just outside
    ([78.34, 78.34, 78.34], [77.9099, 77.9099, 77.9099], False),
])
def test_gate_p_at_bar(tmp_path, mirror, svd, expect):
    gate_runs(tmp_path, mirror, svd)
    rec = C.gate_p_record(C.Final(tmp_path, n_val=N))
    assert rec["pass"] is expect and rec["status"] == (C.PASS if expect else C.FAIL)
    assert rec["arms"]["mirror_exact_astar"]["n"] == 3 and rec["bars"]["svdllm"]["target"] == 78.06
    assert all(s["git_hash"] == GIT for s in rec["sources"])


def test_gate_p_script_pending_then_decided(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    mod = load_script("fp_gate_p")
    out = tmp_path / "results" / "A01" / "gate_p.json"
    gate_runs(tmp_path, [78.34, 78.30], [78.06, 78.06, 78.06])                  # Mirror seed 2 missing
    assert mod.main(["--root", str(tmp_path), "--wait-minutes", "0", "--n-val", str(N)]) == 3
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["pass"] is None and rec["status"] == C.PENDING and "s2 missing" in rec["reason"]
    run(tmp_path, "main", M, "mirror_exact_astar", 2, 77.0)                      # mean 77.88: outside the bar
    log = tmp_path / "notes" / "decisions.md"
    assert mod.main(["--root", str(tmp_path), "--wait-minutes", "0", "--n-val", str(N),
                     "--decisions-log", str(log)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["pass"] is False and rec["status"] == C.FAIL
    assert rec["arms"]["mirror_exact_astar"]["seeds"] == {"0": 78.34, "1": 78.30, "2": 77.0}
    assert "Gate P FAIL" in log.read_text(encoding="utf-8")
    from mirror.final.run.runner import Condition, eval_condition       # the runner's stop_when reads this file
    assert eval_condition(Condition("results/A01/gate_p.json", "pass", "equals", False), tmp_path) is True
    # a pending file of an earlier attempt is removed before waiting: nothing sits at the done marker while waiting
    wj(out, {"pass": None, "stale": True})
    shutil.rmtree(tmp_path / "runs" / "final" / "main" / M / "mirror_exact_astar" / "s2")
    assert mod.main(["--root", str(tmp_path), "--check", "--n-val", str(N)]) == 3
    assert json.loads(out.read_text(encoding="utf-8")).get("stale") is True        # --check writes nothing
    seen = []

    class FakeTime:
        def __init__(self):
            self.t = 1000.0

        def time(self):
            return self.t

        def sleep(self, s):
            seen.append(out.exists())
            self.t += 61.0

        def __getattr__(self, k):
            return getattr(time, k)

    monkeypatch.setattr(mod, "time", FakeTime())
    assert mod.main(["--root", str(tmp_path), "--wait-minutes", "2", "--n-val", str(N)]) == 3
    assert seen == [False, False]
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["pass"] is None and "stale" not in rec


def a01_runs(root: Path, zs: float, w: list, a: list) -> None:
    for s, x in enumerate(w):
        run(root, "main", M, "mirror_exact_wspace", s, x, step0=zs)
    for s, x in enumerate(a):
        run(root, "main", M, "mirror_exact_astar", s, x)


@pytest.mark.parametrize("zs,w,a,adopted", [
    (58.84, [78.49, 78.49], [78.34, 78.34], True),       # zero-shot and healed difference exactly at their bars
    (58.83, [78.60, 78.60], [78.34, 78.34], False),       # zero-shot just below
    (60.00, [78.48, 78.49], [78.34, 78.34], False),       # healed difference 0.145
])
def test_a01_at_bar(tmp_path, zs, w, a, adopted):
    a01_runs(tmp_path, zs, w, a)
    rec = C.a01_record(C.Final(tmp_path, n_val=N))
    assert rec["adopted"] is adopted
    assert rec["main_arm"] == ("mirror_exact_wspace" if adopted else "mirror_exact_astar")
    assert rec["structure"] == C.STRUCTURES["wspace" if adopted else "astar"]


def test_a01_decide_script(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    mod = load_script("fp_a01_decide")
    out = tmp_path / "results" / "A01" / "decision.json"
    a01_runs(tmp_path, 60.0, [78.6], [78.34, 78.34])                            # A*_w seed 1 missing: pending
    assert mod.main(["--root", str(tmp_path), "--wait-minutes", "0", "--n-val", str(N)]) == 3
    assert not out.exists() and (tmp_path / "results" / "A01" / "decision_pending.json").is_file()
    run(tmp_path, "main", M, "mirror_exact_wspace", 1, 78.6)
    assert mod.main(["--root", str(tmp_path), "--wait-minutes", "0", "--n-val", str(N)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["adopted"] is True and rec["main_arm"] == "mirror_exact_wspace"
    assert rec["structure"] == f"results/A01/structure/{M}_wspace_matched.npz"
    assert not (tmp_path / "results" / "A01" / "decision_pending.json").exists()
    # a failed part decides "not adopted" at once, even while the healed seeds are missing
    other = tmp_path / "other"
    a01_runs(other, 50.0, [], [])
    run(other, "main", M, "mirror_exact_wspace", 0, None, step0=50.0, complete=False)
    assert mod.main(["--root", str(other), "--wait-minutes", "0", "--n-val", str(N)]) == 0
    rec = json.loads((other / "results" / "A01" / "decision.json").read_text(encoding="utf-8"))
    assert rec["adopted"] is False and rec["structure"] == f"results/MUST-07/structure/{M}_s0.npz"
    from mirror.final.heal.students import resolve_structure        # the healing code reads the same file
    assert resolve_structure("adopted", other / "results" / "A01" / "decision.json").as_posix().endswith(
        f"results/MUST-07/structure/{M}_s0.npz")


# ------------------------------------------------------------------------------------------------
# Every rule: PASS, FAIL, PENDING and NOT RUN
# ------------------------------------------------------------------------------------------------
def test_every_rule_passes(tmp_path):
    root = build(tmp_path)
    payload, rules = collect(root)
    assert set(rules) == set(C.RULE_ORDER)
    bad = {k: (r["verdict"], r["reason"]) for k, r in rules.items()
           if r["verdict"] != (C.REPORTED if k in REPORT_ONLY else C.PASS)}
    assert not bad, bad
    assert payload["main_arm"]["variant"] == "wspace"
    assert payload["counts"][C.PASS] == payload["counts"]["scored"] == len(C.RULE_ORDER) - len(REPORT_ONLY)
    nums = json.loads((root / "results" / "final" / "numbers.json").read_text(encoding="utf-8"))["numbers"]
    for key in ("gate_p.mirror_exact_astar.knn_mean", "a05.mirror_minus_best.ci_low", "a06.f065.predicted_gain",
                "a07.mirror_minus_best_flipavg", "a10.fullbank_knn_top1.mirror_mean", "p11_2.vexact_s0.max_rel_margin"):
        e = nums[key]
        assert {"value", "unit", "n", "description", "source", "commit"} <= set(e) and e["source"]
    assert nums["gate_p.mirror_exact_astar.knn_mean"]["commit"] == GIT
    assert nums["a05.best_baseline"]["value"] == "svdllm" and nums["a05.superiority"]["value"] is True
    text = (root / "results" / "FINAL_PHASE_REPORT.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and "## Headline numbers" in text and "## Deviations logged after" in text
    for t in ("T1_main_comparison", "T2_stage_ablation", "T3_other_models", "T4_operator_stability",
              "T5_discovery_controls", "T6_systems", "T7_exactness_audit", "T8_standard_metrics"):
        df = pd.read_csv(root / "results" / "final" / "tables" / f"{t}.csv")
        assert len(df) and "source_file" in df and (root / "results" / "final" / "tables" / f"{t}.md").is_file()
    t1 = pd.read_csv(root / "results" / "final" / "tables" / "T1_main_comparison.csv")
    svd = t1[t1["folder"] == "svdllm"].iloc[0]
    assert svd["ci_low"] > 0 and abs(svd["mean"] - 78.06) < 1e-9


def test_every_scored_rule_fails(tmp_path):
    root = build(tmp_path, fail=True)
    payload, rules = collect(root, args=["--no-t1-intervals"])
    assert payload["main_arm"]["variant"] == "astar"
    bad = {k: (r["verdict"], r["reason"]) for k, r in rules.items()
           if r["verdict"] != (C.REPORTED if k in REPORT_ONLY else C.FAIL)}
    assert not bad, bad
    assert payload["counts"][C.FAIL] == payload["counts"]["scored"]


def test_empty_tree_is_pending(tmp_path):
    payload, rules = collect(tmp_path)
    assert {r["verdict"] for r in rules.values()} == {C.PENDING}
    assert all(r["reason"] for k, r in rules.items() if k not in ("A10",)), {k: r["reason"] for k, r in rules.items()}
    assert payload["counts"][C.PENDING] == payload["counts"]["scored"]


def test_cut_jobs_are_not_run(tmp_path):
    root = build(tmp_path)
    for rel in (f"runs/final/a08/{C.LARGE}", f"runs/final/a11/{M}", "results/A15a", "results/A15b", "results/A12b",
                "runs/final/A10/p112"):
        shutil.rmtree(root / rel)
    shutil.rmtree(root / "runs" / "final" / "a06" / M / "svdllm_f050" / "s1")
    shutil.rmtree(root / "runs" / "final" / "a06" / M / "mirror_svdllm_f050" / "s1")
    wj(root / "jobs" / "heal" / "a08_l.json", {"group": "a08", "model": C.LARGE, "seed": 0,
                                               "students": [{"method": "mirror_exact"}, {"method": "svdllm"}]})
    wj(root / "jobs" / "heal" / "a06_f050_s1.json", {"group": "a06", "model": M, "seed": 1, "students": [
        {"method": "svdllm", "variant": "f050"}, {"method": "mirror_svdllm", "variant": "f050"}]})
    jobs = [{"id": "a08_l", "command": "python scripts/fp_heal.py --spec jobs/heal/a08_l.json",
             "done": f"runs/final/a08/{C.LARGE}/_jobs/x/s0/DONE"},
            {"id": "a06_f050_s1", "command": "python scripts/fp_heal.py --spec jobs/heal/a06_f050_s1.json",
             "done": f"runs/final/a06/{M}/_jobs/y/s1/DONE"},
            {"id": "a11_mirror", "command": "python scripts/fp_heal.py --spec jobs/heal/a11.yaml",
             "done": f"runs/final/a11/{M}/mirror_exact_ext/s0/DONE"},
            {"id": "a11_svdllm", "command": "python scripts/fp_heal.py --spec jobs/heal/a11s.yaml",
             "done": f"runs/final/a11/{M}/svdllm_ext/s0/DONE"},
            {"id": "a15a", "command": "python scripts/fp_a15a_siglip.py --out results/A15a",
             "done": "results/A15a/metrics.json"},
            {"id": "a15b_measure", "command": "python scripts/fp_a15b_measure.py", "done": "results/A15b/metrics.json"},
            {"id": "a12b_timing", "command": "python scripts/a12b.py", "done": "results/A12b/metrics.json"},
            {"id": "p112", "command": "python scripts/tier2_ade20k.py --mask-eq --out runs/final/A10/p112",
             "done": "runs/final/A10/p112/t2_ade20k.csv"}]
    (root / "jobs").mkdir(exist_ok=True)
    import yaml
    (root / "jobs" / "final_phase.yaml").write_text(yaml.safe_dump({"version": 1, "jobs": jobs}), encoding="utf-8")
    state = root / "logs" / "final" / "state"
    wj(state / "status.json", {"states": {"a08_l": "cut", "a06_f050_s1": "cut", "a11_svdllm": "cut",
                                          "a15a": "cut", "a15b_measure": "cut", "a12b_timing": "budget",
                                          "a11_mirror": "pending", "p112": "terminated"}})
    wj(state / "cuts.json", {"cuts": [{"rank": 1, "jobs": ["a15a", "a15b_measure"]}, {"rank": 2, "jobs": ["a08_l"]},
                                      {"rank": 3, "jobs": ["a11_svdllm"]}, {"rank": 5, "jobs": ["a06_f050_s1"]}],
                             "jobs": ["a15a", "a15b_measure", "a08_l", "a11_svdllm", "a06_f050_s1"]})
    (state / "a11_mirror.skip").write_text("", encoding="utf-8")
    _, rules = collect(root, args=["--no-t1-intervals"])
    cut = ("A08-2", "A06-1", "A06-2", "A11", "A15a", "A15b", "A12b", "P11.2")
    for rid in cut:
        assert rules[rid]["verdict"] == C.NOT_RUN, (rid, rules[rid]["verdict"], rules[rid]["reason"])
    assert "rank 2" in rules["A08-2"]["reason"] and "budget governor" in rules["A12b"]["reason"]
    assert "ceiling" in rules["P11.2"]["reason"] and "skipped by hand" in rules["A11"]["reason"]
    others = {k: r["verdict"] for k, r in rules.items() if k not in cut}
    assert set(others.values()) <= {C.PASS, C.REPORTED}, others
    text = (root / "results" / "FINAL_PHASE_REPORT.md").read_text(encoding="utf-8")
    assert "Jobs not run" in text and "a08_l" in text


def test_shallow_output_paths_do_not_mark_results_as_cut(tmp_path):
    import yaml
    jobs = [{"id": "report", "command": "python scripts/collect_final_phase.py", "done": "results/FINAL_PHASE_REPORT.md"},
            {"id": "a03", "command": "python scripts/fp_a03_controls.py --out results/A03", "done": "results/A03/x.json"}]
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "final_phase.yaml").write_text(yaml.safe_dump({"jobs": jobs}), encoding="utf-8")
    wj(tmp_path / "logs" / "final" / "state" / "status.json", {"states": {"report": "cut", "a03": "budget"}})
    c = C.Final(tmp_path, n_val=N)
    assert c.cuts.prefix_state("results/A15a") is None
    assert c.cuts.prefix_state("results/A03")[1] == "a03"


def test_recorded_decision_is_used_and_checked(tmp_path):
    root = build(tmp_path)
    wj(root / "results" / "A01" / "decision.json", {"adopted": False, "main_arm": "mirror_exact_astar"})
    payload, rules = collect(root, args=["--no-t1-intervals"])
    assert payload["main_arm"]["variant"] == "astar" and "decision.json" in payload["main_arm"]["how"]
    assert any(n.startswith("WARNING") and "decision.json" in n for n in payload["notes"])
    assert rules["A01"]["verdict"] == C.PASS                     # the rule itself is still recomputed and reported


def test_missing_and_incomplete_inputs_are_pending(tmp_path):
    root = build(tmp_path)
    shutil.rmtree(root / "runs" / "final" / "main" / M / "flap")                      # A04-1: one student missing
    m = json.loads((root / "runs/final/main" / M / "pair_random_h1/s0/metrics.json").read_text(encoding="utf-8"))
    m["complete"] = False                                                              # A03-2: a heal still running
    wj(root / "runs/final/main" / M / "pair_random_h1/s0/metrics.json", m)
    (root / "runs/final/main" / M / "iosvd/s2/eval/step780.json").unlink()            # A05-1: best baseline unknown
    (root / "results" / "A08" / C.DEIT / "params_check.json").unlink()                # A08-1: no parameter check
    _, rules = collect(root, args=["--no-t1-intervals"])
    for rid, part in (("A04-1", "flap/s0 missing"), ("A03-2", "not finished"), ("A05-1", "best baseline not known"),
                      ("A08-1", "no parameter check")):
        assert rules[rid]["verdict"] == C.PENDING and part in rules[rid]["reason"], (rid, rules[rid]["reason"])


# ------------------------------------------------------------------------------------------------
# A05-1: the paired-by-image hierarchical bootstrap
# ------------------------------------------------------------------------------------------------
def test_a05_uses_the_pre_registered_bootstrap(tmp_path, monkeypatch):
    root = build(tmp_path)
    calls = []
    real = HB.hierarchical_bootstrap

    def spy(runs, **kw):
        calls.append((list(runs), dict(kw)))
        return real(runs, **kw)

    monkeypatch.setattr(HB, "hierarchical_bootstrap", spy)
    c = C.Final(root, n_val=N)
    C.score_a05(c)
    r = next(x for x in c.rules if x.id == "A05-1")
    assert r.verdict == C.PASS
    runs, kw = calls[0]
    assert kw["n_resamples"] == 10_000 and kw["seed"] == 0 and kw["alpha"] == 0.05
    assert kw["paired_seeds"] is False and kw["pairs"] == [("mirror", "svdllm")]
    assert len(kw["labels"]) == len(runs) and all(np.array_equal(x, kw["labels"][0]) for x in kw["labels"])
    mir = [(x.fit_seed, x.heal_seed) for x in runs if x.method == "mirror"]
    assert mir == [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
    assert [(x.fit_seed, x.heal_seed) for x in runs if x.method == "svdllm"] == [(0, 0), (0, 1), (0, 2)]
    # images paired: one draw per resample shared by both methods (the interval equals a direct call)
    direct = real(runs, n_resamples=10_000, seed=0, alpha=0.05, pairs=[("mirror", "svdllm")], paired_seeds=False)
    assert c.boot["A05-1"]["pairs"][0]["ci_low"] == direct["pairs"][0]["ci_low"]
    # the point estimate is the mean over fit sets of the fit-set means (PREREG 2.5), not the run mean
    point = np.mean([np.mean([78.60, 78.55, 78.50]), 78.55, 78.50])
    assert abs(c.numbers["a05.mirror_point"]["value"] - point) < 1e-12


# ------------------------------------------------------------------------------------------------
# Outputs: PAPER_NUMBERS.json keeps every key; deviations after the pre-registration commit
# ------------------------------------------------------------------------------------------------
def test_paper_numbers_keeps_existing_keys(tmp_path):
    root = build(tmp_path)
    shutil.copyfile(ROOT / "PAPER_NUMBERS.json", root / "PAPER_NUMBERS.json")
    before = json.loads((root / "PAPER_NUMBERS.json").read_text(encoding="utf-8"))
    collect(root, args=["--no-t1-intervals"])
    collect(root, args=["--no-t1-intervals"])                                       # a rerun replaces its own key
    after = json.loads((root / "PAPER_NUMBERS.json").read_text(encoding="utf-8"))
    assert set(after) == set(before) | {"final_phase"}
    assert all(after[k] == before[k] for k in before)
    fp = after["final_phase"]
    assert fp["counts"]["scored"] > 0 and fp["numbers"]["gate_p.svdllm.knn_mean"]["commit"] == GIT
    assert {r["id"] for r in fp["rules"]} == set(C.RULE_ORDER)


def test_deviations_after_the_prereg_record(tmp_path):
    shutil.copyfile(ROOT / "PREREG_FINAL_PHASE.md", tmp_path / "PREREG_FINAL_PHASE.md")
    wj(tmp_path / "results" / "final" / "prereg_record.json",
       {"file": "PREREG_FINAL_PHASE.md", "sha256": C.sha256_file(tmp_path / "PREREG_FINAL_PHASE.md"),
        "commit": "0" * 40, "recorded_utc": "2026-11-20T10:00:00Z"})
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "decisions.md").write_text(
        "| Date | Decision | Reason | Effect | By |\n|---|---|---|---|---|\n"
        "| 2026-11-19 | before the commit | r | e | x |\n| 2026-11-21 | after the commit | r | e | x |\n",
        encoding="utf-8")
    c = C.Final(tmp_path, n_val=N)
    assert c.prereg["unchanged"] is True and c.prereg["rows"]["Gate P"][1].startswith("Mirror (current A*)")
    rows, how = C.deviations(c)
    assert len(rows) == 1 and "after the commit" in rows[0] and "2026-11-20" in how
    _, rules = collect(tmp_path)
    text = (tmp_path / "results" / "FINAL_PHASE_REPORT.md").read_text(encoding="utf-8")
    assert "after the commit" in text and "before the commit" not in text
    assert all(r["quote_found"] is not False for r in rules.values())

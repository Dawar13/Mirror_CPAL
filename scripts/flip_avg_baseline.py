#!/usr/bin/env python
"""Flip-averaging baseline: exact flip invariance by averaging at test time, measured with the paper protocol.

    python scripts/flip_avg_baseline.py --model dinov2_vitb14_reg --variants original,healed_mirror_k9,vlocal_k9 \
        --vexact-dir runs/m3_vexact_5090 --vexact-fp32 --out runs/t1_flipavg
    python scripts/flip_avg_baseline.py --tiny --variants original --out tiny_out/t1_flipavg     # CPU smoke test

Why: a reviewer can say that exact mirror invariance is trivial, since averaging the features of an image and
of its mirror at test time gives it for any model (at twice the compute). This script measures that baseline
with the protocol of every other number in the paper, so it can sit next to the healed V-exact model, which is
exactly mirror-invariant in one forward pass with fewer parameters.

Modes (per variant):
  plain       the paper's feature f(x), as every Tier-1 and Tier-2 number;
  flipavg     f_avg(x) = (f(x) + f(mirror x)) / 2, for the bank and the queries alike (and for the probe's
              training and test images alike);
  plain_fp32  healed V-exact at full conversion with --vexact-fp32 only: the plain k-NN with fp32 forward
              passes (TF32 off), so its exact invariance is measured on the full protocol and not only on
              the healing run's fp32 subset check. k-NN only, no probe.
A healed V-exact model at full conversion (healed_vexact_k<depth>) is exactly invariant already, so it gets
plain (and plain_fp32), never flipavg. Every other variant gets plain and flipavg.

Feature definitions (where the average is taken):
  k-NN    f = the CLS token after the final norm (adapter.cls_feature, the k-NN feature everywhere else).
          The flip average is taken on that vector BEFORE the L2 normalisation; mirror.knn.extract_features
          then normalises the averaged vector, exactly where it normalises every other feature. This is the
          feature the healed V-exact student at full conversion was distilled to
          (mirror.heal_vexact.VExactTeacher.final_targets: (f(x) + aligned f(mirror x)) / 2 after the final
          norm), and its even head (I + A*)/2 acts at the same place, so the flip-averaged original is that
          student's teacher. The factor 1/2 does not change cosine similarity.
  probe   [CLS after the final norm, mean patch token after the final norm] (tier2_linear.probe_feature),
          averaged as a whole: (p(x) + p(mirror x)) / 2. The probe standardises features, so the 1/2 does
          not matter.
  The two passes are separate forward passes over the same batches as the plain mode (the mirrored batch
  is its own batch), so f(x) inside f_avg is bit for bit the plain-mode feature, and because floating-point
  addition is commutative, f_avg(mirror x) equals f_avg(x) bit for bit in any precision (bf16 included).

k-NN (Tier 1, PAPER_EXPERIMENTS.md Section 1): mirror.protocol.Protocol (bank = 100 ImageNet train images per
class, 100k; queries = all 50k validation images), batch 128, bf16 autocast, k = 20, T = 0.07, cosine.
knn_eval below makes the calls of mirror.knn.evaluate_knn in the same order (extract_features for the bank,
the queries and the mirrored queries; knn_predict twice; a CPU test checks that they agree), and also
records how far the unit feature of each query is from that of its mirror (largest absolute difference, and
the share of queries whose two features are identical bit for bit). For flipavg the adapter is wrapped
(FlipAveragedAdapter), so the k-NN code itself is unchanged. Flip consistency is measured, not assumed: the
mirrored query's feature is computed from mirror x, i.e. (f(mirror x) + f(x)) / 2.

Linear probe (Tier 2): the recipe of scripts/tier2_linear.py unchanged: tier2_linear.extract_features (bf16,
batch 256, float16 storage; flipavg passes the flip-averaged probe feature through its feature_fn hook),
run_probe (weight decay chosen on a fixed 10% split of the bank, never on validation; refit on the bank;
top-1 on the 50k validation images) and knn_check (k-NN on the CLS half of the cached features).

Variants: built by scripts/tier2_linear.VariantFactory with the same names (original, healed_mirror_k9,
vlocal_k9, ...; see that script). --vexact-dir adds healed_vexact_k<K> (K = --vexact-k, default the depth,
i.e. the full conversion), checked after loading as tier2_linear checks it (span exactness and image versus
mirror probe feature in fp32, or the run stops).

Cost: forward passes per image (1 plain, 2 flipavg); analytic GFLOPs per image = forward passes x the
analytic cost of one forward pass from scripts/collect_results.analytic_cost (the cost model of the paper
tables, at the evaluation size: matrix multiply-adds of every block, the converted ones at their kept
parameter counts, one d x d basis rotation per converted block for per-layer operators (Mirror V-local and
random structure), none for V-exact and SVD; not counted: norms, softmax, GELU, the flip and the average, and
the V-exact even head, one d x d product on the CLS token). Parameters: the variant's whole-model count
(tier2_linear's count, as in m1_knn.csv) and its share of the original.

Outputs under --out: flipavg.csv (one row per model, variant and mode), FLIPAVG.md, flipavg_meta_<model>.json,
rows/<model>/<variant>__<mode>.json (k-NN and probe parts with their settings, variant provenance, git hash
and seed), rows/<model>/<variant>_variant.json, knn/<model>/<variant>__<mode>_pred.npz (predictions for the
queries and their mirrors, labels), features/<model>/<variant>__<mode>_<bank|val>.npz and
probe/<model>/<variant>__<mode>_probe.npz. Resumable per (variant, mode): a finished k-NN or probe part with
the same settings is skipped on a rerun (--force redoes them).
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import mirror  # noqa: E402
from mirror.knn import extract_features as knn_features  # noqa: E402
from mirror.knn import knn_predict  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

MEMORY_FRACTION = 0.2
DEFAULT_VARIANTS = "original,healed_mirror_k9,vlocal_k9"


def _load_tier2():
    """scripts/tier2_linear.py as a module (registered once; its import sets its own GPU cap, which main
    replaces with this script's cap through tier2_linear.apply_memory_cap)."""
    key = "_flipavg_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod          # dataclasses need the module registered while it executes
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


T2 = _load_tier2()
T2.apply_memory_cap(MEMORY_FRACTION)

KNN_FEATURE = ("CLS token after the final norm; flipavg averages it over the image and its mirror after the final "
               "norm and before the L2 normalisation of the k-NN (mirror.knn.extract_features normalises the "
               "averaged vector)")
PROBE_FEATURE = ("[CLS after the final norm, mean patch token after the final norm] (tier2_linear.probe_feature); "
                 "flipavg averages the whole vector over the image and its mirror")
FLIPAVG_RECIPE = T2.FEATURE_RECIPE + "; flip-averaged: (p(x) + p(mirror x)) / 2"

MODES = {
    "plain": {"label": "plain (paper feature)", "flip_average": False, "fp32": False, "probe": True},
    "flipavg": {"label": "flip-averaged at test time", "flip_average": True, "fp32": False, "probe": True},
    "plain_fp32": {"label": "plain, fp32 forward passes", "flip_average": False, "fp32": True, "probe": False},
}
MODE_ORDER = {m: i for i, m in enumerate(("plain", "plain_fp32", "flipavg"))}


# ----------------------------------------------------------------------------------------------
# Flip-averaged features
# ----------------------------------------------------------------------------------------------
class FlipAveragedAdapter:
    """The adapter with the flip-averaged k-NN feature; everything else is delegated to `base`.

    cls_feature(x) = (base.cls_feature(x) + base.cls_feature(mirror x)) / 2: the CLS token after the final
    norm, averaged before any L2 normalisation (mirror.knn.extract_features normalises afterwards). Two
    separate forward passes, the first on x itself, so the plain feature inside the average is unchanged."""

    def __init__(self, base):
        self.__dict__["base"] = base

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def cls_feature(self, x: torch.Tensor, model=None) -> torch.Tensor:
        base = self.__dict__["base"]
        return 0.5 * (base.cls_feature(x, model).float() + base.cls_feature(mirror(x), model).float())


def flip_averaged_probe_feature(adapter, x: torch.Tensor, model=None) -> torch.Tensor:
    """(p(x) + p(mirror x)) / 2 with p = tier2_linear.probe_feature ([CLS, mean patch token], float32)."""
    return 0.5 * (T2.probe_feature(adapter, x, model) + T2.probe_feature(adapter, mirror(x), model))


@torch.no_grad()
def knn_eval(adapter, bank_loader, query_loader, device, num_classes: int, k: int = 20, temperature: float = 0.07,
             model=None, amp: str | None = None, flip_average: bool = False) -> dict:
    """mirror.knn.evaluate_knn's computation (the same calls in the same order) with the flip-averaged feature
    when `flip_average`, plus how far the unit feature of each query is from that of its mirror."""
    ad = FlipAveragedAdapter(adapter) if flip_average else adapter
    bf, by = knn_features(ad, bank_loader, device, model, amp=amp)
    qf, qy = knn_features(ad, query_loader, device, model, amp=amp)
    qf_flip, _ = knn_features(ad, query_loader, device, model, flip=True, amp=amp)
    pred = knn_predict(bf, by, qf, num_classes, k, temperature)
    pred_flip = knn_predict(bf, by, qf_flip, num_classes, k, temperature)
    diff = (qf_flip - qf).abs().amax(dim=1)
    return {
        "top1": 100.0 * float((pred == qy).float().mean()),
        "flip_consistency": 100.0 * float((pred == pred_flip).float().mean()),
        "n_bank": int(bf.shape[0]),
        "n_query": int(qf.shape[0]),
        "feature_flip_max_abs": float(diff.max()),
        "feature_identical_pct": 100.0 * float((diff == 0).float().mean()),
        "pred": pred.cpu(),
        "pred_flip": pred_flip.cpu(),
        "labels": qy.cpu(),
    }


# ----------------------------------------------------------------------------------------------
# Cost (scripts/collect_results.analytic_cost)
# ----------------------------------------------------------------------------------------------
def cost_arch(adapter, model_params_dense: int) -> dict:
    """Inputs of the collector's cost model for this adapter at its evaluation size."""
    H, W = adapter.grid_hw
    proj = getattr(getattr(adapter.model, "patch_embed", None), "proj", None)
    return {"d": int(adapter.embed_dim), "h": int(adapter.mlp_hidden), "depth": int(adapter.depth),
            "T": int(adapter.n_tokens), "n_patches": int(H * W), "patch": int(adapter.patch_size),
            "in_chans": int(getattr(proj, "in_channels", 3)), "model_params_dense": int(model_params_dense)}


def rotations_per_block(method: str) -> int:
    """d x d basis rotations per converted block (collector's COST_ASSUMPTIONS): per-layer operators (Mirror
    V-local, MLP only or with attention, and the random-structure controls) pay one; V-exact and SVD none."""
    return 1 if method in ("mirror", "vlocal", "random") else 0


def _split(total: int, k: int) -> list:
    q, r = divmod(int(total), int(k))
    return [q + (1 if i < r else 0) for i in range(int(k))]


def conv_blocks(info: dict, arch: dict) -> list:
    """Per-block costs for analytic_cost from the variant's kept parameter counts (tier2_linear's
    VariantFactory info). analytic_cost depends only on the number of converted blocks and the totals of
    their MLP, attention and rotation counts, so the totals are split evenly (integers summing exactly)."""
    k = int(info.get("k") or 0)
    if info.get("family") == "original" or k == 0:
        return []
    d, h = arch["d"], arch["h"]
    mlp = int(info.get("mlp_params_kept", 2 * h * d * k))
    attn = int(info.get("attn_params_kept", 4 * d * d * k))
    rot = rotations_per_block(str(info.get("method")))
    return [{"mlp": m_, "attn": a_, "rot": rot} for m_, a_ in zip(_split(mlp, k), _split(attn, k))]


def variant_cost(info: dict, arch: dict) -> dict:
    cr = T2.load_script("collect_results")
    c = cr.analytic_cost(arch, conv_blocks(info, arch))
    return {"gflops_per_forward": float(c["gflops_model"]), "macs_per_forward": int(c["macs_model"]),
            "cost_model_params": c["model_params"], "kept_frac_converted": float(c["kept_frac_converted"])}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def modes_for(v, depth: int, vexact_fp32: bool = False) -> list:
    """plain and flipavg for every variant, except a full-conversion healed V-exact model (exactly invariant
    already): plain, plus plain_fp32 with --vexact-fp32."""
    if v.family == "healed_vexact" and int(v.k) == int(depth):
        return ["plain"] + (["plain_fp32"] if vexact_fp32 else [])
    return ["plain", "flipavg"]


def row_json_path(out: Path, model_name: str, variant: str, mode: str) -> Path:
    return Path(out) / "rows" / model_name / f"{variant}__{mode}.json"


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _same(a, b) -> bool:
    return T2._jsonable(a) == T2._jsonable(b)


def vexact_flipcheck(info: dict) -> dict | None:
    """The healing run's own fp32 flip check (scripts/heal_vexact.py flipcheck.json), if present."""
    run = info.get("vexact_run")
    if not run:
        return None
    j = _read_json(Path(run) / "flipcheck.json")
    if not j:
        return None
    return {k: j.get(k) for k in ("flip_consistency", "pass", "n_bank", "n_query", "cls_feature_rel_max", "dtype")}


def row_notes(mode: str, info: dict, knn: dict | None) -> str:
    notes = []
    if mode == "flipavg":
        notes.append("f_avg(x) = (f(x) + f(mirror x)) / 2 for bank and queries; k-NN: CLS after the final norm averaged "
                     "before the L2 normalisation; probe: [CLS, mean patch] averaged")
    elif mode == "plain_fp32":
        notes.append("fp32 forward passes (TF32 off), k-NN only")
    if info.get("family") == "healed_vexact":
        extra = T2.variant_row_extras(info).get("notes")
        if extra:
            notes.append(extra)
        fc = info.get("healing_flipcheck")
        if fc and fc.get("flip_consistency") is not None:
            notes.append(f"healing run's fp32 flip check: {fc['flip_consistency']:.2f}% on {fc.get('n_query')} queries "
                         f"against {fc.get('n_bank')} bank images")
    if knn and mode != "flipavg" and knn.get("flip_consistency", 0) < 100.0 and info.get("family") == "healed_vexact" \
            and info.get("even_head"):
        notes.append("exactly invariant in exact arithmetic; the image and its mirror run through different "
                     "computations, which round differently in this precision")
    return "; ".join(notes)


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
CSV_COLUMNS = ["model", "variant", "label", "family", "method", "k", "mode", "mode_label", "knn_top1", "flip_consistency",
               "probe_top1", "probe_top5", "forwards_per_image", "gflops", "gflops_per_forward", "params", "params_frac",
               "feature_flip_max_abs", "feature_identical_pct", "knn_check_top1", "chosen_wd", "wd_at_grid_edge",
               "knn_amp", "probe_amp", "n_bank", "n_query", "n_probe_bank", "n_probe_val", "t_knn_s", "t_extract_s",
               "t_probe_s", "seed", "git_hash", "git_dirty", "probe_git_hash", "date", "notes"]


def _sort_key(r: dict):
    fam = {"original": 0, "healed_vexact": 1}.get(r["family"], 2)
    return (r["model"], fam, int(r["k"]), r["variant"], MODE_ORDER.get(r["mode"], 9))


def write_tables(out: Path) -> list:
    """Rebuild flipavg.csv and FLIPAVG.md from every row JSON under <out>/rows/."""
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "rows").glob("*/*__*.json")):
        r = _read_json(p)
        if r and r.get("row"):
            rows.append(r["row"])
    if not rows:
        return rows
    rows.sort(key=_sort_key)
    df = pd.DataFrame(rows)
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + [c for c in df.columns if c not in CSV_COLUMNS]]
    df.to_csv(Path(out) / "flipavg.csv", index=False)
    write_summary(Path(out) / "FLIPAVG.md", rows)
    return rows


def _f(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _e(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return "0" if x == 0 else f"{x:.1e}"


def _short_notes(r: dict) -> str:
    bits = []
    if r["family"] == "healed_vexact":
        if r.get("vexact_checks_pass") is not None:
            bits.append(f"checked after loading (fp32): span exactness {_e(r.get('vexact_span_exact_rel_max'))}, image vs "
                        f"mirror probe feature {_e(r.get('vexact_feature_flip_rel_max'))}, "
                        f"{'PASS' if r.get('vexact_checks_pass') else 'FAIL'}")
        if r.get("healing_flip_fp32") is not None:
            bits.append(f"healing run's fp32 flip check {_f(r['healing_flip_fp32'])}%")
    if r["mode"] == "plain_fp32":
        bits.append("fp32 forward passes, k-NN only")
    if r.get("wd_at_grid_edge"):
        bits.append("probe wd at grid edge")
    return "; ".join(bits) or "-"


def write_summary(path: Path, rows: list) -> None:
    lines = [
        "# Flip-averaging baseline: exact flip invariance at test time", "",
        "Question: exact mirror invariance can also be had by averaging the features of an image and of its mirror "
        "at test time (two forward passes per image). This table measures that baseline with the protocol of every "
        "other number, next to the healed V-exact model (exactly mirror-invariant in one forward pass).", "",
        "Protocol. k-NN: bank of 100 ImageNet train images per class (100k), all 50k validation images as queries, "
        "CLS token after the final norm, cosine, k = 20, T = 0.07, bf16 forward passes unless the mode says fp32 "
        "(mirror.knn, as every Tier-1 number). Flip consistency: share of validation images whose k-NN prediction "
        "equals the prediction for their mirror image, measured. Identical features: share of validation images "
        "whose unit k-NN feature equals that of their mirror bit for bit. Linear probe: the Tier-2 recipe of "
        "scripts/tier2_linear.py ([CLS, mean patch token] after the final norm; logistic regression trained on the "
        "100k bank, weight decay chosen on a 10% split of the bank, tested on the 50k validation images).", "",
        "Flip averaging: f_avg(x) = (f(x) + f(mirror x)) / 2, for the bank and the queries alike (and for the "
        "probe's training and test images alike). The k-NN feature is averaged after the final norm and before the "
        "L2 normalisation, where the k-NN normalises every other feature; this is also the feature the healed "
        "V-exact student at full conversion was distilled to. The probe feature is averaged as a whole. The two "
        "forward passes use the same batches as the plain mode, so the average is the same for an image and its "
        "mirror bit for bit, in any precision.", "",
        "Cost: forward passes per image (1 plain, 2 flip-averaged). GFLOPs per image: forward passes times the "
        "analytic cost of one forward pass (scripts/collect_results.analytic_cost, the cost model of the paper "
        "tables; analytic counts, not measured latency). Parameters: whole model, with the share of the original.", ""]
    if any(r["model"] == "tiny" for r in rows):
        lines += ["Tiny rows: CPU smoke test (tiny random ViT, synthetic images); not the protocol above.", ""]
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        orig = next((r for r in rs if r["family"] == "original" and r["mode"] == "plain"), None)
        lines += [f"## {model}", "",
                  "| Variant | What it is | Mode | k-NN top-1 | Change vs original (plain) | Flip consistency (%) | "
                  "Identical features (%) | Linear probe top-1 | Forward passes per image | GFLOPs per image | "
                  "Parameters (share of original) | Notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            delta = "-" if orig is None or r is orig or r.get("knn_top1") is None else _f(r["knn_top1"] - orig["knn_top1"])
            params = (f"{int(r['params']):,} ({_f(r.get('params_frac'), 3)})" if r.get("params") is not None else "-")
            lines.append(f"| {r['variant']} | {r['label']} | {r['mode']} | {_f(r.get('knn_top1'))} | {delta} | "
                         f"{_f(r.get('flip_consistency'))} | {_f(r.get('feature_identical_pct'))} | "
                         f"{_f(r.get('probe_top1'))} | {r['forwards_per_image']} | {_f(r.get('gflops'))} | {params} | "
                         f"{_short_notes(r)} |")
        lines.append("")
        exact = [r for r in rs if (r["mode"] == "flipavg" and r["family"] == "original")
                 or (r["family"] == "healed_vexact" and r.get("vexact_even_head"))]
        if exact:
            lines += ["### Exactly mirror-invariant options", ""]
            for r in exact:
                lines.append(f"- {r['variant']} ({r['mode']}): k-NN {_f(r.get('knn_top1'))}, flip consistency "
                             f"{_f(r.get('flip_consistency'))}%, linear probe {_f(r.get('probe_top1'))}, "
                             f"{r['forwards_per_image']} forward pass{'es' if r['forwards_per_image'] > 1 else ''} per "
                             f"image, {_f(r.get('gflops'))} GFLOPs per image, {int(r['params']):,} parameters "
                             f"({_f(r.get('params_frac'), 3)} of the original).")
            fa = next((r for r in exact if r["mode"] == "flipavg"), None)
            for vx in (r for r in exact if r["family"] == "healed_vexact" and r["mode"] == "plain"):
                if fa and vx.get("gflops") and fa.get("gflops"):
                    bits = [f"{vx['variant']} uses {_f(vx['gflops'] / fa['gflops'], 3)} of the GFLOPs per image and "
                            f"{_f(vx['params'] / fa['params'], 3)} of the parameters of the flip-averaged original"]
                    diffs = []
                    if vx.get("knn_top1") is not None and fa.get("knn_top1") is not None:
                        diffs.append(f"k-NN {_f(vx['knn_top1'] - fa['knn_top1'])}")
                    if vx.get("probe_top1") is not None and fa.get("probe_top1") is not None:
                        diffs.append(f"linear probe {_f(vx['probe_top1'] - fa['probe_top1'])}")
                    if diffs:
                        bits.append("top-1 difference (V-exact minus flip-averaged original, points): " + ", ".join(diffs))
                    lines.append("- " + "; ".join(bits) + ".")
            lines.append("")
    git = sorted({str(r["git_hash"])[:10] + (" (dirty)" if r.get("git_dirty") else "") for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    lines += [f"Source: `flipavg.csv` and `rows/<model>/<variant>__<mode>.json` in this folder (predictions in `knn/`). "
              f"Commit(s): {', '.join(git)}; seed(s): {', '.join(map(str, seeds))}. Full notes, feature checks and "
              "timings are in the CSV.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--variants", default=DEFAULT_VARIANTS,
                   help="comma-separated variant names of scripts/tier2_linear.py (default: %(default)s)")
    p.add_argument("--out", default=None, help="output directory (default runs/t1_flipavg; tiny_out/t1_flipavg with --tiny)")
    p.add_argument("--operators", default=None, help="X2 e1_operators.npz (default per model, tier2_linear MODEL_DEFAULTS)")
    p.add_argument("--m1-structure", default=None, help="M1 m1_structure.npz (default per model)")
    p.add_argument("--heal-dir", default=None, help="M3 output directory with <method>/final/trained_blocks.pt")
    p.add_argument("--heal-config", default=None, help="fallback config when <heal-dir>/<method>/summary.json is missing")
    p.add_argument("--random-seed", type=int, default=1, help="matched random structure (MLP-only variants)")
    p.add_argument("--vexact-dir", default=None,
                   help="scripts/heal_vexact.py output directory (k<K>/final/trained_blocks.pt); adds healed_vexact_k<K>")
    p.add_argument("--vexact-k", type=int, default=None, help="K of the healed V-exact model (default: the depth)")
    p.add_argument("--vexact-config", default=None, help="fallback config when <vexact-dir>/k<K>/summary.json is missing")
    p.add_argument("--vexact-fp32", action="store_true",
                   help="also run the full-conversion V-exact k-NN with fp32 forward passes (mode plain_fp32)")
    p.add_argument("--seed", type=int, default=0, help="global seed and the probe's weight-decay split")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--knn-batch-size", type=int, default=128, help="k-NN batch size (128 for every Tier-1 number)")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--batch-size", type=int, default=256, help="probe feature batch size (256 as tier2_linear)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    p.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--wd-grid", default=T2.DEFAULT_WD_GRID, help="coarse weight-decay grid (tier2_linear's)")
    p.add_argument("--no-wd-refine", action="store_true")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--tol-grad", type=float, default=1e-6)
    p.add_argument("--probe-tf32", action="store_true")
    p.add_argument("--skip-probe", action="store_true", help="k-NN only")
    p.add_argument("--force", action="store_true", help="redo finished k-NN and probe parts and features")
    p.add_argument("--memory-fraction", type=float, default=None, help=f"override the {MEMORY_FRACTION} GPU cap")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    p.add_argument("--tiny-classes", type=int, default=4)
    p.add_argument("--tiny-bank-per-class", type=int, default=20)
    p.add_argument("--tiny-val-per-class", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    model_name = "tiny" if tiny else a.model
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/t1_flipavg" if tiny else "runs/t1_flipavg")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    T2.apply_memory_cap(a.memory_fraction if a.memory_fraction is not None else MEMORY_FRACTION)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    amp = None if a.amp == "none" else a.amp
    fdtype = np.float16 if a.feature_dtype == "float16" else np.float32
    grid = [float(x) for x in a.wd_grid.split(",") if x.strip()]
    solver = {"max_iter": a.max_iter, "history_size": a.history_size, "tol_grad": a.tol_grad}
    probe_settings = {"grid": grid, "refine": not a.no_wd_refine, "holdout_frac": a.holdout_frac, "seed": a.seed,
                      "solver": solver, "tf32": a.probe_tf32}
    variants = T2.parse_variants(a.variants)
    paths = T2.resolve_paths(a)
    log(f"flip-averaging baseline: model {model_name}, variants {[v.name for v in variants]}, out {out}, device "
        f"{device}, commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("load"):
        adapter = T2.build_adapter(a.model, tiny, device, a.img_size)
        if tiny:
            data = T2.TinyData(a.tiny_classes, a.tiny_bank_per_class, a.tiny_val_per_class, seed=a.seed,
                               batch_size=min(a.batch_size, 16))
            knn_bank, knn_query, protocol = data.loader("bank"), data.loader("val"), "tiny synthetic"
        else:
            from mirror.protocol import Protocol

            data = T2.ProtocolData(T2._resolve(a.cache_dir), a.batch_size, a.num_workers, device.type == "cuda")
            proto = Protocol(T2._resolve(a.cache_dir), batch_size=a.knn_batch_size, num_workers=a.num_workers,
                             pin=device.type == "cuda")
            knn_bank, knn_query, protocol = proto.bank_loader(), proto.query_loader(), "v2"
        if a.vexact_dir:
            vk = int(a.vexact_k or adapter.depth)
            if all(v.name != f"healed_vexact_k{vk}" for v in variants):
                variants.append(T2.parse_variant(f"healed_vexact_k{vk}"))
        if tiny and paths["operators"] is None and any(v.family in ("mlp", "m1") for v in variants):
            from mirror import variants as V   # tiny smoke path: fit operators in-run (as tier2_linear does)

            V.load_or_make_structure(None, True, adapter, V.TinyProtocol(), device, out)
            paths["operators"] = str(out / "tiny_fit" / "results" / "e1_operators.npz")
        factory = T2.make_factory(adapter, a, tiny, paths)
        kind = T2.feature_kind(adapter)
    n_model = int(sum(p.numel() for p in adapter.model.parameters()))
    arch = cost_arch(adapter, n_model)
    plan = [(v, modes_for(v, adapter.depth, a.vexact_fp32)) for v in variants]
    log(f"adapter: depth {adapter.depth}, d {adapter.embed_dim}, tokens {adapter.n_tokens}; k-NN feature: {KNN_FEATURE}; "
        f"plan {[(v.name, m) for v, m in plan]}")
    for v, _ in plan:                         # every input file of every variant exists before any work starts
        factory.sources(v)

    results, run_timing = [], {}
    for v, modes in plan:
        vt = StageTimer()
        log(f"=== {v.name}: {v.label}")
        sources = factory.sources(v)
        built = {"model": None, "info": None, "t_build": 0.0}
        vj = out / "rows" / model_name / f"{v.name}_variant.json"

        def get_model():
            if built["model"] is None:
                t0 = time.time()
                with vt.stage(f"build {v.name}"):
                    m, info = factory.build(v)
                if v.family == "healed_vexact":
                    info["healing_flipcheck"] = vexact_flipcheck(info)
                built.update(model=m, info=info, t_build=time.time() - t0)
                T2.write_json(vj, {"info": info, "sources": sources, "git": git})
            return built["model"]

        for mode in modes:
            spec = MODES[mode]
            rp = row_json_path(out, model_name, v.name, mode)
            rec = (None if a.force else _read_json(rp)) or {}
            knn_amp = None if spec["fp32"] else amp
            knn_settings = {"model": model_name, "variant": v.name, "mode": mode, "protocol": protocol,
                            "amp": knn_amp or "none", "k": a.knn_k, "temperature": a.knn_temperature,
                            "batch_size": knn_bank.batch_size, "img_size": adapter.img_size, "sources": sources,
                            "n_bank": len(knn_bank.dataset), "n_query": len(knn_query.dataset), "seed": a.seed}
            # ---- k-NN and flip consistency
            if not (rec.get("knn") and _same(rec.get("knn_settings"), knn_settings)):
                m = get_model()
                log(f"  {v.name} [{mode}]: k-NN ({'flip-averaged' if spec['flip_average'] else 'plain'} feature, "
                    f"{knn_amp or 'fp32'})")
                t0 = time.time()
                with vt.stage(f"knn {v.name} {mode}"):
                    r = knn_eval(adapter, knn_bank, knn_query, device, data.num_classes, a.knn_k, a.knn_temperature,
                                 model=m, amp=knn_amp, flip_average=spec["flip_average"])
                pp = out / "knn" / model_name / f"{v.name}__{mode}_pred.npz"
                pp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pp, pred=r["pred"].numpy().astype(np.int16),
                                    pred_flip=r["pred_flip"].numpy().astype(np.int16),
                                    labels=r["labels"].numpy().astype(np.int16))
                rec["knn"] = {k: r[k] for k in ("top1", "flip_consistency", "n_bank", "n_query", "feature_flip_max_abs",
                                                "feature_identical_pct")}
                rec["knn"].update({"seconds": round(time.time() - t0, 1), "git_hash": git["git_hash"],
                                   "git_dirty": git["git_dirty"], "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "predictions": str(pp), "env": env_info(device)})
                rec["knn_settings"] = knn_settings
                T2.write_json(rp, rec)
                log(f"  {v.name} [{mode}]: k-NN top-1 {r['top1']:.2f}, flip consistency {r['flip_consistency']:.3f}%, "
                    f"identical features {r['feature_identical_pct']:.2f}% (largest unit-feature difference "
                    f"{r['feature_flip_max_abs']:.2e})")
                if spec["flip_average"] and r["flip_consistency"] != 100.0:
                    log(f"WARNING: {v.name} [{mode}]: flip-averaged features gave flip consistency "
                        f"{r['flip_consistency']:.4f}%, not 100%: the forward pass is not deterministic here")
            else:
                log(f"  {v.name} [{mode}]: k-NN done before ({rp.name}): top-1 {rec['knn']['top1']:.2f}; skipping")
            # ---- linear probe
            if spec["probe"] and not a.skip_probe:
                expect = {"model": model_name, "variant": v.name, "mode": mode,
                          "recipe": FLIPAVG_RECIPE if spec["flip_average"] else T2.FEATURE_RECIPE, "amp": a.amp,
                          "dtype": a.feature_dtype, "sources": sources, "img_size": adapter.img_size,
                          "batch_size": data.loader("bank").batch_size}
                want = {**probe_settings, "features": expect}
                if not (rec.get("probe") and _same(rec.get("probe_settings"), want)):
                    fn = flip_averaged_probe_feature if spec["flip_average"] else None
                    feats, t_extract = {}, 0.0
                    for split in T2.SPLITS:
                        fp = Path(out) / "features" / model_name / f"{v.name}__{mode}_{split}.npz"
                        n = len(data.loader(split).dataset)
                        hit = None if a.force else T2.cached_features(fp, {**expect, "split": split, "n": n})
                        if hit is not None:
                            feats[split] = hit
                            log(f"  {split}: cached features {fp.name} {hit[0].shape}")
                            continue
                        m = get_model()
                        t0 = time.time()
                        with vt.stage(f"extract {v.name} {mode} {split}"):
                            X, y = T2.extract_features(adapter, data.loader(split), device, m, amp, fdtype,
                                                       tag=f"{mode} {split}", feature_fn=fn)
                        t_extract += time.time() - t0
                        meta = {**expect, "split": split, "n": int(X.shape[0]), "dim": int(X.shape[1]),
                                "feature_kind": kind, "git_hash": git["git_hash"], "seed": a.seed,
                                "seconds": round(time.time() - t0, 2)}
                        T2.save_features(fp, X, y, data.rows(split), meta)
                        feats[split] = (X, y, meta)
                    (Xb, yb, _), (Xv, yv, _) = feats["bank"], feats["val"]
                    torch.backends.cuda.matmul.allow_tf32 = bool(a.probe_tf32)
                    t0 = time.time()
                    with vt.stage(f"probe {v.name} {mode}"):
                        pr = T2.run_probe(Xb, yb, Xv, yv, data.num_classes, grid, a.holdout_frac, a.seed,
                                          not a.no_wd_refine, device, solver)
                        d_first = int(adapter.embed_dim) if kind == "cls" else int(Xb.shape[1] - adapter.embed_dim)
                        kc = T2.knn_check(Xb, yb, Xv, yv, d_first, data.num_classes, device)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    sel, fin, test = pr["selection"], pr["final"], pr["test"]
                    ppath = out / "probe" / model_name / f"{v.name}__{mode}_probe.npz"
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(ppath, W=pr["W"], b=pr["b"], mean=pr["mean"], std=pr["std"],
                             holdout_positions=pr["holdout_positions"])
                    rec["probe"] = {"top1": test["top1"], "top5": test["top5"],
                                    "mean_per_class_top1": test["mean_per_class_top1"], "knn_check_top1": kc,
                                    "chosen_wd": sel["chosen_wd"], "wd_at_grid_edge": sel["wd_at_grid_edge"],
                                    "holdout_top1_at_chosen": sel["holdout_top1_at_chosen"],
                                    "final_hit_iter_limit": fin["hit_limit"], "final_lbfgs_iters": fin["n_iter"],
                                    "feature_dim": int(Xb.shape[1]), "n_bank": int(Xb.shape[0]), "n_val": int(Xv.shape[0]),
                                    "seconds_extract": round(t_extract, 1), "seconds_probe": round(time.time() - t0, 1),
                                    "selection": sel, "final": fin, "git_hash": git["git_hash"],
                                    "git_dirty": git["git_dirty"], "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "probe_file": str(ppath)}
                    rec["probe_settings"] = want
                    T2.write_json(rp, rec)
                    log(f"  {v.name} [{mode}]: linear probe top-1 {test['top1']:.2f} (wd {sel['chosen_wd']:.3g}); "
                        f"k-NN check {kc:.2f}")
                    del feats, Xb, Xv, pr
                    T2.free_memory(device)
                else:
                    log(f"  {v.name} [{mode}]: probe done before: top-1 {rec['probe']['top1']:.2f}; skipping")
            # ---- row
            info = built["info"] or (_read_json(vj) or {}).get("info") or rec.get("variant_info") or {
                "variant": v.name, "label": v.label, "family": v.family, "method": v.method, "k": v.k}
            if v.family == "healed_vexact" and "healing_flipcheck" not in info:
                info["healing_flipcheck"] = vexact_flipcheck(info)
            rec["variant_info"] = info
            knn, probe = rec.get("knn") or {}, (rec.get("probe") if spec["probe"] and not a.skip_probe else None) or {}
            fw = 2 if spec["flip_average"] else 1
            cost = variant_cost(info, arch) if "model_params" in info else {}
            params = info.get("model_params")
            fc = info.get("healing_flipcheck") or {}
            row = {"model": model_name, "variant": v.name, "label": v.label, "family": v.family, "method": v.method,
                   "k": v.k, "mode": mode, "mode_label": spec["label"], "knn_top1": knn.get("top1"),
                   "flip_consistency": knn.get("flip_consistency"), "probe_top1": probe.get("top1"),
                   "probe_top5": probe.get("top5"), "forwards_per_image": fw,
                   "gflops": fw * cost["gflops_per_forward"] if cost else None,
                   "gflops_per_forward": cost.get("gflops_per_forward"), "params": params,
                   "params_frac": params / n_model if params else None,
                   "feature_flip_max_abs": knn.get("feature_flip_max_abs"),
                   "feature_identical_pct": knn.get("feature_identical_pct"),
                   "knn_check_top1": probe.get("knn_check_top1"), "chosen_wd": probe.get("chosen_wd"),
                   "wd_at_grid_edge": probe.get("wd_at_grid_edge"), "knn_amp": knn_settings["amp"],
                   "probe_amp": a.amp if probe else None, "n_bank": knn.get("n_bank"), "n_query": knn.get("n_query"),
                   "n_probe_bank": probe.get("n_bank"), "n_probe_val": probe.get("n_val"),
                   "t_knn_s": knn.get("seconds"), "t_extract_s": probe.get("seconds_extract"),
                   "t_probe_s": probe.get("seconds_probe"), "seed": a.seed, "git_hash": knn.get("git_hash"),
                   "git_dirty": knn.get("git_dirty"), "probe_git_hash": probe.get("git_hash"),
                   "date": knn.get("date"), "notes": row_notes(mode, info, knn),
                   "healing_flip_fp32": fc.get("flip_consistency"),
                   "cost_model_params": cost.get("cost_model_params"),
                   **{k_: v_ for k_, v_ in T2.variant_row_extras(info).items() if k_ != "notes"}}
            rec["row"] = row
            rec["features"] = {"knn": KNN_FEATURE, "probe": PROBE_FEATURE if spec["probe"] else None,
                               "flip_average": spec["flip_average"]}
            T2.write_json(rp, rec)
            results.append(row)
            write_tables(out)
        if built["model"] is not None and built["model"] is not adapter.model:
            built["model"] = None
        T2.free_memory(device)
        run_timing[v.name] = vt.stages

    rows = write_tables(out)
    T2.write_json(out / f"flipavg_meta_{model_name}.json", {
        "experiment": "flip-averaging baseline (exact flip invariance at test time)", "model": model_name,
        "plan": [{"variant": v.name, "modes": m} for v, m in plan], "args": vars(a), "paths": paths, "git": git,
        "env": env_info(device), "seed": a.seed, "protocol": protocol,
        "features": {"knn": KNN_FEATURE, "probe": PROBE_FEATURE, "probe_recipe_plain": T2.FEATURE_RECIPE,
                     "probe_recipe_flipavg": FLIPAVG_RECIPE},
        "cost": {"arch": arch, "source": "scripts/collect_results.analytic_cost (one forward pass) x forward passes"},
        "timing": {"stages": timer.stages, "variants": run_timing, "total_s": timer.total()}})
    log(f"done in {timer.total() / 60:.1f} min; {len(rows)} rows in {out / 'flipavg.csv'}")
    return {"out": str(out), "rows": results, "all_rows": rows}


if __name__ == "__main__":
    main()

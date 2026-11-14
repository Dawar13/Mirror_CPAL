#!/usr/bin/env python
"""MUST-09 (PROJECT_NEXT_PHASE.md): healed full-model comparison with one recipe for every method.

    python scripts/must09_heal.py --method mirror_exact --seed 0          # seeds 0, 1, 2
    python scripts/must09_heal.py --method svd --seed 0 --hparams-json results/MUST-07/hparams/svd.json
    python scripts/must09_heal.py --method random_structure --seed 0
    python scripts/must09_heal.py --method mirror_exact --seed 0 --latency-only --with-original   # quiet GPU
    python scripts/must09_heal.py --tiny --method mirror_exact                                    # CPU smoke test

Methods: mirror_exact and random_structure (the mirror family: mirror.mirror_family.build_mirror_variant when it
exists, else mirror.heal_vexact.build_vexact_student, the code path of the V-exact k = 12 run and of E3), and the
compressors of MUST-07 (mirror.baselines.METHODS_V2: svd, svdllm, asvd_scale, slicegpt, flap, zipit) at the
Mirror-exact budget of the full model (mirror.budget.mirror_exact_budget), compressed by mirror.baselines.compress
with the hyperparameters chosen in MUST-07 (--hparams-json). Until mirror/budget.py exists, `svd` runs through a
marked shim: the E3 SVD arm (ranks matched per block to V-exact's free-parameter counts).

Recipe (identical for every method; configs/must09.yaml, copied from configs/m3_vexact.yaml): see the config. Every
student trains only its own structured parameters plus the embeddings (mirror/heal_structured.make_trainable);
Mirror-exact re-projects after every step. Targets (--targets, default sym for every method): the final CLS and
patch targets are the flip average of the teacher's features; block targets follow each student's stream
(mirror.heal_structured.distill_teacher). The choice is recorded in meta.json and metrics.json.

Steps (each resumable; rerun the same command after an interruption):
 1. teacher (the original model), heal data (100k train, 5k LR-selection train images), span structure;
 2. the compressed model (once), its trainable form (make_trainable), a check that the trainable form computes the
    compressor's function (CLS features on validation probe images) and that spec.json rebuilds it;
 3. learning rate: 100-step trials per grid rate on the LR-selection set (finished trials are reused after an
    interruption); the extra 2e-4 trial when the best rate is the top of the grid (recorded, not chosen);
 4. the full run (mirror.heal.heal) with the training state saved every train_state.every steps and resumed
    automatically; validation k-NN top-1 and flip consistency (bf16, paper protocol) at the curve steps (full
    validation at the first and last, --curve-queries at the others), written to curves.csv as they come;
 5. final measures: bf16 k-NN (the last curve point), an fp32 pass (bank and all 50k queries in fp32, TF32 off:
    flip_consistency_fp32, the measure for exact-invariance claims), the largest CLS feature difference under flip
    (fp32, raw and unit-normalised), span exactness for the mirror family, parameters, analytic FLOPs
    (scripts/collect_results.analytic_cost), peak training memory, held-out loss;
 6. outputs: runs/MUST-09/<model>/<method>/s<seed>/ (spec.json, structure.npz for the mirror family,
    healed_state.pt, curves.csv, metrics.json, meta.json, lr_selection/, final/, knn predictions) and a row in
    results/MUST-09/summary.csv (and curves.csv there).
--latency-only (after the heal, on a quiet GPU): rebuilds the healed model with load_healed_model and times one
forward at batch 1, 64 and 256 (bf16 weights and inputs, torch.compile, median of 100 runs after 20 warm-up runs;
scripts/m2_latency.time_model), with the peak inference memory per batch; for the mirror family also its even/odd
form (mirror/efficient.py, as M2). Writes latency.json in the run folder and rows in results/MUST-09/latency.csv.
"""
from __future__ import annotations

import torch

DEFAULT_GPU_FRAC = 0.15
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_FRAC)  # cap before anything is allocated; --gpu-frac sets the final cap

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.attention import attn_dims  # noqa: E402
from mirror.heal import HealConfig, exact_involution, heal, load_trained_blocks, probe_extra_lrs, select_lr  # noqa: E402
from mirror.heal_structured import (adapter_spec, block_costs, build_adapter_from_spec, distill_teacher,  # noqa: E402
                                    feature_gap, load_healed_model, make_trainable, stored_params, verify_rebuild,
                                    write_spec)
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, load_config, log, set_seed, setup_torch  # noqa: E402

EXPERIMENT = "MUST-09"
MIRROR_METHODS = ("mirror_exact", "random_structure")
CONTRACT_METHODS_V2 = ("svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit")   # mirror.baselines.METHODS_V2
SHIM_METHODS = ("svd",)
STATE_FILE = "healed_state.pt"
TRAIN_STATE = "train_state.pt"


# ----------------------------------------------------------------------------------------------
# Scripts reused as modules
# ----------------------------------------------------------------------------------------------
_GPU_FRAC = [DEFAULT_GPU_FRAC]


def load_script(name: str):
    """scripts/<name>.py as a module (registered once). Scripts set their own GPU cap at import time, so this
    process's cap is applied again afterwards."""
    key = f"_must09_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
        apply_gpu_cap(_GPU_FRAC[0])
    return sys.modules[key]


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def apply_gpu_cap(frac: float) -> None:
    """(Re)apply this process's cap: imported scripts set their own at import time."""
    _GPU_FRAC[0] = float(frac)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))


# ----------------------------------------------------------------------------------------------
# Arguments and paths
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "must09.yaml"))
    p.add_argument("--from-gate-b", default=None,
                   help="results/MUST-07/gate_b.json: take --method and --hparams-json from gate B's ranking")
    p.add_argument("--gate-rank", type=int, default=1, help="with --from-gate-b: 1 = best baseline, 2 = second best")
    p.add_argument("--allow-incomplete-gate", action="store_true", help="accept a gate_b.json with missing methods")
    p.add_argument("--method", default=None,
                   help=f"one of {', '.join(MIRROR_METHODS + CONTRACT_METHODS_V2)} (or any mirror.baselines.METHODS_V2)")
    p.add_argument("--seed", type=int, default=None, help="healing seed (data order, crops, flips, LR trials); default config")
    p.add_argument("--hparams-json", default=None, help="hyperparameters chosen in MUST-07: a JSON file or a JSON string")
    p.add_argument("--targets", choices=("sym", "raw"), default=None, help="default: config `targets` (sym)")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT and synthetic data on CPU")
    p.add_argument("--out-root", default=None, help="overrides out_root (runs/MUST-09)")
    p.add_argument("--results-root", default=None, help="overrides results_root (results/MUST-09)")
    p.add_argument("--gpu-frac", type=float, default=None, help=f"GPU memory cap per process (default config, {DEFAULT_GPU_FRAC})")
    p.add_argument("--micro-batch", type=int, default=None, help="images per forward/backward (results unchanged)")
    p.add_argument("--curve-queries", type=int, default=None, help="validation queries at the intermediate curve steps")
    p.add_argument("--curve-steps", default=None, help="comma-separated optimiser steps for the k-NN curve")
    p.add_argument("--state-every", type=int, default=None, help="steps between saved training states")
    p.add_argument("--fresh", action="store_true", help="ignore saved training state and finished trials")
    p.add_argument("--keep-state", action="store_true", help="keep train_state.pt after a finished run")
    p.add_argument("--force", action="store_true", help="redo the final measures of a finished run")
    p.add_argument("--lr", type=float, default=None, help="skip the learning-rate selection (debugging only)")
    p.add_argument("--lr-from", default=None,
                   help="run directory of this method's seed-0 run: reuse its chosen learning rate (seeds 1 and 2)")
    p.add_argument("--max-steps", type=int, default=None, help="cap the full run (debugging only)")
    p.add_argument("--no-fp32-pass", action="store_true", help="skip the final fp32 flip-consistency pass")
    p.add_argument("--span-structure", default=None,
                   help="a mirror.mirror_family SpanStructure file (e.g. MUST-07's results/MUST-07/structure/"
                        "<model>_s0.npz) instead of the M1 and X2 structure files")
    p.add_argument("--mirror-builder", choices=("auto", "family", "shim"), default=None)
    p.add_argument("--compressor-builder", choices=("auto", "library", "shim"), default=None)
    p.add_argument("--latency-only", action="store_true", help="time the healed model of this run (after the heal)")
    p.add_argument("--with-original", action="store_true", help="--latency-only: also time the original model")
    p.add_argument("--latency-batches", default=None, help="comma-separated batch sizes (default 1,64,256)")
    p.add_argument("--latency-iters", type=int, default=None)
    p.add_argument("--latency-warmup", type=int, default=None)
    p.add_argument("--no-compile", action="store_true", help="--latency-only: eager mode")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    if a.from_gate_b:
        apply_gate_b(a)
    if not a.method:
        p.error("--method or --from-gate-b is required")
    return a


def apply_gate_b(a) -> None:
    """--from-gate-b: take the method and its MUST-07 hyperparameters from results/MUST-07/gate_b.json (gate B,
    chosen on the method-selection set only), so that a job file can name "the best baseline" before it is known."""
    path = Path(a.from_gate_b)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not gate.get("complete") and not a.allow_incomplete_gate:
        raise SystemExit(f"--from-gate-b: {path} is incomplete (missing {gate.get('missing_methods')}); "
                         "finish MUST-07 at the full size first, or pass --allow-incomplete-gate")
    ranking = sorted(gate.get("ranking") or [], key=lambda r: int(r["rank"]))
    if len(ranking) < a.gate_rank:
        raise SystemExit(f"--from-gate-b: {path} ranks {len(ranking)} methods, rank {a.gate_rank} asked")
    rec = ranking[a.gate_rank - 1]
    if a.method and a.method != rec["method"]:
        raise SystemExit(f"--method {a.method} differs from gate B rank {a.gate_rank} ({rec['method']})")
    a.method = rec["method"]
    if a.hparams_json is None:
        a.hparams_json = json.dumps(rec.get("hparams") or {})
    print(f"gate B ({path}): rank {a.gate_rank} is {a.method} with hyperparameters {a.hparams_json}", flush=True)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def setup(a) -> tuple:
    cfg = load_config(a.config, tiny=a.tiny)
    if a.seed is not None:
        cfg["seed"] = int(a.seed)
    if a.targets:
        cfg["targets"] = a.targets
    if a.out_root:
        cfg["out_root"] = a.out_root
    if a.results_root:
        cfg["results_root"] = a.results_root
    if a.gpu_frac is not None:
        cfg["compute"]["gpu_memory_fraction"] = float(a.gpu_frac)
    if a.micro_batch:
        cfg["heal"]["micro_batch"] = int(a.micro_batch)
    if a.curve_queries is not None:
        cfg["curve"]["queries"] = int(a.curve_queries)
    if a.curve_steps:
        cfg["curve"]["steps"] = [int(s) for s in a.curve_steps.split(",") if s.strip()]
    if a.state_every is not None:
        cfg["train_state"]["every"] = int(a.state_every)
    if a.span_structure:
        cfg["structure"]["span_structure"] = a.span_structure
    if a.mirror_builder:
        cfg["mirror_builder"] = a.mirror_builder
    if a.compressor_builder:
        cfg["compressor_builder"] = a.compressor_builder
    if a.device:
        cfg["compute"]["device"] = a.device
    cfg["method"] = a.method
    return cfg


def model_name(cfg: dict) -> str:
    return "tiny" if cfg.get("tiny_mode") else str(cfg["model"]["name"])


def run_dir_of(cfg: dict) -> Path:
    folder = cfg["method"] if cfg["targets"] == "sym" else f"{cfg['method']}__targets-{cfg['targets']}"
    return resolve(cfg["out_root"]) / model_name(cfg) / folder / f"s{int(cfg['seed'])}"


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, obj) -> None:
    from mirror.heal_structured import _jsonable

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def upsert_csv(path, rows: list, key: list) -> list:
    """Add rows to a CSV, replacing rows with the same key (rewritten atomically)."""
    path = Path(path)
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    new_keys = {tuple(str(r.get(c)) for c in key) for r in rows}
    out = [r for r in old if tuple(str(r.get(c)) for c in key) not in new_keys] + rows
    cols = []
    for r in out:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c) for c in cols})
    os.replace(tmp, path)
    return out


def parse_hparams(text):
    if not text:
        return None
    p = resolve(text) if not text.strip().startswith("{") else None
    if p is not None and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(text)


# ----------------------------------------------------------------------------------------------
# Teacher, data, structure
# ----------------------------------------------------------------------------------------------
def build_teacher(cfg: dict, device) -> tuple:
    crop = int(cfg["heal"]["crop_size"])
    if cfg.get("tiny_mode"):
        spec = adapter_spec("tiny", img_size=crop, seed=int(cfg.get("tiny_model_seed", 0)), dtype="float32")
    else:
        m = cfg["model"]
        spec = adapter_spec("dinov2", hub_repo=m["hub_repo"], name=m["name"], img_size=int(m.get("img_size", crop)))
    return build_adapter_from_spec(spec, device), spec


def fixed_seed_cfg(cfg: dict) -> dict:
    """With --tiny the synthetic data and the tiny random structure come from the config seed; fix it at 0 so the
    healing seed changes only the healing randomness, as outside --tiny."""
    c = copy.deepcopy(cfg)
    if cfg.get("tiny_mode"):
        c["seed"] = 0
    return c


def load_span(cfg: dict, teacher, random_span: bool) -> dict:
    """Span structure for the full model: A*, neuron pairings, head maps and register pairing, fitted or (random_span)
    matched random with structure.random_seed (mirror.m1.matched_random_span_structure, as E3).

    Source: structure.span_structure (--span-structure), a mirror.mirror_family SpanStructure file such as the one
    MUST-07 caches (results/MUST-07/structure/<model>_s<fit seed>.npz), when set; otherwise the M1 structure file and
    the X2 operator file (scripts/heal_vexact.load_vexact_structure: the source of the V-exact k = 12 run and E3)."""
    L = int(teacher.depth)
    path = (cfg["structure"] or {}).get("span_structure")
    if not path:
        HV = load_script("heal_vexact")
        c = fixed_seed_cfg(cfg)
        c["structure"] = dict(c["structure"], random_span=bool(random_span))
        return HV.load_vexact_structure(c, teacher, L)
    from mirror.m1 import matched_random_span_structure
    from mirror.mirror_family import load_span_structure

    src = resolve(path)
    st = load_span_structure(src)
    if st.depth != L:
        raise SystemExit(f"{src} has {st.depth} blocks, the model {L}")
    out = {"A_star": exact_involution(st.A_star[L]), "perms": list(st.perms), "heads": list(st.heads),
           "reg_perm": np.asarray(st.reg_perm[L], dtype=np.int64), "span_source": str(src),
           "operators_source": str(src), "heads_source": str(src), "span_struct": st}
    if random_span:
        seed = int(cfg["structure"].get("random_seed", 1))
        r = matched_random_span_structure(out["A_star"], out["perms"], out["heads"], out["reg_perm"], seed)
        out["random_structure"] = {"seed": seed, "matched_to": str(src)}
        out.update(A_star=exact_involution(r["A_star"]), perms=list(r["perms"]), heads=list(r["heads"]),
                   reg_perm=np.asarray(r["reg_perm"], dtype=np.int64))
    return out


def span_struct_of(fitted: dict, teacher, cfg: dict):
    """mirror.mirror_family.SpanStructure for build_mirror_variant: the loaded file itself, or one built from the M1
    and X2 structure (A_star and reg_perm for the full span only; A_layers is a placeholder, unused by the A*-type
    methods)."""
    if fitted.get("span_struct") is not None:
        return fitted["span_struct"]
    from mirror.mirror_family import SpanStructure

    L = int(teacher.depth)
    A = exact_involution(fitted["A_star"])
    return SpanStructure(model=model_name(cfg), fit_seed=int(cfg["compressor"].get("fit_seed", 0)),
                         A_layers=[A] * (L + 1), perms=[np.asarray(p, dtype=np.int64) for p in fitted["perms"]],
                         heads=list(fitted["heads"]), A_star={L: A},
                         reg_perm={L: np.asarray(fitted["reg_perm"], dtype=np.int64)},
                         meta={"source": f"{fitted.get('span_source')} (A*), {fitted.get('operators_source')} "
                                         f"(pairings), {fitted.get('heads_source')} (head maps)",
                               "A_layers": "placeholder (A* repeated), unused by mirror_exact and random_structure"})


def knn_runtime(cfg: dict, device):
    """(bank loader, full query loader, curve query loader or None, number of classes, protocol, fit loader)."""
    kc = cfg["knn"]
    n_sub = cfg["curve"].get("queries")
    if cfg.get("tiny_mode"):
        from mirror.variants import TinyProtocol

        proto = TinyProtocol(img_size=int(cfg["heal"]["crop_size"]), batch_size=int(kc.get("batch_size", 16)))
        n_full = len(proto.query_idx)
        sub = proto.query_loader(int(n_sub)) if n_sub and int(n_sub) < n_full else None
        return proto.bank_loader(), proto.query_loader(), sub, proto.num_classes, "tiny synthetic", proto
    from mirror.protocol import Protocol

    proto = Protocol(resolve(kc.get("cache_dir", "cache")), int(kc.get("batch_size", 128)), 16,
                     int(kc.get("num_workers", 4)), device.type == "cuda")
    sub = proto.query_loader(int(n_sub)) if n_sub and int(n_sub) < len(proto.val_labels) else None
    return proto.bank_loader(), proto.query_loader(), sub, 1000, "v2", proto


# ----------------------------------------------------------------------------------------------
# Compressed model (the compressor library or the marked shims)
# ----------------------------------------------------------------------------------------------
def _svd_shim(teacher, st: dict, blocks: list, comps: tuple) -> tuple:
    """SHIM (until mirror/budget.py and mirror.baselines.compress exist): plain SVD at the Mirror-exact budget, the E3
    SVD arm: per block and matrix the largest rank r with r (rows + cols) <= the V-exact free-parameter count of that
    matrix (mirror.heal_vexact.vexact_block_params, matched_svd_ranks), applied in absorbed form."""
    from mirror.heal_vexact import matched_svd_ranks, vexact_block_params
    from mirror.m1 import apply_attention_svd
    from mirror.surgery import apply_svd_baseline

    d, h = teacher.embed_dim, teacher.mlp_hidden
    H, dh = attn_dims(teacher.model.blocks[0])
    D = H * dh
    per = vexact_block_params(d, h, st["A_star"], [st["perms"][b] for b in blocks], [st["heads"][b] for b in blocks])
    match = matched_svd_ranks(per, h, d, D)
    m = copy.deepcopy(teacher.model)
    entries, budget = [], 0
    if "mlp" in comps:
        apply_svd_baseline(m, blocks, [tuple(r["mlp_ranks"]) for r in match])
        for b, r in zip(blocks, match):
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": int(k), "params": int(k) * (h + d)}
                        for n, k in zip(("fc1", "fc2"), r["mlp_ranks"])]
        budget += sum(r["mlp_kept_vexact"] for r in match)
    if "attn" in comps:
        apply_attention_svd(m, blocks, [tuple(r["attn_ranks"]) for r in match])
        for b, r in zip(blocks, match):
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": int(k), "params": int(k) * (D + d)}
                        for n, k in zip(("wq", "wk", "wv", "wo"), r["attn_ranks"])]
        budget += sum(r["attn_kept_vexact"] for r in match)
    total = sum(e["params"] for e in entries)
    info = {"kind": "compressed", "method": "svd", "blocks": blocks, "components": list(comps), "per_matrix": entries,
            "params_total": int(total), "budget_total": int(budget), "gap": (budget - total) / budget,
            "source": "SHIM: E3 SVD arm (mirror.surgery.apply_svd_baseline, mirror.m1.apply_attention_svd, ranks from "
                      "mirror.heal_vexact.matched_svd_ranks); replace with mirror.baselines.compress"}
    return m, info


def build_compressed(cfg: dict, teacher, fitted: dict, hparams, device) -> tuple:
    """(compressed model, info for make_trainable, structure used for the targets {A_star, reg_perm})."""
    method = cfg["method"]
    L = int(teacher.depth)
    blocks = list(range(L))
    comps = tuple(cfg["components"])
    s = cfg["structure"]
    if method in MIRROR_METHODS:
        fam = _try_import("mirror.mirror_family") if cfg["mirror_builder"] in ("auto", "family") else None
        if cfg["mirror_builder"] == "family" and fam is None:
            raise SystemExit("--mirror-builder family: mirror.mirror_family is not importable")
        if fam is not None:
            model, info = fam.build_mirror_variant(teacher, method, blocks, comps, span_struct_of(fitted, teacher, cfg),
                                                   seed=int(s.get("random_seed", 1)))
            info = dict(info)
            st_ = info.get("structure") or {}
            info.update(kind="mirror", method=method, blocks=blocks, components=list(comps),
                        builder="mirror.mirror_family.build_mirror_variant")
            for key, alt in (("A_star", "A"), ("perms", "perms"), ("maps_after", "heads_after"), ("reg_perm", "reg_perm")):
                if key not in info and alt in st_:
                    info[key] = st_[alt]
            if method == "random_structure":
                info["random_structure"] = {"seed": int(s.get("random_seed", 1)), **(info.get("structure_info") or {})}
            missing = [k for k in ("A_star", "perms", "maps_after", "reg_perm") if k not in info]
            if missing:
                raise SystemExit(f"mirror.mirror_family info lacks {missing}; make_trainable needs them "
                                 "(or run with --mirror-builder shim)")
        else:
            from mirror.heal_vexact import build_vexact_student

            st = fitted if method == "mirror_exact" else load_span(cfg, teacher, random_span=True)
            model, _, binfo = build_vexact_student(teacher, st["A_star"], L, st["perms"], st["heads"], st["reg_perm"],
                                                   train_embeddings=True)
            info = {"kind": "mirror", "method": method, "blocks": blocks, "components": list(comps),
                    "A_star": exact_involution(st["A_star"]), "perms": [st["perms"][b] for b in blocks],
                    "maps_after": binfo["maps_after"], "reg_perm": st["reg_perm"], "even_head": binfo["even_head"],
                    "random_structure": st.get("random_structure"),
                    "builder": "SHIM: mirror.heal_vexact.build_vexact_student (the V-exact k = 12 and E3 code path); "
                               "replace with mirror.mirror_family.build_mirror_variant"}
        return model, info, {"A_star": info["A_star"], "reg_perm": info["reg_perm"]}
    lib_budget = _try_import("mirror.budget") if cfg["compressor_builder"] in ("auto", "library") else None
    lib = _try_import("mirror.baselines")
    have_lib = lib_budget is not None and lib is not None and hasattr(lib, "METHODS_V2") and hasattr(lib, "compress")
    if cfg["compressor_builder"] == "library" and not have_lib:
        raise SystemExit("--compressor-builder library: mirror.budget or mirror.baselines.compress is not importable")
    targets_struct = {"A_star": fitted["A_star"], "reg_perm": fitted["reg_perm"]}
    if have_lib:
        if method not in lib.METHODS_V2:
            raise SystemExit(f"unknown method {method!r}; mirror.baselines.METHODS_V2 = {lib.METHODS_V2}")
        budget = lib_budget.mirror_exact_budget(teacher.model, fitted["A_star"], blocks, [fitted["perms"][b] for b in blocks],
                                                [fitted["heads"][b] for b in blocks], components=comps,
                                                model_name=model_name(cfg))
        fit_seed = int(cfg["compressor"].get("fit_seed", 0))
        _, _, _, _, _, proto = knn_runtime(cfg, device)
        stats = lib.collect_stats(teacher, proto.fit_loader(fit_seed), device, blocks, [method], comps)
        model, info = lib.compress(teacher.model, method, blocks, comps, budget, stats, hparams)
        info = dict(info)
        info.update(kind="compressed", method=method, blocks=info.get("blocks", blocks),
                    components=info.get("components", list(comps)), source="mirror.baselines.compress",
                    budget={"total": getattr(budget, "total", None)}, fit_seed=fit_seed)
        return model, info, targets_struct
    if method not in SHIM_METHODS:
        raise SystemExit(f"method {method!r} needs mirror/budget.py and mirror.baselines.METHODS_V2, which are not "
                         f"importable yet; without them only {MIRROR_METHODS + SHIM_METHODS} run")
    model, info = _svd_shim(teacher, fitted, blocks, comps)
    return model, info, targets_struct


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def knn_eval(adapter, bank, query, device, n_cls, k, T, model, amp) -> dict:
    """scripts/flip_avg_baseline.knn_eval (mirror.knn.evaluate_knn's computation plus the feature difference
    between each query and its mirror)."""
    FA = load_script("flip_avg_baseline")
    return FA.knn_eval(adapter, bank, query, device, n_cls, k, T, model=model, amp=amp, flip_average=False)


class CurveLogger:
    """Validation k-NN top-1 and flip consistency at the curve steps (the step hook of mirror.heal.heal). Rows go to
    <run>/curves.csv as they come, so a resumed run skips finished steps; the full-validation points also save
    their predictions (for paired bootstraps between methods)."""

    COLUMNS = ["step", "knn_top1", "flip_consistency", "n_query", "n_bank", "full_validation", "amp",
               "feature_flip_max_abs", "feature_identical_pct", "seconds", "git_hash", "seed"]

    def __init__(self, path: Path, adapter, rt: dict, steps: list, full_steps: set, tags: dict):
        self.path, self.adapter, self.rt, self.tags = Path(path), adapter, rt, tags
        self.steps, self.full_steps = sorted(set(steps)), set(full_steps)
        self.rows = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                self.rows = [self._typed(r) for r in csv.DictReader(fh)]

    @staticmethod
    def _typed(r: dict) -> dict:
        out = dict(r)
        for k in ("step", "n_query", "n_bank", "seed"):
            if out.get(k) not in (None, ""):
                out[k] = int(float(out[k]))
        for k in ("knn_top1", "flip_consistency", "feature_flip_max_abs", "feature_identical_pct", "seconds"):
            if out.get(k) not in (None, ""):
                out[k] = float(out[k])
        out["full_validation"] = str(out.get("full_validation")) == "True"
        return out

    def done(self, step: int) -> bool:
        return any(int(r["step"]) == int(step) for r in self.rows)

    def row(self, step: int):
        return next((r for r in self.rows if int(r["step"]) == int(step)), None)

    def __call__(self, step: int, model) -> None:
        if self.done(step):
            return
        rt = self.rt
        full = step in self.full_steps or rt["sub"] is None
        t0 = time.time()
        r = knn_eval(self.adapter, rt["bank"], rt["query"] if full else rt["sub"], rt["device"], rt["n_cls"], rt["k"],
                     rt["T"], model, rt["amp"])
        row = {"step": int(step), "knn_top1": r["top1"], "flip_consistency": r["flip_consistency"],
               "n_query": r["n_query"], "n_bank": r["n_bank"], "full_validation": bool(full), "amp": rt["amp"] or "fp32",
               "feature_flip_max_abs": r["feature_flip_max_abs"], "feature_identical_pct": r["feature_identical_pct"],
               "seconds": round(time.time() - t0, 1), **self.tags}
        if full:
            save_predictions(self.path.parent / "knn" / f"step{int(step)}_{row['amp']}.npz", r)
        self.rows.append(row)
        upsert_csv(self.path, [row], ["step"])
        log(f"curve: step {step}: k-NN top-1 {r['top1']:.2f}, flip consistency {r['flip_consistency']:.2f}% "
            f"({r['n_query']} queries, {row['amp']}, {row['seconds']:.0f} s)")


def save_predictions(path: Path, r: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=r["pred"].numpy().astype(np.int16), pred_flip=r["pred_flip"].numpy().astype(np.int16),
                        labels=r["labels"].numpy().astype(np.int16))


def curve_steps(cfg: dict, total: int) -> tuple:
    """(steps, full-validation steps): the configured steps below the budget, plus the last step."""
    steps = sorted({int(s) for s in cfg["curve"]["steps"] if 0 <= int(s) < total} | {int(total)})
    return steps, {steps[0], int(total)}


def total_steps(cfg: HealConfig, n_train: int) -> int:
    per_epoch = n_train // int(cfg.batch_size)
    return int(cfg.max_steps) if cfg.max_steps is not None else int(math.floor(cfg.epochs * per_epoch))


def probe_images(cfg: dict, adapter, device, n: int) -> torch.Tensor:
    HV = load_script("heal_vexact")
    cd = HV.check_data(fixed_seed_cfg(cfg), device)
    return torch.cat(HV.probe_batches(adapter, cd["probe"], n, device))


def cost_of(cfg: dict, adapter, student, spec: dict, structure: dict | None) -> dict:
    """Parameters and analytic FLOPs per image (scripts/collect_results.analytic_cost, the cost model of the paper
    tables) from the structured form; plus the entries the model stores."""
    cr = load_script("collect_results")
    H, W = adapter.grid_hw
    proj = adapter.model.patch_embed.proj
    n_dense = int(sum(p.numel() for p in adapter.model.parameters()))
    arch = {"d": int(adapter.embed_dim), "h": int(adapter.mlp_hidden), "depth": int(adapter.depth),
            "T": int(adapter.n_tokens), "n_patches": int(H * W), "patch": int(adapter.patch_size),
            "in_chans": int(getattr(proj, "in_channels", 3)), "model_params_dense": n_dense}
    blocks = block_costs(student, spec, structure, arch["d"], arch["h"])
    c = cr.analytic_cost(arch, [{k: b[k] for k in ("mlp", "attn", "rot")} for b in blocks])
    basis = int(sum(b["basis_params"] for b in blocks))
    params = int(c["model_params"]) + basis
    return {"arch": arch, "per_block": blocks, "params_model": params, "params_frac": params / n_dense,
            "params_basis_change": basis, "cost_source": blocks[0]["source"] if blocks else None,
            "params_converted_kept": c["matrix_params_kept"], "params_converted_dense": c["matrix_params_converted_dense"],
            "kept_frac_converted": c["kept_frac_converted"], "gflops": c["gflops_model"],
            "gflops_dense": c["gflops_model_dense"], "gflops_frac": c["gflops_model"] / c["gflops_model_dense"],
            "params_stored": stored_params(student), "params_original": n_dense,
            "cost_model": "scripts/collect_results.analytic_cost: matrices of the converted blocks at the multiply-adds "
                          "of the method's own efficient form (mirror family: free parameters, as its even/odd form "
                          "executes them; compressors: their own per-matrix counts) plus basis changes (d^2 units), "
                          "token mixing 2 T^2 d per block, patch embedding; not norms, softmax, GELU, biases. "
                          "Parameters: whole model, biases and embeddings dense, basis changes included"}


# ----------------------------------------------------------------------------------------------
# Heal
# ----------------------------------------------------------------------------------------------
def final_done(final_dir: Path, fcfg: HealConfig, total: int) -> dict | None:
    from mirror.heal_structured import _jsonable

    m = read_json(final_dir / "metrics.json")
    if not m or m.get("config") != _jsonable(asdict(fcfg)) or int(m.get("steps_done", -1)) != int(total):
        return None
    if m.get("diverged") or not m.get("checkpoint") or not Path(m["checkpoint"]).exists():
        return None
    return m


def run_heal(a, cfg: dict) -> dict:
    t_start = time.time()
    device = setup_torch(cfg)
    apply_gpu_cap(cfg["compute"]["gpu_memory_fraction"])
    seed = int(cfg["seed"])
    set_seed(seed)
    method, targets = cfg["method"], cfg["targets"]
    run = run_dir_of(cfg)
    run.mkdir(parents=True, exist_ok=True)
    results = resolve(cfg["results_root"])
    git = git_info()
    timer = StageTimer()
    tags = {"git_hash": git["git_hash"], "seed": seed}
    done = read_json(run / "metrics.json")
    if done and done.get("complete") and not a.force:
        log(f"MUST-09 {method} s{seed}: finished before ({run / 'metrics.json'}); --force redoes the final measures")
        return done
    hparams = parse_hparams(a.hparams_json)
    if hparams is not None and method in MIRROR_METHODS:
        log("--hparams-json is ignored for the mirror family")
    log(f"MUST-09 heal: method {method}, seed {seed}, targets {targets}, run {run}, git {git['git_hash'][:10]}"
        f"{' (dirty)' if git.get('git_dirty') else ''}")

    HC = load_script("heal")
    load_script("heal_vexact")
    load_script("flip_avg_baseline")
    apply_gpu_cap(cfg["compute"]["gpu_memory_fraction"])
    with timer.stage("load"):
        teacher, teacher_spec = build_teacher(cfg, device)
        L = int(teacher.depth)
        data = HC.load_heal_data(fixed_seed_cfg(cfg))
        fitted = load_span(cfg, teacher, random_span=False)
        bank, query, sub, n_cls, protocol, _ = knn_runtime(cfg, device)
    blocks = list(range(L))

    with timer.stage("compress"):
        compressed, info, tstruct = build_compressed(cfg, teacher, fitted, hparams, device)
        compressed.eval()

    factors = str(cfg.get("lowrank_factors", "balanced"))

    def make_student():
        m = copy.deepcopy(compressed)
        m, cons, _ = make_trainable(m, info, adapter=teacher, lowrank_factors=factors)
        return m, cons

    with timer.stage("structure checks"):
        s0, cons0, spec = make_trainable(copy.deepcopy(compressed), info, adapter=teacher, lowrank_factors=factors)
        spec.update(experiment=EXPERIMENT, adapter=teacher_spec, state_file=STATE_FILE, targets=targets, seed=seed,
                    model=model_name(cfg), hparams=hparams)
        verify_rebuild(teacher, spec, s0)
        xs = probe_images(cfg, teacher, device, int(cfg["checks"].get("exact_images", 64)))
        conv = feature_gap(teacher, compressed, s0, xs)
        tol = float(cfg["checks"].get("conversion_tol", 1e-3))
        conv.update(tol=tol, passed=bool(conv["rel_max"] <= tol))
        log(f"trainable form against the compressed model: CLS feature max relative difference {conv['rel_max']:.2e} "
            f"({'ok' if conv['passed'] else 'FAIL'}, tolerance {tol:g})")
        if not conv["passed"]:
            raise SystemExit("the trainable form does not compute the compressed model's function; see the log")
        mirror_struct = ({"A_star": info["A_star"], "perms": info["perms"], "maps_after": info["maps_after"],
                          "reg_perm": info["reg_perm"], "blocks": blocks} if spec["family"] == "mirror" else None)
        write_spec(run, spec, mirror_struct)
        vteacher = distill_teacher(teacher, spec, targets, tstruct["A_star"], tstruct["reg_perm"])
        del s0, cons0

    hcfg = HealConfig.from_dict({**cfg["heal"], "blocks": blocks, "seed": seed, "train_scope": "all",
                                 "checkpoint_path": str(run / STATE_FILE)})
    if a.max_steps:
        hcfg = replace(hcfg, max_steps=int(a.max_steps))
    sel_cfg = cfg["lr_selection"]
    grid = [float(v) for v in sel_cfg["lrs"]]
    resume = not a.fresh
    if a.fresh:
        for f in (run / TRAIN_STATE, run / "curves.csv"):
            if f.exists():
                f.unlink()
    lr_extra = None
    with timer.stage("lr selection"):
        if a.lr is not None:
            selection = {"skipped": True, "best_lr": float(a.lr), "lrs": [float(a.lr)]}
            chosen = float(a.lr)
        elif a.lr_from:
            # PREREG_NEXT_PHASE.md, MUST-09: the learning rate is chosen once per method (its seed-0 trials) and reused.
            src = Path(a.lr_from) / "metrics.json"
            if not src.exists():
                raise FileNotFoundError(f"--lr-from: {src} missing (the seed-0 run of this method must finish first)")
            prev = json.loads(src.read_text())
            if prev.get("method") not in (None, method):
                raise ValueError(f"--lr-from: {src} is method {prev.get('method')}, not {method}")
            chosen = float(prev["lr"])
            selection = {"reused_from": str(src), "best_lr": chosen, "source_seed": prev.get("seed"),
                         "source_selection": prev.get("lr_selection")}
            log(f"learning rate {chosen:.0e} reused from {src}")
        else:
            selection = select_lr(vteacher, make_student, data, hcfg, lrs=grid, budget_steps=int(sel_cfg["budget_steps"]),
                                  out_dir=run / "lr_selection", resume=resume)
            chosen = float(selection["best_lr"])
            extra, when = [float(v) for v in (sel_cfg.get("extra_lrs") or [])], str(sel_cfg.get("extra_when", "top"))
            if extra and (when == "always" or (when == "top" and chosen == max(grid))):
                lr_extra = probe_extra_lrs(vteacher, make_student, data, hcfg, selection, extra, run, resume=resume,
                                           extra_info={"experiment": EXPERIMENT, "method": method,
                                                       "extra_when": when})
                if sel_cfg.get("extra_can_win") and lr_extra["extra_would_win"]:
                    chosen = float(lr_extra["best_lr_with_extra"])
                    log(f"lr_selection.extra_can_win: the extra rate {chosen:.0e} is used")

    fcfg = replace(hcfg, lr=chosen)
    total = total_steps(fcfg, len(data.train_rows))
    steps, full_steps = curve_steps(cfg, total)
    kc = cfg["knn"]
    rt = {"bank": bank, "query": query, "sub": sub, "device": device, "n_cls": n_cls, "k": int(kc.get("k", 20)),
          "T": float(kc.get("temperature", 0.07)), "amp": (kc.get("amp") if device.type == "cuda" else None) or None}
    if rt["amp"] == "none":
        rt["amp"] = None
    curves = CurveLogger(run / "curves.csv", teacher, rt, steps, full_steps, tags)
    final_dir = run / "final"
    with timer.stage("heal"):
        fin = None if a.fresh else final_done(final_dir, fcfg, total)
        if fin is not None:
            log(f"final run finished before ({final_dir / 'metrics.json'}); loading {fin['checkpoint']}")
            student, cons = make_student()
            load_trained_blocks(student, fin["checkpoint"])
        else:
            student, cons = make_student()
            fin = heal(vteacher, student, data, fcfg, cons, final_dir, step_hook=curves, hook_steps=steps,
                       state_path=run / TRAIN_STATE, state_every=int(cfg["train_state"].get("every", 0)), resume=resume)
            fin = {k: v for k, v in fin.items() if k != "history"}
        if fin.get("diverged"):
            write_json(run / "metrics.json", {"complete": False, "status": "diverged", "final": fin, **tags})
            raise SystemExit(f"the final run diverged at step {fin.get('steps_done')}; the run is invalid")
    student.eval()
    if not curves.done(total):            # the last point, if a crash cut it short (the logger skips finished ones)
        curves(total, student)
    if not curves.done(steps[0]) and steps[0] == 0:
        curves(0, make_student()[0].eval())
    missing = [s for s in steps if not curves.done(s)]
    if missing:
        log(f"WARNING: curve points {missing} are missing (they cannot be recomputed after training)")

    with timer.stage("final measures"):
        final_knn = curves.row(total)
        fp32 = None
        if kc.get("fp32_pass", True) and not a.no_fp32_pass:
            prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            t0 = time.time()
            r = knn_eval(teacher, bank, query, device, n_cls, rt["k"], rt["T"], student, None)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
            save_predictions(run / "knn" / f"step{total}_fp32.npz", r)
            fp32 = {"top1": r["top1"], "flip_consistency": r["flip_consistency"], "n_bank": r["n_bank"],
                    "n_query": r["n_query"], "feature_flip_max_abs_unit": r["feature_flip_max_abs"],
                    "feature_identical_pct": r["feature_identical_pct"], "seconds": round(time.time() - t0, 1),
                    "precision": "fp32 forward passes for the bank and every query and its mirror, TF32 off"}
            log(f"fp32 pass: k-NN {r['top1']:.2f}, flip consistency {r['flip_consistency']:.3f}% on {r['n_query']} "
                f"queries ({fp32['seconds']:.0f} s)")
        from mirror.m1 import feature_invariance

        inv = feature_invariance(teacher, student, xs)
        exact = None
        if spec["family"] == "mirror":
            from mirror.heal_vexact import span_exactness

            exact = span_exactness(teacher, student, [xs], info["A_star"], L,
                                   info["reg_perm"] if len(info["reg_perm"]) else None,
                                   tol=float(cfg["checks"].get("exact_tol", 1e-4)))
        cost = cost_of(cfg, teacher, student, spec, mirror_struct)
        orig = original_knn(cfg, teacher, bank, query, device, n_cls, rt, git)

    sel_lrs = selection.get("lrs")
    metrics = {
        "experiment": EXPERIMENT, "complete": True, "method": method, "family": spec["family"], "model": model_name(cfg),
        "seed": seed, "targets": targets, "teacher": vteacher.describe(), "lr": chosen, "lr_selection": selection,
        "lr_extra": None if lr_extra is None else {k: lr_extra[k] for k in ("chosen_lr", "extra_lrs", "extra_would_win",
                                                                             "winning_extra_lrs", "best_lr_with_extra")},
        "lr_grid": sel_lrs, "total_steps": total, "curve": curves.rows, "knn_bf16": final_knn, "knn_fp32": fp32,
        "knn_original": orig, "cls_flip_fp32_probe": {**inv, "n_images": int(xs.shape[0])}, "exactness": exact,
        "conversion_check": conv, "cost": cost, "peak_train_memory_gb": fin.get("peak_memory_gb"),
        "heldout_initial": fin.get("heldout_initial"), "heldout_final": fin.get("heldout_final"),
        "constraint_checks": fin.get("constraint_checks"), "n_trainable": fin.get("n_trainable"),
        "seconds_train": fin.get("seconds_train"), "compressor": {k: info.get(k) for k in (
            "source", "builder", "params_total", "budget_total", "gap", "budget", "fit_seed") if k in info},
        "hparams": hparams, "random_structure": info.get("random_structure"),
        "timing": {"stages": timer.stages, "total_s": timer.total()}, **git, "env": env_info(device),
    }
    write_json(run / "metrics.json", metrics)
    write_meta(run, experiment=EXPERIMENT, config={"config": cfg, "args": vars(a)},
               seeds={"heal": seed, "compressor_fit": info.get("fit_seed"),
                      "random_structure": cfg["structure"].get("random_seed") if method == "random_structure" else None},
               models={model_name(cfg): teacher_spec}, device=device, wall_seconds=time.time() - t_start,
               extra={"targets": targets, "teacher": vteacher.describe(), "builder": info.get("builder") or info.get("source"),
                      "n_train": int(len(data.train_rows)), "n_lr_selection": int(len(data.heldout_rows)),
                      "protocol": protocol, "run_dir": str(run)})
    row = summary_row(metrics, run)
    upsert_csv(results / "summary.csv", [row], ["model", "method", "targets", "seed"])
    upsert_csv(results / "curves.csv", [{"model": row["model"], "method": method, "targets": targets, "seed": seed, **r}
                                        for r in curves.rows], ["model", "method", "targets", "seed", "step"])
    if (run / TRAIN_STATE).exists() and not a.keep_state:
        (run / TRAIN_STATE).unlink()
    log(f"MUST-09 {method} s{seed} done: k-NN {row['knn_top1']:.2f} ({row['knn_amp']}), flip consistency "
        f"{row['flip_consistency']:.3f}% ({row['knn_amp']}), {row['flip_consistency_fp32']}% (fp32 pass); "
        f"{row['params_frac']:.3f} of the parameters; wrote {run}")
    return metrics


def original_knn(cfg, teacher, bank, query, device, n_cls, rt, git) -> dict:
    """The original model's bf16 k-NN (cache/knn_original, as every healing run; computed for --tiny)."""
    from mirror.variants import original_knn as ok

    tiny = bool(cfg.get("tiny_mode"))
    return ok(teacher, bank, query, device, n_cls, model_name(cfg), rt["amp"] or "none",
              resolve(cfg["knn"].get("cache_dir", "cache")), None, tiny, git["git_hash"], rt["k"], rt["T"])


def summary_row(m: dict, run: Path) -> dict:
    kb, kf, c = m.get("knn_bf16") or {}, m.get("knn_fp32") or {}, m["cost"]
    row = {"experiment": EXPERIMENT, "model": m["model"], "method": m["method"], "family": m["family"],
           "targets": m["targets"], "seed": m["seed"], "lr": m["lr"], "steps": m["total_steps"],
           "knn_top1": kb.get("knn_top1"), "flip_consistency": kb.get("flip_consistency"), "knn_amp": kb.get("amp"),
           "knn_top1_fp32": kf.get("top1"), "flip_consistency_fp32": kf.get("flip_consistency"),
           "n_query": kb.get("n_query"), "feature_flip_max_abs_unit_fp32": kf.get("feature_flip_max_abs_unit"),
           "cls_flip_max_abs_fp32": m["cls_flip_fp32_probe"]["max_abs_diff"],
           "cls_flip_rel_max_fp32": m["cls_flip_fp32_probe"]["rel_max"],
           "exactness_max_rel": (m.get("exactness") or {}).get("max_rel_inside"),
           "knn_original": (m.get("knn_original") or {}).get("top1"),
           "params_model": c["params_model"], "params_frac": c["params_frac"],
           "params_converted_kept": c["params_converted_kept"], "params_stored": c["params_stored"],
           "params_compressor_total": (m.get("compressor") or {}).get("params_total"),
           "budget_total": (m.get("compressor") or {}).get("budget_total"),
           "gflops": c["gflops"], "gflops_frac": c["gflops_frac"], "peak_train_memory_gb": m.get("peak_train_memory_gb"),
           "heldout_initial": (m.get("heldout_initial") or {}).get("total"),
           "heldout_final": (m.get("heldout_final") or {}).get("total"), "seconds_train": m.get("seconds_train"),
           "seconds_total": m["timing"]["total_s"], "git_hash": m["git_hash"], "git_dirty": m.get("git_dirty"),
           "run_dir": str(run), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    for r in m.get("curve") or []:
        row[f"knn_top1_step{int(r['step'])}"] = r["knn_top1"]
    return row


# ----------------------------------------------------------------------------------------------
# Latency (--latency-only)
# ----------------------------------------------------------------------------------------------
def even_odd_form(model, info: dict):
    """The even/odd implementation of a healed mirror-family model (mirror/efficient.py, as M2 times it)."""
    from mirror.efficient import Basis, BlockSpec, build_fast_vit

    st = info["structure"]
    A = exact_involution(st["A_star"])
    L = len(model.blocks)
    specs = [BlockSpec(("mirror", A, A, st["maps_after"][b]), ("mirror", A, A, st["perms"][b])) for b in range(L)]
    return build_fast_vit(model, specs, [Basis.of(A)] * (L + 1), align=int(info.get("align", 8)), wo_mode="fold")


def latency_profile(named: list, batches: list, lat: dict, device, img_px: int, seed: int) -> list:
    """Median latency and peak memory of one bf16 forward per (model, batch): scripts/m2_latency.time_model
    (torch.compile unless lat["compile"] is false; median of lat["iters"] runs after lat["warmup"] runs).
    `named`: (name, module mapping images to features). Peak memory: torch.cuda.max_memory_allocated during the
    compile, warm-up and timed runs at that batch (CUDA only)."""
    M2 = load_script("m2_latency")
    ns = SimpleNamespace(seed=int(seed), img_px=int(img_px), compile_mode=lat.get("compile_mode", "default"),
                         warmup=int(lat["warmup"]), iters=int(lat["iters"]))
    mode = "compile" if lat.get("compile", True) and device.type == "cuda" else "eager"
    rows = []
    for name, mod in named:
        m16 = mod.to(torch.bfloat16).eval()
        for B in batches:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            r = M2.time_model(m16, int(B), mode, ns, device)
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None
            rows.append({"variant": name, "batch": int(B), "mode": r.get("mode_used", mode),
                         "latency_ms_median": r["latency_ms_median"], "latency_ms_p10": r.get("latency_ms_p10"),
                         "latency_ms_p90": r.get("latency_ms_p90"), "throughput_img_s": r.get("throughput_img_s"),
                         "peak_inference_memory_gb": peak, "compile_s": r.get("compile_s"),
                         "error": r.get("error") or r.get("compile_error") or ""})
            log(f"latency {name} batch {B} ({rows[-1]['mode']}): {r['latency_ms_median']:.2f} ms"
                + (f", peak {peak:.2f} GB" if peak is not None else ""))
        del m16
    return rows


def run_latency(a, cfg: dict) -> dict:
    device = setup_torch(cfg)
    apply_gpu_cap(cfg["compute"]["gpu_memory_fraction"])
    seed = int(cfg["seed"])
    run = run_dir_of(cfg)
    if not (run / "spec.json").exists():
        raise SystemExit(f"{run} has no spec.json: heal first")
    M2 = load_script("m2_latency")
    apply_gpu_cap(cfg["compute"]["gpu_memory_fraction"])
    git = git_info()
    lat = dict(cfg["latency"])
    if a.latency_iters:
        lat["iters"] = a.latency_iters
    if a.latency_warmup is not None:
        lat["warmup"] = a.latency_warmup
    if a.no_compile:
        lat["compile"] = False
    batches = [int(b) for b in (a.latency_batches.split(",") if a.latency_batches else lat["batches"])]
    adapter, model, info = load_healed_model(run, device)
    named = []
    if a.with_original:
        named.append(("original", M2.AdapterFeature(adapter, copy.deepcopy(adapter.model))))
    named.append(("healed_as_stored", M2.AdapterFeature(adapter, model)))
    eo_check = None
    if info["family"] == "mirror":
        fv = even_odd_form(model, {**info, "align": lat.get("align", 8)})
        x = probe_images(cfg, adapter, device, 8)
        with torch.no_grad():
            ref = adapter.cls_feature(x, model).double()
            got = fv(x).double()
        eo_check = {"max_abs": float((ref - got).abs().max()), "rel_max": float((ref - got).abs().max() / ref.abs().max()),
                    "dtype": "fp32", "n_images": int(x.shape[0])}
        log(f"even/odd form against the healed model (fp32): max relative difference {eo_check['rel_max']:.2e}")
        named.append(("healed_even_odd", fv))
    rows = latency_profile(named, batches, lat, device, adapter.img_size, seed)
    smi = M2.nvidia_smi() if device.type == "cuda" else "cpu"
    out = {"experiment": EXPERIMENT, "method": cfg["method"], "seed": seed, "targets": cfg["targets"], "rows": rows,
           "even_odd_check": eo_check, "protocol": {**lat, "batches": batches, "precision": "bf16 weights and inputs"},
           "nvidia_smi": smi, **git, "env": env_info(device), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(run / "latency.json", out)
    key = {"model": model_name(cfg), "method": cfg["method"], "targets": cfg["targets"], "seed": seed}
    upsert_csv(resolve(cfg["results_root"]) / "latency.csv",
               [{**key, **r, "git_hash": git["git_hash"]} for r in rows],
               ["model", "method", "targets", "seed", "variant", "batch"])
    return out


def main(argv=None):
    a = parse_args(argv)
    cfg = setup(a)
    if a.latency_only:
        return run_latency(a, cfg)
    return run_heal(a, cfg)


if __name__ == "__main__":
    main()

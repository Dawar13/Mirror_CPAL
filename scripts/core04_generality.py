#!/usr/bin/env python
"""CORE-04: generality, scale and training recipes (PROJECT_NEXT_PHASE.md CORE-04; PREREG_NEXT_PHASE.md P4.1 to P4.3).

    python scripts/core04_generality.py --steps structure --models dinov2_vitg14_reg --gpu-frac 0.25 --workers 16
    python scripts/core04_generality.py --steps structure --workers 16      # every model, one after another
    python scripts/core04_generality.py --steps knn                          # original k-NN of DINOv2 S, B, L, g
    python scripts/core04_generality.py --steps surgery --workers 16        # after MUST-07 has written gate_b.json
    python scripts/core04_generality.py --steps score                        # P4.1 to P4.3 and tables (CPU)
    python scripts/core04_generality.py --tiny --out tiny_out/core04         # random tiny models, synthetic data, CPU

Models (IDs from configs/models/<name>.yaml; mirror/adapters_timm.py and mirror/adapters_extra.py build them):
DINOv2 S, B, L, g with registers and B plain, DeiT-III, MAE, SigLIP2, CLIP, DINO v1, DINOv3 (optional, DINOv3 licence),
EVA-02 (optional), ConvNeXt-B and ResNet-50. An optional model that cannot be built (for example no access) is
skipped and recorded in <out>/structure/<model>/SKIPPED.json; every other step goes on.

Steps
  structure  LR structure plus UD as a control, fit seed --fit-seed, --null-draws null draws:
             * ViTs with fc1/fc2 MLPs and qkv attention (DINOv2 S, B, L, B plain, DeiT-III, MAE, SigLIP2, CLIP, DINO v1)
               through scripts/measure_groups.py (also attention; twin-fraction null off unless --twin-null-draws).
               A MUST-04 run of the same model and fit seed with both groups (--reuse-must04) is used as it is;
             * DINOv2-g (SwiGLU), EVA-02 (SwiGLU, RoPE), DINOv3 (RoPE), ConvNeXt-B (pointwise MLPs) and ResNet-50
               (1 x 1 convolutions, at 225 px) through mirror/cnn_groups.py: the same maths and definitions, MLP or
               1 x 1 convolution units only (no attention; mirror/adapters_extra.py says why). The RoPE models also get
               the numerical RoPE flip check (rope_flip_check) and DINOv2-g the SwiGLU conversion check.
             Per model: <out>/structure/<model>/{layers,blocks,null_draws}.csv, metrics.json, meta.json.
  knn        Original k-NN of DINOv2 S, B, L, g (registers) under the paper protocol (bank 100k train, 50k validation
             queries, k = 20, T = 0.07, bf16 features; cache/knn_original/ is reused, as by every earlier script) and
             their parameter counts: <out>/scale/knn.csv, for the accuracy-against-parameter plot (Fig. 5).
  surgery    Every ViT whose LR ratio to null in blocks 1 to L/2 is at most --surgery-threshold (0.8): zero-shot
             Mirror-exact on the first half against plain SVD and the gate-B best baseline (read from
             results/MUST-07/gate_b.json at run time), at matched parameters, sizes <L/2>mlp and <L/2>both
             (--surgery-components), through
             scripts/must07_compressors.py itself (output <out>/surgery/, rows of results/MUST-07 reused when
             identical). MAE is also scored by the ImageNet linear probe of scripts/tier2_linear.py (its accuracy
             measure in the plan): every method's model rebuilt with must07's own code at the hyperparameters it chose
             on the method-selection set. Models whose architecture the Mirror-exact code cannot convert
             (adapters_extra.SURGERY_SUPPORT) are listed with the reason in <out>/surgery/status.csv; so is a run
             that failed (delete its row to retry).
  score      P4.1 (scale), P4.2 (recipe) and P4.3 (CNNs) with the operational definitions below, plus summary.csv
             (every model, LR and UD), tax_by_depth.csv (Fig. 2), scores.json and SUMMARY.md.

Scoring (PREREG_NEXT_PHASE.md CORE-04; "first half" = blocks 1 to L/2, the range "first_half" of metrics.json):
  P4.1  LR tax (tax_mlp, character formula) in the first half of DINOv2 S, B, L, g with registers. Passes if it
        increases strictly at every step S < B < L < g; the direction (increasing, decreasing or mixed) is reported
        either way, as the plan asks.
  P4.2  Models whose recipe is verified to have no flips (configs/models: recipe_flip "no" without a "probable" or
        "verify" confidence: CLIP) have a lower LR tax in the first half than every flip-trained model
        (recipe_flip "yes") of the same size (same width and depth in the yaml architecture: ViT-B, 768 x 12); PASS
        needs every such peer measured ("incomplete" otherwise), FAIL needs one measured peer at or below. DINOv3
        (recipe_flip "no", confidence "probable") is scored separately as a secondary reading; SigLIP2 (unknown)
        is excluded. scripts/collect_next_phase.py applies the same rule.
  P4.3  ConvNeXt-B: LR ratio to null at most 0.8 and twin fraction (0.5) at least 0.2 in the first half of the
        pointwise MLPs in depth order (units 1 to 18 of 36). The first two stages (units 1 to 6) are reported as a
        secondary reading.

Data roles: fit set fit<seed> of ImageNet train for every statistic (Protocol.fit_loader); the k-NN bank (train)
and validation queries for k-NN reporting only; the method-selection set for must07's choices. Nothing is chosen
on validation. GPU memory is capped with --gpu-frac at import (shared GPU). All outputs are resumable.
"""
from __future__ import annotations

import sys

import torch

GPU_FRAC_DEFAULT = 0.2   # of a 141 GB H200: 28 GB; DINOv2-g with every MLP statistic needs about 15 GB (estimate)


def _gpu_frac_from_argv(default: float) -> float:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if a.startswith("--gpu-frac="):
            return float(a.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_gpu_frac_from_argv(GPU_FRAC_DEFAULT))  # cap before allocating

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import adapters_extra as AX  # noqa: E402
from mirror import cnn_groups as CG  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

STEPS = ("structure", "knn", "surgery", "score")
GROUPS = ("LR", "UD")
# role, path ("measure_groups" or "engine"), kind; size classes come from the yaml architecture at run time.
MODELS = {
    "dinov2_vits14_reg": {"role": "scale", "path": "measure_groups", "kind": "vit", "scale": "S"},
    "dinov2_vitb14_reg": {"role": "main", "path": "measure_groups", "kind": "vit", "scale": "B"},
    "dinov2_vitl14_reg": {"role": "scale", "path": "measure_groups", "kind": "vit", "scale": "L"},
    "dinov2_vitg14_reg": {"role": "scale", "path": "engine", "kind": "vit", "scale": "g"},
    "dinov2_vitb14": {"role": "main (no registers)", "path": "measure_groups", "kind": "vit"},
    "deit3_b16": {"role": "supervised", "path": "measure_groups", "kind": "vit"},
    "mae_b16": {"role": "masked reconstruction", "path": "measure_groups", "kind": "vit", "linear_probe": True},
    "siglip2_b16": {"role": "language-supervised", "path": "measure_groups", "kind": "vit"},
    "clip_b16": {"role": "language-supervised", "path": "measure_groups", "kind": "vit"},
    "dino_b16": {"role": "self-supervised, older", "path": "measure_groups", "kind": "vit"},
    "dinov3_b16": {"role": "optional (no flips, probable)", "path": "engine", "kind": "vit", "optional": True},
    "eva02_b14": {"role": "optional", "path": "engine", "kind": "vit", "optional": True},
    "convnext_b": {"role": "CNN", "path": "engine", "kind": "cnn"},
    "resnet50": {"role": "CNN", "path": "engine", "kind": "cnn"},
}
SCALE_ORDER = ("dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg")
CONVNEXT = "convnext_b"
# Tiny stand-ins (CPU tests): the same paths and scoring on random models; their "recipes" only exercise the plumbing.
TINY_MODELS = {
    "tiny": {"role": "tiny GELU ViT", "path": "measure_groups", "kind": "vit", "recipe_flip": "yes"},
    "tiny_eva": {"role": "tiny SwiGLU + RoPE ViT", "path": "engine", "kind": "vit", "recipe_flip": "no"},
    "tiny_convnext": {"role": "tiny ConvNeXt", "path": "engine", "kind": "cnn"},
    "tiny_resnet": {"role": "tiny ResNet", "path": "engine", "kind": "cnn"},
}
TINY_SCALE_ORDER = ()
TINY_CONVNEXT = "tiny_convnext"


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default=",".join(STEPS), help="comma list of " + ", ".join(STEPS))
    p.add_argument("--models", default="all", help="comma list of model names (default every CORE-04 model)")
    p.add_argument("--groups", default=",".join(GROUPS), help="LR plus UD as the control")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--null-draws", type=int, default=20)
    p.add_argument("--twin-null-draws", type=int, default=0, help="measure_groups twin-fraction null (UD)")
    p.add_argument("--seed", type=int, default=0, help="base seed of every null draw")
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (as every earlier run)")
    p.add_argument("--workers", default="auto", help="processes for the neuron matchings (auto: CPU count - 1)")
    p.add_argument("--reuse-must04", default=str(REPO_ROOT / "results" / "MUST-04"),
                   help="MUST-04 output root whose <model>/metrics.json is reused when it has LR and UD ('' to never)")
    p.add_argument("--surgery-threshold", type=float, default=0.8, help="LR ratio to null in blocks 1 to L/2")
    p.add_argument("--surgery-models", default="auto", help="auto (eligible by the threshold) or a comma list")
    p.add_argument("--surgery-components", default="mlp,both",
                   help="must07 sizes <L/2><component>: mlp (MLP only) and/or both (MLP plus attention)")
    p.add_argument("--gate-b", default=str(REPO_ROOT / "results" / "MUST-07" / "gate_b.json"))
    p.add_argument("--allow-no-gate-b", action="store_true",
                   help="run surgery without gate B (Mirror-exact and plain SVD only); never for paper numbers")
    p.add_argument("--structure-dir", default=str(REPO_ROOT / "results" / "MUST-07" / "structure"),
                   help="must07's cached span structures (shared with MUST-07)")
    p.add_argument("--must07-reuse", default=str(REPO_ROOT / "results" / "MUST-07"),
                   help="must07 output whose identical rows are copied ('' for none)")
    p.add_argument("--fp32-flip", default="none", choices=["full", "exact", "none"],
                   help="must07's fp32 flip pass (first-half sizes are not exactly invariant, so none by default)")
    p.add_argument("--queries", type=int, default=None, help="subsample validation queries (debugging only)")
    p.add_argument("--out", default=None, help="default results/CORE-04 (tiny: tiny_out/core04)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-frac", type=float, default=GPU_FRAC_DEFAULT,
                   help="torch.cuda.set_per_process_memory_fraction (read at import)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--tiny", action="store_true", help="random tiny models and synthetic data on CPU")
    p.add_argument("--tiny-force-surgery", action="store_true",
                   help="tiny only: run the surgery step on the tiny GELU ViT whatever its ratio (plumbing test)")
    return p.parse_args(argv)


def _workers(a) -> int:
    if str(a.workers) == "auto":
        return 1 if a.tiny else max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(a.workers))


def _models(a) -> dict:
    table = TINY_MODELS if a.tiny else MODELS
    if a.models == "all":
        return dict(table)
    names = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in names if m not in table]
    if bad:
        raise SystemExit(f"unknown models {bad}; known: {sorted(table)}")
    return {m: table[m] for m in names}


def _free(device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def load_script(name: str):
    """Import scripts/<name>.py once (its import-time GPU cap reads this process's --gpu-frac)."""
    key = f"_core04_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


# ----------------------------------------------------------------------------------------------
# Model records (yaml, read at run time; never filled from memory)
# ----------------------------------------------------------------------------------------------
def versions() -> dict:
    import timm

    return {"timm": timm.__version__, "torch": torch.__version__, "numpy": np.__version__}


def model_record(name: str, tiny: bool) -> dict:
    """role, recipe and architecture of a model from configs/models/<name>.yaml (tiny: the TINY_MODELS entry)."""
    if tiny:
        e = TINY_MODELS[name]
        return {"name": name, "role": e["role"], "recipe_flip": e.get("recipe_flip", "yes"), "recipe_verified": True,
                "width": "tiny", "blocks": "tiny", "size_class": "CNN" if e["kind"] == "cnn" else "tiny", "yaml": None}
    import yaml

    path = REPO_ROOT / "configs" / "models" / f"{name}.yaml"
    y = yaml.safe_load(path.read_text(encoding="utf-8"))
    arch = y.get("architecture", {}) or {}
    flip = str(y.get("recipe_flip", "unknown")).strip().lower()
    conf = str(y.get("recipe_flip_confidence", "")).lower()
    width, blocks = arch.get("width"), arch.get("blocks")
    kind = MODELS[name]["kind"]
    size = "CNN" if kind == "cnn" else (f"{width}x{blocks}" if isinstance(width, int) else str(width))
    src = y.get("source", {}) or {}
    return {"name": name, "role": MODELS[name]["role"], "recipe_flip": flip,
            "recipe_verified": flip in ("yes", "no") and "probable" not in conf and "verify" not in conf,
            "recipe_flip_confidence": conf or None, "width": width, "blocks": blocks, "size_class": size,
            "mlp_type": arch.get("mlp_type"), "yaml": str(path.relative_to(REPO_ROOT)),
            "id": src.get("timm_name") or src.get("entry_point"), "revision": (y.get("revision") or {}).get("value")}


# ----------------------------------------------------------------------------------------------
# Runtime pieces
# ----------------------------------------------------------------------------------------------
def protocol(a, img_size: int = 56):
    if a.tiny:
        from mirror.variants import TinyProtocol

        return TinyProtocol(img_size=img_size)
    from mirror.protocol import Protocol

    return Protocol(a.cache_dir, a.batch_size, a.stats_batch_size, a.num_workers, torch.cuda.is_available())


@torch.no_grad()
def _randomise_bn(model, seed: int) -> None:
    """Non-trivial BatchNorm statistics and affines for tiny random CNNs (timm zero-initialises bn3.weight)."""
    g = torch.Generator().manual_seed(seed + 777)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            n = mod.num_features
            mod.weight.copy_(1.0 + 0.3 * torch.randn(n, generator=g))
            mod.bias.copy_(0.1 * torch.randn(n, generator=g))
            mod.running_mean.copy_(0.1 * torch.randn(n, generator=g))
            mod.running_var.copy_(1.0 + 0.5 * torch.rand(n, generator=g))


def build_tiny(name: str):
    """Tiny random stand-ins for the engine path (no downloads)."""
    from mirror.adapters import _randomise_affine

    torch.manual_seed(0)
    if name == "tiny_eva":
        ad = AX.build_vit_adapter(AX.EXTRA_REGISTRY["eva02_b14"], img_size=56, pretrained=False, embed_dim=64,
                                  depth=4, num_heads=2, ref_feat_shape=(4, 4))
        _randomise_affine(ad.model, 0)
        return ad
    if name == "tiny_convnext":
        ad = AX.build_cnn_adapter(AX.EXTRA_REGISTRY["convnext_b"], pretrained=False, input_size=64,
                                  depths=(1, 1, 2, 1), dims=(8, 16, 16, 32))
        _randomise_affine(ad.model, 0)
        return ad
    if name == "tiny_resnet":
        ad = AX.build_cnn_adapter(AX.EXTRA_REGISTRY["resnet50"], pretrained=False, input_size=65,
                                  layers=(1, 1, 1, 1), channels=(4, 8, 8, 16), stem_width=8)
        _randomise_bn(ad.model, 0)
        return ad
    raise KeyError(name)


def build_engine_model(name: str, a, device):
    if a.tiny:
        return build_tiny(name).to(device=device)
    return AX.build_extra_adapter(name, device=device, pretrained=True)


def engine_spec(name: str, adapter):
    if isinstance(adapter, AX.CNNAdapter):
        return CG.convnext_spec(adapter, name) if adapter.family == "convnext" else CG.resnet_spec(adapter, name)
    return CG.vit_spec(adapter, name=name)


def device_of(a) -> torch.device:
    if a.tiny or a.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    return torch.device("cuda")


# ----------------------------------------------------------------------------------------------
# Step: structure
# ----------------------------------------------------------------------------------------------
def structure_dir(out: Path, name: str) -> Path:
    return out / "structure" / name


def must04_metrics(a, name: str):
    """A MUST-04 metrics.json of this model and fit seed that has every requested group, else None."""
    if a.tiny or not a.reuse_must04:
        return None
    p = Path(a.reuse_must04) / name / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(m.get("fit_seed", -1)) != int(a.fit_seed):
        return None
    if not all(g in m.get("groups", {}) and "first_half" in m["groups"][g].get("ranges", {}) for g in a.groups_list):
        return None
    return p


def metrics_path(a, out: Path, name: str):
    """Where the model's structure metrics live (own run first, then a reusable MUST-04 run)."""
    own = structure_dir(out, name) / "metrics.json"
    if own.exists():
        return own
    return must04_metrics(a, name)


def run_structure(a, out: Path, name: str, entry: dict, device, git: dict) -> dict:
    sdir = structure_dir(out, name)
    if (sdir / "metrics.json").exists():
        try:
            m = json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
            if all(g in m.get("groups", {}) for g in a.groups_list):
                log(f"{name}: structure present ({sdir / 'metrics.json'})")
                return {"status": "done", "source": str(sdir / "metrics.json")}
        except Exception:
            pass
    reuse = must04_metrics(a, name)
    if reuse is not None:
        log(f"{name}: reusing MUST-04 structure {reuse}")
        return {"status": "reused", "source": str(reuse)}
    t0 = time.time()
    sdir.mkdir(parents=True, exist_ok=True)
    if entry["path"] == "measure_groups":
        mg = load_script("measure_groups")
        argv = ["--groups", ",".join(a.groups_list), "--fit-seed", str(a.fit_seed), "--null-draws", str(a.null_draws),
                "--twin-null-draws", str(a.twin_null_draws), "--seed", str(a.seed), "--top-k", str(a.top_k),
                "--workers", str(_workers(a)), "--experiment", "CORE-04", "--out", str(sdir),
                "--cache-dir", a.cache_dir, "--batch-size", str(a.batch_size),
                "--stats-batch-size", str(a.stats_batch_size), "--num-workers", str(a.num_workers),
                "--gpu-frac", str(a.gpu_frac)]
        argv += ["--tiny", "--no-heldout"] if a.tiny else ["--model", name]
        if a.device == "cpu":
            argv += ["--device", "cpu"]
        mg.main(argv)
        _free(device)
        return {"status": "measured", "source": str(sdir / "metrics.json"), "seconds": round(time.time() - t0, 1)}
    # ---- engine path
    try:
        adapter = build_engine_model(name, a, device)
    except Exception as exc:                                    # optional models: no access, no weights
        if not entry.get("optional"):
            raise
        info = {"model": name, "skipped": True, "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (sdir / "SKIPPED.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log(f"{name}: SKIPPED ({info['reason']})")
        return {"status": "skipped", "reason": info["reason"]}
    checks = {}
    if isinstance(adapter, AX.EvaAdapter):
        checks["rope_flip"] = {g: AX.rope_flip_check_adapter(adapter, g) for g in a.groups_list}
        checks["attention"] = "not measured (RoPE: exact head maps would have to respect the rotary sign map)"
    if isinstance(adapter, AX.Dinov2SwiGLUAdapter):
        checks["swiglu_conversion"] = adapter.check_conversion()
    if not isinstance(adapter, AX.CNNAdapter) and getattr(adapter, "mlp_kind", "gelu") == "swiglu":
        checks.setdefault("attention", "not measured (engine path: MLP units only)")
    spec = engine_spec(name, adapter)
    img = 64 if a.tiny and isinstance(adapter, AX.CNNAdapter) else 56
    proto = protocol(a, img)
    metrics = CG.run_measurement(spec, lambda: proto.fit_loader(a.fit_seed), device, sdir, groups=a.groups_list,
                                 fit_seed=a.fit_seed, null_draws=a.null_draws, seed=a.seed, top_k=a.top_k,
                                 workers=_workers(a), git=git, log=log)
    metrics["checks"] = checks
    metrics["model"] = name
    (sdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    write_meta(sdir, experiment="CORE-04", config=vars(a),
               seeds={"fit_seed": a.fit_seed, "null_seed_base": a.seed, "null_draws": a.null_draws},
               models={name: model_record(name, a.tiny)}, device=device, wall_seconds=time.time() - t0,
               extra={"step": "structure", "engine": "mirror/cnn_groups.py", "checks": checks, "versions": versions(),
                      "peak_gpu_memory_bytes": peak, "spec": spec.info})
    del adapter, spec
    _free(device)
    return {"status": "measured", "source": str(sdir / "metrics.json"), "seconds": round(time.time() - t0, 1),
            "checks": checks}


# ----------------------------------------------------------------------------------------------
# Step: knn (DINOv2 family)
# ----------------------------------------------------------------------------------------------
def run_knn(a, out: Path, device, git: dict) -> list:
    from mirror import variants as V
    from mirror.adapters_timm import build_model_adapter

    names = ["tiny"] if a.tiny else [m for m in SCALE_ORDER if m in _models(a)]
    csv = V.ResultsCsv(out / "scale" / "knn.csv", ["model"])
    rows = []
    for name in names:
        if csv.done((name,)):
            continue
        t0 = time.time()
        if a.tiny:
            from mirror.adapters import TimmTinyAdapter

            adapter, proto, ncls = TimmTinyAdapter(img_size=56, seed=0).to(device=device), protocol(a), 5
        else:
            adapter = build_model_adapter(name, device=device)
            proto, ncls = protocol(a), 1000
        amp = None if a.tiny else "bf16"
        r = V.original_knn(adapter, proto.bank_loader(), proto.query_loader(a.queries), device, ncls, name, amp,
                           a.cache_dir, a.queries, a.tiny, git["git_hash"])
        m = adapter.model
        n_all = int(sum(p.numel() for p in m.parameters()))
        n_blocks = int(sum(p.numel() for blk in m.blocks for p in blk.parameters()))
        row = {"model": name, "top1": r["top1"], "flip_consistency": r["flip_consistency"], "source": r["source"],
               "params_total": n_all, "params_blocks": n_blocks, "depth": len(m.blocks),
               "embed_dim": int(adapter.embed_dim), "mlp_hidden": int(adapter.mlp_hidden),
               "n_query": "all" if a.queries is None else int(a.queries), "amp": amp or "fp32",
               "git_hash": git["git_hash"], "time_s": round(time.time() - t0, 1)}
        csv.add(row)
        rows.append(row)
        log(f"k-NN {name}: top-1 {r['top1']:.2f}, {n_all:,} parameters")
        del adapter
        _free(device)
    return rows


# ----------------------------------------------------------------------------------------------
# Step: surgery
# ----------------------------------------------------------------------------------------------
def read_gate_b(a) -> dict | None:
    p = Path(a.gate_b)
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    return g if g.get("best") else None


def surgery_support(name: str, tiny: bool) -> str:
    if tiny:
        return "must07" if name == "tiny" else "engine model (no Mirror-exact path)"
    return AX.SURGERY_SUPPORT.get(name, "must07")


def lr_first_half(path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    r = m["groups"]["LR"]["ranges"]["first_half"]
    return {"ratio_to_null": r.get("ratio_to_null"), "blocks": r.get("blocks"), "depth": m.get("depth")}


def run_surgery(a, out: Path, device, git: dict) -> list:
    from mirror import variants as V

    status = V.ResultsCsv(out / "surgery" / "status.csv", ["model"])
    gate = read_gate_b(a)
    if gate is None and not (a.allow_no_gate_b or a.tiny):
        log(f"surgery: {a.gate_b} missing or without a best baseline; run MUST-07 first (gate B). Skipped.")
        return []
    best = gate["best"] if gate else None
    methods = ["mirror_exact", "svd"] + ([best] if best and best not in ("mirror_exact", "svd") else [])
    forced = [m.strip() for m in a.surgery_models.split(",")] if a.surgery_models != "auto" else None
    rows = []
    for name, entry in _models(a).items():
        if entry["kind"] != "vit" or status.done((name,)):
            continue
        mp = metrics_path(a, out, name)
        if mp is None:
            log(f"surgery {name}: no structure metrics yet; skipped")
            continue
        lr = lr_first_half(mp)
        ratio = lr["ratio_to_null"]
        eligible = ratio is not None and np.isfinite(ratio) and ratio <= a.surgery_threshold
        if forced is not None:
            eligible = name in forced
        if a.tiny and a.tiny_force_surgery and name == "tiny":
            eligible = True
        support = surgery_support(name, a.tiny)
        row = {"model": name, "lr_ratio_to_null_first_half": ratio, "threshold": a.surgery_threshold,
               "eligible": bool(eligible), "support": support, "gate_b_best": best or "",
               "methods": ",".join(methods), "status": "", "git_hash": git["git_hash"]}
        if not eligible:
            row["status"] = "not eligible (ratio above threshold)"
        elif support != "must07":
            row["status"] = "eligible, not run: " + support
        else:
            t0 = time.time()
            k = len(lr["blocks"]) if lr.get("blocks") else None
            try:
                row.update(run_must07(a, out, name, k, methods, device))
                if MODELS.get(name, {}).get("linear_probe") or (a.tiny and name == "tiny" and a.tiny_force_surgery):
                    row.update(run_linear(a, out, name, k, methods, device, git))
            except Exception as exc:              # one model's failure is recorded; the others still run
                row["status"] = f"failed: {type(exc).__name__}: {exc}"[:500]
                log(f"surgery {name} FAILED: {traceback.format_exc()}")
                if a.tiny:
                    raise
            row["time_s"] = round(time.time() - t0, 1)
        status.add(row)
        rows.append(row)
        log(f"surgery {name}: {row['status']}")
        _free(device)
    return rows


def surgery_sizes(a, k: int) -> list:
    comps = [c.strip() for c in a.surgery_components.split(",") if c.strip()]
    bad = [c for c in comps if c not in ("mlp", "attn", "both")]
    if bad:
        raise SystemExit(f"unknown surgery components {bad}")
    return [f"{k}{c}" for c in comps]


def _must07_argv(a, out: Path, name: str, k: int, methods: list) -> list:
    argv = ["--fit-seeds", str(a.fit_seed), "--sizes", ",".join(surgery_sizes(a, k)), "--methods", ",".join(methods),
            "--out", str(out / "surgery"), "--fp32-flip", a.fp32_flip, "--gpu-frac", str(a.gpu_frac),
            "--workers", str(_workers(a)), "--cache-dir", a.cache_dir, "--num-workers", str(a.num_workers),
            "--stats-batch-size", str(a.stats_batch_size)]
    if a.tiny:
        argv += ["--tiny", "--structure-dir", str(out / "surgery" / "structure")]
    else:
        argv += ["--model", name, "--structure-dir", a.structure_dir]
        if a.must07_reuse:
            argv += ["--reuse-from", a.must07_reuse]
        if a.queries:
            argv += ["--queries", str(a.queries)]
    return argv


def run_must07(a, out: Path, name: str, k: int, methods: list, device) -> dict:
    m7 = load_script("must07_compressors")
    res = m7.main(_must07_argv(a, out, name, k, methods))
    return {"status": f"run: must07 sizes {','.join(surgery_sizes(a, k))}", "must07_rows_added": res.get("rows_added"),
            "must07_out": str(out / "surgery"), "compressor_api": res.get("compressor_api")}


def run_linear(a, out: Path, name: str, k: int, methods: list, device, git: dict) -> dict:
    """ImageNet linear probe (scripts/tier2_linear.py protocol) of the original and every surgery row of `name`:
    models rebuilt with must07's own code (structure from its cache, compressors at the hyperparameters must07
    chose on the method-selection set)."""
    import pandas as pd

    from mirror import mirror_family as MF
    from mirror import variants as V

    m7 = load_script("must07_compressors")
    t2 = load_script("tier2_linear")
    t2.apply_memory_cap(a.gpu_frac)
    model_name = "tiny" if a.tiny else name
    csv = V.ResultsCsv(out / "surgery" / f"linear_{model_name}.csv", ["model", "size", "method"])
    adapter, proto, ncls = V.build_runtime(name, a.tiny, device, a.cache_dir, a.batch_size, a.stats_batch_size,
                                           0 if a.tiny else a.num_workers)
    adapter.model.float().eval()
    ns = argparse.Namespace(force_structure=False, top_k=3 if a.tiny else a.top_k, register_margin=0.0,
                            operators_template=None, workers=_workers(a))
    sdir = (out / "surgery" / "structure") if a.tiny else Path(a.structure_dir)
    struct, _ = m7.get_structure(ns, adapter, proto, device, model_name, a.fit_seed, sdir)
    rows_csv = out / "surgery" / f"rows_{model_name}.csv"
    chosen = pd.read_csv(rows_csv) if rows_csv.exists() else pd.DataFrame()
    api = m7.compressor_api()
    stats = None
    amp = None if a.tiny else "bf16"
    grid = t2.DEFAULT_WD_GRID.split(",")
    grid = [float(x) for x in grid]
    done = 0
    targets = [("original", "original")] + [(sz, m) for sz in surgery_sizes(a, k) for m in methods]
    for size_name, method in targets:
        if csv.done((model_name, size_name, method)):
            continue
        t0 = time.time()
        if method == "original":
            model, hp = adapter.model, {}
        else:
            size = m7.parse_size(size_name, adapter.depth)
            budget = MF.span_budget(adapter.model, struct, range(size.k), size.components, model_name)
            if method in MF.MIRROR_METHODS:
                model, _ = MF.build_mirror_variant(adapter, method, range(size.k), size.components, struct,
                                                   seed=1 + a.fit_seed, budget=budget)
                hp = {}
            else:
                sel = chosen[(chosen.get("size") == size.name) & (chosen.get("method") == method)
                             & (chosen.get("fit_seed") == a.fit_seed) & (chosen.get("status") == "ok")] \
                    if len(chosen) else chosen
                if not len(sel):
                    log(f"linear {model_name} {size_name} {method}: no must07 row; skipped")
                    continue
                hp = json.loads(sel.iloc[0]["hparams"]) if isinstance(sel.iloc[0]["hparams"], str) else {}
                if stats is None:
                    stats = api.collect_stats(adapter, proto.fit_loader(a.fit_seed), device, list(range(adapter.depth)),
                                              [m for m in methods if m in api.methods], MF.COMPONENTS)
                model, _ = api.compress(adapter.model, method, list(range(size.k)), size.components, budget, stats, hp)
        Xb, yb = t2.extract_features(adapter, proto.bank_loader(), device, model=model, amp=amp, dtype=np.float32)
        Xv, yv = t2.extract_features(adapter, proto.query_loader(a.queries), device, model=model, amp=amp,
                                     dtype=np.float32)
        res = t2.run_probe(Xb, yb, Xv, yv, ncls, grid, device=device)
        csv.add({"model": model_name, "size": size_name, "method": method, "top1": res["test"]["top1"],
                 "top5": res["test"].get("top5"), "chosen_wd": res["final"]["wd"], "hparams": json.dumps(hp),
                 "n_bank": int(len(yb)), "n_val": int(len(yv)), "feature": t2.FEATURE_RECIPE,
                 "fit_seed": a.fit_seed, "git_hash": git["git_hash"], "time_s": round(time.time() - t0, 1)})
        done += 1
        log(f"linear {model_name} {size_name} {method}: top-1 {res['test']['top1']:.2f}")
        if model is not adapter.model:
            del model
        _free(device)
    return {"linear_rows_added": done, "linear_csv": str(csv.path)}


# ----------------------------------------------------------------------------------------------
# Step: score
# ----------------------------------------------------------------------------------------------
def _rng(m: dict, g: str, name: str) -> dict:
    return ((m.get("groups", {}).get(g, {}) or {}).get("ranges", {}) or {}).get(name, {}) or {}


def summary_rows(a, out: Path, models: dict) -> tuple:
    import pandas as pd

    rows, depth_rows, missing = [], [], {}
    for name in models:
        rec = model_record(name, a.tiny)
        mp = metrics_path(a, out, name)
        skipped = structure_dir(out, name) / "SKIPPED.json"
        if mp is None:
            missing[name] = "skipped: " + json.loads(skipped.read_text())["reason"] if skipped.exists() else "not run"
            continue
        m = json.loads(Path(mp).read_text(encoding="utf-8"))
        row = {**{k: rec.get(k) for k in ("name", "role", "size_class", "width", "blocks", "recipe_flip",
                                          "recipe_verified", "recipe_flip_confidence", "id")},
               "path": models[name]["path"], "metrics": str(mp), "engine": m.get("engine", "scripts/measure_groups.py")}
        for g in a.groups_list:
            for rn in ("first_half", "second_half"):
                r = _rng(m, g, rn)
                for key in ("ratio_to_null", "twin_fraction_t50", "tax_mlp", "e_ratio_I_mean", "null_over_fit_mean",
                            "ratio_to_null_act", "s_attn_ratio_to_null"):
                    if key in r:
                        row[f"{g}_{key}_{rn}"] = r[key]
                if "s_mlp_null_lo" in r:
                    row[f"{g}_s_mlp_null_95_{rn}"] = f"{r['s_mlp_null_lo']:.4f}-{r['s_mlp_null_hi']:.4f}"
            for rn in ("stages_1-2",):
                r = _rng(m, g, rn)
                for key in ("ratio_to_null", "twin_fraction_t50", "tax_mlp"):
                    if key in r:
                        row[f"{g}_{key}_{rn}"] = r[key]
        rows.append(row)
        bdir = Path(mp).parent / "blocks.csv"
        if bdir.exists():
            b = pd.read_csv(bdir)
            L = int(b["block"].max())
            for _, r in b[b["group"].isin(a.groups_list)].iterrows():
                tax = r.get("tax_mlp")
                if tax is None or (isinstance(tax, float) and not np.isfinite(tax)):
                    den = 2.0 * r["h"] * r["d"]
                    tax = 1.0 - (r["free_fc1"] + r["free_fc2"]) / den
                depth_rows.append({"model": name, "group": r["group"], "block": int(r["block"]), "depth": L,
                                   "depth_frac": (int(r["block"]) - 0.5) / L, "tax_mlp": float(tax),
                                   "s_mlp": r.get("s_mlp"), "s_mlp_null_mean": r.get("s_mlp_null_mean"),
                                   "ratio_to_null": (r.get("s_mlp") / r.get("s_mlp_null_mean")
                                                     if r.get("s_mlp_null_mean") else float("nan")),
                                   "twin_fraction_t50": r.get("frac_2_t50"), "role": rec["role"],
                                   "recipe_flip": rec["recipe_flip"]})
    return rows, depth_rows, missing


def score_p41(rows: list, order) -> dict:
    by = {r["name"]: r for r in rows}
    vals = [(n, (by.get(n) or {}).get("LR_tax_mlp_first_half")) for n in order]
    have = [(n, v) for n, v in vals if v is not None and np.isfinite(v)]
    res = {"prediction": "LR tax in blocks 1 to L/2 increases from DINOv2-S to g (S < B < L < g)",
           "operational": "strictly increasing at every step; direction reported either way",
           "values": dict(vals), "missing": [n for n, v in vals if not (v is not None and np.isfinite(v))]}
    if len(have) < len(order) or not order:
        res.update({"verdict": "not scored", "direction": None})
        return res
    t = [v for _, v in have]
    inc = all(b > a for a, b in zip(t, t[1:]))
    dec = all(b < a for a, b in zip(t, t[1:]))
    res.update({"direction": "increasing" if inc else ("decreasing" if dec else "mixed"),
                "g_minus_S": t[-1] - t[0], "verdict": "PASS" if inc else "FAIL"})
    return res


def score_p42(rows: list, tiny: bool) -> dict:
    res = {"prediction": "models whose recipe is verified to have no flips have a lower LR tax than every "
                         "flip-trained model of the same size",
           "operational": "LR tax_mlp in the first half; verified no-flip = recipe_flip 'no' without a 'probable' or "
                          "'verify' confidence; flip-trained = recipe_flip 'yes'; same size = same yaml width and "
                          "depth; PASS needs every flip-trained peer measured, FAIL needs one measured peer at or "
                          "below the no-flip model"}
    out = {}
    for reading, ok_nf in (("primary (verified no-flip)", lambda r: r["recipe_verified"]),
                           ("secondary (with probable no-flip)", lambda r: True)):
        nf = [r for r in rows if r["recipe_flip"] == "no" and ok_nf(r) and r.get("size_class") != "CNN"]
        comps = []
        for r in nf:
            same = [x for x in rows if x["recipe_flip"] == "yes" and x["size_class"] == r["size_class"]]
            tv = r.get("LR_tax_mlp_first_half")
            others = {x["name"]: x.get("LR_tax_mlp_first_half") for x in same}
            ok_vals = [v for v in others.values() if v is not None and np.isfinite(v)]
            if tv is None or not np.isfinite(tv) or not ok_vals:
                comps.append({"model": r["name"], "tax": tv, "flip_trained": others, "holds": None})
                continue
            holds = bool(tv < min(ok_vals))
            comps.append({"model": r["name"], "tax": tv, "flip_trained": others, "min_flip_trained": min(ok_vals),
                          "holds": holds if (not holds or len(ok_vals) == len(others)) else None,
                          "complete": len(ok_vals) == len(others)})
        if any(c["holds"] is False for c in comps):
            verdict = "FAIL"                              # one measured flip-trained peer is already not above it
        elif comps and all(c["holds"] is True for c in comps):
            verdict = "PASS"
        else:
            verdict = "not scored" if not any(c.get("min_flip_trained") is not None for c in comps) else "incomplete"
        out[reading] = {"comparisons": comps, "verdict": verdict}
    res.update(out)
    res["verdict"] = out["primary (verified no-flip)"]["verdict"]
    return res


def score_p43(rows: list, convnext: str) -> dict:
    by = {r["name"]: r for r in rows}
    res = {"prediction": "twin structure appears in ConvNeXt's pointwise MLPs",
           "operational": "LR ratio to null at most 0.8 and twin fraction at least 0.2 in the first half of the "
                          "pointwise MLPs (depth order)"}
    r = by.get(convnext)
    if r is None:
        res["verdict"] = "not scored"
        return res
    for reading, rn in (("primary", "first_half"), ("secondary (first two stages)", "stages_1-2")):
        ratio, tw = r.get(f"LR_ratio_to_null_{rn}"), r.get(f"LR_twin_fraction_t50_{rn}")
        ok = (ratio is not None and tw is not None and np.isfinite(ratio) and np.isfinite(tw))
        res[reading] = {"range": rn, "ratio_to_null": ratio, "twin_fraction_t50": tw,
                        "verdict": ("PASS" if ratio <= 0.8 and tw >= 0.2 else "FAIL") if ok else "not scored"}
    res["verdict"] = res["primary"]["verdict"]
    return res


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}" if x is not None and np.isfinite(float(x)) else "n/a"
    except (TypeError, ValueError):
        return str(x)


def write_summary(out: Path, a, rows, missing, scores, knn_rows, surg_rows, git) -> Path:
    L = ["# CORE-04: generality, scale and training recipes", "",
         f"Commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}; fit seed {a.fit_seed}; "
         f"{a.null_draws} null draws; groups {', '.join(a.groups_list)}. Every number below is read from the files "
         "named in summary.csv and scores.json.", "",
         "## Pre-registered predictions (PREREG_NEXT_PHASE.md CORE-04)", ""]
    for key in ("P4.1", "P4.2", "P4.3"):
        s = scores[key]
        L.append(f"- **{key}** {s['prediction']}: **{s['verdict']}**" +
                 (f" (direction {s['direction']})" if s.get("direction") else ""))
    L += ["", "## Structure, blocks 1 to L/2 (LR, with UD as the control)", "",
          "| Model | Role | Recipe flips | LR ratio to null | LR twin fraction | LR tax | UD ratio to null | UD tax |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['role']} | {r['recipe_flip']} | {_f(r.get('LR_ratio_to_null_first_half'))} | "
                 f"{_f(r.get('LR_twin_fraction_t50_first_half'))} | {_f(r.get('LR_tax_mlp_first_half'))} | "
                 f"{_f(r.get('UD_ratio_to_null_first_half'))} | {_f(r.get('UD_tax_mlp_first_half'))} |")
    if missing:
        L += ["", "Not measured: " + "; ".join(f"{k} ({v})" for k, v in missing.items()) + "."]
    if knn_rows:
        L += ["", "## DINOv2 family, original k-NN (paper protocol)", "", "| Model | Top-1 | Parameters |",
              "|---|---|---|"]
        for r in knn_rows:
            L.append(f"| {r['model']} | {_f(r['top1'], 2)} | {int(r['params_total']):,} |")
    if surg_rows:
        L += ["", "## Surgery status", "", "| Model | LR ratio to null | Eligible | Status |", "|---|---|---|---|"]
        for r in surg_rows:
            L.append(f"| {r['model']} | {_f(r.get('lr_ratio_to_null_first_half'))} | {r.get('eligible')} | "
                     f"{r.get('status')} |")
    L += ["", "Notes: the RoPE models (DINOv3, EVA-02), DINOv2-g and the CNNs are measured on their MLP (or 1 x 1 "
          "convolution) units only; attention is not measured for them (mirror/adapters_extra.py). ResNet-50 runs at "
          "225 px so that the flip is exact on every feature grid. P4.2 compares ViT-B models only (same yaml width "
          "and depth)."]
    path = out / "SUMMARY.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def run_score(a, out: Path, git: dict) -> dict:
    import pandas as pd

    models = _models(a) if a.models != "all" else (TINY_MODELS if a.tiny else MODELS)
    rows, depth_rows, missing = summary_rows(a, out, models)
    order = TINY_SCALE_ORDER if a.tiny else SCALE_ORDER
    scores = {"P4.1": score_p41(rows, order), "P4.2": score_p42(rows, a.tiny),
              "P4.3": score_p43(rows, TINY_CONVNEXT if a.tiny else CONVNEXT), "missing": missing,
              "git": git, "fit_seed": a.fit_seed, "tiny": bool(a.tiny)}
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(out / "tax_by_depth.csv", index=False)
    (out / "scores.json").write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")
    knn = pd.read_csv(out / "scale" / "knn.csv").to_dict("records") if (out / "scale" / "knn.csv").exists() else []
    if order:                                          # tax against width with the original k-NN (Fig. 2 and 5)
        by, kb = {r["name"]: r for r in rows}, {r["model"]: r for r in knn}
        scale = []
        for n in order:
            rec, r, k = model_record(n, a.tiny), by.get(n, {}), kb.get(n, {})
            scale.append({"model": n, "width": rec["width"], "depth": rec["blocks"],
                          **{key: r.get(key) for key in ("LR_tax_mlp_first_half", "LR_ratio_to_null_first_half",
                                                         "LR_twin_fraction_t50_first_half", "UD_tax_mlp_first_half",
                                                         "UD_ratio_to_null_first_half")},
                          "knn_top1": k.get("top1"), "params_total": k.get("params_total")})
        (out / "scale").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scale).to_csv(out / "scale" / "scale.csv", index=False)
    surg = (pd.read_csv(out / "surgery" / "status.csv").to_dict("records")
            if (out / "surgery" / "status.csv").exists() else [])
    write_summary(out, a, rows, missing, scores, knn, surg, git)
    for k in ("P4.1", "P4.2", "P4.3"):
        log(f"{k}: {scores[k]['verdict']}")
    return scores


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    a.groups_list = [g.strip() for g in a.groups.split(",") if g.strip()]
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"unknown steps {bad}; choose from {STEPS}")
    out = Path(a.out) if a.out else (REPO_ROOT / ("tiny_out/core04" if a.tiny else "results/CORE-04"))
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = device_of(a) if any(s in steps for s in ("structure", "knn", "surgery")) else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    git = git_info()
    models = _models(a)
    log(f"CORE-04 {'TINY ' if a.tiny else ''}steps {steps}, models {list(models)}, device {device}, "
        f"commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")
    result = {"out": str(out), "structure": {}}
    if "structure" in steps:
        for name, entry in models.items():
            result["structure"][name] = run_structure(a, out, name, entry, device, git)
    if "knn" in steps:
        result["knn"] = run_knn(a, out, device, git)
    if "surgery" in steps:
        result["surgery"] = run_surgery(a, out, device, git)
    if "score" in steps:
        result["scores"] = run_score(a, out, git)
    write_meta(out, experiment="CORE-04", config={k: v for k, v in vars(a).items()},
               seeds={"fit_seed": a.fit_seed, "null_seed_base": a.seed, "null_draws": a.null_draws, "global": a.seed},
               models={n: model_record(n, a.tiny) for n in models}, device=device,
               wall_seconds=time.time() - t_start,
               extra={"steps": steps, "structure": result["structure"], "versions": versions(),
                      "peak_gpu_memory_bytes": (int(torch.cuda.max_memory_allocated()) if device.type == "cuda"
                                                else None)})
    log(f"done in {(time.time() - t_start) / 60:.1f} min; outputs in {out}")
    return result


if __name__ == "__main__":
    main()

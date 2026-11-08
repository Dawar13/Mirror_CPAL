#!/usr/bin/env python
"""M2: efficient implementation of mirror-converted blocks and latency (PAPER_EXPERIMENTS.md, M2).

    python scripts/m2_latency.py --model dinov2_vitb14_reg --k 6,9,12 --out runs/m2
    python scripts/m2_latency.py --tiny --out tiny_out/m2        # tiny random ViT, planted structure, CPU

For every k the projected dense models are built with the M1 builders (scripts/m1_attention.make_variant)
from the M1 structure (--structure: head pairings and maps, the A_out choice, span operators A* and
register pairings) and the X2 operators (--operators: per-layer operators and neuron pairings), and
each is turned into an efficient model computing the same function (mirror.efficient.FastViT):
  vlocal     Mirror V-local, MLP + attention: per-layer operators, one d x d stream rotation per converted
             block (counted). When M1 projected Wo with A_(l+1) (its choice), Wo is stored dense in the
             block basis (the rotation folded into it).
  vlocal_r2  (optional) V-local with Wo kept structured and a second rotation (of the attention output).
  vexact     Mirror V-exact, MLP + attention: one operator for the span, no per-token rotation (the
             rotation in is folded into the embeddings; the rotation out acts on the CLS feature only).
  mlp_only   Mirror V-local, MLP only (A6).
  random     (optional) matched random structure (the same costs as vlocal).
Timing baselines in the same implementation (FastViT; folded norms, the same attention kernel):
  dense      the folded original (k = 0), the reference for speed-ups;
  svd        SVD factors of fc1, fc2, Wq, Wk, Wv, Wo at the M1 ranks (matched to Mirror V-local);
  prune      MLP width pruning, attention dense: per block h' = (Mirror V-local matrix parameters - dense
             attention) / (2 d), keeping the neurons with the largest ||W1[i]|| ||W2[:, i]|| (weights only).
plus `original`, the released hub model as it runs out of the box (k = 0).

Checks on --check-images validation images (fixed seed), each efficient model against the dense model it
was built from (projected dense; for svd and prune the dense model with truncated matrices or zeroed
neurons; for dense the original): float64 (both built in float64; max relative error of the CLS feature
under 1e-6: with exact operators the algebra gives about 1e-15 (tests), but fitted operators stored in
float32 fix the all-ones direction only to about 1e-8, so after LayerNorm folding (which centres every write)
a tiny non-equivariant part remains that the even/odd form drops; it is reported as the dropped share and
the operators' |A 1 - 1| is recorded), fp32 with TF32 off (max abs difference of the L2-normalised CLS feature,
the k-NN feature, under 1e-5) and bf16 (under 1e-3). Raw-feature errors and the bf16 noise of the dense
model itself (bf16 against its own fp32 output) are reported next to them.

Timing: bf16 weights and inputs, torch.compile (default mode, static shapes; eager if compilation fails,
recorded) and eager, batch 1, 64 and 256; per run CUDA events around one forward, median of --iters runs
after --warmup runs. Analytic multiply-adds per image: patch embedding, every matrix (free parameters;
low-rank r (in + out)), rotations (d^2 per token), token mixing 2 T^2 D per block, CLS rotation; not
biases, norms, softmax, GELU or the butterfly additions; executed counts add the alignment padding.

Outputs (rows carry the git hash and seed): m2_latency.csv, m2_checks.csv, m2_meta.json, M2_SUMMARY.md.
Rerunning skips rows already in m2_latency.csv (use --force to redo them).
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.attention import HeadMaps, attention_free_params, attn_dims  # noqa: E402
from mirror.data import normalize_for  # noqa: E402
from mirror.efficient import Basis, BlockSpec, build_fast_vit, fast_vit_cost, prune_keep  # noqa: E402
from mirror.fold import fold_layernorm_full  # noqa: E402
from mirror.heal import exact_involution  # noqa: E402
from mirror.m1 import (materialise_pos_embed, matched_random_structure, mlp_mirror_params,  # noqa: E402
                       random_head_structure, random_involution_fixing_ones)
from mirror.matching import random_pairing  # noqa: E402
from mirror.surgery import svd_truncate  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402


def _load_m1():
    """scripts/m1_attention.py (make_variant, svd_ranks, variant_params, load_operators, build_adapter).
    Importing it sets a 0.1 GPU memory cap; main() sets this script's cap again afterwards."""
    spec = importlib.util.spec_from_file_location("m1_attention_for_m2", REPO_ROOT / "scripts" / "m1_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m1 = _load_m1()

MIRROR = ("vlocal", "vlocal_r2", "vexact", "mlp_only", "random")
BASELINES = ("svd", "prune")
REFERENCE = ("original", "dense")
ALL_VARIANTS = REFERENCE + MIRROR + BASELINES
DEFAULT_VARIANTS = ("original", "dense", "vlocal", "vexact", "mlp_only", "svd", "prune")
LABELS = {
    "original": "original (hub model as released)", "dense": "dense (folded original, same implementation)",
    "vlocal": "Mirror V-local, MLP + attention", "vlocal_r2": "Mirror V-local, Wo structured + 2nd rotation",
    "vexact": "Mirror V-exact, MLP + attention", "mlp_only": "Mirror V-local, MLP only",
    "random": "matched random structure", "svd": "SVD at matched parameters",
    "prune": "MLP width pruning at matched parameters",
}
THRESHOLDS = {"float64": ("rel_raw", 1e-6), "float32": ("max_abs_unit", 1e-5), "bfloat16": ("max_abs_unit", 1e-3)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="torch.hub DINOv2 entry")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--structure", default="runs/m1/m1_structure.npz", help="M1 structure (scripts/m1_attention.py)")
    p.add_argument("--operators", default="runs/x2/perlayer_reg_s0/results/e1_operators.npz",
                   help="per-layer operators and neuron pairings (run_e1.py layout)")
    p.add_argument("--k", default=None, help="comma-separated numbers of converted blocks (default 6,9,12; tiny 1,2,4)")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help=f"any of {','.join(ALL_VARIANTS)}")
    p.add_argument("--batches", default=None, help="batch sizes (default 1,64,256; tiny 1,4)")
    p.add_argument("--modes", default=None, help="compile,eager (default both; tiny: eager)")
    p.add_argument("--compile-mode", default="default", help="torch.compile mode")
    p.add_argument("--warmup", type=int, default=None, help="warm-up runs (default 20; tiny 2)")
    p.add_argument("--iters", type=int, default=None, help="timed runs (default 100; tiny 5)")
    p.add_argument("--align", type=int, default=8, help="pad structured GEMM dimensions to multiples of this")
    p.add_argument("--check-images", type=int, default=8)
    p.add_argument("--cache-dir", default="cache", help="validation cache for the check images")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory-fraction", type=float, default=0.3)
    p.add_argument("--out", default=None, help="output directory (default runs/m2; tiny_out/m2 with --tiny)")
    p.add_argument("--force", action="store_true", help="redo rows already in m2_latency.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, planted structure, CPU")
    a = p.parse_args(argv)
    t = a.tiny
    a.ks = [int(k) for k in (a.k or ("1,2,4" if t else "6,9,12")).split(",") if k.strip()]
    a.batch_list = [int(b) for b in (a.batches or ("1,4" if t else "1,64,256")).split(",") if b.strip()]
    a.mode_list = [m.strip() for m in (a.modes or ("eager" if t else "compile,eager")).split(",") if m.strip()]
    a.warmup = (2 if t else 20) if a.warmup is None else a.warmup
    a.iters = (5 if t else 100) if a.iters is None else a.iters
    a.variant_list = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = set(a.variant_list) - set(ALL_VARIANTS)
    if bad:
        raise SystemExit(f"unknown variants {sorted(bad)}")
    if set(a.mode_list) - {"compile", "eager"}:
        raise SystemExit("--modes takes compile and/or eager")
    a.random_seed = 1   # m1_attention.build_adapter and make_variant read these names
    a.register_margin = 0.0
    return a


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvidia_smi() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------
def load_structure(a, adapter) -> tuple[dict, dict, dict]:
    """(S, ops, provenance) from the M1 structure file and the X2 operator file (as m1_attention uses them)."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    ops = m1.load_operators(_resolve(a.operators), L, d, h)
    path = _resolve(a.structure)
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    if str(f["model"].item()) != a.model:
        raise RuntimeError(f"{path} is for {f['model'].item()}, not {a.model}")
    if str(f["operators_sha256"].item()) != ops["sha256"]:
        raise RuntimeError(f"{path} was fitted with operators {f['operators'].item()} (sha256 "
                           f"{str(f['operators_sha256'].item())[:12]}), not {a.operators} ({ops['sha256'][:12]})")
    ks = [int(k) for k in f["ks"]]
    maps = [HeadMaps(sigma=np.asarray(f["sigma"][b], dtype=np.int64), B=f["B"][b], C=f["C"][b]) for b in range(L)]
    for hm in maps:
        hm.check(tol=1e-8)
    R = int(f["reg_perm"].shape[1])
    S = {"maps": maps, "a_out_offset": int(f["a_out_offset"]), "n_reg": R,
         "A_star": {k: exact_involution(f["A_star"][i]) for i, k in enumerate(ks)},
         "reg_perm": {k: np.asarray(f["reg_perm"][i], dtype=np.int64) for i, k in enumerate(ks)}}
    rs = int(f["random_seed"]) if "random_seed" in f else 1
    S["random"] = matched_random_structure(ops["A_layers"], ops["perms"], rs, fix_ones=True, heads=maps)
    prov = {"structure": str(path), "structure_sha256": m1.sha256(path), "operators": ops["path"],
            "operators_sha256": ops["sha256"], "structure_ks": ks, "a_out_offset": S["a_out_offset"],
            "random_seed": rs, "fit_seed": int(f["fit_seed"]) if "fit_seed" in f else None,
            "structure_git_hash": str(f["git_hash"].item()) if "git_hash" in f else None}
    return S, ops, prov


def tiny_structure(a, adapter) -> tuple[dict, dict, dict]:
    """Planted structure for the tiny CPU path: random involutions with A 1 = 1, random pairings, random
    head maps (some pairs, some self-heads), A_out offset 1 (as M1 chose for DINOv2-reg)."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    s = 100 * a.seed
    A_layers = [exact_involution(random_involution_fixing_ones(d, d // 2 - 3 + 3 * (l % 3), seed=s + l))
                for l in range(L + 1)]
    perms = [random_pairing(h, h // 3 + 5 * b, seed=s + 50 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1 + b) % 3, seed=s + 70 + b) for b in range(L)]
    A_star = {k: exact_involution(random_involution_fixing_ones(d, d // 2 + 5, seed=s + 90 + k)) for k in a.ks}
    S = {"maps": maps, "a_out_offset": 1, "n_reg": 0, "A_star": A_star,
         "reg_perm": {k: np.zeros(0, dtype=np.int64) for k in a.ks},
         "random": matched_random_structure(A_layers, perms, 1, fix_ones=True, heads=maps)}
    ops = {"A_layers": A_layers, "perms": perms, "C": None, "G": None, "path": "tiny planted structure", "sha256": ""}
    return S, ops, {"structure": "tiny planted structure", "a_out_offset": 1}


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
class AdapterFeature(nn.Module):
    """The k-NN feature of a hub/timm model (final-norm CLS), as mirror.knn computes it."""

    def __init__(self, adapter, model: nn.Module):
        super().__init__()
        self.model = model
        self._adapter = adapter

    def forward(self, x):
        return self._adapter.cls_feature(x, self.model)


def folded_copy(model: nn.Module, adapter) -> nn.Module:
    """Fully folded copy (position embedding materialised for DINOv2), or a plain copy if already folded."""
    m = copy.deepcopy(model)
    materialise_pos_embed(m, adapter)
    if not isinstance(m.blocks[0].norm1, nn.LayerNorm):
        return m
    fold_layernorm_full(m, final_norm_affine=False)
    return m


def mirror_plan(v: str, k: int, L: int, S: dict, ops: dict):
    """(bases, specs, wo_mode) of a mirror variant; head maps are the ones valid after the gauge."""
    blocks = range(k)
    off = S["a_out_offset"]
    if v == "vexact":
        As = S["A_star"][k]
        basis = Basis.of(As)
        specs = [BlockSpec(("mirror", As, As, S["maps"][b].gauged()), ("mirror", As, As, ops["perms"][b]))
                 if b < k else BlockSpec() for b in range(L)]
        return [basis] * (L + 1), specs, "fold"
    if v == "random":
        A, perms, heads = S["random"]["A_layers"], S["random"]["perms"], S["random"]["heads"]
    else:
        A, perms, heads = ops["A_layers"], ops["perms"], S["maps"]
    lay = [Basis.of(A[l]) for l in range(k + 1)]
    bases = lay + [lay[k]] * (L - k)
    specs = []
    for b in range(L):
        if b not in blocks:
            specs.append(BlockSpec())
            continue
        attn = ("dense",) if v == "mlp_only" else ("mirror", A[b], A[b + off], heads[b].gauged())
        specs.append(BlockSpec(attn, ("mirror", A[b], A[b + 1], perms[b])))
    return bases, specs, ("rotate" if v == "vlocal_r2" else "fold")


def prune_widths(k: int, adapter, S: dict, ops: dict) -> list:
    """Per block h' so that 2 h' d + dense attention = Mirror V-local matrix parameters (M1 formula)."""
    d, h = adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    A, perms, maps, off = ops["A_layers"], ops["perms"], S["maps"], S["a_out_offset"]
    out = []
    for b in range(k):
        kept = mlp_mirror_params(h, d, perms[b], A[b], A[b + 1]) + attention_free_params(d, maps[b], A[b], A[b + off])["matrices"]
        out.append(max(1, int((kept - 4 * H * dh * d) // (2 * d))))
    return out


@torch.no_grad()
def make_pair(v: str, k: int, ad, S: dict, ops: dict, align: int):
    """(reference dense model, efficient FastViT) computing the same function, in ad.model's dtype."""
    L = ad.depth
    if v == "original":
        return ad.model, None
    if v == "dense":
        return ad.model, build_fast_vit(folded_copy(ad.model, ad), [BlockSpec() for _ in range(L)], align=align)
    if v in MIRROR:
        ref = m1.make_variant("vlocal" if v == "vlocal_r2" else v, k, ad, S, ops)
        bases, specs, wo = mirror_plan(v, k, L, S, ops)
        return ref, build_fast_vit(folded_copy(ref, ad), specs, bases, align=align, wo_mode=wo)
    base = folded_copy(ad.model, ad)
    ref = copy.deepcopy(base)
    if v == "svd":
        mlp_r, att_r = m1.svd_ranks(ad, S, ops, list(range(k)))
        specs = [BlockSpec(("svd", att_r[b]), ("svd", mlp_r[b])) if b < k else BlockSpec() for b in range(L)]
        for b in range(k):
            blk = ref.blocks[b]
            W = blk.attn.qkv.weight
            D = W.shape[0] // 3
            for i, r in enumerate(att_r[b][:3]):
                W[i * D:(i + 1) * D] = svd_truncate(W[i * D:(i + 1) * D].double(), int(r)).to(W.dtype)
            for lin, r in ((blk.attn.proj, att_r[b][3]), (blk.mlp.fc1, mlp_r[b][0]), (blk.mlp.fc2, mlp_r[b][1])):
                lin.weight.copy_(svd_truncate(lin.weight.double(), int(r)).to(lin.weight.dtype))
    elif v == "prune":
        widths = prune_widths(k, ad, S, ops)
        specs = [BlockSpec(("dense",), ("prune", widths[b])) if b < k else BlockSpec() for b in range(L)]
        for b in range(k):
            mlp = ref.blocks[b].mlp
            drop = torch.ones(mlp.fc1.out_features, dtype=torch.bool, device=mlp.fc1.weight.device)
            drop[prune_keep(mlp.fc1.weight, mlp.fc2.weight, widths[b])] = False
            mlp.fc2.weight[:, drop] = 0.0
    else:
        raise ValueError(v)
    return ref, build_fast_vit(base, specs, align=align)


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def check_images(a, adapter, device) -> tuple[torch.Tensor, str]:
    n = int(a.check_images)
    if not a.tiny:
        try:
            from mirror.data import load_val_cache

            val, _, _, _ = load_val_cache(_resolve(a.cache_dir), crop=a.img_size)
            rng = np.random.default_rng(a.seed)
            idx = np.sort(rng.choice(len(val), size=n, replace=False))
            x = torch.from_numpy(np.ascontiguousarray(val[idx])).permute(0, 3, 1, 2).to(device)
            return normalize_for(adapter, x).float(), f"validation images {idx.tolist()} (seed {a.seed})"
        except (FileNotFoundError, RuntimeError, OSError) as e:
            log(f"validation cache not available ({e}); checks use random images")
    g = torch.Generator().manual_seed(a.seed)
    x = torch.randn(n, 3, adapter.img_size, adapter.img_size, generator=g)
    return x.to(device), f"random normal images (seed {a.seed})"


def _unit(f: torch.Tensor) -> torch.Tensor:
    return F.normalize(f.double(), dim=-1)


@torch.no_grad()
def compare(ad, ref: nn.Module, eff: nn.Module, x: torch.Tensor, ref_hi: nn.Module | None = None,
            x_hi: torch.Tensor | None = None) -> dict:
    a = ad.cls_feature(x, ref).double()
    b = eff(x).double()
    ma, mr = float((a - b).abs().max()), float(a.abs().max())
    out = {"max_abs_raw": ma, "max_abs_ref_raw": mr, "rel_raw": ma / mr if mr > 0 else float("nan"),
           "max_abs_unit": float((_unit(a) - _unit(b)).abs().max()), "noise_floor_unit": float("nan"),
           "ratio_to_noise": float("nan")}
    if ref_hi is not None:   # bf16 noise of the dense reference itself: bf16 against fp32
        hi = ad.cls_feature(x_hi, ref_hi).double()
        out["noise_floor_unit"] = float((_unit(a) - _unit(hi)).abs().max())
    out["ratio_to_noise"] = (out["max_abs_unit"] / out["noise_floor_unit"] if out["noise_floor_unit"] > 0
                             else float("nan"))
    return out


def _bf16(m: nn.Module) -> nn.Module:
    return copy.deepcopy(m).to(torch.bfloat16)


def run_checks(v: str, k: int, adapter, ad64, S, ops, a, x32, tags) -> tuple[list, tuple]:
    """Checks for one variant; returns (check rows, (ref32, eff32)) so the fp32 pair can be reused."""
    rows = []
    ref32, eff32 = make_pair(v, k, adapter, S, ops, a.align)
    if eff32 is None:
        return rows, (ref32, eff32)
    res = max([blk.attn.cost.get("residual", 0.0) for blk in eff32.blocks]
              + [blk.mlp.cost.get("residual", 0.0) for blk in eff32.blocks])
    ref64, eff64 = make_pair(v, k, ad64, S, ops, a.align)
    res64 = max([blk.attn.cost.get("residual", 0.0) for blk in eff64.blocks]
                + [blk.mlp.cost.get("residual", 0.0) for blk in eff64.blocks])
    results = {"float64": compare(ad64, ref64, eff64, x32.double()),
               "float32": compare(adapter, ref32, eff32, x32)}
    del ref64, eff64
    r16, e16 = _bf16(ref32), _bf16(eff32)
    results["bfloat16"] = compare(adapter, r16, e16, x32.to(torch.bfloat16), ref_hi=ref32, x_hi=x32)
    del r16, e16
    for dt, r in results.items():
        key, thr = THRESHOLDS[dt]
        rows.append({"variant": v, "k": k, "dtype": dt, **r, "criterion": f"{key} < {thr:g}", "threshold": thr,
                     "pass": bool(r[key] < thr), "structure_residual_fp32": res, "structure_residual_fp64": res64,
                     "n_images": int(x32.shape[0]),
                     **tags})
        log(f"check {v} k = {k} {dt}: unit max abs {r['max_abs_unit']:.2e}, raw max abs {r['max_abs_raw']:.2e} "
            f"(rel {r['rel_raw']:.2e}) -> {'pass' if rows[-1]['pass'] else 'FAIL'} ({rows[-1]['criterion']})")
    return rows, (ref32, eff32)


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------
def measure(fn, x: torch.Tensor, warmup: int, iters: int, device) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        _sync(device)
        ts = []
        for _ in range(iters):
            if device.type == "cuda":
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn(x)
                e.record()
                e.synchronize()
                ts.append(float(s.elapsed_time(e)))
            else:
                t0 = time.perf_counter()
                fn(x)
                ts.append(1e3 * (time.perf_counter() - t0))
    t = np.asarray(ts)
    return {"latency_ms_median": float(np.median(t)), "latency_ms_p10": float(np.percentile(t, 10)),
            "latency_ms_p90": float(np.percentile(t, 90)), "latency_ms_mean": float(t.mean()),
            "latency_ms_min": float(t.min())}


def _dynamo_counts() -> dict:
    try:
        from torch._dynamo.utils import counters

        return {"graphs": int(counters["stats"].get("unique_graphs", 0)),
                "graph_breaks": int(sum(counters["graph_break"].values()))}
    except Exception:  # noqa: BLE001
        return {"graphs": None, "graph_breaks": None}


def time_model(model: nn.Module, batch: int, mode: str, a, device) -> dict:
    """Median latency of one bf16 forward at `batch` in `mode` (compile falls back to eager on failure)."""
    g = torch.Generator(device=device).manual_seed(a.seed + batch)
    x = torch.randn(batch, 3, a.img_px, a.img_px, generator=g, device=device).to(torch.bfloat16)
    info = {"mode_used": "eager", "compile_ok": None, "compile_s": None, "compile_error": "", "graphs": None,
            "graph_breaks": None}
    fn = model
    if mode == "compile":
        torch._dynamo.reset()
        try:
            cm = torch.compile(model, mode=a.compile_mode, dynamic=False)
            t0 = time.time()
            with torch.inference_mode():
                cm(x)
            _sync(device)
            info.update({"mode_used": "compile", "compile_ok": True, "compile_s": round(time.time() - t0, 1),
                         **_dynamo_counts()})
            fn = cm
        except Exception as e:  # noqa: BLE001
            info.update({"mode_used": "eager (compile failed)", "compile_ok": False,
                         "compile_error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"})
            log(f"torch.compile failed ({info['compile_error']}); timing in eager mode instead")
            torch._dynamo.reset()
    try:
        r = measure(fn, x, a.warmup, a.iters, device)
    except torch.cuda.OutOfMemoryError as e:
        r = {"latency_ms_median": float("nan"), "error": f"OOM: {str(e).splitlines()[0][:200]}"}
    if mode == "compile":
        torch._dynamo.reset()
    r["throughput_img_s"] = batch * 1e3 / r["latency_ms_median"] if r["latency_ms_median"] > 0 else float("nan")
    return {**info, **r}


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------
def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _e(x):
    try:
        x = float(x)
        return f"{x:.1e}" if math.isfinite(x) else "n/a"
    except (TypeError, ValueError):
        return str(x)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Speed-ups against dense (same implementation) and the hub original, same mode and batch."""
    df = df.copy()
    for col in ("speedup_vs_dense", "speedup_vs_original", "flop_speedup_vs_dense"):
        df[col] = float("nan")
    for (mode, b), g in df.groupby(["mode", "batch"]):
        for ref, col in (("dense", "speedup_vs_dense"), ("original", "speedup_vs_original")):
            r = g[g["variant"] == ref]
            if len(r):
                df.loc[g.index, col] = float(r["latency_ms_median"].iloc[0]) / g["latency_ms_median"]
    dense = df[df["variant"] == "dense"]
    if len(dense):
        df["flop_speedup_vs_dense"] = float(dense["macs_per_image"].iloc[0]) / df["macs_per_image"]
    return df


def ones_deviation(S: dict, ops: dict, d: int) -> dict:
    """Largest ||A 1 - 1|| / sqrt(d) of the operators used (0 for operators that fix 1 exactly)."""
    one = np.ones(d)

    def dev(As):
        return float(max((np.linalg.norm(np.asarray(A, dtype=np.float64) @ one - one) / np.sqrt(d) for A in As),
                         default=float("nan")))

    return {"per_layer_max": dev(ops["A_layers"]), "span_max": dev(list(S["A_star"].values()))}


def check_sentence(chk: pd.DataFrame) -> str:
    """Plain statement of the check outcome, from the numbers."""
    if not len(chk):
        return "No checks were run."
    parts = []
    for dt, name in (("float64", "float64"), ("float32", "fp32"), ("bfloat16", "bf16")):
        sub = chk[chk["dtype"] == dt]
        if len(sub):
            n_ok = int(sub["pass"].astype(bool).sum())
            parts.append(f"{name}: {n_ok} of {len(sub)} pass")
    text = "Outcome: " + "; ".join(parts) + "."
    b = chk[chk["dtype"] == "bfloat16"]
    if len(b) and b["ratio_to_noise"].notna().any():
        text += (f" In bf16 the efficient-versus-dense difference is {_f(b['ratio_to_noise'].min(), 2)} to "
                 f"{_f(b['ratio_to_noise'].max(), 2)} times the dense model's own bf16 rounding (noise floor "
                 f"{_e(b['noise_floor_unit'].min())} to {_e(b['noise_floor_unit'].max())}).")
        if (~b["pass"].astype(bool)).any() and float(b["noise_floor_unit"].min()) > THRESHOLDS["bfloat16"][1]:
            text += (" The bf16 bar of 1e-3 is below that rounding level, so a bf16 failure here reflects bf16 "
                     "arithmetic, not the implementation; the float64 and fp32 rows are the equivalence checks.")
    return text


def _g(x):
    """GFLOPs with enough digits for tiny models."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return _f(x, 2) if x >= 1 else f"{x:.4f}"


def write_summary(out: Path, lat: pd.DataFrame, chk: pd.DataFrame, meta: dict) -> Path:
    g = meta["git"]
    order = {v: i for i, v in enumerate(ALL_VARIANTS)}
    lat = lat.sort_values(["k", "variant", "mode", "batch"], key=lambda s: s.map(order) if s.name == "variant" else s)
    one = lat.drop_duplicates(["variant", "k"])
    batches = sorted(int(b) for b in lat["batch"].unique())
    lines = [
        "# M2: efficient mirror-converted blocks and latency" + (" (TINY TEST, random weights, CPU)" if meta["tiny"] else ""),
        "",
        f"Model `{meta['model']}` ({meta['depth']} blocks, d {meta['embed_dim']}, h {meta['mlp_hidden']}, "
        f"{meta['heads']} heads, {meta['n_tokens']} tokens per image at {meta['img_size']} px). Structure: "
        f"`{meta['provenance'].get('structure')}`, operators `{meta['provenance'].get('operators', 'n/a')}`. "
        f"Git `{str(g['git_hash'])[:10]}`{' (uncommitted changes)' if g.get('git_dirty') else ''}, seed {meta['seed']}. "
        f"Device: {meta['env'].get('gpu', meta['env'].get('device'))}, torch {meta['env']['torch']}. Written {meta['date']}.",
        "",
        "**Measured speed-ups are expected to be smaller than the FLOP savings.** The converted blocks replace a few "
        "large matrix multiplications by more, smaller ones (even and odd parts), plus butterfly additions, gathers "
        "and concatenations that move memory but do no multiply-adds; token mixing (attention over tokens) is "
        "unchanged; at batch 1 the GPU is limited by kernel launches rather than arithmetic; and V-local pays a "
        "d x d basis rotation per converted block. The FLOP column below is analytic; the latency columns are measured.",
        "",
        "## Equivalence checks",
        "",
        f"Each efficient model against the dense model it was built from, on {meta['check_images']}. float64: both "
        "built in float64 (criterion: max relative error of the CLS feature under 1e-6). Dropped: relative size "
        "(Frobenius) of the non-equivariant part of the folded weights that the even/odd form leaves out; it is "
        "zero in exact arithmetic. LayerNorm folding centres every write, which keeps the weights equivariant only "
        "if each operator fixes the all-ones direction exactly; largest |A 1 - 1| / sqrt(d) of the operators used: "
        f"{_e(meta.get('ones_deviation', {}).get('per_layer_max'))} (per-layer), "
        f"{_e(meta.get('ones_deviation', {}).get('span_max'))} (span operators)"
        + (" (fitted operators stored in float32; this is what the dropped part measures)."
           if max(float(meta.get('ones_deviation', {}).get(key, 0) or 0) for key in ("per_layer_max", "span_max")) > 1e-12
           else " (exact here).")
        + " fp32 (TF32 "
        "off) and bf16: max abs difference of the L2-normalised CLS feature (the k-NN feature), under 1e-5 and 1e-3. "
        "Noise floor: the dense model in bf16 against itself in fp32 (same feature). Raw: the CLS feature before "
        "normalisation (absolute; largest entry in brackets).",
        "",
        "| variant | k | float64 rel. | dropped (float64) | fp32 unit | fp32 raw | bf16 unit | bf16 noise floor | bf16 / noise | pass (64 / 32 / bf16) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (v, k), gr in chk.groupby(["variant", "k"], sort=False):
        r = {dt: gr[gr["dtype"] == dt].iloc[0] for dt in ("float64", "float32", "bfloat16") if (gr["dtype"] == dt).any()}
        if len(r) < 3:
            continue
        ok = " / ".join("yes" if bool(r[dt]["pass"]) else "NO" for dt in ("float64", "float32", "bfloat16"))
        lines.append(f"| {LABELS.get(v, v)} | {int(k)} | {_e(r['float64']['rel_raw'])} | "
                     f"{_e(r['float64'].get('structure_residual_fp64'))} | {_e(r['float32']['max_abs_unit'])} | "
                     f"{_e(r['float32']['max_abs_raw'])} ({_f(r['float32']['max_abs_ref_raw'], 1)}) | "
                     f"{_e(r['bfloat16']['max_abs_unit'])} | {_e(r['bfloat16']['noise_floor_unit'])} | "
                     f"{_f(r['bfloat16']['ratio_to_noise'], 2)} | {ok} |")
    lines += ["", check_sentence(chk)]
    lines += [
        "",
        "## Parameters and analytic cost",
        "",
        "Params: whole model, M1 convention (original count minus the dense matrices of the converted blocks plus "
        "the matrices the efficient model stores; biases and embeddings as in the original). Kept: stored matrix "
        "parameters of the converted blocks over dense. M1 formula: the count m1_knn.csv uses for this variant. "
        "Rotations: stored rotation matrices (d x d each, not in Params). GFLOPs per image: analytic, rotations "
        "included; executed adds the alignment padding.",
        "",
        "| k | variant | params | kept | M1 formula | d x d rotations per token | GFLOPs | saving vs dense | GFLOPs executed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    dense_macs = float(one[one["variant"] == "dense"]["macs_per_image"].iloc[0]) if (one["variant"] == "dense").any() else float("nan")
    for _, r in one.iterrows():
        kept = r["kept_frac_converted"]
        m1p = r.get("params_m1_formula")
        lines.append(f"| {int(r['k'])} | {LABELS.get(r['variant'], r['variant'])} | {int(r['params_model']):,} | "
                     f"{_f(kept, 3)} | {('{:,}'.format(int(m1p))) if pd.notna(m1p) else 'n/a'} | "
                     f"{int(r['n_rotations'] + r['n_attn_rotations'])} | {_g(r['gflops'])} | "
                     f"{_f(100 * (1 - r['macs_per_image'] / dense_macs), 1)}% | {_g(r['gflops_exec'])} |")
    for mode in [m for m in ("compile", "eager") if (lat["mode"] == m).any()]:
        sub = lat[lat["mode"] == mode]
        lines += [
            "",
            f"## Latency, {mode} (bf16, median of {meta['iters']} runs after {meta['warmup']} warm-up runs)",
            "",
            "Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); "
            "FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.",
            "",
            "| k | variant | " + " | ".join(f"ms (batch {b})" for b in batches) + " | "
            + " | ".join(f"speed-up (b{b})" for b in batches) + f" | img/s (b{batches[-1]}) | FLOP speed-up | mode used |",
            "|---|---|" + "---|" * (2 * len(batches) + 3),
        ]
        for (v, k), gr in sub.groupby(["variant", "k"], sort=False):
            gb = {int(r["batch"]): r for _, r in gr.iterrows()}
            ms = " | ".join(_f(gb[b]["latency_ms_median"], 2) if b in gb else "n/a" for b in batches)
            sp = " | ".join(_f(gb[b]["speedup_vs_dense"], 2) if b in gb else "n/a" for b in batches)
            last = gb.get(batches[-1])
            used = ", ".join(sorted(set(str(x) for x in gr["mode_used"])))
            lines.append(f"| {int(k)} | {LABELS.get(v, v)} | {ms} | {sp} | "
                         f"{_f(last['throughput_img_s'], 0) if last is not None else 'n/a'} | "
                         f"{_f(gr['flop_speedup_vs_dense'].iloc[0], 2)} | {used} |")
    failed = lat[lat["mode_used"].astype(str).str.contains("failed")]
    off = meta["provenance"].get("a_out_offset")
    lines += [
        "",
        "## What is folded and what is not",
        "",
        "- All norms are RMSNorm on the mean-zero stream (full LayerNorm folding, X4); LayerScale and the norm affines "
        "are folded into the weights. The same holds for the dense, SVD and pruning baselines, so the comparison "
        "with `dense` isolates the structure; `original` also differs by the hub code (LayerNorm, LayerScale, position "
        "interpolation at every forward, its own attention kernel).",
        "- V-exact: the rotation into the A* basis is folded into the patch embedding, CLS token, registers and "
        "position embedding; blocks after the span are rewritten in that basis (free); the rotation out is applied "
        "to the CLS feature only (d x d per image; with the even head of full conversion only the d+ even rows). "
        "No per-token rotation.",
        "- V-local and MLP only: the rotation into the first basis is folded into the embeddings and the blocks after "
        "the span are rewritten in the last basis (free); one d x d rotation of the stream per converted block is "
        "computed explicitly (counted in FLOPs and latency), plus the CLS rotation out.",
        f"- V-local attention output: M1 projected Wo with A_(l+{off})." + (
            " That operator is not the block's input operator A_l, so Wo cannot stay structured in the block's basis "
            "without a second rotation. `vlocal` stores Wo dense in that basis (the rotation folded into it: d x H dh "
            "multiply-adds, more than Wo's free parameters, which is why its kept fraction exceeds M1's formula); "
            "`vlocal_r2` (if run) keeps Wo structured and rotates the attention output (a second d^2 per token)."
            if off else " Wo is structured in the block's basis."),
        "- Self-heads: their head-internal bases (eigenvectors of B and C) are folded into the q, k, v and Wo weights. "
        "Twin heads are computed once as u + v and u - v.",
        f"- Alignment: structured, low-rank and pruned GEMM dimensions are zero-padded to multiples of {meta['align']}; "
        "padding changes no output and is excluded from Params and GFLOPs (included in GFLOPs executed).",
        "",
        "## Caveats",
        "",
        (f"- Measured on one device ({meta['env'].get('gpu', meta['env'].get('device'))}) in one session; latencies "
         "are medians of single forwards (CUDA events on a GPU), no CUDA graphs. Device state before and after "
         "timing (nvidia-smi): " + "; ".join(meta.get("nvidia_smi", {}).values()) + ". The GPU is shared: other "
         "processes running at the same time would inflate latencies."),
        "- torch.compile default mode (no autotuning); the efficient models use plain PyTorch ops (no custom kernels). "
        "The butterfly, gathers and concatenations are fused by the compiler but still cost memory traffic.",
        "- SVD and pruning weights are for timing only (their accuracy comes from M1 and M3); pruning keeps neurons by "
        "weight norms, not activations. Both are matched to Mirror V-local's matrix parameters (M1 formula).",
        "- One structure (fit seed of the M1 run); latency does not depend on the fitted values, only on the shapes.",
    ]
    if len(failed):
        lines.append(f"- torch.compile failed for {len(failed)} rows; those rows were timed in eager mode "
                     f"(see compile_error in m2_latency.csv).")
    lines += ["", "Files: m2_latency.csv, m2_checks.csv, m2_meta.json (config, seed, git hash, environment)."]
    text = "\n".join(lines) + "\n"
    assert chr(0x2014) not in text
    path = out / "M2_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/m2" if tiny else "runs/m2")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(a.memory_fraction))   # importing m1_attention set 0.1
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 checks need full fp32 matmuls
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    tags = {"git_hash": git["git_hash"], "seed": a.seed}
    log(f"M2 {'TINY ' if tiny else ''}run on {device}, out {out}, commit {str(git['git_hash'])[:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")
    smi = {"before": nvidia_smi() if device.type == "cuda" else "cpu"}

    with timer.stage("load"):
        adapter = m1.build_adapter(a, device)
        a.img_px = int(adapter.img_size)
        L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
        H, dh = attn_dims(adapter.model.blocks[0])
        T, n_patches = adapter.n_tokens, adapter.grid_hw[0] * adapter.grid_hw[1]
        S, ops, prov = tiny_structure(a, adapter) if tiny else load_structure(a, adapter)
        ks = [k for k in a.ks if 1 <= k <= L]
        ad64 = copy.copy(adapter)
        ad64.model = copy.deepcopy(adapter.model).double()
        x32, img_src = check_images(a, adapter, device)
        n_original = int(sum(p.numel() for p in adapter.model.parameters()))
    log(f"depth {L}, d {d}, h {h}, heads {H} x {dh}, tokens {T}; ks {ks}; variants {a.variant_list}; "
        f"batches {a.batch_list}; modes {a.mode_list}; A_out offset {S['a_out_offset']}")

    lat_path, chk_path = out / "m2_latency.csv", out / "m2_checks.csv"
    lat_rows = pd.read_csv(lat_path).to_dict("records") if (lat_path.exists() and not a.force) else []
    chk_rows = pd.read_csv(chk_path).to_dict("records") if (chk_path.exists() and not a.force) else []
    done = {(r["variant"], int(r["k"]), int(r["batch"]), r["mode"]) for r in lat_rows}
    checked = {(r["variant"], int(r["k"])) for r in chk_rows}
    skipped = []

    plan = [(v, 0) for v in REFERENCE if v in a.variant_list]
    plan += [(v, k) for k in ks for v in MIRROR + BASELINES if v in a.variant_list]
    dense_cost = None
    with timer.stage("build, check and time"):
        for v, k in plan:
            if v == "vexact" and k not in S["A_star"]:
                skipped.append(f"vexact k = {k}: no span operator for this k in the structure file")
                log(skipped[-1])
                continue
            todo = [(b, md) for md in a.mode_list for b in a.batch_list if (v, k, b, md) not in done]
            need_check = v != "original" and (v, k) not in checked
            if not todo and not need_check:
                continue
            log(f"== {v} k = {k}")
            if need_check:
                rows, (ref32, eff32) = run_checks(v, k, adapter, ad64, S, ops, a, x32, tags)
                chk_rows += rows
                pd.DataFrame(chk_rows).to_csv(chk_path, index=False)
            else:
                ref32, eff32 = make_pair(v, k, adapter, S, ops, a.align)
            if v == "original":
                timed = AdapterFeature(adapter, copy.deepcopy(adapter.model).to(torch.bfloat16)).eval()
                if dense_cost is None:
                    _, dfv = make_pair("dense", 0, adapter, S, ops, a.align)
                    dense_cost = fast_vit_cost(dfv, T, n_patches)
                    del dfv
                cost, fvinfo = dense_cost, {"n_rotations": 0, "n_attn_rotations": 0}
                stored = n_original
            else:
                cost = fast_vit_cost(eff32, T, n_patches)
                if v == "dense":
                    dense_cost = cost
                fvinfo = eff32.info
                stored = int(sum(p.numel() for p in eff32.parameters()))
                timed = _bf16(eff32)
            m1p = None
            if v in MIRROR or v == "svd":
                m1p = m1.variant_params("vlocal" if v == "vlocal_r2" else v, k, adapter, S, ops, n_original)["model_params"]
            elif v in REFERENCE:
                m1p = n_original
            dense_conv = cost["matrix_params_converted_dense"]
            static = {
                "model": a.model if not tiny else "tiny", "variant": v, "label": LABELS[v], "k": k,
                "params_model": n_original - dense_conv + cost["matrix_params_converted"],
                "params_m1_formula": m1p, "params_stored": stored,
                "matrix_params_converted": cost["matrix_params_converted"], "matrix_params_converted_dense": dense_conv,
                "kept_frac_converted": cost["matrix_params_converted"] / dense_conv if dense_conv else 1.0,
                "rotation_params": cost["rotation_params"], "padding_params": cost["padding_params"],
                "n_rotations": int(fvinfo.get("n_rotations", 0)), "n_attn_rotations": int(fvinfo.get("n_attn_rotations", 0)),
                "macs_per_image": cost["macs"], "macs_exec_per_image": cost["macs_exec"], "gflops": cost["gflops"],
                "gflops_exec": cost["gflops_exec"], "rotation_macs": cost["macs_rot"] + cost["macs_head"],
                "mixing_macs": cost["macs_mix"], "align": a.align,
                "wo_mode": fvinfo.get("wo_mode", "") if v in ("vlocal", "vlocal_r2", "random") else "",
                "dtype": "bfloat16", "n_tokens": T,
            }
            del ref32, eff32
            _free(device)
            for b, md in todo:
                r = time_model(timed, b, md, a, device)
                lat_rows.append({**static, "batch": b, "mode": md, **r, "torch": torch.__version__,
                                 "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                 **tags})
                done.add((v, k, b, md))
                pd.DataFrame(lat_rows).to_csv(lat_path, index=False)
                log(f"{v} k = {k} batch {b} {md} ({r['mode_used']}): median {r['latency_ms_median']:.3f} ms, "
                    f"{r['throughput_img_s']:.0f} img/s")
            del timed
            _free(device)

    smi["after"] = nvidia_smi() if device.type == "cuda" else "cpu"
    lat = add_derived(pd.DataFrame(lat_rows))
    lat.to_csv(lat_path, index=False)
    chk = pd.DataFrame(chk_rows)
    meta = {
        "experiment": "M2 efficient implementation and latency", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": a.model if not tiny else "tiny random timm ViT", "tiny": tiny, "seed": a.seed, "git": git,
        "env": env_info(device), "args": {k: v for k, v in vars(a).items()}, "provenance": prov, "ks": ks,
        "variants": a.variant_list, "batches": a.batch_list, "modes": a.mode_list, "warmup": a.warmup,
        "iters": a.iters, "align": a.align, "compile_mode": a.compile_mode, "check_images": img_src,
        "thresholds": {k: f"{c} < {t:g}" for k, (c, t) in THRESHOLDS.items()},
        "depth": L, "embed_dim": d, "mlp_hidden": h, "heads": H, "head_dim": dh, "n_tokens": T,
        "n_patches": n_patches, "img_size": adapter.img_size, "params_original": n_original,
        "nvidia_smi": smi, "skipped": skipped, "ones_deviation": ones_deviation(S, ops, d), "checks_pass": bool(chk["pass"].astype(bool).all()) if len(chk) else None,
        "tf32": False, "timing_dtype": "bfloat16 weights and inputs",
        "timing": {"stages": timer.stages, "total_s": timer.total()},
    }
    with open(out / "m2_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    summary = write_summary(out, lat, chk, meta)
    log(f"wrote {summary} in {timer.total() / 60:.1f} min")
    return {"out": str(out), "meta": meta, "latency": lat, "checks": chk}


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""M2 profile: where the time goes when the mirror-converted (even/odd) models run slower than dense.

    python scripts/m2_profile.py --model dinov2_vitb14_reg --variants dense,vexact,vlocal,mlp_only,svd,prune \
        --k 6,12 --batch 256 --modes compile,eager --out runs/m2_profile
    python scripts/m2_profile.py --tiny --out tiny_out/m2_profile      # tiny random ViT, CPU, smoke test

Measurement only: nothing is optimised. Every model is built exactly as scripts/m2_latency.py builds it
(its make_pair, structure and operator loaders, seed, alignment and bf16 cast; same structure and
operator files), and `dense` (the folded original in the same implementation) is always built as the
reference. For each variant, k and mode (torch.compile default mode with static shapes, or eager):
  1. latency: CUDA events around one forward, median of --iters runs after --warmup runs (as M2), with
     no profiler attached; the input is M2's (randn images, generator seeded with seed + batch);
  2. kernel profile: --prof-iters further forwards under torch.profiler (CPU and CUDA activity). Every
     GPU activity (kernels, memcpy, memset; record_function annotations excluded) is summed per name and
     sorted into categories by name (classify_kernel): GEMM, attention, fused Triton (compile), elementwise
     and activation, normalisation, memory movement (copy, cat, gather, index, permute, pad, fill), other.
     Per forward: ms, share of kernel time and number of kernel launches. Idle time = latency minus kernel
     time (the GPU waiting for the CPU to launch work), so latency gap = sum of category changes + idle
     change, exactly;
  3. sub-modules (converted blocks only): each converted block's attention and MLP, and the V-local stream
     rotation (F.linear(x, rot_t), as FastBlock applies it), timed alone with CUDA events on a bf16 input of
     batch x tokens x d, against the same block of the dense model; so the gap can be split into MLP,
     attention and rotations. Parts left dense by a variant are not timed.
No equivalence checks here (M2 did them; runs/m2_clean/M2_SUMMARY.md).

Outputs (rows carry the git hash and seed): m2_profile.csv (variant, k, mode, category, ms per forward,
share, kernel count, plus run-level columns), m2_submodules.csv, m2_triton.csv (fused Triton time by the
operations named in the kernel), m2_top_kernels.csv and top_kernels/<variant>_k<k>_<mode>.csv (the --top
kernels by GPU time), M2_PROFILE.md (plain-language explanation from the measured numbers),
m2_profile_meta.json (config, seed, git hash, environment, nvidia-smi before and after).

On CPU (--tiny) there are no GPU kernels: categories are filled from CPU operator self times instead
(`time_source` says which). That path is a smoke test of the pipeline, not a measurement.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.attention import attn_dims  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Kernel categories
# ----------------------------------------------------------------------------------------------
CATEGORIES = ("gemm", "attention", "triton", "elementwise", "norm", "memory", "other")
CATEGORY_LABELS = {
    "gemm": "GEMM (cuBLAS, CUTLASS, convolution)",
    "attention": "attention (flash or memory-efficient SDPA)",
    "triton": "fused Triton kernels (compile)",
    "elementwise": "elementwise and activation",
    "norm": "normalisation",
    "memory": "memory movement (copy, cat, gather, index, permute, pad, fill)",
    "other": "other",
}
SHORT = {"gemm": "GEMM", "attention": "attention", "triton": "fused Triton", "elementwise": "elementwise",
         "norm": "normalisation", "memory": "memory movement", "other": "other"}

_ATTN = ("flash", "fmha", "attention", "sdpa", "mem_eff", "softmax")
_GEMM = ("gemm", "cutlass", "cublas", "xmma", "gemv", "splitkreduce", "s16816", "s1688", "h16816", "h1688",
         "wmma", "dot_kernel", "convolution", "convolve", "conv2d", "fprop", "matmul")
# In these models pow, mean and rsqrt occur only inside RMSNorm, so their kernels count as normalisation.
_NORM = ("layer_norm", "layernorm", "rms_norm", "rmsnorm", "group_norm", "batch_norm", "batchnorm", "meanops",
         "welford", "rowwisemoments", "normforward", "rsqrt", "pow_tensor_scalar")
# (not "roll" or "flip": "unrolled_elementwise_kernel" would match)
_MEMORY = ("copy", "index", "gather", "scatter", "permute", "transpose", "pad", "fill", "nchwtonhwc",
           "nhwctonchw", "contiguous", "clone")
_ELEMENTWISE = ("elementwise", "gelu", "silu", "relu", "tanh", "erf", "sigmoid", "functor")
_CAT = re.compile(r"(?<![a-z])cat(?![a-z])")
_TRITON_GEMM = re.compile(r"(^|_)(mm|addmm|bmm|baddbmm|convolution)(_|$)")
_ATEN = {
    "gemm": re.compile(r"^(mm|addmm|bmm|baddbmm|addbmm|matmul|linear|_addmm_activation|mkldnn_linear|.*conv.*)$"),
    "attention": re.compile(r".*(attention|sdpa|softmax).*"),
    "norm": re.compile(r"^(rms_norm|_fused_rms_norm|layer_norm|native_layer_norm|group_norm|batch_norm|mean|rsqrt|"
                       r"pow|var|var_mean|norm)$"),
    "memory": re.compile(r"^(cat|stack|index|index_select|gather|scatter|copy_|_to_copy|to|clone|contiguous|"
                         r"constant_pad_nd|pad|fill_|zero_|zeros|new_zeros|flip|roll|im2col|col2im|repeat)$"),
    "elementwise": re.compile(r"^(add|add_|sub|sub_|rsub|mul|mul_|div|div_|gelu|silu|relu|tanh|erf|neg|exp|where|"
                              r"clamp|sigmoid|addcmul|lerp)$"),
}


def classify_kernel(name: str) -> str:
    """Category of a GPU kernel (or, for the CPU path, an `aten::` operator) from its name."""
    raw = str(name)
    if raw.startswith("aten::"):
        op = raw[len("aten::"):]
        for cat in ("gemm", "attention", "norm", "memory", "elementwise"):
            if _ATEN[cat].match(op):
                return cat
        return "other"
    s = raw.lower().strip()
    if s.startswith("memcpy") or s.startswith("memset"):
        return "memory"
    if any(t in s for t in _ATTN):
        return "attention"
    if s.startswith("triton"):
        return "gemm" if s.startswith("triton_tem") and _TRITON_GEMM.search(s) else "triton"
    if any(t in s for t in _GEMM):
        return "gemm"
    if any(t in s for t in _NORM):
        return "norm"
    if any(t in s for t in _MEMORY) or _CAT.search(s):
        return "memory"
    if "reduce_kernel" in s:          # reductions other than the RMSNorm mean (MeanOps, above)
        return "other"
    if any(t in s for t in _ELEMENTWISE):
        return "elementwise"
    return "other"


_TAG_GROUPS = (
    ("norm", {"mean", "rsqrt", "pow", "var", "rms", "norm"}),
    ("cat/gather/pad", {"cat", "index", "gather", "scatter", "pad", "constant", "clone", "permute", "transpose",
                        "slice", "select", "stack"}),
    ("activation", {"gelu", "erf", "tanh", "silu", "relu", "sigmoid"}),
)


def triton_tag(name: str) -> str:
    """What a fused Triton kernel does, from the operations Inductor lists in its name (e.g.
    triton_poi_fused_add_cat_gelu_sub_3 -> 'cat/gather/pad + activation'); 'pointwise' if none of the groups."""
    s = str(name).lower()
    if "fused" not in s:
        return "unnamed"
    ops = set(t for t in re.split(r"_+", s.split("fused", 1)[1]) if t)
    tags = [g for g, keys in _TAG_GROUPS if ops & keys]
    return " + ".join(tags) if tags else "pointwise"


def short_name(name: str, n: int = 160) -> str:
    s = str(name)
    if s.startswith("void "):
        s = s[5:]
    s = s.split("(", 1)[0]
    return s if len(s) <= n else s[:n] + "..."


def _is_annotation(name: str) -> bool:
    s = str(name)
    return s.startswith("##") or s.startswith("ProfilerStep") or "Compiled" in s or s.startswith("m2_profile")


# ----------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------
_M2 = None


def load_m2():
    """scripts/m2_latency.py (builders, structure loaders, measure, labels). Importing it sets GPU memory
    caps of 0.3 and 0.1 (m1_attention); main() sets this script's cap again afterwards."""
    global _M2
    if _M2 is None:
        spec = importlib.util.spec_from_file_location("m2_latency_for_profile", REPO_ROOT / "scripts" / "m2_latency.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M2 = mod
    return _M2


DEFAULT_VARIANTS = "dense,vexact,vlocal,mlp_only,svd,prune"
ORDER = ("original", "dense", "vexact", "vlocal", "vlocal_r2", "mlp_only", "random", "svd", "prune")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="torch.hub DINOv2 entry (as m2_latency.py)")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--structure", default="runs/m1/m1_structure.npz", help="M1 structure (as m2_latency.py)")
    p.add_argument("--operators", default="runs/x2/perlayer_reg_s0/results/e1_operators.npz",
                   help="per-layer operators and neuron pairings (as m2_latency.py)")
    p.add_argument("--variants", default=DEFAULT_VARIANTS, help="m2_latency.py variant names; dense is always added")
    p.add_argument("--k", default=None, help="numbers of converted blocks (default 6,12; tiny 2)")
    p.add_argument("--batch", type=int, default=None, help="batch size (default 256; tiny 2)")
    p.add_argument("--modes", default=None, help="compile,eager (default both; tiny: eager)")
    p.add_argument("--compile-mode", default="default", help="torch.compile mode (M2 used default)")
    p.add_argument("--warmup", type=int, default=None, help="warm-up forwards before timing (default 20; tiny 1)")
    p.add_argument("--iters", type=int, default=None, help="timed forwards, CUDA events (default 100 as M2; tiny 2)")
    p.add_argument("--prof-iters", type=int, default=None, help="profiled forwards (default 10; tiny 2)")
    p.add_argument("--sub-warmup", type=int, default=None, help="sub-module warm-up runs (default 10; tiny 1)")
    p.add_argument("--sub-iters", type=int, default=None, help="sub-module timed runs (default 50; tiny 2)")
    p.add_argument("--sub-blocks", default="all", help="converted blocks timed alone: all, first, or a list (0,3)")
    p.add_argument("--no-submodules", action="store_true", help="skip the sub-module timings")
    p.add_argument("--top", type=int, default=30, help="kernels kept per variant, k and mode")
    p.add_argument("--align", type=int, default=8, help="as m2_latency.py")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory-fraction", type=float, default=0.25)
    p.add_argument("--reference", default="runs/m2_clean/m2_latency.csv",
                   help="M2 latency file to show next to this run's latencies (skipped if missing)")
    p.add_argument("--out", default=None, help="output directory (default runs/m2_profile; tiny_out/m2_profile)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, planted structure, CPU")
    a = p.parse_args(argv)
    t = a.tiny
    a.ks = [int(k) for k in (a.k or ("2" if t else "6,12")).split(",") if k.strip()]
    a.batch = (2 if t else 256) if a.batch is None else int(a.batch)
    a.mode_list = [m.strip() for m in (a.modes or ("eager" if t else "compile,eager")).split(",") if m.strip()]
    a.warmup = (1 if t else 20) if a.warmup is None else a.warmup
    a.iters = (2 if t else 100) if a.iters is None else a.iters
    a.prof_iters = (2 if t else 10) if a.prof_iters is None else a.prof_iters
    a.sub_warmup = (1 if t else 10) if a.sub_warmup is None else a.sub_warmup
    a.sub_iters = (2 if t else 50) if a.sub_iters is None else a.sub_iters
    a.variant_list = [v.strip() for v in a.variants.split(",") if v.strip()]
    if set(a.mode_list) - {"compile", "eager"}:
        raise SystemExit("--modes takes compile and/or eager")
    if min(a.iters, a.prof_iters, a.sub_iters) < 1:
        raise SystemExit("--iters, --prof-iters and --sub-iters must be at least 1")
    return a


def m2_args(a, m2):
    """The namespace m2_latency.py's builders read (its own parser, so defaults and names match)."""
    argv = ["--model", a.model, "--hub-repo", a.hub_repo, "--structure", a.structure, "--operators", a.operators,
            "--k", ",".join(str(k) for k in a.ks), "--variants", ",".join(a.variant_list),
            "--align", str(a.align), "--img-size", str(a.img_size), "--seed", str(a.seed),
            "--compile-mode", a.compile_mode, "--memory-fraction", str(a.memory_fraction)]
    return m2.parse_args(argv + (["--tiny"] if a.tiny else []))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Timing and profiling
# ----------------------------------------------------------------------------------------------
def prepare(module: nn.Module, x: torch.Tensor, mode: str, a, device, m2):
    """(callable, info): the module itself (eager) or torch.compile of it, compiled on x as
    m2_latency.time_model does (default mode, static shapes; eager if compilation fails, recorded)."""
    info = {"mode_used": "eager", "compile_ok": None, "compile_s": None, "compile_error": "", "graphs": None,
            "graph_breaks": None}
    if mode != "compile":
        return module, info
    torch._dynamo.reset()
    try:
        from torch._dynamo.utils import counters

        counters.clear()   # reset() keeps them; clear so graphs and graph breaks are this model's own
    except Exception:  # noqa: BLE001
        pass
    try:
        cm = torch.compile(module, mode=a.compile_mode, dynamic=False)
        t0 = time.time()
        with torch.inference_mode():
            cm(x)
        _sync(device)
        info.update({"mode_used": "compile", "compile_ok": True, "compile_s": round(time.time() - t0, 1),
                     **m2._dynamo_counts()})
        return cm, info
    except Exception as e:  # noqa: BLE001
        info.update({"mode_used": "eager (compile failed)", "compile_ok": False,
                     "compile_error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"})
        log(f"torch.compile failed ({info['compile_error']}); eager instead")
        torch._dynamo.reset()
        return module, info


def aggregate_events(events, gpu: bool) -> tuple[pd.DataFrame, dict]:
    """Per-name totals: GPU activity (kernels, memcpy, memset) on a GPU, `aten::` operator self times on CPU."""
    from torch.autograd import DeviceType

    acc: dict = {}
    skipped = 0
    lo, hi = float("inf"), float("-inf")
    for e in events:
        name = str(e.name)
        if gpu:
            if e.device_type != DeviceType.CUDA:
                continue
            if bool(getattr(e, "is_user_annotation", False)) or _is_annotation(name) or name.startswith("aten::"):
                skipped += 1
                continue
            us = float(e.time_range.end - e.time_range.start)
            lo, hi = min(lo, float(e.time_range.start)), max(hi, float(e.time_range.end))
        else:
            if e.device_type != DeviceType.CPU or not name.startswith("aten::"):
                continue
            us = float(e.self_cpu_time_total)
        t = acc.setdefault(name, [0.0, 0])
        t[0] += us
        t[1] += 1
    df = pd.DataFrame([{"name": n, "total_us": v[0], "calls": v[1]} for n, v in acc.items()],
                      columns=["name", "total_us", "calls"])
    src = "cuda kernel time (torch.profiler)" if gpu else "cpu operator self time (no GPU; smoke test)"
    # On one stream kernels do not overlap, so their summed time cannot exceed the span from the first kernel's
    # start to the last one's end; a larger sum would mean double counting (e.g. an annotation range kept).
    span = (hi - lo) if hi > lo else float("nan")
    total = float(df["total_us"].sum()) if len(df) else 0.0
    return df, {"time_source": src, "annotations_skipped": skipped, "kernel_span_us": span,
                "kernel_sum_over_span": total / span if span and span > 0 else float("nan")}


def profile_forwards(fn, x: torch.Tensor, iters: int, device) -> tuple[pd.DataFrame, dict]:
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.inference_mode():
            _sync(device)
            with profile(activities=acts, record_shapes=False, profile_memory=False, with_stack=False) as prof:
                for _ in range(iters):
                    fn(x)
                _sync(device)
        events = prof.events()
    return aggregate_events(events, device.type == "cuda")


def kernel_tables(kern: pd.DataFrame, iters: int, top: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(per-category {cat: (ms, count)} per forward, top kernels, all kernels with categories)."""
    k = kern.copy()
    k["category"] = [classify_kernel(n) for n in k["name"]]
    k["ms_per_forward"] = k["total_us"].astype(float) / 1e3 / iters
    k["calls_per_forward"] = k["calls"].astype(float) / iters
    k["mean_us"] = k["total_us"].astype(float) / k["calls"].clip(lower=1)
    k["triton_tag"] = [triton_tag(n) if c == "triton" else "" for n, c in zip(k["name"], k["category"])]
    total = float(k["ms_per_forward"].sum())
    k["share"] = k["ms_per_forward"] / total if total > 0 else float("nan")
    cats = {c: (float(k.loc[k["category"] == c, "ms_per_forward"].sum()),
                float(k.loc[k["category"] == c, "calls_per_forward"].sum())) for c in CATEGORIES}
    t = k.sort_values("ms_per_forward", ascending=False).head(int(top)).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["short_name"] = [short_name(n) for n in t["name"]]
    cols = ["rank", "short_name", "category", "triton_tag", "ms_per_forward", "share", "calls_per_forward",
            "mean_us", "name"]
    return cats, t[cols], k


class StreamRotation(nn.Module):
    """The stream rotation of a V-local FastBlock on its own: F.linear(x, rot_t), as FastBlock.forward applies it."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w)


def time_module(module: nn.Module, x: torch.Tensor, mode: str, a, device, m2) -> dict:
    """Median latency of module(x) (CUDA events; perf_counter on CPU) after warm-up, eager or compiled."""
    fn, info = prepare(module, x, mode, a, device, m2)
    try:
        r = m2.measure(fn, x, a.sub_warmup, a.sub_iters, device)
    except torch.cuda.OutOfMemoryError as e:
        r = {"latency_ms_median": float("nan"), "error": f"OOM: {str(e).splitlines()[0][:200]}"}
    if mode == "compile":
        torch._dynamo.reset()
    return {**info, **r}


def sub_block_list(conv: list, spec: str) -> list:
    conv = sorted(int(b) for b in conv)
    if spec == "all":
        return conv
    if spec == "first":
        return conv[:1]
    want = {int(b) for b in spec.split(",") if b.strip()}
    return [b for b in conv if b in want]


def submodule_parts(blk) -> list:
    """(part, kind, module, macs_token_exec) of one converted FastBlock: the non-dense attention and MLP and
    the stream rotations (FastBlock.rot_t, attn_rot_t)."""
    parts = []
    if blk.attn.cost.get("kind", "dense") != "dense":
        parts.append(("attention", blk.attn.cost["kind"], blk.attn, int(blk.attn.cost["macs_token_exec"])))
    if blk.mlp.cost.get("kind", "dense") != "dense":
        parts.append(("mlp", blk.mlp.cost["kind"], blk.mlp, int(blk.mlp.cost["macs_token_exec"])))
    if blk.rot_t is not None:
        parts.append(("rotation", "stream rotation", StreamRotation(blk.rot_t), int(blk.rot_t.numel())))
    if blk.attn_rot_t is not None:
        parts.append(("rotation", "attention-output rotation", StreamRotation(blk.attn_rot_t),
                      int(blk.attn_rot_t.numel())))
    return parts


# ----------------------------------------------------------------------------------------------
# Derived numbers
# ----------------------------------------------------------------------------------------------
def add_derived(runs: pd.DataFrame) -> pd.DataFrame:
    """Speed-up, deltas against dense (same mode and batch) and the GEMM time dense's throughput would give."""
    df = runs.copy()
    df["idle_ms"] = df["latency_ms_median"] - df["kernel_ms"]
    new = ["speedup_vs_dense", "flop_speedup_vs_dense", "gap_ms", "d_kernel_ms", "d_idle_ms", "d_launches",
           "gemm_ms_at_dense_rate"] + [f"d_{c}" for c in CATEGORIES]
    for col in new:
        df[col] = float("nan")
    for (mode, b), g in df.groupby(["mode", "batch"]):
        r = g[g["variant"] == "dense"]
        if not len(r):
            continue
        r = r.iloc[0]
        df.loc[g.index, "speedup_vs_dense"] = float(r["latency_ms_median"]) / g["latency_ms_median"]
        df.loc[g.index, "flop_speedup_vs_dense"] = float(r["macs_per_image"]) / g["macs_per_image"]
        df.loc[g.index, "gap_ms"] = g["latency_ms_median"] - float(r["latency_ms_median"])
        df.loc[g.index, "d_kernel_ms"] = g["kernel_ms"] - float(r["kernel_ms"])
        df.loc[g.index, "d_idle_ms"] = g["idle_ms"] - float(r["idle_ms"])
        df.loc[g.index, "d_launches"] = g["launches"] - float(r["launches"])
        for c in CATEGORIES:
            df.loc[g.index, f"d_{c}"] = g[f"ms_{c}"] - float(r[f"ms_{c}"])
        if float(r["gemm_gflop"]) > 0:
            df.loc[g.index, "gemm_ms_at_dense_rate"] = float(r["ms_gemm"]) * g["gemm_gflop"] / float(r["gemm_gflop"])
    return df


def long_table(runs: pd.DataFrame) -> pd.DataFrame:
    """m2_profile.csv: one row per variant, k, mode and category."""
    keep = ["model", "variant", "label", "k", "batch", "n_tokens", "mode", "mode_used"]
    run_cols = ["latency_ms_median", "speedup_vs_dense", "flop_speedup_vs_dense", "kernel_ms", "launches",
                "idle_ms", "gpu_busy_share", "gemm_tflops", "gemm_gflop", "gemm_ms_at_dense_rate", "macs_per_image",
                "gflops_per_image", "n_rotations", "ref_m2_clean_ms", "time_source", "kernel_sum_over_span",
                "annotations_skipped", "prof_iters", "iters", "warmup", "compile_s", "graphs", "graph_breaks",
                "compile_error", "error", "torch", "device", "git_hash", "seed"]
    rows = []
    for _, r in runs.iterrows():
        for c in CATEGORIES:
            ms = float(r[f"ms_{c}"])
            tot = float(r["kernel_ms"])
            rows.append({**{key: r[key] for key in keep}, "category": c, "category_label": CATEGORY_LABELS[c],
                         "ms_per_forward": ms, "share": ms / tot if tot > 0 else float("nan"),
                         "kernel_count": float(r[f"n_{c}"]), "delta_ms_vs_dense": r.get(f"d_{c}", float("nan")),
                         **{key: r.get(key) for key in run_cols}})
    return pd.DataFrame(rows)


def submodule_summary(sub: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per variant, k and mode: sums over the timed blocks of the converted and dense parts, and the gap."""
    if not len(sub):
        return pd.DataFrame()
    out = []
    for (v, k, mode), g in sub.groupby(["variant", "k", "mode"], sort=False):
        row = {"variant": v, "k": int(k), "mode": mode, "blocks": ",".join(str(b) for b in sorted(g["block"].unique()))}
        for part in ("mlp", "attention", "rotation"):
            p = g[g["part"] == part]
            row[f"{part}_ms"] = float(p["ms_median"].sum()) if len(p) else float("nan")
            row[f"{part}_dense_ms"] = float(p["dense_ms_median"].sum()) if len(p) else float("nan")
            row[f"{part}_delta_ms"] = float(p["delta_ms"].sum()) if len(p) else float("nan")
        row["sum_delta_ms"] = float(np.nansum([row[f"{p}_delta_ms"] for p in ("mlp", "attention", "rotation")]))
        r = runs[(runs["variant"] == v) & (runs["k"] == k) & (runs["mode"] == mode)]
        row["gap_ms"] = float(r["gap_ms"].iloc[0]) if len(r) else float("nan")
        row["remainder_ms"] = row["gap_ms"] - row["sum_delta_ms"]
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def _s(x, nd=2):
    """Signed number (+1.23 / -0.45)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:+.{nd}f}" if math.isfinite(x) else "n/a"


def _ok(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _range(vals, nd=2) -> str:
    v = [float(x) for x in vals if _ok(x)]
    if not v:
        return "n/a"
    lo, hi = min(v), max(v)
    return _f(lo, nd) if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{_f(lo, nd)} to {_f(hi, nd)}"


def _srange(vals, nd=2) -> str:
    """Range of signed numbers (+0.12 to +3.40)."""
    v = [float(x) for x in vals if _ok(x)]
    if not v:
        return "n/a"
    lo, hi = min(v), max(v)
    return _s(lo, nd) if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{_s(lo, nd)} to {_s(hi, nd)}"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    order = {v: i for i, v in enumerate(ORDER)}
    return df.assign(_o=df["variant"].map(lambda v: order.get(v, 99))).sort_values(["k", "_o"]).drop(columns="_o")


def _labels(m2) -> dict:
    return dict(getattr(m2, "LABELS", {}))


def row_sentence(r, d, label: str, unit: str) -> str:
    """One plain sentence per converted row, from its numbers against dense (same mode and batch)."""
    cats = [(c, float(r[f"d_{c}"])) for c in CATEGORIES if _ok(r[f"d_{c}"])]
    big = [(c, x) for c, x in sorted(cats, key=lambda t: -t[1]) if abs(x) >= 0.05]
    by_cat = ", ".join(f"{SHORT[c]} {_s(x)}" for c, x in big) if big else "no category changed by 0.05 ms or more"
    text = (f"- {label}, k = {int(r['k'])}: {_f(r['latency_ms_median'])} ms per forward against "
            f"{_f(d['latency_ms_median'])} ms for dense ({_f(r['speedup_vs_dense'])}x the dense speed; analytic "
            f"FLOP ratio {_f(r['flop_speedup_vs_dense'])}x). {unit} {_f(r['kernel_ms'])} ms against "
            f"{_f(d['kernel_ms'])} ms ({_s(r['d_kernel_ms'])} ms); by category (ms): {by_cat}. "
            f"Idle time (latency minus kernel time) {_s(r['d_idle_ms'])} ms. Kernel launches per forward "
            f"{_f(r['launches'], 0)} against {_f(d['launches'], 0)}.")
    if _ok(r["gemm_tflops"]) and _ok(d["gemm_tflops"]) and r["variant"] != "dense":
        text += (f" GEMM throughput {_f(r['gemm_tflops'], 1)} against {_f(d['gemm_tflops'], 1)} TFLOP/s; at the "
                 f"dense model's throughput its GEMMs would take {_f(r['gemm_ms_at_dense_rate'])} ms, measured "
                 f"{_f(r['ms_gemm'])} ms.")
    return text


def findings(runs: pd.DataFrame, mode: str, m2, unit: str) -> list:
    """Cross-variant statements for one mode, from the numbers only."""
    sub = runs[(runs["mode"] == mode)]
    d = sub[sub["variant"] == "dense"]
    conv = sub[~sub["variant"].isin(["dense", "original"])]
    if not len(d) or not len(conv):
        return []
    d = d.iloc[0]
    mir = conv[conv["variant"].isin(getattr(m2, "MIRROR", ()))]
    lines = []
    fell = sub[sub["mode_used"].astype(str) != mode]
    if len(fell):
        lines.append(f"- Note: {len(fell)} of {len(sub)} rows ran in mode '{', '.join(sorted(set(fell['mode_used'].astype(str))))}' "
                     f"instead of {mode} (see compile_error in m2_profile.csv).")
    if len(mir):
        slower = int((mir["gap_ms"] > 0).sum())
        lines.append(f"- Mirror rows slower than dense: {slower} of {len(mir)} (speed {_range(mir['speedup_vs_dense'])}x "
                     f"of dense, analytic FLOP ratio {_range(mir['flop_speedup_vs_dense'])}x).")
        more = int((mir["d_gemm"] > 0).sum())
        lines.append(f"- GEMM kernels: {_range(mir['ms_gemm'])} ms per forward in the Mirror rows against "
                     f"{_f(d['ms_gemm'])} ms for dense; Mirror rows spending more time in GEMMs than dense: {more} of "
                     f"{len(mir)}, although they execute {_range(100 * (1 - mir['gemm_gflop'] / float(d['gemm_gflop'])), 1)}% "
                     f"fewer GEMM multiply-adds (analytic). Achieved GEMM throughput {_range(mir['gemm_tflops'], 1)} "
                     f"TFLOP/s against {_f(d['gemm_tflops'], 1)} for dense; at the dense rate their GEMMs would take "
                     f"{_range(mir['gemm_ms_at_dense_rate'])} ms.")
        non = [c for c in CATEGORIES if c not in ("gemm", "attention")]
        mir_non = mir[[f"ms_{c}" for c in non]].sum(axis=1)
        d_non = float(sum(float(d[f"ms_{c}"]) for c in non))
        lines.append(f"- Non-GEMM, non-attention kernels ({', '.join(SHORT[c] for c in non)}): "
                     f"{_range(mir_non)} ms per forward against {_f(d_non)} ms for dense "
                     f"(change {_srange(mir_non - d_non)} ms).")
        lines.append(f"- Attention kernels (token mixing, unchanged by the conversion): {_range(mir['ms_attention'])} ms "
                     f"against {_f(d['ms_attention'])} ms for dense.")
        lines.append(f"- Kernel launches per forward: {_range(mir['launches'], 0)} against {_f(d['launches'], 0)} for "
                     f"dense; idle time (latency minus {unit.lower()}): {_range(mir['idle_ms'])} ms against "
                     f"{_f(d['idle_ms'])} ms.")
        share = []
        for _, r in mir.iterrows():
            if _ok(r["gap_ms"]) and float(r["gap_ms"]) > 0:
                parts = {c: float(r[f"d_{c}"]) for c in CATEGORIES}
                parts["idle"] = float(r["d_idle_ms"])
                top = max(parts, key=parts.get)
                share.append(f"{_labels(m2).get(r['variant'], r['variant'])} k = {int(r['k'])}: "
                             f"{SHORT.get(top, top)} ({_s(parts[top])} of {_f(r['gap_ms'])} ms)")
        if share:
            lines.append("- Largest single contributor to each slower Mirror row's gap: " + "; ".join(share) + ".")
    base = conv[conv["variant"].isin(["svd", "prune"])]
    for _, r in _sorted(base).iterrows():
        lines.append(f"- {_labels(m2).get(r['variant'], r['variant'])}, k = {int(r['k'])}: speed "
                     f"{_f(r['speedup_vs_dense'])}x of dense; GEMM {_f(r['ms_gemm'])} ms "
                     f"({_s(r['d_gemm'])}), other kernels {_s(r['d_kernel_ms'] - r['d_gemm'])} ms.")
    return lines


def write_report(out: Path, runs: pd.DataFrame, sub: pd.DataFrame, subsum: pd.DataFrame, tri: pd.DataFrame,
                 meta: dict, m2) -> Path:
    g = meta["git"]
    labels = _labels(m2)
    gpu = meta["device_type"] == "cuda"
    unit = "GPU kernel time" if gpu else "CPU operator time"
    clock = "CUDA events" if gpu else "wall clock, perf_counter"
    modes = [m for m in ("compile", "eager") if (runs["mode"] == m).any()]
    lines = [
        "# M2 profile: where the time goes in the converted models"
        + (" (TINY TEST, random weights, CPU operator times, not a measurement)" if meta["tiny"] else ""),
        "",
        f"Model `{meta['model']}` ({meta['depth']} blocks, d {meta['embed_dim']}, {meta['n_tokens']} tokens per image), "
        f"batch {meta['batch']}, bf16. Structure `{meta['provenance'].get('structure')}`, operators "
        f"`{meta['provenance'].get('operators', 'n/a')}`. Git `{str(g['git_hash'])[:10]}`"
        f"{' (uncommitted changes)' if g.get('git_dirty') else ''}, seed {meta['seed']}. Device: "
        f"{meta['env'].get('gpu', meta['env'].get('device'))}, torch {meta['env']['torch']}. Written {meta['date']}.",
        "",
        "Method. The models are the ones `scripts/m2_latency.py` times, built by its own code (same structure and "
        f"operator files, alignment {meta['align']}, bf16). Per variant, k and mode: latency is the median of "
        f"{meta['iters']} forwards ({clock}, after {meta['warmup']} warm-up forwards, no profiler attached); then "
        f"{meta['prof_iters']} more forwards were recorded with torch.profiler and the time of every GPU kernel "
        "(including memcpy and memset) was summed and divided by the number of forwards. Kernels are sorted into "
        "categories by name (`classify_kernel` in `scripts/m2_profile.py`). Idle time is latency minus kernel time "
        "(the GPU waiting for work), so for every row the latency gap to dense equals the sum of the category "
        "changes plus the idle change. Nothing was optimised.",
    ]
    if not gpu:
        lines += ["", "On CPU there are no GPU kernels: every number below is CPU operator self time from the "
                  "profiler. This run only checks that the pipeline works."]
    for mode in modes:
        f = findings(runs, mode, m2, unit)
        if f:
            lines += ["", f"## Main findings ({mode}, batch {meta['batch']})", ""] + f
    lines += ["", "## Per variant", ""]
    for mode in modes:
        sub_m = _sorted(runs[runs["mode"] == mode])
        d = sub_m[sub_m["variant"] == "dense"]
        if not len(d):
            continue
        d = d.iloc[0]
        lines += [f"{mode.capitalize()} (mode used for dense: {d['mode_used']}):", ""]
        for _, r in sub_m.iterrows():
            if r["variant"] in ("dense",):
                continue
            lines.append(row_sentence(r, d, labels.get(r["variant"], r["variant"]), unit))
        lines.append("")
    # Category tables.
    ref_col = runs["ref_m2_clean_ms"].notna().any() if "ref_m2_clean_ms" in runs else False
    for mode in modes:
        sub_m = _sorted(runs[runs["mode"] == mode])
        lines += [
            f"## {unit} per forward by category ({mode}, ms)",
            "",
            "Share: category time over total kernel time. GPU busy: kernel time over latency."
            + (" M2 clean: the latency `runs/m2_clean` reported for the same variant, k, batch and mode." if ref_col else ""),
            "",
            "| k | variant | latency | " + ("M2 clean | " if ref_col else "") + "speed vs dense | kernel time | GPU busy | "
            "launches | " + " | ".join(SHORT[c] for c in CATEGORIES) + " |",
            "|---|---|---|" + ("---|" if ref_col else "") + "---|---|---|---|" + "---|" * len(CATEGORIES),
        ]
        for _, r in sub_m.iterrows():
            cells = " | ".join(f"{_f(r[f'ms_{c}'])} ({_f(100 * r[f'ms_{c}'] / r['kernel_ms'], 0) if r['kernel_ms'] > 0 else 'n/a'}%)"
                               for c in CATEGORIES)
            lines.append(f"| {int(r['k'])} | {labels.get(r['variant'], r['variant'])} | {_f(r['latency_ms_median'])} | "
                         + (f"{_f(r.get('ref_m2_clean_ms'))} | " if ref_col else "")
                         + f"{_f(r['speedup_vs_dense'])} | {_f(r['kernel_ms'])} | {_f(100 * r['gpu_busy_share'], 0)}% | "
                         f"{_f(r['launches'], 0)} | {cells} |")
        lines += [
            "",
            f"Change against dense ({mode}, ms per forward; the columns add up to the latency gap):",
            "",
            "| k | variant | gap | " + " | ".join(SHORT[c] for c in CATEGORIES) + " | idle |",
            "|---|---|---|" + "---|" * (len(CATEGORIES) + 1),
        ]
        for _, r in sub_m.iterrows():
            if r["variant"] == "dense":
                continue
            lines.append(f"| {int(r['k'])} | {labels.get(r['variant'], r['variant'])} | {_s(r['gap_ms'])} | "
                         + " | ".join(_s(r[f"d_{c}"]) for c in CATEGORIES) + f" | {_s(r['d_idle_ms'])} |")
        lines.append("")
    # GEMM efficiency.
    lines += [
        "## GEMM efficiency",
        "",
        "Analytic GEMM work: multiply-adds executed by matrix products per forward (patch embedding, every weight "
        "matrix with its alignment padding, rotations; token mixing excluded, it runs inside the attention kernel), "
        "from `mirror.efficient.fast_vit_cost`. Throughput: 2 x that work over the measured GEMM kernel time. "
        "At dense rate: the GEMM time this work would take at the dense model's measured throughput.",
        "",
        "| mode | k | variant | GEMM ms | GEMM launches | analytic GEMM GFLOP | TFLOP/s | at dense rate (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        for _, r in _sorted(runs[runs["mode"] == mode]).iterrows():
            lines.append(f"| {mode} | {int(r['k'])} | {labels.get(r['variant'], r['variant'])} | {_f(r['ms_gemm'])} | "
                         f"{_f(r['n_gemm'], 0)} | {_f(r['gemm_gflop'], 1)} | {_f(r['gemm_tflops'], 1)} | "
                         f"{_f(r['gemm_ms_at_dense_rate'])} |")
    # Triton breakdown.
    if len(tri):
        tags = sorted(tri["triton_tag"].unique())
        lines += [
            "",
            "## Fused Triton kernels by what they fuse (compile)",
            "",
            "From the operations Inductor names in each kernel (e.g. `triton_poi_fused_add_cat_gelu_sub`): norm = mean, "
            "pow or rsqrt; cat/gather/pad = cat, index, gather, scatter, pad, clone, permute, slice or select; "
            "activation = GELU and similar; pointwise = none of these (adds, multiplies, dtype casts). ms per forward "
            "(launches).",
            "",
            "| mode | k | variant | " + " | ".join(tags) + " |",
            "|---|---|---|" + "---|" * len(tags),
        ]
        keys = _sorted(tri.drop_duplicates(["mode", "variant", "k"])[["mode", "variant", "k"]])
        for _, kr in keys.iterrows():
            mode, v, k = kr["mode"], kr["variant"], kr["k"]
            t = tri[(tri["mode"] == mode) & (tri["variant"] == v) & (tri["k"] == k)].set_index("triton_tag")
            lines.append(f"| {mode} | {int(k)} | {labels.get(v, v)} | " + " | ".join(
                f"{_f(t.loc[tg, 'ms_per_forward'])} ({_f(t.loc[tg, 'kernel_count'], 0)})" if tg in t.index else "0"
                for tg in tags) + " |")
    # Sub-modules.
    if len(subsum):
        lines += [
            "",
            "## Converted sub-modules timed alone",
            "",
            f"Each converted block's attention and MLP, and the V-local stream rotation, timed alone on a bf16 input of "
            f"{meta['batch']} x {meta['n_tokens']} tokens x {meta['embed_dim']} (median of {meta['sub_iters']} runs after "
            f"{meta['sub_warmup']} warm-up runs, {clock}), against the same block of the dense model. Sums over the "
            "timed blocks (ms). Remainder: end-to-end gap minus the summed sub-module changes (norms, residual adds, "
            "the embedding and head, and fusion across module boundaries in compile mode, which isolated timing "
            "cannot see).",
            "",
            "| mode | k | variant | blocks | MLP | MLP dense | MLP change | attention | attention dense | "
            "attention change | rotations | sum of changes | end-to-end gap | remainder |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in subsum.iterrows():
            lines.append(f"| {r['mode']} | {int(r['k'])} | {labels.get(r['variant'], r['variant'])} | {r['blocks']} | "
                         f"{_f(r['mlp_ms'])} | {_f(r['mlp_dense_ms'])} | {_s(r['mlp_delta_ms'])} | {_f(r['attention_ms'])} | "
                         f"{_f(r['attention_dense_ms'])} | {_s(r['attention_delta_ms'])} | {_f(r['rotation_ms'])} | "
                         f"{_s(r['sum_delta_ms'])} | {_s(r['gap_ms'])} | {_s(r['remainder_ms'])} |")
        lines += [""]
        for _, r in subsum.iterrows():
            gap = float(r["gap_ms"]) if _ok(r["gap_ms"]) else float("nan")
            parts = []
            for p, name in (("mlp", "MLPs"), ("attention", "attention"), ("rotation", "rotations")):
                x = r[f"{p}_delta_ms"]
                if _ok(x):
                    pct = f" ({_f(100 * float(x) / gap, 0)}% of the gap)" if _ok(gap) and gap > 0 else ""
                    parts.append(f"{name} {_s(x)} ms{pct}")
            lines.append(f"- {r['mode']}, {labels.get(r['variant'], r['variant'])}, k = {int(r['k'])}: converted "
                         + ", ".join(parts) + f"; end-to-end gap {_s(gap)} ms.")
        # Per-part ratios of time against analytic work.
        if len(sub):
            agg = sub.groupby(["mode", "variant", "k", "part"], sort=False).agg(
                ms=("ms_median", "sum"), dense_ms=("dense_ms_median", "sum"), macs=("macs_token_exec", "sum"),
                dense_macs=("dense_macs_token_exec", "sum")).reset_index()
            lines += [
                "",
                "Time against analytic work, per part (sums over the timed blocks): time ratio = converted over dense "
                "time; work ratio = converted over dense executed multiply-adds per token (rotations: d x d against "
                "nothing in dense).",
                "",
                "| mode | k | variant | part | time ratio | work ratio |",
                "|---|---|---|---|---|---|",
            ]
            for _, r in agg.iterrows():
                tr = r["ms"] / r["dense_ms"] if r["dense_ms"] > 0 else float("nan")
                wr = r["macs"] / r["dense_macs"] if r["dense_macs"] > 0 else float("nan")
                lines.append(f"| {r['mode']} | {int(r['k'])} | {labels.get(r['variant'], r['variant'])} | {r['part']} | "
                             f"{_f(tr)} | {_f(wr)} |")
    smi = meta.get("nvidia_smi", {})
    lines += [
        "",
        "## Caveats",
        "",
        f"- One device and one session; the GPU is shared. nvidia-smi before: {smi.get('before', 'n/a')}; after: "
        f"{smi.get('after', 'n/a')}. Other processes running at the same time would inflate latencies; kernel times "
        "are less affected than latency but are not immune (shared memory bandwidth and clocks).",
        "- Latency and the profile come from different forwards of the same compiled model; the profiler adds CPU "
        "overhead, which does not change kernel durations but can change idle time. Idle can be negative if kernels "
        "ran slower under the profiler than in the timed runs.",
        "- Categories come from kernel names. In eager mode the casts to and from float32 inside RMSNorm are copy "
        "kernels and count as memory movement (in dense and converted models alike); in compile mode they are "
        "fused into Triton kernels. A Triton GEMM template (max-autotune, not used here) would count as GEMM.",
        "- Sub-modules timed alone do not see fusion with their neighbours (compile mode) or the cost of the "
        "norms and residual adds around them; their sum need not equal the end-to-end gap.",
        "- torch.compile default mode, no CUDA graphs, plain PyTorch ops as in M2. This is a measurement of that "
        "implementation, not of what a fused kernel could do.",
        "- One structure (the M1 fit seed); latency depends on the shapes, not the fitted values.",
    ]
    if gpu and "kernel_sum_over_span" in runs and runs["kernel_sum_over_span"].notna().any():
        mx = float(runs["kernel_sum_over_span"].max())
        lines.append(f"- Self-check: summed kernel time over the span from the first kernel's start to the last one's "
                     f"end is at most {_f(mx, 3)} over all rows (above 1 would mean overlapping kernels or double "
                     f"counting; record_function ranges were excluded: {int(runs['annotations_skipped'].max())} at most "
                     f"per row).")
    if meta.get("errors"):
        lines.append(f"- Errors: {len(meta['errors'])} (see m2_profile_meta.json).")
    if meta.get("skipped"):
        lines.append("- Skipped: " + "; ".join(meta["skipped"]) + ".")
    lines += ["", "Files: m2_profile.csv, m2_submodules.csv, m2_triton.csv, m2_top_kernels.csv, top_kernels/, "
              "m2_profile_meta.json."]
    text = "\n".join(lines) + "\n"
    assert chr(0x2014) not in text and chr(0x2013) not in text
    path = out / "M2_PROFILE.md"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
def reference_latency(path: str, model: str, tiny: bool) -> pd.DataFrame | None:
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    if tiny or not p.is_file():
        return None
    try:
        ref = pd.read_csv(p)
        if "model" in ref and (ref["model"] != model).all():
            return None
        return ref[["variant", "k", "batch", "mode", "latency_ms_median"]]
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    m2 = load_m2()
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/m2_profile" if tiny else "runs/m2_profile")
    out.mkdir(parents=True, exist_ok=True)
    (out / "top_kernels").mkdir(exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(a.memory_fraction))   # m2_latency import set 0.3, then 0.1
    torch.backends.cuda.matmul.allow_tf32 = False   # as m2_latency.py (timing is bf16 either way)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    tags = {"git_hash": git["git_hash"], "seed": a.seed}
    log(f"M2 profile {'TINY ' if tiny else ''}on {device}, out {out}, commit {str(git['git_hash'])[:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")
    smi = {"before": m2.nvidia_smi() if device.type == "cuda" else "cpu"}
    ma = m2_args(a, m2)
    bad = set(a.variant_list) - set(m2.ALL_VARIANTS)
    if bad:
        raise SystemExit(f"unknown variants {sorted(bad)}")

    with timer.stage("load"):
        adapter = m2.m1.build_adapter(ma, device)
        ma.img_px = int(adapter.img_size)
        a.img_px = ma.img_px
        L, d = adapter.depth, adapter.embed_dim
        H, dh = attn_dims(adapter.model.blocks[0])
        T, n_patches = adapter.n_tokens, adapter.grid_hw[0] * adapter.grid_hw[1]
        S, ops, prov = m2.tiny_structure(ma, adapter) if tiny else m2.load_structure(ma, adapter)
        ks = [k for k in a.ks if 1 <= k <= L]
    log(f"depth {L}, d {d}, heads {H} x {dh}, tokens {T}; ks {ks}; variants {a.variant_list}; batch {a.batch}; "
        f"modes {a.mode_list}")
    ref_lat = reference_latency(a.reference, a.model, tiny)

    plan = [("dense", 0)] + [("original", 0)] * ("original" in a.variant_list)
    plan += [(v, k) for k in ks for v in ORDER if v in a.variant_list and v not in ("dense", "original")]
    run_rows, sub_rows, tri_rows, top_rows, skipped, errors = [], [], [], [], [], []
    dense_bf16, dense_cost = None, None
    dense_sub: dict = {}
    g = torch.Generator(device=device).manual_seed(a.seed + 1)
    x_sub = torch.randn(a.batch, T, d, generator=g, device=device).to(torch.bfloat16)
    common = {"model": a.model if not tiny else "tiny", "batch": a.batch, "n_tokens": T, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", **tags}

    def flush():
        runs = add_derived(pd.DataFrame(run_rows)) if run_rows else pd.DataFrame()
        if len(runs):
            long_table(runs).to_csv(out / "m2_profile.csv", index=False)
        pd.DataFrame(sub_rows).to_csv(out / "m2_submodules.csv", index=False)
        pd.DataFrame(tri_rows, columns=["variant", "k", "mode", "triton_tag", "ms_per_forward", "kernel_count",
                                        "git_hash", "seed"]).to_csv(out / "m2_triton.csv", index=False)
        if top_rows:
            pd.concat(top_rows, ignore_index=True).to_csv(out / "m2_top_kernels.csv", index=False)
        return runs

    with timer.stage("build, time and profile"):
        for v, k in plan:
            if v == "vexact" and k not in S["A_star"]:
                skipped.append(f"vexact k = {k}: no span operator for this k in the structure file")
                log(skipped[-1])
                continue
            log(f"== {v} k = {k}")
            try:
                ref32, eff32 = m2.make_pair(v, k, adapter, S, ops, a.align)
            except Exception as e:  # noqa: BLE001
                errors.append(f"build {v} k = {k}: {type(e).__name__}: {e}")
                log(errors[-1])
                continue
            if v == "original":
                timed = m2.AdapterFeature(adapter, copy.deepcopy(adapter.model).to(torch.bfloat16)).eval()
                cost, conv_blocks = dense_cost, []
            else:
                cost = m2.fast_vit_cost(eff32, T, n_patches)
                conv_blocks = list(eff32.info.get("converted_blocks", []))
                timed = m2._bf16(eff32)
                if v == "dense":
                    dense_cost = cost
            del ref32, eff32
            _free(device)
            gemm_macs = float(cost["macs_exec"] - cost["macs_mix"]) * a.batch   # per forward
            static = {**common, "variant": v, "label": m2.LABELS.get(v, v), "k": k,
                      "macs_per_image": float(cost["macs"]), "gflops_per_image": float(cost["gflops"]),
                      "gemm_gflop": 2.0 * gemm_macs / 1e9, "n_rotations": int(getattr(timed, "info", {}).get("n_rotations", 0)),
                      "iters": a.iters, "warmup": a.warmup, "prof_iters": a.prof_iters}
            gx = torch.Generator(device=device).manual_seed(a.seed + a.batch)      # m2_latency.time_model's input
            x = torch.randn(a.batch, 3, a.img_px, a.img_px, generator=gx, device=device).to(torch.bfloat16)
            for mode in a.mode_list:
                try:
                    fn, info = prepare(timed, x, mode, a, device, m2)
                    lat = m2.measure(fn, x, a.warmup, a.iters, device)
                    kern, pinfo = profile_forwards(fn, x, a.prof_iters, device)
                    err = ""
                except Exception as e:  # noqa: BLE001 (OOM or a failure in the profiler; keep the other rows)
                    msg = str(e).splitlines()[0][:200] if str(e) else ""
                    errors.append(f"{v} k = {k} {mode}: {type(e).__name__}: {msg}")
                    log(errors[-1])
                    torch._dynamo.reset()
                    _free(device)
                    continue
                if mode == "compile":
                    torch._dynamo.reset()
                cats, top, allk = kernel_tables(kern, a.prof_iters, a.top)
                kms = float(sum(ms for ms, _ in cats.values()))
                row = {**static, "mode": mode, **info, **{key: lat[key] for key in lat},
                       "kernel_ms": kms, "launches": float(sum(n for _, n in cats.values())),
                       "gpu_busy_share": kms / lat["latency_ms_median"] if lat["latency_ms_median"] > 0 else float("nan"),
                       **{f"ms_{c}": cats[c][0] for c in CATEGORIES}, **{f"n_{c}": cats[c][1] for c in CATEGORIES},
                       "gemm_tflops": (static["gemm_gflop"] / cats["gemm"][0]) if cats["gemm"][0] > 0 else float("nan"),
                       "error": err, **pinfo}
                if ref_lat is not None:
                    rr = ref_lat[(ref_lat["variant"] == v) & (ref_lat["k"] == k) & (ref_lat["batch"] == a.batch)
                                 & (ref_lat["mode"] == mode)]
                    row["ref_m2_clean_ms"] = float(rr["latency_ms_median"].iloc[0]) if len(rr) else float("nan")
                run_rows.append(row)
                top.insert(0, "mode", mode)
                top.insert(0, "k", k)
                top.insert(0, "variant", v)
                top["git_hash"], top["seed"] = tags["git_hash"], a.seed
                top.to_csv(out / "top_kernels" / f"{v}_k{k}_{mode}.csv", index=False)
                top_rows.append(top)
                tr = allk[allk["category"] == "triton"]
                for tag, gr in tr.groupby("triton_tag"):
                    tri_rows.append({"variant": v, "k": k, "mode": mode, "triton_tag": tag,
                                     "ms_per_forward": float(gr["ms_per_forward"].sum()),
                                     "kernel_count": float(gr["calls_per_forward"].sum()), **tags})
                log(f"{v} k = {k} {mode} ({info['mode_used']}): median {lat['latency_ms_median']:.3f} ms, kernels "
                    f"{kms:.3f} ms in {row['launches']:.0f} launches; GEMM {cats['gemm'][0]:.3f} ms")
                flush()
            # Sub-modules of the converted blocks, against the same blocks of dense.
            if v == "dense":
                dense_bf16 = timed
            elif v != "original" and not a.no_submodules and dense_bf16 is not None:
                blocks = sub_block_list(conv_blocks, a.sub_blocks)
                for mode in a.mode_list:
                    for b in blocks:
                        for part, kind, mod, macs in submodule_parts(timed.blocks[b]):
                            if part == "rotation":
                                dres, dmacs = {"latency_ms_median": 0.0}, 0
                            else:
                                key = (b, part, mode)
                                dmod = dense_bf16.blocks[b].attn if part == "attention" else dense_bf16.blocks[b].mlp
                                if key not in dense_sub:
                                    dense_sub[key] = time_module(dmod, x_sub, mode, a, device, m2)
                                dres, dmacs = dense_sub[key], int(dmod.cost["macs_token_exec"])
                            r = time_module(mod, x_sub, mode, a, device, m2)
                            ms, dms = float(r["latency_ms_median"]), float(dres["latency_ms_median"])
                            sub_rows.append({**common, "variant": v, "label": m2.LABELS.get(v, v), "k": k, "mode": mode,
                                             "mode_used": r["mode_used"], "block": b, "part": part, "kind": kind,
                                             "ms_median": ms, "ms_p10": r.get("latency_ms_p10"),
                                             "ms_p90": r.get("latency_ms_p90"), "dense_ms_median": dms,
                                             "delta_ms": ms - dms, "time_ratio": ms / dms if dms > 0 else float("nan"),
                                             "macs_token_exec": macs, "dense_macs_token_exec": dmacs,
                                             "work_ratio": macs / dmacs if dmacs > 0 else float("nan"),
                                             "dense_mode_used": dres.get("mode_used", ""),
                                             "compile_s": r.get("compile_s"), "error": r.get("error", ""),
                                             "sub_iters": a.sub_iters, "sub_warmup": a.sub_warmup})
                    log(f"{v} k = {k} {mode}: sub-modules of blocks {blocks} timed")
                flush()
            if v != "dense":
                del timed
            _free(device)

    smi["after"] = m2.nvidia_smi() if device.type == "cuda" else "cpu"
    runs = flush()
    sub = pd.DataFrame(sub_rows)
    subsum = submodule_summary(sub, runs) if len(runs) else pd.DataFrame()
    tri = pd.DataFrame(tri_rows)
    meta = {
        "experiment": "M2 profile (why the converted models are slower than dense)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model if not tiny else "tiny random timm ViT",
        "tiny": tiny, "seed": a.seed, "git": git, "env": env_info(device), "device_type": device.type,
        "args": {key: val for key, val in vars(a).items()}, "m2_args": {key: val for key, val in vars(ma).items()},
        "provenance": prov, "ks": ks, "variants": [v for v, _ in plan], "batch": a.batch, "modes": a.mode_list,
        "warmup": a.warmup, "iters": a.iters, "prof_iters": a.prof_iters, "sub_warmup": a.sub_warmup,
        "sub_iters": a.sub_iters, "sub_blocks": a.sub_blocks, "align": a.align, "compile_mode": a.compile_mode,
        "depth": L, "embed_dim": d, "heads": H, "head_dim": dh, "n_tokens": T, "n_patches": n_patches,
        "img_size": a.img_px, "memory_fraction": a.memory_fraction, "nvidia_smi": smi, "skipped": skipped,
        "errors": errors, "reference": a.reference if ref_lat is not None else None,
        "profiler": {"activities": "CPU and CUDA" if device.type == "cuda" else "CPU",
                     "time_source": "cuda kernel time" if device.type == "cuda" else "cpu operator self time",
                     "excluded": "record_function annotations (gpu_user_annotation, '## ... ##', compiled-region markers)"},
        "categories": CATEGORY_LABELS, "timing_dtype": "bfloat16 weights and inputs", "tf32": False,
        "timing": {"stages": timer.stages, "total_s": timer.total()},
        "files": ["m2_profile.csv", "m2_submodules.csv", "m2_triton.csv", "m2_top_kernels.csv", "top_kernels/",
                  "M2_PROFILE.md", "m2_profile_meta.json"],
    }
    with open(out / "m2_profile_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    report = write_report(out, runs, sub, subsum, tri, meta, m2) if len(runs) else None
    log(f"wrote {report} in {timer.total() / 60:.1f} min")
    return {"out": str(out), "meta": meta, "runs": runs, "submodules": sub, "submodule_summary": subsum,
            "triton": tri}


if __name__ == "__main__":
    main()

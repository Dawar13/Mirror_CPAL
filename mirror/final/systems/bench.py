"""A12: systems table and profile of the even/odd form (FINAL_PHASE_PLAN.md v2, item A12; PREREG_FINAL_PHASE.md
Section 2.5 uses the batch-256 time for A07).

For a list of models (the original; Mirror stored densely and in its even/odd form; SVD-LLM; IO-SVD; the dense
students; flip averaging) this module reports:
  * parameters: the analytic count of the method (the paper's count, passed in by the caller) and the entries the
    module actually stores (parameters plus floating-point buffers), with the parameter memory in bf16
    (2 bytes per stored entry; MB = 10^6 bytes);
  * analytic FLOPs per image: the MUST-09 cost accounting (scripts/collect_results.analytic_cost through
    mirror.heal_structured.block_costs; mirror.efficient.fast_vit_cost for an even/odd module), two passes for
    flip averaging;
  * measured latency and throughput at batch 1, 64 and 256: bf16 weights and inputs, torch.compile (default mode,
    static shapes; eager if compilation fails, recorded), median of 100 timed runs after 20 warm-up runs, timed with
    CUDA events, input randn images from a generator seeded with seed + batch (as scripts/m2_latency.time_model),
    and the peak memory of compile, warm-up and timing at that batch (torch.cuda.max_memory_allocated);
  * a torch.profiler summary of the even/odd form (the kernel categories of scripts/m2_profile.py, grouped into
    GEMM, butterflies and permutes, attention and the rest), the achieved TFLOP/s of every GEMM shape of the model
    (each shape timed on its own with random bf16 operands) and the effect of alignment (mirror.efficient align 1
    against 8: d+ = 700 and d- = 68 for DINOv2-B are not multiples of 8; align 8 pads them to 704 and 72).
Under torch.compile the butterfly additions and GELU fuse into one Triton kernel, so the "butterflies and
permutes" group then holds every fused pointwise kernel; the dense original's share of the same group is the
reference for the overhead.

Timing goes through a timer object (CudaEventTimer on CUDA, WallTimer on CPU), so the plumbing is testable on the
CPU with a fake timer. Nothing here allocates GPU memory caps; the script sets the process cap first.

Public API: BenchEntry, FlipAverage, FeatureModule, LogitsModule, CudaEventTimer, WallTimer, param_memory,
analytic_cost, model_arch, structure_cost, even_odd_cost, measure, time_module, run_benchmark, systems_table,
gemm_shapes, gemm_table, profile_summary, group_categories, align_effect.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data import mirror
from ...utils import REPO_ROOT

__all__ = [
    "BenchEntry", "FlipAverage", "FeatureModule", "LogitsModule", "CudaEventTimer", "WallTimer", "param_memory",
    "analytic_cost", "model_arch", "structure_cost", "even_odd_cost", "measure", "time_module", "run_benchmark",
    "systems_table", "gemm_shapes", "gemm_table", "profile_summary", "group_categories", "align_effect", "GROUPS",
]

BATCHES = (1, 64, 256)
GROUPS = {"gemm": "GEMM", "memory": "butterflies and permutes", "elementwise": "butterflies and permutes",
          "triton": "butterflies and permutes", "attention": "attention", "norm": "rest", "other": "rest"}
GPU_FRAC: float | None = None      # set by the caller; re-applied after loading a script that sets its own cap


# ----------------------------------------------------------------------------------------------
# Scripts reused as modules (loaded once, lazily)
# ----------------------------------------------------------------------------------------------
def _script(name: str):
    key = f"_fp_bench_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
        if GPU_FRAC is not None and torch.cuda.is_available():     # scripts/m2_profile.py caps at 0.25 on import
            torch.cuda.set_per_process_memory_fraction(float(GPU_FRAC))
    return sys.modules[key]


# ----------------------------------------------------------------------------------------------
# Entries and wrappers
# ----------------------------------------------------------------------------------------------
@dataclass
class BenchEntry:
    """One model to time. `module` maps images [B, 3, S, S] to features (or logits). `passes` is 2 for flip
    averaging. `params_analytic` is the method's own parameter count (the paper tables); `analytic` holds
    {"gflops": per image, "source": ...} for one pass."""

    name: str
    module: nn.Module
    kind: str = "model"
    passes: int = 1
    params_analytic: int | None = None
    analytic: dict | None = None
    meta: dict = field(default_factory=dict)


class FeatureModule(nn.Module):
    """adapter.cls_feature(x, model): the k-NN feature of a hub or timm model (as scripts/m2_latency.AdapterFeature)."""

    def __init__(self, adapter, model: nn.Module):
        super().__init__()
        self.model = model
        self._adapter = adapter

    def forward(self, x):
        return self._adapter.cls_feature(x, self.model)


class LogitsModule(nn.Module):
    """A classifier's logits through the adapter (DeiT-III): head(fc_norm(pool(norm(tokens))))."""

    def __init__(self, adapter, model: nn.Module):
        super().__init__()
        self.model = model
        self._adapter = adapter

    def forward(self, x):
        m = self.model
        return m.forward_head(m.norm(self._adapter.forward_tokens(x, m)))


class FlipAverage(nn.Module):
    """(f(x) + f(mirror x)) / 2. Two passes by default (the protocol of MUST-10 and A07); batched=True runs one
    pass on [x; mirror x] instead (the same arithmetic, often faster at small batches)."""

    def __init__(self, base: nn.Module, batched: bool = False):
        super().__init__()
        self.base, self.batched = base, bool(batched)

    def forward(self, x):
        if self.batched:
            f = self.base(torch.cat([x, mirror(x)]))
            B = x.shape[0]
            return 0.5 * (f[:B] + f[B:])
        return 0.5 * (self.base(x) + self.base(mirror(x)))


# ----------------------------------------------------------------------------------------------
# Parameters, memory and analytic cost
# ----------------------------------------------------------------------------------------------
def param_memory(module: nn.Module) -> dict:
    """Stored entries (parameters plus floating-point buffers such as the even head, low-rank bases and the even/odd
    form's output map; index buffers excluded) and their memory in bf16 (2 bytes each, MB = 10^6 bytes)."""
    params = int(sum(p.numel() for p in module.parameters()))
    bufs = int(sum(b.numel() for b in module.buffers() if b.is_floating_point()))
    return {"params_stored": params, "float_buffers": bufs, "entries_stored": params + bufs,
            "param_mb_bf16": 2.0 * (params + bufs) / 1e6}


def model_arch(adapter) -> dict:
    """Inputs of the MUST-09 cost model (scripts/must09_heal.cost_of)."""
    H, W = adapter.grid_hw
    proj = adapter.model.patch_embed.proj
    return {"d": int(adapter.embed_dim), "h": int(adapter.mlp_hidden), "depth": int(adapter.depth),
            "T": int(adapter.n_tokens), "n_patches": int(H * W), "patch": int(adapter.patch_size),
            "in_chans": int(getattr(proj, "in_channels", 3)),
            "model_params_dense": int(sum(p.numel() for p in adapter.model.parameters()))}


def analytic_cost(arch: dict, conv_blocks: list) -> dict:
    """scripts/collect_results.analytic_cost (the cost model of the paper tables and MUST-09)."""
    return _script("collect_results").analytic_cost(arch, conv_blocks)


def structure_cost(adapter, model: nn.Module | None = None, spec: dict | None = None,
                   structure: dict | None = None) -> dict:
    """Analytic parameters and GFLOPs per image with the MUST-09 accounting: the original (no spec) or a healed
    student (mirror.heal_structured.block_costs of its spec and, for the mirror family, its structure)."""
    arch = model_arch(adapter)
    if spec is None:
        c = analytic_cost(arch, [])
        return {"gflops": c["gflops_model_dense"], "params_model": arch["model_params_dense"],
                "source": "scripts/collect_results.analytic_cost (dense)"}
    from ...heal_structured import block_costs

    blocks = block_costs(model, spec, structure, arch["d"], arch["h"])
    c = analytic_cost(arch, [{k: b[k] for k in ("mlp", "attn", "rot")} for b in blocks])
    basis = int(sum(b["basis_params"] for b in blocks))
    return {"gflops": c["gflops_model"], "params_model": int(c["model_params"]) + basis,
            "params_frac": (int(c["model_params"]) + basis) / arch["model_params_dense"],
            "gflops_frac": c["gflops_model"] / c["gflops_model_dense"],
            "source": "scripts/collect_results.analytic_cost via mirror.heal_structured.block_costs"}


def even_odd_cost(fv: nn.Module, adapter) -> dict:
    """mirror.efficient.fast_vit_cost of an even/odd module (analytic, without padding; *_exec with padding)."""
    from ...efficient import fast_vit_cost

    H, W = adapter.grid_hw
    c = fast_vit_cost(fv, adapter.n_tokens, H * W)
    return {"gflops": c["gflops"], "gflops_exec": c["gflops_exec"], "padding_params": c["padding_params"],
            "source": "mirror.efficient.fast_vit_cost"}


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------
class CudaEventTimer:
    """Milliseconds of one call, CUDA events around it (as scripts/m2_latency.measure)."""

    def __call__(self, fn, x) -> float:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(x)
        e.record()
        e.synchronize()
        return float(s.elapsed_time(e))


class WallTimer:
    """Milliseconds of one call by perf_counter (CPU; synchronises a CUDA device first and after)."""

    def __init__(self, device=None):
        self.cuda = device is not None and torch.device(device).type == "cuda"

    def __call__(self, fn, x) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if self.cuda:
            torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0)


def default_timer(device):
    return CudaEventTimer() if torch.device(device).type == "cuda" else WallTimer(device)


def measure(fn, x: torch.Tensor, warmup: int, iters: int, timer) -> dict:
    """Median, 10th and 90th percentile, mean and minimum of `iters` timed calls after `warmup` untimed ones."""
    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn(x)
        ts = np.asarray([timer(fn, x) for _ in range(int(iters))], dtype=np.float64)
    return {"latency_ms_median": float(np.median(ts)), "latency_ms_p10": float(np.percentile(ts, 10)),
            "latency_ms_p90": float(np.percentile(ts, 90)), "latency_ms_mean": float(ts.mean()),
            "latency_ms_min": float(ts.min()), "n_timed": int(len(ts)), "n_warmup": int(warmup)}


def _compile(module: nn.Module, x: torch.Tensor, mode: str):
    info = {"mode_used": "eager", "compile_ok": None, "compile_s": None, "compile_error": ""}
    torch._dynamo.reset()
    try:
        cm = torch.compile(module, mode=mode, dynamic=False)
        t0 = time.time()
        with torch.inference_mode():
            cm(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        info.update({"mode_used": "compile", "compile_ok": True, "compile_s": round(time.time() - t0, 1)})
        return cm, info
    except Exception as e:  # noqa: BLE001
        info.update({"mode_used": "eager (compile failed)", "compile_ok": False,
                     "compile_error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"})
        torch._dynamo.reset()
        return module, info


def time_module(module: nn.Module, batch: int, *, img_px: int, device, dtype=torch.bfloat16, compile: bool = True,
                compile_mode: str = "default", warmup: int = 20, iters: int = 100, seed: int = 0, timer=None) -> dict:
    """One (module, batch) measurement: input randn(batch, 3, img_px, img_px) from a generator seeded with
    seed + batch (as scripts/m2_latency.time_model), cast to `dtype`; compile (or eager), warm-up, timed runs; the
    peak memory of all of it on CUDA."""
    device = torch.device(device)
    timer = timer or default_timer(device)
    g = torch.Generator(device=device).manual_seed(int(seed) + int(batch))
    x = torch.randn(int(batch), 3, int(img_px), int(img_px), generator=g, device=device).to(dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    fn, info = (_compile(module, x, compile_mode) if compile else
                (module, {"mode_used": "eager", "compile_ok": None, "compile_s": None, "compile_error": ""}))
    try:
        r = measure(fn, x, warmup, iters, timer)
        err = ""
    except torch.cuda.OutOfMemoryError as e:
        r, err = {"latency_ms_median": float("nan")}, f"OOM: {str(e).splitlines()[0][:200]}"
    if compile and info["mode_used"] == "compile":
        torch._dynamo.reset()
    lat = r["latency_ms_median"]
    out = {**info, **r, "batch": int(batch), "error": err,
           "throughput_img_s": int(batch) * 1e3 / lat if lat and math.isfinite(lat) and lat > 0 else float("nan"),
           "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None}
    return out


@torch.no_grad()
def run_benchmark(entries: list, batches=BATCHES, *, img_px: int = 224, device="cpu", dtype=torch.bfloat16,
                  compile: bool = True, compile_mode: str = "default", warmup: int = 20, iters: int = 100,
                  seed: int = 0, timer=None, done: set | None = None, log=None) -> list:
    """Rows for every (entry, batch) not in `done` ({(name, batch)}): the entry's stored entries, bf16 memory and
    analytic GFLOPs (times its passes) and the measurement of time_module. Each module is moved to `device` and
    cast to `dtype` in place (entries are built for the benchmark; a flip-averaging entry may share its base)."""
    say = log or (lambda m: None)
    done = done or set()
    rows = []
    for e in entries:
        mem = param_memory(e.module)
        todo = [int(b) for b in batches if (e.name, int(b)) not in done]
        if not todo:
            continue
        mod = e.module.to(device=device, dtype=dtype).eval()
        gf = (e.analytic or {}).get("gflops")
        for b in todo:
            r = time_module(mod, b, img_px=img_px, device=device, dtype=dtype, compile=compile,
                            compile_mode=compile_mode, warmup=warmup, iters=iters, seed=seed, timer=timer)
            row = {"name": e.name, "kind": e.kind, "passes": int(e.passes), "params_analytic": e.params_analytic,
                   **mem, "gflops_analytic": None if gf is None else float(gf) * int(e.passes),
                   "gflops_source": (e.analytic or {}).get("source"), **r, "dtype": str(dtype).replace("torch.", ""),
                   "img_px": int(img_px), "seed": int(seed)}
            rows.append(row)
            say(f"A12 {e.name} batch {b} ({r['mode_used']}): {r['latency_ms_median']:.3f} ms, "
                f"{row['throughput_img_s']:.1f} img/s")
        del mod
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return rows


def systems_table(rows: list, ref: str = "original") -> list:
    """One row per model: parameters, bf16 memory, analytic GFLOPs, throughput at every batch, latency at batch 1,
    the largest peak memory, and throughput relative to `ref` at each batch."""
    names = list(dict.fromkeys(r["name"] for r in rows))
    by = {(r["name"], r["batch"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    out = []
    for n in names:
        rs = [r for r in rows if r["name"] == n]
        t = {"name": n, "kind": rs[0]["kind"], "passes": rs[0]["passes"], "params_analytic": rs[0]["params_analytic"],
             "entries_stored": rs[0]["entries_stored"], "param_mb_bf16": rs[0]["param_mb_bf16"],
             "gflops_analytic": rs[0]["gflops_analytic"]}
        peaks = [r["peak_memory_gb"] for r in rs if r.get("peak_memory_gb") is not None]
        t["peak_memory_gb_max"] = max(peaks) if peaks else None
        for b in batches:
            r = by.get((n, b))
            t[f"throughput_b{b}"] = None if r is None else r["throughput_img_s"]
            t[f"latency_ms_b{b}"] = None if r is None else r["latency_ms_median"]
            rr = by.get((ref, b))
            t[f"speed_vs_{ref}_b{b}"] = (r["throughput_img_s"] / rr["throughput_img_s"]
                                         if (r is not None and rr is not None and rr["throughput_img_s"]) else None)
        out.append(t)
    return out


# ----------------------------------------------------------------------------------------------
# GEMM shapes and the alignment effect
# ----------------------------------------------------------------------------------------------
_GEMM_ATTRS = {"MirrorAttention": ("We", "Wodd", "Oe", "Oo", "Wd"), "MirrorMLP": ("E1", "O1", "E2", "O2"),
               "DenseAttention": ("qkv_w", "proj_w"), "DenseMLP": ("W1", "W2"),
               "LowRankMLP": ("V1", "U1", "V2", "U2"), "LowRankAttention": ("V_qkv", "Uq", "Uk", "Uv", "Vo", "Uo")}


def gemm_shapes(module: nn.Module, n_tokens: int, batch: int) -> list:
    """Every GEMM of one forward: (where, N = output width, K = input width, M = batch x tokens) and its FLOPs
    (2 M N K). Even/odd and low-rank modules by their weight attributes, plain nn.Linear layers by their weight."""
    out = []
    M = int(batch) * int(n_tokens)
    for name, mod in module.named_modules():
        attrs = _GEMM_ATTRS.get(type(mod).__name__)
        if attrs is not None:
            for a in attrs:
                w = getattr(mod, a, None)
                if isinstance(w, torch.Tensor) and w.dim() == 2 and w.numel():
                    out.append({"where": f"{name}.{a}", "N": int(w.shape[0]), "K": int(w.shape[1]), "M": M})
        elif isinstance(mod, nn.Linear):
            out.append({"where": name, "N": int(mod.out_features), "K": int(mod.in_features), "M": M})
    for r in out:
        r["flops"] = 2 * r["M"] * r["N"] * r["K"]
        r["N_mult8"], r["K_mult8"] = r["N"] % 8 == 0, r["K"] % 8 == 0
    return out


@torch.no_grad()
def gemm_table(shapes: list, *, device, dtype=torch.bfloat16, warmup: int = 10, iters: int = 50, seed: int = 0,
               timer=None) -> list:
    """Each distinct (M, N, K) timed on its own: F.linear of random [M, K] by [N, K] operands in `dtype`; achieved
    TFLOP/s = 2 M N K / median time. `count` is how often the shape occurs in one forward."""
    device = torch.device(device)
    timer = timer or default_timer(device)
    groups: dict = {}
    for s in shapes:
        groups.setdefault((s["M"], s["N"], s["K"]), []).append(s["where"])
    rows = []
    g = torch.Generator(device=device).manual_seed(int(seed))
    for (M, N, K), where in sorted(groups.items()):
        x = torch.randn(M, K, generator=g, device=device).to(dtype)
        w = torch.randn(N, K, generator=g, device=device).to(dtype)
        r = measure(lambda z, w=w: F.linear(z, w), x, warmup, iters, timer)
        ms = r["latency_ms_median"]
        rows.append({"M": M, "N": N, "K": K, "count": len(where), "where": ",".join(where[:6]),
                     "N_mult8": N % 8 == 0, "K_mult8": K % 8 == 0, "latency_ms_median": ms,
                     "tflops": 2.0 * M * N * K / (ms * 1e-3) / 1e12 if ms > 0 else float("nan"),
                     "ms_per_forward": ms * len(where), "dtype": str(dtype).replace("torch.", "")})
    return rows


def align_effect(build_fn, aligns=(1, 8), *, batches=(64, 256), n_tokens: int, img_px: int = 224, device="cpu",
                 dtype=torch.bfloat16, compile: bool = True, warmup: int = 20, iters: int = 100, seed: int = 0,
                 timer=None, gemm: bool = True, log=None) -> dict:
    """The even/odd form built at each alignment (build_fn(align) -> module) and timed whole; with `gemm`, its GEMM
    shapes timed one by one at the largest batch. Padding changes no output (mirror.efficient)."""
    rows, gemms = [], []
    for a in aligns:
        mod = build_fn(int(a))
        e = BenchEntry(name=f"even_odd_align{a}", module=mod, kind="mirror_even_odd", meta={"align": int(a)})
        for r in run_benchmark([e], batches, img_px=img_px, device=device, dtype=dtype, compile=compile,
                               warmup=warmup, iters=iters, seed=seed, timer=timer, log=log):
            rows.append({**r, "align": int(a)})
        if gemm:
            for r in gemm_table(gemm_shapes(mod, n_tokens, max(batches)), device=device, dtype=dtype,
                                warmup=max(1, warmup // 2), iters=max(1, iters // 2), seed=seed, timer=timer):
                gemms.append({**r, "align": int(a)})
        del mod
    return {"rows": rows, "gemms": gemms}


# ----------------------------------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------------------------------
def group_categories(cats: dict) -> dict:
    """scripts/m2_profile categories {name: (ms, launches)} -> {group: ms} and shares: GEMM, butterflies and permutes
    (memory movement, elementwise, fused pointwise Triton kernels), attention, rest (norms, other)."""
    ms = {}
    for c, v in cats.items():
        g = GROUPS.get(c, "rest")
        ms[g] = ms.get(g, 0.0) + float(v[0] if isinstance(v, (tuple, list)) else v)
    total = sum(ms.values())
    return {"ms": ms, "share": {g: (v / total if total > 0 else float("nan")) for g, v in ms.items()},
            "total_ms": total}


@torch.no_grad()
def profile_summary(module: nn.Module, batch: int, *, img_px: int = 224, device="cpu", dtype=torch.bfloat16,
                    compile: bool = True, compile_mode: str = "default", iters: int = 10, warmup: int = 5,
                    seed: int = 0, top: int = 30) -> dict:
    """torch.profiler over `iters` forwards (after compile and warm-up), kernels sorted into the categories of
    scripts/m2_profile.py (classify_kernel; CUDA kernel time on a GPU, operator self time on the CPU) and grouped by
    group_categories. Returns {"groups", "categories", "top_kernels" (list of dicts), "time_source", ...}."""
    mp = _script("m2_profile")
    device = torch.device(device)
    mod = module.to(device=device, dtype=dtype).eval()          # in place, as run_benchmark
    g = torch.Generator(device=device).manual_seed(int(seed) + int(batch))
    x = torch.randn(int(batch), 3, int(img_px), int(img_px), generator=g, device=device).to(dtype)
    fn, info = (_compile(mod, x, compile_mode) if compile else (mod, {"mode_used": "eager"}))
    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn(x)
    kern, pinfo = mp.profile_forwards(fn, x, int(iters), device)
    cats, topk, _ = mp.kernel_tables(kern, int(iters), int(top))
    out = {"groups": group_categories(cats), "categories": {c: {"ms": v[0], "launches": v[1]} for c, v in cats.items()},
           "top_kernels": topk.to_dict(orient="records"), "batch": int(batch), "iters": int(iters), **info, **pinfo}
    if compile:
        torch._dynamo.reset()
    return out

#!/usr/bin/env python
"""A12b: fused even/odd MLP kernel, GPU correctness test and timing (FINAL_PHASE_PLAN.md v2, A12b; the bar of
PREREG_FINAL_PHASE.md: "throughput at batch 64 and at batch 256, each at least 1.1 times the original's, same device
and settings", bf16, compiled). The kernel, its CPU reference and the builders: mirror/final/systems/kernel.py.

    # W0 smoke step (SMOKE_EXTRA_CMD): GPU correctness only, synthetic weights at the DINOv2-B sizes
    python scripts/fp_a12b_kernel.py --correctness-only --out runs/final/_smoke/a12b
    # W1 timing: zero-shot Mirror-exact conversion from the canonical structure (timing does not depend on the
    # weight values), or a healed Mirror-exact run (as A12's manifest kind "run")
    python scripts/fp_a12b_kernel.py --model dinov2_vitb14_reg \
        --structure results/MUST-07/structure/dinov2_vitb14_reg_s0.npz --out results/A12b
    python scripts/fp_a12b_kernel.py --run-dir runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0 \
        --out results/A12b
    # CPU dry run: tiny random model, the reference path only, eager, no Triton
    python scripts/fp_a12b_kernel.py --dry-run --out tiny_out/A12b

Steps
 1. Correctness per block (always). Synthetic, exactly equivariant MLP weights at the model's sizes: random weights
    projected in float64 (mirror.surgery.project_mlp_two) with a random involution of the model's d+ that fixes the
    all-ones vector (mirror.m1.random_involution_fixing_ones) and random pairings with the model's pair count per
    block (mirror.matching.random_pairing); the even/odd MLP of mirror/efficient.py (align 8) is built from them and
    the fused module from that. Sizes come from --structure or --run-dir, else from the DINOv2-B values recorded in
    notes/paper/method_spec.md (Sections 1 and 2: d+ = 700, twin pairs per block 1168 ... 17). Per block:
      a. layout, float64: the fused computation (reference) equals the dense projected MLP in the A* basis
         (relative error <= 1e-10);
      b. float32: the fused module (the Triton kernel on a GPU) against the float64 computation on the same float32
         weights and inputs; it passes if its error is at most max(8 x the plain even/odd MLP's float32 error,
         64 x 2^-24) (errors are max |difference| / max |reference|; no TF32);
      c. bfloat16: the fused module in bf16 against its own float32 output; it passes if the error is at most
         max(2 x the plain even/odd MLP's bf16-against-fp32 error, 2^-8);
      d. equivariance: with the odd input coordinates negated, the even outputs must be unchanged and the odd
         outputs negated, exactly (bit for bit up to the sign of zero). Reported for the fused and the plain form;
         not a gate (A09 reports bitwise exactness as its own level).
    The factors in b and c are this script's choices for "within float32 / bf16 rounding" (the plan gives no number).
 2. Whole-model correctness, when a converted model is available (--run-dir, or --model with --structure, or the
    dry run): the even/odd model with fused MLPs against the plain even/odd model (mirror.final.audit.exactness.
    build_even_odd), with the criteria of 1b and 1c on the final feature, plus the flip deviation of the feature.
 3. Timing (not with --correctness-only): bf16 weights and inputs, torch.compile (default mode, static shapes; eager
    if compilation fails, recorded), --warmup untimed and --iters timed runs (default 20 and 100), CUDA events,
    median reported, at --batches (default 64 and 256):
      * MLP half-blocks only: the 12 blocks' x <- x + mlp(rms(x)) for the dense MLP (the projected weights stored
        densely, as the original's MLP costs), the plain even/odd MLP and the fused MLP; tokens per image as the
        model (261 for DINOv2-B at 224 px);
      * whole models (mirror.final.systems.bench.run_benchmark, as A12): the original; the plain even/odd form; the
        even/odd form with the fused MLP ("even_odd_fused", the configuration the bar is scored on: the converted
        model with all of its parameter saving); and, unless --no-hybrid, attention dense-stored in the A* basis
        with the fused MLP ("hybrid_fused": same function, MLP saving only; reported, not scored);
      * an estimate (not a measurement): original latency - dense MLP half-blocks + fused MLP half-blocks.
 4. The bar: even_odd_fused throughput / original throughput at batch 64 and at batch 256, both compiled, both at
    least 1.1, and the kernel passed step 1. Without whole-model timings the bar is "not scored"; the script never
    states a speed-up it did not measure.

Outputs (--out, default results/A12b; --correctness-only: results/A12b/correctness_only; dry run: tiny_out/A12b):
metrics.json, timing.csv (timing runs only), correctness.csv, summary.md, meta.json (mirror.final.prereg.
write_final_meta). metrics.json starts with the keys the final collector reads: speed_vs_original_b64 and
speed_vs_original_b256 (whole-model throughput of the converted model with the fused kernel over the original's,
same device and settings, bf16, compiled; null unless measured and scorable, never an MLP-only ratio), bar_met (true
only if both are at least 1.1, false if measured and below, null if not measured or not scorable), bar_status, and
scope ("whole_model"; "mlp_only" when only the MLP half-blocks were timed; "none" without timing; "dry_run").

The script calls mirror.final.prereg.require_prereg() before it touches the GPU, caps its GPU memory (--gpu-frac)
and records the other GPU processes it sees (timing must run alone on the GPU). Exit code 0 only if the correctness
test passed with the Triton kernel on a GPU (or with the reference on the CPU); a full run that is already complete
is skipped unless --force.
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.3

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import mirror as flip_images  # noqa: E402
from mirror.efficient import Basis, FastRMSNorm, build_dense_mlp, build_mirror_mlp  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.final.systems import bench as BE  # noqa: E402
from mirror.final.systems import kernel as K  # noqa: E402
from mirror.m1 import random_involution_fixing_ones  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402
from mirror.surgery import project_mlp_two  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A12b"
BAR = 1.1
BAR_BATCHES = (64, 256)
PRIMARY = "even_odd_fused"
EPS32 = 2.0 ** -24
FP32_FACTOR, FP32_FLOOR = 8.0, 64 * EPS32
BF16_FACTOR, BF16_FLOOR = 2.0, 2.0 ** -8
LAYOUT_TOL = 1e-10
# DINOv2-B/14 reg, fit seed 0, A*_12: notes/paper/method_spec.md Section 1 (d+ = 700) and Section 2 (twin pairs per
# block), from runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/structure.npz; T = 1 + 4 + 256 at 224 px (Section 0).
DEFAULT_SHAPES = {"model": "dinov2_vitb14_reg", "d": 768, "h": 3072, "d_plus": 700, "tokens": 261,
                  "pairs": [1168, 1116, 1013, 963, 878, 672, 554, 406, 155, 93, 19, 17],
                  "source": "notes/paper/method_spec.md Sections 0 to 2 (DINOv2-B reg, fit seed 0)"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="registry name of the original model (timed as 'original'; with "
                                                  "--structure the zero-shot Mirror-exact conversion is built from it)")
    p.add_argument("--structure", default=None, help="SpanStructure .npz (mirror.mirror_family); its A*_L and "
                                                      "pairings give the block shapes")
    p.add_argument("--run-dir", default=None, help="healed Mirror-exact run (replaces --model and --structure)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step of --run-dir (default: the last)")
    p.add_argument("--correctness-only", action="store_true", help="GPU correctness test only (the W0 smoke step)")
    p.add_argument("--batches", default="64,256")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--tokens", type=int, default=None, help="tokens per image for the MLP half-blocks (default: the "
                                                             "model's, else 261)")
    p.add_argument("--check-images", type=int, default=2, help="images (of T tokens) per block in step 1")
    p.add_argument("--model-images", type=int, default=4, help="images in the whole-model check")
    p.add_argument("--no-hybrid", action="store_true", help="skip the dense-attention plus fused-MLP model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="rerun a complete run")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random model, reference path, eager, 2 runs")
    a = p.parse_args(argv)
    if a.run_dir and (a.structure or a.model):
        p.error("--run-dir replaces --model and --structure")
    if a.dry_run:
        a.batches, a.warmup, a.iters, a.no_compile, a.img_size = "2,4", 1, 2, True, 56
    a.batch_list = [int(b) for b in a.batches.split(",") if b.strip()]
    a.mode = "dry_run" if a.dry_run else ("correctness_only" if a.correctness_only else "full")
    return a


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| / max |b| in float64 (the plain maximum difference if b is all zero)."""
    a, b = a.double(), b.double()
    if not b.numel():
        return 0.0
    diff = float((a - b).abs().max())
    den = float(b.abs().max())
    return diff / den if den > 0 else diff


# ----------------------------------------------------------------------------------------------
# Sources: the shapes, and the converted model when there is one
# ----------------------------------------------------------------------------------------------
def n_pairs(perm) -> int:
    p = np.asarray(perm)
    return int((p != np.arange(len(p))).sum() // 2)


def shapes_of(st, adapter=None, label="") -> dict:
    return {"model": label, "d": int(st.d), "h": int(len(st.perms[0])), "d_plus": int(st.d_plus),
            "tokens": int(adapter.n_tokens) if adapter is not None else DEFAULT_SHAPES["tokens"],
            "pairs": [n_pairs(p) for p in st.perms], "source": label}


def _a12_script():
    """scripts/fp_a12_bench.py as a module (its load_run reads healed runs as A12 does)."""
    import importlib.util

    name = "_fp_a12b_fp_a12_bench"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "fp_a12_bench.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def tiny_source() -> dict:
    """A tiny random timm ViT converted with build_vexact (as scripts/fp_a12_bench.py dry_entries)."""
    from mirror.adapters import TimmTinyAdapter
    from mirror.attention import attn_dims
    from mirror.final.audit.exactness import AuditStructure
    from mirror.m1 import build_vexact, random_head_structure

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=3)
    perms = [random_pairing(h, h // 3 + b, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=20 + b) for b in range(L)]
    model, info = build_vexact(ad, A, list(range(L)), perms, maps, even_head=True)
    st = AuditStructure(A=A, perms=perms, maps=info["maps_after"])
    label = "tiny random timm ViT (CPU dry run)"
    return {"kind": "tiny", "label": label, "adapter": ad, "model": model.eval(), "st": st,
            "shapes": shapes_of(st, ad, label)}


def load_source(a, device) -> dict:
    """{"kind", "label", "adapter", "model" (converted, float32) or None, "st" (AuditStructure) or None, "shapes"}."""
    if a.dry_run:
        return tiny_source()
    from mirror.final.audit.exactness import AuditStructure

    if a.run_dir:
        ad, model, info = _a12_script().load_run(a.run_dir, a.step, device)
        st = AuditStructure.from_info(info)
        return {"kind": "run", "label": str(a.run_dir), "adapter": ad, "model": model.float().eval(), "st": st,
                "shapes": shapes_of(st, ad, str(a.run_dir))}
    if a.structure:
        from mirror.mirror_family import load_span_structure

        span = load_span_structure(_resolve(a.structure))
        L = span.depth
        parts = span.exact_parts("mirror_exact", L)
        st0 = AuditStructure(A=parts["A"], perms=parts["perms"], maps=[None] * L, reg_perm=parts["reg_perm"])
        label = f"{a.structure} ({span.model}, fit seed {span.fit_seed})"
        if not a.model:
            return {"kind": "structure", "label": label, "adapter": None, "model": None, "st": None,
                    "shapes": shapes_of(st0, None, label)}
        if span.model and str(span.model) != str(a.model):
            log(f"{EXPERIMENT}: warning: --structure names the model {span.model!r}, --model is {a.model!r}")
        from mirror.adapters_timm import build_model_adapter
        from mirror.m1 import build_vexact

        ad = build_model_adapter(a.model, device, img_size=a.img_size)
        model, binfo = build_vexact(ad, parts["A"], list(range(L)), parts["perms"], parts["heads"],
                                    reg_perm=parts["reg_perm"], even_head=True)
        st = AuditStructure(A=parts["A"], perms=parts["perms"], maps=binfo["maps_after"], reg_perm=parts["reg_perm"])
        return {"kind": "zero_shot", "label": label + ", zero-shot Mirror-exact", "adapter": ad,
                "model": model.float().eval(), "st": st, "shapes": shapes_of(st, ad, label)}
    if a.model:
        if a.model != DEFAULT_SHAPES["model"]:
            raise ValueError(f"give --structure for {a.model} (the default block shapes are DINOv2-B's)")
        from mirror.adapters_timm import build_model_adapter

        ad = build_model_adapter(a.model, device, img_size=a.img_size)
        return {"kind": "original_only", "label": a.model, "adapter": ad, "model": None, "st": None,
                "shapes": dict(DEFAULT_SHAPES)}
    return {"kind": "default_shapes", "label": DEFAULT_SHAPES["source"], "adapter": None, "model": None, "st": None,
            "shapes": dict(DEFAULT_SHAPES)}


# ----------------------------------------------------------------------------------------------
# Step 1: per-block correctness on synthetic equivariant weights
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def synthetic_blocks(shapes: dict, device, seed: int) -> dict:
    """{"A", "basis", "Q", "blocks": [{"W" (projected dense weights, float64), "mm" (MirrorMLP, float64, align 8),
    "perm"}]} on `device`."""
    d, h, dp = int(shapes["d"]), int(shapes["h"]), int(shapes["d_plus"])
    A = random_involution_fixing_ones(d, dp, seed=seed + 1)
    basis = Basis.of(A)
    blocks = []
    for b, npair in enumerate(shapes["pairs"]):
        perm = random_pairing(h, int(npair), seed=seed + 100 + b)
        g = torch.Generator(device=device).manual_seed(seed + 200 + b)
        kw = dict(generator=g, device=device, dtype=torch.float64)
        W = project_mlp_two(torch.randn(h, d, **kw) / math.sqrt(d), 0.1 * torch.randn(h, **kw),
                            torch.randn(d, h, **kw) / math.sqrt(h), 0.1 * torch.randn(d, **kw), A, A, perm)
        mm = build_mirror_mlp(W["W1"], W["b1"], W["W2"], W["b2"], nn.GELU(), basis, basis, A, A, perm, align=8,
                              tol=1e-9)
        blocks.append({"W": W, "mm": mm, "perm": perm})
    return {"A": A, "basis": basis, "Q": torch.as_tensor(basis.Q, dtype=torch.float64, device=device),
            "blocks": blocks}


def dense_forward(x: torch.Tensor, W: dict) -> torch.Tensor:
    return F.linear(F.gelu(F.linear(x, W["W1"], W["b1"])), W["W2"], W["b2"])


@torch.no_grad()
def equivariance(mod: nn.Module, x: torch.Tensor, s_in: torch.Tensor, s_out: torch.Tensor) -> dict:
    """mod(x with odd coordinates negated) against mod(x) with odd outputs negated: exact equality (torch.equal, so
    +0 equals -0) and the largest difference relative to max |mod(x)|."""
    y = mod(x)
    ym = mod(x * s_in)
    want = y * s_out
    return {"exact": bool(torch.equal(ym, want)), "n_differ": int((ym != want).sum()), "rel": rel(ym, want)}


@torch.no_grad()
def check_block(b: int, blk: dict, Q: torch.Tensor, dp: int, backend: str, device, M: int, seed: int) -> dict:
    mm, W = blk["mm"], blk["W"]
    d = int(Q.shape[0])
    g = torch.Generator(device=device).manual_seed(int(seed))
    x32 = torch.randn(int(M), d, generator=g, device=device, dtype=torch.float32)
    x64 = x32.double()
    # a. layout (float64): the fused computation equals the dense projected MLP in the A* basis
    y64 = K.build_fused_mlp(mm, backend="reference")(x64)
    err_layout = rel(y64, dense_forward(x64 @ Q.T, W) @ Q)
    fused = K.build_fused_mlp(mm, backend=backend)
    # b. float32 against float64 on the same float32 weights and inputs
    truth = K.build_fused_mlp(copy.deepcopy(mm).float().double(), backend="reference")(x64)
    f32, p32 = copy.deepcopy(fused).float(), copy.deepcopy(mm).float()
    y_f32, y_p32 = f32(x32), p32(x32)
    err_f32, err_p32 = rel(y_f32, truth), rel(y_p32, truth)
    tol32 = max(FP32_FACTOR * err_p32, FP32_FLOOR)
    # c. bfloat16 against the same module's float32 output
    f16, p16 = copy.deepcopy(fused).to(torch.bfloat16), copy.deepcopy(mm).to(torch.bfloat16)
    x16 = x32.to(torch.bfloat16)
    err_f16, err_p16 = rel(f16(x16), y_f32), rel(p16(x16), y_p32)
    tol16 = max(BF16_FACTOR * err_p16, BF16_FLOOR)
    # d. equivariance (sign flip of the odd coordinates, input and output in the same basis)
    eq = {}
    for name, mod, x in (("fused_fp32", f32, x32), ("fused_bf16", f16, x16), ("plain_fp32", p32, x32),
                         ("plain_bf16", p16, x16)):
        s = K.sign_vector(d, dp, dtype=x.dtype, device=device)
        eq[name] = equivariance(mod, x, s, s)
    lay = fused.layout
    row = {"block": int(b), "n_pairs": lay.get("n_pairs"), "n_self": lay.get("n_self"), "npad": lay["npad"],
           "nspad": lay["nspad"], "ke": lay["ke"], "ko": lay["ko"], "KO": lay["KO"], "tokens": int(M),
           "backend": fused.backend, "err_layout_f64": err_layout, "pass_layout": err_layout <= LAYOUT_TOL,
           "err_fp32_fused": err_f32, "err_fp32_plain": err_p32, "tol_fp32": tol32, "pass_fp32": err_f32 <= tol32,
           "err_bf16_fused": err_f16, "err_bf16_plain": err_p16, "tol_bf16": tol16, "pass_bf16": err_f16 <= tol16}
    for name, e in eq.items():
        row[f"equiv_exact_{name}"] = e["exact"]
        row[f"equiv_rel_{name}"] = e["rel"]
    row["pass"] = bool(row["pass_layout"] and row["pass_fp32"] and row["pass_bf16"])
    return row


# ----------------------------------------------------------------------------------------------
# Step 2: whole-model correctness
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_forms(src: dict, backend: str, hybrid: bool) -> dict:
    """The plain even/odd model, the same with fused MLPs, and (optionally) dense attention with fused MLPs."""
    from mirror.final.audit.exactness import build_even_odd

    fv = build_even_odd(src["model"], src["st"], align=8)
    out = {"even_odd": fv, "even_odd_fused": K.fuse_fast_vit(fv, backend=backend)}
    if hybrid:
        out["hybrid_fused"] = K.fused_even_odd(src["model"], src["st"], align=8, backend=backend, attention="dense")
    return out


@torch.no_grad()
def check_model(forms: dict, device, a) -> dict:
    g = torch.Generator(device=device).manual_seed(a.seed + 7)
    x = torch.randn(int(a.model_images), 3, a.img_size, a.img_size, generator=g, device=device)
    plain = forms["even_odd"]
    y64 = copy.deepcopy(plain).double()(x.double())
    y_p32 = plain(x)
    err_p32 = rel(y_p32, y64)
    p16 = copy.deepcopy(plain).to(torch.bfloat16)
    err_p16 = rel(p16(x.to(torch.bfloat16)), y_p32)
    tol32, tol16 = max(FP32_FACTOR * err_p32, FP32_FLOOR), max(BF16_FACTOR * err_p16, BF16_FLOOR)
    xm = flip_images(x)
    out = {"images": int(a.model_images), "err_fp32_plain": err_p32, "err_bf16_plain": err_p16, "tol_fp32": tol32,
           "tol_bf16": tol16, "flip_rel_fp32_even_odd": rel(plain(xm), y_p32),
           "flip_rel_bf16_even_odd": rel(p16(xm.to(torch.bfloat16)), p16(x.to(torch.bfloat16))), "forms": {}}
    ok = True
    for name, mod in forms.items():
        if name == "even_odd":
            continue
        y32 = mod(x)
        m16 = copy.deepcopy(mod).to(torch.bfloat16)
        y16 = m16(x.to(torch.bfloat16))
        e32, e16 = rel(y32, y64), rel(y16, y32)
        r = {"err_fp32": e32, "pass_fp32": e32 <= tol32, "err_bf16": e16, "pass_bf16": e16 <= tol16,
             "flip_rel_fp32": rel(mod(xm), y32), "flip_rel_bf16": rel(m16(xm.to(torch.bfloat16)), y16)}
        r["pass"] = bool(r["pass_fp32"] and r["pass_bf16"])
        ok = ok and r["pass"]
        out["forms"][name] = r
        del m16
    out["pass"] = bool(ok)
    return out


# ----------------------------------------------------------------------------------------------
# Step 3: timing
# ----------------------------------------------------------------------------------------------
class MLPStack(nn.Module):
    """The MLP halves of the blocks, x <- x + mlp(rms(x)) for every block (FastBlock's MLP step without attention)."""

    def __init__(self, mlps: list, d: int, eps: float = 1e-6):
        super().__init__()
        self.norms = nn.ModuleList(FastRMSNorm(d, eps) for _ in mlps)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x):
        for n, m in zip(self.norms, self.mlps):
            x = x + m(n(x))
        return x


def compile_or_eager(module: nn.Module, x: torch.Tensor, compile: bool, mode: str):
    info = {"mode_used": "eager", "compile_ok": None, "compile_s": None, "compile_error": ""}
    if not compile:
        return module, info
    torch._dynamo.reset()
    try:
        cm = torch.compile(module, mode=mode, dynamic=False)
        t0 = time.time()
        with torch.inference_mode():
            cm(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        info.update(mode_used="compile", compile_ok=True, compile_s=round(time.time() - t0, 1))
        return cm, info
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:300] if str(e) else ""
        info.update(mode_used="eager (compile failed)", compile_ok=False, compile_error=f"{type(e).__name__}: {msg}")
        torch._dynamo.reset()
        return module, info


@torch.no_grad()
def time_tokens(module: nn.Module, batch: int, tokens: int, d: int, *, device, dtype, compile: bool,
                compile_mode: str, warmup: int, iters: int, seed: int, timer) -> dict:
    """One (module, batch) measurement on randn tokens [batch, tokens, d] (generator seeded with seed + batch)."""
    g = torch.Generator(device=device).manual_seed(int(seed) + int(batch))
    x = torch.randn(int(batch), int(tokens), int(d), generator=g, device=device).to(dtype)
    mod = module.to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    fn, info = compile_or_eager(mod, x, compile, compile_mode)
    try:
        r, err = BE.measure(fn, x, warmup, iters, timer), ""
    except torch.cuda.OutOfMemoryError as e:
        r, err = {"latency_ms_median": float("nan")}, f"OOM: {str(e).splitlines()[0][:200]}"
    if info["mode_used"] == "compile":
        torch._dynamo.reset()
    lat = r["latency_ms_median"]
    return {**info, **r, "batch": int(batch), "tokens": int(tokens), "error": err,
            "throughput_img_s": int(batch) * 1e3 / lat if _finite(lat) and lat > 0 else float("nan"),
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None}


def timing(a, device, syn: dict, shapes: dict, tokens: int, forms: dict | None, adapter, backend: str) -> list:
    timer = BE.default_timer(device)
    d = int(shapes["d"])
    common = dict(device=device, dtype=torch.bfloat16, compile=not a.no_compile, compile_mode=a.compile_mode,
                  warmup=a.warmup, iters=a.iters, seed=a.seed, timer=timer)
    rows = []
    builders = {"dense": lambda: [build_dense_mlp(b["W"]["W1"], b["W"]["b1"], b["W"]["W2"], b["W"]["b2"], nn.GELU())
                                  for b in syn["blocks"]],
                "even_odd": lambda: [copy.deepcopy(b["mm"]) for b in syn["blocks"]],
                "fused": lambda: [K.build_fused_mlp(b["mm"], backend=backend) for b in syn["blocks"]]}
    for form, make in builders.items():
        stack = MLPStack(make(), d)
        for B in a.batch_list:
            r = time_tokens(stack, B, tokens, d, **common)
            rows.append({"level": "mlp_stack", "form": form, "backend": backend if form == "fused" else "torch", **r})
            log(f"{EXPERIMENT} MLP half-blocks {form} batch {B} ({r['mode_used']}): {r['latency_ms_median']:.3f} ms")
        del stack
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if forms is not None and adapter is not None:
        entries = [BE.BenchEntry("original", BE.FeatureModule(adapter, adapter.model), "original")]
        entries += [BE.BenchEntry(name, mod, f"mirror_{name}") for name, mod in forms.items()]
        for r in BE.run_benchmark(entries, a.batch_list, img_px=a.img_size, device=device, dtype=torch.bfloat16,
                                  compile=not a.no_compile, compile_mode=a.compile_mode, warmup=a.warmup,
                                  iters=a.iters, seed=a.seed, timer=timer, log=log):
            rows.append({"level": "model", "form": r["name"],
                         "backend": backend if "fused" in r["name"] else "torch", **r})
    elif adapter is not None:                                   # the original alone (for the estimate)
        e = BE.BenchEntry("original", BE.FeatureModule(adapter, adapter.model), "original")
        for r in BE.run_benchmark([e], a.batch_list, img_px=a.img_size, device=device, dtype=torch.bfloat16,
                                  compile=not a.no_compile, compile_mode=a.compile_mode, warmup=a.warmup,
                                  iters=a.iters, seed=a.seed, timer=timer, log=log):
            rows.append({"level": "model", "form": "original", "backend": "torch", **r})
    return rows


def _row(rows, level, form, b):
    return next((r for r in rows if r["level"] == level and r["form"] == form and int(r["batch"]) == int(b)), None)


def _thr(r):
    return float(r["throughput_img_s"]) if r is not None and _finite(r.get("throughput_img_s")) else None


def ratios(rows: list, batches) -> dict:
    out = {"mlp_stack_vs_dense": {}, "model_vs_original": {}, "estimate": {}}
    for b in batches:
        dense, orig = _thr(_row(rows, "mlp_stack", "dense", b)), _thr(_row(rows, "model", "original", b))
        for f in ("even_odd", "fused"):
            t = _thr(_row(rows, "mlp_stack", f, b))
            if t is not None and dense:
                out["mlp_stack_vs_dense"].setdefault(f, {})[int(b)] = t / dense
        for f in ("even_odd", "even_odd_fused", "hybrid_fused"):
            t = _thr(_row(rows, "model", f, b))
            if t is not None and orig:
                out["model_vs_original"].setdefault(f, {})[int(b)] = t / orig
        ro, rd, rf = (_row(rows, "model", "original", b), _row(rows, "mlp_stack", "dense", b),
                      _row(rows, "mlp_stack", "fused", b))
        if ro is not None and rd is not None and rf is not None and all(
                _finite(r["latency_ms_median"]) for r in (ro, rd, rf)):
            est = ro["latency_ms_median"] - rd["latency_ms_median"] + rf["latency_ms_median"]
            out["estimate"][int(b)] = {"original_ms": ro["latency_ms_median"], "estimated_ms": est,
                                       "ratio": ro["latency_ms_median"] / est if est > 0 else None}
    return out


def score_bar(rows: list, corr_pass: bool, backend: str, a) -> dict:
    res = {"threshold": BAR, "batches": list(BAR_BATCHES), "configuration": PRIMARY, "compared_with": "original",
           "met": None, "ratios": {}, "status": ""}
    if a.dry_run:
        res["status"] = "not scored: CPU dry run (no measurement)"
    elif a.correctness_only:
        res["status"] = "not scored: correctness test only (no timing)"
    elif backend != "triton":
        res["status"] = "not scored: the Triton kernel did not run here"
    elif not corr_pass:
        res["status"] = "not scored: the kernel failed the correctness test"
    else:
        missing, eager = [], []
        for b in BAR_BATCHES:
            o, f = _row(rows, "model", "original", b), _row(rows, "model", PRIMARY, b)
            if _thr(o) is None or _thr(f) is None:
                missing.append(b)
                continue
            if o["mode_used"] != "compile" or f["mode_used"] != "compile":
                eager.append(b)                     # an eager ratio stays in metrics["ratios"], never here
                continue
            res["ratios"][int(b)] = _thr(f) / _thr(o)
        if missing:
            res["status"] = (f"not scored: no whole-model timing at batch {', '.join(map(str, missing))} "
                             "(give --run-dir, or --model with --structure, and batches 64 and 256)")
        elif eager:
            res["status"] = f"not scored: not compiled at batch {', '.join(map(str, eager))} (the bar needs compiled)"
        else:
            res["met"] = all(r >= BAR for r in res["ratios"].values())
            res["status"] = "met" if res["met"] else "not met"
    return res


def headline_keys(rows: list, bar: dict, a) -> dict:
    """The keys the final collector reads from metrics.json. speed_vs_original_b64 / _b256: whole-model throughput
    of the converted model with the fused kernel over the original's (same device and settings, bf16, compiled),
    filled only when the bar could be scored at that batch, so never from MLP-only, CPU or eager timings. bar_met:
    true or false when scored at both batches, else null. scope: "whole_model", "mlp_only" (the MLP half-blocks
    were timed but not the whole model), "none" (no timing) or "dry_run"."""
    if a.dry_run:
        scope = "dry_run"
    elif not rows:
        scope = "none"
    elif any(r["level"] == "model" and r["form"] == PRIMARY for r in rows):
        scope = "whole_model"
    else:
        scope = "mlp_only"
    ratios = bar.get("ratios") or {}
    return {"speed_vs_original_b64": ratios.get(64), "speed_vs_original_b256": ratios.get(256),
            "bar_met": bar.get("met"), "bar_status": bar.get("status"), "scope": scope}


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------
def gpu_processes() -> list:
    """Other compute processes on the GPU (nvidia-smi), this process excluded."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi unavailable ({type(e).__name__})"]
    me = str(os.getpid())
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and ln.split(",")[0].strip() != me]


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{nd}f}"


def _e(x):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.2e}"


def write_summary(out: Path, m: dict, a) -> Path:
    bar, corr, rat = m["bar"], m["correctness"], m.get("ratios") or {}
    bs = a.batch_list
    verdict = {"met": "MET", "not met": "NOT MET"}.get(bar["status"], "NOT SCORED")
    lines = ["# A12b fused even/odd MLP kernel", ""]
    head = f"**Pre-registered bar (A12b, at least {BAR}x the original's throughput at batch 64 and 256): {verdict}"
    if verdict == "NOT SCORED":
        head += f" ({bar['status'].replace('not scored: ', '')})"
    lines.append(head + ".**")
    if bar["ratios"]:
        lines.append("The converted model with the fused MLP kernel ran at "
                     + " and ".join(f"{_f(v, 3)}x (batch {b})" for b, v in sorted(bar["ratios"].items()))
                     + f" the original's throughput (bf16, torch.compile, median of {a.iters} runs after "
                       f"{a.warmup} warm-up runs).")
    st = m["triton"]
    tri = f"Triton {st.get('triton_version')}" if st.get("ok") else st.get("reason")
    gpu = f" ({m['gpu']})" if m.get("gpu") else ""
    lines += ["", f"Backend: {m['backend']} ({tri}); device {m['device']}{gpu}; mode {m['mode']}.",
              f"Correctness: {'PASS' if corr['pass'] else 'FAIL'} ({sum(r['pass'] for r in corr['blocks'])} of "
              f"{len(corr['blocks'])} blocks; float32 and bf16 within the tolerances below; float64 layout check)."]
    eqf = [r.get("equiv_exact_fused_bf16") for r in corr["blocks"]]
    lines.append(f"Exact mirror equivariance of the fused MLP (sign flip of the odd coordinates): float32 in "
                 f"{sum(bool(r.get('equiv_exact_fused_fp32')) for r in corr['blocks'])} of {len(eqf)} blocks, bf16 in "
                 f"{sum(bool(v) for v in eqf)} of {len(eqf)} (plain even/odd MLP: float32 "
                 f"{sum(bool(r.get('equiv_exact_plain_fp32')) for r in corr['blocks'])}, bf16 "
                 f"{sum(bool(r.get('equiv_exact_plain_bf16')) for r in corr['blocks'])}).")
    rows = m.get("timing_rows") or []
    if rows and m["device"] == "cpu":
        tiny = ", tiny model" if m["mode"] == "dry_run" else ""
        lines += ["", f"The timings below come from the CPU (eager, a few runs{tiny}): they check the plumbing and "
                      "are not measurements."]
    if any(r["level"] == "model" for r in rows):
        lines += ["", "## Whole model (images per second)", "",
                  "| Model | " + " | ".join(f"b{b}" for b in bs) + " | " + " | ".join(f"vs original b{b}" for b in bs)
                  + " | compiled |", "|---|" + "---|" * (2 * len(bs) + 1)]
        for form in ("original", "even_odd", "even_odd_fused", "hybrid_fused"):
            rs = [_row(rows, "model", form, b) for b in bs]
            if not any(rs):
                continue
            vs = [(rat.get("model_vs_original", {}).get(form, {}) or {}).get(int(b)) if form != "original" else 1.0
                  for b in bs]
            comp = ", ".join(sorted({r["mode_used"] for r in rs if r}))
            lines.append(f"| {form} | " + " | ".join(_f(_thr(r), 1) for r in rs) + " | "
                         + " | ".join(_f(v, 3) for v in vs) + f" | {comp} |")
        lines += ["", "even_odd_fused: the even/odd form with the fused MLP (all of the parameter saving; the bar's "
                      "configuration). hybrid_fused: attention dense-stored in the A* basis with the fused MLP (same "
                      "function, MLP saving only; reported, not scored)."]
    if any(r["level"] == "mlp_stack" for r in rows):
        lines += ["", f"## MLP half-blocks only (x <- x + mlp(rms(x)) for {len(m['shapes']['pairs'])} blocks, "
                      f"{m['tokens']} tokens per image)", "",
                  "| Form | " + " | ".join(f"ms b{b}" for b in bs) + " | " + " | ".join(f"vs dense b{b}" for b in bs)
                  + " |", "|---|" + "---|" * (2 * len(bs))]
        for form in ("dense", "even_odd", "fused"):
            rs = [_row(rows, "mlp_stack", form, b) for b in bs]
            vs = [1.0 if form == "dense" else (rat.get("mlp_stack_vs_dense", {}).get(form, {}) or {}).get(int(b))
                  for b in bs]
            lines.append(f"| {form} | " + " | ".join(_f(r["latency_ms_median"] if r else None, 3) for r in rs) + " | "
                         + " | ".join(_f(v, 3) for v in vs) + " |")
    if rat.get("estimate"):
        lines += ["", "Estimate, not a measurement (original latency minus the dense MLP half-blocks plus the fused "
                      "ones): " + "; ".join(f"batch {b}: {_f(v['ratio'], 3)}x" for b, v in sorted(
                          rat["estimate"].items()))]
    lines += ["", "## Correctness per block (errors: max |difference| / max |reference|)", "",
              "| Block | pairs | self | fp32 fused | fp32 plain | fp32 tol | bf16 fused | bf16 plain | bf16 tol | "
              "layout f64 | pass |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in corr["blocks"]:
        lines.append(f"| {r['block']} | {r['n_pairs']} | {r['n_self']} | {_e(r['err_fp32_fused'])} | "
                     f"{_e(r['err_fp32_plain'])} | {_e(r['tol_fp32'])} | {_e(r['err_bf16_fused'])} | "
                     f"{_e(r['err_bf16_plain'])} | {_e(r['tol_bf16'])} | {_e(r['err_layout_f64'])} | "
                     f"{'yes' if r['pass'] else 'NO'} |")
    errs = [f"block {r['block']}: {r['error']}" for r in corr["blocks"] if r.get("error")]
    if errs:
        lines += ["", "Errors: " + "; ".join(errs[:3]) + (f" (and {len(errs) - 3} more)" if len(errs) > 3 else "")]
    mc = corr.get("model")
    if mc and mc.get("error"):
        lines += ["", f"Whole-model check failed to run: {mc['error']}"]
    elif mc:
        lines += ["", "Whole model (final feature, " + str(mc["images"]) + " random images): "
                  + "; ".join(f"{k}: fp32 {_e(v['err_fp32'])} (tol {_e(mc['tol_fp32'])}), bf16 {_e(v['err_bf16'])} "
                              f"(tol {_e(mc['tol_bf16'])}), {'pass' if v['pass'] else 'FAIL'}"
                              for k, v in mc["forms"].items())]
    lines += ["", "## What is fused", "",
              "One Triton kernel per MLP computes both fc1 GEMMs (even part, K = ke; odd part, K = d - ko), the fc1 "
              "bias, the butterfly u +/- v, GELU and the butterfly ga +/- gb, and writes the fc2 operands once. fc2 "
              "runs as two torch GEMMs (cuBLAS); the output concatenation is left to torch.compile (fused with the "
              "residual addition). Attention is unchanged.", "", "## Caveats", ""]
    others = m.get("gpu_processes_before_timing") or []
    lines.append("- Other GPU processes at the start of timing: " + ("; ".join(others) if others else "none seen")
                 + (" (timings may be contended)" if others else "") + ".")
    lines.append("- Inputs are random images and tokens; timing does not depend on the weight values. Tile "
                 "configurations of the kernel were chosen without measurement; the autotuner picked one per shape "
                 "(metrics.json, autotune).")
    lines += ["", "Files: metrics.json, correctness.csv, timing.csv (timing runs), meta.json."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.dry_run else "cpu"))
    default = {"dry_run": REPO_ROOT / "tiny_out" / EXPERIMENT, "correctness_only": REPO_ROOT / "results" / EXPERIMENT /
               "correctness_only", "full": REPO_ROOT / "results" / EXPERIMENT}[a.mode]
    out = Path(a.out) if a.out else default
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.json"
    if a.mode == "full" and mpath.exists() and not a.force:
        old = json.loads(mpath.read_text(encoding="utf-8"))
        if old.get("complete") and old.get("mode") == "full":
            log(f"{EXPERIMENT}: {out} is complete; --force to rerun")
            return {**old, "ok": bool(old.get("ok")), "skipped": True}
    prereg = require_prereg(allow_missing=True if a.dry_run else None)     # before the GPU is touched
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac, device)    # cap before anything is allocated
        torch.backends.cuda.matmul.allow_tf32 = False                    # float32 checks without TF32
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    status = K.triton_status() if device.type == "cuda" else {
        "ok": False, "reason": "CPU run (reference path only)", "cuda": torch.cuda.is_available(),
        "triton_version": None, "erf": None}
    backend = "triton" if (device.type == "cuda" and status["ok"]) else "reference"
    log(f"{EXPERIMENT}: mode {a.mode}, device {device}, backend {backend}"
        + ("" if status["ok"] else f" ({status['reason']})") + f", out {out}")

    src = load_source(a, device)
    shapes = src["shapes"]
    tokens = int(a.tokens or shapes["tokens"])
    log(f"{EXPERIMENT}: shapes from {src['label']}: d {shapes['d']}, h {shapes['h']}, d+ {shapes['d_plus']}, "
        f"pairs {shapes['pairs']}, {tokens} tokens per image")

    # 1. per-block correctness
    syn = synthetic_blocks(shapes, device, a.seed)
    M = int(a.check_images) * tokens
    blocks = []
    for b, blk in enumerate(syn["blocks"]):
        try:
            r = check_block(b, blk, syn["Q"], int(shapes["d_plus"]), backend, device, M, a.seed + 300 + b)
        except Exception as e:  # noqa: BLE001  (a kernel error is a failed check, reported)
            npair = int(shapes["pairs"][b])
            r = {"block": b, "n_pairs": npair, "n_self": int(shapes["h"]) - 2 * npair, "pass": False,
                 "error": f"{type(e).__name__}: {str(e)[:400]}"}
            for k in ("err_fp32_fused", "err_fp32_plain", "tol_fp32", "err_bf16_fused", "err_bf16_plain", "tol_bf16",
                      "err_layout_f64"):
                r.setdefault(k, None)
        blocks.append(r)
        detail = r["error"] if "error" in r else (
            f"(fp32 {_e(r['err_fp32_fused'])} <= {_e(r['tol_fp32'])}, bf16 {_e(r['err_bf16_fused'])} <= "
            f"{_e(r['tol_bf16'])}, layout {_e(r['err_layout_f64'])}, exact equivariance fused fp32/bf16 "
            f"{r.get('equiv_exact_fused_fp32')}/{r.get('equiv_exact_fused_bf16')})")
        log(f"{EXPERIMENT} block {b}: {'pass' if r['pass'] else 'FAIL'} {detail}")
    corr = {"pass": all(r["pass"] for r in blocks), "backend": backend, "blocks": blocks, "tokens_per_block": M,
            "criteria": {"layout_f64": f"<= {LAYOUT_TOL:g}",
                         "fp32": f"<= max({FP32_FACTOR:g} x plain even/odd float32 error, {FP32_FLOOR:.3g}), against "
                                 "float64 on the same float32 weights",
                         "bf16": f"<= max({BF16_FACTOR:g} x plain even/odd bf16-vs-fp32 error, {BF16_FLOOR:.3g}), "
                                 "against the module's own float32 output",
                         "equivariance": "reported, not a gate"}}

    # 2. whole-model correctness
    forms = None
    if src["model"] is not None and src["st"] is not None:
        try:
            forms = build_forms(src, backend, hybrid=not a.no_hybrid)
            corr["model"] = check_model(forms, device, a)
            corr["pass"] = bool(corr["pass"] and corr["model"]["pass"])
            log(f"{EXPERIMENT} whole model: {'pass' if corr['model']['pass'] else 'FAIL'} "
                + json.dumps(_jsonable({k: (v['err_fp32'], v['err_bf16']) for k, v in corr['model']['forms'].items()})))
        except Exception as e:  # noqa: BLE001
            corr["model"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            corr["pass"] = False
            forms = None
            log(f"{EXPERIMENT} whole model check failed: {corr['model']['error']}")

    # 3. timing
    rows, procs_before, procs_after = [], [], []
    if a.mode != "correctness_only":
        if device.type == "cuda":
            procs_before = gpu_processes()
            if procs_before:
                log(f"{EXPERIMENT}: other GPU processes during timing: {procs_before}")
        rows = timing(a, device, syn, shapes, tokens, forms, src["adapter"], backend)
        if device.type == "cuda":
            procs_after = gpu_processes()
    rat = ratios(rows, a.batch_list) if rows else {}
    bar = score_bar(rows, corr["pass"], backend, a)
    ok = bool(corr["pass"] and (backend == "triton" or device.type == "cpu"))

    import pandas as pd

    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in blocks]).to_csv(
        out / "correctness.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(out / "timing.csv", index=False)
    raw = {"experiment": EXPERIMENT, **headline_keys(rows, bar, a), "mode": a.mode, "ok": ok, "complete": True,
           "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": torch.__version__, "triton": status, "backend": backend, "source": src["label"],
               "source_kind": src["kind"], "shapes": shapes, "tokens": tokens, "correctness": corr,
               "timing_protocol": {"dtype": "bf16 weights and inputs", "compile": not a.no_compile,
                                   "compile_mode": a.compile_mode, "warmup": a.warmup, "iters": a.iters,
                                   "batches": a.batch_list,
                                   "timer": "CUDA events" if device.type == "cuda" else "perf_counter"},
               "timing_rows": rows, "ratios": rat, "bar": bar, "autotune": K.autotune_report(),
               "gpu_processes_before_timing": procs_before, "gpu_processes_after_timing": procs_after,
               "git": git_info(), "prereg": {k: prereg.get(k) for k in ("ok", "reason", "sha256")}}
    write_summary(out, raw, a)                    # from the raw dict: its batch keys are still integers
    metrics = _jsonable(raw)
    mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed},
                     models={"source": src["label"], "kind": src["kind"], "shapes": shapes}, device=device,
                     wall_seconds=time.time() - t_start,
                     extra={"backend": backend, "triton": status, "gpu_processes_before_timing": procs_before,
                            "gpu_processes_after_timing": procs_after, "bar": bar.get("status")})
    log(f"{EXPERIMENT}: correctness {'PASS' if corr['pass'] else 'FAIL'}; bar {bar['status']}"
        + (f" ({json.dumps(bar['ratios'])})" if bar["ratios"] else "") + f"; wrote {out}")
    return {**metrics, "ok": ok}


if __name__ == "__main__":
    sys.exit(0 if main().get("ok") else 1)

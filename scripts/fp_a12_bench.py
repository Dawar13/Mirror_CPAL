#!/usr/bin/env python
"""A12: systems table and profile (FINAL_PHASE_PLAN.md v2, item A12; the batch-256 time also feeds A07). The
measurements are defined in mirror/final/systems/bench.py.

    python scripts/fp_a12_bench.py --manifest jobs/a12_dinov2_vitb14_reg.json --out results/A12/dinov2_vitb14_reg
    python scripts/fp_a12_bench.py --dry-run --out tiny_out/A12            # CPU, tiny random models, eager

Manifest: a JSON list of entries, each one of
  {"name": "original", "kind": "original", "model": "dinov2_vitb14_reg"}                (registry name)
  {"name": "mirror_dense", "kind": "run", "run_dir": "runs/final/.../mirror_exact/s0", "form": "dense"}
  {"name": "mirror_even_odd", "kind": "run", "run_dir": "...", "form": "even_odd", "align": 8}
  {"name": "svdllm", "kind": "run", "run_dir": "runs/final/.../svdllm/s0"}              (any healed student)
  {"name": "flipavg_original", "kind": "flip_average", "base": "original", "batched": false}
optional "output": "logits" (DeiT-III: time the classifier, not the feature) and "step" (checkpoint step). Runs load
through the heal agent's mirror.final.heal.students.load_student when it exists, else
mirror.heal_structured.load_healed_model; analytic parameters and FLOPs use the MUST-09 accounting.

Steps (rerunning skips (model, batch) rows already in systems_rows.csv unless --force):
 1. throughput, latency and peak memory per model and batch (bf16, torch.compile, 20 warm-up and 100 timed runs,
    CUDA events) -> systems_rows.csv and the one-row-per-model systems_table.csv;
 2. torch.profiler of the even/odd form (and of the original for reference): GEMM, butterflies and permutes,
    attention, rest -> profile.json;
 3. every GEMM shape of the even/odd form (and of the original) timed alone: achieved TFLOP/s -> gemm.csv;
 4. the even/odd form at alignment 1 and 8 (700 and 68 are not multiples of 8) -> align.csv, align_gemm.csv;
 5. summary.md, metrics.json, meta.json (mirror.final.prereg.write_final_meta). The script calls
    mirror.final.prereg.require_prereg() before using the GPU.
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.3

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.audit import exactness as EX  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.final.systems import bench as BE  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A12"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=None, help="JSON list of entries (see the docstring)")
    p.add_argument("--batches", default="1,64,256")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--profile", default="auto", help="entry to profile (auto: the first even/odd entry; none: skip)")
    p.add_argument("--prof-batch", type=int, default=256)
    p.add_argument("--prof-iters", type=int, default=10)
    p.add_argument("--no-gemm", action="store_true")
    p.add_argument("--gemm-batch", type=int, default=256)
    p.add_argument("--aligns", default="1,8", help="alignments of the even/odd form to compare ('' to skip)")
    p.add_argument("--align-batches", default="64,256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default results/A12 (dry run: tiny_out/A12)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random models, eager, 1 warm-up and 2 runs")
    a = p.parse_args(argv)
    if a.dry_run:
        a.batches, a.align_batches = "1,2", "2"
        a.warmup, a.iters, a.prof_iters, a.prof_batch, a.gemm_batch = 1, 2, 1, 2, 2
        a.no_compile, a.img_size = True, 56                 # the tiny model's image size
    elif not a.manifest:
        p.error("--manifest is required (or --dry-run)")
    a.batch_list = [int(b) for b in a.batches.split(",") if b.strip()]
    a.align_list = [int(x) for x in a.aligns.split(",") if x.strip()]
    a.align_batch_list = [int(b) for b in a.align_batches.split(",") if b.strip()]
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


# ----------------------------------------------------------------------------------------------
# Entries
# ----------------------------------------------------------------------------------------------
def load_run(run_dir, step, device) -> tuple:
    run = _resolve(run_dir)
    try:
        from mirror.final.heal.students import load_student
        adapter, model, info = load_student(run, step=step, device=device)
    except ImportError:
        from mirror.heal_structured import load_healed_model
        adapter, model, info = load_healed_model(run, device)
    info = info or {}
    if adapter is None:
        from mirror.heal_structured import build_adapter_from_spec
        adapter = build_adapter_from_spec(info["spec"]["adapter"], device)
    return adapter, model, info


def _wrap(adapter, model, output: str):
    return BE.LogitsModule(adapter, model) if output == "logits" else BE.FeatureModule(adapter, model)


def manifest_entries(a, device) -> tuple:
    """(entries, context) from the manifest; context keeps adapters and structures for the profile and alignment."""
    spec = json.loads(_resolve(a.manifest).read_text(encoding="utf-8"))
    entries, ctx, by_name = [], {"even_odd": [], "adapter": None}, {}
    cache = {}
    for e in spec:
        name, kind, output = e["name"], e["kind"], e.get("output", "feature")
        try:
            if kind == "original":
                from mirror.adapters_timm import build_model_adapter
                ad = build_model_adapter(e["model"], device, img_size=a.img_size)
                cost = BE.structure_cost(ad)
                ent = BE.BenchEntry(name, _wrap(ad, ad.model, output), "original",
                                    params_analytic=cost["params_model"], analytic=cost, meta={"model": e["model"]})
                ctx["adapter"] = ctx["adapter"] or ad
            elif kind == "run":
                key = (e["run_dir"], e.get("step"))
                if key not in cache:
                    cache[key] = load_run(e["run_dir"], e.get("step"), device)
                ad, model, info = cache[key]
                ctx["adapter"] = ctx["adapter"] or ad
                try:
                    cost = BE.structure_cost(ad, model, info.get("spec"), info.get("structure"))
                except Exception as exc:  # noqa: BLE001
                    cost = {"gflops": None, "source": f"not available: {type(exc).__name__}: {exc}"[:200]}
                params = info.get("model_params_structured") or cost.get("params_model")
                if e.get("form") == "even_odd":
                    st = EX.AuditStructure.from_info(info)
                    fv = EX.build_even_odd(model, st, align=int(e.get("align", 8)))
                    mod = fv if output != "logits" else _EvenOddLogits(fv, model.head)
                    eo = BE.even_odd_cost(fv, ad)
                    ent = BE.BenchEntry(name, mod, "mirror_even_odd", params_analytic=params,
                                        analytic={**cost, "even_odd": eo}, meta={"run_dir": e["run_dir"],
                                                                                 "align": int(e.get("align", 8))})
                    ctx["even_odd"].append({"name": name, "model": copy.deepcopy(model).cpu(), "st": st,
                                            "adapter": ad, "module": mod})   # a float32 copy: benchmarking casts in place
                else:
                    ent = BE.BenchEntry(name, _wrap(ad, model, output), e.get("label_kind", info.get("method") or "run"),
                                        params_analytic=params, analytic=cost, meta={"run_dir": e["run_dir"]})
            elif kind == "flip_average":
                base = by_name[e["base"]]
                ent = BE.BenchEntry(name, BE.FlipAverage(base.module, batched=bool(e.get("batched", False))),
                                    "flip_average", passes=2, params_analytic=base.params_analytic,
                                    analytic=base.analytic, meta={"base": e["base"], "batched": bool(e.get("batched"))})
            else:
                raise ValueError(f"unknown kind {kind!r}")
        except Exception as exc:  # noqa: BLE001  (one entry failing does not stop the others)
            log(f"A12: entry {name} not loaded: {type(exc).__name__}: {exc}")
            ctx.setdefault("failed", {})[name] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        entries.append(ent)
        by_name[name] = ent
    return entries, ctx


class _EvenOddLogits(torch.nn.Module):
    def __init__(self, fv, head):
        super().__init__()
        self.fv, self.head = fv, head

    def forward(self, x):
        return self.head(self.fv(x))


def dry_entries(a) -> tuple:
    """Tiny random models: the original, Mirror stored densely and in even/odd form, an SVD-factorised model and
    flip averaging of the original."""
    from mirror.adapters import TimmTinyAdapter
    from mirror.attention import attn_dims
    from mirror.efficient import BlockSpec, build_fast_vit
    from mirror.fold import fold_copy
    from mirror.heal_vexact import vexact_block_params
    from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
    from mirror.matching import random_pairing

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=3)
    perms = [random_pairing(h, h // 3, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=20 + b) for b in range(L)]
    model, info = build_vexact(ad, A, list(range(L)), perms, maps, even_head=True)
    st = EX.AuditStructure(A=A, perms=perms, maps=info["maps_after"])
    arch = BE.model_arch(ad)
    blocks = vexact_block_params(d, h, A, perms, info["maps_after"])
    mc = BE.analytic_cost(arch, [{"mlp": b["mlp"], "attn": b["attn"], "rot": 0} for b in blocks])
    mirror_cost = {"gflops": mc["gflops_model"], "params_model": int(mc["model_params"]),
                   "source": "scripts/collect_results.analytic_cost (V-exact free parameters)"}
    orig = BE.BenchEntry("original", BE.FeatureModule(ad, ad.model), "original",
                         params_analytic=arch["model_params_dense"], analytic=BE.structure_cost(ad))
    fv = EX.build_even_odd(model, st, align=8)
    folded, _ = fold_copy(ad.model, final_norm_affine=False)
    svd = build_fast_vit(folded, [BlockSpec(("svd", (8, 8, 8, 8)), ("svd", (16, 16))) for _ in range(L)])
    entries = [orig,
               BE.BenchEntry("mirror_dense", BE.FeatureModule(ad, model), "mirror_dense",
                             params_analytic=mirror_cost["params_model"], analytic=mirror_cost),
               BE.BenchEntry("mirror_even_odd", fv, "mirror_even_odd", params_analytic=mirror_cost["params_model"],
                             analytic={**mirror_cost, "even_odd": BE.even_odd_cost(fv, ad)}),
               BE.BenchEntry("svd_r8", svd, "lowrank", analytic=BE.even_odd_cost(svd, ad)),
               BE.BenchEntry("flipavg_original", BE.FlipAverage(orig.module), "flip_average", passes=2,
                             params_analytic=orig.params_analytic, analytic=orig.analytic)]
    ctx = {"adapter": ad, "even_odd": [{"name": "mirror_even_odd", "model": copy.deepcopy(model), "st": st, "adapter": ad,
                                        "module": fv}]}
    return entries, ctx


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def write_summary(out: Path, table: list, prof: dict, gemm: list, align: dict, a) -> Path:
    def n(x, nd=1):
        return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{nd}f}"

    bs = a.batch_list
    lines = ["# A12 systems table", "",
             f"bf16 weights and inputs, {'eager' if a.no_compile else 'torch.compile (' + a.compile_mode + ')'}, "
             f"median of {a.iters} runs after {a.warmup} warm-up runs; parameters as the method counts them and as "
             "stored; memory of the stored entries in bf16 (MB = 10^6 bytes).", "",
             "| Model | Params (method) | Stored entries | MB (bf16) | GFLOPs | "
             + " | ".join(f"img/s b{b}" for b in bs) + f" | ms b{bs[0]} | peak GB |",
             "|---|---|---|---|---|" + "---|" * len(bs) + "---|---|"]
    for t in table:
        lines.append(f"| {t['name']} | {n(t['params_analytic'], 0) if t['params_analytic'] else 'n/a'} | "
                     f"{t['entries_stored']:,} | {n(t['param_mb_bf16'])} | {n(t['gflops_analytic'], 2)} | "
                     + " | ".join(n(t.get(f'throughput_b{b}')) for b in bs)
                     + f" | {n(t.get(f'latency_ms_b{bs[0]}'), 3)} | {n(t['peak_memory_gb_max'], 2)} |")
    if prof:
        lines += ["", "## Where the time goes (torch.profiler)", ""]
        for name, p in prof.items():
            sh = p["groups"]["share"]
            lines.append(f"- {name}: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in sorted(sh.items(), key=lambda z: -z[1]))
                         + f" (total {p['groups']['total_ms']:.3f} ms per forward, {p.get('time_source')})")
    if gemm:
        lines += ["", "## GEMM shapes (each timed alone)", ""]
        for r in sorted(gemm, key=lambda r: r["tflops"])[:12]:
            lines.append(f"- {r['model']}: M {r['M']}, N {r['N']}, K {r['K']} (x{r['count']} per forward; "
                         f"N and K multiples of 8: {r['N_mult8']} and {r['K_mult8']}): {r['tflops']:.2f} TFLOP/s")
    if align and align.get("rows"):
        lines += ["", "## Alignment of the even/odd form", ""]
        for r in align["rows"]:
            lines.append(f"- align {r['align']}, batch {r['batch']}: {n(r['throughput_img_s'])} img/s "
                         f"({n(r['latency_ms_median'], 3)} ms)")
    lines += ["", "Files: systems_rows.csv, systems_table.csv, profile.json, gemm.csv, align.csv, align_gemm.csv, "
              "metrics.json, meta.json."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.dry_run else "cpu"))
    prereg = require_prereg(allow_missing=True if a.dry_run else None)     # before the GPU is touched
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)              # cap before anything is allocated
    BE.GPU_FRAC = a.gpu_frac if device.type == "cuda" else None             # re-applied after m2_profile's own cap
    set_seed(a.seed)
    git = git_info()
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out" if a.dry_run else "results") / EXPERIMENT
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    entries, ctx = dry_entries(a) if a.dry_run else manifest_entries(a, device)
    log(f"{EXPERIMENT}: {len(entries)} models on {device}: {', '.join(e.name for e in entries)}; out {out}")
    rpath = out / "systems_rows.csv"
    old = [] if (a.force or not rpath.exists()) else pd.read_csv(rpath).to_dict(orient="records")
    done = {(r["name"], int(r["batch"])) for r in old}
    common = dict(img_px=a.img_size, device=device, compile=not a.no_compile, warmup=a.warmup, iters=a.iters,
                  seed=a.seed)
    rows = old + BE.run_benchmark(entries, a.batch_list, compile_mode=a.compile_mode, done=done, log=log, **common)
    pd.DataFrame(rows).to_csv(rpath, index=False)
    table = BE.systems_table(rows)
    pd.DataFrame(table).to_csv(out / "systems_table.csv", index=False)

    ad = ctx.get("adapter")
    T = int(ad.n_tokens) if ad is not None else None
    prof = {}
    eo = ctx.get("even_odd") or []
    target = (eo[0] if eo else None) if a.profile == "auto" else next((x for x in eo if x["name"] == a.profile), None)
    ppath = out / "profile.json"
    if a.profile != "none" and target is not None:
        if ppath.exists() and not a.force:
            prof = json.loads(ppath.read_text(encoding="utf-8"))
        else:
            orig = next((e for e in entries if e.kind == "original"), None)
            for name, mod in [(target["name"], target["module"])] + ([("original", orig.module)] if orig else []):
                prof[name] = BE.profile_summary(mod, a.prof_batch, img_px=a.img_size, device=device,
                                                compile=not a.no_compile, compile_mode=a.compile_mode,
                                                iters=a.prof_iters, warmup=max(1, a.warmup // 4), seed=a.seed)
            ppath.write_text(json.dumps(_jsonable(prof), indent=2), encoding="utf-8")
    gemm = []
    gpath = out / "gemm.csv"
    if not a.no_gemm and target is not None and T is not None:
        if gpath.exists() and not a.force:
            gemm = pd.read_csv(gpath).to_dict(orient="records")
        else:
            orig = next((e for e in entries if e.kind == "original"), None)
            for name, mod in [(target["name"], target["module"])] + ([("original", orig.module)] if orig else []):
                for r in BE.gemm_table(BE.gemm_shapes(mod, T, a.gemm_batch), device=device,
                                       warmup=max(1, a.warmup // 2), iters=max(1, a.iters // 2), seed=a.seed):
                    gemm.append({"model": name, **r})
            pd.DataFrame(gemm).to_csv(gpath, index=False)
    align = {}
    apath = out / "align.csv"
    if a.align_list and target is not None and T is not None:
        if apath.exists() and not a.force:
            align = {"rows": pd.read_csv(apath).to_dict(orient="records")}
        else:
            align = BE.align_effect(lambda al: EX.build_even_odd(target["model"], target["st"], align=al), a.align_list,
                                    batches=a.align_batch_list, n_tokens=T, gemm=not a.no_gemm, log=log, **common)
            pd.DataFrame(align["rows"]).to_csv(apath, index=False)
            pd.DataFrame(align["gemms"]).to_csv(out / "align_gemm.csv", index=False)
    write_summary(out, table, prof, gemm, align, a)
    metrics = {"experiment": EXPERIMENT, "table": table, "profile_groups": {k: v["groups"] for k, v in prof.items()},
               "failed_entries": ctx.get("failed", {}), "device": str(device),
               "protocol": {"dtype": "bf16 weights and inputs", "compile": not a.no_compile,
                            "compile_mode": a.compile_mode, "warmup": a.warmup, "iters": a.iters,
                            "batches": a.batch_list, "timer": "CUDA events" if device.type == "cuda" else "perf_counter"},
               "prereg": {k: prereg.get(k) for k in ("ok", "reason", "sha256")}}
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    nv = None
    if device.type == "cuda":                     # as scripts/m2_latency.nvidia_smi (not imported: it loads m1 scripts)
        import subprocess
        try:
            nv = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            nv = f"unavailable ({type(exc).__name__})"
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed},
                     models={e.name: e.meta for e in entries}, device=device, wall_seconds=time.time() - t_start,
                     extra={"nvidia_smi": nv, "n_rows": len(rows)})
    log(f"{EXPERIMENT}: wrote {out}")
    return {"out": str(out), "rows": rows, "table": table, "profile": prof, "gemm": gemm, "align": align}


if __name__ == "__main__":
    main()

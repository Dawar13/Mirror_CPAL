#!/usr/bin/env python
"""Final phase: one-off k-NN evaluation of checkpoints or released models (the queue server's computation).

    python scripts/fp_eval.py --run-dir runs/final/A05/dinov2_vitb14_reg/mirror_exact/s0 --step 780 --passes bf16,fp32
    python scripts/fp_eval.py --run-dir <run> --all-steps                                # every ckpt/step<S>.pt
    python scripts/fp_eval.py --run-dir <run> --step 300 --n-query 10000                  # the MUST-09 10k subset
    python scripts/fp_eval.py --original dinov2_vits14_reg --out runs/final/A07/dinov2_vits14_reg --flip-average
    python scripts/fp_eval.py --run-dir <run> --step 780 --flip-average --probe --probe-flipavg      # A07 rows
    python scripts/fp_eval.py --dry-run --out tiny_out/fp_eval                            # CPU, tiny model, synthetic

What it does: mirror.final.eval.knn_fast (the MUST-09 k-NN protocol with the 100k bank and the 50k validation images
held on the GPU; passes bf16, fp32, bf16_flipavg, fp32_flipavg), for each (run directory, step) or released model,
with the same outputs as scripts/fp_eval_server.py: <out>/eval/<tag>.json, <out>/knn/<tag>_<key>.npz (MUST-09 format)
and <out>/eval/<tag>_meta.json (<out> = the run directory unless --out is given; tag step<S>, or "original"). Finished
passes with the same settings are skipped (resume). Students are loaded with --loader (default
mirror.final.heal.students:load_student(run_dir, step, device)); released models with --original-builder (default
mirror.adapters_timm.build_model_adapter).
--probe adds the Tier-2 linear probe (scripts/tier2_linear.py run_probe and knn_check on [CLS, mean patch token] of the
100k bank and the 50k validation images; mirror.final.eval.linear_protocol.tier2_probe), --probe-flipavg its
flip-averaged twin (MUST-10's probe feature, A07): <out>/eval/<tag>_probe[_flipavg].json and
<out>/knn/<tag>_probe[_flipavg]_pred.npz.
--cost (on by default with a flip-averaged pass) records the analytic GFLOPs per image (mirror.final.eval.flipavg:
the MUST-09 cost model; flip averaging = 2 forward passes).
GPU: the memory cap is set first (--gpu-frac, default automatic), then mirror.final.prereg.require_prereg().
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.eval import knn_fast as KF  # noqa: E402
from mirror.utils import git_info, log  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", default=None, help="a run directory (repeatable)")
    p.add_argument("--step", type=int, action="append", default=None, help="checkpoint step (repeatable)")
    p.add_argument("--all-steps", action="store_true", help="every <run>/ckpt/step<S>.pt")
    p.add_argument("--original", action="append", default=None, metavar="MODEL", help="a released model (repeatable)")
    p.add_argument("--out", default=None, help="output directory (default: the run directory; needed for --original)")
    p.add_argument("--passes", default="bf16", help="comma list of bf16, fp32, bf16_flipavg, fp32_flipavg")
    p.add_argument("--flip-average", action="store_true", help="add the _flipavg twin of every pass")
    p.add_argument("--n-query", type=int, default=None, help="queries (default all 50k; 10000 = the MUST-09 subset)")
    p.add_argument("--fp32-n-query", type=int, default=None)
    p.add_argument("--fp32-bank-per-class", type=int, default=None)
    p.add_argument("--with-original", default="bf16", help="the original model's passes to add ('none' for none)")
    p.add_argument("--cost", action="store_true", help="record the analytic GFLOPs per image")
    p.add_argument("--probe", action="store_true", help="Tier-2 linear probe")
    p.add_argument("--probe-flipavg", action="store_true", help="Tier-2 linear probe on the flip-averaged feature")
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--experiment", default=None)
    p.add_argument("--loader", default=KF.DEFAULT_LOADER)
    p.add_argument("--original-builder", default=None)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--cache-root", default="runs/final/_cache")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--data-on", default="device", choices=["device", "host"])
    p.add_argument("--gpu-frac", type=float, default=None)
    p.add_argument("--reserve-gb", type=float, default=12.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny model, synthetic data, a stand-in loader")
    return p.parse_args(argv)


def steps_of(run: Path) -> list:
    """Steps of every <run>/ckpt/step<S>.pt, ascending."""
    out = []
    for f in (run / "ckpt").glob("step*.pt"):
        s = f.stem[4:]
        if s.isdigit():
            out.append(int(s))
    return sorted(out)


def probe_part(ev, r: dict, flipavg: bool, a, loader, build_original, adapters, frac) -> dict:
    """Tier-2 probe of one request's model (resumable: a finished probe with the same settings is kept)."""
    from mirror.final.eval.linear_protocol import tier2_probe

    run = None if r["original"] else Path(r["run_dir"])
    out = Path(r["out_dir"]) if r.get("out_dir") else run
    tag = f"step{r['step']}" if r["step"] is not None else ("original" if r["original"] else "final")
    name = f"{tag}_probe{'_flipavg' if flipavg else ''}"
    path = out / "eval" / f"{name}.json"
    ck = KF.checkpoint_path(run, r["step"])
    settings = {"seed": a.probe_seed, "flip_average": flipavg, "data": ev.data.source,
                "checkpoint_sha256": KF.sha256_file(ck) if ck is not None and ck.is_file() else None}
    prev = KF.read_json(path)
    if prev and prev.get("settings") == KF.jsonable(settings):
        log(f"{name}: done before (top-1 {prev['top1']:.2f}); skipping")
        return prev
    adapter, model, info, model_name = KF.load_for_request(r, ev.device, loader, build_original, adapters)
    res = tier2_probe(adapter, model, ev.data, ev.device, flipavg=flipavg, batch_size=a.probe_batch_size,
                      seed=a.probe_seed, gpu_frac=frac, log=log)
    pp = out / "knn" / f"{name}_pred.npz"
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pp, pred=res.pop("val_pred").astype(np.int16), labels=res.pop("val_labels").astype(np.int16))
    rec = {**res, "settings": settings, "model": model_name, "run_dir": None if run is None else str(run),
           "step": r["step"], "predictions": pp.relative_to(out).as_posix(), "git": git_info()}
    KF.write_json_atomic(path, rec)
    return rec


def requests_of(a) -> list:
    base = {"passes": a.passes, "flip_average": a.flip_average, "n_query": a.n_query, "fp32_n_query": a.fp32_n_query,
            "fp32_bank_per_class": a.fp32_bank_per_class, "experiment": a.experiment,
            "with_original": [] if a.with_original.strip().lower() == "none" else a.with_original,
            "cost": True if a.cost else None}
    reqs = []
    for run in a.run_dir or []:
        steps = steps_of(Path(run)) if a.all_steps else (a.step or [None])
        if not steps:
            raise SystemExit(f"{run}: no ckpt/step<S>.pt found")
        for s in steps:
            reqs.append({**base, "run_dir": str(run), "step": s, "out_dir": a.out})
    for name in a.original or []:
        if not a.out:
            raise SystemExit("--original needs --out")
        out = Path(a.out) if len(a.original) == 1 else Path(a.out) / name
        reqs.append({**base, "original": True, "model": name, "out_dir": str(out)})
    return reqs


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    if a.dry_run and not (a.run_dir or a.original):
        out = Path(a.out or "tiny_out/fp_eval")
        a.run_dir = [str(out / "dry_run" / "tiny" / "mirror_exact" / "s0")]
        a.step = a.step or [0, 1]
        a.passes = "bf16,fp32" if a.passes == "bf16" else a.passes
        a.out = None
    ev, device, frac, st = KF.make_evaluator(dry_run=a.dry_run, cache_dir=a.cache_dir, device=a.device,
                                             batch_size=a.batch_size, data_on=a.data_on, gpu_frac=a.gpu_frac,
                                             reserve_gb=a.reserve_gb, cache_root=a.cache_root, log=log)
    loader = KF.tiny_loader if a.dry_run else a.loader
    build_original = (lambda name, dev: KF.tiny_original(name, dev)) if a.dry_run else a.original_builder
    if frac is not None and not a.dry_run:
        if a.run_dir:
            loader = KF.resolve_callable(loader)             # import now, then re-apply this process's cap
        from mirror.utils import cuda_index
        torch.cuda.set_per_process_memory_fraction(float(frac), cuda_index(device))
    reqs = requests_of(a)
    if not reqs:
        raise SystemExit("nothing to evaluate: give --run-dir (with --step or --all-steps) or --original")
    log(f"fp_eval: {len(reqs)} evaluation(s) on {device}, data {ev.data.source}")
    adapters: dict = {}
    results = []
    for req in reqs:
        rec = KF.evaluate_request(ev, req, loader, build_original, adapters)
        r = KF.normalise_request(req)
        if a.probe:
            rec["probe"] = probe_part(ev, r, False, a, loader, build_original, adapters, frac)
        if a.probe_flipavg:
            rec["probe_flipavg"] = probe_part(ev, r, True, a, loader, build_original, adapters, frac)
        results.append(rec)
        KF.free_memory(device)
    log(f"fp_eval: done in {time.time() - t0:.0f} s")
    return {"results": results, "prereg": st, "gpu_frac": frac}


if __name__ == "__main__":
    main()

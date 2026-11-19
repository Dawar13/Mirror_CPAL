#!/usr/bin/env python
"""A08-1: ImageNet top-1 of a supervised model's own classifier (DeiT-III-B/16) and the flip consistency of its
predicted class, for healed students and for the original model (FINAL_PHASE_PLAN.md A08; PREREG_FINAL_PHASE.md
Sections 2.1 and 2.6).

    python scripts/fp_a08_top1.py --run-dir runs/final/a08/deit3_b16/mirror_exact/s0 [--run-dir ...] --passes bf16,fp32
    python scripts/fp_a08_top1.py --original deit3_b16 --passes bf16
    python scripts/fp_a08_top1.py --dry-run --out /tmp/fp_top1          # CPU, a tiny DeiT-III, synthetic images

Every pass runs all 50,000 validation images (mirror.protocol.Protocol, cache/) and their mirror images through
the classifier (mirror.final.adapters.top1_eval): "bf16" under bf16 autocast (the headline top-1), "fp32" in float32
with TF32 off (the exactness claim: 100% flip consistency, or every disagreement a near-tie with a relative top-two
logit margin below 1e-4 in both copies). The classifier of a Mirror student reads the even part of the class token
(the even head installed at full conversion). Outputs per run directory: eval/top1_step<S>.json (one entry per pass)
and top1/step<S>_<pass>.npz (pred, pred_flip, labels, for the bootstrap); for --original under
runs/final/a08/<model>/_original/. The script calls mirror.final.prereg.require_prereg() before using the GPU and
writes meta.json with mirror.final.prereg.write_final_meta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, log  # noqa: E402

EXPERIMENT = "A08"
PASSES = ("bf16", "fp32")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", default=None, help="healed student run directory (repeatable)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: the last)")
    p.add_argument("--original", action="append", default=None, metavar="MODEL",
                   help="also evaluate a released model (for example deit3_b16)")
    p.add_argument("--passes", default="bf16,fp32", help="comma list of bf16 and fp32")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default=None, help="output folder for --original and --dry-run")
    p.add_argument("--gpu-frac", type=float, default=0.2)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="redo passes that are already written")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _passes(text: str) -> list:
    out = [s.strip() for s in str(text).split(",") if s.strip()]
    bad = [s for s in out if s not in PASSES]
    if bad:
        raise SystemExit(f"unknown passes {bad}; choose from {PASSES}")
    return out


def run_passes(adapter, model, loader, device, passes, tf32_note: dict) -> dict:
    """top1_eval for every pass; the fp32 pass with TF32 off and the model in float32."""
    from mirror.final.adapters import top1_eval

    results = {}
    for name in passes:
        t0 = time.time()
        if name == "bf16":
            res = top1_eval(adapter, model, loader, device=device, amp="bf16", flip=True, return_preds=True)
        else:
            prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            try:
                res = top1_eval(adapter, model.float(), loader, device=device, amp=None, flip=True, return_preds=True)
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
            res["precision"] = "float32, TF32 off"
        res["seconds"] = round(time.time() - t0, 1)
        results[name] = res
        log(f"top-1 {name}: {res['top1']:.3f}, flip consistency {res.get('flip_consistency', float('nan')):.4f}% "
            f"({res.get('n_disagree')} disagreements, near-tie pass {res.get('pass_with_near_ties')}; "
            f"{res['seconds']:.0f} s)")
    tf32_note.update(fp32="TF32 off for matmul and cuDNN during the fp32 pass")
    return results


def write_outputs(folder: Path, step, results: dict) -> Path:
    folder = Path(folder)
    (folder / "eval").mkdir(parents=True, exist_ok=True)
    (folder / "top1").mkdir(parents=True, exist_ok=True)
    tag = "original" if step is None else f"step{int(step)}"
    summary = {}
    for name, res in results.items():
        arrays = {k: res.pop(k) for k in ("pred", "pred_flip", "labels") if k in res}
        np.savez(folder / "top1" / f"{tag}_{name}.npz", **arrays)
        summary[name] = res
    path = folder / "eval" / f"top1_{tag}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(summary)
    path.write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
    return path


def main(argv=None) -> int:
    a = parse_args(argv)
    from mirror.final.prereg import require_prereg, write_final_meta

    t_start = time.time()
    passes = _passes(a.passes)
    if a.dry_run:
        from mirror.data import make_loader, synthetic_dataset
        from mirror.final.adapters import tiny_deit3

        device = torch.device(a.device or "cpu")
        pre = require_prereg(allow_missing=True)
        adapter = tiny_deit3(dtype=torch.float32).to(device=device)
        images, labels = synthetic_dataset(10, 4, int(adapter.img_size), seed=0)
        loader = make_loader(images, labels, np.arange(len(labels)), 8)
        out = Path(a.out or (REPO_ROOT / "tiny_out" / "fp_a08_top1"))
        note = {}
        results = run_passes(adapter, adapter.model, loader, device, passes, note)
        path = write_outputs(out, None, results)
        write_final_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={}, models={"tiny_deit3": "random"},
                         device=device, wall_seconds=time.time() - t_start,
                         extra={"prereg_check": pre, "dry_run": True, **note})
        log(f"dry run: wrote {path}")
        return 0

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac), device)
    pre = require_prereg()
    from mirror.final.heal.students import load_student
    from mirror.protocol import Protocol

    proto = Protocol(a.cache_dir, batch_size=int(a.batch_size), num_workers=int(a.num_workers))
    loader = proto.query_loader()
    done = []
    for run in a.run_dir or []:
        adapter, model, info = load_student(run, a.step, device=device)
        if adapter is None:
            raise SystemExit(f"{run}: the adapter could not be built (model weights under HF_HOME?)")
        step = int(info["step"])
        tgt = Path(run) / "eval" / f"top1_step{step}.json"
        have = json.loads(tgt.read_text(encoding="utf-8")) if tgt.exists() and not a.force else {}
        todo = [p for p in passes if p not in have]
        if not todo:
            log(f"{run}: passes {passes} done before ({tgt})")
            continue
        note = {}
        results = run_passes(adapter, model.eval(), loader, device, todo, note)
        path = write_outputs(Path(run), step, results)
        write_final_meta(Path(run) / "eval", name=f"meta_top1_step{step}.json", experiment=EXPERIMENT,
                         config=vars(a), seeds={"heal": info.get("spec", {}).get("seed")},
                         models={info.get("spec", {}).get("model", "?"): info.get("spec", {}).get("adapter")},
                         device=device, wall_seconds=time.time() - t_start, extra={"prereg_check": pre, **note})
        done.append(str(path))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name in a.original or []:
        from mirror.adapters_timm import build_model_adapter

        adapter = build_model_adapter(name, device=device)
        out = Path(a.out or (REPO_ROOT / "runs" / "final" / "a08" / name / "_original"))
        note = {}
        results = run_passes(adapter, adapter.model.eval(), loader, device, passes, note)
        path = write_outputs(out, None, results)
        write_final_meta(out, name="meta_top1.json", experiment=EXPERIMENT, config=vars(a), seeds={},
                         models={name: "released"}, device=device, wall_seconds=time.time() - t_start,
                         extra={"prereg_check": pre, **note})
        done.append(str(path))
    log(f"A08 top-1: wrote {len(done)} result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

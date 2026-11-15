#!/usr/bin/env python
"""Controlled training (MUST-12, CORE-01, UP-01, UP-03): a small ViT from scratch on ImageNet-100 at 64 x 64.

The recipe is identical for every run; only --condition (the symmetry augmentation) changes
(PROJECT_NEXT_PHASE.md, CORE-01). Details of the model, augmentation, conventions, checkpoints and
resume are in mirror/controlled.py.

    # MUST-12 timing pilot (gate C), seed 0, conditions none and LR
    python scripts/train_controlled.py --exp MUST-12 --condition none --seed 0
    python scripts/train_controlled.py --exp MUST-12 --condition LR --seed 0
    # CORE-01 grid (one run per call; 2 or 3 at a time on one H100)
    python scripts/train_controlled.py --condition UD --seed 1 --gpu-frac 0.15
    # 60 epochs if MUST-12 says so: checkpoints become 0, 2, 5, 10, 25, 50, 60
    python scripts/train_controlled.py --condition UD --seed 1 --epochs 60
    # UP-01 and UP-03
    python scripts/train_controlled.py --exp UP-01 --condition UD_p0.1
    python scripts/train_controlled.py --exp UP-03 --arch vit_ti8 --condition UD
    # CPU smoke test (synthetic data, tiny model, about a minute)
    python scripts/train_controlled.py --tiny

Rerunning the same command after a preemption resumes from last.pt exactly where it stopped; a
finished run is left alone. --force-new deletes the run's files and starts again.

Outputs: <out-root>/<exp>/<arch>_<condition>_s<seed>/ with ckpt_e<NNN>.pt, last.pt, train_log.csv,
val_acc.csv and meta.json (meta.json extra.timing holds the MUST-12 timing summary: wall minutes,
steady images per second and epoch seconds, peak memory, projected minutes at 100 and 60 epochs).
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.2) -> float:
    """--gpu-frac from the command line, read before anything touches the GPU."""
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    # Shared GPU: cap before anything is allocated (main() sets it again from the parsed arguments).
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror.controlled import (ARCHS, CONDITIONS, DEFAULT_CKPT_EPOCHS, TrainConfig, cache_store_size,  # noqa: E402
                               ckpt_epochs_for, load_checkpoint_adapter, run_dir_for, train, val_loader,
                               view_size_for_store, write_synthetic_cache)
from mirror.utils import log  # noqa: E402

# Real-run defaults (CORE-01 recipe) and the --tiny overrides.
REAL = {"arch": "vit_s8", "epochs": 100, "batch_size": 512, "warmup_epochs": 10.0, "device": "cuda", "lr": 1e-3,
        "out_root": str(ROOT / "runs"), "cache_dir": str(ROOT / "cache"), "log_every": 50, "eval_batch": 1000}
# Tiny: lr 3.2e-2 at batch 512 is 1e-3 at batch 16, so the tiny model visibly learns.
TINY = {"arch": "vit_test8", "epochs": 3, "batch_size": 16, "warmup_epochs": 1.0, "device": "cpu", "lr": 3.2e-2,
        "out_root": str(ROOT / "tiny_out"), "cache_dir": None, "log_every": 5, "eval_batch": 64}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="CORE-01", help="experiment ID; also the run folder under --out-root")
    p.add_argument("--condition", default="none", choices=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", default=None, choices=list(ARCHS),
                   help="vit_s8 (default), vit_ti8 (UP-03); vit_s16 / vit_ti16 for the 96 x 96 rerun")
    p.add_argument("--img-size", type=int, default=None,
                   help="model input size (default: from the cache's stored size, 72 -> 64, 108 -> 96)")
    p.add_argument("--epochs", type=int, default=None, help="100 (default), or 60 after MUST-12")
    p.add_argument("--ckpt-epochs", default=",".join(str(e) for e in DEFAULT_CKPT_EPOCHS),
                   help="checkpoint epochs; entries beyond --epochs are dropped and the final epoch is added")
    p.add_argument("--batch-size", type=int, default=None, help="512 (default)")
    p.add_argument("--lr", type=float, default=None,
                   help="peak lr at batch 512 (default 1e-3); scaled linearly with the batch")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=None, help="10 (default)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0, help="cosine floor as a fraction of the peak lr")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate (recipe: 0)")
    p.add_argument("--grad-clip", type=float, default=0.0, help="max gradient norm (0: off; the norm is logged)")
    p.add_argument("--amp", choices=("auto", "bf16", "none"), default="auto")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile")
    p.add_argument("--gpu-frac", type=float, default=0.2, help="per-process GPU memory cap (fraction)")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None, help="ImageNet-100 cache (default: cache/)")
    p.add_argument("--out-root", default=None, help="default: runs/ (tiny: tiny_out/)")
    p.add_argument("--run-dir", default=None, help="override the run folder")
    p.add_argument("--save-every-min", type=float, default=5.0,
                   help="write last.pt at least this often (0: off); also at checkpoint epochs and the end")
    p.add_argument("--save-every-epochs", type=int, default=0, help="also write last.pt every n epochs (0: off)")
    p.add_argument("--save-every-steps", type=int, default=0, help="also write last.pt every n steps (0: off)")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--stop-after-steps", type=int, default=None, help="save last.pt and stop (preemption test)")
    p.add_argument("--force-new", action="store_true", help="delete this run's files and start again")
    p.add_argument("--tiny", action="store_true", help="CPU smoke test on synthetic data")
    return p.parse_args(argv)


def build_config(a) -> TrainConfig:
    base = TINY if a.tiny else REAL
    pick = lambda name, val: base[name] if val is None else val  # noqa: E731
    cache_dir = pick("cache_dir", a.cache_dir)
    out_root = pick("out_root", a.out_root)
    if a.tiny and cache_dir is None:
        cache_dir = str(Path(out_root) / "in100_synth")
    epochs = int(pick("epochs", a.epochs))
    requested = [int(s) for s in str(a.ckpt_epochs).split(",") if s.strip()]
    if a.tiny and a.ckpt_epochs == ",".join(str(e) for e in DEFAULT_CKPT_EPOCHS):
        requested = [0, 1, 2]
    return TrainConfig(
        exp=a.exp, arch=pick("arch", a.arch), condition=a.condition, seed=a.seed, epochs=epochs,
        ckpt_epochs=tuple(ckpt_epochs_for(epochs, requested)), batch_size=int(pick("batch_size", a.batch_size)),
        lr=float(pick("lr", a.lr)), weight_decay=a.weight_decay, warmup_epochs=float(pick("warmup_epochs", a.warmup_epochs)),
        min_lr_ratio=a.min_lr_ratio, label_smoothing=a.label_smoothing, drop_path=a.drop_path,
        grad_clip=a.grad_clip, amp=a.amp, compile="off" if a.no_compile else "auto",
        eval_batch=int(pick("eval_batch", a.eval_batch)), save_every_min=a.save_every_min,
        save_every_steps=a.save_every_steps, save_every_epochs=a.save_every_epochs, log_every=int(pick("log_every", a.log_every)),
        cache_dir=cache_dir, out_root=out_root, run_dir=a.run_dir, device=pick("device", a.device),
        stop_after_steps=a.stop_after_steps, force_new=a.force_new)


def main(argv=None):
    a = parse_args(argv)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    cfg = build_config(a)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no GPU visible; use --tiny for a CPU run")
    if a.tiny and not (Path(cfg.cache_dir) / "in100_meta.json").exists():
        write_synthetic_cache(cfg.cache_dir, num_classes=10, train_per_class=8, val_per_class=4, seed=0)
        log(f"synthetic ImageNet-100 stand-in written to {cfg.cache_dir}")
    cfg.img_size = int(a.img_size) if a.img_size is not None else view_size_for_store(cache_store_size(cfg.cache_dir))
    log(f"run {run_dir_for(cfg)}: arch {cfg.arch}, condition {cfg.condition} {CONDITIONS[cfg.condition]}, "
        f"seed {cfg.seed}, {cfg.epochs} epochs, checkpoints {list(cfg.ckpt_epochs)}")
    t0 = time.time()
    out = train(cfg)
    if a.tiny and not out.get("stopped"):
        # Smoke check of the measurement interface on the final checkpoint.
        from mirror.knn import extract_features

        final = run_dir_for(cfg) / f"ckpt_e{cfg.epochs:03d}.pt"
        adapter = load_checkpoint_adapter(final, "cpu")
        feats, labels = extract_features(adapter, val_loader("UD", batch_size=16, cache_dir=cfg.cache_dir), "cpu")
        log(f"interface check: {final.name} -> adapter with grid {adapter.grid_hw}, {adapter.n_tokens} tokens; "
            f"UD validation features {tuple(feats.shape)}")
    log(f"finished in {(time.time() - t0) / 60:.2f} min")
    if not out.get("stopped"):
        # Job-file trigger (jobs/h100_B_core.txt): written only after every checkpoint and meta.json are on disk.
        (run_dir_for(cfg) / "last_done.flag").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()

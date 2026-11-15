#!/usr/bin/env python
"""UP-02 (symmetric start) and UP-04 (tied from scratch; refund and heal) on the controlled ViT (ImageNet-100, 64 px).

    # UP-02: UD run from projected (exactly UD-equivariant) initial weights, trained without constraints;
    #        the UD breaking share under the canonical structure is logged after every epoch
    python scripts/train_controlled_sym.py --mode sym_init --condition UD --seed 0 --epochs 100 --gpu-frac 0.18
    # UP-04: tied from scratch (projected gradient, step, projected weights: the shared parameters only)
    python scripts/train_controlled_sym.py --mode tied --condition UD --seed 0 --epochs 100 --gpu-frac 0.18
    # UP-04: refund and heal a finished CORE-01 UD run (fitted structure, zero-shot projection, 1 epoch kept)
    python scripts/train_controlled_sym.py --mode refund_heal --condition UD --seed 0 --gpu-frac 0.18
    # Collect results/UP-02/summary.md and results/UP-04/summary.md from whatever has finished
    python scripts/train_controlled_sym.py --summarise
    # CPU smoke test (synthetic data, tiny model)
    python scripts/train_controlled_sym.py --tiny --mode tied

The recipe is CORE-01's (mirror/controlled.py) unless stated: same model, initialisation seed, data order, crops and
UD flips, AdamW, schedule, bf16 and torch.compile, checkpoints. Details of the structures, the projection and the
three modes are in mirror/controlled_sym.py. Defaults by mode:
    sym_init, tied   100 epochs, peak lr 1e-3 at batch 512, 10 warm-up epochs, checkpoints 0 2 5 10 25 50 100
    refund_heal      1 epoch, peak lr 1e-4 at batch 512, 0.1 warm-up epochs, checkpoints 0 (zero-shot refund) and 1;
                     source: the final checkpoint of runs/CORE-01/<arch>_UD_s<seed> (or --source-ckpt), structure
                     fitted on fit seed 0 (2,000 ImageNet-100 train images)

Outputs: runs/<exp>/<arch>_UD_<mode>_s<seed>/ (ckpt_e<NNN>.pt in the CORE-01 format, last.pt, train_log.csv,
val_acc.csv, breaking_share.csv, structure.npz for refund_heal, meta.json, last_done.flag) and, at the end,
results/<exp>/<run>/ (summary.json, breaking_share.csv, val_acc.csv, meta.json). exp is UP-02 for sym_init and
UP-04 otherwise. Rerunning the same command resumes from last.pt; a finished run is left alone.
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.2) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))   # shared GPU: cap first

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror import controlled as C  # noqa: E402
from mirror import controlled_sym as S  # noqa: E402
from mirror.utils import log  # noqa: E402

REAL = {"arch": "vit_s8", "batch_size": 512, "device": "cuda", "out_root": str(ROOT / "runs"),
        "results_root": str(ROOT / "results"), "cache_dir": str(ROOT / "cache"), "log_every": 50, "eval_batch": 1000,
        "fit_images": 2000, "workers": 8}
TINY = {"arch": "vit_test8", "batch_size": 16, "device": "cpu", "out_root": str(ROOT / "tiny_out"),
        "results_root": str(ROOT / "tiny_out" / "results"), "cache_dir": None, "log_every": 5, "eval_batch": 64,
        "fit_images": 20, "workers": 1}
# (epochs, lr at batch 512, warm-up epochs, checkpoint epochs); tiny lr 3.2e-2 at batch 512 is 1e-3 at batch 16
MODE_DEFAULTS = {"sym_init": (100, 1e-3, 10.0, C.DEFAULT_CKPT_EPOCHS),
                 "tied": (100, 1e-3, 10.0, C.DEFAULT_CKPT_EPOCHS),
                 "refund_heal": (1, 1e-4, 0.1, (0, 1))}
TINY_MODE = {"sym_init": (2, 3.2e-2, 1.0, (0, 1, 2)), "tied": (2, 3.2e-2, 1.0, (0, 1, 2)),
             "refund_heal": (1, 3.2e-3, 0.2, (0, 1))}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=S.MODES, default="sym_init")
    p.add_argument("--condition", default="UD", choices=["UD"], help="UP-02 and UP-04 are UD runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp", default=None, help="default: UP-02 for sym_init, UP-04 otherwise")
    p.add_argument("--arch", default=None, choices=list(C.ARCHS))
    p.add_argument("--epochs", type=int, default=None, help="default 100 (refund_heal: 1); 60 if MUST-12 says so")
    p.add_argument("--ckpt-epochs", default=None, help="comma list; the final epoch is always added")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="peak lr at batch 512 (default 1e-3; refund_heal 1e-4)")
    p.add_argument("--warmup-epochs", type=float, default=None, help="default 10 (refund_heal 0.1)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--d-minus", type=int, default=0, help="canonical channel action: swapped pairs (0: d / 2)")
    p.add_argument("--hidden-pairs", type=float, default=1.0, help="canonical: share of MLP units in pairs")
    p.add_argument("--init-rescale", action="store_true",
                   help="canonical: after projecting, restore the initialisation variance of paired entries")
    p.add_argument("--no-project-grads", action="store_true", help="constrained modes: project the weights only")
    p.add_argument("--source-ckpt", default=None, help="refund_heal: checkpoint to refund (default: final CORE-01 UD)")
    p.add_argument("--source-run", default=None, help="refund_heal: run folder whose final checkpoint is refunded")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--fit-images", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="processes for the neuron matchings (refund_heal)")
    p.add_argument("--amp", choices=("auto", "bf16", "none"), default="auto")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--gpu-frac", type=float, default=0.2, help="per-process GPU memory cap (fraction)")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--save-every-min", type=float, default=5.0)
    p.add_argument("--save-every-steps", type=int, default=0)
    p.add_argument("--stop-after-steps", type=int, default=None, help="save last.pt and stop (preemption test)")
    p.add_argument("--force-new", action="store_true")
    p.add_argument("--summarise", action="store_true", help="only collect results/UP-02 and results/UP-04 summaries")
    p.add_argument("--core01-runs", default=None, help="CORE-01 run folder for --summarise (default runs/CORE-01)")
    p.add_argument("--tiny", action="store_true", help="CPU smoke test on synthetic data")
    return p.parse_args(argv)


def _tiny_cache(cache_dir: str) -> None:
    if not (Path(cache_dir) / "in100_meta.json").exists():
        C.write_synthetic_cache(cache_dir, num_classes=10, train_per_class=8, val_per_class=4, seed=0)
        log(f"synthetic ImageNet-100 stand-in written to {cache_dir}")


def _tiny_source(a, base: dict, cache_dir: str) -> Path:
    """Tiny unconstrained UD run (controlled.train, as CORE-01) to refund, trained once."""
    run = Path(base["out_root"]) / "CORE-01" / f"{base['arch']}_UD_s{a.seed}"
    if not (run / "last_done.flag").exists():
        cfg = C.TrainConfig(exp="CORE-01", arch=base["arch"], condition="UD", seed=a.seed, epochs=2,
                            ckpt_epochs=(0, 1, 2), batch_size=base["batch_size"], lr=3.2e-2, warmup_epochs=1.0,
                            device="cpu", amp="none", compile="off", eval_batch=base["eval_batch"], save_every_min=0.0,
                            log_every=0, cache_dir=cache_dir, run_dir=str(run))
        C.train(cfg)
        (run / "last_done.flag").write_text("tiny\n")
    return S.final_checkpoint(run)


def build_config(a) -> S.SymTrainConfig:
    base = TINY if a.tiny else REAL
    pick = lambda name, val: base[name] if val is None else val  # noqa: E731
    epochs_d, lr_d, warm_d, ck_d = (TINY_MODE if a.tiny else MODE_DEFAULTS)[a.mode]
    cache_dir = pick("cache_dir", a.cache_dir)
    out_root = pick("out_root", a.out_root)
    if a.tiny and cache_dir is None:
        cache_dir = str(Path(out_root) / "in100_synth")
    if a.tiny:
        _tiny_cache(cache_dir)
    epochs = int(epochs_d if a.epochs is None else a.epochs)
    requested = list(ck_d) if a.ckpt_epochs is None else [int(s) for s in str(a.ckpt_epochs).split(",") if s.strip()]
    source = a.source_ckpt
    if a.mode == "refund_heal" and source is None:
        if a.source_run:
            source = str(S.final_checkpoint(a.source_run))
        elif a.tiny:
            source = str(_tiny_source(a, {**base, "out_root": out_root}, cache_dir))
        else:
            source = str(S.final_checkpoint(ROOT / "runs" / "CORE-01" / f"{pick('arch', a.arch)}_UD_s{a.seed}"))
    return S.SymTrainConfig(
        exp=a.exp or S.DEFAULT_EXP[a.mode], arch=pick("arch", a.arch), condition=a.condition, seed=a.seed,
        epochs=epochs, ckpt_epochs=tuple(C.ckpt_epochs_for(epochs, requested)),
        batch_size=int(pick("batch_size", a.batch_size)), lr=float(lr_d if a.lr is None else a.lr),
        weight_decay=a.weight_decay, warmup_epochs=float(warm_d if a.warmup_epochs is None else a.warmup_epochs),
        label_smoothing=a.label_smoothing, amp=a.amp, compile="off" if a.no_compile else "auto",
        eval_batch=int(base["eval_batch"]), save_every_min=a.save_every_min, save_every_steps=a.save_every_steps,
        log_every=int(base["log_every"]), cache_dir=cache_dir, out_root=out_root,
        device=pick("device", a.device), stop_after_steps=a.stop_after_steps, force_new=a.force_new,
        mode=a.mode, d_minus=a.d_minus, hidden_pairs=a.hidden_pairs, init_rescale=a.init_rescale,
        project_grads=not a.no_project_grads, source_ckpt=source, fit_seed=a.fit_seed,
        fit_images=int(pick("fit_images", a.fit_images)), match_workers=int(pick("workers", a.workers)),
        results_root=pick("results_root", a.results_root))


# ----------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------
def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(x, nd=2):
    x = _f(x)
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def _fmt_e(x):
    x = _f(x)
    return "n/a" if not np.isfinite(x) else f"{x:.2e}"


def core01_ud_runs(core_root: Path, arch: str) -> list:
    """Finished CORE-01 UD runs: final top-1 (orig and UD views) and dense parameter count."""
    rows = []
    for run in sorted(Path(core_root).glob(f"{arch}_UD_s*")):
        meta = run / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("extra", {}).get("status") != "complete":
            continue
        val = _read_csv(run / "val_acc.csv")
        if not val:
            continue
        n_params = next(iter(m.get("models", {}).values()), {}).get("n_params")
        rows.append({"run": run.name, "seed": m.get("config", {}).get("seed"), "epoch": val[-1]["epoch"],
                     "top1_orig": _f(val[-1]["top1_orig"]), "top1_UD": _f(val[-1]["top1_UD"]), "params": n_params})
    return rows


def _summaries(root: Path) -> list:
    out = []
    for p in sorted(root.glob("*/summary.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def summarise(results_root: Path, core_root: Path, arch: str = "vit_s8") -> dict:
    """results/UP-02/summary.{md,csv} and results/UP-04/summary.{md,csv} from finished runs."""
    from mirror.meta import write_meta

    t0 = time.time()
    ud = core01_ud_runs(core_root, arch)
    ud_mean = float(np.mean([r["top1_orig"] for r in ud])) if ud else float("nan")
    ud_line = (f"CORE-01 UD runs ({arch}, final epoch): " + ", ".join(f"{r['run']} {r['top1_orig']:.2f}" for r in ud)
               + f"; mean {ud_mean:.2f} (n = {len(ud)})") if ud else "CORE-01 UD runs: none finished yet."
    out = {}
    # ---- UP-02
    up02 = [s for s in _summaries(results_root / "UP-02") if s.get("mode") == "sym_init"]
    if up02:
        rows, lines = [], []
        for s in up02:
            fv = s.get("final_val", {})
            bs = s.get("breaking_share", {})
            k = next((key for key in bs.get("max_over_epochs", {}) if key.startswith("s_mlp_b1_")), "s_mlp_b1_6")
            same = next((r for r in ud if str(r["seed"]) == str(s["seed"])), None)
            row = {"run": s["run"], "seed": s["seed"], "epochs": s["epochs"], "top1_orig": _f(fv.get("top1_orig")),
                   "top1_UD": _f(fv.get("top1_UD")), "minus_ud_mean": _f(fv.get("top1_orig")) - ud_mean,
                   "minus_ud_same_seed": (_f(fv.get("top1_orig")) - same["top1_orig"]) if same else float("nan"),
                   f"max_{k}": _f(bs.get("max_over_epochs", {}).get(k)), f"end_{k}": _f(bs.get("end", {}).get(k)),
                   "max_s_all": _f(bs.get("max_over_epochs", {}).get("s_all")),
                   "end_s_all": _f(bs.get("end", {}).get("s_all")),
                   "max_s_max_tensor": _f(bs.get("max_over_epochs", {}).get("s_max_tensor"))}
            rows.append(row)
            lines.append(f"| {row['run']} | {_fmt(row['top1_orig'])} | {_fmt(row['minus_ud_mean'])} | "
                         f"{_fmt(row['minus_ud_same_seed'])} | {_fmt_e(row[f'max_{k}'])} | {_fmt_e(row[f'end_{k}'])} | "
                         f"{_fmt_e(row['max_s_all'])} | {_fmt_e(row['max_s_max_tensor'])} |")
        d = results_root / "UP-02"
        _write_csv(d / "summary.csv", rows)
        md = ["# UP-02: symmetric start", "",
              "One UD run (CORE-01 recipe) whose initial weights are projected onto the canonical exactly UD-equivariant",
              "form, then trained without constraints. Breaking shares are measured under that same canonical structure",
              "after every epoch (folded weights for the MLP mean; raw parameters for the pooled share and the largest",
              "single tensor). Plan prediction (PROJECT_NEXT_PHASE.md UP-02, reported, not gating): breaking share below",
              "0.05 throughout training, accuracy within 1 point of the UD runs.", "", ud_line, "",
              f"| run | top-1 | minus UD mean | minus UD same seed | max MLP share ({k}) | final MLP share | "
              "max pooled share | max single tensor |", "|---|---|---|---|---|---|---|---|", *lines, "",
              "Per-epoch values: results/UP-02/<run>/breaking_share.csv.", ""]
        (d / "summary.md").write_text("\n".join(md), encoding="utf-8")
        write_meta(d, experiment="UP-02", config={"results_root": str(results_root), "core01": str(core_root)},
                   wall_seconds=time.time() - t0, extra={"rows": rows, "core01_ud": ud})
        out["UP-02"] = rows
    # ---- UP-04
    up04 = _summaries(results_root / "UP-04")
    if up04 or ud:
        rows = [{"kind": "unconstrained UD (CORE-01)", "run": r["run"], "seed": r["seed"], "top1_orig": r["top1_orig"],
                 "top1_zero_shot": float("nan"), "params_dense": r["params"], "params_free": r["params"],
                 "source_top1": float("nan")} for r in ud]
        for s in up04:
            fv = s.get("final_val", {})
            src = s.get("structure_source", {}) or {}
            src_val = (src.get("source_val") or {}).get("top1_orig") if isinstance(src.get("source_val"), dict) else None
            rows.append({"kind": {"tied": "tied from scratch", "refund_heal": "refund and heal"}.get(s["mode"], s["mode"]),
                         "run": s["run"], "seed": s["seed"], "top1_orig": _f(fv.get("top1_orig")),
                         "top1_zero_shot": _f(s.get("epoch0_val", {}).get("top1_orig")) if s["mode"] == "refund_heal"
                         else float("nan"),
                         "params_dense": s["params_dense"], "params_free": s["params_free"], "source_top1": _f(src_val)})
        d = results_root / "UP-04"
        _write_csv(d / "summary.csv", rows)
        lines = [f"| {r['kind']} | {r['run']} | {_fmt(r['top1_orig'])} | {_fmt(r['top1_zero_shot'])} | "
                 f"{_fmt(r['source_top1'])} | {r['params_free']:,} | {r['params_dense']:,} |"
                 if isinstance(r["params_free"], int) else
                 f"| {r['kind']} | {r['run']} | {_fmt(r['top1_orig'])} | {_fmt(r['top1_zero_shot'])} | "
                 f"{_fmt(r['source_top1'])} | {r['params_free']} | {r['params_dense']} |" for r in rows]
        md = ["# UP-04: tied from scratch", "",
              "Tied: a UD run (CORE-01 recipe) whose weights stay in the canonical exactly UD-equivariant form from the",
              "start (projected gradient, AdamW step, projected weights). Refund and heal: a finished unconstrained UD",
              "run, LayerNorm affines folded, projected onto its fitted UD structure (one shared operator, fitted twins",
              "and head maps; zero-shot column) and trained for 1 epoch with the projection kept. Parameters: free",
              "parameters of the equivariant form against the dense count. Plan prediction (PROJECT_NEXT_PHASE.md UP-04,",
              "reported, not gating): about the same accuracy at the reduced parameter count.", "",
              "| kind | run | top-1 | zero-shot top-1 | source top-1 | free parameters | dense parameters |",
              "|---|---|---|---|---|---|---|", *lines, ""]
        (d / "summary.md").write_text("\n".join(md), encoding="utf-8")
        write_meta(d, experiment="UP-04", config={"results_root": str(results_root), "core01": str(core_root)},
                   wall_seconds=time.time() - t0, extra={"rows": rows})
        out["UP-04"] = rows
    return out


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    a = parse_args(argv)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    base = TINY if a.tiny else REAL
    if a.summarise:
        res_root = Path(a.results_root or base["results_root"])
        core = Path(a.core01_runs) if a.core01_runs else Path(a.out_root or base["out_root"]) / "CORE-01"
        out = summarise(res_root, core, a.arch or base["arch"])
        log(f"summaries written for {sorted(out)} under {res_root}")
        return out
    cfg = build_config(a)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no GPU visible; use --tiny for a CPU run")
    log(f"run {S.sym_run_dir(cfg)}: mode {cfg.mode}, arch {cfg.arch}, seed {cfg.seed}, {cfg.epochs} epochs, "
        f"lr {cfg.lr:g} at batch {cfg.lr_ref_batch}, checkpoints {list(cfg.ckpt_epochs)}"
        + (f", source {cfg.source_ckpt}" if cfg.source_ckpt else ""))
    t0 = time.time()
    out = S.train_sym(cfg)
    log(f"finished in {(time.time() - t0) / 60:.2f} min")
    if not out.get("stopped"):
        (S.sym_run_dir(cfg) / "last_done.flag").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    print(json.dumps({k: v for k, v in out.items() if k not in ("checkpoints", "summary")}, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""MUST-10 (PROJECT_NEXT_PHASE.md): healed Mirror-exact against test-time flip averaging.

    python scripts/must10_flipavg.py --healed-run runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0
    python scripts/must10_flipavg.py --healed-run runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0 --latency-only
    python scripts/must10_flipavg.py --tiny --healed-run tiny_out/MUST-09/tiny/mirror_exact/s0     # CPU smoke test

Rows (model, mode):
  original, plain      the released model, one forward pass, all parameters;
  original, flipavg    f_avg(x) = (f(x) + f(mirror x)) / 2: two forward passes, all parameters, exactly invariant;
  healed, plain        healed Mirror-exact (MUST-09; rebuilt with mirror.heal_structured.load_healed_model): one
                       forward pass, its parameters, the even part of the CLS (its even head), exactly invariant.
Every exact-invariance number is the fp32 one (PREREG_NEXT_PHASE.md, Section 1).

Measures, each with the protocol of every other number:
  k-NN         scripts/flip_avg_baseline.knn_eval (100k train bank, all 50k validation queries, k = 20, T = 0.07,
               cosine) in bf16 (top-1, as every earlier number) and again with fp32 features for the bank and every
               query and its mirror, TF32 off (flip_consistency_fp32: the measure of exact invariance). The flip
               average is taken on the CLS after the final norm, before the L2 normalisation (FlipAveragedAdapter).
  probe        ImageNet linear probe on [CLS, mean patch token] trained on the bank: scripts/tier2_linear.py
               (extract_features, run_probe, knn_check) unchanged; flipavg averages the whole probe feature
               (flip_avg_baseline.flip_averaged_probe_feature).
  paired       paired bootstrap (mirror/bootstrap.py, 10,000 resamples over the validation images) of healed minus
               flip-averaged original, for k-NN (bf16) and the probe.
  cost         analytic GFLOPs per image (scripts/collect_results.analytic_cost; flipavg: twice the original's;
               healed: its structured cost, as MUST-09), parameters, and with --latency (or --latency-only on a quiet
               GPU) the latency of one forward at batch 1, 64 and 256 (bf16, torch.compile, median of 100 runs after
               20 warm-up runs; scripts/must09_heal.latency_profile) with the peak inference memory. The flip average
               is timed as one forward of the batch and its mirror (2B images, the cheapest implementation) and as
               two passes; the healed model as stored and in its even/odd form (mirror/efficient.py).
Outputs: results/MUST-10/flipavg.csv, latency.csv, MUST10.md, meta.json; runs/MUST-10/<model>/ (row JSONs,
predictions, probe features). Resumable: finished parts with the same settings are skipped (--force redoes them).
"""
from __future__ import annotations

import torch

DEFAULT_GPU_FRAC = 0.2
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_FRAC)  # cap before anything is allocated; --gpu-frac sets the final cap

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.bootstrap import paired_bootstrap  # noqa: E402
from mirror.data import mirror  # noqa: E402
from mirror.heal_structured import load_healed_model  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "MUST-10"


def _load(name: str):
    key = f"_must10_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


M9 = _load("must09_heal")          # latency_profile, cost_of, write_json, upsert_csv, GPU cap handling
FA = _load("flip_avg_baseline")    # knn_eval, FlipAveragedAdapter, flip_averaged_probe_feature
T2 = FA.T2                         # scripts/tier2_linear.py: extract_features, run_probe, knn_check
M9.apply_gpu_cap(DEFAULT_GPU_FRAC)

ROWS = (("original", "plain"), ("original", "flipavg"), ("healed", "plain"))
EXACT_ROWS = {("original", "flipavg"), ("healed", "plain")}     # exactly invariant: scored with fp32 features
FORWARDS = {"plain": 1, "flipavg": 2}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--healed-run", required=True, help="a MUST-09 run folder of mirror_exact (spec.json, healed_state.pt)")
    p.add_argument("--tiny", action="store_true", help="tiny model and synthetic data on CPU (the healed run must be tiny)")
    p.add_argument("--out-root", default=None, help="default runs/MUST-10 (tiny: tiny_out/MUST-10)")
    p.add_argument("--results-root", default=None, help="default results/MUST-10 (tiny: tiny_out/results/MUST-10)")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--seed", type=int, default=0, help="global seed and the probe's weight-decay split")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    p.add_argument("--knn-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256, help="probe feature batch size (tier2_linear's)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wd-grid", default=None, help="probe weight-decay grid (default tier2_linear's)")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--skip-fp32", action="store_true", help="skip the fp32 flip-consistency passes")
    p.add_argument("--fp32-all", action="store_true",
                   help="also the fp32 pass of the plain original (not exactly invariant; by default only the two "
                        "exactly invariant rows get it)")
    p.add_argument("--latency", action="store_true", help="also time the models (needs a quiet GPU)")
    p.add_argument("--latency-only", action="store_true", help="only the latency and memory rows")
    p.add_argument("--latency-batches", default=None, help="default 1,64,256 (tiny 1,4)")
    p.add_argument("--latency-iters", type=int, default=None)
    p.add_argument("--latency-warmup", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


class FlipAveragedFeature(nn.Module):
    """Timing form of the flip average: one forward of the batch and its mirror (2B images), then the mean of the two
    halves of the k-NN feature. two_passes=True runs x and mirror(x) as two forwards instead."""

    def __init__(self, adapter, model: nn.Module, two_passes: bool = False):
        super().__init__()
        self.model, self._adapter, self.two_passes = model, adapter, bool(two_passes)

    def forward(self, x):
        if self.two_passes:
            return 0.5 * (self._adapter.cls_feature(x, self.model) + self._adapter.cls_feature(mirror(x), self.model))
        f = self._adapter.cls_feature(torch.cat([x, mirror(x)], 0), self.model)
        B = x.shape[0]
        return 0.5 * (f[:B] + f[B:])


def data_for(a, tiny: bool, device):
    """(k-NN bank, k-NN queries, probe data, number of classes, protocol)."""
    if tiny:
        data = T2.TinyData(4, 20, 8, seed=a.seed, batch_size=16)
        return data.loader("bank"), data.loader("val"), data, data.num_classes, "tiny synthetic"
    from mirror.protocol import Protocol

    pin = device.type == "cuda"
    proto = Protocol(resolve(a.cache_dir), batch_size=a.knn_batch_size, num_workers=a.num_workers, pin=pin)
    data = T2.ProtocolData(resolve(a.cache_dir), a.batch_size, a.num_workers, pin)
    return proto.bank_loader(), proto.query_loader(), data, 1000, "v2"


def knn_part(adapter, model, bank, query, device, n_cls, flipavg: bool, amp, pred_path: Path) -> dict:
    t0 = time.time()
    r = FA.knn_eval(adapter, bank, query, device, n_cls, 20, 0.07, model=model, amp=amp, flip_average=flipavg)
    M9.save_predictions(pred_path, r)
    return {k: r[k] for k in ("top1", "flip_consistency", "n_bank", "n_query", "feature_flip_max_abs",
                              "feature_identical_pct")} | {"amp": amp or "fp32", "seconds": round(time.time() - t0, 1),
                                                           "predictions": str(pred_path)}


def probe_part(a, adapter, model, data, device, n_cls, flipavg: bool, feat_dir: Path, tag: str, git: dict) -> dict:
    """tier2_linear's probe recipe (features cached per split)."""
    fn = FA.flip_averaged_probe_feature if flipavg else None
    recipe = FA.FLIPAVG_RECIPE if flipavg else T2.FEATURE_RECIPE
    feats = {}
    t0 = time.time()
    for split in T2.SPLITS:
        fp = feat_dir / f"{tag}_{split}.npz"
        expect = {"tag": tag, "recipe": recipe, "split": split, "n": len(data.loader(split).dataset), "amp": "bf16"}
        hit = None if a.force else T2.cached_features(fp, expect)
        if hit is None:
            X, y = T2.extract_features(adapter, data.loader(split), device, model, "bf16", np.float16,
                                       tag=f"{tag} {split}", feature_fn=fn)
            meta = {**expect, "n": int(X.shape[0]), "dim": int(X.shape[1]), "git_hash": git["git_hash"]}
            T2.save_features(fp, X, y, data.rows(split), meta)
            hit = (X, y, meta)
        feats[split] = hit
    (Xb, yb, _), (Xv, yv, _) = feats["bank"], feats["val"]
    grid = [float(v) for v in (a.wd_grid or T2.DEFAULT_WD_GRID).split(",") if v.strip()]
    solver = {"max_iter": 1000, "history_size": 20, "tol_grad": 1e-6}
    pr = T2.run_probe(Xb, yb, Xv, yv, n_cls, grid, 0.1, a.seed, True, device, solver)
    d_first = int(adapter.embed_dim)
    kc = T2.knn_check(Xb, yb, Xv, yv, d_first, n_cls, device)
    with torch.no_grad():                     # per-image probe predictions, for the paired bootstrap
        mean, std = (torch.as_tensor(pr[k]).to(device) for k in ("mean", "std"))
        Xt = T2.standardise(torch.as_tensor(Xv).to(device), mean, std).float()
        W, b = (torch.as_tensor(pr[k]).to(device).float() for k in ("W", "b"))
        pred = (Xt @ W + b).argmax(1).cpu().numpy()           # W [D, C], as tier2_linear.evaluate_probe
    np.savez_compressed(feat_dir / f"{tag}_probe_pred.npz", pred=pred.astype(np.int16), labels=np.asarray(yv, dtype=np.int16))
    T2.free_memory(device)
    test, sel = pr["test"], pr["selection"]
    return {"top1": test["top1"], "top5": test["top5"], "chosen_wd": sel["chosen_wd"], "wd_at_grid_edge": sel["wd_at_grid_edge"],
            "knn_check_top1": kc, "n_bank": int(Xb.shape[0]), "n_val": int(Xv.shape[0]), "recipe": recipe,
            "seconds": round(time.time() - t0, 1), "predictions": str(feat_dir / f"{tag}_probe_pred.npz")}


def correct_of(pred_path) -> np.ndarray:
    with np.load(pred_path) as z:
        return (z["pred"] == z["labels"]).astype(np.float64)


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    tiny = a.tiny
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    M9.apply_gpu_cap(a.gpu_frac)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    run = resolve(a.healed_run)
    with timer.stage("load"):
        adapter, healed, info = load_healed_model(run, device)
        spec = info["spec"]
        if spec["family"] != "mirror" or not spec["base"].get("even_head"):
            raise SystemExit(f"{run} is not a full-conversion Mirror-exact run (family {spec['family']})")
        model_name = spec.get("model") or ("tiny" if tiny else "model")
        out = resolve(a.out_root or ("tiny_out/MUST-10" if tiny else "runs/MUST-10")) / model_name
        res = resolve(a.results_root or ("tiny_out/results/MUST-10" if tiny else "results/MUST-10"))
        out.mkdir(parents=True, exist_ok=True)
        cost_h = M9.cost_of({}, adapter, healed, spec, info.get("structure"))
        n_orig = int(cost_h["params_original"])
        g_orig = float(cost_h["gflops_dense"])
    models = {"original": adapter.model, "healed": healed}
    tag_run = {"healed_run": str(run), "healed_seed": spec.get("seed"), "healed_method": spec.get("method")}
    log(f"MUST-10: model {model_name}, healed run {run}, device {device}, git {git['git_hash'][:10]}")

    rows = []
    if not a.latency_only:
        bank, query, data, n_cls, protocol = data_for(a, tiny, device)
        amp = "bf16" if device.type == "cuda" else None
        for who, mode in ROWS:
            rp = out / "rows" / f"{who}__{mode}.json"
            rec = (None if a.force else M9.read_json(rp)) or {}
            if rec.get("settings", {}).get("healed_run") not in (None, str(run)) and who == "healed":
                rec = {}
            fa = mode == "flipavg"
            m = models[who]
            with timer.stage(f"{who} {mode}"):
                if "knn" not in rec:
                    rec["knn"] = knn_part(adapter, m, bank, query, device, n_cls, fa, amp,
                                          out / "knn" / f"{who}__{mode}_{amp or 'fp32'}.npz")
                    M9.write_json(rp, rec)
                if "knn_fp32" not in rec and not a.skip_fp32 and ((who, mode) in EXACT_ROWS or a.fp32_all):
                    rec["knn_fp32"] = rec["knn"] if amp is None else knn_part(
                        adapter, m, bank, query, device, n_cls, fa, None, out / "knn" / f"{who}__{mode}_fp32.npz")
                    M9.write_json(rp, rec)
                if "probe" not in rec and not a.skip_probe:
                    rec["probe"] = probe_part(a, adapter, m, data, device, n_cls, fa, out / "features",
                                              f"{who}__{mode}", git)
                    M9.write_json(rp, rec)
            gf = (g_orig if who == "original" else float(cost_h["gflops"])) * FORWARDS[mode]
            params = n_orig if who == "original" else int(cost_h["params_model"])
            kn, k32, pr = rec.get("knn") or {}, rec.get("knn_fp32") or {}, rec.get("probe") or {}
            row = {"experiment": EXPERIMENT, "model": model_name, "variant": who, "mode": mode,
                   "forwards_per_image": FORWARDS[mode], "knn_top1": kn.get("top1"), "knn_amp": kn.get("amp"),
                   "flip_consistency": kn.get("flip_consistency"), "knn_top1_fp32": k32.get("top1"),
                   "flip_consistency_fp32": k32.get("flip_consistency"),
                   "feature_flip_max_abs_unit_fp32": k32.get("feature_flip_max_abs"),
                   "feature_identical_pct_fp32": k32.get("feature_identical_pct"), "probe_top1": pr.get("top1"),
                   "probe_top5": pr.get("top5"), "probe_wd": pr.get("chosen_wd"), "gflops_per_image": gf,
                   "params": params, "params_frac": params / n_orig, "n_query": kn.get("n_query"),
                   "protocol": protocol, "seed": a.seed, **tag_run, "git_hash": git["git_hash"],
                   "git_dirty": git.get("git_dirty")}
            rec.update(row=row, settings={"healed_run": str(run), "protocol": protocol, "seed": a.seed})
            M9.write_json(rp, rec)
            rows.append(row)
        comp = compare(out, rows, a)
        M9.upsert_csv(res / "flipavg.csv", rows, ["model", "variant", "mode", "healed_run"])
        M9.write_json(res / f"comparison_{model_name}_s{spec.get('seed')}.json", comp)
        write_md(res / "MUST10.md", rows, comp)
    lat_rows = []
    if a.latency or a.latency_only:
        lat = {"batches": [1, 4] if tiny else [1, 64, 256], "warmup": 1 if tiny else 20, "iters": 3 if tiny else 100,
               "compile": not (tiny or a.no_compile), "compile_mode": "default", "align": 8}
        if a.latency_batches:
            lat["batches"] = [int(b) for b in a.latency_batches.split(",") if b.strip()]
        if a.latency_iters:
            lat["iters"] = a.latency_iters
        if a.latency_warmup is not None:
            lat["warmup"] = a.latency_warmup
        M2 = M9.load_script("m2_latency")
        with timer.stage("latency"):
            fv = M9.even_odd_form(healed, {**info, "align": lat["align"]})
            named = [("original_plain", M2.AdapterFeature(adapter, copy.deepcopy(adapter.model))),
                     ("original_flipavg_batched", FlipAveragedFeature(adapter, copy.deepcopy(adapter.model))),
                     ("original_flipavg_two_passes", FlipAveragedFeature(adapter, copy.deepcopy(adapter.model), True)),
                     ("healed_as_stored", M2.AdapterFeature(adapter, copy.deepcopy(healed))),
                     ("healed_even_odd", fv)]
            lat_rows = M9.latency_profile(named, lat["batches"], lat, device, adapter.img_size, a.seed)
        for r in lat_rows:
            r.update(model=model_name, **tag_run, git_hash=git["git_hash"])
        M9.upsert_csv(res / "latency.csv", lat_rows, ["model", "variant", "batch", "healed_run"])
        M9.write_json(out / "latency.json", {"rows": lat_rows, "protocol": lat, "env": env_info(device),
                                             "nvidia_smi": M2.nvidia_smi() if device.type == "cuda" else "cpu"})
    write_meta(res, experiment=EXPERIMENT, config=vars(a), seeds={"probe_split": a.seed, "healed": spec.get("seed")},
               models={model_name: spec.get("adapter")}, device=device, wall_seconds=time.time() - t_start,
               extra={"healed_run": str(run), "healed_spec_method": spec.get("method"), "timing": timer.stages})
    log(f"MUST-10 done in {timer.total() / 60:.1f} min; results in {res}")
    return {"rows": rows, "latency": lat_rows, "out": str(out), "results": str(res)}


def compare(out: Path, rows: list, a) -> dict:
    """Healed Mirror-exact against the flip-averaged original: differences, cost ratios, paired bootstraps."""
    by = {(r["variant"], r["mode"]): r for r in rows}
    h, f = by[("healed", "plain")], by[("original", "flipavg")]
    comp = {"pair": "healed Mirror-exact (one pass) minus flip-averaged original (two passes)",
            "knn_diff": None if h["knn_top1"] is None or f["knn_top1"] is None else h["knn_top1"] - f["knn_top1"],
            "probe_diff": None if h["probe_top1"] is None or f["probe_top1"] is None else h["probe_top1"] - f["probe_top1"],
            "gflops_ratio": h["gflops_per_image"] / f["gflops_per_image"], "params_ratio": h["params"] / f["params"]}
    for key, sub in (("knn_bootstrap", "knn"), ("probe_bootstrap", "probe")):
        hj, fj = (M9.read_json(out / "rows" / f"{n}.json") or {} for n in ("healed__plain", "original__flipavg"))
        hp, fp_ = (hj.get(sub) or {}).get("predictions"), (fj.get(sub) or {}).get("predictions")
        if hp and fp_ and Path(hp).exists() and Path(fp_).exists():
            comp[key] = paired_bootstrap(correct_of(hp), correct_of(fp_), n_resamples=a.n_boot, seed=a.seed)
    return comp


def _f(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def write_md(path: Path, rows: list, comp: dict) -> None:
    lines = ["# MUST-10: healed Mirror-exact against test-time flip averaging", "",
             "k-NN: 100k train bank, 50k validation queries, k = 20, T = 0.07; top-1 with bf16 features; flip "
             "consistency also with fp32 features for the bank and every query (the measure of exact invariance). "
             "Probe: scripts/tier2_linear.py recipe. GFLOPs: analytic, per image (forward passes times one pass).", "",
             "| model | mode | forward passes | k-NN top-1 | flip consistency (fp32) | flip consistency (k-NN precision) | "
             "linear probe top-1 | GFLOPs per image | parameters (share) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['mode']} | {r['forwards_per_image']} | {_f(r['knn_top1'])} | "
                     f"{_f(r['flip_consistency_fp32'], 3)} | {_f(r['flip_consistency'], 3)} | {_f(r['probe_top1'])} | "
                     f"{_f(r['gflops_per_image'])} | {r['params']:,} ({_f(r['params_frac'], 3)}) |")
    lines += ["", f"Healed minus flip-averaged original: k-NN {_f(comp.get('knn_diff'))} points, probe "
                  f"{_f(comp.get('probe_diff'))} points; GFLOPs ratio {_f(comp['gflops_ratio'], 3)}, parameter ratio "
                  f"{_f(comp['params_ratio'], 3)}."]
    for key in ("knn_bootstrap", "probe_bootstrap"):
        b = comp.get(key)
        if b:
            lines.append(f"Paired bootstrap ({key.split('_')[0]}, n = {b['n']}, {b['n_resamples']} resamples): "
                         f"{_f(b['diff'])} points, 95% interval {_f(b['ci_low'])} to {_f(b['ci_high'])}.")
    if any(r["model"] == "tiny" for r in rows):
        lines += ["", "Tiny rows: CPU smoke test (tiny random ViT, synthetic images); not the protocol above."]
    text = "\n".join(lines) + "\n"
    assert chr(0x2014) not in text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

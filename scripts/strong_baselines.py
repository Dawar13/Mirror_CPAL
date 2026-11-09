#!/usr/bin/env python
"""Strong compression baselines, zero-shot (PAPER_EXPERIMENTS.md, M3-grid methods): activation-aware
SVD, SliceGPT-style PCA slicing and activation-weighted MLP width pruning, each at the parameter budget
of Mirror V-local, MLP only, at the same k.

    python scripts/strong_baselines.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0 \
        --fit-seed 0 --ks 6,9 --methods asvd,slicegpt,prune_act --out runs/sb
    python scripts/strong_baselines.py --tiny --ks 1,2 --out tiny_out/sb          # CPU, synthetic data

What is compressed: the MLPs (fc1 and fc2) of blocks 1..k, nothing else; the same blocks as the Track A
default (scripts/stage1_ablate.py). The budget is the MLP weight count of Mirror V-local, MLP only, at
the same k and fit run (variants.mirror_param_counts: sum over blocks of eq_fc1 + eq_fc2, biases
excluded), so each row's params_vs_mirror_first_k is its exact MLP weight count over the Mirror count;
a non-tiny run stops before any evaluation if a method misses the 1% bar. The methods, what each one
slices or keeps and how its allocation is rounded are documented in mirror/baselines.py.

Statistics: one pass of the fit run's fit images (ImageNet train, role fit<seed>, 2,000 images; no
mirrored copies) through the folded copy of the original model (mirror/fold.py), all tokens, float64
sums: fc1 input second moments and fc2 input second moments (asvd), the residual stream entering each
MLP (slicegpt) and mean |GELU output| per neuron (prune_act). Evaluation: the paper protocol
(mirror/protocol.py: bank = 100 train images per class, queries = all 50k validation images unless
--queries; CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features, flip consistency). The
original model's k-NN comes from cache/knn_original/ as in stage1_ablate.py.

Output (resumable; a rerun skips rows already in the CSV and recomputes the statistics only if a row is
missing): <out>/SB_s<fit seed>.csv in the column format of the Track A CSVs (ablation "SB", variant =
method), <out>/SB_s<fit seed>_meta.json (git hash, seed, arguments, budgets, statistics provenance) and
<out>/SB_SUMMARY.md (every SB_s*.csv in <out>).
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.08)  # shared GPU: cap before anything is allocated

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import baselines as B  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.utils import REPO_ROOT, env_info, git_info, log, set_seed  # noqa: E402

ABLATION = "SB"
KEY = ["ablation", "variant", "k", "fit_seed"]
SEED = 0
LABEL = {"asvd": "activation-aware SVD (SVD-LLM style whitening)",
         "slicegpt": "SliceGPT-style PCA slicing of fc1 input and fc2 output",
         "prune_act": "MLP width pruning, activation-weighted importance"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory, e.g. runs/x2/perlayer_reg_s0")
    p.add_argument("--fit-seed", type=int, default=None, help="must agree with the fit run's fit seed")
    p.add_argument("--ks", default="6,9", help="numbers of converted blocks")
    p.add_argument("--methods", default=",".join(B.METHODS), help=f"comma list from {B.METHODS}")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--queries", type=int, default=None, help="subsample validation queries (default: all)")
    p.add_argument("--asvd-damp", type=float, default=B.DEFAULT_DAMP,
                   help="asvd ridge: lam = damp * mean(diag M) before the Cholesky factorisation")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (tiny fit only)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    return p.parse_args(argv)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def data_checks(proto, fit_seed: int) -> dict:
    """The fit images are train images and share no row with the k-NN bank; queries are validation."""
    fit = np.asarray(proto.index[f"fit{fit_seed}_rows"])
    bank = np.asarray(proto.index["bank_rows"])
    return {"fit_role": f"fit{fit_seed} (train)", "n_fit_images": int(len(fit)),
            "fit_bank_overlap": int(len(np.intersect1d(fit, bank))),
            "queries": "validation" if proto.val is not proto.train else "synthetic (tiny)"}


def write_summary(out: Path, model_name: str, git: dict) -> Path:
    """SB_SUMMARY.md from every SB_s*.csv in `out` (plain language, British spelling)."""
    import pandas as pd

    frames = []
    for f in sorted(out.glob("SB_s*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True) if frames else None

    def counts(col):
        return "/".join(f"{int(v):,}" for v in sorted(df[col].unique())) if df is not None else "?"

    L = ["# Strong baselines, zero-shot (M3-grid methods)", "",
         f"Written by `scripts/strong_baselines.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}, commit "
         f"`{git['git_hash'][:10]}`{' (dirty)' if git['git_dirty'] else ''}. Model `{model_name}`.", "",
         "Each method compresses the MLPs of blocks 1 to k and nothing else, at the parameter budget of Mirror "
         "V-local, MLP only, at the same k: the MLP weight count of the converted blocks (biases excluded). "
         f"Statistics come from the fit run's {counts('n_fit_images')} train fit images (no mirrored copies); k-NN "
         f"uses the paper protocol ({counts('n_bank')} train bank images, {counts('n_query')} validation queries, "
         f"k = 20, T = 0.07, {'fp32 on CPU' if model_name == 'tiny' else 'bf16'} features).", "",
         "- **asvd**: activation-aware SVD in the SVD-LLM style. Each folded fc1 and fc2 is whitened by the "
         "Cholesky factor of its input second moment (with a small ridge), truncated and un-whitened. Ranks are "
         "those of the plain SVD baseline (largest rank whose factors fit each matrix's Mirror count).",
         "- **slicegpt**: SliceGPT-style PCA slicing. Per block, the principal directions of the residual stream "
         "entering the MLP (folded RMSNorm form, mirror/fold.py) are computed; fc1 reads only the top d_in of them "
         "and fc2 writes only into the top d_out, with d_in = floor(Mirror fc1 count / h) and d_out = floor(Mirror "
         "fc2 count / h). The residual stream, the norms, attention and the hidden width are not sliced. The change "
         "of stream basis between blocks (one d x d matrix per block in an implementation) is not counted, as Mirror "
         "V-local's basis rotations are not counted in its MLP budget; it is recorded as basis_change_params.",
         "- **prune_act**: structured width pruning. Neurons are scored by mean |activation| on the fit images times "
         "the norm of their fc2 column (LayerScale included); per block the floor(Mirror block count / (2 d)) "
         "highest-scoring neurons are kept and the others removed.", "",
         "Every allocation rounds down, so a baseline never holds more MLP weights than Mirror. Ratio = MLP weights "
         "kept over the Mirror budget (the column params_vs_mirror_first_k of the CSV).", ""]
    if df is None:
        L += ["No rows yet.", ""]
    else:
        order = {m: i for i, m in enumerate(B.METHODS)}
        df = df.assign(_o=df["variant"].map(order).fillna(99)).sort_values(["fit_seed", "k", "_o"])
        L += ["| method | k | fit seed | top-1 | drop | flip consistency | MLP weights kept | Mirror budget | ratio | "
              "within 1% |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            try:
                info = json.loads(r["info"]) if isinstance(r["info"], str) else {}
            except ValueError:
                info = {}
            budget = info.get("mirror_budget")
            ratio = float(r["params_vs_mirror_first_k"])
            within = "yes" if abs(ratio - 1) <= B.PARAM_TOL else "NO"
            L.append(f"| {r['variant']} | {int(r['k'])} | {int(r['fit_seed'])} | {float(r['top1']):.2f} | "
                     f"{float(r['drop_points']):.2f} | {float(r['flip_consistency']):.2f} | "
                     f"{int(r['mlp_params_kept']):,} | {'' if budget is None else f'{int(budget):,}'} | "
                     f"{ratio:.4f} | {within} |")
        orig = df.iloc[0]
        L += ["", f"Original model: top-1 {float(orig['top1_original']):.2f}, flip consistency "
                  f"{float(orig['flip_original']):.2f} (same protocol; drop = original minus method).", ""]
    L += ["Mirror V-local, MLP only and plain SVD at the same k and fit seed are in "
          f"`runs/a/{model_name}/A1_s<fit seed>.csv` (variants per_layer and svd); the canonical Mirror values are "
          "those of X2 in `results/paper` (`T_main_zeroshot`).", "",
          "Caveats: statistics are taken from the original model for every block (not from a model whose earlier "
          "blocks are already compressed); slicegpt is evaluated in the absorbed form of the unfolded model, which "
          "computes the same function as the sliced folded model (tested in float64) and shares the numerics of the "
          "other variants; prune_act has no bias compensation for the removed neurons.", ""]
    path = out / "SB_SUMMARY.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv=None) -> dict:
    a = parse_args(argv)
    methods = list(dict.fromkeys(m.strip() for m in a.methods.split(",") if m.strip()))
    bad = [m for m in methods if m not in B.METHODS]
    if bad or not methods:
        raise SystemExit(f"unknown method(s) {bad}; choose from {B.METHODS}")
    ks = list(dict.fromkeys(int(k) for k in a.ks.split(",")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    if a.tiny:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    amp = None if a.tiny else "bf16"
    git = git_info()
    model_name = "tiny" if a.tiny else a.model
    log(f"strong baselines {methods} on {model_name}, ks {ks}, device {device}, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")

    adapter, proto, num_classes = V.build_runtime(a.model, a.tiny, device, a.cache_dir, a.batch_size,
                                                  a.stats_batch_size, a.num_workers)
    L_depth = adapter.depth
    struct = V.load_or_make_structure(a.fit_run, a.tiny, adapter, proto, device, out, a.fit_seed, a.top_k)
    if (struct.depth, struct.d, struct.h) != (L_depth, adapter.embed_dim, adapter.mlp_hidden):
        raise SystemExit(f"fit run shape {(struct.depth, struct.d, struct.h)} does not match the model")
    run_model = struct.meta.get("model")
    if not a.tiny and run_model and run_model != a.model:
        raise SystemExit(f"fit run is for {run_model}, not {a.model}")
    fit_seed = V.resolve_fit_seed(a.fit_seed, struct)
    for k in ks:
        if not 1 <= k <= L_depth:
            log(f"skip k = {k} (depth {L_depth})")
    ks = [k for k in ks if 1 <= k <= L_depth]
    if not ks:
        raise SystemExit("no valid k")

    # Budgets first: a non-tiny run never evaluates a variant that misses the 1% bar.
    plans = {(m, k): B.plan_for(m, struct, k) for k in ks for m in methods}
    for (m, k), pl in plans.items():
        log(f"budget {m} k = {k}: {pl.params_kept:,} MLP weights vs Mirror {pl.mirror_budget:,} "
            f"(ratio {pl.ratio:.4f}), allocation {pl.as_dict()['allocation']}")
    missed = [f"{m} k = {k} ({pl.ratio:.4f})" for (m, k), pl in plans.items() if not pl.within_tol]
    if missed and not a.tiny:
        raise SystemExit(f"budget outside 1% of the Mirror count: {missed}")
    if missed:
        log(f"tiny model: budget outside 1% for {missed} (granularity of the tiny widths; real model checked in tests)")

    csv_path = out / f"{ABLATION}_s{fit_seed}.csv"
    csv = V.ResultsCsv(csv_path, KEY)
    todo = []
    for k in ks:
        for m in methods:
            if csv.done((ABLATION, m, k, fit_seed)):
                log(f"{m} k = {k} fit seed {fit_seed}: done, skipped")
            else:
                todo.append((m, k))

    t0 = time.time()
    checks = data_checks(proto, fit_seed)
    if checks["fit_bank_overlap"]:
        raise SystemExit(f"fit rows overlap the k-NN bank ({checks['fit_bank_overlap']} rows)")
    orig, stats_info = None, None
    n_new = 0
    if todo:
        bank, query = proto.bank_loader(), proto.query_loader(a.queries)
        orig = V.original_knn(adapter, bank, query, device, num_classes, model_name, amp, a.cache_dir,
                              a.queries, a.tiny, git["git_hash"])
        log(f"original: top-1 {orig['top1']:.2f}, flip consistency {orig['flip_consistency']:.2f} ({orig['source']})")
        need = B.needs_for(sorted({m for m, _ in todo}))
        max_k = max(k for _, k in todo)
        ts = time.time()
        log(f"statistics {sorted(need)} for blocks 1..{max_k} from fit{fit_seed} (train, folded model)")
        stats = B.collect_baseline_stats(adapter, proto.fit_loader(fit_seed), device, range(max_k), need)
        stats_info = {**stats.info, "time_s": round(time.time() - ts, 1)}
        log(f"statistics: {stats.n_images} images, {stats.n_tokens} tokens, {stats_info['time_s']} s")
        _free(device)
        n_fit = int(len(proto.index[f"fit{fit_seed}_rows"]))
        for m, k in todo:
            t = time.time()
            pl = plans[(m, k)]
            model, per_block = B.baseline_copy(adapter.model, m, pl.blocks, pl.alloc, stats, a.asvd_damp)
            sig = B.weights_signature(model, m, pl.blocks, pl.alloc)
            r = evaluate_knn(adapter, bank, query, device, num_classes, k=20, temperature=0.07, model=model, amp=amp)
            del model
            _free(device)
            h, d = struct.h, struct.d
            info = {"baseline": LABEL[m], **pl.as_dict(),
                    "bias_params": B.bias_params(m, h, d, pl.alloc),
                    "per_block": per_block, "statistics": f"fit{fit_seed} train images, folded model, "
                                                          f"{stats.n_tokens} tokens"}
            if m == "asvd":
                info["ridge_damp"] = a.asvd_damp
                info["ranks_as_plain_svd"] = True
            if m == "slicegpt":
                info["basis_change_params"] = len(pl.blocks) * d * d
                info["basis_change_note"] = ("one d x d change of stream basis per converted block, not counted in "
                                             "the MLP budget (as Mirror V-local's basis rotations)")
                info["evaluated_form"] = "absorbed weights in the unfolded model (same function as the folded form)"
            row = {
                "model": model_name, "ablation": ABLATION, "variant": m, "k": k, "fit_seed": fit_seed,
                "blocks": " ".join(str(b + 1) for b in pl.blocks), "method": m, "projection": "", "alpha": "",
                "top1": r["top1"], "flip_consistency": r["flip_consistency"],
                "top1_original": orig["top1"], "flip_original": orig["flip_consistency"],
                "drop_points": orig["top1"] - r["top1"],
                "mlp_params_kept": pl.params_kept, "mlp_params_dense": pl.params_dense,
                "mlp_params_kept_frac": pl.params_kept / pl.params_dense,
                "params_exact": True,
                "params_vs_mirror_first_k": pl.ratio,
                "n_fit_images": n_fit, "n_query": r["n_query"], "n_bank": r["n_bank"],
                "variant_seed": "", "seed": SEED, "fit_run": struct.source, "git_hash": git["git_hash"],
                "git_dirty": git["git_dirty"], "time_s": round(time.time() - t, 1),
                "signature": sig, "reused_from": "",
                "info": json.dumps(info, default=str),
            }
            csv.add(row)
            n_new += 1
            log(f"{m} k = {k}: top-1 {r['top1']:.2f} (drop {row['drop_points']:.2f}), flip "
                f"{r['flip_consistency']:.2f}, MLP weights {pl.params_kept:,} = {pl.ratio:.4f} of Mirror")
        del stats
        _free(device)

    meta = {
        "experiment": "strong baselines, zero-shot (M3-grid methods)", "model": model_name, "ks": ks,
        "methods": methods, "fit_seed": fit_seed, "seed": SEED, "git": git, "env": env_info(device),
        "args": vars(a), "fit_run": struct.source, "structure_meta": struct.meta, "original_knn": orig,
        "knn": {"k": 20, "temperature": 0.07, "amp": amp}, "budgets": {f"{m}_k{k}": pl.as_dict()
                                                                    for (m, k), pl in plans.items()},
        "param_tolerance": B.PARAM_TOL, "data_checks": checks, "statistics": stats_info,
        "asvd_damp": a.asvd_damp, "rows_added": n_new, "rows_total": len(csv.rows),
        "elapsed_s": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / f"{ABLATION}_s{fit_seed}_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    summ = write_summary(out, model_name, git)
    log(f"{n_new} new rows, {len(csv.rows)} in {csv_path}; summary {summ}")
    return {"csv": str(csv_path), "rows_added": n_new, "rows_total": len(csv.rows)}


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""X1: is the SVD collapse on DINOv2-reg (E1b, 8.08% at k = 6) a bug or a real property?

Checks (PAPER_EXPERIMENTS.md X1), for dinov2_vitb14_reg and, for contrast, dinov2_vitb14:
  1. full-rank SVD (r = min(h, d)) on all blocks leaves CLS features unchanged;
  2. folding (norm2 affine, LayerScale) is exact per block for both models (register tokens included);
  3. the per-matrix SVD ranks used in E1b give r (h + d) <= the matched parameter count, within 1%;
  4. precision: everything runs in fp32 (no fp16 anywhere); features after SVD are finite;
plus diagnostics:
  5. weight energy discarded by SVD at the E1b ranks, per block, for both models (folded weights);
  6. single-block SVD at the E1b ranks: k-NN top-1 drop per block (small val protocol: 10 bank
     images per class, 2 queries per class, split seed 1), to locate the collapse.

    python scripts/x1_svd_check.py --out runs/x1
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.1)  # shared GPU

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import Dinov2Adapter  # noqa: E402
from mirror.data import load_val_cache, make_loader, make_splits, normalize  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.surgery import absorb_block, apply_svd_baseline, fold_mlp  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

RUNS = {"dinov2_vitb14_reg": "runs/e1b_perlayer_b14reg", "dinov2_vitb14": "runs/e1b_perlayer_b14"}


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/x1")
    p.add_argument("--cache", default=str(REPO_ROOT / "cache"))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _, _ = load_val_cache(a.cache)
    small = make_splits(labels, {"bank": 10, "query": 2}, seed=1)
    bank = make_loader(images, labels, small["bank"], 128, 4, True)
    query = make_loader(images, labels, small["query"], 128, 4, True)
    x_check = normalize(torch.from_numpy(np.ascontiguousarray(images[small["query"][:64]])).permute(0, 3, 1, 2).to(dev))

    report = {"git": git_info(), "models": {}}
    for name, run in RUNS.items():
        log(f"== {name}")
        ad = Dinov2Adapter(name=name).to(device=dev)
        L, d, h = ad.depth, ad.embed_dim, ad.mlp_hidden
        mlp = pd.read_csv(Path(run) / "results" / "e1_mlp.csv")
        r1 = mlp["svd_rank"].astype(int).tolist()
        r2 = mlp["svd_rank_fc2"].astype(int).tolist()
        res = {"n_prefix_tokens": ad.n_prefix_tokens, "dtype": str(ad.dtype)}

        # 1. full rank on every block
        f0 = ad.cls_feature(x_check)
        m = copy.deepcopy(ad.model)
        apply_svd_baseline(m, list(range(L)), [min(h, d)] * L)
        res["full_rank_max_abs_diff"] = float((ad.cls_feature(x_check, m) - f0).abs().max())
        del m

        # 2. folding exact per block on real tokens
        t = ad.layer0_tokens(x_check)
        fold = []
        for blk in ad.model.blocks:
            b2 = copy.deepcopy(blk)
            absorb_block(b2)
            o = blk(t)
            fold.append(float(((o - b2(t)).abs().max() / o.abs().max())))
            t = o
        res["fold_max_rel_diff_per_block"] = fold

        # 3. parameter match
        eq1, eq2 = mlp["eq_params_fc1"].to_numpy(), mlp["eq_params_fc2"].to_numpy()
        res["rank_param_ratio_fc1"] = (np.array(r1) * (h + d) / eq1).round(4).tolist()
        res["rank_param_ratio_fc2"] = (np.array(r2) * (h + d) / eq2).round(4).tolist()
        res["ranks_fc1"], res["ranks_fc2"] = r1, r2

        # 4 + 5. finite features after the E1b SVD at k = 6, and discarded energy per block
        m = copy.deepcopy(ad.model)
        apply_svd_baseline(m, list(range(6)), list(zip(r1[:6], r2[:6])))
        res["svd_k6_features_finite"] = bool(torch.isfinite(ad.cls_feature(x_check, m)).all())
        del m
        e1, e2 = [], []
        for b, blk in enumerate(ad.model.blocks):
            f = fold_mlp(blk)
            for W, r, acc in ((f["W1"], r1[b], e1), (f["W2"], r2[b], e2)):
                s2 = torch.linalg.svdvals(W) ** 2
                acc.append(float(1 - s2[:r].sum() / s2.sum()))
        res["svd_discarded_energy_fc1"] = [round(x, 4) for x in e1]
        res["svd_discarded_energy_fc2"] = [round(x, 4) for x in e2]

        # 6. single-block SVD k-NN on the small protocol
        base = evaluate_knn(ad, bank, query, dev, 1000, 20, 0.07)["top1"]
        drops = []
        for b in range(L):
            m = copy.deepcopy(ad.model)
            apply_svd_baseline(m, [b], [(r1[b], r2[b])])
            t1 = evaluate_knn(ad, bank, query, dev, 1000, 20, 0.07, model=m)["top1"]
            drops.append(round(base - t1, 2))
            log(f"{name} block {b + 1}: SVD single-block drop {base - t1:.2f}")
            del m
        res["small_knn_top1_original"] = base
        res["single_block_svd_drop"] = drops
        report["models"][name] = res
        del ad
        torch.cuda.empty_cache()
        (out / "x1_svd_check.json").write_text(json.dumps(report, indent=2))
    log("X1 done")


if __name__ == "__main__":
    main()

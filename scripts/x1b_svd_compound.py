#!/usr/bin/env python
"""X1b: where does multi-block SVD collapse on DINOv2-reg? (follow-up to X1, gate G1)

For dinov2_vitb14_reg and dinov2_vitb14, apply SVD (E1b per-matrix ranks) to the first k blocks,
k = 1..6, and record on the small protocol (split seed 1: 10 bank and 2 query images per class):
k-NN top-1, and the mean output-token norm (before the final norm) of CLS, register and patch
tokens, compared with the original model.

    python scripts/x1b_svd_compound.py --out runs/x1
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
from mirror.surgery import apply_svd_baseline  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

RUNS = {"dinov2_vitb14_reg": "runs/e1b_perlayer_b14reg", "dinov2_vitb14": "runs/e1b_perlayer_b14"}


@torch.no_grad()
def token_norms(ad, x, model=None):
    t = ad.forward_tokens(x, model).float()
    n = t.norm(dim=-1)  # [B, T]
    p = ad.n_prefix_tokens
    return {"cls": float(n[:, 0].mean()), "registers": float(n[:, 1:p].mean()) if p > 1 else None,
            "patch": float(n[:, p:].mean()), "patch_max": float(n[:, p:].max())}


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/x1")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _, _ = load_val_cache(REPO_ROOT / "cache")
    small = make_splits(labels, {"bank": 10, "query": 2}, seed=1)
    bank = make_loader(images, labels, small["bank"], 128, 4, True)
    query = make_loader(images, labels, small["query"], 128, 4, True)
    x = normalize(torch.from_numpy(np.ascontiguousarray(images[small["query"][:64]])).permute(0, 3, 1, 2).to(dev))
    report = {"git": git_info(), "models": {}}
    for name, run in RUNS.items():
        ad = Dinov2Adapter(name=name).to(device=dev)
        mlp = pd.read_csv(Path(run) / "results" / "e1_mlp.csv")
        ranks = list(zip(mlp["svd_rank"].astype(int), mlp["svd_rank_fc2"].astype(int)))
        rows = [{"k": 0, "top1": evaluate_knn(ad, bank, query, dev, 1000, 20, 0.07)["top1"], **token_norms(ad, x)}]
        for k in range(1, 7):
            m = copy.deepcopy(ad.model)
            apply_svd_baseline(m, list(range(k)), ranks[:k])
            rows.append({"k": k, "top1": evaluate_knn(ad, bank, query, dev, 1000, 20, 0.07, model=m)["top1"],
                         **token_norms(ad, x, m)})
            log(f"{name} k={k}: {rows[-1]}")
            del m
            torch.cuda.empty_cache()
        report["models"][name] = rows
        del ad
        (out / "x1b_svd_compound.json").write_text(json.dumps(report, indent=2))
    log("X1b done")


if __name__ == "__main__":
    main()

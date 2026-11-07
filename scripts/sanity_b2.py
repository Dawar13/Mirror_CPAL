#!/usr/bin/env python
"""B2: sanity checks on the real model before E1.

1. Folding: block 0 in absorbed form against the original, max abs difference < 1e-4 (fp32).
   Every other block is checked too and reported.
2. Identity surgery (A = I, P = I on all blocks): k-NN top-1 identical to the original.

    python scripts/sanity_b2.py --config configs/e1.yaml [--tiny]
Exit code 0 when both pass. Writes results/sanity_b2.json.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: cap before anything is allocated

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import build_adapter  # noqa: E402
from mirror.data import get_data, make_loader, normalize  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.surgery import absorb_block, apply_surgery  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, load_config, log, set_seed, setup_torch  # noqa: E402

FOLD_TOL = 1e-4


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "e1.yaml"))
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out-root", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, tiny=args.tiny)
    out_root = Path(args.out_root) if args.out_root else (REPO_ROOT / "tiny_out" if args.tiny else REPO_ROOT)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))
    device = setup_torch(cfg)

    adapter = build_adapter(cfg, device)
    images, labels, splits, num_classes = get_data(cfg)
    comp = cfg["compute"]
    nw, pin = int(comp.get("num_workers", 0)), device.type == "cuda"
    fit_loader = make_loader(images, labels, splits["fit"], int(comp["stats_batch_size"]), nw, pin)
    bank = make_loader(images, labels, splits["bank"], int(comp["batch_size"]), nw, pin)
    query = make_loader(images, labels, splits["query"], int(comp["batch_size"]), nw, pin)

    # 1. folding, per block, on tokens that really reach that block
    imgs, _ = next(iter(fit_loader))
    t = adapter.layer0_tokens(normalize(imgs.to(device), adapter.dtype))
    fold = []
    for blk in adapter.model.blocks:
        folded = copy.deepcopy(blk)
        absorb_block(folded)
        out = blk(t)
        fold.append(float((out - folded(t)).abs().max()))
        t = out
    fold_ok = fold[0] < FOLD_TOL
    log(f"folding max abs diff per block: {[f'{v:.2e}' for v in fold]} -> block 0 "
        f"{'PASS' if fold_ok else 'FAIL'} (< {FOLD_TOL:g})")

    # 2. identity surgery on all blocks
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    m = copy.deepcopy(adapter.model)
    apply_surgery(m, list(range(L)), np.eye(d), [np.arange(h)] * L)
    kc = cfg["knn"]
    ref = evaluate_knn(adapter, bank, query, device, num_classes, int(kc["k"]), float(kc["temperature"]))
    new = evaluate_knn(adapter, bank, query, device, num_classes, int(kc["k"]), float(kc["temperature"]), model=m)
    n_diff = int((ref["pred"] != new["pred"]).sum())
    id_ok = ref["top1"] == new["top1"]
    log(f"identity surgery: top-1 {ref['top1']:.3f} -> {new['top1']:.3f}, flip consistency "
        f"{ref['flip_consistency']:.3f} -> {new['flip_consistency']:.3f}, changed predictions {n_diff} -> "
        f"{'PASS' if id_ok else 'FAIL'}")

    out = {"git": git_info(), "seed": int(cfg["seed"]), "tiny": bool(args.tiny),
           "fold_max_abs_diff_per_block": fold, "fold_tol": FOLD_TOL, "fold_pass": fold_ok,
           "identity_top1": [ref["top1"], new["top1"]],
           "identity_flip_consistency": [ref["flip_consistency"], new["flip_consistency"]],
           "identity_changed_predictions": n_diff, "identity_pass": id_ok}
    with open(out_root / "results" / "sanity_b2.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    ok = fold_ok and id_ok
    log("B2 SANITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

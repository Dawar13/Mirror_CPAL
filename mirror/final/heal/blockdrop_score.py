"""The score that chooses which blocks the block-dropped dense student removes (FINAL_PHASE_PLAN.md A04:
"remove 3 of the 12 blocks, chosen greedily by the smallest k-NN loss on the method-selection set").

`msel_scorer(ctx)` is the factory scripts/fp_heal.py calls through the job spec key
`blockdrop_score_fn: "mirror.final.heal.blockdrop_score:msel_scorer"`. It returns score_fn(model) -> float, the
bf16 k-NN top-1 (%) of the 10,000 method-selection train images (cache/msel224_uint8.npy, cache/msel_index.npz)
against the 100,000-image k-NN bank (cache role `bank`), with the protocol of every other k-NN number (k = 20,
T = 0.07, the class token after the final norm); higher is better. Validation images are never used for this choice
(FINAL_PHASE_PLAN.md Section 11). The images are copied to the device once and reused for every candidate model
(mirror/final/eval/knn_fast.py ImageSet and run_pass).

Without the method-selection cache (CPU dry runs) the factory falls back to synthetic images, and says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["msel_scorer", "msel_eval_data"]


def msel_eval_data(cache_dir, device, log=None):
    """EvalData with the k-NN bank as the bank and the method-selection set as the queries."""
    from mirror.final.eval.knn_fast import EvalData, ImageSet

    cache = Path(cache_dir)
    train = np.load(cache / "train224_uint8.npy", mmap_mode="r")
    with np.load(cache / "train_index.npz") as z:
        rows = np.asarray(z["bank_rows"], dtype=np.int64)
        labels = np.asarray(z["bank_labels"], dtype=np.int64)
    msel = np.load(cache / "msel224_uint8.npy", mmap_mode="r")
    with np.load(cache / "msel_index.npz") as z:
        msel_labels = np.asarray(z["msel_labels"], dtype=np.int64)
    bank = ImageSet.from_array(train, rows, labels, device, False, 1024, "k-NN bank", log)
    queries = ImageSet.from_array(msel, None, msel_labels, device, False, 1024, "method-selection queries", log)
    n_cls = int(max(labels.max(), msel_labels.max())) + 1
    return EvalData(bank, queries, n_cls, f"cache {cache.as_posix()} (bank: bank_rows, {len(bank)} images; queries: "
                                          f"method-selection set, {len(queries)} images)")


def msel_scorer(ctx, batch_size: int = 512):
    """score_fn(model) -> bf16 k-NN top-1 (%) on the method-selection queries (see the module docstring)."""
    import torch

    from mirror.final.eval.knn_fast import EvalData, run_pass
    from mirror.utils import log

    device = torch.device(getattr(ctx, "device", "cpu"))
    cache = Path(getattr(ctx, "cache_dir", "cache") or "cache")
    if (cache / "msel224_uint8.npy").exists() and (cache / "train224_uint8.npy").exists():
        data = msel_eval_data(cache, device, log=log)
        pass_name = "bf16" if device.type == "cuda" else "fp32"
    else:
        size = int(getattr(ctx.adapter, "img_size", 56))
        data = EvalData.synthetic(num_classes=4, bank_per_class=8, val_per_class=4, img_size=size, seed=0,
                                  device=device)
        pass_name = "fp32"
        log(f"blockdrop score: no method-selection cache under {cache}; synthetic images (dry run only)")
    adapter = ctx.adapter

    def score_fn(model) -> float:
        was = model.training
        model.eval()
        try:
            res = run_pass(adapter, model, data, pass_name, batch_size=batch_size, device=device)
        finally:
            model.train(was)
        return float(res["summary"]["top1"])

    score_fn.describe = {"queries": "method-selection set (msel), never validation", "pass": pass_name,
                         "source": data.source}
    return score_fn

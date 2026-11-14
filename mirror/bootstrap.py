"""Paired bootstrap for differences between methods on the same images (PROJECT_NEXT_PHASE.md, Section 4.9).

Both methods are scored on the same n images (for example the 50k validation queries of the k-NN protocol,
`evaluate_knn(...)["pred"] == labels`). The bootstrap resamples images with replacement, 10,000 times by
default, and reports the mean difference (A minus B, in percentage points) with its 95% percentile interval.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(correct_a, correct_b, n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05,
                     chunk: int = 100) -> dict:
    """correct_a, correct_b: per-image 0/1 (or real-valued) scores on the same images, same order."""
    a = np.asarray(correct_a, dtype=np.float64).ravel()
    b = np.asarray(correct_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs equal lengths, got {a.shape} and {b.shape}")
    n = a.size
    diff = a - b
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for s in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - s)
        idx = rng.integers(0, n, size=(m, n))
        stats[s:s + m] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "mean_a": 100.0 * float(a.mean()),
        "mean_b": 100.0 * float(b.mean()),
        "diff": 100.0 * float(diff.mean()),
        "ci_low": 100.0 * float(lo),
        "ci_high": 100.0 * float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }

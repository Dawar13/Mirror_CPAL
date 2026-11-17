"""Paired bootstrap of healed Mirror-exact against the healed baselines (MUST-09), from the stored per-image predictions.

Reads runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/knn/step780_bf16.npz (k-NN predictions and labels of the 50k
ImageNet validation images at the end of healing, the headline protocol) and writes results/paper_stats/
healed_bootstrap.json. For each comparison: the per-image correctness of each method is averaged over its healing
seeds, and the difference of means is bootstrapped over images (10,000 resamples, seed 0) with
mirror.bootstrap.paired_bootstrap. Seed-by-seed comparisons are also reported. CPU only.

    .venv/Scripts/python.exe scripts/paper_bootstrap_healed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirror.bootstrap import paired_bootstrap  # noqa: E402

RUN = ROOT / "runs" / "MUST-09" / "dinov2_vitb14_reg"
OUT = ROOT / "results" / "paper_stats" / "healed_bootstrap.json"


def correct(method: str, seed: int):
    p = RUN / method / f"s{seed}" / "knn" / "step780_bf16.npz"
    if not p.exists():
        return None, None
    z = np.load(p)
    return (z["pred"] == z["labels"]).astype(np.float64), z["labels"]


def main() -> None:
    seeds = {m: [s for s in range(3) if (RUN / m / f"s{s}" / "knn" / "step780_bf16.npz").exists()]
             for m in ("mirror_exact", "svdllm", "slicegpt", "svd", "random_structure")}
    ref_labels = None
    mean_correct = {}
    for m, ss in seeds.items():
        cs = []
        for s in ss:
            c, lab = correct(m, s)
            if ref_labels is None:
                ref_labels = lab
            if not np.array_equal(lab, ref_labels):
                raise SystemExit(f"labels differ for {m} s{s}")
            cs.append(c)
        mean_correct[m] = np.mean(cs, axis=0)
        print(f"{m}: seeds {ss}, mean top-1 {100 * mean_correct[m].mean():.3f}")
    out = {"protocol": "k-NN top-1, bf16 features, step 780, 50k ImageNet validation images",
           "seeds": seeds, "n_resamples": 10000, "bootstrap_seed": 0, "comparisons": {}}
    for b in ("svdllm", "slicegpt", "svd", "random_structure"):
        r = paired_bootstrap(mean_correct["mirror_exact"], mean_correct[b], n_resamples=10_000, seed=0)
        out["comparisons"][f"mirror_exact_minus_{b}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                         for k, v in r.items()}
        print(f"Mirror-exact minus {b}: {r}")
    per_seed = {}
    for s in seeds["svdllm"]:
        if s in seeds["mirror_exact"]:
            a, _ = correct("mirror_exact", s)
            c, _ = correct("svdllm", s)
            r = paired_bootstrap(a, c, n_resamples=10_000, seed=0)
            per_seed[f"s{s}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
            print(f"seed {s}: Mirror-exact minus SVD-LLM {r}")
    out["mirror_exact_minus_svdllm_per_seed"] = per_seed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

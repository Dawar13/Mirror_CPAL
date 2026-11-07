#!/usr/bin/env python
"""POST-HOC DIAGNOSTIC for E1 (not part of the pre-registered decision).

E1 fits one shared mirror operator from C summed over layers 1 to 12. This script asks whether the
MLP twin structure holds with each layer's own operator A_l instead, using the saved operators and
neuron pairings (results/e1_operators.npz); no new statistics pass is needed.

For block b (1-based): fc1 is tested against A_in = A_(b-1) (the block's input layer) and fc2
against A_out = A_b (its output layer), so an equivariant MLP maps the input mirror to the output
mirror: MLP(x A_in) = MLP(x) A_out.

1. Breaking shares s1, s2 per block, against a matched null (random involutions with the same d+
   per layer, random pairings with the same numbers of pairs).
2. With --knn: zero-shot surgery on the first k blocks with the per-layer operators, against
   rank-matched SVD (ranks matched to the per-layer parameter counts) and the matched null, scored
   by the same k-NN protocol.

Writes results/e1_diag_perlayer.csv, results/e1_diag_perlayer_surgery.csv and
plots/diag_perlayer_breaking_share.png.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: cap before anything is allocated

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import build_adapter  # noqa: E402
from mirror.data import get_data, make_loader  # noqa: E402
from mirror.fit import involution_trace, random_involution  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402
from mirror.report import C_BLUE, C_INK, C_NULL, C_ORANGE, _style  # noqa: E402
from mirror.surgery import (apply_surgery_two, breaking_share, equivariant_param_count, fold_mlp,  # noqa: E402
                            perm_trace, project_mlp_two, svd_truncate, write_absorbed_mlp)
from mirror.utils import REPO_ROOT, git_info, load_config, log, set_seed, setup_torch  # noqa: E402


def shares(block, A_in, A_out, perm):
    f = fold_mlp(block)
    e = project_mlp_two(**f, A_in=A_in, A_out=A_out, perm=perm)
    return breaking_share(f["W1"], e["W1"]), breaking_share(f["W2"], e["W2"])


@torch.no_grad()
def svd_two(model, blocks, ranks1, ranks2):
    for b, r1, r2 in zip(blocks, ranks1, ranks2):
        blk = model.blocks[b]
        f = fold_mlp(blk)
        write_absorbed_mlp(blk, svd_truncate(f["W1"], int(r1)), f["b1"], svd_truncate(f["W2"], int(r2)), f["b2"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "e1.yaml"))
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out-root", default=None)
    p.add_argument("--knn", action="store_true", help="also run per-layer surgery k-NN")
    p.add_argument("--ks", default="6,3")
    a = p.parse_args(argv)
    cfg = load_config(a.config, tiny=a.tiny)
    root = Path(a.out_root) if a.out_root else (REPO_ROOT / "tiny_out" if a.tiny else REPO_ROOT)
    res_dir, plot_dir = root / "results", root / "plots"
    seed = int(cfg["seed"])
    set_seed(seed)
    device = setup_torch(cfg)
    git = git_info()
    adapter = build_adapter(cfg, device)
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    with np.load(res_dir / "e1_operators.npz") as z:
        A_layers = z["A_layers"].astype(np.float64)
        perms = z["perms"].astype(np.int64)
    # Re-symmetrise after the float32 round trip (keeps them exact involutions up to 1e-7).
    A_layers = 0.5 * (A_layers + np.transpose(A_layers, (0, 2, 1)))
    d_plus = [int((np.linalg.eigvalsh(A) > 0).sum()) for A in A_layers]
    n_pairs = [int((pp != np.arange(h)).sum() // 2) for pp in perms]
    nulls_A = [random_involution(d, d_plus[l], seed=seed + 7000 + l) for l in range(L + 1)]
    nulls_P = [random_pairing(h, n_pairs[b], seed=seed + 8000 + b) for b in range(L)]

    rows = []
    for b in range(L):
        blk = adapter.model.blocks[b]
        s1, s2 = shares(blk, A_layers[b], A_layers[b + 1], perms[b])
        n1, n2 = shares(blk, nulls_A[b], nulls_A[b + 1], nulls_P[b])
        trP = perm_trace(perms[b])
        eq1 = equivariant_param_count(h, d, trP, involution_trace(A_layers[b]))
        eq2 = equivariant_param_count(h, d, trP, involution_trace(A_layers[b + 1]))
        rows.append({"block": b + 1, "dminus_in": d - d_plus[b], "dminus_out": d - d_plus[b + 1],
                     "twin_frac": 2 * n_pairs[b] / h, "s1": s1, "s2": s2, "s1_null": n1, "s2_null": n2,
                     "s1_over_null": s1 / n1, "s2_over_null": s2 / n2,
                     "eq_params_fc1": eq1, "eq_params_fc2": eq2,
                     "param_saving_frac": 1 - (eq1 + eq2) / (2 * h * d),
                     "git_hash": git["git_hash"], "seed": seed})
        log(f"block {b + 1}: s1 {s1:.3f} (null {n1:.3f}), s2 {s2:.3f} (null {n2:.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "e1_diag_perlayer.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(df["block"], df["s1"], "-", color=C_BLUE, marker="o", markersize=5, linewidth=2, label="fc1 (s1), per-layer A")
    ax.plot(df["block"], df["s2"], "-", color=C_ORANGE, marker="s", markersize=5, linewidth=2, label="fc2 (s2), per-layer A")
    ax.plot(df["block"], df["s1_null"], "--", color=C_NULL, linewidth=2, label="matched null, fc1")
    ax.plot(df["block"], df["s2_null"], ":", color=C_NULL, linewidth=2, label="matched null, fc2")
    ax.set_xlabel("block")
    ax.set_ylabel("breaking share")
    ax.set_ylim(0, max(0.6, float(df[["s1", "s2", "s1_null", "s2_null"]].to_numpy().max()) * 1.1))
    ax.set_title("Diagnostic: breaking share with per-layer operators", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "diag_perlayer_breaking_share.png", dpi=150)
    plt.close(fig)

    if a.knn:
        images, labels, splits, num_classes = get_data(cfg)
        comp, kc = cfg["compute"], cfg["knn"]
        nw, pin = int(comp.get("num_workers", 0)), device.type == "cuda"
        bank = make_loader(images, labels, splits["bank"], int(comp["batch_size"]), nw, pin)
        query = make_loader(images, labels, splits["query"], int(comp["batch_size"]), nw, pin)

        def knn(model=None):
            o = evaluate_knn(adapter, bank, query, device, num_classes, int(kc["k"]), float(kc["temperature"]), model=model)
            return o["top1"], o["flip_consistency"]

        t0, f0 = knn()
        srows = [{"k": 0, "variant": "original", "top1": t0, "flip_consistency": f0, "drop_points": 0.0}]
        for k in [int(x) for x in a.ks.split(",") if int(x) <= L]:
            bl = list(range(k))
            r1 = [int(np.floor(df["eq_params_fc1"].iloc[b] / (h + d))) for b in bl]
            r2 = [int(np.floor(df["eq_params_fc2"].iloc[b] / (h + d))) for b in bl]
            variants = {
                "surgery_perlayer": lambda m: apply_surgery_two(m, bl, [A_layers[b] for b in bl],
                                                               [A_layers[b + 1] for b in bl], [perms[b] for b in bl]),
                "svd_perlayer_matched": lambda m: svd_two(m, bl, r1, r2),
                "random_perlayer": lambda m: apply_surgery_two(m, bl, [nulls_A[b] for b in bl],
                                                              [nulls_A[b + 1] for b in bl], [nulls_P[b] for b in bl]),
            }
            for name, fn in variants.items():
                m = copy.deepcopy(adapter.model)
                fn(m)
                t1, f1 = knn(m)
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                srows.append({"k": k, "variant": name, "top1": t1, "flip_consistency": f1, "drop_points": t0 - t1})
                log(f"k = {k} {name}: top-1 {t1:.2f}, flip consistency {f1:.2f}")
                pd.DataFrame(srows).assign(git_hash=git["git_hash"], seed=seed).to_csv(
                    res_dir / "e1_diag_perlayer_surgery.csv", index=False)
    with open(res_dir / "e1_diag_perlayer_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "seed": seed, "knn": a.knn,
                   "ks": a.ks, "note": "post-hoc diagnostic, not part of the pre-registered E1 decision"}, fh, indent=2)
    log("diagnostic done")


if __name__ == "__main__":
    main()

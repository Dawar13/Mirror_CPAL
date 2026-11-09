#!/usr/bin/env python
"""E3 budget table (PREREG_E3.md): per block, the matrix parameters V-exact keeps at k (MLP and attention),
the SVD ranks matched to them and the SVD counts, and a check that the random-structure control keeps the
same counts. CPU only, no model weights: the dimensions are read from the structure files.

    python scripts/e3_budget.py                                   # runs/m1/m1_structure.npz, X2 fit seed 0, k = 12
    python scripts/e3_budget.py --heads <m1_structure.npz> --operators <e1_operators.npz> --k 12 --json out.json

Accounting: mirror.heal_vexact.vexact_block_params (the per-block sums behind vexact_param_counts, which
gives the 64,554,464 kept parameters of runs/m3_vexact_5090/k12/summary.json) and matched_svd_ranks (per
matrix the largest rank r with r (rows + cols) <= the V-exact count), the same functions scripts/heal.py
uses for configs/e3_svd_*.yaml. Random control: mirror.m1.matched_random_span_structure with the seed of
configs/e3_random.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.attention import HeadMaps  # noqa: E402
from mirror.heal import exact_involution  # noqa: E402
from mirror.heal_vexact import matched_svd_ranks, vexact_block_params  # noqa: E402
from mirror.m1 import matched_random_span_structure  # noqa: E402
from mirror.utils import REPO_ROOT, git_info  # noqa: E402


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(heads_path, operators_path, k: int, random_seed: int = 1) -> dict:
    with np.load(heads_path) as z:
        ks = [int(v) for v in z["ks"]]
        if k not in ks:
            raise ValueError(f"{heads_path} holds span operators for k in {ks}, not {k}")
        A = exact_involution(z["A_star"][ks.index(k)])
        reg = np.asarray(z["reg_perm"][ks.index(k)], dtype=np.int64)
        heads = [HeadMaps(sigma=np.asarray(z["sigma"][b], dtype=np.int64), B=z["B"][b], C=z["C"][b])
                 for b in range(len(z["sigma"]))]
    with np.load(operators_path) as z:
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
    d, h = A.shape[0], len(perms[0])
    D = heads[0].H * heads[0].dh
    per = vexact_block_params(d, h, A, perms[:k], heads[:k])
    rows = matched_svd_ranks(per, h, d, D)
    r = matched_random_span_structure(A, perms, heads, reg, random_seed)
    per_r = vexact_block_params(d, h, r["A_star"], r["perms"][:k], r["heads"][:k])
    d_plus = int(round((np.trace(A) + d) / 2))
    tot = {key: int(sum(x[key] for x in rows)) for key in ("mlp_kept_vexact", "attn_kept_vexact", "mlp_kept_svd",
                                                             "attn_kept_svd")}
    return {"k": int(k), "d": int(d), "hidden": int(h), "attn_width": int(D), "d_plus": d_plus, "d_minus": int(d - d_plus),
            "reg_perm": [int(v) for v in reg], "dense_matrix_params": int(k * (2 * h * d + 4 * D * d)), "totals": tot,
            "vexact_kept": tot["mlp_kept_vexact"] + tot["attn_kept_vexact"],
            "svd_kept": tot["mlp_kept_svd"] + tot["attn_kept_svd"],
            "max_block_rel_gap_mlp": max(x["mlp_rel_gap"] for x in rows),
            "max_block_rel_gap_attn": max(x["attn_rel_gap"] for x in rows), "per_block": rows,
            "random_seed": int(random_seed), "random_counts_equal": bool(per == per_r),
            "random_agreement_with_fitted_A": float(np.trace(r["A_star"] @ A) / d),
            "chance_agreement": float((2 * d_plus - d) ** 2 / d ** 2),
            "sources": {"heads": str(heads_path), "heads_sha256": sha256(heads_path), "operators": str(operators_path),
                        "operators_sha256": sha256(operators_path)}, **git_info()}


def table(res: dict) -> str:
    lines = ["| block | MLP ranks (fc1, fc2) | MLP V-exact | MLP SVD | shortfall | attention ranks (q, k, v, o) | "
             "attention V-exact | attention SVD | shortfall |", "|---|---|---|---|---|---|---|---|---|"]
    for x in res["per_block"]:
        lines.append(f"| {x['block'] + 1} | {x['mlp_ranks'][0]}, {x['mlp_ranks'][1]} | {x['mlp_kept_vexact']:,} | "
                     f"{x['mlp_kept_svd']:,} | {100 * x['mlp_rel_gap']:.2f}% | {', '.join(str(v) for v in x['attn_ranks'])} | "
                     f"{x['attn_kept_vexact']:,} | {x['attn_kept_svd']:,} | {100 * x['attn_rel_gap']:.2f}% |")
    t = res["totals"]
    lines.append(f"| total | | {t['mlp_kept_vexact']:,} | {t['mlp_kept_svd']:,} | | | {t['attn_kept_vexact']:,} | "
                 f"{t['attn_kept_svd']:,} | |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heads", default=str(REPO_ROOT / "runs" / "m1" / "m1_structure.npz"))
    p.add_argument("--operators", default=str(REPO_ROOT / "runs" / "x2" / "perlayer_reg_s0" / "results" / "e1_operators.npz"))
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--random-seed", type=int, default=1, help="structure.random_seed of configs/e3_random.yaml")
    p.add_argument("--json", default=None, help="also write the result to this JSON file")
    a = p.parse_args(argv)
    res = budget(a.heads, a.operators, a.k, a.random_seed)
    print(f"k = {res['k']}: A* d+ {res['d_plus']}, d- {res['d_minus']}; V-exact keeps {res['vexact_kept']:,} of "
          f"{res['dense_matrix_params']:,} matrix parameters (MLP {res['totals']['mlp_kept_vexact']:,}, attention "
          f"{res['totals']['attn_kept_vexact']:,}); SVD keeps {res['svd_kept']:,}; largest per-block shortfall "
          f"{100 * res['max_block_rel_gap_mlp']:.2f}% (MLP), {100 * res['max_block_rel_gap_attn']:.2f}% (attention). "
          f"Random control (seed {res['random_seed']}): same counts per block {res['random_counts_equal']}; "
          f"tr(A_random A*)/d {res['random_agreement_with_fitted_A']:.3f} (chance {res['chance_agreement']:.3f}).")
    print(table(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()

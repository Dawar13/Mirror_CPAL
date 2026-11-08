#!/usr/bin/env python
"""E1: Mirror Surgery go/kill experiment.

    python scripts/run_e1.py --config configs/e1.yaml           # DINOv2 ViT-B on the VM
    python scripts/run_e1.py --config configs/e1.yaml --tiny    # tiny random ViT, synthetic data, CPU

Stages: (1) fit statistics on fit images and their mirrors; (2) per-layer and shared mirror
operators, held-out errors against identity and random-involution nulls; (3) position-embedding
check; (4) MLP twin matching and breaking shares; (5) zero-shot surgery against rank-matched SVD and
random-involution surgery, scored by k-NN. Writes CSVs, plots, operators and E1_REPORT.md.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: cap before anything is allocated

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import build_adapter  # noqa: E402
from mirror.data import SPLIT_NAMES, get_data, make_loader, normalize  # noqa: E402
from mirror.fit import agreement, fit_layers, involution_trace, random_involution  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.matching import match_many, random_pairing  # noqa: E402
from mirror.report import decide, plot_breaking_share, plot_equivariance, write_report  # noqa: E402
from mirror.stats import collect_fit_stats, collect_heldout_errors  # noqa: E402
from mirror.surgery import (absorb_block, apply_surgery, apply_surgery_two, apply_svd_baseline,  # noqa: E402
                            equivariant_param_count, mlp_breaking_shares_two, svd_rank_for)
from mirror.utils import (REPO_ROOT, StageTimer, env_info, git_info, load_config, log,  # noqa: E402
                          set_seed, setup_torch)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "e1.yaml"))
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    p.add_argument("--out-root", default=None,
                   help="directory that receives results/ and plots/ (default: repo root, or tiny_out/ with --tiny)")
    p.add_argument("--fit-seed", type=int, default=None, help="override data.fit_seed (protocol v2)")
    p.add_argument("--ks", default=None, help="override surgery.ks, e.g. 6,9")
    return p.parse_args(argv)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def in_run_sanity(adapter, loader, device) -> dict:
    """Cheap checks on one batch: folding is exact; identity surgery changes nothing."""
    imgs, _ = next(iter(loader))
    x = normalize(imgs[:8].to(device), adapter.dtype)
    t0 = adapter.layer0_tokens(x)
    blk = adapter.model.blocks[0]
    folded = copy.deepcopy(blk)
    absorb_block(folded)
    fold_diff = float((blk(t0) - folded(t0)).abs().max())
    m = copy.deepcopy(adapter.model)
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    apply_surgery(m, list(range(L)), np.eye(d), [np.arange(h)] * L)
    id_diff = float((adapter.cls_feature(x) - adapter.cls_feature(x, m)).abs().max())
    del m, folded
    return {"fold_max_abs_diff": fold_diff, "identity_surgery_max_abs_diff": id_diff}


def position_check(adapter, ops: dict) -> dict:
    P = adapter.pos_embed_grid().detach().to(torch.float64).cpu().numpy()
    H, W, d = P.shape
    Pc = P - P.mean(-1, keepdims=True)
    Pg = Pc - Pc.mean((0, 1), keepdims=True)

    def rel(G, A):
        M = G[:, ::-1].reshape(-1, d)
        Q = G.reshape(-1, d) @ A
        return float(np.linalg.norm(M - Q) / max(np.linalg.norm(G), 1e-300))

    return {"channel_centred": {k: rel(Pc, A) for k, A in ops.items()},
            "grid_centred": {k: rel(Pg, A) for k, A in ops.items()}}


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, tiny=args.tiny)
    if args.fit_seed is not None:
        cfg["data"]["fit_seed"] = args.fit_seed
    if args.ks:
        cfg["surgery"]["ks"] = [int(k) for k in args.ks.split(",")]
    tiny = cfg["tiny_mode"]
    out_root = Path(args.out_root) if args.out_root else (REPO_ROOT / "tiny_out" if tiny else REPO_ROOT)
    res_dir, plot_dir = out_root / "results", out_root / "plots"
    res_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_seed(seed)
    device = setup_torch(cfg)
    git = git_info()
    timer = StageTimer()
    log(f"E1 {'TINY ' if tiny else ''}run, device {device}, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")

    # ---------------------------------------------------------------- model and data
    with timer.stage("load"):
        adapter = build_adapter(cfg, device)
        comp = cfg["compute"]
        pin = device.type == "cuda"
        nw = int(comp.get("num_workers", 0))
        protocol = cfg["data"].get("protocol", "v1")
        if protocol == "v2" and not tiny:
            # Paper protocol: fit on ImageNet train (fit seed), held-out errors and k-NN queries on val,
            # k-NN bank of 100 train images per class (PAPER_EXPERIMENTS.md Section 1).
            from mirror.protocol import Protocol

            fit_seed = int(cfg["data"].get("fit_seed", 0))
            proto = Protocol(cfg["data"]["cache_dir"], int(comp["batch_size"]), int(comp["stats_batch_size"]), nw, pin)
            stats_loaders = {"fit": proto.fit_loader(fit_seed), "heldout": proto.heldout_loader()}
            knn_loaders = {"bank": proto.bank_loader(), "query": proto.query_loader()}
            num_classes, split_sizes = 1000, proto.sizes(fit_seed)
        else:
            images, labels, splits, num_classes = get_data(cfg)
            stats_loaders = {s: make_loader(images, labels, splits[s], int(comp["stats_batch_size"]), nw, pin)
                             for s in ("fit", "heldout")}
            knn_loaders = {s: make_loader(images, labels, splits[s], int(comp["batch_size"]), nw, pin)
                           for s in ("bank", "query")}
            split_sizes = {s: int(len(splits[s])) for s in SPLIT_NAMES}
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    log(f"model depth {L}, d {d}, h {h}, grid {adapter.grid_hw}, prefix tokens {adapter.n_prefix_tokens}; "
        f"protocol {protocol}; " + ", ".join(f"{s} {n}" for s, n in split_sizes.items()))

    with timer.stage("sanity"):
        sanity = in_run_sanity(adapter, stats_loaders["fit"], device)
    log(f"sanity: {sanity}")

    # ---------------------------------------------------------------- 1. fit statistics
    st = cfg["stats"]
    center = bool(st.get("center", True))
    mode = str(st.get("operator_mode", "shared"))  # shared (E1) | per_layer (E1b)
    if mode not in ("shared", "per_layer"):
        raise ValueError(f"unknown operator_mode {mode}")
    log(f"operator mode: {mode}")
    with timer.stage("fit_stats"):
        fs = collect_fit_stats(adapter, stats_loaders["fit"], device, center=center, mlp=True)
        res = fs.residual.numpy()

    # ---------------------------------------------------------------- 2. operators and held-out errors
    with timer.stage("fit_operators"):
        shared_layers = (int(st["shared_layers"][0]), min(int(st["shared_layers"][1]), L))
        fit = fit_layers(res["C"], shared_layers, res["sq_x"], res["sq_xt"])
        A_s, A_sn = fit["A_shared"], fit["A_shared_norm"]
        n_null = int(st.get("n_null", 3))
        nulls_shared = [random_involution(d, fit["d_plus_shared"], seed=seed + 101 + s) for s in range(n_null)]
        nulls_layer = [[random_involution(d, fit["d_plus"][l], seed=seed + 1000 + 37 * l + s) for s in range(n_null)]
                       for l in range(L + 1)]
        operators = []
        for l in range(L + 1):
            o = {"A_layer": fit["A_layers"][l], "A_shared": A_s, "A_shared_norm": A_sn}
            o.update({f"null_shared_{s}": N for s, N in enumerate(nulls_shared)})
            o.update({f"null_layer_{s}": N for s, N in enumerate(nulls_layer[l])})
            operators.append(o)
    log(f"shared operator: d+ {fit['d_plus_shared']}, d- {fit['d_minus_shared']}, zero {fit['n_zero_shared']}")

    with timer.stage("heldout_errors"):
        err = collect_heldout_errors(adapter, stats_loaders["heldout"], device, operators, center=center)

    layer_rows = []
    for l in range(L + 1):
        sq, e = err["sq_xt"][l], err["err"][l]
        rel_null_s = float(np.mean([e[f"null_shared_{s}"] for s in range(n_null)]) / sq)
        rel_null_l = float(np.mean([e[f"null_layer_{s}"] for s in range(n_null)]) / sq)
        row = {
            "layer": l, "d": d, "d_plus": fit["d_plus"][l], "d_minus": fit["d_minus"][l],
            "d_zero": fit["n_zero"][l], "dminus_frac": fit["d_minus"][l] / d,
            "agree_shared": agreement(fit["A_layers"][l], A_s),
            "agree_null": float(np.mean([agreement(fit["A_layers"][l], N) for N in nulls_shared])),
            "rel_e_A_layer": e["A_layer"] / sq, "rel_e_A_shared": e["A_shared"] / sq,
            "rel_e_A_shared_norm": e["A_shared_norm"] / sq, "rel_e_I": float(err["err_identity"][l] / sq),
            "rel_e_null_shared": rel_null_s, "rel_e_null_layer": rel_null_l,
            "sq_norm_xt": float(sq), "n_tokens": int(err["n"][l]),
        }
        row["ratio_null_over_shared"] = row["rel_e_null_shared"] / row["rel_e_A_shared"] if row["rel_e_A_shared"] > 0 else float("inf")
        row["ratio_shared_over_I"] = row["rel_e_A_shared"] / row["rel_e_I"] if row["rel_e_I"] > 0 else float("inf")
        layer_rows.append(row)
    layers_df = pd.DataFrame(layer_rows)
    # Columns the decision table reads: the operator under test and its matched null.
    layers_df["rel_e_A_used"] = layers_df["rel_e_A_layer" if mode == "per_layer" else "rel_e_A_shared"]
    layers_df["rel_e_null_used"] = layers_df["rel_e_null_layer" if mode == "per_layer" else "rel_e_null_shared"]

    # ---------------------------------------------------------------- 3. position embedding
    with timer.stage("position_check"):
        pos = position_check(adapter, {"A_shared": A_s, "A_0 (layer 0)": fit["A_layers"][0],
                                       "identity": np.eye(d), "null": nulls_shared[0]})

    # ---------------------------------------------------------------- 4. MLP twins and breaking share
    mt = cfg["matching"]
    with timer.stage("mlp_matching"):
        R_list = [fs.mlp.correlation(b) for b in range(L)]
        del fs
        _free(device)
        workers = mt.get("workers", "auto")
        workers = max(1, min(L, (os.cpu_count() or 2) - 1)) if workers == "auto" else int(workers)
        matches = match_many(R_list, top_k=int(mt["top_k"]), workers=workers)
        diag_scores = [np.diag(0.5 * (R + R.T)).copy() for R in R_list]
        del R_list

    with timer.stage("breaking_share"):
        dc = cfg["decision"]
        lo_d, hi_d = int(dc["blocks"][0]), min(int(dc["blocks"][1]), L)
        n_pairs = [m.n_pairs for m in matches]
        if mode == "per_layer":
            # Block b (0-based) reads layer b and writes layer b+1: fc1 is tested against A_b and
            # fc2 against A_(b+1). Nulls: the per-layer residual nulls (same d+ as each A_l).
            A_in = [fit["A_layers"][b] for b in range(L)]
            A_out = [fit["A_layers"][b + 1] for b in range(L)]
            null_in = [[nulls_layer[b][s] for b in range(L)] for s in range(n_null)]
            null_out = [[nulls_layer[b + 1][s] for b in range(L)] for s in range(n_null)]
            dminus_blocks = [fit["d_minus"][b] / d for b in range(L)]
            dminus_frac = float(np.mean([fit["d_minus"][l] / d for l in range(lo_d, hi_d + 1)]))
        else:
            # One operator everywhere; nulls have the same d+ and d- as A_shared (E1 behaviour).
            A_in = A_out = [A_s] * L
            null_in = null_out = [[nulls_shared[s]] * L for s in range(n_null)]
            dminus_blocks = [fit["d_minus_shared"] / d] * L
            dminus_frac = fit["d_minus_shared"] / d
        # Matched pairings: same numbers of pairs and self-twins as the fitted ones, per block.
        null_P = [[random_pairing(h, n_pairs[b], seed=(seed + 101 + s) * 1000 + 17 + b) for b in range(L)]
                  for s in range(n_null)]
        assert all(int((p != np.arange(h)).sum()) == 2 * n_pairs[b] for ps in null_P for b, p in enumerate(ps))
        mlp_rows = []
        for b in range(L):
            blk, mres = adapter.model.blocks[b], matches[b]
            sh = mlp_breaking_shares_two(blk, A_in[b], A_out[b], mres.perm)
            shn = [mlp_breaking_shares_two(blk, null_in[s][b], null_out[s][b], null_P[s][b]) for s in range(n_null)]
            s1n = float(np.mean([x["s1"] for x in shn]))
            s2n = float(np.mean([x["s2"] for x in shn]))
            eq1 = equivariant_param_count(h, d, mres.trace, involution_trace(A_in[b]))
            eq2 = equivariant_param_count(h, d, mres.trace, involution_trace(A_out[b]))
            r1, r2 = svd_rank_for(h, d, eq1), svd_rank_for(h, d, eq2)
            paired = mres.score[mres.perm != np.arange(h)]
            selfs = mres.score[mres.self_twins]
            twin = 2 * mres.n_pairs / h
            mlp_rows.append({
                "block": b + 1, "h": h, "d": d, "n_pairs": mres.n_pairs, "n_self": mres.n_self,
                "twin_frac": twin, "dminus_frac": dminus_blocks[b],
                "evidence": bool(dminus_blocks[b] >= dc["evidence_dminus"] or twin >= dc["evidence_twin_frac"]),
                "mean_score": float(mres.score.mean()),
                "mean_score_pairs": float(paired.mean()) if paired.size else float("nan"),
                "mean_score_self": float(selfs.mean()) if selfs.size else float("nan"),
                "s1": sh["s1"], "s2": sh["s2"], "sb1": sh["sb1"], "sb2": sh["sb2"],
                "s1_null": s1n, "s2_null": s2n,
                "s1_over_null": sh["s1"] / s1n if s1n > 0 else float("nan"),
                "s2_over_null": sh["s2"] / s2n if s2n > 0 else float("nan"),
                "tr_P": mres.trace, "tr_A_in": involution_trace(A_in[b]), "tr_A_out": involution_trace(A_out[b]),
                "eq_params_fc1": eq1, "eq_params_fc2": eq2, "dense_params": 2 * h * d,
                "param_saving_frac": 1 - (eq1 + eq2) / (2 * h * d),
                "svd_rank": r1, "svd_rank_fc2": r2, "svd_params": (r1 + r2) * (h + d),
            })
        mlp_df = pd.DataFrame(mlp_rows)

    # Save the structure results now, so a failure in the surgery stage cannot lose them.
    for df in (layers_df, mlp_df):
        df["git_hash"] = git["git_hash"]
        df["seed"] = seed
    layers_df.to_csv(res_dir / "e1_layers.csv", index=False)
    mlp_df.to_csv(res_dir / "e1_mlp.csv", index=False)
    np.savez_compressed(
        res_dir / "e1_operators.npz",
        A_layers=np.stack(fit["A_layers"]).astype(np.float32), A_shared=A_s.astype(np.float32),
        A_shared_norm=A_sn.astype(np.float32), eigvals=np.stack(fit["eigvals"]),
        eigvals_shared=fit["eigvals_shared"], C=res["C"].astype(np.float32), G=res["G"].astype(np.float32),
        perms=np.stack([m.perm for m in matches]).astype(np.int32),
        match_scores=np.stack([m.score for m in matches]).astype(np.float32),
        self_corr=np.stack(diag_scores).astype(np.float32),
    )
    log(f"saved structure results to {res_dir}")

    # ---------------------------------------------------------------- 5. zero-shot surgery
    kc = cfg["knn"]
    with timer.stage("surgery_knn"):
        amp = comp.get("knn_amp")  # "bf16" in the paper protocol, identical for every variant

        def knn(model=None):
            out = evaluate_knn(adapter, knn_loaders["bank"], knn_loaders["query"], device, num_classes,
                               k=int(kc["k"]), temperature=float(kc["temperature"]), model=model, amp=amp)
            return out["top1"], out["flip_consistency"]

        # The original model's k-NN does not depend on the fit seed: cache it per model and protocol.
        knn_cache = Path(cfg["data"].get("cache_dir", "cache")) / "knn_original" / \
            f"{cfg['model']['name']}_{protocol}_{amp}.json"
        if protocol == "v2" and not tiny and knn_cache.exists():
            cached = json.loads(knn_cache.read_text())
            top1_0, flip_0 = cached["top1"], cached["flip_consistency"]
            log(f"original k-NN from cache {knn_cache}")
        else:
            top1_0, flip_0 = knn()
            if protocol == "v2" and not tiny:
                knn_cache.parent.mkdir(parents=True, exist_ok=True)
                knn_cache.write_text(json.dumps({"top1": top1_0, "flip_consistency": flip_0, "git": git["git_hash"]}))
        log(f"original: top-1 {top1_0:.2f}, flip consistency {flip_0:.2f}")
        surg_rows = [{"k": 0, "variant": "original", "top1": top1_0, "flip_consistency": flip_0,
                      "drop_points": 0.0, "mlp_params_kept_frac": 1.0}]
        for k in [int(k) for k in cfg["surgery"]["ks"]]:
            if k > L:
                log(f"skip k = {k} (> depth {L})")
                continue
            blocks = list(range(k))
            eq_sum = float((mlp_df["eq_params_fc1"] + mlp_df["eq_params_fc2"]).iloc[:k].sum())
            svd_sum = float(mlp_df["svd_params"].iloc[:k].sum())
            dense = float(mlp_df["dense_params"].iloc[:k].sum())
            ranks = [(int(mlp_df["svd_rank"].iloc[b]), int(mlp_df["svd_rank_fc2"].iloc[b])) for b in blocks]
            variants = {
                "surgery": (lambda m: apply_surgery_two(m, blocks, [A_in[b] for b in blocks], [A_out[b] for b in blocks],
                                                        [matches[b].perm for b in blocks]), eq_sum),
                "svd": (lambda m: apply_svd_baseline(m, blocks, ranks), svd_sum),
                "random": (lambda m: apply_surgery_two(m, blocks, [null_in[0][b] for b in blocks],
                                                       [null_out[0][b] for b in blocks],
                                                       [null_P[0][b] for b in blocks]), eq_sum),
            }
            for name, (fn, kept) in variants.items():
                m = copy.deepcopy(adapter.model)
                fn(m)
                t1, fc = knn(m)
                del m
                _free(device)
                surg_rows.append({"k": k, "variant": name, "top1": t1, "flip_consistency": fc,
                                  "drop_points": top1_0 - t1, "mlp_params_kept_frac": kept / dense})
                log(f"k = {k} {name}: top-1 {t1:.2f}, flip consistency {fc:.2f}")
                pd.DataFrame(surg_rows).to_csv(res_dir / "e1_surgery.csv", index=False)  # partial, kept on failure
        surgery_df = pd.DataFrame(surg_rows).sort_values("k", kind="stable").reset_index(drop=True)

    # ---------------------------------------------------------------- outputs
    decision = decide(layers_df, mlp_df, surgery_df, dminus_frac, cfg["decision"], mode=mode)
    surgery_df["git_hash"] = git["git_hash"]
    surgery_df["seed"] = seed
    surgery_df.to_csv(res_dir / "e1_surgery.csv", index=False)

    suffix = " (TINY TEST)" if tiny else ""
    plot_equivariance(layers_df, plot_dir / "equivariance_vs_depth.png", suffix)
    plot_breaking_share(mlp_df, plot_dir / "breaking_share_vs_depth.png", cfg["decision"], suffix)

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "git": git, "env": env_info(device),
        "config_path": cfg["config_path"], "config": cfg, "tiny": tiny,
        "split_sizes": split_sizes, "protocol": protocol, "fit_seed": cfg["data"].get("fit_seed"),
        "knn_amp": comp.get("knn_amp"),
        "timing": {"stages": timer.stages, "total_s": timer.total()},
        "shared": {"d_plus": fit["d_plus_shared"], "d_minus": fit["d_minus_shared"],
                   "n_zero": fit["n_zero_shared"], "layers": list(shared_layers),
                   "d_plus_norm": fit["d_plus_shared_norm"], "d_minus_norm": fit["d_minus_shared_norm"]},
        "operator_mode": mode, "dminus_frac_decision": dminus_frac,
        "matching_workers": workers, "position_check": pos, "sanity": sanity, "decision": decision,
    }
    with open(res_dir / "e1_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    write_report(res_dir / "E1_REPORT.md", {
        "decision": decision, "meta": meta, "tiny": tiny,
        "model_name": "tiny random ViT" if tiny else cfg["model"]["name"],
        "layers_df": layers_df, "mlp_df": mlp_df, "surgery_df": surgery_df, "pos": pos, "sanity": sanity,
        "d": d, "d_plus_shared": fit["d_plus_shared"], "d_minus_shared": fit["d_minus_shared"],
        "n_zero_shared": fit["n_zero_shared"], "shared_layers": shared_layers, "n_null": n_null,
        "grid": adapter.grid_hw, "decision_blocks": cfg["decision"]["blocks"], "operator_mode": mode,
        "split_seed": cfg["data"].get("split_seed"),
        "surgery_k": cfg["decision"]["surgery_k"], "evidence_dminus": cfg["decision"]["evidence_dminus"],
        "evidence_twin_frac": cfg["decision"]["evidence_twin_frac"],
    })
    log(f"OUTCOME: {decision['outcome']} ({decision['reason']})")
    log(f"wrote {res_dir / 'E1_REPORT.md'} in {timer.total() / 60:.1f} min")
    return {"decision": decision, "out_root": str(out_root)}


if __name__ == "__main__":
    main()
